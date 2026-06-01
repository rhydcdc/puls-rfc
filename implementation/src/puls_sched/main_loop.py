from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from puls_sched.admission import Admission, MicroBatchSpec
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import LayerState
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.kv_accountant import KVAccountant
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType
from puls_sched.request import Request, RequestState
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow

if TYPE_CHECKING:
    from puls_sched.evaluator import AdmissionSnapshot


AdmissionTickCallback = Callable[["AdmissionSnapshot"], None]

# STEP 5.5 — per-mb KV 예산의 분모 = 동시 활성 μ-batch 목표. ARCH §5.6/F2 는 동시 2개
# (M 의 PIM attention ‖ M+1 의 QKV)면 충족; window 3번째 슬롯은 빠지는 M-1 전이 여유라
# KV 를 가득 쓸 필요 없음. 그래서 KV캐파를 window(3) 가 아니라 2 로 나눔 — 배치가
# n_sat 위로 포화 유지 + "2 active + 1 여유" 정합. window.capacity 보다 크지 않게 clamp.
_STAGGERING_TARGET_MB = 2


@dataclass
class SchedulerCore:
    config: Config
    clock: Clock
    queue: EventQueue
    dag: DAG
    window: InFlightWindow
    dispatcher: Dispatcher
    request_queue: RequestQueue
    kv_accountant: KVAccountant
    admission: Admission
    # ---- Impl-6 — Q5 (token decode signal consumer) + Q10 (Request lifecycle owner) ----
    layer_state: LayerState
    completion: Completion
    in_flight_requests: dict[int, Request] = field(default_factory=dict)
    _next_mb_id: int = 0
    # ---- Phase-2 S2 — 합류 게이트(_join_gate_open) 삭제. 풀 모델에선 멤버십=용량이라
    # 자리(KV·batch) 있고 일감 있으면 무조건 backfill (유휴율 게이트 개념 소멸, 배치_생애 §밸런스).
    # ---- Impl-8 — D1 admission tick hook (evaluator 등록 점) ----
    _admission_tick_callbacks: list[AdmissionTickCallback] = field(default_factory=list)
    # ---- Impl-9 Q1 — ADMISSION_TICK self-rescheduling opt-in (Run.init 가 enable). ----
    # Default False: isolated unit test 영역의 single-shot 처리 보존 (R14 setup gap).
    # True: ARCH §6.4 per-iteration admission cadence 정합 (continuous chain).
    enable_admission_tick_rescheduling: bool = False
    # ---- Impl-10-pre-1 (A) — production hot path 위 inter-AB chain wiring. ----
    # Default None: 기존 SchedulerCore fixture 영역 backward-compat (단위 test 무변경).
    # None 아니면 매 O_PROJ 완료 시 instance_pipeline.dispatch(mb) 호출 — ARCH §3.4
    # *forward pass = L × cycle* literal 의 production hot path 영역 wiring.
    # gpu_instance_b activity recording + fixed-shape handoff defensive validation.
    instance_pipeline: Optional[InstancePipeline] = None
    # ---- Impl-10-pre-1 (B) — per-iteration cycle measurement snapshot. ----
    # 이전 ADMISSION_TICK 시점의 IdleTelemetry active_duration 누적값. _measure_cycles 가
    # 현 누적값과 delta 산출 → a_cycle / b_cycle (ARCH §6.4 'previous iteration measurement').
    _prev_a_active_snapshot: float = 0.0
    _prev_b_active_snapshot: float = 0.0

    def on_admission_tick(self, callback: AdmissionTickCallback) -> None:
        """Admission tick snapshot capture 위 callback 등록 (Impl-8 D1 hook).

        Evaluator 같은 외부 inspector 가 admission_convergence series 캡처 위 등록.
        SchedulerCore 자체는 Evaluator 를 모름 (D3 standalone).
        """
        self._admission_tick_callbacks.append(callback)

    def _fire_admission_tick(
        self,
        spec: "MicroBatchSpec | None",
        a_cycle: float,
        b_cycle: float,
        ctx_tokens: int,
    ) -> None:
        """등록된 callback 들에게 admission tick snapshot 통지 (Impl-8 D1 hook fire).

        Spec=None (admission 실패 path) 도 snapshot 누적 — empty admission tick 도
        convergence trace 의 의미 있는 entry (admission cadence 자연 series).
        """
        if not self._admission_tick_callbacks:
            return
        from puls_sched.evaluator import AdmissionSnapshot
        snapshot = AdmissionSnapshot(
            timestamp=self.clock.now,
            gpu_idle_fraction=self.admission.idle_telemetry.gpu_idle_fraction(),
            pim_idle_fraction=self.admission.idle_telemetry.pim_idle_fraction(),
            a_cycle=a_cycle,
            b_cycle=b_cycle,
            ctx_tokens=ctx_tokens,
            spec_admitted=(spec is not None),
            n=spec.n if spec else 0,
        )
        for cb in self._admission_tick_callbacks:
            cb(snapshot)

    def step(self) -> bool:
        if len(self.queue) == 0:
            return False
        event = self.queue.pop()
        self._handle(event)
        return True

    def _handle(self, event: Event) -> None:
        match event.type:
            case EventType.KERNEL_COMPLETION:
                # Impl-9 — 방어: mb 가 이미 evict 된 경우 (auto-evict 또는 explicit) stale event skip.
                # window.admit overflow 위 auto-evict + explicit evict (token 완료) 의 race 정합.
                mb_id_completion = event.payload.get("micro_batch_id")
                if mb_id_completion not in self.dag.nodes:
                    resource = event.payload.get("resource")
                    if resource == "GPU":
                        self.dispatcher.gpu_busy = False
                    elif resource == "PIM":
                        self.dispatcher.pim_busy = False
                    elif resource == "INSTANCE_B":
                        self.dispatcher.instance_b_busy = False
                    self.dispatcher.tick()
                    # STEP 2.5 — 완료 = iteration 경계 = admit 기회 (event-driven admission).
                    if self.enable_admission_tick_rescheduling:
                        self._schedule_admission_tick_with_default_payload()
                    return
                self.dispatcher.on_completion(event)
                # Impl-6 (Q5) — O_PROJ done 분기 → LayerState.advance → L 도달 시 token decode signal
                self._maybe_advance_forward_pass(event)
                self.dispatcher.tick()
                # STEP 2.5 — 완료 시 admission (자원이 비는 유일 시점). 고정 타이머 self-push 폐기.
                if self.enable_admission_tick_rescheduling:
                    self._schedule_admission_tick_with_default_payload()
            case EventType.REQUEST_ARRIVAL:
                req = event.payload["request"]
                self.request_queue.push(req)
                # Impl-9 Q1 — Arrival re-wakes admission chain (idle guard 의 dual entry).
                # ARCH §6.4 'per-iteration admission' 의 arrival-driven 재기동 의미 정합.
                if self.enable_admission_tick_rescheduling:
                    self._schedule_admission_tick_with_default_payload()
            case EventType.ADMISSION_TICK:
                # Impl-8 — admission tick hook 위 spec + cycle 값 산출 (snapshot fire 영역)
                a_cycle = event.payload.get("a_cycle", 0.0)
                b_cycle = event.payload.get("b_cycle", 0.0)
                ctx_tokens = event.payload.get("ctx_tokens", 0)
                # Impl-9 — Window full 시 admission 대기 (ARCH §6.7 '3-μ-batch in-flight window' 의미).
                # Auto-evict (window.admit overflow) 는 *defensive* 영역으로 격하.
                if len(self.window.current_ids()) >= self.window.capacity:
                    self._fire_admission_tick(None, a_cycle, b_cycle, ctx_tokens)
                    # STEP 2.5 — window full 이면 다음 완료(evict) 시 admission 재기동.
                    # 고정 타이머 self-push 폐기 (헛도는 tick 차단).
                    return
                spec = self._invoke_admission(event)
                self._fire_admission_tick(spec, a_cycle, b_cycle, ctx_tokens)
                if spec is not None:
                    mb_id = self._next_mb_id
                    self._next_mb_id += 1
                    # Impl-10-pre-2 (O9.1 + O9.2) — Mixed batch composition.
                    # Lifecycle owner registration first (Q10), then mb populate.
                    for req in spec.decode_requests:
                        if req.state == RequestState.PENDING:
                            req.transition_to(RequestState.PREFILL)
                        self.in_flight_requests[req.id] = req
                    # ARCH §5.2 uniform-chunk + B option (PIM-slack) —
                    # spec.prefill_chunk_tokens = TOTAL budget. Distribute uniformly across prefill reqs.
                    prefill_chunk, decode_tokens, prefill_processed = self._populate_mb_phases(
                        spec.decode_requests, spec.prefill_chunk_tokens,
                    )
                    mb = MicroBatch(
                        id=mb_id,
                        kv_rows_total=spec.kv_rows_total,
                        kv_rows_lockstep=spec.kv_rows_lockstep,   # Impl-8 — F5 ablation 위 signal flow
                        prefill_chunk=prefill_chunk,                  # Impl-10-pre-2 (O9.1)
                        decode_tokens=decode_tokens,
                        prefill_chunk_budget=spec.prefill_chunk_tokens,   # Impl-10-pre-2 B — adaptive budget 보존
                        prefill_processed=prefill_processed,          # Impl-10 main — PREFILL_ATTN causal ctx 산출
                    )
                    self.dispatcher.register(mb)
                    self.window.admit(mb_id)
                    self.dispatcher.tick()
                # STEP 2.5 — 고정 타이머 self-push 폐기. 다음 admission 은 완료
                # (KERNEL_COMPLETION) 또는 신규 도착(REQUEST_ARRIVAL) 시 재기동.

    def _schedule_admission_tick_with_default_payload(self) -> None:
        """완료(KERNEL_COMPLETION) / 신규 도착(REQUEST_ARRIVAL) 시 admission 재기동.

        STEP 2.5 — 고정 타이머 self-push (`_schedule_next_admission_tick`) 폐기 후
        admission 의 유일한 재기동 경로. 이벤트 기반: 자원이 비는 시점(완료)과 새 일감
        도착 시점에만 admission tick push → 헛도는 tick 제거 (REPORT §10).

        Impl-10-pre-1 (B)~(B''') — 'default' = scheduler 자기 상태 위 *진정 측정 payload*.
        """
        if len(self.request_queue) == 0 and len(self.in_flight_requests) == 0:
            return
        next_t = self.clock.now + self.config.admission.tick_interval_us
        self.queue.push(Event(
            timestamp=next_t,
            type=EventType.ADMISSION_TICK,
            payload=self._compose_admission_payload(),
        ))

    def _compose_admission_payload(self) -> dict:
        """ADMISSION_TICK 의 5 개 payload 입력 산출 (Impl-10-pre-1 (B)~(B''')).

        Impl-10 main (Stage 2) — *Stage 1 dummy lookup 영역 폐기*. 사용자 의도 정합 위
        모든 timing 영역 spec-derived per-mb 산출 (ARCH §3.5.2 Computed Wait literal).

        - (B) a_cycle / b_cycle — IdleTelemetry active_duration delta (이전 ADMISSION_TICK 대비)
        - (B') t_proj — last dispatched mb 위 spec-derived (QKV + O_PROJ, compute_gpu_op_time_s)
        - (B'') t_pim_fn — PIMExecutor.op_time 의 closure (avg kv_length 위 동적)
        - (B''') ctx_tokens — in_flight_requests 의 max kv_length (deadband ctx-tier 입력)
        - gpu_op_time_per_token_us — last mb 위 spec-derived per-token (PREFILL_ATTN / chunk)
        """
        from puls_sched.config import compute_gpu_op_time_s

        a_cycle, b_cycle = self._measure_cycles()
        ctx_tokens = max(
            (r.kv_length for r in self.in_flight_requests.values()), default=0,
        )

        # Stage 2 — Path C: last dispatched mb 위 spec-derived 산출 (정확, 평균/dummy 0).
        # Cold start (no mb yet) 위 fallback = 0.0 (admission balance 위 cold start anchor).
        last_mb = self._last_dispatched_mb()
        # Phase-2 — Instance A TP=num_gpus_instance_a 분산 (dispatcher·PIM 단위 통일).
        n_gpu_a = self.config.hw.num_gpus_instance_a
        if last_mb is not None:
            t_qkv_s = compute_gpu_op_time_s(
                NodeType.QKV, last_mb, self.config.calibration, self.config.model,
                num_gpus=n_gpu_a,
            )
            t_oproj_s = compute_gpu_op_time_s(
                NodeType.O_PROJ, last_mb, self.config.calibration, self.config.model,
                num_gpus=n_gpu_a,
            )
            t_proj_us = (t_qkv_s + t_oproj_s) * 1e6
            # per-token = PREFILL_ATTN total / chunk total (last mb 위 per-token 영역)
            chunk_total = sum(len(v) for v in last_mb.prefill_chunk.values())
            if chunk_total > 0:
                t_pattn_s = compute_gpu_op_time_s(
                    NodeType.PREFILL_ATTN, last_mb, self.config.calibration, self.config.model,
                    num_gpus=n_gpu_a,
                )
                per_token_us = (t_pattn_s * 1e6) / chunk_total
            else:
                # Decode-only last mb — per-token spec-derived 위 평균 ctx 산출
                avg_ctx = ctx_tokens if ctx_tokens > 0 else 1
                peak_FLOPS = (
                    self.config.calibration.gpu_fp16_dense_peak_tflops * 1e12
                    * self.config.calibration.gpu_mfu_default * n_gpu_a
                )
                per_token_us = (2 * self.config.model.hidden * avg_ctx / peak_FLOPS) * 1e6
        else:
            t_proj_us = 0.0
            per_token_us = 0.0

        return {
            "t_proj": t_proj_us,
            "t_pim_fn": self._make_t_pim_fn(),
            "a_cycle": a_cycle,
            "b_cycle": b_cycle,
            "ctx_tokens": ctx_tokens,
            "gpu_op_time_per_token_us": per_token_us,
        }

    def _last_dispatched_mb(self):
        """Last dispatched mb (가장 최근 register 된 mb) 위 spec-derived 산출 입력."""
        if not self.dispatcher.micro_batches:
            return None
        last_id = max(self.dispatcher.micro_batches.keys())
        return self.dispatcher.micro_batches[last_id]

    def _measure_cycles(self) -> tuple[float, float]:
        """Impl-10-pre-1 (B) — 이전 ADMISSION_TICK 대비 a_cycle / b_cycle delta 산출.

        a_cycle = Instance A 의 (gpu_instance_a + pim_instance_a) active 증가분
        b_cycle = Instance B 의 (gpu_instance_b) active 증가분
        """
        tel = self.admission.idle_telemetry
        cur_a = tel.active_duration("gpu_instance_a") + tel.active_duration("pim_instance_a")
        cur_b = tel.active_duration("gpu_instance_b")
        delta_a = max(0.0, cur_a - self._prev_a_active_snapshot)
        delta_b = max(0.0, cur_b - self._prev_b_active_snapshot)
        self._prev_a_active_snapshot = cur_a
        self._prev_b_active_snapshot = cur_b
        return delta_a, delta_b

    def _make_t_pim_fn(self):
        """Impl-10-pre-1 (B'') — PIMExecutor.op_time closure (in_flight_requests 의 avg kv 위 동적).

        Impl-10-pre-2 — k_channels 매개변수 폐기 (sequence-parallel PIM 위 무의미).
        Signature: fn(n_decode) → float (단위 = µs).

        STEP 5 단위 버그 수정 — PIMExecutor.op_time() 은 ns 반환. balance_pim_slack 은 이를
        t_proj(µs)와 빼므로 **µs 로 변환(× 1e-3)해야 함** (dispatcher._op_time 의 PIM ×1e-3
        convention 과 동일). 미변환 시 t_pim 이 1000× 부풀어 prefill chunk 예산 과대 →
        GPU 과포화·PIM 유휴. dispatcher 는 Stage 2 에 정정됐으나 이 admission 경로는 누락됐었음.
        """
        in_flight = self.in_flight_requests
        pim_executor = self.dispatcher.pim_executor
        def fn(n_decode: int) -> float:
            if not in_flight or n_decode <= 0:
                return 0.0
            avg_kv = max(1, sum(r.kv_length for r in in_flight.values()) // len(in_flight))
            return pim_executor.op_time(kv_rows_total=n_decode * avg_kv) * 1e-3  # ns → µs
        return fn

    def _maybe_advance_forward_pass(self, event: Event, eos_seen: bool = False) -> None:
        """KERNEL_COMPLETION (FFN done) → LayerState.advance → L 도달 시 token decode signal.

        Phase-2 — trigger = FFN 완료(= layer 종료, inter-AB F3). 이전엔 O_PROJ 였으나 B-FFN
        을 스케줄 노드로 모델링하며 layer 경계가 FFN 완료로 이동.

        Q5 — consumer 는 main_loop 영역. Dispatcher / forward_pass 침범 0.
        Q6 (c) — eos_seen=True 명시 시 EOS branch 발동 (외부 caller / test fixture path).

        Impl-10-pre-1 (A) — 매 O_PROJ 완료 = 1 layer cycle 의 A-side 종료 = B-side 진입 시점.
        ARCH §3.4 *forward pass = L × cycle* literal 정합. `instance_pipeline.dispatch(mb)` 호출 위
        gpu_instance_b activity recording (inter-AB balance signal substrate) + fixed-shape handoff
        defensive validation (ARCH §5.2 production 강제). `instance_pipeline=None` 시 skip (backward-compat).

        Impl-10-pre-2 (O9.1) — L 도달 시 prefill_chunk 의 req 별 prefill_processed 갱신 +
        PREFILL → DECODE 전이 검출. Re-composition 위 다음 cycle 의 prefill_chunk + decode_tokens
        재산출 (남은 prefill 영역 위 새 chunk 영역).
        """
        node_type = event.payload.get("node_type")
        # Phase-2 — layer advance trigger 를 O_PROJ → FFN 으로 이동 (inter-AB, F3).
        # 한 layer = QKV → attn → O_PROJ → FFN. FFN(Instance B) 완료 = 그 layer 종료.
        # B-side timing·gpu_instance_b 기록은 이제 dispatcher.dispatch_instance_b 가 담당
        # (옛 instance_pipeline.dispatch telemetry 호출 대체).
        if node_type is not NodeType.FFN:
            return
        mb_id = event.payload.get("micro_batch_id")
        mb = self.dispatcher.micro_batches.get(mb_id) if mb_id is not None else None
        if mb is None:
            return  # defensive — mb already unregistered
        token_signal = self.layer_state.advance(mb)
        if not token_signal:
            # Impl-9 — 다음 layer 의 fresh dispatch 위 DAG nodes 재생성. ARCH §3.4 L × cycle 정합.
            self.dag.reset_micro_batch(mb.id)
            return
        # ---- L 도달 — token decode signal + prefill chunk advancement ----
        # Impl-10-pre-2 (O9.1) — prefill_chunk 의 req 위 prefill_processed 갱신 + 상태 전이
        for req_id in list(mb.prefill_chunk.keys()):
            req = self.in_flight_requests.get(req_id)
            if req is None:
                continue  # defensive
            chunk_used = len(mb.prefill_chunk[req_id])
            req.prefill_processed += chunk_used
            if req.prefill_processed >= len(req.prompt_tokens) and req.state == RequestState.PREFILL:
                req.transition_to(RequestState.DECODE)
        # Decode token 생성 (decode_tokens 위 reqs 만)
        for req_id in list(mb.decode_tokens.keys()):
            req = self.in_flight_requests.get(req_id)
            if req is None:
                continue
            req.decoded_count += 1
            if req.state == RequestState.PREFILL:
                # Defensive — prefill_processed 이미 도달했을 가능성 (race-free 보장 위)
                req.transition_to(RequestState.DECODE)
            if self.completion.check(req, eos_seen=eos_seen):
                self.completion.finalize(req)
                self.in_flight_requests.pop(req_id, None)
        # 다음 token 의 forward pass 위 reset (multi-token decode 정합)
        mb.current_layer_index = 0
        # Phase-2 S2 — 슬롯 재구성 (1) 잔여 req 재배치 + (2) 풀(queue)에서 연속 backfill.
        # evict 판정 = 재구성 후에도 prefill_chunk·decode_tokens 모두 빔 (완료 + 풀 빔).
        # 배치_생애 §종료 — 요청 단위 종료, 슬롯은 자연히 비면 evict.
        self._recompose_mb(mb)
        if mb.prefill_chunk or mb.decode_tokens:
            self.dag.reset_micro_batch(mb.id)
        else:
            self.window.evict(mb.id)
            self.dispatcher.unregister(mb.id)

    def _recompose_mb(self, mb: MicroBatch) -> None:
        """Phase-2 S2 — 슬롯의 다음 L-cycle 위 prefill_chunk + decode_tokens 갱신.

        잔여 req(완료 안 된 것) sticky 유지 + 풀(queue)에서 빈 자리 연속 backfill.
        Adaptive budget 보존 — mb.prefill_chunk_budget (admission 위 산출). Fallback = default.
        """
        budget = mb.prefill_chunk_budget if mb.prefill_chunk_budget > 0 else self.config.admission.prefill_chunk_default
        active_req_ids = set(mb.prefill_chunk.keys()) | set(mb.decode_tokens.keys())
        active_req_ids = {rid for rid in active_req_ids if rid in self.in_flight_requests}
        # Phase-2 S2 — 풀(queue) 신규 요청을 이 슬롯의 빈 자리에 backfill (배치_생애 §4).
        # prefill/decode 구분은 _populate_mb_phases 가 prompt 유무로 자동 분류. 유휴 게이트 없음.
        joined_ids = self._backfill_slot(active_req_ids)
        active_req_ids |= joined_ids
        active_reqs = [self.in_flight_requests[rid] for rid in active_req_ids]
        new_prefill_chunk, new_decode_tokens, new_prefill_processed = self._populate_mb_phases(
            active_reqs, budget,
        )
        mb.prefill_chunk = new_prefill_chunk
        mb.decode_tokens = new_decode_tokens
        mb.prefill_processed = new_prefill_processed   # Impl-10 main — re-composition causal ctx refresh

    def _backfill_slot(self, active_req_ids: set[int]) -> set[int]:
        """Phase-2 S2 — 슬롯의 빈 자리에 풀(queue) 신규 요청을 연속 backfill (유휴 게이트 없음).

        풀 모델 멤버십 = 용량: 자리(batch·KV) 있고 일감 있으면 무조건 들임 (배치_생애 §밸런스
        "놀까봐 가져온다가 아니라 용량 있고 일감 있으니 넣는다"). prefill/decode 구분은
        `_populate_mb_phases` 가 prompt 유무로 자동 분류.

        가능량 = min(seq 여유, 전역 KV 여유, per-slot KV 예산). FIFO. KV 안 맞는 head 만나면
        중단(head-of-line — 다음 완료 경계에서 캐파 회수 후 재시도). per-slot KV 예산은
        2-슬롯 disjoint 분할의 원리적 한계 (한 슬롯이 KV 독점해 다른 슬롯 굶기는 것 방지).

        Returns: backfill 된 신규 요청 id 집합 (kv_accountant.admit + in_flight 등록 완료).
        """
        seq_room = self.config.admission.max_batch_size - len(active_req_ids)
        if seq_room <= 0:
            return set()
        # per-slot KV 예산 = 전역 KV / 활성 슬롯 목표(2). 한 슬롯의 KV 독점 방지 (disjoint 분할).
        per_slot_kv = self._per_mb_kv_budget()
        slot_kv = sum(
            self.in_flight_requests[rid].kv_length
            for rid in active_req_ids if rid in self.in_flight_requests
        )
        joined: set[int] = set()
        while len(joined) < seq_room:
            req = self.request_queue.peek_oldest()
            if req is None:
                break
            if not self.kv_accountant.can_admit(req):
                break  # 전역 KV 부족 → backfill 중단
            if slot_kv + req.kv_length > per_slot_kv:
                break  # per-slot KV 예산 초과 → 다른 슬롯 몫 보존
            self.request_queue.pop_oldest()
            self.kv_accountant.admit(req)
            slot_kv += req.kv_length
            if req.state == RequestState.PENDING:
                req.transition_to(RequestState.PREFILL)
            self.in_flight_requests[req.id] = req
            joined.add(req.id)
        return joined

    def _populate_mb_phases(
        self, reqs, chunk_budget_total: int,
    ) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
        """Impl-10-pre-2 (O9.1 + O9.2 + B, S1 fix) — mb phase 분리 + Option A 분배.

        ARCH §5.2 uniform-chunk + 사용자 의도 정합. Impl-10 main — *prefill_processed dict*
        동시 산출 (PREFILL_ATTN causal ctx 산출 입력, ARCH §3.5.2 정합).

        Total GPU PREFILL_ATTN work = N × chunk_uniform × per_token ≈ t_pim × margin − t_proj
        → t_qkv + t_prefill_attn + t_oproj ≈ t_pim × margin → 양쪽 idle 최소.

        Edge case: N > budget → chunk_per_req=0 → prefill_chunk 비어 있음 (decode-only cycle).
        Returns (prefill_chunk, decode_tokens, prefill_processed).
        """
        prefill_chunk: dict[int, list[int]] = {}
        decode_tokens: dict[int, int] = {}
        prefill_processed: dict[int, int] = {}
        prefill_reqs = []
        for req in reqs:
            remaining = len(req.prompt_tokens) - req.prefill_processed
            if remaining > 0:
                prefill_reqs.append((req, remaining))
            else:
                decode_tokens[req.id] = 0
        if prefill_reqs and chunk_budget_total > 0:
            n_prefill = len(prefill_reqs)
            chunk_per_req = chunk_budget_total // n_prefill
            min_remaining = min(r[1] for r in prefill_reqs)
            chunk_uniform = min(chunk_per_req, min_remaining)
            if chunk_uniform > 0:
                for req, _ in prefill_reqs:
                    prefill_chunk[req.id] = list(range(
                        req.prefill_processed, req.prefill_processed + chunk_uniform,
                    ))
                    # Impl-10 main — req.prefill_processed 시점 snapshot (causal ctx 산출 입력)
                    prefill_processed[req.id] = req.prefill_processed
        return prefill_chunk, decode_tokens, prefill_processed

    def _per_mb_kv_budget(self) -> int:
        """Phase-2 — per-slot KV 예산 = 전역 KV / 활성 슬롯 목표(2). 2-슬롯 disjoint 분할의
        *원리적* 한계 (Phase-1 의 "땜질" 아님).

        풀 모델 (나) 결정: 디코더를 2 active μ-batch 슬롯으로 disjoint 분할 → 더블 버퍼링
        (A 내부 proj∥attn) + 인스턴스 A∥B(F3) overlap. 한 슬롯이 KV 독점하면 둘째 슬롯이
        못 생겨 overlap 불가 → 슬롯당 KV 절반. 분모 = `_STAGGERING_TARGET_MB`(=2, 활성 슬롯
        목표), window.capacity 보다 크지 않게 clamp (F2 ablation cap=1 → 분모 1 → 단일 슬롯).
        """
        divisor = min(_STAGGERING_TARGET_MB, self.window.capacity)
        if divisor <= 0:
            return self.kv_accountant.capacity
        return self.kv_accountant.capacity // divisor

    def _invoke_admission(self, event: Event) -> MicroBatchSpec | None:
        t_proj = event.payload.get("t_proj", 0.0)
        t_pim_fn = event.payload.get("t_pim_fn", lambda n: 0.0)
        a_cycle = event.payload.get("a_cycle", 0.0)
        b_cycle = event.payload.get("b_cycle", 0.0)
        ctx_tokens = event.payload.get("ctx_tokens", 0)
        gpu_op_time_per_token_us = event.payload.get("gpu_op_time_per_token_us", 0.0)
        return self.admission.layer1(
            t_proj, t_pim_fn, a_cycle, b_cycle, ctx_tokens,
            gpu_op_time_per_token_us=gpu_op_time_per_token_us,
            max_mb_kv_tokens=self._per_mb_kv_budget(),
        )

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
