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

    def on_admission_tick(self, callback: AdmissionTickCallback) -> None:
        """Admission tick snapshot capture 위 callback 등록 (Impl-8 D1 hook).

        Evaluator 같은 외부 inspector 가 admission_convergence series 캡처 위 등록.
        SchedulerCore 자체는 Evaluator 를 모름 (D3 standalone).
        """
        self._admission_tick_callbacks.append(callback)

    def _fire_admission_tick(self, spec: "MicroBatchSpec | None") -> None:
        """등록된 callback 들에게 admission tick snapshot 통지 (Impl-8 D1 hook fire).

        Phase-2 S2 (§2.5) — 동작점 고정으로 cycle 측정 기계장치 삭제. a_cycle/b_cycle 는
        admission 경로에서 산출하지 않음(=0) — idle_telemetry 기반 사후 진단은 evaluator 가
        담당(밸런스 입력에서 분리). snapshot 은 진단용으로만 보존. n = admit 된 디코더 수.
        Spec=None (admission 실패 path) 도 snapshot 누적.
        """
        if not self._admission_tick_callbacks:
            return
        from puls_sched.evaluator import AdmissionSnapshot
        snapshot = AdmissionSnapshot(
            timestamp=self.clock.now,
            gpu_idle_fraction=self.admission.idle_telemetry.gpu_idle_fraction(),
            pim_idle_fraction=self.admission.idle_telemetry.pim_idle_fraction(),
            a_cycle=0.0,
            b_cycle=0.0,
            ctx_tokens=0,
            spec_admitted=(spec is not None),
            n=len(spec.decode_requests) if spec else 0,
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
                        self._schedule_admission_tick()
                    return
                self.dispatcher.on_completion(event)
                # Impl-6 (Q5) — O_PROJ done 분기 → LayerState.advance → L 도달 시 token decode signal
                self._maybe_advance_forward_pass(event)
                self.dispatcher.tick()
                # STEP 2.5 — 완료 시 admission (자원이 비는 유일 시점). 고정 타이머 self-push 폐기.
                if self.enable_admission_tick_rescheduling:
                    self._schedule_admission_tick()
            case EventType.REQUEST_ARRIVAL:
                req = event.payload["request"]
                self.request_queue.push(req)
                # Impl-9 Q1 — Arrival re-wakes admission chain (idle guard 의 dual entry).
                # ARCH §6.4 'per-iteration admission' 의 arrival-driven 재기동 의미 정합.
                if self.enable_admission_tick_rescheduling:
                    self._schedule_admission_tick()
            case EventType.ADMISSION_TICK:
                # Phase-2 S2 (§2.5) — 동작점 고정으로 payload trivial. cycle 측정 인자 없음.
                # Impl-9 — Window full 시 admission 대기 (ARCH §6.7 '3-μ-batch in-flight window' 의미).
                # Auto-evict (window.admit overflow) 는 *defensive* 영역으로 격하.
                if len(self.window.current_ids()) >= self.window.capacity:
                    self._fire_admission_tick(None)
                    # STEP 2.5 — window full 이면 다음 완료(evict) 시 admission 재기동.
                    # 고정 타이머 self-push 폐기 (헛도는 tick 차단).
                    return
                spec = self._invoke_admission()
                self._fire_admission_tick(spec)
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

    def _schedule_admission_tick(self) -> None:
        """완료(KERNEL_COMPLETION) / 신규 도착(REQUEST_ARRIVAL) 시 admission 재기동.

        STEP 2.5 — 고정 타이머 self-push 폐기 후 admission 의 유일한 재기동 경로. 이벤트
        기반: 자원이 비는 시점(완료)과 새 일감 도착 시점에만 admission tick push.

        Phase-2 S2 (§2.5) — 동작점 고정으로 payload trivial(빈 dict). former 는 KV 합·prefill
        512 만 보므로 cycle 측정 payload 산출(`_compose_admission_payload`) 삭제.
        """
        if len(self.request_queue) == 0 and len(self.in_flight_requests) == 0:
            return
        next_t = self.clock.now + self.config.admission.tick_interval_us
        self.queue.push(Event(
            timestamp=next_t,
            type=EventType.ADMISSION_TICK,
            payload={},
        ))

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
        # Phase-2 S2 — 슬롯 재구성 = 잔존 멤버 phase 전진만(backfill 없음, §2.5). evict 판정 =
        # 재구성 후 prefill_chunk·decode_tokens 모두 빔(멤버 전원 완료). 배치_생애 §종료.
        self._recompose_mb(mb)
        if mb.prefill_chunk or mb.decode_tokens:
            self.dag.reset_micro_batch(mb.id)
        else:
            self.window.evict(mb.id)
            self.dispatcher.unregister(mb.id)

    def _recompose_mb(self, mb: MicroBatch) -> None:
        """Phase-2 S2 — 슬롯의 다음 L-cycle 위 prefill_chunk + decode_tokens 갱신.

        잔존 멤버(완료 안 된 req)만 sticky 유지 + phase 전진(다음 토큰용). **풀에서 당겨오지
        않음 — backfill 삭제(§2.5, 사용자 확정).** 밸런스가 정적 동작점으로 형성 시점에 이미
        세 시간이 맞춰져, "균형 맞추려 이미 형성된 배치에 더 합류"시킬 이유가 사라짐. 완료된
        멤버는 빠지고(슬롯 자연 축소), 새 부하는 새 μ-batch(ADMISSION_TICK former)로만 진입
        → 풀→배치 진입 경로가 former 하나로 단일화.
        """
        budget = mb.prefill_chunk_budget if mb.prefill_chunk_budget > 0 else self.config.admission.prefill_chunk_default
        active_req_ids = set(mb.prefill_chunk.keys()) | set(mb.decode_tokens.keys())
        active_req_ids = {rid for rid in active_req_ids if rid in self.in_flight_requests}
        active_reqs = [self.in_flight_requests[rid] for rid in active_req_ids]
        new_prefill_chunk, new_decode_tokens, new_prefill_processed = self._populate_mb_phases(
            active_reqs, budget,
        )
        mb.prefill_chunk = new_prefill_chunk
        mb.decode_tokens = new_decode_tokens
        mb.prefill_processed = new_prefill_processed   # Impl-10 main — re-composition causal ctx refresh

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
        """Phase-2 — per-slot KV 예산 = 전역 KV(60M) / 활성 슬롯 목표(2) = 30M. 2-슬롯 disjoint
        분할(§0.8 A안)의 *선언적* 한계.

        ★ 실제 admission 엔 비바인딩(§0.8/§2.5 사용자 확정): layer1 은 동작점 목표 Σkv=25M 에서
        먼저 멈추므로 이 30M 천장은 production 에서 절대 안 걸린다. 즉 "disjoint 2-슬롯 = 60M/2"
        개념을 *선언*으로 남겨둘 뿐, admission 로직을 실제로 제한하지 않는다(동작점이 답).
        분모 = `_STAGGERING_TARGET_MB`(=2), window.capacity 보다 크지 않게 clamp.
        """
        divisor = min(_STAGGERING_TARGET_MB, self.window.capacity)
        if divisor <= 0:
            return self.kv_accountant.capacity
        return self.kv_accountant.capacity // divisor

    def _invoke_admission(self) -> MicroBatchSpec | None:
        # Phase-2 S2 — 동작점 former (§2.5). cycle 측정·payload 인자 전부 제거 — layer1 은
        # 디코더를 Σkv 동작점(25M)까지 admit + prefill 512 고정. max_mb_kv_tokens 는 비바인딩
        # 선언(per-mb 30M, 목표 25M 이 먼저 멈춤 — _per_mb_kv_budget 각주).
        return self.admission.layer1(max_mb_kv_tokens=self._per_mb_kv_budget())

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
