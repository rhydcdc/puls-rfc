from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from puls_sched.admission import Admission
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
    # ---- Phase-2 S4(b) — 완료 요청 sink (측정용). 완료 시 pop 후 버리지 않고 보존 →
    # measure_steady 가 TTFT/TBT 산출. (스케줄링 로직 무관, 진단 substrate.) ----
    completed_requests: list[Request] = field(default_factory=list)
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

    def _fire_admission_tick(self) -> None:
        """등록된 callback 들에게 admission tick snapshot 통지 (Impl-8 D1 hook fire, 진단용).

        Phase-2 풀 모델 — 동작점 고정으로 cycle 측정 삭제(a/b_cycle=0). n = 현재 활성 μ-batch
        들의 decode 토큰 총수(진단). idle 사후 진단은 evaluator 가 idle_telemetry 로 담당.
        """
        if not self._admission_tick_callbacks:
            return
        from puls_sched.evaluator import AdmissionSnapshot
        n_decode = sum(len(mb.decode_tokens) for mb in self.dispatcher.micro_batches.values())
        snapshot = AdmissionSnapshot(
            timestamp=self.clock.now,
            gpu_idle_fraction=self.admission.idle_telemetry.gpu_idle_fraction(),
            pim_idle_fraction=self.admission.idle_telemetry.pim_idle_fraction(),
            a_cycle=0.0,
            b_cycle=0.0,
            ctx_tokens=0,
            spec_admitted=(n_decode > 0 or len(self.window.current_ids()) > 0),
            n=n_decode,
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
                # Phase-2 풀 모델 (OPERATING_POINT §3) — 세 관심사 분리:
                #  (1) admission = 풀 보충: request_queue → in_flight(PREFILL), KV 게이트만.
                #  (2)+(3) μ-batch 구성: decode-set(123/12.3M) ∥ prefill(256/25.6M) 을 풀에서
                #      *독립적으로* steering+age-cap 으로 구성 → window 를 활성 목표(2)까지 채움.
                self._refill_pool()
                self._fill_window()
                self._fire_admission_tick()
                # 고정 타이머 self-push 폐기. 다음 admission 은 완료/신규도착 시 재기동.

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
            if req.prefill_processed >= req.prompt_len and req.state == RequestState.PREFILL:
                req.transition_to(RequestState.DECODE)
        # Decode token 생성 (decode_tokens 위 reqs 만)
        for req_id in list(mb.decode_tokens.keys()):
            req = self.in_flight_requests.get(req_id)
            if req is None:
                continue
            req.decoded_count += 1
            if req.first_token_time is None:        # S4(b) — 첫 decode 토큰 = TTFT marker
                req.first_token_time = self.clock.now
            if req.state == RequestState.PREFILL:
                # Defensive — prefill_processed 이미 도달했을 가능성 (race-free 보장 위)
                req.transition_to(RequestState.DECODE)
            if self.completion.check(req, eos_seen=eos_seen):
                self.completion.finalize(req)
                self.in_flight_requests.pop(req_id, None)
                self.completed_requests.append(req)   # S4(b) — sink (버리지 않고 보존)
        # 다음 forward pass 위 reset (multi-token decode 정합)
        mb.current_layer_index = 0
        # Phase-2 풀 모델 — forward pass 끝 → 이 μ-batch 를 *풀에서 재구성*(decode-set·prefill
        # 재선택). 직전 멤버는 풀로 반환(in_flight 잔류·미assign)되어 재선택 대상. 재구성 결과
        # 비면(풀에 줄 게 없음) evict. 그 뒤 window 를 활성 목표까지 다시 채움(해제된 멤버 흡수).
        self._recompose_mb(mb)
        if mb.prefill_chunk or mb.decode_tokens:
            self.dag.reset_micro_batch(mb.id)
        else:
            self.window.evict(mb.id)
            self.dispatcher.unregister(mb.id)
        self._fill_window()

    def _refill_pool(self) -> None:
        """Admission (풀 모델, OPERATING_POINT §3) — request_queue → in_flight(PREFILL), KV 게이트만.

        구성 로직 아님: 디코드/prefill 타깃과 *무관* 하게 풀을 보충만 한다. KV 여유(can_admit)
        까지 새 요청을 들이고 PENDING→PREFILL 전이. KV admit 1회(완료 시 release).
        """
        while True:
            req = self.request_queue.peek_oldest()
            if req is None or not self.kv_accountant.can_admit(req):
                break
            self.request_queue.pop_oldest()
            self.kv_accountant.admit(req)
            if req.state == RequestState.PENDING:
                req.transition_to(RequestState.PREFILL)
            # 프롬프트 잔여 0(decode-only, 또는 이미 prefill 끝남) → 즉시 DECODE 풀로.
            if req.prefill_processed >= req.prompt_len and req.state == RequestState.PREFILL:
                req.transition_to(RequestState.DECODE)
            self.in_flight_requests[req.id] = req

    def _assigned_ids(self, exclude_mb_id: int | None = None) -> set[int]:
        """현재 활성 μ-batch 들이 점유(assign)한 요청 id 합 (disjoint 보장).

        exclude_mb_id 의 멤버는 제외 — 그 μ-batch 재구성 시 자기 직전 멤버는 풀로 반환돼
        재선택 대상이므로(다른 μ-batch 의 멤버만 제외).
        """
        assigned: set[int] = set()
        for active in self.dispatcher.micro_batches.values():
            if active.id == exclude_mb_id:
                continue
            assigned |= set(active.decode_tokens) | set(active.prefill_chunk)
        return assigned

    def _select_decoders(self, exclude: set[int]) -> list[Request]:
        """decode-set 구성 (§3) — in_flight DECODE 풀(exclude 제외)서 steering+age-cap 으로
        123/12.3M 선택. prefill 과 *완전 별개* 축."""
        pool = [r for r in self.in_flight_requests.values()
                if r.state == RequestState.DECODE and r.id not in exclude]
        return self.admission.steer_decode_set(pool)

    def _prefill_pool(self, exclude: set[int]) -> list[Request]:
        """prefill 풀 — in_flight PREFILL 이며 프롬프트 잔여 있는 요청(exclude 제외)."""
        return [r for r in self.in_flight_requests.values()
                if r.state == RequestState.PREFILL
                and r.prefill_processed < r.prompt_len and r.id not in exclude]

    def _build_mb_fields(self, decoders, prefill_chunk, prefill_processed) -> dict:
        return dict(
            kv_rows_total=sum(d.kv_length for d in decoders),
            kv_rows_lockstep=max((d.kv_length for d in decoders), default=0) * len(decoders),
            prefill_chunk=prefill_chunk,
            decode_tokens={d.id: 0 for d in decoders},
            prefill_chunk_budget=self.config.admission.prefill_chunk_default,
            prefill_processed=prefill_processed,
        )

    def _compose_microbatch(self) -> "MicroBatch | None":
        """풀에서 새 μ-batch 1개 구성 — decode-set(§3) ∥ prefill(256) *독립* 구성, disjoint.
        둘 다 비면 None (풀에 줄 게 없음)."""
        exclude = self._assigned_ids()
        decoders = self._select_decoders(exclude)
        prefill_chunk, _dec, prefill_processed = self._populate_mb_phases(
            self._prefill_pool(exclude), self.config.admission.prefill_chunk_default,
        )
        if not decoders and not prefill_chunk:
            return None
        mb_id = self._next_mb_id
        self._next_mb_id += 1
        mb = MicroBatch(id=mb_id, **self._build_mb_fields(decoders, prefill_chunk, prefill_processed))
        self.dispatcher.register(mb)
        self.window.admit(mb_id)
        self.dispatcher.tick()
        return mb

    def _fill_window(self) -> None:
        """활성 μ-batch 수를 staggering 목표(2, window.capacity clamp)까지 풀에서 구성·채움
        (F3/더블버퍼링 위 동시 2 μ-batch — ARCH §5.6/§5.7, OPERATING_POINT §3 window=3)."""
        target = min(_STAGGERING_TARGET_MB, self.window.capacity)
        while len(self.window.current_ids()) < target:
            if self._compose_microbatch() is None:
                break

    def _recompose_mb(self, mb: MicroBatch) -> None:
        """forward pass 후 이 μ-batch 를 *풀에서 재구성* (decode-set·prefill 재선택, §3).

        직전 멤버는 풀로 반환(in_flight 잔류, 이 mb 제외 시 미assign)되어 재선택 대상. 다른
        활성 μ-batch 멤버는 제외(disjoint). 풀에 줄 게 없으면 빈 mb (caller 가 evict).
        """
        exclude = self._assigned_ids(exclude_mb_id=mb.id)
        decoders = self._select_decoders(exclude)
        prefill_chunk, _dec, prefill_processed = self._populate_mb_phases(
            self._prefill_pool(exclude), self.config.admission.prefill_chunk_default,
        )
        fields = self._build_mb_fields(decoders, prefill_chunk, prefill_processed)
        mb.decode_tokens = fields["decode_tokens"]
        mb.prefill_chunk = fields["prefill_chunk"]
        mb.prefill_processed = fields["prefill_processed"]
        mb.kv_rows_total = fields["kv_rows_total"]
        mb.kv_rows_lockstep = fields["kv_rows_lockstep"]

    def _populate_mb_phases(
        self, reqs, chunk_budget_total: int,
    ) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
        """Phase-2 former-v2 — mb phase 분리 + prefill steering (OPERATING_POINT §3).

        prefill 토큰 *총수* 는 256 고정(= FFN batch 기여분, decode 와 함께 batch 형성)이되,
        그 256 을 prefill 멤버들에 *어떻게 나누는가* 로 GPU-A PREFILL_ATTN 의 **depth-합**
        (Σ chunk×depth)을 동작점 25.6M(`prefill_kv_work_target_tokens`)에 맞춘다. decode
        steering 과 동형의 로컬 그리디: 매 토큰마다 `ideal=(target−W)/(budget−t)` 깊이에
        가장 가까운 멤버에 1 토큰 배정 → depth-합이 목표에 자기보정 수렴.

        깊은(off-ideal) 멤버는 적게, 얕은 멤버는 많이 받는다(causal: j-번째 토큰 깊이 =
        prefill_processed + j). 멤버별 chunk 가 달라도(ragged) FFN(B)은 batch *총 토큰수* 만
        보므로 fixed-shape 핸드오프와 무관(OPERATING_POINT §3, 사용자 확인 2026-06-02) —
        straggler 제거는 PIM 의 attention 길이변동 흡수에서 오지 prefill chunk 균등이 아님.

        Impl-10 main — *prefill_processed dict* 동시 산출(PREFILL_ATTN causal ctx 입력).
        Edge: 남은 프롬프트 토큰 < budget → 가능한 만큼만(decode-only cycle 이면 prefill 빔).
        Returns (prefill_chunk, decode_tokens, prefill_processed).
        """
        prefill_chunk: dict[int, list[int]] = {}
        decode_tokens: dict[int, int] = {}
        prefill_processed: dict[int, int] = {}
        prefill_reqs = []
        for req in reqs:
            remaining = req.prompt_len - req.prefill_processed
            if remaining > 0:
                prefill_reqs.append((req, remaining))
            else:
                decode_tokens[req.id] = 0
        if prefill_reqs and chunk_budget_total > 0:
            target_work = self.config.admission.prefill_kv_work_target_tokens
            age_cap = self.config.admission.age_cap
            by_id = {req.id: req for req, _ in prefill_reqs}
            remaining = {req.id: rem for req, rem in prefill_reqs}
            alloc = {req.id: 0 for req, _ in prefill_reqs}
            budget = min(chunk_budget_total, sum(remaining.values()))
            W = 0      # 누적 depth-work = Σ(배정 토큰의 causal 깊이)
            t = 0      # 누적 배정 토큰
            while t < budget:
                cand = [rid for rid in alloc if alloc[rid] < remaining[rid]]
                if not cand:
                    break
                # age-cap (decode 와 동형): 이번 사이클 토큰 0개인데 prefill_wait ≥ age_cap 으로
                # 오래 기다린 요청은 steering 무시하고 강제 1 토큰 → prefill starvation 0.
                aged = [rid for rid in cand
                        if alloc[rid] == 0 and by_id[rid].prefill_wait >= age_cap]
                if aged:
                    pick = max(aged, key=lambda rid: by_id[rid].prefill_wait)   # 가장 오래 기다린 것
                else:
                    # steering: depth-합 25.6M 수렴 — 다음 토큰의 이상 깊이에 가장 가까운 요청.
                    # 다음 토큰의 깊이 = prefill_processed + 이미 배정한 수 (causal).
                    ideal = (target_work - W) / (budget - t)
                    pick = min(cand, key=lambda rid: abs(
                        (by_id[rid].prefill_processed + alloc[rid]) - ideal))
                W += by_id[pick].prefill_processed + alloc[pick]
                alloc[pick] += 1
                t += 1
            # 풀 모델: 토큰 받은(c>0) 요청만 prefill_chunk(=이 μ-batch 에 assign). 0토큰 요청은
            # *풀에 잔류*(이 μ-batch 에 안 넣음) → 다음 구성서 재선택, prefill_wait++ 로 age-cap.
            # (sticky 모델의 빈-chunk 멤버십 유지는 풀 모델선 mb 를 미진행 요청으로 부풀려 prefill
            #  256 을 과분산 → prefill 정체·decode 고갈. 그래서 제거.)
            for req, _ in prefill_reqs:
                c = alloc[req.id]
                if c > 0:
                    prefill_chunk[req.id] = list(range(
                        req.prefill_processed, req.prefill_processed + c,
                    ))
                    prefill_processed[req.id] = req.prefill_processed
                    req.prefill_wait = 0
                else:
                    req.prefill_wait += 1
        return prefill_chunk, decode_tokens, prefill_processed

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
