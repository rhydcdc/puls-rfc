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
                    self.dispatcher.tick()
                    return
                self.dispatcher.on_completion(event)
                # Impl-6 (Q5) — O_PROJ done 분기 → LayerState.advance → L 도달 시 token decode signal
                self._maybe_advance_forward_pass(event)
                self.dispatcher.tick()
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
                    if self.enable_admission_tick_rescheduling:
                        self._schedule_next_admission_tick(event)
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
                    prefill_chunk, decode_tokens = self._populate_mb_phases(
                        spec.decode_requests, spec.prefill_chunk_tokens,
                    )
                    mb = MicroBatch(
                        id=mb_id,
                        kv_rows_total=spec.kv_rows_total,
                        kv_rows_lockstep=spec.kv_rows_lockstep,   # Impl-8 — F5 ablation 위 signal flow
                        prefill_chunk=prefill_chunk,                  # Impl-10-pre-2 (O9.1)
                        decode_tokens=decode_tokens,
                        prefill_chunk_budget=spec.prefill_chunk_tokens,   # Impl-10-pre-2 B — adaptive budget 보존
                    )
                    self.dispatcher.register(mb)
                    self.window.admit(mb_id)
                    self.dispatcher.tick()
                # Impl-9 Q1 — self-rescheduling. spec=None 도 chain 유지 (idle termination guard 가 stop 결정).
                # opt-in flag (Run.init 가 enable) — isolated unit test 와 의미 분리 (R14).
                if self.enable_admission_tick_rescheduling:
                    self._schedule_next_admission_tick(event)

    def _schedule_next_admission_tick(self, prev_event: Event) -> None:
        """Impl-9 Q1 — ADMISSION_TICK 처리 직후 다음 tick auto-push.

        ARCH §6.4 'per-iteration basis' 정합. Idle termination guard:
        request_queue + in_flight_requests 동시 empty 시 self-push 중단 (Run.loop 3 조건 정합).

        Impl-10-pre-1 (B)~(B''') — prev_event.payload 영원 propagate (0 영원 trivial) → *진정 측정
        payload* 재구성 (`_compose_admission_payload`). 5 개 입력 모두 scheduler 자기 상태에서 산출.
        """
        if len(self.request_queue) == 0 and len(self.in_flight_requests) == 0:
            return
        next_t = self.clock.now + self.config.admission.tick_interval_us
        self.queue.push(Event(
            timestamp=next_t,
            type=EventType.ADMISSION_TICK,
            payload=self._compose_admission_payload(),
        ))

    def _schedule_admission_tick_with_default_payload(self) -> None:
        """REQUEST_ARRIVAL 위 admission chain 재기동.

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

        ARCH §6.4 *"GPU/PIM idle fractions of the previous iteration are measured to regulate
        next μ-batch's admission"* literal 정합 — 모든 입력 scheduler 자기 상태 derived.

        - (B) a_cycle / b_cycle — IdleTelemetry active_duration delta (이전 ADMISSION_TICK 대비)
        - (B') t_proj — config 의 projection (QKV + O_PROJ) GEMM 시간 합
        - (B'') t_pim_fn — PIMExecutor.op_time 의 closure (avg kv_length 위 동적)
        - (B''') ctx_tokens — in_flight_requests 의 max kv_length (deadband ctx-tier 입력)
        """
        a_cycle, b_cycle = self._measure_cycles()
        t_proj = (
            self.config.time.gpu_op_time_us.get("qkv", 0.0)
            + self.config.time.gpu_op_time_us.get("o_proj", 0.0)
        )
        ctx_tokens = max(
            (r.kv_length for r in self.in_flight_requests.values()), default=0,
        )
        return {
            "t_proj": t_proj,
            "t_pim_fn": self._make_t_pim_fn(),
            "a_cycle": a_cycle,
            "b_cycle": b_cycle,
            "ctx_tokens": ctx_tokens,
            # Impl-10-pre-2 (B option) — PIM-slack adaptive chunk 위 per-token GPU op time 전달
            "gpu_op_time_per_token_us": self.config.time.gpu_op_time_per_token_us,
        }

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
        Signature: fn(n_decode) → float.
        """
        in_flight = self.in_flight_requests
        pim_executor = self.dispatcher.pim_executor
        def fn(n_decode: int) -> float:
            if not in_flight or n_decode <= 0:
                return 0.0
            avg_kv = max(1, sum(r.kv_length for r in in_flight.values()) // len(in_flight))
            return pim_executor.op_time(kv_rows_total=n_decode * avg_kv)
        return fn

    def _maybe_advance_forward_pass(self, event: Event, eos_seen: bool = False) -> None:
        """KERNEL_COMPLETION (O_PROJ done) → LayerState.advance → L 도달 시 token decode signal.

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
        if node_type is not NodeType.O_PROJ:
            return
        mb_id = event.payload.get("micro_batch_id")
        mb = self.dispatcher.micro_batches.get(mb_id) if mb_id is not None else None
        if mb is None:
            return  # defensive — mb already unregistered
        # Impl-10-pre-1 (A) — per-layer A→B chain wiring (production hot path)
        if self.instance_pipeline is not None:
            self.instance_pipeline.dispatch(mb)
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
        # Impl-10-pre-2 (O9.1) — mb 영역의 모든 req 활동 영역 추적 (prefill_chunk + decode_tokens)
        all_req_ids = set(mb.prefill_chunk.keys()) | set(mb.decode_tokens.keys())
        mb_has_active_reqs = any(rid in self.in_flight_requests for rid in all_req_ids)
        if not mb_has_active_reqs:
            self.window.evict(mb.id)
            self.dispatcher.unregister(mb.id)
        else:
            # Re-compose mb.prefill_chunk + mb.decode_tokens — 다음 cycle 영역 위 갱신 (Sarathi 정합)
            self._recompose_mb(mb)
            self.dag.reset_micro_batch(mb.id)

    def _recompose_mb(self, mb: MicroBatch) -> None:
        """Impl-10-pre-2 (O9.1 + B) — mb 의 다음 L-cycle 위 prefill_chunk + decode_tokens 갱신.

        Adaptive budget 보존 — mb.prefill_chunk_budget (admission 위 산출) 영원 사용.
        Fallback (legacy mb 위 budget=0) — prefill_chunk_default.
        """
        budget = mb.prefill_chunk_budget if mb.prefill_chunk_budget > 0 else self.config.admission.prefill_chunk_default
        active_req_ids = set(mb.prefill_chunk.keys()) | set(mb.decode_tokens.keys())
        active_reqs = [
            self.in_flight_requests[rid]
            for rid in active_req_ids
            if rid in self.in_flight_requests
        ]
        new_prefill_chunk, new_decode_tokens = self._populate_mb_phases(
            active_reqs, budget,
        )
        mb.prefill_chunk = new_prefill_chunk
        mb.decode_tokens = new_decode_tokens

    def _populate_mb_phases(
        self, reqs, chunk_budget_total: int,
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        """Impl-10-pre-2 (O9.1 + O9.2 + B, S1 fix) — mb phase 분리 + Option A 분배.

        ARCH §5.2 uniform-chunk + 사용자 의도 정합 — admission 의 chunk_budget_total 은
        *전체 prefill reqs 합산* 위 GPU PREFILL_ATTN 시간이 (t_pim × margin − t_proj) 와 같아지는
        총 token 수. N 개 prefill reqs 위 균등 분배:

            chunk_per_req = chunk_budget_total // N
            chunk_uniform = min(chunk_per_req, min(remaining over prefill reqs))

        Total GPU PREFILL_ATTN work = N × chunk_uniform × per_token ≈ t_pim × margin − t_proj
        → t_qkv + t_prefill_attn + t_oproj ≈ t_pim × margin → 양쪽 idle 최소.

        Edge case: N > budget → chunk_per_req=0 → prefill_chunk 비어 있음 (decode-only cycle).
        Returns (prefill_chunk, decode_tokens).
        """
        prefill_chunk: dict[int, list[int]] = {}
        decode_tokens: dict[int, int] = {}
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
        return prefill_chunk, decode_tokens

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
        )

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
