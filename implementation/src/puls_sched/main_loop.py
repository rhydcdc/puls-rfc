from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from puls_sched.admission import Admission, MicroBatchSpec
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import LayerState
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
            k_total=spec.k_total if spec else 0,
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
                self.dispatcher.on_completion(event)
                # Impl-6 (Q5) — O_PROJ done 분기 → LayerState.advance → L 도달 시 token decode signal
                self._maybe_advance_forward_pass(event)
                self.dispatcher.tick()
            case EventType.REQUEST_ARRIVAL:
                req = event.payload["request"]
                self.request_queue.push(req)
            case EventType.ADMISSION_TICK:
                # Impl-8 — admission tick hook 위 spec + cycle 값 산출 (snapshot fire 영역)
                a_cycle = event.payload.get("a_cycle", 0.0)
                b_cycle = event.payload.get("b_cycle", 0.0)
                ctx_tokens = event.payload.get("ctx_tokens", 0)
                spec = self._invoke_admission(event)
                self._fire_admission_tick(spec, a_cycle, b_cycle, ctx_tokens)
                if spec is None:
                    return
                mb_id = self._next_mb_id
                self._next_mb_id += 1
                mb = MicroBatch(
                    id=mb_id,
                    k_total=spec.k_total,
                    kv_rows_total=spec.kv_rows_total,
                    kv_rows_lockstep=spec.kv_rows_lockstep,   # Impl-8 — F5 ablation 위 signal flow
                    # Q10 (b) — decode_tokens 는 dispatch metadata (placeholder int value). Request 가 lifecycle owner.
                    decode_tokens={req.id: 0 for req in spec.decode_requests},
                )
                # Impl-6 (Q10) — Request lifecycle owner = SchedulerCore.in_flight_requests
                for req in spec.decode_requests:
                    if req.state == RequestState.PENDING:
                        req.transition_to(RequestState.PREFILL)
                    self.in_flight_requests[req.id] = req
                self.dispatcher.register(mb)
                self.window.admit(mb_id)
                self.dispatcher.tick()

    def _maybe_advance_forward_pass(self, event: Event, eos_seen: bool = False) -> None:
        """KERNEL_COMPLETION (O_PROJ done) → LayerState.advance → L 도달 시 token decode signal.

        Q5 — consumer 는 main_loop 영역. Dispatcher / forward_pass 침범 0.
        Q6 (c) — eos_seen=True 명시 시 EOS branch 발동 (외부 caller / test fixture path).
        """
        node_type = event.payload.get("node_type")
        if node_type is not NodeType.O_PROJ:
            return
        mb_id = event.payload.get("micro_batch_id")
        mb = self.dispatcher.micro_batches.get(mb_id) if mb_id is not None else None
        if mb is None:
            return  # defensive — mb already unregistered
        token_signal = self.layer_state.advance(mb)
        if not token_signal:
            return
        # ---- L 도달 — token decode signal ----
        for req_id in list(mb.decode_tokens.keys()):
            req = self.in_flight_requests.get(req_id)
            if req is None:
                continue  # already completed — defensive (Q9 책임 분리 의 correctness invariant, R7)
            req.decoded_count += 1
            if req.state == RequestState.PREFILL:
                req.transition_to(RequestState.DECODE)
            if self.completion.check(req, eos_seen=eos_seen):
                self.completion.finalize(req)
                self.in_flight_requests.pop(req_id, None)
        # 다음 token 의 forward pass 위 reset (multi-token decode 정합)
        mb.current_layer_index = 0

    def _invoke_admission(self, event: Event) -> MicroBatchSpec | None:
        t_proj = event.payload.get("t_proj", 0.0)
        t_pim_fn = event.payload.get("t_pim_fn", lambda k, n: 0.0)
        a_cycle = event.payload.get("a_cycle", 0.0)
        b_cycle = event.payload.get("b_cycle", 0.0)
        ctx_tokens = event.payload.get("ctx_tokens", 0)
        return self.admission.layer1(t_proj, t_pim_fn, a_cycle, b_cycle, ctx_tokens)

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
