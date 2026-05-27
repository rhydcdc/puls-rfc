from dataclasses import dataclass

from puls_sched.admission import Admission, MicroBatchSpec
from puls_sched.clock import Clock
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.kv_accountant import KVAccountant
from puls_sched.micro_batch import MicroBatch
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


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
    _next_mb_id: int = 0

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
                self.dispatcher.tick()
            case EventType.REQUEST_ARRIVAL:
                req = event.payload["request"]
                self.request_queue.push(req)
            case EventType.ADMISSION_TICK:
                spec = self._invoke_admission(event)
                if spec is None:
                    return
                mb_id = self._next_mb_id
                self._next_mb_id += 1
                # Impl-5 (Q1) — spec → MicroBatch 변환. decode_tokens · prefill_chunk 의 실 token data 는
                # Impl-6 영역 (token sampling + prefill chunk schedule). 본 단계는 spec 의 결정 정보 운반.
                mb = MicroBatch(
                    id=mb_id,
                    k_total=spec.k_total,
                    kv_rows_total=spec.kv_rows_total,
                )
                self.dispatcher.register(mb)
                self.window.admit(mb_id)
                self.dispatcher.tick()

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
