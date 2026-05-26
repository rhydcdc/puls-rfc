from dataclasses import dataclass

from puls_sched.clock import Clock
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.window import InFlightWindow


@dataclass
class SchedulerCore:
    config: Config
    clock: Clock
    queue: EventQueue
    dag: DAG
    window: InFlightWindow

    def step(self) -> bool:
        if len(self.queue) == 0:
            return False
        event = self.queue.pop()
        self._handle(event)
        return True

    def _handle(self, event: Event) -> None:
        match event.type:
            case EventType.KERNEL_COMPLETION:
                pass
            case EventType.REQUEST_ARRIVAL:
                pass
            case EventType.ADMISSION_TICK:
                pass

    def run_until_empty(self) -> int:
        n = 0
        while self.step():
            n += 1
        return n
