import heapq

from puls_sched.clock import Clock
from puls_sched.event import Event


class EventQueue:
    def __init__(self, clock: Clock):
        self._heap: list[tuple[float, int, Event]] = []
        self._seq: int = 0
        self._clock = clock

    def push(self, event: Event) -> None:
        if event.timestamp < self._clock.now:
            raise ValueError(
                f"past-event push: {event.timestamp} < clock.now={self._clock.now}"
            )
        heapq.heappush(self._heap, (event.timestamp, self._seq, event))
        self._seq += 1

    def pop(self) -> Event:
        if not self._heap:
            raise IndexError("pop from empty queue")
        timestamp, _, event = heapq.heappop(self._heap)
        self._clock.advance_to(timestamp)
        return event

    def __len__(self) -> int:
        return len(self._heap)

    def peek_timestamp(self) -> float | None:
        return self._heap[0][0] if self._heap else None
