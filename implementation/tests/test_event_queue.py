import pytest

from puls_sched.clock import Clock
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue


def _make_event(t: float, tag: str = "") -> Event:
    return Event(timestamp=t, type=EventType.KERNEL_COMPLETION, payload={"tag": tag})


def test_pop_order_by_timestamp():
    clock = Clock()
    queue = EventQueue(clock)
    for t in [3.0, 1.0, 2.5, 0.5, 5.0]:
        queue.push(_make_event(t))
    popped = []
    while len(queue) > 0:
        popped.append(queue.pop().timestamp)
    assert popped == sorted([3.0, 1.0, 2.5, 0.5, 5.0])


def test_tie_break_by_insertion_order():
    clock = Clock()
    queue = EventQueue(clock)
    queue.push(_make_event(1.0, "first"))
    queue.push(_make_event(1.0, "second"))
    queue.push(_make_event(1.0, "third"))
    assert queue.pop().payload["tag"] == "first"
    assert queue.pop().payload["tag"] == "second"
    assert queue.pop().payload["tag"] == "third"


def test_pop_advances_clock():
    clock = Clock()
    queue = EventQueue(clock)
    queue.push(_make_event(42.0))
    queue.pop()
    assert clock.now == 42.0


def test_past_event_push_rejected():
    clock = Clock()
    queue = EventQueue(clock)
    queue.push(_make_event(10.0))
    queue.pop()  # clock advances to 10.0
    with pytest.raises(ValueError, match="past-event"):
        queue.push(_make_event(5.0))


def test_empty_pop_raises():
    clock = Clock()
    queue = EventQueue(clock)
    with pytest.raises(IndexError):
        queue.pop()
