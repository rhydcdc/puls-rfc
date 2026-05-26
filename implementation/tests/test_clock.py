import pytest

from puls_sched.clock import Clock


def test_now_monotonic():
    clock = Clock()
    timestamps = []
    for t in [0.0, 1.0, 1.5, 2.0, 100.0]:
        clock.advance_to(t)
        timestamps.append(clock.now)
    assert timestamps == sorted(timestamps)


def test_advance_to_updates_time():
    clock = Clock()
    clock.advance_to(42.0)
    assert clock.now == 42.0


def test_backward_advance_rejected():
    clock = Clock()
    clock.advance_to(10.0)
    with pytest.raises(ValueError, match="non-monotonic"):
        clock.advance_to(5.0)
