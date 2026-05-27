import pytest

from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


def _make_req(req_id: int) -> Request:
    return Request(id=req_id, prompt_tokens=[1, 2, 3], kv_length=10)


def test_push_returns_true_when_capacity_available():
    q = RequestQueue(capacity=4)
    assert q.push(_make_req(0)) is True
    assert len(q) == 1


def test_push_returns_false_when_capacity_exceeded():
    q = RequestQueue(capacity=2)
    assert q.push(_make_req(0)) is True
    assert q.push(_make_req(1)) is True
    assert q.push(_make_req(2)) is False
    assert len(q) == 2


def test_pop_oldest_FIFO_order():
    q = RequestQueue(capacity=4)
    reqs = [_make_req(i) for i in range(3)]
    for r in reqs:
        q.push(r)
    popped = [q.pop_oldest() for _ in range(3)]
    assert [r.id for r in popped] == [0, 1, 2]


def test_peek_oldest_does_not_remove():
    q = RequestQueue(capacity=4)
    r = _make_req(7)
    q.push(r)
    assert q.peek_oldest() is r
    assert q.peek_oldest() is r
    assert len(q) == 1


def test_pop_empty_returns_none():
    q = RequestQueue(capacity=4)
    assert q.pop_oldest() is None


def test_peek_empty_returns_none():
    q = RequestQueue(capacity=4)
    assert q.peek_oldest() is None


@pytest.mark.parametrize("capacity", [1, 2, 1024])
def test_capacity_boundary_parametrize(capacity: int):
    q = RequestQueue(capacity=capacity)
    for i in range(capacity):
        assert q.push(_make_req(i)) is True
    assert q.push(_make_req(capacity)) is False
    assert len(q) == capacity
