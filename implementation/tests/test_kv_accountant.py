import pytest

from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int) -> Request:
    return Request(id=req_id, prompt_len=1, kv_length=kv_length)


def test_admit_reduces_remaining():
    acc = KVAccountant(capacity=1000)
    acc.admit(_make_req(0, 100))
    assert acc.remaining == 900


def test_release_restores_remaining():
    acc = KVAccountant(capacity=1000)
    req = _make_req(0, 100)
    acc.admit(req)
    acc.release(req)
    assert acc.remaining == 1000


@pytest.mark.parametrize("n", [1, 10, 100])
def test_admit_release_N_times_bit_exact_remaining(n: int):
    acc = KVAccountant(capacity=10_000)
    initial = acc.remaining
    for i in range(n):
        req = _make_req(i, 50)
        acc.admit(req)
        acc.release(req)
    assert acc.remaining == initial


def test_admit_overflow_raises():
    acc = KVAccountant(capacity=100)
    with pytest.raises(ValueError, match="overflow"):
        acc.admit(_make_req(0, 101))


def test_admit_double_raises():
    acc = KVAccountant(capacity=1000)
    req = _make_req(0, 100)
    acc.admit(req)
    with pytest.raises(ValueError, match="double-admit"):
        acc.admit(req)


def test_release_unadmitted_raises():
    acc = KVAccountant(capacity=1000)
    with pytest.raises(ValueError, match="underflow"):
        acc.release(_make_req(99, 50))


def test_can_admit_is_pure_read():
    acc = KVAccountant(capacity=1000)
    before_used = acc.used
    before_remaining = acc.remaining
    acc.can_admit(_make_req(0, 100))
    assert acc.used == before_used
    assert acc.remaining == before_remaining


def test_can_admit_true_when_fits_exactly():
    acc = KVAccountant(capacity=100)
    assert acc.can_admit(_make_req(0, 100)) is True


def test_can_admit_false_when_overflow_by_one():
    acc = KVAccountant(capacity=100)
    assert acc.can_admit(_make_req(0, 101)) is False


def test_partial_admit_partial_release_correctness():
    acc = KVAccountant(capacity=1000)
    r0 = _make_req(0, 100)
    r1 = _make_req(1, 200)
    r2 = _make_req(2, 300)
    acc.admit(r0)
    acc.admit(r1)
    acc.admit(r2)
    acc.release(r1)
    assert acc.remaining == 1000 - 100 - 300


def test_stress_100_admit_release_no_overflow_underflow():
    acc = KVAccountant(capacity=10_000)
    initial = acc.remaining
    reqs = [_make_req(i, 50) for i in range(100)]
    for r in reqs:
        acc.admit(r)
    for r in reqs:
        acc.release(r)
    assert acc.remaining == initial
    assert acc.used == 0
