"""Impl-6 — Completion (check + finalize) unit tests.

PLAN §4 Impl-6 + §0.5 reminder (KV slot 회수 round-trip + completion 검출 boundary).
Q6 (c) hybrid · Q7 KV release timing · Q9 책임 분리 · Q10 Request lifecycle owner.
"""

import pytest

from puls_sched.completion import Completion
from puls_sched.request import Request, RequestState


def _make_req(req_id: int = 0, kv_length: int = 10, max_tokens: int = 5,
              state: RequestState = RequestState.DECODE) -> Request:
    r = Request(id=req_id, prompt_len=5, kv_length=kv_length,
                max_tokens=max_tokens)
    # transition to target state via valid path
    if state == RequestState.PREFILL or state == RequestState.DECODE:
        r.transition_to(RequestState.PREFILL)
    if state == RequestState.DECODE:
        r.transition_to(RequestState.DECODE)
    return r


# ============================================================================
# check — pure boundary
# ============================================================================

def test_completion_check_max_tokens_reached(completion):
    req = _make_req(max_tokens=10)
    req.decoded_count = 10
    assert completion.check(req) is True


def test_completion_check_max_tokens_below(completion):
    req = _make_req(max_tokens=10)
    req.decoded_count = 5
    assert completion.check(req) is False


def test_completion_check_max_tokens_above(completion):
    req = _make_req(max_tokens=10)
    req.decoded_count = 11
    assert completion.check(req) is True


def test_completion_check_max_tokens_zero(completion):
    """boundary — max_tokens=0, decoded_count=0 → 즉시 종료"""
    req = _make_req(max_tokens=0)
    assert completion.check(req) is True


def test_completion_check_eos_seen_true(completion):
    """Q6 EOS branch — eos_seen=True 위 max_tokens 미도달이어도 True"""
    req = _make_req(max_tokens=100)
    req.decoded_count = 1
    assert completion.check(req, eos_seen=True) is True


def test_completion_check_eos_seen_false_default(completion):
    """Q6 (b) default path — eos_seen 미명시 시 False default"""
    req = _make_req(max_tokens=100)
    req.decoded_count = 1
    assert completion.check(req) is False


def test_completion_check_already_completed_returns_true(completion, kv_accountant):
    """idempotent — COMPLETED 위 True (안전 polling)"""
    req = _make_req(max_tokens=5)
    req.decoded_count = 5
    kv_accountant.admit(req)
    completion.finalize(req)
    assert completion.check(req) is True


def test_completion_check_no_state_mutation(completion):
    """Pure function — check 호출 전후 field 변경 0"""
    req = _make_req(max_tokens=10)
    req.decoded_count = 5
    before = (req.state, req.decoded_count, req.completion_time)
    completion.check(req)
    after = (req.state, req.decoded_count, req.completion_time)
    assert before == after


@pytest.mark.parametrize("decoded_count", [0, 1, 9, 10, 11])
@pytest.mark.parametrize("max_tokens", [0, 1, 10])
@pytest.mark.parametrize("eos_seen", [True, False])
def test_completion_check_boundary_cross_product(completion, decoded_count, max_tokens, eos_seen):
    """5 × 3 × 2 = 30 cell 전수"""
    req = _make_req(max_tokens=max_tokens)
    req.decoded_count = decoded_count
    expected = eos_seen or (decoded_count >= max_tokens)
    assert completion.check(req, eos_seen=eos_seen) is expected


# ============================================================================
# finalize — KV release + completion_time + state transition
# ============================================================================

def test_completion_finalize_kv_release(completion, kv_accountant, clock):
    """Q7 + ARCH §3.3 — finalize 후 kv_accountant.remaining 정확 회수"""
    req = _make_req(kv_length=100, max_tokens=5)
    kv_accountant.admit(req)
    before = kv_accountant.remaining
    completion.finalize(req)
    assert kv_accountant.remaining == before + 100


def test_completion_finalize_completion_time(completion, kv_accountant, clock):
    req = _make_req(max_tokens=5)
    kv_accountant.admit(req)
    clock.advance_to(123.456)
    completion.finalize(req)
    assert req.completion_time == pytest.approx(123.456)


def test_completion_finalize_state_transition(completion, kv_accountant):
    req = _make_req(max_tokens=5)
    kv_accountant.admit(req)
    completion.finalize(req)
    assert req.state == RequestState.COMPLETED


def test_completion_finalize_kv_release_before_state_transition(completion, kv_accountant):
    """Atomic — KV release 실패 시 state 보존"""
    req = _make_req(max_tokens=5)
    # not admitted → release raises → state 유지
    with pytest.raises(ValueError):
        completion.finalize(req)
    assert req.state == RequestState.DECODE  # 변경 0
    assert req.completion_time is None


def test_completion_finalize_double_raises(completion, kv_accountant):
    req = _make_req(max_tokens=5)
    kv_accountant.admit(req)
    completion.finalize(req)
    with pytest.raises(ValueError, match="double finalize"):
        completion.finalize(req)


def test_completion_finalize_pending_raises(completion):
    """PENDING 상태 위 finalize 시도 reject"""
    req = Request(id=0, prompt_len=5, kv_length=10, max_tokens=5)
    # state == PENDING (default)
    with pytest.raises(ValueError, match="PENDING"):
        completion.finalize(req)


def test_completion_finalize_unadmitted_raises(completion):
    """KV release 의 미admit reject 가 propagate"""
    req = _make_req(max_tokens=5)
    with pytest.raises(ValueError):
        completion.finalize(req)


def test_completion_admit_release_roundtrip_x_50(completion, kv_accountant):
    """50 회 (admit → finalize) round-trip → remaining 정확 보존"""
    initial = kv_accountant.remaining
    for i in range(50):
        req = _make_req(req_id=i, kv_length=100)
        kv_accountant.admit(req)
        completion.finalize(req)
    assert kv_accountant.remaining == initial


def test_completion_no_dispatcher_unregister_call(completion, kv_accountant):
    """Q9 책임 분리 lock-in — Completion 이 dispatcher 호출 0"""
    # Completion 의 field 가 dispatcher 미보유 — 구조적 검증
    fields = {f for f in completion.__dataclass_fields__}
    assert fields == {"clock", "kv_accountant"}
    assert "dispatcher" not in fields
