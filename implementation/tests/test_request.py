import pytest

from puls_sched.request import Request, RequestState, _VALID_TRANSITIONS


def test_request_field_round_trip():
    req = Request(id=7, prompt_len=3, kv_length=3, arrival_time=1.5)
    assert req.id == 7
    assert req.prompt_len == 3
    assert req.decoded_tokens == []
    assert req.kv_length == 3
    assert req.arrival_time == 1.5
    assert req.state == RequestState.PENDING


def test_state_forward_transitions():
    req = Request(id=1, prompt_len=1)
    req.transition_to(RequestState.PREFILL)
    assert req.state == RequestState.PREFILL
    req.transition_to(RequestState.DECODE)
    assert req.state == RequestState.DECODE
    req.transition_to(RequestState.COMPLETED)
    assert req.state == RequestState.COMPLETED


def test_state_backward_transition_rejected():
    req = Request(id=1, prompt_len=1, state=RequestState.COMPLETED)
    with pytest.raises(ValueError, match="invalid transition"):
        req.transition_to(RequestState.DECODE)


def test_state_skip_transition_rejected():
    req = Request(id=1, prompt_len=1)
    with pytest.raises(ValueError, match="invalid transition"):
        req.transition_to(RequestState.DECODE)


_INVALID_PAIRS = [
    (a, b)
    for a in RequestState
    for b in RequestState
    if b not in _VALID_TRANSITIONS[a]
]


@pytest.mark.parametrize("from_state,to_state", _INVALID_PAIRS)
def test_request_state_invalid_transitions_rejected(from_state, to_state):
    req = Request(id=1, prompt_len=1, state=from_state)
    with pytest.raises(ValueError, match="invalid transition"):
        req.transition_to(to_state)


# ============================================================================
# Impl-6 lifecycle fields (Q6 · Q10 — Request = lifecycle owner)
# ============================================================================

def test_request_default_max_tokens_zero():
    req = Request(id=0, prompt_len=1)
    assert req.max_tokens == 0


def test_request_default_decoded_count_zero():
    req = Request(id=0, prompt_len=1)
    assert req.decoded_count == 0


def test_request_default_completion_time_none():
    req = Request(id=0, prompt_len=1)
    assert req.completion_time is None


def test_request_explicit_lifecycle_field_roundtrip():
    req = Request(
        id=42, prompt_len=2, kv_length=2, arrival_time=1.0,
        max_tokens=100, decoded_count=50, completion_time=1.5,
    )
    assert req.max_tokens == 100
    assert req.decoded_count == 50
    assert req.completion_time == 1.5


def test_request_decoded_count_distinct_from_decoded_tokens():
    """Q10 (b) lock-in — Request 가 decoded_count (count) + decoded_tokens (list[id]) 둘 다 보유.
    Pre-HW mode 의 *count 만 owner* / *실 token id 는 비어 있음* 의 의도된 분리."""
    fields = Request.__dataclass_fields__
    assert "decoded_count" in fields
    assert "decoded_tokens" in fields
    # 두 field 는 서로 다른 의미 — 자동으로 동기화되지 않음
    req = Request(id=0, prompt_len=0)
    req.decoded_count = 5
    assert req.decoded_tokens == []  # 실 token id list 는 빈 채로 유지 가능


def test_request_completion_time_nullable():
    """None 으로 시작 → finalize 후 float 으로 set"""
    req = Request(id=0, prompt_len=1)
    assert req.completion_time is None
    req.completion_time = 3.14
    assert req.completion_time == 3.14


def test_request_existing_transitions_unchanged():
    """Regression — 기존 transition 정합 보존 (PENDING → PREFILL → DECODE → COMPLETED)"""
    req = Request(id=1, prompt_len=1)
    req.transition_to(RequestState.PREFILL)
    req.transition_to(RequestState.DECODE)
    req.transition_to(RequestState.COMPLETED)
    assert req.state == RequestState.COMPLETED
