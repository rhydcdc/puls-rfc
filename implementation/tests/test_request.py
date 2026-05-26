import pytest

from puls_sched.request import Request, RequestState, _VALID_TRANSITIONS


def test_request_field_round_trip():
    req = Request(id=7, prompt_tokens=[1, 2, 3], kv_length=3, arrival_time=1.5)
    assert req.id == 7
    assert req.prompt_tokens == [1, 2, 3]
    assert req.decoded_tokens == []
    assert req.kv_length == 3
    assert req.arrival_time == 1.5
    assert req.state == RequestState.PENDING


def test_state_forward_transitions():
    req = Request(id=1, prompt_tokens=[1])
    req.transition_to(RequestState.PREFILL)
    assert req.state == RequestState.PREFILL
    req.transition_to(RequestState.DECODE)
    assert req.state == RequestState.DECODE
    req.transition_to(RequestState.COMPLETED)
    assert req.state == RequestState.COMPLETED


def test_state_backward_transition_rejected():
    req = Request(id=1, prompt_tokens=[1], state=RequestState.COMPLETED)
    with pytest.raises(ValueError, match="invalid transition"):
        req.transition_to(RequestState.DECODE)


def test_state_skip_transition_rejected():
    req = Request(id=1, prompt_tokens=[1])
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
    req = Request(id=1, prompt_tokens=[1], state=from_state)
    with pytest.raises(ValueError, match="invalid transition"):
        req.transition_to(to_state)
