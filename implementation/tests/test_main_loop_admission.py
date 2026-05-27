import pytest

from puls_sched.event import Event, EventType
from puls_sched.node import NodeState, NodeType
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length)


def test_request_arrival_pushes_to_queue(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.REQUEST_ARRIVAL,
        payload={"request": _make_req(0)},
    ))
    scheduler_core.step()
    assert len(scheduler_core.request_queue) == 1


def test_request_arrival_overflow_rejects_silently(scheduler_core, admission_config):
    # Fill capacity
    for i in range(admission_config.request_queue_capacity):
        scheduler_core.request_queue.push(_make_req(i))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.REQUEST_ARRIVAL,
        payload={"request": _make_req(9999)},
    ))
    scheduler_core.step()    # no raise
    assert len(scheduler_core.request_queue) == admission_config.request_queue_capacity


def test_admission_tick_empty_queue_no_dispatch(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_TICK, payload={},
    ))
    scheduler_core.step()
    assert scheduler_core.window.current_ids() == ()


def test_admission_tick_admits_and_dispatches(scheduler_core):
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_TICK, payload={},
    ))
    scheduler_core.step()
    # window has 1 mb, QKV dispatched (RUNNING) on GPU
    assert scheduler_core.window.current_ids() == (0,)
    qkv = scheduler_core.dag.get_node(0, NodeType.QKV)
    assert qkv.state is NodeState.RUNNING
    assert scheduler_core.dispatcher.gpu_busy is True


def test_admission_tick_assigns_monotonic_mb_id(scheduler_core):
    """3 회 _handle(ADMISSION_TICK) 직접 호출 → mb_id 0,1,2 단조 증가.
    Queue ordering (KERNEL_COMPLETION 와 ADMISSION_TICK race) 회피 위해
    white-box _handle 직접 호출."""
    for i in range(3):
        scheduler_core.request_queue.push(_make_req(i))
        scheduler_core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        scheduler_core.dispatcher.gpu_busy = False
        scheduler_core.dispatcher.pim_busy = False
    assert scheduler_core._next_mb_id == 3


def test_admission_tick_payload_dummy_values_safe(scheduler_core):
    """payload 의 t_proj / t_pim_fn 미제공 시 default 사용, 정상 동작."""
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_TICK, payload={},   # all defaults
    ))
    scheduler_core.step()    # no raise
    assert 0 in scheduler_core.window.current_ids()


# --- Integration chain (보강) ---

def test_admission_tick_then_completion_natural_progression(scheduler_core):
    """ADMISSION_TICK → KERNEL_COMPLETION → 다음 ADMISSION_TICK chain."""
    # admit mb 0
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_TICK, payload={},
    ))
    scheduler_core.step()
    assert scheduler_core.dispatcher.gpu_busy is True

    # QKV completes
    while len(scheduler_core.queue) > 0:
        scheduler_core.step()
    # All nodes of mb 0 should reach DONE
    for ntype in (NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN, NodeType.O_PROJ):
        assert scheduler_core.dag.get_node(0, ntype).state is NodeState.DONE
    assert scheduler_core.dispatcher.gpu_busy is False
    assert scheduler_core.dispatcher.pim_busy is False


def test_window_eviction_during_admission_dag_consistency(scheduler_core):
    """4번째 admission → eviction → DAG remove → evicted mb 노드 참조 0회.
    White-box _handle 직접 호출로 queue race 회피."""
    for i in range(4):
        scheduler_core.request_queue.push(_make_req(i))
        scheduler_core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        scheduler_core.dispatcher.gpu_busy = False
        scheduler_core.dispatcher.pim_busy = False
    # Window holds last 3 (1, 2, 3); mb 0 evicted from DAG
    assert scheduler_core.window.current_ids() == (1, 2, 3)
    assert 0 not in scheduler_core.dag.nodes


def test_request_arrival_payload_missing_request_key_raises(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={},
    ))
    with pytest.raises(KeyError):
        scheduler_core.step()
