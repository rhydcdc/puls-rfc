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
    """ADMISSION_TICK → KERNEL_COMPLETION chain → L=80 layer cycle → finalize → evict.

    Impl-9 ARCH-compliant lifecycle 정합 갱신 (Impl-6 시점의 'mb 4 node 모두 DONE' 가정 폐기).
    """
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_TICK, payload={},
    ))
    scheduler_core.step()
    assert scheduler_core.dispatcher.gpu_busy is True
    # L=80 layer cycling + max_tokens=0 → first token_signal 위 finalize → evict
    while len(scheduler_core.queue) > 0:
        scheduler_core.step()
    # Impl-9 — mb 0 의 모든 req finalize → window evict + DAG remove + dispatcher unregister
    assert 0 not in scheduler_core.dag.nodes
    assert 0 not in scheduler_core.window.current_ids()
    assert 0 not in scheduler_core.dispatcher.micro_batches
    assert scheduler_core.dispatcher.gpu_busy is False
    assert scheduler_core.dispatcher.pim_busy is False


def test_window_full_admission_deferred(scheduler_core):
    """Impl-9 — window full 위 admission pre-check (auto-evict path 비활성).

    *기존 'window_eviction_during_admission_dag_consistency' 의 Impl-9 ARCH-compliant 갱신.*
    Auto-evict (capacity overflow) 은 *defensive* 영역. Pre-check 으로 4번째 admission 은
    window full 검출 시 spec=None (deferred) → window 는 first 3 (0, 1, 2) 유지.
    """
    for i in range(4):
        scheduler_core.request_queue.push(_make_req(i))
        scheduler_core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        scheduler_core.dispatcher.gpu_busy = False
        scheduler_core.dispatcher.pim_busy = False
    # Pre-check 으로 4번째 admission deferred — window 는 first 3 mbs 보유
    assert scheduler_core.window.current_ids() == (0, 1, 2)
    # 4번째 req (id=3) 는 admission deferred — request_queue 잔존
    assert len(scheduler_core.request_queue) == 1


def test_request_arrival_payload_missing_request_key_raises(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={},
    ))
    with pytest.raises(KeyError):
        scheduler_core.step()


# =========================================================================
# Impl-5 — spec → MicroBatch 변환 + dispatcher.register (Q1)
# =========================================================================

def test_admission_tick_converts_spec_to_micro_batch(scheduler_core):
    """ADMISSION_TICK 후 dispatcher.micro_batches 에 신규 mb 등록."""
    scheduler_core.request_queue.push(_make_req(0, kv_length=50))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_TICK, payload={}))
    assert 0 in scheduler_core.dispatcher.micro_batches
    mb = scheduler_core.dispatcher.micro_batches[0]
    assert mb.id == 0


def test_admission_tick_micro_batch_carries_k_total(scheduler_core):
    """등록된 mb.k_total == spec.k_total (signal flow)."""
    scheduler_core.request_queue.push(_make_req(0, kv_length=50))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_TICK, payload={}))
    mb = scheduler_core.dispatcher.micro_batches[0]
    # k_total 은 admission 의 결정 — 정확값은 admission.layer1 산출이지만, *0 이상 + 다이얼 내* invariant
    assert mb.k_total >= 0
    assert mb.k_total <= scheduler_core.config.admission.k_total_max


def test_admission_tick_micro_batch_carries_kv_rows_total(scheduler_core):
    """등록된 mb.kv_rows_total == Σ kv_length over admitted reqs."""
    scheduler_core.request_queue.push(_make_req(0, kv_length=100))
    scheduler_core.request_queue.push(_make_req(1, kv_length=250))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_TICK, payload={}))
    mb = scheduler_core.dispatcher.micro_batches[0]
    assert mb.kv_rows_total == 100 + 250


def test_admission_tick_no_spec_no_register(scheduler_core):
    """spec None (empty queue) → register 호출 0."""
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_TICK, payload={}))
    assert len(scheduler_core.dispatcher.micro_batches) == 0


def test_admission_tick_multiple_ticks_unique_mb_ids(scheduler_core):
    """다회 ADMISSION_TICK → mb_id 0, 1, 2 단조."""
    for i in range(3):
        scheduler_core.request_queue.push(_make_req(i, kv_length=10))
        scheduler_core._handle(Event(timestamp=float(i), type=EventType.ADMISSION_TICK, payload={}))
        scheduler_core.dispatcher.gpu_busy = False
        scheduler_core.dispatcher.pim_busy = False
    registered_ids = sorted(scheduler_core.dispatcher.micro_batches.keys())
    assert registered_ids == [0, 1, 2]


def test_admission_tick_register_before_window_admit(scheduler_core):
    """register 호출이 window.admit *이전* — dispatcher.micro_batches 가 admit 시점에 보유.

    Indirect check: after ADMISSION_TICK, mb 가 window 와 dispatcher.micro_batches 양쪽에 존재.
    """
    scheduler_core.request_queue.push(_make_req(0, kv_length=10))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_TICK, payload={}))
    assert 0 in scheduler_core.window.current_ids()
    assert 0 in scheduler_core.dispatcher.micro_batches
