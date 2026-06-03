import pytest

from puls_sched.event import Event, EventType
from puls_sched.node import NodeState, NodeType
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_len=1, kv_length=kv_length)


def _make_dec(req_id: int, kv_length: int = 10) -> Request:
    """prompt_len=0 → admission(풀 보충) 후 즉시 DECODE 풀 (디코더). decode-set 구성 대상.
    풀 모델: 새 요청은 PREFILL 부터라 decode_tokens 에 안 들어감 → decode KV 검증은 디코더로."""
    return Request(id=req_id, prompt_len=0, kv_length=kv_length)


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


def test_admission_pass_empty_queue_no_dispatch(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_PASS, payload={},
    ))
    scheduler_core.step()
    assert scheduler_core.window.current_ids() == ()


def test_admission_pass_admits_and_dispatches(scheduler_core):
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_PASS, payload={},
    ))
    scheduler_core.step()
    # window has 1 mb, QKV dispatched (RUNNING) on GPU
    assert scheduler_core.window.current_ids() == (0,)
    qkv = scheduler_core.dag.get_node(0, NodeType.QKV)
    assert qkv.state is NodeState.RUNNING
    assert scheduler_core.dispatcher.gpu_busy is True


def test_admission_pass_assigns_monotonic_mb_id(scheduler_core):
    """풀 모델 — 한 ADMISSION_PASS 이 window 를 활성 목표(2)까지 채움(디코더 풍부 시).
    mb_id 단조 증가(0,1)."""
    for i in range(4):
        scheduler_core.request_queue.push(_make_dec(i, kv_length=7_000_000))   # 2개서 12.3M 초과
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    assert sorted(scheduler_core.dispatcher.micro_batches.keys()) == [0, 1]     # 2 active(목표)
    assert scheduler_core._next_mb_id == 2


def test_admission_pass_payload_dummy_values_safe(scheduler_core):
    """payload 의 t_proj / t_pim_fn 미제공 시 default 사용, 정상 동작."""
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_PASS, payload={},   # all defaults
    ))
    scheduler_core.step()    # no raise
    assert 0 in scheduler_core.window.current_ids()


# --- Integration chain (보강) ---

def test_admission_pass_then_completion_natural_progression(scheduler_core):
    """ADMISSION_PASS → KERNEL_COMPLETION chain → L=80 layer cycle → finalize → evict.

    Impl-9 ARCH-compliant lifecycle 정합 갱신 (Impl-6 시점의 'mb 4 node 모두 DONE' 가정 폐기).
    """
    scheduler_core.request_queue.push(_make_req(0))
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_PASS, payload={},
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


def test_window_fills_to_active_target_not_capacity(scheduler_core):
    """풀 모델 — _fill_window 는 활성 목표(2 = _STAGGERING_TARGET_MB)까지만 채움(capacity 3
    의 '2 active + 1 전이 여유', OPERATING_POINT §3). 디코더 풍부해도 window ≤ 2.

    *기존 test_window_full_admission_deferred(capacity-3 deferral)의 풀 모델 갱신.*
    """
    for i in range(6):
        scheduler_core.request_queue.push(_make_dec(i, kv_length=5_000_000))   # 풍부
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    assert len(scheduler_core.window.current_ids()) == 2     # 목표 2 (< capacity 3)


def test_request_arrival_payload_missing_request_key_raises(scheduler_core):
    scheduler_core.queue.push(Event(
        timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={},
    ))
    with pytest.raises(KeyError):
        scheduler_core.step()


# =========================================================================
# Impl-5 — spec → MicroBatch 변환 + dispatcher.register (Q1)
# =========================================================================

def test_admission_pass_converts_spec_to_micro_batch(scheduler_core):
    """ADMISSION_PASS 후 dispatcher.micro_batches 에 신규 mb 등록."""
    scheduler_core.request_queue.push(_make_req(0, kv_length=50))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    assert 0 in scheduler_core.dispatcher.micro_batches
    mb = scheduler_core.dispatcher.micro_batches[0]
    assert mb.id == 0


def test_admission_pass_micro_batch_carries_kv_rows_total(scheduler_core):
    """등록된 mb.kv_rows_total == Σ kv_length over decode-set (디코더). 풀 모델: decode KV 는
    디코더(prompt_len=0)서 나옴 — 새 PREFILL 요청은 decode_tokens 에 안 들어감."""
    scheduler_core.request_queue.push(_make_dec(0, kv_length=100))
    scheduler_core.request_queue.push(_make_dec(1, kv_length=250))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    mb = scheduler_core.dispatcher.micro_batches[0]
    assert mb.kv_rows_total == 100 + 250


def test_admission_pass_no_spec_no_register(scheduler_core):
    """spec None (empty queue) → register 호출 0."""
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    assert len(scheduler_core.dispatcher.micro_batches) == 0


def test_admission_pass_unique_mb_ids(scheduler_core):
    """풀 모델 — 구성된 μ-batch 들의 id 가 고유·단조(0,1). window 활성 목표 2."""
    for i in range(4):
        scheduler_core.request_queue.push(_make_dec(i, kv_length=7_000_000))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    registered_ids = sorted(scheduler_core.dispatcher.micro_batches.keys())
    assert registered_ids == [0, 1]


def test_admission_pass_register_before_window_admit(scheduler_core):
    """register 호출이 window.admit *이전* — dispatcher.micro_batches 가 admit 시점에 보유.

    Indirect check: after ADMISSION_PASS, mb 가 window 와 dispatcher.micro_batches 양쪽에 존재.
    """
    scheduler_core.request_queue.push(_make_req(0, kv_length=10))
    scheduler_core._handle(Event(timestamp=0.0, type=EventType.ADMISSION_PASS, payload={}))
    assert 0 in scheduler_core.window.current_ids()
    assert 0 in scheduler_core.dispatcher.micro_batches
