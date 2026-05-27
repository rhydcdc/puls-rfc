import pytest

from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeState, NodeType


def _mark_done(dispatcher: Dispatcher, mb_id: int, *ntypes: NodeType) -> None:
    """Force given nodes of mb to DONE (PENDING->READY->RUNNING->DONE)."""
    for ntype in ntypes:
        node = dispatcher.dag.get_node(mb_id, ntype)
        for s in (NodeState.READY, NodeState.RUNNING, NodeState.DONE):
            if node.state is not s:
                node.transition_to(s)


def _register_pim_mb(dispatcher: Dispatcher, mb_id: int) -> MicroBatch:
    """Impl-5 — PIM dispatch 위 MicroBatch register helper. backward-compat 위해
    Impl-4 의 placeholder default args (k_total_max=2048, rtl_fsm_tile_rows=32) 와
    동일 lookup 값 사용 → expected timestamp 변경 0.
    """
    mb = MicroBatch(
        id=mb_id,
        k_total=dispatcher.config.admission.k_total_max,
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    )
    dispatcher.register(mb)
    return mb


# =========================================================================
# refresh_ready
# =========================================================================

def test_refresh_ready_promotes_qkv_first(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    assert qkv.state is NodeState.READY
    # 다른 노드는 PENDING 유지
    for ntype in (NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN, NodeType.O_PROJ):
        assert dispatcher.dag.get_node(0, ntype).state is NodeState.PENDING


def test_refresh_ready_promotes_prefill_decode_after_qkv_done(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    assert dispatcher.dag.get_node(0, NodeType.PREFILL_ATTN).state is NodeState.READY
    assert dispatcher.dag.get_node(0, NodeType.DECODE_ATTN).state is NodeState.READY
    assert dispatcher.dag.get_node(0, NodeType.O_PROJ).state is NodeState.PENDING


def test_refresh_ready_promotes_oproj_after_both_attn_done(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN)
    dispatcher.refresh_ready()
    assert dispatcher.dag.get_node(0, NodeType.O_PROJ).state is NodeState.READY


def test_refresh_ready_idempotent(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    snapshot = {nt: dispatcher.dag.get_node(0, nt).state for nt in NodeType}
    dispatcher.refresh_ready()
    after = {nt: dispatcher.dag.get_node(0, nt).state for nt in NodeType}
    assert snapshot == after


# =========================================================================
# Priority dequeue (GPU)
# =========================================================================

def test_pick_gpu_selects_oproj_when_all_priorities_ready(dispatcher: Dispatcher):
    # mb 0: all three GPU types READY
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN)
    dispatcher.refresh_ready()
    # QKV manually READY again (for the scenario) — but QKV is DONE here, so add mb 1
    dispatcher.dag.add_micro_batch(1)
    dispatcher.refresh_ready()
    # mb 0: O-proj READY; mb 1: QKV READY; need a prefill-attn READY too:
    _mark_done(dispatcher, 1, NodeType.QKV)
    dispatcher.refresh_ready()
    # Now: mb 0 O-proj READY, mb 1 prefill-attn + decode-attn READY (QKV done)
    picked = dispatcher.pick_gpu()
    assert picked is not None
    assert picked.type is NodeType.O_PROJ
    assert picked.micro_batch_id == 0


def test_pick_gpu_selects_prefill_when_no_oproj(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.dag.add_micro_batch(1)
    dispatcher.refresh_ready()
    # mb 0: prefill-attn READY (+ decode-attn READY, PIM); mb 1: QKV READY
    picked = dispatcher.pick_gpu()
    assert picked is not None
    assert picked.type is NodeType.PREFILL_ATTN
    assert picked.micro_batch_id == 0


def test_pick_gpu_selects_qkv_when_only_qkv(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    picked = dispatcher.pick_gpu()
    assert picked is not None
    assert picked.type is NodeType.QKV
    assert picked.micro_batch_id == 0


def test_pick_gpu_returns_none_when_empty(dispatcher: Dispatcher):
    assert dispatcher.pick_gpu() is None


def test_pick_gpu_tie_break_oldest_micro_batch(dispatcher: Dispatcher):
    # Two mb both with prefill-attn READY; oldest (mb 3) wins over mb 7
    for mb_id in (3, 7):
        dispatcher.dag.add_micro_batch(mb_id)
        _mark_done(dispatcher, mb_id, NodeType.QKV)
    dispatcher.refresh_ready()
    picked = dispatcher.pick_gpu()
    assert picked is not None
    assert picked.type is NodeType.PREFILL_ATTN
    assert picked.micro_batch_id == 3


# =========================================================================
# Priority dequeue (PIM)
# =========================================================================

def test_pick_pim_returns_oldest_decode_attn(dispatcher: Dispatcher):
    for mb_id in (2, 5):
        dispatcher.dag.add_micro_batch(mb_id)
        _mark_done(dispatcher, mb_id, NodeType.QKV)
    dispatcher.refresh_ready()
    picked = dispatcher.pick_pim()
    assert picked is not None
    assert picked.type is NodeType.DECODE_ATTN
    assert picked.micro_batch_id == 2


def test_pick_pim_ignores_gpu_node_types(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()  # QKV READY (GPU type)
    assert dispatcher.pick_pim() is None


def test_pick_pim_returns_none_when_empty(dispatcher: Dispatcher):
    assert dispatcher.pick_pim() is None


# =========================================================================
# Dispatch (GPU)
# =========================================================================

def test_dispatch_gpu_qkv_sets_busy_and_pushes_event(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    dispatcher.dispatch_gpu(qkv)
    assert dispatcher.gpu_busy is True
    assert qkv.state is NodeState.RUNNING
    expected_t = dispatcher.config.time.gpu_op_time_us["qkv"]
    assert dispatcher.queue.peek_timestamp() == expected_t
    assert len(dispatcher.queue) == 1


def test_dispatch_gpu_blocks_when_busy(dispatcher: Dispatcher):
    dispatcher.gpu_busy = True
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    with pytest.raises(ValueError, match="I4 violation"):
        dispatcher.dispatch_gpu(qkv)


def test_dispatch_gpu_blocks_prefill_when_qkv_pending(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    # Force prefill-attn READY without QKV DONE (bypass refresh_ready)
    prefill = dispatcher.dag.get_node(0, NodeType.PREFILL_ATTN)
    prefill.transition_to(NodeState.READY)
    with pytest.raises(ValueError, match="I1 violation"):
        dispatcher.dispatch_gpu(prefill)


def test_dispatch_gpu_blocks_oproj_when_attn_pending(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    oproj = dispatcher.dag.get_node(0, NodeType.O_PROJ)
    oproj.transition_to(NodeState.READY)
    with pytest.raises(ValueError, match="I3 violation"):
        dispatcher.dispatch_gpu(oproj)


# =========================================================================
# Dispatch (PIM)
# =========================================================================

def test_dispatch_pim_decode_attn_sets_busy_and_pushes_event(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _register_pim_mb(dispatcher, 0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    dispatcher.dispatch_pim(decode)
    assert dispatcher.pim_busy is True
    assert decode.state is NodeState.RUNNING
    # Impl-4: PIM op_time = pim_executor.op_time(k_max, tile_rows) (placeholder default args)
    expected_t = dispatcher.pim_executor.op_time(
        k_channels=dispatcher.config.admission.k_total_max,
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    )
    assert dispatcher.queue.peek_timestamp() == expected_t


def test_dispatch_pim_blocks_when_busy(dispatcher: Dispatcher):
    dispatcher.pim_busy = True
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    with pytest.raises(ValueError, match="I5 violation"):
        dispatcher.dispatch_pim(decode)


def test_dispatch_pim_blocks_when_qkv_pending(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    decode.transition_to(NodeState.READY)
    with pytest.raises(ValueError, match="I2 violation"):
        dispatcher.dispatch_pim(decode)


# =========================================================================
# on_completion
# =========================================================================

def test_on_completion_gpu_clears_busy_and_marks_done(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    dispatcher.dispatch_gpu(qkv)
    event = dispatcher.queue.pop()
    dispatcher.on_completion(event)
    assert qkv.state is NodeState.DONE
    assert dispatcher.gpu_busy is False


def test_on_completion_pim_clears_busy_and_marks_done(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    _register_pim_mb(dispatcher, 0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    dispatcher.dispatch_pim(decode)
    event = dispatcher.queue.pop()
    dispatcher.on_completion(event)
    assert decode.state is NodeState.DONE
    assert dispatcher.pim_busy is False


def test_on_completion_unknown_resource_raises(dispatcher: Dispatcher):
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    qkv.transition_to(NodeState.RUNNING)
    event = Event(
        timestamp=1.0,
        type=EventType.KERNEL_COMPLETION,
        payload={"micro_batch_id": 0, "node_type": NodeType.QKV, "resource": "NVLINK"},
    )
    with pytest.raises(ValueError, match="unknown resource"):
        dispatcher.on_completion(event)


# =========================================================================
# Cross-module invariant: dispatcher ↔ window/DAG round-trip
# =========================================================================

def test_invariant_dispatcher_dag_admit_evict_roundtrip(dispatcher: Dispatcher, window):
    # admit mb 0 -> refresh -> ready contains mb 0 QKV
    window.admit(0)
    dispatcher.refresh_ready()
    picked = dispatcher.pick_gpu()
    assert picked is not None and picked.micro_batch_id == 0

    # admit two more -> window full
    window.admit(1)
    window.admit(2)
    # 4th admit evicts mb 0 from DAG
    window.admit(3)
    dispatcher.refresh_ready()
    # mb 0 must no longer be selectable
    selected_ids = set()
    for mb_id in dispatcher.dag.nodes:
        selected_ids.add(mb_id)
    assert 0 not in selected_ids
    assert selected_ids == {1, 2, 3}


# =========================================================================
# Impl-4 — PIM executor wiring (cross-module)
# =========================================================================

def test_dispatch_pim_op_time_via_pim_executor(dispatcher: Dispatcher):
    """dispatcher._op_time(pim_node) bit-exact == pim_executor.op_time(...) with placeholder args."""
    dispatcher.dag.add_micro_batch(0)
    _register_pim_mb(dispatcher, 0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    expected = dispatcher.pim_executor.op_time(
        k_channels=dispatcher.config.admission.k_total_max,
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    )
    assert dispatcher._op_time(decode) == expected


def test_dispatch_pim_completion_timestamp_uses_pim_executor(dispatcher: Dispatcher):
    """ARCH §3.5.2 Computed Wait — completion timestamp == clock.now + pim_executor.op_time(...)."""
    dispatcher.dag.add_micro_batch(0)
    _register_pim_mb(dispatcher, 0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    t0 = dispatcher.clock.now
    dispatcher.dispatch_pim(decode)
    pushed_timestamp = dispatcher.queue.peek_timestamp()
    op_time = dispatcher.pim_executor.op_time(
        k_channels=dispatcher.config.admission.k_total_max,
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    )
    assert pushed_timestamp == t0 + op_time


def test_dispatcher_pim_executor_field_present(dispatcher: Dispatcher):
    """Dispatcher dataclass field 에 pim_executor 존재."""
    fields = Dispatcher.__dataclass_fields__
    assert "pim_executor" in fields


# =========================================================================
# Stress (Acceptance) — 100 μ-batch synthetic trace, I1~I5 violation 0
# =========================================================================

def test_stress_100_micro_batch_no_invariant_violation(scheduler_core):
    """100 μ-batch 자연 dispatch — 각 mb 가 in-flight 동안 dispatcher 자연 진행.
    Window capacity 3 이므로 한 시점에 최대 3 mb in-flight. mb i 의 모든 노드 DONE
    이후 다음 admit (orphan event 회피)."""
    for mb_id in range(100):
        _register_pim_mb(scheduler_core.dispatcher, mb_id)
        scheduler_core.window.admit(mb_id)
        scheduler_core.dispatcher.tick()
        while len(scheduler_core.queue) > 0:
            scheduler_core.step()
        # After drain: mb's 4 nodes all DONE
        for ntype in NodeType:
            assert scheduler_core.dag.get_node(mb_id, ntype).state is NodeState.DONE
        # Idle resources between μ-batches
        assert scheduler_core.dispatcher.gpu_busy is False
        assert scheduler_core.dispatcher.pim_busy is False

    # Final state: only last 3 μ-batches in window
    assert scheduler_core.window.current_ids() == (97, 98, 99)
    assert set(scheduler_core.dag.nodes.keys()) == {97, 98, 99}


# =========================================================================
# Impl-5 — register/unregister API + 실 signal flow (O4.1 해소)
# =========================================================================

def test_dispatcher_register_micro_batch(dispatcher):
    mb = MicroBatch(id=42, k_total=512, kv_rows_total=1000)
    dispatcher.register(mb)
    assert dispatcher.micro_batches[42] is mb


def test_dispatcher_register_duplicate_raises(dispatcher):
    mb = MicroBatch(id=42, k_total=512, kv_rows_total=1000)
    dispatcher.register(mb)
    with pytest.raises(RuntimeError, match="already registered"):
        dispatcher.register(mb)


def test_dispatcher_unregister_micro_batch(dispatcher):
    mb = MicroBatch(id=42, k_total=512, kv_rows_total=1000)
    dispatcher.register(mb)
    dispatcher.unregister(42)
    assert 42 not in dispatcher.micro_batches


def test_dispatcher_unregister_unknown_raises(dispatcher):
    with pytest.raises(RuntimeError, match="not registered"):
        dispatcher.unregister(999)


def test_dispatch_pim_uses_real_signal_flow(dispatcher):
    """O4.1 해소 — dispatch_pim event timestamp 가 mb.k_total · mb.kv_rows_total 위 op_time."""
    dispatcher.dag.add_micro_batch(0)
    mb = MicroBatch(id=0, k_total=2048, kv_rows_total=10000)
    dispatcher.register(mb)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    t0 = dispatcher.clock.now
    dispatcher.dispatch_pim(decode)
    expected = t0 + dispatcher.pim_executor.op_time(k_channels=2048, kv_rows_total=10000)
    assert dispatcher.queue.peek_timestamp() == expected


def test_dispatch_pim_unregistered_raises(dispatcher):
    """O4.1 negative — unregistered mb 의 PIM dispatch 시 raise."""
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    with pytest.raises(RuntimeError, match="unregistered MicroBatch"):
        dispatcher.dispatch_pim(decode)


@pytest.mark.parametrize("k_total", [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048])
def test_dispatch_pim_k_total_sweep(dispatcher, k_total):
    """9-step k_total dial 위 _op_time 산식 정합."""
    dispatcher.dag.add_micro_batch(0)
    mb = MicroBatch(id=0, k_total=k_total, kv_rows_total=10000)
    dispatcher.register(mb)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    expected = dispatcher.pim_executor.op_time(k_channels=k_total, kv_rows_total=10000)
    assert dispatcher._op_time(decode) == expected


@pytest.mark.parametrize("kv_rows", [0, 100, 10000, 1_000_000])
def test_dispatch_pim_kv_rows_total_sweep(dispatcher, kv_rows):
    dispatcher.dag.add_micro_batch(0)
    mb = MicroBatch(id=0, k_total=2048, kv_rows_total=kv_rows)
    dispatcher.register(mb)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    expected = dispatcher.pim_executor.op_time(k_channels=2048, kv_rows_total=kv_rows)
    assert dispatcher._op_time(decode) == expected
