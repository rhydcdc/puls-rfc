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
    """Impl-5 — PIM dispatch 위 MicroBatch register helper.
    Impl-10-pre-2 — k_total knob 폐기 위 mb 에 kv_rows_total 만 (k_aggregate PIMExecutor 위 derived).
    """
    mb = MicroBatch(
        id=mb_id,
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
    """Stage 2 — 구조 invariant 보존, op_time 값 영역은 spec-derived (per-mb)."""
    dispatcher.dag.add_micro_batch(0)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(0, NodeType.QKV)
    dispatcher.dispatch_gpu(qkv)
    assert dispatcher.gpu_busy is True
    assert qkv.state is NodeState.RUNNING
    # Stage 2 — op_time = compute_gpu_op_time_s 산출 (구조: queue push + timestamp > 0)
    assert dispatcher.queue.peek_timestamp() > 0
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
    """Stage 2 — PIM op_time unit ns → µs (× 1e-3 정합)."""
    dispatcher.dag.add_micro_batch(0)
    _register_pim_mb(dispatcher, 0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    dispatcher.dispatch_pim(decode)
    assert dispatcher.pim_busy is True
    assert decode.state is NodeState.RUNNING
    # Stage 2 — pim_executor.op_time (ns) × 1e-3 = µs (clock unit 정합)
    mb = dispatcher.micro_batches[0]
    expected_us = dispatcher.pim_executor.op_time(
        kv_rows_total=mb.kv_rows_total,
        kv_rows_lockstep=mb.kv_rows_lockstep,
    ) * 1e-3
    assert dispatcher.queue.peek_timestamp() == pytest.approx(expected_us)


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
    # Stage 2 — PIM op_time (ns) × 1e-3 = µs (clock unit 정합)
    expected_us = dispatcher.pim_executor.op_time(
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    ) * 1e-3
    assert dispatcher._op_time(decode) == pytest.approx(expected_us)


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
    # Stage 2 — PIM op_time (ns) × 1e-3 = µs (clock unit 정합)
    op_time_us = dispatcher.pim_executor.op_time(
        kv_rows_total=dispatcher.config.time.rtl_fsm_tile_rows,
    ) * 1e-3
    assert pushed_timestamp == pytest.approx(t0 + op_time_us)


def test_dispatcher_pim_executor_field_present(dispatcher: Dispatcher):
    """Dispatcher dataclass field 에 pim_executor 존재."""
    fields = Dispatcher.__dataclass_fields__
    assert "pim_executor" in fields


# =========================================================================
# Stress (Acceptance) — 100 μ-batch synthetic trace, I1~I5 violation 0
# =========================================================================

def test_stress_100_micro_batch_no_invariant_violation(scheduler_core):
    """100 μ-batch 자연 dispatch (Impl-9 ARCH-compliant 갱신).

    각 mb 가 L=80 layer cycle 후 finalize → window/DAG/dispatcher 에서 evict.
    Window capacity 3 이지만 mb 가 cycle 완료 후 즉시 evict 되므로 다음 admit 자유.
    Impl-9 lifecycle 영역의 stress — invariant 위반 0 + orphan 0.
    """
    for mb_id in range(100):
        _register_pim_mb(scheduler_core.dispatcher, mb_id)
        scheduler_core.window.admit(mb_id)
        scheduler_core.dispatcher.tick()
        while len(scheduler_core.queue) > 0:
            scheduler_core.step()
        # Impl-9 — mb 의 모든 req (이 setup 에선 decode_tokens 비어있음) finalize 후 evict
        assert mb_id not in scheduler_core.dag.nodes
        assert mb_id not in scheduler_core.window.current_ids()
        assert mb_id not in scheduler_core.dispatcher.micro_batches
        # Idle resources between μ-batches
        assert scheduler_core.dispatcher.gpu_busy is False
        assert scheduler_core.dispatcher.pim_busy is False

    # Final state: all 100 mbs evicted (lifecycle 완료)
    assert scheduler_core.window.current_ids() == ()
    assert scheduler_core.dag.nodes == {}


# =========================================================================
# Impl-5 — register/unregister API + 실 signal flow (O4.1 해소)
# =========================================================================

def test_dispatcher_register_micro_batch(dispatcher):
    mb = MicroBatch(id=42, kv_rows_total=1000)
    dispatcher.register(mb)
    assert dispatcher.micro_batches[42] is mb


def test_dispatcher_register_duplicate_raises(dispatcher):
    mb = MicroBatch(id=42, kv_rows_total=1000)
    dispatcher.register(mb)
    with pytest.raises(RuntimeError, match="already registered"):
        dispatcher.register(mb)


def test_dispatcher_unregister_micro_batch(dispatcher):
    mb = MicroBatch(id=42, kv_rows_total=1000)
    dispatcher.register(mb)
    dispatcher.unregister(42)
    assert 42 not in dispatcher.micro_batches


def test_dispatcher_unregister_unknown_raises(dispatcher):
    with pytest.raises(RuntimeError, match="not registered"):
        dispatcher.unregister(999)


def test_dispatch_pim_uses_real_signal_flow(dispatcher):
    """dispatch_pim event timestamp 가 mb.kv_rows_total 위 op_time (k_aggregate PIMExecutor 위 derived)."""
    dispatcher.dag.add_micro_batch(0)
    mb = MicroBatch(id=0, kv_rows_total=10000)
    dispatcher.register(mb)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    t0 = dispatcher.clock.now
    dispatcher.dispatch_pim(decode)
    # Stage 2 — PIM op_time (ns) × 1e-3 = µs
    expected_us = t0 + dispatcher.pim_executor.op_time(kv_rows_total=10000) * 1e-3
    assert dispatcher.queue.peek_timestamp() == pytest.approx(expected_us)


def test_dispatch_pim_unregistered_raises(dispatcher):
    """negative — unregistered mb 의 PIM dispatch 시 raise."""
    dispatcher.dag.add_micro_batch(0)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    with pytest.raises(RuntimeError, match="unregistered MicroBatch"):
        dispatcher.dispatch_pim(decode)


@pytest.mark.parametrize("kv_rows", [0, 100, 10000, 1_000_000])
def test_dispatch_pim_kv_rows_total_sweep(dispatcher, kv_rows):
    dispatcher.dag.add_micro_batch(0)
    mb = MicroBatch(id=0, kv_rows_total=kv_rows)
    dispatcher.register(mb)
    _mark_done(dispatcher, 0, NodeType.QKV)
    dispatcher.refresh_ready()
    decode = dispatcher.dag.get_node(0, NodeType.DECODE_ATTN)
    # Stage 2 — PIM op_time (ns) × 1e-3 = µs
    expected_us = dispatcher.pim_executor.op_time(kv_rows_total=kv_rows) * 1e-3
    assert dispatcher._op_time(decode) == pytest.approx(expected_us)
