import pytest

from puls_sched.dag import DAG
from puls_sched.invariants import check_I1, check_I2, check_I3, check_I4, check_I5
from puls_sched.node import NodeState, NodeType


def _advance_node_to(dag: DAG, mb_id: int, ntype: NodeType, target: NodeState) -> None:
    """Walk PENDING -> READY -> RUNNING -> DONE up to target."""
    order = [NodeState.PENDING, NodeState.READY, NodeState.RUNNING, NodeState.DONE]
    node = dag.get_node(mb_id, ntype)
    for next_state in order[order.index(node.state) + 1 : order.index(target) + 1]:
        node.transition_to(next_state)


NON_DONE_STATES = [NodeState.PENDING, NodeState.READY, NodeState.RUNNING]


# ---- I1 ----

@pytest.mark.parametrize("qkv_state", NON_DONE_STATES)
def test_I1_blocks_when_qkv_not_done(dag: DAG, qkv_state: NodeState):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, qkv_state)
    with pytest.raises(ValueError, match="I1 violation"):
        check_I1(dag, 0)


def test_I1_passes_when_qkv_done(dag: DAG):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    check_I1(dag, 0)


# ---- I2 ----

@pytest.mark.parametrize("qkv_state", NON_DONE_STATES)
def test_I2_blocks_when_qkv_not_done(dag: DAG, qkv_state: NodeState):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, qkv_state)
    with pytest.raises(ValueError, match="I2 violation"):
        check_I2(dag, 0)


def test_I2_passes_when_qkv_done(dag: DAG):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    check_I2(dag, 0)


# ---- I3 ----

@pytest.mark.parametrize("prefill_state", NON_DONE_STATES)
def test_I3_blocks_when_prefill_not_done(dag: DAG, prefill_state: NodeState):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    _advance_node_to(dag, 0, NodeType.PREFILL_ATTN, prefill_state)
    _advance_node_to(dag, 0, NodeType.DECODE_ATTN, NodeState.DONE)
    with pytest.raises(ValueError, match="I3 violation"):
        check_I3(dag, 0)


@pytest.mark.parametrize("decode_state", NON_DONE_STATES)
def test_I3_blocks_when_decode_not_done(dag: DAG, decode_state: NodeState):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    _advance_node_to(dag, 0, NodeType.PREFILL_ATTN, NodeState.DONE)
    _advance_node_to(dag, 0, NodeType.DECODE_ATTN, decode_state)
    with pytest.raises(ValueError, match="I3 violation"):
        check_I3(dag, 0)


def test_I3_blocks_when_both_not_done(dag: DAG):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    with pytest.raises(ValueError, match="I3 violation"):
        check_I3(dag, 0)


def test_I3_passes_when_both_done(dag: DAG):
    dag.add_micro_batch(0)
    _advance_node_to(dag, 0, NodeType.QKV, NodeState.DONE)
    _advance_node_to(dag, 0, NodeType.PREFILL_ATTN, NodeState.DONE)
    _advance_node_to(dag, 0, NodeType.DECODE_ATTN, NodeState.DONE)
    check_I3(dag, 0)


# ---- I4 / I5 ----

def test_I4_blocks_when_gpu_busy():
    with pytest.raises(ValueError, match="I4 violation"):
        check_I4(True)


def test_I4_passes_when_gpu_idle():
    check_I4(False)


def test_I5_blocks_when_pim_busy():
    with pytest.raises(ValueError, match="I5 violation"):
        check_I5(True)


def test_I5_passes_when_pim_idle():
    check_I5(False)
