from puls_sched.dag import DAG
from puls_sched.node import NodeState, NodeType


def test_add_micro_batch_creates_5_nodes():
    # Phase-2 S0 — μ-batch 당 5 노드 (QKV·PREFILL_ATTN·DECODE_ATTN·O_PROJ·FFN[inter-AB])
    dag = DAG()
    dag.add_micro_batch(0)
    assert set(dag.nodes[0].keys()) == set(NodeType)
    for node_type in NodeType:
        assert dag.nodes[0][node_type].state == NodeState.PENDING
        assert dag.nodes[0][node_type].micro_batch_id == 0


def test_precedence_edges_match_i1_i2_i3():
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0] == {
        NodeType.QKV: set(),
        NodeType.PREFILL_ATTN: {NodeType.QKV},
        NodeType.DECODE_ATTN: {NodeType.QKV},
        NodeType.O_PROJ: {NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN},
        NodeType.FFN: {NodeType.O_PROJ},                 # Phase-2 S0 — inter-AB (F3)
    }


def test_qkv_has_no_precedence():
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0][NodeType.QKV] == set()


def test_o_proj_depends_on_both_attn():
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0][NodeType.O_PROJ] == {
        NodeType.PREFILL_ATTN,
        NodeType.DECODE_ATTN,
    }


def test_invariant_dag_add_remove_roundtrip():
    dag = DAG()
    initial_nodes = dict(dag.nodes)
    initial_precedence = dict(dag.precedence)
    dag.add_micro_batch(7)
    dag.remove_micro_batch(7)
    assert dag.nodes == initial_nodes
    assert dag.precedence == initial_precedence
