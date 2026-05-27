from pathlib import Path

import puls_sched
from puls_sched.dag import DAG
from puls_sched.node import NodeType
from puls_sched.request import RequestState


_EXPECTED_MODULES = {
    "__init__",
    "config",
    "clock",
    "request",
    "micro_batch",
    "node",
    "dag",
    "event",
    "event_queue",
    "window",
    "main_loop",
    "invariants",
    "dispatcher",
}


def test_meta_module_inventory():
    pkg_dir = Path(puls_sched.__file__).parent
    actual = {p.stem for p in pkg_dir.glob("*.py")}
    assert actual == _EXPECTED_MODULES, (
        f"module inventory mismatch — extra: {actual - _EXPECTED_MODULES}, "
        f"missing: {_EXPECTED_MODULES - actual}"
    )


def test_meta_node_types_complete():
    assert set(NodeType) == {
        NodeType.QKV,
        NodeType.PREFILL_ATTN,
        NodeType.DECODE_ATTN,
        NodeType.O_PROJ,
    }


def test_meta_request_states_complete():
    assert set(RequestState) == {
        RequestState.PENDING,
        RequestState.PREFILL,
        RequestState.DECODE,
        RequestState.COMPLETED,
    }


def test_meta_dag_precedence_matches_plan():
    """PLAN.md §6.3 의 I1·I2·I3 표와 정확 일치 (literal 비교)."""
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0] == {
        NodeType.QKV: set(),
        NodeType.PREFILL_ATTN: {NodeType.QKV},                              # I1
        NodeType.DECODE_ATTN: {NodeType.QKV},                               # I2
        NodeType.O_PROJ: {NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN},     # I3
    }
