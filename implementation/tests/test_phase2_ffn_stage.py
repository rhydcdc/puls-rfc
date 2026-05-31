"""Phase-2 S0 — Instance B (FFN) 스케줄 노드 + inter-AB (F3) 검증.

ARCH §3.4 (disaggregation) / §5.7 F3 — B-FFN 을 telemetry 가 아니라 *스케줄*에 반영.
한 layer = QKV → {prefill_attn, decode_attn} → O_PROJ → FFN. FFN 완료 = layer 경계.
"""
import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import EventType
from puls_sched.event_queue import EventQueue
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.invariants import check_I6
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType, NodeState
from puls_sched.pim_emulator import PIMExecutor


def _mk() -> Dispatcher:
    cfg = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    pim = PIMExecutor(config=cfg)
    tel = IdleTelemetry()
    tel.reset(t_start=0.0)
    return Dispatcher(config=cfg, clock=clock, queue=queue, dag=dag,
                      pim_executor=pim, idle_telemetry=tel)


def _add(disp: Dispatcher, mb: MicroBatch) -> None:
    disp.micro_batches[mb.id] = mb
    disp.dag.add_micro_batch(mb.id)


def _drive_to_done(disp: Dispatcher, mb_id: int, *ntypes: NodeType) -> None:
    """Force nodes to DONE, tolerating any current state (PENDING or already READY)."""
    for t in ntypes:
        n = disp.dag.nodes[mb_id][t]
        for s in (NodeState.READY, NodeState.RUNNING, NodeState.DONE):
            if n.state is not s:
                n.transition_to(s)


# ---- DAG 구조 ----

def test_dag_has_5_nodes_with_ffn():
    dag = DAG()
    dag.add_micro_batch(0)
    assert len(dag.nodes[0]) == 5
    assert NodeType.FFN in dag.nodes[0]


def test_ffn_precedence_is_o_proj():
    dag = DAG()
    dag.add_micro_batch(0)
    assert dag.precedence[0][NodeType.FFN] == {NodeType.O_PROJ}


# ---- FFN ready 조건 = O_PROJ done ----

def test_ffn_not_ready_until_o_proj_done():
    disp = _mk()
    mb = MicroBatch(id=0, decode_tokens={1: 0}, kv_rows_total=500)
    _add(disp, mb)
    disp.refresh_ready()
    assert disp.pick_instance_b() is None  # O_PROJ 아직
    _drive_to_done(disp, 0, NodeType.QKV, NodeType.PREFILL_ATTN,
                   NodeType.DECODE_ATTN, NodeType.O_PROJ)
    disp.refresh_ready()
    node = disp.pick_instance_b()
    assert node is not None and node.type is NodeType.FFN


# ---- dispatch_instance_b ----

def test_dispatch_instance_b_busy_and_event():
    disp = _mk()
    mb = MicroBatch(id=0, decode_tokens={1: 0})
    _add(disp, mb)
    node = disp.dag.nodes[0][NodeType.FFN]
    node.transition_to(NodeState.READY)
    disp.dispatch_instance_b(node)
    assert disp.instance_b_busy is True
    evt = disp.queue.pop()
    assert evt.type is EventType.KERNEL_COMPLETION
    assert evt.payload["resource"] == "INSTANCE_B"
    assert evt.payload["node_type"] is NodeType.FFN


def test_ffn_op_time_spec_derived_positive():
    disp = _mk()
    mb = MicroBatch(id=0, decode_tokens={1: 0, 2: 0})
    _add(disp, mb)
    t = disp._op_time(disp.dag.nodes[0][NodeType.FFN])
    assert t > 0


def test_I6_raises_when_instance_b_busy():
    with pytest.raises(ValueError, match="I6 violation"):
        check_I6(True)
    check_I6(False)  # no raise


def test_on_completion_clears_instance_b_busy():
    disp = _mk()
    mb = MicroBatch(id=0, decode_tokens={1: 0})
    _add(disp, mb)
    node = disp.dag.nodes[0][NodeType.FFN]
    node.transition_to(NodeState.READY)
    disp.dispatch_instance_b(node)
    evt = disp.queue.pop()
    disp.on_completion(evt)
    assert disp.instance_b_busy is False


# ---- end-to-end 단일 layer: FFN 이 마지막 ----

def test_single_layer_reaches_ffn():
    disp = _mk()
    mb = MicroBatch(id=0, decode_tokens={1: 0}, kv_rows_total=500)
    _add(disp, mb)
    resources_seen = []
    for _ in range(50):
        disp.tick()
        if len(disp.queue) == 0:
            break
        evt = disp.queue.pop()
        resources_seen.append((evt.payload["resource"], evt.payload["node_type"]))
        disp.on_completion(evt)
    assert any(r == "INSTANCE_B" for r, _ in resources_seen)
    assert resources_seen[-1][1] is NodeType.FFN
    assert disp.dag.nodes[0][NodeType.FFN].state is NodeState.DONE


# ---- F3 overlap: FFN(M) 이 B 점유 중 GPU 가 QKV(N) backfill ----

def test_f3_overlap_gpu_runs_next_mb_while_ffn_busy():
    disp = _mk()
    mb0 = MicroBatch(id=0, decode_tokens={1: 0})
    mb1 = MicroBatch(id=1, decode_tokens={2: 0})
    _add(disp, mb0)
    _add(disp, mb1)
    _drive_to_done(disp, 0, NodeType.QKV, NodeType.PREFILL_ATTN,
                   NodeType.DECODE_ATTN, NodeType.O_PROJ)
    disp.tick()
    assert disp.instance_b_busy is True
    assert disp.gpu_busy is True
    assert disp.dag.nodes[0][NodeType.FFN].state is NodeState.RUNNING
    assert disp.dag.nodes[1][NodeType.QKV].state is NodeState.RUNNING
