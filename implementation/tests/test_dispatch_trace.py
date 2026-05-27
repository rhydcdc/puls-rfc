"""ARCHITECTURE §6.5 dispatch trace meta-test.

3 μ-batch window {P=0, M=1, N=2} + deterministic op time (PIM > GPU) 위에서
Init / T1 / T2 / T3 / T4 / T5 dispatch 시퀀스가 §6.5 표와 정합 재현되는지 검증.

PIM tile time 을 GPU op time 보다 큰 값으로 fixture-override 하는 이유:
§6.5 T3 의 "prefill(M) done → O-proj(M) still not ready (decode-attn(M) cont.)"
narrative 가 `pim_time > gpu_time` 일 때만 자연 emergence. PLAN.md §0.5 의 ratio
property 보존 — 절대값 무의미, ordering 정합만 사용.
"""

import dataclasses

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.main_loop import SchedulerCore
from puls_sched.node import NodeState, NodeType
from puls_sched.window import InFlightWindow


def _make_trace_fixture():
    base = default_dummy_config()
    time_config = dataclasses.replace(
        base.time,
        gpu_op_time_us={"qkv": 1.0, "prefill_attn": 1.0, "o_proj": 1.0},
        pim_tile_time_ns={"FP8": 3.0, "FP16": 6.0},  # PIM > GPU for §6.5 ordering
    )
    config = dataclasses.replace(base, time=time_config)
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag)
    dispatcher = Dispatcher(config=config, clock=clock, queue=queue, dag=dag)
    core = SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag,
        window=window, dispatcher=dispatcher,
    )

    for mb_id in (0, 1, 2):
        window.admit(mb_id)

    # P (mb 0): QKV + PREFILL_ATTN already DONE; DECODE_ATTN RUNNING on PIM
    for ntype in (NodeType.QKV, NodeType.PREFILL_ATTN):
        node = dag.get_node(0, ntype)
        node.transition_to(NodeState.READY)
        node.transition_to(NodeState.RUNNING)
        node.transition_to(NodeState.DONE)
    p_decode = dag.get_node(0, NodeType.DECODE_ATTN)
    p_decode.transition_to(NodeState.READY)
    p_decode.transition_to(NodeState.RUNNING)
    dispatcher.pim_busy = True
    queue.push(Event(
        timestamp=clock.now + config.time.pim_tile_time_ns["FP8"],
        type=EventType.KERNEL_COMPLETION,
        payload={"micro_batch_id": 0, "node_type": NodeType.DECODE_ATTN, "resource": "PIM"},
    ))

    # M (mb 1): QKV already DONE (back-fill finished pre-Init)
    m_qkv = dag.get_node(1, NodeType.QKV)
    m_qkv.transition_to(NodeState.READY)
    m_qkv.transition_to(NodeState.RUNNING)
    m_qkv.transition_to(NodeState.DONE)

    # N (mb 2): all PENDING
    return core


def _running(core) -> set[tuple[int, NodeType]]:
    return {
        (mb_id, ntype)
        for mb_id, nodes in core.dag.nodes.items()
        for ntype, node in nodes.items()
        if node.state is NodeState.RUNNING
    }


def test_dispatch_trace_init_state():
    core = _make_trace_fixture()
    assert _running(core) == {(0, NodeType.DECODE_ATTN)}
    assert core.dispatcher.gpu_busy is False
    assert core.dispatcher.pim_busy is True


def test_dispatch_trace_t1_through_t5_sequence():
    core = _make_trace_fixture()

    # T1: PIM(P) completion -> O-proj(P) [GPU] + decode-attn(M) [PIM] dispatched
    assert core.step() is True
    assert _running(core) == {(0, NodeType.O_PROJ), (1, NodeType.DECODE_ATTN)}
    assert core.dispatcher.gpu_busy is True
    assert core.dispatcher.pim_busy is True

    # T2: O-proj(P) completion -> prefill-attn(M) [GPU] dispatched; M decode still on PIM
    assert core.step() is True
    assert _running(core) == {(1, NodeType.PREFILL_ATTN), (1, NodeType.DECODE_ATTN)}

    # T3: prefill-attn(M) completion -> QKV(N) [GPU back-fill]; M decode still on PIM
    #   (O-proj(M) not ready because decode-attn(M) still RUNNING)
    assert core.step() is True
    assert _running(core) == {(2, NodeType.QKV), (1, NodeType.DECODE_ATTN)}

    # Intermediate: decode-attn(M) completion (PIM) -> no new dispatch (GPU busy on N QKV)
    assert core.step() is True
    assert _running(core) == {(2, NodeType.QKV)}
    assert core.dispatcher.pim_busy is False

    # T4: QKV(N) completion -> O-proj(M) [GPU] + decode-attn(N) [PIM]
    assert core.step() is True
    assert _running(core) == {(1, NodeType.O_PROJ), (2, NodeType.DECODE_ATTN)}

    # T5: O-proj(M) completion -> prefill-attn(N) [GPU]; N decode still on PIM
    assert core.step() is True
    assert _running(core) == {(2, NodeType.PREFILL_ATTN), (2, NodeType.DECODE_ATTN)}


def test_dispatch_trace_terminates_with_all_done():
    core = _make_trace_fixture()
    core.run_until_empty()
    for mb_id in core.dag.nodes:
        for ntype in NodeType:
            assert core.dag.get_node(mb_id, ntype).state is NodeState.DONE
    assert core.dispatcher.gpu_busy is False
    assert core.dispatcher.pim_busy is False
