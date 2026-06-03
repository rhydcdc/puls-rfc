"""ARCHITECTURE §6.5 dispatch trace meta-test.

3 μ-batch window {P=0, M=1, N=2} + deterministic op time (PIM > GPU) 위에서
Init / T1 / T2 / T3 / T4 / T5 dispatch 시퀀스가 §6.5 표와 정합 재현되는지 검증.

PIM tile time 을 GPU op time 보다 큰 값으로 fixture-override 하는 이유:
§6.5 T3 의 "prefill(M) done → O-proj(M) still not ready (decode-attn(M) cont.)"
narrative 가 `pim_time > gpu_time` 일 때만 자연 emergence. PLAN.md §0.5 의 ratio
property 보존 — 절대값 무의미, ordering 정합만 사용.
"""

import dataclasses

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _make_trace_fixture():
    base = default_dummy_config()
    time_config = dataclasses.replace(
        base.time,
        gpu_op_time_us={"qkv": 1.0, "prefill_attn": 1.0, "o_proj": 1.0},
        pim_tile_time_ns={"FP8": 3.0, "FP16": 6.0},  # PIM > GPU for §6.5 ordering
        pim_broadcast_latency_ns_cross_gpu=0.0,  # ordering preservation 위 broadcast 분리
    )
    # Impl-9 — §6.5 single-layer dispatch pattern 검증 위 num_layers=1 (layer cycling 비활성).
    # Layer cycling 의 의미적 검증은 별도 cluster (Impl-9 driver 영역) 의 책임.
    model_config = dataclasses.replace(base.model, num_layers=1)
    config = dataclasses.replace(base, time=time_config, model=model_config)
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag)
    pim_executor = PIMExecutor(config=config)
    dispatcher = Dispatcher(
        config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim_executor,
    )
    request_queue = RequestQueue(capacity=config.admission.request_queue_capacity)
    kv_accountant = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    admission = Admission(
        admission_cfg=config.admission, 
        kv_accountant=kv_accountant, idle_telemetry=IdleTelemetry(),
    )
    core = SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag,
        window=window, dispatcher=dispatcher,
        request_queue=request_queue, kv_accountant=kv_accountant, admission=admission,
        layer_state=LayerState(num_layers=config.model.num_layers),
        completion=Completion(clock=clock, kv_accountant=kv_accountant),
    )

    for mb_id in (0, 1, 2):
        # PIM dispatch signal flow. Impl-10-pre-2 — k_total knob 폐기.
        dispatcher.register(MicroBatch(
            id=mb_id,
            kv_rows_total=config.time.rtl_fsm_tile_rows,
        ))
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


# test_dispatch_trace_t1_through_t5_sequence 삭제 — 옛 4노드 §6.5 시퀀스(O_PROJ 가 layer
# advance 트리거) 검증. S0 가 FFN 노드 + INSTANCE_B 추가, advance 를 O_PROJ→FFN 으로 이동시켜
# 시퀀스 obsolete. FFN/F3 dispatch 동역학은 test_phase2_ffn_stage(test_f3_overlap 등)가 대체.


def test_dispatch_trace_terminates_with_all_done():
    core = _make_trace_fixture()
    core.run_until_empty()
    for mb_id in core.dag.nodes:
        for ntype in NodeType:
            assert core.dag.get_node(mb_id, ntype).state is NodeState.DONE
    assert core.dispatcher.gpu_busy is False
    assert core.dispatcher.pim_busy is False
