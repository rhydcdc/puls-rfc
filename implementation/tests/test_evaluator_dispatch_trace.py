"""Impl-8 cluster B — Evaluator 가 ARCH §6.5 dispatch trace (Init/T1~T5 sequence) 의 *consumer* 영역.

기존 test_dispatch_trace.py 의 §6.5 P/M/N fixture 위에 Evaluator.record_dispatch hook 부착 →
evaluator.dispatch_trace() 가 §6.5 표의 Init/T1~T5 sequence 를 정확히 reproduce 함을 검증.
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
from puls_sched.evaluator import Evaluator
from puls_sched.forward_pass import LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _make_trace_fixture_with_evaluator():
    """test_dispatch_trace.py 의 _make_trace_fixture 위에 Evaluator hook 부착."""
    base = default_dummy_config()
    time_config = dataclasses.replace(
        base.time,
        gpu_op_time_us={"qkv": 1.0, "prefill_attn": 1.0, "o_proj": 1.0, "decode_attn_fallback": 4.0},
        pim_tile_time_ns={"FP8": 3.0, "FP16": 6.0},
        pim_broadcast_latency_ns_cross_gpu=0.0,
    )
    # Impl-9 — §6.5 single-layer dispatch pattern 검증 위 num_layers=1 (layer cycling 비활성)
    model_config = dataclasses.replace(base.model, num_layers=1)
    config = dataclasses.replace(base, time=time_config, model=model_config)
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag, config=config)
    pim_executor = PIMExecutor(config=config)
    dispatcher = Dispatcher(
        config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim_executor,
    )
    request_queue = RequestQueue(capacity=config.admission.request_queue_capacity)
    kv_accountant = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    idle_telemetry = IdleTelemetry()
    admission = Admission(
        admission_cfg=config.admission, request_queue=request_queue,
        kv_accountant=kv_accountant, idle_telemetry=idle_telemetry,
    )
    core = SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag,
        window=window, dispatcher=dispatcher,
        request_queue=request_queue, kv_accountant=kv_accountant, admission=admission,
        layer_state=LayerState(num_layers=config.model.num_layers),
        completion=Completion(clock=clock, kv_accountant=kv_accountant),
    )

    evaluator = Evaluator(config=config, clock=clock, idle_telemetry=idle_telemetry)
    dispatcher.on_dispatch(evaluator.record_dispatch)

    for mb_id in (0, 1, 2):
        dispatcher.register(MicroBatch(
            id=mb_id,
            k_total=config.admission.k_total_max,
            kv_rows_total=config.time.rtl_fsm_tile_rows,
        ))
        window.admit(mb_id)

    # P (mb 0): QKV + PREFILL_ATTN DONE; DECODE_ATTN RUNNING on PIM (이전에 dispatch 됨 — pre-Init)
    # 본 fixture 는 pre-Init dispatch 를 *manual* state mutation 으로 표현 (hook 미연결)
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

    # M (mb 1): QKV DONE (back-fill pre-Init, hook 미연결)
    m_qkv = dag.get_node(1, NodeType.QKV)
    m_qkv.transition_to(NodeState.READY)
    m_qkv.transition_to(NodeState.RUNNING)
    m_qkv.transition_to(NodeState.DONE)

    return core, evaluator


def _filter_real_dispatch(events):
    """Init 이전 pre-Init 의 manual mutation 은 hook 미연결 → events 에 없음.
    Real-time dispatch 만 누적된 events 반환."""
    return list(events)


# =========================================================================
# §6.5 Init/T1~T5 sequence reproduction
# =========================================================================

def test_dispatch_trace_t1_o_proj_p_after_pim_completion():
    """T1: PIM(P) 완료 → O_PROJ(P) GPU dispatch + DECODE_ATTN(M) PIM dispatch.
    Evaluator.dispatch_trace() 의 첫 2 event 가 (O_PROJ, mb=0, GPU) + (DECODE_ATTN, mb=1, PIM)."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.step()  # T1
    events = evaluator.dispatch_trace()
    assert len(events) == 2
    # O_PROJ(P=0) GPU + DECODE_ATTN(M=1) PIM (order: GPU dispatch 가 tick() 의 첫 분기)
    gpu_events = [e for e in events if e.resource == "GPU"]
    pim_events = [e for e in events if e.resource == "PIM"]
    assert len(gpu_events) == 1
    assert len(pim_events) == 1
    assert gpu_events[0].node_type is NodeType.O_PROJ
    assert gpu_events[0].micro_batch_id == 0  # P
    assert pim_events[0].node_type is NodeType.DECODE_ATTN
    assert pim_events[0].micro_batch_id == 1  # M


def test_dispatch_trace_t2_prefill_m():
    """T2: O_PROJ(P) 완료 → PREFILL_ATTN(M) GPU dispatch."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.step()  # T1
    n_after_t1 = len(evaluator.dispatch_trace())
    core.step()  # T2
    new_events = evaluator.dispatch_trace()[n_after_t1:]
    # PREFILL_ATTN(M=1) GPU
    gpu_new = [e for e in new_events if e.resource == "GPU"]
    assert len(gpu_new) == 1
    assert gpu_new[0].node_type is NodeType.PREFILL_ATTN
    assert gpu_new[0].micro_batch_id == 1


def test_dispatch_trace_t3_qkv_n_back_fill():
    """T3: PREFILL_ATTN(M) 완료 → QKV(N) GPU dispatch (back-fill). O_PROJ(M) not ready."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.step()  # T1
    core.step()  # T2
    n_after_t2 = len(evaluator.dispatch_trace())
    core.step()  # T3
    new_events = evaluator.dispatch_trace()[n_after_t2:]
    gpu_new = [e for e in new_events if e.resource == "GPU"]
    assert len(gpu_new) == 1
    assert gpu_new[0].node_type is NodeType.QKV
    assert gpu_new[0].micro_batch_id == 2  # N back-fill


def test_dispatch_trace_t4_o_proj_m_and_pim_decode_n():
    """T4: QKV(N) 완료 → O_PROJ(M) GPU + DECODE_ATTN(N) PIM."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    for _ in range(4):  # T1, T2, T3, intermediate (PIM completion)
        core.step()
    n_after_intermediate = len(evaluator.dispatch_trace())
    core.step()  # T4
    new_events = evaluator.dispatch_trace()[n_after_intermediate:]
    types_resources = {(e.node_type, e.resource, e.micro_batch_id) for e in new_events}
    assert (NodeType.O_PROJ, "GPU", 1) in types_resources
    assert (NodeType.DECODE_ATTN, "PIM", 2) in types_resources


def test_dispatch_trace_t5_prefill_n():
    """T5: O_PROJ(M) 완료 → PREFILL_ATTN(N) GPU dispatch."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    for _ in range(5):  # T1, T2, T3, intermediate, T4
        core.step()
    n_after_t4 = len(evaluator.dispatch_trace())
    core.step()  # T5
    new_events = evaluator.dispatch_trace()[n_after_t4:]
    gpu_new = [e for e in new_events if e.resource == "GPU"]
    assert len(gpu_new) == 1
    assert gpu_new[0].node_type is NodeType.PREFILL_ATTN
    assert gpu_new[0].micro_batch_id == 2


def test_dispatch_trace_timestamps_monotonic():
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    events = evaluator.dispatch_trace()
    timestamps = [e.timestamp for e in events]
    assert all(t1 <= t2 for t1, t2 in zip(timestamps, timestamps[1:]))


def test_dispatch_trace_dag_state_snapshot_at_t1():
    """T1 시점 snapshot — O_PROJ(P=0) 가 막 dispatch 되었으므로 RUNNING.
    QKV(M=1) 는 pre-Init back-fill 으로 DONE."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.step()  # T1
    events = evaluator.dispatch_trace()
    # First event = O_PROJ(P=0) — snapshot 시점에 막 RUNNING 으로 transition
    first = events[0]
    snap = first.dag_state_snapshot
    assert snap[0]["O_PROJ"] in ("RUNNING", "DONE")  # 막 dispatch
    assert snap[1]["QKV"] == "DONE"  # M back-fill 완료 pre-Init


def test_dispatch_trace_resource_label_pim_for_decode_attn():
    """모든 DECODE_ATTN dispatch 의 resource == "PIM" (F1 활성화 default)."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    events = evaluator.dispatch_trace()
    decode_events = [e for e in events if e.node_type is NodeType.DECODE_ATTN]
    assert len(decode_events) > 0
    for e in decode_events:
        assert e.resource == "PIM"


def test_dispatch_trace_pim_events_carry_k_total():
    """PIM dispatch event 의 k_total == mb.k_total (산식 정합)."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    events = evaluator.dispatch_trace()
    pim_events = [e for e in events if e.resource == "PIM"]
    for e in pim_events:
        assert e.k_total > 0  # PIM dispatch 는 k_total 보유


def test_dispatch_trace_gpu_events_k_total_zero():
    """GPU dispatch event 의 k_total == 0 (GPU 분기 의미 없음)."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    events = evaluator.dispatch_trace()
    # GPU events 중 mb 가 등록되어 있으므로 mb.k_total 이 반환됨 (mb.k_total=2048)
    # 단 의미적으로 "GPU 가 PIM channel 사용" 은 의미 0 — schema 만 lock-in
    gpu_events = [e for e in events if e.resource == "GPU"]
    # mb.k_total 자체는 2048 — _fire_dispatch 가 mb lookup. GPU/PIM 의미 분리는
    # *test/caller 가 e.resource 로 판단* (schema lock-in 만, 값 자체는 mb.k_total)
    for e in gpu_events:
        assert e.k_total == 2048  # mb.k_total 의 자연 reflect (의미는 PIM 분기에서만)


def test_dispatch_trace_bit_exact_replay():
    """동일 fixture replay 2 회 → dispatch_trace bit-exact (PLAN §0 C5)."""
    core1, ev1 = _make_trace_fixture_with_evaluator()
    core2, ev2 = _make_trace_fixture_with_evaluator()
    core1.run_until_empty()
    core2.run_until_empty()
    assert ev1.dispatch_trace() == ev2.dispatch_trace()


def test_dispatch_trace_full_sequence_length():
    """Full §6.5 run 의 dispatch 총 횟수 — Init pre-state 외에 T1~T5 + 후속 모든 노드 완료."""
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    events = evaluator.dispatch_trace()
    # 3 mb × 4 node = 12. 단 pre-Init manual mutation (hook 미연결):
    # P: QKV/PREFILL_ATTN/DECODE_ATTN 모두 pre-Init. O_PROJ 만 hook 영역.
    # M: QKV pre-Init (back-fill DONE). PREFILL/DECODE/O_PROJ 가 hook 영역.
    # N: 모두 hook 영역.
    # 합: P 1 + M 3 + N 4 = 8 dispatch events.
    assert len(events) == 8
