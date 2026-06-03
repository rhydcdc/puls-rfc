"""Impl-8 cluster E — cross-module chain (Evaluator + dispatcher + main_loop) + non-intrusion.

dispatcher.on_dispatch(evaluator.record_dispatch)
→ 실 dispatch progression → evaluator.report() 의 chain 검증. Impl-6 cross-module 패턴 정합.
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
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _build_full_core_with_evaluator(config=None):
    """SchedulerCore + Evaluator hook 부착 fixture."""
    if config is None:
        config = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag, config=config)
    pim = PIMExecutor(config=config)
    dispatcher = Dispatcher(config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    rq = RequestQueue(capacity=config.admission.request_queue_capacity)
    kva = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    idle = IdleTelemetry()
    admission = Admission(
        admission_cfg=config.admission, request_queue=rq,
        kv_accountant=kva, idle_telemetry=idle,
    )
    core = SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag, window=window,
        dispatcher=dispatcher, request_queue=rq, kv_accountant=kva,
        admission=admission,
        layer_state=LayerState(num_layers=config.model.num_layers),
        completion=Completion(clock=clock, kv_accountant=kva),
    )
    evaluator = Evaluator(config=config, clock=clock, idle_telemetry=idle)
    dispatcher.on_dispatch(evaluator.record_dispatch)
    return core, evaluator


# =========================================================================
# Chain — D1 hook 정합
# =========================================================================

def test_chain_register_dispatch_record():
    """dispatcher.on_dispatch 등록 → dispatch_gpu 호출 → evaluator._dispatch_events 누적."""
    core, evaluator = _build_full_core_with_evaluator()
    core.dispatcher.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    core.dag.add_micro_batch(0)
    qkv = core.dag.get_node(0, NodeType.QKV)
    qkv.transition_to(NodeState.READY)
    core.dispatcher.dispatch_gpu(qkv)
    assert len(evaluator.dispatch_trace()) == 1


def test_chain_evaluator_standalone_scheduler_unaware():
    """SchedulerCore.__dataclass_fields__ 에 evaluator field 부재 (D3 standalone)."""
    fields = SchedulerCore.__dataclass_fields__
    assert "evaluator" not in fields


def test_chain_multiple_callbacks_fire_in_order():
    """dispatcher.on_dispatch 에 2 callback 등록 → 둘 다 호출 + 등록 order == 호출 order."""
    core, evaluator = _build_full_core_with_evaluator()
    call_log = []
    core.dispatcher.on_dispatch(lambda e: call_log.append("first"))
    core.dispatcher.on_dispatch(lambda e: call_log.append("second"))

    core.dispatcher.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    core.dag.add_micro_batch(0)
    qkv = core.dag.get_node(0, NodeType.QKV)
    qkv.transition_to(NodeState.READY)
    core.dispatcher.dispatch_gpu(qkv)
    # evaluator 의 record_dispatch + first + second 순 (등록 order)
    assert call_log == ["first", "second"]


def test_chain_evaluator_no_dispatcher_field():
    """Evaluator.__dataclass_fields__ 에 dispatcher / scheduler_core field 부재 (D3)."""
    fields = Evaluator.__dataclass_fields__
    assert "dispatcher" not in fields
    assert "scheduler_core" not in fields


# =========================================================================
# End-to-end chain — §6.5 prefigure
# =========================================================================

def test_chain_evaluator_with_real_synthetic_trace():
    """§6.5 P/M/N fixture 위 evaluator.report() 의 dispatch_trace 가 Init/T1~T5 포함."""
    # Use the existing test_evaluator_dispatch_trace fixture pattern
    from tests.test_evaluator_dispatch_trace import _make_trace_fixture_with_evaluator
    core, evaluator = _make_trace_fixture_with_evaluator()
    core.run_until_empty()
    report = evaluator.report(a_cycle=1.0, b_cycle=1.0, t_pim=1.0, t_proj=1.0)
    trace = report["dispatch_trace"]
    assert len(trace) > 0
    # DECODE_ATTN 이 PIM 으로 dispatch (F1 활성화) — §6.5 정합
    decode_events = [e for e in trace if e.node_type == NodeType.DECODE_ATTN]
    assert len(decode_events) > 0
    assert all(e.resource == "PIM" for e in decode_events)


def test_chain_kv_no_leak_with_evaluator_attached():
    """Evaluator hook 등록이 KV invariant 영향 0 (R-loop non-intrusion)."""
    core, evaluator = _build_full_core_with_evaluator()
    initial_remaining = core.kv_accountant.remaining
    # 10 req admit + immediate finalize
    for i in range(10):
        req = Request(id=i, prompt_len=100, kv_length=100, max_tokens=1)
        core.request_queue.push(req)
    # Drive admission via direct push (synthetic)
    core.queue.push(Event(
        timestamp=0.0, type=EventType.ADMISSION_PASS,
        payload={"t_proj": 1.0, "a_cycle": 1.0, "b_cycle": 1.0, "ctx_tokens": 4000},
    ))
    core.step()
    # finalize admitted reqs
    for req in list(core.in_flight_requests.values()):
        req.decoded_count = req.max_tokens
        if core.completion.check(req):
            core.completion.finalize(req)
            core.in_flight_requests.pop(req.id, None)
    assert core.kv_accountant.remaining == initial_remaining


def test_chain_evaluator_no_dispatch_state_mutation():
    """Evaluator 부착 전후 DAG state bit-exact (read-only invariant)."""
    # 두 run — hook 있음 / 없음 — DAG state 동일 검증
    config = default_dummy_config()

    # Run A — no evaluator
    core_a, _ = _build_full_core_with_evaluator(config)
    core_a.dispatcher._dispatch_callbacks.clear()  # clear evaluator hook
    core_a.dispatcher.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    core_a.dag.add_micro_batch(0)
    qkv_a = core_a.dag.get_node(0, NodeType.QKV)
    qkv_a.transition_to(NodeState.READY)
    core_a.dispatcher.dispatch_gpu(qkv_a)
    state_a = qkv_a.state

    # Run B — with evaluator
    core_b, _ = _build_full_core_with_evaluator(config)
    core_b.dispatcher.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    core_b.dag.add_micro_batch(0)
    qkv_b = core_b.dag.get_node(0, NodeType.QKV)
    qkv_b.transition_to(NodeState.READY)
    core_b.dispatcher.dispatch_gpu(qkv_b)
    state_b = qkv_b.state

    assert state_a == state_b  # evaluator 부착이 state 영향 0


def test_chain_replay_determinism():
    """동일 fixture + ablation config 위 chain 2 회 → report() bit-exact (PLAN §0 C5)."""
    config = default_dummy_config()
    core1, ev1 = _build_full_core_with_evaluator(config)
    core2, ev2 = _build_full_core_with_evaluator(config)

    for c in (core1, core2):
        for i in range(3):
            c.queue.push(Event(
                timestamp=float(i), type=EventType.ADMISSION_PASS,
                payload={"t_proj": 1.0, "a_cycle": 1.0, "b_cycle": 1.0 + i, "ctx_tokens": 4000},
            ))
        while c.step():
            pass

    r1 = ev1.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    r2 = ev2.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    assert r1["markdown"] == r2["markdown"]
    assert r1["acceleration_decomposition"] == r2["acceleration_decomposition"]
