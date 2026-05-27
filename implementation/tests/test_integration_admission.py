"""Cross-module integration stress for Impl-3.

PLAN.md §0.5 reminder — cross-module invariant. Impl-2 의
`test_stress_100_micro_batch_no_invariant_violation` 패턴 정합.
"""

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.invariants import (
    check_I1, check_I2, check_I3, check_I4, check_I5,
)
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.node import NodeState, NodeType
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _make_core():
    config = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag)
    dispatcher = Dispatcher(config=config, clock=clock, queue=queue, dag=dag)
    rq = RequestQueue(capacity=config.admission.request_queue_capacity)
    kv = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    admission = Admission(
        admission_cfg=config.admission, request_queue=rq,
        kv_accountant=kv, idle_telemetry=IdleTelemetry(),
    )
    return SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag,
        window=window, dispatcher=dispatcher,
        request_queue=rq, kv_accountant=kv, admission=admission,
    )


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length)


def test_stress_100_cycle_admission_dispatch_no_invariant_violation():
    """100 cycle 위 I1~I5 위반 0 + KV/queue/window state consistency."""
    core = _make_core()
    n_requests = 100
    arrived = 0
    admitted_via_layer1 = 0
    rejected = 0

    for i in range(n_requests):
        req = _make_req(i, kv_length=10)
        pushed = core.request_queue.push(req)
        arrived += 1
        if not pushed:
            rejected += 1
        # Each cycle: admission tick + drain dispatcher
        core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        # Manual reset busy to allow next dispatch (Impl-3 단계 mock — Impl-6 completion 부재)
        if i % 2 == 0:
            core.dispatcher.gpu_busy = False
            core.dispatcher.pim_busy = False

    # Invariant I1~I5 위반 0 — explicit re-check on current DAG state
    for mb_id, nodes in core.dag.nodes.items():
        if nodes[NodeType.PREFILL_ATTN].state is NodeState.RUNNING:
            check_I1(core.dag, mb_id)
        if nodes[NodeType.DECODE_ATTN].state is NodeState.RUNNING:
            check_I2(core.dag, mb_id)
        if nodes[NodeType.O_PROJ].state is NodeState.RUNNING:
            check_I3(core.dag, mb_id)
    # I4·I5: GPU/PIM at most 1 running each
    gpu_running = sum(1 for nodes in core.dag.nodes.values()
                      for n in nodes.values()
                      if n.state is NodeState.RUNNING and n.type is not NodeType.DECODE_ATTN)
    pim_running = sum(1 for nodes in core.dag.nodes.values()
                      for n in nodes.values()
                      if n.state is NodeState.RUNNING and n.type is NodeType.DECODE_ATTN)
    assert gpu_running <= 1
    assert pim_running <= 1

    # KV remaining + queue/window consistency
    assert core.kv_accountant.remaining + core.kv_accountant.used == core.kv_accountant.capacity
    assert len(core.window.current_ids()) <= 3


def test_stress_admission_completion_kv_roundtrip():
    """admission admit N req → mock completion (release) → kv.remaining == initial.
    Impl-3 단독 cross-module 검증의 핵심. Real Completion handler 는 Impl-6."""
    core = _make_core()
    initial = core.kv_accountant.remaining

    # Push 50 req + admit via 50 alternating ticks
    for i in range(50):
        core.request_queue.push(_make_req(i, kv_length=20))
        core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        core.dispatcher.gpu_busy = False
        core.dispatcher.pim_busy = False

    # All KV should now be allocated to admitted requests
    assert core.kv_accountant.used == 50 * 20

    # Mock Completion.finalize for each — release all admitted KV
    # (Real handler in Impl-6; here we directly call release on each admitted req id.)
    admitted_ids = list(core.kv_accountant._admitted.keys())
    for req_id in admitted_ids:
        # Reconstruct req with the recorded kv_length
        kv_length = core.kv_accountant._admitted[req_id]
        core.kv_accountant.release(Request(id=req_id, prompt_tokens=[1], kv_length=kv_length))

    assert core.kv_accountant.remaining == initial
    assert core.kv_accountant.used == 0


def test_determinism_100_cycle_admission_main_loop_bit_exact():
    """PLAN §0 C5 의 Impl-3 단계 prefigure — 동일 event stream 2회 run → bit-exact state."""
    def run_once():
        core = _make_core()
        for i in range(50):
            core.request_queue.push(_make_req(i, kv_length=20))
            core._handle(Event(
                timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
            ))
            core.dispatcher.gpu_busy = False
            core.dispatcher.pim_busy = False
        return (
            len(core.request_queue),
            core.kv_accountant.used,
            core.window.current_ids(),
            core.dispatcher.gpu_busy,
            core.dispatcher.pim_busy,
            core._next_mb_id,
        )

    assert run_once() == run_once()
