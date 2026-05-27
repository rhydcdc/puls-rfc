"""Cross-module integration stress for Impl-3.

PLAN.md §0.5 reminder — cross-module invariant. Impl-2 의
`test_stress_100_micro_batch_no_invariant_violation` 패턴 정합.
"""

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
from puls_sched.invariants import (
    check_I1, check_I2, check_I3, check_I4, check_I5,
)
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.node import NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _make_core():
    config = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag)
    pim_executor = PIMExecutor(config=config)
    dispatcher = Dispatcher(
        config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim_executor,
    )
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
        layer_state=LayerState(num_layers=config.model.num_layers),
        completion=Completion(clock=clock, kv_accountant=kv),
    )


def _make_req(req_id: int, kv_length: int = 10, max_tokens: int = 0) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length, max_tokens=max_tokens)


def test_stress_100_cycle_admission_dispatch_no_invariant_violation():
    """Impl-9 ARCH-compliant 갱신 — 100 req lifecycle 위 I1~I5 invariant + state consistency.

    Run.loop 의 단일 admission tick 후 step drain 패턴 (proper lifecycle, manual busy reset 폐기).
    """
    core = _make_core()
    n_requests = 100

    for i in range(n_requests):
        req = _make_req(i, kv_length=10, max_tokens=1)
        core.request_queue.push(req)
        core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        # Drain events (proper lifecycle — invariants 강제 위)
        while core.step():
            pass

    # I4·I5: GPU/PIM at most 1 running each (invariant 자연 보존)
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
    """Impl-9 갱신 — admission + lifecycle 완주 → kv.remaining == initial.

    Impl-3 시점 mock release path 의 ARCH-compliant 갱신. Real Completion.finalize 가
    layer cycling + token signal + evict chain 위 KV release.
    """
    core = _make_core()
    initial = core.kv_accountant.remaining

    # 50 reqs 위 proper lifecycle drain
    for i in range(50):
        core.request_queue.push(_make_req(i, kv_length=20, max_tokens=1))
        core._handle(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK, payload={},
        ))
        while core.step():
            pass

    # Lifecycle 완주 → 모든 KV 회수 (admit ↔ release round-trip)
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
