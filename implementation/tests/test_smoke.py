import dataclasses

import pytest

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


@pytest.fixture
def scheduler_core_l1():
    """Impl-9 — num_layers=1 fixture (layer cycling 비활성, §6.5 single-layer pattern 영역)."""
    base = default_dummy_config()
    config = dataclasses.replace(base, model=dataclasses.replace(base.model, num_layers=1))
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
    admission = Admission(
        admission_cfg=config.admission, request_queue=request_queue,
        kv_accountant=kv_accountant, idle_telemetry=IdleTelemetry(),
    )
    return SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag,
        window=window, dispatcher=dispatcher,
        request_queue=request_queue, kv_accountant=kv_accountant, admission=admission,
        layer_state=LayerState(num_layers=1),
        completion=Completion(clock=clock, kv_accountant=kv_accountant),
    )


def _register_mb(scheduler_core, mb_id: int) -> None:
    """Impl-5 — PIM dispatch signal flow. backward-compat (k=k_total_max, rows=tile_rows)."""
    scheduler_core.dispatcher.register(MicroBatch(
        id=mb_id,
        k_total=scheduler_core.config.admission.k_total_max,
        kv_rows_total=scheduler_core.config.time.rtl_fsm_tile_rows,
    ))


def test_single_event_dispatch_reaches_handler(scheduler_core):
    _register_mb(scheduler_core, 0)
    scheduler_core.window.admit(0)
    qkv = scheduler_core.dag.get_node(0, NodeType.QKV)
    qkv.transition_to(NodeState.READY)
    qkv.transition_to(NodeState.RUNNING)
    scheduler_core.dispatcher.gpu_busy = True

    event = Event(
        timestamp=1.0,
        type=EventType.KERNEL_COMPLETION,
        payload={"micro_batch_id": 0, "node_type": NodeType.QKV, "resource": "GPU"},
    )
    scheduler_core.queue.push(event)
    assert scheduler_core.step() is True
    assert scheduler_core.clock.now == 1.0
    assert qkv.state is NodeState.DONE


def _drain_micro_batch(scheduler_core, mb_id, base_time):
    """Pre-set the 4 nodes to RUNNING (skipping refresh_ready promotion),
    push 4 completion events, drain. Each mb's nodes complete before the next admit."""
    for node_type in NodeType:
        node = scheduler_core.dag.get_node(mb_id, node_type)
        node.transition_to(NodeState.READY)
        node.transition_to(NodeState.RUNNING)
    for j, node_type in enumerate(NodeType):
        resource = "PIM" if node_type is NodeType.DECODE_ATTN else "GPU"
        scheduler_core.queue.push(Event(
            timestamp=base_time + float(j),
            type=EventType.KERNEL_COMPLETION,
            payload={"micro_batch_id": mb_id, "node_type": node_type, "resource": resource},
        ))
    timestamps = []
    while scheduler_core.step():
        timestamps.append(scheduler_core.clock.now)
    return timestamps


def test_acceptance_10_micro_batch_trace(scheduler_core_l1):
    """Impl-9 ARCH-compliant 갱신 — num_layers=1 fixture 위 10 mb 순차 lifecycle.

    각 mb 가 O_PROJ done → token_signal=True (L=1) → empty decode_tokens → evict.
    Final state: 모든 10 mb evict 완료 (이전 auto-evict overflow path 의 last-3 영역과 다름).
    """
    all_timestamps = []
    for i in range(10):
        _register_mb(scheduler_core_l1, i)
        scheduler_core_l1.window.admit(i)
        all_timestamps.extend(_drain_micro_batch(scheduler_core_l1, i, base_time=float(i * 10)))

    assert all_timestamps == sorted(all_timestamps)
    assert len(all_timestamps) == 10 * len(NodeType)
    # Impl-9 — 각 mb lifecycle 완료 후 evict (auto-evict path 비활성, explicit evict)
    assert scheduler_core_l1.window.current_ids() == ()
    assert scheduler_core_l1.dag.nodes == {}
