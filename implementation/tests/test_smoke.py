from puls_sched.event import Event, EventType
from puls_sched.node import NodeType


def test_single_event_dispatch_reaches_handler(scheduler_core):
    event = Event(timestamp=1.0, type=EventType.KERNEL_COMPLETION)
    scheduler_core.queue.push(event)
    assert scheduler_core.step() is True
    assert scheduler_core.step() is False  # queue empty
    assert scheduler_core.clock.now == 1.0


def _make_10_micro_batch_trace(scheduler_core):
    for i in range(10):
        scheduler_core.window.admit(i)
        for j, node_type in enumerate(NodeType):
            event = Event(
                timestamp=float(i * 10 + j),
                type=EventType.KERNEL_COMPLETION,
                payload={"micro_batch_id": i, "node_type": node_type},
            )
            scheduler_core.queue.push(event)


def test_acceptance_10_micro_batch_trace(scheduler_core):
    _make_10_micro_batch_trace(scheduler_core)

    # 1. 시간순 dequeue
    timestamps_popped = []
    while scheduler_core.step():
        timestamps_popped.append(scheduler_core.clock.now)
    assert timestamps_popped == sorted(timestamps_popped)
    assert len(timestamps_popped) == 10 * len(NodeType)  # 10 micro-batches × 4 nodes

    # 2. Window capacity 유지 — 3 micro-batch (마지막 3 개)
    assert len(scheduler_core.window.current_ids()) == 3
    assert scheduler_core.window.current_ids() == (7, 8, 9)

    # 3. 무한 메모리 누적 없음 — DAG keys == window current_ids
    assert set(scheduler_core.dag.nodes.keys()) == set(scheduler_core.window.current_ids())
