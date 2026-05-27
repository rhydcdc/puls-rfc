from puls_sched.event import Event, EventType
from puls_sched.node import NodeState, NodeType


def test_single_event_dispatch_reaches_handler(scheduler_core):
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


def test_acceptance_10_micro_batch_trace(scheduler_core):
    all_timestamps = []
    for i in range(10):
        scheduler_core.window.admit(i)
        all_timestamps.extend(_drain_micro_batch(scheduler_core, i, base_time=float(i * 10)))

    assert all_timestamps == sorted(all_timestamps)
    assert len(all_timestamps) == 10 * len(NodeType)
    assert len(scheduler_core.window.current_ids()) == 3
    assert scheduler_core.window.current_ids() == (7, 8, 9)
    assert set(scheduler_core.dag.nodes.keys()) == set(scheduler_core.window.current_ids())
