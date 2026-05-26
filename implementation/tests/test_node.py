import pytest

from puls_sched.node import Node, NodeState, NodeType, _VALID_NODE_TRANSITIONS


def test_node_state_forward():
    node = Node(type=NodeType.QKV, micro_batch_id=0)
    node.transition_to(NodeState.READY)
    assert node.state == NodeState.READY
    node.transition_to(NodeState.RUNNING)
    assert node.state == NodeState.RUNNING
    node.transition_to(NodeState.DONE)
    assert node.state == NodeState.DONE


def test_node_state_backward_rejected():
    node = Node(type=NodeType.QKV, micro_batch_id=0, state=NodeState.DONE)
    with pytest.raises(ValueError, match="invalid node transition"):
        node.transition_to(NodeState.RUNNING)


_INVALID_NODE_PAIRS = [
    (a, b)
    for a in NodeState
    for b in NodeState
    if b not in _VALID_NODE_TRANSITIONS[a]
]


@pytest.mark.parametrize("from_state,to_state", _INVALID_NODE_PAIRS)
def test_node_state_invalid_transitions_rejected(from_state, to_state):
    node = Node(type=NodeType.QKV, micro_batch_id=0, state=from_state)
    with pytest.raises(ValueError, match="invalid node transition"):
        node.transition_to(to_state)
