from dataclasses import dataclass
from enum import Enum, auto


class NodeType(Enum):
    QKV = auto()
    PREFILL_ATTN = auto()
    DECODE_ATTN = auto()
    O_PROJ = auto()


class NodeState(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    DONE = auto()


_VALID_NODE_TRANSITIONS: dict[NodeState, set[NodeState]] = {
    NodeState.PENDING: {NodeState.READY},
    NodeState.READY: {NodeState.RUNNING},
    NodeState.RUNNING: {NodeState.DONE},
    NodeState.DONE: set(),
}


@dataclass
class Node:
    type: NodeType
    micro_batch_id: int
    state: NodeState = NodeState.PENDING

    def transition_to(self, new_state: NodeState) -> None:
        if new_state not in _VALID_NODE_TRANSITIONS[self.state]:
            raise ValueError(f"invalid node transition: {self.state} -> {new_state}")
        self.state = new_state
