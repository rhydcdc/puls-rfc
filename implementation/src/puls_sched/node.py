from dataclasses import dataclass
from enum import Enum, auto


class NodeType(Enum):
    QKV = auto()
    PREFILL_ATTN = auto()
    DECODE_ATTN = auto()
    O_PROJ = auto()
    # Phase-2 — Instance B FFN stage (inter-AB pipeline, ARCH §3.4 / §5.7 F3).
    # Instance A 4 노드 뒤 단일 B 노드. O_PROJ → FFN 종속 (A 출력이 B 입력).
    FFN = auto()


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
