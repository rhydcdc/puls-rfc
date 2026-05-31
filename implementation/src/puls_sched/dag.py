from dataclasses import dataclass, field

from puls_sched.node import Node, NodeType


@dataclass
class DAG:
    nodes: dict[int, dict[NodeType, Node]] = field(default_factory=dict)
    precedence: dict[int, dict[NodeType, set[NodeType]]] = field(default_factory=dict)

    def add_micro_batch(self, micro_batch_id: int) -> None:
        if micro_batch_id in self.nodes:
            raise ValueError(f"micro_batch {micro_batch_id} already in DAG")
        self.nodes[micro_batch_id] = {
            t: Node(type=t, micro_batch_id=micro_batch_id) for t in NodeType
        }
        self.precedence[micro_batch_id] = {
            NodeType.QKV: set(),
            NodeType.PREFILL_ATTN: {NodeType.QKV},                              # I1
            NodeType.DECODE_ATTN: {NodeType.QKV},                               # I2
            NodeType.O_PROJ: {NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN},     # I3
            NodeType.FFN: {NodeType.O_PROJ},                                    # Phase-2 — inter-AB (F3): B FFN after A O_PROJ
        }

    def remove_micro_batch(self, micro_batch_id: int) -> None:
        del self.nodes[micro_batch_id]
        del self.precedence[micro_batch_id]

    def reset_micro_batch(self, micro_batch_id: int) -> None:
        """Impl-9 — 4 nodes 를 fresh PENDING 으로 재생성. 다음 layer dispatch 의 entry.

        ARCH §3.4 'L × cycle = forward pass' 정합 — 매 layer 마다 동일 4 op 재실행.
        Precedence (I1·I2·I3) 는 변경 없음 (mb 단위 invariant).
        """
        if micro_batch_id not in self.nodes:
            raise KeyError(f"micro_batch {micro_batch_id} not in DAG (reset failed)")
        self.nodes[micro_batch_id] = {
            t: Node(type=t, micro_batch_id=micro_batch_id) for t in NodeType
        }

    def get_node(self, micro_batch_id: int, node_type: NodeType) -> Node:
        return self.nodes[micro_batch_id][node_type]
