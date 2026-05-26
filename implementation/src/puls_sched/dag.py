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
        }

    def remove_micro_batch(self, micro_batch_id: int) -> None:
        del self.nodes[micro_batch_id]
        del self.precedence[micro_batch_id]

    def get_node(self, micro_batch_id: int, node_type: NodeType) -> Node:
        return self.nodes[micro_batch_id][node_type]
