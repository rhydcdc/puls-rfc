from dataclasses import dataclass

from puls_sched.clock import Clock
from puls_sched.config import Config
from puls_sched.dag import DAG
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.invariants import check_I1, check_I2, check_I3, check_I4, check_I5
from puls_sched.node import Node, NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor


GPU_NODE_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.O_PROJ}
)
PIM_NODE_TYPES: frozenset[NodeType] = frozenset({NodeType.DECODE_ATTN})

GPU_PRIORITY_ORDER: tuple[NodeType, ...] = (
    NodeType.O_PROJ,
    NodeType.PREFILL_ATTN,
    NodeType.QKV,
)


@dataclass
class Dispatcher:
    config: Config
    clock: Clock
    queue: EventQueue
    dag: DAG
    pim_executor: PIMExecutor
    gpu_busy: bool = False
    pim_busy: bool = False

    def refresh_ready(self) -> None:
        for mb_id, nodes in self.dag.nodes.items():
            for ntype, node in nodes.items():
                if node.state is not NodeState.PENDING:
                    continue
                prereqs = self.dag.precedence[mb_id][ntype]
                if all(nodes[p].state is NodeState.DONE for p in prereqs):
                    node.transition_to(NodeState.READY)

    def _ready_of_types(self, types: frozenset[NodeType]) -> list[tuple[int, Node]]:
        out: list[tuple[int, Node]] = []
        for mb_id in sorted(self.dag.nodes.keys()):
            for ntype, node in self.dag.nodes[mb_id].items():
                if node.state is NodeState.READY and ntype in types:
                    out.append((mb_id, node))
        return out

    def pick_gpu(self) -> Node | None:
        for priority_type in GPU_PRIORITY_ORDER:
            candidates = self._ready_of_types(frozenset({priority_type}))
            if candidates:
                return candidates[0][1]
        return None

    def pick_pim(self) -> Node | None:
        candidates = self._ready_of_types(PIM_NODE_TYPES)
        return candidates[0][1] if candidates else None

    def _op_time(self, node: Node) -> float:
        if node.type in GPU_NODE_TYPES:
            return self.config.time.gpu_op_time_us[node.type.name.lower()]
        # PIM (decode-attn). Impl-4 형식 wiring only — args 는 config placeholder default.
        # 진짜 signal flow (MicroBatch 의 k_total · kv_rows_total 필드) 는 Impl-5 영역.
        return self.pim_executor.op_time(
            k_channels=self.config.admission.k_total_max,
            kv_rows_total=self.config.time.rtl_fsm_tile_rows,
        )

    def dispatch_gpu(self, node: Node) -> None:
        check_I4(self.gpu_busy)
        if node.type is NodeType.PREFILL_ATTN:
            check_I1(self.dag, node.micro_batch_id)
        elif node.type is NodeType.O_PROJ:
            check_I3(self.dag, node.micro_batch_id)
        node.transition_to(NodeState.RUNNING)
        self.gpu_busy = True
        self.queue.push(Event(
            timestamp=self.clock.now + self._op_time(node),
            type=EventType.KERNEL_COMPLETION,
            payload={
                "micro_batch_id": node.micro_batch_id,
                "node_type": node.type,
                "resource": "GPU",
            },
        ))

    def dispatch_pim(self, node: Node) -> None:
        check_I5(self.pim_busy)
        check_I2(self.dag, node.micro_batch_id)
        node.transition_to(NodeState.RUNNING)
        self.pim_busy = True
        self.queue.push(Event(
            timestamp=self.clock.now + self._op_time(node),
            type=EventType.KERNEL_COMPLETION,
            payload={
                "micro_batch_id": node.micro_batch_id,
                "node_type": node.type,
                "resource": "PIM",
            },
        ))

    def on_completion(self, event: Event) -> None:
        mb_id: int = event.payload["micro_batch_id"]
        ntype: NodeType = event.payload["node_type"]
        resource: str = event.payload["resource"]
        node = self.dag.get_node(mb_id, ntype)
        node.transition_to(NodeState.DONE)
        if resource == "GPU":
            self.gpu_busy = False
        elif resource == "PIM":
            self.pim_busy = False
        else:
            raise ValueError(f"unknown resource: {resource}")

    def tick(self) -> None:
        self.refresh_ready()
        if not self.gpu_busy:
            node = self.pick_gpu()
            if node is not None:
                self.dispatch_gpu(node)
        if not self.pim_busy:
            node = self.pick_pim()
            if node is not None:
                self.dispatch_pim(node)
