from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from puls_sched.clock import Clock
from puls_sched.config import Config, compute_gpu_op_time_s
from puls_sched.dag import DAG
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.invariants import check_I1, check_I2, check_I3, check_I4, check_I5
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import Node, NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor

if TYPE_CHECKING:
    from puls_sched.evaluator import DispatchEvent


GPU_NODE_TYPES: frozenset[NodeType] = frozenset(
    {NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.O_PROJ}
)
PIM_NODE_TYPES: frozenset[NodeType] = frozenset({NodeType.DECODE_ATTN})

GPU_PRIORITY_ORDER: tuple[NodeType, ...] = (
    NodeType.O_PROJ,
    NodeType.PREFILL_ATTN,
    NodeType.QKV,
)


DispatchCallback = Callable[["DispatchEvent"], None]


@dataclass
class Dispatcher:
    config: Config
    clock: Clock
    queue: EventQueue
    dag: DAG
    pim_executor: PIMExecutor
    micro_batches: dict[int, MicroBatch] = field(default_factory=dict)  # Impl-5 — Q1-bis lookup
    gpu_busy: bool = False
    pim_busy: bool = False
    _dispatch_callbacks: list[DispatchCallback] = field(default_factory=list)  # Impl-8 — D1 hook (evaluator 등록 점)
    # Impl-10-pre-1 O8.2 — optional IdleTelemetry wiring. None 시 record_active skip (backward-compat 영역).
    idle_telemetry: Optional[IdleTelemetry] = None

    def register(self, mb: MicroBatch) -> None:
        """MicroBatch 를 dispatcher 의 lookup 저장소에 등록 (Q1-bis).

        main_loop 의 ADMISSION_TICK body 가 spec → MicroBatch 변환 후 호출.
        """
        if mb.id in self.micro_batches:
            raise RuntimeError(f"MicroBatch {mb.id} already registered")
        self.micro_batches[mb.id] = mb

    def unregister(self, mb_id: int) -> None:
        """Window eviction 시 호출 (Impl-9 wiring). 본 단계는 API 만 노출."""
        if mb_id not in self.micro_batches:
            raise RuntimeError(f"MicroBatch {mb_id} not registered (double unregister?)")
        del self.micro_batches[mb_id]

    def on_dispatch(self, callback: DispatchCallback) -> None:
        """Dispatch 시점 event capture 위 callback 등록 (Impl-8 D1 hook).

        Evaluator 같은 외부 inspector 가 dispatch_trace 캡처 위 등록. Dispatcher 자체는
        Evaluator 를 모름 (D3 standalone — Dispatcher 가 reverse-aware 아님).
        """
        self._dispatch_callbacks.append(callback)

    def _fire_dispatch(self, node: Node, resource: str) -> None:
        """등록된 callback 들에게 dispatch event 통지 (Impl-8 D1 hook fire)."""
        if not self._dispatch_callbacks:
            return
        # Lazy import — dispatcher ↔ evaluator 순환 회피
        from puls_sched.evaluator import DispatchEvent
        event = DispatchEvent(
            timestamp=self.clock.now,
            micro_batch_id=node.micro_batch_id,
            node_type=node.type,
            resource=resource,
            dag_state_snapshot=self._snapshot_dag_state(),
        )
        for cb in self._dispatch_callbacks:
            cb(event)

    def _snapshot_dag_state(self) -> dict[int, dict[str, str]]:
        """DAG state 의 defensive copy — Evaluator 가 받은 snapshot 위 mutation 시 DAG 영향 0."""
        return {
            mb_id: {ntype.name: node.state.name for ntype, node in nodes.items()}
            for mb_id, nodes in self.dag.nodes.items()
        }

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
        """Per-mb spec-derived op_time. Stage 1 fixed lookup 폐기 (Stage 2 Impl-10 main).

        ARCH §3.5.2 *Computed Wait* literal 정합 + 사용자 의도 *"각 trace 위 정확 산출"*
        정합. Stage 2 위:

        - GPU 3 node (QKV, PREFILL_ATTN, O_PROJ) — `compute_gpu_op_time_s` per-mb 산출
            (B200 FP16 peak × MFU + Llama-3 70B spec + per-mb batch + causal ctx)
        - DECODE_ATTN — PIM 산출 (default) 또는 F1 ablation 위 GPU fallback reference
            (gpu_op_time_us["decode_attn_fallback"], Impl-11 calibration 영역)
        """
        mb = self.micro_batches.get(node.micro_batch_id)
        if node.type in GPU_NODE_TYPES:
            if mb is None:
                # Stage 1 test fixture backward-compat — default mb 위 spec-derived 산출
                # (production 영역 위 admission 이 mb register 영원 → 본 fallback 영원 사용 0).
                # 사용자 framing 정합 — dummy lookup 영원 사용 영원, spec-derived 영원 보존.
                from puls_sched.micro_batch import MicroBatch
                mb = MicroBatch(id=node.micro_batch_id, decode_tokens={0: 0})
            # Stage 2 — per-mb spec-derived (seconds → microseconds for clock unit)
            return compute_gpu_op_time_s(node.type, mb, self.config.calibration, self.config.model) * 1e6
        # PIM (decode-attn).
        if self.config.ablation.f1_disabled:
            # F1 ablation — GPU fallback reference (Impl-11 영역). DECODE_ATTN spec-derived 영원
            # closed-form 산식 (Stage 2 위 Impl-11 deferred — reference 보존).
            return self.config.time.gpu_op_time_us["decode_attn_fallback"]
        if mb is None:
            raise RuntimeError(
                f"PIM dispatch for unregistered MicroBatch {node.micro_batch_id}"
            )
        # Stage 2 unit convention — PIMExecutor.op_time() 반환 = ns, clock + GPU op_time = µs.
        # Stage 1 dummy 영역 위 영원 unit mismatch (1.0 dummy 위 의미 0), Stage 2 calibrated
        # 위 정정. PIM (ns) → µs 위 × 1e-3.
        return self.pim_executor.op_time(
            kv_rows_total=mb.kv_rows_total,
            kv_rows_lockstep=mb.kv_rows_lockstep,   # Impl-8 — F5 ablation 위
        ) * 1e-3

    def dispatch_gpu(self, node: Node) -> None:
        check_I4(self.gpu_busy)
        if node.type is NodeType.PREFILL_ATTN:
            check_I1(self.dag, node.micro_batch_id)
        elif node.type is NodeType.O_PROJ:
            check_I3(self.dag, node.micro_batch_id)
        node.transition_to(NodeState.RUNNING)
        self.gpu_busy = True
        op_time = self._op_time(node)
        t_start = self.clock.now
        self.queue.push(Event(
            timestamp=t_start + op_time,
            type=EventType.KERNEL_COMPLETION,
            payload={
                "micro_batch_id": node.micro_batch_id,
                "node_type": node.type,
                "resource": "GPU",
            },
        ))
        # Impl-10-pre-1 O8.2 — gpu_instance_a activity recording (ARCH §6.4 intra-A balance signal)
        if self.idle_telemetry is not None:
            self.idle_telemetry.record_active("gpu_instance_a", t_start, t_start + op_time)
        self._fire_dispatch(node, resource="GPU")    # Impl-8 — D1 hook (evaluator 통지)

    def dispatch_pim(self, node: Node) -> None:
        check_I5(self.pim_busy)
        check_I2(self.dag, node.micro_batch_id)
        node.transition_to(NodeState.RUNNING)
        self.pim_busy = True
        op_time = self._op_time(node)
        t_start = self.clock.now
        self.queue.push(Event(
            timestamp=t_start + op_time,
            type=EventType.KERNEL_COMPLETION,
            payload={
                "micro_batch_id": node.micro_batch_id,
                "node_type": node.type,
                "resource": "PIM",
            },
        ))
        # Impl-10-pre-1 O8.2 — pim_instance_a activity recording (ARCH §6.4 intra-A balance signal)
        if self.idle_telemetry is not None:
            self.idle_telemetry.record_active("pim_instance_a", t_start, t_start + op_time)
        self._fire_dispatch(node, resource="PIM")    # Impl-8 — D1 hook (evaluator 통지). F1 ablation 시에도 resource="PIM" 유지 (I5 invariant 정합)

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
