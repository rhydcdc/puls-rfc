"""Structural Evaluator — D1 증거 산출 + D2 schema 골격.

PLAN.md §4 Impl-8 + ARCH §6.7 정합. *절대 metric 미산출 (TTFT · TPOT · throughput · goodput).
Comparative baseline 미산출 (Sarathi · vLLM).*

7 산출 method + 2 callback method = 9 method total:
- record_dispatch / record_admission_tick — D1 hook callback (dispatcher / main_loop 가 fire)
- dispatch_trace — §6.5 Init/T1~T5 sequence event log
- admission_convergence — §6.4 deadband 위 idle fraction 시간 series 위 oscillation / 수렴 판정
- idle_fraction — Instance A scope (GPU · PIM 2 자원, per-instance A/B split 은 Impl-9 — O8.1)
- pim_utilization — Σ k_total · dt / (k_max · total_time) aggregate channel-time
- pipeline_efficiency — max(A, B) / (A + B) ratio
- acceleration_decomposition — F1·F2·F3·F5 cycle ratio direction 표 (D2 schema 골격, F4 미포함)
- report — Python dict + markdown 표 (PULS 단독, Comparative baseline 없음)
"""

from dataclasses import dataclass, field
from enum import Enum

from puls_sched.clock import Clock
from puls_sched.config import Config
from puls_sched.deadband import in_band, lookup_width
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.node import NodeType


# ============================================================================
# Schema dataclasses
# ============================================================================


@dataclass(frozen=True)
class DispatchEvent:
    """Dispatcher hook 가 fire 하는 dispatch 시점 event snapshot.

    ARCH §6.5 Init/T1~T5 trace 의 *one row* — timestamp · 어느 mb 의 어느 노드가 어디로 dispatch.
    k_total 은 PIM dispatch 시 mb.k_total, GPU dispatch 시 0 (의미 없는 default).
    """

    timestamp: float
    micro_batch_id: int
    node_type: NodeType
    resource: str                                       # "GPU" | "PIM"
    k_total: int                                        # PIM dispatch 시 mb.k_total. GPU 분기 시 0
    dag_state_snapshot: dict                            # {mb_id: {node_type_name: state_name}} — defensive copy


@dataclass(frozen=True)
class AdmissionSnapshot:
    """SchedulerCore admission tick hook 가 fire 하는 snapshot.

    ARCH §6.4 deadband convergence trace 의 *one row* — admission tick 자연 cadence.
    Spec=None (admission 실패) 도 snapshot 누적 (convergence series 의 의미 있는 entry).
    """

    timestamp: float
    gpu_idle_fraction: float
    pim_idle_fraction: float
    a_cycle: float
    b_cycle: float
    ctx_tokens: int
    spec_admitted: bool
    n: int
    k_total: int


@dataclass(frozen=True)
class ConvergenceVerdict:
    """admission_convergence() 의 판정 결과. PLAN §4 Impl-8 acceptance — oscillation / 수렴 판정."""

    converged: bool
    oscillating: bool
    in_band_fraction: float
    samples: int


class AblationSource(Enum):
    """Acceleration source ID — ARCH §5.7 정합. F4 는 precondition 영역 (별도 source 아님)."""

    F1 = "F1_SP_PIM"
    F2 = "F2_DOUBLE_BUFFER"
    F3 = "F3_INSTANCE_AB"
    F5 = "F5_CHANNEL_INDEP"


@dataclass(frozen=True)
class DecompositionCell:
    """acceleration_decomposition() 의 *one cell* — ablation flag on/off 의 cycle ratio direction.

    *정량 절대값 아님* (PLAN §0.5 정합) — direction (cycle_without / cycle_with) 의 *부호* 와 *산식 정합* 만.
    Impl-10 calibrated input 위 정량값으로 교체.
    """

    source: AblationSource
    cycle_with_source: float                            # F-on path 의 cycle 산식 산출값
    cycle_without_source: float                         # F-off path 의 cycle 산식 산출값
    ratio: float                                        # without / with (1.0 이상 = source 가 가속)
    direction_positive: bool                            # ratio > 1.0 sanity check


# ============================================================================
# Evaluator
# ============================================================================


# admission_convergence heuristic 임계값 (통계 의미 const, config 의존 0 — Impl-6 KS p-value 0.05 패턴 정합).
# O8.3 carry-over — Impl-10 calibrated input 위 deadband sensitivity 실측 후 재평가 영역.
_CONVERGE_TAIL_THRESHOLD = 0.8                          # 마지막 tail window 의 in_band 비율 임계값
_OSCILLATE_SIGN_CHANGE_THRESHOLD = 0.4                  # sign change / (n-1) 비율 임계값
_CONVERGE_TAIL_WINDOW = 5                               # tail window size


@dataclass
class Evaluator:
    """Structural evaluator — D1 증거 + D2 schema 골격. Standalone (D3 정합).

    Hook 등록 패턴 (Impl-9 driver 또는 test 가 wiring):
        dispatcher.on_dispatch(evaluator.record_dispatch)
        scheduler_core.on_admission_tick(evaluator.record_admission_tick)
    """

    config: Config
    clock: Clock
    idle_telemetry: IdleTelemetry                       # post-hoc snapshot 영역 (D1 hybrid)
    _dispatch_events: list[DispatchEvent] = field(default_factory=list)
    _admission_snapshots: list[AdmissionSnapshot] = field(default_factory=list)
    _pim_k_dt_accum: float = 0.0                        # Σ k_total · dt 누적 (pim_utilization 산출 위)
    _pim_last_dispatch_t: float | None = None
    _pim_last_k: int = 0

    # ------------------------------------------------------------------------
    # D1 hybrid — hook callback (dispatcher / SchedulerCore 에서 fire)
    # ------------------------------------------------------------------------

    def record_dispatch(self, event: DispatchEvent) -> None:
        """Dispatcher.on_dispatch 가 fire 하면 호출. Series 누적 + PIM utilization 입력."""
        self._dispatch_events.append(event)
        if event.resource == "PIM":
            # 직전 PIM dispatch 의 (k, dt) 누적 — 이번 dispatch 가 직전의 *완료 시점* 을 시사
            if self._pim_last_dispatch_t is not None:
                dt = event.timestamp - self._pim_last_dispatch_t
                self._pim_k_dt_accum += self._pim_last_k * dt
            self._pim_last_dispatch_t = event.timestamp
            self._pim_last_k = event.k_total

    def record_admission_tick(self, snapshot: AdmissionSnapshot) -> None:
        """SchedulerCore.on_admission_tick 가 fire 하면 호출. Series 누적."""
        self._admission_snapshots.append(snapshot)

    # ------------------------------------------------------------------------
    # 7 산출 method (PLAN §4 Impl-8 정합)
    # ------------------------------------------------------------------------

    def dispatch_trace(self) -> tuple[DispatchEvent, ...]:
        """§6.5 Init/T1~T5 sequence 의 event log. Tuple = immutable snapshot."""
        return tuple(self._dispatch_events)

    def admission_convergence(self) -> ConvergenceVerdict:
        """§6.4 deadband 수렴 trace 위 oscillation / 수렴 판정.

        Heuristic (구조 검증, 정량값 미산출):
        - in_band_fraction = (|a-b| ≤ deadband_width 인 snapshot 비율)
        - converged: 마지막 N snapshot 의 in_band 비율 ≥ _CONVERGE_TAIL_THRESHOLD (0.8)
        - oscillating: idle_fraction (a-b) sign change 빈도 ≥ _OSCILLATE_SIGN_CHANGE_THRESHOLD (0.4)

        둘 다 False = transient. 둘 다 True 는 정의상 불가능 (caller 가 해석).
        """
        if not self._admission_snapshots:
            return ConvergenceVerdict(False, False, 0.0, 0)
        in_band_count = 0
        sign_changes = 0
        prev_sign = 0
        for s in self._admission_snapshots:
            width = lookup_width(self.config.admission, s.ctx_tokens)
            if in_band(s.a_cycle - s.b_cycle, width):
                in_band_count += 1
            cur_sign = 1 if s.a_cycle > s.b_cycle else (-1 if s.a_cycle < s.b_cycle else 0)
            if prev_sign != 0 and cur_sign != 0 and cur_sign != prev_sign:
                sign_changes += 1
            if cur_sign != 0:
                prev_sign = cur_sign
        n = len(self._admission_snapshots)
        in_band_fraction = in_band_count / n
        tail_n = min(n, _CONVERGE_TAIL_WINDOW)
        tail_in_band = sum(
            1 for s in self._admission_snapshots[-tail_n:]
            if in_band(s.a_cycle - s.b_cycle, lookup_width(self.config.admission, s.ctx_tokens))
        )
        converged = (tail_in_band / tail_n) >= _CONVERGE_TAIL_THRESHOLD if tail_n > 0 else False
        oscillating = (sign_changes / max(1, n - 1)) >= _OSCILLATE_SIGN_CHANGE_THRESHOLD
        return ConvergenceVerdict(
            converged=converged,
            oscillating=oscillating,
            in_band_fraction=in_band_fraction,
            samples=n,
        )

    def idle_fraction(self) -> dict:
        """3-key idle fraction schema. Impl-10-pre-1 O8.1 정합.

        - `gpu_instance_a` / `pim_instance_a` — intra-A balance signal (ARCH §6.4)
        - `gpu_instance_b` — inter-AB balance 의 B-side substrate

        Instance B activity 의 production 측정 = `InstancePipeline.dispatch` chain.
        스케줄러 driver path 가 ForwardPass.run 미사용 시 gpu_instance_b 는 0 — schema lock-in 의미 (Stage 1).
        """
        return {
            "gpu_instance_a": self.idle_telemetry.gpu_idle_fraction(),
            "pim_instance_a": self.idle_telemetry.pim_idle_fraction(),
            "gpu_instance_b": self.idle_telemetry.gpu_instance_b_idle_fraction(),
        }

    def pim_utilization(self) -> float:
        """`Σ k_total · dt / (k_max · total_time)` aggregate channel-time utilization.

        산식:
        - k_max = config.admission.k_total_max (= 2048, ARCH §3.2 literal)
        - total_time = clock.now - dispatch_events[0].timestamp (PIM 활동 시작 이후 window)
        - Σ k · dt = self._pim_k_dt_accum (record_dispatch PIM 분기 누적)

        Returns 0.0 if no PIM dispatch or zero window.
        *O8.2 carry-over* — 마지막 PIM dispatch 의 completion 미반영 (R1).
        """
        if not self._dispatch_events:
            return 0.0
        total_time = self.clock.now - self._dispatch_events[0].timestamp
        if total_time <= 0:
            return 0.0
        k_max = self.config.admission.k_total_max
        return self._pim_k_dt_accum / (k_max * total_time)

    def pipeline_efficiency(self, a_cycle: float, b_cycle: float) -> float:
        """`max(A, B) / (A + B)` ratio. ARCH §3.4 inter-instance pipeline efficiency.

        Boundary:
        - A = B → 0.5 (perfect balance)
        - A >> B 또는 B >> A → 1.0 (한쪽 dominant — pipeline 무의미)
        - A ≤ 0 또는 B ≤ 0 → ValueError (산식 undefined)
        """
        if a_cycle <= 0 or b_cycle <= 0:
            raise ValueError(
                f"pipeline_efficiency requires positive cycles, got a={a_cycle}, b={b_cycle}"
            )
        return max(a_cycle, b_cycle) / (a_cycle + b_cycle)

    def acceleration_decomposition(
        self, a_cycle: float, b_cycle: float, t_pim: float, t_proj: float,
    ) -> tuple[DecompositionCell, ...]:
        """F1·F2·F3·F5 cycle ratio direction 표. D2 schema 골격.

        *정량 절대값 미산출 (PLAN §0.5 정합)* — 각 cell 의 ratio 의 *direction* (> 1.0) 만 검증.
        Workload regime 격자 위 정량값은 Impl-10 calibrated input.

        산식 (구조 검증, ARCH §5.7 직접 반영):
        - F1: cycle_with = max(t_proj, t_pim) — SP-PIM 활성화
              cycle_without = max(t_proj, t_pim_gpu_fallback) — GPU attention kernel route
        - F2: cycle_with = max(t_proj, t_pim) — double-buffering 활성화
              cycle_without = t_proj + t_pim — 직렬 강제
        - F3: cycle_with = max(a_cycle, b_cycle) — A/B inter-instance pipeline
              cycle_without = a_cycle + b_cycle — single-instance serial
        - F5: cycle_with = t_pim — channel-independent (caller 가 F5 활성화 path 의 t_pim 주입)
              cycle_without = t_pim_lockstep — lock-step max-KV (caller 가 산출 inject — 본 단계는
                              dummy placeholder 2× t_pim 사용. Impl-10 calibrated 영역에서 실측)

        F4 미포함 — ARCH §5.7 F4 정의 ("steady-state precondition for F2·F3, not a standalone
        contribution item") 정합.
        """
        cells: list[DecompositionCell] = []
        # F1 — SP-PIM
        t_pim_fallback = self.config.time.gpu_op_time_us.get("decode_attn_fallback", t_pim)
        cells.append(self._build_cell(
            AblationSource.F1,
            cycle_with=max(t_proj, t_pim),
            cycle_without=max(t_proj, t_pim_fallback),
        ))
        # F2 — double-buffering
        cells.append(self._build_cell(
            AblationSource.F2,
            cycle_with=max(t_proj, t_pim),
            cycle_without=t_proj + t_pim,
        ))
        # F3 — A/B inter-instance (Q7 — Evaluator 직접 산출, InstancePipeline 미터치)
        cells.append(self._build_cell(
            AblationSource.F3,
            cycle_with=max(a_cycle, b_cycle),
            cycle_without=a_cycle + b_cycle,
        ))
        # F5 — channel-independent. 본 단계는 dummy 2× placeholder (Impl-10 calibrated 영역).
        cells.append(self._build_cell(
            AblationSource.F5,
            cycle_with=t_pim,
            cycle_without=t_pim * 2.0,
        ))
        return tuple(cells)

    @staticmethod
    def _build_cell(
        source: AblationSource, cycle_with: float, cycle_without: float,
    ) -> DecompositionCell:
        if cycle_with <= 0:
            ratio = 0.0
            direction_positive = False
        else:
            ratio = cycle_without / cycle_with
            direction_positive = ratio > 1.0
        return DecompositionCell(
            source=source,
            cycle_with_source=cycle_with,
            cycle_without_source=cycle_without,
            ratio=ratio,
            direction_positive=direction_positive,
        )

    def report(
        self, a_cycle: float, b_cycle: float, t_pim: float, t_proj: float,
    ) -> dict:
        """Python dict + markdown 표. PULS 단독 산출 (Comparative baseline 없음).

        Q9 — dict 가 structured (test 검증), markdown 이 human-readable.
        PLAN §0.5 출처 라벨 동반 — markdown 에 ablation flag 명시.

        Returns dict with 8 keys:
            dispatch_trace · convergence · idle_fraction · pim_utilization ·
            pipeline_efficiency · acceleration_decomposition · markdown · ablation_config
        """
        decomp = self.acceleration_decomposition(a_cycle, b_cycle, t_pim, t_proj)
        result = {
            "dispatch_trace": self.dispatch_trace(),
            "convergence": self.admission_convergence(),
            "idle_fraction": self.idle_fraction(),
            "pim_utilization": self.pim_utilization(),
            "pipeline_efficiency": self.pipeline_efficiency(a_cycle, b_cycle),
            "acceleration_decomposition": decomp,
            "ablation_config": self.config.ablation,
        }
        result["markdown"] = self._format_markdown(result)
        return result

    @staticmethod
    def _format_markdown(result: dict) -> str:
        """Markdown 표 — ablation flag · regime label 동반. PLAN §0.5 출처 라벨링 정합."""
        ab = result["ablation_config"]
        lines = ["# PULS Structural Evaluator Report", ""]
        lines.append("## Ablation Flags (Provenance)")
        lines.append(f"- F1 (SP-PIM): {'**DISABLED**' if ab.f1_disabled else 'enabled'}")
        f2 = ab.f2_window_capacity_override if ab.f2_window_capacity_override else "default 3"
        lines.append(f"- F2 (Window cap): {f2}")
        lines.append(f"- F3 (A/B): {'**DISABLED**' if ab.f3_disabled else 'enabled'}")
        lines.append(f"- F5 (Channel-indep): {'**DISABLED**' if ab.f5_disabled else 'enabled'}")
        lines.append("")
        lines.append("## Idle Fraction")
        for k, v in result["idle_fraction"].items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
        lines.append(f"## PIM Utilization: {result['pim_utilization']:.4f}")
        lines.append(f"## Pipeline Efficiency: {result['pipeline_efficiency']:.4f}")
        lines.append("")
        lines.append("## Convergence")
        conv = result["convergence"]
        lines.append(f"- converged: {conv.converged}")
        lines.append(f"- oscillating: {conv.oscillating}")
        lines.append(f"- in_band_fraction: {conv.in_band_fraction:.4f}")
        lines.append(f"- samples: {conv.samples}")
        lines.append("")
        lines.append("## Acceleration Decomposition (Direction Only — Impl-10 calibrated 값)")
        lines.append("| Source | cycle_with | cycle_without | ratio | direction+ |")
        lines.append("|---|---|---|---|---|")
        for cell in result["acceleration_decomposition"]:
            lines.append(
                f"| {cell.source.value} | {cell.cycle_with_source:.4f} | "
                f"{cell.cycle_without_source:.4f} | {cell.ratio:.4f} | {cell.direction_positive} |"
            )
        return "\n".join(lines)
