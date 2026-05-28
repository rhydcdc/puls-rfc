"""Structural Evaluator — D1 증거 산출 + D2 schema 골격.

PLAN.md §4 Impl-8 + ARCH §6.7 정합. *절대 metric 미산출 (TTFT · TPOT · throughput · goodput).
Comparative baseline 미산출 (Sarathi · vLLM).*

7 산출 method + 2 callback method = 9 method total:
- record_dispatch / record_admission_tick — D1 hook callback (dispatcher / main_loop 가 fire)
- dispatch_trace — §6.5 Init/T1~T5 sequence event log
- admission_convergence — §6.4 deadband 위 idle fraction 시간 series 위 oscillation / 수렴 판정
- idle_fraction — Instance A scope (GPU · PIM 2 자원, per-instance A/B split 은 Impl-9 — O8.1)
- pim_utilization — PIM busy time fraction (Impl-10-pre-2: k_total knob 폐기 위 단순 time fraction)
- pipeline_efficiency — max(A, B) / (A + B) ratio
- acceleration_decomposition — F1·F2·F3·F5 cycle ratio direction 표 (D2 schema 골격, F4 미포함)
- report — Python dict + markdown 표 (PULS 단독, Comparative baseline 없음)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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
    Impl-10-pre-2 — k_total field 폐기 (sequence-parallel PIM 위 channel knob 제거, k 영원 k_max).
    """

    timestamp: float
    micro_batch_id: int
    node_type: NodeType
    resource: str                                       # "GPU" | "PIM"
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


class AuxSource(Enum):
    """Aux source ID — ARCH §5.7 정합 (Auxiliary). Stage 2 D2 deliverable.

    Aux1 = mixed batching weight reuse (arithmetic intensity 상승)
    Aux2 = bus traffic reduction (KV cache HBM↔GPU 절감, long-ctx dominant)
    """

    AUX1 = "AUX1_MIXED_BATCH_WEIGHT_REUSE"
    AUX2 = "AUX2_BUS_TRAFFIC_REDUCTION"


@dataclass(frozen=True)
class AuxCell:
    """Aux saving cell — closed-form 산출 + provenance label.

    Stage 2 D2 (PLAN §4 Impl-10 main) — saving_per_layer ns + 80 layer aggregate.
    Source: B200 + HBM4 projection + Llama-3 70B spec + η_HBM (D1) + PIM tile (D5).
    """

    source: AuxSource
    saving_per_layer_ns: float
    saving_total_80_layer_ns: float
    provenance_label: str


@dataclass(frozen=True)
class F3ClosedFormResult:
    """F3 closed-form ratio — α path 위 closed-form 산출.

    a_cycle = max(t_proj, t_attn) — Instance A 내부 double-buffering (§5.6)
    b_cycle = t_FFN — Instance B (α path Carry-1 해소 후)
    F3 = max(a, b) / (a + b)
    """

    ctx_tokens: int
    prefill_chunk: int
    t_qkv_us: float
    t_prefill_attn_us: float
    t_o_proj_us: float
    t_proj_us: float
    t_attn_us: float
    a_cycle_us: float
    b_cycle_us: float
    f3_ratio: float
    provenance: str = "f3_closed_form_alpha_path"


@dataclass(frozen=True)
class F5TraceGroundedResult:
    """F5 trace-grounded ratio — λ=3.40 첫 10 req KV 분포 위.

    ratio = ceil(sum_KV / (k × tile_rows)) / ceil(max_KV × N / (k × tile_rows))
    """

    n_requests: int
    sum_kv: int
    max_kv: int
    tiles_channel_independent: int
    tiles_lockstep_max_kv: int
    f5_ratio: float
    provenance: str = "f5_trace_grounded_longbench_longctx_poisson"


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
    _pim_busy_accum: float = 0.0                        # Σ pim_busy_dt 누적 (pim_utilization 산출 위)
    _pim_last_dispatch_t: float | None = None

    # ------------------------------------------------------------------------
    # D1 hybrid — hook callback (dispatcher / SchedulerCore 에서 fire)
    # ------------------------------------------------------------------------

    def record_dispatch(self, event: DispatchEvent) -> None:
        """Dispatcher.on_dispatch 가 fire 하면 호출. Series 누적 + PIM utilization 입력.

        Impl-10-pre-2 — k_total knob 폐기 위 PIM utilization 산식 단순화 (k_max · time → time fraction).
        """
        self._dispatch_events.append(event)
        if event.resource == "PIM":
            if self._pim_last_dispatch_t is not None:
                self._pim_busy_accum += event.timestamp - self._pim_last_dispatch_t
            self._pim_last_dispatch_t = event.timestamp

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
        """PIM busy time fraction over window since first dispatch.

        Impl-10-pre-2 — k_total knob 폐기 위 산식 단순화: 임의 시점 PIM 은 모든 채널 점유
        (sequence-parallel), 따라서 channel-weighted 영역 시간 fraction 으로 환원.

        Returns 0.0 if no PIM dispatch or zero window.
        *O8.2 carry-over* — 마지막 PIM dispatch 의 completion 미반영 (R1).
        """
        if not self._dispatch_events:
            return 0.0
        total_time = self.clock.now - self._dispatch_events[0].timestamp
        if total_time <= 0:
            return 0.0
        return self._pim_busy_accum / total_time

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

    # ========================================================================
    # Stage 2 D2 deliverable — Aux1·Aux2·F3·F5 closed-form + cross-validate
    # ========================================================================

    def aux1_mixed_batch_weight_reuse(self) -> AuxCell:
        """Aux1 closed-form — mixed batch weight streaming 1 회 절감.

        saving_per_layer = weight_bytes_per_layer / (HBM_ext_BW × η_HBM_ext)
            ↑ separate prefill+decode 위 2 회 weight streaming → mixed 위 1 회

        Provenance: llama3_70b_published_spec + puls_substrate_hbm4_projection +
        puls_d1_eta_hbm3_extended_to_hbm4_framing_A.
        """
        cal = self.config.calibration
        bw_Bps = cal.hbm_ext_bw_aggregate_TBps * 1e12 * cal.eta_hbm_external
        saving_s = cal.weight_bytes_per_layer_fp16 / bw_Bps
        saving_per_layer_ns = saving_s * 1e9
        return AuxCell(
            source=AuxSource.AUX1,
            saving_per_layer_ns=saving_per_layer_ns,
            saving_total_80_layer_ns=saving_per_layer_ns * self.config.model.num_layers,
            provenance_label=cal.provenance.get("weight_bytes_per_layer", "")
            + " + " + cal.provenance.get("hbm_bw", "")
            + " + " + cal.provenance.get("eta_hbm_external", ""),
        )

    def aux2_bus_traffic_reduction(self, kv_length: int) -> AuxCell:
        """Aux2 closed-form — KV cache HBM↔GPU 전송 절감 (long-ctx dominant).

        Without PIM: time_KV = bytes_KV / (HBM_ext_BW × η_ext)
        With PIM:
            time_PIM = num_tiles × pim_tile_time_fp8 × num_layers
            time_result = bytes_result / (HBM_ext_BW × η_ext)
        saving = time_KV − (time_PIM + time_result)

        Args:
            kv_length: per-req KV token 수 (trace 위 산출)
        """
        cal = self.config.calibration
        hw = self.config.hw
        model = self.config.model
        bw_ext_Bps = cal.hbm_ext_bw_aggregate_TBps * 1e12 * cal.eta_hbm_external

        # Without PIM — GPU decode-attn 위 KV bytes 외부 bus 읽기
        bytes_KV = kv_length * cal.kv_bytes_per_token_aggregate_fp8
        time_KV_s = bytes_KV / bw_ext_Bps

        # With PIM — internal 영역 (compute-bound) + result 외부 bus
        k_aggregate = hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack
        tile_rows = self.config.time.rtl_fsm_tile_rows
        num_tiles = math.ceil(kv_length / (k_aggregate * tile_rows)) if kv_length > 0 else 0
        time_PIM_s = num_tiles * cal.pim_tile_time_fp8_ns_calibrated * 1e-9 * model.num_layers

        # Result bytes (PIM → HBM → GPU read) — [batch=1 × hidden] × FP16 × L
        bytes_result = model.hidden * 2 * model.num_layers
        time_result_s = bytes_result / bw_ext_Bps

        saving_s = time_KV_s - (time_PIM_s + time_result_s)
        saving_total_ns = saving_s * 1e9
        return AuxCell(
            source=AuxSource.AUX2,
            saving_per_layer_ns=saving_total_ns / model.num_layers,
            saving_total_80_layer_ns=saving_total_ns,
            provenance_label=cal.provenance.get("kv_bytes", "")
            + " + " + cal.provenance.get("trace", "")
            + " + " + cal.provenance.get("pim_tile_fp8", "")
            + " + " + cal.provenance.get("eta_hbm_external", ""),
        )

    def f3_closed_form(
        self, ctx_tokens: int, prefill_chunk: int = 512, batch: int = 1,
    ) -> F3ClosedFormResult:
        """F3 closed-form ratio — α path 정합.

        a_cycle = max(t_proj, t_attn)
            t_proj = t_QKV + t_PREFILL_ATTN + t_O_PROJ (GPU compute-bound spec-derived)
            t_attn = num_tiles × pim_tile_fp8 (PIM, D5 calibrated)
        b_cycle = t_FFN (Instance B compute-bound spec-derived)

        F3 = max(a, b) / (a + b)

        batch invariance — batch=1 위 산출 후 ratio 보존 (batch linear 위 분자분모 상쇄).
        ctx sweep 위 transition — short-ctx GPU-bound → long-ctx PIM-bound.
        """
        cal = self.config.calibration
        model = self.config.model
        peak_FLOPS = cal.gpu_fp16_dense_peak_tflops * 1e12 * cal.gpu_mfu_default

        # GPU compute (spec-derived per-op, seconds)
        flops_QKV = 2 * batch * model.hidden * (
            model.hidden + 2 * model.num_kv_heads * model.head_dim
        )
        flops_PREFILL_ATTN = 2 * prefill_chunk * model.hidden * ctx_tokens
        flops_O_PROJ = 2 * batch * model.hidden * model.hidden
        flops_FFN = 6 * batch * model.hidden * model.ffn_intermediate

        t_qkv_us = flops_QKV / peak_FLOPS * 1e6
        t_prefill_attn_us = flops_PREFILL_ATTN / peak_FLOPS * 1e6
        t_o_proj_us = flops_O_PROJ / peak_FLOPS * 1e6
        t_proj_us = t_qkv_us + t_prefill_attn_us + t_o_proj_us

        # PIM attention (D5 calibrated, ns → µs)
        hw = self.config.hw
        k_aggregate = hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack
        tile_rows = self.config.time.rtl_fsm_tile_rows
        num_tiles = math.ceil(batch * ctx_tokens / (k_aggregate * tile_rows)) if ctx_tokens > 0 else 0
        t_attn_us = num_tiles * cal.pim_tile_time_fp8_ns_calibrated * 1e-3

        a_cycle_us = max(t_proj_us, t_attn_us)
        b_cycle_us = flops_FFN / peak_FLOPS * 1e6
        total = a_cycle_us + b_cycle_us
        ratio = max(a_cycle_us, b_cycle_us) / total if total > 0 else 0.0

        return F3ClosedFormResult(
            ctx_tokens=ctx_tokens,
            prefill_chunk=prefill_chunk,
            t_qkv_us=t_qkv_us,
            t_prefill_attn_us=t_prefill_attn_us,
            t_o_proj_us=t_o_proj_us,
            t_proj_us=t_proj_us,
            t_attn_us=t_attn_us,
            a_cycle_us=a_cycle_us,
            b_cycle_us=b_cycle_us,
            f3_ratio=ratio,
        )

    def f5_trace_grounded(self, kv_lengths: list[int]) -> F5TraceGroundedResult:
        """F5 trace-grounded — channel-independent vs lock-step max-KV.

        channel-independent: tiles = ceil(sum_KV / (k × tile_rows))
        lock-step max-KV:    tiles = ceil(max_KV × N / (k × tile_rows))
        ratio = tiles_indep / tiles_lockstep (≤ 1.0 — penalty)
        """
        hw = self.config.hw
        k_aggregate = hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack
        tile_rows = self.config.time.rtl_fsm_tile_rows
        denom = k_aggregate * tile_rows
        if not kv_lengths:
            return F5TraceGroundedResult(0, 0, 0, 0, 0, 0.0)
        sum_kv = sum(kv_lengths)
        max_kv = max(kv_lengths)
        n = len(kv_lengths)
        tiles_indep = math.ceil(sum_kv / denom) if sum_kv > 0 else 0
        tiles_lockstep = math.ceil(max_kv * n / denom) if max_kv * n > 0 else 0
        ratio = tiles_indep / tiles_lockstep if tiles_lockstep > 0 else 0.0
        return F5TraceGroundedResult(
            n_requests=n,
            sum_kv=sum_kv,
            max_kv=max_kv,
            tiles_channel_independent=tiles_indep,
            tiles_lockstep_max_kv=tiles_lockstep,
            f5_ratio=ratio,
        )

    def f3_cross_validate(
        self, ctx_tokens: int, prefill_chunk: int = 512,
    ) -> dict:
        """F3 closed-form vs run.loop measured cross-validate (α path 정합).

        closed-form = self.f3_closed_form(ctx, chunk)
        measured = idle_telemetry 위 active duration delta 산출 (a_cycle = gpu_a + pim_a, b_cycle = gpu_b)
        """
        closed = self.f3_closed_form(ctx_tokens, prefill_chunk)
        a_measured = (
            self.idle_telemetry.active_duration("gpu_instance_a")
            + self.idle_telemetry.active_duration("pim_instance_a")
        )
        b_measured = self.idle_telemetry.active_duration("gpu_instance_b")
        total_measured = a_measured + b_measured
        ratio_measured = max(a_measured, b_measured) / total_measured if total_measured > 0 else 0.0
        return {
            "closed_form_ratio": closed.f3_ratio,
            "closed_form_a_cycle_us": closed.a_cycle_us,
            "closed_form_b_cycle_us": closed.b_cycle_us,
            "measured_ratio": ratio_measured,
            "measured_a_cycle_us": a_measured,
            "measured_b_cycle_us": b_measured,
            "abs_diff_ratio": abs(closed.f3_ratio - ratio_measured),
            "provenance": "f3_closed_form_alpha_path + run_loop_measured_b_cycle",
        }

    def d2_report(
        self, kv_lengths: list[int], ctx_sweep: list[int] | None = None,
        aux1_t4_ratio_path: str | Path | None = None,
    ) -> dict:
        """D2 통합 산출 — 4 source closed-form + T4 microbench ingest + markdown.

        Args:
            kv_lengths: trace 첫 N req kv_length 목록 (Aux2 per-req + F5 입력)
            ctx_sweep: F3 ctx sweep 목록 (default [2k, 32k, 128k, 1M])
            aux1_t4_ratio_path: Colab T4 microbench JSON path (optional ingest)
        """
        if ctx_sweep is None:
            ctx_sweep = [2_000, 32_000, 128_000, 1_000_000]
        cal = self.config.calibration

        aux1 = self.aux1_mixed_batch_weight_reuse()
        aux2_per_req = [
            self.aux2_bus_traffic_reduction(kv) for kv in kv_lengths
        ]
        f3_by_ctx = [self.f3_closed_form(ctx) for ctx in ctx_sweep]
        f5 = self.f5_trace_grounded(kv_lengths)

        # η_HBM sensitivity sweep
        eta_sweep = {}
        for eta in cal.eta_hbm_external_sweep:
            # Snapshot calibration with different η
            import dataclasses as dc
            cal_eta = dc.replace(cal, eta_hbm_external=eta)
            cfg_eta = dc.replace(self.config, calibration=cal_eta)
            ev_eta = Evaluator(config=cfg_eta, clock=self.clock, idle_telemetry=self.idle_telemetry)
            eta_sweep[eta] = {
                "aux1_saving_total_ns": ev_eta.aux1_mixed_batch_weight_reuse().saving_total_80_layer_ns,
                "aux2_avg_saving_total_ns": sum(
                    ev_eta.aux2_bus_traffic_reduction(kv).saving_total_80_layer_ns
                    for kv in kv_lengths
                ) / len(kv_lengths) if kv_lengths else 0.0,
            }

        # T4 microbench ingest (optional, disclosure only)
        aux1_t4 = None
        path = aux1_t4_ratio_path or cal.aux1_t4_ratio_path
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = Path(__file__).parent.parent.parent / p
            if p.exists():
                import json
                aux1_t4 = json.loads(p.read_text())

        result = {
            "aux1": aux1,
            "aux2_per_req": aux2_per_req,
            "f3_by_ctx": f3_by_ctx,
            "f5": f5,
            "eta_sweep": eta_sweep,
            "aux1_t4_microbench": aux1_t4,
            "provenance": dict(cal.provenance),
        }
        result["markdown"] = self._format_d2_markdown(result)
        return result

    @staticmethod
    def _format_d2_markdown(d2: dict) -> str:
        """D2 통합 markdown — 4 source + provenance + honest disclosure."""
        lines = ["# D2 Calibrated Projection Report (Stage 2)", ""]
        lines.append("## Substrate Provenance")
        for k, v in d2["provenance"].items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

        # Aux1
        a1 = d2["aux1"]
        lines.append("## Aux1 — Mixed Batching Weight Reuse")
        lines.append(f"- saving_per_layer = {a1.saving_per_layer_ns:.2f} ns")
        lines.append(f"- saving_total (80 layer) = {a1.saving_total_80_layer_ns:.2f} ns = "
                     f"{a1.saving_total_80_layer_ns / 1000:.2f} µs")
        lines.append(f"- provenance: {a1.provenance_label}")
        lines.append("")

        # Aux2
        lines.append("## Aux2 — Bus Traffic Reduction (per-req kv_length)")
        lines.append("| req | saving_total_ns | saving_total_us |")
        lines.append("|---|---|---|")
        for i, c in enumerate(d2["aux2_per_req"]):
            lines.append(f"| {i} | {c.saving_total_80_layer_ns:.2f} | "
                         f"{c.saving_total_80_layer_ns / 1000:.2f} |")
        lines.append("")

        # F3
        lines.append("## F3 — Inter-instance Pipeline Ratio (closed-form, α path)")
        lines.append("| ctx | t_proj_us | t_attn_us | a_cycle_us | b_cycle_us | F3 ratio |")
        lines.append("|---|---|---|---|---|---|")
        for r in d2["f3_by_ctx"]:
            lines.append(f"| {r.ctx_tokens} | {r.t_proj_us:.2f} | {r.t_attn_us:.2f} | "
                         f"{r.a_cycle_us:.2f} | {r.b_cycle_us:.2f} | {r.f3_ratio:.4f} |")
        lines.append("")

        # F5
        f5 = d2["f5"]
        lines.append("## F5 — Channel-independent vs Lock-step max-KV")
        lines.append(f"- n_reqs = {f5.n_requests}, sum_KV = {f5.sum_kv}, max_KV = {f5.max_kv}")
        lines.append(f"- tiles_indep = {f5.tiles_channel_independent}, "
                     f"tiles_lockstep = {f5.tiles_lockstep_max_kv}")
        lines.append(f"- F5 ratio = {f5.f5_ratio:.4f} "
                     f"(< 1.0 = channel-independent 가 lock-step 보다 빠름)")
        lines.append("")

        # η sweep
        lines.append("## η_HBM Sensitivity Sweep")
        lines.append("| η | Aux1 saving (ns) | Aux2 avg saving (ns) |")
        lines.append("|---|---|---|")
        for eta, v in d2["eta_sweep"].items():
            lines.append(f"| {eta} | {v['aux1_saving_total_ns']:.2f} | "
                         f"{v['aux2_avg_saving_total_ns']:.2f} |")
        lines.append("")

        # T4 microbench
        if d2["aux1_t4_microbench"]:
            lines.append("## Aux1 Empirical Anchor — T4 Microbench")
            lines.append(f"- provenance: `{d2['aux1_t4_microbench']['metadata']['provenance_label']}`")
            lines.append("| batch (k → 2k) | t_separate (ms) | t_mixed (ms) | ratio |")
            lines.append("|---|---|---|---|")
            for k, v in d2["aux1_t4_microbench"].get("aux1_ratio", {}).items():
                if isinstance(v, dict) and "ratio_separate_over_mixed" in v:
                    lines.append(f"| {k} | {v['t_separate_ms']:.4f} | "
                                 f"{v['t_mixed_ms']:.4f} | {v['ratio_separate_over_mixed']:.4f} |")
            lines.append("")
            lines.append("*Cross-validation disclosure — T4 위 architectural property (mixed batch "
                         "weight reuse 가 memory-bound regime 위 dominant). B200 위 ingest 0 — "
                         "T4 ≠ B200 substrate mismatch 정합.*")
            lines.append("")

        # Honest disclosure
        lines.append("## Honest Disclosure")
        lines.append("- **F1 (SP-PIM vs GPU attention) — Impl-11 deferred**: Blackwell GPU attention "
                     "kernel calibration 영역. Stage 2 위 reference placeholder `decode_attn_fallback` 보존.")
        lines.append("- **F2 (double-buffering 정량) — Impl-11 deferred**: calibrated cycle 위 "
                     "isolated ratio.")
        lines.append("- **Memory = HBM4 hypothetical projection**: 현재 production 부재, "
                     "ARCH §3.1 *HBM4 logic die* literal 정합.")
        lines.append("- **η_HBM_ext = Framing A**: D1 H100 (HBM3) 측정값 → HBM4 영역 extension. "
                     "직접 measurement 부재.")
        lines.append("- **NVLink 산식 위 등장 0**: ARCH §3.4 *async hidden* + §3.5.3 *result via HBM*.")
        lines.append("- **FP16 PIM tile = reference**: ARCH §6.6 regime A 확정 evidence. "
                     "production 위 `kv_precision = FP8` 영원 (산식 위 미사용).")
        lines.append("- **`compute_*_op_time_s` per-mb spec-derived**: ARCH §3.5.2 Computed Wait + "
                     "사용자 의도 *'각 trace 위 정확 산출'* 정합. 평균/dummy 0.")
        return "\n".join(lines)

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
