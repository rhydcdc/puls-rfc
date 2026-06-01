from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from puls_sched.micro_batch import MicroBatch
    from puls_sched.node import NodeType


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_precision: str  # "FP8" | "FP16" — system-wide PIM regime selector (ARCH §3.1)
    ffn_intermediate: int = 28672  # Llama-3 70B FFN SwiGLU intermediate_size (Stage 2)


@dataclass(frozen=True)
class HWConfig:
    num_gpus_instance_a: int
    num_gpus_instance_b: int
    num_stacks_per_gpu: int
    num_channels_per_stack: int


@dataclass(frozen=True)
class TimeConfig:
    """Stage 2 — gpu_op_time_us fixed lookup 폐기 + per-op spec-derived formula 도입.

    `decode_attn_fallback` 만 reference 보존 (F1 ablation, Impl-11 deferred).
    `pim_tile_time_ns` = D5 calibrated 직접 (FP8 267 ns, FP16 512 ns reference).
    `rtl_fsm_cycle_per_tile` = 347 (D5-pre RTL closure).
    """

    pim_tile_time_ns: Mapping[str, float]
    nvlink_time_per_byte_ns: float
    rtl_fsm_cycle_per_tile: int
    rtl_fsm_tile_rows: int                       # ARCH §3.1 "32-row tile FSM" — RTL 합성 확정값
    pim_broadcast_latency_ns_cross_gpu: float    # SP-PIM cross-GPU lock-step broadcast overhead placeholder
    # Stage 2 — F1 ablation reference 만 보존 (Impl-11 deferred).
    # 다른 GPU op_time (qkv / prefill_attn / o_proj) 영역 = `compute_gpu_op_time_s` per-mb
    # spec-derived 산출 영역 영원. admission payload 의 t_proj 영역도 last mb 위 spec-derived 산출.
    gpu_op_time_us: Mapping[str, float] = field(default_factory=lambda: {
        "decode_attn_fallback": 4.0,    # F1 ablation reference (Impl-11)
    })


@dataclass(frozen=True)
class SLOConfig:
    ttft_target_ms: float
    tpot_target_ms: float


@dataclass(frozen=True)
class AdmissionConfig:
    n_sat: int
    kv_capacity_aggregate: int
    ctx_tier_short_max: int
    ctx_tier_mid_max: int
    deadband_width: Mapping[str, float]
    request_queue_capacity: int
    tick_interval_us: float = 10.0
    prefill_chunk_default: int = 256
    # Phase-2 former-v2 (OPERATING_POINT §1·§3, prefill 256 기본) — 동작점은 *두 타깃*으로
    # 정의되고 former 가 steering 으로 둘에 동시 수렴한다(개수+KV-work 동시 명중, 쪼개기 불가).
    #   decode: 개수 123 (decode_count_target) AND Σkv 12.3M (kv_operating_target_tokens)
    #   prefill: 256 토큰 (prefill_chunk_default) AND depth-합 25.6M (prefill_kv_work_target_tokens)
    # → PIM≈GPU-A≈B≈51µs (spread~1%, op-time 직접 산출 + proto_steering 검증).
    kv_operating_target_tokens: int = 12_300_000
    decode_count_target: int = 123
    # prefill steering 타깃: Σ(chunk_per_req × depth) — GPU-A PREFILL_ATTN=O(chunk×ctx) 균형.
    prefill_kv_work_target_tokens: int = 25_600_000
    # age-cap(공정성): wait ≥ age_cap 인 요청은 steering 무시하고 강제 포함 → starvation 0,
    # 대기 ≤ age_cap+1 batch. sweep 상 2 가 spread·공정·지연 sweet spot (OPERATING_POINT §3).
    age_cap: int = 2


@dataclass(frozen=True)
class AblationConfig:
    """F1~F5 ablation flag — Impl-10 sweep 의 single-flag isolation (D2 정합).

    Default 는 모든 flag off = F1~F5 모두 활성화 = 정상 PULS 동작 (backward-compat).
    """

    f1_disabled: bool = False
    f2_window_capacity_override: int | None = None
    f3_disabled: bool = False
    f5_disabled: bool = False


@dataclass(frozen=True)
class CalibrationConfig:
    """Stage 2 calibrated values. PLAN §0.5 출처 라벨 동반.

    Substrate framing (사용자 결정 정합):
    - GPU compute = DGX B200 standalone, FP16 dense 2,200 TFLOPS per GPU
    - Memory = HBM4 hypothetical projection (16 TB/s per GPU, 8 stack × 2 TB/s)
    - PIM tile = D5 RTL closure (FP8 = 267 ns compute-bound)
    - η_HBM = D1 산출 (0.74 fix + sweep {0.70, 0.74, 0.80})
    - MFU = conservative {0.5, 0.6, 0.7} sweep + report 위 0.6 fix
    - Trace = LongBench λ=3.40 첫 10 req
    - Aux1 = T4 microbench (별도 JSON ingest, 코드 산식 영역 영향 0)
    """

    # GPU compute (B200 standalone, FP16 dense peak)
    gpu_fp16_dense_peak_tflops: float = 2200.0
    gpu_fp8_dense_peak_tflops: float = 4500.0          # Impl-11 reference (F1 deferred)
    gpu_mfu_default: float = 0.6
    gpu_mfu_sweep: tuple[float, ...] = (0.5, 0.6, 0.7)

    # Memory (HBM4 hypothetical projection)
    hbm4_bw_per_stack_TBps: float = 2.0
    num_hbm_stacks_per_gpu: int = 8
    hbm_ext_bw_per_gpu_TBps: float = 16.0              # derived (= 2 × 8)
    hbm_ext_bw_aggregate_TBps: float = 128.0           # derived (= 16 × 8 GPU)
    eta_hbm_external: float = 0.74
    eta_hbm_external_sweep: tuple[float, ...] = (0.70, 0.74, 0.80)
    eta_hbm_internal: float = 1.0                      # ARCH §3.1 literal (bus 부재)

    # PIM tile (D5 산출 직접)
    pim_tile_time_fp8_ns_calibrated: float = 267.0
    pim_tile_time_fp16_ns_calibrated: float = 512.0    # reference only (kv_precision="FP8" 영원)
    rtl_fsm_cycle_per_tile_calibrated: int = 347
    rtl_clock_GHz: float = 1.3

    # Model spec derived (Llama-3 70B)
    weight_bytes_per_layer_fp16: int = 1_712_000_000
    kv_bytes_per_token_per_layer_fp8: int = 2048       # 2 (K,V) × 8 kv_heads × 128 head_dim × 1B
    kv_bytes_per_token_aggregate_fp8: int = 163_840    # × 80 layer

    # Trace
    trace_path: str = "data/longctx_longbench_lambda_3_40.csv"
    trace_first_n_requests: int = 10

    # Aux1 microbench (T4 측정 결과 ingest path — 산식 영역 영향 0)
    aux1_t4_ratio_path: str | None = "data/aux1_ratio_t4_measured.json"

    # Provenance label dict
    provenance: Mapping[str, str] = field(default_factory=lambda: {
        "gpu_fp16_dense_peak_tflops": "nvidia_blackwell_b200_chart_spec",
        "gpu_fp8_dense_peak_tflops": "nvidia_blackwell_b200_chart_spec_impl11_reference",
        "hbm_bw": "puls_substrate_hbm4_projection_hypothetical",
        "eta_hbm_external": "puls_d1_eta_hbm3_extended_to_hbm4_framing_A",
        "eta_hbm_internal": "arch_3_1_literal_bus_absent",
        "pim_tile_fp8": "puls_d5_ramulator_compute_bound_regime_A",
        "pim_tile_fp16": "puls_d5_ramulator_load_bound_reference",
        "rtl_clock": "puls_d5_pre_rtl_clock_1_3GHz_synth_closure",
        "rtl_fsm_cycle_per_tile": "puls_d5_pre_rtl_347cyc",
        "weight_bytes_per_layer": "llama3_70b_published_spec_derived",
        "kv_bytes": "llama3_70b_published_spec_derived",
        "ffn_intermediate": "llama3_70b_published_spec",
        "trace": "longbench_longctx_poisson_lambda_3_40_first_10",
        "mfu": "conservative_estimate_mfu_06_sensitivity_sweep_05_07",
        "aux1_microbench": "colab_t4_free_llama3_70b_ffn_single_layer_measured",
    })


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    hw: HWConfig
    time: TimeConfig
    slo: SLOConfig
    admission: AdmissionConfig
    ablation: AblationConfig
    calibration: CalibrationConfig
    seed: int


# ============================================================================
# Per-op spec-derived formula (Stage 2 — Stage 1 time abstraction 폐기)
# ============================================================================
#
# ARCH §3.5.2 *Computed Wait* literal 정합 + 사용자 의도 *"각 trace 위 정확
# 산출, 평균/압축/퉁치기 0"* 정합. 각 mb 위 정확 batch + ctx 산출.
#
# Stage 1 fixed lookup (`config.time.gpu_op_time_us[node_type]`) 폐기 — 단
# F1 ablation reference (`decode_attn_fallback`) 만 보존 (Impl-11 deferred).
# ============================================================================


def compute_gpu_op_time_s(
    node_type: "NodeType",
    mb: "MicroBatch",
    cal: CalibrationConfig,
    model: ModelConfig,
    time_cfg: TimeConfig | None = None,
    num_gpus: int = 1,
) -> float:
    """Per-mb spec-derived per-op time for DAG 4 node types (Instance A).

    ARCH §3.5.2 *Computed Wait* + 사용자 의도 *"각 trace 위 정확 산출"* 정합.

    Args:
        node_type: NodeType — QKV / PREFILL_ATTN / O_PROJ / DECODE_ATTN (F1 ablation)
        mb: MicroBatch — per-mb batch (decode + prefill) + per-req ctx 산출
        cal: CalibrationConfig — B200 peak × MFU (peak = *per-GPU*, provenance "2200 TFLOPS per GPU")
        model: ModelConfig — hidden, num_kv_heads, head_dim
        time_cfg: TimeConfig — DECODE_ATTN F1 ablation reference (Impl-11 영역)
        num_gpus: tensor-parallel GPU 수 (ARCH §3.4 TP=8). GEMM 이 num_gpus 장에 분산되어
            wall-clock = single-GPU FLOPs / num_gpus. PIM op_time 이 k_aggregate(=8 GPU 채널
            합산)로 이미 분산을 반영하는 것과 *단위 통일*. 기본 1 = 단위 test 의 산식 검증 보존.

    Returns:
        op_time_s (seconds, float)

    Raises:
        ValueError: unknown node_type
    """
    from puls_sched.node import NodeType as _NT

    # Phase-2 — TP=num_gpus 분산: GEMM wall-clock = single-GPU FLOPs / num_gpus.
    # PIM 은 k_aggregate(8-GPU 채널 합산)로 이미 분산 반영 → GPU 도 통일.
    peak_FLOPS = cal.gpu_fp16_dense_peak_tflops * 1e12 * cal.gpu_mfu_default * num_gpus
    batch_decode = len(mb.decode_tokens)
    # prefill_chunk = dict[req_id, list[int]] — list length = chunk size per req
    batch_prefill = sum(len(v) for v in mb.prefill_chunk.values()) if mb.prefill_chunk else 0
    batch_total = batch_decode + batch_prefill

    if node_type == _NT.QKV:
        # ARCH §6.1 — bulk GEMM over entire μ-batch
        # FLOPs = 2 × batch × hidden × (hidden + 2 × n_kv × d_head)
        flops = 2 * batch_total * model.hidden * (
            model.hidden + 2 * model.num_kv_heads * model.head_dim
        )
    elif node_type == _NT.O_PROJ:
        flops = 2 * batch_total * model.hidden * model.hidden
    elif node_type == _NT.PREFILL_ATTN:
        # FlashAttention causal — 각 req 위 prefill_processed + chunk 까지
        # FLOPs(req) = 2 × chunk_per_req × hidden × (prefill_processed + chunk_per_req)
        if not mb.prefill_chunk:
            return 0.0
        flops = 0
        for req_id, chunk_tokens in mb.prefill_chunk.items():
            chunk_per_req = len(chunk_tokens)
            ctx_so_far = mb.prefill_processed.get(req_id, 0) + chunk_per_req
            flops += 2 * chunk_per_req * model.hidden * ctx_so_far
    elif node_type == _NT.DECODE_ATTN:
        # F1 ablation 영역 only — PIM 활성 시 GPU 위 dispatch 0 (정상 PULS 동작).
        # F1 disabled 위 GPU fallback time — TimeConfig.gpu_op_time_us["decode_attn_fallback"]
        # reference (Impl-11 calibration 영역의 placeholder).
        if time_cfg is None:
            raise ValueError("DECODE_ATTN op_time requires time_cfg (F1 ablation reference)")
        return time_cfg.gpu_op_time_us.get("decode_attn_fallback", 4.0) * 1e-6
    else:
        raise ValueError(f"Unknown node_type for spec-derived op_time: {node_type}")

    return flops / peak_FLOPS


def compute_ffn_op_time_s(
    mb: "MicroBatch",
    cal: CalibrationConfig,
    model: ModelConfig,
    num_gpus: int = 1,
) -> float:
    """Instance B FFN op_time (α path Carry-1 해소).

    ARCH §3.4 *Instance B FFN block* — SwiGLU FFN 3 GEMM (gate + up + down).
    DAG 노드 영역 아님 — Instance B substrate 만, NodeType enum 외.

    FLOPs = 6 × batch × hidden × intermediate (Llama-3 70B: intermediate=28672)

    num_gpus: Instance B tensor-parallel GPU 수 (ARCH §3.4, 기본 8). FFN GEMM 이 num_gpus
        장에 분산 → wall-clock = single-GPU FLOPs / num_gpus. compute_gpu_op_time_s 와 동일.
    """
    # Phase-2 — TP=num_gpus 분산 (compute_gpu_op_time_s 와 단위 통일).
    peak_FLOPS = cal.gpu_fp16_dense_peak_tflops * 1e12 * cal.gpu_mfu_default * num_gpus
    batch_decode = len(mb.decode_tokens)
    batch_prefill = sum(len(v) for v in mb.prefill_chunk.values()) if mb.prefill_chunk else 0
    batch_total = batch_decode + batch_prefill
    flops = 6 * batch_total * model.hidden * model.ffn_intermediate
    return flops / peak_FLOPS


def default_dummy_config() -> Config:
    """RFC target = Llama-3 70B class. Stage 2 calibrated values 주입.

    `default_dummy_config` 이름 보존 (backward-compat — fixture 광범위 사용). 단
    Stage 2 이후 *값은 calibrated 영역* — Stage 1 의미 (dummy ratio) 영원 아님.
    """
    return Config(
        model=ModelConfig(
            num_layers=80,
            hidden=8192,
            num_heads=64,
            num_kv_heads=8,
            head_dim=128,
            kv_precision="FP8",
            ffn_intermediate=28672,
        ),
        hw=HWConfig(
            num_gpus_instance_a=8,
            num_gpus_instance_b=8,
            num_stacks_per_gpu=8,
            num_channels_per_stack=32,
        ),
        time=TimeConfig(
            pim_tile_time_ns={"FP8": 267.0, "FP16": 512.0},   # D5 calibrated 직접
            nvlink_time_per_byte_ns=1.0,                       # 보존 (사용자 결정 — NVLink 안 씀)
            rtl_fsm_cycle_per_tile=347,                        # D5-pre RTL closure
            rtl_fsm_tile_rows=32,                              # ARCH §3.1 RTL 합성 확정
            pim_broadcast_latency_ns_cross_gpu=0.5,            # 보존 (산식 위 미세, NVLink 영역)
            gpu_op_time_us={"decode_attn_fallback": 4.0},      # F1 ablation reference (Impl-11) 만
        ),
        slo=SLOConfig(ttft_target_ms=100.0, tpot_target_ms=10.0),
        admission=AdmissionConfig(
            n_sat=16,
            # Phase-2 former-v2 (prefill 256 기준) — 총 30M aggregate (= 배치당 12.3M×2슬롯+여유).
            # main_loop._per_mb_kv_budget = 30M / _STAGGERING_TARGET_MB(2) = 15M per μ-batch
            # → decode 동작점 12.3M 이 배치당 15M 천장 안에 안착. A∥B(F3) 오버랩 위해 2 배치
            # 동시 in-flight(KV 영구 상주) → 총 30M. HW = 80GB/stack·5TB (256 기준, OPERATING_POINT §1).
            kv_capacity_aggregate=30_000_000,
            ctx_tier_short_max=8_000,
            ctx_tier_mid_max=32_000,
            deadband_width={"short": 1.0, "mid": 2.0, "long": 3.0},
            request_queue_capacity=1024,
        ),
        ablation=AblationConfig(),
        calibration=CalibrationConfig(),
        seed=42,
    )
