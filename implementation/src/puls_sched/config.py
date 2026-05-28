from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_precision: str  # "FP8" | "FP16" — system-wide PIM regime selector (ARCH §3.1)


@dataclass(frozen=True)
class HWConfig:
    num_gpus_instance_a: int
    num_gpus_instance_b: int
    num_stacks_per_gpu: int
    num_channels_per_stack: int


@dataclass(frozen=True)
class TimeConfig:
    gpu_op_time_us: Mapping[str, float]
    pim_tile_time_ns: Mapping[str, float]
    nvlink_time_per_byte_ns: float
    rtl_fsm_cycle_per_tile: int
    rtl_fsm_tile_rows: int                       # ARCH §3.1 "32-row tile FSM" — RTL 합성 확정값 (PLAN §0.5 예외)
    pim_broadcast_latency_ns_cross_gpu: float    # SP-PIM cross-GPU lock-step broadcast overhead placeholder
    gpu_op_time_per_token_us: float = 0.01       # Impl-10-pre-2 — PREFILL_ATTN chunk-scaled op_time placeholder. Stage 2 calibrated


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
    idle_theta_low: float
    idle_theta_high: float
    request_queue_capacity: int
    tick_interval_us: float = 10.0           # Impl-9 Q1 — ADMISSION_TICK self-rescheduling cadence placeholder. PLAN §0.5 dummy
    prefill_chunk_default: int = 512         # Impl-10-pre-2 — Hybrid Chunk Size Policy base (Sarathi/vLLM 시중 표준). Cold start fallback
    pim_slack_safety_margin: float = 0.9     # Impl-10-pre-2 (B option) — PIM-GPU TSV BW contention 위 10% conservative margin (ARCH §3.5.3)


@dataclass(frozen=True)
class AblationConfig:
    """F1~F5 ablation flag — Impl-10 sweep 의 single-flag isolation 영역 (D2 정합).

    Default 는 모든 flag off = F1~F5 모두 활성화 = 정상 PULS 동작 (backward-compat).
    PLAN §0.5 — 본 영역의 코드 로직은 flag 의 *분기* 만, *정량 ratio 값* 산출 0.
    """

    f1_disabled: bool = False                       # SP-PIM 비활성화 — dispatcher 의 PIM op_time → GPU fallback
    f2_window_capacity_override: int | None = None  # double-buffering 비활성화 — InFlightWindow CAPACITY override (1 강제 시 직렬)
    f3_disabled: bool = False                       # Instance A/B 비활성화 — Evaluator 가 single_instance_cycle = a + b 산출
    f5_disabled: bool = False                       # channel-independent 비활성화 — PIMExecutor 의 lock-step max-KV penalty 분기


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    hw: HWConfig
    time: TimeConfig
    slo: SLOConfig
    admission: AdmissionConfig
    ablation: AblationConfig
    seed: int


def default_dummy_config() -> Config:
    """RFC target model = Llama-3 70B class (long-ctx 1M motivation 정합).

    PLAN.md §0.5 Numeric Value Policy — model spec 은 logic 검증 영역에서
    자료구조 차원 placeholder. Time field (gpu_op_time_us · pim_tile_time_ns
    · nvlink_time_per_byte_ns) 의 dummy 값은 ratio property 만 보존, 절대값 무의미.
    Impl-10 / Phase 3 에서 실측 / 추정값으로 교체.
    """
    return Config(
        model=ModelConfig(
            num_layers=80,
            hidden=8192,
            num_heads=64,
            num_kv_heads=8,
            head_dim=128,
            kv_precision="FP8",
        ),
        hw=HWConfig(
            num_gpus_instance_a=8,
            num_gpus_instance_b=8,
            num_stacks_per_gpu=8,
            num_channels_per_stack=32,
        ),
        time=TimeConfig(
            gpu_op_time_us={
                "qkv": 1.0,
                "prefill_attn": 1.0,
                "o_proj": 1.0,
                "decode_attn_fallback": 4.0,    # Impl-8 — F1 ablation 시 PIM → GPU fallback placeholder. dummy ratio (Impl-10 calibrated)
            },
            pim_tile_time_ns={"FP8": 1.0, "FP16": 2.0},
            nvlink_time_per_byte_ns=1.0,
            rtl_fsm_cycle_per_tile=1,
            rtl_fsm_tile_rows=32,
            pim_broadcast_latency_ns_cross_gpu=0.5,
        ),
        slo=SLOConfig(ttft_target_ms=100.0, tpot_target_ms=10.0),
        admission=AdmissionConfig(
            n_sat=16,
            kv_capacity_aggregate=4_000_000,   # Llama-3 70B FP8 × 8 H100 (640GB HBM / 0.16MB·token) ≈ 4M token placeholder. PLAN §0.5 ratio 정합.
            ctx_tier_short_max=8_000,
            ctx_tier_mid_max=32_000,
            deadband_width={"short": 1.0, "mid": 2.0, "long": 3.0},
            idle_theta_low=0.1,
            idle_theta_high=0.3,
            request_queue_capacity=1024,
        ),
        ablation=AblationConfig(),
        seed=42,
    )
