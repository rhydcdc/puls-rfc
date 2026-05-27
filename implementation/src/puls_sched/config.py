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
    k_total_step: int
    k_total_max: int


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    hw: HWConfig
    time: TimeConfig
    slo: SLOConfig
    admission: AdmissionConfig
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
            gpu_op_time_us={"qkv": 1.0, "prefill_attn": 1.0, "o_proj": 1.0},
            pim_tile_time_ns={"FP8": 1.0, "FP16": 2.0},
            nvlink_time_per_byte_ns=1.0,
            rtl_fsm_cycle_per_tile=1,
            rtl_fsm_tile_rows=32,
            pim_broadcast_latency_ns_cross_gpu=0.5,
        ),
        slo=SLOConfig(ttft_target_ms=100.0, tpot_target_ms=10.0),
        admission=AdmissionConfig(
            n_sat=16,
            kv_capacity_aggregate=1_000_000,
            ctx_tier_short_max=8_000,
            ctx_tier_mid_max=32_000,
            deadband_width={"short": 1.0, "mid": 2.0, "long": 3.0},
            idle_theta_low=0.1,
            idle_theta_high=0.3,
            request_queue_capacity=1024,
            k_total_step=256,
            k_total_max=2048,
        ),
        seed=42,
    )
