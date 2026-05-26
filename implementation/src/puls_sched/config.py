from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


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


@dataclass(frozen=True)
class SLOConfig:
    ttft_target_ms: float
    tpot_target_ms: float


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    hw: HWConfig
    time: TimeConfig
    slo: SLOConfig
    seed: int


def default_dummy_config() -> Config:
    return Config(
        model=ModelConfig(
            num_layers=32,
            hidden=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
        ),
        hw=HWConfig(
            num_gpus_instance_a=8,
            num_gpus_instance_b=8,
            num_stacks_per_gpu=8,
            num_channels_per_stack=32,
        ),
        time=TimeConfig(
            gpu_op_time_us={"qkv": 1.0, "prefill_attn": 1.0, "decode_attn": 1.0, "o_proj": 1.0},
            pim_tile_time_ns={"FP8": 1.0, "FP16": 2.0},
            nvlink_time_per_byte_ns=1.0,
            rtl_fsm_cycle_per_tile=1,
        ),
        slo=SLOConfig(ttft_target_ms=100.0, tpot_target_ms=10.0),
        seed=42,
    )
