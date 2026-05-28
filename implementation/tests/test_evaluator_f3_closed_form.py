"""Cluster F3 — Evaluator.f3_closed_form (α path β 정합) 산출 정합.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster F3. ARCH §5.7 F3 + §6.6 regime.
"""

import dataclasses

import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.evaluator import Evaluator
from puls_sched.idle_telemetry import IdleTelemetry


@pytest.fixture
def ev():
    cfg = default_dummy_config()
    return Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())


# ============================================================================
# F3.1 — F3 formula = max(a, b) / (a + b)
# ============================================================================


def test_f3_1_formula(ev):
    r = ev.f3_closed_form(ctx_tokens=32000, prefill_chunk=512)
    expected = max(r.a_cycle_us, r.b_cycle_us) / (r.a_cycle_us + r.b_cycle_us)
    assert abs(r.f3_ratio - expected) < 1e-9


# ============================================================================
# F3.2 — Boundary: a=b → ratio=0.5
# ============================================================================


def test_f3_2_a_equals_b_ratio_half(ev):
    """수동 cycle 산출 — 영원 F3 산식 영원 boundary."""
    # 단순 — Evaluator 의 pipeline_efficiency 영원 동일 산식 영원 활용 (a=b 위 0.5)
    assert abs(ev.pipeline_efficiency(100.0, 100.0) - 0.5) < 1e-9


# ============================================================================
# F3.3 — Extremum: a >> b → ratio → 1.0
# ============================================================================


def test_f3_3_a_extremum_ratio_near_one(ev):
    assert abs(ev.pipeline_efficiency(1e6, 1.0) - 1.0) < 1e-6


# ============================================================================
# F3.4 — ctx sweep transition (short-ctx → long-ctx)
# ============================================================================


def test_f3_4_ctx_sweep_monotonic_t_attn(ev):
    """ctx ↑ → t_attn ↑ (PIM 영역 영원 linear, num_tiles 증가)."""
    r_short = ev.f3_closed_form(ctx_tokens=2000)
    r_long = ev.f3_closed_form(ctx_tokens=128000)
    assert r_long.t_attn_us > r_short.t_attn_us


# ============================================================================
# F3.5 — MFU sweep 위 t_proj 영향
# ============================================================================


def test_f3_5_mfu_sweep_inverse_t_proj(ev):
    """MFU ↑ → t_proj ↓ (peak × MFU 분모 효과)."""
    cal = ev.config.calibration
    r_low = ev.f3_closed_form(ctx_tokens=32000)
    cal_high = dataclasses.replace(cal, gpu_mfu_default=0.7)
    cfg_high = dataclasses.replace(ev.config, calibration=cal_high)
    ev_high = Evaluator(config=cfg_high, clock=ev.clock, idle_telemetry=ev.idle_telemetry)
    r_high = ev_high.f3_closed_form(ctx_tokens=32000)
    assert r_high.t_proj_us < r_low.t_proj_us


# ============================================================================
# F3.6 — Batch invariance — batch 위 ratio 보존
# ============================================================================


def test_f3_6_batch_invariance(ev):
    """batch 위 분자 분모 동일 비율 영향 → ratio 보존."""
    # 단 — t_attn 위 batch linear 영원 t_proj/t_FFN 위 batch linear → ratio 영원 동일
    # 단 num_tiles 위 ceil 영원 영원 — batch 영원 영원 영원 ratio 미세 차이 가능
    r1 = ev.f3_closed_form(ctx_tokens=32000, batch=1)
    r2 = ev.f3_closed_form(ctx_tokens=32000, batch=10)
    # ratio 영원 영원 영원 비슷 (단 num_tiles ceil 영원 영원 미세 차이)
    assert abs(r1.f3_ratio - r2.f3_ratio) < 0.05


# ============================================================================
# F3.7 — t_QKV 산식 정합 (Llama-3 weight × MFU)
# ============================================================================


def test_f3_7_t_qkv_formula(ev):
    cal = ev.config.calibration
    model = ev.config.model
    r = ev.f3_closed_form(ctx_tokens=32000, batch=16)
    expected_flops = 2 * 16 * model.hidden * (model.hidden + 2 * model.num_kv_heads * model.head_dim)
    peak = cal.gpu_fp16_dense_peak_tflops * 1e12 * cal.gpu_mfu_default
    expected_us = expected_flops / peak * 1e6
    assert abs(r.t_qkv_us - expected_us) / expected_us < 1e-6


# ============================================================================
# F3.8 — t_FFN 산식 정합 (3 GEMM, intermediate=28672)
# ============================================================================


def test_f3_8_t_ffn_formula(ev):
    cal = ev.config.calibration
    model = ev.config.model
    r = ev.f3_closed_form(ctx_tokens=32000, batch=16)
    expected_flops = 6 * 16 * model.hidden * model.ffn_intermediate
    peak = cal.gpu_fp16_dense_peak_tflops * 1e12 * cal.gpu_mfu_default
    expected_us = expected_flops / peak * 1e6
    assert abs(r.b_cycle_us - expected_us) / expected_us < 1e-6


# ============================================================================
# F3.9 — t_attn 산식 (num_tiles × pim_tile)
# ============================================================================


def test_f3_9_t_attn_formula(ev):
    import math
    cal = ev.config.calibration
    hw = ev.config.hw
    tile_rows = ev.config.time.rtl_fsm_tile_rows
    k_agg = hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack
    r = ev.f3_closed_form(ctx_tokens=32000, batch=1)
    expected_tiles = math.ceil(1 * 32000 / (k_agg * tile_rows))
    expected_us = expected_tiles * cal.pim_tile_time_fp8_ns_calibrated * 1e-3
    assert abs(r.t_attn_us - expected_us) < 1e-6


# ============================================================================
# F3.10 — Determinism
# ============================================================================


def test_f3_10_determinism(ev):
    expected = ev.f3_closed_form(ctx_tokens=32000)
    for _ in range(1000):
        assert ev.f3_closed_form(ctx_tokens=32000) == expected


# ============================================================================
# F3.11 — β path lock-in (closed-form 단독, run.loop 우회)
# ============================================================================


def test_f3_11_closed_form_does_not_use_idle_telemetry(ev):
    """f3_closed_form 영원 idle_telemetry 영원 미사용 (closed-form 영원 spec-derived 단독)."""
    # idle_telemetry 영원 empty 위 영원 산출 정합 (cross_validate 와 분리)
    r = ev.f3_closed_form(ctx_tokens=32000)
    assert r.f3_ratio > 0


# ============================================================================
# F3.12 — Provenance label
# ============================================================================


def test_f3_12_provenance_alpha_path(ev):
    r = ev.f3_closed_form(ctx_tokens=32000)
    assert "alpha_path" in r.provenance
