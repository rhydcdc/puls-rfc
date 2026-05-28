"""Cluster CA — Stage 2 CalibrationConfig + provenance label lock-in.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster CA. 사용자 결정 값 lock-in
(B200 + HBM4 projection + D5 + D1 + Llama-3 70B + MFU sweep).
"""

import dataclasses

import pytest

from puls_sched.config import (
    CalibrationConfig,
    Config,
    TimeConfig,
    default_dummy_config,
)


# ============================================================================
# CA1 — CalibrationConfig default 값 정합
# ============================================================================


def test_ca1_calibration_config_b200_compute_defaults():
    """B200 standalone FP16 dense peak 2200 TFLOPS (사용자 chart 정합)."""
    cal = CalibrationConfig()
    assert cal.gpu_fp16_dense_peak_tflops == 2200.0
    assert cal.gpu_fp8_dense_peak_tflops == 4500.0  # Impl-11 reference
    assert cal.gpu_mfu_default == 0.6


def test_ca1_calibration_config_hbm4_projection():
    """HBM4 hypothetical projection — 16 TB/s per GPU × 8 GPU = 128 TB/s aggregate."""
    cal = CalibrationConfig()
    assert cal.hbm4_bw_per_stack_TBps == 2.0
    assert cal.num_hbm_stacks_per_gpu == 8
    assert cal.hbm_ext_bw_per_gpu_TBps == 16.0
    assert cal.hbm_ext_bw_aggregate_TBps == 128.0
    assert cal.eta_hbm_external == 0.74  # D1 fix
    assert cal.eta_hbm_internal == 1.0  # ARCH §3.1 bus 부재


def test_ca1_calibration_config_pim_d5():
    """PIM tile = D5 calibrated 직접 (FP8 = 267 ns, RTL 347 cycle @ 1.3 GHz)."""
    cal = CalibrationConfig()
    assert cal.pim_tile_time_fp8_ns_calibrated == 267.0
    assert cal.pim_tile_time_fp16_ns_calibrated == 512.0  # reference only
    assert cal.rtl_fsm_cycle_per_tile_calibrated == 347
    assert cal.rtl_clock_GHz == 1.3


# ============================================================================
# CA2 — Provenance dict — 모든 calibrated field 위 출처 라벨
# ============================================================================


def test_ca2_provenance_label_all_fields_present():
    """PLAN §0.5 출처 라벨 동반 — 11 영역 calibrated field 모두 lock-in."""
    cal = CalibrationConfig()
    expected_keys = {
        "gpu_fp16_dense_peak_tflops",
        "hbm_bw",
        "eta_hbm_external",
        "eta_hbm_internal",
        "pim_tile_fp8",
        "pim_tile_fp16",
        "rtl_clock",
        "weight_bytes_per_layer",
        "kv_bytes",
        "trace",
        "mfu",
        "aux1_microbench",
    }
    for key in expected_keys:
        assert key in cal.provenance, f"missing provenance label: {key}"


# ============================================================================
# CA3 — Config 위 calibration field + frozen invariance
# ============================================================================


def test_ca3_config_has_calibration_field():
    cfg = default_dummy_config()
    assert isinstance(cfg.calibration, CalibrationConfig)


def test_ca3_calibration_config_frozen():
    cal = CalibrationConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.gpu_fp16_dense_peak_tflops = 9999.0  # type: ignore


# ============================================================================
# CA4 — default_dummy_config 위 CalibrationConfig default instantiate
# ============================================================================


def test_ca4_default_dummy_config_calibration_present():
    cfg = default_dummy_config()
    assert cfg.calibration is not None
    assert cfg.calibration.gpu_fp16_dense_peak_tflops == 2200.0


# ============================================================================
# CA5 — η_HBM sensitivity sweep tuple
# ============================================================================


def test_ca5_eta_hbm_sweep_tuple():
    cal = CalibrationConfig()
    assert cal.eta_hbm_external_sweep == (0.70, 0.74, 0.80)


# ============================================================================
# CA6 — MFU sensitivity sweep tuple
# ============================================================================


def test_ca6_mfu_sweep_tuple():
    cal = CalibrationConfig()
    assert cal.gpu_mfu_sweep == (0.5, 0.6, 0.7)


# ============================================================================
# CA7 — TimeConfig 위 calibrated 값 교체 (D5 직접)
# ============================================================================


def test_ca7_time_config_pim_tile_d5_calibrated():
    cfg = default_dummy_config()
    assert cfg.time.pim_tile_time_ns["FP8"] == 267.0
    assert cfg.time.pim_tile_time_ns["FP16"] == 512.0
    assert cfg.time.rtl_fsm_cycle_per_tile == 347
    assert cfg.time.rtl_fsm_tile_rows == 32  # ARCH §3.1 RTL 합성


def test_ca7_time_config_gpu_op_lookup_f1_reference_only():
    """Stage 2 — gpu_op_time_us lookup 폐기, F1 ablation reference (decode_attn_fallback) 만 보존.
    다른 GPU op_time 은 compute_gpu_op_time_s per-mb spec-derived 산출."""
    cfg = default_dummy_config()
    assert "decode_attn_fallback" in cfg.time.gpu_op_time_us
    # 폐기 — qkv / prefill_attn / o_proj 영역 lookup 없음
    for deprecated_key in ("qkv", "prefill_attn", "o_proj"):
        assert deprecated_key not in cfg.time.gpu_op_time_us, \
            f"Stage 1 dummy '{deprecated_key}' 영역 폐기 — spec-derived 산출만 (사용자 의도 정합)"


# ============================================================================
# CA8 — Provenance label substring 정합 (출처 영역 명시)
# ============================================================================


def test_ca8_provenance_substring_match():
    cal = CalibrationConfig()
    assert "blackwell_b200" in cal.provenance["gpu_fp16_dense_peak_tflops"]
    assert "hbm4_projection" in cal.provenance["hbm_bw"]
    assert "d1_eta_hbm3" in cal.provenance["eta_hbm_external"]
    assert "d5_ramulator" in cal.provenance["pim_tile_fp8"]
    assert "d5_pre_rtl" in cal.provenance["rtl_clock"]
    assert "llama3_70b" in cal.provenance["weight_bytes_per_layer"]
    assert "longbench" in cal.provenance["trace"]
    assert "mfu" in cal.provenance["mfu"]
    assert "colab_t4" in cal.provenance["aux1_microbench"]
