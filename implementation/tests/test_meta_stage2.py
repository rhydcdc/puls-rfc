"""Cluster M — Stage 2 Impl-10 main meta-test + signature divergence lock-in.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster M. α path signature divergence +
calibration provenance + helper signature inventory + honest disclosure lock-in.
"""

import inspect

from puls_sched.config import (
    CalibrationConfig,
    Config,
    TimeConfig,
    compute_ffn_op_time_s,
    compute_gpu_op_time_s,
    default_dummy_config,
)
from puls_sched.evaluator import (
    AuxCell,
    AuxSource,
    Evaluator,
    F3ClosedFormResult,
    F5TraceGroundedResult,
)
from puls_sched.micro_batch import MicroBatch


# ============================================================================
# M1 — Config calibration sub-config field 존재
# ============================================================================


def test_m1_config_has_calibration_field():
    cfg = default_dummy_config()
    fields = cfg.__dataclass_fields__
    assert "calibration" in fields
    assert isinstance(cfg.calibration, CalibrationConfig)


# ============================================================================
# M2 — CalibrationConfig provenance label 11 영역 lock-in
# ============================================================================


def test_m2_calibration_provenance_inventory():
    cal = CalibrationConfig()
    expected = {
        "gpu_fp16_dense_peak_tflops", "gpu_fp8_dense_peak_tflops",
        "hbm_bw", "eta_hbm_external", "eta_hbm_internal",
        "pim_tile_fp8", "pim_tile_fp16", "rtl_clock", "rtl_fsm_cycle_per_tile",
        "weight_bytes_per_layer", "kv_bytes", "ffn_intermediate",
        "trace", "mfu", "aux1_microbench",
    }
    assert expected.issubset(set(cal.provenance.keys()))


# ============================================================================
# M3 — Evaluator 신규 5 method 영역 존재
# ============================================================================


def test_m3_evaluator_stage2_methods_present():
    methods = [
        "aux1_mixed_batch_weight_reuse",
        "aux2_bus_traffic_reduction",
        "f3_closed_form",
        "f5_trace_grounded",
        "f3_cross_validate",
        "d2_report",
    ]
    for m in methods:
        assert hasattr(Evaluator, m), f"Evaluator.{m} 영역 부재"


# ============================================================================
# M4 — AuxSource enum = {AUX1, AUX2}
# ============================================================================


def test_m4_aux_source_enum():
    members = {m.name for m in AuxSource}
    assert members == {"AUX1", "AUX2"}


# ============================================================================
# M5 — AuxCell frozen + 4 field
# ============================================================================


def test_m5_aux_cell_field_inventory():
    fields = AuxCell.__dataclass_fields__
    expected = {"source", "saving_per_layer_ns", "saving_total_80_layer_ns", "provenance_label"}
    assert set(fields.keys()) == expected
    import dataclasses
    assert AuxCell.__dataclass_params__.frozen


# ============================================================================
# M6 — F3ClosedFormResult + F5TraceGroundedResult frozen
# ============================================================================


def test_m6_f3_f5_result_frozen():
    assert F3ClosedFormResult.__dataclass_params__.frozen
    assert F5TraceGroundedResult.__dataclass_params__.frozen


# ============================================================================
# M7 — Helper signature lock-in
# ============================================================================


def test_m7_compute_gpu_op_time_signature():
    sig = inspect.signature(compute_gpu_op_time_s)
    params = list(sig.parameters.keys())
    assert params == ["node_type", "mb", "cal", "model", "time_cfg"]


def test_m7_compute_ffn_op_time_signature():
    sig = inspect.signature(compute_ffn_op_time_s)
    params = list(sig.parameters.keys())
    assert params == ["mb", "cal", "model"]


# ============================================================================
# M8 — MicroBatch.prefill_processed field 존재
# ============================================================================


def test_m8_micro_batch_prefill_processed_field():
    fields = MicroBatch.__dataclass_fields__
    assert "prefill_processed" in fields


# ============================================================================
# M9 — NVLink 산식 위 등장 0 (evaluator method)
# ============================================================================


def test_m9_no_nvlink_in_aux_f3_f5_methods():
    methods = ["aux1_mixed_batch_weight_reuse", "aux2_bus_traffic_reduction",
               "f3_closed_form", "f5_trace_grounded"]
    for m in methods:
        src = inspect.getsource(getattr(Evaluator, m))
        assert "nvlink" not in src.lower(), f"Evaluator.{m} 위 NVLink substring 등장"


# ============================================================================
# M10 — TimeConfig — gpu_op_time_us lookup 폐기 (decode_attn_fallback 만)
# ============================================================================


def test_m10_time_config_gpu_op_time_us_only_fallback():
    cfg = default_dummy_config()
    keys = set(cfg.time.gpu_op_time_us.keys())
    assert keys == {"decode_attn_fallback"}, f"Stage 1 dummy lookup 잔존: {keys}"


# ============================================================================
# M11 — D2 markdown 위 honest disclosure substring 정합
# ============================================================================


def test_m11_d2_markdown_honest_disclosure():
    from puls_sched.clock import Clock
    from puls_sched.idle_telemetry import IdleTelemetry
    cfg = default_dummy_config()
    ev = Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())
    d2 = ev.d2_report(kv_lengths=[100000, 50000])
    md = d2["markdown"]
    # Honest disclosure 영역
    assert "Impl-11 deferred" in md
    assert "HBM4 hypothetical" in md or "hbm4_projection" in md.lower()
    assert "Framing A" in md
    assert "async hidden" in md  # NVLink async hidden disclosure
    assert "FP16 PIM tile" in md or "Regime A" in md  # FP16 reference disclosure
    assert "per-mb spec-derived" in md  # Stage 2 timing 정합


# ============================================================================
# M12 — SchedulerCore._compose_admission_payload spec-derived t_proj (no qkv lookup)
# ============================================================================


def test_m12_admission_payload_no_stage1_lookup():
    from puls_sched.main_loop import SchedulerCore
    src = inspect.getsource(SchedulerCore._compose_admission_payload)
    # Stage 1 dummy lookup 영역 폐기 — qkv / o_proj lookup 부재
    assert 'gpu_op_time_us["qkv"]' not in src
    assert 'gpu_op_time_us["o_proj"]' not in src
    # Stage 2 — last mb 위 spec-derived 산출
    assert "compute_gpu_op_time_s" in src
