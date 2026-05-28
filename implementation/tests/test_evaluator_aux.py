"""Cluster A1·A2 — Evaluator.aux1·aux2 closed-form 산출 정합.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster A1·A2.
"""

import dataclasses

import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.evaluator import Evaluator, AuxSource
from puls_sched.idle_telemetry import IdleTelemetry


@pytest.fixture
def ev():
    cfg = default_dummy_config()
    return Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())


# ============================================================================
# Cluster A1 — Aux1 mixed batching weight reuse
# ============================================================================


def test_a1_1_aux1_formula(ev):
    """saving_per_layer = weight_bytes / (HBM × η)."""
    cal = ev.config.calibration
    bw = cal.hbm_ext_bw_aggregate_TBps * 1e12 * cal.eta_hbm_external
    expected_ns = cal.weight_bytes_per_layer_fp16 / bw * 1e9
    aux1 = ev.aux1_mixed_batch_weight_reuse()
    assert abs(aux1.saving_per_layer_ns - expected_ns) / expected_ns < 1e-6


def test_a1_2_aux1_eta_monotonic_inverse(ev):
    """η ↑ → saving ↓ (BW 효율 ↑ → 절감 ↓)."""
    cal = ev.config.calibration
    aux1_low = ev.aux1_mixed_batch_weight_reuse()
    cal_high = dataclasses.replace(cal, eta_hbm_external=0.80)
    cfg_high = dataclasses.replace(ev.config, calibration=cal_high)
    ev_high = Evaluator(config=cfg_high, clock=ev.clock, idle_telemetry=ev.idle_telemetry)
    aux1_high = ev_high.aux1_mixed_batch_weight_reuse()
    assert aux1_high.saving_per_layer_ns < aux1_low.saving_per_layer_ns


def test_a1_3_aux1_80_layer_aggregate(ev):
    aux1 = ev.aux1_mixed_batch_weight_reuse()
    assert abs(aux1.saving_total_80_layer_ns - aux1.saving_per_layer_ns * 80) < 1e-6


def test_a1_4_aux1_source_lock_in(ev):
    aux1 = ev.aux1_mixed_batch_weight_reuse()
    assert aux1.source == AuxSource.AUX1


def test_a1_5_aux1_provenance_label(ev):
    aux1 = ev.aux1_mixed_batch_weight_reuse()
    assert "llama3_70b" in aux1.provenance_label
    assert "hbm4_projection" in aux1.provenance_label
    assert "d1_eta_hbm3" in aux1.provenance_label


def test_a1_6_aux1_determinism(ev):
    expected = ev.aux1_mixed_batch_weight_reuse()
    for _ in range(1000):
        assert ev.aux1_mixed_batch_weight_reuse() == expected


# ============================================================================
# Cluster A2 — Aux2 bus traffic reduction
# ============================================================================


def test_a2_1_aux2_formula(ev):
    """saving = time_KV − (time_PIM + time_result)."""
    aux2 = ev.aux2_bus_traffic_reduction(kv_length=128000)
    # 큰 kv → saving > 0 expected
    assert aux2.saving_total_80_layer_ns > 0


def test_a2_2_aux2_kv_monotonic(ev):
    aux2_small = ev.aux2_bus_traffic_reduction(kv_length=2000)
    aux2_large = ev.aux2_bus_traffic_reduction(kv_length=128000)
    # kv ↑ → saving ↑ (long-ctx dominant)
    assert aux2_large.saving_total_80_layer_ns > aux2_small.saving_total_80_layer_ns


def test_a2_3_aux2_eta_external_monotonic(ev):
    """η_ext ↑ → time_KV ↓ → saving ↓."""
    cal = ev.config.calibration
    aux2_low = ev.aux2_bus_traffic_reduction(kv_length=128000)
    cal_high = dataclasses.replace(cal, eta_hbm_external=0.80)
    cfg_high = dataclasses.replace(ev.config, calibration=cal_high)
    ev_high = Evaluator(config=cfg_high, clock=ev.clock, idle_telemetry=ev.idle_telemetry)
    aux2_high = ev_high.aux2_bus_traffic_reduction(kv_length=128000)
    assert aux2_high.saving_total_80_layer_ns < aux2_low.saving_total_80_layer_ns


def test_a2_4_aux2_short_ctx_low_saving(ev):
    """Short-ctx (kv=2k) 위 saving 작음 — long-ctx dominant 정합."""
    aux2_short = ev.aux2_bus_traffic_reduction(kv_length=2000)
    aux2_long = ev.aux2_bus_traffic_reduction(kv_length=128000)
    assert aux2_short.saving_total_80_layer_ns < aux2_long.saving_total_80_layer_ns / 10


def test_a2_5_aux2_source_lock_in(ev):
    aux2 = ev.aux2_bus_traffic_reduction(kv_length=128000)
    assert aux2.source == AuxSource.AUX2


def test_a2_6_aux2_provenance_label(ev):
    aux2 = ev.aux2_bus_traffic_reduction(kv_length=128000)
    assert "llama3_70b" in aux2.provenance_label
    assert "longbench" in aux2.provenance_label
    assert "d5_ramulator" in aux2.provenance_label


def test_a2_7_aux2_zero_kv_handled(ev):
    aux2 = ev.aux2_bus_traffic_reduction(kv_length=0)
    # kv=0 위 time_KV = 0, time_PIM = 0, time_result = constant → saving < 0 가능
    assert aux2.saving_total_80_layer_ns == aux2.saving_total_80_layer_ns  # not NaN


def test_a2_8_aux2_determinism(ev):
    expected = ev.aux2_bus_traffic_reduction(kv_length=64000)
    for _ in range(1000):
        assert ev.aux2_bus_traffic_reduction(kv_length=64000) == expected


def test_a2_9_aux2_k_aggregate_2048(ev):
    """ARCH §3.2 literal — k_aggregate = 8 GPU × 8 stack × 32 ch = 2048."""
    hw = ev.config.hw
    assert hw.num_gpus_instance_a * hw.num_stacks_per_gpu * hw.num_channels_per_stack == 2048


def test_a2_10_aux2_uses_calibrated_pim_tile(ev):
    """Aux2 산식 위 PIM tile = D5 calibrated (267 ns FP8)."""
    cal = ev.config.calibration
    assert cal.pim_tile_time_fp8_ns_calibrated == 267.0
