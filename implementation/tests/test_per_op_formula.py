"""Cluster F — Per-op spec-derived formula 정합 (Stage 1 time abstraction 폐기).

PLAN §4 Impl-10 main + impl_10.md §5 Cluster F. ARCH §3.5.2 Computed Wait literal +
사용자 의도 "각 trace 위 정확 산출, 평균/dummy 0" 정합.
"""

import pytest

from puls_sched.config import (
    compute_ffn_op_time_s,
    compute_gpu_op_time_s,
    default_dummy_config,
)
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cfg():
    return default_dummy_config()


def make_mb(decode_n: int = 0, prefill_chunk_per_req: dict[int, int] | None = None,
            prefill_processed: dict[int, int] | None = None) -> MicroBatch:
    """Test-only mb builder. prefill_chunk_per_req = {req_id: chunk_size}."""
    decode_tokens = {i: 0 for i in range(decode_n)}
    prefill_chunk = {}
    if prefill_chunk_per_req:
        for rid, chunk in prefill_chunk_per_req.items():
            prefill_chunk[rid] = list(range(chunk))
    return MicroBatch(
        id=1,
        decode_tokens=decode_tokens,
        prefill_chunk=prefill_chunk,
        prefill_processed=prefill_processed or {},
        kv_rows_total=decode_n * 1000,
    )


# ============================================================================
# F.1 — QKV formula
# ============================================================================


def test_f1_qkv_formula(cfg):
    """FLOPs = 2 × batch × hidden × (hidden + 2 × n_kv × d_head) / (peak × MFU)."""
    mb = make_mb(decode_n=16, prefill_chunk_per_req={100: 512})
    t_s = compute_gpu_op_time_s(NodeType.QKV, mb, cfg.calibration, cfg.model)
    batch = 16 + 512
    expected_flops = 2 * batch * 8192 * (8192 + 2 * 8 * 128)
    expected_t = expected_flops / (2200e12 * 0.6)
    assert abs(t_s - expected_t) / expected_t < 1e-6, f"QKV formula mismatch"


# ============================================================================
# F.2 — O_PROJ formula
# ============================================================================


def test_f2_o_proj_formula(cfg):
    """FLOPs = 2 × batch × hidden^2 / (peak × MFU)."""
    mb = make_mb(decode_n=16, prefill_chunk_per_req={100: 512})
    t_s = compute_gpu_op_time_s(NodeType.O_PROJ, mb, cfg.calibration, cfg.model)
    batch = 16 + 512
    expected_flops = 2 * batch * 8192 * 8192
    expected_t = expected_flops / (2200e12 * 0.6)
    assert abs(t_s - expected_t) / expected_t < 1e-6, f"O_PROJ formula mismatch"


# ============================================================================
# F.3 — PREFILL_ATTN causal per-req formula
# ============================================================================


def test_f3_prefill_attn_causal_per_req(cfg):
    """FLOPs(req) = 2 × chunk × hidden × (prefill_processed + chunk)."""
    # 2 prefill reqs: req 100 = chunk 256, processed 100; req 200 = chunk 256, processed 500
    mb = make_mb(
        decode_n=0,
        prefill_chunk_per_req={100: 256, 200: 256},
        prefill_processed={100: 100, 200: 500},
    )
    t_s = compute_gpu_op_time_s(NodeType.PREFILL_ATTN, mb, cfg.calibration, cfg.model)
    # FLOPs = 2 × 256 × 8192 × (100 + 256) + 2 × 256 × 8192 × (500 + 256)
    expected_flops = (
        2 * 256 * 8192 * (100 + 256)
        + 2 * 256 * 8192 * (500 + 256)
    )
    expected_t = expected_flops / (2200e12 * 0.6)
    assert abs(t_s - expected_t) / expected_t < 1e-6


def test_f3_prefill_attn_no_prefill_returns_zero(cfg):
    """Decode-only mb 위 PREFILL_ATTN = 0.0."""
    mb = make_mb(decode_n=16, prefill_chunk_per_req=None)
    t_s = compute_gpu_op_time_s(NodeType.PREFILL_ATTN, mb, cfg.calibration, cfg.model)
    assert t_s == 0.0


# ============================================================================
# F.4 — FFN formula (compute_ffn_op_time_s, Instance B α path)
# ============================================================================


def test_f4_ffn_formula(cfg):
    """FLOPs = 6 × batch × hidden × intermediate / (peak × MFU)."""
    mb = make_mb(decode_n=16, prefill_chunk_per_req={100: 512})
    t_s = compute_ffn_op_time_s(mb, cfg.calibration, cfg.model)
    batch = 16 + 512
    expected_flops = 6 * batch * 8192 * 28672
    expected_t = expected_flops / (2200e12 * 0.6)
    assert abs(t_s - expected_t) / expected_t < 1e-6


# ============================================================================
# F.5 — batch monotonic
# ============================================================================


def test_f5_batch_monotonic_linear(cfg):
    """batch ↑ → op_time ↑ (linear). QKV, O_PROJ, FFN 모두 batch linear."""
    mb_small = make_mb(decode_n=8)
    mb_large = make_mb(decode_n=64)
    for nt in [NodeType.QKV, NodeType.O_PROJ]:
        t_small = compute_gpu_op_time_s(nt, mb_small, cfg.calibration, cfg.model)
        t_large = compute_gpu_op_time_s(nt, mb_large, cfg.calibration, cfg.model)
        ratio = t_large / t_small
        assert abs(ratio - 8.0) < 1e-6, f"{nt.name} ratio={ratio} expected 8.0"
    # FFN
    t_small_ffn = compute_ffn_op_time_s(mb_small, cfg.calibration, cfg.model)
    t_large_ffn = compute_ffn_op_time_s(mb_large, cfg.calibration, cfg.model)
    assert abs(t_large_ffn / t_small_ffn - 8.0) < 1e-6


# ============================================================================
# F.6 — MFU sweep monotonic
# ============================================================================


def test_f6_mfu_sweep_inverse_monotonic(cfg):
    """MFU ↑ → op_time ↓ (inverse, 1/MFU scaling)."""
    import dataclasses
    mb = make_mb(decode_n=16)
    times = {}
    for mfu in [0.5, 0.6, 0.7]:
        cal = dataclasses.replace(cfg.calibration, gpu_mfu_default=mfu)
        t = compute_gpu_op_time_s(NodeType.QKV, mb, cal, cfg.model)
        times[mfu] = t
    assert times[0.5] > times[0.6] > times[0.7]
    # Ratio inverse to MFU ratio
    assert abs(times[0.5] / times[0.6] - 0.6 / 0.5) < 1e-6


# ============================================================================
# F.7 — per-mb 정확 산출 (deterministic + different mb different)
# ============================================================================


def test_f7_per_mb_deterministic_and_distinct(cfg):
    """동일 mb 위 동일 산출 (deterministic), 다른 mb 위 다른 산출 (per-mb 정확)."""
    mb_a = make_mb(decode_n=16, prefill_chunk_per_req={100: 256})
    mb_b = make_mb(decode_n=32, prefill_chunk_per_req={200: 256})
    t_a1 = compute_gpu_op_time_s(NodeType.QKV, mb_a, cfg.calibration, cfg.model)
    t_a2 = compute_gpu_op_time_s(NodeType.QKV, mb_a, cfg.calibration, cfg.model)
    t_b = compute_gpu_op_time_s(NodeType.QKV, mb_b, cfg.calibration, cfg.model)
    assert t_a1 == t_a2  # deterministic
    assert t_a1 != t_b   # per-mb 정확 (batch 차이 반영)


# ============================================================================
# F.8 — MicroBatch.prefill_processed dict field
# ============================================================================


def test_f8_micro_batch_prefill_processed_field():
    """MicroBatch 위 prefill_processed: dict[int, int] field 존재."""
    mb = MicroBatch(id=1)
    assert hasattr(mb, "prefill_processed")
    assert mb.prefill_processed == {}
    mb2 = MicroBatch(id=2, prefill_processed={100: 50})
    assert mb2.prefill_processed == {100: 50}


# ============================================================================
# F.9 — DECODE_ATTN F1 ablation fallback (TimeConfig required)
# ============================================================================


def test_f9_decode_attn_f1_fallback_requires_time_cfg(cfg):
    """DECODE_ATTN 산출은 F1 ablation fallback (time_cfg 필수)."""
    mb = make_mb(decode_n=16)
    # With time_cfg
    t_s = compute_gpu_op_time_s(NodeType.DECODE_ATTN, mb, cfg.calibration, cfg.model, cfg.time)
    assert t_s == 4.0e-6  # gpu_op_time_us["decode_attn_fallback"] = 4.0 us → seconds
    # Without time_cfg → raises
    with pytest.raises(ValueError, match="DECODE_ATTN"):
        compute_gpu_op_time_s(NodeType.DECODE_ATTN, mb, cfg.calibration, cfg.model)


# ============================================================================
# F.10 — Unknown node_type ValueError (FFN 위 NodeType enum 아님 — 별도 helper 사용)
# ============================================================================


def test_f10_unknown_node_type_raises(cfg):
    """compute_gpu_op_time_s 위 DAG 4 node 만 cover. 다른 영역 위 ValueError."""
    mb = make_mb(decode_n=16)
    # Mock node type — NodeType enum 외 (FFN 등은 별도 helper 위 위 enum 외)
    class FakeNodeType:
        pass
    with pytest.raises(ValueError, match="Unknown node_type"):
        compute_gpu_op_time_s(FakeNodeType(), mb, cfg.calibration, cfg.model)
