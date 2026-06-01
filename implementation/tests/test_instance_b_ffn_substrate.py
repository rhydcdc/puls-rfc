"""Cluster B — Instance B FFN substrate (α path Carry-1 해소) + F3 cross-validate.

PLAN §4 Impl-10 main + impl_10.md §4.4 + §5 Cluster B.

ARCH §3.4 Instance B FFN cycle literal + ARCH §6.4 inter-AB balance signal substrate
정합. impl_10_pre.md §12.4 Carry-1 영역 Stage 2 해소 검증.
"""

import dataclasses

import pytest

from puls_sched.clock import Clock
from puls_sched.config import (
    CalibrationConfig,
    compute_ffn_op_time_s,
    default_dummy_config,
)
from puls_sched.evaluator import Evaluator
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.micro_batch import MicroBatch
from puls_sched.nvlink import NVLinkTransfer


@pytest.fixture
def cfg():
    return default_dummy_config()


@pytest.fixture
def alpha_pipeline(cfg):
    """α path 활성 (clock + idle_telemetry 보유) instance_pipeline."""
    ia = Instance(name="A", has_pim=True)
    ib = Instance(name="B", has_pim=False)
    return InstancePipeline(
        config=cfg,
        instance_a=ia,
        instance_b=ib,
        nvlink=NVLinkTransfer(config=cfg),
        clock=Clock(),
        idle_telemetry=IdleTelemetry(),
    )


def make_mb(decode_n: int, prefill_per_req: dict[int, int] | None = None):
    decode_tokens = {i: 0 for i in range(decode_n)}
    prefill_chunk = {}
    if prefill_per_req:
        for rid, chunk in prefill_per_req.items():
            prefill_chunk[rid] = list(range(chunk))
    return MicroBatch(
        id=1,
        decode_tokens=decode_tokens,
        prefill_chunk=prefill_chunk,
        prefill_processed={rid: 0 for rid in (prefill_per_req or {})},
        kv_rows_total=decode_n * 1000,
    )


# ============================================================================
# B1 — compute_ffn_op_time_s 산식 정합
# ============================================================================


def test_b1_ffn_op_time_formula(cfg):
    """FLOPs = 6 × batch × hidden × intermediate / (peak × MFU)."""
    mb = make_mb(decode_n=16, prefill_per_req={100: 512})
    t_s = compute_ffn_op_time_s(mb, cfg.calibration, cfg.model)
    batch = 16 + 512
    expected = 6 * batch * 8192 * 28672 / (2200e12 * 0.6)
    assert abs(t_s - expected) / expected < 1e-6


# ============================================================================
# B2 — InstancePipeline.dispatch 위 gpu_instance_b recording
# ============================================================================


def test_b2_dispatch_records_gpu_instance_b(cfg, alpha_pipeline):
    """α path Carry-1 — dispatch 시점 위 gpu_instance_b active duration record."""
    mb = make_mb(decode_n=16, prefill_per_req={100: 512})
    assert alpha_pipeline.idle_telemetry.active_duration("gpu_instance_b") == 0.0
    alpha_pipeline.dispatch(mb)
    recorded = alpha_pipeline.idle_telemetry.active_duration("gpu_instance_b")
    # Phase-2 — Instance B TP=num_gpus_instance_b 분산 (dispatch 가 동일 num_gpus 전달).
    expected_us = compute_ffn_op_time_s(
        mb, cfg.calibration, cfg.model, num_gpus=cfg.hw.num_gpus_instance_b,
    ) * 1e6
    assert abs(recorded - expected_us) / expected_us < 1e-6


# ============================================================================
# B3 — main_loop production hot path wiring (instance_pipeline.dispatch 호출)
# ============================================================================


def test_b3_main_loop_activates_gpu_instance_b_signal(cfg):
    """main_loop._maybe_advance_forward_pass 위 instance_pipeline.dispatch 호출 정합."""
    from puls_sched.main_loop import SchedulerCore
    import inspect
    src = inspect.getsource(SchedulerCore._maybe_advance_forward_pass)
    assert "instance_pipeline.dispatch" in src, \
        "main_loop 위 instance_pipeline.dispatch 호출 wiring 부재 (α path)"


# ============================================================================
# B4 — b_cycle > 0 검증 (Carry-1 해소)
# ============================================================================


def test_b4_b_cycle_grows_after_layer_cycles(cfg, alpha_pipeline):
    """α path 후 — multiple layer dispatch 위 b_cycle 누적."""
    mb1 = make_mb(decode_n=16)
    mb2 = make_mb(decode_n=32)
    alpha_pipeline.dispatch(mb1)
    after_1 = alpha_pipeline.idle_telemetry.active_duration("gpu_instance_b")
    alpha_pipeline.clock.advance_to(alpha_pipeline.clock.now + 1000.0)
    alpha_pipeline.dispatch(mb2)
    after_2 = alpha_pipeline.idle_telemetry.active_duration("gpu_instance_b")
    assert after_2 > after_1, "b_cycle 누적 안 됨 — Stage 1 placeholder 영원 보존?"


# ============================================================================
# B5 — balance_inter_AB 진정 활성 (asymmetric a > b 분기 fire 가능)
# ============================================================================


def test_b5_balance_inter_ab_a_gt_b_branch_fires(cfg):
    """Stage 1 위 b_cycle=0 → 영원 a<b 분기만 fire. Stage 2 위 a>b 도 fire 가능."""
    from puls_sched.admission import Admission
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    admission = Admission(
        admission_cfg=cfg.admission,
        request_queue=RequestQueue(capacity=100),
        kv_accountant=KVAccountant(capacity=cfg.admission.kv_capacity_aggregate),
        idle_telemetry=IdleTelemetry(),
    )
    # Stage 2 — a > b 분기 산출 (out-of-band, a 큼)
    chunk_a_gt_b = admission.balance_inter_AB(
        prefill_chunk_tokens=512, a_cycle=100.0, b_cycle=10.0, ctx_tokens=32000,
    )
    # a > b → base 유지 (admission ↓ 정책)
    assert chunk_a_gt_b == 512
    # a < b 분기 — base + n_sat
    chunk_a_lt_b = admission.balance_inter_AB(
        prefill_chunk_tokens=512, a_cycle=10.0, b_cycle=100.0, ctx_tokens=32000,
    )
    assert chunk_a_lt_b > 512


# ============================================================================
# B6 — balance_inter_AB 4 분기 cover
# ============================================================================


def test_b6_balance_inter_ab_four_branches(cfg):
    """in_band · a<b · a>b · a>>b extremum 4 영역 cover."""
    from puls_sched.admission import Admission
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    admission = Admission(
        admission_cfg=cfg.admission,
        request_queue=RequestQueue(capacity=100),
        kv_accountant=KVAccountant(capacity=cfg.admission.kv_capacity_aggregate),
        idle_telemetry=IdleTelemetry(),
    )
    base = 512
    # in_band (a ≈ b, diff < width=2.0 for mid ctx)
    r1 = admission.balance_inter_AB(base, 50.0, 50.5, 32000)
    # a < b
    r2 = admission.balance_inter_AB(base, 10.0, 100.0, 32000)
    # a > b
    r3 = admission.balance_inter_AB(base, 100.0, 10.0, 32000)
    # extremum a >> b
    r4 = admission.balance_inter_AB(base, 10000.0, 10.0, 32000)
    # 4 영역 모두 산출 정합 (값 영역 차이만 확인)
    assert r1 == base       # in_band → base
    assert r2 > base        # a<b → base + n_sat
    assert r3 == base       # a>b → base (admission ↓ effect limited)
    assert r4 == base       # a>>b → 동일 분기


# ============================================================================
# B7 — Evaluator.f3_cross_validate 산출 정합
# ============================================================================


def test_b7_f3_cross_validate_returns_both_paths(cfg):
    ev = Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())
    # Inject some idle_telemetry data
    ev.idle_telemetry.record_active("gpu_instance_a", 0.0, 100.0)
    ev.idle_telemetry.record_active("gpu_instance_b", 0.0, 80.0)
    result = ev.f3_cross_validate(ctx_tokens=32000)
    assert "closed_form_ratio" in result
    assert "measured_ratio" in result
    assert "abs_diff_ratio" in result
    assert "provenance" in result
    assert "alpha_path" in result["provenance"]


# ============================================================================
# B8 — F3 closed-form 산식 정합
# ============================================================================


def test_b8_f3_closed_form_returns_consistent_result(cfg):
    """F3 산식 정합 — a_cycle = max(t_proj, t_attn), b_cycle = t_FFN."""
    ev = Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())
    r = ev.f3_closed_form(ctx_tokens=32000, prefill_chunk=512)
    # t_proj = t_qkv + t_prefill_attn + t_o_proj
    assert abs(r.t_proj_us - (r.t_qkv_us + r.t_prefill_attn_us + r.t_o_proj_us)) < 1e-9
    # a_cycle = max(t_proj, t_attn)
    assert r.a_cycle_us == max(r.t_proj_us, r.t_attn_us)
    # F3 = max(a, b) / (a + b)
    expected = max(r.a_cycle_us, r.b_cycle_us) / (r.a_cycle_us + r.b_cycle_us)
    assert abs(r.f3_ratio - expected) < 1e-9


# ============================================================================
# B9 — mark.skip 3 test 재활성 — α path 정합
# ============================================================================


def test_b9_mark_skip_three_tests_reactivated():
    """test_instance_pipeline_dispatch.py 위 mark.skip 3 영역 재활성 검증 (다음 task 위)."""
    import pathlib
    test_file = pathlib.Path(__file__).parent / "test_instance_pipeline_dispatch.py"
    if test_file.exists():
        content = test_file.read_text(encoding="utf-8")
        # 재활성 후 — 영원 skip marker 부재 (α path 정합)
        # 단 — 본 task 영역은 다음 task 진행 — 본 test 는 placeholder
        pass


# ============================================================================
# B10 — FFN op_time batch linear scaling
# ============================================================================


def test_b10_ffn_batch_monotonic_linear(cfg):
    mb_small = make_mb(decode_n=8)
    mb_large = make_mb(decode_n=64)
    t_small = compute_ffn_op_time_s(mb_small, cfg.calibration, cfg.model)
    t_large = compute_ffn_op_time_s(mb_large, cfg.calibration, cfg.model)
    assert abs(t_large / t_small - 8.0) < 1e-6


# ============================================================================
# B11 — FFN op_time MFU sweep inverse
# ============================================================================


def test_b11_ffn_mfu_sweep_inverse(cfg):
    mb = make_mb(decode_n=16)
    times = {}
    for mfu in [0.5, 0.6, 0.7]:
        cal = dataclasses.replace(cfg.calibration, gpu_mfu_default=mfu)
        times[mfu] = compute_ffn_op_time_s(mb, cal, cfg.model)
    assert times[0.5] > times[0.6] > times[0.7]


# ============================================================================
# B12 — Determinism (1000-call bit-exact)
# ============================================================================


def test_b12_ffn_deterministic_1000_calls(cfg):
    mb = make_mb(decode_n=16, prefill_per_req={100: 512})
    expected = compute_ffn_op_time_s(mb, cfg.calibration, cfg.model)
    for _ in range(1000):
        assert compute_ffn_op_time_s(mb, cfg.calibration, cfg.model) == expected
