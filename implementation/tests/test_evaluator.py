"""Impl-8 cluster A — Evaluator 7 산출 method + 2 callback method 단위 test."""
import math

import pytest

from puls_sched.clock import Clock
from puls_sched.evaluator import (
    AblationSource,
    AdmissionSnapshot,
    ConvergenceVerdict,
    DecompositionCell,
    DispatchEvent,
    Evaluator,
)
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.node import NodeType


# =========================================================================
# dispatch_trace
# =========================================================================

def test_evaluator_dispatch_trace_empty(evaluator):
    assert evaluator.dispatch_trace() == ()


def test_evaluator_dispatch_trace_single_event(evaluator):
    ev = DispatchEvent(
        timestamp=1.0, micro_batch_id=0, node_type=NodeType.QKV,
        resource="GPU", k_total=0, dag_state_snapshot={},
    )
    evaluator.record_dispatch(ev)
    trace = evaluator.dispatch_trace()
    assert len(trace) == 1
    assert trace[0] == ev


def test_evaluator_dispatch_trace_immutable_snapshot(evaluator):
    """반환 tuple immutable + DispatchEvent frozen → mutation 시 raise."""
    ev = DispatchEvent(
        timestamp=1.0, micro_batch_id=0, node_type=NodeType.QKV,
        resource="GPU", k_total=0, dag_state_snapshot={},
    )
    evaluator.record_dispatch(ev)
    trace = evaluator.dispatch_trace()
    # tuple immutable
    with pytest.raises((TypeError, AttributeError)):
        trace[0] = None  # type: ignore
    # DispatchEvent frozen
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError or AttributeError
        trace[0].timestamp = 99.0  # type: ignore


# =========================================================================
# admission_convergence
# =========================================================================

def test_evaluator_admission_convergence_empty(evaluator):
    v = evaluator.admission_convergence()
    assert v == ConvergenceVerdict(converged=False, oscillating=False, in_band_fraction=0.0, samples=0)


def test_evaluator_admission_convergence_all_in_band(evaluator, dummy_config):
    """10 snapshot 모두 in_band → converged=True, oscillating=False, in_band_fraction=1.0"""
    width = dummy_config.admission.deadband_width["short"]
    for i in range(10):
        snap = AdmissionSnapshot(
            timestamp=float(i), gpu_idle_fraction=0.2, pim_idle_fraction=0.2,
            a_cycle=10.0, b_cycle=10.0 + width * 0.5,  # diff < width
            ctx_tokens=4000, spec_admitted=True, n=1, k_total=256,
        )
        evaluator.record_admission_tick(snap)
    v = evaluator.admission_convergence()
    assert v.converged is True
    assert v.oscillating is False
    assert v.in_band_fraction == 1.0
    assert v.samples == 10


def test_evaluator_admission_convergence_sin_wave_oscillating(evaluator):
    """합성 sin wave (a-b 의 sign 자주 flip) → oscillating=True"""
    # 10 snapshot: a-b 가 +/+/-/- ... 큰 swing
    for i in range(10):
        sign = 1.0 if (i % 2 == 0) else -1.0
        snap = AdmissionSnapshot(
            timestamp=float(i), gpu_idle_fraction=0.2, pim_idle_fraction=0.2,
            a_cycle=100.0 + sign * 50.0, b_cycle=100.0,
            ctx_tokens=4000, spec_admitted=True, n=1, k_total=256,
        )
        evaluator.record_admission_tick(snap)
    v = evaluator.admission_convergence()
    # sign change count = 9 (every step), n-1 = 9, ratio = 1.0 ≥ 0.4
    assert v.oscillating is True


def test_evaluator_admission_convergence_monotonic_convergence(evaluator, dummy_config):
    """a-b 가 100 → 10 → 1 → 0.5 → 0.1 ... → 마지막 in_band 도달 → converged=True"""
    width = dummy_config.admission.deadband_width["short"]
    diffs = [100.0, 50.0, 10.0, 5.0, 1.0, width * 0.5, width * 0.4, width * 0.3, width * 0.2, width * 0.1]
    for i, diff in enumerate(diffs):
        snap = AdmissionSnapshot(
            timestamp=float(i), gpu_idle_fraction=0.2, pim_idle_fraction=0.2,
            a_cycle=100.0 + diff, b_cycle=100.0,
            ctx_tokens=4000, spec_admitted=True, n=1, k_total=256,
        )
        evaluator.record_admission_tick(snap)
    v = evaluator.admission_convergence()
    assert v.converged is True
    assert v.oscillating is False


def test_evaluator_admission_convergence_admitted_and_not_both_captured(evaluator):
    """spec=None (admission 실패) 도 snapshot 누적 — empty admission tick capture invariant"""
    for spec_admitted in [True, False, True, False, True]:
        snap = AdmissionSnapshot(
            timestamp=0.0, gpu_idle_fraction=0.0, pim_idle_fraction=0.0,
            a_cycle=1.0, b_cycle=1.0, ctx_tokens=4000,
            spec_admitted=spec_admitted, n=1 if spec_admitted else 0, k_total=256 if spec_admitted else 0,
        )
        evaluator.record_admission_tick(snap)
    v = evaluator.admission_convergence()
    assert v.samples == 5  # admitted + rejected 모두


# =========================================================================
# idle_fraction
# =========================================================================

def test_evaluator_idle_fraction_keys(evaluator):
    """key set lock-in — Impl-10-pre-1 O8.1 (3-key) schema 정합."""
    result = evaluator.idle_fraction()
    assert set(result.keys()) == {"gpu_instance_a", "pim_instance_a", "gpu_instance_b"}


def test_evaluator_idle_fraction_values_bit_exact_telemetry(evaluator, idle_telemetry):
    """Post-hoc snapshot lock-in (D1) — idle_telemetry 의 값과 bit-exact 일치."""
    # 합성 active duration 주입
    idle_telemetry.reset(0.0)
    idle_telemetry.record_active("GPU", 0.0, 5.0)
    idle_telemetry.record_active("PIM", 0.0, 3.0)
    # Impl-10-pre-1 — gpu_instance_b 신설 slot 도 inject
    idle_telemetry.record_active("gpu_instance_b", 0.0, 2.0)
    result = evaluator.idle_fraction()
    assert result["gpu_instance_a"] == idle_telemetry.gpu_idle_fraction()
    assert result["pim_instance_a"] == idle_telemetry.pim_idle_fraction()
    assert result["gpu_instance_b"] == idle_telemetry.gpu_instance_b_idle_fraction()


# =========================================================================
# pim_utilization
# =========================================================================

def test_evaluator_pim_utilization_no_dispatch(evaluator):
    assert evaluator.pim_utilization() == 0.0


def test_evaluator_pim_utilization_formula(evaluator, clock):
    """합성 PIM dispatch sequence 위 산식 정합 — Σ k·dt / (k_max · total_time)"""
    # PIM dispatch at t=0 with k=1024, then at t=2.0 with k=2048
    # Between t=0 and t=2.0: k=1024 active for dt=2.0 → contribution 1024 × 2.0 = 2048
    # k_max=2048, total_time=2.0 (clock.now needs to be > 2.0)
    clock.advance_to(2.0)
    evaluator.record_dispatch(DispatchEvent(
        timestamp=0.0, micro_batch_id=0, node_type=NodeType.DECODE_ATTN,
        resource="PIM", k_total=1024, dag_state_snapshot={},
    ))
    evaluator.record_dispatch(DispatchEvent(
        timestamp=2.0, micro_batch_id=1, node_type=NodeType.DECODE_ATTN,
        resource="PIM", k_total=2048, dag_state_snapshot={},
    ))
    # Σ k·dt = 1024 × 2.0 = 2048 (마지막 dispatch 의 dt 는 미반영 — O8.2 carry-over)
    # k_max=2048, total_time = clock.now (2.0) - first dispatch (0.0) = 2.0
    # utilization = 2048 / (2048 × 2.0) = 0.5
    util = evaluator.pim_utilization()
    assert util == pytest.approx(0.5)


# =========================================================================
# pipeline_efficiency
# =========================================================================

def test_evaluator_pipeline_efficiency_balanced(evaluator):
    assert evaluator.pipeline_efficiency(10.0, 10.0) == 0.5


def test_evaluator_pipeline_efficiency_2_to_1(evaluator):
    assert evaluator.pipeline_efficiency(20.0, 10.0) == pytest.approx(2.0 / 3.0)


def test_evaluator_pipeline_efficiency_zero_raises(evaluator):
    with pytest.raises(ValueError):
        evaluator.pipeline_efficiency(0.0, 10.0)
    with pytest.raises(ValueError):
        evaluator.pipeline_efficiency(10.0, 0.0)


def test_evaluator_pipeline_efficiency_negative_raises(evaluator):
    with pytest.raises(ValueError):
        evaluator.pipeline_efficiency(-1.0, 10.0)


# =========================================================================
# acceleration_decomposition
# =========================================================================

def test_evaluator_acceleration_decomposition_4_sources(evaluator):
    """4 cell == {F1, F2, F3, F5} — F4 미포함 (ARCH §5.7 precondition 영역)"""
    decomp = evaluator.acceleration_decomposition(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    assert len(decomp) == 4
    sources = {cell.source for cell in decomp}
    assert sources == {AblationSource.F1, AblationSource.F2, AblationSource.F3, AblationSource.F5}


def test_evaluator_acceleration_decomposition_f3_formula(evaluator):
    """F3 cell: a=10, b=20 → with=max(10,20)=20, without=10+20=30, ratio=1.5"""
    decomp = evaluator.acceleration_decomposition(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    f3 = next(c for c in decomp if c.source == AblationSource.F3)
    assert f3.cycle_with_source == 20.0
    assert f3.cycle_without_source == 30.0
    assert f3.ratio == 1.5
    assert f3.direction_positive is True


def test_evaluator_acceleration_decomposition_f2_formula(evaluator):
    """F2 cell: t_proj=10, t_pim=20 → with=max(10,20)=20, without=10+20=30"""
    decomp = evaluator.acceleration_decomposition(a_cycle=1, b_cycle=1, t_pim=20, t_proj=10)
    f2 = next(c for c in decomp if c.source == AblationSource.F2)
    assert f2.cycle_with_source == 20.0
    assert f2.cycle_without_source == 30.0
    assert f2.ratio == 1.5


# =========================================================================
# report
# =========================================================================

def test_evaluator_report_dict_schema(evaluator):
    result = evaluator.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    assert set(result.keys()) == {
        "dispatch_trace", "convergence", "idle_fraction", "pim_utilization",
        "pipeline_efficiency", "acceleration_decomposition", "markdown", "ablation_config",
    }


def test_evaluator_report_markdown_contains_ablation_flags(evaluator):
    """PLAN §0.5 출처 라벨 — markdown 에 F1·F2·F3·F5 flag 명시"""
    result = evaluator.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    md = result["markdown"]
    assert "F1" in md
    assert "F2" in md
    assert "F3" in md
    assert "F5" in md
    assert "Ablation" in md  # provenance section header


def test_evaluator_report_no_absolute_metric_keys(evaluator):
    """PLAN §1.2 lock-in — Comparative baseline / 절대 metric key 부재"""
    result = evaluator.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    forbidden = {"ttft", "tpot", "throughput", "goodput", "slo", "baseline", "sarathi", "vllm"}
    for key in result.keys():
        assert not any(f in key.lower() for f in forbidden), f"key {key} contains forbidden term"


def test_evaluator_report_deterministic_same_input(dummy_config, idle_telemetry):
    """동일 fixture (record sequence 동일) → report() bit-exact (PLAN §0 C5)"""
    c1, c2 = Clock(), Clock()
    e1 = Evaluator(config=dummy_config, clock=c1, idle_telemetry=IdleTelemetry())
    e2 = Evaluator(config=dummy_config, clock=c2, idle_telemetry=IdleTelemetry())
    r1 = e1.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    r2 = e2.report(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    # markdown 은 동일 input → 동일 string
    assert r1["markdown"] == r2["markdown"]
    assert r1["acceleration_decomposition"] == r2["acceleration_decomposition"]
    assert r1["pipeline_efficiency"] == r2["pipeline_efficiency"]
