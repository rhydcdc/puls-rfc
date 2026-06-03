"""Impl-8 cluster A — Evaluator 7 산출 method + 2 callback method 단위 test."""
import math

import pytest

from puls_sched.clock import Clock
from puls_sched.evaluator import (
    AblationSource,
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
        resource="GPU", dag_state_snapshot={},
    )
    evaluator.record_dispatch(ev)
    trace = evaluator.dispatch_trace()
    assert len(trace) == 1
    assert trace[0] == ev


def test_evaluator_dispatch_trace_immutable_snapshot(evaluator):
    """반환 tuple immutable + DispatchEvent frozen → mutation 시 raise."""
    ev = DispatchEvent(
        timestamp=1.0, micro_batch_id=0, node_type=NodeType.QKV,
        resource="GPU", dag_state_snapshot={},
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
    """합성 PIM dispatch sequence 위 산식 정합 — PIM busy time / total window time.

    Impl-10-pre-2 — k_total knob 폐기 위 PIM utilization = 단순 time fraction
    (sequence-parallel PIM 위 임의 시점 한 mb 가 모든 채널 점유).

    Setup: PIM dispatch at t=0, second dispatch at t=2.0, clock.now=2.0.
    PIM busy span = 2.0 - 0.0 = 2.0. total_window = 2.0 - 0.0 = 2.0. utilization = 1.0.
    """
    clock.advance_to(2.0)
    evaluator.record_dispatch(DispatchEvent(
        timestamp=0.0, micro_batch_id=0, node_type=NodeType.DECODE_ATTN,
        resource="PIM", dag_state_snapshot={},
    ))
    evaluator.record_dispatch(DispatchEvent(
        timestamp=2.0, micro_batch_id=1, node_type=NodeType.DECODE_ATTN,
        resource="PIM", dag_state_snapshot={},
    ))
    util = evaluator.pim_utilization()
    assert util == pytest.approx(1.0)


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
        "dispatch_trace", "idle_fraction", "pim_utilization",
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
