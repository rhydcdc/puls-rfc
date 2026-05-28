"""Impl-8 cluster C — admission_convergence() heuristic 합성 oscillation / 수렴 / boundary."""
import pytest

from puls_sched.evaluator import AdmissionSnapshot


def _snap(t, a, b, ctx=4000):
    """Helper — synthetic snapshot 생성."""
    return AdmissionSnapshot(
        timestamp=t, gpu_idle_fraction=0.0, pim_idle_fraction=0.0,
        a_cycle=a, b_cycle=b, ctx_tokens=ctx,
        spec_admitted=True, n=1,
    )


def test_convergence_synth_monotonic_decreasing_diff(evaluator, dummy_config):
    """a-b 100 → 50 → 10 → 1 → 0.5 → ... → 마지막 5 모두 in_band → converged=True."""
    width = dummy_config.admission.deadband_width["short"]
    diffs = [100.0, 50.0, 10.0, 5.0, 1.0, width * 0.5, width * 0.4, width * 0.3, width * 0.2, width * 0.1]
    for i, d in enumerate(diffs):
        evaluator.record_admission_tick(_snap(float(i), 100.0 + d, 100.0))
    v = evaluator.admission_convergence()
    assert v.converged is True
    # diff monotonic decrease + always same sign (a > b) → no sign change → not oscillating
    assert v.oscillating is False


def test_convergence_synth_sin_wave_oscillation(evaluator):
    """a-b 가 +/-/+/- ... 큰 swing → oscillating=True."""
    for i in range(10):
        sign = 1.0 if (i % 2 == 0) else -1.0
        evaluator.record_admission_tick(_snap(float(i), 100.0 + sign * 50.0, 100.0))
    v = evaluator.admission_convergence()
    assert v.oscillating is True
    # 큰 swing → in_band 안 됨 → converged=False
    assert v.converged is False


def test_convergence_synth_all_out_of_band(evaluator):
    """a-b 가 100 으로 고정 (deadband short=1.0 초과) → in_band_fraction=0.0."""
    for i in range(10):
        evaluator.record_admission_tick(_snap(float(i), 200.0, 100.0))
    v = evaluator.admission_convergence()
    assert v.in_band_fraction == 0.0
    assert v.converged is False
    assert v.oscillating is False  # sign 일정 (a > b)


def test_convergence_synth_all_in_band(evaluator):
    """a-b = 0 으로 고정 → in_band_fraction=1.0, converged=True."""
    for i in range(10):
        evaluator.record_admission_tick(_snap(float(i), 100.0, 100.0))
    v = evaluator.admission_convergence()
    assert v.in_band_fraction == 1.0
    assert v.converged is True


def test_convergence_tail_window_size_3(evaluator, dummy_config):
    """snapshot 3 만 — tail_n=3 위 converged 판정 (n < 5 boundary)."""
    width = dummy_config.admission.deadband_width["short"]
    for i in range(3):
        evaluator.record_admission_tick(_snap(float(i), 100.0 + width * 0.1, 100.0))
    v = evaluator.admission_convergence()
    assert v.samples == 3
    assert v.converged is True  # 3/3 in_band 100% ≥ 0.8


def test_convergence_sign_change_count_correct(evaluator):
    """합성 [+, +, -, -, +] 위 sign change == 2 (n-1=4, ratio=0.5 ≥ 0.4 → oscillating)."""
    diffs = [10.0, 10.0, -10.0, -10.0, 10.0]
    for i, d in enumerate(diffs):
        evaluator.record_admission_tick(_snap(float(i), 100.0 + d, 100.0))
    v = evaluator.admission_convergence()
    assert v.oscillating is True


def test_convergence_ctx_tier_deadband_width_consumed(evaluator, dummy_config):
    """ctx tier 별 width lookup 정합 — 동일 diff 위 short / mid / long 위 in_band 다름."""
    # diff = 1.5: short(1.0) → out, mid(2.0) → in, long(3.0) → in
    cfg = dummy_config.admission
    width_short = cfg.deadband_width["short"]  # 1.0
    width_mid = cfg.deadband_width["mid"]      # 2.0
    diff = (width_short + width_mid) / 2.0     # 1.5 → short 영역 out, mid 영역 in
    evaluator.record_admission_tick(_snap(0.0, 100.0 + diff, 100.0, ctx=cfg.ctx_tier_short_max - 100))  # short
    evaluator.record_admission_tick(_snap(1.0, 100.0 + diff, 100.0, ctx=cfg.ctx_tier_mid_max - 100))    # mid
    v = evaluator.admission_convergence()
    assert v.in_band_fraction == 0.5  # 1/2 (mid 만 in_band)


def test_convergence_admitted_vs_not_admitted_both_captured(evaluator):
    """spec=None (admission 실패) 도 snapshot 누적 (empty admission tick — D1 hybrid)."""
    for i, admitted in enumerate([True, False, True, False, True]):
        snap = AdmissionSnapshot(
            timestamp=float(i), gpu_idle_fraction=0.0, pim_idle_fraction=0.0,
            a_cycle=100.0, b_cycle=100.0, ctx_tokens=4000,
            spec_admitted=admitted, n=1 if admitted else 0,
        )
        evaluator.record_admission_tick(snap)
    v = evaluator.admission_convergence()
    assert v.samples == 5  # admitted + rejected 모두 capture


def test_convergence_deterministic_replay(dummy_config, idle_telemetry):
    """동일 snapshot sequence 2 회 → ConvergenceVerdict bit-exact (PLAN §0 C5)."""
    from puls_sched.clock import Clock
    from puls_sched.evaluator import Evaluator
    from puls_sched.idle_telemetry import IdleTelemetry
    e1 = Evaluator(config=dummy_config, clock=Clock(), idle_telemetry=IdleTelemetry())
    e2 = Evaluator(config=dummy_config, clock=Clock(), idle_telemetry=IdleTelemetry())
    for i in range(10):
        s = _snap(float(i), 100.0 + i, 100.0)
        e1.record_admission_tick(s)
        e2.record_admission_tick(s)
    assert e1.admission_convergence() == e2.admission_convergence()


def test_convergence_zero_diff_no_sign_change(evaluator):
    """a-b == 0 인 snapshot → sign 0 (sign change 누적 안 함)."""
    # 모두 0 diff → sign_changes = 0 → oscillating False
    for i in range(5):
        evaluator.record_admission_tick(_snap(float(i), 100.0, 100.0))
    v = evaluator.admission_convergence()
    assert v.oscillating is False
