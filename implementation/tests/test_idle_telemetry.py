import pytest

from puls_sched.idle_telemetry import IdleTelemetry


def test_idle_fraction_zero_when_fully_active():
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 10.0)
    assert tel.gpu_idle_fraction() == 0.0


def test_idle_fraction_half_when_half_active():
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 5.0)
    # window_end advances to 5.0 via record. Need explicit window extension to 10:
    tel.record_active("GPU", 10.0, 10.0)  # zero-duration tag → extends window_end
    assert tel.gpu_idle_fraction() == pytest.approx(0.5)


def test_reproducibility_same_sequence_same_fraction():
    seq = [("GPU", 0.0, 2.0), ("GPU", 3.0, 5.0), ("PIM", 1.0, 4.0)]
    a = IdleTelemetry()
    b = IdleTelemetry()
    a.reset(0.0)
    b.reset(0.0)
    for resource, t0, t1 in seq:
        a.record_active(resource, t0, t1)
        b.record_active(resource, t0, t1)
    assert a.gpu_idle_fraction() == b.gpu_idle_fraction()
    assert a.pim_idle_fraction() == b.pim_idle_fraction()


def test_accumulation_multiple_record_calls():
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 2.0)
    tel.record_active("GPU", 3.0, 5.0)
    tel.record_active("GPU", 7.0, 9.0)
    # active total = 6, window_end = 9, span = 9 → idle = 1 - 6/9
    assert tel.gpu_idle_fraction() == pytest.approx(1.0 - 6.0 / 9.0)


def test_gpu_and_pim_independent():
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 5.0)
    tel.record_active("GPU", 5.0, 10.0)  # establishes window_end=10 via GPU activity
    assert tel.gpu_idle_fraction() == 0.0
    assert tel.pim_idle_fraction() == 1.0


def test_record_active_invalid_timerange_raises():
    tel = IdleTelemetry()
    tel.reset(0.0)
    with pytest.raises(ValueError, match="t_end"):
        tel.record_active("GPU", 5.0, 1.0)


def test_record_active_unknown_resource_raises():
    tel = IdleTelemetry()
    tel.reset(0.0)
    with pytest.raises(ValueError, match="unknown resource"):
        tel.record_active("CPU", 0.0, 1.0)


def test_reset_clears_accumulators():
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 5.0)
    tel.reset(100.0)
    tel.record_active("GPU", 100.0, 110.0)
    assert tel.gpu_idle_fraction() == 0.0  # fully active over new window


@pytest.mark.parametrize("seq", [
    [("GPU", 0.0, 1.0)],
    [("GPU", 0.0, 3.0), ("PIM", 0.0, 2.0), ("GPU", 5.0, 7.0)],
    [("PIM", 0.0, 10.0)],
])
def test_fraction_bounded_0_to_1(seq):
    tel = IdleTelemetry()
    tel.reset(0.0)
    for resource, t0, t1 in seq:
        tel.record_active(resource, t0, t1)
    g = tel.gpu_idle_fraction()
    p = tel.pim_idle_fraction()
    assert 0.0 <= g <= 1.0
    assert 0.0 <= p <= 1.0


def test_idle_fraction_span_zero_returns_zero():
    """Edge: window_start == window_end (no time elapsed) → fraction 0.0 (div-by-zero guard)."""
    tel = IdleTelemetry()
    tel.reset(5.0)
    assert tel.gpu_idle_fraction() == 0.0
    assert tel.pim_idle_fraction() == 0.0


def test_zero_duration_record_extends_window_only():
    """Zero-duration record (t_end == t_start) is the documented mechanism for
    extending window_end without adding active duration."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 10.0, 10.0)
    assert tel.gpu_idle_fraction() == 1.0  # span=10, active=0 → fully idle
