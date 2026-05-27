import pytest

from puls_sched.config import default_dummy_config
from puls_sched.deadband import in_band, lookup_width


@pytest.fixture
def admission_cfg():
    return default_dummy_config().admission


@pytest.mark.parametrize("ctx", [1, 100, 8_000])
def test_lookup_width_short_tier(admission_cfg, ctx):
    assert lookup_width(admission_cfg, ctx) == admission_cfg.deadband_width["short"]


@pytest.mark.parametrize("ctx", [8_001, 16_000, 32_000])
def test_lookup_width_mid_tier(admission_cfg, ctx):
    assert lookup_width(admission_cfg, ctx) == admission_cfg.deadband_width["mid"]


@pytest.mark.parametrize("ctx", [32_001, 128_000, 1_000_000])
def test_lookup_width_long_tier(admission_cfg, ctx):
    assert lookup_width(admission_cfg, ctx) == admission_cfg.deadband_width["long"]


def test_tier_boundary_parametrize(admission_cfg):
    short_max = admission_cfg.ctx_tier_short_max
    mid_max = admission_cfg.ctx_tier_mid_max
    assert lookup_width(admission_cfg, short_max) == admission_cfg.deadband_width["short"]
    assert lookup_width(admission_cfg, short_max + 1) == admission_cfg.deadband_width["mid"]
    assert lookup_width(admission_cfg, mid_max) == admission_cfg.deadband_width["mid"]
    assert lookup_width(admission_cfg, mid_max + 1) == admission_cfg.deadband_width["long"]


@pytest.mark.parametrize("diff", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_in_band_when_diff_within_width(diff):
    assert in_band(diff, 2.0) is True


@pytest.mark.parametrize("diff", [-3.0, 3.0, 100.0])
def test_out_band_when_diff_exceeds_width(diff):
    assert in_band(diff, 2.0) is False


def test_in_band_boundary_inclusive():
    assert in_band(2.0, 2.0) is True
    assert in_band(-2.0, 2.0) is True


def test_deadband_ordering_preserved_in_lookup(admission_cfg):
    short = admission_cfg.deadband_width["short"]
    mid = admission_cfg.deadband_width["mid"]
    long = admission_cfg.deadband_width["long"]
    assert short < mid < long


def test_in_band_width_zero_only_exact_diff():
    """Edge: width=0 → only diff==0 in-band. Impl-10 swap fragility prefigure."""
    assert in_band(0.0, 0.0) is True
    assert in_band(0.001, 0.0) is False
    assert in_band(-0.001, 0.0) is False
