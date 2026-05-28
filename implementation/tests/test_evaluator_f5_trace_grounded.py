"""Cluster F5 — Evaluator.f5_trace_grounded 산출 정합.

PLAN §4 Impl-10 main + impl_10.md §5 Cluster F5. ARCH §5.7 F5.
"""

import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.evaluator import Evaluator
from puls_sched.idle_telemetry import IdleTelemetry


@pytest.fixture
def ev():
    cfg = default_dummy_config()
    return Evaluator(config=cfg, clock=Clock(), idle_telemetry=IdleTelemetry())


# ============================================================================
# F5.1 — F5 formula = tiles_indep / tiles_lockstep
# ============================================================================


def test_f5_1_formula(ev):
    kv_lengths = [128000, 100000, 64000, 50000, 30000]
    r = ev.f5_trace_grounded(kv_lengths)
    # tiles_indep / tiles_lockstep
    assert r.f5_ratio == r.tiles_channel_independent / r.tiles_lockstep_max_kv


# ============================================================================
# F5.2 — Uniform KV → ratio = 1.0 (penalty 0)
# ============================================================================


def test_f5_2_uniform_kv_ratio_one(ev):
    """모든 req kv 동일 위 sum = max × N → ratio = 1.0."""
    r = ev.f5_trace_grounded([128000] * 5)
    assert r.f5_ratio == 1.0


# ============================================================================
# F5.3 — Skewed KV → ratio < 1.0 (lock-step penalty 활성)
# ============================================================================


def test_f5_3_skewed_kv_ratio_less_than_one(ev):
    """max ≫ mean 위 lock-step penalty 활성."""
    r = ev.f5_trace_grounded([128000, 1000, 1000, 1000, 1000])
    assert r.f5_ratio < 1.0


# ============================================================================
# F5.4 — Trace 위 첫 10 req kv_lengths 산출 정합 + R1 lock-in
# ============================================================================


def test_f5_4_trace_kv_distribution_lock_in(ev):
    """λ=3.40 첫 10 req kv_length 영원 산출 정합 — R1 lock-in regression."""
    from pathlib import Path
    from puls_sched.trace import TraceReplayer
    trace_path = Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"
    if not trace_path.exists():
        pytest.skip("trace file not present")
    replayer = TraceReplayer.load(trace_path)
    # kv_length = num_prefill + num_decode (ARCH §3.3 full reservation)
    kvs = [e.num_prefill_tokens + e.num_decode_tokens for e in replayer.entries[:10]]
    assert len(kvs) == 10
    r = ev.f5_trace_grounded(kvs)
    assert r.n_requests == 10
    assert r.sum_kv == sum(kvs)
    assert r.max_kv == max(kvs)


# ============================================================================
# F5.5 — N=1 → ratio = 1.0 (penalty 의미 0)
# ============================================================================


def test_f5_5_single_request_ratio_one(ev):
    r = ev.f5_trace_grounded([50000])
    assert r.f5_ratio == 1.0


# ============================================================================
# F5.6 — Empty list 위 zero handling
# ============================================================================


def test_f5_6_empty_list_returns_zero_ratio(ev):
    r = ev.f5_trace_grounded([])
    assert r.f5_ratio == 0.0
    assert r.n_requests == 0


# ============================================================================
# F5.7 — Determinism
# ============================================================================


def test_f5_7_determinism(ev):
    kvs = [100000, 50000, 30000]
    expected = ev.f5_trace_grounded(kvs)
    for _ in range(1000):
        assert ev.f5_trace_grounded(kvs) == expected


# ============================================================================
# F5.8 — Provenance label
# ============================================================================


def test_f5_8_provenance(ev):
    r = ev.f5_trace_grounded([100000])
    assert "longbench" in r.provenance
