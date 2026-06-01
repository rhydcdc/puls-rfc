"""Impl-6 — TraceReplayer + TraceEntry + TraceStats unit tests.

PLAN §4 Impl-6 + §0.5 reminder + R1 (stats lock-in) + R6 (generator semantic) + R10 (extreme rate).
"""

from pathlib import Path

import pytest

from puls_sched.trace import (
    _PHASE_3_SCHEMAS,
    _SUPPORTED_SCHEMAS,
    TraceEntry,
    TraceReplayer,
    TraceStats,
)


FIXTURES = Path(__file__).parent / "test_trace_fixtures"
DATA = Path(__file__).parent.parent / "data"

REAL_3_40 = DATA / "longctx_longbench_lambda_3_40.csv"
REAL_6_67 = DATA / "longctx_longbench_lambda_6_67.csv"


# ============================================================================
# load — schema validation + malformed cross-product
# ============================================================================

def test_trace_load_valid_minimal():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    assert len(r.entries) == 3
    assert r.entries[0] == TraceEntry(0.1, 100, 10)
    assert r.entries[1] == TraceEntry(0.5, 200, 20)
    assert r.entries[2] == TraceEntry(1.0, 150, 15)


def test_trace_load_real_longbench_3_40():
    r = TraceReplayer.load(REAL_3_40)
    assert len(r.entries) == 12_279
    # first / last row bit-exact
    assert r.entries[0].num_prefill_tokens == 47102
    assert r.entries[0].num_decode_tokens == 350
    assert r.entries[-1].num_prefill_tokens == 157573
    assert r.entries[-1].num_decode_tokens == 350


def test_trace_load_real_longbench_6_67():
    r = TraceReplayer.load(REAL_6_67)
    assert len(r.entries) == 24_054
    assert r.entries[0].num_prefill_tokens == 23771
    assert r.entries[-1].num_prefill_tokens == 128517


def test_trace_load_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TraceReplayer.load(tmp_path / "does_not_exist.csv")


def test_trace_load_empty_csv_raises():
    with pytest.raises(ValueError, match="empty"):
        TraceReplayer.load(FIXTURES / "empty.csv")


def test_trace_load_missing_field_raises():
    with pytest.raises(ValueError, match="missing required fields"):
        TraceReplayer.load(FIXTURES / "missing_field.csv")


def test_trace_load_wrong_type_raises():
    with pytest.raises(ValueError, match="type error"):
        TraceReplayer.load(FIXTURES / "wrong_type.csv")


def test_trace_load_negative_value_raises():
    with pytest.raises(ValueError, match="negative"):
        TraceReplayer.load(FIXTURES / "negative_value.csv")


def test_trace_load_duplicate_arrival_allowed():
    r = TraceReplayer.load(FIXTURES / "duplicate_arrival.csv")
    assert len(r.entries) == 3
    assert r.entries[0].arrived_at == r.entries[1].arrived_at == 0.1


def test_trace_load_unsupported_schema_raises():
    with pytest.raises(ValueError, match="unsupported schema"):
        TraceReplayer.load(FIXTURES / "valid_minimal.csv", schema="unknown")


@pytest.mark.parametrize("phase_3_schema", ["vidur_1m_class", "mooncake_chat"])
def test_trace_load_phase_3_schema_raises(phase_3_schema):
    with pytest.raises(NotImplementedError, match="Phase 3"):
        TraceReplayer.load(FIXTURES / "valid_minimal.csv", schema=phase_3_schema)


# ============================================================================
# replay — yield + max_tokens · kv_length · arrival scaling
# ============================================================================

def test_trace_replay_yields_requests():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay())
    assert len(reqs) == 3
    assert [req.id for req in reqs] == [0, 1, 2]


def test_trace_replay_max_tokens_from_decode_count():
    """Q6 (c) hybrid — Request.max_tokens == TraceEntry.num_decode_tokens"""
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay())
    assert reqs[0].max_tokens == 10
    assert reqs[1].max_tokens == 20
    assert reqs[2].max_tokens == 15


def test_trace_replay_kv_length_full_reservation():
    """ARCH §3.3 — kv_length = num_prefill + num_decode (full reservation at admission)"""
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay())
    assert reqs[0].kv_length == 110  # 100 + 10
    assert reqs[1].kv_length == 220
    assert reqs[2].kv_length == 165


@pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 10.0])
def test_trace_replay_arrival_scaling(rate):
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay(rate_multiplier=rate))
    assert reqs[0].arrival_time == pytest.approx(0.1 / rate)
    assert reqs[1].arrival_time == pytest.approx(0.5 / rate)
    assert reqs[2].arrival_time == pytest.approx(1.0 / rate)


def test_trace_replay_rate_zero_raises():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    with pytest.raises(ValueError, match="positive"):
        list(r.replay(rate_multiplier=0.0))


def test_trace_replay_rate_negative_raises():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    with pytest.raises(ValueError, match="positive"):
        list(r.replay(rate_multiplier=-1.0))


def test_trace_replay_prompt_len():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay())
    # S4(a) — prompt_len = num_prefill_tokens (토큰 내용 미materialize, OOM 회피)
    assert reqs[0].prompt_len == 100


# ============================================================================
# stats — basic + boundary
# ============================================================================

def test_trace_stats_basic():
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    s = r.stats()
    assert s.n_entries == 3
    assert s.kv_length_min == 110  # 100+10
    assert s.kv_length_max == 220  # 200+20
    assert s.kv_length_mean == pytest.approx((110 + 220 + 165) / 3)
    # intervals: 0.5-0.1=0.4, 1.0-0.5=0.5 → mean=0.45
    assert s.arrival_interval_mean == pytest.approx(0.45)


def test_trace_stats_empty_raises():
    r = TraceReplayer(entries=())
    with pytest.raises(RuntimeError, match="empty"):
        r.stats()


def test_trace_stats_single_entry_zero_interval():
    r = TraceReplayer(entries=(TraceEntry(0.5, 100, 10),))
    s = r.stats()
    assert s.n_entries == 1
    assert s.arrival_interval_mean == 0.0
    assert s.arrival_interval_std == 0.0


# ============================================================================
# Q4 KS test — load → replay round-trip (자기 일치, sanity)
# ============================================================================

def test_trace_load_replay_roundtrip_distribution_kv():
    """Q4 — replay 가 entries 를 그대로 yield → KV length 분포 자기 일치 (D=0)"""
    r = TraceReplayer.load(REAL_3_40)
    original_kvs = sorted(e.num_prefill_tokens + e.num_decode_tokens for e in r.entries)
    replayed_kvs = sorted(req.kv_length for req in r.replay())
    assert original_kvs == replayed_kvs  # bit-exact


def test_trace_load_replay_roundtrip_distribution_arrival():
    """Q4 — replay(rate=1.0) → arrival_time 자기 일치 (D=0)"""
    r = TraceReplayer.load(REAL_3_40)
    original = sorted(e.arrived_at for e in r.entries)
    replayed = sorted(req.arrival_time for req in r.replay(rate_multiplier=1.0))
    assert original == replayed


# ============================================================================
# Determinism
# ============================================================================

def test_trace_load_deterministic_1000():
    """동일 path 1000 회 load → entries bit-exact"""
    first = TraceReplayer.load(FIXTURES / "valid_minimal.csv").entries
    for _ in range(999):
        r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
        assert r.entries == first


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 1000])
def test_trace_replay_deterministic_multiseed(seed):
    """seed independence — TraceReplayer 의 RNG 의존 0 (PLAN §0 C5)"""
    import random
    random.seed(seed)
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs1 = [req.kv_length for req in r.replay()]
    random.seed(seed + 1)  # 다른 seed 위 동일 결과
    reqs2 = [req.kv_length for req in r.replay()]
    assert reqs1 == reqs2


def test_trace_replay_multi_iter_same_replayer_bit_exact():
    """동일 TraceReplayer 위 replay() 두 번 호출 → 두 iterator 의 동일 sequence"""
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    out1 = [req.kv_length for req in r.replay()]
    out2 = [req.kv_length for req in r.replay()]
    assert out1 == out2


# ============================================================================
# Q3 source label lock-in
# ============================================================================

def test_trace_source_label_longbench():
    assert "longbench" in REAL_3_40.stem
    assert "longbench" in REAL_6_67.stem


# ============================================================================
# R1 보강 — 실 trace stats 값 lock-in (regression detection)
# ============================================================================

def test_trace_stats_real_trace_values_lock_in_3_40():
    r = TraceReplayer.load(REAL_3_40)
    s = r.stats()
    assert s.n_entries == 12_279
    assert s.kv_length_min == 12190
    assert s.kv_length_max == 5_747_830
    assert s.kv_length_mean == pytest.approx(323232.083150, rel=1e-6)
    assert s.arrival_interval_mean == pytest.approx(0.293145, rel=1e-4)
    assert s.arrival_interval_std == pytest.approx(0.293615, rel=1e-4)


def test_trace_stats_real_trace_values_lock_in_6_67():
    r = TraceReplayer.load(REAL_6_67)
    s = r.stats()
    assert s.n_entries == 24_054
    assert s.kv_length_min == 12190
    assert s.kv_length_max == 5_747_830
    assert s.kv_length_mean == pytest.approx(314838.090962, rel=1e-6)
    assert s.arrival_interval_mean == pytest.approx(0.149651, rel=1e-4)
    assert s.arrival_interval_std == pytest.approx(0.150534, rel=1e-4)


# ============================================================================
# R6 보강 — generator semantic lock-in
# ============================================================================

def test_trace_replay_generator_one_shot_semantic():
    """Python generator semantic — 1회 소진 후 빈 list"""
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    it = r.replay()
    first = list(it)
    second = list(it)  # already exhausted
    assert len(first) == 3
    assert second == []


def test_trace_replay_called_twice_returns_fresh_generator():
    """각 호출 = fresh generator (entries 의 다회 iterate 가능)"""
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    out1 = [req.id for req in r.replay()]
    out2 = [req.id for req in r.replay()]
    assert out1 == out2 == [0, 1, 2]


# ============================================================================
# R10 보강 — extreme rate_multiplier sanity
# ============================================================================

@pytest.mark.parametrize("rate", [1e-9, 1e9])
def test_trace_replay_extreme_rate_multiplier(rate):
    r = TraceReplayer.load(FIXTURES / "valid_minimal.csv")
    reqs = list(r.replay(rate_multiplier=rate))
    # 산식 정합 (overflow / underflow 부재)
    assert reqs[0].arrival_time == pytest.approx(0.1 / rate)
    assert reqs[2].arrival_time == pytest.approx(1.0 / rate)
