"""Cluster C — trace.synthesize(n, seed) staticmethod (Impl-9 Q5).

PLAN §0.5 — 구조 property 만 검증, 정량 일치 0.
"""

import pytest

from puls_sched.trace import TraceReplayer


class TestSynthesize:
    def test_synthesize_n_correct(self):
        r = TraceReplayer.synthesize(100, seed=42)
        assert len(r.entries) == 100

    def test_synthesize_seed_determinism(self):
        a = TraceReplayer.synthesize(50, seed=1)
        b = TraceReplayer.synthesize(50, seed=1)
        assert a.entries == b.entries

    def test_synthesize_seed_isolation(self):
        a = TraceReplayer.synthesize(50, seed=1)
        b = TraceReplayer.synthesize(50, seed=2)
        assert a.entries != b.entries

    def test_synthesize_n_zero_raises(self):
        with pytest.raises(ValueError, match="n >= 1"):
            TraceReplayer.synthesize(0, seed=42)

    def test_synthesize_n_negative_raises(self):
        with pytest.raises(ValueError, match="n >= 1"):
            TraceReplayer.synthesize(-1, seed=42)

    def test_synthesize_arrival_monotonic(self):
        r = TraceReplayer.synthesize(200, seed=7)
        for i in range(len(r.entries) - 1):
            assert r.entries[i + 1].arrived_at >= r.entries[i].arrived_at

    def test_synthesize_positive_tokens(self):
        r = TraceReplayer.synthesize(100, seed=11)
        for e in r.entries:
            assert e.num_prefill_tokens > 0
            assert e.num_decode_tokens > 0

    def test_synthesize_replay_chain(self):
        r = TraceReplayer.synthesize(10, seed=42)
        reqs = list(r.replay(rate_multiplier=1.0))
        assert len(reqs) == 10
        for req in reqs:
            assert req.max_tokens > 0
            assert req.kv_length > 0
            assert req.arrival_time >= 0.0

    def test_synthesize_stats_callable(self):
        r = TraceReplayer.synthesize(50, seed=99)
        s = r.stats()
        assert s.n_entries == 50
        assert s.kv_length_min > 0
