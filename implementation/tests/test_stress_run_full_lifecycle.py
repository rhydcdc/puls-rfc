"""Cluster J — Run full lifecycle stress (multi-seed determinism sweep).

PLAN §0 C5 의 sweep 확장 — 동일 seed 끼리 bit-exact, 다른 seed 끼리 diverge.
"""

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestStressRunLifecycle:
    def test_synthetic_200_invariant_preservation(self, tmp_path):
        """200 req 위 invariant 보존 (raise 없이 완주)."""
        run = Run.init(_CFG, "synthetic:200", tmp_path)
        run.loop()
        assert run.scheduler.in_flight_requests == {}

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_multi_seed_determinism_sweep(self, tmp_path, seed):
        """4 seed cell 위 동일 seed 끼리 bit-exact."""
        out1 = tmp_path / f"s{seed}_a"
        out2 = tmp_path / f"s{seed}_b"
        Run.init(_CFG, "synthetic:20", out1, seed=seed).loop()
        Run.init(_CFG, "synthetic:20", out1, seed=seed).teardown()
        run1 = Run.init(_CFG, "synthetic:20", out1, seed=seed)
        run1.loop()
        run1.teardown()
        run2 = Run.init(_CFG, "synthetic:20", out2, seed=seed)
        run2.loop()
        run2.teardown()
        text1 = (out1 / "report.json").read_text(encoding="utf-8")
        text2 = (out2 / "report.json").read_text(encoding="utf-8")
        assert text1 == text2
