"""Cluster H — C5 acceptance: determinism (동일 seed + trace → bit-exact JSON).

PLAN §0 C5 — 동일 seed + trace → bit-exact 구조 산출.
"""

import json

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


def _read_report_json(out_dir):
    return (out_dir / "report.json").read_text(encoding="utf-8")


class TestC5Determinism:
    def test_bit_exact_same_seed(self, tmp_path):
        """동일 seed 위 2 회 Run → report.json byte-equal."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        for out in (out1, out2):
            run = Run.init(_CFG, "synthetic:20", out, seed=42)
            run.loop()
            run.teardown()
        text1 = _read_report_json(out1)
        text2 = _read_report_json(out2)
        assert text1 == text2

    def test_different_seed_diverges(self, tmp_path):
        """다른 seed 위 report.json 다름 (synthetic 의 seed 영향)."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        Run.init(_CFG, "synthetic:20", out1, seed=1).loop()
        Run.init(_CFG, "synthetic:20", out1, seed=1).teardown()
        run1 = Run.init(_CFG, "synthetic:20", out1, seed=1)
        run1.loop()
        run1.teardown()
        run2 = Run.init(_CFG, "synthetic:20", out2, seed=2)
        run2.loop()
        run2.teardown()
        text1 = _read_report_json(out1)
        text2 = _read_report_json(out2)
        assert text1 != text2

    def test_json_parseable(self, tmp_path):
        """report.json 이 valid JSON."""
        run = Run.init(_CFG, "synthetic:10", tmp_path, seed=42)
        run.loop()
        run.teardown()
        data = json.loads(_read_report_json(tmp_path))
        assert isinstance(data, dict)
        assert "dispatch_trace" in data
