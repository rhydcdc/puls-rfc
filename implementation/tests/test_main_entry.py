"""Cluster B — __main__ argv 파싱 + exit code (Impl-9 Q2 · Q9)."""

import json
import sys

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestMainEntry:
    def test_required_flags_missing(self, capsys):
        """argparse missing required → SystemExit code 2 (Python convention, R2)."""
        with pytest.raises(SystemExit) as exc_info:
            Run.main([])
        assert exc_info.value.code == 2

    def test_normal_synthetic_10(self, tmp_path):
        rc = Run.main([
            "--config-module", _CFG,
            "--trace", "synthetic:10",
            "--output", str(tmp_path),
        ])
        assert rc == 0
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_seed_override_propagates(self, tmp_path):
        rc = Run.main([
            "--config-module", _CFG,
            "--trace", "synthetic:5",
            "--output", str(tmp_path),
            "--seed", "7",
        ])
        assert rc == 0
        # report.json 의 ablation_config 이 직렬화되었는지 sanity 만 확인
        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "ablation_config" in data

    def test_user_error_bad_config(self, tmp_path, capsys):
        """Run.main return 1 path (R2) — argparse SystemExit code 2 와 구분."""
        rc = Run.main([
            "--config-module", "bad_module_path_xyz:fn",
            "--trace", "synthetic:5",
            "--output", str(tmp_path),
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Run failed" in captured.err

    def test_user_error_bad_trace(self, tmp_path, capsys):
        rc = Run.main([
            "--config-module", _CFG,
            "--trace", str(tmp_path / "nonexistent.csv"),
            "--output", str(tmp_path),
        ])
        assert rc == 1

    def test_user_error_bad_synthetic_n(self, tmp_path, capsys):
        rc = Run.main([
            "--config-module", _CFG,
            "--trace", "synthetic:0",
            "--output", str(tmp_path),
        ])
        assert rc == 1
