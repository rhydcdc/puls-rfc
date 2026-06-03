"""Cluster A — Run.init / step / loop / teardown 단위 (Impl-9)."""

import json
import sys

import pytest

from puls_sched.event import EventType
from puls_sched.request import RequestState
from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestRunInit:
    def test_init_normal(self, tmp_path):
        run = Run.init(_CFG, "synthetic:10", tmp_path)
        # 모든 핵심 컴포넌트 instantiate 정합
        assert run.scheduler is not None
        assert run.evaluator is not None
        # Evaluator hook 등록 정합 (1 dispatch callback)
        assert len(run.scheduler.dispatcher._dispatch_callbacks) == 1
        # Q1 self-rescheduling enable 확인 (live admission 재기동 경로)
        assert run.scheduler.continuous_admission is True
        # Q4 eager pre-load — queue 에 REQUEST_ARRIVAL 10 + ADMISSION_PASS 1 = 11
        types = [e[2].type for e in run.scheduler.queue._heap]
        assert types.count(EventType.REQUEST_ARRIVAL) == 10
        assert types.count(EventType.ADMISSION_PASS) == 1

    def test_init_bad_config_module_format(self, tmp_path):
        with pytest.raises(ValueError, match="module.path:factory_fn"):
            Run.init("no_colon", "synthetic:5", tmp_path)

    def test_init_module_not_found(self, tmp_path):
        with pytest.raises(ModuleNotFoundError):
            Run.init("nonexistent_mod_xyz:fn", "synthetic:5", tmp_path)

    def test_init_factory_not_found(self, tmp_path):
        with pytest.raises(AttributeError):
            Run.init("puls_sched.config:nonexistent_fn", "synthetic:5", tmp_path)

    def test_init_synthetic_n_zero(self, tmp_path):
        with pytest.raises(ValueError, match="n >= 1"):
            Run.init(_CFG, "synthetic:0", tmp_path)

    def test_init_trace_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Run.init(_CFG, str(tmp_path / "nonexistent.csv"), tmp_path)

    def test_init_seed_override(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path, seed=99)
        assert run.config.seed == 99

    def test_init_seed_none_preserves_default(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path, seed=None)
        # default_dummy_config 의 seed default = 42
        assert run.config.seed == 42


class TestRunStep:
    def test_step_advances_count(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        before = run._step_count
        run.step()
        assert run._step_count == before + 1

    def test_step_returns_false_when_queue_empty(self, tmp_path):
        run = Run.init(_CFG, "synthetic:1", tmp_path)
        # Drain
        while run.step():
            pass
        # Further step → False
        assert run.step() is False

    def test_step_safety_limit(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        run._safety_step_limit = 3
        with pytest.raises(RuntimeError, match="safety_step_limit"):
            while run.step():
                pass


class TestRunLoop:
    def test_loop_bounded_10req_termination(self, tmp_path):
        run = Run.init(_CFG, "synthetic:10", tmp_path)
        steps = run.loop()
        assert steps > 0
        # Termination 3 조건 정합
        assert len(run.scheduler.queue) == 0
        assert len(run.scheduler.window.current_ids()) == 0
        assert len(run.scheduler.in_flight_requests) == 0


class TestRunTeardown:
    def test_teardown_writes_both_files(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        run.loop()
        run.teardown()
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()

    def test_teardown_json_parseable(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        run.loop()
        run.teardown()
        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "dispatch_trace" in data
        assert "idle_fraction" in data
        assert "acceleration_decomposition" in data

    def test_teardown_md_non_empty(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        run.loop()
        run.teardown()
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "PULS Structural Evaluator Report" in md

    def test_teardown_returns_report_dict(self, tmp_path):
        run = Run.init(_CFG, "synthetic:5", tmp_path)
        run.loop()
        report = run.teardown()
        assert "markdown" in report
        assert "dispatch_trace" in report
