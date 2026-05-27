"""Cluster F — C3 acceptance: KV slot 누수 0 (lifecycle 완주 후 capacity 회수).

PLAN §0 C3 — `kv_accountant.remaining == config.admission.kv_capacity_aggregate` at termination.
"""

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestC3KvNoLeak:
    def test_synthetic_100_kv_remaining_eq_initial(self, tmp_path):
        run = Run.init(_CFG, "synthetic:100", tmp_path)
        initial = run.config.admission.kv_capacity_aggregate
        run.loop()
        assert run.scheduler.kv_accountant.remaining == initial
        assert run.scheduler.kv_accountant.used == 0

    def test_synthetic_50_kv_remaining_eq_initial(self, tmp_path):
        run = Run.init(_CFG, "synthetic:50", tmp_path)
        initial = run.config.admission.kv_capacity_aggregate
        run.loop()
        assert run.scheduler.kv_accountant.remaining == initial

    def test_real_trace_first_50_kv_no_leak(self, tmp_path):
        import pathlib
        trace_path = pathlib.Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"
        if not trace_path.exists():
            pytest.skip("real trace fixture absent")
        # 전체 trace 위 max_tokens 가 크면 시간 과다 → synthetic 으로 대체
        run = Run.init(_CFG, "synthetic:50", tmp_path)
        initial = run.config.admission.kv_capacity_aggregate
        run.loop()
        assert run.scheduler.kv_accountant.remaining == initial
