"""Cluster G — C4 acceptance: I1~I5 invariant 위반 0 회 over full trace.

I1·I2·I3 (DAG precedence) · I4·I5 (resource exclusivity) 가 dispatcher/window/dag
의 raise 영역. Run.loop 가 raise 없이 완주 = invariant 위반 0.
"""

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestC4Invariants:
    def test_synthetic_100_no_invariant_violation(self, tmp_path):
        """Run.loop() 가 raise 없이 완주 = I1~I5 위반 0 회."""
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        # Loop 완주 자체가 invariant 위반 0 증거 (raise propagate)

    def test_synthetic_50_no_invariant_violation(self, tmp_path):
        run = Run.init(_CFG, "synthetic:20", tmp_path)
        run.loop()

    def test_final_resource_state_idle(self, tmp_path):
        """Lifecycle 완료 후 gpu_busy=False · pim_busy=False (I4·I5 자연 보존)."""
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert run.scheduler.dispatcher.gpu_busy is False
        assert run.scheduler.dispatcher.pim_busy is False

    def test_final_dag_empty(self, tmp_path):
        """Lifecycle 완료 후 DAG 가 비어있음 (모든 mb evict, Q9 carry-over 해소)."""
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert run.scheduler.dag.nodes == {}

    def test_final_in_flight_empty(self, tmp_path):
        """Lifecycle 완료 후 in_flight_requests 비어있음."""
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert run.scheduler.in_flight_requests == {}
