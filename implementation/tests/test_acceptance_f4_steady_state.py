"""Cluster N — F4 precondition 명시 검증 (ARCH §5.7).

F4 = 'steady-state precondition for F2·F3, not a standalone contribution item'.
명시 도달 검증 + F4 schema 부재 lock-in.
"""

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestF4SteadyState:
    def test_f4_not_listed_as_acceleration_source(self, tmp_path):
        """ARCH §5.7 F4 precondition 영역 — acceleration_decomposition 에 F4 부재."""
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        t_proj = max(run.config.time.gpu_op_time_us.values())
        t_pim = 2.0
        cells = run.evaluator.acceleration_decomposition(
            a_cycle=3.0, b_cycle=1.0, t_pim=t_pim, t_proj=t_proj,
        )
        sources = {cell.source.name for cell in cells}
        assert "F4" not in sources
        assert sources == {"F1", "F2", "F3", "F5"}
