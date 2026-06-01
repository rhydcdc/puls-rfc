"""Cluster I — Cross-module e2e (실 LongBench trace 위 lifecycle).

Run.init → real CSV eager pre-load → loop → teardown 의 full chain.
HoL skip + capacity 4M 정합 위 *모든 req 길이* 자연 소화.
"""

import pathlib

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"
_REAL_TRACE = pathlib.Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"


def _make_truncated_trace(tmp_path, n: int, decode_override: int = 1) -> str:
    """실 trace first N row 추출 + max_tokens 단축 (시뮬레이션 runtime 단축).

    prefill_tokens 는 *원본* (47K ~ 2.5M 범위) — HoL skip + capacity 4M 정합 영역 정확 검증.
    """
    out = tmp_path / "trace_first_n.csv"
    with open(_REAL_TRACE, "r") as fin, open(out, "w") as fout:
        lines = fin.readlines()
        fout.write(lines[0])
        for line in lines[1:n + 1]:
            parts = line.strip().split(",")
            parts[2] = str(decode_override)
            fout.write(",".join(parts) + "\n")
    return str(out)


# 실 트레이스 cold-start 시뮬레이션은 더 이상 검증 vehicle 이 아님 — prefill 47K~2.5M 콜드스타트가
# 수억 step → 비현실적. 검증은 생성 풀 트레이스(data/sweep_*) + warm-start seed 로 대체(STEP8,
# README Runtime Validation). replay path(TraceReplayer) 자체는 유효(sweep 트레이스도 동일 파서).
@pytest.mark.skip(reason="real-trace cold-start sim 미사용 — sweep_* + warm-start 로 대체(STEP8)")
class TestE2eRealTrace:
    def test_real_trace_first_30_lifecycle(self, tmp_path):
        """30 reqs (prefill max 2.5M) 모두 lifecycle 완주.

        HoL skip + capacity 4M 위 *큰 req 도 자연 admit* (capacity 회수 후).
        """
        trace_path = _make_truncated_trace(tmp_path, n=30)
        run = Run.init(_CFG, trace_path, tmp_path / "out")
        run.loop()
        # 모든 30 req 의 lifecycle 완주
        assert run.scheduler.in_flight_requests == {}
        assert len(run.scheduler.request_queue) == 0
        assert run.scheduler.dag.nodes == {}

    def test_real_trace_kv_no_leak(self, tmp_path):
        """C3 영역의 real trace 확장 — KV release 정합."""
        trace_path = _make_truncated_trace(tmp_path, n=30)
        run = Run.init(_CFG, trace_path, tmp_path / "out")
        initial = run.config.admission.kv_capacity_aggregate
        run.loop()
        assert run.scheduler.kv_accountant.remaining == initial

    def test_real_trace_report_emitted(self, tmp_path):
        """C2 영역의 real trace 확장 — report 산출 정합."""
        trace_path = _make_truncated_trace(tmp_path, n=20)
        out_dir = tmp_path / "out"
        run = Run.init(_CFG, trace_path, out_dir)
        run.loop()
        run.teardown()
        assert (out_dir / "report.json").exists()
        assert (out_dir / "report.md").exists()

    def test_real_trace_hol_skip_preserves_order_relative(self, tmp_path):
        """HoL skip 위 큰 req 도 *결국* admit (영원 잔존 0)."""
        trace_path = _make_truncated_trace(tmp_path, n=30)
        run = Run.init(_CFG, trace_path, tmp_path / "out")
        run.loop()
        # 모든 req 처리 완료 (skip 된 req 도 capacity 회수 후 admit + finalize)
        assert len(run.scheduler.request_queue) == 0
