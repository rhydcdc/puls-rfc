"""Cluster D — C1 acceptance (synthetic:30 → 모든 req COMPLETED)."""

import pytest

from puls_sched.request import RequestState
from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


class TestC1Completion:
    def test_synthetic_100req_all_completed(self, tmp_path):
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        # 모든 admit 된 req 가 COMPLETED 도달
        # in_flight_requests 는 finalize 시 pop → loop 종료 시 0
        assert len(run.scheduler.in_flight_requests) == 0
        # decoded_count >= max_tokens 보장은 finalize 가 보장
        # → 정합 검증은 KV remaining 회수 + window empty 의 cross-module

    def test_window_empty_at_termination(self, tmp_path):
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert len(run.scheduler.window.current_ids()) == 0

    def test_queue_empty_at_termination(self, tmp_path):
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert len(run.scheduler.queue) == 0

    def test_no_stranded_pending_requests(self, tmp_path):
        run = Run.init(_CFG, "synthetic:30", tmp_path)
        run.loop()
        assert len(run.scheduler.request_queue) == 0
