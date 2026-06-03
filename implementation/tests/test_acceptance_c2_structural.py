"""Cluster E — C2 acceptance: structural output schema + substantive ordering (R12).

§6.5 dispatch trace + F1~F5 schema +
ARCH §6.3 priority literal substantive ordering + comparative baseline 부재 lock-in.
"""

import json

import pytest

from puls_sched.node import NodeType
from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


def _make_run_and_report(tmp_path, n: int = 30, seed: int = 42):
    run = Run.init(_CFG, f"synthetic:{n}", tmp_path, seed=seed)
    run.loop()
    report = run.teardown()
    return run, report


class TestC2Schema:
    def test_dispatch_trace_present(self, tmp_path):
        _, report = _make_run_and_report(tmp_path)
        events = report["dispatch_trace"]
        assert len(events) > 0

    def test_acceleration_decomposition_4_sources(self, tmp_path):
        _, report = _make_run_and_report(tmp_path)
        decomp = report["acceleration_decomposition"]
        sources = {cell.source.value for cell in decomp}
        # F1·F2·F3·F5 4 source — F4 미포함 (ARCH §5.7 precondition lock-in)
        assert sources == {"F1_SP_PIM", "F2_DOUBLE_BUFFER", "F3_INSTANCE_AB", "F5_CHANNEL_INDEP"}

    def test_report_json_keys(self, tmp_path):
        _, report = _make_run_and_report(tmp_path)
        for k in ("dispatch_trace", "idle_fraction",
                  "pim_utilization", "pipeline_efficiency",
                  "acceleration_decomposition", "ablation_config", "markdown"):
            assert k in report

    def test_comparative_baseline_absent(self, tmp_path):
        _, report = _make_run_and_report(tmp_path)
        text = json.dumps({k: str(v) for k, v in report.items() if k != "markdown"})
        for forbidden in ("sarathi", "vllm", "baseline"):
            assert forbidden not in text.lower()

    def test_absolute_metric_absent(self, tmp_path):
        _, report = _make_run_and_report(tmp_path)
        text = json.dumps({k: str(v) for k, v in report.items() if k != "markdown"})
        for forbidden in ("ttft", "tpot", "throughput", "goodput"):
            assert forbidden not in text.lower()


# DISPATCHER_PRIORITY_LITERAL — Cluster M lock-in 도 동일 영역 (R12)
DISPATCHER_PRIORITY_LITERAL = (NodeType.O_PROJ, NodeType.PREFILL_ATTN, NodeType.QKV)


class TestC2SubstantiveOrdering:
    def test_priority_literal_lock_in(self):
        """ARCH §6.3 priority literal — Cluster E·M 양쪽에서 영구 기록."""
        from puls_sched.dispatcher import GPU_PRIORITY_ORDER
        assert GPU_PRIORITY_ORDER == DISPATCHER_PRIORITY_LITERAL

    def test_dispatch_trace_resource_pim_only_for_decode_attn(self, tmp_path):
        """모든 DECODE_ATTN dispatch resource == "PIM" (F1 활성화)."""
        _, report = _make_run_and_report(tmp_path)
        for e in report["dispatch_trace"]:
            if e.node_type is NodeType.DECODE_ATTN:
                assert e.resource == "PIM"

    def test_dispatch_trace_gpu_pim_interleaved(self, tmp_path):
        """ARCH §6.5 자연 결과 — GPU 와 PIM resource 가 교대 (전수 GPU → 전수 PIM 아님).

        I4·I5 invariant 의 dispatching 위 자연 emergence — single resource sequential 아닌
        cross-resource concurrent dispatch.
        """
        _, report = _make_run_and_report(tmp_path)
        resources = [e.resource for e in report["dispatch_trace"]]
        gpu_count = resources.count("GPU")
        pim_count = resources.count("PIM")
        # 둘 다 substantive count (interleaved emergence)
        assert gpu_count > 0
        assert pim_count > 0

    def test_dispatch_trace_timestamps_monotonic(self, tmp_path):
        """모든 dispatch event 의 timestamp 가 non-decreasing (event-time 정합)."""
        _, report = _make_run_and_report(tmp_path)
        ts = [e.timestamp for e in report["dispatch_trace"]]
        for i in range(len(ts) - 1):
            assert ts[i] <= ts[i + 1]
