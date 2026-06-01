"""Cluster W — Inter-AB signal end-to-end activation (Impl-10-pre-1 (A) wiring).

ARCH §3.4 (*forward pass = L × cycle*) + §6.4 (admission balance signal driving) 정합 의
*production hot path 검증*. 본 cluster 의 진짜 의도:

- 본 commit 위 *unit test 영역* (Cluster R/S/T) 가 통과해도 *production scheduler 위 실제
  inter-AB signal 활성* 보장 안 함 (silent gap — Disclosure 영역).
- (A) wiring 후 `SchedulerCore._maybe_advance_forward_pass` 가 매 O_PROJ 완료 시
  `instance_pipeline.dispatch(mb)` 호출 → `gpu_instance_b` activity production 위 진정 누적.
- 본 test 가 *end-to-end Run.loop 후 evaluator report* 위 `gpu_instance_b` idle_fraction 의
  *non-trivial 값* (1.0 미만) 검증 — silent gap 영원 차단.

CLAUDE.md §5 *meta-test + cross-module invariant* 정합 — *"진짜 작동하는가"* 영역.
"""

import json
import tempfile
from pathlib import Path

import pytest

from puls_sched.config import default_dummy_config
from puls_sched.run import Run


# ---- End-to-end: Run.loop 후 gpu_instance_b signal 활성 ----

def test_run_loop_activates_gpu_instance_b_signal(tmp_path):
    """End-to-end — Run.init + loop + teardown 후 evaluator report 의
    idle_fraction["gpu_instance_b"] 가 *non-trivial* (1.0 < — 활동 누적 검증).
    """
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    run.loop()
    report = run.teardown()
    assert report["idle_fraction"]["gpu_instance_b"] < 1.0


def test_run_loop_intra_a_signals_also_active(tmp_path):
    """Regression — intra-A signal (gpu_instance_a · pim_instance_a) 도 production 활성.
    Dispatcher.dispatch_gpu/pim 의 record_active wiring 정합 검증."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    run.loop()
    report = run.teardown()
    # gpu_instance_a / pim_instance_a 도 활동 누적 — 1.0 < (Dispatcher wiring)
    assert report["idle_fraction"]["gpu_instance_a"] < 1.0
    assert report["idle_fraction"]["pim_instance_a"] < 1.0


def test_run_loop_three_key_schema_all_present(tmp_path):
    """End-to-end — evaluator report 의 idle_fraction 3-key 모두 존재 (schema lock-in)."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    run.loop()
    report = run.teardown()
    assert set(report["idle_fraction"].keys()) == {
        "gpu_instance_a", "pim_instance_a", "gpu_instance_b",
    }


# ---- Determinism — (A) wiring 후에도 C5 bit-exact 보존 ----

def test_run_loop_gpu_instance_b_deterministic(tmp_path):
    """동일 seed + 동일 trace → gpu_instance_b idle_fraction bit-exact (C5 정합)."""
    results = []
    for trial in range(3):
        out = tmp_path / f"trial_{trial}"
        run = Run.init(
            config_module="puls_sched.config:default_dummy_config",
            trace_path_or_synthetic="synthetic:5",
            output_dir=out,
            seed=42,
        )
        run.loop()
        report = run.teardown()
        results.append(report["idle_fraction"]["gpu_instance_b"])
    assert len(set(results)) == 1, f"non-deterministic gpu_instance_b: {results}"


def test_run_loop_report_json_unchanged_across_seeds(tmp_path):
    """C5 bit-exact — 동일 seed 위 report.json 의 idle_fraction 영역 bit-exact 보존."""
    results = []
    for trial in range(3):
        out = tmp_path / f"trial_{trial}"
        out.mkdir(exist_ok=True)
        run = Run.init(
            config_module="puls_sched.config:default_dummy_config",
            trace_path_or_synthetic="synthetic:5",
            output_dir=out,
            seed=42,
        )
        run.loop()
        run.teardown()
        json_text = (out / "report.json").read_text(encoding="utf-8")
        data = json.loads(json_text)
        results.append(data["idle_fraction"])
    # 3 trial 모두 동일
    assert results[0] == results[1] == results[2]


# ---- (A) wiring 결손 시 silent gap 차단 — schedule_core.instance_pipeline 의 None 확인 ----

def test_run_init_passes_instance_pipeline_to_scheduler(tmp_path):
    """Run.init 후 SchedulerCore.instance_pipeline 가 None 아님 — (A) plumbing 정합."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    assert run.scheduler.instance_pipeline is not None


def test_scheduler_core_without_instance_pipeline_still_works(tmp_path):
    """Backward-compat — SchedulerCore.instance_pipeline=None 위 (단위 test fixture 영역)
    _maybe_advance_forward_pass 가 skip 분기 → 기존 lifecycle 영역 무변경."""
    from puls_sched.clock import Clock
    from puls_sched.event_queue import EventQueue
    from puls_sched.dag import DAG
    from puls_sched.window import InFlightWindow
    from puls_sched.idle_telemetry import IdleTelemetry
    from puls_sched.pim_emulator import PIMExecutor
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.request_queue import RequestQueue
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.admission import Admission
    from puls_sched.forward_pass import LayerState
    from puls_sched.completion import Completion
    from puls_sched.main_loop import SchedulerCore
    cfg = default_dummy_config()
    clock = Clock(); queue = EventQueue(clock); dag = DAG()
    window = InFlightWindow(dag, config=cfg)
    tel = IdleTelemetry(); tel.reset(0.0)
    pim = PIMExecutor(config=cfg)
    disp = Dispatcher(config=cfg, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    rq = RequestQueue(capacity=cfg.admission.request_queue_capacity)
    kv = KVAccountant(capacity=cfg.admission.kv_capacity_aggregate)
    adm = Admission(admission_cfg=cfg.admission, request_queue=rq,
                    kv_accountant=kv, idle_telemetry=tel)
    ls = LayerState(num_layers=cfg.model.num_layers)
    comp = Completion(clock=clock, kv_accountant=kv)
    sc = SchedulerCore(
        config=cfg, clock=clock, queue=queue, dag=dag, window=window,
        dispatcher=disp, request_queue=rq, kv_accountant=kv, admission=adm,
        layer_state=ls, completion=comp,
        # instance_pipeline 미주입 — backward-compat
    )
    assert sc.instance_pipeline is None
    # step() 호출 시 raise 0 (empty queue → False 반환만)
    assert sc.step() is False


# ---- ARCH §3.4 cycle count semantic 정합 ----
# Phase-2 S0/S3 — test_instance_pipeline_dispatch_invoked_per_layer 폐기.
# layer advance 가 O_PROJ→FFN 노드로 이동하며 `instance_pipeline.dispatch` 가 hot path 에서
# 제거됨(호출 0, 의도된 결과). inter-AB(F3) per-layer cycle 은 이제 FFN 노드(INSTANCE_B 자원)
# + gpu_instance_b activity 로 검증 — test_run_loop_activates_gpu_instance_b_signal 가 대체.
