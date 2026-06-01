"""Cluster S — Dispatcher 의 record_active wiring (O8.2).

Impl-10-pre-1 정합. Dispatcher.dispatch_gpu / dispatch_pim 의 record_active 호출 wiring +
cross-module chain (dispatcher → idle_telemetry → admission.balance_intra_A 의 진정 활성).

ARCH §3.5.2 Computed Wait 정합 — dispatch 시점에 (t_start, t_start + op_time) 의 명시 산출.
KERNEL_COMPLETION 의 release 시점 wiring 은 *redundant* (dispatch 시점 push 위 완전 측정).
"""

from dataclasses import dataclass, field

import pytest

from puls_sched.config import default_dummy_config
from puls_sched.dispatcher import Dispatcher
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import Node, NodeState, NodeType


# ---- Mock IdleTelemetry — record_active 호출 capture ----

@dataclass
class _MockTelemetry:
    calls: list[tuple[str, float, float]] = field(default_factory=list)

    def record_active(self, resource, t_start, t_end):
        self.calls.append((resource, t_start, t_end))


# ---- Dispatcher backward-compat (idle_telemetry=None) ----

def test_dispatcher_default_idle_telemetry_is_none():
    """Default Dispatcher (idle_telemetry 미주입) 위 dispatch_gpu/pim 정상 동작 — backward-compat."""
    from puls_sched.clock import Clock
    from puls_sched.dag import DAG
    from puls_sched.event_queue import EventQueue
    from puls_sched.pim_emulator import PIMExecutor
    cfg = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    pim = PIMExecutor(config=cfg)
    d = Dispatcher(config=cfg, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    assert d.idle_telemetry is None  # backward-compat default


# ---- dispatch_gpu 위 record_active call ----

def test_dispatch_gpu_records_gpu_instance_a(dummy_config, clock, event_queue, dag, pim_executor):
    """dispatch_gpu 호출 시 idle_telemetry.record_active('gpu_instance_a', t_start, t_start+op_time)."""
    mock = _MockTelemetry()
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor, idle_telemetry=mock,
    )
    dag.add_micro_batch(0)
    qkv_node = dag.get_node(0, NodeType.QKV)
    qkv_node.transition_to(NodeState.READY)
    d.dispatch_gpu(qkv_node)
    assert len(mock.calls) == 1
    resource, t_start, t_end = mock.calls[0]
    assert resource == "gpu_instance_a"
    assert t_start == clock.now      # dispatch 시점
    assert t_end > t_start            # op_time 추가됨


def test_dispatch_pim_records_pim_instance_a(dummy_config, clock, event_queue, dag, pim_executor):
    """dispatch_pim 호출 시 idle_telemetry.record_active('pim_instance_a', t_start, t_start+op_time)."""
    mock = _MockTelemetry()
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor, idle_telemetry=mock,
    )
    dag.add_micro_batch(0)
    # I2 precondition — QKV done
    qkv_node = dag.get_node(0, NodeType.QKV)
    qkv_node.transition_to(NodeState.READY)
    qkv_node.transition_to(NodeState.RUNNING)
    qkv_node.transition_to(NodeState.DONE)
    mb = MicroBatch(id=0, kv_rows_total=100)
    d.register(mb)
    decode_node = dag.get_node(0, NodeType.DECODE_ATTN)
    decode_node.transition_to(NodeState.READY)
    d.dispatch_pim(decode_node)
    assert len(mock.calls) == 1
    resource, t_start, t_end = mock.calls[0]
    assert resource == "pim_instance_a"
    assert t_start == clock.now
    assert t_end > t_start


# ---- duration = op_time 정합 ----

def test_dispatch_gpu_record_duration_matches_op_time(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    mock = _MockTelemetry()
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor, idle_telemetry=mock,
    )
    dag.add_micro_batch(0)
    node = dag.get_node(0, NodeType.QKV)
    node.transition_to(NodeState.READY)
    d.dispatch_gpu(node)
    _, t_start, t_end = mock.calls[0]
    # Stage 2 — op_time = compute_gpu_op_time_s (spec-derived per-mb fallback)
    expected_op_time = d._op_time(node)
    assert t_end - t_start == pytest.approx(expected_op_time)


# ---- 실 IdleTelemetry 와 cross-module ----

def test_dispatcher_real_idle_telemetry_accumulates(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """실 IdleTelemetry 위 dispatch_gpu 다회 호출 후 gpu_instance_a slot 누적."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor, idle_telemetry=tel,
    )
    op_times = []
    for mb_id in range(3):
        dag.add_micro_batch(mb_id)
        node = dag.get_node(mb_id, NodeType.QKV)
        node.transition_to(NodeState.READY)
        op_times.append(d._op_time(node))
        d.dispatch_gpu(node)
        # 다음 dispatch 위 gpu_busy 해제
        d.gpu_busy = False
    # Stage 2 — 3 dispatch × spec-derived op_time (per-mb, 모두 동일 fallback mb 위 동일)
    expected_active = sum(op_times)
    assert tel._active_duration["gpu_instance_a"] == pytest.approx(expected_active)


# ---- admission.balance_intra_A 진정 활성 chain ----

# test_balance_intra_A_activates_via_dispatch_chain 삭제(S1) — admission.balance_intra_A
# (유일 유휴율 레버)가 S1 에서 제거됨. 검증 대상 메서드 부재 → 테스트 폐기.


# ---- Determinism ----

def test_dispatch_gpu_record_deterministic(dummy_config, clock, event_queue, dag, pim_executor):
    """동일 sequence → 동일 record (1000-iter)."""
    def _seq(cfg, clock_, queue_, dag_, pim_):
        tel = IdleTelemetry()
        tel.reset(0.0)
        d = Dispatcher(
            config=cfg, clock=clock_, queue=queue_, dag=dag_,
            pim_executor=pim_, idle_telemetry=tel,
        )
        dag_.add_micro_batch(0)
        node = dag_.get_node(0, NodeType.QKV)
        node.transition_to(NodeState.READY)
        d.dispatch_gpu(node)
        return tel.gpu_idle_fraction()
    from puls_sched.clock import Clock
    from puls_sched.dag import DAG
    from puls_sched.event_queue import EventQueue
    from puls_sched.pim_emulator import PIMExecutor
    results = []
    for _ in range(100):
        cfg = default_dummy_config()
        c = Clock(); q = EventQueue(c); g = DAG(); p = PIMExecutor(config=cfg)
        results.append(_seq(cfg, c, q, g, p))
    assert len(set(results)) == 1
