"""Impl-8 cluster F — multi-mb stress + multi-seed determinism + I-E1~I-E5 invariants.

Impl-2 의 100-mb stress + Impl-5 multi-mb stress + Impl-6 50-req lifecycle 패턴 정합.

검증 대상 invariants (Impl-8 신설):
- (I-E1) Evaluator._dispatch_events capture loss 0
- (I-E2) Evaluator._admission_snapshots capture loss 0
- (I-E3) report() reproducibility — 동일 seed 위 bit-exact
- (I-E4) Evaluator 가 underlying state 무손상 (KV invariant 보존)
- (I-E5) Ablation flag 4 cell direction (F1·F2·F3·F5) 모두 direction_positive=True over seeds
"""
import dataclasses
import random

import pytest

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.evaluator import AblationSource, Evaluator
from puls_sched.forward_pass import LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request_queue import RequestQueue
from puls_sched.window import InFlightWindow


def _build_stress_core(config=None):
    if config is None:
        config = default_dummy_config()
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag, config=config)
    pim = PIMExecutor(config=config)
    dispatcher = Dispatcher(config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    rq = RequestQueue(capacity=config.admission.request_queue_capacity)
    kva = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    idle = IdleTelemetry()
    admission = Admission(
        admission_cfg=config.admission, request_queue=rq,
        kv_accountant=kva, idle_telemetry=idle,
    )
    core = SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag, window=window,
        dispatcher=dispatcher, request_queue=rq, kv_accountant=kva,
        admission=admission,
        layer_state=LayerState(num_layers=config.model.num_layers),
        completion=Completion(clock=clock, kv_accountant=kva),
    )
    evaluator = Evaluator(config=config, clock=clock, idle_telemetry=idle)
    dispatcher.on_dispatch(evaluator.record_dispatch)
    core.on_admission_tick(evaluator.record_admission_tick)
    return core, evaluator


def _dispatch_n_mbs(core, n):
    """N mb 위 단순 QKV → PREFILL_ATTN/DECODE_ATTN → O_PROJ chain dispatch."""
    for mb_id in range(n):
        core.dispatcher.register(MicroBatch(id=mb_id, k_total=2048, kv_rows_total=32, kv_rows_lockstep=32))
        core.dag.add_micro_batch(mb_id)
        # Manual full chain
        for ntype, resource, busy_field in [
            (NodeType.QKV, "GPU", "gpu_busy"),
            (NodeType.PREFILL_ATTN, "GPU", "gpu_busy"),
            (NodeType.DECODE_ATTN, "PIM", "pim_busy"),
            (NodeType.O_PROJ, "GPU", "gpu_busy"),
        ]:
            node = core.dag.get_node(mb_id, ntype)
            if node.state == NodeState.PENDING:
                node.transition_to(NodeState.READY)
            setattr(core.dispatcher, busy_field, False)
            if resource == "GPU":
                core.dispatcher.dispatch_gpu(node)
            else:
                core.dispatcher.dispatch_pim(node)
            node.transition_to(NodeState.DONE)
            setattr(core.dispatcher, busy_field, False)


# =========================================================================
# (I-E1) dispatch capture loss 0
# =========================================================================

def test_stress_100_mb_dispatch_capture_loss_zero():
    """100 mb × 4 dispatch = 400 events 모두 누적 (loss 0)."""
    core, ev = _build_stress_core()
    _dispatch_n_mbs(core, 100)
    events = ev.dispatch_trace()
    assert len(events) == 400  # 100 mb × 4 nodes


# =========================================================================
# (I-E2) admission snapshot capture loss 0
# =========================================================================

def test_stress_100_admission_ticks_capture_loss_zero():
    """100 admission tick (admit + reject 혼합) → 100 snapshot."""
    core, ev = _build_stress_core()
    for i in range(100):
        core.queue.push(Event(
            timestamp=float(i), type=EventType.ADMISSION_TICK,
            payload={"t_proj": 1.0, "a_cycle": 1.0, "b_cycle": 1.0 + (i % 5), "ctx_tokens": 4000},
        ))
    while core.step():
        pass
    # admission 이 spec 못 만들어도 (queue empty) snapshot 누적
    assert len(ev._admission_snapshots) == 100


# =========================================================================
# (I-E3) Multi-seed determinism (PLAN §0 C5)
# =========================================================================

@pytest.mark.parametrize("seed", [0, 42, 99, 1000])
def test_stress_report_seed_sweep_deterministic(seed):
    """동일 trace + seed → report() bit-exact (4 cell)."""
    config = default_dummy_config()
    core1, ev1 = _build_stress_core(config)
    core2, ev2 = _build_stress_core(config)
    rng1 = random.Random(seed)
    rng2 = random.Random(seed)

    for c, r in [(core1, rng1), (core2, rng2)]:
        for i in range(20):
            a = 10.0 + r.uniform(0, 5)
            b = 10.0 + r.uniform(0, 5)
            c.queue.push(Event(
                timestamp=float(i), type=EventType.ADMISSION_TICK,
                payload={"t_proj": 1.0, "a_cycle": a, "b_cycle": b, "ctx_tokens": 4000},
            ))
        while c.step():
            pass

    r1 = ev1.report(a_cycle=10, b_cycle=20, t_pim=2, t_proj=2)
    r2 = ev2.report(a_cycle=10, b_cycle=20, t_pim=2, t_proj=2)
    assert r1["markdown"] == r2["markdown"]
    assert r1["convergence"] == r2["convergence"]


# =========================================================================
# (I-E4) Non-intrusion — KV no-leak + DAG state no-corruption
# =========================================================================

def test_stress_evaluator_no_kv_leak_stress():
    """100 mb stress 진행 + Evaluator 부착 → KV remaining 보존."""
    core, ev = _build_stress_core()
    initial = core.kv_accountant.remaining
    _dispatch_n_mbs(core, 100)
    # No admission → no KV consumption. Evaluator hook only records events.
    assert core.kv_accountant.remaining == initial


def test_stress_evaluator_no_dag_state_corruption():
    """100 mb dispatch + evaluator hook → DAG 의 final state 가 evaluator 부재 path 와 동일."""
    # With evaluator
    core_with, ev = _build_stress_core()
    _dispatch_n_mbs(core_with, 100)
    final_with = {
        (mb_id, ntype): node.state
        for mb_id, nodes in core_with.dag.nodes.items()
        for ntype, node in nodes.items()
    }
    # Without evaluator
    core_wo, _ev2 = _build_stress_core()
    core_wo.dispatcher._dispatch_callbacks.clear()
    _dispatch_n_mbs(core_wo, 100)
    final_wo = {
        (mb_id, ntype): node.state
        for mb_id, nodes in core_wo.dag.nodes.items()
        for ntype, node in nodes.items()
    }
    assert final_with == final_wo


# =========================================================================
# (I-E5) Ablation direction_positive over multi-seed
# =========================================================================

@pytest.mark.parametrize("seed", [0, 42, 99, 1000])
def test_stress_ablation_f1_direction_positive_multiseed(seed):
    """F1 cell direction_positive=True over seeds (산식 deterministic — sanity)."""
    config = default_dummy_config()
    _core, ev = _build_stress_core(config)
    rng = random.Random(seed)
    t_pim = rng.uniform(0.5, 3.0)  # < decode_attn_fallback (4.0) → F1 가속 보장
    t_proj = rng.uniform(0.5, 3.0)
    decomp = ev.acceleration_decomposition(a_cycle=10, b_cycle=20, t_pim=t_pim, t_proj=t_proj)
    f1 = next(c for c in decomp if c.source == AblationSource.F1)
    assert f1.direction_positive is True


def test_stress_ablation_all_4_direction_positive_default_config(evaluator):
    """Default config + 적당한 input 위 4 cell ratio > 1 + direction_positive."""
    decomp = evaluator.acceleration_decomposition(a_cycle=10, b_cycle=10, t_pim=2, t_proj=2)
    for cell in decomp:
        assert cell.direction_positive is True
        assert cell.ratio > 1.0


# =========================================================================
# Composite — PIM/pipeline boundary
# =========================================================================

def test_stress_pim_utilization_bounded_0_1():
    """100-mb stress 위 pim_utilization 가 [0, 1] 영역 (산식 정의 sanity)."""
    core, ev = _build_stress_core()
    _dispatch_n_mbs(core, 100)
    util = ev.pim_utilization()
    assert 0.0 <= util <= 1.0


def test_stress_pipeline_efficiency_bounded(evaluator):
    """100 (a, b) 쌍 위 efficiency ∈ [0.5, 1.0]."""
    rng = random.Random(42)
    for _ in range(100):
        a = rng.uniform(0.1, 100.0)
        b = rng.uniform(0.1, 100.0)
        eff = evaluator.pipeline_efficiency(a, b)
        assert 0.5 <= eff <= 1.0


# =========================================================================
# Composite — 4 seed × 100 mb 위 모든 I-E1~I-E5 보존
# =========================================================================

@pytest.mark.parametrize("seed", [0, 42, 99, 1000])
def test_stress_composite_invariant_violation_zero(seed):
    """Random seed 위 100-mb dispatch + 50 admission tick → 모든 invariant 보존."""
    config = default_dummy_config()
    core, ev = _build_stress_core(config)
    rng = random.Random(seed)
    initial_kv = core.kv_accountant.remaining
    _dispatch_n_mbs(core, 100)
    # _dispatch_n_mbs 가 manual node transition 으로 DONE 마킹 → 그 과정에서 push 된 KERNEL_COMPLETION
    # event 들을 drain (재방문 시 DONE → DONE invalid transition 방지). 본 test 는 *invariant 검증* 만,
    # event loop 의 자연 progression 은 다른 test 영역.
    while len(core.queue) > 0:
        core.queue.pop()
    for i in range(50):
        a = rng.uniform(1.0, 10.0)
        b = rng.uniform(1.0, 10.0)
        core.queue.push(Event(
            timestamp=100.0 + float(i), type=EventType.ADMISSION_TICK,
            payload={"t_proj": 1.0, "a_cycle": a, "b_cycle": b, "ctx_tokens": 4000},
        ))
    while core.step():
        pass
    # (I-E1) dispatch capture
    assert len(ev.dispatch_trace()) == 400
    # (I-E2) admission capture
    assert len(ev._admission_snapshots) == 50
    # (I-E4) KV invariant — no admission → no consumption
    assert core.kv_accountant.remaining == initial_kv
    # (I-E5) F1·F2·F3·F5 direction_positive
    decomp = ev.acceleration_decomposition(a_cycle=10, b_cycle=10, t_pim=2, t_proj=2)
    for cell in decomp:
        assert cell.direction_positive is True
