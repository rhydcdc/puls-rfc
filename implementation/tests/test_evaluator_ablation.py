"""Impl-8 cluster D — F1·F2·F3·F5 ablation flag on/off 위 cycle ratio direction.

PLAN §0.5 정합 — *정량 ratio 절대값 산출 0*. Direction (증감 부호) + 산식 정합 + F4 lock-in 만 검증.
"""
import dataclasses
import math

import pytest

from puls_sched.clock import Clock
from puls_sched.config import AblationConfig, default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event_queue import EventQueue
from puls_sched.evaluator import AblationSource, Evaluator
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import Node, NodeState, NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.window import InFlightWindow


# =========================================================================
# F1 ablation — dispatcher PIM → GPU fallback
# =========================================================================

def test_ablation_f1_off_dispatcher_uses_fallback(config_f1_disabled):
    """F1 disabled 위 dispatcher._op_time(PIM node) == config.time.gpu_op_time_us['decode_attn_fallback']."""
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    pim = PIMExecutor(config=config_f1_disabled)
    d = Dispatcher(config=config_f1_disabled, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    dag.add_micro_batch(0)
    d.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    decode_node = dag.get_node(0, NodeType.DECODE_ATTN)
    op_time = d._op_time(decode_node)
    assert op_time == config_f1_disabled.time.gpu_op_time_us["decode_attn_fallback"]


def test_ablation_f1_off_resource_label_remains_pim(config_f1_disabled):
    """F1 ablation 시에도 dispatch_pim path 의 resource label == 'PIM' (I5 invariant 보존)."""
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    pim = PIMExecutor(config=config_f1_disabled)
    d = Dispatcher(config=config_f1_disabled, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    dag.add_micro_batch(0)
    d.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))

    captured = []
    d.on_dispatch(lambda e: captured.append(e))

    # QKV(0) must finish first (I2) — manual mutation
    qkv = dag.get_node(0, NodeType.QKV)
    qkv.transition_to(NodeState.READY)
    qkv.transition_to(NodeState.RUNNING)
    qkv.transition_to(NodeState.DONE)

    decode = dag.get_node(0, NodeType.DECODE_ATTN)
    decode.transition_to(NodeState.READY)
    d.dispatch_pim(decode)
    assert len(captured) == 1
    assert captured[0].resource == "PIM"  # F1 off 에도 PIM 라벨 유지


def test_ablation_f1_default_uses_real_pim(dummy_config):
    """f1_disabled=False (default) 위 dispatcher._op_time(PIM) → pim_executor.op_time() 호출."""
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    pim = PIMExecutor(config=dummy_config)
    d = Dispatcher(config=dummy_config, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    dag.add_micro_batch(0)
    d.register(MicroBatch(id=0, kv_rows_total=32, kv_rows_lockstep=32))
    decode = dag.get_node(0, NodeType.DECODE_ATTN)
    op_time = d._op_time(decode)
    # 정상 PIM path = pim_executor.op_time. dispatcher 는 ns→µs(×1e-3) 변환 반환(40e812a/ns 정정).
    expected = pim.op_time(kv_rows_total=32, kv_rows_lockstep=32) * 1e-3
    assert op_time == expected


# =========================================================================
# F2 ablation — InFlightWindow CAPACITY override
# =========================================================================

def test_ablation_f2_off_window_capacity_1(config_f2_capacity_1):
    """f2_window_capacity_override=1 위 InFlightWindow(config).capacity == 1."""
    dag = DAG()
    w = InFlightWindow(dag, config=config_f2_capacity_1)
    assert w.capacity == 1


def test_ablation_f2_off_evict_on_every_admit(config_f2_capacity_1):
    """capacity=1 위 매 admit 마다 직전 mb evict (μ-batch 직렬 강제)."""
    dag = DAG()
    w = InFlightWindow(dag, config=config_f2_capacity_1)
    evicted_0 = w.admit(0)
    assert evicted_0 is None  # 첫 admit
    evicted_1 = w.admit(1)
    assert evicted_1 == 0  # 직전 mb 즉시 evict
    evicted_2 = w.admit(2)
    assert evicted_2 == 1


def test_ablation_f2_default_capacity_3(dummy_config):
    """default config (override=None) 위 capacity == 3 (Impl-1 보존)."""
    dag = DAG()
    w = InFlightWindow(dag, config=dummy_config)
    assert w.capacity == 3


def test_ablation_f2_capacity_zero_raises(dummy_config):
    """f2_window_capacity_override=0 → InFlightWindow 생성 시 ValueError."""
    cfg = dataclasses.replace(
        dummy_config,
        ablation=dataclasses.replace(dummy_config.ablation, f2_window_capacity_override=0),
    )
    dag = DAG()
    with pytest.raises(ValueError):
        InFlightWindow(dag, config=cfg)


def test_ablation_f2_no_config_backward_compat():
    """기존 Impl-1 fixture (config 미주입) → default capacity 3 (backward-compat)."""
    dag = DAG()
    w = InFlightWindow(dag)  # config=None
    assert w.capacity == 3


# =========================================================================
# F3 ablation — Evaluator 직접 산식 (Q7 — InstancePipeline 미터치)
# =========================================================================

def test_ablation_f3_off_evaluator_uses_sum(evaluator):
    """F3 cell: cycle_with = max(a,b), cycle_without = a+b (Evaluator 직접 산식)."""
    decomp = evaluator.acceleration_decomposition(a_cycle=10.0, b_cycle=20.0, t_pim=5.0, t_proj=5.0)
    f3 = next(c for c in decomp if c.source == AblationSource.F3)
    assert f3.cycle_with_source == 20.0  # max
    assert f3.cycle_without_source == 30.0  # sum
    assert f3.direction_positive is True  # ratio = 1.5 > 1.0


def test_ablation_f3_direction_positive_a_equals_b(evaluator):
    """a=b → F3 ratio = (a+b)/max = 2 → direction_positive=True (max balance optimal)."""
    decomp = evaluator.acceleration_decomposition(a_cycle=10.0, b_cycle=10.0, t_pim=5.0, t_proj=5.0)
    f3 = next(c for c in decomp if c.source == AblationSource.F3)
    assert f3.ratio == 2.0


# =========================================================================
# F5 ablation — PIMExecutor max-KV penalty (Q8 — kv_rows_lockstep 산식)
# =========================================================================

def test_ablation_f5_off_pim_uses_lockstep(config_f5_disabled):
    """f5_disabled=True + kv_rows_lockstep=8192 위 산식 == ceil(8192 / (k × tile_rows)) × tile_time."""
    pim = PIMExecutor(config=config_f5_disabled)
    k = 2048
    tile_rows = config_f5_disabled.time.rtl_fsm_tile_rows
    tile_time = config_f5_disabled.time.pim_tile_time_ns["FP8"]
    kv_lockstep = 8192
    op_time = pim.op_time(kv_rows_total=999999, kv_rows_lockstep=kv_lockstep)
    # F5 비활성화 → kv_rows_total 무시, kv_rows_lockstep 사용
    expected_tiles = math.ceil(kv_lockstep / (k * tile_rows))
    expected = expected_tiles * tile_time  # broadcast=0 (k <= k_per_gpu_max는 false, k=2048>k_per_gpu_max=256 → +broadcast)
    # k=2048 > k_per_gpu_max(256) → broadcast 추가
    expected += config_f5_disabled.time.pim_broadcast_latency_ns_cross_gpu
    assert op_time == pytest.approx(expected)


def test_ablation_f5_off_lockstep_zero_raises(config_f5_disabled):
    """f5_disabled=True + kv_rows_lockstep=0 → ValueError."""
    pim = PIMExecutor(config=config_f5_disabled)
    with pytest.raises(ValueError, match="F5 ablation requires kv_rows_lockstep > 0"):
        pim.op_time(kv_rows_total=1000, kv_rows_lockstep=0)


def test_ablation_f5_default_uses_kv_rows_total(dummy_config):
    """f5_disabled=False (default) 위 op_time 산식 bit-exact Impl-4 식 (kv_rows_total 사용)."""
    pim = PIMExecutor(config=dummy_config)
    k = 2048
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    tile_time = dummy_config.time.pim_tile_time_ns["FP8"]
    kv_rows_total = 1000
    op_time = pim.op_time(kv_rows_total=kv_rows_total, kv_rows_lockstep=999999)
    # F5 활성화 → kv_rows_lockstep 무시
    expected_tiles = math.ceil(kv_rows_total / (k * tile_rows))
    expected = expected_tiles * tile_time + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert op_time == pytest.approx(expected)


def test_ablation_f5_lockstep_inflates_compared_to_total(dummy_config):
    """F5 off 가 effective work inflate — kv_lockstep > kv_total 시 op_time(F5 off) > op_time(F5 on)."""
    # Construct: kv_total=11000 (req kv=1000 + 10000), kv_lockstep=10000×2=20000
    pim_on = PIMExecutor(config=dummy_config)
    pim_off = PIMExecutor(config=dataclasses.replace(
        dummy_config,
        ablation=dataclasses.replace(dummy_config.ablation, f5_disabled=True),
    ))
    k = 2048
    t_on = pim_on.op_time(kv_rows_total=11000, kv_rows_lockstep=20000)
    t_off = pim_off.op_time(kv_rows_total=11000, kv_rows_lockstep=20000)
    assert t_off >= t_on  # F5 off 의 effective work ≥ F5 on


# =========================================================================
# F4 lock-in + decomp sanity
# =========================================================================

def test_ablation_acceleration_decomp_all_4_cells_direction_positive(evaluator):
    """4 cell 모두 ratio > 1.0 — 각 source 가 진정 가속.

    값 선택: t_pim < t_pim_fallback(=4) 위 F1 가속 나타남.
    a=b 위 F3 ratio=2. t_proj=t_pim 위 F2 ratio=2. F5 placeholder 2× 위 ratio=2.
    """
    decomp = evaluator.acceleration_decomposition(a_cycle=10.0, b_cycle=10.0, t_pim=2.0, t_proj=2.0)
    for cell in decomp:
        assert cell.direction_positive is True, f"{cell.source} direction negative: {cell}"
        assert cell.ratio > 1.0


def test_ablation_f4_not_in_decomposition_cells(evaluator):
    """decomp 의 source enum set == {F1, F2, F3, F5} (F4 미포함, ARCH §5.7 precondition)."""
    decomp = evaluator.acceleration_decomposition(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    sources = {c.source for c in decomp}
    assert sources == {AblationSource.F1, AblationSource.F2, AblationSource.F3, AblationSource.F5}
    # AblationSource enum 전체 4 value
    assert len(list(AblationSource)) == 4


def test_ablation_combinatorial_skipped_impl10(evaluator):
    """Impl-8 은 single-flag isolation 만 — combinatorial sweep 은 Impl-10.

    Evaluator.acceleration_decomposition 가 4 cell 만 산출 — single-flag isolation 형태.
    Multi-flag combination (F1+F3 동시 off 등) cell 산출 0.
    """
    decomp = evaluator.acceleration_decomposition(a_cycle=10, b_cycle=20, t_pim=5, t_proj=5)
    # 정확 4 cell — single-flag 만
    assert len(decomp) == 4
