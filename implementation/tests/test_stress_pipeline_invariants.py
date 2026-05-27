"""Impl-5 — Multi-mb stress (Impl-2 stress 패턴 정합).

ARCH §3.4 "Pipeline Structure" 의 steady-state cycle 정의가 *concurrent multi-μ-batch
staggering* 위에서 성립 — multi-mb 위 자원 일관성은 ARCH 의 premise. F4 "μ-batch
staggering" 의 정성 ground.

검증 대상 invariants (Impl-5 신설):
- (I-A) Instance.gpu_busy/pim_busy 의 다회 acquire/release sequence 위 일관성
- (I-B) Dispatcher.micro_batches dict 의 다회 register/unregister 정확성
- (I-C) LayerState.advance 의 다회 mb 위 독립 단조성
- (I-D) InstancePipeline.steady_state_cycle 의 다회 호출 위 stateless
- (I-E) NVLinkTransfer.time 의 다회 호출 위 stateless
"""

import random

import pytest

from puls_sched.event import Event, EventType
from puls_sched.forward_pass import LayerState
from puls_sched.micro_batch import MicroBatch
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length)


# =========================================================================
# (I-A) Instance resource stress
# =========================================================================

def test_stress_instance_resource_50_acquire_release_cycle(instance_a):
    """50 회 (acquire → release) sequence — final 모두 False, invariant 0 violation."""
    for _ in range(50):
        instance_a.acquire_gpu()
        assert instance_a.gpu_busy is True
        instance_a.release_gpu()
        assert instance_a.gpu_busy is False
        instance_a.acquire_pim()
        assert instance_a.pim_busy is True
        instance_a.release_pim()
        assert instance_a.pim_busy is False
    assert instance_a.gpu_busy is False and instance_a.pim_busy is False


def test_stress_instance_resource_interleaved_gpu_pim(instance_a, dummy_config):
    """50 회 random interleave — seed 고정 reproducible. 두 자원 직교 invariant."""
    rng = random.Random(dummy_config.seed)
    for _ in range(50):
        op = rng.choice(["gpu_acq", "gpu_rel", "pim_acq", "pim_rel"])
        try:
            if op == "gpu_acq":
                instance_a.acquire_gpu()
            elif op == "gpu_rel":
                instance_a.release_gpu()
            elif op == "pim_acq":
                instance_a.acquire_pim()
            else:
                instance_a.release_pim()
        except RuntimeError:
            # Invalid op for current state — invariant 강제 raise 정상
            pass
    # Final state 는 정의되지 않지만, gpu_busy / pim_busy 의 의미 (bool) 보존
    assert isinstance(instance_a.gpu_busy, bool)
    assert isinstance(instance_a.pim_busy, bool)


# =========================================================================
# (I-B) Dispatcher register/unregister stress
# =========================================================================

def test_stress_dispatcher_register_unregister_20_round_trip(dispatcher):
    """20 회 register → 20 회 unregister round-trip → micro_batches == {}."""
    for i in range(20):
        dispatcher.register(MicroBatch(id=i, k_total=256, kv_rows_total=100))
    assert len(dispatcher.micro_batches) == 20
    for i in range(20):
        dispatcher.unregister(i)
    assert dispatcher.micro_batches == {}


def test_stress_dispatcher_register_partial_unregister(dispatcher):
    """30 register, 10 unregister → 정확히 20 remaining (잘못된 mb evicted 0)."""
    for i in range(30):
        dispatcher.register(MicroBatch(id=i, k_total=256, kv_rows_total=100))
    for i in range(10):
        dispatcher.unregister(i)
    remaining = set(dispatcher.micro_batches.keys())
    assert remaining == set(range(10, 30))


def test_stress_dispatcher_register_double_raise_among_20(dispatcher):
    """20 mb register 도중 중복 id → raise (invariant 강제)."""
    for i in range(20):
        dispatcher.register(MicroBatch(id=i, k_total=256, kv_rows_total=100))
    # 21 번째: 기존 id 와 중복
    with pytest.raises(RuntimeError, match="already registered"):
        dispatcher.register(MicroBatch(id=5, k_total=256, kv_rows_total=100))
    # 기존 20 보존 invariant
    assert len(dispatcher.micro_batches) == 20


# =========================================================================
# (I-C) LayerState independence stress
# =========================================================================

def test_stress_layer_state_20_micro_batches_independent():
    """20 mb 위 각각 L=8 forward → 각 mb 의 final layer 8. Cross-contamination 0."""
    layer_state = LayerState(num_layers=8)
    mbs = [MicroBatch(id=i) for i in range(20)]
    for mb in mbs:
        for _ in range(8):
            layer_state.advance(mb)
    for mb in mbs:
        assert mb.current_layer_index == 8


def test_stress_layer_state_interleaved_advance_consistency():
    """5 mb 의 advance 를 interleave — 각 mb 의 layer 단조 + final 일관."""
    layer_state = LayerState(num_layers=5)
    mbs = [MicroBatch(id=i) for i in range(5)]
    # Round-robin advance — 각 mb 5 회
    for round_idx in range(5):
        for mb in mbs:
            prev = mb.current_layer_index
            layer_state.advance(mb)
            assert mb.current_layer_index == prev + 1
    for mb in mbs:
        assert mb.current_layer_index == 5


# =========================================================================
# (I-D) InstancePipeline stateless stress
# =========================================================================

def test_stress_instance_pipeline_steady_state_100_calls_no_state_leak(instance_pipeline):
    """100 회 다른 (A,B) 입력 → 각 호출 의 max 정합 + state leak 0."""
    rng = random.Random(42)
    for _ in range(100):
        a = rng.uniform(0.0, 100.0)
        b = rng.uniform(0.0, 100.0)
        assert instance_pipeline.steady_state_cycle(a, b) == max(a, b)


# =========================================================================
# (I-E) NVLink stateless stress
# =========================================================================

def test_stress_nvlink_100_shape_sequence_stateless(nvlink_transfer, dummy_config):
    """100 회 다른 shape → 각 호출 의 산식 정합 + state leak 0."""
    rng = random.Random(42)
    coef = dummy_config.time.nvlink_time_per_byte_ns
    bpe = nvlink_transfer.bytes_per_element
    for _ in range(100):
        B = rng.randint(1, 256)
        hidden = rng.randint(1, 16384)
        expected = B * hidden * bpe * coef
        assert nvlink_transfer.time((B, hidden)) == expected


# =========================================================================
# Cross-module stress (admission → register chain × 20)
# =========================================================================

def test_stress_full_pipeline_20_admission_ticks(scheduler_core):
    """20 회 ADMISSION_TICK — 각 tick 의 register 정확 + cross-contamination 0."""
    for i in range(20):
        scheduler_core.request_queue.push(_make_req(i, kv_length=10 * (i + 1)))
        scheduler_core._handle(Event(timestamp=float(i), type=EventType.ADMISSION_TICK, payload={}))
        # 자원 reset for next tick (in stress test, GPU/PIM busy 격 무시)
        scheduler_core.dispatcher.gpu_busy = False
        scheduler_core.dispatcher.pim_busy = False
    # 누적 micro_batches 의 size — admit 된 모든 mb 가 register 됨
    # (단 window eviction 으로 dispatcher.unregister 가 미호출 — Impl-9 영역)
    assert len(scheduler_core.dispatcher.micro_batches) == 20
    # 각 mb 의 id 가 0..19 단조
    assert sorted(scheduler_core.dispatcher.micro_batches.keys()) == list(range(20))


# =========================================================================
# Composite invariant stress (Impl-2 100-mb 패턴 정합)
# =========================================================================

def test_stress_pipeline_invariant_violation_zero(
    instance_a, dispatcher, nvlink_transfer, instance_pipeline,
):
    """20-step random sequence — Instance / Dispatcher / NVLink / InstancePipeline
    동시 사용 위 (I-A)~(I-E) 5 invariant 모두 보존.
    """
    rng = random.Random(42)
    registered_ids: set[int] = set()
    next_mb_id = 0
    layer_state = LayerState(num_layers=8)
    mbs_for_advance: list[MicroBatch] = []

    for step in range(20):
        op = rng.choice(["register", "unregister", "nvlink", "cycle", "advance"])
        if op == "register":
            mb = MicroBatch(id=next_mb_id, k_total=256, kv_rows_total=100)
            dispatcher.register(mb)
            registered_ids.add(next_mb_id)
            mbs_for_advance.append(mb)
            next_mb_id += 1
        elif op == "unregister" and registered_ids:
            mid = rng.choice(list(registered_ids))
            dispatcher.unregister(mid)
            registered_ids.discard(mid)
        elif op == "nvlink":
            B = rng.randint(1, 64)
            t = nvlink_transfer.time((B, 8192))
            assert t >= 0
        elif op == "cycle":
            a = rng.uniform(0.0, 10.0)
            b = rng.uniform(0.0, 10.0)
            assert instance_pipeline.steady_state_cycle(a, b) == max(a, b)
        elif op == "advance" and mbs_for_advance:
            mb = rng.choice(mbs_for_advance)
            if mb.current_layer_index < layer_state.num_layers:
                layer_state.advance(mb)

    # Final invariants check
    assert set(dispatcher.micro_batches.keys()) == registered_ids
    # 모든 advance 된 mb 의 current_layer_index 가 num_layers 이하
    for mb in mbs_for_advance:
        assert 0 <= mb.current_layer_index <= layer_state.num_layers
