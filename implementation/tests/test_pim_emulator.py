"""PIMExecutor unit tests.

Impl-10-pre-2 — k_channels knob 폐기 (sequence-parallel PIM 위 k 영원 k_aggregate).
op_time signature = (kv_rows_total, kv_rows_lockstep). channel count 는 PIMExecutor.k_aggregate
property 가 HWConfig 위 derive.

Test cluster:
- A.1 tile_time — regime lookup + ratio property + determinism
- A.2 op_time — kv_rows_total 위 산식 + monotonicity + ceil equiv
- A.3 broadcast overhead — 영원 cross-GPU (k_aggregate=2048)
- A.4 FSM determinism (PLAN §0.5 Impl-4 reminder)
- A.5 Ramulator2 loader — malformed cross-product
- A.6 ARCH literal property
- A.7 Boundary parametrize
- A.8 Computed Wait (stateless / pure function)
"""

import dataclasses
import inspect
import math
import random
from pathlib import Path

import pytest

from puls_sched.config import default_dummy_config
from puls_sched.pim_emulator import PIMExecutor


FIXTURE_DIR = Path(__file__).parent / "test_pim_loader_fixtures"


# =========================================================================
# A.1  tile_time — regime lookup
# =========================================================================

def test_tile_time_fp8_lookup_matches_config(pim_executor, dummy_config):
    assert pim_executor.tile_time() == dummy_config.time.pim_tile_time_ns["FP8"]


def test_tile_time_fp16_lookup_matches_config(pim_executor_fp16, dummy_config):
    assert pim_executor_fp16.tile_time() == dummy_config.time.pim_tile_time_ns["FP16"]


def test_tile_time_regime_ratio_property(pim_executor, pim_executor_fp16):
    """ARCH §6.6 의 placeholder ratio 2× property. 값 자체는 dummy."""
    assert pim_executor_fp16.tile_time() / pim_executor.tile_time() == 2.0


def test_tile_time_deterministic_1000_calls(pim_executor):
    expected = pim_executor.tile_time()
    for _ in range(1000):
        assert pim_executor.tile_time() == expected


def test_tile_time_kv_precision_swap_isolates_instance(pim_executor, dummy_config):
    """dataclasses.replace 위 새 PIMExecutor 가 기존 instance 영향 0 (frozen safety)."""
    model_fp16 = dataclasses.replace(dummy_config.model, kv_precision="FP16")
    cfg_fp16 = dataclasses.replace(dummy_config, model=model_fp16)
    new_exec = PIMExecutor(config=cfg_fp16)
    assert new_exec.tile_time() == dummy_config.time.pim_tile_time_ns["FP16"]
    assert pim_executor.tile_time() == dummy_config.time.pim_tile_time_ns["FP8"]


# =========================================================================
# A.2  op_time — sequence-parallel 산식 (k_aggregate 영원 k_max)
# =========================================================================

def test_op_time_zero_rows_returns_zero(pim_executor):
    """ARCH §5.1 pure-prefill batch → kv_rows_total = 0 → 0.0."""
    assert pim_executor.op_time(kv_rows_total=0) == 0.0


def test_op_time_single_tile_per_channel(pim_executor, dummy_config):
    """rows = k_aggregate × tile_rows → 1 tile per channel + broadcast."""
    k_agg = pim_executor.k_aggregate
    rows = k_agg * dummy_config.time.rtl_fsm_tile_rows
    expected = 1 * dummy_config.time.pim_tile_time_ns["FP8"] + \
        dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(kv_rows_total=rows) == expected


@pytest.mark.parametrize("rows1,rows2", [(1, 1000), (1000, 10000), (10000, 100000)])
def test_op_time_kv_rows_total_monotonic_increasing(pim_executor, rows1, rows2):
    """rows sweep 위 op_time 단조 비감소."""
    assert pim_executor.op_time(kv_rows_total=rows1) <= pim_executor.op_time(kv_rows_total=rows2)


def test_op_time_signature_no_k_channels(pim_executor):
    """Impl-10-pre-2 — signature 위 k_channels / n_decode / batch 영원 부재."""
    sig = inspect.signature(pim_executor.op_time)
    params = set(sig.parameters.keys())
    assert "k_channels" not in params
    assert "N_decode" not in params
    assert "n_decode" not in params
    assert "batch" not in params
    assert params == {"kv_rows_total", "kv_rows_lockstep"}


def test_op_time_ceil_equivalence_hermite_identity(pim_executor, dummy_config):
    """Hermite identity — `ceil(ceil(a/b)/c) == ceil(a/(b·c))`. 단일 ceil 산식 정합 검증."""
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    k_agg = pim_executor.k_aggregate
    rng = random.Random(42)
    for _ in range(100):
        rows = rng.randint(1, 1_000_000)
        single_ceil = math.ceil(rows / (k_agg * tile_rows))
        two_step = math.ceil(math.ceil(rows / k_agg) / tile_rows)
        assert single_ceil == two_step, f"Hermite divergence at rows={rows}"


# =========================================================================
# A.3  Broadcast overhead — 영원 cross-GPU (k_aggregate=2048 > per-GPU max=256)
# =========================================================================

def test_broadcast_always_applied(pim_executor, dummy_config):
    """k_aggregate 영원 cross-GPU → broadcast latency 항상 포함."""
    rows = 100
    k_agg = pim_executor.k_aggregate
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    per_channel = math.ceil(rows / (k_agg * tile_rows)) * pim_executor.tile_time()
    expected = per_channel + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(kv_rows_total=rows) == expected


# =========================================================================
# A.4  FSM determinism (PLAN §0.5 Impl-4 reminder)
# =========================================================================

def test_op_time_deterministic_1000_calls(pim_executor):
    """동일 rows 1000 회 → bit-exact (ARCH §3.5.2 jitter ±0)."""
    expected = pim_executor.op_time(kv_rows_total=50000)
    for _ in range(1000):
        assert pim_executor.op_time(kv_rows_total=50000) == expected


def test_op_time_deterministic_across_instances(dummy_config):
    """동일 config 의 2 PIMExecutor instance → bit-exact 동일 출력."""
    e1 = PIMExecutor(config=dummy_config)
    e2 = PIMExecutor(config=dummy_config)
    assert e1.op_time(kv_rows_total=30000) == e2.op_time(kv_rows_total=30000)


def test_op_time_no_rng_dependency(dummy_config):
    """seed 변경 → op_time 영향 0 (FSM 은 pure arithmetic)."""
    e1 = PIMExecutor(config=dummy_config)
    cfg2 = dataclasses.replace(dummy_config, seed=99999)
    e2 = PIMExecutor(config=cfg2)
    assert e1.op_time(kv_rows_total=50000) == e2.op_time(kv_rows_total=50000)


def test_tile_time_no_rng_dependency(dummy_config):
    e1 = PIMExecutor(config=dummy_config)
    cfg2 = dataclasses.replace(dummy_config, seed=99999)
    e2 = PIMExecutor(config=cfg2)
    assert e1.tile_time() == e2.tile_time()


# =========================================================================
# A.5  Ramulator2 loader — malformed cross-product
# =========================================================================

def test_loader_valid_minimal_round_trip():
    result = PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "valid_minimal.json")
    assert result == {"FP8": 32, "FP16": 64}


def test_loader_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ramulator2 cycles file not found"):
        PIMExecutor.load_ramulator2_cycles(tmp_path / "nonexistent.json")


def test_loader_empty_list_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "malformed_empty.json")


def test_loader_missing_field_raises():
    with pytest.raises(ValueError, match="missing field 'fsm_freq_ghz'"):
        PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "malformed_missing_field.json")


def test_loader_wrong_type_regime_raises():
    with pytest.raises(ValueError, match="'regime' must be str"):
        PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "malformed_wrong_type.json")


def test_loader_duplicate_regime_raises():
    with pytest.raises(ValueError, match="duplicate regime"):
        PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "malformed_duplicate_regime.json")


def test_loader_non_list_raises(tmp_path):
    p = tmp_path / "non_list.json"
    p.write_text('{"regime": "FP8"}')
    with pytest.raises(ValueError, match="non-empty list"):
        PIMExecutor.load_ramulator2_cycles(p)


def test_loader_wrong_type_tile_cycle_raises(tmp_path):
    p = tmp_path / "bad_tile.json"
    p.write_text('[{"regime": "FP8", "tile_cycle": "abc", "fsm_freq_ghz": 1.3}]')
    with pytest.raises(ValueError, match="'tile_cycle' must be non-negative int"):
        PIMExecutor.load_ramulator2_cycles(p)


def test_loader_negative_tile_cycle_raises(tmp_path):
    p = tmp_path / "neg_tile.json"
    p.write_text('[{"regime": "FP8", "tile_cycle": -1, "fsm_freq_ghz": 1.3}]')
    with pytest.raises(ValueError, match="'tile_cycle' must be non-negative int"):
        PIMExecutor.load_ramulator2_cycles(p)


def test_loader_entry_non_dict_raises(tmp_path):
    p = tmp_path / "bad_entry.json"
    p.write_text('[42, 100]')
    with pytest.raises(ValueError, match="entry must be dict"):
        PIMExecutor.load_ramulator2_cycles(p)


def test_loader_returns_independent_mapping():
    """반환 dict 가 caller mutation 후 loader internal state 영향 0 (staticmethod 정합)."""
    result = PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "valid_minimal.json")
    result["FP8"] = 99999
    result2 = PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "valid_minimal.json")
    assert result2 == {"FP8": 32, "FP16": 64}


# =========================================================================
# A.6  ARCH literal property
# =========================================================================

def test_arch_3_1_tile_rows_literal(dummy_config):
    """ARCH §3.1 '32-row tile FSM' literal."""
    assert dummy_config.time.rtl_fsm_tile_rows == 32


def test_arch_3_2_per_gpu_channel_count(dummy_config):
    """ARCH §3.2 '256 channels per-GPU' literal."""
    per_gpu = dummy_config.hw.num_stacks_per_gpu * dummy_config.hw.num_channels_per_stack
    assert per_gpu == 256


def test_arch_3_4_aggregate_channel_count(pim_executor, dummy_config):
    """ARCH §3.4 '2048 channels in total' literal — k_aggregate property 정합."""
    aggregate = (
        dummy_config.hw.num_gpus_instance_a
        * dummy_config.hw.num_stacks_per_gpu
        * dummy_config.hw.num_channels_per_stack
    )
    assert aggregate == 2048
    assert pim_executor.k_aggregate == 2048


def test_arch_6_6_regime_ratio_in_dummy_placeholder(dummy_config):
    """ARCH §6.6 'roughly 2×' placeholder ratio property. 값 자체 무의미, ordering 만."""
    ratio = (
        dummy_config.time.pim_tile_time_ns["FP16"]
        / dummy_config.time.pim_tile_time_ns["FP8"]
    )
    assert ratio == 2.0


def test_arch_5_1_pure_prefill_zero_rows(pim_executor):
    """ARCH §5.1 'For a pure-prefill batch, decode rows = 0' → op_time = 0."""
    assert pim_executor.op_time(kv_rows_total=0) == 0.0


# =========================================================================
# A.7  Boundary parametrize
# =========================================================================

@pytest.mark.parametrize("rows_offset", [-1, 0, 1])
def test_kv_rows_ceil_boundary_per_channel(pim_executor, dummy_config, rows_offset):
    """rows ∈ {k×T - 1, k×T, k×T + 1} ceil 경계 — k×T 에서 정확 1 tile, +1 에서 2."""
    k_agg = pim_executor.k_aggregate
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    rows_base = k_agg * tile_rows
    rows = rows_base + rows_offset
    expected_tiles = 1 if rows_offset <= 0 else 2
    per_channel = expected_tiles * pim_executor.tile_time()
    expected = per_channel + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(kv_rows_total=rows) == expected


def test_op_time_returns_nonnegative_random_sample(pim_executor):
    """20 random rows 위 op_time ≥ 0."""
    rng = random.Random(2026)
    for _ in range(20):
        rows = rng.randint(0, 1_000_000)
        assert pim_executor.op_time(kv_rows_total=rows) >= 0.0


# =========================================================================
# A.8  Computed Wait — pure function / stateless
# =========================================================================

def test_op_time_purely_deterministic_function_of_inputs(pim_executor):
    """op_time 이 순수 함수 — config 외 external state 의존 0."""
    call1 = pim_executor.op_time(kv_rows_total=50000)
    call2 = pim_executor.op_time(kv_rows_total=50000)
    assert call1 == call2


def test_pim_executor_is_stateless(pim_executor):
    """frozen dataclass + 호출 전후 instance state 변화 0."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        pim_executor.config = None  # type: ignore[misc]
    before = pim_executor.op_time(kv_rows_total=10000)
    _ = pim_executor.op_time(kv_rows_total=50000)
    after = pim_executor.op_time(kv_rows_total=10000)
    assert before == after
