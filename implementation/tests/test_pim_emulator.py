"""PIMExecutor unit tests.

Test cluster:
- A.1 tile_time — regime lookup + ratio property + determinism
- A.2 op_time — SP-PIM aggregate 산식 + monotonicity + batch invariance + ceil equiv
- A.3 broadcast overhead — single-GPU vs cross-GPU boundary
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
    # 기존 instance 영향 0
    assert pim_executor.tile_time() == dummy_config.time.pim_tile_time_ns["FP8"]


# =========================================================================
# A.2  op_time — SP-PIM aggregate 산식
# =========================================================================

def test_op_time_zero_channels_returns_zero(pim_executor):
    """ARCH §5.1 pure-prefill batch → k_total = 0."""
    assert pim_executor.op_time(k_channels=0, kv_rows_total=10000) == 0.0


def test_op_time_zero_rows_returns_broadcast_only(pim_executor, dummy_config):
    """rows=0 → tiles=0 → per_channel=0; k=2048 cross-GPU → broadcast applied."""
    result = pim_executor.op_time(k_channels=2048, kv_rows_total=0)
    assert result == dummy_config.time.pim_broadcast_latency_ns_cross_gpu


def test_op_time_single_tile_per_channel(pim_executor, dummy_config):
    """k=2048, rows = 2048 × 32 → ceil(65536 / (2048×32)) = 1 tile.
    per_channel = 1 × 1.0 = 1.0; broadcast = 0.5 → total = 1.5."""
    rows = 2048 * 32  # tile_rows=32
    expected = 1 * dummy_config.time.pim_tile_time_ns["FP8"] + \
        dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(k_channels=2048, kv_rows_total=rows) == expected


@pytest.mark.parametrize(
    "k_small,k_large",
    [(512, 1024), (1024, 1536), (1536, 2048), (512, 2048)],
)
def test_op_time_k_channels_monotonic_decreasing(pim_executor, k_small, k_large):
    """cross-GPU 영역 (k ≥ 512) 내부 consecutive + corner pair 위 op_time 단조 비증가.
    broadcast latency 동일 영역, per_channel 만 감소 (ARCH §3.4 KV-row sharding)."""
    rows = 100000
    assert pim_executor.op_time(k_large, rows) <= pim_executor.op_time(k_small, rows)


@pytest.mark.parametrize("rows1,rows2", [(0, 1000), (1000, 10000), (10000, 100000)])
def test_op_time_kv_rows_total_monotonic_increasing(pim_executor, rows1, rows2):
    """rows sweep 위 op_time 단조 비감소."""
    k = 2048
    assert pim_executor.op_time(k, rows1) <= pim_executor.op_time(k, rows2)


def test_op_time_kv_row_sharding_ratio_property(pim_executor, dummy_config):
    """ARCH §3.4 KV-row sharding — k 2× 시 per_channel_time 1/2 (ceil 정확 영역)."""
    # rows = 2 × k × tile_rows 위에서 정확 ratio
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    k = 1024
    rows = 2 * (2 * k) * tile_rows  # k=2048 위 2 tile, k=1024 위 4 tile
    # k=2048: ceil(rows / (2048×32)) = 2; k=1024: ceil(rows / (1024×32)) = 4
    t_k2048_per_channel = 2 * pim_executor.tile_time()
    t_k1024_per_channel = 4 * pim_executor.tile_time()
    actual_k2048 = pim_executor.op_time(2048, rows) - \
        dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    actual_k1024 = pim_executor.op_time(1024, rows) - \
        dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert actual_k2048 == t_k2048_per_channel
    assert actual_k1024 == t_k1024_per_channel
    assert actual_k1024 / actual_k2048 == 2.0  # exact 1/2 sharding


def test_op_time_batch_dim_invariance_via_signature(pim_executor):
    """ARCH §3.1 'FSM cycle structure invariant' — signature 에 batch arg 부재 (구조 강제).

    Impl-8 — kv_rows_lockstep 추가 (F5 ablation signal flow, default 0 backward-compat).
    batch arg (N_decode / n_decode) 는 여전히 부재 (ARCH §3.1 invariance 보존).
    """
    sig = inspect.signature(pim_executor.op_time)
    params = set(sig.parameters.keys())
    assert "N_decode" not in params
    assert "batch" not in params
    assert "n_decode" not in params
    assert params == {"k_channels", "kv_rows_total", "kv_rows_lockstep"}


def test_op_time_ceil_equivalence_hermite_identity(pim_executor, dummy_config):
    """Hermite identity (Q11) — `ceil(ceil(a/b)/c) == ceil(a/(b·c))` for positive integers.
    plan 의 단일 ceil 산식과 두 단계 ceil 이 *수학적 등가* 임을 random sample 위 검증."""
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    rng = random.Random(42)
    for _ in range(100):
        k = rng.choice([256, 512, 768, 1024, 1280, 1536, 1792, 2048])
        rows = rng.randint(1, 1_000_000)
        single_ceil = math.ceil(rows / (k * tile_rows))
        two_step = math.ceil(math.ceil(rows / k) / tile_rows)
        assert single_ceil == two_step, f"Hermite divergence at k={k}, rows={rows}"


@pytest.mark.parametrize("n_decode", [1, 8, 64, 256, 1024, 4096])
def test_op_time_batch_dim_invariance_behavioral_sweep(pim_executor, n_decode):
    """행동 검증 (Q12) — caller 가 batch 다른 시나리오 위 동일 (k, rows) 호출 → bit-exact 동일.
    PIMExecutor.op_time 이 외부 batch context 와 무관함을 시연. ARCH §3.1 정합."""
    # caller scenario simulation: n_decode 변화에도 op_time call args 는 동일
    k = 2048
    rows = 65536
    # n_decode 와 무관하게 동일 args 로 호출
    result_baseline = pim_executor.op_time(k, rows)
    # 같은 instance 재호출 (n_decode 변화 시뮬레이션)
    result_after_batch_change = pim_executor.op_time(k, rows)
    assert result_baseline == result_after_batch_change
    # 추가: signature 가 n_decode 받지 않으므로 *어떤 caller* 도 전달 불가
    assert "n_decode" not in inspect.signature(pim_executor.op_time).parameters


# =========================================================================
# A.3  Broadcast overhead — boundary
# =========================================================================

def test_broadcast_zero_at_single_gpu_max(pim_executor, dummy_config):
    """k=256 = num_stacks × num_channels → single-GPU only, broadcast 0."""
    # rows = 256 × 32 → 1 tile per channel
    rows = 256 * 32
    expected = 1 * pim_executor.tile_time()  # broadcast 0
    assert pim_executor.op_time(256, rows) == expected


def test_broadcast_applied_at_257(pim_executor, dummy_config):
    """k=257 → cross-GPU 진입, broadcast = pim_broadcast_latency_ns_cross_gpu."""
    rows = 257 * 32
    expected = (
        math.ceil(rows / (257 * 32)) * pim_executor.tile_time()
        + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    )
    assert pim_executor.op_time(257, rows) == expected


def test_broadcast_boundary_exact_diff(pim_executor, dummy_config):
    """k=257 op_time - k=256 op_time 의 *broadcast 차이* 가 정확히 latency."""
    # rows 를 두 경우 모두 1 tile per channel 로 정규화
    rows_256 = 256 * 32  # k=256: 1 tile
    rows_257 = 257 * 32  # k=257: ceil(257×32 / (257×32)) = 1 tile
    t_256 = pim_executor.op_time(256, rows_256)
    t_257 = pim_executor.op_time(257, rows_257)
    diff = t_257 - t_256
    assert diff == dummy_config.time.pim_broadcast_latency_ns_cross_gpu


@pytest.mark.parametrize("k", [512, 1024, 1536, 2048])
def test_broadcast_constant_across_cross_gpu(pim_executor, dummy_config, k):
    """cross-GPU 영역 모든 k 위 broadcast contribution 동일 (binary model)."""
    rows = 100
    # per_channel_time = ceil(100 / (k×32)) × tile_time = 1 × 1.0 = 1.0 (k×32 > 100)
    per_channel = 1 * pim_executor.tile_time()
    expected = per_channel + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(k, rows) == expected


@pytest.mark.parametrize("k", [1, 32, 64, 128, 256])
def test_broadcast_zero_across_single_gpu_region(pim_executor, k):
    """single-GPU 영역 (k ≤ 256) 모든 값 위 broadcast == 0."""
    rows = 100
    per_channel = math.ceil(rows / (k * 32)) * pim_executor.tile_time()
    assert pim_executor.op_time(k, rows) == per_channel


# =========================================================================
# A.4  FSM determinism (PLAN §0.5 Impl-4 reminder)
# =========================================================================

def test_op_time_deterministic_1000_calls(pim_executor):
    """동일 (k, rows) 1000 회 → bit-exact (ARCH §3.5.2 jitter ±0)."""
    expected = pim_executor.op_time(2048, 50000)
    for _ in range(1000):
        assert pim_executor.op_time(2048, 50000) == expected


def test_op_time_deterministic_across_instances(dummy_config):
    """동일 config 의 2 PIMExecutor instance → bit-exact 동일 출력."""
    e1 = PIMExecutor(config=dummy_config)
    e2 = PIMExecutor(config=dummy_config)
    assert e1.op_time(1024, 30000) == e2.op_time(1024, 30000)


def test_op_time_no_rng_dependency(dummy_config):
    """seed 변경 → op_time 영향 0 (FSM 은 pure arithmetic)."""
    e1 = PIMExecutor(config=dummy_config)
    cfg2 = dataclasses.replace(dummy_config, seed=99999)
    e2 = PIMExecutor(config=cfg2)
    assert e1.op_time(2048, 50000) == e2.op_time(2048, 50000)


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
    """top-level non-list → ValueError."""
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
    result["FP8"] = 99999  # caller modification
    # 재load 시 영향 0
    result2 = PIMExecutor.load_ramulator2_cycles(FIXTURE_DIR / "valid_minimal.json")
    assert result2 == {"FP8": 32, "FP16": 64}


# =========================================================================
# A.6  ARCH literal property
# =========================================================================

def test_arch_3_1_tile_rows_literal(dummy_config):
    """ARCH §3.1 '32-row tile FSM' literal."""
    assert dummy_config.time.rtl_fsm_tile_rows == 32


def test_arch_3_2_per_gpu_channel_count(dummy_config):
    """ARCH §3.2 '256 channels per-GPU' literal — boundary 의 substrate 근거."""
    per_gpu = dummy_config.hw.num_stacks_per_gpu * dummy_config.hw.num_channels_per_stack
    assert per_gpu == 256


def test_arch_3_4_aggregate_channel_count(dummy_config):
    """ARCH §3.4 '2048 channels in total' literal."""
    aggregate = (
        dummy_config.hw.num_gpus_instance_a
        * dummy_config.hw.num_stacks_per_gpu
        * dummy_config.hw.num_channels_per_stack
    )
    assert aggregate == 2048


def test_arch_6_6_regime_ratio_in_dummy_placeholder(dummy_config):
    """ARCH §6.6 'roughly 2×' 의 placeholder ratio property. 값 자체 무의미, ordering 만."""
    ratio = (
        dummy_config.time.pim_tile_time_ns["FP16"]
        / dummy_config.time.pim_tile_time_ns["FP8"]
    )
    assert ratio == 2.0


def test_arch_5_1_pure_prefill_zero_k_total(pim_executor):
    """ARCH §5.1 'For a pure-prefill batch, k_total = 0'."""
    assert pim_executor.op_time(k_channels=0, kv_rows_total=1) == 0.0
    assert pim_executor.op_time(k_channels=0, kv_rows_total=1_000_000) == 0.0


# =========================================================================
# A.7  Boundary parametrize cross-product
# =========================================================================

@pytest.mark.parametrize(
    "rows_offset", [-1, 0, 1]
)
def test_kv_rows_ceil_boundary_per_channel(pim_executor, dummy_config, rows_offset):
    """rows ∈ {k×T - 1, k×T, k×T + 1} ceil 경계 — k×T 에서 정확 1 tile, +1 에서 2."""
    k = 2048
    tile_rows = dummy_config.time.rtl_fsm_tile_rows
    rows_base = k * tile_rows
    rows = rows_base + rows_offset
    expected_tiles = 1 if rows_offset <= 0 else 2
    # rows_offset = -1 → ceil((k×T - 1) / (k×T)) = 1; offset=0 → 1; offset=1 → 2
    per_channel = expected_tiles * pim_executor.tile_time()
    expected = per_channel + dummy_config.time.pim_broadcast_latency_ns_cross_gpu
    assert pim_executor.op_time(k, rows) == expected


@pytest.mark.parametrize("k", [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048])
def test_k_channels_sweep_full_dial(pim_executor, k):
    """k_total dial 9-step 전수 위 op_time 산출 정상 (raise 0)."""
    result = pim_executor.op_time(k, 10000)
    assert result >= 0.0


def test_op_time_returns_nonnegative_random_sample(pim_executor):
    """20 random (k, rows) cases 위 op_time ≥ 0."""
    rng = random.Random(2026)
    for _ in range(20):
        k = rng.choice([0, 256, 512, 1024, 2048])
        rows = rng.randint(0, 1_000_000)
        assert pim_executor.op_time(k, rows) >= 0.0


# =========================================================================
# A.8  Computed Wait 의미 (ARCH §3.5.2) — pure function / stateless
# =========================================================================

def test_op_time_purely_deterministic_function_of_inputs(pim_executor):
    """op_time 이 순수 함수 — config 외 external state 의존 0."""
    call1 = pim_executor.op_time(2048, 50000)
    # 다른 호출 사이에 어떤 외부 변화 도 없음 (clock/queue 의존 0)
    call2 = pim_executor.op_time(2048, 50000)
    assert call1 == call2


def test_pim_executor_is_stateless(pim_executor):
    """frozen dataclass + 호출 전후 instance state 변화 0."""
    # frozen 검증 — assignment 시도 시 FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        pim_executor.config = None  # type: ignore[misc]
    # 호출 전후 같은 출력 (stateless 정합)
    before = pim_executor.op_time(1024, 10000)
    _ = pim_executor.op_time(2048, 50000)  # 다른 호출
    after = pim_executor.op_time(1024, 10000)
    assert before == after
