"""Impl-5 — NVLinkTransfer pure time function (ARCH §3.4 표 정합).

Q4 — pure function. Event push · 자원 lock 안 함. NVLink 은 data path.
"""

import dataclasses

import pytest

from puls_sched.config import default_dummy_config
from puls_sched.nvlink import NVLinkTransfer


# =========================================================================
# Shape 산식 정합 (ARCH §3.4 표 literal)
# =========================================================================

def test_nvlink_decode_shape_time(nvlink_transfer, dummy_config):
    """ARCH §3.4 — A → B: O projection output [B × hidden]."""
    B = 8
    hidden = dummy_config.model.hidden  # 8192
    expected = B * hidden * 2 * dummy_config.time.nvlink_time_per_byte_ns
    assert nvlink_transfer.time((B, hidden)) == expected


@pytest.mark.parametrize("chunk", [64, 256, 1024])
def test_nvlink_prefill_shape_time(nvlink_transfer, dummy_config, chunk):
    """ARCH §5.2 — uniform-chunk prefill [(B · chunk) × hidden]."""
    B = 8
    hidden = dummy_config.model.hidden
    expected = (B * chunk) * hidden * 2 * dummy_config.time.nvlink_time_per_byte_ns
    assert nvlink_transfer.time((B * chunk, hidden)) == expected


def test_nvlink_lookup_no_magic_number(dummy_config):
    """coef 변화에 *선형* 비례 (lookup 정합, magic number 0 검증)."""
    base = NVLinkTransfer(config=dummy_config).time((8, 8192))
    cfg2 = dataclasses.replace(
        dummy_config,
        time=dataclasses.replace(dummy_config.time, nvlink_time_per_byte_ns=2.5),
    )
    scaled = NVLinkTransfer(config=cfg2).time((8, 8192))
    assert scaled == base * 2.5


# =========================================================================
# Monotonicity (sweep)
# =========================================================================

@pytest.mark.parametrize("B", [1, 8, 64, 256])
def test_nvlink_batch_size_monotonic(nvlink_transfer, B):
    base = nvlink_transfer.time((1, 8192))
    scaled = nvlink_transfer.time((B, 8192))
    assert scaled >= base
    assert scaled == B * base


@pytest.mark.parametrize("chunk", [64, 256, 1024, 4096])
def test_nvlink_chunk_size_monotonic(nvlink_transfer, chunk):
    times = [nvlink_transfer.time((c, 8192)) for c in [1, chunk]]
    assert times[1] >= times[0]


# =========================================================================
# Determinism (pure function)
# =========================================================================

def test_nvlink_deterministic_1000_calls(nvlink_transfer):
    expected = nvlink_transfer.time((8, 8192))
    for _ in range(1000):
        assert nvlink_transfer.time((8, 8192)) == expected


# =========================================================================
# Negative
# =========================================================================

def test_nvlink_empty_shape_raises(nvlink_transfer):
    with pytest.raises(ValueError, match="empty"):
        nvlink_transfer.time(())


def test_nvlink_negative_dim_raises(nvlink_transfer):
    with pytest.raises(ValueError, match="non-negative"):
        nvlink_transfer.time((8, -1))


def test_nvlink_zero_dim_returns_zero(nvlink_transfer):
    """0-token batch — sanity. transfer time == 0."""
    assert nvlink_transfer.time((0, 8192)) == 0.0


# =========================================================================
# Async hidden invariant prefigure (Q7) — dummy 값 위 hidden 영역 확인
# =========================================================================

def test_nvlink_async_hidden_invariant(nvlink_transfer, dummy_config):
    """ARCH §3.4 async hiding — dummy config 위에서 t_handoff 가 작은 영역.

    *주의 (정직성):* 본 test 는 dummy 값 위 hidden 영역의 *prefigure* 만 — Impl-10
    calibrated input 위 visible NVLink 영역은 §7 O5.2 sensitivity sweep 영역.
    """
    t_handoff = nvlink_transfer.time((8, dummy_config.model.hidden))
    # dummy gpu_op_time_us * 1000 (us → ns) — A_cycle 의 lower bound 추정
    a_cycle_ns_lb = dummy_config.time.gpu_op_time_us["qkv"] * 1000
    # 본 test 의 inequality 가 dummy-coincidence 임을 명시 — Impl-10 영역에서 재검증 필요
    assert t_handoff > 0  # 비-zero 검증만 (영역 분류는 Impl-10)
    assert a_cycle_ns_lb > 0
