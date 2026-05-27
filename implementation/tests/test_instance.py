"""Impl-5 — Instance 자원 추적 (ARCH §3.4 Case A 정합).

GPU = 1 자원 (TP=8 lock-step, Q2). Instance A 는 PIM 보유, B 는 미보유.
"""

import pytest

from puls_sched.instance import Instance


# =========================================================================
# Basic — Case A 구조 (ARCH §3.4 literal)
# =========================================================================

def test_instance_a_has_pim(instance_a):
    """Instance A 는 PIM 보유 + acquire/release_pim 정상."""
    assert instance_a.has_pim is True
    instance_a.acquire_pim()
    assert instance_a.pim_busy is True
    instance_a.release_pim()
    assert instance_a.pim_busy is False


def test_instance_b_no_pim(instance_b):
    """ARCH §3.4 Case A — Instance B no PIM. acquire_pim 시 raise."""
    assert instance_b.has_pim is False
    with pytest.raises(RuntimeError, match="no PIM"):
        instance_b.acquire_pim()


# =========================================================================
# GPU 자원 (TP=8 lock-step = 1 자원)
# =========================================================================

def test_gpu_acquire_release_roundtrip(instance_a):
    """acquire → release → acquire → release 2 회 round-trip."""
    for _ in range(2):
        instance_a.acquire_gpu()
        assert instance_a.gpu_busy is True
        instance_a.release_gpu()
        assert instance_a.gpu_busy is False


def test_gpu_double_acquire_raises(instance_a):
    instance_a.acquire_gpu()
    with pytest.raises(RuntimeError, match="GPU already busy"):
        instance_a.acquire_gpu()


def test_gpu_double_release_raises(instance_a):
    with pytest.raises(RuntimeError, match="GPU not busy"):
        instance_a.release_gpu()


# =========================================================================
# PIM 자원 (Instance A 한정)
# =========================================================================

def test_pim_acquire_release_roundtrip(instance_a):
    for _ in range(2):
        instance_a.acquire_pim()
        assert instance_a.pim_busy is True
        instance_a.release_pim()
        assert instance_a.pim_busy is False


def test_pim_double_acquire_raises(instance_a):
    instance_a.acquire_pim()
    with pytest.raises(RuntimeError, match="PIM already busy"):
        instance_a.acquire_pim()


def test_pim_release_without_acquire_raises(instance_a):
    with pytest.raises(RuntimeError, match="PIM not busy"):
        instance_a.release_pim()


def test_pim_on_instance_b_release_raises(instance_b):
    """ARCH §3.4 — Instance B 위 release_pim 도 raise."""
    with pytest.raises(RuntimeError, match="no PIM"):
        instance_b.release_pim()


# =========================================================================
# GPU·PIM 독립성 (ARCH §3.2 channel-level independence prefigure)
# =========================================================================

def test_gpu_pim_independent_resources(instance_a):
    """Instance A — GPU busy 상태에서 PIM 자원 점유 가능 (직교 invariant)."""
    instance_a.acquire_gpu()
    instance_a.acquire_pim()
    assert instance_a.gpu_busy is True and instance_a.pim_busy is True
    instance_a.release_gpu()
    assert instance_a.gpu_busy is False and instance_a.pim_busy is True
    instance_a.release_pim()
    assert instance_a.gpu_busy is False and instance_a.pim_busy is False


# =========================================================================
# Determinism — 동일 sequence 위 동일 final state
# =========================================================================

def test_acquire_release_deterministic_state(instance_a):
    """10 회 동일 sequence (acquire_gpu → release → acquire_pim → release) → final 일관."""
    for _ in range(10):
        instance_a.acquire_gpu()
        instance_a.release_gpu()
        instance_a.acquire_pim()
        instance_a.release_pim()
    assert instance_a.gpu_busy is False
    assert instance_a.pim_busy is False
