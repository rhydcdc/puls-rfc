"""Cluster T — IdleTelemetry 3-key 분리 (O8.1).

Impl-10-pre-1 정합. 3 slot (`gpu_instance_a` · `pim_instance_a` · `gpu_instance_b`) 의 독립
누적 + legacy 키 ("GPU"/"PIM") 의 backward-compat mapping + Evaluator.idle_fraction() 의 3-key
schema lock-in.

ARCH §6.4 admission balance 의 *inter-AB (Instance B GPU 활동) · intra-A (Instance A GPU·PIM 활동)*
두 축 정확 반영.
"""

import pytest

from puls_sched.idle_telemetry import IdleTelemetry


# ---- 3-slot 독립 누적 ----

def test_three_slots_independent_accumulation():
    """3 slot 이 각각 독립 누적 — 한 slot 의 record 가 다른 slot 영향 0."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("gpu_instance_a", 0.0, 5.0)
    tel.record_active("pim_instance_a", 0.0, 3.0)
    tel.record_active("gpu_instance_b", 0.0, 2.0)
    # window_end = 5 (max). idle = 1 - active/5
    assert tel.idle_fraction("gpu_instance_a") == pytest.approx(1.0 - 5.0 / 5.0)
    assert tel.idle_fraction("pim_instance_a") == pytest.approx(1.0 - 3.0 / 5.0)
    assert tel.idle_fraction("gpu_instance_b") == pytest.approx(1.0 - 2.0 / 5.0)


def test_three_slots_zero_when_unrecorded():
    """미 record slot 은 fully idle (1.0) — window 가 다른 slot 으로 extend 된 후."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("gpu_instance_a", 0.0, 10.0)   # window_end → 10
    # 미 record slot 들
    assert tel.idle_fraction("pim_instance_a") == 1.0
    assert tel.idle_fraction("gpu_instance_b") == 1.0


# ---- Legacy backward-compat mapping ----

def test_legacy_GPU_maps_to_gpu_instance_a():
    """레거시 'GPU' 키 → gpu_instance_a slot (Impl-3 caller 영역 backward-compat)."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 5.0)
    assert tel.idle_fraction("gpu_instance_a") == tel.idle_fraction("GPU")
    # pim slot 미 record — gpu_instance_a 만 누적
    assert tel.gpu_idle_fraction() == 0.0  # active=5, span=5 → idle=0


def test_legacy_PIM_maps_to_pim_instance_a():
    """레거시 'PIM' 키 → pim_instance_a slot."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("PIM", 0.0, 7.0)
    assert tel.idle_fraction("pim_instance_a") == tel.idle_fraction("PIM")
    assert tel.pim_idle_fraction() == 0.0


def test_legacy_and_new_keys_share_slot():
    """레거시 'GPU' + 신규 'gpu_instance_a' 가 동일 slot 누적."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("GPU", 0.0, 3.0)
    tel.record_active("gpu_instance_a", 5.0, 7.0)
    # 동일 slot 에 5 누적 (3 + 2). window_end = 7.
    assert tel.gpu_idle_fraction() == pytest.approx(1.0 - 5.0 / 7.0)


# ---- Unknown 키 raise ----

def test_unknown_key_record_raises():
    tel = IdleTelemetry()
    tel.reset(0.0)
    with pytest.raises(ValueError, match="unknown resource"):
        tel.record_active("CPU", 0.0, 1.0)


def test_unknown_key_idle_fraction_raises():
    tel = IdleTelemetry()
    tel.reset(0.0)
    with pytest.raises(ValueError, match="unknown resource"):
        tel.idle_fraction("CPU")


@pytest.mark.parametrize("bad_key", ["", "gpu", "pim", "gpu_b", "GPU_A", "instance_a"])
def test_typos_and_partial_keys_raise(bad_key):
    """Typo / partial key 영역 의 cross-product fail-fast — strict schema lock-in."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    with pytest.raises(ValueError, match="unknown resource"):
        tel.record_active(bad_key, 0.0, 1.0)


# ---- 새 helper method ----

def test_gpu_instance_b_idle_fraction_helper():
    """gpu_instance_b_idle_fraction() helper — Instance B 의 GPU idle (Impl-10-pre-1 신설)."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("gpu_instance_b", 0.0, 4.0)
    tel.record_active("gpu_instance_a", 4.0, 10.0)  # window_end → 10
    # gpu_b: active=4, span=10 → idle = 0.6
    assert tel.gpu_instance_b_idle_fraction() == pytest.approx(0.6)


def test_reset_clears_all_three_slots():
    """reset 후 3 slot 모두 0 으로 초기화."""
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("gpu_instance_a", 0.0, 5.0)
    tel.record_active("pim_instance_a", 0.0, 3.0)
    tel.record_active("gpu_instance_b", 0.0, 2.0)
    tel.reset(100.0)
    tel.record_active("gpu_instance_a", 100.0, 110.0)
    assert tel.gpu_idle_fraction() == 0.0  # 새 window 위 fully active
    assert tel.pim_idle_fraction() == 1.0  # 새 window 위 미 record
    assert tel.gpu_instance_b_idle_fraction() == 1.0


# ---- Evaluator.idle_fraction() 3-key schema lock-in ----

def test_evaluator_idle_fraction_returns_three_keys(evaluator):
    """Evaluator.idle_fraction() schema = 3 key (gpu_instance_a · pim_instance_a · gpu_instance_b)."""
    result = evaluator.idle_fraction()
    assert set(result.keys()) == {"gpu_instance_a", "pim_instance_a", "gpu_instance_b"}


def test_evaluator_idle_fraction_propagates_from_telemetry(dummy_config, clock):
    """Evaluator.idle_fraction() 의 각 key 값 = idle_telemetry 의 해당 slot 산출값 bit-exact."""
    from puls_sched.evaluator import Evaluator
    tel = IdleTelemetry()
    tel.reset(0.0)
    tel.record_active("gpu_instance_a", 0.0, 3.0)
    tel.record_active("pim_instance_a", 0.0, 5.0)
    tel.record_active("gpu_instance_b", 0.0, 7.0)
    ev = Evaluator(config=dummy_config, clock=clock, idle_telemetry=tel)
    result = ev.idle_fraction()
    assert result["gpu_instance_a"] == tel.gpu_idle_fraction()
    assert result["pim_instance_a"] == tel.pim_idle_fraction()
    assert result["gpu_instance_b"] == tel.gpu_instance_b_idle_fraction()


def test_evaluator_idle_fraction_determinism(evaluator):
    """동일 시점 두 번 호출 → 동일 결과 (post-hoc snapshot 정합)."""
    a = evaluator.idle_fraction()
    b = evaluator.idle_fraction()
    assert a == b
