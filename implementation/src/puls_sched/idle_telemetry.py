from dataclasses import dataclass, field


# 3 internal resource slot — ARCH §6.4 admission balance 의 'inter-AB · intra-A' 정합.
# `gpu_instance_a` · `pim_instance_a` = Instance A 의 두 자원 (intra-A balance).
# `gpu_instance_b` = Instance B 의 FFN 자원 (inter-AB balance, A vs B cycle).
_RESOURCE_SLOTS = ("gpu_instance_a", "pim_instance_a", "gpu_instance_b")

# Legacy 키 (Impl-3 영역 single-instance) → Impl-10-pre-1 의 3-slot 매핑.
# 기존 caller (evaluator.AdmissionSnapshot · dispatcher 의 단축 호출)
# 가 "GPU"/"PIM" 키 사용 — backward-compat 정합. (balance_intra_A 는 Phase-2 S1 삭제.)
_LEGACY_KEY_MAP = {
    "GPU": "gpu_instance_a",
    "PIM": "pim_instance_a",
}


@dataclass
class IdleTelemetry:
    """3-slot resource activity tracker. Impl-10-pre-1 O8.1 + O8.2 정합.

    Slot 영역 (ARCH §6.4):
    - `gpu_instance_a` / `pim_instance_a` — intra-A balance signal (Impl-3 영역 그대로)
    - `gpu_instance_b` — inter-AB balance 의 B-side substrate (Impl-10-pre-1 신설)

    Legacy 키 "GPU"/"PIM" 는 instance_a slot 으로 자동 매핑 (backward-compat).
    """

    _window_start: float = 0.0
    _window_end: float = 0.0
    _active_duration: dict[str, float] = field(
        default_factory=lambda: {slot: 0.0 for slot in _RESOURCE_SLOTS}
    )

    def reset(self, t_start: float) -> None:
        self._window_start = t_start
        self._window_end = t_start
        for slot in _RESOURCE_SLOTS:
            self._active_duration[slot] = 0.0

    def record_active(self, resource: str, t_start: float, t_end: float) -> None:
        if t_end < t_start:
            raise ValueError(f"record_active: t_end {t_end} < t_start {t_start}")
        slot = _LEGACY_KEY_MAP.get(resource, resource)
        if slot not in self._active_duration:
            raise ValueError(f"unknown resource: {resource}")
        self._active_duration[slot] += t_end - t_start
        if t_end > self._window_end:
            self._window_end = t_end

    def idle_fraction(self, resource: str) -> float:
        """Resource (3-slot key 또는 legacy "GPU"/"PIM") 의 idle fraction.

        Span = window_end - window_start. span ≤ 0 시 0.0 (div-by-zero guard).
        """
        slot = _LEGACY_KEY_MAP.get(resource, resource)
        if slot not in self._active_duration:
            raise ValueError(f"unknown resource: {resource}")
        span = self._window_end - self._window_start
        if span <= 0:
            return 0.0
        return max(0.0, 1.0 - self._active_duration[slot] / span)

    def active_duration(self, resource: str) -> float:
        """Resource 의 누적 활동 시간 (clock time 단위). Impl-10-pre-1 (B) — cycle 측정 substrate.

        SchedulerCore 의 _measure_cycles 가 ADMISSION_TICK 사이 delta 산출 위 호출.
        """
        slot = _LEGACY_KEY_MAP.get(resource, resource)
        if slot not in self._active_duration:
            raise ValueError(f"unknown resource: {resource}")
        return self._active_duration[slot]

    # ---- Backward-compat 영역 (Impl-3 caller — admission · evaluator) ----

    def gpu_idle_fraction(self) -> float:
        """Instance A GPU idle fraction (legacy alias for gpu_instance_a)."""
        return self.idle_fraction("gpu_instance_a")

    def pim_idle_fraction(self) -> float:
        """Instance A PIM idle fraction (legacy alias for pim_instance_a)."""
        return self.idle_fraction("pim_instance_a")

    def gpu_instance_b_idle_fraction(self) -> float:
        """Instance B GPU (FFN) idle fraction. Impl-10-pre-1 O8.1 신설."""
        return self.idle_fraction("gpu_instance_b")
