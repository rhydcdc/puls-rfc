from dataclasses import dataclass


@dataclass
class IdleTelemetry:
    _window_start: float = 0.0
    _window_end: float = 0.0
    _gpu_active_duration: float = 0.0
    _pim_active_duration: float = 0.0

    def reset(self, t_start: float) -> None:
        self._window_start = t_start
        self._window_end = t_start
        self._gpu_active_duration = 0.0
        self._pim_active_duration = 0.0

    def record_active(self, resource: str, t_start: float, t_end: float) -> None:
        if t_end < t_start:
            raise ValueError(f"record_active: t_end {t_end} < t_start {t_start}")
        dt = t_end - t_start
        if resource == "GPU":
            self._gpu_active_duration += dt
        elif resource == "PIM":
            self._pim_active_duration += dt
        else:
            raise ValueError(f"unknown resource: {resource}")
        if t_end > self._window_end:
            self._window_end = t_end

    def gpu_idle_fraction(self) -> float:
        span = self._window_end - self._window_start
        if span <= 0:
            return 0.0
        return max(0.0, 1.0 - self._gpu_active_duration / span)

    def pim_idle_fraction(self) -> float:
        span = self._window_end - self._window_start
        if span <= 0:
            return 0.0
        return max(0.0, 1.0 - self._pim_active_duration / span)
