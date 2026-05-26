class Clock:
    def __init__(self, start: float = 0.0):
        self._t: float = start

    @property
    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> None:
        if t < self._t:
            raise ValueError(f"non-monotonic advance: {self._t} -> {t}")
        self._t = t
