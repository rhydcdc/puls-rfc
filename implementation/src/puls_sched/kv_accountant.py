from puls_sched.request import Request


class KVAccountant:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._used = 0
        self._admitted: dict[int, int] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._capacity - self._used

    def can_admit(self, req: Request) -> bool:
        return req.kv_length <= self.remaining

    def admit(self, req: Request) -> None:
        if req.id in self._admitted:
            raise ValueError(f"KV double-admit: req {req.id}")
        if req.kv_length > self.remaining:
            raise ValueError(
                f"KV overflow: req {req.id} demand={req.kv_length}, remaining={self.remaining}"
            )
        self._admitted[req.id] = req.kv_length
        self._used += req.kv_length

    def release(self, req: Request) -> None:
        if req.id not in self._admitted:
            raise ValueError(f"KV underflow release: req {req.id} not admitted")
        self._used -= self._admitted.pop(req.id)
