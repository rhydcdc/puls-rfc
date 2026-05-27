from collections import deque

from puls_sched.request import Request


class RequestQueue:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._q: deque[Request] = deque()

    def push(self, req: Request) -> bool:
        if len(self._q) >= self._capacity:
            return False
        self._q.append(req)
        return True

    def pop_oldest(self) -> Request | None:
        return self._q.popleft() if self._q else None

    def peek_oldest(self) -> Request | None:
        return self._q[0] if self._q else None

    def __len__(self) -> int:
        return len(self._q)
