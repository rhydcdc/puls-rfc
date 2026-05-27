"""Request lifecycle 종료 검출 + KV slot 회수 (Impl-6).

Q5 — main_loop 의 token decode signal consumer 가 호출.
Q6 (c) — hybrid: max_tokens path (default) + EOS marker path (외부 eos_seen 명시).
Q7 — finalize 직후 즉시 KV release (ARCH §3.3 "permanently resident" 의 종료 시점 회수).
Q9 — Completion 은 KV release 만. dispatcher.unregister 호출 0 (window eviction 시점, Impl-9).
"""

from dataclasses import dataclass

from puls_sched.clock import Clock
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request, RequestState


@dataclass
class Completion:
    """Request lifecycle 종료 검출 + KV slot 회수.

    check 는 pure (idempotent). finalize 는 non-idempotent (중복 시 raise).
    """

    clock: Clock
    kv_accountant: KVAccountant

    def check(self, req: Request, eos_seen: bool = False) -> bool:
        """종료 시점 도달 검사. Pure.

        Idempotent: COMPLETED 상태 위 True 반환 (caller 가 안전 polling 가능).
        """
        if req.state == RequestState.COMPLETED:
            return True
        if eos_seen:
            return True
        if req.decoded_count >= req.max_tokens:
            return True
        return False

    def finalize(self, req: Request) -> None:
        """종료 처리 — KV release → completion_time 기록 → state 전이.

        Q7 — KV release 가 *첫 단계* (atomic semantic: release 실패 시 state 보존).

        Raises:
            ValueError: req.state 가 COMPLETED (중복 finalize) 또는 PENDING (admission 전).
            ValueError: kv_accountant.release 실패 (미admit req).
        """
        if req.state == RequestState.COMPLETED:
            raise ValueError(f"req {req.id}: double finalize (already COMPLETED)")
        if req.state == RequestState.PENDING:
            raise ValueError(f"req {req.id}: finalize before admission (state=PENDING)")
        self.kv_accountant.release(req)
        req.completion_time = self.clock.now
        req.transition_to(RequestState.COMPLETED)
