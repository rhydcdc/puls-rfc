from dataclasses import dataclass, field
from enum import Enum, auto


class RequestState(Enum):
    PENDING = auto()
    PREFILL = auto()
    DECODE = auto()
    COMPLETED = auto()


_VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.PENDING: {RequestState.PREFILL},
    RequestState.PREFILL: {RequestState.DECODE, RequestState.COMPLETED},
    RequestState.DECODE: {RequestState.COMPLETED},
    RequestState.COMPLETED: set(),
}


@dataclass
class Request:
    id: int
    prompt_tokens: list[int]
    decoded_tokens: list[int] = field(default_factory=list)
    kv_length: int = 0
    arrival_time: float = 0.0
    state: RequestState = RequestState.PENDING
    # ---- Impl-6 lifecycle fields (Q6 · Q10 — Request = lifecycle owner) ----
    max_tokens: int = 0
    decoded_count: int = 0
    completion_time: float | None = None
    # ---- Impl-10-pre-2 (O9.1) — prefill chunking position tracker ----
    prefill_processed: int = 0
    # ---- Phase-2 former-v2 — age-cap 공정성 추적. former 가 한 batch 구성에서 이 요청을
    # 선택하지 않으면 +1, 선택(admit)되면 의미 종료. wait ≥ age_cap 이면 steering 무시하고
    # 강제 포함(OPERATING_POINT §3) → starvation 0. ----
    wait: int = 0

    def transition_to(self, new_state: RequestState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {new_state}")
        self.state = new_state
