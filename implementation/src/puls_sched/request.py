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

    def transition_to(self, new_state: RequestState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {new_state}")
        self.state = new_state
