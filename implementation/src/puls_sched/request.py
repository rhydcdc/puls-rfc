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
    # Phase-2 S4(a) — 토큰 *내용* 은 스케줄러가 안 씀(len 만). 실 1M-ctx 트레이스서
    # prompt_tokens=[0]*num_prefill materialize 가 OOM 유발 → 길이(int)만 보존.
    prompt_len: int
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
    # ---- Phase-2 former-v2 — age-cap 공정성 추적 (OPERATING_POINT §3) ----
    # wait: decode admission(request_queue) 대기. former 가 이 요청을 batch 에 안 넣으면 +1,
    #       admit 되면 종료. wait ≥ age_cap → steering 무시 강제 포함.
    # prefill_wait: prefill 토큰 분배 대기(in-flight). 한 사이클서 토큰 0개 받으면 +1, 받으면
    #       0 리셋. prefill_wait ≥ age_cap → 강제 토큰 배정 (prefill starvation 0).
    wait: int = 0
    prefill_wait: int = 0

    def transition_to(self, new_state: RequestState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {new_state}")
        self.state = new_state
