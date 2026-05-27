from dataclasses import dataclass, field


@dataclass
class MicroBatch:
    id: int
    layer_index: int = 0                                 # Impl-1 — *시작 layer* 의미
    prefill_chunk: dict[int, list[int]] = field(default_factory=dict)
    decode_tokens: dict[int, int] = field(default_factory=dict)
    # Impl-5 — admission decision 운반 + forward_pass runtime tracker (O4.1 해소)
    k_total: int = 0                                     # SP-PIM aggregate channel count (admission 산출)
    kv_rows_total: int = 0                               # Σ kv_length over decode_reqs (admission spec 산출)
    current_layer_index: int = 0                         # forward_pass 의 L-iteration 현재 위치 (시작 layer ≠ 현재 — Impl-9 통합 시점 §7 O5.1)

    def request_ids(self) -> set[int]:
        return set(self.prefill_chunk.keys()) | set(self.decode_tokens.keys())

    def is_pure_prefill(self) -> bool:
        return len(self.decode_tokens) == 0

    def is_pure_decode(self) -> bool:
        return len(self.prefill_chunk) == 0
