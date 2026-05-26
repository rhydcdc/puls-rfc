from dataclasses import dataclass, field


@dataclass
class MicroBatch:
    id: int
    layer_index: int = 0
    prefill_chunk: dict[int, list[int]] = field(default_factory=dict)
    decode_tokens: dict[int, int] = field(default_factory=dict)

    def request_ids(self) -> set[int]:
        return set(self.prefill_chunk.keys()) | set(self.decode_tokens.keys())

    def is_pure_prefill(self) -> bool:
        return len(self.decode_tokens) == 0

    def is_pure_decode(self) -> bool:
        return len(self.prefill_chunk) == 0
