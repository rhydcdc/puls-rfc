from dataclasses import dataclass, field


@dataclass
class MicroBatch:
    id: int
    layer_index: int = 0                                 # Impl-1 — *시작 layer* 의미
    prefill_chunk: dict[int, list[int]] = field(default_factory=dict)
    decode_tokens: dict[int, int] = field(default_factory=dict)
    # Impl-5 — admission decision 운반 + forward_pass runtime tracker (O4.1 해소).
    # Impl-10-pre-2 — k_total field 폐기 (sequence-parallel PIM 위 channel knob 무의미).
    kv_rows_total: int = 0                               # Σ kv_length over decode_reqs (admission spec 산출) — F5 활성화 path
    current_layer_index: int = 0                         # forward_pass 의 L-iteration 현재 위치 (시작 layer ≠ 현재 — Impl-9 통합 시점 §7 O5.1)
    # Impl-8 — F5 ablation 위 lock-step max-KV penalty 산식 입력 (max kv_length × num_decode_reqs).
    # F5 활성화 path 위에선 사용 안 함 (default 0). ARCH §5.7 F5 "max-KV straggler bubble within a batch" 정합.
    kv_rows_lockstep: int = 0
    # Impl-10-pre-2 (O9.1 + B option) — Total prefill chunk budget (across all prefill reqs in mb).
    # Stored at admission time → _recompose_mb 위 동일 budget 유지 (PIM-slack adaptive 정합).
    prefill_chunk_budget: int = 0
    # Impl-10 main — per-req prefill_processed at admission time (PREFILL_ATTN causal ctx 산출 입력).
    # Each entry = req.prefill_processed *직전* 산출값 (this mb 의 chunk 가 처리하기 전 상태).
    # PREFILL_ATTN FLOPs = 2 × chunk × hidden × (prefill_processed + chunk) — causal lower bound.
    prefill_processed: dict[int, int] = field(default_factory=dict)

    def request_ids(self) -> set[int]:
        return set(self.prefill_chunk.keys()) | set(self.decode_tokens.keys())

    def is_pure_prefill(self) -> bool:
        return len(self.decode_tokens) == 0

    def is_pure_decode(self) -> bool:
        return len(self.prefill_chunk) == 0
