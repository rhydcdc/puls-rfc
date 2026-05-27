from dataclasses import dataclass

from puls_sched.config import Config
from puls_sched.instance import Instance
from puls_sched.micro_batch import MicroBatch
from puls_sched.nvlink import NVLinkTransfer


@dataclass
class InstancePipeline:
    """Instance A → B → A_next 단일 layer cycle owner. ARCH §3.4 Pipeline Structure 정합.

    - Q3 — *단일 layer* cycle 구조 (A → handoff → B → handoff → A_next). L-loop 은 forward_pass.
    - Q6 — steady_state_cycle() runtime getter (evaluator 는 Impl-8).
    - Q7 — steady_state_cycle = max(A_cycle, B_cycle) ARCH literal. NVLink event 별도 push 안 함.
    """

    config: Config
    instance_a: Instance
    instance_b: Instance
    nvlink: NVLinkTransfer

    def __post_init__(self) -> None:
        if not self.instance_a.has_pim:
            raise ValueError("instance_a must have PIM (ARCH §3.4 Case A)")
        if self.instance_b.has_pim:
            raise ValueError("instance_b must not have PIM (ARCH §3.4 Case A)")

    def validate_handoff_shape(
        self, mb: MicroBatch, tensor_shape: tuple[int, ...]
    ) -> None:
        """A → B handoff tensor 의 fixed-shape 강제 (ARCH §5.2).

        - Decode-only: shape == (B, hidden) where B = len(mb.decode_tokens)
        - Pure-prefill (uniform chunk): shape == (B_prefill * chunk, hidden)
        - Mixed: shape == (B_decode + B_prefill * chunk, hidden)
        - Ragged prefill chunk lengths → raise (ARCH §5.2 uniform-chunk 위반)
        """
        if len(tensor_shape) != 2:
            raise AssertionError(
                f"handoff tensor must be 2D [tokens × hidden], got {tensor_shape}"
            )
        n_tokens, hidden = tensor_shape
        if hidden != self.config.model.hidden:
            raise AssertionError(
                f"handoff hidden dim mismatch: got {hidden}, expected {self.config.model.hidden}"
            )
        chunk_lens = {len(toks) for toks in mb.prefill_chunk.values()}
        if len(chunk_lens) > 1:
            raise AssertionError(
                f"ragged prefill chunk lengths {chunk_lens} — ARCH §5.2 uniform-chunk 위반"
            )
        chunk = next(iter(chunk_lens)) if chunk_lens else 0
        expected = len(mb.decode_tokens) + len(mb.prefill_chunk) * chunk
        if n_tokens != expected:
            raise AssertionError(
                f"handoff tokens mismatch: got {n_tokens}, expected {expected} "
                f"(decode={len(mb.decode_tokens)}, prefill_reqs={len(mb.prefill_chunk)}, chunk={chunk})"
            )

    def steady_state_cycle(self, a_cycle: float, b_cycle: float) -> float:
        """Steady-state pipeline cycle = max(A_cycle, B_cycle). ARCH §3.4 literal.

        Q7 — NVLink 은 async hidden, max(A, B) 산식에 포함 안 함.
        """
        if a_cycle < 0 or b_cycle < 0:
            raise ValueError(f"cycle values must be non-negative: a={a_cycle}, b={b_cycle}")
        return max(a_cycle, b_cycle)
