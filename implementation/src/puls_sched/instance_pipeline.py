from dataclasses import dataclass
from typing import Optional

from puls_sched.clock import Clock
from puls_sched.config import Config
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.instance import Instance
from puls_sched.micro_batch import MicroBatch
from puls_sched.nvlink import NVLinkTransfer


@dataclass
class InstancePipeline:
    """Instance A → B → A_next 단일 layer cycle owner. ARCH §3.4 Pipeline Structure 정합.

    - Q3 — *단일 layer* cycle 구조 (A → handoff → B → handoff → A_next). L-loop 은 forward_pass.
    - Q6 — steady_state_cycle() runtime getter (evaluator 는 Impl-8).
    - Q7 — steady_state_cycle = max(A_cycle, B_cycle) ARCH literal. NVLink event 별도 push 안 함.

    Impl-10-pre-1 O5.1 — `dispatch()` 신설. ARCH §3.4 *forward pass = L × cycle* literal 정합으로
    forward_pass.run 의 매 layer 마다 호출 (Q6 (a)). Substrate 영역 — fixed-shape 검증 + NVLink handoff
    time 측정 + Instance B GPU activity recording (gpu_instance_b — ARCH §6.4 inter-AB balance 의 B-side).

    `clock` · `idle_telemetry` 는 dispatch() 활성화 위 의존. backward-compat 영역에서 Optional —
    기존 caller (steady_state_cycle / validate_handoff_shape 전용) 영향 0.
    """

    config: Config
    instance_a: Instance
    instance_b: Instance
    nvlink: NVLinkTransfer
    clock: Optional[Clock] = None                       # Impl-10-pre-1 O5.1 — dispatch() 위 활성
    idle_telemetry: Optional[IdleTelemetry] = None      # Impl-10-pre-1 O5.1 + O8.1 — gpu_instance_b recording

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

    def dispatch(self, mb: MicroBatch) -> None:
        """단일 layer 의 A → handoff → B → handoff → A_next chain wiring (substrate).

        ARCH §3.4 (Pipeline Structure) + §5.2 (Fixed-shape Handoff) + §5.6 (Double-Buffering)
        정합. Caller (ForwardPass.run) 가 매 layer 마다 호출 — Q6 (a) 결정.

        Effects:
        1. A → B handoff tensor 의 fixed-shape 검증 (ARCH §5.2)
        2. NVLink handoff time 측정 (async hidden under max(A,B), ARCH §3.4 정합)
        3. Instance B GPU activity recording (gpu_instance_b — ARCH §6.4 inter-AB balance 의 B-side substrate)

        Stage 1 영역 — B-FFN duration = NVLink handoff time 의 placeholder substrate (PLAN §0.5
        정합). Stage 2 calibration 영역에서 실 FFN op_time (Llama-3 70B FP8 spec) 으로 교체.

        `clock` · `idle_telemetry` 가 None 이면 record 단계 skip (backward-compat).
        """
        n_tokens = len(mb.decode_tokens) + sum(
            len(t) for t in mb.prefill_chunk.values()
        )
        # 1. Fixed-shape handoff validation (ARCH §5.2)
        self.validate_handoff_shape(mb, (n_tokens, self.config.model.hidden))
        # 2. NVLink handoff time (async hidden, dispatched event 아님)
        handoff_time = self.nvlink.time((n_tokens, self.config.model.hidden))
        # 3. Instance B GPU activity recording (gpu_instance_b substrate)
        if self.clock is not None and self.idle_telemetry is not None:
            t_start = self.clock.now
            self.idle_telemetry.record_active(
                "gpu_instance_b", t_start, t_start + handoff_time,
            )
