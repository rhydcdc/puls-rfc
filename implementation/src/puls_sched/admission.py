from dataclasses import dataclass
from typing import Callable

from puls_sched.config import AdmissionConfig
from puls_sched.deadband import in_band, lookup_width
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


@dataclass(frozen=True)
class MicroBatchSpec:
    prefill_chunk_tokens: int
    decode_requests: tuple[Request, ...]
    n: int
    kv_rows_total: int                                   # Impl-5 — Σ kv_length over decode_requests (signal flow to dispatcher, F5 활성화 path)
    kv_rows_lockstep: int                                # Impl-8 — max(kv_length) × num_decode_reqs (F5 ablation 위 lock-step penalty 산식)


@dataclass
class Admission:
    admission_cfg: AdmissionConfig
    request_queue: RequestQueue
    kv_accountant: KVAccountant
    idle_telemetry: IdleTelemetry

    def mfu_floor(self, n: int) -> int:
        return max(n, self.admission_cfg.n_sat)

    def balance_inter_AB(
        self,
        prefill_chunk_tokens: int,
        a_cycle: float,
        b_cycle: float,
        ctx_tokens: int,
    ) -> int:
        width = lookup_width(self.admission_cfg, ctx_tokens)
        diff = a_cycle - b_cycle
        if in_band(diff, width):
            return prefill_chunk_tokens
        if diff < 0:
            return prefill_chunk_tokens + self.admission_cfg.n_sat
        return prefill_chunk_tokens

    def balance_intra_A(
        self,
        prefill_chunk_tokens: int,
        decode_request_count: int,
    ) -> tuple[int, int]:
        """Impl-10-pre-2 — ARCH §6.4 invert: idle 자원에 일 더 추가 (사용자 logic 정합).

        - GPU idle > θ_high, PIM busy → admit prefill chunk → GPU 윈도우 채움
          (idle GPU 를 PREFILL_ATTN 으로 활용, PIM decode-attn 과 동시)
        - PIM idle > θ_high, GPU busy → admit additional decode → PIM 윈도우 채움
          (idle PIM 을 decode-attn 으로 활용, GPU projection 과 동시)
        """
        gpu_idle = self.idle_telemetry.gpu_idle_fraction()
        pim_idle = self.idle_telemetry.pim_idle_fraction()
        theta_high = self.admission_cfg.idle_theta_high
        if gpu_idle > theta_high and pim_idle <= theta_high:
            return prefill_chunk_tokens + self.admission_cfg.n_sat, decode_request_count
        if pim_idle > theta_high and gpu_idle <= theta_high:
            return prefill_chunk_tokens, decode_request_count + 1
        return prefill_chunk_tokens, decode_request_count

    def balance_pim_slack(
        self,
        prefill_chunk_tokens: int,
        t_pim_fn: Callable[[int], float],
        n_decode: int,
        gpu_op_time_per_token_us: float,
        t_gpu_base: float,
    ) -> int:
        """Impl-10-pre-2 (B option) — PIM-time-driven adaptive *total* chunk budget.

        ARCH §3.5.2 *Computed Wait* + §6.3 *PIM completion time computed at dispatch* + §6.1
        *attention split (PREFILL_ATTN ‖ DECODE_ATTN concurrent)* + §3.5.3 *PIM-GPU TSV
        bandwidth contention margin* + §5.2 *uniform-chunk* literal 정합.

        사용자 의도 — *GPU 전체 cycle (QKV + PREFILL_ATTN + O_PROJ) ≈ t_pim × margin*:
        sequence-parallel PIM 위 임의 시점 한 mb 가 모든 채널 점유 (k 영원 k_max), 그래서
        balance 는 GPU 쪽 prefill chunk 조절 위 cycle 맞춤.

            chunk_total = max(0, t_pim × margin − t_gpu_base) / per_token

        Distribution 은 main_loop._populate_mb_phases 가 Option A (TOTAL ÷ N prefill reqs) 로 처리
        (ARCH §5.2 uniform-chunk).

        Edge case: t_pim × margin ≤ t_gpu_base → chunk_optimal = 0 → base (prefill_chunk_default) 보존.
        이는 projection 만으로 이미 PIM 보다 GPU 가 길어 PIM idle 이 구조적 영역.
        """
        if n_decode <= 0 or gpu_op_time_per_token_us <= 0:
            return prefill_chunk_tokens
        t_pim_predicted = t_pim_fn(n_decode)
        margin = self.admission_cfg.pim_slack_safety_margin
        gpu_slack_us = max(0.0, t_pim_predicted * margin - t_gpu_base)
        chunk_optimal_total = int(gpu_slack_us / gpu_op_time_per_token_us)
        return max(prefill_chunk_tokens, chunk_optimal_total)

    def layer1(
        self,
        t_proj: float,
        t_pim_fn: Callable[[int], float],
        a_cycle: float,
        b_cycle: float,
        ctx_tokens: int,
        gpu_op_time_per_token_us: float = 0.0,
    ) -> MicroBatchSpec | None:
        decode_reqs: list[Request] = []
        # Impl-9 — Head-of-line skip (production scheduler 정합, vLLM/Sarathi 영역).
        # FIFO break 대신 queue 전체 walk + fit 가능한 것만 admit. 큰 req 는 *상대 순서 유지*
        # 위 re-push → 다음 tick 위 capacity 회수 후 재시도. ARCH §3.3 'permanently resident' +
        # §6.4 admission policy 의 *implementation choice* 영역.
        candidates: list[Request] = []
        while True:
            req = self.request_queue.pop_oldest()
            if req is None:
                break
            candidates.append(req)
        # Phase-1 fix — 실제 배치 크기 = min(seq 상한, KV 캐파 허용분).
        # seq 상한 도달 시 KV 여유와 무관하게 나머지는 다음 tick 으로 defer →
        # 한 mb 독점 방지, 동시 다중 mb 형성 (REPORT_baseline §7b).
        max_batch = self.admission_cfg.max_batch_size
        for req in candidates:
            if len(decode_reqs) < max_batch and self.kv_accountant.can_admit(req):
                self.kv_accountant.admit(req)
                decode_reqs.append(req)
            else:
                # Defer — queue 의 그 자리 (상대 순서 유지) 위 re-push
                # (seq 상한 도달 또는 KV 부족 둘 중 하나)
                if not self.request_queue.push(req):
                    raise RuntimeError(
                        f"admission re-push failed for req {req.id} "
                        f"(RequestQueue capacity violation — defensive raise)"
                    )

        if not decode_reqs:
            return None

        # Impl-10-pre-2 — Hybrid Chunk Size Policy: base = prefill_chunk_default, balance 가 adjustment
        prefill_chunk_tokens = self.admission_cfg.prefill_chunk_default
        prefill_chunk_tokens = self.balance_inter_AB(
            prefill_chunk_tokens, a_cycle, b_cycle, ctx_tokens,
        )
        prefill_chunk_tokens, decode_count = self.balance_intra_A(
            prefill_chunk_tokens, len(decode_reqs),
        )

        n = self.mfu_floor(decode_count)
        # Impl-10-pre-2 — PIM-time-driven adaptive *total* chunk budget.
        # t_gpu_base = t_proj (= t_qkv + t_oproj) — GPU non-attention 영역 시간.
        # chunk_total = max(0, t_pim × margin − t_proj) / per_token
        prefill_chunk_tokens = self.balance_pim_slack(
            prefill_chunk_tokens, t_pim_fn, n,
            gpu_op_time_per_token_us, t_gpu_base=t_proj,
        )
        kv_rows_total = sum(r.kv_length for r in decode_reqs)
        # Impl-8 — F5 ablation 위 lock-step penalty 입력. decode_reqs 비어 있으면 0.
        kv_rows_lockstep = max((r.kv_length for r in decode_reqs), default=0) * len(decode_reqs)

        return MicroBatchSpec(
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_requests=tuple(decode_reqs),
            n=n,
            kv_rows_total=kv_rows_total,
            kv_rows_lockstep=kv_rows_lockstep,
        )
