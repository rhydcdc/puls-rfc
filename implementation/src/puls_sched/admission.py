from dataclasses import dataclass
from typing import Callable

from puls_sched.config import AdmissionConfig
from puls_sched.deadband import in_band, lookup_width
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.k_total import solve as k_total_solve
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


@dataclass(frozen=True)
class MicroBatchSpec:
    prefill_chunk_tokens: int
    decode_requests: tuple[Request, ...]
    n: int
    k_total: int
    over_budget: bool
    kv_rows_total: int                                   # Impl-5 — Σ kv_length over decode_requests (signal flow to dispatcher)


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
        gpu_idle = self.idle_telemetry.gpu_idle_fraction()
        pim_idle = self.idle_telemetry.pim_idle_fraction()
        theta_high = self.admission_cfg.idle_theta_high
        if gpu_idle > theta_high and pim_idle <= theta_high:
            return prefill_chunk_tokens, decode_request_count + 1
        if pim_idle > theta_high and gpu_idle <= theta_high:
            return prefill_chunk_tokens + self.admission_cfg.n_sat, decode_request_count
        return prefill_chunk_tokens, decode_request_count

    def layer1(
        self,
        t_proj: float,
        t_pim_fn: Callable[[int, int], float],
        a_cycle: float,
        b_cycle: float,
        ctx_tokens: int,
    ) -> MicroBatchSpec | None:
        decode_reqs: list[Request] = []
        while True:
            req = self.request_queue.peek_oldest()
            if req is None or not self.kv_accountant.can_admit(req):
                break
            self.request_queue.pop_oldest()
            self.kv_accountant.admit(req)
            decode_reqs.append(req)

        if not decode_reqs:
            return None

        prefill_chunk_tokens = 0
        prefill_chunk_tokens = self.balance_inter_AB(
            prefill_chunk_tokens, a_cycle, b_cycle, ctx_tokens,
        )
        prefill_chunk_tokens, decode_count = self.balance_intra_A(
            prefill_chunk_tokens, len(decode_reqs),
        )

        n = self.mfu_floor(decode_count)
        k_result = k_total_solve(t_proj, t_pim_fn, n, self.admission_cfg)
        kv_rows_total = sum(r.kv_length for r in decode_reqs)

        return MicroBatchSpec(
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_requests=tuple(decode_reqs),
            n=n,
            k_total=k_result.k_total,
            over_budget=k_result.over_budget,
            kv_rows_total=kv_rows_total,
        )
