from dataclasses import dataclass
from typing import Callable

from puls_sched.config import AdmissionConfig


@dataclass(frozen=True)
class KTotalResult:
    k_total: int
    over_budget: bool


def solve(
    t_proj: float,
    t_pim_fn: Callable[[int, int], float],
    n_decode: int,
    admission_cfg: AdmissionConfig,
) -> KTotalResult:
    dial = range(0, admission_cfg.k_total_max + 1, admission_cfg.k_total_step)
    feasible = [k for k in dial if t_pim_fn(k, n_decode) <= t_proj]
    if not feasible:
        return KTotalResult(k_total=0, over_budget=True)
    return KTotalResult(k_total=max(feasible), over_budget=False)
