"""Cross-module integration: k_total.solve + PIMExecutor.op_time real injection.

PLAN.md §0.5 reminder — 현재 dummy fn 대신 real PIMExecutor 의 op_time 을
`t_pim_fn` 으로 closure bind 한 후 `k_total.solve` 호출 → 정확한 k_total 선택.
Impl-3 의 `t_pim_fn: Callable[[int, int], float]` signature 와 Impl-4 의
PIMExecutor.op_time(k, rows) 의 *경계* 검증 — n_decode 무시가 ARCH §3.1 정합.
"""

import pytest

from puls_sched.k_total import solve as k_total_solve


def _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total: int):
    """real PIMExecutor.op_time 을 k_total.solve 의 t_pim_fn signature `(k, n) → t_pim` 으로 adapt.
    n_decode 는 무시 (ARCH §3.1 invariance 정합)."""
    return lambda k, n: pim_executor.op_time(k_channels=k, kv_rows_total=kv_rows_total)


def test_k_total_solve_with_real_pim_executor_picks_max_feasible(
    pim_executor, admission_config
):
    """large t_proj 위 모든 k feasible → max k = 2048 반환."""
    t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=1000)
    result = k_total_solve(
        t_proj=1e9,  # 큰 budget
        t_pim_fn=t_pim_fn,
        n_decode=8,
        admission_cfg=admission_config,
    )
    assert result.k_total == 2048
    assert result.over_budget is False


def test_k_total_solve_only_k_zero_feasible_with_real_pim_executor(
    pim_executor, admission_config
):
    """small t_proj 위 k=0 만 feasible (op_time(0, _) = 0 정합, ARCH §5.1)."""
    t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=10_000_000)
    result = k_total_solve(
        t_proj=0.001,  # 거의 0 budget — k≥256 모두 infeasible
        t_pim_fn=t_pim_fn,
        n_decode=8,
        admission_cfg=admission_config,
    )
    # k=0 → op_time=0.0 ≤ 0.001 → feasible. 다른 모든 k 는 infeasible.
    # max feasible = 0, over_budget = False (k=0 은 항상 feasible)
    assert result.k_total == 0
    assert result.over_budget is False


def test_k_total_solve_monotonic_in_kv_rows_via_executor(pim_executor, admission_config):
    """rows 증가 → 같은 t_proj 위 max feasible k 단조 비증가 (KV row 늘면 더 많은 channel 필요)."""
    t_proj = 100.0  # 중간 budget
    rows_seq = [100, 1000, 10000, 100000, 1_000_000]
    k_results = []
    for rows in rows_seq:
        t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=rows)
        result = k_total_solve(t_proj, t_pim_fn, n_decode=8, admission_cfg=admission_config)
        k_results.append(result.k_total)
    # 단조 비증가 검증 (rows 증가 → k 비증가)
    # Note: solve 의 contract 는 max k s.t. t_pim ≤ t_proj. 더 많은 rows → 같은 k 위 t_pim 큼
    # → 같은 t_proj 위 feasible k 영역 축소. 단 k=0 은 항상 feasible (t_pim=0)
    for i in range(len(k_results) - 1):
        # 비증가 보장은 정확 monotonic 아닐 수 있음 — dial step 으로 동일 가능
        # 약한 조건: rows 단조 증가 시 k 단조 비증가
        assert k_results[i] >= k_results[i + 1], (
            f"k regression at rows[{i}]={rows_seq[i]}→k={k_results[i]}, "
            f"rows[{i+1}]={rows_seq[i+1]}→k={k_results[i+1]}"
        )


def test_k_total_dial_stack_granularity_preserved_via_executor(pim_executor, admission_config):
    """real PIMExecutor injection 위에서도 k_total ∈ {0, 256, ..., 2048} dial 정합."""
    t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=50000)
    for t_proj in [0.0, 1.0, 10.0, 100.0, 1000.0, 1e9]:
        result = k_total_solve(t_proj, t_pim_fn, n_decode=8, admission_cfg=admission_config)
        assert result.k_total % admission_config.k_total_step == 0
        assert 0 <= result.k_total <= admission_config.k_total_max


def test_k_total_solve_deterministic_with_pim_executor(pim_executor, admission_config):
    """동일 입력 1000 회 → bit-exact (PIMExecutor stateless 정합 위 cross-module determinism)."""
    t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=10000)
    expected = k_total_solve(100.0, t_pim_fn, n_decode=8, admission_cfg=admission_config)
    for _ in range(1000):
        result = k_total_solve(100.0, t_pim_fn, n_decode=8, admission_cfg=admission_config)
        assert result.k_total == expected.k_total
        assert result.over_budget == expected.over_budget


@pytest.mark.parametrize("n_decode", [1, 8, 64, 256, 1024])
def test_k_total_solve_invariant_under_n_decode_sweep_real_pim(
    pim_executor, admission_config, n_decode
):
    """Cross-module batch invariance (Q12) — t_pim_fn 의 n 무시가 ARCH §3.1 정합.

    n_decode 가 다르더라도 t_pim_fn 이 그 정보를 PIMExecutor 까지 전달하지 않으면
    k_total 결과는 동일. caller 측 (k_total.solve) 의 n_decode 가 PIMExecutor 까지
    *흘러가지 않음* 을 영구 기록.
    """
    t_pim_fn = _bind_pim_to_t_pim_fn(pim_executor, kv_rows_total=10000)
    # baseline = n_decode=8
    baseline = k_total_solve(100.0, t_pim_fn, n_decode=8, admission_cfg=admission_config)
    # sweep n_decode (with same t_pim_fn that ignores n)
    swept = k_total_solve(100.0, t_pim_fn, n_decode=n_decode, admission_cfg=admission_config)
    assert swept.k_total == baseline.k_total
    assert swept.over_budget == baseline.over_budget
