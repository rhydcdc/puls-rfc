import pytest

from puls_sched.config import default_dummy_config
from puls_sched.k_total import KTotalResult, solve


@pytest.fixture
def admission_cfg():
    return default_dummy_config().admission


EXPECTED_DIAL = (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048)


def test_dial_is_9_steps(admission_cfg):
    dial = list(range(0, admission_cfg.k_total_max + 1, admission_cfg.k_total_step))
    assert len(dial) == 9
    assert tuple(dial) == EXPECTED_DIAL


def test_solve_picks_max_feasible(admission_cfg):
    r = solve(t_proj=1000.0, t_pim_fn=lambda k, n: 0.0, n_decode=16, admission_cfg=admission_cfg)
    assert r == KTotalResult(k_total=2048, over_budget=False)


def test_solve_picks_max_when_only_partial_feasible(admission_cfg):
    # t_pim(k, _) = k → feasible iff k ≤ 1000 → max feasible = 768
    r = solve(t_proj=1000.0, t_pim_fn=lambda k, n: float(k), n_decode=16,
              admission_cfg=admission_cfg)
    assert r.k_total == 768
    assert r.over_budget is False


def test_solve_over_budget_when_no_feasible(admission_cfg):
    r = solve(t_proj=1.0, t_pim_fn=lambda k, n: 1e9, n_decode=16, admission_cfg=admission_cfg)
    assert r == KTotalResult(k_total=0, over_budget=True)


def test_solve_at_k_max_boundary(admission_cfg):
    # t_pim(2048) == t_proj → ≤ is True → 2048 feasible
    r = solve(t_proj=2048.0, t_pim_fn=lambda k, n: float(k), n_decode=16,
              admission_cfg=admission_cfg)
    assert r.k_total == 2048
    assert r.over_budget is False


def test_solve_determinism_1000_calls(admission_cfg):
    """Acceptance: kTotalDecider 결정론 — 동일 입력 1000회 bit-exact."""
    fn = lambda k, n: float(k) * 0.5
    results = [solve(t_proj=500.0, t_pim_fn=fn, n_decode=32, admission_cfg=admission_cfg)
               for _ in range(1000)]
    assert len(set(results)) == 1


def test_solve_stack_granularity(admission_cfg):
    """모든 dial 원소가 step (256) 의 배수."""
    dial = range(0, admission_cfg.k_total_max + 1, admission_cfg.k_total_step)
    for k in dial:
        assert k % admission_cfg.k_total_step == 0


@pytest.mark.parametrize("target", EXPECTED_DIAL)
def test_solve_dial_full_enumeration(admission_cfg, target):
    """각 dial 원소가 *유일하게* 선택 가능한 case 위에서 solve 가 정확히 그 원소 반환."""
    # feasible only at k == target → t_pim(target) = 0, else = 1e9
    fn = lambda k, n, t=target: 0.0 if k == t else 1e9
    r = solve(t_proj=1.0, t_pim_fn=fn, n_decode=16, admission_cfg=admission_cfg)
    assert r.k_total == target
    assert r.over_budget is False


def test_over_budget_flag_with_k_max_only_infeasible(admission_cfg):
    """t_pim infeasible at k_max but feasible at smaller k → not over_budget."""
    fn = lambda k, n: 1e9 if k == 2048 else 0.0
    r = solve(t_proj=1.0, t_pim_fn=fn, n_decode=16, admission_cfg=admission_cfg)
    assert r.k_total == 1792
    assert r.over_budget is False


def test_solve_n_decode_passed_through(admission_cfg):
    """t_pim_fn 가 n_decode 를 received → solve 가 caller 의 n_decode 전달."""
    captured = []
    def fn(k, n):
        captured.append(n)
        return 0.0
    solve(t_proj=1.0, t_pim_fn=fn, n_decode=42, admission_cfg=admission_cfg)
    assert all(n == 42 for n in captured)
    assert len(captured) == 9  # one call per dial step
