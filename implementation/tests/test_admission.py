import pytest

from puls_sched.admission import Admission
from puls_sched.request import Request, RequestState


def _make_dec(req_id: int, kv_length: int = 10) -> Request:
    """DECODE 풀 후보 — 풀 모델에선 admission 이 KV admit 후 인플라이트로 들인 디코더."""
    r = Request(id=req_id, prompt_len=0, kv_length=kv_length, state=RequestState.PENDING)
    r.transition_to(RequestState.PREFILL)
    r.transition_to(RequestState.DECODE)
    return r


# Phase-2 풀 모델 (OPERATING_POINT §3) — admission(풀 보충) ∥ decode-set 선택 ∥ prefill 분배
# 3분리. layer1(admission cohort steering)은 폐기되고, decode-set 은 steer_decode_set 가
# *인플라이트 DECODE 풀*에서 순수 선택(KV admit·큐 조작 없음 — KV 는 풀 진입 시 처리).
# admission 의 풀 보충/μ-batch 구성/생애는 main_loop(SchedulerCore) 가 담당 — cross-module
# 검증은 test_cross_module_lifecycle·acceptance(드레인·KV누수0)·test_main_loop_completion.


# --- steer_decode_set (decode-set 구성, §3) ---

def test_steer_empty_returns_empty(admission):
    assert admission.steer_decode_set([]) == []


def test_steer_selects_all_under_target(admission):
    """타깃(123, 12.3M) 미달이면 후보 전부 선택."""
    cands = [_make_dec(i, kv_length=100) for i in range(3)]
    sel = admission.steer_decode_set(cands)
    assert len(sel) == 3


def test_steer_stops_at_count_target(admission):
    """decode 개수 타깃(123) 도달 시 정지 (kv 타깃 미달)."""
    cands = [_make_dec(i, kv_length=100) for i in range(130)]   # Σkv 13K ≪ 12.3M
    sel = admission.steer_decode_set(cands)
    assert len(sel) == admission.admission_cfg.decode_count_target   # 123


def test_steer_stops_at_kv_target(admission):
    """Σkv 타깃(12.3M) 도달 시 정지 (개수 타깃 미달). 쪼개기 불가 → 마지막이 살짝 넘김."""
    cands = [_make_dec(i, kv_length=10_000_000) for i in range(5)]   # 각 10M
    sel = admission.steer_decode_set(cands)
    assert len(sel) == 2                                              # 10M+10M=20M ≥ 12.3M
    assert sum(r.kv_length for r in sel) >= admission.admission_cfg.kv_operating_target_tokens


def test_steer_picks_closest_to_ideal(admission):
    """첫 step ideal=12.3M/123≈100K — kv_length 가 가장 가까운 디코더 먼저."""
    cands = [_make_dec(0, 10_000), _make_dec(1, 100_000), _make_dec(2, 500_000)]
    sel = admission.steer_decode_set(cands)
    assert sel[0].id == 1                                            # closest-to-ideal


def test_steer_age_cap_forces_aged(admission):
    """wait ≥ age_cap 인 off-size 디코더는 steering 무시하고 강제 선택 (공정성)."""
    aged = _make_dec(0, kv_length=2_000_000)                         # off-ideal
    aged.wait = admission.admission_cfg.age_cap
    near = _make_dec(1, kv_length=100_000)                           # ideal 근접
    sel = admission.steer_decode_set([aged, near])
    assert sel[0].id == 0                                            # aged 강제 먼저


def test_steer_wait_accounting(admission):
    """선택분 wait=0, 미선택분 wait+=1 (age-cap 누적, 풀 잔류)."""
    cands = [_make_dec(i, kv_length=10_000_000) for i in range(3)]   # 2개서 12.3M 초과 → 1개 미선택
    sel = admission.steer_decode_set(cands)
    selected_ids = {r.id for r in sel}
    for r in cands:
        assert r.wait == (0 if r.id in selected_ids else 1)


def test_steer_no_kv_admit(admission):
    """steer 는 *순수 선택* — KV admit/큐 조작 없음(KV 는 풀 진입 시, admission)."""
    used_before = admission.kv_accountant.used
    admission.steer_decode_set([_make_dec(i, kv_length=100) for i in range(3)])
    assert admission.kv_accountant.used == used_before
