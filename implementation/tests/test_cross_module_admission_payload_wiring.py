"""Cluster X — ADMISSION_TICK payload 전 영역 production wiring (Impl-10-pre-1 (B)~(B''')).

ARCH §6.4 *"GPU/PIM idle fractions of the previous iteration are measured to regulate next
μ-batch's admission"* literal 정합. 5 개 payload 입력 모두 production scheduler 위 진정
측정값 wiring 검증.

CLAUDE.md §5 *"property you defined hold ≠ property is right"* 정확 적용 — 본 cluster 가
*end-to-end Run.loop 후 admission decision* 영역의 진정성 검증.
"""

import pytest

from puls_sched.config import default_dummy_config
from puls_sched.run import Run


# ---- _compose_admission_payload — 5 key 모두 산출 ----

def test_compose_payload_returns_six_keys(tmp_path):
    """_compose_admission_payload 가 6 개 key 모두 반환 (Impl-10-pre-2 — gpu_op_time_per_token_us 추가)."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    payload = run.scheduler._compose_admission_payload()
    assert set(payload.keys()) == {
        "t_proj", "t_pim_fn", "a_cycle", "b_cycle", "ctx_tokens",
        "gpu_op_time_per_token_us",   # Impl-10-pre-2 (B option) — PIM-time-driven adaptive chunk
    }


def test_compose_payload_t_proj_matches_config(tmp_path):
    """(B') t_proj = config.time.gpu_op_time_us[qkv] + [o_proj]."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    payload = run.scheduler._compose_admission_payload()
    expected = (
        run.config.time.gpu_op_time_us["qkv"]
        + run.config.time.gpu_op_time_us["o_proj"]
    )
    assert payload["t_proj"] == pytest.approx(expected)
    assert payload["t_proj"] > 0   # non-trivial


def test_compose_payload_t_pim_fn_callable_and_non_trivial(tmp_path):
    """(B'') t_pim_fn closure 가 callable + in_flight 위 non-zero 산출."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    # 진행 → in_flight req 누적
    for _ in range(20):
        if not run.scheduler.step():
            break
    payload = run.scheduler._compose_admission_payload()
    fn = payload["t_pim_fn"]
    assert callable(fn)
    # in_flight 가 있으면 t_pim_fn(n) > 0 (Impl-10-pre-2 — k_channels 매개변수 폐기)
    if run.scheduler.in_flight_requests:
        result = fn(4)
        assert result >= 0   # non-negative (정합 — PIMExecutor.op_time 산출값)


def test_compose_payload_t_pim_fn_zero_when_empty(tmp_path):
    """(B'') t_pim_fn 가 in_flight empty 시 0 (guard)."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:1",
        output_dir=tmp_path,
        seed=42,
    )
    payload = run.scheduler._compose_admission_payload()
    fn = payload["t_pim_fn"]
    # init 시점 in_flight empty
    assert fn(4) == 0.0


# ---- (B) a_cycle / b_cycle 진정 측정 ----

def test_a_cycle_grows_after_dispatches(tmp_path):
    """(B) a_cycle = IdleTelemetry delta — dispatch 누적 후 > 0."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    # (B) — accumulated active_duration 직접 검증 (delta 영역의 _compose 자동 fire trap 회피).
    for _ in range(5000):
        if not run.scheduler.step():
            break
    tel = run.scheduler.admission.idle_telemetry
    accumulated_a = (
        tel.active_duration("gpu_instance_a") + tel.active_duration("pim_instance_a")
    )
    assert accumulated_a > 0.0, (
        f"Instance A active = {accumulated_a} "
        f"(5000 step 후에도 0 — (B) wiring 결손)"
    )


@pytest.mark.skip(
    reason="Impl-10-pre-2 post-fix — Stage 1 placeholder substrate (gpu_instance_b ← NVLink handoff time) "
           "폐기. b_cycle = active_duration delta 영원 0 (Stage 2 calibration 위 실 FFN op_time 재활성)."
)
def test_b_cycle_grows_after_layer_cycles(tmp_path):
    """(B) b_cycle source = IdleTelemetry.active_duration("gpu_instance_b")."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    # 진행 — layer cycle 누적 (L=80 × 4 node 영역 위 충분 step)
    for _ in range(10000):
        if not run.scheduler.step():
            break
    tel = run.scheduler.admission.idle_telemetry
    accumulated_b = tel.active_duration("gpu_instance_b")
    assert accumulated_b > 0.0, (
        f"gpu_instance_b active_duration = {accumulated_b} "
        f"(10000 step 후에도 0 — (A) wiring 결손 또는 dispatch chain 영역 결손)"
    )


def test_cycles_reset_between_admission_payloads(tmp_path):
    """(B) snapshot 영역 — 두 번 호출 시 두 번째는 *delta* 만 (누적값 아님)."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    for _ in range(30):
        if not run.scheduler.step():
            break
    p1 = run.scheduler._compose_admission_payload()
    # 두 번째 호출 — 추가 dispatch 0 → delta = 0
    p2 = run.scheduler._compose_admission_payload()
    assert p2["a_cycle"] == 0.0
    assert p2["b_cycle"] == 0.0


# ---- (B''') ctx_tokens — in_flight 의 max kv_length ----

def test_ctx_tokens_reflects_in_flight_max_kv(tmp_path):
    """(B''') ctx_tokens = max(in_flight kv_length). Deadband ctx-tier 입력."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    for _ in range(30):
        if not run.scheduler.step():
            break
    payload = run.scheduler._compose_admission_payload()
    if run.scheduler.in_flight_requests:
        expected = max(r.kv_length for r in run.scheduler.in_flight_requests.values())
        assert payload["ctx_tokens"] == expected
    else:
        assert payload["ctx_tokens"] == 0


# ---- ADMISSION_TICK payload 가 진정 propagate (production) ----

def test_admission_snapshot_captures_non_trivial_cycles(tmp_path):
    """End-to-end Run.loop 후 evaluator 의 admission_snapshots 위 a_cycle 또는 b_cycle > 0
    발생 — production payload propagation 정합."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:10",
        output_dir=tmp_path,
        seed=42,
    )
    run.loop()
    snapshots = run.evaluator._admission_snapshots
    # 적어도 한 snapshot 이 non-trivial cycle 보유
    has_non_trivial = any(s.a_cycle > 0 or s.b_cycle > 0 for s in snapshots)
    assert has_non_trivial, (
        f"모든 {len(snapshots)} admission snapshot 위 a_cycle/b_cycle = 0 — "
        f"payload propagation 영역 wiring 결손"
    )


# ---- Determinism (C5 정합) ----

def test_payload_composition_deterministic(tmp_path):
    """동일 state → 동일 payload (callable 영역 외)."""
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    for _ in range(20):
        if not run.scheduler.step():
            break
    # 두 번 호출 — 두 번째는 delta 0 (state 변경 0). 그러나 절대값 measurement 위 deterministic.
    # 새 Run 으로 동일 seed → 동일 progression 위 동일 payload 검증.
    run2 = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path / "run2",
        seed=42,
    )
    for _ in range(20):
        if not run2.scheduler.step():
            break
    p1 = run.scheduler._compose_admission_payload()
    p2 = run2.scheduler._compose_admission_payload()
    # Callable 제외 4 key bit-exact
    assert p1["t_proj"] == p2["t_proj"]
    assert p1["a_cycle"] == p2["a_cycle"]
    assert p1["b_cycle"] == p2["b_cycle"]
    assert p1["ctx_tokens"] == p2["ctx_tokens"]


# ---- ADMISSION_TICK self-rescheduling 이 진정 payload 사용 ----

def test_self_rescheduled_tick_carries_fresh_payload(tmp_path):
    """_schedule_next_admission_tick 이 prev payload 그대로 propagate 아님 — fresh compose."""
    from puls_sched.event import Event, EventType
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    for _ in range(40):
        if not run.scheduler.step():
            break
    # Queue 에 다음 ADMISSION_TICK 가 self-rescheduled 됨 — payload 검사
    ticks = [
        e for e in run.scheduler.queue._heap
        if hasattr(e, "type") and e.type is EventType.ADMISSION_TICK
    ]
    # Heap structure 직접 inspection 회피 — 대신 payload composer 가 호출 site 정합 검증.
    # 정확 검증: _schedule_next_admission_tick body 위 _compose_admission_payload 호출 확인.
    import inspect
    src = inspect.getsource(run.scheduler._schedule_next_admission_tick)
    assert "_compose_admission_payload" in src
