import pytest

from puls_sched.admission import Admission, MicroBatchSpec
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length)


# Phase-2 S1 — balance_intra_A(유휴율 기반 prefill 증량) 삭제됨.
# Phase-2 S2 — balance_inter_AB·balance_pim_slack 도 삭제(§2.5): 밸런스가 정적 동작점(KV 합
# 25M + prefill 512, §0.8)으로 확정되어 동적 cycle 측정·balance 기계장치가 통째로 moot.
# Phase-2 S2 — mfu_floor + MicroBatchSpec.n 도 삭제(§2.5): 동작점 N_dec≈250≫n_sat=16 이라
# floor 사문 + n 은 len(decode_requests) 와 중복. 관련 테스트(mfu_floor_*, balance_inter_*,
# multi_iteration_converges, out_of_band_returns, mfu_floor_applied) 제거.


# --- layer1 (동작점 former) ---

def test_layer1_empty_queue_returns_none(admission):
    assert admission.layer1() is None


def test_layer1_admits_decode_within_kv_capacity(admission, request_queue):
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=100))
    spec = admission.layer1()
    assert spec is not None
    assert len(spec.decode_requests) == 3


def test_layer1_stops_at_kv_capacity(request_queue, admission_config, idle_telemetry):
    # Custom KVAccountant with tight aggregate capacity → can_admit 가 먼저 막음.
    from puls_sched.kv_accountant import KVAccountant
    small_kv = KVAccountant(capacity=1000)
    adm = Admission(
        admission_cfg=admission_config,
        request_queue=request_queue,
        kv_accountant=small_kv,
        idle_telemetry=idle_telemetry,
    )
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=500))
    spec = adm.layer1()
    assert spec is not None
    assert len(spec.decode_requests) == 2
    assert len(request_queue) == 1


def test_layer1_stops_at_operating_target(admission, request_queue):
    """former-v2 — Σkv 가 kv_operating_target(12.3M) 도달 시 정지, 나머지 defer."""
    for i in range(5):
        request_queue.push(_make_req(i, kv_length=10_000_000))   # 각 10M
    spec = admission.layer1()
    assert spec is not None
    # 10M+10M = 20M ≥ 12.3M → 2개 admit 후 정지 (마지막이 목표 넘김, 쪼개기 불가)
    assert len(spec.decode_requests) == 2
    assert spec.kv_rows_total >= admission.admission_cfg.kv_operating_target_tokens
    assert len(request_queue) == 3          # 나머지 3개는 다음 슬롯/tick 몫


def test_layer1_per_mb_kv_budget_splits(admission, request_queue):
    """max_mb_kv_tokens 한도(per-slot disjoint 분할)까지만 admit, 초과분 defer."""
    for i in range(5):
        request_queue.push(_make_req(i, kv_length=500_000))
    spec = admission.layer1(max_mb_kv_tokens=2_000_000)
    assert spec is not None
    assert len(spec.decode_requests) == 4   # 4×500K = 2.0M ≤ 예산
    assert len(request_queue) == 1           # 5번째 defer → 다음 슬롯/tick 몫


def test_layer1_per_mb_kv_first_req_exception(admission, request_queue):
    """빈 batch 는 per-mb 예산 초과 단일 거대요청도 첫 후보로 허용(starvation 방지)."""
    request_queue.push(_make_req(0, kv_length=3_000_000))   # > 예산 2M
    spec = admission.layer1(max_mb_kv_tokens=2_000_000)
    assert spec is not None
    assert len(spec.decode_requests) == 1    # 거대요청 단독 admit (빈 batch 첫 후보 면제)
    assert spec.decode_requests[0].id == 0
    # (예산 초과분의 후속 defer 는 test_layer1_per_mb_kv_budget_splits 가 커버)


# --- former-v2 steering + age-cap (OPERATING_POINT §3) ---

def test_layer1_steering_picks_closest_to_ideal(admission, request_queue):
    """첫 step ideal=12.3M/123≈100K — kv_length 가 가장 가까운 디코더를 먼저 admit."""
    request_queue.push(_make_req(0, kv_length=10_000))
    request_queue.push(_make_req(1, kv_length=100_000))    # ideal 최근접
    request_queue.push(_make_req(2, kv_length=500_000))
    spec = admission.layer1()
    assert spec is not None
    assert spec.decode_requests[0].id == 1                 # closest-to-ideal 먼저


def test_layer1_age_cap_forces_aged_request(admission, request_queue):
    """wait ≥ age_cap 인 off-size 요청은 steering 무시하고 강제 admit (공정성)."""
    aged = _make_req(0, kv_length=2_000_000)               # off-size — steering 이면 후순위
    aged.wait = admission.admission_cfg.age_cap            # age_cap 도달
    request_queue.push(aged)
    request_queue.push(_make_req(1, kv_length=100_000))    # ideal 근접 (steering 이면 먼저)
    spec = admission.layer1()
    assert spec.decode_requests[0].id == 0                 # 강제 → aged 가 먼저


def test_layer1_deferred_requests_age(admission, request_queue):
    """타깃 도달로 미선택된 후보는 wait += 1 후 re-push (age-cap 누적)."""
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=10_000_000))   # 10M each → 2개서 12.3M 초과
    admission.layer1()
    leftover = request_queue.peek_oldest()
    assert leftover is not None
    assert leftover.wait == 1


def test_layer1_returns_microbatch_spec(admission, request_queue):
    request_queue.push(_make_req(0))
    spec = admission.layer1()
    assert isinstance(spec, MicroBatchSpec)
    # prefill 256 고정 (former-v2 동작점)
    assert spec.prefill_chunk_tokens == admission.admission_cfg.prefill_chunk_default
    assert isinstance(spec.decode_requests, tuple)


def test_layer1_kv_admit_release_roundtrip(admission, request_queue, kv_accountant):
    initial = kv_accountant.remaining
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=100))
    spec = admission.layer1()
    assert kv_accountant.remaining == initial - 300
    for req in spec.decode_requests:
        kv_accountant.release(req)
    assert kv_accountant.remaining == initial


def test_layer1_dispatcher_roundtrip(admission, request_queue, window, dispatcher):
    """Cross-module: admission admit → window.admit(mb_id) → DAG add → refresh_ready → QKV READY."""
    from puls_sched.node import NodeState, NodeType
    request_queue.push(_make_req(0))
    spec = admission.layer1()
    assert spec is not None
    mb_id = 0
    window.admit(mb_id)
    dispatcher.refresh_ready()
    qkv = dispatcher.dag.get_node(mb_id, NodeType.QKV)
    assert qkv.state is NodeState.READY


def test_layer1_determinism_same_inputs_same_spec(admission_config, idle_telemetry):
    """동일 입력 (queue + kv) 2회 reset → 동일 MicroBatchSpec."""
    from puls_sched.admission import Admission
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    def run_once():
        rq = RequestQueue(capacity=admission_config.request_queue_capacity)
        kv = KVAccountant(capacity=admission_config.kv_capacity_aggregate)
        adm = Admission(admission_cfg=admission_config, request_queue=rq,
                        kv_accountant=kv, idle_telemetry=idle_telemetry)
        for i in range(3):
            rq.push(_make_req(i, kv_length=100))
        return adm.layer1()

    a = run_once()
    b = run_once()
    assert a == b


# =========================================================================
# Impl-5 — MicroBatchSpec.kv_rows_total (signal flow to dispatcher)
# =========================================================================

def test_admission_spec_has_kv_rows_total():
    """MicroBatchSpec 에 kv_rows_total 필드 존재 + int 타입."""
    fields = MicroBatchSpec.__dataclass_fields__
    assert "kv_rows_total" in fields
    assert fields["kv_rows_total"].type is int


def test_admission_kv_rows_total_sums_decode_reqs(admission, request_queue):
    """Σ kv_length over decode_reqs == spec.kv_rows_total."""
    request_queue.push(_make_req(0, kv_length=100))
    request_queue.push(_make_req(1, kv_length=250))
    request_queue.push(_make_req(2, kv_length=75))
    spec = admission.layer1()
    assert spec is not None
    assert spec.kv_rows_total == sum(r.kv_length for r in spec.decode_requests)
    assert spec.kv_rows_total == 100 + 250 + 75


def test_admission_kv_rows_total_zero_if_no_decode(admission):
    """Empty queue → spec None (기존 동작 보존)."""
    assert admission.layer1() is None


def test_admission_kv_rows_total_monotonic_with_req_count(admission_config, idle_telemetry):
    """decode req 개수 증가 → kv_rows_total 비감소 (각 req kv_length 동일)."""
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    prev = -1
    for n_reqs in [1, 3, 5, 10]:
        rq = RequestQueue(capacity=admission_config.request_queue_capacity)
        kv = KVAccountant(capacity=admission_config.kv_capacity_aggregate)
        adm = Admission(admission_cfg=admission_config, request_queue=rq,
                        kv_accountant=kv, idle_telemetry=idle_telemetry)
        for i in range(n_reqs):
            rq.push(_make_req(i, kv_length=50))
        spec = adm.layer1()
        assert spec.kv_rows_total >= prev
        prev = spec.kv_rows_total
