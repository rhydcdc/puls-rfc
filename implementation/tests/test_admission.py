import pytest

from puls_sched.admission import Admission, MicroBatchSpec
from puls_sched.request import Request


def _make_req(req_id: int, kv_length: int = 10) -> Request:
    return Request(id=req_id, prompt_tokens=[1], kv_length=kv_length)


# --- mfu_floor ---

@pytest.mark.parametrize("n", [0, 1, 15])
def test_mfu_floor_below_n_sat_clamps_up(admission, admission_config, n):
    assert admission.mfu_floor(n) == admission_config.n_sat


def test_mfu_floor_at_n_sat_unchanged(admission, admission_config):
    assert admission.mfu_floor(admission_config.n_sat) == admission_config.n_sat


@pytest.mark.parametrize("delta", [1, 16])
def test_mfu_floor_above_n_sat_unchanged(admission, admission_config, delta):
    n = admission_config.n_sat + delta
    assert admission.mfu_floor(n) == n


# --- balance_inter_AB ---

def test_balance_inter_in_band_no_change(admission):
    # ctx=100 (short), width=1.0; diff=0 → in-band
    assert admission.balance_inter_AB(0, a_cycle=10.0, b_cycle=10.0, ctx_tokens=100) == 0


def test_balance_inter_a_idle_admits_prefill(admission, admission_config):
    # A < B, diff = -5 outside width 1.0 → prefill 증가
    out = admission.balance_inter_AB(0, a_cycle=5.0, b_cycle=10.0, ctx_tokens=100)
    assert out == admission_config.n_sat


def test_balance_inter_b_idle_no_change(admission):
    # A > B, diff = +5 outside width — §6.4 표 "B idle naturally accepted" → no change
    out = admission.balance_inter_AB(0, a_cycle=10.0, b_cycle=5.0, ctx_tokens=100)
    assert out == 0


def test_balance_inter_ctx_tier_short_outside_mid_inside(admission, admission_config):
    """diff = -1.5: short tier (w=1.0) out-of-band → change; mid tier (w=2.0) in-band → no change."""
    out_short = admission.balance_inter_AB(0, a_cycle=8.5, b_cycle=10.0, ctx_tokens=100)
    out_mid = admission.balance_inter_AB(0, a_cycle=8.5, b_cycle=10.0, ctx_tokens=10_000)
    assert out_short == admission_config.n_sat
    assert out_mid == 0


# --- balance_intra_A ---

# STEP 3-b — balance_intra_A 는 prefill chunk 증량만 담당 (decode 조절은 _try_join 전담).
# 시그니처: balance_intra_A(prefill_chunk_tokens) -> prefill_chunk_tokens.

def test_balance_intra_gpu_idle_admits_prefill(admission, admission_config, idle_telemetry):
    """GPU idle (PIM busy) → prefill chunk + n_sat (idle GPU 를 PREFILL_ATTN 으로 활용)."""
    idle_telemetry.reset(0.0)
    idle_telemetry.record_active("PIM", 0.0, 10.0)
    # gpu_idle = 1.0, pim_idle = 0.0; θ_high = 0.3
    prefill = admission.balance_intra_A(0)
    assert prefill == admission_config.n_sat   # GPU 한테 일 더 줌 (prefill chunk)


def test_balance_intra_pim_idle_no_prefill_change(admission, idle_telemetry):
    """PIM idle (GPU busy) → prefill chunk 변경 0. decode 조절은 _try_join 전담 (여기 무관)."""
    idle_telemetry.reset(0.0)
    idle_telemetry.record_active("GPU", 0.0, 10.0)
    # gpu_idle=0, pim_idle=1.0 → prefill 방향 게이트 닫힘
    prefill = admission.balance_intra_A(0)
    assert prefill == 0                         # prefill chunk 변경 0


def test_balance_intra_both_below_theta_no_change(admission, idle_telemetry):
    # Both fully active → both idle 0 → no change
    idle_telemetry.reset(0.0)
    idle_telemetry.record_active("GPU", 0.0, 10.0)
    idle_telemetry.record_active("PIM", 0.0, 10.0)
    prefill = admission.balance_intra_A(0)
    assert prefill == 0


def test_balance_intra_both_above_theta_no_change(admission, idle_telemetry):
    # No activity → both idle 1.0 → ambiguous → no change (conservative)
    idle_telemetry.reset(0.0)
    idle_telemetry.record_active("GPU", 10.0, 10.0)  # zero-duration window extend
    prefill = admission.balance_intra_A(0)
    assert prefill == 0


# --- layer1 ---

def test_layer1_empty_queue_returns_none(admission):
    spec = admission.layer1(
        t_proj=100.0, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is None


def test_layer1_admits_decode_within_kv_capacity(admission, request_queue):
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=100))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is not None
    assert len(spec.decode_requests) == 3


def test_layer1_stops_at_kv_capacity(request_queue, kv_accountant, admission_config, idle_telemetry):
    # Custom KVAccountant with tight capacity
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.admission import Admission
    small_kv = KVAccountant(capacity=1000)
    adm = Admission(
        admission_cfg=admission_config,
        request_queue=request_queue,
        kv_accountant=small_kv,
        idle_telemetry=idle_telemetry,
    )
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=500))
    spec = adm.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is not None
    assert len(spec.decode_requests) == 2
    assert len(request_queue) == 1


def test_layer1_per_mb_kv_budget_splits(admission, request_queue):
    """STEP 5.5 — max_mb_kv_tokens 한도까지만 admit, 초과분 defer (mb 강제 분할)."""
    for i in range(5):
        request_queue.push(_make_req(i, kv_length=500_000))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        max_mb_kv_tokens=2_000_000,
    )
    assert spec is not None
    assert len(spec.decode_requests) == 4   # 4×500K = 2.0M ≤ 예산
    assert len(request_queue) == 1           # 5번째 defer → 다음 tick 의 mb 1 몫


def test_layer1_per_mb_kv_first_req_exception(admission, request_queue):
    """빈 mb 는 예산 초과 단일 거대요청도 1개 허용(starvation 방지), 이후 req 는 defer."""
    request_queue.push(_make_req(0, kv_length=3_000_000))   # > 예산 2M
    request_queue.push(_make_req(1, kv_length=500_000))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        max_mb_kv_tokens=2_000_000,
    )
    assert spec is not None
    assert len(spec.decode_requests) == 1    # 거대요청 단독 admit (첫 req 예외)
    assert spec.decode_requests[0].id == 0
    assert len(request_queue) == 1            # req1 은 이미 3M 초과라 defer


def test_layer1_returns_microbatch_spec(admission, request_queue):
    request_queue.push(_make_req(0))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert isinstance(spec, MicroBatchSpec)
    assert spec.prefill_chunk_tokens >= 0
    assert isinstance(spec.decode_requests, tuple)
    assert spec.n >= 0


def test_layer1_mfu_floor_applied(admission, admission_config, request_queue):
    request_queue.push(_make_req(0))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec.n == admission_config.n_sat


def test_layer1_kv_admit_release_roundtrip(admission, request_queue, kv_accountant):
    initial = kv_accountant.remaining
    for i in range(3):
        request_queue.push(_make_req(i, kv_length=100))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert kv_accountant.remaining == initial - 300
    for req in spec.decode_requests:
        kv_accountant.release(req)
    assert kv_accountant.remaining == initial


def test_layer1_dispatcher_roundtrip(admission, request_queue, window, dispatcher):
    """Cross-module: admission admit → window.admit(mb_id) → DAG add → refresh_ready → QKV READY."""
    from puls_sched.node import NodeState, NodeType
    request_queue.push(_make_req(0))
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
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
        return adm.layer1(
            t_proj=1e9, t_pim_fn=lambda n: float(n) * 0.5,
            a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        )

    a = run_once()
    b = run_once()
    assert a == b


# --- Multi-iteration convergence (acceptance 진정성) ---

def test_layer1_multi_iteration_converges_in_deadband(admission_config, idle_telemetry):
    """동일 admission 10 iter → spec.prefill_chunk_tokens 시퀀스 stable.
    Balanced input (a_cycle == b_cycle) → balance_inter 가 첫 호출부터 in-band → 변화 0."""
    from puls_sched.admission import Admission
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    rq = RequestQueue(capacity=admission_config.request_queue_capacity)
    kv = KVAccountant(capacity=admission_config.kv_capacity_aggregate)
    adm = Admission(admission_cfg=admission_config, request_queue=rq,
                    kv_accountant=kv, idle_telemetry=idle_telemetry)

    seq = []
    for it in range(10):
        rq.push(_make_req(it, kv_length=10))
        spec = adm.layer1(
            t_proj=1e9, t_pim_fn=lambda n: 0.0,
            a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        )
        seq.append(spec.prefill_chunk_tokens)

    # Impl-10-pre-2 — Hybrid base = prefill_chunk_default (512). Balanced 영역 위 base 유지.
    assert seq == [admission_config.prefill_chunk_default] * 10


def test_layer1_out_of_band_then_returns_to_band(admission_config, idle_telemetry):
    """iter 1 A << B → balance increases prefill; iter 2 balanced → stable."""
    from puls_sched.admission import Admission
    from puls_sched.kv_accountant import KVAccountant
    from puls_sched.request_queue import RequestQueue

    rq = RequestQueue(capacity=admission_config.request_queue_capacity)
    kv = KVAccountant(capacity=admission_config.kv_capacity_aggregate)
    adm = Admission(admission_cfg=admission_config, request_queue=rq,
                    kv_accountant=kv, idle_telemetry=idle_telemetry)

    rq.push(_make_req(0, kv_length=10))
    spec1 = adm.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=5.0, b_cycle=10.0, ctx_tokens=100,
    )
    rq.push(_make_req(1, kv_length=10))
    spec2 = adm.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    # Impl-10-pre-2 — Hybrid base = prefill_chunk_default. balance 가 adjustment 위 합산.
    assert spec1.prefill_chunk_tokens == admission_config.prefill_chunk_default + admission_config.n_sat  # out-of-band: base + n_sat
    assert spec2.prefill_chunk_tokens == admission_config.prefill_chunk_default                            # in-band: base 만


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
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is not None
    assert spec.kv_rows_total == sum(r.kv_length for r in spec.decode_requests)
    assert spec.kv_rows_total == 100 + 250 + 75


def test_admission_kv_rows_total_zero_if_no_decode(admission):
    """Empty queue → spec None (기존 동작 보존)."""
    spec = admission.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is None


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
        spec = adm.layer1(
            t_proj=1e9, t_pim_fn=lambda n: 0.0,
            a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        )
        assert spec.kv_rows_total >= prev
        prev = spec.kv_rows_total
