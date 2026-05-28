"""Cluster Z — PIM-time-driven adaptive chunk (Impl-10-pre-2 B option).

ARCH §3.5.2 *Computed Wait* + §6.3 *PIM completion time computed at dispatch* + §6.1
*attention split (PREFILL_ATTN ‖ DECODE_ATTN concurrent)* + §3.5.3 *PIM-GPU TSV margin* +
§5.2 *uniform-chunk* literal 정합.

Impl-10-pre-2 — k_total knob 폐기 (sequence-parallel PIM 위 k 영원 k_aggregate).
balance_pim_slack signature: (prefill_chunk_tokens, t_pim_fn(n_decode)→float, n_decode, per_token, t_gpu_base).

산식:
    chunk_total = max(0, t_pim × margin − t_gpu_base) / per_token
    chunk_per_req = chunk_total // N_prefill   (main_loop._populate_mb_phases, Option A)
"""

import pytest

from puls_sched.admission import Admission
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.request import Request
from puls_sched.request_queue import RequestQueue


def _make_adm(cfg):
    rq = RequestQueue(capacity=cfg.admission.request_queue_capacity)
    kv = KVAccountant(capacity=cfg.admission.kv_capacity_aggregate)
    tel = IdleTelemetry()
    tel.reset(0.0)
    return Admission(
        admission_cfg=cfg.admission, request_queue=rq,
        kv_accountant=kv, idle_telemetry=tel,
    ), rq, tel


# ---- balance_pim_slack — unit ----

def test_balance_pim_slack_basic_with_margin(dummy_config):
    """TOTAL chunk budget = (t_pim × margin − t_gpu_base) / per_token.

    t_pim=100us, t_gpu_base=25us, per_token=0.01us, margin=0.9
    → chunk_total = (100 × 0.9 − 25) / 0.01 = 6500
    """
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n: 100.0,
        n_decode=4,
        gpu_op_time_per_token_us=0.01,
        t_gpu_base=25.0,
    )
    margin = dummy_config.admission.pim_slack_safety_margin
    expected = max(512, int((100.0 * margin - 25.0) / 0.01))
    assert chunk == expected


def test_balance_pim_slack_small_t_pim_preserves_base(dummy_config):
    """t_pim 작음 위 chunk_optimal < base → base 보존."""
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n: 2.0,
        n_decode=4,
        gpu_op_time_per_token_us=0.01,
        t_gpu_base=0.5,
    )
    # slack = 2*0.9 - 0.5 = 1.3 → chunk = 130 < 512 → base 보존
    assert chunk == 512


def test_balance_pim_slack_zero_n_decode_preserves_base(dummy_config):
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n: 100.0,
        n_decode=0,
        gpu_op_time_per_token_us=0.01,
        t_gpu_base=25.0,
    )
    assert chunk == 512


def test_balance_pim_slack_zero_per_token_preserves_base(dummy_config):
    """per_token=0 위 division 회피 → base 보존."""
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n: 100.0,
        n_decode=4,
        gpu_op_time_per_token_us=0.0,
        t_gpu_base=25.0,
    )
    assert chunk == 512


def test_balance_pim_slack_t_gpu_base_dominates_returns_base(dummy_config):
    """Edge case — t_pim × margin ≤ t_gpu_base → slack=0 → base 보존."""
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n: 100.0,
        n_decode=4,
        gpu_op_time_per_token_us=0.01,
        t_gpu_base=100.0,   # × 0.9 = 90 ≤ t_gpu_base
    )
    assert chunk == 512


# ---- balance_pim_slack — monotonic ----

@pytest.mark.parametrize("t_pim_us", [50.0, 100.0, 500.0, 1000.0, 5000.0])
def test_balance_pim_slack_monotonic_with_t_pim(dummy_config, t_pim_us):
    """t_pim 증가 → chunk_optimal monotonic 증가."""
    adm, _, _ = _make_adm(dummy_config)
    chunk = adm.balance_pim_slack(
        prefill_chunk_tokens=512,
        t_pim_fn=lambda n, _t=t_pim_us: _t,
        n_decode=4,
        gpu_op_time_per_token_us=0.01,
        t_gpu_base=25.0,
    )
    margin = dummy_config.admission.pim_slack_safety_margin
    expected = max(512, int(max(0.0, t_pim_us * margin - 25.0) / 0.01))
    assert chunk == expected


def test_balance_pim_slack_subtracts_t_gpu_base(dummy_config):
    """핵심 — 같은 t_pim, t_gpu_base 증가 → chunk 감소 (slack 축소)."""
    adm, _, _ = _make_adm(dummy_config)
    chunks = []
    for t_gpu_base in [0.0, 10.0, 25.0, 50.0]:
        chunk = adm.balance_pim_slack(
            prefill_chunk_tokens=0,
            t_pim_fn=lambda n: 100.0,
            n_decode=4,
            gpu_op_time_per_token_us=0.01,
            t_gpu_base=t_gpu_base,
        )
        chunks.append(chunk)
    for i in range(len(chunks) - 1):
        assert chunks[i] >= chunks[i + 1], f"non-monotonic: {chunks}"


# ---- admission.layer1 — integration ----

def test_layer1_pim_bound_regime_activates_adaptive_chunk(dummy_config):
    """PIM-bound regime (long-ctx) 위 balance_pim_slack 가 base 초과 chunk 산출.

    t_proj=25us, t_pim=100us, per_token=0.01us, margin=0.9
    → slack = 100*0.9 - 25 = 65us → chunk_total = 6500 (base 512 초과)
    """
    adm, rq, _ = _make_adm(dummy_config)
    rq.push(Request(id=0, prompt_tokens=[0] * 10000, kv_length=100, max_tokens=10))
    spec = adm.layer1(
        t_proj=25.0,
        t_pim_fn=lambda n: 100.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        gpu_op_time_per_token_us=0.01,
    )
    assert spec is not None
    margin = dummy_config.admission.pim_slack_safety_margin
    expected_min = int(max(0.0, 100.0 * margin - 25.0) / 0.01)
    assert spec.prefill_chunk_tokens >= expected_min


def test_layer1_gpu_bound_regime_keeps_base(dummy_config):
    """GPU-bound regime (short-ctx) 위 chunk = base.

    t_proj=200us, t_pim=100us → slack = 90-200 < 0 → chunk_optimal=0 → base.
    """
    adm, rq, _ = _make_adm(dummy_config)
    rq.push(Request(id=0, prompt_tokens=[0] * 1000, kv_length=100, max_tokens=10))
    spec = adm.layer1(
        t_proj=200.0,
        t_pim_fn=lambda n: 100.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
        gpu_op_time_per_token_us=0.01,
    )
    assert spec is not None
    assert spec.prefill_chunk_tokens == dummy_config.admission.prefill_chunk_default


def test_layer1_backward_compat_without_per_token(dummy_config):
    """gpu_op_time_per_token_us 미전달 (default 0.0) 위 base 영원 보존."""
    adm, rq, _ = _make_adm(dummy_config)
    rq.push(Request(id=0, prompt_tokens=[0] * 1000, kv_length=100, max_tokens=10))
    spec = adm.layer1(
        t_proj=200.0, t_pim_fn=lambda n: 100.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,
    )
    assert spec is not None
    assert spec.prefill_chunk_tokens == dummy_config.admission.prefill_chunk_default


# ---- End-to-end Run.init payload 정합 ----

def test_run_init_payload_includes_per_token(tmp_path):
    """Run.init 위 _compose_admission_payload 가 gpu_op_time_per_token_us 영역 포함."""
    from puls_sched.run import Run
    run = Run.init(
        config_module="puls_sched.config:default_dummy_config",
        trace_path_or_synthetic="synthetic:5",
        output_dir=tmp_path,
        seed=42,
    )
    payload = run.scheduler._compose_admission_payload()
    assert "gpu_op_time_per_token_us" in payload
    assert payload["gpu_op_time_per_token_us"] == run.config.time.gpu_op_time_per_token_us


# ---- Determinism ----

def test_balance_pim_slack_deterministic(dummy_config):
    """동일 input → 동일 chunk_optimal (1000-iter)."""
    adm, _, _ = _make_adm(dummy_config)
    results = []
    for _ in range(1000):
        chunk = adm.balance_pim_slack(
            prefill_chunk_tokens=512,
            t_pim_fn=lambda n: 100.0,
            n_decode=4,
            gpu_op_time_per_token_us=0.01,
            t_gpu_base=25.0,
        )
        results.append(chunk)
    assert len(set(results)) == 1
