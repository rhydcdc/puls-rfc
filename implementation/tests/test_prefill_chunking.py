"""Cluster P — Prefill chunking dispatch (Impl-10-pre-2 O9.1).

ARCH §5.5 chunked-prefill primitive 정합. Request.prefill_processed 단조 진행 + MicroBatch.prefill_chunk
populate + Dispatcher.prefill_attn chunk-scaled op_time + main_loop 의 prefill 완료 검출 (PREFILL → DECODE).
"""

import pytest

from puls_sched.admission import Admission
from puls_sched.config import default_dummy_config
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.micro_batch import MicroBatch
from puls_sched.request import Request, RequestState
from puls_sched.request_queue import RequestQueue


# ---- Request.prefill_processed field ----

def test_request_prefill_processed_default_zero():
    """Request.prefill_processed default = 0 (PLAN §0.5 backward-compat)."""
    req = Request(id=0, prompt_len=3, kv_length=3)
    assert req.prefill_processed == 0


def test_request_prefill_processed_advances_monotonic():
    """prefill_processed 영역의 단조 증가 가정 검증."""
    req = Request(id=0, prompt_len=1000, kv_length=1000)
    req.prefill_processed = 256
    assert req.prefill_processed == 256
    req.prefill_processed = 768
    assert req.prefill_processed == 768
    req.prefill_processed = 1000
    assert req.prefill_processed == req.prompt_len


# test_admission_layer1_uses_prefill_chunk_default_as_base 삭제 — layer1(admission cohort)
# 폐기(풀 모델). prefill 예산 = config.prefill_chunk_default 는 main_loop 구성서 _populate
# 호출 시 전달(test_prefill_steering_* 가 분배 검증). admission 분리는 acceptance/lifecycle 커버.


# ---- Dispatcher — PREFILL_ATTN chunk-scaled op_time ----

def test_dispatcher_prefill_attn_chunk_scaled(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """PREFILL_ATTN op_time = spec-derived FlashAttention flops / peak (compute_gpu_op_time_s)."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType, NodeState
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    dag.add_micro_batch(0)
    # mb 위 prefill_chunk 512 token populate
    mb = MicroBatch(
        id=0, kv_rows_total=100,
        prefill_chunk={1: list(range(512))},   # 512 tokens
    )
    d.register(mb)
    node = dag.get_node(0, NodeType.PREFILL_ATTN)
    # Stage 2 — FlashAttention causal: 2 × 512 × hidden × (prefill_processed + 512)
    # prefill_processed default = 0 → ctx = 512
    expected_flops = 2 * 512 * dummy_config.model.hidden * 512
    # TP=8 마이그레이션(40e812a) — GEMM 이 num_gpus_instance_a 장에 분산 (peak ×8).
    peak = (dummy_config.calibration.gpu_fp16_dense_peak_tflops * 1e12
            * dummy_config.calibration.gpu_mfu_default * dummy_config.hw.num_gpus_instance_a)
    expected_us = expected_flops / peak * 1e6
    assert d._op_time(node) == pytest.approx(expected_us)


def test_dispatcher_prefill_attn_decode_only_fallback(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """Decode-only mb (prefill_chunk={}) 위 PREFILL_ATTN op_time = 기존 lookup (backward-compat)."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    dag.add_micro_batch(0)
    mb = MicroBatch(id=0, kv_rows_total=100, decode_tokens={1: 0})  # decode-only
    d.register(mb)
    node = dag.get_node(0, NodeType.PREFILL_ATTN)
    # Stage 2 — decode-only mb (prefill_chunk={}) → PREFILL_ATTN op_time = 0.0 (FLOPs 0)
    assert d._op_time(node) == 0.0


def test_dispatcher_prefill_attn_scales_with_chunk_size(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """Chunk size 위 op_time monotonic 증가."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    sizes = [128, 256, 512, 1024]
    times = []
    for i, sz in enumerate(sizes):
        dag.add_micro_batch(i)
        mb = MicroBatch(id=i, kv_rows_total=100,
                        prefill_chunk={1: list(range(sz))})
        d.register(mb)
        node = dag.get_node(i, NodeType.PREFILL_ATTN)
        times.append(d._op_time(node))
    # Strictly monotonic increasing
    for i in range(len(times) - 1):
        assert times[i] < times[i + 1]


# ---- Multi-req prefill chunk (ARCH §5.2 uniform-chunk) ----

def test_dispatcher_prefill_attn_multi_req_chunk_sum(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """Multi-req prefill chunk 위 PREFILL_ATTN op_time = sum(chunk_lengths) × per-token."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    dag.add_micro_batch(0)
    # 2 req × 256 token each (uniform-chunk 정합)
    mb = MicroBatch(
        id=0, kv_rows_total=100,
        prefill_chunk={1: list(range(256)), 2: list(range(256))},
    )
    d.register(mb)
    node = dag.get_node(0, NodeType.PREFILL_ATTN)
    # Stage 2 — multi-req per-req sum: 2 × Σ(chunk × ctx_so_far)
    # 2 reqs × chunk=256, prefill_processed=0 → 각 req FLOPs = 2 × 256 × hidden × 256
    expected_flops = 2 * (2 * 256 * dummy_config.model.hidden * 256)
    # TP=8 마이그레이션(40e812a) — peak ×num_gpus_instance_a.
    peak = (dummy_config.calibration.gpu_fp16_dense_peak_tflops * 1e12
            * dummy_config.calibration.gpu_mfu_default * dummy_config.hw.num_gpus_instance_a)
    expected_us = expected_flops / peak * 1e6
    assert d._op_time(node) == pytest.approx(expected_us)


# ---- Boundary ----

def test_dispatcher_prefill_attn_empty_chunk_fallback(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """Empty prefill_chunk (key 존재, list 비어있음) 위 fallback."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    dag.add_micro_batch(0)
    mb = MicroBatch(id=0, kv_rows_total=100, prefill_chunk={1: []})
    d.register(mb)
    node = dag.get_node(0, NodeType.PREFILL_ATTN)
    # Stage 2 — empty chunk (sum len = 0) → PREFILL_ATTN op_time = 0.0
    assert d._op_time(node) == 0.0


# ---- Determinism ----

def test_prefill_chunk_op_time_deterministic(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """동일 mb 위 1000-iter 동일 op_time."""
    from puls_sched.dispatcher import Dispatcher
    from puls_sched.node import NodeType
    d = Dispatcher(
        config=dummy_config, clock=clock, queue=event_queue, dag=dag,
        pim_executor=pim_executor,
    )
    dag.add_micro_batch(0)
    mb = MicroBatch(id=0, kv_rows_total=100,
                    prefill_chunk={1: list(range(512))})
    d.register(mb)
    node = dag.get_node(0, NodeType.PREFILL_ATTN)
    results = [d._op_time(node) for _ in range(1000)]
    assert len(set(results)) == 1


# =========================================================================
# former-v2 prefill steering — _populate_mb_phases (OPERATING_POINT §3)
# =========================================================================
#
# prefill 토큰 256 고정, 분배로 depth-합(Σ chunk×depth)을 25.6M 에 맞춤. ideal=
# (target−W)/(budget−t) 깊이 최근접 멤버에 토큰 배정. ragged chunk 허용(B 는 batch
# 총 토큰수만 봄). _populate_mb_phases 는 self.config 만 쓰므로 scheduler_core fixture 로 직접.


def _prefill_req(req_id: int, depth: int, headroom: int = 300) -> Request:
    """depth(prefill_processed) 까지 처리됐고 headroom 만큼 프롬프트 남은 prefill 요청."""
    r = Request(id=req_id, prompt_len=(depth + headroom))
    r.prefill_processed = depth
    return r


def _depth_work(prefill_chunk: dict[int, list[int]]) -> int:
    """Σ over reqs Σ(배정 토큰의 causal 깊이) — chunk = range(pp, pp+c) 라 위치합 = depth-work."""
    return sum(sum(positions) for positions in prefill_chunk.values())


def test_prefill_steering_total_tokens_fixed(scheduler_core):
    """분배 후 총 prefill 토큰 = budget(256) (남은 프롬프트 충분할 때)."""
    reqs = [_prefill_req(0, 80_000), _prefill_req(1, 120_000)]
    prefill_chunk, _, _ = scheduler_core._populate_mb_phases(reqs, 256)
    assert sum(len(c) for c in prefill_chunk.values()) == 256


def test_prefill_steering_avoids_overshoot_on_deep(scheduler_core):
    """ideal(100K) 근접 멤버가 토큰 독식, 과도하게 깊은 멤버는 ~0 → depth-합이 균등분배보다
    25.6M 에 가깝다(깊은 멤버에 토큰 주면 오버슈트)."""
    near, deep = _prefill_req(0, 100_000), _prefill_req(1, 300_000)
    prefill_chunk, _, _ = scheduler_core._populate_mb_phases([near, deep], 256)
    assert len(prefill_chunk.get(0, [])) > len(prefill_chunk.get(1, []))
    target = scheduler_core.config.admission.prefill_kv_work_target_tokens
    w_steered = _depth_work(prefill_chunk)
    w_uniform = 128 * 100_000 + 128 * 300_000   # 균등분배(128 each) 근사
    assert abs(w_steered - target) < abs(w_uniform - target)


def test_prefill_steering_prefers_closer_to_ideal_depth(scheduler_core):
    """둘 다 얕아도(ideal 100K 미만) 더 깊은(=ideal 근접) 멤버가 더 받음."""
    shallow, mid = _prefill_req(0, 50_000), _prefill_req(1, 90_000)
    prefill_chunk, _, _ = scheduler_core._populate_mb_phases([shallow, mid], 256)
    assert len(prefill_chunk.get(1, [])) > len(prefill_chunk.get(0, []))


def test_prefill_single_req_gets_budget(scheduler_core):
    """prefill 멤버 1개면 budget 전량 + chunk = range(depth, depth+256) (causal 연속)."""
    prefill_chunk, _, processed = scheduler_core._populate_mb_phases(
        [_prefill_req(0, 100_000)], 256)
    assert prefill_chunk[0] == list(range(100_000, 100_256))
    assert processed[0] == 100_000


def test_prefill_decode_only_no_chunk(scheduler_core):
    """프롬프트 소진(prefill_processed ≥ len) 요청은 decode_tokens 로, prefill_chunk 제외."""
    done = Request(id=0, prompt_len=100)
    done.prefill_processed = 100
    prefill_chunk, decode_tokens, _ = scheduler_core._populate_mb_phases([done], 256)
    assert 0 not in prefill_chunk
    assert decode_tokens == {0: 0}


def test_prefill_steering_age_cap_forces_starved(scheduler_core):
    """prefill_wait ≥ age_cap 인 요청은 steering 무시하고 강제 ≥1 토큰 (prefill starvation 0)."""
    near = _prefill_req(0, 100_000)              # steering 이 독식할 ideal-근접
    starved = _prefill_req(1, 400_000)           # off-ideal — steering 만이면 0
    starved.prefill_wait = scheduler_core.config.admission.age_cap
    prefill_chunk, _, _ = scheduler_core._populate_mb_phases([near, starved], 256)
    assert len(prefill_chunk[1]) >= 1            # 강제 배정


def test_prefill_steering_zero_token_req_stays_in_pool(scheduler_core):
    """풀 모델 — steering 이 0토큰 준 요청은 prefill_chunk 에 *안 들어감*(이 μ-batch 미assign,
    풀 잔류 → 다음 구성 재선택) + prefill_wait++(age-cap). (sticky 모델의 빈-chunk 멤버십 유지를
    풀 모델선 제거 — mb 를 미진행 요청으로 부풀려 prefill 256 과분산·decode 고갈 회귀 방지.)"""
    near = _prefill_req(0, 100_000)
    deep = _prefill_req(1, 2_000_000)            # 너무 깊어 steering 0토큰
    prefill_chunk, _, _ = scheduler_core._populate_mb_phases([near, deep], 256)
    assert 1 not in prefill_chunk                # 0토큰 → mb 에 안 넣음(풀 잔류)
    assert deep.prefill_wait == 1                # 0토큰 → 대기 누적(age-cap)
