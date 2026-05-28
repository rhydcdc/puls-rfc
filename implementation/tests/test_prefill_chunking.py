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
    req = Request(id=0, prompt_tokens=[1, 2, 3], kv_length=3)
    assert req.prefill_processed == 0


def test_request_prefill_processed_advances_monotonic():
    """prefill_processed 영역의 단조 증가 가정 검증."""
    req = Request(id=0, prompt_tokens=list(range(1000)), kv_length=1000)
    req.prefill_processed = 256
    assert req.prefill_processed == 256
    req.prefill_processed = 768
    assert req.prefill_processed == 768
    req.prefill_processed = 1000
    assert req.prefill_processed == len(req.prompt_tokens)


# ---- Admission.layer1 — Hybrid base = prefill_chunk_default ----

def test_admission_layer1_uses_prefill_chunk_default_as_base(dummy_config):
    """Impl-10-pre-2 — Hybrid policy 의 base = prefill_chunk_default (Sarathi/vLLM 영역 정합)."""
    cfg = dummy_config
    rq = RequestQueue(capacity=cfg.admission.request_queue_capacity)
    kv = KVAccountant(capacity=cfg.admission.kv_capacity_aggregate)
    tel = IdleTelemetry()
    tel.reset(0.0)
    adm = Admission(
        admission_cfg=cfg.admission, request_queue=rq, kv_accountant=kv,
        idle_telemetry=tel,
    )
    req = Request(id=0, prompt_tokens=[0] * 100, kv_length=100, max_tokens=5)
    rq.push(req)
    spec = adm.layer1(
        t_proj=1e9, t_pim_fn=lambda n: 0.0,
        a_cycle=10.0, b_cycle=10.0, ctx_tokens=100,    # balanced → no adjustment
    )
    assert spec is not None
    # Base = prefill_chunk_default (512), balance 위 변경 0 (in-band)
    assert spec.prefill_chunk_tokens == cfg.admission.prefill_chunk_default


# ---- Dispatcher — PREFILL_ATTN chunk-scaled op_time ----

def test_dispatcher_prefill_attn_chunk_scaled(
    dummy_config, clock, event_queue, dag, pim_executor,
):
    """PREFILL_ATTN op_time = chunk_tokens × gpu_op_time_per_token_us (Impl-10-pre-2 (O9.1))."""
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
    peak = dummy_config.calibration.gpu_fp16_dense_peak_tflops * 1e12 * dummy_config.calibration.gpu_mfu_default
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
    peak = dummy_config.calibration.gpu_fp16_dense_peak_tflops * 1e12 * dummy_config.calibration.gpu_mfu_default
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
