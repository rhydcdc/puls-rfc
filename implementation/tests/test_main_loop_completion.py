"""Impl-6 — main_loop._handle(KERNEL_COMPLETION) body 의 token decode signal consumer.

Q5 chain · R2 (EOS path) · R5 (step-by-step sequence).
"""

import pytest

from puls_sched.event import Event, EventType
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType
from puls_sched.request import Request, RequestState


def _admission_event():
    """ADMISSION_TICK event — Phase-2 S2(§2.5): 동작점 고정으로 payload trivial(빈 dict)."""
    return Event(
        timestamp=0.0,
        type=EventType.ADMISSION_TICK,
        payload={},
    )


def _kernel_completion_event(mb_id: int, node_type: NodeType, resource: str = "GPU"):
    return Event(
        timestamp=0.0,
        type=EventType.KERNEL_COMPLETION,
        payload={"micro_batch_id": mb_id, "node_type": node_type, "resource": resource},
    )


def _setup_mb_with_decode_reqs(scheduler_core, n_reqs: int = 1, max_tokens: int = 5,
                                kv_length: int = 100, start_id: int = 0) -> int:
    """admit n_reqs → 1 mb 생성 + dag node 등록.

    Stage 2 — prompt_tokens=[] (이미 prefill 완료 가정) → admission 위 decode-only mb 직접 산출.
    원본 Stage 1 패턴 (prompt_tokens=[0]*5) 위 Stage 2 prefill chunking 위 *first cycle prefill,
    second cycle decode* 영역 변화 — 본 test 의 lifecycle 의도 (decode signal 산출) 와 분리 위 정정.
    """
    for i in range(n_reqs):
        req = Request(id=start_id + i, prompt_tokens=[], kv_length=kv_length,
                      max_tokens=max_tokens)
        scheduler_core.request_queue.push(req)
    scheduler_core._handle(_admission_event())
    mb_id = scheduler_core._next_mb_id - 1
    return mb_id


# ============================================================================
# Q5 — FFN branching (Phase-2 S0: layer advance 트리거 O_PROJ→FFN, inter-AB)
# ============================================================================

def test_ffn_completion_triggers_layer_advance(scheduler_core):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    before = mb.current_layer_index
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    assert mb.current_layer_index == before + 1


# 트리거 = FFN 하나뿐 → O_PROJ 포함 그 외 노드는 advance 없음 (S0 이후).
@pytest.mark.parametrize(
    "ntype",
    [NodeType.QKV, NodeType.PREFILL_ATTN, NodeType.DECODE_ATTN, NodeType.O_PROJ],
)
def test_non_ffn_completion_no_layer_advance(scheduler_core, ntype):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    before = mb.current_layer_index
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, ntype))
    assert mb.current_layer_index == before  # 변경 0


def test_layer_advance_below_l_no_completion_check(scheduler_core, dummy_config):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    # advance once — current_layer_index = 1, num_layers = 80 → below L
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    req = next(iter(scheduler_core.in_flight_requests.values()))
    # decoded_count 증가 안 함 (token decode signal 미발사)
    assert req.decoded_count == 0


def test_layer_advance_at_l_triggers_completion_check(scheduler_core, dummy_config):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    # advance L-1 times manually then trigger signal
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    req = next(iter(scheduler_core.in_flight_requests.values()))
    assert req.decoded_count == 1


def test_token_decode_signal_increments_decoded_count(scheduler_core, dummy_config):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=3, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    # 3 req 모두 +1
    for req in scheduler_core.in_flight_requests.values():
        assert req.decoded_count == 1


def test_token_decode_signal_finalizes_completed_req(scheduler_core, dummy_config, kv_accountant):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=1)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    # max_tokens=1 → 1 step 후 finalize
    assert len(scheduler_core.in_flight_requests) == 0


def test_token_decode_signal_kv_released_on_completion(scheduler_core, dummy_config, kv_accountant):
    initial = kv_accountant.remaining
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=1, kv_length=100)
    # after admit: remaining -= 100
    assert kv_accountant.remaining == initial - 100
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    # finalize → KV release
    assert kv_accountant.remaining == initial


def test_token_decode_signal_does_not_finalize_alive_req(scheduler_core, dummy_config):
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    req = next(iter(scheduler_core.in_flight_requests.values()))
    assert req.state == RequestState.DECODE
    assert req.completion_time is None


def test_layer_index_reset_after_token_signal(scheduler_core, dummy_config):
    """L 도달 후 current_layer_index == 0 (multi-token decode 정합)"""
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    assert mb.current_layer_index == 0


def test_multiple_decode_reqs_independent_finalize(scheduler_core, dummy_config):
    """3 req — max_tokens={1, 5, 10}. 1 step 후 max=1 req 만 finalize"""
    # Stage 2 — prompt_tokens=[] 위 decode-only mb 직접 (prefill chunking 영향 분리)
    for i, mx in enumerate([1, 5, 10]):
        r = Request(id=i, prompt_tokens=[], kv_length=50, max_tokens=mx)
        scheduler_core.request_queue.push(r)
    scheduler_core._handle(_admission_event())
    mb_id = scheduler_core._next_mb_id - 1
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    # max=1 req (id=0) → finalize, 나머지 2 alive
    assert 0 not in scheduler_core.in_flight_requests
    assert 1 in scheduler_core.in_flight_requests
    assert 2 in scheduler_core.in_flight_requests


def test_unregistered_mb_no_crash_defensive(scheduler_core):
    """defensive — unknown mb_id → early return"""
    scheduler_core._maybe_advance_forward_pass(
        _kernel_completion_event(mb_id=9999, node_type=NodeType.FFN)
    )
    # no crash


def test_in_flight_requests_owner_pattern(scheduler_core, dummy_config):
    """Q10 (b) — admit → in_flight_requests 등록, finalize → 제거"""
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=1)
    assert len(scheduler_core.in_flight_requests) == 1
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    assert len(scheduler_core.in_flight_requests) == 0


def test_dispatcher_unregister_called_on_completion(scheduler_core, dummy_config):
    """Impl-9 — Q9 carry-over 해소. mb 의 모든 req finalize 시 dispatcher.unregister + window.evict.

    *Impl-6 시점 의미* (`not_called_on_completion`) 의 ARCH-compliant 갱신 영역.
    """
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=1)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
    # Impl-9 — mb 의 모든 req finalize → window evict + dispatcher unregister + DAG remove
    assert mb_id not in scheduler_core.dispatcher.micro_batches
    assert mb_id not in scheduler_core.window.current_ids()
    assert mb_id not in scheduler_core.dag.nodes


# ============================================================================
# R2 보강 — EOS hybrid path cross-module chain
# ============================================================================

def test_main_loop_eos_seen_external_signal_triggers_finalize(scheduler_core, dummy_config):
    """Q6 (c) EOS branch — eos_seen=True 명시 위 max_tokens 미도달이어도 finalize"""
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(
        _kernel_completion_event(mb_id, NodeType.FFN),
        eos_seen=True,
    )
    # max_tokens=100 이나 EOS path 로 finalize
    assert len(scheduler_core.in_flight_requests) == 0


def test_main_loop_eos_path_kv_release(scheduler_core, dummy_config, kv_accountant):
    initial = kv_accountant.remaining
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=100, kv_length=200)
    assert kv_accountant.remaining == initial - 200
    mb = scheduler_core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = dummy_config.model.num_layers - 1
    scheduler_core._maybe_advance_forward_pass(
        _kernel_completion_event(mb_id, NodeType.FFN),
        eos_seen=True,
    )
    assert kv_accountant.remaining == initial


# ============================================================================
# R5 보강 — multi-token step-by-step sequence
# ============================================================================

@pytest.mark.parametrize("max_tokens", [1, 5, 10, 50])
def test_full_decode_sequence_to_max_tokens(scheduler_core, dummy_config, max_tokens):
    """단일 req 위 max_tokens 회 token decode signal 반복 → step-by-step lifecycle"""
    L = dummy_config.model.num_layers
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=max_tokens, kv_length=10)
    mb = scheduler_core.dispatcher.micro_batches[mb_id]

    for step in range(max_tokens):
        # L-1 회 normal advance + 마지막 1 회는 O_PROJ trigger
        mb.current_layer_index = L - 1
        scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
        if step < max_tokens - 1:
            # alive — decoded_count 증가, mb.current_layer_index reset to 0
            req = scheduler_core.in_flight_requests[0]
            assert req.decoded_count == step + 1
            assert mb.current_layer_index == 0
            assert req.state == RequestState.DECODE
        else:
            # 마지막 step — finalize
            assert 0 not in scheduler_core.in_flight_requests


@pytest.mark.parametrize("max_tokens", [1, 5, 10])
def test_full_decode_sequence_kv_release_only_at_max(scheduler_core, dummy_config,
                                                      kv_accountant, max_tokens):
    """ARCH §3.3 — KV release 가 정확히 마지막 step 에만"""
    L = dummy_config.model.num_layers
    initial = kv_accountant.remaining
    mb_id = _setup_mb_with_decode_reqs(scheduler_core, n_reqs=1, max_tokens=max_tokens, kv_length=100)
    assert kv_accountant.remaining == initial - 100
    mb = scheduler_core.dispatcher.micro_batches[mb_id]

    for step in range(max_tokens):
        mb.current_layer_index = L - 1
        scheduler_core._maybe_advance_forward_pass(_kernel_completion_event(mb_id, NodeType.FFN))
        if step < max_tokens - 1:
            # KV 잔존
            assert kv_accountant.remaining == initial - 100
        else:
            # 회수
            assert kv_accountant.remaining == initial
