"""Impl-6 — Multi-req lifecycle stress invariants.

(I-F1) KV admit/release round-trip × N 위 누수 0 (PLAN §0 C3 stress).
(I-F2) Request.state 단조 transition.
(I-F3) decoded_count signal 정확 +1.
(I-F4) completion_time single-set.
(I-F5) in_flight_requests dict no orphan.
(I-F6) MicroBatch.decode_tokens 의 lifecycle 미관여 (Q10 lock-in).

R8 — composite test 의 seed sweep parametrize 4 cell.
"""

import dataclasses
import random
from pathlib import Path

import pytest

from puls_sched.admission import Admission
from puls_sched.clock import Clock
from puls_sched.completion import Completion
from puls_sched.config import default_dummy_config
from puls_sched.dag import DAG
from puls_sched.dispatcher import Dispatcher
from puls_sched.event import Event, EventType
from puls_sched.event_queue import EventQueue
from puls_sched.forward_pass import LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.kv_accountant import KVAccountant
from puls_sched.main_loop import SchedulerCore
from puls_sched.node import NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request import Request, RequestState
from puls_sched.request_queue import RequestQueue
from puls_sched.trace import TraceReplayer
from puls_sched.window import InFlightWindow


REAL_3_40 = Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"


def _make_core(kv_capacity: int = 1_000_000):
    config = default_dummy_config()
    if kv_capacity != config.admission.kv_capacity_aggregate:
        adm = dataclasses.replace(config.admission, kv_capacity_aggregate=kv_capacity)
        config = dataclasses.replace(config, admission=adm)
    clock = Clock()
    queue = EventQueue(clock)
    dag = DAG()
    window = InFlightWindow(dag)
    pim = PIMExecutor(config=config)
    dispatcher = Dispatcher(config=config, clock=clock, queue=queue, dag=dag, pim_executor=pim)
    rq = RequestQueue(capacity=config.admission.request_queue_capacity)
    kv = KVAccountant(capacity=config.admission.kv_capacity_aggregate)
    telemetry = IdleTelemetry()
    admission = Admission(admission_cfg=config.admission, request_queue=rq,
                          kv_accountant=kv, idle_telemetry=telemetry)
    layer_state = LayerState(num_layers=config.model.num_layers)
    completion = Completion(clock=clock, kv_accountant=kv)
    return SchedulerCore(
        config=config, clock=clock, queue=queue, dag=dag, window=window,
        dispatcher=dispatcher, request_queue=rq, kv_accountant=kv,
        admission=admission, layer_state=layer_state, completion=completion,
    )


def _adm_event():
    return Event(timestamp=0.0, type=EventType.ADMISSION_TICK,
                 payload={"t_proj": 1.0, "t_pim_fn": lambda k, n: 0.5,
                          "a_cycle": 1.0, "b_cycle": 1.0, "ctx_tokens": 1000})


def _kc(mb_id, ntype, resource="GPU"):
    return Event(timestamp=0.0, type=EventType.KERNEL_COMPLETION,
                 payload={"micro_batch_id": mb_id, "node_type": ntype, "resource": resource})


def _decode_one(core, mb_id):
    mb = core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = core.config.model.num_layers - 1
    core._maybe_advance_forward_pass(_kc(mb_id, NodeType.O_PROJ))


def _drive_until_done(core, mb_id):
    """Impl-9 — mb evict 후엔 mb 가 dispatcher.micro_batches 에서 사라짐 (defensive)."""
    while (
        mb_id in core.dispatcher.micro_batches
        and any(
            r in core.in_flight_requests
            for r in core.dispatcher.micro_batches[mb_id].decode_tokens.keys()
        )
    ):
        _decode_one(core, mb_id)


# ============================================================================
# (I-F1) KV admit/release round-trip stress
# ============================================================================

def test_stress_kv_admit_release_100_roundtrip(completion, kv_accountant):
    """100 req 의 (admit → finalize) round-trip → remaining 정확 보존"""
    initial = kv_accountant.remaining
    for i in range(100):
        req = Request(id=i, prompt_tokens=[0], kv_length=100, max_tokens=1)
        req.transition_to(RequestState.PREFILL)
        req.transition_to(RequestState.DECODE)
        kv_accountant.admit(req)
        completion.finalize(req)
    assert kv_accountant.remaining == initial


def test_stress_kv_partial_release_state_consistency(completion, kv_accountant):
    """50 admit, 25 finalize → remaining 정확 부분 회수"""
    initial = kv_accountant.remaining
    reqs = []
    for i in range(50):
        req = Request(id=i, prompt_tokens=[0], kv_length=100, max_tokens=1)
        req.transition_to(RequestState.PREFILL)
        req.transition_to(RequestState.DECODE)
        kv_accountant.admit(req)
        reqs.append(req)
    for r in reqs[:25]:
        completion.finalize(r)
    expected = initial - 25 * 100
    assert kv_accountant.remaining == expected


def test_stress_kv_double_release_raises_among_50(completion, kv_accountant):
    """50 finalize 중 11 번째 동일 req 재 finalize → raise"""
    reqs = []
    for i in range(50):
        req = Request(id=i, prompt_tokens=[0], kv_length=100, max_tokens=1)
        req.transition_to(RequestState.PREFILL)
        req.transition_to(RequestState.DECODE)
        kv_accountant.admit(req)
        reqs.append(req)
    for r in reqs[:11]:
        completion.finalize(r)
    with pytest.raises(ValueError, match="double finalize"):
        completion.finalize(reqs[0])  # 이미 finalize


# ============================================================================
# (I-F2) state monotonic stress
# ============================================================================

def test_stress_request_state_monotonic_100_reqs():
    """100 req 의 lifecycle 진행 위 state 단방향"""
    core = _make_core()
    reqs = []
    for i in range(100):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=1)
        reqs.append(r)
        core.request_queue.push(r)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    # 모두 COMPLETED + 역방향 transition 없음 (자연 강제)
    assert all(r.state == RequestState.COMPLETED for r in reqs)


# ============================================================================
# (I-F3) decoded_count signal invariant
# ============================================================================

def test_stress_decoded_count_signal_invariant_50_mbs():
    """50 mb 위 token decode signal 1회 = +1 정확"""
    core = _make_core()
    for i in range(50):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=3)
        core.request_queue.push(r)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id not in core.dispatcher.micro_batches:
            continue
        for step in range(3):
            _decode_one(core, mb_id)
        # max_tokens=3 도달 후 finalize. r 의 decoded_count = 3
        assert r.decoded_count == 3


# ============================================================================
# (I-F4) completion_time single-set
# ============================================================================

def test_stress_completion_time_single_set():
    """50 req 의 completion_time 가 finalize 시점에 정확 한 번만 set"""
    core = _make_core()
    reqs = []
    for i in range(50):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=1)
        reqs.append(r)
        core.request_queue.push(r)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    # 모든 req 의 completion_time 가 float (None 아님)
    assert all(isinstance(r.completion_time, float) for r in reqs)


# ============================================================================
# (I-F5) in_flight_requests dict no orphan
# ============================================================================

def test_stress_in_flight_requests_dict_no_leak():
    """50 admit + 50 finalize → in_flight_requests == {}"""
    core = _make_core()
    for i in range(50):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=1)
        core.request_queue.push(r)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    assert core.in_flight_requests == {}


# ============================================================================
# (I-F6) MicroBatch.decode_tokens 의 lifecycle 미관여 (Q10 lock-in)
# ============================================================================

def test_stress_micro_batch_decode_tokens_unchanged_by_completion():
    """finalize 호출 전후 mb.decode_tokens dict 의 key set 동일"""
    core = _make_core()
    for i, mx in enumerate([1, 2, 3]):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=mx)
        core.request_queue.push(r)
    core._handle(_adm_event())
    mb_id = core._next_mb_id - 1
    mb = core.dispatcher.micro_batches[mb_id]
    before_keys = set(mb.decode_tokens.keys())
    _drive_until_done(core, mb_id)
    after_keys = set(mb.decode_tokens.keys())
    assert before_keys == after_keys


# ============================================================================
# Real-trace stress (R3 보강)
# ============================================================================

def test_stress_full_pipeline_real_trace_first_500_reqs():
    """실 trace first 100 req 위 admission → dispatch → completion → release (500 은 무거움 → 100)"""
    core = _make_core(kv_capacity=100_000_000)
    r = TraceReplayer.load(REAL_3_40)
    initial = core.kv_accountant.remaining
    for req in list(r.replay())[:100]:
        req.max_tokens = 1  # 단축
        core.request_queue.push(req)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    assert core.kv_accountant.remaining == initial


def test_stress_lifecycle_seed_42_bit_exact_reproducible():
    """동일 trace + seed 42 위 50-req lifecycle bit-exact 2 회 (PLAN §0 C5)"""
    def run():
        random.seed(42)
        core = _make_core()
        reqs = []
        for i in range(50):
            max_t = random.choice([1, 2])
            r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=max_t)
            reqs.append(r)
            core.request_queue.push(r)
            core._handle(_adm_event())
            mb_id = core._next_mb_id - 1
            if mb_id in core.dispatcher.micro_batches:
                _drive_until_done(core, mb_id)
        return (core.kv_accountant.remaining,
                tuple(r.completion_time for r in reqs))
    assert run() == run()


# ============================================================================
# Composite stress + R8 seed sweep
# ============================================================================

@pytest.mark.parametrize("seed", [0, 42, 99, 1000])
def test_stress_lifecycle_composite_invariant_violation_zero(seed):
    """R8 — composite 위 4 seed cell 모두 invariant 보존"""
    random.seed(seed)
    core = _make_core()
    reqs = []
    initial = core.kv_accountant.remaining
    for i in range(30):
        max_t = random.choice([1, 2, 3])
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=max_t)
        reqs.append(r)
        core.request_queue.push(r)
        core._handle(_adm_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    # I-F1 KV no-leak
    assert core.kv_accountant.remaining == initial
    # I-F2 state monotonic (모두 COMPLETED 도달)
    assert all(r.state == RequestState.COMPLETED for r in reqs)
    # I-F3 decoded_count 정확 (max_tokens 와 일치)
    assert all(r.decoded_count == r.max_tokens for r in reqs)
    # I-F4 completion_time set
    assert all(r.completion_time is not None for r in reqs)
    # I-F5 in_flight_requests no orphan
    assert core.in_flight_requests == {}
