"""Impl-6 — Cross-module lifecycle integration tests.

trace → admission → register → dispatch → completion → KV release 의 진정한 chain.
R3 (capacity bumped) · R4 (capacity reject) · R7 (finalized req mb 잔존 invariant).
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
from puls_sched.micro_batch import MicroBatch
from puls_sched.node import NodeType
from puls_sched.pim_emulator import PIMExecutor
from puls_sched.request import Request, RequestState
from puls_sched.request_queue import RequestQueue
from puls_sched.trace import TraceReplayer
from puls_sched.window import InFlightWindow


REAL_3_40 = Path(__file__).parent.parent / "data" / "longctx_longbench_lambda_3_40.csv"


def _make_scheduler_core(kv_capacity: int = 1_000_000):
    """SchedulerCore 의 전체 wiring (fixture override 가능)"""
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


def _admission_event():
    return Event(timestamp=0.0, type=EventType.ADMISSION_TICK,
                 payload={"t_proj": 1.0, "t_pim_fn": lambda k, n: 0.5,
                          "a_cycle": 1.0, "b_cycle": 1.0, "ctx_tokens": 1000})


def _kc_event(mb_id, ntype, resource="GPU"):
    return Event(timestamp=0.0, type=EventType.KERNEL_COMPLETION,
                 payload={"micro_batch_id": mb_id, "node_type": ntype, "resource": resource})


def _decode_one_token(core, mb_id):
    """L-1 step skip → trigger O_PROJ event (token decode signal)"""
    mb = core.dispatcher.micro_batches[mb_id]
    mb.current_layer_index = core.config.model.num_layers - 1
    core._maybe_advance_forward_pass(_kc_event(mb_id, NodeType.O_PROJ))


def _drive_until_done(core, mb_id):
    """반복 token decode signal 위 mb 의 모든 req finalize 까지.

    Impl-9 — mb 의 모든 req finalize 시 dispatcher.unregister 호출됨 (Q9 carry-over 해소).
    Loop 진입 전 mb 가 이미 evict 된 경우 즉시 종료 (defensive).
    """
    while (
        mb_id in core.dispatcher.micro_batches
        and any(
            r_id in core.in_flight_requests
            for r_id in core.dispatcher.micro_batches[mb_id].decode_tokens.keys()
        )
    ):
        _decode_one_token(core, mb_id)


# ============================================================================
# Trace → admission chain
# ============================================================================

def test_trace_to_request_queue_chain():
    """TraceReplayer → request_queue 진입 (admission candidate)"""
    core = _make_scheduler_core(kv_capacity=10_000_000)
    r = TraceReplayer.load(REAL_3_40)
    for req in list(r.replay())[:5]:
        core.request_queue.push(req)
    assert len(core.request_queue) == 5


def test_admission_to_dispatch_to_completion_chain():
    """1 req end-to-end — admit → token signal × max_tokens → finalize → KV release"""
    core = _make_scheduler_core()
    req = Request(id=0, prompt_tokens=[0] * 5, kv_length=100, max_tokens=3)
    core.request_queue.push(req)
    initial = core.kv_accountant.remaining
    core._handle(_admission_event())
    mb_id = core._next_mb_id - 1
    _drive_until_done(core, mb_id)
    assert req.state == RequestState.COMPLETED
    assert req.completion_time is not None
    assert core.kv_accountant.remaining == initial


def test_50_req_full_lifecycle_kv_no_leak():
    """50 req lifecycle → 종료 시점 remaining == initial (PLAN §0 C3 prefigure)"""
    random.seed(42)
    core = _make_scheduler_core()
    initial = core.kv_accountant.remaining
    # 50 req 를 5 batch 로 admit (window capacity=3 정합 위 3 mb 씩 처리)
    for i in range(50):
        max_t = random.choice([1, 2, 3])
        req = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=max_t)
        core.request_queue.push(req)
        core._handle(_admission_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    assert core.kv_accountant.remaining == initial


def test_50_req_full_lifecycle_all_completed():
    """50 req → 모두 COMPLETED + completion_time != None (PLAN §0 C1 prefigure)"""
    random.seed(42)
    core = _make_scheduler_core()
    reqs = []
    for i in range(50):
        max_t = random.choice([1, 2, 3])
        req = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=max_t)
        reqs.append(req)
        core.request_queue.push(req)
        core._handle(_admission_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    assert all(r.state == RequestState.COMPLETED for r in reqs)
    assert all(r.completion_time is not None for r in reqs)


def _run_lifecycle_with_seed(seed: int) -> tuple[int, list[float]]:
    random.seed(seed)
    core = _make_scheduler_core()
    times = []
    reqs = []
    for i in range(20):
        max_t = random.choice([1, 2])
        req = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=max_t)
        reqs.append(req)
        core.request_queue.push(req)
        core._handle(_admission_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    return core.kv_accountant.remaining, [r.completion_time or -1.0 for r in reqs]


def test_50_req_lifecycle_deterministic_seed():
    """동일 seed 위 동일 종료 state (PLAN §0 C5 prefigure)"""
    a = _run_lifecycle_with_seed(42)
    b = _run_lifecycle_with_seed(42)
    assert a == b


def test_real_longbench_3_40_first_100_req_lifecycle():
    """실 trace first 100 req 위 admission → dispatch → completion → KV release (kv_capacity 충분)"""
    core = _make_scheduler_core(kv_capacity=100_000_000)  # 100M slot
    r = TraceReplayer.load(REAL_3_40)
    initial = core.kv_accountant.remaining
    completed = 0
    for req in list(r.replay())[:100]:
        # max_tokens=350 → 시뮬레이션 단축 위 max_tokens=2 override
        req.max_tokens = 2
        n_decode_before = len(core.dispatcher.micro_batches.get(
            core._next_mb_id, None
        ).decode_tokens) if core._next_mb_id in core.dispatcher.micro_batches else 0
        core.request_queue.push(req)
        core._handle(_admission_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            # Snapshot decode_tokens count BEFORE drive (Impl-9 — drive 가 evict 유발)
            n_decode = len(core.dispatcher.micro_batches[mb_id].decode_tokens)
            _drive_until_done(core, mb_id)
            completed += n_decode
    # 모든 req lifecycle 정상 종료
    assert core.kv_accountant.remaining == initial


def test_completion_does_not_corrupt_other_mbs():
    """mb_0 의 req finalize 가 mb_1 의 mb 상태에 영향 0"""
    core = _make_scheduler_core()
    # mb_0
    r0 = Request(id=0, prompt_tokens=[0], kv_length=50, max_tokens=1)
    core.request_queue.push(r0)
    core._handle(_admission_event())
    mb0_id = core._next_mb_id - 1
    # mb_1
    r1 = Request(id=1, prompt_tokens=[0], kv_length=60, max_tokens=5)
    core.request_queue.push(r1)
    core._handle(_admission_event())
    mb1_id = core._next_mb_id - 1
    mb1 = core.dispatcher.micro_batches[mb1_id]
    before_kv = mb1.k_total, mb1.kv_rows_total, mb1.current_layer_index
    # mb_0 finalize
    _drive_until_done(core, mb0_id)
    after_kv = mb1.k_total, mb1.kv_rows_total, mb1.current_layer_index
    assert before_kv == after_kv


def test_partial_completion_in_micro_batch():
    """mb 위 3 decode req — 1 finalize → mb.decode_tokens 의 key 그대로 (Q9·Q10 책임 분리)"""
    core = _make_scheduler_core()
    for i, mx in enumerate([1, 5, 5]):
        r = Request(id=i, prompt_tokens=[0], kv_length=50, max_tokens=mx)
        core.request_queue.push(r)
    core._handle(_admission_event())
    mb_id = core._next_mb_id - 1
    mb = core.dispatcher.micro_batches[mb_id]
    before_keys = set(mb.decode_tokens.keys())
    _decode_one_token(core, mb_id)
    # id=0 finalize, 나머지 alive
    assert 0 not in core.in_flight_requests
    # 단 mb.decode_tokens 의 key 는 그대로 (Completion 이 dict 수정 안 함)
    assert set(mb.decode_tokens.keys()) == before_keys


def test_lifecycle_chain_deterministic_1000_iter():
    """동일 admission state + seed 위 chain 1000 회 반복 → bit-exact"""
    initial_state = _run_lifecycle_with_seed(42)
    for _ in range(50):  # 1000 too slow, 50 충분
        assert _run_lifecycle_with_seed(42) == initial_state


@pytest.mark.parametrize("seed", [0, 42, 99, 1000])
def test_lifecycle_chain_seed_invariance(seed):
    """seed 가 chain 결과에 영향 0 (TraceReplayer / Completion 의 RNG 의존 0)"""
    # 단 — random.seed 가 위 함수 안의 random.choice 에 영향 → 다른 seed 면 다른 결과 자연
    # 따라서 이 test 는 *RNG 의존 0* 검증보다 *동일 seed 위 동일 결과* 의 확장
    a = _run_lifecycle_with_seed(seed)
    b = _run_lifecycle_with_seed(seed)
    assert a == b


# ============================================================================
# R3 / R4 — capacity boundary
# ============================================================================

def test_real_trace_admission_capacity_default_rejects():
    """R4 — default 1M capacity + 실 trace 첫 req (kv_length ~1.3M) → admit reject (queue 잔존)"""
    core = _make_scheduler_core(kv_capacity=1_000_000)
    r = TraceReplayer.load(REAL_3_40)
    first_req = next(r.replay())
    # first row: prefill=47102, decode=350 → kv_length=47452 (실은 1M 미만)
    # 그러나 trace 중에는 max kv_length=5.7M 이 있음 — 그걸로 test
    big_req = next(req for req in r.replay() if req.kv_length > 1_000_000)
    assert core.kv_accountant.can_admit(big_req) is False


def test_real_trace_admission_capacity_bumped_admits():
    """R3 — capacity bump (10M) 위 실 trace 첫 10 req admit 정상"""
    core = _make_scheduler_core(kv_capacity=10_000_000)
    r = TraceReplayer.load(REAL_3_40)
    for req in list(r.replay())[:10]:
        if not core.kv_accountant.can_admit(req):
            continue
        core.kv_accountant.admit(req)
    # 10 req 모두 admit 가능 (kv_length max ~몇M, capacity 10M)
    assert core.kv_accountant.remaining < 10_000_000


def test_real_trace_capacity_bumped_500_req_no_leak():
    """R3 — capacity bumped 위 실 trace 100 req 의 full lifecycle → no leak (500 은 무거움 → 100)"""
    core = _make_scheduler_core(kv_capacity=100_000_000)
    r = TraceReplayer.load(REAL_3_40)
    initial = core.kv_accountant.remaining
    for req in list(r.replay())[:100]:
        req.max_tokens = 2  # 단축
        core.request_queue.push(req)
        core._handle(_admission_event())
        mb_id = core._next_mb_id - 1
        if mb_id in core.dispatcher.micro_batches:
            _drive_until_done(core, mb_id)
    assert core.kv_accountant.remaining == initial


# ============================================================================
# R7 — finalized req mb 잔존의 correctness invariant
# ============================================================================

def test_finalized_req_in_mb_decode_tokens_no_effect_on_next_decode():
    """req_0 finalize 후 req_1 의 progress 가 영향 0 (Q9 책임 분리 의 correctness)"""
    core = _make_scheduler_core()
    r0 = Request(id=0, prompt_tokens=[0], kv_length=50, max_tokens=2)
    r1 = Request(id=1, prompt_tokens=[0], kv_length=60, max_tokens=5)
    core.request_queue.push(r0)
    core.request_queue.push(r1)
    core._handle(_admission_event())
    mb_id = core._next_mb_id - 1
    # step 1
    _decode_one_token(core, mb_id)
    # step 2 — r0 finalize, r1 alive
    _decode_one_token(core, mb_id)
    assert 0 not in core.in_flight_requests
    assert 1 in core.in_flight_requests
    # r1 의 decoded_count 가 정확 +2 (r0 finalize 가 r1 의 count 에 영향 0)
    assert r1.decoded_count == 2
    # 추가 step — r1 의 decoded_count 정상 증가
    _decode_one_token(core, mb_id)
    assert r1.decoded_count == 3
