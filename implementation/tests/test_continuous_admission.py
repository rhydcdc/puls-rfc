"""Cluster K — ADMISSION_PASS self-rescheduling (Impl-9 Q1).

ARCH §6.4 'per-iteration basis' 정합. Idle termination guard 검증.
"""

import pytest

from puls_sched.event import Event, EventType
from puls_sched.request import Request, RequestState


def _admission_event(timestamp: float = 0.0) -> Event:
    return Event(
        timestamp=timestamp,
        type=EventType.ADMISSION_PASS,
        payload={
            "t_proj": 0.0, "t_pim_fn": lambda n: 0.0,
            "a_cycle": 0.0, "b_cycle": 0.0, "ctx_tokens": 0,
        },
    )


def _make_request(req_id: int, arrival: float = 0.0) -> Request:
    return Request(
        id=req_id,
        prompt_len=256,
        kv_length=256 + 32,
        arrival_time=arrival,
        max_tokens=32,
    )


def _admission_pass_count(scheduler) -> int:
    return sum(
        1 for entry in scheduler.queue._heap
        if entry[2].type == EventType.ADMISSION_PASS
    )


def _admission_passes(scheduler) -> list[Event]:
    return [entry[2] for entry in scheduler.queue._heap
            if entry[2].type == EventType.ADMISSION_PASS]


class TestSelfRescheduling:
    def test_idle_termination_no_next_tick(self, scheduler_core):
        """request_queue + in_flight 동시 empty → 다음 tick push 안 함."""
        scheduler_core.continuous_admission = True
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_pass_count(scheduler_core) == 0

    def test_arrival_pushes_admission_pass(self, scheduler_core):
        """STEP 2.5 — ADMISSION_PASS 자체는 self-push 안 함. 재기동은 REQUEST_ARRIVAL.

        고정 타이머 self-push 폐기 (REPORT §10) → ADMISSION_PASS 단독 처리 후 다음 tick
        없음. 새 요청 도착(REQUEST_ARRIVAL) 이 admission 을 재기동.
        """
        scheduler_core.continuous_admission = True
        req = _make_request(req_id=0, arrival=0.0)
        scheduler_core.queue.push(Event(
            timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={"request": req},
        ))
        scheduler_core.step()
        assert _admission_pass_count(scheduler_core) >= 1

    def test_admission_pass_no_self_push(self, scheduler_core):
        """STEP 2.5 — ADMISSION_PASS 처리는 다음 tick 을 self-push 하지 않는다.

        옛 타이머 모델(처리 후 +interval 에 self-push) 폐기. queue 에 req 있어도
        ADMISSION_PASS 단독으론 새 tick 안 생김 (헛도는 tick 제거 핵심).
        """
        scheduler_core.continuous_admission = True
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        # admit 성공 후 dispatcher.tick → 그 자체로는 self-push 없음. admit 된 mb 의
        # KERNEL_COMPLETION 이 다음 재기동 경로 (이 단위 test 범위 밖).
        assert _admission_pass_count(scheduler_core) == 0

    def test_arrival_tick_payload_trivial(self, scheduler_core):
        """REQUEST_ARRIVAL 재기동 tick payload = trivial(빈 dict).

        Phase-2 S2 (§2.5) — 동작점 고정으로 cycle 측정 payload(`_compose_admission_payload`)
        삭제. former 는 KV 합·prefill 512 만 보므로 payload 가 비어 있음. 재기동 cadence·
        push 자체는 유지(빈 슬롯 재충전 트리거, §2.2.1).
        """
        scheduler_core.continuous_admission = True
        req = _make_request(req_id=0)
        scheduler_core.queue.push(Event(
            timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={"request": req},
        ))
        scheduler_core.step()
        ticks = _admission_passes(scheduler_core)
        assert len(ticks) >= 1
        assert ticks[0].payload == {}
        # cadence — 재기동 pass 는 트리거와 동일 시각(now)에 즉시 push (지연 0)
        assert ticks[0].timestamp == pytest.approx(0.0)

    def test_default_flag_off_no_self_push(self, scheduler_core):
        """Default (flag=False) — 기존 isolated test 영역 보존 (R14)."""
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_pass_count(scheduler_core) == 0
