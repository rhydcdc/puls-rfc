"""Cluster K — ADMISSION_TICK self-rescheduling (Impl-9 Q1).

ARCH §6.4 'per-iteration basis' 정합. Idle termination guard 검증.
"""

import pytest

from puls_sched.event import Event, EventType
from puls_sched.request import Request, RequestState


def _admission_event(timestamp: float = 0.0) -> Event:
    return Event(
        timestamp=timestamp,
        type=EventType.ADMISSION_TICK,
        payload={
            "t_proj": 0.0, "t_pim_fn": lambda n: 0.0,
            "a_cycle": 0.0, "b_cycle": 0.0, "ctx_tokens": 0,
        },
    )


def _make_request(req_id: int, arrival: float = 0.0) -> Request:
    return Request(
        id=req_id,
        prompt_tokens=[0] * 256,
        kv_length=256 + 32,
        arrival_time=arrival,
        max_tokens=32,
    )


def _admission_tick_count(scheduler) -> int:
    return sum(
        1 for entry in scheduler.queue._heap
        if entry[2].type == EventType.ADMISSION_TICK
    )


def _admission_ticks(scheduler) -> list[Event]:
    return [entry[2] for entry in scheduler.queue._heap
            if entry[2].type == EventType.ADMISSION_TICK]


class TestSelfRescheduling:
    def test_idle_termination_no_next_tick(self, scheduler_core):
        """request_queue + in_flight 동시 empty → 다음 tick push 안 함."""
        scheduler_core.enable_admission_tick_rescheduling = True
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) == 0

    def test_non_idle_pushes_next_tick(self, scheduler_core):
        """request_queue 에 req 있으면 다음 tick auto-push."""
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0, arrival=0.0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) >= 1

    def test_next_tick_cadence(self, scheduler_core, dummy_config):
        """연속 2 회 ADMISSION_TICK 의 timestamp 차 = tick_interval_us."""
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        ticks = _admission_ticks(scheduler_core)
        assert len(ticks) >= 1
        assert ticks[0].timestamp == pytest.approx(
            dummy_config.admission.tick_interval_us
        )

    def test_next_tick_payload_freshly_composed(self, scheduler_core):
        """self-push 된 ADMISSION_TICK payload = scheduler._compose_admission_payload() 의 fresh 산출.

        Impl-10-pre-1 (B)~(B''') — prev payload identity propagation 영역 → fresh composition 영역.
        Production 위 scheduler 자기 상태 측정값 (a_cycle/b_cycle/t_proj/t_pim_fn/ctx_tokens) 진정 주입.
        """
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        orig = _admission_event(timestamp=0.0)
        scheduler_core.queue.push(orig)
        scheduler_core.step()
        ticks = _admission_ticks(scheduler_core)
        # Fresh composition — identity 다름 + 5 key 모두 존재 (composer schema)
        assert ticks[0].payload is not orig.payload
        assert set(ticks[0].payload.keys()) == {
            "t_proj", "t_pim_fn", "a_cycle", "b_cycle", "ctx_tokens",
        }

    def test_idle_guard_with_in_flight(self, scheduler_core):
        """in_flight_requests 잔재 시 self-push (queue 비어도)."""
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=99)
        req.transition_to(RequestState.PREFILL)
        scheduler_core.in_flight_requests[99] = req
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) >= 1

    def test_default_flag_off_no_self_push(self, scheduler_core):
        """Default (flag=False) — 기존 isolated test 영역 보존 (R14)."""
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) == 0
