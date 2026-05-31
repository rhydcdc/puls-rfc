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

    def test_arrival_pushes_admission_tick(self, scheduler_core):
        """STEP 2.5 — ADMISSION_TICK 자체는 self-push 안 함. 재기동은 REQUEST_ARRIVAL.

        고정 타이머 self-push 폐기 (REPORT §10) → ADMISSION_TICK 단독 처리 후 다음 tick
        없음. 새 요청 도착(REQUEST_ARRIVAL) 이 admission 을 재기동.
        """
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0, arrival=0.0)
        scheduler_core.queue.push(Event(
            timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={"request": req},
        ))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) >= 1

    def test_admission_tick_no_self_push(self, scheduler_core):
        """STEP 2.5 — ADMISSION_TICK 처리는 다음 tick 을 self-push 하지 않는다.

        옛 타이머 모델(처리 후 +interval 에 self-push) 폐기. queue 에 req 있어도
        ADMISSION_TICK 단독으론 새 tick 안 생김 (헛도는 tick 제거 핵심).
        """
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        # admit 성공 후 dispatcher.tick → 그 자체로는 self-push 없음. admit 된 mb 의
        # KERNEL_COMPLETION 이 다음 재기동 경로 (이 단위 test 범위 밖).
        assert _admission_tick_count(scheduler_core) == 0

    def test_arrival_tick_payload_freshly_composed(self, scheduler_core):
        """REQUEST_ARRIVAL 재기동 tick payload = _compose_admission_payload 의 fresh 산출.

        STEP 2.5 — 재기동 경로(`_schedule_admission_tick_with_default_payload`)가 scheduler
        자기 상태 측정값 (6 key) 진정 주입.
        """
        scheduler_core.enable_admission_tick_rescheduling = True
        req = _make_request(req_id=0)
        scheduler_core.queue.push(Event(
            timestamp=0.0, type=EventType.REQUEST_ARRIVAL, payload={"request": req},
        ))
        scheduler_core.step()
        ticks = _admission_ticks(scheduler_core)
        assert len(ticks) >= 1
        assert set(ticks[0].payload.keys()) == {
            "t_proj", "t_pim_fn", "a_cycle", "b_cycle", "ctx_tokens",
            "gpu_op_time_per_token_us",
        }
        # cadence — 재기동 tick 은 now + tick_interval_us 에 push
        assert ticks[0].timestamp == pytest.approx(
            scheduler_core.config.admission.tick_interval_us
        )

    def test_default_flag_off_no_self_push(self, scheduler_core):
        """Default (flag=False) — 기존 isolated test 영역 보존 (R14)."""
        req = _make_request(req_id=0)
        scheduler_core.request_queue.push(req)
        scheduler_core.queue.push(_admission_event(timestamp=0.0))
        scheduler_core.step()
        assert _admission_tick_count(scheduler_core) == 0
