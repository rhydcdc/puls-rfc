"""STEP 3-a — prefill 합류 (진행 중 mb 에 큐의 신규 요청 backfill).

배치_생애 §4 합류 + §5 게이트. `_try_join_prefill`:
- 게이트: gpu_idle > idle_theta_high (GPU 빈자리 있을 때만; PIM-bound 구간)
- 가능량: min(seq 여유, KV 여유), FIFO
- KV admit + in_flight 등록 + PREFILL 전이
"""
from puls_sched.request import Request, RequestState


def _set_gpu_idle(tel, frac):
    """idle_telemetry 의 gpu_instance_a idle fraction 을 frac 으로 설정.

    gpu_idle = 1 − gpu_active/span. gpu_active=10, span 을 pim active 로 확장 →
    gpu_idle = 1 − 10/span = frac.
    """
    tel.reset(0.0)
    if frac >= 1.0:
        return
    active = 10.0
    span = active / (1.0 - frac)
    tel.record_active("gpu_instance_a", 0.0, active)
    tel.record_active("pim_instance_a", 0.0, span)


class TestPrefillJoin:
    def test_join_fires_when_gpu_idle(self, scheduler_core):
        """GPU idle > theta_high → 큐의 신규 요청이 mb 에 prefill 합류."""
        rq = scheduler_core.request_queue
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join_prefill({99})  # 기존 active 1개 가정
        assert len(joined) == 5
        assert len(rq) == 0
        for rid in joined:
            assert rid in scheduler_core.in_flight_requests
            assert scheduler_core.in_flight_requests[rid].state == RequestState.PREFILL
        assert scheduler_core.kv_accountant.used == 5000

    def test_join_gated_off_when_gpu_saturated(self, scheduler_core):
        """GPU idle <= theta_high (GPU 포화) → 합류 안 함 (게이트 닫힘)."""
        rq = scheduler_core.request_queue
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.05)
        joined = scheduler_core._try_join_prefill({99})
        assert joined == set()
        assert len(rq) == 5
        assert scheduler_core.kv_accountant.used == 0

    def test_join_bounded_by_seq_room(self, scheduler_core):
        """seq 여유만큼만 합류 (max_batch_size − 현재 active)."""
        rq = scheduler_core.request_queue
        for i in range(10):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        cap = scheduler_core.config.admission.max_batch_size
        active = set(range(1000, 1000 + cap - 2))  # seq_room = 2
        joined = scheduler_core._try_join_prefill(active)
        assert len(joined) == 2
        assert len(rq) == 8

    def test_join_bounded_by_kv(self, scheduler_core):
        """KV 여유 부족 시 head 에서 중단 (가능량 = min(seq, KV))."""
        rq = scheduler_core.request_queue
        cap = scheduler_core.kv_accountant.capacity
        rq.push(Request(id=0, prompt_tokens=[0] * 10, kv_length=cap, max_tokens=10))   # 캐파 전부
        rq.push(Request(id=1, prompt_tokens=[0] * 10, kv_length=1000, max_tokens=10))  # 안 들어감
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join_prefill({99})
        assert joined == {0}
        assert len(rq) == 1  # head-of-line, 다음 완료 경계에서 재시도

    def test_join_empty_queue_noop(self, scheduler_core):
        """큐 비면 합류 0 (no-op)."""
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        assert scheduler_core._try_join_prefill({99}) == set()
