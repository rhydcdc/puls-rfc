"""STEP 3 — 양방향 합류 (진행 중 mb 에 큐의 신규 요청 backfill).

배치_생애 §4 합류 + §5 게이트. `_try_join`:
- 게이트: gpu_idle > θ OR pim_idle > θ (어느 한쪽이라도 놀면 합류; 양쪽 포화면 닫힘)
- 가능량: min(seq 여유, KV 여유), FIFO
- KV admit + in_flight 등록 + (PENDING 이면) PREFILL 전이
- prefill/decode 분류는 _populate_mb_phases 가 prompt 유무로 자동 (끌어오기만)
"""
from puls_sched.request import Request, RequestState


def _set_gpu_idle(tel, frac):
    """GPU 만 idle=frac, PIM 포화(idle 0). prefill 합류 방향.

    gpu active=10, pim active=span(=10/(1−frac)) → window_end=span →
    gpu_idle = 1−10/span = frac, pim_idle = 1−span/span = 0.
    """
    tel.reset(0.0)
    if frac >= 1.0:
        return
    active = 10.0
    span = active / (1.0 - frac)
    tel.record_active("gpu_instance_a", 0.0, active)
    tel.record_active("pim_instance_a", 0.0, span)


def _set_pim_idle(tel, frac):
    """PIM 만 idle=frac, GPU 포화(idle 0). decode 합류 방향 (위 대칭)."""
    tel.reset(0.0)
    if frac >= 1.0:
        return
    active = 10.0
    span = active / (1.0 - frac)
    tel.record_active("pim_instance_a", 0.0, active)
    tel.record_active("gpu_instance_a", 0.0, span)


class TestPrefillJoin:
    def test_join_fires_when_gpu_idle(self, scheduler_core):
        """GPU idle > theta_high → 큐의 신규 요청이 mb 에 prefill 합류."""
        rq = scheduler_core.request_queue
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join({99})  # 기존 active 1개 가정
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
        joined = scheduler_core._try_join({99})
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
        joined = scheduler_core._try_join(active)
        assert len(joined) == 2
        assert len(rq) == 8

    def test_join_bounded_by_per_mb_kv(self, scheduler_core):
        """STEP 5.5 — 합류도 per-mb KV 예산(= KV캐파/2)까지만. 초과 head 에서 중단."""
        rq = scheduler_core.request_queue
        per_mb = scheduler_core._per_mb_kv_budget()   # 4M // 2 = 2,000,000
        # kv 500K × 5 → 4개(2.0M = 예산) 합류, 5번째(2.5M > 예산) 중단
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 10, kv_length=500_000, max_tokens=10))
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join({99})
        assert 4 * 500_000 <= per_mb < 5 * 500_000   # 예산 관계 sanity (2.0M)
        assert len(joined) == 4
        assert len(rq) == 1  # 5번째는 예산 초과 → 다음 mb 몫으로 보존

    def test_join_empty_queue_noop(self, scheduler_core):
        """큐 비면 합류 0 (no-op)."""
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.9)
        assert scheduler_core._try_join({99}) == set()


class TestDecodeJoin:
    """STEP 3-b — decode 방향 합류 (GPU 포화·PIM idle 구간)."""

    def test_join_fires_when_pim_idle(self, scheduler_core):
        """PIM idle > theta_high (GPU 포화) → 큐의 신규 요청 합류 (게이트 OR 의 PIM 쪽)."""
        rq = scheduler_core.request_queue
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        _set_pim_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join({99})
        assert len(joined) == 5
        assert len(rq) == 0
        assert scheduler_core.kv_accountant.used == 5000

    def test_join_gated_off_when_both_saturated(self, scheduler_core):
        """gpu·pim 둘 다 포화(idle ≤ θ) → 합류 안 함 (게이트 닫힘 = 밸런스 맞음)."""
        rq = scheduler_core.request_queue
        for i in range(5):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))
        tel = scheduler_core.admission.idle_telemetry
        # gpu active=pim active=span → 양쪽 idle 0
        tel.reset(0.0)
        tel.record_active("gpu_instance_a", 0.0, 100.0)
        tel.record_active("pim_instance_a", 0.0, 100.0)
        assert tel.gpu_idle_fraction() == 0.0 and tel.pim_idle_fraction() == 0.0
        joined = scheduler_core._try_join({99})
        assert joined == set()
        assert len(rq) == 5
        assert scheduler_core.kv_accountant.used == 0

    def test_decode_req_classified_to_decode_tokens(self, scheduler_core):
        """prompt 없는(decode 단계) 합류 req 는 _populate_mb_phases 가 decode_tokens 로 분류.

        합류 자체는 prefill/decode 무구분(끌어오기). 분류는 prompt 유무로 자동 —
        prompt=[] req 는 prefill 단계 없이 바로 decode_tokens 에 들어가 PIM 을 채운다.
        """
        rq = scheduler_core.request_queue
        for i in range(3):
            rq.push(Request(id=i, prompt_tokens=[], kv_length=500, max_tokens=10))
        _set_pim_idle(scheduler_core.admission.idle_telemetry, 0.9)
        joined = scheduler_core._try_join(set())
        assert joined == {0, 1, 2}
        # _populate_mb_phases 가 prompt 없는 req 를 decode_tokens 로 분류
        active_reqs = [scheduler_core.in_flight_requests[r] for r in joined]
        prefill_chunk, decode_tokens, _ = scheduler_core._populate_mb_phases(active_reqs, 512)
        assert set(decode_tokens.keys()) == {0, 1, 2}
        assert prefill_chunk == {}


class TestHysteresis:
    """STEP 5 — 합류 게이트 hysteresis deadband (θ_high=0.3, θ_low=0.05 default).

    신호 = max(gpu_idle, pim_idle). 닫힘→θ_high 초과 시 열림, 열림→θ_low 미만 시 닫힘,
    [θ_low, θ_high] 사이는 직전 상태 유지 (경계 진동 방지).
    """

    def _push(self, scheduler_core, k=5):
        rq = scheduler_core.request_queue
        for i in range(k):
            rq.push(Request(id=i, prompt_tokens=[0] * 1000, kv_length=1000, max_tokens=10))

    def test_opens_above_theta_high(self, scheduler_core):
        """닫힘 상태에서 신호 > θ_high → 열림 (합류 발동)."""
        self._push(scheduler_core)
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.4)  # > 0.3
        assert scheduler_core._join_gate_open is False
        joined = scheduler_core._try_join({99})
        assert len(joined) == 5
        assert scheduler_core._join_gate_open is True

    def test_stays_closed_in_deadband(self, scheduler_core):
        """닫힘 상태에서 신호가 deadband(0.05~0.3) 안 → 닫힘 유지 (합류 안 함)."""
        self._push(scheduler_core)
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.2)  # θ_low < 0.2 < θ_high
        assert scheduler_core._join_gate_open is False
        joined = scheduler_core._try_join({99})
        assert joined == set()
        assert scheduler_core._join_gate_open is False
        assert len(scheduler_core.request_queue) == 5

    def test_stays_open_in_deadband(self, scheduler_core):
        """열림 상태에서 신호가 deadband 안 → 열림 유지 (합류 계속)."""
        self._push(scheduler_core)
        scheduler_core._join_gate_open = True
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.2)
        joined = scheduler_core._try_join({99})
        assert len(joined) == 5
        assert scheduler_core._join_gate_open is True

    def test_closes_below_theta_low(self, scheduler_core):
        """열림 상태에서 신호 < θ_low → 닫힘 (합류 중단)."""
        self._push(scheduler_core)
        scheduler_core._join_gate_open = True
        _set_gpu_idle(scheduler_core.admission.idle_telemetry, 0.03)  # < 0.05
        joined = scheduler_core._try_join({99})
        assert joined == set()
        assert scheduler_core._join_gate_open is False
        assert len(scheduler_core.request_queue) == 5

    def test_pim_direction_opens(self, scheduler_core):
        """신호 = max(gpu, pim) — pim 만 idle 이어도 열림 (양방향 대칭)."""
        self._push(scheduler_core)
        _set_pim_idle(scheduler_core.admission.idle_telemetry, 0.4)
        joined = scheduler_core._try_join({99})
        assert len(joined) == 5
        assert scheduler_core._join_gate_open is True
