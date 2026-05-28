"""Cluster R — InstancePipeline.dispatch() (O5.1).

Impl-10-pre-1 정합. ARCH §3.4 (Pipeline Structure) + §5.2 (Fixed-shape Handoff) +
§5.6 (Double-Buffering) 위 단일 layer chain wiring substrate.

ForwardPass.run() 의 매 layer 마다 dispatch 호출 (Q6 (a)) — L=80 회 반복.

`clock` · `idle_telemetry` 가 None 이면 record 단계 skip (backward-compat). Optional 의존.
"""

import pytest

from puls_sched.clock import Clock
from puls_sched.config import default_dummy_config
from puls_sched.forward_pass import ForwardPass, LayerState
from puls_sched.idle_telemetry import IdleTelemetry
from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.micro_batch import MicroBatch
from puls_sched.nvlink import NVLinkTransfer


# ---- dispatch() method 존재 + signature ----

def test_dispatch_method_exists(instance_pipeline):
    """InstancePipeline.dispatch method 신설 — O5.1 substrate."""
    assert hasattr(instance_pipeline, "dispatch")


def test_dispatch_signature():
    """dispatch(self, mb) — 단순 substrate, layer_idx arg 부재 (caller 가 mb.current_layer_index 보유)."""
    import inspect
    params = list(inspect.signature(InstancePipeline.dispatch).parameters.keys())
    assert params == ["self", "mb"]


# ---- Backward-compat (clock/idle_telemetry=None) ----

def test_dispatch_without_telemetry_works(instance_pipeline):
    """clock · idle_telemetry None 위 dispatch 호출 — fixed-shape 검증만, record skip."""
    mb = MicroBatch(id=0, decode_tokens={0: 0, 1: 0})  # 2 decode req
    instance_pipeline.dispatch(mb)  # raises 0


# ---- Fixed-shape validation ----

def test_dispatch_validates_handoff_shape_pure_decode(instance_pipeline):
    """Pure-decode mb → handoff shape (B, hidden) 정합 자연 통과."""
    mb = MicroBatch(id=0, decode_tokens={i: 0 for i in range(4)})
    instance_pipeline.dispatch(mb)


def test_dispatch_validates_handoff_shape_pure_prefill(instance_pipeline):
    """Pure-prefill uniform-chunk → handoff shape (P*chunk, hidden) 정합."""
    mb = MicroBatch(id=0, prefill_chunk={0: [1, 2, 3], 1: [4, 5, 6]})  # uniform chunk=3
    instance_pipeline.dispatch(mb)


def test_dispatch_validates_handoff_shape_mixed(instance_pipeline):
    """Mixed mb → handoff shape (B_decode + P*chunk, hidden) 정합."""
    mb = MicroBatch(
        id=0,
        prefill_chunk={0: [1, 2], 1: [3, 4]},   # uniform chunk=2
        decode_tokens={2: 0, 3: 0},              # 2 decode
    )
    instance_pipeline.dispatch(mb)


def test_dispatch_raises_on_ragged_prefill_chunk(instance_pipeline):
    """Ragged prefill chunk → AssertionError (ARCH §5.2 violation)."""
    mb = MicroBatch(id=0, prefill_chunk={0: [1, 2], 1: [3, 4, 5]})  # ragged
    with pytest.raises(AssertionError, match="ragged"):
        instance_pipeline.dispatch(mb)


# ---- Instance B GPU activity recording (with telemetry) ----

def _make_pipeline_with_telemetry(cfg, clock, tel):
    a = Instance(name="A", has_pim=True)
    b = Instance(name="B", has_pim=False)
    nvlink = NVLinkTransfer(config=cfg)
    return InstancePipeline(
        config=cfg, instance_a=a, instance_b=b, nvlink=nvlink,
        clock=clock, idle_telemetry=tel,
    )


@pytest.mark.skip(
    reason="Impl-10-pre-2 post-fix — Stage 1 placeholder substrate (NVLink handoff time → gpu_instance_b) "
           "폐기 위 window_end 오염 제거. Stage 2 calibration 위 실 Instance B FFN op_time substrate "
           "도입 시점 재활성 (gpu_instance_b 의 진정 active duration 측정)."
)
def test_dispatch_records_gpu_instance_b(dummy_config):
    """dispatch 호출 시 gpu_instance_b slot 에 activity 누적 (O8.1 substrate)."""
    clock = Clock()
    tel = IdleTelemetry()
    tel.reset(0.0)
    p = _make_pipeline_with_telemetry(dummy_config, clock, tel)
    mb = MicroBatch(id=0, decode_tokens={i: 0 for i in range(4)})
    p.dispatch(mb)
    assert tel._active_duration["gpu_instance_b"] > 0


def test_dispatch_does_not_record_other_slots(dummy_config):
    """dispatch 가 gpu_instance_a · pim_instance_a 영향 0 (slot 분리 lock-in)."""
    clock = Clock()
    tel = IdleTelemetry()
    tel.reset(0.0)
    p = _make_pipeline_with_telemetry(dummy_config, clock, tel)
    mb = MicroBatch(id=0, decode_tokens={0: 0})
    p.dispatch(mb)
    assert tel._active_duration["gpu_instance_a"] == 0.0
    assert tel._active_duration["pim_instance_a"] == 0.0


# ---- Multi-layer (L=80) chain via ForwardPass.run ----

def test_forward_pass_run_calls_dispatch_per_layer(dummy_config):
    """ForwardPass.run() 의 매 layer 마다 instance_pipeline.dispatch 호출 (Q6 a 결정).
    L=80 회 반복.

    Impl-10-pre-2 post-fix — gpu_instance_b active_duration 검증 제거 (placeholder 폐기).
    Dispatch 호출 횟수 검증 위 fp.run 의 count + mb.current_layer_index 만 유지.
    """
    clock = Clock()
    tel = IdleTelemetry()
    tel.reset(0.0)
    p = _make_pipeline_with_telemetry(dummy_config, clock, tel)
    layer_state = LayerState(num_layers=dummy_config.model.num_layers)
    fp = ForwardPass(config=dummy_config, instance_pipeline=p, layer_state=layer_state)
    mb = MicroBatch(id=0, decode_tokens={0: 0, 1: 0})
    count = fp.run(mb)
    assert count == dummy_config.model.num_layers
    assert mb.current_layer_index == dummy_config.model.num_layers


def test_forward_pass_run_with_layer_count_parametrize(dummy_config):
    """L=1, 8, 32, 80 parametrize — 각 L 위 dispatch 호출 횟수 정합."""
    import dataclasses as dc
    for L in (1, 8, 32, 80):
        cfg = dc.replace(dummy_config, model=dc.replace(dummy_config.model, num_layers=L))
        clock = Clock()
        tel = IdleTelemetry()
        tel.reset(0.0)
        p = _make_pipeline_with_telemetry(cfg, clock, tel)
        ls = LayerState(num_layers=L)
        fp = ForwardPass(config=cfg, instance_pipeline=p, layer_state=ls)
        mb = MicroBatch(id=0, decode_tokens={0: 0})
        count = fp.run(mb)
        assert count == L


# ---- Determinism ----

def test_dispatch_deterministic(dummy_config):
    """동일 mb · 동일 clock → 동일 record (100-iter bit-exact)."""
    results = []
    for _ in range(100):
        clock = Clock()
        tel = IdleTelemetry()
        tel.reset(0.0)
        p = _make_pipeline_with_telemetry(dummy_config, clock, tel)
        mb = MicroBatch(id=0, decode_tokens={i: 0 for i in range(4)})
        p.dispatch(mb)
        results.append(tel.gpu_instance_b_idle_fraction())
    assert len(set(results)) == 1


def test_dispatch_no_event_push(dummy_config):
    """NVLink event push 0 — InstancePipeline.dispatch 은 *측정만*, queue 미 touch (Q7 정합)."""
    clock = Clock()
    tel = IdleTelemetry()
    tel.reset(0.0)
    p = _make_pipeline_with_telemetry(dummy_config, clock, tel)
    # Pipeline 자체에 queue dep 없음 — 미 dispatch
    mb = MicroBatch(id=0, decode_tokens={0: 0})
    p.dispatch(mb)
    # nothing to assert about queue — pipeline doesn't have one. confirmed by missing field
    import dataclasses as dc
    fields = set(InstancePipeline.__dataclass_fields__.keys())
    assert "queue" not in fields
    assert "event_queue" not in fields
