"""Impl-5 — ForwardPass + LayerState. ARCH §3.4 "Pass through L layers = L × cycle".

Q3 — forward_pass = L-loop owner. PLAN §0.5 reminder — L-iteration meta-count +
token decode signal trigger.

*정직성:* Impl-5 의 run() 은 LayerState advance 의 L 회 반복 + signal trigger 검증 만 —
실 instance_pipeline.dispatch 통합은 Impl-9 영역 (§7 O5.7).
"""

import pytest

from puls_sched.forward_pass import ForwardPass, LayerState
from puls_sched.micro_batch import MicroBatch


# =========================================================================
# LayerState.advance — basic
# =========================================================================

def test_layer_state_advance_increments_index(layer_state):
    mb = MicroBatch(id=0)
    assert mb.current_layer_index == 0
    layer_state.advance(mb)
    assert mb.current_layer_index == 1


def test_layer_state_advance_returns_true_at_l(layer_state):
    """L 도달 시 token decode signal trigger."""
    mb = MicroBatch(id=0, current_layer_index=layer_state.num_layers - 1)
    assert layer_state.advance(mb) is True


def test_layer_state_advance_returns_false_below_l(layer_state):
    mb = MicroBatch(id=0, current_layer_index=0)
    assert layer_state.advance(mb) is False
    # And another step still below L
    while mb.current_layer_index < layer_state.num_layers - 1:
        assert layer_state.advance(mb) is False


def test_layer_state_advance_negative_raises(layer_state):
    mb = MicroBatch(id=0, current_layer_index=-1)
    with pytest.raises(ValueError, match="non-negative"):
        layer_state.advance(mb)


def test_layer_state_advance_already_done_raises(layer_state):
    mb = MicroBatch(id=0, current_layer_index=layer_state.num_layers)
    with pytest.raises(ValueError, match="already reached"):
        layer_state.advance(mb)


# =========================================================================
# ForwardPass.run — L-iteration meta-count (PLAN §0.5 Impl-5 reminder)
# =========================================================================

@pytest.mark.parametrize("L", [1, 8, 32, 80])
def test_forward_pass_run_l_iterations(dummy_config, instance_pipeline, L):
    layer_state = LayerState(num_layers=L)
    fp = ForwardPass(config=dummy_config, instance_pipeline=instance_pipeline, layer_state=layer_state)
    mb = MicroBatch(id=0)
    fp.run(mb)
    assert mb.current_layer_index == L


@pytest.mark.parametrize("L", [1, 8, 32, 80])
def test_forward_pass_run_count_matches_l(dummy_config, instance_pipeline, L):
    layer_state = LayerState(num_layers=L)
    fp = ForwardPass(config=dummy_config, instance_pipeline=instance_pipeline, layer_state=layer_state)
    mb = MicroBatch(id=0)
    assert fp.run(mb) == L


@pytest.mark.parametrize("current", [1, 50])
def test_forward_pass_entry_non_zero_raises(forward_pass, current):
    mb = MicroBatch(id=0, current_layer_index=current)
    with pytest.raises(ValueError, match="must be 0"):
        forward_pass.run(mb)


def test_forward_pass_layer_index_monotonic_increasing(forward_pass, dummy_config):
    """run() 중 current_layer_index 의 단방향 증가 (역방향 transition reject 자연 강제)."""
    mb = MicroBatch(id=0)
    prev = mb.current_layer_index
    # Simulate one step at a time
    while mb.current_layer_index < dummy_config.model.num_layers:
        prev_index = mb.current_layer_index
        done = forward_pass.layer_state.advance(mb)
        assert mb.current_layer_index == prev_index + 1
        if done:
            break


def test_forward_pass_deterministic_run(dummy_config, instance_pipeline):
    """다회 run (각 새 mb instance) → 동일 layer trajectory."""
    layer_state = LayerState(num_layers=dummy_config.model.num_layers)
    fp = ForwardPass(config=dummy_config, instance_pipeline=instance_pipeline, layer_state=layer_state)
    trajectories = []
    for _ in range(10):
        mb = MicroBatch(id=0)
        count = fp.run(mb)
        trajectories.append((count, mb.current_layer_index))
    assert all(t == trajectories[0] for t in trajectories)


def test_forward_pass_default_num_layers_80(dummy_config):
    """ARCH (config default) — Llama-3 70B class L=80."""
    assert dummy_config.model.num_layers == 80


def test_forward_pass_token_decode_signal_at_l(forward_pass, dummy_config):
    """run() 종료 시점이 정확 L 회 advance 후 True signal."""
    mb = MicroBatch(id=0)
    forward_pass.run(mb)
    # Final state: current_layer_index == L (signal triggered at last advance)
    assert mb.current_layer_index == dummy_config.model.num_layers
    # Next advance must raise (already done)
    with pytest.raises(ValueError, match="already reached"):
        forward_pass.layer_state.advance(mb)
