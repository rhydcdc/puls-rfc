"""Impl-5 — InstancePipeline (A → B handoff + steady_state_cycle + fixed-shape gate).

ARCH §3.4 · §5.2 정합. Q3 (단일 layer cycle owner), Q5 (raise on ragged),
Q6 (runtime getter), Q7 (max(A,B) literal, NVLink hidden).
"""

import pytest

from puls_sched.instance import Instance
from puls_sched.instance_pipeline import InstancePipeline
from puls_sched.micro_batch import MicroBatch
from puls_sched.nvlink import NVLinkTransfer


# =========================================================================
# Init guard (ARCH §3.4 Case A)
# =========================================================================

def test_pipeline_init_requires_instance_a_has_pim(dummy_config, nvlink_transfer):
    """instance_a 가 PIM 미보유 시 raise."""
    a_no_pim = Instance(name="A", has_pim=False)
    b = Instance(name="B", has_pim=False)
    with pytest.raises(ValueError, match="instance_a must have PIM"):
        InstancePipeline(config=dummy_config, instance_a=a_no_pim, instance_b=b, nvlink=nvlink_transfer)


def test_pipeline_init_rejects_instance_b_has_pim(dummy_config, nvlink_transfer):
    """instance_b 가 PIM 보유 시 raise."""
    a = Instance(name="A", has_pim=True)
    b_with_pim = Instance(name="B", has_pim=True)
    with pytest.raises(ValueError, match="instance_b must not have PIM"):
        InstancePipeline(config=dummy_config, instance_a=a, instance_b=b_with_pim, nvlink=nvlink_transfer)


# =========================================================================
# Steady-state cycle = max(A, B) (ARCH §3.4 literal, Q6+Q7)
# =========================================================================

def test_steady_state_cycle_max_a_gt_b(instance_pipeline):
    assert instance_pipeline.steady_state_cycle(10.0, 5.0) == 10.0


def test_steady_state_cycle_max_a_lt_b(instance_pipeline):
    assert instance_pipeline.steady_state_cycle(5.0, 10.0) == 10.0


def test_steady_state_cycle_max_a_eq_b(instance_pipeline):
    assert instance_pipeline.steady_state_cycle(7.0, 7.0) == 7.0


@pytest.mark.parametrize("a,b", [(-1.0, 5.0), (5.0, -1.0), (-0.001, -0.001)])
def test_steady_state_cycle_negative_raises(instance_pipeline, a, b):
    with pytest.raises(ValueError, match="non-negative"):
        instance_pipeline.steady_state_cycle(a, b)


def test_steady_state_cycle_deterministic_1000_calls(instance_pipeline):
    expected = instance_pipeline.steady_state_cycle(3.5, 7.2)
    for _ in range(1000):
        assert instance_pipeline.steady_state_cycle(3.5, 7.2) == expected


# =========================================================================
# Fixed-shape handoff gate (ARCH §5.2 literal)
# =========================================================================

def test_validate_handoff_decode_only_pass(instance_pipeline, dummy_config):
    """Decode-only: shape == (B, hidden)."""
    mb = MicroBatch(id=0, decode_tokens={1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8})
    instance_pipeline.validate_handoff_shape(mb, (8, dummy_config.model.hidden))


def test_validate_handoff_pure_prefill_pass(instance_pipeline, dummy_config):
    """Pure-prefill uniform chunk: shape == (B * chunk, hidden)."""
    chunk = 256
    mb = MicroBatch(
        id=0,
        prefill_chunk={1: list(range(chunk)), 2: list(range(chunk)), 3: list(range(chunk)), 4: list(range(chunk))},
    )
    instance_pipeline.validate_handoff_shape(mb, (4 * chunk, dummy_config.model.hidden))


def test_validate_handoff_mixed_uniform_pass(instance_pipeline, dummy_config):
    """Mixed: shape == (B_decode + B_prefill * chunk, hidden)."""
    chunk = 256
    mb = MicroBatch(
        id=0,
        decode_tokens={1: 1, 2: 2, 3: 3, 4: 4},
        prefill_chunk={5: list(range(chunk)), 6: list(range(chunk))},
    )
    instance_pipeline.validate_handoff_shape(mb, (4 + 2 * chunk, dummy_config.model.hidden))


def test_validate_handoff_ragged_prefill_chunk_raises(instance_pipeline, dummy_config):
    """ARCH §5.2 — uniform-chunk 위반 reject."""
    mb = MicroBatch(
        id=0,
        prefill_chunk={1: list(range(256)), 2: list(range(128))},  # ragged
    )
    with pytest.raises(AssertionError, match="ragged prefill chunk"):
        instance_pipeline.validate_handoff_shape(mb, (384, dummy_config.model.hidden))


def test_validate_handoff_wrong_hidden_dim_raises(instance_pipeline, dummy_config):
    mb = MicroBatch(id=0, decode_tokens={1: 1})
    with pytest.raises(AssertionError, match="hidden dim mismatch"):
        instance_pipeline.validate_handoff_shape(mb, (1, dummy_config.model.hidden + 1))


@pytest.mark.parametrize("off", [-1, 1])
def test_validate_handoff_wrong_n_tokens_raises(instance_pipeline, dummy_config, off):
    mb = MicroBatch(id=0, decode_tokens={i: i for i in range(8)})
    with pytest.raises(AssertionError, match="tokens mismatch"):
        instance_pipeline.validate_handoff_shape(mb, (8 + off, dummy_config.model.hidden))


def test_validate_handoff_3d_shape_raises(instance_pipeline, dummy_config):
    mb = MicroBatch(id=0, decode_tokens={1: 1})
    with pytest.raises(AssertionError, match="must be 2D"):
        instance_pipeline.validate_handoff_shape(mb, (1, dummy_config.model.hidden, 1))


@pytest.mark.parametrize("decode_B", [1, 8, 64])
@pytest.mark.parametrize("prefill_B", [0, 2, 4])
@pytest.mark.parametrize("chunk", [64, 256])
def test_validate_handoff_fixed_shape_cross_product(
    instance_pipeline, dummy_config, decode_B, prefill_B, chunk,
):
    """ARCH §5.2 정확 반영 — decode_B × prefill_B × chunk cross-product."""
    mb = MicroBatch(
        id=0,
        decode_tokens={i: i for i in range(decode_B)},
        prefill_chunk={1000 + i: list(range(chunk)) for i in range(prefill_B)},
    )
    expected_tokens = decode_B + prefill_B * chunk
    if expected_tokens == 0:
        # validate 가 의미 없음 — skip (decode 도 prefill 도 없는 경우)
        return
    instance_pipeline.validate_handoff_shape(mb, (expected_tokens, dummy_config.model.hidden))
