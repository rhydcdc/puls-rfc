import dataclasses
import random

import pytest

from puls_sched.config import Config, ModelConfig, default_dummy_config


def test_required_field_missing_fails_fast():
    with pytest.raises(TypeError):
        ModelConfig()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Config()  # type: ignore[call-arg]


def test_seed_changes_affect_rng_pipeline():
    c_a = dataclasses.replace(default_dummy_config(), seed=42)
    c_b = dataclasses.replace(default_dummy_config(), seed=42)
    c_c = dataclasses.replace(default_dummy_config(), seed=43)
    out_a = [random.Random(c_a.seed).random() for _ in range(3)]
    out_b = [random.Random(c_b.seed).random() for _ in range(3)]
    out_c = [random.Random(c_c.seed).random() for _ in range(3)]
    assert out_a == out_b
    assert out_a != out_c


def test_invariant_config_immutability():
    config = default_dummy_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 999  # type: ignore[misc]
