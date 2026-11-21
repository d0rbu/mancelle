from __future__ import annotations

import dataclasses

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_max_steps():
    assert FeatureGcgConfig().max_steps == 64


def test_default_candidate_batch_size():
    assert FeatureGcgConfig().candidate_batch_size == 8


def test_default_top_k():
    assert FeatureGcgConfig().top_k == 32


def test_default_temperature():
    assert FeatureGcgConfig().temperature == 1.0


def test_default_master_seed():
    assert FeatureGcgConfig().master_seed == 0


def test_default_minibatch_size():
    assert FeatureGcgConfig().minibatch_size == 16


def test_default_reg_lambda():
    assert FeatureGcgConfig().reg_lambda == 0.0


def test_default_min_k_pos_zero():
    assert FeatureGcgConfig().min_k_pos == 0


def test_default_max_k_pos_100():
    assert FeatureGcgConfig().max_k_pos == 100


def test_default_min_k_neg_zero():
    assert FeatureGcgConfig().min_k_neg == 0


def test_default_max_k_neg_100():
    assert FeatureGcgConfig().max_k_neg == 100


# ---------------------------------------------------------------------------
# Frozen (immutable)
# ---------------------------------------------------------------------------


def test_frozen_raises_on_assignment():
    cfg = FeatureGcgConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_steps = 10  # type: ignore[misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# feature_masking was REMOVED (#720) -- pinned by absence, not just by omission
# from the tests above. A re-added inert `feature_masking` field would pass every
# other test in this file silently; these assert it directly.
# ---------------------------------------------------------------------------


def test_feature_masking_is_not_a_dataclass_field():
    field_names = {f.name for f in dataclasses.fields(FeatureGcgConfig)}
    assert "feature_masking" not in field_names


def test_feature_masking_is_not_an_attribute_of_an_instance():
    cfg = FeatureGcgConfig()
    assert not hasattr(cfg, "feature_masking")


# ---------------------------------------------------------------------------
# universal_minibatch_size property
# ---------------------------------------------------------------------------


def test_universal_minibatch_size_equals_minibatch_size():
    cfg = FeatureGcgConfig(minibatch_size=32)
    assert cfg.universal_minibatch_size == 32


def test_universal_minibatch_size_default():
    assert FeatureGcgConfig().universal_minibatch_size == 16


def test_universal_minibatch_size_int_cast():
    cfg = FeatureGcgConfig(minibatch_size=8)
    assert int(cfg.universal_minibatch_size) == 8


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validate_raises_on_top_k_zero():
    with pytest.raises(BeartypeCallHintParamViolation, match="top_k"):
        FeatureGcgConfig(top_k=0)


def test_validate_raises_on_top_k_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="top_k"):
        FeatureGcgConfig(top_k=-1)


def test_validate_raises_on_max_steps_zero():
    with pytest.raises(BeartypeCallHintParamViolation, match="max_steps"):
        FeatureGcgConfig(max_steps=0)


def test_validate_raises_on_max_steps_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="max_steps"):
        FeatureGcgConfig(max_steps=-5)


def test_validate_raises_on_candidate_batch_size_zero():
    with pytest.raises(BeartypeCallHintParamViolation, match="candidate_batch_size"):
        FeatureGcgConfig(candidate_batch_size=0)


def test_validate_raises_on_candidate_batch_size_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="candidate_batch_size"):
        FeatureGcgConfig(candidate_batch_size=-2)


def test_validate_raises_on_temperature_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="temperature"):
        FeatureGcgConfig(temperature=-0.1)


def test_validate_rejects_temperature_zero():
    # temperature=0.0 causes div-by-zero in exp(-Δ/T); config must reject it
    with pytest.raises(BeartypeCallHintParamViolation, match="temperature"):
        FeatureGcgConfig(temperature=0.0)


def test_validate_raises_on_minibatch_size_zero():
    with pytest.raises(BeartypeCallHintParamViolation, match="minibatch_size"):
        FeatureGcgConfig(minibatch_size=0)


def test_validate_raises_on_minibatch_size_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="minibatch_size"):
        FeatureGcgConfig(minibatch_size=-1)


def test_validate_raises_on_reg_lambda_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="reg_lambda"):
        FeatureGcgConfig(reg_lambda=-0.5)


def test_validate_allows_reg_lambda_zero():
    cfg = FeatureGcgConfig(reg_lambda=0.0)
    assert cfg.reg_lambda == 0.0


def test_validate_raises_on_min_k_pos_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="min_k_pos"):
        FeatureGcgConfig(min_k_pos=-1)


def test_validate_raises_on_max_k_pos_not_greater_than_min_k_pos():
    # Both values satisfy their unary domains; the relationship remains bespoke.
    with pytest.raises(FeatureGcgError, match="max_k_pos"):
        FeatureGcgConfig(min_k_pos=1, max_k_pos=1)


def test_validate_raises_on_min_k_neg_negative():
    with pytest.raises(BeartypeCallHintParamViolation, match="min_k_neg"):
        FeatureGcgConfig(min_k_neg=-1)


def test_validate_raises_on_max_k_neg_not_greater_than_min_k_neg():
    with pytest.raises(FeatureGcgError, match="max_k_neg"):
        FeatureGcgConfig(min_k_neg=1, max_k_neg=1)


# ---------------------------------------------------------------------------
# Valid non-default construction
# ---------------------------------------------------------------------------


def test_custom_construction():
    cfg = FeatureGcgConfig(
        max_steps=256,
        candidate_batch_size=16,
        top_k=64,
        temperature=0.5,
        master_seed=42,
        minibatch_size=8,
        reg_lambda=2.0,
        min_k_pos=1,
        max_k_pos=5,
        min_k_neg=1,
        max_k_neg=5,
    )
    assert cfg.max_steps == 256
    assert cfg.candidate_batch_size == 16
    assert cfg.top_k == 64
    assert cfg.temperature == 0.5
    assert cfg.master_seed == 42
    assert cfg.minibatch_size == 8
    assert cfg.reg_lambda == 2.0
    assert cfg.min_k_pos == 1
    assert cfg.max_k_pos == 5
    assert cfg.min_k_neg == 1
    assert cfg.max_k_neg == 5
    assert cfg.universal_minibatch_size == 8
