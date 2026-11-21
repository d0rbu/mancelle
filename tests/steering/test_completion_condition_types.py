"""CompletionSample condition labels are enum-backed and validated."""

from __future__ import annotations

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.multiseg_generator import (
    AddedBucketCondition,
    CompletionCondition,
    CompletionSample,
    added_bucket_condition,
)


def test_completion_sample_coerces_fixed_condition_to_enum() -> None:
    sample = CompletionSample(condition="real", text="hello", seed=1)

    assert sample.condition is CompletionCondition.REAL
    assert sample.condition == "real"
    assert sample.bucket_index is None


def test_completion_sample_coerces_added_bucket_condition() -> None:
    sample = CompletionSample(condition="added_bucket_3", text="hello", seed=1)

    assert isinstance(sample.condition, AddedBucketCondition)
    assert sample.condition == "added_bucket_3"
    assert sample.bucket_index == 3


def test_added_bucket_condition_factory_validates_bucket_index() -> None:
    assert added_bucket_condition(2) == "added_bucket_2"

    with pytest.raises(BeartypeCallHintParamViolation, match="bucket_index"):
        added_bucket_condition(-1)


def test_completion_sample_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError, match="Unknown completion condition"):
        CompletionSample(condition="plain", text="hello", seed=1)


def test_completion_sample_rejects_bucket_mismatch() -> None:
    with pytest.raises(ValueError, match="bucket_index does not match"):
        CompletionSample(
            condition="added_bucket_3",
            text="hello",
            seed=1,
            bucket_index=4,
        )


def test_completion_sample_rejects_bucket_index_for_fixed_condition() -> None:
    with pytest.raises(ValueError, match="bucket_index is only valid"):
        CompletionSample(
            condition=CompletionCondition.REAL,
            text="hello",
            seed=1,
            bucket_index=0,
        )
