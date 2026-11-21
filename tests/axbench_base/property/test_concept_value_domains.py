"""Property oracles for shared AxBench concept value domains."""

from __future__ import annotations

import random
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.types import CandidateConcept, Concept

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)


def _concept(**overrides: Any) -> Concept:
    values: dict[str, Any] = {
        "concept_id": 0,
        "feature_index": 1,
        "layer": 8,
        "sae_source": "sae",
        "description": "description",
        "corpus_count": 2,
        "description_source": "source",
    }
    values.update(overrides)
    return Concept(**cast(Any, values))


def _candidate(**overrides: Any) -> CandidateConcept:
    values: dict[str, Any] = {
        "feature_index": 1,
        "layer": 8,
        "sae_source": "sae",
        "description": "description",
        "corpus_count": 2,
        "description_source": "source",
    }
    values.update(overrides)
    return CandidateConcept(**cast(Any, values))


def _build(kind: str, field: str, value: object) -> Concept | CandidateConcept:
    overrides = {field: value}
    return _concept(**overrides) if kind == "concept" else _candidate(**overrides)


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_blank_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("concept", "concept_id"),
        ("concept", "feature_index"),
        ("concept", "corpus_count"),
        ("candidate", "feature_index"),
        ("candidate", "corpus_count"),
    ],
)
@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_non_negative_integer_fields_match_independent_oracle(
    kind: str,
    field: str,
    value: object,
) -> None:
    if _is_non_negative_int(value):
        assert getattr(_build(kind, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _build(kind, field, value)


@pytest.mark.parametrize("kind", ["concept", "candidate"])
@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_layer_fields_match_strict_integer_oracle(kind: str, value: object) -> None:
    if _is_strict_int(value):
        assert _build(kind, "layer", value).layer == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="layer"):
            _build(kind, "layer", value)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        (kind, field)
        for kind in ("concept", "candidate")
        for field in ("sae_source", "description", "description_source")
    ],
)
@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_non_blank_string_fields_match_independent_oracle(
    kind: str,
    field: str,
    value: object,
) -> None:
    if _is_non_blank_str(value):
        assert getattr(_build(kind, field, value), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _build(kind, field, value)


def test_concept_construction_preserves_global_python_rng() -> None:
    random.seed(31_415)
    before = random.getstate()

    _concept()
    _candidate()

    assert random.getstate() == before
