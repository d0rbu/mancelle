"""Property oracles for clustered-evaluation contract value domains."""

from __future__ import annotations

import random
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.cluster_contract_identity import (
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
)
from steering.eval.cluster_feature_contract import ClusterFeatureTarget

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=80),
    st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
)
NON_NEGATIVE_INTS = st.integers(min_value=0, max_value=100)
INVALID_TUPLE_ITEM = st.one_of(
    st.integers(min_value=-100, max_value=-1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)
CANONICAL_OR_INVALID_TUPLE = st.one_of(
    st.lists(NON_NEGATIVE_INTS, unique=True, max_size=12).map(lambda values: tuple(sorted(values))),
    st.none(),
    st.lists(NON_NEGATIVE_INTS, max_size=8),
    st.tuples(INVALID_TUPLE_ITEM),
)

_DEFAULTS: dict[str, dict[str, Any]] = {
    "directory": {
        "segment_size": 128,
        "layer": 8,
        "activation_type": "residual",
        "pooling": "max_pool",
        "width": "16k",
        "model": "gemma-2-2b",
        "dataset_phase": "base",
        "context_independent_features": False,
        "ignore_bos_features": True,
    },
    "statistics": {
        "file_sha256": "a" * 64,
        "percentile_table_checksum": "b" * 64,
        "n_sae_features": 101,
        "feature_id_domain": (),
    },
    "sae": {
        "release": "release",
        "sae_id": "sae-id",
        "hook_name": "hook",
        "model": "model",
        "layer": 8,
        "activation_type": "residual",
        "width": "16k",
        "actual_loaded_width": 16_384,
    },
    "target": {
        "representative_id": 0,
        "allowed_member_ids": (0,),
        "projecting_source_ids": (),
        "feature_contract_fingerprint": "c" * 64,
    },
}


def _build(kind: str, **overrides: Any) -> Any:
    constructors = {
        "directory": FeatureDirectoryIdentity,
        "statistics": FeatureStatisticsIdentity,
        "sae": SaeIdentity,
        "target": ClusterFeatureTarget,
    }
    values = {**_DEFAULTS[kind], **overrides}
    return constructors[kind](**cast(Any, values))


def _is_strict_int(value: object, *, positive: bool) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= (1 if positive else 0)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@pytest.mark.parametrize(
    ("kind", "field", "domain"),
    [
        ("directory", "segment_size", "positive"),
        ("statistics", "n_sae_features", "positive"),
        ("sae", "actual_loaded_width", "positive"),
        ("directory", "layer", "non_negative"),
        ("sae", "layer", "non_negative"),
        ("target", "representative_id", "non_negative"),
        *(
            ("directory", field, "non_empty")
            for field in ("activation_type", "pooling", "width", "model", "dataset_phase")
        ),
        *(
            ("sae", field, "non_empty")
            for field in ("release", "sae_id", "hook_name", "model", "activation_type", "width")
        ),
        ("statistics", "file_sha256", "sha256"),
        ("statistics", "percentile_table_checksum", "sha256"),
        ("target", "feature_contract_fingerprint", "sha256"),
        ("directory", "context_independent_features", "bool"),
        ("directory", "ignore_bos_features", "bool"),
    ],
)
@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_scalar_fields_match_independent_oracles(
    kind: str,
    field: str,
    domain: str,
    value: object,
) -> None:
    valid = {
        "positive": lambda item: _is_strict_int(item, positive=True),
        "non_negative": lambda item: _is_strict_int(item, positive=False),
        "non_empty": lambda item: isinstance(item, str) and bool(item),
        "sha256": _is_sha256,
        "bool": lambda item: isinstance(item, bool),
    }[domain](value)
    overrides = {field: value}
    if kind == "target" and field == "representative_id" and valid:
        overrides["allowed_member_ids"] = (value,)
    if valid:
        assert getattr(_build(kind, **overrides), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _build(kind, **overrides)


@PROPERTY_SETTINGS
@given(
    kind=st.sampled_from(("statistics", "allowed", "projecting")),
    value=CANONICAL_OR_INVALID_TUPLE,
)
def test_tuple_fields_match_independent_oracles(kind: str, value: object) -> None:
    valid = isinstance(value, tuple) and all(_is_strict_int(item, positive=False) for item in value)
    if kind == "allowed":
        valid = valid and bool(value)
        field = "allowed_member_ids"
        representative_id = cast(tuple[int, ...], value)[0] if valid else 0
        overrides = {field: value, "representative_id": representative_id}
        build_kind = "target"
    else:
        field = "feature_id_domain" if kind == "statistics" else "projecting_source_ids"
        overrides = {field: value}
        build_kind = "statistics" if kind == "statistics" else "target"
    if valid:
        assert getattr(_build(build_kind, **overrides), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _build(build_kind, **overrides)


@PROPERTY_SETTINGS
@given(
    representative_id=NON_NEGATIVE_INTS,
    allowed_member_ids=st.lists(NON_NEGATIVE_INTS, unique=True, min_size=1, max_size=12).map(tuple),
)
def test_target_retains_order_and_representative_relationships(
    representative_id: int,
    allowed_member_ids: tuple[int, ...],
) -> None:
    valid = (
        allowed_member_ids == tuple(sorted(allowed_member_ids))
        and allowed_member_ids[0] == representative_id
    )
    if valid:
        assert (
            _build(
                "target",
                representative_id=representative_id,
                allowed_member_ids=allowed_member_ids,
            ).allowed_member_ids
            == allowed_member_ids
        )
    else:
        with pytest.raises(ClusterFeatureContractError):
            _build(
                "target",
                representative_id=representative_id,
                allowed_member_ids=allowed_member_ids,
            )


def test_cluster_contract_value_construction_preserves_global_python_rng() -> None:
    random.seed(31_415)
    before = random.getstate()
    _build("directory")
    _build("statistics", feature_id_domain=(1, 2, 5))
    _build("sae")
    _build("target", allowed_member_ids=(0, 1), projecting_source_ids=(2, 3))
    assert random.getstate() == before
