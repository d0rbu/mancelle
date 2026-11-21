"""Independent property oracles for Feature-GCG refined domains."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.feature_gcg.candidate_search import (
    rank_replacements_from_gradients,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.edit_primitives import TokenReplacement
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import (
    FeatureConditioning,
    FeatureConditioningEntries,
)
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureConditioningEditSpace,
)
from steering.conditional.feature_gcg.gradient import _batched_masked_losses

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=25),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=12),
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_entries(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    seen_fids: set[int] = set()
    for entry in value:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return False
        fid, bucket = entry
        if not _strict_int(fid) or not _strict_int(bucket):
            return False
        fid_int = cast(int, fid)
        bucket_int = cast(int, bucket)
        if fid_int < 0 or bucket_int < 0 or fid_int in seen_fids:
            return False
        seen_fids.add(fid_int)
    return True


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "max_steps",
            "candidate_batch_size",
            "top_k",
            "minibatch_size",
            "max_forward_batch",
            "max_k_pos",
            "max_k_neg",
        )
    ),
    value=SCALARS,
)
def test_config_positive_integer_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        FeatureGcgConfig(**{field: cast(Any, value)})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            FeatureGcgConfig(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("min_k_pos", "min_k_neg")),
    value=SCALARS,
)
def test_config_non_negative_integer_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        FeatureGcgConfig(**{field: cast(Any, value)})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            FeatureGcgConfig(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("temperature", "reg_lambda")),
    value=SCALARS,
)
def test_config_finite_numeric_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = _finite_number(value) and (
        value > 0 if field == "temperature" else value >= 0  # ty: ignore[unsupported-operator]
    )
    if valid:
        FeatureGcgConfig(**{field: cast(Any, value)})
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            FeatureGcgConfig(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_config_master_seed_is_a_strict_integer(value: object) -> None:
    if _strict_int(value):
        FeatureGcgConfig(master_seed=cast(Any, value))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="master_seed"):
            FeatureGcgConfig(master_seed=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    partition=st.sampled_from(("pos", "neg")),
    minimum=st.integers(min_value=0, max_value=100),
    maximum=st.integers(min_value=1, max_value=100),
)
def test_partition_relationship_remains_explicit(
    partition: str,
    minimum: int,
    maximum: int,
) -> None:
    kwargs = {f"min_k_{partition}": minimum, f"max_k_{partition}": maximum}
    if maximum > minimum:
        FeatureGcgConfig(**kwargs)
    else:
        with pytest.raises(FeatureGcgError, match=f"max_k_{partition}"):
            FeatureGcgConfig(**kwargs)


@PROPERTY_SETTINGS
@given(entries=st.lists(st.tuples(SCALARS, SCALARS), max_size=8).map(tuple))
def test_conditioning_entry_collection_matches_independent_oracle(
    entries: tuple[tuple[object, object], ...],
) -> None:
    valid = _valid_entries(entries)
    assert isinstance(entries, FeatureConditioningEntries) is valid
    if valid:
        conditioning = FeatureConditioning(entries=cast(Any, entries))
        assert conditioning.entries == tuple(sorted(entries))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
            FeatureConditioning(entries=cast(Any, entries))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("fid", "bucket")), value=SCALARS)
def test_conditioning_mutation_scalars_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {"fid": 1, "bucket": 2}
    kwargs[field] = value
    conditioning = FeatureConditioning(entries=())
    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if valid:
        conditioning.with_added(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            conditioning.with_added(**kwargs)


@PROPERTY_SETTINGS
@given(operation=st.sampled_from(("replace", "remove")), value=SCALARS)
def test_conditioning_slot_domain_precedes_upper_bound_relationship(
    operation: str,
    value: object,
) -> None:
    conditioning = FeatureConditioning(entries=((1, 2),))

    def call() -> object:
        if operation == "replace":
            return conditioning.with_replaced(cast(Any, value), 3, 4)
        return conditioning.with_removed(cast(Any, value))

    valid = _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]
    if not valid:
        with pytest.raises(BeartypeCallHintParamViolation, match="slot"):
            call()
    elif value == 0:
        call()
    else:
        with pytest.raises(FeatureGcgError, match="out of range"):
            call()


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("top_k", "distance_weight")), value=SCALARS)
def test_candidate_ranking_scalars_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {"top_k": 1, "distance_weight": 0.0}
    kwargs[field] = value
    if field == "top_k":
        valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    else:
        valid = _finite_number(value) and value >= 0  # ty: ignore[unsupported-operator]

    def call() -> object:
        return rank_replacements_from_gradients(
            embedding_gradient=t.zeros(2, 1),
            embedding_weight=t.zeros(2, 1),
            editable_positions=(1,),
            allowed_token_ids=(0, 1),
            current_tokens=(0, 1),
            **kwargs,
        )

    if valid:
        assert cast(dict[int, object], call())[1]
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            call()


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("batch_size", "temperature")), value=SCALARS)
def test_candidate_sampling_scalars_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {"batch_size": 1, "temperature": 1.0}
    kwargs[field] = value
    if field == "batch_size":
        valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    else:
        valid = _finite_number(value) and value > 0  # ty: ignore[unsupported-operator]

    def call() -> object:
        return sample_candidate_sequences(
            current_tokens=(0, 1),
            replacements_by_position={1: (TokenReplacement(position=1, token_id=0, score=0.0),)},
            rng=random.Random(0),
            **kwargs,
        )

    if valid:
        assert call()
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            call()


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("batch_size", "temperature")), value=SCALARS)
def test_edit_space_sampling_scalars_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {"batch_size": 1, "temperature": 1.0}
    kwargs[field] = value
    if field == "batch_size":
        valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    else:
        valid = _finite_number(value) and value > 0  # ty: ignore[unsupported-operator]

    def call() -> object:
        return FeatureConditioningEditSpace.sample_candidates(
            cast(Any, object()),
            SimpleNamespace(edits=()),
            FeatureConditioning(entries=()),
            rng=random.Random(0),
            visited=set(),
            **kwargs,
        )

    if valid:
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_forward_batch_size_domain_precedes_structural_checks(value: object) -> None:
    def call() -> object:
        return _batched_masked_losses(
            object(),
            [],
            [],
            "cpu",
            max_forward_batch=cast(Any, value),
        )

    valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        with pytest.raises(FeatureGcgError, match="no members"):
            call()
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="max_forward_batch"):
            call()


def test_typed_container_boundaries_preserve_global_python_rng() -> None:
    """Beartype must not sample generic containers through the global RNG."""
    random.seed(31_415)
    state = random.getstate()

    conditioning = FeatureConditioning.from_dict({2: 1, 5: 3})
    assert conditioning.feature_ids() == (2, 5)
    ranked = rank_replacements_from_gradients(
        embedding_gradient=t.zeros(2, 1),
        embedding_weight=t.zeros(3, 1),
        editable_positions=(1,),
        allowed_token_ids=(0, 1, 2),
        current_tokens=(0, 1),
        top_k=1,
        distance_weight=0.0,
    )
    assert sample_candidate_sequences(
        current_tokens=(0, 1),
        replacements_by_position=ranked,
        batch_size=1,
        temperature=1.0,
        rng=random.Random(7),
    )
    assert (
        FeatureConditioningEditSpace.sample_candidates(
            cast(Any, object()),
            SimpleNamespace(edits=()),
            FeatureConditioning(entries=()),
            batch_size=1,
            temperature=1.0,
            rng=random.Random(9),
            visited=set(),
        )
        == []
    )
    with pytest.raises(FeatureGcgError, match="no members"):
        _batched_masked_losses(
            object(),
            [],
            [],
            "cpu",
            max_forward_batch=1,
        )

    assert random.getstate() == state
