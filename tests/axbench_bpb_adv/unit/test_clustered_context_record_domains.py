"""Property contracts for clustered-context runtime record domains."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, assert_type, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.eval.axbench_bpb_adv.clustered_context_cache as cache_module
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    CanonicalPrefixPopulation,
    RawClusteredFeatureSegment,
    RawClusteredPrefix,
    canonicalize_prefix_population,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import align_char_ranges_to_decoded_prefix

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
NON_NEGATIVE_INTS = st.integers(min_value=0, max_value=1_000_000)
POSITIVE_INTS = st.integers(min_value=1, max_value=1_000_000)
INVALID_NON_NEGATIVE_INT = st.one_of(
    st.integers(min_value=-1_000_000, max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.none(),
    st.text(max_size=32),
)
SCALARS = st.one_of(
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.none(),
    st.text(max_size=32),
)
HASHISH = st.one_of(
    SCALARS,
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        max_size=80,
    ),
    st.text(alphabet="0123456789abcdef", min_size=16, max_size=16),
    st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
)


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_prefix_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _raw_segment(*, char_start: int = 0, char_end: int = 0) -> RawClusteredFeatureSegment:
    return RawClusteredFeatureSegment(
        feature_ids=t.empty(0, dtype=t.int64),
        feature_magnitudes=t.empty(0, dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


def _context_segment() -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=(),
        buckets=(),
        char_start=0,
        char_end=0,
    )


@PROPERTY_SETTINGS
@given(values=st.lists(NON_NEGATIVE_INTS, max_size=40))
def test_alignment_accepts_every_valid_character_offset_list(values: list[int]) -> None:
    decoded_len = values[-1] if values else 0
    starts, ends, kept = align_char_ranges_to_decoded_prefix(
        char_starts=list(values),
        char_ends=list(values),
        decoded_len=decoded_len,
    )

    assert starts == values
    assert ends == values
    assert kept == list(range(len(values)))


@PROPERTY_SETTINGS
@given(
    prefix=st.lists(NON_NEGATIVE_INTS, max_size=20),
    invalid=INVALID_NON_NEGATIVE_INT,
    suffix=st.lists(NON_NEGATIVE_INTS, max_size=20),
)
def test_alignment_rejects_an_invalid_character_start_at_every_position(
    prefix: list[int],
    invalid: object,
    suffix: list[int],
) -> None:
    starts = [*prefix, invalid, *suffix]
    with pytest.raises(BeartypeCallHintParamViolation, match="char_starts"):
        align_char_ranges_to_decoded_prefix(
            char_starts=cast(Any, starts),
            char_ends=[0] * len(starts),
            decoded_len=0,
        )


@PROPERTY_SETTINGS
@given(
    prefix=st.lists(NON_NEGATIVE_INTS, max_size=20),
    invalid=INVALID_NON_NEGATIVE_INT,
    suffix=st.lists(NON_NEGATIVE_INTS, max_size=20),
)
def test_alignment_rejects_an_invalid_character_end_at_every_position(
    prefix: list[int],
    invalid: object,
    suffix: list[int],
) -> None:
    ends = [*prefix, invalid, *suffix]
    with pytest.raises(BeartypeCallHintParamViolation, match="char_ends"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0] * len(ends),
            char_ends=cast(Any, ends),
            decoded_len=0,
        )


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_alignment_decoded_length_matches_independent_oracle(value: object) -> None:
    if _is_non_negative_int(value):
        assert align_char_ranges_to_decoded_prefix(
            char_starts=[],
            char_ends=[],
            decoded_len=cast(int, value),
        ) == ([], [], [])
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="decoded_len"):
            align_char_ranges_to_decoded_prefix(
                char_starts=[],
                char_ends=[],
                decoded_len=cast(Any, value),
            )


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_raw_segment_character_offsets_match_independent_oracle(value: object) -> None:
    if _is_non_negative_int(value):
        segment = _raw_segment(
            char_start=cast(int, value),
            char_end=cast(int, value),
        )
        assert segment.char_start == value
        assert segment.char_end == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="char_start"):
            RawClusteredFeatureSegment(
                feature_ids=t.empty(0, dtype=t.int64),
                feature_magnitudes=t.empty(0, dtype=t.float32),
                char_start=cast(Any, value),
                char_end=0,
            )


@PROPERTY_SETTINGS
@given(
    char_start=NON_NEGATIVE_INTS,
    char_end=NON_NEGATIVE_INTS,
)
def test_raw_segment_character_range_relationship_is_preserved(
    char_start: int,
    char_end: int,
) -> None:
    if char_end >= char_start:
        segment = _raw_segment(char_start=char_start, char_end=char_end)
        assert (segment.char_start, segment.char_end) == (char_start, char_end)
    else:
        with pytest.raises(AxBenchBpbAdvError, match="char_end"):
            _raw_segment(char_start=char_start, char_end=char_end)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_raw_prefix_token_count_matches_independent_oracle(value: object) -> None:
    if _is_positive_int(value):
        prefix = RawClusteredPrefix(segments=(), n_tokens=cast(int, value))
        assert prefix.n_tokens == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_tokens"):
            RawClusteredPrefix(segments=(), n_tokens=cast(Any, value))


@PROPERTY_SETTINGS
@given(
    value=st.one_of(
        st.lists(st.just(_raw_segment()), max_size=12),
        st.lists(st.just(_raw_segment()), max_size=12).map(tuple),
        st.tuples(
            st.lists(st.just(_raw_segment()), max_size=6).map(tuple),
            SCALARS,
            st.lists(st.just(_raw_segment()), max_size=6).map(tuple),
        ).map(lambda parts: (*parts[0], parts[1], *parts[2])),
    )
)
def test_raw_prefix_segment_collection_checks_every_item(value: object) -> None:
    valid = isinstance(value, tuple) and all(
        isinstance(item, RawClusteredFeatureSegment) for item in value
    )
    if valid:
        assert (
            RawClusteredPrefix(
                segments=cast(tuple[RawClusteredFeatureSegment, ...], value),
                n_tokens=1,
            ).segments
            == value
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
            RawClusteredPrefix(segments=cast(Any, value), n_tokens=1)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_materialized_segment_counts_match_independent_oracle(value: object) -> None:
    kwargs = {
        "context": _context_segment(),
        "positive_percentiles_uint8": t.empty(0, dtype=t.uint8),
        "n_positive": value,
        "n_negative": value if _is_non_negative_int(value) else 0,
    }
    if _is_non_negative_int(value):
        segment = cache_module._MaterializedSegment(**cast(Any, kwargs))
        assert segment.n_positive == value
        assert segment.n_negative == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_positive"):
            cache_module._MaterializedSegment(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(value=HASHISH)
def test_raw_metadata_prefix_hash_matches_independent_oracle(value: object) -> None:
    kwargs = {
        "contract": SimpleNamespace(feature_contract_fingerprint="f" * 64),
        "prefix_hash": value,
        "prefix_text_sha256": "a" * 64,
        "extraction_identity_sha256": "b" * 64,
        "n_tokens": 1,
    }
    if _is_prefix_hash(value):
        assert cache_module._raw_metadata(**cast(Any, kwargs))["prefix_hash"] == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="prefix_hash"):
            cache_module._raw_metadata(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("prefix_text_sha256", "extraction_identity_sha256")),
    value=HASHISH,
)
def test_raw_metadata_sha256_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "contract": SimpleNamespace(feature_contract_fingerprint="f" * 64),
        "prefix_hash": "0" * 16,
        "prefix_text_sha256": "a" * 64,
        "extraction_identity_sha256": "b" * 64,
        "n_tokens": 1,
    }
    kwargs[field] = value
    if _is_sha256(value):
        assert cache_module._raw_metadata(**cast(Any, kwargs))[field] == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            cache_module._raw_metadata(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(prefixes=st.dictionaries(st.text(max_size=40), st.none(), max_size=30))
def test_canonical_population_valid_mapping_has_exact_sorted_snapshot(
    prefixes: dict[str, None],
) -> None:
    mapping = {prefix: make_prefix_hash(prefix) for prefix in prefixes}
    population = canonicalize_prefix_population(mapping)

    assert population.entries == tuple(sorted(mapping.items()))
    assert population.as_dict() == mapping


@PROPERTY_SETTINGS
@given(
    key=st.one_of(
        st.integers(),
        st.booleans(),
        st.none(),
        st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_canonical_population_mapping_rejects_every_non_string_key(key: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
        canonicalize_prefix_population(cast(Any, {key: "digest"}))


@PROPERTY_SETTINGS
@given(value=st.one_of(st.integers(), st.booleans(), st.none(), st.floats(), st.lists(SCALARS)))
def test_canonical_population_mapping_rejects_every_non_string_value(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
        canonicalize_prefix_population(cast(Any, {"prefix": value}))


@PROPERTY_SETTINGS
@given(
    prefix_count=st.integers(min_value=0, max_value=30),
    invalid=st.one_of(
        SCALARS,
        st.lists(SCALARS, max_size=4),
        st.tuples(
            st.text(max_size=8),
            st.one_of(
                st.integers(),
                st.booleans(),
                st.floats(allow_nan=True, allow_infinity=True),
                st.none(),
                st.lists(SCALARS, max_size=3),
            ),
        ),
        st.tuples(st.text(max_size=8)),
    ),
    suffix_count=st.integers(min_value=0, max_value=30),
)
def test_canonical_population_entries_reject_an_invalid_item_at_every_position(
    prefix_count: int,
    invalid: object,
    suffix_count: int,
) -> None:
    valid_entry = ("alpha", make_prefix_hash("alpha"))
    entries = (
        *(valid_entry for _ in range(prefix_count)),
        invalid,
        *(valid_entry for _ in range(suffix_count)),
    )
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        CanonicalPrefixPopulation(entries=cast(Any, entries))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_ddp_rank_domain_matches_independent_oracle(value: object) -> None:
    if _is_non_negative_int(value):
        rank = cast(int, value)
        try:
            cache_module._validate_ddp(rank=rank, world_size=rank + 1)
        except AxBenchBpbAdvError:
            # A valid type-level coordinate can still require a live process
            # group; that relationship is intentionally checked explicitly.
            pass
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="rank"):
            cache_module._validate_ddp(rank=cast(Any, value), world_size=1)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_ddp_world_size_domain_matches_independent_oracle(value: object) -> None:
    if _is_positive_int(value):
        try:
            cache_module._validate_ddp(rank=0, world_size=cast(int, value))
        except AxBenchBpbAdvError:
            pass
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="world_size"):
            cache_module._validate_ddp(rank=0, world_size=cast(Any, value))


def test_clustered_record_fields_expose_nominal_refined_types() -> None:
    raw_segment = _raw_segment(char_start=2, char_end=3)
    raw_prefix = RawClusteredPrefix(segments=(raw_segment,), n_tokens=7)
    materialized = cache_module._MaterializedSegment(
        context=_context_segment(),
        positive_percentiles_uint8=t.empty(0, dtype=t.uint8),
        n_positive=0,
        n_negative=0,
    )

    assert_type(raw_segment.char_start, NonNegativeInt)
    assert_type(raw_segment.char_end, NonNegativeInt)
    assert_type(raw_prefix.n_tokens, PositiveInt)
    assert_type(materialized.n_positive, NonNegativeInt)
    assert_type(materialized.n_negative, NonNegativeInt)


def test_full_collection_predicates_do_not_consume_global_rng() -> None:
    valid_offsets = list(range(10_000))
    invalid_offsets: list[Any] = [*valid_offsets, True]
    segments = tuple(_raw_segment() for _ in range(2_000))
    invalid_segments: tuple[Any, ...] = (*segments, object())
    python_state = random.getstate()
    torch_state = t.random.get_rng_state().clone()

    align_char_ranges_to_decoded_prefix(
        char_starts=valid_offsets,
        char_ends=valid_offsets,
        decoded_len=valid_offsets[-1],
    )
    RawClusteredPrefix(segments=segments, n_tokens=1)
    with pytest.raises(BeartypeCallHintParamViolation, match="char_starts"):
        align_char_ranges_to_decoded_prefix(
            char_starts=invalid_offsets,
            char_ends=[0] * len(invalid_offsets),
            decoded_len=0,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
        RawClusteredPrefix(segments=invalid_segments, n_tokens=1)

    assert random.getstate() == python_state
    assert t.equal(t.random.get_rng_state(), torch_state)


def test_canonical_population_entries_are_immutable_and_relationship_checked() -> None:
    valid = CanonicalPrefixPopulation(entries=(("alpha", make_prefix_hash("alpha")),))
    assert valid.as_dict() == {"alpha": make_prefix_hash("alpha")}
    with pytest.raises(BeartypeCallHintParamViolation, match="entries"):
        CanonicalPrefixPopulation(entries=cast(Any, [["alpha", make_prefix_hash("alpha")]]))
