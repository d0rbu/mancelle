"""Static, runtime, and property contracts for reusable refined value types."""

import random
from dataclasses import replace
from typing import assert_type

import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.splice_resource_admission import AdmissionPolicy
from steering.refined_types import (
    AtLeastTwoInt,
    FiniteNumber,
    LowercaseSha256,
    NonBlankStr,
    NonEmptyNonBlankStrListLike,
    NonEmptyNonNegativeIntListListLike,
    NonEmptyNonNegativeIntTuple,
    NonNegativeFiniteNumber,
    NonNegativeFloat,
    NonNegativeInt,
    NonNegativeIntListListLike,
    OpenUnitInterval,
    PartsPerMillion,
    PositiveFiniteNumber,
    PositiveFloat,
    PositiveInt,
    UniqueNonEmptyNonNegativeIntSequenceLike,
    UniqueNonEmptyNonNegativeIntTuple,
    UnitInterval,
    UnitIntervalOpenLower,
    UnitIntervalOpenUpper,
    ZstdCompressionLevel,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


@beartype
def _accept_positive_float(value: PositiveFloat) -> PositiveFloat:
    return value


@beartype
def _accept_non_empty_non_blank_str_list(
    value: NonEmptyNonBlankStrListLike,
) -> NonEmptyNonBlankStrListLike:
    return value


@beartype
def _accept_non_negative_int_list_list(
    value: NonNegativeIntListListLike,
) -> NonNegativeIntListListLike:
    return value


@beartype
def _accept_non_empty_non_negative_int_list_list(
    value: NonEmptyNonNegativeIntListListLike,
) -> NonEmptyNonNegativeIntListListLike:
    return value


@beartype
def _accept_unique_non_empty_non_negative_int_sequence(
    value: UniqueNonEmptyNonNegativeIntSequenceLike,
) -> UniqueNonEmptyNonNegativeIntSequenceLike:
    return value


@pytest.mark.parametrize("value", [0, 1, 1_000_000])
def test_non_negative_int_accepts_strict_non_negative_integers(value: int) -> None:
    parsed = NonNegativeInt.parse(value)
    assert_type(parsed, NonNegativeInt)
    assert parsed == value
    assert type(parsed) is int


@pytest.mark.parametrize("value", [-1, True, False, 1.0])
def test_non_negative_int_rejects_other_values(value: object) -> None:
    with pytest.raises(TypeError):
        NonNegativeInt.parse(value)


@pytest.mark.parametrize("value", [1, 10])
def test_positive_int_accepts_strict_positive_integers(value: int) -> None:
    assert PositiveInt.parse(value) == value


@pytest.mark.parametrize("value", [0, -1, True])
def test_positive_int_rejects_non_positive_or_bool(value: object) -> None:
    with pytest.raises(TypeError):
        PositiveInt.parse(value)


@pytest.mark.parametrize("value", [0.25, 1.0, float("inf")])
def test_positive_float_accepts_positive_floats(value: float) -> None:
    parsed = PositiveFloat.parse(value)
    assert_type(parsed, PositiveFloat)
    assert parsed == value


@pytest.mark.parametrize("value", [0.0, -1.0, 1, True, False, float("nan")])
def test_positive_float_rejects_non_positive_non_float_or_nan(value: object) -> None:
    with pytest.raises(TypeError):
        PositiveFloat.parse(value)


@pytest.mark.parametrize("value", [0.0, 0.25, float("inf")])
def test_non_negative_float_accepts_non_negative_floats(value: float) -> None:
    parsed = NonNegativeFloat.parse(value)
    assert_type(parsed, NonNegativeFloat)
    assert parsed == value


@pytest.mark.parametrize("value", [-1.0, 0, True, False, float("nan")])
def test_non_negative_float_rejects_negative_non_float_or_nan(value: object) -> None:
    with pytest.raises(TypeError):
        NonNegativeFloat.parse(value)


@pytest.mark.parametrize("value", [2, 10])
def test_at_least_two_int_includes_its_lower_bound(value: int) -> None:
    assert AtLeastTwoInt.parse(value) == value


@pytest.mark.parametrize("value", [1, 0, -1, True])
def test_at_least_two_int_rejects_values_below_two_or_bool(value: object) -> None:
    with pytest.raises(TypeError):
        AtLeastTwoInt.parse(value)


@pytest.mark.parametrize("value", [0, 1_000_000])
def test_parts_per_million_includes_both_bounds(value: int) -> None:
    assert PartsPerMillion.parse(value) == value


@pytest.mark.parametrize("value", [-1, 1_000_001, True])
def test_parts_per_million_rejects_out_of_range_or_bool(value: object) -> None:
    with pytest.raises(TypeError):
        PartsPerMillion.parse(value)


@pytest.mark.parametrize("value", [1, 1.0, 0.25, 10**100])
def test_positive_finite_number_accepts_finite_positive_numbers(value: int | float) -> None:
    assert PositiveFiniteNumber.parse(value) == value


@pytest.mark.parametrize(
    "value",
    [0, 0.0, -1, True, False, float("nan"), float("inf"), 10**1000],
)
def test_positive_finite_number_rejects_other_values(value: object) -> None:
    with pytest.raises(TypeError):
        PositiveFiniteNumber.parse(value)


@pytest.mark.parametrize("value", [-10, -1.5, 0, 0.0, 2.5, 10**100])
def test_finite_number_accepts_finite_non_boolean_numbers(value: int | float) -> None:
    assert FiniteNumber.parse(value) == value


@pytest.mark.parametrize(
    "value",
    [True, False, float("nan"), float("inf"), float("-inf"), 10**1000, "1"],
)
def test_finite_number_rejects_non_finite_or_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError):
        FiniteNumber.parse(value)


@pytest.mark.parametrize("value", [0, 0.0, 0.25, 1, 1.0])
def test_unit_interval_includes_both_bounds(value: int | float) -> None:
    assert UnitInterval.parse(value) == value


@pytest.mark.parametrize(
    "value",
    [-1, -0.1, 1.1, 2, True, False, float("nan"), float("inf"), 10**1000],
)
def test_unit_interval_rejects_values_outside_its_numeric_domain(value: object) -> None:
    with pytest.raises(TypeError):
        UnitInterval.parse(value)


@PROPERTY_SETTINGS
@given(st.lists(st.integers(min_value=0), min_size=1).map(tuple))
def test_non_empty_non_negative_int_tuple_accepts_complete_valid_domain(
    value: tuple[int, ...],
) -> None:
    parsed = NonEmptyNonNegativeIntTuple.parse(value)
    assert_type(parsed, NonEmptyNonNegativeIntTuple)
    assert isinstance(value, NonEmptyNonNegativeIntTuple)
    assert parsed == value
    assert type(parsed) is tuple


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.just(()),
        st.tuples(
            st.lists(st.integers(min_value=0)),
            st.one_of(
                st.integers(max_value=-1),
                st.booleans(),
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(),
                st.none(),
            ),
            st.lists(st.integers(min_value=0)),
        ).map(lambda parts: (*parts[0], parts[1], *parts[2])),
        st.lists(st.integers(min_value=0), min_size=1),
    )
)
def test_non_empty_non_negative_int_tuple_rejects_complete_invalid_classes(
    value: object,
) -> None:
    assert not isinstance(value, NonEmptyNonNegativeIntTuple)
    with pytest.raises(TypeError):
        NonEmptyNonNegativeIntTuple.parse(value)


@PROPERTY_SETTINGS
@given(st.lists(st.text(min_size=1).filter(str.strip), min_size=1))
def test_non_empty_non_blank_str_list_accepts_every_valid_item(
    value: list[str],
) -> None:
    assert _accept_non_empty_non_blank_str_list(value) is value


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.just([]),
        st.tuples(
            st.lists(st.text(min_size=1).filter(str.strip)),
            st.sampled_from(("", " ", "\t", "\n")),
            st.lists(st.text(min_size=1).filter(str.strip)),
        ).map(lambda parts: [*parts[0], parts[1], *parts[2]]),
        st.lists(st.text(min_size=1).filter(str.strip), min_size=1).map(tuple),
    )
)
def test_non_empty_non_blank_str_list_rejects_every_invalid_shape_or_item(
    value: object,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        _accept_non_empty_non_blank_str_list(value)  # ty: ignore[invalid-argument-type]


@PROPERTY_SETTINGS
@given(st.lists(st.lists(st.integers(min_value=0))))
def test_non_negative_int_list_list_accepts_empty_outer_and_inner_lists(
    value: list[list[int]],
) -> None:
    assert _accept_non_negative_int_list_list(value) is value
    if value:
        assert _accept_non_empty_non_negative_int_list_list(value) is value
    else:
        with pytest.raises(BeartypeCallHintParamViolation):
            _accept_non_empty_non_negative_int_list_list(value)


@PROPERTY_SETTINGS
@given(
    prefix=st.lists(st.lists(st.integers(min_value=0))),
    invalid=st.one_of(
        st.integers(max_value=-1),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    ),
    suffix=st.lists(st.integers(min_value=0)),
)
def test_non_negative_int_list_list_rejects_invalid_items_at_every_position(
    prefix: list[list[int]],
    invalid: object,
    suffix: list[int],
) -> None:
    value: list[object] = [*prefix, [*suffix, invalid]]
    with pytest.raises(BeartypeCallHintParamViolation):
        _accept_non_negative_int_list_list(value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(BeartypeCallHintParamViolation):
        _accept_non_empty_non_negative_int_list_list(value)  # ty: ignore[invalid-argument-type]


@PROPERTY_SETTINGS
@given(
    strings=st.lists(st.text(min_size=1).filter(str.strip), min_size=1),
    token_lists=st.lists(st.lists(st.integers(min_value=0)), min_size=1),
)
def test_refined_list_boundaries_do_not_consume_global_random_state(
    strings: list[str],
    token_lists: list[list[int]],
) -> None:
    state = random.getstate()
    _accept_non_empty_non_blank_str_list(strings)
    _accept_non_negative_int_list_list(token_lists)
    _accept_non_empty_non_negative_int_list_list(token_lists)
    assert random.getstate() == state


@PROPERTY_SETTINGS
@given(st.integers())
def test_non_negative_int_predicate_matches_the_complete_integer_domain(
    value: int,
) -> None:
    expected = value >= 0
    assert isinstance(value, NonNegativeInt) is expected
    if expected:
        parsed = NonNegativeInt.parse(value)
        assert parsed == value
        assert type(parsed) is int
    else:
        with pytest.raises(TypeError):
            NonNegativeInt.parse(value)


@PROPERTY_SETTINGS
@given(st.integers())
def test_positive_int_predicate_matches_the_complete_integer_domain(value: int) -> None:
    expected = value > 0
    assert isinstance(value, PositiveInt) is expected
    if expected:
        parsed = PositiveInt.parse(value)
        assert parsed == value
        assert type(parsed) is int
    else:
        with pytest.raises(TypeError):
            PositiveInt.parse(value)


@PROPERTY_SETTINGS
@given(st.floats(allow_nan=True, allow_infinity=True))
def test_positive_float_predicate_matches_the_complete_float_domain(value: float) -> None:
    expected = value > 0
    assert isinstance(value, PositiveFloat) is expected
    if expected:
        assert _accept_positive_float(value) == value  # ty: ignore[invalid-argument-type]
    else:
        with pytest.raises(BeartypeCallHintParamViolation):
            _accept_positive_float(value)  # ty: ignore[invalid-argument-type]


@PROPERTY_SETTINGS
@given(st.floats(allow_nan=True, allow_infinity=True))
def test_non_negative_float_predicate_matches_the_complete_float_domain(
    value: float,
) -> None:
    expected = value >= 0
    assert isinstance(value, NonNegativeFloat) is expected
    if expected:
        parsed = NonNegativeFloat.parse(value)
        assert parsed == value
        assert type(parsed) is float
    else:
        with pytest.raises(TypeError):
            NonNegativeFloat.parse(value)


@PROPERTY_SETTINGS
@given(st.integers())
def test_at_least_two_int_predicate_matches_the_complete_integer_domain(
    value: int,
) -> None:
    expected = value >= 2
    assert isinstance(value, AtLeastTwoInt) is expected
    if expected:
        parsed = AtLeastTwoInt.parse(value)
        assert parsed == value
        assert type(parsed) is int
    else:
        with pytest.raises(TypeError):
            AtLeastTwoInt.parse(value)


@PROPERTY_SETTINGS
@given(st.integers())
def test_parts_per_million_predicate_matches_the_complete_integer_domain(
    value: int,
) -> None:
    expected = 0 <= value <= 1_000_000
    assert isinstance(value, PartsPerMillion) is expected
    if expected:
        parsed = PartsPerMillion.parse(value)
        assert parsed == value
        assert type(parsed) is int
    else:
        with pytest.raises(TypeError):
            PartsPerMillion.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_positive_finite_number_matches_the_complete_supported_domain(
    value: object,
) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
            and float("-inf") < float(value) < float("inf")
        )
    except OverflowError:
        expected = False
    assert isinstance(value, PositiveFiniteNumber) is expected
    if expected:
        assert PositiveFiniteNumber.parse(value) == value
    else:
        with pytest.raises(TypeError):
            PositiveFiniteNumber.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_finite_number_matches_the_complete_supported_domain(value: object) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, FiniteNumber) is expected
    if expected:
        assert FiniteNumber.parse(value) == value
    else:
        with pytest.raises(TypeError):
            FiniteNumber.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_unit_interval_matches_the_complete_supported_domain(value: object) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value <= 1
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, UnitInterval) is expected
    if expected:
        assert UnitInterval.parse(value) == value
    else:
        with pytest.raises(TypeError):
            UnitInterval.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_open_unit_interval_matches_the_complete_supported_domain(
    value: object,
) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value < 1
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, OpenUnitInterval) is expected
    if expected:
        assert OpenUnitInterval.parse(value) == value
    else:
        with pytest.raises(TypeError):
            OpenUnitInterval.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_open_lower_unit_interval_matches_the_complete_supported_domain(
    value: object,
) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, UnitIntervalOpenLower) is expected
    if expected:
        assert UnitIntervalOpenLower.parse(value) == value
    else:
        with pytest.raises(TypeError):
            UnitIntervalOpenLower.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_open_upper_unit_interval_matches_the_complete_supported_domain(
    value: object,
) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value < 1
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, UnitIntervalOpenUpper) is expected
    if expected:
        assert UnitIntervalOpenUpper.parse(value) == value
    else:
        with pytest.raises(TypeError):
            UnitIntervalOpenUpper.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.lists(st.integers(min_value=0), min_size=1, unique=True).map(tuple),
        st.just(()),
        st.lists(st.integers(min_value=0), min_size=1).map(
            lambda values: (*values, values[0])
        ),
        st.tuples(
            st.lists(st.integers(min_value=0)),
            st.one_of(
                st.integers(max_value=-1),
                st.booleans(),
                st.floats(allow_nan=True, allow_infinity=True),
                st.text(),
                st.none(),
            ),
            st.lists(st.integers(min_value=0)),
        ).map(lambda parts: (*parts[0], parts[1], *parts[2])),
        st.lists(st.integers(min_value=0), min_size=1, unique=True),
    )
)
def test_unique_non_empty_non_negative_int_tuple_matches_complete_domain(
    value: object,
) -> None:
    expected = (
        isinstance(value, tuple)
        and bool(value)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
        and len(value) == len(set(value))
    )
    assert isinstance(value, UniqueNonEmptyNonNegativeIntTuple) is expected
    if expected:
        assert UniqueNonEmptyNonNegativeIntTuple.parse(value) == value
    else:
        with pytest.raises(TypeError):
            UniqueNonEmptyNonNegativeIntTuple.parse(value)


@PROPERTY_SETTINGS
@given(
    values=st.lists(
        st.one_of(
            st.integers(min_value=-100, max_value=100),
            st.booleans(),
            st.none(),
            st.text(max_size=8),
        ),
        max_size=20,
    ),
    as_tuple=st.booleans(),
)
def test_unique_non_empty_non_negative_int_sequence_matches_complete_domain(
    values: list[object],
    as_tuple: bool,
) -> None:
    sequence = tuple(values) if as_tuple else values
    expected = (
        bool(values)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        )
        and len(values) == len(set(values))
    )
    state = random.getstate()
    if expected:
        assert (
            _accept_unique_non_empty_non_negative_int_sequence(sequence)  # ty: ignore[invalid-argument-type]
            is sequence
        )
    else:
        with pytest.raises(BeartypeCallHintParamViolation):
            _accept_unique_non_empty_non_negative_int_sequence(sequence)  # ty: ignore[invalid-argument-type]
    assert random.getstate() == state


@PROPERTY_SETTINGS
@given(st.text())
def test_non_blank_str_matches_the_complete_string_domain(value: str) -> None:
    expected = bool(value.strip())
    assert isinstance(value, NonBlankStr) is expected
    if expected:
        assert NonBlankStr.parse(value) == value
    else:
        with pytest.raises(TypeError):
            NonBlankStr.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
    )
)
def test_non_blank_str_rejects_every_non_string_class(value: object) -> None:
    assert not isinstance(value, NonBlankStr)
    with pytest.raises(TypeError):
        NonBlankStr.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.none(),
        st.text(),
    )
)
def test_non_negative_finite_number_matches_supported_domain(value: object) -> None:
    try:
        expected = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
            and float("-inf") < float(value) < float("inf")
        )
    except (OverflowError, TypeError):
        expected = False
    assert isinstance(value, NonNegativeFiniteNumber) is expected
    if expected:
        assert NonNegativeFiniteNumber.parse(value) == value
    else:
        with pytest.raises(TypeError):
            NonNegativeFiniteNumber.parse(value)


@PROPERTY_SETTINGS
@given(st.text(min_size=64, max_size=64))
def test_lowercase_sha256_matches_exact_text_domain(value: str) -> None:
    expected = all(char in "0123456789abcdef" for char in value)
    assert isinstance(value, LowercaseSha256) is expected
    if expected:
        assert LowercaseSha256.parse(value) == value
    else:
        with pytest.raises(TypeError):
            LowercaseSha256.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.text(max_size=63),
        st.text(min_size=65, max_size=80),
        st.integers(),
        st.booleans(),
        st.none(),
    )
)
def test_lowercase_sha256_rejects_wrong_lengths_and_non_strings(
    value: object,
) -> None:
    assert not isinstance(value, LowercaseSha256)
    with pytest.raises(TypeError):
        LowercaseSha256.parse(value)


@pytest.mark.parametrize("value", ["0" * 64, "abcdef0123456789" * 4])
def test_lowercase_sha256_accepts_canonical_digests(value: str) -> None:
    assert isinstance(value, LowercaseSha256)
    assert LowercaseSha256.parse(value) == value


@PROPERTY_SETTINGS
@given(st.integers())
def test_zstd_compression_level_matches_complete_integer_domain(value: int) -> None:
    expected = 1 <= value <= 22
    assert isinstance(value, ZstdCompressionLevel) is expected
    if expected:
        assert ZstdCompressionLevel.parse(value) == value
    else:
        with pytest.raises(TypeError):
            ZstdCompressionLevel.parse(value)


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_zstd_compression_level_rejects_every_non_integer_class(
    value: object,
) -> None:
    assert not isinstance(value, ZstdCompressionLevel)
    with pytest.raises(TypeError):
        ZstdCompressionLevel.parse(value)


@PROPERTY_SETTINGS
@given(st.one_of(st.booleans(), st.integers(max_value=-1)))
def test_beartype_rejects_every_invalid_non_negative_policy_value(
    value: int,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        AdmissionPolicy(
            store_free_reserve_bytes=value  # ty: ignore[invalid-argument-type]
        )


@PROPERTY_SETTINGS
@given(st.one_of(st.booleans(), st.integers(max_value=0)))
def test_beartype_rejects_every_invalid_positive_policy_value(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        AdmissionPolicy(
            worker_packed_ratio_denominator=value  # ty: ignore[invalid-argument-type]
        )


@PROPERTY_SETTINGS
@given(
    st.one_of(
        st.booleans(),
        st.integers(max_value=-1),
        st.integers(min_value=1_000_001),
    )
)
def test_beartype_rejects_every_invalid_parts_per_million_policy_value(
    value: int,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        AdmissionPolicy(
            host_reserve_fraction_ppm=value  # ty: ignore[invalid-argument-type]
        )


@PROPERTY_SETTINGS
@given(
    spool_min=st.integers(min_value=0),
    spool_max=st.integers(min_value=0),
    host_min=st.integers(min_value=0),
    host_max=st.integers(min_value=0),
)
def test_admission_policy_relationships_hold_for_all_non_negative_bounds(
    spool_min: int,
    spool_max: int,
    host_min: int,
    host_max: int,
) -> None:
    policy = replace(
        AdmissionPolicy(),
        spool_min_free_reserve_bytes=NonNegativeInt.parse(spool_min),
        spool_max_free_reserve_bytes=NonNegativeInt.parse(spool_max),
        host_min_reserve_bytes=NonNegativeInt.parse(host_min),
        host_max_reserve_bytes=NonNegativeInt.parse(host_max),
    )
    if spool_min > spool_max:
        with pytest.raises(ValueError, match="spool minimum reserve"):
            policy.validate()
    elif host_min > host_max:
        with pytest.raises(ValueError, match="host minimum reserve"):
            policy.validate()
    else:
        policy.validate()
