"""Reusable refined value types validated at runtime by ``beartype``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

from beartype.vale import Is
from phantom import Phantom

__all__ = [
    "AtLeastTwoInt",
    "AtLeastTwoIntLike",
    "BinaryInt",
    "BinaryIntLike",
    "FiniteNumber",
    "FiniteNumberLike",
    "FivePointScore",
    "FivePointScoreLike",
    "LowercaseSha256",
    "LowercaseSha256Like",
    "NonBlankPathComponent",
    "NonBlankPathComponentLike",
    "NonBlankStr",
    "NonBlankStrLike",
    "NonEmptyStr",
    "NonEmptyStrLike",
    "NonEmptyNonBlankStrListLike",
    "NonEmptyNonNegativeIntListLike",
    "NonEmptyNonNegativeIntListListLike",
    "NonEmptyNonNegativeIntMappingListLike",
    "NonEmptyNonNegativeIntTuple",
    "NonEmptyNonNegativeIntTupleLike",
    "NonNegativeFiniteNumber",
    "NonNegativeFiniteNumberLike",
    "NonNegativeFiniteNumberTuple",
    "NonNegativeFiniteNumberTupleLike",
    "NonNegativeFloat",
    "NonNegativeInt",
    "NonNegativeIntListLike",
    "NonNegativeIntListListLike",
    "NonNegativeIntLike",
    "NonNegativeIntMappingItems",
    "NonNegativeIntMappingLike",
    "NonNegativeIntSetLike",
    "NonNegativeIntTuple",
    "NonNegativeIntTupleLike",
    "OpenUnitInterval",
    "OpenUnitIntervalLike",
    "OneToFiveInt",
    "OneToFiveIntLike",
    "PartsPerMillion",
    "PositiveFloat",
    "PositiveFiniteNumber",
    "PositiveFiniteNumberLike",
    "PositiveInt",
    "PositiveIntLike",
    "StrictInt",
    "StrictIntLike",
    "SingletonNonNegativeIntTuple",
    "SingletonNonNegativeIntTupleLike",
    "StrTuple",
    "StrTupleLike",
    "UnitInterval",
    "UnitIntervalLike",
    "UnitIntervalOpenLower",
    "UnitIntervalOpenLowerLike",
    "UnitIntervalOpenUpper",
    "UnitIntervalOpenUpperLike",
    "UniqueNonEmptyNonNegativeIntTuple",
    "UniqueNonEmptyNonNegativeIntListLike",
    "UniqueNonEmptyNonNegativeIntSequenceLike",
    "UniqueNonEmptyNonNegativeIntTupleLike",
    "UniqueNonEmptyNonBlankStrTuple",
    "UniqueNonEmptyNonBlankStrTupleLike",
    "UniqueNonEmptyNonNegativeFiniteNumberTuple",
    "UniqueNonEmptyNonNegativeFiniteNumberTupleLike",
    "UniqueNonEmptyPositiveIntTuple",
    "UniqueNonEmptyPositiveIntTupleLike",
    "ZstdCompressionLevel",
    "ZeroInt",
    "ZeroIntLike",
]


def _is_at_least_two_int(value: int) -> bool:
    return not isinstance(value, bool) and value >= 2


def _is_binary_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_five_point_score(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 5


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_negative_float(value: float) -> bool:
    return isinstance(value, float) and value >= 0


def _is_non_negative_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _is_non_negative_finite_number_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(_is_non_negative_finite_number(item) for item in value)


def _is_non_blank_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_str_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, str) for item in value)


def _is_non_blank_path_component(value: object) -> bool:
    return (
        isinstance(value, str)
        and _is_non_blank_str(value)
        and "/" not in value
        and "\\" not in value
        and ".." not in value
    )


def _is_non_empty_non_negative_int_tuple(value: tuple[int, ...]) -> bool:
    return bool(value) and _is_non_negative_int_tuple(value)


def _is_non_negative_int_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(_is_non_negative_int(item) for item in value)


def _is_singleton_non_negative_int_tuple(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 1 and _is_non_negative_int(value[0])


def _is_non_negative_int_mapping_items(value: object) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and _is_non_negative_int(item[0])
        and _is_non_negative_int(item[1])
        for item in value
    )


def _is_non_empty_non_negative_int_mapping_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, dict)
            and isinstance(
                tuple(item.items()),
                NonNegativeIntMappingItems,
            )
            for item in value
        )
    )


def _is_non_empty_non_blank_str_list(value: object) -> bool:
    return (
        isinstance(value, list) and bool(value) and all(_is_non_blank_str(item) for item in value)
    )


def _is_non_negative_int_list_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, list) and all(_is_non_negative_int(token_id) for token_id in item)
        for item in value
    )


def _is_non_empty_non_negative_int_list_list(value: object) -> bool:
    return bool(value) and _is_non_negative_int_list_list(value)


def _is_unique_non_empty_non_negative_int_tuple(value: tuple[int, ...]) -> bool:
    return _is_non_empty_non_negative_int_tuple(value) and len(value) == len(set(value))


def _is_unique_non_empty_non_blank_str_tuple(value: tuple[str, ...]) -> bool:
    return (
        bool(value)
        and all(_is_non_blank_str(item) for item in value)
        and len(value) == len(set(value))
    )


def _is_unique_non_empty_non_negative_finite_number_tuple(
    value: tuple[int | float, ...],
) -> bool:
    return (
        bool(value)
        and all(_is_non_negative_finite_number(item) for item in value)
        and len(value) == len(set(value))
    )


def _is_unique_non_empty_positive_int_tuple(value: tuple[int, ...]) -> bool:
    return (
        bool(value)
        and all(_is_positive_int(item) for item in value)
        and len(value) == len(set(value))
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_float(value: float) -> bool:
    return isinstance(value, float) and value > 0


def _is_positive_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _is_parts_per_million(value: int) -> bool:
    return not isinstance(value, bool) and 0 <= value <= 1_000_000


def _is_open_unit_interval(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 < value < 1
    except OverflowError:
        return False


def _is_one_to_five_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5


def _is_unit_interval(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        return False


def _is_unit_interval_open_lower(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 < value <= 1
    except OverflowError:
        return False


def _is_unit_interval_open_upper(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and 0 <= value < 1
    except OverflowError:
        return False


def _is_zstd_compression_level(value: int) -> bool:
    return not isinstance(value, bool) and 1 <= value <= 22


def _is_zero_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


class AtLeastTwoInt(int, Phantom, predicate=_is_at_least_two_int):
    """A strict integer greater than or equal to two (``bool`` excluded)."""


class BinaryInt(int, Phantom, predicate=_is_binary_int):
    """The strict integer zero or one (``bool`` excluded)."""


class FiniteNumber(float, Phantom, predicate=_is_finite_number):
    """A finite ``int`` or ``float`` (``bool`` excluded)."""


class FivePointScore(int, Phantom, predicate=_is_five_point_score):
    """A strict integer score in the inclusive range ``[0, 5]``."""


class LowercaseSha256(str, Phantom, predicate=_is_lowercase_sha256):
    """A 64-character lowercase hexadecimal SHA-256 digest."""


class NonBlankPathComponent(str, Phantom, predicate=_is_non_blank_path_component):
    """A non-blank string without path traversal or separator tokens."""


class NonBlankStr(str, Phantom, predicate=_is_non_blank_str):
    """A string containing at least one non-whitespace character."""


class NonEmptyStr(str, Phantom, predicate=_is_non_empty_str):
    """A string containing at least one character; whitespace is permitted."""


class NonEmptyNonNegativeIntTuple(
    tuple[int, ...],
    Phantom,
    predicate=_is_non_empty_non_negative_int_tuple,
):
    """An immutable non-empty tuple of strict non-negative integers."""


class NonNegativeIntTuple(
    tuple[int, ...],
    Phantom,
    predicate=_is_non_negative_int_tuple,
):
    """An immutable, possibly empty tuple of strict non-negative integers."""


class NonNegativeIntMappingItems(
    tuple[tuple[int, int], ...],
    Phantom,
    predicate=_is_non_negative_int_mapping_items,
):
    """Immutable items from a non-negative integer mapping."""


class NonNegativeFiniteNumber(
    float,
    Phantom,
    predicate=_is_non_negative_finite_number,
):
    """A finite non-negative ``int`` or ``float`` (``bool`` excluded)."""


class NonNegativeFiniteNumberTuple(
    tuple[int | float, ...],
    Phantom,
    predicate=_is_non_negative_finite_number_tuple,
):
    """An immutable, possibly empty tuple of finite non-negative numbers."""


class NonNegativeInt(int, Phantom, predicate=_is_non_negative_int):
    """A strict integer greater than or equal to zero (``bool`` excluded)."""


class NonNegativeFloat(float, Phantom, predicate=_is_non_negative_float):
    """A strict float greater than or equal to zero."""


class PositiveInt(int, Phantom, predicate=_is_positive_int):
    """A strict integer greater than zero (``bool`` excluded)."""


class StrictInt(int, Phantom, predicate=_is_strict_int):
    """An integer that excludes ``bool`` without imposing a range."""


class SingletonNonNegativeIntTuple(
    tuple[int],
    Phantom,
    predicate=_is_singleton_non_negative_int_tuple,
):
    """An immutable singleton tuple containing one non-negative integer."""


class StrTuple(tuple[str, ...], Phantom, predicate=_is_str_tuple):
    """An immutable, possibly empty tuple containing only strings."""


class PositiveFloat(float, Phantom, predicate=_is_positive_float):
    """A strict float greater than zero."""


class PositiveFiniteNumber(float, Phantom, predicate=_is_positive_finite_number):
    """A finite positive ``int`` or ``float`` (``bool`` excluded)."""


class PartsPerMillion(int, Phantom, predicate=_is_parts_per_million):
    """A strict integer in the inclusive range ``[0, 1_000_000]``."""


class OpenUnitInterval(float, Phantom, predicate=_is_open_unit_interval):
    """A finite non-boolean number in ``(0, 1)``."""


class OneToFiveInt(int, Phantom, predicate=_is_one_to_five_int):
    """A strict integer in the inclusive range ``[1, 5]``."""


class UnitInterval(float, Phantom, predicate=_is_unit_interval):
    """A finite ``int`` or ``float`` in the inclusive range ``[0, 1]``."""


class UnitIntervalOpenLower(float, Phantom, predicate=_is_unit_interval_open_lower):
    """A finite non-boolean number in ``(0, 1]``."""


class UnitIntervalOpenUpper(float, Phantom, predicate=_is_unit_interval_open_upper):
    """A finite non-boolean number in ``[0, 1)``."""


class UniqueNonEmptyNonNegativeIntTuple(
    tuple[int, ...],
    Phantom,
    predicate=_is_unique_non_empty_non_negative_int_tuple,
):
    """A non-empty tuple of unique strict non-negative integers."""


class UniqueNonEmptyNonBlankStrTuple(
    tuple[str, ...],
    Phantom,
    predicate=_is_unique_non_empty_non_blank_str_tuple,
):
    """A non-empty tuple of unique non-blank strings."""


class UniqueNonEmptyNonNegativeFiniteNumberTuple(
    tuple[int | float, ...],
    Phantom,
    predicate=_is_unique_non_empty_non_negative_finite_number_tuple,
):
    """A non-empty tuple of unique finite non-negative numbers."""


class UniqueNonEmptyPositiveIntTuple(
    tuple[int, ...],
    Phantom,
    predicate=_is_unique_non_empty_positive_int_tuple,
):
    """A non-empty tuple of unique strict positive integers."""


class ZstdCompressionLevel(int, Phantom, predicate=_is_zstd_compression_level):
    """A strict integer in zstandard's supported compression range ``[1, 22]``."""


class ZeroInt(int, Phantom, predicate=_is_zero_int):
    """The strict integer zero (``bool`` excluded)."""


# Phantom values are structural: ``parse`` returns the original builtin value,
# and ``isinstance(value, PhantomType)`` evaluates the predicate. These aliases
# expose that builtin shape to static checkers while letting beartype reuse the
# exact phantom predicate at runtime. They are intended for declarative ingress
# boundaries that retain builtin values; normalized domain objects can instead
# annotate the nominal phantom type after calling ``parse``.
AtLeastTwoIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, AtLeastTwoInt)],
]
BinaryIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, BinaryInt)],
]
FiniteNumberLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, FiniteNumber)],
]
FivePointScoreLike = Annotated[
    int,
    Is[lambda value: isinstance(value, FivePointScore)],
]
LowercaseSha256Like = Annotated[
    str,
    Is[lambda value: isinstance(value, LowercaseSha256)],
]
NonBlankPathComponentLike = Annotated[
    str,
    Is[lambda value: isinstance(value, NonBlankPathComponent)],
]
NonBlankStrLike = Annotated[
    str,
    Is[lambda value: isinstance(value, NonBlankStr)],
]
NonEmptyStrLike = Annotated[
    str,
    Is[lambda value: isinstance(value, NonEmptyStr)],
]
NonNegativeFiniteNumberLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, NonNegativeFiniteNumber)],
]
NonNegativeIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, NonNegativeInt)],
]
OpenUnitIntervalLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, OpenUnitInterval)],
]
OneToFiveIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, OneToFiveInt)],
]
PositiveFiniteNumberLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, PositiveFiniteNumber)],
]
PositiveIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, PositiveInt)],
]
StrictIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, StrictInt)],
]
ZeroIntLike = Annotated[
    int,
    Is[lambda value: isinstance(value, ZeroInt)],
]
UnitIntervalLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, UnitInterval)],
]
UnitIntervalOpenLowerLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, UnitIntervalOpenLower)],
]
UnitIntervalOpenUpperLike = Annotated[
    int | float,
    Is[lambda value: isinstance(value, UnitIntervalOpenUpper)],
]
# Parameterized container hints make beartype consume Python's global RNG for
# O(1) element sampling. These domains already perform deterministic full
# collection checks through their phantom predicates, so runtime hints use only
# the unsubscripted outer container. Static checkers retain each exact shape.
if TYPE_CHECKING:
    NonEmptyNonBlankStrListLike = list[str]
    NonEmptyNonNegativeIntMappingListLike = list[dict[int, int]]
    NonEmptyNonNegativeIntListLike = list[int]
    NonEmptyNonNegativeIntListListLike = list[list[int]]
    NonEmptyNonNegativeIntTupleLike = tuple[int, ...]
    NonNegativeFiniteNumberTupleLike = tuple[int | float, ...]
    NonNegativeIntListLike = list[int]
    NonNegativeIntListListLike = list[list[int]]
    NonNegativeIntMappingLike = dict[int, int]
    NonNegativeIntSetLike = set[int]
    NonNegativeIntTupleLike = tuple[int, ...]
    SingletonNonNegativeIntTupleLike = tuple[int]
    StrTupleLike = tuple[str, ...]
    UniqueNonEmptyNonNegativeIntListLike = list[int]
    UniqueNonEmptyNonNegativeIntSequenceLike = Sequence[int]
    UniqueNonEmptyNonNegativeIntTupleLike = tuple[int, ...]
    UniqueNonEmptyNonBlankStrTupleLike = tuple[str, ...]
    UniqueNonEmptyNonNegativeFiniteNumberTupleLike = tuple[int | float, ...]
    UniqueNonEmptyPositiveIntTupleLike = tuple[int, ...]
else:
    NonEmptyNonBlankStrListLike = Annotated[
        list,
        Is[_is_non_empty_non_blank_str_list],
    ]
    NonEmptyNonNegativeIntMappingListLike = Annotated[
        list,
        Is[lambda value: _is_non_empty_non_negative_int_mapping_list(value)],
    ]
    NonEmptyNonNegativeIntListLike = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list)
                and isinstance(
                    tuple(value),
                    NonEmptyNonNegativeIntTuple,
                )
            )
        ],
    ]
    NonEmptyNonNegativeIntListListLike = Annotated[
        list,
        Is[_is_non_empty_non_negative_int_list_list],
    ]
    NonEmptyNonNegativeIntTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, NonEmptyNonNegativeIntTuple)],
    ]
    NonNegativeFiniteNumberTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, NonNegativeFiniteNumberTuple)],
    ]
    NonNegativeIntMappingLike = Annotated[
        dict,
        Is[
            lambda value: (
                isinstance(value, dict)
                and isinstance(
                    tuple(value.items()),
                    NonNegativeIntMappingItems,
                )
            )
        ],
    ]
    NonNegativeIntListLike = Annotated[
        list,
        Is[
            lambda value: (
                isinstance(value, list)
                and isinstance(
                    tuple(value),
                    NonNegativeIntTuple,
                )
            )
        ],
    ]
    NonNegativeIntListListLike = Annotated[
        list,
        Is[_is_non_negative_int_list_list],
    ]
    NonNegativeIntSetLike = Annotated[
        set,
        Is[
            lambda value: (
                isinstance(value, set)
                and isinstance(
                    tuple(value),
                    NonNegativeIntTuple,
                )
            )
        ],
    ]
    NonNegativeIntTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, NonNegativeIntTuple)],
    ]
    SingletonNonNegativeIntTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, SingletonNonNegativeIntTuple)],
    ]
    StrTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, StrTuple)],
    ]
    UniqueNonEmptyNonNegativeIntListLike = Annotated[
        list,
        Is[
            lambda value: isinstance(
                tuple(value),
                UniqueNonEmptyNonNegativeIntTuple,
            )
        ],
    ]
    UniqueNonEmptyNonNegativeIntSequenceLike = Annotated[
        Sequence,
        Is[
            lambda value: isinstance(
                tuple(value),
                UniqueNonEmptyNonNegativeIntTuple,
            )
        ],
    ]
    UniqueNonEmptyNonNegativeIntTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, UniqueNonEmptyNonNegativeIntTuple)],
    ]
    UniqueNonEmptyNonBlankStrTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, UniqueNonEmptyNonBlankStrTuple)],
    ]
    UniqueNonEmptyNonNegativeFiniteNumberTupleLike = Annotated[
        tuple,
        Is[
            lambda value: isinstance(
                value,
                UniqueNonEmptyNonNegativeFiniteNumberTuple,
            )
        ],
    ]
    UniqueNonEmptyPositiveIntTupleLike = Annotated[
        tuple,
        Is[lambda value: isinstance(value, UniqueNonEmptyPositiveIntTuple)],
    ]
