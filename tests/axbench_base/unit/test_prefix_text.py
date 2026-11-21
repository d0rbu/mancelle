"""Exact unit contract for shared decoded-text prefix bounds and windowing."""

from __future__ import annotations

from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base.prefix_text import (
    validate_prefix_char_bounds,
    window_prefix_text,
)
from steering.eval.axbench_base.types import AxBenchBaseError


@pytest.mark.parametrize(
    ("min_prefix_chars", "max_prefix_chars"),
    [
        (0, 1),
        (0, 2048),
        (1, 1),
        (1, 2048),
        (100, 2048),
        (2048, 2048),
    ],
)
def test_validate_prefix_char_bounds_accepts_closed_integer_domain(
    min_prefix_chars: int,
    max_prefix_chars: int,
) -> None:
    validate_prefix_char_bounds(min_prefix_chars, max_prefix_chars)


@pytest.mark.parametrize(
    "bad_min",
    [
        True,
        False,
        -1,
        -10,
        0.0,
        1.0,
        "0",
        "",
        None,
        [],
    ],
)
def test_validate_prefix_char_bounds_rejects_non_integer_or_negative_minimum(
    bad_min: object,
) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"min_prefix_chars",
    ) as exc_info:
        validate_prefix_char_bounds(cast(Any, bad_min), 2048)

    assert repr(bad_min) in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_max",
    [
        True,
        False,
        -10,
        -1,
        0,
        0.0,
        1.0,
        "1",
        "",
        None,
        [],
    ],
)
def test_validate_prefix_char_bounds_rejects_non_integer_or_nonpositive_maximum(
    bad_max: object,
) -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"max_prefix_chars",
    ) as exc_info:
        validate_prefix_char_bounds(0, cast(Any, bad_max))

    assert repr(bad_max) in str(exc_info.value)


@pytest.mark.parametrize(
    ("min_prefix_chars", "max_prefix_chars"),
    [
        (2, 1),
        (101, 100),
        (2049, 2048),
    ],
)
def test_validate_prefix_char_bounds_rejects_inverted_closed_interval(
    min_prefix_chars: int,
    max_prefix_chars: int,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=(
            rf"min_prefix_chars must be <= max_prefix_chars, got "
            rf"{min_prefix_chars} > {max_prefix_chars}"
        ),
    ):
        validate_prefix_char_bounds(min_prefix_chars, max_prefix_chars)


@pytest.mark.parametrize(
    ("full_text", "end_char", "max_prefix_chars", "expected"),
    [
        ("", 0, 1, ""),
        ("abcdef", 0, 4, ""),
        ("abcdef", 2, 4, "ab"),
        ("abcdef", 4, 4, "abcd"),
        ("abcdef", 6, 4, "cdef"),
        ("abcdef", 5, 1, "e"),
        ("abcdef", 6, 100, "abcdef"),
        ("0123456789", 7, 4, "3456"),
    ],
)
def test_window_prefix_text_uses_tail_ending_at_requested_character(
    full_text: str,
    end_char: int,
    max_prefix_chars: int,
    expected: str,
) -> None:
    actual = window_prefix_text(full_text, end_char, max_prefix_chars)

    assert actual == expected
    assert len(actual) <= max_prefix_chars
    assert actual == full_text[max(0, end_char - max_prefix_chars) : end_char]


def test_window_prefix_text_counts_unicode_code_points_not_utf8_bytes() -> None:
    full_text = "head-😀e\u0301漢Z"

    actual = window_prefix_text(
        full_text,
        end_char=len(full_text),
        max_prefix_chars=4,
    )

    # The combining accent remains its own Python character. The four retained
    # code points are e, COMBINING ACUTE ACCENT, CJK ideograph, and Z.
    assert actual == "e\u0301漢Z"
    assert len(actual) == 4
    assert len(actual.encode("utf-8")) > 4


def test_window_prefix_text_does_not_mutate_or_normalize_input() -> None:
    decomposed = "prefix-Cafe\u0301"
    original = decomposed[:]

    actual = window_prefix_text(decomposed, len(decomposed), 5)

    assert decomposed == original
    assert actual == "Cafe\u0301"
    assert actual != "Café"


def test_window_prefix_text_is_a_character_slice_not_a_word_or_sentence_slice() -> None:
    full_text = "first sentence. second sentence"

    actual = window_prefix_text(full_text, len(full_text), 11)

    assert actual == "nd sentence"
    assert not actual.startswith("second")
