"""Slice 16 audit — pin ``parse_axbench_rating`` regex branches.

Map P1-5 walked through the regex's lookahead semantics and noted the
risk of a future contributor weakening the lookaheads. This test
locks in the documented behaviour for each branch:

- ``"Rating: [[N]]"`` and ``"Rating: N"`` for valid scores 0/1/2.
- Multi-rating responses use the LAST match.
- Decimals (``Rating: 1.5``) reject (must not coerce to int).
- Word-suffix (``Rating: 1abc``) rejects.
- ``None`` input is distinct from ``0`` (the legacy bug fixed by
  ``judge_parser.py``).
- Out-of-range values (negatives, > JUDGE_MAX_SCORE) reject.
- Empty / whitespace-only / no-match inputs return ``None``.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.judge_parser import (
    JUDGE_MAX_SCORE,
    parse_axbench_rating,
)


@pytest.mark.parametrize("score", [0, 1, 2])
def test_double_bracketed_valid_score(score: int) -> None:
    """``Rating: [[N]]`` for valid N parses cleanly."""
    out = parse_axbench_rating(f"Rating: [[{score}]]")
    assert out == score


@pytest.mark.parametrize("score", [0, 1, 2])
def test_unbracketed_valid_score(score: int) -> None:
    """``Rating: N`` for valid N parses cleanly."""
    out = parse_axbench_rating(f"Rating: {score}")
    assert out == score


def test_none_input_returns_none() -> None:
    """The legacy bug fixed by judge_parser.parse_axbench_rating."""
    assert parse_axbench_rating(None) is None


def test_empty_input_returns_none() -> None:
    """Empty string returns ``None`` (not 0)."""
    assert parse_axbench_rating("") is None


def test_whitespace_only_returns_none() -> None:
    assert parse_axbench_rating("   \n\t  ") is None


def test_no_match_returns_none() -> None:
    assert parse_axbench_rating("Hello world") is None


def test_decimal_rejected() -> None:
    """``Rating: 1.5`` does NOT silently coerce to 1."""
    out = parse_axbench_rating("Rating: 1.5")
    # Per regex docstring: lookahead ``(?!\.\d)`` blocks decimal forms.
    assert out is None


def test_word_suffix_rejected() -> None:
    """``Rating: 1abc`` rejects — lookahead ``(?!\\w)``."""
    out = parse_axbench_rating("Rating: 1abc")
    assert out is None


def test_negative_rejected() -> None:
    """Negative scores reject (out of [0, JUDGE_MAX_SCORE])."""
    out = parse_axbench_rating("Rating: -1")
    assert out is None


def test_out_of_range_above_max_rejected() -> None:
    """Scores > JUDGE_MAX_SCORE return ``None``."""
    out = parse_axbench_rating(f"Rating: {JUDGE_MAX_SCORE + 1}")
    assert out is None


def test_multi_rating_last_match_wins() -> None:
    """When multiple ``Rating: N`` lines appear, the LAST is taken."""
    text = "Rating: 0\nSecond opinion\nRating: 2"
    out = parse_axbench_rating(text)
    assert out == 2


def test_judge_max_score_is_two() -> None:
    """Pin the documented maximum score."""
    assert JUDGE_MAX_SCORE == 2


def test_bare_digit_no_label_rejected() -> None:
    """A bare digit without 'Rating:' prefix returns ``None`` — pinned in
    the regex docstring as the bug-fix that closed the legacy
    ``score=None vs 0`` confusion.
    """
    assert parse_axbench_rating("2") is None
