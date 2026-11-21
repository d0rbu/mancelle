"""Two-field interp/consistency rating parser."""
from steering.eval.feature_filter.interp_judge import (
    INTERP_JUDGE_MAX_SCORE,
    InterpJudgeRating,
    parse_interp_rating,
)


def test_bracketed_format() -> None:
    assert parse_interp_rating("Interpretability: [[3]]\nConsistency: [[5]]") == InterpJudgeRating(3, 5)


def test_bare_format() -> None:
    assert parse_interp_rating("Interpretability: 3\nConsistency: 5") == InterpJudgeRating(3, 5)


def test_mixed_formats() -> None:
    assert parse_interp_rating("Interpretability: [[0]]\nConsistency: 4") == InterpJudgeRating(0, 4)


def test_max_score_value_5() -> None:
    assert INTERP_JUDGE_MAX_SCORE == 5
    assert parse_interp_rating("Interpretability: [[5]]\nConsistency: [[5]]") == InterpJudgeRating(5, 5)


def test_zero_accepted() -> None:
    assert parse_interp_rating("Interpretability: [[0]]\nConsistency: [[0]]") == InterpJudgeRating(0, 0)


def test_out_of_range_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[7]]\nConsistency: [[2]]") is None


def test_negative_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[-1]]\nConsistency: [[2]]") is None


def test_decimal_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[3.5]]\nConsistency: [[2]]") is None


def test_missing_field_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[3]]") is None
    assert parse_interp_rating("Consistency: [[3]]") is None


def test_both_missing_returns_none() -> None:
    assert parse_interp_rating("hello world") is None


def test_multiple_occurrences_last_wins() -> None:
    text = """
    Interpretability: [[1]]
    Consistency: [[1]]
    ... model reconsiders ...
    Interpretability: [[4]]
    Consistency: [[5]]
    """
    assert parse_interp_rating(text) == InterpJudgeRating(4, 5)


def test_case_insensitive_labels() -> None:
    assert parse_interp_rating("INTERPRETABILITY: [[3]]\nCONSISTENCY: [[5]]") == InterpJudgeRating(3, 5)


def test_half_open_bracket_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[3\nConsistency: [[5]]") is None
    assert parse_interp_rating("Interpretability: 3]]\nConsistency: [[5]]") is None
    assert parse_interp_rating("Interpretability: [3]]\nConsistency: [[5]]") is None
    assert parse_interp_rating("Interpretability: [[3]\nConsistency: [[5]]") is None


def test_inconsistency_word_does_not_match_consistency() -> None:
    # "inconsistency" must NOT satisfy the consistency field
    assert parse_interp_rating("Interpretability: [[3]]\nThe inconsistency: [[2]]") is None


def test_bracketed_six_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[6]]\nConsistency: [[5]]") is None


def test_bare_negative_returns_none() -> None:
    assert parse_interp_rating("Interpretability: -1\nConsistency: 3") is None


def test_empty_brackets_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[]]\nConsistency: [[3]]") is None


def test_extra_content_in_brackets_returns_none() -> None:
    assert parse_interp_rating("Interpretability: [[3 extra]]\nConsistency: [[3]]") is None


def test_missing_colon_returns_none() -> None:
    assert parse_interp_rating("Interpretability 3\nConsistency 3") is None


def test_tabs_and_multispace_ok() -> None:
    assert parse_interp_rating("Interpretability:\t\t[[3]]\nConsistency:   [[5]]") == InterpJudgeRating(3, 5)


def test_newline_between_label_and_value_rejected() -> None:
    # We don't want to cross newlines — judge emits one label per line.
    assert parse_interp_rating("Interpretability:\n[[3]]\nConsistency: [[5]]") is None


def test_label_suffix_does_not_match() -> None:
    # "Interpretability_score" should not satisfy "interpretability"
    assert parse_interp_rating("Interpretability_score: [[3]]\nConsistency: [[4]]") is None


def test_score_word_between_label_and_colon_rejected() -> None:
    # "Interpretability score: 3" — additional token breaks the label-then-colon shape
    assert parse_interp_rating("Interpretability score: [[3]]\nConsistency: [[4]]") is None


def test_double_label_collision() -> None:
    # If a judge repeats the same label twice on the same line (unusual but possible)
    # finditer still picks the last match per label; semantic check.
    assert parse_interp_rating("Interpretability: [[1]] Interpretability: [[4]]\nConsistency: [[3]]") == InterpJudgeRating(4, 3)
