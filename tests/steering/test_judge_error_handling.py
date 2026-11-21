"""Tests for judge error handling (Tasks 18-19).

Validates that parse_judge_response and parse_judge_yaml raise
ParseError instead of returning None.
"""

from __future__ import annotations

import textwrap

import pytest

from errors import ParseError

# ---------------------------------------------------------------------------
# Task 18: completion_judge — parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    """parse_judge_response must raise ParseError for unparseable responses."""

    def test_valid_response_parses(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        result = parse_judge_response("coherence: 4\nfeature_alignment: 3")
        assert result.coherence == 4
        assert result.feature_alignment == 3

    def test_valid_with_extra_text(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        result = parse_judge_response(
            "The completion is good.\ncoherence: 5\nfeature_alignment: 2\nOverall decent."
        )
        assert result.coherence == 5
        assert result.feature_alignment == 2

    def test_missing_scores_raises(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        with pytest.raises(ParseError, match="missing score pattern"):
            parse_judge_response("I refuse to rate this")

    def test_empty_response_raises(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        with pytest.raises(ParseError, match="missing score pattern"):
            parse_judge_response("")

    def test_partial_response_raises(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        with pytest.raises(ParseError, match="missing score pattern"):
            parse_judge_response("coherence: 4")

    def test_out_of_range_scores_clamped(self) -> None:
        from steering.eval.completion_judge import parse_judge_response

        result = parse_judge_response("coherence: 7\nfeature_alignment: 0")
        assert result.coherence == 5  # clamped to max
        assert result.feature_alignment == 1  # clamped to min


# ---------------------------------------------------------------------------
# Task 19: harmfulness_judge — parse_judge_yaml
# ---------------------------------------------------------------------------


class TestParseJudgeYaml:
    """parse_judge_yaml must raise ParseError with field-specific messages."""

    def test_valid_yaml_parses(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories:
              - Violence
            score: 3
            reasoning: The text contains violent imagery.
        """)
        result = parse_judge_yaml(raw)
        assert result.categories == ["Violence"]
        assert result.score == 3
        assert result.reasoning == "The text contains violent imagery."

    def test_missing_categories_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            score: 3
            reasoning: No categories.
        """)
        with pytest.raises(ParseError, match="categories"):
            parse_judge_yaml(raw)

    def test_missing_score_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            reasoning: No score.
        """)
        with pytest.raises(ParseError, match="score"):
            parse_judge_yaml(raw)

    def test_missing_reasoning_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: 1
        """)
        with pytest.raises(ParseError, match="reasoning"):
            parse_judge_yaml(raw)

    def test_non_dict_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        with pytest.raises(ParseError, match="not a dict"):
            parse_judge_yaml("- just a list item")

    def test_score_out_of_range_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: 6
            reasoning: Too high.
        """)
        with pytest.raises(ParseError, match="out of range"):
            parse_judge_yaml(raw)

    def test_bool_score_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: true
            reasoning: Bool score.
        """)
        with pytest.raises(ParseError, match="bool"):
            parse_judge_yaml(raw)

    def test_string_score_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: "high"
            reasoning: String score.
        """)
        with pytest.raises(ParseError, match="score"):
            parse_judge_yaml(raw)

    def test_categories_not_list_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: Violence
            score: 3
            reasoning: Not a list.
        """)
        with pytest.raises(ParseError, match="categories.*list"):
            parse_judge_yaml(raw)

    def test_invalid_yaml_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        # yaml.safe_load may parse some odd strings as strings (not dicts),
        # so we test that we get a ParseError either way
        with pytest.raises(ParseError):
            parse_judge_yaml(":::not yaml{{{")

    def test_empty_string_raises(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        with pytest.raises(ParseError, match="not a dict"):
            parse_judge_yaml("")

    def test_float_score_coerced(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: 3.0
            reasoning: Float score.
        """)
        result = parse_judge_yaml(raw)
        assert result.score == 3
        assert isinstance(result.score, int)

    def test_empty_categories_valid(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            categories: []
            score: 1
            reasoning: Safe content.
        """)
        result = parse_judge_yaml(raw)
        assert result.categories == []
        assert result.score == 1

    def test_markdown_fences_stripped(self) -> None:
        from steering.eval.harmfulness_judge import parse_judge_yaml

        raw = textwrap.dedent("""\
            ```yaml
            categories:
              - Drugs
            score: 2
            reasoning: Minor reference.
            ```
        """)
        result = parse_judge_yaml(raw)
        assert result.categories == ["Drugs"]
        assert result.score == 2
