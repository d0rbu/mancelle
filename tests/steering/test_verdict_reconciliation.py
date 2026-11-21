"""Exhaustive tests for verdict reconciliation logic in steering_eval."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pytest

from steering.eval.steering_eval import (
    JudgeChoice,
    PairwiseResult,
    Verdict,
    format_judge_prompt,
    judge_pair,
    reconcile_judgments,
)

# ---------------------------------------------------------------------------
# Test 1: All 9 valid combinations (3x3 grid)
# ---------------------------------------------------------------------------

EXPECTED_VERDICTS: dict[tuple[JudgeChoice, JudgeChoice], Verdict] = {
    (JudgeChoice.A, JudgeChoice.B): Verdict.WIN,
    (JudgeChoice.B, JudgeChoice.A): Verdict.LOSS,
    (JudgeChoice.TIE, JudgeChoice.TIE): Verdict.TIE,
    (JudgeChoice.A, JudgeChoice.A): Verdict.INCONSISTENT,
    (JudgeChoice.B, JudgeChoice.B): Verdict.INCONSISTENT,
    (JudgeChoice.A, JudgeChoice.TIE): Verdict.INCONSISTENT,
    (JudgeChoice.TIE, JudgeChoice.A): Verdict.INCONSISTENT,
    (JudgeChoice.B, JudgeChoice.TIE): Verdict.INCONSISTENT,
    (JudgeChoice.TIE, JudgeChoice.B): Verdict.INCONSISTENT,
}


class TestReconcileJudgments:
    """Test reconcile_judgments covers the full 3x3 grid."""

    @pytest.mark.parametrize(
        ("ab", "ba", "expected"),
        [
            (ab, ba, expected)
            for (ab, ba), expected in EXPECTED_VERDICTS.items()
        ],
        ids=[
            f"AB={ab.name}_BA={ba.name}"
            for ab, ba in EXPECTED_VERDICTS
        ],
    )
    def test_reconcile_all_combinations(
        self,
        ab: JudgeChoice,
        ba: JudgeChoice,
        expected: Verdict,
    ) -> None:
        assert reconcile_judgments(ab, ba) == expected

    def test_full_grid_coverage(self) -> None:
        """Verify we tested every combination of JudgeChoice x JudgeChoice."""
        all_combos = set(product(JudgeChoice, JudgeChoice))
        tested_combos = set(EXPECTED_VERDICTS.keys())
        assert tested_combos == all_combos, (
            f"Missing combinations: {all_combos - tested_combos}"
        )

    def test_win_means_conditioned_preferred(self) -> None:
        """WIN: conditioned picked in both positions (A in AB, B in BA)."""
        assert reconcile_judgments(JudgeChoice.A, JudgeChoice.B) == Verdict.WIN

    def test_loss_means_baseline_preferred(self) -> None:
        """LOSS: baseline picked in both positions (B in AB, A in BA)."""
        assert reconcile_judgments(JudgeChoice.B, JudgeChoice.A) == Verdict.LOSS

    def test_tie_requires_both_tie(self) -> None:
        """TIE only when both rounds report TIE."""
        assert reconcile_judgments(JudgeChoice.TIE, JudgeChoice.TIE) == Verdict.TIE


# ---------------------------------------------------------------------------
# Mock judge helpers
# ---------------------------------------------------------------------------


@dataclass
class MockJudge:
    """Judge that returns a sequence of pre-configured responses."""

    responses: list[str]
    prompts_received: list[str] | None = None

    def __post_init__(self) -> None:
        self._index = 0
        if self.prompts_received is None:
            self.prompts_received = []

    def judge(self, prompt: str) -> str:
        self.prompts_received.append(prompt)  # ty: ignore[unresolved-attribute]
        response = self.responses[self._index]
        self._index += 1
        return response


# ---------------------------------------------------------------------------
# Test 2: judge_pair with valid responses
# ---------------------------------------------------------------------------


class TestJudgePairValid:
    """judge_pair with valid judge responses."""

    def test_win(self) -> None:
        mock = MockJudge(responses=["A", "B"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.WIN
        assert result.ab_response == JudgeChoice.A
        assert result.ba_response == JudgeChoice.B

    def test_loss(self) -> None:
        mock = MockJudge(responses=["B", "A"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.LOSS
        assert result.ab_response == JudgeChoice.B
        assert result.ba_response == JudgeChoice.A

    def test_tie(self) -> None:
        mock = MockJudge(responses=["TIE", "TIE"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.TIE
        assert result.ab_response == JudgeChoice.TIE
        assert result.ba_response == JudgeChoice.TIE

    def test_inconsistent(self) -> None:
        mock = MockJudge(responses=["A", "A"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INCONSISTENT
        assert result.ab_response == JudgeChoice.A
        assert result.ba_response == JudgeChoice.A

    def test_result_is_pairwise_result(self) -> None:
        mock = MockJudge(responses=["A", "B"])
        result = judge_pair("cond", "base", "trait", mock)
        assert isinstance(result, PairwiseResult)


# ---------------------------------------------------------------------------
# Test 3: judge_pair with INVALID responses (both garbage)
# ---------------------------------------------------------------------------


class TestJudgePairInvalid:
    """judge_pair when judge returns unparseable responses."""

    def test_both_garbage(self) -> None:
        mock = MockJudge(responses=["MAYBE", "PERHAPS"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID

    def test_first_garbage(self) -> None:
        """First response is garbage -> INVALID, raw strings stored."""
        mock = MockJudge(responses=["MAYBE", "B"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID
        assert result.ab_response == "MAYBE"
        assert result.ba_response == "B"

    def test_empty_response(self) -> None:
        mock = MockJudge(responses=["", "A"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID

    def test_numeric_response(self) -> None:
        mock = MockJudge(responses=["1", "2"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID


# ---------------------------------------------------------------------------
# Test 4: judge_pair with mixed valid/invalid
# ---------------------------------------------------------------------------


class TestJudgePairMixed:
    """One valid, one invalid response."""

    def test_first_valid_second_invalid(self) -> None:
        mock = MockJudge(responses=["A", "GARBAGE"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID
        # ab_response was successfully parsed to JudgeChoice
        assert result.ab_response == JudgeChoice.A
        # ba_response remains as the raw (uppercased) string
        assert result.ba_response == "GARBAGE"

    def test_first_invalid_second_valid(self) -> None:
        mock = MockJudge(responses=["NOPE", "B"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.INVALID
        # When first fails, both raw strings are stored
        assert result.ab_response == "NOPE"
        assert result.ba_response == "B"


# ---------------------------------------------------------------------------
# Test 5: format_judge_prompt correctness
# ---------------------------------------------------------------------------


class TestFormatJudgePrompt:
    """Verify the prompt template is correctly formatted."""

    def test_contains_description(self) -> None:
        prompt = format_judge_prompt("curiosity", "text_a", "text_b")
        assert "curiosity" in prompt

    def test_contains_texts(self) -> None:
        prompt = format_judge_prompt("trait", "ALPHA_TEXT", "BETA_TEXT")
        assert "ALPHA_TEXT" in prompt
        assert "BETA_TEXT" in prompt

    def test_text_a_before_text_b(self) -> None:
        prompt = format_judge_prompt("trait", "FIRST", "SECOND")
        assert prompt.index("FIRST") < prompt.index("SECOND")

    def test_contains_instruction(self) -> None:
        prompt = format_judge_prompt("trait", "a", "b")
        assert '"A"' in prompt
        assert '"B"' in prompt
        assert '"TIE"' in prompt

    def test_template_sections(self) -> None:
        prompt = format_judge_prompt("my_trait", "aaa", "bbb")
        assert "Trait: my_trait" in prompt
        assert "Text A:" in prompt
        assert "Text B:" in prompt


# ---------------------------------------------------------------------------
# Test 6: Position swap correctness
# ---------------------------------------------------------------------------


class TestPositionSwap:
    """Verify judge_pair actually swaps positions between calls."""

    def test_conditioned_is_a_in_first_call(self) -> None:
        mock = MockJudge(responses=["A", "B"])
        judge_pair("CONDITIONED", "BASELINE", "trait", mock)

        ab_prompt = mock.prompts_received[0]  # ty: ignore[not-subscriptable]
        # In the AB prompt, conditioned is text_a and baseline is text_b
        assert ab_prompt.index("CONDITIONED") < ab_prompt.index("BASELINE")

    def test_conditioned_is_b_in_second_call(self) -> None:
        mock = MockJudge(responses=["A", "B"])
        judge_pair("CONDITIONED", "BASELINE", "trait", mock)

        ba_prompt = mock.prompts_received[1]  # ty: ignore[not-subscriptable]
        # In the BA prompt, baseline is text_a and conditioned is text_b
        assert ba_prompt.index("BASELINE") < ba_prompt.index("CONDITIONED")

    def test_both_prompts_contain_description(self) -> None:
        mock = MockJudge(responses=["TIE", "TIE"])
        judge_pair("cond", "base", "my_special_trait", mock)

        assert len(mock.prompts_received) == 2  # ty: ignore[invalid-argument-type]
        for prompt in mock.prompts_received:  # ty: ignore[not-iterable]
            assert "my_special_trait" in prompt

    def test_exactly_two_judge_calls(self) -> None:
        mock = MockJudge(responses=["A", "B"])
        judge_pair("cond", "base", "trait", mock)
        assert len(mock.prompts_received) == 2  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Test 7: Verdict and JudgeChoice string values
# ---------------------------------------------------------------------------


class TestEnumValues:
    """Verify enum members have the expected string values."""

    def test_verdict_values(self) -> None:
        assert Verdict.WIN == "win"
        assert Verdict.LOSS == "loss"
        assert Verdict.TIE == "tie"
        assert Verdict.INCONSISTENT == "inconsistent"
        assert Verdict.INVALID == "invalid"

    def test_judge_choice_values(self) -> None:
        assert JudgeChoice.A == "A"
        assert JudgeChoice.B == "B"
        assert JudgeChoice.TIE == "TIE"

    def test_verdict_is_str(self) -> None:
        for member in Verdict:
            assert isinstance(member, str)

    def test_judge_choice_is_str(self) -> None:
        for member in JudgeChoice:
            assert isinstance(member, str)

    def test_verdict_member_count(self) -> None:
        assert len(Verdict) == 5

    def test_judge_choice_member_count(self) -> None:
        assert len(JudgeChoice) == 3


# ---------------------------------------------------------------------------
# Test 8: Response normalization
# ---------------------------------------------------------------------------


class TestResponseNormalization:
    """Verify judge responses are stripped and uppercased before parsing."""

    def test_lowercase_a(self) -> None:
        mock = MockJudge(responses=["a", "b"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.WIN

    def test_whitespace_padded(self) -> None:
        mock = MockJudge(responses=["  A  ", "  B  "])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.WIN

    def test_newline_padded(self) -> None:
        mock = MockJudge(responses=[" a\n", "\nb\n"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.WIN

    def test_mixed_case_tie(self) -> None:
        mock = MockJudge(responses=["Tie", "tie"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.TIE

    def test_tab_padded(self) -> None:
        mock = MockJudge(responses=["\tA\t", "\tB\t"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.WIN

    def test_lowercase_loss(self) -> None:
        mock = MockJudge(responses=["b", "a"])
        result = judge_pair("cond", "base", "trait", mock)
        assert result.verdict == Verdict.LOSS
