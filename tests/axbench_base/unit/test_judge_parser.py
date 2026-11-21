"""Unit tests for :func:`parse_axbench_rating`.

Three tiers:
1. Corpus tests — every row from
   ``tests/axbench_base/fixtures/judge_response_samples.jsonl`` must parse
   to the hand-annotated expected score. This closes the gap that
   shipped the original bug: the parser was only tested against the
   format the prompt *asks for*, not against what the judge *actually
   returns*.
2. Adversarial / format-drift tests — hand-written cases covering
   formats the empirical corpus may not hit, so future judge-model
   drift that matches one of these shapes still parses correctly.
3. Property tests — Hypothesis-driven sweeps over
   ``(score, prefix, suffix)`` to catch regex pathologies the hand-
   written cases miss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.judge_parser import (
    JUDGE_MAX_SCORE,
    parse_axbench_rating,
)

# ---------------------------------------------------------------------------
# Corpus-driven tests
# ---------------------------------------------------------------------------


_FIXTURE = Path(__file__).parent.parent.parent / "axbench_base" / "fixtures" / "judge_response_samples.jsonl"


def _load_corpus() -> list[dict[str, Any]]:
    """Read every row from the committed corpus file."""
    rows: list[dict[str, Any]] = []
    with open(_FIXTURE, encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            rows.append(json.loads(stripped_line))
    return rows


# Hand-annotated expected parse result for every row in the committed
# corpus. Keyed on the row's ``id`` field. If the corpus grows (re-run
# ``capture_judge_samples.py``), add the new ids here.
#
# The value is what ``parse_axbench_rating`` must return — an int in
# ``[0, JUDGE_MAX_SCORE]`` when the judge gave a well-formed rating,
# ``None`` when the response genuinely doesn't contain one (the test
# author does NOT fudge ambiguous cases; if the judge didn't say a
# score, the parser must not invent one).
_EXPECTED: dict[str, int | None] = {
    # examples_* — judge_concept_examples template
    "examples_000": 2,  # "Rating: 2"
    "examples_001": 2,  # "Rating: [[2]]"
    "examples_002": 2,  # "Rating: [[2]]"
    "examples_003": 2,  # "Rating: [[2]]"
    "examples_004": 0,  # "Rating: 0"
    "examples_005": 0,  # "Rating: [[0]]"
    "examples_006": 2,  # "Rating: [[2]]"
    "examples_007": 2,  # "Rating: [[2]]"
    "examples_008": 2,  # "Rating: [[2]]"
    "examples_009": 2,  # "Rating: 2"
    # relevance_* — judge_concept_relevance template
    "relevance_000": 2,  # strong match, fortifications
    "relevance_001": 0,  # unrelated, pancakes
    "relevance_002": 2,  # nervous tension
    "relevance_003": 0,  # calm confidence
    "relevance_004": 2,  # Schrödinger/wave function
    "relevance_005": 0,  # farmer's market
    "relevance_006": 2,  # corporate restructuring
    "relevance_007": 0,  # market stalls
    "relevance_008": 2,  # French culinary — uses bare "Rating: 2"
    "relevance_009": 0,  # laptop battery
    # fluency_* — judge_fluency template
    "fluency_000": 2,  # fluent sunset
    "fluency_001": 2,  # fluent Paris
    "fluency_002": 1,  # awkward walking
    "fluency_003": 1,  # repetitive dog ran
    "fluency_004": 0,  # purple electric sandwich
    "fluency_005": 2,  # fluent chess
    "fluency_006": 2,  # colorless green ideas
    "fluency_007": 2,  # team won game
    "fluency_008": 0,  # broken "The went running store"
    "fluency_009": 2,  # fluent long sentence
}


@pytest.mark.parametrize(
    "row",
    _load_corpus(),
    ids=lambda r: r["id"],
)
def test_parser_matches_hand_annotated_expected_on_real_corpus(
    row: dict[str, Any],
) -> None:
    """Regression pin: every real judge response in the corpus must
    parse to the hand-annotated expected score.

    If this test fails after a future capture_judge_samples.py re-run,
    either (a) the parser is broken on a new format the judge started
    emitting, or (b) the expected annotation is wrong. Investigate;
    don't just update the expected to make the test pass."""
    expected = _EXPECTED.get(row["id"])
    assert row["id"] in _EXPECTED, (
        f"corpus row {row['id']!r} has no entry in _EXPECTED; "
        f"add it to tests/axbench_base/unit/test_judge_parser.py"
    )
    got = parse_axbench_rating(row["raw_response"])
    assert got == expected, (
        f"parser returned {got!r} for corpus row {row['id']!r}, "
        f"expected {expected!r}. Raw response tail: "
        f"{row['raw_response'][-200:]!r}"
    )


def test_corpus_file_has_at_least_30_rows() -> None:
    """Regression: the corpus must not silently shrink. If a future
    capture-script edit drops rows, this fails loudly."""
    rows = _load_corpus()
    assert len(rows) >= 30, f"corpus shrank to {len(rows)} rows"


def test_every_corpus_row_has_expected_annotation() -> None:
    """Catch a missing annotation for a row that was added to the
    corpus without updating ``_EXPECTED``. Parametrize skipif would
    hide this; a dedicated test surfaces it clearly."""
    rows = _load_corpus()
    missing = [r["id"] for r in rows if r["id"] not in _EXPECTED]
    assert not missing, (
        f"{len(missing)} corpus row(s) have no entry in _EXPECTED: "
        f"{missing}. Add expected annotations."
    )


def test_corpus_covers_all_three_templates() -> None:
    """Coverage invariant: the corpus must exercise every judge template
    so parser changes don't regress any single template."""
    rows = _load_corpus()
    templates = {r["template"] for r in rows}
    assert templates == {
        "judge_concept_examples",
        "judge_concept_relevance",
        "judge_fluency",
    }, f"corpus covers {templates}, expected all three"


def test_corpus_contains_both_canonical_and_bare_formats() -> None:
    """Regression: the corpus must include at least one ``Rating: N``
    (bare) and one ``Rating: [[N]]`` (canonical) response, so the parser
    is tested against both shapes the real judge emits."""
    rows = _load_corpus()
    # Use a cheap heuristic: the presence of "[[" in the Rating line.
    has_canonical = False
    has_bare = False
    for r in rows:
        raw = r["raw_response"]
        if "Rating: [[" in raw:
            has_canonical = True
        # Look for "Rating: N" where N is a digit and NOT preceded by "[["
        # A simple way is checking the response has a bare pattern
        # somewhere, not "Rating: [" at that point.
        import re
        if re.search(r"Rating:\s*\d", raw):
            has_bare = True
    assert has_canonical, "corpus missing any `Rating: [[N]]` canonical responses"
    assert has_bare, "corpus missing any `Rating: N` bare responses"


# ---------------------------------------------------------------------------
# Adversarial / format-drift tests
# ---------------------------------------------------------------------------


class TestCanonicalAndBare:
    """Formats observed in the empirical corpus."""

    @pytest.mark.parametrize("score", [0, 1, 2])
    def test_canonical_double_bracket(self, score: int) -> None:
        assert parse_axbench_rating(f"Rating: [[{score}]]") == score

    @pytest.mark.parametrize("score", [0, 1, 2])
    def test_bare_no_brackets(self, score: int) -> None:
        """Observed format in 5/30 real responses."""
        assert parse_axbench_rating(f"Rating: {score}") == score

    @pytest.mark.parametrize("score", [0, 1, 2])
    def test_canonical_after_long_explanation(self, score: int) -> None:
        """concept_relevance and fluency always emit prose first."""
        prose = "The sentence has a clear subject and verb. " * 5
        response = f"{prose}\n\nRating: [[{score}]]"
        assert parse_axbench_rating(response) == score

    @pytest.mark.parametrize("score", [0, 1, 2])
    def test_bare_after_long_explanation(self, score: int) -> None:
        prose = "Evaluation: the feature is well-matched. " * 5
        response = f"{prose}\n\nRating: {score}"
        assert parse_axbench_rating(response) == score


class TestWhitespaceVariants:
    """Whitespace tolerances — all should accept."""

    def test_no_space_after_colon(self) -> None:
        assert parse_axbench_rating("Rating:[[1]]") == 1
        assert parse_axbench_rating("Rating:1") == 1

    def test_double_space_after_colon(self) -> None:
        assert parse_axbench_rating("Rating:  [[1]]") == 1
        assert parse_axbench_rating("Rating:  1") == 1

    def test_spaces_inside_brackets(self) -> None:
        assert parse_axbench_rating("Rating: [[ 1 ]]") == 1

    def test_trailing_whitespace(self) -> None:
        assert parse_axbench_rating("Rating: 1   ") == 1
        assert parse_axbench_rating("Rating: [[1]]   \n") == 1

    def test_tab_after_colon(self) -> None:
        assert parse_axbench_rating("Rating:\t1") == 1


class TestFormatDrift:
    """Formats the empirical corpus may not hit but future model drift could."""

    def test_markdown_bold_around_value(self) -> None:
        """E.g. ``Rating: **2**`` — GPT models sometimes emit markdown."""
        assert parse_axbench_rating("Rating: **2**") == 2

    def test_markdown_bold_around_bracketed(self) -> None:
        assert parse_axbench_rating("Rating: **[[1]]**") == 1

    def test_single_brackets(self) -> None:
        """``Rating: [1]`` — single bracket pair.

        The empirical corpus has no instance of this, but an LLM easily
        drifts to it. Accept rather than reject to minimize silent
        drops; out-of-range checks still hold."""
        assert parse_axbench_rating("Rating: [1]") == 1

    def test_mismatched_brackets(self) -> None:
        """Pathological but easy for a model to emit."""
        assert parse_axbench_rating("Rating: [[1]") == 1
        assert parse_axbench_rating("Rating: [1]]") == 1

    def test_trailing_punctuation(self) -> None:
        assert parse_axbench_rating("Rating: 1.") == 1
        assert parse_axbench_rating("Rating: [[1]].") == 1
        assert parse_axbench_rating("Rating: 1,") == 1

    def test_multiline_rating_at_end(self) -> None:
        response = "Line one.\nLine two.\nLine three.\n\nRating: [[0]]"
        assert parse_axbench_rating(response) == 0

    def test_rating_followed_by_more_text(self) -> None:
        """Rare but possible: judge appends commentary after the rating."""
        response = "Rating: [[2]]\n\nFinal remarks: looks good."
        assert parse_axbench_rating(response) == 2

    def test_rating_inside_code_block(self) -> None:
        """Markdown code block surrounding the rating."""
        response = (
            "Here is my evaluation:\n\n```\nRating: [[2]]\n```\n\n"
            "Hope this helps."
        )
        assert parse_axbench_rating(response) == 2


class TestRejectionCases:
    """Formats that MUST return None so callers can retry / mark invalid."""

    def test_empty_string(self) -> None:
        assert parse_axbench_rating("") is None

    def test_none_input(self) -> None:
        assert parse_axbench_rating(None) is None

    def test_lowercase_rating_rejected(self) -> None:
        """Case-sensitive: the prompt template uses ``Rating:`` capitalized.
        A lowercase emission is a strong signal the judge went off-script."""
        assert parse_axbench_rating("rating: [[1]]") is None
        assert parse_axbench_rating("rating: 1") is None

    def test_score_wrong_word_rejected(self) -> None:
        assert parse_axbench_rating("Score: 2") is None
        assert parse_axbench_rating("Grade: [[1]]") is None

    def test_word_digit_rejected(self) -> None:
        """``Rating: one`` — words not numbers."""
        assert parse_axbench_rating("Rating: one") is None
        assert parse_axbench_rating("Rating: [[two]]") is None

    def test_out_of_range_rejected(self) -> None:
        """Out-of-rubric values MUST return None rather than clamp."""
        assert parse_axbench_rating("Rating: 3") is None
        assert parse_axbench_rating("Rating: [[3]]") is None
        assert parse_axbench_rating("Rating: 10") is None
        assert parse_axbench_rating("Rating: [[100]]") is None

    def test_negative_rejected(self) -> None:
        assert parse_axbench_rating("Rating: -1") is None
        assert parse_axbench_rating("Rating: [[-1]]") is None

    def test_decimal_rejected(self) -> None:
        """Decimals are off-rubric (judge should emit an integer).

        The parser must NOT silently truncate ``1.5`` to ``1``."""
        assert parse_axbench_rating("Rating: 1.5") is None
        assert parse_axbench_rating("Rating: [[1.5]]") is None
        assert parse_axbench_rating("Rating: 0.9") is None

    def test_multi_digit_decimal_rejected_no_backtracking(self) -> None:
        """Greedy ``\\d+`` must NOT backtrack into a shorter match that
        sneaks past the decimal lookahead. ``Rating: 12.5`` must NOT
        return ``1`` — that would re-introduce exactly the silent-wrong-
        score class of bug this PR exists to fix.

        Pathology: greedy ``\\d+`` matches ``12``, lookahead fails on
        ``.5``, engine backtracks to ``\\d+`` matching ``1``, lookahead
        sees ``2`` (not ``.``) and accepts. Returns ``1``. **WRONG.**"""
        assert parse_axbench_rating("Rating: 12.5") is None
        assert parse_axbench_rating("Rating: 23.5") is None
        assert parse_axbench_rating("Rating: 100.0") is None
        assert parse_axbench_rating("Rating: 100.5") is None
        # Same pathology in the long-prose path.
        assert (
            parse_axbench_rating("final answer Rating: 12.5") is None
        )

    def test_hex_notation_rejected(self) -> None:
        """``Rating: 0x10`` is a non-decimal numeric literal: greedy
        ``\\d+`` matches ``0`` and the lookaheads accept (``x`` isn't
        a digit or period). Range check accepts ``0`` -> the parser
        would return ``0`` (silently wrong: a 0x10 emission means the
        judge emitted hex, not a rubric score)."""
        assert parse_axbench_rating("Rating: 0x10") is None
        assert parse_axbench_rating("Rating: 0xFF") is None

    def test_exponential_notation_rejected(self) -> None:
        """``Rating: 1e2`` is exponential; greedy ``\\d+`` would match
        ``1`` and the ``(?!\\w)`` lookahead rejects on ``e`` (word char)."""
        assert parse_axbench_rating("Rating: 1e2") is None
        assert parse_axbench_rating("Rating: 2e0") is None

    def test_digit_followed_by_letter_rejected(self) -> None:
        """``Rating: 1abc`` signals the judge emitted a non-numeric
        token (the 1 is probably an item index or version number, not
        a rubric score). Reject rather than silently truncate to 1."""
        assert parse_axbench_rating("Rating: 1abc") is None
        assert parse_axbench_rating("Rating: 2nd") is None

    def test_digit_followed_by_space_accepted(self) -> None:
        """A digit followed by whitespace is a valid rating -- the
        ``(?!\\w)`` guard only rejects word-char suffixes."""
        assert parse_axbench_rating("Rating: 2 final") == 2
        assert parse_axbench_rating("Rating: 1)") == 1
        assert parse_axbench_rating("Rating: 0\n") == 0

    def test_huge_integer_rejected(self) -> None:
        """A 10-digit number is not a valid rubric score; range check
        must reject."""
        assert parse_axbench_rating("Rating: 1234567890") is None

    def test_leading_zero_in_range_accepted(self) -> None:
        """``Rating: 01`` parses to integer 1 — leading zeros are
        meaningless to ``int()``. Accept (in range)."""
        assert parse_axbench_rating("Rating: 01") == 1

    def test_leading_zero_out_of_range_rejected(self) -> None:
        """``Rating: 0123`` parses to integer 123 — out of range,
        rejected. Verifies the leading-zero path doesn't accidentally
        truncate."""
        assert parse_axbench_rating("Rating: 0123") is None

    def test_no_rating_word_rejected(self) -> None:
        assert parse_axbench_rating("The answer is 2") is None
        assert parse_axbench_rating("I give it a 2") is None

    def test_rating_colon_without_number(self) -> None:
        assert parse_axbench_rating("Rating:") is None
        assert parse_axbench_rating("Rating: ") is None
        assert parse_axbench_rating("Rating: [[]]") is None


class TestLastRatingWins:
    """When a response contains multiple Rating: lines, the LAST one is
    authoritative. The prompt template asks for a concluding rating, so
    any earlier mentions are commentary / hypothetical — the final
    answer is what we score."""

    def test_two_different_ratings_picks_last(self) -> None:
        """This is a real drift risk: the judge might emit
        "I initially thought Rating: [[1]], but on reflection Rating: [[2]]"."""
        response = (
            "Initially I thought Rating: [[1]], but on closer reading, "
            "the concept is more strongly present than I first judged. "
            "Rating: [[2]]"
        )
        assert parse_axbench_rating(response) == 2

    def test_two_same_ratings_fine(self) -> None:
        assert (
            parse_axbench_rating("Rating: [[1]] (confirmed) Rating: [[1]]")
            == 1
        )

    def test_three_ratings_picks_last(self) -> None:
        response = (
            "Rating: [[0]] initially. Wait, Rating: [[1]] on reflection. "
            "Final: Rating: [[2]]"
        )
        assert parse_axbench_rating(response) == 2


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------


_safe_text = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),  # no surrogates
        blacklist_characters=("\x00",),  # no NULs
    ),
    max_size=500,
)


class TestProperties:
    @given(
        score=st.integers(min_value=0, max_value=JUDGE_MAX_SCORE),
        prefix=_safe_text,
        suffix=_safe_text,
    )
    @settings(max_examples=400)
    @example(score=0, prefix="", suffix="")
    @example(score=JUDGE_MAX_SCORE, prefix="Explanation: ", suffix="")
    def test_canonical_roundtrip_finds_score(
        self, score: int, prefix: str, suffix: str,
    ) -> None:
        """For any ``(score in [0, MAX], prefix, suffix)``, a well-formed
        canonical response concludes with ``Rating: [[score]]`` — the
        parser must recover ``score`` as long as the prefix/suffix
        don't introduce their own Rating lines."""
        # Strip any "Rating:" substring from the fuzz input so it doesn't
        # create a competing match.
        clean_prefix = prefix.replace("Rating:", "")
        clean_suffix = suffix.replace("Rating:", "")
        response = f"{clean_prefix}Rating: [[{score}]]{clean_suffix}"
        assert parse_axbench_rating(response) == score

    @given(
        score=st.integers(min_value=0, max_value=JUDGE_MAX_SCORE),
        prefix=_safe_text,
        suffix=_safe_text,
    )
    @settings(max_examples=200)
    def test_bare_roundtrip_finds_score(
        self, score: int, prefix: str, suffix: str,
    ) -> None:
        """Bare ``Rating: N`` variant — same round-trip, different format."""
        clean_prefix = prefix.replace("Rating:", "")
        clean_suffix = suffix.replace("Rating:", "")
        # Put a non-digit sentinel before the suffix so a fuzzed digit
        # doesn't turn "Rating: 2" + "3" into "Rating: 23" (valid per
        # the regex, but 23 is out of range → parser returns None
        # which violates the property).
        response = f"{clean_prefix}Rating: {score}\n{clean_suffix}"
        assert parse_axbench_rating(response) == score

    @given(
        score=st.integers().filter(
            lambda s: s < 0 or s > JUDGE_MAX_SCORE,
        ),
    )
    @settings(max_examples=200)
    def test_out_of_range_always_rejected(self, score: int) -> None:
        """Any score outside ``[0, JUDGE_MAX_SCORE]`` must return None,
        whatever the surrounding format."""
        assert parse_axbench_rating(f"Rating: [[{score}]]") is None
        assert parse_axbench_rating(f"Rating: {score}") is None

    @given(
        int_part=st.integers(min_value=0, max_value=99999),
        frac_part=st.integers(min_value=0, max_value=99999),
    )
    @settings(max_examples=300)
    def test_decimal_strings_always_rejected(
        self, int_part: int, frac_part: int,
    ) -> None:
        """Exhaustive fuzzing over decimal-shaped ``Rating: N.M`` strings.

        This is the property that would have caught the
        multi-digit-decimal backtracking bug (C1) before it shipped:
        for ANY ``int_part`` / ``frac_part`` combination, the parser
        must return None. If the regex backtracks into a shorter
        match that happens to land in ``[0, 2]``, this property
        fails."""
        response = f"Rating: {int_part}.{frac_part}"
        assert parse_axbench_rating(response) is None, (
            f"parser returned non-None for {response!r}"
        )

    @given(
        int_part=st.integers(min_value=0, max_value=99999),
        frac_part=st.integers(min_value=0, max_value=99999),
        prefix=_safe_text,
    )
    @settings(max_examples=200)
    def test_decimal_with_prose_always_rejected(
        self, int_part: int, frac_part: int, prefix: str,
    ) -> None:
        """Same property under arbitrary prose prefix."""
        clean_prefix = prefix.replace("Rating:", "")
        response = f"{clean_prefix}Rating: {int_part}.{frac_part}"
        assert parse_axbench_rating(response) is None

    @given(
        value=st.integers(min_value=0, max_value=0xFFFFFF),
    )
    @settings(max_examples=200)
    def test_hex_notation_always_rejected(self, value: int) -> None:
        """``Rating: 0xNN`` must never silently parse as a valid score."""
        response = f"Rating: 0x{value:X}"
        assert parse_axbench_rating(response) is None

    @given(nonsense=_safe_text)
    @settings(max_examples=200)
    def test_no_rating_word_returns_none_or_parses_nonsense(
        self, nonsense: str,
    ) -> None:
        """Responses without ``Rating:`` must not parse — unless the
        fuzz input happens to contain ``Rating: N`` literally (vanishingly
        unlikely but Hypothesis can still find it via shrinking)."""
        result = parse_axbench_rating(nonsense)
        if result is not None:
            # If the fuzzer happened to generate a Rating: line, the
            # parser is correct to find it. But the score must be valid.
            assert 0 <= result <= JUDGE_MAX_SCORE


# ---------------------------------------------------------------------------
# JUDGE_MAX_SCORE constant (preserved from prior tests)
# ---------------------------------------------------------------------------


def test_judge_max_score_value_is_two() -> None:
    """The AxBench judge rates on ``{0, 1, 2}``; max is 2."""
    assert JUDGE_MAX_SCORE == 2


def test_rating_at_judge_max_score_accepted() -> None:
    assert parse_axbench_rating(f"Rating: [[{JUDGE_MAX_SCORE}]]") == JUDGE_MAX_SCORE


def test_rating_one_above_judge_max_score_rejected() -> None:
    assert parse_axbench_rating(f"Rating: [[{JUDGE_MAX_SCORE + 1}]]") is None


def test_every_in_range_rating_accepted() -> None:
    for score in range(JUDGE_MAX_SCORE + 1):
        assert parse_axbench_rating(f"Rating: [[{score}]]") == score
