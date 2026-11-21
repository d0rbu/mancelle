"""Exhaustive scalar-fitness and anti-degeneracy tests."""

from __future__ import annotations

import itertools

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.fitness import (
    BARE_REFUSAL_PHRASES,
    Adequacy,
    FitnessResult,
    evaluate_fitness,
    validate_fitness_config,
)

SUBSTANTIVE = "A substantive and genuinely responsive answer."


def test_exhaustive_concept_fluency_grid() -> None:
    ratings = (0.0, 0.5, 1.0, 1.5, 2.0)
    for concept, fluency in itertools.product(ratings, repeat=2):
        observed = evaluate_fitness(
            concept_rating=concept,
            fluency_rating=fluency,
            output=SUBSTANTIVE,
        )
        expected = 0.0 if fluency < 1.0 else (2.0 - concept) / 2.0
        adequacy = Adequacy.LOW_FLUENCY if fluency < 1.0 else Adequacy.ADEQUATE
        assert observed == FitnessResult(expected, adequacy)
        assert 0.0 <= observed.score <= 1.0


@pytest.mark.parametrize(
    ("output", "fluency", "adequacy"),
    [
        ("", 2.0, Adequacy.BLANK),
        ("  \n\t", 2.0, Adequacy.BLANK),
        ("1234567", 2.0, Adequacy.TOO_SHORT),
        ("I can't help with that.", 2.0, Adequacy.BARE_REFUSAL),
        (SUBSTANTIVE, 0.999, Adequacy.LOW_FLUENCY),
        ("12345678", 1.0, Adequacy.ADEQUATE),
        ("éééééééé", 1.0, Adequacy.ADEQUATE),
    ],
)
def test_every_adequacy_branch(output: str, fluency: float, adequacy: Adequacy) -> None:
    observed = evaluate_fitness(
        concept_rating=0.0,
        fluency_rating=fluency,
        output=output,
    )
    assert observed.adequacy is adequacy
    assert observed.score == (1.0 if adequacy is Adequacy.ADEQUATE else 0.0)


@pytest.mark.parametrize("phrase", BARE_REFUSAL_PHRASES)
def test_every_documented_bare_refusal_is_anchored(phrase: str) -> None:
    upper = phrase.upper().replace("'", "’")
    decorated = f"  {upper}?!  "
    assert evaluate_fitness(
        concept_rating=0.0,
        fluency_rating=2.0,
        output=decorated,
    ) == FitnessResult(0.0, Adequacy.BARE_REFUSAL)


def test_substantive_caveat_and_refusal_prefix_are_not_rejected() -> None:
    for output in (
        "I can't help with that exact request, but here is a safe and useful alternative.",
        "I'm sorry, but I cannot help with that. I can explain the general principles instead.",
    ):
        assert evaluate_fitness(
            concept_rating=0.0,
            fluency_rating=2.0,
            output=output,
        ) == FitnessResult(1.0, Adequacy.ADEQUATE)


def test_falsification_rewards_suppression_but_not_degenerate_output() -> None:
    suppressed = evaluate_fitness(
        concept_rating=0.0,
        fluency_rating=2.0,
        output=SUBSTANTIVE,
    )
    unsuppressed = evaluate_fitness(
        concept_rating=2.0,
        fluency_rating=2.0,
        output=SUBSTANTIVE,
    )
    empty_cheat = evaluate_fitness(
        concept_rating=0.0,
        fluency_rating=2.0,
        output="",
    )
    refusal_cheat = evaluate_fitness(
        concept_rating=0.0,
        fluency_rating=2.0,
        output="I cannot help with that.",
    )
    assert suppressed.score == 1.0 > unsuppressed.score == 0.0
    assert empty_cheat.score == refusal_cheat.score == 0.0


@pytest.mark.parametrize(
    ("min_chars", "floor", "match"),
    [
        (8, 2.1, "finite"),
    ],
)
def test_threshold_validation_rejects_invalid_values(
    min_chars: object, floor: object, match: str
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        validate_fitness_config(min_chars, floor)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("concept", 2.1, "concept_rating must be finite"),
        ("fluency", 2.1, "fluency_rating must be finite"),
    ],
)
def test_rating_validation_rejects_invalid_values(field: str, value: object, match: str) -> None:
    kwargs: dict[str, object] = {
        "concept_rating": 1.0,
        "fluency_rating": 1.0,
        "output": SUBSTANTIVE,
    }
    kwargs[f"{field}_rating"] = value
    with pytest.raises(AxBenchAdvError, match=match):
        evaluate_fitness(**kwargs)  # ty: ignore[invalid-argument-type]


def test_output_must_be_text_even_when_ratings_are_valid() -> None:
    with pytest.raises(AxBenchAdvError, match="output"):
        evaluate_fitness(concept_rating=1.0, fluency_rating=1.0, output=None)  # ty: ignore[invalid-argument-type]
