"""Execution-scope environment parsing tests."""

from __future__ import annotations

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import parse_concept_ids, parse_method_filter
from steering.eval.axbench_adv.types import ALL_METHODS


@pytest.mark.parametrize("max_concepts", [True, 0, -1, 2.5])
def test_concept_slice_rejects_invalid_configured_population(max_concepts: object) -> None:
    with pytest.raises(AxBenchAdvError, match="max_concepts must be a positive integer"):
        parse_concept_ids(max_concepts, environ={})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, (0, 1, 2, 3, 4)),
        ({"AXBENCH_ADV_CONCEPT_LO": "2"}, (2, 3, 4)),
        ({"AXBENCH_ADV_CONCEPT_HI": "3"}, (0, 1, 2)),
        (
            {"AXBENCH_ADV_CONCEPT_LO": " 1 ", "AXBENCH_ADV_CONCEPT_HI": "4"},
            (1, 2, 3),
        ),
    ],
)
def test_concept_slice_parses_half_open_bounds(
    environ: dict[str, str], expected: tuple[int, ...]
) -> None:
    assert parse_concept_ids(5, environ=environ) == expected


@pytest.mark.parametrize(
    ("environ", "match"),
    [
        ({"AXBENCH_ADV_CONCEPT_LO": "-1"}, "non-negative integer"),
        ({"AXBENCH_ADV_CONCEPT_HI": "1.5"}, "non-negative integer"),
        ({"AXBENCH_ADV_CONCEPT_LO": ""}, "non-negative integer"),
        (
            {"AXBENCH_ADV_CONCEPT_LO": "3", "AXBENCH_ADV_CONCEPT_HI": "3"},
            "must satisfy",
        ),
        (
            {"AXBENCH_ADV_CONCEPT_LO": "4", "AXBENCH_ADV_CONCEPT_HI": "2"},
            "must satisfy",
        ),
        ({"AXBENCH_ADV_CONCEPT_HI": "6"}, "must not exceed"),
    ],
)
def test_concept_slice_rejects_invalid_values(
    environ: dict[str, str], match: str
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        parse_concept_ids(5, environ=environ)


def test_method_filter_selects_configured_subset_in_canonical_order() -> None:
    assert parse_method_filter(
        ALL_METHODS,
        environ={"AXBENCH_ADV_METHODS": "NLSteering, DiffMean"},
    ) == ("DiffMean", "NLSteering")
    assert parse_method_filter(ALL_METHODS, environ={}) == ALL_METHODS


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("Unknown", "unknown AxBench-Adv methods"),
        ("NLSteering", "outside the configured source contract"),
        ("DiffMean,DiffMean", "must be unique"),
        ("", "comma-separated nonempty"),
        ("DiffMean,", "comma-separated nonempty"),
    ],
)
def test_method_filter_rejects_unknown_duplicate_and_empty_names(
    value: str, match: str
) -> None:
    configured = ALL_METHODS[:-1] if value == "NLSteering" else ALL_METHODS
    with pytest.raises(AxBenchAdvError, match=match):
        parse_method_filter(configured, environ={"AXBENCH_ADV_METHODS": value})
