"""Configuration and path contracts for issue #1057."""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_base.config import validate_config
from steering.eval.axbench_base.paths import AxBenchBasePaths
from steering.eval.axbench_base.types import AxBenchBaseError


def test_issue_1057_defaults(make_valid_axbench_cfg) -> None:
    cfg = make_valid_axbench_cfg()

    assert cfg.max_concept_drop_fraction == 0.5
    assert cfg.max_dropped_concept_fraction == 0.5
    validate_config(cfg)


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("max_concept_drop_fraction", True),
        ("max_concept_drop_fraction", "0.5"),
        ("max_concept_drop_fraction", float("nan")),
        ("max_concept_drop_fraction", float("inf")),
        ("max_concept_drop_fraction", -0.01),
        ("max_concept_drop_fraction", 1.0),
        ("max_dropped_concept_fraction", True),
        ("max_dropped_concept_fraction", "0.5"),
        ("max_dropped_concept_fraction", float("-inf")),
        ("max_dropped_concept_fraction", -0.01),
        ("max_dropped_concept_fraction", 1.0),
    ],
)
def test_issue_1057_fraction_validation(
    make_valid_axbench_cfg,
    field_name: str,
    bad_value: object,
) -> None:
    cfg = make_valid_axbench_cfg(**{field_name: bad_value})

    with pytest.raises(AxBenchBaseError, match=field_name):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, 0.0, 0.25, 0.999999])
@pytest.mark.parametrize(
    "field_name",
    ["max_concept_drop_fraction", "max_dropped_concept_fraction"],
)
def test_issue_1057_valid_fraction_boundaries(
    make_valid_axbench_cfg,
    field_name: str,
    value: float,
) -> None:
    validate_config(make_valid_axbench_cfg(**{field_name: value}))


def test_issue_1057_paths_are_run_scoped(tmp_path: Path) -> None:
    output_paths = AxBenchBasePaths(tmp_path, "issue-1057")

    assert output_paths.stage1b_concept_yield_report() == (
        tmp_path / "issue-1057" / "stage1b_concept_yield.json"
    )
    assert output_paths.stage1c_concept_yield_report() == (
        tmp_path / "issue-1057" / "stage1c_concept_yield.json"
    )
    assert output_paths.stage1b_concepts() == (
        tmp_path / "issue-1057" / "concepts_stage1b.json"
    )
    assert output_paths.stage1c_concepts() == (
        tmp_path / "issue-1057" / "concepts_stage1c.json"
    )
    assert output_paths.downstream_concepts(run_negative_mode=False) == (
        output_paths.stage1b_concepts()
    )
    assert output_paths.downstream_concepts(run_negative_mode=True) == (
        output_paths.stage1c_concepts()
    )
