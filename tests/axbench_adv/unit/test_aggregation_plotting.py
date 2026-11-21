"""Exact proposal-axis aggregation and plotting tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pytest
from matplotlib.figure import Figure

from steering.eval.axbench_adv.aggregation import (
    AGGREGATE_SCHEMA,
    aggregate_point,
    build_aggregate,
    validate_aggregate,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.plotting import (
    CONCEPT_AXIS_LABEL,
    FLUENCY_AXIS_LABEL,
    FOOTER,
    method_trajectory,
    plot_concept_vs_fluency,
    render_axes,
)
from steering.eval.axbench_adv.types import METHODS
from steering.eval.axbench_bpb_adv.aggregation import strength_color_for_rank


def _judged(method: str, factor: float, pairs: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "concept_id": concept_id,
            "results": {
                "LMJudgeEvaluator": {
                    method: {
                        "factor": [factor],
                        "relevance_concept_ratings": [concept],
                        "fluency_ratings": [fluency],
                    }
                }
            },
        }
        for concept_id, (concept, fluency) in enumerate(pairs)
    ]


def _aggregate_table(rows: list[dict] | None = None) -> pa.Table:
    default = [
        {
            "method": method,
            "proposal_count": proposal_count,
            "coefficient": 1.0,
            "n_concepts": 2,
            "concept_score": 1.0 - rank * 0.1,
            "fluency_score": 1.5 - rank * 0.05,
            "concept_score_sem": 0.1,
            "fluency_score_sem": 0.2,
        }
        for method in METHODS
        for rank, proposal_count in enumerate((0, 1, 2, 4, 8))
    ]
    return pa.Table.from_pylist(rows if rows is not None else default, schema=AGGREGATE_SCHEMA)


def test_aggregate_point_exact_mean_and_sem() -> None:
    row = aggregate_point(
        _judged("DiffMean", 1.0, [(0.5, 1.0), (1.5, 2.0)]),
        method="DiffMean",
        factor=1.0,
        proposal_count=4,
    )
    assert row == {
        "method": "DiffMean",
        "proposal_count": 4,
        "coefficient": 1.0,
        "n_concepts": 2,
        "concept_score": 1.0,
        "fluency_score": 1.5,
        "concept_score_sem": 0.5,
        "fluency_score_sem": 0.5,
    }
    single = aggregate_point(
        _judged("LoRA", 1.0, [(1.0, 1.0)]),
        method="LoRA",
        factor=1.0,
        proposal_count=0,
    )
    assert single["concept_score_sem"] == single["fluency_score_sem"] == 0.0


def test_build_aggregate_uses_regenerated_control_and_all_independent_levels(
    tmp_path: Path,
) -> None:
    levels = (0, 1, 2, 4, 8)
    best = dict.fromkeys(METHODS, 1.0)
    judged = {
        (method, level): _judged(
            method,
            1.0,
            [(1.0 - 0.01 * level, 1.5), (1.2 - 0.01 * level, 1.7)],
        )
        for method in best
        for level in levels
    }
    table = build_aggregate(
        best_factors=best,
        judged_points=judged,
        expected_n_concepts=2,
        output_path=tmp_path / "aggregate.parquet",
    )
    assert table.num_rows == 30
    assert [(row["method"], row["proposal_count"]) for row in table.to_pylist()] == [
        (method, level) for method in sorted(best) for level in levels
    ]
    assert table.to_pylist()[0]["concept_score"] == pytest.approx(1.1)


@pytest.mark.parametrize("concept_ids", [(0,), (0, 0), (1, 0), (False, 1), (0.0, 1)])
def test_aggregate_rejects_invalid_explicit_population(tmp_path: Path, concept_ids: tuple) -> None:
    with pytest.raises(AxBenchAdvError, match="expected_concept_ids"):
        build_aggregate(
            best_factors={"DiffMean": 1.0}, judged_points={}, expected_n_concepts=2,
            expected_concept_ids=concept_ids, output_path=tmp_path / "aggregate.parquet",
        )
    assert not (tmp_path / "aggregate.parquet").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"proposal_count": True}, "must be an integer"),
        ({"proposal_count": -1}, "must be one of"),
        ({"proposal_count": 3}, "must be one of"),
    ],
)
def test_aggregate_point_rejects_invalid_level(kwargs: dict, match: str) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        aggregate_point(
            _judged("DiffMean", 1.0, [(1.0, 1.0)]),
            method="DiffMean",
            factor=1.0,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (None, "does not conform"),
        ([], "must not be empty"),
        ([{"proposal_count": -1}], "must be one of"),
        ([{"method": "Unknown"}], "unknown aggregate method"),
        ([{"n_concepts": 0}], "positive"),
        ([{"coefficient": float("nan")}], "coefficient"),
        ([{"coefficient": 7.0}], "factor grid"),
        ([{"concept_score": 3.0}], "concept_score"),
        ([{"fluency_score_sem": -1.0}], "fluency_score_sem"),
    ],
)
def test_validate_aggregate_rejects_semantic_mutations(rows: list[dict] | None, match: str) -> None:
    if rows is None:
        table = pa.table({"bad": [1]})
    elif rows == []:
        table = pa.Table.from_pylist([], schema=AGGREGATE_SCHEMA)
    else:
        base = _aggregate_table().to_pylist()[0]
        base.update(rows[0])
        table = _aggregate_table([base])
    with pytest.raises(AxBenchAdvError, match=match):
        validate_aggregate(table)
    duplicate = _aggregate_table([_aggregate_table().to_pylist()[0]] * 2)
    with pytest.raises(AxBenchAdvError, match="duplicate"):
        validate_aggregate(duplicate)


@pytest.mark.parametrize(
    ("best", "match"),
    [({}, "nonempty subset"), ({"LoRA": 1.0}, "coverage differs")],
)
def test_build_aggregate_rejects_empty_inputs(
    tmp_path: Path, best: dict[str, float], match: str
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        build_aggregate(
            best_factors=best,
            judged_points={},
            expected_n_concepts=2,
            output_path=tmp_path / "aggregate.parquet",
        )


def test_build_aggregate_rejects_incomplete_point_grid(tmp_path: Path) -> None:
    with pytest.raises(AxBenchAdvError, match="coverage differs"):
        build_aggregate(
            best_factors=dict.fromkeys(METHODS, 1.0),
            judged_points={},
            expected_n_concepts=2,
            output_path=tmp_path / "aggregate.parquet",
        )


@pytest.mark.parametrize("expected", [True, 0])
def test_build_aggregate_rejects_invalid_expected_concept_count(
    tmp_path: Path, expected: object
) -> None:
    with pytest.raises(AxBenchAdvError, match="positive integer"):
        build_aggregate(
            best_factors=dict.fromkeys(METHODS, 1.0),
            judged_points={},
            expected_n_concepts=expected,  # ty: ignore[invalid-argument-type]
            output_path=tmp_path / "aggregate.parquet",
        )


def test_build_aggregate_rejects_uniformly_incomplete_judged_points(tmp_path: Path) -> None:
    judged = {
        (method, level): _judged(method, 1.0, [(1.0, 1.0)])
        for method in METHODS
        for level in (0, 1, 2, 4, 8)
    }
    with pytest.raises(AxBenchAdvError, match="exactly once in ascending order"):
        build_aggregate(
            best_factors=dict.fromkeys(METHODS, 1.0),
            judged_points=judged,
            expected_n_concepts=2,
            output_path=tmp_path / "aggregate.parquet",
        )


def test_validate_aggregate_rejects_incomplete_or_inconsistent_grid() -> None:
    rows = _aggregate_table().to_pylist()
    with pytest.raises(AxBenchAdvError, match="exact method/proposal-count"):
        validate_aggregate(_aggregate_table(rows[:-1]))
    rows[1]["n_concepts"] = 3
    with pytest.raises(AxBenchAdvError, match="n_concepts must be identical"):
        validate_aggregate(_aggregate_table(rows))
    rows = _aggregate_table().to_pylist()
    rows[1]["coefficient"] = 2.0
    with pytest.raises(AxBenchAdvError, match="coefficient must be constant"):
        validate_aggregate(_aggregate_table(rows))


def test_validate_aggregate_rejects_null() -> None:
    rows = _aggregate_table().to_pylist()
    rows[0]["method"] = None
    with pytest.raises(AxBenchAdvError, match="null values"):
        validate_aggregate(_aggregate_table(rows))


def test_trajectory_colors_axes_footer_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = _aggregate_table()
    points = method_trajectory(table, "DiffMean")
    assert [point.proposal_count for point in points] == [0, 1, 2, 4, 8]
    assert points[0].color == tuple(strength_color_for_rank("#1f77b4", rank=0, n_ranks=5))
    assert points[-1].color == tuple(strength_color_for_rank("#1f77b4", rank=4, n_ranks=5))
    assert points[0].color == pytest.approx((0.7364705882352941, 0.84, 0.9117647058823529))
    assert points[2].color == pytest.approx(
        (0.42901960784313725, 0.6533333333333333, 0.8088235294117647)
    )
    assert points[-1].color == pytest.approx(
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    )
    assert strength_color_for_rank("black", rank=10, n_ranks=1) == pytest.approx((0.7, 0.7, 0.7))
    expected_footer = (
        "0 is the regenerated empty-suffix control; 1/2/4/8 are independent "
        "fixed-seed GEPA runs, and larger proposal counts are darker."
    )
    assert FOOTER == expected_footer

    figure, ax = plt.subplots()
    try:
        render_axes(ax, table, base_colors={"DiffMean": "#123456"})
        assert ax.get_xlabel() == FLUENCY_AXIS_LABEL
        assert ax.get_ylabel() == CONCEPT_AXIS_LABEL
        lines = {line.get_label(): line for line in ax.lines if line.get_label() in METHODS}
        assert set(lines) == set(METHODS)
        assert np.asarray(lines["DiffMean"].get_xdata()).tolist() == [1.5, 1.45, 1.4, 1.35, 1.3]
        assert np.asarray(lines["DiffMean"].get_ydata()).tolist() == [1.0, 0.9, 0.8, 0.7, 0.6]
    finally:
        plt.close(figure)
    footer_calls: list[str] = []
    original_text = Figure.text

    def spy_text(figure: Figure, x: float, y: float, text: str, **kwargs: Any) -> object:
        footer_calls.append(text)
        return original_text(figure, x, y, text, **kwargs)

    monkeypatch.setattr(Figure, "text", spy_text)
    paths = plot_concept_vs_fluency(table, tmp_path)
    assert [path.name for path in paths] == [
        "concept_vs_fluency.png",
        "concept_vs_fluency_DiffMean.png",
        "concept_vs_fluency_LinearProbe.png",
        "concept_vs_fluency_LoRA.png",
        "concept_vs_fluency_PromptSteering.png",
        "concept_vs_fluency_SelfConditioning.png",
        "concept_vs_fluency_SelfConditioningMasked.png",
    ]
    assert footer_calls == [expected_footer] * 7
    assert all(path.stat().st_size > 0 for path in paths)


def test_trajectory_rejects_unknown_duplicate_and_wrong_schema() -> None:
    with pytest.raises(AxBenchAdvError, match="no rows"):
        method_trajectory(_aggregate_table(), "Unknown")
    duplicate = _aggregate_table([_aggregate_table().to_pylist()[0]] * 2)
    with pytest.raises(AxBenchAdvError, match="duplicate"):
        method_trajectory(duplicate, "DiffMean")
    with pytest.raises(AxBenchAdvError, match="does not conform"):
        method_trajectory(pa.table({"bad": [1]}), "DiffMean")
