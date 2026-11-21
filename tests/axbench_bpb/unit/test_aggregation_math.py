from __future__ import annotations

import pyarrow as pa
import pytest

from steering.eval.axbench_bpb.bpb_aggregation import (
    compute_macro_rollup,
    compute_per_concept_rollup,
    compute_per_tuple_rollup,
    write_leaderboard_csv,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError


def _score_row(
    *,
    sample_id: str,
    content_class: str,
    bpb: float,
    mode: str = "positive",
    coefficient_str: str = "0",
    method_name: str = "m",
    target: str = "encdec",
    truncated: bool = False,
) -> dict:
    return {
        "method_name": method_name,
        "concept_id": 1,
        "mode": mode,
        "content_class": content_class,
        "target": target,
        "coefficient_str": coefficient_str,
        "sample_id": sample_id,
        "bpb_intervened": bpb,
        "n_output_tokens": 10,
        "n_output_bytes": 40,
        "truncated": truncated,
    }


def test_per_tuple_positive_primary_margin_is_avoid_minus_rich():
    scores = pa.Table.from_pylist(
        [
            _score_row(sample_id="r0", content_class="concept_rich", bpb=1.0),
            _score_row(sample_id="a0", content_class="concept_avoiding", bpb=2.0),
        ]
    )
    row = compute_per_tuple_rollup(scores).to_pylist()[0]
    assert row["margin_intervened"] == pytest.approx(1.0)
    assert row["primary_margin"] == pytest.approx(1.0)


def test_per_tuple_negative_flips_primary_margin():
    scores = pa.Table.from_pylist(
        [
            _score_row(
                sample_id="r0",
                content_class="concept_rich",
                bpb=2.0,
                mode="negative",
            ),
            _score_row(
                sample_id="a0",
                content_class="concept_avoiding",
                bpb=1.0,
                mode="negative",
            ),
        ]
    )
    row = compute_per_tuple_rollup(scores).to_pylist()[0]
    assert row["margin_intervened"] == pytest.approx(-1.0)
    assert row["primary_margin"] == pytest.approx(1.0)


def test_per_concept_selects_coefficient_closest_to_feature_diffmean_target():
    per_tuple = pa.Table.from_pylist(
        [
            {
                "method_name": "feature_diffmean_encdec",
                "concept_id": 1,
                "mode": "positive",
                "target": "encdec",
                "coefficient_str": "None",
                "mean_bpb_intervened_rich": 1.0,
                "mean_bpb_intervened_avoid": 1.4,
                "n_rich": 2,
                "n_avoid": 2,
                "n_truncated_rich": 0,
                "n_truncated_avoid": 0,
                "margin_intervened": 0.4,
                "primary_margin": 0.4,
                "primary_bpb": 1.0,
            },
            {
                "method_name": "m",
                "concept_id": 1,
                "mode": "positive",
                "target": "encdec",
                "coefficient_str": "0",
                "mean_bpb_intervened_rich": 1.0,
                "mean_bpb_intervened_avoid": 1.2,
                "n_rich": 2,
                "n_avoid": 2,
                "n_truncated_rich": 0,
                "n_truncated_avoid": 0,
                "margin_intervened": 0.2,
                "primary_margin": 0.2,
                "primary_bpb": 1.0,
            },
            {
                "method_name": "m",
                "concept_id": 1,
                "mode": "positive",
                "target": "encdec",
                "coefficient_str": "1",
                "mean_bpb_intervened_rich": 1.0,
                "mean_bpb_intervened_avoid": 1.7,
                "n_rich": 2,
                "n_avoid": 2,
                "n_truncated_rich": 0,
                "n_truncated_avoid": 0,
                "margin_intervened": 0.7,
                "primary_margin": 0.7,
                "primary_bpb": 1.0,
            },
        ]
    )
    rows = compute_per_concept_rollup(per_tuple, min_rows_per_side=1).to_pylist()
    row = next(row for row in rows if row["method_name"] == "m")
    assert row["best_coef"] == "0"
    assert row["best_primary_margin"] == pytest.approx(0.2)


def _per_tuple_row(
    *,
    method_name: str,
    coefficient_str: str,
    primary_margin: float,
    concept_id: int = 1,
    mode: str = "positive",
    target: str = "encdec",
) -> dict:
    return {
        "method_name": method_name,
        "concept_id": concept_id,
        "mode": mode,
        "target": target,
        "coefficient_str": coefficient_str,
        "mean_bpb_intervened_rich": 1.0,
        "mean_bpb_intervened_avoid": 1.0 + primary_margin,
        "n_rich": 2,
        "n_avoid": 2,
        "n_truncated_rich": 0,
        "n_truncated_avoid": 0,
        "margin_intervened": primary_margin,
        "primary_margin": primary_margin,
        "primary_bpb": 1.0,
    }


def test_per_concept_closest_to_feature_diffmean_tie_prefers_higher_score():
    per_tuple = pa.Table.from_pylist(
        [
            _per_tuple_row(
                method_name="feature_diffmean_encdec",
                coefficient_str="None",
                primary_margin=0.75,
            ),
            _per_tuple_row(
                method_name="m",
                coefficient_str="0",
                primary_margin=0.5,
            ),
            _per_tuple_row(
                method_name="m",
                coefficient_str="1",
                primary_margin=1.0,
            ),
        ]
    )
    rows = compute_per_concept_rollup(
        per_tuple,
        min_rows_per_side=1,
    ).to_pylist()
    row = next(row for row in rows if row["method_name"] == "m")
    assert row["best_coef"] == "1"
    assert row["best_primary_margin"] == pytest.approx(1.0)


def test_per_concept_closest_to_feature_diffmean_missing_target_raises():
    per_tuple = pa.Table.from_pylist(
        [
            _per_tuple_row(
                method_name="m",
                coefficient_str="0",
                primary_margin=0.5,
            ),
            _per_tuple_row(
                method_name="m",
                coefficient_str="1",
                primary_margin=1.0,
            ),
        ]
    )
    with pytest.raises(AxBenchBpbError, match="missing fair feature-DiffMean"):
        compute_per_concept_rollup(
            per_tuple,
            min_rows_per_side=1,
        )


def test_macro_rollup_uses_conditioned_margin_columns():
    per_concept = pa.Table.from_pylist(
        [
            {
                "method_name": "m",
                "concept_id": 1,
                "mode": "positive",
                "target": "encdec",
                "best_coef": "0",
                "best_primary_margin": 0.4,
                "best_margin_intervened": 0.4,
                "best_primary_bpb": 1.0,
                "dose_auc_margin": 0.1,
                "n_rich_total": 2,
                "n_avoid_total": 2,
                "truncation_rate": 0.0,
            },
            {
                "method_name": "m",
                "concept_id": 2,
                "mode": "positive",
                "target": "encdec",
                "best_coef": "0",
                "best_primary_margin": 0.6,
                "best_margin_intervened": 0.6,
                "best_primary_bpb": 1.0,
                "dose_auc_margin": 0.3,
                "n_rich_total": 2,
                "n_avoid_total": 2,
                "truncation_rate": 0.0,
            },
        ]
    )
    macro, report = compute_macro_rollup(
        per_concept,
        invalid_warning_threshold=0.1,
        invalid_drop_threshold=0.2,
    )
    row = macro.to_pylist()[0]
    assert row["macro_mean_best_primary_margin"] == pytest.approx(0.5)
    assert row["macro_mean_dose_auc_margin"] == pytest.approx(0.2)
    assert report == {"warnings": [], "dropped_methods": []}


def test_write_leaderboard_rejects_unknown_metric(tmp_path):
    macro = pa.Table.from_pylist([{"method_name": "m", "macro_mean_best_primary_margin": 0.5}])
    with pytest.raises(AxBenchBpbError):
        write_leaderboard_csv(macro, tmp_path / "leaderboard.csv", primary_metric="bad")
