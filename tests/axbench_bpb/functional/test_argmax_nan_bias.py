"""Slice 17 audit (2026-05-05) P0.E regression test.

``compute_per_concept_rollup`` previously called ``idxmax()`` directly on
the full group's ``primary_margin`` column. When every value was NaN,
pandas silently returned the first index — emitting a per-concept row
with ``best_primary_margin = NaN`` that downstream macro rollup propagated.

Fix: skip the group entirely if no row has a finite primary_margin,
otherwise restrict ``idxmax`` to the valid subset.
"""

from __future__ import annotations

import math

import pandas as pd
import pyarrow as pa

from steering.eval.axbench_bpb.bpb_aggregation import compute_per_concept_rollup


def _build_minimal_per_tuple(
    *,
    primary_margin_values: list[float],
    n_rows_per_side: int = 5,
) -> pa.Table:
    """Build a minimal per_tuple Table with one (method, concept, mode, target)
    group spread over `len(primary_margin_values)` coefficients."""
    rows = []
    rows.append(
        {
            "method_name": "feature_diffmean_encdec",
            "concept_id": 0,
            "mode": "positive",
            "coefficient_str": "None",
            "target": "encdec",
            "mean_bpb_intervened_rich": 1.0,
            "mean_bpb_intervened_avoid": 1.0,
            "n_rich": n_rows_per_side,
            "n_avoid": n_rows_per_side,
            "n_truncated_rich": 0,
            "n_truncated_avoid": 0,
            "margin_intervened": 0.6,
            "primary_margin": 0.6,
            "primary_bpb": 1.0,
        }
    )
    for i, margin in enumerate(primary_margin_values):
        rows.append(
            {
                "method_name": "test_method",
                "concept_id": 0,
                "mode": "positive",
                "coefficient_str": str(float(i)),
                "target": "decoder_only",
                "mean_bpb_intervened_rich": 1.0,
                "mean_bpb_intervened_avoid": 1.0,
                "n_rich": n_rows_per_side,
                "n_avoid": n_rows_per_side,
                "n_truncated_rich": 0,
                "n_truncated_avoid": 0,
                "margin_intervened": margin,
                "primary_margin": margin,
                "primary_bpb": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    return pa.Table.from_pandas(df, preserve_index=False)


def test_all_nan_primary_margin_skips_group():
    """All-NaN primary_margin produces NO row for that method (group skipped)."""
    nan = float("nan")
    per_tuple = _build_minimal_per_tuple(
        primary_margin_values=[nan, nan, nan],
    )
    per_concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    assert not any(row["method_name"] == "test_method" for row in per_concept.to_pylist())


def test_partial_nan_uses_valid_subset_for_argmax():
    """If some values are finite, argmax is computed only over the finite subset."""
    nan = float("nan")
    # Coefficients sorted lexically: "0.0", "1.0", "2.0", "3.0"
    # Values:                        nan,    0.5,   nan,    0.7
    # Best should be coef "3.0" with margin=0.7.
    per_tuple = _build_minimal_per_tuple(
        primary_margin_values=[nan, 0.5, nan, 0.7],
    )
    per_concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    df = per_concept.to_pandas()
    row = df[df["method_name"] == "test_method"].iloc[0]
    assert row["best_coef"] == "3.0"
    assert math.isclose(row["best_primary_margin"], 0.7)


def test_no_nan_unchanged_behaviour():
    """No NaN at all → standard argmax behaviour preserved."""
    per_tuple = _build_minimal_per_tuple(
        primary_margin_values=[0.1, 0.5, 0.3],
    )
    per_concept = compute_per_concept_rollup(per_tuple, min_rows_per_side=1)
    row = per_concept.to_pandas()
    row = row[row["method_name"] == "test_method"].iloc[0]
    assert row["best_coef"] == "1.0"
    assert math.isclose(row["best_primary_margin"], 0.5)
