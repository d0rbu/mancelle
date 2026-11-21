"""Negative-mode AUC uses the sign-corrected conditioned margin."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from steering.eval.axbench_bpb.bpb_aggregation import compute_per_concept_rollup

_trapezoid = getattr(np, "trapezoid", np.trapz)


def _per_tuple_row(
    *,
    mode: str,
    coef: float | str,
    raw_margin: float,
    method_name: str = "test_method",
    target: str = "decoder_only",
) -> dict:
    primary_margin = -raw_margin if mode == "negative" else raw_margin
    return {
        "method_name": method_name,
        "concept_id": 0,
        "mode": mode,
        "coefficient_str": str(coef),
        "target": target,
        "n_rich": 10,
        "n_avoid": 10,
        "n_truncated_rich": 0,
        "n_truncated_avoid": 0,
        "mean_bpb_intervened_rich": 1.0,
        "mean_bpb_intervened_avoid": 1.0 + raw_margin,
        "margin_intervened": raw_margin,
        "primary_margin": primary_margin,
        "primary_bpb": 1.0,
    }


def test_per_concept_negative_mode_auc_uses_flipped_margin():
    coefs = [1.0, 2.0, 3.0]
    rows = [
        _per_tuple_row(
            mode="negative",
            coef="None",
            raw_margin=-0.5,
            method_name="neg_feature_diffmean_encdec",
            target="encdec",
        ),
        *[_per_tuple_row(mode="negative", coef=c, raw_margin=-0.5) for c in coefs],
    ]
    out = compute_per_concept_rollup(pa.Table.from_pylist(rows), min_rows_per_side=1).to_pandas()
    out = out[out["method_name"] == "test_method"]
    assert len(out) == 1
    expected = float(_trapezoid([0.5, 0.5, 0.5], coefs))
    assert out["dose_auc_margin"].iloc[0] == pytest.approx(expected)


def test_per_concept_positive_mode_auc_uses_raw_margin():
    coefs = [1.0, 2.0, 3.0]
    rows = [
        _per_tuple_row(
            mode="positive",
            coef="None",
            raw_margin=0.5,
            method_name="feature_diffmean_encdec",
            target="encdec",
        ),
        *[_per_tuple_row(mode="positive", coef=c, raw_margin=0.5) for c in coefs],
    ]
    out = compute_per_concept_rollup(pa.Table.from_pylist(rows), min_rows_per_side=1).to_pandas()
    out = out[out["method_name"] == "test_method"]
    assert len(out) == 1
    expected = float(_trapezoid([0.5, 0.5, 0.5], coefs))
    assert out["dose_auc_margin"].iloc[0] == pytest.approx(expected)
