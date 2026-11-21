"""Aggregate conditioned AxBench-BPB score rows into leaderboard tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from steering.eval.axbench_base.class_margin import margin_sign
from steering.eval.axbench_bpb.representative_selection import (
    FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE,
    coefficient_abs,
    coefficient_lex,
)
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    LeaderboardMetric,
    SteeringMode,
    TargetModel,
)

_trapezoid = getattr(np, "trapezoid", np.trapz)


def compute_per_tuple_rollup(scores_tbl: pa.Table) -> pa.Table:
    """Aggregate per (method, concept, mode, coefficient, target).

    The conditioned-only metric is the class margin in teacher-forced BPB:
    ``BPB(avoiding) - BPB(rich)``. Positive-mode methods should make rich
    text easier than avoiding text, so the raw margin is the primary margin.
    Negative-mode methods should make avoiding text easier, so the sign is
    flipped to keep "larger is better" uniform.
    """
    df = scores_tbl.to_pandas()
    group_cols = ["method_name", "concept_id", "mode", "coefficient_str", "target"]

    def _agg_side(side_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if side_df.empty:
            cols = [
                *group_cols,
                f"mean_bpb_intervened_{suffix}",
                f"n_{suffix}",
                f"n_truncated_{suffix}",
            ]
            return pd.DataFrame(columns=cols)
        return side_df.groupby(group_cols, as_index=False).agg(
            **{
                f"mean_bpb_intervened_{suffix}": ("bpb_intervened", "mean"),
                f"n_{suffix}": ("sample_id", "count"),
                f"n_truncated_{suffix}": ("truncated", "sum"),
            }
        )

    rich_g = _agg_side(df[df["content_class"] == "concept_rich"], "rich")
    avoid_g = _agg_side(df[df["content_class"] == "concept_avoiding"], "avoid")
    merged = rich_g.merge(avoid_g, on=group_cols, how="outer")

    for col in ("n_rich", "n_avoid", "n_truncated_rich", "n_truncated_avoid"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int64")

    merged["margin_intervened"] = (
        merged["mean_bpb_intervened_avoid"] - merged["mean_bpb_intervened_rich"]
    )
    # The sign rule lives ONLY in steering.eval.axbench_base.class_margin; map()
    # over the mode column applies it elementwise and surfaces any unknown mode
    # loudly (ValueError) instead of silently treating it as positive.
    signs = merged["mode"].map(margin_sign)
    merged["primary_margin"] = signs * merged["margin_intervened"]
    is_negative = merged["mode"] == SteeringMode.NEGATIVE.value
    merged["primary_bpb"] = merged["mean_bpb_intervened_rich"].where(
        ~is_negative,
        merged["mean_bpb_intervened_avoid"],
    )
    return pa.Table.from_pandas(merged, preserve_index=False)


def compute_per_concept_rollup(
    per_tuple: pa.Table,
    *,
    min_rows_per_side: int,
) -> pa.Table:
    """Reduce per-tuple rows to per (method, concept, mode, target)."""
    df = per_tuple.to_pandas()
    df = df[(df["n_rich"] >= min_rows_per_side) & (df["n_avoid"] >= min_rows_per_side)]

    # Plain-BPB counterpart of axbench_bpb_adv.aggregation._feature_diffmean_target_scores:
    # the mean primary_margin of the fair feature-DiffMean encdec rows is the target
    # every method's representative coefficient is chosen closest to. The two are
    # intentionally NOT shared -- plain BPB has no adversarial axis (keys on
    # concept_id; no is_control filter) and operates on a pandas frame rather than
    # list-of-dicts, and axbench_bpb cannot import axbench_bpb_adv. Only the method-name
    # definition is shared, via FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE; keep the averaging
    # rule in step if you change one.
    feature_diffmean_targets: dict[tuple[int, str], float] = {}
    for mode, method_name in FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.items():
        target_rows = df[
            (df["mode"] == mode)
            & (df["method_name"] == method_name)
            & (df["target"] == TargetModel.ENCDEC.value)
            & (df["primary_margin"].notna())
        ]
        for concept_id, concept_group in target_rows.groupby("concept_id"):
            feature_diffmean_targets[(int(concept_id), mode)] = float(
                concept_group["primary_margin"].mean()
            )

    per_concept_rows: list[dict] = []
    group_keys = ["method_name", "concept_id", "mode", "target"]
    for group_key, group in df.groupby(group_keys):
        method, cid, mode, target = group_key
        group_sorted = group.sort_values("coefficient_str")

        valid_mask = group_sorted["primary_margin"].notna()
        if not valid_mask.any():
            continue

        target_score = feature_diffmean_targets.get((int(cid), str(mode)))
        if target_score is None:
            target_method = FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.get(str(mode))
            raise AxBenchBpbError(
                "missing fair feature-DiffMean target row for representative "
                f"selection: concept_id={int(cid)} mode={mode!r} "
                f"target_method={target_method!r} target='encdec'. "
                "Run scoring with feature_diffmean_encdec / "
                "neg_feature_diffmean_encdec enabled."
            )
        valid_rows = group_sorted.loc[valid_mask]
        # itertuples reads each row's values ONCE; the previous label-based
        # ``valid_rows.loc[idx, col]`` accesses re-did four (slow) pandas scalar
        # lookups per candidate, twice for primary_margin.
        best_idx = min(
            valid_rows.itertuples(),
            key=lambda row: (
                abs(float(row.primary_margin) - target_score),
                -float(row.primary_margin),
                coefficient_abs(row.coefficient_str),
                coefficient_lex(row.coefficient_str),
            ),
        ).Index
        best_row = group_sorted.loc[best_idx]

        coef_values: list[float] = []
        primary_margin_values: list[float] = []
        for _, row in group_sorted.iterrows():
            try:
                coef_value = float(row["coefficient_str"])
            except ValueError:
                continue
            coef_values.append(coef_value)
            primary_margin_values.append(float(row["primary_margin"]))

        if len(coef_values) >= 2:
            coef_arr = np.array(coef_values)
            order = np.argsort(coef_arr)
            margin_arr = np.array(primary_margin_values)[order]
            dose_auc_margin = float(_trapezoid(margin_arr, coef_arr[order]))
        else:
            dose_auc_margin = float("nan")

        side_totals = (group_sorted["n_rich"] + group_sorted["n_avoid"]).astype("int64")
        per_coef_truncated = (
            group_sorted["n_truncated_rich"] + group_sorted["n_truncated_avoid"]
        ).astype("int64")
        per_coef_rates = per_coef_truncated / side_totals.where(
            side_totals > 0,
            1,
        )

        per_concept_rows.append(
            {
                "method_name": method,
                "concept_id": int(cid),
                "mode": mode,
                "target": target,
                "best_coef": str(best_row["coefficient_str"]),
                "best_primary_margin": float(best_row["primary_margin"]),
                "best_margin_intervened": float(best_row["margin_intervened"]),
                "best_primary_bpb": float(best_row["primary_bpb"]),
                "dose_auc_margin": dose_auc_margin,
                "n_rich_total": int(group_sorted["n_rich"].iloc[0]),
                "n_avoid_total": int(group_sorted["n_avoid"].iloc[0]),
                "truncation_rate": float(per_coef_rates.mean()),
            }
        )

    return pa.Table.from_pylist(per_concept_rows)


def compute_macro_rollup(
    per_concept: pa.Table,
    *,
    invalid_warning_threshold: float,
    invalid_drop_threshold: float,
) -> tuple[pa.Table, dict]:
    """Reduce per-concept rows to per (method, mode, target)."""
    df = per_concept.to_pandas()
    warnings: list[str] = []
    dropped: list[str] = []

    macro_rows: list[dict] = []
    if not df.empty:
        for group_key, group in df.groupby(["method_name", "mode", "target"]):
            method, mode, target = group_key
            trunc_rate = float(group["truncation_rate"].mean())
            label = f"{method}/{mode}/{target}"
            if trunc_rate > invalid_drop_threshold:
                dropped.append(label)
                continue
            if trunc_rate > invalid_warning_threshold:
                warnings.append(f"{label}: truncation_rate={trunc_rate:.3f}")

            modes_series = group["best_coef"].mode()
            best_coef_mode = str(modes_series.iloc[0]) if len(modes_series) else ""

            macro_rows.append(
                {
                    "method_name": method,
                    "mode": mode,
                    "target": target,
                    "n_concepts_scored": int(len(group)),
                    LeaderboardMetric.MACRO_MEAN_BEST_PRIMARY_MARGIN.value: float(
                        group["best_primary_margin"].mean()
                    ),
                    LeaderboardMetric.MACRO_MEAN_BEST_MARGIN_INTERVENED.value: float(
                        group["best_margin_intervened"].mean()
                    ),
                    LeaderboardMetric.MACRO_MEAN_DOSE_AUC_MARGIN.value: float(
                        group["dose_auc_margin"].mean(skipna=True)
                    ),
                    LeaderboardMetric.MACRO_MEAN_TRUNCATION_RATE.value: trunc_rate,
                    "best_coef_mode": best_coef_mode,
                }
            )

    return (
        pa.Table.from_pylist(macro_rows),
        {"warnings": warnings, "dropped_methods": dropped},
    )


def write_leaderboard_csv(
    macro: pa.Table,
    path: Path,
    *,
    primary_metric: LeaderboardMetric | str,
) -> None:
    """Sort macro table by ``primary_metric`` descending and write CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if macro.num_rows == 0:
        fieldnames = [field.name for field in macro.schema]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        return

    if primary_metric not in [f.name for f in macro.schema]:
        raise AxBenchBpbError(
            f"primary_metric {primary_metric!r} not in macro table columns; "
            f"available: {[f.name for f in macro.schema]}"
        )

    df = macro.to_pandas().sort_values(primary_metric, ascending=False)
    df.to_csv(path, index=False)


__all__ = [
    "compute_macro_rollup",
    "compute_per_concept_rollup",
    "compute_per_tuple_rollup",
    "write_leaderboard_csv",
]
