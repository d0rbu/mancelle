"""Compare two AxBench-BPB-Adv round-trip score tables for semantic drift."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import pyarrow.parquet as pq

from steering.eval.axbench_bpb_adv.schemas import ROUNDTRIP_SCORE_SCHEMA

_KEY_COLUMNS = ("candidate_id", "sample_index")
_DRIFT_COLUMNS = frozenset(
    {
        "generated_text",
        "n_generated_tokens",
        "stop_reason",
        "control_generated_text",
        "control_n_generated_tokens",
        "control_stop_reason",
        "roundtrip_pre_act",
        "roundtrip_post_act",
        "roundtrip_percentile",
        "control_roundtrip_pre_act",
        "control_roundtrip_post_act",
        "control_roundtrip_percentile",
    }
)
_CONTROL_DRIFT_COLUMNS = frozenset(
    {
        "control_generated_text",
        "control_n_generated_tokens",
        "control_stop_reason",
        "control_generation_seed",
        "control_roundtrip_pre_act",
        "control_roundtrip_post_act",
        "control_roundtrip_percentile",
        "control_n_roundtrip_features",
    }
)
_RESEEDING_DRIFT_COLUMNS = _DRIFT_COLUMNS | {
    "generation_seed",
    "control_generation_seed",
    "n_roundtrip_features",
    "control_n_roundtrip_features",
}
_EXACT_COLUMN_SET = frozenset(ROUNDTRIP_SCORE_SCHEMA.names) - _DRIFT_COLUMNS
_EXACT_COLUMNS = tuple(
    column for column in ROUNDTRIP_SCORE_SCHEMA.names if column in _EXACT_COLUMN_SET
)
_SCIENTIFIC_DRIFT_COLUMNS = (
    "roundtrip_pre_act",
    "roundtrip_post_act",
    "roundtrip_percentile",
    "control_roundtrip_pre_act",
    "control_roundtrip_post_act",
    "control_roundtrip_percentile",
)

if set(ROUNDTRIP_SCORE_SCHEMA.names) != set(_EXACT_COLUMNS) | _DRIFT_COLUMNS:
    raise RuntimeError("roundtrip comparison columns do not partition the score schema")


class RoundtripScoreComparisonError(ValueError):
    """Raised when input tables cannot be compared safely."""


def _row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["candidate_id"]), int(row["sample_index"])


def _load_rows(
    path: Path,
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], dict[str, Any]]]:
    table = pq.read_table(path)
    if not table.schema.equals(ROUNDTRIP_SCORE_SCHEMA):
        raise RoundtripScoreComparisonError(
            f"roundtrip score schema mismatch at {path}: expected "
            f"ROUNDTRIP_SCORE_SCHEMA, observed {table.schema}"
        )
    table_rows = table.to_pylist()
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    ordered_keys: list[tuple[str, int]] = []
    for row in table_rows:
        key = _row_key(row)
        ordered_keys.append(key)
        if key in rows:
            duplicates.append(key)
        rows[key] = row
    if duplicates:
        raise RoundtripScoreComparisonError(
            f"duplicate {_KEY_COLUMNS} keys at {path}: {duplicates[:10]}"
        )
    return ordered_keys, rows


def _row_order_violation(
    keys_a: Sequence[tuple[str, int]],
    keys_b: Sequence[tuple[str, int]],
) -> dict[str, Any] | None:
    for position, (key_a, key_b) in enumerate(zip(keys_a, keys_b, strict=False)):
        if key_a != key_b:
            return {
                "first_differing_position": position,
                "table_a_key": key_a,
                "table_b_key": key_b,
            }
    if len(keys_a) != len(keys_b):
        position = min(len(keys_a), len(keys_b))
        return {
            "first_differing_position": position,
            "table_a_key": keys_a[position] if position < len(keys_a) else None,
            "table_b_key": keys_b[position] if position < len(keys_b) else None,
        }
    return None


def _distribution_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "std": pstdev(values) if values else 0.0,
    }


def _max_empirical_cdf_gap(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """Return the two-sample maximum empirical CDF gap."""

    sorted_a = sorted(values_a)
    sorted_b = sorted(values_b)
    if not sorted_a and not sorted_b:
        return 0.0
    index_a = 0
    index_b = 0
    max_gap = 0.0
    while index_a < len(sorted_a) or index_b < len(sorted_b):
        next_a = sorted_a[index_a] if index_a < len(sorted_a) else float("inf")
        next_b = sorted_b[index_b] if index_b < len(sorted_b) else float("inf")
        value = min(next_a, next_b)
        while index_a < len(sorted_a) and sorted_a[index_a] <= value:
            index_a += 1
        while index_b < len(sorted_b) and sorted_b[index_b] <= value:
            index_b += 1
        cdf_a = index_a / len(sorted_a) if sorted_a else 0.0
        cdf_b = index_b / len(sorted_b) if sorted_b else 0.0
        max_gap = max(max_gap, abs(cdf_a - cdf_b))
    return max_gap


def _validate_finite_drift_values(
    *,
    table_name: str,
    rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    for key, row in rows.items():
        for column in _DRIFT_COLUMNS | _CONTROL_DRIFT_COLUMNS:
            value = row[column]
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise RoundtripScoreComparisonError(
                    f"non-finite numeric drift value in table {table_name}, "
                    f"row key {key!r}, column {column!r}: {value!r}"
                )


def compare_roundtrip_score_tables(
    path_a: Path,
    path_b: Path,
    *,
    expect_control_drift: bool = False,
    expect_reseeding: bool = False,
) -> dict[str, Any]:
    """Return exact-contract and tolerated-drift statistics for two tables."""
    if expect_control_drift and expect_reseeding:
        raise RoundtripScoreComparisonError(
            "expect_control_drift and expect_reseeding are mutually exclusive"
        )
    ordered_keys_a, rows_a = _load_rows(path_a)
    ordered_keys_b, rows_b = _load_rows(path_b)
    keys_a = set(rows_a)
    keys_b = set(rows_b)
    if keys_a != keys_b:
        missing_from_a = sorted(keys_b - keys_a)
        missing_from_b = sorted(keys_a - keys_b)
        raise RoundtripScoreComparisonError(
            "roundtrip score key sets differ: "
            f"missing from A ({len(missing_from_a)}): {missing_from_a[:10]}; "
            f"missing from B ({len(missing_from_b)}): {missing_from_b[:10]}"
        )
    _validate_finite_drift_values(table_name="A", rows=rows_a)
    _validate_finite_drift_values(table_name="B", rows=rows_b)

    order_violation = _row_order_violation(ordered_keys_a, ordered_keys_b)
    row_order_equal = order_violation is None
    keys = ordered_keys_a
    if expect_reseeding:
        drift_columns = _RESEEDING_DRIFT_COLUMNS
    elif expect_control_drift:
        drift_columns = _CONTROL_DRIFT_COLUMNS
    else:
        drift_columns = _DRIFT_COLUMNS
    exact_column_names = tuple(
        column
        for column in ROUNDTRIP_SCORE_SCHEMA.names
        if column not in drift_columns
    )
    exact_columns: dict[str, dict[str, Any]] = {}
    exact_contract_equal = row_order_equal
    for column in exact_column_names:
        mismatched_keys = [
            key for key in keys if rows_a[key][column] != rows_b[key][column]
        ]
        equal = not mismatched_keys
        exact_contract_equal = exact_contract_equal and equal
        exact_columns[column] = {
            "equal": equal,
            "n_mismatched": len(mismatched_keys),
            "mismatched_key_examples": mismatched_keys[:10],
        }

    n_rows = len(keys)
    n_identical_generated_text = sum(
        rows_a[key]["generated_text"] == rows_b[key]["generated_text"] for key in keys
    )
    n_identical_control_text = sum(
        rows_a[key]["control_generated_text"] == rows_b[key]["control_generated_text"]
        for key in keys
    )
    allowlisted_drift: dict[str, dict[str, Any]] = {}
    for column in ROUNDTRIP_SCORE_SCHEMA.names:
        if column not in drift_columns:
            continue
        n_equal = sum(rows_a[key][column] == rows_b[key][column] for key in keys)
        allowlisted_drift[column] = {
            "n_equal": n_equal,
            "n_different": n_rows - n_equal,
            "identity_rate": n_equal / n_rows if n_rows else 1.0,
        }

    numeric_drift: dict[str, dict[str, float]] = {}
    aggregate_means: dict[str, dict[str, float]] = {}
    for column in _SCIENTIFIC_DRIFT_COLUMNS:
        values_a = [float(rows_a[key][column]) for key in keys]
        values_b = [float(rows_b[key][column]) for key in keys]
        deltas = [
            abs(value_a - value_b)
            for value_a, value_b in zip(values_a, values_b, strict=True)
        ]
        numeric_drift[column] = {
            "max_abs_delta": max(deltas, default=0.0),
            "mean_abs_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        }
        mean_before = sum(values_a) / len(values_a) if values_a else 0.0
        mean_after = sum(values_b) / len(values_b) if values_b else 0.0
        aggregate_means[column] = {
            "mean_before": mean_before,
            "mean_after": mean_after,
            "delta": mean_after - mean_before,
        }

    control_percentiles_a = [
        float(rows_a[key]["control_roundtrip_percentile"]) for key in keys
    ]
    control_percentiles_b = [
        float(rows_b[key]["control_roundtrip_percentile"]) for key in keys
    ]
    control_seed_mismatches = sum(
        rows_a[key]["control_generation_seed"]
        != rows_b[key]["control_generation_seed"]
        for key in keys
    )

    return {
        "table_a": str(path_a.resolve()),
        "table_b": str(path_b.resolve()),
        "n_rows": n_rows,
        "expect_control_drift": expect_control_drift,
        "expect_reseeding": expect_reseeding,
        "exact_contract_equal": exact_contract_equal,
        "row_order_equal": row_order_equal,
        "row_order_violation": order_violation,
        "exact_columns": exact_columns,
        "allowlisted_drift": allowlisted_drift,
        "n_identical_generated_text": n_identical_generated_text,
        "generated_text_identity_rate": (
            n_identical_generated_text / n_rows if n_rows else 1.0
        ),
        "n_identical_control_text": n_identical_control_text,
        "control_text_identity_rate": (
            n_identical_control_text / n_rows if n_rows else 1.0
        ),
        "control_drift": {
            "control_generation_seed": {
                "n_equal": n_rows - control_seed_mismatches,
                "n_different": control_seed_mismatches,
                "identity_rate": (
                    (n_rows - control_seed_mismatches) / n_rows if n_rows else 1.0
                ),
            },
            "control_generated_text": {
                "n_equal": n_identical_control_text,
                "n_different": n_rows - n_identical_control_text,
                "identity_rate": (
                    n_identical_control_text / n_rows if n_rows else 1.0
                ),
            },
            "control_roundtrip_percentile_distribution": {
                "table_a": _distribution_stats(control_percentiles_a),
                "table_b": _distribution_stats(control_percentiles_b),
                "max_empirical_cdf_gap": _max_empirical_cdf_gap(
                    control_percentiles_a,
                    control_percentiles_b,
                ),
            },
        },
        "numeric_drift": numeric_drift,
        "aggregate_means": aggregate_means,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_a", type=Path)
    parser.add_argument("table_b", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    drift_group = parser.add_mutually_exclusive_group()
    drift_group.add_argument(
        "--expect-control-drift",
        action="store_true",
        help=(
            "report but do not fail on control seed drift; steered seeds and "
            "all identity columns remain exact"
        ),
    )
    drift_group.add_argument(
        "--expect-reseeding",
        action="store_true",
        help=(
            "report but do not fail on PR7 generation reseeding and downstream "
            "probe drift; identity, prompt, editable-span, and teacher-forced "
            "loss columns remain exact"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = compare_roundtrip_score_tables(
            args.table_a,
            args.table_b,
            expect_control_drift=args.expect_control_drift,
            expect_reseeding=args.expect_reseeding,
        )
    except RoundtripScoreComparisonError as exc:
        report = {
            "table_a": str(args.table_a.resolve()),
            "table_b": str(args.table_b.resolve()),
            "exact_contract_equal": False,
            "comparison_error": str(exc),
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered, encoding="utf-8")
    return 0 if report["exact_contract_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
