"""Aggregate regenerated-control and GEPA proposal-level judge results."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from scripts.eval.axbench_adv.pressures import validate_magnitudes
from steering.eval.axbench_adv.artifacts import write_validated_parquet
from steering.eval.axbench_adv.best_point import per_concept_ratings
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.types import (
    ALL_METHODS,
    PLOT_EFFORT_LEVELS,
    validate_method_subset,
)

AGGREGATE_SCHEMA: pa.Schema = pa.schema(
    [
        ("method", pa.string()),
        ("proposal_count", pa.int32()),
        ("coefficient", pa.float64()),
        ("n_concepts", pa.int32()),
        ("concept_score", pa.float64()),
        ("fluency_score", pa.float64()),
        ("concept_score_sem", pa.float64()),
        ("fluency_score_sem", pa.float64()),
    ]
)


def _sem(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else statistics.stdev(values) / math.sqrt(len(values))


def validate_aggregate(table: pa.Table) -> None:
    """Validate aggregate schema, keys, ranges, and finite statistics."""

    if not isinstance(table, pa.Table) or not table.schema.equals(AGGREGATE_SCHEMA):
        raise AxBenchAdvError("table does not conform to AGGREGATE_SCHEMA")
    if table.num_rows == 0:
        raise AxBenchAdvError("aggregate table must not be empty")
    keys: set[tuple[str, int]] = set()
    concept_counts: set[int] = set()
    coefficients: dict[str, float] = {}
    for row in table.to_pylist():
        if any(value is None for value in row.values()):
            raise AxBenchAdvError("aggregate rows must not contain null values")
        key = (str(row["method"]), int(row["proposal_count"]))
        if key in keys:
            raise AxBenchAdvError(f"duplicate aggregate key {key}")
        keys.add(key)
        if key[0] not in ALL_METHODS:
            raise AxBenchAdvError(f"unknown aggregate method {key[0]!r}")
        if key[1] not in PLOT_EFFORT_LEVELS:
            raise AxBenchAdvError(f"aggregate proposal_count must be one of {PLOT_EFFORT_LEVELS}")
        n_concepts = int(row["n_concepts"])
        if n_concepts <= 0:
            raise AxBenchAdvError("aggregate n_concepts must be positive")
        concept_counts.add(n_concepts)
        coefficient = float(row["coefficient"])
        if not math.isfinite(coefficient):
            raise AxBenchAdvError("aggregate coefficient must be finite")
        if coefficient not in tuple(float(value) for value in ALL_METHOD_FACTORS[key[0]]):
            raise AxBenchAdvError("aggregate coefficient is outside the method's factor grid")
        if key[0] in coefficients and coefficients[key[0]] != coefficient:
            raise AxBenchAdvError("aggregate coefficient must be constant within each method")
        coefficients[key[0]] = coefficient
        for field in ("concept_score", "fluency_score"):
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise AxBenchAdvError(f"aggregate {field} must be finite in [0, 2]")
        for field in ("concept_score_sem", "fluency_score_sem"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise AxBenchAdvError(f"aggregate {field} must be finite and non-negative")
    if len(concept_counts) != 1:
        raise AxBenchAdvError("aggregate n_concepts must be identical at every point")
    methods = validate_method_subset(coefficients)
    expected_keys = {(method, level) for method in methods for level in PLOT_EFFORT_LEVELS}
    if keys != expected_keys:
        raise AxBenchAdvError("aggregate must contain the exact method/proposal-count grid")


def aggregate_point(
    steering_jsonl: Path | Sequence[Mapping[str, Any]],
    *,
    method: str,
    factor: float,
    proposal_count: int,
    expected_concept_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Aggregate one independently judged method/proposal-level point."""

    if isinstance(proposal_count, bool) or not isinstance(proposal_count, int):
        raise AxBenchAdvError("proposal_count must be an integer")
    if proposal_count not in PLOT_EFFORT_LEVELS:
        raise AxBenchAdvError(f"proposal_count must be one of {PLOT_EFFORT_LEVELS}")
    concept_values, fluency_values = per_concept_ratings(
        steering_jsonl,
        method,
        factor,
        expected_concept_ids=expected_concept_ids,
    )
    count = len(concept_values)
    return {
        "method": method,
        "proposal_count": proposal_count,
        "coefficient": float(factor),
        "n_concepts": count,
        "concept_score": sum(concept_values) / count,
        "fluency_score": sum(fluency_values) / count,
        "concept_score_sem": _sem(concept_values),
        "fluency_score_sem": _sem(fluency_values),
    }


def build_aggregate(
    *,
    best_factors: Mapping[str, float],
    judged_points: Mapping[tuple[str, int], Path | Sequence[Mapping[str, Any]]],
    expected_n_concepts: int,
    output_path: Path,
    expected_concept_ids: Sequence[int] | None = None,
) -> pa.Table:
    """Build every method's regenerated control plus independent GEPA levels."""

    validate_method_subset(best_factors)
    if type(expected_n_concepts) is not int or expected_n_concepts <= 0:
        raise AxBenchAdvError("expected_n_concepts must be a positive integer")
    levels = PLOT_EFFORT_LEVELS
    concept_ids = (
        tuple(range(expected_n_concepts))
        if expected_concept_ids is None
        else tuple(expected_concept_ids)
    )
    if (
        len(concept_ids) != expected_n_concepts
        or any(type(concept_id) is not int for concept_id in concept_ids)
        or concept_ids != tuple(sorted(set(concept_ids)))
    ):
        raise AxBenchAdvError(
            "expected_concept_ids must be unique, ascending, and match expected_n_concepts"
        )
    expected_keys = {(method, level) for method in best_factors for level in levels}
    observed_keys = set(judged_points)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AxBenchAdvError(
            f"judged point coverage differs from the aggregate grid; missing={missing}, extra={extra}"
        )
    rows = [
        aggregate_point(
            judged_points[(method, level)],
            method=method,
            factor=best_factors[method],
            proposal_count=level,
            expected_concept_ids=concept_ids,
        )
        for method in sorted(best_factors)
        for level in levels
    ]
    if any(row["n_concepts"] != len(concept_ids) for row in rows):
        raise AxBenchAdvError(f"judged points must each contain {expected_n_concepts} concepts")
    table = pa.Table.from_pylist(rows, schema=AGGREGATE_SCHEMA)
    write_validated_parquet(table, Path(output_path), validator=validate_aggregate)
    return table


__all__ = [
    "AGGREGATE_SCHEMA",
    "aggregate_point",
    "build_aggregate",
    "validate_aggregate",
]


COUNTER_AGGREGATE_SCHEMA = pa.schema(
    [
        ("magnitude", pa.float64()) if field.name == "proposal_count" else (field.name, field.type)
        for field in AGGREGATE_SCHEMA
    ]
    + [("mean_fitness", pa.float64())]
)


def validate_counter_aggregate(table: pa.Table) -> None:
    """Validate a complete, common magnitude grid, including an explicit zero."""
    if not table.schema.equals(COUNTER_AGGREGATE_SCHEMA) or not table.num_rows:
        raise AxBenchAdvError("table does not conform to COUNTER_AGGREGATE_SCHEMA")
    rows = table.to_pylist()
    if any(
        row["mean_fitness"] is None
        or not math.isfinite(row["mean_fitness"])
        or not 0 <= row["mean_fitness"] <= 1
        for row in rows
    ):
        raise AxBenchAdvError("counter mean_fitness must lie in [0, 1]")
    magnitudes = sorted({row["magnitude"] for row in rows if row["magnitude"] is not None})
    if (
        not magnitudes
        or magnitudes[0] != 0
        or any(not math.isfinite(v) or v < 0 for v in magnitudes)
    ):
        raise AxBenchAdvError("counter aggregate requires finite magnitudes and zero control")
    methods = validate_method_subset({row["method"] for row in rows})
    keys = [(row["method"], row["magnitude"]) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != {(m, v) for m in methods for v in magnitudes}:
        raise AxBenchAdvError("counter aggregate must contain the exact method/magnitude grid")
    # Reuse all range, factor, null and population checks from the GEPA schema,
    # validating each magnitude as a constant trajectory with the legacy levels.
    for magnitude in magnitudes:
        projected = []
        for row in rows:
            if row["magnitude"] == magnitude:
                base = {
                    key: value
                    for key, value in row.items()
                    if key not in {"magnitude", "mean_fitness"}
                }
                projected.extend({**base, "proposal_count": level} for level in PLOT_EFFORT_LEVELS)
        validate_aggregate(pa.Table.from_pylist(projected, schema=AGGREGATE_SCHEMA))
    if len({r["n_concepts"] for r in rows}) != 1 or any(
        len({r["coefficient"] for r in rows if r["method"] == m}) != 1 for m in methods
    ):
        raise AxBenchAdvError(
            "counter population and method factors must be constant across magnitudes"
        )


def build_counter_aggregate(
    *,
    best_factors: Mapping[str, float],
    judged_points: Mapping[tuple[str, float], Path | Sequence[Mapping[str, Any]]],
    magnitudes: Sequence[float],
    fitness: Mapping[tuple[str, float], float],
    concept_ids: Sequence[int],
    output_path: Path,
) -> pa.Table:
    """Use the identical per-concept judge reduction for each additive point."""
    levels = (0.0, *validate_magnitudes(magnitudes))
    if set(judged_points) != {(method, level) for method in best_factors for level in levels}:
        raise AxBenchAdvError("counter judged points differ from the method/magnitude grid")
    if set(fitness) != set(judged_points):
        raise AxBenchAdvError("counter fitness coverage differs from judged points")
    rows = []
    for method in sorted(best_factors):
        for magnitude in levels:
            row = aggregate_point(
                judged_points[(method, magnitude)],
                method=method,
                factor=best_factors[method],
                proposal_count=0,
                expected_concept_ids=concept_ids,
            )
            del row["proposal_count"]
            rows.append(
                {**row, "magnitude": magnitude, "mean_fitness": fitness[(method, magnitude)]}
            )
    table = pa.Table.from_pylist(rows, schema=COUNTER_AGGREGATE_SCHEMA)
    write_validated_parquet(table, output_path, validator=validate_counter_aggregate)
    return table


GCG_AGGREGATE_SCHEMA = pa.schema(
    [
        ("method", pa.string()),
        ("token_count", pa.int32()),
        ("coefficient", pa.float64()),
        ("n_concepts", pa.int32()),
        ("concept_score", pa.float64()),
        ("fluency_score", pa.float64()),
        ("instruction_score", pa.float64()),
        ("concept_score_sem", pa.float64()),
        ("fluency_score_sem", pa.float64()),
        ("instruction_score_sem", pa.float64()),
    ]
)


def validate_gcg_aggregate(table: pa.Table) -> None:
    """Validate the separate, integer-token GCG aggregate contract."""

    if not isinstance(table, pa.Table) or not table.schema.equals(GCG_AGGREGATE_SCHEMA):
        raise AxBenchAdvError("table does not conform to GCG_AGGREGATE_SCHEMA")
    if table.num_rows == 0:
        raise AxBenchAdvError("GCG aggregate table must not be empty")
    rows = table.to_pylist()
    if any(any(value is None for value in row.values()) for row in rows):
        raise AxBenchAdvError("GCG aggregate rows must not contain null values")
    methods = validate_method_subset({str(row["method"]) for row in rows})
    token_counts = sorted({int(row["token_count"]) for row in rows})
    if not token_counts or token_counts[0] != 0 or any(value < 0 for value in token_counts):
        raise AxBenchAdvError("GCG aggregate requires nonnegative token counts and zero control")
    keys = [(str(row["method"]), int(row["token_count"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (method, token_count) for method in methods for token_count in token_counts
    }:
        raise AxBenchAdvError("GCG aggregate must contain the exact method/token-count grid")
    populations = {int(row["n_concepts"]) for row in rows}
    if len(populations) != 1 or next(iter(populations)) <= 0:
        raise AxBenchAdvError("GCG n_concepts must be one constant positive population")
    for method in methods:
        coefficients = {float(row["coefficient"]) for row in rows if row["method"] == method}
        if len(coefficients) != 1 or next(iter(coefficients)) not in tuple(
            float(value) for value in ALL_METHOD_FACTORS[method]
        ):
            raise AxBenchAdvError("GCG coefficient must be constant and on the method grid")
    for row in rows:
        for field in ("concept_score", "fluency_score", "instruction_score"):
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise AxBenchAdvError(f"GCG {field} must be finite in [0, 2]")
        for field in (
            "concept_score_sem",
            "fluency_score_sem",
            "instruction_score_sem",
        ):
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise AxBenchAdvError(f"GCG {field} must be finite and nonnegative")


def build_gcg_aggregate(
    *,
    best_factors: Mapping[str, float],
    judged_points: Mapping[tuple[str, int], Path | Sequence[Mapping[str, Any]]],
    token_counts: Sequence[int],
    concept_ids: Sequence[int],
    output_path: Path,
) -> pa.Table:
    """Aggregate independently judged retained GCG suffixes over concepts."""

    methods = validate_method_subset(best_factors)
    counts = tuple(token_counts)
    concepts = tuple(concept_ids)
    if (
        not counts
        or any(type(value) is not int or value < 0 for value in counts)
        or counts != tuple(sorted(set(counts)))
        or counts[0] != 0
    ):
        raise AxBenchAdvError("GCG token_counts must be unique, ascending, and start at zero")
    if (
        not concepts
        or any(type(value) is not int or value < 0 for value in concepts)
        or concepts != tuple(sorted(set(concepts)))
    ):
        raise AxBenchAdvError("GCG concept_ids must be nonempty, unique, and ascending")
    expected = {(method, count) for method in methods for count in counts}
    if set(judged_points) != expected:
        raise AxBenchAdvError("GCG judged points differ from the method/token-count grid")
    rows: list[dict[str, Any]] = []
    for method in sorted(methods):
        factor = float(best_factors[method])
        for count in counts:
            source = judged_points[(method, count)]
            concept, fluency, instruction = per_concept_ratings(
                source,
                method,
                factor,
                expected_concept_ids=concepts,
                rating_fields=(
                    "relevance_concept_ratings",
                    "fluency_ratings",
                    "relevance_instruction_ratings",
                ),
            )
            rows.append(
                {
                    "method": method,
                    "token_count": count,
                    "coefficient": factor,
                    "n_concepts": len(concepts),
                    "concept_score": statistics.fmean(concept),
                    "fluency_score": statistics.fmean(fluency),
                    "instruction_score": statistics.fmean(instruction),
                    "concept_score_sem": _sem(concept),
                    "fluency_score_sem": _sem(fluency),
                    "instruction_score_sem": _sem(instruction),
                }
            )
    table = pa.Table.from_pylist(rows, schema=GCG_AGGREGATE_SCHEMA)
    write_validated_parquet(table, output_path, validator=validate_gcg_aggregate)
    return table


__all__.extend(
    [
        "GCG_AGGREGATE_SCHEMA",
        "build_gcg_aggregate",
        "validate_gcg_aggregate",
    ]
)
