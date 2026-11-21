"""Stage 6 leaderboard aggregation math helpers plus ``aggregate()`` orchestration.

The bottom half of the module (``hmean``, ``best_of_grid``, ``bootstrap_ci``)
holds the pure math helpers — no I/O, no parquet, no dataframes. They are unit-tested in
``tests/axbench_base/unit/test_aggregation_math.py`` and stayed
untouched when Task 40 appended the orchestration.

The top-half :func:`aggregate` function joins the Stage 4
generations directory-dataset with the Stage 5 judge scores
directory-dataset on ``generation_key``, computes per-method
best-of-grid harmonic-mean scores, and writes three reports:

- :func:`paths.report_yaml` — per-method mean + CI plus a top-level
  ``metadata`` block.
- :func:`paths.leaderboard_csv` — flat sorted leaderboard.
- :func:`paths.aggregation_report` — diagnostic counts per stage.

Methods whose judge invalid-rate crosses
``cfg.judge.invalid_drop_threshold`` are EXCLUDED from the leaderboard;
methods that cross ``cfg.judge.invalid_warning_threshold`` are kept but
logged with a warning. The row-score formula flips the concept axis in
negative mode (``JUDGE_MAX_SCORE - concept_score``) so "low concept
presence" reads as "high score".
"""

from __future__ import annotations

import csv
import datetime as _datetime
import math
import random
from pathlib import Path

import yaml
from loguru import logger

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.judge_parser import JUDGE_MAX_SCORE
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.parquet_writer import load_existing
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode


def hmean(a: float, b: float) -> float:
    """Harmonic mean with zero-bottom per AxBench.

    If either input is 0, returns 0.0 (AxBench's "zero-bottom" rule).
    Otherwise returns the standard 2ab/(a+b).

    AxBench's per-example callers pass integer ratings. AxBench-Adv passes
    non-negative macro-averaged floating-point ratings. The calculation is
    identical for both; this widened annotation documents the shared contract.
    The helper never inspects ``JUDGE_MAX_SCORE`` itself.
    """
    if a == 0 or b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def best_of_grid(scores: list[float]) -> float:
    """Return max(scores). Raises AxBenchBaseError on empty list."""
    if not scores:
        raise AxBenchBaseError("best_of_grid received an empty scores list.")
    return max(scores)


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval (lower, upper) at alpha level.

    Uses ``random.Random(seed).choices`` to resample with replacement, computes
    the mean of each bootstrap sample, and returns the (alpha/2, 1 - alpha/2)
    percentiles.

    Single-value input returns (value, value).
    Empty input raises AxBenchBaseError.
    """
    if not values:
        raise AxBenchBaseError("bootstrap_ci received an empty values list.")

    if len(values) == 1:
        v = values[0]
        return (v, v)

    rng = random.Random(seed)
    n = len(values)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(values, k=n)
        boot_means.append(sum(sample) / n)

    boot_means.sort()

    lower_idx = math.floor((alpha / 2) * n_boot)
    upper_idx = math.floor((1.0 - alpha / 2) * n_boot)
    # Clamp upper index to last valid position
    upper_idx = min(upper_idx, n_boot - 1)

    return (boot_means[lower_idx], boot_means[upper_idx])


# ===========================================================================
# Stage 6 orchestration — ``aggregate``
# ===========================================================================


def _registry_mode_lookup() -> dict[str, SteeringMode]:
    """Build a ``{method_name: mode}`` lookup over the static registry.

    The joined rows already carry a ``mode`` column from Stage 4. Pinning
    the mode to the registry keeps aggregation self-consistent with Stage
    4's dispatch.
    """
    return {m.name: m.mode for m in ALL_METHODS}


def _compute_row_score(
    *,
    concept_score: int,
    fluency_score: int,
    mode: SteeringMode,
) -> float:
    """Return the per-row score used by best-of-grid.

    Positive mode: ``hmean(concept, fluency)`` — both want to be high.
    Negative mode: ``hmean(JUDGE_MAX_SCORE - concept, fluency)`` — we
    want LOW concept presence, so we flip the axis. Fluency always reads
    "high is good".
    """
    if mode == SteeringMode.NEGATIVE:
        return hmean(JUDGE_MAX_SCORE - concept_score, fluency_score)
    return hmean(concept_score, fluency_score)


def _build_judge_lookup(
    judge_rows: list[dict],
) -> dict[str, dict]:
    """Index judge rows by ``generation_key`` for an O(1) join."""
    out: dict[str, dict] = {}
    for row in judge_rows:
        out[row["generation_key"]] = row
    return out


def assert_complete_judge_coverage(
    generation_rows: list[dict],
    judge_rows: list[dict],
) -> None:
    """Raise when any Stage 4 generation lacks a Stage 5 judge row."""
    judge_keys = {row["generation_key"] for row in judge_rows}
    missing_keys = [
        row["generation_key"]
        for row in generation_rows
        if row["generation_key"] not in judge_keys
    ]
    if not missing_keys:
        return

    missing_examples = list(dict.fromkeys(missing_keys))[:5]
    raise AxBenchBaseError(
        "Stage 6 requires complete judge coverage: "
        f"{len(missing_keys)} missing of {len(generation_rows)} generation rows "
        f"with {len(judge_rows)} judge rows; missing generation_key examples: "
        f"{missing_examples!r}. Run or resume Stage 5 (judge_all) before Stage 6."
    )


def _row_is_invalid(judge_row: dict) -> bool:
    """Return True if either judge axis failed to parse."""
    return bool(judge_row["concept_invalid"]) or bool(judge_row["fluency_invalid"])


def _method_aggregate_entry(
    *,
    method_name: str,
    mode: SteeringMode,
    per_concept_scores: list[float],
    n_valid_rows: int,
    n_invalid_rows: int,
) -> dict:
    """Build the per-method report-entry dict.

    ``per_concept_scores`` is the list of best-of-grid scores, one per
    concept.
    """
    mean_score = sum(per_concept_scores) / len(per_concept_scores)
    ci_lower, ci_upper = bootstrap_ci(per_concept_scores)
    n_total = n_valid_rows + n_invalid_rows
    invalid_rate = n_invalid_rows / n_total if n_total > 0 else 0.0
    entry: dict = {
        "mean": mean_score,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_concepts": len(per_concept_scores),
        "n_valid_rows": n_valid_rows,
        "n_invalid_rows": n_invalid_rows,
        "invalid_rate": invalid_rate,
    }
    del method_name, mode
    return entry


def _write_report_yaml(
    report: dict,
    out_path: Path,
) -> None:
    """Serialize the full report dict to YAML."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)


def _write_leaderboard_csv(
    report: dict,
    mode_lookup: dict[str, SteeringMode],
    out_path: Path,
) -> None:
    """Serialize the per-method entries to a flat CSV, sorted within mode.

    The ``report`` dict passed in MUST be the per-method-only slice (no
    ``metadata`` key) — caller responsibility.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method_rows: list[dict] = []
    for method_name, entry in report.items():
        mode_value = mode_lookup.get(method_name)
        method_rows.append(
            {
                "method_name": method_name,
                # Write the raw string value rather than the enum repr so
                # the CSV stays byte-identical to the pre-enum output.
                "mode": str(mode_value) if mode_value is not None else "",
                "mean": float(entry["mean"]),
                "ci_lower": float(entry["ci_lower"]),
                "ci_upper": float(entry["ci_upper"]),
                "n_valid_rows": int(entry["n_valid_rows"]),
            },
        )
    # Sort by mode (stable), then by mean desc within mode.
    method_rows.sort(key=lambda r: (r["mode"], -r["mean"]))
    fieldnames = [
        "method_name",
        "mode",
        "mean",
        "ci_lower",
        "ci_upper",
        "n_valid_rows",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_rows:
            writer.writerow(row)


def _write_aggregation_report(
    diag: dict,
    out_path: Path,
) -> None:
    """Serialize the diagnostic counts dict to YAML."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(diag, f, sort_keys=False)


def aggregate(
    cfg: AxBenchBaseConfig,
) -> Path:
    """Stage 6 entry point: join, score, aggregate, and write reports.

    Joins the Stage 4 generations directory-dataset with the Stage 5
    judge scores directory-dataset on ``generation_key``, computes
    per-row harmonic-mean scores (positive mode: ``hmean(concept,
    fluency)``; negative mode: ``hmean(2 - concept, fluency)``),
    best-of-grid-reduces per concept, and takes the mean across
    concepts with a 95% bootstrap CI for each method.

    Parameters
    ----------
    cfg:
        Pipeline config. ``cfg.run_tag`` must be set.

    Returns
    -------
    Path
        The ``report.yaml`` path (also written: ``leaderboard.csv``
        and ``aggregation_report.yaml``).

    Raises
    ------
    AxBenchBaseError
        If ``cfg.run_tag`` is unset, the generations directory is
        missing, the judge_scores directory is missing, or any
        generation has no matching judge row.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling aggregate",
        )
    run_tag = cfg.run_tag
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    gen_dir = output_paths.generations_dir()
    if not gen_dir.exists():
        raise AxBenchBaseError(
            f"generations directory does not exist: {gen_dir}. Run Stage 4 (generate_all) first.",
        )
    judge_dir = output_paths.judge_scores_dir()
    if not judge_dir.exists():
        raise AxBenchBaseError(
            f"judge_scores directory does not exist: {judge_dir}. Run Stage 5 (judge_all) first.",
        )

    # Load tables + convert to pylist so pyarrow dependence stays at the
    # boundary. The join is a plain dict + loop — simple, ~O(N) and
    # pyarrow isn't particularly fast on row-wise access anyway.
    with profile_region("stage6.load_generations", path=gen_dir):
        generations = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
        gen_rows = generations.to_pylist()
    with profile_region("stage6.load_judge_scores", path=judge_dir):
        judges = load_existing(judge_dir, schema=_JUDGE_SCORES_SCHEMA)
        judge_rows = judges.to_pylist()
    assert_complete_judge_coverage(gen_rows, judge_rows)
    with profile_region("stage6.build_judge_lookup", rows=len(judge_rows)):
        judge_lookup = _build_judge_lookup(judge_rows)

    # Accumulators keyed on method_name.
    # per_method_best_scores: {method_name: {concept_id: [row_scores]}}
    per_method_best_scores: dict[str, dict[int, list[float]]] = {}
    per_method_n_valid: dict[str, int] = {}
    per_method_n_invalid: dict[str, int] = {}

    n_joined = 0
    n_invalid = 0

    with profile_region("stage6.join_and_score_rows", rows=len(gen_rows)):
        for gen_row in gen_rows:
            key = gen_row["generation_key"]
            judge_row = judge_lookup[key]
            n_joined += 1
            method_name = gen_row["method_name"]
            # Parquet stores ``mode`` as ``pa.string()``; round-trip it
            # through :class:`SteeringMode` so downstream comparisons are
            # strongly typed.
            mode = SteeringMode(gen_row["mode"])
            concept_id = int(gen_row["concept_id"])

            if _row_is_invalid(judge_row):
                per_method_n_invalid[method_name] = (
                    per_method_n_invalid.get(method_name, 0) + 1
                )
                n_invalid += 1
                continue

            row_score = _compute_row_score(
                concept_score=int(judge_row["concept_score"]),
                fluency_score=int(judge_row["fluency_score"]),
                mode=mode,
            )
            by_concept = per_method_best_scores.setdefault(method_name, {})
            by_concept.setdefault(concept_id, []).append(row_score)
            per_method_n_valid[method_name] = (
                per_method_n_valid.get(method_name, 0) + 1
            )

    # Per-method best-of-grid reduction: collapse the coefficient list
    # for each (method, concept) down to max(row_score).
    per_method_best_of_grid: dict[str, dict[int, float]] = {}
    with profile_region("stage6.best_of_grid_reduce", rows=len(per_method_best_scores)):
        for method_name, by_concept in per_method_best_scores.items():
            per_method_best_of_grid[method_name] = {
                cid: best_of_grid(scores) for cid, scores in by_concept.items()
            }

    # Apply invalid-rate gating. Methods exceeding the drop threshold
    # are removed entirely; methods between warn and drop get a log
    # line but are kept.
    all_seen_methods = (
        set(per_method_n_valid.keys())
        | set(per_method_n_invalid.keys())
    )
    dropped_methods: set[str] = set()
    for method_name in sorted(all_seen_methods):
        n_valid = per_method_n_valid.get(method_name, 0)
        n_invalid_m = per_method_n_invalid.get(method_name, 0)
        total = n_valid + n_invalid_m
        # ``all_seen_methods`` is the union of keys from the valid /
        # invalid counters, so ``total`` is always >= 1 here.
        rate = n_invalid_m / total
        if rate > cfg.judge.invalid_drop_threshold:
            logger.error(
                f"aggregate: dropping method {method_name!r} "
                f"(invalid_rate={rate:.3f} > drop_threshold="
                f"{cfg.judge.invalid_drop_threshold:.3f})",
            )
            dropped_methods.add(method_name)
        elif rate > cfg.judge.invalid_warning_threshold:
            logger.warning(
                f"aggregate: method {method_name!r} has invalid_rate="
                f"{rate:.3f} > warning_threshold="
                f"{cfg.judge.invalid_warning_threshold:.3f}",
            )

    # Build the per-method report entries.
    mode_lookup = _registry_mode_lookup()
    # Also pin mode from actually-seen rows as a fallback for any method
    # name not in the registry (defensive — shouldn't happen in practice).
    for gen_row in gen_rows:
        mode_lookup.setdefault(
            gen_row["method_name"], SteeringMode(gen_row["mode"]),
        )

    report: dict = {}
    n_concepts_total = len({int(r["concept_id"]) for r in gen_rows})
    for method_name in sorted(per_method_best_of_grid.keys()):
        if method_name in dropped_methods:
            continue
        by_concept = per_method_best_of_grid[method_name]
        # ``per_method_best_of_grid`` only contains methods that had at
        # least one valid row, so ``by_concept`` is always non-empty.
        per_concept_scores = list(by_concept.values())
        mode = mode_lookup.get(method_name, SteeringMode.POSITIVE)
        entry = _method_aggregate_entry(
            method_name=method_name,
            mode=mode,
            per_concept_scores=per_concept_scores,
            n_valid_rows=per_method_n_valid.get(method_name, 0),
            n_invalid_rows=per_method_n_invalid.get(method_name, 0),
        )
        report[method_name] = entry

    # Metadata block — rendered at the TOP of the report dict so readers
    # land on provenance first. Python dict insertion order is stable so
    # re-inserting the method entries after the metadata gives us that
    # ordering for free in ``yaml.safe_dump`` output.
    metadata = {
        "run_tag": run_tag,
        "timestamp": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "feature_dir": cfg.feature_dir,
        "n_concepts_total": n_concepts_total,
        "n_generations": len(gen_rows),
        "n_judge_scores": len(judge_rows),
        "n_joined": n_joined,
        "n_invalid": n_invalid,
        "n_methods_included": len(report),
        "n_methods_dropped": len(dropped_methods),
    }
    ordered_report: dict = {"metadata": metadata}
    ordered_report.update(report)

    report_yaml_path = output_paths.report_yaml()
    with profile_region("stage6.write_report_yaml", rows=len(ordered_report)):
        _write_report_yaml(ordered_report, report_yaml_path)

    leaderboard_path = output_paths.leaderboard_csv()
    with profile_region("stage6.write_leaderboard_csv", rows=len(report)):
        _write_leaderboard_csv(
            {k: v for k, v in ordered_report.items() if k != "metadata"},
            mode_lookup,
            leaderboard_path,
        )

    diag = {
        "n_generations": len(gen_rows),
        "n_judge_scores": len(judge_rows),
        "n_joined": n_joined,
        "n_invalid": n_invalid,
        "n_methods_included": len(report),
        "n_methods_dropped": len(dropped_methods),
    }
    with profile_region("stage6.write_aggregation_report"):
        _write_aggregation_report(diag, output_paths.aggregation_report())
    record_counter("stage6.n_generations", len(gen_rows))
    record_counter("stage6.n_judge_scores", len(judge_rows))
    record_counter("stage6.n_joined", n_joined)
    record_counter("stage6.n_invalid", n_invalid)

    logger.info(
        f"aggregate wrote {len(report)} method entries to {report_yaml_path} "
        f"({len(dropped_methods)} dropped, {n_invalid} invalid rows)",
    )
    return report_yaml_path
