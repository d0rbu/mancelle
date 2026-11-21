"""Benchmark AxBench-BPB-Adv class-margin and roundtrip scoring.

This harness deliberately runs the production CLI in a fresh subprocess for
each metric. It reuses one already-materialized candidate table, sends score
artifacts to a separate benchmark directory, and records the profiler output
alongside immutable candidate-table fingerprints.

Example::

        uv run python -m scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip \
        --output-root out/eval/axbench_bpb_adv \
        --run-tag l22_f65k_clustered_std50_m_gcg2 \
        --benchmark-dir out/benchmarks/issue-1257 \
        source_run_dir=... feature_dir=... feature_statistics_path=... \
        decoder_only_target=... encdec_target=...

The trailing arguments are Hydra overrides. The harness owns ``stage``,
``steering_metric``, ``run_tag``, ``output_root``, ``scores_path``, and the
profile output location so that both metric runs consume the same candidate
artifact without overwriting production score files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from steering.eval.axbench_bpb_adv.artifact_identity import (
    ARTIFACT_FAMILY_METADATA_KEY,
    FEATURE_CONTRACT_METADATA_KEY,
    PRECOMPUTED_BAG_SET_METADATA_KEY,
    SOURCE_CONTRACT_METADATA_KEY,
    restamp_candidates_bag_set,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
)

_RESERVED_OVERRIDE_KEYS = frozenset(
    {
        "stage",
        "steering_metric",
        "run_tag",
        "output_root",
        "scores_path",
        "profile.enabled",
        "profile.output_dir",
    }
)
_CANDIDATES_FAMILY_METADATA_KEYS = frozenset(
    {
        FEATURE_CONTRACT_METADATA_KEY,
        SOURCE_CONTRACT_METADATA_KEY,
        ARTIFACT_FAMILY_METADATA_KEY,
        PRECOMPUTED_BAG_SET_METADATA_KEY,
    }
)
_DERIVED_RUN_EXCLUDED_PATH_ACCESSORS = frozenset(
    {
        "aggregates_dir",
        "candidates_dir",
        "candidates_path",
        "general_val_bpb_path",
        "plots_dir",
        "profile_dir",
        "roundtrip_aggregates_dir",
        "roundtrip_checkpoint_dir",
        "roundtrip_checkpoint_manifest",
        "roundtrip_plots_dir",
        "roundtrip_scores_path",
        "samples_dir",
        "scores_path",
        "steering_strength_plots_dir",
        "universal_gcg_checkpoint_dir",
        "universal_gcg_checkpoint_groups_dir",
        "universal_gcg_checkpoint_manifest",
        "universal_triggers_path",
    }
)
_DERIVED_RUN_REQUIRED_INPUT_ACCESSORS = frozenset(
    {
        "attack_rows_dir",
        "attack_rows_path",
        "clustered_context_cache_partial_marker",
        "clustered_context_cache_raw_dir",
        "context_cache_dir",
        "context_cache_manifest",
        "evaluation_population",
        "general_val_bags_dir",
        "ours_gcg_bags_dir",
        "source_contract",
    }
)


def _classified_relative_paths(accessors: frozenset[str]) -> frozenset[Path]:
    paths = AxBenchBpbAdvPaths(
        output_root=Path("__benchmark_path_contract__"),
        run_tag="run",
    )
    return frozenset(
        getattr(paths, accessor)().relative_to(paths.root) for accessor in accessors
    )


_DERIVED_RUN_EXCLUDED_PATHS = _classified_relative_paths(
    _DERIVED_RUN_EXCLUDED_PATH_ACCESSORS
)
_DERIVED_RUN_REQUIRED_INPUT_PATHS = _classified_relative_paths(
    _DERIVED_RUN_REQUIRED_INPUT_ACCESSORS
) | frozenset({Path("feature_contract.json")})


@dataclass(frozen=True)
class CandidateFingerprint:
    """Stable identity of the candidate table consumed by both subprocesses."""

    path: str
    sha256: str
    schema_sha256: str
    candidate_ids_sha256: str
    rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class CandidateSample:
    """Deterministic bounded candidate population and its allocation."""

    table: pa.Table
    sample_size: int
    sample_seed: int
    n_strata: int
    strata_counts: dict[str, int]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def candidate_fingerprint(path: Path) -> CandidateFingerprint:
    """Hash candidate bytes, schema, and ordered IDs before any model loads."""
    if not path.is_file():
        raise FileNotFoundError(f"candidate table does not exist: {path}")
    payload = path.read_bytes()
    table = pq.read_table(path)
    if "candidate_id" not in table.column_names:
        raise ValueError(f"candidate table lacks candidate_id: {path}")
    ids = [str(value) for value in table.column("candidate_id").to_pylist()]
    ids_payload = json.dumps(
        ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return CandidateFingerprint(
        path=str(path.resolve()),
        sha256=_sha256_bytes(payload),
        schema_sha256=_sha256_bytes(str(table.schema).encode("utf-8")),
        candidate_ids_sha256=_sha256_bytes(ids_payload),
        rows=table.num_rows,
    )


def _stratum_slug(stratum: tuple[str, str, str]) -> str:
    return json.dumps(list(stratum), ensure_ascii=False, separators=(",", ":"))


def _candidate_strata(
    table: pa.Table,
) -> dict[tuple[str, str, str], list[int]]:
    required = {"candidate_id", "method_name", "target_model", "mode"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"candidate table lacks sampling columns: {missing}")
    columns = {
        name: table.column(name).to_pylist()
        for name in ("candidate_id", "method_name", "target_model", "mode")
    }
    strata: dict[tuple[str, str, str], list[int]] = {}
    for index in range(table.num_rows):
        stratum = (
            str(columns["method_name"][index]),
            str(columns["target_model"][index]),
            str(columns["mode"][index]),
        )
        strata.setdefault(stratum, []).append(index)
    return strata


def _largest_remainder_allocation(
    strata: dict[tuple[str, str, str], list[int]],
    sample_size: int,
    table_rows: int,
) -> dict[tuple[str, str, str], int]:
    n_strata = len(strata)
    if table_rows == 0:
        raise ValueError(
            f"sample_size={sample_size} cannot sample table.num_rows={table_rows}"
        )
    if sample_size < n_strata:
        raise ValueError(
            f"sample_size={sample_size} is smaller than n_strata={n_strata}"
        )
    if sample_size > table_rows:
        raise ValueError(
            f"sample_size={sample_size} exceeds table.num_rows={table_rows}"
        )
    ordered = sorted(strata)
    quotas = {
        stratum: sample_size * len(strata[stratum]) / table_rows
        for stratum in ordered
    }
    allocation = {
        stratum: max(1, math.floor(quotas[stratum])) for stratum in ordered
    }
    while sum(allocation.values()) > sample_size:
        removable = [stratum for stratum in ordered if allocation[stratum] > 1]
        if not removable:
            # Defensive: unreachable while sample_size >= n_strata; removable is empty only when every allocation is 1, bounding the sum by n_strata <= sample_size.
            raise AssertionError(  # pragma: no cover
                "largest-remainder allocation cannot satisfy minimum-one bound: "
                f"sample_size={sample_size}, n_strata={n_strata}"
            )
        stratum = min(
            removable,
            key=lambda key: (-(allocation[key] - quotas[key]), key),
        )
        allocation[stratum] -= 1
    while sum(allocation.values()) < sample_size:
        addable = [
            stratum
            for stratum in ordered
            if allocation[stratum] < len(strata[stratum])
        ]
        if not addable:
            raise AssertionError(
                "largest-remainder allocation exhausted stratum capacity: "
                f"sample_size={sample_size}, table.num_rows={table_rows}"
            )
        stratum = min(
            addable,
            key=lambda key: (allocation[key] - quotas[key], key),
        )
        allocation[stratum] += 1
    return allocation


def sample_candidates(
    table: pa.Table,
    *,
    sample_size: int,
    sample_seed: int,
) -> CandidateSample:
    """Select a stable, proportional, stratified candidate subset."""
    strata = _candidate_strata(table)
    allocation = _largest_remainder_allocation(
        strata,
        sample_size,
        table.num_rows,
    )
    candidate_ids = table.column("candidate_id").to_pylist()
    selected: list[int] = []
    for stratum in sorted(strata):
        base_order = sorted(
            strata[stratum],
            key=lambda index: (str(candidate_ids[index]), index),
        )
        seed_payload = json.dumps([sample_seed, *stratum]).encode()
        seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:8],
            "little",
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        positions = generator.choice(
            len(base_order),
            size=allocation[stratum],
            replace=False,
        )
        selected.extend(base_order[int(position)] for position in positions)
    selected.sort()
    subset = table.take(pa.array(selected, type=pa.int64()))
    if subset.schema.serialize().to_pybytes() != table.schema.serialize().to_pybytes():
        raise RuntimeError(
            "candidate subset schema drifted before write, including schema metadata"
        )
    return CandidateSample(
        table=subset,
        sample_size=sample_size,
        sample_seed=sample_seed,
        n_strata=len(strata),
        strata_counts={
            _stratum_slug(stratum): allocation[stratum] for stratum in sorted(strata)
        },
    )


def _write_subset_table_atomic(
    table: pa.Table,
    destination: Path,
    *,
    source_schema: pa.Schema,
) -> None:
    _assert_subset_schema_fidelity(
        table.schema,
        source_schema=source_schema,
        surface="before write",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path)
        written_schema = pq.read_schema(temporary_path)
        _assert_subset_schema_fidelity(
            written_schema,
            source_schema=source_schema,
            surface="during parquet write",
        )
        if written_schema.metadata != table.schema.metadata:
            raise RuntimeError(
                "candidate subset schema drifted during parquet write, including "
                "schema metadata"
            )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _assert_subset_schema_fidelity(
    observed: pa.Schema,
    *,
    source_schema: pa.Schema,
    surface: str,
) -> None:
    if not observed.remove_metadata().equals(source_schema.remove_metadata()):
        raise RuntimeError(f"candidate subset schema drifted {surface}, including columns")
    source_metadata = dict(source_schema.metadata or {})
    observed_metadata = dict(observed.metadata or {})
    clustered_keys_present = _CANDIDATES_FAMILY_METADATA_KEYS.intersection(
        source_metadata
    )
    if clustered_keys_present:
        missing = _CANDIDATES_FAMILY_METADATA_KEYS - observed_metadata.keys()
        bag_set_stamp = observed_metadata.get(PRECOMPUTED_BAG_SET_METADATA_KEY, b"")
        source_without_bag_set = {
            key: value
            for key, value in source_metadata.items()
            if key != PRECOMPUTED_BAG_SET_METADATA_KEY
        }
        observed_without_bag_set = {
            key: value
            for key, value in observed_metadata.items()
            if key != PRECOMPUTED_BAG_SET_METADATA_KEY
        }
        if missing or not bag_set_stamp or observed_without_bag_set != source_without_bag_set:
            raise RuntimeError(
                f"candidate subset schema drifted {surface}, including schema metadata"
            )
        return
    if observed_metadata != source_metadata:
        raise RuntimeError(
            f"candidate subset schema drifted {surface}, including schema metadata"
        )


def _is_excluded_from_derived_run(relative_path: Path) -> bool:
    return any(
        relative_path == excluded or excluded in relative_path.parents
        for excluded in _DERIVED_RUN_EXCLUDED_PATHS
    )


def _hardlink_copy_run_dir(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"source run directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"derived run directory already exists: {destination}")
    destination.mkdir(parents=True)
    for source_path in sorted(source.rglob("*")):
        relative_path = source_path.relative_to(source)
        if _is_excluded_from_derived_run(relative_path):
            continue
        if source_path.is_symlink():
            raise ValueError(
                f"source run directory contains forbidden symlink: {source_path}"
            )
        destination_path = destination / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, destination_path)
        else:
            raise ValueError(
                f"source run directory contains unsupported entry: {source_path}"
            )


def _reject_reserved_overrides(overrides: Sequence[str]) -> None:
    conflicts = [
        override
        for override in overrides
        if override.lstrip("+").split("=", 1)[0] in _RESERVED_OVERRIDE_KEYS
    ]
    if conflicts:
        raise ValueError(
            "benchmark owns these Hydra overrides; remove them from the command: "
            + ", ".join(conflicts)
        )


def _reject_benchmark_dir_inside_source(
    *,
    benchmark_dir: Path,
    source_run_dir: Path,
) -> None:
    if benchmark_dir == source_run_dir or source_run_dir in benchmark_dir.parents:
        raise ValueError(
            "benchmark directory must not equal or descend from the source run "
            f"directory: benchmark_dir={benchmark_dir}, source_run_dir={source_run_dir}"
        )


def _counter_totals(summary: dict[str, Any]) -> dict[str, float]:
    counters = summary.get("counters", [])
    if not isinstance(counters, list):
        raise ValueError("profile summary counters must be a list")
    out: dict[str, float] = {}
    for row in counters:
        if not isinstance(row, dict) or "name" not in row or "total" not in row:
            raise ValueError(f"malformed profile counter row: {row!r}")
        out[str(row["name"])] = float(row["total"])
    return out


def validate_regression_sentinel(
    optimized_counters: dict[str, float],
    reference_counters: dict[str, float],
    *,
    required_increases: Sequence[str],
) -> dict[str, Any]:
    """Validate that a deliberately slower reference run is observable.

    The production benchmark does not silently manufacture a reference run.
    Tests and future benchmark invocations can pass counters from an explicitly
    disabled-cache/reference execution here. Every requested counter must be
    strictly larger in that execution, otherwise the harness is not sensitive
    enough to the regression it claims to detect.
    """
    missing = [
        name
        for name in required_increases
        if name not in optimized_counters or name not in reference_counters
    ]
    if missing:
        raise ValueError(f"regression sentinel counters are missing: {missing}")
    regressions = {
        name: {
            "optimized": optimized_counters[name],
            "reference": reference_counters[name],
        }
        for name in required_increases
        if reference_counters[name] <= optimized_counters[name]
    }
    if regressions:
        raise AssertionError(
            "reference run did not increase the required counters: "
            f"{regressions}"
        )
    return {
        "passed": True,
        "required_increases": {
            name: {
                "optimized": optimized_counters[name],
                "reference": reference_counters[name],
            }
            for name in required_increases
        },
    }


def _load_profile_summary(profile_dir: Path) -> dict[str, Any]:
    summary_path = profile_dir / "summary.yaml"
    if not summary_path.is_file():
        raise FileNotFoundError(f"profile summary does not exist: {summary_path}")
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"profile summary is not a mapping: {summary_path}")
    return summary


def _score_table_summary(metric: str, score_path: Path) -> dict[str, Any]:
    table = pq.read_table(score_path)
    expected = ROUNDTRIP_SCORE_SCHEMA if metric == "roundtrip" else SCORE_SCHEMA
    if not table.schema.equals(expected):
        raise ValueError(
            f"{metric} score schema mismatch at {score_path}: "
            f"expected {expected}, observed {table.schema}"
        )
    generated_tokens = 0
    if metric == "roundtrip":
        generated_tokens = int(sum(table.column("n_generated_tokens").to_pylist()))
        generated_tokens += int(
            sum(table.column("control_n_generated_tokens").to_pylist())
        )
    return {
        "path": str(score_path.resolve()),
        "rows": table.num_rows,
        "candidate_ids": len(
            {str(value) for value in table.column("candidate_id").to_pylist()}
        ),
        "generated_tokens": generated_tokens,
    }


def compare_roundtrip_baselines(
    class_margin_path: Path,
    roundtrip_path: Path,
    *,
    abs_tolerance: float = 1e-7,
    rel_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare the retained BPB losses for shared candidate IDs.

    Roundtrip rows can be absent when the current empty-completion contract
    skips a sample, so missing roundtrip IDs are reported rather than treated
    as a baseline mismatch.
    """
    class_table = pq.read_table(class_margin_path)
    roundtrip_table = pq.read_table(roundtrip_path)
    if not class_table.schema.equals(SCORE_SCHEMA):
        raise ValueError("class-margin benchmark output has the wrong schema")
    if not roundtrip_table.schema.equals(ROUNDTRIP_SCORE_SCHEMA):
        raise ValueError("roundtrip benchmark output has the wrong schema")
    class_rows = {
        str(row["candidate_id"]): row for row in class_table.to_pylist()
    }
    observed: set[str] = set()
    max_abs = 0.0
    max_rel = 0.0
    mismatches: list[str] = []
    for row in roundtrip_table.to_pylist():
        candidate_id = str(row["candidate_id"])
        observed.add(candidate_id)
        baseline = class_rows.get(candidate_id)
        if baseline is None:
            mismatches.append(candidate_id)
            continue
        for field in ("loss_concept_rich", "loss_concept_avoiding"):
            left = float(baseline[field])
            right = float(row[field])
            absolute = abs(left - right)
            relative = absolute / max(abs(left), abs(right), 1e-12)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            if not math.isclose(
                left,
                right,
                abs_tol=abs_tolerance,
                rel_tol=rel_tolerance,
            ):
                mismatches.append(f"{candidate_id}:{field}")
    return {
        "class_margin_candidates": len(class_rows),
        "roundtrip_candidates": len(observed),
        "roundtrip_missing_candidates": len(set(class_rows) - observed),
        "max_abs_loss_delta": max_abs,
        "max_rel_loss_delta": max_rel,
        "within_tolerance": not mismatches,
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
    }


def _run_metric(
    *,
    metric: str,
    args: argparse.Namespace,
    overrides: Sequence[str],
    candidate: CandidateFingerprint,
    sample: CandidateSample | None = None,
) -> dict[str, Any]:
    metric_dir = args.benchmark_dir / metric
    metric_dir.mkdir(parents=True, exist_ok=False)
    score_path = metric_dir / (
        "roundtrip_scores.parquet" if metric == "roundtrip" else "scores.parquet"
    )
    profile_dir = metric_dir / "profile"
    run_output_root = args.output_root
    source_paths = AxBenchBpbAdvPaths(
        output_root=args.output_root,
        run_tag=args.run_tag,
    )
    run_dir = source_paths.root
    consumed_candidate = candidate
    if sample is not None:
        run_output_root = args.benchmark_dir / "derived" / metric
        derived_paths = AxBenchBpbAdvPaths(
            output_root=run_output_root,
            run_tag=args.run_tag,
        )
        run_dir = derived_paths.root
        _hardlink_copy_run_dir(source_paths.root, run_dir)
        derived_candidate_path = derived_paths.candidates_path()
        source_schema = pq.read_schema(Path(candidate.path))
        derived_table = sample.table
        if _CANDIDATES_FAMILY_METADATA_KEYS.intersection(
            source_schema.metadata or {}
        ):
            derived_table = restamp_candidates_bag_set(
                derived_table,
                paths=derived_paths,
            )
        _write_subset_table_atomic(
            derived_table,
            derived_candidate_path,
            source_schema=source_schema,
        )
        if os.stat(derived_candidate_path).st_ino == os.stat(candidate.path).st_ino:
            raise RuntimeError(
                "derived candidates table unexpectedly shares its inode with the "
                f"source table: {derived_candidate_path}"
            )
        consumed_candidate = candidate_fingerprint(derived_candidate_path)
        if consumed_candidate.rows != sample.sample_size:
            raise RuntimeError(
                "derived candidates row count mismatch: "
                f"expected sample_size={sample.sample_size}, "
                f"observed rows={consumed_candidate.rows}"
            )
        source_before_run = candidate_fingerprint(Path(candidate.path))
        if source_before_run != candidate:
            raise RuntimeError(
                "source candidate table changed while constructing derived run: "
                f"expected={candidate.as_dict()}, observed={source_before_run.as_dict()}"
            )
    command = [
        args.python,
        "-m",
        "scripts.eval.axbench_bpb_adv_eval",
        *overrides,
        "stage=score",
        f"steering_metric={metric}",
        f"run_tag={args.run_tag}",
        f"output_root={run_output_root}",
        f"scores_path={score_path}",
        "+profile.enabled=true",
        f"+profile.output_dir={profile_dir}",
    ]
    started = time.perf_counter()
    subprocess_error: Exception | None = None
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        subprocess_error = exc
    elapsed_s = time.perf_counter() - started
    integrity_errors: list[Exception] = []
    try:
        consumed_after = candidate_fingerprint(Path(consumed_candidate.path))
        if consumed_after != consumed_candidate:
            raise RuntimeError(
                "candidate table changed during benchmark; refusing to compare metrics"
            )
    except Exception as exc:
        integrity_errors.append(exc)
    try:
        source_after = candidate_fingerprint(Path(candidate.path))
        if source_after != candidate:
            raise RuntimeError(
                "source candidate table changed during benchmark; production state was "
                f"mutated: expected={candidate.as_dict()}, "
                f"observed={source_after.as_dict()}"
            )
    except Exception as exc:
        integrity_errors.append(exc)
    if subprocess_error is not None:
        for integrity_error in integrity_errors:
            subprocess_error.add_note(
                "benchmark integrity check also failed: "
                f"{type(integrity_error).__name__}: {integrity_error}"
            )
        raise subprocess_error
    if integrity_errors:
        primary_integrity_error = integrity_errors[0]
        for additional_error in integrity_errors[1:]:
            primary_integrity_error.add_note(
                "additional benchmark integrity check failed: "
                f"{type(additional_error).__name__}: {additional_error}"
            )
        raise primary_integrity_error
    summary = _load_profile_summary(profile_dir)
    score_summary = _score_table_summary(metric, score_path)
    generated_tokens = int(score_summary["generated_tokens"])
    result = {
        "metric": metric,
        "command": command,
        "elapsed_s": elapsed_s,
        "candidates_per_second": (
            consumed_candidate.rows / elapsed_s if elapsed_s > 0 else 0.0
        ),
        "generated_tokens_per_second": (
            generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
        ),
        "candidate_fingerprint": consumed_candidate.as_dict(),
        "score": score_summary,
        "profile_dir": str(profile_dir.resolve()),
        "profile_elapsed_s": float(summary.get("elapsed_s", 0.0)),
        "profile_counters": _counter_totals(summary),
    }
    if sample is not None:
        result["run_dir"] = str(run_dir.resolve())
    return result


def run_benchmark(args: argparse.Namespace, overrides: Sequence[str]) -> dict[str, Any]:
    """Run selected metrics in independent production subprocesses."""
    _reject_reserved_overrides(overrides)
    args.output_root = Path(args.output_root).resolve()
    args.benchmark_dir = Path(args.benchmark_dir).resolve()
    source_run_dir = AxBenchBpbAdvPaths(
        output_root=args.output_root,
        run_tag=args.run_tag,
    ).root.resolve()
    _reject_benchmark_dir_inside_source(
        benchmark_dir=args.benchmark_dir,
        source_run_dir=source_run_dir,
    )
    args.benchmark_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = AxBenchBpbAdvPaths(
        output_root=args.output_root,
        run_tag=args.run_tag,
    ).candidates_path()
    candidate = candidate_fingerprint(candidate_path)
    sample = None
    if args.sample_size is not None:
        source_table = pq.read_table(candidate_path)
        source_after_read = candidate_fingerprint(candidate_path)
        if source_after_read != candidate:
            raise RuntimeError(
                "source candidate table changed while resolving benchmark sample: "
                f"expected={candidate.as_dict()}, observed={source_after_read.as_dict()}"
            )
        sample = sample_candidates(
            source_table,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
        )
    metric_names = (
        ("bpb_class_margin", "roundtrip")
        if args.metrics == "both"
        else (args.metrics,)
    )
    results = [
        _run_metric(
            metric=metric,
            args=args,
            overrides=overrides,
            candidate=candidate,
            sample=sample,
        )
        for metric in metric_names
    ]
    if args.metrics == "both":
        result_by_metric = {result["metric"]: result for result in results}
        baseline_comparison = compare_roundtrip_baselines(
            Path(result_by_metric["bpb_class_margin"]["score"]["path"]),
            Path(result_by_metric["roundtrip"]["score"]["path"]),
        )
    else:
        baseline_comparison = {"skipped": "single-metric run"}
    report = {
        "issue": 1257,
        "run_tag": args.run_tag,
        "output_root": str(args.output_root),
        "candidate": candidate.as_dict(),
        "metrics": results,
        "baseline_comparison": baseline_comparison,
    }
    if sample is not None:
        subset_hashes = {
            str(result["candidate_fingerprint"]["sha256"]) for result in results
        }
        if len(subset_hashes) != 1:
            raise RuntimeError(
                "derived candidate subset parquet bytes differ across metrics: "
                f"sha256={sorted(subset_hashes)}"
            )
        report["sample"] = {
            "sample_size": sample.sample_size,
            "sample_seed": sample.sample_seed,
            "numpy_version": np.__version__,
            "n_strata": sample.n_strata,
            "pyarrow_version": pa.__version__,
            "strata_counts": sample.strata_counts,
            "subset_sha256": next(iter(subset_hashes)),
            "source_candidate_fingerprint": candidate.as_dict(),
        }
    (args.benchmark_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"sample size must be greater than zero, observed {parsed}"
        )
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--metrics",
        choices=("both", "bpb_class_margin", "roundtrip"),
        default="both",
    )
    parser.add_argument("--sample-size", type=_positive_int)
    parser.add_argument("--sample-seed", type=int, default=0)
    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def main(argv: Sequence[str] | None = None) -> None:
    args, overrides = _parse_args(argv)
    report = run_benchmark(args, overrides)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
