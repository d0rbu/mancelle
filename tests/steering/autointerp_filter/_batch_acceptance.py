"""Acceptance-only helpers kept out of the production benchmark entry point."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from steering.eval.autointerp_filter.types import AutoInterpDataError
from steering.stats_io import find_statistics_path, load_statistics

_VALIDATION_PLAN_SCHEMA = "autointerp_batch_validation_plan_v2"
_VALIDATION_SELECTION_METHOD = (
    "autointerp_batch_validation_targets_v5_stats_grid10_midpoint_split_isolated"
)
_VALIDATION_SPLIT_MATERIALIZATION_POLICY = "opposite_anchor_dummy_v1"


def feature_activation_counts(feature_dir: Path) -> list[tuple[int, int]]:
    """Load exact per-segment feature-fire counts for validation stratification.

    ``positive_refs`` is top-k capped by the source-index configuration and is
    therefore not a population-density measurement. The statistics artifact
    is the canonical uncapped producer for the activation-frequency mask used
    by this run, so its integer counts are the matching auditable input.
    """
    statistics_path = find_statistics_path(feature_dir)
    if statistics_path is None:
        raise AutoInterpDataError(
            f"validation target selection requires {feature_dir / 'feature_statistics.safetensors'}"
        )
    statistics = load_statistics(statistics_path)
    counts: list[tuple[int, int]] = []
    for raw_feature_id, feature_stat in statistics.per_feature_stats.items():
        try:
            feature_id = int(raw_feature_id)
        except (TypeError, ValueError) as exc:
            raise AutoInterpDataError(
                f"feature statistics contain non-integer feature ID {raw_feature_id!r}"
            ) from exc
        counts.append((feature_id, feature_stat.count))
    return sorted(counts)


@dataclass(frozen=True)
class ValidationTargetPlan:
    """Preselected 60/40 anchors plus all deterministic block companions."""

    calibration_target_ids: tuple[int, ...]
    heldout_target_ids: tuple[int, ...]
    materialization_target_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    batch_size: int
    resolved_target_count: int
    completed_reference_target_count: int
    selection_candidate_count: int
    selection_inputs_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise AutoInterpDataError("validation plan batch_size must be a strict int")
        if self.batch_size < 2:
            raise AutoInterpDataError("validation plan batch_size must be >= 2")
        for field_name in (
            "resolved_target_count",
            "completed_reference_target_count",
            "selection_candidate_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AutoInterpDataError(
                    f"validation plan {field_name} must be a strict positive int"
                )
        for field_name in (
            "calibration_target_ids",
            "heldout_target_ids",
            "materialization_target_ids",
            "block_ids",
        ):
            values = getattr(self, field_name)
            invalid_values = not isinstance(values, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            if invalid_values:
                raise AutoInterpDataError(
                    f"validation plan {field_name} must be a sorted unique tuple "
                    "of strict non-negative ints"
                )
            integer_values = cast(tuple[int, ...], values)
            if tuple(sorted(set(integer_values))) != integer_values:
                raise AutoInterpDataError(
                    f"validation plan {field_name} must be a sorted unique tuple "
                    "of strict non-negative ints"
                )
        if not self.calibration_target_ids or not self.heldout_target_ids:
            raise AutoInterpDataError(
                "validation plan calibration and heldout sets must be non-empty"
            )
        if set(self.calibration_target_ids) & set(self.heldout_target_ids):
            raise AutoInterpDataError("validation plan calibration and heldout sets overlap")
        if not set(self.anchor_target_ids) <= set(self.materialization_target_ids):
            raise AutoInterpDataError("validation plan materialization set lost an anchor")
        if self.resolved_target_count < len(self.materialization_target_ids):
            raise AutoInterpDataError(
                "validation plan resolved_target_count is smaller than its materialization set"
            )
        if self.completed_reference_target_count < len(self.anchor_target_ids):
            raise AutoInterpDataError(
                "validation plan completed_reference_target_count is smaller than its anchors"
            )
        if not (
            len(self.anchor_target_ids)
            <= self.selection_candidate_count
            <= self.completed_reference_target_count
            <= self.resolved_target_count
        ):
            raise AutoInterpDataError(
                "validation plan population counts violate "
                "anchors <= candidates <= completed <= resolved"
            )
        expected_blocks = tuple(
            sorted({feature_id // self.batch_size for feature_id in self.anchor_target_ids})
        )
        if self.block_ids != expected_blocks:
            raise AutoInterpDataError(
                f"validation plan block IDs {self.block_ids} != anchors {expected_blocks}"
            )
        if any(
            feature_id // self.batch_size not in set(self.block_ids)
            for feature_id in self.materialization_target_ids
        ):
            raise AutoInterpDataError(
                "validation plan materialization target lies outside its anchor blocks"
            )
        calibration_materialization = set(self.calibration_materialization_target_ids)
        heldout_materialization = set(self.heldout_materialization_target_ids)
        if not set(self.calibration_target_ids) <= calibration_materialization:
            raise AutoInterpDataError("validation plan calibration materialization lost an anchor")
        if not set(self.heldout_target_ids) <= heldout_materialization:
            raise AutoInterpDataError("validation plan heldout materialization lost an anchor")
        if calibration_materialization & set(self.heldout_target_ids):
            raise AutoInterpDataError(
                "validation plan calibration materialization leaked heldout anchors"
            )
        if heldout_materialization & set(self.calibration_target_ids):
            raise AutoInterpDataError(
                "validation plan heldout materialization leaked calibration anchors"
            )
        if calibration_materialization | heldout_materialization != set(
            self.materialization_target_ids
        ):
            raise AutoInterpDataError(
                "validation plan split materializations do not cover the combined blocks"
            )
        for field_name in ("selection_inputs_sha256", "selection_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise AutoInterpDataError(
                    f"validation plan {field_name} must be 64 lowercase hex chars"
                )
        expected_selection_sha = _validation_selection_sha256(
            calibration=self.calibration_target_ids,
            heldout=self.heldout_target_ids,
            materialization=self.materialization_target_ids,
            block_ids=self.block_ids,
            batch_size=self.batch_size,
            resolved_target_count=self.resolved_target_count,
            completed_reference_target_count=self.completed_reference_target_count,
            selection_candidate_count=self.selection_candidate_count,
            selection_inputs_sha256=self.selection_inputs_sha256,
        )
        if self.selection_sha256 != expected_selection_sha:
            raise AutoInterpDataError(
                "validation plan selection_sha256 does not match its exact contents"
            )

    @property
    def anchor_target_ids(self) -> tuple[int, ...]:
        return tuple(sorted((*self.calibration_target_ids, *self.heldout_target_ids)))

    @property
    def calibration_block_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted({feature_id // self.batch_size for feature_id in self.calibration_target_ids})
        )

    @property
    def heldout_block_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted({feature_id // self.batch_size for feature_id in self.heldout_target_ids})
        )

    @property
    def calibration_materialization_target_ids(self) -> tuple[int, ...]:
        """Calibration blocks with every held-out anchor left as a dummy row."""
        heldout = set(self.heldout_target_ids)
        blocks = set(self.calibration_block_ids)
        return tuple(
            feature_id
            for feature_id in self.materialization_target_ids
            if feature_id // self.batch_size in blocks and feature_id not in heldout
        )

    @property
    def heldout_materialization_target_ids(self) -> tuple[int, ...]:
        """Held-out blocks with every calibration anchor left as a dummy row."""
        calibration = set(self.calibration_target_ids)
        blocks = set(self.heldout_block_ids)
        return tuple(
            feature_id
            for feature_id in self.materialization_target_ids
            if feature_id // self.batch_size in blocks and feature_id not in calibration
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": _VALIDATION_PLAN_SCHEMA,
            "selection_method": _VALIDATION_SELECTION_METHOD,
            "split_materialization_policy": _VALIDATION_SPLIT_MATERIALIZATION_POLICY,
            "calibration_target_ids": list(self.calibration_target_ids),
            "heldout_target_ids": list(self.heldout_target_ids),
            "anchor_target_ids": list(self.anchor_target_ids),
            "materialization_target_ids": list(self.materialization_target_ids),
            "block_ids": list(self.block_ids),
            "batch_size": self.batch_size,
            "resolved_target_count": self.resolved_target_count,
            "completed_reference_target_count": self.completed_reference_target_count,
            "selection_candidate_count": self.selection_candidate_count,
            "selection_inputs_sha256": self.selection_inputs_sha256,
            "selection_sha256": self.selection_sha256,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ValidationTargetPlan:
        expected_keys = {
            "schema",
            "selection_method",
            "split_materialization_policy",
            "calibration_target_ids",
            "heldout_target_ids",
            "anchor_target_ids",
            "materialization_target_ids",
            "block_ids",
            "batch_size",
            "resolved_target_count",
            "completed_reference_target_count",
            "selection_candidate_count",
            "selection_inputs_sha256",
            "selection_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise AutoInterpDataError(
                "validation plan JSON must contain exactly the versioned plan fields"
            )
        if payload["schema"] != _VALIDATION_PLAN_SCHEMA:
            raise AutoInterpDataError(
                f"validation plan JSON has unsupported schema {payload['schema']!r}"
            )
        if payload["selection_method"] != _VALIDATION_SELECTION_METHOD:
            raise AutoInterpDataError(
                "validation plan JSON has unsupported selection_method "
                f"{payload['selection_method']!r}"
            )
        if payload["split_materialization_policy"] != _VALIDATION_SPLIT_MATERIALIZATION_POLICY:
            raise AutoInterpDataError(
                "validation plan JSON has unsupported split_materialization_policy "
                f"{payload['split_materialization_policy']!r}"
            )

        def ids(field_name: str) -> tuple[int, ...]:
            values = payload[field_name]
            if not isinstance(values, list):
                raise AutoInterpDataError(f"validation plan JSON {field_name} must be a list")
            return tuple(values)

        plan = cls(
            calibration_target_ids=ids("calibration_target_ids"),
            heldout_target_ids=ids("heldout_target_ids"),
            materialization_target_ids=ids("materialization_target_ids"),
            block_ids=ids("block_ids"),
            batch_size=payload["batch_size"],
            resolved_target_count=payload["resolved_target_count"],
            completed_reference_target_count=payload["completed_reference_target_count"],
            selection_candidate_count=payload["selection_candidate_count"],
            selection_inputs_sha256=payload["selection_inputs_sha256"],
            selection_sha256=payload["selection_sha256"],
        )
        if ids("anchor_target_ids") != plan.anchor_target_ids:
            raise AutoInterpDataError(
                "validation plan JSON anchor_target_ids does not match calibration+heldout"
            )
        return plan


def _validation_selection_sha256(
    *,
    calibration: tuple[int, ...],
    heldout: tuple[int, ...],
    materialization: tuple[int, ...],
    block_ids: tuple[int, ...],
    batch_size: int,
    resolved_target_count: int,
    completed_reference_target_count: int,
    selection_candidate_count: int,
    selection_inputs_sha256: str,
) -> str:
    payload = {
        "schema": _VALIDATION_PLAN_SCHEMA,
        "method": _VALIDATION_SELECTION_METHOD,
        "split_materialization_policy": _VALIDATION_SPLIT_MATERIALIZATION_POLICY,
        "calibration": calibration,
        "heldout": heldout,
        "materialization": materialization,
        "block_ids": block_ids,
        "batch_size": batch_size,
        "resolved_target_count": resolved_target_count,
        "completed_reference_target_count": completed_reference_target_count,
        "selection_candidate_count": selection_candidate_count,
        "selection_inputs_sha256": selection_inputs_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _midpoint_spread_order(values: list[int]) -> list[int]:
    """Breadth-first interval midpoints over an already sorted sequence."""
    ordered: list[int] = []
    intervals = [(0, len(values))]
    while intervals:
        next_intervals: list[tuple[int, int]] = []
        for start, end in intervals:
            if start >= end:
                continue
            midpoint = (start + end) // 2
            ordered.append(values[midpoint])
            next_intervals.extend(((start, midpoint), (midpoint + 1, end)))
        intervals = next_intervals
    return ordered


def _select_validation_anchors(
    nonzero: list[tuple[int, int]],
    *,
    total: int,
    mandatory: tuple[int, ...],
) -> tuple[int, ...]:
    """Cover a 10x10 ID/density grid, then spread repeats within cells."""
    if total < len(mandatory) or total > len(nonzero):
        raise AutoInterpDataError(
            f"validation anchor count {total} outside [{len(mandatory)}, {len(nonzero)}]"
        )
    present = {feature_id for feature_id, _count in nonzero}
    missing = sorted(set(mandatory) - present)
    if missing:
        raise AutoInterpDataError(
            f"mandatory validation anchors are absent from positive candidates: {missing}"
        )
    by_id = [feature_id for feature_id, _count in nonzero]
    by_density = [
        feature_id for feature_id, _count in sorted(nonzero, key=lambda item: (item[1], item[0]))
    ]
    grid_bins = min(10, len(nonzero))

    def bins(ordered: list[int]) -> dict[int, int]:
        return {
            feature_id: min(grid_bins - 1, grid_bins * index // len(ordered))
            for index, feature_id in enumerate(ordered)
        }

    id_bin = bins(by_id)
    density_bin = bins(by_density)
    cells: dict[tuple[int, int], list[int]] = {}
    for feature_id in by_id:
        cells.setdefault((id_bin[feature_id], density_bin[feature_id]), []).append(feature_id)

    chosen = set(mandatory)
    selected = list(mandatory)
    represented_cells = {(id_bin[feature_id], density_bin[feature_id]) for feature_id in mandatory}
    queues = {
        cell: _midpoint_spread_order(
            [feature_id for feature_id in bucket if feature_id not in chosen]
        )
        for cell, bucket in cells.items()
    }
    cell_order = sorted(
        cells,
        key=lambda cell: (
            hashlib.sha256(f"autointerp-batch-grid-v4:{cell[0]}:{cell[1]}".encode()).hexdigest(),
            cell,
        ),
    )
    offsets = dict.fromkeys(cell_order, 0)
    for cell in cell_order:
        if cell in represented_cells or len(selected) == total:
            continue
        queue = queues[cell]
        # An unrepresented populated cell cannot have an empty queue: only
        # mandatory IDs are removed, and any mandatory ID would represent it.
        assert queue
        selected.append(queue[0])
        chosen.add(queue[0])
        offsets[cell] = 1
    while len(selected) < total:
        progressed = False
        for cell in cell_order:
            queue = queues.get(cell, [])
            offset = offsets[cell]
            if offset >= len(queue):
                continue
            feature_id = queue[offset]
            offsets[cell] += 1
            selected.append(feature_id)
            chosen.add(feature_id)
            progressed = True
            if len(selected) == total:
                break
        if not progressed:
            raise AutoInterpDataError(
                "validation grid exhausted positive candidates before reaching target count"
            )
    return tuple(sorted(selected))


def select_validation_target_plan(
    activation_counts: Sequence[tuple[int, int]],
    *,
    allowed: set[int],
    anchor_allowed: set[int] | None = None,
    batch_size: int = 128,
    n_calibration: int = 60,
    n_heldout: int = 40,
    include: Sequence[int] = (),
) -> ValidationTargetPlan:
    """Select reference-backed anchors, then fill touched production blocks."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise AutoInterpDataError(
            f"validation batch_size must be a strict int >= 2, got {batch_size!r}"
        )
    for name, value in (("n_calibration", n_calibration), ("n_heldout", n_heldout)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AutoInterpDataError(f"{name} must be a strict positive int, got {value!r}")
    if not isinstance(allowed, set) or not allowed:
        raise AutoInterpDataError("validation target plan requires a non-empty allowed set")
    if any(isinstance(fid, bool) or not isinstance(fid, int) or fid < 0 for fid in allowed):
        raise AutoInterpDataError("validation allowed target IDs must be non-negative strict ints")
    resolved_anchor_allowed = allowed if anchor_allowed is None else anchor_allowed
    if not isinstance(resolved_anchor_allowed, set) or not resolved_anchor_allowed:
        raise AutoInterpDataError("validation target plan requires a non-empty anchor_allowed set")
    if any(
        isinstance(fid, bool) or not isinstance(fid, int) or fid < 0
        for fid in resolved_anchor_allowed
    ):
        raise AutoInterpDataError("validation anchor_allowed IDs must be non-negative strict ints")
    if not resolved_anchor_allowed <= allowed:
        raise AutoInterpDataError("validation anchor_allowed IDs must be a subset of allowed IDs")
    if (
        isinstance(include, (str, bytes))
        or any(
            isinstance(feature_id, bool) or not isinstance(feature_id, int) or feature_id < 0
            for feature_id in include
        )
        or len(set(include)) != len(include)
    ):
        raise AutoInterpDataError(
            "validation included target IDs must be unique strict non-negative ints"
        )

    normalized_counts: list[tuple[int, int]] = []
    seen_count_ids: set[int] = set()
    for item in activation_counts:
        if not isinstance(item, tuple) or len(item) != 2:
            raise AutoInterpDataError(
                "validation activation counts must be (feature_id, count) tuples"
            )
        feature_id, count = item
        if (
            isinstance(feature_id, bool)
            or not isinstance(feature_id, int)
            or feature_id < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise AutoInterpDataError(
                "validation activation counts require strict non-negative integer IDs "
                "and strict non-negative integer counts"
            )
        if feature_id in seen_count_ids:
            raise AutoInterpDataError(
                f"validation activation counts duplicate feature ID {feature_id}"
            )
        seen_count_ids.add(feature_id)
        normalized_counts.append((feature_id, count))
    normalized_counts.sort()
    missing_allowed_counts = sorted(allowed - seen_count_ids)
    if missing_allowed_counts:
        preview = missing_allowed_counts[:10]
        suffix = "..." if len(missing_allowed_counts) > len(preview) else ""
        raise AutoInterpDataError(
            f"validation activation statistics are missing resolved target IDs: {preview}{suffix}"
        )
    nonzero = [
        (feature_id, count)
        for feature_id, count in normalized_counts
        if count > 0 and feature_id in resolved_anchor_allowed
    ]
    total = n_calibration + n_heldout
    if not nonzero:
        raise AutoInterpDataError("validation target plan has no positive allowed candidates")
    distinct_density_count = len({count for _feature_id, count in nonzero})
    required_density_count = min(10, total)
    if distinct_density_count < required_density_count:
        raise AutoInterpDataError(
            "validation activation counts do not provide enough density diversity: "
            f"{distinct_density_count} distinct values, need {required_density_count}; "
            "do not use top-k-capped source-index positive-ref counts"
        )
    by_density = sorted(nonzero, key=lambda item: (item[1], item[0]))
    by_id = [fid for fid, _count in nonzero]
    boundary_candidates = {
        by_id[0],
        by_id[-1],
        by_density[0][0],
        by_density[-1][0],
        min(by_id, key=lambda fid: (fid % batch_size, fid)),
        min(by_id, key=lambda fid: (-(fid % batch_size), fid)),
    }
    mandatory = tuple(sorted({int(fid) for fid in include} | boundary_candidates))
    anchors = _select_validation_anchors(
        nonzero,
        total=total,
        mandatory=mandatory,
    )

    # This assignment is frozen before any serial/batch values are observed.
    split_order = sorted(
        anchors,
        key=lambda fid: (
            hashlib.sha256(f"autointerp-batch-calibration-v1:{fid}".encode()).hexdigest(),
            fid,
        ),
    )
    calibration = tuple(sorted(split_order[:n_calibration]))
    heldout = tuple(sorted(split_order[n_calibration:]))
    grid_bins = min(10, len(nonzero))
    id_ranked = [feature_id for feature_id, _count in nonzero]
    density_ranked = [
        feature_id for feature_id, _count in sorted(nonzero, key=lambda item: (item[1], item[0]))
    ]

    def population_bins(ordered: list[int]) -> dict[int, int]:
        return {
            feature_id: min(grid_bins - 1, grid_bins * index // len(ordered))
            for index, feature_id in enumerate(ordered)
        }

    id_bin = population_bins(id_ranked)
    density_bin = population_bins(density_ranked)
    expected_bins = set(range(grid_bins))
    for split_name, split in (("calibration", calibration), ("heldout", heldout)):
        if len(split) < grid_bins:
            continue
        missing_id_bins = sorted(expected_bins - {id_bin[feature_id] for feature_id in split})
        missing_density_bins = sorted(
            expected_bins - {density_bin[feature_id] for feature_id in split}
        )
        if missing_id_bins or missing_density_bins:
            raise AutoInterpDataError(
                f"validation {split_name} hash split lost population coverage: "
                f"missing_id_bins={missing_id_bins} "
                f"missing_density_bins={missing_density_bins}"
            )
    block_ids = tuple(sorted({fid // batch_size for fid in anchors}))
    materialization = tuple(sorted(fid for fid in allowed if fid // batch_size in set(block_ids)))

    source_payload = {
        "activation_counts": normalized_counts,
        "allowed": sorted(allowed),
        "anchor_allowed": sorted(resolved_anchor_allowed),
    }
    selection_inputs_sha = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    selection_sha = _validation_selection_sha256(
        calibration=calibration,
        heldout=heldout,
        materialization=materialization,
        block_ids=block_ids,
        batch_size=batch_size,
        resolved_target_count=len(allowed),
        completed_reference_target_count=len(resolved_anchor_allowed),
        selection_candidate_count=len(nonzero),
        selection_inputs_sha256=selection_inputs_sha,
    )
    return ValidationTargetPlan(
        calibration_target_ids=calibration,
        heldout_target_ids=heldout,
        materialization_target_ids=materialization,
        block_ids=block_ids,
        batch_size=batch_size,
        resolved_target_count=len(allowed),
        completed_reference_target_count=len(resolved_anchor_allowed),
        selection_candidate_count=len(nonzero),
        selection_inputs_sha256=selection_inputs_sha,
        selection_sha256=selection_sha,
    )


def evaluate_batch_performance_replicates(
    *,
    serial_seconds: Sequence[float],
    batched_seconds: Sequence[float],
    prefetched_seconds: Sequence[float],
    bootstrap_samples: int = 10_000,
    seed: int = 696,
) -> dict[str, Any]:
    """Apply the approved 1.2x batch and 3% prefetch acceptance thresholds."""
    series = {
        "serial_seconds": tuple(float(value) for value in serial_seconds),
        "batched_seconds": tuple(float(value) for value in batched_seconds),
        "prefetched_seconds": tuple(float(value) for value in prefetched_seconds),
    }
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 3:
        raise AutoInterpDataError(
            "performance acceptance requires three equally-sized series with at least 3 repeats"
        )
    if any(
        not value > 0.0 or not math.isfinite(value)
        for values in series.values()
        for value in values
    ):
        raise AutoInterpDataError("performance repeat durations must be finite and > 0")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 100
    ):
        raise AutoInterpDataError("bootstrap_samples must be an int >= 100")

    serial_median = statistics.median(series["serial_seconds"])
    batched_median = statistics.median(series["batched_seconds"])
    prefetched_median = statistics.median(series["prefetched_seconds"])
    batch_speedup = serial_median / batched_median
    prefetch_speedup = batched_median / prefetched_median

    paired_ratios = tuple(
        off / on
        for off, on in zip(
            series["batched_seconds"],
            series["prefetched_seconds"],
            strict=True,
        )
    )
    rng = random.Random(seed)
    bootstrap_medians = sorted(
        statistics.median(rng.choice(paired_ratios) for _ in paired_ratios)
        for _ in range(bootstrap_samples)
    )
    lower_index = int(0.025 * (bootstrap_samples - 1))
    upper_index = int(0.975 * (bootstrap_samples - 1))
    ci = (bootstrap_medians[lower_index], bootstrap_medians[upper_index])
    batch_pass = batch_speedup >= 1.2
    prefetch_pass = prefetch_speedup >= 1.03 and ci[0] > 1.0
    return {
        **series,
        "serial_median_seconds": serial_median,
        "batched_median_seconds": batched_median,
        "prefetched_median_seconds": prefetched_median,
        "batch_speedup": batch_speedup,
        "prefetch_speedup": prefetch_speedup,
        "prefetch_paired_speedup_95ci": list(ci),
        "batch_threshold": 1.2,
        "prefetch_threshold": 1.03,
        "batch_pass": batch_pass,
        "prefetch_pass": prefetch_pass,
        "retain_prefetch": prefetch_pass,
    }
