"""Coverage stats for manually reviewed autointerp feature filters."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch as t
import yaml

from errors import ManualFeatureSelectionError
from steering.compute_statistics import _stats_domain_shards
from steering.eval.autointerp_filter.types import AutoInterpConfigError
from steering.feature_io import TensorKey, load_features
from steering.manual_feature_ids import (
    parse_manual_feature_yaml_ids,
    validate_feature_ids_in_range,
)
from steering.stats_io import find_statistics_path, load_statistics

MANUAL_COVERAGE_PERCENTILES = (0, 1, 5, 10, 50, 90, 95, 99, 100)


def load_manual_feature_ids(path: str | Path) -> list[int]:
    try:
        return parse_manual_feature_yaml_ids(path)
    except ManualFeatureSelectionError as exc:
        raise AutoInterpConfigError(str(exc)) from exc


def validate_manual_feature_ids(ids: list[int], *, n_sae_features: int) -> None:
    try:
        validate_feature_ids_in_range(ids, n_sae_features=n_sae_features)
    except ManualFeatureSelectionError as exc:
        raise AutoInterpConfigError(str(exc)) from exc


def compute_manual_filter_coverage(
    *,
    feature_dir: str | Path,
    selection_yaml_path: str | Path,
    exclude_last_shard: bool = True,
) -> dict[str, Any]:
    feature_dir = Path(feature_dir)
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise AutoInterpConfigError(
            f"feature statistics file not found under {feature_dir}; run compute_statistics first"
        )
    stats = load_statistics(stats_path)
    selected_ids = load_manual_feature_ids(selection_yaml_path)
    validate_manual_feature_ids(
        selected_ids,
        n_sae_features=int(stats.metadata.n_sae_features),
    )
    if not selected_ids:
        histogram = {0: int(stats.features.n_segments)}
    else:
        histogram = _stream_selected_count_histogram(
            feature_dir=feature_dir,
            selected_ids=selected_ids,
            n_sae_features=int(stats.metadata.n_sae_features),
            exclude_last_shard=exclude_last_shard,
        )
    n_segments = sum(histogram.values())
    return {
        "feature_dir": str(feature_dir),
        "selection_yaml_path": str(selection_yaml_path),
        "exclude_last_shard": bool(exclude_last_shard),
        "n_selected_features": len(selected_ids),
        "n_segments": n_segments,
        "histogram": dict(sorted(histogram.items())),
        "percentiles": _histogram_percentiles(
            histogram,
            percentiles=MANUAL_COVERAGE_PERCENTILES,
        ),
    }


def dump_manual_coverage_report(report: dict[str, Any]) -> str:
    return yaml.safe_dump(report, sort_keys=False, allow_unicode=False)


def _stream_selected_count_histogram(
    *,
    feature_dir: Path,
    selected_ids: list[int],
    n_sae_features: int,
    exclude_last_shard: bool,
) -> dict[int, int]:
    allowed = t.zeros(n_sae_features, dtype=t.bool)
    allowed[t.tensor(selected_ids, dtype=t.int64)] = True
    histogram: Counter[int] = Counter()
    paths = _stats_domain_shards(feature_dir, exclude_last_shard)
    if not paths:
        raise AutoInterpConfigError(f"No feature shards found in {feature_dir}")
    for path in paths:
        data = load_features(
            path,
            keys={str(TensorKey.OFFSETS), str(TensorKey.FEATURE_IDS)},
        )
        offsets = data[TensorKey.OFFSETS].to(t.int64)
        feature_ids = data[TensorKey.FEATURE_IDS].to(t.int64)
        in_range = (feature_ids >= 0) & (feature_ids < n_sae_features)
        selected = t.zeros_like(feature_ids, dtype=t.int64)
        if bool(in_range.any()):
            valid_positions = in_range.nonzero(as_tuple=False).flatten()
            selected[valid_positions] = allowed[feature_ids[valid_positions]].to(t.int64)
        prefix = t.zeros(int(selected.numel()) + 1, dtype=t.int64)
        if int(selected.numel()) > 0:
            prefix[1:] = selected.cumsum(0)
        counts = prefix[offsets[1:]] - prefix[offsets[:-1]]
        for value, count in zip(*t.unique(counts, return_counts=True), strict=True):
            histogram[int(value.item())] += int(count.item())
    return dict(histogram)


def _histogram_percentiles(
    histogram: dict[int, int],
    *,
    percentiles: tuple[int, ...],
) -> dict[str, int]:
    n = sum(histogram.values())
    if n <= 0:
        raise AutoInterpConfigError("cannot compute coverage percentiles for zero segments")
    cumulative = 0
    sorted_items = sorted(histogram.items())
    out: dict[str, int] = {}
    for percentile in percentiles:
        target_index = min(int(percentile / 100 * n), n - 1)
        while sorted_items:
            value, count = sorted_items[0]
            if cumulative + count > target_index:
                out[f"p{percentile}"] = value
                break
            cumulative += count
            sorted_items.pop(0)
    return out
