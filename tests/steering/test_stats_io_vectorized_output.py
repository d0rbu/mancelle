"""Regression tests for bulk per-feature statistics serialization.

The scalar reference in this file is the pre-vectorization implementation. It
is intentionally test-only: production keeps one implementation, while these
tests pin the exact tensor contents produced by the old loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

import steering.stats_io as stats_io
from steering.segmentation import SegmentMode
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

_PER_FEATURE_TENSOR_NAMES = (
    "feature_ids",
    "counts",
    "pct_segments_active",
    "mean_mag",
    "mean_mag_incl_zeros",
    "std_mag",
    "min_mag",
    "max_mag",
    "median_mag",
    "percentile_values",
    "histogram_counts",
)


def _stat(
    seed: int,
    *,
    percentiles: dict[str, float],
    histogram: dict[str, object] | None,
) -> PerFeatureStat:
    """Build a feature row whose fields are easy to distinguish."""
    return PerFeatureStat(
        count=2**40 + seed,
        pct_segments_active=seed / 7,
        mean_mag=seed + 0.125,
        mean_mag_incl_zeros=seed + 0.25,
        std_mag=seed + 0.375,
        min_mag=-(seed + 0.5),
        max_mag=seed + 0.625,
        median_mag=seed + 0.75,
        percentiles=percentiles,
        histogram=histogram,
    )


def _statistics(
    per_feature_stats: dict[str, PerFeatureStat],
) -> ExtractionStatistics:
    """Build a complete object so the public writer exercises every section."""
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=22,
            n_sae_features=65_536,
            generated_at="2026-08-12T00:00:00+00:00",
            activation_percentile=0.5,
        ),
        corpus=CorpusStats(
            n_shards_processed=3,
            n_docs=10,
            n_tokens_gpt2=20,
            mean_tokens_per_doc=2.0,
            median_tokens_per_doc=2,
        ),
        features=SegmentFeatureStats(
            n_segments=11,
            n_features_seen=len(per_feature_stats),
            mean_active_per_segment=2.5,
            median_active_per_segment=2,
            p90_active_per_segment=4,
            max_active_per_segment=5,
            min_active_per_segment=0,
            n_total_sae_features=65_536,
            pct_features_active=0.01,
            count_percentiles={"p50": 2},
            magnitude_stats={"mean": 1.25},
            active_per_segment_histogram={
                "bin_edges": [0.0, 1.0],
                "counts": [11],
            },
            active_per_segment_percentiles={"p50": 2.0},
            mag_histogram_bin_edges=[0.0, 0.5, 1.0],
            segment_length_histogram=None,
            segment_length_percentiles=None,
            features_per_token_histogram=None,
            features_per_token_percentiles=None,
        ),
        storage=StorageStats(
            total_bytes=123,
            mean_bytes_per_doc=12.3,
            mean_bytes_per_token=6.15,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=5_000_000_000,
            estimated_storage_gb=57.0,
            estimated_wall_time_hours=3.5,
        ),
        per_feature_thresholds={"3": 0.75},
        ignore_bos_features=True,
        health_metrics={"finite_fraction": "1.0"},
    )


def _scalar_reference(
    per_feature_stats: dict[str, PerFeatureStat],
) -> tuple[dict[str, t.Tensor], dict[str, str]]:
    """Construct per-feature outputs exactly as the old scalar loop did."""
    tensors: dict[str, t.Tensor] = {}
    metadata: dict[str, str] = {}
    fids = list(per_feature_stats.keys())
    n_active = len(fids)

    if n_active == 0:
        return tensors, metadata

    tensors["feature_ids"] = t.tensor([int(fid) for fid in fids], dtype=t.int32)
    first_stat = next(iter(per_feature_stats.values()))
    pct_keys = sorted(first_stat.percentiles.keys())
    metadata["percentile_names"] = ",".join(pct_keys)

    counts = t.zeros(n_active, dtype=t.int64)
    pct_segments_active = t.zeros(n_active, dtype=t.float32)
    mean_mag = t.zeros(n_active, dtype=t.float64)
    mean_mag_incl_zeros = t.zeros(n_active, dtype=t.float64)
    std_mag = t.zeros(n_active, dtype=t.float64)
    min_mag = t.zeros(n_active, dtype=t.float64)
    max_mag = t.zeros(n_active, dtype=t.float64)
    median_mag = t.zeros(n_active, dtype=t.float64)
    percentile_values = t.zeros(n_active, len(pct_keys), dtype=t.float64)
    has_histogram = False
    hist_n_bins = 0

    for i, fstat in enumerate(per_feature_stats.values()):
        counts[i] = fstat.count
        pct_segments_active[i] = fstat.pct_segments_active
        mean_mag[i] = fstat.mean_mag
        mean_mag_incl_zeros[i] = fstat.mean_mag_incl_zeros
        std_mag[i] = fstat.std_mag
        min_mag[i] = fstat.min_mag
        max_mag[i] = fstat.max_mag
        median_mag[i] = fstat.median_mag
        for j, percentile_name in enumerate(pct_keys):
            if percentile_name in fstat.percentiles:
                percentile_values[i, j] = fstat.percentiles[percentile_name]
        if fstat.histogram and "counts" in fstat.histogram and not has_histogram:
            has_histogram = True
            hist_n_bins = len(
                fstat.histogram["counts"]  # ty: ignore[invalid-argument-type]
            )

    tensors["counts"] = counts
    tensors["pct_segments_active"] = pct_segments_active
    tensors["mean_mag"] = mean_mag
    tensors["mean_mag_incl_zeros"] = mean_mag_incl_zeros
    tensors["std_mag"] = std_mag
    tensors["min_mag"] = min_mag
    tensors["max_mag"] = max_mag
    tensors["median_mag"] = median_mag
    tensors["percentile_values"] = percentile_values

    if has_histogram and hist_n_bins > 0:
        histogram_counts = t.zeros(n_active, hist_n_bins, dtype=t.int64)
        for i, fstat in enumerate(per_feature_stats.values()):
            if fstat.histogram and "counts" in fstat.histogram:
                histogram_counts[i] = t.tensor(fstat.histogram["counts"], dtype=t.int64)
        tensors["histogram_counts"] = histogram_counts

    return tensors, metadata


def _capture_writer_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    statistics: ExtractionStatistics,
) -> tuple[dict[str, t.Tensor], dict[str, str]]:
    captured: dict[str, object] = {}

    def capture(
        tensors: dict[str, t.Tensor],
        path: Path,
        *,
        metadata: dict[str, str],
    ) -> None:
        captured["tensors"] = tensors
        captured["metadata"] = metadata

    monkeypatch.setattr(stats_io, "atomic_safetensors_save", capture)
    result = stats_io.save_statistics(statistics, tmp_path / "statistics.json")

    assert result == tmp_path / "statistics.safetensors"
    assert set(captured) == {"tensors", "metadata"}
    return captured["tensors"], captured["metadata"]  # ty: ignore[invalid-return-type]


def _assert_matches_scalar_reference(
    actual_tensors: dict[str, t.Tensor],
    actual_metadata: dict[str, str],
    per_feature_stats: dict[str, PerFeatureStat],
) -> None:
    expected_tensors, expected_metadata = _scalar_reference(per_feature_stats)

    actual_per_feature = {
        key: value for key, value in actual_tensors.items() if key in _PER_FEATURE_TENSOR_NAMES
    }
    assert tuple(actual_per_feature) == tuple(expected_tensors)
    for key, expected in expected_tensors.items():
        t.testing.assert_close(actual_per_feature[key], expected, rtol=0, atol=0, equal_nan=True)

    assert actual_metadata.get("percentile_names") == expected_metadata.get("percentile_names")


def test_vectorized_output_matches_scalar_reference_for_diverse_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    per_feature_stats = {
        # Deliberately not feature-id sorted: insertion order is the on-disk row order.
        "42": _stat(
            1,
            percentiles={"p90": 9.0, "p10": 1.0, "p50": 5.0},
            histogram=None,
        ),
        # Missing p50 is zero-filled; p99 is ignored because the first row
        # defines the serialized percentile columns.
        "3": _stat(
            2,
            percentiles={"p10": -1.5, "p90": float("inf"), "p99": 99.0},
            histogram={"counts": [2, 0, 7]},
        ),
        "1001": _stat(
            3,
            percentiles={"p10": float("nan"), "p50": -0.0},
            histogram={"counts": [0, 5, 1]},
        ),
    }
    statistics = _statistics(per_feature_stats)
    tensors, metadata = _capture_writer_output(monkeypatch, tmp_path, statistics)

    _assert_matches_scalar_reference(tensors, metadata, per_feature_stats)
    assert tensors["feature_ids"].tolist() == [42, 3, 1001]
    assert tensors["percentile_values"].shape == (3, 3)
    assert tensors["percentile_values"][1].tolist() == [-1.5, 0.0, float("inf")]
    assert tensors["histogram_counts"].tolist() == [
        [0, 0, 0],
        [2, 0, 7],
        [0, 5, 1],
    ]
    assert tensors["counts"].dtype == t.int64
    assert tensors["pct_segments_active"].dtype == t.float32
    for key in (
        "mean_mag",
        "mean_mag_incl_zeros",
        "std_mag",
        "min_mag",
        "max_mag",
        "median_mag",
        "percentile_values",
    ):
        assert tensors[key].dtype == t.float64
    assert tensors["histogram_counts"].dtype == t.int64


def test_vectorized_output_preserves_empty_first_row_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    per_feature_stats = {
        "7": _stat(1, percentiles={}, histogram={"counts": []}),
        "8": _stat(
            2,
            percentiles={"p50": 5.0},
            histogram={"counts": [1, 2]},
        ),
    }
    tensors, metadata = _capture_writer_output(
        monkeypatch, tmp_path, _statistics(per_feature_stats)
    )

    _assert_matches_scalar_reference(tensors, metadata, per_feature_stats)
    assert metadata["percentile_names"] == ""
    assert tensors["percentile_values"].shape == (2, 0)
    assert "histogram_counts" not in tensors


def test_vectorized_output_preserves_empty_feature_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tensors, metadata = _capture_writer_output(monkeypatch, tmp_path, _statistics({}))

    _assert_matches_scalar_reference(tensors, metadata, {})
    assert not set(tensors).intersection(_PER_FEATURE_TENSOR_NAMES)
    assert "percentile_names" not in metadata
