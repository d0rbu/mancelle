"""Tests for steering.compute_statistics percentile threshold logic."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch as t

import steering.compute_statistics as statistics_module
from errors import EmptyExtractionSegmentError
from steering.compute_statistics import (
    SegmentStatsAccumulator,
    accumulate_segment_stats,
    accumulate_shard,
    compute_per_feature_percentiles,
    compute_per_feature_thresholds,
    compute_statistics,
    filter_shard_by_percentile,
    gpu_accumulate_shard,
    init_accumulators,
    init_segment_stats,
    load_all_shards,
    sanity_check_percentiles,
)
from steering.feature_io import glob_features, load_features, save_features
from steering.stats_io import load_statistics
from tests.steering._compute_statistics_reference import (
    gpu_exact_percentiles,
    gpu_merge_per_feature_percentiles,
    merge_per_feature_percentiles,
    optimized_pass1,
    stream_pass1,
    stream_per_feature_percentiles,
    vectorized_pass2_percentiles,
)
from tests.steering._helpers import save_raw_shard


class TestComputePerFeatureThresholds:
    """Unit tests for compute_per_feature_thresholds."""

    def test_basic_single_feature(self):
        """Single feature with known activations → threshold matches torch.quantile."""
        feat_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([1.0, 5.0, 6.0, 7.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert thresholds.shape == (1,)  # max_id=0 → size 1
        expected = t.quantile(t.tensor([1.0, 5.0, 6.0, 7.0], dtype=t.float64), 0.5)
        assert abs(thresholds[0].item() - expected.item()) < 1e-6

    def test_multiple_features(self):
        """Two features with distinct distributions."""
        feat_ids = t.tensor([0, 0, 1, 1, 1], dtype=t.int32)
        feat_mags = t.tensor([2.0, 8.0, 1.0, 3.0, 5.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert thresholds.shape == (2,)
        # Feature 0: quantile([2, 8], 0.5) = 5.0
        assert abs(thresholds[0].item() - 5.0) < 1e-6
        # Feature 1: quantile([1, 3, 5], 0.5) = 3.0
        assert abs(thresholds[1].item() - 3.0) < 1e-6

    def test_zeros_are_ignored(self):
        """Zero magnitudes should be excluded from percentile computation."""
        feat_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([0.0, 0.0, 1.0, 5.0, 9.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        expected = t.quantile(t.tensor([1.0, 5.0, 9.0], dtype=t.float64), 0.5)
        assert abs(thresholds[0].item() - expected.item()) < 1e-6

    def test_all_zeros_gives_inf(self):
        """Feature with only zero activations → threshold = inf (nothing passes)."""
        feat_ids = t.tensor([0, 0], dtype=t.int32)
        feat_mags = t.tensor([0.0, 0.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert thresholds[0].item() == float("inf")

    def test_empty_input(self):
        """Empty tensors → empty result."""
        feat_ids = t.tensor([], dtype=t.int32)
        feat_mags = t.tensor([], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert len(thresholds) == 0

    def test_sparse_feature_ids(self):
        """Non-contiguous feature IDs → gaps filled with inf."""
        feat_ids = t.tensor([0, 5], dtype=t.int32)
        feat_mags = t.tensor([3.0, 7.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert thresholds.shape == (6,)  # IDs 0..5
        assert thresholds[0].item() == 3.0
        assert thresholds[5].item() == 7.0
        for i in [1, 2, 3, 4]:
            assert thresholds[i].item() == float("inf"), f"Gap ID {i} should be inf"

    def test_percentile_zero_returns_min(self):
        """Percentile 0.0 → threshold = minimum non-zero activation."""
        feat_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([1.0, 5.0, 9.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.0)
        assert abs(thresholds[0].item() - 1.0) < 1e-6

    def test_percentile_one_returns_max(self):
        """Percentile 1.0 → threshold = maximum activation."""
        feat_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([1.0, 5.0, 9.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 1.0)
        assert abs(thresholds[0].item() - 9.0) < 1e-6

    def test_output_dtype_is_float64(self):
        """Output tensor should be float64 for precision."""
        feat_ids = t.tensor([0], dtype=t.int32)
        feat_mags = t.tensor([1.0], dtype=t.float32)
        thresholds = compute_per_feature_thresholds(feat_ids, feat_mags, 0.5)
        assert thresholds.dtype == t.float64


def _make_shard(
    feat_ids: list[int],
    feat_mags: list[float],
    offsets: list[int],
    n_segments: int,
    *,
    n_docs: int = 1,
    doc_token_counts: list[int] | None = None,
    segment_mode: str = "fixed_tokens",
    segment_size: int = 128,
    context_window: int = 1024,
    pooling: str = "mean_pool",
    layer: int = 8,
) -> dict:
    """Build a minimal shard dict matching the format of feature shard files."""
    if doc_token_counts is None:
        doc_token_counts = [1024] * n_docs
    return {
        "feature_ids": t.tensor(feat_ids, dtype=t.int32),
        "feature_mags": t.tensor(feat_mags, dtype=t.float32),
        "offsets": t.tensor(offsets, dtype=t.int64),
        "n_segments": n_segments,
        "n_docs": n_docs,
        "doc_ids": t.zeros(n_segments, dtype=t.int64),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "segment_mode": segment_mode,
        "segment_size": segment_size,
        "context_window": context_window,
        "pooling": pooling,
        "layer": layer,
    }


class TestFilterShardByPercentile:
    """Unit tests for filter_shard_by_percentile."""

    def test_no_filtering_when_all_above(self):
        """All activations above threshold → shard unchanged."""
        shard = _make_shard(
            feat_ids=[0, 0, 1],
            feat_mags=[5.0, 6.0, 7.0],
            offsets=[0, 2, 3],
            n_segments=2,
        )
        thresholds = t.tensor([1.0, 1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert result["feature_ids"].tolist() == [0, 0, 1]
        assert result["feature_mags"].tolist() == [5.0, 6.0, 7.0]
        assert result["offsets"].tolist() == [0, 2, 3]

    def test_some_filtered(self):
        """Some activations below threshold → correctly removed."""
        shard = _make_shard(
            feat_ids=[0, 1, 0],
            feat_mags=[1.0, 5.0, 3.0],
            offsets=[0, 2, 3],
            n_segments=2,
        )
        thresholds = t.tensor([2.0, 1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert result["feature_ids"].tolist() == [1, 0]
        assert result["feature_mags"].tolist() == [5.0, 3.0]
        assert result["offsets"].tolist() == [0, 1, 2]

    def test_all_filtered_out(self):
        """All activations below threshold → empty result with correct offsets."""
        shard = _make_shard(
            feat_ids=[0, 1],
            feat_mags=[1.0, 2.0],
            offsets=[0, 1, 2],
            n_segments=2,
        )
        thresholds = t.tensor([10.0, 10.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert len(result["feature_ids"]) == 0
        assert len(result["feature_mags"]) == 0
        assert result["offsets"].tolist() == [0, 0, 0]

    def test_empty_shard(self):
        """Shard with no activations → unchanged."""
        shard = _make_shard(
            feat_ids=[],
            feat_mags=[],
            offsets=[0, 0],
            n_segments=1,
        )
        thresholds = t.tensor([1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert len(result["feature_ids"]) == 0
        assert result["offsets"].tolist() == [0, 0]

    def test_feature_id_beyond_threshold_tensor(self):
        """Feature ID beyond threshold tensor size → NOT kept (threshold = inf)."""
        shard = _make_shard(
            feat_ids=[0, 5],
            feat_mags=[1.0, 2.0],
            offsets=[0, 1, 2],
            n_segments=2,
        )
        thresholds = t.tensor([0.5], dtype=t.float64)  # Only covers feature 0
        result = filter_shard_by_percentile(shard, thresholds)
        assert result["feature_ids"].tolist() == [0]
        assert result["feature_mags"].tolist() == [1.0]

    def test_offsets_consistency(self):
        """Verify offsets[i+1] - offsets[i] == count of features in segment i."""
        shard = _make_shard(
            feat_ids=[0, 1, 2, 0, 1, 2],
            feat_mags=[1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            offsets=[0, 3, 6],
            n_segments=2,
        )
        thresholds = t.tensor([2.0, 3.0, 4.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        offsets = result["offsets"]
        ids = result["feature_ids"]
        for seg_idx in range(2):
            start = offsets[seg_idx].item()
            end = offsets[seg_idx + 1].item()
            seg_ids = ids[start:end]
            seg_mags = result["feature_mags"][start:end]
            for fid, mag in zip(seg_ids.tolist(), seg_mags.tolist(), strict=True):
                assert mag > thresholds[fid].item(), (
                    f"Segment {seg_idx}: feature {fid} mag {mag} "
                    f"<= threshold {thresholds[fid].item()}"
                )

    def test_metadata_preserved(self):
        """Non-feature fields (n_docs, doc_token_counts, etc.) are preserved."""
        shard = _make_shard(
            feat_ids=[0],
            feat_mags=[5.0],
            offsets=[0, 1],
            n_segments=1,
            n_docs=3,
            doc_token_counts=[100, 200, 300],
        )
        thresholds = t.tensor([1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert result["n_segments"] == 1
        assert result["n_docs"] == 3
        assert result["doc_token_counts"].tolist() == [100, 200, 300]
        assert result["segment_mode"] == "fixed_tokens"


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def _per_doc_segment_geometry(
    doc_ids: t.Tensor,
    segment_size: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Build realistic indices/positions that restart at each document."""
    indices: list[int] = []
    positions: list[int] = []
    previous_doc: int | None = None
    segment_index = 0
    for raw_doc_id in doc_ids.tolist():
        doc_id = int(raw_doc_id)
        if doc_id != previous_doc:
            previous_doc = doc_id
            segment_index = 0
        indices.append(segment_index)
        positions.append(segment_index * segment_size)
        segment_index += 1
    return (
        t.tensor(indices, dtype=t.int32),
        t.tensor(positions, dtype=t.int32),
    )


def _write_synthetic_shards(tmp_dir: Path, n_shards: int = 2) -> None:
    """Write synthetic feature shard files for integration tests.

    Creates shards with 3 features:
      - Feature 0: magnitudes ~ [1, 2, 3, 4, 5] (spread evenly)
      - Feature 1: magnitudes ~ [10, 20, 30, 40, 50] (high activation)
      - Feature 2: magnitudes ~ [0.1, 0.2, 0.3, 0.4, 0.5] (low activation)
    """
    for shard_idx in range(n_shards):
        n_segments = 5
        feat_ids = []
        feat_mags = []
        offsets = [0]
        for seg in range(n_segments):
            seg_feats = [0, 1, 2]
            seg_mags = [
                float(seg + 1),
                float((seg + 1) * 10),
                float((seg + 1) * 0.1),
            ]
            feat_ids.extend(seg_feats)
            feat_mags.extend(seg_mags)
            offsets.append(offsets[-1] + len(seg_feats))

        doc_ids = t.zeros(n_segments, dtype=t.int32)
        segment_indices, segment_positions = _per_doc_segment_geometry(doc_ids, 128)
        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": 1,
            "doc_ids": doc_ids,
            "doc_token_counts": t.tensor([640], dtype=t.int32),
            "segment_indices": segment_indices,
            "segment_positions": segment_positions,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestComputeStatisticsIntegration:
    """Integration tests for compute_statistics with activation_percentile."""

    def test_no_percentile_processes_all(self, tmp_path):
        """activation_percentile=0.0 -> all activations processed."""
        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.0,
        )
        assert out.with_suffix(".safetensors").exists()
        assert len(stats.per_feature_stats) == 3
        f0 = stats.per_feature_stats["0"]
        assert f0.count == 10  # 5 segments * 2 shards

    def test_percentile_filters_activations(self, tmp_path):
        """activation_percentile > 0 -> low activations filtered out."""
        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.5,
            streaming=False,
        )
        f0 = stats.per_feature_stats["0"]
        assert f0.count < 10, "Percentile filter should reduce activation count"

    def test_high_percentile_keeps_only_top(self, tmp_path):
        """activation_percentile=0.9 -> only very high activations survive."""
        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.9,
            streaming=False,
        )
        for fid, fstat in stats.per_feature_stats.items():
            assert fstat.count <= 3, (
                f"Feature {fid} has {fstat.count} activations at p90, expected <= 3"
            )

    def test_percentile_metadata_recorded(self, tmp_path):
        """activation_percentile value should be recorded in metadata."""
        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.3,
            streaming=False,
        )
        assert stats.metadata.activation_percentile == 0.3

    def test_per_feature_threshold_values_in_output(self, tmp_path):
        """Output should include resolved per-feature threshold magnitudes."""
        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.5,
            streaming=False,
        )
        assert stats.per_feature_thresholds is not None
        assert len(stats.per_feature_thresholds) > 0
        for fid, thresh in stats.per_feature_thresholds.items():
            assert thresh > 0, f"Feature {fid} threshold should be positive"
            assert thresh != float("inf"), f"Feature {fid} threshold should be finite"

    def test_round_trip(self, tmp_path):
        """Stats file can be loaded back and has expected structure."""

        _write_synthetic_shards(tmp_path)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.5,
            streaming=False,
        )
        sf_path = tmp_path / "stats.safetensors"
        assert sf_path.exists()
        data = load_statistics(sf_path)
        assert data.per_feature_thresholds is not None
        assert data.metadata.activation_percentile == 0.5
        assert data.metadata.n_sae_features == stats.metadata.n_sae_features
        assert len(data.per_feature_stats) == len(stats.per_feature_stats)

    def test_mean_magnitude_increases_with_percentile(self, tmp_path):
        """Higher percentile -> higher mean magnitude for surviving activations."""
        _write_synthetic_shards(tmp_path)
        mean_mags = {}
        for pct in [0.0, 0.3, 0.7]:
            out = tmp_path / f"stats_p{pct}"
            stats = compute_statistics(
                feature_dir=str(tmp_path),
                output_path=str(out),
                activation_percentile=pct,
                streaming=False,
            )
            mags = [f.mean_mag for f in stats.per_feature_stats.values()]
            mean_mags[pct] = sum(mags) / len(mags) if mags else 0

        assert mean_mags[0.3] >= mean_mags[0.0], (
            f"p0.3 mean_mag {mean_mags[0.3]} should >= p0.0 {mean_mags[0.0]}"
        )
        assert mean_mags[0.7] >= mean_mags[0.3], (
            f"p0.7 mean_mag {mean_mags[0.7]} should >= p0.3 {mean_mags[0.3]}"
        )


class TestComputeStatisticsEdgeCases:
    """Edge case tests for compute_statistics with percentile filtering."""

    def test_percentile_zero_same_as_no_filter(self, tmp_path):
        """activation_percentile=0.0 produces identical output to omitting it."""
        _write_synthetic_shards(tmp_path)
        out_base = tmp_path / "base"
        out_zero = tmp_path / "zero"

        stats_base = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_base),
        )
        stats_zero = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_zero),
            activation_percentile=0.0,
        )

        assert len(stats_base.per_feature_stats) == len(stats_zero.per_feature_stats)
        for fid in stats_base.per_feature_stats:
            assert stats_base.per_feature_stats[fid].count == (
                stats_zero.per_feature_stats[fid].count
            )

    def test_single_shard(self, tmp_path):
        """Works with a single shard."""
        _write_synthetic_shards(tmp_path, n_shards=1)
        out = tmp_path / "stats"
        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.5,
            streaming=False,
        )
        assert len(stats.per_feature_stats) > 0

    def test_n_segments_unchanged_by_filtering(self, tmp_path):
        """Total segment count should be same regardless of percentile."""
        _write_synthetic_shards(tmp_path)
        out_base = tmp_path / "base"
        out_filtered = tmp_path / "filtered"

        stats_base = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_base),
            activation_percentile=0.0,
        )
        stats_filtered = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_filtered),
            activation_percentile=0.5,
            streaming=False,
        )

        assert stats_base.features.n_segments == stats_filtered.features.n_segments


# ---------------------------------------------------------------------------
# Vectorized per-feature percentiles correctness tests
# ---------------------------------------------------------------------------


class TestVectorizedPerFeaturePercentiles:
    """Compare vectorized compute_per_feature_percentiles() against
    manual per-feature torch.quantile() reference."""

    def _reference_percentiles(
        self, feat_ids: t.Tensor, feat_mags: t.Tensor, percentiles=(10, 50, 90, 99)
    ):
        """Reference: per-feature loop with torch.quantile."""
        result = {}
        unique_ids = feat_ids.unique()
        for fid in unique_ids:
            mask = feat_ids == fid
            mags = feat_mags[mask].to(t.float64)
            pcts = {}
            for p in percentiles:
                pcts[f"p{p}"] = round(t.quantile(mags, p / 100.0).item(), 6)
            median = round(t.quantile(mags, 0.5).item(), 6)
            result[int(fid.item())] = {"percentiles": pcts, "median": median}
        return result

    def test_single_feature(self):
        feat_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=t.float32)
        result, _ = compute_per_feature_percentiles(feat_ids, feat_mags)
        ref = self._reference_percentiles(feat_ids, feat_mags)
        for fid in ref:
            assert abs(result[fid].median - ref[fid]["median"]) < 1e-4
            for pkey in ref[fid]["percentiles"]:
                if pkey in result[fid].percentiles:
                    assert (
                        abs(
                            result[fid].percentiles[pkey]
                            - ref[fid]["percentiles"][pkey]
                        )
                        < 1e-4
                    )

    def test_multiple_features(self):
        feat_ids = t.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=t.int32)
        feat_mags = t.tensor(
            [1.0, 5.0, 9.0, 2.0, 4.0, 6.0, 0.1, 0.5, 0.9, 1.3], dtype=t.float32
        )
        result, edges = compute_per_feature_percentiles(feat_ids, feat_mags)
        ref = self._reference_percentiles(feat_ids, feat_mags)
        assert edges is not None
        for fid in ref:
            assert fid in result
            assert abs(result[fid].median - ref[fid]["median"]) < 1e-4

    def test_large_synthetic(self):
        """Test with 1000 features, 100 activations each."""
        t.manual_seed(42)
        n_features = 1000
        n_per_feature = 100
        feat_ids = t.repeat_interleave(
            t.arange(n_features, dtype=t.int32), n_per_feature
        )
        feat_mags = t.rand(n_features * n_per_feature, dtype=t.float32) * 10 + 0.1
        result, edges = compute_per_feature_percentiles(feat_ids, feat_mags)
        assert len(result) == n_features
        assert edges is not None
        # Spot-check a few features against reference
        for fid in [0, 500, 999]:
            mask = feat_ids == fid
            mags = feat_mags[mask].to(t.float64)
            ref_median = t.quantile(mags, 0.5).item()
            assert abs(result[fid].median - ref_median) < 1e-4

    def test_empty_input(self):
        feat_ids = t.tensor([], dtype=t.int32)
        feat_mags = t.tensor([], dtype=t.float32)
        result, edges = compute_per_feature_percentiles(feat_ids, feat_mags)
        assert result == {}
        assert edges is None


class TestVectorizedPerFeatureThresholds:
    """Compare vectorized compute_per_feature_thresholds() against
    manual per-feature loop reference."""

    def _reference_thresholds(
        self, feat_ids: t.Tensor, feat_mags: t.Tensor, percentile: float
    ):
        """Reference: per-feature loop."""
        max_id = int(feat_ids.max().item())
        thresholds = t.full((max_id + 1,), float("inf"), dtype=t.float64)
        for fid in range(max_id + 1):
            mask = (feat_ids == fid) & (feat_mags > 0)
            mags = feat_mags[mask].to(t.float64)
            if len(mags) > 0:
                thresholds[fid] = t.quantile(mags, percentile)
        return thresholds

    def test_matches_reference(self):
        t.manual_seed(0)
        feat_ids = t.repeat_interleave(t.arange(50, dtype=t.int32), 20)
        feat_mags = t.rand(1000, dtype=t.float32) * 5 + 0.01
        for pct in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            result = compute_per_feature_thresholds(feat_ids, feat_mags, pct)
            ref = self._reference_thresholds(feat_ids, feat_mags, pct)
            assert t.allclose(result, ref, atol=1e-4), (
                f"Mismatch at percentile {pct}: max diff = "
                f"{(result - ref).abs().max().item()}"
            )


class TestPreloadedParameter:
    """Verify _preloaded gives identical results and doesn't mutate original."""

    def test_preloaded_matches_fresh(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        out_fresh = tmp_path / "fresh"
        out_preloaded = tmp_path / "preloaded"

        stats_fresh = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_fresh),
            activation_percentile=0.5,
            streaming=False,
        )

        loaded = load_all_shards(str(tmp_path))
        stats_preloaded = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_preloaded),
            activation_percentile=0.5,
            _preloaded=loaded,
            streaming=False,
        )

        assert len(stats_fresh.per_feature_stats) == len(
            stats_preloaded.per_feature_stats
        )
        for fid in stats_fresh.per_feature_stats:
            assert stats_fresh.per_feature_stats[fid].count == (
                stats_preloaded.per_feature_stats[fid].count
            )

    def test_preloaded_not_mutated(self, tmp_path):
        _write_synthetic_shards(tmp_path)
        loaded = load_all_shards(str(tmp_path))
        original_ids = loaded.feature_ids.clone()
        original_mags = loaded.feature_mags.clone()

        out = tmp_path / "stats"
        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            activation_percentile=0.5,
            _preloaded=loaded,
            streaming=False,
        )

        assert t.equal(loaded.feature_ids, original_ids)
        assert t.equal(loaded.feature_mags, original_mags)


class TestKthvalueP99:
    """Verify kthvalue-based p99 bin edge matches sorted-based reference."""

    def test_kthvalue_matches_sorted(self):
        t.manual_seed(123)
        mags = t.rand(100_000, dtype=t.float64)
        # Reference: sort-based
        sorted_vals = mags.sort().values
        k99 = max(1, int(len(mags) * 0.99))
        ref_p99 = sorted_vals[k99 - 1].item()  # 0-indexed
        # kthvalue (1-indexed)
        kth_p99 = mags.kthvalue(k99).values.item()
        assert abs(kth_p99 - ref_p99) < 1e-10

    def test_kthvalue_small_tensor(self):
        mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=t.float64)
        k99 = max(1, int(5 * 0.99))  # k=4
        result = mags.kthvalue(k99).values.item()
        sorted_vals = mags.sort().values
        assert result == sorted_vals[k99 - 1].item()


# ---------------------------------------------------------------------------
# Streaming Pass 1 equivalence tests
# ---------------------------------------------------------------------------


class TestStreamingPass1Equivalence:
    """Verify streaming Pass 1 produces identical results to in-memory."""

    def test_accumulators_match(self, tmp_path):
        """Streaming scatter_add must match eager scatter_add."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        # Reference: load all, accumulate
        loaded = load_all_shards(str(tmp_path))
        ref_acc = init_accumulators(3)  # 3 features in synthetic data
        for data in loaded.shard_data_list:
            accumulate_shard(ref_acc, data["feature_ids"], data["feature_mags"])

        # Streaming
        p1 = stream_pass1(str(tmp_path))

        assert t.equal(p1.acc.occurrence, ref_acc.occurrence)
        assert t.allclose(p1.acc.mag_sum, ref_acc.mag_sum)
        assert t.allclose(p1.acc.mag_sum_sq, ref_acc.mag_sum_sq)
        assert t.equal(p1.acc.mag_min, ref_acc.mag_min)
        assert t.equal(p1.acc.mag_max, ref_acc.mag_max)

    def test_segment_stats_match(self, tmp_path):
        """Segment-level stats must match."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        loaded = load_all_shards(str(tmp_path))
        ref_seg = init_segment_stats()
        for data in loaded.shard_data_list:
            accumulate_segment_stats(ref_seg, data)

        p1 = stream_pass1(str(tmp_path))

        assert p1.seg_acc.features_per_segment == ref_seg.features_per_segment
        assert p1.seg_acc.tokens_per_doc == ref_seg.tokens_per_doc

    def test_metadata_matches(self, tmp_path):
        """Metadata and corpus stats must match."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        loaded = load_all_shards(str(tmp_path))
        p1 = stream_pass1(str(tmp_path))

        assert p1.total_docs == loaded.total_docs
        assert p1.total_tokens == loaded.total_tokens
        assert p1.n_shards == loaded.n_shards
        assert p1.metadata.segment_size == loaded.metadata.segment_size
        assert p1.metadata.layer == loaded.metadata.layer


# ---------------------------------------------------------------------------
# Streaming Pass 2 (batched percentiles) equivalence tests
# ---------------------------------------------------------------------------


class TestStreamingPass2Equivalence:
    """Verify batched streaming percentiles match in-memory reference."""

    def test_percentiles_match_exactly(self, tmp_path):
        """Streaming percentiles must be bit-identical to reference."""
        _write_synthetic_shards(tmp_path, n_shards=3)
        loaded = load_all_shards(str(tmp_path))

        # Reference
        ref_result, ref_edges = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        # Build occurrence counts
        p1 = stream_pass1(str(tmp_path))

        stream_result, stream_edges = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        # Same features
        assert set(ref_result.keys()) == set(stream_result.keys())

        # Same percentiles per feature
        for fid in ref_result:
            ref_pcts = ref_result[fid].percentiles
            stream_pcts = stream_result[fid].percentiles
            for key in ref_pcts:
                assert abs(ref_pcts[key] - stream_pcts[key]) < 1e-6, (
                    f"Feature {fid} {key}: ref={ref_pcts[key]} stream={stream_pcts[key]}"
                )
            assert abs(ref_result[fid].median - stream_result[fid].median) < 1e-6

        # Same bin edges
        if ref_edges is not None:
            assert stream_edges is not None
            assert len(ref_edges) == len(stream_edges)
            for r, s in zip(ref_edges, stream_edges, strict=True):
                assert abs(r - s) < 1e-6

    def test_batch_size_does_not_affect_result(self, tmp_path):
        """Different batch sizes must produce identical results."""
        _write_synthetic_shards(tmp_path, n_shards=2)
        p1 = stream_pass1(str(tmp_path))

        result_b1, edges_b1 = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
            batch_size=1,
        )
        result_b2, edges_b2 = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
            batch_size=2,
        )
        result_b100, edges_b100 = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
            batch_size=100,
        )

        for fid in result_b1:
            for key in result_b1[fid].percentiles:
                assert (
                    result_b1[fid].percentiles[key] == result_b2[fid].percentiles[key]
                )
                assert (
                    result_b1[fid].percentiles[key] == result_b100[fid].percentiles[key]
                )
            assert result_b1[fid].median == result_b2[fid].median
            assert result_b1[fid].median == result_b100[fid].median

        # Bin edges must also be identical across batch sizes
        assert edges_b1 is not None
        assert edges_b1 == edges_b2
        assert edges_b1 == edges_b100

    def test_single_shard(self, tmp_path):
        """Streaming works correctly with a single shard."""
        _write_synthetic_shards(tmp_path, n_shards=1)
        loaded = load_all_shards(str(tmp_path))
        ref_result, ref_edges = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        stream_result, stream_edges = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        assert set(ref_result.keys()) == set(stream_result.keys())
        for fid in ref_result:
            for key in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[key]
                        - stream_result[fid].percentiles[key]
                    )
                    < 1e-6
                )

    def test_histograms_match(self, tmp_path):
        """Per-feature histogram counts must match reference."""
        _write_synthetic_shards(tmp_path, n_shards=3)
        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        stream_result, _ = stream_per_feature_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        for fid in ref_result:
            assert ref_result[fid].histogram == stream_result[fid].histogram, (
                f"Feature {fid}: histogram mismatch"
            )

    def test_empty_shards(self, tmp_path):
        """Empty occurrence tensor produces empty results."""
        _write_synthetic_shards(tmp_path, n_shards=1)
        p1 = stream_pass1(str(tmp_path))

        # Zero out occurrence to simulate no active features
        empty_occurrence = t.zeros_like(p1.acc.occurrence)
        result, edges = stream_per_feature_percentiles(
            p1.shard_paths,
            empty_occurrence,
        )
        assert result == {}
        assert edges is None


# ---------------------------------------------------------------------------
# Streaming end-to-end equivalence tests
# ---------------------------------------------------------------------------


class TestGPUAccumulateShard:
    """Verify GPU scatter accumulation matches CPU reference."""

    @staticmethod
    def _requires_cuda():
        if not t.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_gpu_matches_cpu(self, tmp_path):
        """GPU accumulate_shard must produce identical accumulators to CPU."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)

        pt_files = glob_features(str(tmp_path))

        cpu_acc = init_accumulators(3)
        gpu_acc = init_accumulators(3)

        for pt_path in pt_files:
            data = load_features(pt_path)
            accumulate_shard(cpu_acc, data["feature_ids"], data["feature_mags"])
            gpu_accumulate_shard(gpu_acc, data["feature_ids"], data["feature_mags"])

        assert t.equal(cpu_acc.occurrence, gpu_acc.occurrence)
        assert t.allclose(cpu_acc.mag_sum, gpu_acc.mag_sum, atol=1e-10)
        assert t.allclose(cpu_acc.mag_sum_sq, gpu_acc.mag_sum_sq, atol=1e-10)
        assert t.equal(cpu_acc.mag_min, gpu_acc.mag_min)
        assert t.equal(cpu_acc.mag_max, gpu_acc.mag_max)

    def test_gpu_small_chunk_size(self, tmp_path):
        """Small chunk size still produces correct results."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=2)

        pt_files = glob_features(str(tmp_path))

        cpu_acc = init_accumulators(3)
        gpu_acc = init_accumulators(3)

        for pt_path in pt_files:
            data = load_features(pt_path)
            accumulate_shard(cpu_acc, data["feature_ids"], data["feature_mags"])
            gpu_accumulate_shard(
                gpu_acc,
                data["feature_ids"],
                data["feature_mags"],
                gpu_chunk_size=5,  # tiny chunks to stress chunking logic
            )

        assert t.equal(cpu_acc.occurrence, gpu_acc.occurrence)
        assert t.allclose(cpu_acc.mag_sum, gpu_acc.mag_sum, atol=1e-10)

    def test_gpu_transfers_storage_dtypes_before_widening(self, monkeypatch):
        """Host-to-device copies stay int32/float32; widening happens on CUDA."""
        self._requires_cuda()
        feature_ids = t.tensor([0, 1, 1, 2], dtype=t.int32)
        feature_mags = t.tensor([1.0, 2.0, 3.0, 4.0], dtype=t.float32)
        original_to = t.Tensor.to
        first_cpu_conversion: dict[t.dtype, object] = {}

        def recording_to(tensor, *args, **kwargs):
            if (
                tensor.device.type == "cpu"
                and tensor.numel() == feature_ids.numel()
                and tensor.dtype in (t.int32, t.float32)
            ):
                first_cpu_conversion.setdefault(
                    tensor.dtype,
                    args[0] if args else kwargs,
                )
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(t.Tensor, "to", recording_to)
        gpu_accumulate_shard(
            init_accumulators(3),
            feature_ids,
            feature_mags,
            gpu_chunk_size=len(feature_ids),
        )

        assert first_cpu_conversion == {
            t.int32: t.device("cuda"),
            t.float32: t.device("cuda"),
        }

    def test_gpu_chunk_admission_accounts_for_device_first_transfer(self, monkeypatch):
        """The automatic chunk limit includes the observed 28-byte CUDA peak."""
        monkeypatch.setattr(t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            t.cuda,
            "mem_get_info",
            lambda: (2_800_000_000, 4_000_000_000),
        )

        device, chunk_size = statistics_module._resolve_exact_runtime(
            t.device("cuda"),
            gpu_chunk_size=1_000_000_000,
        )

        assert device == t.device("cuda")
        assert chunk_size == 50_000_000

    def test_gpu_stream_pass1_matches(self, tmp_path):
        """GPU stream_pass1 matches CPU stream_pass1."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)

        cpu_p1 = stream_pass1(str(tmp_path), device="cpu")
        gpu_p1 = stream_pass1(str(tmp_path), device="cuda")

        assert t.equal(cpu_p1.acc.occurrence, gpu_p1.acc.occurrence)
        assert t.allclose(cpu_p1.acc.mag_sum, gpu_p1.acc.mag_sum, atol=1e-10)
        assert t.allclose(cpu_p1.acc.mag_sum_sq, gpu_p1.acc.mag_sum_sq, atol=1e-10)
        assert t.equal(cpu_p1.acc.mag_min, gpu_p1.acc.mag_min)
        assert t.equal(cpu_p1.acc.mag_max, gpu_p1.acc.mag_max)
        assert cpu_p1.total_docs == gpu_p1.total_docs
        assert cpu_p1.total_tokens == gpu_p1.total_tokens

    def test_gpu_empty_shard(self):
        """GPU path handles empty tensors."""
        self._requires_cuda()
        acc = init_accumulators(3)
        gpu_accumulate_shard(
            acc,
            t.tensor([], dtype=t.int32),
            t.tensor([], dtype=t.float32),
        )
        assert acc.occurrence.sum().item() == 0


class TestGPUExactPercentiles:
    """Verify GPU exact percentiles match the reference implementation."""

    @staticmethod
    def _requires_cuda():
        if not t.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_matches_reference_percentiles(self, tmp_path):
        """GPU exact percentiles must match compute_per_feature_percentiles."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)
        loaded = load_all_shards(str(tmp_path))

        ref_result, ref_edges = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        gpu_result, gpu_edges = gpu_exact_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        # Same features
        assert set(ref_result.keys()) == set(gpu_result.keys()), (
            f"Feature set mismatch: ref={sorted(ref_result.keys())} gpu={sorted(gpu_result.keys())}"
        )

        # Same percentiles per feature
        for fid in ref_result:
            ref_pcts = ref_result[fid].percentiles
            gpu_pcts = gpu_result[fid].percentiles
            for key in ref_pcts:
                assert abs(ref_pcts[key] - gpu_pcts[key]) < 1e-5, (
                    f"Feature {fid} {key}: ref={ref_pcts[key]} gpu={gpu_pcts[key]}"
                )
            assert abs(ref_result[fid].median - gpu_result[fid].median) < 1e-5, (
                f"Feature {fid} median: ref={ref_result[fid].median} gpu={gpu_result[fid].median}"
            )

    def test_histograms_match(self, tmp_path):
        """GPU exact per-feature histograms must match reference."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)
        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        gpu_result, _ = gpu_exact_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        for fid in ref_result:
            assert ref_result[fid].histogram == gpu_result[fid].histogram, (
                f"Feature {fid}: histogram mismatch"
            )

    def test_bin_edges_match(self, tmp_path):
        """Shared histogram bin edges must match reference."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)
        loaded = load_all_shards(str(tmp_path))
        _, ref_edges = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        _, gpu_edges = gpu_exact_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        assert ref_edges is not None
        assert gpu_edges is not None
        assert len(ref_edges) == len(gpu_edges)
        for r, g in zip(ref_edges, gpu_edges, strict=True):
            assert abs(r - g) < 1e-5, f"Bin edge mismatch: ref={r} gpu={g}"

    def test_single_shard(self, tmp_path):
        """Works correctly with a single shard."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=1)
        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        gpu_result, _ = gpu_exact_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
        )

        for fid in ref_result:
            for key in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[key]
                        - gpu_result[fid].percentiles[key]
                    )
                    < 1e-5
                )

    def test_small_chunk_size(self, tmp_path):
        """Tiny GPU chunk size still produces correct results."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=2)
        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1 = stream_pass1(str(tmp_path))
        gpu_result, _ = gpu_exact_percentiles(
            p1.shard_paths,
            p1.acc.occurrence,
            gpu_chunk_size=5,  # very small chunks
        )

        for fid in ref_result:
            for key in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[key]
                        - gpu_result[fid].percentiles[key]
                    )
                    < 1e-5
                )

    def test_empty_occurrence(self, tmp_path):
        """Empty occurrence tensor produces empty results."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=1)
        p1 = stream_pass1(str(tmp_path))
        empty_occ = t.zeros_like(p1.acc.occurrence)
        result, edges = gpu_exact_percentiles(p1.shard_paths, empty_occ)
        assert result == {}
        assert edges is None


class TestGPUEndToEnd:
    """Full pipeline: GPU compute_statistics must match CPU version."""

    @staticmethod
    def _requires_cuda():
        if not t.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_gpu_matches_cpu_full_pipeline(self, tmp_path):
        """GPU pipeline (safetensors) must produce identical stats to CPU (JSON)."""
        self._requires_cuda()

        _write_synthetic_shards(tmp_path, n_shards=3)

        # CPU reference (JSON)
        cpu_out = tmp_path / "cpu_stats"
        cpu_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(cpu_out),
            streaming=True,
            device="cpu",
        )

        # GPU (safetensors)
        gpu_out = tmp_path / "gpu_stats"
        gpu_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(gpu_out),
            streaming=True,
            device="cuda",
        )

        # Compare per-feature stats
        for fid in cpu_stats.per_feature_stats:
            cpu_f = cpu_stats.per_feature_stats[fid]
            gpu_f = gpu_stats.per_feature_stats[fid]
            assert cpu_f.count == gpu_f.count, f"Feature {fid} count mismatch"
            assert abs(cpu_f.mean_mag - gpu_f.mean_mag) < 1e-4
            assert abs(cpu_f.median_mag - gpu_f.median_mag) < 1e-4
            for pkey in cpu_f.percentiles:
                assert abs(cpu_f.percentiles[pkey] - gpu_f.percentiles[pkey]) < 1e-4, (
                    f"Feature {fid} {pkey}: cpu={cpu_f.percentiles[pkey]} gpu={gpu_f.percentiles[pkey]}"
                )

        # Compare corpus stats
        assert cpu_stats.corpus.n_docs == gpu_stats.corpus.n_docs
        assert cpu_stats.corpus.n_tokens_gpt2 == gpu_stats.corpus.n_tokens_gpt2
        assert cpu_stats.features.n_segments == gpu_stats.features.n_segments
        assert cpu_stats.features.n_features_seen == gpu_stats.features.n_features_seen

        # Verify safetensors file loads correctly
        sf_path = tmp_path / "gpu_stats.safetensors"
        assert sf_path.exists()
        loaded = load_statistics(sf_path)
        assert len(loaded.per_feature_stats) == len(cpu_stats.per_feature_stats)


class TestStreamingEndToEnd:
    """Full pipeline: streaming compute_statistics must match eager version."""

    def test_full_output_matches(self, tmp_path):
        """JSON output from streaming must match reference eager computation."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        # Reference (eager)
        ref_out = tmp_path / "ref_stats"
        ref_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(ref_out),
            streaming=False,
        )

        # Streaming
        stream_out = tmp_path / "stream_stats"
        stream_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(stream_out),
            streaming=True,
        )

        # Compare per-feature stats
        for fid in ref_stats.per_feature_stats:
            ref_f = ref_stats.per_feature_stats[fid]
            stream_f = stream_stats.per_feature_stats[fid]
            assert ref_f.count == stream_f.count, f"Feature {fid} count mismatch"
            assert abs(ref_f.mean_mag - stream_f.mean_mag) < 1e-4
            assert abs(ref_f.median_mag - stream_f.median_mag) < 1e-4
            for pkey in ref_f.percentiles:
                assert abs(ref_f.percentiles[pkey] - stream_f.percentiles[pkey]) < 1e-4

        # Compare corpus stats
        assert ref_stats.corpus.n_docs == stream_stats.corpus.n_docs
        assert ref_stats.corpus.n_tokens_gpt2 == stream_stats.corpus.n_tokens_gpt2

        # Compare segment feature stats
        assert ref_stats.features.n_segments == stream_stats.features.n_segments
        assert (
            ref_stats.features.n_features_seen == stream_stats.features.n_features_seen
        )


# ---------------------------------------------------------------------------
# Optimized Pass 1 + Merge Pass 2 tests
# ---------------------------------------------------------------------------


class TestOptimizedPass1:
    """Tests for optimized_pass1 (no double-read, writes sorted temp files)."""

    def test_accumulators_match_stream_pass1(self, tmp_path):
        """Accumulators from optimized_pass1 must match stream_pass1."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        old_p1 = stream_pass1(str(tmp_path))
        new_p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        assert t.equal(old_p1.acc.occurrence, new_p1.acc.occurrence)
        assert t.allclose(old_p1.acc.mag_sum, new_p1.acc.mag_sum, atol=1e-10)
        assert t.allclose(old_p1.acc.mag_sum_sq, new_p1.acc.mag_sum_sq, atol=1e-10)
        assert t.allclose(old_p1.acc.mag_min, new_p1.acc.mag_min)
        assert t.allclose(old_p1.acc.mag_max, new_p1.acc.mag_max)
        assert old_p1.n_sae_features == new_p1.n_sae_features
        assert old_p1.total_docs == new_p1.total_docs
        assert old_p1.total_tokens == new_p1.total_tokens
        assert old_p1.n_shards == new_p1.n_shards

    def test_sorted_shard_counts_consistent(self, tmp_path):
        """Per-shard feature counts must sum to global occurrence."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        p1, sorted_shards, _, _ = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        n = p1.n_sae_features
        total_counts = t.zeros(n, dtype=t.int64)
        for si in sorted_shards:
            counts = si.per_feature_counts
            if len(counts) < n:
                counts = t.cat([counts, t.zeros(n - len(counts), dtype=t.int64)])
            total_counts += counts

        assert t.equal(total_counts, p1.acc.occurrence)

    def test_temp_files_round_trip_via_torch_load(self, tmp_path):
        """Sorted temp files must round-trip via torch.load with correct shape/dtype."""
        _write_synthetic_shards(tmp_path, n_shards=2)

        p1, sorted_shards, _, _ = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        for si in sorted_shards:
            if si.n_activations == 0:
                continue
            assert si.mags_path is not None
            assert si.mags_path.exists()
            loaded = t.load(si.mags_path, mmap=True, weights_only=True)
            assert loaded.shape == (si.n_activations,)
            # These shards were written through save_features, which stores
            # feature_mags as fp16 and load_features upcasts them to fp32.
            # optimized_pass1 only permutes those values, so the spill should
            # take the lossless fp16 path. Unsafe direct fp32 inputs are covered
            # by test_compute_statistics_fp16_spill.py.
            assert loaded.dtype == t.float16
            assert t.isfinite(loaded).all()

    def test_subsample_nonempty(self, tmp_path):
        """Magnitude subsample must be non-empty for non-empty data."""
        _write_synthetic_shards(tmp_path, n_shards=2)

        _, _, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        assert len(mag_sub) > 0
        assert mag_min > 0  # synthetic data has positive magnitudes


class TestMergePerFeaturePercentiles:
    """Tests for merge_per_feature_percentiles vs reference implementation."""

    def test_percentiles_match_eager(self, tmp_path):
        """Merge percentiles must match eager compute_per_feature_percentiles."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        # Reference (eager)
        loaded = load_all_shards(str(tmp_path))
        ref_result, ref_edges = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        # Merge-based
        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        merge_result, merge_edges = merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        # Compare percentiles and medians (these should match exactly)
        assert set(ref_result.keys()) == set(merge_result.keys())
        for fid in ref_result:
            ref_fp = ref_result[fid]
            merge_fp = merge_result[fid]
            assert abs(ref_fp.median - merge_fp.median) < 1e-5, (
                f"Feature {fid} median: ref={ref_fp.median}, merge={merge_fp.median}"
            )
            for pkey in ref_fp.percentiles:
                assert (
                    abs(ref_fp.percentiles[pkey] - merge_fp.percentiles[pkey]) < 1e-5
                ), (
                    f"Feature {fid} {pkey}: "
                    f"ref={ref_fp.percentiles[pkey]}, merge={merge_fp.percentiles[pkey]}"
                )

    def test_occurrence_assertion_fires(self, tmp_path):
        """If temp data is corrupted, the occurrence assertion should fire."""
        _write_synthetic_shards(tmp_path, n_shards=1)

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        # Corrupt occurrence count
        corrupted_occurrence = p1.acc.occurrence.clone()
        corrupted_occurrence[0] += 999

        with pytest.raises(AssertionError, match="merge count"):
            merge_per_feature_percentiles(
                sorted_shards,
                p1.n_sae_features,
                corrupted_occurrence,
                mag_sub,
                mag_min,
            )

    def test_single_shard(self, tmp_path):
        """Works correctly with a single shard."""
        _write_synthetic_shards(tmp_path, n_shards=1)

        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        merge_result, _ = merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        for fid in ref_result:
            for pkey in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[pkey]
                        - merge_result[fid].percentiles[pkey]
                    )
                    < 1e-5
                )

    def test_many_features(self, tmp_path):
        """Correctness with more features (100) and multiple shards."""
        n_features = 100
        n_shards = 3
        for shard_idx in range(n_shards):
            n_segments = 10
            feat_ids = []
            feat_mags = []
            offsets = [0]
            for seg in range(n_segments):
                # Random subset of features per segment
                t.manual_seed(shard_idx * 1000 + seg)
                n_active = t.randint(5, 20, (1,)).item()
                seg_ids = t.randint(0, n_features, (n_active,)).tolist()  # ty: ignore[no-matching-overload]
                seg_mags = (t.rand(n_active) * 10 + 0.1).tolist()  # ty: ignore[no-matching-overload]
                feat_ids.extend(seg_ids)
                feat_mags.extend(seg_mags)
                offsets.append(offsets[-1] + n_active)

            doc_ids = t.tensor([0] * 5 + [1] * 5, dtype=t.int32)
            segment_indices, segment_positions = _per_doc_segment_geometry(doc_ids, 128)
            shard = {
                "feature_ids": t.tensor(feat_ids, dtype=t.int32),
                "feature_mags": t.tensor(feat_mags, dtype=t.float32),
                "offsets": t.tensor(offsets, dtype=t.int64),
                "n_segments": n_segments,
                "n_docs": 2,
                "doc_ids": doc_ids,
                "doc_token_counts": t.tensor([640, 640], dtype=t.int32),
                "segment_indices": segment_indices,
                "segment_positions": segment_positions,
                "segment_mode": "fixed_tokens",
                "segment_size": 128,
                "context_window": 1024,
                "pooling": "mean_pool",
                "layer": 8,
            }
            save_features(shard, tmp_path / f"shard_{shard_idx}_features")

        # Reference
        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        # Merge
        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        merge_result, _ = merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        assert set(ref_result.keys()) == set(merge_result.keys())
        for fid in ref_result:
            for pkey in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[pkey]
                        - merge_result[fid].percentiles[pkey]
                    )
                    < 1e-5
                ), f"Feature {fid} {pkey} mismatch"


class TestOptimizedFullPipeline:
    """End-to-end: optimized streaming pipeline vs eager reference."""

    def test_full_stats_match(self, tmp_path):
        """Full compute_statistics output must match between eager and streaming."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        ref_out = tmp_path / "ref_stats"
        ref_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(ref_out),
            streaming=False,
        )

        stream_out = tmp_path / "stream_stats"
        stream_stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(stream_out),
            streaming=True,
        )

        # Corpus stats must match exactly
        assert ref_stats.corpus.n_docs == stream_stats.corpus.n_docs
        assert ref_stats.corpus.n_tokens_gpt2 == stream_stats.corpus.n_tokens_gpt2
        assert (
            ref_stats.corpus.n_shards_processed
            == stream_stats.corpus.n_shards_processed
        )

        # Feature stats must match exactly
        assert ref_stats.features.n_segments == stream_stats.features.n_segments
        assert (
            ref_stats.features.n_features_seen == stream_stats.features.n_features_seen
        )

        # Per-feature percentiles must match closely
        for fid in ref_stats.per_feature_stats:
            ref_f = ref_stats.per_feature_stats[fid]
            stream_f = stream_stats.per_feature_stats[fid]
            assert ref_f.count == stream_f.count, f"Feature {fid} count"
            assert abs(ref_f.mean_mag - stream_f.mean_mag) < 1e-4, f"Feature {fid} mean"
            assert abs(ref_f.median_mag - stream_f.median_mag) < 1e-4, (
                f"Feature {fid} median"
            )
            for pkey in ref_f.percentiles:
                assert (
                    abs(ref_f.percentiles[pkey] - stream_f.percentiles[pkey]) < 1e-4
                ), f"Feature {fid} {pkey}"

    def test_temp_files_cleaned_up(self, tmp_path):
        """Temp sorted files must be cleaned up after compute_statistics."""
        _write_synthetic_shards(tmp_path, n_shards=2)

        import glob

        # Count temp dirs before
        before = set(glob.glob("/tmp/stats_sorted_*"))

        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats"),
            streaming=True,
        )

        # No new temp dirs should remain
        after = set(glob.glob("/tmp/stats_sorted_*"))
        new_dirs = after - before
        assert len(new_dirs) == 0, f"Temp dirs not cleaned up: {new_dirs}"


# ---------------------------------------------------------------------------
# GPU Merge Pass 2 tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
class TestGPUMergePerFeaturePercentiles:
    """Tests for gpu_merge_per_feature_percentiles vs CPU merge reference."""

    def test_matches_cpu_merge(self, tmp_path):
        """GPU merge percentiles must match CPU merge percentiles exactly."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        cpu_result, cpu_edges = merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        # Re-create sorted shards (memmaps may have been consumed)
        p1b, sorted_shards_b, mag_sub_b, mag_min_b = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp_gpu",
        )

        gpu_result, gpu_edges = gpu_merge_per_feature_percentiles(
            sorted_shards_b,
            p1b.n_sae_features,
            p1b.acc.occurrence,
            mag_sub_b,
            mag_min_b,
        )

        assert set(cpu_result.keys()) == set(gpu_result.keys())
        for fid in cpu_result:
            cpu_fp = cpu_result[fid]
            gpu_fp = gpu_result[fid]
            assert abs(cpu_fp.median - gpu_fp.median) < 1e-5, (
                f"Feature {fid} median: cpu={cpu_fp.median}, gpu={gpu_fp.median}"
            )
            for pkey in cpu_fp.percentiles:
                assert (
                    abs(cpu_fp.percentiles[pkey] - gpu_fp.percentiles[pkey]) < 1e-5
                ), (
                    f"Feature {fid} {pkey}: "
                    f"cpu={cpu_fp.percentiles[pkey]}, gpu={gpu_fp.percentiles[pkey]}"
                )

    def test_matches_eager_reference(self, tmp_path):
        """GPU merge percentiles must match eager compute_per_feature_percentiles."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        gpu_result, _ = gpu_merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        assert set(ref_result.keys()) == set(gpu_result.keys())
        for fid in ref_result:
            ref_fp = ref_result[fid]
            gpu_fp = gpu_result[fid]
            assert abs(ref_fp.median - gpu_fp.median) < 1e-5, (
                f"Feature {fid} median: ref={ref_fp.median}, gpu={gpu_fp.median}"
            )
            for pkey in ref_fp.percentiles:
                assert (
                    abs(ref_fp.percentiles[pkey] - gpu_fp.percentiles[pkey]) < 1e-5
                ), (
                    f"Feature {fid} {pkey}: "
                    f"ref={ref_fp.percentiles[pkey]}, gpu={gpu_fp.percentiles[pkey]}"
                )

    def test_single_shard(self, tmp_path):
        """Works correctly with a single shard on GPU."""
        _write_synthetic_shards(tmp_path, n_shards=1)

        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        gpu_result, _ = gpu_merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        for fid in ref_result:
            for pkey in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[pkey]
                        - gpu_result[fid].percentiles[pkey]
                    )
                    < 1e-5
                )

    def test_many_features(self, tmp_path):
        """Correctness with 100 features and multiple shards on GPU."""
        n_features = 100
        n_shards = 3
        for shard_idx in range(n_shards):
            n_segments = 10
            feat_ids = []
            feat_mags = []
            offsets = [0]
            for seg in range(n_segments):
                t.manual_seed(shard_idx * 1000 + seg)
                n_active = t.randint(5, 20, (1,)).item()
                seg_ids = t.randint(0, n_features, (n_active,)).tolist()  # ty: ignore[no-matching-overload]
                seg_mags = (t.rand(n_active) * 10 + 0.1).tolist()  # ty: ignore[no-matching-overload]
                feat_ids.extend(seg_ids)
                feat_mags.extend(seg_mags)
                offsets.append(offsets[-1] + n_active)

            doc_ids = t.tensor([0] * 5 + [1] * 5, dtype=t.int32)
            segment_indices, segment_positions = _per_doc_segment_geometry(doc_ids, 128)
            shard = {
                "feature_ids": t.tensor(feat_ids, dtype=t.int32),
                "feature_mags": t.tensor(feat_mags, dtype=t.float32),
                "offsets": t.tensor(offsets, dtype=t.int64),
                "n_segments": n_segments,
                "n_docs": 2,
                "doc_ids": doc_ids,
                "doc_token_counts": t.tensor([640, 640], dtype=t.int32),
                "segment_indices": segment_indices,
                "segment_positions": segment_positions,
                "segment_mode": "fixed_tokens",
                "segment_size": 128,
                "context_window": 1024,
                "pooling": "mean_pool",
                "layer": 8,
            }
            save_features(shard, tmp_path / f"shard_{shard_idx}_features")

        loaded = load_all_shards(str(tmp_path))
        ref_result, _ = compute_per_feature_percentiles(
            loaded.feature_ids,
            loaded.feature_mags,
        )

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )
        gpu_result, _ = gpu_merge_per_feature_percentiles(
            sorted_shards,
            p1.n_sae_features,
            p1.acc.occurrence,
            mag_sub,
            mag_min,
        )

        assert set(ref_result.keys()) == set(gpu_result.keys())
        for fid in ref_result:
            for pkey in ref_result[fid].percentiles:
                assert (
                    abs(
                        ref_result[fid].percentiles[pkey]
                        - gpu_result[fid].percentiles[pkey]
                    )
                    < 1e-5
                ), f"Feature {fid} {pkey} mismatch"

    def test_occurrence_assertion_fires(self, tmp_path):
        """If temp data is corrupted, the occurrence assertion should fire."""
        _write_synthetic_shards(tmp_path, n_shards=1)

        p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "sorted_tmp",
        )

        corrupted_occurrence = p1.acc.occurrence.clone()
        corrupted_occurrence[0] += 999

        with pytest.raises(AssertionError, match="merge count"):
            gpu_merge_per_feature_percentiles(
                sorted_shards,
                p1.n_sae_features,
                corrupted_occurrence,
                mag_sub,
                mag_min,
            )


class TestSanityCheckPercentiles:
    """Tests for the post-compute percentile sanity check."""

    @staticmethod
    def _write_large_shards(tmp_dir: Path, n_shards: int = 2, n_features: int = 5):
        """Write shards with enough activations per feature for meaningful checks."""
        t.manual_seed(123)
        for shard_idx in range(n_shards):
            n_segments = 200
            feat_ids = []
            feat_mags = []
            offsets = [0]
            for _ in range(n_segments):
                # Each segment activates all features
                seg_ids = list(range(n_features))
                seg_mags = [
                    abs(float(t.randn(1).item()) * (fid + 1) + 1.0) for fid in seg_ids
                ]
                feat_ids.extend(seg_ids)
                feat_mags.extend(seg_mags)
                offsets.append(offsets[-1] + len(seg_ids))

            doc_ids = t.tensor(
                [i // 50 for i in range(n_segments)], dtype=t.int32
            )
            segment_indices, segment_positions = _per_doc_segment_geometry(doc_ids, 128)
            shard = {
                "feature_ids": t.tensor(feat_ids, dtype=t.int32),
                "feature_mags": t.tensor(feat_mags, dtype=t.float32),
                "offsets": t.tensor(offsets, dtype=t.int64),
                "n_segments": n_segments,
                "n_docs": 4,
                "doc_ids": doc_ids,
                "doc_token_counts": t.tensor([6400] * 4, dtype=t.int32),
                "segment_indices": segment_indices,
                "segment_positions": segment_positions,
                "segment_mode": "fixed_tokens",
                "segment_size": 128,
                "context_window": 6400,
                "pooling": "mean_pool",
                "layer": 8,
            }
            save_features(shard, tmp_dir / f"shard_{shard_idx}_features")

    def test_passes_on_correct_percentiles(self, tmp_path):
        """Sanity check passes when percentiles are computed correctly."""

        self._write_large_shards(tmp_path, n_shards=2, n_features=5)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )
        # The sanity check runs automatically inside compute_statistics,
        # but also verify it returns True when called directly.
        assert sanity_check_percentiles(
            str(tmp_path),
            stats.per_feature_stats,
            n_features_to_check=5,
            tolerance=0.02,
        )

    def test_fails_on_wrong_percentiles(self, tmp_path):
        """Sanity check detects deliberately corrupted percentiles."""
        import dataclasses

        self._write_large_shards(tmp_path, n_shards=2, n_features=5)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )

        # Corrupt the percentiles: set all values to 0 (way too low)
        corrupted = {}
        for fid, stat in stats.per_feature_stats.items():
            corrupted[fid] = dataclasses.replace(
                stat,
                percentiles=dict.fromkeys(stat.percentiles, 0.0),
            )

        assert not sanity_check_percentiles(
            str(tmp_path),
            corrupted,
            n_features_to_check=5,
            tolerance=0.02,
        )


class TestGpuOptimizedPass1:
    """Verify GPU-accelerated optimized_pass1 matches CPU results."""

    @staticmethod
    def _requires_cuda():
        if not t.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_gpu_pass1_accumulators_match_cpu(self, tmp_path):
        """GPU and CPU optimized_pass1 produce identical accumulators."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=3)

        # CPU pass
        cpu_tmp = tempfile.mkdtemp()
        try:
            cpu_p1, cpu_shards, cpu_sub, cpu_min = optimized_pass1(
                str(tmp_path),
                cpu_tmp,
                device="cpu",
            )
        finally:
            shutil.rmtree(cpu_tmp)

        # GPU pass
        gpu_tmp = tempfile.mkdtemp()
        try:
            gpu_p1, gpu_shards, gpu_sub, gpu_min = optimized_pass1(
                str(tmp_path),
                gpu_tmp,
                device="cuda",
            )
        finally:
            shutil.rmtree(gpu_tmp)

        # Accumulators must match exactly (both use float64)
        assert t.equal(cpu_p1.acc.occurrence, gpu_p1.acc.occurrence)
        assert t.allclose(cpu_p1.acc.mag_sum, gpu_p1.acc.mag_sum, atol=1e-10)
        assert t.allclose(cpu_p1.acc.mag_sum_sq, gpu_p1.acc.mag_sum_sq, atol=1e-10)
        assert t.allclose(cpu_p1.acc.mag_min, gpu_p1.acc.mag_min, atol=1e-10)
        assert t.allclose(cpu_p1.acc.mag_max, gpu_p1.acc.mag_max, atol=1e-10)

        # Sorted shard counts must match
        for cs, gs in zip(cpu_shards, gpu_shards, strict=True):
            assert cs.n_activations == gs.n_activations

    def test_gpu_pass1_sorted_output_matches_cpu(self, tmp_path):
        """GPU-sorted temp files contain same data as CPU-sorted ones."""
        self._requires_cuda()
        _write_synthetic_shards(tmp_path, n_shards=2)

        cpu_tmp = tempfile.mkdtemp()
        gpu_tmp = tempfile.mkdtemp()
        try:
            cpu_p1, cpu_shards, _, _ = optimized_pass1(
                str(tmp_path),
                cpu_tmp,
                device="cpu",
            )
            gpu_p1, gpu_shards, _, _ = optimized_pass1(
                str(tmp_path),
                gpu_tmp,
                device="cuda",
            )
            # Compare sorted magnitudes in temp files
            for cs, gs in zip(cpu_shards, gpu_shards, strict=True):
                if cs.mags_path is None:
                    continue
                cpu_mags = np.fromfile(str(cs.mags_path), dtype=np.float32)
                gpu_mags = np.fromfile(str(gs.mags_path), dtype=np.float32)
                assert len(cpu_mags) == len(gpu_mags)
                # Per-feature counts must match.
                cpu_counts = cs.per_feature_counts
                gpu_counts = gs.per_feature_counts
                max_len = max(len(cpu_counts), len(gpu_counts))
                cpu_padded = t.cat(
                    [
                        cpu_counts,
                        t.zeros(max_len - len(cpu_counts), dtype=t.int64),
                    ]
                )
                gpu_padded = t.cat(
                    [
                        gpu_counts,
                        t.zeros(max_len - len(gpu_counts), dtype=t.int64),
                    ]
                )
                assert t.equal(cpu_padded, gpu_padded)
        finally:
            shutil.rmtree(cpu_tmp)
            shutil.rmtree(gpu_tmp)


class TestVectorizedPass2:
    """Verify vectorized Pass 2 matches merge-based CPU Pass 2."""

    def test_vectorized_matches_merge(self, tmp_path):
        """vectorized_pass2_percentiles matches merge_per_feature_percentiles."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            # Reference: CPU merge
            ref_result, ref_edges = merge_per_feature_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            # New: vectorized
            vec_result, vec_edges = vectorized_pass2_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            # All features must be present
            assert set(ref_result.keys()) == set(vec_result.keys())
            # Percentiles must match within tolerance
            max_diff = 0.0
            for fid in ref_result:
                for pk in ref_result[fid].percentiles:
                    ref_val = ref_result[fid].percentiles[pk]
                    vec_val = vec_result[fid].percentiles[pk]
                    max_diff = max(max_diff, abs(ref_val - vec_val))
                assert abs(ref_result[fid].median - vec_result[fid].median) < 1e-5
            assert max_diff < 1e-5, f"Max percentile diff: {max_diff}"
        finally:
            shutil.rmtree(tmp)

    def test_vectorized_histograms_match(self, tmp_path):
        """Histograms from vectorized Pass 2 match merge-based."""
        _write_synthetic_shards(tmp_path, n_shards=2)

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            ref_result, _ = merge_per_feature_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            vec_result, _ = vectorized_pass2_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            for fid in ref_result:
                if ref_result[fid].histogram is not None:
                    assert vec_result[fid].histogram is not None
                    # Histogram counts must match exactly
                    assert ref_result[fid].histogram == vec_result[fid].histogram
        finally:
            shutil.rmtree(tmp)

    def test_vectorized_single_shard(self, tmp_path):
        """Works with a single shard."""
        _write_synthetic_shards(tmp_path, n_shards=1)

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            vec_result, vec_edges = vectorized_pass2_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            assert len(vec_result) > 0
        finally:
            shutil.rmtree(tmp)

    def test_vectorized_device_cpu_matches_reference(self, tmp_path):
        """Forcing device='cpu' must produce the same percentiles as default."""
        _write_synthetic_shards(tmp_path, n_shards=3)

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            ref_result, ref_edges = merge_per_feature_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            cpu_result, cpu_edges = vectorized_pass2_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
                device="cpu",  # ty: ignore[invalid-argument-type]
            )
            assert set(ref_result.keys()) == set(cpu_result.keys())
            for fid in ref_result:
                for pk in ref_result[fid].percentiles:
                    ref_val = ref_result[fid].percentiles[pk]
                    cpu_val = cpu_result[fid].percentiles[pk]
                    assert abs(ref_val - cpu_val) < 1e-5
                assert abs(ref_result[fid].median - cpu_result[fid].median) < 1e-5
                assert ref_result[fid].histogram == cpu_result[fid].histogram
            assert ref_edges == cpu_edges
        finally:
            shutil.rmtree(tmp)

    def test_vectorized_device_cpu_runs_even_when_cuda_available(self, tmp_path):
        """device='cpu' must not touch CUDA, even if CUDA is available.

        Patches torch.cuda.mem_get_info to raise so that if the code accidentally
        takes the GPU branch, the test fails loudly instead of silently using GPU.
        """
        _write_synthetic_shards(tmp_path, n_shards=2)

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            orig_mem_get_info = t.cuda.mem_get_info

            def boom(*args, **kwargs):
                raise RuntimeError(
                    "device='cpu' should not call cuda.mem_get_info"
                )

            t.cuda.mem_get_info = boom  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
            try:
                result, _ = vectorized_pass2_percentiles(
                    sorted_shards,
                    p1.n_sae_features,
                    p1.acc.occurrence,
                    mag_sub,
                    mag_min,
                    device="cpu",  # ty: ignore[invalid-argument-type]
                )
            finally:
                t.cuda.mem_get_info = orig_mem_get_info  # type: ignore[assignment]
            assert len(result) > 0
        finally:
            shutil.rmtree(tmp)

    def test_vectorized_empty_shards(self, tmp_path):
        """Empty shards produce empty result."""
        # Create a shard with no activations
        shard = {
            "feature_ids": t.tensor([], dtype=t.int32),
            "feature_mags": t.tensor([], dtype=t.float32),
            "offsets": t.tensor([0], dtype=t.int64),
            "n_segments": 0,
            "n_docs": 0,
            "doc_ids": t.tensor([], dtype=t.int32),
            "doc_token_counts": t.tensor([], dtype=t.int32),
            "segment_indices": t.tensor([], dtype=t.int32),
            "segment_positions": t.tensor([], dtype=t.int32),
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_path / "shard_0_features")

        tmp = tempfile.mkdtemp()
        try:
            p1, sorted_shards, mag_sub, mag_min = optimized_pass1(
                str(tmp_path),
                tmp,
                device="cpu",
            )
            vec_result, vec_edges = vectorized_pass2_percentiles(
                sorted_shards,
                p1.n_sae_features,
                p1.acc.occurrence,
                mag_sub,
                mag_min,
            )
            assert vec_result == {}
            assert vec_edges is None
        finally:
            shutil.rmtree(tmp)


class TestEndToEndOptimized:
    """End-to-end: optimized streaming matches CPU reference."""

    def test_streaming_gpu_matches_non_streaming(self, tmp_path):
        """Streaming GPU pipeline produces same stats as non-streaming CPU."""
        _write_synthetic_shards(tmp_path, n_shards=3)
        out_stream = tmp_path / "stream_stats"
        out_eager = tmp_path / "eager_stats"

        stats_stream = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_stream),
            streaming=True,
            device="cuda",
        )
        stats_eager = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out_eager),
            streaming=False,
            device="cpu",
        )

        # Corpus stats must match exactly
        assert stats_stream.corpus.n_docs == stats_eager.corpus.n_docs
        assert stats_stream.corpus.n_tokens_gpt2 == stats_eager.corpus.n_tokens_gpt2
        assert stats_stream.features.n_segments == stats_eager.features.n_segments

        # Per-feature counts must match
        for fid in stats_eager.per_feature_stats:
            assert fid in stats_stream.per_feature_stats
            assert stats_stream.per_feature_stats[fid].count == (
                stats_eager.per_feature_stats[fid].count
            )

        # Per-feature percentiles within tolerance
        max_diff = 0.0
        for fid in stats_eager.per_feature_stats:
            for pk in stats_eager.per_feature_stats[fid].percentiles:
                ref = stats_eager.per_feature_stats[fid].percentiles[pk]
                new = stats_stream.per_feature_stats[fid].percentiles[pk]
                max_diff = max(max_diff, abs(ref - new))
        assert max_diff < 1e-4, f"Max percentile diff: {max_diff}"


# ---------------------------------------------------------------------------
# Tests for bracket-test sanity check and quantized distributions
# ---------------------------------------------------------------------------


def _write_quantized_shards(tmp_dir: Path, n_shards: int = 3) -> None:
    """Write shards where one feature has a heavily quantized distribution.

    Feature 0: smooth distribution (normal * 10 + 20)
    Feature 1: ~70% of values at exactly 16.125, rest spread around it
    Feature 2: smooth distribution (uniform 1-5)
    """
    rng = t.Generator().manual_seed(777)
    for shard_idx in range(n_shards):
        n_segments = 200
        feat_ids = []
        feat_mags = []
        offsets = [0]
        for _ in range(n_segments):
            # Feature 0: smooth normal
            mag0 = abs(float(t.randn(1, generator=rng).item()) * 10 + 20)
            # Feature 1: quantized — 70% at 16.125, 30% from uniform(1, 30)
            if t.rand(1, generator=rng).item() < 0.7:
                mag1 = 16.125
            else:
                mag1 = float(t.rand(1, generator=rng).item() * 29 + 1)
            # Feature 2: smooth uniform
            mag2 = float(t.rand(1, generator=rng).item() * 4 + 1)

            feat_ids.extend([0, 1, 2])
            feat_mags.extend([mag0, mag1, mag2])
            offsets.append(offsets[-1] + 3)

        doc_ids = t.tensor([i // 50 for i in range(n_segments)], dtype=t.int32)
        segment_indices, segment_positions = _per_doc_segment_geometry(doc_ids, 128)
        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": n_segments,
            "n_docs": 4,
            "doc_ids": doc_ids,
            "doc_token_counts": t.tensor([6400] * 4, dtype=t.int32),
            "segment_indices": segment_indices,
            "segment_positions": segment_positions,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 6400,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


class TestSanityCheckBracketTest:
    """Tests for bracket-test sanity check on quantized distributions."""

    def test_passes_on_quantized_distribution(self, tmp_path):
        """Sanity check passes when features have concentrated/quantized values.

        Feature 1 has ~70% of values at exactly 16.125, causing many
        percentiles (p30-p70) to correctly map to the same value.
        The bracket test handles this correctly.
        """
        _write_quantized_shards(tmp_path, n_shards=3)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )

        # The internal sanity check should have passed. Verify explicitly
        # with all 3 features checked.
        assert sanity_check_percentiles(
            str(tmp_path),
            stats.per_feature_stats,
            n_features_to_check=3,
            n_shards_to_check=3,
            tolerance=0.02,
        )

    def test_fails_on_shifted_percentiles(self, tmp_path):
        """Bracket test catches percentiles shifted far from truth."""
        import dataclasses

        _write_quantized_shards(tmp_path, n_shards=3)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )

        # Corrupt: shift every percentile value up by 1000
        corrupted = {}
        for fid, stat in stats.per_feature_stats.items():
            corrupted[fid] = dataclasses.replace(
                stat,
                percentiles={k: v + 1000.0 for k, v in stat.percentiles.items()},
            )

        assert not sanity_check_percentiles(
            str(tmp_path),
            corrupted,
            n_features_to_check=3,
            n_shards_to_check=3,
            tolerance=0.02,
        )

    def test_fails_on_zeroed_percentiles(self, tmp_path):
        """Bracket test catches percentiles set to zero."""
        import dataclasses

        _write_quantized_shards(tmp_path, n_shards=3)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )

        corrupted = {}
        for fid, stat in stats.per_feature_stats.items():
            corrupted[fid] = dataclasses.replace(
                stat,
                percentiles=dict.fromkeys(stat.percentiles, 0.0),
            )

        assert not sanity_check_percentiles(
            str(tmp_path),
            corrupted,
            n_features_to_check=3,
            n_shards_to_check=3,
            tolerance=0.02,
        )


class TestVectorizedPass2Streaming:
    """Verify streaming Pass 2 (memmap reads per-batch) is correct."""

    def test_quantized_data_bracket_validation(self, tmp_path):
        """Streaming Pass 2 percentiles satisfy bracket test on quantized data.

        For each computed percentile p_X = v, verifies:
            fraction(data < v) <= X/100 <= fraction(data <= v)
        This is the correct validation for quantized distributions where
        many percentile ranks map to the same value.
        """
        _write_quantized_shards(tmp_path, n_shards=3)

        stats = compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(tmp_path / "stats.safetensors"),
            streaming=True,
            device="cpu",
        )

        # Collect all magnitudes per feature from raw shards
        paths = glob_features(str(tmp_path))
        raw_mags: dict[int, list[float]] = {}
        for p in paths:
            data = load_features(p)
            fids = data["feature_ids"]
            fmags = data["feature_mags"]
            for fid_val in range(3):
                mask = fids == fid_val
                if mask.any():
                    raw_mags.setdefault(fid_val, []).extend(fmags[mask].tolist())

        tolerance = 0.02
        for fid_val in range(3):
            fid_key = str(fid_val)
            if fid_key not in stats.per_feature_stats:
                continue
            stat = stats.per_feature_stats[fid_key]
            mags_arr = np.array(raw_mags[fid_val], dtype=np.float64)
            n = len(mags_arr)

            for pkey, pval in stat.percentiles.items():
                expected_frac = int(pkey[1:]) / 100.0
                frac_below = float(np.sum(mags_arr < pval - 1e-7)) / n
                frac_at_or_below = float(np.sum(mags_arr <= pval + 1e-7)) / n

                assert expected_frac >= frac_below - tolerance, (
                    f"Feature {fid_val} {pkey}: value {pval:.6f} too high — "
                    f"expected ~{expected_frac:.0%} but {frac_below:.1%} "
                    f"are strictly below"
                )
                assert expected_frac <= frac_at_or_below + tolerance, (
                    f"Feature {fid_val} {pkey}: value {pval:.6f} too low — "
                    f"expected ~{expected_frac:.0%} but only {frac_at_or_below:.1%} "
                    f"are at or below"
                )


def _corrupt_empty_segment_shard_data() -> dict:
    """Base-shard data dict with an empty middle segment (issue #522).

    ``offsets=[0,2,2,4]`` → 3 segments (so the legitimate ``n_segments==0``
    whole-shard skip does NOT apply), with segment 1 empty. CSR + per-segment
    boundary tensors are internally consistent so ``load_features`` reads it
    back cleanly; the base-reading guard fires right after the load.
    """
    return {
        "offsets": t.tensor([0, 2, 2, 4], dtype=t.int64),  # segment 1 empty
        "feature_ids": t.tensor([1, 5, 2, 7], dtype=t.int32),
        "feature_mags": t.tensor([0.1, 0.2, 0.3, 0.4], dtype=t.float32),
        "doc_ids": t.zeros(3, dtype=t.int32),
        "segment_indices": t.tensor([0, 1, 2], dtype=t.int32),
        "segment_positions": t.tensor([0, 8, 16], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 8, 16], dtype=t.int32),
        "segment_char_ends": t.tensor([8, 16, 24], dtype=t.int32),
        "doc_token_counts": t.tensor([24], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 3,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 8,
    }


class TestComputeStatisticsEmptySegmentGuard:
    """Issue #522 site-f guard: compute_statistics reads base shards directly
    via load_features (NOT via FeatureIndex), so each raw-base-shard load point
    rejects a shard with any empty segment (offsets[i]==offsets[i+1]).

    The guard sits at the *raw* load, BEFORE percentile filtering (which
    legitimately empties segments), so it must NOT live inside
    accumulate_segment_stats — that also runs on filtered data."""

    def test_load_all_shards_raises_on_empty_segment(self, tmp_path):
        save_raw_shard(tmp_path, "shard_00000", _corrupt_empty_segment_shard_data())
        with pytest.raises(EmptyExtractionSegmentError, match=r"segment 1\b"):
            load_all_shards(str(tmp_path))

    def test_stream_pass1_raises_on_empty_segment(self, tmp_path):
        save_raw_shard(tmp_path, "shard_00000", _corrupt_empty_segment_shard_data())
        with pytest.raises(EmptyExtractionSegmentError, match=r"segment 1\b"):
            stream_pass1(str(tmp_path))

    def test_optimized_pass1_raises_on_empty_segment(self, tmp_path):
        save_raw_shard(tmp_path, "shard_00000", _corrupt_empty_segment_shard_data())
        with pytest.raises(EmptyExtractionSegmentError, match=r"segment 1\b"):
            optimized_pass1(str(tmp_path), tmp_path / "sorted_tmp")

    def test_accumulate_segment_stats_allows_post_filter_empty_segment(self):
        """The guard must NOT live in accumulate_segment_stats: percentile
        filtering legitimately empties segments, and that filtered data still
        flows through accumulate_segment_stats in the eager path."""
        acc = SegmentStatsAccumulator(
            features_per_segment=[],
            tokens_per_doc=[],
            segment_lengths=[],
            features_per_token_ratios=[],
        )
        data = {
            "offsets": t.tensor([0, 2, 2, 4], dtype=t.int64),  # segment 1 empty
            "doc_token_counts": t.tensor([24], dtype=t.int32),
            "doc_ids": t.zeros(3, dtype=t.int32),
            "n_segments": 3,
            "context_window": 1024,
            "segment_positions": t.tensor([0, 8, 16], dtype=t.int32),
            "segment_size": 8,
            "segment_mode": "fixed_tokens",
        }
        # Must NOT raise — empty segments are valid post-filter.
        accumulate_segment_stats(acc, data)
        assert acc.features_per_segment == [2, 0, 2]
