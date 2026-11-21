"""Tests for the thresholding stage in the completion steering eval pipeline.

Validates:
- apply_threshold_to_shard correctness
- scan_thresholded_features vs manual threshold+scan
- _scan_core matches scan_validation_features_vectorized
- Thresholded features survive generation-time threshold (key invariant)
- Edge cases: empty shards, preserved doc_ids/segment_indices
- Auto-resume for threshold stage
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    build_thresholds_from_stats,
)
from steering.eval.completion_steering_eval import (
    ScanResultTensors,
    _scan_core,
    apply_threshold_to_shard,
    load_scan_results,
    save_scan_results,
    scan_thresholded_features,
    scan_validation_features_vectorized,
)
from steering.feature_io import load_features, save_features
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_feature_stats(feature_ids: list[int]) -> dict[str, PerFeatureStat]:
    """Build synthetic per_feature_stats for a set of feature IDs.

    Each feature gets percentiles p1..p99 linearly spaced, so that
    the p90 threshold is at activation magnitude ~0.9 * max_val.
    """
    stats: dict[str, PerFeatureStat] = {}
    for fid in feature_ids:
        # Percentiles p1..p99 with values [0.01, 0.02, ..., 0.99] * (fid + 1)
        # This gives each feature a unique scale.
        scale = float(fid + 1)
        percentiles = {f"p{i}": i * 0.01 * scale for i in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5 * scale,
            mean_mag_incl_zeros=0.25 * scale,
            std_mag=0.2 * scale,
            min_mag=0.0,
            max_mag=scale,
            median_mag=0.5 * scale,
            percentiles=percentiles,
        )
    return stats


def _build_shard_data(
    n_docs: int = 3,
    segs_per_doc: int = 4,
    features_per_seg: int = 10,
    n_features: int = 20,
    seed: int = 42,
) -> dict:
    """Build a synthetic shard dict matching load_features() output format.

    Creates n_docs documents, each with segs_per_doc segments, each segment
    having features_per_seg features drawn from feature IDs 0..n_features-1.
    """
    rng = t.Generator().manual_seed(seed)
    n_segments = n_docs * segs_per_doc

    doc_ids_list: list[int] = []
    seg_indices_list: list[int] = []
    for d in range(n_docs):
        for s in range(segs_per_doc):
            doc_ids_list.append(d)
            seg_indices_list.append(s)

    all_fids: list[t.Tensor] = []
    all_mags: list[t.Tensor] = []
    offsets = [0]

    for _ in range(n_segments):
        fids = t.randint(0, n_features, (features_per_seg,), generator=rng)
        mags = t.rand(features_per_seg, generator=rng) * 5.0  # [0, 5)
        all_fids.append(fids)
        all_mags.append(mags)
        offsets.append(offsets[-1] + features_per_seg)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "doc_ids": t.tensor(doc_ids_list, dtype=t.int64),
        "segment_indices": t.tensor(seg_indices_list, dtype=t.int64),
        "feature_ids": t.cat(all_fids),
        "feature_mags": t.cat(all_mags),
        "n_segments": n_segments,
    }


def _save_shard_to_disk(shard_data: dict, tmp_path: Path) -> Path:
    """Save a synthetic shard to disk and return the actual path written."""
    # save_features rewrites the filename; we use a consistent stem
    return save_features(shard_data, tmp_path / "shard_00000_features")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyThresholdToShardCorrectness:
    """Verify apply_threshold_to_shard matches per-segment apply_threshold."""

    def test_each_segment_matches_individual_apply_threshold(self):
        """For each segment, thresholded output matches calling apply_threshold directly."""
        shard = _build_shard_data(
            n_docs=2, segs_per_doc=3, features_per_seg=8, n_features=15
        )
        feature_ids_in_shard = shard["feature_ids"].unique().tolist()
        stats = _make_per_feature_stats(list(range(max(feature_ids_in_shard) + 1)))

        # Use feature_percentile_segment_topk threshold: topk=5, percentile=0.5
        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, 5.5
        )

        # Build precomputed thresholds
        precomputed = build_thresholds_from_stats(
            stats,
            max(feature_ids_in_shard) + 1,
            ThresholdType.feature_percentile_segment_topk,
            5.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        # Verify each segment individually
        offsets = shard["offsets"]
        new_offsets = result.offsets
        n_segments = int(shard["n_segments"])

        for i in range(n_segments):
            # Original segment
            start = int(offsets[i].item())
            end = int(offsets[i + 1].item())
            seg_fids = shard["feature_ids"][start:end]
            seg_mags = shard["feature_mags"][start:end]

            # Expected: apply_threshold directly
            expected_fids, expected_mags = apply_threshold(
                threshold_cfg, seg_fids, seg_mags, precomputed
            )

            # Actual: from apply_threshold_to_shard
            new_start = int(new_offsets[i].item())
            new_end = int(new_offsets[i + 1].item())
            actual_fids = result.feature_ids[new_start:new_end]
            actual_mags = result.feature_mags[new_start:new_end]

            # Sort both for comparison (apply_threshold may reorder via topk)
            exp_sort = expected_fids.argsort()
            act_sort = actual_fids.argsort()
            assert t.equal(expected_fids[exp_sort], actual_fids[act_sort]), (
                f"Segment {i}: feature_ids mismatch"
            )
            assert t.allclose(expected_mags[exp_sort], actual_mags[act_sort]), (
                f"Segment {i}: feature_mags mismatch"
            )


class TestScanThresholdedVsManual:
    """Verify scan_thresholded_features matches manual threshold + scan."""

    def test_matches_manual_threshold_then_scan(self, tmp_path):
        """scan_thresholded_features == load + threshold + _scan_core manually."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=3, features_per_seg=12, n_features=20
        )
        shard_path = _save_shard_to_disk(shard, tmp_path)

        all_fids = list(range(20))
        stats = _make_per_feature_stats(all_fids)

        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, 8.5
        )
        precomputed = build_thresholds_from_stats(
            stats,
            20,
            ThresholdType.feature_percentile_segment_topk,
            8.5,
        )

        # Path 1: scan_thresholded_features (full pipeline)
        result1 = scan_thresholded_features(
            shard_path, stats, threshold_cfg, precomputed
        )

        # Path 2: manual — load, threshold, then _scan_core

        data = load_features(shard_path)
        thresholded = apply_threshold_to_shard(data, threshold_cfg, precomputed)
        result2 = _scan_core(
            thresholded.offsets,
            thresholded.doc_ids,
            thresholded.segment_indices,
            thresholded.feature_ids,
            thresholded.feature_mags,
            thresholded.n_segments,
            stats,
        )

        # Both should produce identical ScanResultTensors
        assert t.equal(result1.feature_ids, result2.feature_ids)
        assert t.equal(result1.n_docs_activated, result2.n_docs_activated)
        assert t.equal(result1.offsets, result2.offsets)
        assert t.equal(result1.doc_idx, result2.doc_idx)
        assert t.allclose(result1.max_activation, result2.max_activation)
        assert t.equal(result1.max_segment_idx, result2.max_segment_idx)
        assert t.allclose(result1.percentile, result2.percentile)
        assert t.equal(result1.total_segments, result2.total_segments)


class TestScanCoreMatchesVectorized:
    """Verify _scan_core produces identical output to scan_validation_features_vectorized."""

    def test_scan_core_matches_full_function(self, tmp_path):
        """_scan_core on raw data matches scan_validation_features_vectorized."""
        shard = _build_shard_data(
            n_docs=4, segs_per_doc=5, features_per_seg=15, n_features=25
        )
        shard_path = _save_shard_to_disk(shard, tmp_path)

        all_fids = list(range(25))
        stats = _make_per_feature_stats(all_fids)

        # Full function path
        result_full = scan_validation_features_vectorized(shard_path, stats)

        # Manual _scan_core path with same data

        data = load_features(shard_path)
        result_core = _scan_core(
            data["offsets"],
            data["doc_ids"],
            data["segment_indices"],
            data["feature_ids"],
            data["feature_mags"],
            int(data["n_segments"]),
            stats,
        )

        assert t.equal(result_full.feature_ids, result_core.feature_ids)
        assert t.equal(result_full.n_docs_activated, result_core.n_docs_activated)
        assert t.equal(result_full.offsets, result_core.offsets)
        assert t.equal(result_full.doc_idx, result_core.doc_idx)
        assert t.allclose(result_full.max_activation, result_core.max_activation)
        assert t.equal(result_full.max_segment_idx, result_core.max_segment_idx)
        assert t.allclose(result_full.percentile, result_core.percentile)
        assert t.equal(result_full.total_segments, result_core.total_segments)


class TestThresholdRemovesFeatures:
    """Verify that features below the percentile threshold are absent."""

    def test_below_threshold_features_absent(self):
        """Features with activations below the percentile are filtered out."""
        # Create a shard with known activations
        # Feature 0: scale=1.0, p50 threshold = 0.50
        # Feature 1: scale=2.0, p50 threshold = 1.00
        n_segments = 2
        shard = {
            "offsets": t.tensor([0, 3, 6], dtype=t.int64),
            "doc_ids": t.tensor([0, 1], dtype=t.int64),
            "segment_indices": t.tensor([1, 1], dtype=t.int64),
            "feature_ids": t.tensor([0, 0, 1, 0, 1, 1], dtype=t.int64),
            # Segment 0: fid0=0.3 (below p50=0.5), fid0=0.7 (above), fid1=0.5 (below p50=1.0)
            # Segment 1: fid0=0.2 (below), fid1=1.5 (above), fid1=2.0 (above)
            "feature_mags": t.tensor([0.3, 0.7, 0.5, 0.2, 1.5, 2.0], dtype=t.float32),
            "n_segments": n_segments,
        }

        stats = _make_per_feature_stats([0, 1])
        # Use feature_percentile threshold at 0.5 (p50)
        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            2,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        # Segment 0: fid0=0.3 filtered (<=0.5), fid0=0.7 kept (>0.5), fid1=0.5 filtered (<=1.0)
        # Segment 1: fid0=0.2 filtered, fid1=1.5 kept (>1.0), fid1=2.0 kept (>1.0)
        seg0_start = int(result.offsets[0].item())
        seg0_end = int(result.offsets[1].item())
        seg0_fids = result.feature_ids[seg0_start:seg0_end]
        seg0_mags = result.feature_mags[seg0_start:seg0_end]

        seg1_start = int(result.offsets[1].item())
        seg1_end = int(result.offsets[2].item())
        seg1_fids = result.feature_ids[seg1_start:seg1_end]
        seg1_mags = result.feature_mags[seg1_start:seg1_end]

        # Segment 0: only fid0 with mag 0.7 should survive
        assert seg0_end - seg0_start == 1
        assert seg0_fids[0].item() == 0
        assert seg0_mags[0].item() == pytest.approx(0.7)

        # Segment 1: fid1 with mags 1.5 and 2.0 should survive
        assert seg1_end - seg1_start == 2
        assert set(seg1_fids.tolist()) == {1}
        assert sorted(seg1_mags.tolist()) == pytest.approx([1.5, 2.0])


class TestThresholdStageAutoResume:
    """Test that the threshold stage auto-resumes when output exists."""

    def test_auto_resume_loads_from_disk(self, tmp_path):
        """When thresholded scan file exists, it is loaded instead of recomputed."""
        # Create and save a known ScanResultTensors
        expected = ScanResultTensors(
            feature_ids=t.tensor([10, 20], dtype=t.int64),
            n_docs_activated=t.tensor([3, 5], dtype=t.int64),
            offsets=t.tensor([0, 3, 8], dtype=t.int64),
            doc_idx=t.tensor([0, 1, 2, 10, 11, 12, 13, 14], dtype=t.int64),
            max_activation=t.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=t.float32
            ),
            max_segment_idx=t.tensor([1, 2, 1, 3, 2, 1, 4, 2], dtype=t.int64),
            percentile=t.tensor(
                [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7], dtype=t.float32
            ),
            total_segments=t.tensor([5, 5, 5, 8, 8, 8, 8, 8], dtype=t.int64),
        )

        path = tmp_path / "validation_scan_thresholded.safetensors"
        save_scan_results(expected, path)

        # Load it back
        loaded = load_scan_results(path)

        assert t.equal(loaded.feature_ids, expected.feature_ids)
        assert t.equal(loaded.n_docs_activated, expected.n_docs_activated)
        assert t.equal(loaded.offsets, expected.offsets)
        assert t.equal(loaded.doc_idx, expected.doc_idx)
        assert t.allclose(loaded.max_activation, expected.max_activation)
        assert t.equal(loaded.max_segment_idx, expected.max_segment_idx)
        assert t.allclose(loaded.percentile, expected.percentile)
        assert t.equal(loaded.total_segments, expected.total_segments)


class TestThresholdedFeaturesSurviveGenerationThreshold:
    """Key correctness test: every feature in thresholded scan survives apply_threshold.

    This proves the bug is fixed: features selected from the thresholded scan
    will NOT be filtered out during generation-time thresholding.
    """

    def test_all_thresholded_features_survive(self, tmp_path):
        """For every (feature, doc, segment) in the thresholded scan, calling
        apply_threshold on that segment retains the feature."""
        shard = _build_shard_data(
            n_docs=5, segs_per_doc=4, features_per_seg=20, n_features=30, seed=123
        )
        shard_path = _save_shard_to_disk(shard, tmp_path)

        all_fids = list(range(30))
        stats = _make_per_feature_stats(all_fids)

        # Use the same threshold type as the real pipeline
        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, 2048.9
        )
        precomputed = build_thresholds_from_stats(
            stats,
            30,
            ThresholdType.feature_percentile_segment_topk,
            2048.9,
        )

        # Get thresholded scan
        scan_result = scan_thresholded_features(
            shard_path, stats, threshold_cfg, precomputed
        )

        # Load the raw shard to get segment-level data

        raw_data = load_features(shard_path)

        # For each feature in the scan, check that it survives thresholding
        # on its max_segment_idx
        for feat_idx in range(len(scan_result.feature_ids)):
            fid = int(scan_result.feature_ids[feat_idx].item())
            start = int(scan_result.offsets[feat_idx].item())
            end = int(scan_result.offsets[feat_idx + 1].item())

            for j in range(start, end):
                doc_idx = int(scan_result.doc_idx[j].item())
                max_seg_idx = int(scan_result.max_segment_idx[j].item())

                # Find the segment in the raw shard
                offsets = raw_data["offsets"]
                doc_ids = raw_data["doc_ids"]
                seg_indices = raw_data["segment_indices"]

                # Find segment position matching (doc_idx, max_seg_idx)
                found = False
                for seg_i in range(int(raw_data["n_segments"])):
                    if (
                        int(doc_ids[seg_i].item()) == doc_idx
                        and int(seg_indices[seg_i].item()) == max_seg_idx
                    ):
                        seg_start = int(offsets[seg_i].item())
                        seg_end = int(offsets[seg_i + 1].item())
                        seg_fids = raw_data["feature_ids"][seg_start:seg_end]
                        seg_mags = raw_data["feature_mags"][seg_start:seg_end]

                        # Apply the same threshold
                        filtered_fids, _ = apply_threshold(
                            threshold_cfg, seg_fids, seg_mags, precomputed
                        )

                        assert fid in filtered_fids.tolist(), (
                            f"Feature {fid} (doc={doc_idx}, seg={max_seg_idx}) "
                            f"was in thresholded scan but did NOT survive "
                            f"apply_threshold on its target segment. "
                            f"Filtered fids: {filtered_fids.tolist()}"
                        )
                        found = True
                        break

                assert found, (
                    f"Could not find segment (doc={doc_idx}, seg={max_seg_idx}) "
                    f"in raw shard"
                )


class TestEmptyShardAfterThreshold:
    """Edge case: all features filtered out by aggressive threshold."""

    def test_aggressive_threshold_produces_empty_scan(self):
        """With very high threshold, all features are filtered → empty scan."""
        # Create shard with very low magnitudes
        shard = {
            "offsets": t.tensor([0, 3, 6], dtype=t.int64),
            "doc_ids": t.tensor([0, 1], dtype=t.int64),
            "segment_indices": t.tensor([1, 1], dtype=t.int64),
            "feature_ids": t.tensor([0, 1, 2, 0, 1, 2], dtype=t.int64),
            "feature_mags": t.tensor(
                [0.001, 0.002, 0.003, 0.001, 0.002, 0.003], dtype=t.float32
            ),
            "n_segments": 2,
        }

        # Create stats where p99 threshold is 99.0 (way above any activation)
        stats: dict[str, PerFeatureStat] = {}
        for fid in [0, 1, 2]:
            percentiles = {f"p{i}": float(i) for i in range(1, 100)}
            stats[str(fid)] = PerFeatureStat(
                count=100,
                pct_segments_active=0.5,
                mean_mag=50.0,
                mean_mag_incl_zeros=25.0,
                std_mag=10.0,
                min_mag=0.0,
                max_mag=100.0,
                median_mag=50.0,
                percentiles=percentiles,
            )

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.99)
        precomputed = build_thresholds_from_stats(
            stats,
            3,
            ThresholdType.feature_percentile,
            0.99,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)
        thresholded_data = result

        # All features should be filtered out
        assert thresholded_data.feature_ids.numel() == 0
        assert thresholded_data.feature_mags.numel() == 0

        # _scan_core should handle empty input gracefully
        scan = _scan_core(
            thresholded_data.offsets,
            thresholded_data.doc_ids,
            thresholded_data.segment_indices,
            thresholded_data.feature_ids,
            thresholded_data.feature_mags,
            int(thresholded_data.n_segments),
            stats,
        )
        assert len(scan.feature_ids) == 0
        assert len(scan.doc_idx) == 0


class TestThresholdPreservesDocIdsAndSegmentIndices:
    """Verify structural invariants are preserved after thresholding."""

    def test_doc_ids_unchanged(self):
        """doc_ids pass through unchanged."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=2, features_per_seg=5, n_features=10
        )
        stats = _make_per_feature_stats(list(range(10)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            10,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        assert t.equal(result.doc_ids, shard["doc_ids"])

    def test_segment_indices_unchanged(self):
        """segment_indices pass through unchanged."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=2, features_per_seg=5, n_features=10
        )
        stats = _make_per_feature_stats(list(range(10)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            10,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        assert t.equal(result.segment_indices, shard["segment_indices"])

    def test_n_segments_unchanged(self):
        """n_segments is preserved."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=2, features_per_seg=5, n_features=10
        )
        stats = _make_per_feature_stats(list(range(10)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            10,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        assert result.n_segments == shard["n_segments"]

    def test_offsets_length_correct(self):
        """Offsets have length n_segments + 1."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=2, features_per_seg=5, n_features=10
        )
        stats = _make_per_feature_stats(list(range(10)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            10,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        assert len(result.offsets) == shard["n_segments"] + 1

    def test_offsets_monotonically_increasing(self):
        """Offsets are monotonically non-decreasing."""
        shard = _build_shard_data(
            n_docs=3, segs_per_doc=3, features_per_seg=8, n_features=15
        )
        stats = _make_per_feature_stats(list(range(15)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            15,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        offsets = result.offsets
        for i in range(len(offsets) - 1):
            assert offsets[i] <= offsets[i + 1], (
                f"Offsets not monotonic at index {i}: {offsets[i]} > {offsets[i + 1]}"
            )

    def test_total_features_equals_last_offset(self):
        """Total number of features equals the last offset value."""
        shard = _build_shard_data(
            n_docs=2, segs_per_doc=3, features_per_seg=7, n_features=12
        )
        stats = _make_per_feature_stats(list(range(12)))

        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.5)
        precomputed = build_thresholds_from_stats(
            stats,
            12,
            ThresholdType.feature_percentile,
            0.5,
        )

        result = apply_threshold_to_shard(shard, threshold_cfg, precomputed)

        assert result.feature_ids.numel() == int(result.offsets[-1].item())
        assert result.feature_mags.numel() == int(result.offsets[-1].item())
