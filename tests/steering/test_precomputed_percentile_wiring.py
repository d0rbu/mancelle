"""Tests for precomputed percentile wiring through the training pipeline.

Verifies:
1. FeatureIndex loads precomputed percentiles when checksum matches
2. Dataloader passes precomputed percentiles to sample_features_batched
3. Fallback to runtime only when checksum is absent; mismatches fail loudly
4. Delta filtering correctly subsets precomputed percentiles
5. End-to-end: augmented shard → dataloader → sampling produces valid results
6. Stats file checksum persistence wires to FeatureIndex
"""

from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_pipeline import FeaturePipelineConfig, ThresholdSelectionConfig
from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
    sample_features_batched,
)
from steering.feature_io import (
    TensorKey,
    save_features,
)
from steering.overlay_io import (
    OverlayIncompleteError,
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.stats_io import (
    load_percentile_checksum,
    save_percentile_checksum,
)

N_SAE = 200
SEGMENT_SIZE = 16
N_DOCS_PER_SHARD = 10
N_SEGS_PER_DOC = 2
N_FEATS_PER_SEG = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer for dataloader tests."""

    def __init__(self, vocab_size=100):
        self._vocab_size = vocab_size

    def get_vocab_size(self):
        return self._vocab_size

    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        tokens = list(range(1, len(text.split()) + 1))
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        return "".join(f"w{tid:04d} " for tid in token_ids)


def _make_pct_table():
    """Build a monotonic percentile table [N_SAE, 99]."""
    t.manual_seed(0)
    return t.rand(N_SAE, 99, dtype=t.float64).cumsum(dim=1)


def _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True,
                                 wrong_checksum=False):
    """Create synthetic parquet + feature shards, optionally with precomputed percentiles.

    Returns:
        dict with data_dir, feature_dir, checksum, pct_table
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    checksum = compute_percentile_checksum(pct_table)

    for shard_idx in range(1):  # 1 shard for simplicity
        shard_stem = f"shard_{shard_idx:05d}"

        # Parquet file
        texts = []
        for _ in range(N_DOCS_PER_SHARD):
            words = [f"word{w}" for w in range(40)]
            texts.append(" ".join(words))
        pq_path = data_dir / f"{shard_stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        # Feature shard
        total_segs = N_DOCS_PER_SHARD * N_SEGS_PER_DOC
        total_feats = total_segs * N_FEATS_PER_SEG

        all_fids = t.zeros(total_feats, dtype=t.int32)
        all_mags = t.zeros(total_feats, dtype=t.float32)
        offsets = t.zeros(total_segs + 1, dtype=t.int64)
        doc_ids = t.zeros(total_segs, dtype=t.int32)
        seg_indices = t.zeros(total_segs, dtype=t.int32)
        seg_positions = t.zeros(total_segs, dtype=t.int32)

        feat_offset = 0
        seg_global = 0
        for doc_idx in range(N_DOCS_PER_SHARD):
            for seg_idx in range(N_SEGS_PER_DOC):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
                offsets[seg_global] = feat_offset

                for f in range(N_FEATS_PER_SEG):
                    fid = (doc_idx * 10 + seg_idx * 5 + f) % N_SAE
                    all_fids[feat_offset + f] = fid
                    all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx

                feat_offset += N_FEATS_PER_SEG
                seg_global += 1

        offsets[total_segs] = feat_offset

        segment_char_starts = seg_positions * 6
        segment_char_ends = (seg_positions + SEGMENT_SIZE) * 6

        shard_data = {
            "offsets": offsets,
            "feature_ids": all_fids,
            "feature_mags": all_mags,
            "doc_ids": doc_ids,
            "segment_indices": seg_indices,
            "segment_positions": seg_positions,
            "doc_token_counts": t.full((N_DOCS_PER_SHARD,), 41, dtype=t.int32),
            "segment_char_starts": segment_char_starts,
            "segment_char_ends": segment_char_ends,
            "n_docs": N_DOCS_PER_SHARD,
            "n_segments": total_segs,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": SEGMENT_SIZE,
        }

        save_features(shard_data, feature_dir / f"{shard_stem}_features")

        if with_percentiles:
            # Write percentile overlay shard (not embedded in base shard)
            effective_checksum = "wrong_checksum_value" if wrong_checksum else checksum
            pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, effective_checksum)
            write_manifest(pct_dir, effective_checksum, OverlayKind.PERCENTILE)
            pct_scores = magnitude_to_percentile(
                all_fids.to(t.int64), all_mags, pct_table
            )
            save_overlay_shard(
                {TensorKey.FEATURE_PERCENTILES: percentile_to_uint8(pct_scores)},
                overlay_shard_path(pct_dir, shard_stem),
            )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "checksum": checksum,
        "pct_table": pct_table,
    }


def _make_sampling_pipeline(pct_table, num_buckets=4):
    """Build a SamplingSelection-based pipeline config."""
    rng = t.Generator().manual_seed(42)
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0,
        std=3.0,
        min_k=1,
        max_k=N_FEATS_PER_SEG,
    )
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=pct_table,
            rng=rng,
        ),
        boundaries=t.zeros(N_SAE, num_buckets - 1, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=num_buckets,
    )


def _make_threshold_pipeline(num_buckets=4):
    """Build a ThresholdSelection-based pipeline config (no sampling path)."""
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(threshold_type="segment_topk", threshold=3),  # ty: ignore[invalid-argument-type]
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(N_SAE, num_buckets - 1, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=num_buckets,
    )


# ---------------------------------------------------------------------------
# 1. FeatureIndex checksum wiring
# ---------------------------------------------------------------------------


class TestFeatureIndexChecksumWiring:
    """Verify FeatureIndex correctly gates precomputed percentiles on checksum match."""

    def test_matching_checksum_enables_precomputed(self, tmp_path):
        """FeatureIndex with correct checksum enables has_precomputed_percentiles."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

        # Must trigger shard load via lookup
        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert idx.has_precomputed_percentiles is True

    def test_no_checksum_disables_precomputed(self, tmp_path):
        """FeatureIndex without checksum param never exposes precomputed percentiles."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"])  # no checksum

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert idx.has_precomputed_percentiles is False
        # Segment percentiles should be None when checksum not provided
        assert segs[0].feature_percentiles is None

    def test_wrong_checksum_fails_loudly(self, tmp_path):
        """A requested checksum mismatch must not silently fall back."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True,
                                           wrong_checksum=True)
        with pytest.raises(OverlayIncompleteError, match="Requested percentile overlay"):
            FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

    def test_shard_without_percentiles_fails_loudly(self, tmp_path):
        """A requested overlay without payloads must not silently fall back."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=False)
        with pytest.raises(OverlayIncompleteError, match="Requested percentile overlay"):
            FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

    def test_precomputed_segment_features_have_correct_dtype(self, tmp_path):
        """When enabled, segment feature_percentiles are uint8 with correct length."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        for seg in segs:
            assert seg.feature_percentiles is not None
            assert seg.feature_percentiles.dtype == t.uint8
            assert len(seg.feature_percentiles) == len(seg.feature_ids)

    def test_all_docs_get_precomputed_percentiles(self, tmp_path):
        """Every document's segments get precomputed percentiles when enabled."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

        for doc_id in range(N_DOCS_PER_SHARD):
            segs = idx.lookup("shard_00000", doc_id)
            assert segs is not None, f"doc_id={doc_id} returned None"
            for seg in segs:
                assert seg.feature_percentiles is not None, (
                    f"doc_id={doc_id}, seg_idx={seg.segment_idx} missing percentiles"
                )


# ---------------------------------------------------------------------------
# 2. Stats file → FeatureIndex checksum wiring
# ---------------------------------------------------------------------------


class TestStatsToFeatureIndexWiring:
    """Verify the stats file checksum flows correctly to FeatureIndex."""

    def test_save_and_load_checksum_roundtrip(self, tmp_path):
        """Checksum saved to stats file can be loaded and used with FeatureIndex."""
        from safetensors.torch import save_file

        # Create a minimal stats file
        stats_path = tmp_path / "feature_statistics.safetensors"
        save_file(
            {"dummy": t.tensor([0], dtype=t.int32)},
            str(stats_path),
            metadata={"n_sae_features": str(N_SAE), "layer": "8"},
        )

        pct_table = _make_pct_table()
        checksum = compute_percentile_checksum(pct_table)

        # Save checksum to stats file
        save_percentile_checksum(stats_path, checksum)

        # Load it back
        loaded_checksum = load_percentile_checksum(stats_path)
        assert loaded_checksum == checksum

        # Use it with FeatureIndex
        env = _make_augmented_feature_dir(tmp_path / "sub", pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=loaded_checksum)
        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert idx.has_precomputed_percentiles is True

    def test_missing_checksum_in_stats_returns_none(self, tmp_path):
        """Stats file without checksum returns None, FeatureIndex disables precomputed."""
        from safetensors.torch import save_file

        stats_path = tmp_path / "feature_statistics.safetensors"
        save_file(
            {"dummy": t.tensor([0], dtype=t.int32)},
            str(stats_path),
            metadata={"n_sae_features": str(N_SAE)},
        )

        loaded_checksum = load_percentile_checksum(stats_path)
        assert loaded_checksum is None

        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path / "sub", pct_table, with_percentiles=True)
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=loaded_checksum)
        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert idx.has_precomputed_percentiles is False


# ---------------------------------------------------------------------------
# 3. sample_features_batched precomputed path
# ---------------------------------------------------------------------------


class TestSampleFeaturesWithPrecomputed:
    """Verify sample_features_batched correctly uses precomputed percentiles."""

    def test_precomputed_produces_valid_output(self):
        """Precomputed percentile path produces valid sampled features."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0, std=3.0, min_k=1, max_k=5,
        )
        rng = t.Generator().manual_seed(42)

        # Create 3 segments with known features
        seg_sizes = [5, 5, 5]
        n_total = sum(seg_sizes)
        fids = t.randint(0, N_SAE, (n_total,), dtype=t.int64)
        mags = t.rand(n_total) + 0.01

        # Compute ground-truth percentiles and encode as precomputed
        true_pcts = magnitude_to_percentile(fids, mags, pct_table)
        precomputed = percentile_to_uint8(true_pcts).float() / 255.0

        results = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )

        assert len(results) == 3
        for sampled_ids, sampled_mags in results:
            assert sampled_ids.dtype == t.int64
            assert sampled_mags.dtype == t.float32
            assert len(sampled_ids) == len(sampled_mags)
            assert len(sampled_ids) > 0  # min_k=1

    def test_precomputed_skips_magnitude_to_percentile(self):
        """When precomputed_percentiles given, magnitude_to_percentile is not called."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0, std=3.0, min_k=1, max_k=5,
        )
        rng = t.Generator().manual_seed(42)

        seg_sizes = [5]
        fids = t.randint(0, N_SAE, (5,), dtype=t.int64)
        mags = t.rand(5) + 0.01

        precomputed = t.rand(5, dtype=t.float64)

        with patch(
            "steering.conditional.sampling.magnitude_to_percentile",
            side_effect=AssertionError("Should not be called"),
        ):
            results = sample_features_batched(
                fids, mags, seg_sizes, pct_table, config, rng,
                precomputed_percentiles=precomputed,
            )
            assert len(results) == 1

    def test_none_precomputed_falls_back_to_runtime(self):
        """Without precomputed percentiles, runtime computation is used."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0, std=3.0, min_k=1, max_k=5,
        )

        seg_sizes = [5]
        fids = t.randint(0, N_SAE, (5,), dtype=t.int64)
        mags = t.rand(5) + 0.01

        # Run twice with same seed, no precomputed — should use runtime path
        rng1 = t.Generator().manual_seed(42)
        results1 = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng1,
            precomputed_percentiles=None,
        )

        rng2 = t.Generator().manual_seed(42)
        results2 = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng2,
            precomputed_percentiles=None,
        )

        # Same seed → same output
        assert t.equal(results1[0][0], results2[0][0])

    def test_precomputed_vs_runtime_agreement(self):
        """Precomputed path and runtime path produce identical selections with same seed.

        The uint8 encoding introduces small quantization error, so we verify
        that when given exact (non-quantized) percentiles, results match exactly.
        """
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0, std=3.0, min_k=1, max_k=5,
        )

        seg_sizes = [5, 5]
        n_total = 10
        fids = t.randint(0, N_SAE, (n_total,), dtype=t.int64)
        mags = t.rand(n_total) + 0.01

        # Compute exact percentiles (not uint8-quantized)
        exact_pcts = magnitude_to_percentile(fids, mags, pct_table)

        rng1 = t.Generator().manual_seed(42)
        results_precomputed = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng1,
            precomputed_percentiles=exact_pcts,
        )

        rng2 = t.Generator().manual_seed(42)
        results_runtime = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng2,
            precomputed_percentiles=None,
        )

        for i in range(len(seg_sizes)):
            assert t.equal(results_precomputed[i][0], results_runtime[i][0]), (
                f"Segment {i}: precomputed selected {results_precomputed[i][0].tolist()} "
                f"vs runtime {results_runtime[i][0].tolist()}"
            )


# ---------------------------------------------------------------------------
# 4. Dataloader integration with precomputed percentiles
# ---------------------------------------------------------------------------


class TestDataloaderWiringLogic:
    """Verify the dataloader's precomputed percentile wiring logic.

    Instead of running the full dataloader (which requires a complete data pipeline
    and can hang in test environments), these tests verify the specific wiring logic:
    FeatureIndex → segment percentiles → accumulation → sample_features_batched.
    """

    def test_precomputed_path_accumulates_and_decodes(self, tmp_path):
        """Simulate what the dataloader does: accumulate segment percentiles, decode, pass to sampling."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)

        feature_index = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

        # Simulate dataloader's per-segment accumulation
        rg_all_fids_parts: list[t.Tensor] = []
        rg_all_mags_parts: list[t.Tensor] = []
        rg_all_pcts_parts: list[t.Tensor] = []
        rg_seg_sizes: list[int] = []

        has_precomputed = feature_index.has_precomputed_percentiles
        # has_precomputed is False before any shard is loaded
        # After first lookup, it becomes True if checksum matches
        for doc_idx in range(N_DOCS_PER_SHARD):
            segs = feature_index.lookup("shard_00000", doc_idx)
            assert segs is not None
            has_precomputed = feature_index.has_precomputed_percentiles  # check after load

            for seg in segs:
                rg_all_fids_parts.append(seg.feature_ids.to(t.int64))
                rg_all_mags_parts.append(seg.feature_mags)
                rg_seg_sizes.append(len(seg.feature_ids))

                if has_precomputed and seg.feature_percentiles is not None:
                    decoded_pcts = seg.feature_percentiles.float() / 255.0
                    rg_all_pcts_parts.append(decoded_pcts)
                else:
                    rg_all_pcts_parts.append(None)  # ty: ignore[invalid-argument-type]

        assert has_precomputed is True
        assert all(p is not None for p in rg_all_pcts_parts)

        # Concatenate and call sample_features_batched (exactly as dataloader does)
        rg_all_fids = t.cat(rg_all_fids_parts)
        rg_all_mags = t.cat(rg_all_mags_parts)
        rg_precomputed_pcts = t.cat(rg_all_pcts_parts)

        assert rg_precomputed_pcts.shape == rg_all_fids.shape

        config = SamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0,
            min_k=1, max_k=N_FEATS_PER_SEG,
        )
        rng = t.Generator().manual_seed(42)
        results = sample_features_batched(
            rg_all_fids, rg_all_mags, rg_seg_sizes,
            pct_table, config, rng,
            precomputed_percentiles=rg_precomputed_pcts,
        )

        assert len(results) == N_DOCS_PER_SHARD * N_SEGS_PER_DOC
        for sampled_ids, _ in results:
            assert len(sampled_ids) > 0
            assert sampled_ids.dtype == t.int64

    def test_fallback_path_no_checksum(self, tmp_path):
        """Without checksum, dataloader accumulates None percentiles and falls back to runtime."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)

        feature_index = FeatureIndex(env["feature_dir"])  # No checksum!

        rg_all_fids_parts: list[t.Tensor] = []
        rg_all_mags_parts: list[t.Tensor] = []
        rg_all_pcts_parts: list[t.Tensor | None] = []
        rg_seg_sizes: list[int] = []

        for doc_idx in range(N_DOCS_PER_SHARD):
            segs = feature_index.lookup("shard_00000", doc_idx)
            assert segs is not None
            has_precomputed = feature_index.has_precomputed_percentiles

            for seg in segs:
                rg_all_fids_parts.append(seg.feature_ids.to(t.int64))
                rg_all_mags_parts.append(seg.feature_mags)
                rg_seg_sizes.append(len(seg.feature_ids))

                if has_precomputed and seg.feature_percentiles is not None:
                    rg_all_pcts_parts.append(seg.feature_percentiles.float() / 255.0)
                else:
                    rg_all_pcts_parts.append(None)

        # Without checksum, all percentiles should be None
        assert not feature_index.has_precomputed_percentiles
        assert all(p is None for p in rg_all_pcts_parts)

        # Dataloader would pass None to sample_features_batched
        rg_all_fids = t.cat(rg_all_fids_parts)
        rg_all_mags = t.cat(rg_all_mags_parts)

        config = SamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0,
            min_k=1, max_k=N_FEATS_PER_SEG,
        )
        rng = t.Generator().manual_seed(42)
        results = sample_features_batched(
            rg_all_fids, rg_all_mags, rg_seg_sizes,
            pct_table, config, rng,
            precomputed_percentiles=None,  # runtime path
        )

        assert len(results) == N_DOCS_PER_SHARD * N_SEGS_PER_DOC
        for sampled_ids, _ in results:
            assert len(sampled_ids) > 0

    def test_wrong_checksum_fails_loudly(self, tmp_path):
        """Wrong checksum fails before the dataloader can fall back."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True,
                                           wrong_checksum=True)

        with pytest.raises(OverlayIncompleteError, match="Requested percentile overlay"):
            FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

    def test_precomputed_decoded_values_match_encoding(self, tmp_path):
        """Decoded precomputed percentiles from shard match the original encoding."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)

        feature_index = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])

        # Load a document and check percentile values
        segs = feature_index.lookup("shard_00000", 0)
        assert segs is not None
        seg = segs[0]
        assert seg.feature_percentiles is not None

        # Recompute expected percentiles
        expected_pcts = magnitude_to_percentile(
            seg.feature_ids.to(t.int64), seg.feature_mags, pct_table
        )
        expected_uint8 = percentile_to_uint8(expected_pcts)

        # Stored uint8 should match
        assert t.equal(seg.feature_percentiles, expected_uint8)

        # Decoded float should be close to normalized runtime
        decoded = seg.feature_percentiles.float() / 255.0
        expected_normalized = expected_pcts.float() / 100.0
        max_err = (decoded - expected_normalized).abs().max()
        assert max_err < 0.005

    def test_precomputed_and_runtime_sampling_produce_same_k(self, tmp_path):
        """Precomputed and runtime paths sample the same number of features (same k)."""
        pct_table = _make_pct_table()
        env = _make_augmented_feature_dir(tmp_path, pct_table, with_percentiles=True)

        # Build feature tensors from index
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["checksum"])
        fid_parts, mag_parts, pct_parts, seg_sizes = [], [], [], []
        for doc_idx in range(N_DOCS_PER_SHARD):
            segs = idx.lookup("shard_00000", doc_idx)
            for seg in segs:
                fids = seg.feature_ids.to(t.int64)
                fid_parts.append(fids)
                mag_parts.append(seg.feature_mags)
                pct_parts.append(seg.feature_percentiles.float() / 255.0)  # ty: ignore[unresolved-attribute]
                seg_sizes.append(len(fids))

        all_fids = t.cat(fid_parts)
        all_mags = t.cat(mag_parts)
        all_pcts = t.cat(pct_parts)

        config = SamplingSelection(
            distribution=Distribution.geometric, mean=3.0, std=3.0,
            min_k=1, max_k=N_FEATS_PER_SEG,
        )

        # With precomputed: k is determined by RNG, not percentile values
        rng1 = t.Generator().manual_seed(99)
        results_pre = sample_features_batched(
            all_fids, all_mags, seg_sizes, pct_table, config, rng1,
            precomputed_percentiles=all_pcts,
        )

        # Without precomputed (runtime): same seed → same k values
        rng2 = t.Generator().manual_seed(99)
        results_rt = sample_features_batched(
            all_fids, all_mags, seg_sizes, pct_table, config, rng2,
            precomputed_percentiles=None,
        )

        # k (number sampled) should be identical since it's determined by RNG, not weights
        for i, ((pre_ids, _), (rt_ids, _)) in enumerate(zip(results_pre, results_rt, strict=True)):
            assert len(pre_ids) == len(rt_ids), (
                f"Segment {i}: precomputed sampled {len(pre_ids)} vs runtime {len(rt_ids)}"
            )


# ---------------------------------------------------------------------------
# 5. Precomputed percentile accuracy
# ---------------------------------------------------------------------------


class TestPrecomputedPercentileAccuracy:
    """Verify uint8-encoded percentiles decode to values close to runtime computation."""

    def test_decoded_percentiles_approximate_runtime(self):
        """Decoded uint8 percentiles are within quantization error of runtime values."""
        pct_table = _make_pct_table()

        n = 50
        fids = t.randint(0, N_SAE, (n,), dtype=t.int64)
        mags = t.rand(n) + 0.01

        # Runtime percentiles
        runtime_pcts = magnitude_to_percentile(fids, mags, pct_table)

        # Encode → decode roundtrip
        encoded = percentile_to_uint8(runtime_pcts)
        decoded = encoded.float() / 255.0  # same as dataloader does

        # Runtime values are in [0.5, 99.5], normalized to [0.005, 0.995]
        runtime_normalized = runtime_pcts.float() / 100.0

        # Max quantization error is 1/255 ≈ 0.004
        max_error = (decoded - runtime_normalized).abs().max()
        assert max_error < 0.005, f"Max quantization error {max_error:.6f} exceeds 0.005"

    def test_percentile_ordering_preserved(self):
        """Relative ordering of percentiles is preserved through uint8 encoding."""
        pct_table = _make_pct_table()

        n = 100
        fids = t.randint(0, N_SAE, (n,), dtype=t.int64)
        mags = t.rand(n) + 0.01

        runtime_pcts = magnitude_to_percentile(fids, mags, pct_table)
        encoded = percentile_to_uint8(runtime_pcts)

        # Where runtime has strict ordering, encoded should preserve it
        # (within quantization limits — ties may be introduced)
        runtime_sorted_indices = runtime_pcts.argsort()
        encoded_at_sorted = encoded[runtime_sorted_indices]
        # Non-decreasing after sorting by runtime order
        assert (encoded_at_sorted[1:] >= encoded_at_sorted[:-1]).all()


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestPrecomputedEdgeCases:
    """Edge cases for precomputed percentile wiring."""

    def test_empty_segment_precomputed(self):
        """Segments with 0 features work correctly with precomputed path."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0, std=3.0, min_k=0, max_k=5,
        )
        rng = t.Generator().manual_seed(42)

        # Mixed segments: 0, 5, 0
        seg_sizes = [0, 5, 0]
        fids = t.randint(0, N_SAE, (5,), dtype=t.int64)
        mags = t.rand(5) + 0.01
        precomputed = t.rand(5, dtype=t.float64)

        results = sample_features_batched(
            fids, mags, seg_sizes, pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )

        assert len(results) == 3
        assert len(results[0][0]) == 0  # empty segment
        assert len(results[2][0]) == 0  # empty segment

    def test_single_feature_precomputed(self):
        """Single-feature segment works with precomputed path."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0, std=1.0, min_k=1, max_k=2,
        )
        rng = t.Generator().manual_seed(42)

        fids = t.tensor([42], dtype=t.int64)
        mags = t.tensor([0.5], dtype=t.float32)
        precomputed = t.tensor([0.5], dtype=t.float64)

        results = sample_features_batched(
            fids, mags, [1], pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )

        assert len(results) == 1
        assert len(results[0][0]) == 1
        assert results[0][0].item() == 42

    def test_precomputed_all_same_percentile(self):
        """All features with identical precomputed percentile → uniform sampling."""
        pct_table = _make_pct_table()
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0, std=2.0, min_k=1, max_k=3,
        )
        rng = t.Generator().manual_seed(42)

        n = 10
        fids = t.arange(n, dtype=t.int64)
        mags = t.ones(n, dtype=t.float32)
        precomputed = t.full((n,), 0.5, dtype=t.float64)

        results = sample_features_batched(
            fids, mags, [n], pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )

        assert len(results) == 1
        sampled_ids = results[0][0]
        assert len(sampled_ids) >= 1
        assert len(sampled_ids) <= 3
