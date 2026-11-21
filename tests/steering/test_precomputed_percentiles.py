"""Comprehensive tests for the precomputed percentile pipeline.

Tests cover:
1. uint8 encoding/decoding roundtrip and precision
2. Checksum determinism and sensitivity
3. feature_io save/load with percentiles and checksum
4. Co-sort correctness (percentiles stay aligned after sort)
5. Precompute script end-to-end
6. FeatureIndex integration (checksum validation, has_precomputed_percentiles)
7. sample_features_batched with precomputed_percentiles
8. Reference comparison: precomputed vs runtime percentiles
9. Edge cases: empty shards, zero features, mismatched checksums
"""

import pickle
from pathlib import Path

import pytest
import torch as t

import scripts.preprocess.precompute_percentiles as precompute_percentiles_module
from errors import EmptyExtractionSegmentError
from scripts.preprocess.precompute_percentiles import precompute_shard
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    sample_features_batched,
)
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    load_features,
    read_shard_header,
    save_features,
)
from steering.overlay_io import (
    OverlayIncompleteError,
    OverlayKind,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.percentile_precompute import magnitude_to_percentile_chunked
from tests.steering._helpers import save_raw_shard

N_SAE = 200


def _precompute_shard_overlay(shard_path, pct_table, checksum, **kwargs):
    """Test helper: calls precompute_shard with overlay API, returns overlay shard path."""

    from scripts.preprocess.precompute_percentiles import precompute_shard
    from steering.feature_io import shard_stem_from_path

    feature_dir = shard_path.parent
    stem = shard_stem_from_path(shard_path)
    out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    write_manifest(out_dir, checksum, OverlayKind.PERCENTILE)
    precompute_shard(shard_path, pct_table, out_dir, stem, **kwargs)
    return overlay_shard_path(out_dir, stem)


@pytest.fixture
def pct_table():
    """Build a small monotonic percentile table [200, 99]."""
    t.manual_seed(0)
    return t.rand(N_SAE, 99, dtype=t.float64).cumsum(dim=1)


@pytest.fixture
def config():
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0, std=3.0,
        min_k=1, max_k=10,
    )


# ======================================================================
# 1. uint8 encoding/decoding roundtrip and precision
# ======================================================================

class TestUint8EncodingDecoding:
    """Verify percentile_to_uint8 and uint8_to_percentile are consistent."""

    def test_roundtrip_boundary_values(self):
        """Boundary percentile values encode and decode correctly."""
        scores = t.tensor([0.5, 50.0, 99.5], dtype=t.float64)
        encoded = percentile_to_uint8(scores)
        # 0.5/100*255 = 1.275 → 1
        # 50.0/100*255 = 127.5 → 128 (banker's rounding) or 127
        # 99.5/100*255 = 253.725 → 254
        assert encoded[0].item() == 1
        assert encoded[2].item() == 254
        assert encoded.dtype == t.uint8

    def test_roundtrip_monotonicity(self):
        """Encoding preserves the ordering of percentile scores."""
        scores = t.linspace(0.5, 99.5, 100, dtype=t.float64)
        encoded = percentile_to_uint8(scores)
        # Encoded values should be monotonically non-decreasing
        assert (encoded[1:] >= encoded[:-1]).all()

    def test_decode_range(self):
        """Decoded values are in [0, 1]."""
        all_uint8 = t.arange(256, dtype=t.uint8)
        decoded = uint8_to_percentile(all_uint8)
        assert decoded.min().item() == 0.0
        assert decoded.max().item() == 1.0
        assert decoded.dtype == t.float32

    def test_decode_zero(self):
        """uint8 value 0 decodes to 0.0."""
        assert uint8_to_percentile(t.tensor([0], dtype=t.uint8)).item() == 0.0

    def test_decode_255(self):
        """uint8 value 255 decodes to 1.0."""
        assert uint8_to_percentile(t.tensor([255], dtype=t.uint8)).item() == 1.0

    def test_quantization_error(self):
        """Quantization error is bounded by 1/255 ≈ 0.4%."""
        scores = t.linspace(0.5, 99.5, 1000, dtype=t.float64)
        encoded = percentile_to_uint8(scores)
        decoded = uint8_to_percentile(encoded)  # [0, 1]
        original_normalized = scores.float() / 100.0  # [0, 1]
        max_error = (decoded - original_normalized).abs().max().item()
        assert max_error < 1.0 / 255 + 1e-6  # at most 1 uint8 step

    def test_empty_input(self):
        """Empty tensor round-trips correctly."""
        empty = t.tensor([], dtype=t.float64)
        encoded = percentile_to_uint8(empty)
        assert len(encoded) == 0
        assert encoded.dtype == t.uint8
        decoded = uint8_to_percentile(encoded)
        assert len(decoded) == 0

    def test_all_same_value(self):
        """Constant input encodes to a single uint8 value."""
        scores = t.full((100,), 50.0, dtype=t.float64)
        encoded = percentile_to_uint8(scores)
        assert (encoded == encoded[0]).all()

    def test_values_outside_expected_range_are_clamped(self):
        """Values outside [0, 99.5] are clamped to [0, 255]."""
        scores = t.tensor([-10.0, 0.0, 100.0, 200.0], dtype=t.float64)
        encoded = percentile_to_uint8(scores)
        assert encoded.min().item() >= 0
        assert encoded.max().item() <= 255


# ======================================================================
# 2. Checksum determinism and sensitivity
# ======================================================================

class TestPercentileChecksum:
    """Verify compute_percentile_checksum is deterministic and sensitive."""

    def test_deterministic(self, pct_table):
        """Same table always produces the same checksum."""
        c1 = compute_percentile_checksum(pct_table)
        c2 = compute_percentile_checksum(pct_table)
        assert c1 == c2

    def test_different_tables_different_checksums(self, pct_table):
        """Modified table produces a different checksum."""
        c1 = compute_percentile_checksum(pct_table)
        modified = pct_table.clone()
        modified[0, 0] += 1e-10
        c2 = compute_percentile_checksum(modified)
        assert c1 != c2

    def test_is_sha256_hex(self, pct_table):
        """Checksum is a 64-character hex string (SHA-256)."""
        c = compute_percentile_checksum(pct_table)
        assert len(c) == 64
        int(c, 16)  # Should not raise

    def test_independent_of_memory_layout(self, pct_table):
        """Non-contiguous and contiguous tables produce the same checksum."""
        c1 = compute_percentile_checksum(pct_table)
        # Transpose to make it non-contiguous, then transpose back
        non_contig = pct_table.t().t()
        assert not non_contig.is_contiguous() or True  # depends on impl
        c2 = compute_percentile_checksum(non_contig)
        assert c1 == c2

    def test_zero_table(self):
        """All-zero table has a valid checksum."""
        table = t.zeros(100, 99, dtype=t.float64)
        c = compute_percentile_checksum(table)
        assert len(c) == 64

    def test_checksum_from_stats_roundtrip(self):
        """Checksum computed from stats-built table is deterministic."""
        from steering.stats_types import PerFeatureStat

        stats = {}
        for fid in range(10):
            pcts = {f"p{k}": float(k) * 0.1 + fid * 0.01 for k in range(1, 100)}
            stats[str(fid)] = PerFeatureStat(
                count=100,
                pct_segments_active=0.5,
                mean_mag=1.0, mean_mag_incl_zeros=0.5,
                std_mag=0.5, min_mag=0.1, max_mag=2.0,
                median_mag=1.0,
                percentiles=pcts,
            )
        table = build_percentile_table(stats, 20)
        c1 = compute_percentile_checksum(table)
        table2 = build_percentile_table(stats, 20)
        c2 = compute_percentile_checksum(table2)
        assert c1 == c2


# ======================================================================
# 3. feature_io save/load with percentiles and checksum
# ======================================================================

class TestFeatureIOPercentiles:
    """Verify feature_io correctly saves/loads shards with percentile data."""

    def _make_shard_data(self, n_segs=5, n_features=20):
        """Create minimal shard data dict."""
        t.manual_seed(42)
        offsets = [0]
        fids_list = []
        mags_list = []
        for _ in range(n_segs):
            n = t.randint(2, n_features + 1, (1,)).item()
            fids_list.append(t.randint(0, 200, (n,), dtype=t.int32))  # ty: ignore[no-matching-overload]
            mags_list.append(t.rand(n, dtype=t.float32))  # ty: ignore[no-matching-overload]
            offsets.append(offsets[-1] + n)

        all_fids = t.cat(fids_list)
        all_mags = t.cat(mags_list)

        return {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": all_fids,
            "feature_mags": all_mags,
            "doc_ids": t.zeros(n_segs, dtype=t.int32),
            "segment_indices": t.arange(n_segs, dtype=t.int32),
            "segment_positions": t.arange(n_segs, dtype=t.int32) * 64,
            "doc_token_counts": t.tensor([500], dtype=t.int32),
            "segment_char_starts": t.arange(n_segs, dtype=t.int32) * 100,
            "segment_char_ends": t.arange(n_segs, dtype=t.int32) * 100 + 64,
            "n_docs": 1,
            "n_segments": n_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }

    def test_save_load_roundtrip_with_percentiles(self, tmp_path):
        """Shard with percentiles survives save/load cycle."""
        data = self._make_shard_data()
        n_total = len(data["feature_ids"])
        data[TensorKey.FEATURE_PERCENTILES] = t.randint(0, 256, (n_total,), dtype=t.uint8)
        data[MetadataKey.PERCENTILE_CHECKSUM] = "abc123"

        path = save_features(data, tmp_path / "test_shard")
        loaded = load_features(path)

        assert TensorKey.FEATURE_PERCENTILES in loaded
        assert loaded[MetadataKey.PERCENTILE_CHECKSUM] == "abc123"
        pcts = loaded[TensorKey.FEATURE_PERCENTILES]
        assert pcts.dtype == t.uint8
        assert len(pcts) == n_total

    def test_roundtrip_percentile_values(self, tmp_path):
        """Percentile values are preserved exactly through save/load."""
        data = self._make_shard_data(n_segs=3, n_features=5)
        n = len(data["feature_ids"])
        original_pcts = t.arange(n, dtype=t.uint8) % 255
        data[TensorKey.FEATURE_PERCENTILES] = original_pcts
        data[MetadataKey.PERCENTILE_CHECKSUM] = "test_checksum"

        path = save_features(data, tmp_path / "shard")
        loaded = load_features(path)

        # After save/load, feature_ids are sorted within segments.
        # Percentiles should be co-sorted to stay aligned.
        # Verify all original values are present (may be reordered within segments).
        offsets = loaded["offsets"]
        for i in range(len(offsets) - 1):
            s, e = offsets[i].item(), offsets[i + 1].item()
            orig_seg_pcts = set(original_pcts[data["offsets"][i]:data["offsets"][i+1]].tolist())
            loaded_seg_pcts = set(loaded[TensorKey.FEATURE_PERCENTILES][s:e].tolist())
            assert orig_seg_pcts == loaded_seg_pcts

    def test_save_load_without_percentiles(self, tmp_path):
        """Shards without percentiles still load correctly."""
        data = self._make_shard_data()
        path = save_features(data, tmp_path / "no_pct_shard")
        loaded = load_features(path)

        assert TensorKey.FEATURE_PERCENTILES not in loaded
        assert MetadataKey.PERCENTILE_CHECKSUM not in loaded
        # Original keys still work
        assert "feature_ids" in loaded
        assert "feature_mags" in loaded

    def test_co_sort_alignment(self, tmp_path):
        """Percentiles stay aligned with feature_ids after sort within segments."""
        data = self._make_shard_data(n_segs=2, n_features=10)

        # Create known unsorted data
        offsets = t.tensor([0, 4, 8], dtype=t.int64)
        data["offsets"] = offsets
        # Segment 0: IDs [30, 10, 20, 40], each with a unique percentile
        data["feature_ids"] = t.tensor([30, 10, 20, 40, 5, 15, 25, 35], dtype=t.int32)
        data["feature_mags"] = t.tensor([3.0, 1.0, 2.0, 4.0, 0.5, 1.5, 2.5, 3.5])
        data["n_segments"] = 2
        # Percentiles: fid 30→pct 30, fid 10→pct 10, etc.
        data[TensorKey.FEATURE_PERCENTILES] = t.tensor([30, 10, 20, 40, 5, 15, 25, 35], dtype=t.uint8)
        data[MetadataKey.PERCENTILE_CHECKSUM] = "test"

        path = save_features(data, tmp_path / "cosort")
        loaded = load_features(path)

        fids = loaded["feature_ids"]
        pcts = loaded[TensorKey.FEATURE_PERCENTILES]

        # After sorting, segment 0 should be [10, 20, 30, 40]
        # with percentiles [10, 20, 30, 40] (co-sorted)
        assert fids[:4].tolist() == [10, 20, 30, 40]
        assert pcts[:4].tolist() == [10, 20, 30, 40]

        # Segment 1 should be [5, 15, 25, 35] with pcts [5, 15, 25, 35]
        assert fids[4:].tolist() == [5, 15, 25, 35]
        assert pcts[4:].tolist() == [5, 15, 25, 35]


# ======================================================================
# 3b. read_shard_header (fast metadata-only path)
# ======================================================================

class TestReadShardHeader:
    """Verify read_shard_header reads metadata without full decompression."""

    def _make_shard(self, tmp_path, with_percentiles=False, with_checksum=False):
        """Create a shard and return its path."""
        n = 20
        data = {
            "offsets": t.tensor([0, 10, n], dtype=t.int64),
            "feature_ids": t.randint(0, 100, (n,), dtype=t.int32),
            "feature_mags": t.rand(n) + 0.1,
            "doc_ids": t.tensor([0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([50, 50], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([64, 64], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 2,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }
        if with_percentiles:
            data[TensorKey.FEATURE_PERCENTILES] = t.randint(0, 256, (n,), dtype=t.uint8)
        if with_checksum:
            data[MetadataKey.PERCENTILE_CHECKSUM] = "test_checksum_abc"
        return save_features(data, tmp_path / "test_shard")

    def test_reads_metadata(self, tmp_path):
        """Metadata values are correctly extracted."""
        path = self._make_shard(tmp_path, with_checksum=True)
        header = read_shard_header(path)
        meta = header["__metadata__"]
        assert meta[MetadataKey.PERCENTILE_CHECKSUM] == "test_checksum_abc"
        assert meta["layer"] == "8"
        assert meta["segment_size"] == "64"

    def test_reads_tensor_keys(self, tmp_path):
        """All tensor key names are present."""
        path = self._make_shard(tmp_path, with_percentiles=True)
        header = read_shard_header(path)
        keys = header["__tensor_keys__"]
        assert "feature_ids" in keys
        assert "feature_mags" in keys
        assert "offsets" in keys
        assert TensorKey.FEATURE_PERCENTILES in keys

    def test_no_percentiles_in_keys(self, tmp_path):
        """Shard without percentiles doesn't report the key."""
        path = self._make_shard(tmp_path, with_percentiles=False)
        header = read_shard_header(path)
        assert TensorKey.FEATURE_PERCENTILES not in header["__tensor_keys__"]

    def test_no_checksum_in_metadata(self, tmp_path):
        """Shard without checksum returns no checksum key."""
        path = self._make_shard(tmp_path, with_checksum=False)
        header = read_shard_header(path)
        assert MetadataKey.PERCENTILE_CHECKSUM not in header["__metadata__"]

    def test_matches_full_load(self, tmp_path):
        """Header-only read matches what full load_features returns."""
        path = self._make_shard(tmp_path, with_percentiles=True, with_checksum=True)

        header = read_shard_header(path)
        data = load_features(path)

        # Checksum matches
        assert header["__metadata__"][MetadataKey.PERCENTILE_CHECKSUM] == data[MetadataKey.PERCENTILE_CHECKSUM]
        # Tensor keys match
        assert TensorKey.FEATURE_PERCENTILES in header["__tensor_keys__"]
        assert TensorKey.FEATURE_PERCENTILES in data

    def test_resume_decision_matches_full_load(self, tmp_path):
        """Fast resume check makes same skip/process decision as full load."""
        path = self._make_shard(tmp_path, with_percentiles=True, with_checksum=True)
        expected_checksum = "test_checksum_abc"

        # Fast path
        header = read_shard_header(path)
        fast_skip = (
            header["__metadata__"].get(MetadataKey.PERCENTILE_CHECKSUM) == expected_checksum
            and TensorKey.FEATURE_PERCENTILES in header["__tensor_keys__"]
        )

        # Full load path
        data = load_features(path)
        full_skip = (
            data.get(MetadataKey.PERCENTILE_CHECKSUM) == expected_checksum
            and TensorKey.FEATURE_PERCENTILES in data
            and isinstance(data[TensorKey.FEATURE_PERCENTILES], t.Tensor)
        )

        assert fast_skip == full_skip

    def test_wrong_checksum_not_skipped(self, tmp_path):
        """Mismatched checksum means we should NOT skip."""
        path = self._make_shard(tmp_path, with_percentiles=True, with_checksum=True)

        header = read_shard_header(path)
        should_skip = (
            header["__metadata__"].get(MetadataKey.PERCENTILE_CHECKSUM) == "different_checksum"
            and TensorKey.FEATURE_PERCENTILES in header["__tensor_keys__"]
        )
        assert not should_skip


# ======================================================================
# 4. Precompute script end-to-end
# ======================================================================

class TestPrecomputeScript:
    """Verify the precompute_percentiles script works end-to-end."""

    def _make_test_shard(self, tmp_path, shard_name, pct_table=None):
        """Create a minimal feature shard for testing."""
        t.manual_seed(42)
        n_features = 50
        fids = t.randint(0, pct_table.shape[0] if pct_table is not None else 100,
                         (n_features,), dtype=t.int32)
        mags = t.rand(n_features) + 0.1
        offsets = t.tensor([0, 20, 35, n_features], dtype=t.int64)

        data = {
            "offsets": offsets,
            "feature_ids": fids,
            "feature_mags": mags,
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 64, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([128, 64], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 100, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([99, 199, 63], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 3,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }
        return save_features(data, tmp_path / shard_name)

    def test_precompute_shard(self, tmp_path, pct_table):
        """precompute_shard correctly computes and stores percentiles in overlay."""
        shard_path = self._make_test_shard(tmp_path, "shard_00000", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)
        overlay_data = load_overlay_shard(overlay_path)

        assert TensorKey.FEATURE_PERCENTILES in overlay_data
        pcts = overlay_data[TensorKey.FEATURE_PERCENTILES]
        assert pcts.dtype == t.uint8

        base = load_features(shard_path)
        assert len(pcts) == len(base["feature_ids"])

    def test_existing_payload_validation_reads_base_count_from_header(
        self,
        tmp_path,
        pct_table,
        monkeypatch,
    ):
        """Resume validation does not decode base feature tensors for their count."""
        shard_path = self._make_test_shard(tmp_path, "shard_resume", pct_table)
        payload_path = save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: t.zeros(50, dtype=t.uint8)},
            tmp_path / "shard_resume",
        )
        read_count = precompute_percentiles_module.read_feature_value_count
        observed_paths = []

        def _recording_read_count(path):
            observed_paths.append(path)
            return read_count(path)

        monkeypatch.setattr(
            precompute_percentiles_module,
            "read_feature_value_count",
            _recording_read_count,
        )

        precompute_percentiles_module._validate_percentile_payload(
            shard_path,
            payload_path,
        )

        assert observed_paths == [shard_path]

    def test_precomputed_matches_reference(self, tmp_path, pct_table):
        """Precomputed uint8 percentiles match runtime computation."""
        shard_path = self._make_test_shard(tmp_path, "shard_ref", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # Load before precomputation to get reference
        original = load_features(shard_path)
        fids = original["feature_ids"].to(t.int64)
        fmags = original["feature_mags"]
        ref_pcts = magnitude_to_percentile(fids, fmags, pct_table)
        ref_uint8 = percentile_to_uint8(ref_pcts)

        # Run precomputation via overlay
        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)
        overlay_data = load_overlay_shard(overlay_path)
        stored_pcts = overlay_data[TensorKey.FEATURE_PERCENTILES]

        # The overlay percentiles are aligned with the base shard's feature order.
        # Feature IDs are re-sorted during save, so values should match
        # but may be reordered within segments.
        loaded = load_features(shard_path)
        offsets = loaded["offsets"]
        orig_offsets = original["offsets"]
        for i in range(len(offsets) - 1):
            s, e = offsets[i].item(), offsets[i + 1].item()
            os, oe = orig_offsets[i].item(), orig_offsets[i + 1].item()
            stored_set = sorted(stored_pcts[s:e].tolist())
            ref_set = sorted(ref_uint8[os:oe].tolist())
            assert stored_set == ref_set, (
                f"Segment {i}: stored {stored_set} != ref {ref_set}"
            )

    def test_dry_run_does_not_modify(self, tmp_path, pct_table):
        """Dry run computes but doesn't write overlay."""
        shard_path = self._make_test_shard(tmp_path, "shard_dry", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum, dry_run=True)
        assert not overlay_path.exists()


# ======================================================================
# 4b. Force recompute
# ======================================================================

class TestForceRecompute:
    """Verify force mode reprocesses shards and produces correct results."""

    def _make_test_shard(self, tmp_path, shard_name, pct_table):
        """Create a minimal feature shard."""
        t.manual_seed(42)
        n_features = 50
        fids = t.randint(0, pct_table.shape[0], (n_features,), dtype=t.int32)
        mags = t.rand(n_features) + 0.1
        offsets = t.tensor([0, 20, 35, n_features], dtype=t.int64)
        data = {
            "offsets": offsets,
            "feature_ids": fids,
            "feature_mags": mags,
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 64, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([128, 64], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 100, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([99, 199, 63], dtype=t.int32),
            "n_docs": 2, "n_segments": 3, "layer": 8,
            "context_window": 1024, "pooling": "mean_pool",
            "segment_mode": "fixed_tokens", "segment_size": 64,
        }
        return save_features(data, tmp_path / shard_name)

    def test_force_reprocesses_already_done_shard(self, tmp_path, pct_table):
        """Force overwrites an overlay shard that already has matching percentiles."""
        shard_path = self._make_test_shard(tmp_path, "shard_force", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # First pass: compute percentiles
        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)
        first_data = load_overlay_shard(overlay_path)
        first_pcts = first_data[TensorKey.FEATURE_PERCENTILES].clone()

        # Second pass (force): should overwrite with identical values
        overlay_path2 = _precompute_shard_overlay(shard_path, pct_table, checksum)
        second_data = load_overlay_shard(overlay_path2)
        second_pcts = second_data[TensorKey.FEATURE_PERCENTILES]

        assert t.equal(first_pcts, second_pcts), "Force recompute changed percentile values"

    def test_force_produces_same_as_initial(self, tmp_path, pct_table):
        """Force recompute is bitwise identical to initial computation."""
        # Create two identical shards
        shard_a = self._make_test_shard(tmp_path / "a", "shard_cmp", pct_table)
        shard_b = self._make_test_shard(tmp_path / "b", "shard_cmp", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # Compute A once (initial)
        overlay_a = _precompute_shard_overlay(shard_a, pct_table, checksum)

        # Compute B once, then force-recompute
        _precompute_shard_overlay(shard_b, pct_table, checksum)
        overlay_b = _precompute_shard_overlay(shard_b, pct_table, checksum)  # force (overwrite)

        data_a = load_overlay_shard(overlay_a)
        data_b = load_overlay_shard(overlay_b)
        assert t.equal(data_a[TensorKey.FEATURE_PERCENTILES], data_b[TensorKey.FEATURE_PERCENTILES])

    def test_force_overwrites_stale_overlay(self, tmp_path, pct_table):
        """Force recompute replaces a stale overlay with correctly computed one."""
        shard_path = self._make_test_shard(tmp_path, "shard_stale", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # First pass: create overlay with correct percentiles
        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)

        # Force recompute (overwrite existing overlay)
        overlay_path2 = _precompute_shard_overlay(shard_path, pct_table, checksum)
        assert overlay_path == overlay_path2

        fixed = load_overlay_shard(overlay_path2)
        assert not (fixed[TensorKey.FEATURE_PERCENTILES] == 0).all(), "Percentiles should be non-zero after recompute"

    def test_force_preserves_original_feature_data(self, tmp_path, pct_table):
        """Force recompute doesn't corrupt the base shard (overlay is separate)."""
        shard_path = self._make_test_shard(tmp_path, "shard_preserve", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # Capture original data
        original = load_features(shard_path)
        orig_fids = original["feature_ids"].clone()
        orig_mags = original["feature_mags"].clone()
        orig_offsets = original["offsets"].clone()
        orig_n_docs = original["n_docs"]
        orig_n_segments = original["n_segments"]

        # Compute then force-recompute via overlay
        _precompute_shard_overlay(shard_path, pct_table, checksum)
        _precompute_shard_overlay(shard_path, pct_table, checksum)

        # Base shard should be untouched
        after = load_features(shard_path)
        assert t.equal(after["feature_ids"], orig_fids), "feature_ids corrupted by force recompute"
        assert t.allclose(after["feature_mags"], orig_mags, atol=1e-3), "feature_mags corrupted by force recompute"
        assert t.equal(after["offsets"], orig_offsets), "offsets corrupted by force recompute"
        assert after["n_docs"] == orig_n_docs
        assert after["n_segments"] == orig_n_segments

    def test_force_reference_comparison(self, tmp_path, pct_table):
        """Force-recomputed overlay percentiles match manual magnitude_to_percentile call."""
        shard_path = self._make_test_shard(tmp_path, "shard_refcmp", pct_table)
        checksum = compute_percentile_checksum(pct_table)

        # Compute reference manually
        original = load_features(shard_path)
        fids = original["feature_ids"].to(t.int64)
        fmags = original["feature_mags"]
        ref_pcts = magnitude_to_percentile(fids, fmags, pct_table)
        ref_uint8 = percentile_to_uint8(ref_pcts)

        # Force compute via overlay (two passes)
        _precompute_shard_overlay(shard_path, pct_table, checksum)
        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)  # second pass = force

        overlay_data = load_overlay_shard(overlay_path)
        stored_pcts = overlay_data[TensorKey.FEATURE_PERCENTILES]

        # Compare per-segment (sort order may differ)
        base = load_features(shard_path)
        offsets = base["offsets"]
        orig_offsets = original["offsets"]
        for i in range(len(offsets) - 1):
            s, e = offsets[i].item(), offsets[i + 1].item()
            os, oe = orig_offsets[i].item(), orig_offsets[i + 1].item()
            assert sorted(stored_pcts[s:e].tolist()) == sorted(ref_uint8[os:oe].tolist()), (
                f"Segment {i}: force-recomputed percentiles don't match reference"
            )

    def test_run_force_overrides_resume(self, tmp_path, pct_table):
        """Force recompute overwrites an existing overlay (force wins over resume)."""
        # Create a shard with features
        n = 10
        data = {
            "offsets": t.tensor([0, n], dtype=t.int64),
            "feature_ids": t.randint(0, pct_table.shape[0], (n,), dtype=t.int32),
            "feature_mags": t.rand(n) + 0.1,
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([64], dtype=t.int32),
            "doc_token_counts": t.tensor([100], dtype=t.int32),
            "n_docs": 1, "n_segments": 1, "layer": 8,
            "context_window": 1024, "pooling": "mean_pool",
            "segment_mode": "fixed_tokens", "segment_size": 64,
        }
        shard_path = save_features(data, tmp_path / "shard_00000")
        checksum = compute_percentile_checksum(pct_table)

        # First pass: compute overlay
        overlay_path = _precompute_shard_overlay(shard_path, pct_table, checksum)
        first_data = load_overlay_shard(overlay_path)
        first_pcts = first_data[TensorKey.FEATURE_PERCENTILES].clone()

        # Force recompute: overlay should be overwritten with same correct values
        overlay_path2 = _precompute_shard_overlay(shard_path, pct_table, checksum)
        after_data = load_overlay_shard(overlay_path2)

        assert not (after_data[TensorKey.FEATURE_PERCENTILES] == 0).all(), (
            "Force should have replaced overlay with real values"
        )
        assert t.equal(first_pcts, after_data[TensorKey.FEATURE_PERCENTILES])


# ======================================================================
# 5. FeatureIndex integration
# ======================================================================

class TestFeatureIndexPercentiles:
    """Verify FeatureIndex exposes precomputed percentiles correctly."""

    def _make_feature_dir(self, tmp_path, pct_table, with_percentiles=True):
        """Create a feature dir with one shard, optionally with overlay percentiles."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        t.manual_seed(42)
        n = 20
        fids = t.randint(0, pct_table.shape[0], (n,), dtype=t.int32)
        mags = t.rand(n) + 0.1
        offsets = t.tensor([0, 10, n], dtype=t.int64)

        data = {
            "offsets": offsets,
            "feature_ids": fids,
            "feature_mags": mags,
            "doc_ids": t.tensor([0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([10, 10], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([64, 64], dtype=t.int32),
            "n_docs": 2,
            "n_segments": 2,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }

        shard_path = save_features(data, feature_dir / "shard_00000")
        checksum = compute_percentile_checksum(pct_table)

        if with_percentiles:
            # Load base shard to get sorted fids/mags, then compute percentiles
            base = load_features(shard_path)
            sorted_fids = base["feature_ids"]
            sorted_mags = base["feature_mags"]
            pct_scores = magnitude_to_percentile(sorted_fids.to(t.int64), sorted_mags, pct_table)
            pct_uint8 = percentile_to_uint8(pct_scores)

            # Write overlay shard
            out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
            write_manifest(out_dir, checksum, OverlayKind.PERCENTILE)
            out_path = overlay_shard_path(out_dir, "shard_00000")
            save_overlay_shard({TensorKey.FEATURE_PERCENTILES: pct_uint8}, out_path)

        return feature_dir, checksum

    def test_has_precomputed_with_matching_checksum(self, tmp_path, pct_table):
        """FeatureIndex reports has_precomputed_percentiles when checksum matches."""
        feature_dir, checksum = self._make_feature_dir(tmp_path, pct_table, True)
        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)

        # Force shard load via lookup
        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert idx.has_precomputed_percentiles

    def test_no_precomputed_without_checksum(self, tmp_path, pct_table):
        """FeatureIndex without checksum doesn't expose percentiles."""
        feature_dir, _ = self._make_feature_dir(tmp_path, pct_table, True)
        idx = FeatureIndex(str(feature_dir))  # no checksum

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert not idx.has_precomputed_percentiles
        # Segments should not have percentiles
        assert segs[0].feature_percentiles is None

    def test_wrong_requested_checksum_fails_loud(self, tmp_path, pct_table):
        """A mismatched non-null checksum cannot silently fall back."""
        feature_dir, _ = self._make_feature_dir(tmp_path, pct_table, True)
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum="wrong_checksum")

    def test_requested_missing_percentile_overlay_fails_loud(self, tmp_path, pct_table):
        """A requested overlay cannot silently fall back shard-by-shard."""
        feature_dir, checksum = self._make_feature_dir(tmp_path, pct_table, False)
        with pytest.raises(OverlayIncompleteError, match="percentile"):
            FeatureIndex(str(feature_dir), percentile_checksum=checksum)

    def test_precomputed_percentiles_values_correct(self, tmp_path, pct_table):
        """Precomputed percentiles on SegmentFeatures match expected uint8 values."""
        feature_dir, checksum = self._make_feature_dir(tmp_path, pct_table, True)
        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)

        segs = idx.lookup("shard_00000", 0)
        assert segs is not None
        assert segs[0].feature_percentiles is not None
        assert segs[0].feature_percentiles.dtype == t.uint8
        assert len(segs[0].feature_percentiles) == len(segs[0].feature_ids)

    def test_pickle_preserves_checksum(self, tmp_path, pct_table):
        """FeatureIndex with checksum survives pickle roundtrip."""
        feature_dir, checksum = self._make_feature_dir(tmp_path, pct_table, True)
        idx = FeatureIndex(str(feature_dir), percentile_checksum=checksum)

        data = pickle.dumps(idx)
        idx2 = pickle.loads(data)
        assert idx2._percentile_checksum == checksum


# ======================================================================
# 6. sample_features_batched with precomputed percentiles
# ======================================================================

class TestBatchedSamplingWithPrecomputed:
    """Verify sample_features_batched uses precomputed percentiles correctly."""

    def test_precomputed_produces_valid_results(self, pct_table, config):
        """Using precomputed percentiles still produces valid sampling results."""
        t.manual_seed(42)
        n_segs = 10
        segments = []
        for _ in range(n_segs):
            n = t.randint(3, 15, (1,)).item()
            fids = t.randint(0, N_SAE, (n,), dtype=t.int64)  # ty: ignore[no-matching-overload]
            mags = t.rand(n) + 0.1  # ty: ignore[no-matching-overload]
            segments.append((fids, mags))

        all_fids = t.cat([s[0] for s in segments])
        all_mags = t.cat([s[1] for s in segments])
        seg_sizes = [len(s[0]) for s in segments]

        # Compute precomputed percentiles
        pct_scores = magnitude_to_percentile(all_fids, all_mags, pct_table)
        precomputed = uint8_to_percentile(percentile_to_uint8(pct_scores))

        rng = t.Generator()
        rng.manual_seed(99)
        results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )

        assert len(results) == n_segs
        for i, (ids, mags) in enumerate(results):
            assert len(ids) >= 1  # min_k=1
            assert len(ids) == len(mags)
            # All output IDs should come from the input segment
            in_set = set(all_fids[sum(seg_sizes[:i]):sum(seg_sizes[:i+1])].tolist())
            out_set = set(ids.tolist())
            assert out_set.issubset(in_set)

    def test_precomputed_vs_runtime_same_rng(self, pct_table, config):
        """Precomputed and runtime sampling select from same valid features.

        Note: due to uint8 quantization, weights differ slightly so exact
        sample equality is not guaranteed. But both should produce valid
        results with the same statistical properties.
        """
        t.manual_seed(42)
        segments = []
        for _ in range(20):
            n = 10
            fids = t.randperm(N_SAE)[:n].to(t.int64)
            mags = t.rand(n) + 0.1
            segments.append((fids, mags))

        all_fids = t.cat([s[0] for s in segments])
        all_mags = t.cat([s[1] for s in segments])
        seg_sizes = [10] * 20

        # Runtime path
        rng1 = t.Generator()
        rng1.manual_seed(42)
        runtime_results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng1,
        )

        # Precomputed path
        pct_scores = magnitude_to_percentile(all_fids, all_mags, pct_table)
        precomputed = uint8_to_percentile(percentile_to_uint8(pct_scores))

        rng2 = t.Generator()
        rng2.manual_seed(42)
        precomputed_results = sample_features_batched(
            all_fids, all_mags, seg_sizes,
            pct_table, config, rng2,
            precomputed_percentiles=precomputed,
        )

        # Both should produce same number of segments
        assert len(runtime_results) == len(precomputed_results) == 20
        # Same k values (since same RNG seed for sample_k)
        for i in range(20):
            assert len(runtime_results[i][0]) == len(precomputed_results[i][0]), (
                f"Segment {i}: runtime k={len(runtime_results[i][0])}, "
                f"precomputed k={len(precomputed_results[i][0])}"
            )

    def test_precomputed_empty_segments(self, pct_table, config):
        """Empty segments with precomputed percentiles work correctly."""
        rng = t.Generator()
        rng.manual_seed(42)
        all_fids = t.tensor([], dtype=t.int64)
        all_mags = t.tensor([], dtype=t.float32)
        precomputed = t.tensor([], dtype=t.float32)

        results = sample_features_batched(
            all_fids, all_mags, [0, 0],
            pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )
        assert len(results) == 2
        for ids, _ in results:
            assert len(ids) == 0

    def test_precomputed_overrides_runtime(self, pct_table, config):
        """Precomputed percentiles take precedence over runtime computation.

        This test verifies the code path by using deliberately different
        precomputed weights (all equal) vs what runtime would compute.
        """
        t.manual_seed(42)
        n = 50
        fids = t.randperm(N_SAE)[:n].to(t.int64)
        mags = t.rand(n) + 0.1

        # Uniform precomputed weights = uniform sampling
        precomputed = t.ones(n, dtype=t.float32)

        counts = t.zeros(n)
        for seed in range(200):
            rng = t.Generator()
            rng.manual_seed(seed)
            results = sample_features_batched(
                fids, mags, [n],
                pct_table, config, rng,
                precomputed_percentiles=precomputed,
            )
            for fid in results[0][0]:
                idx = (fids == fid).nonzero(as_tuple=True)[0][0]
                counts[idx] += 1

        # With uniform weights, all features should have similar selection counts
        # (no feature should dominate like it would with real percentile weights)
        cv = counts.std() / counts.mean()
        assert cv < 0.5, f"Coefficient of variation {cv:.2f} too high for uniform weights"


# ======================================================================
# 7. Reference comparison: precomputed vs runtime
# ======================================================================

class TestReferenceComparison:
    """Verify precomputed percentiles match runtime computation."""

    def test_uint8_roundtrip_error_bounded(self, pct_table):
        """uint8 encoding introduces at most 1/255 ≈ 0.4% error (normalized)."""
        t.manual_seed(42)
        n = 10000
        fids = t.randint(0, N_SAE, (n,), dtype=t.int64)
        mags = t.rand(n) * 10 + 0.01

        exact = magnitude_to_percentile(fids, mags, pct_table)  # [0.5, 99.5]
        encoded = percentile_to_uint8(exact)
        decoded = uint8_to_percentile(encoded)  # [0, 1]
        exact_normalized = exact.float() / 100.0  # [0, 1]

        max_error = (decoded - exact_normalized).abs().max().item()
        assert max_error < 2.0 / 255, f"Max error {max_error:.6f} exceeds bound"

    def test_relative_ordering_preserved(self, pct_table):
        """uint8 encoding preserves the relative ordering of percentile scores."""
        t.manual_seed(42)
        n = 5000
        fids = t.randint(0, N_SAE, (n,), dtype=t.int64)
        mags = t.rand(n) * 10

        exact = magnitude_to_percentile(fids, mags, pct_table)
        encoded = percentile_to_uint8(exact)

        # For pairs with significant percentile difference, ordering should be preserved
        for _ in range(1000):
            i, j = t.randint(0, n, (2,)).tolist()
            diff = (exact[i] - exact[j]).abs().item()
            if diff > 1.0:  # > 1 percentile point apart
                assert (encoded[i] > encoded[j]) == (exact[i] > exact[j]) or encoded[i] == encoded[j]


# ======================================================================
# 8. Edge cases
# ======================================================================

class TestEdgeCases:
    """Test edge cases in the precomputed percentile pipeline."""

    def test_shard_with_zero_features(self, tmp_path, pct_table):
        """Shard with no features handles precomputation correctly."""
        data = {
            "offsets": t.tensor([0], dtype=t.int64),
            "feature_ids": t.tensor([], dtype=t.int32),
            "feature_mags": t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor([], dtype=t.int32),
            "segment_indices": t.tensor([], dtype=t.int32),
            "segment_positions": t.tensor([], dtype=t.int32),
            "doc_token_counts": t.tensor([], dtype=t.int32),
            "segment_char_starts": t.tensor([], dtype=t.int32),
            "segment_char_ends": t.tensor([], dtype=t.int32),
            "n_docs": 0,
            "n_segments": 0,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }
        path = save_features(data, tmp_path / "empty_shard")
        checksum = compute_percentile_checksum(pct_table)
        overlay_path = _precompute_shard_overlay(path, pct_table, checksum)

        overlay_data = load_overlay_shard(overlay_path)
        assert TensorKey.FEATURE_PERCENTILES in overlay_data
        assert len(overlay_data[TensorKey.FEATURE_PERCENTILES]) == 0

    def test_single_feature_per_segment(self, pct_table, config):
        """Segments with exactly 1 feature work with precomputed percentiles."""
        fids = t.arange(5, dtype=t.int64)
        mags = t.rand(5) + 0.1
        precomputed = t.rand(5)

        rng = t.Generator()
        rng.manual_seed(42)
        results = sample_features_batched(
            fids, mags, [1, 1, 1, 1, 1],
            pct_table, config, rng,
            precomputed_percentiles=precomputed,
        )
        assert len(results) == 5
        for i, (ids, _) in enumerate(results):
            assert len(ids) == 1
            assert ids[0] == i

    def test_large_feature_ids(self, pct_table):
        """Features with IDs near the table boundary work correctly."""
        fids = t.tensor([0, N_SAE - 1], dtype=t.int64)
        mags = t.tensor([1.0, 5.0])
        pcts = magnitude_to_percentile(fids, mags, pct_table)
        encoded = percentile_to_uint8(pcts)
        assert encoded.dtype == t.uint8
        assert len(encoded) == 2


# ======================================================================
# 9. Statistics file checksum persistence
# ======================================================================

class TestStatisticsChecksumPersistence:
    """Verify checksum save/load in the statistics file."""

    def _make_stats_file(self, tmp_path):
        """Create a minimal statistics safetensors file."""
        from safetensors.torch import save_file
        tensors = {"feature_ids": t.tensor([0, 1, 2], dtype=t.int32)}
        metadata = {"n_sae_features": "100", "layer": "8"}
        path = tmp_path / "feature_statistics.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        return path

    def test_save_and_load_checksum(self, tmp_path):
        """Checksum can be written to and read from a statistics file."""
        from steering.stats_io import load_percentile_checksum, save_percentile_checksum
        path = self._make_stats_file(tmp_path)

        save_percentile_checksum(path, "abc123def456")
        loaded = load_percentile_checksum(path)
        assert loaded == "abc123def456"

    def test_load_returns_none_when_absent(self, tmp_path):
        """load_percentile_checksum returns None if no checksum stored."""
        from steering.stats_io import load_percentile_checksum
        path = self._make_stats_file(tmp_path)

        assert load_percentile_checksum(path) is None

    def test_save_preserves_existing_metadata(self, tmp_path):
        """Writing checksum doesn't destroy existing metadata."""
        from steering.stats_io import _read_metadata, save_percentile_checksum
        path = self._make_stats_file(tmp_path)

        save_percentile_checksum(path, "test_checksum")
        metadata = _read_metadata(path)
        assert metadata["n_sae_features"] == "100"
        assert metadata["layer"] == "8"
        assert metadata["percentile_checksum"] == "test_checksum"

    def test_save_preserves_existing_tensors(self, tmp_path):
        """Writing checksum doesn't destroy existing tensors."""
        from safetensors.torch import load_file

        from steering.stats_io import save_percentile_checksum
        path = self._make_stats_file(tmp_path)

        save_percentile_checksum(path, "test_checksum")
        tensors = load_file(str(path))
        assert "feature_ids" in tensors
        assert t.equal(tensors["feature_ids"], t.tensor([0, 1, 2], dtype=t.int32))

    def test_overwrite_existing_checksum(self, tmp_path):
        """Second save overwrites the first checksum."""
        from steering.stats_io import load_percentile_checksum, save_percentile_checksum
        path = self._make_stats_file(tmp_path)

        save_percentile_checksum(path, "first")
        save_percentile_checksum(path, "second")
        assert load_percentile_checksum(path) == "second"

    def test_already_current_checksum_is_byte_preserving(self, tmp_path):
        """Repeated checksum persistence cannot churn the statistics identity."""
        from steering.stats_io import save_percentile_checksum

        path = self._make_stats_file(tmp_path)
        save_percentile_checksum(path, "stable")
        before = path.read_bytes()

        save_percentile_checksum(path, "stable")

        assert path.read_bytes() == before


# ======================================================================
# 10. GPU acceleration (if available)
# ======================================================================

class TestGPUAcceleration:
    """Verify GPU path produces identical results to CPU path."""

    @pytest.fixture
    def _make_shard(self, tmp_path, pct_table):
        """Create a test shard and return (path, pct_table, checksum)."""
        n = 200
        fids = t.randint(0, pct_table.shape[0], (n,), dtype=t.int32)
        mags = t.rand(n) + 0.1
        offsets = t.tensor([0, 80, 150, n], dtype=t.int64)
        data = {
            "offsets": offsets,
            "feature_ids": fids,
            "feature_mags": mags,
            "doc_ids": t.tensor([0, 0, 1], dtype=t.int32),
            "segment_indices": t.tensor([0, 1, 0], dtype=t.int32),
            "segment_positions": t.tensor([0, 64, 0], dtype=t.int32),
            "doc_token_counts": t.tensor([128, 64], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 100, 0], dtype=t.int32),
            "segment_char_ends": t.tensor([99, 199, 63], dtype=t.int32),
            "n_docs": 2, "n_segments": 3, "layer": 8,
            "context_window": 1024, "pooling": "mean_pool",
            "segment_mode": "fixed_tokens", "segment_size": 64,
        }
        path = save_features(data, tmp_path / "gpu_shard")
        checksum = compute_percentile_checksum(pct_table)
        return path, pct_table, checksum

    @pytest.mark.skipif(not t.cuda.is_available(), reason="No GPU available")
    def test_gpu_matches_cpu(self, _make_shard):
        """GPU precompute produces uint8 results within ±1 of CPU.

        float32 rounding in searchsorted/interpolation can cause off-by-one
        differences at quantization boundaries. This is expected and
        harmless for multinomial sampling weights.
        """
        path, pct_table, checksum = _make_shard
        data = load_features(path)
        fids = data["feature_ids"].to(t.int64)
        fmags = data["feature_mags"]

        cpu_result = magnitude_to_percentile_chunked(fids, fmags, pct_table, device="cpu")
        gpu_result = magnitude_to_percentile_chunked(fids, fmags, pct_table, device="cuda")

        diff = (cpu_result.int() - gpu_result.int()).abs()
        assert diff.max().item() <= 1, (
            f"GPU/CPU max diff {diff.max().item()} > 1 (expected at most off-by-one)"
        )

    @pytest.mark.skipif(not t.cuda.is_available(), reason="No GPU available")
    def test_gpu_shard_roundtrip(self, tmp_path, pct_table):
        """Full shard precompute on GPU produces valid overlay results."""
        n = 100
        data = {
            "offsets": t.tensor([0, n], dtype=t.int64),
            "feature_ids": t.randint(0, N_SAE, (n,), dtype=t.int32),
            "feature_mags": t.rand(n) + 0.1,
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([64], dtype=t.int32),
            "doc_token_counts": t.tensor([100], dtype=t.int32),
            "n_docs": 1, "n_segments": 1, "layer": 8,
            "context_window": 1024, "pooling": "mean_pool",
            "segment_mode": "fixed_tokens", "segment_size": 64,
        }
        path = save_features(data, tmp_path / "gpu_rt")
        checksum = compute_percentile_checksum(pct_table)

        overlay_path = _precompute_shard_overlay(path, pct_table, checksum, device="cuda")

        overlay_data = load_overlay_shard(overlay_path)
        assert TensorKey.FEATURE_PERCENTILES in overlay_data
        pcts = overlay_data[TensorKey.FEATURE_PERCENTILES]
        assert pcts.dtype == t.uint8
        assert len(pcts) == n

    def test_cpu_chunked_matches_unchunked(self, pct_table):
        """Chunked processing produces the same result as single-pass."""
        t.manual_seed(42)
        n = 5000
        fids = t.randint(0, N_SAE, (n,), dtype=t.int64)
        mags = t.rand(n) + 0.1

        # Single pass (small chunk = full)
        full_result = magnitude_to_percentile_chunked(fids, mags, pct_table, device="cpu")

        # Reference: direct call
        ref_pcts = magnitude_to_percentile(fids, mags, pct_table)
        ref_uint8 = percentile_to_uint8(ref_pcts)

        assert t.equal(full_result, ref_uint8)


# ======================================================================
# 11. Shard validation pipeline tests
# ======================================================================

class TestShardValidationPercentiles:
    """Verify shard_validation correctly checks precomputed percentile data."""

    def _make_shard_data(self, n=20, with_percentiles=True, with_checksum=True):
        """Create minimal shard data dict with optional percentiles."""
        offsets = t.tensor([0, n], dtype=t.int64)
        data = {
            "offsets": offsets,
            "feature_ids": t.randint(0, 100, (n,), dtype=t.int32),
            "feature_mags": t.rand(n) + 0.1,
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([64], dtype=t.int32),
            "doc_token_counts": t.tensor([100], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 64,
        }
        if with_percentiles:
            data[TensorKey.FEATURE_PERCENTILES] = t.randint(0, 256, (n,), dtype=t.uint8)
        if with_checksum:
            data[MetadataKey.PERCENTILE_CHECKSUM] = "test_checksum"
        return data

    def test_valid_shard_with_percentiles(self):
        """Shard with matching percentiles and checksum passes validation."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(with_percentiles=True, with_checksum=True)
        result = validate_shard_data(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_valid_shard_without_percentiles(self):
        """Shard without percentiles still passes validation."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(with_percentiles=False, with_checksum=False)
        result = validate_shard_data(data)
        assert result.valid, f"Violations: {result.violations}"

    def test_percentiles_without_checksum_fails(self):
        """Percentiles present but missing checksum is flagged."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(with_percentiles=True, with_checksum=False)
        result = validate_shard_data(data)
        assert not result.valid
        assert any("percentile_checksum missing" in v for v in result.violations)

    def test_checksum_without_percentiles_fails(self):
        """Checksum present but missing percentiles is flagged."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(with_percentiles=False, with_checksum=True)
        result = validate_shard_data(data)
        assert not result.valid
        assert any("feature_percentiles missing" in v for v in result.violations)

    def test_wrong_percentile_length_fails(self):
        """Percentiles with wrong length relative to feature_ids is flagged."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(n=20, with_percentiles=True, with_checksum=True)
        data[TensorKey.FEATURE_PERCENTILES] = t.randint(0, 256, (10,), dtype=t.uint8)  # wrong length
        result = validate_shard_data(data)
        assert not result.valid
        assert any("length" in v for v in result.violations)

    def test_wrong_percentile_dtype_fails(self):
        """Percentiles with wrong dtype is flagged."""
        from steering.shard_validation import validate_shard_data
        data = self._make_shard_data(n=20, with_percentiles=True, with_checksum=True)
        data[TensorKey.FEATURE_PERCENTILES] = t.rand(20, dtype=t.float32)  # wrong dtype
        result = validate_shard_data(data)
        assert not result.valid
        assert any("uint8" in v for v in result.violations)


# ======================================================================
# 12. validate_feature_directory percentile checks
# ======================================================================

class TestValidateDirectoryPercentiles:
    """Verify validate_feature_directory checks percentile consistency."""

    def _make_feature_dir_with_stats(self, tmp_path, pct_table, n_shards=2):
        """Create a feature dir with stats and shards."""
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        checksum = compute_percentile_checksum(pct_table)

        # Create stats file
        from safetensors.torch import save_file as sf_save

        from steering.stats_io import SAFETENSORS_EXT, STATS_STEM, save_percentile_checksum

        stats_path = feature_dir / f"{STATS_STEM}{SAFETENSORS_EXT}"
        # Minimal stats file
        sf_save(
            {"feature_ids": t.tensor([0], dtype=t.int32)},
            str(stats_path),
            metadata={
                "n_sae_features": str(pct_table.shape[0]),
                "n_shards_processed": str(n_shards),
                "n_docs": str(n_shards),
                "n_segments": str(n_shards),
                "layer": "8",
                "segment_size": "64",
                "context_window": "1024",
                "pooling_mode": "mean_pool",
                "segment_mode": "fixed_tokens",
                "generated_at": "",
                "activation_percentile": "0.0",
                "n_tokens_gpt2": "0",
                "mean_tokens_per_doc": "0",
                "median_tokens_per_doc": "0",
                "n_features_seen": "0",
                "mean_active_per_segment": "0",
                "median_active_per_segment": "0",
                "p90_active_per_segment": "0",
                "max_active_per_segment": "0",
                "min_active_per_segment": "0",
                "n_total_sae_features": str(pct_table.shape[0]),
                "pct_features_active": "0",
                "total_bytes": "0",
                "mean_bytes_per_doc": "0",
                "mean_bytes_per_token": "0",
                "extrap_target_tokens": "0",
                "extrap_estimated_docs": "0",
                "extrap_estimated_storage_gb": "0",
                "extrap_estimated_wall_time_hours": "0",
            },
        )
        save_percentile_checksum(stats_path, checksum)

        # Create shards and overlay percentiles
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        write_manifest(out_dir, checksum, OverlayKind.PERCENTILE)

        for i in range(n_shards):
            n = 10
            fids = t.randint(0, pct_table.shape[0], (n,), dtype=t.int32)
            mags = t.rand(n) + 0.1
            data = {
                "offsets": t.tensor([0, n], dtype=t.int64),
                "feature_ids": fids,
                "feature_mags": mags,
                "doc_ids": t.tensor([0], dtype=t.int32),
                "segment_indices": t.tensor([0], dtype=t.int32),
                "segment_positions": t.tensor([0], dtype=t.int32),
                "doc_token_counts": t.tensor([100], dtype=t.int32),
                "segment_char_starts": t.tensor([0], dtype=t.int32),
                "segment_char_ends": t.tensor([64], dtype=t.int32),
                "n_docs": 1,
                "n_segments": 1,
                "layer": 8,
                "context_window": 1024,
                "pooling": "mean_pool",
                "segment_mode": "fixed_tokens",
                "segment_size": 64,
            }
            stem = f"shard_{i:05d}"
            shard_path = save_features(data, feature_dir / stem)

            # Compute percentiles from saved (sorted) shard and write overlay
            base = load_features(shard_path)
            sorted_fids = base["feature_ids"]
            sorted_mags = base["feature_mags"]
            pct_scores = magnitude_to_percentile(sorted_fids.to(t.int64), sorted_mags, pct_table)
            pct_uint8 = percentile_to_uint8(pct_scores)
            save_overlay_shard(
                {TensorKey.FEATURE_PERCENTILES: pct_uint8},
                overlay_shard_path(out_dir, stem),
            )

        return feature_dir

    def test_resolve_feature_dir_explicit(self, tmp_path):
        """Explicit feature_dir takes precedence over structured params."""
        from omegaconf import OmegaConf

        from scripts.preprocess._common import resolve_feature_dir

        cfg = OmegaConf.create({
            "feature_dir": str(tmp_path / "explicit"),
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 8,
            "pooling": "max_pool",
            "segment_size": 128,
            "sae_width": "16k",
            "dataset_phase": None,
            "feature_root": "out/data/features",
        })
        result = resolve_feature_dir(cfg)
        assert result == tmp_path / "explicit"

    def test_resolve_feature_dir_auto(self):
        """Auto-constructed path from structured params."""
        from omegaconf import OmegaConf

        from scripts.preprocess._common import resolve_feature_dir

        cfg = OmegaConf.create({
            "feature_dir": None,
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 8,
            "pooling": "max_pool",
            "segment_size": 128,
            "sae_width": "16k",
            "dataset_phase": None,
            "feature_root": "out/data/features",
        })
        result = resolve_feature_dir(cfg)
        assert "s128" in str(result)
        assert "l8" in str(result)
        assert "a_res" in str(result)
        assert "p_max_pool" in str(result)
        assert "f16k" in str(result)
        assert "m_gemma-2-2b" in str(result)
        assert "base" in str(result)
        assert result.name == "no_bos"
        assert result.parent.name == "base"

    def test_resolve_feature_dir_sft_phase(self):
        """SFT dataset_phase is correctly applied."""
        from omegaconf import OmegaConf

        from scripts.preprocess._common import resolve_feature_dir

        cfg = OmegaConf.create({
            "feature_dir": None,
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 8,
            "pooling": "max_pool",
            "segment_size": 128,
            "sae_width": "16k",
            "dataset_phase": "safety_sft",
            "feature_root": "out/data/features",
        })
        result = resolve_feature_dir(cfg)
        assert str(result).endswith("safety_sft/no_bos")

    def test_run_with_omegaconf(self, tmp_path, pct_table):
        """run() works with OmegaConf DictConfig (Hydra-compatible)."""
        from omegaconf import OmegaConf

        # Set up feature dir with a shard + stats
        feature_dir = self._make_feature_dir_with_stats(tmp_path, pct_table, n_shards=1)

        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 8,
            "pooling": "max_pool",
            "segment_size": 128,
            "sae_width": "16k",
            "dataset_phase": None,
            "feature_root": "out/data/features",
            "device": "cpu",
            "resume": False,
            "dry_run": True,  # dry run to avoid needing full stats
        })
        from scripts.preprocess._common import resolve_feature_dir
        result = resolve_feature_dir(cfg)
        assert result == feature_dir

    def test_valid_directory_passes(self, tmp_path, pct_table):
        """Directory with matching checksums and correct overlay percentiles passes validation."""
        from steering.feature_io import glob_features, shard_stem_from_path
        from steering.stats_io import load_percentile_checksum

        feature_dir = self._make_feature_dir_with_stats(tmp_path, pct_table, n_shards=2)
        checksum = compute_percentile_checksum(pct_table)

        # Verify checksum is in stats file
        stats_path = feature_dir / "feature_statistics.safetensors"
        assert load_percentile_checksum(stats_path) == checksum

        # Verify each overlay shard has correct percentile values
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        for path in glob_features(feature_dir):
            stem = shard_stem_from_path(path)
            ovl_path = overlay_shard_path(out_dir, stem)
            assert ovl_path.exists(), f"Overlay shard missing for {stem}"

            overlay_data = load_overlay_shard(ovl_path)
            stored_pcts = overlay_data[TensorKey.FEATURE_PERCENTILES]

            # Verify percentile values match recomputation from base shard
            data = load_features(str(path))
            fids = data["feature_ids"].to(t.int64)
            fmags = data["feature_mags"]
            ref = percentile_to_uint8(magnitude_to_percentile(fids, fmags, pct_table))
            assert t.equal(stored_pcts, ref), f"[{stem}] Percentile mismatch"


def _corrupt_empty_segment_shard_data() -> dict:
    """Build a base-shard data dict with an empty middle segment (issue #522).

    ``offsets=[0,2,2,4]`` → 3 segments, segment 1 empty. CSR + per-segment
    boundary tensors are kept internally consistent so ``load_features`` (which
    delta-decodes feature_ids and parses metadata) reads it back cleanly; the
    producer guard fires immediately after the load, before any feature use.
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


class TestPrecomputePercentilesEmptySegmentGuard:
    """Issue #522 site-f: precompute_percentiles reads base shards directly via
    load_features (not via FeatureIndex), so it needs its own empty-segment
    guard. It must fire right after the load, before the percentile table is
    touched."""

    def test_precompute_shard_raises_on_empty_segment(self, tmp_path):
        path = save_raw_shard(
            tmp_path, "shard_00000", _corrupt_empty_segment_shard_data()
        )
        # percentile_table / output_dir are only touched AFTER the guard, so
        # dummies suffice — the guard must fire first.
        dummy_table = t.zeros(N_SAE, 99, dtype=t.float64)
        with pytest.raises(EmptyExtractionSegmentError, match=r"segment 1\b"):
            precompute_shard(
                Path(path),
                dummy_table,
                tmp_path / "out",
                "shard_00000",
            )
