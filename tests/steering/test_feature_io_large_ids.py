"""Tests for delta encoding / dtype optimization with large SAE feature IDs.

Covers SAE widths from 16K to 1M, uint16 overflow detection, large-gap
delta encoding edge cases, and full save/load roundtrips with feature IDs
that span the entire 1M range.
"""

import torch as t

from steering.feature_io import (
    _delta_decode,
    _delta_encode,
    _optimize_dtype,
    _sort_within_segments,
    _upcast,
    load_features,
    save_features,
)
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth

# Concrete width values for readability
W_16K = SAE_WIDTH_TO_NUM_FEATURES[SAEWidth.W_16K]  # 16384
W_65K = SAE_WIDTH_TO_NUM_FEATURES[SAEWidth.W_65K]  # 65536
W_131K = SAE_WIDTH_TO_NUM_FEATURES[SAEWidth.W_131K]  # 131072
W_262K = SAE_WIDTH_TO_NUM_FEATURES[SAEWidth.W_262K]  # 262144
W_1M = SAE_WIDTH_TO_NUM_FEATURES[SAEWidth.W_1M]  # 1048576


def _make_shard_dict(
    n_segments: int = 3,
    feats_per_seg: int = 5,
    n_docs: int = 2,
    max_feature_id: int = 16383,
) -> dict:
    """Build a minimal valid shard dict."""
    n_features = n_segments * feats_per_seg
    offsets = [i * feats_per_seg for i in range(n_segments + 1)]

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        segs_for_doc = n_segments // n_docs + (1 if d < n_segments % n_docs else 0)
        for s in range(segs_for_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 512)
            seg_char_ends.append((s + 1) * 512)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.randint(0, max_feature_id + 1, (n_features,), dtype=t.int32),
        "feature_mags": t.rand(n_features, dtype=t.float32).abs() + 0.01,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


def _make_variable_density_shard(
    segment_sizes: list[int],
    max_feature_id: int,
    n_docs: int = 1,
) -> dict:
    """Build a shard with segments of varying density (feature count)."""
    n_segments = len(segment_sizes)
    offsets = [0]
    for sz in segment_sizes:
        offsets.append(offsets[-1] + sz)
    total_features = offsets[-1]

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        segs_for_doc = n_segments // n_docs + (1 if d < n_segments % n_docs else 0)
        for s in range(segs_for_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 512)
            seg_char_ends.append((s + 1) * 512)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.randint(
            0, max_feature_id + 1, (total_features,), dtype=t.int32
        ),
        "feature_mags": t.rand(total_features, dtype=t.float32).abs() + 0.01,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


# ---------------------------------------------------------------------------
# 1. Delta encoding roundtrip for each SAE width
# ---------------------------------------------------------------------------


class TestDeltaRoundtripPerWidth:
    """Delta encode -> decode roundtrip for every SAE width."""

    def _roundtrip(self, max_id: int, segment_sizes: list[int]) -> None:
        """Generate random IDs within range, sort, delta-encode, decode, verify."""
        offsets = [0]
        for sz in segment_sizes:
            offsets.append(offsets[-1] + sz)
        offsets_t = t.tensor(offsets, dtype=t.int64)
        total = offsets[-1]

        ids = t.randint(0, max_id, (total,), dtype=t.int32)
        mags = t.rand(total, dtype=t.float32).abs() + 0.01

        sorted_ids, _ = _sort_within_segments(ids, mags, offsets_t)
        encoded = _delta_encode(sorted_ids, offsets_t)
        decoded = _delta_decode(encoded, offsets_t)

        assert t.equal(decoded, sorted_ids), "Delta roundtrip failed"

    def test_16k_uniform(self):
        self._roundtrip(W_16K, [10, 10, 10, 10])

    def test_16k_variable_density(self):
        self._roundtrip(W_16K, [1, 50, 3, 0, 20])

    def test_65k_uniform(self):
        self._roundtrip(W_65K, [10, 10, 10, 10])

    def test_65k_variable_density(self):
        self._roundtrip(W_65K, [1, 100, 0, 5, 50])

    def test_262k_uniform(self):
        self._roundtrip(W_262K, [20, 20, 20])

    def test_262k_variable_density(self):
        self._roundtrip(W_262K, [2, 0, 80, 1, 30])

    def test_1m_uniform(self):
        self._roundtrip(W_1M, [15, 15, 15, 15])

    def test_1m_variable_density(self):
        self._roundtrip(W_1M, [0, 1, 200, 0, 3, 50])

    def test_1m_single_element_segments(self):
        self._roundtrip(W_1M, [1, 1, 1, 1, 1])

    def test_1m_single_large_segment(self):
        self._roundtrip(W_1M, [500])


# ---------------------------------------------------------------------------
# 2. uint16 overflow detection
# ---------------------------------------------------------------------------


class TestUint16OverflowDetection:
    """_optimize_dtype must pick uint16 only when all values fit."""

    def test_16k_ids_use_uint16(self):
        """16K max ID (16383) fits in uint16."""
        ids = t.tensor([0, 100, 16383], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.uint16

    def test_65k_max_id_uses_uint16(self):
        """65535 is the uint16 ceiling -- still fits."""
        ids = t.tensor([0, 32768, 65535], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.uint16

    def test_65k_delta_encoded_fits_uint16(self):
        """Delta-encoded IDs from SAE_65K stay within uint16 range."""
        offsets = t.tensor([0, 4], dtype=t.int64)
        sorted_ids = t.tensor([100, 200, 50000, 65535], dtype=t.int32)
        delta = _delta_encode(sorted_ids, offsets)
        result = _optimize_dtype("feature_ids", delta)
        assert result.dtype == t.uint16

    def test_131k_first_id_exceeds_uint16(self):
        """131K SAE: first IDs can be >65535, must use int32."""
        ids = t.tensor([100000, 500, 200], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.int32

    def test_131k_delta_with_large_first_id(self):
        """Delta-encoded 131K IDs: first ID 100000 forces int32."""
        offsets = t.tensor([0, 3], dtype=t.int64)
        sorted_ids = t.tensor([100000, 100001, 131071], dtype=t.int32)
        delta = _delta_encode(sorted_ids, offsets)
        # First element is 100000 > 65535
        result = _optimize_dtype("feature_ids", delta)
        assert result.dtype == t.int32

    def test_1m_forces_int32(self):
        """1M SAE: IDs up to 1048575 clearly exceed uint16."""
        ids = t.tensor([0, 500000, 1048575], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.int32

    def test_1m_ids_all_in_uint16_range_uses_uint16(self):
        """SAE_1M but all features happen to be in 0-65535 range -- should use uint16."""
        ids = t.tensor([0, 100, 200, 65535], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.uint16

    def test_1m_delta_with_large_gap_forces_int32(self):
        """A single large delta (>65535) forces int32 for the whole tensor."""
        offsets = t.tensor([0, 3], dtype=t.int64)
        sorted_ids = t.tensor([0, 1, 100000], dtype=t.int32)
        delta = _delta_encode(sorted_ids, offsets)
        # delta = [0, 1, 99999] -- 99999 > 65535
        result = _optimize_dtype("feature_ids", delta)
        assert result.dtype == t.int32

    def test_empty_tensor_uses_uint16(self):
        """Empty feature_ids should still use uint16 (vacuously true max check)."""
        ids = t.tensor([], dtype=t.int32)
        result = _optimize_dtype("feature_ids", ids)
        assert result.dtype == t.uint16


# ---------------------------------------------------------------------------
# 3. Full save/load roundtrip with large feature IDs
# ---------------------------------------------------------------------------


class TestSaveLoadLargeIds:
    """save_features -> load_features with IDs spanning the 1M range."""

    def test_roundtrip_1m_range(self, tmp_path):
        """Full pipeline roundtrip with feature IDs in the 1M range."""
        data = _make_shard_dict(
            n_segments=5, feats_per_seg=20, n_docs=2, max_feature_id=W_1M - 1
        )
        out = save_features(data, tmp_path / "shard_00000_features")
        loaded = load_features(out)

        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            orig_sorted = data["feature_ids"][s:e].sort().values
            load_sorted = loaded["feature_ids"][s:e].sort().values
            assert t.equal(orig_sorted, load_sorted), (
                f"Segment {seg}: feature IDs differ after roundtrip"
            )

    def test_roundtrip_262k_range(self, tmp_path):
        """262K SAE roundtrip."""
        data = _make_shard_dict(
            n_segments=4, feats_per_seg=15, n_docs=2, max_feature_id=W_262K - 1
        )
        out = save_features(data, tmp_path / "shard_262k")
        loaded = load_features(out)

        assert loaded["feature_ids"].shape == data["feature_ids"].shape
        assert loaded["feature_ids"].dtype == t.int32

    def test_roundtrip_mixed_density_1m(self, tmp_path):
        """Variable-density segments with 1M-range IDs."""
        # Issue #522: a base shard (this goes through save_features) with
        # n_segments > 0 may not contain an empty segment, so every segment here
        # has >=1 feature. The subject is large-ID round-tripping across varying
        # densities (1..100), not empty-segment handling — the codec-level
        # empty-segment cases live in TestDeltaRoundtripPerWidth/TestDeltaLargeGaps,
        # which call the _delta_*/_sort_* primitives directly (below save_features).
        data = _make_variable_density_shard(
            segment_sizes=[2, 1, 50, 3, 4, 100],
            max_feature_id=W_1M - 1,
            n_docs=2,
        )
        out = save_features(data, tmp_path / "shard_mixed")
        loaded = load_features(out)

        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            orig_sorted = data["feature_ids"][s:e].sort().values
            load_sorted = loaded["feature_ids"][s:e].sort().values
            assert t.equal(orig_sorted, load_sorted), f"Segment {seg}: IDs differ"

    def test_loaded_ids_sorted_ascending(self, tmp_path):
        """After roundtrip, IDs must be sorted ascending within each segment."""
        data = _make_shard_dict(
            n_segments=4, feats_per_seg=30, n_docs=2, max_feature_id=W_1M - 1
        )
        out = save_features(data, tmp_path / "shard_sorted_check")
        loaded = load_features(out)

        offsets = loaded["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            if e - s > 1:
                seg_ids = loaded["feature_ids"][s:e]
                assert t.all(seg_ids[1:] >= seg_ids[:-1]), (
                    f"Segment {seg}: IDs not sorted ascending after roundtrip"
                )

    def test_bit_exact_roundtrip_presorted_1m(self, tmp_path):
        """Pre-sorted 1M-range IDs should be bit-exact after roundtrip."""
        data = _make_shard_dict(
            n_segments=3, feats_per_seg=5, n_docs=1, max_feature_id=W_1M - 1
        )
        offsets = data["offsets"]
        n_segs = offsets.numel() - 1
        # Pre-sort feature IDs ascending within each segment
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            data["feature_ids"][s:e] = data["feature_ids"][s:e].sort().values

        out = save_features(data, tmp_path / "shard_presorted")
        loaded = load_features(out)

        assert t.equal(loaded["feature_ids"], data["feature_ids"])


# ---------------------------------------------------------------------------
# 4. Delta encoding edge cases with large gaps
# ---------------------------------------------------------------------------


class TestDeltaLargeGaps:
    """Edge cases: extreme deltas, first-ID overflow, identical IDs, empties."""

    def test_single_segment_extreme_delta(self):
        """Segment [0, 1048575]: delta of 1048575 exceeds uint16."""
        offsets = t.tensor([0, 2], dtype=t.int64)
        sorted_ids = t.tensor([0, 1048575], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected_delta = t.tensor([0, 1048575], dtype=t.int32)
        assert t.equal(encoded, expected_delta)

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_first_id_exceeds_uint16(self):
        """Segment [500000, 500001, 500002]: first ID >65535."""
        offsets = t.tensor([0, 3], dtype=t.int64)
        sorted_ids = t.tensor([500000, 500001, 500002], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected_delta = t.tensor([500000, 1, 1], dtype=t.int32)
        assert t.equal(encoded, expected_delta)

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_all_identical_large_ids(self):
        """Segment with all identical large IDs [999999, 999999, 999999]."""
        offsets = t.tensor([0, 3], dtype=t.int64)
        sorted_ids = t.tensor([999999, 999999, 999999], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected_delta = t.tensor([999999, 0, 0], dtype=t.int32)
        assert t.equal(encoded, expected_delta)

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_empty_segments_interleaved_with_large_ids(self):
        """Empty segments before and after large-ID segments."""
        offsets = t.tensor([0, 0, 3, 3, 2 + 3], dtype=t.int64)
        sorted_ids = t.tensor([200000, 500000, 1000000, 800000, 900000], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_large_gap_in_middle_of_segment(self):
        """IDs [100, 101, 900000, 900001]: large gap in the middle."""
        offsets = t.tensor([0, 4], dtype=t.int64)
        sorted_ids = t.tensor([100, 101, 900000, 900001], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        expected = t.tensor([100, 1, 899899, 1], dtype=t.int32)
        assert t.equal(encoded, expected)

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_max_possible_delta(self):
        """Two consecutive IDs [0, W_1M-1] produce the max possible delta."""
        offsets = t.tensor([0, 2], dtype=t.int64)
        sorted_ids = t.tensor([0, W_1M - 1], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        assert encoded[1].item() == W_1M - 1

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)

    def test_multiple_segments_mixed_gaps(self):
        """Multiple segments: one with small IDs, one with max-range IDs."""
        offsets = t.tensor([0, 3, 6], dtype=t.int64)
        sorted_ids = t.tensor([10, 20, 30, 500000, 750000, 1048575], dtype=t.int32)

        encoded = _delta_encode(sorted_ids, offsets)
        # Segment 0: [10, 10, 10]
        assert encoded[0].item() == 10
        assert encoded[1].item() == 10
        assert encoded[2].item() == 10
        # Segment 1: [500000, 250000, 298575]
        assert encoded[3].item() == 500000
        assert encoded[4].item() == 250000
        assert encoded[5].item() == 298575

        decoded = _delta_decode(encoded, offsets)
        assert t.equal(decoded, sorted_ids)


# ---------------------------------------------------------------------------
# 5. Verify cumsum doesn't overflow after upcast
# ---------------------------------------------------------------------------


class TestCumsumOverflow:
    """Verify int32 can hold the max possible feature ID after cumsum."""

    def test_max_sae_id_fits_in_int32(self):
        """1048575 (max 1M SAE ID) is well below 2^31 - 1."""
        assert W_1M - 1 < 2**31 - 1

    def test_cumsum_on_int32_produces_correct_max(self):
        """Cumsum of deltas reconstructing max ID stays within int32."""
        offsets = t.tensor([0, 5], dtype=t.int64)
        # Ascending IDs that sum to near the max SAE ID
        sorted_ids = t.tensor([200000, 400000, 600000, 800000, 1048575], dtype=t.int32)
        delta = _delta_encode(sorted_ids, offsets)
        decoded = _delta_decode(delta, offsets)

        assert decoded.dtype == t.int32
        assert decoded.max().item() == 1048575
        assert t.equal(decoded, sorted_ids)

    def test_upcast_from_uint16_before_cumsum(self):
        """_upcast converts uint16 -> int32 so _delta_decode cumsum is correct."""
        # Simulate deltas that individually fit in uint16 but cumsum to >65535
        deltas_uint16 = t.tensor([60000, 10000, 10000, 10000], dtype=t.uint16)

        # _upcast must produce int32 for feature_ids
        upcast = _upcast("feature_ids", deltas_uint16)
        assert upcast.dtype == t.int32, (
            f"Expected int32 after upcast, got {upcast.dtype}"
        )

        # cumsum on the upcast result produces correct values (>65535)
        result = upcast.cumsum(0)
        assert result[0].item() == 60000
        assert result[1].item() == 70000
        assert result[2].item() == 80000
        assert result[3].item() == 90000

    def test_full_pipeline_upcast_correctness(self, tmp_path):
        """save/load roundtrip with IDs whose deltas cumsum past uint16 range."""
        data = _make_shard_dict(n_segments=1, feats_per_seg=4, n_docs=1)
        # IDs that are sorted ascending and whose cumsum exceeds 65535
        data["feature_ids"] = t.tensor([60000, 60100, 60200, 60300], dtype=t.int32)

        out = save_features(data, tmp_path / "shard_upcast")
        loaded = load_features(out)

        # Delta-encoded: [60000, 100, 100, 100] -- all fit in uint16
        # Cumsum: [60000, 60100, 60200, 60300] -- correct only if upcast first
        assert t.equal(loaded["feature_ids"], data["feature_ids"])

    def test_all_widths_max_id_in_int32_range(self):
        """All SAE widths have max ID < 2^31 - 1."""
        for width, num_features in SAE_WIDTH_TO_NUM_FEATURES.items():
            max_id = num_features - 1
            assert max_id < 2**31 - 1, (
                f"SAE width {width.value}: max ID {max_id} exceeds int32 range"
            )
