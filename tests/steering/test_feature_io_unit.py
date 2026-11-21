"""Unit tests for steering/feature_io.py — slice 01 Phase 1 audit.

Pins fixes for plan items B-1 (header size cap), B-2 (zstd level validation),
B-7/B-8 (dtype optimization branches).
"""

import struct

import pytest
import torch as t
import zstandard

from steering.feature_io import (
    FP16_SAFE_MAX,
    ZSTD_LEVEL,
    RegularFileIdentity,
    TensorKey,
    _optimize_dtype,
    _upcast,
    read_shard_header,
    save_features,
)

# ---------------------------------------------------------------------------
# B-1: read_shard_header upper bound on header_size
# ---------------------------------------------------------------------------


class TestReadShardHeaderHeaderSizeCap:
    """A corrupted shard with an absurd header_size must raise, not allocate."""

    def test_corrupted_header_size_raises(self, tmp_path):
        """A shard claiming `header_size = 2**62` must raise."""
        path = tmp_path / "fake_features.safetensors.zst"
        # Build a file: 8-byte little-endian header_size, then a tiny payload
        # that does not match the claim. Compress so read_shard_header decodes.
        evil_header_size = 2**62  # absurdly large
        sf_bytes = struct.pack("<Q", evil_header_size) + b"{}"
        cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        path.write_bytes(cctx.compress(sf_bytes))

        with pytest.raises(ValueError, match="header"):
            read_shard_header(path)

    def test_legitimate_header_size_passes(self, tmp_path):
        """A shard with a normal header_size (a few KB) must not raise."""
        # Build a real (tiny) shard via save_features and verify read_shard_header
        # works on it. This guards against B-1's cap being set too low.
        offsets = t.tensor([0, 2], dtype=t.int64)
        feature_ids = t.tensor([0, 1], dtype=t.int32)
        feature_mags = t.tensor([1.0, 2.0], dtype=t.float32)
        doc_ids = t.tensor([0, 0], dtype=t.int32)
        seg_indices = t.tensor([0], dtype=t.int32)
        seg_positions = t.tensor([0], dtype=t.int32)
        seg_char_starts = t.tensor([0], dtype=t.int32)
        seg_char_ends = t.tensor([10], dtype=t.int32)
        path = tmp_path / "shard_features.safetensors.zst"
        save_features(
            {
                "offsets": offsets,
                "feature_ids": feature_ids,
                "feature_mags": feature_mags,
                "doc_ids": doc_ids,
                "segment_indices": seg_indices,
                "segment_positions": seg_positions,
                "segment_char_starts": seg_char_starts,
                "segment_char_ends": seg_char_ends,
            },
            path,
        )
        header = read_shard_header(path)
        assert "__metadata__" in header
        assert "__tensor_keys__" in header
        assert TensorKey.FEATURE_IDS.value in header["__tensor_keys__"]


# ---------------------------------------------------------------------------
# B-2: save_features validates zstd_level range
# ---------------------------------------------------------------------------


class TestSaveFeaturesZstdLevelValidation:
    """zstd_level must be in [1, 22] (zstd library range)."""

    def _make_minimal_shard_dict(self) -> dict:
        return {
            "offsets": t.tensor([0, 1], dtype=t.int64),
            "feature_ids": t.tensor([0], dtype=t.int32),
            "feature_mags": t.tensor([1.0], dtype=t.float32),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([10], dtype=t.int32),
        }

    def test_zstd_level_zero_raises(self, tmp_path):
        path = tmp_path / "shard_features.safetensors.zst"
        with pytest.raises(ValueError, match="zstd_level"):
            save_features(self._make_minimal_shard_dict(), path, zstd_level=0)

    def test_zstd_level_negative_raises(self, tmp_path):
        path = tmp_path / "shard_features.safetensors.zst"
        with pytest.raises(ValueError, match="zstd_level"):
            save_features(self._make_minimal_shard_dict(), path, zstd_level=-1)

    def test_zstd_level_too_high_raises(self, tmp_path):
        path = tmp_path / "shard_features.safetensors.zst"
        with pytest.raises(ValueError, match="zstd_level"):
            save_features(self._make_minimal_shard_dict(), path, zstd_level=23)

    def test_zstd_level_one_passes(self, tmp_path):
        """Level 1 is the minimum valid level."""
        path = tmp_path / "shard_features.safetensors.zst"
        out = save_features(self._make_minimal_shard_dict(), path, zstd_level=1)
        assert out.exists()

    def test_zstd_level_twenty_two_passes(self, tmp_path):
        """Level 22 is the maximum valid level."""
        path = tmp_path / "shard_features.safetensors.zst"
        out = save_features(self._make_minimal_shard_dict(), path, zstd_level=22)
        assert out.exists()


# ---------------------------------------------------------------------------
# B-7: _optimize_dtype int32 vs uint16 switch on max ID
# ---------------------------------------------------------------------------


class TestOptimizeDtypeFeatureIds:
    def test_max_id_at_uint16_boundary_uses_uint16(self):
        """feature_ids with max == 65535 must be uint16."""
        ids = t.tensor([0, 1, 65535], dtype=t.int32)
        out = _optimize_dtype(TensorKey.FEATURE_IDS.value, ids)
        assert out.dtype == t.uint16

    def test_max_id_above_uint16_keeps_int32(self):
        """feature_ids with max > 65535 must stay int32."""
        ids = t.tensor([0, 1, 65536], dtype=t.int32)
        out = _optimize_dtype(TensorKey.FEATURE_IDS.value, ids)
        assert out.dtype == t.int32

    def test_empty_ids_uses_uint16(self):
        ids = t.tensor([], dtype=t.int32)
        out = _optimize_dtype(TensorKey.FEATURE_IDS.value, ids)
        assert out.dtype == t.uint16


# ---------------------------------------------------------------------------
# B-8: _optimize_dtype offsets int32 vs int64
# ---------------------------------------------------------------------------


class TestOptimizeDtypeOffsets:
    def test_offsets_at_int32_boundary_uses_int32(self):
        offsets = t.tensor([0, 2**31 - 1], dtype=t.int64)
        out = _optimize_dtype(TensorKey.OFFSETS.value, offsets)
        assert out.dtype == t.int32

    def test_offsets_above_int32_keeps_int64(self):
        offsets = t.tensor([0, 2**31], dtype=t.int64)
        out = _optimize_dtype(TensorKey.OFFSETS.value, offsets)
        assert out.dtype == t.int64

    def test_empty_offsets_uses_int32(self):
        offsets = t.tensor([], dtype=t.int64)
        out = _optimize_dtype(TensorKey.OFFSETS.value, offsets)
        assert out.dtype == t.int32

    def test_upcast_int32_to_int64_offsets(self):
        offsets = t.tensor([0, 1], dtype=t.int32)
        out = _upcast(TensorKey.OFFSETS.value, offsets)
        assert out.dtype == t.int64


# ---------------------------------------------------------------------------
# B-6: feature_mags fp16 saturation guard
# ---------------------------------------------------------------------------


class TestFp16SaturationGuard:
    def test_safe_max_minus_epsilon_passes(self):
        """A value just below the safe max should succeed."""
        mags = t.tensor([FP16_SAFE_MAX - 1.0], dtype=t.float32)
        out = _optimize_dtype(TensorKey.FEATURE_MAGS.value, mags)
        assert out.dtype == t.float16

    def test_safe_max_raises(self):
        """Exactly FP16_SAFE_MAX must raise."""
        mags = t.tensor([FP16_SAFE_MAX], dtype=t.float32)
        with pytest.raises(ValueError, match="fp16"):
            _optimize_dtype(TensorKey.FEATURE_MAGS.value, mags)

    def test_above_safe_max_raises(self):
        mags = t.tensor([FP16_SAFE_MAX + 100.0], dtype=t.float32)
        with pytest.raises(ValueError, match="fp16"):
            _optimize_dtype(TensorKey.FEATURE_MAGS.value, mags)

    def test_empty_mags_passes(self):
        mags = t.tensor([], dtype=t.float32)
        out = _optimize_dtype(TensorKey.FEATURE_MAGS.value, mags)
        assert out.dtype == t.float16


class TestFromVerifiedPublication:
    def test_agreeing_stat_binds_unchanged_without_warning(self, tmp_path, caplog):
        path = tmp_path / "artifact"
        path.write_bytes(b"payload bytes")
        result = path.lstat()
        bound = RegularFileIdentity.from_verified_publication(
            result,
            verified_byte_length=len(b"payload bytes"),
            path=path,
        )
        assert bound == RegularFileIdentity.from_stat(result)
        assert not caplog.records

    def test_stale_stat_size_is_replaced_by_verified_length(self, tmp_path, caplog):
        path = tmp_path / "artifact"
        path.write_bytes(b"payload bytes")
        result = path.lstat()
        bound = RegularFileIdentity.from_verified_publication(
            result,
            verified_byte_length=len(b"payload bytes") + 7,
            path=path,
        )
        expected = RegularFileIdentity.from_stat(result)
        assert bound.size == len(b"payload bytes") + 7
        assert (bound.mode, bound.device, bound.inode) == (
            expected.mode,
            expected.device,
            expected.inode,
        )
        assert (bound.mtime_ns, bound.ctime_ns) == (
            expected.mtime_ns,
            expected.ctime_ns,
        )
        assert len(caplog.records) == 1
        assert "binding the verified length" in caplog.records[0].message
        assert str(path) in caplog.records[0].message
