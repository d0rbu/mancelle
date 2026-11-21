"""Unit tests for packed row shard I/O, CSR reconstruction, and selection shards."""

from __future__ import annotations

import gc
from contextlib import nullcontext

import pytest
import torch as t
import zstandard
from safetensors import SafetensorError
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode

from steering.overlay_io import save_overlay_shard
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    compute_selection_hash,
    load_packed_row_shard,
    load_packed_row_shard_bytes,
    load_packed_row_shard_safetensors_bytes,
    load_selection_shard,
    load_selection_shard_bytes,
    reconstruct_batch,
    save_packed_row_shard,
    save_selection_shard,
)


def _make_identity_selection(shard: PackedRowShard) -> SelectionShard:
    """Create a SelectionShard that selects all features from the base shard."""
    return SelectionShard(
        sel_offsets=shard.seg_feat_offsets.to(t.int32),
        sel_fids=shard.feature_ids,
        sel_buckets=t.ones(len(shard.feature_ids), dtype=t.uint8),
        sel_pcts=shard.feature_pcts,
        neg_offsets=t.zeros(shard.total_segments + 1, dtype=t.int32),
        neg_fids=t.tensor([], dtype=t.uint16),
    )


_PACKED_TENSOR_FIELDS = (
    "tokens",
    "row_seg_offsets",
    "seg_token_start",
    "seg_token_end",
    "seg_doc_id",
    "seg_feat_offsets",
    "feature_ids",
    "feature_pcts",
)


def _assert_packed_shards_exact(*shards: PackedRowShard) -> None:
    reference = shards[0]
    for shard in shards:
        assert type(shard) is PackedRowShard
        assert shard.backing_segment is None
        assert shard.n_rows == reference.n_rows
        assert shard.context_len == reference.context_len
        assert shard.total_segments == reference.total_segments
        assert shard.total_features == reference.total_features
        for field in _PACKED_TENSOR_FIELDS:
            actual = getattr(shard, field)
            expected = getattr(reference, field)
            assert actual.dtype == expected.dtype, field
            assert actual.shape == expected.shape, field
            assert actual.stride() == expected.stride(), field
            assert actual.device == expected.device, field
            assert actual.numel() == expected.numel(), field
            assert (actual.numel() == 0) == (expected.numel() == 0), field
            assert t.equal(actual, expected), field


def _make_test_shard(
    n_rows: int = 4,
    context_len: int = 32,
    segs_per_row: int = 3,
    feats_per_seg: int = 5,
    n_sae: int = 16384,
) -> PackedRowShard:
    """Create a synthetic packed row shard for testing."""
    T = context_len
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg

    # Tokens: random uint16 in vocab range
    tokens = t.randint(0, 32000, (n_rows, T + 1), dtype=t.uint16)

    # Row segment offsets: uniform segments per row
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)

    # Segment data: evenly spaced within each row
    seg_size = T // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
            seg_doc_id[idx] = 0  # single doc per row for simplicity

    # Feature offsets: uniform features per segment
    if feats_per_seg > 0:
        seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    else:
        seg_feat_offsets = t.zeros(total_segs + 1, dtype=t.int32)

    # Feature data
    feature_ids = t.randint(0, n_sae, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(0, 255, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


class TestPackedRowShardProperties:
    def test_n_rows(self):
        shard = _make_test_shard(n_rows=8)
        assert shard.n_rows == 8

    def test_context_len(self):
        shard = _make_test_shard(context_len=64)
        assert shard.context_len == 64

    def test_total_segments(self):
        shard = _make_test_shard(n_rows=4, segs_per_row=3)
        assert shard.total_segments == 12

    def test_total_features(self):
        shard = _make_test_shard(n_rows=4, segs_per_row=3, feats_per_seg=5)
        assert shard.total_features == 60


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path):
        """Save and load preserves all tensors exactly."""
        shard = _make_test_shard()
        path = save_packed_row_shard(shard, tmp_path / "shard_00000")
        loaded = load_packed_row_shard(path)

        assert t.equal(shard.tokens, loaded.tokens)
        assert t.equal(shard.row_seg_offsets, loaded.row_seg_offsets)
        assert t.equal(shard.seg_token_start, loaded.seg_token_start)
        assert t.equal(shard.seg_token_end, loaded.seg_token_end)
        assert t.equal(shard.seg_doc_id, loaded.seg_doc_id)
        assert t.equal(shard.seg_feat_offsets, loaded.seg_feat_offsets)
        assert t.equal(shard.feature_ids, loaded.feature_ids)
        assert t.equal(shard.feature_pcts, loaded.feature_pcts)

    def test_dtypes_preserved(self, tmp_path):
        """All compact dtypes survive save/load."""
        shard = _make_test_shard()
        path = save_packed_row_shard(shard, tmp_path / "test")
        loaded = load_packed_row_shard(path)

        assert loaded.tokens.dtype == t.uint16
        assert loaded.row_seg_offsets.dtype == t.int32
        assert loaded.seg_token_start.dtype == t.uint16
        assert loaded.seg_token_end.dtype == t.uint16
        assert loaded.seg_doc_id.dtype == t.uint8
        assert loaded.seg_feat_offsets.dtype == t.int32
        assert loaded.feature_ids.dtype == t.uint16
        assert loaded.feature_pcts.dtype == t.uint8

    def test_uint16_tokens_lossless(self, tmp_path):
        """Vocab 32K fits in uint16 with no information loss."""
        shard = _make_test_shard()
        # Set a specific known value
        shard.tokens[0, 0] = 32000
        shard.tokens[0, 1] = 0
        shard.tokens[0, 2] = 65535  # max uint16
        path = save_packed_row_shard(shard, tmp_path / "tok_test")
        loaded = load_packed_row_shard(path)
        assert loaded.tokens[0, 0].item() == 32000
        assert loaded.tokens[0, 1].item() == 0
        assert loaded.tokens[0, 2].item() == 65535

    def test_path_and_bytes_loaders_are_dataclass_and_layout_exact(self, tmp_path):
        shard = _make_test_shard(
            n_rows=11,
            context_len=127,
            segs_per_row=7,
            feats_per_seg=13,
            n_sae=65536,
        )
        shard.tokens[0, 0] = 65535
        shard.feature_ids[-1] = 65535
        path = save_packed_row_shard(shard, tmp_path / "packed")

        from_path = load_packed_row_shard(path)
        from_bytes = load_packed_row_shard_bytes(path.read_bytes())
        assert type(from_path) is type(from_bytes) is PackedRowShard
        for field in (
            "tokens",
            "row_seg_offsets",
            "seg_token_start",
            "seg_token_end",
            "seg_doc_id",
            "seg_feat_offsets",
            "feature_ids",
            "feature_pcts",
        ):
            path_tensor = getattr(from_path, field)
            bytes_tensor = getattr(from_bytes, field)
            assert path_tensor.dtype == bytes_tensor.dtype
            assert path_tensor.shape == bytes_tensor.shape
            assert path_tensor.stride() == bytes_tensor.stride()
            assert path_tensor.storage_offset() == bytes_tensor.storage_offset()
            assert t.equal(path_tensor, bytes_tensor)
            assert t.equal(
                path_tensor.contiguous().view(t.uint8),
                bytes_tensor.contiguous().view(t.uint8),
            )
        assert from_path.n_rows == from_bytes.n_rows == 11
        assert from_path.context_len == from_bytes.context_len == 127
        assert from_path.total_segments == from_bytes.total_segments
        assert from_path.total_features == from_bytes.total_features

    @pytest.mark.parametrize(
        "case",
        ["ordinary", "zero_values", "empty_features", "zero_sized"],
    )
    def test_three_way_loaders_are_tensor_exact(self, tmp_path, case):
        if case == "ordinary":
            shard = _make_test_shard()
        elif case == "zero_values":
            shard = _make_test_shard()
            shard.feature_ids.zero_()
            shard.feature_pcts.zero_()
        elif case == "empty_features":
            shard = _make_test_shard(feats_per_seg=0)
        else:
            shard = _make_test_shard(n_rows=0, feats_per_seg=0)

        path = save_packed_row_shard(shard, tmp_path / f"packed-{case}")
        compressed = path.read_bytes()
        decoded = zstandard.ZstdDecompressor().decompress(compressed)
        from_path = load_packed_row_shard(path)
        from_compressed = load_packed_row_shard_bytes(compressed)
        from_decoded = load_packed_row_shard_safetensors_bytes(decoded)

        _assert_packed_shards_exact(from_path, from_compressed, from_decoded)

    def test_decoded_loader_rejects_malformed_or_compressed_bytes(self, tmp_path):
        path = save_packed_row_shard(_make_test_shard(), tmp_path / "packed")
        compressed = path.read_bytes()
        decoded = zstandard.ZstdDecompressor().decompress(compressed)

        for invalid in (b"not-safetensors", decoded[:-3], compressed):
            with pytest.raises(SafetensorError):
                load_packed_row_shard_safetensors_bytes(invalid)

    def test_path_and_bytes_loaders_raise_identical_missing_key_error(self, tmp_path):
        shard = _make_test_shard()
        incomplete = {
            key: value
            for key, value in shard.__dict__.items()
            if key != "feature_pcts" and isinstance(value, t.Tensor)
        }
        path = save_overlay_shard(incomplete, tmp_path / "incomplete-packed")

        with pytest.raises(KeyError) as path_error:
            load_packed_row_shard(path)
        with pytest.raises(KeyError) as bytes_error:
            load_packed_row_shard_bytes(path.read_bytes())
        decoded = zstandard.ZstdDecompressor().decompress(path.read_bytes())
        with pytest.raises(KeyError) as decoded_error:
            load_packed_row_shard_safetensors_bytes(decoded)
        assert (
            path_error.value.args
            == bytes_error.value.args
            == decoded_error.value.args
            == ("feature_pcts",)
        )


class TestReconstructBatch:
    def test_shapes(self):
        """Reconstructed batch has correct shapes."""
        shard = _make_test_shard(n_rows=8, context_len=32, segs_per_row=3, feats_per_seg=5)
        rb = reconstruct_batch(shard, [0, 1, 2, 3], _make_identity_selection(shard), 10)
        inputs, targets, seg_map, seg_doc_id, enc_fids, enc_pcts, enc_mask = (
            rb.inputs, rb.targets, rb.seg_map, rb.seg_doc_id,
            rb.enc_fids, rb.enc_pcts, rb.enc_mask,
        )

        B, T = 4, 32
        assert inputs.shape == (B, T)
        assert targets.shape == (B, T)
        assert seg_map.shape == (B, T)
        assert seg_doc_id.shape[0] == B
        assert enc_fids.shape[0] == B
        assert enc_pcts.shape[0] == B
        assert enc_mask.shape[0] == B
        # n_seg and F_max are dynamic
        assert enc_fids.shape[1] >= 3  # at least 3 segs per row
        assert enc_fids.shape[2] >= 5  # at least 5 feats per seg

    def test_inputs_targets_relationship(self):
        """targets = tokens[:,1:], inputs = tokens[:,:T]."""
        shard = _make_test_shard(n_rows=4, context_len=16)
        inputs, targets, *_ = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10)

        raw_tokens = shard.tokens[:2].to(t.int64)
        assert t.equal(inputs, raw_tokens[:, :16])
        assert t.equal(targets, raw_tokens[:, 1:17])

    def test_seg_map_covers_segments(self):
        """seg_map assigns covered positions to correct segment indices."""
        shard = _make_test_shard(n_rows=4, context_len=30, segs_per_row=3)
        seg_map = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10).seg_map

        # With seg_size=10, segments cover [0,10), [10,20), [20,30)
        assert seg_map.shape == (1, 30)
        assert seg_map[0, 0].item() == 0
        assert seg_map[0, 10].item() == 1
        assert seg_map[0, 20].item() == 2

    def test_enc_mask_matches_feature_count(self):
        """enc_mask True count per segment matches actual feature count."""
        shard = _make_test_shard(n_rows=2, segs_per_row=2, feats_per_seg=4)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        enc_mask = rb.enc_mask

        # Each segment has exactly 4 features
        for s in range(2):
            assert enc_mask[0, s].sum().item() == 4

    def test_single_row_batch(self):
        """Batch of 1 row works correctly."""
        shard = _make_test_shard(n_rows=4, context_len=16)
        rb = reconstruct_batch(shard, [2], _make_identity_selection(shard), 10)
        assert rb.inputs.shape == (1, 16)
        assert rb.enc_fids.shape[0] == 1

    def test_non_contiguous_row_indices(self):
        """Can select non-contiguous rows from the shard."""
        shard = _make_test_shard(n_rows=8, context_len=16)
        inputs, *_ = reconstruct_batch(shard, [0, 3, 7], _make_identity_selection(shard), 10)

        assert inputs.shape[0] == 3
        assert t.equal(inputs[0], shard.tokens[0, :16].to(t.int64))
        assert t.equal(inputs[1], shard.tokens[3, :16].to(t.int64))
        assert t.equal(inputs[2], shard.tokens[7, :16].to(t.int64))

    def test_feature_ids_upcast_to_int64(self):
        """Feature IDs are upcast from uint16 to int64 for downstream compatibility."""
        shard = _make_test_shard(n_rows=2, segs_per_row=1, feats_per_seg=3)
        enc_fids = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10).enc_fids
        assert enc_fids.dtype == t.int64

    def test_percentiles_normalized_to_float(self):
        """Percentiles are decoded from uint8 to float32 in [0, 1]."""
        shard = _make_test_shard(n_rows=2, segs_per_row=1, feats_per_seg=3)
        # Set known percentile values
        shard.feature_pcts[:3] = t.tensor([0, 128, 255], dtype=t.uint8)

        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        enc_pcts, enc_mask = rb.enc_pcts, rb.enc_mask
        valid_pcts = enc_pcts[0, 0, enc_mask[0, 0]]
        assert valid_pcts[0].item() == 0.0
        assert abs(valid_pcts[1].item() - 128 / 255.0) < 1e-6
        assert abs(valid_pcts[2].item() - 1.0) < 1e-6


class TestEdgeCases:
    def test_empty_segment(self):
        """Segment with 0 features is handled correctly."""
        shard = _make_test_shard(n_rows=2, segs_per_row=2, feats_per_seg=0)
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        enc_mask = rb.enc_mask

        # All mask entries should be False (no features)
        # But enc_mask shape should still be valid (F_max=1 minimum)
        assert enc_mask.shape[2] >= 1
        assert enc_mask.sum().item() == 0

    def test_variable_features_per_segment(self, tmp_path):
        """Segments with different feature counts pad correctly."""
        # Build a shard manually with variable feature counts
        n_rows, T = 2, 16
        tokens = t.randint(0, 1000, (n_rows, T + 1), dtype=t.uint16)
        # Row 0: 2 segments with 3 and 7 features
        # Row 1: 1 segment with 5 features
        row_seg_offsets = t.tensor([0, 2, 3], dtype=t.int32)
        seg_token_start = t.tensor([0, 8, 0], dtype=t.uint16)
        seg_token_end = t.tensor([8, 16, 16], dtype=t.uint16)
        seg_doc_id = t.tensor([0, 0, 0], dtype=t.uint8)
        seg_feat_offsets = t.tensor([0, 3, 10, 15], dtype=t.int32)
        feature_ids = t.randint(0, 100, (15,), dtype=t.uint16)
        feature_pcts = t.randint(0, 255, (15,), dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens, row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start, seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id, seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids, feature_pcts=feature_pcts,
        )

        rb = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10)
        enc_fids, enc_mask = rb.enc_fids, rb.enc_mask
        # F_max should be 7 (max features in any segment)
        assert enc_fids.shape[2] == 7
        # n_seg should be 2 (max segments in any row)
        assert enc_fids.shape[1] == 2
        # Row 0, seg 0: 3 features
        assert enc_mask[0, 0].sum().item() == 3
        # Row 0, seg 1: 7 features
        assert enc_mask[0, 1].sum().item() == 7
        # Row 1, seg 0: 5 features
        assert enc_mask[1, 0].sum().item() == 5
        # Row 1, seg 1: padding (no segment)
        assert enc_mask[1, 1].sum().item() == 0

    def test_multi_doc_row(self):
        """Row with multiple documents has correct seg_doc_id."""
        n_rows, T = 1, 32
        tokens = t.randint(0, 1000, (n_rows, T + 1), dtype=t.uint16)
        # 4 segments: doc 0 has segs 0-1, doc 1 has segs 2-3
        row_seg_offsets = t.tensor([0, 4], dtype=t.int32)
        seg_token_start = t.tensor([0, 8, 16, 24], dtype=t.uint16)
        seg_token_end = t.tensor([8, 16, 24, 32], dtype=t.uint16)
        seg_doc_id = t.tensor([0, 0, 1, 1], dtype=t.uint8)
        seg_feat_offsets = t.tensor([0, 2, 4, 6, 8], dtype=t.int32)
        feature_ids = t.randint(0, 100, (8,), dtype=t.uint16)
        feature_pcts = t.randint(0, 255, (8,), dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens, row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start, seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id, seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids, feature_pcts=feature_pcts,
        )

        seg_doc_id_out = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10).seg_doc_id
        assert seg_doc_id_out[0, 0].item() == 0
        assert seg_doc_id_out[0, 1].item() == 0
        assert seg_doc_id_out[0, 2].item() == 1
        assert seg_doc_id_out[0, 3].item() == 1

    def test_large_shard_save_load(self, tmp_path):
        """Shard with many rows (simulating R=16384 at small T)."""
        shard = _make_test_shard(n_rows=1024, context_len=16, segs_per_row=2, feats_per_seg=3)
        path = save_packed_row_shard(shard, tmp_path / "large_shard")
        loaded = load_packed_row_shard(path)

        assert loaded.n_rows == 1024
        assert t.equal(shard.tokens, loaded.tokens)
        assert t.equal(shard.feature_ids, loaded.feature_ids)


# ---------------------------------------------------------------------------
# Selection shard tests
# ---------------------------------------------------------------------------


def _make_test_selection_shard(total_segs: int = 12, sel_per_seg: int = 3, neg_per_seg: int = 2):
    """Create a synthetic selection shard."""
    total_sel = total_segs * sel_per_seg
    total_neg = total_segs * neg_per_seg

    sel_offsets = t.arange(0, (total_segs + 1) * sel_per_seg, sel_per_seg, dtype=t.int32)
    sel_fids = t.randint(0, 16384, (total_sel,), dtype=t.uint16)
    sel_buckets = t.randint(0, 10, (total_sel,), dtype=t.uint8)
    sel_pcts = t.randint(0, 255, (total_sel,), dtype=t.uint8)

    neg_offsets = t.arange(0, (total_segs + 1) * neg_per_seg, neg_per_seg, dtype=t.int32)
    neg_fids = t.randint(0, 16384, (total_neg,), dtype=t.uint16)

    return SelectionShard(
        sel_offsets=sel_offsets, sel_fids=sel_fids, sel_buckets=sel_buckets,
        sel_pcts=sel_pcts, neg_offsets=neg_offsets, neg_fids=neg_fids,
    )


class TestSelectionShardIO:
    def test_round_trip(self, tmp_path):
        """Save and load preserves all tensors exactly."""
        shard = _make_test_selection_shard()
        path = save_selection_shard(shard, tmp_path / "sel_00000")
        loaded = load_selection_shard(path)

        assert t.equal(shard.sel_offsets, loaded.sel_offsets)
        assert t.equal(shard.sel_fids, loaded.sel_fids)
        assert t.equal(shard.sel_buckets, loaded.sel_buckets)
        assert t.equal(shard.sel_pcts, loaded.sel_pcts)
        assert t.equal(shard.neg_offsets, loaded.neg_offsets)
        assert t.equal(shard.neg_fids, loaded.neg_fids)

    def test_dtypes_preserved(self, tmp_path):
        shard = _make_test_selection_shard()
        path = save_selection_shard(shard, tmp_path / "sel")
        loaded = load_selection_shard(path)

        assert loaded.sel_offsets.dtype == t.int32
        assert loaded.sel_fids.dtype == t.uint16
        assert loaded.sel_buckets.dtype == t.uint8
        assert loaded.sel_pcts.dtype == t.uint8
        assert loaded.neg_offsets.dtype == t.int32
        assert loaded.neg_fids.dtype == t.uint16

    def test_path_and_bytes_loaders_are_dataclass_and_layout_exact(self, tmp_path):
        shard = _make_test_selection_shard(
            total_segs=257,
            sel_per_seg=11,
            neg_per_seg=7,
        )
        shard.sel_fids[0] = 65535
        shard.neg_fids[-1] = 65535
        path = save_selection_shard(shard, tmp_path / "selection")

        from_path = load_selection_shard(path)
        from_bytes = load_selection_shard_bytes(path.read_bytes())
        assert type(from_path) is type(from_bytes) is SelectionShard
        for field in (
            "sel_offsets",
            "sel_fids",
            "sel_buckets",
            "sel_pcts",
            "neg_offsets",
            "neg_fids",
        ):
            path_tensor = getattr(from_path, field)
            bytes_tensor = getattr(from_bytes, field)
            assert path_tensor.dtype == bytes_tensor.dtype
            assert path_tensor.shape == bytes_tensor.shape
            assert path_tensor.stride() == bytes_tensor.stride()
            assert path_tensor.storage_offset() == bytes_tensor.storage_offset()
            assert t.equal(path_tensor, bytes_tensor)
            assert t.equal(
                path_tensor.contiguous().view(t.uint8),
                bytes_tensor.contiguous().view(t.uint8),
            )
        assert from_path.total_segments == from_bytes.total_segments == 257
        assert from_path.total_selected == from_bytes.total_selected
        assert from_path.total_negative == from_bytes.total_negative

    def test_path_and_bytes_loaders_raise_identical_missing_key_error(self, tmp_path):
        shard = _make_test_selection_shard()
        incomplete = {key: value for key, value in shard.__dict__.items() if key != "neg_fids"}
        path = save_overlay_shard(incomplete, tmp_path / "incomplete-selection")

        with pytest.raises(KeyError) as path_error:
            load_selection_shard(path)
        with pytest.raises(KeyError) as bytes_error:
            load_selection_shard_bytes(path.read_bytes())
        assert path_error.value.args == bytes_error.value.args == ("neg_fids",)

    def test_properties(self):
        shard = _make_test_selection_shard(total_segs=10, sel_per_seg=4, neg_per_seg=2)
        assert shard.total_segments == 10
        assert shard.total_selected == 40
        assert shard.total_negative == 20


class TestReconstructBatchWithSelection:
    def test_shapes(self):
        """Reconstruction with selection produces correct shapes."""
        base = _make_test_shard(n_rows=4, context_len=32, segs_per_row=3, feats_per_seg=5)
        sel = _make_test_selection_shard(total_segs=12, sel_per_seg=3, neg_per_seg=2)

        rb = reconstruct_batch(base, [0, 1], sel, 10)

        assert rb.inputs.shape == (2, 32)
        assert rb.targets.shape == (2, 32)
        assert rb.enc_fids.shape[0] == 2
        assert rb.enc_mags.shape == rb.enc_fids.shape  # now filled (not zeros)
        assert rb.enc_mask.shape == rb.enc_fids.shape

    def test_positive_features_have_buckets(self):
        """Selected positive features have non-zero bucket values."""
        base = _make_test_shard(n_rows=2, context_len=16, segs_per_row=2, feats_per_seg=4)
        sel = _make_test_selection_shard(total_segs=4, sel_per_seg=2, neg_per_seg=1)
        # Ensure at least one non-zero bucket
        sel.sel_buckets[0] = 5

        rb = reconstruct_batch(base, [0], sel, 10)

        # Positive features should have their actual bucket values
        # (not all zeros like in the base-only reconstruction)
        assert rb.enc_mags[rb.enc_mask].max().item() > 0 or rb.enc_mags[rb.enc_mask].numel() == 0

    def test_negative_features_have_effective_bucket(self):
        """Negative features get the effective_buckets value."""
        base = _make_test_shard(n_rows=2, context_len=16, segs_per_row=1, feats_per_seg=3)
        # 1 selected, 2 negative per segment
        sel = _make_test_selection_shard(total_segs=2, sel_per_seg=1, neg_per_seg=2)
        sel.sel_buckets[:] = 3  # positive buckets

        rb = reconstruct_batch(base, [0], sel, 10)

        # For the first segment, feature at index 0 is positive (bucket=3),
        # features at indices 1-2 are negative (bucket=10)
        if rb.enc_mask[0, 0].sum().item() >= 3:
            assert rb.enc_mags[0, 0, 0].item() == 3   # positive bucket
            assert rb.enc_mags[0, 0, 1].item() == 10   # negative = effective_buckets
            assert rb.enc_mags[0, 0, 2].item() == 10


def _selected_payload_fixture():
    shard = _make_test_shard(n_rows=3, context_len=8, segs_per_row=2, feats_per_seg=1)
    selection = SelectionShard(
        sel_offsets=t.tensor([0, 0, 2, 3, 3, 5, 6], dtype=t.int32),
        sel_fids=t.tensor([11, 12, 21, 41, 42, 51], dtype=t.uint16),
        sel_buckets=t.tensor([1, 2, 3, 4, 5, 6], dtype=t.uint8),
        sel_pcts=t.tensor([0, 51, 102, 153, 204, 255], dtype=t.uint8),
        neg_offsets=t.tensor([0, 1, 1, 1, 3, 4, 4], dtype=t.int32),
        neg_fids=t.tensor([101, 131, 132, 141], dtype=t.uint16),
    )
    # Keep each selection array noncontiguous, including both CSR offset arrays.
    for name, value in vars(selection).items():
        backing = t.empty(2 * value.numel(), dtype=value.dtype)
        backing[::2] = value
        setattr(selection, name, backing[::2])
    expected = {
        "inputs": shard.tokens[[2, 0, 2], :8].long(),
        "targets": shard.tokens[[2, 0, 2], 1:].long(),
        "seg_map": t.tensor([[0] * 4 + [1] * 4] * 3),
        "seg_doc_id": t.tensor([[0, 0, -1]] * 3),
        "enc_fids": t.tensor([
            [[41, 42, 141, 0], [51, 0, 0, 0], [0, 0, 0, 0]],
            [[101, 0, 0, 0], [11, 12, 0, 0], [0, 0, 0, 0]],
            [[41, 42, 141, 0], [51, 0, 0, 0], [0, 0, 0, 0]],
        ]),
        "enc_mags": t.tensor([
            [[4, 5, 9, 0], [6, 0, 0, 0], [0, 0, 0, 0]],
            [[9, 0, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]],
            [[4, 5, 9, 0], [6, 0, 0, 0], [0, 0, 0, 0]],
        ]),
        "enc_mask": t.tensor([
            [[True, True, True, False], [True, False, False, False], [False] * 4],
            [[True, False, False, False], [True, True, False, False], [False] * 4],
            [[True, True, True, False], [True, False, False, False], [False] * 4],
        ]),
        "enc_pcts": t.tensor([
            [[153, 204, 0, 0], [255, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 51, 0, 0], [0, 0, 0, 0]],
            [[153, 204, 0, 0], [255, 0, 0, 0], [0, 0, 0, 0]],
        ], dtype=t.float32) * (1.0 / 255.0),
    }
    return shard, selection, expected


def _plain_payload(value):
    if isinstance(value, FunctionalTensor):
        value = value.from_functional()
    return value if type(value) is t.Tensor else value.as_subclass(t.Tensor)


def _assert_selected_payload(batch, expected):
    assert set(batch._fields) == set(expected)
    for name, target in expected.items():
        actual = _plain_payload(getattr(batch, name))
        assert actual.dtype == target.dtype, name
        assert actual.shape == target.shape, name
        assert t.equal(actual, target), name


class _StorageTensor(t.Tensor):
    """Ordinary storage-backed subclass; no custom dispatch behavior."""


class TestSelectedPayloadExpansion:
    @pytest.mark.parametrize("mode", ["ordinary", "no_grad", "inference"])
    def test_repeated_rows_with_mixed_counts(self, mode):
        shard, selection, expected = _selected_payload_fixture()
        sources = [*vars(shard).values(), *vars(selection).values()]
        sources = [value for value in sources if isinstance(value, t.Tensor)]
        before = [(value._version, value.clone()) for value in sources]
        context = {"ordinary": nullcontext, "no_grad": t.no_grad, "inference": t.inference_mode}[mode]
        with context():
            batch = reconstruct_batch(shard, [2, 0, 2], selection, 9, pad_n_seg=3, pad_F=4)
            _assert_selected_payload(batch, expected)
            batch.enc_pcts.add_(1)
            for name in expected.keys() - {"enc_pcts"}:
                assert t.equal(getattr(batch, name), expected[name]), name
        for value, (version, original) in zip(sources, before, strict=True):
            assert value._version == version
            assert t.equal(value, original)

    @pytest.mark.parametrize("dtype", [t.float32, t.float64])
    def test_percentile_backward_after_repeated_row_reconstruction(self, dtype):
        shard, selection, expected = _selected_payload_fixture()
        selection.sel_pcts = selection.sel_pcts.to(dtype).requires_grad_()
        before = selection.sel_pcts.detach().clone()
        version = selection.sel_pcts._version
        batch = reconstruct_batch(shard, [2, 0, 2], selection, 9, pad_n_seg=3, pad_F=4)
        _assert_selected_payload(batch, expected)
        # Backward runs after the reconstruction frame and its row-index views die.
        gc.collect()
        weights = t.arange(1, batch.enc_pcts.numel() + 1, dtype=t.float32).reshape(batch.enc_pcts.shape)
        (batch.enc_pcts * weights).sum().backward()
        # Row0's two positives occur once; row2's three occur twice; row1 is absent.
        expected_grad = t.tensor([17, 18, 0, 1 + 25, 2 + 26, 5 + 29], dtype=t.float32) * (1.0 / 255.0)
        t.testing.assert_close(selection.sel_pcts.grad, expected_grad.to(dtype), rtol=1e-6, atol=1e-7)
        assert selection.sel_pcts._version == version
        assert t.equal(selection.sel_pcts.detach(), before)

    @pytest.mark.parametrize("mode", ["storage", "functional"])
    def test_supported_tensor_modes_preserve_values_and_ownership(self, mode):
        shard, selection, expected = _selected_payload_fixture()
        context = FunctionalTensorMode() if mode == "functional" else nullcontext()
        with context:
            sources = []
            for owner in (shard, selection):
                for name, value in vars(owner).items():
                    if isinstance(value, t.Tensor):
                        wrapped = FunctionalTensor.to_functional(value) if mode == "functional" else value.as_subclass(_StorageTensor)
                        setattr(owner, name, wrapped)
                        sources.append(wrapped)
            before = [(value._version, _plain_payload(value).clone()) for value in sources]
            batch = reconstruct_batch(shard, [2, 0, 2], selection, 9, pad_n_seg=3, pad_F=4)
            _assert_selected_payload(batch, expected)
            for name in batch._fields:
                expected_class = FunctionalTensor if mode == "functional" else _StorageTensor if name in ("inputs", "targets") else t.Tensor
                assert type(getattr(batch, name)) is expected_class, name
            source_storage = {_plain_payload(value).untyped_storage()._cdata for value in sources}
            output_storage = {_plain_payload(value).untyped_storage()._cdata for value in batch}
            assert source_storage.isdisjoint(output_storage)
            assert len(output_storage) == len(batch)
            batch.enc_pcts.add_(1)
            for name in expected.keys() - {"enc_pcts"}:
                assert t.equal(_plain_payload(getattr(batch, name)), expected[name]), name
            for value, (version, original) in zip(sources, before, strict=True):
                assert value._version == version
                assert t.equal(_plain_payload(value), original)


class TestSelectionHash:
    def test_deterministic(self):
        """Same config produces same hash."""
        h1 = compute_selection_hash(
            {"type": "sampling", "mean": 5.0}, {"mean": 3.0}, {"mode": "prev_segment"}, 10, 42
        )
        h2 = compute_selection_hash(
            {"type": "sampling", "mean": 5.0}, {"mean": 3.0}, {"mode": "prev_segment"}, 10, 42
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_different_configs_different_hash(self):
        h1 = compute_selection_hash(
            {"type": "sampling", "mean": 5.0}, None, None, 10, 42
        )
        h2 = compute_selection_hash(
            {"type": "sampling", "mean": 3.0}, None, None, 10, 42
        )
        assert h1 != h2

    def test_seed_affects_hash(self):
        h1 = compute_selection_hash({"type": "threshold"}, None, None, 10, 42)
        h2 = compute_selection_hash({"type": "threshold"}, None, None, 10, 99)
        assert h1 != h2

    def test_rank_affects_hash(self):
        h1 = compute_selection_hash({"type": "threshold"}, None, None, 10, 42, ddp_rank=0)
        h2 = compute_selection_hash({"type": "threshold"}, None, None, 10, 42, ddp_rank=1)
        assert h1 != h2
