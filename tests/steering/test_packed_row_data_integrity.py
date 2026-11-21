"""Deep data integrity tests for packed row CSR format.

These tests scrutinize the packed row precomputation for subtle data corruption:
1. Percentile encoding round-trip fidelity (uint8 → float32 → uint8)
2. Feature ID preservation through the full pipeline
3. Token data consistency between input and output
4. CSR structural invariants under various conditions
5. Doc ID consistency across packing operations
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from scripts.preprocess.precompute_packed_rows import (
    precompute_packed_rows,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import (
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_packed_row_shard,
    reconstruct_batch,
    save_packed_row_shard,
)
from steering.segmentation import token_to_char_offsets


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


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        return " ".join(f"t{tid}" for tid in token_ids)


# ======================================================================
# 1. Percentile encoding round-trip fidelity
# ======================================================================


class TestPercentileRoundTrip:
    """Test that percentiles survive encoding round-trips.

    Note: percentile_to_uint8 takes percentile scores in [0.5, 99.5]
    (from magnitude_to_percentile), normalizes by /100, then encodes.
    uint8_to_percentile reverses: uint8/255 → [0, 1].
    """

    def test_uint8_round_trip_preserves_ordering(self):
        """Monotonic ordering is preserved through encoding."""
        # percentile_to_uint8 expects values in [0.5, 99.5]
        values = t.linspace(0.5, 99.5, 100)
        encoded = percentile_to_uint8(values)
        decoded = uint8_to_percentile(encoded)

        # Decoded values should be approximately monotonically non-decreasing
        diffs = decoded[1:] - decoded[:-1]
        assert (diffs >= -0.005).all(), "Percentile ordering violated after round-trip"

    def test_uint8_to_percentile_boundaries(self):
        """uint8_to_percentile: 0 → 0.0, 128 → ~0.502, 255 → 1.0."""
        encoded = t.tensor([0, 128, 255], dtype=t.uint8)
        decoded = uint8_to_percentile(encoded)

        assert abs(decoded[0].item() - 0.0) < 0.01
        assert abs(decoded[1].item() - 128.0 / 255.0) < 0.01
        assert abs(decoded[2].item() - 1.0) < 0.01

    def test_reconstruct_percentile_range(self):
        """Reconstructed percentiles from packed rows are in [0, 1]."""
        n_rows, T = 2, 16
        segs_per_row, feats_per_seg = 2, 3
        total_segs = n_rows * segs_per_row
        total_feats = total_segs * feats_per_seg

        # Use a range of uint8 percentile values
        pct_values = [0, 64, 128, 191, 255, 10, 0, 50, 100, 200, 250, 30]
        assert len(pct_values) == total_feats

        shard = PackedRowShard(
            tokens=t.randint(1, 100, (n_rows, T + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2, 4], dtype=t.int32),
            seg_token_start=t.tensor([0, 8, 0, 8], dtype=t.uint16),
            seg_token_end=t.tensor([8, 16, 8, 16], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0, 0, 0], dtype=t.uint8),
            seg_feat_offsets=t.arange(0, total_feats + 1, feats_per_seg, dtype=t.int32),
            feature_ids=t.randint(0, 100, (total_feats,), dtype=t.uint16),
            feature_pcts=t.tensor(pct_values, dtype=t.uint8),
        )

        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)

        # All valid percentiles should be in [0, 1]
        valid_pcts = rb.enc_pcts[rb.enc_mask]
        assert (valid_pcts >= 0).all()
        assert (valid_pcts <= 1).all()

        # First segment: features with uint8 [0, 64, 128]
        valid_pcts_s0 = rb.enc_pcts[0, 0, rb.enc_mask[0, 0]]
        assert len(valid_pcts_s0) == 3
        assert abs(valid_pcts_s0[0].item() - 0.0 / 255.0) < 0.01
        assert abs(valid_pcts_s0[1].item() - 64.0 / 255.0) < 0.01
        assert abs(valid_pcts_s0[2].item() - 128.0 / 255.0) < 0.01


# ======================================================================
# 2. Feature ID preservation
# ======================================================================


class TestFeatureIdPreservation:
    """Ensure feature IDs are not corrupted during precomputation."""

    def test_feature_ids_in_valid_range(self):
        """All reconstructed feature IDs are in [0, n_sae)."""
        n_sae = 16384
        # 4 rows, 2 segments each tiling [0, 32)
        seg_starts = t.tensor([0, 16] * 4, dtype=t.uint16)
        seg_ends = t.tensor([16, 32] * 4, dtype=t.uint16)
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (4, 33), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2, 4, 6, 8], dtype=t.int32),
            seg_token_start=seg_starts,
            seg_token_end=seg_ends,
            seg_doc_id=t.zeros(8, dtype=t.uint8),
            seg_feat_offsets=t.arange(0, 25, 3, dtype=t.int32),
            feature_ids=t.randint(0, n_sae, (24,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (24,), dtype=t.uint8),
        )

        rb = reconstruct_batch(shard, [0, 1, 2, 3], _make_identity_selection(shard), 10)
        valid_fids = rb.enc_fids[rb.enc_mask]
        assert (valid_fids >= 0).all()
        assert (valid_fids < n_sae).all()

    def test_feature_ids_round_trip_through_disk(self, tmp_path):
        """Feature IDs survive save → load → reconstruct."""
        known_fids = t.tensor([100, 200, 500, 1000, 15000, 65535], dtype=t.uint16)

        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, 17), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
            seg_token_start=t.tensor([0, 8], dtype=t.uint16),
            seg_token_end=t.tensor([8, 16], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 3, 6], dtype=t.int32),
            feature_ids=known_fids,
            feature_pcts=t.randint(0, 255, (6,), dtype=t.uint8),
        )

        path = save_packed_row_shard(shard, tmp_path / "fid_test")
        loaded = load_packed_row_shard(path)

        assert t.equal(loaded.feature_ids, known_fids)


# ======================================================================
# 3. Token data consistency
# ======================================================================


class TestTokenDataConsistency:
    """Verify token data is not corrupted during precomputation."""

    def test_inputs_targets_offset_by_one(self):
        """inputs = tokens[:T], targets = tokens[1:T+1]."""
        T = 16
        raw_tokens = t.arange(0, T + 1, dtype=t.int32).to(t.uint16).unsqueeze(0)

        shard = PackedRowShard(
            tokens=raw_tokens,
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([T], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 0], dtype=t.int32),
            feature_ids=t.tensor([], dtype=t.uint16),
            feature_pcts=t.tensor([], dtype=t.uint8),
        )

        inputs, targets, *_ = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        expected_inputs = t.arange(0, T, dtype=t.int64).unsqueeze(0)
        expected_targets = t.arange(1, T + 1, dtype=t.int64).unsqueeze(0)

        assert t.equal(inputs, expected_inputs)
        assert t.equal(targets, expected_targets)

    def test_uint16_no_overflow(self):
        """Token values up to 65535 are preserved (no uint16 overflow)."""
        T = 4
        raw_tokens = t.tensor([[0, 32000, 65535, 100, 1]], dtype=t.uint16)

        shard = PackedRowShard(
            tokens=raw_tokens,
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([T], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 0], dtype=t.int32),
            feature_ids=t.tensor([], dtype=t.uint16),
            feature_pcts=t.tensor([], dtype=t.uint8),
        )

        inputs, targets, *_ = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        assert inputs[0, 0].item() == 0
        assert inputs[0, 1].item() == 32000
        assert inputs[0, 2].item() == 65535
        assert inputs[0, 3].item() == 100
        assert targets[0, 0].item() == 32000
        assert targets[0, 3].item() == 1


# ======================================================================
# 4. CSR structural invariants
# ======================================================================


class TestCSRStructuralInvariants:
    """Deep verification of CSR format correctness."""

    def test_row_seg_offsets_cover_all_segments(self):
        """row_seg_offsets[-1] == total_segments."""
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (4, 17), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3, 5, 8, 10], dtype=t.int32),
            seg_token_start=t.zeros(10, dtype=t.uint16),
            seg_token_end=t.full((10,), 16, dtype=t.uint16),
            seg_doc_id=t.zeros(10, dtype=t.uint8),
            seg_feat_offsets=t.arange(0, 31, 3, dtype=t.int32),
            feature_ids=t.randint(0, 100, (30,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (30,), dtype=t.uint8),
        )

        assert shard.row_seg_offsets[-1].item() == 10
        assert shard.total_segments == 10

    def test_feat_offsets_sum_to_total(self):
        """seg_feat_offsets[-1] == total_features."""
        feats_per_seg = [2, 5, 0, 3, 1, 4]  # variable
        offsets = [0]
        for c in feats_per_seg:
            offsets.append(offsets[-1] + c)

        total_feats = sum(feats_per_seg)
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (2, 17), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3, 6], dtype=t.int32),
            seg_token_start=t.zeros(6, dtype=t.uint16),
            seg_token_end=t.full((6,), 16, dtype=t.uint16),
            seg_doc_id=t.zeros(6, dtype=t.uint8),
            seg_feat_offsets=t.tensor(offsets, dtype=t.int32),
            feature_ids=t.randint(0, 100, (total_feats,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (total_feats,), dtype=t.uint8),
        )

        assert shard.seg_feat_offsets[-1].item() == total_feats
        assert shard.total_features == total_feats

    def test_zero_feature_segment(self):
        """Segments with 0 features have equal consecutive offsets."""
        # Segment 1 has 0 features
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, 33), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 3], dtype=t.int32),
            seg_token_start=t.tensor([0, 8, 16], dtype=t.uint16),
            seg_token_end=t.tensor([8, 16, 32], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0, 0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 3, 3, 6], dtype=t.int32),  # seg 1 has 0 feats
            feature_ids=t.randint(0, 100, (6,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (6,), dtype=t.uint8),
        )

        # Segment 1 offset range is [3, 3) → 0 features
        seg1_start = shard.seg_feat_offsets[1].item()
        seg1_end = shard.seg_feat_offsets[2].item()
        assert seg1_start == seg1_end

        # Reconstruction should still work
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        assert rb.enc_mask[0, 1].sum().item() == 0  # segment 1 has no features


# ======================================================================
# 5. Doc ID consistency
# ======================================================================


class TestDocIdConsistency:
    """Verify document IDs are consistent through precomputation."""

    def test_doc_ids_monotonic_within_row(self):
        """Within a packed row, doc IDs should be monotonically non-decreasing."""
        # 2 documents packed into 1 row
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, 33), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 4], dtype=t.int32),
            seg_token_start=t.tensor([0, 8, 16, 24], dtype=t.uint16),
            seg_token_end=t.tensor([8, 16, 24, 32], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0, 1, 1], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 2, 4, 6, 8], dtype=t.int32),
            feature_ids=t.randint(0, 100, (8,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (8,), dtype=t.uint8),
        )

        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        doc_ids = rb.seg_doc_id[0]
        # Check monotonicity
        for i in range(len(doc_ids) - 1):
            assert doc_ids[i] <= doc_ids[i + 1]

    def test_doc_id_uint8_max(self):
        """Doc IDs up to 255 are representable in uint8."""
        # Create a row with doc_id at the max uint8 value
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, 17), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
            seg_token_start=t.tensor([0, 8], dtype=t.uint16),
            seg_token_end=t.tensor([8, 16], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 255], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1, 2], dtype=t.int32),
            feature_ids=t.tensor([10, 20], dtype=t.uint16),
            feature_pcts=t.tensor([100, 200], dtype=t.uint8),
        )

        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        assert rb.seg_doc_id[0, 0].item() == 0
        assert rb.seg_doc_id[0, 1].item() == 255


# ======================================================================
# 6. End-to-end pipeline integrity
# ======================================================================


class TestEndToEndPipelineIntegrity:
    """Full pipeline: documents → precompute → shard → reconstruct → verify."""

    def _make_pipeline_env(self, tmp_path: Path) -> dict:
        """Create a test environment for end-to-end testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        # Create documents with known content (enough to fill buffer)
        texts = [" ".join(f"word{w}" for w in range(n)) for n in [15, 20, 25, 30, 12, 18, 22, 28, 35, 40]]
        stem = "shard_00000"

        pq_path = data_dir / f"{stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path))

        # Build feature shard
        n_docs = len(texts)
        seg_size = 8
        all_fids, all_mags = [], []
        offsets = [0]
        doc_ids, seg_indices, seg_positions = [], [], []
        seg_char_starts, seg_char_ends = [], []
        seg_token_starts, seg_token_ends = [], []

        for doc_idx, text in enumerate(texts):
            tokens = tok.encode(text, prepend=bos)
            char_offsets = token_to_char_offsets(tok, tokens)
            n_tokens = len(tokens)
            n_segs = max(1, (n_tokens + seg_size - 1) // seg_size)
            for s in range(n_segs):
                doc_ids.append(doc_idx)
                seg_indices.append(s)
                seg_positions.append(s * seg_size)
                tok_start = s * seg_size
                tok_end = min((s + 1) * seg_size, n_tokens)
                c_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
                c_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]
                seg_char_starts.append(c_start)
                seg_char_ends.append(c_end)
                seg_token_starts.append(tok_start)
                seg_token_ends.append(tok_end)
                for f in range(3):
                    fid = doc_idx * 10 + s * 3 + f
                    all_fids.append(fid)
                    all_mags.append(0.5 + 0.1 * f)
                offsets.append(len(all_fids))

        total_segs = len(doc_ids)
        shard_data = {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.tensor(all_fids, dtype=t.int32),
            "feature_mags": t.tensor(all_mags, dtype=t.float32),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(seg_indices, dtype=t.int32),
            "segment_positions": t.tensor(seg_positions, dtype=t.int32),
            "doc_token_counts": t.tensor(
                [len(tok.encode(text, prepend=bos)) for text in texts], dtype=t.int32,
            ),
            "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
            "n_docs": n_docs,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": seg_size,
        }
        save_features(shard_data, feature_dir / f"{stem}_features")

        # Create token boundary overlay
        tok_checksum = "test_tok_checksum" + "0" * 47
        tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tok_checksum)
        write_manifest(tok_dir, tok_checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: t.tensor(seg_token_starts, dtype=t.int32),
                TensorKey.SEGMENT_TOKEN_ENDS: t.tensor(seg_token_ends, dtype=t.int32),
            },
            overlay_shard_path(tok_dir, stem),
        )

        return {
            "data_dir": str(data_dir),
            "feature_dir": str(feature_dir),
            "parquet_paths": [str(pq_path)],
            "texts": texts,
            "tokenizer": tok,
            "tok_checksum": tok_checksum,
        }

    def test_pipeline_produces_valid_model_input(self, tmp_path):
        """Full pipeline output is a valid model input batch."""
        env = self._make_pipeline_env(tmp_path)
        tok = env["tokenizer"]
        idx = FeatureIndex(env["feature_dir"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        total, *_ = precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=10,
            rows_per_shard=100,
        )
        assert total > 0

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        B = min(2, shard.n_rows)
        rb = reconstruct_batch(shard, list(range(B)), _make_identity_selection(shard), 10)

        # Verify shapes are compatible with model forward pass
        T = 32
        assert rb.inputs.shape == (B, T)
        assert rb.targets.shape == (B, T)
        assert rb.seg_map.shape == (B, T)

        # seg_map values should be valid segment indices
        n_seg = rb.enc_fids.shape[1]
        assert (rb.seg_map >= 0).all()
        assert (rb.seg_map < n_seg).all()

        # enc_mask should have at least some True values
        assert rb.enc_mask.any(), "No features at all in the batch"

    def test_pipeline_no_nan_values(self, tmp_path):
        """No NaN values in any output tensor."""
        env = self._make_pipeline_env(tmp_path)
        tok = env["tokenizer"]
        idx = FeatureIndex(env["feature_dir"], tokenizer_checksum=env["tok_checksum"])

        output_dir = tmp_path / "packed"
        output_dir.mkdir()

        precompute_packed_rows(
            parquet_paths=env["parquet_paths"],
            tokenizer=tok,
            feature_index=idx,
            context_len=32,
            output_dir=output_dir,
            buffer_size=10,
            rows_per_shard=100,
        )

        shard = load_packed_row_shard(sorted(output_dir.glob("shard_*.safetensors.zst"))[0])
        rb = reconstruct_batch(shard, list(range(min(4, shard.n_rows))), _make_identity_selection(shard), 10)

        for name, tensor in [
            ("inputs", rb.inputs), ("targets", rb.targets), ("seg_map", rb.seg_map),
            ("seg_doc_id", rb.seg_doc_id), ("enc_fids", rb.enc_fids), ("enc_pcts", rb.enc_pcts),
        ]:
            if tensor.is_floating_point():
                assert not tensor.isnan().any(), f"NaN found in {name}"
                assert not tensor.isinf().any(), f"Inf found in {name}"
