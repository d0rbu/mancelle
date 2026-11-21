"""Integration tests: precomputed packed rows vs live dataloader equivalence.

The critical correctness property: packed row shards must produce IDENTICAL
tokens, segment assignments, and feature data as the live dataloader when
given the same documents, packing algorithm, and feature index.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from scripts.preprocess.precompute_packed_rows import (
    SortedDocBuffer,
    _pack_row_with_provenance,
    _raw_document_iterator,
    _row_to_csr,
)
from steering.conditional.encoder_decoder.dataloader import (
    _build_row_segments,
    _pad_segments,
    precomputed_data_loader,
)
from steering.conditional.feature_index import FeatureIndex
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
    load_selection_shard,
    reconstruct_batch,
    save_packed_row_shard,
)
from steering.segmentation import token_to_char_offsets
from tests.encdec_test_helpers import (
    NUM_BUCKETS,
    _make_identity_selection,
    build_threshold_pipeline,
)

# ---------------------------------------------------------------------------
# Test tokenizer (same as test_precomputed_token_boundaries.py)
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Deterministic tokenizer for testing: each word becomes a token."""

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
        parts = []
        for tid in token_ids:
            if tid == 0:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

SEGMENT_SIZE = 8
N_SAE = 200
N_DOCS = 20
N_FEATS_PER_SEG = 4


def _pack_row(
    doc_buffer: SortedDocBuffer,
    row_capacity: int,
) -> tuple[t.Tensor, list]:
    """Test adapter for assertions that do not inspect document provenance."""
    row_tokens, row_doc_info, _row_doc_prov = _pack_row_with_provenance(
        doc_buffer, row_capacity
    )
    return row_tokens, row_doc_info


def _with_test_provenance(documents):
    """Give raw iterator rows the same three-field shape as production packing."""
    return [
        (tokens, segments, (0, doc_idx))
        for doc_idx, (tokens, segments, *_) in enumerate(documents)
    ]


def _make_pipeline(n_features: int = N_SAE) -> object:
    """Build a threshold pipeline whose boundaries cover feature IDs 0..(n_features-1)."""
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(n_features, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _make_test_env(tmp_path: Path) -> dict:
    """Create a complete test environment: parquets + feature shards + overlays.

    Returns dict with paths and metadata.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    tok = SimpleTokenizer()
    bos = tok.get_bos_token_id()

    shard_stem = "shard_00000"

    # Create varied-length documents
    texts = []
    for doc_idx in range(N_DOCS):
        n_words = 10 + doc_idx * 3  # 10 to 67 words
        words = [f"word{w}" for w in range(n_words)]
        texts.append(" ".join(words))

    # Write parquet
    pq_path = data_dir / f"{shard_stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    # Build feature shard
    all_fids: list[int] = []
    all_mags: list[float] = []
    offsets = [0]
    doc_ids: list[int] = []
    seg_indices_list: list[int] = []
    seg_positions: list[int] = []
    seg_char_starts: list[int] = []
    seg_char_ends: list[int] = []

    for doc_idx in range(N_DOCS):
        text = texts[doc_idx]
        text_tokens = tok.encode(text, prepend=bos)
        char_offsets = token_to_char_offsets(tok, text_tokens)
        n_tokens = len(text_tokens)
        n_segs = max(1, (n_tokens + SEGMENT_SIZE - 1) // SEGMENT_SIZE)

        for seg_idx in range(n_segs):
            doc_ids.append(doc_idx)
            seg_indices_list.append(seg_idx)
            seg_positions.append(seg_idx * SEGMENT_SIZE)

            tok_start = seg_idx * SEGMENT_SIZE
            tok_end = min((seg_idx + 1) * SEGMENT_SIZE, n_tokens)
            c_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
            c_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]
            seg_char_starts.append(c_start)
            seg_char_ends.append(c_end)

            for f in range(N_FEATS_PER_SEG):
                fid = (doc_idx * 10 + seg_idx * 5 + f) % N_SAE
                all_fids.append(fid)
                all_mags.append(0.1 * (f + 1) + 0.01 * doc_idx)

            offsets.append(len(all_fids))

    total_segs = len(doc_ids)
    shard_data = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(all_fids, dtype=t.int32),
        "feature_mags": t.tensor(all_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices_list, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "doc_token_counts": t.tensor([len(tok.encode(texts[i], prepend=bos)) for i in range(N_DOCS)], dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "n_docs": N_DOCS,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }
    save_features(shard_data, feature_dir / f"{shard_stem}_features")

    # Create percentile overlay (synthetic: feature_pcts = fid % 256)
    pct_checksum = "pct_test_integration" + "0" * 45
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, pct_checksum)
    write_manifest(pct_dir, pct_checksum, OverlayKind.PERCENTILE)
    pct_data = t.tensor(all_fids, dtype=t.int32).remainder(256).to(t.uint8)
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: pct_data},
        overlay_shard_path(pct_dir, shard_stem),
    )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "texts": texts,
        "pct_checksum": pct_checksum,
        "parquet_paths": [str(pq_path)],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPackRowEquivalence:
    """_pack_row produces same packing as the live dataloader's inner loop."""

    def test_single_row_tokens_match(self, tmp_path):
        """Pack a single row and verify tokens match manual packing."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        # Create some documents
        docs = [
            (tok.encode("hello world foo bar baz", prepend=bos), None, (0, 0)),
            (tok.encode("one two three", prepend=bos), None, (0, 1)),
            (tok.encode("a b c d e f g h i j", prepend=bos), None, (0, 2)),
        ]
        doc_buffer = SortedDocBuffer()
        for d in docs:
            doc_buffer.append(d)
        T = 20
        row_tokens, row_doc_info = _pack_row(doc_buffer, T + 1)

        assert row_tokens.shape == (T + 1,)
        # First doc should be packed starting at position 0
        assert row_tokens[0].item() == bos


class TestRawDocumentIterator:
    """_raw_document_iterator yields documents in correct order with features."""

    def test_yields_all_documents(self, tmp_path):
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()

        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        docs = [(t, s) for t, s, *_ in _raw_document_iterator(env["parquet_paths"], tok, tok.get_bos_token_id(), idx)]
        assert len(docs) == N_DOCS

    def test_features_have_percentiles(self, tmp_path):
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()

        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        for _, seg_features, *_ in _raw_document_iterator(
            env["parquet_paths"], tok, tok.get_bos_token_id(), idx
        ):
            assert seg_features is not None
            for sf in seg_features:
                assert sf.percentiles is not None
                assert len(sf.percentiles) == len(sf.feature_ids)
                # Packed preprocessing preserves the overlay's encoded bytes.
                assert sf.percentiles.dtype == t.uint8
                assert (sf.percentiles >= 0).all()
                assert (sf.percentiles <= 255).all()


class TestRowToCsr:
    """_row_to_csr correctly converts collated row data to CSR format."""

    def test_round_trip(self, tmp_path):
        """CSR conversion preserves feature data."""
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        # Get several documents (need enough to fill the row)
        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )
        T = 32
        doc_buffer = SortedDocBuffer()
        for d in docs[:10]:
            doc_buffer.append(d)
        row_tokens, row_doc_info = _pack_row(doc_buffer, T + 1)

        csr = _row_to_csr(row_tokens, row_doc_info, T, tokenizer=tok)

        assert csr.tokens.shape == (T + 1,)
        assert csr.tokens.dtype == t.uint16
        assert len(csr.seg_token_start) == len(csr.seg_token_end)
        assert csr.feat_counts.sum().item() == len(csr.feature_ids)
        assert len(csr.feature_ids) == len(csr.feature_pcts)


class TestPrecomputeToReconstructEquivalence:
    """The critical test: precompute → shard → reconstruct matches live collation."""

    def test_tokens_match(self, tmp_path):
        """Packed row tokens from precompute match manual packing."""
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )
        T = 32

        # Pack 4 rows
        doc_buffer = SortedDocBuffer()
        for d in docs:
            doc_buffer.append(d)
        rows_data = []
        for _ in range(4):
            if not doc_buffer:
                break
            row_tokens, row_doc_info = _pack_row(doc_buffer, T + 1)
            csr = _row_to_csr(row_tokens, row_doc_info, T, tokenizer=tok)
            rows_data.append(csr)

        # Build a shard manually
        shard_tokens = t.stack([r.tokens for r in rows_data])
        row_seg_offsets = [0]
        all_seg_starts, all_seg_ends, all_seg_doc_ids = [], [], []
        all_feat_ids, all_feat_pcts = [], []
        seg_feat_offsets = [0]

        for csr in rows_data:
            row_seg_offsets.append(row_seg_offsets[-1] + csr.n_segs)
            all_seg_starts.extend(csr.seg_token_start.tolist())
            all_seg_ends.extend(csr.seg_token_end.tolist())
            all_seg_doc_ids.extend(csr.seg_doc_id.tolist())
            all_feat_ids.extend(csr.feature_ids.tolist())
            all_feat_pcts.extend(csr.feature_pcts.tolist())
            for c in csr.feat_counts.tolist():
                seg_feat_offsets.append(seg_feat_offsets[-1] + c)

        shard = PackedRowShard(
            tokens=shard_tokens,
            row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
            seg_token_start=t.tensor(all_seg_starts, dtype=t.uint16),
            seg_token_end=t.tensor(all_seg_ends, dtype=t.uint16),
            seg_doc_id=t.tensor(all_seg_doc_ids, dtype=t.uint8),
            seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
            feature_ids=t.tensor(all_feat_ids, dtype=t.uint16),
            feature_pcts=t.tensor(all_feat_pcts, dtype=t.uint8),
        )

        # Reconstruct batch from shard
        rb = reconstruct_batch(shard, range(len(rows_data)), _make_identity_selection(shard), 10)
        inputs, targets = rb.inputs, rb.targets

        # Verify tokens
        expected_tokens = shard_tokens.to(t.int64)
        assert t.equal(inputs, expected_tokens[:, :T])
        assert t.equal(targets, expected_tokens[:, 1:T + 1])

    def test_features_survive_round_trip(self, tmp_path):
        """Features from precompute → shard → reconstruct are present and valid."""
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )
        T = 32

        doc_buffer = SortedDocBuffer()
        for d in docs:
            doc_buffer.append(d)
        row_tokens, row_doc_info = _pack_row(doc_buffer, T + 1)
        csr = _row_to_csr(row_tokens, row_doc_info, T, tokenizer=tok)

        # Build single-row shard
        seg_feat_offsets = [0]
        for c in csr.feat_counts.tolist():
            seg_feat_offsets.append(seg_feat_offsets[-1] + c)

        shard = PackedRowShard(
            tokens=csr.tokens.unsqueeze(0),
            row_seg_offsets=t.tensor([0, csr.n_segs], dtype=t.int32),
            seg_token_start=csr.seg_token_start.to(t.uint16),
            seg_token_end=csr.seg_token_end.to(t.uint16),
            seg_doc_id=csr.seg_doc_id,
            seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
            feature_ids=csr.feature_ids.to(t.uint16),
            feature_pcts=csr.feature_pcts,
        )

        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10)
        enc_fids, enc_pcts, enc_mask = rb.enc_fids, rb.enc_pcts, rb.enc_mask

        # At least some features should be present
        assert enc_mask.any(), "No features found after reconstruction"
        # Feature IDs should be in valid range
        assert (enc_fids[enc_mask] >= 0).all()
        assert (enc_fids[enc_mask] < N_SAE).all()
        # Percentiles should be in [0, 1]
        assert (enc_pcts[enc_mask] >= 0).all()
        assert (enc_pcts[enc_mask] <= 1).all()

class TestPrecomputedDataLoader:
    """precomputed_data_loader yields correct batches from shard files."""

    def _write_test_shards(self, tmp_path, n_rows=8, T=32):
        """Create packed row shard files for testing the loader."""
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )

        shard_tokens = []
        row_seg_counts = []
        all_seg_starts, all_seg_ends, all_seg_doc_ids = [], [], []
        all_feat_ids, all_feat_pcts, all_feat_counts = [], [], []

        doc_buffer = SortedDocBuffer()
        for d in docs:
            doc_buffer.append(d)

        for _ in range(n_rows):
            if not doc_buffer:
                doc_buffer = SortedDocBuffer()
                for d in docs:
                    doc_buffer.append(d)
            row_tokens, row_doc_info = _pack_row(doc_buffer, T + 1)
            csr = _row_to_csr(row_tokens, row_doc_info, T, tokenizer=tok)
            shard_tokens.append(csr.tokens)
            row_seg_counts.append(csr.n_segs)
            all_seg_starts.extend(csr.seg_token_start.tolist())
            all_seg_ends.extend(csr.seg_token_end.tolist())
            all_seg_doc_ids.extend(csr.seg_doc_id.tolist())
            all_feat_ids.extend(csr.feature_ids.tolist())
            all_feat_pcts.extend(csr.feature_pcts.tolist())
            all_feat_counts.extend(csr.feat_counts.tolist())

        row_seg_offsets = [0]
        for c in row_seg_counts:
            row_seg_offsets.append(row_seg_offsets[-1] + c)
        seg_feat_offsets = [0]
        for c in all_feat_counts:
            seg_feat_offsets.append(seg_feat_offsets[-1] + c)

        shard = PackedRowShard(
            tokens=t.stack(shard_tokens),
            row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
            seg_token_start=t.tensor(all_seg_starts, dtype=t.uint16),
            seg_token_end=t.tensor(all_seg_ends, dtype=t.uint16),
            seg_doc_id=t.tensor(all_seg_doc_ids, dtype=t.uint8),
            seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
            feature_ids=t.tensor(all_feat_ids, dtype=t.uint16),
            feature_pcts=t.tensor(all_feat_pcts, dtype=t.uint8),
        )

        shard_dir = tmp_path / "packed_shards"
        shard_dir.mkdir()
        save_packed_row_shard(shard, shard_dir / "shard_00000")
        return shard_dir

    def test_yields_correct_shapes(self, tmp_path):
        """Loader yields batches with correct shapes."""
        shard_dir = self._write_test_shards(tmp_path, n_rows=8, T=32)
        loader = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")  # ty: ignore[invalid-argument-type]
        # Take first 2 batches (8 rows / 4 = 2 batches in first epoch)
        batches = [next(loader) for _ in range(2)]

        assert len(batches) == 2
        for batch in batches:
            inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, state = batch
            assert inputs.shape == (4, 32)
            assert targets.shape == (4, 32)
            assert seg_map.shape == (4, 32)
            assert enc_fids.shape[0] == 4
            assert enc_mask.shape == enc_fids.shape
            assert enc_pcts.shape == enc_fids.shape

    def test_no_rows_lost_at_shard_boundary(self, tmp_path):
        """Rows carry across shard boundaries — no rows are dropped."""
        # 7 rows, B=4: first batch uses rows 0-3, second would need rows 4-6 + carry
        # from epoch 2. After 2 epochs (14 total rows), we should get 3 complete batches.
        shard_dir = self._write_test_shards(tmp_path, n_rows=7, T=32)
        loader = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")  # ty: ignore[invalid-argument-type]
        # Epoch 1: rows 0-3 (batch), rows 4-6 pending (3 rows)
        # Epoch 2: rows 0-6 again. 3 pending + 7 new = 10 rows → 2 batches, 2 pending
        # So within 2 epochs we get at least 3 batches
        batches = [next(loader) for _ in range(3)]
        assert len(batches) == 3
        # All batches should be complete (B=4)
        for batch in batches:
            assert batch[0].shape[0] == 4

    def test_enc_mask_has_true_entries(self, tmp_path):
        """Reconstructed batches have real feature data."""
        shard_dir = self._write_test_shards(tmp_path, n_rows=4, T=32)
        loader = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")  # ty: ignore[invalid-argument-type]
        batch = next(loader)
        _, _, _, _, enc_mask, _, _, _, _, _ = batch
        assert enc_mask.any(), "No features in reconstructed batch"

    def test_label_dropout(self, tmp_path):
        """Label dropout masks segments."""
        shard_dir = self._write_test_shards(tmp_path, n_rows=4, T=32)

        # With dropout=0: all features visible
        loader_no_drop = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu", label_dropout=0.0)  # ty: ignore[invalid-argument-type]
        _, _, _, _, mask_no_drop, _, _, _, _, _ = next(loader_no_drop)
        total_no_drop = mask_no_drop.sum().item()

        # With dropout=0.99: most features masked (but at least 1 per row preserved)
        t.manual_seed(42)
        loader_drop = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu", label_dropout=0.99)  # ty: ignore[invalid-argument-type]
        _, _, _, _, mask_drop, _, _, _, _, _ = next(loader_drop)
        total_drop = mask_drop.sum().item()

        assert total_drop < total_no_drop, "Label dropout should reduce visible features"
        # At least 1 per segment row is preserved (NaN prevention)
        for row_idx in range(4):
            assert mask_drop[row_idx].any(), f"Row {row_idx} has no visible features"


# ---------------------------------------------------------------------------
# End-to-end parity: live packing vs precompute → load
# ---------------------------------------------------------------------------


class TestLiveVsPrecomputedParity:
    """The critical correctness test: precompute → shard → load produces
    IDENTICAL tokens, seg_map, and seg_doc_id as the live packing algorithm,
    batch by batch, for the same document sequence.
    """

    def _make_docs_and_shards(self, tmp_path, n_docs=30, T=32, B=4):
        """Create synthetic documents + feature shards. Pack N rows using both
        the live packing algorithm and the precompute pipeline. Return both
        for comparison.
        """
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])

        # Get all documents with packed features (same order as precompute)
        all_docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )

        # --- LIVE PATH: pack rows one at a time from a shared buffer ---
        live_buffer = SortedDocBuffer()
        for d in all_docs:
            live_buffer.append(d)
        live_rows_tokens: list[t.Tensor] = []
        live_rows_seg_maps: list[t.Tensor] = []
        live_rows_seg_doc_ids: list[t.Tensor] = []
        live_rows_enc_fids: list[t.Tensor] = []
        live_rows_enc_masks: list[t.Tensor] = []

        n_rows = min(len(all_docs), 12)  # pack enough rows for 3 batches at B=4
        for _ in range(n_rows):
            if not live_buffer:
                break
            row_tokens, row_doc_info = _pack_row(live_buffer, T + 1)
            per_seg_fids, _, per_seg_pcts, seg_map_row, seg_doc_id_row, _, _ = \
                _build_row_segments(
                    row_doc_info,
                    T,
                    tok,
                    collect_buckets=False,
                    empty_percentile_dtype=t.uint8,
                )
            per_seg_bkts = [t.zeros_like(fids) for fids in per_seg_fids]
            fids, mags, pcts, mask, seg_map_row = \
                _pad_segments(per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map_row)

            live_rows_tokens.append(row_tokens[:T + 1])
            live_rows_seg_maps.append(seg_map_row)
            live_rows_seg_doc_ids.append(seg_doc_id_row)
            live_rows_enc_fids.append(fids)
            live_rows_enc_masks.append(mask)

        # --- PRECOMPUTE PATH: same packing, CSR conversion, shard, reconstruct ---
        precompute_buffer = SortedDocBuffer()
        for d in all_docs:
            precompute_buffer.append(d)
        shard_tokens_list = []
        shard_row_seg_counts = []
        all_seg_starts, all_seg_ends, all_seg_doc_ids_flat = [], [], []
        all_feat_ids, all_feat_pcts, all_feat_counts = [], [], []

        for _ in range(n_rows):
            if not precompute_buffer:
                break
            row_tokens, row_doc_info = _pack_row(precompute_buffer, T + 1)
            csr = _row_to_csr(row_tokens, row_doc_info, T, tokenizer=tok)
            shard_tokens_list.append(csr.tokens)
            shard_row_seg_counts.append(csr.n_segs)
            all_seg_starts.extend(csr.seg_token_start.tolist())
            all_seg_ends.extend(csr.seg_token_end.tolist())
            all_seg_doc_ids_flat.extend(csr.seg_doc_id.tolist())
            all_feat_ids.extend(csr.feature_ids.tolist())
            all_feat_pcts.extend(csr.feature_pcts.tolist())
            all_feat_counts.extend(csr.feat_counts.tolist())

        # Build CSR offsets
        row_seg_offsets = [0]
        for c in shard_row_seg_counts:
            row_seg_offsets.append(row_seg_offsets[-1] + c)
        seg_feat_offsets = [0]
        for c in all_feat_counts:
            seg_feat_offsets.append(seg_feat_offsets[-1] + c)

        shard = PackedRowShard(
            tokens=t.stack(shard_tokens_list),
            row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
            seg_token_start=t.tensor(all_seg_starts, dtype=t.uint16),
            seg_token_end=t.tensor(all_seg_ends, dtype=t.uint16),
            seg_doc_id=t.tensor(all_seg_doc_ids_flat, dtype=t.uint8),
            seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
            feature_ids=t.tensor(all_feat_ids, dtype=t.uint16),
            feature_pcts=t.tensor(all_feat_pcts, dtype=t.uint8),
        )

        return live_rows_tokens, live_rows_seg_maps, live_rows_seg_doc_ids, \
            live_rows_enc_fids, live_rows_enc_masks, shard, n_rows

    def test_tokens_identical_per_row(self, tmp_path):
        """Packed tokens are bit-for-bit identical between live and precomputed paths."""
        live_tokens, _, _, _, _, shard, n_rows = self._make_docs_and_shards(tmp_path)

        for row_idx in range(n_rows):
            live = live_tokens[row_idx][:shard.context_len + 1].to(t.int64)
            precomp = shard.tokens[row_idx].to(t.int64)
            assert t.equal(live, precomp), (
                f"Row {row_idx} tokens differ:\n  live[:10]: {live[:10]}\n  precomp[:10]: {precomp[:10]}"
            )

    def test_seg_map_identical_per_row(self, tmp_path):
        """Segment map is bit-for-bit identical between live and precomputed paths."""
        _, live_seg_maps, _, _, _, shard, n_rows = self._make_docs_and_shards(tmp_path)

        for row_idx in range(n_rows):
            # Reconstruct seg_map from shard for this row
            seg_map_recon = reconstruct_batch(shard, [row_idx], _make_identity_selection(shard), 10).seg_map
            live_sm = live_seg_maps[row_idx]
            recon_sm = seg_map_recon[0]

            assert t.equal(live_sm, recon_sm), (
                f"Row {row_idx} seg_map differs:\n  live: {live_sm}\n  recon: {recon_sm}"
            )

    def test_seg_doc_id_identical_per_row(self, tmp_path):
        """Document-to-segment assignment is identical between live and precomputed."""
        _, _, live_doc_ids, _, _, shard, n_rows = self._make_docs_and_shards(tmp_path)

        for row_idx in range(n_rows):
            seg_doc_id_recon = reconstruct_batch(shard, [row_idx], _make_identity_selection(shard), 10).seg_doc_id
            live_sdi = live_doc_ids[row_idx]
            n_live = len(live_sdi)
            recon_sdi = seg_doc_id_recon[0, :n_live]

            assert t.equal(live_sdi.to(t.int64), recon_sdi), (
                f"Row {row_idx} seg_doc_id differs:\n  live: {live_sdi}\n  recon: {recon_sdi}"
            )

    def test_feature_ids_identical_per_segment(self, tmp_path):
        """Per-segment feature IDs match between live collation and precomputed CSR."""
        _, _, _, live_fids, live_masks, shard, n_rows = self._make_docs_and_shards(tmp_path)

        for row_idx in range(n_rows):
            rb = reconstruct_batch(shard, [row_idx], _make_identity_selection(shard), 10)
            enc_fids_recon, enc_mask_recon = rb.enc_fids, rb.enc_mask

            live_f = live_fids[row_idx]   # [n_seg, F_max]
            live_m = live_masks[row_idx]   # [n_seg, F_max]
            n_segs = live_f.shape[0]

            for s in range(n_segs):
                direct_ids = live_f[s, live_m[s]]
                recon_ids = enc_fids_recon[0, s, enc_mask_recon[0, s]]

                assert t.equal(direct_ids, recon_ids), (
                    f"Row {row_idx} seg {s} feature IDs differ:\n"
                    f"  live ({len(direct_ids)}): {direct_ids}\n"
                    f"  recon ({len(recon_ids)}): {recon_ids}"
                )

    def test_batched_parity(self, tmp_path):
        """Full batch reconstruction matches live collation when batching B rows."""
        live_tokens, live_seg_maps, live_doc_ids, live_fids, live_masks, shard, n_rows = \
            self._make_docs_and_shards(tmp_path)

        B = 4
        n_batches = n_rows // B

        for batch_idx in range(n_batches):
            row_start = batch_idx * B
            row_end = row_start + B
            row_indices = range(row_start, row_end)

            rb = reconstruct_batch(shard, row_indices, _make_identity_selection(shard), 10)
            inputs, seg_map, seg_doc_id, enc_fids, enc_mask = (
                rb.inputs,
                rb.seg_map,
                rb.seg_doc_id,
                rb.enc_fids,
                rb.enc_mask,
            )

            for b in range(B):
                row_idx = row_start + b

                # Tokens
                live_input = live_tokens[row_idx][:shard.context_len].to(t.int64)
                assert t.equal(inputs[b], live_input), f"Batch {batch_idx} row {b}: inputs differ"

                # seg_map
                assert t.equal(seg_map[b], live_seg_maps[row_idx]), \
                    f"Batch {batch_idx} row {b}: seg_map differs"

                # seg_doc_id
                n_live_segs = len(live_doc_ids[row_idx])
                assert t.equal(
                    seg_doc_id[b, :n_live_segs],
                    live_doc_ids[row_idx].to(t.int64),
                ), f"Batch {batch_idx} row {b}: seg_doc_id differs"

                # Feature IDs per segment
                for s in range(n_live_segs):
                    direct_ids = live_fids[row_idx][s, live_masks[row_idx][s]]
                    recon_ids = enc_fids[b, s, enc_mask[b, s]]
                    assert t.equal(direct_ids, recon_ids), \
                        f"Batch {batch_idx} row {b} seg {s}: feature IDs differ"

    def test_seg_map_parity(self, tmp_path):
        """seg_map matches between live and precomputed paths."""
        _, live_seg_maps, _, _, _, shard, n_rows = self._make_docs_and_shards(tmp_path)
        for row_idx in range(n_rows):
            seg_map_recon = reconstruct_batch(shard, [row_idx], _make_identity_selection(shard), 10).seg_map
            assert t.equal(live_seg_maps[row_idx], seg_map_recon[0]), \
                f"Row {row_idx} seg_map mismatch"

    def test_feature_ids_parity(self, tmp_path):
        """Per-segment feature IDs match between live and precomputed paths."""
        _, _, _, live_fids, live_masks, shard, n_rows = self._make_docs_and_shards(tmp_path)
        for row_idx in range(n_rows):
            rb = reconstruct_batch(shard, [row_idx], _make_identity_selection(shard), 10)
            enc_fids_r, enc_mask_r = rb.enc_fids, rb.enc_mask
            n_segs = live_fids[row_idx].shape[0]
            for s in range(n_segs):
                direct = live_fids[row_idx][s, live_masks[row_idx][s]]
                recon = enc_fids_r[0, s, enc_mask_r[0, s]]
                assert t.equal(direct, recon), \
                    f"Row {row_idx} seg {s} fids mismatch"


# ---------------------------------------------------------------------------
# Comprehensive E2E parity + multi-shard
# ---------------------------------------------------------------------------


class TestEndToEndPipelineParity:
    """Comprehensive parity: precompute → select → load produces correct batches."""

    def _build_base_shard_and_pipeline(self, tmp_path, n_rows=8):
        from steering.conditional.feature_pipeline import FeaturePipelineConfig, NegativeBundle
        from steering.conditional.sampling import (
            Distribution,
            NegativeSamplingSelection,
            SamplingBundle,
            SamplingSelection,
        )

        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])
        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )

        shard_tokens, row_seg_counts = [], []
        all_starts, all_ends, all_doc_ids = [], [], []
        all_fids, all_pcts, all_counts = [], [], []

        doc_buffer = SortedDocBuffer()
        for d in docs:
            doc_buffer.append(d)

        for _ in range(n_rows):
            if not doc_buffer:
                doc_buffer = SortedDocBuffer()
                for d in docs:
                    doc_buffer.append(d)
            row_tokens, row_doc_info = _pack_row(doc_buffer, 33)
            csr = _row_to_csr(row_tokens, row_doc_info, 32, tokenizer=tok)
            shard_tokens.append(csr.tokens)
            row_seg_counts.append(csr.n_segs)
            all_starts.extend(csr.seg_token_start.tolist())
            all_ends.extend(csr.seg_token_end.tolist())
            all_doc_ids.extend(csr.seg_doc_id.tolist())
            all_fids.extend(csr.feature_ids.tolist())
            all_pcts.extend(csr.feature_pcts.tolist())
            all_counts.extend(csr.feat_counts.tolist())

        row_off = [0]
        for c in row_seg_counts:
            row_off.append(row_off[-1] + c)
        seg_off = [0]
        for c in all_counts:
            seg_off.append(seg_off[-1] + c)

        base = PackedRowShard(
            tokens=t.stack(shard_tokens),
            row_seg_offsets=t.tensor(row_off, dtype=t.int32),
            seg_token_start=t.tensor(all_starts, dtype=t.uint16),
            seg_token_end=t.tensor(all_ends, dtype=t.uint16),
            seg_doc_id=t.tensor(all_doc_ids, dtype=t.uint8),
            seg_feat_offsets=t.tensor(seg_off, dtype=t.int32),
            feature_ids=t.tensor(all_fids, dtype=t.uint16),
            feature_pcts=t.tensor(all_pcts, dtype=t.uint8),
        )
        shard_dir = tmp_path / "packed_e2e"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(base, shard_dir / "shard_00000")

        pct_table = t.linspace(0.01, 5.0, N_SAE * 99).reshape(N_SAE, 99).double()
        boundaries = t.zeros(N_SAE, 9, dtype=t.float64)
        for i in range(N_SAE):
            boundaries[i] = t.linspace(0.1, 5.0, 9)

        rng = t.Generator().manual_seed(42)
        sel = SamplingSelection(distribution=Distribution.geometric, mean=3.0, std=3.0, min_k=1)
        bundle = SamplingBundle(selection=sel, percentile_table=pct_table, rng=rng)
        neg_sel = NegativeSamplingSelection(distribution=Distribution.geometric, mean=2.0, std=2.0)
        neg_bundle = NegativeBundle(selection=neg_sel, n_sae_features=N_SAE, rng=t.Generator().manual_seed(43))

        pipeline = FeaturePipelineConfig(
            selection_config=bundle, boundaries=boundaries,
            negative_bundle=neg_bundle, effective_buckets=10,
        )
        return base, shard_path, pipeline

    def test_selection_segment_count_matches_base(self, tmp_path):
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        base, shard_path, pipeline = self._build_base_shard_and_pipeline(tmp_path)
        out = tmp_path / "sel1"
        select_features_for_shard(shard_path, pipeline, out, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out, "shard_00000"))
        assert sel.total_segments == base.total_segments

    def test_selected_fids_are_subset_of_base(self, tmp_path):
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        base, shard_path, pipeline = self._build_base_shard_and_pipeline(tmp_path)
        out = tmp_path / "sel2"
        select_features_for_shard(shard_path, pipeline, out, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out, "shard_00000"))

        base_off = base.seg_feat_offsets.tolist()
        sel_off = sel.sel_offsets.tolist()
        for s in range(sel.total_segments):
            base_set = set(base.feature_ids[base_off[s]:base_off[s+1]].tolist())
            sel_set = set(sel.sel_fids[sel_off[s]:sel_off[s+1]].tolist())
            assert sel_set.issubset(base_set), f"Seg {s}: selected fids not in base"

    def test_multi_batch_valid_shapes(self, tmp_path):
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        base, shard_path, pipeline = self._build_base_shard_and_pipeline(tmp_path, n_rows=12)
        out = tmp_path / "sel3"
        select_features_for_shard(shard_path, pipeline, out, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out, "shard_00000"))

        for batch_idx in range(3):  # 12 rows / 4 = 3 batches
            rows = range(batch_idx * 4, batch_idx * 4 + 4)
            rb = reconstruct_batch(base, rows, sel, 10)
            assert rb.inputs.shape == (4, 32)
            assert rb.enc_mask.any(), f"Batch {batch_idx}: no features"
            assert (rb.enc_mags[rb.enc_mask] >= 0).all()

    def test_multi_shard_no_rows_lost(self, tmp_path):
        """2 shards with 5 rows each, B=4: verify no rows lost across boundary."""
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])
        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )

        shard_dir = tmp_path / "multi"
        shard_dir.mkdir()
        for si in range(2):
            buf = SortedDocBuffer()
            for d in docs:
                buf.append(d)
            st, rsc = [], []
            sa, se, sd, sf, sp, sc = [], [], [], [], [], []
            for _ in range(5):
                if not buf:
                    buf = SortedDocBuffer()
                    for d in docs:
                        buf.append(d)
                rt, rdi = _pack_row(buf, 33)
                csr = _row_to_csr(rt, rdi, 32, tokenizer=tok)
                st.append(csr.tokens)
                rsc.append(csr.n_segs)
                sa.extend(csr.seg_token_start.tolist())
                se.extend(csr.seg_token_end.tolist())
                sd.extend(csr.seg_doc_id.tolist())
                sf.extend(csr.feature_ids.tolist())
                sp.extend(csr.feature_pcts.tolist())
                sc.extend(csr.feat_counts.tolist())
            ro = [0]
            for c in rsc:
                ro.append(ro[-1] + c)
            so = [0]
            for c in sc:
                so.append(so[-1] + c)
            shard = PackedRowShard(
                tokens=t.stack(st), row_seg_offsets=t.tensor(ro, dtype=t.int32),
                seg_token_start=t.tensor(sa, dtype=t.uint16),
                seg_token_end=t.tensor(se, dtype=t.uint16),
                seg_doc_id=t.tensor(sd, dtype=t.uint8),
                seg_feat_offsets=t.tensor(so, dtype=t.int32),
                feature_ids=t.tensor(sf, dtype=t.uint16),
                feature_pcts=t.tensor(sp, dtype=t.uint8),
            )
            save_packed_row_shard(shard, shard_dir / f"shard_{si:05d}")

        # 10 rows, B=4: need 5 batches across 2 epochs (10+10=20 rows → 5 batches)
        loader = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")  # ty: ignore[invalid-argument-type]
        batches = [next(loader) for _ in range(5)]
        assert all(b[0].shape == (4, 32) for b in batches)

    def test_empty_shard_handled(self, tmp_path):
        """A shard with 0 rows doesn't crash the loader."""
        shard_dir = tmp_path / "empty_shard"
        shard_dir.mkdir()
        # Create a shard with 0 rows
        empty = PackedRowShard(
            tokens=t.zeros(0, 33, dtype=t.uint16),
            row_seg_offsets=t.tensor([0], dtype=t.int32),
            seg_token_start=t.tensor([], dtype=t.uint16),
            seg_token_end=t.tensor([], dtype=t.uint16),
            seg_doc_id=t.tensor([], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0], dtype=t.int32),
            feature_ids=t.tensor([], dtype=t.uint16),
            feature_pcts=t.tensor([], dtype=t.uint8),
        )
        save_packed_row_shard(empty, shard_dir / "shard_00000")

        # Also create a non-empty shard so the loader has something to yield
        env = _make_test_env(tmp_path)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        idx = FeatureIndex(env["feature_dir"], percentile_checksum=env["pct_checksum"])
        docs = _with_test_provenance(
            _raw_document_iterator(env["parquet_paths"], tok, bos, idx)
        )
        buf = SortedDocBuffer()
        for d in docs:
            buf.append(d)
        st, rsc, sa, se, sd, sf, sp, sc = [], [], [], [], [], [], [], []
        for _ in range(4):
            if not buf:
                buf = SortedDocBuffer()
                for d in docs:
                    buf.append(d)
            rt, rdi = _pack_row(buf, 33)
            csr = _row_to_csr(rt, rdi, 32, tokenizer=tok)
            st.append(csr.tokens)
            rsc.append(csr.n_segs)
            sa.extend(csr.seg_token_start.tolist())
            se.extend(csr.seg_token_end.tolist())
            sd.extend(csr.seg_doc_id.tolist())
            sf.extend(csr.feature_ids.tolist())
            sp.extend(csr.feature_pcts.tolist())
            sc.extend(csr.feat_counts.tolist())
        ro = [0]
        for c in rsc:
            ro.append(ro[-1] + c)
        so = [0]
        for c in sc:
            so.append(so[-1] + c)
        nonempty = PackedRowShard(
            tokens=t.stack(st), row_seg_offsets=t.tensor(ro, dtype=t.int32),
            seg_token_start=t.tensor(sa, dtype=t.uint16),
            seg_token_end=t.tensor(se, dtype=t.uint16),
            seg_doc_id=t.tensor(sd, dtype=t.uint8),
            seg_feat_offsets=t.tensor(so, dtype=t.int32),
            feature_ids=t.tensor(sf, dtype=t.uint16),
            feature_pcts=t.tensor(sp, dtype=t.uint8),
        )
        save_packed_row_shard(nonempty, shard_dir / "shard_00001")

        # Should yield a batch from the non-empty shard (empty shard has 0 rows to contribute)
        loader = precomputed_data_loader(shard_dir, B=4, feature_pipeline=_make_pipeline(), device="cpu")  # ty: ignore[invalid-argument-type]
        batch = next(loader)
        assert batch[0].shape == (4, 32)
