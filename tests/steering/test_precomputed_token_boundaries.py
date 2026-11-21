"""Tests for precomputed nanochat token boundary wiring.

Verifies:
1. Exact equivalence: precomputed token boundaries produce IDENTICAL seg_map, fids, mask
   as the current char-level path
2. BOS consistency: BOS token assigned to segment 0 in both paths
3. Cropped doc: boundaries correct when doc is cropped to fewer tokens
4. Multi-segment: multiple segments, identical assignments both paths
5. Fallback: missing/mismatched checksum → falls back to char-level path
6. Round-trip: precompute for a shard, load back, verify matches on-the-fly
7. Checksum mismatch: wrong checksum → boundaries ignored
8. FeatureIndex integration: tokenizer_checksum param wires correctly
9. compute_tokenizer_checksum: deterministic, different tokenizer → different hash
10. Parallel precomputation: sequential vs parallel produce identical results
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from errors import EmptyExtractionSegmentError
from scripts.preprocess.precompute_token_boundaries import _init_worker, precompute_shard
from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
)
from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import (
    TensorKey,
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
from steering.segmentation import (
    SegmentMode,
    assign_tokens_to_segments,
    compute_tokenizer_checksum,
    token_to_char_offsets,
)
from steering.stats_io import (
    load_tokenizer_checksum,
    save_tokenizer_checksum,
)
from tests.steering._helpers import save_raw_shard

# ---------------------------------------------------------------------------
# Test tokenizer — deterministic, simple char-level decoding
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Tokenizer where each token decodes to a fixed-length string.

    Token 0 = BOS = "<bos>" (5 chars)
    Token n (n>0) = f"t{n:03d} " (5 chars each)

    This gives deterministic char offsets for testing.
    """

    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        # Each word becomes a token
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
# Helpers
# ---------------------------------------------------------------------------


def _compute_char_level_assignments(tokenizer, text_tokens, seg_char_starts, seg_char_ends):
    """Replicate the char-level path from _build_row_segments.

    Matches production (``_char_level_assignments`` in
    ``scripts/precompute_token_boundaries.py``): tokens that start at or
    beyond ``seg_char_ends[-1]`` are outside the source context window and
    must be truncated before calling ``assign_tokens_to_segments`` — which
    now strictly rejects tokens with zero overlap.
    """
    char_offsets = token_to_char_offsets(tokenizer, text_tokens)
    tok_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends = t.tensor(char_offsets[1:], dtype=t.int64)

    # Truncate tokens to the source segment range (mirrors production).
    seg_range_end = int(seg_char_ends[-1])
    in_range = tok_starts < seg_range_end
    n_in_range = int(in_range.sum().item())
    tok_starts = tok_starts[:n_in_range]
    tok_ends = tok_ends[:n_in_range]

    assignments = assign_tokens_to_segments(
        tok_starts, tok_ends,
        t.tensor(seg_char_starts, dtype=t.int64),
        t.tensor(seg_char_ends, dtype=t.int64),
    )
    assignments = t.cummax(assignments, dim=0).values
    return assignments


def _compute_token_boundaries(tokenizer, text_tokens, seg_char_starts, seg_char_ends):
    """Compute precomputed token boundaries for segments (same as precompute script)."""
    assignments = _compute_char_level_assignments(
        tokenizer, text_tokens, seg_char_starts, seg_char_ends
    )
    n_segs = len(seg_char_starts)
    token_starts = []
    token_ends = []
    for s in range(n_segs):
        mask = assignments == s
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            token_starts.append(int(indices[0].item()))
            token_ends.append(int(indices[-1].item()) + 1)
        else:
            prev_end = token_ends[-1] if token_ends else 0
            token_starts.append(prev_end)
            token_ends.append(prev_end)
    return token_starts, token_ends


def _make_seg_features(
    n_segs, seg_char_starts, seg_char_ends,
    token_starts=None, token_ends=None,
    n_feats_per_seg=3,
):
    """Create SegFeatures list for testing _build_row_segments."""
    seg_features = []
    for s in range(n_segs):
        fids = t.arange(s * n_feats_per_seg, (s + 1) * n_feats_per_seg, dtype=t.int64)
        buckets = t.ones(n_feats_per_seg, dtype=t.int64)
        pcts = t.full((n_feats_per_seg,), 0.5, dtype=t.float32)
        seg_features.append(SegFeatures(
            feature_ids=fids,
            buckets=buckets,
            percentiles=pcts,
            char_start=seg_char_starts[s],
            char_end=seg_char_ends[s],
            token_start=token_starts[s] if token_starts else None,
            token_end=token_ends[s] if token_ends else None,
        ))
    return seg_features


def test_precomputed_first_segment_must_start_at_token_zero() -> None:
    """F7: a nonzero first boundary would silently leave token 0 on segment 0.

    Use a second packed document so the exception identifies both its document
    index and packed-row start position.
    """
    valid = _make_seg_features(
        1,
        [0],
        [5],
        token_starts=[0],
        token_ends=[3],
    )
    corrupt = _make_seg_features(
        1,
        [0],
        [5],
        token_starts=[1],
        token_ends=[3],
    )
    row_docs = [
        (0, valid, [0, 1, 2]),
        (3, corrupt, [0, 3, 4]),
    ]

    with pytest.raises(
        ValueError,
        match=r"document 1.*doc_start=3.*token_start=0.*got 1",
    ):
        _build_row_segments(row_docs, T=6, tokenizer=None)


SEGMENT_SIZE = 16
N_DOCS = 4
N_SEGS_PER_DOC = 3
N_FEATS_PER_SEG = 5


def _make_test_feature_dir(tmp_path, include_token_boundaries=True,
                            wrong_checksum=False, tokenizer_checksum="test_checksum"):
    """Create synthetic parquet + feature shard with optional token boundaries."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    tokenizer = SimpleTokenizer()
    bos_token = tokenizer.get_bos_token_id()

    shard_stem = "shard_00000"
    texts = []
    for doc_idx in range(N_DOCS):
        words = [f"word{w}" for w in range(20 + doc_idx * 5)]
        texts.append(" ".join(words))

    pq_path = data_dir / f"{shard_stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    total_segs = N_DOCS * N_SEGS_PER_DOC
    total_feats = total_segs * N_FEATS_PER_SEG

    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)
    seg_char_starts_t = t.zeros(total_segs, dtype=t.int32)
    seg_char_ends_t = t.zeros(total_segs, dtype=t.int32)
    seg_token_starts_t = t.zeros(total_segs, dtype=t.int32)
    seg_token_ends_t = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0

    for doc_idx in range(N_DOCS):
        text = texts[doc_idx]
        text_tokens = tokenizer.encode(text, prepend=bos_token)

        # Compute char offsets for this doc
        char_offsets = token_to_char_offsets(tokenizer, text_tokens)
        n_tokens = len(text_tokens)

        for seg_idx in range(N_SEGS_PER_DOC):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset

            # Char boundaries based on token positions
            tok_start = seg_idx * SEGMENT_SIZE
            tok_end = min((seg_idx + 1) * SEGMENT_SIZE, n_tokens)
            char_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
            char_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]

            seg_char_starts_t[seg_global] = char_start
            seg_char_ends_t[seg_global] = char_end

            for f in range(N_FEATS_PER_SEG):
                fid = (doc_idx * 10 + seg_idx * 5 + f) % 200
                all_fids[feat_offset + f] = fid
                all_mags[feat_offset + f] = 0.1 * (f + 1)

            feat_offset += N_FEATS_PER_SEG
            seg_global += 1

    offsets[total_segs] = feat_offset

    if include_token_boundaries:
        # Compute token boundaries for each doc
        seg_global = 0
        for doc_idx in range(N_DOCS):
            text = texts[doc_idx]
            doc_seg_char_starts = []
            doc_seg_char_ends = []
            for seg_idx in range(N_SEGS_PER_DOC):
                doc_seg_char_starts.append(int(seg_char_starts_t[seg_global + seg_idx].item()))
                doc_seg_char_ends.append(int(seg_char_ends_t[seg_global + seg_idx].item()))

            text_tokens = tokenizer.encode(text, prepend=bos_token)
            tok_starts, tok_ends = _compute_token_boundaries(
                tokenizer, text_tokens, doc_seg_char_starts, doc_seg_char_ends
            )
            for seg_idx in range(N_SEGS_PER_DOC):
                seg_token_starts_t[seg_global + seg_idx] = tok_starts[seg_idx]
                seg_token_ends_t[seg_global + seg_idx] = tok_ends[seg_idx]

            seg_global += N_SEGS_PER_DOC

    shard_data = {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((N_DOCS,), 40, dtype=t.int32),
        "segment_char_starts": seg_char_starts_t,
        "segment_char_ends": seg_char_ends_t,
        "n_docs": N_DOCS,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }

    save_features(shard_data, feature_dir / f"{shard_stem}_features")

    if include_token_boundaries:
        # Write token boundary overlay shard (not embedded in base shard)
        effective_checksum = "wrong_value" if wrong_checksum else tokenizer_checksum
        tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, effective_checksum)
        write_manifest(tok_dir, effective_checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: seg_token_starts_t,
                TensorKey.SEGMENT_TOKEN_ENDS: seg_token_ends_t,
            },
            overlay_shard_path(tok_dir, shard_stem),
        )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "tokenizer_checksum": tokenizer_checksum,
        "texts": texts,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExactEquivalence:
    """Precomputed token boundaries produce IDENTICAL seg_map as char-level path."""

    def test_single_doc_single_segment(self):
        """Single doc, single segment — both paths produce same seg_map."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = "hello world foo bar baz"
        text_tokens = tok.encode(text, prepend=bos)
        n_tokens = len(text_tokens)  # 6 tokens (BOS + 5 words)

        char_offsets = token_to_char_offsets(tok, text_tokens)
        seg_char_starts = [char_offsets[0]]
        seg_char_ends = [char_offsets[-1]]

        tok_starts, tok_ends = _compute_token_boundaries(
            tok, text_tokens, seg_char_starts, seg_char_ends
        )

        T = n_tokens
        seg_features_precomp = _make_seg_features(
            1, seg_char_starts, seg_char_ends, tok_starts, tok_ends
        )
        seg_features_char = _make_seg_features(
            1, seg_char_starts, seg_char_ends
        )

        row_docs_precomp = [(0, seg_features_precomp, text_tokens)]
        row_docs_char = [(0, seg_features_char, text_tokens)]

        result_precomp = _build_row_segments(row_docs_precomp, T, tok)
        result_char = _build_row_segments(row_docs_char, T, tok)

        # seg_map should be identical (index 3 in new return)
        assert t.equal(result_precomp[3], result_char[3]), (
            f"seg_map mismatch:\n  precomp: {result_precomp[3]}\n  char: {result_char[3]}"
        )
        # per_seg_fids should be identical (index 0, list of tensors)
        assert len(result_precomp[0]) == len(result_char[0])
        for i, (a, b) in enumerate(zip(result_precomp[0], result_char[0], strict=True)):
            assert t.equal(a, b), f"per_seg_fids[{i}] mismatch"

    def test_multi_segment_doc(self):
        """Multi-segment doc — both paths produce identical seg_map."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        # Create a doc with 30 words → 31 tokens (with BOS)
        text = " ".join(f"w{i}" for i in range(30))
        text_tokens = tok.encode(text, prepend=bos)
        n_tokens = len(text_tokens)

        char_offsets = token_to_char_offsets(tok, text_tokens)

        # 3 segments: tokens [0,10), [10,20), [20,31)
        seg_char_starts = [char_offsets[0], char_offsets[10], char_offsets[20]]
        seg_char_ends = [char_offsets[10], char_offsets[20], char_offsets[-1]]

        tok_starts, tok_ends = _compute_token_boundaries(
            tok, text_tokens, seg_char_starts, seg_char_ends
        )

        T = n_tokens
        seg_features_precomp = _make_seg_features(
            3, seg_char_starts, seg_char_ends, tok_starts, tok_ends
        )
        seg_features_char = _make_seg_features(
            3, seg_char_starts, seg_char_ends
        )

        row_docs_precomp = [(0, seg_features_precomp, text_tokens)]
        row_docs_char = [(0, seg_features_char, text_tokens)]

        result_precomp = _build_row_segments(row_docs_precomp, T, tok)
        result_char = _build_row_segments(row_docs_char, T, tok)

        assert t.equal(result_precomp[3], result_char[3]), (
            f"seg_map mismatch:\n  precomp: {result_precomp[3]}\n  char: {result_char[3]}"
        )


class TestBOSConsistency:
    """BOS token is assigned to segment 0 in both paths."""

    def test_bos_in_segment_zero(self):
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = "a b c d e f g h"
        text_tokens = tok.encode(text, prepend=bos)

        char_offsets = token_to_char_offsets(tok, text_tokens)

        # 2 segments
        mid = len(text_tokens) // 2
        seg_char_starts = [char_offsets[0], char_offsets[mid]]
        seg_char_ends = [char_offsets[mid], char_offsets[-1]]

        tok_starts, tok_ends = _compute_token_boundaries(
            tok, text_tokens, seg_char_starts, seg_char_ends
        )

        # BOS should be in segment 0
        assert tok_starts[0] == 0, f"BOS not at segment 0 start: {tok_starts[0]}"

        T = len(text_tokens)
        seg_features = _make_seg_features(
            2, seg_char_starts, seg_char_ends, tok_starts, tok_ends
        )
        row_docs = [(0, seg_features, text_tokens)]
        result = _build_row_segments(row_docs, T, tok)
        seg_map = result[3]
        assert seg_map[0].item() == 0, "BOS position not mapped to segment 0"


class TestCroppedDoc:
    """Precomputed boundaries handle cropped documents correctly."""

    def test_cropped_doc_matches_char_level(self):
        """When a doc is cropped, min(token_end, n_doc_tokens) handles it correctly."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = " ".join(f"w{i}" for i in range(40))
        full_tokens = tok.encode(text, prepend=bos)

        char_offsets = token_to_char_offsets(tok, full_tokens)

        # 3 segments for full doc
        seg_char_starts = [char_offsets[0], char_offsets[15], char_offsets[30]]
        seg_char_ends = [char_offsets[15], char_offsets[30], char_offsets[-1]]

        # Precompute on FULL doc
        tok_starts, tok_ends = _compute_token_boundaries(
            tok, full_tokens, seg_char_starts, seg_char_ends
        )

        # Now crop to only 20 tokens
        cropped_tokens = full_tokens[:20]
        T = 20

        seg_features_precomp = _make_seg_features(
            3, seg_char_starts, seg_char_ends, tok_starts, tok_ends
        )
        seg_features_char = _make_seg_features(
            3, seg_char_starts, seg_char_ends
        )

        row_docs_precomp = [(0, seg_features_precomp, cropped_tokens)]
        row_docs_char = [(0, seg_features_char, cropped_tokens)]

        result_precomp = _build_row_segments(row_docs_precomp, T, tok)
        result_char = _build_row_segments(row_docs_char, T, tok)

        assert t.equal(result_precomp[3], result_char[3]), (
            f"seg_map mismatch on cropped doc:\n"
            f"  precomp: {result_precomp[3]}\n  char: {result_char[3]}"
        )


class TestFallback:
    """Missing/mismatched token boundaries → falls back to char-level path."""

    def test_missing_boundaries_uses_char_level(self):
        """When token_start/token_end are None, the char-level path is used."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        text = "hello world foo bar"
        text_tokens = tok.encode(text, prepend=bos)

        char_offsets = token_to_char_offsets(tok, text_tokens)
        seg_char_starts = [char_offsets[0]]
        seg_char_ends = [char_offsets[-1]]

        # No token boundaries
        seg_features = _make_seg_features(
            1, seg_char_starts, seg_char_ends
        )
        assert seg_features[0].token_start is None
        assert seg_features[0].token_end is None

        T = len(text_tokens)
        row_docs = [(0, seg_features, text_tokens)]
        # Should not crash — falls back to char-level path
        result = _build_row_segments(row_docs, T, tok)
        assert result[3].shape == (T,)


class TestFeatureIndexIntegration:
    """FeatureIndex correctly loads and exposes precomputed token boundaries."""

    def test_token_boundaries_loaded_with_matching_checksum(self, tmp_path):
        checksum = "test_checksum"
        env = _make_test_feature_dir(tmp_path, include_token_boundaries=True,
                                      tokenizer_checksum=checksum)
        idx = FeatureIndex(env["feature_dir"], tokenizer_checksum=checksum)

        # Flag is set after first shard load via lookup
        segs = idx.lookup("shard_00000", 0)
        assert idx.has_precomputed_token_boundaries is True
        assert segs is not None
        for seg in segs:
            assert seg.token_start is not None
            assert seg.token_end is not None

    def test_wrong_requested_token_checksum_fails_loud(self, tmp_path):
        env = _make_test_feature_dir(tmp_path, include_token_boundaries=True,
                                      tokenizer_checksum="correct_checksum")
        with pytest.raises(OverlayIncompleteError, match="token"):
            FeatureIndex(
                env["feature_dir"], tokenizer_checksum="different_checksum"
            )

    def test_token_boundaries_ignored_without_checksum(self, tmp_path):
        env = _make_test_feature_dir(tmp_path, include_token_boundaries=True,
                                      tokenizer_checksum="some_checksum")
        # No tokenizer_checksum provided to FeatureIndex
        idx = FeatureIndex(env["feature_dir"])
        assert idx.has_precomputed_token_boundaries is False

    def test_requested_missing_token_overlay_fails_loud(self, tmp_path):
        env = _make_test_feature_dir(tmp_path, include_token_boundaries=False,
                                      tokenizer_checksum="test")
        with pytest.raises(OverlayIncompleteError, match="token"):
            FeatureIndex(env["feature_dir"], tokenizer_checksum="test")


class TestTokenizerChecksum:
    """compute_tokenizer_checksum is deterministic and sensitive to tokenizer changes."""

    def test_deterministic(self):
        tok = SimpleTokenizer()
        c1 = compute_tokenizer_checksum(tok)
        c2 = compute_tokenizer_checksum(tok)
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex digest

    def test_different_tokenizers_different_checksum(self):
        tok1 = SimpleTokenizer()

        class AltTokenizer:
            def encode(self, text, prepend=None, num_threads=None):
                # Different encoding: reverse word order
                tokens = list(range(len(text.split()), 0, -1))
                if prepend is not None:
                    tokens.insert(0, prepend if isinstance(prepend, int) else 0)
                return tokens

        tok2 = AltTokenizer()
        assert compute_tokenizer_checksum(tok1) != compute_tokenizer_checksum(tok2)


class TestStatsIOTokenizerChecksum:
    """save_tokenizer_checksum / load_tokenizer_checksum round-trip correctly."""

    def test_round_trip(self, tmp_path):
        from safetensors.torch import save_file
        # Create a minimal stats file
        stats_path = tmp_path / "feature_statistics.safetensors"
        save_file({"dummy": t.tensor([1.0])}, str(stats_path), metadata={"test": "1"})

        assert load_tokenizer_checksum(stats_path) is None

        save_tokenizer_checksum(stats_path, "abc123")
        assert load_tokenizer_checksum(stats_path) == "abc123"

        # Update overwrites
        save_tokenizer_checksum(stats_path, "def456")
        assert load_tokenizer_checksum(stats_path) == "def456"

    def test_already_current_checksum_is_byte_preserving(self, tmp_path):
        """Repeated checksum persistence cannot churn the statistics identity."""
        from safetensors.torch import save_file

        stats_path = tmp_path / "feature_statistics.safetensors"
        save_file({"dummy": t.tensor([1.0])}, str(stats_path))
        save_tokenizer_checksum(stats_path, "stable")
        before = stats_path.read_bytes()

        save_tokenizer_checksum(stats_path, "stable")

        assert stats_path.read_bytes() == before


class TestRoundTripPrecomputation:
    """Precomputed boundaries stored in shard match on-the-fly computation."""

    def test_shard_round_trip(self, tmp_path):
        """Load precomputed boundaries from shard and verify they match on-the-fly."""
        checksum = "test_checksum"
        env = _make_test_feature_dir(tmp_path, include_token_boundaries=True,
                                      tokenizer_checksum=checksum)
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        idx = FeatureIndex(env["feature_dir"], tokenizer_checksum=checksum)

        for doc_idx in range(N_DOCS):
            segs = idx.lookup("shard_00000", doc_idx)
            assert segs is not None

            text = env["texts"][doc_idx]
            text_tokens = tok.encode(text, prepend=bos)

            seg_char_starts = [s.char_start for s in segs]
            seg_char_ends = [s.char_end for s in segs]

            expected_tok_starts, expected_tok_ends = _compute_token_boundaries(
                tok, text_tokens, seg_char_starts, seg_char_ends
            )

            for s_idx, seg in enumerate(segs):
                assert seg.token_start == expected_tok_starts[s_idx], (
                    f"doc {doc_idx} seg {s_idx}: token_start mismatch "
                    f"{seg.token_start} != {expected_tok_starts[s_idx]}"
                )
                assert seg.token_end == expected_tok_ends[s_idx], (
                    f"doc {doc_idx} seg {s_idx}: token_end mismatch "
                    f"{seg.token_end} != {expected_tok_ends[s_idx]}"
                )


# ---------------------------------------------------------------------------
# Helpers for parallel precomputation tests
# ---------------------------------------------------------------------------


def _make_parallel_test_shard(tmp_path, n_docs=8, n_segs_per_doc=4, suffix=""):
    """Create a synthetic parquet + feature shard for parallel testing.

    Returns (shard_path, parquet_path, texts).
    """
    data_dir = tmp_path / f"data{suffix}"
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = tmp_path / f"features{suffix}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    tok = SimpleTokenizer()
    bos = tok.get_bos_token_id()

    shard_stem = "shard_00000"
    texts = []
    for doc_idx in range(n_docs):
        # Ensure each doc fits ENTIRELY within n_segs_per_doc segments.
        # SimpleTokenizer: each word = 1 token, plus BOS = n_words + 1 tokens.
        # Total capacity = n_segs_per_doc * SEGMENT_SIZE tokens; leave a small
        # variable tail that still fits. Wave 3D hardened
        # ``assign_tokens_to_segments`` to reject tokens that fall outside the
        # segment char range, so every token must be covered by some segment.
        capacity = n_segs_per_doc * SEGMENT_SIZE
        n_words = capacity - 1 - (doc_idx % 4)  # -1 for BOS, small doc variance
        words = [f"word{w}" for w in range(n_words)]
        texts.append(" ".join(words))

    pq_path = data_dir / f"{shard_stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    total_segs = n_docs * n_segs_per_doc
    n_feats_per_seg = 3
    total_feats = total_segs * n_feats_per_seg

    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)
    seg_char_starts_t = t.zeros(total_segs, dtype=t.int32)
    seg_char_ends_t = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0

    for doc_idx in range(n_docs):
        text = texts[doc_idx]
        text_tokens = tok.encode(text, prepend=bos)
        char_offsets = token_to_char_offsets(tok, text_tokens)

        for seg_idx in range(n_segs_per_doc):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset

            tok_start = seg_idx * SEGMENT_SIZE
            tok_end = min((seg_idx + 1) * SEGMENT_SIZE, len(text_tokens))
            char_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
            char_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]

            seg_char_starts_t[seg_global] = char_start
            seg_char_ends_t[seg_global] = char_end

            for f in range(n_feats_per_seg):
                all_fids[feat_offset + f] = (doc_idx * 10 + seg_idx * 5 + f) % 200
                all_mags[feat_offset + f] = 0.1 * (f + 1)
            feat_offset += n_feats_per_seg
            seg_global += 1

    offsets[total_segs] = feat_offset

    shard_data = {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((n_docs,), 80, dtype=t.int32),
        "segment_char_starts": seg_char_starts_t,
        "segment_char_ends": seg_char_ends_t,
        "n_docs": n_docs,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }

    actual_path = save_features(shard_data, feature_dir / f"{shard_stem}_features")
    return actual_path, pq_path, texts


# ---------------------------------------------------------------------------
# Parallel precomputation tests
# ---------------------------------------------------------------------------


@contextmanager
def _worker_pool(
    tok,
    source_bos_char_len,
    source_segment_size,
    n_workers=2,
    max_nanochat_tokens_per_segment=None,
    min_nanochat_tokens_per_segment=None,
):
    """Build a run-level spawn pool so these tests can exercise precompute_shard's pooled path.

    Mirrors run()'s pool setup (worker globals fixed at construction via _init_worker, including
    the validation bounds). precompute_shard runs sequentially when ``worker_pool`` is None and in
    parallel when handed this pool (its ``max_workers`` sets the parallelism); the old per-shard
    self-spawn fallback and the ``n_workers`` parameter were removed in #529. The token bounds
    default to None (multiplier-based) — pass them when a pooled test needs non-default bounds, so
    the pooled path validates with the same bounds as the sequential comparison.
    """
    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(
            tok,
            source_bos_char_len,
            source_segment_size,
            max_nanochat_tokens_per_segment,
            min_nanochat_tokens_per_segment,
        ),
    )
    try:
        yield pool
    finally:
        pool.shutdown()


class TestParallelPrecomputation:
    """Sequential vs parallel precompute_shard produce identical results."""

    def test_sequential_vs_parallel_exact_match(self, tmp_path):
        """Sequential (no pool) vs pooled precompute_shard produce identical results."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        checksum = "test_checksum" + "0" * 51

        # Create two identical shards (one for each run)
        shard_seq, pq_seq, _ = _make_parallel_test_shard(tmp_path, suffix="_seq")
        shard_par, pq_par, _ = _make_parallel_test_shard(tmp_path, suffix="_par")

        # Create overlay dirs
        out_seq = overlay_dir(shard_seq.parent, OverlayKind.SEG_BOUNDARY, checksum)
        out_par = overlay_dir(shard_par.parent, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(out_seq, checksum, OverlayKind.SEG_BOUNDARY)
        write_manifest(out_par, checksum, OverlayKind.SEG_BOUNDARY)

        # Sequential (no pool)
        precompute_shard(shard_seq, pq_seq, tok, bos, out_seq, "shard_00000", source_bos_char_len=5, source_segment_size=SEGMENT_SIZE)
        # Parallel (pooled)
        with _worker_pool(tok, 5, SEGMENT_SIZE) as pool:
            precompute_shard(shard_par, pq_par, tok, bos, out_par, "shard_00000", source_bos_char_len=5, source_segment_size=SEGMENT_SIZE, worker_pool=pool)

        data_seq = load_overlay_shard(overlay_shard_path(out_seq, "shard_00000"))
        data_par = load_overlay_shard(overlay_shard_path(out_par, "shard_00000"))

        assert t.equal(
            data_seq[TensorKey.SEGMENT_TOKEN_STARTS],
            data_par[TensorKey.SEGMENT_TOKEN_STARTS],
        ), (
            f"token_starts mismatch:\n"
            f"  seq: {data_seq[TensorKey.SEGMENT_TOKEN_STARTS]}\n"
            f"  par: {data_par[TensorKey.SEGMENT_TOKEN_STARTS]}"
        )
        assert t.equal(
            data_seq[TensorKey.SEGMENT_TOKEN_ENDS],
            data_par[TensorKey.SEGMENT_TOKEN_ENDS],
        ), (
            f"token_ends mismatch:\n"
            f"  seq: {data_seq[TensorKey.SEGMENT_TOKEN_ENDS]}\n"
            f"  par: {data_par[TensorKey.SEGMENT_TOKEN_ENDS]}"
        )

    def test_parallel_round_trip_through_feature_index(self, tmp_path):
        """Parallel-precomputed shard loaded via FeatureIndex matches on-the-fly."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        checksum = "test_checksum" + "0" * 51

        shard_path, pq_path, texts = _make_parallel_test_shard(tmp_path, n_docs=6)
        out_dir = overlay_dir(shard_path.parent, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(out_dir, checksum, OverlayKind.SEG_BOUNDARY)
        with _worker_pool(tok, 5, SEGMENT_SIZE) as pool:
            precompute_shard(shard_path, pq_path, tok, bos, out_dir, "shard_00000", source_bos_char_len=5, source_segment_size=SEGMENT_SIZE, worker_pool=pool)

        feature_dir = str(shard_path.parent)
        idx = FeatureIndex(feature_dir, tokenizer_checksum=checksum)

        for doc_idx in range(6):
            segs = idx.lookup("shard_00000", doc_idx)
            assert segs is not None

            text = texts[doc_idx]
            text_tokens = tok.encode(text, prepend=bos)
            seg_char_starts = [s.char_start for s in segs]
            seg_char_ends = [s.char_end for s in segs]

            expected_starts, expected_ends = _compute_token_boundaries(
                tok, text_tokens, seg_char_starts, seg_char_ends
            )

            for s_idx, seg in enumerate(segs):
                assert seg.token_start == expected_starts[s_idx], (
                    f"doc {doc_idx} seg {s_idx}: token_start {seg.token_start} != {expected_starts[s_idx]}"
                )
                assert seg.token_end == expected_ends[s_idx], (
                    f"doc {doc_idx} seg {s_idx}: token_end {seg.token_end} != {expected_ends[s_idx]}"
                )

    def test_parallel_dataloader_equivalence(self, tmp_path):
        """Parallel-precomputed boundaries produce identical seg_map as char-level path."""
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        checksum = "test_checksum" + "0" * 51

        shard_path, pq_path, texts = _make_parallel_test_shard(tmp_path, n_docs=4, n_segs_per_doc=3)
        out_dir = overlay_dir(shard_path.parent, OverlayKind.SEG_BOUNDARY, checksum)
        write_manifest(out_dir, checksum, OverlayKind.SEG_BOUNDARY)
        with _worker_pool(tok, 5, SEGMENT_SIZE) as pool:
            precompute_shard(shard_path, pq_path, tok, bos, out_dir, "shard_00000", source_bos_char_len=5, source_segment_size=SEGMENT_SIZE, worker_pool=pool)

        feature_dir = str(shard_path.parent)
        idx = FeatureIndex(feature_dir, tokenizer_checksum=checksum)

        for doc_idx in range(4):
            segs = idx.lookup("shard_00000", doc_idx)
            assert segs is not None

            text = texts[doc_idx]
            text_tokens = tok.encode(text, prepend=bos)
            T = len(text_tokens)

            seg_char_starts = [s.char_start for s in segs]
            seg_char_ends = [s.char_end for s in segs]

            # Build SegFeatures WITH precomputed boundaries
            seg_features_precomp = _make_seg_features(
                len(segs), seg_char_starts, seg_char_ends,
                [s.token_start for s in segs], [s.token_end for s in segs],
            )
            # Build SegFeatures WITHOUT (char-level fallback)
            seg_features_char = _make_seg_features(
                len(segs), seg_char_starts, seg_char_ends,
            )

            row_docs_precomp = [(doc_idx, seg_features_precomp, text_tokens)]
            row_docs_char = [(doc_idx, seg_features_char, text_tokens)]

            result_precomp = _build_row_segments(row_docs_precomp, T, tok)
            result_char = _build_row_segments(row_docs_char, T, tok)

            assert t.equal(result_precomp[3], result_char[3]), (
                f"doc {doc_idx} seg_map mismatch:\n"
                f"  precomp: {result_precomp[3]}\n  char: {result_char[3]}"
            )


def _corrupt_empty_segment_shard_data() -> dict:
    """Base-shard data dict with an empty middle segment (issue #522).

    ``offsets=[0,2,2,4]`` → 3 segments (so the legitimate ``n_segments==0``
    whole-shard skip does NOT apply), with segment 1 empty. CSR + per-segment
    boundary tensors are internally consistent so ``load_features`` reads it
    back cleanly; the producer guard fires right after the load.
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


class TestPrecomputeTokenBoundariesEmptySegmentGuard:
    """Issue #522 site-f: precompute_token_boundaries reads base shards directly
    via load_features (not via FeatureIndex), so it needs its own empty-segment
    guard. It must fire AFTER the legitimate n_segments==0 whole-shard skip but
    BEFORE the heavy parquet+tokenizer work."""

    def test_precompute_shard_raises_on_empty_segment(self, tmp_path):
        shard_path = Path(
            save_raw_shard(tmp_path, "shard_00000", _corrupt_empty_segment_shard_data())
        )
        # A real (but otherwise unused) parquet — the guard fires before it is
        # opened, so its contents are irrelevant.
        pq_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": ["hello world"]}), str(pq_path))

        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(EmptyExtractionSegmentError, match=r"segment 1\b"):
            precompute_shard(
                shard_path,
                pq_path,
                tok,
                bos,
                out_dir,
                "shard_00000",
                source_bos_char_len=5,
                source_segment_size=SEGMENT_SIZE,
            )


class TestPrecomputeTokenBoundariesEmptyShardSkip:
    """Issue #522 exemption: a WHOLE-shard-empty base shard (n_segments==0,
    offsets==[0]) is legitimate and must be SKIPPED, not rejected.

    The n_segments==0 early-return precedes the empty-segment guard, so this
    confirms the guard's ordering: ``empty_segment_indices([0]) == []`` would
    also pass the guard, but the early-return short-circuits first and avoids
    any parquet/tokenizer work."""

    def test_precompute_shard_skips_empty_shard(self, tmp_path):
        # Whole-shard-empty base shard is acceptable to save_features
        # (the empty-base-SEGMENT guard only fires for n_segments>0 shards
        # that carry a zero-feature segment).
        empty_shard = {
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
            "segment_size": SEGMENT_SIZE,
        }
        shard_path = save_features(empty_shard, tmp_path / "shard_00000_features")

        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Must NOT raise (the empty-segment guard would crash on offsets==[0]
        # only if reached); returns False == "empty shard, skipping".
        wrote = precompute_shard(
            Path(shard_path),
            None,
            tok,
            bos,
            out_dir,
            "shard_00000",
            source_bos_char_len=5,
            source_segment_size=SEGMENT_SIZE,
        )
        assert wrote is False
        # No overlay file should have been written for a skipped shard.
        assert not (out_dir / "shard_00000.safetensors.zst").exists()


def test_precompute_shard_rejects_turn_delimited_metadata(tmp_path):
    with pytest.raises(
        ValueError,
        match="token-boundary precompute supports fixed_tokens shards only",
    ):
        precompute_shard(
            tmp_path / "shard_00000.safetensors.zst",
            None,
            SimpleTokenizer(),
            0,
            tmp_path / "out",
            "shard_00000",
            source_bos_char_len=5,
            source_segment_size=SEGMENT_SIZE,
            preloaded_data={"segment_mode": SegmentMode.turn_delimited_fixed_tokens.value},
        )
