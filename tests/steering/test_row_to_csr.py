"""Tests for _row_to_csr direct-cat CSR conversion.

Verifies that the direct torch.cat path produces identical output to the
old pad-then-unpad path, and exercises all branches of the CSR conversion.
"""

from __future__ import annotations

import random

import pytest
import torch as t

from errors import PackedRowInvariantError
from scripts.preprocess.precompute_packed_rows import (
    _row_to_csr,
    _seg_features_to_packed,
)
from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
    _pad_segments,
)
from steering.conditional.feature_index import SegmentFeatures as IndexedSegmentFeatures
from steering.conditional.percentile_table import normalized_percentile_to_uint8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg_features(
    token_start: int,
    token_end: int,
    n_features: int = 5,
    char_start: int = 0,
    char_end: int = 100,
) -> SegFeatures:
    return SegFeatures(
        feature_ids=t.arange(n_features, dtype=t.int64),
        buckets=t.ones(n_features, dtype=t.int64),
        percentiles=t.rand(n_features, dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
    )


def _make_row_doc_info(
    doc_specs: list[tuple[int, int, int]],
    T: int,
) -> tuple[t.Tensor, list]:
    """Create row_tokens and row_doc_info from doc specs.

    Each spec is (doc_start, doc_len, n_segments).
    """
    row_tokens = t.randint(0, 1000, (T + 1,), dtype=t.long)
    row_doc_info = []

    for doc_start, doc_len, n_segments in doc_specs:
        seg_size = max(1, doc_len // n_segments) if n_segments > 0 else doc_len
        seg_features = []
        for s in range(n_segments):
            tok_lo = s * seg_size
            tok_hi = doc_len if s == n_segments - 1 else (s + 1) * seg_size
            seg_features.append(_make_seg_features(
                token_start=tok_lo,
                token_end=tok_hi,
                n_features=random.randint(1, 10),
                char_start=tok_lo * 5,
                char_end=tok_hi * 5,
            ))

        doc_tokens = list(range(doc_start, doc_start + doc_len))
        if n_segments == 0:
            row_doc_info.append((doc_start, None, doc_tokens))
        else:
            row_doc_info.append((doc_start, seg_features, doc_tokens))

    return row_tokens, row_doc_info


class _MockTokenizer:
    """Minimal tokenizer mock for _build_row_segments."""

    def get_bos_token_id(self):
        return 1

    def decode(self, ids):
        return "x" * len(ids)


TOKENIZER = _MockTokenizer()


def _old_pad_path(row_doc_info, T):
    """Reference: the old pad-then-unpad path for comparison."""
    per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map, seg_doc_id, _, _ = \
        _build_row_segments(row_doc_info, T, TOKENIZER)

    n_segs = len(per_seg_fids)

    # Pad to matrix
    fids, mags, pcts, mask, seg_map_out = _pad_segments(
        per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map,
    )

    # Extract with mask (old way)
    all_active_fids = fids[mask]
    all_active_pcts = (pcts[mask] * 255.0).round().clamp(0, 255).to(t.uint8)
    feat_counts = mask.sum(dim=1).to(t.int32)

    return all_active_fids, all_active_pcts, feat_counts, seg_map, seg_doc_id, n_segs


def _new_cat_path(row_doc_info, T):
    """The new direct-cat path."""
    per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map, seg_doc_id, _, _ = \
        _build_row_segments(row_doc_info, T, TOKENIZER)

    n_segs = len(per_seg_fids)

    all_active_fids = t.cat(per_seg_fids) if per_seg_fids else t.tensor([], dtype=t.int64)
    raw_pcts = t.cat(per_seg_pcts) if per_seg_pcts else t.tensor([], dtype=t.float32)
    all_active_pcts = (raw_pcts * 255.0).round().clamp(0, 255).to(t.uint8)
    feat_counts = t.tensor([len(f) for f in per_seg_fids], dtype=t.int32)

    return all_active_fids, all_active_pcts, feat_counts, seg_map, seg_doc_id, n_segs


def _encode_row_doc_info(row_doc_info):
    """Convert the live float fixture to the packed producer's uint8 carrier."""
    encoded_info = []
    for start, segments, tokens in row_doc_info:
        encoded_segments = None
        if segments is not None:
            encoded_segments = []
            for segment_idx, segment in enumerate(segments):
                indexed = IndexedSegmentFeatures(
                    segment_idx=segment_idx,
                    position=segment_idx,
                    feature_ids=segment.feature_ids,
                    feature_mags=t.ones(len(segment.feature_ids)),
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                    token_start=segment.token_start,
                    token_end=segment.token_end,
                    feature_percentiles=normalized_percentile_to_uint8(
                        segment.percentiles
                    ),
                )
                encoded_segments.append(_seg_features_to_packed(indexed))
        encoded_info.append((start, encoded_segments, tokens))
    return encoded_info


def _packed_csr(row_tokens, row_doc_info, T):
    return _row_to_csr(row_tokens, _encode_row_doc_info(row_doc_info), T, TOKENIZER)


# ---------------------------------------------------------------------------
# Equivalence tests: direct cat matches pad-then-unpad
# ---------------------------------------------------------------------------

class TestCatVsPadEquivalence:
    """Verify direct-cat produces byte-identical results to pad-then-unpad."""

    def _assert_equivalent(self, doc_specs, T):
        _, row_doc_info = _make_row_doc_info(doc_specs, T)

        old_fids, old_pcts, old_counts, old_seg_map, old_doc_id, old_n = \
            _old_pad_path(row_doc_info, T)
        new_fids, new_pcts, new_counts, new_seg_map, new_doc_id, new_n = \
            _new_cat_path(row_doc_info, T)

        assert old_n == new_n, f"n_segs: {old_n} vs {new_n}"
        assert t.equal(old_fids, new_fids), "feature_ids mismatch"
        assert t.equal(old_pcts, new_pcts), "feature_pcts mismatch"
        assert t.equal(old_counts, new_counts), f"feat_counts: {old_counts} vs {new_counts}"
        assert t.equal(old_seg_map, new_seg_map), "seg_map mismatch"
        assert t.equal(old_doc_id, new_doc_id), "seg_doc_id mismatch"

    def test_single_doc_single_segment(self):
        self._assert_equivalent([(0, 100, 1)], T=100)

    def test_single_doc_multiple_segments(self):
        self._assert_equivalent([(0, 200, 4)], T=200)

    def test_two_docs(self):
        self._assert_equivalent([(0, 100, 2), (100, 100, 2)], T=200)

    def test_three_docs_varying_segments(self):
        self._assert_equivalent(
            [(0, 150, 3), (150, 80, 2), (230, 70, 1)], T=300,
        )

    def test_doc_with_no_features(self):
        self._assert_equivalent([(0, 100, 0), (100, 100, 2)], T=200)

    def test_all_docs_no_features(self):
        self._assert_equivalent([(0, 100, 0), (100, 100, 0)], T=200)

    @pytest.mark.parametrize("seed", range(20))
    def test_randomized(self, seed):
        rng = random.Random(seed)
        n_docs = rng.randint(1, 5)
        T = rng.randint(100, 500)
        pos = 0
        specs = []
        for _ in range(n_docs):
            doc_len = rng.randint(10, T // n_docs)
            n_segs = rng.randint(0, min(4, doc_len // 5 + 1))
            specs.append((pos, doc_len, n_segs))
            pos += doc_len
            if pos >= T:
                break
        if specs:
            self._assert_equivalent(specs, T)


# ---------------------------------------------------------------------------
# _row_to_csr unit tests
# ---------------------------------------------------------------------------

class TestRowToCSR:
    """Test _row_to_csr directly."""

    def test_basic_output_shapes(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 100, 2), (100, 100, 2)], T=200,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 200)

        assert csr.tokens.shape == (201,)
        assert csr.tokens.dtype == t.uint16
        assert csr.seg_doc_id.dtype == t.uint8
        assert csr.feature_pcts.dtype == t.uint8
        # O3 (issue #531): _RowCSR holds feature_ids at the final uint16 dtype (was int64),
        # cutting accumulator memory ~4×. flush_shard's cat(...).to(uint16) keeps the assembled
        # shard byte-identical (ids < 65536 ⇒ lossless), proven by the packed-rows golden.
        assert csr.feature_ids.dtype == t.uint16
        assert csr.feat_counts.dtype == t.int32
        assert csr.seg_token_start.dtype == t.int32
        assert csr.seg_token_end.dtype == t.int32

    def test_segment_boundaries_contiguous(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 200, 4)], T=200,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 200)

        # Segments should tile [0, T) without gaps
        for i in range(len(csr.seg_token_start) - 1):
            assert csr.seg_token_end[i] == csr.seg_token_start[i + 1], (
                f"Gap between segment {i} end={csr.seg_token_end[i]} "
                f"and segment {i+1} start={csr.seg_token_start[i+1]}"
            )
        assert csr.seg_token_start[0] == 0
        assert csr.seg_token_end[-1] == 200

    def test_feat_counts_sum_matches_fids_length(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 150, 3), (150, 50, 1)], T=200,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 200)

        assert csr.feat_counts.sum().item() == len(csr.feature_ids)
        assert csr.feat_counts.sum().item() == len(csr.feature_pcts)

    def test_no_features_produces_zero_counts(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 100, 0)], T=100,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 100)

        assert len(csr.feature_ids) == 0
        assert len(csr.feature_pcts) == 0
        assert csr.feat_counts.sum().item() == 0

    def test_pcts_in_valid_range(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 200, 4)], T=200,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 200)

        assert (csr.feature_pcts >= 0).all()
        assert (csr.feature_pcts <= 255).all()

    @pytest.mark.parametrize(
        "specs,T",
        [
            ([(0, 200, 4)], 200),
            ([(0, 90, 2), (90, 110, 3)], 200),
            ([(0, 100, 0)], 100),
            ([(0, 63, 3)], 100),
        ],
    )
    def test_encoded_path_matches_legacy_tensor_reference(self, specs, T):
        row_tokens, float_info = _make_row_doc_info(specs, T=T)
        ref_fids, ref_pcts, ref_counts, ref_seg_map, ref_doc_id, ref_n = _old_pad_path(
            float_info, T
        )
        actual = _packed_csr(row_tokens, float_info, T)

        ref_starts = t.tensor(
            [int(t.nonzero(ref_seg_map == seg_id)[0]) for seg_id in range(ref_n)],
            dtype=t.int32,
        )
        ref_ends = t.cat([ref_starts[1:], t.tensor([T], dtype=t.int32)])
        assert t.equal(actual.tokens, row_tokens.to(t.uint16))
        assert t.equal(actual.seg_token_start, ref_starts)
        assert t.equal(actual.seg_token_end, ref_ends)
        assert t.equal(actual.seg_doc_id, ref_doc_id.to(t.uint8))
        assert t.equal(actual.feature_ids, ref_fids.to(t.uint16))
        assert t.equal(actual.feature_pcts, ref_pcts)
        assert t.equal(actual.feat_counts, ref_counts)
        assert actual.n_segs == ref_n

    def test_rejects_float_percentiles(self):
        row_tokens, row_doc_info = _make_row_doc_info([(0, 100, 2)], T=100)

        with pytest.raises(PackedRowInvariantError, match="non-uint8"):
            _row_to_csr(row_tokens, row_doc_info, 100, TOKENIZER)

    def test_doc_id_correct(self):
        row_tokens, row_doc_info = _make_row_doc_info(
            [(0, 100, 2), (100, 100, 3)], T=200,
        )
        csr = _packed_csr(row_tokens, row_doc_info, 200)

        # First 2 segments → doc 0, next 3 → doc 1
        assert csr.seg_doc_id[0] == 0
        assert csr.seg_doc_id[1] == 0
        assert csr.seg_doc_id[2] == 1
        assert csr.seg_doc_id[3] == 1
        assert csr.seg_doc_id[4] == 1

    @pytest.mark.parametrize("seed", range(10))
    def test_randomized_invariants(self, seed):
        rng = random.Random(seed + 500)
        T = rng.randint(100, 500)
        n_docs = rng.randint(1, 4)
        pos = 0
        specs = []
        for i in range(n_docs):
            remaining = T - pos
            if remaining <= 0:
                break
            if i == n_docs - 1:
                doc_len = remaining  # last doc fills to T
            else:
                doc_len = rng.randint(10, max(10, remaining - 10 * (n_docs - i - 1)))
            n_segs = rng.randint(1, min(4, doc_len // 5 + 1))
            specs.append((pos, doc_len, n_segs))
            pos += doc_len

        row_tokens, row_doc_info = _make_row_doc_info(specs, T)
        csr = _packed_csr(row_tokens, row_doc_info, T)

        # Invariants
        assert csr.tokens.shape == (T + 1,)
        assert csr.feat_counts.sum().item() == len(csr.feature_ids)
        assert len(csr.seg_token_start) == len(csr.seg_token_end)
        assert len(csr.seg_token_start) == len(csr.seg_doc_id)
        assert len(csr.seg_token_start) == len(csr.feat_counts)
        assert csr.seg_token_start[0] == 0


def test_uint16_feature_ids_lossless_at_65535_boundary():
    """O3 (issue #531 B-A2/F-A4): _RowCSR stores feature_ids as uint16. SAE widths up to 65k put
    ids across the full [0, 65535] uint16 range, so the int64->uint16 round-trip MUST be lossless
    at the boundary — the premise the real-corpus f65k byte-identity relies on. Every other fixture
    caps ids well under 256, so this is the only test that pins the boundary directly. A regression
    to a narrower dtype (uint8) or a truncating cast would corrupt the largest ids (e.g. 65535->255)
    while passing every small-id test."""
    T = 64
    boundary_ids = [0, 1, 65000, 65534, 65535]  # spans the uint16 range, incl. its max value
    seg = IndexedSegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor(boundary_ids, dtype=t.int64),
        feature_mags=t.ones(len(boundary_ids)),
        char_start=0,
        char_end=100,
        token_start=0,
        token_end=T,
        feature_percentiles=t.arange(len(boundary_ids), dtype=t.uint8),
    )
    row_tokens = t.randint(0, 1000, (T + 1,), dtype=t.long)
    row_doc_info = [(0, [_seg_features_to_packed(seg)], list(range(T)))]

    csr = _row_to_csr(row_tokens, row_doc_info, T, TOKENIZER)

    assert csr.feature_ids.dtype == t.uint16
    # uint16 holds 0..65535 losslessly: every id (incl. 65535) survives, in order.
    assert csr.feature_ids.tolist() == boundary_ids
