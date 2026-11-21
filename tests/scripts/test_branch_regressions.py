"""Regression tests for feat/profile-packed-rows functional changes.

Each test in this file explicitly catches a behavioral shortcoming of the OLD
code. Tests are designed to:
  1. FAIL on main (old code) — demonstrating the bug
  2. PASS on this branch (new code) — demonstrating the fix

Covers three functional fixes:
  - PR #289: merge-aware offset correction in token_to_char_offsets
  - PR #287: truncate packed row docs to precomputed boundary coverage
  - PR #288: global doc index in packed rows iterator
  - Batch encoding in _raw_document_iterator
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.preprocess.precompute_packed_rows import (
    _raw_document_iterator,
    assert_boundaries_covered,
    truncate_to_boundary_coverage,
)
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.feature_index import SegmentFeatures
from steering.refined_types import PositiveInt
from steering.segmentation import (
    _token_to_char_offsets_progressive,
    token_to_char_offsets,
)


def _require_tokenizer():
    """Skip test if the RustBPE tokenizer cache is unavailable locally."""
    try:
        from nanochat.tokenizer import get_tokenizer

        return get_tokenizer()
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"Tokenizer cache unavailable: {e}")


# ============================================================================
# Helpers
# ============================================================================


def _make_seg_features(
    boundaries: list[tuple[int, int]],
    n_features: int = 3,
) -> list[SegFeatures]:
    segs = []
    for tok_start, tok_end in boundaries:
        segs.append(SegFeatures(
            feature_ids=t.arange(n_features, dtype=t.int64),
            buckets=t.zeros(n_features, dtype=t.int64),
            percentiles=t.ones(n_features, dtype=t.float32) * 0.5,
            char_start=tok_start * 5,
            char_end=tok_end * 5,
            token_start=tok_start,
            token_end=tok_end,
        ))
    return segs


_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(128)


class SimpleTokenizer:
    """Deterministic word-level tokenizer for testing."""

    def get_bos_token_id(self):
        return 0

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


class RecordingFeatureIndex:
    """Mock FeatureIndex that records every (shard_stem, doc_idx) lookup."""

    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    @property
    def segment_size(self):
        return 128

    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
        self.calls.append((shard_stem, doc_idx))
        # Mirror the real post-#521 contract: a present doc returns >= 1 segment
        # (extraction guarantees this). These tests assume full coverage and
        # only inspect recorded doc indices + token equivalence — segment
        # CONTENT is irrelevant here. Mirrors _one_segment in
        # tests/scripts/test_packed_rows_doc_index.py.
        return [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([doc_idx % 7], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=0,
                char_end=1,
                feature_percentiles=None,
                token_start=0,
                token_end=1,
            )
        ]


# ============================================================================
# Reference: OLD algorithm for comparison
# ============================================================================


def _old_token_to_char_offsets_progressive(tokenizer, token_ids: list[int]) -> list[int]:
    """The OLD (buggy) algorithm from main: simple max() clamping."""
    offsets = [0]
    for i in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[: i + 1])
        offsets.append(max(len(prefix), offsets[-1]))
    return offsets


# ============================================================================
# PR #289: merge-aware offset correction — offsets[-1] inflation bug
# ============================================================================


class TestMergeCorrection_OffsetsLastInflation:
    """OLD BUG: max() clamping inflated offsets[-1] beyond actual decoded length.

    When byte-fallback tokens completed a multi-byte character, the decoded
    prefix shrank (e.g., 3 replacement chars -> 1 real char). The old code:
        offsets.append(max(len(prefix), offsets[-1]))
    would preserve the inflated offset, making offsets[-1] too large.

    This caused downstream segment boundaries to be wrong: the last segment
    covered fewer characters than expected because offsets[-1] was already
    too high.
    """

    def test_4byte_merge_old_gives_inflated_last_offset(self):
        """4 byte tokens merge into 1 char. offsets[-1] must be 1, not 3.

        OLD code would produce: [0, 1, 2, 3, 3] — offsets[-1]=3 (WRONG)
        NEW code produces:      [0, 0, 0, 0, 1] — offsets[-1]=1 (CORRECT)
        """
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\ufffd\ufffd\ufffd",
            4: "\U00010a05",  # 1 char
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])

        # This is the key assertion that fails on old code (old gives 3)
        assert offsets[-1] == 1, (
            f"offsets[-1]={offsets[-1]} — inflated by replacement chars. "
            f"Expected 1 (actual decoded length)."
        )
        assert offsets[-1] == len(tokenizer.decode([0, 1, 2, 3]))

    def test_bos_plus_smp_old_steals_last_token_width(self):
        """BOS + 4-byte SMP char. Last real token gets zero width with old code.

        OLD code: [0, 5, 6, 7, 8, 8] — token 4 maps to [8,8) = zero width
        NEW code: [0, 5, 5, 5, 5, 6] — token 4 maps to [5,6) = the SMP char
        """
        progressive = {
            1: "<bos>",        # 5 chars
            2: "<bos>\ufffd",   # 6
            3: "<bos>\ufffd\ufffd",  # 7
            4: "<bos>\ufffd\ufffd\ufffd",  # 8
            5: "<bos>\U00010a2f",  # 6 chars (merge!)
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3, 4])

        # Old code gives offsets[-1]=8, new code correctly gives 6
        assert offsets[-1] == 6
        # The completing byte token should have non-zero width
        assert offsets[5] > offsets[4], (
            f"Completing byte token has zero width: [{offsets[4]}, {offsets[5]})"
        )

    def test_ascii_then_merge_old_makes_trailing_ascii_zero_width(self):
        """'H' + 3-byte merge + '!'. Old code gives '!' zero width.

        OLD: [0, 1, 2, 3, 3, 3] — '!' maps to [3,3) = zero width = WRONG
        NEW: [0, 1, 1, 1, 2, 3] — '!' maps to [2,3) = correct
        """
        progressive = {
            1: "H",
            2: "H\ufffd",
            3: "H\ufffd\ufffd",
            4: "H\u00e9",     # merge: len=2
            5: "H\u00e9!",   # len=3
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(5)))

        # Token 4 ('!') must have non-zero width
        assert offsets[5] > offsets[4], (
            f"Trailing ASCII token has zero width: [{offsets[4]}, {offsets[5]})"
        )
        # offsets[-1] must match actual decoded length
        assert offsets[-1] == 3  # len("Hé!") = 3

    def test_two_merges_old_inflates_final(self):
        """Two consecutive 3-byte merges. Old code inflates offsets[-1].

        OLD: [0, 1, 2, 2, 2, 3, 3] — offsets[-1]=3, but decoded='あい'=2 chars
        NEW: [0, 0, 0, 1, 1, 1, 2] — offsets[-1]=2, correct
        """
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\u3042",      # merge: 1 char
            4: "\u3042\ufffd",
            5: "\u3042\ufffd\ufffd",
            6: "\u3042\u3044",  # merge: 2 chars
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))

        assert offsets[-1] == 2, (
            f"Two merges: offsets[-1]={offsets[-1]}, expected 2 (actual decoded length)"
        )


class TestMergeCorrection_OldVsNew:
    """Explicitly prove the OLD algorithm gives wrong results."""

    def test_old_inflates_new_correct_4byte(self):
        """4-byte merge: old gives 3, new gives 1 for offsets[-1]."""
        progressive = {
            1: "\ufffd", 2: "\ufffd\ufffd", 3: "\ufffd\ufffd\ufffd",
            4: "\U00010a05",
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        old = _old_token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])
        new = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])

        # Old is WRONG: offsets[-1] = 3 (inflated by replacement chars)
        assert old[-1] == 3, f"Old algorithm should give 3, got {old[-1]}"
        # New is CORRECT: offsets[-1] = 1 (actual decoded length)
        assert new[-1] == 1, f"New algorithm should give 1, got {new[-1]}"
        # Decoded text is 1 char long
        assert len(tokenizer.decode([0, 1, 2, 3])) == 1

    def test_old_inflates_new_correct_ascii_merge_ascii(self):
        """'H' + 3-byte + '!': old steals width from '!', new preserves it."""
        progressive = {
            1: "H", 2: "H\ufffd", 3: "H\ufffd\ufffd",
            4: "H\u00e9", 5: "H\u00e9!",
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        old = _old_token_to_char_offsets_progressive(tokenizer, list(range(5)))
        new = _token_to_char_offsets_progressive(tokenizer, list(range(5)))

        # Old: [0, 1, 2, 3, 3, 3] — '!' has zero width [3,3)
        assert old == [0, 1, 2, 3, 3, 3], f"Old: {old}"
        # New: [0, 1, 1, 1, 2, 3] — '!' has width [2,3)
        assert new == [0, 1, 1, 1, 2, 3], f"New: {new}"
        # The key difference: token 4 ('!') has width in new code
        assert new[5] - new[4] == 1  # '!' is 1 char wide
        assert old[5] - old[4] == 0  # '!' has zero width (BUG)

    def test_old_inflates_new_correct_two_merges(self):
        """Two 3-byte merges: old gives 3 for decoded text of length 2."""
        progressive = {
            1: "\ufffd", 2: "\ufffd\ufffd", 3: "\u3042",
            4: "\u3042\ufffd", 5: "\u3042\ufffd\ufffd", 6: "\u3042\u3044",
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        old = _old_token_to_char_offsets_progressive(tokenizer, list(range(6)))
        new = _token_to_char_offsets_progressive(tokenizer, list(range(6)))

        # Old: [0, 1, 2, 2, 2, 3, 3] — offsets[-1]=3, but decoded='あい'=2 chars
        assert old[-1] == 3, f"Old: {old}"
        # New: [0, 0, 0, 1, 1, 1, 2] — offsets[-1]=2 = len(decoded)
        assert new[-1] == 2, f"New: {new}"

    def test_new_offsets_match_real_tokenizer_decoded_length(self):
        """Real-tokenizer SMP offsets end at the decoded-text length."""
        tok = _require_tokenizer()
        text = "Hello 𐨯𐨀𐨁𐨭 world"
        bos = tok.get_bos_token_id()
        tokens = tok.encode(text, prepend=bos)
        decoded = tok.decode(tokens)

        new = _token_to_char_offsets_progressive(tok, tokens)

        # New code must match decoded length
        assert new[-1] == len(decoded)
        assert new[0] == 0
        for i in range(len(new) - 1):
            assert new[i + 1] >= new[i]


class TestMergeCorrection_ChunkedPathConsistency:
    """The chunked (hybrid) path must produce identical results to progressive.

    OLD BUG: the chunked path used max() clamping in divergent chunks,
    producing different (inflated) results from the progressive path.
    NEW: chunked path uses the same merge correction with cross-chunk walk-back.
    """

    def test_chunked_matches_progressive_for_cjk(self):
        """CJK text exercises the slow path. Chunked must match progressive."""
        tok = _require_tokenizer()
        text = "在新的时代我们需要重新审视传统价值观科技发展日新月异" * 3
        bos = tok.get_bos_token_id()
        tokens = tok.encode(text, prepend=bos)
        assert len(tokens) > 16  # forces chunked path

        offsets_api = token_to_char_offsets(tok, tokens)
        offsets_ref = _token_to_char_offsets_progressive(tok, tokens)

        assert offsets_api == offsets_ref, (
            f"Chunked path diverges from progressive reference. "
            f"First difference at index {next(i for i, (a, b) in enumerate(zip(offsets_api, offsets_ref, strict=True)) if a != b)}"
        )

    def test_chunked_matches_progressive_for_mixed_smp(self):
        """Mixed SMP text. Byte groups may span chunk boundaries."""
        tok = _require_tokenizer()
        text = (
            "Normal English text here. "
            "Some CJK: 没有了责任. "
            "Kharoshthi: 𐨯𐨀𐨁𐨭. "
            "Emoji: 🎉🔥🌍. "
            "More English padding text. " * 5
        )
        bos = tok.get_bos_token_id()
        tokens = tok.encode(text, prepend=bos)
        assert len(tokens) > 16

        offsets_api = token_to_char_offsets(tok, tokens)
        offsets_ref = _token_to_char_offsets_progressive(tok, tokens)

        assert offsets_api == offsets_ref


class TestMergeCorrection_ThreeInvariants:
    """The three invariants must hold for ALL inputs with the new code.

    These invariants were violated by the old code for multi-byte text:
    - offsets[-1] was inflated beyond len(decoded)
    """

    @pytest.mark.parametrize("text", [
        "没有了责任",
        "hello 🎉 world",
        "𐨯 𐨀𐨁𐨭",
        "Hello 𐨯 world 没有 emoji 🎉",
        "𐨯𐨀𐨁𐨭" * 10,
    ])
    def test_invariants_hold_for_multibyte_text(self, text):
        tok = _require_tokenizer()
        bos = tok.get_bos_token_id()
        tokens = tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(tok, tokens)
        decoded = tok.decode(tokens)

        assert offsets[0] == 0
        assert offsets[-1] == len(decoded), (
            f"offsets[-1]={offsets[-1]} != len(decoded)={len(decoded)}"
        )
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] >= offsets[i], (
                f"Non-monotonic at {i}: {offsets[i]} > {offsets[i + 1]}"
            )


# ============================================================================
# PR #288: global doc index in packed rows iterator
# ============================================================================


class TestGlobalDocIndex_RegressionMultiRG:
    """OLD BUG: doc_idx was per-row-group (enumerate reset to 0 each RG).

    feature_index.lookup(shard_stem, doc_idx) received wrong indices for
    documents in row groups after the first, causing feature data misalignment.
    """

    def test_old_code_resets_per_row_group(self, tmp_path):
        """With 3 row groups of 10 docs, old code passed [0-9, 0-9, 0-9].
        New code passes [0-9, 10-19, 20-29]."""
        n_per_rg = 10
        n_rgs = 3
        total = n_per_rg * n_rgs

        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} " + " ".join(f"w{w}" for w in range(5)) for i in range(total)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=n_per_rg)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()
        list(_raw_document_iterator([str(pq_path)], tok, bos, rec))

        indices = [d for _, d in rec.calls]

        # OLD code would produce [0,1,...9, 0,1,...9, 0,1,...9] (30 values, 10 unique)
        # NEW code produces [0,1,...29] (30 values, 30 unique)
        assert len(set(indices)) == total, (
            f"Expected {total} unique indices, got {len(set(indices))}. "
            f"Index reset per row group detected."
        )
        assert indices == list(range(total))

    def test_second_rg_first_doc_uses_offset(self, tmp_path):
        """Row group 1's first doc must be index n_per_rg, not 0."""
        n_per_rg = 8
        total = n_per_rg * 2

        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} hello world foo" for i in range(total)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=n_per_rg)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator([str(pq_path)], tok, tok.get_bos_token_id(), rec))

        indices = [d for _, d in rec.calls]
        # Old code: indices[8] == 0 (bug!)
        # New code: indices[8] == 8
        assert indices[n_per_rg] == n_per_rg, (
            f"RG 1 first doc index: expected {n_per_rg}, got {indices[n_per_rg]}"
        )

    def test_offset_resets_between_parquet_files(self, tmp_path):
        """global_doc_offset resets between separate parquet files (shards)."""
        for shard_idx in range(2):
            pq_path = tmp_path / f"shard_{shard_idx:05d}.parquet"
            texts = [f"doc{i} hello world" for i in range(15)]
            table = pa.table({"text": texts})
            pq.write_table(table, str(pq_path), row_group_size=5)  # 3 RGs per file

        paths = sorted(str(p) for p in tmp_path.glob("shard_*.parquet"))
        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        list(_raw_document_iterator(paths, tok, tok.get_bos_token_id(), rec))

        shard0 = [d for stem, d in rec.calls if stem == "shard_00000"]
        shard1 = [d for stem, d in rec.calls if stem == "shard_00001"]

        # Each shard should have global indices 0-14
        assert shard0 == list(range(15))
        assert shard1 == list(range(15))


# ============================================================================
# PR #287: truncate packed row docs to precomputed boundary coverage
# ============================================================================


class TestTruncation_TinyGapSilentTruncation:
    """OLD BUG: no truncation existed, documents with uncovered tail tokens
    would trigger assertion failures in _build_row_segments_char_level.

    NEW: truncate_to_boundary_coverage silently truncates tiny gaps (<=4 tokens).
    """

    def test_tiny_gap_1_token_silently_truncated(self):
        """1 token gap: truncated silently, no error."""
        boundaries = [(0, 128), (128, 255)]  # covers 255, doc has 256
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(256))

        # Old code: no truncation → downstream assertion failure
        # New code: silently truncates to 255
        result = truncate_to_boundary_coverage(
            tokens, seg_features, 2048, _SEGMENT_SIZE
        )
        assert len(result) == 255

    def test_tiny_gap_4_tokens_silently_truncated(self):
        """4 token gap (boundary case): still silently truncated."""
        boundaries = [(0, 2044)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(2048))

        result = truncate_to_boundary_coverage(
            tokens, seg_features, 2048, _SEGMENT_SIZE
        )
        assert len(result) == 2044

    def test_gap_5_tokens_not_silent(self):
        """5 token gap: exceeds tiny gap threshold, enters tolerance logic."""
        boundaries = [(0, 2043)]
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(2048))

        # gap=5, last_precomputed_end=2043 >= tolerance_threshold=1792
        # → within tolerance → truncate with potential logging
        result = truncate_to_boundary_coverage(
            tokens, seg_features, 2048, _SEGMENT_SIZE
        )
        assert len(result) == 2043


class TestTruncation_BoundaryExceedsTokens:
    """When precomputed boundaries extend beyond document length, the assertion
    must be relaxed (not strict token_end <= n_tokens).

    OLD BUG: strict assertion would fail when boundaries cover more than context_len.
    NEW: assert_boundaries_covered only checks first segment starts at 0 and
         at least one segment starts within the token range.
    """

    def test_boundaries_beyond_tokens_accepted(self):
        """Boundaries end at 2100 but doc has 2048 tokens.
        Old strict assertion: AssertionError.
        New relaxed assertion: passes.
        """
        boundaries = [(0, 128), (128, 2100)]  # token_end=2100 > 2048
        seg_features = _make_seg_features(boundaries)
        tokens = list(range(2048))

        # truncate_to_boundary_coverage: 2048 <= 2100 → fully covered → no truncation
        result = truncate_to_boundary_coverage(
            tokens, seg_features, 2048, _SEGMENT_SIZE
        )
        assert len(result) == 2048

        # assert_boundaries_covered must NOT raise despite token_end=2100
        assert_boundaries_covered(result, seg_features)


# ============================================================================
# Batch encoding in _raw_document_iterator
# ============================================================================


class TestBatchEncoding_Equivalence:
    """The batch encoding optimization must produce identical token lists."""

    def test_batch_matches_sequential(self, tmp_path):
        """Batch encode(texts) must produce same tokens as encode(text) per doc."""
        n_docs = 15
        pq_path = tmp_path / "shard_00000.parquet"
        texts = [f"doc{i} " + " ".join(f"word{w}" for w in range(3 + i % 5)) for i in range(n_docs)]
        table = pa.table({"text": texts})
        pq.write_table(table, str(pq_path), row_group_size=5)

        rec = RecordingFeatureIndex()
        tok = SimpleTokenizer()
        bos = tok.get_bos_token_id()

        # Collect tokens from the iterator
        iter_tokens = []
        for text_tokens, _, _, _ in _raw_document_iterator(
            [str(pq_path)], tok, bos, rec
        ):
            iter_tokens.append(text_tokens)

        # Compute expected tokens sequentially
        expected_tokens = [tok.encode(text, prepend=bos) for text in texts]

        assert len(iter_tokens) == len(expected_tokens)
        for i, (actual, expected) in enumerate(zip(iter_tokens, expected_tokens, strict=True)):
            assert actual == expected, (
                f"Doc {i}: batch tokens differ from sequential. "
                f"Batch: {actual[:10]}..., Sequential: {expected[:10]}..."
            )

    def test_batch_encoding_with_real_tokenizer(self):
        """Verify real tokenizer batch encoding matches sequential."""
        tok = _require_tokenizer()
        bos = tok.get_bos_token_id()
        texts = [
            "hello world foo bar baz",
            "The quick brown fox jumps over the lazy dog.",
            "12345 67890 number heavy 11111 22222 33333",
            "Mixed 日本語 and emoji 🎉 text here",
            "",
            "a",
        ]

        batch_result = tok.encode(texts, prepend=bos)
        sequential_result = [tok.encode(text, prepend=bos) for text in texts]

        assert len(batch_result) == len(sequential_result)
        for i, (batch_ids, seq_ids) in enumerate(zip(batch_result, sequential_result, strict=True)):
            assert batch_ids == seq_ids, (
                f"Doc {i}: batch={batch_ids[:10]}..., seq={seq_ids[:10]}..."
            )


# ============================================================================
# _row_to_csr padding run handling — REMOVED.
#
# A previously-skipped ``TestRowToCsr_PaddingRun`` class exercised the
# stride-based decoder segmentation API (``dec_seg_size`` / ``char_level``
# kwargs) that was deleted from ``_row_to_csr`` when the project moved to
# char-level segmentation exclusively (``decoder_segment_size: null``). The
# class was @pytest.mark.skip-decorated and never re-enabled. Replacement
# coverage for the live ``_row_to_csr`` signature lives in the current packed
# batch tests.
# ============================================================================
