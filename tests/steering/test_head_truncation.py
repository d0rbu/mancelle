"""Tests for head-truncation behavior (SAE-only truncation)."""

from __future__ import annotations

import pytest

from steering.extract_features import (
    _precompute_batch_cpu_work,
    _tokenize_and_segment_chunk,
    _tokenize_chunk,
    _truncate_to_training_window,
)
from steering.segmentation import SegmentMode
from tests.cross_tokenizer_helpers import SimpleTokenizer


class TestTruncateToTrainingWindow:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_no_truncation_short_doc(self):
        """Short doc fits entirely -- all SAE tokens kept."""
        text = "Hello"
        kept_ids, n_full = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=10,
            sae_max_context=100,
        )
        assert kept_ids == self.sae_tok.encode(text)
        assert n_full == len(self.sae_tok.encode(text))

    def test_truncation_applied(self):
        """Document exceeding context_window is truncated to context_window SAE tokens."""
        text = "Hello World!"  # 12 chars = 12 SAE tokens
        kept_ids, n_full = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=8,
            sae_max_context=100,
        )
        assert len(kept_ids) == 8
        assert n_full == 12
        assert self.sae_tok.decode(kept_ids) == "Hello Wo"

    def test_sae_cap_applied(self):
        """SAE max context caps the kept tokens."""
        text = "A" * 100
        kept_ids, _ = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=20,
            sae_max_context=5,
        )
        assert len(kept_ids) == 5

    def test_exact_context_window_no_truncation(self):
        """Doc with exactly context_window SAE tokens -- no truncation."""
        text = "ABCDEFGH"  # 8 chars = 8 SAE tokens
        kept_ids, _ = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=8,
            sae_max_context=100,
        )
        assert kept_ids == self.sae_tok.encode(text)

    def test_empty_text(self):
        text = ""
        kept_ids, n_full = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=10,
            sae_max_context=100,
        )
        assert len(kept_ids) == 0
        assert n_full == 0

    def test_head_not_tail(self):
        """Truncation keeps the HEAD of the document, not the tail."""
        text = "START_" + "x" * 100 + "_END"
        kept_ids, _ = _truncate_to_training_window(
            text,
            self.sae_tok,
            context_window=10,
            sae_max_context=100,
        )
        decoded = self.sae_tok.decode(kept_ids)
        assert decoded.startswith("START")  # head is preserved
        assert "END" not in decoded  # tail is gone


class TestTokenizeChunk:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_returns_two_tuple(self):
        """Returns (kept_ids, n_tokens)."""
        results = _tokenize_chunk(self.sae_tok, ["Hello"], 10, 100)
        kept_ids, n_tokens = results[0]  # 2-tuple, not 3

    def test_head_truncation_for_long_doc(self):
        """Long doc is head-truncated based on SAE tokenizer."""
        text = "START_" + "x" * 100 + "_END"
        results = _tokenize_chunk(self.sae_tok, [text], 10, 100)
        kept_ids, n_full = results[0]
        decoded = self.sae_tok.decode(kept_ids)
        assert decoded.startswith("START")
        assert "END" not in decoded

    def test_batch_all_head_truncated(self):
        """All docs in a batch use head truncation."""
        texts = ["Short", "A" * 100, "B" * 50]
        results = _tokenize_chunk(self.sae_tok, texts, 5, 100)
        assert len(results) == 3
        for kept_ids, _ in results:
            assert len(kept_ids) > 0

    def test_no_head_truncate_flag(self):
        """The head_truncate parameter no longer exists."""
        with pytest.raises(TypeError):
            _tokenize_chunk(self.sae_tok, ["hi"], 10, 100, head_truncate=True)  # ty: ignore[unknown-argument]

    def test_empty_text_raises(self):
        """Empty text produces 0 tokens -- no error since text is empty."""
        results = _tokenize_chunk(self.sae_tok, [""], 10, 100)
        kept_ids, n_full = results[0]
        assert len(kept_ids) == 0
        assert n_full == 0


class TestPrecomputeBatchCpuWork:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_input_is_two_tuple(self):
        """Batch items are (kept_ids, n_tokens)."""
        batch = [(self.sae_tok.encode("Hello"), 5)]  # 2-tuple
        _precompute_batch_cpu_work(batch, self.sae_tok, SegmentMode.fixed_tokens, 3)
        # Should not raise

    def test_output_is_four_tuple(self):
        """Output is (segments, n_tokens, char_starts, char_ends)."""
        batch = [(self.sae_tok.encode("Hello World"), 11)]
        results = _precompute_batch_cpu_work(
            batch, self.sae_tok, SegmentMode.fixed_tokens, 3
        )
        segments, n_tokens, char_starts, char_ends = results[0]  # 4-tuple

    def test_char_starts_are_absolute_from_zero(self):
        """Char starts are absolute positions (no offset shift)."""
        text = "Hello World! Nice day"
        ids = self.sae_tok.encode(text)
        batch = [(ids, len(ids))]
        results = _precompute_batch_cpu_work(
            batch, self.sae_tok, SegmentMode.fixed_tokens, 3
        )
        segments, n_tokens, char_starts, char_ends = results[0]
        # First segment starts at 0
        assert char_starts[0] == 0


class TestTokenizeAndSegmentChunk:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_accepts_sae_max_context(self):
        token_info, precomputed = _tokenize_and_segment_chunk(
            self.sae_tok,
            ["Hello World!"],
            context_window=10,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=3,
            sae_max_context=100,
        )
        kept_ids, n_tokens = token_info[0]  # 2-tuple
        segments, n_tokens2, char_starts, char_ends = precomputed[0]  # 4-tuple

    def test_no_head_truncate_param(self):
        """head_truncate parameter removed."""
        with pytest.raises(TypeError):
            _tokenize_and_segment_chunk(
                self.sae_tok,
                ["hi"],
                10,
                SegmentMode.fixed_tokens,
                3,
                sae_max_context=100,
                head_truncate=True,  # ty: ignore[unknown-argument]
            )

    def test_head_truncation_applied(self):
        """Long doc is head-truncated in combined function."""
        text = "START_" + "x" * 100 + "_END"
        token_info, precomputed = _tokenize_and_segment_chunk(
            self.sae_tok,
            [text],
            context_window=10,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=3,
            sae_max_context=100,
        )
        kept_ids, n_full = token_info[0]
        decoded = self.sae_tok.decode(kept_ids)
        assert decoded.startswith("START")
        assert "END" not in decoded


class TestCharOffsetInvariants:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_offsets_always_start_at_zero(self):
        """First segment's char_start is always 0, regardless of doc length."""
        for text in ["Short", "A" * 50, "Hello World! This is longer."]:
            ids = self.sae_tok.encode(text)
            batch = [(ids, len(ids))]
            results = _precompute_batch_cpu_work(
                batch, self.sae_tok, SegmentMode.fixed_tokens, 3
            )
            _, _, char_starts, _ = results[0]
            assert char_starts[0] == 0, f"Failed for text={text!r}"

    def test_head_truncation_offsets_start_at_zero(self):
        """With head truncation of a long doc, char_starts[0] is still 0."""
        text = "A" * 200
        kept_ids, _ = _truncate_to_training_window(text, self.sae_tok, 10, 100)
        batch = [(kept_ids, 200)]
        results = _precompute_batch_cpu_work(
            batch, self.sae_tok, SegmentMode.fixed_tokens, 8
        )
        _, _, char_starts, _ = results[0]
        assert char_starts[0] == 0

    def test_segments_cover_correct_text(self):
        """text[char_start:char_end] matches the segment's decoded tokens."""
        text = "Hello World! Nice day today."
        ids = self.sae_tok.encode(text)
        batch = [(ids, len(ids))]
        results = _precompute_batch_cpu_work(
            batch, self.sae_tok, SegmentMode.fixed_tokens, 5
        )
        segments, _, char_starts, char_ends = results[0]
        decoded_full = self.sae_tok.decode(ids)
        for (s, e), cs, ce in zip(segments, char_starts, char_ends, strict=True):
            segment_ids = ids[s:e]
            segment_text = self.sae_tok.decode(segment_ids)
            expected_text = decoded_full[cs:ce]
            assert segment_text == expected_text, f"Segment [{s}:{e}] text mismatch"

    def test_segments_tile_without_gaps(self):
        """Adjacent segments: char_ends[i] == char_starts[i+1]."""
        text = "The quick brown fox jumps over the lazy dog"
        ids = self.sae_tok.encode(text)
        batch = [(ids, len(ids))]
        results = _precompute_batch_cpu_work(
            batch, self.sae_tok, SegmentMode.fixed_tokens, 7
        )
        _, _, char_starts, char_ends = results[0]
        for i in range(len(char_starts) - 1):
            assert char_ends[i] == char_starts[i + 1], (
                f"Gap between segment {i} end ({char_ends[i]}) and "
                f"segment {i + 1} start ({char_starts[i + 1]})"
            )

    def test_truncation_keeps_head(self):
        """After truncation, decoded text comes from the head of the document."""
        text = "Hello World!"
        kept_ids, _ = _truncate_to_training_window(text, self.sae_tok, 8, 100)
        decoded = self.sae_tok.decode(kept_ids)
        assert decoded == "Hello Wo"
        # All decoded chars are from the head
        assert text.startswith(decoded)


class TestHeadTruncationIntegration:
    def setup_method(self):
        self.sae_tok = SimpleTokenizer()

    def test_full_pipeline_produces_valid_output(self):
        """_tokenize_and_segment_chunk produces valid segments and char offsets."""
        text = "The quick brown fox jumps over the lazy dog."
        token_info, precomputed = _tokenize_and_segment_chunk(
            self.sae_tok,
            [text],
            context_window=100,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=8,
            sae_max_context=200,
        )
        kept_ids, n_full = token_info[0]
        segments, n_tokens, char_starts, char_ends = precomputed[0]
        assert len(segments) > 0
        assert len(char_starts) == len(segments)
        assert char_starts[0] == 0

    def test_truncated_doc_keeps_head(self):
        """Features cover beginning of document, not end."""
        text = "START " + "middle " * 50 + "END"
        token_info, _ = _tokenize_and_segment_chunk(
            self.sae_tok,
            [text],
            context_window=10,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=4,
            sae_max_context=200,
        )
        kept_ids, n_full = token_info[0]
        decoded = self.sae_tok.decode(kept_ids)
        assert decoded.startswith("START")
        assert "END" not in decoded

    def test_multiple_docs_in_batch(self):
        """Multiple docs in a batch all processed correctly."""
        texts = [
            "First document with some text.",
            "Second document is a bit longer with more words.",
            "Third short.",
        ]
        token_info, precomputed = _tokenize_and_segment_chunk(
            self.sae_tok,
            texts,
            context_window=100,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=5,
            sae_max_context=200,
        )
        assert len(token_info) == 3
        assert len(precomputed) == 3
        for _, (segments, _, char_starts, _) in zip(
            token_info, precomputed, strict=True
        ):
            assert len(segments) > 0
            assert char_starts[0] == 0

    def test_sae_cap_in_pipeline(self):
        """SAE max context cap is applied through the pipeline."""
        text = "A" * 200
        token_info, _ = _tokenize_and_segment_chunk(
            self.sae_tok,
            [text],
            context_window=100,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=4,
            sae_max_context=10,
        )
        kept_ids, n_full = token_info[0]
        assert len(kept_ids) <= 10
        assert n_full == 200
