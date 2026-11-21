"""Tests for the hybrid chunked char offset path.

Verifies that token_to_char_offsets produces identical output to the
reference _token_to_char_offsets_progressive for all inputs, including
property-based testing with hypothesis.
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.segmentation import (
    _token_to_char_offsets_progressive,
    token_to_char_offsets,
)
from tests.conftest import MockTokenizer

# ---------------------------------------------------------------------------
# Reference comparison tests with MockTokenizer
# ---------------------------------------------------------------------------


class TestHybridMatchesReference:
    """Verify hybrid path matches progressive for all edge cases."""

    @pytest.fixture()
    def tok(self):
        return MockTokenizer()

    def test_empty(self, tok):
        assert token_to_char_offsets(tok, []) == [0]

    def test_single_token(self, tok):
        ids = tok.encode("a")
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_short_doc(self, tok):
        ids = tok.encode("hello world")
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_long_doc(self, tok):
        ids = tok.encode("a" * 500)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_exactly_16_tokens(self, tok):
        """Boundary: n=16 uses small_n path (progressive)."""
        ids = tok.encode("a" * 16)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_17_tokens(self, tok):
        """Boundary: n=17 uses fast/hybrid path."""
        ids = tok.encode("a" * 17)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_chunk_boundary_exact(self, tok):
        """Document length exactly at chunk boundary (32 tokens)."""
        ids = tok.encode("a" * 32)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_chunk_boundary_plus_one(self, tok):
        ids = tok.encode("a" * 33)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)

    def test_multiple_chunks(self, tok):
        ids = tok.encode("a" * 200)
        assert token_to_char_offsets(tok, ids) == _token_to_char_offsets_progressive(tok, ids)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestHybridPropertyBased:
    """Property-based tests ensuring hybrid always matches progressive."""

    @given(text=st.text(min_size=17, max_size=500, alphabet=st.characters(codec="ascii")))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_ascii_texts_match_reference(self, text):
        """For any ASCII text, hybrid matches progressive."""
        tok = MockTokenizer()
        ids = tok.encode(text)
        if len(ids) <= 16:
            return  # small_n path, not testing hybrid
        ref = _token_to_char_offsets_progressive(tok, ids)  # ty: ignore[invalid-argument-type]
        result = token_to_char_offsets(tok, ids)  # ty: ignore[invalid-argument-type]
        assert result == ref

    @given(text=st.text(min_size=17, max_size=200))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_unicode_texts_match_reference(self, text):
        """For any Unicode text, hybrid matches progressive."""
        tok = MockTokenizer()
        ids = tok.encode(text)
        if len(ids) <= 16:
            return
        ref = _token_to_char_offsets_progressive(tok, ids)  # ty: ignore[invalid-argument-type]
        result = token_to_char_offsets(tok, ids)  # ty: ignore[invalid-argument-type]
        assert result == ref

    @given(
        n_tokens=st.integers(min_value=17, max_value=300),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_random_token_ids_match_reference(self, n_tokens):
        """For random token ID sequences, hybrid matches progressive."""
        tok = MockTokenizer()
        ids = [i % (tok.vocab_size - 2) + 2 for i in range(n_tokens)]
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref


# ---------------------------------------------------------------------------
# Tests with real tokenizer (if available)
# ---------------------------------------------------------------------------


class TestHybridWithRealTokenizer:
    """Tests using the real nanochat tokenizer for byte-fallback scenarios."""

    @pytest.fixture()
    def tok(self):
        try:
            import pickle
            with open("nanochat_cache/tokenizer/tokenizer.pkl", "rb") as f:
                enc = pickle.load(f)
            from nanochat.tokenizer import RustBPETokenizer
            return RustBPETokenizer(enc, bos_token="<|bos|>")
        except (FileNotFoundError, ImportError):
            pytest.skip("Real tokenizer not available")

    def test_japanese_matches(self, tok):
        ids = tok.encode("日本語のテキスト。" * 10, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_mixed_script_matches(self, tok):
        ids = tok.encode(
            "Hello 世界 こんにちは café résumé 🎉 " * 5,
            prepend=tok.get_bos_token_id(),
        )
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_mostly_ascii_with_emoji(self, tok):
        text = "normal text " * 50 + "🎉" + " more text " * 50
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_korean_matches(self, tok):
        ids = tok.encode("한국어 텍스트입니다. " * 20, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_early_bailout_all_cjk(self, tok):
        """All-CJK documents should trigger early bailout (2+ divergent in first 4 chunks)."""
        ids = tok.encode("日本語テスト文章。" * 80, prepend=tok.get_bos_token_id())
        assert len(ids) > 128  # enough for multiple chunks
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_divergence_only_at_start(self, tok):
        """Non-ASCII at start, ASCII for the rest — only first chunk diverges."""
        text = "日本語。" + "normal english text here. " * 40
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_divergence_only_at_end(self, tok):
        """ASCII at start, non-ASCII at end — only last chunk diverges."""
        text = "normal english text here. " * 40 + "日本語テスト。"
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_divergence_scattered(self, tok):
        """Non-ASCII scattered throughout — multiple divergent chunks with fast ones between."""
        parts = []
        for i in range(10):
            parts.append("normal text " * 5)
            parts.append("café" if i % 3 == 0 else "日本")
        text = "".join(parts)
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref

    def test_offsets_monotonic(self, tok):
        """Hybrid path must produce monotonically non-decreasing offsets."""
        text = "Hello 世界 " * 30 + "café " * 20 + "한국어 " * 10
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        offsets = token_to_char_offsets(tok, ids)
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] >= offsets[i], f"Non-monotonic at {i}: {offsets[i]} > {offsets[i+1]}"

    def test_integration_with_assign_tokens(self, tok):
        """Hybrid offsets work correctly with assign_tokens_to_segments downstream."""
        from steering.segmentation import assign_tokens_to_segments
        text = "normal text " * 30 + "🎉 emoji test " + "more text " * 30
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        offsets = token_to_char_offsets(tok, ids)
        tok_starts = t.tensor(offsets[:-1], dtype=t.int64)
        tok_ends = t.tensor(offsets[1:], dtype=t.int64)

        # Create tiling segments over the full character range
        total_chars = offsets[-1]
        n_segs = 4
        seg_size = total_chars // n_segs
        seg_starts = t.tensor([i * seg_size for i in range(n_segs)], dtype=t.int64)
        seg_ends = t.tensor(
            [total_chars if i == n_segs - 1 else (i + 1) * seg_size for i in range(n_segs)],
            dtype=t.int64,
        )

        # Should not raise — every token must overlap with a segment
        assignments = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert len(assignments) == len(ids)
        assert (assignments >= 0).all()
        assert (assignments < n_segs).all()

    @given(
        ascii_part=st.text(min_size=20, max_size=100, alphabet=st.characters(codec="ascii")),
        unicode_part=st.text(
            min_size=5, max_size=50,
            alphabet=st.characters(
                categories=("L", "N", "P", "S"),
                exclude_characters="\x00",
            ),
        ),
        repeat=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_hypothesis_mixed_text(self, tok, ascii_part, unicode_part, repeat):
        """Property: hybrid always matches progressive for mixed ASCII + Unicode."""
        text = (ascii_part + unicode_part) * repeat
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        if len(ids) <= 16:
            return
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref, (
            f"Mismatch for {len(ids)} tokens. "
            f"First diff at {next(i for i,(a,b) in enumerate(zip(ref,result, strict=True)) if a!=b)}"
        )

    @given(
        text=st.text(
            min_size=30, max_size=200,
            alphabet=st.characters(
                categories=("L", "N", "P", "S", "Z"),
                exclude_characters="\x00",
            ),
        ),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_hypothesis_arbitrary_unicode(self, tok, text):
        """Property: hybrid matches progressive for arbitrary Unicode text."""
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        if len(ids) <= 16:
            return
        ref = _token_to_char_offsets_progressive(tok, ids)
        result = token_to_char_offsets(tok, ids)
        assert result == ref
