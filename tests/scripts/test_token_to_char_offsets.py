"""Tests for token_to_char_offsets correctness with multi-byte UTF-8.

Verifies three invariants:
1. offsets[-1] == len(tokenizer.decode(token_ids))
2. offsets[i+1] >= offsets[i] for all i (non-decreasing)
3. offsets[0] == 0

Tests both the nanochat tokenizer and the Gemma tokenizer (skipped if unavailable).
"""

from unittest.mock import MagicMock

import pytest

from nanochat.tokenizer import get_tokenizer
from steering.segmentation import (
    _token_to_char_offsets_progressive,
    token_to_char_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nanochat_tok():
    try:
        return get_tokenizer()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"nanochat tokenizer cache unavailable: {exc}")


@pytest.fixture(scope="module")
def gemma_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        return tok
    except Exception:
        pytest.skip("Gemma tokenizer not available")


def _assert_three_invariants(offsets, token_ids, tokenizer):
    """Assert the three core invariants for token_to_char_offsets."""
    decoded = tokenizer.decode(token_ids)
    assert offsets[0] == 0, f"offsets[0]={offsets[0]}, expected 0"
    assert offsets[-1] == len(decoded), (
        f"offsets[-1]={offsets[-1]} != len(decoded)={len(decoded)} "
        f"for {len(token_ids)} tokens"
    )
    for i in range(len(offsets) - 1):
        assert offsets[i + 1] >= offsets[i], (
            f"Non-monotonic at index {i}: offsets[{i}]={offsets[i]} "
            f"> offsets[{i + 1}]={offsets[i + 1]}"
        )
    assert len(offsets) == len(token_ids) + 1


# ---------------------------------------------------------------------------
# 1. Three invariants — nanochat tokenizer
# ---------------------------------------------------------------------------


class TestThreeInvariantsNanochat:
    """offsets[0]==0, non-decreasing, offsets[-1]==len(decoded) — nanochat."""

    def test_ascii(self, nanochat_tok):
        text = "Hello world, this is a test."
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_cjk(self, nanochat_tok):
        text = "没有了责任"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_emoji(self, nanochat_tok):
        text = "hello 🎉 world"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_kharoshthi(self, nanochat_tok):
        text = "𐨯 𐨀𐨁𐨭"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_mixed(self, nanochat_tok):
        text = "Hello 𐨯 world 没有 emoji 🎉"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_long_ascii(self, nanochat_tok):
        """Long ASCII text — exercises fast path for n > 16."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        assert len(tokens) > 16
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_long_cjk(self, nanochat_tok):
        """Long CJK text — exercises slow/chunked path for n > 16."""
        text = "在新的时代我们需要重新审视传统价值观科技发展日新月异" * 3
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        assert len(tokens) > 16
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_long_mixed_smp(self, nanochat_tok):
        """Long text with scattered SMP characters — exercises chunked path."""
        text = (
            "Normal English text here. "
            "Some CJK: 没有了责任. "
            "Kharoshthi: 𐨯𐨀𐨁𐨭. "
            "Emoji: 🎉🔥🌍. "
            "More English padding text. " * 5
        )
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        assert len(tokens) > 16
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)


# ---------------------------------------------------------------------------
# 2. Three invariants — Gemma tokenizer
# ---------------------------------------------------------------------------


class TestThreeInvariantsGemma:
    """offsets[0]==0, non-decreasing, offsets[-1]==len(decoded) — Gemma."""

    def test_ascii(self, gemma_tok):
        text = "Hello world, this is a test."
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_cjk(self, gemma_tok):
        text = "没有了责任"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_emoji(self, gemma_tok):
        text = "hello 🎉 world"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_kharoshthi(self, gemma_tok):
        """Single Kharoshthi character — the exact reproduction case."""
        text = "𐨯"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        decoded = gemma_tok.decode(tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)
        # Specific check: the bug report example
        assert offsets[-1] == len(decoded), (
            f"Kharoshthi bug: offsets[-1]={offsets[-1]} but "
            f"len(decoded)={len(decoded)} (decoded={decoded!r})"
        )

    def test_kharoshthi_script(self, gemma_tok):
        text = "𐨯 𐨀𐨁𐨭"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_mixed(self, gemma_tok):
        text = "Hello 𐨯 world 没有 emoji 🎉"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)


# ---------------------------------------------------------------------------
# 3. Character classes — nanochat
# ---------------------------------------------------------------------------


class TestCharacterClassesNanochat:
    """Specific character classes with the nanochat tokenizer."""

    def test_pure_ascii_strictly_increasing(self, nanochat_tok):
        """Pure ASCII should have strictly increasing offsets (no zero-width)."""
        text = "Hello world, this is a test."
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] > offsets[i], (
                f"Expected strictly increasing for ASCII at {i}: "
                f"offsets[{i}]={offsets[i]}, offsets[{i + 1}]={offsets[i + 1]}"
            )

    def test_cjk_final_offset_correct(self, nanochat_tok):
        """CJK text: final offset must equal decoded length, not inflated."""
        text = "没有了责任"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)
        assert offsets[-1] == len(decoded)

    def test_emoji_final_offset_correct(self, nanochat_tok):
        """Emoji text: final offset must equal decoded length."""
        text = "hello 🎉 world"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)
        assert offsets[-1] == len(decoded)

    def test_kharoshthi_final_offset_correct(self, nanochat_tok):
        """Kharoshthi text: final offset must equal decoded length."""
        text = "𐨯 𐨀𐨁𐨭"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)
        assert offsets[-1] == len(decoded)

    def test_mixed_final_offset_correct(self, nanochat_tok):
        """Mixed text: final offset must equal decoded length."""
        text = "Hello 𐨯 world 没有 emoji 🎉"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)
        assert offsets[-1] == len(decoded)


# ---------------------------------------------------------------------------
# 4. Edge cases — both tokenizers
# ---------------------------------------------------------------------------


class TestEdgeCasesNanochat:
    """Edge cases with nanochat tokenizer."""

    def test_single_smp_char(self, nanochat_tok):
        text = "𐨯"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_consecutive_smp_chars(self, nanochat_tok):
        text = "𐨯𐨀𐨁𐨭"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_smp_at_start(self, nanochat_tok):
        text = "𐨯 hello world"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_smp_at_end(self, nanochat_tok):
        text = "hello world 𐨯"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_bos_plus_single_smp(self, nanochat_tok):
        """BOS + single SMP character — minimal reproduction."""
        text = "𐨯"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)
        assert offsets[-1] == len(decoded)
        assert offsets[0] == 0

    def test_long_scattered_smp(self, nanochat_tok):
        """Long text with SMP characters scattered throughout."""
        text = (
            "The quick brown 𐨯 fox jumps over the lazy 𐨀 dog. "
            "This is a test with emoji 🎉 and CJK 没有 characters. "
            "More text with 𐨁𐨭 Kharoshthi at various 🔥 points. "
            "The end."
        )
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_progressive_path_directly(self, nanochat_tok):
        """Call _token_to_char_offsets_progressive directly with SMP text."""
        text = "Hello 𐨯 world"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = _token_to_char_offsets_progressive(nanochat_tok, tokens)
        _assert_three_invariants(offsets, tokens, nanochat_tok)


class TestEdgeCasesGemma:
    """Edge cases with Gemma tokenizer."""

    def test_single_smp_char(self, gemma_tok):
        text = "𐨯"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_consecutive_smp_chars(self, gemma_tok):
        text = "𐨯𐨀𐨁𐨭"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_smp_at_start(self, gemma_tok):
        text = "𐨯 hello world"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)

    def test_smp_at_end(self, gemma_tok):
        text = "hello world 𐨯"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        offsets = token_to_char_offsets(gemma_tok, tokens)
        _assert_three_invariants(offsets, tokens, gemma_tok)


# ---------------------------------------------------------------------------
# 5. Mock tests for algorithm correctness
# ---------------------------------------------------------------------------


class TestAlgorithmMock:
    """Mock-based tests verifying the exact merge-walkback behavior."""

    def test_4byte_merge_produces_correct_final_offset(self):
        """4 byte tokens merging into 1 char: offsets[-1] must equal 1."""
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\ufffd\ufffd\ufffd",
            4: "\U00010a05",  # 1 char
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])
        assert offsets[-1] == 1, f"Expected offsets[-1]=1, got {offsets[-1]}"
        assert offsets[0] == 0

    def test_two_3byte_merges_correct_final_offset(self):
        """Two 3-byte chars: offsets[-1] must equal 2."""
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "あ",  # merge: 1 char
            4: "あ\ufffd",
            5: "あ\ufffd\ufffd",
            6: "あい",  # merge: 2 chars
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))
        assert offsets[-1] == 2, f"Expected offsets[-1]=2, got {offsets[-1]}"
        assert offsets[0] == 0

    def test_ascii_plus_merge_correct_offsets(self):
        """'Hi' + 3-byte char + '!': offsets must match decoded length."""
        progressive = {
            1: "H",
            2: "Hi",
            3: "Hi\ufffd",
            4: "Hi\ufffd\ufffd",
            5: "Hi\u00e9",  # merge: len=3
            6: "Hi\u00e9!",  # len=4
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))
        assert offsets[-1] == 4, f"Expected offsets[-1]=4, got {offsets[-1]}"
        assert offsets[0] == 0
        # Non-decreasing
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] >= offsets[i]

    def test_bos_plus_4byte_merge(self):
        """BOS (5 chars) + 4-byte SMP char: offsets[-1] must be 6."""
        progressive = {
            1: "<bos>",  # 5 chars
            2: "<bos>\ufffd",  # 6
            3: "<bos>\ufffd\ufffd",  # 7
            4: "<bos>\ufffd\ufffd\ufffd",  # 8
            5: "<bos>\U00010a2f",  # 6 chars (merge!)
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(
            tokenizer, [0, 1, 2, 3, 4],
        )
        assert offsets[-1] == 6, f"Expected offsets[-1]=6, got {offsets[-1]}"
        assert offsets[0] == 0
        assert offsets[1] == 5  # BOS offset preserved

    def test_ascii_unaffected_by_fix(self):
        """Pure ASCII tokens: no merge, offsets strictly increasing."""
        def mock_decode(ids):
            return "a" * len(ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(5)))
        assert offsets == [0, 1, 2, 3, 4, 5]

    def test_dangling_bytes_never_merge(self):
        """Byte-fallback tokens that never form a valid character stay as
        individual replacement chars — no merge, no walk-back."""
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\ufffd\ufffd\ufffd",
        }
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2])
        assert offsets == [0, 1, 2, 3]
        # Strictly increasing — each replacement char is 1 position
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] > offsets[i]


# ---------------------------------------------------------------------------
# 6. Fast path unaffected
# ---------------------------------------------------------------------------


class TestFastPathUnaffected:
    """Pure ASCII should still use the fast path and produce correct results."""

    def test_fast_path_ascii_nanochat(self, nanochat_tok):
        from steering.segmentation import _path_stats

        for v in _path_stats.values():
            v.clear()

        text = "The quick brown fox jumps over the lazy dog. " * 10
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        assert len(tokens) > 16
        offsets = token_to_char_offsets(nanochat_tok, tokens)

        from steering.segmentation import get_path_stats

        stats = get_path_stats()
        assert stats["fast"]["count"] == 1, "ASCII text should use fast path"
        _assert_three_invariants(offsets, tokens, nanochat_tok)

    def test_fast_path_matches_progressive(self, nanochat_tok):
        """Fast path and progressive produce identical results for ASCII."""
        text = "Simple ASCII text for testing fast path equivalence."
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets_fast = token_to_char_offsets(nanochat_tok, tokens)
        offsets_progressive = _token_to_char_offsets_progressive(
            nanochat_tok, tokens,
        )
        assert offsets_fast == offsets_progressive


# ---------------------------------------------------------------------------
# 7. Upstream expansion integration
# ---------------------------------------------------------------------------


class TestUpstreamExpansion:
    """_boundaries_from_offsets works correctly with corrected offsets."""

    def test_cjk_boundaries(self, nanochat_tok):
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "没有了责任 就没有了方向。"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)

        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, nanochat_tok,
            source_bos_char_len=5,
        )
        assert tok_starts[0] == 0
        assert tok_ends[0] > 0

    def test_smp_boundaries(self, nanochat_tok):
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Hello 𐨯 world"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)

        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, nanochat_tok,
            source_bos_char_len=5,
        )
        assert tok_starts[0] == 0
        assert tok_ends[0] > 0

    def test_mixed_boundaries_two_segments(self, nanochat_tok):
        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Hello 世界! Testing 日本語 here."
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)

        mid = offsets[-1] // 2
        seg_starts = [0, mid]
        seg_ends = [mid, offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, nanochat_tok,
            source_bos_char_len=5,
        )
        assert len(tok_starts) == 2
        assert tok_starts[0] == 0
        assert tok_ends[0] == tok_starts[1]  # segments tile


# ---------------------------------------------------------------------------
# 8. Synthetic regression test (simulates shard 81 doc 43130 scenario)
# ---------------------------------------------------------------------------


class TestSyntheticRegression:
    """Simulate the real failing scenario: a document with Kharoshthi script
    where segment char boundaries are wrong due to inflated offsets.

    In the real bug, Gemma tokens 1024-1152 covering a 59-char segment
    were reported as 17 chars because offsets were inflated by replacement
    characters from SMP byte-fallback tokens.
    """

    def test_segment_char_range_not_inflated(self, nanochat_tok):
        """Verify that a segment containing SMP characters has the correct
        character range (not inflated by replacement chars)."""
        # Build a document with ASCII + SMP characters
        ascii_prefix = "Normal English text. " * 10  # ~200 chars
        smp_segment = "𐨯𐨀𐨁𐨭" * 10  # 40 SMP chars = 40 Python chars
        ascii_suffix = "More normal text. " * 10  # ~180 chars

        text = ascii_prefix + smp_segment + ascii_suffix
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)
        decoded = nanochat_tok.decode(tokens)

        # Critical: offsets[-1] must match actual decoded length
        assert offsets[-1] == len(decoded), (
            f"Inflated offsets: offsets[-1]={offsets[-1]} but "
            f"len(decoded)={len(decoded)}"
        )

        # Find the token range covering the SMP segment by looking at
        # the character positions of the SMP text
        bos_decoded = nanochat_tok.decode([bos])
        smp_start_char = len(bos_decoded) + len(ascii_prefix)
        smp_end_char = smp_start_char + len(smp_segment)

        # Find tokens that fall in the SMP range
        smp_token_start = None
        smp_token_end = None
        for i in range(len(tokens)):
            if offsets[i + 1] > smp_start_char and smp_token_start is None:
                smp_token_start = i
            if offsets[i] >= smp_end_char and smp_token_end is None:
                smp_token_end = i
                break
        if smp_token_end is None:
            smp_token_end = len(tokens)
        assert smp_token_start is not None

        # The char range for the SMP segment should be ~40 chars, not inflated
        char_range = offsets[smp_token_end] - offsets[smp_token_start]
        assert char_range <= len(smp_segment) + 10, (
            f"Char range {char_range} is too large for {len(smp_segment)}-char SMP segment "
            f"(likely inflated by replacement chars)"
        )

    def test_assign_tokens_after_fix(self, nanochat_tok):
        """Verify that assign_tokens_to_segments works correctly after the fix
        for text containing SMP characters."""

        from scripts.preprocess.precompute_token_boundaries import _boundaries_from_offsets

        text = "Hello 𐨯𐨀𐨁𐨭 world 没有了责任 end"
        bos = nanochat_tok.get_bos_token_id()
        tokens = nanochat_tok.encode(text, prepend=bos)
        offsets = token_to_char_offsets(nanochat_tok, tokens)

        # Single segment covering the full text
        seg_starts = [0]
        seg_ends = [offsets[-1]]

        tok_starts, tok_ends = _boundaries_from_offsets(
            tokens, seg_starts, seg_ends, nanochat_tok,
            source_bos_char_len=5,
        )
        assert tok_starts[0] == 0
        assert tok_ends[0] == len(tokens)

        # Two segments
        mid = offsets[-1] // 2
        seg_starts_2 = [0, mid]
        seg_ends_2 = [mid, offsets[-1]]

        tok_starts_2, tok_ends_2 = _boundaries_from_offsets(
            tokens, seg_starts_2, seg_ends_2, nanochat_tok,
            source_bos_char_len=5,
        )
        assert len(tok_starts_2) == 2
        assert tok_starts_2[0] == 0
        assert tok_ends_2[0] == tok_starts_2[1]
        assert tok_ends_2[1] <= len(tokens)


# ---------------------------------------------------------------------------
# Regression: chunked vs progressive disagreement on long byte-fallback runs
# ---------------------------------------------------------------------------


class TestChunkedVsProgressiveAgreement:
    """The chunked (default) and progressive (slow) char-offset paths must
    return identical offsets for the same input.

    Regression guard for a bug where the chunked path's slow per-chunk
    fallback's merge-correction walk-back could cross chunk boundaries and
    clamp offsets from a previous chunk, producing zero-width source
    segments. The fix (bail to progressive on first divergent chunk)
    guarantees the two paths always agree. If per-chunk slow fallback
    logic is ever re-introduced, these tests will catch any divergence.

    The bug was observed in production on ``shard_00355/doc_id=15373``
    (a Wikipedia article containing an Old Italic inscription whose
    every byte becomes its own Gemma token). The long byte-fallback run
    collapsed ~50 token offsets onto a single character position, even
    though by UTF-8 design 32 bytes always span at least 8 characters.
    """

    # The Old Italic inscription that triggered the bug in shard_00355.
    # Two leading ASCII chars give the chunked path enough non-fallback
    # context to start with valid offsets, then the long byte-fallback run
    # exposes the walk-back over-correction.
    KHAROSHTHI_REPRO_TEXT = (
        "x " + "𐌇𐌀𐌓𐌉𐌙𐌀𐌔𐌕𐌉𐌕𐌄𐌉𐌅𐌀" * 4 + " y"
    )

    def test_chunked_agrees_with_progressive_on_short_unicode(self, gemma_tok):
        """A short multi-byte sequence is handled identically by both paths."""
        text = "Hello 𐨯 world"
        tokens = gemma_tok.encode(text, add_special_tokens=True)
        chunked = token_to_char_offsets(gemma_tok, tokens)
        slow = _token_to_char_offsets_progressive(gemma_tok, tokens)
        assert chunked == slow, (
            f"chunked vs progressive disagree on short input.\n"
            f"chunked={chunked}\nslow={slow}"
        )

    def test_chunked_agrees_with_progressive_on_long_byte_fallback_run(
        self, gemma_tok
    ):
        """The chunked path must match the progressive path on the exact
        Old Italic / Kharoshthi-style sequence that broke shard_00355.

        Regression test for the chunked-path walk-back over-correction
        that produced zero-width source segments. The buggy slow per-chunk
        fallback in ``_token_to_char_offsets_chunked`` was removed in the
        same commit; the function now bails to the full progressive path
        on the first divergent chunk, so the only path that diverges from
        progressive on long byte-fallback runs is no longer reachable."""
        tokens = gemma_tok.encode(
            self.KHAROSHTHI_REPRO_TEXT, add_special_tokens=True
        )
        # Sanity: this input should actually exercise the long byte-fallback
        # path, not just trivially agree.
        assert len(tokens) > 32, (
            f"test fixture too short to exercise chunked path: {len(tokens)} tokens"
        )
        chunked = token_to_char_offsets(gemma_tok, tokens)
        slow = _token_to_char_offsets_progressive(gemma_tok, tokens)
        # The two paths must agree element-for-element. When they disagree,
        # the chunked path is wrong (it produces zero-width offsets that
        # cannot represent any UTF-8 byte sequence).
        diffs = [
            (i, chunked[i], slow[i])
            for i in range(min(len(chunked), len(slow)))
            if chunked[i] != slow[i]
        ]
        assert not diffs, (
            f"chunked vs progressive disagree on {len(diffs)} indices "
            f"(first 5: {diffs[:5]}). Total tokens: {len(tokens)}."
        )

    def test_progressive_path_advances_at_least_one_char_per_four_bytes(
        self, gemma_tok
    ):
        """Sanity check on the progressive path: it must always advance by
        at least 1 character per 4 byte-fallback tokens, because UTF-8
        guarantees that 5 bytes span at least 2 characters."""
        tokens = gemma_tok.encode(
            self.KHAROSHTHI_REPRO_TEXT, add_special_tokens=True
        )
        slow = _token_to_char_offsets_progressive(gemma_tok, tokens)
        # For every window of 5 consecutive tokens (≥5 bytes), the offset
        # must have advanced by at least 1 character.
        for i in range(len(slow) - 5):
            advance = slow[i + 5] - slow[i]
            assert advance >= 1, (
                f"progressive path did not advance over 5 tokens at i={i}: "
                f"slow[{i}]={slow[i]}, slow[{i + 5}]={slow[i + 5]} "
                f"(advance={advance}, expected ≥1)"
            )
