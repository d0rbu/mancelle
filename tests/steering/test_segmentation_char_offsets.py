"""Tests for token_to_char_offsets in segmentation module.

Covers:
- Mock tests for fast path / slow path / small-n routing
- Real GPT-2 tokenizer tests (ASCII, CJK, mixed multi-byte)
- Hypothesis equivalence tests (fast path == progressive for all inputs)
- Edge cases (empty, single token, special tokens, empty-decode tokens)
- Path-tracking instrumentation tests
"""

from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from transformers import GPT2Tokenizer

from steering.segmentation import (
    _path_stats,
    _token_to_char_offsets_progressive,
    get_path_stats,
    token_to_char_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def _reset_path_stats():
    """Clear path stats before a test."""
    for v in _path_stats.values():
        v.clear()


# ---------------------------------------------------------------------------
# 2a. Mock tests for fast / slow / small-n routing
# ---------------------------------------------------------------------------


class TestFastPath:
    """n > 16, per-token concat == full decode → fast O(n) path."""

    def test_repeated_singletons_are_reused_only_within_the_current_call(self):
        words = {0: "", 1: "ab", 2: "日"}
        calls = []

        def decode(ids):
            calls.append(tuple(ids))
            return "".join(words[tid] for tid in ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = decode
        ids = [2, 0, 1] * 20
        for replacement in ("ab", "longer"):
            words[1] = replacement
            calls.clear()
            expected = [len(decode(ids[:i])) for i in range(len(ids) + 1)]
            calls.clear()
            assert token_to_char_offsets(tokenizer, ids) == expected
            assert calls == [(2,), (0,), (1,), tuple(ids)]

    def test_ascii_fast_path(self):
        """20 ASCII tokens where per-token decode concatenates correctly."""
        n = 20
        words = [f"w{i}" for i in range(n)]

        def mock_decode(ids):
            if len(ids) == 1:
                return words[ids[0]]
            return "".join(words[tid] for tid in ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        token_ids = list(range(n))
        offsets = token_to_char_offsets(tokenizer, token_ids)

        # Verify offsets are correct
        assert offsets[0] == 0
        assert len(offsets) == n + 1
        expected = [0]
        cum = 0
        for w in words:
            cum += len(w)
            expected.append(cum)
        assert offsets == expected

    def test_fast_path_records_stats(self):
        """Fast path records timing in _path_stats['fast']."""
        n = 20
        words = [f"x{i}" for i in range(n)]

        def mock_decode(ids):
            if len(ids) == 1:
                return words[ids[0]]
            return "".join(words[tid] for tid in ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        token_to_char_offsets(tokenizer, list(range(n)))
        stats = get_path_stats()
        assert stats["fast"]["count"] == 1
        assert stats["slow"]["count"] == 0
        assert stats["small_n"]["count"] == 0


class TestSlowPathMultiByte:
    """n > 16, per-token decode differs from full decode (multi-byte)."""

    def test_repeated_context_dependent_tokens_keep_progressive_offsets(self):
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: " ".join("word" for _ in ids)
        ids = [7] * 70
        assert token_to_char_offsets(tokenizer, ids) == _token_to_char_offsets_progressive(
            tokenizer, ids
        )

    def test_multibyte_triggers_slow_path(self):
        """When per-token decode produces replacement chars, slow path is used."""
        n = 20
        # Simulate: per-token decode gives replacement chars for multi-byte tokens
        per_token = ["a"] * n

        # Full decode has a multi-byte char that doesn't match concatenation
        per_token[5] = "\ufffd"  # replacement char individually
        full_text = "a" * 5 + "日" + "a" * 14  # actual char in full decode

        progressive_results = {}
        cum = ""
        for i in range(n):
            if i < 5:
                cum += "a"
            elif i == 5:
                cum += "日"
            else:
                cum += "a"
            progressive_results[i + 1] = cum

        def mock_decode(ids):
            if len(ids) == 1:
                return per_token[ids[0]]
            if len(ids) <= n:
                return progressive_results.get(len(ids), "")
            return full_text

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        offsets = token_to_char_offsets(tokenizer, list(range(n)))
        stats = get_path_stats()

        assert stats["slow"]["count"] == 1
        assert stats["fast"]["count"] == 0
        assert len(offsets) == n + 1
        assert offsets[0] == 0


class TestSlowPathWhitespace:
    """n > 16, SentencePiece-style ▁ whitespace stripping triggers slow path."""

    def test_sentencepiece_space_triggers_slow_path(self):
        """Per-token decode strips leading space, full decode preserves it."""
        n = 20
        ["word"] * n
        # Full decode has spaces between tokens
        full_text = " ".join(["word"] * n)

        # Progressive: decode([0]) = "word", decode([0,1]) = "word word", etc.
        def mock_decode(ids):
            if len(ids) == 1:
                return "word"
            return " ".join(["word"] * len(ids))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        offsets = token_to_char_offsets(tokenizer, list(range(n)))
        stats = get_path_stats()

        # Concat of per-token = "word" * 20 = 80 chars
        # Full decode = "word word word..." = 99 chars → mismatch → slow path
        assert stats["slow"]["count"] == 1
        assert len(offsets) == n + 1
        assert offsets[-1] == len(full_text)


class TestSlowPathCancellation:
    """Cancellation: length matches but characters differ → string check catches it."""

    def test_length_match_but_content_differs(self):
        """▁ additions cancel byte-merge subtractions in length but not content."""
        n = 20

        # Craft per-token decodes whose total length equals full decode length
        # but with different characters
        per_token_texts = ["ab"] * n  # each 2 chars, total = 40

        # Full decode: same total length (40) but different content
        full_text = "ba" * n  # also 40 chars, but different characters

        # Progressive results for slow path
        progressive_results = {}
        for i in range(1, n + 1):
            progressive_results[i] = "ba" * i

        def mock_decode(ids):
            if len(ids) == 1:
                return per_token_texts[ids[0]]
            return progressive_results.get(len(ids), full_text)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        offsets = token_to_char_offsets(tokenizer, list(range(n)))
        stats = get_path_stats()

        # String check catches mismatch despite equal length
        assert stats["slow"]["count"] == 1
        assert stats["fast"]["count"] == 0
        assert len(offsets) == n + 1
        assert offsets[-1] == len(full_text)


class TestSmallN:
    """n ≤ 16: uses progressive decode directly, never calls per-token decode."""

    def test_small_n_uses_progressive(self):
        """With n=5, should use progressive path, not per-token decode."""
        call_log = []

        def mock_decode(ids):
            call_log.append(("decode", tuple(ids)))
            # Progressive: returns prefix decodes
            chars = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
            return "".join(chars[tid] for tid in ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        offsets = token_to_char_offsets(tokenizer, [0, 1, 2, 3, 4])
        stats = get_path_stats()

        assert stats["small_n"]["count"] == 1
        assert stats["fast"]["count"] == 0
        assert stats["slow"]["count"] == 0
        assert offsets == [0, 1, 2, 3, 4, 5]

        # Progressive path calls decode with growing prefixes: [0], [0,1], [0,1,2], ...
        # NOT with individual token IDs like [0], [1], [2], ... (fast path pattern)
        for _, call_ids in call_log:
            # Each call should be a prefix of the original token_ids
            assert list(call_ids) == [0, 1, 2, 3, 4][: len(call_ids)]

    def test_n_equals_16_uses_progressive(self):
        """Boundary: n=16 should use progressive."""

        def mock_decode(ids):
            return "x" * len(ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        token_to_char_offsets(tokenizer, list(range(16)))
        stats = get_path_stats()
        assert stats["small_n"]["count"] == 1

    def test_n_equals_17_uses_fast_or_slow(self):
        """Boundary: n=17 should NOT use progressive directly."""

        def mock_decode(ids):
            return "x" * len(ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        _reset_path_stats()
        token_to_char_offsets(tokenizer, list(range(17)))
        stats = get_path_stats()
        assert stats["small_n"]["count"] == 0
        assert stats["fast"]["count"] + stats["slow"]["count"] == 1


# ---------------------------------------------------------------------------
# 2b. Real tokenizer tests (GPT-2)
# ---------------------------------------------------------------------------


class TestRealTokenizerGPT2:
    """Tests with real GPT-2 tokenizer."""

    def test_ascii_text_properties(self, gpt2_tokenizer):
        """ASCII text: offsets start at 0, end at len(text), monotonic."""
        text = "The quick brown fox jumps over the lazy dog."
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        assert offsets[0] == 0
        assert offsets[-1] == len(text)
        assert len(offsets) == len(ids) + 1
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]

    def test_cjk_text_properties(self, gpt2_tokenizer):
        """CJK text produces valid monotonic offsets."""
        text = "Hello. 日本語テスト. Goodbye."
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        assert offsets[0] == 0
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))
        assert len(offsets) == len(ids) + 1
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]

    def test_mixed_multibyte(self, gpt2_tokenizer):
        """Mixed multi-byte: combining characters, emoji."""
        text = "Hello 🌍 café naïve"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        assert offsets[0] == 0
        assert len(offsets) == len(ids) + 1
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]

    def test_long_ascii_fast_path(self, gpt2_tokenizer):
        """Long ASCII text (n > 16) exercises the fast path."""
        text = "The quick brown fox " * 20
        ids = gpt2_tokenizer.encode(text)
        assert len(ids) > 16

        _reset_path_stats()
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        stats = get_path_stats()

        assert offsets[0] == 0
        assert offsets[-1] == len(text)
        # ASCII GPT-2 text should hit fast path
        assert stats["fast"]["count"] == 1

    def test_boundary_n16_vs_n17(self, gpt2_tokenizer):
        """n=16 uses progressive, n=17 uses fast/slow path."""
        # Find text that encodes to exactly 16 tokens, then 17
        text = "a " * 20  # each "a " is typically 2 tokens in GPT-2
        ids = gpt2_tokenizer.encode(text)

        ids_16 = ids[:16]
        ids_17 = ids[:17]

        _reset_path_stats()
        token_to_char_offsets(gpt2_tokenizer, ids_16)
        stats_16 = get_path_stats()
        assert stats_16["small_n"]["count"] == 1

        _reset_path_stats()
        token_to_char_offsets(gpt2_tokenizer, ids_17)
        stats_17 = get_path_stats()
        assert stats_17["small_n"]["count"] == 0


# ---------------------------------------------------------------------------
# 2c. Equivalence tests (hypothesis)
# ---------------------------------------------------------------------------


class TestEquivalence:
    """Property-based tests: token_to_char_offsets == _token_to_char_offsets_progressive."""

    @given(
        n=st.integers(min_value=1, max_value=200),
    )
    @settings(
        max_examples=150,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_identical_output_random_ids(self, gpt2_tokenizer, n):
        """Fast path produces identical output to progressive for random token IDs."""
        # Use valid GPT-2 token IDs (0 to 50256)
        import random

        random.seed(n)
        token_ids = [random.randint(0, 50255) for _ in range(n)]

        result_new = token_to_char_offsets(gpt2_tokenizer, token_ids)
        result_ref = _token_to_char_offsets_progressive(gpt2_tokenizer, token_ids)

        assert result_new == result_ref

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                min_codepoint=32,
                max_codepoint=127,
            ),
            min_size=1,
            max_size=500,
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_identical_output_encoded_ascii(self, gpt2_tokenizer, text):
        """Equivalence for encoded ASCII text."""
        token_ids = gpt2_tokenizer.encode(text)
        if not token_ids:
            return

        result_new = token_to_char_offsets(gpt2_tokenizer, token_ids)
        result_ref = _token_to_char_offsets_progressive(gpt2_tokenizer, token_ids)

        assert result_new == result_ref

    def test_offsets_length(self, gpt2_tokenizer):
        """len(offsets) == len(token_ids) + 1 for various inputs."""
        for text in ["Hello world", "日本語", "a" * 100, "🌍🎉"]:
            ids = gpt2_tokenizer.encode(text)
            offsets = token_to_char_offsets(gpt2_tokenizer, ids)
            assert len(offsets) == len(ids) + 1

    def test_monotonically_nondecreasing(self, gpt2_tokenizer):
        """Offsets are monotonically non-decreasing."""
        for text in ["Hello world", "日本語テスト", "a b c d e f g h " * 10]:
            ids = gpt2_tokenizer.encode(text)
            offsets = token_to_char_offsets(gpt2_tokenizer, ids)
            for i in range(1, len(offsets)):
                assert offsets[i] >= offsets[i - 1]

    def test_last_offset_equals_decoded_length(self, gpt2_tokenizer):
        """offsets[-1] == len(tokenizer.decode(token_ids))."""
        for text in ["The quick brown fox.", "café naïve", "Hello 🌍"]:
            ids = gpt2_tokenizer.encode(text)
            offsets = token_to_char_offsets(gpt2_tokenizer, ids)
            assert offsets[-1] == len(gpt2_tokenizer.decode(ids))


# ---------------------------------------------------------------------------
# 2d. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input(self):
        offsets = token_to_char_offsets(MagicMock(), [])
        assert offsets == [0]

    def test_single_token(self):
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: "abc"
        offsets = token_to_char_offsets(tokenizer, [42])
        assert offsets == [0, 3]

    def test_all_same_token_x20(self, gpt2_tokenizer):
        """20 copies of the same token."""
        # Use a common token like "the" (token 1169 in GPT-2)
        token_id = gpt2_tokenizer.encode("the")[0]
        ids = [token_id] * 20
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        ref = _token_to_char_offsets_progressive(gpt2_tokenizer, ids)
        assert offsets == ref
        assert len(offsets) == 21

    def test_special_tokens_bos_eos(self, gpt2_tokenizer):
        """BOS/EOS tokens (GPT-2 uses <|endoftext|> = 50256)."""
        eos = gpt2_tokenizer.eos_token_id
        ids = [eos]
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        assert offsets[0] == 0
        assert len(offsets) == 2

    def test_token_decodes_to_empty_string(self):
        """Token that decodes to empty string."""
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: {
            (0,): "",
            (0, 1): "hello",
        }.get(tuple(ids), "")
        offsets = token_to_char_offsets(tokenizer, [0, 1])
        assert offsets == [0, 0, 5]

    def test_prefix_dependent_whitespace(self, gpt2_tokenizer):
        """Prefix-dependent whitespace: ' hello' followed by ' world'."""
        text = " hello world"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        ref = _token_to_char_offsets_progressive(gpt2_tokenizer, ids)
        assert offsets == ref

    def test_suffix_dependent_utf8(self, gpt2_tokenizer):
        """Suffix-dependent: partial UTF-8 byte tokens."""
        # Encode text with multi-byte chars to get byte-level tokens
        text = "日本語"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        ref = _token_to_char_offsets_progressive(gpt2_tokenizer, ids)
        assert offsets == ref


# ---------------------------------------------------------------------------
# 2e. Multi-byte byte-fallback monotonicity tests
# ---------------------------------------------------------------------------


class TestByteBackoffMonotonicity:
    """Tests for the merge-aware offset correction in _token_to_char_offsets_progressive.

    When a tokenizer splits multi-byte Unicode characters into individual
    byte-fallback tokens (e.g., Gemma's handling of Kharoshthi U+10A00),
    progressive prefix decoding can produce non-monotonic char offsets:
    incomplete byte sequences decode as N replacement chars, but when the
    final byte completes the character, the prefix shrinks to 1 char.

    The fix detects the merge and retroactively resets preceding offsets for
    the byte-fallback group, instead of clamping with max() which inflates
    offsets[-1].
    """

    def test_shrinking_prefix_mock(self):
        """Simulates 4-byte char: 3 replacement chars → 1 real char."""
        # Simulate a tokenizer that splits a 4-byte char into 4 byte tokens.
        # decode([:1]) = "�" (1 char)
        # decode([:2]) = "��" (2 chars)
        # decode([:3]) = "���" (3 chars)
        # decode([:4]) = "𐨅" (1 char) — SHRINKS
        progressive_results = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\ufffd\ufffd\ufffd",
            4: "\U00010a05",  # Kharoshthi char — 1 char
        }

        def mock_decode(ids):
            return progressive_results.get(len(ids), "x" * len(ids))

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])

        # Merge correction: intermediate bytes reset to base, final = actual len
        assert offsets == [0, 0, 0, 0, 1]
        assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))

    def test_multiple_shrinks_mock(self):
        """Multiple multi-byte chars in sequence: each shrinks independently."""
        # Two 3-byte chars back to back:
        # tokens 0-2: first char, tokens 3-5: second char
        # decode([:1])="�", decode([:2])="��", decode([:3])="あ" (shrink)
        # decode([:4])="あ�", decode([:5])="あ��", decode([:6])="あい" (shrink)
        progressive_results = {
            1: "\ufffd",  # 1 char
            2: "\ufffd\ufffd",  # 2 chars
            3: "あ",  # 1 char (shrink!)
            4: "あ\ufffd",  # 2 chars
            5: "あ\ufffd\ufffd",  # 3 chars
            6: "あい",  # 2 chars (shrink!)
        }

        def mock_decode(ids):
            return progressive_results.get(len(ids), "")

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))

        # Merge correction: each group's intermediate bytes reset to base
        assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))
        assert offsets == [0, 0, 0, 1, 1, 1, 2]

    def test_no_clamp_needed_for_ascii_mock(self):
        """ASCII tokens: no clamping needed, offsets are naturally monotonic."""

        def mock_decode(ids):
            return "a" * len(ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(5)))
        assert offsets == [0, 1, 2, 3, 4, 5]

    def test_single_byte_token_no_shrink(self):
        """Single token that doesn't shrink."""

        def mock_decode(ids):
            return "x"

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, [42])
        assert offsets == [0, 1]

    def test_monotonicity_with_real_gpt2_emoji(self, gpt2_tokenizer):
        """GPT-2 emoji encoding: byte tokens that merge into emoji chars."""
        text = "Hello 🌍🎉 world"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        assert offsets[0] == 0
        assert len(offsets) == len(ids) + 1
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1], (
                f"Non-monotonic at {i}: {offsets[i]} < {offsets[i - 1]}"
            )

    def test_monotonicity_with_real_gpt2_mixed_scripts(self, gpt2_tokenizer):
        """Mixed scripts with byte fallback: Arabic, Thai, Devanagari."""
        texts = [
            "Hello مرحبا world",  # Arabic
            "สวัสดี test ทดสอบ",  # Thai
            "नमस्ते test अनुवाद",  # Devanagari
            "Hello 世界 🌍 café naïve ñ",  # Mixed
        ]
        for text in texts:
            ids = gpt2_tokenizer.encode(text)
            offsets = token_to_char_offsets(gpt2_tokenizer, ids)

            assert offsets[0] == 0
            assert len(offsets) == len(ids) + 1
            for i in range(1, len(offsets)):
                assert offsets[i] >= offsets[i - 1], (
                    f"Non-monotonic at token {i} in {text!r}: "
                    f"{offsets[i]} < {offsets[i - 1]}"
                )

    def test_token_to_char_offsets_monotonic_via_slow_path(self):
        """Ensure the public API clamps via slow path for multi-byte tokens."""
        n = 20

        # Simulate tokenizer where per-token decode differs from full decode
        # AND the progressive decode shrinks at token 10
        per_token = ["a"] * n
        per_token[10] = "\ufffd"  # This token is a byte-fallback

        progressive = {}
        for i in range(1, n + 1):
            if i <= 10:
                progressive[i] = "a" * i
            elif i == 11:
                # Shrink: the byte at position 10 completes a 2-byte char
                progressive[i] = "a" * 9 + "ñ" + "a" * 0  # shorter than 11
            else:
                progressive[i] = "a" * 9 + "ñ" + "a" * (i - 11)

        full_text = progressive[n]

        def mock_decode(ids):
            if len(ids) == 1:
                return per_token[ids[0]]
            return progressive.get(len(ids), full_text)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = token_to_char_offsets(tokenizer, list(range(n)))

        assert len(offsets) == n + 1
        assert offsets[0] == 0
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1], (
                f"Non-monotonic at {i}: {offsets[i]} < {offsets[i - 1]}"
            )

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z", "S"),
                min_codepoint=32,
                max_codepoint=0x1FFFF,  # Include supplementary planes
            ),
            min_size=1,
            max_size=200,
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.differing_executors],
    )
    def test_monotonicity_hypothesis_all_unicode(self, gpt2_tokenizer, text):
        """Property test: offsets are always monotonic for arbitrary Unicode."""
        ids = gpt2_tokenizer.encode(text)
        if not ids:
            return
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        assert offsets[0] == 0
        assert len(offsets) == len(ids) + 1
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1], (
                f"Non-monotonic at token {i}: {offsets[i]} < {offsets[i - 1]}"
            )


# ---------------------------------------------------------------------------
# Path-tracking instrumentation tests
# ---------------------------------------------------------------------------


class TestPathStats:
    def test_get_path_stats_returns_all_paths(self):
        _reset_path_stats()
        stats = get_path_stats()
        assert set(stats.keys()) == {"fast", "slow", "small_n"}

    def test_get_path_stats_resets_counters(self):
        _reset_path_stats()
        _path_stats["fast"].append(1.0)
        _path_stats["slow"].append(2.0)
        stats = get_path_stats()
        assert stats["fast"]["count"] == 1
        assert stats["slow"]["count"] == 1

        # Second call should be empty
        stats2 = get_path_stats()
        assert stats2["fast"]["count"] == 0
        assert stats2["slow"]["count"] == 0

    def test_stats_mean_and_total(self):
        _reset_path_stats()
        _path_stats["fast"].extend([1.0, 3.0])
        stats = get_path_stats()
        assert stats["fast"]["count"] == 2
        assert stats["fast"]["mean_ms"] == pytest.approx(2.0)
        assert stats["fast"]["total_ms"] == pytest.approx(4.0)

    def test_empty_path_stats(self):
        _reset_path_stats()
        stats = get_path_stats()
        for path in ["fast", "slow", "small_n"]:
            assert stats[path]["count"] == 0
            assert stats[path]["mean_ms"] == 0.0
            assert stats[path]["total_ms"] == 0.0
