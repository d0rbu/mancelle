"""Supplementary Unicode and integration tests for token_to_char_offsets.

Extends test_segmentation_char_offsets.py with:
- Diverse Unicode edge cases (supplementary planes, ZWJ sequences, RTL)
- Integration tests verifying segment boundary consistency with char offsets
- Gemma-style byte-fallback simulation with replacement char coalescence
"""

from unittest.mock import MagicMock

import pytest
from transformers import GPT2Tokenizer

from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import (
    SegmentMode,
    _token_to_char_offsets_progressive,
    segment_tokens,
    token_to_char_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


def _assert_offset_invariants(offsets, token_ids):
    """Check universal invariants on a char-offset list."""
    assert offsets[0] == 0, f"offsets[0] must be 0, got {offsets[0]}"
    assert len(offsets) == len(token_ids) + 1, (
        f"len(offsets) must be len(ids)+1={len(token_ids) + 1}, got {len(offsets)}"
    )
    for i in range(1, len(offsets)):
        assert offsets[i] >= offsets[i - 1], (
            f"Non-monotonic at index {i}: {offsets[i]} < {offsets[i - 1]}"
        )


# ---------------------------------------------------------------------------
# 1. Emoji edge cases (4-byte UTF-8)
# ---------------------------------------------------------------------------


class TestEmojiEdgeCases:
    """GPT-2 splits emoji into byte-level tokens; verify offset sanity."""

    def test_single_emoji(self, gpt2_tokenizer):
        text = "Hello \U0001f30d"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_consecutive_emoji(self, gpt2_tokenizer):
        text = "\U0001f30d\U0001f389\U0001f38a\U0001f388"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_emoji_surrounded_by_text(self, gpt2_tokenizer):
        text = "start \U0001f30d end"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_zwj_emoji_sequence(self, gpt2_tokenizer):
        """Zero-width joiner sequences (family emoji, skin tones)."""
        # Woman technologist: Woman + ZWJ + Laptop
        text = "Hi \U0001f469\u200d\U0001f4bb there"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_flag_emoji_regional_indicators(self, gpt2_tokenizer):
        """Flag emoji are two regional indicator symbols."""
        # US flag: U+1F1FA U+1F1F8
        text = "Flag: \U0001f1fa\U0001f1f8 test"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_keycap_sequence(self, gpt2_tokenizer):
        """Keycap sequences: digit + U+FE0F + U+20E3."""
        text = "Press 1\ufe0f\u20e3 now"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)


# ---------------------------------------------------------------------------
# 2. CJK characters (3-byte UTF-8)
# ---------------------------------------------------------------------------


class TestCJKCharacters:
    def test_japanese(self, gpt2_tokenizer):
        text = "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_chinese(self, gpt2_tokenizer):
        text = "\u4e2d\u6587\u6d4b\u8bd5"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_korean(self, gpt2_tokenizer):
        text = "\ud55c\uad6d\uc5b4"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_mixed_ascii_cjk(self, gpt2_tokenizer):
        text = "Hello \u65e5\u672c world \u4e2d\u6587 test"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_cjk_long_text_forces_nonsmall_path(self, gpt2_tokenizer):
        """CJK text long enough (n > 16) to exercise fast/slow path decision."""
        # Each CJK char typically becomes ~1.5 byte tokens in GPT-2;
        # use enough chars to guarantee > 16 tokens
        text = "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8\u4e2d\u6587\u6d4b\u8bd5\u6771\u4eac\u5927\u962a"
        ids = gpt2_tokenizer.encode(text)
        assert len(ids) > 16, f"Expected > 16 tokens, got {len(ids)}"
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)


# ---------------------------------------------------------------------------
# 3. Accented / Latin-extended characters (2-byte UTF-8)
# ---------------------------------------------------------------------------


class TestAccentedCharacters:
    def test_accented_latin(self, gpt2_tokenizer):
        text = "caf\u00e9 na\u00efve r\u00e9sum\u00e9"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_spanish_tilde(self, gpt2_tokenizer):
        text = "\u00d1o\u00f1o"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_german_umlauts(self, gpt2_tokenizer):
        text = "\u00dcber stra\u00dfe Gr\u00fc\u00dfe"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_accented_long_text_fast_path(self, gpt2_tokenizer):
        """GPT-2 BPE handles common accented chars natively; verify fast path
        is plausible for largely-Latin text with a few accents."""
        text = "The caf\u00e9 served a cr\u00e8me br\u00fbl\u00e9e on a lovely morning."
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))


# ---------------------------------------------------------------------------
# 4. Byte-fallback monotonicity (mock simulations)
# ---------------------------------------------------------------------------


class TestByteFallbackMonotonicity:
    """Simulate tokenizers that produce shrinking prefixes."""

    def test_4byte_char_shrink(self):
        """4 byte-tokens for one char: decode shrinks at completion."""
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\ufffd\ufffd\ufffd",
            4: "\U0001d573",  # mathematical bold fraktur H (1 char)
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1, 2, 3])
        assert offsets == [0, 0, 0, 0, 1], f"Expected [0,0,0,0,1], got {offsets}"

    def test_two_3byte_chars_sequential_shrinks(self):
        """Two 3-byte CJK chars: each byte-triplet shrinks independently."""
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\u3042",  # 'a' hiragana (1 char, shrink from 2)
            4: "\u3042\ufffd",
            5: "\u3042\ufffd\ufffd",
            6: "\u3042\u3044",  # 'i' hiragana (2 chars, shrink from 3)
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))
        # Merge correction: each group's intermediate bytes reset to base
        assert offsets == [0, 0, 0, 1, 1, 1, 2]
        assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))

    def test_three_4byte_chars_triple_shrink(self):
        """Three consecutive 4-byte chars, each shrinks from 3 replacement chars."""

        # Tokens: 0-3 -> char1, 4-7 -> char2, 8-11 -> char3
        def mock_decode(ids):
            n = len(ids)
            result = ""
            full_chars = ["\U0001f600", "\U0001f601", "\U0001f602"]
            pos = 0
            while pos < n:
                char_idx = pos // 4
                byte_within = n - char_idx * 4
                if char_idx < len(full_chars) and byte_within >= 4:
                    # This char is complete
                    result += full_chars[char_idx]
                    pos = (char_idx + 1) * 4
                else:
                    # Incomplete: replacement chars for remaining bytes
                    remaining = n - pos
                    result += "\ufffd" * remaining
                    break
            return result

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(12)))
        _assert_offset_invariants(offsets, list(range(12)))

    def test_mixed_ascii_and_shrinking_byte_tokens(self):
        """ASCII tokens interspersed with byte-fallback tokens."""
        # "Hi" + 3-byte char + "!"
        # tokens 0='H', 1='i', 2-4=byte fallback, 5='!'
        progressive = {
            1: "H",
            2: "Hi",
            3: "Hi\ufffd",
            4: "Hi\ufffd\ufffd",
            5: "Hi\u00e9",  # shrink: 2 replacement -> 1 accented char
            6: "Hi\u00e9!",
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(6)))
        _assert_offset_invariants(offsets, list(range(6)))
        # After merge at token 4: prefix is "Hié" (3 chars), so offsets[5] = 3
        # Token 5: prefix is "Hié!" (4 chars), so offsets[6] = 4
        assert offsets[5] == 3  # corrected: actual prefix length, not clamped

    def test_real_gpt2_mathematical_script(self, gpt2_tokenizer):
        """Mathematical script chars in supplementary plane with real GPT-2."""
        text = "Hello \U0001d573ello"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)


# ---------------------------------------------------------------------------
# 5. Segment boundary consistency (integration)
# ---------------------------------------------------------------------------


class TestSegmentBoundaryConsistency:
    """Verify that segment char boundaries are consistent with token offsets."""

    def _get_segment_char_ranges(self, tokenizer, token_ids, segment_size):
        """Compute per-segment (char_start, char_end) from offsets + segments."""
        offsets = token_to_char_offsets(tokenizer, token_ids)
        segments = segment_tokens(
            token_ids,
            mode=SegmentMode.fixed_tokens,
            segment_size=segment_size,
        )
        char_ranges = []
        for tok_start, tok_end in segments:
            char_ranges.append((offsets[tok_start], offsets[tok_end]))
        return char_ranges, offsets

    def test_ascii_segments_tile_without_gaps(self, gpt2_tokenizer):
        text = "The quick brown fox jumps over the lazy dog and more words here."
        ids = gpt2_tokenizer.encode(text)
        char_ranges, offsets = self._get_segment_char_ranges(
            gpt2_tokenizer,
            ids,
            segment_size=4,
        )

        # No gaps between consecutive segments
        for i in range(1, len(char_ranges)):
            assert char_ranges[i][0] == char_ranges[i - 1][1], (
                f"Gap at segment {i}: end={char_ranges[i - 1][1]}, "
                f"start={char_ranges[i][0]}"
            )

        # First segment starts at char 0
        assert char_ranges[0][0] == 0

        # Last segment ends at decoded text length
        assert char_ranges[-1][1] == offsets[-1]

    def test_cjk_segments_tile_without_gaps(self, gpt2_tokenizer):
        text = "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8\u4e2d\u6587\u6d4b\u8bd5"
        ids = gpt2_tokenizer.encode(text)
        char_ranges, offsets = self._get_segment_char_ranges(
            gpt2_tokenizer,
            ids,
            segment_size=8,
        )

        for i in range(1, len(char_ranges)):
            assert char_ranges[i][0] == char_ranges[i - 1][1]
        assert char_ranges[0][0] == 0
        assert char_ranges[-1][1] == offsets[-1]

    def test_mixed_ascii_cjk_emoji_segments(self, gpt2_tokenizer):
        """Mixed script text: verify segment tiling with no gaps/overlaps."""
        text = "Hello \u4e16\u754c \U0001f30d caf\u00e9 \u65e5\u672c\u8a9e test"
        ids = gpt2_tokenizer.encode(text)
        char_ranges, offsets = self._get_segment_char_ranges(
            gpt2_tokenizer,
            ids,
            segment_size=6,
        )

        # Tiling: no gaps
        for i in range(1, len(char_ranges)):
            assert char_ranges[i][0] == char_ranges[i - 1][1], (
                f"Gap at segment {i}: prev_end={char_ranges[i - 1][1]}, "
                f"cur_start={char_ranges[i][0]}"
            )

        # Non-overlapping: each range is non-negative width
        for start, end in char_ranges:
            assert end >= start, f"Negative width segment: [{start}, {end})"

        # Complete coverage
        assert char_ranges[0][0] == 0
        assert char_ranges[-1][1] == offsets[-1]

    def test_segment_size_1_gives_per_token_ranges(self, gpt2_tokenizer):
        """segment_size=1: each segment is one token; char ranges are per-token spans."""
        text = "Hello world"
        ids = gpt2_tokenizer.encode(text)
        char_ranges, offsets = self._get_segment_char_ranges(
            gpt2_tokenizer,
            ids,
            segment_size=1,
        )

        assert len(char_ranges) == len(ids)
        for i, (start, end) in enumerate(char_ranges):
            assert start == offsets[i]
            assert end == offsets[i + 1]

    def test_segment_size_larger_than_input(self, gpt2_tokenizer):
        """Single segment covering the entire token sequence."""
        text = "Short"
        ids = gpt2_tokenizer.encode(text)
        char_ranges, offsets = self._get_segment_char_ranges(
            gpt2_tokenizer,
            ids,
            segment_size=1024,
        )

        assert len(char_ranges) == 1
        assert char_ranges[0] == (0, offsets[-1])


# ---------------------------------------------------------------------------
# 6. Mixed scripts in long text (n > 16)
# ---------------------------------------------------------------------------


class TestMixedScriptsLong:
    """Long multi-script texts that exceed the small-n threshold."""

    def test_arabic_mixed(self, gpt2_tokenizer):
        text = "Hello \u0645\u0631\u062d\u0628\u0627 world"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_thai(self, gpt2_tokenizer):
        text = (
            "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 test \u0e17\u0e14\u0e2a\u0e2d\u0e1a"
        )
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_devanagari(self, gpt2_tokenizer):
        text = "\u0928\u092e\u0938\u094d\u0924\u0947 test \u0905\u0928\u0941\u0935\u093e\u0926"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_hebrew(self, gpt2_tokenizer):
        text = "\u05e9\u05dc\u05d5\u05dd world \u05e2\u05d5\u05dc\u05dd"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_long_mixed_exceeds_threshold(self, gpt2_tokenizer):
        """Combine many scripts into a text guaranteed to produce > 16 tokens."""
        text = (
            "English caf\u00e9 "
            "\u65e5\u672c\u8a9e "
            "\u0645\u0631\u062d\u0628\u0627 "
            "\U0001f30d\U0001f389 "
            "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 "
            "\u0928\u092e\u0938\u094d\u0924\u0947 "
            "\ud55c\uad6d\uc5b4 "
            "end"
        )
        ids = gpt2_tokenizer.encode(text)
        assert len(ids) > 16, f"Expected > 16 tokens, got {len(ids)}"
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_equivalence_with_progressive_for_long_multiscript(self, gpt2_tokenizer):
        """For long multi-script text, public API must match progressive reference."""
        text = (
            "The quick brown fox \u65e5\u672c\u8a9e "
            "\U0001f30d caf\u00e9 r\u00e9sum\u00e9 "
            "\u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd "
            "extra padding words for length"
        )
        ids = gpt2_tokenizer.encode(text)
        assert len(ids) > 16

        offsets_api = token_to_char_offsets(gpt2_tokenizer, ids)
        offsets_ref = _token_to_char_offsets_progressive(gpt2_tokenizer, ids)
        assert offsets_api == offsets_ref


# ---------------------------------------------------------------------------
# 7. Gemma byte-fallback simulation (mock)
# ---------------------------------------------------------------------------


class TestGemmaByteFallbackSimulation:
    """Simulate Gemma-style tokenizer where byte tokens produce replacement
    chars individually but combine into valid characters together."""

    def test_replacement_chars_coalesce_to_valid_char(self):
        """Individual byte tokens -> U+FFFD, but together -> valid char."""
        # Simulate "A" + 3-byte CJK char + "B"
        # Token 0: 'A', tokens 1-3: byte fallback for CJK, token 4: 'B'
        progressive = {
            1: "A",
            2: "A\ufffd",
            3: "A\ufffd\ufffd",
            4: "A\u6d4b",  # byte fallback complete -> CJK char
            5: "A\u6d4bB",
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(5)))
        _assert_offset_invariants(offsets, list(range(5)))
        # After merge correction: offsets[4] = len("A\u6d4b") = 2
        assert offsets[4] == 2

    def test_multiple_coalescence_events(self):
        """Multiple groups of byte-fallback tokens coalescing."""
        # Token layout: [byte1, byte2, byte3, 'X', byte1, byte2, byte3, byte4]
        # First group (0-2) -> 3-byte char; second group (4-7) -> 4-byte char
        progressive = {
            1: "\ufffd",
            2: "\ufffd\ufffd",
            3: "\u3042",  # hiragana 'a'
            4: "\u3042X",
            5: "\u3042X\ufffd",
            6: "\u3042X\ufffd\ufffd",
            7: "\u3042X\ufffd\ufffd\ufffd",
            8: "\u3042X\U0001f600",  # grinning face
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(8)))
        _assert_offset_invariants(offsets, list(range(8)))

    def test_coalescence_with_segment_boundaries(self):
        """Verify segment boundaries are valid even with byte-fallback coalescence."""
        # 8 tokens: 2 ASCII + 3-byte char + 3 ASCII
        # progressive decode simulates the coalescence
        progressive = {
            1: "H",
            2: "Hi",
            3: "Hi\ufffd",
            4: "Hi\ufffd\ufffd",
            5: "Hi\u00e9",  # coalescence: 2 FFFD -> 1 char
            6: "Hi\u00e9 o",
            7: "Hi\u00e9 ok",
            8: "Hi\u00e9 ok!",
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        token_ids = list(range(8))
        offsets = _token_to_char_offsets_progressive(tokenizer, token_ids)
        _assert_offset_invariants(offsets, token_ids)

        # Simulate segment_tokens with segment_size=4
        segments = [(0, 4), (4, 8)]
        for tok_start, tok_end in segments:
            char_start = offsets[tok_start]
            char_end = offsets[tok_end]
            assert char_end >= char_start, (
                f"Negative segment: [{char_start}, {char_end})"
            )

        # Tiling
        assert offsets[segments[1][0]] == offsets[segments[0][1]]

    def test_all_replacement_chars_never_coalesce(self):
        """Edge case: byte tokens that never form valid characters."""

        # Every prefix is just more replacement chars
        def mock_decode(ids):
            return "\ufffd" * len(ids)

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = mock_decode

        offsets = _token_to_char_offsets_progressive(tokenizer, list(range(5)))
        assert offsets == [0, 1, 2, 3, 4, 5]

    def test_immediate_coalescence_at_token_2(self):
        """2-byte char: replacement char at token 1, valid char at token 2."""
        progressive = {
            1: "\ufffd",
            2: "\u00e9",  # 1 char (shrink from 1)
        }

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids: progressive[len(ids)]

        offsets = _token_to_char_offsets_progressive(tokenizer, [0, 1])
        # new_len=1 at step 0 (append 1), new_len=1 at step 1 (no merge, append 1)
        assert offsets == [0, 1, 1]


# ---------------------------------------------------------------------------
# 8. Supplementary plane and special Unicode (additional coverage)
# ---------------------------------------------------------------------------


class TestSupplementaryPlaneUnicode:
    """Characters outside the Basic Multilingual Plane (> U+FFFF)."""

    def test_musical_symbols(self, gpt2_tokenizer):
        """Musical symbols are in U+1D100 range."""
        text = "Music: \U0001d11e \U0001d122"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_ancient_scripts(self, gpt2_tokenizer):
        """Egyptian hieroglyphs (U+13000 range)."""
        text = "Ancient: \U00013000\U00013001"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_math_symbols_supplementary(self, gpt2_tokenizer):
        """Mathematical alphanumeric symbols (U+1D400 range)."""
        text = "\U0001d400\U0001d401\U0001d402 = ABC bold"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_mixed_bmp_and_supplementary(self, gpt2_tokenizer):
        """Mix of BMP chars (CJK) and supplementary plane chars (emoji)."""
        text = "\u4e2d\U0001f30d\u6587\U0001f389test"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)
        assert offsets[-1] == len(gpt2_tokenizer.decode(ids))

    def test_combining_characters(self, gpt2_tokenizer):
        """Combining diacritical marks (e.g., e + combining acute)."""
        # 'e' + combining acute accent = visual 'e with accent' but 2 codepoints
        text = "caf\u0065\u0301 test"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)

    def test_surrogate_free_text(self, gpt2_tokenizer):
        """Verify handling of text that would use surrogate pairs in UTF-16."""
        # Deseret small letter long I (U+10428) - 4 bytes in UTF-8
        text = "Deseret: \U00010428\U00010429\U0001042a"
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        _assert_offset_invariants(offsets, ids)


# ---------------------------------------------------------------------------
# 9. Integration: char offsets + segment_tokens end-to-end
# ---------------------------------------------------------------------------


class TestCharOffsetsSegmentIntegration:
    """End-to-end: tokenize -> char offsets -> segment -> verify consistency."""

    @pytest.mark.parametrize("segment_size", [4, 8, 16, 32])
    def test_various_segment_sizes(self, gpt2_tokenizer, segment_size):
        text = "The quick brown fox jumps over the lazy dog."
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        segments = segment_tokens(
            ids,
            mode=SegmentMode.fixed_tokens,
            segment_size=segment_size,
        )

        # Every segment has valid char range
        for tok_start, tok_end in segments:
            assert offsets[tok_end] >= offsets[tok_start]

        # Complete coverage
        assert offsets[segments[0][0]] == 0
        assert offsets[segments[-1][1]] == offsets[-1]

        # No gaps
        for i in range(1, len(segments)):
            prev_char_end = offsets[segments[i - 1][1]]
            cur_char_start = offsets[segments[i][0]]
            assert cur_char_start == prev_char_end

    def test_multibyte_segment_char_text_extraction(self, gpt2_tokenizer):
        """Char ranges from segments can correctly index into decoded text."""
        text = "Hello world test foo bar"
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        segments = segment_tokens(
            ids,
            mode=SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(2),
        )

        # Each segment's char range extracts a valid substring
        full_reconstructed = ""
        for tok_start, tok_end in segments:
            char_start = offsets[tok_start]
            char_end = offsets[tok_end]
            segment_text = decoded[char_start:char_end]
            full_reconstructed += segment_text

        assert full_reconstructed == decoded

    @pytest.mark.parametrize(
        "text",
        [
            "Hello \U0001f30d world \u65e5\u672c end",
            "\u4e2d\u6587 test \u0645\u0631\u062d\u0628\u0627",
            "caf\u00e9 \U0001d573 \u00d1o\u00f1o \U0001f389",
        ],
    )
    def test_multiscript_segment_tiling(self, gpt2_tokenizer, text):
        """Segment char ranges tile the full decoded text for multi-script inputs."""
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)
        segments = segment_tokens(
            ids,
            mode=SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(4),
        )

        reconstructed = ""
        for tok_start, tok_end in segments:
            reconstructed += decoded[offsets[tok_start] : offsets[tok_end]]

        assert reconstructed == decoded

    def test_token_offset_with_offset_parameter(self, gpt2_tokenizer):
        """segment_tokens with token_offset: char ranges still tile correctly
        over the offset region."""
        text = "Prompt tokens here. Response tokens here and more."
        ids = gpt2_tokenizer.encode(text)
        offsets = token_to_char_offsets(gpt2_tokenizer, ids)

        # Simulate SFT: segmentation starts at token 5
        token_offset = 5
        segments = segment_tokens(
            ids,
            mode=SegmentMode.fixed_tokens,
            segment_size=PositiveInt.parse(4),
            token_offset=NonNegativeInt.parse(token_offset),
        )

        # First segment starts at the offset token
        assert segments[0][0] == token_offset

        # Last segment covers up to the end
        assert segments[-1][1] == len(ids)

        # No gaps in the offset region
        for i in range(1, len(segments)):
            prev_char_end = offsets[segments[i - 1][1]]
            cur_char_start = offsets[segments[i][0]]
            assert cur_char_start == prev_char_end
