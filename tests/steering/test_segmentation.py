"""Tests for steering.segmentation module.

Run with:
    uv run pytest tests/test_segmentation.py -v
"""

import math

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import (
    SegmentMode,
    assert_turn_roundtrip,
    normalize_roundtrip_text,
    segment_document,
    segment_mode_from_metadata,
    segment_tokens,
    turn_segments_for_document,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


class _CharacterTokenizer:
    def decode(self, token_ids):
        return "".join(chr(96 + token_id) for token_id in token_ids)


class _BosCharacterTokenizer:
    bos_token_id = 0

    def decode(self, token_ids):
        return "".join("<bos>" if token_id == 0 else chr(96 + token_id) for token_id in token_ids)


def _segment_tokens(
    token_ids: list[int],
    mode: SegmentMode = SegmentMode.fixed_tokens,
    segment_size: int = 128,
    token_offset: int = 0,
) -> list[tuple[int, int]]:
    """Refine test inputs before exercising the production boundary."""
    return segment_tokens(
        token_ids,
        mode,
        PositiveInt.parse(segment_size),
        NonNegativeInt.parse(token_offset),
    )


class TestSegmentMode:
    def test_expected_modes_exist(self):
        assert SegmentMode.fixed_tokens.value == "fixed_tokens"
        assert SegmentMode.turn_delimited_fixed_tokens.value == "turn_delimited_fixed_tokens"
        assert len(SegmentMode) == 2

    def test_empty_raises(self):
        with pytest.raises(AssertionError, match="empty token list"):
            _segment_tokens([], SegmentMode.fixed_tokens)

    def test_round_trip(self):
        for mode in SegmentMode:
            assert SegmentMode(mode.value) is mode

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SegmentMode("nonexistent")

    def test_removed_modes_raise_value_error(self):
        """Removed modes (sentence, newline) cannot be constructed."""
        with pytest.raises(ValueError):
            SegmentMode("sentence")
        with pytest.raises(ValueError):
            SegmentMode("newline")


def test_segment_document_rejects_mismatched_precomputed_char_offsets():
    with pytest.raises(ValueError, match="char_offsets.*one boundary per token"):
        segment_document(
            [1, 2, 3],
            _CharacterTokenizer(),
            SegmentMode.fixed_tokens,
            PositiveInt.parse(2),
            char_offsets=[0, 1, 2],
        )


def test_segment_document_precomputed_char_offsets_match_recomputation():
    args = (
        [1, 2, 3],
        _CharacterTokenizer(),
        SegmentMode.fixed_tokens,
        PositiveInt.parse(2),
    )
    assert segment_document(*args, char_offsets=[0, 1, 2, 3]) == segment_document(*args)


class TestAssertTurnRoundtrip:
    def test_exact_prefix_returns_none(self):
        assert assert_turn_roundtrip(
            "<bos>abc", "abc", bos_char_len=5, boundaries=[]
        ) is None

    def test_equal_length_mismatch_returns_decoded_coordinate(self):
        assert assert_turn_roundtrip(
            "<bos>abX", "abc", bos_char_len=5, boundaries=[]
        ) == 7

    def test_tail_window_edge_is_inclusive(self):
        text = "a" * 40
        decoded = "<bos>" + "a" * 20 + "!" + "a" * 19
        assert assert_turn_roundtrip(
            decoded, text, bos_char_len=5, boundaries=[9], window=20
        ) == 25

    def test_beyond_tail_window_raises(self):
        text = "a" * 40
        decoded = "<bos>" + "a" * 19 + "!" + "a" * 20
        with pytest.raises(ValueError, match=r"tail_delta=21, window=20"):
            assert_turn_roundtrip(
                decoded, text, bos_char_len=5, boundaries=[9], window=20
            )

    def test_boundary_at_mismatch_raises(self):
        with pytest.raises(
            ValueError,
            match=r"tail_delta=12, window=20, retained_boundaries=\[13\]",
        ):
            assert_turn_roundtrip(
                "<bos>aaaaaaaa!aaaaaaaaaaa",
                "a" * 20,
                bos_char_len=5,
                boundaries=[13],
                window=20,
            )

    def test_decode_overrun_returns_first_extra_character(self):
        assert assert_turn_roundtrip(
            "<bos>abcXY", "abc", bos_char_len=5, boundaries=[]
        ) == 8

    def test_long_decode_overrun_raises(self):
        with pytest.raises(ValueError, match="tail_delta=40"):
            assert_turn_roundtrip(
                "<bos>hi" + "Z" * 40,
                "hi",
                bos_char_len=5,
                boundaries=[],
            )

    def test_decode_overrun_at_boundary_raises(self):
        with pytest.raises(ValueError, match=r"retained_boundaries=\[8\]"):
            assert_turn_roundtrip(
                "<bos>abcdef",
                "abc",
                bos_char_len=5,
                boundaries=[8],
            )

    def test_bos_length_precondition(self):
        """Pin the defensive branch even though production callers derive it."""
        with pytest.raises(
            ValueError,
            match="BOS character length 9 exceeds decoded length 5",
        ):
            assert_turn_roundtrip("<bos>", "abc", bos_char_len=9, boundaries=[])

    # -- sentencepiece word-boundary marker (U+2581) in the SOURCE text ------
    # Sentencepiece decodes its marker as a space, so a source that literally
    # contains it (SmolLM rows about BPE) never round-trips byte-for-byte; the
    # substitution is width-preserving so offsets stay aligned (Engaging 3B
    # campaign, shards 6/17 of standard_steered, 2026-09-05).

    def test_source_marker_decoded_as_space_is_not_a_mismatch(self):
        text = "replace the '▁' marker with a space and keep going for a while"
        decoded = "<bos>" + text.replace("▁", " ")
        assert decoded != "<bos>" + text
        assert assert_turn_roundtrip(
            decoded, text, bos_char_len=5, boundaries=[30, 50], window=20
        ) is None

    def test_marker_is_symmetric_and_only_that_character_is_normalized(self):
        assert normalize_roundtrip_text("a▁b▁") == "a b "
        assert normalize_roundtrip_text("a b") == "a b"
        assert normalize_roundtrip_text(" 　") == " 　", (
            "no other whitespace-like character is normalized"
        )
        # A literal marker on the decoded side is treated the same way.
        assert assert_turn_roundtrip(
            "<bos>x▁y", "x y", bos_char_len=5, boundaries=[]
        ) is None

    def test_marker_does_not_hide_a_later_genuine_mismatch(self):
        text = "aaaa▁" + "a" * 30
        decoded = "<bos>aaaa " + "a" * 9 + "!" + "a" * 20
        with pytest.raises(ValueError, match=r"index 19; decoded_len=40, tail_delta=21"):
            assert_turn_roundtrip(decoded, text, bos_char_len=5, boundaries=[], window=20)

    def test_marker_does_not_hide_a_boundary_at_a_genuine_mismatch(self):
        text = "aa▁" + "a" * 17
        decoded = "<bos>aa aaaaaa!aaaaaaaaaaa"  # '!' is decoded index 5 + 9 = 14
        assert assert_turn_roundtrip(decoded, text, bos_char_len=5, boundaries=[13], window=20) == 14
        with pytest.raises(ValueError, match=r"index 14; .*retained_boundaries=\[14\]"):
            assert_turn_roundtrip(decoded, text, bos_char_len=5, boundaries=[14], window=20)

    def test_real_gemma_tokenizer_round_trips_a_marker_document(self):
        """The production repro: Gemma decodes '▁' as ' ' at the same width."""
        pytest.importorskip("transformers")
        from steering.models import resolve_tokenizer

        try:
            tokenizer = resolve_tokenizer("gemma-3-4b-pt")
        except Exception as exc:  # noqa: BLE001 - offline machines without the cache
            pytest.skip(f"gemma-3-4b-pt tokenizer unavailable: {exc}")
        text = (
            "<user_start>Replace the BPE split character '▁' (U+2581) with a space."
            "<user_end><assistant_start>def f(s): return s.replace('▁', ' ')<assistant_end>"
        )
        ids = tokenizer(text)["input_ids"]
        decoded = tokenizer.decode(ids)
        bos_char_len = len(decoded) - len(text)
        assert bos_char_len >= 0 and decoded[bos_char_len:] != text, (
            "the repro requires the tokenizer to normalize the marker"
        )
        boundary = text.index("<assistant_start>")
        assert assert_turn_roundtrip(
            decoded, text, bos_char_len=bos_char_len, boundaries=[boundary]
        ) is None


def test_turn_segments_for_document_returns_geometry_and_tail_delta():
    assert turn_segments_for_document(
        [0, 1, 2, 3],
        _BosCharacterTokenizer(),
        "abc",
        [0, 2],
        PositiveInt.parse(2),
    ) == ([(0, 2), (2, 3), (3, 4)], [0, 6, 7], [6, 7, 8], None)


def test_turn_segments_for_document_returns_tolerated_tail_drift():
    """A decode that drifts inside the tail window reports the drift length."""
    assert turn_segments_for_document(
        [0, 1, 2, 3],
        _BosCharacterTokenizer(),
        "abd",
        [0],
        PositiveInt.parse(2),
    ) == ([(0, 2), (2, 4)], [0, 6], [6, 8], 1)


def test_turn_segments_for_document_requires_leading_bos():
    with pytest.raises(
        ValueError,
        match="assumes a leading BOS token; expected 0, got 1 for document 'abc'",
    ):
        turn_segments_for_document(
            [1, 2, 3],
            _BosCharacterTokenizer(),
            "abc",
            [0],
            PositiveInt.parse(2),
        )


def test_turn_segments_for_document_names_document_on_boundary_error():
    with pytest.raises(
        ValueError,
        match="strictly increasing.*for document 'abc'",
    ):
        turn_segments_for_document(
            [0, 1, 2, 3],
            _BosCharacterTokenizer(),
            "abc",
            [0, 2, 2],
            PositiveInt.parse(2),
        )


def test_segment_mode_from_metadata_contract():
    assert segment_mode_from_metadata({}, source="legacy") is SegmentMode.fixed_tokens
    for mode in SegmentMode:
        assert segment_mode_from_metadata(
            {"segment_mode": mode.value}, source="shard"
        ) is mode
    with pytest.raises(
        ValueError,
        match=(
            "checkpoint has unknown sft_segment_mode='bogus'.*fixed_tokens.*"
            "turn_delimited_fixed_tokens"
        ),
    ):
        segment_mode_from_metadata(
            {"segment_mode": "fixed_tokens", "sft_segment_mode": "bogus"},
            key="sft_segment_mode",
            source="checkpoint",
        )


class TestFixedTokens:
    def test_exact_multiple(self):
        segs = _segment_tokens(
            list(range(8)), SegmentMode.fixed_tokens, segment_size=4
        )
        assert segs == [(0, 4), (4, 8)]

    def test_with_remainder(self):
        segs = _segment_tokens(
            list(range(10)), SegmentMode.fixed_tokens, segment_size=4
        )
        assert segs == [(0, 4), (4, 8), (8, 10)]

    def test_smaller_than_segment(self):
        segs = _segment_tokens(
            [1, 2, 3], SegmentMode.fixed_tokens, segment_size=10
        )
        assert segs == [(0, 3)]

    def test_single_token(self):
        segs = _segment_tokens([42], SegmentMode.fixed_tokens, segment_size=4)
        assert segs == [(0, 1)]

    def test_empty_input(self):
        with pytest.raises(AssertionError):
            _segment_tokens([], SegmentMode.fixed_tokens, segment_size=4)

    def test_segment_size_one(self):
        segs = _segment_tokens(
            [1, 2, 3], SegmentMode.fixed_tokens, segment_size=1
        )
        assert segs == [(0, 1), (1, 2), (2, 3)]

    @PROPERTY_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=5000),
        seg_size=st.integers(min_value=1, max_value=500),
    )
    def test_full_coverage(self, n, seg_size):
        segs = _segment_tokens(
            list(range(n)), SegmentMode.fixed_tokens, segment_size=seg_size
        )
        assert segs[0][0] == 0
        assert segs[-1][1] == n
        for i in range(1, len(segs)):
            assert segs[i][0] == segs[i - 1][1]

    @PROPERTY_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=5000),
        seg_size=st.integers(min_value=1, max_value=500),
    )
    def test_segment_count(self, n, seg_size):
        segs = _segment_tokens(
            list(range(n)), SegmentMode.fixed_tokens, segment_size=seg_size
        )
        assert len(segs) == math.ceil(n / seg_size)

    @PROPERTY_SETTINGS
    @given(
        n=st.integers(min_value=1, max_value=5000),
        seg_size=st.integers(min_value=1, max_value=500),
    )
    def test_all_segments_nonempty(self, n, seg_size):
        segs = _segment_tokens(
            list(range(n)), SegmentMode.fixed_tokens, segment_size=seg_size
        )
        for start, end in segs:
            assert end > start
