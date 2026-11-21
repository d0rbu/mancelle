"""Differential tests for the exact tiktoken byte offset path (issue #1269)."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pytest
import tiktoken
from hypothesis import given, settings
from hypothesis import strategies as st

from nanochat.tokenizer import get_tokenizer
from steering.segmentation import (
    _HYBRID_CHUNK_SIZE,
    _IncrementalReplaceLength,
    _tiktoken_token_bytes,
    _token_to_char_offsets_chunked_bytes,
    _token_to_char_offsets_progressive_bytes,
    get_path_stats,
    token_to_char_offsets,
)


# Frozen pre-#1269 reference implementation (copied from segmentation.py @ 78c841725).
# Do NOT "fix" or update; the differential is the point.
def _reference_token_to_char_offsets_progressive(
    tokenizer, token_ids: list[int]
) -> list[int]:
    """O(n²) progressive prefix decode with merge-aware offset correction.

    When byte-fallback tokens complete a multi-byte character, the decoded
    prefix shrinks (e.g., 3 replacement chars → 1 real char).  Instead of
    clamping with max() (which inflates offsets[-1]), we detect the merge and
    retroactively fix the preceding offsets for the byte-fallback group.

    Guarantees:
        offsets[0] == 0
        offsets[-1] == len(tokenizer.decode(token_ids))
        offsets[i+1] >= offsets[i]  (non-decreasing)

    Zero-width tokens (offsets[i] == offsets[i+1]) occur for intermediate
    byte-fallback tokens within a multi-byte group.  Callers that need
    non-zero width must expand these upstream (e.g. _boundaries_from_offsets).
    """
    offsets = [0]
    for i in range(len(token_ids)):
        prefix = tokenizer.decode(token_ids[: i + 1])
        new_len = len(prefix)

        if new_len < offsets[-1]:
            # Merge detected: byte-fallback group completed a multi-byte char.
            # The preceding offsets were inflated by replacement characters.
            # Walk back and reset them to the base offset (just before the
            # merged character).  A completed multi-byte merge replaces N
            # replacement chars with 1 real char, so the char position just
            # before the merged char is at new_len - 1.
            base = max(0, new_len - 1)
            for j in range(len(offsets) - 1, 0, -1):
                if offsets[j] <= base:
                    break
                offsets[j] = base

        offsets.append(new_len)
    return offsets


def _reference_token_to_char_offsets_chunked(
    tokenizer,
    token_ids: list[int],
    per_token_texts: list[str],
) -> list[int]:
    """Hybrid path: chunk the sequence and fast-check each chunk.

    For each chunk, decodes the prefix up to the chunk boundary and checks
    if the actual length matches the running optimistic offset from per-token
    lengths. Chunks that match use fast per-token offsets. The first chunk
    that diverges triggers a fall-back to the full progressive O(n²) path
    for the whole document — guaranteed correct, no chunk-boundary corner
    cases. Most pretraining text has no divergent chunks so the fast O(n)
    path runs to completion.

    The previous slow per-chunk fallback was buggy: a walk-back triggered
    inside a divergent chunk could cross into a previously fast-processed
    chunk and clamp its offsets to a low ``base`` value, producing a long
    flat region whose offsets were inconsistent with what the progressive
    path would produce. This manifested as zero-width source segments in
    feature shards — observed in production on shard_00355/doc_id=15373,
    a Wikipedia article containing an Old Italic inscription whose 32+
    consecutive byte-fallback Gemma tokens collapsed onto a single char
    position. By UTF-8 design, 32 bytes always span at least 8 characters,
    so a 32-token zero-width range is impossible — proving the chunked
    path was wrong rather than the data being pathological.
    """
    n = len(token_ids)
    chunk_size = _HYBRID_CHUNK_SIZE
    per_tok_lens = [len(s) for s in per_token_texts]

    offsets = [0] * (n + 1)
    chunk_start = 0

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)

        # Optimistic: assume per-token lengths are correct for this chunk
        optimistic_end = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            optimistic_end += per_tok_lens[i]

        # Validate: does the actual prefix decode length match?
        actual_prefix_len = len(tokenizer.decode(token_ids[:chunk_end]))

        if actual_prefix_len != optimistic_end:
            # Any divergence ⇒ fall back to the full progressive path for
            # the whole document. See the docstring for why we don't try
            # to slow-process just this chunk.
            return _reference_token_to_char_offsets_progressive(tokenizer, token_ids)

        # Fast: per-token offsets are correct for this chunk
        pos = offsets[chunk_start]
        for i in range(chunk_start, chunk_end):
            pos += per_tok_lens[i]
            offsets[i + 1] = pos

        chunk_start = chunk_end

    return offsets


def _reference_token_to_char_offsets(
    tokenizer, token_ids: list[int]
) -> tuple[str, list[int]]:
    """Convert token boundaries to character offsets.

    Uses an O(n) fast path when per-token decodes concatenate to the full
    decoded text (true for ASCII-heavy byte-level BPE). For inputs where the
    fast path fails on the whole-sequence check, hands off to a
    chunked fast path that processes the document chunk-by-chunk; the first
    chunk that diverges triggers a fall-back to the O(n²) progressive prefix
    decode for the whole document. The progressive path handles
    context-dependent decoding (multi-byte UTF-8 merges, SentencePiece
    whitespace, byte fallback) correctly; an earlier per-chunk slow
    fallback was removed because its merge-correction walk-back could
    cross chunk boundaries and produce inconsistent offsets — see
    ``_token_to_char_offsets_chunked`` for the post-mortem.

    Returns a list of length len(token_ids) + 1 where offsets[i] is the character
    position where token i starts, and offsets[-1] is the total character count.
    """
    if not token_ids:
        return "fast", [0]

    n = len(token_ids)
    if n <= 16:
        return "small_n", _reference_token_to_char_offsets_progressive(tokenizer, token_ids)

    # O(n): decode each token individually + full sequence once
    per_token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    full_text = tokenizer.decode(token_ids)

    # String comparison catches all context-dependent decode effects:
    # - Multi-byte UTF-8 merges (replacement chars vs actual chars)
    # - SentencePiece leading space stripping vs preservation
    # - Byte fallback hex strings vs merged characters
    if "".join(per_token_texts) == full_text:
        offsets = [0, *itertools.accumulate(len(text) for text in per_token_texts)]
        return "fast", offsets

    # Hybrid path: chunk-by-chunk fast check. Each clean chunk uses
    # cumulative per-token lengths (O(n) total). The first chunk that
    # diverges triggers a fall-back to the full progressive O(n²) path
    # for the whole document — see ``_reference_token_to_char_offsets_chunked``
    # for why we do not try to slow-process individual chunks any more.
    return "slow", _reference_token_to_char_offsets_chunked(
        tokenizer, token_ids, per_token_texts
    )


ADVERSARIAL_TEXTS = [
    "",
    "hello world",
    "short",
    "x " + "𐌇𐌀𐌓𐌉𐌙𐌀𐌔𐌕𐌉𐌕𐌄𐌉𐌅𐌀" * 4 + " y",
    "안녕하세요 세계",
    "日本語のテキストです。" * 40,
    "café née naïve",
    "emoji 👩‍👩‍👧‍👦 zwj 🏳️‍🌈 seq",
    "mixed 𐌀 ascii 한 more € text" * 30,
    "a" * 5000,
    ("word " * 200 + "𐌀") * 3,
]


@pytest.fixture(scope="module")
def tok():
    try:
        tokenizer = get_tokenizer()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"nanochat tokenizer cache unavailable: {exc}")
    enc = getattr(tokenizer, "enc", None)
    if not isinstance(enc, tiktoken.Encoding):
        pytest.skip("nanochat tokenizer is not tiktoken-backed")
    return tokenizer


class TestDifferentialRealTokenizer:
    def test_texts_with_and_without_bos(self, tok) -> None:
        bos = tok.get_bos_token_id()
        for text in ADVERSARIAL_TEXTS:
            for prepend in (None, bos):
                ids = tok.encode(text, prepend=prepend)
                expected_path, expected = _reference_token_to_char_offsets(tok, ids)
                get_path_stats()
                actual = token_to_char_offsets(tok, ids)
                stats = get_path_stats()
                assert actual == expected
                if ids:
                    assert stats[expected_path]["count"] == 1
                    assert sum(bucket["count"] for bucket in stats.values()) == 1
                else:
                    assert sum(bucket["count"] for bucket in stats.values()) == 0

    @given(st.data())
    @settings(max_examples=30, deadline=None)
    def test_hypothesis_random_token_ids(self, tok, data) -> None:
        token_ids = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=tok.get_vocab_size() - 1),
                min_size=0,
                max_size=300,
            )
        )
        special_ids = set(tok.enc._special_tokens.values())
        token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
        expected_path, expected = _reference_token_to_char_offsets(tok, token_ids)
        get_path_stats()
        actual = token_to_char_offsets(tok, token_ids)
        stats = get_path_stats()
        assert actual == expected
        if token_ids:
            assert stats[expected_path]["count"] == 1
        else:
            assert sum(bucket["count"] for bucket in stats.values()) == 0


class TestTiktokenByteInvariants:
    def test_per_token_and_joined_bytes_match_decode(self, tok) -> None:
        bos = tok.get_bos_token_id()
        samples = [
            [bos],
            tok.encode("hello 안녕 𐌀 👩‍👩‍👧‍👦", prepend=bos),
            [bos, 0, 1, 2, 3, 32759],
        ]
        for ids in samples:
            token_bytes = tok.enc.decode_tokens_bytes(ids)
            assert [data.decode("utf-8", errors="replace") for data in token_bytes] == [
                tok.decode([token_id]) for token_id in ids
            ]
            assert b"".join(token_bytes) == tok.enc.decode_bytes(ids)
            assert b"".join(token_bytes).decode("utf-8", errors="replace") == tok.decode(ids)


class TestIncrementalReplaceLength:
    @given(st.lists(st.binary(max_size=8), min_size=0, max_size=30))
    @settings(max_examples=50, deadline=None)
    def test_matches_one_shot_on_every_prefix_hypothesis(self, blobs: list[bytes]) -> None:
        engine = _IncrementalReplaceLength()
        accumulated = b""
        for blob in blobs:
            accumulated += blob
            assert engine.feed(blob) == len(accumulated.decode("utf-8", errors="replace"))

    def test_split_multibyte_every_way(self) -> None:
        for character in ["€", "𐌀", "한", "👩"]:
            payload = character.encode()
            for n_cuts in range(len(payload)):
                for cuts in itertools.combinations(range(1, len(payload)), n_cuts):
                    points = (0, *cuts, len(payload))
                    engine = _IncrementalReplaceLength()
                    accumulated = b""
                    for start, end in itertools.pairwise(points):
                        blob = payload[start:end]
                        accumulated += blob
                        assert engine.feed(blob) == len(
                            accumulated.decode("utf-8", errors="replace")
                        )


class TestDispatchSafety:
    def test_magicmock_takes_generic_path(self) -> None:
        mock = MagicMock()
        mock.decode.side_effect = lambda ids: "a" * len(ids)
        assert _tiktoken_token_bytes(mock, [1]) is None
        assert token_to_char_offsets(mock, list(range(20))) == list(range(21))
        assert mock.enc is not None

    def test_hf_style_tokenizer_takes_generic_path(self) -> None:
        class HFStyleTokenizer:
            tokenizer = object()

            def decode(self, ids):
                return "a" * len(ids)

        tokenizer = HFStyleTokenizer()
        assert _tiktoken_token_bytes(tokenizer, [1]) is None
        assert token_to_char_offsets(tokenizer, list(range(20))) == list(range(21))


class TestByteHelpersDirect:
    def test_progressive_bytes_equals_reference_progressive(self, tok) -> None:
        for text in ADVERSARIAL_TEXTS:
            ids = tok.encode(text, prepend=tok.get_bos_token_id())
            token_bytes = tok.enc.decode_tokens_bytes(ids)
            assert _token_to_char_offsets_progressive_bytes(token_bytes) == (
                _reference_token_to_char_offsets_progressive(tok, ids)
            )

    def test_chunked_bytes_equals_reference_chunked(self, tok) -> None:
        for text in ADVERSARIAL_TEXTS:
            ids = tok.encode(text, prepend=tok.get_bos_token_id())
            if len(ids) <= 16:
                continue
            token_bytes = tok.enc.decode_tokens_bytes(ids)
            per_token_texts = [data.decode("utf-8", errors="replace") for data in token_bytes]
            if "".join(per_token_texts) == tok.decode(ids):
                continue
            assert _token_to_char_offsets_chunked_bytes(token_bytes, per_token_texts) == (
                _reference_token_to_char_offsets_chunked(tok, ids, per_token_texts)
            )


def _toy_encoding() -> tiktoken.Encoding:
    """Hermetic byte-level tiktoken Encoding: 256 single-byte ranks + one special.

    Review finding (#1269): every real-tokenizer differential above skips when
    the nanochat tokenizer artifact is absent. This toy Encoding is a REAL
    tiktoken.Encoding (so ``_tiktoken_token_bytes``'s isinstance gate dispatches
    to the byte path) that needs no on-disk artifact, and its ids ARE raw byte
    values, so adversarial UTF-8 layouts can be constructed exactly.
    """
    return tiktoken.Encoding(
        name="toy-bytes-1269",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<sp>": 256},
    )


class _ToyTokenizer:
    def __init__(self) -> None:
        self.enc = _toy_encoding()

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


class _RealObjectNonTiktokenEnc:
    """A tokenizer whose ``enc`` is a genuine (non-Mock) non-tiktoken object."""

    def __init__(self) -> None:
        self.enc = object()

    def decode(self, ids: list[int]) -> str:
        return "a" * len(ids)


class TestHermeticToyEncodingDifferential:
    """Byte-path vs frozen reference on a hermetic Encoding — never skips."""

    def _assert_case(self, ids: list[int]) -> None:
        toy = _ToyTokenizer()
        _ = get_path_stats()
        actual = token_to_char_offsets(toy, ids)
        actual_stats = get_path_stats()
        expected_path, expected = _reference_token_to_char_offsets(toy, ids)
        assert actual == expected
        assert actual_stats[expected_path]["count"] == 1

    def test_incomplete_utf8_across_the_chunk_boundary(self) -> None:
        # 30 ASCII bytes, then U+10348 (f0 90 8d 88) split across token
        # indices 30..33 — the incomplete character straddles the 32-token
        # chunk edge — then ASCII out past a second chunk.
        char = "\U00010348".encode()
        ids = [65] * 30 + list(char) + [66] * 40
        assert len(ids) > 2 * 32
        self._assert_case(ids)

    def test_special_token_mid_sequence(self) -> None:
        char = "é".encode()  # c3 a9
        ids = [97] * 10 + [256] + list(char) + [98] * 10 + [256] + [99] * 5
        self._assert_case(ids)

    def test_continuation_byte_soup(self) -> None:
        # Lone continuation bytes render one replacement char each; long
        # enough to exercise the chunked path and its progressive bail-out.
        ids = ([0x80, 0x81, 0xBF] * 40)[:110]
        self._assert_case(ids)

    def test_pure_ascii_fast_path(self) -> None:
        ids = [104, 101, 108, 108, 111] * 8
        self._assert_case(ids)

    def test_small_n_toy(self) -> None:
        ids = [65, 0xF0, 0x90, 0x8D, 0x88, 66]
        self._assert_case(ids)

    def test_real_object_non_tiktoken_enc_takes_generic_path(self) -> None:
        tok = _RealObjectNonTiktokenEnc()
        assert _tiktoken_token_bytes(tok, [1, 2, 3]) is None
        assert token_to_char_offsets(tok, list(range(20))) == list(range(21))

    def test_chunked_bytes_all_clean_path_direct(self) -> None:
        """Pin the clean-chunk fill of the chunked byte twin directly.

        The dispatcher never routes all-clean inputs here (they pass the
        whole-doc fast check first — see the reachability note on
        ``_token_to_char_offsets_chunked_bytes``), so a mutation campaign
        found the fill otherwise untested (equivalent mutant). Exercise the
        final-return path as a unit: pure ASCII single-byte tokens, every
        chunk clean, offsets must be the exact cumulative lengths and agree
        with the progressive twin.
        """
        # Single chunk (n <= _HYBRID_CHUNK_SIZE): the ONLY shape where the fill
        # decides the returned value directly — in multi-chunk inputs a
        # corrupted fill poisons the next chunk's optimistic base, trips the
        # divergence check, and self-heals through the progressive bail (the
        # first mutant survived exactly that way).
        token_bytes = [bytes([65 + (i % 26)]) for i in range(20)]
        per_token_texts = [b.decode("utf-8", errors="replace") for b in token_bytes]
        offsets = _token_to_char_offsets_chunked_bytes(token_bytes, per_token_texts)
        assert offsets == list(range(21))
        assert offsets == _token_to_char_offsets_progressive_bytes(token_bytes)

        # Multi-chunk all-clean: pins the cumulative fill across chunk seams.
        token_bytes = [bytes([65 + (i % 26)]) for i in range(100)]
        per_token_texts = [b.decode("utf-8", errors="replace") for b in token_bytes]
        offsets = _token_to_char_offsets_chunked_bytes(token_bytes, per_token_texts)
        assert offsets == list(range(101))
        assert offsets == _token_to_char_offsets_progressive_bytes(token_bytes)
