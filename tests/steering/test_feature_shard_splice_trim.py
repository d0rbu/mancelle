"""Multibyte-truncation trim: truncated docs cut mid-multibyte-char are trimmed
back (byte-fallback tokens only) to round-trip stability; everything else raises."""

from __future__ import annotations

import pytest
import torch as t

from steering.feature_io import load_features, save_features
from steering.feature_shard_splice import (
    SourceDocNotStableError,
    _is_byte_fallback_token,
    load_source_segments,
)
from tests.steering._splice_fixtures import doc_seg_ranges, make_base_shard
from tests.steering._splice_test_tokenizers import (
    CharMergeTokenizer,
    Gemma3LikeMarkerTokenizer,
)

# ---------------------------------------------------------------------------
# Minimal test-only tokenizers
# ---------------------------------------------------------------------------


class _AppendXDecodeTokenizer:
    """Deterministic unstable tokenizer: encodes char-by-char, decode appends 'X'.

    Encoding: each char maps to its ord value; no BOS.
    Decoding: concatenate chars, then append 'X' — so round-tripping always
    changes the token sequence (decode produces a longer text, which re-encodes
    with an extra token for 'X').
    No convert_ids_to_tokens → _is_byte_fallback_token always returns False.
    """

    def __call__(self, text, *, padding=False, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) for c in txt] for txt in text]}
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids) + "X"


class _ByteFallbackTestTokenizer:
    """Tokenizer with byte-fallback token 200 (<0xC8>) for trim edge-case tests.

    Encoding: prepends BOS (999) then encodes chars; chr(0xC8) maps to id 200.
    Decoding: 200 → 'XY' (two chars), so any doc ending in token 200 is unstable.
    convert_ids_to_tokens: token 200 → '<0xC8>' (matches byte-fallback regex).
    """

    bos_token_id = 999

    def __call__(self, text, *, padding=False, add_special_tokens=True):
        if isinstance(text, list):
            return {
                "input_ids": [
                    self(txt, padding=padding, add_special_tokens=add_special_tokens)[
                        "input_ids"
                    ]
                    for txt in text
                ]
            }
        ids = [self.bos_token_id]
        for ch in text:
            ids.append(200 if ch == "\xc8" else ord(ch))
        return {"input_ids": ids}

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.bos_token_id:
                continue
            out.append("XY" if i == 200 else chr(i))
        return "".join(out)

    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, (list, tuple)):
            return [self.convert_ids_to_tokens(i) for i in token_id]
        if int(token_id) == 200:
            return "<0xC8>"
        if int(token_id) == self.bos_token_id:
            return "<BOS>"
        return chr(int(token_id))


_TOK_UNSTABLE = _AppendXDecodeTokenizer()
_TOK_BYTEFB = _ByteFallbackTestTokenizer()
_TOK_CLEAN = CharMergeTokenizer(bos_id=500)


# ---------------------------------------------------------------------------
# slow: real gemma midchar truncation trim
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trim_real_gemma_midchar_truncation(gemma_tok, tmp_path):
    """𐍈 (U+10348) → 4 byte-fallback tokens; cutting to 3 dangling → pops 3."""
    text = "hello world " * 3 + "\U00010348"
    full = gemma_tok(text, padding=False)["input_ids"]
    limit = len(full) - 1  # cut off the 4th byte-fallback of U+10348, leaving 3

    # Build base shard through the real save/load path (fp16-quantised like extraction).
    # segment_size=6: kept_ids is 11 tokens → segments (0..6) and (6..11).
    # The last segment has 5 tokens (2 normal + 3 byte-fallbacks); after trimming
    # 3 byte-fallbacks, it has 2 tokens remaining — not empty, trim succeeds.
    segment_size = 6
    base_dict = make_base_shard([text], gemma_tok, segment_size, context_window=limit)
    base_path = save_features(base_dict, tmp_path / "base_trim")
    base = load_features(base_path)

    segs, stats = load_source_segments(
        base,
        [text],
        gemma_tok,
        segment_size=segment_size,
        context_window=limit,
        sae_max_context=None,
        shard_name="trim_shard",
    )

    assert stats.n_trimmed_docs == 1
    assert stats.n_trimmed_tokens == 3  # 3 dangling fragment byte-fallbacks popped

    # The last segment's features are preserved verbatim (features untouched).
    ranges = doc_seg_ranges(base)
    last_gi = ranges[0][1] - 1  # global index of the last segment for doc 0
    offsets = base["offsets"]
    expected_fids = base["feature_ids"][int(offsets[last_gi]) : int(offsets[last_gi + 1])]
    expected_fmags = base["feature_mags"][int(offsets[last_gi]) : int(offsets[last_gi + 1])]
    last_seg = segs[0][-1]
    assert t.equal(last_seg.feature_ids, expected_fids)
    assert t.equal(last_seg.feature_mags, expected_fmags)
    # Tokens were shortened: the last token in the trimmed segment is NOT a byte-fallback.
    assert len(last_seg.tokens) >= 1
    assert not _is_byte_fallback_token(gemma_tok, last_seg.tokens[-1])


# ---------------------------------------------------------------------------
# Non-truncated unstable doc raises (no trim attempted)
# ---------------------------------------------------------------------------


def test_nontruncated_unstable_doc_raises():
    """A non-truncated unstable doc raises SourceDocNotStableError; no trim is tried."""
    tok = _TOK_UNSTABLE
    texts = ["hi"]
    shard = make_base_shard(texts, tok, segment_size=2, context_window=1000)
    with pytest.raises(SourceDocNotStableError, match="not round-trip stable and was not truncated"):
        load_source_segments(
            shard,
            texts,
            tok,
            segment_size=2,
            context_window=1000,
            sae_max_context=None,
            shard_name="shard_test",
        )


# ---------------------------------------------------------------------------
# Trim stops at non-byte-fallback token and raises
# ---------------------------------------------------------------------------


def test_trim_stops_at_non_byte_fallback_and_raises():
    """Truncated doc; last token is NOT a byte-fallback → trim raises immediately."""
    # _AppendXDecodeTokenizer has no convert_ids_to_tokens →
    # _is_byte_fallback_token always returns False for every token.
    tok = _TOK_UNSTABLE
    texts = ["hi"]  # encodes to [104, 105]; context_window=1 keeps only [104]
    shard = make_base_shard(texts, tok, segment_size=1, context_window=1)
    with pytest.raises(SourceDocNotStableError, match="not round-trip stable after popping"):
        load_source_segments(
            shard,
            texts,
            tok,
            segment_size=1,
            context_window=1,
            sae_max_context=None,
            shard_name="shard_test",
        )


# ---------------------------------------------------------------------------
# Trim that would empty the last segment raises
# ---------------------------------------------------------------------------


def test_trim_that_would_empty_last_segment_raises():
    """Trimming pops the only token in the last segment → SourceDocNotStableError."""
    tok = _TOK_BYTEFB
    # "abcde\xc8xxx": 10 tokens total (BOS + 9 chars); context_window=7 keeps 7.
    # kept_ids = [999, 97, 98, 99, 100, 101, 200]; with segment_size=6:
    #   segment 0: tokens 0..5 = [999,a,b,c,d,e] (6 tokens)
    #   segment 1: token  6   = [200]            (1 byte-fallback token)
    # _trim_truncated_doc_tail pops token 200 (byte-fallback) making ids stable,
    # but n_popped=1 == len(last_seg.tokens)=1 → last segment would be empty → raise.
    texts = ["abcde\xc8xxx"]
    shard = make_base_shard(texts, tok, segment_size=6, context_window=7)
    with pytest.raises(SourceDocNotStableError, match="byte-fallback trim"):
        load_source_segments(
            shard,
            texts,
            tok,
            segment_size=6,
            context_window=7,
            sae_max_context=None,
            shard_name="shard_test",
        )


# ---------------------------------------------------------------------------
# slow: byte-fallback predicate on real gemma
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_byte_fallback_predicate(gemma_tok):
    """_is_byte_fallback_token identifies <0xNN> byte-fallback tokens correctly."""
    # U+10348 (𐍈) → 4 byte-fallback tokens in gemma; the last one is a fragment.
    frag_ids = gemma_tok("\U00010348", padding=False)["input_ids"]
    last_frag = frag_ids[-1]
    assert _is_byte_fallback_token(gemma_tok, last_frag)
    # A normal token (last token of " hello") is NOT a byte-fallback.
    normal_ids = gemma_tok(" hello", padding=False)["input_ids"]
    assert not _is_byte_fallback_token(gemma_tok, normal_ids[-1])


# ---------------------------------------------------------------------------
# Stats are zero on a clean stable corpus
# ---------------------------------------------------------------------------


def test_stats_zero_on_clean_corpus():
    """No trims or marker normalizations on a clean CharMergeTokenizer corpus."""
    tok = _TOK_CLEAN
    texts = ["alpha beta", "gamma delta epsilon", "zeta eta theta"]
    shard = make_base_shard(texts, tok, segment_size=4, context_window=2048)
    segs, stats = load_source_segments(
        shard,
        texts,
        tok,
        segment_size=4,
        context_window=2048,
        sae_max_context=None,
        shard_name="clean_shard",
    )
    assert stats.n_trimmed_docs == 0
    assert stats.n_trimmed_tokens == 0
    assert stats.n_marker_normalized_docs == 0
    assert len(segs) == len(texts)


# ---------------------------------------------------------------------------
# Marker normalization still applies before trim check
# ---------------------------------------------------------------------------


def test_marker_norm_still_applies_before_trim():
    """Gemma3LikeMarkerTokenizer: marker normalization is applied and counted."""
    tok = Gemma3LikeMarkerTokenizer()
    # Source text contains ▁ (U+2581); normalization rewrites token 900 → 901.
    # After normalization the doc IS stable, so no trim is needed.
    marker = "▁"
    texts = [marker + "hello"]
    shard = make_base_shard(texts, tok, segment_size=8, context_window=100)
    segs, stats = load_source_segments(
        shard,
        texts,
        tok,
        segment_size=8,
        context_window=100,
        sae_max_context=None,
        shard_name="marker_shard",
    )
    assert stats.n_marker_normalized_docs == 1
    assert stats.n_trimmed_docs == 0  # normalization fixed the instability; no trim
    assert len(segs) == 1
    assert len(segs[0]) >= 1


# ---------------------------------------------------------------------------
# Combo: marker normalization + byte-fallback trim in the same shard
# ---------------------------------------------------------------------------


def test_marker_norm_and_byte_fallback_trim_combined():
    """Two docs in one shard: doc 0 needs marker normalization, doc 1 needs byte-fallback trim.

    Uses Gemma3LikeMarkerTokenizer(byte_fallback_id=500) which encodes '\\xff' as token
    500, decodes 500 as 'Z', and reports convert_ids_to_tokens(500)='<0xFF>' so that
    _is_byte_fallback_token recognises it.

    Doc 0 ('▁ab'): 3 tokens [900,97,98]; context_window=3 does NOT truncate (n_full=3).
    Normalization rewrites 900→901 (marker→space, same length, same ▁ piece) → stable.

    Doc 1 ('cd\\xff' + 'X'*100): 103 tokens; context_window=3 truncates to [99,100,500]
    (was_truncated=True). No normalization (no ▁ in text). is_roundtrip_stable fails
    (500 decodes to 'Z', re-encodes to ord('Z')=90 ≠ 500). _trim_truncated_doc_tail
    recognises token 500 as byte-fallback, pops it → [99,100] which IS stable.
    Last segment has 3 tokens [99,100,500] so n_popped=1 < 3 (no empty-segment raise).

    Net result: n_marker_normalized_docs=1, n_trimmed_docs=1, n_trimmed_tokens=1.
    """
    _BFID = 500  # byte-fallback id: '\\xff' encodes to 500; decode(500)='Z' (unstable)
    tok = Gemma3LikeMarkerTokenizer(byte_fallback_id=_BFID)

    # Doc 0: exactly fills context_window (no truncation); doc 1: well beyond it.
    doc0 = "▁ab"          # [900, 97, 98] = 3 tokens  (no BOS in Gemma3LikeMarkerTokenizer)
    doc1 = "cd\xff" + "X" * 100  # [99, 100, 500, 88*100] = 103 tokens; truncated to 3
    texts = [doc0, doc1]
    segment_size = 3
    context_window = 3

    shard = make_base_shard(texts, tok, segment_size=segment_size, context_window=context_window)
    segs, stats = load_source_segments(
        shard,
        texts,
        tok,
        segment_size=segment_size,
        context_window=context_window,
        sae_max_context=None,
        shard_name="combo_shard",
    )

    # Doc 0: marker normalization incremented; no trim needed.
    assert stats.n_marker_normalized_docs == 1
    # Doc 1: byte-fallback trim kicked in; 1 token (500) popped.
    assert stats.n_trimmed_docs == 1
    assert stats.n_trimmed_tokens == 1
    assert len(segs) == 2
