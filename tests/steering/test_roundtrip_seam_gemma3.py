"""Unit tests for the gemma-3 U+2581 source-data normalization (issue #590).

Gemma-3 parquet text can contain literal SentencePiece space markers (U+2581)
that decode as normal spaces, so the source doc's own tokenization is not
idempotent. The normalization repairs ONLY that narrow case (length-preserving,
piece-equal, marker-containing mismatch) and otherwise leaves the ids untouched.
"""

from __future__ import annotations

import pytest

from steering._roundtrip_seam import (
    _GEMMA3_MARKER_MAX_DOCS_PER_SHARD,
    Gemma3MarkerError,
    _is_gemma3_tokenizer,
    normalize_gemma3_space_marker_doc,
)
from tests.steering._splice_test_tokenizers import (
    CharMergeTokenizer,
    Gemma3LikeMarkerTokenizer,
)

MARKER = "▁"


def test_is_gemma3_tokenizer_detects_by_name():
    assert _is_gemma3_tokenizer(Gemma3LikeMarkerTokenizer()) is True
    assert _is_gemma3_tokenizer(CharMergeTokenizer()) is False


def test_normalizes_literal_marker_source_doc():
    tok = Gemma3LikeMarkerTokenizer()
    kept_ids = tok.encode(MARKER + "x")  # [900, 120]
    out, docs, mism = normalize_gemma3_space_marker_doc(
        kept_ids, tok, source_text=MARKER + "x", doc_index=0,
        truncated=False, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == [901, 120]            # marker id 900 rewritten to space id 901
    assert (docs, mism) == (1, 1)       # one doc, one token mismatch normalized
    # idempotent: the re-encoded ids round-trip and need no further change
    out2, docs2, mism2 = normalize_gemma3_space_marker_doc(
        out, tok, source_text=MARKER + "x", doc_index=0,
        truncated=False, normalized_docs_seen=1, token_mismatches_seen=1,
    )
    assert out2 == [901, 120] and (docs2, mism2) == (1, 1)


def test_no_op_for_non_gemma3_tokenizer():
    tok = CharMergeTokenizer()
    out, docs, mism = normalize_gemma3_space_marker_doc(
        [97, 98], tok, source_text=MARKER + "x", doc_index=0,
        truncated=False, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == [97, 98] and (docs, mism) == (0, 0)


def test_no_op_when_source_has_no_marker():
    tok = Gemma3LikeMarkerTokenizer()
    kept_ids = tok.encode("xy")
    out, docs, mism = normalize_gemma3_space_marker_doc(
        kept_ids, tok, source_text="xy", doc_index=0,
        truncated=False, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == kept_ids and (docs, mism) == (0, 0)


def test_no_op_when_decoded_does_not_match_source():
    # source says marker+x but ids decode to " y" -> not the U+2581 case, leave alone.
    tok = Gemma3LikeMarkerTokenizer()
    out, docs, mism = normalize_gemma3_space_marker_doc(
        [900, 121], tok, source_text=MARKER + "x", doc_index=0,
        truncated=False, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == [900, 121] and (docs, mism) == (0, 0)


def test_truncated_source_uses_prefix_match():
    # truncated: normalized_source.startswith(decoded). Source "marker+xyz" but
    # ids only cover "marker+x" -> still normalizable.
    tok = Gemma3LikeMarkerTokenizer()
    out, docs, mism = normalize_gemma3_space_marker_doc(
        [900, 120], tok, source_text=MARKER + "xyz", doc_index=0,
        truncated=True, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == [901, 120] and (docs, mism) == (1, 1)


def test_doc_cap_overflow_raises():
    tok = Gemma3LikeMarkerTokenizer()
    with pytest.raises(Gemma3MarkerError):
        normalize_gemma3_space_marker_doc(
            [900, 120], tok, source_text=MARKER + "x", doc_index=7,
            truncated=False,
            normalized_docs_seen=_GEMMA3_MARKER_MAX_DOCS_PER_SHARD,  # already at cap
            token_mismatches_seen=0,
        )


@pytest.mark.slow
def test_real_gemma3_tokenizer_detected_and_noop_on_plain_text():
    from steering.models import resolve_tokenizer

    tok = resolve_tokenizer("gemma-3-1b-pt")
    assert _is_gemma3_tokenizer(tok) is True
    ids = tok("plain ascii text without markers", padding=False)["input_ids"]
    out, docs, mism = normalize_gemma3_space_marker_doc(
        list(ids), tok, source_text="plain ascii text without markers",
        doc_index=0, truncated=False, normalized_docs_seen=0, token_mismatches_seen=0,
    )
    assert out == list(ids) and (docs, mism) == (0, 0)
