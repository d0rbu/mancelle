"""Tests for _batch_roundtrip_stable row coercion (FIX-5, issue #590).

Verifies that _batch_roundtrip_stable handles tokenizers whose batched output rows
are tuples, ndarray-like objects exposing .tolist(), or flat single lists — exactly
mirroring the single-doc oracle (is_roundtrip_stable) for each case.
"""

from __future__ import annotations

import pytest

from steering._roundtrip_seam import is_roundtrip_stable
from steering.feature_shard_splice import SpliceInternalError, _batch_roundtrip_stable
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

_BASE_TOK = CharMergeTokenizer(bos_id=500)


class _TolistObj:
    """Minimal ndarray-like with .tolist() exposing an int list."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = list(ids)

    def tolist(self) -> list[int]:
        return self._ids

    def __iter__(self):
        return iter(self._ids)


class _TupleRowTokenizer:
    """Wraps CharMergeTokenizer, returns batched input_ids as a tuple of tuples."""

    def __init__(self, delegate) -> None:
        self._d = delegate

    def __call__(self, texts, **kwargs):
        result = self._d(texts, **kwargs)
        ids = result["input_ids"]
        # Only convert when batched (list of lists). Single-text output is flat.
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return {"input_ids": tuple(tuple(row) for row in ids)}
        return result

    def decode(self, ids):
        return self._d.decode(ids)

    # Forward BOS attribute so to_text works
    @property
    def bos_token_id(self):
        return self._d.bos_id


class _ArrayRowTokenizer:
    """Wraps CharMergeTokenizer, returns batched input_ids as a list of _TolistObj."""

    def __init__(self, delegate) -> None:
        self._d = delegate

    def __call__(self, texts, **kwargs):
        result = self._d(texts, **kwargs)
        ids = result["input_ids"]
        # Only convert when batched (list of lists). Single-text output is flat.
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return {"input_ids": [_TolistObj(row) for row in ids]}
        return result

    def decode(self, ids):
        return self._d.decode(ids)

    @property
    def bos_token_id(self):
        return self._d.bos_id


class _FlatListTokenizer:
    """Non-HF tokenizer returning a flat list for a SINGLE text (no nesting).

    For batched input with one text, returns a flat list like [id1, id2, ...] instead
    of [[id1, id2, ...]]. This is the non-HF flat-single-list regime.
    """

    def __init__(self, delegate) -> None:
        self._d = delegate

    def __call__(self, texts, **kwargs):
        result = self._d(texts, **kwargs)
        ids = result["input_ids"]
        # For exactly one text, flatten the outer list to expose the flat-list regime.
        if isinstance(ids, list) and len(ids) == 1:
            return {"input_ids": ids[0]}
        return result

    def decode(self, ids):
        return self._d.decode(ids)

    @property
    def bos_token_id(self):
        return self._d.bos_id


_TUPLE_TOK = _TupleRowTokenizer(_BASE_TOK)
_ARRAY_TOK = _ArrayRowTokenizer(_BASE_TOK)
_FLAT_TOK = _FlatListTokenizer(_BASE_TOK)


# Two stable docs
_STABLE_IDS = [
    tuple(_BASE_TOK("hello world", padding=False)["input_ids"]),
    tuple(_BASE_TOK("abcde", padding=False)["input_ids"]),
]


# ---------------------------------------------------------------------------
# Test (i): batched rows as tuples
# ---------------------------------------------------------------------------


def test_tuple_rows_batch_stable():
    """Tuple rows (tuple of tuples) produce correct stability verdicts."""
    verdicts = _batch_roundtrip_stable(_STABLE_IDS, _TUPLE_TOK)
    assert verdicts == [True, True], f"expected [True, True], got {verdicts}"


def test_tuple_rows_single_doc():
    """Tuple rows work for a single-doc batch."""
    ids = _STABLE_IDS[:1]
    verdicts = _batch_roundtrip_stable(ids, _TUPLE_TOK)
    assert verdicts == [True], f"expected [True], got {verdicts}"


# ---------------------------------------------------------------------------
# Test (ii): rows as objects exposing .tolist() (ndarray-like)
# ---------------------------------------------------------------------------


def test_array_rows_batch_stable():
    """ndarray-like rows (with .tolist()) produce correct stability verdicts."""
    verdicts = _batch_roundtrip_stable(_STABLE_IDS, _ARRAY_TOK)
    assert verdicts == [True, True], f"expected [True, True], got {verdicts}"


def test_array_rows_single_doc():
    """ndarray-like rows work for a single-doc batch."""
    ids = _STABLE_IDS[:1]
    verdicts = _batch_roundtrip_stable(ids, _ARRAY_TOK)
    assert verdicts == [True], f"expected [True], got {verdicts}"


# ---------------------------------------------------------------------------
# Test (iii): flat-single-list regime (non-HF returning one flat list)
# ---------------------------------------------------------------------------


def test_flat_list_single_doc_stable():
    """Flat-list regime (non-HF, single text) returns correct verdict."""
    ids = _STABLE_IDS[:1]
    verdicts = _batch_roundtrip_stable(ids, _FLAT_TOK)
    assert verdicts == [True], f"expected [True], got {verdicts}"


# ---------------------------------------------------------------------------
# Test (iv): equality with per-doc is_roundtrip_stable across all row types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tok_name,tok", [
    ("base", _BASE_TOK),
    ("tuple_rows", _TUPLE_TOK),
    ("array_rows", _ARRAY_TOK),
])
def test_batch_agrees_with_per_doc_oracle(tok_name, tok):
    """_batch_roundtrip_stable == is_roundtrip_stable per-doc for all row types."""
    docs_ids = _STABLE_IDS
    batch_verdicts = _batch_roundtrip_stable(docs_ids, tok)
    per_doc_verdicts = [is_roundtrip_stable(ids, tok) for ids in docs_ids]
    assert batch_verdicts == per_doc_verdicts, (
        f"{tok_name}: batch {batch_verdicts} != per-doc {per_doc_verdicts}"
    )


def test_flat_regime_agrees_with_per_doc_oracle():
    """Flat-list regime agrees with is_roundtrip_stable for single doc."""
    ids = _STABLE_IDS[:1]
    batch_v = _batch_roundtrip_stable(ids, _FLAT_TOK)
    per_doc_v = [is_roundtrip_stable(ids[0], _FLAT_TOK)]
    assert batch_v == per_doc_v, f"batch {batch_v} != per-doc {per_doc_v}"


# ---------------------------------------------------------------------------
# Test (v): mixed stable/unstable rows
# ---------------------------------------------------------------------------


class _UnstableSuffixTokenizer:
    """Stable for BOS-led ids that start with 500; unstable for anything else.

    decode() appends 'X', so re-encoding always produces an extra token.
    But for the BOS case, to_text strips the BOS and the body re-encodes cleanly
    (we use CharMergeTokenizer for stable texts). Unstable if no BOS.
    """

    def __init__(self, base) -> None:
        self._b = base

    def __call__(self, texts, **kwargs):
        result = self._b(texts, **kwargs)
        ids = result["input_ids"]
        # Return as tuples to test coercion
        if isinstance(ids, list):
            return {"input_ids": tuple(tuple(row) for row in ids)}
        return result

    def decode(self, ids):
        return self._b.decode(ids)

    @property
    def bos_token_id(self):
        return self._b.bos_id


def test_mixed_stable_unstable_tuple_rows():
    """Mixed stable/unstable docs with tuple rows returns correct per-doc verdicts."""
    stable = tuple(_BASE_TOK("hello", padding=False)["input_ids"])
    # Craft unstable ids: give raw chars with no BOS — decode strips nothing,
    # but re-encoding via CharMergeTokenizer adds BOS → different ids.
    # Actually for CharMergeTokenizer, to_text strips BOS, then re-encoding adds it back.
    # So (BOS, chars...) is stable. For unstable: omit BOS or use wrong ids.
    # Use ids that don't correspond to any decodable stable text:
    # (97,) = 'a', to_text sees no BOS → decode([97]) = 'a',
    # re-encode('a') = [500, 97] (BOS + a) ≠ (97,) → unstable
    unstable = (97,)  # 'a' without BOS → not round-trip stable

    mixed_ids = [stable, unstable]
    tok = _UnstableSuffixTokenizer(_BASE_TOK)
    verdicts = _batch_roundtrip_stable(mixed_ids, tok)
    assert verdicts[0] is True, f"stable doc should pass: {verdicts[0]}"
    assert verdicts[1] is False, f"unstable doc should fail: {verdicts[1]}"


# ---------------------------------------------------------------------------
# Test (vi): empty docs_ids raises SpliceInternalError (ITEM-1)
# ---------------------------------------------------------------------------


def test_empty_docs_ids_raises():
    """_batch_roundtrip_stable raises SpliceInternalError on empty docs_ids.

    An empty batch is a caller logic error: load_source_segments always has >= 1 doc,
    and _first_unstable_piece is only reached when target_k >= 2 (guaranteeing at
    least one non-identity piece to check). Passing [] is never legitimate.
    """
    with pytest.raises(SpliceInternalError, match="empty docs_ids"):
        _batch_roundtrip_stable([], _BASE_TOK)
