"""Edge-case tests for the round-trip seam machinery (issue #590)."""

from __future__ import annotations

import itertools

import pytest
import torch as t

from steering import _roundtrip_seam as rs
from steering._roundtrip_seam import (
    _HF_WINDOW_GUARD_TOKENS,
    _NON_HF_WINDOW_SIDE_TOKENS,
    WindowedSeamChecker,
    _infer_window_side_tokens,
    _input_ids_tuple,
    is_roundtrip_stable,
    to_text,
)
from steering.feature_shard_splice import recombination_seed
from tests.steering._seam_test_helpers import swap_is_stable
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

# -- _input_ids_tuple normalization -----------------------------------------


def test_input_ids_tuple_from_plain_list():
    assert _input_ids_tuple({"input_ids": [1, 2, 3]}) == (1, 2, 3)


def test_input_ids_tuple_from_batched_list():
    # Some tokenizers return [[...]] for a single string under certain calls.
    assert _input_ids_tuple({"input_ids": [[4, 5, 6]]}) == (4, 5, 6)


def test_input_ids_tuple_from_tensor():
    assert _input_ids_tuple({"input_ids": t.tensor([7, 8, 9])}) == (7, 8, 9)


def test_input_ids_tuple_missing_field_raises():
    with pytest.raises(TypeError, match="input_ids"):
        _input_ids_tuple({"not_input_ids": [1]})


def test_input_ids_tuple_non_integer_raises():
    with pytest.raises(TypeError):
        _input_ids_tuple({"input_ids": ["a", "b"]})


# -- to_text / round-trip ---------------------------------------------------


def test_to_text_no_bos_tokenizer_keeps_all_tokens():
    tok = CharMergeTokenizer()  # bos_id None
    ids = tuple(ord(c) for c in "hello")
    assert to_text(ids, tok) == "hello"


def test_roundtrip_stable_merge_within_a_segment_is_stable():
    # "ab" is a single merge token; its own round-trip is stable.
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    ids = tuple(tok("xaby", padding=False)["input_ids"])  # x, ab, y after bos
    assert is_roundtrip_stable(ids, tok)


# -- window-side inference --------------------------------------------------


def test_window_side_non_hf_is_fixed():
    assert _infer_window_side_tokens(CharMergeTokenizer()) == _NON_HF_WINDOW_SIDE_TOKENS


def test_window_side_hf_scales_with_max_byte_length():
    # Single-char merges -> max decoded piece is 2 bytes.
    tok2 = CharMergeTokenizer(merge_pairs=[("a", "b")], merge_id_base=64, expose_vocab=True)
    assert _infer_window_side_tokens(tok2) == 2 + _HF_WINDOW_GUARD_TOKENS


def test_window_side_cached_per_tokenizer():
    tok = CharMergeTokenizer(merge_pairs=[("a", "b")], merge_id_base=64, expose_vocab=True)
    checker = WindowedSeamChecker(tok)
    a = checker.window_side_tokens
    b = checker.window_side_tokens
    assert a == b
    assert id(tok) in rs._WINDOW_SIDE_TOKENS_CACHE


# -- checker stats counters (proves the windowed path actually ran) ---------


def _seg_tokens(texts, tok, seg):
    out = []
    for text in texts:
        ids = tok(text, padding=False)["input_ids"]
        out.append([tuple(ids[i:i + seg]) for i in range(0, len(ids), seg)])
    return out


def test_checker_records_local_accepts_and_rejects():
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    seg_tokens = _seg_tokens(["az", "xb", "pq"], tok, 2)
    checker = WindowedSeamChecker(tok)
    for a, b in itertools.combinations(range(3), 2):
        la, lb = len(seg_tokens[a]), len(seg_tokens[b])
        for k in range(min(la, lb) - 1):
            chain_a = [(a, j) for j in range(la)]
            chain_b = [(b, j) for j in range(lb)]
            swap_is_stable(checker, chain_a, chain_b, k, seg_tokens)
    assert checker.stats.local_checks > 0
    assert checker.stats.local_accepts + checker.stats.local_rejects == checker.stats.local_checks



# -- determinism of the seed ------------------------------------------------


def test_seed_is_64_bit_and_varied():
    seeds = {recombination_seed(42, i, 0.5) for i in range(100)}
    assert len(seeds) == 100  # no collisions over 100 indices
    assert all(0 <= s < 2**64 for s in seeds)


# -- real gemma byte-fallback + BOS at seams (the realistic stress) ---------


@pytest.mark.slow
def test_windowed_equals_oracle_with_byte_fallback_text(gemma_tok):
    # Rare scripts trigger byte-fallback tokens; the windowed trim path must
    # still agree with the full oracle for every candidate swap.
    texts = ["Kharoshthi 𐨀𐨁𐨂 sample", "emoji 🎉🔥 mixed run", "ascii baseline text here"]
    seg_tokens = _seg_tokens(texts, gemma_tok, 4)
    checker = WindowedSeamChecker(gemma_tok)
    for a, b in itertools.combinations(range(len(texts)), 2):
        la, lb = len(seg_tokens[a]), len(seg_tokens[b])
        for k in range(min(la, lb) - 1):
            chain_a = [(a, j) for j in range(la)]
            chain_b = [(b, j) for j in range(lb)]
            windowed = swap_is_stable(checker, chain_a, chain_b, k, seg_tokens)
            a_new = tuple(tid for s, j in chain_a[:k + 1] + chain_b[k + 1:] for tid in seg_tokens[s][j])
            b_new = tuple(tid for s, j in chain_b[:k + 1] + chain_a[k + 1:] for tid in seg_tokens[s][j])
            oracle = is_roundtrip_stable(a_new, gemma_tok) and is_roundtrip_stable(b_new, gemma_tok)
            assert windowed == oracle, (a, b, k)
