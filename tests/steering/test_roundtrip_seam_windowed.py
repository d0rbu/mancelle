"""Unit tests for the windowed seam checker (issue #590 feature-shard splice).

The central correctness property: the cheap windowed seam check is EXACTLY
equivalent to the full-document round-trip oracle for every candidate tail-swap.
If the window is ever too small the gate would wrongly accept an unstable swap,
silently corrupting the derived corpus — so the differential tests below also
assert that the unstable branch is actually exercised (non-vacuous).
"""

from __future__ import annotations

import itertools

import pytest

from steering._roundtrip_seam import (
    _HF_WINDOW_GUARD_TOKENS,
    _NON_HF_WINDOW_SIDE_TOKENS,
    WindowedSeamChecker,
    _infer_window_side_tokens,
    is_roundtrip_stable,
)
from tests.steering._seam_test_helpers import swap_is_stable
from tests.steering._splice_test_tokenizers import CharMergeTokenizer


def _seg_tokens(texts, tokenizer, segment_size):
    """[doc][seg] tuples mirroring how the splicer segments re-tokenized docs."""
    out = []
    for text in texts:
        ids = tokenizer(text, padding=False)["input_ids"]
        out.append([tuple(ids[i : i + segment_size]) for i in range(0, len(ids), segment_size)])
    return out


def _chain_ids(chain, seg_tokens):
    return tuple(tid for s, j in chain for tid in seg_tokens[s][j])


def _assert_windowed_equals_oracle(tokenizer, texts, segment_size):
    """For every candidate tail-swap, windowed == oracle. Returns (#checked, #unstable)."""
    seg_tokens = _seg_tokens(texts, tokenizer, segment_size)
    checker = WindowedSeamChecker(tokenizer)
    n = len(texts)
    checked = unstable = 0
    for a, b in itertools.combinations(range(n), 2):
        la, lb = len(seg_tokens[a]), len(seg_tokens[b])
        for k in range(min(la, lb) - 1):
            chain_a = [(a, j) for j in range(la)]
            chain_b = [(b, j) for j in range(lb)]
            windowed = swap_is_stable(checker, chain_a, chain_b, k, seg_tokens)
            a_new = _chain_ids(chain_a[: k + 1] + chain_b[k + 1 :], seg_tokens)
            b_new = _chain_ids(chain_b[: k + 1] + chain_a[k + 1 :], seg_tokens)
            oracle = is_roundtrip_stable(a_new, tokenizer) and is_roundtrip_stable(b_new, tokenizer)
            assert windowed == oracle, (a, b, k, "windowed", windowed, "oracle", oracle)
            checked += 1
            unstable += not oracle
    return checked, unstable


def test_windowed_equals_oracle_merge_tokenizer_nonhf():
    # 'ab' merges. With BOS + segment_size=2, a doc "az" segments to
    # [(bos,a),(z,)] (seg 0 ends with 'a') and "xb" to [(bos,x),(b,)] (seg 1
    # starts with 'b'); swapping their tails at k=0 forms the text "ab", which
    # re-tokenizes to the merge id -> an unstable seam the gate must reject.
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    texts = ["az", "xb", "ay", "yb", "aq", "stuv"]
    checked, unstable = _assert_windowed_equals_oracle(tok, texts, segment_size=2)
    assert checked > 0
    assert unstable > 0  # the rejection branch was actually exercised


def test_windowed_equals_oracle_hf_inference_tokenizer():
    # expose_vocab=True -> HF window-side inference path (decode the vocab).
    tok = CharMergeTokenizer(
        bos_id=200, merge_pairs=[("a", "b"), ("c", "d")], merge_id_base=128, expose_vocab=True
    )
    texts = ["aaacc", "bbbdd", "acbd ac", "cab dba", "xaybzc"]
    checked, unstable = _assert_windowed_equals_oracle(tok, texts, segment_size=2)
    assert checked > 0 and unstable > 0


def test_window_side_tokens_inferred_from_vocab_bytes():
    # Max decoded piece is a 2-char merge ('ab') = 2 UTF-8 bytes -> side = 2 + guard.
    tok = CharMergeTokenizer(merge_pairs=[("a", "b")], merge_id_base=128, expose_vocab=True)
    assert _infer_window_side_tokens(tok) == 2 + _HF_WINDOW_GUARD_TOKENS


def test_window_side_tokens_fixed_for_non_hf_tokenizer():
    tok = CharMergeTokenizer(merge_pairs=[("a", "b")])  # no vocab_size -> non-HF
    assert _infer_window_side_tokens(tok) == _NON_HF_WINDOW_SIDE_TOKENS


def test_swap_is_stable_false_on_known_merge_seam():
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    # doc0 "az" -> [bos, a, z] -> segs [(bos,a),(z,)]; seg 0 ENDS with 'a'.
    # doc1 "xb" -> [bos, x, b] -> segs [(bos,x),(b,)]; seg 1 STARTS with 'b'.
    seg_tokens = _seg_tokens(["az", "xb"], tok, segment_size=2)
    chain_a = [(0, j) for j in range(len(seg_tokens[0]))]
    chain_b = [(1, j) for j in range(len(seg_tokens[1]))]
    # swap at k=0: a_new = [(bos,a)] + [(b,)] -> text "ab" -> merge -> unstable.
    checker = WindowedSeamChecker(tok)
    stable = swap_is_stable(checker, chain_a, chain_b, 0, seg_tokens)
    assert stable is False
    # the symmetric chain b_new = [(bos,x)] + [(z,)] -> "xz" is itself stable,
    # so the rejection is driven purely by the a-side seam.



def test_windowed_localizes_when_doc_far_exceeds_window():
    # Prove the bounded window is the thing being tested (not the whole doc):
    # two ~26-token docs whose only interaction is an 'ab' seam in the middle.
    # window side = 2+8 = 10, doubled to 20 < 26 tokens, so the long stable 'z'
    # runs are excluded yet windowed still equals the full-document oracle.
    tok = CharMergeTokenizer(merge_pairs=[("a", "b")], merge_id_base=128, expose_vocab=True)
    # No BOS here, so char index == token index. seg j == tokens[2j:2j+2].
    doc_a = "z" * 21 + "a" + "z" * 4   # token 21 == 'a' -> ENDS seg 10; tails follow
    doc_b = "z" * 22 + "b" + "z" * 3   # token 22 == 'b' -> STARTS seg 11
    seg_tokens = _seg_tokens([doc_a, doc_b], tok, segment_size=2)
    checker = WindowedSeamChecker(tok)
    side = checker.window_side_tokens
    doc_a_tokens = sum(len(s) for s in seg_tokens[0])
    assert doc_a_tokens > side * 2  # window genuinely smaller than the doc
    chain_a = [(0, j) for j in range(len(seg_tokens[0]))]
    chain_b = [(1, j) for j in range(len(seg_tokens[1]))]
    k = 10
    windowed = swap_is_stable(checker, chain_a, chain_b, k, seg_tokens)
    a_new = _chain_ids(chain_a[: k + 1] + chain_b[k + 1 :], seg_tokens)
    b_new = _chain_ids(chain_b[: k + 1] + chain_a[k + 1 :], seg_tokens)
    oracle = is_roundtrip_stable(a_new, tok) and is_roundtrip_stable(b_new, tok)
    assert oracle is False  # the 'ab' seam is unstable
    assert windowed == oracle  # localized window agrees with the full oracle


@pytest.mark.slow
def test_windowed_equals_oracle_on_real_gemma(gemma_tok):
    # Real merges + byte-fallback at seams; the window must still equal the oracle.
    texts = [
        "The unhappy customer left a review.",
        "happily ever after, the end.",
        "café résumé naïve façade",
        "tokenization boundaries 12345",
        "mixed 日本語 text and ascii",
    ]
    checked, unstable = _assert_windowed_equals_oracle(gemma_tok, texts, segment_size=4)
    assert checked > 0  # equivalence held across every candidate swap
