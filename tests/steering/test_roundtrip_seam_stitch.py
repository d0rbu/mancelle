"""stitch_is_stable: windowed verdict for one junction == full-doc oracle.

Given the FULL true document context on both sides, the windowed check must
agree with is_roundtrip_stable on the concatenation, at any junction index.
"""

from __future__ import annotations

import itertools

import pytest

from steering._roundtrip_seam import WindowedSeamChecker, is_roundtrip_stable
from tests.steering._splice_test_tokenizers import CharMergeTokenizer


def _ids(tok, text):
    enc = tok(text, padding=False)["input_ids"]
    return tuple(enc[0] if enc and isinstance(enc[0], list) else enc)


@pytest.mark.parametrize("expose_vocab", [False, True])
def test_stitch_equals_oracle_any_junction(expose_vocab):
    tok = CharMergeTokenizer(
        bos_id=500, merge_pairs=[("a", "b"), ("c", "d")],
        merge_id_base=128, expose_vocab=expose_vocab,
    )
    texts = ["azcq", "xbdy", "aycb", "qqab", "ccaa", "bdbd"]
    docs = [_ids(tok, t) for t in texts]
    checker = WindowedSeamChecker(tok)
    checked = unstable = 0
    for da, db in itertools.permutations(range(len(docs)), 2):
        for i in range(1, len(docs[da])):          # any prefix cut of doc a
            for j in range(1, len(docs[db])):      # any suffix start of doc b
                prefix, suffix = docs[da][:i], docs[db][j:]
                joined = prefix + suffix
                windowed = checker.stitch_is_stable(prefix, suffix)
                oracle = is_roundtrip_stable(joined, tok)
                assert windowed == oracle, (da, i, db, j, windowed, oracle)
                checked += 1
                unstable += not oracle
    assert checked > 0 and unstable > 0   # both branches exercised


def test_stitch_handles_short_contexts():
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    # 1-token prefix/suffix: window == whole side; must not crash, must match oracle.
    a, b = _ids(tok, "a"), _ids(tok, "b")
    assert WindowedSeamChecker(tok).stitch_is_stable(a, b[1:]) == is_roundtrip_stable(a + b[1:], tok)


@pytest.mark.slow
def test_stitch_equals_oracle_real_gemma_differential(gemma_tok):
    """stitch_is_stable matches is_roundtrip_stable on real gemma junctions.

    Uses byte-fallback content to exercise edge-trim and unstable paths.
    Mirrors the corpus from test_windowed_equals_oracle_with_byte_fallback_text
    (test_roundtrip_seam_edge.py) plus a fourth text for broader coverage.
    """
    texts = [
        "Kharoshthi 𐨀𐨁𐨂 sample",
        "emoji 🎉🔥 mixed run",
        "ascii baseline text here",
        "café résumé naïve façade",
    ]
    checker = WindowedSeamChecker(gemma_tok)
    checked = unstable = 0
    for da, text_a in enumerate(texts):
        ids_a = _ids(gemma_tok, text_a)
        for db, text_b in enumerate(texts):
            if da == db:
                continue
            ids_b = _ids(gemma_tok, text_b)
            n_a, n_b = len(ids_a), len(ids_b)
            stride_a = max(1, n_a // 4)
            stride_b = max(1, n_b // 4)
            for i in range(1, n_a, stride_a):
                for j in range(0, n_b, stride_b):
                    suffix = ids_b[j:]
                    if not suffix:
                        continue
                    prefix = ids_a[:i]
                    windowed = checker.stitch_is_stable(prefix, suffix)
                    oracle = is_roundtrip_stable(prefix + suffix, gemma_tok)
                    assert windowed == oracle, (da, i, db, j)
                    checked += 1
                    unstable += not oracle
    assert checked > 0 and unstable > 0
