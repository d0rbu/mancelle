"""Differential + property tests for the v2 splice records (issue #596).

The reference implementation here recomputes the would-have-followed mapping
from the raw provenance triples alone (the approach-B inversion), completely
independent of build_provenance's placement bookkeeping. Emitted records must
match it exactly on every randomized recombination.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.feature_shard_splice import (
    build_provenance,
    load_source_segments,
    recombination_seed,
    recombine_segments,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

_SEG = 2
_TOK = CharMergeTokenizer(bos_id=500)
_ALPHA = "abcdefghijklmnopqrst"

_corpus_st = st.lists(
    st.text(alphabet=_ALPHA, min_size=1, max_size=15), min_size=2, max_size=10,
)
_cut_st = st.sampled_from([0.15, 0.5, 0.9])
_seed_st = st.integers(min_value=0, max_value=999)


def recompute_splices_from_triples(prov: dict) -> dict[int, dict[int, dict]]:
    """Approach-B reference: {doc_id: {pos: record}} from raw triples only."""
    placement: dict[tuple[int, int], tuple[int, int]] = {}
    for rec in prov["docs"]:
        for p, (_, d, o) in enumerate(rec["provenance"]):
            placement[(d, o)] = (rec["doc_id"], p)
    out: dict[int, dict[int, dict]] = {}
    for rec in prov["docs"]:
        tri = rec["provenance"]
        doc_out: dict[int, dict] = {}
        for p in range(1, len(tri)):
            if (tri[p][1], tri[p][2]) == (tri[p - 1][1], tri[p - 1][2] + 1):
                continue
            whf = (tri[p - 1][1], tri[p - 1][2] + 1)
            assert whf in placement, f"existence invariant violated at doc {rec['doc_id']} pos {p}"
            derived = placement[whf]
            doc_out[p] = {
                "pos": p,
                "whf_base": [prov["shard_index"], whf[0], whf[1]],
                "whf_derived": [derived[0], derived[1]],
            }
        out[rec["doc_id"]] = doc_out
    return out


def _prov_for(texts: list[str], *, cut_fraction: float, seed: int) -> dict:
    shard = make_base_shard(texts, _TOK, _SEG)
    segs, _stats = load_source_segments(
        shard, texts, _TOK, segment_size=_SEG, context_window=2048,
        sae_max_context=None, shard_name="shard_00007",
    )
    shard_seed = recombination_seed(seed, 7, cut_fraction)
    result = recombine_segments(
        segs, _TOK, cut_fraction=cut_fraction, shard_seed=shard_seed,
        shard_name="shard_00007", limit=2048,
    )
    return build_provenance(
        result, segs, params={}, shard_index=7,
        shard_name="shard_00007", shard_seed=shard_seed,
    )


@given(texts=_corpus_st, cut_fraction=_cut_st, seed=_seed_st)
@settings(max_examples=60, deadline=None)
def test_emitted_records_match_independent_recomputation(texts, cut_fraction, seed):
    prov = _prov_for(texts, cut_fraction=cut_fraction, seed=seed)
    expected = recompute_splices_from_triples(prov)
    for rec in prov["docs"]:
        got = {r["pos"]: r for r in rec["splices"]}
        assert got == expected[rec["doc_id"]]


@given(texts=_corpus_st, cut_fraction=_cut_st, seed=_seed_st)
@settings(max_examples=40, deadline=None)
def test_existence_invariant_across_randomized_recombinations(texts, cut_fraction, seed):
    prov = _prov_for(texts, cut_fraction=cut_fraction, seed=seed)
    all_refs = {
        (tri[1], tri[2]) for rec in prov["docs"] for tri in rec["provenance"]
    }
    for rec in prov["docs"]:
        for r in rec["splices"]:
            assert (r["whf_base"][1], r["whf_base"][2]) in all_refs


def test_strategy_reaches_splices():
    """Non-vacuity: a known corpus/seed produces at least one record and at
    least one SAME-DOC ordinal jump across a seed sweep."""
    texts = ["abcdefgh", "ijklmnop", "qrstabcd", "efghijkl", "mnopqrst"]
    any_records = False
    any_same_doc_jump = False
    for seed in range(40):
        prov = _prov_for(texts, cut_fraction=0.9, seed=seed)
        for rec in prov["docs"]:
            tri = rec["provenance"]
            for r in rec["splices"]:
                any_records = True
                p = r["pos"]
                if tri[p][1] == tri[p - 1][1]:
                    any_same_doc_jump = True
    assert any_records
    assert any_same_doc_jump, "seed sweep must exercise same-doc ordinal jumps"


def test_v2_minus_new_keys_is_exactly_v1_shape():
    """Strip splices + n_splice_positions, set schema_version to 1 ⇒ exactly
    the v1 key structure and ordering (byte-compat oracle at dict level)."""
    prov = _prov_for(["alpha beta", "gamma delta", "epsi zeta"], cut_fraction=0.6, seed=3)
    stripped = dict(prov)
    stripped["schema_version"] = 1
    stripped["params"] = {
        k: v for k, v in prov["params"].items() if k != "n_splice_positions"
    }
    stripped["docs"] = [
        {k: v for k, v in d.items() if k != "splices"} for d in prov["docs"]
    ]
    assert list(stripped.keys()) == [
        "schema_version", "shard_name", "shard_index", "shard_seed",
        "params", "n_docs", "docs",
    ]
    for d in stripped["docs"]:
        assert list(d.keys()) == [
            "doc_id", "seed_doc_id", "n_segments", "n_switches",
            "provenance", "segment_token_starts",
        ]
