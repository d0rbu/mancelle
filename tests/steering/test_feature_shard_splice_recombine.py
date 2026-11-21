"""Tests for the batch select->derange->stitch recombination engine (#590, Task 6).

The engine replaces the old index-aligned tail-swap. It:

  * enumerates every splice point, seeds a shuffle, selects ``target_k`` cuts and a
    derangement over them,
  * pre-filters the k stitch seams with the windowed checker (redrawing from a pool
    on any unstable seam),
  * reads root chains + loops from the successor graph,
  * turns each loop into a fresh ``new_started`` document (BOS-led),
  * overflow-splits every piece to fit ``limit`` (remainders are ``new_started``),
  * and runs a final batched full-document oracle, redrawing the earliest stitched
    cut of any piece that still fails.

Output docs are ``DerivedDoc(segment_refs, new_started)`` in deterministic order:
root chains in root order (each followed by its overflow remainders), then loop
docs by lowest involved cut ref (each followed by its remainders).
"""

from __future__ import annotations

import random

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import steering.feature_shard_splice as _fsm_mod
from steering._roundtrip_seam import WindowedSeamChecker, is_roundtrip_stable
from steering.feature_shard_splice import (
    PoolExhaustedError,
    SpliceInternalError,
    UnsplittableChainError,
    _build_successors,
    _draw_cut_selection,
    _draw_derangement,
    _enumerate_splice_points,
    _piece_ids,
    _prefilter_seams,
    load_source_segments,
    recombination_seed,
    recombine_segments,
)
from steering.segmentation import resolve_bos_token_id
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_reference import reference_recombine
from tests.steering._splice_test_tokenizers import CharMergeTokenizer, SpanMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)  # no merges -> every stitch is stable
MERGE = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])  # 'a'|'b' seams merge

CORPUS = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta", "iota kappa"]


# ---------------------------------------------------------------------------
# Helpers — build source segments the production way (tokenize WITH BOS, tile).
# ---------------------------------------------------------------------------


def _segments(texts, *, tok=TOK, segment_size=2, context_window=2048, shard_name="shard_00007"):
    shard = make_base_shard(texts, tok, segment_size, context_window=context_window)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=segment_size, context_window=context_window,
        sae_max_context=None, shard_name=shard_name,
    )
    return segs


def _recombine(texts, *, cut_fraction, seed=42, tok=TOK, segment_size=2, limit=2048,
               shard_index=7):
    segs = _segments(texts, tok=tok, segment_size=segment_size)
    res = recombine_segments(
        segs, tok, cut_fraction=cut_fraction,
        shard_seed=recombination_seed(seed, shard_index, cut_fraction),
        shard_name=f"shard_{shard_index:05d}", limit=limit,
    )
    return segs, res


def _seg_tokens(segs):
    return [[s.tokens for s in doc] for doc in segs]


def _all_refs(res):
    return [ref for doc in res.docs for ref in doc.segment_refs]


def _source_refs(segs):
    return [(d, j) for d, doc in enumerate(segs) for j in range(len(doc))]


def _assert_bijection(res, segs):
    placed = _all_refs(res)
    source = _source_refs(segs)
    assert sorted(placed) == sorted(source)   # every source segment placed once
    assert len(placed) == len(source)         # no reuse, no loss


def _assert_oracle(res, segs, tok):
    """Every emitted doc's reconstructed ids round-trip (the §3.2 invariant)."""
    seg_tokens = _seg_tokens(segs)
    bos_id = resolve_bos_token_id(tok)
    for doc in res.docs:
        ids = _piece_ids(list(doc.segment_refs), doc.new_started, seg_tokens, bos_id)  # ty: ignore[invalid-argument-type]
        assert is_roundtrip_stable(ids, tok), (doc.segment_refs, doc.new_started)


def _has_any_index_graft(res):
    """True iff some output doc holds a ref at output position != its src ordinal."""
    return any(
        ref[1] != pos
        for doc in res.docs
        for pos, ref in enumerate(doc.segment_refs)
    )


# ---------------------------------------------------------------------------
# Identity short-circuit (target_k < 2, before any RNG draw).
# ---------------------------------------------------------------------------


def test_identity_when_cut_fraction_zero():
    segs, res = _recombine(CORPUS, cut_fraction=0.0)
    assert (res.n_cuts_final, res.n_redrawn, res.n_loops, res.n_splits) == (0, 0, 0, 0)
    assert len(res.docs) == len(segs)
    for d, doc in enumerate(res.docs):
        assert doc.new_started is False
        assert doc.segment_refs == tuple((d, j) for j in range(len(segs[d])))


def test_identity_when_target_k_is_one():
    # cut_fraction = 1/n_points -> round(n_points * 1/n_points) = round(1.0) = 1,
    # so target_k == 1 < 2 -> identity short-circuit (distinct from cut_fraction=0).
    texts = ["abc de", "fg"]
    segs = _segments(texts)
    n_points = sum(len(doc) - 1 for doc in segs)
    assert n_points >= 2, segs  # a genuine multi-cut corpus reduced to target_k == 1
    cut_fraction = 1.0 / n_points
    segs2, res = _recombine(texts, cut_fraction=cut_fraction)
    assert (res.n_cuts_final, res.n_redrawn, res.n_loops, res.n_splits) == (0, 0, 0, 0)
    for d, doc in enumerate(res.docs):
        assert doc.new_started is False
        assert doc.segment_refs == tuple((d, j) for j in range(len(segs2[d])))


# ---------------------------------------------------------------------------
# Golden: fixed corpus + seed pins the exact RNG contract (run once, hardcode).
# ---------------------------------------------------------------------------

# 6 docs of 3-5 segments (segment_size=2, BOS-led). Distinct content so the
# hardcoded refs stay legible. Seg counts: [3, 3, 4, 4, 5, 5].
GOLDEN_CORPUS = [
    "cats",       # doc 0 -> 3 segs
    "zebra",      # doc 1 -> 3 segs
    "dolphin",    # doc 2 -> 4 segs
    "penguin",    # doc 3 -> 4 segs
    "elephant",   # doc 4 -> 5 segs
    "kangaroos",  # doc 5 -> 5 segs
]
GOLDEN_SEED = 42
# Pinned from a single verified run (this test's own asserts checked bijection +
# any-index grafts + the full oracle before this list was frozen). This is THE
# RNG-contract pin: any change to the shuffle / derangement / redraw / stitch
# order flips it. Seeded by recombination_seed(42, 7, 0.5); no loops for this seed.
GOLDEN_EXPECTED: list[tuple[tuple[tuple[int, int], ...], bool]] = [
    (((0, 0), (5, 3), (5, 4)), False),
    (((1, 0), (1, 1), (3, 1), (3, 2), (3, 3)), False),
    (((2, 0), (0, 1), (0, 2)), False),
    (((3, 0), (4, 3), (2, 1), (1, 2)), False),
    (((4, 0), (2, 2), (2, 3)), False),
    (((5, 0), (5, 1), (5, 2), (4, 1), (4, 2), (4, 4)), False),
]


def test_golden_small_shard():
    segs, res = _recombine(GOLDEN_CORPUS, cut_fraction=0.5, seed=GOLDEN_SEED)

    # -- invariants verified BY CODE (the eyeball sanity checks) --------------
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, TOK)
    assert _has_any_index_graft(res), "golden must exercise an any-index graft"
    assert res.n_redrawn == 0, "no-merge tokenizer: the pre-filter never redraws"

    # -- exact RNG-contract pin ----------------------------------------------
    actual = [(doc.segment_refs, doc.new_started) for doc in res.docs]
    assert actual == GOLDEN_EXPECTED


# ---------------------------------------------------------------------------
# Redraw path: a single merge-unstable stitch is redrawn from the pool front.
# ---------------------------------------------------------------------------

# A merge corpus + seed for which exactly one stitch in the seeded selection is
# 'a'|'b'-unstable, resolved by one pool redraw (found by the sweep below, pinned).
REDRAW_CORPUS = ["az by", "bx ya", "cz dy", "wa vb"]
REDRAW_SEED = 23  # pinned: the seeded selection has one 'a'|'b' seam, fixed by 1 redraw


def test_redraw_on_unstable_seam():
    segs, res = _recombine(REDRAW_CORPUS, cut_fraction=0.5, seed=REDRAW_SEED, tok=MERGE)
    assert res.n_redrawn == 1
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, MERGE)  # the redraw produced an all-stable result


# ---------------------------------------------------------------------------
# Pool exhaustion: every redraw target is unstable / the pool is empty.
# ---------------------------------------------------------------------------


def test_pool_exhaustion_raises():
    # cut_fraction=1.0 cuts BOTH splice points (empty redraw pool). The 2-cut
    # derangement is [1,0], so one stitch is always (BOS,a)|(b) -> 'ab' merges ->
    # unstable -> the pre-filter redraws -> empty pool -> PoolExhaustedError.
    segs = _segments(["az", "xb"], tok=MERGE)
    with pytest.raises(PoolExhaustedError):
        recombine_segments(
            segs, MERGE, cut_fraction=1.0, shard_seed=recombination_seed(1, 0, 1.0),
            shard_name="shard_00000", limit=2048,
        )


# ---------------------------------------------------------------------------
# Any-index graft: guards against an index-alignment regression.
# ---------------------------------------------------------------------------


def test_any_index_graft_occurs():
    _segs, res = _recombine(GOLDEN_CORPUS, cut_fraction=0.5, seed=GOLDEN_SEED)
    assert _has_any_index_graft(res), (
        "some output doc must hold a ref whose src ordinal != its output position"
    )


# ---------------------------------------------------------------------------
# Loops: a cycle in the successor graph becomes a fresh new_started document.
# ---------------------------------------------------------------------------

# A single 5-segment doc + a seed whose selection forms exactly one loop (pinned).
LOOP_CORPUS = ["abcdefghi"]  # 1 doc, 5 segments (BOS + 9 chars, segment_size=2)
LOOP_CUT_FRACTION = 0.5
LOOP_SEED = 0  # pinned: this selection's derangement produces exactly one loop


def test_loop_becomes_new_doc():
    segs, res = _recombine(LOOP_CORPUS, cut_fraction=LOOP_CUT_FRACTION, seed=LOOP_SEED)
    assert res.n_loops == 1
    new_started = [doc for doc in res.docs if doc.new_started]
    assert len(new_started) == 1                    # the loop became one new doc
    assert len(res.docs) == 2                        # 1 root chain + 1 loop doc
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, TOK)


# ---------------------------------------------------------------------------
# Overflow: a small limit forces splits; remainders are new_started.
# ---------------------------------------------------------------------------


def test_overflow_produces_new_started_remainder():
    # Long docs + a small (segment-size-multiple) limit force overflow splits.
    texts = ["aaaa bbbb cccc", "dddd eeee ffff", "gggg hhhh iiii"]
    segs, res = _recombine(texts, cut_fraction=0.5, seed=3, limit=6)
    assert res.n_splits > 0, "the small limit must force at least one split"
    seg_tokens = _seg_tokens(segs)
    bos_id = resolve_bos_token_id(TOK)
    for doc in res.docs:
        ids = _piece_ids(list(doc.segment_refs), doc.new_started, seg_tokens, bos_id)  # ty: ignore[invalid-argument-type]
        assert len(ids) <= 6, (doc.segment_refs, doc.new_started, len(ids))
    # every overflow remainder is a fresh BOS-led document
    assert any(doc.new_started for doc in res.docs)
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, TOK)


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_determinism_two_runs_identical():
    _s1, r1 = _recombine(CORPUS, cut_fraction=0.6, seed=123)
    _s2, r2 = _recombine(CORPUS, cut_fraction=0.6, seed=123)
    assert [(d.segment_refs, d.new_started) for d in r1.docs] == \
           [(d.segment_refs, d.new_started) for d in r2.docs]
    assert (r1.n_cuts_final, r1.n_redrawn, r1.n_loops, r1.n_splits) == \
           (r2.n_cuts_final, r2.n_redrawn, r2.n_loops, r2.n_splits)


# ---------------------------------------------------------------------------
# The §3.2 invariant: every emitted doc passes the full round-trip oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut_fraction", [0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_all_output_docs_pass_full_oracle(cut_fraction, seed):
    segs, res = _recombine(CORPUS, cut_fraction=cut_fraction, seed=seed)
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, TOK)


# ---------------------------------------------------------------------------
# FIX-15: prefilter memo reduces stitch_is_stable calls across iterations.
# ---------------------------------------------------------------------------


class _CountingChecker:
    """Wraps WindowedSeamChecker and counts stitch_is_stable invocations."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.call_count = 0

    def stitch_is_stable(self, prefix, suffix):
        self.call_count += 1
        return self._wrapped.stitch_is_stable(prefix, suffix)

    @property
    def window_side_tokens(self):
        return self._wrapped.window_side_tokens


def test_prefilter_memo_reduces_stitch_calls_on_second_run():
    """A memo filled on the first _prefilter_seams call yields 0 checker calls on
    the second call with the same cuts/sigma — simulating a fixpoint restart where
    no cuts were changed.  With no memo, the second call would make k calls."""
    tok = CharMergeTokenizer(bos_id=500)  # no merges — all seams stable
    texts = CORPUS  # 5 docs, each with multiple segments
    shard = make_base_shard(texts, tok, segment_size=2, context_window=2048)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=2, context_window=2048,
        sae_max_context=None, shard_name="shard_00007",
    )
    seg_tokens = [[s.tokens for s in doc] for doc in segs]
    bos_id = resolve_bos_token_id(tok)

    rng = random.Random(42)
    all_sp = _enumerate_splice_points(segs)
    target_k = max(2, round(0.5 * len(all_sp)))
    sel_idx, _pool = _draw_cut_selection(rng, len(all_sp), target_k)
    sigma = _draw_derangement(rng, target_k)
    cuts = [all_sp[i] for i in sel_idx]

    # build successors to ensure cuts form valid graph (no PoolExhaustedError)
    _ = _build_successors(segs, cuts, sigma)

    real_checker = WindowedSeamChecker(tok)
    counting_checker = _CountingChecker(real_checker)
    memo: dict = {}

    def _noop_replace(slot):
        pass  # no redraws expected since tok has no merges

    # First call: fills memo with k entries (k unique seam (prefix,suffix) pairs).
    _prefilter_seams(cuts, sigma, seg_tokens, counting_checker, _noop_replace, bos_id, memo)  # ty: ignore[invalid-argument-type]
    calls_first = counting_checker.call_count
    assert calls_first == target_k, (
        f"first call should make exactly k={target_k} stitch_is_stable calls, got {calls_first}"
    )
    assert len(memo) == target_k, f"memo should have {target_k} entries after first call"

    # Reset counter for second call.
    counting_checker.call_count = 0

    # Second call with same cuts/sigma/memo: all seams hit the memo -> 0 checker calls.
    _prefilter_seams(cuts, sigma, seg_tokens, counting_checker, _noop_replace, bos_id, memo)  # ty: ignore[invalid-argument-type]
    calls_second = counting_checker.call_count
    assert calls_second == 0, (
        f"second call with same cuts should make 0 stitch_is_stable calls (all memo hits), "
        f"got {calls_second}"
    )
    assert calls_second < calls_first, "memo must reduce calls vs first run"

    # Also verify that recombine_segments (which owns the memo) produces deterministic
    # output — the memo does not change verdicts.
    segs2, res1 = _recombine(texts, cut_fraction=0.5, seed=42)
    segs3, res2 = _recombine(texts, cut_fraction=0.5, seed=42)
    assert [(d.segment_refs, d.new_started) for d in res1.docs] == \
           [(d.segment_refs, d.new_started) for d in res2.docs], \
           "recombine_segments must be deterministic with memo enabled"


# ---------------------------------------------------------------------------
# ITEM-3: Memo hardening tests
# ---------------------------------------------------------------------------

# Shared hypothesis strategies for the memo differential tests.
_TEXT_ST = st.text(alphabet="abcdefghijklmnopqrst", min_size=1, max_size=8)
_CORPUS_ST = st.lists(_TEXT_ST, min_size=2, max_size=6)
_CF_ST = st.sampled_from([0.3, 0.5, 0.8])
_SEED_ST = st.integers(0, 99)
_LIMIT_ST = st.one_of(st.just(2048), st.integers(3, 18))


@given(
    texts=_CORPUS_ST, cut_fraction=_CF_ST, seed=_SEED_ST,
    use_merges=st.booleans(), limit=_LIMIT_ST,
)
@settings(max_examples=30, deadline=None)
def test_memo_vs_no_memo_identical_result(texts, cut_fraction, seed, use_merges, limit):
    """Bypassing the memo from the test produces an identical ShardSpliceResult.

    Spans corpora that exercise: merge-triggered redraws (use_merges=True),
    overflow splits (small limit), and loop formation (multi-segment docs).
    Because the memo only caches pure functions of the (prefix, suffix) key,
    enabling or disabling it must never change the final docs or counters.
    """
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")] if use_merges else [])
    shard = make_base_shard(texts, tok, segment_size=2, context_window=2048)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=2, context_window=2048,
        sae_max_context=None, shard_name="hyp_memo",
    )
    # Require at least 2 splice points so real recombination is exercised.
    assume(sum(len(doc) - 1 for doc in segs) >= 2)

    shard_seed = recombination_seed(seed, 0, cut_fraction)
    kwargs = {"cut_fraction": cut_fraction, "shard_seed": shard_seed, "shard_name": "hyp_memo", "limit": limit}

    # Run with memo enabled (default).
    try:
        res_with = recombine_segments(segs, tok, **kwargs)  # ty: ignore[invalid-argument-type]
    except (PoolExhaustedError, UnsplittableChainError):
        return  # legal terminal state; skip this sample

    # Keep memo suppression in the test: replace the production seam helper
    # with a wrapper that discards only its optional memo argument.
    real_prefilter = _fsm_mod._prefilter_seams

    def prefilter_without_memo(
        cuts, sigma, seg_tokens, checker, replace_fn, bos_id, _memo=None,
    ):
        return real_prefilter(
            cuts, sigma, seg_tokens, checker, replace_fn, bos_id, memo=None,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_fsm_mod, "_prefilter_seams", prefilter_without_memo)
        try:
            res_without = recombine_segments(segs, tok, **kwargs)  # ty: ignore[invalid-argument-type]
        except (PoolExhaustedError, UnsplittableChainError):
            return  # legal terminal state; skip this sample

    assert [(d.segment_refs, d.new_started) for d in res_with.docs] == \
           [(d.segment_refs, d.new_started) for d in res_without.docs], \
           "memo-enabled and memo-disabled runs must produce identical docs"
    assert (res_with.n_cuts_final, res_with.n_redrawn, res_with.n_loops, res_with.n_splits) == \
           (res_without.n_cuts_final, res_without.n_redrawn, res_without.n_loops, res_without.n_splits), \
           "memo-enabled and memo-disabled runs must produce identical counters"


def test_memo_content_matches_fresh_checker():
    """Every memo entry, when re-checked with a fresh WindowedSeamChecker, gives the same verdict.

    Uses the merge tokenizer so some seams may produce False verdicts (merge-unstable).
    A mismatch would mean the stored verdict depends on something other than the
    (prefix, suffix) key — e.g. checker state or global context (which are both
    illegal by the purity contract).  The test covers all entries regardless of whether
    redraws occurred (a corpus where all seams happen to be stable still exercises the
    True-verdict purity contract).
    """
    tok = CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])
    # Use a larger corpus to increase the chance of at least one merge-unstable seam.
    texts = REDRAW_CORPUS + CORPUS
    shard = make_base_shard(texts, tok, segment_size=2, context_window=2048)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=2, context_window=2048,
        sae_max_context=None, shard_name="shard_00007",
    )
    seg_tokens = [[s.tokens for s in doc] for doc in segs]
    bos_id = resolve_bos_token_id(tok)

    rng = random.Random(REDRAW_SEED)
    all_sp = _enumerate_splice_points(segs)
    target_k = max(2, round(0.5 * len(all_sp)))
    sel_idx, pool = _draw_cut_selection(rng, len(all_sp), target_k)
    sigma = _draw_derangement(rng, target_k)
    cuts = [all_sp[i] for i in sel_idx]

    memo: dict = {}

    def _replace(slot: int) -> None:
        # Raise (not no-op) on an empty pool: a silent no-op would leave the seam
        # unstable forever and hang _prefilter_seams. A loud failure is hang-proof.
        assert pool, "redraw pool exhausted — _prefilter_seams would loop forever"
        cuts[slot] = all_sp[pool.popleft()]

    checker = WindowedSeamChecker(tok)
    _prefilter_seams(cuts, sigma, seg_tokens, checker, _replace, bos_id, memo)  # ty: ignore[invalid-argument-type]

    assert memo, "memo must be non-empty after _prefilter_seams"

    # Re-check every memo entry with a fresh checker and verify the verdict matches.
    # This proves the stored verdict is a pure function of the (prefix, suffix) key.
    fresh_checker = WindowedSeamChecker(tok)
    for (prefix, suffix), stored_verdict in memo.items():
        fresh_verdict = fresh_checker.stitch_is_stable(prefix, suffix)
        assert fresh_verdict == stored_verdict, (
            f"memo entry mismatch: stored={stored_verdict!r}, "
            f"fresh={fresh_verdict!r} for prefix={prefix!r}, suffix={suffix!r}"
        )


def test_memo_no_cross_call_leakage():
    """Two back-to-back _prefilter_seams calls with separate memos see identical fresh-call counts.

    Simulates two consecutive recombine_segments calls: each creates its own memo
    dict (fresh ``{}``).  If memo entries from call 1 leaked into call 2, the
    counting checker would see fewer invocations in call 2 (memo hits would skip them).
    With separate dicts, call 2 must make the same number of invocations as call 1.
    """
    tok = CharMergeTokenizer(bos_id=500)  # no merges — all seams stable
    texts = CORPUS
    shard = make_base_shard(texts, tok, segment_size=2, context_window=2048)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=2, context_window=2048,
        sae_max_context=None, shard_name="shard_00007",
    )
    seg_tokens = [[s.tokens for s in doc] for doc in segs]
    bos_id = resolve_bos_token_id(tok)

    rng = random.Random(42)
    all_sp = _enumerate_splice_points(segs)
    target_k = max(2, round(0.5 * len(all_sp)))
    sel_idx, _pool = _draw_cut_selection(rng, len(all_sp), target_k)
    sigma = _draw_derangement(rng, target_k)
    cuts = [all_sp[i] for i in sel_idx]

    real_checker = WindowedSeamChecker(tok)

    def _noop(slot: int) -> None:
        pass

    # Call 1: fresh memo dict (as recombine_segments would create).
    memo1: dict = {}
    counter1 = _CountingChecker(real_checker)
    _prefilter_seams(cuts, sigma, seg_tokens, counter1, _noop, bos_id, memo1)  # ty: ignore[invalid-argument-type]
    calls1 = counter1.call_count

    # Call 2: a second, separate memo dict — simulates the next recombine_segments call.
    # Cuts are identical (same corpus / seed), so the same seams are checked.
    memo2: dict = {}
    counter2 = _CountingChecker(real_checker)
    _prefilter_seams(cuts, sigma, seg_tokens, counter2, _noop, bos_id, memo2)  # ty: ignore[invalid-argument-type]
    calls2 = counter2.call_count

    # Both calls must make the same number of invocations.  If memo1 had leaked into
    # memo2, calls2 would be 0 (all hits) rather than calls1 (all misses on fresh dict).
    assert calls2 == calls1, (
        f"call 2 made {calls2} invocations, call 1 made {calls1}; "
        f"they must be equal — any difference signals cross-call memo leakage"
    )
    assert calls2 > 0, "both calls must check at least one seam"
    assert calls2 == target_k, (
        f"expected {target_k} fresh invocations in call 2 (no memo hits), got {calls2}"
    )


# ---------------------------------------------------------------------------
# A. ENGINE RECOVERY PATHS — the rare full-fixpoint redraw branches.
#
# These are exercised by the run-local seam BLIND SPOT: a SpanMergeTokenizer span
# whose trigger straddles a whole middle run (``prev_tail + middle_run + next_head``)
# is invisible to every run-local seam window (the mid-run's out-stitch prefix is
# synthetic-BOS-led, masking the left context; its in-stitch misses the right
# context) yet the full-document oracle sees the assembled span. That gap drives:
#   A1 — Phase-E full-document oracle redraw (`_first_cut_slot_in_piece`),
#   A2 — unbreakable-loop culprit redraw (`_lowest_cut_slot_in_cycle`).
# Both corpora + seeds are pinned from a computational sweep (see the wave-5 report);
# each is proven against the independent reference (`reference_recombine`) so the
# RNG-contract and the recovery bookkeeping are both nailed down. A3 pins the
# BOS-less engine raise.
# ---------------------------------------------------------------------------


def _recovery_segs(corpus, tok):
    shard = make_base_shard(corpus, tok, 2, context_window=2048)
    segs, _ = load_source_segments(
        shard, corpus, tok, segment_size=2, context_window=2048,
        sae_max_context=None, shard_name="recovery",
    )
    return segs


def _assert_prod_equals_reference(res, segs, tok, *, cut_fraction, shard_seed, limit=2048):
    """The independent reference must reproduce prod's docs AND counters exactly.

    Removes the differential blind spot: any orchestration divergence on this
    recovery path (redraw ordering, piece ordering, counter attribution) would
    surface as a mismatch here even though the full oracle passes on both.
    """
    ref_docs, ref_counters = reference_recombine(
        segs, tok, cut_fraction=cut_fraction, shard_seed=shard_seed, limit=limit,
    )
    prod_docs = [(d.segment_refs, d.new_started) for d in res.docs]
    assert prod_docs == ref_docs, "prod docs must equal the reference docs exactly"
    assert ref_counters == {
        "n_cuts_final": res.n_cuts_final, "n_redrawn": res.n_redrawn,
        "n_loops": res.n_loops, "n_splits": res.n_splits,
    }, "prod counters must equal the reference counters exactly"


# A1 pin: span "pqrs" = p (prev_tail) + qr (a whole middle run) + s (next_head).
# Docs supply p-ending / "qr" / s-starting segments; each run-local seam sees only
# "..pqr" or "qrs.." (both stable) while some assembled doc contains "pqrs".
_PHASE_E_TOK = SpanMergeTokenizer(bos_id=500, spans=["pqrs"])
_PHASE_E_CORPUS = ["epqrfs", "gpXqrYsZ", "kpmqrnso", "apbqrcsd", "wpxqrysz", "pqXsrp"]
_PHASE_E_CF = 0.7
_PHASE_E_SEED = 56
_PHASE_E_SHARD_INDEX = 7


def test_phase_e_full_doc_oracle_redraw(monkeypatch):
    """A piece passing every run-local seam still fails the full-document oracle →
    Phase E redraws the earliest stitched cut → engine completes with a valid result.

    Proves the Phase-E branch actually fired by wrapping ``_first_cut_slot_in_piece``
    with a call-recording passthrough (delegating to the real function).
    """
    segs = _recovery_segs(_PHASE_E_CORPUS, _PHASE_E_TOK)
    shard_seed = recombination_seed(_PHASE_E_SEED, _PHASE_E_SHARD_INDEX, _PHASE_E_CF)

    real = _fsm_mod._first_cut_slot_in_piece
    fired = {"n": 0}

    def _recording(bad, cuts):
        fired["n"] += 1
        return real(bad, cuts)  # delegate to the real Phase-E cut mapping

    monkeypatch.setattr(_fsm_mod, "_first_cut_slot_in_piece", _recording)

    res = recombine_segments(
        segs, _PHASE_E_TOK, cut_fraction=_PHASE_E_CF, shard_seed=shard_seed,
        shard_name="recovery", limit=2048,
    )

    assert fired["n"] >= 1, "the Phase-E full-document oracle branch must have fired"
    assert res.n_redrawn >= 1, "a Phase-E failure must trigger at least one redraw"
    assert res.n_loops == 0, "this pin isolates Phase E (no loop culprit involved)"
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, _PHASE_E_TOK)
    _assert_prod_equals_reference(
        res, segs, _PHASE_E_TOK, cut_fraction=_PHASE_E_CF, shard_seed=shard_seed,
    )


def test_phase_e_pin_has_a_genuine_blind_spot():
    """Non-vacuity: the pinned corpus really does hide a full-doc failure from the
    windowed pre-filter. Encoding the *assembled* span 'pqrs' must merge (so the
    full oracle can fail) while its run-local fragments 'pqr' and 'qrs' must not —
    otherwise the pre-filter would have caught it and Phase E could never fire."""
    tok = _PHASE_E_TOK
    assert tok("pqrs")["input_ids"] == [500, 1000], "assembled span must merge to one id"
    assert tok("pqr")["input_ids"] == [500, ord("p"), ord("q"), ord("r")]
    assert tok("qrs")["input_ids"] == [500, ord("q"), ord("r"), ord("s")]


# A2 pin: three dormant covering spans over a 3-cycle of single-segment middle runs
# (AB, CD, EF). Each span straddles a whole run — "BCDE" = B + CD + E,
# "DEFA" = D + EF + A, "FABC" = F + AB + C — so EVERY rotation of the 3-loop has an
# interior run whose merge the run-local windows miss → `_break_loop` returns None →
# the loop's lowest cut slot is the culprit → the driver redraws it and completes.
_LOOP_TOK = SpanMergeTokenizer(bos_id=500, spans=["BCDE", "DEFA", "FABC"])
_LOOP_CORPUS = ["wAB12x", "yCD45z", "kEF78m"]
_LOOP_CF = 0.8
_LOOP_SEED = 1355
_LOOP_SHARD_INDEX = 7


def test_loop_culprit_redraw(monkeypatch):
    """An unbreakable loop (every rotation fails the ``(bos,)+ids`` oracle) makes
    ``_resolve_loops`` return the loop's lowest involved cut slot as the culprit; the
    driver redraws it and the engine completes with a valid result.

    Proves the culprit path fired by wrapping BOTH ``_lowest_cut_slot_in_cycle``
    (the culprit selector) and ``_break_loop`` (must have returned None at least once).
    """
    segs = _recovery_segs(_LOOP_CORPUS, _LOOP_TOK)
    shard_seed = recombination_seed(_LOOP_SEED, _LOOP_SHARD_INDEX, _LOOP_CF)

    real_low = _fsm_mod._lowest_cut_slot_in_cycle
    real_break = _fsm_mod._break_loop
    fired = {"low": 0, "break_none": 0}

    def _recording_low(cyc, cuts):
        fired["low"] += 1
        return real_low(cyc, cuts)  # delegate to the real culprit selector

    def _recording_break(cyc, seg_tokens, tokenizer, bos_id):
        result = real_break(cyc, seg_tokens, tokenizer, bos_id)  # delegate
        if result is None:
            fired["break_none"] += 1
        return result

    monkeypatch.setattr(_fsm_mod, "_lowest_cut_slot_in_cycle", _recording_low)
    monkeypatch.setattr(_fsm_mod, "_break_loop", _recording_break)

    res = recombine_segments(
        segs, _LOOP_TOK, cut_fraction=_LOOP_CF, shard_seed=shard_seed,
        shard_name="recovery", limit=2048,
    )

    assert fired["break_none"] >= 1, "a loop must have been unbreakable (all rotations failed)"
    assert fired["low"] >= 1, "the loop-culprit selector must have fired"
    assert res.n_redrawn >= 1, "the culprit must trigger at least one redraw"
    _assert_bijection(res, segs)
    _assert_oracle(res, segs, _LOOP_TOK)
    _assert_prod_equals_reference(
        res, segs, _LOOP_TOK, cut_fraction=_LOOP_CF, shard_seed=shard_seed,
    )


def test_loop_pin_covering_spans_are_dormant_in_sources():
    """Non-vacuity: the three covering spans are absent from every source doc (so
    the docs stay round-trip stable) yet each straddles a whole middle run when the
    3-cycle assembles. If any span appeared in a source doc, load would have raised
    and the loop could never form."""
    for span in ("BCDE", "DEFA", "FABC"):
        assert all(span not in doc for doc in _LOOP_CORPUS), f"{span} must be dormant"
    # And each span DOES collapse when assembled (proving the oracle can fail).
    assert _LOOP_TOK("BCDE")["input_ids"] == [500, 1000]


def test_bos_none_engine_raises():
    """A BOS-less tokenizer with target_k >= 2 cannot form new documents → the
    engine raises SpliceInternalError (past the identity short-circuit)."""
    tok = CharMergeTokenizer()  # bos_id=None -> resolve_bos_token_id is None
    assert resolve_bos_token_id(tok) is None
    segs = _recovery_segs(["abcdef", "ghijkl"], tok)
    # cut_fraction=1.0 -> target_k == n_splice_points >= 2, so the BOS check is reached.
    assert round(1.0 * len(_enumerate_splice_points(segs))) >= 2
    with pytest.raises(SpliceInternalError, match="tokenizer has no BOS"):
        recombine_segments(
            segs, tok, cut_fraction=1.0,
            shard_seed=recombination_seed(1, 0, 1.0),
            shard_name="shard_00000", limit=2048,
        )
