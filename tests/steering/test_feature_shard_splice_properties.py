"""Property tests for the recombination engine (Task 7, issue #590).

Includes a differential test against the independent reference implementation in
``tests.steering._splice_reference``. The TDD sequence was:
  RED:   write these tests first (reference module did not exist yet -> ImportError).
  GREEN: implement ``_splice_reference.py`` -> all tests pass.

Tests at a glance:
  test_differential_reference_matches_production  -- headline: ref == prod exactly
  test_bijection_property                         -- all source segs placed once
  test_bounded_length_property                    -- piece len <= limit (BOS counted)
  test_tiling_property                            -- interior refs always full-size
  test_realized_cut_fraction_tracks_target        -- n_cuts_final ~ round(cf*n_sp)
  test_full_oracle_property                       -- every piece round-trips
  test_determinism_property                       -- two identical seeds -> same output
  test_new_started_iff_not_source_seg0            -- new_started <-> first ref != (d,0)
"""

from __future__ import annotations

import random

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from steering._roundtrip_seam import is_roundtrip_stable
from steering.feature_shard_splice import (
    PoolExhaustedError,
    UnsplittableChainError,
    load_source_segments,
    recombination_seed,
    recombine_segments,
)
from steering.segmentation import resolve_bos_token_id
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_reference import reference_recombine
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

# ---------------------------------------------------------------------------
# Constants and tokenizer factories
# ---------------------------------------------------------------------------

_SEGMENT_SIZE = 2
_CONTEXT_WINDOW = 2048


def _tok() -> CharMergeTokenizer:
    """No-merge tokenizer: every stitch is stable -> no redraws."""
    return CharMergeTokenizer(bos_id=500)


def _tok_merge() -> CharMergeTokenizer:
    """Merge tokenizer: 'a'|'b' seam is unstable -> prefilter redraws."""
    return CharMergeTokenizer(bos_id=500, merge_pairs=[("a", "b")])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_segs(texts: list[str], tok: CharMergeTokenizer):
    """Build source segments the production way: tokenize WITH BOS, tile fixed-size."""
    shard = make_base_shard(texts, tok, _SEGMENT_SIZE, context_window=_CONTEXT_WINDOW)
    segs, _ = load_source_segments(
        shard, texts, tok,
        segment_size=_SEGMENT_SIZE,
        context_window=_CONTEXT_WINDOW,
        sae_max_context=None,
        shard_name="prop_test",
    )
    return segs


def _seg_tokens(segs):
    return [[s.tokens for s in doc] for doc in segs]


def _piece_ids_local(refs, new_started, seg_tokens, bos_id: int) -> tuple[int, ...]:
    """Local reimplementation of _piece_ids -- does NOT import the private helper."""
    ids = tuple(tid for ref in refs for tid in seg_tokens[ref[0]][ref[1]])
    return ((bos_id,) + ids) if new_started else ids


def _all_source_refs(segs):
    return [(d, j) for d, doc in enumerate(segs) for j in range(len(doc))]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Lowercase alphabet keeps tokens in a predictable range (ord 97-116),
# avoiding accidental merge triggers when use_merges=False.
_ALPHA = "abcdefghijklmnopqrst"
_text_st = st.text(
    alphabet=st.sampled_from(list(_ALPHA)),
    min_size=1,
    max_size=15,
)
_corpus_st = st.lists(_text_st, min_size=2, max_size=10)
_cut_fractions_st = st.sampled_from([0.15, 0.5, 0.9])
_seeds_st = st.integers(min_value=0, max_value=999)

# Limits: generous (2048) or small-to-force-splits (3..30 tokens).
_limit_st = st.one_of(
    st.just(2048),
    st.integers(min_value=3, max_value=30),
)


# ---------------------------------------------------------------------------
# Differential: reference must agree with production on every output
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
    use_merges=st.booleans(),
)
@settings(max_examples=60, deadline=None)
def test_differential_reference_matches_production(
    texts, cut_fraction, seed, limit, use_merges
):
    """Reference and production must agree exactly on docs and all four counters."""
    tok = _tok_merge() if use_merges else _tok()
    segs = _build_segs(texts, tok)
    shard_seed = recombination_seed(seed, 0, cut_fraction)

    # Non-vacuity guard: ensure real recombination is exercised.
    # The identity short-circuit fires when target_k < 2, so we require at least
    # 2 splice points (sum of doc lengths - 1 for each doc).
    assume(sum(len(doc) - 1 for doc in segs) >= 2)

    # --- production ---
    prod_docs = None
    prod_counters = None
    prod_exc: type | None = None
    try:
        res = recombine_segments(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            shard_name="prop_test",
            limit=limit,
        )
        prod_docs = [(doc.segment_refs, doc.new_started) for doc in res.docs]
        prod_counters = {
            "n_cuts_final": res.n_cuts_final,
            "n_redrawn": res.n_redrawn,
            "n_loops": res.n_loops,
            "n_splits": res.n_splits,
        }
    except (PoolExhaustedError, UnsplittableChainError) as exc:
        prod_exc = type(exc)

    # --- reference ---
    ref_docs = None
    ref_counters = None
    ref_exc: type | None = None
    try:
        ref_docs, ref_counters = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=limit,
        )
    except (PoolExhaustedError, UnsplittableChainError) as exc:
        ref_exc = type(exc)

    # Both must raise the same exception type or both succeed.
    assert prod_exc == ref_exc, (
        f"exception mismatch: prod={prod_exc}, ref={ref_exc}\n"
        f"texts={texts!r}, cut_fraction={cut_fraction}, seed={seed}, limit={limit}"
    )
    if prod_exc is not None:
        return  # both raised identically -- acceptable

    assert prod_docs == ref_docs, (
        f"docs mismatch:\n  prod={prod_docs}\n  ref={ref_docs}\n"
        f"texts={texts!r}, cut_fraction={cut_fraction}, seed={seed}, limit={limit}"
    )
    assert prod_counters == ref_counters, (
        f"counters mismatch:\n  prod={prod_counters}\n  ref={ref_counters}\n"
        f"texts={texts!r}, cut_fraction={cut_fraction}, seed={seed}, limit={limit}"
    )


# ---------------------------------------------------------------------------
# Recombination path exercise: verify the differential corpus strategy
# reaches interesting algorithm branches (non-identity, n_splits/n_loops)
# ---------------------------------------------------------------------------


def test_differential_strategy_exercises_recombination():
    """Fixed-seed corpus sweep proves differential strategy exercises recombination.

    The identity short-circuit (target_k < 2) must not dominate. This test runs
    ~20 fixed seeds over diverse corpus sizes and cut fractions, asserting:
      1. At least one run yields n_cuts_final > 0 (non-identity)
      2. At least one run yields n_splits > 0 or n_loops > 0 (algorithm branches)
    """
    seen_cuts = False
    seen_splits_or_loops = False

    for seed in range(20):
        rng = random.Random(seed)

        # Mirror the hypothesis strategy: 2-10 docs, 1-15 chars each
        n_docs = rng.randint(2, 10)
        texts = []
        for _ in range(n_docs):
            text_len = rng.randint(1, 15)
            text = "".join(rng.choice(_ALPHA) for _ in range(text_len))
            texts.append(text)

        cut_fraction = rng.choice([0.15, 0.5, 0.9])
        limit = rng.choice([2048, rng.randint(3, 30)])
        use_merges = rng.choice([True, False])

        tok = _tok_merge() if use_merges else _tok()
        segs = _build_segs(texts, tok)
        n_sp = sum(len(doc) - 1 for doc in segs)

        # Skip if not enough splice points for recombination
        if n_sp < 2:
            continue

        shard_seed = recombination_seed(seed, 0, cut_fraction)

        try:
            res = recombine_segments(
                segs, tok,
                cut_fraction=cut_fraction,
                shard_seed=shard_seed,
                shard_name="prop_test",
                limit=limit,
            )
        except (PoolExhaustedError, UnsplittableChainError):
            continue

        # Track whether we hit non-identity branch (n_cuts_final > 0)
        if res.n_cuts_final > 0:
            seen_cuts = True

        # Track whether we hit algorithm branches (splits or loops)
        if res.n_splits > 0 or res.n_loops > 0:
            seen_splits_or_loops = True

        # Early exit if both conditions met
        if seen_cuts and seen_splits_or_loops:
            break

    assert seen_cuts, (
        "differential corpus strategy never produced n_cuts_final > 0 "
        "(identity short-circuit dominated)"
    )
    assert seen_splits_or_loops, (
        "differential corpus strategy never produced n_splits > 0 or n_loops > 0 "
        "(algorithm branches not exercised)"
    )


# ---------------------------------------------------------------------------
# Bijection: every source segment appears exactly once
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
)
@settings(max_examples=40, deadline=None)
def test_bijection_property(texts, cut_fraction, seed, limit):
    tok = _tok()
    segs = _build_segs(texts, tok)
    shard_seed = recombination_seed(seed, 0, cut_fraction)
    try:
        ref_docs, _ = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=limit,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)  # skip this example
        return

    placed = [ref for refs, _ns in ref_docs for ref in refs]
    source = _all_source_refs(segs)

    assert sorted(placed) == sorted(source), "placed refs != source refs"
    assert len(placed) == len(source), "refs count mismatch"


# ---------------------------------------------------------------------------
# Bounded length: no piece exceeds limit tokens (BOS counted for new_started)
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
)
@settings(max_examples=40, deadline=None)
def test_bounded_length_property(texts, cut_fraction, seed, limit):
    """Every piece fits within limit tokens (BOS overhead counted for new_started pieces).

    Only tested when recombination actually occurs (target_k >= 2): the identity
    short-circuit (target_k < 2) returns source docs verbatim without consulting
    ``limit``, so the property is vacuous for that branch.
    """
    tok = _tok()
    segs = _build_segs(texts, tok)
    n_sp = sum(len(doc) - 1 for doc in segs)
    # Skip the identity branch: limit only constrains recombined pieces.
    assume(round(cut_fraction * n_sp) >= 2)
    seg_toks = _seg_tokens(segs)
    bos_id = resolve_bos_token_id(tok)
    assert bos_id is not None
    shard_seed = recombination_seed(seed, 0, cut_fraction)
    try:
        ref_docs, _ = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=limit,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    for refs, new_started in ref_docs:
        ids = _piece_ids_local(list(refs), new_started, seg_toks, bos_id)
        assert len(ids) <= limit, (
            f"piece {refs} (new_started={new_started}) has {len(ids)} tokens > limit={limit}"
        )


# ---------------------------------------------------------------------------
# Tiling: interior refs are always full-size; only the last ref may be partial
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
)
@settings(max_examples=40, deadline=None)
def test_tiling_property(texts, cut_fraction, seed):
    tok = _tok()
    segs = _build_segs(texts, tok)
    seg_toks = _seg_tokens(segs)
    shard_seed = recombination_seed(seed, 0, cut_fraction)

    # Need at least one multi-segment doc to derive the full segment size.
    full_size: int | None = None
    for tok_list in seg_toks:
        if len(tok_list) > 1:
            full_size = len(tok_list[0])
            break
    assume(full_size is not None)

    try:
        ref_docs, _ = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=2048,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    for refs, _ns in ref_docs:
        for i, ref in enumerate(refs[:-1]):  # every ref except the last
            d, j = ref
            actual_len = len(seg_toks[d][j])
            assert actual_len == full_size, (
                f"piece {refs}: interior ref {ref} (index {i}) has {actual_len} tokens "
                f"!= full_size={full_size}"
            )


# ---------------------------------------------------------------------------
# Realized cut fraction tracks target (statistical, merge-free, tolerance)
# ---------------------------------------------------------------------------


@given(
    texts=st.lists(_text_st, min_size=3, max_size=8),
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
)
@settings(max_examples=60, deadline=None)
def test_realized_cut_fraction_tracks_target(texts, cut_fraction, seed):
    """n_cuts_final / n_splice_points should be within 1/n_sp of cut_fraction.

    For a merge-free tokenizer (no redraws), n_cuts_final == round(cut_fraction * n_sp),
    so the relative error is at most 0.5 / n_sp (vanishes as the corpus grows).
    """
    tok = _tok()  # merge-free: no redraws, so n_cuts_final is always round(cf * n_sp)
    segs = _build_segs(texts, tok)
    n_sp = sum(len(doc) - 1 for doc in segs)
    assume(n_sp >= 2)  # need at least 2 splice points for recombination

    shard_seed = recombination_seed(seed, 0, cut_fraction)
    try:
        _, counters = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=2048,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    n_cuts = counters["n_cuts_final"]
    expected = round(cut_fraction * n_sp)
    # The identity short-circuit fires when target_k < 2; n_cuts_final = 0 in that
    # case regardless of the nominal target.  Only assert the round()-tracks-target
    # relationship when recombination actually runs (target_k >= 2).
    assume(expected >= 2)
    assert n_cuts == expected, (
        f"n_cuts_final={n_cuts} != round({cut_fraction}*{n_sp})={expected}"
    )
    # Relative error: always <= 0.5 / n_sp
    tolerance = 0.5 / n_sp + 1e-9
    realized = n_cuts / n_sp
    assert abs(realized - cut_fraction) <= tolerance, (
        f"realized fraction {realized:.4f} deviates from target {cut_fraction} by "
        f"{abs(realized - cut_fraction):.4f} > tolerance {tolerance:.4f}"
    )


# ---------------------------------------------------------------------------
# Full oracle: every emitted piece passes is_roundtrip_stable
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
)
@settings(max_examples=40, deadline=None)
def test_full_oracle_property(texts, cut_fraction, seed, limit):
    """Every piece's token ids round-trip through the tokenizer (§3.2 invariant)."""
    tok = _tok()
    segs = _build_segs(texts, tok)
    seg_toks = _seg_tokens(segs)
    bos_id = resolve_bos_token_id(tok)
    assert bos_id is not None
    shard_seed = recombination_seed(seed, 0, cut_fraction)
    try:
        ref_docs, _ = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=limit,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    for refs, new_started in ref_docs:
        ids = _piece_ids_local(list(refs), new_started, seg_toks, bos_id)
        assert is_roundtrip_stable(ids, tok), (
            f"piece {refs} (new_started={new_started}) failed round-trip"
        )


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical output on two runs
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
)
@settings(max_examples=30, deadline=None)
def test_determinism_property(texts, cut_fraction, seed, limit):
    tok = _tok()
    segs = _build_segs(texts, tok)
    shard_seed = recombination_seed(seed, 0, cut_fraction)
    kwargs = {"cut_fraction": cut_fraction, "shard_seed": shard_seed, "limit": limit}

    try:
        out1 = reference_recombine(segs, tok, **kwargs)
        out2 = reference_recombine(segs, tok, **kwargs)
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    assert out1 == out2, "two identical-seed runs produced different outputs"


# ---------------------------------------------------------------------------
# new_started iff first ref is not (d, 0) for any d
# ---------------------------------------------------------------------------


@given(
    texts=_corpus_st,
    cut_fraction=_cut_fractions_st,
    seed=_seeds_st,
    limit=_limit_st,
)
@settings(max_examples=40, deadline=None)
def test_new_started_iff_not_source_seg0(texts, cut_fraction, seed, limit):
    """A piece is new_started exactly when its first ref is not any (d, 0).

    Root chains start at (d, 0) with new_started=False. Loop docs and overflow
    remainders never start at (d, 0) (stitches target ordinal >= 1; root ordinals
    are exclusively served by root-walks and can never enter a cycle).
    """
    tok = _tok()
    segs = _build_segs(texts, tok)
    shard_seed = recombination_seed(seed, 0, cut_fraction)
    try:
        ref_docs, _ = reference_recombine(
            segs, tok,
            cut_fraction=cut_fraction,
            shard_seed=shard_seed,
            limit=limit,
        )
    except (PoolExhaustedError, UnsplittableChainError):
        assume(False)
        return

    for refs, new_started in ref_docs:
        first_ref = refs[0]
        is_root = first_ref[1] == 0  # (d, 0) for any d
        if new_started:
            assert not is_root, (
                f"new_started piece starts at {first_ref} which is a (d,0) root ref"
            )
        else:
            assert is_root, (
                f"non-new_started piece starts at {first_ref} which is not a (d,0) root ref"
            )
