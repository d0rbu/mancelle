"""Property tests for FeatureVocab (#695 PR A).

The example-based tests pin the cases I thought of. These hunt for the ones I didn't.

Every property here is a load-bearing claim from the spec:

* V1 — ``from_mask`` is a PURE function of ``(mask, n_sae)``. This is what lets the
  existing ``bl_``/``wl_`` checkpoint leaf pin the vocab with no new hash component.
* V2 — ``to_original(to_compact(id)) == id`` for every vocab id; every id OUTSIDE the
  vocab **RAISES**. (Issue #695's explicit round-trip requirement.)

  There is no sentinel and no opt-in to one: ``to_compact`` NEVER returns ``-1``, because
  ``-1`` silently reads the LAST embedding row under ``torch.compile``, so a caller handed
  one would condition on — and score — the WRONG SAE feature with no error anywhere. The
  two halves of V2 are therefore pinned as a BICONDITIONAL: ``to_compact`` raises **iff**
  some id in its input is out of vocab, and everything it returns is a real row of the
  table. That is strictly stronger than the pair of one-directional claims it replaces.

  The ``-1`` still exists in ``old_to_new_tensor()`` — the RAW map, which the encoder
  consumes mask-aware and guards with ``torch._assert_async``. The properties below pin it
  there, where it is guarded, and not in a public return value, where it would not be.
* ``select_rows`` is a BITWISE gather — the basis of slice-init byte-exactness (§1.4).
* ``old_to_new`` and ``new_to_old`` are exact inverses.
"""

from __future__ import annotations

import contextlib

import pytest
import torch as t
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from errors import FeatureVocabError
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def vocabs(draw, min_n=2, max_n=64):
    """A random non-identity, non-empty FeatureVocab."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    ids = draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), min_size=1, max_size=n, unique=True)
    )
    return FeatureVocab(n_sae_features=n, original_ids=tuple(sorted(ids)))


@st.composite
def masks_and_widths(draw, min_n=2, max_n=64):
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    mode = draw(st.sampled_from([FeatureMaskMode.blacklist, FeatureMaskMode.whitelist]))
    # A blacklist of EVERY id leaves an empty universe (rejected); keep >= 1 survivor.
    max_size = n - 1 if mode is FeatureMaskMode.blacklist else n
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=1,
            max_size=max(1, max_size),
            unique=True,
        )
    )
    assume(not (mode is FeatureMaskMode.blacklist and len(ids) >= n))
    return FeatureMask(mode=mode, ids=tuple(sorted(ids))), n


# ===========================================================================
# V1 — purity / determinism
# ===========================================================================


@SETTINGS
@given(masks_and_widths())
def test_from_mask_is_deterministic(mask_and_n):
    """V1: same (mask, n_sae) ⇒ identical vocab, every time."""
    mask, n = mask_and_n
    assert FeatureVocab.from_mask(mask, n) == FeatureVocab.from_mask(mask, n)


@SETTINGS
@given(masks_and_widths())
def test_vocab_is_exactly_the_allowed_universe(mask_and_n):
    """The vocab IS ``mask.allowed_universe_mask(n).nonzero()`` — for BOTH modes.

    If this ever drifts, a feature the mask permits would have no encoder row (or a
    feature it forbids would get one), which is silent corruption in both directions.
    """
    mask, n = mask_and_n
    vocab = FeatureVocab.from_mask(mask, n)
    allowed = mask.allowed_universe_mask(n)
    expected = tuple(int(i) for i in allowed.nonzero().flatten().tolist())
    if vocab is None:  # identity is normalized to None
        assert expected == tuple(range(n))
    else:
        assert vocab.original_ids == expected


@SETTINGS
@given(masks_and_widths())
def test_every_vocab_id_survives_the_masks_keep_mask(mask_and_n):
    """Cross-check against the OTHER mask primitive (value-based keep_mask), so the
    two can't drift apart."""
    mask, n = mask_and_n
    vocab = FeatureVocab.from_mask(mask, n)
    if vocab is None:
        return
    ids = t.tensor(vocab.original_ids, dtype=t.int64)
    assert bool(mask.keep_mask(ids).all()), "a vocab id was rejected by keep_mask"

    excluded = [i for i in range(n) if i not in set(vocab.original_ids)]
    if excluded:
        assert not bool(mask.keep_mask(t.tensor(excluded, dtype=t.int64)).any()), (
            "an excluded id was accepted by keep_mask"
        )


# ===========================================================================
# V2 — round-trip
# ===========================================================================


@SETTINGS
@given(vocabs())
def test_round_trip_is_the_identity_on_every_vocab_id(vocab):
    """V2 / issue #695: new→old(old→new(id)) == id for every id in the vocab."""
    ids = t.tensor(vocab.original_ids, dtype=t.int64)
    assert t.equal(vocab.to_original(vocab.to_compact(ids)), ids)


@SETTINGS
@given(vocabs())
def test_EVERY_out_of_vocab_id_RAISES_and_is_NAMED(vocab):
    """V2's second half: every id OUTSIDE the vocab raises. Never a -1, never a row.

    There is no opt-in, so this is the WHOLE contract for an excluded id — and an error is
    only actionable if it names the id that caused it, so that is pinned here too.

    (The map itself still carries the -1; ``test_old_to_new_is_minus_one_EXACTLY_on_the_
    complement`` pins it there. This test and that one together are what the deleted
    sentinel property used to assert in one — and they assert it about the two surfaces
    SEPARATELY, which is the distinction the whole tightening turns on.)
    """
    outside = [i for i in range(vocab.n_sae_features) if i not in set(vocab.original_ids)]
    for oov in outside:
        with pytest.raises(FeatureVocabError) as exc:
            vocab.to_compact(t.tensor([oov], dtype=t.int64))
        assert f"[{oov}]" in str(exc.value), f"the error must name the offending id {oov}"


@SETTINGS
@given(vocabs())
def test_compact_ids_are_exactly_the_contiguous_range(vocab):
    """Compact ids must be 0..V-1 with no gaps — the embedding table has no holes."""
    ids = t.tensor(vocab.original_ids, dtype=t.int64)
    assert t.equal(vocab.to_compact(ids), t.arange(vocab.size, dtype=t.int64))


@SETTINGS
@given(vocabs())
def test_compaction_is_order_preserving(vocab):
    """Compact id == rank of the original id. Sorted in, sorted out."""
    ids = t.tensor(vocab.original_ids, dtype=t.int64)
    compact = vocab.to_compact(ids)
    assert t.equal(compact.sort().values, compact), "compaction must be monotonic"


# ===========================================================================
# Tensor maps are exact inverses
# ===========================================================================


@SETTINGS
@given(vocabs())
def test_old_to_new_and_new_to_old_are_exact_inverses(vocab):
    o2n, n2o = vocab.old_to_new_tensor(), vocab.new_to_old_tensor()
    assert t.equal(o2n[n2o], t.arange(vocab.size, dtype=t.int64))
    assert t.equal(n2o, t.tensor(vocab.original_ids, dtype=t.int64))
    assert int((o2n >= 0).sum()) == vocab.size, "exactly V entries may be in-vocab"


# ===========================================================================
# select_rows — the slice-init primitive
# ===========================================================================


@SETTINGS
@given(vocabs(max_n=32), st.integers(min_value=1, max_value=6))
def test_select_rows_is_a_bitwise_gather(vocab, dim):
    """Slice-init byte-exactness rests on this: compact[j] IS full[original_ids[j]],
    bit for bit. No arithmetic, no rescale, no reordering."""
    full = t.randn(vocab.n_sae_features, dim, dtype=t.float32)
    got = vocab.select_rows(full)
    assert got.shape == (vocab.size, dim)
    for j, orig in enumerate(vocab.original_ids):
        assert t.equal(got[j], full[orig]), f"row {j} != full[{orig}]"


@SETTINGS
@given(vocabs(max_n=32))
def test_select_rows_agrees_with_to_compact(vocab):
    """The two views of the map must agree: gathering row j gives the row whose
    ORIGINAL id compacts to j."""
    full = t.arange(vocab.n_sae_features, dtype=t.float32).unsqueeze(1)
    got = vocab.select_rows(full)
    for j in range(vocab.size):
        original_id = int(got[j, 0].item())
        assert int(vocab.to_compact(t.tensor([original_id]))[0]) == j


# ===========================================================================
# Serialization
# ===========================================================================


@SETTINGS
@given(vocabs())
def test_fingerprint_round_trips_through_canonical_dict(vocab):
    rebuilt = FeatureVocab(
        n_sae_features=vocab.canonical_dict()["n_sae_features"],
        original_ids=tuple(vocab.canonical_dict()["original_ids"]),
    )
    assert rebuilt == vocab
    assert rebuilt.fingerprint() == vocab.fingerprint()


@SETTINGS
@given(vocabs(), vocabs())
def test_fingerprint_collides_only_for_equal_vocabs(a, b):
    """Distinct vocabs ⇒ distinct fingerprints (this is what the resume guard trusts)."""
    assert (a.fingerprint() == b.fingerprint()) == (a == b)


# ===========================================================================
# The n_sae_features collision — the one the bl_/wl_ leaf CANNOT see
# ===========================================================================
#
# ``test_fingerprint_collides_only_for_equal_vocabs`` above technically covers this, but
# only by accident: its two vocabs are drawn independently, so hypothesis will
# essentially never produce a pair whose ids MATCH and whose widths DIFFER — the exact
# shape of the bug. Construct that pair deliberately.


@st.composite
def same_ids_two_widths(draw, max_n=64):
    """Two vocabs with IDENTICAL ids and DIFFERENT ``n_sae_features``."""
    n1 = draw(st.integers(min_value=2, max_value=max_n))
    n2 = draw(st.integers(min_value=2, max_value=max_n))
    assume(n1 != n2)
    lo = min(n1, n2)
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=lo - 1),
            min_size=1,
            max_size=lo,
            unique=True,
        )
    )
    # Keep both non-identity, so neither would have normalized to None upstream.
    assume(len(ids) < lo)
    ids = tuple(sorted(ids))
    return (
        FeatureVocab(n_sae_features=n1, original_ids=ids),
        FeatureVocab(n_sae_features=n2, original_ids=ids),
    )


@SETTINGS
@given(same_ids_two_widths())
def test_fingerprint_separates_vocabs_differing_ONLY_in_n_sae_features(pair):
    """Spec §4.9.3 — the silent-wrong-model bug.

    ``{whitelist, ids:[1,2,3]}`` against a 16k SAE and against a 65k SAE produce the
    SAME ``bl_``/``wl_`` checkpoint leaf (it hashes only mode + ids), the same id list,
    and the same V-row table shape. ``n_sae_features`` is the ONLY thing that separates
    them, so the fingerprint — which the resume guard compares — must carry it.
    """
    a, b = pair
    assert a.original_ids == b.original_ids, "fixture must hold the ids fixed"
    assert a.n_sae_features != b.n_sae_features
    assert a != b
    assert a.fingerprint() != b.fingerprint()
    assert a.metadata_dict() != b.metadata_dict()


@SETTINGS
@given(same_ids_two_widths())
def test_the_id_list_ALONE_cannot_tell_these_apart(pair):
    """States the premise the test above defends: everything the checkpoint leaf and the
    table shape can see is IDENTICAL between the two. If this ever stops holding, the
    fingerprint's n_sae component has become redundant — but until then it is the only
    guard, so it may not be dropped."""
    a, b = pair
    assert a.size == b.size, "same V ⇒ same table shape ⇒ load_state_dict is happy"
    assert a.canonical_dict()["original_ids"] == b.canonical_dict()["original_ids"]


# ===========================================================================
# from_mask is order-independent (the purity claim, stated adversarially)
# ===========================================================================


@SETTINGS
@given(masks_and_widths(), st.randoms(use_true_random=False))
def test_from_mask_ignores_the_ORDER_the_mask_ids_arrive_in(mask_and_n, rng):
    """V1 restated where it can actually break.

    ``from_mask`` inherits its purity from ``FeatureMask.__post_init__``'s
    canonicalisation. If that canonicalisation were ever dropped, two configs listing
    the same ids in a different order would silently resolve to DIFFERENT vocabs — and
    the ``bl_``/``wl_`` leaf, which hashes the canonical ids, would still claim they are
    the same checkpoint. Pin the composition, not just the happy path.
    """
    mask, n = mask_and_n
    shuffled = list(mask.ids)
    rng.shuffle(shuffled)
    remade = FeatureMask(mode=mask.mode, ids=tuple(shuffled))

    original = FeatureVocab.from_mask(mask, n)
    reshuffled = FeatureVocab.from_mask(remade, n)
    assert original == reshuffled
    if original is not None:
        assert reshuffled is not None
        assert original.fingerprint() == reshuffled.fingerprint()


@SETTINGS
@given(masks_and_widths())
def test_duplicated_mask_ids_cannot_change_the_vocab(mask_and_n):
    """A config that repeats an id must resolve to the same vocab as one that does not."""
    mask, n = mask_and_n
    doubled = FeatureMask(mode=mask.mode, ids=tuple(mask.ids) + tuple(mask.ids))
    assert FeatureVocab.from_mask(doubled, n) == FeatureVocab.from_mask(mask, n)


@SETTINGS
@given(st.integers(min_value=1, max_value=64))
def test_whitelisting_EVERY_id_always_normalizes_to_None(n):
    """The identity vocab must never be constructed, at any width — otherwise the
    zero-added-ops (byte-identical) encoder path could be left silently unentered."""
    m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(n)))
    assert FeatureVocab.from_mask(m, n) is None


@SETTINGS
@given(st.integers(min_value=1, max_value=64))
def test_blacklisting_EVERY_id_always_raises(n):
    m = FeatureMask(mode=FeatureMaskMode.blacklist, ids=tuple(range(n)))
    with pytest.raises(FeatureVocabError, match="zero features"):
        FeatureVocab.from_mask(m, n)


# ===========================================================================
# to_compact on ARBITRARY id tensors — mixed in-vocab / out-of-vocab
# ===========================================================================
#
# The round-trip properties feed to_compact tensors that are ALL in-vocab, and the OOV
# property feeds it ids that are ALL out-of-vocab, one at a time. Real encoder inputs are
# neither: they interleave real ids, repeats, and padding zeros in one tensor. A bug in
# the gather (a stray sort, a scatter that overwrites, a broadcast) survives both of the
# homogeneous tests and dies here.


@st.composite
def vocab_and_arbitrary_ids(draw, max_n=32):
    """A vocab, and a tensor of ARBITRARY legal SAE ids — in-vocab, out-of-vocab, mixed."""
    vocab = draw(vocabs(max_n=max_n))
    n = vocab.n_sae_features
    rows = draw(st.integers(min_value=1, max_value=4))
    cols = draw(st.integers(min_value=1, max_value=6))
    flat = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return vocab, t.tensor(flat, dtype=t.int64).reshape(rows, cols)


@st.composite
def vocab_and_in_vocab_ids(draw, max_n=32):
    """A vocab, and a tensor drawn ONLY from ids it holds — dups and any order allowed.

    Everything ``to_compact`` ACCEPTS is of this shape, now that a single out-of-vocab id
    is fatal for the whole tensor. So this is the strategy for every property about what
    it RETURNS.
    """
    vocab = draw(vocabs(max_n=max_n))
    rows = draw(st.integers(min_value=1, max_value=4))
    cols = draw(st.integers(min_value=1, max_value=6))
    flat = draw(
        st.lists(
            st.sampled_from(vocab.original_ids),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return vocab, t.tensor(flat, dtype=t.int64).reshape(rows, cols)


@SETTINGS
@given(vocab_and_in_vocab_ids())
def test_to_compact_is_the_ELEMENTWISE_rank_of_each_id(vocab_ids):
    """For EVERY element independently: the rank of its OWN original id.

    Duplicates, repeats and unsorted order all at once, against an independent python-dict
    reference. A bug in the gather (a stray sort, a scatter that overwrites, a broadcast)
    dies here.

    This ran through the ``-1`` sentinel on a MIXED tensor before. Drawing the ids from the
    vocab instead runs it through the STRICT path — the one the encoder actually takes —
    and the mixed tensor's RAISE is pinned as a biconditional immediately below, so nothing
    is lost by the swap.
    """
    vocab, ids = vocab_ids
    lookup = {orig: j for j, orig in enumerate(vocab.original_ids)}
    expected = t.tensor([[lookup[int(x)] for x in row] for row in ids], dtype=t.int64)
    assert t.equal(vocab.to_compact(ids), expected)


@SETTINGS
@given(vocab_and_arbitrary_ids())
def test_to_compact_RAISES_iff_some_id_is_out_of_vocab(vocab_ids):
    """V2 as a BICONDITIONAL over arbitrary mixed tensors — the strongest form, and the
    one that replaces the old sentinel/raise pair:

    * it fires on EVERY tensor holding an out-of-vocab id, and COUNTS every distinct
      offender that shares the tensor with it (the list may truncate; the count may not);
    * it fires on NO tensor that is fully in-vocab — and then returns exactly the
      elementwise ranks, every one of them a REAL row of the embedding table.

    The second bullet is what makes this stronger than what it replaces. The old else-branch
    compared ``to_compact(ids)`` against ``to_compact(ids, on_missing="sentinel")`` — the
    same gather with one branch skipped, i.e. the implementation against ITSELF. It could
    not have caught a wrong rank. This compares against an independent reference, and adds
    the non-negativity claim the whole tightening exists to buy.
    """
    vocab, ids = vocab_ids
    lookup = {orig: j for j, orig in enumerate(vocab.original_ids)}
    offenders = {int(x) for x in ids.flatten().tolist() if int(x) not in lookup}

    if offenders:
        with pytest.raises(FeatureVocabError) as exc:
            vocab.to_compact(ids)
        assert f"{len(offenders)} feature id(s)" in str(exc.value), (
            "the COUNT of distinct offenders must be exact, however many there are"
        )
        return

    got = vocab.to_compact(ids)
    expected = t.tensor([[lookup[int(x)] for x in row] for row in ids], dtype=t.int64)
    assert t.equal(got, expected)
    assert bool((got >= 0).all()), "an ACCEPTED input must never yield a negative row"
    assert bool((got < vocab.size).all()), "every returned row must exist in the table"


@SETTINGS
@given(vocab_and_in_vocab_ids())
def test_to_compact_preserves_shape_and_returns_int64(vocab_ids):
    """Shape and dtype — the sentinel was only ever incidental scaffolding here."""
    vocab, ids = vocab_ids
    got = vocab.to_compact(ids)
    assert got.shape == ids.shape
    assert got.dtype == t.int64
    assert t.equal(vocab.to_compact(ids.to(t.int32)), got), (
        "producers are free to emit int32; embedding indices must come back int64 and "
        "must not change value in the upcast"
    )


@SETTINGS
@given(vocab_and_arbitrary_ids())
def test_to_compact_is_a_pure_function_no_input_mutation(vocab_ids):
    """The encoder hands ``to_compact`` tensors it still needs. In-place scatter into the
    caller's ids would corrupt magnitudes/masks downstream in a way nothing else checks.

    Now checked on the RAISE path too, which the sentinel version could never reach: the
    failure path re-indexes the caller's tensor (``ids64[compact < 0]``) to build its
    message, and an error handler that corrupted the very ids it is reporting on would be
    an exceptionally nasty thing to debug.
    """
    vocab, ids = vocab_ids
    before = ids.clone()
    with contextlib.suppress(FeatureVocabError):
        vocab.to_compact(ids)
    assert t.equal(ids, before), "to_compact must not mutate its input"


@SETTINGS
@given(vocabs())
def test_to_original_ALWAYS_raises_on_the_minus_one_that_old_to_new_emits(vocab):
    """The composition guard, as a property: an out-of-vocab id must never survive a
    round trip by silently wrapping to the last row (torch indexes -1 from the END).

    ``to_compact`` no longer emits the -1 — it raises, and that is asserted here too. But
    ``old_to_new_tensor()`` is public and still carries the sentinel, so it remains a live
    route into ``to_original``, and ``to_original``'s bounds check remains the only thing
    standing between a -1 and a plausible, WRONG original id.
    """
    outside = [i for i in range(vocab.n_sae_features) if i not in set(vocab.original_ids)]
    if not outside:
        return
    oov = t.tensor(outside, dtype=t.int64)

    with pytest.raises(FeatureVocabError):
        vocab.to_compact(oov)  # the producer refuses to hand out the -1 at all

    compact = vocab.old_to_new_tensor()[oov]  # ... but the raw map still says -1
    assert bool((compact == -1).all())
    with pytest.raises(FeatureVocabError, match="out of range"):
        vocab.to_original(compact)  # ... and the consumer guards it anyway


@SETTINGS
@given(vocabs())
def test_old_to_new_is_minus_one_EXACTLY_on_the_complement(vocab):
    """Not merely 'V entries are non-negative' — the non-negative entries must be at
    exactly the vocab's original ids, and nowhere else."""
    o2n = vocab.old_to_new_tensor()
    in_vocab = t.zeros(vocab.n_sae_features, dtype=t.bool)
    in_vocab[t.tensor(vocab.original_ids, dtype=t.int64)] = True
    assert t.equal(o2n >= 0, in_vocab)
    assert t.equal(o2n[~in_vocab], t.full(((~in_vocab).sum().item(),), -1, dtype=t.int64))  # ty: ignore[no-matching-overload]


@SETTINGS
@given(vocabs(max_n=32))
def test_select_rows_and_to_compact_are_the_same_map_viewed_two_ways(vocab):
    """The slice-init gather and the forward-pass remap MUST agree, or a compacted model
    embeds row j using weights that were initialised for a different feature."""
    full = t.randn(vocab.n_sae_features, 3, dtype=t.float32)
    compact_table = vocab.select_rows(full)
    for orig in vocab.original_ids:
        j = int(vocab.to_compact(t.tensor([orig], dtype=t.int64))[0])
        assert t.equal(compact_table[j], full[orig]), (
            f"row {j} of the compacted table is not full[{orig}]"
        )
