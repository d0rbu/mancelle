"""Encoder-input partition modifiers for Stage 4 generation (Tasks 27-29).

A "partition" in this module is a ``dict[int, int]`` mapping
``feature_id -> bucket_index``. Stage 4 builds an encoder input for the
target model by combining two partitions for each segment:

  * ``active`` — features whose magnitude crossed the training-time
    extraction threshold on the segment.
  * ``inactive`` — features that should be FORCED to the
    ``num_buckets - 1`` "inactive" bucket (only used when the target
    model was trained with negative-feature sampling).

Each ``ours_*`` and ``neg_ours_*`` method in the AxBench-Base evaluation
applies a different set of edits to these partitions before merging
them into a single ``feature_conditions`` dict that is forwarded to
``InferenceEngine.generate_stream``. The functions below are the
ENTIRELY-PURE primitives that perform those edits.

Each modifier:

  * Accepts ``active`` and ``inactive`` positional partitions plus a
    set of keyword-only modifier-specific args.
  * Accepts ``**_`` so the orchestrator (``build_mode_aware_encoder_input``,
    Task 29) can pass a uniform argument set to every modifier without
    each function needing to spell out the args it ignores.
  * Returns a NEW ``(active, inactive)`` tuple. Inputs are never
    mutated.
  * Performs no I/O, holds no global state, and never references a
    model.

Three positive-mode modifiers (Task 27) and four negative-mode
modifiers (Task 28) are dispatched by the
``build_mode_aware_encoder_input`` orchestrator (Task 29), which:

  1. Optionally samples an inactive partition for negative-mode methods
     on targets that were trained with negative-feature sampling
     (otherwise the inactive partition is whatever the caller passed in).
  2. Looks up ``method.partition_modifier_name`` in ``_MODIFIER_REGISTRY``
     and dispatches. Methods whose modifier is ``None`` (DiffMean /
     LinearProbe / Prompting / LoRA) get an empty ``feature_conditions``
     dict — they don't take a partition-based encoder input.
  3. Resolves any fid that ends up in BOTH partitions via the
     ``collision_filter`` helper, which prefers the active partition
     for positive-mode methods and the inactive partition for
     negative-mode methods.
  4. Merges the two partitions into a flat ``dict[int, int]`` suitable
     for ``InferenceEngine.generate_stream``.

The canonical ``Method`` dataclass lives in ``method_registry.py``. This
module uses the lightweight ``_MethodLike`` protocol below to avoid an import
cycle while retaining its exact dispatch contract.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from loguru import logger

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
    TargetModelMetadata,
)

Partition = dict[int, int]


class PartitionModifier(StrEnum):
    """Keys into partition-modifier functions for ``ours`` methods."""

    OURS_EXTRACT_AND_ADD = "ours_extract_and_add"
    OURS_REPLACE_RANDOM = "ours_replace_random"
    OURS_REPLACE_ALL = "ours_replace_all"
    NEG_OURS_REMOVE = "neg_ours_remove"
    NEG_OURS_ADD_TO_NEGATIVES = "neg_ours_add_to_negatives"
    NEG_OURS_REPLACE_RANDOM_NEGATIVES = "neg_ours_replace_random_negatives"
    NEG_OURS_REPLACE_ALL_NEGATIVES = "neg_ours_replace_all_negatives"
    # Precomputed conditioning-bag methods. The BPB/BPB-Adv loader intercepts
    # these before the placeholder modifier is called.
    OURS_GCG_ENCDEC = "ours_gcg_encdec"
    NEG_OURS_GCG_ENCDEC = "neg_ours_gcg_encdec"
    # Fair feature-GCG again, but sweeping the class-margin alpha instead of the
    # regularisation lambda: same FAIR bag family, different coefficient axis.
    OURS_GCG_ALPHA_ENCDEC = "ours_gcg_alpha_encdec"
    NEG_OURS_GCG_ALPHA_ENCDEC = "neg_ours_gcg_alpha_encdec"
    OURS_GCG_ROOFLINE_ENCDEC = "ours_gcg_roofline_encdec"
    NEG_OURS_GCG_ROOFLINE_ENCDEC = "neg_ours_gcg_roofline_encdec"
    FEATURE_DIFFMEAN_ENCDEC = "feature_diffmean_encdec"
    NEG_FEATURE_DIFFMEAN_ENCDEC = "neg_feature_diffmean_encdec"
    FEATURE_DIFFMEAN_ROOFLINE_ENCDEC = "feature_diffmean_roofline_encdec"
    NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC = "neg_feature_diffmean_roofline_encdec"


class _MethodLike(Protocol):
    """Structural protocol the orchestrator uses to read a method spec.

    The real ``Method`` dataclass lives in ``method_registry.py``. The
    orchestrator uses ``method.name`` for logging / error messages,
    ``method.mode`` for collision-filter direction, and
    ``method.partition_modifier_name`` as the sole dispatch key.
    """

    @property
    def name(self) -> str: ...  # pragma: no cover

    @property
    def mode(self) -> SteeringMode: ...  # pragma: no cover

    @property
    def partition_modifier_name(self) -> PartitionModifier | None: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Invariant helper shared by every modifier
# ---------------------------------------------------------------------------


def _assert_partition_invariants(
    new_active: Partition,
    new_inactive: Partition,
) -> None:
    """Runtime assertion that a modifier returned a well-formed pair.

    Three invariants are documented + enforced here:

      1. ``new_active`` has no intra-partition duplicates. This is
         guaranteed by dict semantics, but asserting it documents the
         intent and pins it down if a future refactor ever converts a
         modifier to use a list.
      2. ``new_inactive`` has no intra-partition duplicates — same.
      3. ``set(new_active) & set(new_inactive) == set()`` — i.e. the two
         partitions are disjoint BEFORE the orchestrator's collision
         filter runs. The filter is a safety net, not a crutch: every
         modifier must produce disjoint output on its own so the filter
         is a no-op on well-formed inputs.

    These assertions never fire on valid inputs; they are pure
    documentation of modifier contract.
    """
    assert len(set(new_active.keys())) == len(new_active), "intra-active duplicate"
    assert len(set(new_inactive.keys())) == len(new_inactive), "intra-inactive duplicate"
    assert set(new_active.keys()).isdisjoint(new_inactive.keys()), (
        f"cross-partition collision: {set(new_active) & set(new_inactive)}"
    )


# ---------------------------------------------------------------------------
# Positive-mode modifiers (Task 27)
# ---------------------------------------------------------------------------


def modify_ours_extract_and_add(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    coefficient: int,
    **_: object,
) -> tuple[Partition, Partition]:
    """Inject the top-K concept fids into ``active`` at ``coefficient``.

    ``coefficient`` is named after the spec's outer-loop variable but in
    this method it represents the bucket index the top-K features should
    occupy in the encoder input. If a top-K fid is already in ``active``,
    its existing bucket is OVERWRITTEN (dict semantics — no duplicates).
    A top-K fid that was ALSO in ``inactive`` is proactively removed from
    ``inactive`` so we never hand the collision filter a cross-partition
    collision for a positive-mode method.
    """
    new_active = dict(active)
    new_inactive = dict(inactive)
    for fid in top_k_concept_fids:
        new_active[fid] = coefficient
        new_inactive.pop(fid, None)
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


def modify_ours_replace_random(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    coefficient: int,
    rng: random.Random,
    **_: object,
) -> tuple[Partition, Partition]:
    """Drop K random NON-top-K fids from ``active`` then add the top-K.

    "K" is ``len(top_k_concept_fids)``. The drop candidates deliberately
    EXCLUDE any fid that is already in ``top_k_concept_fids``: otherwise
    a top-K fid that happens to be in ``active`` could be dropped and
    then re-added, reducing the effective drop count to K-1.

    If K > |candidates| (the non-top-K subset of active), drop all of
    them and log a warning. The sampling uses ``rng.sample`` over the
    SORTED candidate list so the dropped set is deterministic for a
    given ``rng``.

    The top-K are added LAST; if a top-K fid was originally in
    ``active`` (and thus not dropped by virtue of the exclusion above),
    its bucket is overwritten with ``coefficient``. Top-K fids that
    exist in ``inactive`` are proactively removed from ``inactive`` so
    we hand the collision filter a disjoint-by-construction pair.
    """
    top_k_set = set(top_k_concept_fids)
    new_active = dict(active)
    new_inactive = dict(inactive)

    drop_candidates = [fid for fid in sorted(new_active.keys()) if fid not in top_k_set]
    k = len(top_k_concept_fids)
    drop_count = min(k, len(drop_candidates))
    if k > len(drop_candidates):
        logger.warning(
            f"modify_ours_replace_random: top-K size {k} exceeds "
            f"drop-candidate count {len(drop_candidates)} "
            f"(active size = {len(new_active)}, top-K already in active = "
            f"{len(new_active) - len(drop_candidates)}); "
            f"dropping all non-top-K active features.",
        )
    if drop_count > 0:
        dropped = rng.sample(drop_candidates, k=drop_count)
        for fid in dropped:
            del new_active[fid]
    for fid in top_k_concept_fids:
        new_active[fid] = coefficient
        new_inactive.pop(fid, None)
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


def modify_ours_replace_all(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    coefficient: int,
    **_: object,
) -> tuple[Partition, Partition]:
    """Replace ``active`` entirely with the top-K concept fids.

    Top-K fids in ``inactive`` are proactively removed so the
    collision filter never sees a cross-partition collision for a
    positive-mode method.
    """
    del active
    new_active: Partition = dict.fromkeys(top_k_concept_fids, coefficient)
    new_inactive = {fid: b for fid, b in inactive.items() if fid not in new_active}
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


# ---------------------------------------------------------------------------
# Negative-mode modifiers (Task 28)
# ---------------------------------------------------------------------------


def modify_neg_ours_remove(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    **_: object,
) -> tuple[Partition, Partition]:
    """Remove the top-K concept fids from ``active``; leave ``inactive`` alone.

    The simplest negative-mode method: signal "this concept should NOT be
    expressed" by deleting it from the active partition without further
    edits. Top-K fids that aren't in active are ignored (no KeyError).
    Any pre-existing cross-partition collision (a fid in both ``active``
    and ``inactive`` on entry) is resolved in favor of ``inactive`` —
    that's the negative-mode convention, so we strip colliding fids
    from ``new_active`` before returning.
    """
    new_active = dict(active)
    new_inactive = dict(inactive)
    for fid in top_k_concept_fids:
        new_active.pop(fid, None)
    # Negative mode: inactive wins on cross-partition collisions.
    for fid in list(new_active):
        if fid in new_inactive:
            del new_active[fid]
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


def modify_neg_ours_add_to_negatives(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    inactive_bucket: int,
    **_: object,
) -> tuple[Partition, Partition]:
    """Move the top-K concept fids from ``active`` into ``inactive`` at the last bucket.

    ``inactive_bucket`` is the encoder's reserved "inactive" bucket
    index (i.e., ``num_buckets - 1``). If a top-K fid is already in
    ``inactive``, its bucket is OVERWRITTEN to ``inactive_bucket``.
    Any pre-existing cross-partition collision is resolved in favor of
    ``inactive`` (negative-mode convention) by stripping colliding fids
    from ``new_active`` before returning.
    """
    new_active = dict(active)
    new_inactive = dict(inactive)
    for fid in top_k_concept_fids:
        new_active.pop(fid, None)
        new_inactive[fid] = inactive_bucket
    for fid in list(new_active):
        if fid in new_inactive:
            del new_active[fid]
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


def modify_neg_ours_replace_random_negatives(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    inactive_bucket: int,
    rng: random.Random,
    **_: object,
) -> tuple[Partition, Partition]:
    """Drop K random existing NON-top-K inactives, then add the top-K.

    Also strips the top-K from ``active`` (mirrors
    ``modify_neg_ours_add_to_negatives``). The drop candidates
    deliberately EXCLUDE any fid in ``top_k_concept_fids`` so a top-K
    fid that happens to be in ``inactive`` on entry is not dropped and
    then re-added (which would reduce effective drop count to K-1).
    If ``len(candidates) < K``, drops fewer (warning logged) — the
    goal is to keep the inactive partition's cardinality roughly
    stable, not to crash.
    """
    top_k_set = set(top_k_concept_fids)
    new_active = dict(active)
    new_inactive = dict(inactive)
    for fid in top_k_concept_fids:
        new_active.pop(fid, None)
    drop_candidates = [fid for fid in sorted(new_inactive.keys()) if fid not in top_k_set]
    k = len(top_k_concept_fids)
    drop_count = min(k, len(drop_candidates))
    if k > len(drop_candidates):
        logger.warning(
            f"modify_neg_ours_replace_random_negatives: top-K size {k} "
            f"exceeds non-top-K inactive count {len(drop_candidates)} "
            f"(inactive size = {len(new_inactive)}, top-K already in "
            f"inactive = "
            f"{len(new_inactive) - len(drop_candidates)}); "
            f"dropping all non-top-K inactives.",
        )
    if drop_count > 0:
        dropped = rng.sample(drop_candidates, k=drop_count)
        for fid in dropped:
            del new_inactive[fid]
    for fid in top_k_concept_fids:
        new_inactive[fid] = inactive_bucket
    for fid in list(new_active):
        if fid in new_inactive:
            del new_active[fid]
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


def modify_neg_ours_replace_all_negatives(
    active: Partition,
    inactive: Partition,
    *,
    top_k_concept_fids: list[int],
    inactive_bucket: int,
    **_: object,
) -> tuple[Partition, Partition]:
    """Strip every existing inactive and replace with the top-K.

    Also strips the top-K from ``active``. This is the most aggressive
    negative-mode method: it tells the encoder "the only inactive
    features are the concept fids; nothing else should be suppressed".
    Any pre-existing cross-partition collision with a top-K fid is
    resolved by stripping it from ``active``.
    """
    del inactive
    new_active = dict(active)
    for fid in top_k_concept_fids:
        new_active.pop(fid, None)
    new_inactive: Partition = dict.fromkeys(top_k_concept_fids, inactive_bucket)
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


# ---------------------------------------------------------------------------
# Modifier registry (Task 29)
# ---------------------------------------------------------------------------


_PartitionModifierFn = Callable[..., tuple[Partition, Partition]]


# ---------------------------------------------------------------------------
# Precomputed conditioning-bag placeholder modifier.
#
# At BPB-Adv SCORE time, ``_build_ours_intervention``
# (``axbench_bpb/intervention.py``) intercepts bag-backed methods BEFORE
# ``build_mode_aware_encoder_input`` is called, so this modifier is NOT
# invoked by the BPB-Adv scoring path.  It IS invoked by the future
# generation path (Task N) where the caller resolves the precomputed bag
# and passes ``precomputed_bag`` explicitly.
#
# ONE implementation serves all bag-backed registry entries: the
# steering DIRECTION is baked into the bag's bucket VALUES by the derive and
# the FAMILY only selects WHICH bag the caller loads, so the modifier body is
# byte-identical across all bag-backed methods.
#
# Calling it WITHOUT a bag raises :exc:`FeatureGcgError` — the same "bag
# not precomputed" sentinel that the original Task-1 placeholders raised.  The
# registry contract
# (``test_gcg_registry.py::test_gcg_modifier_placeholders_raise_feature_gcg_error``)
# is therefore preserved: ``fn({}, {})`` still raises.
# ---------------------------------------------------------------------------


def _modify_precomputed_conditioning_bag(
    active: Partition,
    inactive: Partition,
    *,
    precomputed_bag: dict[int, int] | None = None,
    **_: object,
) -> tuple[Partition, Partition]:
    """Return a precomputed conditioning bag as the active partition.

    Shared by all bag-backed registry entries. They differ ONLY in the bag the
    caller resolves and hands in as ``precomputed_bag``, so one body serves every
    family:

    * DIRECTION (positive vs negative) is carried by the bag's bucket VALUES
      (positive vs inactive bucket index, baked in by the derive), so
      the modifier itself is direction-agnostic.
    * FAMILY selects WHICH bag is loaded: the fair entries receive the
      universal-over-the-optimize-subset bag; the roofline entries receive the
      per-prefix bag the caller selects by ``prefix_id`` at resolve time.

    Because the bag already carries its own bucket assignments, the full bag
    becomes ``new_active`` and ``new_inactive`` is always empty; the collision
    filter + merge in ``build_mode_aware_encoder_input`` then produce
    ``{**bag, **{}}`` = ``bag`` as the final ``feature_conditions``.

    Raises :exc:`FeatureGcgError` when called without a bag — the BPB-Adv
    scoring path bypasses this modifier entirely, so a ``None`` bag only arises
    when the generation path has not yet resolved the precomputed bag.
    """
    if precomputed_bag is None:
        raise FeatureGcgError("precomputed conditioning bag not available")
    new_active = dict(precomputed_bag)
    new_inactive: Partition = {}
    _assert_partition_invariants(new_active, new_inactive)
    return new_active, new_inactive


_MODIFIER_REGISTRY: dict[PartitionModifier, _PartitionModifierFn] = {
    PartitionModifier.OURS_EXTRACT_AND_ADD: modify_ours_extract_and_add,
    PartitionModifier.OURS_REPLACE_RANDOM: modify_ours_replace_random,
    PartitionModifier.OURS_REPLACE_ALL: modify_ours_replace_all,
    PartitionModifier.NEG_OURS_REMOVE: modify_neg_ours_remove,
    PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES: modify_neg_ours_add_to_negatives,
    PartitionModifier.NEG_OURS_REPLACE_RANDOM_NEGATIVES: modify_neg_ours_replace_random_negatives,
    PartitionModifier.NEG_OURS_REPLACE_ALL_NEGATIVES: modify_neg_ours_replace_all_negatives,
    # Precomputed-bag methods: BPB/BPB-Adv scoring bypasses these via
    # _build_ours_intervention; generation invokes them with the loaded bag.
    # Direction lives in the bag's bucket values and family only selects which
    # bag the caller loads.
    PartitionModifier.OURS_GCG_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.NEG_OURS_GCG_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.OURS_GCG_ALPHA_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.NEG_OURS_GCG_ALPHA_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.FEATURE_DIFFMEAN_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC: _modify_precomputed_conditioning_bag,
    PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC: _modify_precomputed_conditioning_bag,
}


# Modifiers whose semantics REQUIRE a meaningful inactive partition
# (i.e. ``num_buckets is not None``). Running these on a decoder-only
# target is nonsensical because there is no "inactive" bucket to assign
# fids to — the orchestrator raises ``AxBenchBaseError`` in that case
# (Item 12 of the batch R1 review).
_INACTIVE_MANIPULATING_MODIFIERS: frozenset[PartitionModifier] = frozenset(
    {
        PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES,
        PartitionModifier.NEG_OURS_REPLACE_RANDOM_NEGATIVES,
        PartitionModifier.NEG_OURS_REPLACE_ALL_NEGATIVES,
    }
)


# ---------------------------------------------------------------------------
# Collision filter + orchestrator (Task 29)
# ---------------------------------------------------------------------------


def collision_filter(
    active: Partition,
    inactive: Partition,
    *,
    mode: SteeringMode,
) -> tuple[Partition, Partition]:
    """Resolve fid collisions between ``active`` and ``inactive``.

    A fid that ends up in BOTH partitions after a modifier runs is
    ambiguous — the encoder needs every fid in at most one partition.
    The "winner" depends on which mode the method is expressing:

      * ``mode == SteeringMode.POSITIVE``: ``active`` wins, the
        colliding fid is removed from ``inactive``.
      * ``mode == SteeringMode.NEGATIVE``: ``inactive`` wins, the
        colliding fid is removed from ``active``.

    Both inputs are returned as new dicts. The non-colliding partition
    is also copied so callers can freely mutate either return value.

    Accepts either a :class:`SteeringMode` member or the equivalent
    bare string (``"positive"`` / ``"negative"``) because
    :class:`SteeringMode` is a :class:`StrEnum`, so the equality checks
    below resolve either form.
    """
    overlap = set(active.keys()) & set(inactive.keys())
    if not overlap:
        return dict(active), dict(inactive)
    if mode == SteeringMode.POSITIVE:
        new_inactive = {fid: b for fid, b in inactive.items() if fid not in overlap}
        return dict(active), new_inactive
    if mode == SteeringMode.NEGATIVE:
        new_active = {fid: b for fid, b in active.items() if fid not in overlap}
        return new_active, dict(inactive)
    # Use AxBenchBaseError (not ValueError) so operator log-grepping for
    # the package's exception type catches this site too. Mirrors the
    # equivalent raise in ``generation.py::_eval_seeds_for_mode``.
    raise AxBenchBaseError(
        f"unknown method mode {mode!r}; expected 'positive' or 'negative'",
    )


def build_mode_aware_encoder_input(
    *,
    method: _MethodLike,
    active: Partition,
    inactive: Partition,
    top_k_concept_fids: list[int],
    coefficient: int,
    target_model_meta: TargetModelMetadata,
    inactive_sampler: Callable[..., Partition] | None,
    rng: random.Random,
) -> Partition:
    """Dispatch to a partition modifier and return a flat ``feature_conditions`` dict.

    Steps:

      1. If the method is in negative mode AND the target model was
         trained with negative-feature sampling AND the caller supplied
         an ``inactive_sampler``, call the sampler to populate the
         ``inactive`` partition. Otherwise the inactive partition is
         whatever the caller passed in directly.
      2. Read ``method.partition_modifier_name``. If it is ``None``, this
         is a baseline method (DiffMean / LinearProbe / Prompting / LoRA)
         that doesn't take a partition-based encoder input — return ``{}``.
         Otherwise, look up that explicit key in the modifier registry.
      3. If the modifier manipulates the inactive partition but the
         target has no ``num_buckets`` (decoder-only), raise
         ``AxBenchBaseError`` — a meaningful ``inactive_bucket`` is
         not defined for such targets, so running an inactive-
         manipulating method on them is nonsensical.
      4. Call the modifier with ``(active, inactive)`` plus the uniform
         keyword set every modifier accepts. Modifiers produce
         collision-free output via their end-of-function
         ``_assert_partition_invariants`` call.
      5. Apply the collision filter as a safety net (it is a no-op on
         well-formed modifier output) and assert disjointness before
         the merge.
      6. Merge ``active`` and ``inactive`` into a single dict. Step 5
         guarantees the partitions are disjoint, so merge order is
         irrelevant — we use a single fixed expression (Item 13).
    """
    if (
        method.mode == SteeringMode.NEGATIVE
        and target_model_meta.negative_selection is not None
        and inactive_sampler is not None
    ):
        sampled_inactive = inactive_sampler(rng=rng)
    else:
        sampled_inactive = dict(inactive)

    modifier_key = method.partition_modifier_name
    if modifier_key is None:
        return {}
    modifier = _MODIFIER_REGISTRY[modifier_key]

    num_buckets = target_model_meta.num_buckets
    if num_buckets is None and modifier_key in _INACTIVE_MANIPULATING_MODIFIERS:
        raise AxBenchBaseError(
            f"Method {method.name!r} uses modifier {modifier_key!r} which "
            "manipulates the inactive partition, but the target model "
            "has ``num_buckets=None`` (decoder-only / no negative-feature "
            "bucket). This combination is nonsensical — either pick a "
            "method whose modifier does not touch inactive (e.g. "
            "``neg_ours_remove``), or run this method against an "
            "encoder-decoder target."
        )
    inactive_bucket = (num_buckets - 1) if num_buckets is not None else 0

    new_active, new_inactive = modifier(
        active,
        sampled_inactive,
        top_k_concept_fids=top_k_concept_fids,
        coefficient=coefficient,
        rng=rng,
        inactive_bucket=inactive_bucket,
    )
    new_active, new_inactive = collision_filter(
        new_active,
        new_inactive,
        mode=method.mode,
    )
    assert set(new_active) & set(new_inactive) == set(), "collision_filter left a collision"
    # After collision_filter, the two partitions are disjoint, so the
    # merge order is irrelevant — we use a single fixed expression.
    return {**new_active, **new_inactive}
