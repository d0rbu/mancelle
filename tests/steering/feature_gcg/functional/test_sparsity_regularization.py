"""Sparsity / regularization integration tests for Task 8.

Covers count-penalty wiring, partition-cap enforcement, and search-vocabulary
constraints across the full optimize loop.

NOT duplicated from test_optimize_loop.py:
  ``test_count_penalty_drives_sparsity`` covers the λ=0 vs λ=100 binary case;
  these tests add broader coverage (λ-sweep monotonicity, cap enforcement,
  objective accounting, partition counting on real bags, vocabulary gates).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch

from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import FeatureConditioningEditSpace
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.optimize import (
    _grad_batch,
    _Group,
    _n_sae_features,
    _penalized_obj_batch,
    optimize_feature_conditioning_universal,
)
from steering.conditional.feature_gcg.partition import (
    count_penalty,
    is_active,
    partition_bounds_from_model,
)
from steering.refined_types import NonEmptyNonNegativeIntTuple
from tests.steering.feature_gcg.functional.test_optimize_loop import (
    _MODEL,
    D_EMBD,
    N_FEATURES,
    NUM_BUCKETS,
    SCALE,
    SCALE_B,
    _make_config,
    _synth_factory,
)


def _make_stub_model_with_neg() -> SimpleNamespace:
    """Stub model identical to _MODEL but with negative_selection in user_config.

    This makes partition_bounds_from_model return has_negative=True, enabling
    min_k_neg/max_k_neg cap enforcement in the edit space.
    """
    encoder_config = SimpleNamespace(num_buckets=NUM_BUCKETS, n_sae_features=N_FEATURES)
    encoder = SimpleNamespace(
        config=encoder_config,
        feature_embed=SimpleNamespace(weight=torch.zeros(N_FEATURES, D_EMBD)),
    )
    return SimpleNamespace(
        encoder=encoder,
        feature_vocab=None,  # uncompacted stub (#720 universe path reads model.feature_vocab)
        _loaded_meta_data={
            "user_config": {
                "negative_selection": {
                    "distribution": "geometric",
                    "mean": 3.0,
                },
            }
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# High-fid synthetic objective — maximizes sum(fid) via ADD/REPLACE of high ids
# ─────────────────────────────────────────────────────────────────────────────


class _HighFidObjective:
    """Gradient-guided objective: minimize -sum(fid) → prefers HIGH feature ids.

    Inverts the ``id_weight`` axis so that ADD/REPLACE proposals with higher fids
    receive lower (better) GCG scores, making the optimizer choose high-fid
    features.  Used to make vocabulary restrictions and the ADD-cap become the
    binding constraints under test.

    With::

        id_weight[i, :] = (N_FEATURES - 1 - i) * SCALE   (inverted vs _SyntheticObjective)

    and uniform ``id_grad = 1/D_EMBD``, the GCG score for REPLACE slot s to f' is:

        ((N-1-f') - (N-1-f)) * SCALE = (f - f') * SCALE

    which is LOWER (better in minimization) for HIGHER f'.  ADD proposals
    similarly prefer the highest available fid because the pooled gradient
    contracts with the same inverted weight matrix.
    """

    def __init__(self) -> None:
        # Inverted: id_weight[i] = (N-1-i)*SCALE → lower weight for higher fids → scored first.
        self._id_weight = (
            torch.arange(N_FEATURES - 1, -1, -1, dtype=torch.float32)
            .mul(SCALE)
            .unsqueeze(1)
            .expand(-1, D_EMBD)
            .contiguous()
        )
        # bucket_weight: same as _SyntheticObjective (lower buckets preferred).
        self._bucket_weight = (
            torch.arange(NUM_BUCKETS, dtype=torch.float32)
            .mul(SCALE_B)
            .unsqueeze(1)
            .expand(-1, D_EMBD)
            .contiguous()
        )

    def losses(self, items: Sequence[Any]) -> list[float]:
        return [float(-sum(fid for fid, _ in item.bag.entries)) for item in items]

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        return [self._grad(item.bag) for item in items]

    def _grad(self, bag: FeatureConditioning) -> FeatureGrad:
        n_real = len(bag)
        uni = 1.0 / D_EMBD
        id_grad = torch.full((n_real, D_EMBD), uni)
        bucket_grad = torch.full((n_real, D_EMBD), uni)
        return FeatureGrad(
            id_grad=id_grad,
            bucket_grad=bucket_grad,
            id_weight=self._id_weight,
            bucket_weight=self._bucket_weight,
        )


def _high_fid_factory(_model: Any) -> _HighFidObjective:
    return _HighFidObjective()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — partition counting on a real optimized bag
# ─────────────────────────────────────────────────────────────────────────────


def test_partition_counting_matches_is_active() -> None:
    """``is_active`` correctly classifies both ACTIVE and INACTIVE entries.

    Warm start: ``((4, 3), (8, 0))`` — one entry in the INACTIVE bucket (3 ==
    ``num_buckets - 1``) and one in an ACTIVE bucket (0).  With a model that has
    ``negative_selection`` (so ``has_negative=True``) and ``min_k_pos=1,
    min_k_neg=1``, the optimizer cannot drop either partition below 1 entry.
    The final bag therefore always has at least one active (bucket < num_buckets-1)
    and at least one inactive (bucket == num_buckets-1) entry, exercising BOTH
    branches of ``is_active``.

    Asserts:
    - ``n_active > 0`` and ``n_inactive > 0``  — both branches exercised.
    - ``n_active + n_inactive == len(bag.entries)``  — partition exhausts all entries.
    - The direct bucket comparison is consistent with ``is_active``.

    TEETH: if ``is_active`` always returns True (or is removed), ``n_inactive``
    is 0 and ``assert n_inactive > 0`` REDs.
    """
    # Use a model with negative_selection → has_negative=True → min_k_neg is enforced.
    model = _make_stub_model_with_neg()
    cfg = _make_config(
        max_steps=20,
        reg_lambda=0.0,
        min_k_pos=1,
        min_k_neg=1,
    )
    bounds = partition_bounds_from_model(model, cfg)
    num_buckets = bounds.num_buckets  # 4 for the stub model

    # bucket=3 is inactive (num_buckets-1), bucket=0 is active.
    warm = FeatureConditioning(entries=((4, 3), (8, 0)))

    bag = optimize_feature_conditioning_universal(
        model,
        ["m0"],
        warm,
        objective_factory=_synth_factory,
        config=cfg,
    )

    n_active = 0
    n_inactive = 0
    for _fid, bucket in bag.entries:
        if is_active(bucket, num_buckets):
            assert bucket < num_buckets - 1, (
                f"is_active returned True but bucket={bucket} >= num_buckets-1={num_buckets - 1}"
            )
            n_active += 1
        else:
            assert bucket == num_buckets - 1, (
                f"is_active returned False but bucket={bucket} != num_buckets-1={num_buckets - 1}"
            )
            n_inactive += 1

    assert n_active + n_inactive == len(bag.entries), (
        f"Partition counts {n_active}+{n_inactive} != total {len(bag.entries)}"
    )
    # Both partitions must be populated — the warm-start buckets are preserved.
    assert n_active > 0, (
        f"Expected at least one active entry (bucket < {num_buckets - 1}); bag={bag.entries}"
    )
    assert n_inactive > 0, (
        f"Expected at least one inactive entry (bucket == {num_buckets - 1}); bag={bag.entries}"
    )
    # Verify equivalence with the direct predicate.
    expected_active = sum(1 for _, b in bag.entries if b < num_buckets - 1)
    assert n_active == expected_active, (
        f"Active count mismatch: n_active={n_active}, expected {expected_active}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — partition caps respected end-to-end
# ─────────────────────────────────────────────────────────────────────────────


def test_caps_respected_max_k_pos_never_exceeded() -> None:
    """max_k_pos-1 active features is never exceeded even when the objective rewards adding.

    Uses ``_HighFidObjective`` (minimize -sum(fid)), which strictly improves by
    ADD-ing any new feature with fid > 0.  Without the ADD-cap gate, the optimizer
    would fill the entire SAE vocabulary.  With ``max_k_pos=3`` (realized cap = 2
    active features), ADD is blocked once ``n_pos == 2``.

    TEETH (two-part):
    1. ``assert n_tight <= 2`` REDs when the ``_add_allowed`` / ``_within_bounds``
       gate is removed — the optimizer then exceeds the cap.
    2. ``assert n_loose > n_tight`` proves the cap BINDS at max_k_pos=3: the same
       objective with ``max_k_pos=99`` (no binding cap) accumulates strictly more
       active features, confirming the optimizer WOULD exceed 2 if unconstrained.

    Warm start: ``((0, 0),)`` — one feature with fid=0, n_pos=1, below the tight cap.
    """
    warm = FeatureConditioning(entries=((0, 0),))

    cfg_tight = _make_config(
        max_steps=60,
        min_k_pos=0,
        max_k_pos=3,
        reg_lambda=0.0,
        master_seed=1,
    )
    cfg_loose = _make_config(
        max_steps=60,
        min_k_pos=0,
        max_k_pos=99,
        reg_lambda=0.0,
        master_seed=1,
    )
    bounds_tight = partition_bounds_from_model(_MODEL, cfg_tight)
    bounds_loose = partition_bounds_from_model(_MODEL, cfg_loose)

    bag_tight = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm,
        objective_factory=_high_fid_factory,
        config=cfg_tight,
    )
    bag_loose = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm,
        objective_factory=_high_fid_factory,
        config=cfg_loose,
    )

    n_tight = sum(1 for _, b in bag_tight.entries if is_active(b, bounds_tight.num_buckets))
    n_loose = sum(1 for _, b in bag_loose.entries if is_active(b, bounds_loose.num_buckets))

    assert n_tight <= bounds_tight.max_k_pos - 1, (
        f"max_k_pos cap violated: n_tight={n_tight} > max_k_pos-1={bounds_tight.max_k_pos - 1}; "
        f"bag={bag_tight.entries}"
    )
    # TEETH: removing the ADD cap gate lets n_tight exceed 2; the loose cap
    # shows the optimizer CAN exceed 2 when unconstrained (same objective, same seed).
    assert n_loose > n_tight, (
        f"wider cap (max_k_pos=99) should yield strictly more active features than "
        f"tight cap (max_k_pos=3); got n_loose={n_loose}, n_tight={n_tight}"
    )


def test_caps_respected_min_k_pos_never_underrun() -> None:
    """Optimized bag never drops below min_k_pos active features.

    Config: min_k_pos=1 → at least 1 active feature must remain.
    Warm start: 2 active features; the optimizer can REMOVE at most one.
    """
    cfg = _make_config(max_steps=40, min_k_pos=1, max_k_pos=6, reg_lambda=0.0)
    bounds = partition_bounds_from_model(_MODEL, cfg)

    warm = FeatureConditioning(entries=((4, 0), (8, 0)))  # 2 active features
    bag = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm,
        objective_factory=_synth_factory,
        config=cfg,
    )

    n_active = sum(1 for _fid, bucket in bag.entries if is_active(bucket, bounds.num_buckets))
    assert n_active >= bounds.min_k_pos, (
        f"min_k_pos floor violated: n_active={n_active} < min_k_pos={bounds.min_k_pos}; "
        f"bag={bag.entries}"
    )


def test_caps_respected_both_bounds_simultaneously() -> None:
    """Both max_k_pos and min_k_pos are satisfied in the optimized result.

    Tight window: min_k_pos=1, max_k_pos=3 → n_active ∈ {1, 2}.
    reg_lambda=2.0 adds a count penalty to exercise the penalty interaction
    with both floors and ceilings.
    """
    cfg = _make_config(max_steps=40, min_k_pos=1, max_k_pos=3, reg_lambda=2.0)
    bounds = partition_bounds_from_model(_MODEL, cfg)

    warm = FeatureConditioning(entries=((4, 0), (8, 0)))  # 2 active — within [1,2]
    bag = optimize_feature_conditioning_universal(
        _MODEL, ["m0"], warm,
        objective_factory=_synth_factory,
        config=cfg,
    )

    n_active = sum(1 for _fid, bucket in bag.entries if is_active(bucket, bounds.num_buckets))
    assert bounds.min_k_pos <= n_active <= bounds.max_k_pos - 1, (
        f"Active count {n_active} outside allowed range "
        f"[{bounds.min_k_pos}, {bounds.max_k_pos - 1}]; bag={bag.entries}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — recorded objective == base_loss + count_penalty
# ─────────────────────────────────────────────────────────────────────────────


def test_recorded_objective_equals_base_plus_penalty() -> None:
    """``payload['objective']`` equals ``base_loss(bag) + count_penalty(bag, λ)``.

    Uses ``run_gcg`` directly to access the full payload.  The test uses
    ``_HighFidObjective`` (minimize ``-sum(fid)``), which strictly improves by
    ADD-ing features with fid > 0: the optimizer always ends with ``n_pos >= 1``
    active entries, so ``count_penalty > 0`` in the final bag — a non-trivial
    verification target.

    Mutation check:
        Removing ``+ count_penalty(...)`` from ``_penalized_obj_batch`` in
        ``optimize.py`` makes ``payload['objective']`` equal the bare base loss
        only; since ``penalty > 0`` in the final bag, ``recorded_obj != expected``
        and the ``isclose`` assertion REDs.
    """
    reg_lambda = 2.0
    warm = FeatureConditioning(entries=((0, 0),))  # 1 low-fid feature to start

    cfg = _make_config(
        max_steps=15,
        reg_lambda=reg_lambda,
        master_seed=7,
        min_k_pos=0,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)
    n_sae = _n_sae_features(_MODEL)

    high_fid_obj = _HighFidObjective()
    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(range(n_sae))),
        warm_start=warm,
    )

    group = _Group(key_str="penalty_check", members=["m0"])
    results = run_gcg(
        edit_space=edit_space,
        groups=[group],
        objective_batch=_penalized_obj_batch(high_fid_obj, reg_lambda),
        gradient_provider_batch=_grad_batch(high_fid_obj),
        cfg=cfg,
    )

    payload = results[0].payload
    bag: FeatureConditioning = payload["bag"]
    recorded_obj: float = payload["objective"]

    # Independent computation: base_loss = -sum(fid) for _HighFidObjective.
    base_loss = float(-sum(fid for fid, _ in bag.entries))
    penalty = count_penalty(bag, reg_lambda)
    expected = base_loss + penalty

    # _HighFidObjective ADD-drives the bag to n_pos >= 1, guaranteeing penalty > 0.
    assert penalty > 0, (
        f"penalty should be > 0 with _HighFidObjective (n_pos >= 1); "
        f"got penalty={penalty}, bag={bag.entries}"
    )
    assert math.isclose(recorded_obj, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"recorded objective {recorded_obj} != "
        f"base_loss {base_loss} + penalty {penalty} = {expected}; "
        f"bag={bag.entries}, reg_lambda={reg_lambda}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — λ-sweep monotonicity
# ─────────────────────────────────────────────────────────────────────────────


def test_lambda_sweep_monotone_decreasing() -> None:
    """Higher reg_lambda drives weakly fewer active features at convergence.

    Design
    ------
    Warm start: ``((0, 2),)`` — one feature, fid=0, bucket=2 (active).
    Base objective = ``sum(fid)`` = 0, already at its global minimum.

    - λ=0:  ``count_penalty(bag, 0.0) == 0``; REMOVE gives base loss
      0 == 0, NOT strictly less → not accepted.  Bag stays at 1 entry.
    - λ=1:  ``count_penalty(bag, 1.0) == 1 * 1 = 1`` for a 1-entry bag.
      REMOVE gives 0 + 0 = 0 < 0 + 1 → accepted.  Bag becomes empty.
    - λ=8:  ``count_penalty(bag, 8.0) == 8``.  Same logic → bag becomes empty.

    Expected counts: [1, 0, 0] — weakly monotone-decreasing.
    The key mutation-detectable check is counts[0] > 0 (λ=0 does NOT remove)
    and counts[1] == 0 (λ=1 DOES remove via the penalty).
    """
    warm = FeatureConditioning(entries=((0, 2),))  # fid=0, base_loss=0, 1 active entry

    counts: list[int] = []
    for reg_lambda in (0.0, 1.0, 8.0):
        cfg = _make_config(
            max_steps=20,
            candidate_batch_size=4,
            top_k=4,
            master_seed=0,
            min_k_pos=0,
            max_k_pos=6,
            reg_lambda=reg_lambda,
        )
        bounds = partition_bounds_from_model(_MODEL, cfg)
        bag = optimize_feature_conditioning_universal(
            _MODEL, ["m0"], warm,
            objective_factory=_synth_factory,
            config=cfg,
        )
        n_active = sum(
            1 for _fid, bucket in bag.entries
            if is_active(bucket, bounds.num_buckets)
        )
        counts.append(n_active)

    # Weakly monotone-decreasing: higher λ ⇒ at most as many active features.
    assert counts[0] >= counts[1] >= counts[2], (
        f"Expected weakly decreasing counts for λ∈{{0,1,8}}; got {counts}"
    )
    # λ=0: base objective already at minimum; REMOVE gives 0==0, not strictly <.
    assert counts[0] > 0, (
        f"λ=0 should not remove the fid=0 entry (base unchanged); got count={counts[0]}"
    )
    # λ=1: count_penalty makes REMOVE strictly better; must drive to empty.
    assert counts[1] == 0, (
        f"λ=1 should drive the bag to empty (REMOVE improves penalized obj); "
        f"got count={counts[1]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — allowed_feature_ids vocabulary constraints
# ─────────────────────────────────────────────────────────────────────────────


def test_optimized_bag_only_contains_allowed_feature_ids() -> None:
    """With ``allowed_feature_ids=[0..7]``, the optimized bag stays within that set.

    Uses ``_HighFidObjective`` (minimize -sum(fid)) which PREFERS high fids.
    Without the ``allowed_feature_ids`` filter, the optimizer would choose fids
    8–15 (higher = better for the objective).  With the restriction to [0..7],
    only fids 0–7 are ever proposed for ADD or REPLACE, so the final bag
    contains only ids from the restricted set.

    Warm start: ``((0, 0),)`` — a single low-fid entry within the restricted
    half-vocab.  The optimizer pushes toward higher fids but is gated to [0..7].

    TEETH: removing the ``allowed_feature_ids`` filter in ``rank_candidates``
    (so all fids 0–15 become eligible) causes fids ≥ 8 to appear in proposals
    and be accepted → ``bag_fids ⊆ restricted_set`` REDs.
    """
    restricted_ids = list(range(N_FEATURES // 2))  # [0..7] from 16-feature vocab
    warm_low = FeatureConditioning(entries=((0, 0),))

    cfg = _make_config(
        max_steps=40,
        reg_lambda=0.0,
        master_seed=3,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)

    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(restricted_ids)),
        warm_start=warm_low,
    )

    high_fid_obj = _HighFidObjective()
    group = _Group(key_str="restricted_vocab_check", members=["m0"])
    results = run_gcg(
        edit_space=edit_space,
        groups=[group],
        objective_batch=_penalized_obj_batch(high_fid_obj, 0.0),
        gradient_provider_batch=_grad_batch(high_fid_obj),
        cfg=cfg,
    )

    bag: FeatureConditioning = results[0].payload["bag"]
    bag_fids = set(bag.feature_ids())
    restricted_set = set(restricted_ids)

    assert bag_fids <= restricted_set, (
        f"Bag fids {sorted(bag_fids)} not a subset of restricted_ids {restricted_ids}; "
        f"bag.entries={bag.entries}"
    )


def test_search_only_full_vocab_admits_high_feature_ids() -> None:
    """With the full vocabulary [0..15], the high-fid objective picks fids ≥ 8.

    Uses the same ``_HighFidObjective`` as the restricted test but with the full
    vocabulary.  The optimizer is free to choose any fid and will
    preferentially pick high fids (15, 14, 13, …), so at least one fid ≥ 8
    must appear in the final bag.

    This proves that the restriction in the previous test is the binding
    constraint: widening ``allowed_feature_ids`` from [0..7] to [0..15] admits
    fids the restricted case blocks.

    Warm start: ``((0, 0),)`` — same as the restricted test (same objective,
    same seed) to make the comparison apples-to-apples.

    TEETH: replacing ``allowed_feature_ids`` with [0..7] (the restriction from
    the previous test) causes no fid ≥ 8 to appear → ``any(fid >= 8 ...)`` REDs.
    """
    warm_low = FeatureConditioning(entries=((0, 0),))

    cfg = _make_config(
        max_steps=40,
        reg_lambda=0.0,
        master_seed=3,
    )
    bounds = partition_bounds_from_model(_MODEL, cfg)
    n_sae = _n_sae_features(_MODEL)  # 16

    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(tuple(range(n_sae))),
        warm_start=warm_low,
    )

    high_fid_obj = _HighFidObjective()
    group = _Group(key_str="full_vocab_check", members=["m0"])
    results = run_gcg(
        edit_space=edit_space,
        groups=[group],
        objective_batch=_penalized_obj_batch(high_fid_obj, 0.0),
        gradient_provider_batch=_grad_batch(high_fid_obj),
        cfg=cfg,
    )

    bag: FeatureConditioning = results[0].payload["bag"]
    assert any(fid >= N_FEATURES // 2 for fid, _ in bag.entries), (
        f"Expected at least one fid >= {N_FEATURES // 2} with full vocab and "
        f"high-fid objective; got bag.entries={bag.entries}"
    )
