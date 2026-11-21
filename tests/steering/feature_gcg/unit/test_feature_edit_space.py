"""Tests for feature_edit_space.py (Task 6) — FeatureConditioningEditSpace.

Each of the 12 EditSpace seams is exercised in isolation with a synthetic stub
``model`` and a FAKE ``FeatureGrad`` provider, so the candidate-generation /
sampling / aggregation logic is tested WITHOUT a real encoder forward.

Coverage map (brief Step 1):
- REPLACE-id excludes in-bag feature ids (dup guard).
- REPLACE-bucket gated OFF at ``num_buckets == 1``; ON when multibucket.
- REPLACE-joint emits (id, bucket) joint edits when magnitude is active.
- ADD rejected at the active cap (``n_pos == max_k_pos - 1``) and the inactive cap
  (``n_neg == max_k_neg - 1``); ADD always generated (unconditional).
- REMOVE always generated, rejected at ``min_k``.
- sample_candidates dedups on digest against ``visited`` + respects partition caps.
- model_item identity; digest pure; event fields disjoint from _RESERVED_EVENT_KEYS.
- aggregate_gradient means slot-aligned + consumes the strict zip.
- validate / is_trivial / finalize_trivial / init_target / finalize.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.conditional.feature_gcg.feature_edit_space as feature_edit_space_module
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.core import _RESERVED_EVENT_KEYS
from steering.conditional.feature_gcg.edit_space import GroupResult
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureAddition,
    FeatureConditioningEditSpace,
    FeatureProposals,
    FeatureRemoval,
    FeatureReplacement,
)
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.partition import PartitionBounds, is_active
from steering.refined_types import NonEmptyNonNegativeIntTuple

# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

_D = 4
_N_SAE = 8


def _grad(
    n_real: int,
    *,
    d: int = _D,
    n_sae: int = _N_SAE,
    num_buckets: int = 2,
    sum_kv: int = 0,
    seed: int = 0,
) -> FeatureGrad:
    """A FAKE FeatureGrad with deterministic small random tensors.

    ``bucket_grad`` mirrors ``id_grad[:, :d]`` (the separable contract), but the
    EditSpace treats the four tensors independently so the equality is incidental.
    """
    g = t.Generator().manual_seed(seed)
    id_dim = d + sum_kv
    id_grad = t.randn(n_real, id_dim, generator=g)
    return FeatureGrad(
        id_grad=id_grad,
        bucket_grad=id_grad[:, :d].clone(),
        id_weight=t.randn(n_sae, id_dim, generator=g),
        bucket_weight=t.randn(num_buckets, d, generator=g),
    )


def _bounds(
    *,
    num_buckets: int = 2,
    has_negative: bool = False,
    min_k_pos: int = 0,
    max_k_pos: int = 40,
    min_k_neg: int = 0,
    max_k_neg: int = 40,
) -> PartitionBounds:
    return PartitionBounds(
        num_buckets=num_buckets,
        has_negative=has_negative,
        min_k_pos=min_k_pos,
        max_k_pos=max_k_pos,
        min_k_neg=min_k_neg,
        max_k_neg=max_k_neg,
    )


def _space(
    *,
    bounds: PartitionBounds | None = None,
    allowed: Any = None,
    warm_start: FeatureConditioning | None = None,
) -> FeatureConditioningEditSpace:
    allowed_feature_ids = NonEmptyNonNegativeIntTuple.parse(
        tuple(allowed) if allowed is not None else tuple(range(_N_SAE))
    )
    return FeatureConditioningEditSpace(
        bounds=bounds if bounds is not None else _bounds(),
        allowed_feature_ids=allowed_feature_ids,
        warm_start=warm_start if warm_start is not None else FeatureConditioning(entries=((0, 0),)),
    )


@dataclass
class _Group:
    key_str: str
    members: list[Any]
    concept_id: str | None = None
    mode: str | None = None


def _bag(*entries: tuple[int, int]) -> FeatureConditioning:
    return FeatureConditioning(entries=tuple(entries))


def _count_pos_neg(bag: FeatureConditioning, bounds: PartitionBounds) -> tuple[int, int]:
    if not bounds.has_negative:
        return len(bag), 0
    n_pos = sum(1 for _f, b in bag.entries if is_active(b, bounds.num_buckets))
    return n_pos, len(bag) - n_pos


def _replace_edits(props: FeatureProposals) -> list[FeatureReplacement]:
    return [e for e in props.edits if isinstance(e, FeatureReplacement)]


def _add_edits(props: FeatureProposals) -> list[FeatureAddition]:
    return [e for e in props.edits if isinstance(e, FeatureAddition)]


def _remove_edits(props: FeatureProposals) -> list[FeatureRemoval]:
    return [e for e in props.edits if isinstance(e, FeatureRemoval)]


# ─────────────────────────────────────────────────────────────────────────────
# validate / is_trivial / finalize_trivial / init_target / finalize
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_accepts_wellformed_group():
    space = _space(warm_start=_bag((0, 0), (1, 0)))
    group = _Group(key_str="concept-7|mode1", members=[object(), object()])
    validated = space.validate(group)
    assert validated.key_str == "concept-7|mode1"
    assert validated.members == group.members
    assert validated.payload is group


def test_validate_rejects_non_str_key():
    space = _space()
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str=None, members=[object()]))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_validate_rejects_empty_members():
    space = _space()
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str="g", members=[]))


def test_validate_rejects_none_member():
    space = _space()
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str="g", members=[object(), None]))


def test_validate_rejects_warm_start_above_active_cap():
    # max_k_pos=3 ⇒ realized max active = 2; a 3-active warm-start violates.
    space = _space(bounds=_bounds(max_k_pos=3), warm_start=_bag((0, 0), (1, 0), (2, 0)))
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str="g", members=[object()]))


def test_validate_rejects_warm_start_below_min_k():
    space = _space(bounds=_bounds(min_k_pos=2), warm_start=_bag((0, 0)))
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str="g", members=[object()]))


def test_validate_rejects_warm_start_bad_bucket():
    space = _space(bounds=_bounds(num_buckets=2), warm_start=_bag((0, 5)))
    with pytest.raises(FeatureGcgError):
        space.validate(_Group(key_str="g", members=[object()]))


@pytest.mark.parametrize("invalid_allowed", [(), (-1, 0, 1), (True,), [0, 1]])
def test_constructor_rejects_invalid_allowed_feature_ids(
    invalid_allowed: Any,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        FeatureConditioningEditSpace(
            bounds=_bounds(),
            allowed_feature_ids=invalid_allowed,
            warm_start=_bag((0, 0)),
        )


def test_is_trivial_is_always_false():
    space = _space()
    validated = space.validate(_Group(key_str="g", members=[object()]))
    assert space.is_trivial(validated) is False


def test_finalize_trivial_raises():
    space = _space()
    validated = space.validate(_Group(key_str="g", members=[object()]))
    with pytest.raises(FeatureGcgError):
        space.finalize_trivial(validated)


def test_init_target_returns_warm_start():
    warm = _bag((3, 0), (5, 1))
    space = _space(bounds=_bounds(num_buckets=2, has_negative=True, max_k_neg=40), warm_start=warm)
    validated = space.validate(_Group(key_str="g", members=[object()]))
    assert space.init_target(validated) is warm


def test_finalize_payload_shape():
    bounds = _bounds(num_buckets=2, has_negative=True, max_k_pos=40, max_k_neg=40)
    target = _bag((0, 0), (1, 0), (2, 1))  # 2 active, 1 inactive
    space = _space(bounds=bounds, warm_start=target)
    validated = space.validate(_Group(key_str="g", members=[object()]))
    result = space.finalize(validated, target, objective=1.5, accepted_steps=3, visited_count=10)
    assert isinstance(result, GroupResult)
    assert result.payload["bag"] is target
    assert result.payload["objective"] == 1.5
    assert result.payload["accepted_steps"] == 3
    assert result.payload["visited"] == 10
    assert result.payload["n_pos"] == 2
    assert result.payload["n_neg"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# digest (pure) / model_item (identity)
# ─────────────────────────────────────────────────────────────────────────────


def test_digest_is_pure_and_matches_bag():
    space = _space()
    bag = _bag((2, 0), (5, 1))
    before = bag.entries
    assert space.digest(bag) == bag.digest()
    assert space.digest(bag) == space.digest(bag)  # deterministic
    assert bag.entries == before  # no mutation


def test_model_item_is_identity_pair():
    space = _space()
    member = object()
    target = _bag((1, 0))
    assert space.model_item(member, target) == (member, target)


# ─────────────────────────────────────────────────────────────────────────────
# event fields — concept identity only, disjoint from _RESERVED_EVENT_KEYS
# ─────────────────────────────────────────────────────────────────────────────


def test_event_fields_disjoint_from_reserved_keys():
    space = _space()
    validated = space.validate(
        _Group(key_str="g", members=[object(), object()], concept_id="c1", mode="mode1")
    )
    group_fields = space.group_event_fields(validated)
    step_fields = space.step_event_fields(validated)
    assert set(group_fields) & _RESERVED_EVENT_KEYS == set()
    assert set(step_fields) & _RESERVED_EVENT_KEYS == set()


def test_event_fields_carry_concept_identity():
    space = _space()
    validated = space.validate(
        _Group(key_str="k", members=[object(), object()], concept_id="c1", mode="mode2")
    )
    fields = space.group_event_fields(validated)
    assert fields["n_members"] == 2
    assert fields["concept_id"] == "c1"
    assert fields["mode"] == "mode2"


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_gradient — slot-aligned mean + consumes the strict zip
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_gradient_means_slot_aligned():
    space = _space()
    g1 = FeatureGrad(
        id_grad=t.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        bucket_grad=t.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        id_weight=t.zeros(_N_SAE, _D),
        bucket_weight=t.zeros(2, _D),
    )
    g2 = FeatureGrad(
        id_grad=t.tensor([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]),
        bucket_grad=t.tensor([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]),
        id_weight=t.zeros(_N_SAE, _D),
        bucket_weight=t.zeros(2, _D),
    )
    agg = space.aggregate_gradient(zip([g1, g2], ["m1", "m2"], strict=True))
    assert t.allclose(agg.id_grad, t.tensor([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]))
    assert t.allclose(agg.bucket_grad, t.tensor([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]))


def test_aggregate_gradient_shares_readonly_weights():
    space = _space()
    g1 = _grad(2, seed=1)
    g2 = _grad(2, seed=2)
    agg = space.aggregate_gradient(zip([g1, g2], ["m1", "m2"], strict=True))
    # id_weight / bucket_weight are shared read-only tensors taken from the first member.
    assert agg.id_weight is g1.id_weight
    assert agg.bucket_weight is g1.bucket_weight


def test_aggregate_gradient_consumes_strict_zip():
    space = _space()
    g1 = _grad(2, seed=1)
    g2 = _grad(2, seed=2)
    # strict=True zip raises ValueError ONLY if fully consumed (lengths differ).
    with pytest.raises(ValueError):
        space.aggregate_gradient(zip([g1, g2], ["only-one"], strict=True))


def test_aggregate_gradient_empty_raises():
    space = _space()
    with pytest.raises(FeatureGcgError):
        space.aggregate_gradient(zip([], [], strict=True))


# ─────────────────────────────────────────────────────────────────────────────
# magnitude auto-off gate
# ─────────────────────────────────────────────────────────────────────────────


def test_magnitude_active_off_when_num_buckets_one():
    space = _space(bounds=_bounds(num_buckets=1))
    assert space.magnitude_active() is False


def test_magnitude_active_on_when_multibucket():
    space = _space(bounds=_bounds(num_buckets=2))
    assert space.magnitude_active() is True


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE-id — excludes in-bag ids
# ─────────────────────────────────────────────────────────────────────────────


def test_replace_id_excludes_in_bag_ids():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), allowed=list(range(_N_SAE)), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    in_bag = set(current.feature_ids())
    # id-replace proposals: new fid (not in bag), same bucket as the current slot's bucket
    id_replaces = [
        e for e in _replace_edits(props)
        if e.fid not in in_bag and e.bucket == current.entries[e.slot][1]
    ]
    assert id_replaces, "expected REPLACE-id proposals"
    for edit in id_replaces:
        assert edit.fid not in in_bag  # never proposes an id already present
        assert edit.fid in set(range(_N_SAE))
        # REPLACE-id keeps the slot's existing bucket
        assert edit.bucket == current.entries[edit.slot][1]


def test_replace_id_empty_when_all_allowed_in_bag():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(allowed=[0, 1], warm_start=current)  # all allowed already present
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    in_bag = set(current.feature_ids())
    # No id-replace candidates (all allowed fids are in bag) and no ADD candidates.
    id_replaces = [e for e in _replace_edits(props) if e.fid not in in_bag]
    assert not id_replaces, "no id-replace allowed when all allowed fids are in bag"
    assert not _add_edits(props), "no ADD allowed when all allowed fids are in bag"


def test_replace_id_keeps_dynamic_allowed_sets_on_uncached_rank_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = feature_edit_space_module.rank_replacements_from_gradients
    calls: list[tuple[tuple[int, ...], Any]] = []

    def recording_rank(**kwargs):
        calls.append(
            (
                tuple(int(fid) for fid in kwargs["allowed_token_ids"]),
                kwargs.get("allowed_embedding_cache"),
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(
        feature_edit_space_module,
        "rank_replacements_from_gradients",
        recording_rank,
    )
    space = _space(allowed=list(range(_N_SAE)))
    cfg = FeatureGcgConfig()
    first = _bag((0, 0), (1, 0))
    second = _bag((0, 0), (2, 0))
    space.rank_candidates(_grad(len(first), seed=11), first, cfg)
    space.rank_candidates(_grad(len(second), seed=12), second, cfg)

    assert calls == [
        (tuple(range(2, _N_SAE)), None),
        ((1, *range(3, _N_SAE)), None),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE-bucket — gated by magnitude_active; REPLACE-joint
# ─────────────────────────────────────────────────────────────────────────────


def test_replace_bucket_present_when_magnitude_active():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    in_bag = set(current.feature_ids())
    # A REPLACE-bucket edit keeps the slot's fid (in-bag) and changes the bucket.
    bucket_edits = [
        e for e in _replace_edits(props)
        if e.fid in in_bag and e.bucket != current.entries[e.slot][1]
    ]
    assert bucket_edits, "expected REPLACE-bucket proposals when magnitude is active"


def test_replace_bucket_absent_at_num_buckets_one():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=1), warm_start=current)
    props = space.rank_candidates(_grad(len(current), num_buckets=1), current, cfg)
    in_bag = set(current.feature_ids())
    # magnitude auto-OFF ⇒ no proposal keeps the fid (no bucket-only edits).
    assert not any(e.fid in in_bag for e in _replace_edits(props))


def test_replace_joint_present_when_magnitude_active():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    in_bag = set(current.feature_ids())
    # A joint edit changes BOTH axes: new fid (not in bag) AND a bucket != current.
    joint = [
        e for e in _replace_edits(props)
        if e.fid not in in_bag and e.bucket != current.entries[e.slot][1]
    ]
    assert joint, "expected REPLACE-joint proposals (new id + new bucket)"


# ─────────────────────────────────────────────────────────────────────────────
# ADD — always generated; rejected at the partition cap
# ─────────────────────────────────────────────────────────────────────────────


def test_add_present_with_headroom():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0))
    space = _space(bounds=_bounds(max_k_pos=40), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    adds = _add_edits(props)
    assert adds, "expected ADD proposals when below the cap"
    for edit in adds:
        assert edit.fid not in set(current.feature_ids())


def test_add_rejected_at_active_cap():
    # max_k_pos=3 ⇒ realized max active = 2; current has n_pos=2 (at cap).
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2, max_k_pos=3), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    assert _add_edits(props) == [], "no ADD allowed when the active partition is at its cap"


def test_add_rejected_at_inactive_cap_but_active_allowed():
    # has_negative, num_buckets=2: bucket 0 active, bucket 1 inactive.
    # max_k_neg=2 ⇒ realized max inactive = 1; current n_neg=1 (at cap), n_pos has headroom.
    cfg = FeatureGcgConfig()
    bounds = _bounds(num_buckets=2, has_negative=True, max_k_pos=40, max_k_neg=2)
    current = _bag((0, 0), (1, 0), (2, 1))  # n_pos=2, n_neg=1
    space = _space(bounds=bounds, warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    adds = _add_edits(props)
    assert adds, "active ADDs should still be proposed (positive partition has headroom)"
    # No ADD targets the inactive bucket (its partition is at the cap).
    assert all(e.bucket != bounds.num_buckets - 1 for e in adds)


# ─────────────────────────────────────────────────────────────────────────────
# REMOVE — always generated; rejected at min_k
# ─────────────────────────────────────────────────────────────────────────────


def test_remove_present_above_min_k():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0), (2, 0))
    space = _space(bounds=_bounds(min_k_pos=0), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    removes = _remove_edits(props)
    assert len(removes) == 3  # one per slot
    assert {e.slot for e in removes} == {0, 1, 2}


def test_remove_rejected_at_min_k():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))  # n_pos = 2 == min_k_pos
    space = _space(bounds=_bounds(min_k_pos=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    assert _remove_edits(props) == [], "no REMOVE allowed at the min_k floor"


# ─────────────────────────────────────────────────────────────────────────────
# sample_candidates — dedup on digest + respect caps + one edit per candidate
# ─────────────────────────────────────────────────────────────────────────────


def test_sample_candidates_returns_feature_conditioning_one_edit_away():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    cands = space.sample_candidates(
        props, current, batch_size=4, temperature=0.5,
        rng=random.Random(0), visited={current.digest()},
    )
    assert cands
    for cand in cands:
        assert isinstance(cand, FeatureConditioning)
        assert cand.digest() != current.digest()


def test_sample_candidates_dedups_against_visited():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    visited = {current.digest()}
    first = space.sample_candidates(
        props, current, batch_size=3, temperature=0.5, rng=random.Random(7), visited=visited
    )
    seen = {c.digest() for c in first}
    # A second draw with the SAME (mutated) visited set never repeats an earlier bag.
    second = space.sample_candidates(
        props, current, batch_size=3, temperature=0.5, rng=random.Random(7), visited=visited
    )
    for cand in second:
        assert cand.digest() not in seen


def test_sample_candidates_respects_partition_caps():
    # n_neg at the inactive cap; REPLACE-bucket toggling active→inactive WOULD overflow it.
    cfg = FeatureGcgConfig()
    bounds = _bounds(num_buckets=2, has_negative=True, max_k_pos=40, max_k_neg=2)
    current = _bag((0, 0), (1, 0), (2, 1))  # n_pos=2, n_neg=1 (at cap)
    space = _space(bounds=bounds, warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    # Non-vacuity: the pool DOES contain a doomed active→inactive toggle.
    assert any(
        isinstance(e, FeatureReplacement) and e.bucket == 1 and current.entries[e.slot][1] == 0
        for e in props.edits
    )
    cands = space.sample_candidates(
        props, current, batch_size=8, temperature=0.5,
        rng=random.Random(3), visited={current.digest()},
    )
    for cand in cands:
        n_pos, n_neg = _count_pos_neg(cand, bounds)
        assert n_neg <= bounds.max_k_neg - 1
        assert n_pos <= bounds.max_k_pos - 1


def test_sample_candidates_empty_proposals_returns_empty():
    space = _space()
    out = space.sample_candidates(
        FeatureProposals(edits=()), _bag((0, 0)), batch_size=4,
        temperature=0.5, rng=random.Random(0), visited=set(),
    )
    assert out == []


def test_sample_candidates_respects_batch_size():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    out = space.sample_candidates(
        props, current, batch_size=2, temperature=0.5,
        rng=random.Random(0), visited={current.digest()},
    )
    assert len(out) <= 2


def test_sample_candidates_rejects_bad_temperature():
    space = _space()
    props = FeatureProposals(edits=(FeatureRemoval(slot=0, score=0.0),))
    with pytest.raises(BeartypeCallHintParamViolation, match="temperature"):
        space.sample_candidates(
            props, _bag((0, 0)), batch_size=1, temperature=0.0,
            rng=random.Random(0), visited=set(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# rank_candidates return type
# ─────────────────────────────────────────────────────────────────────────────


def test_rank_candidates_returns_feature_proposals():
    cfg = FeatureGcgConfig()
    current = _bag((0, 0), (1, 0))
    space = _space(bounds=_bounds(num_buckets=2), warm_start=current)
    props = space.rank_candidates(_grad(len(current)), current, cfg)
    assert isinstance(props, FeatureProposals)
    assert all(
        isinstance(e, (FeatureRemoval, FeatureReplacement, FeatureAddition))
        for e in props.edits
    )


# ─────────────────────────────────────────────────────────────────────────────
# validate: warm_start fid not in allowed_feature_ids raises
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_raises_when_warm_start_fid_not_in_allowed():
    """warm_start with a feature id not in allowed_feature_ids raises FeatureGcgError.

    Teeth: if the guard in validate() is removed, this test passes vacuously
    (no FeatureGcgError raised) → pytest.raises context exits without exception → fails.
    """
    # warm_start fid=99 is NOT in allowed_feature_ids [0..7]
    warm = _bag((99, 0))
    space = FeatureConditioningEditSpace(
        bounds=_bounds(),
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(
            tuple(range(_N_SAE))
        ),  # 0..7
        warm_start=warm,  # fid=99 NOT in 0..7
    )
    with pytest.raises(FeatureGcgError, match="not in allowed_feature_ids") as exc:
        space.validate(_Group(key_str="g", members=[object()]))
    # FIX 4 (#720 T5): post-#720 allowed_feature_ids is a SPARSE subset V, so the message
    # must not imply it is a contiguous range — no "max allowed" framing. It still fails
    # loud and still names the offending fid (checked above via `match`).
    msg = str(exc.value)
    assert "max allowed" not in msg, (
        "the message implies allowed_feature_ids is a contiguous [0, max] range, which "
        "is false for a sparse post-#720 vocabulary V"
    )
    assert str(_N_SAE) in msg, "must still report how many ids are actually allowed"
