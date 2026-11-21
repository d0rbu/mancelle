"""Unit tests for the build_mode_aware_encoder_input orchestrator (Task 29).

The orchestrator dispatches through ``method.partition_modifier_name``,
optionally samples a negative partition for negative-mode methods on targets
that were trained with negative sampling, runs the modifier output through a
collision filter that prefers the mode-appropriate partition on overlap, and
merges into a flat ``feature_conditions`` dict that
``InferenceEngine.generate_stream`` can consume directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import pytest

from steering.eval.axbench_base import encoder_input
from steering.eval.axbench_base.encoder_input import (
    Partition,
    PartitionModifier,
    build_mode_aware_encoder_input,
    collision_filter,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
    TargetModelMetadata,
)


@dataclass(frozen=True)
class _StubMethod:
    name: str
    mode: SteeringMode
    partition_modifier_name: PartitionModifier | None


def _meta(
    *,
    num_buckets: int | None = 8,
    negative_selection: dict | None = None,
) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=num_buckets,
        n_sae_features=16384,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


# ---------------------------------------------------------------------------
# collision_filter
# ---------------------------------------------------------------------------


def test_collision_filter_no_overlap_passes_through() -> None:
    active: Partition = {1: 2, 2: 3}
    inactive: Partition = {10: 7, 11: 7}
    new_active, new_inactive = collision_filter(
        active, inactive, mode=SteeringMode.POSITIVE,
    )
    assert new_active == active
    assert new_inactive == inactive
    # Returned dicts are NEW (caller-safe).
    assert new_active is not active
    assert new_inactive is not inactive


def test_collision_filter_positive_mode_drops_from_inactive() -> None:
    active: Partition = {1: 2, 2: 3, 5: 9}
    inactive: Partition = {5: 7, 11: 7}
    new_active, new_inactive = collision_filter(
        active, inactive, mode=SteeringMode.POSITIVE,
    )
    assert new_active == {1: 2, 2: 3, 5: 9}
    assert new_inactive == {11: 7}


def test_collision_filter_negative_mode_drops_from_active() -> None:
    active: Partition = {1: 2, 5: 9}
    inactive: Partition = {5: 7, 11: 7}
    new_active, new_inactive = collision_filter(
        active, inactive, mode=SteeringMode.NEGATIVE,
    )
    assert new_active == {1: 2}
    assert new_inactive == {5: 7, 11: 7}


def test_collision_filter_does_not_mutate_inputs() -> None:
    active: Partition = {1: 2, 5: 9}
    inactive: Partition = {5: 7, 11: 7}
    collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert active == {1: 2, 5: 9}
    assert inactive == {5: 7, 11: 7}


def test_collision_filter_unknown_mode_raises() -> None:
    # ``collision_filter`` raises on an unknown mode. We smuggle a
    # bogus string into a ``SteeringMode`` slot via ``cast`` so the
    # runtime raise branch is exercised while keeping ``ty`` happy.
    with pytest.raises(AxBenchBaseError, match="unknown method mode"):
        collision_filter({1: 2}, {1: 7}, mode=cast(SteeringMode, "bogus"))


# ---------------------------------------------------------------------------
# Orchestrator: dispatch
# ---------------------------------------------------------------------------


_POSITIVE_METHOD_NAMES = [
    "ours_extract_and_add",
    "ours_replace_random",
    "ours_replace_all",
]

_NEGATIVE_METHOD_NAMES = [
    "neg_ours_remove",
    "neg_ours_add_to_negatives",
    "neg_ours_replace_random_negatives",
    "neg_ours_replace_all_negatives",
]

# DISPATCH-tested names — these modifiers actually run. The feature-GCG Phase-2
# modifiers are EXCLUDED here because their placeholder fns raise until Task 7;
# dispatching them in test_dispatch_to_correct_modifier would (correctly) raise.
# The registry-completeness test below checks the GCG keys ARE registered without
# dispatching them.
_ALL_METHOD_NAMES = _POSITIVE_METHOD_NAMES + _NEGATIVE_METHOD_NAMES

_GCG_MODIFIER_NAMES = [
    "ours_gcg_encdec",
    "ours_gcg_roofline_encdec",
    "neg_ours_gcg_encdec",
    "neg_ours_gcg_roofline_encdec",
    # Class-margin alpha sweep: same precomputed-bag dispatch as the fair pair,
    # different coefficient axis.
    "ours_gcg_alpha_encdec",
    "neg_ours_gcg_alpha_encdec",
    "feature_diffmean_encdec",
    "feature_diffmean_roofline_encdec",
    "neg_feature_diffmean_encdec",
    "neg_feature_diffmean_roofline_encdec",
]


def _mode_for(name: str) -> SteeringMode:
    return SteeringMode.NEGATIVE if name.startswith("neg_") else SteeringMode.POSITIVE


@pytest.mark.parametrize("method_name", _ALL_METHOD_NAMES)
def test_dispatch_to_correct_modifier(method_name: str, monkeypatch) -> None:
    """The orchestrator should call the method's explicit modifier."""
    spy_calls: list[dict[str, Any]] = []

    def make_spy(real_modifier):
        def spy(active, inactive, **kwargs):
            spy_calls.append(
                {"name": method_name, "active": active, "inactive": inactive, "kwargs": kwargs}
            )
            return real_modifier(active, inactive, **kwargs)

        return spy

    # _MODIFIER_REGISTRY is keyed by :class:`PartitionModifier`; reify
    # the bare method_name string into the matching enum member for
    # type-checked lookup and monkeypatch replacement.
    modifier_key = PartitionModifier(method_name)
    real = encoder_input._MODIFIER_REGISTRY[modifier_key]
    monkeypatch.setitem(
        encoder_input._MODIFIER_REGISTRY, modifier_key, make_spy(real),
    )

    method = _StubMethod(
        name=method_name,
        mode=_mode_for(method_name),
        partition_modifier_name=modifier_key,
    )
    build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={},
        top_k_concept_fids=[100, 101],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert len(spy_calls) == 1
    assert spy_calls[0]["name"] == method_name


# ---------------------------------------------------------------------------
# Orchestrator: inactive-sampling gating
# ---------------------------------------------------------------------------


def test_no_inactive_sampling_when_target_has_no_negative_selection() -> None:
    """``negative_selection`` is ``None`` -> sampler must NOT be called."""
    sampler_calls: list[Any] = []

    def sampler(*, rng):
        sampler_calls.append(rng)
        return {99: 7}

    method = _StubMethod(
        name="neg_ours_remove",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    build_mode_aware_encoder_input(
        method=method,
        active={1: 2},
        inactive={},
        top_k_concept_fids=[1],
        coefficient=4,
        target_model_meta=_meta(negative_selection=None),
        inactive_sampler=sampler,
        rng=random.Random(0),
    )
    assert sampler_calls == []


def test_inactive_sampling_called_when_target_has_negative_selection() -> None:
    sampler_calls: list[random.Random] = []

    def sampler(*, rng) -> Partition:
        sampler_calls.append(rng)
        return {99: 7, 100: 7}

    method = _StubMethod(
        name="neg_ours_add_to_negatives",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES,
    )
    rng = random.Random(0)
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 5: 9},
        inactive={},
        top_k_concept_fids=[5],
        coefficient=4,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=sampler,
        rng=rng,
    )
    assert len(sampler_calls) == 1
    assert sampler_calls[0] is rng
    # Sampled inactives are present (not dropped by collision filter).
    assert 99 in out
    assert 100 in out
    # Top-K (5) was added to inactive at the inactive bucket = 8 - 1 = 7.
    assert out[5] == 7


def test_no_inactive_sampling_for_positive_mode_method() -> None:
    """Positive-mode methods don't need inactive sampling even if target supports it."""
    sampler_calls: list[Any] = []

    def sampler(*, rng):
        sampler_calls.append(rng)
        return {99: 7}

    method = _StubMethod(
        name="ours_extract_and_add",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=PartitionModifier.OURS_EXTRACT_AND_ADD,
    )
    build_mode_aware_encoder_input(
        method=method,
        active={1: 2},
        inactive={},
        top_k_concept_fids=[100],
        coefficient=3,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=sampler,
        rng=random.Random(0),
    )
    assert sampler_calls == []


def test_no_inactive_sampling_when_sampler_is_none() -> None:
    """Even if target has negative_selection, missing sampler -> no call, no error."""
    method = _StubMethod(
        name="neg_ours_remove",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={},  # caller-supplied empty
        top_k_concept_fids=[1],
        coefficient=4,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    # neg_ours_remove drops fid 1 from active.
    assert 1 not in out
    assert out == {2: 3}


# ---------------------------------------------------------------------------
# Orchestrator: collision behavior
# ---------------------------------------------------------------------------


def test_collision_filter_deduplicates() -> None:
    """A modifier producing overlapping fids -> filter removes from the loser."""
    sampler_calls: list[Any] = []

    def sampler(*, rng):
        sampler_calls.append(rng)
        # A non-empty inactive partition that overlaps active fid 5.
        return {5: 7, 99: 7}

    method = _StubMethod(
        name="neg_ours_remove",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 5: 9},
        inactive={},
        top_k_concept_fids=[100],
        coefficient=4,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=sampler,
        rng=random.Random(0),
    )
    # neg_ours_remove leaves active intact (fid 100 not present), so active = {1: 2, 5: 9}
    # Sampler returned inactive = {5: 7, 99: 7}.
    # Collision filter (mode=negative): inactive wins, active loses fid 5.
    # Final: {1: 2, 99: 7, 5: 7}.
    assert out == {1: 2, 99: 7, 5: 7}


def test_output_is_flat_dict_no_duplicate_keys() -> None:
    method = _StubMethod(
        name="ours_extract_and_add",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=PartitionModifier.OURS_EXTRACT_AND_ADD,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={10: 7},
        top_k_concept_fids=[100, 101],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert isinstance(out, dict)
    assert len(out) == len(set(out.keys()))


def test_negative_mode_collision_prefers_inactive() -> None:
    """If a fid is in BOTH partitions, negative mode keeps the inactive bucket."""
    sampler_calls: list[Any] = []

    def sampler(*, rng):
        sampler_calls.append(rng)
        return {5: 7}  # fid 5 at inactive bucket

    method = _StubMethod(
        name="neg_ours_remove",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={5: 9, 1: 2},  # fid 5 at bucket 9
        inactive={},
        top_k_concept_fids=[],  # remove nothing — keep collision intact
        coefficient=0,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=sampler,
        rng=random.Random(0),
    )
    assert out[5] == 7  # inactive bucket wins
    assert out[1] == 2  # non-colliding active fid still there


# ---------------------------------------------------------------------------
# Orchestrator: edge cases
# ---------------------------------------------------------------------------


def test_empty_extracted_active_handled() -> None:
    """active={} should still call the modifier and produce a valid output."""
    method = _StubMethod(
        name="ours_extract_and_add",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=PartitionModifier.OURS_EXTRACT_AND_ADD,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=[100, 101],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert out == {100: 4, 101: 4}


def test_none_modifier_does_not_fall_back_to_method_name() -> None:
    """A modifier-like display name cannot replace the explicit dispatch key."""
    method = _StubMethod(
        name=PartitionModifier.OURS_EXTRACT_AND_ADD.value,
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=None,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={},
        top_k_concept_fids=[100],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert out == {}


def test_none_modifier_negative_method_returns_empty_dict() -> None:
    """A negative-mode method without a modifier also short-circuits to {}."""
    sampler_calls: list[Any] = []

    def sampler(*, rng):
        sampler_calls.append(rng)
        return {99: 7}

    method = _StubMethod(
        name="some_other_baseline",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=None,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2},
        inactive={},
        top_k_concept_fids=[1],
        coefficient=4,
        target_model_meta=_meta(negative_selection={"strategy": "global"}),
        inactive_sampler=sampler,
        rng=random.Random(0),
    )
    assert out == {}


def test_build_mode_aware_encoder_input_raises_for_inactive_manipulating_method_on_decoder_only_target() -> None:
    """num_buckets=None + an inactive-manipulating method -> explicit raise.

    The previous behavior silently fell back to ``inactive_bucket=0``,
    which hid the bug of running a negative-mode method that
    manipulates the inactive partition on a decoder-only target
    (where there is no meaningful ``num_buckets - 1`` bucket). Item 12
    of the batch R1 review flipped this to an explicit raise so
    callers see the incompatibility immediately.
    """
    method = _StubMethod(
        name="neg_ours_add_to_negatives",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES,
    )
    with pytest.raises(AxBenchBaseError, match="num_buckets=None"):
        build_mode_aware_encoder_input(
            method=method,
            active={5: 9},
            inactive={},
            top_k_concept_fids=[5],
            coefficient=0,
            target_model_meta=_meta(num_buckets=None, negative_selection=None),
            inactive_sampler=None,
            rng=random.Random(0),
        )


def test_build_mode_aware_encoder_input_allows_neg_remove_on_decoder_only() -> None:
    """neg_ours_remove does NOT manipulate inactive, so running it on
    a decoder-only target (``num_buckets=None``) is valid.
    """
    method = _StubMethod(
        name="neg_ours_remove",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={5: 9, 1: 2},
        inactive={},
        top_k_concept_fids=[5],
        coefficient=0,
        target_model_meta=_meta(num_buckets=None, negative_selection=None),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    # fid 5 dropped from active; 1 remains.
    assert out == {1: 2}


def test_orchestrator_returns_new_dict_each_call() -> None:
    method = _StubMethod(
        name="ours_extract_and_add",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=PartitionModifier.OURS_EXTRACT_AND_ADD,
    )
    out1 = build_mode_aware_encoder_input(
        method=method,
        active={1: 2},
        inactive={},
        top_k_concept_fids=[],
        coefficient=0,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    out2 = build_mode_aware_encoder_input(
        method=method,
        active={1: 2},
        inactive={},
        top_k_concept_fids=[],
        coefficient=0,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert out1 == out2
    assert out1 is not out2


def test_orchestrator_inactive_bucket_uses_num_buckets_minus_one() -> None:
    """inactive_bucket should equal num_buckets - 1 for the modifier's input."""
    method = _StubMethod(
        name="neg_ours_add_to_negatives",
        mode=SteeringMode.NEGATIVE,
        partition_modifier_name=PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={5: 9},
        inactive={},
        top_k_concept_fids=[5],
        coefficient=0,
        target_model_meta=_meta(num_buckets=16, negative_selection=None),
        inactive_sampler=None,
        rng=random.Random(0),
    )
    # fid 5 ends up at inactive bucket 16 - 1 = 15.
    assert out == {5: 15}


def test_orchestrator_passes_rng_to_modifier_for_random_methods() -> None:
    """random.Random should be threaded through ours_replace_random."""
    method = _StubMethod(
        name="ours_replace_random",
        mode=SteeringMode.POSITIVE,
        partition_modifier_name=PartitionModifier.OURS_REPLACE_RANDOM,
    )
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    out_a = build_mode_aware_encoder_input(
        method=method,
        active=dict.fromkeys(range(20), 1),
        inactive={},
        top_k_concept_fids=[100, 101, 102],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=rng_a,
    )
    out_b = build_mode_aware_encoder_input(
        method=method,
        active=dict.fromkeys(range(20), 1),
        inactive={},
        top_k_concept_fids=[100, 101, 102],
        coefficient=4,
        target_model_meta=_meta(),
        inactive_sampler=None,
        rng=rng_b,
    )
    assert out_a == out_b


def test_registry_contains_all_modifiers() -> None:
    """Sanity check: the registry has exactly the expected modifier keys.

    Original 7 modifiers + 10 precomputed-bag placeholders (the 8 fair/roofline
    GCG + feature-DiffMean pairs, plus the 2 class-margin alpha methods).
    """
    assert set(encoder_input._MODIFIER_REGISTRY.keys()) == set(_ALL_METHOD_NAMES) | set(
        _GCG_MODIFIER_NAMES
    )
