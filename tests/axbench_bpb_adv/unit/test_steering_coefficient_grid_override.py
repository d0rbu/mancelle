"""``steering_coefficient_grid``: pin the scalar steering axis for costly runs.

Why the knob exists: in an adversarial run every steering coefficient costs a
full attack optimization, but the robustness plot keeps ONE representative
coefficient per family (``_select_representative_coefficient``) and discards
the rest. The default 10-value grid therefore multiplies attack cost ~10x for
output that is thrown away.

The two properties that make it safe:

  * unset changes NOTHING — same grid, and the key is stripped from the run
    identity payload, so every existing run tag stays byte-for-byte valid;
  * set changes the run tag — a restricted grid scores a different coefficient
    set, so it must never share an auto-generated tag with a full sweep.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_base.coefficient_grids import (
    _DIFFMEAN_SCALAR_GRID,
    diffmean_scalar_grid,
)
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict

_META = TargetModelMetadata(
    num_buckets=None,
    n_sae_features=None,
    negative_selection=None,
    feature_selection=None,
    per_feature_stats=None,
    device="cpu",
    sae_layer=8,
    segment_size=256,
)


def _cfg(**overrides) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        target_models=["decoder_only"],
        base_methods=["diffmean_decoder_only"],
        adversarial_methods=[FasterGcgConfig()],
        **overrides,
    )


@pytest.mark.parametrize("cfg", [object(), SimpleNamespace(), SimpleNamespace(steering_coefficient_grid=None)])
def test_unset_yields_the_full_default_grid(cfg) -> None:
    """Absent, unset, and explicitly-None all mean the full default sweep.

    Teeth: treating an absent attribute as an empty grid would silently drop
    every steering coefficient from every run that never sets this field.
    """
    assert diffmean_scalar_grid(_META, cfg) == list(_DIFFMEAN_SCALAR_GRID)


def test_override_restricts_the_axis_in_order() -> None:
    """A set override IS the grid, preserving the configured order."""
    cfg = SimpleNamespace(steering_coefficient_grid=(0.6, 0.8, 4.0, 6.0))
    assert diffmean_scalar_grid(_META, cfg) == [0.6, 0.8, 4.0, 6.0]


def test_override_values_are_floats() -> None:
    """Ints from YAML become floats, matching the default grid's element type."""
    cfg = SimpleNamespace(steering_coefficient_grid=(1, 2))
    produced = diffmean_scalar_grid(_META, cfg)
    assert produced == [1.0, 2.0]
    assert all(isinstance(value, float) for value in produced)


def test_only_none_means_default_not_any_falsy_value() -> None:
    """``None`` selects the default; an empty grid does NOT silently become it.

    The factory is duck-typed over ``cfg``, so it is reachable with values the
    dataclass domain would reject. Testing ``if override:`` instead of
    ``if override is not None:`` would make an empty grid quietly expand to the
    full 10-value sweep — a silently WRONG population rather than a visibly
    empty one. The dataclass separately forbids empty (see
    ``test_invalid_override_grids_are_rejected``); this pins the factory's own
    contract, which is reachable independently of that domain.
    """
    assert diffmean_scalar_grid(_META, SimpleNamespace(steering_coefficient_grid=())) == []


def test_unset_is_stripped_from_the_identity_payload() -> None:
    """Unset keeps every existing run tag byte-for-byte.

    Teeth: emitting the key unconditionally would change the identity of every
    already-scored clustered run and fail it at the next stage boundary.
    """
    payload = axbench_bpb_adv_config_to_dict(_cfg())
    assert "steering_coefficient_grid" not in payload


def test_set_participates_in_identity() -> None:
    """A restricted grid MUST change the tag — it scores a different population."""
    full = axbench_bpb_adv_config_to_dict(_cfg())
    pinned = axbench_bpb_adv_config_to_dict(
        dataclasses.replace(_cfg(), steering_coefficient_grid=(0.6, 0.8))
    )
    assert "steering_coefficient_grid" in pinned
    assert full != pinned


def test_two_different_overrides_have_different_identities() -> None:
    """Two restricted grids are distinguishable from each other, not just from unset."""
    a = axbench_bpb_adv_config_to_dict(
        dataclasses.replace(_cfg(), steering_coefficient_grid=(0.6,))
    )
    b = axbench_bpb_adv_config_to_dict(
        dataclasses.replace(_cfg(), steering_coefficient_grid=(0.8,))
    )
    assert a != b


@pytest.mark.parametrize(
    ("bad_grid", "why"),
    [
        ((), "empty would sweep nothing"),
        ((0.6, 0.6), "duplicates score one coefficient twice under one identity"),
        ((-0.1,), "a negative steering factor is not a point on this axis"),
        ((float("inf"),), "non-finite would poison the path slug and the stamp"),
    ],
)
def test_invalid_override_grids_are_rejected(bad_grid, why) -> None:
    """The declared tuple domain rejects each malformed grid.

    Pinned to ``BeartypeCallHintParamViolation`` rather than bare ``Exception``
    so a domain widened to a plain tuple still REDs here instead of passing on
    some unrelated error.
    """
    with pytest.raises(BeartypeCallHintParamViolation):
        _cfg(steering_coefficient_grid=bad_grid)
