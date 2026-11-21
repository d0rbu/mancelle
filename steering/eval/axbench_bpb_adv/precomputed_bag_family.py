"""Precomputed conditioning-bag families and method-name classification.

This module is the neutral home for BPB/BPB-Adv methods that load conditioning
bags from disk. Feature-GCG and feature-DiffMean both use the same artifact
layout, but feature-DiffMean is not a GCG method.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from steering.eval.axbench_base.encoder_input import PartitionModifier
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.gcg_family import (
    FAIR_GCG_METHOD_MODES,
    ROOFLINE_GCG_METHOD_MODES,
    GcgFamily,
    active_gcg_modes,
)


class PrecomputedBagFamily(StrEnum):
    """Families stored under the historical ``ours_gcg_bags`` artifact tree."""

    FAIR_GCG = GcgFamily.FAIR.value
    ROOFLINE_GCG = GcgFamily.ROOFLINE.value
    FAIR_FEATURE_DIFFMEAN = "feature_diffmean"
    ROOFLINE_FEATURE_DIFFMEAN = "feature_diffmean_roofline"


FAIR_FEATURE_DIFFMEAN_METHOD_MODES: dict[str, SteeringMode] = {
    PartitionModifier.FEATURE_DIFFMEAN_ENCDEC.value: SteeringMode.POSITIVE,
    PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC.value: SteeringMode.NEGATIVE,
}

ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES: dict[str, SteeringMode] = {
    PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC.value: SteeringMode.POSITIVE,
    PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC.value: SteeringMode.NEGATIVE,
}


PRECOMPUTED_BAG_METHOD_NAME_TO_FAMILY: dict[str, PrecomputedBagFamily] = {
    **dict.fromkeys(FAIR_GCG_METHOD_MODES, PrecomputedBagFamily.FAIR_GCG),
    **dict.fromkeys(ROOFLINE_GCG_METHOD_MODES, PrecomputedBagFamily.ROOFLINE_GCG),
    **dict.fromkeys(FAIR_FEATURE_DIFFMEAN_METHOD_MODES, PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN),
    **dict.fromkeys(ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES, PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN),
}

ROOFLINE_PRECOMPUTED_BAG_FAMILIES: frozenset[PrecomputedBagFamily] = frozenset(
    {
        PrecomputedBagFamily.ROOFLINE_GCG,
        PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
    }
)


def coerce_precomputed_bag_family(
    family: PrecomputedBagFamily | GcgFamily | str,
) -> PrecomputedBagFamily:
    """Coerce exact historical family values to :class:`PrecomputedBagFamily`."""
    return PrecomputedBagFamily(str(family))


def precomputed_bag_method_family(method_name: str) -> PrecomputedBagFamily | None:
    """Return the precomputed-bag family for *method_name*, or ``None``."""
    return PRECOMPUTED_BAG_METHOD_NAME_TO_FAMILY.get(method_name)


def active_precomputed_bag_modes(
    base_methods: Sequence[str] | None,
    method_modes: dict[str, SteeringMode],
) -> set[SteeringMode]:
    """Steering modes whose precomputed-bag methods are active.

    Delegates to :func:`active_gcg_modes`, the single home of the "which modes
    are active given *base_methods*" rule, so the feature-GCG and feature-DiffMean
    derives can never drift on which modes to build bags for.
    """
    return active_gcg_modes(base_methods, method_modes)


__all__ = [
    "FAIR_FEATURE_DIFFMEAN_METHOD_MODES",
    "PRECOMPUTED_BAG_METHOD_NAME_TO_FAMILY",
    "ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES",
    "ROOFLINE_PRECOMPUTED_BAG_FAMILIES",
    "PrecomputedBagFamily",
    "active_precomputed_bag_modes",
    "coerce_precomputed_bag_family",
    "precomputed_bag_method_family",
]
