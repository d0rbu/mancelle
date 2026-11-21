"""Feature-GCG families and method-name -> family classification."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from steering.eval.axbench_base.encoder_input import PartitionModifier
from steering.eval.axbench_base.types import SteeringMode


class GcgFamily(StrEnum):
    """The feature-GCG bag families.

    A ``StrEnum`` whose values are the exact ``"fair"`` / ``"roofline"`` strings
    the artifacts and paths have always used, so serialization is byte-identical
    (``str(GcgFamily.FAIR) == "fair"``): the persisted ``mode_tag`` metadata key
    (its NAME is kept for artifact compatibility), the ``paths.ours_gcg_*``
    directory slugs, and the scored-prefix keying are all unchanged.  It
    replaces the former ``_FAIR_MODE_TAG`` / ``_ROOFLINE_MODE_TAG`` magic
    strings.

    * ``FAIR``: the universal-over-the-optimize-subset derive (Task 5); the score
      stage scores each concept's HELD-OUT prefixes (clean generalization).
    * ``ROOFLINE``: the per-prefix in-sample upper bound (Task 6); the score
      stage scores that same HELD-OUT prefix population, with one oracle bag per
      held-out prefix.
    """

    FAIR = "fair"
    ROOFLINE = "roofline"


# The fair Phase-2 method names (Method.name == PartitionModifier.value) and the
# steering mode each one derives bags for.
FAIR_GCG_METHOD_MODES: dict[str, SteeringMode] = {
    PartitionModifier.OURS_GCG_ENCDEC.value: SteeringMode.POSITIVE,
    PartitionModifier.NEG_OURS_GCG_ENCDEC.value: SteeringMode.NEGATIVE,
    # The alpha sweep reads the SAME fair bags -- it differs only in which axis
    # its coefficient indexes (objective_alpha, not reg_lambda), so it belongs
    # here: a run selecting only these methods must still derive fair bags.
    PartitionModifier.OURS_GCG_ALPHA_ENCDEC.value: SteeringMode.POSITIVE,
    PartitionModifier.NEG_OURS_GCG_ALPHA_ENCDEC.value: SteeringMode.NEGATIVE,
}

# Bag-backed methods whose coefficient is the class-margin ``objective_alpha``
# rather than ``reg_lambda``. The bag family is unchanged (fair); only the
# meaning of the scalar the score stage carries differs, so this is a set of
# method NAMES, deliberately separate from the family mapping. Every consumer
# that turns a coefficient into a bag path must consult it -- reading a
# coefficient as the wrong axis silently loads a bag optimized under a
# different objective and reports it under this method's name.
ALPHA_AXIS_GCG_METHODS: frozenset[str] = frozenset(
    {
        PartitionModifier.OURS_GCG_ALPHA_ENCDEC.value,
        PartitionModifier.NEG_OURS_GCG_ALPHA_ENCDEC.value,
    }
)

# The roofline Phase-2 method names (the per-prefix in-sample upper bound) and
# the steering mode each one derives bags for — the sibling of the fair mapping.
ROOFLINE_GCG_METHOD_MODES: dict[str, SteeringMode] = {
    PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC.value: SteeringMode.POSITIVE,
    PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC.value: SteeringMode.NEGATIVE,
}

def active_gcg_modes(
    base_methods: Sequence[str] | None,
    method_modes: dict[str, SteeringMode],
) -> set[SteeringMode]:
    """Steering modes whose gcg method (a key of *method_modes*) is active.

    ``base_methods is None`` means the method whitelist is disabled, so ALL of
    *method_modes*' methods are active (a full run derives every bag); otherwise
    only the listed method names count.  This is the ONE home of the
    "which modes derive?" rule — the fair/roofline derive predicates both call
    it, so the classification can never drift between them.

    Roofline general-val is implemented per document, so there is no config-time
    roofline rejection; this helper is only a method-name classification utility.
    """
    if base_methods is None:
        return set(method_modes.values())
    active = set(base_methods)
    return {mode for name, mode in method_modes.items() if name in active}


__all__ = [
    "FAIR_GCG_METHOD_MODES",
    "ROOFLINE_GCG_METHOD_MODES",
    "GcgFamily",
    "active_gcg_modes",
]
