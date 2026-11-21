"""Dependency-light pressure identities shared with the isolated vendor worker."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum


class Pressure(StrEnum):
    GEPA = "gepa"
    LINEAR_PROBE_COUNTER = "linear_probe_counter"
    DIFFMEAN_COUNTER = "diffmean_counter"
    GCG = "gcg"
    SOFT_PROMPT = "soft_prompt"


class DirectionMethod(StrEnum):
    LINEAR_PROBE = "LinearProbe"
    DIFFMEAN = "DiffMean"


PRESSURES_ENV = "AXBENCH_ADV_PRESSURES"
DEFAULT_MAGNITUDES = (0.25, 0.5, 1.0, 2.0)
COEFFICIENT_SEMANTICS = "residual -= magnitude * saved_direction; raw_direction_norm_units"
DIRECTION_METHOD = {
    Pressure.LINEAR_PROBE_COUNTER: DirectionMethod.LINEAR_PROBE,
    Pressure.DIFFMEAN_COUNTER: DirectionMethod.DIFFMEAN,
}
DIRECTION_FAMILY = {
    Pressure.LINEAR_PROBE_COUNTER: "linear_probe",
    Pressure.DIFFMEAN_COUNTER: "diffmean",
}

# Issue #1673's completion receipt is permanently defined over these three
# pressure families.  Keep it independent of enum growth.
LEGACY_PRESSURES = (
    Pressure.GEPA,
    Pressure.LINEAR_PROBE_COUNTER,
    Pressure.DIFFMEAN_COUNTER,
)
COUNTER_PRESSURES = (
    Pressure.LINEAR_PROBE_COUNTER,
    Pressure.DIFFMEAN_COUNTER,
)


def parse_pressures(
    configured: Sequence[str] = (Pressure.GEPA,),
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Pressure, ...]:
    """Validate config even when overridden, and retain canonical pressure order."""
    if isinstance(configured, str) or not configured:
        raise ValueError("pressures must be a nonempty list")
    values = tuple(Pressure(value) for value in configured)
    if len(set(values)) != len(values):
        raise ValueError("pressures must be unique")
    raw = (os.environ if environ is None else environ).get(PRESSURES_ENV)
    if raw is not None:
        values = tuple(Pressure(part.strip()) for part in raw.split(","))
        if len(set(values)) != len(values):
            raise ValueError(f"{PRESSURES_ENV} must contain unique pressures")
    return tuple(pressure for pressure in Pressure if pressure in values)


def validate_magnitudes(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, str) or not values:
        raise ValueError("counter_magnitudes must be a nonempty positive grid")
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise ValueError(
            "counter_magnitudes must be finite and positive; zero is the shared control"
        )
    result = tuple(float(v) for v in values)
    if result != tuple(sorted(set(result))):
        raise ValueError("counter_magnitudes must be unique and ascending")
    return result


def magnitude_label(magnitude: float) -> str:
    return f"m{float(magnitude)!r}"
