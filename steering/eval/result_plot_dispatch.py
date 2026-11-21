"""Shared linestyle/color dispatch for SuiteResultVariant.

Plotting convention:
- StaticYaml: dashed (--), color-by-name.
- TargetRealLockstep: dotted (:), color-by-name.
- TargetRealPooled: dash-dot (-.), color-by-name.

Color-by-name uses sha256(name) modulo the C1..C20 tableau, so a
lockstep + pooled pair sharing the same name gets the same color
(visual pairing across linestyles).
"""

from __future__ import annotations

import hashlib

from steering.eval.progression_results import (
    SelfConditionedSuiteResult,
    StaticYamlSuiteResult,
    SuiteResultVariant,
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
)


def _color_by_name(name: str) -> str:
    """Stable per-name color from matplotlib's tab20 palette."""
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    return f"C{(h % 20) + 1}"


def _get_linestyle_for_variant(variant: SuiteResultVariant) -> tuple[str, str]:
    """Return (linestyle, color) per the PR-D plotting convention.

    Lockstep + pooled pairs sharing the same name share color, so they
    visually pair across the two distinct linestyles.
    """
    match variant:
        case SelfConditionedSuiteResult(name=n):
            return "-", _color_by_name(n)
        case StaticYamlSuiteResult(name=n):
            return "--", _color_by_name(n)
        case TargetRealLockstepSuiteResult(name=n):
            return ":", _color_by_name(n)
        case TargetRealPooledSuiteResult(name=n):
            return "-.", _color_by_name(n)
        case _:
            raise TypeError(f"unknown SuiteResultVariant: {type(variant).__name__}")
