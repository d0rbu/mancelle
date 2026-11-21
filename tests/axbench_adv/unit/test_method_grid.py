"""Pin axbench_adv's ratified six-method grid against upstream growth."""

from __future__ import annotations

import pytest

from scripts.eval.axbench_upstream.merge_methods import METHOD_FACTORS
from steering.eval.axbench_adv.method_grid import (
    ADV_FULL_FACTORS,
    ADV_INPUTS_PER_CONCEPT,
    ADV_METHOD_FACTORS,
)
from steering.eval.axbench_adv.types import METHODS, SELF_CONDITIONING_METHODS

_RATIFIED_FULL_FACTORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_adv_method_set_is_the_ratified_six_method_grid() -> None:
    # Literal values on purpose: the ratified grid must not follow the
    # upstream FULL_FACTORS if that tuple is ever widened.
    assert dict(ADV_METHOD_FACTORS) == {
        "DiffMean": _RATIFIED_FULL_FACTORS,
        "LinearProbe": _RATIFIED_FULL_FACTORS,
        "LoRA": (1.0,),
        "PromptSteering": (1.0,),
        "SelfConditioning": _RATIFIED_FULL_FACTORS,
        "SelfConditioningMasked": _RATIFIED_FULL_FACTORS,
    }
    assert ADV_FULL_FACTORS == _RATIFIED_FULL_FACTORS
    assert SELF_CONDITIONING_METHODS == ("SelfConditioning", "SelfConditioningMasked")


def test_adv_method_set_is_decoupled_from_the_upstream_table() -> None:
    # The upstream comparison table grew to ten methods (the EncDec baseline
    # analogs); the adv grid deliberately did NOT follow — that is the exact
    # decoupling this module's docstring promises. The six ratified adv
    # methods must still exist upstream (their cells are judged there), while
    # the upstream-only additions must stay out of the adv grid.
    assert set(ADV_METHOD_FACTORS) < set(METHOD_FACTORS)
    assert set(METHOD_FACTORS) - set(ADV_METHOD_FACTORS) == {
        "EncDecDiffMean",
        "EncDecLinearProbe",
        "EncDecPromptSteering",
        "EncDecLoRA",
    }
    assert tuple(ADV_METHOD_FACTORS) == METHODS
    assert tuple(METHOD_FACTORS) != METHODS


def test_adv_method_grid_covers_exactly_the_pipeline_method_tuple() -> None:
    # types.METHODS drives the worker/judge loops that index this grid; a key
    # mismatch there surfaces as a bare mid-run KeyError, so pin the sets.
    assert set(ADV_METHOD_FACTORS) == set(METHODS)


def test_adv_method_grid_is_immutable() -> None:
    with pytest.raises(TypeError):
        ADV_METHOD_FACTORS["SelfConditioningMasked"] = _RATIFIED_FULL_FACTORS  # ty: ignore[invalid-assignment]


def test_adv_grid_geometry_literals_are_pinned() -> None:
    # The factor values and per-concept input count are ratified literals; a
    # widened upstream FULL_FACTORS/INPUTS_PER_CONCEPT must not reach the adv
    # key-product and declared-ratings checks.
    assert ADV_FULL_FACTORS == _RATIFIED_FULL_FACTORS
    assert ADV_INPUTS_PER_CONCEPT == 10
