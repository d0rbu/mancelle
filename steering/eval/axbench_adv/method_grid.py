"""The adv pipeline's ratified six-method factor grid.

The upstream comparison table
(:data:`scripts.eval.axbench_upstream.merge_methods.METHOD_FACTORS`) may grow
independently. Keep this map local so an upstream comparison change cannot
silently redefine an AxBench-Adv experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

ADV_FULL_FACTORS: tuple[float, ...] = (
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
)
ADV_INPUTS_PER_CONCEPT: int = 10

ADV_METHOD_FACTORS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "DiffMean": ADV_FULL_FACTORS,
        "LinearProbe": ADV_FULL_FACTORS,
        "LoRA": (1.0,),
        "PromptSteering": (1.0,),
        "SelfConditioning": ADV_FULL_FACTORS,
        "SelfConditioningMasked": ADV_FULL_FACTORS,
    }
)
ALL_METHOD_FACTORS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {**ADV_METHOD_FACTORS, "NLSteering": ADV_FULL_FACTORS}
)

__all__ = ["ADV_FULL_FACTORS", "ADV_INPUTS_PER_CONCEPT", "ADV_METHOD_FACTORS", "ALL_METHOD_FACTORS"]
