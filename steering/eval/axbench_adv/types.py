"""Closed constants shared by AxBench-Adv's pure stages."""

from __future__ import annotations

from collections.abc import Collection

from steering.eval.axbench_adv.errors import AxBenchAdvError

QWEN_METHODS: tuple[str, ...] = ("DiffMean", "LinearProbe", "LoRA", "PromptSteering")
SELF_CONDITIONING_METHODS: tuple[str, ...] = ("SelfConditioning", "SelfConditioningMasked")
METHODS: tuple[str, ...] = (*QWEN_METHODS, *SELF_CONDITIONING_METHODS)
NL_METHOD = "NLSteering"
NATIVE_METHOD = "NativeDecoderPromptSteering"
ENCDEC_BASELINE_METHODS: tuple[str, ...] = (
    "EncDecPromptSteering",
    "EncDecDiffMean",
    "EncDecLinearProbe",
    "EncDecLoRA",
)
ALL_METHODS: tuple[str, ...] = (*METHODS, NL_METHOD)
ROOT_METHODS: tuple[str, ...] = (*SELF_CONDITIONING_METHODS, NL_METHOD)
PROPOSAL_COUNTS: tuple[int, ...] = (1, 2, 4, 8)
PLOT_EFFORT_LEVELS: tuple[int, ...] = (0, *PROPOSAL_COUNTS)


def validate_method_set(methods: Collection[str]) -> tuple[str, ...]:
    """Return canonical order for the six methods, optionally with retrieval NL."""
    for allowed in (METHODS, ALL_METHODS):
        if len(methods) == len(allowed) and set(methods) == set(allowed):
            return allowed
    raise AxBenchAdvError("methods must contain exactly the six baseline methods, optionally plus NLSteering")


def validate_method_subset(methods: Collection[str]) -> tuple[str, ...]:
    """Return canonical order for a nonempty subset of AxBench-Adv methods."""

    requested = tuple(methods)
    if not requested:
        raise AxBenchAdvError("methods must be a nonempty subset of the configured methods")
    if len(set(requested)) != len(requested):
        raise AxBenchAdvError("methods must be unique")
    if requested == (NATIVE_METHOD,):
        return requested
    unknown = set(requested) - set(ALL_METHODS)
    if unknown:
        raise AxBenchAdvError(f"unknown AxBench-Adv methods: {sorted(unknown)}")
    return tuple(method for method in ALL_METHODS if method in requested)


__all__ = [
    "ALL_METHODS",
    "ENCDEC_BASELINE_METHODS",
    "METHODS",
    "NL_METHOD",
    "PLOT_EFFORT_LEVELS",
    "PROPOSAL_COUNTS",
    "QWEN_METHODS",
    "ROOT_METHODS",
    "SELF_CONDITIONING_METHODS",
    "validate_method_set",
    "validate_method_subset",
]
