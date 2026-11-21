"""Placement-free AxBench-Upstream selection and generation seed contracts."""

from __future__ import annotations

import hashlib
import math

from scripts.eval.axbench_upstream.merge_methods import (
    NL_PROJECT_METHODS,
    FrameSource,
    methods_for_source,
)

SEED_MODULUS = 2**31 - 1
MAX_SIGNED_63_BIT = (1 << 63) - 1
_GENERATION_SEED_METHOD_ORDER = tuple(
    method
    for source in (FrameSource.ENCDEC_MAIN, FrameSource.ENCDEC_AUX)
    for method in methods_for_source(source)
) + NL_PROJECT_METHODS
_GENERATION_SEED_METHODS = frozenset(_GENERATION_SEED_METHOD_ORDER)


def _seed_identity_integer(value: int, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SIGNED_63_BIT:
        raise ValueError(f"{name} must be an integer in [0, 2^63 - 1]")
    return value


def derive_generation_seed(
    base_seed: int,
    method: str,
    concept_id: int,
    input_id: int,
    factor: float,
) -> int:
    """Derive a process-stable seed for one method/input/factor cell."""

    base_seed = _seed_identity_integer(base_seed, name="base_seed")
    concept_id = _seed_identity_integer(concept_id, name="concept_id")
    input_id = _seed_identity_integer(input_id, name="input_id")
    if method not in _GENERATION_SEED_METHODS:
        choices = ", ".join(repr(name) for name in _GENERATION_SEED_METHOD_ORDER[:-1])
        raise ValueError(
            "generation seed method must be one of "
            f"{choices}, or {_GENERATION_SEED_METHOD_ORDER[-1]!r}"
        )
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError("generation seed factor must be numeric")
    numeric_factor = float(factor)
    if not math.isfinite(numeric_factor) or numeric_factor < 0:
        raise ValueError("generation seed factor must be finite and non-negative")

    fields = (
        str(base_seed),
        method,
        str(concept_id),
        str(input_id),
        numeric_factor.hex(),
    )
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def derive_selection_seed(base_seed: int, concept_id: int, input_id: int) -> int:
    """Derive one factor-independent feature-cardinality seed per input."""

    base_seed = _seed_identity_integer(base_seed, name="base_seed")
    concept_id = _seed_identity_integer(concept_id, name="concept_id")
    input_id = _seed_identity_integer(input_id, name="input_id")

    fields = (
        "axbench-upstream-selfconditioning-selection-v1",
        str(base_seed),
        str(concept_id),
        str(input_id),
    )
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


__all__ = (
    "MAX_SIGNED_63_BIT",
    "SEED_MODULUS",
    "derive_generation_seed",
    "derive_selection_seed",
)
