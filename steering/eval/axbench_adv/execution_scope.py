"""Strict environment parsing for one AxBench-Adv execution scope."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.types import validate_method_set, validate_method_subset

CONCEPT_LO_ENV = "AXBENCH_ADV_CONCEPT_LO"
CONCEPT_HI_ENV = "AXBENCH_ADV_CONCEPT_HI"
METHODS_ENV = "AXBENCH_ADV_METHODS"


def _bound(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value.isascii() or not value.isdecimal():
        raise AxBenchAdvError(f"{name} must be a base-10 non-negative integer")
    return int(value)


def parse_concept_ids(
    max_concepts: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Parse the optional half-open concept slice against the configured population."""

    if type(max_concepts) is not int or max_concepts <= 0:
        raise AxBenchAdvError("max_concepts must be a positive integer")
    environment = os.environ if environ is None else environ
    lo = _bound(environment, CONCEPT_LO_ENV, 0)
    hi = _bound(environment, CONCEPT_HI_ENV, max_concepts)
    if lo >= hi:
        raise AxBenchAdvError(
            f"concept slice must satisfy 0 <= {CONCEPT_LO_ENV} < {CONCEPT_HI_ENV}"
        )
    if hi > max_concepts:
        raise AxBenchAdvError(
            f"{CONCEPT_HI_ENV} must not exceed configured max_concepts={max_concepts}"
        )
    return tuple(range(lo, hi))


def validate_concept_ids(
    concept_ids: Sequence[int],
    *,
    max_concepts: int,
) -> tuple[int, ...]:
    """Validate an explicit ascending concept subset used by downstream stages."""

    if type(max_concepts) is not int or max_concepts <= 0:
        raise AxBenchAdvError("max_concepts must be a positive integer")
    parsed = tuple(concept_ids)
    if not parsed or any(type(concept_id) is not int for concept_id in parsed):
        raise AxBenchAdvError("concept_ids must be a nonempty sequence of integers")
    if parsed != tuple(sorted(set(parsed))):
        raise AxBenchAdvError("concept_ids must be unique and ascending")
    if parsed[0] < 0 or parsed[-1] >= max_concepts:
        raise AxBenchAdvError("concept_ids must lie within configured max_concepts")
    return parsed


def parse_method_filter(
    configured_methods: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Parse an optional comma-separated execution subset in canonical order."""

    configured = validate_method_set(configured_methods)
    environment = os.environ if environ is None else environ
    raw = environment.get(METHODS_ENV)
    if raw is None:
        return configured
    requested = tuple(part.strip() for part in raw.split(","))
    if any(not part for part in requested):
        raise AxBenchAdvError(f"{METHODS_ENV} must be a comma-separated nonempty method list")
    selected = validate_method_subset(requested)
    unknown = set(selected) - set(configured)
    if unknown:
        raise AxBenchAdvError(
            f"{METHODS_ENV} contains methods outside the configured source contract: "
            f"{sorted(unknown)}"
        )
    return tuple(method for method in configured if method in selected)


__all__ = [
    "CONCEPT_HI_ENV",
    "CONCEPT_LO_ENV",
    "METHODS_ENV",
    "parse_concept_ids",
    "parse_method_filter",
    "validate_concept_ids",
]
