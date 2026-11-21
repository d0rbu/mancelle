"""Shared pairing helper for synth Stage 1b/1c (issue #508).

A concept whose rows are all ``concept_rich`` with zero paired
``concept_avoiding`` rows would crash Stage 3 with "Empty negatives". Both synth
modes instead drop the unpaired rich rows so the concept is simply *absent* from
the corpus (consistent with the chosen non-"error" qc_fallback policy).
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
)


def drop_unpaired_rich_rows(
    rows: list[dict[str, Any]], concept_id: int,
) -> list[dict[str, Any]]:
    """Drop a concept's rows entirely if it has ``concept_rich`` rows but no
    ``concept_avoiding`` row; otherwise return ``rows`` unchanged.

    Mode-agnostic (positive Stage 1b + negative Stage 1c). Returns ``[]`` when
    ``n_rich > 0 and n_avoid == 0``; ``[]``, avoid-only, and paired inputs pass
    through unchanged.
    """
    n_rich = sum(1 for r in rows if r["content_class"] == CONCEPT_RICH_LABEL)
    n_avoid = sum(1 for r in rows if r["content_class"] == CONCEPT_AVOIDING_LABEL)
    if n_rich > 0 and n_avoid == 0:
        logger.warning(
            f"Concept {concept_id}: {n_rich} concept_rich rows but 0 "
            "concept_avoiding rows (all avoid QC dropped under a non-'error' "
            "fallback); dropping the unpaired rich rows so the concept is "
            "absent rather than crashing Stage 3 with empty negatives."
        )
        return []
    return rows
