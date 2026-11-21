"""Construct the exact attacked-prompt frame for a method/effort point."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pandas as pd

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import (
    ADV_INPUTS_PER_CONCEPT,
    ADV_METHOD_FACTORS,
)
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix, insert_suffix
from steering.eval.axbench_adv.types import PLOT_EFFORT_LEVELS

ATTACK_COLUMN: dict[str, str] = {
    "DiffMean": "input",
    "LinearProbe": "input",
    "LoRA": "input",
    "PromptSteering": "steered_input",
}


def _selected_factor_frame(source_frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    missing = sorted(set(INPUT_COLUMNS) - set(source_frame.columns))
    if missing:
        raise AxBenchAdvError(f"source frame is missing input columns {missing}")
    selected = source_frame.loc[
        source_frame["factor"].eq(float(factor)), list(INPUT_COLUMNS)
    ].copy()
    if selected.empty:
        raise AxBenchAdvError(f"source frame has no rows at factor {factor}")
    selected = selected.sort_values(["concept_id", "input_id"], kind="stable").reset_index(
        drop=True
    )
    if selected.duplicated(["concept_id", "input_id"]).any():
        raise AxBenchAdvError("selected frame has duplicate (concept_id, input_id) rows")
    concept_ids = sorted(int(value) for value in selected["concept_id"].unique())
    if concept_ids != list(range(len(concept_ids))):
        raise AxBenchAdvError("selected frame concept IDs must be contiguous from zero")
    for concept_id, group in selected.groupby("concept_id", sort=True):
        input_ids = sorted(int(value) for value in group["input_id"].tolist())
        if input_ids != list(range(ADV_INPUTS_PER_CONCEPT)):
            raise AxBenchAdvError(
                f"concept {concept_id} input IDs must be 0..{ADV_INPUTS_PER_CONCEPT - 1}"
            )
    return selected


def build_attacked_frame(
    *,
    source_frame: pd.DataFrame,
    suffixes: Mapping[int, str],
    method: str,
    factor: float,
    proposal_count: int,
    token_count: Callable[[str], int],
    max_prompt_tokens: int,
) -> pd.DataFrame:
    """Return a full-input-column frame with one method's suffix inserted.

    ``proposal_count == 0`` is the regenerated control and therefore requires
    every concept's canonical suffix to be empty.
    """

    if isinstance(proposal_count, bool) or not isinstance(proposal_count, int):
        raise AxBenchAdvError("proposal_count must be an integer")
    if proposal_count not in PLOT_EFFORT_LEVELS:
        raise AxBenchAdvError(f"proposal_count must be one of {PLOT_EFFORT_LEVELS}")
    if method not in ATTACK_COLUMN:
        raise AxBenchAdvError(f"unknown attacked method: {method!r}")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise AxBenchAdvError("factor must be numeric")
    numeric_factor = float(factor)
    if not math.isfinite(numeric_factor):
        raise AxBenchAdvError("factor must be finite")
    if numeric_factor not in tuple(float(value) for value in ADV_METHOD_FACTORS[method]):
        raise AxBenchAdvError(f"factor is not on {method}'s declared grid")
    selected = _selected_factor_frame(source_frame, numeric_factor)
    concept_ids = {int(value) for value in selected["concept_id"].unique()}
    supplied_ids = set(suffixes)
    if supplied_ids != concept_ids:
        raise AxBenchAdvError(
            f"suffix concept coverage must be exactly {sorted(concept_ids)}, "
            f"observed {sorted(supplied_ids)}"
        )
    canonical = {
        concept_id: canonicalize_suffix(suffixes[concept_id]) for concept_id in concept_ids
    }
    if proposal_count == 0 and any(canonical.values()):
        raise AxBenchAdvError("proposal_count=0 control requires an empty suffix per concept")

    target_column = ATTACK_COLUMN[method]
    override = selected.copy()
    override[target_column] = [
        insert_suffix(
            str(prompt),
            canonical[int(concept_id)],
            token_count=token_count,
            max_prompt_tokens=max_prompt_tokens,
        )
        for concept_id, prompt in zip(selected["concept_id"], selected[target_column], strict=True)
    ]
    if proposal_count == 0 and not override.equals(selected):
        raise AxBenchAdvError("empty-suffix control must be byte-for-byte frame-identical")
    return override


__all__ = ["ATTACK_COLUMN", "build_attacked_frame"]
