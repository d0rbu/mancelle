"""Per-prefix Faster-GCG reference oracle (TEST-ONLY).

``build_faster_gcg_candidates`` is the frozen backward-compat reference that the
production universal optimizer (``build_universal_gcg_candidates`` in
``steering.eval.axbench_bpb_adv.optimize``) is proven byte-identical against
(``test_universal_gcg_equiv_golden``). It has NO production callers, so it lives
here rather than cluttering the production module. The private helpers it uses
(``_candidate_row``, ``_validate_attack_row(s)``, ``build_control_candidate``,
etc.) ARE production code -- shared with the universal path -- so they stay in
``optimize`` and are imported here.

The body must stay byte-for-byte behaviourally identical to the captured golden
parquet; do not "improve" it. The real feature-GCG loop is ``run_gcg`` in
``steering/conditional/feature_gcg/core.py`` (issue #657).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa
import torch as t

from steering.conditional.feature_gcg.candidate_search import (
    CandidateSequence,
    deterministic_step_rng,
    rank_replacements_from_gradients,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.core import _emit
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_bpb_adv.candidate_metadata import FasterGcgFinalMetadata
from steering.eval.axbench_bpb_adv.config import AdversarialMethod, FasterGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.optimize import (
    ProgressCallback,
    _candidate_row,
    _validate_attack_row,
    _validate_attack_rows,
    build_control_candidate,
)
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA
from steering.refined_types import NonNegativeInt

ObjectiveFn = Callable[[dict[str, Any], Sequence[int]], float]
GradientProvider = Callable[[dict[str, Any], Sequence[int], int], t.Tensor]


def _choose_best_candidate(
    candidates: Sequence[CandidateSequence],
    objective: Callable[[Sequence[int]], float],
) -> tuple[CandidateSequence, float]:
    """Frozen scalar selection used only by the historical test oracle."""
    if not candidates:
        raise FeatureGcgError("cannot choose from an empty candidate list")
    best_candidate = candidates[0]
    best_value = float(objective(best_candidate.prompt_tokens))
    for candidate in candidates[1:]:
        value = float(objective(candidate.prompt_tokens))
        if value < best_value:
            best_candidate = candidate
            best_value = value
    return best_candidate, best_value


def build_faster_gcg_candidates(
    *,
    attack_rows: pa.Table,
    tokenizer: Any,
    cfg: FasterGcgConfig,
    embedding_weight: t.Tensor,
    allowed_token_ids: Sequence[int],
    objective: ObjectiveFn,
    gradient_provider: GradientProvider,
    progress_callback: ProgressCallback | None = None,
) -> pa.Table:
    """Build one final Faster-GCG candidate per attack row (test-only oracle)."""

    _validate_attack_rows(attack_rows)
    rows: list[dict[str, Any]] = []
    source_rows = [
        row
        for row in attack_rows.to_pylist()
        if str(row["adversarial_method"]) == AdversarialMethod.FASTER_GCG.value
    ]
    total_rows = len(source_rows)
    for row_index, row in enumerate(source_rows, start=1):
        source_tokens, span = _validate_attack_row(row)
        if span.count == 0:
            rows.append(build_control_candidate(row=row, tokenizer=tokenizer))
            _emit(
                progress_callback,
                event="row_done",
                row_index=row_index,
                total_rows=total_rows,
                attack_row_id=str(row["attack_row_id"]),
                target_model=str(row["target_model"]),
                method_name=str(row["method_name"]),
                coefficient_str=str(row["coefficient_str"]),
                adversarial_strength=float(row["adversarial_strength"]),
                context_signal_mode=str(row["context_signal_mode"]),
                editable_token_count=0,
                objective_value=None,
                accepted_steps=0,
                visited_sequences=1,
                control=True,
            )
            continue

        current_tokens = list(source_tokens)
        current_value = float(objective(row, current_tokens))
        visited = {token_sequence_digest(current_tokens)}
        accepted_steps = 0
        _emit(
            progress_callback,
            event="row_start",
            row_index=row_index,
            total_rows=total_rows,
            attack_row_id=str(row["attack_row_id"]),
            target_model=str(row["target_model"]),
            method_name=str(row["method_name"]),
            coefficient_str=str(row["coefficient_str"]),
            adversarial_strength=float(row["adversarial_strength"]),
            context_signal_mode=str(row["context_signal_mode"]),
            editable_token_count=span.count,
            objective_value=current_value,
        )
        for step in range(int(cfg.max_steps)):
            gradient = gradient_provider(row, current_tokens, step)
            try:
                replacements = rank_replacements_from_gradients(
                    embedding_gradient=gradient,
                    embedding_weight=embedding_weight,
                    editable_positions=list(span.positions()),
                    allowed_token_ids=allowed_token_ids,
                    current_tokens=current_tokens,
                    top_k=cfg.top_k,
                    distance_weight=cfg.distance_weight,
                )
            except FeatureGcgError as exc:
                raise AxBenchBpbAdvError(str(exc)) from exc
            try:
                candidates = sample_candidate_sequences(
                    current_tokens=current_tokens,
                    replacements_by_position=replacements,
                    batch_size=cfg.candidate_batch_size,
                    temperature=cfg.temperature,
                    rng=deterministic_step_rng(
                        candidate_seed=str(row["attack_row_id"]),
                        step=step,
                        master_seed=cfg.master_seed,
                    ),
                    visited_digests=visited,
                )
            except FeatureGcgError as exc:
                raise AxBenchBpbAdvError(str(exc)) from exc
            if not candidates:
                _emit(
                    progress_callback,
                    event="step_done",
                    row_index=row_index,
                    total_rows=total_rows,
                    attack_row_id=str(row["attack_row_id"]),
                    target_model=str(row["target_model"]),
                    method_name=str(row["method_name"]),
                    step_index=step,
                    max_steps=int(cfg.max_steps),
                    candidate_count=0,
                    accepted=False,
                    objective_value=current_value,
                    best_value=None,
                    visited_sequences=len(visited),
                )
                continue
            try:
                best, best_value = _choose_best_candidate(
                    candidates,
                    # row is consumed immediately within this iteration (before
                    # it rebinds), so the loop-var capture is intentional/safe.
                    lambda tokens: objective(row, tokens),  # noqa: B023
                )
            except FeatureGcgError as exc:
                raise AxBenchBpbAdvError(str(exc)) from exc
            accepted = best_value < current_value
            if best_value < current_value:
                current_tokens = list(best.prompt_tokens)
                current_value = best_value
                accepted_steps += 1
            _emit(
                progress_callback,
                event="step_done",
                row_index=row_index,
                total_rows=total_rows,
                attack_row_id=str(row["attack_row_id"]),
                target_model=str(row["target_model"]),
                method_name=str(row["method_name"]),
                step_index=step,
                max_steps=int(cfg.max_steps),
                candidate_count=len(candidates),
                accepted=accepted,
                objective_value=current_value,
                best_value=best_value,
                visited_sequences=len(visited),
            )
        rows.append(
            _candidate_row(
                row=row,
                tokenizer=tokenizer,
                source_tokens=source_tokens,
                attacked_tokens=current_tokens,
                objective=current_value,
                metadata=FasterGcgFinalMetadata(
                    max_steps=NonNegativeInt.parse(int(cfg.max_steps)),
                    accepted_steps=NonNegativeInt.parse(accepted_steps),
                    visited_sequences=NonNegativeInt.parse(len(visited)),
                ),
            )
        )
        _emit(
            progress_callback,
            event="row_done",
            row_index=row_index,
            total_rows=total_rows,
            attack_row_id=str(row["attack_row_id"]),
            target_model=str(row["target_model"]),
            method_name=str(row["method_name"]),
            coefficient_str=str(row["coefficient_str"]),
            adversarial_strength=float(row["adversarial_strength"]),
            context_signal_mode=str(row["context_signal_mode"]),
            editable_token_count=span.count,
            objective_value=current_value,
            accepted_steps=accepted_steps,
            visited_sequences=len(visited),
            control=False,
        )
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)
