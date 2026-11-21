"""Edit-space-generic GCG optimisation loop.

Extracted from ``build_universal_gcg_candidates`` in
``steering/eval/axbench_bpb_adv/optimize.py`` but operates over any
``EditSpace`` implementation rather than the token-trigger space directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from steering.conditional.feature_gcg.aggregation import aggregate_objective, minibatch_for_step
from steering.conditional.feature_gcg.candidate_search import deterministic_step_rng
from steering.conditional.feature_gcg.edit_space import EditSpace, GroupResult, ValidatedGroup
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.refined_types import NonNegativeInt, PositiveInt

_RESERVED_EVENT_KEYS = frozenset({
    "event", "group_index", "total_groups", "objective_value", "accepted_steps",
    "visited_sequences", "control", "step_index", "max_steps", "candidate_count",
    "accepted", "best_value",
})


def _emit(cb: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if cb is not None:
        cb(event)


def _check_event_fields(provided: dict[str, Any], source: str) -> None:
    clash = _RESERVED_EVENT_KEYS & provided.keys()
    if clash:
        raise FeatureGcgError(
            f"EditSpace.{source} returned reserved progress-event key(s) {sorted(clash)}; "
            f"these collide with run_gcg's own event fields."
        )


def _group_objective(
    edit_space: EditSpace,
    members: Sequence[Any],
    target: Any,
    objective_batch: Callable[[Sequence[Any]], list[float]],
) -> float:
    return aggregate_objective(
        objective_batch([edit_space.model_item(m, target) for m in members])
    )


def _objectives_of_candidates(
    edit_space: EditSpace,
    members: Sequence[Any],
    candidate_targets: Sequence[Any],
    objective_batch: Callable[[Sequence[Any]], list[float]],
) -> list[float]:
    items = [edit_space.model_item(m, ct) for ct in candidate_targets for m in members]
    flat = objective_batch(items)
    width = len(members)
    return [aggregate_objective(flat[i * width:(i + 1) * width]) for i in range(len(candidate_targets))]


def run_gcg(
    *,
    edit_space: EditSpace,
    groups: Sequence[Any],
    objective_batch: Callable[[Sequence[Any]], list[float]],
    gradient_provider_batch: Callable[[Sequence[Any], int], list[Any]],
    cfg: Any,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    group_boundary_callback: Callable[[], None] | None = None,
    full_batch_pool_exhausted: Callable[[Any, Any, AbstractSet[str]], bool]
    | None = None,
    restore_group_result: Callable[[ValidatedGroup], GroupResult | None] | None = None,
    commit_group_result: Callable[[ValidatedGroup, GroupResult], None] | None = None,
) -> list[GroupResult]:
    """Run the GCG optimisation loop over each group in ``groups``.

    Accept gate: ``best_obj < _group_objective(edit_space, batch, target, objective_batch)``
    where ``batch`` is the CURRENT MINIBATCH (not the full eval batch).  On accept,
    ``cur_obj`` is re-measured on the full eval batch and the BEST full-batch target
    is retained as ``retained_target`` / ``retained_obj``.

    Tie-break: ``min(range(len(values)), key=...)`` returns the FIRST index on ties.

    When ``full_batch_pool_exhausted`` is supplied, groups whose complete member
    count fits in ``universal_minibatch_size`` retain one ranked proposal mapping
    per current target and may stop after an empty sample proves exact exhaustion.
    Other callers keep the per-step proposal path unchanged.
    """

    results: list[GroupResult] = []
    total = len(groups)

    for gi, group in enumerate(groups, start=1):
        if group_boundary_callback is not None:
            group_boundary_callback()

        validated: ValidatedGroup = edit_space.validate(group)
        fields = edit_space.group_event_fields(validated)
        _check_event_fields(fields, "group_event_fields")

        if edit_space.is_trivial(validated):
            results.append(edit_space.finalize_trivial(validated))
            _emit(
                progress_callback,
                event="group_done",
                group_index=gi,
                total_groups=total,
                objective_value=None,
                accepted_steps=0,
                visited_sequences=0,
                control=True,
                **fields,
            )
            continue

        if restore_group_result is not None:
            restored = restore_group_result(validated)
            if restored is not None:
                results.append(restored)
                continue

        # step_event_fields is only needed on the optimization path (control groups skip it).
        step_fields = edit_space.step_event_fields(validated)
        _check_event_fields(step_fields, "step_event_fields")
        members = validated.members
        target = edit_space.init_target(validated)
        cur_obj = _group_objective(edit_space, members, target, objective_batch)
        retained_target, retained_obj = target, cur_obj
        visited: set[str] = {edit_space.digest(target)}
        accepted_steps = 0
        fixed_pool_eligible = (
            full_batch_pool_exhausted is not None
            and len(members) <= int(cfg.universal_minibatch_size)
        )
        cached_proposals: Any | None = None

        _emit(
            progress_callback,
            event="group_start",
            group_index=gi,
            total_groups=total,
            objective_value=cur_obj,
            **fields,
        )

        for step in range(int(cfg.max_steps)):
            # Per-phase profiling of the SHARED production GCG loop (issue #657).
            # These regions wrap the model-heavy calls at the eager level (the
            # gradient backward and the candidate objective forwards) -- NOT any
            # compiled graph -- so they are byte-identical: profile_region is a
            # no-op when no profiler is bound (every non-BPB-Adv feature_gcg
            # caller), and under a bound profiler it only adds a sync + snapshot.
            # They answer "of feature-GCG time, how much is gradient vs
            # candidate-eval", which the family-level regions cannot.
            record_counter("feature_gcg.run_gcg.steps", 1)
            idx = minibatch_for_step(
                NonNegativeInt.parse(len(members)),
                step=NonNegativeInt.parse(step),
                minibatch_size=PositiveInt.parse(int(cfg.universal_minibatch_size)),
                group_seed=f"{validated.key_str}|seed{cfg.master_seed}",
            )
            batch = [members[i] for i in idx]
            if not fixed_pool_eligible or cached_proposals is None:
                with profile_region("feature_gcg.run_gcg.gradient"):
                    grad_items = [edit_space.model_item(m, target) for m in batch]
                    grads = gradient_provider_batch(grad_items, step)
                    agg = edit_space.aggregate_gradient(zip(grads, batch, strict=True))
                with profile_region("feature_gcg.run_gcg.rank_candidates"):
                    proposals = edit_space.rank_candidates(agg, target, cfg)
                if fixed_pool_eligible:
                    cached_proposals = proposals
            else:
                proposals = cached_proposals
            rng = deterministic_step_rng(
                candidate_seed=validated.key_str,
                step=step,
                master_seed=cfg.master_seed,
            )
            candidates = edit_space.sample_candidates(
                proposals,
                target,
                batch_size=int(cfg.candidate_batch_size),
                temperature=float(cfg.temperature),
                rng=rng,
                visited=visited,
            )

            if not candidates:
                _emit(
                    progress_callback,
                    event="step_done",
                    group_index=gi,
                    total_groups=total,
                    step_index=step,
                    max_steps=int(cfg.max_steps),
                    candidate_count=0,
                    accepted=False,
                    objective_value=cur_obj,
                    best_value=None,
                    visited_sequences=len(visited),
                    **step_fields,
                )
                if fixed_pool_eligible and full_batch_pool_exhausted is not None:
                    if full_batch_pool_exhausted(proposals, target, visited):
                        break
                continue

            cand_targets = list(candidates)
            with profile_region(
                "feature_gcg.run_gcg.candidate_eval", candidates=len(cand_targets)
            ):
                values = _objectives_of_candidates(
                    edit_space, batch, cand_targets, objective_batch
                )

            # First-as-best tie-break: min keeps the earliest candidate on a tie.
            best_i = min(range(len(values)), key=lambda i: values[i])
            best_obj = values[best_i]
            best_target = cand_targets[best_i]

            # Accept iff best_obj strictly improves over the CURRENT TARGET on the SAME MINIBATCH
            current_minibatch_obj = _group_objective(edit_space, batch, target, objective_batch)
            accepted = best_obj < current_minibatch_obj

            if accepted:
                target = best_target
                cur_obj = _group_objective(edit_space, members, target, objective_batch)
                accepted_steps += 1
                cached_proposals = None
                record_counter("feature_gcg.run_gcg.steps_accepted", 1)
                if cur_obj < retained_obj:
                    retained_obj, retained_target = cur_obj, target

            _emit(
                progress_callback,
                event="step_done",
                group_index=gi,
                total_groups=total,
                step_index=step,
                max_steps=int(cfg.max_steps),
                candidate_count=len(candidates),
                accepted=accepted,
                objective_value=cur_obj,
                best_value=best_obj,
                visited_sequences=len(visited),
                **step_fields,
            )

        result = edit_space.finalize(
            validated,
            retained_target,
            objective=retained_obj,
            accepted_steps=accepted_steps,
            visited_count=len(visited),
        )
        if commit_group_result is not None:
            commit_group_result(validated, result)
        results.append(result)
        _emit(
            progress_callback,
            event="group_done",
            group_index=gi,
            total_groups=total,
            objective_value=retained_obj,
            accepted_steps=accepted_steps,
            visited_sequences=len(visited),
            control=False,
            **fields,
        )

    return results
