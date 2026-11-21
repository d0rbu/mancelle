"""Stage 3: bpb_score — enumerate work units and partition across DDP ranks.

Per spec §11: stable SHA256-based hash partitioning. A WorkUnit is a
(method, concept, sample, coefficient) tuple; each rank owns exactly the
subset of units whose hash mod world_size equals rank.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from loguru import logger

from steering.eval.axbench_base.direction_scale import (
    STEERING_SCALE_MARKER_NAME,
    DirectionScaleRequest,
    enforce_steering_scale_identity,
    preflight_direction_scales,
)
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_bpb.bpb_cache import BpbCache, reconcile_cache
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.live_ours_vocab_contract import (
    enforce_live_ours_vocab_contract,
)
from steering.eval.axbench_bpb.parquet_writer import write_bpb_score_rows
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    score_sample,
)
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    BpbScoreRow,
    LoraParams,
    ScoringInputs,
    SteeringMode,
    TargetModel,
    WorkUnit,
)


def is_mine(unit: WorkUnit, *, rank: int, world_size: int) -> bool:
    """Stable hash-based partitioning across DDP ranks.

    Returns True iff ``unit`` belongs to ``rank`` in a world of ``world_size``
    ranks. Partitioning is deterministic across processes and across
    invocations — two calls with identical inputs always agree.
    """
    payload = f"{unit.method_name}|{unit.concept_id}|{unit.sample_id}|{unit.coefficient_str}"
    h = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
    return h % world_size == rank


def enumerate_work_units(
    *,
    methods: Iterable[Any],
    concepts: Iterable[Any],
    samples_by_concept_mode: Mapping[tuple[int, Any], Iterable[Any]],
    coefficient_grids: Mapping[str, Iterable[Any]],
) -> Iterator[WorkUnit]:
    """Yield every ``(method, concept, sample, coefficient)`` work unit.

    Order: methods × concepts × samples × coefficients.

    Args:
        methods: Iterable of method dataclasses; each must expose
            ``.name`` (str) and ``.mode`` (SteeringMode or compatible).
        concepts: Iterable of concept dataclasses with ``.concept_id``.
        samples_by_concept_mode: Keyed by (concept_id, mode); each value
            is a list of objects with ``.sample_id``.
        coefficient_grids: Keyed by method.name; each value is the
            iterable of coefficient values for that method.
    """
    for method in methods:
        for concept in concepts:
            key = (concept.concept_id, method.mode)
            for sample in samples_by_concept_mode.get(key, []):
                for coef in coefficient_grids[method.name]:
                    yield WorkUnit(
                        method_name=method.name,
                        concept_id=concept.concept_id,
                        sample_id=sample.sample_id,
                        coefficient_str=str(coef),
                    )


def _coefficient_from_str(s: str) -> int | float | None:
    """Reverse of str(coef). Handles None/int/float/scientific.

    Raises:
        AxBenchBpbError: if ``s`` is neither "None" nor a valid numeric
            string — surface failures at the conversion boundary rather
            than deep inside intervention dispatch.
    """
    if s == "None":
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError as exc:
        raise AxBenchBpbError(f"unrecognized coefficient string: {s!r}") from exc


def _next_batch_index(rank_dir: Path) -> int:
    """Return the next batch_NNNNNN index based on existing shards."""
    if not rank_dir.exists():
        return 0
    existing = sorted(rank_dir.glob("batch_*.parquet"))
    if not existing:
        return 0
    last = existing[-1].stem  # "batch_000042"
    try:
        return int(last.split("_")[-1]) + 1
    except ValueError:
        return len(existing)


def run_bpb_score_stage(
    *,
    cfg: Any,  # AxBenchBpbConfig
    output_paths: Any,
    source_paths: Any,
    methods: list[Any],
    concepts: list[Any],
    samples_by_concept_mode: Mapping[tuple[int, Any], Iterable[ScoringInputs]],
    coefficient_grids: Mapping[str, Iterable[Any]],
    decoder_only_bundle: TargetModelBundle,
    encdec_bundle: TargetModelBundle,
    target_model_metas: dict[str, Any],
    tokenizer_nanochat: Any,
    device: Any,
    ddp_rank: int,
    ddp_world_size: int,
    diffmean_layer: int,
    inactive_sampler: Callable[..., dict[int, int]] | None = None,
    precomputed_bag_paths: Any | None = None,
    lora: LoraParams | None = None,
    batch_flush_size: int = 1000,
) -> Path:
    """Run the bpb_score stage on ``ddp_rank``.

    Enumerates every (method, concept, sample, coefficient) work unit,
    filters to rank-owned via hash partitioning, builds the intervention
    per spec §10, scores via ``score_sample``, and flushes results to
    rank-local parquet shards every ``batch_flush_size`` rows. The
    ``BpbCache`` provides O(1) skip-if-already-scored lookups; on resume
    ``reconcile_cache`` rebuilds the index from existing parquet.

    Returns the rank's parquet directory.
    """
    rank_dir = output_paths.bpb_scores_rank_dir(ddp_rank)
    cache_path = output_paths.bpb_cache_path(ddp_rank)

    # The cache key predates checkpoint-V-restricted live ``ours`` semantics.
    # Guard the run-level durable rows before reconciliation or any cache skip,
    # so pre-fix rows can never be silently mixed into this scoring contract.
    enforce_live_ours_vocab_contract(
        scores_dir=rank_dir.parent,
        methods=methods,
        model=encdec_bundle.model,
    )

    # Rebuild cache from any durable parquet shards (handles prior crashes)
    reconcile_cache(rank_dir=rank_dir, cache_path=cache_path)

    scalar_methods = [
        m
        for m in methods
        if getattr(m, "baseline", "")
        in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
    ]
    preflight_values: dict[tuple[str, str, str, int], float] = {}
    if scalar_methods:
        # Snapshot the exact direction-owned values before accepting cache hits
        # or writing durable rows. The marker and post-build comparison bind the
        # resumed stage to this artifact/value set.
        requests = []
        for method in scalar_methods:
            baseline = BaselineKind(method.baseline)
            target = TargetModel.coerce(method.target_model)
            for concept in concepts:
                path_fn = (
                    source_paths.diffmean_direction
                    if baseline is BaselineKind.DIFFMEAN
                    else source_paths.linear_probe_direction
                )
                requests.append(
                    DirectionScaleRequest(
                        family=baseline.value,
                        target_model=target.value,
                        mode=method.mode.value,
                        concept_id=concept.concept_id,
                        path=path_fn(target, method.mode.value, concept.concept_id),
                    )
                )
        preflight = preflight_direction_scales(
            requests,
            expected_layer=diffmean_layer,
            stage_label="bpb-score",
            error_cls=AxBenchBpbError,
        )
        preflight_values = preflight.values
        enforce_steering_scale_identity(
            rank_dir / STEERING_SCALE_MARKER_NAME,
            identity=preflight.identity,
            resume_active=_next_batch_index(rank_dir) > 0,
        )

    # Pre-index methods + concepts + samples for O(1) lookup during the hot loop
    method_by_name: dict[str, Any] = {m.name: m for m in methods}
    concept_by_id: dict[int, Any] = {c.concept_id: c for c in concepts}

    batch_rows: list[BpbScoreRow] = []
    batch_idx = _next_batch_index(rank_dir)
    processed = 0
    skipped = 0

    with BpbCache(cache_path) as cache:
        for wu in enumerate_work_units(
            methods=methods,
            concepts=concepts,
            samples_by_concept_mode=samples_by_concept_mode,
            coefficient_grids=coefficient_grids,
        ):
            if not is_mine(wu, rank=ddp_rank, world_size=ddp_world_size):
                continue

            method = method_by_name[wu.method_name]
            concept = concept_by_id[wu.concept_id]
            target = method.target_model
            bundle = (
                decoder_only_bundle
                if target == TargetModel.DECODER_ONLY
                else encdec_bundle
            )

            if cache.has(
                method_name=wu.method_name,
                concept_id=wu.concept_id,
                sample_id=wu.sample_id,
                coefficient_str=wu.coefficient_str,
                target=target,
            ):
                skipped += 1
                continue

            # Locate the sample in samples_by_concept_mode
            samples_for_cm = samples_by_concept_mode[(concept.concept_id, method.mode)]
            sample = next(
                (s for s in samples_for_cm if s.sample_id == wu.sample_id),
                None,
            )
            if sample is None:
                # enumerate_work_units only yields units for existing samples,
                # so this should be unreachable — surface loudly if it ever fires.
                raise RuntimeError(
                    f"work unit {wu} has no matching sample in "
                    f"samples_by_concept_mode[({concept.concept_id}, {method.mode})]"
                )

            coef = _coefficient_from_str(wu.coefficient_str)
            # Per-unit deterministic seed keeps randomness rank-invariant: changing
            # ddp_world_size re-distributes units across ranks, but each unit gets
            # the same seed regardless of which rank processes it.
            rng_seed_bytes = hashlib.sha256(
                f"{wu.method_name}|{wu.concept_id}|{wu.sample_id}|{wu.coefficient_str}".encode()
            ).digest()
            intervention = build_intervention(
                method=method,
                concept=concept,
                coefficient=coef,
                source_paths=source_paths,
                diffmean_layer=diffmean_layer,
                target_model_meta=target_model_metas[target],
                inactive_sampler=inactive_sampler,
                tokenizer_nanochat=tokenizer_nanochat,
                rng=random.Random(rng_seed_bytes),
                apply_max_act_scale=True,
                adv_paths=precomputed_bag_paths,
                model=bundle.model if bundle is not None else None,
            )
            method_baseline = getattr(method, "baseline", "")
            if method_baseline in (
                BaselineKind.DIFFMEAN,
                BaselineKind.LINEAR_PROBE,
            ):
                artifact_key = (
                    BaselineKind(method_baseline).value,
                    TargetModel.coerce(method.target_model).value,
                    method.mode.value,
                    concept.concept_id,
                )
                signed_coefficient = float(cast(int | float, coef))
                if method.mode == SteeringMode.NEGATIVE:
                    signed_coefficient = -abs(signed_coefficient)
                expected_coefficient = (
                    signed_coefficient * preflight_values[artifact_key]
                )
                if intervention.direction_coefficient != expected_coefficient:
                    raise AxBenchBpbError(
                        "bpb-score direction-scale mid-run regeneration hazard: "
                        "the built intervention reopened a direction artifact "
                        "whose effective coefficient does not match the stamped "
                        f"preflight value for artifact key {artifact_key!r}; "
                        f"expected {expected_coefficient!r}, got "
                        f"{intervention.direction_coefficient!r}. Stage 3 may "
                        "have regenerated the unclustered artifact while scoring; "
                        "use a fresh run tag."
                    )
            row = score_sample(
                inputs=sample,
                target_bundle=bundle,
                intervention=intervention,
                device=device,
                coefficient_str=wu.coefficient_str,
                method_name=wu.method_name,
                lora=lora,
            )
            cache.put(
                method_name=row.method_name,
                concept_id=row.concept_id,
                sample_id=row.sample_id,
                coefficient_str=row.coefficient_str,
                target=row.target,
                bpb=row.bpb_intervened,
                n_output_tokens=row.n_output_tokens,
                n_output_bytes=row.n_output_bytes,
                truncated=row.truncated,
            )
            batch_rows.append(row)
            processed += 1

            if processed % cfg.log_progress_every == 0:
                logger.info(
                    f"bpb_score rank={ddp_rank}: processed={processed}, "
                    f"skipped={skipped}, buffered={len(batch_rows)}"
                )

            if len(batch_rows) >= batch_flush_size:
                batch_path = rank_dir / f"batch_{batch_idx:06d}.parquet"
                write_bpb_score_rows(batch_rows, batch_path)
                batch_rows.clear()
                batch_idx += 1

        if batch_rows:
            batch_path = rank_dir / f"batch_{batch_idx:06d}.parquet"
            write_bpb_score_rows(batch_rows, batch_path)

    logger.info(f"bpb_score rank={ddp_rank} complete: processed={processed}, skipped={skipped}")
    return rank_dir


__all__ = [
    "enumerate_work_units",
    "is_mine",
    "run_bpb_score_stage",
]
