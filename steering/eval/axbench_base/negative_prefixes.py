"""Stage 0c: build the negative-mode prefix pool (spec §3.3).

For each concept, find validation-shard prefixes where the target
base SAE LM **reliably** continues in concept-rich fashion.
Reliability is defined as: across ``cfg.n_verification_seeds`` (default
5) verification continuations at different RNG seeds, at least
``cfg.min_consistency_passes`` (default 3) have a per-feature
percentile > ``cfg.verification_percentile_threshold * 100`` (default
0.7 -> 70th percentile).

Candidate prefixes come from segments in the validation feature shards
where the concept feature is active. For each candidate, take the
parquet text up to the segment's character end as the prefix, tokenize
with the extraction/Gemma tokenizer, and clamp the length to
``[cfg.min_prefix_chars, cfg.max_prefix_chars]``.

Stage 0c consumes the full ``candidate_concepts.json`` pool emitted by
Stage 0a and writes the final ``concepts.json`` only after it has found
``cfg.max_concepts`` candidates with enough verified prefixes. It also
writes ``negative_mode_prefixes.parquet`` with one row per verified
prefix. Each row carries the 3-5 passing verification continuations
along with their seeds and percentiles so Stage 1c can reuse them
directly as negative-mode training rows.

Edge cases:
- Candidate has 0 verified prefixes -> reject and continue.
- Candidate has fewer verified than
  ``cfg.n_verified_prefixes_per_concept_target`` -> reject and continue.
- Prefix shorter than ``cfg.min_prefix_chars`` -> skipped during
  candidate build.
- Prefix longer than ``cfg.max_prefix_chars`` -> truncated at char
  ``max_prefix_chars``.
- Prefix tokenizes longer than the base SAE LM can accept before
  the verification continuation -> dropped during candidate scanning.

Internal candidate/verified rows are frozen dataclasses
(:class:`_PrefixCandidate`, :class:`_VerifiedNegativePrefix`) to avoid
the dict-with-magic-keys pattern that had tests reaching into literal
string keys for correctness. Dataclass instances are converted to
dicts only at the parquet write boundary via
:func:`dataclasses.asdict`.

Module structure (PR-4 cycle break):

This module is the public Stage 0c entrypoint. The bulk of the private
helper logic (dataclasses, schemas, candidate scanning, verification,
checkpoint I/O, generation adapters) lives in
:mod:`steering.eval.axbench_base._stage0c_internals` so that
:mod:`steering.eval.axbench_base.stage0c_runner` can share those
symbols without forming a circular import. Private implementation consumers
import that canonical module directly; this entrypoint does not re-export its
internals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch.distributed as dist
from loguru import logger

from steering.eval.axbench_base import (
    _stage0c_internals as _si,
)
from steering.eval.axbench_base import (
    extraction as _extraction,
)
from steering.eval.axbench_base import (
    paths,
)
from steering.eval.axbench_base import (
    stage0c_runner as _stage0c_runner,
)
from steering.eval.axbench_base._stats_reader import read_n_sae_features_from_stats
from steering.eval.axbench_base.cluster_artifacts import (
    assert_live_statistics_match_committed_marker,
    clustered_source_enabled,
    validate_cluster_candidate_identity_with_collective_agreement,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.percentile_units import (
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    QcOutcome,
    RejectedConceptReason,
    TargetModelMetadata,
)
from steering.eval.axbench_base.validate import (
    load_and_validate_candidate_concepts,
    load_and_validate_concepts,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def _broadcast_resume_decision(
    resume_path: Path | None,
    *,
    rank: int,
    world_size: int,
    resume_error: str | None = None,
) -> Path | None:
    """Rank 0 makes the resume short-circuit decision; all ranks adopt it.

    On NFS / cache-lagged shared filesystems, or with heterogeneous
    slurm env, ranks can disagree about whether the three Stage 0c
    artifacts are present + complete. Even a millisecond of disagreement
    is enough for some ranks to short-circuit (return early) while
    others enter ``run_single_rank`` and block on
    ``dist.all_gather_object`` forever. The hang isn't bounded by gloo
    because the surviving ranks see a consistent rendezvous count among
    themselves.

    The fix: rank 0 makes the decision (it has the same on-disk view
    eventually anyway), broadcasts it to all ranks, and every rank
    acts on rank 0's verdict. The broadcast is via ``broadcast_object_list``
    with a small payload containing either a resume path, a fail-loud
    validation error, or neither (continue with a fresh run).

    Single-rank or pre-init paths short-circuit: only the multi-rank
    AND initialized-process-group case actually broadcasts.
    """
    if world_size <= 1:
        if resume_error is not None:
            raise AxBenchBaseError(resume_error)
        return resume_path
    if not (dist.is_available() and dist.is_initialized()):
        if resume_error is not None:
            raise AxBenchBaseError(resume_error)
        return resume_path
    decision: list[dict[str, str | None] | None] = [
        {
            "resume_path": str(resume_path) if resume_path is not None else None,
            "resume_error": resume_error,
        }
        if rank == 0
        else None
    ]
    dist.broadcast_object_list(decision, src=0)
    rank0_decision = decision[0]
    if rank0_decision is None:
        raise AxBenchBaseError(
            "Stage 0c resume broadcast returned no rank-0 decision"
        )
    resume_error_value = rank0_decision["resume_error"]
    if resume_error_value is not None:
        raise AxBenchBaseError(resume_error_value)
    resume_path_value = rank0_decision["resume_path"]
    if resume_path_value is None:
        return None
    return Path(resume_path_value)


def _assert_no_partial_clustered_final_artifacts(
    cfg: AxBenchBaseConfig,
) -> None:
    """Reject a torn clustered Stage 0c commit before any expensive work."""

    if not clustered_source_enabled(cfg):
        return
    assert cfg.run_tag is not None
    canonical_paths = (
        paths.AxBenchBasePaths.from_config(cfg).concepts_json(),
        paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json(),
        paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes(),
    )
    qc_report_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    any_canonical = any(path.exists() for path in canonical_paths) or (qc_report_path.exists())
    complete = all(path.is_file() for path in canonical_paths) and (
        qc_report_path.is_file() if cfg.enable_concept_examples_qc else not qc_report_path.exists()
    )
    if any_canonical and not complete:
        present = [str(path) for path in (*canonical_paths, qc_report_path) if path.exists()]
        raise AxBenchBaseError(
            "Clustered Stage 0c found a partial canonical artifact set "
            f"under run_tag={cfg.run_tag!r}: {present}. Individual "
            "cluster-owned files are immutable; use a fresh run tag."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def prepare_negative_prefixes(
    cfg: AxBenchBaseConfig,
    concepts_path: Path,
    base_model: Any,
    base_sae: Any,
    target_model_meta: TargetModelMetadata,
    response_cache: SynthResponseCache | None = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Path:
    """Stage 0c entrypoint. Returns the path to the written parquet.

    Parameters:
        cfg: pipeline config.
        concepts_path: path to candidate_concepts.json (written by Stage 0a).
        base_model: loaded base Gemma-2-2b model (for explicit-null local
            generation and for SAE activation measurement in both generation
            modes).
        base_sae: loaded base GemmaScope SAE.
        target_model_meta: runtime metadata from
            :func:`detect_target_model_metadata`. Carries
            ``sae_layer`` so the QC helpers run at the correct
            residual-stream layer.
        response_cache: optional already-open cache for tests or callers
            that manage cache lifetime externally. When omitted, Stage 0c
            opens ``paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()`` if examples
            QC or API verification generation is enabled.
        rank: this process's rank (0..world_size-1). Default 0.
        world_size: total number of ranks in the distributed group. Default 1
            (single-GPU). Multi-GPU runs (world_size > 1) require
            :mod:`torch.distributed` to be initialized via
            :func:`nanochat.common.compute_init` BEFORE this entrypoint
            is called; the runner uses ``dist.all_gather_object`` for
            block-barrier coordination.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling prepare_negative_prefixes"
        )
    if clustered_source_enabled(cfg) != (cluster_contract is not None):
        raise AxBenchBaseError(
            "Clustered Stage 0c requires config and validated feature-contract state "
            "to be present together"
        )
    # Computed here (moved ahead of the cluster-identity check below, which
    # the oracle-inherited fix batch now also needs) rather than after the
    # sae_layer validation that used to precede it -- ``_infer_n_sae_
    # features`` is a cheap local read off the already-loaded ``base_sae``,
    # with no ordering dependency on that check.
    n_sae_features = _extraction._infer_n_sae_features(base_sae)
    if cluster_contract is not None:
        # Codex cross-review fix batch (FU-D-stage0c, Bug 1 HIGH + Bug 2
        # MEDIUM): mirrors stage0c_runner.dispatch_prepare_negative_
        # prefixes's own fix -- see ``cluster_artifacts.validate_cluster_
        # candidate_identity_with_collective_agreement``'s docstring for
        # the full rationale.
        validate_cluster_candidate_identity_with_collective_agreement(
            cfg,
            cluster_contract,
            candidate_path=concepts_path,
            n_sae_features=n_sae_features,
            world_size=world_size,
        )
    runtime_sae_layer = target_model_meta.sae_layer
    if (
        not isinstance(runtime_sae_layer, int)
        or isinstance(runtime_sae_layer, bool)
        or runtime_sae_layer < 0
    ):
        raise AxBenchBaseError(
            "Stage 0c target_model_meta.sae_layer must be a non-negative "
            f"int, got {runtime_sae_layer!r}"
        )
    # Resume decision is rank-0-makes, broadcast-to-all so ranks cannot
    # disagree about whether to short-circuit. See
    # ``_broadcast_resume_decision`` for the deadlock this prevents.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        local_resume_error: str | None = None
        if rank == 0:
            try:
                local_resume = _resume_if_artifacts_complete(
                    cfg,
                    n_sae_features=n_sae_features,
                    candidate_concepts_path=concepts_path,
                    expected_sae_layer=runtime_sae_layer,
                )
            except Exception as exc:
                local_resume = None
                local_resume_error = str(exc)
        else:
            local_resume = None
        resume = _broadcast_resume_decision(
            local_resume,
            rank=rank,
            world_size=world_size,
            resume_error=local_resume_error,
        )
    else:
        resume = _resume_if_artifacts_complete(
            cfg,
            n_sae_features=n_sae_features,
            candidate_concepts_path=concepts_path,
            expected_sae_layer=runtime_sae_layer,
        )
    if resume is not None:
        return resume
    cluster_kwargs: dict[str, Any] = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    if (
        cfg.enable_concept_examples_qc or cfg.verification_api_model is not None
    ) and response_cache is None:
        with SynthResponseCache(
            paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()
        ) as cache:
            return _stage0c_runner.dispatch_prepare_negative_prefixes(
                cfg,
                concepts_path,
                base_model,
                base_sae,
                target_model_meta,
                n_sae_features=n_sae_features,
                response_cache=cache,
                rank=rank,
                world_size=world_size,
                **cluster_kwargs,
            )
    return _stage0c_runner.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model,
        base_sae,
        target_model_meta,
        n_sae_features=n_sae_features,
        response_cache=response_cache,
        rank=rank,
        world_size=world_size,
        **cluster_kwargs,
    )


def resume_negative_prefixes_if_complete(cfg: AxBenchBaseConfig) -> Path | None:
    """Public lightweight Stage 0c resume check for CLI dispatch.

    Unlike :func:`prepare_negative_prefixes`, this helper does not need
    loaded models. It first checks whether the three canonical Stage 0c
    artifacts are
    present; only then does it read the feature-statistics header, validate the
    current ``candidate_concepts.json``, and compare its ordered-pool hash with
    the completed parquet metadata. This lets the CLI skip loading the base
    model, SAE, and decoder-only target when Stage 0c is already complete,
    without reusing output from an earlier Stage 0a pool. ``None`` means only
    that the canonical set is absent or incomplete. Once all three artifacts
    exist, every validation failure raises before model loading because an
    automatic Stage 0c rerun could overwrite diagnostic evidence or consume a
    stale Stage 0a candidate pool.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before checking Stage 0c resume artifacts")
    _assert_no_partial_clustered_final_artifacts(cfg)
    if not (
        paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
        and paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()
        and paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes().exists()
    ):
        return None
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    _validate_completed_artifact_identity_and_schema(
        parquet_path,
        cfg=cfg,
    )
    n_sae_features = read_n_sae_features_from_stats(
        Path(cfg.feature_statistics_path),
    )
    return _resume_if_artifacts_complete(
        cfg,
        n_sae_features=n_sae_features,
        candidate_concepts_path=paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json(),
    )


def _resume_if_artifacts_complete(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
    candidate_concepts_path: Path,
    expected_sae_layer: int | None = None,
) -> Path | None:
    """Return the existing ``negative_mode_prefixes.parquet`` path when
    Stage 0c already has a valid artifact set for this run_tag. Return ``None``
    only when the three canonical outputs are absent or incomplete, so the
    caller may proceed with Stage 0c. Once all three exist, any validation
    failure raises :class:`AxBenchBaseError`; completed state is never silently
    overwritten or rerun.

    "Complete" means the three canonical Stage 0c output artifacts exist and
    still
    satisfy the current config's structural invariants:

    - ``concepts.json`` validates as a final concept list and contains
      exactly ``cfg.max_concepts`` rows.
    - ``rejected_concepts.json`` is a list of Stage 0c rejection records.
    - ``negative_mode_prefixes.parquet`` has the expected schema, exactly
      ``cfg.n_verified_prefixes_per_concept_target`` rows for every final
      concept, and every row still satisfies the current verification
      consistency/threshold settings.
    - Its persisted candidate-pool hash equals the content hash of the current
      ordered ``candidate_concepts.json`` at ``candidate_concepts_path``.
    - When examples QC is enabled, ``concept_examples_qc_report.json`` is
      structurally valid and its semantic hash matches parquet metadata. When
      QC is disabled, the parquet pins a null-report sentinel and the report
      path must be absent.

    Crash resume is the primary motivation: a Stage 0c that verified 5
    concepts in 5 hours has written all three artifacts already; a
    subsequent run (say, after a Stage 1b crash took the pipeline down)
    should short-circuit Stage 0c rather than redo the 5 hours of
    verification. Operators who want to force a same-identity re-run delete
    the ``concepts.json`` file (or any of the three artifacts) — the DAG
    semantics match the rest of the pipeline. Identity changes require a new
    ``run_tag`` or regeneration from Stage 0a.

    The validation catches config changes that would make a previous
    artifact stale: the Stage 0a candidate pool, ``max_concepts``, per-concept
    prefix target, ``min_consistency_passes``, and verification percentile
    threshold.
    """
    assert cfg.run_tag is not None
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    qc_report_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    _assert_no_partial_clustered_final_artifacts(cfg)
    if not (concepts_path.exists() and rejected_path.exists() and parquet_path.exists()):
        return None
    try:
        # Codex cross-review fix batch (FU-D-stage0c, Bug 1 HIGH): this
        # entrypoint (and its public caller,
        # ``resume_negative_prefixes_if_complete``) deliberately never
        # resolves a full ``ClusterFeatureContract`` -- it must decide
        # whether to reuse a completed run without loading a checkpoint or
        # SAE. Re-validate the LIVE statistics against the already-
        # committed ``feature_contract.json`` marker (a cheap JSON read,
        # not a contract rebuild) before trusting ANY completed artifact
        # below. No-op when clustering is disabled.
        assert_live_statistics_match_committed_marker(
            cfg,
            n_sae_features=n_sae_features,
            surface="Stage 0c resume",
        )
        _validate_completed_artifact_identity_and_schema(
            parquet_path,
            cfg=cfg,
        )
        candidate_concepts = load_and_validate_candidate_concepts(
            candidate_concepts_path,
            n_sae_features=n_sae_features,
        )
        if not candidate_concepts:
            raise _si._CompletedStage0cCandidatePoolIdentityError(
                "Stage 0c resume: current candidate pool at "
                f"{candidate_concepts_path} is empty; refusing completed "
                "artifact reuse because its identity cannot match a real "
                "Stage 0c run"
            )
        if expected_sae_layer is not None:
            candidate_layers = {
                candidate.layer for candidate in candidate_concepts
            }
            if candidate_layers != {expected_sae_layer}:
                raise AxBenchBaseError(
                    f"Stage 0c resume: current candidate pool at "
                    f"{candidate_concepts_path} has SAE layer(s) "
                    f"{sorted(candidate_layers)!r}, but runtime metadata "
                    f"requires sae_layer={expected_sae_layer}."
                )
        pool_identity = _si.candidate_pool_identity(candidate_concepts)
        _si.validate_negative_prefixes_candidate_pool_identity_metadata(
            parquet_path,
            consumer="Stage 0c resume",
            expected_candidate_pool_identity=pool_identity,
        )
        if cfg.enable_concept_examples_qc:
            if not qc_report_path.is_file():
                raise AxBenchBaseError(
                    "Stage 0c resume: the canonical artifacts exist, but "
                    f"examples QC is enabled and {qc_report_path} is missing"
                )
            qc_report_payload = _validate_concept_examples_qc_report_json(
                qc_report_path,
                expected_candidates=candidate_concepts,
                expected_judge_min_score=(
                    cfg.activation_example_judge_min_score
                ),
            )
        else:
            if qc_report_path.exists():
                raise AxBenchBaseError(
                    "Stage 0c resume: examples QC is disabled, but stale "
                    f"diagnostic state exists at {qc_report_path}. The "
                    "completed disabled-QC artifact set requires that report "
                    "to be absent."
                )
            qc_report_payload = None
        existing_concepts = load_and_validate_concepts(
            concepts_path, n_sae_features=n_sae_features,
        )
        rejected_concepts_payload = _validate_rejected_concepts_json(rejected_path)
        _si.validate_negative_prefixes_completed_artifact_identities(
            parquet_path,
            consumer="Stage 0c resume",
            expected_concepts=existing_concepts,
            expected_rejected_concepts=rejected_concepts_payload,
            expected_concept_examples_qc_report=qc_report_payload,
            expected_stage0c_fingerprint=_si._fingerprint_with_pool(
                cfg,
                candidate_pool_identity=pool_identity,
            ),
        )
        if len(existing_concepts) != cfg.max_concepts:
            raise AxBenchBaseError(
                f"Stage 0c resume: {concepts_path} has "
                f"{len(existing_concepts)} concepts but cfg.max_concepts="
                f"{cfg.max_concepts}. Refusing an automatic Stage 0c rerun "
                "because it could overwrite diagnostic evidence or consume "
                "a stale Stage 0a candidate pool. Use a new `run_tag`, or "
                "delete the Stage 0 artifacts and rerun Stage 0a then Stage "
                "0c."
            )
        _validate_completed_sidecars_against_candidate_pool(
            candidate_concepts=candidate_concepts,
            concepts=existing_concepts,
            rejected_concepts=rejected_concepts_payload,
            concept_examples_qc_report=qc_report_payload,
            cfg=cfg,
        )
        _validate_negative_prefixes_parquet(
            parquet_path, concepts=existing_concepts, cfg=cfg,
        )
        concept_sources = {concept.sae_source for concept in existing_concepts}
        if concept_sources != {cfg.neuronpedia_source}:
            raise _si._CompletedStage0cExtractorIdentityError(
                f"Stage 0c resume: {concepts_path} has SAE source(s) "
                f"{sorted(concept_sources)!r}, but the current config and "
                "negative_mode_prefixes metadata use "
                f"neuronpedia_source={cfg.neuronpedia_source!r}. Refusing "
                "mixed-identity completed artifacts. Use a new `run_tag`, or "
                "delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c."
            )
    except (
        _si._CompletedStage0cExtractorIdentityError,
        _si._CompletedStage0cCandidatePoolIdentityError,
        _si._CompletedStage0cArtifactSetIdentityError,
    ) as exc:
        logger.error(
            "Stage 0c resume: completed artifact identity is "
            f"unsafe to reuse ({exc})"
        )
        raise
    except (
        AxBenchBaseError,
        OSError,
        json.JSONDecodeError,
        pa.ArrowException,
        TypeError,
        ValueError,
    ) as exc:
        message = (
            "Stage 0c resume: the completed artifact set is invalid for the "
            f"current config ({exc}). Refusing an automatic Stage 0c rerun "
            "because it could overwrite diagnostic evidence or consume a "
            "stale Stage 0a candidate pool. Use a new `run_tag`, or delete "
            "the Stage 0 artifacts and rerun Stage 0a then Stage 0c."
        )
        logger.error(message)
        raise AxBenchBaseError(message) from exc
    logger.info(
        f"Stage 0c resume: all artifacts present for run_tag={cfg.run_tag!r} "
        f"(concepts.json has {len(existing_concepts)} entries; "
        f"skipping verification — delete any of concepts.json, "
        f"rejected_concepts.json, or negative_mode_prefixes.parquet "
        f"to force a re-run)."
    )
    return parquet_path


def _validate_rejected_concepts_json(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AxBenchBaseError(
            f"rejected_concepts.json at {path} must be a JSON array, "
            f"got {type(raw).__name__}"
        )
    required_keys = {
        "candidate_index",
        "feature_index",
        "description",
        "reason",
        "n_verified_found",
        "llm_judge_score",
    }
    valid_reasons = {reason.value for reason in RejectedConceptReason}
    validated: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AxBenchBaseError(
                f"rejected_concepts[{i}] at {path} must be a JSON object, "
                f"got {type(item).__name__}"
            )
        record = cast(dict[str, Any], item)
        if set(record) != required_keys:
            raise AxBenchBaseError(
                f"rejected_concepts[{i}] at {path} has keys "
                f"{sorted(record)}, expected {sorted(required_keys)}"
            )
        candidate_index = record["candidate_index"]
        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or candidate_index < 0
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].candidate_index must be a "
                "non-negative int"
            )
        feature_index = record["feature_index"]
        if (
            not isinstance(feature_index, int)
            or isinstance(feature_index, bool)
            or feature_index < 0
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].feature_index must be a "
                "non-negative int"
            )
        description = record["description"]
        if not isinstance(description, str) or not description.strip():
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].description is empty or not a string"
            )
        reason = record["reason"]
        if reason not in valid_reasons:
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].reason has unknown value {reason!r}"
            )
        n_verified_found = record["n_verified_found"]
        if n_verified_found is not None and (
            not isinstance(n_verified_found, int)
            or isinstance(n_verified_found, bool)
            or n_verified_found < 0
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].n_verified_found must be null or "
                "a non-negative int"
            )
        llm_judge_score = record["llm_judge_score"]
        if llm_judge_score is not None and (
            not isinstance(llm_judge_score, int)
            or isinstance(llm_judge_score, bool)
            or not 0 <= llm_judge_score <= 2
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{i}].llm_judge_score must be null or an "
                "int in [0, 2]"
            )
        validated.append(record)
    return validated


def _validate_concept_examples_qc_report_json(
    path: Path,
    *,
    expected_candidates: list[CandidateConcept],
    expected_judge_min_score: int,
) -> dict[str, Any]:
    """Validate and return the persisted Stage 0c QC diagnostic.

    The report is human-facing, but it is part of a completed QC-enabled run.
    Validate its redundant summary fields against ``per_feature`` before its
    semantic identity is compared with the canonical parquet commit marker.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _validate_concept_examples_qc_report_payload(
        raw,
        source=f"concept_examples_qc_report.json at {path}",
        expected_candidates=expected_candidates,
        expected_judge_min_score=expected_judge_min_score,
    )


def _validate_concept_examples_qc_report_payload(
    raw: object,
    *,
    source: str,
    expected_candidates: list[CandidateConcept],
    expected_judge_min_score: int,
) -> dict[str, Any]:
    """Validate persisted or in-memory QC-report semantics identically."""
    if not isinstance(raw, dict):
        raise AxBenchBaseError(
            f"{source} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    report = cast(dict[str, Any], raw)
    expected_fields = {
        "n_total",
        "n_accepted",
        "n_rejected",
        "n_fetch_errors",
        "n_judge_errors",
        "n_empty_examples",
        "score_histogram",
        "per_feature",
        "judge_min_score",
        "n_processed",
    }
    if set(report) != expected_fields:
        raise AxBenchBaseError(
            f"{source} has keys "
            f"{sorted(report)}, expected {sorted(expected_fields)}"
        )

    def _nonnegative_int(field: str) -> int:
        value = report[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AxBenchBaseError(
                f"concept_examples_qc_report.{field} must be a non-negative "
                f"int, got {value!r}"
            )
        return value

    n_total = _nonnegative_int("n_total")
    expected_n_total = len(expected_candidates)
    if n_total != expected_n_total:
        raise AxBenchBaseError(
            f"concept_examples_qc_report.n_total={n_total} does not match the "
            f"current candidate pool size {expected_n_total}"
        )
    judge_min_score = _nonnegative_int("judge_min_score")
    if judge_min_score > 2 or judge_min_score != expected_judge_min_score:
        raise AxBenchBaseError(
            "concept_examples_qc_report.judge_min_score must equal the current "
            f"activation_example_judge_min_score={expected_judge_min_score}, "
            f"got {judge_min_score}"
        )
    n_processed = _nonnegative_int("n_processed")
    summary_counts = {
        QcOutcome.ACCEPTED: _nonnegative_int("n_accepted"),
        QcOutcome.REJECTED: _nonnegative_int("n_rejected"),
        QcOutcome.FETCH_ERROR: _nonnegative_int("n_fetch_errors"),
        QcOutcome.JUDGE_ERROR: _nonnegative_int("n_judge_errors"),
        QcOutcome.EMPTY_EXAMPLES: _nonnegative_int("n_empty_examples"),
    }

    raw_rows = report["per_feature"]
    if not isinstance(raw_rows, list):
        raise AxBenchBaseError(
            "concept_examples_qc_report.per_feature must be a JSON array"
        )
    if n_processed != len(raw_rows) or n_processed > n_total:
        raise AxBenchBaseError(
            "concept_examples_qc_report.n_processed must equal the number of "
            f"per_feature rows and be <= n_total; got n_processed={n_processed}, "
            f"rows={len(raw_rows)}, n_total={n_total}"
        )

    expected_row_fields = {
        "feature_index",
        "description",
        "outcome",
        "score",
        "n_examples",
        "error_message",
    }
    actual_counts = dict.fromkeys(QcOutcome, 0)
    actual_histogram: dict[int, int] = {}
    seen_feature_ids: set[int] = set()
    candidates_by_feature = {
        candidate.feature_index: candidate for candidate in expected_candidates
    }
    for row_idx, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}] must be "
                "a JSON object"
            )
        row = cast(dict[str, Any], raw_row)
        if set(row) != expected_row_fields:
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}] has keys "
                f"{sorted(row)}, expected {sorted(expected_row_fields)}"
            )
        feature_id = row["feature_index"]
        if (
            not isinstance(feature_id, int)
            or isinstance(feature_id, bool)
            or feature_id < 0
        ):
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}]."
                f"feature_index must be a non-negative int, got {feature_id!r}"
            )
        if feature_id in seen_feature_ids:
            raise AxBenchBaseError(
                "concept_examples_qc_report.per_feature contains duplicate "
                f"feature_index={feature_id}"
            )
        seen_feature_ids.add(feature_id)
        description = row["description"]
        if not isinstance(description, str) or not description.strip():
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}].description "
                "must be a non-empty string"
            )
        expected_candidate = candidates_by_feature.get(feature_id)
        if expected_candidate is None or description != expected_candidate.description:
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}] does not "
                "match any current candidate's feature_index and description"
            )
        try:
            outcome = QcOutcome(row["outcome"])
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}].outcome "
                f"is invalid: {row['outcome']!r}"
            ) from exc
        actual_counts[outcome] += 1
        score = row["score"]
        if score is not None and (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 2
        ):
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}].score "
                f"must be null or an int in [0, 2], got {score!r}"
            )
        n_examples = row["n_examples"]
        if (
            not isinstance(n_examples, int)
            or isinstance(n_examples, bool)
            or n_examples < 0
        ):
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}].n_examples "
                f"must be a non-negative int, got {n_examples!r}"
            )
        error_message = row["error_message"]
        if error_message is not None and not isinstance(error_message, str):
            raise AxBenchBaseError(
                f"concept_examples_qc_report.per_feature[{row_idx}]."
                "error_message must be null or a string"
            )

        if outcome in (QcOutcome.ACCEPTED, QcOutcome.REJECTED):
            if score is None or n_examples == 0 or error_message is not None:
                raise AxBenchBaseError(
                    f"concept_examples_qc_report.per_feature[{row_idx}] "
                    f"outcome={outcome.value!r} requires a score, at least one "
                    "example, and no error_message"
                )
            if (
                outcome is QcOutcome.ACCEPTED
                and score < judge_min_score
            ) or (
                outcome is QcOutcome.REJECTED
                and score >= judge_min_score
            ):
                raise AxBenchBaseError(
                    f"concept_examples_qc_report.per_feature[{row_idx}] score="
                    f"{score} is inconsistent with outcome={outcome.value!r} "
                    f"at judge_min_score={judge_min_score}"
                )
            actual_histogram[score] = actual_histogram.get(score, 0) + 1
        else:
            if score is not None:
                raise AxBenchBaseError(
                    f"concept_examples_qc_report.per_feature[{row_idx}] "
                    f"outcome={outcome.value!r} must not have a score"
                )
            if outcome in (QcOutcome.FETCH_ERROR, QcOutcome.EMPTY_EXAMPLES):
                if n_examples != 0:
                    raise AxBenchBaseError(
                        f"concept_examples_qc_report.per_feature[{row_idx}] "
                        f"outcome={outcome.value!r} requires n_examples=0"
                    )
            elif n_examples == 0:
                raise AxBenchBaseError(
                    f"concept_examples_qc_report.per_feature[{row_idx}] "
                    "judge_error requires at least one fetched example"
                )
            has_error = outcome in (QcOutcome.FETCH_ERROR, QcOutcome.JUDGE_ERROR)
            if has_error != isinstance(error_message, str):
                raise AxBenchBaseError(
                    f"concept_examples_qc_report.per_feature[{row_idx}] "
                    f"outcome={outcome.value!r} has inconsistent error_message"
                )

    if actual_counts != summary_counts:
        raise AxBenchBaseError(
            "concept_examples_qc_report summary counts disagree with "
            f"per_feature outcomes: stored={summary_counts!r}, "
            f"actual={actual_counts!r}"
        )
    if sum(summary_counts.values()) != n_processed:
        raise AxBenchBaseError(
            "concept_examples_qc_report outcome counts do not sum to "
            f"n_processed={n_processed}"
        )

    raw_histogram = report["score_histogram"]
    if not isinstance(raw_histogram, dict):
        raise AxBenchBaseError(
            "concept_examples_qc_report.score_histogram must be a JSON object"
        )
    stored_histogram: dict[int, int] = {}
    for raw_score, raw_count in raw_histogram.items():
        if raw_score not in {"0", "1", "2"}:
            raise AxBenchBaseError(
                "concept_examples_qc_report.score_histogram keys must be "
                f"decimal scores 0, 1, or 2, got {raw_score!r}"
            )
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count <= 0
        ):
            raise AxBenchBaseError(
                "concept_examples_qc_report.score_histogram counts must be "
                f"positive ints, got {raw_count!r} for score {raw_score}"
            )
        stored_histogram[int(raw_score)] = raw_count
    if stored_histogram != actual_histogram:
        raise AxBenchBaseError(
            "concept_examples_qc_report.score_histogram disagrees with "
            f"per_feature scores: stored={stored_histogram!r}, "
            f"actual={actual_histogram!r}"
        )
    return report


def _validate_completed_sidecars_against_candidate_pool(
    *,
    candidate_concepts: list[CandidateConcept],
    concepts: list[Concept],
    rejected_concepts: list[dict[str, Any]],
    concept_examples_qc_report: dict[str, Any] | None,
    cfg: AxBenchBaseConfig,
) -> None:
    """Prove that every canonical sidecar row belongs to the live pool."""
    candidate_by_feature = {
        candidate.feature_index: (candidate_idx, candidate)
        for candidate_idx, candidate in enumerate(candidate_concepts)
    }
    candidate_fields = (
        "feature_index",
        "layer",
        "sae_source",
        "description",
        "corpus_count",
        "description_source",
    )
    accepted_indices: list[int] = []
    for concept in concepts:
        candidate_match = candidate_by_feature.get(concept.feature_index)
        if candidate_match is None:
            raise AxBenchBaseError(
                f"Final concept feature_index={concept.feature_index} is absent "
                "from the current candidate pool"
            )
        candidate_idx, candidate = candidate_match
        mismatched = [
            field
            for field in candidate_fields
            if getattr(concept, field) != getattr(candidate, field)
        ]
        if mismatched:
            raise AxBenchBaseError(
                f"Final concept_id={concept.concept_id} disagrees with current "
                f"candidate_concepts[{candidate_idx}] on fields {mismatched!r}"
            )
        accepted_indices.append(candidate_idx)
    if accepted_indices != sorted(accepted_indices):
        raise AxBenchBaseError(
            "Final concepts are not ordered by their current candidate indices: "
            f"{accepted_indices!r}"
        )

    rejected_indices: list[int] = []
    for row_idx, record in enumerate(rejected_concepts):
        candidate_idx = record["candidate_index"]
        if candidate_idx >= len(candidate_concepts):
            raise AxBenchBaseError(
                f"rejected_concepts[{row_idx}].candidate_index={candidate_idx} "
                f"is outside [0, {len(candidate_concepts)})"
            )
        candidate = candidate_concepts[candidate_idx]
        if (
            record["feature_index"] != candidate.feature_index
            or record["description"] != candidate.description
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{row_idx}] disagrees with current "
                f"candidate_concepts[{candidate_idx}]"
            )
        rejected_indices.append(candidate_idx)
    if rejected_indices != sorted(set(rejected_indices)):
        raise AxBenchBaseError(
            "rejected_concepts candidate indices must be unique and sorted, "
            f"got {rejected_indices!r}"
        )
    overlap = sorted(set(accepted_indices) & set(rejected_indices))
    if overlap:
        raise AxBenchBaseError(
            "Completed Stage 0c sidecars mark candidate indices as both "
            f"accepted and rejected: {overlap!r}"
        )
    if len(concepts) > cfg.max_concepts:
        raise AxBenchBaseError(
            "Stage 0c sidecars contain more accepted concepts than "
            f"cfg.max_concepts={cfg.max_concepts}: {len(concepts)}"
        )
    terminal_indices = set(accepted_indices) | set(rejected_indices)
    if terminal_indices:
        if len(concepts) == cfg.max_concepts:
            # A successful run stops at its final accepted candidate. Any
            # later outcome is stale or a distributed overvisit that must not
            # enter the canonical artifact set.
            terminal_cutoff = accepted_indices[-1]
        else:
            # Failed exhaustion/cap runs intentionally persist diagnostics
            # after the last accept. They still must describe one contiguous
            # prefix of the ordered pool with no holes or duplicates.
            terminal_cutoff = max(terminal_indices)
        expected_terminal_indices = set(range(terminal_cutoff + 1))
        if terminal_indices != expected_terminal_indices:
            raise AxBenchBaseError(
                "Completed Stage 0c sidecars must contain exactly one terminal "
                "outcome for every candidate through the canonical cutoff "
                f"candidate; expected={sorted(expected_terminal_indices)!r}, "
                f"actual={sorted(terminal_indices)!r}"
            )

    qc_by_feature: dict[int, dict[str, Any]] = {}
    if cfg.enable_concept_examples_qc:
        if concept_examples_qc_report is None:
            raise AxBenchBaseError(
                "Completed Stage 0c QC-enabled sidecars require a validated "
                "concept_examples_qc_report"
            )
        qc_by_feature = {
            row["feature_index"]: row
            for row in concept_examples_qc_report["per_feature"]
        }
        qc_indices: list[int] = []
        for row_idx, row in enumerate(
            concept_examples_qc_report["per_feature"]
        ):
            candidate_match = candidate_by_feature.get(row["feature_index"])
            if candidate_match is None:
                raise AxBenchBaseError(
                    "concept_examples_qc_report.per_feature"
                    f"[{row_idx}].feature_index={row['feature_index']} is "
                    "absent from the current candidate pool"
                )
            qc_indices.append(candidate_match[0])
        expected_qc_indices = sorted(terminal_indices)
        if qc_indices != expected_qc_indices:
            raise AxBenchBaseError(
                "Completed Stage 0c QC report must contain exactly one row "
                "for every terminal candidate in canonical candidate order; "
                f"expected={expected_qc_indices!r}, actual={qc_indices!r}"
            )
    elif concept_examples_qc_report is not None:
        raise AxBenchBaseError(
            "Completed Stage 0c QC-disabled sidecars must use a null report "
            "sentinel"
        )

    for concept, candidate_idx in zip(concepts, accepted_indices, strict=True):
        if not cfg.enable_concept_examples_qc:
            continue
        qc_row = qc_by_feature.get(concept.feature_index)
        if qc_row is None or QcOutcome(qc_row["outcome"]) is not QcOutcome.ACCEPTED:
            raise AxBenchBaseError(
                f"Final accepted candidate {candidate_idx} lacks an accepted "
                "QC-report outcome"
            )

    target = cfg.n_verified_prefixes_per_concept_target
    for row_idx, record in enumerate(rejected_concepts):
        reason = RejectedConceptReason(record["reason"])
        n_verified_found = record["n_verified_found"]
        llm_judge_score = record["llm_judge_score"]
        if reason is RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES:
            if (
                not isinstance(n_verified_found, int)
                or isinstance(n_verified_found, bool)
                or not 0 <= n_verified_found < target
            ):
                raise AxBenchBaseError(
                    f"rejected_concepts[{row_idx}] reason={reason.value!r} "
                    f"requires 0 <= n_verified_found < {target}"
                )
            expected_qc_outcome = QcOutcome.ACCEPTED
        else:
            if not cfg.enable_concept_examples_qc:
                raise AxBenchBaseError(
                    f"rejected_concepts[{row_idx}] uses QC-only reason "
                    f"{reason.value!r} while examples QC is disabled"
                )
            if n_verified_found is not None:
                raise AxBenchBaseError(
                    f"rejected_concepts[{row_idx}] QC rejection must have "
                    "n_verified_found=null"
                )
            expected_qc_outcome = {
                RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE: (
                    QcOutcome.REJECTED
                ),
                RejectedConceptReason.LLM_JUDGE_FETCH_ERROR: (
                    QcOutcome.FETCH_ERROR
                ),
                RejectedConceptReason.LLM_JUDGE_ERROR: QcOutcome.JUDGE_ERROR,
                RejectedConceptReason.LLM_JUDGE_EMPTY_EXAMPLES: (
                    QcOutcome.EMPTY_EXAMPLES
                ),
            }[reason]
        if not cfg.enable_concept_examples_qc:
            if llm_judge_score is not None:
                raise AxBenchBaseError(
                    f"rejected_concepts[{row_idx}] stores an LLM score while "
                    "examples QC is disabled"
                )
            continue
        qc_row = qc_by_feature.get(record["feature_index"])
        if qc_row is None:
            raise AxBenchBaseError(
                f"rejected_concepts[{row_idx}] lacks a corresponding QC row"
            )
        actual_qc_outcome = QcOutcome(qc_row["outcome"])
        if (
            actual_qc_outcome is not expected_qc_outcome
            or qc_row["score"] != llm_judge_score
        ):
            raise AxBenchBaseError(
                f"rejected_concepts[{row_idx}] disagrees with its QC-report "
                "outcome or score"
            )


def _validate_negative_prefixes_parquet(
    path: Path, *, concepts: list[Concept], cfg: AxBenchBaseConfig,
) -> None:
    table = pq.read_table(path)
    _validate_negative_prefixes_table(
        table,
        path=path,
        concepts=concepts,
        cfg=cfg,
    )


def _validate_negative_prefixes_table(
    table: pa.Table,
    *,
    path: Path,
    concepts: list[Concept],
    cfg: AxBenchBaseConfig,
) -> None:
    """Validate an in-memory or persisted canonical Stage 0c table."""
    if table.schema.remove_metadata() != _si._NEG_PREFIXES_SCHEMA:
        raise AxBenchBaseError(
            f"negative_mode_prefixes parquet at {path} has schema "
            f"{table.schema}, expected {_si._NEG_PREFIXES_SCHEMA}"
        )
    _si.validate_negative_prefixes_extractor_identity_metadata(
        path,
        cfg=cfg,
        consumer="Stage 0c resume",
        schema=table.schema,
    )
    _si.validate_negative_prefixes_tokenizer_metadata(
        path,
        expected_tokenizer_id=cfg.extraction_tokenizer_id,
        consumer="Stage 0c resume",
        schema=table.schema,
    )

    expected_concept_ids = {concept.concept_id for concept in concepts}
    rows_per_concept = {concept.concept_id: 0 for concept in concepts}
    raw_threshold = percentile_fraction_to_points(
        cfg.verification_percentile_threshold,
        field_name="verification_percentile_threshold",
    )
    for row_idx, row in enumerate(table.to_pylist()):
        concept_id = _expect_resume_int(row, "concept_id", row_idx)
        if concept_id not in expected_concept_ids:
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].concept_id={concept_id} "
                "does not correspond to a final concept"
            )
        rows_per_concept[concept_id] += 1

        prefix_text = row["prefix_text"]
        if not isinstance(prefix_text, str) or not prefix_text:
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].prefix_text is empty "
                "or not a string"
            )
        if not (cfg.min_prefix_chars <= len(prefix_text) <= cfg.max_prefix_chars):
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].prefix_text length "
                f"{len(prefix_text)} is outside current bounds "
                f"[{cfg.min_prefix_chars}, {cfg.max_prefix_chars}]"
            )

        for field in (
            "prefix_source_shard",
            "prefix_source_doc",
            "prefix_activating_seg_idx",
        ):
            value = _expect_resume_int(row, field, row_idx)
            if value < 0:
                raise AxBenchBaseError(
                    f"negative_mode_prefixes[{row_idx}].{field} must be "
                    f"non-negative, got {value}"
                )

        continuations = row["verification_continuations"]
        seeds = row["verification_seeds"]
        percentiles = row["verification_percentiles"]
        if not (
            isinstance(continuations, list)
            and isinstance(seeds, list)
            and isinstance(percentiles, list)
        ):
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}] verification fields "
                "must all be lists"
            )
        n_passes = len(percentiles)
        if len(continuations) != n_passes or len(seeds) != n_passes:
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}] verification list "
                "lengths differ"
            )
        if n_passes < cfg.min_consistency_passes:
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}] has {n_passes} passing "
                f"continuations, below cfg.min_consistency_passes="
                f"{cfg.min_consistency_passes}"
            )
        if n_passes > cfg.n_verification_seeds:
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}] has {n_passes} passing "
                f"continuations, above cfg.n_verification_seeds="
                f"{cfg.n_verification_seeds}"
            )
        if any(not isinstance(cont, str) for cont in continuations):
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].verification_continuations "
                "must contain only strings"
            )
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].verification_seeds must "
                "contain only ints"
            )
        if len(set(seeds)) != len(seeds):
            raise AxBenchBaseError(
                f"negative_mode_prefixes[{row_idx}].verification_seeds "
                "contains duplicates"
            )
        for pct in percentiles:
            pct_value = float(pct)
            if not math.isfinite(pct_value) or pct_value <= raw_threshold:
                raise AxBenchBaseError(
                    f"negative_mode_prefixes[{row_idx}] has percentile "
                    f"{pct_value}, which does not exceed current threshold "
                    f"{raw_threshold}"
                )

    expected_rows = cfg.n_verified_prefixes_per_concept_target
    for concept_id, row_count in rows_per_concept.items():
        if row_count != expected_rows:
            raise AxBenchBaseError(
                f"negative_mode_prefixes parquet at {path} has {row_count} "
                f"rows for concept_id={concept_id}, expected {expected_rows}"
            )


def _validate_completed_artifact_identity_and_schema(
    path: Path,
    *,
    cfg: AxBenchBaseConfig,
) -> None:
    """Fail closed on completed schema/provenance before other sidecars."""
    try:
        schema = pq.read_schema(path)
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Stage 0c resume: completed negative_mode_prefixes parquet at "
            f"{path} is unreadable ({exc}). Refusing an automatic rerun; use "
            "a new `run_tag`, or delete the Stage 0 artifacts and rerun "
            "Stage 0a then Stage 0c."
        ) from exc
    if schema.remove_metadata() != _si._NEG_PREFIXES_SCHEMA:
        raise AxBenchBaseError(
            f"Stage 0c resume: completed negative_mode_prefixes parquet at "
            f"{path} has schema {schema.remove_metadata()}, expected "
            f"{_si._NEG_PREFIXES_SCHEMA}. Refusing an automatic rerun; use a new "
            "`run_tag`, or delete the Stage 0 artifacts and rerun Stage 0a "
            "then Stage 0c."
        )
    _si.validate_negative_prefixes_tokenizer_metadata(
        path,
        expected_tokenizer_id=cfg.extraction_tokenizer_id,
        consumer="Stage 0c resume",
        schema=schema,
    )
    _si.validate_negative_prefixes_extractor_identity_metadata(
        path,
        cfg=cfg,
        consumer="Stage 0c resume",
        schema=schema,
    )
    _si.validate_negative_prefixes_candidate_pool_identity_metadata(
        path,
        consumer="Stage 0c resume",
        schema=schema,
    )
    _si.validate_negative_prefixes_completed_artifact_identities(
        path,
        consumer="Stage 0c resume",
        schema=schema,
    )


def _expect_resume_int(row: dict[str, Any], field: str, row_idx: int) -> int:
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise AxBenchBaseError(
            f"negative_mode_prefixes[{row_idx}].{field} must be an int, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value
