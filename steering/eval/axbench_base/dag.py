"""Stage dependency DAG + artifact existence checks for axbench_base.

The pipeline is built as a 14-stage graph (``Stage.PREPARE_CONCEPTS``
through ``Stage.AGGREGATE`` — 0a/0b/0c/1b/1c/2/3a/3b/3c/3d/3e/4/5/6)
with a simple declarative dependency map.
Before running any stage, we check that every required artifact already
exists on disk; if one is missing we raise ``AxBenchBaseError`` with the
full list of missing paths so the operator knows exactly which prior
stage to re-run.

This enables crash-resume: re-running ``stage=Stage.ALL`` after a
partial failure will only re-run the stages whose outputs do not yet
exist, because each stage's runtime entrypoint calls
:func:`check_stage_dependencies` against its own fresh config before
doing work.

Two rules are encoded here beyond the raw dep map:

1. ``stage == Stage.ALL`` is a no-op (the pipeline checks individual
   stages inside its dispatcher loop).
2. When ``cfg.run_negative_mode`` is ``False``:
   - Stage ``Stage.SYNTH_NEGATIVE_MODE`` is skipped entirely — the dep
     check is a no-op.
   - Stage ``Stage.GENERATE`` drops its
     ``concept_features_neg.safetensors`` requirement (it was never
     produced).

All artifact key → path resolution goes through ``_ARTIFACT_PATHS`` so
the file/directory layout stays owned by :mod:`paths`.

Stage identifiers are the :class:`Stage` :class:`StrEnum` from
:mod:`steering.eval.axbench_base.types`. Because :class:`Stage`
subclasses :class:`str`, every call site that passes a bare string
(``"0a"``, ``"all"``) still works — the enum coerces the input at the
top of :func:`check_stage_dependencies` / :func:`list_missing_artifacts`
and :data:`STAGE_DEPENDENCIES` hash-equals the string form thanks to
the :class:`StrEnum` contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from steering.eval.axbench_base import paths
from steering.eval.axbench_base._stats_reader import read_n_sae_features_from_stats
from steering.eval.axbench_base.aggregation import assert_complete_judge_coverage
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.method_registry import BaselineKind, get_method_by_name
from steering.eval.axbench_base.parquet_writer import load_existing
from steering.eval.axbench_base.synth.concept_yield import (
    concept_yield_report_is_committed,
    load_concept_yield_surviving_ids,
    load_filtered_concepts,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    Stage,
    SteeringMode,
)
from steering.eval.axbench_base.validate import load_and_validate_concepts
from steering.eval.cluster_config import CLUSTERED_AXBENCH_BASE_STAGE4_METHODS

#: Declarative stage → required-artifact-keys map. Keys are
#: :class:`Stage` members; values are lists of keys into
#: :data:`_ARTIFACT_PATHS`. Ordering within each list is irrelevant;
#: the dep check evaluates them as a set.
#:
#: :attr:`Stage.ALL` is intentionally absent: it is a pseudo-stage and
#: gets expanded to the full ordered stage list by
#: ``_resolve_stage_list`` before any dep check runs.
STAGE_DEPENDENCIES: dict[Stage, list[str]] = {
    Stage.PREPARE_CONCEPTS: [],
    Stage.PREPARE_SEED_POOLS: ["candidate_concepts.json"],
    Stage.PREPARE_NEGATIVE_PREFIXES: [
        "candidate_concepts.json",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
    ],
    Stage.SYNTH_POSITIVE_MODE: [
        "concepts.json",
        "seeds_train.safetensors",
        "seed_pool_manifest.json",
    ],
    Stage.SYNTH_NEGATIVE_MODE: [
        "concepts.json",
        "concepts_stage1b.json",
        "negative_mode_prefixes.parquet",
        "stage1b_concept_yield.json",
    ],
    Stage.CONCEPT_FEATURE_MAPPING: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "synth_training_positive.parquet",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.TRAIN_DIFFMEAN: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.TRAIN_LINEAR_PROBE: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.TRAIN_PROMPTING: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.TRAIN_LORA: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.ENUMERATE_STAGE4_CONTEXTS: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    ],
    Stage.GENERATE: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
        "stage4_contexts_manifest.json",
    ],
    Stage.JUDGE: [
        "concepts.json",
        "concepts_stage1b.json",
        "concepts_stage1c.json",
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
        "generations",
    ],
    Stage.AGGREGATE: ["generations", "judge_scores"],
}


#: Map from artifact key to the function that resolves the on-disk path
#: for a given ``run_tag``. Keeps the dep table dry and keeps path
#: conventions owned by :mod:`paths`.
_ARTIFACT_PATH_METHODS: dict[str, str] = {
    "candidate_concepts.json": "candidate_concepts_json",
    "concepts.json": "concepts_json",
    "concepts_stage1b.json": "stage1b_concepts",
    "concepts_stage1c.json": "stage1c_concepts",
    "seeds_train.safetensors": "seeds_train",
    "seeds_eval.safetensors": "seeds_eval",
    "seed_pool_manifest.json": "seed_pool_manifest",
    "stage4_contexts_manifest.json": "stage4_contexts_manifest",
    "negative_mode_prefixes.parquet": "negative_mode_prefixes",
    "synth_training_positive.parquet": "synth_training_positive",
    "synth_training_negative.parquet": "synth_training_negative",
    "stage1b_concept_yield.json": "stage1b_concept_yield_report",
    "stage1c_concept_yield.json": "stage1c_concept_yield_report",
    "concept_features.safetensors": "concept_features",
    "concept_features_neg.safetensors": "concept_features_neg",
    "generations": "generations_dir",
    "judge_scores": "judge_scores_dir",
}


#: Artifact keys whose on-disk entity is a directory, not a regular
#: file. Used so existence checks do the right thing (``is_dir`` vs
#: ``is_file``). Everything not listed here is treated as a file.
_DIRECTORY_ARTIFACTS: frozenset[str] = frozenset({"generations", "judge_scores"})


#: Artifact keys that should be dropped from a stage's dep set when
#: ``cfg.run_negative_mode`` is ``False``. These are the
#: negative-mode-only products of Stage 1c / Stage 2.
#:
#: - ``concept_features_neg.safetensors``: Stage 2 product, required by
#:   Stage 4 only when negative-mode generation is enabled.
#: - ``synth_training_negative.parquet``: Stage 1c product, required by
#:   Stages 3a/3b/3d (training) only when negative-mode training
#:   baselines are enabled. Stage 3c (prompting) doesn't read it at all.
_NEG_MODE_ONLY_ARTIFACTS: frozenset[str] = frozenset({
    "concepts_stage1c.json",
    "concept_features_neg.safetensors",
    "stage1c_concept_yield.json",
    "synth_training_negative.parquet",
})


def _resolved_deps(cfg: AxBenchBaseConfig, stage: Stage) -> list[str]:
    """Return the artifact keys required by ``stage`` under ``cfg``.

    Applies the ``run_negative_mode=False`` skip rule: negative-mode-only
    artifacts are dropped from the dep list so downstream stages don't
    wait on a file that was never produced.

    Assumes ``stage`` is a known key in :data:`STAGE_DEPENDENCIES` —
    callers must validate this first.
    """
    deps = STAGE_DEPENDENCIES[stage]
    if cfg.run_negative_mode:
        return list(deps)
    return [d for d in deps if d not in _NEG_MODE_ONLY_ARTIFACTS]


def _artifact_exists(key: str, path: Path) -> bool:
    """Return True iff ``path`` exists as the right kind of entity for
    ``key`` (directory vs regular file).
    """
    if key in _DIRECTORY_ARTIFACTS:
        return path.is_dir()
    if key in {
        "stage1b_concept_yield.json",
        "stage1c_concept_yield.json",
    }:
        return concept_yield_report_is_committed(path)
    return path.is_file()


def _assert_stage_6_judge_coverage(cfg: AxBenchBaseConfig) -> None:
    """Require one matching Stage 5 row for every Stage 4 generation."""
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before checking Stage 6 judge coverage")
    generation_rows = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
    ).to_pylist()
    judge_rows = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
    ).to_pylist()
    assert_complete_judge_coverage(generation_rows, judge_rows)


#: The target-model names the Stage 4 per-concept dep check looks under
#: for DiffMean / LinearProbe artifacts. Stages 3a/3b emit one direction
#: per concept per target per mode and Stage 4 reads all of them.
_STAGE_4_TARGET_MODELS: tuple[str, ...] = ("decoder_only", "encdec")


def _load_concepts_for_dep_check(
    cfg: AxBenchBaseConfig,
    output_paths: paths.AxBenchBasePaths,
) -> list[Concept]:
    """Load the final Stage 1 concept authority for Stage 4 enumeration.

    The dense Stage 0c list is immutable provenance. The filtered Stage 1
    artifact is the population boundary and may retain sparse original IDs.

    Raises :class:`AxBenchBaseError` for any of:

    - unreadable feature_statistics_path (propagates from
      :func:`read_n_sae_features_from_stats`);
    - unreadable or malformed ``concepts.json`` (propagates from
      :func:`load_and_validate_concepts`).
    """
    n_sae_features = read_n_sae_features_from_stats(
        Path(cfg.feature_statistics_path),
    )
    stage0c_concepts = load_and_validate_concepts(
        output_paths.concepts_json(),
        n_sae_features=n_sae_features,
    )
    stage1b_survivors = load_concept_yield_surviving_ids(
        cfg,
        stage="stage1b",
        input_concepts=stage0c_concepts,
        path=output_paths.stage1b_concept_yield_report(),
    )
    stage1b_concepts = load_filtered_concepts(
        output_paths.stage1b_concepts(),
        parent_concepts=stage0c_concepts,
    )
    if frozenset(concept.concept_id for concept in stage1b_concepts) != (
        stage1b_survivors
    ):
        raise AxBenchBaseError(
            "Stage 1b filtered concepts do not match its concept-yield report"
        )
    if not cfg.run_negative_mode:
        return stage1b_concepts
    stage1c_survivors = load_concept_yield_surviving_ids(
        cfg,
        stage="stage1c",
        input_concepts=stage1b_concepts,
        path=output_paths.stage1c_concept_yield_report(),
    )
    stage1c_concepts = load_filtered_concepts(
        output_paths.stage1c_concepts(),
        parent_concepts=stage1b_concepts,
    )
    if frozenset(concept.concept_id for concept in stage1c_concepts) != (
        stage1c_survivors
    ):
        raise AxBenchBaseError(
            "Stage 1c filtered concepts do not match its concept-yield report"
        )
    return stage1c_concepts


def _stage_4_missing_artifacts(cfg: AxBenchBaseConfig) -> list[Path]:
    """Stage 4 (GENERATE) per-concept dep check.

    Stage 4 requires every per-concept × per-target × per-mode training
    artifact produced by Stages 3a (DiffMean), 3b (LinearProbe), 3c
    (Prompting), and 3d (LoRA) in addition to the generic Stage 0 /
    Stage 2 outputs. This helper expands the generic dep list with a
    per-concept enumeration driven by the final filtered Stage 1 list.

    The enumeration fans out as follows, where C = number of concepts
    in that final list and M = 1 + (1 if ``cfg.run_negative_mode``
    else 0):

    - DiffMean: C × M × 2 directions (one per target ×
      ``{decoder_only, encdec}``).
    - LinearProbe: C × M × 2 directions.
    - LoRA: C × M adapters (decoder-only only; no ``encdec`` variant).
    - Prompting: M description files (per mode only, NOT per concept).

    Called by :func:`list_missing_artifacts` when its ``stage`` argument
    coerces to :attr:`Stage.GENERATE`. Callers get an already-populated
    missing list if the generic deps fail and can surface those errors
    without enumerating concepts first.

    Implementation note: if any concept-handoff artifact is in the generic
    missing list we return early—we cannot validate and enumerate the final
    population until the complete Stage 1 handoff exists.

    Raises:
        AxBenchBaseError: if ``cfg.run_tag`` is unset.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before checking Stage 4 dependencies")
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    # Step 1: the generic Stage 4 dep set (Stage 0 + Stage 2 outputs,
    # with neg-mode filtering already applied).
    missing: list[Path] = []
    for key in _resolved_deps(cfg, Stage.GENERATE):
        method = getattr(output_paths, _ARTIFACT_PATH_METHODS[key])
        path = method()
        if not _artifact_exists(key, path):
            missing.append(path)

    # Can't enumerate per-concept artifacts without the concept list.
    # Return whatever generic deps are missing so the operator fixes
    # ``concepts.json`` first, then re-runs the dep check.
    concept_handoff_paths = {
        output_paths.concepts_json(),
        output_paths.stage1b_concepts(),
        output_paths.stage1b_concept_yield_report(),
    }
    if cfg.run_negative_mode:
        concept_handoff_paths.update(
            {
                output_paths.stage1c_concepts(),
                output_paths.stage1c_concept_yield_report(),
            }
        )
    if concept_handoff_paths.intersection(missing):
        return missing

    # Step 2: expand the per-concept × per-mode × per-target matrix.
    concepts = _load_concepts_for_dep_check(cfg, output_paths)
    modes: list[SteeringMode] = [SteeringMode.POSITIVE]
    if cfg.run_negative_mode:
        modes.append(SteeringMode.NEGATIVE)

    clustered = bool(
        isinstance(cfg.conditioning, Mapping)
        and cfg.conditioning.get("feature_clusters") is not None
    )
    if clustered:
        if cfg.enabled_methods is None:
            raise AxBenchBaseError(
                "Clustered Stage 4 requires an explicit enabled_methods allowlist"
            )
        unsupported = sorted(
            set(cfg.enabled_methods) - set(CLUSTERED_AXBENCH_BASE_STAGE4_METHODS)
        )
        if unsupported:
            raise AxBenchBaseError(
                "Clustered Stage 4 enabled_methods contains unsupported methods "
                f"{unsupported}; allowed="
                f"{list(CLUSTERED_AXBENCH_BASE_STAGE4_METHODS)}"
            )
        for name in cfg.disabled_methods:
            get_method_by_name(name)
        methods = [get_method_by_name(name) for name in cfg.enabled_methods]
        methods = [
            method
            for method in methods
            if method.name not in cfg.disabled_methods
            and (cfg.run_negative_mode or method.mode is SteeringMode.POSITIVE)
        ]
        prompting_modes = {
            method.mode for method in methods if method.baseline is BaselineKind.PROMPTING
        }
        for mode in sorted(prompting_modes, key=lambda value: value.value):
            prompting_path = output_paths.prompting_descriptions(mode)
            if not prompting_path.is_file():
                missing.append(prompting_path)
        scalar_methods = [
            method
            for method in methods
            if method.baseline
            in {BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE}
        ]
        lora_modes = {
            method.mode
            for method in methods
            if method.baseline is BaselineKind.LORA
        }
        for concept in concepts:
            for method in scalar_methods:
                path_fn = (
                    output_paths.diffmean_direction
                    if method.baseline is BaselineKind.DIFFMEAN
                    else output_paths.linear_probe_direction
                )
                artifact = path_fn(
                    method.target_model,
                    method.mode,
                    concept.concept_id,
                )
                if not artifact.is_file():
                    missing.append(artifact)
            for mode in sorted(lora_modes, key=lambda value: value.value):
                adapter = output_paths.lora_adapter(mode, concept.concept_id)
                if not adapter.is_file():
                    missing.append(adapter)
        return missing

    # Prompting: one JSON file per mode (NOT per concept).
    for mode in modes:
        prompting_path = output_paths.prompting_descriptions(mode)
        if not prompting_path.is_file():
            missing.append(prompting_path)

    # DiffMean + LinearProbe: one safetensors per (concept, target, mode).
    for concept in concepts:
        for mode in modes:
            for target in _STAGE_4_TARGET_MODELS:
                diffmean_path = output_paths.diffmean_direction(
                    target,
                    mode,
                    concept.concept_id,
                )
                if not diffmean_path.is_file():
                    missing.append(diffmean_path)
                probe_path = output_paths.linear_probe_direction(
                    target,
                    mode,
                    concept.concept_id,
                )
                if not probe_path.is_file():
                    missing.append(probe_path)

    # LoRA: decoder-only only, one .pt file per (concept, mode).
    for concept in concepts:
        for mode in modes:
            lora_path = output_paths.lora_adapter(mode, concept.concept_id)
            if not lora_path.is_file():
                missing.append(lora_path)

    return missing


def list_missing_artifacts(
    cfg: AxBenchBaseConfig, stage: Stage | str,
) -> list[Path]:
    """Return the list of on-disk paths that a given ``stage`` requires
    but that are missing (or of the wrong entity type).

    An empty list means every dependency is satisfied and the stage may
    run.

    ``stage`` may be passed as a :class:`Stage` member or its
    bare-string equivalent; the call coerces to an enum internally.

    :attr:`Stage.ALL` always returns ``[]`` — the pipeline dispatcher
    checks each sub-stage at its own runtime.

    When ``cfg.run_negative_mode`` is ``False``:

    - :attr:`Stage.SYNTH_NEGATIVE_MODE` returns ``[]`` regardless of
      on-disk state.
    - :attr:`Stage.GENERATE` does not require
      ``concept_features_neg.safetensors`` and does not enumerate
      negative-mode per-concept artifacts.
    - :attr:`Stage.TRAIN_DIFFMEAN` / :attr:`Stage.TRAIN_LINEAR_PROBE` /
      :attr:`Stage.TRAIN_LORA` do not require
      ``synth_training_negative.parquet``.

    :attr:`Stage.GENERATE` is special-cased to enumerate every
    per-concept × per-target × per-mode training artifact (DiffMean,
    LinearProbe, Prompting, LoRA) in addition to its generic dep list.

    Raises:
        AxBenchBaseError: if ``stage`` is unknown, if ``cfg.run_tag``
            is unset (required to resolve artifact paths), or if the
            Stage 4 dep check cannot read the feature statistics file
            to determine ``n_sae_features``.
    """
    stage_enum = Stage.coerce(stage)
    if stage_enum is Stage.ALL:
        return []
    # Skip rule: the entire negative-mode synth stage is a no-op when
    # negative mode is off.
    if stage_enum is Stage.SYNTH_NEGATIVE_MODE and not cfg.run_negative_mode:
        return []
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before checking stage dependencies"
        )
    # Stage 4 fans out into a per-concept × per-target × per-mode matrix
    # that the generic path can't express. Dispatch to the dedicated
    # helper BEFORE walking the generic dep list.
    if stage_enum is Stage.GENERATE:
        return _stage_4_missing_artifacts(cfg)
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    missing: list[Path] = []
    for key in _resolved_deps(cfg, stage_enum):
        method = getattr(output_paths, _ARTIFACT_PATH_METHODS[key])
        path = method()
        if not _artifact_exists(key, path):
            missing.append(path)
    return missing


def check_stage_dependencies(
    cfg: AxBenchBaseConfig, stage: Stage | str,
) -> None:
    """Check that every artifact required by ``stage`` is ready on disk.

    Raises ``AxBenchBaseError`` with a single message listing every
    missing path if any are absent. Stage 6 additionally requires a
    matching judge row for every generation; does nothing on success.

    ``stage`` may be passed as a :class:`Stage` member or its
    bare-string equivalent; the call coerces to an enum internally.

    :attr:`Stage.ALL` is a no-op (the pipeline dispatcher checks each
    sub-stage individually at runtime).

    :attr:`Stage.SYNTH_NEGATIVE_MODE` is a no-op when
    ``cfg.run_negative_mode`` is ``False`` — negative mode is disabled
    end-to-end.

    Raises:
        AxBenchBaseError: for unknown stages, unset ``run_tag``, or
            missing or incomplete required artifacts.
    """
    stage_enum = Stage.coerce(stage)
    if stage_enum is Stage.ALL:
        return
    if stage_enum is Stage.SYNTH_NEGATIVE_MODE and not cfg.run_negative_mode:
        return
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before checking stage dependencies"
        )
    missing = list_missing_artifacts(cfg, stage_enum)
    if not missing:
        if stage_enum is Stage.AGGREGATE:
            _assert_stage_6_judge_coverage(cfg)
        return
    bullets = "\n".join(f"  - {p}" for p in missing)
    raise AxBenchBaseError(
        f"Stage {stage_enum.value!r} cannot run: {len(missing)} required artifact"
        f"{'s' if len(missing) != 1 else ''} missing on disk:\n{bullets}"
    )
