"""Stage 4 ``generate_all`` orchestration (Task 38).

The outer loop for Stage 4: iterates every combination of
``(method, concept, eval_seed, coefficient)`` in the
:data:`~steering.eval.axbench_base.method_registry.ALL_METHODS` registry,
calls :func:`~steering.eval.axbench_base.intervention_dispatch.run_intervention`
for each combination, and appends the resulting
:class:`~steering.eval.axbench_base.generation_helpers.GenerationResult`
to the generations directory-dataset via
:class:`~steering.eval.axbench_base.parquet_writer.ParquetAppendWriter`.

Iteration structure
-------------------
For every method in :data:`ALL_METHODS` passed through the per-target
filter (see :func:`_filter_methods_for_targets`):

1. Pick the right target model and metadata from the two
   caller-supplied dicts (keyed on ``"decoder_only"`` / ``"encdec"``).
2. Filter out methods whose ``requires_negative_sampling`` is True and
   whose target model does NOT have a ``negative_selection`` config
   (there's no inactive partition for the modifier to mutate).
3. For each concept and for each eval_seed in the mode-appropriate pool
   (positive-mode methods iterate ``eval_seeds_positive``, negative-mode
   methods iterate ``eval_seeds_negative``), expand the method's
   coefficient grid via ``method.coefficient_grid_fn(meta, cfg)`` and
   emit one row per ``(method, concept, eval_seed, coefficient)``.
4. Skip rows whose ``generation_key`` is already present in the output
   directory (crash-safe resume).

Schema
------
The output parquet rows match :func:`judge._expected_generations_schema`
so Stage 5 (``judge_all``) can ``load_existing`` the directory and iterate
rows without a schema mismatch. The first seven columns carry the stable
human-facing generation identity and result. Seven additional columns carry
the target, visible seed prefix, exact model-ready prompt key/count, committed
seed-pool fingerprint, generated-token count, and stop reason. See
:data:`GENERATIONS_SCHEMA` for the canonical definition.

The ``generation_key`` format is deterministic and human-readable:

    "{method_name}|c{concept_id}|s{eval_seed}|k{coef_str}|{target_model_name}"

where ``coef_str`` is ``str(coefficient)`` (so ``"None"`` for single-condition
grids, ``"1.5"`` for scalar grids, ``"3"`` for bucket grids).

Concept eligibility
-------------------
The orchestrator receives the final filtered Stage 1 concept authority.
Corpus coverage is validated before this entry point, so every enabled method
sees that same authoritative concept list.

Resume strategy
---------------
On every call, the writer scans the generations directory for existing
batches via :func:`load_existing`. Every existing row must have a unique,
well-formed ``generation_key`` plus an exact resume identity consisting of
``prefix_text``, ``prompt_token_key``, ``prompt_token_count``, and
``seed_pool_fingerprint``. A key is skipped only when all four current values
match. Combined with ``ParquetAppendWriter``'s zero-padded batch cursor, a
crash mid-Stage-4 resumes exactly; stale or malformed rows fail loudly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch as t
from loguru import logger
from phantom.sized import NonEmptyStr

from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.cluster_artifacts import (
    commit_cluster_generation_marker,
    validate_cluster_generation_resume,
)
from steering.eval.axbench_base.cluster_context_cache import (
    clustered_context_cache_metadata_for_contract,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.direction_scale import (
    STEERING_SCALE_MARKER_NAME,
    DirectionScaleRequest,
    enforce_steering_scale_identity,
    preflight_direction_scales,
)
from steering.eval.axbench_base.intervention_dispatch import (
    _load_scalar_direction_artifact,
    _load_top_k_concept_fids,
    run_intervention,
)
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    BaselineKind,
    Method,
    get_method_by_name,
)
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    _scan_batch_indices,
    load_existing,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.seed_contract import (
    load_seed_pool_fingerprint,
    load_seed_pool_snapshot,
    sha256_bytes,
)
from steering.eval.axbench_base.stage4_contexts import (
    load_stage4_contexts_manifest,
    validate_stage4_contexts_for_generation,
)
from steering.eval.axbench_base.stage4_prompts import (
    build_stage4_prompt_tokens,
    load_prompting_descriptions,
    prompt_token_key,
    prompting_prefix_text,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    stream_hash_file_stable,
)
from steering.segmentation import align_segments_to_decoded_prefix

_STAGE4_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.axbench_base_stage4_preflight"
)


@dataclass(frozen=True)
class Stage4MethodPlan:
    """One executable Stage 4 method after target/seed/grid resolution."""

    method: Method
    target_name: str
    bundle: TargetBundle
    target_meta: TargetModelMetadata
    seeds_for_mode: list[SeedRow]
    coefficients: list[Any]
    concept_features_path: Path
    cluster_source_corpus_sha256: str | None = None


@dataclass(frozen=True)
class PendingGenerationRow:
    """One pending Stage 4 row after resume filtering."""

    generation_key: str
    concept: Concept
    eval_seed: int
    coefficient: Any
    seed_row: SeedRow
    prompt_tokens: tuple[int, ...]
    prompt_token_key: str


@dataclass(frozen=True)
class ExistingGenerationIdentity:
    """Resume-critical identity committed with one generation row."""

    prefix_text: str
    prompt_token_key: str
    prompt_token_count: int
    seed_pool_fingerprint: str


def _derive_stage4_direction_scale_requests(
    method_plans: Sequence[Stage4MethodPlan],
    *,
    eligible_by_mode: Mapping[SteeringMode, Sequence[Concept]],
    output_paths: paths.AxBenchBasePaths,
) -> list[DirectionScaleRequest]:
    """Return exactly the scalar direction artifacts Stage 4 can consume."""
    requests: list[DirectionScaleRequest] = []
    for plan in method_plans:
        method = plan.method
        if method.baseline not in (
            BaselineKind.DIFFMEAN,
            BaselineKind.LINEAR_PROBE,
        ):
            continue
        path_fn = (
            output_paths.diffmean_direction
            if method.baseline is BaselineKind.DIFFMEAN
            else output_paths.linear_probe_direction
        )
        for concept in eligible_by_mode[method.mode]:
            requests.append(
                DirectionScaleRequest(
                    family=method.baseline.value,
                    target_model=plan.target_name,
                    mode=method.mode.value,
                    concept_id=concept.concept_id,
                    path=path_fn(
                        plan.target_name,
                        method.mode,
                        concept.concept_id,
                    ),
                )
            )
    return requests


__all__ = (
    "GENERATIONS_SCHEMA",
    "GENERATIONS_WRITE_BATCH_SIZE",
    "build_generation_key",
    "generate_all",
)


# Row buffer size before an incremental parquet flush. Small enough that
# a crash mid-Stage-4 doesn't lose much work; large enough that per-flush
# metadata overhead stays negligible on long sweeps.
GENERATIONS_WRITE_BATCH_SIZE = 100


# Canonical Stage 4 generations schema. The first seven columns match
# :func:`judge._expected_generations_schema` exactly so Stage 5 can
# ``load_existing`` the dir and iterate rows. Additional columns carry
# full provenance that later debugging stages can join back on the
# ``generation_key`` field.
GENERATIONS_SCHEMA: pa.Schema = pa.schema(
    [
        ("generation_key", pa.string()),
        ("concept_id", pa.int32()),
        ("method_name", pa.string()),
        ("mode", pa.string()),
        ("eval_seed", pa.int32()),
        ("coefficient", pa.string()),
        ("generation_text", pa.string()),
        ("target_model", pa.string()),
        ("prefix_text", pa.string()),
        ("prompt_token_key", pa.string()),
        ("prompt_token_count", pa.int32()),
        ("seed_pool_fingerprint", pa.string()),
        ("token_count", pa.int32()),
        ("stop_reason", pa.string()),
    ],
)


def build_generation_key(
    *,
    method_name: str,
    concept_id: int,
    eval_seed: int,
    coefficient: Any,
    target_model_name: str,
) -> str:
    """Return the deterministic, human-readable generation key.

    The format is::

        "{method_name}|c{concept_id}|s{eval_seed}|k{coef_str}|{target_model_name}"

    where ``coef_str`` is the plain ``str()`` of the coefficient. This
    produces ``"None"`` for single-condition grids,
    ``"1.5"`` for scalar grids, and ``"3"`` for integer bucket grids —
    the downstream parquet ``coefficient`` column stores the same
    stringified form so they match trivially.
    """
    coef_str = str(coefficient)
    return (
        f"{method_name}|c{concept_id}|s{eval_seed}|k{coef_str}"
        f"|{target_model_name}"
    )


def _method_target_has_negative_sampling(
    method: Method,
    target_model_metas: dict[str, TargetModelMetadata],
) -> bool:
    """Return True if the method's target model was trained with negative sampling.

    The filter is per-method because different methods route to
    different target models; a single run may load both a
    ``decoder_only`` target (no encoder, no negative sampling) and an
    ``encdec`` target (which may or may not have negative sampling
    depending on how it was trained).
    """
    meta = target_model_metas.get(method.target_model)
    if meta is None:
        return False
    return meta.negative_selection is not None


def _filter_methods_for_targets(
    all_methods: list[Method],
    *,
    enabled_methods: list[str] | None,
    disabled_methods: list[str],
    target_model_metas: dict[str, TargetModelMetadata],
    run_negative_mode: bool,
) -> list[Method]:
    """Apply Stage 4's method filters across heterogeneous target models.

    The single-call :func:`filter_methods` helper takes a boolean
    ``target_model_has_negative_sampling`` flag, but Stage 4 runs with
    up to two distinct target models simultaneously, so we evaluate the
    flag per method before invoking the filter. Enabled / disabled
    string filters are applied uniformly.

    When ``run_negative_mode`` is ``False``, every method whose
    ``mode == SteeringMode.NEGATIVE`` is excluded unconditionally.  This
    prevents Stage 4 from iterating negative-mode methods when Stages
    1c/2/3 never produced negative-mode artifacts (concept_features_neg,
    negative baselines, synth_training_negative), which would otherwise
    cause a ``FileNotFoundError`` or ``AxBenchBaseError`` at load time.
    The check is applied BEFORE the ``enabled_methods`` / ``disabled_methods``
    / ``requires_negative_sampling`` filters so negative methods are
    dropped early.

    ``enabled_methods`` validation: if any name in ``enabled_methods`` is
    not a registered method, we raise :class:`AxBenchBaseError` at the
    top of the loop rather than silently dropping the name, so a typo in
    a user-supplied config surfaces early.
    """
    if enabled_methods is not None:
        # Surface typos in enabled_methods early — otherwise a silent
        # drop would make the user wonder why their method never ran.
        for name in enabled_methods:
            get_method_by_name(name)  # raises on unknown
    for name in disabled_methods:
        get_method_by_name(name)  # raises on unknown

    out: list[Method] = []
    for method in all_methods:
        if not run_negative_mode and method.mode == SteeringMode.NEGATIVE:
            continue
        if enabled_methods is not None and method.name not in enabled_methods:
            continue
        if method.name in disabled_methods:
            continue
        if method.requires_negative_sampling and not (
            _method_target_has_negative_sampling(method, target_model_metas)
        ):
            continue
        out.append(method)
    return out


def _load_existing_generation_metadata(
    generations_dir: Path,
    *,
    expected_max_new_tokens: int,
) -> dict[str, ExistingGenerationIdentity]:
    """Build the exact per-key identity required for safe resume.

    A missing / empty directory returns an empty dict. A populated
    directory reads every batch file and extracts the ``generation_key``
    and exact prompt/seed identity columns. Schema-mismatched batches raise
    :class:`AxBenchBaseError` via ``load_existing``, bubbling up so the
    caller sees the corruption rather than silently re-generating.

    The key format is historically index-based (``s0``, ``s1``, ...), so
    the visible prefix, exact model-ready token key/count, and committed seed
    fingerprint are all required to prevent a stale row from being skipped
    against a different logical seed or prompt.
    """
    if (
        not isinstance(expected_max_new_tokens, int)
        or isinstance(expected_max_new_tokens, bool)
        or expected_max_new_tokens <= 0
    ):
        raise AxBenchBaseError(
            "Stage 4 resume expected_max_new_tokens must be a positive "
            f"integer, got {expected_max_new_tokens!r}."
        )
    if not generations_dir.exists():
        return {}
    with profile_region("stage4.load_existing_generations", path=generations_dir):
        existing = load_existing(generations_dir, schema=GENERATIONS_SCHEMA)
    out: dict[str, ExistingGenerationIdentity] = {}
    for row_index, row in enumerate(existing.to_pylist()):
        key = row["generation_key"]
        concept_id = row["concept_id"]
        method_name = row["method_name"]
        mode = row["mode"]
        eval_seed = row["eval_seed"]
        coefficient = row["coefficient"]
        generation_text = row["generation_text"]
        target_model = row["target_model"]
        prefix = row["prefix_text"]
        prompt_key = row["prompt_token_key"]
        prompt_count = row["prompt_token_count"]
        seed_fingerprint = row["seed_pool_fingerprint"]
        token_count = row["token_count"]
        stop_reason = row["stop_reason"]
        valid_prompt_key = (
            isinstance(prompt_key, str)
            and prompt_key.startswith("tokens_sha256:")
            and len(prompt_key) == len("tokens_sha256:") + 64
            and all(
                char in "0123456789abcdef"
                for char in prompt_key.removeprefix("tokens_sha256:")
            )
        )
        valid_seed_fingerprint = (
            isinstance(seed_fingerprint, str)
            and len(seed_fingerprint) == 64
            and all(char in "0123456789abcdef" for char in seed_fingerprint)
        )
        basic_row_is_valid = (
            isinstance(concept_id, int)
            and not isinstance(concept_id, bool)
            and concept_id >= 0
            and isinstance(method_name, str)
            and bool(method_name)
            and isinstance(mode, str)
            and mode in {item.value for item in SteeringMode}
            and isinstance(eval_seed, int)
            and not isinstance(eval_seed, bool)
            and eval_seed >= 0
            and isinstance(coefficient, str)
            and bool(coefficient)
            and isinstance(generation_text, str)
            and isinstance(target_model, str)
            and bool(target_model)
            and isinstance(token_count, int)
            and not isinstance(token_count, bool)
            and token_count >= 0
            and stop_reason in {"bos_token", "max_tokens"}
        )
        termination_is_valid = (
            stop_reason == "max_tokens"
            and token_count == expected_max_new_tokens
        ) or (
            stop_reason == "bos_token"
            and isinstance(token_count, int)
            and not isinstance(token_count, bool)
            and 0 <= token_count < expected_max_new_tokens
        )
        method: Method | None = None
        if basic_row_is_valid:
            try:
                method = get_method_by_name(method_name)
            except AxBenchBaseError:
                basic_row_is_valid = False
        semantic_row_is_valid = (
            method is not None
            and mode == method.mode.value
            and target_model == method.target_model
            and key
            == build_generation_key(
                method_name=method_name,
                concept_id=concept_id,
                eval_seed=eval_seed,
                coefficient=coefficient,
                target_model_name=target_model,
            )
        )
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(prefix, str)
            or not prefix
            or not valid_prompt_key
            or not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or prompt_count <= 0
            or not valid_seed_fingerprint
            or not basic_row_is_valid
            or not termination_is_valid
            or not semantic_row_is_valid
        ):
            raise AxBenchBaseError(
                "Stage 4 resume found a malformed exact generation identity "
                f"or row at row_index={row_index} in {generations_dir}: "
                f"row={row!r}. Use a fresh run tag; malformed generation "
                "rows cannot be resumed safely."
            )
        if key in out:
            raise AxBenchBaseError(
                f"Duplicate generation_key {key!r} found in "
                f"{generations_dir}. Refusing Stage 4 resume because the "
                "existing generations directory is not key-unique."
            )
        out[key] = ExistingGenerationIdentity(
            prefix_text=prefix,
            prompt_token_key=prompt_key,
            prompt_token_count=prompt_count,
            seed_pool_fingerprint=seed_fingerprint,
        )
    return out


def _eval_seeds_for_mode(
    mode: SteeringMode,
    *,
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    cap: int | None = None,
) -> list[SeedRow]:
    """Return the seed pool for a given method mode.

    Positive-mode methods consume the ``eval_seeds_positive`` pool
    (neutral prefixes with no concept preference); negative-mode methods
    consume ``eval_seeds_negative`` (concept-rich prefixes whose
    continuation should drift away from the concept). Both pools are
    produced by Stage 0b.

    ``cap`` truncates the pool to the first ``cap`` rows. ``None``
    means "no cap" (use the full pool); any ``int`` — including ``0``
    — is a literal truncation size. ``cap == 0`` therefore yields an
    empty list, which is the intuitive "run zero seeds for this mode"
    semantics operators expect when they want to disable one mode's
    evaluation without touching method enablement.

    Stage 0b shuffles the pool deterministically on the
    ``cfg.seed_pool_seed`` before splitting train/eval, so taking the
    head is a random-but-reproducible subsample. Callers pass
    ``cfg.generation.eval_seeds_per_concept_{positive|negative}`` so
    the generation matrix stays bounded in practice — the full eval
    pool contains thousands of rows and iterating it fully per
    (concept, method, coefficient) would be intractable.

    Raises:
        AxBenchBaseError: if ``cap`` is a negative integer (``cap < 0``
            is never a sensible cap; callers who want "no cap" pass
            ``None``).
    """
    if mode == SteeringMode.POSITIVE:
        pool = eval_seeds_positive
    elif mode == SteeringMode.NEGATIVE:
        pool = eval_seeds_negative
    else:
        raise AxBenchBaseError(
            f"unknown method mode {mode!r}; expected 'positive' or 'negative'"
        )
    if cap is None:
        return pool
    if cap < 0:
        raise AxBenchBaseError(
            f"eval_seeds_per_concept cap must be None (no cap) or a "
            f"non-negative int; got {cap}"
        )
    return pool[:cap]


def _concept_features_path_for_mode(
    output_paths: paths.AxBenchBasePaths,
    mode: SteeringMode,
) -> Path:
    """Return the Stage 2 concept-features artifact path for a given mode.

    Positive-mode methods read ``concept_features.safetensors``;
    negative-mode methods read ``concept_features_neg.safetensors``.
    Baseline methods (DiffMean / LinearProbe / LoRA / Prompting) never
    touch this file, so the path is still returned for the ``ours``
    dispatcher's consumption — non-ours dispatch ignores it.
    """
    if mode == SteeringMode.POSITIVE:
        return output_paths.concept_features()
    return output_paths.concept_features_neg()


def _build_stage4_method_plans(
    cfg: AxBenchBaseConfig,
    *,
    output_paths: paths.AxBenchBasePaths,
    targets: list[TargetBundle],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
) -> list[Stage4MethodPlan]:
    """Resolve the executable Stage 4 method plans for this run.

    The main writer loop needs target lookup, method filtering, seed-pool
    resolution, coefficient expansion, and per-mode concept-features path
    selection; centralizing that planning keeps it in one place.
    """
    with profile_region("stage4.build_target_lookup", rows=len(targets)):
        target_lookup: dict[str, TargetBundle] = {t.name: t for t in targets}
        target_model_metas: dict[str, TargetModelMetadata] = {
            t.name: t.meta for t in targets
        }

    with profile_region("stage4.filter_methods", rows=len(ALL_METHODS)):
        methods = _filter_methods_for_targets(
            ALL_METHODS,
            enabled_methods=cfg.enabled_methods,
            disabled_methods=cfg.disabled_methods,
            target_model_metas=target_model_metas,
            run_negative_mode=cfg.run_negative_mode,
        )
    concept_features_paths: dict[SteeringMode, Path] = {
        SteeringMode.POSITIVE: _concept_features_path_for_mode(
            output_paths,
            SteeringMode.POSITIVE,
        ),
        SteeringMode.NEGATIVE: _concept_features_path_for_mode(
            output_paths,
            SteeringMode.NEGATIVE,
        ),
    }

    plans: list[Stage4MethodPlan] = []
    cluster_corpus_hashes: dict[SteeringMode, str] = {}
    for method in methods:
        target_name = method.target_model
        bundle = target_lookup.get(target_name)
        if bundle is None:
            raise AxBenchBaseError(
                f"Stage 4 method {method.name!r} requires target "
                f"{target_name!r}, but no matching TargetBundle was supplied. "
                "Refusing to silently omit an enabled method from generation."
            )
        if target_name == "encdec" and bundle.context_provider is None:
            raise AxBenchBaseError(
                "AxBench-Base Stage 4 encoder-decoder generation requires "
                "TargetBundle.context_provider with real prompt/context "
                "features. Unconditioned or target-only encoder-decoder "
                f"generation is unsupported; method {method.name!r} would run "
                "without full-context conditioning."
            )

        seeds_cap = (
            cfg.generation.eval_seeds_per_concept_positive
            if method.mode == SteeringMode.POSITIVE
            else cfg.generation.eval_seeds_per_concept_negative
        )
        with profile_region(
            "stage4.resolve_method_seed_pool",
            method=method.name,
            mode=method.mode,
        ):
            seeds_for_mode = _eval_seeds_for_mode(
                method.mode,
                eval_seeds_positive=eval_seeds_positive,
                eval_seeds_negative=eval_seeds_negative,
                cap=seeds_cap,
            )

        with profile_region(
            "stage4.coefficient_grid",
            method=method.name,
            target=target_name,
        ):
            coefficients = method.coefficient_grid_fn(bundle.meta, cfg)
        if not coefficients:
            logger.warning(
                f"generate_all: method {method.name!r} produced an "
                "empty coefficient grid; no rows will be written for "
                f"this (method, target={method.target_model}) and it "
                "will be absent from the Stage 6 report.",
            )
            continue

        cluster_source_corpus_sha256: str | None = None
        if bundle.cluster_contract is not None and method.baseline in (
            BaselineKind.DIFFMEAN,
            BaselineKind.LINEAR_PROBE,
        ):
            cluster_source_corpus_sha256 = cluster_corpus_hashes.get(method.mode)
            if cluster_source_corpus_sha256 is None:
                corpus_path = output_paths.synth_training(method.mode)
                try:
                    cluster_source_corpus_sha256 = stream_hash_file_stable(corpus_path).sha256
                except Exception as exc:
                    raise AxBenchBaseError(
                        f"Clustered Stage 4 could not freeze scalar source corpus {corpus_path}"
                    ) from exc
                cluster_corpus_hashes[method.mode] = cluster_source_corpus_sha256

        plans.append(
            Stage4MethodPlan(
                method=method,
                target_name=target_name,
                bundle=bundle,
                target_meta=bundle.meta,
                seeds_for_mode=seeds_for_mode,
                coefficients=coefficients,
                concept_features_path=concept_features_paths[method.mode],
                cluster_source_corpus_sha256=cluster_source_corpus_sha256,
            ),
        )
    return plans


def _authoritative_concepts_by_mode(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
) -> dict[SteeringMode, list[Concept]]:
    """Expose the one Stage 1 authority only for enabled Stage 4 modes."""
    return {
        SteeringMode.POSITIVE: concepts,
        SteeringMode.NEGATIVE: concepts if cfg.run_negative_mode else [],
    }


def _required_prompting_descriptions(
    cfg: AxBenchBaseConfig,
    plans: list[Stage4MethodPlan],
) -> dict[SteeringMode, dict[int, str]]:
    """Load each enabled prompting-mode mapping exactly once."""

    modes = {
        plan.method.mode
        for plan in plans
        if plan.method.baseline is BaselineKind.PROMPTING
    }
    return {
        mode: load_prompting_descriptions(cfg, mode)
        for mode in sorted(modes, key=lambda value: value.value)
    }


def _prompt_tokens_for_row(
    plan: Stage4MethodPlan,
    concept: Concept,
    seed_row: SeedRow,
    prompting_descriptions: dict[SteeringMode, dict[int, str]],
) -> list[int]:
    """Construct the live Stage 4 prompt through the shared exact builder."""

    prompting_prefix: str | None = None
    if plan.method.baseline is BaselineKind.PROMPTING:
        descriptions = prompting_descriptions.get(plan.method.mode, {})
        description = descriptions.get(concept.concept_id)
        if description is None:
            raise AxBenchBaseError(
                "Prompting description missing for "
                f"mode={plan.method.mode.value}, concept_id={concept.concept_id}. "
                "Rerun Stage 3c under a fresh run tag."
            )
        prompting_prefix = prompting_prefix_text(plan.method.mode, description)
    return build_stage4_prompt_tokens(
        tokenizer=plan.bundle.tokenizer,
        seed_content_tokens=seed_row.prefix_content_nanochat_tokens,
        prompting_prefix=prompting_prefix,
    )


def _target_sequence_len(model: Any, *, target_name: str) -> int:
    """Resolve a decoder limit from either nanochat target model shape.

    Decoder-only models expose ``config.sequence_len``.  The raw conditional
    encoder-decoder loaded by :class:`RuntimeState` exposes the corresponding
    decoder config as ``dec_config``; only its per-prompt eval wrapper has a
    ``config`` alias.  Stage 4 preflight runs before that wrapper exists.
    """

    for config_name in ("config", "dec_config"):
        config = getattr(model, config_name, None)
        raw = getattr(config, "sequence_len", None)
        if isinstance(raw, Integral) and not isinstance(raw, bool):
            sequence_len = int(raw)
            if sequence_len > 0:
                return sequence_len
    raise AxBenchBaseError(
        f"Target {target_name!r} does not expose a positive sequence_len "
        "through model.config or model.dec_config"
    )


def _preflight_stage4_prompt_lengths(
    cfg: AxBenchBaseConfig,
    *,
    plans: list[Stage4MethodPlan],
    eligible_by_mode: dict[SteeringMode, list[Concept]],
    prompting_descriptions: dict[SteeringMode, dict[int, str]],
) -> None:
    """Validate every method plan against its checkpoint before any write."""

    for plan in plans:
        sequence_len = _target_sequence_len(
            plan.bundle.model, target_name=plan.target_name
        )
        concepts = eligible_by_mode[plan.method.mode]
        if not concepts or not plan.seeds_for_mode:
            continue
        concepts_to_check = (
            concepts
            if plan.method.baseline is BaselineKind.PROMPTING
            else concepts[:1]
        )
        for concept in concepts_to_check:
            for seed_index, seed_row in enumerate(plan.seeds_for_mode):
                tokens = _prompt_tokens_for_row(
                    plan, concept, seed_row, prompting_descriptions
                )
                requested = len(tokens) + cfg.generation.max_new_tokens
                if requested > sequence_len:
                    raise AxBenchBaseError(
                        "Stage 4 prompt preflight exceeds the target checkpoint "
                        f"length before generation: method={plan.method.name!r}, "
                        f"concept_id={concept.concept_id}, eval_seed={seed_index}, "
                        f"prompt_token_count={len(tokens)}, "
                        f"max_new_tokens={cfg.generation.max_new_tokens}, "
                        f"sequence_len={sequence_len}."
                    )


def _validate_stage4_context_cache(
    cfg: AxBenchBaseConfig,
    *,
    plans: list[Stage4MethodPlan],
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
) -> str | None:
    """Validate Stage 3e and every required cache row before generation."""

    encdec_plans = [plan for plan in plans if plan.target_name == "encdec"]
    if not encdec_plans:
        return None
    bundle = encdec_plans[0].bundle
    if any(plan.bundle is not bundle for plan in encdec_plans):
        raise AxBenchBaseError("Stage 4 encdec plans do not share one target bundle")
    sequence_len = _target_sequence_len(bundle.model, target_name="encdec")
    manifest, records = validate_stage4_contexts_for_generation(
        cfg,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=bundle.tokenizer,
        encdec_meta=bundle.meta,
        encdec_sequence_len=sequence_len,
        cluster_contract=bundle.cluster_contract,
    )
    provider = bundle.context_provider
    if provider is None:
        raise AxBenchBaseError("Encoder-decoder Stage 4 context provider is missing")
    cache_manifest = getattr(provider, "manifest", None)
    if not isinstance(cache_manifest, dict):
        raise AxBenchBaseError(
            "Encoder-decoder Stage 4 requires a validated safetensors context provider"
        )
    cache_metadata = cache_manifest.get("metadata")
    if not isinstance(cache_metadata, dict):
        raise AxBenchBaseError("Context cache manifest metadata is malformed")
    expected_cache_metadata = {
        "model_max_seq_len": sequence_len,
        "contexts_fingerprint": manifest["contexts_fingerprint"],
    }
    if bundle.cluster_contract is not None:
        expected_cache_metadata.update(
            clustered_context_cache_metadata_for_contract(
                bundle.cluster_contract
            )
        )
    for key, expected in expected_cache_metadata.items():
        if cache_metadata.get(key) != expected:
            raise AxBenchBaseError(
                f"Context cache metadata mismatch for {key}: "
                f"cache={cache_metadata.get(key)!r}, expected={expected!r}"
            )
    for record in records:
        tokens = record["token_ids"]
        segments = provider.get(tokens)
        if not isinstance(segments, tuple) or not segments:
            raise AxBenchBaseError(
                "Stage 4 context cache is missing a non-empty segment tuple for "
                f"{record['context_key']}"
            )
        previous_end = -1
        for segment in segments:
            feature_ids = getattr(segment, "feature_ids", None)
            buckets = getattr(segment, "buckets", None)
            char_start = getattr(segment, "char_start", None)
            char_end = getattr(segment, "char_end", None)
            # Feature order is part of the training-selection contract:
            # positive sampler order comes first, followed by negative sampler
            # order. Require uniqueness and aligned buckets, but never impose a
            # cosmetic sort on either clustered or ordinary cache rows.
            if (
                not isinstance(feature_ids, tuple)
                or not isinstance(buckets, tuple)
                or len(feature_ids) != len(buckets)
                or any(
                    not isinstance(feature_id, int)
                    or isinstance(feature_id, bool)
                    or feature_id < 0
                    or (
                        bundle.meta.n_sae_features is not None
                        and feature_id >= bundle.meta.n_sae_features
                    )
                    for feature_id in feature_ids
                )
                or len(set(feature_ids)) != len(feature_ids)
                or any(
                    not isinstance(bucket, int)
                    or isinstance(bucket, bool)
                    or bucket < 0
                    or (
                        bundle.meta.num_buckets is not None
                        and bucket >= bundle.meta.num_buckets
                    )
                    for bucket in buckets
                )
                or not isinstance(char_start, int)
                or not isinstance(char_end, int)
                or isinstance(char_start, bool)
                or isinstance(char_end, bool)
                or char_start < 0
                or char_end <= char_start
                or char_start < previous_end
            ):
                raise AxBenchBaseError(
                    "Stage 4 context cache returned a structurally invalid "
                    f"segment for {record['context_key']}: {segment!r}"
                )
            previous_end = char_end
            if bundle.cluster_contract is not None:
                bundle.cluster_contract.consolidation.assert_representatives_only(
                    t.tensor(feature_ids, dtype=t.int64),
                    surface=(
                        "AxBench Stage 4 cached context segment "
                        f"{record['context_key']}"
                    ),
                )
        try:
            aligned_segments = align_segments_to_decoded_prefix(
                tokenizer=bundle.tokenizer,
                prefix_tokens=tokens,
                segments=segments,
                counter_name=_STAGE4_ALIGNMENT_COUNTER,
            )
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError(
                "Stage 4 context cache segments cannot be aligned to the "
                "exact decoded prompt before generation: "
                f"context_key={record['context_key']!r}, error={exc}"
            ) from exc
        if not aligned_segments:
            raise AxBenchBaseError(
                "Stage 4 context cache alignment removed every segment before "
                f"generation for {record['context_key']}"
            )
        decoded_length = len(bundle.tokenizer.decode(tokens))
        previous_end = -1
        for segment in aligned_segments:
            if (
                isinstance(segment.char_start, bool)
                or isinstance(segment.char_end, bool)
                or segment.char_start < 0
                or segment.char_end <= segment.char_start
                or segment.char_end > decoded_length
                or segment.char_start < previous_end
            ):
                raise AxBenchBaseError(
                    "Stage 4 context cache alignment returned a structurally "
                    f"invalid segment for {record['context_key']}: {segment!r}"
                )
            previous_end = segment.char_end
    return str(manifest["contexts_fingerprint"])


def _cluster_contract_for_targets(
    targets: list[TargetBundle],
) -> ClusterFeatureContract | None:
    """Return the one common clustered contract, rejecting mixed targets."""

    contracts = [
        target.cluster_contract
        for target in targets
        if target.cluster_contract is not None
    ]
    if not contracts:
        return None
    if len(contracts) != len(targets):
        raise AxBenchBaseError(
            "Stage 4 target bundles mix clustered and unclustered contract state"
        )
    fingerprint = contracts[0].feature_contract_fingerprint
    if any(
        contract.feature_contract_fingerprint != fingerprint
        for contract in contracts[1:]
    ):
        raise AxBenchBaseError(
            "Stage 4 target bundles carry different feature contracts"
        )
    return contracts[0]


def _stable_cluster_artifact_sha256(path: Path, *, surface: str) -> str:
    """Freeze one clustered Stage 4 input without trusting a mutable stat."""

    try:
        return stream_hash_file_stable(path).sha256
    except Exception as exc:
        raise AxBenchBaseError(
            f"Clustered Stage 4 could not stably hash {surface} at {path}"
        ) from exc


def _bind_cluster_generation_sources(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    consumed_eval_seeds: list[SeedRow],
) -> tuple[str, str]:
    """Bind hashed source artifacts to the exact in-memory Stage 4 inputs."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("Clustered Stage 4 source binding requires cfg.run_tag")
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    try:
        before = stream_hash_file_stable(concepts_path)
        payload = before.path.read_bytes()
        after = stream_hash_file_stable(concepts_path)
    except Exception as exc:
        raise AxBenchBaseError(
            "Clustered Stage 4 could not stably load final concepts at "
            f"{concepts_path}"
        ) from exc
    if (
        before != after
        or len(payload) != before.byte_length
        or sha256_bytes(payload) != before.sha256
    ):
        raise AxBenchBaseError(
            "Clustered Stage 4 final concepts changed while being bound"
        )
    try:
        disk_concepts = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            f"Clustered Stage 4 final concepts are not valid JSON: {concepts_path}"
        ) from exc

    seed_snapshot = load_seed_pool_snapshot(paths.AxBenchBasePaths.from_config(cfg))
    mismatches: list[str] = []
    if disk_concepts != [asdict(concept) for concept in concepts]:
        mismatches.append("concepts.json")
    if seed_snapshot.eval_rows != tuple(consumed_eval_seeds):
        mismatches.append("seeds_eval.safetensors")
    if mismatches:
        raise AxBenchBaseError(
            "Clustered Stage 4 source artifacts do not match the in-memory "
            "inputs consumed by generation: "
            + ", ".join(mismatches)
        )
    return before.sha256, seed_snapshot.fingerprint


def _context_cache_manifest_sha256(bundle: TargetBundle) -> str:
    """Hash the exact manifest bytes consumed by the validated provider."""

    # ACCEPTED RESIDUAL: ``save_context_cache`` supports ``overwrite=True``
    # and installs its manifest through ``tmp.rename(path)`` -- an atomic
    # replacement, not a create-once install. ``generate_all`` only reads it;
    # a separate cache builder produces it out-of-band before the eval run.
    # The provider has already read each manifest-listed shard once, verified
    # the digest of those bytes, and decoded those same bytes into memory. The
    # generation marker binds the manifest digest returned here. A later rename
    # therefore cannot mix new cache rows into this run, but it can make the
    # on-disk path differ from the digest this run correctly records.
    # Re-reading immediately before marker installation would only move that
    # boundary: a non-cooperating writer can rename again afterward. The next
    # run rejects the drift through the marker/manifest cross-check (or the
    # provider's shard checksums).
    # Closing the disk-state residual would require a construction change such
    # as immutable publication, not another check. The
    # ``initial_contexts_fingerprint`` versus
    # ``validated_contexts_fingerprint`` cross-check separately guards which
    # Stage-3e contexts were enumerated.
    provider = bundle.context_provider
    if provider is None:
        raise AxBenchBaseError(
            "Clustered Stage 4 encoder-decoder target has no context provider"
        )
    manifest_path = getattr(provider, "manifest_path", None)
    if isinstance(manifest_path, Path) and manifest_path.is_file():
        try:
            digest = stream_hash_file_stable(manifest_path)
            payload = digest.path.read_bytes()
            if (
                len(payload) != digest.byte_length
                or sha256_bytes(payload) != digest.sha256
            ):
                raise AxBenchBaseError(
                    "Clustered Stage 4 context-cache manifest changed while "
                    f"it was being loaded: {manifest_path}"
                )
            current = canonical_json(json.loads(payload))
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(
                "Clustered Stage 4 could not freeze the context-cache manifest "
                f"at {manifest_path}"
            ) from exc
        loaded_manifest = getattr(provider, "manifest", None)
        if (
            not isinstance(loaded_manifest, dict)
            or current != canonical_json(loaded_manifest)
        ):
            raise AxBenchBaseError(
                "Clustered Stage 4 context-cache manifest changed after the "
                "validated provider loaded it"
            )
        return digest.sha256
    manifest = getattr(provider, "manifest", None)
    if not isinstance(manifest, dict):
        raise AxBenchBaseError(
            "Clustered Stage 4 context provider has no exact manifest identity"
        )
    # Unit/fake providers may expose only the already-validated manifest.
    return sha256_bytes(canonical_json(manifest).encode("utf-8"))


def _checkpoint_refs_for_plans(
    cfg: AxBenchBaseConfig,
    plans: list[Stage4MethodPlan],
) -> dict[str, dict[str, Any]]:
    """Freeze explicit checkpoint references for every active target."""

    active_targets = {plan.target_name for plan in plans}
    refs: dict[str, dict[str, Any]] = {}
    for name, raw in (
        ("decoder_only", cfg.decoder_only),
        ("encdec", cfg.encdec),
    ):
        if name not in active_targets:
            continue
        missing = [
            key for key in ("source", "model_tag", "step") if key not in raw
        ]
        if missing:
            raise AxBenchBaseError(
                "Clustered Stage 4 checkpoint reference is missing required "
                f"field(s) for {name}: {missing}"
            )
        refs[name] = {
            key: raw[key] for key in ("source", "model_tag", "step")
        }
        if name == "decoder_only":
            pins = [
                plan.bundle.checkpoint_files_pin.semantic_dict()
                for plan in plans
                if plan.target_name == name
                and plan.bundle.checkpoint_files_pin is not None
            ]
            if not pins or any(pin != pins[0] for pin in pins[1:]):
                raise AxBenchBaseError(
                    "Clustered decoder-only Stage 4 requires one exact loaded "
                    "checkpoint pin shared by every active method"
                )
            pin_payload = pins[0]
            refs[name]["step"] = pin_payload["step"]
            refs[name]["loaded_checkpoint"] = pin_payload
    return refs


def _cluster_plan_payload(
    plan: Stage4MethodPlan,
    eligible: list[Concept],
) -> dict[str, Any]:
    """Return the wire-format identity for one Stage 4 method plan."""

    method = plan.method
    return {
        "method": method.name,
        "mode": method.mode.value,
        "baseline": method.baseline.value,
        "target": plan.target_name,
        "requires_negative_sampling": method.requires_negative_sampling,
        "partition_modifier": (
            None
            if method.partition_modifier_name is None
            else method.partition_modifier_name.value
        ),
        "coefficients": plan.coefficients,
        "seed_count": len(plan.seeds_for_mode),
        "eligible_concept_ids": [concept.concept_id for concept in eligible],
    }


def _preflight_cluster_plan(
    cfg: AxBenchBaseConfig,
    *,
    output_paths: paths.AxBenchBasePaths,
    plan: Stage4MethodPlan,
    eligible: list[Concept],
    cluster_contract: ClusterFeatureContract,
    artifact_hashes: dict[str, str],
    scalar_concepts: set[int],
) -> str | None:
    """Validate and hash one clustered Stage 4 plan's artifact inputs."""

    method = plan.method
    if method.baseline in (
        BaselineKind.DIFFMEAN,
        BaselineKind.LINEAR_PROBE,
    ):
        corpus_sha256 = plan.cluster_source_corpus_sha256
        if corpus_sha256 is None:
            raise AxBenchBaseError(
                "Clustered scalar Stage 4 plan lacks frozen source-corpus identity"
            )
        path_fn = (
            output_paths.diffmean_direction
            if method.baseline is BaselineKind.DIFFMEAN
            else output_paths.linear_probe_direction
        )
        for concept in eligible:
            scalar_concepts.add(concept.feature_index)
            direction_path = path_fn(
                plan.target_name,
                method.mode,
                concept.concept_id,
            )
            _load_scalar_direction_artifact(
                direction_path,
                method=method,
                concept=concept,
                target_model_name=plan.target_name,
                layer=cfg.layer_hooks.diffmean_and_probe_layer,
                output_paths=output_paths,
                cluster_contract=cluster_contract,
                source_corpus_sha256=corpus_sha256,
            )
            artifact_hashes[
                f"scalar:{method.name}:concept:{concept.concept_id}"
            ] = _stable_cluster_artifact_sha256(
                direction_path,
                surface=(
                    f"{method.name} direction for concept "
                    f"{concept.concept_id}"
                ),
            )
        return corpus_sha256
    if method.baseline is BaselineKind.OURS:
        for concept in eligible:
            _load_top_k_concept_fids(
                plan.concept_features_path,
                concept.concept_id,
                expected_tokenizer_id=cfg.extraction_tokenizer_id,
                cluster_contract=cluster_contract,
                # Required under a cluster contract: the loader binds the
                # artifact's canonical_feature_id row to the representative
                # actually being steered, so a fingerprint-correct file with
                # swapped rows cannot silently drive a different bag. Mirrors
                # the dispatch call site in intervention_dispatch._dispatch_ours.
                expected_representative_id=concept.feature_index,
            )
        artifact_key = f"concept_features:{method.mode.value}"
        if artifact_key not in artifact_hashes:
            artifact_hashes[artifact_key] = _stable_cluster_artifact_sha256(
                plan.concept_features_path,
                surface=f"{method.mode.value} concept features",
            )
        return None
    if method.baseline is not BaselineKind.CONTROL:
        raise AxBenchBaseError(
            "Clustered AxBench-Base Stage 4 reached an unsupported method "
            f"family during artifact preflight: {method.name}"
        )
    return None


def _preflight_scalar_direction_plans(
    cfg: AxBenchBaseConfig,
    *,
    output_paths: paths.AxBenchBasePaths,
    plans: list[Stage4MethodPlan],
    eligible_by_mode: dict[SteeringMode, list[Concept]],
    cluster_contract: ClusterFeatureContract | None,
) -> None:
    """Validate every active scalar direction before Stage 4 can write.

    The direct dispatch loader repeats the same check as defense in depth.  This
    up-front pass prevents a late concept/target/layer mismatch from appearing
    after earlier methods have already generated tokens or appended Parquet
    batches.
    """

    for plan in plans:
        method = plan.method
        if method.baseline not in (
            BaselineKind.DIFFMEAN,
            BaselineKind.LINEAR_PROBE,
        ):
            continue
        path_fn = (
            output_paths.diffmean_direction
            if method.baseline is BaselineKind.DIFFMEAN
            else output_paths.linear_probe_direction
        )
        for concept in eligible_by_mode[method.mode]:
            _load_scalar_direction_artifact(
                path_fn(
                    plan.target_name,
                    method.mode,
                    concept.concept_id,
                ),
                method=method,
                concept=concept,
                target_model_name=plan.target_name,
                layer=cfg.layer_hooks.diffmean_and_probe_layer,
                output_paths=output_paths,
                cluster_contract=cluster_contract,
                source_corpus_sha256=plan.cluster_source_corpus_sha256,
            )


def _cluster_generation_inputs(
    cfg: AxBenchBaseConfig,
    *,
    cluster_contract: ClusterFeatureContract,
    concepts: list[Concept],
    plans: list[Stage4MethodPlan],
    eligible_by_mode: dict[SteeringMode, list[Concept]],
    seed_pool_fingerprint: str | None,
    consumed_eval_seeds: list[SeedRow],
    steering_scale_identity: str | None,
) -> dict[str, Any]:
    """Validate and fingerprint every clustered determinant before writes."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("Clustered Stage 4 identity requires cfg.run_tag")
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    concepts_sha256, bound_seed_fingerprint = _bind_cluster_generation_sources(
        cfg,
        concepts=concepts,
        consumed_eval_seeds=consumed_eval_seeds,
    )
    if seed_pool_fingerprint is not None and seed_pool_fingerprint != bound_seed_fingerprint:
        raise AxBenchBaseError("Clustered Stage 4 seed-pool fingerprint changed before binding")
    seed_pool_fingerprint = bound_seed_fingerprint
    artifact_hashes: dict[str, str] = {
        "concepts.json": concepts_sha256,
        "stage4_contexts_manifest.json": _stable_cluster_artifact_sha256(
            output_paths.stage4_contexts_manifest(),
            surface="Stage 3e manifest",
        ),
    }
    corpus_hashes: dict[str, str] = {}
    cache_hashes: dict[str, str] = {}
    plan_payloads: list[dict[str, Any]] = []
    scalar_concepts: set[int] = set()
    target_metadata: dict[str, dict[str, Any]] = {}

    for plan in plans:
        method = plan.method
        eligible = eligible_by_mode[method.mode]
        plan_payloads.append(_cluster_plan_payload(plan, eligible))
        if plan.target_name not in target_metadata:
            meta = plan.target_meta
            target_metadata[plan.target_name] = {
                "num_buckets": meta.num_buckets,
                "n_sae_features": meta.n_sae_features,
                "negative_selection": meta.negative_selection,
                "feature_selection": meta.feature_selection,
                "feature_mask": meta.feature_mask,
                "sae_layer": meta.sae_layer,
                "segment_size": meta.segment_size,
            }
            if plan.target_name == "encdec":
                cache_hashes[plan.target_name] = _context_cache_manifest_sha256(
                    plan.bundle
                )

        corpus_sha256 = _preflight_cluster_plan(
            cfg,
            output_paths=output_paths,
            plan=plan,
            eligible=eligible,
            cluster_contract=cluster_contract,
            artifact_hashes=artifact_hashes,
            scalar_concepts=scalar_concepts,
        )
        if corpus_sha256 is not None:
            corpus_hashes[method.mode.value] = corpus_sha256

    if scalar_concepts and steering_scale_identity is None:
        raise AxBenchBaseError(
            "Clustered scalar Stage 4 requires a direction-scale identity"
        )

    return {
        "schema_version": 1,
        "feature_contract_fingerprint": (
            cluster_contract.feature_contract_fingerprint
        ),
        "generation": asdict(cfg.generation),
        "layer_hooks": asdict(cfg.layer_hooks),
        "seed_pool_fingerprint": seed_pool_fingerprint,
        "concepts": [asdict(concept) for concept in concepts],
        "present_concept_ids": {
            mode.value: [
                concept.concept_id for concept in eligible_by_mode[mode]
            ]
            for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE)
        },
        "method_plans": plan_payloads,
        "checkpoint_refs": _checkpoint_refs_for_plans(cfg, plans),
        "target_metadata": target_metadata,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "source_corpus_sha256": dict(sorted(corpus_hashes.items())),
        "context_cache_manifest_sha256": dict(sorted(cache_hashes.items())),
        "steering_scale_identity": steering_scale_identity,
        "inactive_sampler_policy": "axbench_stage4_training_aligned_v1",
    }


def _collect_pending_generation_rows(
    plan: Stage4MethodPlan,
    *,
    concepts: list[Concept],
    already_done: dict[str, ExistingGenerationIdentity],
    run_tag: str,
    seed_pool_fingerprint: str,
    prompting_descriptions: dict[SteeringMode, dict[int, str]],
) -> list[PendingGenerationRow]:
    """Expand one method plan into the rows that still need work."""
    pending_rows: list[PendingGenerationRow] = []
    for concept in concepts:
        for eval_seed_idx, seed_row in enumerate(plan.seeds_for_mode):
            prompt_tokens = _prompt_tokens_for_row(
                plan, concept, seed_row, prompting_descriptions
            )
            exact_prompt_key = prompt_token_key(prompt_tokens)
            for coefficient in plan.coefficients:
                generation_key = build_generation_key(
                    method_name=plan.method.name,
                    concept_id=concept.concept_id,
                    eval_seed=eval_seed_idx,
                    coefficient=coefficient,
                    target_model_name=plan.target_name,
                )
                existing_identity = already_done.get(generation_key)
                if existing_identity is not None:
                    expected_identity = ExistingGenerationIdentity(
                        prefix_text=seed_row.prefix_text,
                        prompt_token_key=exact_prompt_key,
                        prompt_token_count=len(prompt_tokens),
                        seed_pool_fingerprint=seed_pool_fingerprint,
                    )
                    if existing_identity != expected_identity:
                        raise AxBenchBaseError(
                            "Stage 4 resume refused to skip an existing "
                            "generation row because its generation_key "
                            "matches the current loop but its exact resume "
                            "identity does not. This means the eval seed, "
                            "model-ready prompt tokens, or seed pool changed "
                            f"under the same run_tag={run_tag!r}. "
                            f"generation_key={generation_key!r}, "
                            f"existing_identity={existing_identity!r}, "
                            f"current_identity={expected_identity!r}. "
                            "Use a new run_tag or delete the stale "
                            "generations directory."
                        )
                    continue
                pending_rows.append(
                    PendingGenerationRow(
                        generation_key=generation_key,
                        concept=concept,
                        eval_seed=eval_seed_idx,
                        coefficient=coefficient,
                        seed_row=seed_row,
                        prompt_tokens=tuple(prompt_tokens),
                        prompt_token_key=exact_prompt_key,
                    ),
                )
    return pending_rows


def generate_all(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    targets: list[TargetBundle],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    apply_max_act_scale: bool,
    inactive_sampler: Callable[..., dict[int, int]] | None = None,
) -> Path:
    """Stage 4 entry point: generate the full method x concept x seed x coef matrix.

    Parameters
    ----------
    cfg:
        Pipeline config. ``cfg.run_tag`` must be set.
    concepts:
        Final filtered Stage 1 concept authority.
    targets:
        One :class:`TargetBundle` per target model (e.g., decoder-only
        and encoder-decoder). Each bundle carries the live model, its
        metadata, tokenizer, and optional exact context provider. An enabled
        method whose ``target_model`` field is not present in any bundle's
        ``name`` fails loudly before generation.
    eval_seeds_positive, eval_seeds_negative:
        The Stage 0b eval pools, one per mode.
    inactive_sampler:
        Optional callable forwarded to the ``ours`` negative-mode
        dispatch. Required only for methods with
        ``requires_negative_sampling=True``; ignored otherwise.

    Returns
    -------
    Path
        The invocation path contract's generations directory-dataset root.

    Raises
    ------
    AxBenchBaseError
        If ``cfg.run_tag`` is unset, if ``cfg.enabled_methods`` contains
        an unknown method name, or (from the dispatch helper) if any
        baseline artifact is missing / malformed.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling generate_all",
        )
    run_tag = cfg.run_tag
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    generations_dir = output_paths.generations_dir()
    cluster_contract = _cluster_contract_for_targets(targets)
    seed_pool_fingerprint: str | None = None
    initial_contexts_fingerprint: str | None = None
    generation_inputs: dict[str, Any] | None = None
    already_done: dict[str, ExistingGenerationIdentity] = {}
    clustered_batch_indices: tuple[int, ...] | None = None
    # Preserve the legacy call order literally. Clustered runs defer resume
    # acceptance until every artifact has been fully preflighted and hashed.
    if cluster_contract is None:
        seed_pool_fingerprint = load_seed_pool_fingerprint(output_paths)
        validate_cluster_generation_resume(
            cfg,
            None,
            contexts_fingerprint=None,
        )
        already_done = _load_existing_generation_metadata(
            generations_dir,
            expected_max_new_tokens=cfg.generation.max_new_tokens,
        )
        if already_done:
            logger.info(
                f"generate_all resume: {len(already_done)} rows already written "
                f"at {generations_dir}; skipping them",
            )

    method_plans = _build_stage4_method_plans(
        cfg,
        output_paths=output_paths,
        targets=targets,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
    )
    logger.info(
        f"generate_all dispatching {len(method_plans)} methods after filters "
        f"(enabled={cfg.enabled_methods}, disabled={cfg.disabled_methods})",
    )
    if cluster_contract is not None:
        initial_manifest = load_stage4_contexts_manifest(output_paths.stage4_contexts_manifest())
        if initial_manifest.get("feature_contract_fingerprint") != (
            cluster_contract.feature_contract_fingerprint
        ):
            raise AxBenchBaseError("Stage 3e contexts do not match the clustered Stage 4 target")
        initial_contexts_fingerprint = str(initial_manifest["contexts_fingerprint"])
    eligible_by_mode = _authoritative_concepts_by_mode(cfg, concepts)
    direction_requests = (
        _derive_stage4_direction_scale_requests(
            method_plans,
            eligible_by_mode=eligible_by_mode,
            output_paths=output_paths,
        )
        if apply_max_act_scale
        else []
    )
    # Snapshot the exact direction-owned values before any durable generation
    # row is accepted. The marker and per-consumer reopen comparison bind every
    # resumed scalar row to this artifact/value set.
    direction_preflight = (
        preflight_direction_scales(
            direction_requests,
            expected_layer=cfg.layer_hooks.diffmean_and_probe_layer,
            stage_label="Stage 4",
            error_cls=AxBenchBaseError,
        )
        if direction_requests
        else None
    )
    # Loading/identity validation is orthogonal to max-activation scaling and
    # therefore runs in both modes. The scalar consumer repeats it at load time.
    _preflight_scalar_direction_plans(
        cfg,
        output_paths=output_paths,
        plans=method_plans,
        eligible_by_mode=eligible_by_mode,
        cluster_contract=cluster_contract,
    )
    prompting_descriptions = _required_prompting_descriptions(cfg, method_plans)
    _preflight_stage4_prompt_lengths(
        cfg,
        plans=method_plans,
        eligible_by_mode=eligible_by_mode,
        prompting_descriptions=prompting_descriptions,
    )
    validated_contexts_fingerprint = _validate_stage4_context_cache(
        cfg,
        plans=method_plans,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
    )
    if cluster_contract is not None and (
        validated_contexts_fingerprint is not None
        and validated_contexts_fingerprint != initial_contexts_fingerprint
    ):
        raise AxBenchBaseError(
            "Stage 4 context identity changed during clustered preflight"
        )
    if cluster_contract is not None:
        generation_inputs = _cluster_generation_inputs(
            cfg,
            cluster_contract=cluster_contract,
            concepts=concepts,
            plans=method_plans,
            eligible_by_mode=eligible_by_mode,
            seed_pool_fingerprint=seed_pool_fingerprint,
            consumed_eval_seeds=eval_seeds_positive,
            steering_scale_identity=(
                None if direction_preflight is None else direction_preflight.identity
            ),
        )
        bound_seed_fingerprint = generation_inputs.get("seed_pool_fingerprint")
        if not isinstance(bound_seed_fingerprint, str):
            raise AxBenchBaseError(
                "Clustered Stage 4 generation inputs lack a bound seed fingerprint"
            )
        seed_pool_fingerprint = bound_seed_fingerprint
        validate_cluster_generation_resume(
            cfg,
            cluster_contract,
            contexts_fingerprint=initial_contexts_fingerprint,
            generation_inputs=generation_inputs,
        )
        batches_before_resume_read = tuple(
            _scan_batch_indices(generations_dir)
        )
        already_done = _load_existing_generation_metadata(
            generations_dir,
            expected_max_new_tokens=cfg.generation.max_new_tokens,
        )
        clustered_batch_indices = tuple(_scan_batch_indices(generations_dir))
        if clustered_batch_indices != batches_before_resume_read:
            raise AxBenchBaseError(
                "Clustered Stage 4 generation batches changed while resume "
                "metadata was being loaded; use a fresh run tag"
            )
        if already_done:
            logger.info(
                f"generate_all resume: {len(already_done)} rows already written "
                f"at {generations_dir}; skipping them",
            )
    if cluster_contract is not None:
        commit_cluster_generation_marker(
            cfg,
            cluster_contract,
            contexts_fingerprint=initial_contexts_fingerprint,
            generation_inputs=generation_inputs,
        )
    assert seed_pool_fingerprint is not None

    n_rows_written = 0
    n_rows_skipped = 0

    stage_started_at = time.monotonic()
    with ParquetAppendWriter(
        out_dir=generations_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=GENERATIONS_WRITE_BATCH_SIZE,
        immutable_batches=cluster_contract is not None,
        expected_existing_indices=clustered_batch_indices,
    ) as writer:
        if direction_preflight is not None:
            scaled_method_names = {
                plan.method.name
                for plan in method_plans
                if plan.method.baseline
                in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
            }
            resume_active = any(
                key.split("|", 1)[0] in scaled_method_names for key in already_done
            )
            enforce_steering_scale_identity(
                generations_dir / STEERING_SCALE_MARKER_NAME,
                identity=direction_preflight.identity,
                resume_active=resume_active,
            )
        for method_idx, plan in enumerate(method_plans):
            method = plan.method
            eligible_concepts = eligible_by_mode[plan.method.mode]
            method_expected_rows = (
                len(eligible_concepts) * len(plan.seeds_for_mode) * len(plan.coefficients)
            )
            method_started_at = time.monotonic()
            method_rows_written = 0
            pending_rows = _collect_pending_generation_rows(
                plan,
                concepts=eligible_concepts,
                already_done=already_done,
                run_tag=run_tag,
                seed_pool_fingerprint=seed_pool_fingerprint,
                prompting_descriptions=prompting_descriptions,
            )
            method_rows_skipped = method_expected_rows - len(pending_rows)
            n_rows_skipped += method_rows_skipped
            last_method_heartbeat = method_started_at
            logger.info(
                f"[Stage 4] Method {method_idx + 1}/{len(method_plans)}: "
                f"{method.name!r} "
                f"(target={plan.target_name}, mode={method.mode}, "
                f"concepts={len(eligible_concepts)}, seeds={len(plan.seeds_for_mode)}, "
                f"coefficients={len(plan.coefficients)}, "
                f"expected_rows={method_expected_rows})"
            )

            for pending_row in pending_rows:
                concept = pending_row.concept
                eval_seed_idx = pending_row.eval_seed
                coefficient = pending_row.coefficient
                seed_row = pending_row.seed_row
                generation_key = pending_row.generation_key

                with profile_region(
                    "stage4.run_intervention",
                    method=method.name,
                    concept_id=concept.concept_id,
                    target=plan.target_name,
                ):
                    expected_max_activation = None
                    if (
                        apply_max_act_scale
                        and method.baseline
                        in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE)
                    ):
                        assert direction_preflight is not None
                        expected_max_activation = direction_preflight.values[
                            (
                                method.baseline.value,
                                plan.target_name,
                                method.mode.value,
                                concept.concept_id,
                            )
                        ]
                    result = run_intervention(
                        method=method,
                        concept=concept,
                        eval_seed=eval_seed_idx,
                        coefficient=coefficient,
                        prefix_tokens=list(pending_row.prompt_tokens),
                        target=plan.bundle,
                        cfg=cfg,
                        concept_features_path=plan.concept_features_path,
                        apply_max_act_scale=apply_max_act_scale,
                        expected_max_activation=expected_max_activation,
                        inactive_sampler=inactive_sampler,
                        cluster_source_corpus_sha256=(
                            plan.cluster_source_corpus_sha256
                        ),
                    )
                with profile_region(
                    "stage4.writer_append",
                    method=method.name,
                    concept_id=concept.concept_id,
                ):
                    writer.append(
                        {
                            "generation_key": generation_key,
                            "concept_id": concept.concept_id,
                            "method_name": method.name,
                            "mode": method.mode,
                            "eval_seed": eval_seed_idx,
                            "coefficient": str(coefficient),
                            "generation_text": result.text,
                            "target_model": plan.target_name,
                            "prefix_text": seed_row.prefix_text,
                            "prompt_token_key": pending_row.prompt_token_key,
                            "prompt_token_count": len(pending_row.prompt_tokens),
                            "seed_pool_fingerprint": seed_pool_fingerprint,
                            "token_count": result.token_count,
                            "stop_reason": result.stop_reason,
                        },
                    )
                n_rows_written += 1
                method_rows_written += 1
                record_counter("stage4.rows_written", 1, method=method.name)

                now = time.monotonic()
                if (now - last_method_heartbeat) >= 30.0:
                    elapsed = now - method_started_at
                    processed = method_rows_written + method_rows_skipped
                    pct = (
                        100.0 * processed / method_expected_rows
                        if method_expected_rows else 0.0
                    )
                    rate = method_rows_written / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, method_expected_rows - processed)
                    eta = remaining / rate if rate > 0 else float("inf")
                    logger.info(
                        f"[Stage 4] {method.name!r}: "
                        f"{processed}/{method_expected_rows} "
                        f"({pct:.0f}% — written={method_rows_written}, "
                        f"skipped={method_rows_skipped}, "
                        f"rate={rate:.2f} rows/s, "
                        f"eta≈{eta:.0f}s)"
                    )
                    last_method_heartbeat = now

            method_elapsed = time.monotonic() - method_started_at
            logger.info(
                f"[Stage 4] Method {method.name!r} done: "
                f"wrote {method_rows_written} rows, "
                f"skipped {method_rows_skipped} (already cached), "
                f"elapsed {method_elapsed:.0f}s"
            )
            record_counter(
                "stage4.method_rows_written",
                method_rows_written,
                method=method.name,
            )
            record_counter(
                "stage4.method_rows_skipped",
                method_rows_skipped,
                method=method.name,
            )

    stage_elapsed = time.monotonic() - stage_started_at
    logger.info(
        f"[Stage 4] generate_all wrote {n_rows_written} rows to "
        f"{generations_dir} ({n_rows_skipped} skipped as already done, "
        f"stage elapsed {stage_elapsed:.0f}s)"
    )
    record_counter("stage4.total_rows_written", n_rows_written)
    record_counter("stage4.total_rows_skipped", n_rows_skipped)
    return generations_dir
