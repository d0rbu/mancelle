"""Stage 4 ``_run_intervention`` dispatch (Task 37).

Given a :class:`~steering.eval.axbench_base.method_registry.Method` plus a
concept, an ``eval_seed``, and a coefficient value drawn from the method's
grid, this module is responsible for actually running the intervention:

    1. Load the baseline artifact (DiffMean / LinearProbe direction or
       LoRA adapter). "Ours" methods load the concept features file instead;
       prompting receives its fully constructed prompt from Stage 4.
    2. Install any side effects on the target model:
       * DiffMean / LinearProbe → a residual-stream forward hook via
         :func:`~steering.eval.axbench_base.baselines.hooks.register_residual_hook`.
       * LoRA → :func:`~steering.eval.axbench_base.baselines.lora.loader.attach_lora_from_disk`.
       * Prompting → no model side effect; generation uses the exact
         model-ready prompt supplied by the caller.
       * Ours → build a ``feature_conditions`` dict via
         :func:`~steering.eval.axbench_base.encoder_input.build_mode_aware_encoder_input`.
    3. Call :func:`~steering.eval.axbench_base.generation_helpers._generate_with_nanochat_tokens`.
    4. Unconditionally remove the hook / detach the adapter in a
       ``finally`` block so the model is left pristine for the next
       method.

Negative mode
-------------
Only the DiffMean / LinearProbe path treats ``method.mode == "negative"``
specially — it negates the incoming ``coefficient`` (subtract the
direction instead of adding). The LoRA and Prompting paths rely on the
fact that negative-mode artifacts were trained on a different corpus
(Stage 1c), so the same machinery applied with ``method.mode`` forwarded
to the path resolver / prompting text is sufficient. The ``ours`` path
also forwards ``method.mode`` verbatim to
:func:`~steering.eval.axbench_base.encoder_input.build_mode_aware_encoder_input`
which itself dispatches on that field.

Sampling-seed determinism
-------------------------
Each ``(method, concept, eval_seed)`` triple derives a deterministic but
method-specific sampling seed via a SHA256 of the string
``"{method_name}|{concept_id}|{eval_seed}|{master_seed}"``. This is a
local helper so each method samples from a distinct deterministic stream
and each generation's stochasticity is uncorrelated.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch as t
import torch.nn as nn
from safetensors import safe_open

from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
)
from steering.eval.axbench_base.baselines.hooks import (
    register_residual_hook,
)
from steering.eval.axbench_base.baselines.lora.loader import attach_lora_from_disk
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    expected_artifact_kind,
    load_direction_artifact,
)
from steering.eval.axbench_base.direction_scale import (
    normalize_direction,
    require_direction_max_activation,
)
from steering.eval.axbench_base.encoder_input import (
    build_mode_aware_encoder_input,
)
from steering.eval.axbench_base.generation_helpers import (
    GenerationResult,
    _generate_with_nanochat_tokens,
)
from steering.eval.axbench_base.method_registry import BaselineKind, Method
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    stream_hash_file_stable,
)

__all__ = ("run_intervention",)


# ---------------------------------------------------------------------------
# Sampling seed derivation (method-specific per-call)
# ---------------------------------------------------------------------------


def _sampling_seed(
    method_name: str,
    *,
    concept_id: int,
    eval_seed: int,
    master_seed: int,
) -> int:
    """Deterministic per-call sampling seed for Stage 4 interventions.

    SHA256 of ``"{method_name}|{concept_id}|{eval_seed}|{master_seed}"``
    truncated to 8 little-endian bytes and modded to an int31. The
    int31 envelope is wide enough for every sampling RNG we use
    downstream and safe for JSON / parquet round-trip without overflow.
    """
    digest = hashlib.sha256(
        f"{method_name}|{concept_id}|{eval_seed}|{master_seed}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (1 << 31)


# ---------------------------------------------------------------------------
# Top-K concept feature lookup
# ---------------------------------------------------------------------------


_VALID_CONCEPT_FEATURES_MODES: frozenset[str] = frozenset(
    {"canonical", "delta_ranked"},
)


def _load_top_k_concept_fids(
    concept_features_path: Path,
    concept_id: int,
    *,
    expected_tokenizer_id: str | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    expected_representative_id: int | None = None,
) -> list[int]:
    """Load the top-K SAE feature ids for one concept from a features file.

    The file is the safetensors artifact written by the Stage 2 canonical or
    delta-ranked producer.
    Row ``concept_id`` of its ``feature_ids`` tensor holds that
    concept's ordered top-K list; ``-1`` is a sentinel for unused slots and is filtered
    out before the list is returned.

    Validates the safetensors ``metadata={"mode": ...}`` annotation
    written by Stage 2 — the value must be ``"canonical"`` or
    ``"delta_ranked"``. A wrong mode would silently load the wrong
    feature rankings (e.g., a canonical-mode file generated against
    one SAE alongside a delta_ranked-mode dispatch). Defense surfaced
    by Phase 2 cross-cutting Stream 1 (data-flow audit).
    """
    with safe_open(str(concept_features_path), framework="pt") as handle:
        with profile_region(
            "stage4.load_concept_features_metadata",
            concept_id=concept_id,
            path=concept_features_path,
        ):
            metadata = handle.metadata() or {}
            mode = metadata.get("mode")
            if mode not in _VALID_CONCEPT_FEATURES_MODES:
                raise AxBenchBaseError(
                    f"concept_features safetensors at {concept_features_path} "
                    f"has unexpected metadata mode={mode!r}; expected one of "
                    f"{sorted(_VALID_CONCEPT_FEATURES_MODES)}. Did Stage 2 "
                    f"finish cleanly? Re-run Stage 2 with a fresh run_tag."
                )
            if mode == "delta_ranked" and expected_tokenizer_id is not None:
                actual_tokenizer_id = metadata.get(EXTRACTION_TOKENIZER_METADATA_KEY)
                if actual_tokenizer_id is None:
                    raise AxBenchBaseError(
                        f"concept_features safetensors at {concept_features_path} "
                        f"is delta_ranked but missing "
                        f"{EXTRACTION_TOKENIZER_METADATA_KEY!r} metadata. "
                        "Re-run Stage 2 with the current extraction tokenizer."
                    )
                if actual_tokenizer_id != expected_tokenizer_id:
                    raise AxBenchBaseError(
                        f"concept_features safetensors at {concept_features_path} "
                        f"was created with extraction tokenizer "
                        f"{actual_tokenizer_id!r}, but the current config uses "
                        f"{expected_tokenizer_id!r}. Re-run Stage 2 with the "
                        "current extraction tokenizer."
                    )
            artifact_fingerprint = metadata.get("feature_contract_fingerprint")
            if cluster_contract is None:
                if artifact_fingerprint is not None:
                    raise AxBenchBaseError(
                        "Unclustered Stage 4 rejects cluster-stamped "
                        f"concept features at {concept_features_path}"
                    )
            elif artifact_fingerprint != (cluster_contract.feature_contract_fingerprint):
                raise AxBenchBaseError(
                    f"concept features at {concept_features_path} do not match "
                    "the clustered target feature contract"
                )
            if cluster_contract is not None:
                if expected_representative_id is None:
                    raise AxBenchBaseError(
                        "Clustered Stage 4 requires the expected "
                        "Concept.feature_index representative"
                    )
                if "canonical_feature_id" not in handle.keys():
                    raise AxBenchBaseError(
                        f"clustered concept features at {concept_features_path} "
                        "are missing canonical_feature_id"
                    )
        with profile_region(
            "stage4.load_concept_features_tensor",
            concept_id=concept_id,
            path=concept_features_path,
        ):
            feature_ids = handle.get_tensor("feature_ids")
            canonical_feature_ids = (
                handle.get_tensor("canonical_feature_id") if cluster_contract is not None else None
            )
    if concept_id < 0 or concept_id >= int(feature_ids.shape[0]):
        raise AxBenchBaseError(
            f"concept_id={concept_id} is out of bounds for "
            f"{concept_features_path} feature_ids rows={feature_ids.shape[0]}"
        )
    if canonical_feature_ids is not None:
        if canonical_feature_ids.ndim != 1 or int(canonical_feature_ids.shape[0]) != int(
            feature_ids.shape[0]
        ):
            raise AxBenchBaseError(
                f"clustered concept features at {concept_features_path} have "
                "canonical_feature_id shape "
                f"{tuple(canonical_feature_ids.shape)!r}, expected "
                f"({int(feature_ids.shape[0])},)"
            )
        actual_representative_id = int(canonical_feature_ids[concept_id].item())
        if actual_representative_id != expected_representative_id:
            raise AxBenchBaseError(
                f"clustered concept features canonical_feature_id row "
                f"{concept_id}={actual_representative_id} does not match "
                f"Concept.feature_index={expected_representative_id}"
            )
    fids_tensor = feature_ids[concept_id]
    raw_fids = [int(fid) for fid in fids_tensor.tolist()]
    invalid_negative_ids = sorted({fid for fid in raw_fids if fid < -1})
    if invalid_negative_ids:
        raise AxBenchBaseError(
            f"concept_features safetensors at {concept_features_path} row "
            f"concept_id={concept_id} contains invalid negative feature ids "
            f"{invalid_negative_ids}; only -1 is a valid unused-slot sentinel."
        )
    result = [fid for fid in raw_fids if fid != -1]
    if cluster_contract is not None:
        cluster_contract.consolidation.assert_representatives_only(
            t.tensor(result, dtype=t.int64),
            surface=f"Stage 4 concept feature row {concept_id}",
        )
        if result != [expected_representative_id]:
            raise AxBenchBaseError(
                f"clustered canonical concept features at "
                f"{concept_features_path} row concept_id={concept_id} are "
                f"{result}, expected exactly [{expected_representative_id}]"
            )
    return result


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


def run_intervention(
    *,
    method: Method,
    concept: Concept,
    eval_seed: int,
    coefficient: Any,
    prefix_tokens: list[int],
    target: TargetBundle,
    cfg: AxBenchBaseConfig,
    concept_features_path: Path,
    apply_max_act_scale: bool,
    expected_max_activation: float | None = None,
    inactive_sampler: Callable[..., dict[int, int]] | None = None,
    cluster_source_corpus_sha256: str | None = None,
) -> GenerationResult:
    """Dispatch to the right intervention implementation for ``method``.

    Args:
        method: The Stage 4 method registry entry (baseline, mode,
            target model, etc.).
        concept: The concept being evaluated.
        eval_seed: The outer evaluation-seed index (a deterministic
            per-call sampling seed is derived from
            ``(method, concept_id, eval_seed, master_seed)``).
        coefficient: Depends on the baseline: a scalar float for
            DiffMean / LinearProbe, an int bucket for ``ours``
            SAE-feature methods, or ``None`` for LoRA / Prompting.
        prefix_tokens: The fully constructed model-ready nanochat prompt.
            Stage 4 owns its leading BOS and, for prompting methods, has
            already inserted the raw prompting prefix before the seed content.
        target: A :class:`TargetBundle` grouping the live target model,
            its metadata, tokenizer, and optional context provider. The
            ``target.model`` is the ``nn.Module`` whose forward path
            the intervention temporarily mutates (hook install / LoRA
            attach). The ``target.name`` is ``"decoder_only"`` or
            ``"encdec"`` -- a path segment saved with every baseline
            artifact at training time.
        cfg: The run-level config. Only a few sub-fields are read
            directly: ``generation.*``, ``layer_hooks.*``, ``lora``,
            ``run_tag``.
        concept_features_path: Path to the safetensors artifact written
            by Stage 2. Only read by the "ours" path (and ignored by
            DiffMean / LinearProbe / LoRA / Prompting).
        inactive_sampler: Optional callable forwarded to
            :func:`build_mode_aware_encoder_input`. Required for
            negative-mode ours methods that mutate the inactive
            partition; ignored for every other baseline.

    Returns:
        A :class:`GenerationResult` -- the newly generated text, token
        count, and stop reason. The prefix tokens are NOT included.

    Raises:
        AxBenchBaseError: on an unknown ``method.baseline`` or a
            missing prompting description for ``concept``.
    """
    target_model = target.model
    target_model_name = target.name
    target_model_meta = target.meta
    tokenizer = target.tokenizer
    device = target.meta.device
    context_provider = target.context_provider
    pre_forward_guard = (
        target.ensure_ready_for_forward if target.cluster_contract is not None else None
    )
    sampling_seed = _sampling_seed(
        method.name,
        concept_id=concept.concept_id,
        eval_seed=eval_seed,
        master_seed=cfg.generation.master_seed,
    )

    if method.baseline in (BaselineKind.DIFFMEAN, BaselineKind.LINEAR_PROBE):
        return _dispatch_scalar_direction(
            method=method,
            concept=concept,
            coefficient=coefficient,
            prefix_tokens=prefix_tokens,
            model=target_model,
            tokenizer=tokenizer,
            device=device,
            target_model_name=target_model_name,
            cfg=cfg,
            context_provider=context_provider,
            sampling_seed=sampling_seed,
            apply_max_act_scale=apply_max_act_scale,
            expected_max_activation=expected_max_activation,
            cluster_contract=target.cluster_contract,
            cluster_source_corpus_sha256=cluster_source_corpus_sha256,
            pre_forward_guard=pre_forward_guard,
        )
    if method.baseline == BaselineKind.LORA:
        return _dispatch_lora(
            method=method,
            concept=concept,
            prefix_tokens=prefix_tokens,
            model=target_model,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg,
            context_provider=context_provider,
            sampling_seed=sampling_seed,
            pre_forward_guard=pre_forward_guard,
        )
    if method.baseline == BaselineKind.PROMPTING:
        return _dispatch_prompting(
            prefix_tokens=prefix_tokens,
            model=target_model,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg,
            context_provider=context_provider,
            sampling_seed=sampling_seed,
            pre_forward_guard=pre_forward_guard,
        )
    if method.baseline == BaselineKind.OURS:
        return _dispatch_ours(
            method=method,
            concept=concept,
            coefficient=coefficient,
            prefix_tokens=prefix_tokens,
            model=target_model,
            tokenizer=tokenizer,
            device=device,
            target_model_meta=target_model_meta,
            cfg=cfg,
            context_provider=context_provider,
            concept_features_path=concept_features_path,
            inactive_sampler=inactive_sampler,
            sampling_seed=sampling_seed,
            cluster_contract=target.cluster_contract,
            pre_forward_guard=pre_forward_guard,
        )
    if method.baseline == BaselineKind.CONTROL:
        return _dispatch_control(
            prefix_tokens=prefix_tokens,
            model=target_model,
            tokenizer=tokenizer,
            device=device,
            cfg=cfg,
            context_provider=context_provider,
            sampling_seed=sampling_seed,
            pre_forward_guard=pre_forward_guard,
        )
    raise AxBenchBaseError(f"Unknown method baseline: {method.baseline!r}")


# ---------------------------------------------------------------------------
# Per-baseline dispatch helpers
# ---------------------------------------------------------------------------


def _dispatch_control(
    *,
    prefix_tokens: list[int],
    model: nn.Module,
    tokenizer: Any,
    device: str,
    cfg: AxBenchBaseConfig,
    context_provider: ContextProvider | None,
    sampling_seed: int,
    pre_forward_guard: Callable[[], None] | None = None,
) -> GenerationResult:
    """CONTROL: NO steering. No model hook / adapter / encoder delta is installed
    (``encoder_input=None``). Decoder-only ⇒ the unmodified base model; encdec ⇒
    prompt/context-conditioned (via ``context_provider``) with NO concept
    injection. This is the no-steering reference baseline; it reuses the same
    generation primitive as every other dispatch path.
    """
    if pre_forward_guard is not None:
        pre_forward_guard()
    return _generate_with_nanochat_tokens(
        model=model,
        tokenizer=tokenizer,
        device=device,
        nanochat_tokens=prefix_tokens,
        sampling_seed=sampling_seed,
        gen_cfg=cfg.generation,
        encoder_input=None,
        context_provider=context_provider,
    )


def _dispatch_scalar_direction(
    *,
    method: Method,
    concept: Concept,
    coefficient: Any,
    prefix_tokens: list[int],
    model: nn.Module,
    tokenizer: Any,
    device: str,
    target_model_name: str,
    cfg: AxBenchBaseConfig,
    context_provider: ContextProvider | None,
    sampling_seed: int,
    apply_max_act_scale: bool,
    expected_max_activation: float | None = None,
    cluster_contract: ClusterFeatureContract | None,
    cluster_source_corpus_sha256: str | None = None,
    pre_forward_guard: Callable[[], None] | None = None,
) -> GenerationResult:
    """DiffMean / LinearProbe: load direction, install hook, generate.

    Negates ``coefficient`` for negative mode (subtract the direction), then
    in scaled mode normalizes the vector and multiplies by the direction
    artifact's own ``max_activation`` (#1228). The hook is
    unconditionally removed in a ``finally`` block so the model is pristine
    on return even if generation raised.
    """
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    path_fn = (
        output_paths.diffmean_direction
        if method.baseline == BaselineKind.DIFFMEAN
        else output_paths.linear_probe_direction
    )
    direction_path = path_fn(
        target_model_name,
        method.mode,
        concept.concept_id,
    )
    with profile_region(
        "stage4.load_scalar_direction",
        method=method.name,
        concept_id=concept.concept_id,
    ):
        direction, metadata = _load_scalar_direction_artifact(
            direction_path,
            method=method,
            concept=concept,
            target_model_name=target_model_name,
            layer=cfg.layer_hooks.diffmean_and_probe_layer,
            output_paths=output_paths,
            cluster_contract=cluster_contract,
            source_corpus_sha256=cluster_source_corpus_sha256,
        )

    effective_coefficient = float(coefficient)
    if method.mode == SteeringMode.NEGATIVE:
        effective_coefficient = -abs(effective_coefficient)
    if apply_max_act_scale:
        direction = normalize_direction(direction, error_cls=AxBenchBaseError)
        artifact_max_activation = require_direction_max_activation(
            metadata,
            direction=direction,
            expected_concept_id=concept.concept_id,
            expected_mode=method.mode.value,
            expected_target_model=target_model_name,
            expected_layer=int(cfg.layer_hooks.diffmean_and_probe_layer),
            surface=(
                f"stage-4 {method.baseline.value} direction "
                f"concept={concept.concept_id}"
            ),
            error_cls=AxBenchBaseError,
        )
        if (
            expected_max_activation is not None
            and artifact_max_activation != expected_max_activation
        ):
            artifact_key = (
                method.baseline.value,
                target_model_name,
                method.mode.value,
                concept.concept_id,
            )
            raise AxBenchBaseError(
                "stage-4 direction-scale mid-run regeneration hazard: the "
                "reopened direction artifact value does not match preflight for "
                f"artifact key {artifact_key!r}; expected "
                f"{expected_max_activation!r}, got {artifact_max_activation!r}. "
                "Stage 3 may have regenerated the unclustered artifact while "
                "generation was running; use a fresh run tag."
            )
        effective_coefficient *= artifact_max_activation

    with profile_region("stage4.register_residual_hook", method=method.name):
        handle = register_residual_hook(
            model=model,
            layer=cfg.layer_hooks.diffmean_and_probe_layer,
            direction=direction,
            coefficient=effective_coefficient,
        )
    try:
        if pre_forward_guard is not None:
            pre_forward_guard()
        return _generate_with_nanochat_tokens(
            model=model,
            tokenizer=tokenizer,
            device=device,
            nanochat_tokens=prefix_tokens,
            sampling_seed=sampling_seed,
            gen_cfg=cfg.generation,
            encoder_input=None,
            context_provider=context_provider,
        )
    finally:
        with profile_region("stage4.remove_residual_hook", method=method.name):
            handle.remove()


def _load_scalar_direction_artifact(
    direction_path: Path,
    *,
    method: Method,
    concept: Concept,
    target_model_name: str,
    layer: int,
    output_paths: paths.AxBenchBasePaths,
    cluster_contract: ClusterFeatureContract | None,
    source_corpus_sha256: str | None = None,
) -> tuple[t.Tensor, dict[str, str]]:
    """Load one scalar artifact through the shared fused identity contract."""

    family = (
        BaselineKind.DIFFMEAN.value
        if method.baseline is BaselineKind.DIFFMEAN
        else BaselineKind.LINEAR_PROBE.value
    )
    request = DirectionArtifactRequest(
        family=family,
        target_model=target_model_name,
        mode=method.mode.value,
        concept_id=concept.concept_id,
        layer=layer,
    )
    loaded = load_direction_artifact(
        direction_path,
        request=request,
        error_type=AxBenchBaseError,
    )
    metadata = loaded.metadata_dict()
    if cluster_contract is None:
        cluster_fields = {
            "feature_contract_fingerprint",
            "source_corpus_sha256",
        }
        if any(field in metadata for field in cluster_fields):
            raise AxBenchBaseError(
                f"Unclustered Stage 4 rejects cluster-stamped scalar artifact {direction_path}"
            )
        return loaded.direction, metadata

    corpus_sha256 = source_corpus_sha256
    if corpus_sha256 is None:
        corpus_path = (
            output_paths.synth_training_positive()
            if method.mode is SteeringMode.POSITIVE
            else output_paths.synth_training_negative()
        )
        try:
            corpus_sha256 = stream_hash_file_stable(corpus_path).sha256
        except Exception as exc:
            raise AxBenchBaseError(
                "Clustered Stage 4 could not verify the scalar artifact's "
                f"source corpus {corpus_path}"
            ) from exc
    if (
        not isinstance(corpus_sha256, str)
        or len(corpus_sha256) != 64
        or any(char not in "0123456789abcdef" for char in corpus_sha256)
    ):
        raise AxBenchBaseError(
            "Clustered Stage 4 source-corpus identity must be a lowercase SHA-256"
        )
    expected_metadata = {
        "feature_contract_fingerprint": (cluster_contract.feature_contract_fingerprint),
        "source_corpus_sha256": corpus_sha256,
        "artifact_kind": expected_artifact_kind(family),
    }
    mismatches = {
        field: {"expected": expected, "actual": metadata.get(field)}
        for field, expected in expected_metadata.items()
        if metadata.get(field) != expected
    }
    if mismatches:
        raise AxBenchBaseError(
            "Clustered Stage 4 scalar artifact identity mismatch at "
            f"{direction_path}: {mismatches}. Use a fresh run_tag."
        )
    return loaded.direction, metadata


def _dispatch_lora(
    *,
    method: Method,
    concept: Concept,
    prefix_tokens: list[int],
    model: nn.Module,
    tokenizer: Any,
    device: str,
    cfg: AxBenchBaseConfig,
    context_provider: ContextProvider | None,
    sampling_seed: int,
    pre_forward_guard: Callable[[], None] | None = None,
) -> GenerationResult:
    """LoRA: attach adapter, generate, detach.

    ``attach_lora_from_disk`` returns a ``(adapters, detach_fn)`` pair;
    we call ``detach_fn`` in a ``finally`` block. Idempotent on the
    inner side (see loader docstring).
    """
    adapter_path = paths.AxBenchBasePaths.from_config(cfg).lora_adapter(
        method.mode,
        concept.concept_id,
    )
    with profile_region(
        "stage4.attach_lora",
        method=method.name,
        concept_id=concept.concept_id,
    ):
        _, detach_fn = attach_lora_from_disk(
            model=model,
            adapter_path=adapter_path,
            lora_cfg=cfg.lora,
            layer=cfg.layer_hooks.diffmean_and_probe_layer,
        )
    try:
        if pre_forward_guard is not None:
            pre_forward_guard()
        return _generate_with_nanochat_tokens(
            model=model,
            tokenizer=tokenizer,
            device=device,
            nanochat_tokens=prefix_tokens,
            sampling_seed=sampling_seed,
            gen_cfg=cfg.generation,
            encoder_input=None,
            context_provider=context_provider,
        )
    finally:
        with profile_region("stage4.detach_lora", method=method.name):
            detach_fn()


def _dispatch_prompting(
    *,
    prefix_tokens: list[int],
    model: Any,
    tokenizer: Any,
    device: str,
    cfg: AxBenchBaseConfig,
    context_provider: ContextProvider | None,
    sampling_seed: int,
    pre_forward_guard: Callable[[], None] | None = None,
) -> GenerationResult:
    """Prompting: generate from an already constructed model-ready prompt."""

    if pre_forward_guard is not None:
        pre_forward_guard()
    return _generate_with_nanochat_tokens(
        model=model,
        tokenizer=tokenizer,
        device=device,
        nanochat_tokens=prefix_tokens,
        sampling_seed=sampling_seed,
        gen_cfg=cfg.generation,
        encoder_input=None,
        context_provider=context_provider,
    )


def _dispatch_ours(
    *,
    method: Method,
    concept: Concept,
    coefficient: Any,
    prefix_tokens: list[int],
    model: Any,
    tokenizer: Any,
    device: str,
    target_model_meta: TargetModelMetadata,
    cfg: AxBenchBaseConfig,
    context_provider: ContextProvider | None,
    concept_features_path: Path,
    inactive_sampler: Callable[..., dict[int, int]] | None,
    sampling_seed: int,
    cluster_contract: ClusterFeatureContract | None,
    pre_forward_guard: Callable[[], None] | None = None,
) -> GenerationResult:
    """``ours`` encoder-decoder method dispatch.

    Builds a partition modifier input via
    :func:`build_mode_aware_encoder_input` and forwards the merged
    ``feature_conditions`` dict to the generation helper.
    """
    top_k_concept_fids = _load_top_k_concept_fids(
        concept_features_path,
        concept.concept_id,
        expected_tokenizer_id=cfg.extraction_tokenizer_id,
        cluster_contract=cluster_contract,
        expected_representative_id=concept.feature_index,
    )
    record_counter(
        "stage4.top_k_concept_fids",
        len(top_k_concept_fids),
        concept_id=concept.concept_id,
        method=method.name,
    )
    # Under the "common path" documented in the module header, the
    # caller supplies a precomputed active partition via the
    # target-model features artifact. The dispatch itself has no way to
    # observe Stage 1 features yet, so start with an empty active
    # partition — the modifier will populate it from the top-K list
    # (extract_and_add / replace_random / replace_all) or leave it empty
    # (neg_ours_remove).
    active: dict[int, int] = {}
    inactive: dict[int, int] = {}

    rng = random.Random(sampling_seed)

    # ``coefficient`` for an ``ours`` method is an int bucket. Coerce
    # ``None`` is accepted for single-condition grids; modifiers that need
    # a bucket receive 0 in that case.
    effective_coefficient = int(coefficient) if coefficient is not None else 0

    with profile_region(
        "stage4.build_encoder_input",
        method=method.name,
        concept_id=concept.concept_id,
    ):
        encoder_input = build_mode_aware_encoder_input(
            method=method,
            active=active,
            inactive=inactive,
            top_k_concept_fids=top_k_concept_fids,
            coefficient=effective_coefficient,
            target_model_meta=target_model_meta,
            inactive_sampler=inactive_sampler,
            rng=rng,
        )
    if cluster_contract is not None:
        cluster_contract.consolidation.assert_representatives_only(
            t.tensor(sorted(encoder_input), dtype=t.int64),
            surface=f"Stage 4 {method.name} encoder input",
        )
    if pre_forward_guard is not None:
        pre_forward_guard()
    return _generate_with_nanochat_tokens(
        model=model,
        tokenizer=tokenizer,
        device=device,
        nanochat_tokens=prefix_tokens,
        sampling_seed=sampling_seed,
        gen_cfg=cfg.generation,
        encoder_input=encoder_input,
        context_provider=context_provider,
    )
