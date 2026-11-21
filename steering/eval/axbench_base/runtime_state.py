"""Lazy-loaded, refcount-managed runtime state for axbench_base stages.

The axbench_base pipeline has stages that need heavyweight runtime
resources -- a feature-extraction base model + SAE pair for feature
extraction, nanochat target models for generation, and an OpenRouter
judge client for scoring. These resources can't be reconstructed from ``cfg`` alone:
they need network downloads, disk IO, GPU memory, and OpenAPI
credentials.

:class:`RuntimeState` centralizes this state with three design
properties:

1. **Lazy loading**: each resource is built on first access and cached.
   Callers that don't touch a resource don't pay for it. This is how
   stage 0a can run without touching the base model, and stage 6 can
   run without loading any models at all.

2. **Tiered resources**: pure-data artifacts (concepts, seeds, tokenizers)
   are cheap, small, and stay resident once
   loaded. Heavy models (base feature-extraction model, SAE, target
   models) are expensive and participate in refcounted context managers
   that offload them to CPU when not in use. This keeps GPU peak memory
   bounded.

3. **Explicit teardown**: ``close()`` is idempotent and called by the
   CLI's outer ``try/finally``. Subsequent accessor calls raise a
   clear error rather than silently reloading a dead state. Each
   cache close is guarded in its own ``try/except`` block so a
   single faulty ``.close()`` cannot leak the other resources.

Thread safety
-------------
:class:`RuntimeState` is **NOT thread-safe**. The refcount fields
(``_base_model_refcount``, etc.) are plain integers without locks,
and lazy-loader sentinels use the "check then set" pattern that is
not atomic. All stage dispatch in this repository is single-threaded
(``_run`` iterates stages sequentially in the main thread), and every
access to a :class:`RuntimeState` instance must come from the same
thread.

Every accessor asserts ``threading.current_thread() is
threading.main_thread()`` and raises :class:`AxBenchBaseError` from a
non-main thread. If you add distributed or parallel stage execution in
the future, wrap every refcount mutation and every ``is None`` check in
a ``threading.Lock``, or move to an async context-var-based design.

Accessor surface
----------------
The dataclass exposes four categories of accessors:

* **Pure-data** -- :meth:`RuntimeState.concepts`,
  :meth:`RuntimeState.seeds_train`, :meth:`RuntimeState.seeds_eval`,
  :meth:`RuntimeState.nanochat_tokenizer`,
  :meth:`RuntimeState.extraction_tokenizer`. Lazy-loaded once, cached
  forever.
* **Heavy models** -- context managers
  :meth:`RuntimeState.base_model`, :meth:`RuntimeState.base_sae`,
  :meth:`RuntimeState.decoder_only_model`,
  :meth:`RuntimeState.encdec_model` each maintain a refcount and
  offload to CPU when the refcount drops to 0.
* **Derived dataclasses** --
  :meth:`RuntimeState.decoder_only_meta`,
  :meth:`RuntimeState.encdec_meta`,
  :meth:`RuntimeState.inactive_sampler`. Cached after first call.
* **Client** -- :meth:`RuntimeState.judge_client`. SQLite-backed, opened
  lazily and closed in :meth:`RuntimeState.close`.

Import cost
-----------
This module imports :class:`steering.eval.axbench_base.judge.AxBenchJudgeClient`
at module scope per the project rule that imports must live at file top. Its
dependencies include torch + HuggingFace weights-loading code, so the module
import is not cheap.

Design reference
----------------
See the AxBench-Base implementation plan, section 3.1
("RuntimeState scaffold") for the full life-cycle contract this
module implements.
"""

from __future__ import annotations

import contextlib
import os
import random
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger
from transformers import AutoTokenizer

from nanochat.checkpoint_manager import ModelPhase, load_model
from nanochat.tokenizer import get_tokenizer as _nanochat_get_tokenizer
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.conditional.sampling import (
    parse_negative_selection,
    sample_inactive_features,
    sample_k,
)
from steering.eval.axbench_base import paths
from steering.eval.axbench_base._log_helpers import StepwiseTimer, is_log_rank
from steering.eval.axbench_base._stats_reader import read_n_sae_features_from_stats
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    FeatureStatisticsPathConfig,
    bare_model_id,
    effective_extraction_model_id,
)
from steering.eval.axbench_base.judge import AxBenchJudgeClient
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.openrouter import OPENROUTER_API_KEY_ENV
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_base.seed_contract import (
    LoadedSeedSplit,
    expectation_from_config,
    load_seed_pool_with_rows_identity,
)
from steering.eval.axbench_base.seed_pools import load_seed_pool
from steering.eval.axbench_base.synth.concept_yield import (
    load_concept_yield_surviving_ids,
    load_filtered_concepts,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    Stage,
    TargetModelMetadata,
)
from steering.eval.axbench_base.validate import load_and_validate_concepts
from steering.eval.cluster_checkpoint_pin import (
    _mapping_get,
    assert_checkpoint_files_pin_unchanged,
    resolve_checkpoint_files_pin,
    resolve_eval_checkpoint_directory,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    assert_loaded_cluster_target,
)
from steering.eval.neuronpedia_client import NEURONPEDIA_API_KEY_ENV
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.extract_features import build_nnterp_model, load_sae_and_model
from steering.sae_resolver import (
    SAERelease,
    parse_neuronpedia_source,
    resolve_neuronpedia_ids,
    resolve_sae,
)

# ---------------------------------------------------------------------------
# Environment-variable validation for stages
#
# Public helper called by the CLI ``_run`` (and by any future harness)
# BEFORE any heavy model load, so missing credentials are surfaced to
# operators immediately rather than after a 30-second Gemma download.
# The set is intentionally kept next to :class:`RuntimeState` because
# future heavy-resource additions (e.g. Anthropic judge, Gemini judge)
# may land new env-var requirements, and this file already owns the
# "what does this stage need" knowledge.
# ---------------------------------------------------------------------------


#: Stages that unconditionally require :data:`OPENROUTER_API_KEY_ENV`
#: to be present in the environment before dispatch. Stage 0c is NOT
#: in this set because its OpenRouter dependency is conditional on
#: :attr:`AxBenchBaseConfig.enable_concept_examples_qc` or
#: :attr:`AxBenchBaseConfig.verification_api_model`; see
#: :func:`validate_env_for_stages` for the conditional handling.
_STAGES_REQUIRING_OPENROUTER: frozenset[Stage] = frozenset(
    {
        # Stage 1b / 1c drive OpenRouter for synth + QC.
        Stage.SYNTH_POSITIVE_MODE,
        Stage.SYNTH_NEGATIVE_MODE,
        # Stage 5 drives the AxBench judge via OpenRouter.
        Stage.JUDGE,
    }
)


#: Stages that unconditionally require :data:`NEURONPEDIA_API_KEY_ENV`.
#: Stage 0a fetches feature descriptions from Neuronpedia regardless of
#: ``cfg.enable_concept_examples_qc`` (Neuronpedia is the description
#: source; QC is a separate post-fetch judge step that touches OpenRouter).
_STAGES_REQUIRING_NEURONPEDIA: frozenset[Stage] = frozenset(
    {
        Stage.PREPARE_CONCEPTS,
    }
)


def validate_env_for_stages(
    stages: list[Stage],
    *,
    cfg: AxBenchBaseConfig | None = None,
) -> None:
    """Validate required environment variables are set for the given stages.

    Call BEFORE any model load so operators see missing credentials
    immediately, not after a 30-second Gemma download.

    Currently validates:

    - :data:`OPENROUTER_API_KEY_ENV` for stages that hit the OpenRouter
      API.
    - :data:`NEURONPEDIA_API_KEY_ENV` for stages that hit Neuronpedia.

    Stage 0c's concept-examples QC and API verification generator are
    conditional on :attr:`AxBenchBaseConfig.enable_concept_examples_qc` and
    :attr:`AxBenchBaseConfig.verification_api_model`. Either feature requires
    both OpenRouter and Neuronpedia. When ``cfg`` is :data:`None` (defensive
    default), Stage 0c is treated as requiring both credentials.

    Parameters
    ----------
    stages:
        The stages that are about to run. An ``ALL``-expanded stage
        list is acceptable; the function inspects each element.
    cfg:
        Optional config. When provided, enables the Stage 0c QC
        exemption. When :data:`None` (defensive default), Stage 0c is
        treated as requiring the examples-QC env vars.

    Raises
    ------
    AxBenchBaseError
        With an actionable message naming the missing env var, which
        stages in the list need it, and how to set it.
    """
    needs_openrouter: list[Stage] = []
    needs_neuronpedia: list[Stage] = []
    for stage in stages:
        if stage is Stage.PREPARE_CONCEPTS:
            # Neuronpedia is unconditional for Stage 0a.
            needs_neuronpedia.append(stage)
            continue
        if stage is Stage.PREPARE_NEGATIVE_PREFIXES:
            if (
                cfg is None
                or cfg.enable_concept_examples_qc
                or cfg.verification_api_model is not None
            ):
                needs_openrouter.append(stage)
                needs_neuronpedia.append(stage)
            continue
        if stage in _STAGES_REQUIRING_OPENROUTER:
            needs_openrouter.append(stage)
        if stage in _STAGES_REQUIRING_NEURONPEDIA:
            needs_neuronpedia.append(stage)

    missing: list[str] = []
    if needs_openrouter and not os.environ.get(OPENROUTER_API_KEY_ENV):
        triggering = sorted({s.value for s in needs_openrouter})
        missing.append(
            f"{OPENROUTER_API_KEY_ENV} is not set in the environment, but "
            f"the following requested stages require it: {triggering}. "
            f"Set the env var via "
            f"`export {OPENROUTER_API_KEY_ENV}=sk-or-...`."
        )
    if needs_neuronpedia and not os.environ.get(NEURONPEDIA_API_KEY_ENV):
        triggering = sorted({s.value for s in needs_neuronpedia})
        missing.append(
            f"{NEURONPEDIA_API_KEY_ENV} is not set in the environment, but "
            f"the following requested stages require it: {triggering}. "
            f"Set the env var via "
            f"`export {NEURONPEDIA_API_KEY_ENV}=np-...`."
        )
    if missing:
        raise AxBenchBaseError(
            "\n".join(missing)
            + "\nFix the missing env var(s) and re-run, or reduce the stage "
            "list to only stages that don't need them."
        )


# ---------------------------------------------------------------------------
# Heavy-model loader stubs.
#
# Tests replace these at module scope (see ``stub_heavy_loaders`` in
# ``tests/axbench_base/unit/test_runtime_state.py``). The production
# bodies defer to the real project helpers so a full run invokes
# the same loader paths as the legacy scripts. Tests never actually
# reach these bodies.
# ---------------------------------------------------------------------------


def _load_feature_extraction_pair(
    cfg: AxBenchBaseConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[Any, Any]:
    """Load the feature-extraction base model + SAE once via ``load_sae_and_model``.

    Returns ``(model, sae)`` where ``model`` is a TransformerLens
    :class:`HookedTransformer` with an optional nnterp
    :class:`StandardizedTransformer` speedup wrapper attached as
    ``model.nnterp_model`` (``None`` when nnterp cannot wrap the
    architecture). ``sae`` is the SAE object from SAELens.

    Device layout: the nnterp wrapper (when available) and the SAE
    live on ``cfg.device``; the :class:`HookedTransformer` itself is
    parked on CPU by :func:`_build_nnterp_gemma2` to avoid a
    simultaneous fp32 + HF-load VRAM spike. Extractors drive forwards
    through nnterp (bf16 on GPU) and fall back to
    ``HookedTransformer.run_with_cache`` on CPU when the nnterp path
    isn't populated.

    We intentionally return the :class:`HookedTransformer` — NOT the
    nnterp wrapper — because downstream axbench_base extractors call
    ``model.run_with_cache(...)`` / ``model.cfg.n_ctx`` on the base
    model; the nnterp wrapper does not expose that API. The nnterp
    wrapper is kept accessible as a side-channel attribute so callers
    that want the speedup path (e.g. the extraction pipeline in
    :mod:`steering.extract_features`) can opt in by reading
    ``getattr(model, "nnterp_model", None)`` and passing it through.

    This single-call loader replaces the old separate
    ``_load_base_model`` / ``_load_base_sae`` functions that each called
    ``load_sae_and_model`` independently, wasting ~30s by downloading
    and loading the full model+SAE pair twice.

    SAE identifier resolution: ``cfg.neuronpedia_source`` is a
    Neuronpedia-format string (e.g., ``"8-gemmascope-res-16k"``) that
    encodes ``(layer, release, activation_type, width)``. We parse it
    via :func:`steering.sae_resolver.parse_neuronpedia_source` and
    feed the structured tuple into :func:`resolve_sae` to obtain the
    concrete SAELens ``release`` + ``sae_id`` + TransformerLens
    ``hook_name`` strings. The local model id used for this loading path
    is ``cfg.feature_extraction_model_id or cfg.neuronpedia_model_id``:
    Neuronpedia's public API namespace can differ from the local
    extraction model id (for example Gemma-3 ``gemma-3-4b`` vs
    ``gemma-3-4b-pt``). The TransformerLens model is loaded at the SAE's
    encoded layer, NOT at ``cfg.layer_hooks.diffmean_and_probe_layer``
    which refers to the target-model hook layer used by Stage 3 baselines
    -- those two concepts are intentionally independent.

    Post-load identity validation applies to clustered and unclustered loads.
    Against the pinned SAELens 6.42.0 registry, real loads never populate
    ``sae_id`` (``LLM_METADATA_KEYS`` does not contain that field), and
    ``neuronpedia_id`` is null for 1,148 of 1,264 GemmaScope-2 combinations.
    Consequently, an unclustered GemmaScope-2 run on one of those unpublished
    combinations fails closed because its L0 tier cannot be proven. The shipped
    ``gemma-3-4b-pt`` / ``22-gemmascope-2-res-65k`` configuration is published
    as ``gemma-3-4b/22-gemmascope-2-res-65k`` and remains accepted.

    Raises:
        AxBenchBaseError: if ``cfg.neuronpedia_source`` is malformed, or
            if the model or SAE cannot be downloaded, loaded, placed on
            the requested device, or validated against the resolved SAE
            identity. The original exception is chained as ``__cause__``.
    """
    try:
        with profile_region("stage0c.load_feature_extraction_pair"):
            timer = StepwiseTimer()
            feature_extraction_model_id = effective_extraction_model_id(cfg)
            if is_log_rank():
                logger.info(
                    "[runtime_state] resolving Neuronpedia source ({source!r})",
                    source=cfg.neuronpedia_source,
                )
            _release_enum, activation_type, sae_layer, width, l0_tier = (
                parse_neuronpedia_source(cfg.neuronpedia_source)
            )
            timer.step(
                "[runtime_state] resolving SAE release/sae_id "
                "(model={model!r}, layer={layer}, width={width})",
                model=feature_extraction_model_id, layer=sae_layer, width=width,
            )
            resolved = resolve_sae(
                model=bare_model_id(str(feature_extraction_model_id)),
                release=_release_enum,
                activation_type=activation_type,
                layer=sae_layer,
                width=width,
                l0=l0_tier,
            )
            if cluster_contract is not None:
                contract_sae = cluster_contract.sae
                configured_identity = (
                    # Both sides bare-normalized: the configured id may be
                    # org-prefixed (SAELens force-prefixes Gemma-3) while a
                    # contract may record either form. Normalizing one side
                    # only moves the mismatch.
                    bare_model_id(str(feature_extraction_model_id)),
                    activation_type.value,
                    sae_layer,
                    width.value,
                    resolved.release,
                    resolved.sae_id,
                    resolved.hook_name,
                )
                contract_identity = (
                    bare_model_id(str(contract_sae.model)),
                    contract_sae.activation_type,
                    contract_sae.layer,
                    contract_sae.width,
                    contract_sae.release,
                    contract_sae.sae_id,
                    contract_sae.hook_name,
                )
                if configured_identity != contract_identity:
                    raise ValueError(
                        "runtime extraction SAE identity does not match the "
                        "cluster feature contract: "
                        f"configured={configured_identity!r}, "
                        f"contract={contract_identity!r}"
                    )
            timer.step(
                "[runtime_state] loading TransformerLens base model + SAE "
                "(device={device!r})",
                device=cfg.device,
            )
            tl_model, sae, _tokenizer = load_sae_and_model(
                model_name=feature_extraction_model_id,
                layer=sae_layer,
                device=cfg.device,
                resolved=resolved,
            )
            sae_cfg = getattr(sae, "cfg", None)
            loaded_metadata = _mapping_get(sae_cfg, "metadata", None)
            if cluster_contract is not None:
                loaded_width = (
                    sae_cfg.get("d_sae")
                    if isinstance(sae_cfg, Mapping)
                    else getattr(sae_cfg, "d_sae", None)
                )
                loaded_hook = (
                    sae_cfg.get("hook_name")
                    if isinstance(sae_cfg, Mapping)
                    else getattr(sae_cfg, "hook_name", None)
                )
                if not loaded_hook:
                    sae_metadata = (
                        sae_cfg.get("metadata")
                        if isinstance(sae_cfg, Mapping)
                        else getattr(sae_cfg, "metadata", None)
                    )
                    loaded_hook = (
                        sae_metadata.get("hook_name")
                        if isinstance(sae_metadata, Mapping)
                        else getattr(sae_metadata, "hook_name", None)
                    )
                decoder = getattr(sae, "W_dec", None)
                if (
                    type(loaded_width) is not int
                    or loaded_width != cluster_contract.n_sae_features
                    or not isinstance(loaded_hook, str)
                    or not loaded_hook
                    or loaded_hook != cluster_contract.sae.hook_name
                    or not isinstance(decoder, t.Tensor)
                    or decoder.ndim != 2
                    or int(decoder.shape[0]) != cluster_contract.n_sae_features
                ):
                    raise ValueError(
                        "loaded runtime extraction SAE does not match the cluster "
                        "feature contract: "
                        f"d_sae={loaded_width!r}, hook={loaded_hook!r}, "
                        f"W_dec_shape={getattr(decoder, 'shape', None)!r}"
                    )

            # Shape/hook checks cannot distinguish GemmaScope-2 L0 tiers.
            # Bind every loaded identity field to the resolver output, and
            # fail closed for v2 when the returned SAE exposes no tier-
            # discriminating sae_id or Neuronpedia identity at all.
            direct_release = _mapping_get(sae_cfg, "release", None)
            if direct_release is not None and direct_release != resolved.release:
                raise ValueError(
                    "loaded runtime extraction SAE cfg.release "
                    f"{direct_release!r} does not match resolved release "
                    f"{resolved.release!r}"
                )
            direct_model_name = _mapping_get(sae_cfg, "model_name", None)
            metadata_model_name = _mapping_get(
                loaded_metadata,
                "model_name",
                None,
            )
            for loaded_model_name in (
                direct_model_name,
                metadata_model_name,
            ):
                if (
                    loaded_model_name is not None
                    and bare_model_id(loaded_model_name)
                    != bare_model_id(feature_extraction_model_id)
                ):
                    raise ValueError(
                        "loaded runtime extraction SAE model_name "
                        f"{loaded_model_name!r} does not match resolved model "
                        f"{feature_extraction_model_id!r}"
                    )
            direct_sae_id = _mapping_get(sae_cfg, "sae_id", None)
            metadata_sae_id = _mapping_get(loaded_metadata, "sae_id", None)
            tier_identity_proven = False
            for loaded_sae_id in (direct_sae_id, metadata_sae_id):
                if loaded_sae_id is None:
                    continue
                if loaded_sae_id != resolved.sae_id:
                    raise ValueError(
                        "loaded runtime extraction SAE sae_id "
                        f"{loaded_sae_id!r} does not match resolved sae_id "
                        f"{resolved.sae_id!r} -- likely the wrong L0 tier "
                        "was loaded"
                    )
                tier_identity_proven = True
            metadata_neuronpedia_id = _mapping_get(
                loaded_metadata,
                "neuronpedia_id",
                None,
            )
            expected_neuronpedia_id: str | None = None
            try:
                expected_model_id, expected_source = resolve_neuronpedia_ids(
                    model=feature_extraction_model_id,
                    release=_release_enum,
                    activation_type=activation_type,
                    layer=sae_layer,
                    width=width,
                    l0=l0_tier,
                )
            except ValueError:
                pass
            else:
                expected_neuronpedia_id = f"{expected_model_id}/{expected_source}"
            if metadata_neuronpedia_id is not None:
                if metadata_neuronpedia_id != expected_neuronpedia_id:
                    raise ValueError(
                        "loaded runtime extraction SAE "
                        "cfg.metadata.neuronpedia_id "
                        f"{metadata_neuronpedia_id!r} does not match "
                        "resolved Neuronpedia id "
                        f"{expected_neuronpedia_id!r} -- likely the "
                        "wrong L0 tier was loaded"
                    )
                tier_identity_proven = True
            if (
                _release_enum is SAERelease.gemmascope_2
                and not tier_identity_proven
            ):
                raise ValueError(
                    "loaded runtime extraction SAE cannot prove its "
                    "GemmaScope-2 L0 tier: SAEConfig exposes no matching "
                    "sae_id and cfg.metadata.neuronpedia_id is absent/None"
                )
            timer.step(
                "[runtime_state] building nnterp wrapper (dtype=bf16)",
            )
            # Build nnterp in bf16 to match the extraction-pipeline default
            # and halve the VRAM footprint. The HookedTransformer is already
            # offloaded to CPU inside _build_nnterp_gemma2 to avoid a
            # simultaneous fp32+HF-load spike during load, so after this
            # call the only GPU resident is the nnterp bf16 model + SAE.
            # Cast the SAE to bf16 too so the extractor's encode path runs
            # in the same dtype as the upstream activations (mirrors the
            # extraction pipeline's ``sae.to(torch.bfloat16)`` step).
            nn_model = build_nnterp_model(
                tl_model,
                device=cfg.device,
                model_dtype=t.bfloat16,
            )
            if nn_model is not None:
                timer.step("[runtime_state] casting SAE to bf16")
                sae.to(t.bfloat16)
            else:
                timer.step(
                    "[runtime_state] nnterp wrapper unavailable for this "
                    "architecture; SAE stays at default dtype",
                )
            # Attach nnterp as a side-channel attribute so downstream
            # callers that opt in to the speedup path can read it, while
            # the model itself keeps HookedTransformer's run_with_cache /
            # .cfg.n_ctx API that the axbench_base extractors rely on.
            # Use setattr() because HookedTransformer declares a custom
            # __setattr__ that makes direct attribute assignment fail ty's
            # unresolved-attribute check.
            setattr(tl_model, "nnterp_model", nn_model)  # noqa: B010
            if is_log_rank():
                logger.info(
                    "[runtime_state] feature-extraction pair ready "
                    "[total_elapsed_ms={ms}]",
                    ms=timer.total_ms(),
                )
            return tl_model, sae
    except Exception as exc:
        feature_extraction_model_id = effective_extraction_model_id(cfg)
        raise AxBenchBaseError(
            f"Failed to load feature-extraction model + SAE "
            f"(model={feature_extraction_model_id!r}, "
            f"neuronpedia_model_id={cfg.neuronpedia_model_id!r}, "
            f"neuronpedia_source={cfg.neuronpedia_source!r}) on device "
            f"{cfg.device!r}: {exc}. The pair is loaded via "
            f"steering.extract_features.load_sae_and_model using IDs "
            f"derived from cfg.neuronpedia_source -- verify the source "
            f"string parses into (layer, release, activation, width), the "
            f"SAELens release and sae_id are reachable, the HF model id "
            f"is cached or downloadable, and that the requested device "
            f"has enough VRAM."
        ) from exc


def _load_decoder_only_model(
    cfg: AxBenchBaseConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Any:
    """Load the nanochat decoder-only target model on ``cfg.device``.

    Reads ``cfg.decoder_only`` for ``source`` / ``model_tag`` / ``step``
    and dispatches to :func:`nanochat.checkpoint_manager.load_model`,
    which handles source-chain resolution, tag discovery, and step
    discovery.

    Raises:
        AxBenchBaseError: if the checkpoint cannot be located, loaded,
            or placed on the requested device. The original exception
            is chained as ``__cause__``.
    """
    decoder_only = cfg.decoder_only
    try:
        source = decoder_only["source"]
        model_tag = decoder_only.get("model_tag")
        raw_step = decoder_only.get("step")
        checkpoint_pin = None
        if cluster_contract is None:
            step = raw_step
        else:
            if not isinstance(source, str) or not source:
                raise AxBenchBaseError(
                    "clustered decoder_only.source must be a non-empty string"
                )
            if not isinstance(model_tag, str) or not model_tag:
                raise AxBenchBaseError(
                    "clustered decoder_only.model_tag must be a non-empty string"
                )
            if raw_step in (None, "last"):
                requested_step = None
            elif isinstance(raw_step, Integral) and not isinstance(raw_step, bool):
                requested_step = int(raw_step)
            else:
                raise AxBenchBaseError(
                    "clustered decoder_only.step must be a non-negative integer, "
                    f"null, or 'last'; got {raw_step!r}"
                )
            if requested_step is not None and requested_step < 0:
                raise AxBenchBaseError(
                    "clustered decoder_only.step must be non-negative"
                )
            checkpoint_dir, resolved_tag = resolve_eval_checkpoint_directory(
                checkpoint_dir=None,
                checkpoint_source=source,
                checkpoint_model_tag=model_tag,
            )
            checkpoint_pin = resolve_checkpoint_files_pin(
                checkpoint_dir,
                step=requested_step,
                resolved_tag=resolved_tag,
            )[0]
            # The nanochat loader below reopens these pathnames. The pin's
            # post-load digest + lstat-signature recheck brackets that reopen
            # with unchanged file identity and content; the descriptor-bound
            # state dict returned here is validation-only and is not injected
            # into nanochat's model-build transform chain.
            step = checkpoint_pin.step
        device = t.device(cfg.device)
        model, _tokenizer, _meta = load_model(
            source=source,
            device=device,
            phase=ModelPhase.EVAL,
            model_tag=model_tag,
            step=step,
            eval_conditioning=EvalConditioning.deferred("axbench_base"),
        )
        reject_feature_clusters_metadata(_meta, "axbench_base decoder checkpoint")
        if checkpoint_pin is not None:
            assert_checkpoint_files_pin_unchanged(checkpoint_pin)
            setattr(model, "_loaded_checkpoint_files_pin", checkpoint_pin)  # noqa: B010
        return model
    except Exception as exc:
        raise AxBenchBaseError(
            f"Failed to load decoder-only nanochat model from "
            f"cfg.decoder_only={decoder_only!r} on device "
            f"{cfg.device!r}: {exc}. The checkpoint is loaded via "
            "nanochat.checkpoint_manager.load_model; verify the "
            "source/model_tag/step triple exists under "
            "$NANOCHAT_BASE_DIR and that the device has enough VRAM."
        ) from exc


def _load_encdec_model(
    cfg: AxBenchBaseConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Any:
    """Load the nanochat encoder-decoder target model on ``cfg.device``.

    Reads ``cfg.encdec`` for ``source`` / ``model_tag`` / ``step`` and
    dispatches to :func:`nanochat.checkpoint_manager.load_model`. The
    returned model has the checkpoint ``meta_data`` dict attached as
    ``model._loaded_meta_data`` -- downstream inference code needs it
    to rebuild the tokenizer / sampler wrapper, and the runtime state
    is the one place we can guarantee this attachment happens exactly
    once.

    Raises:
        AxBenchBaseError: if the checkpoint cannot be located, loaded,
            or placed on the requested device. The original exception
            is chained as ``__cause__``.
    """
    encdec = cfg.encdec
    try:
        source = encdec["source"]
        model_tag = encdec.get("model_tag")
        step = encdec.get("step")
        device = t.device(cfg.device)
        model, _tokenizer, meta_data = load_model(
            source=source,
            device=device,
            phase=ModelPhase.EVAL,
            model_tag=model_tag,
            step=step,
            eval_conditioning=(
                EvalConditioning.deferred("axbench_base")
                if cluster_contract is None
                else EvalConditioning.conditions_within_vocab(
                    "axbench_base_clustered"
                )
            ),
        )
        if cluster_contract is None:
            reject_feature_clusters_metadata(meta_data, "axbench_base encdec checkpoint")
        else:
            assert_loaded_cluster_target(
                cluster_contract,
                model,
                meta_data,
                surface="axbench_base encdec checkpoint",
            )
    except Exception as exc:
        raise AxBenchBaseError(
            f"Failed to load encoder-decoder nanochat model from "
            f"cfg.encdec={encdec!r} on device {cfg.device!r}: {exc}. "
            "The checkpoint is loaded via "
            "nanochat.checkpoint_manager.load_model; verify the "
            "source/model_tag/step triple exists under "
            "$NANOCHAT_BASE_DIR and that the device has enough VRAM."
        ) from exc
    # Outside the loader's try: a turn-delimited checkpoint is a supported-mode
    # rejection, not a "checkpoint could not be loaded" failure, so it must
    # surface the guard's own message rather than the VRAM/path hint above.
    require_fixed_checkpoint_segment_mode(
        meta_data,
        surface="axbench_base",
    )
    # Attach checkpoint metadata so downstream callers (see
    # ``steering.eval.axbench_base.model_detection``) don't need to
    # re-read the checkpoint to rebuild tokenizer state.
    setattr(model, "_loaded_meta_data", meta_data)  # noqa: B010
    return model


# ---------------------------------------------------------------------------
# Inactive-sampler builder.
# ---------------------------------------------------------------------------


#: Upper bound for the int64 seeds pulled from ``random.Random`` in the
#: :func:`_build_inactive_sampler` closure. Must stay strictly below
#: ``2**63`` so the seed fits in a signed 64-bit int (the representation
#: ``torch.Generator.manual_seed`` expects). Keeping this as a module
#: constant makes the rng-bridge contract explicit and gives tests a
#: single patch point when verifying the seed-derivation path.
_TORCH_SEED_UPPER_BOUND: int = (1 << 63) - 1


def _build_inactive_sampler(
    *,
    encdec_meta: TargetModelMetadata,
    cfg: FeatureStatisticsPathConfig,
    allowed_feature_ids: t.Tensor | None = None,
) -> Callable[..., dict[int, int]]:
    """Build an inactive-feature sampler from the encdec training config.

    The encdec target's training-time ``user_config["negative_selection"]``
    describes how many inactive features to sample per Stage 4
    generation call (distribution, mean, std, min_k, max_k). This
    function parses that config into a
    :class:`steering.conditional.sampling.NegativeSamplingSelection`
    and returns a callable ``(*, rng: random.Random) -> dict[int, int]``
    that every call samples a fresh set of inactive fids and maps
    them all to the inactive bucket (``num_buckets - 1``).

    The orchestrator
    (:func:`steering.eval.axbench_base.encoder_input.build_mode_aware_encoder_input`)
    threads a :class:`random.Random` through the partition modifiers,
    but
    :func:`steering.conditional.sampling.sample_inactive_features`
    expects a :class:`torch.Generator`. The returned closure bridges
    the two by pulling a fresh int64 seed from the incoming
    ``random.Random`` on every call and seeding a fresh
    :class:`torch.Generator` with it. This keeps the orchestrator's
    ``random.Random`` API unchanged (so other downstream partition
    modifiers that use ``rng.sample(...)`` continue to work) and
    produces a deterministic sequence of inactive samples for a fixed
    ``random.Random`` seed. The Option-A rationale is documented in
    the runtime-state PR; briefly, bridging the two RNG types at
    the closure boundary keeps the orchestrator's public
    ``rng: random.Random`` signature unchanged.

    The legacy closure invokes
    :func:`steering.conditional.sampling.sample_inactive_features`
    with ``active_feature_ids=<empty int64 tensor>``. The empty-active
    decision is deliberate: Stage 4 does not pass the per-segment
    active fid set down to the sampler (the orchestrator signature
    only forwards ``rng``), and the downstream
    :func:`steering.eval.axbench_base.encoder_input.collision_filter`
    already handles the rare case where a sampled inactive fid
    collides with an active fid. The restricted-universe branch samples the
    same configured count, clamps it to the universe size, and indexes a uniform
    random permutation of the captured allowed-ID snapshot. Sampling
    directly without replacement avoids the low-ID bias that sorted
    deduplication in generic rejection sampling would introduce.

    Args:
        encdec_meta: The encdec target's metadata. Must have both
            :attr:`num_buckets` (``int``, ``>= 2``) and
            :attr:`negative_selection` (non-``None`` dict) set. If
            either is missing we raise :class:`AxBenchBaseError`
            because
            :meth:`RuntimeState.inactive_sampler` is the only caller
            and it short-circuits to ``None`` on
            ``negative_selection is None`` before reaching this
            helper — so a missing field here indicates a contract
            violation, not a user-facing "no negative sampling"
            condition.
        cfg: Any config exposing ``feature_statistics_path``. Used so the sampler knows the
            total SAE feature count (needed for rejection sampling).
        allowed_feature_ids: Optional validated conditioning universe. ``None``
            selects the exact legacy full-SAE-universe path. A non-``None``
            value must be a sorted, unique, non-empty, proper-subset int64
            vector and produces a uniform source-level sampler over that
            captured universe. Ordinary compacted callers pass checkpoint
            vocabulary ``V``; clustered AxBench passes only the representative
            universe from its exact feature contract.

    Returns:
        A callable with the signature
        ``(*, rng: random.Random) -> dict[int, int]``. The dict maps
        sampled inactive fids to ``encdec_meta.num_buckets - 1``.

    Raises:
        AxBenchBaseError: if ``encdec_meta.num_buckets`` or
            ``encdec_meta.negative_selection`` is ``None``, if
            :func:`parse_negative_selection` returns ``None`` from a
            non-``None`` config dict (defensive — should never happen
            given the upstream validator), or if reading the total
            SAE feature count from
            ``feature_statistics_path`` fails.
    """
    if encdec_meta.num_buckets is None:
        raise AxBenchBaseError(
            "Cannot build inactive sampler: encdec_meta.num_buckets is "
            "None. The encoder-decoder target was not trained with "
            "buckets, so there is no meaningful 'inactive bucket' to "
            "assign sampled fids to. RuntimeState.inactive_sampler "
            "should have short-circuited to None before reaching this "
            "helper — this error indicates a contract violation."
        )
    if encdec_meta.negative_selection is None:
        raise AxBenchBaseError(
            "Cannot build inactive sampler: encdec_meta.negative_selection "
            "is None. The encoder-decoder target was not trained with "
            "negative feature sampling. RuntimeState.inactive_sampler "
            "should have short-circuited to None before reaching this "
            "helper — this error indicates a contract violation."
        )

    selection = parse_negative_selection(encdec_meta.negative_selection)
    if selection is None:
        raise AxBenchBaseError(
            "parse_negative_selection returned None from a non-None "
            f"negative_selection dict {encdec_meta.negative_selection!r}. "
            "This should never happen; check the encdec's training "
            "user_config or the parse_negative_selection implementation."
        )

    n_sae_features = read_n_sae_features_from_stats(
        Path(cfg.feature_statistics_path),
    )
    inactive_bucket = encdec_meta.num_buckets - 1

    if allowed_feature_ids is None:
        def _legacy_sampler(*, rng: random.Random) -> dict[int, int]:
            torch_seed = rng.randint(0, _TORCH_SEED_UPPER_BOUND)
            torch_rng = t.Generator(device="cpu")
            torch_rng.manual_seed(torch_seed)
            # Pass an empty active set because the sampler is invoked BEFORE
            # the per-row active partition is known. During Stage 4 generation,
            # build_mode_aware_encoder_input calls this sampler to populate the
            # inactive partition, then the partition modifier adds/removes concept
            # features, and finally collision_filter removes any overlap between
            # active and inactive. So excluding active features here would be
            # premature -- the collision_filter downstream is the correct place
            # to resolve active/inactive overlap.
            active_empty = t.empty(0, dtype=t.int64)
            sampled = sample_inactive_features(
                active_feature_ids=active_empty,
                n_sae_features=n_sae_features,
                config=selection,
                rng=torch_rng,
            )
            return {int(fid): inactive_bucket for fid in sampled.tolist()}

        return _legacy_sampler

    if allowed_feature_ids.ndim != 1:
        raise AxBenchBaseError(
            "allowed_feature_ids must be one-dimensional, got "
            f"shape={tuple(allowed_feature_ids.shape)}"
        )
    if allowed_feature_ids.dtype != t.int64:
        raise AxBenchBaseError(
            "allowed_feature_ids must have dtype torch.int64, got "
            f"{allowed_feature_ids.dtype}"
        )
    if encdec_meta.n_sae_features is None:
        raise AxBenchBaseError(
            "Cannot build checkpoint-V-restricted inactive sampler: "
            "encdec_meta.n_sae_features is None."
        )
    if encdec_meta.n_sae_features != n_sae_features:
        raise AxBenchBaseError(
            "Cannot build checkpoint-V-restricted inactive sampler: target "
            f"metadata n_sae_features={encdec_meta.n_sae_features} does not "
            f"match feature statistics n_sae_features={n_sae_features}."
        )

    allowed_cpu = (
        allowed_feature_ids.detach()
        .to(device="cpu", dtype=t.int64)
        .contiguous()
        .clone()
    )
    if allowed_cpu.numel() == 0:
        raise AxBenchBaseError(
            "allowed_feature_ids must be non-empty for a compacted checkpoint."
        )
    if bool((allowed_cpu[1:] <= allowed_cpu[:-1]).any()):
        raise AxBenchBaseError(
            "allowed_feature_ids must be strictly increasing (sorted and unique)."
        )
    minimum = int(allowed_cpu[0].item())
    maximum = int(allowed_cpu[-1].item())
    if minimum < 0:
        raise AxBenchBaseError(
            f"allowed_feature_ids must be non-negative, got minimum={minimum}."
        )
    if maximum >= n_sae_features:
        raise AxBenchBaseError(
            f"allowed_feature_ids maximum={maximum} must be less than "
            f"n_sae_features={n_sae_features}."
        )
    if allowed_cpu.numel() >= n_sae_features:
        raise AxBenchBaseError(
            "a non-None allowed_feature_ids vocabulary must be a genuine "
            f"proper subset of the SAE universe, got cardinality="
            f"{allowed_cpu.numel()} and n_sae_features={n_sae_features}."
        )

    def _restricted_sampler(*, rng: random.Random) -> dict[int, int]:
        torch_seed = rng.randint(0, _TORCH_SEED_UPPER_BOUND)
        torch_rng = t.Generator(device="cpu")
        torch_rng.manual_seed(torch_seed)
        k = sample_k(
            selection.distribution,
            selection.mean,
            selection.std,
            torch_rng,
            min_k=selection.min_k,
            max_k=selection.max_k,
        )
        k = min(k, int(allowed_cpu.numel()))
        if k <= 0:
            return {}
        permutation = t.randperm(allowed_cpu.numel(), generator=torch_rng)
        sampled = allowed_cpu[permutation[:k]]
        return {int(fid): inactive_bucket for fid in sampled.tolist()}

    return _restricted_sampler


# ---------------------------------------------------------------------------
# RuntimeState
# ---------------------------------------------------------------------------


@dataclass
class RuntimeState:
    """Lazy-loaded, refcount-managed runtime state for axbench_base stages.

    See the module docstring for the overall design contract. This
    dataclass exposes:

    * **Pure-data accessors** for cheap artifacts (concepts, seeds, tokenizers).
    * **Heavy-model context managers** :meth:`base_model`,
      :meth:`base_sae`, :meth:`decoder_only_model`, :meth:`encdec_model`
      that refcount the underlying model and park it on CPU when the
      refcount drops to 0.
    * **Derived accessors** for detected target-model metas
      (:meth:`decoder_only_meta`, :meth:`encdec_meta`) and the inactive-feature
      sampler (:meth:`inactive_sampler`).
    * **SQLite-backed resource** :meth:`judge_client`, opened lazily and closed
      in :meth:`close`.
    """

    cfg: AxBenchBaseConfig

    # DDP coordinates. Default to single-rank when not running under
    # torchrun. The CLI populates these at construction (see
    # :func:`scripts.eval.axbench_base_eval._run`); stages that need
    # rank-aware dispatch (Stage 0c since PR-4) read them through the
    # state instance rather than reaching into ``torch.distributed``
    # again.
    ddp_rank: int = 0
    ddp_world_size: int = 1
    cluster_contract: ClusterFeatureContract | None = None

    # Pure-data artifacts -- cheap, loaded from disk on demand.
    _concepts: list[Concept] | None = field(default=None, repr=False)
    _stage1b_concepts: list[Concept] | None = field(default=None, repr=False)
    _downstream_concepts: list[Concept] | None = field(default=None, repr=False)
    _seeds_train: LoadedSeedSplit | None = field(default=None, repr=False)
    _seeds_eval: list[SeedRow] | None = field(default=None, repr=False)
    # Tokenizers -- CPU-cheap, resident once loaded.
    _nanochat_tokenizer: Any = field(default=None, repr=False)
    _extraction_tokenizer: Any = field(default=None, repr=False)

    # Heavy models. While the corresponding refcount is > 0 the model
    # lives on ``self._device``; at refcount 0 it is moved to the CPU
    # (but the instance is kept so re-entry avoids the loader).
    #
    # The base model and SAE are loaded together via
    # ``_load_feature_extraction_pair`` on first access to either one.
    # The ``_feat_extraction_loaded`` flag tracks whether the shared
    # loader has run; the two objects are cached independently so they
    # can be offloaded to CPU at different times.
    _feat_extraction_loaded: bool = field(default=False, repr=False)
    _base_model: Any = field(default=None, repr=False)
    _base_model_refcount: int = field(default=0, repr=False)

    _base_sae: Any = field(default=None, repr=False)
    _base_sae_refcount: int = field(default=0, repr=False)

    _decoder_only_model: Any = field(default=None, repr=False)
    _decoder_only_refcount: int = field(default=0, repr=False)

    _encdec_model: Any = field(default=None, repr=False)
    _encdec_refcount: int = field(default=0, repr=False)

    # Metas (dataclass instances), the SQLite-backed judge client, and the
    # derived inactive sampler. Metas are cached on first access and never
    # rebuilt; the client is opened lazily and closed in :meth:`close`; the
    # inactive sampler is a derived field that may
    # legitimately resolve to ``None`` (target not trained with
    # negative feature sampling), so the ``_inactive_sampler_loaded``
    # sentinel distinguishes "not yet loaded" from "loaded and None".
    _decoder_only_meta: TargetModelMetadata | None = field(default=None, repr=False)
    _encdec_meta: TargetModelMetadata | None = field(default=None, repr=False)

    _judge_client: AxBenchJudgeClient | None = field(default=None, repr=False)

    _inactive_sampler: Callable[..., dict[int, int]] | None = field(
        default=None,
        repr=False,
    )
    _inactive_sampler_loaded: bool = field(default=False, repr=False)

    # Populated in ``__post_init__`` from ``cfg.device``. Stored
    # separately so heavy accessors can read a plain str without
    # reaching into cfg on every entry.
    _device: str = field(init=False, default="", repr=False)
    _output_paths: paths.AxBenchBasePaths | None = field(
        init=False,
        default=None,
        repr=False,
    )

    _closed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the target device string from ``cfg.device``.

        Called automatically by the :mod:`dataclasses` machinery after
        ``__init__``. Stored as a plain attribute (not a dataclass
        field with ``init=True``) so construction signatures stay
        minimal and the value is guaranteed to come from cfg.
        """
        self._device = self.cfg.device
        if self.cfg.run_tag is not None:
            self._output_paths = paths.AxBenchBasePaths.from_config(self.cfg)

    def output_paths(self) -> paths.AxBenchBasePaths:
        """Return the immutable output-path contract for this invocation."""
        self._require_run_tag()
        if self._output_paths is None:
            self._output_paths = paths.AxBenchBasePaths.from_config(self.cfg)
        return self._output_paths

    # -----------------------------------------------------------------
    # Invariants
    # -----------------------------------------------------------------

    def _check_not_closed(self) -> None:
        """Raise :class:`AxBenchBaseError` if :meth:`close` has been called.

        The CLI always wraps stage dispatch in a ``try/finally`` that
        calls ``close()``. Any accessor call after that point indicates
        a stage that cached the state and is trying to touch it after
        the pipeline teardown -- a programming error.
        """
        if self._closed:
            raise AxBenchBaseError(
                "RuntimeState is closed. Construct a fresh instance; "
                "accessor calls after close() indicate a resource leak in "
                "a stage that cached a RuntimeState reference past the "
                "pipeline teardown."
            )

    def _check_main_thread(self) -> None:
        """Raise :class:`AxBenchBaseError` if called off the main thread.

        See the module docstring for why :class:`RuntimeState` is not
        thread-safe and why every accessor asserts the main-thread
        invariant.
        """
        if threading.current_thread() is not threading.main_thread():
            raise AxBenchBaseError(
                "RuntimeState must be accessed from the main thread only. "
                "Worker-thread access would race the lazy-load sentinels "
                "and future refcount fields. Dispatch stages on the main "
                "thread, or introduce a lock before allowing off-thread "
                "access."
            )

    # -----------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Idempotent teardown.

        Drops every cached reference so the garbage collector can
        reclaim the loaded artifacts, and marks the state as
        ``_closed=True`` so subsequent accessor calls crash loudly.

        Teardown is best-effort: every SQLite-backed resource
        (generation cache, synth response cache, judge client) is
        closed in an isolated ``try/except`` block, and the failure
        is logged at WARNING level via :mod:`loguru` without
        propagating. This ensures that a single faulty
        ``.close()`` cannot leak the other resources by
        short-circuiting the rest of teardown. ``_closed = True``
        is set unconditionally at the end so a second
        :meth:`close` call after a partial failure is still
        idempotent.

        Raises:
            AxBenchBaseError: if any heavy-model context manager has
                a non-zero refcount when :meth:`close` is called. This
                catches bugs where a ``with state.base_model() as m:``
                block escapes the ``finally`` via a bare ``return`` or
                similar control-flow mistake, which would otherwise
                leave the caller holding a dangling reference to a
                model whose owning state has been half-torn-down.

                The :attr:`_closed` flag is NOT set when this raises,
                so callers can close their context managers and
                re-invoke :meth:`close` successfully.
        """
        if self._closed:
            return
        self._check_main_thread()
        active_refcounts = {
            name: count
            for name, count in (
                ("base_model", self._base_model_refcount),
                ("base_sae", self._base_sae_refcount),
                ("decoder_only_model", self._decoder_only_refcount),
                ("encdec_model", self._encdec_refcount),
            )
            if count > 0
        }
        if active_refcounts:
            raise AxBenchBaseError(
                "RuntimeState.close() called with active model contexts: "
                f"{active_refcounts}. Exit the with-block(s) before "
                "closing the runtime state."
            )
        # Best-effort teardown: every SQLite-backed resource gets its
        # own try/except so a raise from one cache.close() cannot
        # prevent the other resources from being closed. Each failure
        # is logged at WARNING level (naming the resource and the
        # exception type) so operators can spot a dirty teardown in
        # the logs. Order: SQLite-backed resources first (so their
        # connections close cleanly while the rest of the state is
        # still intact), then derived refs, then pure-data and
        # heavy-model caches.
        sqlite_resources: list[tuple[str, Any]] = [
            ("judge_client", self._judge_client),
        ]
        for resource_name, resource in sqlite_resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RuntimeState.close(): {name}.close() raised "
                    "{exc_type}: {exc}. Continuing teardown.",
                    name=resource_name,
                    exc_type=type(exc).__name__,
                    exc=exc,
                )
        # Drop references unconditionally -- the assignments below
        # cannot raise, so we run them regardless of whether any
        # individual cache close() above raised.
        self._judge_client = None
        # Drop derived / cached-dataclass references.
        self._inactive_sampler = None
        self._inactive_sampler_loaded = False
        self._decoder_only_meta = None
        self._encdec_meta = None
        # Drop pure-data references.
        self._concepts = None
        self._stage1b_concepts = None
        self._downstream_concepts = None
        self._seeds_train = None
        self._seeds_eval = None
        self._nanochat_tokenizer = None
        self._extraction_tokenizer = None
        # Drop heavy-model references.
        self._feat_extraction_loaded = False
        self._base_model = None
        self._base_sae = None
        self._decoder_only_model = None
        self._encdec_model = None
        # Best-effort CUDA cache empty -- don't let a torch failure
        # leave the state un-closed.
        if t.cuda.is_available():
            try:
                t.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RuntimeState.close(): torch.cuda.empty_cache() raised "
                    "{exc_type}: {exc}.",
                    exc_type=type(exc).__name__,
                    exc=exc,
                )
        self._closed = True

    # -----------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------

    def concepts(self) -> list[Concept]:
        """Return the validated concept list from ``concepts.json``.

        Reads and validates on first call, then caches. See
        :func:`steering.eval.axbench_base.validate.load_and_validate_concepts`.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._concepts is None:
            self._concepts = self._load_concepts()
        return self._concepts

    def stage1b_concepts(self) -> list[Concept]:
        """Return Stage 1b's authoritative filtered population for Stage 1c."""

        self._check_not_closed()
        self._check_main_thread()
        if self._stage1b_concepts is None:
            output_paths = self.output_paths()
            path = output_paths.stage1b_concepts()
            if not path.is_file():
                raise AxBenchBaseError(
                    f"Stage 1b filtered concepts not found at {path}. Run "
                    "Stage 1b (SYNTH_POSITIVE_MODE) first."
                )
            input_concepts = self.concepts()
            survivor_ids = load_concept_yield_surviving_ids(
                self.cfg,
                stage="stage1b",
                input_concepts=input_concepts,
                path=output_paths.stage1b_concept_yield_report(),
            )
            filtered = load_filtered_concepts(
                path,
                parent_concepts=input_concepts,
            )
            filtered_ids = frozenset(
                concept.concept_id for concept in filtered
            )
            if filtered_ids != survivor_ids:
                raise AxBenchBaseError(
                    "Stage 1b filtered concepts do not match "
                    f"{output_paths.stage1b_concept_yield_report()}"
                )
            self._stage1b_concepts = filtered
        return self._stage1b_concepts

    def downstream_concepts(self) -> list[Concept]:
        """Return the single filtered population consumed after synthesis."""

        self._check_not_closed()
        self._check_main_thread()
        if self._downstream_concepts is None:
            if not self.cfg.run_negative_mode:
                self._downstream_concepts = self.stage1b_concepts()
            else:
                output_paths = self.output_paths()
                path = output_paths.stage1c_concepts()
                if not path.is_file():
                    raise AxBenchBaseError(
                        f"Stage 1c filtered concepts not found at {path}. Run "
                        "Stage 1c (SYNTH_NEGATIVE_MODE) first."
                    )
                input_concepts = self.stage1b_concepts()
                survivor_ids = load_concept_yield_surviving_ids(
                    self.cfg,
                    stage="stage1c",
                    input_concepts=input_concepts,
                    path=output_paths.stage1c_concept_yield_report(),
                )
                filtered = load_filtered_concepts(
                    path,
                    parent_concepts=input_concepts,
                )
                filtered_ids = frozenset(
                    concept.concept_id for concept in filtered
                )
                if filtered_ids != survivor_ids:
                    raise AxBenchBaseError(
                        "Stage 1c filtered concepts do not match "
                        f"{output_paths.stage1c_concept_yield_report()}"
                    )
                self._downstream_concepts = filtered
        return self._downstream_concepts

    def seeds_train(self) -> list[SeedRow]:
        """Return the Stage 0b train seed pool.

        Reads the CSR-packed safetensors file on first call and caches
        the resulting list of :class:`SeedRow` instances.
        """
        return self._seeds_train_snapshot().rows

    def seeds_train_rows_sha256(self) -> str:
        """Return the ordered train-row identity from the cached seed load."""

        return self._seeds_train_snapshot().rows_sha256

    def _seeds_train_snapshot(self) -> LoadedSeedSplit:
        self._check_not_closed()
        self._check_main_thread()
        if self._seeds_train is None:
            self._seeds_train = self._load_seeds_train()
        return self._seeds_train

    def seeds_eval(self) -> list[SeedRow]:
        """Return the Stage 0b eval seed pool.

        Reads the CSR-packed safetensors file on first call and caches
        the resulting list of :class:`SeedRow` instances.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._seeds_eval is None:
            self._seeds_eval = self._load_seeds_eval()
        return self._seeds_eval

    def nanochat_tokenizer(self) -> Any:
        """Return the project's nanochat tokenizer.

        Loaded from disk on first call via
        :func:`nanochat.tokenizer.get_tokenizer`.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._nanochat_tokenizer is None:
            self._nanochat_tokenizer = self._load_nanochat_tokenizer()
        return self._nanochat_tokenizer

    def extraction_tokenizer(self) -> Any:
        """Return the HuggingFace tokenizer for the feature-extraction model.

        Loaded via :func:`transformers.AutoTokenizer.from_pretrained`
        using :attr:`AxBenchBaseConfig.extraction_tokenizer_id`. The
        default config points at ``"google/gemma-2-2b"`` but any valid
        HuggingFace model id is accepted.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._extraction_tokenizer is None:
            self._extraction_tokenizer = self._load_extraction_tokenizer()
        return self._extraction_tokenizer

    # -----------------------------------------------------------------
    # Heavy-model context managers
    #
    # All four follow the same pattern: on first entry the loader is
    # invoked (and the loader is responsible for placing the model on
    # ``self._device``); on re-entry after a full exit the cached model
    # is woken back up with ``.to(self._device)``; on final exit
    # (refcount drops to 0) the model is parked on CPU and the CUDA
    # allocator cache is emptied. Nested ``with`` blocks share the
    # single live instance -- the innermost exit just decrements the
    # refcount without moving anything.
    # -----------------------------------------------------------------

    def _ensure_feature_extraction_loaded(self) -> None:
        """Load the feature-extraction model + SAE pair on first access.

        Called by both :meth:`base_model` and :meth:`base_sae` so the
        single ``load_sae_and_model`` call is shared. The
        ``_feat_extraction_loaded`` flag prevents re-loading once both
        objects are cached.

        Sibling-acquisition contract
        ----------------------------
        The loader places BOTH the base_model and the base_sae on
        ``cfg.device``; only the triggering accessor's refcount bumps
        to 1, so the sibling sits live on device at refcount 0 until
        someone enters its accessor. In all current callers both
        siblings are always acquired together, so the idle-on-device
        window is bounded. If a future stage enters only one, the
        sibling will leak on GPU until :meth:`close` — pin the
        "both-together" contract there with an assert before
        introducing such a caller.

        First-entry wake-up note
        ------------------------
        The sibling's first context-manager entry observes
        ``refcount == 0`` AND ``self._base_sae is not None`` (the
        loader already populated it), so the accessor's
        ``elif refcount == 0`` wake-up branch fires and re-moves the
        already-on-device tensor to the same device. This is a no-op
        in PyTorch but still re-walks the parameter list (<1 ms for a
        2 B-param base model + 16 k SAE). Pinned by the integration
        test's ``base_sae`` CPU-park count being 1 higher than
        ``base_model``. Acceptable; not worth a flag + branch in every
        accessor to avoid.
        """
        if not self._feat_extraction_loaded:
            with profile_region("runtime.load_feature_extraction_pair"):
                if self.cluster_contract is None:
                    model, sae = _load_feature_extraction_pair(self.cfg)
                else:
                    model, sae = _load_feature_extraction_pair(
                        self.cfg,
                        cluster_contract=self.cluster_contract,
                    )
            self._base_model = model
            self._base_sae = sae
            self._feat_extraction_loaded = True

    @contextlib.contextmanager
    def base_model(self) -> Iterator[Any]:
        """Yield the feature-extraction base model live on :attr:`_device`.

        On exit at refcount 0 the model is moved to CPU and the CUDA
        allocator cache is emptied (if CUDA is available). The model
        instance is kept as a cached attribute, so re-entering the
        context manager re-uses the same loaded weights rather than
        calling the loader a second time.

        The model and SAE are loaded together via a single call to
        :func:`_load_feature_extraction_pair` the first time either
        :meth:`base_model` or :meth:`base_sae` is entered.

        Yields:
            The feature-extraction base model. Concrete type is the
            return type of :func:`_load_feature_extraction_pair` (an
            nnterp ``StandardizedTransformer`` in production).

        Raises:
            AxBenchBaseError: if :meth:`close` has been called, if
                invoked from a worker thread, or if the loader raises
                (the underlying exception is included as ``__cause__``).
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._base_model is None:
            self._ensure_feature_extraction_loaded()
        elif self._base_model_refcount == 0:
            # Cached instance is parked on CPU -- wake it back up.
            with profile_region("runtime.base_model.to_device", device=self._device):
                self._base_model.to(self._device)
        self._base_model_refcount += 1
        try:
            yield self._base_model
        finally:
            self._base_model_refcount -= 1
            if self._base_model_refcount == 0:
                with profile_region("runtime.base_model.to_cpu"):
                    self._base_model.to("cpu")
                if t.cuda.is_available():
                    with profile_region("runtime.cuda.empty_cache"):
                        t.cuda.empty_cache()

    @contextlib.contextmanager
    def base_sae(self) -> Iterator[Any]:
        """Yield the base SAE live on :attr:`_device`.

        Same refcount + CPU-offload contract as :meth:`base_model`.
        The model and SAE are loaded together via a single call to
        :func:`_load_feature_extraction_pair` the first time either
        :meth:`base_model` or :meth:`base_sae` is entered.

        Yields:
            The base SAE. Concrete type is an ``sae_lens.SAE`` in
            production.

        Raises:
            AxBenchBaseError: if closed, off-thread, or loader failure.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._base_sae is None:
            self._ensure_feature_extraction_loaded()
        elif self._base_sae_refcount == 0:
            with profile_region("runtime.base_sae.to_device", device=self._device):
                self._base_sae.to(self._device)
        self._base_sae_refcount += 1
        try:
            yield self._base_sae
        finally:
            self._base_sae_refcount -= 1
            if self._base_sae_refcount == 0:
                with profile_region("runtime.base_sae.to_cpu"):
                    self._base_sae.to("cpu")
                if t.cuda.is_available():
                    with profile_region("runtime.cuda.empty_cache"):
                        t.cuda.empty_cache()

    @contextlib.contextmanager
    def decoder_only_model(self) -> Iterator[Any]:
        """Yield the nanochat decoder-only target model live on device.

        Same refcount + CPU-offload contract as :meth:`base_model`.
        Loader reads ``cfg.decoder_only`` (``source``, ``model_tag``,
        ``step``).

        Yields:
            The decoder-only GPT. Concrete type is the return type of
            :func:`_load_decoder_only_model` (a :class:`nanochat.gpt.GPT`
            in production).

        Raises:
            AxBenchBaseError: if closed, off-thread, or loader failure.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._decoder_only_model is None:
            with profile_region("runtime.load_decoder_only_model"):
                if self.cluster_contract is None:
                    self._decoder_only_model = _load_decoder_only_model(self.cfg)
                else:
                    self._decoder_only_model = _load_decoder_only_model(
                        self.cfg,
                        self.cluster_contract,
                    )
        elif self._decoder_only_refcount == 0:
            with profile_region(
                "runtime.decoder_only_model.to_device",
                device=self._device,
            ):
                self._decoder_only_model.to(self._device)
        self._decoder_only_refcount += 1
        try:
            yield self._decoder_only_model
        finally:
            self._decoder_only_refcount -= 1
            if self._decoder_only_refcount == 0:
                with profile_region("runtime.decoder_only_model.to_cpu"):
                    self._decoder_only_model.to("cpu")
                if t.cuda.is_available():
                    with profile_region("runtime.cuda.empty_cache"):
                        t.cuda.empty_cache()

    @contextlib.contextmanager
    def encdec_model(self) -> Iterator[Any]:
        """Yield the nanochat encoder-decoder target model live on device.

        Same refcount + CPU-offload contract as :meth:`base_model`.
        Loader reads ``cfg.encdec`` (``source``, ``model_tag``,
        ``step``). The loader also attaches the checkpoint's
        ``meta_data`` dict to the returned model as
        ``model._loaded_meta_data`` so downstream inference code that
        needs tokenizer configuration at eval time can recover it
        without re-reading the checkpoint.

        Yields:
            The encoder-decoder GPT with ``_loaded_meta_data``
            attached.

        Raises:
            AxBenchBaseError: if closed, off-thread, or loader failure.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._encdec_model is None:
            with profile_region("runtime.load_encdec_model"):
                if self.cluster_contract is None:
                    self._encdec_model = _load_encdec_model(self.cfg)
                else:
                    self._encdec_model = _load_encdec_model(
                        self.cfg,
                        self.cluster_contract,
                    )
        elif self._encdec_refcount == 0:
            with profile_region("runtime.encdec_model.to_device", device=self._device):
                self._encdec_model.to(self._device)
        self._encdec_refcount += 1
        try:
            yield self._encdec_model
        finally:
            self._encdec_refcount -= 1
            if self._encdec_refcount == 0:
                with profile_region("runtime.encdec_model.to_cpu"):
                    self._encdec_model.to("cpu")
                if t.cuda.is_available():
                    with profile_region("runtime.cuda.empty_cache"):
                        t.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Metas
    #
    # Metas are plain :class:`TargetModelMetadata` dataclass instances
    # that describe the target model's training-time bucket + selection
    # semantics. Detection requires a live model context because
    # :func:`detect_target_model_metadata` reads attributes directly off
    # the model (``encoder.config``, ``_loaded_meta_data``). The meta
    # accessor enters the corresponding model context internally on the
    # first call, caches the result, and returns the cached dataclass
    # on every subsequent call without touching the model at all.
    # -----------------------------------------------------------------

    def decoder_only_meta(self) -> TargetModelMetadata:
        """Return the detected metadata for the decoder-only target.

        On first call, briefly enters :meth:`decoder_only_model` so the
        detector can read attributes off a live model, then caches the
        resulting :class:`TargetModelMetadata`. Subsequent calls return
        the cached instance without entering the model context.

        Raises:
            AxBenchBaseError: if the state is closed, called off the
                main thread, or :func:`detect_target_model_metadata`
                raises. The underlying exception is chained as
                ``__cause__``.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._decoder_only_meta is None:
            with self.decoder_only_model() as model:
                try:
                    self._decoder_only_meta = detect_target_model_metadata(
                        target_model=model,
                        cfg=self.cfg,
                    )
                except AxBenchBaseError:
                    raise
                except Exception as exc:
                    raise AxBenchBaseError(
                        f"Failed to detect decoder-only target model metadata: {exc}"
                    ) from exc
        return self._decoder_only_meta

    def encdec_meta(self) -> TargetModelMetadata:
        """Return the detected metadata for the encoder-decoder target.

        On first call, briefly enters :meth:`encdec_model` so the
        detector can read attributes off a live model, then caches the
        resulting :class:`TargetModelMetadata`. Subsequent calls return
        the cached instance without entering the model context.

        Raises:
            AxBenchBaseError: if the state is closed, called off the
                main thread, or :func:`detect_target_model_metadata`
                raises. The underlying exception is chained as
                ``__cause__``.
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._encdec_meta is None:
            with self.encdec_model() as model:
                try:
                    if self.cluster_contract is None:
                        self._encdec_meta = detect_target_model_metadata(
                            target_model=model,
                            cfg=self.cfg,
                        )
                    else:
                        self._encdec_meta = detect_target_model_metadata(
                            target_model=model,
                            cfg=self.cfg,
                            cluster_contract=self.cluster_contract,
                        )
                except AxBenchBaseError:
                    raise
                except Exception as exc:
                    raise AxBenchBaseError(
                        f"Failed to detect encdec target model metadata: {exc}"
                    ) from exc
        return self._encdec_meta

    # -----------------------------------------------------------------
    # Judge client
    # -----------------------------------------------------------------

    def judge_client(self) -> AxBenchJudgeClient:
        """Return the OpenRouter judge client. Lazily constructed on first access.

        Uses :attr:`AxBenchBaseConfig.judge` plus the invocation path contract
        so the client shares a per-run SQLite cache with other callers.

        Raises:
            AxBenchBaseError: if closed, off-thread, if ``cfg.run_tag``
                is unset, or if :class:`AxBenchJudgeClient` raises on
                construction (e.g., missing OpenRouter API key).
        """
        self._check_not_closed()
        self._check_main_thread()
        if self._judge_client is None:
            run_tag = self._require_run_tag()
            try:
                self._judge_client = AxBenchJudgeClient(
                    self.cfg.judge,
                    cache_path=self.output_paths().judge_cache(),
                )
            except AxBenchBaseError:
                raise
            except Exception as exc:
                raise AxBenchBaseError(
                    f"Failed to construct judge client for run_tag {run_tag!r}: {exc}"
                ) from exc
        return self._judge_client

    # -----------------------------------------------------------------
    # Inactive sampler
    #
    # Derived from :meth:`encdec_meta`. Returns ``None`` when the
    # encdec target was not trained with negative feature sampling.
    # The ``_inactive_sampler_loaded`` sentinel distinguishes "not yet
    # loaded" from "loaded and the value is None" so we never
    # re-invoke the builder on the ``None`` branch.
    # -----------------------------------------------------------------

    def inactive_sampler(self) -> Callable[..., dict[int, int]] | None:
        """Return the inactive-feature sampler for negative-mode methods.

        ``None`` if the encdec target was not trained with negative
        feature sampling (``encdec_meta().negative_selection is None``).
        Otherwise a callable with the signature
        ``(*, rng: random.Random) -> dict[int, int]`` suitable for
        forwarding to :func:`build_mode_aware_encoder_input`.

        On first call, enters :meth:`encdec_meta` (which in turn
        briefly wakes the encdec model) and inspects the training-time
        negative selection. Cached thereafter: the ``None`` return
        value is cached too, via the ``_inactive_sampler_loaded``
        sentinel.

        Raises:
            AxBenchBaseError: if closed, off-thread, or if
                :func:`_build_inactive_sampler` raises (parse failure,
                missing ``num_buckets``, stats-file read errors, etc.).
        """
        self._check_not_closed()
        self._check_main_thread()
        if not self._inactive_sampler_loaded:
            encdec_meta = self.encdec_meta()
            if encdec_meta.negative_selection is None:
                self._inactive_sampler = None
            else:
                try:
                    if self.cluster_contract is None:
                        # Preserve the legacy call exactly. In particular, do
                        # not add a semantically-empty keyword to the
                        # unclustered path: tests and downstream wrappers are
                        # entitled to the original callable contract.
                        self._inactive_sampler = _build_inactive_sampler(
                            encdec_meta=encdec_meta,
                            cfg=self.cfg,
                        )
                    else:
                        self._inactive_sampler = _build_inactive_sampler(
                            encdec_meta=encdec_meta,
                            cfg=self.cfg,
                            allowed_feature_ids=t.tensor(
                                self.cluster_contract.representative_universe,
                                dtype=t.int64,
                            ),
                        )
                except AxBenchBaseError:
                    raise
                except Exception as exc:
                    raise AxBenchBaseError(
                        f"Failed to build inactive sampler from encdec meta: {exc}"
                    ) from exc
            self._inactive_sampler_loaded = True
        return self._inactive_sampler

    # -----------------------------------------------------------------
    # Private loaders
    # -----------------------------------------------------------------

    def _require_run_tag(self) -> str:
        if self.cfg.run_tag is None:
            raise AxBenchBaseError(
                "cfg.run_tag must be set before accessing run-tag-scoped "
                "resources. The CLI normally auto-generates one via "
                "scripts.eval.axbench_base_eval._generate_run_tag; explicit "
                "callers should set cfg.run_tag before constructing a "
                "RuntimeState."
            )
        return self.cfg.run_tag

    def _load_concepts(self) -> list[Concept]:
        self._require_run_tag()
        path = self.output_paths().concepts_json()
        if not path.exists():
            raise AxBenchBaseError(
                f"concepts.json not found at {path}. Run Stage 0c "
                "(PREPARE_NEGATIVE_PREFIXES) first."
            )
        n_sae_features = read_n_sae_features_from_stats(
            Path(self.cfg.feature_statistics_path),
        )
        try:
            return load_and_validate_concepts(
                path,
                n_sae_features=n_sae_features,
            )
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(f"Failed to load concepts from {path}: {exc}") from exc

    def _load_seeds_train(self) -> LoadedSeedSplit:
        self._require_run_tag()
        path = self.output_paths().seeds_train()
        if not path.exists():
            raise AxBenchBaseError(
                f"seeds_train safetensors not found at {path}. Run "
                "Stage 0b (PREPARE_SEED_POOLS) first."
            )
        try:
            tokenizer = _nanochat_get_tokenizer()
            return load_seed_pool_with_rows_identity(
                path,
                expected_split="train",
                expectation=expectation_from_config(self.cfg, tokenizer),
                tokenizer=tokenizer,
            )
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(f"Failed to load seeds_train from {path}: {exc}") from exc

    def _load_seeds_eval(self) -> list[SeedRow]:
        self._require_run_tag()
        path = self.output_paths().seeds_eval()
        if not path.exists():
            raise AxBenchBaseError(
                f"seeds_eval safetensors not found at {path}. Run "
                "Stage 0b (PREPARE_SEED_POOLS) first."
            )
        try:
            tokenizer = _nanochat_get_tokenizer()
            return load_seed_pool(
                path,
                expected_split="eval",
                expectation=expectation_from_config(self.cfg, tokenizer),
                tokenizer=tokenizer,
            )
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(
                f"Failed to load seeds_eval from {path}: {exc}"
            ) from exc

    def _load_nanochat_tokenizer(self) -> Any:
        try:
            return _nanochat_get_tokenizer()
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(
                f"Failed to load nanochat tokenizer: {exc}. The nanochat "
                "tokenizer is usually bundled under $NANOCHAT_BASE_DIR/"
                "tokenizer/; verify the base dir is set correctly and "
                "that the tokenizer artifacts have been downloaded."
            ) from exc

    def _load_extraction_tokenizer(self) -> Any:
        model_id = self.cfg.extraction_tokenizer_id
        try:
            return AutoTokenizer.from_pretrained(model_id)
        except AxBenchBaseError:
            raise
        except Exception as exc:
            raise AxBenchBaseError(
                f"Failed to load extraction tokenizer "
                f"{model_id!r}: {exc}. The HuggingFace tokenizer is "
                f"fetched via transformers.AutoTokenizer.from_pretrained; "
                f"verify the model id is reachable (HF Hub or a local "
                f"cache) and that the transformers package is installed."
            ) from exc
