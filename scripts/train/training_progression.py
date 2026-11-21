"""Training progression evaluation: collect metrics across checkpoints and plot.

Evaluates checkpoints across a model's training trajectory (pretraining + SFT),
saves results incrementally to a JSON file, and generates line charts showing
mean +/- std over training.

By default, reads checkpoints from the local filesystem.  Set
``checkpoint_source=remote`` to pull checkpoints one-at-a-time from a remote
machine via SCP (useful when checkpoints don't fit locally).

Usage:
    uv run python -m scripts.train.training_progression mode=collect
    uv run python -m scripts.train.training_progression mode=plot
    uv run python -m scripts.train.training_progression mode=both
    uv run python -m scripts.train.training_progression mode=collect checkpoint_source=remote
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import steering.eval.feature_config as _feature_config
import steering.eval.harmfulness_judge as harmfulness_judge
from nanochat.checkpoint_manager import (
    ModelPhase,
    build_encdec_model,
    meta_looks_encdec,
)
from nanochat.checkpoint_manager import (
    build_model as build_decoder_model,
)
from nanochat.common import autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.hydra_cli import parse_config_name_and_overrides
from scripts.eval.base_eval import (
    _run_core_target_real_eval as _base_eval_run_core_target_real,
)
from scripts.eval.base_eval import (
    evaluate_core as _base_eval_evaluate_core,
)
from scripts.eval.chat_eval import (
    ALL_CHAT_TASKS as _CHAT_ALL_TASKS,
)
from scripts.eval.chat_eval import (
    BASELINE_ACCURACIES as _CHAT_BASELINE_ACCURACIES,
)
from scripts.eval.chat_eval import (
    _chat_eval_target_real_per_config,
)
from scripts.eval.chat_eval import (
    run_chat_eval as _chat_eval_run_chat_eval,
)
from scripts.eval.validate_eval_context_cache import enumerate_expected_eval_contexts
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.chat_generation import (
    render_prompt_for_chat_completion,
)
from steering.eval.compacted_eval_loading import (
    assert_conditioned_eval_model,
    conditioned_eval_cache_contract_metadata,
    conditioned_eval_conditioning,
    conditioned_eval_feature_vocab,
)
from steering.eval.context_cache_io import (
    CONTEXT_LEN_DIR_RE,
    MANIFEST_FILENAME,
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    feature_selection_hash_leaf,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.feature_config import (
    FeatureConfig,
    parse_strict_feature_configs,
    require_nonempty_feature_configs,
    resolve_next_segment_conditioning_offset,
)
from steering.eval.harmfulness_artifact import (
    HARMFULNESS_EVAL_SCHEMA_VERSION,
    build_harmfulness_payload,
    build_judge_provenance,
    validate_harmfulness_contract,
)
from steering.eval.harmfulness_generation import (
    SELF_CONDITIONED_SEED_SCHEDULE,
    STRENGTH_SWEEP_SPEC_SHA256_KEY,
    absolute_prompt_batch_seed,
    capture_file_inventory,
    recursive_file_paths,
    validate_harmfulness_completion_provenance,
)
from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.eval.harmfulness_self_conditioning import (
    generate_harmfulness_response_self_conditioned,
    harmfulness_completion_provenance,
    parse_harmfulness_safety_settings,
    prepare_harmfulness_self_conditioning,
)
from steering.eval.inference_engine import InferenceEngine, build_wrapper_for_features
from steering.eval.progression_plotting import (
    BASELINE_COLOR,
    BASELINE_LINESTYLE,
    ChartSeries,
    MultiSeriesSpec,
    _phase_boundaries_from_series,
    build_multi_series_spec,
    render_multi_series_chart,
)
from steering.eval.progression_results import SuiteResultKind, load_suite_from_dict
from steering.eval.prompt_coverage import assert_exact_prompt_coverage
from steering.eval.result_plot_dispatch import _get_linestyle_for_variant
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    PooledTargetRealConfig,
    TargetConditioningMode,
    context_cache_generation_instructions,
    validate_feature_dict,
)
from steering.eval.target_real_runner import parse_target_real_configs_from_cfg
from steering.immutable_io import stream_hash_file_descriptor_bound
from steering.overlay_io import COMPRESSED_EXT
from steering.profiling.markers import timed_region, timing_tags

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "training_progression"
_DECODER_WORK_VARIANT = "decoder"
_legacy_checkpoint_identity_warned = False
# Version 2 binds the complete-transcript target-aware decoding contract from
# issue #1339. Version-1 artifacts were produced when forced Engine tokens were
# discarded and must not be resumed under the current evaluator.
_HARMFULNESS_PROGRESSION_SPEC_VERSION = 2
_SELF_CONDITIONED_CONFIG_NAME = SuiteResultKind.SELF_CONDITIONED.value
_SELF_CONDITIONED_CACHE_DOMAIN = "harmfulness_progression_context_cache_v1"


class _ModelFamily(StrEnum):
    AUTO = "auto"
    DECODER = "decoder"
    ENCODER_DECODER = "encoder_decoder"


def _model_family(cfg: DictConfig) -> _ModelFamily:
    raw = str(cfg.model_family).lower()
    try:
        return _ModelFamily(raw)
    except ValueError as exc:
        raise ValueError(
            f"model_family must be one of auto, decoder, encoder_decoder; got {raw!r}"
        ) from exc


def _variant_linestyle_resolver(config_name: str, config_entry: dict) -> tuple[str, str]:
    """Dispatch linestyle/color for a conditioned config via PR-D variant types.

    Builds a ``SuiteResultVariant`` from the raw config entry dict (using
    ``load_suite_from_dict``) and returns ``(linestyle, color)`` via
    ``_get_linestyle_for_variant``. Entries without an explicit ``kind`` are
    rejected by ``load_suite_from_dict``.
    """
    variant = load_suite_from_dict(config_entry, name=config_name)
    return _get_linestyle_for_variant(variant)


# ---------------------------------------------------------------------------
# Checkpoint specification
# ---------------------------------------------------------------------------


@dataclass
class CheckpointSpec:
    step: int
    phase: str  # "pretrain" or "sft"
    subdir: str  # relative path under base_checkpoints/model_tag (e.g. "" or "standard_sft_checkpoints")
    checkpoint_identity: dict[str, Any] | None = None


def build_checkpoint_specs(cfg: DictConfig) -> list[CheckpointSpec]:
    """Build ordered list of checkpoint specs from config phases."""
    specs: list[CheckpointSpec] = []
    for phase_cfg in cfg.phases:
        for step in phase_cfg.steps:
            specs.append(
                CheckpointSpec(
                    step=int(step),
                    phase=str(phase_cfg.phase),
                    subdir=str(phase_cfg.subdir),
                )
            )
    unknown = set(_checkpoint_overrides(cfg)) - {(spec.phase, spec.step) for spec in specs}
    if unknown:
        raise ValueError(f"self_conditioned_overrides reference unconfigured checkpoints: {sorted(unknown)}")
    return specs


# ---------------------------------------------------------------------------
# Encoder-decoder conditioning config helpers
# ---------------------------------------------------------------------------


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _conditioning_cfg(cfg: DictConfig) -> dict:
    raw = cfg.get("conditioning")
    if raw is None:
        return {}
    plain = _plain(raw)
    return plain if isinstance(plain, dict) else {}


def _self_conditioned_cfg(cfg: DictConfig) -> dict:
    raw = _plain(cfg.get("self_conditioned"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("self_conditioned must be a mapping")
    if set(raw) != {"safety_steering"}:
        raise ValueError("self_conditioned requires exactly safety_steering (null or targets/strength)")
    return raw


def _checkpoint_overrides(cfg: DictConfig) -> dict[tuple[str, int], dict]:
    overrides = {}
    for override in cfg.get("self_conditioned_overrides", []) or []:
        plain = _plain(override)
        if not isinstance(plain, dict) or set(plain) != {"phase", "step", "config"}:
            raise ValueError("self_conditioned_overrides entries require phase, step, config")
        if not isinstance(plain["config"], dict):
            raise TypeError("self_conditioned_overrides.config must be a mapping")
        if set(plain["config"]) - {
            "self_conditioned", "conditioning", "stats_path", "self_cond_segment_size",
            "extraction_tokenizer_id",
        }:
            raise ValueError("self_conditioned_overrides.config accepts only runtime/target inputs")
        if isinstance(plain["step"], bool) or not isinstance(plain["step"], int):
            raise TypeError("self_conditioned_overrides.step must be an integer")
        if not isinstance(plain["phase"], str) or not plain["phase"].strip():
            raise ValueError("self_conditioned_overrides.phase must be nonblank text")
        key = (plain["phase"], plain["step"])
        if key in overrides:
            raise ValueError(f"Duplicate self_conditioned override for {key}")
        overrides[key] = plain["config"]
    return overrides


def _cfg_for_checkpoint(cfg: DictConfig, spec: CheckpointSpec) -> DictConfig:
    """Resolve explicit per-checkpoint cache/target overrides before scheduling."""
    overrides = _checkpoint_overrides(cfg)
    key = (spec.phase, spec.step)
    if key not in overrides:
        return cfg
    resolved = copy.deepcopy(cfg)
    resolved.merge_with(overrides[key])
    return resolved


def _validate_harmfulness_conditioning(cfg: DictConfig, phase: str) -> None:
    if _raw_conditioned_configs_for_phase(cfg, phase):
        raise ValueError(
            "Encoder-decoder harmfulness progression no longer accepts hand-supplied "
            "conditioning.conditioned_configs feature maps; use "
            "self_conditioned.safety_steering={targets: PATH, strength: NUMBER}. "
            "Run legacy static CORE/chat variants in a separate invocation."
        )
    if not _self_conditioned_cfg(cfg):
        raise ValueError(
            "Encoder-decoder harmfulness progression requires "
            "self_conditioned={safety_steering: null} for unsteered feedback, "
            "or self_conditioned.safety_steering={targets: PATH, strength: NUMBER}"
        )


def _validate_no_legacy_conditioning_modes(cfg: DictConfig) -> None:
    """Reject old no-context switches; progression eval is always full-context."""

    cond = _conditioning_cfg(cfg)
    if "require_context_features" in cond:
        raise ValueError(
            "conditioning.require_context_features is no longer supported. "
            "Encoder-decoder progression eval always requires full cached "
            "prompt/context features."
        )
    if "context_mode" in cond:
        raise ValueError(
            "conditioning.context_mode is no longer supported. Encoder-decoder "
            "progression eval always uses full cached prompt/context features."
        )
    if "run_unconditioned" in cond:
        raise ValueError(
            "conditioning.run_unconditioned is no longer supported; "
            "encoder-decoder progression eval requires configured full-context "
            "conditioning variants."
        )


def _context_cache_identity_for_cfg(cfg: DictConfig):
    cond = _conditioning_cfg(cfg)
    raw = cond.get("identity")
    if raw is None:
        raise ValueError(
            "conditioning.identity is required when using a context feature cache "
            "(model_id/sae_id/layer/pooling/feature_selection_hash)"
        )
    if not isinstance(raw, dict):
        raise TypeError("conditioning.identity must resolve to a mapping")
    return cache_identity_from_config(raw, cond)


def _raw_conditioned_configs_for_phase(cfg: DictConfig, phase: str):
    cond = _conditioning_cfg(cfg)
    raw = cond.get("conditioned_configs")
    if raw is None:
        return None
    raw = _plain(raw)
    if isinstance(raw, dict):
        # Dicts with feature keys are inline global configs.  Otherwise the
        # mapping is interpreted as phase -> YAML/list.
        feature_keys = {"features", "features_file", "name"}
        if any(key in raw for key in feature_keys):
            return raw
        return raw.get(phase)
    return raw


def conditioned_configs_for_phase(
    cfg: DictConfig,
    phase: str,
) -> list[FeatureConfig]:
    """Resolve explicit static target-delta feature configs for a phase."""

    configs = parse_strict_feature_configs(_raw_conditioned_configs_for_phase(cfg, phase))
    require_nonempty_feature_configs(
        configs,
        surface=f"training_progression.conditioned_configs[{phase}]",
    )
    return configs


def _context_provider_for_cfg(
    cfg: DictConfig,
    *,
    model_max_seq_len: int,
    expected_n_sae_features: int | None = None,
    expected_num_buckets: int | None = None,
    expected_feature_vocab: FeatureVocab | None = None,
    expected_contract_metadata: Mapping[str, Any] | None = None,
) -> SafetensorsContextFeatureProvider:
    cond = _conditioning_cfg(cfg)
    path = cond.get("context_cache_dir")
    if not path:
        raise RuntimeError(
            "conditioned_configs require conditioning.context_cache_dir; "
            f"{context_cache_generation_instructions()}"
        )
    identity = _context_cache_identity_for_cfg(cfg)
    cache_dir = context_length_cache_dir(
        path,
        model_max_seq_len,
        hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
    )
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=expected_n_sae_features,
        expected_num_buckets=expected_num_buckets,
        expected_feature_vocab=expected_feature_vocab,
        expected_metadata={
            "model_max_seq_len": int(model_max_seq_len),
            **dict(expected_contract_metadata or {}),
        },
    )


def _target_real_providers_for_cfg(
    cfg,
    *,
    model_max_seq_len: int,
    target_real_configs,
    expected_feature_vocab: FeatureVocab | None = None,
    expected_contract_metadata: Mapping[str, Any] | None = None,
) -> dict[str, SafetensorsContextFeatureProvider]:
    """Construct one SafetensorsContextFeatureProvider per unique cache_dir
    referenced by parsed target_real_configs. Configs sharing a cache_dir share
    the provider instance.

    Returns a dict mapping config-name -> provider so callers can look up
    by name without re-checking cache_dir equality. Providers are constructed
    per checkpoint because their validation is bound to that checkpoint's
    effective feature vocabulary and mask/cluster contract.

    Args:
        cfg: Hydra config carrying ``conditioning.identity`` etc.
        model_max_seq_len: context length of the loaded checkpoint, used
            both for cache key construction (when applicable) and for the
            provider's ``expected_metadata`` validation.
        target_real_configs: Pre-parsed list of
            ``LockstepTargetRealConfig`` / ``PooledTargetRealConfig``
            instances. Entries with null/missing cache dirs must already
            have been auto-resolved by ``parse_target_real_configs_from_cfg``.
    """
    cond = _conditioning_cfg(cfg)
    trc_list = list(target_real_configs)
    if not trc_list:
        return {}
    if not cond:
        raise ValueError("target_real provider construction requires cfg.conditioning")

    identity = _context_cache_identity_for_cfg(cfg)
    n_sae_features = int(cond.get("n_sae_features", 16384))
    # H7 fix: default num_buckets to 2 (binary-bucket pipeline) to match
    # build_context_feature_provider's default and the YAML defaults.
    num_buckets = int(cond.get("num_buckets", 2))

    by_cache_dir: dict[str, SafetensorsContextFeatureProvider] = {}
    by_name: dict[str, SafetensorsContextFeatureProvider] = {}
    for trc in trc_list:
        name = str(trc.name)
        cache_dir_path = Path(str(trc.cache_dir))
        cache_key = str(cache_dir_path)
        full_cache_dir = context_length_cache_dir(
            cache_key,
            model_max_seq_len,
            hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash),
        )
        if cache_key not in by_cache_dir:
            by_cache_dir[cache_key] = SafetensorsContextFeatureProvider(
                full_cache_dir,
                identity=identity,
                expected_n_sae_features=n_sae_features,
                expected_num_buckets=num_buckets,
                expected_feature_vocab=expected_feature_vocab,
                expected_metadata={
                    "model_max_seq_len": int(model_max_seq_len),
                    **dict(expected_contract_metadata or {}),
                },
            )
        by_name[name] = by_cache_dir[cache_key]
    return by_name


def _context_cache_dirs_under_root(cache_root: str | Path) -> list[Path]:
    root = Path(cache_root)
    if CONTEXT_LEN_DIR_RE.fullmatch(root.name):
        raise ValueError(
            "conditioning.context_cache_dir must point at the cache root, not "
            f"the context-length directory {root}. Use {root.parent} so "
            "progression can resolve c{model.max_seq_len}/{hash[:16]} for each "
            "checkpoint."
        )
    if not root.exists():
        raise FileNotFoundError(
            f"Context cache root not found: {root}. Expected context-length "
            "subdirectories such as c1024/<hash>/manifest.json or "
            "c2048/<hash>/manifest.json.\n\n"
            f"{context_cache_generation_instructions()}"
        )
    if not root.is_dir():
        raise NotADirectoryError(f"Context cache root is not a directory: {root}")
    return sorted(
        leaf
        for context_dir in root.iterdir()
        if context_dir.is_dir() and CONTEXT_LEN_DIR_RE.fullmatch(context_dir.name)
        for leaf in context_dir.iterdir()
        if leaf.is_dir() and (leaf / "manifest.json").is_file()
    )


def _conditioned_phases_requiring_context_cache(
    cfg: DictConfig,
    specs: list[CheckpointSpec],
) -> list[str]:
    phases: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.phase in seen:
            continue
        if conditioned_configs_for_phase(cfg, spec.phase):
            phases.append(spec.phase)
            seen.add(spec.phase)
    return phases


def _validate_required_context_cache(
    cfg: DictConfig,
    specs: list[CheckpointSpec],
) -> None:
    """Fail before starting conditioned evals when the context cache is absent."""
    phases = _conditioned_phases_requiring_context_cache(cfg, specs)
    if not phases:
        return

    path = _conditioning_cfg(cfg).get("context_cache_dir")
    if not path:
        raise RuntimeError(
            "conditioned progression eval requires conditioning.context_cache_dir "
            f"for phase(s): {', '.join(phases)}.\n\n"
            f"{context_cache_generation_instructions()}"
        )

    cache_dirs = _context_cache_dirs_under_root(path)
    if not cache_dirs:
        cache_root = Path(path)
        raise FileNotFoundError(
            f"No context-length cache manifest found under {path}. Expected "
            "the cache builder to write manifest.json under "
            f"{cache_root / 'c1024' / '<hash[:16]>'}/ or "
            f"{cache_root / 'c2048' / '<hash[:16]>'}/, depending on the "
            "checkpoint metadata max_seq_len/model.max_seq_len.\n\n"
            f"{context_cache_generation_instructions()}"
        )

    nonempty_dirs = []
    identity = _context_cache_identity_for_cfg(cfg)
    for cache_dir in cache_dirs:
        context_len = int(cache_dir.parent.name[1:])
        provider = SafetensorsContextFeatureProvider(
            cache_dir,
            identity=identity,
            expected_metadata={"model_max_seq_len": context_len},
        )
        if len(provider) > 0:
            nonempty_dirs.append(cache_dir)
    if not nonempty_dirs:
        raise RuntimeError(
            "All context-length feature caches under "
            f"{path} are empty: {', '.join(str(p) for p in cache_dirs)}\n\n"
            f"{context_cache_generation_instructions()}"
        )


def _is_encoder_decoder_model(model) -> bool:
    return isinstance(model, EncoderDecoderGPT)


def _validate_conditioned_configs(
    model,
    configs: list[FeatureConfig],
    meta: dict,
) -> None:
    require_nonempty_feature_configs(configs, surface="training_progression")
    enc_cfg = model.encoder.config
    _feature_config.validate_inactive_bucket_feature_configs(
        configs,
        num_buckets=enc_cfg.num_buckets,
        checkpoint_meta=meta,
    )
    for feature_cfg in configs:
        validate_feature_dict(
            feature_cfg.features,
            enc_cfg.n_sae_features,
            enc_cfg.num_buckets,
            name=feature_cfg.name,
            feature_vocab=getattr(model, "feature_vocab", None),
        )


def _assert_static_conditioning_context_coverage(
    *,
    cfg: DictConfig,
    tokenizer,
    provider: SafetensorsContextFeatureProvider,
    eval_suites: list[str],
    seeds: list[int],
    model_max_seq_len: int,
    surface: str,
) -> None:
    """Reject an incomplete static cache before the first scored example."""

    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_cfg, dict):
        raise TypeError("progression config must resolve to a mapping")
    coverage_cfg = OmegaConf.create(raw_cfg)
    coverage_cfg.model_max_seq_len = model_max_seq_len
    records: list[EvalContextRecord] = []
    for suite in eval_suites:
        if suite not in {"core", "chat", "harmfulness"}:
            continue
        for seed in _eval_seeds_for_suite(suite, seeds):
            coverage_cfg.evals = suite
            coverage_cfg.base_seed = seed
            coverage_cfg.num_seeds = 1
            records.extend(enumerate_expected_eval_contexts(coverage_cfg, tokenizer))
    if records:
        assert_exact_prompt_coverage(provider, records, surface=surface)


def _wrap_conditioned_model(
    model,
    tokenizer,
    feature_cfg: FeatureConfig,
    context_provider: SafetensorsContextFeatureProvider,
    target_mode: TargetConditioningMode,
):
    if not _is_encoder_decoder_model(model):
        raise ValueError("Conditioned progression evals require an EncoderDecoderGPT checkpoint")
    return build_wrapper_for_features(
        model,
        tokenizer,
        features=feature_cfg.features,
        context_provider=context_provider,
        target_mode=target_mode,
    )


def _run_eval_suite_target_real(
    *,
    eval_name: str,
    model,
    tokenizer,
    device,
    seed: int,
    cfg: DictConfig,
    trc,
    provider: SafetensorsContextFeatureProvider,
    target_mode: TargetConditioningMode = TargetConditioningMode.CURRENT,
) -> dict | None:
    """Run one eval suite for one target_real config and one seed (PR-D
    Batch H2 — load-bearing replacement for the prior stub).

    Iterates examples in the suite. For each example, the
    :func:`build_target_real_artifacts_for_example` helper rebuilds a fresh
    wrapper with an example-specific ``target=``. The example's eval is then
    dispatched via the same primitives ``_run_eval_suite`` uses (categorical /
    generative for chat, single-task CORE eval for core).

    Returns the same per-suite result-dict shape ``_run_eval_suite``
    returns. Returns ``None`` for unsupported eval suites (the caller
    skips writing).
    """
    if not _is_encoder_decoder_model(model):
        raise ValueError("target_real_configs require an EncoderDecoderGPT checkpoint")
    if eval_name == "core":
        with timed_region("run_core_target_real_eval"):
            return _run_core_target_real_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                cfg=cfg,
                trc=trc,
                provider=provider,
                target_mode=target_mode,
                seed=seed,
            )
    if eval_name == "chat":
        with timed_region("run_chat_target_real_eval"):
            return _run_chat_target_real_eval(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                trc=trc,
                provider=provider,
                target_mode=target_mode,
            )
    if eval_name == "harmfulness":
        # Harmfulness eval has no per-example target cache (prompts are
        # judged by an external API, not measured per-token). Skip with a
        # warning so operators see the gap but the sweep still completes.
        logger.warning(
            f"  target_real eval is not supported for suite={eval_name!r}; "
            f"skipping (no per-example cache contract for harmfulness)"
        )
        return None
    logger.warning(f"  target_real eval: unknown suite {eval_name}")
    return None


def _run_core_target_real_eval(
    *,
    model,
    tokenizer,
    device,
    cfg: DictConfig,
    trc,
    provider: SafetensorsContextFeatureProvider,
    target_mode: TargetConditioningMode,
    seed: int,
) -> dict:
    """Lift base_eval._run_core_target_real_eval into progression.

    The CORE evaluator dispatches per-example through
    ``base_eval._run_core_target_real_eval``: per row, look up cache,
    build wrapper, forward to score the answer span. Returns the same
    dict shape ``run_core_eval`` returns (results / centered_results /
    core_metric — mirrored on ``per_task`` / ``centered_per_task`` for
    consistency with progression's existing schema).
    """
    core_out = _base_eval_run_core_target_real(
        raw_encdec=model,
        tokenizer=tokenizer,
        device=device,
        trc=trc,
        provider=provider,
        max_per_task=int(cfg.max_per_task),
        target_mode=target_mode,
        seed=seed,
    )
    return {
        "core_metric": core_out["core_metric"],
        "per_task": core_out["results"],
        "centered_per_task": core_out["centered_results"],
    }


def _run_chat_target_real_eval(
    *,
    model,
    tokenizer,
    cfg: DictConfig,
    trc,
    provider: SafetensorsContextFeatureProvider,
    target_mode: TargetConditioningMode = TargetConditioningMode.CURRENT,
) -> dict:
    """Lift chat_eval._chat_eval_target_real_per_config into progression.

    The chat evaluator iterates the configured chat tasks and runs each
    task per-example with a freshly-built wrapper. Returns the per-task
    accuracy dict + chatcore_metric (mirrors ``run_chat_eval_suite``).
    """
    task_names = (
        list(_CHAT_ALL_TASKS)
        if cfg.chat_tasks is None
        else [t.strip() for t in cfg.chat_tasks.split(",")]
    )
    body = _chat_eval_target_real_per_config(
        model=model,
        tokenizer=tokenizer,
        trc=trc,
        provider=provider,
        task_names=task_names,
        num_samples=int(cfg.chat_num_samples),
        max_new_tokens=int(cfg.chat_max_new_tokens),
        temperature=float(cfg.chat_temperature),
        top_k=int(cfg.chat_top_k),
        max_problems=cfg.chat_max_problems,
        target_mode=target_mode,
    )
    if not isinstance(body, Mapping):
        raise ValueError(
            "chat target_real progression expected "
            "_chat_eval_target_real_per_config to return a flat eval body "
            f"mapping; got {type(body).__name__}"
        )
    if "seeds" in body or "kind" in body:
        raise ValueError(
            "chat target_real progression expected "
            "_chat_eval_target_real_per_config to return a flat eval body "
            "with top-level 'results'. Got SuiteResult-style payload keys "
            f"{list(body.keys())}; wrap with build_target_real_payload only "
            "at the standalone chat_eval YAML boundary."
        )
    results = body.get("results")
    if not isinstance(results, Mapping) or not results:
        raise ValueError(
            "chat target_real progression expected a non-empty top-level "
            f"'results' mapping; got keys {list(body.keys())}"
        )
    result: dict = {}
    for task_name, acc in results.items():
        if not isinstance(task_name, str):
            raise ValueError(
                "chat target_real progression expected string task names in "
                f"'results'; got {type(task_name).__name__}"
            )
        if isinstance(acc, bool) or not isinstance(acc, Real):
            raise ValueError(
                "chat target_real progression expected numeric accuracy for "
                f"task '{task_name}'; got {type(acc).__name__}"
            )
        result[task_name] = float(acc)
    if "chatcore_metric" in body:
        if isinstance(body["chatcore_metric"], bool) or not isinstance(
            body["chatcore_metric"], Real
        ):
            raise ValueError(
                "chat target_real progression expected numeric "
                f"'chatcore_metric'; got {type(body['chatcore_metric']).__name__}"
            )
        result["chatcore_metric"] = float(body["chatcore_metric"])
    return result


def _trc_kind_for(trc) -> str:
    """Map a TargetRealConfig variant -> SuiteResultKind.value (CI-5 helper).

    Stored in the per-cell ``cfg_result`` dict so analyze_sweep can dispatch
    by variant (PR-D Task 10).
    """
    if isinstance(trc, LockstepTargetRealConfig):
        return SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    if isinstance(trc, PooledTargetRealConfig):
        return SuiteResultKind.TARGET_REAL_POOLED.value
    raise TypeError(f"unexpected TargetRealConfig variant: {type(trc).__name__}")


def _run_target_real_for_suite(
    *,
    eval_name: str,
    eval_data: dict,
    eval_seeds: list[int],
    model,
    tokenizer,
    device,
    cfg: DictConfig,
    target_real_providers: dict[str, SafetensorsContextFeatureProvider],
    target_real_configs: list,
    target_mode: TargetConditioningMode,
    eval_pass_tags: Callable,
    should_run: Callable[[str, str, str, int], bool],
) -> None:
    """Per-suite target_real_configs collect-loop (PR-D Batch H2 fix).

    Mirrors the conditioned_configs peer in
    :func:`_evaluate_single_checkpoint_impl`:
      1. Per trc, write the cfg_result skeleton (kind / cache_dir /
         seeds) under ``eval_data['target_real']['configs'][trc.name]``.
      2. Iterate ``eval_seeds`` (CI-6 — uses ``_eval_seeds_for_suite``,
         not the full sweep seed list).
      3. Per seed, invoke :func:`_run_eval_suite_target_real` (CI-1 —
         the load-bearing fix; replaces the prior no-op stub).

    The provider lifecycle is plumbed via ``target_real_providers``
    (constructed once per checkpoint —
    see :func:`_target_real_providers_for_cfg`).
    """
    if not target_real_configs:
        return
    target_real_root = eval_data.setdefault("target_real", {}).setdefault(
        "configs",
        {},
    )
    for trc in target_real_configs:
        provider = target_real_providers[trc.name]
        trc_kind = _trc_kind_for(trc)
        cfg_result = target_real_root.setdefault(
            trc.name,
            {
                "kind": trc_kind,
                "cache_dir": str(trc.cache_dir),
                "target_conditioning_mode": target_mode.value,
                "seeds": {},
            },
        )
        cfg_result.setdefault("seeds", {})
        cfg_result["target_conditioning_mode"] = target_mode.value

        for seed in eval_seeds:
            if not should_run(eval_name, "target_real", trc.name, seed):
                continue
            if str(seed) in cfg_result["seeds"]:
                logger.info(
                    f"  [{device}] {eval_name}/target_real/{trc.name} seed={seed} already done"
                )
                continue
            logger.info(f"  [{device}] Running {eval_name}/target_real/{trc.name} (seed={seed})...")
            with eval_pass_tags(eval_name, "target_real", trc.name, seed):
                result = _run_eval_suite_target_real(
                    eval_name=eval_name,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    seed=seed,
                    cfg=cfg,
                    trc=trc,
                    provider=provider,
                    target_mode=target_mode,
                )
            if result is None:
                # Suite returned None (unsupported eval kind); stop iterating
                # the rest of the seed list — they will all yield None.
                break
            cfg_result["seeds"][str(seed)] = result
            logger.info(f"  [{device}] {eval_name}/target_real/{trc.name} seed={seed} complete")


# ---------------------------------------------------------------------------
# Results JSON helpers
# ---------------------------------------------------------------------------


def _results_dir(cfg: DictConfig) -> Path:
    """Directory containing per-checkpoint result files."""
    if cfg.results_file:
        return Path(cfg.results_file).parent / "checkpoints"
    return Path("out") / "progression" / cfg.model_tag / "checkpoints"


def _results_path(cfg: DictConfig) -> Path:
    """Path to the merged results.json (used for plotting)."""
    if cfg.results_file:
        return Path(cfg.results_file)
    return Path("out") / "progression" / cfg.model_tag / "results.json"


def _checkpoint_result_path(cfg: DictConfig, step: int, phase: str) -> Path:
    """Path to an individual checkpoint's result file."""
    return _results_dir(cfg) / f"{phase}_step_{step:06d}.json"


def _fig_dir(cfg: DictConfig) -> Path:
    if cfg.fig_dir:
        return Path(cfg.fig_dir)
    return Path("fig") / "progression" / cfg.model_tag


def load_checkpoint_result(cfg: DictConfig, step: int, phase: str) -> dict | None:
    """Load a single checkpoint's result file, or None if missing."""
    path = _checkpoint_result_path(cfg, step, phase)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    legacy_path = _results_path(cfg)
    if legacy_path.exists():
        with open(legacy_path) as f:
            legacy = json.load(f)
        return find_checkpoint_entry(legacy, step, phase)
    return None


def save_checkpoint_result(
    cfg: DictConfig,
    step: int,
    phase: str,
    evals: dict,
    checkpoint_identity: dict[str, Any] | None = None,
) -> None:
    """Atomically write a single checkpoint's result file."""
    path = _checkpoint_result_path(cfg, step, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"step": step, "phase": phase, "evals": evals}
    if checkpoint_identity is not None:
        data["checkpoint_identity"] = checkpoint_identity
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def merge_checkpoint_results(cfg: DictConfig) -> dict:
    """Merge all per-checkpoint result files into a single results dict."""
    results_dir = _results_dir(cfg)
    results = {"model_tag": cfg.model_tag, "checkpoints": []}
    if not results_dir.exists():
        return load_or_init_results(cfg)
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        return load_or_init_results(cfg)
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        results["checkpoints"].append(data)
    return results


def _configured_checkpoint_keys(cfg: DictConfig) -> set[tuple[int, str]] | None:
    """Return the checkpoint keys explicitly requested by config phases.

    Returns ``None`` (no filter) when ``cfg.phases`` is missing/empty, or when
    any phase entry lacks an explicit ``steps`` list — partially-specified
    phases (e.g. ``[{"phase": "safety_sft"}]``) cannot pin a `(step, phase)`
    key set, so we fall back to plotting whatever checkpoints exist rather
    than silently dropping data.
    """
    if not cfg.get("phases"):
        return None
    keys: set[tuple[int, str]] = set()
    for phase_cfg in cfg.phases:
        phase = str(phase_cfg.phase)
        steps = phase_cfg.get("steps")
        if steps is None:
            return None
        for step in steps:
            keys.add((int(step), phase))
    return keys


def _filter_results_to_configured_checkpoints(results: dict, cfg: DictConfig) -> dict:
    """Keep plotted checkpoints aligned with the current progression config."""
    keys = _configured_checkpoint_keys(cfg)
    if keys is None:
        return results

    checkpoints = results.get("checkpoints", [])
    filtered_checkpoints = [
        ckpt
        for ckpt in checkpoints
        if (int(ckpt.get("step", 0)), str(ckpt.get("phase", ""))) in keys
    ]
    if len(filtered_checkpoints) == len(checkpoints):
        return results

    skipped = [
        f"{ckpt.get('phase')}:{ckpt.get('step')}"
        for ckpt in checkpoints
        if (int(ckpt.get("step", 0)), str(ckpt.get("phase", ""))) not in keys
    ]
    logger.info(
        "Skipping {n} checkpoint result(s) not present in the current config: {items}",
        n=len(skipped),
        items=", ".join(skipped),
    )
    filtered = dict(results)
    filtered["checkpoints"] = filtered_checkpoints
    return filtered


def save_merged_results(cfg: DictConfig) -> dict:
    """Merge per-checkpoint files and write the combined results.json."""
    results = merge_checkpoint_results(cfg)
    path = _results_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)
    return results


# Legacy helpers (used by plotting and single-GPU path)
def load_or_init_results(cfg: DictConfig) -> dict:
    path = _results_path(cfg)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"model_tag": cfg.model_tag, "checkpoints": []}


def save_results(results: dict, cfg: DictConfig) -> None:
    path = _results_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.rename(path)


def find_checkpoint_entry(results: dict, step: int, phase: str) -> dict | None:
    for entry in results["checkpoints"]:
        if entry["step"] == step and entry["phase"] == phase:
            return entry
    return None


def _existing_checkpoint_entry(
    cfg: DictConfig,
    merged_results: dict,
    spec: CheckpointSpec,
) -> dict | None:
    entry = load_checkpoint_result(cfg, spec.step, spec.phase) or find_checkpoint_entry(
        merged_results,
        spec.step,
        spec.phase,
    )
    if entry is None:
        return None

    persisted = entry.get("checkpoint_identity")
    if persisted is None:
        global _legacy_checkpoint_identity_warned
        if not _legacy_checkpoint_identity_warned:
            logger.warning(
                f"Legacy progression result for {spec.phase} step {spec.step} has no "
                "checkpoint_identity; accepting it without checkpoint validation"
            )
            _legacy_checkpoint_identity_warned = True
        return entry

    try:
        current = _checkpoint_identity_for_spec(spec, cfg)
    except FileNotFoundError as exc:
        logger.warning(
            f"Cannot validate cached result for {spec.phase} step {spec.step}: "
            f"{exc}; scheduling it again"
        )
        return None
    if current != persisted:
        logger.warning(
            f"Checkpoint identity changed for {spec.phase} step {spec.step}; "
            "discarding cached measurements and scheduling them again"
        )
        return None
    return entry


def _decoder_seed_done(
    entry: dict | None,
    eval_name: str,
    seed: int,
    *,
    cfg: DictConfig | None = None,
) -> bool:
    """Check the legacy-compatible top-level decoder seed namespace."""

    if entry is None:
        return False
    eval_data = entry.get("evals", {}).get(eval_name, {})
    seed_payload = eval_data.get("seeds", {}).get(str(seed))
    return _eval_seed_payload_done(
        seed_payload,
        eval_name=eval_name,
        seed=seed,
        cfg=cfg,
    )


def _conditioned_seed_done(
    entry: dict | None,
    eval_name: str,
    config_name: str,
    seed: int,
    *,
    cfg: DictConfig | None = None,
) -> bool:
    if entry is None:
        return False
    eval_data = entry.get("evals", {}).get(eval_name, {})
    cfg_data = eval_data.get("conditioned", {}).get("configs", {}).get(config_name, {})
    seed_payload = cfg_data.get("seeds", {}).get(str(seed))
    return _eval_seed_payload_done(
        seed_payload,
        eval_name=eval_name,
        seed=seed,
        cfg=cfg,
    )


def _harmfulness_progression_spec_sha256(
    cfg: DictConfig,
    seed: int,
    *,
    self_conditioned_receipt: dict[str, Any] | None = None,
    strength_sweep: bool = False,
) -> str:
    """Hash every pre-generation input available at resume scheduling time."""
    spec = {
        "schema_version": _HARMFULNESS_PROGRESSION_SPEC_VERSION,
        "generation_contract_version": 3,
        "chat_prompt_renderer": "render_prompt_for_chat_completion_v1",
        "target_aware_response_decode": (
            "complete_generated_transcript_including_forced_tokens_v1"
        ),
        "non_target_aware_response_decode": "inference_engine_chat_response_v1",
        "seed": int(seed),
        "batch_size": 8,
        "temperature": float(cfg.harm_temperature),
        "top_k": None if cfg.harm_top_k is None else int(cfg.harm_top_k),
        "max_new_tokens": int(cfg.harm_max_new_tokens),
        "num_benign": int(cfg.harm_num_benign),
        "num_harmful": (
            None if cfg.get("harm_num_harmful", None) is None else int(cfg.harm_num_harmful)
        ),
        "render_as_chat": True,
        "device_type": str(cfg.get("device_type", "")),
        "dtype": str(cfg.get("dtype", "")),
        "prompt_loader_contract_version": 1,
        "harmful_dataset": {
            "id": "PKU-Alignment/BeaverTails-Evaluation",
            "split": "test",
            "revision": harmfulness_judge.BEAVERTAILS_EVALUATION_REVISION,
        },
        "benign_dataset": {
            "id": harmfulness_judge.WILDGUARD_MIX_DATASET_ID,
            "config": harmfulness_judge.WILDGUARD_MIX_CONFIG,
            "split": harmfulness_judge.WILDGUARD_MIX_SPLIT,
            "revision": harmfulness_judge.WILDGUARD_MIX_REVISION,
            "prompt_harm_label": harmfulness_judge.BENIGN_PROMPT_HARM_LABEL,
            "prompt_harm_agreement": (
                harmfulness_judge.BENIGN_PROMPT_HARM_AGREEMENT
            ),
            "subcategory": harmfulness_judge.BENIGN_PROMPT_SUBCATEGORY,
        },
    }
    if _self_conditioned_cfg(cfg):
        spec["self_conditioned"] = (
            _self_conditioned_progression_receipt(cfg)
            if self_conditioned_receipt is None else self_conditioned_receipt
        )
        spec["batch_size"] = 1
        spec["seed_schedule"] = SELF_CONDITIONED_SEED_SCHEDULE
        spec["non_target_aware_response_decode"] = "self_conditioned_complete_generated_transcript_v1"
    if strength_sweep:
        # Compare controlled studies by content: moving identical cache/statistics
        # files is harmless, whereas changing anything except strength is not.
        receipt = copy.deepcopy(spec["self_conditioned"])
        receipt.pop("cache_dir")
        receipt.pop("stats_path")
        receipt["safety_steering"]["strength"] = 0.0
        spec["self_conditioned"] = receipt
        judge = build_judge_provenance(model=str(cfg.harm_judge_model), pairs=[])
        judge.pop("ordered_pairs_sha256")
        spec["judge"] = judge
    rendered = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


def _harmfulness_progression_provenance(
    cfg: DictConfig,
    seed: int,
) -> dict[str, Any]:
    receipt = _self_conditioned_progression_receipt(cfg) if _self_conditioned_cfg(cfg) else None
    result = {
        "schema_version": _HARMFULNESS_PROGRESSION_SPEC_VERSION,
        "evaluation_spec_sha256": _harmfulness_progression_spec_sha256(
            cfg,
            seed,
            self_conditioned_receipt=receipt,
        ),
    }
    if receipt is not None:
        result["self_conditioned"] = receipt
        if receipt["safety_steering"] is not None:
            result[STRENGTH_SWEEP_SPEC_SHA256_KEY] = _harmfulness_progression_spec_sha256(
                cfg, seed, self_conditioned_receipt=receipt, strength_sweep=True,
            )
    return result


def _self_conditioned_progression_receipt(cfg: DictConfig) -> dict[str, Any]:
    """Bind current bundle bytes and runtime inputs at scheduling and execution."""
    settings = parse_harmfulness_safety_settings(_plain(cfg))
    cache_dir = _conditioning_cfg(cfg).get("context_cache_dir")
    if not cache_dir:
        raise ValueError("self-conditioned progression requires conditioning.context_cache_dir")
    cache_root = Path(cache_dir).resolve(strict=True)
    cache_files = [path for path in recursive_file_paths(cache_root)
                   if path.name == MANIFEST_FILENAME or path.name.endswith(COMPRESSED_EXT)]
    if not any(path.name == MANIFEST_FILENAME for path in cache_files):
        raise ValueError(f"self-conditioned progression cache has no {MANIFEST_FILENAME}: {cache_root}")
    inventory = capture_file_inventory(cache_root, cache_files, domain=_SELF_CONDITIONED_CACHE_DOMAIN)
    inventory.assert_unchanged()
    return {
        "cache_dir": str(cache_dir),
        "cache_sha256": inventory.sha256,
        "stats_path": cfg.get("stats_path"),
        "stats_sha256": (
            None if cfg.get("stats_path") is None else
            stream_hash_file_descriptor_bound(Path(cfg.stats_path)).sha256
        ),
        "self_cond_segment_size": cfg.get("self_cond_segment_size"),
        "extraction_tokenizer_id": cfg.get("extraction_tokenizer_id"),
        "safety_steering": None if settings is None else {
            "targets_sha256": stream_hash_file_descriptor_bound(settings.targets).sha256,
            "strength": float(settings.strength),
        },
    }


def _eval_seed_payload_done(
    seed_payload: Any,
    *,
    eval_name: str,
    seed: int,
    cfg: DictConfig | None,
) -> bool:
    if seed_payload is None:
        return False
    if eval_name != "harmfulness":
        return True
    if cfg is None or not isinstance(seed_payload, dict):
        return False
    try:
        validate_harmfulness_contract(
            seed_payload,
            model=str(cfg.harm_judge_model),
        )
    except ValueError:
        return False
    expected = _harmfulness_progression_provenance(cfg, seed)
    if _self_conditioned_cfg(cfg):
        completion = seed_payload.get("completion")
        try:
            validate_harmfulness_completion_provenance(completion)
        except ValueError:
            return False
        if "safety_steering" not in completion or completion["safety_steering"] != expected["self_conditioned"]["safety_steering"]:
            return False
    return seed_payload.get("progression_provenance") == expected


def _target_real_seed_done(
    entry: dict | None,
    eval_name: str,
    trc_name: str,
    seed: int,
) -> bool:
    if entry is None:
        return False
    eval_data = entry.get("evals", {}).get(eval_name, {})
    cfg_data = eval_data.get("target_real", {}).get("configs", {}).get(trc_name, {})
    seed_payload = cfg_data.get("seeds", {}).get(str(seed))
    return seed_payload is not None


EvalWorkKey = tuple[str, str, str, int]


def _eval_work_key(
    eval_name: str,
    variant: str,
    config_name: str,
    seed: int,
) -> EvalWorkKey:
    return (eval_name, variant, config_name, int(seed))


def _pending_eval_work_keys(
    entry: dict | None,
    eval_suites: list[str],
    seeds: list[int],
    cfg: DictConfig,
    phase: str | None,
) -> list[EvalWorkKey]:
    """Return missing eval/variant/seed units for one checkpoint."""
    conditioned_configs = conditioned_configs_for_phase(cfg, phase) if phase is not None else []
    runs_decoder = _is_decoder_progression(cfg)
    target_real_configs = list(_conditioning_cfg(cfg).get("target_real_configs", []) or [])
    if target_real_configs and not _should_search_nested_checkpoints(cfg):
        target_real_configs = []
    work: list[EvalWorkKey] = []
    for eval_name in eval_suites:
        eval_seeds = _eval_seeds_for_suite(eval_name, seeds)
        if eval_name == "harmfulness" and not runs_decoder and phase is not None:
            _validate_harmfulness_conditioning(cfg, phase)
            for seed in eval_seeds:
                if not _conditioned_seed_done(
                    entry, eval_name, _SELF_CONDITIONED_CONFIG_NAME, seed, cfg=cfg
                ):
                    work.append(_eval_work_key(
                        eval_name, "conditioned", _SELF_CONDITIONED_CONFIG_NAME, seed
                    ))
            continue
        if runs_decoder:
            for seed in eval_seeds:
                if not _decoder_seed_done(entry, eval_name, seed, cfg=cfg):
                    work.append(
                        _eval_work_key(
                            eval_name,
                            _DECODER_WORK_VARIANT,
                            "",
                            seed,
                        )
                    )
        if conditioned_configs:
            for feature_cfg in conditioned_configs:
                for seed in eval_seeds:
                    if not _conditioned_seed_done(
                        entry,
                        eval_name,
                        feature_cfg.name,
                        seed,
                        cfg=cfg,
                    ):
                        work.append(
                            _eval_work_key(
                                eval_name,
                                "conditioned",
                                feature_cfg.name,
                                seed,
                            )
                        )
        if eval_name not in {"core", "chat"}:
            continue
        for trc in target_real_configs:
            if isinstance(trc, (LockstepTargetRealConfig, PooledTargetRealConfig)):
                trc_name = str(trc.name)
            elif isinstance(trc, Mapping):
                trc_name = str(trc["name"])
            else:
                raise TypeError(
                    "target_real_configs entries must be mappings or "
                    "TargetRealConfig instances; got "
                    f"{type(trc).__name__}"
                )
            for seed in eval_seeds:
                if not _target_real_seed_done(
                    entry,
                    eval_name,
                    trc_name,
                    seed,
                ):
                    work.append(_eval_work_key(eval_name, "target_real", trc_name, seed))
    return work


def _merge_eval_results(base: dict, update: dict) -> dict:
    """Deep-merge eval result dictionaries, preserving completed seeds."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_eval_results(base[key], value)
        else:
            base[key] = value
    return base


def _merge_checkpoint_entry(base: dict, update: dict) -> dict:
    if not base:
        base.update(
            {
                "step": update["step"],
                "phase": update["phase"],
                "evals": {},
            }
        )
    _merge_eval_results(base.setdefault("evals", {}), update.get("evals", {}))
    if "checkpoint_identity" in update:
        base["checkpoint_identity"] = update["checkpoint_identity"]
    return base


def _split_work_queue_paths(cfg: DictConfig, queue_id: str) -> tuple[Path, Path]:
    queue_dir = _results_dir(cfg) / "work_queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_path = queue_dir / f"split_eval_{queue_id}.jsonl"
    lock_path = queue_dir / f"split_eval_{queue_id}.lock"
    return queue_path, lock_path


def _cleanup_split_work_queue(queue_path: Path, lock_path: Path) -> None:
    for path in (queue_path, lock_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _serialize_split_work_item(
    spec: CheckpointSpec,
    work_key: EvalWorkKey,
) -> str:
    return json.dumps(
        {
            "step": spec.step,
            "phase": spec.phase,
            "subdir": spec.subdir,
            "checkpoint_identity": spec.checkpoint_identity,
            "work_key": list(work_key),
        }
    )


def _deserialize_split_work_item(line: str) -> tuple[CheckpointSpec, EvalWorkKey]:
    data = json.loads(line)
    work_key_data = data["work_key"]
    work_key: EvalWorkKey = (
        str(work_key_data[0]),
        str(work_key_data[1]),
        str(work_key_data[2]),
        int(work_key_data[3]),
    )
    return (
        CheckpointSpec(
            step=int(data["step"]),
            phase=str(data["phase"]),
            subdir=str(data["subdir"]),
            checkpoint_identity=data.get("checkpoint_identity"),
        ),
        work_key,
    )


def _initialize_split_work_queue(
    queue_path: Path,
    lock_path: Path,
    all_work: list[tuple[CheckpointSpec, EvalWorkKey]],
) -> None:
    """Create a shared queue once for torchrun ranks in split mode."""
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if queue_path.exists():
                return
            tmp = queue_path.with_suffix(f".tmp.{os.getpid()}")
            with open(tmp, "w") as f:
                for spec, work_key in all_work:
                    f.write(_serialize_split_work_item(spec, work_key))
                    f.write("\n")
            tmp.replace(queue_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _pop_split_work_item(
    queue_path: Path,
    lock_path: Path,
) -> tuple[CheckpointSpec, EvalWorkKey] | None:
    """Atomically pop one split eval work item from the shared queue."""
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if not queue_path.exists():
                return None
            with open(queue_path) as f:
                lines = f.readlines()
            if not lines:
                return None
            line = lines[0].strip()
            tmp = queue_path.with_suffix(f".tmp.{os.getpid()}")
            with open(tmp, "w") as f:
                f.writelines(lines[1:])
            tmp.replace(queue_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    if not line:
        return None
    return _deserialize_split_work_item(line)


# ---------------------------------------------------------------------------
# Checkpoint resolution (local + remote)
# ---------------------------------------------------------------------------


def _local_checkpoints_root(cfg: DictConfig) -> Path:
    """Resolve the local checkpoint-family root directory."""
    root_name = str(cfg.checkpoint_root)
    if cfg.checkpoints_dir:
        return Path(cfg.checkpoints_dir) / root_name
    return Path(get_base_dir()) / root_name


def _checkpoint_prefix(cfg: DictConfig) -> Path:
    prefix = str(cfg.checkpoint_prefix or "")
    return Path(prefix) if prefix else Path()


def _should_search_nested_checkpoints(cfg: DictConfig) -> bool:
    return (
        _model_family(cfg) is _ModelFamily.ENCODER_DECODER
        or str(cfg.checkpoint_root) == "conditional_encdec_checkpoints"
    )


def _is_decoder_progression(cfg: DictConfig) -> bool:
    """Return whether config selects the unconditional decoder eval variant.

    The decoder variant is deliberately config-scoped: encoder-decoder
    checkpoints still require a real conditioned or target-real variant.
    """

    # Static variants require an encoder-decoder checkpoint even when the
    # current schema delegates family selection to ``auto``.
    if _conditioning_cfg(cfg).get("conditioned_configs") or _self_conditioned_cfg(cfg):
        return False

    family = _model_family(cfg)
    if family is _ModelFamily.DECODER:
        return True
    if family is _ModelFamily.ENCODER_DECODER:
        return False
    return not _should_search_nested_checkpoints(cfg)


def _find_checkpoint_dir_for_step(root: Path, step: int) -> Path | None:
    """Find a nested directory under *root* containing model/meta files."""
    model_name = f"model_{step:06d}.pt"
    meta_name = f"meta_{step:06d}.json"
    if (root / model_name).exists() and (root / meta_name).exists():
        return root
    candidates = sorted(
        {
            model_file.parent
            for model_file in root.rglob(model_name)
            if (model_file.parent / meta_name).exists()
        },
        key=str,
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise ValueError(
        f"Ambiguous nested checkpoint for step {step} under {root}: "
        f"found {len(candidates)} model/meta candidates:\n{formatted}\n"
        "Set checkpoint_prefix to the precise checkpoint lineage."
    )


def resolve_local_checkpoint_dir(spec: CheckpointSpec, cfg: DictConfig) -> str:
    """Resolve checkpoint directory from local filesystem."""
    root = _local_checkpoints_root(cfg)
    ckpt_dir = root / cfg.model_tag / _checkpoint_prefix(cfg)
    if spec.subdir:
        ckpt_dir = ckpt_dir / spec.subdir

    model_file = ckpt_dir / f"model_{spec.step:06d}.pt"
    meta_file = ckpt_dir / f"meta_{spec.step:06d}.json"
    if _should_search_nested_checkpoints(cfg) and (
        not model_file.exists() or not meta_file.exists()
    ):
        nested = _find_checkpoint_dir_for_step(ckpt_dir, spec.step)
        if nested is not None:
            return str(nested)

    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Meta not found: {meta_file}")

    return str(ckpt_dir)


def _scp_cmd(cfg: DictConfig) -> list[str]:
    """Base scp command with SSH options."""
    return [
        "scp",
        "-P",
        str(cfg.remote.port),
        "-i",
        os.path.expanduser(cfg.remote.ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
    ]


def _ssh_cmd(cfg: DictConfig) -> list[str]:
    """Base ssh command matching the SCP transport options."""
    return [
        "ssh",
        "-p",
        str(cfg.remote.port),
        "-i",
        os.path.expanduser(cfg.remote.ssh_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
    ]


def _remote_checkpoint_local_dir(spec: CheckpointSpec, cfg: DictConfig) -> Path:
    local_dir = Path(cfg.remote.local_temp_dir) / cfg.model_tag / _checkpoint_prefix(cfg)
    if spec.subdir:
        local_dir = local_dir / spec.subdir
    return local_dir


def _checkpoint_lock_path(local_dir: Path, step: int) -> Path:
    return local_dir / f".checkpoint_{step:06d}.lock"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_file_identity(
    cfg: DictConfig,
    remote_path: str,
    pattern: str,
) -> tuple[int, str]:
    quoted_path = shlex.quote(remote_path)
    remote_command = f"stat -c %s -- {quoted_path} && sha256sum -- {quoted_path}"
    result = subprocess.run(
        _ssh_cmd(cfg) + [str(cfg.remote.host), remote_command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"remote checkpoint identity query failed for {pattern}: {result.stderr.strip()}"
        )

    lines = result.stdout.splitlines()
    try:
        size = int(lines[0].strip())
        sha256 = lines[1].split()[0].lower()
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"remote checkpoint identity query returned malformed output for {pattern}: "
            f"{result.stdout!r}"
        ) from exc
    if size < 0 or len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise RuntimeError(
            f"remote checkpoint identity query returned invalid size/SHA-256 "
            f"for {pattern}: size={size}, sha256={sha256!r}"
        )
    return size, sha256


def _local_file_identity(path: Path) -> tuple[int, str] | None:
    if not path.is_file():
        return None
    return path.stat().st_size, _sha256_file(path)


def _validate_staged_remote_file(
    staged_file: Path,
    pattern: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    actual = _local_file_identity(staged_file)
    if actual is None:
        raise RuntimeError(
            f"download validation failed for {pattern}: SCP produced no regular file"
        )
    actual_size, actual_sha256 = actual
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"download validation failed for {pattern}: "
            f"remote size={expected_size} sha256={expected_sha256}, "
            f"staged size={actual_size} sha256={actual_sha256}"
        )


def fetch_remote_checkpoint(spec: CheckpointSpec, cfg: DictConfig) -> str:
    """Transfer model + meta from remote to local temp dir. Returns local dir path."""
    model_tag = cfg.model_tag
    local_dir = _remote_checkpoint_local_dir(spec, cfg)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_base = f"{cfg.remote.base_path}/{model_tag}"
    prefix = str(_checkpoint_prefix(cfg))
    if prefix and prefix != ".":
        remote_base = f"{remote_base}/{prefix}"
    if spec.subdir:
        remote_base = f"{remote_base}/{spec.subdir}"

    lock_path = _checkpoint_lock_path(local_dir, spec.step)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            for pattern in [
                f"model_{spec.step:06d}.pt",
                f"meta_{spec.step:06d}.json",
            ]:
                local_file = local_dir / pattern
                remote_file = f"{remote_base}/{pattern}"
                expected_size, expected_sha256 = _remote_file_identity(
                    cfg,
                    remote_file,
                    pattern,
                )
                if _local_file_identity(local_file) == (
                    expected_size,
                    expected_sha256,
                ):
                    logger.info(f"  {pattern} already verified locally, skipping transfer")
                    continue

                staged_file = local_dir / (f".{pattern}.stage.{os.getpid()}.{uuid.uuid4().hex}")
                remote_path = f"{cfg.remote.host}:{remote_file}"
                cmd = _scp_cmd(cfg) + [remote_path, str(staged_file)]
                logger.info(f"  Fetching {pattern}...")
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"scp failed for {pattern}: {result.stderr.strip()}")
                    _validate_staged_remote_file(
                        staged_file,
                        pattern,
                        expected_size,
                        expected_sha256,
                    )
                    os.replace(staged_file, local_file)
                finally:
                    try:
                        staged_file.unlink()
                    except FileNotFoundError:
                        pass
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return str(local_dir)


def _file_identity(path: Path) -> dict[str, int | str]:
    """Return the exact byte length and SHA-256 of one local file."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    return {"size": size, "sha256": sha256}


def _local_checkpoint_identity(checkpoint_dir: str, step: int) -> dict[str, Any]:
    directory = Path(checkpoint_dir)
    return {
        "model": _file_identity(directory / f"model_{step:06d}.pt"),
        "meta": _file_identity(directory / f"meta_{step:06d}.json"),
    }


def _checkpoint_identity_for_spec(
    spec: CheckpointSpec,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Resolve/fetch lazily, then hash only the local checkpoint files."""
    if spec.checkpoint_identity is not None:
        return spec.checkpoint_identity
    checkpoint_dir = (
        fetch_remote_checkpoint(spec, cfg)
        if cfg.checkpoint_source == "remote"
        else resolve_local_checkpoint_dir(spec, cfg)
    )
    identity = _local_checkpoint_identity(checkpoint_dir, spec.step)
    spec.checkpoint_identity = identity
    return identity


def _read_checkpoint_metadata(checkpoint_dir: str, step: int) -> dict:
    meta_path = Path(checkpoint_dir) / f"meta_{step:06d}.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _should_load_encoder_decoder(
    checkpoint_dir: str,
    step: int,
    cfg: DictConfig,
) -> bool:
    family = _model_family(cfg)
    if family is _ModelFamily.ENCODER_DECODER:
        return True
    if family is _ModelFamily.DECODER:
        return False

    path_parts = set(Path(checkpoint_dir).parts)
    if "conditional_encdec_checkpoints" in path_parts:
        return True

    meta = _read_checkpoint_metadata(checkpoint_dir, step)
    return meta_looks_encdec(meta)


def load_progression_model(
    checkpoint_dir: str,
    step: int,
    device: torch.device,
    cfg: DictConfig,
):
    """Load a decoder-only or encoder-decoder checkpoint for progression eval."""

    if _should_load_encoder_decoder(checkpoint_dir, step, cfg):
        model, tokenizer, meta = build_encdec_model(
            checkpoint_dir,
            step,
            torch.device("cpu") if device.type == "cuda" else device,
            ModelPhase.EVAL,
            eval_conditioning=conditioned_eval_conditioning("training_progression"),
        )
        checkpoint_contract = assert_conditioned_eval_model(
            model,
            meta,
            checkpoint_dir=checkpoint_dir,
            eval_name="training_progression",
        )
        if not (_self_conditioned_cfg(cfg) and cfg.evals == "harmfulness"):
            require_fixed_checkpoint_segment_mode(
                meta,
                surface="training_progression (turn checkpoints require evals=harmfulness and self_conditioned)",
                checkpoint_dir=checkpoint_dir,
            )
        if device.type == "cuda":
            model = model.to(device=device, dtype=(
                torch.float32 if cfg.dtype == "float32" else torch.bfloat16
            ))
        meta["model_type"] = "encoder_decoder"
    else:
        model, tokenizer, meta = build_decoder_model(
            checkpoint_dir,
            step,
            device,
            ModelPhase.EVAL,
        )
        checkpoint_contract = assert_conditioned_eval_model(
            model,
            meta,
            checkpoint_dir=checkpoint_dir,
            eval_name="training_progression",
        )
        meta["model_type"] = "decoder"
    meta["checkpoint_dir"] = checkpoint_dir
    return model, tokenizer, meta, checkpoint_contract


def cleanup_remote_checkpoint(local_dir: str, step: int) -> None:
    """Remove the large .pt file after evaluation (remote mode only)."""
    checkpoint_dir = Path(local_dir)
    if not checkpoint_dir.exists():
        return
    lock_path = _checkpoint_lock_path(checkpoint_dir, step)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            pt_file = checkpoint_dir / f"model_{step:06d}.pt"
            if pt_file.exists():
                pt_file.unlink()
                logger.info(f"  Cleaned up {pt_file.name}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_cleanup_after_eval(cfg: DictConfig) -> None:
    if bool(cfg.get("cleanup_after_eval", False)) and cfg.checkpoint_source != "remote":
        raise ValueError(
            "cleanup_after_eval=true requires checkpoint_source=remote; "
            "refusing to delete a local checkpoint"
        )


# ---------------------------------------------------------------------------
# Eval dispatch
# ---------------------------------------------------------------------------


def run_core_eval(model, tokenizer, device, seed: int, cfg: DictConfig) -> dict:
    """Run CORE eval with a specific seed."""
    core_out = _base_eval_evaluate_core(
        model,
        tokenizer,
        device,
        max_per_task=cfg.max_per_task,
        seed=seed,
        core_example_batch_size=cfg.get("core_example_batch_size", 1),
    )
    return {
        "core_metric": core_out["core_metric"],
        "per_task": core_out["results"],
        "centered_per_task": core_out["centered_results"],
    }


def run_chat_eval_suite(
    model,
    tokenizer,
    cfg: DictConfig,
) -> dict:
    """Run all chat eval tasks. Returns per-task accuracies + chatcore."""
    task_names = (
        list(_CHAT_ALL_TASKS)
        if cfg.chat_tasks is None
        else [t.strip() for t in cfg.chat_tasks.split(",")]
    )

    engine = Engine(model, tokenizer)
    results: dict = {}
    for task_name in task_names:
        logger.info(f"    Chat task: {task_name}")
        acc = _chat_eval_run_chat_eval(
            task_name,
            model,
            tokenizer,
            engine,
            batch_size=cfg.chat_batch_size,
            num_samples=cfg.chat_num_samples,
            max_new_tokens=cfg.chat_max_new_tokens,
            temperature=cfg.chat_temperature,
            top_k=cfg.chat_top_k,
            max_problems=cfg.chat_max_problems,
        )
        results[task_name] = acc
        logger.info(f"    {task_name}: {100 * acc:.2f}%")

    # Compute ChatCORE if all standard tasks are present
    all_present = all(t in results for t in _CHAT_ALL_TASKS)
    if all_present:
        centered_sum = 0.0
        for t_name, acc in results.items():
            baseline = _CHAT_BASELINE_ACCURACIES.get(t_name, 0.0)
            centered_sum += (acc - baseline) / (1.0 - baseline)
        results["chatcore_metric"] = centered_sum / len(results)

    return results


def run_harmfulness_eval(
    model,
    tokenizer,
    device,
    seed: int,
    cfg: DictConfig,
) -> dict:
    """Decoder-only generation; encoder-decoder runs require the shared runtime."""
    if _is_encoder_decoder_model(model) or getattr(model, "is_target_aware_encdec", False) is True:
        raise ValueError(
            "Encoder-decoder harmfulness requires run_self_conditioned_harmfulness_eval "
            "and self_conditioned.safety_steering; legacy static wrappers are unsupported"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set — cannot run harmfulness eval")
    engine = InferenceEngine(model, tokenizer, device)
    pool = resolve_harmfulness_prompt_pool(
        seed=seed, max_harmful=cfg.get("harm_num_harmful"), num_benign=cfg.harm_num_benign
    )
    all_prompts = [dict(row) for row in pool.rows]
    logger.info(f"    Generating responses for {len(all_prompts)} prompts (seed={seed})...")
    batch_size = 8
    for index in range(0, len(all_prompts), batch_size):
        batch = all_prompts[index : index + batch_size]
        responses = engine.generate_batch(
            [row["prompt"] for row in batch], max_new_tokens=cfg.harm_max_new_tokens,
            temperature=cfg.harm_temperature, top_k=cfg.harm_top_k,
            seed=absolute_prompt_batch_seed(seed, index), render_as_chat=True,
        )
        if len(responses) != len(batch):
            raise ValueError(
                "Harmfulness generation returned "
                f"{len(responses)} responses for {len(batch)} prompts "
                f"at batch starting index {index}"
            )
        for prompt, response in zip(batch, responses, strict=True):
            prompt["response"] = response
    return _judge_harmfulness_progression_rows(all_prompts, seed=seed, cfg=cfg)


def _judge_harmfulness_progression_rows(
    all_prompts: list[dict], *, seed: int, cfg: DictConfig
) -> dict:
    """One judge/artifact contract for decoder and self-conditioned progression."""
    logger.info("    Judging responses...")
    client = harmfulness_judge.JudgeClient(model=cfg.harm_judge_model)
    judge_results = list(client.score_all(all_prompts))
    if len(judge_results) != len(all_prompts):
        raise ValueError(
            "Harmfulness judge returned "
            f"{len(judge_results)} results for {len(all_prompts)} prompts"
        )
    failed_indices = [index for index, result in enumerate(judge_results) if result is None]
    if failed_indices:
        raise ValueError(
            "Harmfulness judge failed "
            f"{len(failed_indices)} of {len(all_prompts)} prompts; "
            f"failed indices={failed_indices}"
        )
    metrics = harmfulness_judge.compute_metrics(all_prompts, judge_results)
    benign_prompts = [prompt for prompt in all_prompts if prompt["pool"] == "benign"]
    compliance_results = list(client.score_compliance_all(benign_prompts))
    if len(compliance_results) != len(benign_prompts):
        raise ValueError(
            "Benign compliance judge returned "
            f"{len(compliance_results)} results for {len(benign_prompts)} prompts"
        )
    failed_compliance_indices = [
        index for index, result in enumerate(compliance_results) if result is None
    ]
    if failed_compliance_indices:
        raise ValueError(
            "Benign compliance judge failed "
            f"{len(failed_compliance_indices)} of {len(benign_prompts)} prompts; "
            f"failed benign indices={failed_compliance_indices}"
        )
    compliance_metrics = harmfulness_judge.compute_compliance_metrics(
        benign_prompts,
        compliance_results,
    )
    payload = build_harmfulness_payload(
        metrics=metrics,
        compliance_metrics=compliance_metrics,
        model=str(cfg.harm_judge_model),
        pairs=all_prompts,
    )
    payload["progression_provenance"] = _harmfulness_progression_provenance(
        cfg,
        seed,
    )
    return payload


def run_self_conditioned_harmfulness_eval(
    *, model, tokenizer, device, seed: int, cfg: DictConfig,
    meta: Mapping[str, Any], feature_contract,
) -> dict:
    """Validate this checkpoint's own target/cache identity before generation."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set — cannot run harmfulness eval")
    pool = resolve_harmfulness_prompt_pool(
        seed=seed, max_harmful=cfg.get("harm_num_harmful"), num_benign=cfg.harm_num_benign
    )
    rows = [dict(row) for row in pool.rows]
    prompt_ids = [render_prompt_for_chat_completion(tokenizer, row["prompt"]).token_ids for row in rows]
    provenance = _harmfulness_progression_provenance(cfg, seed)
    setup = prepare_harmfulness_self_conditioning(
        _plain(cfg), model=model, tokenizer=tokenizer, device=device, meta=meta,
        feature_contract=feature_contract,
        prompt_token_ids=prompt_ids,
    )
    if _harmfulness_progression_provenance(cfg, seed) != provenance:
        raise ValueError("Self-conditioning inputs changed during progression runtime preparation")
    completion = harmfulness_completion_provenance(setup)
    if completion["safety_steering"] != provenance["self_conditioned"]["safety_steering"]:
        raise ValueError("Safety target changed between runtime validation and progression receipt")
    for index, row in enumerate(rows):
        row["response"] = generate_harmfulness_response_self_conditioned(
            setup=setup, stable_pool_index=index, prompt=row["prompt"],
            pool=row["pool"], category=row["category"], base_seed=seed,
            max_new_tokens=int(cfg.harm_max_new_tokens), temperature=float(cfg.harm_temperature),
            top_k=cfg.harm_top_k,
        )
    if _harmfulness_progression_provenance(cfg, seed) != provenance:
        raise ValueError("Self-conditioning inputs changed during progression generation")
    setup.assert_target_unchanged()
    payload = _judge_harmfulness_progression_rows(rows, seed=seed, cfg=cfg)
    setup.assert_target_unchanged()
    if payload["progression_provenance"] != provenance:
        raise ValueError("Self-conditioning inputs changed during progression judging")
    payload["completion"] = completion
    return payload


def _eval_seeds_for_suite(eval_name: str, seeds: list[int]) -> list[int]:
    # Chat eval tasks are deterministic under the configured generation params;
    # keep the historical progression behavior of recording a single seed.
    return seeds[:1] if eval_name == "chat" else seeds


def _run_eval_suite(
    eval_name: str,
    model,
    tokenizer,
    device,
    seed: int,
    cfg: DictConfig,
):
    if eval_name == "core":
        with timed_region("run_core_eval"):
            return run_core_eval(model, tokenizer, device, seed, cfg)
    if eval_name == "chat":
        with timed_region("run_chat_eval_suite"):
            return run_chat_eval_suite(model, tokenizer, cfg)
    if eval_name == "harmfulness":
        with timed_region("run_harmfulness_eval"):
            return run_harmfulness_eval(model, tokenizer, device, seed, cfg)
    logger.warning(f"  Unknown eval suite: {eval_name}")
    return None


# ---------------------------------------------------------------------------
# Collection orchestrator
# ---------------------------------------------------------------------------


def _needs_eval(
    entry: dict | None,
    eval_suites: list[str],
    seeds: list[int],
    cfg: DictConfig | None = None,
    phase: str | None = None,
) -> bool:
    """Return True if any eval+seed combination is not yet done."""
    if cfg is not None and _is_decoder_progression(cfg):
        return any(
            not _decoder_seed_done(entry, eval_name, seed, cfg=cfg)
            for eval_name in eval_suites
            for seed in _eval_seeds_for_suite(eval_name, seeds)
        )

    return cfg is not None and bool(
        _pending_eval_work_keys(entry, eval_suites, seeds, cfg, phase)
    )


def _evaluate_single_checkpoint_impl(
    spec: CheckpointSpec,
    device: torch.device,
    ptdtype: torch.dtype,
    eval_suites: list[str],
    seeds: list[int],
    cfg: DictConfig,
    existing_entry: dict | None = None,
    work_keys: set[EvalWorkKey] | None = None,
) -> dict:
    """Evaluate a single checkpoint on a specific device. Returns eval results dict.

    This is the core per-checkpoint work unit, used by both single-GPU and
    multi-GPU code paths. It loads the model, runs all requested evals/seeds,
    and returns the results without touching the shared JSON file.
    """
    cfg = _cfg_for_checkpoint(cfg, spec)
    if "harmfulness" in eval_suites and not _is_decoder_progression(cfg):
        _validate_harmfulness_conditioning(cfg, spec.phase)
    is_remote = cfg.checkpoint_source == "remote"
    _validate_cleanup_after_eval(cfg)
    # Split eval cells share one staged checkpoint. Their coordinator cleans it
    # only after every rank has acknowledged its completed cells.
    do_cleanup = is_remote and bool(cfg.get("cleanup_after_eval", False)) and work_keys is None
    device_type = "cuda" if device.type == "cuda" else device.type
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # Resolve checkpoint directory
    if is_remote:
        local_dir = fetch_remote_checkpoint(spec, cfg)
    else:
        local_dir = resolve_local_checkpoint_dir(spec, cfg)

    # Load model
    try:
        with timed_region("load_progression_model"):
            model, tokenizer, _meta, checkpoint_contract = load_progression_model(
                local_dir,
                spec.step,
                device,
                cfg,
            )
        if device.type == "cuda":
            model = model.to(dtype=ptdtype)
    except Exception as e:
        if do_cleanup:
            cleanup_remote_checkpoint(local_dir, spec.step)
        raise RuntimeError(f"Failed to load model: {e}") from e

    model_path = Path(local_dir) / f"model_{spec.step:06d}.pt"
    if model_path.is_file():
        current_identity = _local_checkpoint_identity(local_dir, spec.step)
        persisted_identity = (
            existing_entry.get("checkpoint_identity") if existing_entry is not None else None
        )
        if persisted_identity is not None and persisted_identity != current_identity:
            logger.warning("Checkpoint changed after resume validation; rerunning all measurements")
            existing_entry = None
            work_keys = None
        spec.checkpoint_identity = current_identity
        _meta["loaded_model_sha256"] = current_identity["model"]["sha256"]

    # Build eval results, reusing any already-completed seeds.
    evals: dict = {}
    if existing_entry:
        evals = copy.deepcopy(existing_entry.get("evals", {}))

    def should_run(
        eval_name: str,
        variant: str,
        config_name: str,
        seed: int,
    ) -> bool:
        if work_keys is None:
            return True
        return _eval_work_key(eval_name, variant, config_name, seed) in work_keys

    def has_requested_work(
        eval_name: str,
        variant: str,
        config_name: str,
        eval_seeds: list[int],
    ) -> bool:
        if work_keys is None:
            return True
        return any(should_run(eval_name, variant, config_name, seed) for seed in eval_seeds)

    def eval_pass_tags(
        eval_name: str,
        variant: str,
        config_name: str,
        seed: int | None = None,
    ):
        tags: dict[str, Any] = {
            "suite": eval_name,
            "eval_variant": variant,
            "conditioned_config": config_name,
        }
        if seed is not None:
            tags["seed"] = int(seed)
        return timing_tags(**tags)

    try:
        conditioned_configs = conditioned_configs_for_phase(cfg, spec.phase)
        has_conditioned_variants = len(conditioned_configs) > 0
        provider_contract_kwargs: dict[str, Any] = {}
        if _is_encoder_decoder_model(model):
            assert checkpoint_contract is not None
            feature_vocab = conditioned_eval_feature_vocab(model)
            contract_metadata = conditioned_eval_cache_contract_metadata(
                checkpoint_contract
            )
            if feature_vocab is not None:
                provider_contract_kwargs["expected_feature_vocab"] = feature_vocab
            if contract_metadata:
                provider_contract_kwargs["expected_contract_metadata"] = contract_metadata
        if has_conditioned_variants:
            if not _is_encoder_decoder_model(model):
                raise ValueError(
                    "conditioned_configs were provided, but the checkpoint is decoder-only"
                )
            _validate_conditioned_configs(model, conditioned_configs, _meta)
        if has_conditioned_variants:
            enc_cfg = model.encoder.config
            model_context_len = int(model.dec_config.sequence_len)
            static_context_provider = _context_provider_for_cfg(
                cfg,
                model_max_seq_len=model_context_len,
                expected_n_sae_features=enc_cfg.n_sae_features,
                expected_num_buckets=enc_cfg.num_buckets,
                **provider_contract_kwargs,
            )
            with timed_region("validate_context_coverage"):
                _assert_static_conditioning_context_coverage(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    provider=static_context_provider,
                    eval_suites=eval_suites,
                    seeds=seeds,
                    model_max_seq_len=model_context_len,
                    surface=(
                        "training_progression static conditioning "
                        f"phase={spec.phase} step={spec.step}"
                    ),
                )

        # PR-D Batch H2 (CI-5): pre-compute target_real_configs +
    # providers once per checkpoint. The per-suite loop below iterates these for each
        # eval_name — keying under evals[eval_name] so multi-suite runs
        # don't conflate.
        target_real_raw = list(_conditioning_cfg(cfg).get("target_real_configs", []) or [])
        if target_real_raw and _is_encoder_decoder_model(model) and any(
            suite in {"core", "chat"} for suite in eval_suites
        ):
            target_real_configs_parsed = parse_target_real_configs_from_cfg(
                cfg.conditioning,
                suite=f"{spec.phase}_target",
                model_max_seq_len=int(model.dec_config.sequence_len),
            )
            target_real_providers = _target_real_providers_for_cfg(
                cfg,
                model_max_seq_len=int(model.dec_config.sequence_len),
                target_real_configs=target_real_configs_parsed,
                **provider_contract_kwargs,
            )
        else:
            target_real_configs_parsed = []
            target_real_providers = {}
        target_mode = TargetConditioningMode.from_next_segment_offset(
            resolve_next_segment_conditioning_offset(
                {"next_segment_conditioning_offset": "auto"},
                _meta,
            )
        )
        with autocast_ctx:
            for eval_name in eval_suites:
                eval_seeds = _eval_seeds_for_suite(eval_name, seeds)

                if eval_name == "harmfulness" and _self_conditioned_cfg(cfg):
                    if not _is_encoder_decoder_model(model):
                        raise ValueError("self_conditioned harmfulness requires an encoder-decoder checkpoint")
                    assert checkpoint_contract is not None
                    config_result = evals.setdefault(eval_name, {}).setdefault(
                        "conditioned", {}
                    ).setdefault("configs", {}).setdefault(_SELF_CONDITIONED_CONFIG_NAME, {})
                    config_result.update({
                        "kind": SuiteResultKind.SELF_CONDITIONED.value,
                        "cache_dir": str(_conditioning_cfg(cfg)["context_cache_dir"]),
                    })
                    recorded_seeds = config_result.setdefault("seeds", {})
                    for seed in eval_seeds:
                        if not should_run(eval_name, "conditioned", _SELF_CONDITIONED_CONFIG_NAME, seed):
                            continue
                        if _eval_seed_payload_done(
                            recorded_seeds.get(str(seed)), eval_name=eval_name, seed=seed, cfg=cfg
                        ):
                            continue
                        with eval_pass_tags(eval_name, "conditioned", _SELF_CONDITIONED_CONFIG_NAME, seed):
                            recorded_seeds[str(seed)] = run_self_conditioned_harmfulness_eval(
                                model=model, tokenizer=tokenizer, device=device, seed=seed, cfg=cfg,
                                meta=_meta, feature_contract=checkpoint_contract,
                            )
                    continue

                if _is_decoder_progression(cfg):
                    if _is_encoder_decoder_model(model):
                        raise ValueError(
                            "The unconditional decoder progression variant cannot "
                            "evaluate an EncoderDecoderGPT checkpoint"
                        )
                    eval_entry = evals.setdefault(eval_name, {"seeds": {}})
                    decoder_seeds = eval_entry.setdefault("seeds", {})
                    for seed in eval_seeds:
                        if not should_run(
                            eval_name,
                            _DECODER_WORK_VARIANT,
                            "",
                            seed,
                        ):
                            continue
                        if _eval_seed_payload_done(
                            decoder_seeds.get(str(seed)),
                            eval_name=eval_name,
                            seed=seed,
                            cfg=cfg,
                        ):
                            logger.info(
                                f"  [{device}] {eval_name}/decoder seed={seed} already done"
                            )
                            continue
                        logger.info(f"  [{device}] Running {eval_name}/decoder (seed={seed})...")
                        try:
                            with eval_pass_tags(
                                eval_name,
                                _DECODER_WORK_VARIANT,
                                "",
                                seed,
                            ):
                                result = _run_eval_suite(
                                    eval_name,
                                    model,
                                    tokenizer,
                                    device,
                                    seed,
                                    cfg,
                                )
                            if result is not None:
                                decoder_seeds[str(seed)] = result
                                logger.info(
                                    f"  [{device}] {eval_name}/decoder seed={seed} complete"
                                )
                        except Exception as e:
                            logger.error(
                                f"  [{device}] {eval_name}/decoder seed={seed} failed: {e}"
                            )
                            if eval_name == "harmfulness":
                                raise
                    continue

                if not has_conditioned_variants:
                    has_target_real = bool(target_real_configs_parsed)
                    if not has_target_real:
                        continue
                    if has_target_real:
                        eval_data = evals.setdefault(eval_name, {})
                        _run_target_real_for_suite(
                            eval_name=eval_name,
                            eval_data=eval_data,
                            eval_seeds=eval_seeds,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            cfg=cfg,
                            target_real_providers=target_real_providers,
                            target_real_configs=target_real_configs_parsed,
                            target_mode=target_mode,
                            eval_pass_tags=eval_pass_tags,
                            should_run=should_run,
                        )
                    continue

                requested_conditioned_configs = [
                    feature_cfg
                    for feature_cfg in conditioned_configs
                    if has_requested_work(
                        eval_name,
                        "conditioned",
                        feature_cfg.name,
                        eval_seeds,
                    )
                ]
                has_target_real = bool(target_real_configs_parsed)
                if not requested_conditioned_configs and not has_target_real:
                    continue

                eval_data = evals.setdefault(eval_name, {})

                if not requested_conditioned_configs:
                    # Skip the conditioned_configs loop entirely, but still
                    # run target_real below (CI-5 — target_real runs per
                    # eval_name independent of conditioned_configs).
                    if has_target_real:
                        _run_target_real_for_suite(
                            eval_name=eval_name,
                            eval_data=eval_data,
                            eval_seeds=eval_seeds,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            cfg=cfg,
                            target_real_providers=target_real_providers,
                            target_real_configs=target_real_configs_parsed,
                            target_mode=target_mode,
                            eval_pass_tags=eval_pass_tags,
                            should_run=should_run,
                        )
                    continue

                cond_root = eval_data.setdefault("conditioned", {}).setdefault(
                    "configs",
                    {},
                )
                for feature_cfg in requested_conditioned_configs:
                    resolved_context_cache_dir = str(static_context_provider.cache_dir)
                    cfg_result = cond_root.setdefault(
                        feature_cfg.name,
                        {
                            "kind": "static_yaml",
                            "display_name": feature_cfg.display_name,
                            "target_features_file": feature_cfg.source,
                            "context_cache_dir": resolved_context_cache_dir,
                            "target_conditioning_mode": target_mode.value,
                            "seeds": {},
                        },
                    )
                    cfg_result.setdefault("seeds", {})
                    cfg_result["display_name"] = feature_cfg.display_name
                    cfg_result["target_features_file"] = feature_cfg.source
                    cfg_result["context_cache_dir"] = resolved_context_cache_dir
                    cfg_result["target_conditioning_mode"] = target_mode.value
                    with (
                        eval_pass_tags(
                            eval_name,
                            "conditioned",
                            feature_cfg.name,
                        ),
                        timed_region("build_eval_wrapper"),
                    ):
                        eval_model = _wrap_conditioned_model(
                            model,
                            tokenizer,
                            feature_cfg,
                            static_context_provider,
                            target_mode,
                        )
                    for seed in eval_seeds:
                        if not should_run(
                            eval_name,
                            "conditioned",
                            feature_cfg.name,
                            seed,
                        ):
                            continue
                        if _eval_seed_payload_done(
                            cfg_result["seeds"].get(str(seed)),
                            eval_name=eval_name,
                            seed=seed,
                            cfg=cfg,
                        ):
                            logger.info(
                                f"  [{device}] {eval_name}/{feature_cfg.name} seed={seed} already done"
                            )
                            continue
                        logger.info(
                            f"  [{device}] Running {eval_name}/{feature_cfg.name} (seed={seed})..."
                        )
                        try:
                            with eval_pass_tags(
                                eval_name,
                                "conditioned",
                                feature_cfg.name,
                                seed,
                            ):
                                result = _run_eval_suite(
                                    eval_name,
                                    eval_model,
                                    tokenizer,
                                    device,
                                    seed,
                                    cfg,
                                )
                            if result is not None:
                                cfg_result["seeds"][str(seed)] = result
                                logger.info(
                                    f"  [{device}] {eval_name}/{feature_cfg.name} seed={seed} complete"
                                )
                        except KeyError as e:
                            logger.error(
                                f"  [{device}] {eval_name}/{feature_cfg.name} seed={seed} failed: {e}"
                            )
                            raise
                        except Exception as e:
                            logger.error(
                                f"  [{device}] {eval_name}/{feature_cfg.name} seed={seed} failed: {e}"
                            )
                            if eval_name == "harmfulness":
                                raise

                # PR-D Batch H2 (CI-1/CI-2/CI-5/CI-6): per-suite
                # target_real loop (after the conditioned_configs branch).
                if has_target_real:
                    _run_target_real_for_suite(
                        eval_name=eval_name,
                        eval_data=eval_data,
                        eval_seeds=eval_seeds,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        cfg=cfg,
                        target_real_providers=target_real_providers,
                        target_real_configs=target_real_configs_parsed,
                        target_mode=target_mode,
                        eval_pass_tags=eval_pass_tags,
                        should_run=should_run,
                    )
    finally:
        if do_cleanup:
            cleanup_remote_checkpoint(local_dir, spec.step)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return evals


def evaluate_single_checkpoint(
    spec: CheckpointSpec,
    device: torch.device,
    ptdtype: torch.dtype,
    eval_suites: list[str],
    seeds: list[int],
    cfg: DictConfig,
    existing_entry: dict | None = None,
    work_keys: set[EvalWorkKey] | None = None,
) -> dict:
    with timed_region("evaluate_single_checkpoint"):
        return _evaluate_single_checkpoint_impl(
            spec,
            device,
            ptdtype,
            eval_suites,
            seeds,
            cfg,
            existing_entry,
            work_keys,
        )


def collect_rank_results(
    rank_fn: Callable[[int], Any],
    world_size: int,
    *,
    max_workers: int | None = None,
) -> dict:
    """Dispatch per-rank work in parallel and classify failures.

    Runs ``rank_fn(rank)`` for each rank in ``[0, world_size)`` via a
    ``ThreadPoolExecutor`` so CI can exercise the classification logic
    without requiring CUDA or a real multi-process torchrun setup. The
    production multi-GPU path uses the same any-failure rule after
    gathering rank-local failures across the torchrun workers.

    Behavior:
      * On a per-rank exception, logs the rank id AND the full traceback at
        ``ERROR`` severity and records the rank id in the failure list.
      * If any rank fails, raises ``RuntimeError`` summarizing the failed
        ranks. This prevents silently consuming partial eval output.
      * Otherwise returns an aggregate ``{"rank_results": [...],
        "partial_ranks_failed": []}``.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    successes: list[tuple[int, Any]] = []
    failures: list[int] = []
    first_exc: BaseException | None = None

    effective_max_workers = max_workers if max_workers is not None else world_size
    if effective_max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {effective_max_workers}")

    def _handle_result(rank: int, result_fn: Callable[[], Any]) -> None:
        nonlocal first_exc
        try:
            result = result_fn()
        except KeyError:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(
                f"rank {rank} failed with {type(exc).__name__}: {exc}\n"
                f"Traceback (from rank {rank}):\n{tb}"
            )
            failures.append(rank)
            if first_exc is None:
                first_exc = exc
        else:
            successes.append((rank, result))

    if effective_max_workers == 1:
        for rank in range(world_size):
            _handle_result(rank, lambda r=rank: rank_fn(r))
    else:
        with ThreadPoolExecutor(max_workers=effective_max_workers) as pool:
            ordered_futures = [(pool.submit(rank_fn, r), r) for r in range(world_size)]
            for fut, rank in ordered_futures:
                _handle_result(rank, fut.result)

    failures.sort()
    _raise_on_any_rank_failure(failures, world_size, cause=first_exc)

    successes.sort(key=lambda pair: pair[0])
    aggregate: dict[str, Any] = {
        "rank_results": [r for _rank, r in successes],
        "partial_ranks_failed": [],
    }
    return aggregate


def _raise_on_any_rank_failure(
    failed_ranks: list[int],
    world_size: int,
    *,
    cause: BaseException | None = None,
) -> None:
    """Raise if any DDP rank failed."""
    failures = sorted(set(failed_ranks))
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {world_size} ranks failed "
            f"(any rank failure is fatal); ranks={failures}. "
            "See ERROR logs above for per-rank tracebacks."
        ) from cause


def collect_results(cfg: DictConfig) -> None:
    """Evaluate checkpoints and save results incrementally.

    Checkpoint source is controlled by ``cfg.checkpoint_source``:
      - ``"local"`` (default): reads from the local filesystem
      - ``"remote"``: SCP each checkpoint on demand

    When ``num_gpus > 1``, torchrun launches one process per GPU and stripes
    pending evaluation work across ranks.

    When ``checkpoint_source=remote``, the next checkpoint is prefetched
    in a background thread while the current one is being evaluated.
    Set ``cleanup_after_eval=true`` to delete each checkpoint's ``.pt``
    file after evaluation (useful when disk space is limited).
    """
    _validate_cleanup_after_eval(cfg)
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ptdtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16

    is_remote = cfg.checkpoint_source == "remote"
    specs = build_checkpoint_specs(cfg)
    results = load_or_init_results(cfg)
    eval_suites = [e.strip() for e in cfg.evals.split(",")]
    seeds = [cfg.base_seed + i * 1000 for i in range(cfg.num_seeds)]

    # Determine GPU count.
    num_gpus = cfg.get("num_gpus", 1)
    if num_gpus is None and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    num_gpus = max(num_gpus or 1, 1)

    logger.info(
        f"Collecting: {len(specs)} checkpoints, {eval_suites} evals, "
        f"{len(seeds)} seeds, source={cfg.checkpoint_source}, gpus={num_gpus}"
    )

    # Filter to specs that still need evaluation.
    # Check per-checkpoint result files (robust to interruption).
    pending_specs = [
        spec
        for spec in specs
        if _needs_eval(
            _existing_checkpoint_entry(cfg, results, spec),
            eval_suites,
            seeds,
            _cfg_for_checkpoint(cfg, spec),
            spec.phase,
        )
    ]
    logger.info(f"  {len(pending_specs)} checkpoints need evaluation")

    if not pending_specs:
        logger.info("Nothing to do")
        return

    _validate_required_context_cache(cfg, pending_specs)

    # --- Multi-GPU path (torchrun) ---
    if num_gpus > 1 and device_type == "cuda":
        rank = int(os.environ.get("RANK", "-1"))
        if rank == -1:
            # Not launched via torchrun — relaunch ourselves.
            snapshot_path = _results_dir(cfg) / "torchrun_config.yaml"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            snapshot_cfg.mode = "collect"
            snapshot_cfg.num_gpus = num_gpus
            OmegaConf.save(snapshot_cfg, snapshot_path)

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "-m",
                "scripts.train.training_progression",
                "mode=collect",
                f"num_gpus={num_gpus}",
                f"config_snapshot={snapshot_path}",
            ]
            # Forward hydra overrides. Skip comma-containing values
            # (e.g. evals=core,chat,harmfulness) as they're in the
            # config file and Hydra misparses bare commas.
            for key in [
                "model_tag",
                "checkpoint_source",
                "num_seeds",
                "base_seed",
                "max_per_task",
                "core_example_batch_size",
                "chat_batch_size",
                "chat_max_problems",
                "chat_max_new_tokens",
                "harm_temperature",
                "harm_num_benign",
                "harm_num_harmful",
                "dtype",
                "checkpoints_dir",
                "cleanup_after_eval",
                "results_file",
                "fig_dir",
                "device_type",
                "model_family",
                "checkpoint_root",
                "checkpoint_prefix",
            ]:
                val = cfg.get(key)
                if val is not None and "," not in str(val) and str(val) != "":
                    cmd.append(f"{key}={val}")
            logger.info(f"Relaunching with torchrun: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"torchrun exited with code {result.returncode}")
            logger.info("Collection complete")
            return

        # We are inside a torchrun worker.
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Clear torchrun env vars so eval code (core_eval, chat_eval)
        # treats each rank as a single-GPU process. Each rank evaluates
        # independent checkpoints — no cross-rank communication needed.
        # We keep rank/world_size locally for striping and the final barrier.
        saved_env = {}
        for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            saved_env[var] = os.environ.pop(var, None)

        logger.info(f"[rank {rank}/{world_size}] Evaluating on {device}")

        # Stripe checkpoints across ranks. When the resumed tail has fewer
        # checkpoints than ranks, split the remaining work by
        # eval/variant/seed so SFT-only tails do not leave most GPUs idle.
        split_eval_work = len(pending_specs) < world_size
        rank_failures: list[dict[str, Any]] = []
        rank_entries: list[dict[str, Any]] = []
        split_queue_paths: tuple[Path, Path] | None = None

        if split_eval_work:
            all_work: list[tuple[CheckpointSpec, EvalWorkKey]] = []
            for spec in pending_specs:
                existing = _existing_checkpoint_entry(cfg, results, spec)
                for work_key in _pending_eval_work_keys(
                    existing,
                    eval_suites,
                    seeds,
                    _cfg_for_checkpoint(cfg, spec),
                    spec.phase,
                ):
                    all_work.append((spec, work_key))
            logger.info(
                f"[rank {rank}/{world_size}] Splitting {len(all_work)} "
                f"eval/seed work unit(s) across {world_size} rank(s) for "
                f"{len(pending_specs)} pending checkpoint(s)"
            )
            queue_id = str(os.getppid())
            queue_path, lock_path = _split_work_queue_paths(cfg, queue_id)
            split_queue_paths = (queue_path, lock_path)
            _initialize_split_work_queue(queue_path, lock_path, all_work)

            completed_units = 0
            while True:
                work_item = _pop_split_work_item(queue_path, lock_path)
                if work_item is None:
                    break
                spec, work_key = work_item
                existing = _existing_checkpoint_entry(cfg, results, spec)
                try:
                    evals = evaluate_single_checkpoint(
                        spec,
                        device,
                        ptdtype,
                        eval_suites,
                        seeds,
                        cfg,
                        existing,
                        work_keys={work_key},
                    )
                    rank_entries.append(
                        {
                            "step": spec.step,
                            "phase": spec.phase,
                            "evals": evals,
                            "checkpoint_identity": spec.checkpoint_identity,
                        }
                    )
                    completed_units += 1
                    logger.info(
                        f"  [rank {rank}] Finished split work unit {work_key} "
                        f"for {spec.phase} step {spec.step}"
                    )
                except KeyError:
                    if rank == 0 and split_queue_paths is not None:
                        _cleanup_split_work_queue(*split_queue_paths)
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(
                        f"  [rank {rank}] Failed {spec.phase} step {spec.step}: "
                        f"{type(e).__name__}: {e}\n"
                        f"Traceback (from rank {rank}):\n{tb}"
                    )
                    rank_failures.append(
                        {
                            "rank": rank,
                            "phase": spec.phase,
                            "step": spec.step,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
            logger.info(f"  [rank {rank}] Finished {completed_units} split unit(s)")
        else:
            my_specs = pending_specs[rank::world_size]

            def _eval_one_spec(spec_idx: int) -> dict[str, Any]:
                spec = my_specs[spec_idx]
                existing = _existing_checkpoint_entry(cfg, results, spec)
                evals = evaluate_single_checkpoint(
                    spec, device, ptdtype, eval_suites, seeds, cfg, existing
                )
                save_checkpoint_result(cfg, spec.step, spec.phase, evals, spec.checkpoint_identity)
                logger.info(f"  [rank {rank}] Saved {spec.phase} step {spec.step}")
                return {"step": spec.step, "phase": spec.phase}

            if my_specs:
                try:
                    collect_rank_results(
                        _eval_one_spec,
                        world_size=len(my_specs),
                        max_workers=1,
                    )
                except KeyError:
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(
                        f"  [rank {rank}] Failed local checkpoint stripe: "
                        f"{type(e).__name__}: {e}\n"
                        f"Traceback (from rank {rank}):\n{tb}"
                    )
                    first_spec = my_specs[0]
                    rank_failures.append(
                        {
                            "rank": rank,
                            "phase": first_spec.phase,
                            "step": first_spec.step,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

        # Restore env vars and init process group to synchronize completion
        # and gather rank-local failures. A single torchrun worker has no
        # peer ranks to gather from, so avoid initializing distributed state.
        for var, val in saved_env.items():
            if val is not None:
                os.environ[var] = val
        process_group_initialized = False
        if world_size > 1:
            dist.init_process_group(backend="nccl", device_id=device)
            process_group_initialized = True
            gathered_failures: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered_failures, rank_failures)
            if split_eval_work:
                gathered_entries: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
                dist.all_gather_object(gathered_entries, rank_entries)
        else:
            gathered_failures = [rank_failures]
        failed_ranks = sorted(
            {
                int(failure["rank"])
                for failures_for_rank in gathered_failures
                for failure in failures_for_rank
            }
        )
        if failed_ranks:
            logger.error(
                f"[rank {rank}] Rank failure detected: "
                f"{len(failed_ranks)}/{world_size} ranks failed ({failed_ranks})"
            )
        try:
            _raise_on_any_rank_failure(failed_ranks, world_size)
        except RuntimeError:
            if split_eval_work and rank == 0 and split_queue_paths is not None:
                _cleanup_split_work_queue(*split_queue_paths)
            if process_group_initialized:
                dist.destroy_process_group()
            raise

        # Rank 0 merges split partials, then all per-checkpoint files into
        # results.json.
        if rank == 0:
            if split_eval_work:
                merged_by_spec: dict[tuple[str, int], dict] = {}
                for spec in pending_specs:
                    base = {"step": spec.step, "phase": spec.phase, "evals": {}}
                    existing = _existing_checkpoint_entry(cfg, results, spec)
                    if existing:
                        _merge_checkpoint_entry(base, existing)
                    merged_by_spec[(spec.phase, spec.step)] = base

                for entries_for_rank in gathered_entries:
                    for entry in entries_for_rank:
                        spec_id = (entry["phase"], entry["step"])
                        base = merged_by_spec.setdefault(
                            spec_id,
                            {
                                "step": entry["step"],
                                "phase": entry["phase"],
                                "evals": {},
                            },
                        )
                        _merge_checkpoint_entry(base, entry)

                for spec in pending_specs:
                    entry = merged_by_spec[(spec.phase, spec.step)]
                    save_checkpoint_result(
                        cfg,
                        spec.step,
                        spec.phase,
                        entry.get("evals", {}),
                        entry.get("checkpoint_identity"),
                    )
                    if _needs_eval(entry, eval_suites, seeds, _cfg_for_checkpoint(cfg, spec), spec.phase):
                        logger.warning(
                            f"Split eval left incomplete results for "
                            f"{spec.phase} step {spec.step}; a later resume "
                            "will retry missing units"
                        )
                    else:
                        logger.info(f"Saved split results for {spec.phase} step {spec.step}")
                        if is_remote and bool(cfg.get("cleanup_after_eval", False)):
                            cleanup_remote_checkpoint(
                                str(_remote_checkpoint_local_dir(spec, cfg)),
                                spec.step,
                            )
                if split_queue_paths is not None:
                    _cleanup_split_work_queue(*split_queue_paths)
            save_merged_results(cfg)
            logger.info(f"Merged checkpoint results into {_results_path(cfg)}")

        if process_group_initialized:
            dist.destroy_process_group()
        logger.info(f"[rank {rank}] Done")
        return

    # --- Single-GPU path (with remote prefetch) ---
    device = torch.device(device_type)
    prefetch_executor = ThreadPoolExecutor(max_workers=1) if is_remote else None
    prefetch_future = None

    for i, spec in enumerate(pending_specs):
        logger.info(f"=== {spec.phase} step {spec.step} ===")

        # In remote mode, collect prefetch or start fresh fetch.
        if is_remote:
            if prefetch_future is not None:
                try:
                    prefetch_future.result()
                except Exception:
                    pass  # fetch_remote_checkpoint will be retried below
                prefetch_future = None

            # Kick off prefetch for the next pending checkpoint.
            if prefetch_executor is not None and i + 1 < len(pending_specs):
                next_spec = pending_specs[i + 1]
                logger.info(f"  Prefetching next: {next_spec.phase} step {next_spec.step}")
                prefetch_future = prefetch_executor.submit(fetch_remote_checkpoint, next_spec, cfg)

        existing = _existing_checkpoint_entry(cfg, results, spec)
        try:
            evals = evaluate_single_checkpoint(
                spec, device, ptdtype, eval_suites, seeds, cfg, existing
            )
            save_checkpoint_result(cfg, spec.step, spec.phase, evals, spec.checkpoint_identity)
            logger.info(f"  Saved results for {spec.phase} step {spec.step}")
        except (RuntimeError, FileNotFoundError) as e:
            logger.error(f"  Failed: {e}")

    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=False)

    # Merge all per-checkpoint files into results.json for plotting.
    save_merged_results(cfg)
    logger.info("Collection complete")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


# Metric specifications: (eval_name, metric_key_path, display_name, filename)
METRIC_SPECS = [
    # CORE
    (["core", "core_metric"], "CORE Metric", "core_metric"),
    # Validation BPB — populated via the "val" eval namespace in results.json.
    # Charts are skipped gracefully when the "val" eval suite has not yet
    # been wired into the progression collection run.
    (["val", "bpb"], "Validation BPB (conditioned)", "val_bpb"),
    # Chat tasks
    (["chat", "ARC-Easy"], "ARC-Easy Accuracy", "chat_arc_easy"),
    (["chat", "ARC-Challenge"], "ARC-Challenge Accuracy", "chat_arc_challenge"),
    (["chat", "MMLU"], "MMLU Accuracy", "chat_mmlu"),
    (["chat", "GSM8K"], "GSM8K Accuracy", "chat_gsm8k"),
    (["chat", "HumanEval"], "HumanEval Accuracy", "chat_humaneval"),
    (["chat", "SpellingBee"], "SpellingBee Accuracy", "chat_spellingbee"),
    (["chat", "chatcore_metric"], "ChatCORE Metric", "chatcore_metric"),
    # Harmfulness
    (
        ["harmfulness", "harmful_prompts", "asr"],
        "Attack Success Rate",
        "harmfulness_asr",
    ),
    (
        ["harmfulness", "overall", "mean_score"],
        "Harmfulness Mean Score",
        "harmfulness_mean_score",
    ),
    (
        ["harmfulness", "harmful_prompts", "safe_response_rate"],
        "Harmful-Prompt Safe Response Rate (score = 1)",
        "harmfulness_harmful_safe_response_rate",
    ),
    (
        ["harmfulness", "benign_prompts", "compliance_rate"],
        "Benign Compliance Rate (non-refusal)",
        "harmfulness_benign_compliance_rate",
    ),
]


def _preflight_harmfulness_plot_schemas(checkpoints: list[dict]) -> None:
    """Reject stale harmfulness seeds before any progression metric extraction."""

    for checkpoint in checkpoints:
        phase = checkpoint.get("phase")
        step = checkpoint.get("step")
        harmfulness = checkpoint.get("evals", {}).get("harmfulness")
        if not isinstance(harmfulness, Mapping):
            continue
        seed_groups: list[tuple[str, Any]] = [
            ("seeds", harmfulness.get("seeds")),
        ]
        conditioned = harmfulness.get("conditioned")
        if isinstance(conditioned, Mapping):
            configs = conditioned.get("configs")
            if isinstance(configs, Mapping):
                for config_name, config_data in configs.items():
                    config_seeds = (
                        config_data.get("seeds")
                        if isinstance(config_data, Mapping)
                        else None
                    )
                    seed_groups.append(
                        (f"conditioned.configs[{config_name!r}].seeds", config_seeds)
                    )
        for group_path, seeds in seed_groups:
            if seeds is None:
                continue
            if not isinstance(seeds, Mapping):
                raise ValueError(
                    f"checkpoint phase={phase} step={step} harmfulness.{group_path} "
                    "must be a mapping"
                )
            for seed_id, payload in seeds.items():
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"checkpoint phase={phase} step={step} harmfulness.{group_path}"
                        f"[{seed_id!r}] must be a mapping"
                    )
                schema_version = payload.get("harmfulness_eval_schema_version")
                if schema_version != HARMFULNESS_EVAL_SCHEMA_VERSION:
                    raise ValueError(
                        f"checkpoint phase={phase} step={step} harmfulness.{group_path}"
                        f"[{seed_id!r}].harmfulness_eval_schema_version must equal "
                        f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got {schema_version!r}"
                    )


def _decoder_series_for_metric(
    checkpoints: list[dict],
    *,
    metric_path: list[str],
    phase_order: dict[str, int],
) -> ChartSeries | None:
    """Extract the unconditional decoder series from top-level seed payloads."""

    eval_name = metric_path[0]
    tail_keys = metric_path[1:]
    ordered = sorted(
        checkpoints,
        key=lambda checkpoint: (
            phase_order.get(str(checkpoint.get("phase", "")), 10_000),
            int(checkpoint.get("step", 0)),
        ),
    )
    steps: list[int] = []
    phases: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for checkpoint in ordered:
        seeds = checkpoint.get("evals", {}).get(eval_name, {}).get("seeds", {})
        values: list[float] = []
        for seed_result in seeds.values():
            value: Any = seed_result
            for key in tail_keys:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
        if not values:
            continue
        steps.append(int(checkpoint["step"]))
        phases.append(str(checkpoint["phase"]))
        means.append(float(statistics.fmean(values)))
        stds.append(float(statistics.pstdev(values)) if len(values) > 1 else 0.0)
    if not steps:
        return None
    return ChartSeries(
        label="Decoder (unconditional)",
        steps=tuple(steps),
        phases=tuple(phases),
        means=tuple(means),
        stds=tuple(stds),
        color=BASELINE_COLOR,
        linestyle=BASELINE_LINESTYLE,
        is_baseline=True,
    )


def _build_progression_chart_spec(
    checkpoints: list[dict],
    *,
    metric_path: list[str],
    display_name: str,
    model_tag: str,
    phase_order: dict[str, int],
) -> MultiSeriesSpec | None:
    """Build a chart spec including the decoder-only baseline namespace."""

    variant_spec = build_multi_series_spec(
        checkpoints,
        metric_path=metric_path,
        display_name=display_name,
        model_tag=model_tag,
        phase_order=phase_order,
        linestyle_resolver=_variant_linestyle_resolver,
    )
    decoder_series = _decoder_series_for_metric(
        checkpoints,
        metric_path=metric_path,
        phase_order=phase_order,
    )
    if decoder_series is None:
        return variant_spec
    other_series = () if variant_spec is None else variant_spec.series
    series = (decoder_series, *other_series)
    return MultiSeriesSpec(
        title=f"{model_tag} — {display_name}",
        metric_name=display_name,
        model_tag=model_tag,
        series=series,
        phase_boundaries=_phase_boundaries_from_series(decoder_series),
        legend_outside=sum(not item.is_baseline for item in series) > 2,
    )


def plot_results(cfg: DictConfig) -> None:
    """Generate combined progression charts from saved results.

    Each metric in ``METRIC_SPECS`` produces one PNG when configured
    conditioned or target-real results exist for that metric. Encoder-decoder
    progressions overlay one line per configured conditioning variant.

    The merged ``results.json`` is filtered to checkpoints whose
    ``(step, phase)`` key appears in the active ``cfg.phases`` so charts and
    JSON stay aligned with the requested checkpoint set. Per-checkpoint files
    under ``_results_dir(cfg)/checkpoints/`` are never modified by this
    filter, so re-adding a step to ``cfg.phases`` restores it on the next
    merge. A config with no overlap with on-disk results emits an empty
    merged JSON and skips chart generation; this is recoverable because the
    per-checkpoint files persist.
    """
    results = merge_checkpoint_results(cfg)
    if results.get("checkpoints"):
        results = _filter_results_to_configured_checkpoints(results, cfg)
        save_results(results, cfg)
    else:
        results = load_or_init_results(cfg)
        results = _filter_results_to_configured_checkpoints(results, cfg)
    fig_dir = _fig_dir(cfg)
    fig_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = results.get("checkpoints", [])
    if not checkpoints:
        logger.warning("No checkpoints in results file, nothing to plot")
        return
    _preflight_harmfulness_plot_schemas(checkpoints)

    phase_order: dict[str, int] = {}
    if cfg.get("phases"):
        for idx, phase_cfg in enumerate(cfg.phases):
            phase_name = str(phase_cfg.phase)
            if phase_name not in phase_order:
                phase_order[phase_name] = idx

    plotted = 0
    for metric_path, display_name, filename in METRIC_SPECS:
        spec = _build_progression_chart_spec(
            checkpoints,
            metric_path=list(metric_path),
            display_name=display_name,
            model_tag=cfg.model_tag,
            phase_order=phase_order,
        )
        if spec is None:
            logger.info(f"No data for {display_name}, skipping")
            continue
        fig = render_multi_series_chart(spec)
        out_path = fig_dir / f"{filename}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        plotted += 1

    logger.info(f"Generated {plotted} charts in {fig_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _validate_pr_d_target_real_configs(cfg) -> None:
    """Hard-fail at config-load when target_real_configs has name collisions."""
    cond = cfg.get("conditioning", None)
    if cond is None:
        return
    _validate_no_legacy_conditioning_modes(cfg)
    target_real = list(cond.get("target_real_configs", []) or [])
    if not target_real:
        return

    for idx, trc in enumerate(target_real):
        if not isinstance(trc, Mapping):
            raise ValueError(
                f"target_real_configs[{idx}]: each entry must be a YAML mapping; "
                f"got {type(trc).__name__} {trc!r}."
            )
    target_real_names = [str(trc["name"]) for trc in target_real]
    reserved = {"unconditioned", "unconditional"}
    reserved_used = sorted(set(target_real_names) & reserved)
    if reserved_used:
        raise ValueError(
            "target_real_configs uses reserved no-context names that are no "
            f"longer supported: {reserved_used}"
        )
    if len(set(target_real_names)) != len(target_real_names):
        seen: dict[str, int] = {}
        for n in target_real_names:
            seen[n] = seen.get(n, 0) + 1
        dups = [n for n, c in seen.items() if c > 1]
        raise ValueError(f"target_real_configs has duplicate names: {sorted(dups)}")

    conditioned_names = _configured_conditioned_config_names(cond.get("conditioned_configs"))
    collision = set(target_real_names) & set(conditioned_names)
    if collision:
        raise ValueError(
            f"name collision between conditioned_configs and target_real_configs: "
            f"{sorted(collision)}"
        )


def _configured_conditioned_config_names(raw) -> list[str]:
    """Return every static target-delta config name declared in raw YAML."""

    raw = _plain(raw)
    if raw is None:
        return []

    names: list[str] = []
    feature_keys = {"features", "features_file", "name"}

    def add_from_entry(entry, path: str, fallback_name: str | None = None) -> None:
        entry = _plain(entry)
        if isinstance(entry, str):
            names.append(fallback_name or Path(entry).stem)
            return
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"conditioned_configs{path} must be a mapping, list, or string path; "
                f"got {type(entry).__name__}"
            )
        if "name" in entry:
            names.append(str(entry["name"]))
            return
        if fallback_name is not None and any(key in entry for key in feature_keys):
            names.append(fallback_name)
            return
        add_from_mapping(entry, path)

    def add_from_mapping(mapping: Mapping, path: str) -> None:
        for key, value in mapping.items():
            plain_value = _plain(value)
            key_name = str(key)
            if plain_value is None:
                continue
            if isinstance(plain_value, list):
                for idx, entry in enumerate(plain_value):
                    add_from_entry(entry, f"{path}.{key_name}[{idx}]")
            elif isinstance(plain_value, Mapping):
                if any(feature_key in plain_value for feature_key in feature_keys):
                    add_from_entry(
                        plain_value,
                        f"{path}.{key_name}",
                        fallback_name=key_name,
                    )
                else:
                    add_from_mapping(plain_value, f"{path}.{key_name}")
            elif isinstance(plain_value, str):
                names.append(key_name)
            else:
                raise ValueError(
                    f"conditioned_configs{path}.{key_name} must be a mapping, "
                    f"list, string path, or null; got {type(plain_value).__name__}"
                )

    if isinstance(raw, str):
        return [Path(raw).stem]

    if isinstance(raw, list):
        for idx, entry in enumerate(raw):
            add_from_entry(entry, f"[{idx}]")
        return names

    if not isinstance(raw, Mapping):
        raise ValueError(
            "conditioning.conditioned_configs must be a mapping, list, string path, "
            f"or null; got {type(raw).__name__}"
        )

    if any(key in raw for key in feature_keys):
        add_from_entry(raw, "")
        return names

    add_from_mapping(raw, "")
    return names


def main() -> None:
    config_name, overrides = parse_config_name_and_overrides(
        sys.argv[1:], default_config_name=_CONFIG_NAME
    )
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    if cfg.get("config_snapshot"):
        snapshot_path = str(cfg.config_snapshot)
        cfg = OmegaConf.merge(
            OmegaConf.load(snapshot_path),
            OmegaConf.create({"config_snapshot": snapshot_path}),
        )
    if not isinstance(cfg, DictConfig):
        raise TypeError("training_progression config must be a DictConfig")

    _validate_pr_d_target_real_configs(cfg)

    logger.info(f"Mode: {cfg.mode}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.mode in ("collect", "both"):
        collect_results(cfg)
    if cfg.mode in ("plot", "both"):
        plot_results(cfg)


if __name__ == "__main__":
    main()
