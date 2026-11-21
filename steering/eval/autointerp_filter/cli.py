"""Hydra-facing helpers for the autointerp filter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from errors import RemovedFeatureError
from steering.conditional.feature_clusters import reject_feature_clusters_config
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    IntruderJudgeConfig,
    NeuronpediaDescriptionConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
    validate_config,
)
from steering.eval.autointerp_filter.intruder import IntruderDetectionJudge
from steering.eval.autointerp_filter.runner import AutoInterpFilterRunner
from steering.eval.autointerp_filter.types import AutoInterpConfigError
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient


def build_config(cfg_dict: DictConfig) -> AutoInterpFilterConfig:
    """Convert OmegaConf into frozen dataclasses."""
    raw = OmegaConf.to_container(cfg_dict, resolve=True)
    if not isinstance(raw, dict):
        raise AutoInterpConfigError(f"Expected mapping config, got {type(raw).__name__}")
    resolved: dict[str, Any] = {str(k): v for k, v in raw.items()}
    reject_feature_clusters_config(resolved, "autointerp_filter config")
    if resolved.get("feature_dir") is None:
        # Preserve the required-path sentinel while the base config is being
        # inspected; validate_config rejects it before pipeline work begins.
        resolved["feature_dir"] = ""
    for key, cls in (
        ("judge", IntruderJudgeConfig),
        ("token_examples", TokenExamplesConfig),
        ("source_index", SourceIndexConfig),
        ("neuronpedia", NeuronpediaDescriptionConfig),
    ):
        value = resolved.get(key, {})
        if not isinstance(value, dict):
            raise AutoInterpConfigError(f"{key} must be a mapping")
        try:
            resolved[key] = cls(**cast(dict[str, Any], value))
        except (BeartypeCallHintParamViolation, TypeError) as exc:
            raise AutoInterpConfigError(f"{key} has invalid fields: {exc}") from exc
    try:
        cfg = AutoInterpFilterConfig(**resolved)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AutoInterpConfigError(f"autointerp config has invalid fields: {exc}") from exc
    path_updates: dict[str, str] = {}
    for field_name in (
        "token_example_cache_path",
        "judge_cache_path",
        "target_scores_path",
        "ranked_yaml_path",
        "accepted_features_path",
        "manual_feature_filter_path",
        "manual_feature_filter_meta_path",
        "report_output_path",
        "trial_audit_path",
        "trial_audit_yaml_path",
        "runner_manifest_path",
        "lockfile_path",
    ):
        value = str(getattr(cfg, field_name))
        if "${output_dir}" in value:
            path_updates[field_name] = value.replace("${output_dir}", cfg.output_dir)
    if path_updates:
        cfg = replace(cfg, **path_updates)
    source_updates: dict[str, str] = {}
    for field_name in ("path", "manifest_path", "shared_cache_dir"):
        raw_value = getattr(cfg.source_index, field_name)
        if raw_value is None:
            continue
        value = str(raw_value)
        if "${output_dir}" in value:
            source_updates[field_name] = value.replace("${output_dir}", cfg.output_dir)
    if source_updates:
        cfg = replace(cfg, source_index=replace(cfg.source_index, **source_updates))
    return cfg


def run_from_config(cfg_dict: DictConfig) -> int:
    """Build dependencies and run. Returns process exit code."""
    try:
        cfg = build_config(cfg_dict)
        validate_config(cfg)
        llm_cfg = LLMClientConfig(
            provider="openrouter",
            model=cfg.judge.model,
            requests_per_minute=cfg.judge.requests_per_minute,
        )
        client = OpenRouterLLMClient(llm_cfg)
        judge = IntruderDetectionJudge(client, cfg.judge)
        report = AutoInterpFilterRunner(
            cfg,
            judge=judge,
        ).run()
    except (AutoInterpConfigError, RemovedFeatureError) as exc:
        logger.error(f"Autointerp config validation failed: {exc}")
        return 2
    except Exception as exc:
        logger.error(f"Autointerp filter failed: {exc}")
        return 1

    logger.info(
        f"Autointerp filter complete: targets={report.n_targets_total} "
        f"accepted_targets={report.n_targets_accepted} "
        f"accepted_features={report.n_features_accepted}"
    )
    if cfg.fail_on_empty and report.n_features_accepted == 0:
        return 1
    return 0
