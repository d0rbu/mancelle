"""Hydra CLI entrypoint for the feature_filter pipeline."""
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from beartype.roar import BeartypeCallHintParamViolation
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoTokenizer

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.feature_filter.config import (
    FeatureFilterConfig,
    JudgeConfig,
    NeuronpediaInlineConfig,
    validate_config,
)
from steering.eval.feature_filter.interp_judge import (
    InterpJudgeConfig,
    make_interp_judge,
)
from steering.eval.feature_filter.neuronpedia_activations import (
    NeuronpediaActivationCache,
)
from steering.eval.feature_filter.runner import FeatureFilterRunner
from steering.eval.feature_filter.types import (
    ConcurrentRunnerError,
    ConfigValidationError,
    CostCapExceeded,
)
from steering.eval.neuronpedia_client import (
    NeuronpediaAuthError,
    NeuronpediaNotPublishedError,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

cs = ConfigStore.instance()
cs.store(name="feature_filter_schema", node=FeatureFilterConfig)


def _build_cfg(cfg_dict: DictConfig | ListConfig) -> FeatureFilterConfig:
    # OmegaConf's structured config → real dataclass
    raw = OmegaConf.to_container(cfg_dict, resolve=True)
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Expected mapping at top level of feature_filter config, got {type(raw).__name__}"
        )
    resolved: dict[str, Any] = {str(k): v for k, v in raw.items()}

    def nested_mapping(name: str) -> dict[str, Any]:
        value = resolved.pop(name, {})
        if not isinstance(value, Mapping):
            raise ConfigValidationError(
                f"feature_filter config: '{name}' must be a mapping"
            )
        return {str(key): item for key, item in value.items()}

    try:
        neuronpedia = NeuronpediaInlineConfig(**nested_mapping("neuronpedia"))
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise ConfigValidationError(f"invalid neuronpedia config: {exc}") from exc
    try:
        judge = JudgeConfig(**nested_mapping("judge"))
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise ConfigValidationError(f"invalid judge config: {exc}") from exc
    try:
        return FeatureFilterConfig(
            neuronpedia=neuronpedia,
            judge=judge,
            **resolved,
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise ConfigValidationError(f"invalid feature_filter config: {exc}") from exc


@hydra.main(
    config_path="../../../configs/eval/feature_filter",
    config_name="feature_filter",
    version_base=None,
)
def main(cfg_dict: DictConfig) -> int:
    try:
        cfg = _build_cfg(cfg_dict)
        validate_config(cfg)
    except ConfigValidationError as exc:
        logger.error(f"Config validation failed: {exc}")
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id)
        np_cfg = NeuronpediaConfig(
            timeout_seconds=cfg.neuronpedia.timeout_seconds,
            max_retries=cfg.neuronpedia.max_retries,
            backoff_base_seconds=cfg.neuronpedia.backoff_base_seconds,
            backoff_max_seconds=cfg.neuronpedia.backoff_max_seconds,
            max_concurrent=cfg.neuronpedia.max_concurrent,
        )
        cache = NeuronpediaActivationCache(
            cfg.model_id, cfg.sae_source, cfg.neuronpedia_cache_dir, config=np_cfg,
        )
        llm_cfg = LLMClientConfig(
            provider="openrouter",
            model=cfg.judge.model,
            requests_per_minute=cfg.judge.requests_per_minute,
        )
        response_cache = SynthResponseCache(Path(cfg.judge_cache_path))
        llm_client = OpenRouterLLMClient(llm_cfg, response_cache=response_cache)
        judge = make_interp_judge(llm_client, InterpJudgeConfig(
            max_prompt_chars=cfg.judge.max_prompt_chars,
            max_examples_in_prompt=cfg.judge.max_examples_in_prompt,
            example_window_chars=cfg.judge.example_window_chars,
            parse_max_retries=cfg.judge.parse_max_retries,
        ))
    except NeuronpediaAuthError as exc:
        logger.error(f"Neuronpedia auth failed: {exc}")
        return 3

    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=tokenizer)
    try:
        report = runner.run()
    except ConcurrentRunnerError as exc:
        logger.error(f"Concurrent runner conflict: {exc}")
        return 1
    except CostCapExceeded as exc:
        logger.error(f"Cost cap exceeded: {exc}")
        return 4
    except NeuronpediaNotPublishedError as exc:
        # Defense-in-depth: the runner classifies these per-feature and
        # should never let one escape; if one does, it is a known
        # limitation (un-ingested v2 SAE), not a hard failure.
        logger.warning(f"Neuronpedia not-published surfaced to CLI: {exc}")
        return 0

    total = report.n_total
    if report.n_not_published > 0:
        logger.warning(
            f"{report.n_not_published}/{total} features not published on "
            f"Neuronpedia for {cfg.sae_source}; excluded from error rate"
        )
    # not_published is deliberately NOT in `errored` — an un-ingested v2
    # SAE is a known limitation, not a failure, so it must not trip
    # error_abort_fraction.
    errored = report.n_fetch_errors + report.n_judge_errors + report.n_prompt_errors
    if total > 0 and (errored / total) > cfg.error_abort_fraction:
        logger.error(
            f"Error rate {errored}/{total} > {cfg.error_abort_fraction}; exit 1"
        )
        return 1
    logger.info(
        f"Done: total={report.n_total} accepted={report.n_accepted} "
        f"rejected={report.n_rejected} errors={errored}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
