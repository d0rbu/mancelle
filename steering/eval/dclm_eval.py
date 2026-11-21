"""DCLM CORE benchmark evaluation for decoder-only and encoder-decoder models.

Provides multi-config evaluation: run CORE with different feature-conditioning
setups and collect all results into a single YAML report.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import hydra
import torch as t
import yaml
from loguru import logger
from omegaconf import DictConfig

from nanochat.checkpoint_manager import ModelPhase, load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.gpt import GPT
from scripts.eval.base_eval import _run_core_target_real_eval, evaluate_core
from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.feature_config import (
    FeatureConfig,
    ensure_core_eval_has_work,
    parse_feature_configs,
    require_nonempty_feature_configs,
    resolve_next_segment_conditioning_offset,
    validate_inactive_bucket_feature_configs,
)
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.target_real_runner import (
    assert_target_real_requires_encdec,
    build_shared_context_feature_provider,
    build_target_real_payload,
    parse_target_real_configs_from_cfg,
    require_shared_context_cache_dir,
    run_target_real_evals,
)


@dataclass
class ConfigResult:
    """Results from evaluating a single feature configuration."""

    name: str
    results: dict[str, float]
    centered_results: dict[str, float]
    core_metric: float


@dataclass
class EvalReport:
    """Complete evaluation report across one or more feature configurations."""

    model: str | dict
    evaluation: str
    config_results: list[ConfigResult]

    @property
    def core_scores(self) -> dict[str, float]:
        return {cr.name: cr.core_metric for cr in self.config_results}

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "evaluation": self.evaluation,
            "results": [asdict(cr) for cr in self.config_results],
            "summary": {
                "configs_evaluated": len(self.config_results),
                "core_scores": self.core_scores,
            },
        }


def evaluate_dclm_core(
    model: GPT | EncoderDecoderGPT,
    tokenizer,
    device,
    *,
    feature_conditions: dict | None = None,
    max_per_task: int = -1,
    core_example_batch_size: int = 1,
    context_provider: ContextProvider | None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> dict:
    """Evaluate DCLM CORE, requiring full context conditioning for encdec models.

    Args:
        model: A ``GPT`` or ``EncoderDecoderGPT`` instance.
        tokenizer: Tokenizer compatible with ``evaluate_core``.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        feature_conditions: Optional feature mapping for conditioning.
        max_per_task: Cap on examples per task (``-1`` = all).
        core_example_batch_size: Target-aware encoder-decoder examples per forward.
        context_provider: Shared context provider for prompt/context features.

    Returns:
        Dict with ``results``, ``centered_results``, and ``core_metric``.

    Raises:
        ValueError: If a decoder-only model is given with non-null features,
            or if full-context cache configuration is missing.
    """
    is_encdec = isinstance(model, EncoderDecoderGPT)
    has_features = feature_conditions is not None and len(feature_conditions) > 0

    if not is_encdec and has_features:
        raise ValueError(
            "Decoder-only models do not support feature conditioning. "
            "Pass feature_conditions=None or use an EncoderDecoderGPT."
        )

    if is_encdec:
        if context_provider is None:
            raise ValueError(
                "Encoder-decoder dclm_eval requires a context_provider with "
                "real prompt/context features."
            )
        eval_model = build_wrapper_for_features(
            model,
            tokenizer,
            features=feature_conditions,
            context_provider=context_provider,
            target_mode=target_mode,
        )
    else:
        eval_model = model

    return evaluate_core(
        eval_model,
        tokenizer,
        device,
        max_per_task=max_per_task,
        core_example_batch_size=core_example_batch_size,
    )


def run_evaluation(
    model: GPT | EncoderDecoderGPT,
    tokenizer: object,
    device: str,
    configs: list[FeatureConfig] | None = None,
    max_per_task: int = -1,
    core_example_batch_size: int = 1,
    *,
    context_provider: ContextProvider | None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> EvalReport:
    """Evaluate a model under one or more feature-conditioning configs.

    Args:
        model: A ``GPT`` or ``EncoderDecoderGPT`` instance.
        tokenizer: Tokenizer compatible with ``evaluate_core``.
        device: Torch device string.
        configs: List of :class:`FeatureConfig` to evaluate. ``None`` means no
            static target-delta configs.
        max_per_task: Cap on examples per task (``-1`` = all).
        core_example_batch_size: Target-aware encoder-decoder examples per forward.
        context_provider: Shared SafetensorsContextFeatureProvider for encdec eval.
    """
    if configs is None:
        configs = []
    if configs and isinstance(model, EncoderDecoderGPT):
        require_nonempty_feature_configs(configs, surface="dclm_eval")
        if context_provider is None:
            raise ValueError(
                "Encoder-decoder dclm_eval requires conditioning.context_cache_dir; "
                "target-only eval is intentionally unsupported."
            )

    config_results: list[ConfigResult] = []

    for cfg_fc in configs:
        logger.info(f"Evaluating config: {cfg_fc.name}")
        core_out = evaluate_dclm_core(
            model,
            tokenizer,
            device,
            feature_conditions=cfg_fc.features,
            max_per_task=max_per_task,
            core_example_batch_size=core_example_batch_size,
            context_provider=context_provider,
            target_mode=target_mode,
        )
        config_results.append(
            ConfigResult(
                name=cfg_fc.name,
                results=core_out["results"],
                centered_results=core_out["centered_results"],
                core_metric=core_out["core_metric"],
            )
        )

    model_type = "encoder_decoder" if isinstance(model, EncoderDecoderGPT) else "decoder_only"
    return EvalReport(
        model=model_type,
        evaluation="dclm-core",
        config_results=config_results,
    )


def write_results_yaml(report: EvalReport, output_path: str) -> None:
    """Write evaluation results to a YAML file.

    Args:
        report: The :class:`EvalReport` to serialize.
        output_path: Filesystem path for the output YAML file.
    """
    data = report.to_dict()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Results written to {output_path}")


def _dclm_core_target_real_per_config(
    *,
    raw_encdec,
    tokenizer,
    device,
    trc,
    provider,
    cfg,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> dict:
    """Build the per-target-real-config payload for dclm CORE eval.

    Delegates to :func:`base_eval._run_core_target_real_eval` (same CORE
    pipeline). Returns the payload dict (orchestrator writes the yaml).
    """
    core_out = _run_core_target_real_eval(
        raw_encdec=raw_encdec,
        tokenizer=tokenizer,
        device=device,
        trc=trc,
        provider=provider,
        max_per_task=int(cfg.max_per_task),
        target_mode=target_mode,
    )
    return build_target_real_payload(
        trc,
        {
            "results": core_out["results"],
            "centered_results": core_out["centered_results"],
            "core_metric": core_out["core_metric"],
        },
        target_mode=target_mode,
    )


@hydra.main(config_path="../../configs/eval", config_name="dclm_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    """DCLM CORE evaluation entry point with DDP support.

    Loads a decoder-only or encoder-decoder model, runs DCLM CORE evaluation
    across the specified feature configurations, and writes a combined YAML
    report.
    """
    device_type = autodetect_device_type() if cfg.device_type == "" else cfg.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        t.amp.autocast(device_type=device_type, dtype=t.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Validate config before model load (fail fast; mirrors base_eval ordering)
    configs = parse_feature_configs(cfg.feature_configs)
    # Suite name picks the cache subdirectory for null-cache_dir auto-resolution
    # (Task 2). dclm_eval consumes the same target cache layout as base_eval CORE.
    target_real_configs = parse_target_real_configs_from_cfg(cfg, suite="core_target")

    # Load model via chain-based source resolution
    source = cfg.source
    model, tokenizer, meta = load_model(
        source,
        device,
        phase=ModelPhase.EVAL,
        model_tag=cfg.model_tag,
        step=cfg.step,
        eval_conditioning=EvalConditioning.deferred("dclm_eval"),
    )
    reject_feature_clusters_metadata(meta, "dclm_eval checkpoint")
    is_encdec = isinstance(model, EncoderDecoderGPT)
    if is_encdec:
        require_fixed_checkpoint_segment_mode(meta, surface="dclm_eval")
    target_mode = TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(cfg, meta)
    )
    assert_target_real_requires_encdec(target_real_configs, is_encdec=is_encdec)
    # Empty feature_configs must not silently run zero passes: decoder-only models
    # fall back to one unconditioned baseline; encoder-decoder runs require either
    # static configs or target_real work (otherwise hard error).
    configs = ensure_core_eval_has_work(
        configs,
        is_encdec=is_encdec,
        has_target_real=bool(target_real_configs),
        surface="dclm_eval",
    )
    model_info = {
        "type": "encoder_decoder" if is_encdec else "decoder_only",
        "source": source,
        "checkpoint_dir": meta.get("checkpoint_dir"),
        "step": int(meta["step"]),
    }
    logger.info(f"Model loaded: {model_info}")

    if is_encdec and configs:
        require_nonempty_feature_configs(configs, surface="dclm_eval")
        require_shared_context_cache_dir(cfg, eval_name="dclm_eval")
        validate_inactive_bucket_feature_configs(
            configs,
            num_buckets=model.encoder.config.num_buckets,
            checkpoint_meta=meta,
        )

    shared_context_provider = (
        build_shared_context_feature_provider(cfg, eval_name="dclm_eval")
        if is_encdec and configs
        else None
    )
    with autocast_ctx:
        report = run_evaluation(
            model,
            tokenizer,
            device,
            configs,
            max_per_task=cfg.max_per_task,
            core_example_batch_size=cfg.core_example_batch_size,
            context_provider=shared_context_provider,
            target_mode=target_mode,
        )

    # Enrich report with detailed model info
    report.model = model_info

    # Write per-config YAML output (rank 0 only)
    if ddp_rank == 0:
        default_output = os.path.join(meta["checkpoint_dir"], "dclm_eval")
        output_dir = cfg.output_dir or default_output
        for cr in report.config_results:
            output_path = os.path.join(output_dir, f"{cr.name}.yaml")
            config_report = EvalReport(
                model=model_info,
                evaluation="dclm-core",
                config_results=[cr],
            )
            write_results_yaml(config_report, output_path)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("DCLM CORE Evaluation Summary")
        logger.info("=" * 60)
        for name, score in report.core_scores.items():
            logger.info(f"  {name}: {score:.4f}")

    # PR-C: per-target-real-config eval loop. Orchestrated by the shared
    # ``run_target_real_evals`` helper — rank-0 barrier, per-config provider
    # construction, output dir creation, and yaml writing all live there.
    if target_real_configs and ddp_rank == 0:
        default_output = os.path.join(meta["checkpoint_dir"], "dclm_eval")
        output_dir_base = cfg.output_dir or default_output
        raw_encdec = model if is_encdec else None
        with autocast_ctx:
            run_target_real_evals(
                target_real_configs=target_real_configs,
                cfg=cfg,
                output_dir_base=output_dir_base,
                ddp_rank=ddp_rank,
                per_config_eval_fn=lambda trc, provider: _dclm_core_target_real_per_config(
                    raw_encdec=raw_encdec,
                    tokenizer=tokenizer,
                    device=device,
                    trc=trc,
                    provider=provider,
                    cfg=cfg,
                    target_mode=target_mode,
                ),
                checkpoint_dir=meta["checkpoint_dir"],
                report_section_prefix=f"DCLM eval {cfg.source}",
            )

    compute_cleanup()


if __name__ == "__main__":
    main()
