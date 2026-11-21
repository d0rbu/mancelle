"""Steering evaluation CLI — run SAE feature steering eval on encoder-decoder models.

Usage:
    uv run python scripts/steering_eval.py source=encdec steering_config=path/to/config.yaml
    uv run python scripts/steering_eval.py source=encdec steering_config=config.yaml output_dir=/tmp/out
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch as t
import yaml
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

from nanochat.checkpoint_manager import (
    ModelPhase,
    _is_encdec_root,
    build_encdec_model,
    find_last_step,
    get_checkpoint_dir,
    load_model,
)
from nanochat.common import autodetect_device_type
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.eval.checkpoint_selection import parse_checkpoint_feature_selection
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.harmfulness_judge import JudgeClient
from steering.eval.segment_mode_guard import require_fixed_checkpoint_segment_mode
from steering.eval.steering_eval import (
    PairwiseJudge,
    load_config,
    run_steering_eval,
    validate_config_against_model,
)
from steering.eval.target_conditioning import TargetConditioningMode
from steering.eval.target_real_runner import build_context_feature_provider

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "steering_eval"
_RESULTS_SUBDIR = "steering_eval"


def _load_model_for_steering_eval(cfg, device: t.device):
    """Load the configured model and pin explicit encdec feature selections.

    ``load_model`` can auto-discover a bare encdec checkpoint, but it has no
    feature-selection arguments.  Resolve and build an explicitly selected
    checkpoint directly so the requested leaf cannot be ignored.
    """
    feature_selection = parse_checkpoint_feature_selection(
        cfg.get("feature_selection"),
        live_context="steering_eval live selection",
    )
    segment_size = cfg.get("segment_size")
    is_encdec_root = _is_encdec_root(cfg.source)

    if not is_encdec_root and (feature_selection is not None or segment_size is not None):
        raise ValueError(
            "steering_eval segment_size/feature_selection checkpoint pins require "
            "an encdec-rooted source"
        )

    if is_encdec_root and feature_selection is not None:
        if segment_size is None:
            raise ValueError(
                "steering_eval requires segment_size when feature_selection is set"
            )
        checkpoint_dir = get_checkpoint_dir(
            cfg.source,
            model_tag=cfg.model_tag,
            segment_size=segment_size,
            feature_selection=feature_selection,
        )
        step = cfg.step if cfg.step is not None else find_last_step(checkpoint_dir)
        model, tokenizer, meta = build_encdec_model(
            checkpoint_dir,
            step,
            device,
            ModelPhase.EVAL,
            eval_conditioning=EvalConditioning.deferred("steering_eval"),
        )
        meta["checkpoint_dir"] = checkpoint_dir
        return model, tokenizer, meta

    if is_encdec_root and segment_size is not None:
        raise ValueError(
            "steering_eval.segment_size without feature_selection is ambiguous; "
            "set both to pin a checkpoint or leave both null for auto-discovery"
        )

    return load_model(
        source=cfg.source,
        device=device,
        phase=ModelPhase.EVAL,
        model_tag=cfg.model_tag,
        step=cfg.step,
        eval_conditioning=EvalConditioning.deferred("steering_eval"),
    )


def _resolve_output_dir(
    output_dir: str | None,
    checkpoint_dir: str | None,
) -> Path:
    """Resolve the output directory for steering eval results.

    When *output_dir* is explicitly set, use it directly.  Otherwise
    auto-derive from the directory of the checkpoint that was actually loaded.

    Parameters
    ----------
    checkpoint_dir:
        Concrete checkpoint leaf recorded by the model loader. Using this
        rather than independently resolving config keeps loading and report
        placement on the same checkpoint.

    Raises
    ------
    ValueError
        If neither *output_dir* nor a concrete *checkpoint_dir* is provided.
    """
    if output_dir is not None:
        return Path(output_dir)
    if checkpoint_dir is None:
        raise ValueError(
            "Cannot determine output directory: set output_dir or load a checkpoint "
            "with a concrete checkpoint_dir"
        )
    return Path(checkpoint_dir) / "report" / _RESULTS_SUBDIR


def main() -> None:
    """Run the steering evaluation pipeline."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name=_CONFIG_NAME, overrides=overrides)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Validate required fields.
    if not cfg.steering_config:
        logger.error(
            "steering_config is required. Pass steering_config=path/to/config.yaml"
        )
        sys.exit(1)

    steering_config_path = Path(cfg.steering_config)
    if not steering_config_path.exists():
        logger.error(f"Steering config not found: {steering_config_path}")
        sys.exit(1)

    # Verify API key.
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Load model.
    device_type = autodetect_device_type()
    device = t.device(device_type)
    logger.info(
        f"Loading model: source={cfg.source}, tag={cfg.model_tag}, step={cfg.step}"
    )
    model, tokenizer, meta = _load_model_for_steering_eval(cfg, device)
    reject_feature_clusters_metadata(meta, "steering_eval checkpoint")

    if not isinstance(model, EncoderDecoderGPT):
        logger.error(
            f"Expected EncoderDecoderGPT but got {type(model).__name__}. "
            "Steering eval requires an encoder-decoder model (source=encdec)."
        )
        sys.exit(1)
    require_fixed_checkpoint_segment_mode(meta, surface="steering_eval")

    # Load steering eval config.
    steering_config = load_config(steering_config_path)
    logger.info(
        f"Steering config: {len(steering_config.features)} features, "
        f"magnitude_levels={steering_config.magnitude_levels}, "
        f"{steering_config.generation.samples_per_condition} samples/condition"
    )

    # Validate config against model.
    enc_config = model.encoder.config
    validate_config_against_model(
        steering_config,
        n_sae_features=enc_config.n_sae_features,
        num_buckets=enc_config.num_buckets,
    )

    cond = cfg.get("conditioning", None)
    cache_dir = None if cond is None else cond.get("context_cache_dir", None)
    if cache_dir is None:
        raise ValueError(
            "Encoder-decoder steering eval requires conditioning.context_cache_dir "
            "so chat prompts receive real cached SAE features."
        )
    OmegaConf.update(
        cfg,
        "model_max_seq_len",
        int(model.dec_config.sequence_len),
        merge=False,
        force_add=True,
    )
    context_provider = build_context_feature_provider(
        cfg,
        cache_dir=cache_dir,
        error_prefix="conditioning.context_cache_dir",
    )

    # Create judge.
    judge_client = JudgeClient(
        model=steering_config.judge.model,
        requests_per_minute=cfg.requests_per_minute,
        cache_path=cfg.judge_cache_path or None,
    )
    judge = PairwiseJudge(judge_client)

    # Resolve output directory.
    output_dir = _resolve_output_dir(
        output_dir=cfg.output_dir,
        checkpoint_dir=meta.get("checkpoint_dir"),
    )

    logger.info(f"Output directory: {output_dir}")

    # Run eval.
    autocast_ctx = (
        t.amp.autocast(device_type=device_type, dtype=t.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Resolve next-segment target timing from checkpoint metadata so a
    # checkpoint trained with next_segment_conditioning_offset=true is evaluated
    # with NEXT routing instead of silently defaulting to CURRENT.
    target_mode = TargetConditioningMode.from_next_segment_offset(
        resolve_next_segment_conditioning_offset(cfg, meta)
    )

    with autocast_ctx:
        run_steering_eval(
            steering_config,
            judge,
            output_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_provider=context_provider,
            target_mode=target_mode,
        )

    # Print summary.
    report_path = output_dir / "report.yaml"
    if report_path.exists():
        with open(report_path) as f:
            report = yaml.safe_load(f)

        logger.info("=== Steering Eval Results ===")
        if "features" in report:
            for feat in report["features"]:
                wr = feat.get("standard_win_rate", "N/A")
                fid = feat["id"]
                logger.info(f"  Feature {fid}: win_rate={wr}")
                dose = feat.get("dose_response", {})
                if dose:
                    rho = dose.get("spearman_rho", "N/A")
                    r2 = dose.get("isotonic_r2", "N/A")
                    logger.info(f"    spearman_rho={rho}, isotonic_r2={r2}")

        if "aggregate" in report:
            agg = report["aggregate"]
            logger.info(
                f"  Aggregate win rate: {agg.get('confidence_weighted_win_rate', 'N/A')}"
            )
            logger.info(f"  Fraction monotonic: {agg.get('fraction_monotonic', 'N/A')}")


if __name__ == "__main__":
    main()
