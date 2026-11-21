"""Encoder-decoder SFT pipeline: configurable phase order.

Orchestrates a chained SFT sequence on an encoder-decoder model. Each phase
finetunes from the previous phase's checkpoint (the first phase from
pretrain). For each conditioned phase, feature statistics are computed
(idempotently) before kicking off the SFT job.

The default phase_order is ``safety,capabilities`` (the original pipeline).
Pass ``phase_order=capabilities,safety`` to flip the order. The pipeline
supports every ``SFTMixture`` phase (``CONDITIONED_FEATURE_DATASETS`` is
enum-derived): ``standard``, ``capabilities``, ``safety``, ``standard_steered``
and ``standard_steered_balanced``. A steered phase needs its
``steered_jsonl=<path>`` override, which reaches the SFT subprocess through the
forwarded Hydra overrides.

Usage:
    # Single GPU, default order (safety → capabilities)
    uv run python -m scripts.train.run_encdec_sft_pipeline model_tag=d12_baseline

    # Reverse order (capabilities → safety)
    uv run python -m scripts.train.run_encdec_sft_pipeline model_tag=d12_baseline phase_order=capabilities,safety

    # Multi-GPU (each SFT phase runs via torchrun)
    uv run python -m scripts.train.run_encdec_sft_pipeline model_tag=d12_baseline num_gpus=4

    # With hardware config overrides
    uv run python -m scripts.train.run_encdec_sft_pipeline model_tag=d12_baseline num_gpus=2 device_batch_size=8 max_seq_len=1024

    # With a hardware config file
    uv run python -m scripts.train.run_encdec_sft_pipeline config_name=conditional_encdec_sft_2xh200 model_tag=d12_baseline

    # Use turn-delimited SFT feature directories
    uv run python -m scripts.train.run_encdec_sft_pipeline model_tag=d12_baseline feature_segment_mode=turn_delimited_fixed_tokens
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from beartype import beartype
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf

from errors import FeatureClustersError
from nanochat.common import get_base_dir
from nanochat.hydra_cli import parse_config_name_and_overrides
from steering.conditional.feature_mask import feature_mask_leaf_from_cfg
from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.sampling import assert_no_gradient_ranking, parse_feature_selection
from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.refined_types import PositiveInt, PositiveIntLike
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import find_statistics_path
from tasks.mixtures import SFTMixture

DEFAULT_SFT_CONFIG_NAME = "conditional_encdec_sft"
DEFAULT_PHASE_ORDER: tuple[str, ...] = ("safety", "capabilities")
CONDITIONED_FEATURE_DATASETS: frozenset[str] = frozenset(m.value for m in SFTMixture)
_PIPELINE_PARAMS = {
    "feature_root",
    "pooling",
    "num_gpus",
    "phase_order",
    "feature_segment_mode",
}


# ---------------------------------------------------------------------------
# CLI / validation helpers
# ---------------------------------------------------------------------------


def parse_phase_order(raw: str | Sequence[str] | None) -> list[str]:
    """Parse a phase-order value from CLI or direct Python calls.

    Accepted forms:
    - ``None`` (returns the default ``("safety", "capabilities")``)
    - Comma-separated string: ``"safety,capabilities"`` or ``" safety , capabilities "``
    - Already-parsed sequence: ``["safety", "capabilities"]``

    Hydra-bracket syntax (``"[safety,capabilities]"``, ``'["safety","capabilities"]'``,
    or any sequence element containing ``[`` / ``]``) is rejected with a
    clear error directing the operator at the comma form. Brackets in input
    are a common source of subtle whitespace/quoting bugs and don't add
    expressiveness over the comma form, so we fail fast — for both string
    CLI input and Python-API list input.
    """
    if raw is None:
        return list(DEFAULT_PHASE_ORDER)
    if isinstance(raw, str):
        if "[" in raw or "]" in raw:
            raise ValueError(
                f"phase_order uses comma-separated form, not Hydra brackets; got "
                f"{raw!r}. Use 'phase_order=safety,capabilities' instead."
            )
        raw_parts: Sequence[str] = raw.split(",")
    else:
        raw_parts = raw

    parts = [str(part).strip().strip("\"'") for part in raw_parts if str(part).strip().strip("\"'")]
    for part in parts:
        if "[" in part or "]" in part:
            raise ValueError(
                f"phase_order uses comma-separated form, not Hydra brackets; got "
                f"entry {part!r}. Use 'phase_order=safety,capabilities' instead."
            )
    return parts


def validate_phase_order(phase_order: Sequence[str]) -> list[str]:
    """Validate and normalize the requested SFT phase order.

    Conditioned SFT requires phase-specific feature directories before
    launch, so only datasets with extraction phases are accepted.
    """
    phases = [str(phase).strip() for phase in phase_order if str(phase).strip()]
    if not phases:
        raise ValueError("phase_order must list at least one dataset")

    seen: set[str] = set()
    duplicates: list[str] = []
    for phase in phases:
        if phase in seen and phase not in duplicates:
            duplicates.append(phase)
        seen.add(phase)
    if duplicates:
        raise ValueError(
            f"phase_order must not contain duplicate datasets; duplicates: {duplicates}"
        )

    unsupported = [phase for phase in phases if phase not in CONDITIONED_FEATURE_DATASETS]
    if unsupported:
        valid = sorted(CONDITIONED_FEATURE_DATASETS)
        raise ValueError(
            "phase_order datasets must have pre-extracted SFT feature phases; "
            f"got {unsupported}. Supported datasets: {valid}"
        )

    return phases


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def build_checkpoint_dir(
    base_dir: str,
    model_tag: str,
    segment_size: int,
    feature_selection_dir: str,
) -> str:
    """Construct the pretrain checkpoint directory path."""
    return os.path.join(
        base_dir,
        "conditional_encdec_checkpoints",
        model_tag,
        f"s{segment_size}",
        feature_selection_dir,
    )


def build_sft_checkpoint_dir(
    checkpoint_dir: str,
    dataset: str,
) -> str:
    """Construct the SFT checkpoint leaf directory."""
    return os.path.join(checkpoint_dir, f"{dataset}_sft_conditioned_checkpoints")


def build_feature_dir_for_phase(
    feature_root: str,
    segment_size: int,
    layer: int,
    activation_type: str,
    pooling: str,
    sae_width: str,
    model: str,
    dataset_phase: str,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> str:
    """Construct the feature directory path for a dataset phase."""
    return str(
        build_feature_dir(
            root=feature_root,
            segment_size=segment_size,
            layer=layer,
            activation_type=ActivationType(activation_type),
            pooling=PoolingMode(pooling),
            width=SAEWidth(sae_width),
            model=model,
            dataset_phase=DatasetPhase(dataset_phase),
            ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
            segment_mode=segment_mode,
        )
    )


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def ensure_feature_statistics(feature_dir: str, dataset_phase: str) -> None:
    """Compute feature statistics if not already present."""
    stats_path = find_statistics_path(feature_dir)
    if stats_path is not None:
        logger.info(f"Feature statistics already exist: {stats_path}")
        return

    logger.info(f"Computing feature statistics for {dataset_phase} at {feature_dir}")
    cmd = [
        sys.executable,
        "-m",
        "steering.compute_statistics",
        f"feature_dir={feature_dir}",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"compute_statistics failed for {dataset_phase} (exit code {result.returncode})"
        )
    # Verify stats were created
    stats_path = find_statistics_path(feature_dir)
    assert stats_path is not None, (
        f"compute_statistics completed but no feature_statistics.safetensors found in {feature_dir}"
    )
    logger.info(f"Feature statistics saved: {stats_path}")


@beartype
def run_sft(
    dataset: str,
    model_tag: str,
    sft_overrides: list[str],
    base_checkpoint_dir: str | None = None,
    feature_dir: str | None = None,
    num_gpus: PositiveIntLike = 1,
    sft_config_name: str = DEFAULT_SFT_CONFIG_NAME,
) -> int:
    """Run a single SFT phase via subprocess.

    When ``num_gpus > 1``, launches via ``torchrun --standalone``.
    ``@beartype`` owns the positive GPU-count contract.

    Returns the subprocess exit code (0 = success).
    """
    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "scripts.train.conditional_encdec_sft",
            "--",
        ]
    else:
        cmd = [sys.executable, "-m", "scripts.train.conditional_encdec_sft"]
    if sft_config_name != DEFAULT_SFT_CONFIG_NAME:
        cmd.append(f"config_name={sft_config_name}")
    cmd.extend(
        [
            f"dataset={dataset}",
            f"model_tag={model_tag}",
        ]
    )
    if base_checkpoint_dir is not None:
        cmd.append(f"base_checkpoint_dir={base_checkpoint_dir}")
    if feature_dir is not None:
        cmd.append(f"feature_dir={feature_dir}")
    cmd.extend(sft_overrides)

    logger.info(f"Running {dataset} SFT: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{dataset} SFT failed (exit code {result.returncode})")
    return result.returncode


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@beartype
def run_pipeline(
    model_tag: str,
    segment_size: int = 128,
    feature_root: str = "out/data/features",
    layer: int = 8,
    activation_type: str = "residual",
    pooling: str = "max_pool",
    sae_width: str = "16k",
    sae_model: str = "gemma-2-2b",
    sft_overrides: list[str] | None = None,
    num_gpus: PositiveIntLike = 1,
    phase_order: str | Sequence[str] | None = None,
    sft_config_name: str = DEFAULT_SFT_CONFIG_NAME,
    feature_segment_mode: str = "fixed_tokens",
) -> dict[str, str]:
    """Run the chained SFT pipeline in ``phase_order`` (default safety → capabilities).

    When ``num_gpus > 1``, each SFT phase is launched via torchrun.
    ``@beartype`` owns the positive GPU-count contract.

    Returns a dict with checkpoint paths for each completed phase.
    """
    if sft_overrides is None:
        sft_overrides = []
    segment_mode = SegmentMode(feature_segment_mode)
    phase_order = validate_phase_order(parse_phase_order(phase_order))
    base_dir = get_base_dir()

    # Parse feature_selection from the SFT config to build the path
    config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=sft_config_name, overrides=sft_overrides)
    if cfg.get("feature_clusters") is not None:
        raise FeatureClustersError(
            "clustered chained SFT is not supported by the pipeline driver; run "
            "phases individually with explicit base_checkpoint_dir"
        )
    feature_selection_cfg = OmegaConf.to_container(
        cfg.feature_selection,
        resolve=True,
    )
    if not isinstance(feature_selection_cfg, dict):
        raise TypeError(
            "training config feature_selection must resolve to a mapping, "
            f"got {type(feature_selection_cfg).__name__}"
        )
    feature_selection = parse_feature_selection(cast(dict[str, Any], feature_selection_cfg))
    assert_no_gradient_ranking(
        feature_selection, "run_encdec_sft_pipeline (SFT is live-selection only)"
    )
    fs_dirname = feature_selection_dirname(feature_selection)

    # Build pretrain checkpoint dir (with optional feature_mask leaf).
    # feature_mask_leaf_from_cfg computes the dirname from the raw
    # cfg block without needing n_sae_features (the orchestrator has no
    # checkpoint to load from); the hash depends only on mode+ids.
    # When no mask is configured the leaf is "" and os.path.join is skipped
    # (I2: path is byte-identical to the legacy no-mask path).
    _mask_leaf = feature_mask_leaf_from_cfg(cfg)
    _base_pretrain_dir = build_checkpoint_dir(base_dir, model_tag, segment_size, fs_dirname)
    pretrain_dir = (
        os.path.join(_base_pretrain_dir, _mask_leaf) if _mask_leaf else _base_pretrain_dir
    )
    logger.info(f"Pretrain checkpoint dir: {pretrain_dir}")
    logger.info(f"Phase order: {' → '.join(phase_order)}")

    results: dict[str, str] = {}
    parent_dir: str | None = None  # None for first phase = chain from pretrain

    for idx, dataset in enumerate(phase_order, start=1):
        feature_dir = build_feature_dir_for_phase(
            feature_root,
            segment_size,
            layer,
            activation_type,
            pooling,
            sae_width,
            sae_model,
            f"{dataset}_sft",
            segment_mode=segment_mode,
        )
        ensure_feature_statistics(feature_dir, f"{dataset}_sft")
        verify_feature_dir(feature_dir)

        chained_from = "pretrain" if parent_dir is None else phase_order[idx - 2]
        logger.info("=" * 60)
        logger.info(f"Phase {idx}: {dataset} SFT (chaining from {chained_from})")
        logger.info("=" * 60)
        run_sft(
            dataset=dataset,
            model_tag=model_tag,
            sft_overrides=sft_overrides,
            base_checkpoint_dir=parent_dir,
            feature_dir=feature_dir,
            num_gpus=num_gpus,
            sft_config_name=sft_config_name,
        )
        sft_dir = build_sft_checkpoint_dir(
            parent_dir if parent_dir is not None else pretrain_dir,
            dataset,
        )
        results[f"{dataset}_sft"] = sft_dir
        logger.info(f"{dataset} SFT complete: {sft_dir}")
        parent_dir = sft_dir

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    for phase, path in results.items():
        logger.info(f"  {phase}: {path}")
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    # Pipeline-only params are NOT in the SFT Hydra config. Extract them
    # before passing overrides to Hydra compose.
    sft_config_name, all_overrides = parse_config_name_and_overrides(
        sys.argv[1:],
        default_config_name=DEFAULT_SFT_CONFIG_NAME,
    )
    pipeline_params: dict[str, str] = {}
    hydra_overrides: list[str] = []
    for ov in all_overrides:
        key = ov.split("=", 1)[0]
        if key in _PIPELINE_PARAMS:
            pipeline_params[key] = ov.split("=", 1)[1]
        else:
            hydra_overrides.append(ov)

    config_dir = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=sft_config_name, overrides=hydra_overrides)

    # Extract pipeline-level params
    model_tag = cfg.model_tag
    feature_root = pipeline_params.get("feature_root", "out/data/features")
    pooling = pipeline_params.get("pooling", "max_pool")
    num_gpus = PositiveInt.parse(int(pipeline_params.get("num_gpus", "1")))
    phase_order = parse_phase_order(pipeline_params.get("phase_order"))
    feature_segment_mode = pipeline_params.get("feature_segment_mode", "fixed_tokens")

    # Overrides that pass through to the SFT subprocess
    sft_overrides = [a for a in hydra_overrides if not a.startswith("model_tag=")]

    run_pipeline(
        model_tag=model_tag,
        segment_size=cfg.segment_size,
        feature_root=feature_root,
        layer=cfg.sae.layer,
        activation_type=cfg.sae.activation_type,
        sae_width=cfg.sae.width,
        sae_model=cfg.sae.model,
        pooling=pooling,
        sft_overrides=sft_overrides,
        num_gpus=num_gpus,
        phase_order=phase_order,
        sft_config_name=sft_config_name,
        feature_segment_mode=feature_segment_mode,
    )


if __name__ == "__main__":
    main()
