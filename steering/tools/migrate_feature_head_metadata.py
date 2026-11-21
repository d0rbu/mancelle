"""Explicit, metadata-only conversion of legacy linear/residual-MLP checkpoints."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from steering.conditional.encoder_decoder.config import (
    parse_feature_prediction_config,
    serialize_feature_prediction_config,
    validate_feature_prediction_head_layers,
)


def _legacy_config(raw: object) -> dict:
    if not isinstance(raw, Mapping):
        raise ValueError("legacy feature-prediction metadata must be a mapping")
    if "block" in raw or "head_layers" in raw:
        raise ValueError("canonical, mixed, or partial head schema cannot be migrated as legacy metadata")
    converted = dict(raw)
    depth = validate_feature_prediction_head_layers(converted.pop("mlp_layers", 0))
    # The legacy canonical checkpoint reader used best_token when this field
    # was absent. Make that interpretation explicit rather than substituting
    # the fresh-construction max_pool default during conversion.
    converted.setdefault("loss_mode", "best_token")
    converted.update(block="mlp", head_layers=depth)
    config = serialize_feature_prediction_config(parse_feature_prediction_config(converted, checkpoint=True))
    assert config is not None
    return config


def _validate_tensors(weights: Path, config: dict, meta: Mapping) -> None:
    # mmap avoids loading a multi-GB model into RAM. Meta placement inspects
    # tensor metadata only; neither historical weights nor optimizer files change.
    state = t.load(weights, map_location="meta", weights_only=True, mmap=True)
    if not isinstance(state, Mapping):
        raise ValueError("paired weights must be a model state dictionary")
    head = {}
    prefix = "feature_pred_head."
    for raw_name, tensor in state.items():
        if not isinstance(raw_name, str):
            raise ValueError("model tensor keys must be strings")
        name = raw_name.removeprefix("_orig_mod.")
        if name.startswith(prefix):
            key = name.removeprefix(prefix)
            if key in head or not isinstance(tensor, t.Tensor):
                raise ValueError("duplicate or non-tensor feature-head state")
            head[key] = tensor
    projection = head.get("weight")
    if projection is None or projection.ndim != 2:
        raise ValueError("paired weights do not contain a complete feature-head projection")
    width, hidden = projection.shape
    if width < 1 or hidden < 1:
        raise ValueError("feature-head projection must have positive dimensions")
    model_config = meta.get("model_config", {})
    if isinstance(model_config, Mapping) and model_config.get("n_embd", hidden) != hidden:
        raise ValueError("head tensor width disagrees with decoder metadata")
    expected = {"weight": (width, hidden), "bias": (width,)}
    for index in range(config["head_layers"]):
        expected[f"residual_blocks.{index}.c_fc.weight"] = (4 * hidden, hidden)
        expected[f"residual_blocks.{index}.c_proj.weight"] = (hidden, 4 * hidden)
    if set(head) != set(expected):
        raise ValueError("declared head depth disagrees with the paired tensor layout")
    for key, shape in expected.items():
        if tuple(head[key].shape) != shape or not head[key].is_floating_point():
            raise ValueError(f"feature-head tensor shape/dtype disagrees for {key}: expected {shape}")


def migrate_feature_head_metadata(
    source_metadata: Path, weights: Path, output_metadata: Path
) -> dict:
    """Validate legacy architecture and atomically publish a new metadata file."""
    if output_metadata.exists() or output_metadata.is_symlink():
        raise FileExistsError(f"refusing to overwrite destination: {output_metadata}")
    with source_metadata.open() as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict) or not isinstance(meta.get("user_config", {}), dict):
        raise ValueError("checkpoint metadata and user_config must be objects")
    user = meta.get("user_config", {})
    raw_mirrors = []
    if "feature_prediction" in user:
        raw_mirrors.append(user["feature_prediction"])
    if "feature_prediction_config" in meta:
        raw_mirrors.append(meta["feature_prediction_config"])
    if not raw_mirrors or all(raw is None for raw in raw_mirrors):
        raise ValueError("headless checkpoints need no feature-head metadata conversion")
    mirrors = [_legacy_config(raw) for raw in raw_mirrors]
    if any(mirror != mirrors[0] for mirror in mirrors[1:]):
        raise ValueError("conflicting feature-prediction metadata mirrors")
    config = mirrors[0]
    _validate_tensors(weights, config, meta)
    result = copy.deepcopy(meta)
    result_user = result.setdefault("user_config", {})
    if "feature_prediction_block" in result_user or "feature_prediction_head_layers" in result_user:
        raise ValueError("mixed legacy and canonical SFT override schema")
    if "feature_prediction_mlp_layers" in result_user:
        override = validate_feature_prediction_head_layers(result_user.pop("feature_prediction_mlp_layers"))
        if override != config["head_layers"]:
            raise ValueError("recorded SFT depth disagrees with checkpoint architecture")
        result_user.update(feature_prediction_block="mlp", feature_prediction_head_layers=override)
    result_user["feature_prediction"] = copy.deepcopy(config)
    result["feature_prediction_config"] = config
    # Linking a fully fsynced temporary inode is atomic AND refuses overwrite,
    # including a concurrent destination creator (os.replace would clobber it).
    fd, temporary = tempfile.mkstemp(prefix=f".{output_metadata.name}.", dir=output_metadata.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_metadata)
    finally:
        os.unlink(temporary)
    logger.info(f"Converted feature-head metadata: {source_metadata} -> {output_metadata}")
    return result


@hydra.main(version_base=None, config_path="../../configs/utils", config_name="migrate_feature_head_metadata")
def main(cfg: DictConfig) -> None:
    migrate_feature_head_metadata(Path(cfg.source_metadata), Path(cfg.weights), Path(cfg.output_metadata))


if __name__ == "__main__":
    main()
