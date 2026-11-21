"""CPU-only Campaign 2 artifact checks and isolated encoder stage bases."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.immutable_io import (
    StreamedFileDigest,
    install_bytes_once,
    stream_hash_file_descriptor_bound,
)
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft.segment_mode_identity import resolve_checkpoint_sft_segment_mode
from tasks.mixture_names import SFTMixture


class Stage(StrEnum):
    INSTRUCT = "instruct"
    SAFETY = "safety"
    CAPABILITIES = "capabilities"


class Action(StrEnum):
    BASE = "base"
    RESUME = "resume"
    INVENTORY = "inventory"


PREVIOUS_MIXTURE = {
    Stage.SAFETY: SFTMixture.STANDARD_STEERED,
    Stage.CAPABILITIES: SFTMixture.SAFETY,
}

def load_strict_json_object(path: Path) -> dict[str, Any]:
    """Load one stable JSON object with duplicate/nonfinite values rejected."""
    snapshot = stream_hash_file_descriptor_bound(
        path.resolve(strict=True),
        retain_payload=True,
    )
    assert snapshot.payload is not None
    value = json.loads(
        snapshot.payload,
        object_pairs_hook=strict_json_object,
        parse_constant=reject_nonstandard_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value

def nonempty(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"required nonempty file: {path}")
    return path

def required_file_digest(path: Path, *, retain_payload: bool = False) -> StreamedFileDigest:
    """Hash a required nonempty file while preserving symlink-transparent inputs."""
    return stream_hash_file_descriptor_bound(
        nonempty(path).resolve(strict=True),
        retain_payload=retain_payload,
    )

def install_json_once(path: Path, payload: Any) -> None:
    """Install exact legacy Campaign JSON bytes through immutable I/O."""
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    install_bytes_once(encoded, path, surface="Campaign 2 artifact")

def prepare_base(source: Path, step: int, base: Path, stage: Stage) -> Path:
    """Pin an explicit prior SFT step; never follow latest or an RL checkpoint."""
    if stage not in PREVIOUS_MIXTURE or type(step) is not int or step < 0:
        raise ValueError("base requires safety/capabilities and a nonnegative explicit step")
    source = source.resolve(strict=True)
    model = nonempty(source / f"model_{step:06d}.pt")
    metadata = nonempty(source / f"meta_{step:06d}.json")
    meta = load_strict_json_object(metadata)
    if meta.get("dataset") != PREVIOUS_MIXTURE[stage].value:
        raise ValueError(f"{stage.value} must start from {PREVIOUS_MIXTURE[stage].value} SFT")
    if meta.get("step") != step or meta.get("sft_loss_mask") != LossMaskMode.COMPLETION_ONLY:
        raise ValueError("previous checkpoint must match the step and completion_only mask")
    if any(key in meta for key in ("selfcond", "selfcond_data", "grpo")):
        raise ValueError("Campaign 2 starts from ordinary masked SFT, not RL/selfcond training")
    if resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=str(source)) is not SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError("previous checkpoint must be turn-delimited SFT")
    if meta.get("user_config", {}).get("segment_size") != 32:
        raise ValueError("previous checkpoint must use source segment_size=32")
    if base.is_symlink() or base.resolve() == source:
        raise ValueError("stage base must be a distinct real directory")
    base.mkdir(parents=True, exist_ok=True)
    allowed = {model.name, metadata.name, f"{stage.value}_sft_conditioned_checkpoints"}
    if any(path.name not in allowed for path in base.iterdir()):
        raise ValueError(f"stage base contains unrelated artifacts: {base}")
    for target in (model, metadata):
        link = base / target.name
        if link.is_symlink():
            if link.resolve(strict=True) != target:
                raise ValueError(f"refusing to change pinned stage parent: {link}")
        elif link.exists():
            raise ValueError(f"stage parent must be a symlink: {link}")
        else:
            link.symlink_to(target)
    return base / f"{stage.value}_sft_conditioned_checkpoints"

def latest_complete(leaf: Path, world_size: int = 4) -> int:
    """Ignore partial saves and require exact, end-anchored optimizer rank names."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    steps = sorted(
        (int(match[1]) for path in leaf.glob("model_*.pt")
         if (match := re.fullmatch(r"model_(\d{6,})\.pt", path.name))), reverse=True,
    )
    for step in steps:
        expected = {f"optim_{step:06d}_rank{rank}.pt" for rank in range(world_size)}
        present = {
            path.name for path in leaf.glob(f"optim_{step:06d}_rank*.pt")
            if re.fullmatch(r"optim_\d+_rank\d+\.pt", path.name)
        }
        files = [leaf / f"model_{step:06d}.pt", leaf / f"meta_{step:06d}.json",
                 *(leaf / name for name in expected)]
        if present == expected and all(path.is_file() and path.stat().st_size for path in files):
            meta = load_strict_json_object(leaf / f"meta_{step:06d}.json")
            if meta.get("step") != step:
                raise ValueError("checkpoint filename/metadata step mismatch")
            return step
    if any(leaf.glob("model_*.pt")) or any(leaf.glob("optim_*.pt")):
        raise ValueError(f"no complete resumable checkpoint in nonempty leaf: {leaf}")
    return -1

def inventory(parquet: Path, features: Path | None = None) -> int:
    meta = load_strict_json_object(nonempty(parquet / "conversion_meta.json"))
    count = meta.get("total_shards")
    if type(count) is not int or count <= 0 or meta.get("completed_shards") != count:
        raise ValueError("conversion must be complete with positive total_shards")
    expected = {f"shard_{index:05d}.parquet" for index in range(count)}
    if {path.name for path in parquet.glob("shard_*.parquet")} != expected:
        raise ValueError("parquet shard set differs from completed conversion")
    for name in expected:
        nonempty(parquet / name)
    if features is not None:
        expected_features = {f"shard_{index:05d}_features.safetensors.zst" for index in range(count)}
        if {path.name for path in features.glob("shard_*_features.safetensors.zst")} != expected_features:
            raise ValueError("feature shard set differs from completed conversion")
        for name in expected_features:
            nonempty(features / name)
    return count

@hydra.main(version_base=None, config_path="../../configs/utils", config_name="campaign2")
def main(cfg: DictConfig) -> None:
    action = Action(cfg.action)
    if action is Action.BASE:
        print(prepare_base(Path(cfg.source), cfg.step, Path(cfg.base), Stage(cfg.stage)))
    elif action is Action.RESUME:
        print(latest_complete(Path(cfg.leaf), cfg.world_size))
    else:
        print(inventory(Path(cfg.parquet), None if cfg.features is None else Path(cfg.features)))

if __name__ == "__main__":
    main()
