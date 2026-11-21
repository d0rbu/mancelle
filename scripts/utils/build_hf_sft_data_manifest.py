"""Freeze or verify the actual HF SFT task bodies on CPU.

The trainer consumes only this file's SHA-256 as data_snapshot_sha256; its
huggingface_standard_sft format describes exported models, not data manifests.
Task content hashes plus the shared index-map fingerprints bind every ordered
conversation, including synthetic spelling data and both identity repetitions.
Run with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 after staging the HF datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.utils.campaign2 import (
    load_strict_json_object,
    nonempty,
    required_file_digest,
)
from steering.eval.hf_chat_generation import resolve_repository_hf_checkpoint_dir
from steering.sft.hf_contract import validate_repository_hf_sft_metadata
from steering.sft.hf_decoder import load_hf_config_and_tokenizer
from steering.sft.iteration import sft_data_generator_bos_bestfit_core
from steering.sft.loss_mask import LossMaskMode
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft_schema import SFTSplit
from tasks.common import TaskMixture
from tasks.mixtures import SFTMixture, build_task_mixtures, steered_mixture_overrides
from tasks.spellingbee import ensure_word_list_available

DEFAULT_SFT_CONFIG_NAME = "chat_sft_pythia_2_8b"


class ManifestMode(StrEnum):
    BUILD = "build"
    VERIFY = "verify"


MANIFEST_FORMAT = "hf_sft_data_manifest_v1"


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if path.stat().st_size == 0:
        raise ValueError(f"SFT input is empty: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}


def _parent_identity(path: Path, expected_mixture: SFTMixture) -> dict[str, Any]:
    root = resolve_repository_hf_checkpoint_dir(path)
    meta = validate_repository_hf_sft_metadata(
        load_strict_json_object(root / "sft_metadata.json")
    )
    if meta.get("mixture") != expected_mixture.value:
        raise ValueError("wrong parent mixture for chained Pythia stage")
    if meta.get("sft_loss_mask") != LossMaskMode.COMPLETION_ONLY.value:
        raise ValueError("Pythia parent must be completion_only SFT")
    if (
        meta["output"]["model_type"] != "gpt_neox"
        or meta["output"]["architecture"] != "GPTNeoXForCausalLM"
    ):
        raise ValueError("Campaign 2 requires a Pythia GPTNeoX export")
    if meta["completed_optimizer_step"] != int(root.name.removeprefix("step_")):
        raise ValueError("HF export directory/metadata step mismatch")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        nonempty(root / name)
    weights = list(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("parent export has no safetensors weights")
    index = root / "model.safetensors.index.json"
    if len(weights) > 1 or index.exists():
        weight_map = load_strict_json_object(index).get("weight_map", {})
        if not weight_map or set(weight_map.values()) != {weight.name for weight in weights}:
            raise ValueError("parent export shard index mismatch")
    files = {
        file.name: required_file_digest(file).sha256
        for file in sorted(root.iterdir())
        if file.is_file()
    }
    return {"path": str(root), "source_revision": meta["source"]["revision"], "files": files}


def _mixture_identity(
    dataset: TaskMixture, split: SFTSplit, mixture: SFTMixture = SFTMixture.STANDARD_STEERED
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError(f"SFT {split.value} mixture is empty")
    tasks = []
    for task in dataset.tasks:
        digest = hashlib.sha256()
        for index in range(len(task)):
            conversation = json.dumps(
                task[index],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            digest.update(conversation.encode("utf-8") + b"\n")
        tasks.append(
            {
                "class_name": type(task).__name__,
                "length": len(task),
                "conversations_sha256": digest.hexdigest(),
            }
        )
        logger.info("Hashed {} {}: {} conversations", split.value, type(task).__name__, len(task))
    return {
        "fingerprint": compute_mixture_fingerprint(
            dataset,
            mixture_name=mixture.value,
            split=split,
        ),
        "tasks": tasks,
    }


def measure_epoch(
    dataset: TaskMixture,
    tokenizer: Any,
    *,
    device_batch_size: int,
    loss_mask: LossMaskMode,
    max_seq_len: int = 2048,
    total_batch_size: int = 524288,
    world_size: int = 4,
) -> dict[str, Any]:
    """Replay the trainer's packer on CPU to each rank's first epoch boundary.

    Round each rank up to a complete optimizer update, then take the maximum
    so every rank reaches one epoch of committed conversation consumption.
    Best-fit lookahead and final-batch rounding can repeat/retain boundary
    conversations, exactly as the existing single-rank dataset horizon does.
    """
    if isinstance(device_batch_size, bool) or device_batch_size <= 0 or len(dataset) < world_size:
        raise ValueError("positive device batch and at least one conversation per rank required")
    micro_tokens = world_size * device_batch_size * max_seq_len
    if total_batch_size % micro_tokens:
        raise ValueError("total_batch_size must be divisible by world micro-batch tokens")
    accumulation = total_batch_size // micro_tokens
    per_rank_steps = []
    for rank in range(world_size):
        rank_size = len(range(rank, len(dataset), world_size))
        batches = sft_data_generator_bos_bestfit_core(
            dataset=dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            device_batch_size=device_batch_size,
            ddp_rank=rank,
            ddp_world_size=world_size,
            device=t.device("cpu"),
            pin_memory=False,
            loss_mask_mode=loss_mask,
        )
        for _inputs, _targets, progress in batches:
            if progress.micro_batch_count % 10000 == 0:
                logger.info(
                    "Epoch rank {}: {}/{} conversations consumed",
                    rank,
                    progress.consumed,
                    rank_size,
                )
            if progress.consumed >= rank_size:
                steps = (progress.micro_batch_count + accumulation - 1) // accumulation
                per_rank_steps.append(steps)
                logger.info("Epoch rank {}: {} optimizer steps", rank, steps)
                break
    return {
        "world_size": world_size,
        "device_batch_size": device_batch_size,
        "max_seq_len": max_seq_len,
        "total_batch_size": total_batch_size,
        "sft_loss_mask": loss_mask.value,
        "gradient_accumulation_steps": accumulation,
        "per_rank_steps": per_rank_steps,
        "num_iterations": max(per_rank_steps),
        "scheduled_tokens": max(per_rank_steps) * total_batch_size,
    }


def validate_epoch(
    epoch: Any, *, device_batch_size: int, loss_mask: LossMaskMode
) -> dict[str, Any]:
    """Check the frozen CPU measurement without replaying it on a GPU allocation."""
    if not isinstance(epoch, dict):
        raise ValueError("manifest epoch must be an object")
    steps = epoch.get("per_rank_steps")
    if (
        not isinstance(steps, list)
        or len(steps) != 4
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
    ):
        raise ValueError("manifest epoch requires four positive per-rank step counts")
    if isinstance(device_batch_size, bool) or device_batch_size <= 0 or 64 % device_batch_size:
        raise ValueError("device_batch_size must be a positive divisor of 64")
    expected = {
        "world_size": 4,
        "device_batch_size": device_batch_size,
        "max_seq_len": 2048,
        "total_batch_size": 524288,
        "sft_loss_mask": loss_mask.value,
        "gradient_accumulation_steps": 64 // device_batch_size,
        "per_rank_steps": steps,
        "num_iterations": max(steps),
        "scheduled_tokens": max(steps) * 524288,
    }
    if epoch != expected or any(
        type(epoch[key]) is not type(value) for key, value in expected.items()
    ):
        raise ValueError("manifest epoch geometry or derived step count differs")
    return expected


def build_manifest(
    *,
    base_dir: Path,
    steered_jsonl: Path | None = None,
    mixture: SFTMixture = SFTMixture.STANDARD_STEERED,
    device_batch_size: int = 1,
    loss_mask: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    measured_epoch: dict[str, Any] | None = None,
    sft_config_name: str = DEFAULT_SFT_CONFIG_NAME,
) -> dict[str, Any]:
    """Use the trainer's default mixture builder; never read prepared parquet.

    ``sft_config_name`` selects the training config whose pinned Hugging Face
    model/revision provides the tokenizer identity recorded in the manifest.
    """
    if not isinstance(sft_config_name, str) or not sft_config_name.strip():
        raise ValueError("sft_config_name must be a nonblank training config name")
    base_dir = base_dir.resolve(strict=True)
    if mixture not in (SFTMixture.STANDARD_STEERED, SFTMixture.SAFETY, SFTMixture.CAPABILITIES):
        raise ValueError(f"unsupported production mixture: {mixture.value}")
    steered_jsonl = None if steered_jsonl is None else steered_jsonl.resolve(strict=True)
    overrides = steered_mixture_overrides(mixture, None if steered_jsonl is None else str(steered_jsonl))
    if not base_dir.is_dir():
        raise ValueError(f"base_dir must be a directory: {base_dir}")
    previous_base = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)
    try:
        inputs = {
            "identity": _file_identity(base_dir / "identity_conversations.jsonl"),
        }
        if steered_jsonl is not None:
            inputs["steered"] = _file_identity(steered_jsonl)
        if mixture is not SFTMixture.SAFETY:
            inputs["spelling_words"] = _file_identity(ensure_word_list_available(allow_network=False))
        train, val = build_task_mixtures(mixture, **overrides)
        if val is None:
            raise ValueError(f"{mixture.value} requires a validation mixture")
        config_dir = Path(__file__).resolve().parents[2] / "configs/training"
        config_path = config_dir / f"{sft_config_name}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"unknown SFT training config: {config_path}")
        cfg = OmegaConf.load(config_path)
        _config, tokenizer, contract = load_hf_config_and_tokenizer(
            model_id=cfg.huggingface.model_id,
            revision=cfg.huggingface.revision,
        )
        payload = {
            "epoch": (
                measure_epoch(
                    train, tokenizer, device_batch_size=device_batch_size, loss_mask=loss_mask
                )
                if measured_epoch is None
                else validate_epoch(
                    measured_epoch, device_batch_size=device_batch_size, loss_mask=loss_mask
                )
            ),
            "tokenizer": {
                "model_id": cfg.huggingface.model_id,
                "revision": cfg.huggingface.revision,
                "chat_template_sha256": contract.chat_template_sha256,
            },
            "format": MANIFEST_FORMAT,
            "mixture": mixture.value,
            "nanochat_base_dir": str(base_dir),
            "inputs": inputs,
            "train": _mixture_identity(train, SFTSplit.TRAIN, mixture),
            "validation": _mixture_identity(val, SFTSplit.VAL, mixture),
        }
        return payload
    finally:
        if previous_base is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous_base


def write_manifest(payload: dict[str, Any], output_path: Path) -> str:
    """Publish a new JSON manifest and sha256sum-compatible digest sidecar."""
    sidecar = Path(f"{output_path}.sha256")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite a frozen manifest: {output_path}")
    body = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        output.write(body)
    with sidecar.open("x", encoding="utf-8") as output:
        output.write(f"{digest}  {output_path.name}\n")
    return digest


def verify_manifest(payload: dict[str, Any], output_path: Path) -> str:
    """Reject missing, malformed, or changed data before GPU work begins."""
    body = output_path.read_bytes()
    if json.loads(body) != payload:
        raise ValueError(f"live SFT data differs from frozen manifest: {output_path}")
    return hashlib.sha256(body).hexdigest()


@hydra.main(
    version_base=None, config_path="../../configs/utils", config_name="hf_sft_data_manifest"
)
def main(cfg: DictConfig) -> None:
    mode = ManifestMode(cfg.mode)
    output_path = Path(cfg.output_path)
    if mode is ManifestMode.BUILD and (
        output_path.exists() or Path(f"{output_path}.sha256").exists()
    ):
        raise FileExistsError(f"refusing to overwrite a frozen manifest: {output_path}")
    if mode is ManifestMode.VERIFY and not output_path.is_file():
        raise FileNotFoundError(output_path)
    parent_values = (cfg.parent_export, cfg.expected_parent_mixture, cfg.data_manifest)
    supplied = tuple(value is not None for value in parent_values)
    if any(supplied) and not all(supplied):
        names = ("parent_export", "expected_parent_mixture", "data_manifest")
        missing = sorted(name for name, present in zip(names, supplied, strict=True) if not present)
        raise ValueError(
            "parent_export, expected_parent_mixture, and data_manifest must be supplied "
            f"together; missing={missing}"
        )
    has_parent = all(supplied)
    data_manifest = Path(cfg.data_manifest) if has_parent else None
    measured_epoch = (
        load_strict_json_object(data_manifest).get("epoch")
        if data_manifest is not None
        else validate_epoch(
            load_strict_json_object(output_path).get("epoch"),
            device_batch_size=cfg.device_batch_size,
            loss_mask=LossMaskMode(cfg.sft_loss_mask),
        )
        if mode is ManifestMode.VERIFY
        else None
    )
    payload = build_manifest(
        base_dir=Path(cfg.base_dir),
        steered_jsonl=None if cfg.steered_jsonl is None else Path(cfg.steered_jsonl),
        mixture=SFTMixture(cfg.mixture),
        device_batch_size=cfg.device_batch_size,
        loss_mask=LossMaskMode(cfg.sft_loss_mask),
        measured_epoch=measured_epoch,
        sft_config_name=str(cfg.sft_config_name),
    )
    if data_manifest is not None:
        data_digest = verify_manifest(payload, data_manifest)
        parent = _parent_identity(
            Path(cfg.parent_export), SFTMixture(cfg.expected_parent_mixture)
        )
        _, _, parent_contract = load_hf_config_and_tokenizer(
            model_id=parent["path"], revision=parent["source_revision"]
        )
        if parent_contract.chat_template_sha256 != payload["tokenizer"][
            "chat_template_sha256"
        ]:
            raise ValueError("parent tokenizer template differs from the measured data epoch")
        payload = {
            **payload,
            "campaign2_parent": parent,
            "campaign2_data_sha256": data_digest,
        }
    digest = (
        write_manifest(payload, output_path)
        if mode is ManifestMode.BUILD
        else verify_manifest(payload, output_path)
    )
    logger.info("{} manifest {}; SHA-256={}", mode.value, output_path, digest)


if __name__ == "__main__":
    main()
