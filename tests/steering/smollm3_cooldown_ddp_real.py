"""Distributed stock-Trainer save/resume acceptance harness.

This module is intentionally not collected by ordinary pytest. Run it twice
under torchrun, first with ``SMOLLM3_COOLDOWN_DDP_TARGET_STEP=1`` and then 2,
against the same output directory and a prepared real-shard receipt. The
expected world and local-world sizes default to the original 2/2 gate and must
be overridden explicitly for any other topology.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch as t
import torch.distributed as dist

from nanochat.common import get_peak_flops
from steering.sft.hf_decoder import HFDecoderModelAdapter
from steering.smollm3_cooldown import (
    CONTEXT_LENGTH,
    PREPARED_SCHEMA_VERSION,
    SOURCE_MODEL_ID,
    SOURCE_REVISION,
    CooldownCallback,
    CooldownDataset,
    CooldownTrainer,
    PreparedShard,
    build_training_arguments,
    cooldown_collator,
    gradient_accumulation_steps,
    load_source_model,
    load_source_tokenizer,
    resolve_resume_checkpoint,
    semantic_digest,
    sha256_file,
    tokenizer_identity,
)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} must be a nonempty environment variable")
    return value


def _expected_topology() -> tuple[int, int]:
    expected_world_size = int(os.environ.get("SMOLLM3_COOLDOWN_DDP_WORLD_SIZE", "2"))
    expected_local_world_size = int(
        os.environ.get("SMOLLM3_COOLDOWN_DDP_LOCAL_WORLD_SIZE", "2")
    )
    if (
        expected_world_size <= 0
        or expected_local_world_size <= 0
        or expected_world_size % expected_local_world_size
    ):
        raise RuntimeError(
            "the expected DDP world size must be a positive multiple of local-world size"
        )
    gradient_accumulation_steps(expected_world_size)
    return expected_world_size, expected_local_world_size


def _prepared_shard(receipt_path: Path) -> tuple[dict[str, Any], PreparedShard]:
    receipt = json.loads(receipt_path.read_text())
    token_count = int(receipt["token_count"])
    expected_contexts = max(0, (token_count - 1) // CONTEXT_LENGTH)
    if (
        receipt.get("schema_version") != PREPARED_SCHEMA_VERSION
        or receipt.get("context_stride") != CONTEXT_LENGTH
        or receipt.get("context_count") != expected_contexts
    ):
        raise RuntimeError("DDP prepared-shard causal-stride contract mismatch")
    stem = receipt_path.stem
    token_path = receipt_path.with_name(f"{stem}.ds")
    index_path = receipt_path.with_name(f"{stem}.ds.index")
    return receipt, PreparedShard(
        stem=stem,
        token_path=token_path,
        index_path=index_path,
        token_count=token_count,
        document_count=int(receipt["document_count"]),
        context_count=int(receipt["context_count"]),
        token_sha256=str(receipt["files"]["tokens"]["sha256"]),
        index_sha256=str(receipt["files"]["index"]["sha256"]),
    )


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size, expected_local_world_size = _expected_topology()
    if world_size != expected_world_size or local_world_size != expected_local_world_size:
        raise RuntimeError(
            "the DDP acceptance topology differs from the explicit expectation: "
            f"actual={world_size}/{local_world_size}, "
            f"expected={expected_world_size}/{expected_local_world_size}"
        )
    target_step = int(_required_environment("SMOLLM3_COOLDOWN_DDP_TARGET_STEP"))
    if target_step not in (1, 2):
        raise RuntimeError("SMOLLM3_COOLDOWN_DDP_TARGET_STEP must be 1 or 2")
    receipt_path = Path(_required_environment("SMOLLM3_COOLDOWN_DDP_RECEIPT")).resolve()
    output_dir = Path(_required_environment("SMOLLM3_COOLDOWN_DDP_OUTPUT")).resolve()

    t.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    receipt, shard = _prepared_shard(receipt_path)
    if shard.context_count < 1_152:
        raise RuntimeError(f"DDP receipt has only {shard.context_count} contexts; 1,152 required")
    if sha256_file(shard.token_path) != shard.token_sha256:
        raise RuntimeError("DDP token body digest mismatch")
    if sha256_file(shard.index_path) != shard.index_sha256:
        raise RuntimeError("DDP index body digest mismatch")
    tokenizer = load_source_tokenizer(local_files_only=True)
    if tokenizer_identity(tokenizer) != receipt["tokenizer_sha256"]:
        raise RuntimeError("DDP tokenizer differs from prepared real data")

    semantic = {
        "format": "smollm3_cooldown_distributed_acceptance_v1",
        "source": {"model_id": SOURCE_MODEL_ID, "revision": SOURCE_REVISION},
        "receipt_sha256": sha256_file(receipt_path),
        "token_sha256": shard.token_sha256,
        "index_sha256": shard.index_sha256,
        "batch": {
            "world_size": world_size,
            "local_world_size": local_world_size,
            "contexts": 1_152,
        },
        "execution": {"gradient_checkpointing": True},
    }
    digest = semantic_digest(semantic)
    resume = resolve_resume_checkpoint(output_dir, expected_digest=digest) if rank == 0 else None
    objects: list[Any] = [None if resume is None else str(resume)]
    dist.broadcast_object_list(objects, src=0)
    resume = None if objects[0] is None else Path(str(objects[0]))
    expected_resume_step = target_step - 1
    if expected_resume_step == 0 and resume is not None:
        raise RuntimeError("step-1 DDP acceptance requires an empty output directory")
    if expected_resume_step > 0 and (
        resume is None or resume.name != f"checkpoint-{expected_resume_step}"
    ):
        raise RuntimeError(
            f"step-{target_step} DDP acceptance requires checkpoint-{expected_resume_step}"
        )

    model = load_source_model(
        local_files_only=True,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    dataset = CooldownDataset([shard], length=1_152)
    args = build_training_arguments(
        output_dir=output_dir,
        world_size=world_size,
        logging_steps=1,
        save_steps=2_000,
        dataloader_workers=0,
    )
    callback = CooldownCallback(
        semantic=semantic,
        canary_steps=target_step,
        flops_per_token=HFDecoderModelAdapter(
            model,
            max_seq_len=CONTEXT_LENGTH,
        ).estimate_flops(),
        peak_flops=get_peak_flops(t.cuda.get_device_name(local_rank)),
        dataset=dataset,
    )
    trainer = CooldownTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=cooldown_collator,
        callbacks=[callback],
    )
    result = trainer.train(resume_from_checkpoint=None if resume is None else str(resume))
    if trainer.state.global_step != target_step or int(result.global_step) != target_step:
        raise RuntimeError(
            f"DDP acceptance stopped at {trainer.state.global_step}; expected {target_step}"
        )
    dist.barrier()
    if rank == 0:
        marker_path = output_dir / f"checkpoint-{target_step}" / "cooldown_complete.json"
        marker = json.loads(marker_path.read_text())
        if marker["completed_cooldown_step"] != target_step:
            raise RuntimeError("DDP checkpoint marker step mismatch")
        print(
            json.dumps(
                {
                    "target_step": target_step,
                    "resume": None if resume is None else str(resume),
                    "checkpoint": str(marker_path.parent),
                    "train_loss": result.training_loss,
                    "metrics": result.metrics,
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
