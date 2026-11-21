"""Atomic model-only Hugging Face checkpoint publication for decoder SFT."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from steering._common.ddp import rank_synchronized_stage
from steering.sft.hf_contract import HFDecoderContract
from steering.sft.hf_decoder import (
    HF_ATTENTION_IMPLEMENTATION,
    HFSFTTokenizer,
)


def repository_version() -> dict[str, str | None]:
    """Return package and current Git identities for export provenance."""
    try:
        package_version = importlib.metadata.version("nano-gpt-pretrain-steer")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    return {
        "package": package_version,
        "git_revision": git_revision,
    }


def _tensor_alias_key(tensor: t.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def build_cpu_bf16_state_dict(model: t.nn.Module) -> dict[str, t.Tensor]:
    """Cast a detached state dict without mutating live FP32 parameters.

    Exact shared-storage aliases (notably tied input/output embeddings) reuse
    one cast tensor so publication does not accidentally duplicate them.
    """
    aliases: dict[tuple[Any, ...], t.Tensor] = {}
    output: dict[str, t.Tensor] = {}
    for name, value in model.state_dict().items():
        key = _tensor_alias_key(value)
        converted = aliases.get(key)
        if converted is None:
            target_dtype = t.bfloat16 if value.is_floating_point() else value.dtype
            converted = value.detach().to(device="cpu", dtype=target_dtype, copy=True)
            aliases[key] = converted
        output[name] = converted
    return output


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stamp_bf16_export_config(path: Path) -> None:
    """Record the serialized weight precision, not the live FP32 precision."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("HF config.json must contain a JSON object")
    payload["dtype"] = "bfloat16"
    # Transformers 5 writes ``dtype``. Remove the legacy spelling if a
    # version transition ever emits both so the contract remains unambiguous.
    payload.pop("torch_dtype", None)
    _write_strict_json(path, payload)


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in directory.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [directory]:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _inspect_staged_checkpoint(
    directory: Path,
    expected_metadata: dict[str, Any],
    contract: HFDecoderContract,
) -> None:
    model_files = sorted(directory.glob("model*.safetensors"))
    if not model_files:
        raise RuntimeError("HF export produced no model*.safetensors files")
    for filename in ("config.json", "generation_config.json", "sft_metadata.json"):
        if not (directory / filename).is_file():
            raise RuntimeError(f"HF export is missing required {filename}")
    with (directory / "sft_metadata.json").open(encoding="utf-8") as handle:
        actual_metadata = json.load(handle)
    if actual_metadata != expected_metadata:
        raise RuntimeError("sft_metadata.json did not round-trip exactly")
    config = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    if config.model_type != contract.model_type:
        raise RuntimeError("staged HF config has the wrong model type")
    if config.vocab_size != contract.model_vocab_size:
        raise RuntimeError("staged HF config has the wrong model vocabulary size")
    if config.bos_token_id != contract.generation_bos_token_id:
        raise RuntimeError("staged HF config has the wrong BOS token id")
    if config.eos_token_id != contract.generation_eos_token_id:
        raise RuntimeError("staged HF config has the wrong EOS token id")
    if config.pad_token_id != contract.generation_pad_token_id:
        raise RuntimeError("staged HF config has the wrong PAD token id")
    if config.dtype != t.bfloat16:
        raise RuntimeError(f"staged HF config dtype must be bfloat16, got {config.dtype}")
    generation_config = GenerationConfig.from_pretrained(directory)
    if generation_config.bos_token_id != contract.generation_bos_token_id:
        raise RuntimeError("staged generation config has the wrong BOS token id")
    if generation_config.eos_token_id != contract.generation_eos_token_id:
        raise RuntimeError("staged generation config has the wrong EOS token id")
    if generation_config.pad_token_id != contract.generation_pad_token_id:
        raise RuntimeError("staged generation config has the wrong PAD token id")
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(directory, trust_remote_code=False),
    )
    if not tokenizer.chat_template:
        raise RuntimeError("staged HF tokenizer has no chat template")
    if (
        tokenizer.bos_token_id != contract.generation_bos_token_id
        or tokenizer.eos_token_id != contract.generation_eos_token_id
        or tokenizer.pad_token_id != contract.generation_pad_token_id
    ):
        raise RuntimeError("staged HF tokenizer BOS/EOS/PAD contract drifted")
    if len(tokenizer) != contract.tokenizer_vocab_size:
        raise RuntimeError("staged HF tokenizer vocabulary size drifted")


def publish_hf_checkpoint(
    *,
    output_root: str | Path,
    completed_step: int,
    model: PreTrainedModel,
    tokenizer: HFSFTTokenizer,
    metadata: dict[str, Any],
    max_shard_size: str = "2GB",
) -> Path:
    """Atomically publish one BF16 HF checkpoint with exclusive ownership."""
    if completed_step < 0:
        raise ValueError(f"completed_step must be non-negative, got {completed_step}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    step_name = f"step_{completed_step:06d}"
    final_dir = root / step_name
    lock_path = root / f".{step_name}.lock"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing HF checkpoint {final_dir}")

    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    temp_dir = root / f".{step_name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    publication_started = False
    try:
        if final_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing HF checkpoint {final_dir}")
        temp_dir.mkdir(mode=0o700)
        state_dict = build_cpu_bf16_state_dict(model)
        # Transformers annotates the in-memory config with architecture and
        # dtype during save_pretrained. Export must not mutate the live FP32
        # training model, so preserve object identity and restore both config
        # dictionaries even when serialization raises.
        config_state = deepcopy(model.config.__dict__)
        generation_config_state = deepcopy(model.generation_config.__dict__)
        try:
            model.save_pretrained(
                temp_dir,
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size=max_shard_size,
            )
        finally:
            model.config.__dict__.clear()
            model.config.__dict__.update(config_state)
            model.generation_config.__dict__.clear()
            model.generation_config.__dict__.update(generation_config_state)
        del state_dict
        _stamp_bf16_export_config(temp_dir / "config.json")
        tokenizer.save(temp_dir)
        _write_strict_json(temp_dir / "sft_metadata.json", metadata)
        _inspect_staged_checkpoint(temp_dir, metadata, tokenizer.contract)
        _fsync_tree(temp_dir)
        publication_started = True
        temp_dir.rename(final_dir)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        return final_dir
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if publication_started and final_dir.exists():
            shutil.rmtree(final_dir)
        raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def publish_hf_checkpoint_distributed(
    *,
    output_root: str | Path,
    completed_step: int,
    model: PreTrainedModel,
    tokenizer: HFSFTTokenizer,
    metadata: dict[str, Any],
    master_process: bool,
    device: str | t.device,
) -> Path:
    """Publish on rank zero and propagate failure before another collective."""
    final_dir = Path(output_root) / f"step_{completed_step:06d}"
    with rank_synchronized_stage(
        f"HF checkpoint publication step {completed_step}",
        device=device,
    ):
        if master_process:
            final_dir = publish_hf_checkpoint(
                output_root=output_root,
                completed_step=completed_step,
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
            )
    return final_dir


def validate_export_smoke(
    checkpoint_dir: str | Path,
    *,
    contract: HFDecoderContract,
    device: str | t.device,
    max_new_tokens: int = 16,
) -> None:
    """Reload the final export and run serialization-focused forward/generation gates."""
    checkpoint_dir = Path(checkpoint_dir)
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            trust_remote_code=False,
            dtype=t.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=HF_ATTENTION_IMPLEMENTATION,
        ),
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=False),
    )
    t.nn.Module.to(model, device=device)
    model.eval()
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {t.bfloat16}:
        raise RuntimeError(f"reloaded HF floating parameter dtypes are {floating_dtypes}")
    if not isinstance(model, contract.model_class):
        raise RuntimeError(
            f"reloaded HF class is {type(model).__name__}, expected {contract.architecture}"
        )
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if tied != contract.tied_embeddings:
        raise RuntimeError("reloaded HF checkpoint embedding ownership drifted")
    if int(model.config.vocab_size) != contract.model_vocab_size:
        raise RuntimeError(f"reloaded HF vocab size is {model.config.vocab_size}")
    if not tokenizer.chat_template:
        raise RuntimeError("reloaded tokenizer lost its chat template")
    if (
        tokenizer.bos_token_id != contract.generation_bos_token_id
        or tokenizer.eos_token_id != contract.generation_eos_token_id
        or tokenizer.pad_token_id != contract.generation_pad_token_id
    ):
        raise RuntimeError("reloaded tokenizer BOS/EOS/PAD contract drifted")
    if len(tokenizer) != contract.tokenizer_vocab_size:
        raise RuntimeError("reloaded tokenizer vocabulary size drifted")

    messages = [{"role": "user", "content": "Write one short greeting."}]
    prompt_rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(prompt_rendered, BatchEncoding):
        raise RuntimeError("saved chat template did not return a BatchEncoding")
    raw_prompt_ids = cast(Any, prompt_rendered)["input_ids"]
    if not isinstance(raw_prompt_ids, t.Tensor):
        raise RuntimeError("saved chat template did not return a prompt tensor")
    prompt_ids = raw_prompt_ids.to(device)
    with t.inference_mode():
        logits = model(input_ids=prompt_ids, use_cache=False).logits
    if not t.isfinite(logits).all():
        raise RuntimeError("reloaded HF checkpoint produced nonfinite logits")
    with t.inference_mode():
        generated = cast(Any, model).generate(
            prompt_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=contract.generation_eos_token_id,
            pad_token_id=contract.generation_pad_token_id,
        )
    if not t.equal(generated[:, : prompt_ids.shape[1]], prompt_ids):
        raise RuntimeError("HF generation did not preserve the prompt prefix")
    continuation = generated[0, prompt_ids.shape[1] :].tolist()
    if not continuation:
        raise RuntimeError("HF generation produced an empty continuation")
    if any(token_id < 0 or token_id >= contract.tokenizer_vocab_size for token_id in continuation):
        raise RuntimeError("HF generation produced an out-of-vocabulary token id")
    decoded = tokenizer.decode(continuation, skip_special_tokens=False)
    if not decoded:
        raise RuntimeError("HF generation continuation is not decodable")
    if contract.generation_eos_token_id in continuation:
        first_eos = continuation.index(contract.generation_eos_token_id)
        if first_eos != len(continuation) - 1:
            raise RuntimeError("HF generation emitted tokens after its EOS token")
    del generated, logits, prompt_ids, tokenizer, model
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
