"""Transformers-native Hugging Face decoder backend for standard SFT."""

from __future__ import annotations

import gc
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from nanochat.optim import DistMuonAdamW, MuonAdamW
from steering._common.ddp import rank_synchronized_stage
from steering.sft.hf_contract import (
    DecoderSFTBackend,
    HFAttentionImplementation,
    HFConfigTokenIds,
    HFDecoderContract,
    HFDistributedImplementation,
)
from steering.sft.pythia import PYTHIA_CONTRACT
from steering.sft.smollm3 import SMOLLM3_CONTRACT

HF_ATTENTION_IMPLEMENTATION = HFAttentionImplementation.SDPA.value

_CONTRACTS_BY_MODEL_TYPE = {
    SMOLLM3_CONTRACT.model_type: SMOLLM3_CONTRACT,
    PYTHIA_CONTRACT.model_type: PYTHIA_CONTRACT,
}
_SFT_METADATA_FILENAME = "sft_metadata.json"


def _field(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name)


def _finite_number(
    raw: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a finite number; got {raw!r}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {raw!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive; got {raw!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative; got {raw!r}")
    return value


def validate_decoder_sft_config(
    cfg: Any,
    *,
    source_config: Any | None = None,
    world_size: int | None = None,
) -> DecoderSFTBackend:
    """Validate the backend discriminator and its disjoint config surface."""
    raw_backend = _field(cfg, "model_backend")
    try:
        backend = DecoderSFTBackend(raw_backend)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"model_backend must be exactly 'nanochat' or 'huggingface'; got {raw_backend!r}"
        ) from exc

    num_iterations = _field(cfg, "num_iterations")
    if (
        isinstance(num_iterations, bool)
        or not isinstance(num_iterations, int)
        or num_iterations == 0
        or num_iterations < -1
    ):
        raise ValueError(
            f"num_iterations must be -1 or a positive optimizer-step count; got {num_iterations!r}"
        )
    clip = _field(cfg, "grad_clip_max_norm")
    if clip is not None:
        _finite_number(clip, name="grad_clip_max_norm", positive=True)

    hf = _field(cfg, "huggingface")
    if backend is DecoderSFTBackend.NANOCHAT:
        if hf is not None:
            raise ValueError("model_backend=nanochat requires huggingface: null")
        return backend

    if hf is None:
        raise ValueError("model_backend=huggingface requires a huggingface config block")
    model_id = _field(hf, "model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("huggingface.model_id must be a nonempty string")
    revision = _field(hf, "revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "huggingface.revision must be a full 40-character lowercase commit SHA; "
            f"got {revision!r}"
        )
    if _field(cfg, "dtype") != "bfloat16":
        raise ValueError("model_backend=huggingface requires dtype=bfloat16")
    for name in ("source", "model_tag", "model_step"):
        if _field(cfg, name) is not None:
            raise ValueError(f"model_backend=huggingface requires {name}=null")
    for name in ("embedding_lr", "unembedding_lr", "matrix_lr"):
        if _field(cfg, name) is not None:
            raise ValueError(
                f"model_backend=huggingface rejects native optimizer knob {name}; set it to null"
            )
    for name in ("gradient_checkpointing", "validate_export", "torch_compile"):
        value = _field(hf, name)
        if not isinstance(value, bool):
            raise ValueError(f"huggingface.{name} must be a boolean; got {value!r}")
    try:
        HFAttentionImplementation(_field(hf, "attention_implementation"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "huggingface.attention_implementation must be exactly 'sdpa' or "
            "'flash_attention_4'"
        ) from exc
    try:
        HFDistributedImplementation(_field(hf, "distributed_implementation"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "huggingface.distributed_implementation must be exactly "
            "'dist_muon_adamw' or 'ddp_fused_adamw'"
        ) from exc
    restart_from_step = _field(hf, "restart_from_step")
    if (
        isinstance(restart_from_step, bool)
        or not isinstance(restart_from_step, int)
        or restart_from_step < -1
    ):
        raise ValueError(
            "huggingface.restart_from_step must be -1 or a non-negative integer; "
            f"got {restart_from_step!r}"
        )
    restart_save_every = _field(hf, "restart_save_every")
    if (
        isinstance(restart_save_every, bool)
        or not isinstance(restart_save_every, int)
        or restart_save_every < 0
    ):
        raise ValueError(
            "huggingface.restart_save_every must be a non-negative integer; "
            f"got {restart_save_every!r}"
        )
    data_snapshot_sha256 = _field(hf, "data_snapshot_sha256")
    if data_snapshot_sha256 is not None and (
        not isinstance(data_snapshot_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", data_snapshot_sha256) is None
    ):
        raise ValueError(
            "huggingface.data_snapshot_sha256 must be null or a 64-character "
            "lowercase hexadecimal SHA-256 digest"
        )
    output_dir = _field(hf, "output_dir")
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        raise ValueError("huggingface.output_dir must be null or a nonempty path string")

    max_seq_len = _field(cfg, "max_seq_len")
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be a positive integer; got {max_seq_len!r}")
    _finite_number(_field(cfg, "weight_decay"), name="weight_decay", nonnegative=True)
    _finite_number(_field(hf, "learning_rate"), name="huggingface.learning_rate", positive=True)
    _finite_number(_field(hf, "adam_eps"), name="huggingface.adam_eps", positive=True)
    betas = _field(hf, "adam_betas")
    if not isinstance(betas, Sequence) or isinstance(betas, (str, bytes)) or len(betas) != 2:
        raise ValueError("huggingface.adam_betas must contain exactly two finite values in [0, 1)")
    for idx, beta in enumerate(betas):
        value = _finite_number(beta, name=f"huggingface.adam_betas[{idx}]")
        if value < 0 or value >= 1:
            raise ValueError(f"huggingface.adam_betas[{idx}] must be in [0, 1); got {beta!r}")

    if source_config is not None:
        source_limit = int(source_config.max_position_embeddings)
        if max_seq_len > source_limit:
            raise ValueError(f"max_seq_len={max_seq_len} exceeds source model limit {source_limit}")
    if world_size is not None:
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"world_size must be a positive integer; got {world_size!r}")
        if world_size > 1 and num_iterations == -1:
            raise ValueError(
                "distributed Hugging Face SFT requires a positive num_iterations; "
                "dataset-horizon runs are rank-asymmetric"
            )
    return backend


def validate_hf_source_config(
    config: PretrainedConfig,
    *,
    model_id: str | None = None,
    local_source: bool = False,
) -> HFDecoderContract:
    """Validate a supported pristine source or repository SFT export config."""
    model_type = str(getattr(config, "model_type", ""))
    try:
        contract = _CONTRACTS_BY_MODEL_TYPE[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Hugging Face decoder model_type {model_type!r}; "
            f"supported={sorted(_CONTRACTS_BY_MODEL_TYPE)}"
        ) from exc
    common_expected = {
        "model_type": contract.model_type,
        "vocab_size": contract.model_vocab_size,
        "tie_word_embeddings": contract.tied_embeddings,
    }
    model_id_fields = contract.source_config_fields_by_model_id
    infer_local_geometry = (
        local_source
        and model_id_fields is not None
        and (model_id is None or model_id not in model_id_fields)
    )
    if infer_local_geometry:
        candidates = [
            fields
            for fields in (contract.source_config_fields, *model_id_fields.values())
            if all(getattr(config, name, None) == expected for name, expected in fields.items())
        ]
        unique_candidates = {tuple(sorted(fields.items())): fields for fields in candidates}
        if len(unique_candidates) != 1:
            raise ValueError(
                f"{contract.family.value} local source geometry does not match exactly one "
                f"supported model size; source model_id={model_id!r}"
            )
        source_config_fields = next(iter(unique_candidates.values()))
    else:
        source_config_fields = contract.source_config_fields_for_model_id(model_id)
    expected_fields = {**common_expected, **source_config_fields}
    for name, expected in expected_fields.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise ValueError(
                f"{contract.family.value} source config mismatch for {name}: "
                f"expected {expected!r}, got {actual!r}"
            )
    architectures = list(getattr(config, "architectures", []) or [])
    if architectures != [contract.architecture]:
        raise ValueError(
            f"{contract.family.value} source architectures must resolve exactly to "
            f"{contract.architecture}; got {architectures!r}"
        )
    token_ids = HFConfigTokenIds(config.bos_token_id, config.eos_token_id, config.pad_token_id)
    if (
        any(type(token_id) is not int for token_id in token_ids[:2])
        or (token_ids.pad_token_id is not None and type(token_ids.pad_token_id) is not int)
        or token_ids not in contract.accepted_source_token_ids
    ):
        raise ValueError(
            f"{contract.family.value} source config mismatch for BOS/EOS/PAD token ids: "
            f"expected one of {contract.accepted_source_token_ids!r}, got {token_ids!r}"
        )
    return contract


def _source_model_id_for_contract(model_id: str) -> str | None:
    """Recover a repository export's pinned Hub identity when available."""
    source = Path(model_id).expanduser()
    if not source.is_dir():
        return model_id
    metadata_path = source / _SFT_METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid repository HF metadata at {metadata_path}: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f"repository HF metadata must be an object: {metadata_path}")
    raw_source = metadata.get("source")
    if raw_source is None:
        # Compatibility for the minimal staged exports used by the chaining
        # regression: their monkeypatched family contract owns the geometry.
        return None
    if not isinstance(raw_source, Mapping):
        raise ValueError(f"repository HF metadata source must be an object: {metadata_path}")
    source_model_id = raw_source.get("model_id")
    if not isinstance(source_model_id, str) or not source_model_id.strip():
        raise ValueError(
            f"repository HF metadata source.model_id must be nonempty: {metadata_path}"
        )
    return source_model_id


def validate_hf_source_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    contract: HFDecoderContract,
) -> None:
    """Require the source vocabulary and structural token identities."""
    if len(tokenizer) != contract.tokenizer_vocab_size:
        raise ValueError(
            f"{contract.family.value} tokenizer size must be "
            f"{contract.tokenizer_vocab_size}; got {len(tokenizer)}"
        )
    for token, expected_id in contract.required_token_ids.items():
        raw_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError(f"token {token!r} must resolve to one integer id; got {raw_id!r}")
        if raw_id != expected_id:
            raise ValueError(f"token {token!r} must have id {expected_id}; got {raw_id}")


class HFSFTTokenizer:
    """Nanochat-compatible adapter around one validated HF chat tokenizer."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        contract: HFDecoderContract,
    ) -> None:
        validate_hf_source_tokenizer(tokenizer, contract)
        self.tokenizer = tokenizer
        self.contract = contract
        original_size = len(tokenizer)
        if contract.prepare_tokenizer is not None:
            contract.prepare_tokenizer(tokenizer)
        tokenizer.bos_token = contract.bos_token
        tokenizer.eos_token = contract.eos_token
        tokenizer.pad_token = contract.pad_token
        tokenizer.chat_template = contract.chat_template
        if len(tokenizer) != original_size:
            raise RuntimeError("installing the HF SFT contract changed vocabulary size")

    def get_bos_token_id(self) -> int:
        return self.contract.generation_bos_token_id

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode expects str, got {type(text).__name__}")
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        decoded = self.tokenizer.decode(list(ids), skip_special_tokens=False)
        if not isinstance(decoded, str):
            raise RuntimeError("single-sequence HF decode returned multiple strings")
        return decoded

    @staticmethod
    def _validated_messages(conversation: Any) -> list[dict[str, Any]]:
        if not isinstance(conversation, Mapping):
            raise ValueError("conversation must be a mapping with a messages list")
        raw_messages = conversation.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("conversation.messages must be a nonempty list")
        for index, message in enumerate(raw_messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"message {index} must be a mapping")
        messages = cast(list[dict[str, Any]], raw_messages)
        first_conversation_index = 0
        if messages[0].get("role") == "system":
            if len(messages) < 2 or messages[1].get("role") != "user":
                raise ValueError("System message must be immediately followed by a user message")
            if not isinstance(messages[0].get("content"), str):
                raise ValueError("System message content must be a string")
            first_conversation_index = 1
        for index, message in enumerate(messages[first_conversation_index:]):
            expected_role = "user" if index % 2 == 0 else "assistant"
            role = message.get("role")
            if role != expected_role:
                raise ValueError(f"message {index} has role {role!r}; expected {expected_role!r}")
            content = message.get("content")
            if role == "user":
                if not isinstance(content, str):
                    raise ValueError("User message content must be a string")
                continue
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise ValueError("Assistant content must be a string or list of parts")
            for part_index, part in enumerate(content):
                if not isinstance(part, Mapping):
                    raise ValueError(
                        f"assistant part {part_index} in message {index} must be a mapping"
                    )
                part_type = part.get("type")
                if part_type not in {"text", "python", "python_output"}:
                    raise ValueError(f"Unknown assistant part type: {part_type!r}")
                if not isinstance(part.get("text"), str):
                    raise ValueError(
                        f"assistant part {part_index} in message {index} must have string text"
                    )
        return messages

    def render_conversation(
        self,
        conversation: Any,
        max_tokens: int | None = None,
    ) -> tuple[list[int], list[int]]:
        messages = self._validated_messages(conversation)
        rendered = cast(
            Mapping[str, Any],
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            ),
        )
        ids = [int(value) for value in rendered["input_ids"]]
        mask = [int(value) for value in rendered["assistant_masks"]]
        if max_tokens is not None:
            ids = ids[:max_tokens]
            mask = mask[:max_tokens]
        if len(ids) != len(mask):
            raise RuntimeError("HF chat template returned a misaligned assistant mask")
        return ids, mask

    def render_for_completion(self, conversation: Any) -> list[int]:
        messages = self._validated_messages(conversation)
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must be from the assistant")
        return cast(
            list[int],
            self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            ),
        )

    def save(self, tokenizer_dir: str | Path) -> None:
        self.tokenizer.save_pretrained(tokenizer_dir)


class HFDecoderModelAdapter(t.nn.Module):
    """Expose a Transformers causal LM through the nanochat decoder loss API."""

    def __init__(self, model: PreTrainedModel, *, max_seq_len: int) -> None:
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: t.Tensor,
        targets: t.Tensor | None = None,
        loss_reduction: str = "mean",
    ) -> t.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits.float()
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

    def get_device(self) -> t.device:
        return next(self.model.parameters()).device

    def estimate_flops(self) -> float:
        """Approximate training FLOPs/token from matrix weights + full attention."""
        matrix_parameters = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.ndim >= 2
        )
        config = self.model.config
        head_dim = config.hidden_size // config.num_attention_heads
        attention = (
            12 * config.num_hidden_layers * config.num_attention_heads * head_dim * self.max_seq_len
        )
        return float(6 * matrix_parameters + attention)


def load_hf_config_and_tokenizer(
    *,
    model_id: str,
    revision: str,
) -> tuple[PretrainedConfig, HFSFTTokenizer, HFDecoderContract]:
    """Run cheap immutable source checks before any weight-shard loading."""
    local_source = Path(model_id).expanduser().is_dir()
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    contract = validate_hf_source_config(
        config,
        model_id=_source_model_id_for_contract(model_id),
        local_source=local_source,
    )
    raw_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        ),
    )
    return config, HFSFTTokenizer(raw_tokenizer, contract), contract


def validated_local_world_size(
    *,
    local_rank: int,
    device: str | t.device,
) -> int:
    """Return strict local-rank geometry used by restart and model loading."""
    local_world_size: int | None = None
    with rank_synchronized_stage("HF local-world geometry", device=device):
        raw = os.environ.get("LOCAL_WORLD_SIZE")
        if raw is None:
            if dist.is_initialized() and dist.get_world_size() > 1:
                raise ValueError("distributed HF loading requires LOCAL_WORLD_SIZE")
            local_world_size = 1
        else:
            try:
                local_world_size = int(raw)
            except ValueError as exc:
                raise ValueError(f"LOCAL_WORLD_SIZE must be an integer; got {raw!r}") from exc
        if local_world_size <= 0 or not 0 <= local_rank < local_world_size:
            raise ValueError(
                f"invalid local rank geometry: LOCAL_RANK={local_rank}, "
                f"LOCAL_WORLD_SIZE={local_world_size}"
            )
        raw_local_rank = os.environ.get("LOCAL_RANK")
        if raw_local_rank is not None:
            try:
                environment_local_rank = int(raw_local_rank)
            except ValueError as exc:
                raise ValueError(f"LOCAL_RANK must be an integer; got {raw_local_rank!r}") from exc
            if environment_local_rank != local_rank:
                raise ValueError(
                    f"compute local_rank={local_rank} disagrees with "
                    f"LOCAL_RANK={environment_local_rank}"
                )
    assert local_world_size is not None
    if dist.is_initialized() and dist.get_world_size() > 1:
        bounds = t.tensor([local_world_size, local_world_size], dtype=t.int32, device=device)
        dist.all_reduce(bounds[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(bounds[1:], op=dist.ReduceOp.MAX)
        if int(bounds[0].item()) != int(bounds[1].item()):
            raise ValueError("LOCAL_WORLD_SIZE differs across distributed ranks")
        if dist.get_world_size() % local_world_size != 0:
            raise ValueError(
                f"WORLD_SIZE={dist.get_world_size()} must be divisible by "
                f"LOCAL_WORLD_SIZE={local_world_size}"
            )
    return local_world_size


def load_hf_model_in_local_rank_slots(
    *,
    model_id: str,
    revision: str,
    contract: HFDecoderContract,
    device: str | t.device,
    local_rank: int,
    gradient_checkpointing: bool,
    attention_implementation: HFAttentionImplementation | str = HF_ATTENTION_IMPLEMENTATION,
) -> PreTrainedModel:
    """Materialize at most one FP32 CPU model per node at a time."""
    try:
        attention = HFAttentionImplementation(attention_implementation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported HF attention implementation {attention_implementation!r}"
        ) from exc
    local_world_size = validated_local_world_size(local_rank=local_rank, device=device)

    model: PreTrainedModel | None = None
    for slot in range(local_world_size):
        with rank_synchronized_stage(f"HF source load local-rank slot {slot}", device=device):
            if local_rank == slot:
                loaded = cast(
                    PreTrainedModel,
                    AutoModelForCausalLM.from_pretrained(
                        model_id,
                        revision=revision,
                        trust_remote_code=False,
                        dtype=t.float32,
                        low_cpu_mem_usage=True,
                        attn_implementation=attention.value,
                    ),
                )
                if not isinstance(loaded, contract.model_class):
                    raise TypeError(
                        f"source must load as {contract.architecture}; got {type(loaded).__name__}"
                    )
                loaded.config.use_cache = False
                if gradient_checkpointing:
                    loaded.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                t.nn.Module.to(loaded, device=device, dtype=t.float32)
                loaded.train()
                model = loaded
                gc.collect()
    if model is None:
        raise RuntimeError("HF source loading completed without a local model")
    with rank_synchronized_stage("HF loaded-model validation", device=device):
        validate_loaded_hf_model(model, contract)
    return model


def validate_loaded_hf_model(
    model: PreTrainedModel,
    contract: HFDecoderContract,
) -> None:
    """Verify architecture, FP32 training, and embedding ownership."""
    if not isinstance(model, contract.model_class):
        raise TypeError(f"expected {contract.architecture}, got {type(model).__name__}")
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if (input_weight is output_weight) != contract.tied_embeddings:
        expected = "tied" if contract.tied_embeddings else "untied"
        raise ValueError(f"{contract.family.value} input/output embeddings must be {expected}")
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.dtype != t.float32
    ]
    if bad:
        raise ValueError(f"all trainable HF parameters must be FP32; first mismatch: {bad[0]}")


def build_hf_optimizer(
    model: t.nn.Module,
    *,
    ddp: bool,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    init_lr_frac: float,
    distributed_implementation: HFDistributedImplementation | str = (
        HFDistributedImplementation.DIST_MUON_ADAMW
    ),
) -> t.optim.Optimizer:
    """Build the repository optimizer with exactly one AdamW parameter group."""
    try:
        implementation = HFDistributedImplementation(distributed_implementation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported HF distributed implementation {distributed_implementation!r}"
        ) from exc
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("HF decoder has no trainable parameters")
    group = {
        "kind": "adamw",
        "params": parameters,
        "lr": learning_rate * init_lr_frac,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
    }
    if implementation is HFDistributedImplementation.DIST_MUON_ADAMW:
        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer: t.optim.Optimizer = factory([group])
    else:
        optimizer = t.optim.AdamW([group], fused=parameters[0].is_cuda)
    optimizer.param_groups[0]["initial_lr"] = optimizer.param_groups[0]["lr"]
    return optimizer


def prepare_hf_training_model(
    model: t.nn.Module,
    *,
    ddp: bool,
    local_rank: int,
    distributed_implementation: HFDistributedImplementation | str,
    torch_compile: bool,
) -> t.nn.Module:
    """Apply optional compile and DDP without changing raw-model ownership."""
    implementation = HFDistributedImplementation(distributed_implementation)
    training_model: t.nn.Module = model
    if torch_compile:
        training_model = cast(t.nn.Module, t.compile(training_model, dynamic=None))
    if ddp and implementation is HFDistributedImplementation.DDP_FUSED_ADAMW:
        training_model = t.nn.parallel.DistributedDataParallel(
            training_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
    return training_model


@dataclass(frozen=True)
class GlobalTargetCounts:
    """Global target counts for one optimizer update."""

    nonpadding: int
    supervised: int


def reduce_global_target_counts(
    *,
    local_nonpadding: int,
    local_supervised: int,
    device: str | t.device,
) -> GlobalTargetCounts:
    """Sum exact non-padding and supervised target counts across ranks."""
    if local_nonpadding < 0 or local_supervised < 0:
        raise ValueError("local target counts must be non-negative")
    if local_supervised > local_nonpadding:
        raise ValueError(
            f"supervised targets ({local_supervised}) exceed non-padding targets "
            f"({local_nonpadding})"
        )
    counts = t.tensor(
        [local_nonpadding, local_supervised],
        dtype=t.int64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    nonpadding, supervised = (int(value) for value in counts.tolist())
    if supervised <= 0:
        raise RuntimeError("decoder SFT optimizer step has zero global supervised targets")
    return GlobalTargetCounts(nonpadding=nonpadding, supervised=supervised)


def distributed_loss_backward_scale(global_supervised: int, *, world_size: int) -> float:
    """Scale a rank-local summed loss before an averaging gradient reduction."""
    if global_supervised <= 0:
        raise ValueError(f"global_supervised must be positive, got {global_supervised}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return world_size / global_supervised


def reduce_global_loss_mean(
    local_loss_sum: t.Tensor,
    *,
    global_supervised: int,
) -> float:
    """Return the detached global summed-loss numerator divided by its token count."""
    if local_loss_sum.numel() != 1:
        raise ValueError("local_loss_sum must be a scalar tensor")
    if global_supervised <= 0:
        raise ValueError(f"global_supervised must be positive, got {global_supervised}")
    numerator = local_loss_sum.detach().to(dtype=t.float64)
    if dist.is_initialized():
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
    mean = float((numerator / global_supervised).item())
    if not math.isfinite(mean):
        raise FloatingPointError(
            f"decoder SFT global supervised-token loss is non-finite: {mean!r}"
        )
    return mean


def resolve_hf_output_root(
    *,
    base_dir: str | Path,
    model_id: str,
    revision: str,
    output_dir: str | Path | None,
) -> Path:
    """Resolve the model-only checkpoint root without creating it."""
    if output_dir is not None:
        return Path(output_dir)
    escaped_model_id = model_id.replace("/", "--")
    return (
        Path(base_dir) / "hf_checkpoints" / escaped_model_id / revision / "standard_sft_checkpoints"
    )


def configure_generation_contract(
    model: PreTrainedModel,
    contract: HFDecoderContract,
) -> None:
    """Install exported instruct BOS/EOS/PAD identities without resizing."""
    model.config.bos_token_id = contract.generation_bos_token_id
    model.config.eos_token_id = contract.generation_eos_token_id
    model.config.pad_token_id = contract.generation_pad_token_id
    model.generation_config.bos_token_id = contract.generation_bos_token_id
    model.generation_config.eos_token_id = contract.generation_eos_token_id
    model.generation_config.pad_token_id = contract.generation_pad_token_id
    model.generation_config.do_sample = False
