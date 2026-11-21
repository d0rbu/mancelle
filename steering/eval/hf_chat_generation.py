"""Strict local repository-HF loading and harmfulness chat generation."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype import BeartypeConf, beartype
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
    string_keyed_mapping,
)
from steering.eval.harmfulness_generation import (
    MAX_TORCH_SEED,
    capture_file_inventory,
    recursive_file_paths,
)
from steering.immutable_io import ImmutableArtifactError, stream_hash_file_descriptor_bound
from steering.refined_types import (
    NonNegativeFiniteNumberLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.sft.hf_contract import validate_repository_hf_sft_metadata

_STEP_DIR_RE = re.compile(r"step_([0-9]{6})")
_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_VALUE_ERROR_CONF = BeartypeConf(violation_param_type=ValueError)


@dataclass(frozen=True)
class RepositoryHFCheckpoint:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_identity: dict[str, Any]
    source_model_id: str
    source_revision: str


class _BatchedSamplingLogitsProcessor(LogitsProcessor):
    """Sample the complete score matrix from one outer-batch RNG stream.

    Selecting the tokens here and returning one-hot score rows lets the common
    greedy decode loop retain batched model forwards. One matrix-wide
    ``multinomial`` avoids a Python loop and one sampling launch per prompt.
    The RNG trajectory intentionally depends on outer-batch partitioning.
    """

    def __init__(
        self,
        *,
        device: t.device,
        seed: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        self._generator = t.Generator(device=device).manual_seed(seed)
        self._temperature = TemperatureLogitsWarper(temperature)
        self._top_k = (
            None if top_k is None else TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)
        )

    def __call__(self, input_ids: t.LongTensor, scores: t.FloatTensor) -> t.FloatTensor:
        warped_scores = self._temperature(input_ids, scores)
        if self._top_k is not None:
            warped_scores = self._top_k(input_ids, warped_scores)
        probabilities = t.nn.functional.softmax(warped_scores, dim=-1)

        selected = t.multinomial(probabilities, num_samples=1, generator=self._generator)
        forced_scores = t.full_like(scores, -t.inf)
        return cast(t.FloatTensor, forced_scores.scatter_(1, selected, 0.0))


def _read_strict_json(path: Path, *, surface: str) -> dict[str, Any]:
    try:
        digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
    except (OSError, ImmutableArtifactError) as exc:
        raise ValueError(f"{surface} is missing or unreadable: {path}") from exc
    assert digest.payload is not None
    try:
        text = digest.payload.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=strict_json_object,
            parse_constant=reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{surface} must be strict UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{surface} must contain a JSON object: {path}")
    return payload


def _token_ids(owner: Any, *, path: str) -> tuple[int, int, int]:
    ids = tuple(getattr(owner, f"{name}_token_id", None) for name in ("bos", "eos", "pad"))
    if not all(isinstance(token_id, NonNegativeInt) for token_id in ids):
        raise ValueError(f"{path} must define non-negative integer BOS/EOS/PAD token ids")
    return cast(tuple[int, int, int], ids)


def _validate_weight_inventory(root: Path, paths: Sequence[Path]) -> None:
    forbidden = [
        path.relative_to(root).as_posix()
        for path in paths
        if path.suffix.lower() in {".bin", ".pt", ".pth"}
    ]
    if forbidden:
        raise ValueError(
            "repository HF checkpoints must be safetensors-only; found pickle-based "
            f"weights/files: {forbidden}"
        )
    model_files = sorted(
        path
        for path in paths
        if path.parent == root and path.name.startswith("model") and path.suffix == ".safetensors"
    )
    if not model_files:
        raise ValueError("repository HF checkpoint contains no safetensors model files")
    index_path = root / "model.safetensors.index.json"
    if len(model_files) > 1 and not index_path.is_file():
        raise ValueError("sharded safetensors checkpoint is missing model.safetensors.index.json")
    if index_path.exists():
        index = _read_strict_json(index_path, surface="safetensors shard index")
        weight_map = string_keyed_mapping(index.get("weight_map"))
        if not weight_map:
            raise ValueError("safetensors shard index.weight_map must be a nonempty mapping")
        referenced_values = list(weight_map.values())
        if not all(
            isinstance(name, str) and Path(name).name == name for name in referenced_values
        ):
            raise ValueError("safetensors shard index filenames must be top-level basenames")
        referenced = {name for name in referenced_values if isinstance(name, str)}
        actual = {path.name for path in model_files}
        if referenced != actual:
            raise ValueError(
                "safetensors shard index files do not exactly match the model inventory: "
                f"referenced={sorted(referenced)}, actual={sorted(actual)}"
            )


def resolve_repository_hf_checkpoint_dir(path: str | Path) -> Path:
    """Resolve one concrete local repository export without loading weights."""

    supplied = Path(path).expanduser()
    try:
        supplied_info = supplied.lstat()
    except OSError as exc:
        raise ValueError(f"HF checkpoint directory is missing or unreadable: {supplied}") from exc
    if stat.S_ISLNK(supplied_info.st_mode):
        raise ValueError(f"HF checkpoint directory must not be a symlink: {supplied}")
    if not stat.S_ISDIR(supplied_info.st_mode):
        raise ValueError(f"HF checkpoint path must be a directory: {supplied}")
    resolved = supplied.resolve(strict=True)
    match = _STEP_DIR_RE.fullmatch(resolved.name)
    if match is None:
        raise ValueError(
            "hf_checkpoint_dir must name one concrete immutable step_<six digits> directory; "
            f"got {resolved}"
        )
    return resolved


def load_repository_hf_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | t.device,
) -> RepositoryHFCheckpoint:
    """Validate, content-bind, and locally load one repository HF SFT export."""

    resolved = resolve_repository_hf_checkpoint_dir(checkpoint_dir)
    directory_step = int(resolved.name.removeprefix("step_"))
    selected_device = t.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "repository HF harmfulness generation supports only CPU or CUDA devices, "
            f"got {selected_device}"
        )

    initial_paths = recursive_file_paths(resolved, exclude_top_level={"report"})
    _validate_weight_inventory(resolved, initial_paths)
    inventory = capture_file_inventory(
        resolved,
        initial_paths,
        domain="harmfulness_repository_hf_checkpoint_v1",
    )
    tokenizer_inventory = capture_file_inventory(
        resolved,
        [
            path
            for path in initial_paths
            if path.name in _TOKENIZER_FILENAMES
            or path.name.startswith("tokenizer.")
            or "chat_template" in path.name
        ],
        domain="harmfulness_repository_hf_tokenizer_v1",
    )
    _read_strict_json(resolved / "config.json", surface="config.json")
    _read_strict_json(
        resolved / "generation_config.json",
        surface="generation_config.json",
    )
    metadata = validate_repository_hf_sft_metadata(
        _read_strict_json(resolved / "sft_metadata.json", surface="sft_metadata.json")
    )
    completed_step = metadata["completed_optimizer_step"]
    if completed_step != directory_step:
        raise ValueError(
            "HF checkpoint directory step does not match sft_metadata.completed_optimizer_step: "
            f"directory={directory_step}, metadata={completed_step}"
        )
    load_kwargs = {
        "trust_remote_code": False,
        "local_files_only": True,
    }
    config = AutoConfig.from_pretrained(resolved, **load_kwargs)
    exported_generation_config = GenerationConfig.from_pretrained(
        resolved,
        trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(resolved, **load_kwargs),
    )
    ids = _token_ids(config, path="config")
    generation_ids = _token_ids(exported_generation_config, path="generation_config")
    tokenizer_ids = _token_ids(tokenizer, path="tokenizer")
    if ids != generation_ids or ids != tokenizer_ids:
        raise ValueError(
            "HF checkpoint BOS/EOS/PAD token identities disagree: "
            f"config={ids}, generation_config={generation_ids}, tokenizer={tokenizer_ids}"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise ValueError("repository HF tokenizer must contain a nonempty saved chat template")
    actual_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if actual_template_sha256 != metadata["chat_template_sha256"]:
        raise ValueError(
            "repository HF tokenizer chat template does not match sft_metadata.chat_template_sha256"
        )
    if not isinstance(getattr(config, "max_position_embeddings", None), PositiveInt):
        raise ValueError("repository HF config.max_position_embeddings must be a positive integer")

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            resolved,
            config=config,
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=t.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ),
    )
    t.nn.Module.to(model, device=selected_device)
    model.eval()

    output = cast(Mapping[str, Any], metadata["output"])
    configured_architectures = list(getattr(config, "architectures", None) or ())
    if type(model).__name__ != output["architecture"] or configured_architectures != [
        output["architecture"]
    ]:
        raise ValueError(
            "loaded HF model class/config architectures disagree with metadata: "
            f"loaded={type(model).__name__!r}, config={configured_architectures!r}, "
            f"metadata={output['architecture']!r}"
        )
    if (
        str(config.model_type) != output["model_type"]
        or str(model.config.model_type) != output["model_type"]
    ):
        raise ValueError(
            "loaded HF model/config model_type disagree with metadata: "
            f"loaded={model.config.model_type!r}, config={config.model_type!r}, "
            f"metadata={output['model_type']!r}"
        )
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {t.bfloat16}:
        raise ValueError(
            f"loaded repository HF floating parameter dtypes must be bfloat16, got {floating_dtypes}"
        )
    vocabulary = cast(Mapping[str, int], metadata["vocabulary"])
    model_vocab_size = int(vocabulary["model"])
    tokenizer_vocab_size = int(vocabulary["tokenizer"])
    config_vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(config_vocab_size, PositiveInt):
        raise ValueError("HF config.vocab_size must be a positive integer")
    if (
        config_vocab_size != model_vocab_size
        or int(model.config.vocab_size) != model_vocab_size
        or len(tokenizer) != tokenizer_vocab_size
    ):
        raise ValueError(
            "HF model/config/tokenizer vocabulary sizes disagree: "
            f"model={model.config.vocab_size}, config={config_vocab_size}, "
            f"tokenizer={len(tokenizer)}, metadata={dict(vocabulary)}"
        )
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    embedding_weights = (
        getattr(input_embeddings, "weight", None),
        getattr(output_embeddings, "weight", None),
    )
    if any(weight is None for weight in embedding_weights) or {
        int(cast(t.Tensor, weight).shape[0]) for weight in embedding_weights
    } != {model_vocab_size}:
        raise ValueError("HF input/output embedding vocabulary sizes disagree with config")
    if _token_ids(model.config, path="loaded model config") != ids:
        raise ValueError("loaded HF model token identities disagree with preflight config")

    inventory.assert_unchanged(recursive_file_paths(resolved, exclude_top_level={"report"}))
    model_identity = {
        "format": "repository_hf_sft",
        "backend": metadata["family"],
        "architecture": output["architecture"],
        "model_type": output["model_type"],
        "step": completed_step,
        "checkpoint_sha256": inventory.sha256,
        "tokenizer_sha256": tokenizer_inventory.sha256,
    }
    return RepositoryHFCheckpoint(
        model=model,
        tokenizer=tokenizer,
        model_identity=model_identity,
        source_model_id=str(metadata["source"]["model_id"]),
        source_revision=str(metadata["source"]["revision"]),
    )


class HuggingFaceChatGenerationEngine:
    """The minimal ``generate_batch`` surface harmfulness evaluation needs."""

    def __init__(self, checkpoint: RepositoryHFCheckpoint) -> None:
        self._model = checkpoint.model
        self._tokenizer = checkpoint.tokenizer
        self._device = next(checkpoint.model.parameters()).device
        self._max_position_embeddings = int(checkpoint.model.config.max_position_embeddings)
        self._bos_id, self._eos_id, self._pad_id = _token_ids(
            checkpoint.model.config,
            path="config",
        )

    def _render_prompt(self, prompt: str, *, prompt_index: int) -> t.Tensor:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt {prompt_index} must be a string, got {type(prompt).__name__}")
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        raw_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
        if not isinstance(raw_ids, t.Tensor):
            raise ValueError(
                "HF chat template must return a tensor or mapping containing input_ids"
            )
        if raw_ids.ndim == 1:
            raw_ids = raw_ids.unsqueeze(0)
        if raw_ids.ndim != 2 or raw_ids.shape[0] != 1 or raw_ids.shape[1] == 0:
            raise ValueError(
                "HF chat template must return exactly one nonempty token sequence; "
                f"prompt_index={prompt_index}, shape={tuple(raw_ids.shape)}"
            )
        if raw_ids.dtype == t.bool or raw_ids.is_floating_point() or raw_ids.is_complex():
            raise ValueError(
                f"HF chat template input_ids must use an integer tensor dtype, got {raw_ids.dtype}"
            )
        return raw_ids.to(device=self._device, dtype=t.long)

    def _decode_continuation(self, continuation_tensor: t.Tensor) -> str:
        continuation = [int(value) for value in continuation_tensor.tolist()]
        for index, token_id in enumerate(continuation):
            if token_id in {self._eos_id, self._bos_id}:
                continuation = continuation[:index]
                break
        decoded = self._tokenizer.decode(
            continuation,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise ValueError("HF tokenizer.decode must return one string")
        return decoded

    def _build_left_padded_batch(
        self,
        prompt_ids: list[t.Tensor],
    ) -> tuple[t.Tensor, t.Tensor]:
        max_prompt_tokens = max(int(ids.shape[1]) for ids in prompt_ids)
        input_ids = t.full(
            (len(prompt_ids), max_prompt_tokens),
            self._pad_id,
            dtype=t.long,
            device=self._device,
        )
        attention_mask = t.zeros_like(input_ids)
        for row, ids in enumerate(prompt_ids):
            prompt_tokens = int(ids.shape[1])
            input_ids[row, -prompt_tokens:] = ids[0]
            attention_mask[row, -prompt_tokens:] = 1
        return input_ids, attention_mask

    @beartype(conf=_VALUE_ERROR_CONF)
    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: NonNegativeIntLike = 256,
        temperature: NonNegativeFiniteNumberLike = 0.7,
        top_k: PositiveIntLike | None = None,
        seed: NonNegativeIntLike | None = None,
        render_as_chat: bool = False,
    ) -> list[str]:
        if render_as_chat is not True:
            raise ValueError("HF harmfulness generation requires render_as_chat=True")
        if not prompts:
            return []
        temperature = float(temperature)
        if temperature == 0.0 and top_k is not None:
            raise ValueError("greedy HF generation rejects non-null top_k")
        if temperature > 0.0 and seed is None:
            raise ValueError("seed must be a non-negative integer when temperature > 0")
        if seed is not None and seed > MAX_TORCH_SEED:
            raise ValueError("seed exceeds PyTorch's valid range")

        prompt_ids = [
            self._render_prompt(prompt, prompt_index=index)
            for index, prompt in enumerate(prompts)
        ]
        for prompt_index, ids in enumerate(prompt_ids):
            prompt_tokens = int(ids.shape[1])
            if prompt_tokens + max_new_tokens > self._max_position_embeddings:
                raise ValueError(
                    "HF harmfulness prompt exceeds the architecture context window: "
                    f"prompt_index={prompt_index}, prompt_tokens={prompt_tokens}, "
                    f"max_new_tokens={max_new_tokens}, "
                    f"max_position_embeddings={self._max_position_embeddings}"
                )
        if max_new_tokens == 0:
            return [""] * len(prompts)

        input_ids, attention_mask = self._build_left_padded_batch(prompt_ids)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            bos_token_id=self._bos_id,
            eos_token_id=[self._eos_id, self._bos_id],
            pad_token_id=self._pad_id,
        )
        logits_processor = None
        if temperature > 0.0:
            assert seed is not None
            logits_processor = LogitsProcessorList(
                [
                    _BatchedSamplingLogitsProcessor(
                        device=self._device,
                        seed=seed,
                        temperature=temperature,
                        top_k=top_k,
                    )
                ]
            )

        with t.inference_mode():
            generated = cast(Any, self._model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                logits_processor=logits_processor,
            )

        prompt_width = int(input_ids.shape[1])
        if not isinstance(generated, t.Tensor) or generated.ndim != 2:
            raise ValueError(
                "HF generate() must return a rank-2 tensor; "
                f"got {type(generated).__name__} shape={getattr(generated, 'shape', None)}"
            )
        if generated.shape[0] != len(prompts):
            raise ValueError(
                f"HF generate() returned {generated.shape[0]} rows for {len(prompts)} prompts"
            )
        if generated.shape[1] < prompt_width or not t.equal(
            generated[:, :prompt_width],
            input_ids,
        ):
            raise ValueError(
                "HF generate() did not preserve the complete left-padded prompt prefix batch"
            )
        return [self._decode_continuation(row[prompt_width:]) for row in generated]
