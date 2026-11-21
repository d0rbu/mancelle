"""Token-exact GCG model math for the persistent vendored AxBench worker."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib.util import find_spec
from itertools import batched, repeat
from operator import is_
from pathlib import Path
from typing import Any

import axbench
import pandas as pd
import torch as t
import torch.nn.functional as F
from axbench.utils.model_utils import decoder_layers
from pyvene import CollectIntervention, IntervenableConfig, IntervenableModel
from safetensors.torch import save_file
from tokenizers.decoders import ByteLevel
from torch.nn.modules import module as module_hooks
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import is_in_torch_dispatch_mode
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, masking_utils
from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM, SmolLM3Model

from scripts.eval.axbench_adv.gcg_math import (
    activation_l2_pair,
    activation_l2_row,
    response_margin_gradient,
    response_margin_value,
    weighted_bpb,
)
from scripts.eval.axbench_adv.gcg_runtime_state import preserve_gcg_model_state
from scripts.eval.axbench_adv.soft_prompt_tensors import (
    load_soft_embeddings,
    validate_soft_embeddings,
)
from scripts.eval.axbench_adv.vendored.gcg_mlp_scoring import fused_scoring_mlps
from scripts.eval.axbench_adv.vendored.prompt_recovery import (
    QUESTION_MARKER,
    PromptWrapper,
    recover_concept_steering_prompts,
    rendered_prompt_wrapper_from_manifest,
)

if find_spec("peft") is not None:
    from peft import PeftModelForCausalLM
    from peft.tuners.lora.model import LoraModel
else:
    # Root tests isolate the worker from the vendored environment's dependencies.
    PeftModelForCausalLM = None
    LoraModel = None

_OBJECTIVES = frozenset({"response_margin", "activation_l2"})
_ADDITIVE_METHODS = frozenset({"DiffMean", "LinearProbe"})
_HF_DECODE = PreTrainedTokenizerBase.decode
_INPUT_FORWARDS = {
    kind: getattr(kind, "forward", None)
    for kind in (
        SmolLM3ForCausalLM,
        SmolLM3Model,
        t.nn.Embedding,
        IntervenableModel,
        PeftModelForCausalLM,
        LoraModel,
    )
    if kind is not None
}
_INPUT_EMBEDDINGS = SmolLM3ForCausalLM.get_input_embeddings
_PEFT_INPUT_HOOKS = {
    kind: kind._enable_peft_forward_hooks
    for kind in (PeftModelForCausalLM, LoraModel)
    if kind is not None
}
_INPUT_MODEL_GLOBALS = inspect.unwrap(SmolLM3Model.forward).__globals__
_CREATE_CAUSAL_MASK = masking_utils.create_causal_mask
_SDPA_MASK = masking_utils.sdpa_mask
_EXPLICIT_CAUSAL_MASK = "allow_is_causal_skip" in inspect.signature(_CREATE_CAUSAL_MASK).parameters


def _native_input_module(module: t.nn.Module) -> bool:
    forward = _INPUT_FORWARDS.get(type(module))
    return (
        forward is not None
        and "forward" not in module.__dict__
        and getattr(module.forward, "__func__", None) is forward
        and module._compiled_call_impl is None
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and not module._backward_hooks
        and not module._backward_pre_hooks
    )


def _can_stage_scoring_inputs(runtime: Any, sequences: list[tuple[int, ...]]) -> bool:
    """Recognize the native forward path for a necessarily padded causal mask."""
    if (
        not _EXPLICIT_CAUSAL_MASK
        or t.is_grad_enabled()
        or t.is_autocast_enabled()
        or t.jit.is_tracing()
        or t.compiler.is_compiling()
        or _get_current_function_mode() is not None
        or is_in_torch_dispatch_mode()
        or t.device(runtime.device).type != "cuda"
        or module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_backward_hooks
        or module_hooks._global_backward_pre_hooks
        or masking_utils.create_causal_mask is not _CREATE_CAUSAL_MASK
        or _INPUT_MODEL_GLOBALS.get("create_causal_mask") is not _CREATE_CAUSAL_MASK
        or masking_utils.ALL_MASK_ATTENTION_FUNCTIONS.get("sdpa") is not _SDPA_MASK
    ):
        return False
    lengths = list(map(len, sequences))
    if not lengths or min(lengths) <= 0 or min(lengths) == max(lengths):
        return False
    model = runtime.benchmark_model.model
    backbone = getattr(model, "model", None)
    if type(model) is not SmolLM3ForCausalLM or type(backbone) is not SmolLM3Model:
        return False
    if (
        model.config is not backbone.config
        or model.config._attn_implementation != "sdpa"
        or not getattr(model.config, "is_causal", True)
        or backbone.has_sliding_layers
        or model.config.sliding_window is not None
        or "get_input_embeddings" in model.__dict__
        or getattr(model.get_input_embeddings, "__func__", None) is not _INPUT_EMBEDDINGS
    ):
        return False
    embedding = model.get_input_embeddings()
    if type(embedding) is not t.nn.Embedding or type(embedding.weight) is not t.nn.Parameter:
        return False
    modules = [model, backbone, embedding]
    if runtime.method in _ADDITIVE_METHODS:
        wrapper = runtime.benchmark_model.ax_model
        if type(wrapper) is not IntervenableModel or wrapper.model is not model:
            return False
        modules.append(wrapper)
    elif runtime.method == "LoRA":
        wrapper = runtime.benchmark_model.ax_model
        tuner = getattr(wrapper, "base_model", None)
        if (
            type(wrapper) is not PeftModelForCausalLM
            or type(tuner) is not LoraModel
            or tuner.model is not model
            or wrapper.active_peft_config.is_prompt_learning
        ):
            return False
        for module in (wrapper, tuner):
            if (
                "_enable_peft_forward_hooks" in module.__dict__
                or getattr(module._enable_peft_forward_hooks, "__func__", None)
                is not _PEFT_INPUT_HOOKS[type(module)]
            ):
                return False
        modules.extend((wrapper, tuner))
    elif runtime.method != "PromptSteering":
        return False
    return all(map(_native_input_module, modules))


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}, got {observed}")
    return value


def _text(value: object, *, name: str, nonblank: bool = True) -> str:
    if not isinstance(value, str) or (nonblank and not value.strip()):
        qualifier = "nonblank " if nonblank else ""
        raise ValueError(f"{name} must be a {qualifier}string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{name} must be finite and within its allowed range")
    return parsed


def _write_tensor_sidecar(path: Path, key: str, tensor: t.Tensor) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    save_file({key: tensor}, str(temporary))
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)
    return {"shape": list(tensor.shape), "sha256": _sha256_file(path)}


def _token_ids(value: object, *, name: str, count: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(_integer(token_id, name=f"{name} token", minimum=0) for token_id in value)
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must contain exactly {count} token ids")
    return result


def _ids(encoded: object, *, label: str) -> tuple[int, ...]:
    # Subclasses retain the general path's indexing and iteration behavior.
    if type(encoded) is list and len(encoded) == 1 and type(encoded[0]) is int:
        return (encoded[0],)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, t.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"{label} unexpectedly returned a batch")
        encoded = encoded[0]
    # Keep the original generator for subclasses, including iterator cleanup order.
    if not isinstance(encoded, list) or not (
        all(map(is_, map(type, encoded), repeat(int)))
        if type(encoded) is list
        else all(type(value) is int for value in encoded)
    ):
        raise ValueError(f"{label} returned invalid input_ids")
    return tuple(encoded)


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
    return _ids(
        tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"],
        label="tokenizer",
    )


def _encode_many(
    tokenizer: Any, texts: list[str], *, add_special_tokens: bool
) -> Iterator[tuple[int, ...]]:
    if texts and getattr(tokenizer, "is_fast", False):
        try:
            batch = tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            if not isinstance(batch, list) or len(batch) != len(texts):
                raise ValueError("tokenizer returned an invalid batch")
            encoded = tuple(_ids(row, label="tokenizer") for row in batch)
        except (UnicodeError, ValueError, TypeError, RuntimeError, KeyError):
            # Retry in original row order so the first scalar failure or framing
            # mismatch still determines the candidate's exact rejection.
            pass
        else:
            yield from encoded
            return
    for text in texts:
        yield _encode(tokenizer, text, add_special_tokens=add_special_tokens)


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    value = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return _text(value, name="decoded candidate", nonblank=False)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _token_byte_weights(tokenizer: Any, token_ids: tuple[int, ...]) -> tuple[int, ...]:
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    byte_level = isinstance(getattr(backend, "decoder", None), ByteLevel)
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_ids = (
        {int(value) for value in get_added_vocab().values()}
        if byte_level and callable(get_added_vocab)
        else set()
    )
    weights: list[int] = []
    for token_id in token_ids:
        if token_id in special_ids:
            raise ValueError("raw response must not contain tokenizer special/control ids")
        if byte_level and token_id not in added_ids:
            piece = tokenizer.convert_ids_to_tokens(token_id)
            if not isinstance(piece, str):
                raise ValueError(f"ByteLevel token id {token_id} has no vocabulary piece")
            weight = len(piece)
        else:
            weight = len(_decode(tokenizer, (token_id,)).encode("utf-8"))
        if weight <= 0:
            raise ValueError("raw response contains a zero-byte tokenizer control id")
        weights.append(weight)
    return tuple(weights)


@dataclass(frozen=True)
class _CompletionLayout:
    sample_id: int
    content: str
    clean_prompt_ids: tuple[int, ...]
    fixed_before: tuple[int, ...]
    fixed_after: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_byte_weights: tuple[int, ...]

    @property
    def aligned_token_count(self) -> int:
        return len(self.fixed_after) + len(self.response_ids)


@dataclass(frozen=True)
class _PairLayout:
    pair_id: int
    positive: _CompletionLayout
    negative: _CompletionLayout


@dataclass(frozen=True)
class _SideItem:
    layout: _CompletionLayout
    tokens: tuple[int, ...]


def _completion_layout(
    runtime: Any,
    value: object,
    *,
    wrapper: PromptWrapper,
    steering_prefix: str,
    separator: str,
    seed_text: str,
    seed_ids: tuple[int, ...],
    label: str,
) -> _CompletionLayout:
    fields = _exact_keys(value, {"prompt", "response", "sample_id"}, label=label)
    sample_id = _integer(fields["sample_id"], name=f"{label}.sample_id")
    prompt = _text(fields["prompt"], name=f"{label}.prompt")
    response = _text(fields["response"], name=f"{label}.response")
    content = (
        f"{steering_prefix}{QUESTION_MARKER}{prompt}"
        if runtime.method == "PromptSteering"
        else prompt
    )
    clean_text = f"{wrapper.prefix}{content}{wrapper.suffix}"
    seeded_text = f"{wrapper.prefix}{content}{separator}{seed_text}{wrapper.suffix}"
    clean_ids = _encode(runtime.tokenizer, clean_text, add_special_tokens=True)
    seeded_ids = _encode(runtime.tokenizer, seeded_text, add_special_tokens=True)
    after = _encode(runtime.tokenizer, wrapper.suffix, add_special_tokens=False)
    before = seeded_ids[: -(len(seed_ids) + len(after))]
    # Locate the editable span in the actual prompt: Pythia's separator tokenization
    # changes when a non-whitespace suffix follows it.
    if seeded_ids != before + seed_ids + after:
        raise ValueError("seed text must occupy exactly the requested token span before the tail")
    if len(after) == 0 or len(clean_ids) < len(after) or clean_ids[-len(after) :] != after:
        raise ValueError(f"{label} clean and attacked wrappers do not have an identical fixed tail")
    response_ids = _encode(runtime.tokenizer, response, add_special_tokens=False)
    if not response_ids:
        raise ValueError(f"{label}.response must encode to at least one raw response token")
    byte_weights = _token_byte_weights(runtime.tokenizer, response_ids)
    maximum = int(runtime.worker_max_prompt_tokens)
    total = len(seeded_ids) + len(response_ids)
    clean_total = len(clean_ids) + len(response_ids)
    if total > maximum or clean_total > maximum:
        raise ValueError(
            f"{label} has attacked={total}, clean={clean_total} tokens, exceeding configured "
            f"token limit {maximum}"
        )
    return _CompletionLayout(
        sample_id=sample_id,
        content=content,
        clean_prompt_ids=clean_ids,
        fixed_before=before,
        fixed_after=after,
        response_ids=response_ids,
        response_byte_weights=byte_weights,
    )


def _eligible_token_texts(runtime: Any, embedding_rows: int) -> Iterator[tuple[int, str]]:
    tokenizer = runtime.tokenizer
    excluded = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    added = getattr(tokenizer, "added_tokens_decoder", {})
    if hasattr(added, "keys"):
        excluded.update(int(value) for value in added.keys())
    for token_id in range(embedding_rows):
        if token_id in excluded:
            continue
        try:
            # range supplies plain ints: native decode only converts the list
            # before delegating. Recheck overrides for each token, including
            # changes made while this generator yields to round-trip encoding.
            if (
                type(tokenizer) is PreTrainedTokenizerFast
                and "decode" not in vars(tokenizer)
                and PreTrainedTokenizerFast.decode is _HF_DECODE
            ):
                text = _text(
                    tokenizer._decode(
                        token_ids=[token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    name="decoded candidate",
                    nonblank=False,
                )
            else:
                text = _decode(tokenizer, (token_id,))
            if not text or "\x00" in text:
                continue
            # ASCII is already NFC; only its controls are nonprintable.
            if type(text) is str and text.isascii():
                if not text.isprintable():
                    continue
            elif unicodedata.normalize("NFC", text) != text or any(
                unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in text
            ):
                continue
        except (UnicodeError, ValueError, TypeError, RuntimeError):
            continue
        yield token_id, text


def _allowed_token_ids(runtime: Any, embedding_rows: int) -> tuple[int, ...]:
    tokenizer = runtime.tokenizer
    # Bound temporary tokenizer output while retaining the original scalar path
    # for slow tokenizers. Each failed scalar encoding skips only its own ID.
    chunk_size = 512 if getattr(tokenizer, "is_fast", False) else 1
    allowed: list[int] = []
    for chunk in batched(  # noqa: B911 - Python 3.12; retain the final partial batch.
        _eligible_token_texts(runtime, embedding_rows), chunk_size
    ):
        if chunk_size > 1:
            try:
                encoded = list(
                    _encode_many(
                        tokenizer, [text for _token_id, text in chunk], add_special_tokens=False
                    )
                )
            except (UnicodeError, ValueError, TypeError, RuntimeError):
                pass
            else:
                allowed.extend(
                    token_id
                    for (token_id, _text), token_ids in zip(chunk, encoded, strict=True)
                    if token_ids == (token_id,)
                )
                continue
        for token_id, text in chunk:
            try:
                if _encode(tokenizer, text, add_special_tokens=False) == (token_id,):
                    allowed.append(token_id)
            except (UnicodeError, ValueError, TypeError, RuntimeError):
                continue
    if not allowed:
        raise ValueError("tokenizer has no canonical non-special GCG token ids")
    return tuple(allowed)


def _preserve_runtime(runtime: Any) -> AbstractContextManager[None]:
    base_model = runtime.benchmark_model.model
    intervention = getattr(runtime.benchmark_model, "ax", None)
    roots = (base_model, intervention) if isinstance(intervention, t.nn.Module) else (base_model,)
    return preserve_gcg_model_state(
        roots,
        padding_owner=runtime.tokenizer,
    )


def _subspace(runtime: Any, batch_size: int) -> dict[str, t.Tensor]:
    maximum = runtime.benchmark_model.max_activations.get(runtime.concept_id)
    if maximum is None:
        raise ValueError(f"{runtime.method} has no max activation for GCG forward")
    device = t.device(runtime.device)
    return {
        "idx": t.full((batch_size,), runtime.concept_id, dtype=t.long, device=device),
        "mag": t.full((batch_size,), runtime.factor, dtype=t.float32, device=device),
        "max_act": t.full((batch_size,), float(maximum), dtype=t.float32, device=device),
        "prefix_length": t.zeros(batch_size, dtype=t.long, device=device),
    }


def _forward(
    runtime: Any,
    *,
    embeds: t.Tensor,
    attention_mask: t.Tensor,
    logits_to_keep: int | t.Tensor = 0,
) -> Any:
    kwargs: dict[str, Any] = {"attention_mask": attention_mask, "inputs_embeds": embeds}
    if isinstance(logits_to_keep, t.Tensor) or logits_to_keep:
        kwargs["logits_to_keep"] = logits_to_keep
    if runtime.method in _ADDITIVE_METHODS:
        _base, output = runtime.benchmark_model.ax_model(
            kwargs,
            unit_locations=None,
            subspaces=[_subspace(runtime, embeds.shape[0])],
            use_cache=False,
        )
        return output
    if runtime.method == "LoRA":
        return runtime.benchmark_model.ax_model(**kwargs, use_cache=False)
    return runtime.benchmark_model.model(**kwargs, use_cache=False)


@contextmanager
def _decoder_layer_prefix(model: Any, layer: int) -> Iterator[None]:
    """Skip blocks beyond an activation target, restoring the registered list."""
    kind = getattr(model.config, "model_type", None)
    if kind not in {"llama", "qwen2", "smollm3", "gpt_neox"}:
        yield
        return
    backbone = getattr(model, "gpt_neox" if kind == "gpt_neox" else "model", None)
    layers = getattr(backbone, "layers", None)
    stop = layer + 1
    if backbone is None or not isinstance(layers, t.nn.ModuleList) or not 0 < stop < len(layers):
        yield
        return
    # Keep the original blocks and their parameters/hooks. Only the temporary
    # container changes; model configuration and normal generation stay intact.
    backbone.layers = layers[:stop]
    try:
        yield
    finally:
        backbone.layers = layers


class GCGSession:
    """Prepared GCG state for one persistent AxBench worker cell."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.objective: str | None = None
        self.alpha = 1.0
        self.separator = "\n\n"
        self.device_batch_size = 1
        self.seed_token_ids: tuple[int, ...] = ()
        self.layouts: tuple[_PairLayout, ...] = ()
        self._allowed_token_id_set: frozenset[int] = frozenset()
        self._wrapper: PromptWrapper | None = None
        self._sidecar_path: Path | None = None
        self._direction_offset: t.Tensor | None = None
        self._activation_model: IntervenableModel | None = None
        self._clean_hidden: dict[int, t.Tensor] = {}
        self._response_tensors: dict[tuple[int, t.device], tuple[t.Tensor, t.Tensor]] = {}
        self._soft_prepared = False

    def _require_prepared(self) -> None:
        if self.objective is None:
            raise ValueError("gcg_prepare must succeed before this GCG operation")

    def _steering_prefix(self, wrapper: PromptWrapper) -> str:
        if self.runtime.method != "PromptSteering":
            return ""
        aux_dir = getattr(self.runtime, "_gcg_aux_dump_dir", None)
        if not isinstance(aux_dir, Path):
            raise ValueError("PromptSteering GCG requires the bound auxiliary dump directory")
        frame = pd.read_parquet(aux_dir / "inference" / "steering_data.parquet")
        recovered = recover_concept_steering_prompts(
            frame,
            max_concepts=int(self.runtime.max_concepts),
            wrapper=wrapper,
        )
        return recovered[self.runtime.concept_id]

    def _reset(self) -> None:
        if self._sidecar_path is not None:
            self._sidecar_path.unlink(missing_ok=True)
        self.objective = None
        self.layouts = ()
        self._allowed_token_id_set = frozenset()
        self._wrapper = None
        self.seed_token_ids = ()
        self._sidecar_path = None
        self._direction_offset = None
        self._activation_model = None
        self._clean_hidden.clear()
        self._response_tensors.clear()
        self._soft_prepared = False
        sidecar_dir = getattr(self.runtime, "_gcg_sidecar_dir", None)
        if isinstance(sidecar_dir, Path):
            (sidecar_dir / "soft_embeddings.safetensors").unlink(missing_ok=True)

    def prepare_soft_prompt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # A malformed re-prepare also invalidates the previous soft cell.
        self._reset()
        result = self.prepare(payload)
        weight = self.runtime.benchmark_model.model.get_input_embeddings().weight
        result["identity"] = {
            **result["identity"],
            "embedding_layout": [
                {
                    "name": "input_embeddings",
                    "width": int(weight.shape[1]),
                    "dtype": str(weight.dtype),
                }
            ],
            "generation_conditioning": "ordinary_prompt",
        }
        self._soft_prepared = True
        return result

    def _require_soft_prepared(self) -> None:
        if not self._soft_prepared or self.objective is None:
            raise ValueError("soft_prepare must succeed before this soft-prompt operation")

    def _load_soft_embeddings(self) -> t.Tensor:
        self._require_soft_prepared()
        path = self.runtime._gcg_sidecar_dir / "soft_embeddings.safetensors"
        try:
            return load_soft_embeddings(
                path=path, shape=(len(self.seed_token_ids), self._embedding_dim())
            )
        finally:
            # A consumed (including rejected) request cannot replay a stale sidecar.
            path.unlink(missing_ok=True)

    def _native_soft_embeddings(self, embeddings: t.Tensor) -> t.Tensor:
        validate_soft_embeddings(embeddings, (len(self.seed_token_ids), self._embedding_dim()))
        weight = self.runtime.benchmark_model.model.get_input_embeddings().weight
        native = embeddings.detach().to(device=weight.device, dtype=weight.dtype)
        if not bool(t.isfinite(native).all()):
            raise ValueError("soft embeddings become nonfinite in the model's native dtype")
        return native

    def prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._soft_prepared = False
        fields = _exact_keys(
            payload,
            {
                "alpha",
                "device_batch_size",
                "objective",
                "rows",
                "separator",
                "suffix_token_count",
            },
            label="gcg_prepare payload",
        )
        self._reset()
        try:
            objective = _text(fields["objective"], name="objective")
            if objective not in _OBJECTIVES:
                raise ValueError(f"objective must be one of {sorted(_OBJECTIVES)}")
            alpha = _float(fields["alpha"], name="alpha", minimum=0.0)
            separator = _text(fields["separator"], name="separator", nonblank=False)
            if separator != "\n\n":
                raise ValueError("separator must be exactly '\\n\\n'")
            k = _integer(fields["suffix_token_count"], name="suffix_token_count", minimum=1)
            device_batch_size = _integer(
                fields["device_batch_size"], name="device_batch_size", minimum=1
            )
            rows = fields["rows"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("rows must be a nonempty ordered list")
            main_dir = getattr(self.runtime, "_gcg_main_dump_dir", None)
            if not isinstance(main_dir, Path):
                raise ValueError("GCG requires the worker's bound main dump directory")
            wrapper = rendered_prompt_wrapper_from_manifest(
                main_dir / "run_manifest.json",
                tokenizer=self.runtime.tokenizer,
                model_type=self.runtime.benchmark_model.model.config.model_type,
            )
            if (
                wrapper.source_model_name != self.runtime.model_name
                or wrapper.revision != self.runtime.model_revision
            ):
                raise ValueError(
                    "GCG runtime model identity differs from the source prompt wrapper"
                )
            steering_prefix = self._steering_prefix(wrapper)
            seed_text = " ".join(["!"] * k)
            seed_ids = _encode(self.runtime.tokenizer, seed_text, add_special_tokens=False)
            if len(seed_ids) != k:
                raise ValueError(
                    f"seed text {seed_text!r} must encode to exactly {k} token ids, got {len(seed_ids)}"
                )
            layouts: list[_PairLayout] = []
            pair_ids: set[int] = set()
            sample_ids: set[int] = set()
            for index, value in enumerate(rows):
                row = _exact_keys(value, {"negative", "pair_id", "positive"}, label=f"row {index}")
                pair_id = _integer(row["pair_id"], name=f"row {index}.pair_id")
                if pair_id in pair_ids:
                    raise ValueError(f"duplicate pair_id {pair_id}")
                pair_ids.add(pair_id)
                positive = _completion_layout(
                    self.runtime,
                    row["positive"],
                    wrapper=wrapper,
                    steering_prefix=steering_prefix,
                    separator=separator,
                    seed_text=seed_text,
                    seed_ids=seed_ids,
                    label=f"row {index}.positive",
                )
                negative = _completion_layout(
                    self.runtime,
                    row["negative"],
                    wrapper=wrapper,
                    steering_prefix=steering_prefix,
                    separator=separator,
                    seed_text=seed_text,
                    seed_ids=seed_ids,
                    label=f"row {index}.negative",
                )
                for sample_id in (positive.sample_id, negative.sample_id):
                    if sample_id in sample_ids:
                        raise ValueError(f"duplicate sample_id {sample_id}")
                    sample_ids.add(sample_id)
                layouts.append(_PairLayout(pair_id, positive, negative))

            embedding = (
                self.runtime.benchmark_model.model.get_input_embeddings()
                .weight.detach()
                .float()
                .cpu()
                .contiguous()
            )
            allowed = _allowed_token_ids(self.runtime, embedding.shape[0])
            allowed_set = frozenset(allowed)
            if any(token_id not in allowed_set for token_id in seed_ids):
                raise ValueError(
                    "one or more seed token ids are absent from the canonical allowed pool"
                )
            all_sides = [side for pair in layouts for side in (pair.positive, pair.negative)]
            if any(not side.fixed_before for side in all_sides):
                raise ValueError("GCG row has no immutable wrapper token")
            first_wrapper_tokens = {side.fixed_before[0] for side in all_sides}
            if len(first_wrapper_tokens) != 1:
                raise ValueError("GCG rows disagree on the first immutable wrapper token")
            sentinel = next(iter(first_wrapper_tokens))
            if not 0 <= sentinel < embedding.shape[0]:
                raise ValueError(
                    "first immutable wrapper token is outside the embedding vocabulary"
                )

            sidecar_dir = self.runtime._gcg_sidecar_dir
            if not isinstance(sidecar_dir, Path):
                raise ValueError("GCG prepare requires a root-owned sidecar directory")
            path = sidecar_dir / "input_embeddings.safetensors"
            self._sidecar_path = path
            embedding_sidecar = _write_tensor_sidecar(path, "embedding_weight", embedding)

            self.objective = objective
            self.alpha = alpha
            self.separator = separator
            self.device_batch_size = device_batch_size
            self.seed_token_ids = seed_ids
            self.layouts = tuple(layouts)
            self._allowed_token_id_set = allowed_set
            self._wrapper = wrapper
            if objective == "activation_l2":
                with _preserve_runtime(self.runtime):
                    calibration = self._prepare_activation()
            else:
                calibration = None
            identity_rows = [
                {
                    "pair_id": pair.pair_id,
                    "positive": [
                        pair.positive.sample_id,
                        len(pair.positive.fixed_before),
                        len(pair.positive.fixed_after),
                        len(pair.positive.response_ids),
                    ],
                    "negative": [
                        pair.negative.sample_id,
                        len(pair.negative.fixed_before),
                        len(pair.negative.fixed_after),
                        len(pair.negative.response_ids),
                    ],
                }
                for pair in layouts
            ]
            layouts_sha256 = hashlib.sha256(
                json.dumps(identity_rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            identity = {
                "concept_id": self.runtime.concept_id,
                "factor": self.runtime.factor,
                "layer": self.runtime.layer,
                "method": self.runtime.method,
                "model_name": self.runtime.model_name,
                "model_revision": self.runtime.model_revision,
                "rank_sentinel_id": sentinel,
                "tokenizer_class": type(self.runtime.tokenizer).__name__,
                "vocab_size": int(embedding.shape[0]),
            }
            return {
                "embedding_sidecar": embedding_sidecar,
                "allowed_token_ids": list(allowed),
                "seed_token_ids": list(seed_ids),
                "member_count": len(layouts),
                "identity": identity,
                "layouts_sha256": layouts_sha256,
                "direction_calibration": calibration,
            }
        except BaseException:
            self._reset()
            raise

    def _prepare_activation(self) -> dict[str, Any]:
        main_dir = self.runtime._gcg_main_dump_dir
        train_dir = main_dir / "train"
        inference_dir = main_dir / "inference"
        weight_path = train_dir / "DiffMean_weight.pt"
        latent_path = inference_dir / "latent_data.parquet"
        if not weight_path.is_file() or not latent_path.is_file():
            raise ValueError(
                "activation_l2 requires canonical DiffMean weight and latent artifacts"
            )
        canonical = pd.read_parquet(latent_path, columns=["concept_id", "DiffMean_max_act"])
        selected = canonical.loc[canonical["concept_id"].eq(self.runtime.concept_id)]
        if selected.empty:
            raise ValueError("canonical latent_data.parquet lacks the selected concept")
        raw_max = float(selected["DiffMean_max_act"].max())
        fallback_used = not math.isfinite(raw_max) or raw_max <= 0.0
        for path in sorted(
            set(inference_dir.glob("latent_*.parquet"))
            | set(inference_dir.glob("rank_*_latent_data.parquet"))
        ):
            if path == latent_path:
                continue
            frame = pd.read_parquet(path, columns=["concept_id"])
            if bool(frame["concept_id"].eq(self.runtime.concept_id).any()):
                raise ValueError(f"noncanonical latent artifact overlaps selected concept: {path}")

        sibling = axbench.DiffMean(
            self.runtime.benchmark_model.model,
            self.runtime.tokenizer,
            layer=self.runtime.layer,
            training_args=None,
            low_rank_dimension=self.runtime.max_concepts,
            device=self.runtime.device,
            steering_layers=[self.runtime.layer],
        )
        sibling.load(
            dump_dir=train_dir,
            mode="steering",
            priority_mode="compute_priority",
            intervention_type="addition",
            concept_id=self.runtime.concept_id,
            low_rank_dimension=1,
        )
        sibling.to(self.runtime.device)
        sibling.ax.eval()
        sibling.ax.to(t.bfloat16)
        sibling.pre_compute_mean_activations(inference_dir)
        maximum = sibling.max_activations.get(self.runtime.concept_id)
        if maximum is None:
            raise ValueError("sibling DiffMean did not calibrate the selected concept")
        maximum = _float(maximum, name="DiffMean max activation", minimum=0.0)
        direction = sibling.ax.proj.weight[self.runtime.concept_id].detach().clone()
        if direction.dtype != t.bfloat16:
            raise RuntimeError(
                "DiffMean direction must preserve the worker's bfloat16 runtime dtype"
            )
        if (
            not t.isfinite(direction.float()).all()
            or float(t.linalg.vector_norm(direction.float())) == 0.0
        ):
            raise ValueError("DiffMean calibration direction must be finite and nonzero")
        max_tensor = t.tensor([maximum], dtype=t.float32, device=direction.device)
        mag_tensor = t.tensor([self.runtime.factor], dtype=t.float32, device=direction.device)
        self._direction_offset = (
            max_tensor.unsqueeze(-1) * mag_tensor.unsqueeze(-1) * direction.unsqueeze(0)
        )[0].detach()

        model_config = getattr(self.runtime.benchmark_model.model, "config", None)
        model_type = getattr(model_config, "model_type", None)
        component = (
            "block_output"
            if model_type == "gpt_neox"
            else f"model.layers[{self.runtime.layer}].output"
        )
        if self.runtime.method in _ADDITIVE_METHODS:
            actual = copy.deepcopy(self.runtime.benchmark_model.ax)
            hidden = int(self.runtime.benchmark_model.model.config.hidden_size)
            collector = CollectIntervention(embed_dim=hidden, keep_last_dim=True)
            config = IntervenableConfig(
                representations=[
                    {"layer": self.runtime.layer, "component": component, "intervention": actual},
                    {
                        "layer": self.runtime.layer,
                        "component": component,
                        "intervention": collector,
                    },
                ]
            )
            self._activation_model = IntervenableModel(config, self.runtime.benchmark_model.model)
            self._activation_model.set_device(self.runtime.device)
            # PyVene uses this fixed size as a Python slice bound. Keeping it
            # on the host avoids a CUDA scalar synchronization on every capture.
            collector.set_interchange_dim(t.tensor(hidden, dtype=t.long, device="cpu"))
        self._compute_clean_hidden()
        return {
            "site": (
                "block_output.after_actual_intervention"
                if model_type == "gpt_neox"
                else f"model.layers[{self.runtime.layer}].output_after_actual_intervention"
            ),
            "layer": self.runtime.layer,
            "concept_id": self.runtime.concept_id,
            "direction_path": str(weight_path.resolve()),
            "direction_sha256": _sha256_file(weight_path),
            "latent_path": str(latent_path.resolve()),
            "latent_sha256": _sha256_file(latent_path),
            "raw_max_activation": raw_max if math.isfinite(raw_max) else None,
            "max_activation": maximum,
            "fallback_used": fallback_used,
            "factor": self.runtime.factor,
            "resolved_scale": maximum * self.runtime.factor,
            "direction_norm": float(t.linalg.vector_norm(direction.float()).cpu()),
            "offset_dtype_semantics": (
                "float32 [1] max_act * float32 [1] factor * saved bfloat16 direction -> float32"
            ),
        }

    def _candidate(self, token_ids: tuple[int, ...]) -> tuple[str, str | None]:
        if len(token_ids) != len(self.seed_token_ids):
            return "", f"candidate must contain exactly {len(self.seed_token_ids)} token ids"
        if any(token_id not in self._allowed_token_id_set for token_id in token_ids):
            return "", "candidate contains a token outside the allowed pool"
        try:
            text = _decode(self.runtime.tokenizer, token_ids)
            if (
                not text
                or text.strip() != text
                or "\x00" in text
                or unicodedata.normalize("NFC", text) != text
            ):
                return (
                    text,
                    "candidate text is blank, outer-spaced, contains NUL, or changes under NFC",
                )
            if _encode(self.runtime.tokenizer, text, add_special_tokens=False) != token_ids:
                return text, "candidate decode/re-encode changes optimized token ids"
            wrapper = self._wrapper
            if wrapper is None:
                return ("", "candidate wrapper is unavailable") if self.layouts else (text, None)
            sides = [side for pair in self.layouts for side in (pair.positive, pair.negative)]
            texts = [
                f"{wrapper.prefix}{side.content}{self.separator}{text}{wrapper.suffix}"
                for side in sides
            ]
            observed_rows = _encode_many(self.runtime.tokenizer, texts, add_special_tokens=True)
            for side, observed in zip(sides, observed_rows, strict=True):
                if observed != side.fixed_before + token_ids + side.fixed_after:
                    return text, "candidate changes a fixed rendered wrapper or tail"
            return text, None
        except (UnicodeError, ValueError, TypeError, RuntimeError) as exc:
            return "", f"candidate canonicalization failed: {exc}"

    def filter(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_prepared()
        fields = _exact_keys(payload, {"candidate_token_ids"}, label="gcg_filter payload")
        values = fields["candidate_token_ids"]
        if not isinstance(values, list):
            raise ValueError("candidate_token_ids must be a list")
        candidates: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            try:
                token_ids = _token_ids(value, name=f"candidate {index}")
                text, error = self._candidate(token_ids)
            except ValueError as exc:
                token_ids, text, error = (), "", str(exc)
            candidates.append(
                {"token_ids": list(token_ids), "text": text, "valid": error is None, "error": error}
            )
        return {"candidates": candidates}

    def _validated_candidate(self, value: object, *, name: str) -> tuple[int, ...]:
        token_ids = _token_ids(value, name=name, count=len(self.seed_token_ids))
        _candidate_text, error = self._candidate(token_ids)
        if error is not None:
            raise ValueError(f"{name} is invalid: {error}")
        return token_ids

    def _validated_indices(self, value: object) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError("member_indices must be a nonempty list")
        indices = tuple(_integer(index, name="member index") for index in value)
        if len(set(indices)) != len(indices):
            raise ValueError("member_indices must be unique")
        if any(index >= len(self.layouts) for index in indices):
            raise ValueError("member index is outside the prepared layout range")
        return indices

    def _sequence(self, item: _SideItem) -> tuple[int, ...]:
        layout = item.layout
        return layout.fixed_before + item.tokens + layout.fixed_after + layout.response_ids

    def _padded_embeds(
        self, sequences: list[tuple[int, ...]], *, gradient: bool, stage_inputs: bool = False
    ) -> tuple[t.Tensor, t.Tensor]:
        embedding = self.runtime.benchmark_model.model.get_input_embeddings()
        vocab = int(embedding.weight.shape[0])
        pad_id = getattr(self.runtime.tokenizer, "pad_token_id", None)
        if type(pad_id) is not int or not 0 <= pad_id < vocab:
            pad_id = getattr(self.runtime.tokenizer, "eos_token_id", None)
        if type(pad_id) is not int or not 0 <= pad_id < vocab:
            pad_id = 0
        maximum = max(map(len, sequences))
        if stage_inputs:
            # Each copy owns its pinned allocation until the current-stream copy
            # completes. No reusable host buffer or cross-call tensor is retained.
            device = t.device(self.runtime.device)
            ids = t.tensor(
                [list(sequence) + [pad_id] * (maximum - len(sequence)) for sequence in sequences],
                dtype=t.long,
                device="cpu",
                pin_memory=True,
            ).to(device, non_blocking=True)
            attention = t.tensor(
                [[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences],
                dtype=t.long,
                device="cpu",
                pin_memory=True,
            ).to(device, non_blocking=True)
        else:
            ids = t.full(
                (len(sequences), maximum),
                pad_id,
                dtype=t.long,
                device=t.device(self.runtime.device),
            )
            attention = t.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                ids[row, : len(sequence)] = t.tensor(sequence, dtype=t.long, device=ids.device)
                attention[row, : len(sequence)] = 1
        embeds = embedding(ids).detach()
        embeds.requires_grad_(gradient)
        return embeds, attention

    def _capture_hidden(self, embeds: t.Tensor, attention: t.Tensor) -> t.Tensor:
        model = self.runtime.benchmark_model.model
        logits_to_keep: int | t.Tensor = 1
        if getattr(model.config, "model_type", None) in {"llama", "qwen2", "smollm3", "gpt_neox"}:
            # These HF forwards accept token indices. An empty selection keeps
            # the normal forward/collector path without projecting unused logits.
            logits_to_keep = t.empty(0, dtype=t.long, device=embeds.device)
        with _decoder_layer_prefix(model, self.runtime.layer):
            # The collector needs every position at its layer, but no later
            # blocks or vocabulary logits. Retain the existing HF forward path.
            if self.runtime.method in _ADDITIVE_METHODS:
                if self._activation_model is None:
                    raise ValueError("activation collector is unavailable")
                (_base, collected), _output = self._activation_model(
                    {
                        "inputs_embeds": embeds,
                        "attention_mask": attention,
                        "logits_to_keep": logits_to_keep,
                    },
                    unit_locations=None,
                    subspaces=[_subspace(self.runtime, embeds.shape[0]), None],
                    use_cache=False,
                )
                if not isinstance(collected, list) or len(collected) != 1:
                    raise RuntimeError(
                        "activation collector did not return exactly one block output"
                    )
                return collected[0]
            layer = decoder_layers(model)[self.runtime.layer]
            captured: list[t.Tensor] = []

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                captured.append(output[0] if isinstance(output, (tuple, list)) else output)

            handle = layer.register_forward_hook(capture, always_call=True)
            try:
                _forward(
                    self.runtime,
                    embeds=embeds,
                    attention_mask=attention,
                    logits_to_keep=logits_to_keep,
                )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("activation_l2 did not capture exactly one decoder-layer output")
            return captured[0]

    def _compute_clean_hidden(self) -> None:
        sides = [side for pair in self.layouts for side in (pair.positive, pair.negative)]
        with _preserve_runtime(self.runtime), t.no_grad():
            for start in range(0, len(sides), self.device_batch_size):
                chunk = sides[start : start + self.device_batch_size]
                sequences = [side.clean_prompt_ids + side.response_ids for side in chunk]
                embeds, attention = self._padded_embeds(sequences, gradient=False)
                hidden = self._capture_hidden(embeds, attention)
                for row, side in enumerate(chunk):
                    begin = len(side.clean_prompt_ids) - len(side.fixed_after)
                    end = len(sequences[row])
                    selected = hidden[row, begin:end].detach().float().cpu().contiguous()
                    if selected.shape[0] != side.aligned_token_count:
                        raise RuntimeError(
                            "clean activation tail alignment omitted a fixed or response token"
                        )
                    self._clean_hidden[side.sample_id] = selected

    def _side_batch(
        self,
        items: list[_SideItem],
        *,
        gradient: bool,
        soft_embeddings: t.Tensor | None = None,
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        sequences = [self._sequence(item) for item in items]
        if (
            not gradient
            and self.objective == "response_margin"
            and getattr(self._padded_embeds, "__func__", None) is _ORIGINAL_PADDED_EMBEDS
            and _can_stage_scoring_inputs(self.runtime, sequences)
        ):
            embeds, attention = self._padded_embeds(sequences, gradient=False, stage_inputs=True)
            # Unequal positive lengths imply a zero within q_length == kv_length.
            # Both native skip checks are therefore false. Ask the same factory
            # to build that mask without reading GPU scalars back to the host.
            attention = _CREATE_CAUSAL_MASK(
                config=self.runtime.benchmark_model.model.config,
                inputs_embeds=embeds,
                attention_mask=attention,
                past_key_values=None,
                position_ids=None,
                allow_is_causal_skip=False,
            )
        else:
            embeds, attention = self._padded_embeds(
                sequences, gradient=gradient and soft_embeddings is None
            )
        if soft_embeddings is not None:
            for row, item in enumerate(items):
                begin = len(item.layout.fixed_before)
                embeds[row, begin : begin + len(item.tokens)] = soft_embeddings
            # The VJP root is the complete native input, just as for GCG. The
            # root optimizer owns the independent FP32 master and cast policy.
            embeds.requires_grad_(gradient)
        # Finish the reference copies before forward, so loss kernels can be
        # queued afterwards without an intervening host-to-device transfer.
        clean_batch = (
            [self._clean_hidden[item.layout.sample_id].to(embeds.device) for item in items]
            if self.objective == "activation_l2"
            else []
        )
        if self.objective == "response_margin":
            output = _forward(self.runtime, embeds=embeds, attention_mask=attention)
            if output is None or not hasattr(output, "logits"):
                raise RuntimeError("response-margin forward returned no logits")
            hidden = None
        else:
            output = None
            hidden = self._capture_hidden(embeds, attention)
        losses: list[t.Tensor] = []
        for row, (item, sequence) in enumerate(zip(items, sequences, strict=True)):
            layout = item.layout
            response_start = len(layout.fixed_before) + len(item.tokens) + len(layout.fixed_after)
            if self.objective == "response_margin":
                selected_logits = output.logits[row, response_start - 1 : len(sequence) - 1].float()
                key = (layout.sample_id, selected_logits.device)
                constants = self._response_tensors.get(key)
                if constants is None:
                    # Fixed responses also serve later backward calls, even if
                    # the first scoring call is inside inference mode.
                    with t.inference_mode(False):
                        constants = (
                            t.tensor(
                                layout.response_ids, dtype=t.long, device=selected_logits.device
                            ),
                            t.tensor(
                                layout.response_byte_weights,
                                dtype=t.float32,
                                device=selected_logits.device,
                            ),
                        )
                    self._response_tensors[key] = constants
                targets, weights = constants
                token_losses = F.cross_entropy(selected_logits, targets, reduction="none")
                loss = weighted_bpb(token_losses, weights, tensor_ln2=False)
            else:
                if hidden is None or self._direction_offset is None:
                    raise RuntimeError("activation_l2 state is incomplete")
                aligned_start = len(layout.fixed_before) + len(item.tokens)
                attacked = hidden[row, aligned_start : len(sequence)]
                clean = clean_batch[row]
                loss = activation_l2_row(attacked, clean, self._direction_offset)
            losses.append(loss)
        # Rows are independent: the joint VJP preserves each row's gradient
        # without traversing the full batch's graph once per scalar loss.
        full_gradient = (
            t.autograd.grad(tuple(losses), embeds, create_graph=False)[0] if gradient else None
        )
        results: list[tuple[t.Tensor, t.Tensor | None]] = []
        for row, (item, loss) in enumerate(zip(items, losses, strict=True)):
            if full_gradient is not None:
                begin = len(item.layout.fixed_before)
                gradient_value = (
                    full_gradient[row, begin : begin + len(item.tokens)]
                    .detach()
                    .float()
                    .cpu()
                    .contiguous()
                )
            else:
                gradient_value = None
            results.append((loss.detach(), gradient_value))
        return results

    def _all_side_values(
        self,
        items: list[_SideItem],
        *,
        gradient: bool,
        batch_size: int,
        soft_embeddings: t.Tensor | None = None,
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        results: list[tuple[t.Tensor, t.Tensor | None]] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            results.extend(
                self._side_batch(chunk, gradient=gradient, soft_embeddings=soft_embeddings)
            )
        return results

    def score(
        self, payload: Mapping[str, Any], *, soft_embeddings: t.Tensor | None = None
    ) -> dict[str, Any]:
        self._require_prepared()
        fields = _exact_keys(
            payload,
            {"candidate_eval_device_batch_size", "candidate_token_ids", "member_indices"},
            label="gcg_score payload",
        )
        candidates = fields["candidate_token_ids"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidate_token_ids must be a nonempty list")
        override = fields["candidate_eval_device_batch_size"]
        batch_size = (
            self.device_batch_size
            if override is None
            else _integer(override, name="candidate_eval_device_batch_size", minimum=1)
        )
        indices = self._validated_indices(fields["member_indices"])
        parsed = [
            self._validated_candidate(value, name=f"candidate {index}")
            for index, value in enumerate(candidates)
        ]
        items: list[_SideItem] = []
        for tokens in parsed:
            for index in indices:
                pair = self.layouts[index]
                items.extend((_SideItem(pair.negative, tokens), _SideItem(pair.positive, tokens)))
        with (
            _preserve_runtime(self.runtime),
            t.no_grad(),
            fused_scoring_mlps(self.runtime.benchmark_model.model),
        ):
            side_values = self._all_side_values(
                items, gradient=False, batch_size=batch_size, soft_embeddings=soft_embeddings
            )
        losses: list[float] = []
        for offset in range(0, len(side_values), 2):
            negative = side_values[offset][0]
            positive = side_values[offset + 1][0]
            pair_loss = (
                activation_l2_pair(positive, negative)
                if self.objective == "activation_l2"
                else response_margin_value(negative, positive, alpha=self.alpha)
            )
            value = float(pair_loss)
            if not math.isfinite(value):
                raise RuntimeError("GCG score produced a nonfinite loss")
            losses.append(value)
        return {"candidate_count": len(parsed), "member_count": len(indices), "losses": losses}

    def gradient(
        self, payload: Mapping[str, Any], *, soft_embeddings: t.Tensor | None = None
    ) -> dict[str, Any]:
        self._require_prepared()
        fields = _exact_keys(payload, {"member_indices", "token_ids"}, label="gcg_gradient payload")
        tokens = self._validated_candidate(fields["token_ids"], name="token_ids")
        indices = self._validated_indices(fields["member_indices"])
        items: list[_SideItem] = []
        for index in indices:
            pair = self.layouts[index]
            items.extend((_SideItem(pair.negative, tokens), _SideItem(pair.positive, tokens)))
        with _preserve_runtime(self.runtime), t.enable_grad():
            side_values = self._all_side_values(
                items,
                gradient=True,
                batch_size=self.device_batch_size,
                soft_embeddings=soft_embeddings,
            )
        gradients: list[t.Tensor] = []
        for position in range(len(indices)):
            negative = side_values[2 * position][1]
            positive = side_values[2 * position + 1][1]
            if negative is None or positive is None:
                raise RuntimeError("GCG gradient operation returned no side VJP")
            gradient_value = (
                activation_l2_pair(positive, negative)
                if self.objective == "activation_l2"
                else response_margin_gradient(negative, positive, alpha=self.alpha)
            ).contiguous()
            if (
                gradient_value.shape != (len(tokens), self._embedding_dim())
                or not t.isfinite(gradient_value).all()
            ):
                raise RuntimeError("GCG pair VJP has an invalid shape or nonfinite value")
            gradients.append(gradient_value)
        gradient_sidecar = _write_tensor_sidecar(
            self.runtime._gcg_sidecar_dir / "gradients.safetensors",
            "gradients",
            t.stack(gradients).contiguous(),
        )
        return {"gradient_sidecar": gradient_sidecar, "member_indices": list(indices)}

    def soft_score(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_soft_prepared()
        fields = _exact_keys(payload, {"member_indices"}, label="soft_score payload")
        embeddings = self._load_soft_embeddings()
        indices = self._validated_indices(fields["member_indices"])
        result = self.score(
            {
                "candidate_token_ids": [list(self.seed_token_ids)],
                "candidate_eval_device_batch_size": None,
                "member_indices": list(indices),
            },
            soft_embeddings=self._native_soft_embeddings(embeddings),
        )
        return {"losses": result["losses"], "member_indices": list(indices)}

    def soft_gradient(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._require_soft_prepared()
        fields = _exact_keys(payload, {"member_indices"}, label="soft_gradient payload")
        embeddings = self._load_soft_embeddings()
        return self.gradient(
            {"token_ids": list(self.seed_token_ids), "member_indices": fields["member_indices"]},
            soft_embeddings=self._native_soft_embeddings(embeddings),
        )

    @contextmanager
    def soft_generation(
        self, embeddings: t.Tensor, *, prompt: str, token_ids: list[int]
    ) -> Iterator[None]:
        """Replace the verified suffix inside each actual full-prefix forward.

        PyVene removes embedding hooks at generate entry. Register this hook
        inside the base forward, after its own intervention hooks are installed.
        """
        self._require_soft_prepared()
        wrapper = self._wrapper
        seed_text = " ".join(["!"] * len(self.seed_token_ids))
        if (
            wrapper is None
            or not prompt.startswith(wrapper.prefix)
            or not prompt.endswith(f"{self.separator}{seed_text}{wrapper.suffix}")
        ):
            raise ValueError("soft generation prompt does not preserve the prepared wrapper")
        tail = _encode(self.runtime.tokenizer, wrapper.suffix, add_special_tokens=False)
        begin = len(token_ids) - len(tail) - len(self.seed_token_ids)
        end = begin + len(self.seed_token_ids)
        if (
            begin <= 0
            or not tail
            or tuple(token_ids[begin:end]) != self.seed_token_ids
            or tuple(token_ids[end:]) != tail
        ):
            raise ValueError("soft generation prompt has an invalid suffix span")
        model = self.runtime.benchmark_model.model
        embedding = model.get_input_embeddings()
        native = self._native_soft_embeddings(embeddings)
        expected = t.tensor([token_ids], dtype=t.long, device=native.device)
        original_forward = model.forward
        had_instance_forward = "forward" in vars(model)
        instance_forward = vars(model).get("forward")
        injections = 0

        @wraps(original_forward)
        def forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal injections
            ids = kwargs.get("input_ids", args[0] if args else None)
            if (
                len(args) > 1
                or kwargs.get("inputs_embeds") is not None
                or not isinstance(ids, t.Tensor)
                or ids.ndim != 2
                or ids.shape[0] != 1
            ):
                raise ValueError("soft generation requires batch-one input_ids forwards")
            past = kwargs.get("past_key_values")
            if past is None:
                past_length = 0
            else:
                get_length = getattr(past, "get_seq_length", None)
                if not callable(get_length):
                    raise ValueError("soft generation received an unsupported cache")
                past_length = get_length()
                if type(past_length) is not int or past_length < 0:
                    raise ValueError("soft generation received an invalid cache length")
            if past_length:
                if not injections or past_length < expected.shape[1] or ids.shape[1] != 1:
                    raise ValueError("soft generation received unsupported cached-decode geometry")
                return original_forward(*args, **kwargs)
            # No-cache generation can repeat the full prefix with new tokens.
            # Its suffix remains at the original verified positions.
            if ids.shape[1] < expected.shape[1] or not t.equal(
                ids[:, : expected.shape[1]], expected
            ):
                raise ValueError("soft generation forward changed the verified prompt prefix")
            calls = 0

            def replace(_module: Any, inputs: Any, output: Any) -> t.Tensor:
                nonlocal calls, injections
                if (
                    calls
                    or not inputs
                    or not isinstance(inputs[0], t.Tensor)
                    or not t.equal(inputs[0], ids)
                    or not isinstance(output, t.Tensor)
                    or tuple(output.shape) != (*ids.shape, native.shape[1])
                    or output.dtype != native.dtype
                ):
                    raise ValueError("soft generation embedding lookup has unexpected geometry")
                result = output.clone()
                result[:, begin:end] = native
                calls += 1
                injections += 1
                return result

            handle = embedding.register_forward_hook(replace)
            try:
                result = original_forward(*args, **kwargs)
                if calls != 1:
                    raise RuntimeError("soft generation did not inject the suffix embeddings")
                return result
            finally:
                handle.remove()

        model.forward = forward
        try:
            yield
            if not injections:
                raise RuntimeError("soft generation did not inject the suffix embeddings")
        finally:
            if had_instance_forward:
                model.forward = instance_forward
            else:
                del model.forward

    def _embedding_dim(self) -> int:
        return int(self.runtime.benchmark_model.model.get_input_embeddings().weight.shape[1])

    def close(self) -> None:
        self._reset()


_ORIGINAL_PADDED_EMBEDS = GCGSession._padded_embeds

__all__ = ["GCGSession"]
