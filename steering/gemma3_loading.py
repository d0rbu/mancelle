"""Correct loading of Gemma-3 text decoders, including VLM (4b/12b/27b) repos.

The gemma-3 4b/12b/27b repos are ``Gemma3ForConditionalGeneration`` VLM
checkpoints whose text weights live under ``language_model.model.*``.
``Gemma3ForCausalLM`` expects ``model.*`` and does NOT remap them (transformers
5.x), silently random-initialising the entire text decoder. This module loads
the text decoder correctly and fails loud if any weight is unmaterialised.
See issue #549 and ``docs/pipelines/gemma-scope-2.md``.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from typing import Any, cast

import torch as t
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, Gemma3ForCausalLM
from transformers.initialization import no_init_weights
from transformers.modeling_utils import PreTrainedModel

from errors import ModelWeightLoadError

_VLM_TEXT_PREFIX = "language_model.model."
_VLM_LM_HEAD_PREFIX = "language_model.lm_head."


def assert_weights_materialized(
    *,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    mismatched_keys: Iterable[str] = (),
    tie_word_embeddings: bool,
    context: str,
) -> None:
    """Raise ``ModelWeightLoadError`` if any weight is missing/unexpected/mismatched.

    A tied ``lm_head.weight`` may be 'missing' (resolved by ``tie_weights()``);
    any other missing key, any unexpected key, or any mismatched key means
    weights were silently (random-)initialised — fatal.

    All-strict guard: fail-loud is always preferred over silent random-init.
    Benign-unexpected risk is accepted — callers must ensure checkpoint
    consistency before loading (issue #549, [3]).
    """
    allowed_missing = {"lm_head.weight"} if tie_word_embeddings else set()
    bad_missing = [k for k in missing_keys if k not in allowed_missing]
    unexpected = list(unexpected_keys)
    mismatched = list(mismatched_keys)
    if bad_missing or unexpected or mismatched:
        raise ModelWeightLoadError(
            f"model weights not fully materialised for {context!r}: "
            f"{len(bad_missing)} missing (e.g. {bad_missing[:5]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:5]}), "
            f"{len(mismatched)} mismatched (e.g. {mismatched[:5]}). "
            "This is the issue #549 silent-random-weight class of failure."
        )


def from_pretrained_checked(
    loader_cls: Any,
    hf_id: str,
    *,
    dtype: t.dtype,
    attn_implementation: str = "sdpa",
) -> PreTrainedModel:
    """Load via ``loader_cls.from_pretrained`` and run the strict materialisation
    guard (issue #549). info values are heterogeneous (key sets + error_msgs list).

    This is the shared strict helper used by both the text-only and (via Task 3)
    the native/gradient Gemma-2 load paths. Any missing, unexpected, or
    mismatched key surfaces ``ModelWeightLoadError`` immediately — no silent
    random-init leakage.
    """
    model, info = cast(
        tuple[Any, dict[str, Any]],
        loader_cls.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        ),
    )
    assert_weights_materialized(
        missing_keys=info["missing_keys"],
        unexpected_keys=info["unexpected_keys"],
        mismatched_keys=info["mismatched_keys"],
        tie_word_embeddings=bool(getattr(model.config, "tie_word_embeddings", False)),
        context=hf_id,
    )
    return cast(PreTrainedModel, model)


def is_gemma3_vlm(config_or_hf_id: Any) -> bool:
    """True iff the Gemma-3 config is a VLM (carries a nested ``text_config``)."""
    config = config_or_hf_id
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)
    return getattr(config, "text_config", None) is not None


def _resolve_snapshot(hf_id: str) -> str:
    """Local dir → itself; hub id → cached snapshot (offline-friendly)."""
    if os.path.isdir(hf_id):
        return hf_id
    return snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.json"])


def _read_remapped_text_state_dict(hf_id: str) -> dict[str, t.Tensor]:
    """Read only text-decoder tensors from a VLM checkpoint, remapped
    ``language_model.model.* -> model.*`` / ``language_model.lm_head.* -> lm_head.*``."""
    snapshot = _resolve_snapshot(hf_id)
    shard_paths = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    if not shard_paths:
        raise ModelWeightLoadError(f"no safetensors shards under {snapshot!r} for {hf_id!r}")
    remapped: dict[str, t.Tensor] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(_VLM_TEXT_PREFIX):
                    remapped["model." + key[len(_VLM_TEXT_PREFIX):]] = handle.get_tensor(key)
                elif key.startswith(_VLM_LM_HEAD_PREFIX):
                    remapped["lm_head." + key[len(_VLM_LM_HEAD_PREFIX):]] = handle.get_tensor(key)
    return remapped


def load_gemma3_text_decoder(
    hf_id: str,
    *,
    dtype: t.dtype,
    attn_implementation: str = "sdpa",
) -> Gemma3ForCausalLM:
    """Load the Gemma-3 text decoder, correctly handling VLM (4b/12b/27b) repos.

    Text-only repos (270m/1b) take the ``from_pretrained_checked`` path which
    runs the strict materialisation guard. VLM repos load via a prefix remap
    into a ``Gemma3ForCausalLM`` constructed under ``no_init_weights()`` to
    skip the wasted random initialisation of weights (buffers such as
    ``embed_scale = sqrt(d_model)`` and rotary ``inv_freq`` are still
    materialised correctly by ``__init__`` before ``load_state_dict`` replaces
    params). The ``load_state_dict`` call is wrapped to turn any ``RuntimeError``
    (commonly shape/key drift) into the typed ``ModelWeightLoadError`` (#549 F1).
    Both paths fail loud on any unmaterialised weight. Note: the VLM path holds
    an fp32 model in RAM while loading — a meta-device follow-up is needed for
    12b/27b. ``hf_id`` may be a hub id or a local dir.
    """
    config = AutoConfig.from_pretrained(hf_id)
    if not is_gemma3_vlm(config):  # F9: branch via discriminator, covers text-only
        return cast(Gemma3ForCausalLM, from_pretrained_checked(
            Gemma3ForCausalLM, hf_id, dtype=dtype, attn_implementation=attn_implementation
        ))

    text_config = config.text_config
    text_config._attn_implementation = attn_implementation
    remapped = _read_remapped_text_state_dict(hf_id)
    with no_init_weights():  # F6: skip wasted random init; buffers still materialised
        model = Gemma3ForCausalLM(text_config)
    try:  # F1: RuntimeError (commonly shape/key drift) → typed fatal error
        result = model.load_state_dict(remapped, strict=False)
    except RuntimeError as e:
        raise ModelWeightLoadError(
            f"gemma-3 VLM text-weight load_state_dict failed for {hf_id!r} "
            f"(issue #549 fail-loud; commonly a shape/key mismatch): {e}"
        ) from e
    model.tie_weights()
    assert_weights_materialized(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        tie_word_embeddings=bool(text_config.tie_word_embeddings),
        context=hf_id,
    )
    logger.info(
        f"load_gemma3_text_decoder: VLM remap for {hf_id} "
        f"({len(remapped)} text tensors → Gemma3ForCausalLM, dtype={dtype})"
    )
    # ty resolves .to() to a transformers-wrapped overload that expects
    # PreTrainedModel; cast to plain nn.Module to hit the correct signature.
    cast(t.nn.Module, model).to(dtype)  # .to() is in-place and returns self
    return model
