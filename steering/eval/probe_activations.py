# steering/eval/probe_activations.py
"""Unified probe activation dispatcher — tries nnterp, falls back to TransformerLens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch as t
from loguru import logger

from errors import ExtractionError
from steering.extract_features import (
    _get_activations_nnterp,
    _get_activations_tl,
    build_nnterp_model,
    load_sae_and_model,
    maybe_gc_cuda,
)
from steering.sae_resolver import ActivationType

if TYPE_CHECKING:
    from nnterp import StandardizedTransformer
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

from steering.sae_resolver import ResolvedSAE
from steering.tokenizer_protocols import TextTokenizer, require_text_tokenizer


def get_probe_activations(
    tl_model: HookedTransformer | None,
    tokens: t.Tensor,
    layer: int,
    hook_name: str,
    activation_type: ActivationType,
    nnterp_model: StandardizedTransformer | None = None,
) -> t.Tensor:
    """Get activations for SAE probing, trying nnterp first with TL fallback.

    Args:
        tl_model: TransformerLens HookedTransformer (used as fallback), or
            ``None`` when the caller released that fallback after building
            nnterp.
        tokens: Input token IDs, shape [batch, seq_len].
        layer: Target layer index (0-indexed).
        hook_name: TransformerLens hook name (e.g. blocks.8.hook_resid_post).
        activation_type: Activation site (residual, mlp, attention).
        nnterp_model: Optional nnterp StandardizedTransformer.

    Returns:
        Activation tensor, shape [batch, seq_len, d_model].
    """
    if nnterp_model is not None:
        try:
            acts = _get_activations_nnterp(
                nnterp_model, tokens, layer, activation_type, hook_name=hook_name
            )
            device = tokens.device
            maybe_gc_cuda(device)
            logger.debug("Probe activations via nnterp (layer {})", layer)
            return acts
        except ExtractionError as exc:
            if tl_model is None:
                raise ExtractionError(
                    "nnterp probe failed and the TransformerLens fallback was released"
                ) from exc
            logger.opt(exception=True).debug(
                "nnterp probe failed at layer {}, falling back to TransformerLens", layer
            )

    if tl_model is None:
        raise ExtractionError("probe activation requested without an activation backend")
    return _get_activations_tl(tl_model, tokens, layer, hook_name=hook_name)


def trim_hf_model(hf_model: object, layer: int) -> None:
    """Delete HF transformer layers beyond ``layer`` (in-place).

    Supports two common layouts:
    - ``model.model.layers`` (Gemma-2, Llama, etc.)
    - ``model.transformer.h`` (GPT-2)
    """
    keep = layer + 1

    # Gemma-2 / Llama style
    layers_list = getattr(getattr(hf_model, "model", None), "layers", None)
    if layers_list is not None and len(layers_list) > keep:
        original = len(layers_list)
        del layers_list[keep:]
        logger.info("Trimmed HF model.model.layers from {} to {}", original, keep)
        return

    # GPT-2 style
    h_list = getattr(getattr(hf_model, "transformer", None), "h", None)
    if h_list is not None and len(h_list) > keep:
        original = len(h_list)
        del h_list[keep:]
        logger.info("Trimmed HF model.transformer.h from {} to {}", original, keep)
        return


def build_probe_models(
    model_name: str,
    layer: int,
    device: str,
    *,
    resolved: ResolvedSAE,
) -> tuple[
    HookedTransformer,
    SAE,
    TextTokenizer,
    StandardizedTransformer | None,
]:
    """Load TL model + SAE + tokenizer, and build a trimmed nnterp model on CUDA."""
    tl_model, sae, tokenizer = load_sae_and_model(
        model_name=model_name,
        layer=layer,
        device=device,
        resolved=resolved,
    )
    text_tokenizer = require_text_tokenizer(tokenizer, surface="feature probe")

    nnterp_model = None
    if device.startswith("cuda"):
        nnterp_model = build_nnterp_model(tl_model, device=device)
        if nnterp_model is not None:
            trim_hf_model(nnterp_model.model, layer)
            logger.info("Built trimmed nnterp probe model (layers 0-{})", layer)

    return tl_model, sae, text_tokenizer, nnterp_model
