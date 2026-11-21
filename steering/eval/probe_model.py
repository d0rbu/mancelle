"""Reusable trimmed base-model + SAE loader for feature activation probing."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import torch as t
from loguru import logger

from steering.eval.axbench_base.config import bare_model_id
from steering.eval.probe_activations import trim_hf_model
from steering.extract_features import build_nnterp_model, load_sae_and_model
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)


def _context_length(model: Any) -> int | None:
    """Return the first recognized context-length field from a model."""
    for container in (
        getattr(model, "cfg", None),
        getattr(model, "config", None),
        model,
    ):
        for name in ("n_ctx", "context_size", "max_position_embeddings"):
            value = getattr(container, name, None)
            if value is not None:
                return int(value)
    return None


def _restore_transformerlens_probe_device(
    model: Any,
    *,
    layer: int,
    device: t.device | str,
) -> None:
    """Restore only the TransformerLens modules needed by the fallback path.

    The accelerated nnterp builder temporarily moves the full TransformerLens
    model to CPU before loading its HF counterpart.  If that builder returns
    ``None`` (for example when an offline Gemma-3 config is unavailable), the
    fallback must move the embedding and executed blocks back without bringing
    the deliberately offloaded tail back onto the GPU.
    """

    for name in ("embed", "pos_embed"):
        module = getattr(model, name, None)
        if isinstance(module, t.nn.Module):
            module.to(device)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "TransformerLens probe fallback has no blocks collection"
        )
    if not 0 <= int(layer) < len(blocks):
        raise RuntimeError(
            f"TransformerLens probe layer {layer} is outside [0, {len(blocks)})"
        )
    for block in blocks[: int(layer) + 1]:
        block.to(device)

    # ``_build_nnterp_gemma3`` calls ``HookedTransformer.to("cpu")`` before
    # loading the HF companion.  TransformerLens' custom ``to`` updates
    # ``cfg.device`` as well as moving parameters, while the targeted moves
    # above intentionally leave the offloaded tail on CPU.  Restore the
    # config separately so ``run_with_cache`` places tokens and attention
    # masks on the active block device instead of moving them back to CPU.
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "device"):
        cfg.device = t.device(device).type


@dataclass
class ProbeModel:
    """Trimmed base model + SAE for feature activation probing."""

    # ``model`` is optional for consumers that use the nnterp fast path and
    # deliberately release the TransformerLens fallback after construction.
    model: Any | None
    sae: Any
    tokenizer: Any
    hook_name: str
    layer: int
    nnterp_model: Any = None
    context_length: int | None = None
    activation_type: ActivationType = ActivationType.residual
    apply_b_dec: bool = field(init=False)

    def __post_init__(self) -> None:
        self.apply_b_dec = bool(self.sae.cfg.apply_b_dec_to_input)

    @property
    def device(self) -> t.device:
        if self.nnterp_model is not None:
            nn_model = getattr(self.nnterp_model, "model", self.nnterp_model)
            return next(nn_model.parameters()).device
        if self.model is not None:
            return next(self.model.parameters()).device
        raise RuntimeError("ProbeModel has neither a TransformerLens nor nnterp model")

    def to(self, device: t.device | str) -> ProbeModel:
        """Move model and SAE to device."""
        # The nnterp backend is the active CUDA path when present.  The
        # retained TransformerLens model is deliberately parked on CPU as a
        # fallback; moving both would unnecessarily co-reside two full probe
        # models and can OOM the training/eval process.
        if self.nnterp_model is not None:
            self.nnterp_model.to(device)
        elif self.model is not None:
            self.model.to(device)
        self.sae.to(device)
        return self


def load_probe_model(
    model_name: str,
    layer: int,
    activation_type: str,
    sae_release: str,
    sae_width: str,
    device: t.device | str,
    *,
    l0: int | L0Tier | None = None,
    keep_transformerlens_model: bool = True,
) -> ProbeModel:
    """Load a trimmed base model + SAE for feature probing.

    ``keep_transformerlens_model=False`` is intended for callers that require
    only the nnterp CUDA path.  Once nnterp builds successfully, releasing the
    TransformerLens fallback avoids retaining two copies of the probe model.
    The default remains ``True`` because other evaluators rely on the
    TransformerLens fallback when nnterp is unavailable or fails at runtime.
    """
    # The SAE registry is keyed on BARE model ids while probe callers may pass
    # an org-prefixed HF id (the BPB-Adv roundtrip validator requires
    # probe.model_name == feature_extraction_model_id, which production
    # presets pin as ``google/gemma-3-4b-pt``). Same class as the
    # bare_model_id() sites in gcg_bags.py / runtime_state.py: without the
    # normalization a real gemma-scope-2 probe load raises
    # "Unsupported model 'google/gemma-3-4b-pt'" against a registry that
    # contains the bare id. Weight loading below keeps the caller's id.
    resolved = resolve_sae(
        model=bare_model_id(model_name),
        release=SAERelease(sae_release),
        activation_type=ActivationType(activation_type),
        layer=layer,
        width=SAEWidth(sae_width),
        l0=l0,
    )
    logger.info(
        "Loading probe model: {} layer {} | SAE: {}/{} hook={}",
        model_name,
        layer,
        resolved.release,
        resolved.sae_id,
        resolved.hook_name,
    )

    base_model, sae, tokenizer = load_sae_and_model(
        model_name=model_name,
        layer=layer,
        device=str(device),
        resolved=resolved,
        # ProbeModel permanently deletes this same tail below and every
        # activation call early-stops at ``layer``.  Offload it before the wide
        # SAE load as well, avoiding an otherwise unnecessary CUDA peak.
        offload_transformer_tail=True,
    )

    nnterp_mdl = None
    if str(device).startswith("cuda"):
        nnterp_mdl = build_nnterp_model(base_model, device=str(device))
        if nnterp_mdl is not None:
            trim_hf_model(nnterp_mdl.model, layer)
            logger.info("Built trimmed nnterp probe model")
        else:
            _restore_transformerlens_probe_device(
                base_model,
                layer=layer,
                device=device,
            )
            logger.info(
                "Restored TransformerLens probe fallback active layers to {}",
                device,
            )

    context_length = _context_length(base_model)
    if context_length is None and nnterp_mdl is not None:
        context_length = _context_length(nnterp_mdl.model)

    n_original = len(base_model.blocks)
    if n_original > layer + 1:
        del base_model.blocks[layer + 1 :]
        logger.info("Trimmed probe TL model from {} to {} blocks", n_original, layer + 1)

    if hasattr(base_model, "unembed"):
        del base_model.unembed
    if hasattr(base_model, "ln_final"):
        del base_model.ln_final

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in sae.parameters():
        p.requires_grad_(False)

    logger.info(
        "Probe model ready on {}: {:.0f}M params",
        device,
        sum(p.numel() for p in base_model.parameters()) / 1e6,
    )

    retained_model = base_model
    if nnterp_mdl is not None and not keep_transformerlens_model:
        # The nnterp path is the normal CUDA path for live highlighting.  Do
        # not keep the CPU-offloaded TransformerLens tail and active blocks
        # alive solely for a fallback that this caller explicitly declined.
        retained_model = None
        del base_model
        gc.collect()
        if str(device).startswith("cuda") and t.cuda.is_available():
            t.cuda.empty_cache()
        logger.info("Released TransformerLens probe fallback after nnterp build")

    return ProbeModel(
        model=retained_model,
        sae=sae,
        tokenizer=tokenizer,
        hook_name=resolved.hook_name,
        layer=layer,
        nnterp_model=nnterp_mdl,
        context_length=context_length,
        activation_type=ActivationType(activation_type),
    )
