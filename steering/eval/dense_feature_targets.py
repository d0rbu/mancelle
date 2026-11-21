"""Document percentile contrasts shared by AxBench and offline ChatCORE fitting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch as t
from sae_lens import SAE
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import MissingPercentileResolver
from steering.extract_features import build_nnterp_model, load_sae_and_model
from steering.models import resolve_tokenizer
from steering.native_collector import build_native_model
from steering.sae_resolver import ResolvedSAE


def load_dense_extraction_stack(*, sae_model: str, sae_layer: int, resolved_sae: ResolvedSAE, device: str) -> tuple[Any, Any]:
    """Load Gemma and its SAE once for prompt preparation plus dense fits."""

    if sae_model.startswith("gemma-3"):
        model, sae, _tokenizer = load_sae_and_model(
            model_name=sae_model,
            layer=sae_layer,
            device="cpu",
            resolved=resolved_sae,
        )
        nnterp_model = build_nnterp_model(
            tl_model=None,
            device=device,
            model_dtype=t.bfloat16,
            model_name=sae_model,
        )
        if nnterp_model is None:
            raise RuntimeError(f"nnterp could not load extraction model {sae_model!r}")
        model.nnterp_model = nnterp_model
        sae.to(device)
        sae.to(t.bfloat16)
        return model, sae
    model, sae, _tokenizer = load_sae_and_model(
        model_name=sae_model,
        layer=sae_layer,
        device=device,
        resolved=resolved_sae,
    )
    return model, sae


def load_weight_free_gemma3_extraction_stack(
    *, sae_model: str, resolved_sae: ResolvedSAE, device: str,
    native: bool = False, sae_layer: int | None = None,
) -> tuple[Any, Any]:
    """Load the project extractor without unused TransformerLens weights."""
    if not sae_model.startswith("gemma-3"):
        raise ValueError("weight-free extraction requires a Gemma-3 model")
    if native and (
        t.device(device).type != "cuda"
        or sae_layer is None
        or resolved_sae.hook_name != f"blocks.{sae_layer}.hook_resid_post"
    ):
        raise ValueError("native extraction requires CUDA and a matching residual-post SAE")
    cfg = get_pretrained_model_config(sae_model, device="cpu")
    tokenizer = resolve_tokenizer(sae_model)
    sae = SAE.from_pretrained(
        release=resolved_sae.release, sae_id=resolved_sae.sae_id, device="cpu",
    )
    if native:
        assert sae_layer is not None
        collector = build_native_model(sae_model, device, t.bfloat16, sae_layer)
        if collector is None:
            raise RuntimeError(f"native collector could not load extraction model {sae_model!r}")
        model = SimpleNamespace(cfg=cfg, tokenizer=tokenizer, native_model=collector)
    else:
        nnterp_model = build_nnterp_model(
            tl_model=None, device=device, model_dtype=t.bfloat16, model_name=sae_model,
        )
        if nnterp_model is None:
            raise RuntimeError(f"nnterp could not load extraction model {sae_model!r}")
        model = SimpleNamespace(cfg=cfg, tokenizer=tokenizer, nnterp_model=nnterp_model)
    sae.to(device)
    sae.to(t.bfloat16)
    return model, sae


def dense_mean_percentiles(
    activations: t.Tensor, *, percentile_table: t.Tensor, allowed_ids: t.Tensor,
    consolidation: Any, missing: MissingPercentileResolver, label: str = "dense DiffMean",
) -> t.Tensor:
    """Validate once, preserving AxBench's two historical reduction orders."""
    if activations.ndim != 2 or activations.shape[0] == 0:
        raise ValueError(f"{label} activations must be nonempty rank 2")
    n_sae_features = percentile_table.shape[0]
    if activations.shape[1] != n_sae_features:
        raise ValueError(f"{label} width {activations.shape[1]} != {n_sae_features}")
    if not bool(t.isfinite(activations).all()):
        raise ValueError(f"{label} activations contain NaN or Inf")
    if consolidation is not None:
        if consolidation.n_sae_features != n_sae_features:
            raise ValueError("dense DiffMean activation width disagrees with clusters")
        allowed_ids = t.nonzero(consolidation.allowed_universe, as_tuple=False).squeeze(1)
        if allowed_ids.numel() == 0:
            raise ValueError("dense DiffMean has no mask-allowed features")
        table = percentile_table.detach().to(device="cpu", dtype=t.float64)
        total = t.zeros(n_sae_features, dtype=t.float64)
        for activation_row in activations:
            ranks = missing.ranks(
                allowed_ids,
                activation_row.detach().to(dtype=t.float32, device="cpu")[allowed_ids],
                table,
            )
            representatives, _normalized, projected_ranks = consolidation.consolidate_segment(
                allowed_ids, ranks / 100.0, ranks,
            )
            row = t.zeros_like(total)
            row[representatives] = projected_ranks.to(t.float64)
            total += row
        return total / float(activations.shape[0])

    # Keep the unclustered stack/mean: replacing it with the clustered running
    # sum can change floating-point results even when the formulas agree.
    ids = allowed_ids.to(dtype=t.int64, device="cpu")
    rows = [
        missing.ranks(
            ids,
            activation.detach().to(dtype=t.float32, device="cpu")[ids],
            percentile_table,
        )
        for activation in activations
    ]
    mean = t.zeros(n_sae_features, dtype=t.float64)
    mean[ids] = t.stack(rows).to(t.float64).mean(dim=0)
    return mean


def dense_diffmean_from_means(
    positive_mean: t.Tensor, negative_mean: t.Tensor, *,
    feature_vocab: FeatureVocab | None, label: str,
) -> t.Tensor:
    """Subtract FP64 document means, then gather the exact checkpoint head rows."""
    if positive_mean.shape != negative_mean.shape:
        raise ValueError("dense DiffMean mean shapes disagree")
    original_scores = (positive_mean - negative_mean) / 100.0
    scores = (
        original_scores
        if feature_vocab is None
        else feature_vocab.select_rows(original_scores)
    )
    result = scores.detach().to(device="cpu", dtype=t.float32).contiguous()
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{label} produced no rows")
    if not bool(t.isfinite(result).all()):
        raise ValueError("dense DiffMean contains NaN or Inf")
    if not bool(((result > -1.0) & (result < 1.0)).all()):
        raise ValueError("dense DiffMean values must lie strictly between -1 and 1")
    return result
