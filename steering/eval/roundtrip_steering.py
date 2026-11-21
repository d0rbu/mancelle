"""Low-level SAE probe primitives used by AxBench-BPB-Adv round-trip scoring."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
)
from steering.eval.feature_probe import (
    apply_sae_activation_batched,
    build_completion_mask,
    compute_raw_probe_batched,
    max_pool_masked_batched,
)
from steering.eval.probe_activations import get_probe_activations
from steering.eval.probe_model import ProbeModel, load_probe_model
from steering.refined_types import (
    NonEmptyNonNegativeIntTupleLike,
    PositiveIntLike,
)
from steering.sae_resolver import L0Tier
from steering.segmentation import token_to_char_offsets
from steering.stats_io import load_percentile_checksum, load_statistics

ROUNDTRIP_BLOCK_KEY = "roundtrip_steering"


@dataclass(frozen=True)
class RoundtripProbeConfig:
    """Probe model + SAE identity used for round-trip scoring."""

    model_name: str = "gemma-2-2b"
    sae_release: str = "gemmascope"
    activation_type: str = "residual"
    layer: int = 8
    sae_width: str = "16k"
    l0: int | L0Tier | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        raw_l0 = self.l0
        if isinstance(raw_l0, bool):
            raise ValueError("roundtrip probe l0 must not be a boolean")
        if raw_l0 is None or isinstance(raw_l0, (int, L0Tier)):
            return
        if isinstance(raw_l0, str):
            try:
                parsed_numeric_l0 = int(raw_l0)
            except ValueError:
                try:
                    parsed_l0_tier = L0Tier(raw_l0)
                except ValueError as exc:
                    raise ValueError(
                        "roundtrip probe l0 string must be a numeric GemmaScope "
                        "v1 average L0 or a GemmaScope-2 tier "
                        f"({', '.join(tier.value for tier in L0Tier)}), "
                        f"got {raw_l0!r}"
                    ) from exc
                object.__setattr__(self, "l0", parsed_l0_tier)
            else:
                # Preserve the legacy config parser's int("42") behaviour.
                object.__setattr__(self, "l0", parsed_numeric_l0)
            return
        raise ValueError(
            "roundtrip probe l0 must be an integer GemmaScope v1 average L0, "
            "a GemmaScope-2 tier string, or None; "
            f"got {type(raw_l0).__name__}"
        )


@beartype
@dataclass(frozen=True)
class RoundtripEvalConfig:
    """AxBench-BPB-Adv round-trip probe settings."""

    enabled: bool = False
    feature_statistics_path: str | None = None
    feature_chunk_size: PositiveIntLike = 512
    probe: RoundtripProbeConfig = RoundtripProbeConfig()


@dataclass(frozen=True)
class RoundtripFeatureScores:
    """Per-record score against one target feature set."""

    n_features: int
    pre_act_mean: float
    post_act_mean: float
    percentile_mean: float


if TYPE_CHECKING:
    _IntIterable = Iterable[int]
else:
    # Domain objects own their element invariants. Unsubscripted runtime
    # containers avoid beartype's global-RNG-backed element sampling.
    _IntIterable = Iterable


def _plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def parse_roundtrip_eval_config(cfg: Mapping[str, object]) -> RoundtripEvalConfig:
    """Parse AxBench-BPB-Adv ``roundtrip_steering`` probe settings."""

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"round-trip eval config must be a mapping, got {type(cfg).__name__}"
        )
    raw = _plain(cfg.get(ROUNDTRIP_BLOCK_KEY))
    if raw is None:
        return RoundtripEvalConfig(enabled=False)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{ROUNDTRIP_BLOCK_KEY} must be a mapping, got {type(raw).__name__}"
        )
    enabled = bool(raw.get("enabled", False))
    if "empty_response_policy" in raw:
        raise ValueError(
            f"{ROUNDTRIP_BLOCK_KEY}.empty_response_policy is no longer "
            "supported; empty round-trip responses always raise"
        )
    probe_raw = raw.get("probe", {}) or {}
    if not isinstance(probe_raw, Mapping):
        raise TypeError(
            f"{ROUNDTRIP_BLOCK_KEY}.probe must be a mapping, "
            f"got {type(probe_raw).__name__}"
        )
    raw_l0 = probe_raw.get("l0", None)
    probe = RoundtripProbeConfig(
        model_name=str(probe_raw.get("model_name", "gemma-2-2b")),
        sae_release=str(probe_raw.get("sae_release", "gemmascope")),
        activation_type=str(probe_raw.get("activation_type", "residual")),
        layer=int(probe_raw.get("layer", 8)),
        sae_width=str(probe_raw.get("sae_width", "16k")),
        l0=(
            None
            if raw_l0 is None
            else raw_l0
            if isinstance(raw_l0, (str, bool))
            else int(raw_l0)
        ),
        device=str(probe_raw.get("device", "auto")),
    )
    try:
        out = RoundtripEvalConfig(
            enabled=enabled,
            feature_statistics_path=(
                None
                if raw.get("feature_statistics_path", None) is None
                else str(raw["feature_statistics_path"])
            ),
            feature_chunk_size=raw.get("feature_chunk_size", 512),
            probe=probe,
        )
    except BeartypeCallHintParamViolation as exc:
        raise ValueError(
            f"{ROUNDTRIP_BLOCK_KEY}.feature_chunk_size must be a positive integer, "
            f"got {raw.get('feature_chunk_size', 512)!r}"
        ) from exc
    return out


def resolve_probe_device(device: str) -> t.device:
    if device == "auto":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    return t.device(device)


def load_probe_model_for_roundtrip(config: RoundtripProbeConfig) -> ProbeModel:
    """Load the real probe model for production scoring."""

    return load_probe_model(
        model_name=config.model_name,
        layer=config.layer,
        activation_type=config.activation_type,
        sae_release=config.sae_release,
        sae_width=config.sae_width,
        device=resolve_probe_device(config.device),
        l0=config.l0,
    )

def load_percentile_table_for_roundtrip(
    stats_path: str | Path,
) -> tuple[t.Tensor, str]:
    """Build the percentile lookup table and return ``(table, checksum)``."""

    stats = load_statistics(stats_path)
    table = build_percentile_table(
        stats.per_feature_stats,
        int(stats.metadata.n_sae_features),
    )
    checksum = load_percentile_checksum(stats_path)
    if checksum is None:
        checksum = compute_percentile_checksum(table)
    return table, checksum


def _probe_text(
    *,
    prompt_text: str,
    response_text: str,
    probe_model: ProbeModel,
    get_activations_fn: Callable[..., t.Tensor],
) -> tuple[t.Tensor, t.Tensor]:
    full_text = prompt_text + response_text
    if not full_text:
        raise ValueError("roundtrip probe text is empty")

    tokens = probe_model.tokenizer.encode(full_text)
    if not tokens:
        raise ValueError(
            "probe_model.tokenizer.encode returned no tokens for non-empty "
            f"roundtrip text of length {len(full_text)}"
        )
    char_offsets = token_to_char_offsets(probe_model.tokenizer, tokens)
    mask = build_completion_mask(char_offsets, len(prompt_text))
    if not bool(mask.any()):
        raise ValueError(
            "roundtrip completion mask is all-False: response text produced no "
            f"probe tokens (prompt_chars={len(prompt_text)}, "
            f"response_chars={len(response_text)}, n_tokens={len(tokens)})"
        )
    tokens_tensor = t.tensor([tokens], device=probe_model.device)
    with t.inference_mode():
        act = get_activations_fn(
            probe_model.model,
            tokens_tensor,
            layer=probe_model.layer,
            hook_name=probe_model.hook_name,
            activation_type=probe_model.activation_type,
            nnterp_model=probe_model.nnterp_model,
        )
    return act.squeeze(0).float(), mask


@beartype
def _score_feature_ids_from_activation(
    *,
    act: t.Tensor,
    mask: t.Tensor,
    feature_ids: NonEmptyNonNegativeIntTupleLike,
    probe_model: ProbeModel,
    percentile_table: t.Tensor | None,
    feature_chunk_size: PositiveIntLike,
) -> RoundtripFeatureScores:
    if act.ndim != 2:
        raise ValueError(
            "roundtrip probe activations must have shape [seq_len, d_model], "
            f"got shape {tuple(act.shape)}"
        )
    if mask.ndim != 1 or mask.shape[0] != act.shape[0]:
        raise ValueError(
            "roundtrip completion mask must have shape [seq_len] matching "
            f"activations; got mask shape {tuple(mask.shape)} and activation "
            f"shape {tuple(act.shape)}"
        )

    n_sae_features = int(probe_model.sae.W_enc.shape[1])
    min_fid = min(feature_ids)
    max_fid = max(feature_ids)
    if max_fid >= n_sae_features:
        raise ValueError(
            "roundtrip feature ids are outside the probe SAE width: "
            f"min={min_fid}, max={max_fid}, n_sae_features={n_sae_features}"
        )
    percentile_table_cpu = None
    if percentile_table is not None:
        if percentile_table.ndim != 2:
            raise ValueError(
                "roundtrip percentile_table must have shape [n_features, bins], "
                f"got shape {tuple(percentile_table.shape)}"
            )
        if max_fid >= int(percentile_table.shape[0]):
            raise ValueError(
                "roundtrip percentile_table does not cover all target features: "
                f"max_feature_id={max_fid}, table_n_features={int(percentile_table.shape[0])}"
            )
        percentile_table_cpu = percentile_table.detach().cpu()

    with t.no_grad():
        mask = mask.to(device=act.device)
        pre_chunks: list[t.Tensor] = []
        post_chunks: list[t.Tensor] = []
        pct_chunks: list[t.Tensor] = []
        for start in range(0, len(feature_ids), feature_chunk_size):
            chunk_ids = feature_ids[start : start + feature_chunk_size]
            fid_t = t.tensor(chunk_ids, dtype=t.int64, device=act.device)
            pre_probes = compute_raw_probe_batched(
                act,
                probe_model.sae.W_enc,
                probe_model.sae.b_enc,
                probe_model.sae.b_dec,
                fid_t,
                apply_b_dec=probe_model.apply_b_dec,
            )
            pre_vals = max_pool_masked_batched(pre_probes, mask)
            post_probes = apply_sae_activation_batched(
                pre_probes,
                probe_model.sae,
                fid_t,
            )
            post_vals = max_pool_masked_batched(post_probes, mask)

            if not bool(t.isfinite(pre_vals).all()):
                raise ValueError(
                    f"non-finite roundtrip pre-activation values for features {chunk_ids}"
                )
            if not bool(t.isfinite(post_vals).all()):
                raise ValueError(
                    f"non-finite roundtrip post-activation values for features {chunk_ids}"
                )

            pre_chunks.append(pre_vals.detach().float().cpu())
            post_chunks.append(post_vals.detach().float().cpu())
            if percentile_table_cpu is not None:
                pct = magnitude_to_percentile(
                    fid_t.detach().cpu(),
                    post_vals.detach().float().cpu(),
                    percentile_table_cpu,
                )
                pct_chunks.append(pct.detach().cpu())

        pre_all = t.cat(pre_chunks)
        post_all = t.cat(post_chunks)
        pct_mean = float("nan")
        if pct_chunks:
            pct_all = t.cat(pct_chunks)
            if not bool(t.isfinite(pct_all).all()):
                raise ValueError("non-finite roundtrip percentile values")
            pct_mean = float(pct_all.mean().item())
        return RoundtripFeatureScores(
            n_features=len(feature_ids),
            pre_act_mean=float(pre_all.mean().item()),
            post_act_mean=float(post_all.mean().item()),
            percentile_mean=pct_mean,
        )


@beartype
def probe_text_against_features(
    *,
    prompt_text: str,
    response_text: str,
    feature_ids: _IntIterable,
    probe_model: ProbeModel,
    percentile_table: t.Tensor | None,
    feature_chunk_size: PositiveIntLike = 512,
    get_activations_fn: Callable[..., t.Tensor] = get_probe_activations,
) -> RoundtripFeatureScores:
    """Probe one prompt/response pair against one feature set.

    Round-trip scoring is inference-only and does not record gradients for the
    activation or SAE probe computations, even when the supplied probe
    parameters are trainable.
    """

    act, mask = _probe_text(
        prompt_text=prompt_text,
        response_text=response_text,
        probe_model=probe_model,
        get_activations_fn=get_activations_fn,
    )
    return _score_feature_ids_from_activation(
        act=act,
        mask=mask,
        feature_ids=tuple(sorted({int(fid) for fid in feature_ids})),
        probe_model=probe_model,
        percentile_table=percentile_table,
        feature_chunk_size=feature_chunk_size,
    )
