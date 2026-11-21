"""Live SAE feature activation highlighting for completion TUI text segments."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger

from scripts.completion_tui.highlight_types import (
    FeatureHighlightRequest,
    FeatureHighlightResult,
    FeatureHighlightSpan,
    SegmentSpanSnapshot,
)
from steering.conditional.percentile_table import magnitude_to_percentile
from steering.eval.feature_probe import apply_sae_activation, compute_raw_probe
from steering.eval.probe_activations import get_probe_activations
from steering.eval.probe_model import ProbeModel, load_probe_model
from steering.feature_paths import parse_feature_dir
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import token_to_char_offsets


@dataclass(frozen=True)
class ProbeSpec:
    """Structured provenance needed to load the SAE base probe model."""

    model_name: str
    release: str
    activation_type: str
    layer: int
    width: str
    l0: int | L0Tier | None = None
    hook_name: str | None = None
    sae_id: str | None = None


ActivationProvider = Callable[[ProbeModel, t.Tensor], t.Tensor]
ProbeModelFactory = Callable[[], ProbeModel]


def _mapping_get(mapping: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    get = getattr(mapping, "get", None)
    if callable(get):
        return get(key)
    return None


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def resolve_probe_spec(
    user_cfg: Mapping[str, Any] | Any,
    feature_dir: str | Path,
) -> ProbeSpec:
    """Resolve probe model provenance from checkpoint config and feature dir.

    The checkpoint ``user_config.sae`` is authoritative because it includes
    release-family and optional L0 information that the feature directory does
    not encode.  The feature directory is still parsed and cross-checked for
    the fields it does encode, so a mismatched checkpoint/data pairing fails
    before any large model is loaded.
    """
    parsed = _parse_probe_feature_dir(feature_dir)
    sae_cfg = _mapping_get(user_cfg, "sae")
    if sae_cfg is None:
        release_s = SAERelease.gemmascope.value
        model_name_s = parsed.model
        activation_s = parsed.activation_type.value
        layer_i = int(parsed.layer)
        width_s = parsed.width.value
        resolved = resolve_sae(
            model=model_name_s,
            release=SAERelease(release_s),
            activation_type=ActivationType(activation_s),
            layer=layer_i,
            width=SAEWidth(width_s),
        )
        return ProbeSpec(
            model_name=model_name_s,
            release=release_s,
            activation_type=activation_s,
            layer=layer_i,
            width=width_s,
            hook_name=resolved.hook_name,
            sae_id=resolved.sae_id,
        )

    model_name = _mapping_get(sae_cfg, "model")
    release = _mapping_get(sae_cfg, "release")
    activation_type = _mapping_get(sae_cfg, "activation_type")
    layer = _mapping_get(sae_cfg, "layer")
    width = _mapping_get(sae_cfg, "width")
    l0_raw = _mapping_get(sae_cfg, "l0")
    missing = [
        name
        for name, value in (
            ("model", model_name),
            ("release", release),
            ("activation_type", activation_type),
            ("layer", layer),
            ("width", width),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Checkpoint user_config.sae is missing required field(s): "
            + ", ".join(missing)
        )

    model_name_s = str(model_name)
    release_s = _enum_value(release)
    activation_s = _enum_value(activation_type)
    width_s = _enum_value(width)
    layer_i = int(layer)

    dir_values = {
        "model": parsed.model,
        "activation_type": parsed.activation_type.value,
        "layer": parsed.layer,
        "width": parsed.width.value,
    }
    cfg_values = {
        "model": model_name_s,
        "activation_type": activation_s,
        "layer": layer_i,
        "width": width_s,
    }
    for key, cfg_value in cfg_values.items():
        dir_value = dir_values[key]
        if cfg_value != dir_value:
            raise ValueError(
                f"Checkpoint SAE {key}={cfg_value!r} does not match "
                f"feature_dir {key}={dir_value!r}"
            )

    resolved = resolve_sae(
        model=model_name_s,
        release=SAERelease(release_s),
        activation_type=ActivationType(activation_s),
        layer=layer_i,
        width=SAEWidth(width_s),
        l0=coerce_config_l0(SAERelease(release_s), l0_raw),
    )
    return ProbeSpec(
        model_name=model_name_s,
        release=release_s,
        activation_type=activation_s,
        layer=layer_i,
        width=width_s,
        l0=coerce_config_l0(SAERelease(release_s), l0_raw),
        hook_name=resolved.hook_name,
        sae_id=resolved.sae_id,
    )


_SPLICE_OVERLAY_RE = re.compile(r"^cut_frac\d+(?:\.\d+)?$")


def _parse_probe_feature_dir(feature_dir: str | Path):
    """Parse the canonical SAE stack, accepting a splice-derived leaf.

    Splice-counterfactual TUI launches point at ``.../splice/cut_frac*/seed*``
    directories.  That overlay changes document composition, not the SAE
    provenance needed to load the probe, so strip it before parsing.
    """
    path = Path(feature_dir)
    parts = path.parts
    if (
        len(parts) >= 4
        and parts[-3] == "splice"
        and _SPLICE_OVERLAY_RE.fullmatch(parts[-2])
        and parts[-1].startswith("seed")
    ):
        path = Path(*parts[:-3])
    return parse_feature_dir(path)


def _percentiles_to_levels(activations: t.Tensor, percentiles: t.Tensor) -> t.Tensor:
    """Map post-SAE activations and corpus percentiles to display levels.

    Level 0 stays gray/inactive.  Positive activations get at least level 1,
    with corpus-top activations promoted to stronger colors.
    """
    acts = activations.detach().cpu()
    pcts = percentiles.detach().cpu()
    levels = t.zeros(acts.shape, dtype=t.int64)
    active = acts > 0
    levels[active] = 1
    levels[active & (pcts >= 90.0)] = 2
    levels[active & (pcts >= 99.0)] = 3
    return levels


def _coalesce_token_highlights(
    *,
    token_starts: t.Tensor,
    token_ends: t.Tensor,
    levels: t.Tensor,
    percentiles: t.Tensor,
    activations: t.Tensor,
) -> tuple[FeatureHighlightSpan, ...]:
    """Collapse token-level grades into non-overlapping character spans.

    Zero-width token spans are ignored because the Rich text renderer has no
    visible character to color.  If tokenizer quirks produce overlapping
    visible spans, the highest level wins per character; ties keep the highest
    percentile/activation for display metadata.
    """
    best: dict[int, tuple[int, float, float]] = {}
    for start_t, end_t, level_t, pct_t, act_t in zip(
        token_starts.tolist(),
        token_ends.tolist(),
        levels.tolist(),
        percentiles.tolist(),
        activations.tolist(), strict=True,
    ):
        start = int(start_t)
        end = int(end_t)
        level = int(level_t)
        if level <= 0 or end <= start:
            continue
        pct = float(pct_t)
        act = float(act_t)
        for pos in range(start, end):
            existing = best.get(pos)
            if (
                existing is None
                or level > existing[0]
                or (level == existing[0] and pct > existing[1])
            ):
                best[pos] = (level, pct, act)

    if not best:
        return ()

    spans: list[FeatureHighlightSpan] = []
    sorted_positions = sorted(best)
    span_start = sorted_positions[0]
    prev_pos = span_start
    span_level, span_pct, span_act = best[span_start]

    for pos in sorted_positions[1:]:
        level, pct, act = best[pos]
        if pos == prev_pos + 1 and level == span_level:
            span_pct = max(span_pct, pct)
            span_act = max(span_act, act)
            prev_pos = pos
            continue
        spans.append(
            FeatureHighlightSpan(
                char_start=span_start,
                char_end=prev_pos + 1,
                level=span_level,
                percentile=span_pct,
                activation=span_act,
            )
        )
        span_start = pos
        prev_pos = pos
        span_level = level
        span_pct = pct
        span_act = act

    spans.append(
        FeatureHighlightSpan(
            char_start=span_start,
            char_end=prev_pos + 1,
            level=span_level,
            percentile=span_pct,
            activation=span_act,
        )
    )
    return tuple(spans)


def _model_context_length(model: Any) -> int | None:
    cfg = getattr(model, "cfg", None)
    for name in ("n_ctx", "context_size", "max_position_embeddings"):
        value = getattr(cfg, name, None)
        if value is not None:
            return int(value)
    for name in ("n_ctx", "context_size", "max_position_embeddings"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)
    return None


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _target_token_mask(
    token_starts: t.Tensor,
    token_ends: t.Tensor,
    segments: tuple[SegmentSpanSnapshot, ...],
    target_segment_index: int,
    prefix_end: int,
) -> t.Tensor:
    visible = token_ends > token_starts
    if not bool(visible.any()):
        return t.zeros_like(visible, dtype=t.bool)

    usable_segments = [
        seg
        for seg in segments
        if seg.char_end > seg.char_start
        and seg.char_start < prefix_end
        and seg.char_end > 0
    ]
    target_col = next(
        (
            i
            for i, seg in enumerate(usable_segments)
            if seg.list_index == target_segment_index
        ),
        None,
    )
    if target_col is None:
        return t.zeros_like(visible, dtype=t.bool)

    seg_starts = t.tensor(
        [max(0, seg.char_start) for seg in usable_segments], dtype=t.int64
    )
    seg_ends = t.tensor(
        [min(prefix_end, seg.char_end) for seg in usable_segments], dtype=t.int64
    )
    lo = t.maximum(token_starts.unsqueeze(1), seg_starts.unsqueeze(0))
    hi = t.minimum(token_ends.unsqueeze(1), seg_ends.unsqueeze(0))
    overlap = (hi - lo).clamp(min=0)
    max_overlap, assignments = overlap.max(dim=1)
    return visible & (max_overlap > 0) & (assignments == target_col)


class LiveFeatureHighlightService:
    """Lazy live SAE probe service used by the completion TUI."""

    def __init__(
        self,
        spec: ProbeSpec,
        percentile_table: t.Tensor,
        *,
        device: t.device | str | None = None,
        probe_model_factory: ProbeModelFactory | None = None,
        activation_provider: ActivationProvider | None = None,
    ) -> None:
        self._spec = spec
        self._percentile_table = percentile_table.detach().to(t.float64).cpu()
        self._device = device
        self._probe_model_factory = probe_model_factory
        self._activation_provider = activation_provider
        self._probe_model: ProbeModel | None = None

    @property
    def spec(self) -> ProbeSpec:
        return self._spec

    def _ensure_probe_model(self) -> ProbeModel:
        if self._probe_model is None:
            if self._probe_model_factory is not None:
                self._probe_model = self._probe_model_factory()
            else:
                device = (
                    self._device
                    if self._device is not None
                    else ("cuda" if t.cuda.is_available() else "cpu")
                )
                self._probe_model = load_probe_model(
                    model_name=self._spec.model_name,
                    layer=self._spec.layer,
                    activation_type=self._spec.activation_type,
                    sae_release=self._spec.release,
                    sae_width=self._spec.width,
                    device=device,
                    l0=self._spec.l0,
                    # Live highlighting is a CUDA nnterp consumer.  Releasing
                    # the unused TransformerLens fallback avoids retaining a
                    # second Gemma model in the TUI process.
                    keep_transformerlens_model=False,
                )
        return self._probe_model

    def _get_activations(self, probe_model: ProbeModel, tokens: t.Tensor) -> t.Tensor:
        if self._activation_provider is not None:
            return self._activation_provider(probe_model, tokens)
        return get_probe_activations(
            probe_model.model,
            tokens,
            layer=probe_model.layer,
            hook_name=probe_model.hook_name,
            activation_type=probe_model.activation_type,
            nnterp_model=probe_model.nnterp_model,
        )

    def compute(self, request: FeatureHighlightRequest) -> FeatureHighlightResult:
        """Compute live token highlights for one immutable request snapshot."""
        feature_ids = request.feature_ids or (request.feature_id,)
        if not (0 <= request.target_segment_index < len(request.segments)):
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=-1,
                feature_id=request.feature_id,
                status="unavailable",
                message="target segment no longer exists",
            )

        target = request.segments[request.target_segment_index]
        invalid_ids = [
            feature_id
            for feature_id in feature_ids
            if feature_id < 0 or feature_id >= self._percentile_table.shape[0]
        ]
        if invalid_ids:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="unavailable",
                message=f"feature id(s) outside the percentile table: {invalid_ids}",
            )
        missing_stats = [
            feature_id
            for feature_id in feature_ids
            if bool((self._percentile_table[feature_id] == 0).all())
        ]
        if missing_stats:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="unavailable",
                message=f"feature id(s) missing corpus percentile statistics: {missing_stats}",
            )

        prefix_end = min(target.char_end, len(request.document_text))
        if target.char_end <= target.char_start or prefix_end <= target.char_start:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="empty",
                message="target segment has no visible tokens",
            )

        probe_model = self._ensure_probe_model()
        prefix_text = request.document_text[:prefix_end]
        token_ids = _encode_without_special_tokens(probe_model.tokenizer, prefix_text)
        if not token_ids:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="empty",
                message="tokenizer returned no tokens",
            )

        char_offsets = token_to_char_offsets(probe_model.tokenizer, token_ids)
        if char_offsets[-1] != len(prefix_text):
            logger.warning(
                "Probe tokenizer decode length mismatch: offsets end {} != prefix len {}",
                char_offsets[-1],
                len(prefix_text),
            )

        n_ctx = probe_model.context_length
        if n_ctx is None and probe_model.model is not None:
            n_ctx = _model_context_length(probe_model.model)
        start_token = max(0, len(token_ids) - n_ctx) if n_ctx is not None else 0
        window_token_ids = token_ids[start_token:]
        if not window_token_ids:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="empty",
                message="context window clipped all tokens",
            )

        token_starts = t.tensor(char_offsets[start_token:-1], dtype=t.int64)
        token_ends = t.tensor(char_offsets[start_token + 1 :], dtype=t.int64)
        target_mask = _target_token_mask(
            token_starts,
            token_ends,
            request.segments,
            request.target_segment_index,
            prefix_end,
        )
        n_target_tokens = int(target_mask.sum().item())
        if n_target_tokens == 0:
            return FeatureHighlightResult(
                request_key=request.request_key,
                segment_idx=target.segment_idx,
                feature_id=request.feature_id,
                status="empty",
                message="target segment has no visible tokens in context window",
            )

        tokens_tensor = t.tensor(
            [window_token_ids], dtype=t.long, device=probe_model.device
        )
        with t.no_grad():
            acts = self._get_activations(probe_model, tokens_tensor).squeeze(0).float()
            post_by_feature: list[t.Tensor] = []
            for feature_id in feature_ids:
                raw = compute_raw_probe(
                    acts,
                    probe_model.sae.W_enc,
                    probe_model.sae.b_enc,
                    probe_model.sae.b_dec,
                    feature_id,
                    apply_b_dec=probe_model.apply_b_dec,
                )
                post_by_feature.append(
                    apply_sae_activation(raw, probe_model.sae, feature_id)
                )

        # A composite highlight is the pointwise maximum of its component
        # activations. Keep the winning component so percentile normalization
        # still uses that component's own corpus distribution.
        post_by_feature_tensor = t.stack(post_by_feature, dim=1)
        post, winning_columns = post_by_feature_tensor.max(dim=1)
        winning_feature_ids = t.tensor(
            feature_ids, dtype=t.int64, device=winning_columns.device
        )[winning_columns]

        mask_device = target_mask.to(post.device)
        target_post = post[mask_device].detach().cpu()
        winning_feature_ids = winning_feature_ids[mask_device].detach().cpu()
        percentiles = magnitude_to_percentile(
            winning_feature_ids,
            target_post,
            self._percentile_table,
        )
        levels = _percentiles_to_levels(target_post, percentiles)
        selected_starts = token_starts[target_mask]
        selected_ends = token_ends[target_mask]
        spans = _coalesce_token_highlights(
            token_starts=selected_starts,
            token_ends=selected_ends,
            levels=levels,
            percentiles=percentiles,
            activations=target_post,
        )

        positive = target_post > 0
        max_percentile = None
        max_activation = None
        if bool(positive.any()):
            max_percentile = float(percentiles[positive].max().item())
            max_activation = float(target_post[positive].max().item())

        per_feature_max_percentile: list[tuple[int, float]] = []
        per_feature_max_activation: list[tuple[int, float]] = []
        for column, feature_id in enumerate(feature_ids):
            feature_post = post_by_feature_tensor[:, column][mask_device].detach().cpu()
            feature_positive = feature_post > 0
            if not bool(feature_positive.any()):
                continue
            feature_ids_tensor = t.full(
                (n_target_tokens,), feature_id, dtype=t.int64
            )
            feature_percentiles = magnitude_to_percentile(
                feature_ids_tensor,
                feature_post,
                self._percentile_table,
            )
            per_feature_max_percentile.append(
                (feature_id, float(feature_percentiles[feature_positive].max().item()))
            )
            per_feature_max_activation.append(
                (feature_id, float(feature_post[feature_positive].max().item()))
            )

        return FeatureHighlightResult(
            request_key=request.request_key,
            segment_idx=target.segment_idx,
            feature_id=request.feature_id,
            status="ready",
            spans=spans,
            max_percentile=max_percentile,
            max_activation=max_activation,
            n_tokens=n_target_tokens,
            message=None,
            per_feature_max_percentile=tuple(per_feature_max_percentile),
            per_feature_max_activation=tuple(per_feature_max_activation),
        )
