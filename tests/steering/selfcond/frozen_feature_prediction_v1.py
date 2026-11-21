"""FROZEN verbatim copy of steering/conditional/encoder_decoder/feature_prediction.py

Captured at 0c28b16d (bit-identical to origin/main 2c0dba78) BEFORE the
primitive-consolidation refactor. This is the known-working reference the
refactored implementation must match BIT-EXACTLY. Never edit this file to
make a test pass; it exists so the production module can change while the
behavior provably cannot.
"""


import math

import torch
import torch.nn.functional as F

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.feature_mask import FeatureMask


def pool_fp_logits(
    fp_logits: torch.Tensor,
    mode: FeaturePredLossMode,
) -> torch.Tensor:
    """Pool feature-prediction head logits over a time span to produce a
    single per-feature prediction vector.

    Used by self-conditioned generation: after K decoded tokens, this pools
    the K (n_sae,) per-token logits into one (n_sae,) prediction that
    drives the encoder for the NEXT segment.

    Modes mirror the training-time loss aggregation:

    - :attr:`FeaturePredLossMode.MAX_POOL`: per-feature max across time.
    - :attr:`FeaturePredLossMode.LAST_TOKEN`: read position ``-1``.
    - :attr:`FeaturePredLossMode.FULL_SEGMENT`: average over time.
    - :attr:`FeaturePredLossMode.BEST_TOKEN` is **forbidden** here. The
      "best" token is selected by argmin over the per-token loss against
      the *target* segment's features — that target is unavailable at
      inference time, so this mode cannot run during self-conditioned
      generation. Operators training a model intended for self-cond eval
      should keep the default (``MAX_POOL``) or pick ``LAST_TOKEN`` /
      ``FULL_SEGMENT``.

    Args:
        fp_logits: ``(B, T, n_sae)`` raw logits from the prediction head.
        mode: Pooling mode.

    Returns:
        ``(B, n_sae)`` pooled logits.
    """
    if fp_logits.dim() != 3:
        raise ValueError(
            f"pool_fp_logits expects a 3D (B, T, n_sae) tensor, got shape {tuple(fp_logits.shape)}"
        )
    if fp_logits.shape[1] == 0:
        raise ValueError("pool_fp_logits requires T >= 1; got empty time dimension")

    if mode == FeaturePredLossMode.MAX_POOL:
        return fp_logits.amax(dim=1)
    if mode == FeaturePredLossMode.LAST_TOKEN:
        return fp_logits[:, -1, :]
    if mode == FeaturePredLossMode.FULL_SEGMENT:
        return fp_logits.mean(dim=1)
    if mode == FeaturePredLossMode.BEST_TOKEN:
        raise ValueError(
            "BEST_TOKEN pooling is not supported at inference: it requires "
            "the next segment's target features to identify the 'best' "
            "token. Use MAX_POOL, LAST_TOKEN, or FULL_SEGMENT for "
            "self-conditioned generation."
        )
    raise ValueError(f"Unknown pooling mode: {mode!r}")


def resolve_layer_index(layer: int, n_layer: int) -> int:
    """Resolve Python-style layer index to absolute index.

    Args:
        layer: Python-style index (-1=last, 0=first, etc.)
        n_layer: Total number of layers.

    Returns:
        Absolute layer index in [0, n_layer).
    """
    resolved = layer if layer >= 0 else layer + n_layer
    assert 0 <= resolved < n_layer, (
        f"Layer index {layer} out of range for {n_layer} layers (resolved to {resolved})"
    )
    return resolved


def build_dense_targets(
    enc_fids: torch.Tensor,
    enc_percentiles: torch.Tensor,
    enc_mask: torch.Tensor,
    n_sae_features: int,
    *,
    feature_mask: FeatureMask | None = None,
) -> torch.Tensor:
    """Build dense target vectors from sparse feature data.

    Args:
        enc_fids: (n_seg, F_max) int64 feature IDs.
        enc_percentiles: (n_seg, F_max) float32 percentile values.
        enc_mask: (n_seg, F_max) bool mask (True=real feature).
        n_sae_features: Total number of SAE features.
        feature_mask: Optional feature blacklist/whitelist. When set, columns
            for masked-out features are zeroed in the returned dense tensor so
            those dims are excluded from the FP objective. ``None`` ⇒ no-op
            (I5: byte-identical to the pre-mask code path).

    Returns:
        (n_seg, n_sae_features) float32 dense target vectors.
    """
    n_seg = enc_fids.shape[0]
    targets = torch.zeros(
        n_seg, n_sae_features, dtype=torch.float32, device=enc_fids.device
    )

    # Scatter percentile values into dense vector at feature ID positions
    valid_fids = enc_fids[enc_mask]  # (N_active,)
    valid_pcts = enc_percentiles[enc_mask].float()  # (N_active,)

    # Build segment indices for scatter
    seg_indices = (
        torch.arange(n_seg, device=enc_fids.device).unsqueeze(1).expand_as(enc_fids)
    )
    valid_seg_indices = seg_indices[enc_mask]  # (N_active,)

    targets[valid_seg_indices, valid_fids.long()] = valid_pcts

    # Gate: zero out columns for masked features so they cannot enter the FP
    # objective. Static multiply — no grad, no RNG.  None ⇒ I5-safe no-op.
    # Gate is always float32 (targets is float32; explicit cast for clarity
    # and forward-safety in case targets.dtype ever changes).
    if feature_mask is not None:
        allowed = feature_mask.allowed_universe_mask(n_sae_features).to(dtype=torch.float32, device=targets.device)
        targets = targets * allowed

    return targets


def _compute_per_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: FeaturePredLossFn,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token feature prediction loss (mean over feature dim).

    Args:
        logits: (N, n_sae_features) raw predictions.
        targets: (N, n_sae_features) target values in [0, 1].
        loss_fn: BCE or MSE.
        allowed: Optional bool[n_sae_features] gate. When provided, masked dims
            are excluded from BOTH the numerator and the denominator of the
            mean, so they contribute exactly zero loss AND zero gradient.
            Critically for BCE: binary_cross_entropy_with_logits(0, 0) = log(2)
            ≠ 0 and has nonzero gradient — zeroing the target/logit is NOT
            sufficient; the dim must be excluded from the reduction entirely.
            ``None`` ⇒ I5-safe unchanged code path (byte-identical).

    Returns:
        (N,) per-token scalar losses.
    """
    if loss_fn == FeaturePredLossFn.BCE:
        per_feature = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
    else:
        per_feature = F.mse_loss(torch.sigmoid(logits), targets, reduction="none")

    if allowed is None:
        # I5: unchanged code path — mean over all feature dims
        return per_feature.mean(dim=-1)  # (N,)

    # Mask: zero contributions from disallowed dims, divide by allowed count
    # so the mean is over allowed dims only.  allowed is bool[n_sae], broadcast
    # over the N token/segment dimension.
    # all-masked edge (e.g. whitelist excludes every active dim): clamp keeps the mean well-defined → loss is exactly 0
    n_allowed = allowed.sum().clamp(min=1)
    return (per_feature * allowed).sum(dim=-1) / n_allowed


def _target_segment_validity(
    seg_lengths: torch.Tensor,
    seg_doc_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source/target segment masks plus target lengths for FP loss.

    SFT can append a prediction-target-only segment that has no decoder token
    positions but is still a real segment with feature targets. Padding keeps
    ``seg_doc_id=-1``. Treat non-padding target-only segments as valid targets
    and give them unit length so weighted modes do not silently drop them.
    """
    seg_has_tokens = seg_lengths > 0
    target_is_real = seg_doc_id >= 0
    target_seg_valid = seg_has_tokens | target_is_real
    effective_target_lengths = torch.where(
        seg_has_tokens,
        seg_lengths,
        target_is_real.to(dtype=seg_lengths.dtype),
    )
    return seg_has_tokens, target_seg_valid, effective_target_lengths


def compute_feature_pred_loss(
    logits: torch.Tensor,  # (B, T, n_sae)
    seg_map: torch.Tensor,  # (B, T)
    targets: torch.Tensor,  # (B, n_seg, n_sae)
    seg_doc_id: torch.Tensor,  # (B, n_seg)
    loss_mode: FeaturePredLossMode,
    loss_fn: FeaturePredLossFn,
    *,
    feature_mask: FeatureMask | None = None,
) -> torch.Tensor:
    """Compute feature prediction loss.

    Each segment predicts the next segment's feature activations, but only
    when consecutive segments belong to the same document (determined by
    seg_doc_id).

    Args:
        logits: (B, T, n_sae_features) raw head output.
        seg_map: (B, T) int64 segment index per position.
        targets: (B, n_seg, n_sae_features) dense target vectors.
        seg_doc_id: (B, n_seg) int64 document index per segment. Padded
            segments use -1.
        loss_mode: FULL_SEGMENT, LAST_TOKEN, or BEST_TOKEN.
        loss_fn: BCE or MSE.
        feature_mask: Optional feature blacklist/whitelist. When set, masked
            dims are excluded from the loss mean (numerator AND denominator)
            so they contribute exactly zero loss and zero gradient. Critical
            for BCE correctness: BCE(logit=0, target=0) = log(2) ≠ 0, so
            zeroing targets is insufficient; dims must be dropped from the
            reduction. ``None`` ⇒ I5-safe no-op (byte-identical).

    Returns:
        Scalar loss tensor.
    """
    n_sae = logits.shape[-1]
    allowed: torch.Tensor | None = None
    if feature_mask is not None:
        # Compute the gate in float32 regardless of logits.dtype.
        # Rationale: allowed is a 0/1 integer mask; its sum is the denominator
        # of the masked mean.  Under fp16, sum over >=65504 ones overflows to
        # inf → denominator is inf → loss = 0.0 and grad = 0.0 silently.
        # bf16 is safe from hard overflow but has only 7 mantissa bits so the
        # integer count can be rounded for large widths.  float32 is exact for
        # integer counts up to 2^24 (~16 M), which covers all practical SAE
        # widths.  The None path is untouched (I5-safe).
        allowed = feature_mask.allowed_universe_mask(n_sae).to(device=logits.device, dtype=torch.float32)

    if loss_mode == FeaturePredLossMode.FULL_SEGMENT:
        return _full_segment_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)
    if loss_mode == FeaturePredLossMode.LAST_TOKEN:
        return _last_token_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)
    if loss_mode == FeaturePredLossMode.MAX_POOL:
        return _max_pool_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)
    return _best_token_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)


def _full_segment_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed=None):
    """Full segment loss: mean over all source tokens globally.

    Fully vectorized -- no Python for-loops, no .item() calls.

    ``allowed``: bool[n_sae] or None. When set, masked feature dims are
    excluded from the per-token loss mean (numerator + denominator) so they
    contribute zero loss and zero gradient. None ⇒ I5-safe unchanged path.
    """
    B, T, n_sae = logits.shape
    n_seg = targets.shape[1]
    device = logits.device

    # Compute segment lengths via scatter_add
    seg_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    seg_lengths.scatter_add_(
        1, seg_map, torch.ones(B, T, dtype=torch.float32, device=device)
    )
    seg_has_tokens, target_seg_valid, _effective_target_lengths = (
        _target_segment_validity(seg_lengths, seg_doc_id)
    )

    # valid_source_seg[b, s] = True iff segment s can predict segment s+1
    # (same document, source has tokens, and target is real)
    # The `seg_doc_id[:, :-1] >= 0` guard filters padding slots (seg_doc_id=-1):
    # without it, two adjacent padded slots both match as -1 == -1 and the
    # pair leaks into the loss if any real tokens got clamped into them.
    valid_source_seg = torch.zeros(B, n_seg, dtype=torch.bool, device=device)
    if n_seg > 1:
        valid_source_seg[:, :-1] = (
            (seg_doc_id[:, :-1] == seg_doc_id[:, 1:])
            & (seg_doc_id[:, :-1] >= 0)
            & seg_has_tokens[:, :-1]
            & target_seg_valid[:, 1:]
        )

    # Map segment-level validity to token-level
    source_mask = valid_source_seg.gather(1, seg_map)  # (B, T)
    n_source_tokens = source_mask.sum().clamp(
        min=1
    )  # avoid div-by-zero, no graph break

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(
        1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae)
    )

    # Compute per-token loss
    per_token_loss = _compute_per_token_loss(
        logits.reshape(-1, n_sae),
        per_token_targets.reshape(-1, n_sae),
        loss_fn,
        allowed,
    ).reshape(B, T)

    return (per_token_loss * source_mask).sum() / n_source_tokens


def _last_token_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed=None):
    """Last token loss: only last token per source segment, weighted by next seg length.

    Fully vectorized -- no Python for-loops, no .item() calls.

    ``allowed``: bool[n_sae] or None. When set, masked feature dims are
    excluded from the per-token loss mean (numerator + denominator) so they
    contribute zero loss and zero gradient. None ⇒ I5-safe unchanged path.
    """
    B, T, n_sae = logits.shape
    n_seg = targets.shape[1]
    device = logits.device

    # Compute segment lengths via scatter_add
    seg_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    seg_lengths.scatter_add_(
        1, seg_map, torch.ones(B, T, dtype=torch.float32, device=device)
    )
    seg_has_tokens, target_seg_valid, effective_target_lengths = (
        _target_segment_validity(seg_lengths, seg_doc_id)
    )

    # valid_source_seg[b, s] = True iff segment s can predict segment s+1
    # The `seg_doc_id[:, :-1] >= 0` guard filters padding slots (seg_doc_id=-1):
    # without it, two adjacent padded slots both match as -1 == -1 and the
    # pair leaks into the loss if any real tokens got clamped into them.
    valid_source_seg = torch.zeros(B, n_seg, dtype=torch.bool, device=device)
    if n_seg > 1:
        valid_source_seg[:, :-1] = (
            (seg_doc_id[:, :-1] == seg_doc_id[:, 1:])
            & (seg_doc_id[:, :-1] >= 0)
            & seg_has_tokens[:, :-1]
            & target_seg_valid[:, 1:]
        )

    # Last token detection: token where segment changes or end of sequence
    is_last = torch.ones(B, T, dtype=torch.bool, device=device)
    is_last[:, :-1] = seg_map[:, :-1] != seg_map[:, 1:]

    valid_source_at_token = valid_source_seg.gather(1, seg_map)
    last_token_mask = is_last & valid_source_at_token  # (B, T)

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(
        1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae)
    )

    # Compute per-token loss
    per_token_loss = _compute_per_token_loss(
        logits.reshape(-1, n_sae),
        per_token_targets.reshape(-1, n_sae),
        loss_fn,
        allowed,
    ).reshape(B, T)

    # Per-token weight: next_seg_length / total_target_tokens
    next_seg_lengths = effective_target_lengths.gather(1, target_seg_indices)
    valid_next_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    if n_seg > 1:
        valid_next_lengths[:, :-1] = torch.where(
            valid_source_seg[:, :-1],
            effective_target_lengths[:, 1:],
            torch.zeros_like(effective_target_lengths[:, 1:]),
        )
    total_target_tokens = valid_next_lengths.sum().clamp(min=1.0)

    per_token_weight = next_seg_lengths / total_target_tokens
    return (per_token_loss * per_token_weight * last_token_mask).sum()


def _best_token_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed=None):
    """Best token loss: min-loss token per source segment, weighted by next seg length.

    Uses scatter_reduce amin for per-segment min -- autograd flows through
    the min element. Fully vectorized -- no Python for-loops, no .item() calls.

    ``allowed``: bool[n_sae] or None. When set, masked feature dims are
    excluded from the per-token loss mean (numerator + denominator). This
    ensures: (1) masked dims contribute zero loss and zero gradient, and (2)
    token selection (argmin per segment) is determined only by allowed dims
    so the best-token choice is not skewed by masked dimensions.
    None ⇒ I5-safe unchanged path.
    """
    B, T, n_sae = logits.shape
    n_seg = targets.shape[1]
    device = logits.device

    # Compute segment lengths via scatter_add
    seg_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    seg_lengths.scatter_add_(
        1, seg_map, torch.ones(B, T, dtype=torch.float32, device=device)
    )
    seg_has_tokens, target_seg_valid, effective_target_lengths = (
        _target_segment_validity(seg_lengths, seg_doc_id)
    )

    # valid_source_seg[b, s] = True iff segment s can predict segment s+1
    # The `seg_doc_id[:, :-1] >= 0` guard filters padding slots (seg_doc_id=-1):
    # without it, two adjacent padded slots both match as -1 == -1 and the
    # pair leaks into the loss if any real tokens got clamped into them.
    valid_source_seg = torch.zeros(B, n_seg, dtype=torch.bool, device=device)
    if n_seg > 1:
        valid_source_seg[:, :-1] = (
            (seg_doc_id[:, :-1] == seg_doc_id[:, 1:])
            & (seg_doc_id[:, :-1] >= 0)
            & seg_has_tokens[:, :-1]
            & target_seg_valid[:, 1:]
        )

    valid_source_at_token = valid_source_seg.gather(1, seg_map)

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(
        1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae)
    )

    # Compute per-token loss (with allowed gate so token selection uses only
    # allowed dims — prevents masked dims from skewing the best-token choice)
    per_token_loss = _compute_per_token_loss(
        logits.reshape(-1, n_sae),
        per_token_targets.reshape(-1, n_sae),
        loss_fn,
        allowed,
    ).reshape(B, T)

    # For non-source tokens, set loss to +inf so they never win argmin
    source_loss = torch.where(
        valid_source_at_token, per_token_loss, torch.full_like(per_token_loss, math.inf)
    )

    # Per-segment min loss via scatter_reduce (autograd flows through min element)
    seg_min_loss = torch.full(
        (B, n_seg), math.inf, dtype=per_token_loss.dtype, device=device
    )
    seg_min_loss = seg_min_loss.scatter_reduce(
        1, seg_map, source_loss, reduce="amin", include_self=True
    )

    # Replace inf with 0 to avoid inf * 0 = NaN
    safe_min_loss = torch.where(
        seg_min_loss.isinf(), torch.zeros_like(seg_min_loss), seg_min_loss
    )

    # Weights per segment: next_seg_length / total_target_tokens
    valid_next_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    if n_seg > 1:
        valid_next_lengths[:, :-1] = torch.where(
            valid_source_seg[:, :-1],
            effective_target_lengths[:, 1:],
            torch.zeros_like(effective_target_lengths[:, 1:]),
        )
    total_target_tokens = valid_next_lengths.sum().clamp(min=1.0)

    seg_weights = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    if n_seg > 1:
        seg_weights[:, :-1] = torch.where(
            valid_source_seg[:, :-1],
            effective_target_lengths[:, 1:] / total_target_tokens,
            torch.zeros_like(effective_target_lengths[:, 1:]),
        )

    return (safe_min_loss * seg_weights).sum()


def _max_pool_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed=None):
    """Max-pool loss: per-feature max of logits across segment positions,
    then BCE/MSE against the next segment's target.

    For each source segment ``s``, computes ``head_logits = max_t(logits[t])``
    over tokens ``t`` belonging to segment ``s``, then evaluates the
    per-feature loss (BCE or MSE on sigmoid) against ``targets[s+1]``.
    Per-segment losses are weighted by the next segment's length so the
    aggregation matches ``LAST_TOKEN`` and ``BEST_TOKEN`` weighting
    semantics. Autograd flows through the max element via
    ``scatter_reduce(reduce="amax")``.

    Fully vectorized -- no Python for-loops, no ``.item()`` calls.

    ``allowed``: bool[n_sae] or None. When set, masked feature dims are
    excluded from the per-segment loss mean (numerator + denominator) so they
    contribute zero loss and zero gradient. None ⇒ I5-safe unchanged path.
    """
    B, T, n_sae = logits.shape
    n_seg = targets.shape[1]
    device = logits.device

    seg_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    seg_lengths.scatter_add_(
        1, seg_map, torch.ones(B, T, dtype=torch.float32, device=device)
    )
    seg_has_tokens, target_seg_valid, effective_target_lengths = (
        _target_segment_validity(seg_lengths, seg_doc_id)
    )

    valid_source_seg = torch.zeros(B, n_seg, dtype=torch.bool, device=device)
    if n_seg > 1:
        valid_source_seg[:, :-1] = (
            (seg_doc_id[:, :-1] == seg_doc_id[:, 1:])
            & (seg_doc_id[:, :-1] >= 0)
            & seg_has_tokens[:, :-1]
            & target_seg_valid[:, 1:]
        )

    # Per-segment max-pool of logits along the time dim. Initialise with
    # -inf so segments with no tokens stay -inf (and are masked out later).
    seg_max_logits = torch.full(
        (B, n_seg, n_sae), float("-inf"), dtype=logits.dtype, device=device,
    )
    seg_idx_3d = seg_map.unsqueeze(-1).expand(B, T, n_sae)
    seg_max_logits = seg_max_logits.scatter_reduce(
        1, seg_idx_3d, logits, reduce="amax", include_self=True,
    )
    # Replace -inf with 0 for the loss computation; weights will mask
    # padded segments out anyway.
    safe_seg_max = torch.where(
        seg_max_logits.isinf(),
        torch.zeros_like(seg_max_logits),
        seg_max_logits,
    )

    # Targets for source segment s come from segment s+1.
    target_seg_indices = (
        torch.arange(n_seg, device=device).unsqueeze(0).expand(B, -1) + 1
    ).clamp(max=n_seg - 1)
    per_seg_targets = targets.gather(
        1, target_seg_indices.unsqueeze(-1).expand(B, n_seg, n_sae),
    )

    per_seg_loss = _compute_per_token_loss(
        safe_seg_max.reshape(-1, n_sae),
        per_seg_targets.reshape(-1, n_sae),
        loss_fn,
        allowed,
    ).reshape(B, n_seg)

    # Weight per source segment by next segment's length, normalised so
    # totals match LAST_TOKEN/BEST_TOKEN aggregation.
    valid_next_lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    if n_seg > 1:
        valid_next_lengths[:, :-1] = torch.where(
            valid_source_seg[:, :-1],
            effective_target_lengths[:, 1:],
            torch.zeros_like(effective_target_lengths[:, 1:]),
        )
    total_target_tokens = valid_next_lengths.sum().clamp(min=1.0)

    seg_weights = torch.zeros(B, n_seg, dtype=torch.float32, device=device)
    if n_seg > 1:
        seg_weights[:, :-1] = torch.where(
            valid_source_seg[:, :-1],
            effective_target_lengths[:, 1:] / total_target_tokens,
            torch.zeros_like(effective_target_lengths[:, 1:]),
        )

    return (per_seg_loss * seg_weights).sum()
