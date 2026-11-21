"""Feature prediction head loss computation.

Computes auxiliary loss for predicting the next segment's SAE feature activations
from the current segment's hidden states.

All loss functions are fully vectorized (no Python for-loops, no .item() calls)
for torch.compile compatibility. Prediction across groups is prevented via
seg_doc_id, the PREDICTION-GROUP id: a group boundary sits at every packed
document/conversation boundary and, for a steered SFT conversation, also at its
first assistant-turn segment.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from beartype import beartype

from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn as _FeaturePredLossFn,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossMode as _FeaturePredLossMode,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt


@beartype
@dataclass(frozen=True)
class FeaturePredictionTargetLayout:
    """Runtime-only identity and row map for feature-prediction targets.

    Original SAE IDs enter this boundary.  Compact row IDs leave it and never
    escape into artifacts, cache metadata, logs, or user-facing reports.  The
    layout is derived solely from the already-authoritative mask, cluster
    consolidation, and model vocabulary; it is intentionally not serialized as
    another identity descriptor.

    ``feature_vocab is None`` always means the literal full-width legacy layout.
    A non-``None`` vocabulary always means genuine compaction, matching the
    invariant owned by :class:`FeatureVocab`.

    The original and effective widths are strict positive integers.
    ``@beartype`` owns those unary contracts; :meth:`build` retains only the
    relationships among the mask, clusters, vocabulary, and derived row map.
    """

    n_sae_features: PositiveInt
    output_width: PositiveInt
    feature_mask: FeatureMask | None
    feature_clusters: FeatureClusterConsolidation | None
    feature_vocab: FeatureVocab | None
    original_to_output_row: torch.Tensor | None

    @classmethod
    def build(
        cls,
        *,
        n_sae_features: PositiveInt,
        feature_mask: FeatureMask | None,
        feature_clusters: FeatureClusterConsolidation | None,
        feature_vocab: FeatureVocab | None,
    ) -> "FeaturePredictionTargetLayout":
        """Validate authoritative state and derive the compact target map."""
        if feature_vocab is not None and (feature_vocab.n_sae_features != n_sae_features):
            raise ValueError(
                "feature_vocab.n_sae_features does not match the configured SAE "
                f"width: {feature_vocab.n_sae_features} != {n_sae_features}"
            )
        if feature_clusters is not None:
            if feature_clusters.n_sae_features != n_sae_features:
                raise ValueError(
                    "feature_clusters.n_sae_features does not match the configured "
                    f"SAE width: {feature_clusters.n_sae_features} != "
                    f"{n_sae_features}"
                )
            if feature_clusters.feature_mask != feature_mask:
                raise ValueError(
                    "feature_clusters was derived from a different FeatureMask than "
                    "the FP target layout"
                )

        expected_vocab = effective_feature_vocab(
            feature_clusters if feature_clusters is not None else feature_mask,
            n_sae_features,
        )
        if expected_vocab != feature_vocab:
            raise ValueError(
                "feature_vocab does not equal effective_feature_vocab(mask/clusters, "
                f"n_sae_features): supplied={feature_vocab!r}, "
                f"derived={expected_vocab!r}"
            )

        if feature_vocab is None:
            return cls(
                n_sae_features=n_sae_features,
                output_width=n_sae_features,
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
                feature_vocab=None,
                original_to_output_row=None,
            )

        # Start with the authoritative compact rows (representatives only).
        representative_to_row = feature_vocab.old_to_new_tensor()
        original_to_output_row = torch.full((n_sae_features,), -1, dtype=torch.int64)
        allowed = (
            feature_mask.allowed_universe_mask(n_sae_features)
            if feature_mask is not None
            else torch.ones(n_sae_features, dtype=torch.bool)
        )
        if feature_clusters is None:
            representatives = torch.arange(n_sae_features, dtype=torch.int64)
        else:
            representatives = feature_clusters.rep_of

        allowed_ids = allowed.nonzero(as_tuple=False).squeeze(1)
        allowed_rows = representative_to_row[representatives[allowed_ids]]
        if bool((allowed_rows < 0).any()):
            bad_id = int(allowed_ids[allowed_rows < 0][0].item())
            bad_rep = int(representatives[bad_id].item())
            raise ValueError(
                "mask-allowed FP source feature does not map to a vocabulary row: "
                f"original_id={bad_id}, representative={bad_rep}"
            )
        if bool((allowed_rows >= feature_vocab.size).any()):
            raise ValueError("FP target row map contains an out-of-range compact row")
        original_to_output_row[allowed_ids] = allowed_rows

        # The vocabulary itself must contain exactly the allowed representatives;
        # masked and consolidated-away IDs may map as sources but may not own rows.
        expected_row_ids = (
            (feature_clusters.cluster_allowed_universe if feature_clusters is not None else allowed)
            .nonzero(as_tuple=False)
            .squeeze(1)
        )
        if not torch.equal(expected_row_ids, feature_vocab.new_to_old_tensor()):
            raise ValueError(
                "feature_vocab rows are not exactly the mask-allowed, "
                "non-consolidated representative IDs"
            )

        return cls(
            n_sae_features=n_sae_features,
            output_width=feature_vocab.size,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
            feature_vocab=feature_vocab,
            original_to_output_row=original_to_output_row,
        )

    @property
    def is_compact(self) -> bool:
        return self.feature_vocab is not None

    @property
    def has_active_clusters(self) -> bool:
        return self.feature_clusters is not None and not (
            self.feature_clusters.is_identity_after_mask
        )


def segment_lengths(seg_map: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Token count per segment: ``(B, n_seg)`` float32 via ``scatter_add``.

    THE single implementation of the per-segment length idiom shared by every
    FP loss mode and by self-conditioning's rollout gating
    (``selfcond_training.continuation_mask``) — consolidated so the copies
    cannot drift (review finding #12).
    """
    B, T = seg_map.shape
    lengths = torch.zeros(B, n_seg, dtype=torch.float32, device=seg_map.device)
    lengths.scatter_add_(1, seg_map, torch.ones(B, T, dtype=torch.float32, device=seg_map.device))
    return lengths


class SourceSegmentMasks(NamedTuple):
    """Per-segment gating masks for next-segment feature prediction."""

    valid_source_seg: torch.Tensor  # (B, n_seg) bool — segment s may predict s+1
    seg_has_tokens: torch.Tensor  # (B, n_seg) bool
    target_seg_valid: torch.Tensor  # (B, n_seg) bool
    effective_target_lengths: torch.Tensor  # (B, n_seg) float32


def source_segment_masks(
    seg_lengths: torch.Tensor,
    seg_doc_id: torch.Tensor,
) -> SourceSegmentMasks:
    """The prediction-group gate shared by all four FP loss modes.

    ``valid_source_seg[b, s]`` is True iff segment ``s`` may predict ``s+1``:
    same prediction group, source has tokens, and the target is real. A group
    is one packed document/conversation, except that a steered SFT conversation
    starts a second group at its first assistant-turn segment. The
    ``seg_doc_id[:, :-1] >= 0`` guard filters padding slots (``-1``): without
    it, two adjacent padded slots both match as ``-1 == -1`` and the pair
    leaks into the loss if any real tokens got clamped into them.

    Self-conditioning's ``continuation_mask`` is the target-indexed shift of
    ``valid_source_seg`` — sharing this implementation keeps the rollout's
    slot gating provably aligned with the loss the head was trained under.
    """
    seg_has_tokens, target_seg_valid, effective_target_lengths = _target_segment_validity(
        seg_lengths, seg_doc_id
    )
    B, n_seg = seg_doc_id.shape
    valid_source_seg = torch.zeros(B, n_seg, dtype=torch.bool, device=seg_doc_id.device)
    if n_seg > 1:
        valid_source_seg[:, :-1] = (
            (seg_doc_id[:, :-1] == seg_doc_id[:, 1:])
            & (seg_doc_id[:, :-1] >= 0)
            & seg_has_tokens[:, :-1]
            & target_seg_valid[:, 1:]
        )
    return SourceSegmentMasks(
        valid_source_seg=valid_source_seg,
        seg_has_tokens=seg_has_tokens,
        target_seg_valid=target_seg_valid,
        effective_target_lengths=effective_target_lengths,
    )


def pool_segment_amax(values: torch.Tensor, seg_map: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Per-segment per-feature max: ``(B, n_seg, n_feat)`` with ``-inf`` empties.

    Returns the RAW pooled tensor; each consumer applies its OWN sentinel
    handling, and the two deliberately differ: the FP loss zeroes every
    infinity (``.isinf()`` — pre-existing behavior, padded segments are
    weight-masked anyway), while self-conditioning zeroes ONLY ``-inf``
    (``isneginf``) so a diverged ``+inf`` head logit propagates into its
    finiteness guard instead of silently ranking mid-scale.
    """
    B, T, n_feat = values.shape
    pooled = torch.full(
        (B, n_seg, n_feat),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )
    seg_idx_3d = seg_map.unsqueeze(-1).expand(B, T, n_feat)
    return pooled.scatter_reduce(1, seg_idx_3d, values, reduce="amax", include_self=True)


def last_token_mask(seg_map: torch.Tensor) -> torch.Tensor:
    """True at each segment's final token position (``(B, T)`` bool)."""
    is_last = torch.ones(*seg_map.shape, dtype=torch.bool, device=seg_map.device)
    is_last[:, :-1] = seg_map[:, :-1] != seg_map[:, 1:]
    return is_last


def pool_fp_logits(
    fp_logits: torch.Tensor,
    mode: _FeaturePredLossMode,
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

    if mode == _FeaturePredLossMode.MAX_POOL:
        return fp_logits.amax(dim=1)
    if mode == _FeaturePredLossMode.LAST_TOKEN:
        return fp_logits[:, -1, :]
    if mode == _FeaturePredLossMode.FULL_SEGMENT:
        return fp_logits.mean(dim=1)
    if mode == _FeaturePredLossMode.BEST_TOKEN:
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


def _compute_per_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: _FeaturePredLossFn,
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
    if loss_fn == _FeaturePredLossFn.BCE:
        per_feature = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
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
    loss_mode: _FeaturePredLossMode,
    loss_fn: _FeaturePredLossFn,
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
        seg_doc_id: (B, n_seg) int64 FP prediction-group id per segment; -1
            for padding/empty segments. Segment ``s`` may predict ``s+1`` only
            within one group (one packed conversation, split again at a
            steered conversation's first assistant-turn segment).
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
        allowed = feature_mask.allowed_universe_mask(n_sae).to(
            device=logits.device, dtype=torch.float32
        )

    if loss_mode == _FeaturePredLossMode.FULL_SEGMENT:
        return _full_segment_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)
    if loss_mode == _FeaturePredLossMode.LAST_TOKEN:
        return _last_token_loss(logits, seg_map, targets, seg_doc_id, loss_fn, allowed)
    if loss_mode == _FeaturePredLossMode.MAX_POOL:
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

    valid_source_seg = source_segment_masks(
        segment_lengths(seg_map, n_seg), seg_doc_id
    ).valid_source_seg

    # Map segment-level validity to token-level
    source_mask = valid_source_seg.gather(1, seg_map)  # (B, T)
    n_source_tokens = source_mask.sum().clamp(min=1)  # avoid div-by-zero, no graph break

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae))

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

    masks = source_segment_masks(segment_lengths(seg_map, n_seg), seg_doc_id)
    valid_source_seg = masks.valid_source_seg
    effective_target_lengths = masks.effective_target_lengths

    valid_source_at_token = valid_source_seg.gather(1, seg_map)
    source_last_tokens = last_token_mask(seg_map) & valid_source_at_token  # (B, T)

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae))

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
    return (per_token_loss * per_token_weight * source_last_tokens).sum()


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

    masks = source_segment_masks(segment_lengths(seg_map, n_seg), seg_doc_id)
    valid_source_seg = masks.valid_source_seg
    effective_target_lengths = masks.effective_target_lengths

    valid_source_at_token = valid_source_seg.gather(1, seg_map)

    # For each token, get the target (next segment's features)
    target_seg_indices = (seg_map + 1).clamp(max=n_seg - 1)
    per_token_targets = targets.gather(1, target_seg_indices.unsqueeze(-1).expand(B, T, n_sae))

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
    seg_min_loss = torch.full((B, n_seg), math.inf, dtype=per_token_loss.dtype, device=device)
    seg_min_loss = seg_min_loss.scatter_reduce(
        1, seg_map, source_loss, reduce="amin", include_self=True
    )

    # Replace inf with 0 to avoid inf * 0 = NaN
    safe_min_loss = torch.where(seg_min_loss.isinf(), torch.zeros_like(seg_min_loss), seg_min_loss)

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

    masks = source_segment_masks(segment_lengths(seg_map, n_seg), seg_doc_id)
    valid_source_seg = masks.valid_source_seg
    effective_target_lengths = masks.effective_target_lengths

    seg_max_logits = pool_segment_amax(logits, seg_map, n_seg)
    # Replace every infinity with 0 for the loss computation; weights will
    # mask padded segments out anyway. Deliberately BOTH signs (`.isinf()`,
    # unlike self-conditioning's isneginf sentinel) — pre-existing loss
    # behavior preserved bit-for-bit.
    safe_seg_max = torch.where(
        seg_max_logits.isinf(),
        torch.zeros_like(seg_max_logits),
        seg_max_logits,
    )

    # Targets for source segment s come from segment s+1.
    target_seg_indices = (torch.arange(n_seg, device=device).unsqueeze(0).expand(B, -1) + 1).clamp(
        max=n_seg - 1
    )
    per_seg_targets = targets.gather(
        1,
        target_seg_indices.unsqueeze(-1).expand(B, n_seg, n_sae),
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
