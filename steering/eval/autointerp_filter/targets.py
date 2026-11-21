"""Target resolution for autointerp filtering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from beartype import beartype

from steering.conditional.feature_mask import resolve_feature_mask
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig
from steering.eval.autointerp_filter.types import (
    AutoInterpConfigError,
    ScoringTarget,
    TargetKind,
)
from steering.refined_types import (
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntListLike,
)


def _target_key(feature_id: int) -> str:
    return f"feature:{feature_id}"


def _display_name(feature_id: int) -> str:
    return f"feature_{feature_id}"


def target_cache_payload(target: ScoringTarget) -> dict:
    """Canonical target identity payload used by cache/report metadata."""
    return {
        "kind": target.kind.value,
        "feature_ids": list(target.feature_ids),
        "target_key": target.target_key,
    }


def target_payload_hash(target: ScoringTarget) -> str:
    """Stable hash of a target's canonical identity."""
    payload = json.dumps(
        target_cache_payload(target), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@beartype
def _allowed_feature_ids(
    feature_ids: UniqueNonEmptyNonNegativeIntListLike | None,
    feature_mask: dict | None,
    n_sae_features: PositiveIntLike,
) -> set[int]:
    if feature_ids is None:
        allowed = set(range(n_sae_features))
    else:
        for fid in feature_ids:
            if fid >= n_sae_features:
                raise AutoInterpConfigError(
                    f"feature_id {fid} out of range [0, {n_sae_features})"
                )
        allowed = set(feature_ids)
    if feature_mask is None:
        return allowed
    mask = resolve_feature_mask({"feature_mask": feature_mask}, n_sae_features)
    if mask is None:
        return allowed
    mask_allowed = mask.allowed_universe_mask(n_sae_features)
    return {fid for fid in allowed if bool(mask_allowed[fid].item())}


def _make_target(feature_id: int) -> ScoringTarget:
    key = _target_key(feature_id)
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=key,
        display_name=_display_name(feature_id),
    )


def resolve_targets(
    cfg: AutoInterpFilterConfig,
    *,
    n_sae_features: int,
    feature_dir: Path,
) -> list[ScoringTarget]:
    """Resolve sorted singleton feature targets after optional input masking."""
    del feature_dir
    allowed = _allowed_feature_ids(cfg.feature_ids, cfg.feature_mask, n_sae_features)
    targets = [_make_target(fid) for fid in sorted(allowed)]
    if cfg.max_targets is not None:
        targets = targets[: cfg.max_targets]
    return targets
