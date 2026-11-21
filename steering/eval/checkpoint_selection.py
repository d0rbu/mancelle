"""Shared parsing for eval checkpoint feature-selection paths."""

from __future__ import annotations

from omegaconf import OmegaConf

from steering.conditional.sampling import (
    FeatureSelection,
    SamplingSelection,
    ThresholdSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
)


def parse_checkpoint_feature_selection(
    raw,
    *,
    live_context: str,
) -> FeatureSelection | None:
    """Parse a raw Hydra block and reject offline-only live path policies."""
    if raw is None:
        return None
    if isinstance(raw, (SamplingSelection, ThresholdSelection)):
        return raw
    plain = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else raw
    if not isinstance(plain, dict):
        raise TypeError(
            "feature_selection must be a mapping with either 'distribution' "
            "or 'threshold_type' keys"
        )
    parsed = parse_feature_selection(plain)
    assert_no_gradient_ranking(parsed, live_context)
    return parsed
