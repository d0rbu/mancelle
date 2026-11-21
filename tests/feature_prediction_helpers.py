"""Test-data helpers for feature-prediction loss inputs."""

import torch as t


def dense_targets_from_sparse(
    feature_ids: t.Tensor,
    percentiles: t.Tensor,
    mask: t.Tensor,
    n_features: int,
) -> t.Tensor:
    """Scatter sparse per-segment values into dense synthetic test targets."""
    n_segments = feature_ids.shape[0]
    targets = t.zeros(
        n_segments,
        n_features,
        dtype=t.float32,
        device=feature_ids.device,
    )
    segment_indices = (
        t.arange(n_segments, device=feature_ids.device).unsqueeze(1).expand_as(feature_ids)
    )
    targets[segment_indices[mask], feature_ids[mask].long()] = percentiles[mask].float()
    return targets
