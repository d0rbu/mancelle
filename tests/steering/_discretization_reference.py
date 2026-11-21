"""Reference boundary construction used only by discretization tests."""

import torch as t


def compute_bucket_boundaries(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    num_buckets: int,
) -> t.Tensor:
    """Compute exact per-feature quantiles directly from raw activations."""
    assert num_buckets >= 2, "Need at least 2 buckets"
    assert len(feature_ids) == len(feature_mags), (
        "feature_ids and feature_mags must have same length"
    )
    n_boundaries = num_buckets - 1

    if len(feature_ids) == 0:
        return t.zeros(0, n_boundaries, dtype=t.float64)

    max_id = int(feature_ids.max().item())
    boundaries = t.zeros(max_id + 1, n_boundaries, dtype=t.float64)
    percentiles = t.linspace(
        1.0 / num_buckets,
        1.0 - 1.0 / num_buckets,
        n_boundaries,
        dtype=t.float64,
    )

    sort_idx = feature_ids.argsort()
    sorted_ids = feature_ids[sort_idx]
    sorted_mags = feature_mags[sort_idx].to(t.float64)
    unique_ids, group_counts = t.unique_consecutive(sorted_ids, return_counts=True)
    splits = group_counts.cumsum(0)
    starts = t.cat([t.zeros(1, dtype=splits.dtype), splits[:-1]])

    fids = unique_ids.long()
    starts_list = starts.tolist()
    splits_list = splits.tolist()
    for i in range(len(unique_ids)):
        group_mags = sorted_mags[starts_list[i] : splits_list[i]]
        if len(group_mags) == 1:
            boundaries[fids[i]] = group_mags[0].expand(n_boundaries)
        else:
            boundaries[fids[i]] = t.quantile(group_mags, percentiles)

    return boundaries
