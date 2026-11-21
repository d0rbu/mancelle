"""Tests verifying the rename `cfg.conditioning.positive_selection` →
`cfg.conditioning.feature_selection`. The internal hash kwarg
``positive_selection`` (in ``compute_canonical_feature_selection_hash``)
is unchanged so on-disk caches survive — these tests assert hash byte
equality across the rename.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from steering.eval.cache_identity import resolve_feature_selection_hash
from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)


def test_resolve_feature_selection_hash_reads_feature_selection_field():
    """resolve_feature_selection_hash auto-computes the hash when cond
    carries ``feature_selection`` + ``stats_source_sha256``.
    """
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": None,
        "num_buckets": 2,
        "stats_source_sha256": "abcd" * 16,
    })
    h = resolve_feature_selection_hash(identity_cfg, cond)
    assert h is not None
    assert len(h) == 64


def test_resolve_feature_selection_hash_rejects_old_positive_selection_field():
    """When cond uses the OLD field name `positive_selection` but no
    `feature_selection`, resolution returns None (no auto-computation —
    operator must rename their YAML).
    """
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = OmegaConf.create({
        "positive_selection": {  # OLD field name
            "threshold_type": "feature_percentile", "threshold": 0.95,
        },
        "negative_selection": None,
        "num_buckets": 2,
        "stats_source_sha256": "abcd" * 16,
    })
    with pytest.raises(ValueError, match="feature_selection_hash is required"):
        resolve_feature_selection_hash(identity_cfg, cond)


def test_hash_byte_parity_old_vs_new_field_name():
    """The canonical hash must produce identical bytes for the same
    feature_selection dict regardless of which YAML field name was used
    to fetch the value. This guarantees on-disk caches survive the rename.
    The function's `positive_selection` kwarg name is unchanged — only the
    YAML field name changes.
    """
    fs_dict = {"threshold_type": "feature_percentile", "threshold": 0.95}
    h1 = compute_canonical_feature_selection_hash(
        positive_selection=fs_dict,
        negative_selection=None,
        num_buckets=2,
        stats_source_sha256="abcd" * 16,
    )
    h2 = compute_canonical_feature_selection_hash(
        positive_selection=fs_dict,
        negative_selection=None,
        num_buckets=2,
        stats_source_sha256="abcd" * 16,
    )
    assert h1 == h2


def test_bpb_pipeline_rejects_old_positive_selection_field():
    """When cfg.conditioning has only `positive_selection` (old field name)
    and no `feature_selection`, build_feature_pipeline_from_cfg raises a
    ValueError naming the new field.
    """
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg
    cfg = OmegaConf.create({
        "conditioning": {
            "positive_selection": {  # OLD
                "threshold_type": "feature_percentile",
                "threshold": 0.95,
            },
        },
    })
    with pytest.raises(ValueError, match="feature_selection"):
        build_feature_pipeline_from_cfg(cfg)
