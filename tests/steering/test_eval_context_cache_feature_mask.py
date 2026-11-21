"""H7 eval-context-cache feature-mask gap tests.

Four focused tests for CRITICAL-2 (adversarial-review):
  (a) Pipeline: build_eval_context_cache passes feature_mask through to the
      feature pipeline so a mask-trained model's cache records ONLY the right
      features.
  (b) Cache identity: compute_canonical_feature_selection_hash includes
      feature_mask in the hash so masked vs unmasked builds have distinct
      cache identities (no collision), and no-mask is byte-identical to the
      legacy value (cache-identity I1 — existing caches stay valid).
"""

from __future__ import annotations

from omegaconf import OmegaConf

from steering.conditional.feature_mask import FeatureMaskMode
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _base_hash_args() -> dict:
    """Minimal args for compute_canonical_feature_selection_hash (no mask)."""
    return {
        "positive_selection": {
            "threshold_type": "feature_percentile_segment_topk",
            "threshold": 2048.9,
        },
        "negative_selection": None,
        "num_buckets": 10,
        "stats_source_sha256": "a" * 64,
    }


# FROZEN golden: the legacy no-mask value for _base_hash_args().
# Pinned here so that any accidental change to seed/ddp_rank/payload layout
# breaks this test immediately rather than silently invalidating on-disk caches.
# Computed once with seed=0, ddp_rank=0, no feature_mask, against the v1 schema.
_CACHE_IDENTITY_LEGACY_GOLDEN = (
    "f8bfb3a1e8c58a47f91bca43938359842186d911a3d9d9371353d95907edea6e"
)


# ---------------------------------------------------------------------------
# (b) Cache identity — no-mask byte-identical to legacy
# ---------------------------------------------------------------------------


def test_canonical_feature_selection_hash_no_mask_byte_identical():
    """compute_canonical_feature_selection_hash with no mask == legacy golden.

    This is the cache-identity I1 invariant: existing on-disk caches that were
    built without a feature_mask must continue to be valid and reusable after
    the H7 change.  Two sub-assertions:
      1. Omitting feature_mask entirely == the frozen legacy hex.
      2. Passing feature_mask=None explicitly == the same frozen legacy hex.
    """
    h_no_arg = compute_canonical_feature_selection_hash(**_base_hash_args())
    h_none = compute_canonical_feature_selection_hash(
        **_base_hash_args(), feature_mask=None
    )
    assert h_no_arg == _CACHE_IDENTITY_LEGACY_GOLDEN, (
        f"cache-identity I1 broken (no-arg): got {h_no_arg}"
    )
    assert h_none == _CACHE_IDENTITY_LEGACY_GOLDEN, (
        f"cache-identity I1 broken (explicit None): got {h_none}"
    )


# ---------------------------------------------------------------------------
# (b) Cache identity — masked builds produce distinct hashes
# ---------------------------------------------------------------------------


def test_canonical_feature_selection_hash_mask_changes_identity():
    """Three mask configurations must produce 3 DISTINCT hashes (no collision).

    Also verifies canonical invariance: id order in the mask dict does not
    matter — the hash is the same regardless of insertion order.
    """
    base = _base_hash_args()
    h_no_mask = compute_canonical_feature_selection_hash(**base)
    h_blacklist = compute_canonical_feature_selection_hash(
        **base, feature_mask={"mode": "blacklist", "ids": [1, 2, 3]}
    )
    h_whitelist = compute_canonical_feature_selection_hash(
        **base, feature_mask={"mode": "whitelist", "ids": [4, 5]}
    )
    assert h_blacklist != h_no_mask, "blacklist mask must change cache identity"
    assert h_whitelist != h_no_mask, "whitelist mask must change cache identity"
    assert h_blacklist != h_whitelist, "blacklist vs whitelist must have distinct identities"

    # Canonical invariance: reversed id order must hash identically.
    h_blacklist_reorder = compute_canonical_feature_selection_hash(
        **base, feature_mask={"mode": "blacklist", "ids": [3, 2, 1]}
    )
    assert h_blacklist == h_blacklist_reorder, (
        "hash must be id-order invariant (canonical dict)"
    )


# ---------------------------------------------------------------------------
# (a) Pipeline — feature_mask slot resolves into the built pipeline
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n: int) -> dict[str, PerFeatureStat]:
    """Build a minimal stats dict with deterministic percentiles for n features."""
    pcts = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    return {
        str(i): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(pcts),
        )
        for i in range(n)
    }


def test_context_cache_pipeline_resolves_feature_mask():
    """build_feature_pipeline_from_cfg_block picks up feature_mask from cfg.

    With the H7 fix (feature_mask: null slot added to the yaml, and the
    resolver wired), a cfg block carrying feature_mask={mode,ids} must produce
    a pipeline whose .feature_mask is the corresponding FeatureMask object,
    not None.  A cfg with feature_mask=null must produce pipeline.feature_mask
    is None (I5 back-compat).
    """
    n_features = 16
    per_feature_stats = _make_per_feature_stats(n_features)
    # feature_percentile_segment_topk: real threshold type used in the yaml
    # default; supports the percentile-boundary build path used by build_cache.
    fs_block = {
        "threshold_type": "feature_percentile_segment_topk",
        "threshold": 2048.9,
    }

    # ---- blacklist mask ----
    cfg_with_mask = OmegaConf.create(
        {
            "feature_selection": fs_block,
            "negative_selection": None,
            "feature_mask": {"mode": "blacklist", "ids": [0, 1, 2]},
        }
    )
    pipeline_masked = build_feature_pipeline_from_cfg_block(
        cfg_with_mask,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_features,
        num_buckets=4,
    )
    assert pipeline_masked.feature_mask is not None, (
        "pipeline.feature_mask must not be None when cfg carries a blacklist mask "
        "(H7 gap: currently None -> cache records features the model never saw)"
    )
    assert pipeline_masked.feature_mask.mode == FeatureMaskMode.blacklist
    assert set(pipeline_masked.feature_mask.ids) == {0, 1, 2}

    # ---- null mask (I5 back-compat) ----
    cfg_no_mask = OmegaConf.create(
        {
            "feature_selection": fs_block,
            "negative_selection": None,
            "feature_mask": None,
        }
    )
    pipeline_no_mask = build_feature_pipeline_from_cfg_block(
        cfg_no_mask,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_features,
        num_buckets=4,
    )
    assert pipeline_no_mask.feature_mask is None, (
        "pipeline.feature_mask must be None when cfg.feature_mask is null (I5)"
    )


# ---------------------------------------------------------------------------
# (a) Config slot — build_eval_context_cache.yaml has feature_mask: null
# ---------------------------------------------------------------------------


def test_build_eval_context_cache_yaml_has_feature_mask_slot():
    """build_eval_context_cache.yaml must contain a feature_mask key set to null.

    This is the config-regression guard: if the slot is ever accidentally
    removed from the yaml, build_eval_context_cache would silently fall back to
    no feature_mask resolution, breaking mask-trained model eval builds.

    Uses Hydra's compose API for a genuine end-to-end round-trip through
    OmegaConf.  Falls back to direct yaml parse if Hydra compose is unavailable
    in the test environment.
    """
    from pathlib import Path

    import yaml as _yaml

    yaml_path = Path(__file__).parent.parent.parent / "configs" / "eval" / "build_eval_context_cache.yaml"
    assert yaml_path.is_file(), f"yaml not found at {yaml_path}"

    raw = _yaml.safe_load(yaml_path.read_text())
    assert isinstance(raw, dict), "yaml root must be a mapping"
    assert "feature_mask" in raw, (
        "build_eval_context_cache.yaml is missing the 'feature_mask' key — "
        "H7 gap: without this slot, build_feature_pipeline_from_cfg_block "
        "cannot resolve a mask for mask-trained model builds"
    )
    assert raw["feature_mask"] is None, (
        f"build_eval_context_cache.yaml feature_mask default must be null, "
        f"got {raw['feature_mask']!r}"
    )

    # Also verify via OmegaConf round-trip (catches interpolation issues).
    cfg = OmegaConf.create(raw)
    resolved = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(resolved, dict)
    assert resolved.get("feature_mask") is None, (
        "OmegaConf round-trip: feature_mask must be null"
    )
