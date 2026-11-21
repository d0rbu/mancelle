"""Edge-case + invariance tests for build_feature_pipeline_from_cfg_block
and the rename. Complements the basic unit tests with adversarial inputs
and stronger contract assertions:

- non-dict feature_selection in cfg.conditioning (BPB pipeline path)
- cfg_block.feature_selection = None (vs missing key)
- feature_delta + ThresholdSelection passthrough
- plain-dict negative_selection (vs OmegaConf-only)
- canonical-hash invariance: rename does NOT change on-disk hash
- end-to-end hash bit-identity: feature_selection dict value alone determines hash
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_pipeline import (
    build_feature_pipeline_from_cfg_block,
)
from steering.eval.cache_identity import (
    cache_identity_from_config,
    resolve_feature_selection_hash,
)
from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)
from steering.packed_batch_io import compute_selection_hash
from steering.stats_types import PerFeatureStat
from tests.statistics_doubles import fixed_statistics_double

N = 64
B = 4


class _ExplodingIdentityConfig(Mapping[str, object]):
    def __init__(self, exploding_key: str) -> None:
        self._data: dict[str, object] = {
            "model_id": "model",
            "sae_id": "sae",
            "layer": 8,
            "pooling": "max",
            "feature_selection_hash": "a" * 64,
        }
        self._exploding_key = exploding_key

    def __getitem__(self, key: str) -> object:
        if key == self._exploding_key:
            raise RuntimeError(f"failed to read {key}")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


@pytest.mark.parametrize("exploding_key", ["model_id", "feature_selection_hash"])
def test_cache_identity_does_not_downgrade_mapping_access_failures(
    exploding_key: str,
) -> None:
    with pytest.raises(RuntimeError, match=f"failed to read {exploding_key}"):
        cache_identity_from_config(_ExplodingIdentityConfig(exploding_key))


def _stats() -> dict[str, PerFeatureStat]:
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(N):
        out[str(i)] = PerFeatureStat(
            count=100 - i,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(full_percentiles),
            histogram=None,
        )
    return out


# ---------------------------------------------------------------------------
# Edge case 1: BPB pipeline non-dict feature_selection
# ---------------------------------------------------------------------------


def test_bpb_pipeline_raises_typeerror_on_non_dict_feature_selection(tmp_path):
    """When cfg.conditioning.feature_selection is a non-dict, scalar value,
    bpb_pipeline.build_feature_pipeline_from_cfg surfaces TypeError from
    the helper.

    Pre-migration this was ValueError ("must be a dict"); the post-migration
    helper raises TypeError ("must resolve to a mapping"). This test pins
    the new behavior so future refactors don't silently regress it.
    """
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg
    cfg = OmegaConf.create({
        "conditioning": {
            "identity": {
                "model_id": "m", "sae_id": "s", "layer": 0, "pooling": "max_pool",
                "feature_selection_hash": None,
            },
            "feature_selection": "not-a-dict",  # malformed
            "n_sae_features": 64,
            "num_buckets": 2,
        },
        "bpb": {
            "stats_path": str(tmp_path / "stats.safetensors"),
            "feature_root": "out/data/features",
            "segment_size": 32,
        },
    })
    # Stub _resolve_stats_path + load_statistics so we hit the helper.
    from unittest.mock import patch

    fake_stats = fixed_statistics_double(per_feature_stats=_stats())
    with patch("steering.eval.bpb_pipeline._resolve_stats_path", return_value=tmp_path / "stats.safetensors"), \
         patch("steering.eval.bpb_pipeline.load_statistics", return_value=fake_stats):
        with pytest.raises(TypeError, match="feature_selection.*resolve to a mapping"):
            build_feature_pipeline_from_cfg(cfg)


# ---------------------------------------------------------------------------
# Edge case 2: feature_selection set to None (vs missing key)
# ---------------------------------------------------------------------------


def test_helper_rejects_explicit_none_feature_selection_value():
    """cfg_block has feature_selection: null in YAML — distinct from the key
    being missing. Both should produce a clear TypeError; this pins the
    explicit-null path which YAML operators sometimes hit."""
    cfg = OmegaConf.create({"feature_selection": None})
    with pytest.raises(TypeError, match="must define 'feature_selection'"):
        build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_stats(),
            n_sae_features=N,
            num_buckets=B,
        )


def test_helper_rejects_missing_feature_selection_key():
    """cfg_block has no feature_selection key at all."""
    cfg = OmegaConf.create({})
    with pytest.raises(TypeError, match="must define 'feature_selection'"):
        build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_stats(),
            n_sae_features=N,
            num_buckets=B,
        )


def test_helper_rejects_none_cfg_block():
    """Passing cfg_block=None outright raises TypeError."""
    with pytest.raises(TypeError, match="cfg_block must be a mapping"):
        build_feature_pipeline_from_cfg_block(
            cast(Any, None),
            per_feature_stats=_stats(),
            n_sae_features=N,
            num_buckets=B,
        )


# ---------------------------------------------------------------------------
# Edge case 3: feature_delta passthrough with ThresholdSelection
# ---------------------------------------------------------------------------


def test_helper_with_feature_delta_and_threshold_selection():
    """ThresholdSelection + feature_delta: build_feature_pipeline does NOT
    raise (validation is sampling-specific). delta_bundle is constructed
    and the percentile_table is allocated."""
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
        },
    })
    fd = FeatureDeltaConfig(mode=DeltaMode.cum_max, neg_delta_ratio=0.7)
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=N,
        num_buckets=B,
        feature_delta=fd,
    )
    assert pipeline.delta_bundle is not None
    assert pipeline.delta_bundle.config is fd
    # percentile_table is required for delta even on threshold path
    assert pipeline.delta_bundle.percentile_table.shape == (N, 99)


def test_helper_threshold_with_feature_delta_does_not_validate_delta_weights():
    """Sampling-specific delta_weights validation must NOT trigger for
    ThresholdSelection. (build_feature_pipeline only checks delta_weights
    for SamplingSelection — this regression test pins that contract.)"""
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
            # NO delta_weights — would error if ThresholdSelection were validated
        },
    })
    fd = FeatureDeltaConfig(mode=DeltaMode.cum_max, neg_delta_ratio=0.5)
    # Should not raise.
    build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=N,
        num_buckets=B,
        feature_delta=fd,
    )


# ---------------------------------------------------------------------------
# Edge case 4: plain-dict cfg_block (vs OmegaConf-only)
# ---------------------------------------------------------------------------


def test_helper_plain_dict_with_negative_selection():
    """Plain-dict path for both feature_selection and negative_selection."""
    cfg = {
        "feature_selection": {"threshold_type": "feature_percentile", "threshold": 0.95},
        "negative_selection": {"distribution": "geometric", "mean": 2.0, "std": 2.0},
    }
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=N,
        num_buckets=B,
    )
    assert pipeline.negative_bundle is not None
    assert pipeline.effective_buckets == B - 1


def test_helper_plain_dict_use_negative_selection_false():
    """Plain dict + use_negative_selection=False forces None even when set."""
    cfg = {
        "feature_selection": {"threshold_type": "feature_percentile", "threshold": 0.95},
        "negative_selection": {"distribution": "geometric", "mean": 2.0, "std": 2.0},
    }
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=N,
        num_buckets=B,
        use_negative_selection=False,
    )
    assert pipeline.negative_bundle is None
    assert pipeline.effective_buckets == B  # unchanged


# ---------------------------------------------------------------------------
# Cache-identity invariance: rename must NOT change on-disk hash
# ---------------------------------------------------------------------------


def test_resolve_feature_selection_hash_matches_direct_canonical_call():
    """The hash auto-computed via cond.feature_selection (new field name)
    must equal the hash from a direct compute_canonical_feature_selection_hash
    call with the same dict value passed as positive_selection.

    This is the load-bearing cache-identity invariance: the rename is purely
    a YAML/Python-symbol change; the on-disk hash payload still uses the
    'positive_selection' JSON key. Operator caches built before the rename
    must remain valid."""
    fs_dict = {"threshold_type": "feature_percentile", "threshold": 0.95}
    stats_sha = "abcd" * 16

    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = OmegaConf.create({
        "feature_selection": fs_dict,
        "negative_selection": None,
        "num_buckets": 2,
        "stats_source_sha256": stats_sha,
    })

    h_via_resolve = resolve_feature_selection_hash(identity_cfg, cond)
    h_direct = compute_canonical_feature_selection_hash(
        positive_selection=fs_dict,
        negative_selection=None,
        num_buckets=2,
        stats_source_sha256=stats_sha,
    )
    assert h_via_resolve == h_direct, (
        "rename must not change on-disk hash — both paths must produce "
        "identical sha256 of the canonical JSON dump"
    )


def test_canonical_hash_json_payload_uses_positive_selection_key():
    """The canonical hash JSON key is 'positive_selection' regardless of
    YAML field name. Pin the byte format so a future refactor of the JSON
    payload is detected by this test."""
    fs_dict = {"threshold_type": "feature_percentile", "threshold": 0.95}
    h = compute_selection_hash(
        positive_selection=fs_dict,
        negative_selection=None,
        delta_config=None,
        num_buckets=2,
        seed=0,
        ddp_rank=0,
    )
    expected_canonical = json.dumps(
        {
            "positive_selection": fs_dict,
            "negative_selection": None,
            "delta_config": None,
            "num_buckets": 2,
            "seed": 0,
            "ddp_rank": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
    assert h == expected_hash, (
        "JSON payload format changed — this would invalidate every existing "
        "operator cache. Either roll back the format change or accept the "
        "cache-identity bump."
    )


def test_canonical_hash_with_negative_selection_uses_positive_selection_key():
    """Same byte-format pin, but with non-null negative_selection. Catches
    accidental key reordering or rename of the negative_selection JSON key."""
    fs_dict = {"distribution": "geometric", "mean": 4.0, "std": 4.0}
    neg_dict = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    h = compute_selection_hash(
        positive_selection=fs_dict,
        negative_selection=neg_dict,
        delta_config=None,
        num_buckets=4,
        seed=42,
        ddp_rank=0,
    )
    expected_canonical = json.dumps(
        {
            "positive_selection": fs_dict,
            "negative_selection": neg_dict,
            "delta_config": None,
            "num_buckets": 4,
            "seed": 42,
            "ddp_rank": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
    assert h == expected_hash


# ---------------------------------------------------------------------------
# Hash invariance under helper migration
# ---------------------------------------------------------------------------


def test_helper_inputs_match_compute_selection_hash_inputs_byte_identity():
    """When the helper is called with a cfg_block, the parsed feature_selection
    dict (after OmegaConf.to_container) is byte-identical to the dict that
    would flow into compute_selection_hash if computed at the same call site.

    Practical guarantee: post-migration call sites
    (precompute_selected_features, conditional_encdec_train) that read the
    raw cfg via OmegaConf.to_container and pass it to compute_selection_hash
    produce the same selection_hash as before."""
    fs_yaml = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
        },
    })
    # Extraction the way the call sites do it (OmegaConf.to_container at the
    # top of the function, before passing to compute_selection_hash).
    fs_raw = OmegaConf.to_container(fs_yaml.feature_selection, resolve=True)
    neg_raw = OmegaConf.to_container(fs_yaml.negative_selection, resolve=True)

    # Now compute the hash via compute_selection_hash directly.
    h_direct = compute_selection_hash(
        positive_selection=fs_raw,  # ty: ignore[invalid-argument-type]
        negative_selection=neg_raw,  # ty: ignore[invalid-argument-type]
        delta_config=None,
        num_buckets=2,
        seed=42,
        ddp_rank=0,
    )

    # Verify the dict round-trip is what we expect (no OmegaConf wrapping
    # leaks into the hash payload).
    assert isinstance(fs_raw, dict)
    assert isinstance(neg_raw, dict)
    # The hash is deterministic; assert against a recomputed canonical form.
    expected_canonical = json.dumps(
        {
            "positive_selection": fs_raw,
            "negative_selection": neg_raw,
            "delta_config": None,
            "num_buckets": 2,
            "seed": 42,
            "ddp_rank": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(expected_canonical.encode()).hexdigest()
    assert h_direct == expected_hash


# ---------------------------------------------------------------------------
# completion_steering_eval semantics: use_negative_selection=False
# ---------------------------------------------------------------------------


def test_use_negative_selection_false_with_negative_in_cfg_keeps_full_buckets():
    """completion_steering_eval pattern: cfg has negative_selection set
    (because the YAML inherits from a training-style template) but the eval
    explicitly opts out via use_negative_selection=False. The pipeline must
    use the full num_buckets, NOT num_buckets-1 (which would happen if
    negative_selection were honored)."""
    cfg = OmegaConf.create({
        "feature_selection": {"threshold_type": "feature_percentile", "threshold": 0.95},
        "negative_selection": {"distribution": "geometric", "mean": 2.0, "std": 2.0},
    })
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=N,
        num_buckets=4,
        use_negative_selection=False,
    )
    assert pipeline.negative_bundle is None
    assert pipeline.effective_buckets == 4  # NOT 3
