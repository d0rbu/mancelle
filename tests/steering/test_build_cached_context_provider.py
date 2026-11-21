"""Tests for _build_cached_context_provider field-guard behaviour (I-2)."""

from __future__ import annotations

import json

import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors.torch import save_file

from errors import FeatureClustersError
from scripts.train.conditional_encdec_train import _build_cached_context_provider
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.packed_batch_io import compute_selection_hash
from steering.refined_types import PositiveInt
from steering.stats_io import content_sha256

_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(8)
_NUM_BUCKETS: PositiveInt = PositiveInt.parse(5)


def test_build_cached_context_provider_returns_none_when_conditioning_block_missing():
    cfg = OmegaConf.create({})
    assert _build_cached_context_provider(cfg) is None


def test_build_cached_context_provider_returns_none_when_cache_dir_unset():
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": None}})
    assert _build_cached_context_provider(cfg) is None


def test_build_cached_context_provider_raises_when_identity_missing():
    cfg = OmegaConf.create({
        "conditioning": {"context_cache_dir": "/tmp/x"},
        "model_max_seq_len": 512,
    })
    with pytest.raises(ValueError, match="conditioning.identity"):
        _build_cached_context_provider(cfg)


def test_build_cached_context_provider_raises_with_field_name_when_layer_null():
    cfg = OmegaConf.create({
        "conditioning": {
            "context_cache_dir": "/tmp/x",
            "identity": {
                "model_id": "gemma-2-2b", "sae_id": "test",
                "layer": None,  # null
                "pooling": "max_pool",
            },
        },
        "model_max_seq_len": 512,
    })
    with pytest.raises(ValueError, match="layer"):
        _build_cached_context_provider(cfg)


def test_build_cached_context_provider_raises_with_field_name_when_layer_string():
    cfg = OmegaConf.create({
        "conditioning": {
            "context_cache_dir": "/tmp/x",
            "identity": {
                "model_id": "gemma-2-2b", "sae_id": "test",
                "layer": "not-an-int",
                "pooling": "max_pool",
            },
        },
        "model_max_seq_len": 512,
    })
    with pytest.raises(ValueError, match="layer"):
        _build_cached_context_provider(cfg)


def _clustered_provider_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(cluster_path)}},
        8,
    )
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    vocab = effective_feature_vocab(clusters, 8)
    assert vocab is not None
    stats_path = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(stats_path), metadata={"k": "v"})
    stats_sha = content_sha256(stats_path)
    seed = 42
    feature_selection = {
        "distribution": "geometric",
        "mean": 12.0,
        "std": 12.0,
        "min_k": 1,
        "max_k": 49,
        "delta_weights": "percentile_delta",
    }
    # D-seed Codex cross-review fix (Finding 1/2, HIGH): the clustered
    # prompt-cache identity is now real-seed + delta-free with
    # delta_weights RETAINED as an explicit null (not popped) --
    # steering/eval/cache_identity.py's clustered branch and
    # scripts/train/conditional_encdec_train.py's derived_hash cross-check
    # both build this exact payload shape. Retaining the key (rather than
    # compute_canonical_feature_selection_hash's legacy hardcoded seed=0 +
    # stats_source_sha256-folded-into-delta_config formula this fixture
    # used to call) matches the real packed-row producer's convention --
    # compute_selection_hash hashes raw key presence, so popping the key
    # would silently diverge from what the production code now computes.
    prompt_selection = dict(feature_selection)
    prompt_selection["delta_weights"] = None  # ty: ignore[invalid-assignment]
    negative_selection = {
        "distribution": "geometric",
        "mean": 3.0,
        "min_k": 0,
        "max_k": 15,
    }
    derived_hash = compute_selection_hash(
        positive_selection=prompt_selection,
        negative_selection=negative_selection,
        delta_config=None,
        num_buckets=_NUM_BUCKETS,
        seed=seed,
        ddp_rank=0,
        feature_mask=None,
        feature_clusters=resolved.descriptor_dict(),
        splice_batch_size=None,
    )
    identity = CacheIdentity("model", "sae", 0, "max_pool", derived_hash)
    cache_root = tmp_path / "cache"
    cache_dir = context_length_cache_dir(
        cache_root,
        16,
        hash_leaf=derived_hash[:16],
    )
    metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
            stats_source_sha256=stats_sha,
        ),
    }
    save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                [1, 2],
                [ContextFeatureSegment((1,), (3,), 0, 1)],
            )
        ],
        identity=identity,
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=_NUM_BUCKETS,
        metadata=metadata,
    )
    cfg = OmegaConf.create(
        {
            "model_max_seq_len": 16,
            "seed": seed,
            "feature_selection": feature_selection,
            "negative_selection": negative_selection,
            "conditioning": {
                "context_cache_dir": str(cache_root),
                "n_sae_features": 8,
                "num_buckets": 5,
                "identity": {
                    "model_id": "model",
                    "sae_id": "sae",
                    "layer": 0,
                    "pooling": "max_pool",
                    "feature_selection_hash": derived_hash,
                },
            },
        }
    )
    return cfg, stats_path, clusters, vocab


def test_clustered_provider_derives_identity_from_training_state(tmp_path):
    cfg, stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    provider = _build_cached_context_provider(
        cfg,
        stats_path=stats_path,
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=_NUM_BUCKETS,
        feature_mask=None,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )
    assert provider is not None
    segments = provider.get([1, 2])
    assert segments is not None
    assert segments[0].feature_ids == (1,)


def test_clustered_provider_uses_exact_original_splice_statistics_source(tmp_path):
    cfg, original_stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    cfg.conditioning.stats_source_sha256 = content_sha256(original_stats_path)
    active_stats_path = tmp_path / "derived_feature_statistics.safetensors"
    save_file(
        {"weights": t.tensor([4.0, 5.0, 6.0])}, str(active_stats_path), metadata={"k": "other"}
    )

    provider = _build_cached_context_provider(
        cfg,
        stats_path=active_stats_path,
        original_stats_path=original_stats_path,
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=_NUM_BUCKETS,
        feature_mask=None,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )

    assert provider is not None
    assert len(provider) == 1


def test_clustered_provider_rejects_unknown_statistics_source(tmp_path):
    cfg, original_stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    cfg.conditioning.stats_source_sha256 = "f" * 64
    active_stats_path = tmp_path / "derived_feature_statistics.safetensors"
    save_file(
        {"weights": t.tensor([4.0, 5.0, 6.0])}, str(active_stats_path), metadata={"k": "other"}
    )

    with pytest.raises(
        ValueError,
        match="does not match either the active training statistics source",
    ):
        _build_cached_context_provider(
            cfg,
            stats_path=active_stats_path,
            original_stats_path=original_stats_path,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )


def test_clustered_provider_uses_encoder_decoder_max_seq_len(tmp_path):
    cfg, stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    cfg.max_seq_len = cfg.pop("model_max_seq_len")

    provider = _build_cached_context_provider(
        cfg,
        stats_path=stats_path,
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=_NUM_BUCKETS,
        feature_mask=None,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )

    assert provider is not None
    assert provider.get([1, 2]) is not None


def test_clustered_provider_rejects_explicit_hash_and_shape_drift(tmp_path):
    cfg, stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    cfg.conditioning.identity.feature_selection_hash = "f" * 64
    with pytest.raises(ValueError, match="does not match.*derived"):
        _build_cached_context_provider(
            cfg,
            stats_path=stats_path,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )

    cfg, stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path / "other")
    cfg.conditioning.num_buckets = 4
    with pytest.raises(ValueError, match="conditioning.num_buckets"):
        _build_cached_context_provider(
            cfg,
            stats_path=stats_path,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )


def test_clustered_provider_rejects_threshold_prompt_cache(tmp_path):
    cfg, stats_path, clusters, vocab = _clustered_provider_fixture(tmp_path)
    cfg.feature_selection = {
        "threshold_type": "feature_percentile",
        "threshold": 0.9,
    }

    with pytest.raises(
        FeatureClustersError,
        match="prompt caches support sampling selection only",
    ):
        _build_cached_context_provider(
            cfg,
            stats_path=stats_path,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
        )
