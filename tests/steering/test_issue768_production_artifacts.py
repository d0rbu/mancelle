"""Real-artifact preflight for the canonical issue-#768 production config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train.conditional_encdec_train import (
    _build_cached_context_provider,
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    canonical_feature_clusters_descriptor,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import canonical_dict, resolve_feature_mask
from steering.eval.axbench_base.seed_contract import sha256_file
from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)
from steering.packed_batch_io import compute_selection_hash

pytestmark = [pytest.mark.slow, pytest.mark.requires_feature_data]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSET_ROOT = Path(
    os.environ.get(
        "ISSUE768_ASSET_ROOT",
        "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer",
    )
)
_CONFIG_NAME = (
    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000"
)
_N_SAE = 65_536
_TOKEN_VOCAB = 32_768
_MASK_SHA256 = "2c7cc26c05f83297770babad80be95c3228292647c8dbc2359640329d557562c"
_CLUSTERS_SHA256 = "53e8758da02eb6c399d77aa75135ee901989d5f5a8171ccd3bb3953fae630cca"
_STATS_SHA256 = "dadf6d4f8d3b56ba1b40e0f49f4bb92bfb635c64f78a530f04f624b1d5773b5d"
_DERIVED_SELECTION_HASH = (
    "e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3"
)
_BASE_SELECTION_HASH = (
    "6be4095ad0116d3e1b7ca93a2fe0a47439eaba770c6d0ef08ab164a0652a19bc"
)
# Legacy pre-D-seed value: what steering.eval.feature_selection_hash's
# compute_canonical_feature_selection_hash wrapper computes for this
# config's clustered prompt-selection inputs (hardcoded seed=0,
# stats_source_sha256 folded into delta_config). Still exactly what that
# unmodified wrapper produces today -- pinned as an API-shape reference,
# NOT as this config's actual clustered cache identity (see
# _PROMPT_SELECTION_HASH below).
_PROMPT_SELECTION_HASH_LEGACY = "63aeb505576814764888abb660e9e9154c19bb399541a59bf3f50fc697bcea0f"
# D-seed (PR #739 Track D, 2026-07-24): this config's clustered
# eval-context-cache identity is now resolved by
# steering/eval/cache_identity.py's clustered branch -- the checkpoint's
# REAL training-preselection seed (conditioning.seed=42), delta_config=
# None (statistics identity is independently pinned by the clustered
# feature contract + cache manifest, not folded into this hash), AND
# delta_weights explicitly retained as null (matching the packed-row
# producer's own delta-free payload convention -- Codex cross-review
# Finding 1, HIGH: compute_selection_hash hashes raw key PRESENCE, so
# dropping delta_weights entirely produces a DIFFERENT hash than
# retaining it as null, even though parsing treats both identically).
# This value is therefore IDENTICAL to _BASE_SELECTION_HASH above -- the
# intended unification: the same delta-free selection/mask/cluster/seed
# inputs produce the same identity for both the clustered eval-context
# cache (c1024/<hash>/) and the packed-row "original/base validation"
# selection (sel/<hash>/), even though they live under different
# directory trees. Migration history: legacy (hardcoded seed=0) was
# 63aeb505576814764888abb660e9e9154c19bb399541a59bf3f50fc697bcea0f; a
# first real-seed attempt that incorrectly DROPPED delta_weights instead
# of retaining it as null was briefly
# 9b2085a6f814a6a63128ea186cfaa83a1369c6658a1bcdced1f5c97143d6434f (never
# a valid target -- did not match the packed-row identity).
_PROMPT_SELECTION_HASH = _BASE_SELECTION_HASH
_VOCAB_SHA256 = "8c183369ce4a8ff1e5f153a749e9eb71f9def65283d0a22b4d33bf5ff2d6f16c"


def _plain(value: Any) -> dict[str, Any]:
    decoded = OmegaConf.to_container(value, resolve=True)
    assert isinstance(decoded, dict)
    return cast(dict[str, Any], decoded)


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        pytest.skip(f"missing issue-#768 {label}: {path}")
    return path


def test_canonical_production_artifacts_policy_hashes_and_model_shape() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str((_REPO_ROOT / "configs/training").resolve()),
    ):
        cfg = compose(config_name=_CONFIG_NAME)

    mask_path = _require(_ASSET_ROOT / str(cfg.feature_mask.path), "manual whitelist")
    clusters_path = _require(
        _ASSET_ROOT / str(cfg.feature_clusters.path),
        "schema-v1 clusters",
    )
    stats_path = _require(
        _ASSET_ROOT
        / "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
        "m_gemma-3-4b-pt/base/context_independent/no_bos/"
        "feature_statistics.safetensors",
        "feature statistics",
    )
    derived_stats_path = _require(
        _ASSET_ROOT
        / "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
        "m_gemma-3-4b-pt/base/context_independent/no_bos/"
        "splice/cut_frac0.15/seed42/feature_statistics.safetensors",
        "derived feature statistics",
    )
    assert sha256_file(mask_path) == _MASK_SHA256
    assert sha256_file(clusters_path) == _CLUSTERS_SHA256
    assert sha256_file(stats_path) == _STATS_SHA256
    assert sha256_file(derived_stats_path) != _STATS_SHA256

    mask = resolve_feature_mask(
        {"feature_mask": {"mode": "whitelist", "path": str(mask_path)}},
        _N_SAE,
    )
    resolved_clusters = resolve_feature_clusters(
        {"feature_clusters": {"path": str(clusters_path)}},
        _N_SAE,
    )
    assert mask is not None
    assert resolved_clusters is not None
    clusters = FeatureClusterConsolidation.build(resolved_clusters, mask)
    vocab = effective_feature_vocab(clusters, _N_SAE)
    assert len(mask.ids) == 16_254
    assert int(clusters.consolidated_away.sum()) == 14_180
    assert vocab is not None
    assert vocab.size == 2_074
    assert vocab.fingerprint() == _VOCAB_SHA256

    positive = _plain(cfg.feature_selection)
    negative = _plain(cfg.negative_selection)
    delta = _plain(cfg.feature_delta)
    mask_canonical = canonical_dict(mask)
    clusters_canonical = canonical_feature_clusters_descriptor(resolved_clusters)
    assert (
        compute_selection_hash(
            positive_selection=positive,
            negative_selection=negative,
            delta_config=delta,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.seed),
            ddp_rank=0,
            feature_mask=mask_canonical,
            feature_clusters=clusters_canonical,
            splice_batch_size=cfg.get("selection_splice_batch_size"),
        )
        == _DERIVED_SELECTION_HASH
    )
    base_positive = dict(positive)
    base_positive["delta_weights"] = None
    assert (
        compute_selection_hash(
            positive_selection=base_positive,
            negative_selection=negative,
            delta_config=None,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.seed),
            ddp_rank=0,
            feature_mask=mask_canonical,
            feature_clusters=clusters_canonical,
        )
        == _BASE_SELECTION_HASH
    )
    prompt_positive = dict(positive)
    prompt_positive.pop("delta_weights")
    # Legacy pre-D-seed formula (unmodified: still exactly what the
    # unclustered eval-context-cache path uses today). Pinned as an
    # API-shape reference only -- NOT this config's actual clustered cache
    # identity, which is resolved via cache_identity.py's clustered branch
    # below (real seed, delta_config=None).
    assert (
        compute_canonical_feature_selection_hash(
            positive_selection=prompt_positive,
            negative_selection=negative,
            num_buckets=int(cfg.num_buckets),
            stats_source_sha256=_STATS_SHA256,
            feature_delta=None,
            feature_mask=mask_canonical,
            feature_clusters=clusters_canonical,
        )
        == _PROMPT_SELECTION_HASH_LEGACY
    )
    # D-seed (PR #739 Track D): this config's ACTUAL clustered
    # eval-context-cache identity -- steering/eval/cache_identity.py's
    # clustered branch, real conditioning.seed (42, not the legacy
    # formula's hardcoded seed=0), delta_config=None (stats identity is
    # independently pinned by the clustered feature contract + cache
    # manifest -- see steering/eval/cache_identity.py's clustered-branch
    # comment and D-contract's scheduled checkpoint cross-check), AND
    # delta_weights RETAINED as null (Codex cross-review Finding 1, HIGH:
    # compute_selection_hash hashes raw key presence -- popping the key
    # entirely, as the legacy assertion above does, produces a DIFFERENT
    # hash than retaining delta_weights=None, even though parsing treats
    # them identically). Must match the pinned
    # conditioning.identity.feature_selection_hash so
    # _build_cached_context_provider below (the real training-loop call
    # path) resolves without raising -- and must equal
    # _BASE_SELECTION_HASH above, the real packed-row "original/base
    # validation" selection identity for this same delta-free policy
    # (the unification this PR fixes).
    prompt_positive_retained = dict(prompt_positive)
    prompt_positive_retained["delta_weights"] = None
    assert (
        compute_selection_hash(
            positive_selection=prompt_positive_retained,
            negative_selection=negative,
            delta_config=None,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.conditioning.seed),
            ddp_rank=0,
            feature_mask=mask_canonical,
            feature_clusters=clusters_canonical,
            splice_batch_size=None,
        )
        == _PROMPT_SELECTION_HASH
        == _BASE_SELECTION_HASH
        == str(cfg.conditioning.identity.feature_selection_hash)
    )

    with t.device("meta"):
        model = build_encdec_meta_model_from_training_cfg(
            cfg,
            base_vocab=_TOKEN_VOCAB,
            n_sae_features=_N_SAE,
            feature_vocab=vocab,
        )
    assert model.feature_prediction_width == 2_074
    assert model.feature_pred_head is not None
    assert sum(parameter.numel() for parameter in model.feature_pred_head.parameters()) == 1_594_906
    assert model.encoder.feature_embed.num_embeddings == 2_074
    assert int(cfg.core_metric_max_per_task) == 500

    cfg.conditioning.context_cache_dir = str(
        _ASSET_ROOT / str(cfg.conditioning.context_cache_dir)
    )
    cfg.conditioning.feature_mask.path = str(mask_path)
    cfg.conditioning.feature_clusters.path = str(clusters_path)
    provider = _build_cached_context_provider(
        cfg,
        stats_path=derived_stats_path,
        original_stats_path=stats_path,
        n_sae_features=_N_SAE,
        num_buckets=int(cfg.num_buckets),
        feature_mask=mask,
        feature_clusters=clusters,
        feature_vocab=vocab,
    )
    assert provider is not None
    assert len(provider) == 8_239
