"""Cluster-aware prompt-cache identity, selection, and provider contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.conditional.cluster_segment_selection as selection_module
from errors import FeatureClustersError
from steering.conditional.discretization import discretize_activations
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import build_feature_pipeline
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
    sample_features,
    sample_inactive_features,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.cache_identity import resolve_feature_selection_hash
from steering.eval.context_cache_build import _select_and_bucket_segment
from steering.eval.context_cache_io import (
    CLUSTER_SELECTION_SEMANTICS,
    CLUSTER_SELECTION_SEMANTICS_KEY,
    CacheIdentity,
    ContextCacheEntry,
    ContextCacheValidationError,
    SafetensorsContextFeatureProvider,
    clustered_context_cache_metadata,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.overlay_io import load_overlay_shard
from steering.refined_types import PositiveInt
from steering.stats_types import PerFeatureStat

N_FEATURES: PositiveInt = PositiveInt.parse(8)
N_BUCKETS: PositiveInt = PositiveInt.parse(5)
STATS_SHA = "a" * 64


def _write_clusters(path: Path, clusters: list[list[int]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": clusters}),
        encoding="utf-8",
    )
    return path


def _stats() -> dict[str, PerFeatureStat]:
    out: dict[str, PerFeatureStat] = {}
    for feature_id in range(N_FEATURES):
        # Feature 3 deliberately has a different raw scale. A magnitude of
        # 1.6 is its p80 while feature 1 uses 0.2 at p20.
        scale = 0.02 if feature_id == 3 else 0.01
        percentiles = {f"p{p}": p * scale for p in range(1, 100)}
        out[str(feature_id)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles=percentiles,
        )
    return out


def _cluster_state(
    tmp_path: Path,
    *,
    clusters: list[list[int]] | None = None,
    mask: FeatureMask | None = None,
) -> FeatureClusterConsolidation:
    cluster_path = _write_clusters(
        tmp_path / "clusters.json",
        clusters or [[1, 3], [2, 4]],
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(cluster_path)}},
        N_FEATURES,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _pipeline(
    *,
    mask: FeatureMask | None,
    seed: int = 42,
    negatives: bool = True,
):
    return build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=(
            NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
            )
            if negatives
            else None
        ),
        per_feature_stats=_stats(),
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        seed=seed,
        feature_mask=mask,
    )


def _frozen_legacy_sampling_selection(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    pipeline,
) -> tuple[t.Tensor, t.Tensor]:
    """PR-1 no-cluster selection boundary, frozen as a parity oracle."""
    pos_fids, pos_mags = raw_fids, raw_mags
    if pipeline.feature_mask is not None:
        pos_fids, pos_mags = pipeline.feature_mask.apply_to_positive(
            raw_fids,
            raw_mags,
        )
    assert isinstance(pipeline.selection_config, SamplingBundle)
    fids, mags = sample_features(
        pos_fids,
        pos_mags,
        pipeline.selection_config.percentile_table,
        pipeline.selection_config.selection,
        pipeline.selection_config.rng,
    )
    buckets = (
        discretize_activations(fids, mags, pipeline.boundaries)
        if fids.numel() > 0
        else t.tensor([], dtype=t.int64)
    )
    if pipeline.negative_bundle is not None:
        negative_ids = sample_inactive_features(
            raw_fids,
            N_FEATURES,
            pipeline.negative_bundle.selection,
            pipeline.negative_bundle.rng,
            feature_mask=pipeline.feature_mask,
        )
        if negative_ids.numel() > 0:
            fids = t.cat((fids.to(t.int64), negative_ids.to(t.int64)))
            buckets = t.cat(
                (
                    buckets.to(t.int64),
                    t.full(
                        (negative_ids.numel(),),
                        pipeline.effective_buckets,
                        dtype=t.int64,
                    ),
                )
            )
    return fids.to(t.int64), buckets.to(t.int64)


def _identity_cond(
    cluster_path: Path,
    *,
    members: list[int] | None = None,
    seed: int = 42,
):
    if members is not None:
        _write_clusters(cluster_path, [members])
    return OmegaConf.create(
        {
            "n_sae_features": N_FEATURES,
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "min_k": 0,
                "max_k": 15,
            },
            "num_buckets": N_BUCKETS,
            "stats_source_sha256": STATS_SHA,
            "feature_mask": {"mode": "whitelist", "ids": [6, 4, 3, 2, 1, 0]},
            "feature_clusters": {"path": str(cluster_path)},
            "feature_delta": None,
            # D-seed: clustered identity is pinned to the real training
            # seed (steering/eval/cache_identity.py's
            # site6-eval-context-cache-clustered branch). Defaulted here
            # (42, matching build_feature_pipeline's own legacy default) so
            # every pre-existing caller of _identity_cond keeps passing
            # unmodified; new D-seed tests pass seed= explicitly to build
            # two conds that differ ONLY in seed.
            "seed": seed,
        }
    )


def test_clustered_identity_is_canonical_and_membership_sensitive(tmp_path: Path):
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3], [2, 4]])
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = _identity_cond(cluster_path)
    first = resolve_feature_selection_hash(identity_cfg, cond)

    cond.feature_mask.ids = [0, 1, 2, 3, 4, 6]
    reordered = resolve_feature_selection_hash(identity_cfg, cond)
    assert reordered == first

    _write_clusters(cluster_path, [[1, 3], [2, 6]])
    changed = resolve_feature_selection_hash(identity_cfg, cond)
    assert changed != first


def test_clustered_identity_differs_by_seed_only(tmp_path: Path):
    """D-seed: two conds identical except ``seed`` must hash differently --
    clustered cache identity is pinned to the real training-preselection
    seed (steering/eval/cache_identity.py's clustered branch), not the
    legacy hardcoded ``seed=0``.
    """
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3], [2, 4]])
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond_a = _identity_cond(cluster_path, seed=17)
    cond_b = _identity_cond(cluster_path, seed=18)

    hash_a = resolve_feature_selection_hash(identity_cfg, cond_a)
    hash_b = resolve_feature_selection_hash(identity_cfg, cond_b)
    assert hash_a != hash_b


def test_clustered_identity_raises_on_missing_seed(tmp_path: Path):
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3], [2, 4]])
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = _identity_cond(cluster_path)
    del cond["seed"]
    with pytest.raises(ValueError, match="conditioning.seed"):
        resolve_feature_selection_hash(identity_cfg, cond)


@pytest.mark.parametrize(
    "seed_value,label",
    [
        (-1, "negative"),
        (3.5, "float"),
        ("42", "numeric-string"),
        (True, "bool"),
    ],
)
def test_clustered_identity_raises_on_invalid_seed(
    tmp_path: Path,
    seed_value: object,
    label: str,
):
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3], [2, 4]])
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = _identity_cond(cluster_path)
    cond.seed = seed_value
    with pytest.raises(ValueError, match="conditioning.seed"):
        resolve_feature_selection_hash(identity_cfg, cond)


def test_unclustered_identity_does_not_require_seed():
    """U-invariant: the unclustered branch never reads/requires ``seed``."""
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = OmegaConf.create(
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
            },
            "negative_selection": None,
            "num_buckets": N_BUCKETS,
            "stats_source_sha256": STATS_SHA,
        }
    )
    # No 'seed' key at all -- must resolve without raising.
    resolve_feature_selection_hash(identity_cfg, cond)


def test_no_cluster_selection_rng_and_cache_payload_match_frozen_legacy(
    tmp_path: Path,
):
    mask = FeatureMask(FeatureMaskMode.blacklist, (7,))
    current_pipeline = _pipeline(mask=mask, seed=29, negatives=True)
    legacy_pipeline = _pipeline(mask=mask, seed=29, negatives=True)
    raw_fids = t.tensor([0, 1, 3, 6], dtype=t.int64)
    raw_mags = t.tensor([0.31, 0.20, 1.60, 0.70], dtype=t.float32)

    current = _select_and_bucket_segment(
        raw_fids,
        raw_mags,
        pipeline=current_pipeline,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        source="current",
        feature_clusters=None,
    )
    legacy = _frozen_legacy_sampling_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipeline,
    )
    assert t.equal(current[0], legacy[0])
    assert t.equal(current[1], legacy[1])
    assert t.equal(
        current_pipeline.selection_config.rng.get_state(),
        legacy_pipeline.selection_config.rng.get_state(),
    )
    assert current_pipeline.negative_bundle is not None
    assert legacy_pipeline.negative_bundle is not None
    assert t.equal(
        current_pipeline.negative_bundle.rng.get_state(),
        legacy_pipeline.negative_bundle.rng.get_state(),
    )

    identity = CacheIdentity("model", "sae", 0, "max_pool", "e" * 64)
    def entry(selected) -> ContextCacheEntry:
        return ContextCacheEntry.from_sequences(
            [1, 2],
            [
                ContextFeatureSegment(
                    tuple(selected[0].tolist()),
                    tuple(selected[1].tolist()),
                    0,
                    1,
                )
            ],
        )

    manifests = []
    for name, selected in (("current", current), ("legacy", legacy)):
        cache_dir = context_length_cache_dir(
            tmp_path / name,
            16,
            hash_leaf="e" * 16,
        )
        manifests.append(
            save_context_cache(
                cache_dir,
                [entry(selected)],
                identity=identity,
                n_sae_features=N_FEATURES,
                num_buckets=N_BUCKETS,
                metadata={"model_max_seq_len": 16},
            )
        )
    current_manifest = json.loads(manifests[0].read_text())
    legacy_manifest = json.loads(manifests[1].read_text())
    current_sha = current_manifest["shards"][0].pop("sha256")
    legacy_sha = legacy_manifest["shards"][0].pop("sha256")
    assert current_sha != ""
    assert legacy_sha != ""
    assert current_manifest == legacy_manifest
    shard_name = current_manifest["shards"][0]["file"]
    current_shard = manifests[0].parent / shard_name
    legacy_shard = manifests[1].parent / shard_name
    # safetensors metadata maps are not emitted in a stable key order across
    # independent writes, so compare every serialized tensor payload exactly.
    current_tensors = load_overlay_shard(current_shard)
    legacy_tensors = load_overlay_shard(legacy_shard)
    assert current_tensors.keys() == legacy_tensors.keys()
    assert all(
        t.equal(current_tensors[key], legacy_tensors[key])
        for key in current_tensors
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("feature_delta", {"mode": "splice_counterfactual"}, "feature_delta=null"),
        ("delta_weights", None, "omit delta_weights entirely"),
        ("delta_weights", "percentile_delta", "omit delta_weights entirely"),
    ],
)
def test_clustered_prompt_identity_rejects_delta_semantics(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
):
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3]])
    cond = _identity_cond(cluster_path)
    if field == "feature_delta":
        cond.feature_delta = value
    else:
        cond.feature_selection[field] = value
    with pytest.raises(ValueError, match=match):
        resolve_feature_selection_hash(
            OmegaConf.create({"feature_selection_hash": None}),
            cond,
        )


def test_clustered_selection_uses_sibling_max_percentile_and_rep_buckets(
    tmp_path: Path,
):
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]])
    pipeline = _pipeline(mask=None, negatives=False)

    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 3], dtype=t.int64),
        t.tensor([0.2, 1.6], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )

    # Feature 1 is around p20; sibling 3 is around p80 on its own scale. The
    # representative keeps the latter and maps it directly to bucket 4 of 5.
    assert feature_ids.tolist() == [1]
    assert buckets.tolist() == [4]


def test_post_mask_identity_clusters_still_use_clustered_percentile_semantics(
    tmp_path: Path,
    monkeypatch,
):
    """D-dedup (PR #739 Track D, axis C) re-pin: BEFORE delegation, percentile
    was scored strictly AFTER masking (this builder called
    ``magnitude_to_percentile`` only on the post-mask survivor, ``[1]``).
    The merged A5 primitive (``select_clustered_segment``) requires the
    RAW (pre-mask) ids and owns masking internally, matching training's
    real preselection call
    (``scripts/preprocess/precompute_selected_features.py:1244`` passes
    ``select_clustered_segment(raw_unmasked_fids, uint8_pcts, ...)``
    directly) -- so the delegating builder now scores the FULL raw ids
    ``[1, 3]`` (masked sibling included) before masking removes 3. This is
    training-currency, not a regression: percentile scoring is per-feature
    independent (each id's score depends only on its own magnitude and its
    own percentile-table row), so scoring the masked sibling too cannot
    change survivor 1's own score or the final selected bucket -- verified
    empirically in
    ``tests/steering/test_d_dedup_clustered_selection_differential.py::
    test_cell_e1_post_mask_identity_percentile_call_site_replay`` (Cell E1),
    which replays this exact fixture through both the frozen pre-delegation
    body and the delegating replacement side by side. See also Cell E2 in
    that file for the reachable consequence of this ordering change: the
    shared primitive loudly drops a masked feature with no percentile
    statistics before scoring, while an unmasked stat-less feature remains
    fatal.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]], mask=mask)
    assert clusters.is_identity_after_mask
    pipeline = _pipeline(mask=mask, negatives=False)
    seen: list[list[int]] = []

    def fake_percentiles(feature_ids, feature_mags, percentile_table):
        seen.append(feature_ids.tolist())
        return t.full(feature_ids.shape, 90.0, dtype=t.float64)

    monkeypatch.setattr(
        selection_module,
        "magnitude_to_percentile",
        fake_percentiles,
    )
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 3], dtype=t.int64),
        t.tensor([0.2, 1.6], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )

    # Re-pinned (was [[1]] pre-delegation): the delegating builder scores
    # the FULL raw ids [1, 3] before select_clustered_segment masks 3 away
    # internally -- see docstring above and differential Cell E1.
    assert seen == [[1, 3]]
    assert feature_ids.tolist() == [1]
    assert buckets.tolist() == [4]


def test_clustered_selection_masks_before_representative_choice_and_negatives(
    tmp_path: Path,
):
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    clusters = _cluster_state(tmp_path, clusters=[[1, 3, 5]], mask=mask)
    pipeline = _pipeline(mask=mask, negatives=True)

    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 3, 5], dtype=t.int64),
        t.tensor([0.9, 1.6, 0.4], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )

    assert feature_ids[0].item() == 3
    assert buckets[0].item() == 3  # four active buckets; p80 -> bucket 3
    assert len(set(feature_ids.tolist())) == feature_ids.numel()
    assert 1 not in feature_ids.tolist()
    assert 5 not in feature_ids.tolist()
    assert 7 not in feature_ids.tolist()
    if feature_ids.numel() > 1:
        assert buckets[1:].tolist() == [pipeline.effective_buckets]


def test_clustered_selection_reuses_one_sequential_rng_stream(tmp_path: Path):
    clusters = _cluster_state(tmp_path, clusters=[[1, 3], [2, 4]])
    first_pipeline = _pipeline(mask=None, seed=17, negatives=True)
    second_pipeline = _pipeline(mask=None, seed=17, negatives=True)
    segments = (
        (
            t.tensor([0, 1, 3, 6], dtype=t.int64),
            t.tensor([0.31, 0.20, 1.60, 0.70], dtype=t.float32),
        ),
        (
            t.tensor([2, 4, 5, 7], dtype=t.int64),
            t.tensor([0.40, 0.80, 0.55, 0.95], dtype=t.float32),
        ),
    )

    def run(pipeline):
        return [
            _select_and_bucket_segment(
                ids,
                mags,
                pipeline=pipeline,
                n_sae_features=N_FEATURES,
                num_buckets=N_BUCKETS,
                source=f"segment-{idx}",
                feature_clusters=clusters,
            )
            for idx, (ids, mags) in enumerate(segments)
        ]

    first = run(first_pipeline)
    second = run(second_pipeline)
    assert [ids.tolist() for ids, _ in first] == [
        ids.tolist() for ids, _ in second
    ]
    assert [buckets.tolist() for _, buckets in first] == [
        buckets.tolist() for _, buckets in second
    ]
    assert t.equal(
        first_pipeline.selection_config.rng.get_state(),
        second_pipeline.selection_config.rng.get_state(),
    )
    assert first_pipeline.negative_bundle is not None
    assert second_pipeline.negative_bundle is not None
    assert t.equal(
        first_pipeline.negative_bundle.rng.get_state(),
        second_pipeline.negative_bundle.rng.get_state(),
    )


def test_clustered_threshold_context_selection_is_explicitly_unsupported(
    tmp_path: Path,
):
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]])
    pipeline = build_feature_pipeline(
        feature_selection=ThresholdSelection(
            threshold_type=ThresholdType.segment_topk,
            threshold=1,
        ),
        negative_selection=None,
        per_feature_stats=_stats(),
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        seed=42,
    )

    with pytest.raises(
        FeatureClustersError,
        match="clustered threshold context selection",
    ):
        _select_and_bucket_segment(
            t.tensor([1, 3], dtype=t.int64),
            t.tensor([0.2, 1.6], dtype=t.float32),
            pipeline=pipeline,
            n_sae_features=N_FEATURES,
            num_buckets=N_BUCKETS,
            source="unit",
            feature_clusters=clusters,
        )


def test_provider_enforces_effective_original_id_universe_before_rows(
    tmp_path: Path,
):
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]])
    vocab = effective_feature_vocab(clusters, N_FEATURES)
    assert vocab is not None
    identity = CacheIdentity("model", "sae", 0, "max_pool", "b" * 64)
    metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
            stats_source_sha256=STATS_SHA,
        ),
    }
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        16,
        hash_leaf="b" * 16,
    )
    save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                [1, 2],
                [ContextFeatureSegment((3,), (1,), 0, 1)],
            )
        ],
        identity=identity,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        metadata=metadata,
    )

    with pytest.raises(
        ContextCacheValidationError,
        match="outside the expected effective original-ID vocabulary",
    ):
        SafetensorsContextFeatureProvider(
            cache_dir,
            identity=identity,
            expected_n_sae_features=N_FEATURES,
            expected_num_buckets=N_BUCKETS,
            expected_metadata=metadata,
            expected_feature_vocab=vocab,
        )


def test_clustered_cache_roundtrip_keeps_original_representative_ids(
    tmp_path: Path,
):
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]])
    vocab = effective_feature_vocab(clusters, N_FEATURES)
    assert vocab is not None
    identity = CacheIdentity("model", "sae", 0, "max_pool", "c" * 64)
    metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
            stats_source_sha256=STATS_SHA,
        ),
    }
    assert metadata[CLUSTER_SELECTION_SEMANTICS_KEY] == CLUSTER_SELECTION_SEMANTICS
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        16,
        hash_leaf="c" * 16,
    )
    save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                [1, 2],
                [ContextFeatureSegment((1, 6), (3, 4), 0, 1)],
            )
        ],
        identity=identity,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        metadata=metadata,
    )
    provider = SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=N_FEATURES,
        expected_num_buckets=N_BUCKETS,
        expected_metadata=metadata,
        expected_feature_vocab=vocab,
    )

    segments = provider.get([1, 2])
    assert segments is not None
    assert segments[0].feature_ids == (1, 6)
    assert 3 not in segments[0].feature_ids


def test_stale_cluster_selection_semantics_cache_is_rejected(tmp_path: Path):
    """D-dedup review Finding 1 (HIGH): a clustered context cache written
    under an OLD ``cluster_selection_semantics`` value (e.g. by main
    pre-D-dedup, whose serialized rows were selected by the retired
    multinomial/raw-float/adaptive-negative-factor algorithm) must be
    REJECTED by the current loader, not silently accepted and reused.
    ``feature_selection_hash`` (the cache path/identity key) has no
    implementation-version input, so an untouched pre-D-dedup cache
    directory sits at the SAME path with the SAME identity and would pass
    every other metadata check -- only the semantics-version bump (this
    test) stops training from silently loading stale, wrong rows.

    Simulates an old-semantics cache by writing one normally (which stamps
    the CURRENT, post-bump semantics value) and then tampering the
    on-disk manifest to hold the retired v1 value -- exactly what an
    untouched pre-D-dedup cache directory looks like to the new loader.
    """
    clusters = _cluster_state(tmp_path, clusters=[[1, 3]])
    vocab = effective_feature_vocab(clusters, N_FEATURES)
    assert vocab is not None
    identity = CacheIdentity("model", "sae", 0, "max_pool", "f" * 64)
    metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=clusters,
            feature_vocab=vocab,
            stats_source_sha256=STATS_SHA,
        ),
    }
    assert metadata[CLUSTER_SELECTION_SEMANTICS_KEY] == CLUSTER_SELECTION_SEMANTICS
    assert CLUSTER_SELECTION_SEMANTICS != "post_mask_representative_max_percentile_v1", (
        "the bump must actually change the value, or this test proves nothing"
    )
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        16,
        hash_leaf="f" * 16,
    )
    manifest_path = save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                [1, 2],
                [ContextFeatureSegment((1, 6), (3, 4), 0, 1)],
            )
        ],
        identity=identity,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        metadata=metadata,
    )
    # Sanity: the freshly-written (current-semantics) cache loads fine
    # before tampering.
    SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_n_sae_features=N_FEATURES,
        expected_num_buckets=N_BUCKETS,
        expected_metadata=metadata,
        expected_feature_vocab=vocab,
    )

    # Tamper the on-disk manifest to simulate a STALE (pre-bump, e.g.
    # origin/main pre-D-dedup) cache -- same path, same identity, old rows.
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"][CLUSTER_SELECTION_SEMANTICS_KEY] = (
        "post_mask_representative_max_percentile_v1"
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContextCacheValidationError, match="cluster_selection_semantics"):
        SafetensorsContextFeatureProvider(
            cache_dir,
            identity=identity,
            expected_n_sae_features=N_FEATURES,
            expected_num_buckets=N_BUCKETS,
            expected_metadata=metadata,
            expected_feature_vocab=vocab,
        )


def test_same_effective_vocab_with_different_cluster_membership_is_rejected(
    tmp_path: Path,
):
    first_clusters = _cluster_state(
        tmp_path,
        clusters=[[1, 3], [2, 4]],
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    second_clusters = _cluster_state(
        other_dir,
        clusters=[[1, 4], [2, 3]],
    )
    first_vocab = effective_feature_vocab(first_clusters, N_FEATURES)
    second_vocab = effective_feature_vocab(second_clusters, N_FEATURES)
    assert first_vocab is not None
    assert first_vocab == second_vocab

    identity = CacheIdentity("model", "sae", 0, "max_pool", "d" * 64)
    first_metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=first_clusters,
            feature_vocab=first_vocab,
            stats_source_sha256=STATS_SHA,
        ),
    }
    second_metadata = {
        "model_max_seq_len": 16,
        **clustered_context_cache_metadata(
            feature_mask=None,
            feature_clusters=second_clusters,
            feature_vocab=second_vocab,
            stats_source_sha256=STATS_SHA,
        ),
    }
    assert first_metadata["feature_vocab"] == second_metadata["feature_vocab"]
    assert (
        first_metadata["feature_clusters_descriptor"]
        != second_metadata["feature_clusters_descriptor"]
    )
    cache_dir = context_length_cache_dir(
        tmp_path / "cache",
        16,
        hash_leaf="d" * 16,
    )
    save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                [1, 2],
                [ContextFeatureSegment((1, 2), (3, 3), 0, 1)],
            )
        ],
        identity=identity,
        n_sae_features=N_FEATURES,
        num_buckets=N_BUCKETS,
        metadata=first_metadata,
    )

    with pytest.raises(ContextCacheValidationError, match="metadata mismatch"):
        SafetensorsContextFeatureProvider(
            cache_dir,
            identity=identity,
            expected_n_sae_features=N_FEATURES,
            expected_num_buckets=N_BUCKETS,
            expected_metadata=second_metadata,
            expected_feature_vocab=second_vocab,
        )


def test_unclustered_hash_mismatch_message_still_names_stats_source_sha256():
    """Codex cross-review fix (Finding 4, MEDIUM): resolve_feature_selection_
    hash's mismatch error message is now CONDITIONAL on whether the config is
    clustered (the two branches of _canonical_hash_from_conditioning hash a
    genuinely different input set). This is the U-invariant proof for that
    shared-code change: for an UNCLUSTERED cond, the produced message text
    must be byte-identical to what it always was -- naming
    stats_source_sha256, NOT seed/feature_mask/feature_clusters.
    """
    identity_cfg = OmegaConf.create({"feature_selection_hash": "f" * 64})
    cond = OmegaConf.create(
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
            },
            "negative_selection": None,
            "num_buckets": N_BUCKETS,
            "stats_source_sha256": STATS_SHA,
        }
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_feature_selection_hash(identity_cfg, cond)
    message = str(excinfo.value)
    assert (
        "conditioning.feature_selection, conditioning.negative_selection, "
        "conditioning.num_buckets, and conditioning.stats_source_sha256"
    ) in message
    assert "conditioning.seed" not in message
    assert "conditioning.feature_mask" not in message
    assert "conditioning.feature_clusters" not in message


def test_clustered_hash_mismatch_message_names_seed_not_stats(tmp_path: Path):
    """Codex cross-review fix (Finding 4, MEDIUM): for a CLUSTERED cond, the
    mismatch message must name conditioning.seed/feature_mask/feature_clusters
    (the clustered branch's real hash inputs) and must NOT claim
    stats_source_sha256 is part of the hash -- the pre-fix message was
    actively wrong for clustered configs (it always named
    stats_source_sha256, even though the clustered branch's delta_config=None
    never folds it in).
    """
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 3], [2, 4]])
    identity_cfg = OmegaConf.create({"feature_selection_hash": "f" * 64})
    cond = _identity_cond(cluster_path, seed=17)
    with pytest.raises(ValueError) as excinfo:
        resolve_feature_selection_hash(identity_cfg, cond)
    message = str(excinfo.value)
    assert "conditioning.seed" in message
    assert "conditioning.feature_mask" in message
    assert "conditioning.feature_clusters" in message
    assert "conditioning.stats_source_sha256" not in message


# ---------------------------------------------------------------------------
# D-dedup (PR #739 Track D) regression anchors: pin the NEW (A5-delegated)
# behavior directly against production ``_select_and_bucket_segment``, not
# just the frozen-vs-delegating differential in
# tests/steering/test_d_dedup_clustered_selection_differential.py. Each test
# cites the differential cell it is anchoring.
# ---------------------------------------------------------------------------


def _identity_pct_stats(n_sae_features: int) -> dict[str, PerFeatureStat]:
    """Percentile table row p_k = k (k=1..99): magnitude_to_percentile(fid,
    mag, table) == mag for mag in (1, 99) exactly -- lets a fixture target
    an exact percentile by setting the raw magnitude to it directly. Mirrors
    test_d_dedup_clustered_selection_differential.py's ``_identity_stats``.
    """
    out: dict[str, PerFeatureStat] = {}
    for fid in range(n_sae_features):
        out[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=100.0,
            median_mag=0.5,
            percentiles={f"p{k}": float(k) for k in range(1, 100)},
        )
    return out


def test_axis_a_quantization_flips_bucket_at_boundary_regression_anchor(
    tmp_path: Path,
):
    """Regression anchor for differential Cell B1
    (test_d_dedup_clustered_selection_differential.py::
    test_cell_b1_bucket_boundary_crossing_from_quantization), pinned
    directly against production ``_select_and_bucket_segment``. A single
    candidate at raw percentile 79.9% quantizes to uint8 byte 204, which
    decodes to EXACTLY 0.8 (204/255) -- a clean effective_buckets=5
    boundary. floor(0.799 * 5) = 3 was the pre-delegation (raw-float)
    bucket; floor(0.8 * 5) = 4 is the post-delegation (uint8-quantized,
    training-currency) bucket for the SAME selected feature.
    """
    n: PositiveInt = PositiveInt.parse(16)
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[10, 11]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.normal,
            mean=1.0,
            std=0.01,
            min_k=1,
            max_k=2,
            mode=SamplingMode.top_k_percentile,
        ),
        negative_selection=None,
        per_feature_stats=_identity_pct_stats(n),
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=1,
    )
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([5], dtype=t.int64),
        t.tensor([79.9], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )
    assert feature_ids.tolist() == [5]
    assert buckets.tolist() == [4], (
        "uint8-quantized percentile 204/255=0.8 exactly -> floor(0.8*5)=4 "
        "(pre-delegation raw-float would have given bucket 3)"
    )


def test_axis_b_weighted_mode_multi_candidate_regression_anchor(tmp_path: Path):
    """D-dedup review Finding 3 (differential coverage gap): a DIRECT
    production regression anchor for axis B, pinned against the SHIPPED
    ``_select_and_bucket_segment`` (not just the differential's copied
    prototype in test_d_dedup_clustered_selection_differential.py, which a
    production-only reversion -- e.g. re-introducing ``sample_features``'s
    ``t.multinomial`` call in the real code -- would leave green). 4
    distinct candidates (20/40/60/80%), ``SamplingMode.weighted`` (the
    dataclass default + every production fixture's mode), fixed seed=7001:
    production (Gumbel-max via A5) selects ``[3, 2]``/buckets ``[3, 2]``;
    the frozen pre-delegation legacy body (``t.multinomial``) selects
    ``[3, 4]``/buckets ``[3, 4]`` for the SAME seed -- confirmed to diverge,
    so a revert of the shipped sampling algorithm trips this anchor.
    """
    n: PositiveInt = PositiveInt.parse(16)
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[10, 11]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
            mode=SamplingMode.weighted,
        ),
        negative_selection=None,
        per_feature_stats=_identity_pct_stats(n),
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=7001,
    )
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 2, 3, 4], dtype=t.int64),
        t.tensor([20.0, 40.0, 60.0, 80.0], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )
    assert feature_ids.tolist() == [3, 2], (
        "Gumbel-max weighted selection at seed=7001 -- a t.multinomial "
        "revert would select [3, 4] instead (verified against the frozen "
        "legacy body)"
    )
    assert buckets.tolist() == [3, 2]


def _axis_c_stats(n: int, *, missing_fid: int | None) -> dict[str, PerFeatureStat]:
    stats = {}
    for fid in range(n):
        if fid == missing_fid:
            continue  # NO stats -> all-zero percentile-table row
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles={f"p{p}": float(p) * 0.01 for p in range(1, 100)},
        )
    return stats


def test_axis_c_masked_stat_less_feature_passes_loud_regression_anchor(
    tmp_path: Path,
    caplog,
):
    """D-dedup review Finding 2 (owner decision), pinned directly against
    production ``_select_and_bucket_segment``. Case (a): a masked-out
    feature with NO percentile statistics (an all-zero percentile-table
    row) no longer raises -- the build now PASSES, with the masked
    stat-less feature excluded (pre-detected before scoring) and a loud
    ``logger.error`` surfacing the mismatch. Selection is correct: the
    masked feature never appears, and the non-masked candidate is
    selected exactly as it always was.
    """
    n: PositiveInt = PositiveInt.parse(8)
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 2]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    stats = _axis_c_stats(n, missing_fid=3)
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=42,
        feature_mask=mask,
    )
    caplog.clear()
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 3], dtype=t.int64),  # 3 is masked AND has zero stats
        t.tensor([0.2, 0.5], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )
    assert feature_ids.tolist() == [1], "masked stat-less feature 3 must be excluded, not raise"
    assert 3 not in feature_ids.tolist()
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error_records, "expected a loud logger.error for the masked stat-less feature"
    assert "3" in error_records[0].message
    assert "no percentile statistics" in error_records[0].message


def test_axis_c_non_masked_stat_less_feature_still_raises_regression_anchor(
    tmp_path: Path,
):
    """Case (b) of Finding 2: a NON-masked feature with no percentile
    statistics is a genuinely different anomaly -- it WOULD affect the
    selection -- and must keep raising, pinned directly against production
    ``_select_and_bucket_segment``.
    """
    n: PositiveInt = PositiveInt.parse(8)
    mask = FeatureMask(FeatureMaskMode.blacklist, (7,))  # does NOT cover feature 3
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 2]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    stats = _axis_c_stats(n, missing_fid=3)
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=42,
        feature_mask=mask,
    )
    with pytest.raises(ValueError, match="Percentile statistics are missing"):
        _select_and_bucket_segment(
            t.tensor([1, 3], dtype=t.int64),  # 3 has zero stats but is NOT masked
            t.tensor([0.2, 0.5], dtype=t.float32),
            pipeline=pipeline,
            n_sae_features=n,
            num_buckets=N_BUCKETS,
            source="unit",
            feature_clusters=clusters,
        )


def test_axis_c_ordinary_no_stat_less_case_unchanged_regression_anchor(tmp_path: Path):
    """Case (c) of Finding 2: with a mask configured but no stat-less
    feature anywhere, production behavior is unaffected by the new
    pre-filter (nothing is dropped, no log fires).
    """
    n: PositiveInt = PositiveInt.parse(8)
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[1, 2]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    stats = _axis_c_stats(n, missing_fid=None)  # every feature has stats
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=None,
        per_feature_stats=stats,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=42,
        feature_mask=mask,
    )
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1, 3], dtype=t.int64),  # 3 is masked but HAS stats
        t.tensor([0.2, 0.5], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )
    assert feature_ids.tolist() == [1]


def test_axis_d_small_pool_negative_sampling_regression_anchor(tmp_path: Path):
    """D-dedup review Finding 3 (differential coverage gap): a DIRECT
    production regression anchor for axis D, pinned against the SHIPPED
    ``_select_and_bucket_segment`` (not just the differential's copied
    prototype -- a production-only reversion of the fixed oversample
    factor back to the adaptive ``max(3, n_sae_features//pool_size+1)``
    formula would leave the differential green but should trip this
    anchor). n_sae_features=512, a 24-id whitelist (~4.7% pool), negative
    mean=5/min_k=3/max_k=8, fixed seed=8001: production (A5's fixed
    ``MIN_OVERSAMPLE_FACTOR=3``) selects negative ids, in RNG-draw order,
    ``[6, 11, 14, 21, 3, 7]``; the frozen pre-delegation legacy body
    (adaptive factor ``max(3, 512//24+1)=22``) selects ``[6, 21, 14]`` for
    the SAME seed -- confirmed to diverge (different set AND different
    draw order).
    """
    n_sae: PositiveInt = PositiveInt.parse(512)
    allowed_ids = list(range(24))
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(allowed_ids))
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[100, 101]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n_sae)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, mask)
    stats = {}
    for fid in allowed_ids + [100, 101]:
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles={f"p{p}": float(p) * 0.01 for p in range(1, 100)},
        )
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=3,
            max_k=8,
        ),
        per_feature_stats=stats,
        n_sae_features=n_sae,
        num_buckets=N_BUCKETS,
        seed=8001,
        feature_mask=mask,
    )
    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([1], dtype=t.int64),
        t.tensor([0.5], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n_sae,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )
    assert feature_ids.tolist() == [1, 6, 11, 14, 21, 3, 7], (
        "fixed-oversample-factor negative draw at seed=8001 -- an adaptive-"
        "factor revert would select [1, 6, 21, 14] instead (verified "
        "against the frozen legacy body)"
    )
    assert buckets.tolist() == [2, 4, 4, 4, 4, 4, 4]


def test_empty_positive_segment_no_longer_draws_negatives_regression_anchor(
    tmp_path: Path,
):
    """Regression anchor for differential Cell G1/G2
    (test_d_dedup_clustered_selection_differential.py::
    test_cell_g1_raw_empty_with_negatives_legacy_draws_new_does_not /
    test_cell_g2_post_mask_empty_with_negatives_legacy_draws_new_does_not),
    pinned directly against production ``_select_and_bucket_segment``. A
    raw-empty segment with negative sampling CONFIGURED now returns a fully
    empty result and consumes ZERO RNG (A5's select_clustered_segment
    short-circuits before ever attempting negative sampling). The
    pre-delegation body drew negatives from the full universe regardless of
    whether there were any positive candidates -- a real training-parity
    bug (precompute_selected_features.py's per-segment loop has its own
    caller-level ``if n_feat == 0: continue`` before it would ever call
    select_clustered_segment for a raw-empty segment, so training never
    drew negatives for one either).
    """
    n: PositiveInt = PositiveInt.parse(16)
    cluster_path = _write_clusters(tmp_path / "clusters.json", [[10, 11]])
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, n)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    pipeline = build_feature_pipeline(
        feature_selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=3,
        ),
        per_feature_stats=_identity_pct_stats(n),
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        seed=42,
    )
    assert pipeline.negative_bundle is not None
    state_before = pipeline.negative_bundle.rng.get_state().clone()

    feature_ids, buckets = _select_and_bucket_segment(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.float32),
        pipeline=pipeline,
        n_sae_features=n,
        num_buckets=N_BUCKETS,
        source="unit",
        feature_clusters=clusters,
    )

    assert feature_ids.numel() == 0
    assert buckets.numel() == 0
    assert t.equal(pipeline.negative_bundle.rng.get_state(), state_before), (
        "a raw-empty positive segment must consume ZERO RNG, including for the negative sampler"
    )
