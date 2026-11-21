from __future__ import annotations

import shutil
from contextlib import nullcontext
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from errors import RemovedFeatureError
from scripts.eval import completion_steering_eval
from scripts.preprocess import precompute_selected_features
from scripts.train import conditional_encdec_sft, conditional_encdec_train
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.eval import cache_identity
from steering.overlay_io import OverlayKind, write_manifest
from steering.removed_features import (
    reject_autointerp_clustering_manifest,
    reject_feature_clustering_config,
    reject_feature_clustering_metadata,
)
from steering.stats_types import PerFeatureStat


def _stats(n: int = 8) -> dict[str, PerFeatureStat]:
    return {
        str(i): PerFeatureStat(
            count=1,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles={"p50": 1.0},
        )
        for i in range(n)
    }


@pytest.mark.parametrize(
    "block",
    [
        {"enabled": True},
        {"enabled": False},
        None,
        {"path": "/tmp/old"},
        {"clusters": [[1, 2]]},
    ],
)
def test_top_level_feature_clustering_config_rejected(block) -> None:
    cfg = OmegaConf.create({"feature_clustering": block})
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        reject_feature_clustering_config(cfg, "unit surface")


def test_feature_pipeline_builder_rejects_stale_feature_clustering_config() -> None:
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "type": "threshold",
                "threshold_type": "segment_topk",
                "threshold": 2,
            },
            "feature_clustering": {"enabled": False},
        }
    )
    with pytest.raises(RemovedFeatureError, match="feature pipeline"):
        build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_stats(),
            n_sae_features=8,
            num_buckets=4,
        )


def test_precompute_selected_features_rejects_before_path_validation() -> None:
    cfg = OmegaConf.create({"feature_clustering": None})
    with pytest.raises(RemovedFeatureError, match="precompute_selected_features"):
        precompute_selected_features.main.__wrapped__(cfg)


def test_train_shared_state_rejects_before_ddp_setup() -> None:
    cfg = OmegaConf.create({"feature_clustering": None, "device_type": "cpu"})
    with pytest.raises(RemovedFeatureError, match="conditional_encdec_train shared setup"):
        conditional_encdec_train.build_shared_state(cfg)


def test_train_one_config_rejects_before_shared_state_use() -> None:
    cfg = OmegaConf.create({"feature_clustering": None})
    with pytest.raises(RemovedFeatureError, match="conditional_encdec_train"):
        conditional_encdec_train.train_one_config(cfg, shared=object(), wandb_run=None)  # ty: ignore[invalid-argument-type]


def test_sft_entrypoint_rejects_legacy_cluster_config_during_compose(hydra_compose) -> None:
    config_dir = str(Path(conditional_encdec_sft.__file__).parents[2] / "configs" / "training")
    with pytest.raises(
        ConfigCompositionException,
        match="Could not override 'feature_clustering'",
    ):
        hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            overrides=["feature_clustering=null"],
        )


def test_sft_entrypoint_rejects_legacy_cluster_config_from_inherited_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config_dir = Path(conditional_encdec_sft.__file__).parents[2] / "configs" / "training"
    config_dir = tmp_path / "training"
    shutil.copytree(source_config_dir, config_dir)
    (config_dir / "conditional_encdec_sft_legacy_clustering.yaml").write_text(
        "defaults:\n"
        "  - conditional_encdec_sft\n"
        "  - _self_\n"
        "feature_clustering:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
    initialize = conditional_encdec_sft.initialize_config_dir
    monkeypatch.setattr(
        conditional_encdec_sft,
        "initialize_config_dir",
        lambda **_: initialize(config_dir=str(config_dir), version_base=None),
    )
    monkeypatch.setattr(
        conditional_encdec_sft.sys,
        "argv",
        [
            "conditional_encdec_sft.py",
            "config_name=conditional_encdec_sft_legacy_clustering",
        ],
    )

    with pytest.raises(RemovedFeatureError, match="conditional_encdec_sft"):
        conditional_encdec_sft.main()


def test_completion_eval_entrypoint_rejects_after_compose_before_feature_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create({"feature_clustering": None})
    monkeypatch.setattr(completion_steering_eval.sys, "argv", ["completion_steering_eval.py"])
    monkeypatch.setattr(
        completion_steering_eval, "initialize_config_dir", lambda **_: nullcontext()
    )
    monkeypatch.setattr(completion_steering_eval, "compose", lambda **_: cfg)

    with pytest.raises(RemovedFeatureError, match="completion_steering_eval"):
        completion_steering_eval.main()


def test_eval_cache_identity_rejects_stale_clustering_before_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_cfg = OmegaConf.create({"feature_selection_hash": None})
    cond = OmegaConf.create(
        {
            "feature_selection": {
                "threshold_type": "segment_topk",
                "threshold": 2,
            },
            "stats_source_sha256": "a" * 64,
            "num_buckets": 4,
            "feature_clustering": {"enabled": True},
        }
    )

    def fail_if_hash_called(**_kwargs):
        raise AssertionError("stale feature_clustering reached hash computation")

    monkeypatch.setattr(
        cache_identity,
        "compute_canonical_feature_selection_hash",
        fail_if_hash_called,
    )

    with pytest.raises(RemovedFeatureError, match="eval context cache conditioning"):
        cache_identity.resolve_feature_selection_hash(identity_cfg, cond)


def test_existing_selection_manifest_rejects_stale_clustering_before_reuse(
    tmp_path,
) -> None:
    sel_dir = tmp_path / "sel" / ("a" * 64)
    write_manifest(
        sel_dir,
        "a" * 64,
        OverlayKind.PACKED_ROW,
        metadata={"feature_clustering": {"clusters": [[1, 2]]}},
    )

    with pytest.raises(RemovedFeatureError, match="sel/ manifest"):
        precompute_selected_features._assert_feature_clusters_sel_manifest_current(
            sel_dir,
            None,
            selection_hash="a" * 64,
            packed_rows_hash="b" * 64,
            resume=True,
        )


@pytest.mark.parametrize(
    "meta",
    [
        {"feature_clustering_canonical": {"clusters": [[1, 2]]}},
        {"feature_clustering": {"clusters": [[1, 2]]}},
        {"user_config": {"feature_clustering_canonical": {"clusters": [[1, 2]]}}},
        {"user_config": {"feature_clustering": {"enabled": True}}},
        {"user_config": {"feature_clustering": {"path": "/tmp/old"}}},
        {"user_config": {"feature_clustering": {"checksum": "abc"}}},
        {"user_config": {"feature_clustering": {"clusters": [[1, 2]]}}},
    ],
)
def test_removed_feature_clustering_metadata_rejected(meta) -> None:
    with pytest.raises(RemovedFeatureError, match="metadata surface"):
        reject_feature_clustering_metadata(meta, "metadata surface")


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {"feature_clustering_canonical": None},
        {"feature_clustering": None},
        {"user_config": None},
        {"user_config": {}},
        {"user_config": {"feature_clustering": None}},
        {"user_config": {"feature_clustering": {"enabled": False}}},
    ],
)
def test_unclustered_legacy_metadata_forms_allowed(meta) -> None:
    reject_feature_clustering_metadata(meta, "metadata surface")


@pytest.mark.parametrize(
    "manifest",
    [
        {"clustering_enabled": True},
        {"feature_clustering": {"enabled": False}},
        {"feature_clustering_canonical": {"clusters": [[1, 2]]}},
        {"targets": [{"kind": "cluster", "feature_ids": [1], "target_key": "cluster:1"}]},
        {"targets": [{"kind": "feature", "feature_ids": [1, 2], "target_key": "feature:1"}]},
        {"targets": [{"kind": "feature", "feature_ids": [1], "target_key": "cluster:1"}]},
    ],
)
def test_autointerp_cluster_manifest_forms_rejected(manifest) -> None:
    with pytest.raises(RemovedFeatureError, match="autointerp manifest"):
        reject_autointerp_clustering_manifest(manifest, "autointerp manifest")


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"clustering_enabled": False},
        {"feature_clustering": None},
        {"feature_clustering_canonical": None},
        {"targets": [{"kind": "feature", "feature_ids": [1], "target_key": "feature:1"}]},
    ],
)
def test_autointerp_legacy_unclustered_manifest_forms_allowed(manifest) -> None:
    reject_autointerp_clustering_manifest(manifest, "autointerp manifest")
