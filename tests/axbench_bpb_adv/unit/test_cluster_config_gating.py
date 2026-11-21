from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from steering.eval.axbench_bpb.config import ContextCacheConfig
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_family import ALPHA_AXIS_GCG_METHODS
from steering.eval.cluster_config import CLUSTERED_AXBENCH_BPB_ADV_METHODS

_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "eval"
    / "axbench_bpb_adv"
)
_MASK = {"mode": "whitelist", "path": "/tmp/mask.txt"}
_CLUSTERS = {"path": "/tmp/clusters.yaml"}


def _clustered_config(**overrides: Any) -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/output",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/feature_statistics.safetensors",
        feature_mask=dict(_MASK),
        feature_clusters=dict(_CLUSTERS),
        base_methods=["diffmean_decoder_only"],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_clustered_config_accepts_locked_method_subset() -> None:
    validate_config(_clustered_config())


@pytest.mark.parametrize(
    ("base_methods", "match"),
    [
        (None, "requires an explicit method allowlist"),
        ([], "method allowlist must not be empty"),
        (
            ["diffmean_decoder_only", "diffmean_decoder_only"],
            "method allowlist contains duplicates",
        ),
        (
            ["ours_replace_all_encdec"],
            "contains unsupported methods",
        ),
    ],
)
def test_clustered_config_rejects_invalid_method_allowlists(
    base_methods: list[str] | None,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_config(_clustered_config(base_methods=base_methods))


@pytest.mark.parametrize(
    ("feature_mask", "feature_clusters", "missing"),
    [
        (_MASK, None, "feature_clusters"),
        (None, _CLUSTERS, "feature_mask"),
    ],
)
def test_clustered_config_requires_mask_and_clusters_together(
    feature_mask: dict[str, str] | None,
    feature_clusters: dict[str, str] | None,
    missing: str,
) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match=rf"{missing} is null",
    ):
        validate_config(
            _clustered_config(
                feature_mask=feature_mask,
                feature_clusters=feature_clusters,
            )
        )


def test_clustered_config_rejects_context_cache_force_rebuild() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="force_rebuild=true is forbidden"):
        validate_config(
            _clustered_config(
                context_cache=ContextCacheConfig(force_rebuild=True)
            )
        )


def test_unclustered_config_retains_legacy_none_allowlist_and_force_rebuild() -> None:
    cfg = _clustered_config(
        feature_mask=None,
        feature_clusters=None,
        base_methods=None,
        context_cache=ContextCacheConfig(force_rebuild=True),
    )

    validate_config(cfg)


def test_forced_revalidation_requires_clustered_config() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires feature_clusters",
    ):
        validate_config(
            _clustered_config(
                feature_mask=None,
                feature_clusters=None,
                force_cluster_shard_revalidation=True,
            )
        )


def test_forced_revalidation_rejects_bool_lookalike() -> None:
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="force_cluster_shard_revalidation",
    ):
        AxBenchBpbAdvConfig(force_cluster_shard_revalidation=1)  # ty: ignore[invalid-argument-type]


def test_hydra_default_accepts_new_null_keys_and_preserves_context_width() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        hydra_cfg = compose(
            config_name="axbench_bpb_adv_eval",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )

    cfg = cfg_to_dataclass(hydra_cfg)

    assert cfg.feature_mask is None
    assert cfg.feature_clusters is None
    assert cfg.extraction_max_tokens == 1024


def test_clustered_hydra_config_requires_assets_and_locks_fourteen_methods() -> None:
    """The canonical clustered config selects every ALLOWED method except the
    class-margin alpha pair.

    The allowlist and the canonical selection are different things: the
    ``ours_gcg_alpha*`` methods are PERMITTED under clustered runs (they read the
    same fair bags as ``ours_gcg_encdec``) but are deliberately NOT part of the
    default population -- an alpha sweep is opt-in via its own config, and adding
    it here would change the method set of every existing clustered run.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        unresolved = compose(
            config_name="axbench_bpb_adv_clustered",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )

    assert OmegaConf.is_missing(unresolved.feature_mask, "path")
    assert OmegaConf.is_missing(unresolved.feature_clusters, "path")
    expected_methods = [
        method
        for method in CLUSTERED_AXBENCH_BPB_ADV_METHODS
        if method not in ALPHA_AXIS_GCG_METHODS
    ]
    assert list(unresolved.base_methods) == expected_methods
    # ... and the alpha pair really is allowed, just not selected -- otherwise
    # the filter above would be hiding a missing allowlist entry.
    assert ALPHA_AXIS_GCG_METHODS <= set(CLUSTERED_AXBENCH_BPB_ADV_METHODS)

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        resolved = compose(
            config_name="axbench_bpb_adv_clustered",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
                "feature_mask.path=/tmp/mask.txt",
                "feature_clusters.path=/tmp/clusters.yaml",
                "steering_metric=bpb_class_margin",
            ],
        )

    cfg = cfg_to_dataclass(resolved)
    validate_config(cfg)
    assert cfg.feature_mask == {"mode": "whitelist", "path": "/tmp/mask.txt"}
    assert cfg.feature_clusters == {"path": "/tmp/clusters.yaml"}
