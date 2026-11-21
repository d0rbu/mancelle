"""Pin: startup fails fast when in-training evals are enabled but no cache
is configured. No silent fallback to all-masked eval inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_train as _encdec_train_mod
from scripts.train.conditional_encdec_train import (
    _assert_remote_packed_row_manifest_identity,
    _feature_index_percentile_checksum_for_training,
    _feature_index_required_token_boundary_stems_for_training,
    _feature_index_tokenizer_checksum_for_training,
    _load_steering_probe_model,
    _validate_eval_config,
)
from steering.conditional.feature_index import FeatureIndex
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    OverlayStaleError,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.remote_training import RemoteTrainingConfig
from steering.sae_resolver import L0Tier
from tests.steering._helpers import make_valid_shard_data


def _cfg(**overrides):
    base = {
        "core_metric_every": 0,
        "sample_every": 0,
        "conditioning": {"context_cache_dir": None},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_validation_passes_when_evals_disabled():
    _validate_eval_config(_cfg())


def test_validation_passes_when_cache_dir_set_and_evals_enabled():
    _validate_eval_config(_cfg(
        core_metric_every=1000,
        conditioning={"context_cache_dir": "/some/path", "identity": {}},
    ))


def test_validation_fails_when_core_metric_enabled_without_cache():
    with pytest.raises(ValueError, match="conditioning.context_cache_dir"):
        _validate_eval_config(_cfg(core_metric_every=1000))


def test_validation_fails_when_sample_every_enabled_without_cache():
    with pytest.raises(ValueError, match="conditioning.context_cache_dir"):
        _validate_eval_config(_cfg(sample_every=500))


def test_validation_error_message_names_the_cache_builder_script():
    with pytest.raises(ValueError, match="build_eval_context_cache"):
        _validate_eval_config(_cfg(core_metric_every=1000))


def test_validation_error_message_lists_required_suites():
    with pytest.raises(ValueError, match="core.*samples|samples.*core"):
        _validate_eval_config(_cfg(core_metric_every=1000, sample_every=500))


def test_validation_passes_when_only_one_eval_disabled_and_other_enabled_with_cache():
    _validate_eval_config(_cfg(
        core_metric_every=1000,
        sample_every=0,
        conditioning={"context_cache_dir": "/some/path", "identity": {}},
    ))


def test_validation_missing_conditioning_block_treated_as_unset_cache():
    """conditioning entirely missing -> same as context_cache_dir=null."""
    cfg = OmegaConf.create({
        "core_metric_every": 1000,
        "sample_every": 0,
    })
    with pytest.raises(ValueError, match="conditioning.context_cache_dir"):
        _validate_eval_config(cfg)


def test_training_probe_loader_coerces_gemmascope2_l0_tier(monkeypatch):
    captured = {}

    def _capture_probe_loader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        _encdec_train_mod,
        "load_probe_model",
        _capture_probe_loader,
    )
    cfg = OmegaConf.create(
        {
            "sae": {
                "release": "gemmascope_2",
                "model": "gemma-3-4b-pt",
                "activation_type": "residual",
                "layer": 22,
                "width": "65k",
                "l0": "medium",
            }
        }
    )

    result = _load_steering_probe_model(cfg, "cpu")

    assert result is not None
    assert captured["l0"] is L0Tier.medium
    assert captured["sae_release"] == "gemmascope_2"
    assert captured["sae_width"] == "65k"


def test_clustered_precomputed_training_does_not_require_raw_token_overlay():
    cfg = OmegaConf.create(
        {
            "feature_clusters": {"path": "/artifacts/clusters.yaml"},
            "use_precomputed_batches": True,
            "steering_eval_every": 0,
        }
    )

    assert _feature_index_tokenizer_checksum_for_training(cfg, "tokenizer-sha") is None


def _remote_training_config() -> RemoteTrainingConfig:
    remote = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 1,
            "checkpoint_retry_timeout_seconds": 1,
            "retry_interval_seconds": 0.01,
            "resume_latest": True,
            "train_packed": {},
        }
    )
    assert remote is not None
    return remote


def test_remote_clustered_precomputed_training_does_not_require_raw_percentile_overlay():
    cfg = OmegaConf.create(
        {
            "feature_clusters": {"path": "/artifacts/clusters.yaml"},
            "use_precomputed_batches": True,
            "steering_eval_every": 250,
        }
    )

    assert (
        _feature_index_percentile_checksum_for_training(
            cfg,
            _remote_training_config(),
            "percentile-sha",
        )
        is None
    )


@pytest.mark.parametrize(
    ("remote", "feature_clusters", "use_precomputed_batches"),
    [
        (False, {"path": "/artifacts/clusters.yaml"}, True),
        (True, None, True),
        (True, {"path": "/artifacts/clusters.yaml"}, False),
    ],
)
def test_other_training_modes_keep_raw_percentile_overlay_guard(
    remote,
    feature_clusters,
    use_precomputed_batches,
):
    cfg = OmegaConf.create(
        {
            "feature_clusters": feature_clusters,
            "use_precomputed_batches": use_precomputed_batches,
        }
    )

    assert (
        _feature_index_percentile_checksum_for_training(
            cfg,
            _remote_training_config() if remote else None,
            "percentile-sha",
        )
        == "percentile-sha"
    )


def test_remote_percentile_bypass_allows_intentionally_partial_feature_stage(
    tmp_path: Path,
):
    checksum = "a" * 64
    base_paths = [
        save_features(
            make_valid_shard_data(seed=index),
            tmp_path / f"source_{index:05d}_features",
        )
        for index in range(2)
    ]
    pct_dir = overlay_dir(tmp_path, OverlayKind.PERCENTILE, checksum)
    write_manifest(
        pct_dir,
        checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=base_paths,
    )
    for index, base_path in enumerate(base_paths):
        shard = make_valid_shard_data(seed=index)
        save_overlay_shard(
            {
                TensorKey.FEATURE_PERCENTILES: t.zeros(
                    shard["feature_ids"].numel(),
                    dtype=t.uint8,
                )
            },
            overlay_shard_path(pct_dir, f"source_{index:05d}"),
        )
        assert base_path.is_file()
    base_paths[1].unlink()

    cfg = OmegaConf.create(
        {
            "feature_clusters": {"path": "/artifacts/clusters.yaml"},
            "use_precomputed_batches": True,
        }
    )
    effective_checksum = _feature_index_percentile_checksum_for_training(
        cfg,
        _remote_training_config(),
        checksum,
    )
    remote_index = FeatureIndex(str(tmp_path), percentile_checksum=effective_checksum)
    assert remote_index.shard_stems == ["source_00000"]

    local_checksum = _feature_index_percentile_checksum_for_training(
        cfg,
        None,
        checksum,
    )
    with pytest.raises(OverlayStaleError, match="current base shards hash"):
        FeatureIndex(str(tmp_path), percentile_checksum=local_checksum)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checksum", "wrong"),
        ("kind", OverlayKind.PERCENTILE.value),
        ("pct_checksum", "wrong"),
        ("tok_checksum", "wrong"),
    ],
)
def test_remote_packed_manifest_identity_requires_every_full_checksum_field(
    tmp_path: Path,
    field: str,
    value: str,
):
    percentile_checksum = "a" * 64
    tokenizer_checksum = "b" * 64
    packed_dir = tmp_path / "packed"
    write_manifest(
        packed_dir,
        f"{percentile_checksum}:{tokenizer_checksum}",
        OverlayKind.PACKED_ROW,
        metadata={
            "pct_checksum": percentile_checksum,
            "tok_checksum": tokenizer_checksum,
        },
    )
    manifest_path = packed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field in {"checksum", "kind"}:
        manifest[field] = value
    else:
        manifest["metadata"][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        _encdec_train_mod.RemoteTrainingError,
        match="packed-row manifest identity mismatch",
    ):
        _assert_remote_packed_row_manifest_identity(
            packed_dir,
            percentile_checksum=percentile_checksum,
            tokenizer_checksum=tokenizer_checksum,
            label="test family",
        )


@pytest.mark.parametrize(
    ("feature_clusters", "use_precomputed_batches", "steering_eval_every"),
    [
        (None, True, 0),
        ({"path": "/artifacts/clusters.yaml"}, False, 0),
        ({"path": "/artifacts/clusters.yaml"}, True, 250),
    ],
)
def test_boundary_consuming_training_modes_keep_raw_token_overlay_guard(
    feature_clusters,
    use_precomputed_batches,
    steering_eval_every,
):
    cfg = OmegaConf.create(
        {
            "feature_clusters": feature_clusters,
            "use_precomputed_batches": use_precomputed_batches,
            "steering_eval_every": steering_eval_every,
        }
    )

    assert (
        _feature_index_tokenizer_checksum_for_training(cfg, "tokenizer-sha")
        == "tokenizer-sha"
    )


def test_clustered_static_steering_requires_only_derived_validation_token_overlay(
    monkeypatch,
):
    cfg = OmegaConf.create(
        {
            "feature_clusters": {"path": "/artifacts/clusters.yaml"},
            "use_precomputed_batches": True,
            "steering_eval_every": 250,
            "data_dir": "/derived/parquets",
        }
    )
    monkeypatch.setattr(
        _encdec_train_mod,
        "list_parquet_files",
        lambda data_dir: [
            "/derived/parquets/shard_00000.parquet",
            "/derived/parquets/shard_00369.parquet",
        ],
    )

    assert _feature_index_required_token_boundary_stems_for_training(cfg) == {
        "shard_00369"
    }


@pytest.mark.parametrize(
    ("feature_clusters", "use_precomputed_batches", "steering_eval_every"),
    [
        (None, True, 250),
        ({"path": "/artifacts/clusters.yaml"}, False, 250),
        ({"path": "/artifacts/clusters.yaml"}, True, 0),
    ],
)
def test_other_training_modes_do_not_weaken_token_overlay_coverage(
    monkeypatch,
    feature_clusters,
    use_precomputed_batches,
    steering_eval_every,
):
    cfg = OmegaConf.create(
        {
            "feature_clusters": feature_clusters,
            "use_precomputed_batches": use_precomputed_batches,
            "steering_eval_every": steering_eval_every,
        }
    )
    monkeypatch.setattr(
        _encdec_train_mod,
        "list_parquet_files",
        lambda _data_dir: pytest.fail("non-clustered-static path listed parquets"),
    )

    assert _feature_index_required_token_boundary_stems_for_training(cfg) is None


def test_clustered_static_steering_fails_when_validation_parquet_is_absent(
    monkeypatch,
):
    cfg = OmegaConf.create(
        {
            "feature_clusters": {"path": "/artifacts/clusters.yaml"},
            "use_precomputed_batches": True,
            "steering_eval_every": 250,
            "data_dir": "/derived/parquets",
        }
    )
    monkeypatch.setattr(
        _encdec_train_mod,
        "list_parquet_files",
        lambda _data_dir: [],
    )

    with pytest.raises(FileNotFoundError, match="derived validation parquet"):
        _feature_index_required_token_boundary_stems_for_training(cfg)


def test_make_shared_state_fails_fast_when_evals_enabled_no_cache():
    """End-to-end: the actual shared-state builder entry point fails fast
    when in-training evals are enabled without a cache.

    The validation runs first; we don't need the rest of the function to
    succeed (it'd require a real model checkpoint, etc.).
    """
    # Find the shared-state builder. The function may be named
    # _make_shared_state or build_shared_state depending on local rename.
    builder = (
        getattr(_encdec_train_mod, "_make_shared_state", None)
        or getattr(_encdec_train_mod, "build_shared_state", None)
    )
    assert builder is not None, (
        "expected _make_shared_state or build_shared_state on conditional_encdec_train"
    )

    cfg = OmegaConf.create({
        "core_metric_every": 100,
        "sample_every": 0,
        "conditioning": {"context_cache_dir": None},
    })
    with pytest.raises(ValueError, match="conditioning.context_cache_dir"):
        builder(cfg)


# ---------------------------------------------------------------------------
# Gradient startup gates: sel-shard COVERAGE (not just existence)
# ---------------------------------------------------------------------------


def _touch_shards(d, stems):
    d.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (d / f"{s}.safetensors.zst").write_bytes(b"x")


def test_sel_coverage_missing_lists_uncovered_stems(tmp_path):
    """The loader pairs sel shards per base-stem and silently falls back to
    fused live selection for uncovered stems — the gradient startup gate
    compares stem SETS via this helper so partial coverage fails at startup
    instead of GradientRankingUnsupportedError hours into training."""
    from scripts.train.conditional_encdec_train import _sel_coverage_missing

    packed = tmp_path / "packed"
    sel = tmp_path / "packed" / "sel" / "h"
    _touch_shards(packed, ["shard_00000", "shard_00001", "shard_00002"])
    _touch_shards(sel, ["shard_00000", "shard_00002"])
    assert _sel_coverage_missing(packed, sel) == ["shard_00001"]
    _touch_shards(sel, ["shard_00001"])
    assert _sel_coverage_missing(packed, sel) == []
    # Extra sel shards (stale leftovers) are not an error of THIS check.
    _touch_shards(sel, ["shard_00099"])
    assert _sel_coverage_missing(packed, sel) == []
