"""Config-loading test for splice_feature_shards (issue #590, Task E1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf.errors import ConfigKeyError

from scripts.preprocess.splice_feature_shards import (
    _resolve_run_params,
    _resolve_zstd_level,
    load_splice_config,
)


def test_config_loads_with_documented_defaults():
    config_dir = str(Path(__file__).parents[2] / "configs" / "extraction")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="splice_feature_shards")
    assert cfg.feature_dir is None
    assert cfg.data_dir is None
    assert cfg.output_dir is None
    assert cfg.model is None
    assert cfg.segment_size is None
    assert cfg.context_window == 1024
    assert cfg.cut_fraction == 0.15
    assert cfg.seed == 42
    assert cfg.skip_existing is True
    assert cfg.write_provenance is True
    assert cfg.shard_parallelism == 1
    assert cfg.zstd_threads == 0
    assert cfg.zstd_level is None
    assert cfg.async_feature_save_depth == 0
    assert cfg.profile is False
    assert cfg.profile_summary_output.endswith("splice_feature_shards_profile.json")
    assert cfg.remote.enabled is False
    assert cfg.remote.bucket_id == "d0rb/nano-gpt-pretrain-steer-artifacts"
    assert cfg.remote.local_features_root == "out/data/features"
    assert cfg.remote.remote_features_root == "features"
    assert cfg.remote.staging_dir is None
    assert cfg.remote.expected_shard_count is None
    assert cfg.remote.retry_timeout_seconds == 900.0
    assert cfg.remote.retry_interval_seconds == 5.0


def test_config_has_no_legacy_keys():
    """The config must NOT expose 'p', 'start_shard', or 'end_shard' keys."""
    config_path = (
        Path(__file__).parents[2] / "configs" / "extraction" / "splice_feature_shards.yaml"
    )
    raw = yaml.safe_load(config_path.read_text())
    assert "p" not in raw, "legacy 'p' key must not appear in the config"
    assert "start_shard" not in raw, "legacy 'start_shard' key must not appear in the config"
    assert "end_shard" not in raw, "legacy 'end_shard' key must not appear in the config"


def test_overrides_apply():
    config_dir = str(Path(__file__).parents[2] / "configs" / "extraction")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="splice_feature_shards",
            overrides=[
                "cut_fraction=0.5",
                "seed=7",
                "shard_parallelism=4",
                "async_feature_save_depth=2",
                "zstd_threads=2",
                "zstd_level=8",
            ],
        )
    assert cfg.cut_fraction == 0.5 and cfg.seed == 7
    assert cfg.shard_parallelism == 4
    assert cfg.async_feature_save_depth == 2
    assert cfg.zstd_threads == 2
    assert cfg.zstd_level == 8


def test_programmatic_loader_completes_the_canonical_schema():
    cfg = load_splice_config(
        {
            "feature_dir": "/tmp/features",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
        }
    )

    assert cfg.skip_existing is True
    assert cfg.write_provenance is True
    assert cfg.async_feature_save_depth == 0
    assert cfg.zstd_threads == 0
    assert cfg.zstd_level is None
    assert cfg.profile_summary_output.endswith("splice_feature_shards_profile.json")
    assert cfg.remote.enabled is False


def test_programmatic_loader_rejects_retired_keys():
    with pytest.raises(ConfigKeyError, match="start_shard"):
        load_splice_config({"start_shard": 0})


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "2", None])
def test_invalid_threads_rejected_before_reading_source(tmp_path, value):
    cfg = load_splice_config({"zstd_threads": value})
    with pytest.raises(ValueError, match="zstd_threads must be an integer >= 0"):
        _resolve_run_params(cfg, tmp_path, tmp_path / "missing.safetensors.zst")


@pytest.mark.parametrize("value", [None, 1, 8, 12, 22])
def test_valid_compression_levels(value):
    assert _resolve_zstd_level(load_splice_config({"zstd_level": value})) == value


@pytest.mark.parametrize("value", [True, False, -1, 0, 23, 1.5, "8"])
def test_invalid_level_rejected_before_reading_source(tmp_path, value):
    cfg = load_splice_config({"zstd_level": value})
    with pytest.raises(ValueError, match="zstd_level must be null or an integer"):
        _resolve_run_params(cfg, tmp_path, tmp_path / "missing.safetensors.zst")
