"""The packed-rows YAML must declare the profiling + parallel-compress keys so CLI
overrides like ``profile=true`` don't crash under Hydra struct mode (issue #531; mirrors
#529's 22a2e7c1 fix for the token-boundaries config)."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import ConfigKeyError

from scripts.preprocess.precompute_packed_rows import load_packed_rows_config
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW

CFG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "extraction")


def test_profiling_keys_overridable_via_cli():
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(
            config_name="precompute_packed_rows",
            overrides=[
                "profile=true",
                "profile_summary_output=/tmp/x.json",
                "compress_workers=4",
            ],
        )
    assert cfg.profile is True
    assert cfg.profile_summary_output == "/tmp/x.json"
    assert cfg.compress_workers == 4


def test_remote_raw_input_keys_are_declared_and_overridable():
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(
            config_name="precompute_packed_rows",
            overrides=[
                "remote.enabled=true",
                "remote.staging_dir=/tmp/packed-raw",
                "remote.expected_shard_count=1400",
            ],
        )
    assert cfg.remote.enabled is True
    assert cfg.remote.staging_dir == "/tmp/packed-raw"
    assert cfg.remote.expected_shard_count == 1400
    assert cfg.remote.bucket_id == "henryac/nano-gpt-pretrain-steer"


def test_profiling_defaults_preserve_behavior():
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="precompute_packed_rows")
    assert cfg.profile is False
    assert cfg.profile_summary_output is None
    assert cfg.compress_workers == 0
    assert cfg.context_len == DEFAULT_CONTEXT_WINDOW
    assert "decoder_segment_size" not in cfg
    assert "preserve_percentile_uint8" not in cfg


def test_programmatic_config_loader_uses_complete_strict_schema():
    cfg = load_packed_rows_config(
        {
            "feature_dir": "/tmp/features",
            "sae_width": "65k",
            "max_shards": 4,
        }
    )
    assert cfg.buffer_size == 1_000
    assert cfg.rows_per_shard == 16_384
    assert cfg.zstd_level == 12
    assert cfg.resume_from_shard is None
    assert cfg.remote.enabled is False

    with pytest.raises(ConfigKeyError, match="decoder_segment_size"):
        load_packed_rows_config({"decoder_segment_size": 128})
    with pytest.raises(ConfigKeyError, match="preserve_percentile_uint8"):
        load_packed_rows_config({"preserve_percentile_uint8": True})
