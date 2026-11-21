"""Opt-in two-shard real-data parity and compact-size acceptance gate.

Set ``TOKEN_BOUNDARY_REAL_FEATURE_DIR`` and ``TOKEN_BOUNDARY_REAL_PARQUET_DIR``
to an exact paired production family. This CPU-only gate copies two raw shards,
publishes a filesystem-backed compact source, and compares it with the living
local raw runner. The removed remote-raw reference is commit ``5a726b14d``;
before removal, ``test_remote_artifacts_are_byte_identical_to_local_oracle``
proved that revision byte-identical to the same local oracle under a frozen
clock.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nanochat.tokenizer import get_tokenizer
from scripts.preprocess import precompute_token_boundaries as consumer
from scripts.preprocess import publish_token_boundary_source as publisher
from steering.feature_io import glob_features
from steering.feature_paths import parse_feature_dir
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    compute_base_shard_content_hash,
    overlay_dir,
    overlay_shard_path,
)
from steering.segmentation import compute_tokenizer_checksum
from steering.stats_io import save_tokenizer_checksum
from steering.token_boundary_source import (
    parse_token_boundary_source_manifest,
)
from tests.steering.test_token_boundary_source import FakeBucket

pytestmark = pytest.mark.requires_feature_data


def _require_file(path: Path, *, purpose: str) -> Path:
    if not path.is_file():
        pytest.skip(f"missing real token-boundary {purpose}: {path}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_real_shards_are_byte_identical_and_compact_below_one_percent(
    tmp_path: Path,
) -> None:
    feature_raw = os.environ.get("TOKEN_BOUNDARY_REAL_FEATURE_DIR")
    parquet_raw = os.environ.get("TOKEN_BOUNDARY_REAL_PARQUET_DIR")
    if feature_raw is None or parquet_raw is None:
        pytest.skip(
            "set TOKEN_BOUNDARY_REAL_FEATURE_DIR and "
            "TOKEN_BOUNDARY_REAL_PARQUET_DIR to an exact paired production family"
        )
    source = Path(feature_raw).resolve()
    parquet_source = Path(parquet_raw).resolve()
    shards = glob_features(source)[:2]
    if len(shards) != 2:
        pytest.skip(f"real parity gate requires two raw shards under {source}")
    parsed = parse_feature_dir(source)
    marker = f"s{parsed.segment_size}"
    try:
        marker_index = source.parts.index(marker)
    except ValueError:
        pytest.skip(f"real feature path has no canonical {marker} component: {source}")
    relative_feature_path = Path(*source.parts[marker_index:])
    oracle = tmp_path / "oracle-root" / relative_feature_path
    logical = tmp_path / "logical-root" / relative_feature_path
    oracle.mkdir(parents=True)
    logical.mkdir(parents=True)
    data_dir = tmp_path / "parquets"
    data_dir.mkdir()

    bucket = FakeBucket(tmp_path / "bucket")
    raw_root = bucket._path("raw")
    raw_root.mkdir(parents=True)
    for shard in shards:
        shutil.copyfile(shard, oracle / shard.name)
        shutil.copyfile(shard, raw_root / shard.name)
        stem = shard.name.split("_features", 1)[0]
        parquet = _require_file(
            parquet_source / f"{stem}.parquet",
            purpose=f"parquet for {stem}",
        )
        shutil.copyfile(parquet, data_dir / parquet.name)

    stats = _require_file(
        source / "feature_statistics.safetensors",
        purpose="statistics file",
    )
    shutil.copyfile(stats, oracle / stats.name)
    shutil.copyfile(stats, logical / stats.name)
    try:
        tokenizer = get_tokenizer()
    except Exception as exc:
        pytest.skip(f"real nanochat tokenizer is unavailable: {exc}")
    checksum = compute_tokenizer_checksum(tokenizer)
    save_tokenizer_checksum(oracle / stats.name, checksum)
    save_tokenizer_checksum(logical / stats.name, checksum)
    stats_before = (logical / stats.name).read_bytes()

    staging = tmp_path / "publisher-staging"
    staging.mkdir()
    publisher_cfg = OmegaConf.create(
        {
            "bucket_id": "owner/filesystem-real-parity",
            "raw_prefix": "raw",
            "staging_dir": str(staging),
            "expected_shard_count": 2,
            "retry_timeout_seconds": 1.0,
            "retry_interval_seconds": 0.001,
        }
    )
    assert publisher.run(publisher_cfg, backend=bucket) == 0

    common = {
        "segment_size": parsed.segment_size,
        "data_dir": str(data_dir),
        "workers": 1,
        "dry_run": False,
        "resume": False,
        "force": False,
        "profile": False,
        "profile_trace": False,
        "zstd_level": 12,
    }
    local_cfg = OmegaConf.create({**common, "feature_dir": str(oracle)})
    consumer_staging = tmp_path / "consumer-staging"
    remote_cfg = OmegaConf.create(
        {
            **common,
            "feature_dir": str(logical),
            "remote": {
                "enabled": True,
                "bucket_id": "owner/filesystem-real-parity",
                "local_features_root": str(logical),
                "remote_features_root": "raw",
                "staging_dir": str(consumer_staging),
                "expected_shard_count": 2,
                "retry_timeout_seconds": 1.0,
                "retry_interval_seconds": 0.001,
            },
        }
    )
    frozen = (2026, 1, 2, 3, 4, 5, 4, 2, 0)
    with (
        patch.object(consumer, "get_tokenizer", return_value=tokenizer),
        patch("steering.overlay_io.time.gmtime", return_value=frozen),
    ):
        assert consumer.run(local_cfg) == 0
        with patch(
            "steering.token_boundary_remote.HfBucketBackend",
            return_value=bucket,
        ):
            assert consumer.run(remote_cfg) == 0

    local_output = overlay_dir(oracle, OverlayKind.SEG_BOUNDARY, checksum)
    remote_output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert (remote_output / MANIFEST_FILENAME).read_bytes() == (
        local_output / MANIFEST_FILENAME
    ).read_bytes()
    for shard in shards:
        stem = shard.name.split("_features", 1)[0]
        local_overlay = overlay_shard_path(local_output, stem)
        remote_overlay = overlay_shard_path(remote_output, stem)
        assert remote_overlay.read_bytes() == local_overlay.read_bytes(), (
            stem,
            _sha256(local_overlay),
            _sha256(remote_overlay),
        )
    assert (logical / stats.name).read_bytes() == stats_before

    manifest_paths = list(
        bucket.root.glob("raw/_token_boundary_source/v1/*/manifest.json")
    )
    assert len(manifest_paths) == 1
    compact_manifest = parse_token_boundary_source_manifest(
        manifest_paths[0].read_bytes()
    )
    assert compact_manifest.base_shard_content_hash == compute_base_shard_content_hash(
        sorted(raw_root.glob("*.safetensors.zst"))
    )
    raw_bytes = sum(path.stat().st_size for path in raw_root.glob("*.safetensors.zst"))
    compact_bytes = sum(
        path.stat().st_size
        for path in manifest_paths[0].parent.glob("shard_*_features.safetensors.zst")
    )
    assert compact_bytes <= raw_bytes * 0.01
