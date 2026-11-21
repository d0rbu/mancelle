"""Parity tests for the split read+zstd and safetensors decode stages."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
import zstandard

from steering.feature_io import (
    FeatureDirectoryMismatchError,
    RegularFileIdentity,
    decode_feature_safetensors_bytes,
    load_feature_safetensors_bytes,
    load_feature_safetensors_bytes_stable,
    load_features,
    load_features_stable,
    read_feature_compressed_bytes_stable,
)
from tests.steering.test_exact_statistics_integration import _write_shards


def _assert_feature_dicts_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, t.Tensor):
            assert actual_value.dtype == expected_value.dtype, key
            assert actual_value.shape == expected_value.shape, key
            assert t.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value, key


@pytest.mark.parametrize("keys", [None, {"offsets", "feature_ids", "feature_mags"}])
def test_split_loader_exactly_matches_public_loader(
    tmp_path: Path,
    keys: set[str] | None,
) -> None:
    path = _write_shards(tmp_path)[0]

    expected = load_features(path, keys=keys)
    actual = decode_feature_safetensors_bytes(
        load_feature_safetensors_bytes(path),
        keys=keys,
    )

    _assert_feature_dicts_equal(actual, expected)


def test_split_stable_loader_exactly_matches_public_stable_loader(
    tmp_path: Path,
) -> None:
    path = _write_shards(tmp_path)[0]
    identity = RegularFileIdentity.capture(path)

    expected = load_features_stable(path, expected_identity=identity)
    actual = decode_feature_safetensors_bytes(
        load_feature_safetensors_bytes_stable(
            path,
            expected_identity=identity,
        )
    )

    _assert_feature_dicts_equal(actual, expected)


def test_stable_compressed_reader_returns_exact_on_disk_bytes(tmp_path: Path) -> None:
    path = _write_shards(tmp_path)[0]
    identity = RegularFileIdentity.capture(path)
    assert read_feature_compressed_bytes_stable(
        path,
        expected_identity=identity,
    ) == path.read_bytes()


def test_prepared_loader_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "shard.bin"
    path.write_bytes(b"unused")

    with pytest.raises(ValueError, match="Unknown feature file format"):
        load_feature_safetensors_bytes(path)


def test_prepared_loader_propagates_zstd_corruption(tmp_path: Path) -> None:
    path = tmp_path / "shard_features.safetensors.zst"
    path.write_bytes(b"not a zstd frame")

    with pytest.raises(zstandard.ZstdError):
        load_feature_safetensors_bytes(path)


def test_prepared_stable_loader_rejects_changed_identity(tmp_path: Path) -> None:
    path = _write_shards(tmp_path)[0]
    identity = RegularFileIdentity.capture(path)
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(FeatureDirectoryMismatchError, match="changed before stable read"):
        load_feature_safetensors_bytes_stable(path, expected_identity=identity)


def test_decode_preserves_subset_validation(tmp_path: Path) -> None:
    path = _write_shards(tmp_path)[0]
    prepared = load_feature_safetensors_bytes(path)

    with pytest.raises(ValueError, match="must also include 'offsets'"):
        decode_feature_safetensors_bytes(prepared, keys={"feature_ids"})
    with pytest.raises(ValueError, match="not present"):
        decode_feature_safetensors_bytes(prepared, keys={"does_not_exist"})
