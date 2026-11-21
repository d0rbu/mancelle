from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest
import torch as t

from nanochat.atomic_io import atomic_write_parquet
from tests.scripts import _splice_artifact_compare as cmp


def test_feature_comparator_detects_tensor_value_dtype_and_metadata(monkeypatch, tmp_path):
    left = tmp_path / "left.safetensors.zst"
    right = tmp_path / "right.safetensors.zst"
    left.touch()
    right.touch()

    cases = [
        (
            {
                left: {"x": t.tensor([1], dtype=t.int32), "n_docs": 1},
                right: {"x": t.tensor([2], dtype=t.int32), "n_docs": 1},
            },
            "tensor values differ",
        ),
        (
            {
                left: {"x": t.tensor([1], dtype=t.int32), "n_docs": 1},
                right: {"x": t.tensor([1], dtype=t.int64), "n_docs": 1},
            },
            "dtype mismatch",
        ),
        (
            {
                left: {"x": t.tensor([1], dtype=t.int32), "n_docs": 1},
                right: {"x": t.tensor([1], dtype=t.int32), "n_docs": 2},
            },
            "metadata mismatch",
        ),
    ]

    for payloads, match in cases:
        monkeypatch.setattr(
            cmp,
            "load_features",
            lambda path, payloads=payloads: payloads[Path(path)],
        )
        with pytest.raises(AssertionError, match=match):
            cmp.assert_feature_shards_equal(left, right)


def test_parquet_comparator_detects_bytes_and_rows(tmp_path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    atomic_write_parquet(pa.table({"text": ["alpha", "beta"]}), left)
    atomic_write_parquet(pa.table({"text": ["alpha", "changed"]}), right)

    with pytest.raises(AssertionError, match="parquet row text differs"):
        cmp.assert_parquets_equal(left, right)


def test_provenance_comparator_detects_byte_and_json_mismatch(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"a": 1}, indent=2), encoding="utf-8")
    right.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(AssertionError, match="provenance bytes differ"):
        cmp.assert_provenance_equal(left, right)

    right.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(AssertionError, match="provenance JSON differs"):
        cmp.assert_provenance_equal(left, right)


def test_directory_comparator_detects_missing_stem(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    (left / "_parquets").mkdir(parents=True)
    (right / "_parquets").mkdir(parents=True)
    (left / "_splice_provenance").mkdir()
    (right / "_splice_provenance").mkdir()
    (left / "shard_00000_features.safetensors.zst").touch()
    (right / "shard_00000_features.safetensors.zst").touch()
    (left / "_splice_provenance" / "shard_00000.json").write_text(
        json.dumps({"same": True}), encoding="utf-8"
    )
    (right / "_splice_provenance" / "shard_00000.json").write_text(
        json.dumps({"same": True}), encoding="utf-8"
    )
    atomic_write_parquet(pa.table({"text": ["alpha"]}), left / "_parquets" / "shard_00000.parquet")

    with pytest.raises(AssertionError, match="parquet stem mismatch"):
        cmp.assert_splice_dirs_equal(left, right)


def test_directory_comparator_requires_artifact_subdirs(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    with pytest.raises(AssertionError, match="_parquets dir missing"):
        cmp.assert_splice_dirs_equal(left, right)


def test_directory_comparator_rejects_empty_artifact_classes(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "_parquets").mkdir(parents=True)
        (root / "_splice_provenance").mkdir()

    with pytest.raises(AssertionError, match="no feature shards found"):
        cmp.assert_splice_dirs_equal(left, right)
