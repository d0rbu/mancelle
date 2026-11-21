"""Test-only artifact comparators for feature-shard splice outputs."""

from __future__ import annotations

import json
from pathlib import Path

import torch as t

from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.parquet_io import read_parquet_texts


def _fail(errors: list[str]) -> None:
    if errors:
        raise AssertionError("; ".join(errors))


def assert_feature_shards_equal(left: str | Path, right: str | Path) -> None:
    left_path = Path(left)
    right_path = Path(right)
    left_data = load_features(left_path)
    right_data = load_features(right_path)
    left_tensor_keys = {k for k, v in left_data.items() if isinstance(v, t.Tensor)}
    right_tensor_keys = {k for k, v in right_data.items() if isinstance(v, t.Tensor)}
    left_scalar_keys = set(left_data) - left_tensor_keys
    right_scalar_keys = set(right_data) - right_tensor_keys
    errors: list[str] = []
    if left_tensor_keys != right_tensor_keys:
        errors.append(
            f"tensor key mismatch: {sorted(left_tensor_keys)} != {sorted(right_tensor_keys)}"
        )
    if left_scalar_keys != right_scalar_keys:
        errors.append(
            f"metadata key mismatch: {sorted(left_scalar_keys)} != {sorted(right_scalar_keys)}"
        )
    for key in sorted(left_tensor_keys & right_tensor_keys):
        lv = left_data[key]
        rv = right_data[key]
        if lv.dtype != rv.dtype:
            errors.append(f"{key}: dtype mismatch {lv.dtype} != {rv.dtype}")
        if tuple(lv.shape) != tuple(rv.shape):
            errors.append(f"{key}: shape mismatch {tuple(lv.shape)} != {tuple(rv.shape)}")
        if lv.dtype == rv.dtype and tuple(lv.shape) == tuple(rv.shape) and not t.equal(lv, rv):
            errors.append(f"{key}: tensor values differ")
    for key in sorted(left_scalar_keys & right_scalar_keys):
        if left_data[key] != right_data[key]:
            errors.append(f"{key}: metadata mismatch {left_data[key]!r} != {right_data[key]!r}")
    _fail(errors)


def assert_parquets_equal(left: str | Path, right: str | Path) -> None:
    left_path = Path(left)
    right_path = Path(right)
    errors: list[str] = []
    if left_path.read_bytes() != right_path.read_bytes():
        errors.append("parquet bytes differ")
    left_texts = read_parquet_texts(left_path)
    right_texts = read_parquet_texts(right_path)
    if left_texts != right_texts:
        errors.append("parquet row text differs")
    _fail(errors)


def assert_provenance_equal(left: str | Path, right: str | Path) -> None:
    left_path = Path(left)
    right_path = Path(right)
    errors: list[str] = []
    left_bytes = left_path.read_bytes()
    right_bytes = right_path.read_bytes()
    if left_bytes != right_bytes:
        errors.append("provenance bytes differ")
    left_json = json.loads(left_bytes.decode("utf-8"))
    right_json = json.loads(right_bytes.decode("utf-8"))
    if left_json != right_json:
        errors.append("provenance JSON differs")
    _fail(errors)


def _stemmed(paths: list[Path], *, suffix: str | None = None) -> dict[str, Path]:
    if suffix is None:
        return {shard_stem_from_path(path): path for path in paths}
    return {path.name.removesuffix(suffix): path for path in paths}


def assert_splice_dirs_equal(left: str | Path, right: str | Path) -> None:
    left_dir = Path(left)
    right_dir = Path(right)
    errors: list[str] = []
    for label, directory in (("left", left_dir), ("right", right_dir)):
        if not directory.is_dir():
            errors.append(f"{label} splice dir missing: {directory}")
        for child in ("_parquets", "_splice_provenance"):
            child_path = directory / child
            if not child_path.is_dir():
                errors.append(f"{label} {child} dir missing: {child_path}")
    _fail(errors)

    left_features = _stemmed(glob_features(left_dir))
    right_features = _stemmed(glob_features(right_dir))
    left_parquets = _stemmed(sorted((left_dir / "_parquets").glob("*.parquet")), suffix=".parquet")
    right_parquets = _stemmed(sorted((right_dir / "_parquets").glob("*.parquet")), suffix=".parquet")
    left_prov = _stemmed(sorted((left_dir / "_splice_provenance").glob("*.json")), suffix=".json")
    right_prov = _stemmed(sorted((right_dir / "_splice_provenance").glob("*.json")), suffix=".json")
    errors = []
    if not left_features:
        errors.append(f"no feature shards found in {left_dir}")
    if not right_features:
        errors.append(f"no feature shards found in {right_dir}")
    if not left_parquets:
        errors.append(f"no parquets found in {left_dir / '_parquets'}")
    if not right_parquets:
        errors.append(f"no parquets found in {right_dir / '_parquets'}")
    if not left_prov:
        errors.append(f"no provenance sidecars found in {left_dir / '_splice_provenance'}")
    if not right_prov:
        errors.append(f"no provenance sidecars found in {right_dir / '_splice_provenance'}")
    if set(left_features) != set(right_features):
        errors.append(
            f"feature shard stem mismatch: {sorted(left_features)} != {sorted(right_features)}"
        )
    if set(left_parquets) != set(right_parquets):
        errors.append(f"parquet stem mismatch: {sorted(left_parquets)} != {sorted(right_parquets)}")
    if set(left_prov) != set(right_prov):
        errors.append(f"provenance stem mismatch: {sorted(left_prov)} != {sorted(right_prov)}")
    _fail(errors)
    for stem in sorted(left_features):
        assert_feature_shards_equal(left_features[stem], right_features[stem])
    for stem in sorted(left_parquets):
        assert_parquets_equal(left_parquets[stem], right_parquets[stem])
    for stem in sorted(left_prov):
        assert_provenance_equal(left_prov[stem], right_prov[stem])
