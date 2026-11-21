"""Certified sparse-source regressions for remote clustered inline steering."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.conditional.feature_index import (
    FeatureIndex,
    FeatureIndexCatalogError,
    SegmentFeatures,
)
from steering.conditional.splice_counterfactual import derived_feature_fingerprint
from steering.feature_io import load_features, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.remote_inline_eval import (
    REMOTE_INLINE_EVAL_SOURCE_FILENAME,
    CertifiedRemoteInlineEvalSource,
    build_remote_inline_eval_certificate,
    parse_remote_inline_eval_certificate,
    publish_remote_inline_eval_source,
)
from steering.remote_training import (
    RemoteObjectInfo,
    RemoteTrainingConfig,
    RemoteTrainingError,
    _stable_file_identity,
)
from steering.stats_io import (
    save_statistics,
    save_tokenizer_checksum,
)
from tests.steering._helpers import make_valid_shard_data
from tests.steering.test_atomic_writes import _minimal_stats

TOKENIZER_CHECKSUM = "b" * 64


class _Bucket:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        if not path.is_file():
            return None
        byte_length, _, _, xet_hash = _stable_file_identity(path)
        return RemoteObjectInfo(
            path=remote_path,
            byte_length=byte_length,
            xet_hash=xet_hash,
        )

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        del prefix
        return []

    def download(self, remote_path: str, local_path: Path) -> None:
        shutil.copyfile(self._path(remote_path), local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)


def _remote() -> RemoteTrainingConfig:
    parsed = RemoteTrainingConfig.parse(
        {
            "bucket_id": "owner/bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 2},
            "selection_prefix": "features",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 0.1,
            "checkpoint_retry_timeout_seconds": 0.1,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
            "train_packed": {},
        }
    )
    assert parsed is not None
    return parsed


def _segment_snapshot(segment: SegmentFeatures) -> tuple[Any, ...]:
    return (
        segment.segment_idx,
        segment.position,
        segment.feature_ids,
        segment.feature_mags,
        segment.char_start,
        segment.char_end,
        segment.feature_percentiles,
        segment.token_start,
        segment.token_end,
    )


def _assert_docs_equal(
    left: list[SegmentFeatures],
    right: list[SegmentFeatures],
) -> None:
    assert len(left) == len(right)
    for left_segment, right_segment in zip(left, right, strict=True):
        left_values = _segment_snapshot(left_segment)
        right_values = _segment_snapshot(right_segment)
        for left_value, right_value in zip(left_values, right_values, strict=True):
            if isinstance(left_value, t.Tensor):
                assert isinstance(right_value, t.Tensor)
                assert t.equal(left_value, right_value)
            else:
                assert left_value == right_value


def _build_producer(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    feature_dir = root / "derived"
    data_dir = root / "parquets"
    feature_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    raw_paths: list[Path] = []
    for index in range(2):
        stem = f"shard_{index:05d}"
        raw_paths.append(
            save_features(
                make_valid_shard_data(seed=index),
                feature_dir / f"{stem}_features",
            )
        )
        pq.write_table(
            pa.table({"text": [f"document {index}-{doc}" for doc in range(3)]}),
            data_dir / f"{stem}.parquet",
        )
    token_dir = overlay_dir(
        feature_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    write_manifest(
        token_dir,
        TOKENIZER_CHECKSUM,
        OverlayKind.SEG_BOUNDARY,
        base_shard_paths=raw_paths,
    )
    for index, raw_path in enumerate(raw_paths):
        stem = f"shard_{index:05d}"
        n_segments = int(load_features(raw_path)["n_segments"])
        save_overlay_shard(
            {
                "segment_token_starts": t.arange(n_segments, dtype=t.int32),
                "segment_token_ends": t.arange(1, n_segments + 1, dtype=t.int32),
            },
            overlay_shard_path(token_dir, stem),
        )
    stats_path = save_statistics(
        replace(
            _minimal_stats(),
            corpus=replace(_minimal_stats().corpus, n_shards_processed=2),
        ),
        feature_dir / "feature_statistics",
    )
    save_tokenizer_checksum(stats_path, TOKENIZER_CHECKSUM)
    certificate = publish_remote_inline_eval_source(
        feature_dir=feature_dir,
        data_dir=str(data_dir),
        stem="shard_00001",
        fingerprint_cache_dir=feature_dir,
    )
    return feature_dir, data_dir, certificate


def _stage_sparse(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    _Bucket,
    dict[str, Any],
    FeatureIndex,
]:
    producer_root = tmp_path / "producer"
    producer_dir, producer_data, certificate = _build_producer(producer_root)
    full_index = FeatureIndex(
        str(producer_dir),
        tokenizer_checksum=TOKENIZER_CHECKSUM,
    )
    local_root = tmp_path / "local"
    local_dir = local_root / "derived"
    local_data = local_root / "parquets"
    local_dir.mkdir(parents=True)
    local_data.mkdir(parents=True)
    raw_name = certificate["raw_feature"]["name"]
    token_name = certificate["token_boundary"]["name"]
    parquet_name = certificate["parquet"]["name"]
    shutil.copyfile(producer_dir / raw_name, local_dir / raw_name)
    producer_token_dir = overlay_dir(
        producer_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    local_token_dir = overlay_dir(
        local_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    local_token_dir.mkdir(parents=True)
    shutil.copyfile(
        producer_token_dir / "manifest.json",
        local_token_dir / "manifest.json",
    )
    shutil.copyfile(
        producer_token_dir / token_name,
        local_token_dir / token_name,
    )
    shutil.copyfile(
        producer_data / parquet_name,
        local_data / parquet_name,
    )
    bucket = _Bucket(tmp_path / "bucket")
    remote_certificate = bucket.root / "features" / "derived" / REMOTE_INLINE_EVAL_SOURCE_FILENAME
    remote_certificate.parent.mkdir(parents=True)
    shutil.copyfile(
        producer_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME,
        remote_certificate,
    )
    return (
        local_root,
        local_dir,
        local_data,
        bucket,
        certificate,
        full_index,
    )


def _discover(
    local_root: Path,
    local_dir: Path,
    local_data: Path,
    bucket: _Bucket,
) -> CertifiedRemoteInlineEvalSource:
    return CertifiedRemoteInlineEvalSource.discover(
        backend=bucket,
        remote=_remote(),
        feature_root=local_root,
        feature_dir=local_dir,
        data_dir=str(local_data),
        tokenizer_checksum=TOKENIZER_CHECKSUM,
        segment_size=128,
    )


def test_publisher_validates_full_family_and_writes_self_authenticating_certificate(
    tmp_path: Path,
) -> None:
    feature_dir, _data_dir, certificate = _build_producer(tmp_path)

    assert certificate["stem"] == "shard_00001"
    assert certificate["derived_feature_fingerprint"] == derived_feature_fingerprint(
        feature_dir,
        cache_dir=feature_dir,
    )
    assert (
        parse_remote_inline_eval_certificate(
            json.loads(
                (feature_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME).read_text(encoding="utf-8")
            ),
            surface="test certificate",
        )
        == certificate
    )


def test_certified_sparse_index_is_exactly_equal_to_full_feature_index(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, certificate, full_index = _stage_sparse(tmp_path)

    source = _discover(local_root, local_dir, local_data, bucket)

    assert source.aggregate_sha256 == certificate["aggregate_sha256"]
    assert source.feature_index.shard_stems == ["shard_00001"]
    for doc_idx in range(3):
        _assert_docs_equal(
            full_index.lookup("shard_00001", doc_idx),
            source.feature_index.lookup("shard_00001", doc_idx),
        )
    source.feature_index.drop_except(())
    assert not source.feature_index._resident
    _assert_docs_equal(
        full_index.lookup("shard_00001", 0),
        source.feature_index.lookup("shard_00001", 0),
    )


@pytest.mark.parametrize("target", ["raw", "token", "parquet"])
def test_certified_sparse_source_rejects_one_byte_tamper(
    tmp_path: Path,
    target: str,
) -> None:
    local_root, local_dir, local_data, bucket, certificate, _ = _stage_sparse(tmp_path)
    paths = {
        "raw": local_dir / certificate["raw_feature"]["name"],
        "token": (
            overlay_dir(local_dir, OverlayKind.SEG_BOUNDARY, TOKENIZER_CHECKSUM)
            / certificate["token_boundary"]["name"]
        ),
        "parquet": local_data / certificate["parquet"]["name"],
    }
    path = paths[target]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        _discover(local_root, local_dir, local_data, bucket)


def test_certified_sparse_source_rejects_unapproved_raw_shard(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    save_features(
        make_valid_shard_data(seed=99),
        local_dir / "shard_99999_features",
    )

    with pytest.raises(RemoteTrainingError, match="exactly the certified shard"):
        _discover(local_root, local_dir, local_data, bucket)


def test_certified_sparse_source_rejects_unapproved_token_leaf(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    token_dir = overlay_dir(
        local_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    shutil.copyfile(
        next(token_dir.glob("*.safetensors.zst")),
        token_dir / "shard_99999.safetensors.zst",
    )

    with pytest.raises(RemoteTrainingError, match="exactly the certified leaf"):
        _discover(local_root, local_dir, local_data, bucket)


@pytest.mark.parametrize("target", ["raw", "token", "parquet"])
def test_certified_sparse_source_rejects_missing_body(
    tmp_path: Path,
    target: str,
) -> None:
    local_root, local_dir, local_data, bucket, certificate, _ = _stage_sparse(tmp_path)
    paths = {
        "raw": local_dir / certificate["raw_feature"]["name"],
        "token": (
            overlay_dir(local_dir, OverlayKind.SEG_BOUNDARY, TOKENIZER_CHECKSUM)
            / certificate["token_boundary"]["name"]
        ),
        "parquet": local_data / certificate["parquet"]["name"],
    }
    paths[target].unlink()

    with pytest.raises(RemoteTrainingError):
        _discover(local_root, local_dir, local_data, bucket)


def test_local_certificate_must_equal_authoritative_remote_bytes(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    (local_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RemoteTrainingError, match="conflicts with the authoritative"):
        _discover(local_root, local_dir, local_data, bucket)


def test_certified_sparse_source_rejects_token_manifest_semantic_drift(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    token_dir = overlay_dir(
        local_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    manifest = read_manifest(token_dir)
    assert manifest is not None
    manifest["base_shard_content_hash"] = "f" * 64
    (token_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RemoteTrainingError, match="full-family base hash"):
        _discover(local_root, local_dir, local_data, bucket)


def test_nonsemantic_token_manifest_timestamp_restamp_is_accepted(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    token_dir = overlay_dir(
        local_dir,
        OverlayKind.SEG_BOUNDARY,
        TOKENIZER_CHECKSUM,
    )
    manifest = read_manifest(token_dir)
    assert manifest is not None
    manifest["created_at"] = "2099-01-01T00:00:00Z"
    (token_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    source = _discover(local_root, local_dir, local_data, bucket)
    assert source.stem == "shard_00001"


def test_certificate_parser_rejects_unknown_field_and_aggregate_tamper(
    tmp_path: Path,
) -> None:
    _feature_dir, _data_dir, certificate = _build_producer(tmp_path)
    unknown = {**certificate, "unexpected": True}
    with pytest.raises(RemoteTrainingError, match="contain exactly"):
        parse_remote_inline_eval_certificate(unknown, surface="unknown")

    tampered = json.loads(json.dumps(certificate))
    tampered["raw_feature"]["byte_length"] += 1
    with pytest.raises(RemoteTrainingError, match="aggregate_sha256 mismatch"):
        parse_remote_inline_eval_certificate(tampered, surface="tampered")


def test_certificate_cross_binding_rejects_selection_family_mismatch(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, _certificate, _ = _stage_sparse(tmp_path)
    source = _discover(local_root, local_dir, local_data, bucket)

    with pytest.raises(RemoteTrainingError, match="derived-feature fingerprint"):
        source.assert_derived_feature_fingerprint("f" * 64)


def test_source_signature_recheck_rejects_post_authentication_replacement(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, certificate, _ = _stage_sparse(tmp_path)
    source = _discover(local_root, local_dir, local_data, bucket)
    parquet_path = local_data / certificate["parquet"]["name"]
    parquet_path.write_bytes(parquet_path.read_bytes() + b"x")

    with pytest.raises(FeatureIndexCatalogError, match="changed after parent validation"):
        source.validate_signatures()


def test_concurrent_discovery_publishes_one_stable_local_certificate(
    tmp_path: Path,
) -> None:
    local_root, local_dir, local_data, bucket, certificate, _ = _stage_sparse(tmp_path)
    local_certificate = local_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME
    local_certificate.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = list(
            pool.map(
                lambda _index: _discover(
                    local_root,
                    local_dir,
                    local_data,
                    bucket,
                ),
                range(2),
            )
        )

    assert {source.aggregate_sha256 for source in sources} == {certificate["aggregate_sha256"]}
    assert not list(local_root.glob(f".{REMOTE_INLINE_EVAL_SOURCE_FILENAME}.*"))


def test_build_certificate_rejects_bool_as_integer() -> None:
    with pytest.raises(RemoteTrainingError, match="strict int"):
        build_remote_inline_eval_certificate(
            stem="shard_00000",
            derived_feature_fingerprint_value="a" * 64,
            tokenizer_checksum=TOKENIZER_CHECKSUM,
            token_overlay_base_shard_content_hash="c" * 64,
            raw_feature={
                "name": "shard_00000_features.safetensors.zst",
                "byte_length": True,
                "content_sha256": "d" * 64,
                "file_fingerprint": "e" * 64,
                "n_documents": 1,
                "n_segments": 1,
                "segment_size": 128,
            },
            token_boundary={
                "name": "shard_00000.safetensors.zst",
                "byte_length": 1,
                "content_sha256": "f" * 64,
            },
            parquet={
                "name": "shard_00000.parquet",
                "byte_length": 1,
                "content_sha256": "0" * 64,
                "n_documents": 1,
            },
        )
