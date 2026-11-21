"""Compact token-boundary format, publisher, and commit-transaction contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest
import torch as t
import zstandard
from omegaconf import OmegaConf
from safetensors.torch import save as safetensors_save

from scripts.preprocess import publish_token_boundary_source as publisher
from steering.feature_io import glob_features, load_features
from steering.overlay_io import compute_base_shard_content_hash
from steering.remote_feature_inventory import (
    RemoteShardFamily,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
)
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import NormalizedRemotePrefix
from steering.token_boundary_source import (
    TOKEN_BOUNDARY_SOURCE_KEYS,
    MissingTokenBoundarySourceError,
    TokenBoundarySourceError,
    TokenBoundarySourceManifest,
    assert_token_boundary_source_tensor_equal,
    compact_token_boundary_payload_from_raw,
    decode_token_boundary_source_bytes,
    derive_token_boundary_source_prefix,
    parse_token_boundary_source_manifest,
    save_token_boundary_source_shard,
    serialize_token_boundary_source_manifest,
    token_boundary_source_manifest_path,
    token_boundary_source_shard_path,
    validate_token_boundary_source_payload,
)
from tests.scripts.test_token_boundaries_manifest_v2 import _make_env


def _payload(*, n_segments: int = 2, n_docs: int = 2) -> dict[str, t.Tensor]:
    if n_segments == 0:
        doc_ids = t.empty(0, dtype=t.int32)
        segment_indices = t.empty(0, dtype=t.int32)
        starts = t.empty(0, dtype=t.int32)
        ends = t.empty(0, dtype=t.int32)
        offsets = t.tensor([0], dtype=t.int64)
    else:
        doc_ids = t.arange(n_segments, dtype=t.int32) % n_docs
        segment_indices = t.arange(n_segments, dtype=t.int32)
        starts = t.arange(n_segments, dtype=t.int32) * 10
        ends = starts + 10
        offsets = t.arange(n_segments + 1, dtype=t.int64) + t.arange(n_segments + 1, dtype=t.int64)
    return {
        "offsets": offsets,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "segment_char_starts": starts,
        "segment_char_ends": ends,
        "doc_token_counts": t.arange(n_docs, dtype=t.int32) + 1,
        "n_docs": t.tensor(n_docs, dtype=t.int64),
        "n_segments": t.tensor(n_segments, dtype=t.int64),
        "context_window": t.tensor(2048, dtype=t.int64),
        "segment_size": t.tensor(128, dtype=t.int64),
    }


def test_format_round_trip_is_exact_and_does_not_mutate_input(tmp_path: Path) -> None:
    payload = _payload()
    before = {key: value.clone() for key, value in payload.items()}
    first = save_token_boundary_source_shard(payload, tmp_path / "first")
    second = save_token_boundary_source_shard(payload, tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        "6d7386527a59dd6b6a56664afebd462ff7a41641c2925c3d9b68e612c25c15ae"
    )
    loaded = decode_token_boundary_source_bytes(first.read_bytes())
    assert set(loaded) == TOKEN_BOUNDARY_SOURCE_KEYS
    for key in sorted(payload):
        assert t.equal(payload[key], before[key])
        assert t.equal(loaded[key], payload[key])


def test_empty_whole_shard_is_valid_but_empty_segment_is_not() -> None:
    validate_token_boundary_source_payload(_payload(n_segments=0, n_docs=0))
    malformed = _payload()
    malformed["offsets"] = t.tensor([0, 2, 2], dtype=t.int64)
    with pytest.raises(TokenBoundarySourceError, match="strictly increasing"):
        validate_token_boundary_source_payload(malformed)


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        (lambda p: p.pop("doc_ids"), "exactly"),
        (lambda p: p.update({"unexpected": t.tensor(1)}), "exactly"),
        (lambda p: p.update({"doc_ids": p["doc_ids"].to(t.int64)}), "dtype"),
        (lambda p: p.update({"doc_ids": p["doc_ids"].reshape(1, -1)}), "rank"),
        (lambda p: p.update({"n_docs": t.tensor(-1, dtype=t.int64)}), "non-negative"),
        (lambda p: p.update({"n_segments": t.tensor(-1, dtype=t.int64)}), "non-negative"),
        (lambda p: p.update({"context_window": t.tensor(0, dtype=t.int64)}), "positive"),
        (lambda p: p.update({"segment_size": t.tensor(0, dtype=t.int64)}), "positive"),
        (lambda p: p.update({"offsets": t.empty(0, dtype=t.int64)}), "empty"),
        (lambda p: p.update({"offsets": t.tensor([1, 2, 4])}), r"offsets\[0\]"),
        (lambda p: p.update({"offsets": t.tensor([0, 3, 2])}), "non-decreasing"),
        (lambda p: p.update({"doc_ids": t.tensor([-1, 0], dtype=t.int32)}), "doc_ids"),
        (lambda p: p.update({"doc_ids": t.tensor([0, 2], dtype=t.int32)}), "doc_ids"),
        (
            lambda p: p.update({"segment_indices": t.tensor([-1, 0], dtype=t.int32)}),
            "segment_indices",
        ),
        (
            lambda p: p.update({"segment_char_starts": t.tensor([-1, 0], dtype=t.int32)}),
            "char_starts",
        ),
        (lambda p: p.update({"segment_char_ends": t.tensor([-1, 20], dtype=t.int32)}), "precede"),
        (
            lambda p: p.update({"doc_token_counts": t.tensor([-1, 2], dtype=t.int32)}),
            "token_counts",
        ),
        (lambda p: p.update({"doc_token_counts": t.tensor([1], dtype=t.int32)}), "n_docs"),
        (lambda p: p.update({"segment_indices": t.tensor([0], dtype=t.int32)}), "n_segments"),
    ],
)
def test_validator_rejects_each_malformed_condition(
    mutate: Callable[[dict[str, t.Tensor]], object],
    pattern: str,
) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(TokenBoundarySourceError, match=pattern):
        validate_token_boundary_source_payload(payload)


def test_format_helpers_reject_noncanonical_runtime_inputs() -> None:
    payload = _payload()
    payload["doc_ids"] = cast(Any, "not-a-tensor")
    with pytest.raises(TokenBoundarySourceError, match="must be a tensor"):
        validate_token_boundary_source_payload(payload)

    raw = {key: cast(Any, value) for key, value in _payload().items()}
    for key in ("n_docs", "n_segments", "context_window", "segment_size"):
        raw[key] = int(raw[key].item())
    raw.pop("offsets")
    with pytest.raises(TokenBoundarySourceError, match="lacks compact source fields"):
        compact_token_boundary_payload_from_raw(raw)

    for invalid in (True, 1.5, "1"):
        raw = {key: cast(Any, value) for key, value in _payload().items()}
        for key in ("n_docs", "n_segments", "context_window", "segment_size"):
            raw[key] = int(raw[key].item())
        raw["n_docs"] = cast(Any, invalid)
        with pytest.raises(TokenBoundarySourceError, match="strict int"):
            compact_token_boundary_payload_from_raw(raw)

    with pytest.raises(TokenBoundarySourceError, match="64 lowercase"):
        derive_token_boundary_source_prefix(
            NormalizedRemotePrefix.parse("raw/family"), "not-a-digest"
        )
    for invalid in (True, -1, 1.5):
        with pytest.raises(ValueError, match="non-negative int"):
            token_boundary_source_shard_path(
                NormalizedRemotePrefix.parse("raw/family"), cast(Any, invalid)
            )


def test_decode_and_reload_equality_fail_closed_on_every_body_surface() -> None:
    with pytest.raises(TokenBoundarySourceError, match="not valid zstd"):
        decode_token_boundary_source_bytes(b"not-zstd")

    compressor = zstandard.ZstdCompressor()
    with pytest.raises(TokenBoundarySourceError, match="safetensors body is invalid"):
        decode_token_boundary_source_bytes(compressor.compress(b"not-safetensors"))

    metadata_body = safetensors_save(_payload(), metadata={"unexpected": "value"})
    with pytest.raises(TokenBoundarySourceError, match="metadata must be empty"):
        decode_token_boundary_source_bytes(compressor.compress(metadata_body))

    expected = _payload()
    missing = dict(expected)
    missing.pop("doc_ids")
    with pytest.raises(TokenBoundarySourceError, match="key set"):
        assert_token_boundary_source_tensor_equal(expected, missing)
    changed = {key: value.clone() for key, value in expected.items()}
    changed["doc_ids"][0] += 1
    with pytest.raises(TokenBoundarySourceError, match="changed tensor doc_ids"):
        assert_token_boundary_source_tensor_equal(expected, changed)


def test_independent_hash_seeds_write_identical_bytes(tmp_path: Path) -> None:
    script = """
from pathlib import Path
import torch as t
from steering.token_boundary_source import (
    TokenBoundarySourceManifest,
    save_token_boundary_source_shard,
    serialize_token_boundary_source_manifest,
)
keys = set(['offsets','doc_ids','segment_indices','segment_char_starts','segment_char_ends','doc_token_counts','n_docs','n_segments','context_window','segment_size'])
values = {
 'offsets': t.tensor([0,2], dtype=t.int64), 'doc_ids': t.tensor([0], dtype=t.int32),
 'segment_indices': t.tensor([0], dtype=t.int32), 'segment_char_starts': t.tensor([0], dtype=t.int32),
 'segment_char_ends': t.tensor([3], dtype=t.int32), 'doc_token_counts': t.tensor([1], dtype=t.int32),
 'n_docs': t.tensor(1, dtype=t.int64), 'n_segments': t.tensor(1, dtype=t.int64),
 'context_window': t.tensor(8, dtype=t.int64), 'segment_size': t.tensor(4, dtype=t.int64),
}
save_token_boundary_source_shard({key: values[key] for key in keys}, Path(__import__('sys').argv[1]))
Path(__import__('sys').argv[2]).write_bytes(serialize_token_boundary_source_manifest(
    TokenBoundarySourceManifest('a' * 64, 'b' * 64, 1, 1, 'c' * 64)
))
"""
    outputs = []
    for seed in ("1", "987654"):
        output = tmp_path / f"seed-{seed}.safetensors.zst"
        manifest_output = tmp_path / f"seed-{seed}.manifest.json"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        subprocess.run(
            [sys.executable, "-c", script, str(output), str(manifest_output)],
            check=True,
            env=env,
        )
        outputs.append((output.read_bytes(), manifest_output.read_bytes()))
    assert outputs[0] == outputs[1]


def test_manifest_is_canonical_strict_and_namespace_is_inventory_bound() -> None:
    manifest = TokenBoundarySourceManifest(
        base_shard_content_hash="a" * 64,
        raw_inventory_sha256="b" * 64,
        schema_version=1,
        shard_count=2,
        source_inventory_sha256="c" * 64,
    )
    expected = (
        b'{"base_shard_content_hash":"'
        + b"a" * 64
        + b'","raw_inventory_sha256":"'
        + b"b" * 64
        + b'","schema_version":1,"shard_count":2,'
        + b'"source_inventory_sha256":"'
        + b"c" * 64
        + b'"}\n'
    )
    assert serialize_token_boundary_source_manifest(manifest) == expected
    assert parse_token_boundary_source_manifest(expected) == manifest
    with pytest.raises(TokenBoundarySourceError, match="canonical"):
        parse_token_boundary_source_manifest(expected.replace(b"\n", b" \n"))
    with pytest.raises(TokenBoundarySourceError, match="schema_version"):
        parse_token_boundary_source_manifest(
            expected.replace(b'"schema_version":1', b'"schema_version":true')
        )
    first = derive_token_boundary_source_prefix(
        NormalizedRemotePrefix.parse("raw/family"), "b" * 64
    )
    second = derive_token_boundary_source_prefix(
        NormalizedRemotePrefix.parse("raw/family"), "d" * 64
    )
    assert first != second
    assert str(first).endswith("/_token_boundary_source/v1/" + "b" * 64)


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (b"\xff", "not valid UTF-8 JSON"),
        (b'{"schema_version":1,"schema_version":1}\n', "not valid UTF-8 JSON"),
        (b"[]\n", "contain exactly"),
        (b"{}\n", "contain exactly"),
    ],
)
def test_manifest_parser_rejects_malformed_json_surfaces(
    payload: bytes,
    pattern: str,
) -> None:
    with pytest.raises(TokenBoundarySourceError, match=pattern):
        parse_token_boundary_source_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("schema_version", 2, "schema_version"),
        ("shard_count", True, "shard_count"),
        ("shard_count", 0, "shard_count"),
        ("base_shard_content_hash", "A" * 64, "64 lowercase"),
        ("raw_inventory_sha256", "b" * 63, "64 lowercase"),
        ("source_inventory_sha256", 1, "64 lowercase"),
    ],
)
def test_manifest_parser_rejects_invalid_typed_fields(
    field: str,
    value: object,
    pattern: str,
) -> None:
    raw: dict[str, object] = {
        "base_shard_content_hash": "a" * 64,
        "raw_inventory_sha256": "b" * 64,
        "schema_version": 1,
        "shard_count": 2,
        "source_inventory_sha256": "c" * 64,
    }
    raw[field] = value
    payload = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(TokenBoundarySourceError, match=pattern):
        parse_token_boundary_source_manifest(payload)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"expected_shard_count": 3}, "shard_count mismatch"),
        ({"expected_raw_inventory_sha256": "d" * 64}, "raw inventory digest"),
        ({"expected_source_inventory_sha256": "e" * 64}, "compact inventory digest"),
    ],
)
def test_manifest_parser_enforces_expected_bindings(
    kwargs: dict[str, object],
    pattern: str,
) -> None:
    manifest = TokenBoundarySourceManifest(
        base_shard_content_hash="a" * 64,
        raw_inventory_sha256="b" * 64,
        schema_version=1,
        shard_count=2,
        source_inventory_sha256="c" * 64,
    )
    with pytest.raises(TokenBoundarySourceError, match=pattern):
        parse_token_boundary_source_manifest(
            serialize_token_boundary_source_manifest(manifest),
            **cast(Any, kwargs),
        )


class FakeBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.downloads: list[str] = []
        self.uploads: list[str] = []
        self.fail_canonical_upload_after: int | None = None
        self._canonical_uploads = 0

    def _path(self, remote_path: str) -> Path:
        path = self.root / remote_path
        assert path.resolve().is_relative_to(self.root.resolve())
        return path

    @staticmethod
    def _info(path: Path, remote_path: str) -> RemoteObjectInfo:
        xet = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(xet.hash))

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        base = self._path(prefix)
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)
        self.downloads.append(remote_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        if "/_verified_upload_claims/" not in remote_path:
            self._canonical_uploads += 1
            if (
                self.fail_canonical_upload_after is not None
                and self._canonical_uploads >= self.fail_canonical_upload_after
            ):
                raise OSError("injected canonical upload failure")
        destination = self._path(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)
        self.uploads.append(remote_path)


def _publisher_env(tmp_path: Path, *, n_shards: int = 2):
    source, _data, _tokenizer = _make_env(tmp_path / "source", n_shards=n_shards)
    bucket = FakeBucket(tmp_path / "bucket")
    raw_root = bucket._path("raw/family")
    raw_root.mkdir(parents=True)
    for path in glob_features(source):
        shutil.copyfile(path, raw_root / path.name)
    staging = tmp_path / "staging"
    staging.mkdir()
    cfg = OmegaConf.create(
        {
            "bucket_id": "owner/test-bucket",
            "raw_prefix": "raw/family",
            "staging_dir": str(staging),
            "expected_shard_count": n_shards,
            "retry_timeout_seconds": 0.03,
            "retry_interval_seconds": 0.001,
        }
    )
    raw_paths = sorted(raw_root.glob("*.safetensors.zst"))
    raw_inventory = normalize_remote_feature_inventory(
        bucket.list("raw/family"),
        prefix=NormalizedRemotePrefix.parse("raw/family"),
        family=RemoteShardFamily.RAW_FEATURES,
    )
    compact_prefix = derive_token_boundary_source_prefix(
        NormalizedRemotePrefix.parse("raw/family"),
        remote_feature_inventory_digest(raw_inventory),
    )
    return bucket, cfg, raw_paths, compact_prefix, staging


def _publisher_config(*, raw_prefix: object, staging_dir: object):
    return OmegaConf.create(
        {
            "bucket_id": "owner/test-bucket",
            "raw_prefix": raw_prefix,
            "staging_dir": staging_dir,
            "expected_shard_count": 1,
            "retry_timeout_seconds": 0.03,
            "retry_interval_seconds": 0.001,
        }
    )


def test_publisher_config_requires_explicit_real_staging_directory(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError, match="raw_prefix is required"):
        publisher._resolve_config(_publisher_config(raw_prefix=None, staging_dir=staging))
    with pytest.raises(ValueError, match="staging_dir is required"):
        publisher._resolve_config(_publisher_config(raw_prefix="raw", staging_dir=None))
    with pytest.raises(ValueError, match="existing real nonsymlink"):
        publisher._resolve_config(
            _publisher_config(raw_prefix="raw", staging_dir=tmp_path / "missing")
        )

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("not a directory")
    staging_link = tmp_path / "staging-link"
    staging_link.symlink_to(staging, target_is_directory=True)
    for invalid in (regular_file, staging_link):
        with pytest.raises(ValueError, match="existing real nonsymlink"):
            publisher._resolve_config(_publisher_config(raw_prefix="raw", staging_dir=invalid))


def test_publisher_cleanup_is_owned_and_preserves_primary_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = tmp_path / "not-publisher-owned"
    unsafe.mkdir()
    with pytest.raises(RuntimeError, match="refusing cleanup"):
        publisher._remove_run_directory(unsafe)
    assert unsafe.is_dir()

    bucket, cfg, _raw_paths, _compact_prefix, _staging = _publisher_env(
        tmp_path / "primary", n_shards=1
    )
    bucket.fail_canonical_upload_after = 1
    original_cleanup = publisher._remove_run_directory

    def cleanup_then_fail(path: Path) -> None:
        original_cleanup(path)
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(publisher, "_remove_run_directory", cleanup_then_fail)
    with pytest.raises(RuntimeError, match="timed out") as caught:
        publisher.run(cfg, backend=bucket)
    assert any(
        "publisher staging cleanup also failed" in note
        for note in getattr(caught.value, "__notes__", [])
    )


@pytest.mark.parametrize("cleanup_mode", ["missing", "failure"])
def test_publisher_cleanup_only_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, _staging = _publisher_env(tmp_path, n_shards=1)
    original_cleanup = publisher._remove_run_directory

    def cleanup(path: Path) -> None:
        original_cleanup(path)
        if cleanup_mode == "missing":
            raise FileNotFoundError(path)
        raise OSError("injected cleanup-only failure")

    monkeypatch.setattr(publisher, "_remove_run_directory", cleanup)
    if cleanup_mode == "missing":
        assert publisher.run(cfg, backend=bucket) == 0
    else:
        with pytest.raises(OSError, match="cleanup-only"):
            publisher.run(cfg, backend=bucket)
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is not None


def test_publisher_commits_manifest_last_and_rerun_skips_raw_bodies(tmp_path: Path) -> None:
    bucket, cfg, raw_paths, compact_prefix, staging = _publisher_env(tmp_path)
    assert publisher.run(cfg, backend=bucket) == 0
    canonical_uploads = [path for path in bucket.uploads if "/_verified_upload_claims/" not in path]
    assert canonical_uploads[-1] == token_boundary_source_manifest_path(compact_prefix)
    assert len(canonical_uploads) == 3
    manifest = parse_token_boundary_source_manifest(
        bucket._path(token_boundary_source_manifest_path(compact_prefix)).read_bytes()
    )
    assert manifest.base_shard_content_hash == compute_base_shard_content_hash(raw_paths)
    assert manifest.shard_count == 2
    assert list(staging.iterdir()) == []

    bucket.downloads.clear()
    bucket.uploads.clear()
    assert publisher.run(cfg, backend=bucket) == 0
    assert bucket.uploads == []
    assert bucket.downloads == [token_boundary_source_manifest_path(compact_prefix)]
    assert list(staging.iterdir()) == []


def test_publisher_cli_wrapper_uses_default_backend_and_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, cfg, _raw_paths, _compact_prefix, _staging = _publisher_env(tmp_path, n_shards=1)
    assert publisher.run(cfg, backend=bucket) == 0
    monkeypatch.setattr(publisher, "HfBucketBackend", lambda _bucket_id: bucket)
    with pytest.raises(SystemExit) as caught:
        publisher.main.__wrapped__(cfg)
    assert caught.value.code == 0


@pytest.mark.parametrize("compact_list_to_mutate", [1, 2])
def test_publisher_rejects_compact_mutation_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact_list_to_mutate: int,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, _staging = _publisher_env(tmp_path, n_shards=1)
    original_list = bucket.list
    compact_lists = 0

    def mutate_compact(prefix: str) -> Sequence[RemoteObjectInfo]:
        nonlocal compact_lists
        if prefix == str(compact_prefix):
            compact_lists += 1
            if compact_lists == compact_list_to_mutate:
                body = bucket._path(f"{compact_prefix}/shard_00000_features.safetensors.zst")
                body.write_bytes(body.read_bytes() + b"mutated")
        return original_list(prefix)

    monkeypatch.setattr(bucket, "list", mutate_compact)
    with pytest.raises(RuntimeError, match="inventory"):
        publisher.run(cfg, backend=bucket)
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is None


def test_publisher_rejects_raw_mutation_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, _staging = _publisher_env(tmp_path, n_shards=1)
    original_list = bucket.list
    raw_lists = 0

    def mutate_raw(prefix: str) -> Sequence[RemoteObjectInfo]:
        nonlocal raw_lists
        if prefix == "raw/family":
            raw_lists += 1
            if raw_lists == 2:
                body = bucket._path("raw/family/shard_00000_features.safetensors.zst")
                body.write_bytes(body.read_bytes() + b"mutated")
        return original_list(prefix)

    monkeypatch.setattr(bucket, "list", mutate_raw)
    with pytest.raises(RuntimeError, match="inventory changed"):
        publisher.run(cfg, backend=bucket)
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is None


def test_publisher_rejects_divergent_same_path_sidecar(
    tmp_path: Path,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, _staging = _publisher_env(tmp_path, n_shards=1)
    divergent = bucket._path(f"{compact_prefix}/shard_00000_features.safetensors.zst")
    divergent.parent.mkdir(parents=True)
    divergent.write_bytes(b"divergent immutable sidecar")

    with pytest.raises(RuntimeError, match="same-path remote object collision"):
        publisher.run(cfg, backend=bucket)
    assert divergent.read_bytes() == b"divergent immutable sidecar"
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is None


def test_publisher_failure_leaves_uncommitted_sidecars_and_reruns(
    tmp_path: Path,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, staging = _publisher_env(tmp_path)
    bucket.fail_canonical_upload_after = 2
    with pytest.raises(RuntimeError, match="timed out"):
        publisher.run(cfg, backend=bucket)
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is None
    assert list(staging.iterdir()) == []

    bucket.fail_canonical_upload_after = None
    bucket._canonical_uploads = 0
    assert publisher.run(cfg, backend=bucket) == 0
    assert bucket.stat(token_boundary_source_manifest_path(compact_prefix)) is not None


def test_existing_malformed_manifest_is_never_overwritten_or_followed_by_raw_download(
    tmp_path: Path,
) -> None:
    bucket, cfg, _raw_paths, compact_prefix, _staging = _publisher_env(tmp_path)
    raw_path = next(bucket._path("raw/family").glob("*.safetensors.zst"))
    compact = compact_token_boundary_payload_from_raw(load_features(raw_path))
    for index in range(2):
        save_token_boundary_source_shard(
            compact,
            bucket._path(f"{compact_prefix}/shard_{index:05d}_features.safetensors.zst"),
        )
    manifest_path = bucket._path(token_boundary_source_manifest_path(compact_prefix))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    malformed = b"{}\n"
    manifest_path.write_bytes(malformed)

    with pytest.raises(TokenBoundarySourceError, match="exactly"):
        publisher.run(cfg, backend=bucket)
    assert manifest_path.read_bytes() == malformed
    assert all("/_token_boundary_source/" in path for path in bucket.downloads)


def test_missing_source_error_type_is_dedicated() -> None:
    assert issubclass(MissingTokenBoundarySourceError, TokenBoundarySourceError)
