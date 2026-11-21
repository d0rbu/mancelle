"""HF-streamed raw-source contracts for packed-row precomputation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hf_xet
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file as safetensors_save_file

import scripts.preprocess.precompute_packed_rows as ppr
import steering.conditional.feature_index as feature_index_module
import steering.overlay_io as overlay_io
import steering.packed_rows_remote as packed_remote
import steering.token_boundary_remote as token_boundary_remote
from steering.conditional.encoder_decoder.sorted_doc_buffer import SortedDocBuffer
from steering.conditional.feature_index import (
    ArtifactSignature,
    FeatureIndex,
    FeatureIndexCatalog,
    FeatureIndexCatalogError,
    SegmentFeatures,
)
from steering.feature_io import (
    FeatureDirectoryMismatchError,
    MetadataKey,
    RegularFileIdentity,
    TensorKey,
    glob_features,
    read_shard_header_stable,
    save_features,
    shard_stem_from_path,
)
from steering.overlay_io import (
    OverlayKind,
    compute_base_shard_content_hash,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_rows_remote import (
    RemotePackedRowsError,
    RemotePackedRowsFeatureLookup,
    SelectedParquet,
    required_packed_rows_staging_capacity_bytes,
)
from steering.remote_feature_inventory import RemoteFeatureShard
from steering.remote_training import RemoteObjectInfo, RemoteTrainingError
from steering.token_boundary_remote import (
    PreparedTokenBoundaryShard,
    RemoteTokenBoundaryTransport,
    TokenBoundaryRemoteError,
)
from tests.scripts._packed_rows_fixtures import (
    SEGMENT_SIZE,
    SimpleTokenizer,
    _make_precompute_env,
)
from tests.steering._helpers import make_valid_shard_data


class _FakeBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.downloads: list[str] = []
        self.download_failures: dict[str, int] = {}
        self.upload_failures: dict[str, int] = {}
        self.uploads: list[str] = []
        self.list_calls = 0

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
        self.list_calls += 1
        base = self._path(prefix)
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"), reverse=True)
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        failures = self.download_failures.get(remote_path, 0)
        if failures:
            self.download_failures[remote_path] = failures - 1
            raise OSError("injected transient download failure")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)
        self.downloads.append(remote_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        # Only the offload path uploads; every other remote packed-row test
        # pins `bucket.uploads == []` to keep production read-only against the
        # sealed input family.
        self.uploads.append(remote_path)
        failures = self.upload_failures.get(remote_path, 0)
        if failures:
            self.upload_failures[remote_path] = failures - 1
            raise OSError("injected transient upload failure")
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)


class _StubTransport:
    def __init__(
        self,
        inventory: Sequence[RemoteFeatureShard],
        prepared: Sequence[PreparedTokenBoundaryShard],
    ) -> None:
        self.inventory = tuple(inventory)
        self.prepared = tuple(prepared)
        self.inventory_sha256 = "a" * 64

    def stream(self, *, start_position: int = 0, stop_position: int | None = None):
        stop = len(self.prepared) if stop_position is None else stop_position
        yield from self.prepared[start_position:stop]


class _StubFeatureIndex:
    def ensure_resident(self, _stem: str) -> None:
        return None

    def lookup(self, _stem: str, doc_idx: int):
        return [("segment", doc_idx)]


def _load_checkpoint_payload(path: Path) -> dict:
    """torch payload of a digest-prefixed checkpoint file (skip `sha256hex\\n`)."""
    return t.load(io.BytesIO(path.read_bytes()[65:]), map_location="cpu", weights_only=True)


def _checkpoint_header(path: Path) -> dict:
    return _load_checkpoint_payload(path)["header"]


def _copy_percentile_family_to_bucket(
    *,
    feature_dir: Path,
    bucket_root: Path,
    checksum: str,
) -> Path:
    local = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    remote = bucket_root / "raw" / "pct" / checksum[:16]
    remote.mkdir(parents=True, exist_ok=True)
    for path in local.iterdir():
        if path.is_file():
            shutil.copyfile(path, remote / path.name)
    return remote


def _canonical_env(tmp_path: Path, *, n_shards: int = 3) -> tuple[dict, Path]:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    env = _make_precompute_env(fixture_root, n_shards=n_shards)
    original_feature_dir = Path(env["feature_dir"])
    canonical = (
        tmp_path
        / "features"
        / f"s{SEGMENT_SIZE}"
        / "l8"
        / "a_res"
        / "p_max_pool"
        / "f16k"
        / "m_gemma-2-2b"
        / "base"
    )
    canonical.parent.mkdir(parents=True)
    shutil.move(original_feature_dir, canonical)
    env["feature_dir"] = str(canonical)

    raw_paths = glob_features(canonical)
    base_hash = compute_base_shard_content_hash(raw_paths)
    pct_dir = overlay_dir(canonical, OverlayKind.PERCENTILE, env["pct_checksum"])
    tok_dir = overlay_dir(canonical, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    write_manifest(
        pct_dir,
        env["pct_checksum"],
        OverlayKind.PERCENTILE,
        base_shard_paths=raw_paths,
        precomputed_base_hash=base_hash,
    )
    write_manifest(
        tok_dir,
        env["tok_checksum"],
        OverlayKind.SEG_BOUNDARY,
        base_shard_paths=raw_paths,
        precomputed_base_hash=base_hash,
    )
    safetensors_save_file(
        {"stub": t.tensor([1], dtype=t.int32)},
        str(canonical / "feature_statistics.safetensors"),
        metadata={"percentile_checksum": env["pct_checksum"]},
    )

    bucket_root = tmp_path / "bucket"
    remote_prefix = bucket_root / "raw"
    remote_prefix.mkdir(parents=True)
    for path in raw_paths:
        shutil.copyfile(path, remote_prefix / path.name)
    _copy_percentile_family_to_bucket(
        feature_dir=canonical,
        bucket_root=bucket_root,
        checksum=env["pct_checksum"],
    )
    return env, bucket_root


def _cfg(env: dict, *, staging: Path, enabled: bool) -> DictConfig:
    cfg = OmegaConf.load(ppr._PACKED_ROWS_CONFIG_PATH)
    cfg.feature_dir = env["feature_dir"]
    cfg.data_dir = env["data_dir"]
    cfg.segment_size = SEGMENT_SIZE
    cfg.sae_width = "16k"
    cfg.context_len = 32
    cfg.buffer_size = 4
    cfg.rows_per_shard = 3
    cfg.max_shards = None
    cfg.compress_workers = 1
    cfg.profile = False
    cfg.remote.enabled = enabled
    cfg.remote.bucket_id = "owner/test-bucket"
    cfg.remote.local_features_root = env["feature_dir"]
    cfg.remote.remote_features_root = "raw"
    cfg.remote.staging_dir = str(staging)
    cfg.remote.expected_shard_count = env["n_shards"]
    assert isinstance(cfg, DictConfig)
    return cfg


def _product_files(feature_dir: Path) -> dict[str, tuple[str, bytes | str]]:
    result: dict[str, tuple[str, bytes | str]] = {}
    for path in sorted(feature_dir.rglob("*")):
        relative = path.relative_to(feature_dir).as_posix()
        if path.name.startswith("."):
            continue
        if "/c32/" not in f"/{relative}/":
            continue
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
    return result


def _patch_remote_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict,
    bucket: _FakeBucket,
) -> None:
    monkeypatch.setattr(ppr, "get_tokenizer", lambda: SimpleTokenizer())
    monkeypatch.setattr(
        ppr,
        "compute_tokenizer_checksum",
        lambda _tokenizer: env["tok_checksum"],
    )
    monkeypatch.setattr(
        overlay_io,
        "time",
        type(
            "_FrozenTime",
            (),
            {
                "strftime": staticmethod(lambda *_args, **_kwargs: "2026-08-12T00:00:00Z"),
                "gmtime": staticmethod(lambda: object()),
            },
        ),
    )
    original_snapshot = RemoteTokenBoundaryTransport.snapshot

    def _snapshot(cls, config, **kwargs):
        # setdefault, not override: callers that already hold a backend (the
        # offload inventory pass) pass their own.
        kwargs.setdefault("backend", bucket)
        return original_snapshot(config, **kwargs)

    monkeypatch.setattr(
        ppr._RemotePackedRowsTransport,
        "snapshot",
        classmethod(_snapshot),
    )


# Shared prelude for `subprocess.Popen([sys.executable, "-c", bootstrap, ...])`
# child scripts below: it loads argv[1]/argv[2] into `env`/`bucket` and
# in-process-patches `ppr`/`overlay_io` the same way `_patch_remote_runtime`
# does for the parent process. Each bootstrap concatenates its own gating
# logic (plus any additional stdlib imports it needs) after this constant.
_BOOTSTRAP_PRELUDE = """
import json
import sys
from pathlib import Path

import scripts.preprocess.precompute_packed_rows as ppr
import steering.overlay_io as overlay_io
from steering.token_boundary_remote import RemoteTokenBoundaryTransport
from tests.scripts._packed_rows_fixtures import SimpleTokenizer
from tests.scripts.test_packed_rows_remote import _FakeBucket, _cfg

env = json.loads(sys.argv[1])
bucket = _FakeBucket(Path(sys.argv[2]))
ppr.get_tokenizer = lambda: SimpleTokenizer()
ppr.compute_tokenizer_checksum = lambda _tokenizer: env["tok_checksum"]
overlay_io.time = type(
    "_FrozenTime",
    (),
    {
        "strftime": staticmethod(lambda *_args, **_kwargs: "2026-08-12T00:00:00Z"),
        "gmtime": staticmethod(lambda: object()),
    },
)
original_snapshot = RemoteTokenBoundaryTransport.snapshot

def snapshot(cls, config, **kwargs):
    kwargs.setdefault("backend", bucket)
    return original_snapshot(config, **kwargs)

ppr._RemotePackedRowsTransport.snapshot = classmethod(snapshot)
"""


def _remote_output_dir(feature_dir: Path, env: dict, max_shards: int | None) -> Path:
    suffix = "sall" if max_shards is None else f"s{max_shards}"
    return (
        feature_dir
        / "pct"
        / env["pct_checksum"][:16]
        / "tok"
        / env["tok_checksum"][:16]
        / "c32"
        / suffix
    )


def _seeded_remote_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_shards: int = 3,
    compress_workers: int = 1,
    max_shards: int | None = None,
    local_dirname: str = "local",
    unlink_local_raw: bool = False,
    unlink_remote_raw: bool = True,
) -> tuple[dict, dict, dict, Path, _FakeBucket, Path, Path]:
    """Fork a canonical feature tree into `local_dirname`/ and `remote`/
    copies, patch remote runtime globals, and run the local (`enabled=False`)
    config as the byte-identical reference product tree.

    Returns `(env, local_env, remote_env, bucket_root, bucket,
    local_feature_dir, remote_feature_dir)`.
    """
    env, bucket_root = _canonical_env(tmp_path, n_shards=n_shards)
    seed_feature_dir = Path(env["feature_dir"])
    relative_feature_dir = seed_feature_dir.relative_to(tmp_path / "features")
    local_feature_dir = tmp_path / local_dirname / relative_feature_dir
    remote_feature_dir = tmp_path / "remote" / relative_feature_dir
    local_feature_dir.parent.mkdir(parents=True)
    remote_feature_dir.parent.mkdir(parents=True)
    shutil.copytree(seed_feature_dir, local_feature_dir)
    shutil.copytree(seed_feature_dir, remote_feature_dir)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)

    local_env = {**env, "feature_dir": str(local_feature_dir)}
    local_cfg = _cfg(local_env, staging=tmp_path / "unused", enabled=False)
    local_cfg.compress_workers = compress_workers
    local_cfg.max_shards = max_shards
    assert ppr.run(local_cfg) == 0
    if unlink_local_raw:
        for raw in glob_features(local_feature_dir):
            raw.unlink()
    if unlink_remote_raw:
        for raw in glob_features(remote_feature_dir):
            raw.unlink()

    remote_env = {**env, "feature_dir": str(remote_feature_dir)}
    return env, local_env, remote_env, bucket_root, bucket, local_feature_dir, remote_feature_dir


def _buffer_pop_sequence(buffer: SortedDocBuffer) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    while buffer:
        entry = buffer.pop_best_fit(10_000)
        assert entry is not None and len(entry) == 3
        result.append(cast(tuple[int, int], entry[2]))
    return result


def test_sorted_doc_buffer_snapshot_rebuild_preserves_randomized_tie_order() -> None:
    rng = random.Random(1437)
    for _case in range(100):
        entries = [([index] * rng.choice((2, 2, 3, 3, 5)), None, (0, index)) for index in range(40)]
        original = SortedDocBuffer()
        for entry in entries:
            original.append(entry)
        rebuilt = SortedDocBuffer()
        for entry in original.snapshot():
            rebuilt.append(entry)
        control = SortedDocBuffer()
        for entry in entries:
            control.append(entry)
        assert _buffer_pop_sequence(rebuilt) == _buffer_pop_sequence(control)


def test_sorted_doc_buffer_deliberate_resort_changes_duplicate_pop_order() -> None:
    entries = [([index, index], None, (0, index)) for index in range(6)]
    original = SortedDocBuffer()
    for entry in entries:
        original.append(entry)
    snapshot = original.snapshot()
    faithful = SortedDocBuffer()
    deliberately_resorted = SortedDocBuffer()
    for entry in snapshot:
        faithful.append(entry)
    for entry in reversed(snapshot):
        deliberately_resorted.append(entry)
    assert _buffer_pop_sequence(faithful) != _buffer_pop_sequence(deliberately_resorted)


def test_checkpoint_buffer_round_trip_serializes_production_segment_fields() -> None:
    first = ppr._PackedSegFeatures(
        feature_ids=t.tensor([1, 65535], dtype=t.int64),
        percentiles=t.tensor([7, 255], dtype=t.uint8),
        char_start=3,
        char_end=19,
        token_start=1,
        token_end=4,
    )
    second = ppr._PackedSegFeatures(
        feature_ids=t.tensor([4], dtype=t.int64),
        percentiles=t.tensor([11], dtype=t.uint8),
        char_start=20,
        char_end=25,
        token_start=None,
        token_end=None,
    )
    buffer = SortedDocBuffer()
    buffer.append(([10, 11, 12], [first, second], (5, 8)))
    restored = ppr._deserialize_checkpoint_buffer(ppr._serialize_checkpoint_buffer(buffer))
    assert len(restored) == 1
    tokens, segments, provenance = restored[0]
    assert tokens == [10, 11, 12]
    assert provenance == (5, 8)
    assert len(segments) == 2
    for expected, actual in zip((first, second), segments, strict=True):
        assert t.equal(actual.feature_ids, expected.feature_ids)
        assert t.equal(actual.percentiles, expected.percentiles)
        assert actual.char_start == expected.char_start
        assert actual.char_end == expected.char_end
        assert actual.token_start == expected.token_start
        assert actual.token_end == expected.token_end


@pytest.mark.parametrize("flip_at_fraction", [0.1, 0.5, 0.9])
def test_checkpoint_any_flipped_byte_is_rejected_to_replay(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    flip_at_fraction: float,
) -> None:
    """The content digest is the integrity contract: torch.load alone accepts a
    flipped payload byte silently, so corruption anywhere in the file must fail
    the digest and fall back to replay."""
    parquet = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": ["alpha", "beta"]}), parquet)
    context = ppr._RemotePackingCheckpointContext(
        checkpoint_dir=tmp_path / "ckpt",
        identity_digest="d" * 64,
        split="train",
        parquet_inventory=((parquet, 0),),
        end_position=1,
        rows_per_shard=4,
    )
    context.checkpoint_dir.mkdir()
    segment = ppr._PackedSegFeatures(
        feature_ids=t.tensor([1, 2], dtype=t.int64),
        percentiles=t.tensor([3, 4], dtype=t.uint8),
        char_start=0,
        char_end=3,
        token_start=0,
        token_end=2,
    )
    buffer = SortedDocBuffer()
    buffer.append(([10, 11], [segment], (0, 0)))
    ppr._write_packing_checkpoint(
        context,
        shard_idx=1,
        total_rows=4,
        documents_loaded=1,
        documents_consumed=0,
        cursor=ppr._DocumentCursor(0, 0, 1),
        doc_buffer=buffer,
        compress_workers=1,
    )
    path = ppr._checkpoint_path(context.checkpoint_dir, 1)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is not None

    raw = bytearray(path.read_bytes())
    position = 65 + int((len(raw) - 65) * flip_at_fraction)
    raw[position] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None
    assert any("digest mismatch" in record.message for record in caplog.records)


def test_checkpoint_buffer_write_rejects_negative_present_token_boundary() -> None:
    segment = ppr._PackedSegFeatures(
        feature_ids=t.tensor([1], dtype=t.int64),
        percentiles=t.tensor([3], dtype=t.uint8),
        char_start=0,
        char_end=3,
        token_start=-1,
        token_end=2,
    )
    buffer = SortedDocBuffer()
    buffer.append(([10], [segment], (0, 0)))
    with pytest.raises(ppr.PackedRowInvariantError, match="token boundaries"):
        ppr._serialize_checkpoint_buffer(buffer)


def test_raw_document_iterator_cursor_skips_without_tokenize_or_lookup(tmp_path: Path) -> None:
    parquet = tmp_path / "shard_00000.parquet"
    pq.write_table(
        pa.table({"text": ["zero", "one", "two", "three"]}),
        parquet,
        row_group_size=2,
    )
    encoded: list[list[str]] = []
    looked_up: list[tuple[str, int]] = []

    class _Tokenizer:
        def encode(self, texts, *, prepend):
            encoded.append(list(texts))
            return [[prepend, len(text)] for text in texts]

    class _Lookup:
        def lookup(self, shard_stem, doc_idx):
            looked_up.append((shard_stem, doc_idx))
            return [
                SegmentFeatures(
                    segment_idx=0,
                    position=0,
                    feature_ids=t.tensor([doc_idx], dtype=t.int64),
                    feature_mags=t.ones(1),
                    char_start=0,
                    char_end=1,
                    feature_percentiles=t.tensor([doc_idx], dtype=t.uint8),
                    token_start=0,
                    token_end=2,
                )
            ]

    cursor = ppr._DocumentCursor(
        parquet_file_idx=0,
        row_group_idx=1,
        rows_consumed_in_group=1,
    )
    yielded = list(
        ppr._raw_document_iterator(
            [str(parquet)],
            _Tokenizer(),
            99,
            _Lookup(),
            cursor=cursor,
        )
    )
    assert [item[3] for item in yielded] == [3]
    assert encoded == [["three"]]
    assert looked_up == [("shard_00000", 3)]
    assert cursor == ppr._DocumentCursor(parquet_file_idx=1)


def _empty_checkpoint_context(tmp_path: Path) -> ppr._RemotePackingCheckpointContext:
    return ppr._RemotePackingCheckpointContext(
        checkpoint_dir=ppr._remote_checkpoint_dir(tmp_path / "packed"),
        identity_digest="a" * 64,
        split="train",
        parquet_inventory=(),
        end_position=0,
        rows_per_shard=3,
    )


def test_checkpoint_size_cap_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversize.ckpt"
    monkeypatch.setattr(ppr, "_REMOTE_PACKING_CHECKPOINT_MAX_BYTES", 0)
    with pytest.raises(ppr.PackedRowInvariantError, match="exceeds 4 GiB"):
        ppr._atomic_save_packing_checkpoint(
            {"header": {}, "buffer": {"tensor": t.ones(1)}},
            path,
        )
    assert not path.exists()


def test_checkpoint_retention_keeps_compression_window_plus_two(tmp_path: Path) -> None:
    context = _empty_checkpoint_context(tmp_path)
    for shard_idx in range(1, 9):
        ppr._write_packing_checkpoint(
            context,
            shard_idx=shard_idx,
            total_rows=shard_idx * context.rows_per_shard,
            documents_loaded=0,
            documents_consumed=0,
            cursor=ppr._DocumentCursor(),
            doc_buffer=SortedDocBuffer(),
            compress_workers=4,
        )
    assert [path.name for path in sorted(context.checkpoint_dir.glob("*.ckpt"))] == [
        f"shard_{index:05d}.ckpt" for index in range(2, 9)
    ]


def test_checkpoint_identity_drift_and_partial_tmp_are_ineligible(tmp_path: Path) -> None:
    context = _empty_checkpoint_context(tmp_path)
    ppr._write_packing_checkpoint(
        context,
        shard_idx=1,
        total_rows=3,
        documents_loaded=0,
        documents_consumed=0,
        cursor=ppr._DocumentCursor(),
        doc_buffer=SortedDocBuffer(),
        compress_workers=1,
    )
    path = ppr._checkpoint_path(context.checkpoint_dir, 1)
    payload = _load_checkpoint_payload(path)
    payload["header"]["identity_digest"] = "b" * 64
    ppr._atomic_save_packing_checkpoint(payload, path)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None

    path.unlink()
    partial = path.with_name(f"{path.name}.tmp.{os.getpid()}.deadbeef")
    partial.write_bytes(b"truncated")
    assert list(ppr._non_debris_entries(context.checkpoint_dir)) == []
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None
    assert ppr._sweep_atomic_write_debris(tmp_path / "packed") == 1
    assert not partial.exists()


def test_nonterminal_checkpoint_rejects_total_rows_mutation_alone(tmp_path: Path) -> None:
    parquet = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": ["zero", "one"]}), parquet, row_group_size=2)
    context = ppr._RemotePackingCheckpointContext(
        checkpoint_dir=ppr._remote_checkpoint_dir(tmp_path / "packed"),
        identity_digest="a" * 64,
        split="train",
        parquet_inventory=((parquet, 0),),
        end_position=1,
        rows_per_shard=3,
    )
    ppr._write_packing_checkpoint(
        context,
        shard_idx=1,
        total_rows=3,
        documents_loaded=1,
        documents_consumed=1,
        cursor=ppr._DocumentCursor(rows_consumed_in_group=1),
        doc_buffer=SortedDocBuffer(),
        compress_workers=1,
    )
    path = ppr._checkpoint_path(context.checkpoint_dir, 1)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is not None
    payload = _load_checkpoint_payload(path)
    payload["header"]["total_rows"] = 2
    ppr._atomic_save_packing_checkpoint(payload, path)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None


def test_checkpoint_cursor_is_bound_to_absolute_document_position(tmp_path: Path) -> None:
    parquet = tmp_path / "shard_00000.parquet"
    rows = [f"row-{index}" for index in range(6)]
    pq.write_table(pa.table({"text": rows}), parquet, row_group_size=3)
    context = ppr._RemotePackingCheckpointContext(
        checkpoint_dir=ppr._remote_checkpoint_dir(tmp_path / "packed"),
        identity_digest="a" * 64,
        split="train",
        parquet_inventory=((parquet, 0),),
        end_position=1,
        rows_per_shard=3,
    )
    ppr._write_packing_checkpoint(
        context,
        shard_idx=1,
        total_rows=3,
        documents_loaded=4,
        documents_consumed=4,
        cursor=ppr._DocumentCursor(row_group_idx=1, rows_consumed_in_group=1),
        doc_buffer=SortedDocBuffer(),
        compress_workers=1,
    )
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is not None
    pq.write_table(pa.table({"text": rows}), parquet, row_group_size=2)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None


@pytest.mark.parametrize("kind", ["symlink", "directory", "hardlink"])
def test_checkpoint_load_rejects_unstable_descriptor_targets(tmp_path: Path, kind: str) -> None:
    context = _empty_checkpoint_context(tmp_path)
    path = ppr._checkpoint_path(context.checkpoint_dir, 1)
    path.parent.mkdir(parents=True)
    target = tmp_path / "target.ckpt"
    ppr._atomic_save_packing_checkpoint({"header": {}, "buffer": {}}, target)
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        os.link(target, path)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None


def test_checkpoint_load_rejects_oversized_serialized_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _empty_checkpoint_context(tmp_path)
    ppr._write_packing_checkpoint(
        context,
        shard_idx=1,
        total_rows=3,
        documents_loaded=0,
        documents_consumed=0,
        cursor=ppr._DocumentCursor(),
        doc_buffer=SortedDocBuffer(),
        compress_workers=1,
    )
    path = ppr._checkpoint_path(context.checkpoint_dir, 1)
    monkeypatch.setattr(ppr, "_REMOTE_PACKING_CHECKPOINT_MAX_BYTES", path.stat().st_size - 66)
    assert ppr._load_eligible_packing_checkpoint(context, n_kept=1) is None


def _stub_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    headers: Sequence[tuple[int, int, int]] = ((1, 1, SEGMENT_SIZE),),
    selected_positions: Sequence[int] = (0,),
    prepared_count: int | None = None,
    start_position: int = 0,
) -> tuple[RemotePackedRowsFeatureLookup, _StubTransport]:
    staging = tmp_path / "stub-staging"
    pct_dir = tmp_path / "stub-pct"
    tok_dir = tmp_path / "stub-tok"
    parquet_dir = tmp_path / "stub-parquet"
    for directory in (staging, pct_dir, tok_dir, parquet_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inventory: list[RemoteFeatureShard] = []
    prepared: list[PreparedTokenBoundaryShard] = []
    percentile_inventory: list[RemoteFeatureShard] = []
    percentile_prepared: list[PreparedTokenBoundaryShard] = []
    header_by_name: dict[str, dict[str, object]] = {}
    selected: list[SelectedParquet] = []
    for index, (n_docs, n_segments, segment_size) in enumerate(headers):
        name = f"shard_{index:05d}_features.safetensors.zst"
        remote_info = RemoteObjectInfo(
            path=f"raw/{name}",
            byte_length=index + 1,
            xet_hash=f"{index + 1:064x}",
        )
        inventory.append(RemoteFeatureShard(index=index, info=remote_info))
        raw = staging / name
        raw.write_bytes(bytes([index + 1]))
        prepared.append(
            PreparedTokenBoundaryShard(
                index=index,
                logical_path=Path(remote_info.path),
                local_path=raw,
                remote_identity=remote_info,
                file_identity=RegularFileIdentity.capture(raw),
            )
        )
        header_by_name[name] = {
            "__metadata__": {
                MetadataKey.N_DOCS: str(n_docs),
                MetadataKey.N_SEGMENTS: str(n_segments),
                MetadataKey.SEGMENT_SIZE: str(segment_size),
            },
            "__tensor_keys__": set(),
        }
        stem = f"shard_{index:05d}"
        pct = pct_dir / f"{stem}.safetensors.zst"
        pct.write_bytes(b"pct")
        pct_remote_info = RemoteObjectInfo(
            path=f"raw/pct/checksum/{pct.name}",
            byte_length=index + 1,
            xet_hash=f"{index + 100:064x}",
        )
        percentile_inventory.append(RemoteFeatureShard(index=index, info=pct_remote_info))
        percentile_prepared.append(
            PreparedTokenBoundaryShard(
                index=index,
                logical_path=Path(pct_remote_info.path),
                local_path=pct,
                remote_identity=pct_remote_info,
                file_identity=RegularFileIdentity.capture(pct),
            )
        )
        if n_segments > 0:
            (tok_dir / f"{stem}.safetensors.zst").write_bytes(b"tok")
        if index in selected_positions:
            parquet = parquet_dir / f"{stem}.parquet"
            parquet.write_bytes(b"parquet")
            selected.append(
                SelectedParquet(
                    stem=stem,
                    path=parquet,
                    n_rows=n_docs,
                )
            )

    monkeypatch.setattr(
        packed_remote,
        "read_shard_header_stable",
        lambda path, **_kwargs: header_by_name[Path(path).name],
    )
    monkeypatch.setattr(
        packed_remote.FeatureIndex,
        "from_certified_sparse_source",
        staticmethod(lambda **_kwargs: _StubFeatureIndex()),
    )
    transport = _StubTransport(
        inventory,
        prepared if prepared_count is None else prepared[:prepared_count],
    )
    percentile_transport = _StubTransport(
        percentile_inventory,
        percentile_prepared if prepared_count is None else percentile_prepared[:prepared_count],
    )
    lookup = RemotePackedRowsFeatureLookup(
        transport=cast(Any, transport),
        percentile_transport=cast(Any, percentile_transport),
        segment_size=SEGMENT_SIZE,
        percentile_checksum="p" * 64,
        token_overlay_dir=tok_dir,
        tokenizer_checksum="t" * 64,
        selected_parquets=selected,
        start_position=start_position,
    )
    return lookup, transport


def test_disabled_remote_fields_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create({"remote": {"enabled": False, "bucket_id": "???"}})
    monkeypatch.setattr(ppr, "_run_local", lambda _cfg: 17)
    monkeypatch.setattr(
        ppr,
        "_run_remote",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("remote runner called")),
    )
    assert ppr.run(cfg) == 17


@pytest.mark.parametrize("enabled", [None, 1, "true"])
def test_remote_enabled_requires_literal_bool(enabled: object) -> None:
    cfg = OmegaConf.create({"remote": {"enabled": enabled}})
    with pytest.raises(ValueError, match="remote.enabled"):
        ppr.run(cfg)


def test_remote_enabled_absent_and_nonmapping_contract() -> None:
    assert ppr._remote_enabled(OmegaConf.create({})) is False
    with pytest.raises(ValueError, match="configuration mapping"):
        ppr._remote_enabled(OmegaConf.create({"remote": 7}))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "missing or unreadable"),
        ("{", "missing or unreadable"),
        ([], "not an object"),
        ({}, "schema_version=2"),
        ({"schema_version": 2, "kind": "tok"}, "expected kind"),
        (
            {"schema_version": 2, "kind": "pct", "checksum": "wrong"},
            "expected exact checksum",
        ),
        (
            {"schema_version": 2, "kind": "pct", "checksum": "checksum"},
            "lowercase SHA-256",
        ),
    ],
)
def test_required_manifest_rejects_malformed_inputs(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    directory = tmp_path / "overlay"
    directory.mkdir()
    if payload is not None:
        path = directory / "manifest.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    with pytest.raises(RemotePackedRowsError, match=message):
        ppr._read_required_manifest_path(
            directory / "manifest.json",
            kind=OverlayKind.PERCENTILE,
            checksum="checksum",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "overlay directory is missing"),
        ("directory", "not a regular file"),
        ("dangling", "not a regular file"),
        ("noncanonical", "noncanonical direct leaf"),
    ],
)
def test_overlay_leaf_stems_rejects_unsafe_entries(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    directory = tmp_path / "overlay"
    if case != "missing":
        directory.mkdir()
    leaf = directory / "shard_00000.safetensors.zst"
    if case == "directory":
        leaf.mkdir()
    elif case == "dangling":
        leaf.symlink_to(directory / "missing-target")
    elif case == "noncanonical":
        (directory / "bad.safetensors.zst").write_bytes(b"bad")

    with pytest.raises(RemotePackedRowsError, match=message):
        ppr._overlay_leaf_stems(directory)


def test_overlay_leaf_stems_accepts_symlinked_regular_files(tmp_path: Path) -> None:
    """Local coverage follows valid links while still rejecting dangling ones."""
    directory = tmp_path / "overlay"
    directory.mkdir()
    target = tmp_path / "mount" / "shard_00000.safetensors.zst"
    target.parent.mkdir()
    target.write_bytes(b"leaf")
    (directory / "shard_00000.safetensors.zst").symlink_to(target)
    (directory / "shard_00001.safetensors.zst").write_bytes(b"leaf")

    assert ppr._overlay_leaf_stems(directory) == ["shard_00000", "shard_00001"]


def test_certified_sparse_source_loads_percentiles_and_stable_raw(tmp_path: Path) -> None:
    env, _bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    raw = glob_features(feature_dir)[0]
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, env["pct_checksum"])
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    index = FeatureIndex.from_certified_sparse_source(
        feature_dir=feature_dir,
        shard_path=raw,
        token_overlay_dir=tok_dir,
        tokenizer_checksum=env["tok_checksum"],
        segment_size=SEGMENT_SIZE,
        validated_content_fingerprint="a" * 64,
        percentile_overlay_dir=pct_dir,
        percentile_checksum=env["pct_checksum"],
        skip_feature_mags=True,
        expected_shard_identity=RegularFileIdentity.capture(raw),
    )
    stem = shard_stem_from_path(raw)
    segments = index.lookup(stem, 0)
    assert segments
    assert segments[0].feature_percentiles is not None
    assert segments[0].feature_mags.numel() == 0
    assert str(feature_dir) not in index._scan_artifact_signatures

    with pytest.raises(FeatureIndexCatalogError, match="both present or both absent"):
        FeatureIndex.from_certified_sparse_source(
            feature_dir=feature_dir,
            shard_path=raw,
            token_overlay_dir=tok_dir,
            tokenizer_checksum=env["tok_checksum"],
            segment_size=SEGMENT_SIZE,
            validated_content_fingerprint="a" * 64,
            percentile_overlay_dir=pct_dir,
        )


def test_certified_sparse_source_rejects_raw_replacement_before_materialization(
    tmp_path: Path,
) -> None:
    env, _bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    raw = glob_features(feature_dir)[0]
    index = FeatureIndex.from_certified_sparse_source(
        feature_dir=feature_dir,
        shard_path=raw,
        token_overlay_dir=overlay_dir(
            feature_dir,
            OverlayKind.SEG_BOUNDARY,
            env["tok_checksum"],
        ),
        tokenizer_checksum=env["tok_checksum"],
        segment_size=SEGMENT_SIZE,
        validated_content_fingerprint="a" * 64,
        percentile_overlay_dir=overlay_dir(
            feature_dir,
            OverlayKind.PERCENTILE,
            env["pct_checksum"],
        ),
        percentile_checksum=env["pct_checksum"],
        skip_feature_mags=True,
        expected_shard_identity=RegularFileIdentity.capture(raw),
    )
    replacement = raw.with_suffix(".replacement")
    replacement.write_bytes(raw.read_bytes())
    os.replace(replacement, raw)
    with pytest.raises(FeatureDirectoryMismatchError, match="changed before stable read"):
        index.ensure_resident(shard_stem_from_path(raw))


def _staged_percentile_index(tmp_path, *, env=None):
    """Build a FeatureIndex over a staged (out-of-tree) percentile leaf."""
    if env is None:
        env, _bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    raw = glob_features(feature_dir)[0]
    stem = shard_stem_from_path(raw)
    local_pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, env["pct_checksum"])
    staged_pct_dir = tmp_path / "staged-pct"
    staged_pct_dir.mkdir()
    staged_pct = overlay_shard_path(staged_pct_dir, stem)
    shutil.copyfile(overlay_shard_path(local_pct_dir, stem), staged_pct)
    index = FeatureIndex.from_certified_sparse_source(
        feature_dir=feature_dir,
        shard_path=raw,
        token_overlay_dir=overlay_dir(
            feature_dir,
            OverlayKind.SEG_BOUNDARY,
            env["tok_checksum"],
        ),
        tokenizer_checksum=env["tok_checksum"],
        segment_size=SEGMENT_SIZE,
        validated_content_fingerprint="a" * 64,
        percentile_overlay_dir=staged_pct_dir,
        percentile_checksum=env["pct_checksum"],
        skip_feature_mags=True,
        expected_shard_identity=RegularFileIdentity.capture(raw),
        expected_percentile_identity=RegularFileIdentity.capture(staged_pct),
    )
    return index, stem, staged_pct_dir, staged_pct


def test_certified_sparse_source_accepts_identity_bound_staged_percentile_leaf(
    tmp_path: Path,
) -> None:
    index, stem, staged_pct_dir, staged_pct = _staged_percentile_index(tmp_path)
    segments = index.lookup(stem, 0)
    assert segments[0].feature_percentiles is not None
    # The staged leaf itself is identity-bound; the mutable staging directory
    # and its absent manifest are not.
    assert str(staged_pct_dir.absolute()) not in index._scan_artifact_signatures
    assert str((staged_pct_dir / "manifest.json").absolute()) not in index._scan_artifact_signatures
    assert str(staged_pct.absolute()) in index._scan_artifact_signatures


def test_certified_sparse_source_rejects_percentile_replacement_before_materialization(
    tmp_path: Path,
) -> None:
    index, stem, _staged_pct_dir, staged_pct = _staged_percentile_index(tmp_path)
    replacement = staged_pct.with_suffix(".replacement")
    replacement.write_bytes(staged_pct.read_bytes())
    os.replace(replacement, staged_pct)
    with pytest.raises(FeatureDirectoryMismatchError, match="changed before its stable read"):
        index.ensure_resident(stem)


def test_certified_sparse_zero_segment_source_may_omit_token_leaf(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    empty_data = make_valid_shard_data(n_docs=0, segs_per_doc=0)
    empty_data["segment_size"] = SEGMENT_SIZE
    raw = save_features(empty_data, feature_dir / "shard_00000_features")
    pct_checksum = "p" * 64
    tok_checksum = "t" * 64
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, pct_checksum)
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tok_checksum)
    base_hash = compute_base_shard_content_hash([raw])
    write_manifest(
        pct_dir,
        pct_checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=[raw],
        precomputed_base_hash=base_hash,
    )
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: t.empty(0, dtype=t.uint8)},
        overlay_shard_path(pct_dir, "shard_00000"),
    )
    write_manifest(
        tok_dir,
        tok_checksum,
        OverlayKind.SEG_BOUNDARY,
        base_shard_paths=[raw],
        precomputed_base_hash=base_hash,
    )
    index = FeatureIndex.from_certified_sparse_source(
        feature_dir=feature_dir,
        shard_path=raw,
        token_overlay_dir=tok_dir,
        tokenizer_checksum=tok_checksum,
        segment_size=SEGMENT_SIZE,
        validated_content_fingerprint="a" * 64,
        percentile_overlay_dir=pct_dir,
        percentile_checksum=pct_checksum,
        token_overlay_required=False,
        skip_feature_mags=True,
        expected_shard_identity=RegularFileIdentity.capture(raw),
    )
    index.ensure_resident("shard_00000")
    with pytest.raises(KeyError, match="Document 0 not found"):
        index.lookup("shard_00000", 0)

    nonzero_root = tmp_path / "nonzero"
    nonzero_root.mkdir()
    nonzero_env, _bucket_root = _canonical_env(nonzero_root, n_shards=2)
    nonzero_feature_dir = Path(nonzero_env["feature_dir"])
    nonzero_raw = glob_features(nonzero_feature_dir)[0]
    with pytest.raises(FeatureIndexCatalogError, match="zero-segment"):
        FeatureIndex.from_certified_sparse_source(
            feature_dir=nonzero_feature_dir,
            shard_path=nonzero_raw,
            token_overlay_dir=overlay_dir(
                nonzero_feature_dir,
                OverlayKind.SEG_BOUNDARY,
                nonzero_env["tok_checksum"],
            ),
            tokenizer_checksum=nonzero_env["tok_checksum"],
            segment_size=SEGMENT_SIZE,
            validated_content_fingerprint="a" * 64,
            token_overlay_required=False,
        )


def test_certified_sparse_source_preserves_token_only_inline_contract(
    tmp_path: Path,
) -> None:
    env, _bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    raw = glob_features(feature_dir)[0]
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    index = FeatureIndex.from_certified_sparse_source(
        feature_dir=feature_dir,
        shard_path=raw,
        token_overlay_dir=tok_dir,
        tokenizer_checksum=env["tok_checksum"],
        segment_size=SEGMENT_SIZE,
        validated_content_fingerprint="a" * 64,
    )
    assert index.lookup(shard_stem_from_path(raw), 0)[0].feature_mags.numel() > 0

    with pytest.raises(TypeError, match="token_overlay_required must be a bool"):
        FeatureIndex.from_certified_sparse_source(
            feature_dir=feature_dir,
            shard_path=raw,
            token_overlay_dir=tok_dir,
            tokenizer_checksum=env["tok_checksum"],
            segment_size=SEGMENT_SIZE,
            validated_content_fingerprint="a" * 64,
            token_overlay_required=cast(Any, 1),
        )
    with pytest.raises(FeatureIndexCatalogError, match="direct child"):
        FeatureIndex.from_certified_sparse_source(
            feature_dir=tmp_path,
            shard_path=raw,
            token_overlay_dir=tok_dir,
            tokenizer_checksum=env["tok_checksum"],
            segment_size=SEGMENT_SIZE,
            validated_content_fingerprint="a" * 64,
        )


def test_certified_sparse_zero_segment_metadata_and_catalog_flag_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    token_dir = tmp_path / "tok"
    feature_dir.mkdir()
    token_dir.mkdir()
    raw = feature_dir / "shard_00000_features.safetensors.zst"
    raw.write_bytes(b"raw")
    (token_dir / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        feature_index_module,
        "read_shard_header",
        lambda _path: {"__metadata__": {MetadataKey.N_SEGMENTS: "invalid"}},
    )
    with pytest.raises(FeatureIndexCatalogError, match="invalid n_segments"):
        FeatureIndex.from_certified_sparse_source(
            feature_dir=feature_dir,
            shard_path=raw,
            token_overlay_dir=token_dir,
            tokenizer_checksum="t" * 64,
            segment_size=SEGMENT_SIZE,
            validated_content_fingerprint="a" * 64,
            token_overlay_required=False,
        )

    catalog = FeatureIndexCatalog(
        schema_version=1,
        feature_dir=str(feature_dir.absolute()),
        shard_entries=(),
        segment_size=None,
        percentile_checksum=None,
        percentile_overlay_dir=None,
        percentile_overlay_entries=(),
        tokenizer_checksum=None,
        token_overlay_dir=None,
        token_overlay_entries=(),
        signatures=(ArtifactSignature.capture(feature_dir.absolute()),),
        feature_dir_signature_required=cast(Any, 1),
    )
    with pytest.raises(FeatureIndexCatalogError, match="must be a bool"):
        catalog.validate()
    with pytest.raises(FeatureIndexCatalogError, match="percentile_overlay_dir_signature"):
        replace(
            catalog,
            feature_dir_signature_required=True,
            percentile_overlay_dir_signature_required=cast(Any, 1),
        ).validate()


def test_streaming_lookup_is_monotonic_and_does_not_drain_unselected_body(
    tmp_path: Path,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=3)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    cfg = ppr._PackedRowsRemoteTransportConfig(
        bucket_id=ppr.HfBucketId.parse("owner/test-bucket"),
        remote_prefix=ppr.NormalizedRemotePrefix.parse("raw"),
        staging_dir=tmp_path / "staging",
        expected_shard_count=3,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )
    transport = RemoteTokenBoundaryTransport.snapshot(cfg, backend=bucket)
    percentile_cfg = replace(
        cfg,
        remote_prefix=ppr.NormalizedRemotePrefix.parse(f"raw/pct/{env['pct_checksum'][:16]}"),
    )
    percentile_transport = RemoteTokenBoundaryTransport.snapshot(
        percentile_cfg,
        backend=bucket,
        family=ppr.RemoteShardFamily.OVERLAY_LEAF,
    )
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    selected = tuple(
        SelectedParquet(
            stem=Path(path).stem,
            path=Path(path).absolute(),
            n_rows=15,
        )
        for path in env["parquet_paths"][:2]
    )
    with transport, percentile_transport:
        lookup = RemotePackedRowsFeatureLookup(
            transport=transport,
            percentile_transport=percentile_transport,
            segment_size=SEGMENT_SIZE,
            percentile_checksum=env["pct_checksum"],
            token_overlay_dir=tok_dir,
            tokenizer_checksum=env["tok_checksum"],
            selected_parquets=selected,
            profile_loads=True,
        )
        first = lookup.lookup("shard_00000", 0)
        assert first is lookup.lookup("shard_00000", 0)
        lookup.lookup("shard_00001", 0)
        with pytest.raises(RemotePackedRowsError, match="move backwards"):
            lookup.lookup("shard_00000", 0)
        assert lookup.finish() is None
    assert len(bucket.downloads) == 4
    assert lookup.profile_payload()["decoded_selected_shards"] == 2
    per_body = cast(list[dict], lookup.profile_payload()["per_body"])
    assert len(per_body) == 2
    assert [body["selected"] for body in per_body] == [True, True]
    assert all(body["compressed_bytes"] > 0 for body in per_body)
    assert all(body["transport_wait_seconds"] >= 0 for body in per_body)
    assert all(body["header_seconds"] >= 0 for body in per_body)
    assert per_body[0]["activate_seconds"] > 0
    assert per_body[1]["activate_seconds"] > 0
    assert len(lookup.feature_index_loads) == 2
    assert all(record["phase"] == "resident_load" for record in lookup.feature_index_loads)


def test_staged_body_timestamp_drift_rebinds_and_release_tolerates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    cfg = ppr._PackedRowsRemoteTransportConfig(
        bucket_id=ppr.HfBucketId.parse("owner/test-bucket"),
        remote_prefix=ppr.NormalizedRemotePrefix.parse("raw"),
        staging_dir=tmp_path / "staging",
        expected_shard_count=2,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )
    transport = RemoteTokenBoundaryTransport.snapshot(cfg, backend=bucket)
    real_capture = token_boundary_remote._capture_direct_single_link
    post_publication_drifted = False

    def drift_once_after_publication(
        path: Path,
        *,
        surface: str,
    ) -> RegularFileIdentity:
        nonlocal post_publication_drifted
        if surface == "verified staged body" and not post_publication_drifted:
            before = path.stat()
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
            )
            post_publication_drifted = True
        return real_capture(path, surface=surface)

    monkeypatch.setattr(
        token_boundary_remote,
        "_capture_direct_single_link",
        drift_once_after_publication,
    )

    with transport:
        iterator = transport.stream()
        first = next(iterator)
        assert read_shard_header_stable(
            first.local_path,
            expected_identity=first.file_identity,
        )
        before = first.local_path.stat()
        os.utime(
            first.local_path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
        )
        assert [prepared.index for prepared in iterator] == [1]

    warnings = [
        record.message
        for record in caplog.records
        if "staged body identity drift tolerated" in record.message
    ]
    assert len(warnings) == 2
    assert all(str(first.local_path) in warning for warning in warnings)
    assert all("mtime_ns" in warning for warning in warnings)
    assert list((tmp_path / "staging").iterdir()) == []


def test_staged_body_stale_size_stat_tolerated_and_reads_still_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    cfg = ppr._PackedRowsRemoteTransportConfig(
        bucket_id=ppr.HfBucketId.parse("owner/test-bucket"),
        remote_prefix=ppr.NormalizedRemotePrefix.parse("raw"),
        staging_dir=tmp_path / "staging",
        expected_shard_count=2,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )
    transport = RemoteTokenBoundaryTransport.snapshot(cfg, backend=bucket)
    real_capture = token_boundary_remote._capture_direct_single_link
    stale_returned = False

    def stale_size_once(
        path: Path,
        *,
        surface: str,
    ) -> RegularFileIdentity:
        nonlocal stale_returned
        observed = real_capture(path, surface=surface)
        if surface == "verified staged body" and not stale_returned:
            stale_returned = True
            return replace(observed, size=observed.size - 5)
        return observed

    monkeypatch.setattr(
        token_boundary_remote,
        "_capture_direct_single_link",
        stale_size_once,
    )

    with transport:
        iterator = transport.stream()
        first = next(iterator)
        # The bound identity must carry the inventory-verified size, not the
        # stale stat, so descriptor-sealed reads verify against the truth.
        assert first.file_identity.size == first.local_path.stat().st_size
        assert read_shard_header_stable(
            first.local_path,
            expected_identity=first.file_identity,
        )
        assert [prepared.index for prepared in iterator] == [1]

    warnings = [
        record.message
        for record in caplog.records
        if "staged body identity drift tolerated" in record.message
    ]
    assert len(warnings) == 1
    assert "size" in warnings[0]
    assert list((tmp_path / "staging").iterdir()) == []


def test_staged_body_inode_swap_during_consumption_is_fatal(tmp_path: Path) -> None:
    _env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    cfg = ppr._PackedRowsRemoteTransportConfig(
        bucket_id=ppr.HfBucketId.parse("owner/test-bucket"),
        remote_prefix=ppr.NormalizedRemotePrefix.parse("raw"),
        staging_dir=tmp_path / "staging",
        expected_shard_count=2,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )
    transport = RemoteTokenBoundaryTransport.snapshot(cfg, backend=bucket)

    with transport:
        iterator = transport.stream()
        first = next(iterator)
        replacement = first.local_path.with_name(f"{first.local_path.name}.replacement")
        replacement.write_bytes(first.local_path.read_bytes())
        os.replace(replacement, first.local_path)
        with pytest.raises(
            TokenBoundaryRemoteError,
            match="changed while consumed",
        ):
            list(iterator)

    assert list((tmp_path / "staging").iterdir()) == []


def test_double_sigterm_persists_remote_failure_profile_and_removes_staging_run_dir(
    tmp_path: Path,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    staging = tmp_path / "staging"
    profile_path = tmp_path / "preempted-profile.json"
    ready_path = tmp_path / "download-started"
    bootstrap = (
        _BOOTSTRAP_PRELUDE
        + """
import time

ready_path = Path(sys.argv[5])
real_download = bucket.download

def slow_download(remote_path, local_path):
    real_download(remote_path, local_path)
    if remote_path.endswith("/manifest.json"):
        return
    ready_path.write_text("ready")
    time.sleep(2.0)

bucket.download = slow_download
cfg = _cfg(env, staging=Path(sys.argv[3]), enabled=True)
cfg.profile = True
cfg.profile_summary_output = sys.argv[4]
ppr.main.__wrapped__(cfg)
"""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            json.dumps(env),
            str(bucket_root),
            str(staging),
            str(profile_path),
            str(ready_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60.0
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("remote fixture subprocess did not start a download")
            time.sleep(0.01)
        assert ready_path.is_file()
        process.send_signal(signal.SIGTERM)
        time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=60.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=30.0)

    assert process.returncode == 143, output
    assert profile_path.is_file()
    profile = json.loads(profile_path.read_text())
    assert profile["remote_transport"]["status"] != "success"
    assert profile["remote_transport"]["terminal_error"]["type"] == "SystemExit"
    assert not list(staging.glob(".token-boundary-remote.*"))


def test_sigterm_during_checkpoint_load_exits_143_without_replay_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=4)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = 4
    real_write = ppr._write_packing_checkpoint

    def _interrupt_after_checkpoint(context, **kwargs):
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 1:
            raise RuntimeError("injected checkpoint setup interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint setup interruption"):
        ppr.run(cfg)
    output_dir = _remote_output_dir(feature_dir, env, None)
    assert ppr._contiguous_complete_shard_count(output_dir) == 1

    ready_path = tmp_path / "checkpoint-load-ready"
    download_path = tmp_path / "unexpected-download"
    bootstrap = (
        _BOOTSTRAP_PRELUDE
        + r"""
import threading

ready_path = Path(sys.argv[4])
download_path = Path(sys.argv[5])
real_download = bucket.download

def marked_download(remote_path, local_path):
    if not remote_path.endswith("/manifest.json"):
        download_path.write_text(remote_path)
    return real_download(remote_path, local_path)

bucket.download = marked_download

def blocking_load(*_args, **_kwargs):
    ready_path.write_text("ready")
    threading.Event().wait()

ppr.t.load = blocking_load
cfg = _cfg(env, staging=Path(sys.argv[3]), enabled=True)
cfg.compress_workers = 4
cfg.resume = True
ppr.main.__wrapped__(cfg)
"""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            json.dumps(env),
            str(bucket_root),
            str(tmp_path / "staging-b"),
            str(ready_path),
            str(download_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60.0
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("checkpoint-load subprocess did not enter torch.load")
            time.sleep(0.01)
        assert process.poll() is None
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=60.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=30.0)
    assert process.returncode == 143, output
    assert not download_path.exists()


@pytest.mark.parametrize(("compress_workers", "max_shards"), [(1, None), (4, 2)])
def test_remote_run_matches_local_product_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compress_workers: int,
    max_shards: int | None,
) -> None:
    env, _local_env, remote_env, bucket_root, bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(
            tmp_path,
            monkeypatch,
            n_shards=3,
            compress_workers=compress_workers,
            max_shards=max_shards,
            unlink_remote_raw=False,
        )
    )

    bucket_pct_manifest = bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "manifest.json"
    expected_base_hash = json.loads(bucket_pct_manifest.read_text())["base_shard_content_hash"]
    remote_pct_dir = overlay_dir(
        remote_feature_dir,
        OverlayKind.PERCENTILE,
        env["pct_checksum"],
    )
    shutil.rmtree(remote_pct_dir)

    retain_local_residue = max_shards is None
    if retain_local_residue:
        for raw in glob_features(remote_feature_dir):
            raw.write_bytes(f"ignored-local-residue:{raw.name}".encode())
    else:
        for raw in glob_features(remote_feature_dir):
            raw.unlink()
    residue_before = {raw.name: raw.read_bytes() for raw in glob_features(remote_feature_dir)}
    remote_cfg = _cfg(remote_env, staging=tmp_path / "staging", enabled=True)
    remote_cfg.compress_workers = compress_workers
    remote_cfg.max_shards = max_shards
    assert ppr.run(remote_cfg) == 0

    assert _product_files(remote_feature_dir) == _product_files(local_feature_dir)
    if retain_local_residue:
        assert {
            raw.name: raw.read_bytes() for raw in glob_features(remote_feature_dir)
        } == residue_before
    else:
        assert glob_features(remote_feature_dir) == []
    assert not list((tmp_path / "staging").rglob("*.safetensors.zst"))
    assert not (remote_feature_dir / ".base_shard_hash_cache.json").exists()
    assert bucket.uploads == []
    state_path, _lock_path = ppr._remote_state_paths(
        _remote_output_dir(remote_feature_dir, env, max_shards)
    )
    assert not state_path.exists()
    selected_downloads = env["n_shards"] if max_shards is None else max_shards
    expected_downloads = 1 + 2 * selected_downloads
    assert len(bucket.downloads) == expected_downloads
    tok_dir = overlay_dir(remote_feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    packed_manifest = json.loads(
        (_remote_output_dir(remote_feature_dir, env, max_shards) / "manifest.json").read_text()
    )
    assert packed_manifest["base_shard_content_hash"] == expected_base_hash
    assert not (remote_pct_dir / "manifest.json").exists()
    assert ppr.glob_overlay_shards(remote_pct_dir) == []
    assert (
        ppr.discover_overlay(
            remote_feature_dir,
            OverlayKind.SEG_BOUNDARY,
            env["tok_checksum"],
            precomputed_base_hash=expected_base_hash,
        )
        == tok_dir
    )


@pytest.mark.parametrize("compress_workers", [1, 4])
@pytest.mark.parametrize("delete_eligible_checkpoint", [False, True])
def test_remote_interrupted_flush_seek_or_replay_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compress_workers: int,
    delete_eligible_checkpoint: bool,
) -> None:
    env, _local_env, remote_env, bucket_root, bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(
            tmp_path,
            monkeypatch,
            n_shards=4,
            compress_workers=compress_workers,
        )
    )
    real_lock = ppr._remote_packed_rows_output_lock
    lock_events: list[str] = []

    @contextmanager
    def _tracked_lock(path):
        lock_events.append("enter")
        try:
            with real_lock(path):
                yield
        finally:
            lock_events.append("exit")

    monkeypatch.setattr(ppr, "_remote_packed_rows_output_lock", _tracked_lock)
    cfg = _cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = compress_workers
    real_write = ppr._write_packing_checkpoint
    interrupted = False

    def _interrupt_after_second_train_checkpoint(context, **kwargs):
        nonlocal interrupted
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("injected post-checkpoint interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_second_train_checkpoint)
    with pytest.raises(RuntimeError, match="post-checkpoint interruption"):
        ppr.run(cfg)
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    n_kept = ppr._contiguous_complete_shard_count(output_dir)
    assert n_kept == 2
    eligible = ppr._checkpoint_path(ppr._remote_checkpoint_dir(output_dir), n_kept)
    # consumed_bodies is derived, not stored: for the canonical fixture's
    # mid-train cursor, train parquet i maps to body position i, so the
    # cursor's file index IS the resume body position.
    resume_position = _checkpoint_header(eligible)["parquet_file_idx"]
    assert type(resume_position) is int
    prior_checkpoint = ppr._checkpoint_path(ppr._remote_checkpoint_dir(output_dir), n_kept - 1)
    prior_position = _checkpoint_header(prior_checkpoint)["parquet_file_idx"]
    if delete_eligible_checkpoint:
        eligible.unlink()

    downloads_before_resume = len(bucket.downloads)
    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    profile_path = tmp_path / "resume-profile.json"
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)
    assert ppr.run(cfg) == 0
    assert _product_files(remote_feature_dir) == _product_files(local_feature_dir)
    resume_downloads = bucket.downloads[downloads_before_resume:]
    raw_resume_indices = [
        int(Path(path).name.split("_")[1])
        for path in resume_downloads
        if Path(path).name.endswith("_features.safetensors.zst")
    ]
    percentile_resume_indices = [
        int(Path(path).name.split("_")[1].split(".")[0])
        for path in resume_downloads
        if Path(path).name.startswith("shard_")
        and not Path(path).name.endswith("_features.safetensors.zst")
    ]
    profile = json.loads(profile_path.read_text())
    if delete_eligible_checkpoint:
        assert profile["resume"]["mode"]["train"] == "checkpoint_seek"
        assert profile["resume"]["rows_replayed"] == int(cfg.rows_per_shard)
        assert raw_resume_indices == list(range(prior_position, env["n_shards"]))
        assert percentile_resume_indices == raw_resume_indices
        expected_start_position = prior_position
    else:
        assert profile["resume"]["mode"]["train"] == "checkpoint_seek"
        assert profile["resume"]["rows_replayed"] == 0
        assert profile["resume"]["checkpoint_seek_seconds"] > 0
        assert raw_resume_indices == list(range(resume_position, env["n_shards"]))
        assert percentile_resume_indices == raw_resume_indices
        expected_start_position = resume_position
    assert profile["resume"]["checkpoint_write_seconds"] > 0
    assert profile["remote_transport"]["cursor_position"] == env["n_shards"]
    assert profile["remote_transport"]["bodies_processed"] == (
        env["n_shards"] - expected_start_position
    )
    assert not ppr._remote_checkpoint_dir(output_dir).exists()
    assert not ppr._remote_checkpoint_dir(output_dir.parent / "val_sall").exists()
    assert lock_events == ["enter", "exit", "enter", "exit"]


def test_row_group_layout_drift_rejects_seek_and_replay_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=4)
    for parquet_name in env["parquet_paths"]:
        parquet = Path(parquet_name)
        table = pq.read_table(parquet)
        pq.write_table(table, parquet, row_group_size=3)
    seed_feature_dir = Path(env["feature_dir"])
    relative_feature_dir = seed_feature_dir.relative_to(tmp_path / "features")
    reference_feature_dir = tmp_path / "reference" / relative_feature_dir
    remote_feature_dir = tmp_path / "remote" / relative_feature_dir
    reference_feature_dir.parent.mkdir(parents=True)
    remote_feature_dir.parent.mkdir(parents=True)
    shutil.copytree(seed_feature_dir, reference_feature_dir)
    shutil.copytree(seed_feature_dir, remote_feature_dir)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)

    reference_env = {**env, "feature_dir": str(reference_feature_dir)}
    reference_cfg = _cfg(reference_env, staging=tmp_path / "unused", enabled=False)
    reference_cfg.compress_workers = 4
    assert ppr.run(reference_cfg) == 0
    for raw in glob_features(remote_feature_dir):
        raw.unlink()
    remote_env = {**env, "feature_dir": str(remote_feature_dir)}
    cfg = _cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = 4
    real_write = ppr._write_packing_checkpoint

    def _interrupt_after_checkpoint(context, **kwargs):
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 1:
            raise RuntimeError("injected layout-drift interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="layout-drift interruption"):
        ppr.run(cfg)
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    n_kept = ppr._contiguous_complete_shard_count(output_dir)
    assert n_kept == 1

    for parquet_name in env["parquet_paths"]:
        parquet = Path(parquet_name)
        table = pq.read_table(parquet)
        pq.write_table(table, parquet, row_group_size=2)
    warnings: list[str] = []
    monkeypatch.setattr(ppr.logger, "warning", lambda message: warnings.append(str(message)))
    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "layout-resume-profile.json")
    assert ppr.run(cfg) == 0
    assert _product_files(remote_feature_dir) == _product_files(reference_feature_dir)
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"]["train"] == "replay"
    assert profile["resume"]["rows_replayed"] == n_kept * int(cfg.rows_per_shard)
    assert any(
        "absolute document position" in warning or "canonical next-row offset" in warning
        for warning in warnings
    )


def test_remote_double_interruption_skips_completed_train_on_val_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=4)
    seed_feature_dir = Path(env["feature_dir"])
    local_feature_dir = tmp_path / "local" / seed_feature_dir.relative_to(tmp_path / "features")
    remote_feature_dir = tmp_path / "remote" / seed_feature_dir.relative_to(tmp_path / "features")
    local_feature_dir.parent.mkdir(parents=True)
    remote_feature_dir.parent.mkdir(parents=True)
    shutil.copytree(seed_feature_dir, local_feature_dir)
    shutil.copytree(seed_feature_dir, remote_feature_dir)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)

    local_env = {**env, "feature_dir": str(local_feature_dir)}
    local_cfg = _cfg(local_env, staging=tmp_path / "unused", enabled=False)
    local_cfg.compress_workers = 4
    local_cfg.max_shards = 3
    assert ppr.run(local_cfg) == 0

    for raw in glob_features(remote_feature_dir):
        raw.unlink()
    remote_env = {**env, "feature_dir": str(remote_feature_dir)}
    cfg = _cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = 4
    cfg.max_shards = 3
    real_write = ppr._write_packing_checkpoint
    phase = "train"

    def _interrupt_each_split(context, **kwargs):
        nonlocal phase
        real_write(context, **kwargs)
        if phase == "train" and context.split == "train" and kwargs["shard_idx"] == 2:
            phase = "val"
            raise RuntimeError("injected train interruption")
        if phase == "val" and context.split == "val" and kwargs["shard_idx"] == 1:
            phase = "done"
            raise RuntimeError("injected val interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_each_split)
    with pytest.raises(RuntimeError, match="train interruption"):
        ppr.run(cfg)
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    with pytest.raises(RuntimeError, match="val interruption"):
        ppr.run(cfg)

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    real_precompute = ppr.precompute_packed_rows
    called_outputs: list[Path] = []

    def _record_packer_call(**kwargs):
        called_outputs.append(Path(kwargs["output_dir"]))
        return real_precompute(**kwargs)

    monkeypatch.setattr(ppr, "precompute_packed_rows", _record_packer_call)
    downloads_before_second_resume = len(bucket.downloads)
    cfg.remote.staging_dir = str(tmp_path / "staging-c")
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "second-resume-profile.json")
    assert ppr.run(cfg) == 0
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, 3)
    val_output_dir = output_dir.parent / "val_s3"
    # Both packers run; the train packer restores its terminal checkpoint and
    # exits without loading documents, so only the val body (raw + percentile
    # lockstep) is re-downloaded alongside the percentile manifest.
    assert called_outputs == [output_dir, val_output_dir]
    resume_names = [Path(path).name for path in bucket.downloads[downloads_before_second_resume:]]
    # max_shards=3 selects parquets 0-2: train = bodies 0-1, val = body 2.
    assert [
        int(name.split("_")[1])
        for name in resume_names
        if name.endswith("_features.safetensors.zst")
    ] == [2]
    assert _product_files(remote_feature_dir) == _product_files(local_feature_dir)
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"] == {
        "train": "checkpoint_seek",
        "val": "checkpoint_seek",
    }
    assert profile["resume"]["rows_replayed"] == 0


def test_resume_sweeps_orphaned_atomic_write_debris(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SIGKILL between temp-file creation and rename orphans
    `<name>.tmp.<pid>.<uuid>` debris; the run must sweep it under the output
    lock so exact product trees stay comparable and fresh runs are not
    blocked by debris-only namespaces."""
    _env, _local_env, remote_env, _bucket_root, _bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3, unlink_local_raw=True)
    )
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    val_output_dir = output_dir.parent / "val_sall"
    debris: list[Path] = []
    for packed_dir in (output_dir, val_output_dir):
        for namespace in (
            packed_dir,
            packed_dir / "_dims",
            ppr.provenance_dir(packed_dir),
            ppr._remote_checkpoint_dir(packed_dir),
        ):
            namespace.mkdir(parents=True, exist_ok=True)
            planted = namespace / f"shard_00000.json.tmp.{os.getpid()}.deadbeef"
            planted.write_bytes(b"torn write")
            debris.append(planted)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)
    state_debris = state_path.with_name(f"{state_path.name}.tmp.{os.getpid()}.deadbeef")
    state_debris.write_bytes(b"torn state write")
    debris.append(state_debris)
    unrelated = state_path.with_name(f".other-state.json.tmp.{os.getpid()}.deadbeef")
    unrelated.write_bytes(b"not owned")

    cfg = _cfg(remote_env, staging=tmp_path / "staging", enabled=True)
    assert ppr.run(cfg) == 0
    for planted in debris:
        assert not planted.exists()
    assert unrelated.read_bytes() == b"not owned"
    assert _product_files(remote_feature_dir) == _product_files(local_feature_dir)


@pytest.mark.parametrize("kind", ["directory", "fifo", "symlink"])
def test_atomic_temp_lookalikes_are_payload_and_sweep_fails_loudly(
    tmp_path: Path,
    kind: str,
) -> None:
    output_dir = tmp_path / "packed"
    output_dir.mkdir()
    lookalike = output_dir / f"shard_00000.json.tmp.{os.getpid()}.deadbeef"
    if kind == "directory":
        lookalike.mkdir()
    elif kind == "fifo":
        os.mkfifo(lookalike)
    else:
        target = tmp_path / "target"
        target.write_bytes(b"target")
        lookalike.symlink_to(target)
    assert list(ppr._non_debris_entries(output_dir)) == [lookalike]
    with pytest.raises(RemotePackedRowsError, match="lookalike"):
        ppr._sweep_atomic_write_debris(output_dir)


def test_local_packed_shard_completeness_does_not_require_dims(tmp_path: Path) -> None:
    output_dir = tmp_path / "packed"
    packed = ppr.overlay_shard_path(output_dir, "shard_00000")
    provenance = ppr.overlay_shard_path(ppr.provenance_dir(output_dir), "shard_00000")
    packed.parent.mkdir(parents=True)
    provenance.parent.mkdir(parents=True)
    packed.write_bytes(b"packed")
    provenance.write_bytes(b"provenance")
    assert ppr._packed_shard_complete(output_dir, 0)
    assert not ppr._packed_shard_complete(output_dir, 0, require_dims=True)


def test_remote_missing_dims_drops_only_torn_shard_and_rewrites_all_companions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _env,
        _local_env,
        remote_env,
        _bucket_root,
        _bucket,
        reference_feature_dir,
        remote_feature_dir,
    ) = _seeded_remote_run(
        tmp_path, monkeypatch, n_shards=4, compress_workers=4, local_dirname="reference"
    )
    cfg = _cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = 4
    real_write = ppr._write_packing_checkpoint

    def _interrupt_after_second_checkpoint(context, **kwargs):
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 2:
            raise RuntimeError("injected post-checkpoint interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_second_checkpoint)
    with pytest.raises(RuntimeError, match="post-checkpoint interruption"):
        ppr.run(cfg)
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    assert ppr._contiguous_complete_shard_count(output_dir) == 2
    shard_idx = 1
    companions = (
        ppr.overlay_shard_path(output_dir, f"shard_{shard_idx:05d}"),
        ppr.overlay_shard_path(ppr.provenance_dir(output_dir), f"shard_{shard_idx:05d}"),
        ppr._dims_companion_path(output_dir, shard_idx),
    )
    expected_bytes = [path.read_bytes() for path in companions]
    companions[-1].unlink()
    assert ppr._contiguous_complete_shard_count(output_dir) == 1
    assert ppr._packed_shard_complete(output_dir, shard_idx, require_dims=False)

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "dims-resume-profile.json")
    assert ppr.run(cfg) == 0
    assert [path.read_bytes() for path in companions] == expected_bytes
    assert _product_files(remote_feature_dir) == _product_files(reference_feature_dir)
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"]["train"] == "checkpoint_seek"


def test_remote_sigkill_after_checkpoint_resumes_byte_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _env,
        _local_env,
        remote_env,
        bucket_root,
        _bucket,
        reference_feature_dir,
        remote_feature_dir,
    ) = _seeded_remote_run(
        tmp_path, monkeypatch, n_shards=4, compress_workers=4, local_dirname="reference"
    )
    ready_path = tmp_path / "checkpoint-ready"
    staging = tmp_path / "staging-child"
    bootstrap = (
        _BOOTSTRAP_PRELUDE
        + r"""
import time

real_write = ppr._write_packing_checkpoint

def pause_after_checkpoint(context, **kwargs):
    real_write(context, **kwargs)
    if context.split == "train" and kwargs["shard_idx"] == 2:
        Path(sys.argv[4]).write_text("ready")
        while True:
            time.sleep(1)

ppr._write_packing_checkpoint = pause_after_checkpoint
cfg = _cfg(env, staging=Path(sys.argv[3]), enabled=True)
cfg.compress_workers = 4
ppr.main.__wrapped__(cfg)
"""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            json.dumps(remote_env),
            str(bucket_root),
            str(staging),
            str(ready_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    try:
        deadline = time.monotonic() + 60.0
        while (
            not ready_path.exists() or ppr._contiguous_complete_shard_count(output_dir) < 2
        ) and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("checkpoint SIGKILL subprocess did not reach a durable shard prefix")
            time.sleep(0.01)
        assert process.poll() is None
        process.kill()
        output, _ = process.communicate(timeout=60.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=30.0)
    assert process.returncode == -signal.SIGKILL, output
    assert ppr._contiguous_complete_shard_count(output_dir) == 2

    cfg = _cfg(remote_env, staging=tmp_path / "staging-resume", enabled=True)
    cfg.compress_workers = 4
    cfg.resume = True
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "sigkill-resume-profile.json")
    assert ppr.run(cfg) == 0
    assert _product_files(remote_feature_dir) == _product_files(reference_feature_dir)
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"]["train"] == "checkpoint_seek"
    assert profile["resume"]["rows_replayed"] == 0


@pytest.mark.parametrize(
    ("phase", "expected_kept", "expected_rows_replayed"),
    [
        ("packed_before_provenance", 1, 0),
        ("provenance_before_dims", 1, 0),
        ("checkpoint_temp_before_rename", 2, 3),
    ],
)
def test_remote_phase_gated_sigkill_resume_is_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_kept: int,
    expected_rows_replayed: int,
) -> None:
    (
        _env,
        _local_env,
        remote_env,
        bucket_root,
        _bucket,
        reference_feature_dir,
        remote_feature_dir,
    ) = _seeded_remote_run(
        tmp_path, monkeypatch, n_shards=4, compress_workers=4, local_dirname="reference"
    )
    ready_path = tmp_path / f"{phase}-ready"
    bootstrap = (
        _BOOTSTRAP_PRELUDE
        + r"""
import threading

import nanochat.atomic_io as atomic_io
import torch as t

staging = Path(sys.argv[3])
ready_path = Path(sys.argv[4])
phase = sys.argv[5]
required_dims_idx = 1 if phase == "checkpoint_temp_before_rename" else 0
required_dims = threading.Event()
checkpoint_published = threading.Event()
real_json_dump = ppr.atomic_json_dump

def tracked_json_dump(obj, path):
    result = real_json_dump(obj, path)
    path = Path(path)
    if path.parent.name == "_dims" and path.name == f"shard_{required_dims_idx:05d}.json":
        required_dims.set()
    return result

ppr.atomic_json_dump = tracked_json_dump
real_checkpoint_write = ppr._write_packing_checkpoint

def tracked_checkpoint_write(context, **kwargs):
    real_checkpoint_write(context, **kwargs)
    if context.split == "train" and kwargs["shard_idx"] == 2:
        checkpoint_published.set()

ppr._write_packing_checkpoint = tracked_checkpoint_write

def signal_and_block(*, need_checkpoint):
    if not required_dims.wait(20):
        raise RuntimeError("required prior dims companion was not published")
    if need_checkpoint and not checkpoint_published.wait(20):
        raise RuntimeError("required checkpoint was not published")
    ready_path.write_text("ready")
    threading.Event().wait()

if phase == "packed_before_provenance":
    real_save = ppr.save_packed_row_shard

    def gated_save(shard, path, *, zstd_level=None):
        result = real_save(shard, path, zstd_level=zstd_level)
        if Path(path).name.startswith("shard_00001"):
            signal_and_block(need_checkpoint=True)
        return result

    ppr.save_packed_row_shard = gated_save
elif phase == "provenance_before_dims":
    real_provenance = ppr._write_provenance_shard

    def gated_provenance(path, rows, **kwargs):
        result = real_provenance(path, rows, **kwargs)
        if Path(path).name.startswith("shard_00001"):
            signal_and_block(need_checkpoint=True)
        return result

    ppr._write_provenance_shard = gated_provenance
else:
    real_atomic_write_bytes = ppr.atomic_write_bytes

    def gated_atomic_write_bytes(data, path):
        path = Path(path)
        if path.name != "shard_00002.ckpt":
            return real_atomic_write_bytes(data, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = atomic_io._tmp_path_for(path)
        tmp.write_bytes(data)
        atomic_io._fsync_path(tmp)
        signal_and_block(need_checkpoint=False)

    ppr.atomic_write_bytes = gated_atomic_write_bytes

cfg = _cfg(env, staging=staging, enabled=True)
cfg.compress_workers = 4
ppr.main.__wrapped__(cfg)
"""
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            bootstrap,
            json.dumps(remote_env),
            str(bucket_root),
            str(tmp_path / "staging-child"),
            str(ready_path),
            phase,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_dir = _remote_output_dir(remote_feature_dir, remote_env, None)
    try:
        deadline = time.monotonic() + 60.0
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail(f"{phase} subprocess did not reach its gated kill point")
            time.sleep(0.01)
        assert process.poll() is None
        process.kill()
        output, _ = process.communicate(timeout=60.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=30.0)
    assert process.returncode == -signal.SIGKILL, output
    assert ppr._contiguous_complete_shard_count(output_dir) == expected_kept
    if phase == "packed_before_provenance":
        assert ppr.overlay_shard_path(output_dir, "shard_00001").is_file()
        assert not ppr.overlay_shard_path(ppr.provenance_dir(output_dir), "shard_00001").exists()
    elif phase == "provenance_before_dims":
        assert ppr._packed_shard_complete(output_dir, 1, require_dims=False)
        assert not ppr._dims_companion_path(output_dir, 1).exists()
    else:
        checkpoint_dir = ppr._remote_checkpoint_dir(output_dir)
        assert any(
            path.name.startswith("shard_00002.ckpt.tmp.") for path in checkpoint_dir.iterdir()
        )

    cfg = _cfg(remote_env, staging=tmp_path / "staging-resume", enabled=True)
    cfg.compress_workers = 4
    cfg.resume = True
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / f"{phase}-resume-profile.json")
    assert ppr.run(cfg) == 0
    assert _product_files(remote_feature_dir) == _product_files(reference_feature_dir)
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"]["train"] == "checkpoint_seek"
    assert profile["resume"]["rows_replayed"] == expected_rows_replayed


def test_remote_corrupt_eligible_checkpoint_falls_back_loudly_and_heals_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=4)
    feature_dir = Path(env["feature_dir"])
    reference_feature_dir = tmp_path / "reference" / feature_dir.relative_to(tmp_path / "features")
    reference_feature_dir.parent.mkdir(parents=True)
    shutil.copytree(feature_dir, reference_feature_dir)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    reference_env = {**env, "feature_dir": str(reference_feature_dir)}
    reference_cfg = _cfg(reference_env, staging=tmp_path / "unused", enabled=False)
    reference_cfg.compress_workers = 4
    assert ppr.run(reference_cfg) == 0
    for raw in glob_features(feature_dir):
        raw.unlink()
    cfg = _cfg(env, staging=tmp_path / "staging-a", enabled=True)
    cfg.compress_workers = 4
    real_assemble = ppr._assemble_and_save

    def _leave_gap(shard_idx, *args, **kwargs):
        if shard_idx == 1:
            raise RuntimeError("injected compression gap")
        return real_assemble(shard_idx, *args, **kwargs)

    monkeypatch.setattr(ppr, "_assemble_and_save", _leave_gap)
    with pytest.raises(RuntimeError, match="compression gap"):
        ppr.run(cfg)
    output_dir = _remote_output_dir(feature_dir, env, None)
    assert ppr._contiguous_complete_shard_count(output_dir) == 1
    assert ppr.overlay_shard_path(output_dir, "shard_00002").is_file()
    eligible = ppr._checkpoint_path(ppr._remote_checkpoint_dir(output_dir), 1)
    eligible.write_bytes(b"truncated checkpoint")

    warnings: list[str] = []
    monkeypatch.setattr(ppr.logger, "warning", lambda message: warnings.append(str(message)))
    monkeypatch.setattr(ppr, "_assemble_and_save", real_assemble)
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "gap-profile.json")
    assert ppr.run(cfg) == 0
    profile = json.loads(Path(cfg.profile_summary_output).read_text())
    assert profile["resume"]["mode"]["train"] == "replay"
    assert profile["resume"]["rows_replayed"] == int(cfg.rows_per_shard)
    assert any("checkpoint seek rejected" in message.lower() for message in warnings)
    assert ppr._packed_output_inventory(output_dir) > 3
    assert _product_files(feature_dir) == _product_files(reference_feature_dir)


def test_expected_count_mismatch_fails_before_download_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=3)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.remote.expected_shard_count = 4

    with pytest.raises(RuntimeError, match="exact contiguous family"):
        ppr.run(cfg)

    state_path, _lock_path = ppr._remote_state_paths(_remote_output_dir(feature_dir, env, None))
    assert bucket.downloads == []
    assert not os.path.lexists(state_path)


@pytest.mark.parametrize("case", ["missing", "feature", "data", "profile"])
def test_remote_staging_preflight_fails_before_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    if case == "missing":
        cfg.remote.staging_dir = None
        expected = "staging_dir is required"
    elif case == "feature":
        cfg.remote.staging_dir = env["feature_dir"]
        expected = "must be disjoint"
    elif case == "data":
        cfg.remote.staging_dir = str(Path(env["data_dir"]) / "scratch")
        expected = "must be disjoint"
    else:
        cfg.profile = True
        cfg.profile_summary_output = str(tmp_path / "staging" / "profile.json")
        expected = "must be disjoint"
    with pytest.raises(ValueError, match=expected):
        ppr.run(cfg)
    assert bucket.list_calls == 0
    assert bucket.downloads == []


def test_remote_rejects_explicit_resume_and_one_selected_parquet_before_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.resume_from_shard = 1
    with pytest.raises(RemotePackedRowsError, match="does not support resume_from_shard"):
        ppr.run(cfg)
    assert bucket.list_calls == 0

    cfg.resume_from_shard = None
    cfg.max_shards = 1
    with pytest.raises(RemotePackedRowsError, match="at least 2 selected"):
        ppr.run(cfg)
    assert bucket.list_calls == 0


def test_remote_joint_admission_is_rechecked_inside_dual_transport_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    calls = 0

    def _shrinking_disk_usage(_path):
        nonlocal calls
        calls += 1
        free = 1 << 40 if calls <= 3 else 1 << 30
        return type("_Usage", (), {"free": free})()

    monkeypatch.setattr(ppr.shutil, "disk_usage", _shrinking_disk_usage)
    with pytest.raises(RemotePackedRowsError, match="joint staging admission failed adjacent"):
        ppr.run(cfg)
    assert calls == 4
    assert bucket.downloads == [f"raw/pct/{env['pct_checksum'][:16]}/manifest.json"]


def test_staging_admission_fails_before_manifest_download_or_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An under-provisioned box must fail before ANY download or state write,
    so a plain relaunch (no resume, no manual cleanup) is the recovery."""
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    monkeypatch.setattr(ppr.shutil, "disk_usage", lambda _path: type("_Usage", (), {"free": 1})())

    with pytest.raises(RemotePackedRowsError, match="admission failed before download"):
        ppr.run(cfg)

    assert bucket.downloads == []
    state_path, _lock_path = ppr._remote_state_paths(_remote_output_dir(feature_dir, env, None))
    assert not os.path.lexists(state_path)


def test_state_atomic_write_precedes_transport_and_packed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    output_dir = _remote_output_dir(feature_dir, env, None)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)
    real_atomic_json_dump = ppr.atomic_json_dump

    def _fail_state_write(payload, path):
        if Path(path) == state_path:
            raise OSError("injected state publication failure")
        return real_atomic_json_dump(payload, path)

    monkeypatch.setattr(ppr, "atomic_json_dump", _fail_state_write)
    with pytest.raises(OSError, match="state publication failure"):
        ppr.run(cfg)
    assert bucket.list_calls == 2
    assert bucket.downloads == [f"raw/pct/{env['pct_checksum'][:16]}/manifest.json"]
    assert not os.path.lexists(state_path)
    assert not ppr._output_has_payload_or_manifest(output_dir)
    assert not ppr._output_has_payload_or_manifest(output_dir.parent / "val_sall")


def test_remote_profile_reports_selected_transport_and_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=3)
    feature_dir = Path(env["feature_dir"])
    empty = make_valid_shard_data(n_docs=0, segs_per_doc=0)
    empty["segment_size"] = SEGMENT_SIZE
    save_features(empty, bucket_root / "raw" / "shard_00002_features.safetensors.zst")
    pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, env["pct_checksum"])
    tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: t.empty(0, dtype=t.uint8)},
        overlay_shard_path(pct_dir, "shard_00002"),
    )
    overlay_shard_path(tok_dir, "shard_00002").unlink()
    base_hash = compute_base_shard_content_hash(
        sorted((bucket_root / "raw").glob("*.safetensors.zst"))
    )
    write_manifest(
        pct_dir,
        env["pct_checksum"],
        OverlayKind.PERCENTILE,
        precomputed_base_hash=base_hash,
    )
    write_manifest(
        tok_dir,
        env["tok_checksum"],
        OverlayKind.SEG_BOUNDARY,
        precomputed_base_hash=base_hash,
    )
    _copy_percentile_family_to_bucket(
        feature_dir=feature_dir,
        bucket_root=bucket_root,
        checksum=env["pct_checksum"],
    )
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    bucket.download_failures["raw/shard_00001_features.safetensors.zst"] = 1
    bucket.download_failures[f"raw/pct/{env['pct_checksum'][:16]}/shard_00001.safetensors.zst"] = 1
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    profile_path = tmp_path / "profile.json"
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.max_shards = 2
    cfg.compress_workers = 0
    cfg.remote.expected_shard_count = None
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)

    assert ppr.run(cfg) == 0
    profile = json.loads(profile_path.read_text())
    assert "preserve_percentile_uint8" not in profile["config"]
    remote = profile["remote_transport"]
    assert remote["inventory_count"] == 3
    assert remote["bodies_processed"] == 2
    assert remote["cursor_position"] == 2
    assert remote["decoded_selected_shards"] == 2
    assert remote["materialized_bytes"] == remote["downloaded_bytes"]
    assert 1 <= remote["staging_high_water_bodies"] <= 2
    assert remote["final_fence_status"] == "passed"
    assert remote["status"] == "success"
    assert remote["snapshot_list_seconds"] >= 0
    assert remote["fence_list_seconds"] >= 0
    assert remote["list_calls"] == 2
    per_body_transport = remote["per_body_transport"]
    assert [record["position"] for record in per_body_transport] == [0, 1]
    assert [record["attempts"] for record in per_body_transport] == [1, 2]
    assert remote["total_attempts"] == 3
    for record in per_body_transport:
        assert record["status"] == "ok"
        assert record["error"] is None
        assert record["materialized_bytes"] >= 0
        # Real transfers/Xet hashes on nontrivial fixture bodies are strictly
        # positive at perf_counter resolution — a stub returning 0.0 fails.
        for key in ("transfer_seconds", "download_seconds", "xet_seconds"):
            assert record[key] > 0
        assert record["sha256_seconds"] == 0.0
    percentile = remote["percentile"]
    assert percentile["inventory_count"] == 3
    assert percentile["materialized_bytes"] == percentile["downloaded_bytes"]
    assert percentile["final_fence_status"] == "passed"
    assert percentile["list_calls"] == 2
    assert [record["position"] for record in percentile["per_body_transport"]] == [0, 1]
    assert [record["attempts"] for record in percentile["per_body_transport"]] == [1, 2]
    assert percentile["total_attempts"] == 3
    assert remote["admission_required_bytes"] > percentile["admission_required_bytes"]
    assert remote["joint_admission_required_bytes"] > remote["admission_required_bytes"]
    assert remote["joint_admission_required_bytes"] > percentile["admission_required_bytes"]
    assert remote["admission_observed_free_bytes"] is not None
    assert remote["joint_admission_observed_free_bytes"] is not None
    per_body = remote["per_body"]
    assert len(per_body) == 2
    assert len(profile["feature_index_loads"]) == 2
    loads_by_stem = {record["shard_stem"]: record for record in profile["feature_index_loads"]}
    assert len(loads_by_stem) == len(profile["feature_index_loads"])
    for body in per_body:
        matching = loads_by_stem.get(body["shard_stem"])
        if body["selected"]:
            assert matching is not None
        for phase in ("header", "activate"):
            wall = body[f"{phase}_seconds"]
            cpu = body[f"{phase}_cpu_seconds"]
            assert 0 <= cpu <= wall + 0.05
    for record in profile["feature_index_loads"]:
        assert record["detail_seconds"]["doc_index"] > 0
        assert set(record["detail_cpu_seconds"]) == set(record["detail_seconds"])
        for key, cpu in record["detail_cpu_seconds"].items():
            assert 0 <= cpu <= record["detail_seconds"][key] + 0.05
    assert profile["manifest_pre_pass"] == {
        "seconds": 0.0,
        "bytes_read": 0,
        "cache_hit": False,
        "n_shards_hashed": 0,
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("segment", "segment_size mismatch"),
        ("width", "sae_width mismatch"),
        ("statistics", "No feature statistics artifact"),
        ("checksum", "No percentile checksum"),
        ("expected_bool", "expected_shard_count"),
        ("expected_zero", "expected_shard_count"),
        ("expected_string", "expected_shard_count"),
    ],
)
def test_remote_preflight_rejects_source_and_config_before_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    statistics = Path(env["feature_dir"]) / "feature_statistics.safetensors"
    if case == "segment":
        cfg.segment_size = SEGMENT_SIZE + 1
    elif case == "width":
        cfg.sae_width = "32k"
    elif case == "statistics":
        statistics.unlink()
    elif case == "checksum":
        monkeypatch.setattr(ppr, "load_percentile_checksum", lambda _path: None)
    elif case == "expected_bool":
        cfg.remote.expected_shard_count = True
    elif case == "expected_zero":
        cfg.remote.expected_shard_count = 0
    else:
        cfg.remote.expected_shard_count = "2"

    with pytest.raises((RemotePackedRowsError, ppr.SegmentSizeMismatch, ValueError), match=message):
        ppr.run(cfg)
    assert bucket.list_calls == 0
    assert bucket.downloads == []


def test_remote_missing_logical_feature_dir_returns_failure(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.load(ppr._PACKED_ROWS_CONFIG_PATH)
    cfg.feature_dir = str(tmp_path / "missing")
    cfg.remote.enabled = True
    assert isinstance(cfg, DictConfig)
    assert ppr.run(cfg) == 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("parquet_stem", "same-position prefix"),
        ("base_hash", "pin different"),
        ("percentile_missing", "exact contiguous family"),
        ("percentile_misspelled", "malformed direct overlay_leaf"),
        ("percentile_manifest", "exactly one direct manifest"),
        ("token_extra", "outside the remote family"),
    ],
)
def test_remote_post_inventory_preflight_fails_without_body_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    if case == "parquet_stem":
        real_selected = ppr._selected_parquets

        def _wrong_selected(paths):
            selected = list(real_selected(paths))
            selected[0] = replace(selected[0], stem="shard_99999")
            return tuple(selected)

        monkeypatch.setattr(ppr, "_selected_parquets", _wrong_selected)
    elif case == "base_hash":
        token_manifest = (
            overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"])
            / "manifest.json"
        )
        payload = json.loads(token_manifest.read_text())
        payload["base_shard_content_hash"] = "b" * 64
        token_manifest.write_text(json.dumps(payload))
    elif case == "percentile_missing":
        (
            bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "shard_00000.safetensors.zst"
        ).unlink()
    elif case == "percentile_misspelled":
        percentile_leaf = (
            bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "shard_00000.safetensors.zst"
        )
        percentile_leaf.rename(percentile_leaf.with_name("shard_00000_features.safetensors.zst"))
    elif case == "percentile_manifest":
        (bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "manifest.json").unlink()
    elif case == "token_extra":
        extra = overlay_shard_path(
            overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, env["tok_checksum"]),
            "shard_99999",
        )
        extra.write_bytes(b"extra")
    with pytest.raises(
        (RemotePackedRowsError, TokenBoundaryRemoteError, ValueError),
        match=message,
    ):
        ppr.run(cfg)
    expected_list_calls = 1 if case == "parquet_stem" else 2
    assert bucket.list_calls == expected_list_calls
    expected_downloads = (
        []
        if case
        in {
            "parquet_stem",
            "percentile_missing",
            "percentile_misspelled",
            "percentile_manifest",
        }
        else [f"raw/pct/{env['pct_checksum'][:16]}/manifest.json"]
    )
    assert bucket.downloads == expected_downloads


def test_remote_failure_profile_preserves_primary_error_and_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    profile_path = tmp_path / "failed-profile.json"
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)
    monkeypatch.setattr(
        ppr,
        "certify_fp_target_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected certification failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected certification failure"):
        ppr.run(cfg)
    profile = json.loads(profile_path.read_text())
    assert profile["remote_transport"]["status"] == "failed"
    assert profile["remote_transport"]["terminal_error"] == {
        "type": "RuntimeError",
        "message": "injected certification failure",
    }
    assert len(profile["remote_transport"]["per_body_transport"]) == 2
    assert profile["remote_transport"]["total_attempts"] == 2
    assert profile["remote_transport"]["list_calls"] == 2
    state_path, _lock_path = ppr._remote_state_paths(_remote_output_dir(feature_dir, env, None))
    assert os.path.lexists(state_path)
    state = json.loads(state_path.read_text())
    assert state["schema_version"] == 3
    assert len(state["remote"]["percentile_inventory_sha256"]) == 64
    manifest_info = bucket.stat(f"raw/pct/{env['pct_checksum'][:16]}/manifest.json")
    assert manifest_info is not None
    assert state["remote"]["percentile_manifest"] == {
        "path": f"raw/pct/{env['pct_checksum'][:16]}/manifest.json",
        "byte_length": len(
            (bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "manifest.json").read_bytes()
        ),
        "xet_hash": manifest_info.xet_hash,
    }
    assert state["source"]["selected_parquet_inventory"] == [
        {"stem": f"shard_{index:05d}", "n_rows": 15} for index in range(2)
    ]
    assert not any("signature" in key for key in state["source"])


def test_remote_failure_sidecar_records_the_exhausted_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failing body's transport record must reach the failure sidecar.

    Records are published from the downloader thread in the same locked update
    as the run aggregates, so an exhausted retry loop still leaves a
    status="failed" record with its attempt count (the most diagnostically
    relevant transfer of the run).
    """
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    bucket.download_failures["raw/shard_00001_features.safetensors.zst"] = 999
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    profile_path = tmp_path / "exhausted-profile.json"
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)
    cfg.remote.retry_timeout_seconds = 0.05
    cfg.remote.retry_interval_seconds = 0.001

    with pytest.raises(RemoteTrainingError, match="timed out after"):
        ppr.run(cfg)

    remote = json.loads(profile_path.read_text())["remote_transport"]
    assert remote["status"] == "failed"
    records = {record["position"]: record for record in remote["per_body_transport"]}
    assert records[0]["status"] == "ok"
    assert records[0]["error"] is None
    failed = records[1]
    assert failed["status"] == "failed"
    assert failed["error"] == "RemoteTrainingError"
    assert failed["attempts"] >= 2
    assert failed["materialized_bytes"] == 0
    assert remote["total_attempts"] == records[0]["attempts"] + failed["attempts"]


@pytest.mark.parametrize("case", ["percentile", "percentile_manifest", "token"])
def test_remote_authentication_failure_precedes_final_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    output_dir = _remote_output_dir(feature_dir, env, None)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)
    real_discover = ppr.discover_overlay
    target_kind = (
        OverlayKind.PERCENTILE
        if case in {"percentile", "percentile_manifest"}
        else OverlayKind.SEG_BOUNDARY
    )

    def _fail_one_overlay(feature_root, kind, checksum, **kwargs):
        if kind is target_kind:
            return None
        return real_discover(feature_root, kind, checksum, **kwargs)

    if case in {"percentile", "percentile_manifest"}:
        real_fence = RemoteTokenBoundaryTransport.final_inventory_fence

        def _mutate_percentile_before_fence(self):
            if case == "percentile" and "/pct/" in str(self.config.remote_prefix):
                path = bucket_root / self.inventory[0].info.path
                path.write_bytes(path.read_bytes() + b"changed")
            result = real_fence(self)
            if case == "percentile_manifest" and "/pct/" in str(self.config.remote_prefix):
                manifest_path = (
                    bucket_root / "raw" / "pct" / env["pct_checksum"][:16] / "manifest.json"
                )
                replacement = manifest_path.with_suffix(".replacement")
                replacement.write_bytes(manifest_path.read_bytes() + b" ")
                os.replace(replacement, manifest_path)
            return result

        monkeypatch.setattr(
            RemoteTokenBoundaryTransport,
            "final_inventory_fence",
            _mutate_percentile_before_fence,
        )
        expected = (
            "input inventory changed" if case == "percentile" else "manifest identity changed"
        )
    else:
        monkeypatch.setattr(ppr, "discover_overlay", _fail_one_overlay)
        expected = "token overlay could not be authenticated"

    with pytest.raises(
        (RemotePackedRowsError, TokenBoundaryRemoteError, overlay_io.OverlayStaleError),
        match=expected,
    ):
        ppr.run(cfg)
    assert os.path.lexists(state_path)
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir.parent / "val_sall" / "manifest.json").exists()


def test_remote_final_parquet_inventory_fence_precedes_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    output_dir = _remote_output_dir(feature_dir, env, None)
    real_finish = ppr.RemotePackedRowsFeatureLookup.finish

    def _finish_then_add_parquet(lookup):
        result = real_finish(lookup)
        shutil.copyfile(
            env["parquet_paths"][0],
            Path(env["data_dir"]) / "shard_99999.parquet",
        )
        return result

    monkeypatch.setattr(
        ppr.RemotePackedRowsFeatureLookup,
        "finish",
        _finish_then_add_parquet,
    )
    with pytest.raises(
        RemotePackedRowsError,
        match="selected parquet inventory changed after remote packed-row preflight",
    ):
        ppr.run(cfg)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)
    assert os.path.lexists(state_path)
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir.parent / "val_sall" / "manifest.json").exists()


def test_remote_final_staged_body_release_precedes_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    output_dir = _remote_output_dir(feature_dir, env, None)
    real_finish = ppr.RemotePackedRowsFeatureLookup.finish

    def _replace_final_body_then_finish(lookup):
        prepared = lookup._current_prepared
        assert prepared is not None
        replacement = prepared.local_path.with_name(f"{prepared.local_path.name}.replacement")
        replacement.write_bytes(prepared.local_path.read_bytes())
        os.replace(replacement, prepared.local_path)
        return real_finish(lookup)

    monkeypatch.setattr(
        ppr.RemotePackedRowsFeatureLookup,
        "finish",
        _replace_final_body_then_finish,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="changed while consumed"):
        ppr.run(cfg)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)
    assert os.path.lexists(state_path)
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir.parent / "val_sall" / "manifest.json").exists()


def test_remote_changed_state_blocks_completion_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    monkeypatch.setattr(ppr, "_read_remote_state", lambda _path: {"changed": True})
    with pytest.raises(RemotePackedRowsError, match="state changed before completion removal"):
        ppr.run(cfg)
    state_path, _lock_path = ppr._remote_state_paths(_remote_output_dir(feature_dir, env, None))
    assert os.path.lexists(state_path)


def test_remote_profile_write_failures_do_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "profile.json")
    monkeypatch.setattr(
        ppr.PackedRowsProfile,
        "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("profile failed")),
    )
    with pytest.raises(RuntimeError, match="profile failed"):
        ppr.run(cfg)

    # The completed canonical outputs can be reauthenticated. The profile
    # exception is attached as a note while the authentication failure wins.
    cfg.resume = True
    monkeypatch.setattr(
        ppr,
        "discover_overlay",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("authentication failed")),
    )
    with pytest.raises(RuntimeError, match="authentication failed") as exc_info:
        ppr.run(cfg)
    assert exc_info.value.__notes__ == [
        "remote packed-row profile write also failed: RuntimeError('profile failed')"
    ]


def test_remote_profile_before_lookup_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    profile_path = tmp_path / "profile.json"
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)
    real_dump = ppr.atomic_json_dump

    def _fail_state(payload, path):
        if Path(path).name.endswith("remote-packing-state.json"):
            raise OSError("state failed")
        return real_dump(payload, path)

    monkeypatch.setattr(ppr, "atomic_json_dump", _fail_state)
    with pytest.raises(OSError, match="state failed"):
        ppr.run(cfg)
    remote = json.loads(profile_path.read_text())["remote_transport"]
    assert remote["status"] == "failed"
    assert remote["per_body_transport"] == []
    assert remote["total_attempts"] == 0
    assert remote["snapshot_list_seconds"] >= 0
    assert remote["fence_list_seconds"] == 0
    assert remote["list_calls"] == 1


def test_interrupted_finalization_resumes_and_completed_resume_reauthenticates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=3)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging-a", enabled=True)
    output_dir = _remote_output_dir(feature_dir, env, None)
    state_path, _lock_path = ppr._remote_state_paths(output_dir)

    real_certify = ppr.certify_fp_target_source
    certification_calls = 0

    def _fail_first_certification(*args, **kwargs):
        nonlocal certification_calls
        certification_calls += 1
        if certification_calls == 1:
            raise RuntimeError("injected validation certification failure")
        return real_certify(*args, **kwargs)

    monkeypatch.setattr(ppr, "certify_fp_target_source", _fail_first_certification)
    with pytest.raises(RuntimeError, match="injected validation certification failure"):
        ppr.run(cfg)
    assert os.path.lexists(state_path)
    assert ppr._contiguous_complete_shard_count(output_dir) > 0
    assert ppr._contiguous_complete_shard_count(output_dir.parent / "val_sall") > 0
    first_train = ppr.overlay_shard_path(output_dir, "shard_00000")
    first_provenance = ppr.overlay_shard_path(
        ppr.provenance_dir(output_dir),
        "shard_00000",
    )
    first_train_bytes = first_train.read_bytes()
    first_provenance_bytes = first_provenance.read_bytes()
    assert ppr.overlay_shard_path(output_dir, "shard_00001").is_file()
    first_train.unlink()
    first_provenance.unlink()
    assert ppr._contiguous_complete_shard_count(output_dir) == 0

    monkeypatch.setattr(ppr, "certify_fp_target_source", real_certify)
    cfg.resume = True
    cfg.remote.staging_dir = str(tmp_path / "staging-b")
    cfg.remote.retry_timeout_seconds = 2.0
    cfg.remote.retry_interval_seconds = 0.002
    cfg.compress_workers = 4
    cfg.profile = True
    cfg.profile_summary_output = str(tmp_path / "resume-profile.json")
    assert ppr.run(cfg) == 0
    assert not os.path.lexists(state_path)
    assert first_train.read_bytes() == first_train_bytes
    assert first_provenance.read_bytes() == first_provenance_bytes
    completed = _product_files(feature_dir)
    downloads_after_resume = len(bucket.downloads)

    mirror = next(
        path
        for path in feature_dir.rglob("*.safetensors.zst")
        if path.is_symlink() and "/tok/" in path.as_posix() and "/pct/" in path.as_posix()
    )
    mirror.unlink()
    log_messages: list[str] = []
    monkeypatch.setattr(ppr.logger, "info", lambda message: log_messages.append(str(message)))
    assert ppr.run(cfg) == 0
    assert _product_files(feature_dir) == completed
    assert mirror.is_symlink()
    assert not os.path.lexists(state_path)
    assert len(bucket.downloads) == downloads_after_resume + 1
    assert (
        "Remote packed-row completed artifact reauthenticated "
        "(train/val manifests + certificates revalidated)"
    ) in log_messages


def test_completed_resume_rejects_manifest_dims_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)
    assert ppr.run(cfg) == 0

    output_dir = _remote_output_dir(feature_dir, env, None)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["max_n_seg"] += 1
    manifest_path.write_text(json.dumps(manifest))

    cfg.resume = True
    with pytest.raises(RemotePackedRowsError, match=r"max_n_seg=\d+, expected \d+"):
        ppr.run(cfg)


@pytest.mark.parametrize("drift_field", ["zstd_level", "buffer_size"])
def test_resume_rejects_semantic_state_drift_before_body_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_field: str,
) -> None:
    env, bucket_root = _canonical_env(tmp_path, n_shards=2)
    feature_dir = Path(env["feature_dir"])
    for raw in glob_features(feature_dir):
        raw.unlink()
    bucket = _FakeBucket(bucket_root)
    _patch_remote_runtime(monkeypatch, env=env, bucket=bucket)
    cfg = _cfg(env, staging=tmp_path / "staging", enabled=True)

    real_precompute = ppr.precompute_packed_rows

    def _fail_before_packed_body(**_kwargs):
        raise RuntimeError("injected packing failure")

    monkeypatch.setattr(ppr, "precompute_packed_rows", _fail_before_packed_body)
    with pytest.raises(RuntimeError, match="injected packing failure"):
        ppr.run(cfg)
    downloads_after_failure = len(bucket.downloads)

    monkeypatch.setattr(ppr, "precompute_packed_rows", real_precompute)
    cfg.resume = True
    setattr(cfg, drift_field, int(getattr(cfg, drift_field)) + 1)
    with pytest.raises(RemotePackedRowsError, match="does not exactly match"):
        ppr.run(cfg)
    assert len(bucket.downloads) == downloads_after_failure + 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "is unreadable"),
        (b"\xff", "is unreadable"),
        (b"not-json", "is unreadable"),
        (b"[]", "must be a JSON object"),
    ],
)
def test_resume_state_rejects_invalid_json(
    tmp_path: Path,
    payload: bytes | None,
    message: str,
) -> None:
    missing = tmp_path / "missing.json"
    if payload is not None:
        missing.write_bytes(payload)
    with pytest.raises(RemotePackedRowsError, match=message):
        ppr._read_remote_state(missing)


def test_output_inventory_and_mirror_rejections(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.symlink_to(tmp_path / "missing-output")
    assert ppr._output_has_payload_or_manifest(output)
    output.unlink()

    output.mkdir()
    checkpoint_dir = ppr._remote_checkpoint_dir(output)
    checkpoint_dir.mkdir()
    (checkpoint_dir / "shard_00001.ckpt").write_bytes(b"stale")
    assert ppr._output_has_payload_or_manifest(output)
    shutil.rmtree(checkpoint_dir)
    provenance = ppr.provenance_dir(output)
    provenance.mkdir()
    (output / "bad.safetensors.zst").write_bytes(b"bad")
    with pytest.raises(RemotePackedRowsError, match="noncanonical"):
        ppr._packed_output_inventory(output)
    (output / "bad.safetensors.zst").unlink()

    (output / "shard_00001.safetensors.zst").write_bytes(b"gap")
    with pytest.raises(RemotePackedRowsError, match="not the exact contiguous family"):
        ppr._packed_output_inventory(output)
    (output / "shard_00001.safetensors.zst").unlink()

    (output / "shard_00000.safetensors.zst").write_bytes(b"packed")
    with pytest.raises(RemotePackedRowsError, match="must be nonempty, paired"):
        ppr._packed_output_inventory(output)

    source = tmp_path / "source"
    mirror = tmp_path / "mirror"
    source.mkdir()
    mirror.mkdir()
    body = source / "shard_00000.safetensors.zst"
    body.write_bytes(b"body")
    with pytest.raises(RemotePackedRowsError, match="inventory mismatch"):
        ppr._validate_mirror(source, mirror)
    wrong = mirror / body.name
    wrong.write_bytes(b"body")
    with pytest.raises(RemotePackedRowsError, match="expected file symlink"):
        ppr._validate_mirror(source, mirror)


def test_remote_output_classification_rejects_unauthenticated_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    val = tmp_path / "val"
    state = tmp_path / "state.json"
    kwargs = {
        "state_path": state,
        "expected_state": {},
        "output_dir": output,
        "val_output_dir": val,
        "identity": ppr._PackedArtifactIdentity(
            checksum="p:t",
            base_hash="a" * 64,
            base_shard_stems=("shard_00000",),
            context_len=8,
            max_shards=None,
            buffer_size=1,
            rows_per_shard=1,
            n_sae_features=1,
        ),
        "identity_digest": "d" * 64,
    }
    assert ppr._classify_remote_outputs(resume=False, **kwargs) == "fresh"
    output.mkdir()
    (output / "manifest.json").write_text("{}")
    with pytest.raises(RemotePackedRowsError, match="fresh remote packed-row output"):
        ppr._classify_remote_outputs(resume=False, **kwargs)
    (output / "manifest.json").unlink()
    with pytest.raises(RemotePackedRowsError, match="no authenticated output"):
        ppr._classify_remote_outputs(resume=True, **kwargs)
    (output / "shard_00000.safetensors.zst").write_bytes(b"payload")
    with pytest.raises(RemotePackedRowsError, match="manifest is missing or unreadable"):
        ppr._classify_remote_outputs(resume=True, **kwargs)


def test_remote_output_classification_rejects_genuine_v2_state_schema(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.json"
    # The version gate fires before any other state comparison, so a minimal
    # v2 marker suffices — the full v2 field layout is not this test's contract.
    state.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(
        RemotePackedRowsError,
        match=r"state schema version mismatch: state has 2.*current version is 3",
    ):
        ppr._classify_remote_outputs(
            resume=True,
            state_path=state,
            expected_state={"schema_version": 3},
            output_dir=tmp_path / "output",
            val_output_dir=tmp_path / "val",
            identity=ppr._PackedArtifactIdentity(
                checksum="p:t",
                base_hash="a" * 64,
                base_shard_stems=("shard_00000",),
                context_len=8,
                max_shards=None,
                buffer_size=1,
                rows_per_shard=1,
                n_sae_features=1,
            ),
            identity_digest="d" * 64,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("base_hash", "base hash mismatch"),
        ("metadata", "metadata is missing"),
        ("identity", "incompatible context_len"),
        ("provenance", "provenance_shards mismatch"),
        ("negative_max", "invalid max_n_seg"),
        ("wrong_max", "max_seg_size=2, expected 1"),
    ],
)
def test_packed_manifest_validation_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    metadata: object = {
        "type": "packed_rows",
        "context_len": 8,
        "max_shards": None,
        "buffer_size": 1,
        "rows_per_shard": 1,
        "pct_checksum": "p",
        "tok_checksum": "t",
        "base_shard_stems": ["shard_00000"],
        "provenance_shards": 1,
        "max_n_seg": 1,
        "max_seg_size": 1,
    }
    manifest = {
        "base_shard_content_hash": "a" * 64,
        "metadata": metadata,
    }
    if case == "base_hash":
        manifest["base_shard_content_hash"] = "b" * 64
    elif case == "metadata":
        manifest["metadata"] = None
    else:
        assert isinstance(metadata, dict)
        if case == "identity":
            metadata["context_len"] = True
        elif case == "provenance":
            metadata["provenance_shards"] = 2
        elif case == "negative_max":
            metadata["max_n_seg"] = -1
        else:
            metadata["max_seg_size"] = 2
    monkeypatch.setattr(ppr, "_read_required_manifest_path", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(ppr, "_packed_output_inventory", lambda _path, *, offloaded: 1)
    monkeypatch.setattr(
        ppr,
        "validate_fp_target_source_certificate",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RemotePackedRowsError, match=message):
        ppr._validate_packed_manifest(
            tmp_path,
            identity=ppr._PackedArtifactIdentity(
                checksum="p:t",
                base_hash="a" * 64,
                base_shard_stems=("shard_00000",),
                context_len=8,
                max_shards=None,
                buffer_size=1,
                rows_per_shard=1,
                n_sae_features=1,
            ),
            expected_max_n_seg=1,
            expected_max_seg_size=1,
        )


def test_product_helper_is_digest_stable(tmp_path: Path) -> None:
    """Keep the parity helper itself from silently following dot caches."""
    root = tmp_path / "features"
    body = root / "pct" / "x" / "tok" / "y" / "c32" / "sall" / "body"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"body")
    (body.parent / ".cache").write_bytes(b"cache")
    products = _product_files(root)
    assert list(products) == ["pct/x/tok/y/c32/sall/body"]
    assert hashlib.sha256(cast(bytes, products[next(iter(products))][1])).hexdigest()


@pytest.mark.parametrize(
    ("metadata", "key", "message"),
    [
        ({}, MetadataKey.N_DOCS, "lacks n_docs"),
        ({MetadataKey.N_DOCS: "bad"}, MetadataKey.N_DOCS, "invalid n_docs"),
        ({MetadataKey.N_SEGMENTS: "-1"}, MetadataKey.N_SEGMENTS, "negative n_segments"),
    ],
)
def test_raw_header_integer_parser_is_strict(metadata, key, message) -> None:
    with pytest.raises(RemotePackedRowsError, match=message):
        packed_remote._strict_nonnegative_metadata_int(
            metadata,
            key,
            shard_path=Path("shard"),
        )


@pytest.mark.parametrize("segment_size", [True, 0, "8"])
def test_lookup_requires_positive_integer_segment_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_size: object,
) -> None:
    lookup, transport = _stub_lookup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="positive integer"):
        RemotePackedRowsFeatureLookup(
            transport=cast(Any, transport),
            percentile_transport=cast(Any, lookup._percentile_transport),
            segment_size=cast(Any, segment_size),
            percentile_checksum="p" * 64,
            token_overlay_dir=lookup._token_overlay_dir,
            tokenizer_checksum="t" * 64,
            selected_parquets=tuple(lookup._selected.values()),
        )


def test_dual_stream_capacity_and_inventory_alignment_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, raw_transport = _stub_lookup(
        tmp_path,
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE), (1, 1, SEGMENT_SIZE)),
    )
    percentile_transport = lookup._percentile_transport
    assert isinstance(percentile_transport, _StubTransport)
    assert required_packed_rows_staging_capacity_bytes(
        raw_transport.inventory,
        percentile_transport.inventory,
    ) == sum(entry.info.byte_length for entry in raw_transport.inventory) + sum(
        entry.info.byte_length for entry in percentile_transport.inventory
    ) + (1 << 30)

    truncated = _StubTransport(
        (percentile_transport.inventory[0],),
        percentile_transport.prepared,
    )
    with pytest.raises(RemotePackedRowsError, match="align position-for-position"):
        RemotePackedRowsFeatureLookup(
            transport=cast(Any, raw_transport),
            percentile_transport=cast(Any, truncated),
            segment_size=SEGMENT_SIZE,
            percentile_checksum="p" * 64,
            token_overlay_dir=lookup._token_overlay_dir,
            tokenizer_checksum="t" * 64,
            selected_parquets=tuple(lookup._selected.values()),
        )


def test_lookup_seek_seeds_absolute_position_and_rejects_backwards_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path,
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE),) * 3,
        selected_positions=(0, 1, 2),
        start_position=2,
    )
    initial_profile = lookup.profile_payload()
    assert initial_profile["bodies_processed"] == 0
    assert initial_profile["cursor_position"] == 2
    assert lookup._consumed_bodies == 2
    assert lookup._activated_stems == {"shard_00000", "shard_00001"}
    with pytest.raises(RemotePackedRowsError, match="before its seek cursor"):
        lookup.lookup("shard_00001", 0)
    assert lookup.lookup("shard_00002", 0) == [("segment", 0)]


def test_lookup_rejects_skipped_nonempty_selected_and_allows_empty_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path / "nonempty",
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE), (1, 1, SEGMENT_SIZE)),
        selected_positions=(0, 1),
    )
    with pytest.raises(RemotePackedRowsError, match="not completely empty"):
        lookup.lookup("shard_00001", 0)

    lookup, _transport = _stub_lookup(
        tmp_path / "empty",
        monkeypatch,
        headers=((0, 0, SEGMENT_SIZE), (1, 1, SEGMENT_SIZE)),
        selected_positions=(0, 1),
    )
    assert lookup.lookup("shard_00001", 0) == [("segment", 0)]


def test_lookup_rejects_document_count_and_nonempty_zero_segment_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(tmp_path / "docs", monkeypatch)
    selected = lookup._selected["shard_00000"]
    lookup._selected["shard_00000"] = SelectedParquet(
        stem=selected.stem,
        path=selected.path,
        n_rows=2,
    )
    with pytest.raises(RemotePackedRowsError, match="document-count mismatch"):
        lookup.lookup("shard_00000", 0)

    lookup, _transport = _stub_lookup(
        tmp_path / "segments",
        monkeypatch,
        headers=((1, 0, SEGMENT_SIZE),),
    )
    with pytest.raises(RemotePackedRowsError, match="zero raw feature segments"):
        lookup.lookup("shard_00000", 0)


def test_lookup_request_and_finish_terminal_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(tmp_path / "unknown", monkeypatch)
    with pytest.raises(KeyError, match="Unknown remote shard"):
        lookup.lookup("shard_99999", 0)

    lookup, _transport = _stub_lookup(
        tmp_path / "short",
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE), (1, 1, SEGMENT_SIZE)),
        selected_positions=(1,),
        prepared_count=1,
    )
    with pytest.raises(RemotePackedRowsError, match="ended before requested"):
        lookup.lookup("shard_00001", 0)

    lookup, _transport = _stub_lookup(tmp_path / "finished", monkeypatch)
    lookup.lookup("shard_00000", 0)
    assert lookup.finish() is None
    assert lookup.finish() is None
    with pytest.raises(RemotePackedRowsError, match="already finished"):
        lookup.lookup("shard_00000", 0)


def test_finish_requires_every_nonempty_selected_stem_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path,
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE), (1, 1, SEGMENT_SIZE)),
        selected_positions=(0, 1),
    )
    lookup.lookup("shard_00000", 0)
    with pytest.raises(RemotePackedRowsError) as exc_info:
        lookup.finish()
    assert "missing=['shard_00001']" in str(exc_info.value)


def test_finish_preseeds_zero_row_selected_stem_without_body_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path,
        monkeypatch,
        headers=((0, 0, SEGMENT_SIZE),),
        selected_positions=(0,),
    )
    assert lookup._consumed_bodies == 0
    assert lookup.finish() is None
    assert lookup._consumed_bodies == 0


def test_percentile_stream_ending_early_is_loud_not_clean_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pct StopIteration mid-consume must never masquerade as end-of-stream."""
    lookup, transport = _stub_lookup(
        tmp_path / "percentile-short",
        monkeypatch,
        selected_positions=(),
    )
    lookup._stream = iter(  # ty: ignore[invalid-assignment]
        (transport.prepared[0],)
    )
    lookup._percentile_stream = iter(())  # ty: ignore[invalid-assignment]
    with pytest.raises(RemotePackedRowsError, match="percentile transport ended before"):
        lookup._consume_next()


def test_consume_rejects_malformed_header_segment_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path / "size",
        monkeypatch,
        headers=((1, 1, SEGMENT_SIZE + 1),),
        selected_positions=(),
    )
    with pytest.raises(RemotePackedRowsError, match="segment_size mismatch"):
        lookup._consume_next()


def test_activate_rejects_unselected_and_allows_zero_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup, _transport = _stub_lookup(
        tmp_path / "unselected",
        monkeypatch,
        selected_positions=(),
    )
    with pytest.raises(RemotePackedRowsError, match="unselected parquet"):
        lookup._activate("shard_00000")

    lookup, _transport = _stub_lookup(
        tmp_path / "zero",
        monkeypatch,
        headers=((0, 0, SEGMENT_SIZE),),
    )
    assert lookup.lookup("shard_00000", 0) == [("segment", 0)]
