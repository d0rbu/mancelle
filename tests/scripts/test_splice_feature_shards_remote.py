"""Bounded HF-backed execution tests for ``splice_feature_shards``."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import hf_xet
import pyarrow as pa
import pytest

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import COMPRESSED_EXT, save_features
from steering.remote_training import (
    VERIFIED_UPLOAD_CLAIMS_DIRNAME,
    RemoteObjectInfo,
    RemoteTrainingError,
)
from tests.scripts._splice_artifact_compare import assert_splice_dirs_equal
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)
SHARD_TEXTS = {
    0: ["alpha beta", "gamma delta epsilon", "zeta eta"],
    1: ["one two three", "four five", "six seven eight nine"],
    2: ["red green blue", "cyan magenta", "yellow black white"],
}


class FakeBucket:
    def __init__(self, root: Path):
        self.root = root
        self.download_order: list[str] = []
        self.upload_order: list[str] = []

    def _path(self, remote_path: str) -> Path:
        path = self.root / remote_path
        assert path.resolve().is_relative_to(self.root.resolve())
        return path

    @staticmethod
    def _info(path: Path, remote_path: str) -> RemoteObjectInfo:
        result = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(result.hash))

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        source = self._path(remote_path)
        if not source.is_file():
            raise FileNotFoundError(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)
        self.download_order.append(remote_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)
        self.upload_order.append(remote_path)

    def put(self, local_path: Path, remote_path: str) -> None:
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return TOK

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda _model: 8192)


def _build_base(tmp_path: Path) -> tuple[Path, Path, Path]:
    local_features_root = tmp_path / "features"
    feature_dir = local_features_root / "base"
    data_dir = tmp_path / "parquets"
    feature_dir.mkdir(parents=True)
    data_dir.mkdir()
    for index, texts in SHARD_TEXTS.items():
        shard = make_base_shard(texts, TOK, 2, context_window=2048)
        save_features(shard, feature_dir / f"shard_{index:05d}")
        atomic_write_parquet(
            pa.table({"text": texts}), data_dir / f"shard_{index:05d}.parquet"
        )
    return local_features_root, feature_dir, data_dir


def _base_cfg(
    *,
    local_features_root: Path,
    feature_dir: Path,
    data_dir: Path,
    output_dir: Path,
    staging_dir: Path,
    profile_path: Path | None = None,
    expected_shard_count: int | None = None,
):
    return sfs.load_splice_config(
        {
            "feature_dir": str(feature_dir),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "model": "gemma-2-2b",
            "segment_size": 2,
            "context_window": 2048,
            "cut_fraction": 0.6,
            "seed": 42,
            "profile": profile_path is not None,
            "profile_summary_output": str(profile_path or "unused.json"),
            "remote": {
                "enabled": True,
                "bucket_id": "owner/test-bucket",
                "local_features_root": str(local_features_root),
                "remote_features_root": "features",
                "staging_dir": str(staging_dir),
                "expected_shard_count": expected_shard_count,
                "retry_timeout_seconds": 1.0,
                "retry_interval_seconds": 0.001,
            },
        }
    )


def _seed_sources(bucket: FakeBucket, feature_dir: Path) -> None:
    for source in sorted(feature_dir.glob(f"shard_*_features{COMPRESSED_EXT}")):
        bucket.put(source, f"features/base/{source.name}")


def _run_local_reference(
    feature_dir: Path, data_dir: Path, output_dir: Path, *,
    zstd_threads: int = 0, zstd_level: int | None = None
) -> None:
    sfs.run(
        sfs.load_splice_config(
            {
                "feature_dir": str(feature_dir),
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "model": "gemma-2-2b",
                "segment_size": 2,
                "context_window": 2048,
                "cut_fraction": 0.6,
                "zstd_threads": zstd_threads,
                "zstd_level": zstd_level,
            }
        )
    )


def _download_output_tree(bucket: FakeBucket, remote_prefix: str, local: Path) -> None:
    for info in bucket.list(remote_prefix):
        relative = Path(info.path).relative_to(remote_prefix)
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME in relative.parts:
            continue
        destination = local / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bucket._path(info.path), destination)


def test_inventory_filters_nested_objects_and_uses_stable_modulo_ownership() -> None:
    valid = [
        RemoteObjectInfo(
            f"features/base/shard_{index:05d}_features{COMPRESSED_EXT}",
            index + 1,
            f"{index + 1:064x}",
        )
        for index in (0, 2, 5)
    ]
    listing = [
        *valid,
        RemoteObjectInfo("features/base/feature_statistics.safetensors", 4, "a" * 64),
        RemoteObjectInfo(
            f"features/base/splice/cut/shard_00001_features{COMPRESSED_EXT}",
            1,
            "b" * 64,
        ),
    ]
    sources, digest = sfs._discover_remote_sources(
        listing, prefix="features/base", expected_shard_count=None
    )
    assert sorted(sources) == [0, 2, 5]
    assert len(digest) == 64
    rank_zero = [index for index in sources if index % 2 == 0]
    rank_one = [index for index in sources if index % 2 == 1]
    assert rank_zero == [0, 2]
    assert rank_one == [5]
    sources_with_gap_fill, _ = sfs._discover_remote_sources(
        [
            *listing,
            RemoteObjectInfo(
                f"features/base/shard_00001_features{COMPRESSED_EXT}",
                2,
                "c" * 64,
            ),
        ],
        prefix="features/base",
        expected_shard_count=None,
    )
    assert [index for index in sources_with_gap_fill if index % 2 == 0] == rank_zero


def test_inventory_rejects_malformed_duplicate_empty_hash_and_expected_count() -> None:
    good = RemoteObjectInfo(
        f"features/base/shard_00000_features{COMPRESSED_EXT}", 1, "a" * 64
    )
    with pytest.raises(ValueError, match="malformed"):
        sfs._discover_remote_sources(
            [
                RemoteObjectInfo(
                    f"features/base/shard_0_features{COMPRESSED_EXT}", 1, "a" * 64
                )
            ],
            prefix="features/base",
            expected_shard_count=None,
        )
    with pytest.raises(ValueError, match="duplicate"):
        sfs._discover_remote_sources(
            [good, good], prefix="features/base", expected_shard_count=None
        )
    for bad in (
        RemoteObjectInfo(good.path, 0, good.xet_hash),
        RemoteObjectInfo(good.path, 1, "not-a-hash"),
    ):
        with pytest.raises(ValueError, match="invalid"):
            sfs._discover_remote_sources(
                [bad], prefix="features/base", expected_shard_count=None
            )
    with pytest.raises(ValueError, match="exact expected contiguous set"):
        sfs._discover_remote_sources(
            [good], prefix="features/base", expected_shard_count=2
        )


@pytest.mark.parametrize(
    ("parquet", "provenance", "feature", "valid"),
    [
        (False, False, False, True),
        (True, False, False, True),
        (True, True, False, True),
        (True, True, True, True),
        (False, False, True, False),
        (False, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_remote_publication_state_matrix(
    parquet: bool,
    provenance: bool,
    feature: bool,
    valid: bool,
) -> None:
    info = RemoteObjectInfo("unused", 1, "a" * 64)
    states = {
        0: sfs._RemoteOutputState(
            parquet=info if parquet else None,
            provenance=info if provenance else None,
            feature=info if feature else None,
        )
    }
    if valid:
        sfs._validate_remote_output_states(states, owned_indices=[0])
    else:
        with pytest.raises(ValueError, match="impossible remote publication state"):
            sfs._validate_remote_output_states(states, owned_indices=[0])


def test_remote_config_mapping_and_guards(tmp_path: Path) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=feature_dir / "splice" / "cut_frac0.6" / "seed42",
        staging_dir=tmp_path / "staging",
    )
    layout = sfs._resolve_remote_layout(cfg)
    assert layout.feature_prefix == "features/base"
    assert layout.output_prefix == "features/base/splice/cut_frac0.6/seed42"

    cfg.remote.staging_dir = str(feature_dir / "scratch")
    with pytest.raises(ValueError, match="disjoint"):
        sfs._resolve_remote_layout(cfg)
    cfg.remote.staging_dir = str(tmp_path / "staging")
    cfg.output_dir = str(tmp_path / "escape")
    with pytest.raises(ValueError, match="escapes"):
        sfs._resolve_remote_layout(cfg)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"feature_dir": None}, "feature_dir"),
        ({"data_dir": None}, "data_dir"),
        ({"remote.staging_dir": None}, "staging_dir"),
        ({"cut_fraction": 1.01}, "cut_fraction"),
        ({"shard_parallelism": 2}, "shard_parallelism=1"),
        ({"async_feature_save_depth": 1}, "async_feature_save_depth=0"),
        ({"zstd_threads": -1}, "zstd_threads"),
        ({"zstd_threads": "2"}, "zstd_threads"),
        ({"zstd_threads": 1.5}, "zstd_threads"),
        ({"zstd_threads": True}, "zstd_threads"),
        ({"zstd_threads": None}, "zstd_threads"),
        *[({"zstd_level": value}, "zstd_level") for value in [False, True, 0, 23, 1.5, "8"]],
        ({"skip_existing": False}, "skip_existing=true"),
        ({"write_provenance": False}, "write_provenance=true"),
        ({"remote.expected_shard_count": 0}, "expected_shard_count"),
        ({"remote.retry_timeout_seconds": 0.0}, "retry_timeout_seconds"),
        ({"remote.retry_interval_seconds": -1.0}, "retry_interval_seconds"),
    ],
)
def test_remote_config_rejects_unsupported_or_invalid_values(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    for dotted_key, value in override.items():
        target: Any = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    with pytest.raises(ValueError, match=match):
        sfs._resolve_remote_layout(cfg)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("remote.retry_timeout_seconds", float("nan"), "finite"),
        ("remote.retry_timeout_seconds", float("inf"), "finite"),
        ("remote.retry_interval_seconds", float("nan"), "finite"),
        ("shard_parallelism", True, "integer"),
        ("async_feature_save_depth", False, "integer"),
        ("seed", True, "seed must be an integer"),
    ],
)
def test_remote_config_rejects_nonfinite_and_boolean_numeric_values(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    target: Any = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(ValueError, match=match):
        sfs._resolve_remote_layout(cfg)


@pytest.mark.parametrize(
    ("rank", "world_size", "match"),
    [
        ("not-an-int", "2", "must be integers"),
        ("0", "0", "WORLD_SIZE must be >= 1"),
        ("-1", "2", "0 <= RANK < WORLD_SIZE"),
        ("2", "2", "0 <= RANK < WORLD_SIZE"),
    ],
)
def test_remote_rank_world_size_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: str,
    world_size: str,
    match: str,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)
    with pytest.raises(ValueError, match=match):
        sfs._resolve_remote_layout(cfg)


def test_remote_path_and_output_inventory_metadata_are_fail_loud() -> None:
    with pytest.raises(ValueError, match="non-normalized"):
        sfs._relative_remote_parts("/features/base/body", "features/base")
    with pytest.raises(ValueError, match="outside the prefix"):
        sfs._relative_remote_parts("features/sibling/body", "features/base")

    info = RemoteObjectInfo(
        f"features/derived/shard_00000_features{COMPRESSED_EXT}",
        1,
        "a" * 64,
    )
    with pytest.raises(ValueError, match="duplicate remote output"):
        sfs._discover_remote_output_states(
            [info, info],
            prefix="features/derived",
            source_indices={0},
        )
    with pytest.raises(ValueError, match="invalid Xet hash"):
        sfs._discover_remote_output_states(
            [RemoteObjectInfo(info.path, 1, "invalid")],
            prefix="features/derived",
            source_indices={0},
        )


def test_disabled_remote_does_not_construct_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = sfs.load_splice_config({"feature_dir": str(tmp_path / "missing")})
    monkeypatch.setattr(
        sfs,
        "HfBucketBackend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("backend constructed")),
    )
    with pytest.raises(ValueError, match="no base feature shards"):
        sfs.run(cfg)


def test_remote_empty_inventory_and_missing_parquet_fail_before_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    empty_bucket = FakeBucket(tmp_path / "empty-bucket")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: empty_bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    with pytest.raises(ValueError, match="no direct raw feature shard"):
        sfs.run(cfg)
    assert empty_bucket.download_order == []
    assert empty_bucket.upload_order == []

    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    (data_dir / "shard_00002.parquet").unlink()
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    with pytest.raises(ValueError, match="missing local source parquet"):
        sfs.run(cfg)
    assert bucket.download_order == []
    assert bucket.upload_order == []


def test_rank_with_no_owned_shards_is_a_transfer_free_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )

    sfs.run(cfg)

    assert bucket.download_order == []
    assert bucket.upload_order == []
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize("zstd_threads,zstd_level", [(0, None), (2, 8)])
def test_remote_end_to_end_parity_cleanup_and_all_skip_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zstd_threads: int,
    zstd_level: int | None,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    reference = root / "derived"
    _run_local_reference(
        feature_dir, data_dir, reference, zstd_threads=zstd_threads, zstd_level=zstd_level
    )

    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    profile_path = tmp_path / "profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=reference,
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
        expected_shard_count=3,
    )
    cfg.zstd_threads = zstd_threads
    cfg.zstd_level = zstd_level
    sfs.run(cfg)

    remote_prefix = "features/derived"
    downloaded = tmp_path / "downloaded"
    _download_output_tree(bucket, remote_prefix, downloaded)
    assert_splice_dirs_equal(reference, downloaded)
    for path in sorted((downloaded / sfs.PROVENANCE_DIRNAME).glob("*.json")):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        assert provenance["params"]["base_feature_dir"] == str(feature_dir)
        assert "staging" not in provenance["params"]["base_feature_dir"]
    expected_order: list[str] = []
    for index in range(3):
        stem = f"shard_{index:05d}"
        expected_order.extend(
            [
                f"{remote_prefix}/{sfs.PARQUET_DIRNAME}/{stem}.parquet",
                f"{remote_prefix}/{sfs.PROVENANCE_DIRNAME}/{stem}.json",
                f"{remote_prefix}/{stem}_features{COMPRESSED_EXT}",
            ]
        )
    canonical_uploads = [
        path
        for path in bucket.upload_order
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(path).parts
    ]
    claim_uploads = [
        path
        for path in bucket.upload_order
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME in PurePosixPath(path).parts
    ]
    assert canonical_uploads == expected_order
    assert len(claim_uploads) == 9
    rank_dir = tmp_path / "staging" / "rank00000"
    assert not list((rank_dir / "source").iterdir())
    assert not [path for path in (rank_dir / "output").iterdir() if path.is_file()]
    assert not list((rank_dir / "output" / sfs.PARQUET_DIRNAME).iterdir())
    assert not list((rank_dir / "output" / sfs.PROVENANCE_DIRNAME).iterdir())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["config"]["remote"]["source_inventory_digest"]
    assert profile["config"]["zstd_threads"] == zstd_threads
    assert profile["config"]["zstd_level"] == zstd_level
    assert profile["totals"]["committed"] == 3
    assert profile["totals"]["n_ok"] == 3

    bucket.download_order.clear()
    bucket.upload_order.clear()
    before = {info.path: (info.byte_length, info.xet_hash) for info in bucket.list(remote_prefix)}
    monkeypatch.setattr(
        sfs, "resolve_tokenizer", Mock(side_effect=AssertionError("all-skip must not load tokenizer"))
    )
    sfs.run(cfg)
    after = {info.path: (info.byte_length, info.xet_hash) for info in bucket.list(remote_prefix)}
    assert before == after
    assert bucket.upload_order == []
    assert bucket.download_order == [
        f"features/base/shard_00000_features{COMPRESSED_EXT}"
    ] + [
        f"{remote_prefix}/{sfs.PROVENANCE_DIRNAME}/shard_{index:05d}.json"
        for index in range(3)
    ]
    rerun_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert rerun_profile["totals"]["n_skipped"] == 3
    assert rerun_profile["totals"]["committed"] == 0
    source_zero = bucket.stat(f"features/base/shard_00000_features{COMPRESSED_EXT}")
    assert source_zero is not None
    assert rerun_profile["totals"]["source_bytes_downloaded"] == source_zero.byte_length


@pytest.mark.parametrize("setting,initial,changed", [("zstd_threads", 2, 0), ("zstd_level", 8, None)])
def test_remote_restart_preserves_claims_and_rejects_changed_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, initial: int, changed: int | None
) -> None:
    feature_name = f"shard_00000_features{COMPRESSED_EXT}"

    class InterruptingBucket(FakeBucket):
        interrupted = True

        def upload(self, local_path: Path, remote_path: str) -> None:
            if self.interrupted and remote_path.endswith(feature_name):
                raise OSError("injected feature upload interruption")
            super().upload(local_path, remote_path)

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = InterruptingBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    cfg[setting] = initial
    cfg.remote.retry_timeout_seconds = 0.01
    with pytest.raises(RemoteTrainingError, match="timed out"):
        sfs.run(cfg)
    before = list(bucket.list("features/derived"))
    assert not bucket._path(f"features/derived/{feature_name}").exists()
    bucket.interrupted = False
    cfg[setting] = changed
    with pytest.raises(RemoteTrainingError, match="collision"):
        sfs.run(cfg)
    assert list(bucket.list("features/derived")) == before
    cfg[setting] = initial
    sfs.run(cfg)
    assert bucket._path(f"features/derived/{feature_name}").is_file()
    bucket.upload_order.clear()
    cfg[setting] = changed
    with pytest.raises(ValueError, match=setting):
        sfs.run(cfg)
    assert bucket.upload_order == []


def test_all_skip_params_are_resolved_from_raw_source_not_existing_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
        expected_shard_count=3,
    )
    sfs.run(cfg)

    for index in range(3):
        provenance_path = bucket._path(
            f"features/derived/{sfs.PROVENANCE_DIRNAME}/shard_{index:05d}.json"
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["params"]["model"] = "wrong-model"
        provenance["params"]["segment_size"] = 4
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    cfg.model = None
    cfg.segment_size = None
    monkeypatch.setattr(
        sfs,
        "parse_feature_dir",
        lambda _path: SimpleNamespace(model="gemma-2-2b"),
    )
    bucket.download_order.clear()
    bucket.upload_order.clear()

    with pytest.raises(ValueError, match="mixed-provenance"):
        sfs.run(cfg)

    assert bucket.download_order[:2] == [
        f"features/base/shard_00000_features{COMPRESSED_EXT}",
        f"features/derived/{sfs.PROVENANCE_DIRNAME}/shard_00000.json",
    ]
    assert bucket.upload_order == []


def test_all_skip_provenance_is_bound_to_logical_feature_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
        expected_shard_count=3,
    )
    sfs.run(cfg)

    for index in range(3):
        provenance_path = bucket._path(
            f"features/derived/{sfs.PROVENANCE_DIRNAME}/shard_{index:05d}.json"
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["params"]["base_feature_dir"] = str(root / "different-layout")
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    bucket.download_order.clear()
    bucket.upload_order.clear()

    with pytest.raises(ValueError, match="base_feature_dir"):
        sfs.run(cfg)

    assert bucket.download_order[:2] == [
        f"features/base/shard_00000_features{COMPRESSED_EXT}",
        f"features/derived/{sfs.PROVENANCE_DIRNAME}/shard_00000.json",
    ]
    assert bucket.upload_order == []


def test_partial_resume_bootstraps_from_first_pending_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    reference = root / "reference"
    _run_local_reference(feature_dir, data_dir, reference)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    remote_prefix = "features/derived"
    for relative in (
        Path(sfs.PARQUET_DIRNAME) / "shard_00000.parquet",
        Path(sfs.PROVENANCE_DIRNAME) / "shard_00000.json",
        Path(f"shard_00000_features{COMPRESSED_EXT}"),
    ):
        bucket.put(reference / relative, f"{remote_prefix}/{relative.as_posix()}")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    profile_path = tmp_path / "profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
        expected_shard_count=3,
    )

    sfs.run(cfg)

    raw_zero = f"features/base/shard_00000_features{COMPRESSED_EXT}"
    raw_one = f"features/base/shard_00001_features{COMPRESSED_EXT}"
    raw_two = f"features/base/shard_00002_features{COMPRESSED_EXT}"
    assert raw_zero not in bucket.download_order
    assert bucket.download_order[0] == raw_one
    assert bucket.download_order.count(raw_one) == 1
    assert bucket.download_order.count(raw_two) == 1
    info_one = bucket.stat(raw_one)
    info_two = bucket.stat(raw_two)
    assert info_one is not None and info_two is not None
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["totals"]["source_bytes_downloaded"] == (
        info_one.byte_length + info_two.byte_length
    )


def test_two_rank_remote_union_matches_serial_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    reference = root / "reference"
    _run_local_reference(feature_dir, data_dir, reference)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    output_dir = root / "striped"
    profile_path = tmp_path / "striped-profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
        expected_shard_count=3,
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    sfs.run(cfg)
    monkeypatch.setenv("RANK", "1")
    sfs.run(cfg)

    downloaded = tmp_path / "striped-downloaded"
    _download_output_tree(bucket, "features/striped", downloaded)
    assert_splice_dirs_equal(reference, downloaded)
    rank_zero_profile = json.loads(
        profile_path.with_name("striped-profile.rank00000.json").read_text(
            encoding="utf-8"
        )
    )
    rank_one_profile = json.loads(
        profile_path.with_name("striped-profile.rank00001.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["shard_index"] for record in rank_zero_profile["per_shard"]] == [
        0,
        2,
    ]
    assert [record["shard_index"] for record in rank_one_profile["per_shard"]] == [1]
    assert (
        rank_zero_profile["config"]["remote"]["source_inventory_digest"]
        == rank_one_profile["config"]["remote"]["source_inventory_digest"]
    )


@pytest.mark.parametrize("existing_sidecars", [1, 2])
def test_valid_partial_publication_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_sidecars: int,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    reference = root / "derived"
    _run_local_reference(feature_dir, data_dir, reference)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    remote_prefix = "features/derived"
    parquet = reference / sfs.PARQUET_DIRNAME / "shard_00000.parquet"
    bucket.put(
        parquet,
        f"{remote_prefix}/{sfs.PARQUET_DIRNAME}/shard_00000.parquet",
    )
    if existing_sidecars == 2:
        bucket.put(
            reference / sfs.PROVENANCE_DIRNAME / "shard_00000.json",
            f"{remote_prefix}/{sfs.PROVENANCE_DIRNAME}/shard_00000.json",
        )
    preexisting_bytes = sum(info.byte_length for info in bucket.list(remote_prefix))
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    profile_path = tmp_path / "partial-profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=reference,
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
        expected_shard_count=3,
    )

    sfs.run(cfg)

    shard_zero_uploads = [
        path
        for path in bucket.upload_order
        if "shard_00000" in path
        and VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(path).parts
    ]
    if existing_sidecars == 1:
        assert shard_zero_uploads == [
            f"{remote_prefix}/{sfs.PROVENANCE_DIRNAME}/shard_00000.json",
            f"{remote_prefix}/shard_00000_features{COMPRESSED_EXT}",
        ]
    else:
        assert shard_zero_uploads == [
            f"{remote_prefix}/shard_00000_features{COMPRESSED_EXT}"
        ]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    final_canonical_bytes = sum(
        info.byte_length
        for info in bucket.list(remote_prefix)
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(info.path).parts
    )
    assert profile["totals"]["output_bytes_uploaded"] == (
        final_canonical_bytes - preexisting_bytes
    )


def test_differing_existing_sidecar_is_collision_and_feature_is_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    remote_prefix = "features/derived"
    collision = bucket._path(
        f"{remote_prefix}/{sfs.PARQUET_DIRNAME}/shard_00000.parquet"
    )
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"different deterministic sidecar")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    profile_path = tmp_path / "failed-profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
    )

    with pytest.raises(RemoteTrainingError, match="collision"):
        sfs.run(cfg)

    assert not any(
        path.endswith(f"shard_00000_features{COMPRESSED_EXT}")
        for path in bucket.upload_order
    )
    staged_output = tmp_path / "staging" / "rank00000" / "output"
    assert (staged_output / f"shard_00000_features{COMPRESSED_EXT}").is_file()
    assert (
        staged_output / sfs.PARQUET_DIRNAME / "shard_00000.parquet"
    ).is_file()
    assert (
        staged_output / sfs.PROVENANCE_DIRNAME / "shard_00000.json"
    ).is_file()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["per_shard"][0]["status"] == "failed"
    assert profile["per_shard"][0]["error_type"] == "RemoteTrainingError"
    assert "collision" in profile["per_shard"][0]["error_message"]


def test_differing_concurrent_publications_claim_before_canonical_upload(
    tmp_path: Path,
) -> None:
    class ClaimBarrierBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.claim_barrier = threading.Barrier(2)

        def upload(self, local_path: Path, remote_path: str) -> None:
            super().upload(local_path, remote_path)
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME in PurePosixPath(remote_path).parts:
                self.claim_barrier.wait(timeout=5.0)

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = ClaimBarrierBucket(tmp_path / "bucket")
    layouts: list[sfs._RemoteLayout] = []
    for suffix in ("a", "b"):
        cfg = _base_cfg(
            local_features_root=root,
            feature_dir=feature_dir,
            data_dir=data_dir,
            output_dir=root / "derived",
            staging_dir=tmp_path / f"staging-{suffix}",
        )
        layouts.append(sfs._resolve_remote_layout(cfg))

    errors: list[BaseException] = []
    with sfs._locked_rank_staging(layouts[0]) as staging_a:
        with sfs._locked_rank_staging(layouts[1]) as staging_b:
            jobs: list[tuple[sfs._RemoteLayout, sfs._RemoteStaging, sfs.SpliceJob]] = []
            for suffix, layout, staging in (
                ("a", layouts[0], staging_a),
                ("b", layouts[1], staging_b),
            ):
                stem = "shard_00000"
                parquet = staging.parquet_dir / f"{stem}.parquet"
                provenance = staging.provenance_dir / f"{stem}.json"
                feature = staging.output_dir / f"{stem}_features{COMPRESSED_EXT}"
                parquet.write_bytes(f"parquet-{suffix}".encode())
                provenance.write_bytes(f"provenance-{suffix}".encode())
                feature.write_bytes(f"feature-{suffix}".encode())
                jobs.append(
                    (
                        layout,
                        staging,
                        sfs.SpliceJob(
                            base_shard_path=staging.source_dir
                            / f"{stem}_features{COMPRESSED_EXT}",
                            parquet_path=data_dir / f"{stem}.parquet",
                            shard_index=0,
                            shard_name=stem,
                            derived_shard_path=staging.output_dir / stem,
                            derived_parquet_path=parquet,
                            provenance_path=provenance,
                        ),
                    )
                )

            result = sfs.SpliceJobResult(
                shard_name="shard_00000",
                shard_index=0,
                status=sfs.SpliceJobStatus.ok,
                n_docs=1,
                n_switches=0,
            )

            def publish(
                layout: sfs._RemoteLayout,
                staging: sfs._RemoteStaging,
                job: sfs.SpliceJob,
            ) -> None:
                try:
                    sfs._commit_remote_job(bucket, layout, staging, job, result)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [
                threading.Thread(target=publish, args=job_args) for job_args in jobs
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10.0)
                assert not thread.is_alive()

    assert len(errors) == 2
    assert all(
        isinstance(exc, RemoteTrainingError) and "collision" in str(exc)
        for exc in errors
    )
    assert bucket.stat("features/derived/_parquets/shard_00000.parquet") is None
    assert bucket.stat("features/derived/_splice_provenance/shard_00000.json") is None
    assert bucket.stat(f"features/derived/shard_00000_features{COMPRESSED_EXT}") is None


@pytest.mark.parametrize(
    ("fail_suffix", "published_before_failure"),
    [
        ("_parquets/shard_00000.parquet", []),
        (
            "_splice_provenance/shard_00000.json",
            ["_parquets/shard_00000.parquet"],
        ),
        (
            f"shard_00000_features{COMPRESSED_EXT}",
            [
                "_parquets/shard_00000.parquet",
                "_splice_provenance/shard_00000.json",
            ],
        ),
    ],
)
def test_each_publication_interruption_boundary_restarts_safely(
    tmp_path: Path,
    fail_suffix: str,
    published_before_failure: list[str],
) -> None:
    class InterruptingBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.fail_suffix: str | None = fail_suffix

        def upload(self, local_path: Path, remote_path: str) -> None:
            if self.fail_suffix is not None and remote_path.endswith(self.fail_suffix):
                raise OSError(f"injected interruption at {self.fail_suffix}")
            super().upload(local_path, remote_path)

    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    cfg.remote.retry_timeout_seconds = 0.01
    cfg.remote.retry_interval_seconds = 0.001
    layout = sfs._resolve_remote_layout(cfg)
    bucket = InterruptingBucket(tmp_path / "bucket")
    result = sfs.SpliceJobResult(
        shard_name="shard_00000",
        shard_index=0,
        status=sfs.SpliceJobStatus.ok,
        n_docs=1,
        n_switches=0,
    )

    with sfs._locked_rank_staging(layout) as staging:
        stem = "shard_00000"
        parquet = staging.parquet_dir / f"{stem}.parquet"
        provenance = staging.provenance_dir / f"{stem}.json"
        feature = staging.output_dir / f"{stem}_features{COMPRESSED_EXT}"
        parquet.write_bytes(b"parquet")
        provenance.write_bytes(b"provenance")
        feature.write_bytes(b"feature")
        job = sfs.SpliceJob(
            base_shard_path=staging.source_dir / f"{stem}_features{COMPRESSED_EXT}",
            parquet_path=data_dir / f"{stem}.parquet",
            shard_index=0,
            shard_name=stem,
            derived_shard_path=staging.output_dir / stem,
            derived_parquet_path=parquet,
            provenance_path=provenance,
        )

        with pytest.raises(RemoteTrainingError, match="timed out"):
            sfs._commit_remote_job(bucket, layout, staging, job, result)

        canonical = [
            info.path
            for info in bucket.list("features/derived")
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(info.path).parts
        ]
        assert canonical == [
            f"features/derived/{relative}" for relative in published_before_failure
        ]
        assert all(path.is_file() for path in (parquet, provenance, feature))

        bucket.fail_suffix = None
        committed = sfs._commit_remote_job(
            bucket,
            layout,
            staging,
            job,
            result,
            frozenset(canonical),
        )

        assert committed.job_result == result
        assert all(not path.exists() for path in (parquet, provenance, feature))
        final_canonical = [
            info.path
            for info in bucket.list("features/derived")
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(info.path).parts
        ]
        assert final_canonical == [
            "features/derived/_parquets/shard_00000.parquet",
            "features/derived/_splice_provenance/shard_00000.json",
            f"features/derived/shard_00000_features{COMPRESSED_EXT}",
        ]


def test_cleanup_failure_after_feature_verification_is_idempotently_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    layout = sfs._resolve_remote_layout(cfg)
    bucket = FakeBucket(tmp_path / "bucket")
    result = sfs.SpliceJobResult(
        shard_name="shard_00000",
        shard_index=0,
        status=sfs.SpliceJobStatus.ok,
        n_docs=1,
        n_switches=0,
    )

    with sfs._locked_rank_staging(layout) as staging:
        stem = "shard_00000"
        parquet = staging.parquet_dir / f"{stem}.parquet"
        provenance = staging.provenance_dir / f"{stem}.json"
        feature = staging.output_dir / f"{stem}_features{COMPRESSED_EXT}"
        parquet.write_bytes(b"parquet")
        provenance.write_bytes(b"provenance")
        feature.write_bytes(b"feature")
        job = sfs.SpliceJob(
            base_shard_path=staging.source_dir / f"{stem}_features{COMPRESSED_EXT}",
            parquet_path=data_dir / f"{stem}.parquet",
            shard_index=0,
            shard_name=stem,
            derived_shard_path=staging.output_dir / stem,
            derived_parquet_path=parquet,
            provenance_path=provenance,
        )
        original_unlink = Path.unlink
        failed_once = False

        def fail_first_output_cleanup(path: Path, *args, **kwargs) -> None:
            nonlocal failed_once
            if path == parquet and not failed_once:
                failed_once = True
                raise OSError("injected cleanup interruption")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_first_output_cleanup)
        with pytest.raises(OSError, match="cleanup interruption"):
            sfs._commit_remote_job(bucket, layout, staging, job, result)

        assert all(path.is_file() for path in (parquet, provenance, feature))
        canonical = [
            info.path
            for info in bucket.list("features/derived")
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(info.path).parts
        ]
        assert len(canonical) == 3
        canonical_upload_order = [
            path
            for path in bucket.upload_order
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(path).parts
        ]

        committed = sfs._commit_remote_job(
            bucket,
            layout,
            staging,
            job,
            result,
            frozenset(canonical),
        )

        assert committed.output_bytes == 0
        assert all(not path.exists() for path in (parquet, provenance, feature))
        assert [
            path
            for path in bucket.upload_order
            if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in PurePosixPath(path).parts
        ] == canonical_upload_order


def test_complete_mismatched_provenance_fails_before_any_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    reference = root / "derived"
    _run_local_reference(feature_dir, data_dir, reference)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    remote_prefix = "features/derived"
    for relative in (
        Path(sfs.PARQUET_DIRNAME) / "shard_00000.parquet",
        Path(sfs.PROVENANCE_DIRNAME) / "shard_00000.json",
        Path(f"shard_00000_features{COMPRESSED_EXT}"),
    ):
        bucket.put(reference / relative, f"{remote_prefix}/{relative.as_posix()}")
    provenance_path = bucket._path(
        f"{remote_prefix}/{sfs.PROVENANCE_DIRNAME}/shard_00000.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["params"]["seed"] = 999
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=reference,
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(ValueError, match="mixed-provenance"):
        sfs.run(cfg)
    assert bucket.upload_order == []


def test_blocked_upload_backpressures_before_third_splice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.first_upload_started = threading.Event()
            self.allow_first_upload = threading.Event()
            self.active_downloads = 0
            self.max_active_downloads = 0
            self.active_uploads = 0
            self.max_active_uploads = 0

        def download(self, remote_path: str, local_path: Path) -> None:
            self.active_downloads += 1
            self.max_active_downloads = max(self.max_active_downloads, self.active_downloads)
            try:
                super().download(remote_path, local_path)
            finally:
                self.active_downloads -= 1

        def upload(self, local_path: Path, remote_path: str) -> None:
            self.active_uploads += 1
            self.max_active_uploads = max(self.max_active_uploads, self.active_uploads)
            try:
                if not self.first_upload_started.is_set():
                    self.first_upload_started.set()
                    assert self.allow_first_upload.wait(timeout=5.0)
                super().upload(local_path, remote_path)
            finally:
                self.active_uploads -= 1

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = BlockingBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    started_splices: list[int] = []
    second_finished = threading.Event()
    original_process = sfs._process_job

    def observe_process(*args, **kwargs):
        job = args[0]
        started_splices.append(job.shard_index)
        result = original_process(*args, **kwargs)
        if job.shard_index == 1:
            second_finished.set()
        return result

    monkeypatch.setattr(sfs, "_process_job", observe_process)
    failure: list[BaseException] = []

    def target() -> None:
        try:
            sfs.run(cfg)
        except BaseException as exc:  # pragma: no cover - asserted below
            failure.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    assert bucket.first_upload_started.wait(timeout=5.0)
    assert second_finished.wait(timeout=5.0)
    time.sleep(0.05)
    assert started_splices == [0, 1]
    output = tmp_path / "staging" / "rank00000" / "output"
    assert len(list(output.glob(f"shard_*_features{COMPRESSED_EXT}"))) == 2
    assert len(list((output / sfs.PARQUET_DIRNAME).glob("*.parquet"))) == 2
    assert len(list((output / sfs.PROVENANCE_DIRNAME).glob("*.json"))) == 2
    bucket.allow_first_upload.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert failure == []
    assert bucket.max_active_downloads == 1
    assert bucket.max_active_uploads == 1


def test_shard_failure_cancels_retrying_prefetch_without_waiting_for_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryingPrefetchBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.prefetch_attempted = threading.Event()

        def download(self, remote_path: str, local_path: Path) -> None:
            if remote_path.endswith(f"shard_00001_features{COMPRESSED_EXT}"):
                self.prefetch_attempted.set()
                raise OSError("injected transient prefetch failure")
            super().download(remote_path, local_path)

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = RetryingPrefetchBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    cfg.remote.retry_timeout_seconds = 30.0
    cfg.remote.retry_interval_seconds = 30.0

    def fail_first_job(job, *_args, **_kwargs):
        assert job.shard_index == 0
        assert bucket.prefetch_attempted.wait(timeout=5.0)
        return sfs.SpliceJobResult(
            shard_name=job.shard_name,
            shard_index=job.shard_index,
            status=sfs.SpliceJobStatus.failed,
            n_docs=0,
            n_switches=0,
            error_type="InjectedFailure",
            error_message="deterministic splice failure",
        )

    monkeypatch.setattr(sfs, "_process_job", fail_first_job)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="deterministic splice failure"):
        sfs.run(cfg)
    elapsed = time.perf_counter() - started

    assert bucket.prefetch_attempted.is_set()
    assert elapsed < 2.0
    staged_source = tmp_path / "staging" / "rank00000" / "source"
    assert (
        staged_source / f"shard_00000_features{COMPRESSED_EXT}"
    ).is_file()
    assert bucket.upload_order == []


def test_failed_previous_upload_blocks_current_publication_and_keeps_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailFirstParquetBucket(FakeBucket):
        def upload(self, local_path: Path, remote_path: str) -> None:
            if remote_path.endswith("_parquets/shard_00000.parquet"):
                raise OSError("injected first publication failure")
            super().upload(local_path, remote_path)

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FailFirstParquetBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    cfg.remote.retry_timeout_seconds = 0.01
    cfg.remote.retry_interval_seconds = 0.001
    started: list[int] = []
    original_process = sfs._process_job

    def observe_process(*args, **kwargs):
        started.append(args[0].shard_index)
        return original_process(*args, **kwargs)

    monkeypatch.setattr(sfs, "_process_job", observe_process)

    with pytest.raises(RemoteTrainingError, match="timed out"):
        sfs.run(cfg)

    assert started == [0, 1]
    assert not any("shard_00001" in path for path in bucket.upload_order)
    output = tmp_path / "staging" / "rank00000" / "output"
    assert (output / f"shard_00001_features{COMPRESSED_EXT}").is_file()
    assert (output / sfs.PARQUET_DIRNAME / "shard_00001.parquet").is_file()
    assert (output / sfs.PROVENANCE_DIRNAME / "shard_00001.json").is_file()


def test_simultaneous_publication_and_splice_failures_both_reach_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailFirstParquetBucket(FakeBucket):
        def upload(self, local_path: Path, remote_path: str) -> None:
            if remote_path.endswith("_parquets/shard_00000.parquet"):
                raise OSError("injected first publication failure")
            super().upload(local_path, remote_path)

    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FailFirstParquetBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    profile_path = tmp_path / "double-failure-profile.json"
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
        profile_path=profile_path,
    )
    cfg.remote.retry_timeout_seconds = 0.01
    cfg.remote.retry_interval_seconds = 0.001
    original_process = sfs._process_job

    def fail_second_splice(job, *args, **kwargs):
        if job.shard_index == 1:
            return sfs.SpliceJobResult(
                shard_name=job.shard_name,
                shard_index=job.shard_index,
                status=sfs.SpliceJobStatus.failed,
                n_docs=0,
                n_switches=0,
                error_type="InjectedSpliceFailure",
                error_message="second splice failed",
                profile={
                    "shard_stem": job.shard_name,
                    "shard_index": job.shard_index,
                    "status": "failed",
                    "error_type": "InjectedSpliceFailure",
                    "error_message": "second splice failed",
                },
            )
        return original_process(job, *args, **kwargs)

    monkeypatch.setattr(sfs, "_process_job", fail_second_splice)

    with pytest.raises(RemoteTrainingError, match="timed out"):
        sfs.run(cfg)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert [record["shard_index"] for record in profile["per_shard"]] == [0, 1]
    assert [record["status"] for record in profile["per_shard"]] == ["failed", "failed"]
    assert profile["per_shard"][0]["error_type"] == "RemoteTrainingError"
    assert profile["per_shard"][1]["error_type"] == "InjectedSpliceFailure"
    assert profile["totals"]["n_failed"] == 2
    assert not any("shard_00001" in path for path in bucket.upload_order)


def test_startup_cleans_only_recognized_rank_residue(tmp_path: Path) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    layout = sfs._resolve_remote_layout(cfg)
    rank_dir = tmp_path / "staging" / "rank00000"
    source = rank_dir / "source"
    output = rank_dir / "output"
    verify = rank_dir / "verify"
    parquet = output / sfs.PARQUET_DIRNAME
    provenance = output / sfs.PROVENANCE_DIRNAME
    for directory in (source, output, verify, parquet, provenance):
        directory.mkdir(parents=True, exist_ok=True)
    residue = (
        source / f"shard_00000_features{COMPRESSED_EXT}",
        output / f"shard_00000_features{COMPRESSED_EXT}",
        parquet / "shard_00000.parquet",
        provenance / "shard_00000.json",
        verify / "shard_00000.json",
    )
    for path in residue:
        path.write_bytes(b"recognized runner residue")

    with sfs._locked_rank_staging(layout):
        assert all(not path.exists() for path in residue)


def test_impossible_remote_state_fails_before_download_or_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    bad = bucket._path(f"features/derived/shard_00000_features{COMPRESSED_EXT}")
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"impossible")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )

    with pytest.raises(ValueError, match="impossible remote publication state"):
        sfs.run(cfg)
    assert bucket.download_order == []
    assert bucket.upload_order == []


def test_unknown_staging_file_is_preserved_and_blocks_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    bucket = FakeBucket(tmp_path / "bucket")
    _seed_sources(bucket, feature_dir)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    staging = tmp_path / "staging"
    unknown = staging / "rank00000" / "source" / "do-not-delete.txt"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("owned by someone else", encoding="utf-8")
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=staging,
    )

    with pytest.raises(ValueError, match="unknown file or directory"):
        sfs.run(cfg)
    assert unknown.read_text(encoding="utf-8") == "owned by someone else"
    assert bucket.download_order == []
    assert bucket.upload_order == []


def test_same_rank_staging_lock_contention(tmp_path: Path) -> None:
    root, feature_dir, data_dir = _build_base(tmp_path)
    cfg = _base_cfg(
        local_features_root=root,
        feature_dir=feature_dir,
        data_dir=data_dir,
        output_dir=root / "derived",
        staging_dir=tmp_path / "staging",
    )
    layout = sfs._resolve_remote_layout(cfg)
    with sfs._locked_rank_staging(layout):
        with pytest.raises(ValueError, match="already in use"):
            with sfs._locked_rank_staging(layout):
                pass
