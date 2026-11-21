from __future__ import annotations

import builtins
import json
import os
import shutil
import stat
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import hf_xet
import pytest
import torch as t
from loguru import logger

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from steering.remote_training import (
    REMOTE_CHECKPOINT_MARKER_RE,
    REMOTE_RUN_META_KEY,
    REMOTE_SEMANTIC_META_KEY,
    HfBucketBackend,
    RemoteCheckpointStore,
    RemoteObjectInfo,
    RemoteTrainingConfig,
    RemoteTrainingError,
    build_remote_semantic_fingerprint_v2,
    build_remote_semantic_identity,
    parse_checkpoint_marker,
)
from tests.steering.test_remote_checkpoint_lineage import (
    _config as _v2_config,
)
from tests.steering.test_remote_checkpoint_lineage import (
    _contracts as _v2_contracts,
)
from tests.steering.test_remote_training import (
    FakeBucket,
    PrefixGuardedBucket,
    _generation_remote_path,
    _install_legacy_flat_group,
    _remote_config,
    _require_safe_live_test_bucket,
)


class CountingBucket:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.ops: list[tuple[str, str]] = []

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        self.ops.append(("stat", remote_path))
        return self.inner.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        self.ops.append(("list", prefix))
        return self.inner.list(prefix)

    def download(self, remote_path: str, local_path: Path) -> None:
        self.ops.append(("download", remote_path))
        self.inner.download(remote_path, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.ops.append(("upload", remote_path))
        self.inner.upload(local_path, remote_path)


class RecordingHfApi:
    def __init__(
        self,
        *,
        tree: Sequence[SimpleNamespace] = (),
        path_infos: Sequence[SimpleNamespace] = (),
    ) -> None:
        self.tree = tree
        self.path_infos = path_infos
        self.list_calls: list[tuple[str, str, bool]] = []
        self.paths_info_calls: list[tuple[str, list[str]]] = []

    def list_bucket_tree(
        self,
        bucket_id: str,
        *,
        prefix: str,
        recursive: bool,
    ) -> Sequence[SimpleNamespace]:
        self.list_calls.append((bucket_id, prefix, recursive))
        return self.tree

    def get_bucket_paths_info(
        self,
        bucket_id: str,
        paths: Sequence[str],
    ) -> Sequence[SimpleNamespace]:
        self.paths_info_calls.append((bucket_id, list(paths)))
        return self.path_infos


def test_hf_bucket_backend_list_is_one_tree_call_without_exact_path_lookups() -> None:
    api = RecordingHfApi(
        tree=(
            SimpleNamespace(type="file", path="leaf/z", size=30, xet_hash="c" * 64),
            SimpleNamespace(type="directory", path="leaf/nested", size=0, xet_hash=None),
            SimpleNamespace(type="file", path="leaf/a", size=10, xet_hash="a" * 64),
            SimpleNamespace(type="file", path="leaf/m", size=20, xet_hash="b" * 64),
        ),
    )
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    listing = backend.list("leaf")

    assert api.list_calls == [("owner/bucket", "leaf", True)]
    assert api.paths_info_calls == []
    assert listing == [
        RemoteObjectInfo(path="leaf/a", byte_length=10, xet_hash="a" * 64),
        RemoteObjectInfo(path="leaf/m", byte_length=20, xet_hash="b" * 64),
        RemoteObjectInfo(path="leaf/z", byte_length=30, xet_hash="c" * 64),
    ]


def test_hf_bucket_backend_stat_is_one_exact_lookup_without_tree_call() -> None:
    api = RecordingHfApi(
        path_infos=(SimpleNamespace(path="leaf/object", size=42, xet_hash="d" * 64),),
    )
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    info = backend.stat("leaf/object")

    assert api.paths_info_calls == [("owner/bucket", ["leaf/object"])]
    assert api.list_calls == []
    assert info == RemoteObjectInfo(
        path="leaf/object",
        byte_length=42,
        xet_hash="d" * 64,
    )


def reference_latest_complete_step(store: RemoteCheckpointStore) -> int | None:
    # Frozen verbatim from 94ae9a38:steering/remote_training.py. Keep this
    # independent of the optimized implementation as the equality oracle.
    self = store
    markers: list[tuple[int, str]] = []
    prefix = self.remote_dir + "/"
    for info in self.backend.list(self.remote_dir):
        if not info.path.startswith(prefix):
            raise RemoteTrainingError(f"bucket listing escaped checkpoint prefix: {info.path}")
        relative = info.path[len(prefix) :]
        if "/" in relative:
            continue
        match = REMOTE_CHECKPOINT_MARKER_RE.fullmatch(relative)
        if match is not None:
            markers.append((int(match.group("step")), info.path))
        elif relative.startswith("remote_complete_"):
            raise RemoteTrainingError(
                f"malformed checkpoint completion-marker filename: {info.path}"
            )
    if not markers:
        return None
    parsed_by_step: dict[int, dict[str, Any]] = {}
    checkpoint_lineages: set[str] = set()
    for filename_step, path in sorted(markers):
        marker = self._read_remote_marker(path)
        if marker["step"] != filename_step:
            raise RemoteTrainingError(f"marker filename/payload step mismatch at {path}")
        if marker["ddp_world_size"] != self.world_size:
            raise RemoteTrainingError(f"checkpoint world-size drift at {path}")
        if filename_step in parsed_by_step:
            raise RemoteTrainingError(f"duplicate completion marker for step {filename_step}")
        self._assert_remote_group(marker)
        checkpoint_lineages.add(marker["remote_run_fingerprint"])
        parsed_by_step[filename_step] = marker
    if len(checkpoint_lineages) != 1:
        raise RemoteTrainingError(
            f"checkpoint leaf contains mixed lineages: {sorted(checkpoint_lineages)}"
        )
    latest_step = max(parsed_by_step)
    self._resolve_checkpoint_lineage(parsed_by_step[latest_step])
    if checkpoint_lineages != {self.remote_run_fingerprint}:
        raise RemoteTrainingError("checkpoint lineage resolution did not cover the full leaf")
    return latest_step


@dataclass
class LeafFixture:
    bucket: FakeBucket
    remote: RemoteTrainingConfig
    checkpoint_root: Path
    checkpoint_dir: Path
    remote_run_fingerprint: str
    world_size: int
    semantic_fingerprint_v2: str | None
    resolved_training_config: Mapping[str, Any] | None
    selection_contracts: Mapping[str, Mapping[str, Any]] | None

    def store(
        self,
        *,
        backend: Any | None = None,
        checkpoint_root: Path | None = None,
        world_size: int | None = None,
    ) -> RemoteCheckpointStore:
        root = checkpoint_root or self.checkpoint_root
        return RemoteCheckpointStore(
            backend=backend or self.bucket,
            remote=self.remote,
            checkpoint_dir=root / "model" / "leaf",
            checkpoint_root=root,
            remote_run_fingerprint=self.remote_run_fingerprint,
            world_size=world_size or self.world_size,
            semantic_fingerprint_v2=self.semantic_fingerprint_v2,
            resolved_training_config=self.resolved_training_config,
            selection_contracts=self.selection_contracts,
        )


def _save_group(
    checkpoint_dir: Path,
    *,
    step: int,
    world_size: int,
    lineage: str,
    semantic: str | None = None,
    config: Mapping[str, Any] | None = None,
    weight_offset: int = 0,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "step": step,
        REMOTE_RUN_META_KEY: lineage,
    }
    if config is not None:
        metadata["user_config"] = dict(config)
    if semantic is not None:
        metadata[REMOTE_SEMANTIC_META_KEY] = build_remote_semantic_identity(semantic)
    for rank in range(world_size):
        save_checkpoint(
            str(checkpoint_dir),
            step,
            {"weight": t.arange(6).reshape(2, 3) + step + weight_offset},
            {"state": {rank: {"step": t.tensor(step + rank)}}},
            metadata,
            rank=rank,
        )


def build_leaf(
    tmp_path: Path,
    *,
    steps: Sequence[int],
    world_size: int = 2,
    v2: bool = False,
    fingerprint: str = "d" * 64,
) -> LeafFixture:
    checkpoint_root = tmp_path / "checkpoints-local"
    checkpoint_dir = checkpoint_root / "model" / "leaf"
    bucket = FakeBucket(tmp_path / "bucket")
    remote = _remote_config()
    config: Mapping[str, Any] | None = _v2_config() if v2 else None
    contracts: Mapping[str, Mapping[str, Any]] | None = _v2_contracts() if v2 else None
    semantic: str | None = None
    if config is not None and contracts is not None:
        semantic = build_remote_semantic_fingerprint_v2(
            resolved_training_config=config,
            world_size=world_size,
            selection_contracts=contracts,
        )
    lineage = semantic or fingerprint
    fixture = LeafFixture(
        bucket=bucket,
        remote=remote,
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        remote_run_fingerprint=lineage,
        world_size=world_size,
        semantic_fingerprint_v2=semantic,
        resolved_training_config=config,
        selection_contracts=contracts,
    )
    store = fixture.store()
    for step in steps:
        _save_group(
            checkpoint_dir,
            step=step,
            world_size=world_size,
            lineage=lineage,
            semantic=semantic,
            config=config,
        )
        store.publish(step)
    return fixture


def _marker_path(fixture: LeafFixture, step: int = 10) -> Path:
    return fixture.bucket._path(fixture.store().remote_dir + f"/remote_complete_{step:06d}.json")


def _read_marker(fixture: LeafFixture, step: int = 10) -> dict[str, Any]:
    return json.loads(_marker_path(fixture, step).read_text(encoding="utf-8"))


def _write_leaf_object(fixture: LeafFixture, name: str, data: bytes) -> None:
    path = fixture.bucket._path(fixture.store().remote_dir + f"/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _copy_remote_leaf(source: LeafFixture, target: LeafFixture) -> None:
    source_dir = source.bucket._path(source.store().remote_dir)
    target_dir = target.bucket._path(target.store().remote_dir)
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        destination = target_dir / path.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)


Mutation = Callable[[LeafFixture], None]


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[int, ...] = ()
    mutate: Mutation | None = None
    world_size: int = 2
    v2: bool = False


def _malformed_marker_name(fixture: LeafFixture) -> None:
    _write_leaf_object(fixture, "remote_complete_latest.json", b"{}")


def _junk_marker_json(fixture: LeafFixture) -> None:
    _write_leaf_object(fixture, "remote_complete_000060.json", b"{}")


def _syntactic_junk_marker_json(fixture: LeafFixture) -> None:
    _write_leaf_object(fixture, "remote_complete_000060.json", b"{")


def _step_payload_mismatch(fixture: LeafFixture) -> None:
    shutil.copyfile(_marker_path(fixture, 20), _marker_path(fixture, 30))


def _world_size_drift(fixture: LeafFixture) -> None:
    producer_root = fixture.checkpoint_root
    _save_group(
        fixture.checkpoint_dir,
        step=10,
        world_size=1,
        lineage=fixture.remote_run_fingerprint,
    )
    RemoteCheckpointStore(
        backend=fixture.bucket,
        remote=fixture.remote,
        checkpoint_dir=fixture.checkpoint_dir,
        checkpoint_root=producer_root,
        remote_run_fingerprint=fixture.remote_run_fingerprint,
        world_size=1,
    ).publish(10)


def _duplicate_step_padding(fixture: LeafFixture) -> None:
    shutil.copyfile(
        _marker_path(fixture, 10),
        fixture.bucket._path(fixture.store().remote_dir + "/remote_complete_0000010.json"),
    )


def _mixed_lineage(fixture: LeafFixture) -> None:
    first = build_leaf(
        fixture.checkpoint_root.parent / "first",
        steps=(10,),
        world_size=fixture.world_size,
        fingerprint="d" * 64,
    )
    second = build_leaf(
        fixture.checkpoint_root.parent / "second",
        steps=(20,),
        world_size=fixture.world_size,
        fingerprint="e" * 64,
    )
    _copy_remote_leaf(first, fixture)
    _copy_remote_leaf(second, fixture)


def _generation_entry(
    fixture: LeafFixture,
    *,
    prefix: str,
    step: int = 10,
) -> Path:
    marker = _read_marker(fixture, step)
    entry = next(item for item in marker["files"] if item["name"].startswith(prefix))
    return fixture.bucket._path(_generation_remote_path(fixture.store(), marker, entry["name"]))


def _generation_missing_file(fixture: LeafFixture) -> None:
    _generation_entry(fixture, prefix="optim_").unlink()


def _generation_drift_len(fixture: LeafFixture) -> None:
    path = _generation_entry(fixture, prefix="model_")
    path.write_bytes(path.read_bytes() + b"tamper")


def _generation_drift_xet_same_len(fixture: LeafFixture) -> None:
    path = _generation_entry(fixture, prefix="model_")
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)


def _legacy_flat_ok(fixture: LeafFixture) -> None:
    _install_legacy_flat_group(
        fixture.store(),
        fixture.bucket,
        _read_marker(fixture),
    )


def _legacy_flat_missing(fixture: LeafFixture) -> None:
    _legacy_flat_ok(fixture)
    flat = fixture.bucket._path(fixture.store().remote_dir + "/optim_000010_rank0.pt")
    flat.unlink()


def _partial_generation_with_flat(fixture: LeafFixture) -> None:
    marker = _read_marker(fixture)
    generation_source = _generation_entry(fixture, prefix="model_")
    generation_bytes = generation_source.read_bytes()
    _install_legacy_flat_group(fixture.store(), fixture.bucket, marker)
    generation_source.parent.mkdir(parents=True, exist_ok=True)
    generation_source.write_bytes(generation_bytes)


def _higher_unmarked_debris(fixture: LeafFixture) -> None:
    _write_leaf_object(fixture, "model_000099.pt", b"junk")


def _stale_local_marker(fixture: LeafFixture) -> None:
    (fixture.checkpoint_dir / "remote_complete_000010.json").write_bytes(b"malformed")


def _symlink_local_marker(fixture: LeafFixture) -> None:
    marker = fixture.checkpoint_dir / "remote_complete_000010.json"
    target = fixture.checkpoint_dir / "marker-copy.json"
    shutil.copyfile(marker, target)
    marker.unlink()
    marker.symlink_to(target)


def _v2_tampered_meta(fixture: LeafFixture) -> None:
    path = _generation_entry(fixture, prefix="meta_")
    path.write_bytes(path.read_bytes() + b"tamper")


SCENARIOS = (
    Scenario("empty_leaf"),
    Scenario("single_marker", (10,)),
    Scenario("five_markers", (10, 20, 30, 40, 50)),
    Scenario("malformed_marker_name", mutate=_malformed_marker_name),
    Scenario("junk_marker_json", mutate=_junk_marker_json),
    Scenario("syntactic_junk_marker_json", mutate=_syntactic_junk_marker_json),
    Scenario("step_payload_mismatch", (20,), _step_payload_mismatch),
    Scenario("world_size_drift", mutate=_world_size_drift),
    Scenario("duplicate_step_padding", (10,), _duplicate_step_padding),
    Scenario("mixed_lineage", mutate=_mixed_lineage),
    Scenario("generation_missing_file", (10,), _generation_missing_file),
    Scenario("generation_drift_len", (10,), _generation_drift_len),
    Scenario("generation_drift_xet_same_len", (10,), _generation_drift_xet_same_len),
    Scenario("legacy_flat_ok", (10,), _legacy_flat_ok),
    Scenario("legacy_flat_missing", (10,), _legacy_flat_missing),
    Scenario("partial_generation_with_flat", (10,), _partial_generation_with_flat),
    Scenario("higher_unmarked_debris", (10,), _higher_unmarked_debris),
    Scenario("stale_local_marker", (10,), _stale_local_marker),
    Scenario("symlink_local_marker", (10,), _symlink_local_marker),
    Scenario("v2_direct", (10,), world_size=1, v2=True),
    Scenario("v2_tampered_meta", (10,), _v2_tampered_meta, world_size=1, v2=True),
)


def _build_scenario(tmp_path: Path, scenario: Scenario) -> LeafFixture:
    fixture = build_leaf(
        tmp_path,
        steps=scenario.steps,
        world_size=scenario.world_size,
        v2=scenario.v2,
    )
    if scenario.mutate is not None:
        scenario.mutate(fixture)
    return fixture


def _outcome(call: Callable[[], int | None]) -> tuple[str, Any]:
    try:
        return ("return", call())
    except Exception as exc:
        return ("raise", (type(exc), str(exc)))


def _lineage_state(store: RemoteCheckpointStore) -> tuple[str, str, str | None]:
    return (
        store.remote_run_fingerprint,
        store.compatibility_mode,
        store._resolved_checkpoint_lineage,
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_reference_walk_matches_current_production(tmp_path: Path, scenario: Scenario) -> None:
    reference_fixture = _build_scenario(tmp_path / "reference", scenario)
    production_fixture = build_leaf(
        tmp_path / "production",
        steps=scenario.steps,
        world_size=scenario.world_size,
        v2=scenario.v2,
    )
    reference_store = reference_fixture.store(
        world_size=2 if scenario.name == "world_size_drift" else None
    )
    production_store = production_fixture.store(
        world_size=2 if scenario.name == "world_size_drift" else None
    )
    if scenario.name == "mixed_lineage":
        assert scenario.mutate is not None
        scenario.mutate(production_fixture)
        _outcome(production_store.latest_complete_step)
        production_store = production_fixture.store()
    elif scenario.name in {
        "single_marker",
        "five_markers",
        "generation_drift_len",
        "generation_drift_xet_same_len",
    }:
        production_store.latest_complete_step()
        production_store = production_fixture.store()
        if scenario.mutate is not None:
            scenario.mutate(production_fixture)
    elif scenario.mutate is not None:
        scenario.mutate(production_fixture)

    reference = _outcome(lambda: reference_latest_complete_step(reference_store))
    production = _outcome(production_store.latest_complete_step)

    assert production == reference
    assert _lineage_state(production_store) == _lineage_state(reference_store)
    if scenario.name == "stale_local_marker":
        assert (
            production_fixture.checkpoint_dir / "remote_complete_000010.json"
        ).read_bytes() == b"malformed"
    if scenario.name in {"step_payload_mismatch", "duplicate_step_padding"}:
        adopted_name = (
            "remote_complete_000030.json"
            if scenario.name == "step_payload_mismatch"
            else "remote_complete_0000010.json"
        )
        assert (production_fixture.checkpoint_dir / adopted_name).read_bytes() == (
            production_fixture.bucket._path(
                production_store.remote_dir + f"/{adopted_name}"
            ).read_bytes()
        )


def _published_marker_fixture(
    tmp_path: Path,
) -> tuple[LeafFixture, str, Path, bytes, RemoteObjectInfo]:
    fixture = build_leaf(tmp_path, steps=(10,))
    remote_path = fixture.store().remote_dir + "/remote_complete_000010.json"
    local_path = fixture.checkpoint_dir / "remote_complete_000010.json"
    remote_bytes = fixture.bucket._path(remote_path).read_bytes()
    expected = fixture.bucket.stat(remote_path)
    assert expected is not None
    return fixture, remote_path, local_path, remote_bytes, expected


def test_marker_mirror_candidate_hit_has_no_remote_ops(tmp_path: Path) -> None:
    fixture, remote_path, _, remote_bytes, expected = _published_marker_fixture(tmp_path)
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting)

    marker = store._marker_mirror_candidate(remote_path, expected=expected)

    assert marker == parse_checkpoint_marker(
        json.loads(remote_bytes),
        surface=f"remote checkpoint marker {remote_path}",
    )
    assert counting.ops == []


@pytest.mark.parametrize(
    "mutation",
    ("absent", "symlink", "hardlink", "size", "xet"),
)
def test_marker_mirror_candidate_identity_shortfall_is_miss(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture, remote_path, local_path, _, expected = _published_marker_fixture(tmp_path)
    if mutation == "absent":
        local_path.unlink()
    elif mutation == "symlink":
        target = local_path.with_name("marker-target.json")
        shutil.copyfile(local_path, target)
        local_path.unlink()
        local_path.symlink_to(target)
    elif mutation == "hardlink":
        os.link(local_path, local_path.with_name("marker-hardlink.json"))
    elif mutation == "size":
        local_path.write_bytes(local_path.read_bytes() + b"x")
    else:
        data = bytearray(local_path.read_bytes())
        index = data.index(ord("{")) + 1
        data[index] ^= 1
        local_path.write_bytes(data)
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting)

    assert store._marker_mirror_candidate(remote_path, expected=expected) is None
    assert counting.ops == []


def test_marker_mirror_candidate_remote_identical_invalid_json_uses_remote_surface(
    tmp_path: Path,
) -> None:
    fixture, remote_path, local_path, _, _ = _published_marker_fixture(tmp_path)
    junk = b"{"
    fixture.bucket._path(remote_path).write_bytes(junk)
    local_path.write_bytes(junk)
    expected = fixture.bucket.stat(remote_path)
    assert expected is not None

    with pytest.raises(
        RemoteTrainingError,
        match=f"^remote checkpoint marker {remote_path} is invalid JSON$",
    ):
        fixture.store()._marker_mirror_candidate(remote_path, expected=expected)


def test_marker_mirror_candidate_fifo_misses_without_xet_and_walk_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_leaf(tmp_path, steps=(10,))
    local_path = fixture.checkpoint_dir / "remote_complete_000010.json"
    local_path.unlink()
    os.mkfifo(local_path)
    listing = fixture.bucket.list(fixture.store().remote_dir)
    monkeypatch.setattr(fixture.bucket, "list", lambda _prefix: listing)
    real_hash_files = cast(Any, hf_xet).hash_files

    def reject_mirror_hash(paths: Sequence[str]) -> Any:
        if str(paths[0]).startswith("/proc/"):
            pytest.fail("FIFO mirror classification must not invoke Xet")
        return real_hash_files(paths)

    monkeypatch.setattr(
        "steering.remote_training.hf_xet.hash_files",
        reject_mirror_hash,
    )
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting)

    zero_length_expected = RemoteObjectInfo(
        path=store.remote_dir + "/remote_complete_000010.json",
        byte_length=0,
        xet_hash="0" * 64,
    )
    assert (
        store._marker_mirror_candidate(
            zero_length_expected.path,
            expected=zero_length_expected,
        )
        is None
    )
    assert store.latest_complete_step() == 10
    assert counting.ops == [
        ("list", store.remote_dir),
        ("download", store.remote_dir + "/remote_complete_000010.json"),
    ]
    assert stat.S_ISFIFO(local_path.lstat().st_mode)
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


def test_marker_mirror_candidate_requires_o_nofollow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, remote_path, _, _, expected = _published_marker_fixture(tmp_path)
    monkeypatch.delattr(os, "O_NOFOLLOW")

    with pytest.raises(
        RemoteTrainingError,
        match="^checkpoint marker mirror reads require os.O_NOFOLLOW$",
    ):
        fixture.store()._marker_mirror_candidate(remote_path, expected=expected)


def test_marker_mirror_candidate_wrong_xet_file_size_is_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, remote_path, _, _, expected = _published_marker_fixture(tmp_path)
    monkeypatch.setattr(
        "steering.remote_training.hf_xet.hash_files",
        lambda _paths: [
            SimpleNamespace(
                hash=expected.xet_hash,
                file_size=expected.byte_length + 1,
            )
        ],
    )

    assert fixture.store()._marker_mirror_candidate(remote_path, expected=expected) is None


def test_marker_mirror_candidate_final_fstat_identity_change_is_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, remote_path, _, _, expected = _published_marker_fixture(tmp_path)
    real_fstat = os.fstat
    calls = 0

    def corrupt_final_fstat(fd: int) -> Any:
        nonlocal calls
        calls += 1
        result = real_fstat(fd)
        if calls != 3:
            return result
        return SimpleNamespace(
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns + 1,
            st_ctime_ns=result.st_ctime_ns,
            st_mode=result.st_mode,
            st_nlink=result.st_nlink,
        )

    monkeypatch.setattr(os, "fstat", corrupt_final_fstat)

    assert fixture.store()._marker_mirror_candidate(remote_path, expected=expected) is None
    assert calls == 3


def test_read_leaf_marker_hit_has_no_remote_ops(tmp_path: Path) -> None:
    fixture, remote_path, _, _, expected = _published_marker_fixture(tmp_path)
    counting = CountingBucket(fixture.bucket)

    marker = fixture.store(backend=counting)._read_leaf_marker(remote_path, expected=expected)

    assert marker["step"] == 10
    assert counting.ops == []


def test_read_leaf_marker_miss_adopts_downloaded_bytes(tmp_path: Path) -> None:
    fixture, remote_path, local_path, remote_bytes, expected = _published_marker_fixture(tmp_path)
    local_path.unlink()
    counting = CountingBucket(fixture.bucket)

    marker = fixture.store(backend=counting)._read_leaf_marker(remote_path, expected=expected)

    assert marker["step"] == 10
    assert counting.ops == [("download", remote_path)]
    assert local_path.read_bytes() == remote_bytes
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


def test_read_leaf_marker_atomic_adoption_does_not_clobber_racing_creator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, remote_path, local_path, _, expected = _published_marker_fixture(tmp_path)
    local_path.unlink()
    racing_bytes = b"created-concurrently"
    real_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(racing_bytes)
        real_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)

    marker = fixture.store()._read_leaf_marker(remote_path, expected=expected)

    assert marker["step"] == 10
    assert local_path.read_bytes() == racing_bytes
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


def test_read_leaf_marker_adoption_failure_is_nonfatal_and_cleans_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, remote_path, local_path, _, expected = _published_marker_fixture(tmp_path)
    local_path.unlink()
    sentinel = fixture.checkpoint_dir / "sentinel"
    sentinel.write_bytes(b"untouched")
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    def deny_link(_source: Path, _destination: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "link", deny_link)
    try:
        marker = fixture.store()._read_leaf_marker(remote_path, expected=expected)
    finally:
        logger.remove(sink)

    assert marker["step"] == 10
    assert not local_path.exists()
    assert sentinel.read_bytes() == b"untouched"
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []
    assert any(
        f"remote_training.marker_mirror_adopt_failed path={remote_path}" in message
        and "PermissionError('denied')" in message
        for message in messages
    )


def test_marker_mirror_probe_oserror_falls_back_to_download_with_reference_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(10,))
    production_fixture = build_leaf(tmp_path / "production", steps=(10,))
    real_hash_files = cast(Any, hf_xet).hash_files

    def fail_proc_hash(paths: Sequence[str]) -> Any:
        if str(paths[0]).startswith("/proc/"):
            raise OSError("simulated mirror EIO")
        return real_hash_files(paths)

    monkeypatch.setattr(
        "steering.remote_training.hf_xet.hash_files",
        fail_proc_hash,
    )
    counting = CountingBucket(production_fixture.bucket)
    production_store = production_fixture.store(backend=counting)

    reference = _outcome(lambda: reference_latest_complete_step(reference_fixture.store()))
    production = _outcome(production_store.latest_complete_step)

    assert production == reference == ("return", 10)
    assert counting.ops == [
        ("list", production_store.remote_dir),
        ("download", production_store.remote_dir + "/remote_complete_000010.json"),
    ]
    assert list(production_fixture.checkpoint_dir.glob(".*.read.*")) == []


def test_read_leaf_marker_stale_local_is_not_replaced(tmp_path: Path) -> None:
    fixture, remote_path, local_path, _, expected = _published_marker_fixture(tmp_path)
    local_path.write_bytes(b"stale")
    counting = CountingBucket(fixture.bucket)

    marker = fixture.store(backend=counting)._read_leaf_marker(remote_path, expected=expected)

    assert marker["step"] == 10
    assert counting.ops == [("download", remote_path)]
    assert local_path.read_bytes() == b"stale"
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


def test_read_leaf_marker_download_failure_leaves_no_local_state(tmp_path: Path) -> None:
    fixture, remote_path, local_path, _, expected = _published_marker_fixture(tmp_path)
    local_path.unlink()
    fixture.bucket.download_failures[remote_path] = 1_000_000

    with pytest.raises(RemoteTrainingError, match="timed out"):
        fixture.store()._read_leaf_marker(remote_path, expected=expected)

    assert not local_path.exists()
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


@pytest.mark.parametrize(
    "payload",
    (b"{", b"\xff\xfe{"),
    ids=("invalid-json", "invalid-utf8"),
)
def test_read_leaf_marker_cold_invalid_bytes_fail_without_local_mutation(
    tmp_path: Path,
    payload: bytes,
) -> None:
    fixture, remote_path, local_path, _, _ = _published_marker_fixture(tmp_path)
    fixture.bucket._path(remote_path).write_bytes(payload)
    local_path.unlink()
    expected = fixture.bucket.stat(remote_path)
    assert expected is not None

    with pytest.raises(
        RemoteTrainingError,
        match=f"^remote checkpoint marker {remote_path} is invalid JSON$",
    ):
        fixture.store()._read_leaf_marker(remote_path, expected=expected)

    assert not local_path.exists()
    assert list(fixture.checkpoint_dir.glob(".*.read.*")) == []


GROUP_LISTING_SCENARIOS = tuple(
    scenario
    for scenario in SCENARIOS
    if scenario.name
    in {
        "single_marker",
        "legacy_flat_ok",
        "generation_missing_file",
        "generation_drift_len",
        "generation_drift_xet_same_len",
        "partial_generation_with_flat",
        "legacy_flat_missing",
    }
)


def _return_or_error(call: Callable[[], object]) -> tuple[str, Any]:
    try:
        call()
    except Exception as exc:
        return ("raise", (type(exc), str(exc)))
    return ("return", None)


@pytest.mark.parametrize(
    "scenario",
    GROUP_LISTING_SCENARIOS,
    ids=lambda scenario: scenario.name,
)
def test_listing_group_assertion_matches_remote_resolver(
    tmp_path: Path,
    scenario: Scenario,
) -> None:
    reference_fixture = _build_scenario(tmp_path / "reference", scenario)
    listing_fixture = _build_scenario(tmp_path / "listing", scenario)
    reference_store = reference_fixture.store()
    counting = CountingBucket(listing_fixture.bucket)
    listing_store = listing_fixture.store(backend=counting)
    reference_marker = _read_marker(reference_fixture)
    listing_marker = _read_marker(listing_fixture)
    listing = {info.path: info for info in listing_fixture.bucket.list(listing_store.remote_dir)}

    reference = _return_or_error(lambda: reference_store._resolve_committed_group(reference_marker))
    snapshot = _return_or_error(
        lambda: listing_store._assert_group_in_listing(listing_marker, listing)
    )

    assert snapshot == reference
    assert counting.ops == []


def test_listing_group_assertion_rejects_info_path_identity_drift(
    tmp_path: Path,
) -> None:
    fixture = build_leaf(tmp_path, steps=(10,))
    store = fixture.store()
    marker = _read_marker(fixture)
    listing = {info.path: info for info in fixture.bucket.list(store.remote_dir)}
    entry = marker["files"][0]
    remote_path = _generation_remote_path(store, marker, entry["name"])
    info = listing[remote_path]
    listing[remote_path] = RemoteObjectInfo(
        path=remote_path + ".wrong",
        byte_length=info.byte_length,
        xet_hash=info.xet_hash,
    )

    with pytest.raises(
        RemoteTrainingError,
        match=(
            "^marked checkpoint object identity drift "
            f"layout=immutable_generation path={remote_path}$"
        ),
    ):
        store._assert_group_in_listing(marker, listing)


def _op_counts(ops: Sequence[tuple[str, str]]) -> Counter[str]:
    return Counter(op for op, _ in ops)


@pytest.mark.parametrize("count", (1, 2, 6, 13, 17))
def test_warm_v1_walk_is_exactly_one_listing(tmp_path: Path, count: int) -> None:
    fixture = build_leaf(
        tmp_path,
        steps=tuple(range(10, (count + 1) * 10, 10)),
    )
    counting = CountingBucket(fixture.bucket)

    assert fixture.store(backend=counting).latest_complete_step() == count * 10
    assert counting.ops == [("list", fixture.store().remote_dir)]


@pytest.mark.parametrize("count", (6, 13))
def test_cold_v1_walk_downloads_each_marker_once_then_is_warm(
    tmp_path: Path,
    count: int,
) -> None:
    fixture = build_leaf(
        tmp_path,
        steps=tuple(range(10, (count + 1) * 10, 10)),
    )
    cold_root = tmp_path / "cold" / "checkpoints-local"
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting, checkpoint_root=cold_root)

    assert store.latest_complete_step() == count * 10
    assert counting.ops == [
        ("list", store.remote_dir),
        *[
            ("download", store.remote_dir + f"/remote_complete_{step:06d}.json")
            for step in range(10, (count + 1) * 10, 10)
        ],
    ]
    counting.ops.clear()
    assert store.latest_complete_step() == count * 10
    assert counting.ops == [("list", store.remote_dir)]


def _warm_v2_walk_ops(tmp_path: Path, count: int) -> list[tuple[str, str]]:
    fixture = build_leaf(
        tmp_path,
        steps=tuple(range(10, (count + 1) * 10, 10)),
        world_size=1,
        v2=True,
    )
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting)
    assert store.latest_complete_step() == count * 10
    return counting.ops


def test_warm_v2_walk_has_constant_lineage_resolution_tail(tmp_path: Path) -> None:
    three = _warm_v2_walk_ops(tmp_path / "three", 3)
    nine = _warm_v2_walk_ops(tmp_path / "nine", 9)

    assert _op_counts(three) == Counter({"stat": 3, "list": 1, "download": 1})
    assert _op_counts(nine) == _op_counts(three)


def test_publish_walk_remote_ops_are_independent_of_marker_count(tmp_path: Path) -> None:
    fixture = build_leaf(tmp_path, steps=())
    counting = CountingBucket(fixture.bucket)
    store = fixture.store(backend=counting)
    publish_ops: list[list[tuple[str, str]]] = []
    for step in range(10, 90, 10):
        _save_group(
            fixture.checkpoint_dir,
            step=step,
            world_size=fixture.world_size,
            lineage=fixture.remote_run_fingerprint,
        )
        counting.ops.clear()
        store.publish(step)
        publish_ops.append(list(counting.ops))

    assert _op_counts(publish_ops[2]) == _op_counts(publish_ops[7])


def _cold_restore_ops(tmp_path: Path, count: int) -> list[tuple[str, str]]:
    fixture = build_leaf(
        tmp_path,
        steps=tuple(range(10, (count + 1) * 10, 10)),
    )
    counting = CountingBucket(fixture.bucket)
    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    store = fixture.store(backend=counting, checkpoint_root=fresh_root)
    store.restore(count * 10)
    return counting.ops


def test_restore_remote_stat_and_listing_counts_are_marker_count_independent(
    tmp_path: Path,
) -> None:
    six = _cold_restore_ops(tmp_path / "six", 6)
    twelve = _cold_restore_ops(tmp_path / "twelve", 12)
    six_counts = _op_counts(six)
    twelve_counts = _op_counts(twelve)

    assert six_counts["list"] == twelve_counts["list"]
    assert six_counts["stat"] == twelve_counts["stat"]
    assert twelve_counts["download"] - six_counts["download"] == 6


class AfterListBucket:
    def __init__(self, inner: FakeBucket, after_list: Callable[[], None]) -> None:
        self.inner = inner
        self.after_list = after_list
        self.triggered = False

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        return self.inner.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        snapshot = self.inner.list(prefix)
        if not self.triggered:
            self.triggered = True
            self.after_list()
        return snapshot

    def download(self, remote_path: str, local_path: Path) -> None:
        self.inner.download(remote_path, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.inner.upload(local_path, remote_path)


class ListingTransformBucket:
    def __init__(
        self,
        inner: FakeBucket,
        transform: Callable[
            [builtins.list[RemoteObjectInfo]],
            Sequence[RemoteObjectInfo],
        ],
    ) -> None:
        self.inner = inner
        self.transform = transform

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        return self.inner.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        return self.transform(list(self.inner.list(prefix)))

    def download(self, remote_path: str, local_path: Path) -> None:
        self.inner.download(remote_path, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.inner.upload(local_path, remote_path)


def _install_two_differently_failing_markers(fixture: LeafFixture) -> None:
    shutil.copyfile(_marker_path(fixture, 20), _marker_path(fixture, 30))
    _marker_path(fixture, 20).write_bytes(b"{")


def _local_marker_state(fixture: LeafFixture) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in fixture.checkpoint_dir.glob("remote_complete_*.json")
    }


def test_reversed_listing_preserves_first_error_and_local_mirror_state(
    tmp_path: Path,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(20,))
    optimized_fixture = build_leaf(tmp_path / "optimized", steps=(20,))
    _install_two_differently_failing_markers(reference_fixture)
    _install_two_differently_failing_markers(optimized_fixture)
    reference_before = _local_marker_state(reference_fixture)
    optimized_before = _local_marker_state(optimized_fixture)
    reference_backend = ListingTransformBucket(
        reference_fixture.bucket,
        lambda listing: list(reversed(listing)),
    )
    optimized_backend = ListingTransformBucket(
        optimized_fixture.bucket,
        lambda listing: list(reversed(listing)),
    )

    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_backend))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_backend).latest_complete_step)

    assert optimized == reference
    assert optimized == (
        "raise",
        (
            RemoteTrainingError,
            "remote checkpoint marker "
            f"{optimized_fixture.store().remote_dir}/remote_complete_000020.json "
            "is invalid JSON",
        ),
    )
    assert _local_marker_state(reference_fixture) == reference_before
    assert _local_marker_state(optimized_fixture) == optimized_before
    assert _local_marker_state(optimized_fixture) == _local_marker_state(reference_fixture)


def test_escaped_prefix_listing_fails_before_download_or_adoption(tmp_path: Path) -> None:
    fixture = build_leaf(tmp_path, steps=(10,))
    cold_root = tmp_path / "cold" / "checkpoints-local"
    escaped = RemoteObjectInfo(
        path="outside/checkpoint-leaf/object",
        byte_length=1,
        xet_hash="e" * 64,
    )
    transformed = ListingTransformBucket(
        fixture.bucket,
        lambda listing: [escaped, *listing],
    )
    counting = CountingBucket(transformed)
    store = fixture.store(backend=counting, checkpoint_root=cold_root)

    with pytest.raises(
        RemoteTrainingError,
        match=("^bucket listing escaped checkpoint prefix: outside/checkpoint-leaf/object$"),
    ):
        store.latest_complete_step()

    assert counting.ops == [("list", store.remote_dir)]
    assert not store.checkpoint_dir.exists()


def _duplicate_group_record(
    listing: list[RemoteObjectInfo],
    *,
    conflicting: bool,
) -> Sequence[RemoteObjectInfo]:
    group = next(info for info in listing if "/_groups/" in info.path)
    duplicate = (
        RemoteObjectInfo(
            path=group.path,
            byte_length=group.byte_length,
            xet_hash="0" * 64,
        )
        if conflicting
        else group
    )
    return [*listing, duplicate]


def test_conflicting_duplicate_listing_record_is_precise_delta_d6(
    tmp_path: Path,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(10,))
    optimized_fixture = build_leaf(tmp_path / "optimized", steps=(10,))
    reference_backend = ListingTransformBucket(
        reference_fixture.bucket,
        lambda listing: _duplicate_group_record(listing, conflicting=True),
    )
    optimized_backend = ListingTransformBucket(
        optimized_fixture.bucket,
        lambda listing: _duplicate_group_record(listing, conflicting=True),
    )
    conflict_path = next(
        info.path
        for info in optimized_fixture.bucket.list(optimized_fixture.store().remote_dir)
        if "/_groups/" in info.path
    )

    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_backend))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_backend).latest_complete_step)

    # Delta D6: a conflicting duplicate is an out-of-contract bucket API
    # anomaly. The reference silently ignores it through later exact-path stats.
    assert reference == ("return", 10)
    assert optimized == (
        "raise",
        (
            RemoteTrainingError,
            f"bucket listing repeated a path with conflicting identity: {conflict_path}",
        ),
    )


def test_identical_duplicate_group_listing_record_preserves_equality(
    tmp_path: Path,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(10,))
    optimized_fixture = build_leaf(tmp_path / "optimized", steps=(10,))
    reference_backend = ListingTransformBucket(
        reference_fixture.bucket,
        lambda listing: _duplicate_group_record(listing, conflicting=False),
    )
    optimized_backend = ListingTransformBucket(
        optimized_fixture.bucket,
        lambda listing: _duplicate_group_record(listing, conflicting=False),
    )

    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_backend))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_backend).latest_complete_step)

    assert optimized == reference == ("return", 10)


def _malformed_marker_listing_identity(
    listing: list[RemoteObjectInfo],
) -> Sequence[RemoteObjectInfo]:
    return [
        RemoteObjectInfo(
            path=info.path,
            byte_length=info.byte_length,
            xet_hash="None",
        )
        if PurePosixPath(info.path).name.startswith("remote_complete_")
        else info
        for info in listing
    ]


def test_malformed_marker_listing_xet_is_precise_delta_d7(tmp_path: Path) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(10,))
    optimized_fixture = build_leaf(tmp_path / "optimized", steps=(10,))
    reference_backend = ListingTransformBucket(
        reference_fixture.bucket,
        _malformed_marker_listing_identity,
    )
    optimized_backend = ListingTransformBucket(
        optimized_fixture.bucket,
        _malformed_marker_listing_identity,
    )
    marker_path = optimized_fixture.store().remote_dir + "/remote_complete_000010.json"

    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_backend))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_backend).latest_complete_step)

    # Delta D7: malformed listing identity is an out-of-contract bucket API
    # anomaly. The reference repairs it incidentally through a fresh stat.
    assert reference == ("return", 10)
    assert optimized == (
        "raise",
        (
            RemoteTrainingError,
            "checkpoint marker listing entry has a malformed Xet identity "
            f"(bucket API anomaly): path={marker_path} xet_hash='None'",
        ),
    )


def test_listing_snapshot_pins_group_existence_under_single_writer_contract(
    tmp_path: Path,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference", steps=(10,))
    optimized_fixture = build_leaf(tmp_path / "optimized", steps=(10,))
    reference_target = _generation_entry(reference_fixture, prefix="optim_")
    optimized_target = _generation_entry(optimized_fixture, prefix="optim_")
    reference_bucket = AfterListBucket(
        reference_fixture.bucket,
        reference_target.unlink,
    )
    optimized_bucket = AfterListBucket(
        optimized_fixture.bucket,
        optimized_target.unlink,
    )

    # Delta D2: under the unchanged single-writer contract the recursive listing
    # is one authoritative snapshot. The reference's later stat observes an
    # impossible concurrent deletion; the optimized walk consistently uses T_list.
    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_bucket))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_bucket).latest_complete_step)

    assert reference[0] == "raise"
    assert "marked checkpoint group layout=immutable_generation is missing" in reference[1][1]
    assert optimized == ("return", 10)


def _replacement_leaf(tmp_path: Path) -> LeafFixture:
    fixture = build_leaf(tmp_path, steps=())
    _save_group(
        fixture.checkpoint_dir,
        step=10,
        world_size=fixture.world_size,
        lineage=fixture.remote_run_fingerprint,
        weight_offset=100,
    )
    fixture.store().publish(10)
    return fixture


def _swap_leaf(source: LeafFixture, target: LeafFixture) -> None:
    _copy_remote_leaf(source, target)


def test_listing_identity_rejects_marker_replacement_after_listing(
    tmp_path: Path,
) -> None:
    reference_fixture = build_leaf(tmp_path / "reference-target", steps=(10,))
    reference_replacement = _replacement_leaf(tmp_path / "reference-replacement")
    optimized_fixture = build_leaf(tmp_path / "optimized-target", steps=(10,))
    optimized_replacement = _replacement_leaf(tmp_path / "optimized-replacement")
    (optimized_fixture.checkpoint_dir / "remote_complete_000010.json").unlink()
    reference_bucket = AfterListBucket(
        reference_fixture.bucket,
        lambda: _swap_leaf(reference_replacement, reference_fixture),
    )
    optimized_bucket = AfterListBucket(
        optimized_fixture.bucket,
        lambda: _swap_leaf(optimized_replacement, optimized_fixture),
    )

    # Delta D3: marker replacement after T_list violates marker immutability and
    # the single-writer contract. The pinned listing identity fails loudly,
    # while the reference's later stat accepts the replacement.
    reference = _outcome(
        lambda: reference_latest_complete_step(reference_fixture.store(backend=reference_bucket))
    )
    optimized = _outcome(optimized_fixture.store(backend=optimized_bucket).latest_complete_step)

    assert reference == ("return", 10)
    assert optimized[0] == "raise"
    assert "timed out after 0.2s downloading and verifying" in optimized[1][1]


def test_final_lineage_coverage_postcondition_fails_if_resolution_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_leaf(tmp_path, steps=(10,), fingerprint="d" * 64)
    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    store = RemoteCheckpointStore(
        backend=fixture.bucket,
        remote=fixture.remote,
        checkpoint_dir=fresh_root / "model" / "leaf",
        checkpoint_root=fresh_root,
        remote_run_fingerprint="e" * 64,
        world_size=fixture.world_size,
    )
    monkeypatch.setattr(store, "_resolve_checkpoint_lineage", lambda _marker: None)

    with pytest.raises(
        RemoteTrainingError,
        match="^checkpoint lineage resolution did not cover the full leaf$",
    ):
        store.latest_complete_step()


def _run_publish_crash_resume_lifecycle(
    tmp_path: Path,
    *,
    group_count: int,
) -> Counter[str]:
    fixture = build_leaf(tmp_path, steps=())
    counting = CountingBucket(fixture.bucket)
    publishing_store = fixture.store(backend=counting)
    publish_ops: list[list[tuple[str, str]]] = []
    for step in range(10, (group_count + 1) * 10, 10):
        _save_group(
            fixture.checkpoint_dir,
            step=step,
            world_size=fixture.world_size,
            lineage=fixture.remote_run_fingerprint,
        )
        counting.ops.clear()
        publishing_store.publish(step)
        publish_ops.append(list(counting.ops))

    steady_publish_ops = _op_counts(publish_ops[2])
    assert all(_op_counts(ops) == steady_publish_ops for ops in publish_ops[2:])

    latest = group_count * 10
    marker = _read_marker(fixture, latest)
    expected_group = {
        entry["name"]: (fixture.checkpoint_dir / entry["name"]).read_bytes()
        for entry in marker["files"]
    }
    fresh_root = tmp_path / "crashed-host" / "checkpoints-local"
    counting.ops.clear()
    cold_store = fixture.store(backend=counting, checkpoint_root=fresh_root)
    assert cold_store.latest_complete_step() == latest
    assert cold_store.restore(latest) == marker
    for name, expected in expected_group.items():
        assert (cold_store.checkpoint_dir / name).read_bytes() == expected

    counting.ops.clear()
    warm_store = fixture.store(backend=counting, checkpoint_root=fresh_root)
    assert warm_store.latest_complete_step() == latest
    assert warm_store.restore(latest) == marker
    warm_resume_ops = _op_counts(counting.ops)

    model, optimizer, metadata = load_checkpoint(
        str(warm_store.checkpoint_dir),
        latest,
        "cpu",
        load_optimizer=True,
        rank=1,
    )
    assert t.equal(model["weight"], t.arange(6).reshape(2, 3) + latest)
    assert optimizer["state"][1]["step"].item() == latest + 1
    assert metadata["step"] == latest
    assert metadata[REMOTE_RUN_META_KEY] == fixture.remote_run_fingerprint
    return warm_resume_ops


def test_full_publish_crash_resume_lifecycle_has_constant_remote_ops(
    tmp_path: Path,
) -> None:
    ten_groups = _run_publish_crash_resume_lifecycle(
        tmp_path / "ten-groups",
        group_count=10,
    )
    four_groups = _run_publish_crash_resume_lifecycle(
        tmp_path / "four-groups",
        group_count=4,
    )

    assert ten_groups == four_groups


@pytest.mark.slow
@pytest.mark.api
def test_live_bucket_listing_xet_identity_and_warm_walk(tmp_path: Path) -> None:
    bucket_id = os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET")
    if not bucket_id:
        pytest.skip("set HF_REMOTE_TRAINING_TEST_BUCKET=owner/throwaway-bucket")
    _require_safe_live_test_bucket(
        bucket_id,
        os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET_WRITE_CONFIRM"),
    )
    test_prefix = f"codex-tests/issue-888/{uuid.uuid4().hex}"
    guarded_backend = PrefixGuardedBucket(
        HfBucketBackend(bucket_id),
        allowed_prefix=test_prefix,
    )
    assert guarded_backend.list(test_prefix) == []
    remote = _remote_config(
        bucket_id=bucket_id,
        checkpoint_prefix=f"{test_prefix}/checkpoints",
    )
    checkpoint_root = tmp_path / "publishing-root"
    checkpoint_dir = checkpoint_root / "model" / "leaf"
    lineage = "d" * 64
    _save_group(
        checkpoint_dir,
        step=10,
        world_size=1,
        lineage=lineage,
    )
    publishing_store = RemoteCheckpointStore(
        backend=guarded_backend,
        remote=remote,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root=checkpoint_root,
        remote_run_fingerprint=lineage,
        world_size=1,
    )
    publishing_store.publish(10)

    listing = guarded_backend.list(publishing_store.remote_dir)
    assert len(listing) == 4
    for listed in listing:
        assert listed.xet_hash
        assert listed.xet_hash != "None"
        exact = guarded_backend.stat(listed.path)
        assert exact is not None
        assert listed.path == exact.path
        assert listed.byte_length == exact.byte_length
        assert listed.xet_hash == exact.xet_hash

    fresh_root = tmp_path / "fresh-root"
    counting = CountingBucket(guarded_backend)
    fresh_store = RemoteCheckpointStore(
        backend=counting,
        remote=remote,
        checkpoint_dir=fresh_root / "model" / "leaf",
        checkpoint_root=fresh_root,
        remote_run_fingerprint=lineage,
        world_size=1,
    )
    assert fresh_store.latest_complete_step() == 10
    assert _op_counts(counting.ops)["download"] == 1
    counting.ops.clear()
    assert fresh_store.latest_complete_step() == 10
    assert counting.ops == [("list", fresh_store.remote_dir)]
