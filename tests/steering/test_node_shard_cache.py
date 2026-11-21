"""Unit tests for the budget-capped node-local compressed-shard cache."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger

from steering import node_shard_cache as cache_module
from steering.node_shard_cache import (
    NodeLocalShardCache,
    NodeShardCacheConfig,
    resolve_train_node_cache,
)
from steering.refined_types import NonNegativeInt, PositiveInt


def _cache(root: Path, *, max_bytes: int = 1_000_000, reserve_bytes: int = 0):
    return NodeLocalShardCache(
        NodeShardCacheConfig(
            dir=root,
            max_bytes=PositiveInt.parse(max_bytes),
            reserve_bytes=NonNegativeInt.parse(reserve_bytes),
        )
    )


def _warnings():
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )
    return messages, sink


def _offer(
    cache: NodeLocalShardCache,
    source: Path,
    compressed: bytes | None = None,
) -> None:
    source_stat = os.stat(source)
    cache.offer(
        source,
        source.read_bytes() if compressed is None else compressed,
        source_stat,
    )


def test_fill_hit_roundtrip_and_sidecar_schema(tmp_path: Path) -> None:
    source = tmp_path / "source" / "shard.safetensors.zst"
    source.parent.mkdir()
    source.write_bytes(b"compressed shard bytes")
    cache = _cache(tmp_path / "cache")

    assert cache.read(source) is None
    _offer(cache, source)
    assert cache.read(source) == source.read_bytes()
    assert cache.counters() == {
        "hits": 1,
        "misses": 1,
        "fills": 1,
        "fill_skips": 0,
        "verify_failures": 0,
        "fallbacks": 0,
    }
    payloads = [path for path in cache.dir.iterdir() if not path.name.endswith(".meta.json")]
    assert len(payloads) == 1
    assert payloads[0].name.endswith("_shard.safetensors.zst")
    sidecar = payloads[0].with_name(f"{payloads[0].name}.meta.json")
    assert sidecar.is_file()
    assert '"schema_version":1' in sidecar.read_text()


def test_zero_byte_source_is_cacheable(tmp_path: Path) -> None:
    source = tmp_path / "empty.safetensors.zst"
    source.write_bytes(b"")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source, b"")
    assert cache.read(source) == b""


def test_corrupt_payload_warns_misses_and_refills(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    payload.write_bytes(b"corrupted")
    messages, sink = _warnings()
    try:
        assert cache.read(source) is None
    finally:
        logger.remove(sink)
    assert any("payload verification failed" in message for message in messages)
    assert not payload.exists()
    assert not sidecar.exists()
    _offer(cache, source)
    assert cache.read(source) == b"canonical"
    assert cache.counters()["verify_failures"] == 1


def test_corrupt_sidecar_invalidates_payload_and_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    sidecar.write_text("{not-json", encoding="utf-8")
    assert cache.read_path(source) is None
    assert not payload.exists()
    assert not sidecar.exists()
    assert cache.counters()["verify_failures"] == 1


@pytest.mark.parametrize(
    "case",
    [
        "non_dict_root",
        "missing_key",
        "extra_key",
        "wrong_schema_version",
        "mismatched_source_abspath",
        "malformed_sha256",
        "non_integer_source_size",
        "non_integer_source_mtime_ns",
        "non_integer_byte_length",
    ],
)
def test_sidecar_metadata_corruption_warns_invalidates_and_refills(
    tmp_path: Path,
    case: str,
) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    source_abspath, payload, sidecar = cache._entry_paths(source)
    metadata: object = json.loads(sidecar.read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
    if case == "non_dict_root":
        metadata = []
    elif case == "missing_key":
        metadata.pop("byte_length")
    elif case == "extra_key":
        metadata["extra"] = "rejected"
    elif case == "wrong_schema_version":
        metadata["schema_version"] = 2
    elif case == "mismatched_source_abspath":
        metadata["source_abspath"] = f"{source_abspath}.other"
    elif case == "malformed_sha256":
        metadata["sha256"] = "A" * 64
    else:
        field = case.removeprefix("non_integer_")
        assert field in {"source_size", "source_mtime_ns", "byte_length"}
        metadata[field] = "not-an-integer"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    messages, sink = _warnings()
    try:
        assert cache.read(source) is None
    finally:
        logger.remove(sink)
    assert any(
        "sidecar handling failed" in message
        and "invalid node shard cache sidecar metadata" in message
        for message in messages
    )
    assert not payload.exists()
    assert not sidecar.exists()
    assert cache.counters()["verify_failures"] == 1
    _offer(cache, source)
    assert cache.read(source) == b"canonical"


def test_orphan_sidecar_without_payload_invalidates_misses_and_refills(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    payload.unlink()

    messages, sink = _warnings()
    try:
        assert cache.read(source) is None
    finally:
        logger.remove(sink)
    assert any("payload handling failed" in message for message in messages)
    assert not sidecar.exists()
    _offer(cache, source)
    assert cache.read(source) == b"canonical"


def test_sidecar_stat_oserror_invalidates_misses_and_refills(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    real_stat = Path.stat

    def flaky_stat(path: Path, *args, **kwargs):
        if path == sidecar:
            raise OSError("sidecar stat fault")
        return real_stat(path, *args, **kwargs)

    messages, sink = _warnings()
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "stat", flaky_stat)
            assert cache.read(source) is None
    finally:
        logger.remove(sink)
    assert any("sidecar stat fault" in message for message in messages)
    assert not payload.exists()
    assert not sidecar.exists()
    _offer(cache, source)
    assert cache.read(source) == b"canonical"


@pytest.mark.parametrize("change", ["mtime", "size"])
def test_changed_source_invalidates_and_refills(tmp_path: Path, change: str) -> None:
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"old bytes")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    if change == "mtime":
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    else:
        stat = source.stat()
        source.write_bytes(b"new longer bytes")
        # Preserve the original timestamp so a size-vs-mtime mutation cannot
        # mistake the changed source for the cached identity.
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert cache.read(source) is None
    _offer(cache, source)
    assert cache.read(source) == source.read_bytes()


def test_budget_stop_caches_first_k_without_eviction(tmp_path: Path) -> None:
    sources = []
    for index in range(3):
        source = tmp_path / "sources" / f"shard_{index}.zst"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(bytes([index]) * 1024)
        sources.append(source)
    cache = _cache(tmp_path / "cache", max_bytes=2700)
    for source in sources:
        _offer(cache, source)

    observed = [cache.read(source) or source.read_bytes() for source in sources]
    assert observed == [source.read_bytes() for source in sources]
    assert cache.read(sources[0]) == sources[0].read_bytes()
    assert cache.read(sources[1]) == sources[1].read_bytes()
    assert cache.read(sources[2]) is None
    assert cache.counters()["fills"] == 2
    assert cache.counters()["fill_skips"] == 1


def test_cache_byte_accounting_includes_payload_sidecar_and_live_tmp(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "cache")
    (cache.dir / "payload").write_bytes(b"p" * 3)
    (cache.dir / "payload.meta.json").write_bytes(b"m" * 5)
    (cache.dir / f"payload.tmp.{os.getpid()}.deadbeef").write_bytes(b"t" * 7)
    subdirectory = cache.dir / "nested"
    subdirectory.mkdir()
    (subdirectory / "not-cache-bytes").write_bytes(b"ignored")
    assert cache._cache_bytes() == 15


def test_reserve_stop_uses_free_after_fill(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"1234")
    cache = _cache(tmp_path / "cache", reserve_bytes=100)
    monkeypatch.setattr(
        cache_module.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=25, f_frsize=4),
    )
    _offer(cache, source)
    assert cache.read(source) is None
    assert cache.counters()["fill_skips"] == 1


def test_admit_statvfs_oserror_skips_and_warns(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    monkeypatch.setattr(
        cache_module.os,
        "statvfs",
        lambda _path: (_ for _ in ()).throw(OSError("statvfs fault")),
    )
    messages, sink = _warnings()
    try:
        _offer(cache, source)
    finally:
        logger.remove(sink)
    assert any("admission check failed" in message and "statvfs fault" in message for message in messages)
    assert cache.counters()["fill_skips"] == 1


def test_reserve_fill_tmp_stat_exception_skips_and_returns_none(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    monkeypatch.setattr(
        cache_module.os,
        "stat",
        lambda _path: (_ for _ in ()).throw(OSError("reservation stat fault")),
    )
    messages, sink = _warnings()
    try:
        assert cache.reserve_fill_tmp(source) is None
    finally:
        logger.remove(sink)
    assert any("worker-tee reservation failed" in message for message in messages)
    assert cache.counters()["fill_skips"] == 1


def test_orphan_payload_is_a_miss_and_refill_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _source, payload, sidecar = cache._entry_paths(source)
    payload.write_bytes(b"orphan")
    assert not sidecar.exists()
    assert cache.read(source) is None
    _offer(cache, source)
    _offer(cache, source)
    assert cache.read(source) == b"canonical"


def test_concurrent_same_file_fills_are_consistent(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"same bytes" * 1000)
    cache = _cache(tmp_path / "cache")
    barrier = threading.Barrier(9)
    errors: list[BaseException] = []

    def fill() -> None:
        try:
            barrier.wait()
            _offer(cache, source)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=fill) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert not errors
    assert cache.read(source) == source.read_bytes()


def test_runtime_unwritable_fill_warns_and_bypasses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")

    def deny_write(_data: bytes, _path: Path) -> None:
        raise PermissionError("runtime cache permissions changed")

    monkeypatch.setattr(cache_module, "atomic_write_bytes", deny_write)
    messages, sink = _warnings()
    try:
        _offer(cache, source)
    finally:
        logger.remove(sink)
    assert any("runtime cache permissions changed" in message for message in messages)
    assert cache.read(source) is None


def test_runtime_chmod_unwritable_fill_warns_and_bypasses(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    messages, sink = _warnings()
    cache.dir.chmod(0o500)
    try:
        _offer(cache, source)
    finally:
        cache.dir.chmod(0o700)
        logger.remove(sink)
    assert any("Permission denied" in message for message in messages)
    assert cache.read(source) is None


def test_unlink_read_only_directory_warns_on_non_filenotfound_oserror(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "cache")
    victim = cache.dir / "victim"
    victim.write_bytes(b"keep")
    messages, sink = _warnings()
    cache.dir.chmod(0o500)
    try:
        cache._unlink(victim)
    finally:
        cache.dir.chmod(0o700)
        logger.remove(sink)
    assert victim.is_file()
    assert any("could not unlink" in message and "Permission denied" in message for message in messages)


def test_constructor_mkdir_failure_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    config = NodeShardCacheConfig(
        dir=blocker / "cache",
        max_bytes=PositiveInt.parse(1),
        reserve_bytes=NonNegativeInt.parse(0),
    )
    with pytest.raises(OSError, match="not-a-directory"):
        NodeLocalShardCache(config)


def test_worker_tmp_size_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical bytes")
    cache = _cache(tmp_path / "cache")
    fill = cache.reserve_fill_tmp(source)
    assert fill is not None
    fill.tmp_path.write_bytes(b"short")
    cache.commit_fill_tmp(fill)
    assert cache.read(source) is None
    assert cache.counters()["verify_failures"] == 1


def test_offer_rejects_bytes_that_disagree_with_caller_stat(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical bytes")
    cache = _cache(tmp_path / "cache")
    cache.offer(source, b"short", os.stat(source))
    assert cache.read(source) is None
    assert cache.counters()["fills"] == 0
    assert cache.counters()["verify_failures"] == 1


def test_offer_rejects_source_changed_after_caller_read(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"original")
    before = os.stat(source)
    compressed = source.read_bytes()
    source.write_bytes(b"changed!")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    cache = _cache(tmp_path / "cache")
    cache.offer(source, compressed, before)
    assert cache.read(source) is None
    assert cache.counters()["fills"] == 0
    assert cache.counters()["verify_failures"] == 1


def test_commit_fill_tmp_rejects_source_changed_after_reservation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"original")
    cache = _cache(tmp_path / "cache")
    fill = cache.reserve_fill_tmp(source)
    assert fill is not None
    fill.tmp_path.write_bytes(b"original")
    source.write_bytes(b"changed!")
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, fill.source_mtime_ns + 1_000_000))
    cache.commit_fill_tmp(fill)
    assert cache.read(source) is None
    assert cache.counters()["fills"] == 0
    assert cache.counters()["verify_failures"] == 1


def test_commit_fill_tmp_renames_prepared_inode_without_read_bytes_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical bytes")
    cache = _cache(tmp_path / "cache")
    fill = cache.reserve_fill_tmp(source)
    assert fill is not None
    fill.tmp_path.write_bytes(source.read_bytes())
    prepared_inode = fill.tmp_path.stat().st_ino
    cache.commit_fill_tmp(fill)
    assert fill.payload_path.stat().st_ino == prepared_inode
    assert not fill.tmp_path.exists()
    assert cache.read(source) == source.read_bytes()


def test_commit_fill_tmp_publish_exception_warns_skips_and_unlinks_tmp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical bytes")
    cache = _cache(tmp_path / "cache")
    fill = cache.reserve_fill_tmp(source)
    assert fill is not None
    fill.tmp_path.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        cache,
        "_publish_prepared_tmp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("publish fault")),
    )
    messages, sink = _warnings()
    try:
        cache.commit_fill_tmp(fill)
    finally:
        logger.remove(sink)
    assert any("worker-tee commit failed" in message and "publish fault" in message for message in messages)
    assert cache.counters()["fill_skips"] == 1
    assert not fill.tmp_path.exists()


def test_reserve_fill_tmp_rejects_budget_without_returning_a_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache", max_bytes=source.stat().st_size - 1)
    assert cache.reserve_fill_tmp(source) is None
    assert cache.counters()["fill_skips"] == 1
    assert not list(cache.dir.iterdir())


def test_read_path_verifies_and_invalidates_corrupt_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    assert cache.read_path(source) == payload
    payload.write_bytes(b"corrupt!!")
    assert cache.read_path(source) is None
    assert not payload.exists()
    assert not sidecar.exists()
    assert cache.counters()["verify_failures"] == 1


def test_source_stat_failure_misses_without_invalidating_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.zst"
    source.write_bytes(b"canonical")
    cache = _cache(tmp_path / "cache")
    _offer(cache, source)
    _source, payload, sidecar = cache._entry_paths(source)
    real_stat = cache_module.os.stat

    def flaky_stat(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == os.path.abspath(source):
            raise OSError("transient source stat failure")
        return real_stat(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(cache_module.os, "stat", flaky_stat)
        assert cache.read_path(source) is None
    assert payload.is_file()
    assert sidecar.is_file()
    assert cache.read(source) == b"canonical"


def test_constructor_sweeps_only_proven_or_old_orphan_temporaries(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    live = cache_dir / f"payload.tmp.{os.getpid()}.deadbeef"
    old_live = cache_dir / f"old.tmp.{os.getpid()}.feedface"
    dead = cache_dir / "payload.tmp.999999999.deadbeef"
    malformed_old = cache_dir / "payload.tmp.unparseable"
    malformed_fresh = cache_dir / "payload.tmp.fresh"
    for path in (live, old_live, dead, malformed_old, malformed_fresh):
        path.write_bytes(b"tmp")
    old_ns = time.time_ns() - (25 * 60 * 60 * 1_000_000_000)
    os.utime(old_live, ns=(old_ns, old_ns))
    os.utime(malformed_old, ns=(old_ns, old_ns))

    _cache(cache_dir)

    assert live.is_file()
    assert not old_live.exists()
    assert malformed_fresh.is_file()
    assert not dead.exists()
    assert not malformed_old.exists()


def test_pid_is_alive_permission_generic_and_huge_pid_constructor_survival(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # PID 1 is owned by root on this test host, so the real probe raises
    # PermissionError and must still classify the process as alive.
    assert NodeLocalShardCache._pid_is_alive(1)
    monkeypatch.setattr(
        cache_module.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(OSError("indeterminate liveness")),
    )
    assert NodeLocalShardCache._pid_is_alive(123)
    monkeypatch.undo()

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    huge_pid = "9" * 40
    huge = cache_dir / f"payload.tmp.{huge_pid}.deadbeef"
    huge.write_bytes(b"tmp")
    old_ns = time.time_ns() - (25 * 60 * 60 * 1_000_000_000)
    os.utime(huge, ns=(old_ns, old_ns))
    _cache(cache_dir)
    assert not huge.exists()


def test_sweep_entry_stat_oserror_warns_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    class Entry:
        name = "payload.tmp.unparseable"
        path = str(cache_dir / name)

        def is_file(self, *, follow_symlinks: bool) -> bool:
            assert not follow_symlinks
            return True

        def stat(self, *, follow_symlinks: bool):
            assert not follow_symlinks
            raise OSError("entry stat fault")

    class Scandir:
        def __enter__(self):
            return iter([Entry()])

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(cache_module.os, "scandir", lambda _path: Scandir())
    messages, sink = _warnings()
    try:
        _cache(cache_dir)
    finally:
        logger.remove(sink)
    assert any("could not inspect temporary" in message and "entry stat fault" in message for message in messages)


@pytest.mark.parametrize("failure", ["oserror", "concurrent_filenotfound"])
def test_sweep_unlink_errors_warn_or_continue(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dead = cache_dir / "payload.tmp.999999999.deadbeef"
    dead.write_bytes(b"tmp")
    real_unlink = Path.unlink

    def failing_unlink(path: Path, *args, **kwargs):
        if path == dead:
            if failure == "concurrent_filenotfound":
                real_unlink(path)
                raise FileNotFoundError("concurrent unlink")
            raise OSError("sweep unlink fault")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    messages, sink = _warnings()
    try:
        _cache(cache_dir)
    finally:
        logger.remove(sink)
    if failure == "oserror":
        assert dead.is_file()
        assert any("could not sweep temporary" in message and "sweep unlink fault" in message for message in messages)
    else:
        assert not dead.exists()
        assert not any("could not sweep temporary" in message for message in messages)


def test_fiftieth_event_emits_info_summary(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        level="INFO",
        format="{message}",
    )
    try:
        for _ in range(50):
            cache.record_verify_failure()
    finally:
        logger.remove(sink)
    assert any(
        "node shard cache: hits=0 misses=0 fills=0 fill_skips=0 "
        "verify_failures=50 fallbacks=0" in message
        for message in messages
    )


def test_cache_key_includes_lexical_directory(tmp_path: Path) -> None:
    first = tmp_path / "one" / "same.zst"
    second = tmp_path / "two" / "same.zst"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache = _cache(tmp_path / "cache")
    _offer(cache, first)
    _offer(cache, second)
    assert cache.read(first) == b"first"
    assert cache.read(second) == b"second"
    assert cache._entry_paths(first)[1] != cache._entry_paths(second)[1]


def test_cache_key_preserves_lexical_symlink_identity(tmp_path: Path) -> None:
    target = tmp_path / "target.safetensors.zst"
    target.write_bytes(b"canonical")
    link = tmp_path / "link.safetensors.zst"
    link.symlink_to(target.name)
    cache = _cache(tmp_path / "cache")

    _offer(cache, link)
    link_abspath = os.path.abspath(link)
    link_prefix = hashlib.sha256(link_abspath.encode()).hexdigest()[:16]
    link_payload = cache._entry_paths(link)[1]
    assert link_payload.name.startswith(f"{link_prefix}_link.safetensors.zst")

    _offer(cache, target)
    assert cache._entry_paths(target)[1] != link_payload
    payloads = [
        path
        for path in cache.dir.iterdir()
        if not path.name.endswith(".meta.json")
    ]
    assert len(payloads) == 2


def test_resolve_train_node_cache_boundaries(tmp_path: Path) -> None:
    assert resolve_train_node_cache(None, None, 0) is None
    assert resolve_train_node_cache(None, None, None) is None
    required = "train_node_cache_max_bytes and train_node_cache_reserve_bytes are required"
    with pytest.raises(ValueError, match=required):
        resolve_train_node_cache(tmp_path / "cache", None, 0)
    with pytest.raises(ValueError, match=required):
        resolve_train_node_cache(tmp_path / "cache", 1, None)
    for blank in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="must not be blank or whitespace-only"):
            resolve_train_node_cache(blank, 1, 0)
    resolved = resolve_train_node_cache(tmp_path / "cache", 123, 45)
    assert resolved == NodeShardCacheConfig(
        dir=tmp_path / "cache",
        max_bytes=123,
        reserve_bytes=45,
    )


@pytest.mark.parametrize(
    "args",
    [
        (123, 1, 0),
        ("cache", 0, 0),
        ("cache", True, 0),
        ("cache", 1, -1),
        ("cache", 1, True),
    ],
)
def test_resolve_train_node_cache_beartype_rejections(args: tuple[object, ...]) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="violates type hint"):
        resolve_train_node_cache(*args)
