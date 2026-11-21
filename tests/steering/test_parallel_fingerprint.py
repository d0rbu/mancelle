# tests/steering/test_parallel_fingerprint.py
"""Correctness of the parallel file fingerprint (issue #740 PR-C startup opt).

`_fingerprint_files` was a single serial SHA-256 stream over ~1.34 TB (a
25-72 min startup cost). It is now per-file SHA-256 (name + streamed bytes)
folded in name-sorted order, hashed concurrently via a thread pool. The VALUE
differs from the old scheme (a one-time migration), but it must be:
- deterministic and INDEPENDENT of thread count / scheduling,
- independent of the input list order (folded in name-sorted order),
- fully content-based: any byte change, or a file rename, changes the digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import steering.conditional.splice_counterfactual as splice_module
from steering.conditional.splice_counterfactual import (
    _DERIVED_FEATURE_FINGERPRINT_CACHE,
    _drop_page_cache,
    _file_fingerprint,
    _fingerprint_files,
    _fingerprint_files_cached,
)
from steering.conditional.splice_provenance import SpliceProvenanceError


def _make_files(tmp_path: Path, n: int) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = tmp_path / f"shard_{i:03d}.bin"
        p.write_bytes(os.urandom(300_000 + i * 777))
        paths.append(p)
    return paths


def _fingerprint_with_cpu_count(paths: list[Path], cpu_count: int | None) -> str:
    """Exercise production worker selection under a controlled CPU count."""
    with patch.object(splice_module.os, "cpu_count", return_value=cpu_count):
        return _fingerprint_files(paths)


def test_parallel_equals_serial_and_deterministic(tmp_path):
    files = _make_files(tmp_path, 16)
    ref = _fingerprint_with_cpu_count(files, 1)
    for cpu_count in (2, 4, 8, None):
        got = _fingerprint_with_cpu_count(files, cpu_count)
        assert got == ref, f"cpu_count={cpu_count} changed the fingerprint"
    # repeated calls are stable
    assert _fingerprint_files(files) == ref


def test_input_order_independent(tmp_path):
    files = _make_files(tmp_path, 10)
    ref = _fingerprint_with_cpu_count(files, 1)
    shuffled = files[:]
    random.Random(0).shuffle(shuffled)
    assert _fingerprint_with_cpu_count(shuffled, 4) == ref


def test_content_change_detected(tmp_path):
    files = _make_files(tmp_path, 6)
    ref = _fingerprint_files(files)
    data = bytearray(files[2].read_bytes())
    data[0] ^= 0xFF
    files[2].write_bytes(bytes(data))
    assert _fingerprint_files(files) != ref


def test_appended_byte_detected(tmp_path):
    files = _make_files(tmp_path, 4)
    ref = _fingerprint_files(files)
    files[1].write_bytes(files[1].read_bytes() + b"\x00")
    assert _fingerprint_files(files) != ref


def test_name_included_in_fingerprint(tmp_path):
    """Two files with identical bytes but different names must differ."""
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    payload = os.urandom(100_000)
    a.write_bytes(payload)
    b.write_bytes(payload)
    assert _file_fingerprint(a) != _file_fingerprint(b)


def test_empty_and_single(tmp_path):
    assert _fingerprint_files([]) == hashlib.sha256().hexdigest()
    (tmp_path / "only.bin").write_bytes(b"hello")
    one = _fingerprint_files([tmp_path / "only.bin"])
    assert len(one) == 64 and one == _fingerprint_files([tmp_path / "only.bin"])


def test_streaming_matches_full_read(tmp_path):
    """The per-file digest must be independent of the streaming chunk size —
    i.e. equal to hashing the whole file at once."""
    p = tmp_path / "big.bin"
    p.write_bytes(os.urandom(_STREAM_TEST_SIZE))
    h = hashlib.sha256()
    h.update(p.name.encode("utf-8"))
    h.update(b"\0")
    h.update(p.read_bytes())
    assert _file_fingerprint(p) == h.hexdigest()


_STREAM_TEST_SIZE = 16 * 1024 * 1024 + 12345  # spans multiple 16 MiB chunks


@pytest.mark.parametrize("n", [0, 1, 2, 33])
def test_various_counts_stable(tmp_path, n):
    files = _make_files(tmp_path, n)
    assert _fingerprint_with_cpu_count(files, 1) == _fingerprint_with_cpu_count(
        files, 8
    )


# --- Regression tests for the Codex review findings ---


def test_duplicate_basename_raises(tmp_path):
    """The fold orders by basename, so a duplicate basename would be
    order-ambiguous — fail loud rather than return an order-dependent value."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1 = tmp_path / "a" / "shard_000.bin"
    p2 = tmp_path / "b" / "shard_000.bin"
    p1.write_bytes(b"x")
    p2.write_bytes(b"y")
    with pytest.raises(ValueError, match="duplicate basename"):
        _fingerprint_files([p1, p2])


def test_file_count_inequality_changes_fingerprint(tmp_path):
    files = _make_files(tmp_path, 5)
    ref = _fingerprint_files(files)
    assert _fingerprint_files(files[:4]) != ref  # dropping a file changes it


def test_actual_rename_changes_fingerprint(tmp_path):
    files = _make_files(tmp_path, 4)
    ref = _fingerprint_files(files)
    renamed = tmp_path / "zzz_renamed.bin"
    files[1].rename(renamed)
    new_set = [renamed if i == 1 else p for i, p in enumerate(files)]
    assert _fingerprint_files(new_set) != ref


def test_missing_file_propagates(tmp_path):
    files = _make_files(tmp_path, 3)
    files[1].unlink()
    with pytest.raises(FileNotFoundError):
        _fingerprint_files(files)


def test_symlink_hashes_target_bytes(tmp_path):
    real = tmp_path / "real.bin"
    payload = os.urandom(50_000)
    real.write_bytes(payload)
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    # symlink is followed: its per-file digest = sha256(link.name + target bytes)
    h = hashlib.sha256()
    h.update(b"link.bin\0")
    h.update(payload)
    assert _file_fingerprint(link) == h.hexdigest()


def test_cpu_count_none_falls_back_to_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "steering.conditional.splice_counterfactual.os.cpu_count", lambda: None
    )
    files = _make_files(tmp_path, 6)
    # max_workers resolves to min(len, 1) == 1 → serial, still correct
    assert _fingerprint_files(files) == _fingerprint_with_cpu_count(files, 1)


def test_fadvise_does_not_change_fingerprint_value(tmp_path):
    """Issue #740: the posix_fadvise(DONTNEED) cache-drop in _file_fingerprint must NOT change the
    hash value (else every existing sel/ manifest would read as stale). The digest must equal a
    plain name+bytes SHA-256 over the exact file contents, at any size incl. multi-chunk."""
    for size in (0, 10, _CHUNK := 16 * 1024 * 1024, _CHUNK * 2 + 123):
        p = tmp_path / f"f_{size}.bin"
        payload = bytes((i * 7) % 256 for i in range(min(size, 4096))) * (size // 4096 + 1)
        payload = payload[:size]
        p.write_bytes(payload)
        h = hashlib.sha256()
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(payload)
        assert _file_fingerprint(p) == h.hexdigest(), f"size={size}"


def test_drop_page_cache_is_safe(tmp_path):
    """_drop_page_cache must never corrupt or error on the file; contents stay intact + re-hashable."""
    p = tmp_path / "shard.bin"
    p.write_bytes(b"packed-shard-bytes" * 100000)
    before = _file_fingerprint(p)
    _drop_page_cache(p)  # drop cache
    assert p.read_bytes() == b"packed-shard-bytes" * 100000  # unchanged
    assert _file_fingerprint(p) == before  # identical after cache drop
    _drop_page_cache(tmp_path / "does-not-exist.bin")  # missing file: no error


class TestStatCertifiedFingerprintCache:
    """The cache may save reads, but can never define a fingerprint value."""

    @staticmethod
    def _cached(
        paths: list[Path], cache_dir: Path, *, namespace: str = "test"
    ) -> str:
        return _fingerprint_files_cached(
            paths,
            cache_dir=cache_dir,
            cache_filename=_DERIVED_FEATURE_FINGERPRINT_CACHE,
            namespace=namespace,
        )

    def test_cold_value_equals_reference_and_warm_hit_reads_zero_bodies(
        self, tmp_path, monkeypatch
    ):
        files = _make_files(tmp_path / "inputs", 4)
        cache_dir = tmp_path / "cache"
        reference = _fingerprint_with_cpu_count(files, 1)

        cold = self._cached(files, cache_dir)
        assert cold == reference
        assert (cache_dir / _DERIVED_FEATURE_FINGERPRINT_CACHE).is_file()

        def forbidden_body_read(_path):
            raise AssertionError("warm cache hit read an artifact body")

        monkeypatch.setattr(splice_module, "_file_fingerprint", forbidden_body_read)
        assert self._cached(list(reversed(files)), cache_dir) == reference

    def test_warm_hit_ignores_cluster_client_device_number(
        self, tmp_path, monkeypatch
    ):
        """The same shared file may have a different st_dev on each node."""
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        expected = self._cached(files, cache_dir)
        cache_path = cache_dir / _DERIVED_FEATURE_FINGERPRINT_CACHE
        raw_cache = json.loads(cache_path.read_text())
        assert raw_cache["schema_version"] == 2
        for entry in raw_cache["files"]:
            assert "dev" not in entry["signature"]["lstat"]
            assert "dev" not in entry["signature"]["stat"]

        real_lstat = Path.lstat
        real_stat = Path.stat

        def with_other_device(result):
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_dev=result.st_dev + 10_000,
                st_ino=result.st_ino,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns,
                st_ctime_ns=result.st_ctime_ns,
            )

        def other_device_lstat(path):
            return with_other_device(real_lstat(path))

        def other_device_stat(path, *args, **kwargs):
            return with_other_device(real_stat(path, *args, **kwargs))

        monkeypatch.setattr(Path, "lstat", other_device_lstat)
        monkeypatch.setattr(Path, "stat", other_device_stat)

        def forbidden_body_read(_path):
            raise AssertionError("cross-node cache hit read an artifact body")

        monkeypatch.setattr(splice_module, "_file_fingerprint", forbidden_body_read)
        assert self._cached(files, cache_dir) == expected

    def test_stale_content_recomputes_exact_reference(self, tmp_path, monkeypatch):
        files = _make_files(tmp_path / "inputs", 3)
        cache_dir = tmp_path / "cache"
        old = self._cached(files, cache_dir)
        files[1].write_bytes(files[1].read_bytes() + b"changed")
        reference = _fingerprint_with_cpu_count(files, 1)

        real_hash = _file_fingerprint
        body_reads: list[str] = []

        def recording_hash(path):
            body_reads.append(path.name)
            return real_hash(path)

        monkeypatch.setattr(splice_module, "_file_fingerprint", recording_hash)
        new = self._cached(files, cache_dir)
        assert new != old
        assert new == reference
        assert sorted(body_reads) == sorted(path.name for path in files)

    def test_atomic_same_size_same_mtime_replacement_invalidates(
        self, tmp_path, monkeypatch
    ):
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        old = self._cached(files, cache_dir)
        victim = files[0]
        before = victim.stat()
        replacement = victim.with_suffix(".replacement")
        prior_bytes = victim.read_bytes()
        replacement.write_bytes(bytes([prior_bytes[0] ^ 0xFF]) + prior_bytes[1:])
        os.utime(
            replacement,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        replacement.replace(victim)
        assert victim.stat().st_size == before.st_size
        assert victim.stat().st_mtime_ns == before.st_mtime_ns
        assert victim.stat().st_ino != before.st_ino

        real_hash = _file_fingerprint
        body_reads = 0

        def recording_hash(path):
            nonlocal body_reads
            body_reads += 1
            return real_hash(path)

        monkeypatch.setattr(splice_module, "_file_fingerprint", recording_hash)
        new = self._cached(files, cache_dir)
        assert body_reads == len(files)
        assert new != old

    @pytest.mark.parametrize(
        "corrupt_payload",
        [
            b"not json",
            json.dumps([]).encode(),
            json.dumps({"schema_version": True}).encode(),
        ],
    )
    def test_corrupt_cache_is_a_visible_full_recompute(
        self, tmp_path, monkeypatch, corrupt_payload
    ):
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        expected = self._cached(files, cache_dir)
        cache_path = cache_dir / _DERIVED_FEATURE_FINGERPRINT_CACHE
        cache_path.write_bytes(corrupt_payload)
        real_hash = _file_fingerprint
        body_reads = 0

        def recording_hash(path):
            nonlocal body_reads
            body_reads += 1
            return real_hash(path)

        monkeypatch.setattr(splice_module, "_file_fingerprint", recording_hash)
        assert self._cached(files, cache_dir) == expected
        assert body_reads == len(files)

    def test_namespace_mismatch_cannot_reuse_cache(self, tmp_path, monkeypatch):
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        expected = self._cached(files, cache_dir, namespace="first")
        real_hash = _file_fingerprint
        body_reads = 0

        def recording_hash(path):
            nonlocal body_reads
            body_reads += 1
            return real_hash(path)

        monkeypatch.setattr(splice_module, "_file_fingerprint", recording_hash)
        assert self._cached(files, cache_dir, namespace="second") == expected
        assert body_reads == len(files)

    def test_cache_write_failure_preserves_computed_value(
        self, tmp_path, monkeypatch
    ):
        files = _make_files(tmp_path / "inputs", 2)
        expected = _fingerprint_with_cpu_count(files, 1)

        def fail_write(_payload, _path):
            raise OSError("read-only cache directory")

        monkeypatch.setattr(splice_module, "atomic_json_dump", fail_write)
        assert self._cached(files, tmp_path / "cache") == expected

    def test_mutation_during_hash_fails_without_certifying(self, tmp_path, monkeypatch):
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        real_hash = _file_fingerprint
        mutated = False

        def mutate_after_hash(path):
            nonlocal mutated
            digest = real_hash(path)
            if not mutated:
                path.write_bytes(path.read_bytes() + b"raced")
                mutated = True
            return digest

        monkeypatch.setattr(splice_module, "_file_fingerprint", mutate_after_hash)
        with pytest.raises(SpliceProvenanceError, match="changed while computing"):
            self._cached(files, cache_dir)
        assert not (cache_dir / _DERIVED_FEATURE_FINGERPRINT_CACHE).exists()

    def test_mutation_during_cache_lookup_fails_instead_of_returning_hit(
        self, tmp_path, monkeypatch
    ):
        files = _make_files(tmp_path / "inputs", 2)
        cache_dir = tmp_path / "cache"
        self._cached(files, cache_dir)
        real_load = splice_module._load_file_fingerprint_cache

        def mutate_after_cache_read(*args, **kwargs):
            cached = real_load(*args, **kwargs)
            files[0].write_bytes(files[0].read_bytes() + b"raced-hit")
            return cached

        monkeypatch.setattr(
            splice_module,
            "_load_file_fingerprint_cache",
            mutate_after_cache_read,
        )
        with pytest.raises(SpliceProvenanceError, match="validating cached"):
            self._cached(files, cache_dir)

    def test_symlink_retarget_invalidates_warm_entry(self, tmp_path, monkeypatch):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        target_a = tmp_path / "target_a"
        target_b = tmp_path / "target_b"
        target_a.write_bytes(b"a" * 100)
        target_b.write_bytes(b"b" * 100)
        link = inputs / "shard.bin"
        link.symlink_to(target_a)
        cache_dir = tmp_path / "cache"
        old = self._cached([link], cache_dir)
        link.unlink()
        link.symlink_to(target_b)

        real_hash = _file_fingerprint
        body_reads = 0

        def recording_hash(path):
            nonlocal body_reads
            body_reads += 1
            return real_hash(path)

        monkeypatch.setattr(splice_module, "_file_fingerprint", recording_hash)
        new = self._cached([link], cache_dir)
        assert body_reads == 1
        assert new != old
