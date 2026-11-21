"""Base-shard content-hash cache: byte-identity + invalidation (issue #529).

The cache is opt-in (``cache_dir``); it must return EXACTLY what the streaming SHA-256 returns
over the same on-disk files. These tests pin that equality and every invalidation path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    BASE_SHARD_HASH_CACHE_VERSION,
    _compute_base_hash,
    _load_hash_cache,
    _stream_base_hash,
    compute_base_shard_content_hash,
    save_overlay_shard,
)


def _shard(d: Path, name: str, payload: dict) -> Path:
    return save_overlay_shard(payload, d / name)


def _legacy_hash(paths: list[Path]) -> str:
    """The exact pre-#529 byte stream: sorted by str(path), name + \\0 + bytes + \\0."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda q: str(q)):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def test_stream_matches_legacy_formula(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    b = _shard(tmp_path, "shard_00001", {"x": t.tensor([4, 5, 6])})
    expected = _legacy_hash([a, b])

    assert compute_base_shard_content_hash([a, b]) == expected
    digest, nbytes = _stream_base_hash(sorted([a, b], key=lambda q: str(q)))
    assert digest == expected
    assert nbytes == a.stat().st_size + b.stat().st_size


def test_cache_hit_equals_stream(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    streamed = _compute_base_hash([a], cache_dir=None)
    first = _compute_base_hash([a], cache_dir=tmp_path)  # miss -> writes cache
    second = _compute_base_hash([a], cache_dir=tmp_path)  # hit

    assert first.hexdigest == streamed.hexdigest == second.hexdigest
    assert first.cache_hit is False and second.cache_hit is True
    assert first.bytes_read == a.stat().st_size
    assert second.bytes_read == 0
    assert (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).is_file()


def test_cache_teeth_no_byte_read_on_hit(tmp_path, monkeypatch):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)  # populate

    calls = {"n": 0}
    orig = Path.read_bytes

    def spy(self):
        if self.name.endswith(".safetensors.zst"):
            calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is True
    assert calls["n"] == 0, "cache hit must not read any shard bytes"


def test_cache_invalidation_size(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    h1 = _compute_base_hash([a], cache_dir=tmp_path).hexdigest
    a.unlink()
    _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3, 4, 5, 6, 7])})  # different size
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is False
    assert r.hexdigest == _compute_base_hash([a], cache_dir=None).hexdigest
    assert r.hexdigest != h1


def test_cache_invalidation_mtime_same_size(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    st = a.stat()
    os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # same size, new mtime
    assert _compute_base_hash([a], cache_dir=tmp_path).cache_hit is False


def test_cache_invalidation_atomic_replace(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    # Re-create the SAME content at the same path via unlink+rewrite. Even if the inode is
    # reused, the rewrite happens at a later time → mtime_ns AND ctime_ns change, so the cache
    # MUST miss. Assert this UNCONDITIONALLY (a guarded assert would go vacuous on inode reuse
    # and stop covering the ctime/mtime invalidation path).
    a.unlink()
    _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    assert _compute_base_hash([a], cache_dir=tmp_path).cache_hit is False


def test_cache_invalid_utf8_is_miss(tmp_path):
    # read_text() raises UnicodeDecodeError (a ValueError, NOT OSError) on non-UTF-8 bytes;
    # _load_hash_cache must treat it as a miss, not crash the whole run.
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_bytes(b"\xff\xfe not valid utf-8 \x80")
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is False
    assert r.hexdigest == _compute_base_hash([a], cache_dir=None).hexdigest


def test_cache_wrong_length_digest_is_miss(tmp_path):
    # A structurally-valid cache whose directory_hash is not a 64-char hex string must miss
    # (validated in _load_hash_cache, not by a bare assert that vanishes under `python -O`).
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    p = tmp_path / BASE_SHARD_HASH_CACHE_FILENAME
    data = json.loads(p.read_text())
    data["directory_hash"] = "deadbeef"  # 8 chars, not 64
    p.write_text(json.dumps(data))
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is False
    assert r.hexdigest == _compute_base_hash([a], cache_dir=None).hexdigest


def test_cache_invalidation_add_remove(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1])})
    _compute_base_hash([a], cache_dir=tmp_path)  # cache = {a}
    b = _shard(tmp_path, "shard_00001", {"x": t.tensor([2])})
    assert _compute_base_hash([a, b], cache_dir=tmp_path).cache_hit is False  # added
    assert _compute_base_hash([a], cache_dir=tmp_path).cache_hit is False  # removed


def test_cache_corrupt_json_is_miss(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text("{not json")
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is False
    assert r.hexdigest == _compute_base_hash([a], cache_dir=None).hexdigest


def test_cache_wrong_version_is_miss(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)
    p = tmp_path / BASE_SHARD_HASH_CACHE_FILENAME
    data = json.loads(p.read_text())
    data["version"] = BASE_SHARD_HASH_CACHE_VERSION + 999
    p.write_text(json.dumps(data))
    assert _compute_base_hash([a], cache_dir=tmp_path).cache_hit is False


def test_cache_order_independent(tmp_path):
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1])})
    b = _shard(tmp_path, "shard_00001", {"x": t.tensor([2])})
    h_ab = _compute_base_hash([a, b], cache_dir=tmp_path).hexdigest
    h_ba = _compute_base_hash([b, a], cache_dir=tmp_path).hexdigest
    assert h_ab == h_ba == compute_base_shard_content_hash([a, b])


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    ops=st.lists(
        st.sampled_from(["add", "mutate", "remove", "touch"]), min_size=1, max_size=12
    )
)
def test_cached_equals_streamed_under_mutations(tmp_path_factory, ops):
    """For ANY sequence of add/mutate/remove/touch, the cached hash equals the streamed hash."""
    d = tmp_path_factory.mktemp("cache_prop")
    shards: dict[str, Path] = {}

    def resync():
        paths = list(shards.values())
        cached = _compute_base_hash(paths, cache_dir=d).hexdigest if paths else None
        streamed = _compute_base_hash(paths, cache_dir=None).hexdigest if paths else None
        assert cached == streamed

    for i, op in enumerate(ops):
        if op == "add" or not shards:
            name = f"shard_{len(shards):05d}"
            shards[name] = _shard(d, name, {"x": t.tensor([i, i + 1])})
        elif op == "mutate":
            name = sorted(shards)[0]
            shards[name].unlink()
            shards[name] = _shard(d, name, {"x": t.tensor([i, i, i, i, i])})
        elif op == "remove":
            name = sorted(shards)[0]
            shards.pop(name).unlink()
        elif op == "touch":
            name = sorted(shards)[0]
            stt = shards[name].stat()
            os.utime(shards[name], ns=(stt.st_atime_ns, stt.st_mtime_ns + 1))
        resync()


# ---------------------------------------------------------------------------
# Cache-miss logging (issue #529 follow-up): every cache miss must be logged so a
# silently-ignored cache is visible. _load_hash_cache covers the cache-FILE cases on a severity
# ladder (absent: DEBUG, stale version: INFO, structural corruption: WARNING); _compute_base_hash
# logs the signature-mismatch ("base shards changed") miss at INFO. Captured via the project's
# caplog→loguru bridge (tests/conftest.py), not a hand-rolled sink.
# ---------------------------------------------------------------------------


def _has_log(caplog, level: str, needle: str) -> bool:
    return any(r.levelname == level and needle in r.getMessage() for r in caplog.records)


def test_load_cache_missing_file_logs_debug(tmp_path, caplog):
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "DEBUG", "no cache file")


def test_load_cache_corrupt_json_logs_warning(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text("{not json")
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "WARNING", "corrupt/unreadable")


def test_load_cache_invalid_utf8_logs_warning(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_bytes(b"\xff\xfe\x80 not utf-8")
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "WARNING", "corrupt/unreadable")


def test_load_cache_non_object_logs_warning(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text("[1, 2, 3]")  # valid JSON, not a dict
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "WARNING", "not a JSON object")


def test_load_cache_wrong_version_logs_info(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text(
        json.dumps(
            {
                "version": BASE_SHARD_HASH_CACHE_VERSION + 99,
                "directory_hash": "a" * 64,
                "shards": [],
            }
        )
    )
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "INFO", "version")


def test_load_cache_malformed_digest_logs_warning(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text(
        json.dumps(
            {"version": BASE_SHARD_HASH_CACHE_VERSION, "directory_hash": "deadbeef", "shards": []}
        )
    )
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "WARNING", "malformed directory_hash")


def test_load_cache_non_list_shards_logs_warning(tmp_path, caplog):
    (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).write_text(
        json.dumps(
            {"version": BASE_SHARD_HASH_CACHE_VERSION, "directory_hash": "a" * 64, "shards": "nope"}
        )
    )
    assert _load_hash_cache(tmp_path) is None
    assert _has_log(caplog, "WARNING", "non-list 'shards'")


def test_load_cache_valid_emits_no_warning(tmp_path, caplog):
    # A structurally-valid cache file emits no WARNING/ERROR from _load_hash_cache.
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)  # writes a valid cache
    caplog.clear()  # isolate the _load_hash_cache call under test
    assert _load_hash_cache(tmp_path) is not None
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]


def test_compute_base_hash_sig_mismatch_logs_info(tmp_path, caplog):
    # The common "base shards changed" miss: the cache file loads fine but its signatures no
    # longer match the on-disk shards, so _compute_base_hash logs at INFO before recomputing.
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    _compute_base_hash([a], cache_dir=tmp_path)  # write cache
    a.unlink()
    _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3, 4, 5])})  # same name, new bytes/stat
    caplog.clear()
    r = _compute_base_hash([a], cache_dir=tmp_path)
    assert r.cache_hit is False
    assert _has_log(caplog, "INFO", "base shards changed")


def test_store_cache_write_failure_degrades_gracefully(tmp_path, monkeypatch):
    # A cache WRITE failure (e.g. a read-only feature_dir at load/eval/serving time, which #530's
    # resolve-path cache write can now reach) must NOT crash the caller: the streamed hexdigest is
    # still returned and no cache file is left behind. (#530 best-effort cache write.)
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    expected = _compute_base_hash([a], cache_dir=None).hexdigest

    def _boom(*args, **kwargs):
        raise PermissionError("Read-only file system")

    monkeypatch.setattr("steering.overlay_io.atomic_json_dump", _boom)
    result = _compute_base_hash([a], cache_dir=tmp_path)  # must NOT raise

    assert result.hexdigest == expected
    assert result.cache_hit is False
    assert not (tmp_path / BASE_SHARD_HASH_CACHE_FILENAME).exists()


def test_compute_base_hash_missing_shard_raises(tmp_path):
    # A base shard listed but absent on disk is a hard error, never a silent skip — both the cache
    # path (the is_file pre-check) and the cache_dir=None streaming path surface it as
    # FileNotFoundError so a truncated/incomplete base-shard set can never hash to a bogus digest.
    a = _shard(tmp_path, "shard_00000", {"x": t.tensor([1, 2, 3])})
    missing = tmp_path / "shard_99999.safetensors.zst"  # never created
    with pytest.raises(FileNotFoundError):
        _compute_base_hash([a, missing], cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        _compute_base_hash([a, missing], cache_dir=None)
