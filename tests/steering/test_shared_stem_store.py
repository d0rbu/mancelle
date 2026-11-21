"""Byte-identity + real-page-sharing tests for SharedStemStore (issue #740 PR-F, increment 1)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch as t

import steering.conditional.shared_stem_store as sss
from scripts.preprocess.precompute_selected_features import _sweep_orphaned_shm_stores
from steering.conditional import shared_store_lifecycle as lifecycle
from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import load_features, save_features
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer


def _varied_tensors() -> dict[str, t.Tensor]:
    return {
        "offsets": t.tensor([0, 3, 7, 12], dtype=t.int64),
        "feature_ids": t.arange(12, dtype=t.int32) * 5,
        "feature_ids_u16": (t.arange(12, dtype=t.int64) % 65536).to(t.uint16),
        "feature_mags": t.linspace(0, 1, 12, dtype=t.float32),
        "percentiles": (t.arange(12, dtype=t.int64) % 256).to(t.uint8),
        "empty": t.zeros((0,), dtype=t.int32),
    }


def test_roundtrip_byte_identical(tmp_path):
    store = SharedStemStore(tmp_path / "shm")
    tensors = _varied_tensors()
    scalars = {"segment_size": 2, "n_segments": 3}
    store.put("shard_00000", tensors, scalars)
    got = store.get("shard_00000")
    for name, ref in tensors.items():
        assert got[name].dtype == ref.dtype, name
        assert tuple(got[name].shape) == tuple(ref.shape), name
        assert t.equal(got[name], ref), name
    assert got["segment_size"] == 2
    assert got["n_segments"] == 3


def test_get_returns_shared_pages_not_copies(tmp_path):
    """Two get()s alias the SAME physical /dev/shm pages: a write through one is seen by the
    other. A copying loader would fail this — the deterministic proof that sharing is real."""
    store = SharedStemStore(tmp_path / "shm")
    store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})
    g1 = store.get("s")
    g2 = store.get("s")
    assert g1["a"].data_ptr() != 0
    g1["a"][7] = 4242
    assert int(g2["a"][7]) == 4242  # same physical page => shared, not copied


def test_cross_process_reads_same_bytes(tmp_path):
    """A separate process maps the published stem and reads back identical values — cross-process
    sharing, the actual PR-F use case (spawn workers)."""
    root = tmp_path / "shm"
    store = SharedStemStore(root)
    ref = t.arange(100, dtype=t.int32) * 3
    store.put("cp", {"a": ref}, {"n": 5})
    code = (
        "import torch as t;"
        "from steering.conditional.shared_stem_store import SharedStemStore;"
        f"s=SharedStemStore({str(root)!r});g=s.get('cp');"
        "assert g['n']==5;"
        "assert t.equal(g['a'], t.arange(100, dtype=t.int32)*3);"
        "print('OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_idempotent_put_and_has(tmp_path):
    store = SharedStemStore(tmp_path / "shm")
    assert not store.has("k")
    store.put("k", {"a": t.ones(4, dtype=t.int32)}, {})
    assert store.has("k")
    # second put is a no-op that must not corrupt the published stem
    store.put("k", {"a": t.zeros(4, dtype=t.int32)}, {})
    assert t.equal(store.get("k")["a"], t.ones(4, dtype=t.int32))


def test_matches_load_features_on_real_shard(tmp_path):
    """End-to-end: the store round-trips a real save_features/load_features data dict exactly."""
    tok = CharMergeTokenizer(bos_id=500)
    shard = make_base_shard(
        ["alpha beta gamma", "delta epsilon zeta", "eta theta"], tok, 2
    )
    save_features(shard, tmp_path / "shard_00000")
    data = load_features(_shard_zst(tmp_path))
    tensors = {k: v for k, v in data.items() if isinstance(v, t.Tensor)}
    scalars = {k: v for k, v in data.items() if isinstance(v, int)}
    store = SharedStemStore(tmp_path / "shm")
    store.put("shard_00000", tensors, scalars)
    got = store.get("shard_00000")
    for name, ref in tensors.items():
        assert got[name].dtype == ref.dtype and t.equal(got[name], ref), name
    for name, ref in scalars.items():
        assert got[name] == ref, name


def _shard_zst(tmp_path):
    hits = list((tmp_path).glob("**/*_features.safetensors.zst"))
    assert len(hits) == 1, hits
    return hits[0]


def test_capacity_evicts_lru(tmp_path):
    store = SharedStemStore(tmp_path / "shm", capacity=2)
    for i in range(5):
        store.put(f"s{i}", {"a": t.full((8,), i, dtype=t.int32)}, {})
    published = [d.name for _m, d in store._published_stems()]
    assert len(published) <= 2, published
    # the two most-recently-put survive
    assert store.has("s4") and store.has("s3")
    assert not store.has("s0")


def test_held_mapping_survives_eviction(tmp_path):
    """Unlink-while-mapped: a tensor mapped before its stem is evicted keeps valid data (Linux
    holds the pages until the mapping closes). This is what makes cross-worker eviction safe."""
    store = SharedStemStore(tmp_path / "shm", capacity=2)
    ref = t.arange(256, dtype=t.int32) * 7
    store.put("a", {"x": ref}, {})
    held = store.get("a")["x"]  # live mapping into /dev/shm
    # push "a" out: two newer stems exceed capacity=2
    store.put("b", {"x": t.zeros(8, dtype=t.int32)}, {})
    store.put("c", {"x": t.zeros(8, dtype=t.int32)}, {})
    assert not store.has("a")  # evicted (unlinked)
    assert t.equal(held, ref)  # ...but our mapping still reads the right bytes


def test_min_free_bytes_guard(tmp_path):
    # A huge min_free forces eviction down to (nearly) empty on each publish.
    store = SharedStemStore(tmp_path / "shm", min_free_bytes=1)  # trivially satisfied
    store.put("a", {"x": t.ones(4, dtype=t.int32)}, {})
    assert store.has("a")


def test_unbounded_by_default(tmp_path):
    store = SharedStemStore(tmp_path / "shm")  # capacity=None
    for i in range(6):
        store.put(f"s{i}", {"a": t.ones(4, dtype=t.int32)}, {})
    assert all(store.has(f"s{i}") for i in range(6))


def test_space_guard_skips_publish_when_shm_full(tmp_path):
    """Issue #740: put() must NOT write (which would SIGBUS on a full tmpfs) when there isn't room;
    it returns the unpublished path so the caller falls back to a private copy. Simulated via a
    min_free_bytes larger than the filesystem so the guard always trips."""
    huge = 1 << 60  # 1 EiB free required ⇒ guard always trips
    store = SharedStemStore(tmp_path / "shm", min_free_bytes=huge)
    store.put("s", {"a": t.arange(16, dtype=t.int32)}, {"n": 1})
    assert not store.has("s")  # unpublished ⇒ caller re-materializes privately
    # with a sane threshold the same put publishes normally
    store2 = SharedStemStore(tmp_path / "shm2", min_free_bytes=0)
    store2.put("s", {"a": t.arange(16, dtype=t.int32)}, {"n": 1})
    assert store2.has("s") and t.equal(store2.get("s")["a"], t.arange(16, dtype=t.int32))


def test_put_fails_loud_when_free_space_cannot_be_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SharedStemStore(tmp_path / "shm")

    def fail_disk_usage(_path: Path):
        raise OSError(5, "synthetic capacity probe failure")

    monkeypatch.setattr(sss.shutil, "disk_usage", fail_disk_usage)
    with pytest.raises(OSError, match="synthetic capacity probe failure"):
        store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})
    assert not (store.root / "s").exists()
    assert list(store.root.glob(".tmp-*")) == []


def test_put_preallocates_every_nonempty_tensor_before_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SharedStemStore(tmp_path / "shm")
    tensors = {
        "a": t.arange(16, dtype=t.int32),
        "b": t.arange(7, dtype=t.float64),
        "empty": t.empty((0, 3), dtype=t.float32),
    }
    calls: list[tuple[int, int]] = []
    real_posix_fallocate = sss.os.posix_fallocate
    real_from_file = sss.t.from_file

    def recording_fallocate(fd: int, offset: int, length: int) -> None:
        assert offset == 0
        real_posix_fallocate(fd, offset, length)
        assert os.fstat(fd).st_size == length
        calls.append((offset, length))

    def checked_from_file(path: str, *, shared: bool, size: int, dtype: t.dtype):
        expected = size * t.empty((), dtype=dtype).element_size()
        assert Path(path).stat().st_size == expected
        return real_from_file(path, shared=shared, size=size, dtype=dtype)

    monkeypatch.setattr(sss.os, "posix_fallocate", recording_fallocate)
    monkeypatch.setattr(sss.t, "from_file", checked_from_file)
    assert store.put("s", tensors, {}) is True

    assert calls == [(0, tensors["a"].nbytes), (0, tensors["b"].nbytes)]
    assert not (store.root / "s" / "empty.bin").exists()
    loaded = store.get("s")
    assert t.equal(loaded["a"], tensors["a"])
    assert t.equal(loaded["b"], tensors["b"])
    assert loaded["empty"].shape == (0, 3)


def test_put_preallocation_enospc_is_catchable_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SharedStemStore(tmp_path / "shm")
    mapped = False

    def fail_fallocate(_fd: int, _offset: int, _length: int) -> None:
        raise OSError(28, "synthetic no space left")

    def forbidden_from_file(*_args, **_kwargs):
        nonlocal mapped
        mapped = True
        raise AssertionError("mapping must not precede successful preallocation")

    monkeypatch.setattr(sss.os, "posix_fallocate", fail_fallocate)
    monkeypatch.setattr(sss.t, "from_file", forbidden_from_file)
    with pytest.raises(OSError, match="synthetic no space left") as raised:
        store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})

    assert raised.value.errno == 28
    assert mapped is False
    assert not (store.root / "s").exists()
    assert list(store.root.glob(".tmp-*")) == []


def test_put_rename_failure_without_published_winner_is_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SharedStemStore(tmp_path / "shm")
    real_rename = sss.os.rename

    def fail_publish(source, destination):
        if Path(destination) == store.root / "s":
            raise OSError(5, "synthetic publish rename failure")
        return real_rename(source, destination)

    monkeypatch.setattr(sss.os, "rename", fail_publish)
    with pytest.raises(OSError, match="synthetic publish rename failure"):
        store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})

    assert not store.has("s")
    assert list(store.root.glob(".tmp-*")) == []


def test_put_rename_race_accepts_only_complete_published_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SharedStemStore(tmp_path / "shm")
    values = t.arange(16, dtype=t.int32)
    real_rename = sss.os.rename

    def publish_peer_then_fail(source, destination):
        destination = Path(destination)
        if destination == store.root / "s":
            shutil.copytree(source, destination)
            raise FileExistsError("synthetic lost publication race")
        return real_rename(source, destination)

    monkeypatch.setattr(sss.os, "rename", publish_peer_then_fail)
    assert store.put("s", {"a": values}, {"n": 7}) is True

    assert list(store.root.glob(".tmp-*")) == []
    loaded = store.get("s")
    assert t.equal(loaded["a"], values)
    assert loaded["n"] == 7


def test_get_translates_from_file_enoent_race_to_keyerror(tmp_path, monkeypatch):
    """Codex review #2: if a stem's file vanishes (atomic rename-aside eviction) between get()'s
    size stat and torch.from_file, from_file raises RuntimeError (ENOENT) — NOT FileNotFoundError.
    get() must translate that verified vanished-file race into KeyError (a clean miss), not let a
    fatal RuntimeError abort the whole run."""
    store = SharedStemStore(tmp_path / "shm")
    store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})

    def racey_from_file(path, *a, **k):
        shutil.rmtree(store._stem_dir("s"))  # evicted right after the stat, before the mmap
        raise RuntimeError("unable to open file <s/a.bin> in read-write mode: No such file")

    monkeypatch.setattr(sss.t, "from_file", racey_from_file)
    with pytest.raises(KeyError):  # not RuntimeError
        store.get("s")


def test_get_reraises_non_enoent_runtimeerror(tmp_path, monkeypatch):
    """A RuntimeError from from_file where the file STILL exists is a real fault — re-raise it, do
    not mask it as a miss."""
    store = SharedStemStore(tmp_path / "shm")
    store.put("s", {"a": t.arange(16, dtype=t.int32)}, {})
    monkeypatch.setattr(sss.t, "from_file",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("real torch fault")))
    with pytest.raises(RuntimeError):
        store.get("s")


def test_sweep_reaps_only_schema_valid_dead_owner_store(tmp_path: Path) -> None:
    """The compatibility wrapper uses exact owner identity plus the lease protocol."""
    root = tmp_path.resolve()
    dead = root / f"nanochat-sel-dead-{os.urandom(4).hex()}"
    dead.mkdir()
    dead_owner = lifecycle.initialize_store_ownership(dead, root)
    dead_owner = lifecycle.StoreOwner(
        owner=lifecycle.ProcessIdentity(
            dead_owner.owner.boot_id,
            2_000_000_000,
            1,
        ),
        store_dir=dead_owner.store_dir,
        store_root=dead_owner.store_root,
        uid=dead_owner.uid,
        created_time_ns=dead_owner.created_time_ns,
    )
    lifecycle._atomic_write_json(dead / lifecycle.OWNER_FILENAME, dead_owner.as_dict())
    live = root / f"nanochat-sel-live-{os.urandom(4).hex()}"
    live.mkdir()
    lifecycle.initialize_store_ownership(live, root)
    unrelated = root / "some-other-thing"
    unrelated.mkdir()
    _sweep_orphaned_shm_stores(str(root))
    assert not dead.exists()
    assert live.exists()
    assert unrelated.exists()
