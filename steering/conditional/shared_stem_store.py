"""Shared-memory source-stem store for parallel splice preselection (issue #740, PR-F).

Parallel splice workers (`selection_num_workers>1`) each hold their own decompressed copy of a
source feature stem (~4 GB with PR-E+E2), so per-worker RAM — not CPU — caps the worker count.
This store keeps **one physical copy per stem** on a shared-memory filesystem (`/dev/shm`): the
first worker to need a stem materializes its already-decompressed tensors to a file there, and
every worker maps them with ``torch.from_file(shared=True)`` — the OS hands all mappers the same
physical pages. Downstream code holds zero-copy **views** into that shared buffer.

Byte-identical BY CONSTRUCTION: the stored bytes are the same tensors ``load_features`` produced;
mapping them back yields tensors that are ``t.equal`` (and same dtype/shape) to the originals.

This module is the foundational layer (materialize + shared reload of ONE stem). The FeatureIndex
integration, global LRU eviction, and parallel wiring build on top of it.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch as t

# Tensor dtypes that appear in a materialized feature-stem data dict. Stored by name in the
# manifest and mapped back on load — torch.from_file needs an explicit dtype.
_DTYPE_BY_NAME: dict[str, t.dtype] = {
    str(dt): dt
    for dt in (
        t.uint8, t.int8, t.int16, t.uint16, t.int32, t.int64,
        t.float16, t.bfloat16, t.float32, t.float64, t.bool,
    )
}
_DTYPE_SIZE: dict[t.dtype, int] = {
    dt: t.empty((), dtype=dt).element_size() for dt in _DTYPE_BY_NAME.values()
}


class SharedStemStore:
    """Materialize feature-stem tensors to ``/dev/shm`` once; map them back shared across procs.

    A stem is stored as a directory ``<root>/<key>/`` containing one ``<name>.bin`` per tensor
    (raw native-endian bytes) plus ``manifest.json`` (per-tensor dtype/shape + scalar metadata).
    The directory is published with an atomic ``os.rename`` so a racing worker either sees the
    complete stem or re-materializes its own (byte-identical) copy — never a partial read.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        capacity: int | None = None,
        min_free_bytes: int = 0,
    ) -> None:
        """`capacity` caps the number of stems resident on /dev/shm (LRU eviction beyond it);
        None ⇒ unbounded. `min_free_bytes`, if >0, evicts LRU stems until at least that many bytes
        are free before a publish. Eviction is cross-process-safe: the shared filesystem is the
        authority (stem-dir mtime = last use), and `unlink`-while-mapped keeps active readers valid.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.min_free_bytes = min_free_bytes

    def _stem_dir(self, key: str) -> Path:
        return self.root / key

    @contextmanager
    def materialize_lock(self) -> Iterator[None]:
        """Cross-process exclusive lock (flock) serializing the EXPENSIVE decompress+publish of a
        stem. Each worker's materialize transiently holds the decompressed stem (~2× its resident
        size) in private RAM; without this, N workers materializing at once spike memory N× and can
        OOM the box. Serializing costs only the one-time cold path (get()/mmap is never locked)."""
        fd = os.open(str(self.root / ".materialize.lock"), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def has(self, key: str) -> bool:
        return (self._stem_dir(key) / "manifest.json").exists()

    def _touch(self, path: Path) -> None:
        """Mark a stem most-recently-used (best-effort; a concurrent evict may have removed it)."""
        try:
            os.utime(path, None)
        except OSError:
            pass

    def _published_stems(self) -> list[tuple[float, Path]]:
        """(mtime, dir) for every fully-published stem under root, oldest first."""
        out: list[tuple[float, Path]] = []
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            return out
        for e in entries:
            # skip in-flight temp (.tmp-) and being-evicted (.evicting-) dirs
            if not e.name.startswith(".") and (Path(e.path) / "manifest.json").exists():
                try:
                    out.append((e.stat().st_mtime, Path(e.path)))
                except OSError:
                    continue
        out.sort(key=lambda mp: mp[0])
        return out

    def _enforce_limits(self, protect: str | None = None) -> None:
        """Evict LRU stems until under `capacity` and above `min_free_bytes`. `protect` is a stem
        that must not be evicted (the one just published / about to be read). Best-effort and
        idempotent: racing evictors that both remove a dir just no-op on the second unlink."""
        if self.capacity is None and self.min_free_bytes <= 0:
            return
        stems = self._published_stems()
        if self.capacity is not None:
            n_over = len(stems) - self.capacity
            for _mtime, d in stems:
                if n_over <= 0:
                    break
                if d.name == protect:
                    continue
                self._atomic_remove(d)
                n_over -= 1
        if self.min_free_bytes > 0:
            for _mtime, d in self._published_stems():
                try:
                    if shutil.disk_usage(self.root).free >= self.min_free_bytes:
                        break
                except OSError:
                    break
                if d.name == protect:
                    continue
                self._atomic_remove(d)

    def put(self, key: str, tensors: dict[str, t.Tensor], scalars: dict[str, int]) -> bool:
        """Write a stem's tensors + scalar metadata to shared memory (atomic publish).

        Returns True if the stem is published (or already present), False if the tmpfs space guard
        declined to write it — the caller then uses a private copy instead of re-materializing.
        Idempotent: racing writers each build a private temp dir and atomically rename; the loser's
        rename fails and is discarded, leaving the (byte-identical) winner in place.
        """
        final = self._stem_dir(key)
        if self.has(key):
            self._touch(final)
            return True
        # Free room BEFORE writing this stem's bytes (protect it once named).
        self._enforce_limits(protect=key)
        # tmpfs SPACE GUARD (issue #740): writing to a torch.from_file(shared=True) mapping raises
        # SIGBUS (a fatal Bus error, not a catchable exception) if /dev/shm can't back the pages.
        # On tmpfs, evicting a stem a worker still maps does NOT free its pages, so the published-
        # count capacity can undercount real usage. If there isn't room for this stem + a margin,
        # DON'T publish — return the (unpublished) path so the caller falls back to a private RAM
        # materialize. Prevents the crash; degrades to less sharing under /dev/shm pressure.
        stem_bytes = sum(int(x.numel()) * x.element_size() for x in tensors.values())
        margin = max(self.min_free_bytes, 512 * 1024 * 1024)
        # A capacity probe failure is not evidence that tmpfs can back the
        # mapping. Let it propagate rather than risking an uncatchable SIGBUS.
        free = shutil.disk_usage(self.root).free
        if free < stem_bytes + margin:
            return False  # unpublished ⇒ caller uses a private copy (no re-materialize)
        tmp = self.root / f".tmp-{key}-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        manifest: dict[str, object] = {"tensors": {}, "scalars": scalars}
        tmeta: dict[str, object] = {}
        try:
            for name, tsr in tensors.items():
                arr = tsr.detach().contiguous().cpu()
                numel = int(arr.numel())
                # Map a fresh shared file sized to the tensor and copy the bytes in — this writes
                # exactly the layout torch.from_file(shared=True) reads back. Empty tensors carry
                # no data ⇒ no file (get() reconstructs them from the manifest shape).
                if numel:
                    tensor_path = tmp / f"{name}.bin"
                    tensor_bytes = numel * arr.element_size()
                    fd = os.open(
                        tensor_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    try:
                        # Linux tmpfs otherwise allocates lazily and can SIGBUS
                        # during ``copy_`` when it runs out of backing space.
                        # Preallocation turns that fatal signal into a normal,
                        # fail-loud ENOSPC before any mapping is exposed.
                        os.posix_fallocate(fd, 0, tensor_bytes)
                    finally:
                        os.close(fd)
                    dst = t.from_file(
                        str(tensor_path), shared=True, size=numel, dtype=arr.dtype
                    )
                    dst.copy_(arr.reshape(-1))
                tmeta[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
            manifest["tensors"] = tmeta
            (tmp / "manifest.json").write_text(json.dumps(manifest))
            try:
                os.rename(tmp, final)
            except OSError:
                # Only a fully published winner makes a rename failure an
                # idempotent race. Permission, I/O, or other rename failures
                # with no winner must remain loud; returning True there would
                # falsely report an absent stem as published.
                if not self.has(key):
                    raise
                shutil.rmtree(tmp, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._touch(final)
        self._enforce_limits(protect=key)
        return True

    def get(self, key: str) -> dict:
        """Map a published stem back as shared tensors (zero-copy views into shared pages).

        Raises KeyError if the stem is not present. The returned tensors alias the on-``/dev/shm``
        buffer: reading them touches the shared physical pages, not a per-process copy.
        """
        d = self._stem_dir(key)
        manifest_path = d / "manifest.json"
        self._touch(d)  # mark MRU so a concurrent evictor picks a colder stem
        try:
            manifest = json.loads(manifest_path.read_text())
            out: dict = {}
            for name, meta in manifest["tensors"].items():
                dtype = _DTYPE_BY_NAME[meta["dtype"]]
                shape = tuple(meta["shape"])
                numel = 1
                for s in shape:
                    numel *= s
                if numel == 0:
                    out[name] = t.empty(shape, dtype=dtype)  # no file for empty tensors
                    continue
                binp = d / f"{name}.bin"
                # CRITICAL: torch.from_file(shared=True) CREATES a zero-filled file if the path
                # is missing — so an eviction race would silently yield zeros, not an error.
                # Require the file to already exist at exactly the expected size; a missing/short
                # file (stat raises or size mismatches) is a clean miss → caller re-materializes.
                if binp.stat().st_size != numel * _DTYPE_SIZE[dtype]:
                    raise KeyError(
                        f"SharedStemStore: stem {key!r} tensor {name!r} truncated (race)"
                    )
                try:
                    mapped = t.from_file(str(binp), shared=True, size=numel, dtype=dtype)
                except RuntimeError as exc:
                    # torch.from_file raises RuntimeError (NOT FileNotFoundError) when the path is
                    # gone (ENOENT) — the stem was atomically renamed aside (evicted) between the
                    # stat above and this open. Classify by the ERROR MESSAGE, not a second path
                    # lookup: the message reflects what happened AT from_file time, so a peer
                    # republishing the stem before a stat can't mask the miss or expose a real
                    # fault (issue #740 review). Only a genuine ENOENT becomes a clean miss.
                    msg = str(exc).lower()
                    if "no such file" in msg or "os error 2" in msg:
                        raise KeyError(
                            f"SharedStemStore: stem {key!r} evicted mid-read (race)"
                        ) from exc
                    raise
                out[name] = mapped.reshape(shape)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            # Evicted (or mid-publish) between has()/touch and the read — a clean miss; the
            # caller re-materializes. Any process still holding a mapping keeps valid pages.
            raise KeyError(
                f"SharedStemStore: stem {key!r} not present under {self.root}"
            ) from exc
        out.update(manifest["scalars"])
        return out

    def _atomic_remove(self, d: Path) -> None:
        """Delete a published stem dir so a concurrent reader sees it EITHER fully present OR fully
        gone — never mid-delete with some .bin files removed (which torch.from_file would recreate
        as zeros). Rename the whole dir aside atomically, then remove the renamed copy at leisure."""
        aside = d.parent / f".evicting-{os.getpid()}-{d.name}"
        try:
            os.rename(d, aside)
        except OSError:
            return  # already gone, or another evictor renamed it first
        shutil.rmtree(aside, ignore_errors=True)
