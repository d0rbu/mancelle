"""Crash-safe create-once installation for immutable artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from beartype import beartype

from steering.refined_types import PositiveIntLike


class ImmutableArtifactError(RuntimeError):
    """An existing immutable artifact is unreadable, changing, or different."""


def _replay_matches(path: Path, payload: bytes, *, surface: str) -> bool:
    """True iff the artifact already at ``path`` is exactly ``payload``.

    The original pathname is resolved once, then the streamed hash binds the
    pre-open pathname identity, opened descriptor, read, and post-read pathname
    identity to one concrete regular file. A final following ``stat`` of the
    declared pathname requires it to still name that same concrete file.
    ``not_a_regular_file`` and ``read_failed`` retain the historical
    stable-read message; ``changed_before_open``, ``changed_during_read``, and
    ``changed_after_read`` retain the historical changed-during-read message.

    The ``OSError`` wrapper preserves the contract that every filesystem
    failure becomes :class:`ImmutableArtifactError`. It cannot reclassify a
    mutation-detection raise because ``_on_error`` always returns
    :class:`ImmutableArtifactError`, a ``RuntimeError`` rather than an
    ``OSError`` subclass.

    Intentional consequences of requiring a real regular file are that a FIFO
    fails loudly without blocking and a character device such as ``/dev/null``
    fails loudly rather than comparing as an empty payload. The payload
    contract remains exactly ``bytes``: passing a non-contiguous ``memoryview``
    is a contract violation and may expose ``hashlib.sha256``'s ``BufferError``;
    it is deliberately not copied into a contiguous ``bytes`` allocation.

    Residuals: (a) a declared-path retarget that is reverted before the final
    declared-path ``stat`` is not distinguished, and (b) any change landing
    after that final ``stat`` is outside the function's observations. The
    latter is the inherent check-then-use window, identical to the pre-existing
    implementation's; this function makes no guarantee about what the declared
    path names after it returns.
    """

    def _on_error(reason: str) -> Exception:
        # MUST NOT be an OSError subclass -- see the `except OSError` below.
        if reason in ("not_a_regular_file", "read_failed"):
            return ImmutableArtifactError(f"Could not stably read {surface} at {path}")
        return ImmutableArtifactError(
            f"{surface} changed while it was being read at {path}; use a fresh run tag"
        )

    try:
        resolved = path.resolve(strict=True)
        digest = stream_hash_file_descriptor_bound(resolved, on_error=_on_error)
        declared = path.stat()
    except OSError as exc:
        raise ImmutableArtifactError(f"Could not stably read {surface} at {path}") from exc
    if (declared.st_dev, declared.st_ino) != (
        digest.stat_signature[1],  # index 1 is dev; see _fd_bound_signature
        digest.stat_signature[2],  # index 2 is ino; see _fd_bound_signature
    ):
        raise ImmutableArtifactError(
            f"{surface} changed while it was being read at {path}; use a fresh run tag"
        )
    return (
        digest.byte_length == len(payload) and digest.sha256 == hashlib.sha256(payload).hexdigest()
    )


def install_bytes_once(
    payload: bytes,
    path: Path,
    *,
    surface: str,
) -> bool:
    """Create ``path`` without replacement, or validate an exact replay.

    The final path is installed with a same-filesystem hard link from a fully
    flushed temporary file. A concurrent writer can win the link race, but it
    can never be overwritten and is accepted only when descriptor-bound
    streaming validates that its content exactly matches ``payload``.
    """

    if path.exists():
        if not _replay_matches(path, payload, surface=surface):
            raise ImmutableArtifactError(
                f"{surface} at {path} differs from the current result; "
                "refusing to overwrite it. Use a fresh run tag."
            )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    installed = False
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
            installed = True
        except FileExistsError:
            if not _replay_matches(path, payload, surface=surface):
                raise ImmutableArtifactError(
                    f"A concurrent writer committed different {surface} "
                    f"bytes at {path}; use a fresh run tag"
                ) from None
        if installed:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return installed


def _fd_bound_signature(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """``(mode, dev, ino, size, mtime_ns, ctime_ns)`` from an already-obtained
    ``os.stat_result`` (an ``lstat()`` or ``fstat()`` result).

    This is the shared comparison tuple :func:`stream_hash_file_descriptor_bound`
    evaluates at every checkpoint. Field order mirrors
    ``steering.conditional.feature_clusters._stat_identity``.
    """
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


@dataclass(frozen=True)
class StreamedFileDigest:
    """Result of :func:`stream_hash_file_descriptor_bound`.

    ``stat_signature`` is in-memory provenance only — a ``(mode, dev, ino,
    size, mtime_ns, ctime_ns)`` tuple used for same-process mutation
    detection. Callers must not serialize it into any persisted/semantic
    payload (it identifies a moment in time on one machine, not content).
    """

    path: Path
    sha256: str
    byte_length: int
    stat_signature: tuple[int, int, int, int, int, int]
    payload: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
    )


_DESCRIPTOR_BOUND_ERROR_MESSAGES: dict[str, str] = {
    "not_a_regular_file": "descriptor-bound hash target is not a regular file: {path}",
    "changed_before_open": (
        "descriptor-bound hash target changed before it could be opened: {path}"
    ),
    "read_failed": "could not read descriptor-bound hash target: {path}",
    "changed_during_read": ("descriptor-bound hash target changed while it was being read: {path}"),
    "changed_after_read": "descriptor-bound hash target changed after it was read: {path}",
}


def _descriptor_bound_error(
    reason: str, *, path: Path, on_error: Callable[[str], Exception] | None
) -> Exception:
    if on_error is not None:
        return on_error(reason)
    return ImmutableArtifactError(_DESCRIPTOR_BOUND_ERROR_MESSAGES[reason].format(path=path))


@beartype
def stream_hash_file_descriptor_bound(
    path: str | Path,
    *,
    on_error: Callable[[str], Exception] | None = None,
    chunk_size: PositiveIntLike = 8_388_608,
    retain_payload: bool = False,
) -> StreamedFileDigest:
    """Stream-hash one real regular file with descriptor-bound mutation detection.

    Sequence: ``lstat`` (pre-open identity) -> ``open`` -> ``fstat`` (must
    match the pre-open identity — catches a swap staged between the lstat
    and the open) -> stream-read -> ``fstat`` (must match the just-opened
    identity — catches in-place mutation during the read, through the same
    descriptor) -> ``lstat`` (must match the pre-open identity — catches a
    swap, rename, or symlink substitution staged at the path after the read
    completes).

    Every checkpoint compares against the *same* original ``lstat``, and the
    two mid-read checkpoints are descriptor-bound (``os.fstat(fd)``, immune
    to path-level swaps once the descriptor is open): an equal-size A->B swap
    staged immediately before ``open()`` and restored to A immediately after
    is detected at the very first ``fstat``, because the descriptor is bound
    to B's inode regardless of what the path is later restored to — a
    pathname-only before/after ``stat()`` pair (no fd-binding) would see A
    both times and miss the substitution entirely. This also catches a swap
    of the *parent directory* around ``open()`` (rename the live directory
    aside, rename a staged sibling into its place, open, then rename both
    back) — the pathname resolves to the staged file's inode for the
    duration of the open, which the descriptor-bound ``fstat`` observes even
    though the parent-directory dance leaves the original path's own
    pathname-``lstat`` signature unchanged by the time any pathname-only
    before/after check would run.

    ``retain_payload=True`` returns the exact bytes read from the same open
    descriptor in :attr:`StreamedFileDigest.payload`. The default is ``False``
    so existing streaming-hash callers retain their bounded-memory behavior.
    ``retain_payload`` itself must be a strict ``bool``, enforced at runtime
    by :func:`beartype.beartype`; accepting truthy objects would make an
    accidental large allocation too easy.

    ``chunk_size`` uses :class:`~steering.refined_types.PositiveIntLike`: a
    strict positive ``int`` whose phantom predicate excludes ``bool``. The
    :func:`beartype.beartype` boundary validates it eagerly, before any
    filesystem call, because ``bytes.__bool__`` makes a size-``0`` read loop silently
    indistinguishable from EOF: ``handle.read(0)`` always returns ``b""``
    without touching the underlying file, so a ``chunk_size=0`` call used to
    return ``sha256(b"")``/``byte_length=0`` for an arbitrarily large,
    completely unread, unchanged file. As defense in depth against the same
    failure shape from any other source (e.g. a wrapped or monkeypatched
    ``.read()`` that returns short/empty chunks before genuine EOF), the
    accumulated ``byte_length`` is also checked against the post-read
    ``fstat`` size once the loop ends — a mismatch raises
    ``"changed_during_read"`` even when every stat-signature checkpoint
    otherwise reports no change, mirroring the ``len(payload) !=
    before.st_size`` completeness invariant of a buffered whole-file read.

    Rejects symlinks at the top level: ``lstat`` reports the link itself
    (not the target), so ``S_ISREG`` fails and this raises rather than
    silently dereferencing. Callers that want symlink-transparent hashing
    must resolve the path themselves first (e.g. ``Path.resolve(strict=True)``)
    — the resulting concrete path is then still symlink-substitution-checked
    by this function for the duration of the call.

    ``on_error`` receives a short, stable reason keyword (one of
    ``"not_a_regular_file"``, ``"changed_before_open"``, ``"read_failed"``,
    ``"changed_during_read"``, ``"changed_after_read"``) and must return the
    exception to raise — never raised implicitly — which keeps this module
    free of any dependency on a caller's error hierarchy (e.g.
    ``steering.eval.cluster_contract_identity.ClusterFeatureContractError``).
    Defaults to :class:`ImmutableArtifactError` with a generic message per
    reason. A missing path raises the raw ``FileNotFoundError`` from
    ``lstat`` unwrapped (not routed through ``on_error``), matching this
    repo's convention of letting native filesystem errors surface directly
    for the "doesn't exist at all" case.

    Every ``except OSError`` in this function wraps exactly one individual
    filesystem call (``open``, ``fstat``, or ``read``) and nothing else. In
    particular, the mutation-detection ``raise _descriptor_bound_error(...)``
    call sites are never inside one of those ``except OSError`` regions: if
    ``on_error`` returns an ``OSError`` subclass (a caller is free to, since
    the return type is just ``Exception``), that exception propagates
    unchanged and ``on_error`` is invoked exactly once for it — it is not
    caught by a broader ``except OSError`` and re-classified as a second,
    different reason. The file handle's own close (via the ``with`` block
    below) is deliberately *not* individually re-wrapped in its own
    ``except OSError`` for the same reason: a broad wrapper around the
    whole ``with`` block previously caught the mutation-detection raises
    described above, and a `close()`-time failure is not otherwise
    observed by any test in this suite or its delegating caller.
    """

    resolved = Path(path).expanduser()
    before_result = resolved.lstat()
    if not stat.S_ISREG(before_result.st_mode):
        raise _descriptor_bound_error("not_a_regular_file", path=resolved, on_error=on_error)
    before = _fd_bound_signature(before_result)

    digest = hashlib.sha256()
    byte_length = 0
    payload_chunks: list[bytes] | None = [] if retain_payload else None
    try:
        opened_file = resolved.open("rb")
    except OSError as exc:
        raise _descriptor_bound_error("read_failed", path=resolved, on_error=on_error) from exc

    with opened_file as handle:
        try:
            opened = _fd_bound_signature(os.fstat(handle.fileno()))
        except OSError as exc:
            raise _descriptor_bound_error("read_failed", path=resolved, on_error=on_error) from exc
        if opened != before:
            raise _descriptor_bound_error("changed_before_open", path=resolved, on_error=on_error)

        while True:
            try:
                chunk = handle.read(chunk_size)
            except OSError as exc:
                raise _descriptor_bound_error(
                    "read_failed", path=resolved, on_error=on_error
                ) from exc
            if not chunk:
                break
            byte_length += len(chunk)
            digest.update(chunk)
            if payload_chunks is not None:
                payload_chunks.append(chunk)

        try:
            after_read = _fd_bound_signature(os.fstat(handle.fileno()))
        except OSError as exc:
            raise _descriptor_bound_error("read_failed", path=resolved, on_error=on_error) from exc
        if after_read != opened:
            raise _descriptor_bound_error("changed_during_read", path=resolved, on_error=on_error)
        if byte_length != after_read[3]:  # index 3 is size; see _fd_bound_signature
            raise _descriptor_bound_error("changed_during_read", path=resolved, on_error=on_error)

    try:
        final_result = resolved.lstat()
    except OSError as exc:
        raise _descriptor_bound_error(
            "changed_after_read", path=resolved, on_error=on_error
        ) from exc
    if _fd_bound_signature(final_result) != before:
        raise _descriptor_bound_error("changed_after_read", path=resolved, on_error=on_error)

    return StreamedFileDigest(
        path=resolved,
        sha256=digest.hexdigest(),
        byte_length=byte_length,
        stat_signature=before,
        payload=(None if payload_chunks is None else b"".join(payload_chunks)),
    )


def is_sha256_hex(value: object) -> bool:
    """True iff ``value`` is a lowercase 64-char hex string -- the exact shape
    of every SHA-256 digest this repo produces (``hashlib.sha256().hexdigest()``).

    THE canonical digest-shape check (PR #893 review dedup: six identical
    private ``_is_sha256`` copies previously existed across ``steering/``).
    Import this instead of re-implementing. Lowercase-only is deliberate:
    our digests are always ``hexdigest()`` output, so an uppercase string is
    not one of ours and must not validate.
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


__all__ = [
    "ImmutableArtifactError",
    "StreamedFileDigest",
    "install_bytes_once",
    "is_sha256_hex",
    "stream_hash_file_descriptor_bound",
]
