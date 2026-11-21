"""Race-minimizing filesystem identity checks for on-disk artifacts."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class PathSafetyError(RuntimeError):
    """A pathname no longer names the regular inode opened by this process."""


@dataclass(frozen=True)
class FileStatSignature:
    """Named stat fields used to detect an observed file-identity change.

    This is an in-memory comparison snapshot, not a content digest.  Byte
    integrity is established separately with SHA-256 where required.
    """

    mode: int
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


def file_stat_signature(value: os.stat_result) -> FileStatSignature:
    """Mode, inode identity, size, and timestamp signature of one stat result.

    Two equal signatures for one path mean no rename, replacement, resize,
    rewrite, or permission change was observed between the two stat calls;
    callers use this to fail loudly on concurrent mutation.
    """
    return FileStatSignature(
        mode=value.st_mode,
        device=value.st_dev,
        inode=value.st_ino,
        size_bytes=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


def paths_alias(left: Path, right: Path) -> bool:
    """Return whether two pathnames resolve to one path or existing inode.

    ``Path.resolve()`` catches lexical aliases and aliases through parent
    symlinks.  Comparing the existing ``(st_dev, st_ino)`` pair additionally
    catches hard links.  A missing endpoint is not an inode alias; publication
    code still uses atomic replacement and role-specific locks after this
    preflight check.

    Unexpected stat failures deliberately propagate so callers can fail closed
    with their domain-specific configuration error.
    """
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True

    identities: list[tuple[int, int]] = []
    for path in (left, right):
        try:
            value = path.stat()
        except FileNotFoundError:
            return False
        identities.append((int(value.st_dev), int(value.st_ino)))
    return identities[0] == identities[1]


def verify_open_regular_file_binding(
    file_descriptor: int,
    path: Path,
) -> os.stat_result:
    """Prove an fd and pathname name one stable regular file.

    Call before and after consuming bytes.  This closes the lstat/open race:
    the descriptor is opened with O_NOFOLLOW by the caller, and every snapshot
    must name the same inode rather than merely a stable followed target.
    """
    try:
        opened_before = os.fstat(file_descriptor)
        pathname = path.lstat()
        opened_after = os.fstat(file_descriptor)
    except OSError as exc:
        raise PathSafetyError(f"cannot inspect {path}: {exc}") from exc
    snapshots = (opened_before, pathname, opened_after)
    if not all(stat.S_ISREG(value.st_mode) for value in snapshots):
        raise PathSafetyError(f"path is not a regular file: {path}")
    identity = (opened_before.st_dev, opened_before.st_ino)
    if identity != (pathname.st_dev, pathname.st_ino) or identity != (
        opened_after.st_dev,
        opened_after.st_ino,
    ):
        raise PathSafetyError(f"path changed while its descriptor was open: {path}")
    if file_stat_signature(opened_before) != file_stat_signature(opened_after):
        raise PathSafetyError(f"path changed while it was being measured: {path}")
    return opened_after


def _close_file_descriptors(file_descriptors: list[int]) -> OSError | None:
    """Attempt every close and return the first failure without masking later ones."""
    first_error: OSError | None = None
    for file_descriptor in reversed(file_descriptors):
        try:
            os.close(file_descriptor)
        except OSError as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"an additional descriptor close failed for fd {file_descriptor}: {exc}"
                )
    return first_error


def _regular_file_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise PathSafetyError("safe artifact reads require os.O_NOFOLLOW support")
    return os.O_RDONLY | os.O_NONBLOCK | no_follow | getattr(os, "O_CLOEXEC", 0)


def _verify_opened_file_type(file_descriptor: int, path: Path) -> None:
    try:
        opened = os.fstat(file_descriptor)
    except OSError as exc:
        raise PathSafetyError(f"cannot inspect safely opened path {path}: {exc}") from exc
    if not stat.S_ISREG(opened.st_mode):
        raise PathSafetyError(f"path is not a regular file: {path}")


def _open_regular_file_unrooted(path: Path) -> int:
    file_descriptor: int | None = None
    try:
        try:
            # This diagnostic check is not relied on for safety: O_NOFOLLOW in
            # the open flags is the enforcement point.  It preserves the more
            # useful final-symlink error used by existing callers.
            if stat.S_ISLNK(path.lstat().st_mode):
                raise PathSafetyError(f"path must not be a symlink: {path}")
            file_descriptor = os.open(path, _regular_file_open_flags())
            _verify_opened_file_type(file_descriptor, path)
        except PathSafetyError:
            raise
        except (OSError, NotImplementedError) as exc:
            raise PathSafetyError(f"cannot safely open {path}: {exc}") from exc
    except BaseException as primary_error:
        cleanup_error = _close_file_descriptors(
            [] if file_descriptor is None else [file_descriptor]
        )
        if cleanup_error is not None:
            primary_error.add_note(
                f"descriptor cleanup also failed while opening {path}: {cleanup_error}"
            )
        raise
    assert file_descriptor is not None
    return file_descriptor


def _open_regular_file_beneath(
    path: Path,
    *,
    root: Path,
    root_identity: tuple[int, int] | None,
) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    path_only = getattr(os, "O_PATH", None)
    if no_follow is None or directory is None or path_only is None:
        raise PathSafetyError(
            "safe rooted artifact reads require os.O_PATH, os.O_DIRECTORY, "
            "and os.O_NOFOLLOW support"
        )
    if not _OPEN_SUPPORTS_DIR_FD:
        raise PathSafetyError("safe rooted artifact reads require os.open dir_fd support")
    if root_identity is None:
        raise PathSafetyError("safe rooted artifact reads require the original root identity")
    if not root.is_absolute() or not path.is_absolute():
        raise PathSafetyError("safe rooted artifact reads require absolute canonical paths")
    try:
        named_root_before = root.lstat()
    except OSError as exc:
        raise PathSafetyError(f"cannot inspect artifact root {root}: {exc}") from exc
    if (
        not stat.S_ISDIR(named_root_before.st_mode)
        or (int(named_root_before.st_dev), int(named_root_before.st_ino)) != root_identity
    ):
        raise PathSafetyError(f"artifact root changed before file open: {root}")
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise PathSafetyError(
            f"artifact path is outside its root: {path} not beneath {root}"
        ) from exc
    if not relative_path.parts or any(part in ("", ".", "..") for part in relative_path.parts):
        raise PathSafetyError(f"artifact path must name a normalized file beneath its root: {path}")

    directory_flags = path_only | directory | no_follow | getattr(os, "O_CLOEXEC", 0)
    directory_fds: list[int] = []
    result_fd: int | None = None
    try:
        try:
            current_fd = os.open(os.sep, directory_flags)
            directory_fds.append(current_fd)
            for component in root.parts[1:]:
                current_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                )
                directory_fds.append(current_fd)

            opened_root = os.fstat(current_fd)
            named_root = root.lstat()
            expected_identity = tuple(int(value) for value in root_identity)
            opened_identity = (int(opened_root.st_dev), int(opened_root.st_ino))
            named_identity = (int(named_root.st_dev), int(named_root.st_ino))
            if (
                not stat.S_ISDIR(opened_root.st_mode)
                or not stat.S_ISDIR(named_root.st_mode)
                or opened_identity != expected_identity
                or named_identity != expected_identity
            ):
                raise PathSafetyError(f"artifact root changed before file open: {root}")

            for component in relative_path.parts[:-1]:
                current_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                )
                directory_fds.append(current_fd)
            result_fd = os.open(
                relative_path.parts[-1],
                _regular_file_open_flags(),
                dir_fd=current_fd,
            )
            _verify_opened_file_type(result_fd, path)
        except PathSafetyError:
            raise
        except (OSError, NotImplementedError) as exc:
            raise PathSafetyError(f"cannot safely open {path} beneath {root}: {exc}") from exc
    except BaseException as primary_error:
        descriptors = [*directory_fds]
        if result_fd is not None:
            descriptors.append(result_fd)
        cleanup_error = _close_file_descriptors(descriptors)
        if cleanup_error is not None:
            primary_error.add_note(
                f"descriptor cleanup also failed while opening {path}: {cleanup_error}"
            )
        raise

    cleanup_error = _close_file_descriptors(directory_fds)
    if cleanup_error is not None:
        assert result_fd is not None
        result_cleanup_error = _close_file_descriptors([result_fd])
        if result_cleanup_error is not None:
            cleanup_error.add_note(
                f"result descriptor cleanup also failed for fd {result_fd}: {result_cleanup_error}"
            )
        raise PathSafetyError(
            f"cannot close artifact directory descriptors while opening {path}"
        ) from cleanup_error
    assert result_fd is not None
    return result_fd


def open_regular_file_no_follow(
    path: Path,
    *,
    root: Path | None = None,
    root_identity: tuple[int, int] | None = None,
) -> int:
    """Open and type-check a regular file without following path symlinks.

    A rooted open walks every component descriptor-relatively from ``/`` and
    binds that walk to a root inode captured before candidate resolution.  It
    therefore cannot be redirected through an intermediate symlink between a
    containment check and the final open.
    """
    if root is None:
        if root_identity is not None:
            raise PathSafetyError("root_identity requires an artifact root")
        return _open_regular_file_unrooted(path)
    return _open_regular_file_beneath(
        path,
        root=root,
        root_identity=root_identity,
    )
