from __future__ import annotations

import os
from pathlib import Path

import pytest

import steering.eval.autointerp_filter.path_safety as path_safety_module
from steering.eval.autointerp_filter.path_safety import (
    FileStatSignature,
    PathSafetyError,
    file_stat_signature,
    open_regular_file_no_follow,
)


def test_file_stat_signature_is_stable_for_untouched_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("{}\n")

    assert file_stat_signature(path.stat()) == file_stat_signature(path.stat())


def test_file_stat_signature_maps_every_documented_field_exactly(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("{}\n")
    observed = path.stat()

    assert file_stat_signature(observed) == FileStatSignature(
        mode=observed.st_mode,
        device=observed.st_dev,
        inode=observed.st_ino,
        size_bytes=observed.st_size,
        mtime_ns=observed.st_mtime_ns,
        ctime_ns=observed.st_ctime_ns,
    )


def test_file_stat_signature_changes_on_permission_change(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("{}\n")
    path.chmod(0o600)
    before = file_stat_signature(path.stat())

    path.chmod(0o400)

    assert file_stat_signature(path.stat()) != before


def test_file_stat_signature_changes_on_size_change(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("{}\n")
    before = file_stat_signature(path.stat())

    with path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    assert file_stat_signature(path.stat()) != before


def test_file_stat_signature_changes_on_atomic_replacement(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("{}\n")
    before = file_stat_signature(path.stat())

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text("{}\n")
    os.replace(replacement, path)

    assert file_stat_signature(path.stat()) != before


def test_rooted_open_rejects_mismatched_original_root_identity(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "artifact"
    path.write_bytes(b"payload")

    with pytest.raises(PathSafetyError, match="artifact root changed"):
        open_regular_file_no_follow(
            path,
            root=root,
            root_identity=(0, 1),
        )


def test_rooted_open_requires_dir_fd_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "artifact"
    path.write_bytes(b"payload")
    monkeypatch.setattr(path_safety_module, "_OPEN_SUPPORTS_DIR_FD", False)

    with pytest.raises(PathSafetyError, match="require os.open dir_fd support"):
        open_regular_file_no_follow(path, root=root)


def test_rooted_open_close_failure_does_not_misclassify_outer_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    path = nested / "artifact"
    path.write_bytes(b"payload")
    root_stat = root.stat()
    real_open = os.open
    real_close = os.close
    opened_fds: list[int] = []
    close_calls: list[int] = []
    injected = False

    def recording_open(
        candidate: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        file_descriptor = real_open(candidate, flags, mode, dir_fd=dir_fd)
        opened_fds.append(file_descriptor)
        return file_descriptor

    def fail_first_close(file_descriptor: int) -> None:
        nonlocal injected
        close_calls.append(file_descriptor)
        real_close(file_descriptor)
        if not injected:
            injected = True
            raise OSError("injected close failure")

    monkeypatch.setattr(path_safety_module.os, "open", recording_open)
    monkeypatch.setattr(path_safety_module.os, "close", fail_first_close)

    try:
        raise ValueError("unrelated outer exception")
    except ValueError:
        with pytest.raises(
            PathSafetyError,
            match="cannot close artifact directory descriptors",
        ):
            open_regular_file_no_follow(
                path,
                root=root,
                root_identity=(root_stat.st_dev, root_stat.st_ino),
            )

    assert injected
    assert set(opened_fds) <= set(close_calls)
    for file_descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(file_descriptor)
