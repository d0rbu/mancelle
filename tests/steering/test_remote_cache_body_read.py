"""Bare-metal contracts for descriptor-pinned remote cache consumption.

These tests exercise real Linux file descriptors and inode transitions. They
do not replace the materializer, loader, or multiprocess tests: the purpose of
this file is to prove the lowest-level seam independently of those callers.
"""

from __future__ import annotations

import errno
import hashlib
import os
import signal
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

import steering.remote_training as remote_module
from steering.remote_training import RemoteTrainingError


@contextmanager
def _blocking_open_watchdog() -> Iterator[None]:
    """Fail instead of hanging the suite if a special-file open blocks."""

    def fail_blocked_open(signum: int, frame: Any) -> None:
        del signum, frame
        raise RuntimeError("special-file open blocked")

    previous_handler = signal.signal(signal.SIGALRM, fail_blocked_open)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _entry(
    data: bytes,
    path: Path,
    *,
    packed: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "byte_length": len(data),
        "content_sha256": hashlib.sha256(data).hexdigest(),
    }
    if packed:
        digest = hashlib.sha256()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        result["file_fingerprint"] = digest.hexdigest()
    return result


def _install_unchanged_signature_same_length_mutation(
    path: Path,
    mutated: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Mutate descriptor bytes while returning one retained signature twice."""
    real_fstat = os.fstat
    target_info = path.stat()
    target_identity = (target_info.st_dev, target_info.st_ino)
    state: dict[str, Any] = {
        "calls": 0,
        "first": None,
        "second_actual": None,
        "returned": [],
    }

    def mutating_fstat(fd: int) -> os.stat_result:
        actual = real_fstat(fd)
        if (actual.st_dev, actual.st_ino) != target_identity:
            return actual
        state["calls"] += 1
        assert state["calls"] <= 2, (
            "_read_cache_body_bytes must perform exactly two body-descriptor fstat calls"
        )
        if state["calls"] == 1:
            first = actual
            state["first"] = first
            assert len(mutated) == first.st_size
            with path.open("r+b") as handle:
                assert handle.write(mutated) == len(mutated)
                handle.flush()
                os.fsync(handle.fileno())
            state["returned"].append(first)
            return first

        state["second_actual"] = actual
        first = state["first"]
        assert isinstance(first, os.stat_result)
        assert (actual.st_dev, actual.st_ino) == target_identity
        state["returned"].append(first)
        return first

    monkeypatch.setattr(remote_module.os, "fstat", mutating_fstat)
    return state


def test_cache_body_fd_signature_contains_exact_seven_fields() -> None:
    info = cast(
        os.stat_result,
        SimpleNamespace(
            st_mode=11,
            st_dev=22,
            st_ino=33,
            st_size=44,
            st_mtime_ns=55,
            st_ctime_ns=66,
            st_nlink=77,
        ),
    )

    assert remote_module._cache_body_fd_signature(info) == (
        11,
        22,
        33,
        44,
        55,
        66,
        77,
    )


def _assert_original_fd_closed(
    fd: int,
    original_identity: tuple[int, int],
) -> None:
    try:
        current = os.fstat(fd)
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        # Long-lived test processes may reuse a just-closed descriptor
        # immediately. Reuse is still proof that the original open file
        # description was closed, provided it now identifies another inode.
        assert (current.st_dev, current.st_ino) != original_identity


@pytest.mark.parametrize("packed", [False, True])
def test_verify_cache_body_payload_accepts_exact_signed_bytes(
    tmp_path: Path,
    packed: bool,
) -> None:
    path = tmp_path / "shard_00017.safetensors.zst"
    data = bytes(range(256)) * 3

    remote_module._verify_cache_body_payload(
        data,
        path=path,
        entry=_entry(data, path, packed=packed),
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("length", "integrity mismatch"),
        ("sha256", "integrity mismatch"),
        ("fingerprint", "remote packed body integrity mismatch"),
    ],
)
def test_verify_cache_body_payload_rejects_each_independent_inventory_mismatch(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    path = tmp_path / "shard_00002.safetensors.zst"
    data = b"authenticated-compressed-payload"
    entry = _entry(data, path, packed=True)
    if mutation == "length":
        entry["byte_length"] += 1
    elif mutation == "sha256":
        entry["content_sha256"] = "0" * 64
    else:
        entry["file_fingerprint"] = "f" * 64

    with pytest.raises(RemoteTrainingError, match=match):
        remote_module._verify_cache_body_payload(
            data,
            path=path,
            entry=entry,
        )


def test_payload_fingerprint_is_salted_by_final_inventory_name(
    tmp_path: Path,
) -> None:
    original = tmp_path / "shard_00000.safetensors.zst"
    renamed = tmp_path / "shard_00001.safetensors.zst"
    data = b"same bytes, different committed name"
    original.write_bytes(data)
    renamed.write_bytes(data)
    original_entry = _entry(data, original, packed=True)

    size, content_sha256, renamed_fingerprint = remote_module._dual_digest(renamed)
    assert size == len(data)
    assert content_sha256 == original_entry["content_sha256"]
    assert renamed_fingerprint != original_entry["file_fingerprint"]
    with pytest.raises(
        RemoteTrainingError,
        match="remote packed body integrity mismatch",
    ):
        remote_module._verify_cache_body_payload(
            data,
            path=renamed,
            entry=original_entry,
        )


@pytest.mark.parametrize("verify_content", [False, True])
def test_read_cache_body_returns_exact_bytes_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_content: bool,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = os.urandom(64 * 1024 + 17)
    path.write_bytes(data)
    entry = _entry(data, path, packed=True)
    real_open = os.open
    opened: list[tuple[int, tuple[int, int]]] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path:
            info = os.fstat(fd)
            opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    assert (
        remote_module._read_cache_body_bytes(
            path,
            entry=entry,
            verify_content=verify_content,
        )
        == data
    )
    assert len(opened) == 1
    _assert_original_fd_closed(*opened[0])


def test_trusted_stable_body_skips_payload_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"trusted-stable-body"
    path.write_bytes(data)

    def forbidden_verification(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("stable trusted cache hit must not hash content")

    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        forbidden_verification,
    )
    assert (
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path, packed=True),
            verify_content=False,
        )
        == data
    )


def test_trusted_stable_body_preserves_explicit_same_length_corruption_tradeoff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    signed = b"signed-payload"
    corrupt = b"bad-payload!!!"
    assert len(corrupt) == len(signed)
    path.write_bytes(corrupt)
    entry = _entry(signed, path, packed=True)

    assert (
        remote_module._read_cache_body_bytes(
            path,
            entry=entry,
            verify_content=False,
        )
        == corrupt
    )
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        remote_module._read_cache_body_bytes(
            path,
            entry=entry,
            verify_content=True,
        )


def test_strict_stable_body_hashes_the_exact_returned_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"strict-stable-body"
    path.write_bytes(data)
    real_verify = remote_module._verify_cache_body_payload
    seen: list[bytes] = []

    def recording_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        seen.append(payload)
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        recording_verify,
    )
    returned = remote_module._read_cache_body_bytes(
        path,
        entry=_entry(data, path, packed=True),
        verify_content=True,
    )
    assert returned == data
    assert seen == [returned]


def test_missing_body_preserves_filenotfound_for_bounded_materializer_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.safetensors.zst"
    with pytest.raises(FileNotFoundError):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(b"x", path),
            verify_content=True,
        )


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        "dangling_symlink",
        "directory",
        "fifo",
        "unix_socket",
        "hardlink",
    ],
)
def test_read_cache_body_rejects_every_non_direct_single_link_file_kind(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    source = tmp_path / "source.safetensors.zst"
    source.write_bytes(b"x")
    bound_socket: socket.socket | None = None
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "dangling_symlink":
        path.symlink_to(tmp_path / "does-not-exist")
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "unix_socket":
        bound_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_socket.bind(str(path))
    else:
        os.link(source, path)

    started = time.monotonic()
    try:
        with _blocking_open_watchdog():
            with pytest.raises(RemoteTrainingError):
                remote_module._read_cache_body_bytes(
                    path,
                    entry=_entry(b"x", path),
                    verify_content=True,
                )
    finally:
        if bound_socket is not None:
            bound_socket.close()
    assert time.monotonic() - started < 1.0


def test_body_open_uses_nofollow_nonblock_and_cloexec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    data = b"flags"
    path.write_bytes(data)
    real_open = os.open
    seen_flags: list[int] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(os.fsdecode(candidate)) == path:
            seen_flags.append(flags)
        return real_open(candidate, flags, mode)

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    assert (
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path),
            verify_content=True,
        )
        == data
    )
    assert len(seen_flags) == 1
    assert seen_flags[0] & os.O_NOFOLLOW
    assert seen_flags[0] & os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        assert seen_flags[0] & os.O_CLOEXEC


def test_atomic_replacement_after_open_returns_verified_old_inode_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    replacement = tmp_path / "replacement"
    old_data = b"A" * 31
    new_data = b"B" * 31
    assert len(old_data) == len(new_data)
    path.write_bytes(old_data)
    replacement.write_bytes(new_data)
    entry = _entry(old_data, path, packed=True)
    real_open = os.open
    real_replace = os.replace
    fired = False
    verification_payloads: list[bytes] = []
    real_verify = remote_module._verify_cache_body_payload

    def replacing_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal fired
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path and not fired:
            fired = True
            real_replace(replacement, path)
        return fd

    def recording_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        verification_payloads.append(payload)
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(remote_module.os, "open", replacing_open)
    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        recording_verify,
    )
    returned = remote_module._read_cache_body_bytes(
        path,
        entry=entry,
        verify_content=False,
    )
    assert fired
    assert returned == old_data
    assert path.read_bytes() == new_data
    assert verification_payloads == [old_data]


def test_atomic_replacement_cannot_make_reader_accept_bad_opened_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    replacement = tmp_path / "replacement"
    expected = b"E" * 25
    opened_bad = b"B" * 25
    assert len(expected) == len(opened_bad)
    path.write_bytes(opened_bad)
    replacement.write_bytes(expected)
    real_open = os.open
    real_replace = os.replace

    def replacing_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path:
            real_replace(replacement, path)
        return fd

    monkeypatch.setattr(remote_module.os, "open", replacing_open)
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(expected, path, packed=True),
            verify_content=False,
        )
    assert path.read_bytes() == expected


def test_unlink_after_open_forces_verification_and_returns_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"authenticated-unlinked-open-inode"
    path.write_bytes(data)
    real_open = os.open
    calls = 0
    real_verify = remote_module._verify_cache_body_payload

    def unlinking_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path:
            path.unlink()
        return fd

    def counted_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        nonlocal calls
        calls += 1
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(remote_module.os, "open", unlinking_open)
    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        counted_verify,
    )
    assert (
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path, packed=True),
            verify_content=False,
        )
        == data
    )
    assert calls == 1
    assert not path.exists()


def test_trusted_metadata_only_signature_change_verifies_exact_read_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"metadata-only-transition"
    path.write_bytes(data)
    real_fstat = os.fstat
    fstat_calls = 0
    verification_payloads: list[bytes] = []
    real_verify = remote_module._verify_cache_body_payload

    def changing_fstat(fd: int) -> os.stat_result:
        nonlocal fstat_calls
        actual = real_fstat(fd)
        fstat_calls += 1
        if fstat_calls == 1:
            return actual
        assert fstat_calls == 2
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_mode=actual.st_mode,
                st_dev=actual.st_dev,
                st_ino=actual.st_ino,
                st_size=actual.st_size,
                st_mtime_ns=actual.st_mtime_ns + 1,
                st_ctime_ns=actual.st_ctime_ns,
                st_nlink=actual.st_nlink,
            ),
        )

    def recording_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        verification_payloads.append(payload)
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(remote_module.os, "fstat", changing_fstat)
    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        recording_verify,
    )
    returned = remote_module._read_cache_body_bytes(
        path,
        entry=_entry(data, path, packed=True),
        verify_content=False,
    )

    assert returned == data
    assert verification_payloads == [returned]
    assert fstat_calls == 2


@pytest.mark.parametrize("receipt_drift", ["device", "timestamps"])
def test_fresh_receipt_cross_client_metadata_drift_forces_payload_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_drift: str,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"fresh-owner-verified-body"
    path.write_bytes(data)
    receipt = remote_module._cache_body_descriptor(path.stat())
    if receipt_drift == "device":
        receipt["device"] += 1
    else:
        receipt["mtime_ns"] += 1
        receipt["ctime_ns"] += 1
    real_verify = remote_module._verify_cache_body_payload
    verification_payloads: list[bytes] = []

    def recording_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        verification_payloads.append(payload)
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        recording_verify,
    )
    returned = remote_module._read_cache_body_bytes(
        path,
        entry=_entry(data, path, packed=True),
        verify_content=False,
        expected_descriptor=receipt,
    )

    assert returned == data
    assert verification_payloads == [returned]


def test_fresh_receipt_metadata_drift_rejects_same_length_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    signed = b"signed-payload"
    corrupt = b"bad-payload!!!"
    assert len(corrupt) == len(signed)
    path.write_bytes(signed)
    receipt = remote_module._cache_body_descriptor(path.stat())
    receipt["device"] += 1
    path.write_bytes(corrupt)

    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(signed, path, packed=True),
            verify_content=False,
            expected_descriptor=receipt,
        )


def test_trusted_unchanged_signature_accepts_same_length_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance is the explicit trusted-cache limitation, not desired integrity."""
    path = tmp_path / "shard_00000.safetensors.zst"
    original = b"A" * 4096
    mutated = b"B" * 4096
    path.write_bytes(original)
    original_info = path.stat()
    state = _install_unchanged_signature_same_length_mutation(
        path,
        mutated,
        monkeypatch,
    )
    unrelated_path = tmp_path / "unrelated"
    unrelated_path.write_bytes(b"unrelated")
    unrelated_fd = os.open(unrelated_path, os.O_RDONLY)
    try:
        unrelated_info = os.fstat(unrelated_fd)
    finally:
        os.close(unrelated_fd)
    assert (unrelated_info.st_dev, unrelated_info.st_ino) == (
        unrelated_path.stat().st_dev,
        unrelated_path.stat().st_ino,
    )
    assert state["calls"] == 0
    forbidden_verification = Mock(
        side_effect=AssertionError("trusted unchanged-signature read must not verify the payload")
    )

    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        forbidden_verification,
    )
    result = remote_module._read_cache_body_bytes(
        path,
        entry=_entry(original, path),
        verify_content=False,
    )

    final_info = path.stat()
    first = state["first"]
    second_actual = state["second_actual"]
    assert isinstance(first, os.stat_result)
    assert isinstance(second_actual, os.stat_result)
    assert result == mutated
    forbidden_verification.assert_not_called()
    assert (final_info.st_dev, final_info.st_ino) == (
        original_info.st_dev,
        original_info.st_ino,
    )
    assert first.st_size == second_actual.st_size == final_info.st_size == len(original)
    assert len(result) == len(original)
    assert first.st_nlink == second_actual.st_nlink == final_info.st_nlink == 1
    assert state["calls"] == 2
    assert len(state["returned"]) == 2
    assert state["returned"][0] is first
    assert state["returned"][1] is first
    assert path.read_bytes() == mutated


def test_strict_unchanged_signature_rejects_same_length_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode authenticates the read buffer despite an unchanged signature."""
    path = tmp_path / "shard_00000.safetensors.zst"
    original = b"A" * 4096
    mutated = b"B" * 4096
    path.write_bytes(original)
    original_info = path.stat()
    state = _install_unchanged_signature_same_length_mutation(
        path,
        mutated,
        monkeypatch,
    )
    real_verify = remote_module._verify_cache_body_payload
    verification_payloads: list[bytes] = []

    def recording_verify(
        payload: bytes,
        *,
        path: Path,
        entry: dict[str, Any],
    ) -> None:
        verification_payloads.append(payload)
        real_verify(payload, path=path, entry=entry)

    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        recording_verify,
    )
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(original, path),
            verify_content=True,
        )

    final_info = path.stat()
    first = state["first"]
    second_actual = state["second_actual"]
    assert isinstance(first, os.stat_result)
    assert isinstance(second_actual, os.stat_result)
    assert verification_payloads == [mutated]
    assert (final_info.st_dev, final_info.st_ino) == (
        original_info.st_dev,
        original_info.st_ino,
    )
    assert first.st_size == second_actual.st_size == final_info.st_size == len(original)
    assert first.st_nlink == second_actual.st_nlink == final_info.st_nlink == 1
    assert state["calls"] == 2
    assert len(state["returned"]) == 2
    assert state["returned"][0] is first
    assert state["returned"][1] is first
    assert path.read_bytes() == mutated


@pytest.mark.parametrize("read_kind", ["short", "long"])
def test_trusted_incomplete_read_fails_before_payload_handoff_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_kind: str,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    data = b"signed-complete-body"
    path.write_bytes(data)
    synthetic_data = data[:-1] if read_kind == "short" else data + b"!"
    real_open = os.open
    opened: list[tuple[int, tuple[int, int]]] = []
    forbidden_verification = Mock(
        side_effect=AssertionError("incomplete trusted read must fail before hashing")
    )

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        assert Path(os.fsdecode(candidate)) == path
        info = os.fstat(fd)
        opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    @contextmanager
    def synthetic_fdopen(*args: Any, **kwargs: Any) -> Iterator[Any]:
        del args, kwargs
        yield SimpleNamespace(read=lambda: synthetic_data)

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    monkeypatch.setattr(remote_module.os, "fdopen", synthetic_fdopen)
    monkeypatch.setattr(
        remote_module,
        "_verify_cache_body_payload",
        forbidden_verification,
    )
    with pytest.raises(RemoteTrainingError, match="integrity mismatch while reading"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path),
            verify_content=False,
        )

    assert len(opened) == 1
    forbidden_verification.assert_not_called()
    _assert_original_fd_closed(*opened[0])


def test_size_growth_after_read_fails_before_payload_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    data = b"fixed-size"
    path.write_bytes(data)
    real_fstat = os.fstat
    fstat_calls = 0

    def growing_fstat(fd: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            with path.open("ab") as handle:
                handle.write(b"!")
                handle.flush()
                os.fsync(handle.fileno())
        return real_fstat(fd)

    monkeypatch.setattr(remote_module.os, "fstat", growing_fstat)
    with pytest.raises(RemoteTrainingError, match="integrity mismatch while reading"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path),
            verify_content=False,
        )
    assert fstat_calls == 2


def test_hardlink_added_during_read_fails_before_payload_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard_00000.safetensors.zst"
    added_link = tmp_path / "added-hardlink"
    data = b"single-link-at-open"
    path.write_bytes(data)
    real_fstat = os.fstat
    fstat_calls = 0

    def linking_fstat(fd: int) -> os.stat_result:
        nonlocal fstat_calls
        result = real_fstat(fd)
        fstat_calls += 1
        if fstat_calls == 1:
            os.link(path, added_link)
        return result

    monkeypatch.setattr(remote_module.os, "fstat", linking_fstat)
    with pytest.raises(RemoteTrainingError, match="gained multiple hard links"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path),
            verify_content=False,
        )
    assert fstat_calls == 2
    assert path.stat().st_nlink == added_link.stat().st_nlink == 2


def test_read_failure_closes_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    path.write_bytes(b"wrong-size")
    real_open = os.open
    opened: list[tuple[int, tuple[int, int]]] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path:
            info = os.fstat(fd)
            opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(b"different-size", path),
            verify_content=True,
        )
    assert len(opened) == 1
    _assert_original_fd_closed(*opened[0])


def test_read_oserror_is_wrapped_and_closes_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    data = b"read-error"
    path.write_bytes(data)
    real_open = os.open
    opened: list[tuple[int, tuple[int, int]]] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == path:
            info = os.fstat(fd)
            opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    def failing_fdopen(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError(errno.EIO, "injected descriptor read failure")

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    monkeypatch.setattr(remote_module.os, "fdopen", failing_fdopen)
    with pytest.raises(
        RemoteTrainingError,
        match="cannot read remote cache body",
    ):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(data, path),
            verify_content=True,
        )
    assert len(opened) == 1
    _assert_original_fd_closed(*opened[0])


def test_open_cache_lock_creates_and_reopens_one_direct_inode(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".shard.lock"
    first_fd = remote_module._open_cache_lock(lock_path)
    try:
        first = os.fstat(first_fd)
        linked = lock_path.lstat()
        assert (first.st_dev, first.st_ino) == (linked.st_dev, linked.st_ino)
        assert first.st_nlink == linked.st_nlink == 1
    finally:
        os.close(first_fd)

    second_fd = remote_module._open_cache_lock(lock_path)
    try:
        second = os.fstat(second_fd)
        assert (second.st_dev, second.st_ino) == (first.st_dev, first.st_ino)
    finally:
        os.close(second_fd)


@pytest.mark.parametrize(
    "kind",
    ["symlink", "directory", "fifo", "unix_socket", "hardlink"],
)
def test_open_cache_lock_rejects_unsafe_lock_file_kinds(
    tmp_path: Path,
    kind: str,
) -> None:
    lock_path = tmp_path / ".shard.lock"
    source = tmp_path / "other-lock"
    source.write_bytes(b"")
    bound_socket: socket.socket | None = None
    if kind == "symlink":
        lock_path.symlink_to(source)
    elif kind == "directory":
        lock_path.mkdir()
    elif kind == "fifo":
        os.mkfifo(lock_path)
    elif kind == "unix_socket":
        bound_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_socket.bind(str(lock_path))
    else:
        os.link(source, lock_path)

    try:
        with _blocking_open_watchdog():
            with pytest.raises(RemoteTrainingError):
                remote_module._open_cache_lock(lock_path)
    finally:
        if bound_socket is not None:
            bound_socket.close()


def test_open_cache_lock_rejects_fd_path_inode_mismatch_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".shard.lock"
    lock_path.write_bytes(b"")
    alternate = tmp_path / ".alternate.lock"
    alternate.write_bytes(b"")
    real_open = os.open
    opened: list[tuple[int, tuple[int, int]]] = []

    def wrong_inode_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        candidate_path = Path(os.fsdecode(candidate))
        fd = real_open(
            alternate if candidate_path == lock_path else candidate,
            flags,
            mode,
        )
        if candidate_path == lock_path:
            info = os.fstat(fd)
            opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    monkeypatch.setattr(remote_module.os, "open", wrong_inode_open)
    with pytest.raises(RemoteTrainingError, match="path changed while opening"):
        remote_module._open_cache_lock(lock_path)
    assert len(opened) == 1
    _assert_original_fd_closed(*opened[0])


def test_open_cache_lock_wraps_validation_oserror_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".shard.lock"
    real_open = os.open
    real_lstat = Path.lstat
    opened: list[tuple[int, tuple[int, int]]] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        fd = real_open(candidate, flags, mode)
        if Path(os.fsdecode(candidate)) == lock_path:
            info = os.fstat(fd)
            opened.append((fd, (info.st_dev, info.st_ino)))
        return fd

    def disappearing_lstat(candidate: Path) -> os.stat_result:
        if candidate == lock_path:
            raise FileNotFoundError(lock_path)
        return real_lstat(candidate)

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    monkeypatch.setattr(Path, "lstat", disappearing_lstat)
    with pytest.raises(RemoteTrainingError, match="cannot validate remote cache lock"):
        remote_module._open_cache_lock(lock_path)
    assert len(opened) == 1
    _assert_original_fd_closed(*opened[0])


def test_lock_open_uses_nofollow_and_cloexec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / ".shard.lock"
    real_open = os.open
    seen_flags: list[int] = []

    def recording_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(os.fsdecode(candidate)) == lock_path:
            seen_flags.append(flags)
        return real_open(candidate, flags, mode)

    monkeypatch.setattr(remote_module.os, "open", recording_open)
    fd = remote_module._open_cache_lock(lock_path)
    os.close(fd)
    assert len(seen_flags) == 1
    assert seen_flags[0] & os.O_NOFOLLOW
    assert seen_flags[0] & os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        assert seen_flags[0] & os.O_CLOEXEC


def test_helpers_fail_loudly_if_platform_has_no_nofollow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "body.safetensors.zst"
    path.write_bytes(b"x")
    monkeypatch.delattr(remote_module.os, "O_NOFOLLOW")

    with pytest.raises(RemoteTrainingError, match=r"require(?:s)? os.O_NOFOLLOW"):
        remote_module._read_cache_body_bytes(
            path,
            entry=_entry(b"x", path),
            verify_content=True,
        )
    with pytest.raises(RemoteTrainingError, match=r"require(?:s)? os.O_NOFOLLOW"):
        remote_module._open_cache_lock(tmp_path / ".body.lock")
