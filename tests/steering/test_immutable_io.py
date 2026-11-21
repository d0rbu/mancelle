"""Dedicated suite for the create-once immutable artifact primitive and the
descriptor-bound streaming hash helper (Codex B1 fix batch finding 1)."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import re
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.immutable_io as immutable_io
from steering.immutable_io import (
    ImmutableArtifactError,
    StreamedFileDigest,
    install_bytes_once,
    stream_hash_file_descriptor_bound,
)
from steering.refined_types import PositiveInt, PositiveIntLike

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


def test_first_install_writes_bytes_and_returns_true(tmp_path: Path):
    target = tmp_path / "artifact.json"
    assert install_bytes_once(b"payload-1", target, surface="test artifact") is True
    assert target.read_bytes() == b"payload-1"


def test_replay_with_identical_bytes_is_noop_returning_false(tmp_path: Path):
    target = tmp_path / "artifact.json"
    install_bytes_once(b"payload-1", target, surface="test artifact")
    before = target.stat().st_mtime_ns
    assert install_bytes_once(b"payload-1", target, surface="test artifact") is False
    assert target.stat().st_mtime_ns == before
    assert target.read_bytes() == b"payload-1"


def test_replay_with_different_bytes_raises_and_preserves_original(tmp_path: Path):
    target = tmp_path / "artifact.json"
    install_bytes_once(b"payload-1", target, surface="test artifact")
    with pytest.raises(ImmutableArtifactError, match="differs from the current result"):
        install_bytes_once(b"payload-2", target, surface="test artifact")
    assert target.read_bytes() == b"payload-1"


def test_creates_missing_parent_directories(tmp_path: Path):
    target = tmp_path / "a" / "b" / "artifact.json"
    assert install_bytes_once(b"x", target, surface="nested artifact") is True
    assert target.read_bytes() == b"x"


def test_no_tmp_files_left_behind(tmp_path: Path):
    target = tmp_path / "artifact.json"
    install_bytes_once(b"payload-1", target, surface="test artifact")
    with pytest.raises(ImmutableArtifactError):
        install_bytes_once(b"payload-2", target, surface="test artifact")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "artifact.json"]
    assert leftovers == []


def test_externally_tampered_artifact_rejected_on_replay(tmp_path: Path):
    target = tmp_path / "artifact.json"
    install_bytes_once(b"payload-1", target, surface="test artifact")
    target.write_bytes(b"tampered")
    with pytest.raises(ImmutableArtifactError, match="differs"):
        install_bytes_once(b"payload-1", target, surface="test artifact")


def test_surface_name_appears_in_error(tmp_path: Path):
    target = tmp_path / "artifact.json"
    install_bytes_once(b"payload-1", target, surface="candidate provenance sidecar")
    with pytest.raises(ImmutableArtifactError, match="candidate provenance sidecar"):
        install_bytes_once(b"payload-2", target, surface="candidate provenance sidecar")


def test_replay_through_a_symlink_with_identical_target_bytes_is_a_noop(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"payload")
    link = tmp_path / "artifact.json"
    link.symlink_to(target)

    assert install_bytes_once(b"payload", link, surface="test artifact") is False


def test_replay_through_a_symlink_with_different_target_bytes_raises(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"installed")
    link = tmp_path / "artifact.json"
    link.symlink_to(target)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            f"^test artifact at {re.escape(str(link))} differs from the current result; "
            r"refusing to overwrite it\. Use a fresh run tag\.$"
        ),
    ):
        install_bytes_once(b"current", link, surface="test artifact")


@pytest.mark.parametrize(
    "path_kind",
    ["plain_regular_file", "symlinked_final_component", "symlinked_parent"],
)
def test_exact_replay_honest_paths_remain_noops(tmp_path: Path, path_kind: str) -> None:
    payload = b"payload"
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real_target = real_parent / "artifact.json"
    real_target.write_bytes(payload)

    if path_kind == "plain_regular_file":
        declared = real_target
    elif path_kind == "symlinked_final_component":
        declared = tmp_path / "artifact-link.json"
        declared.symlink_to(real_target)
    else:
        parent_link = tmp_path / "live"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        declared = parent_link / real_target.name

    assert install_bytes_once(payload, declared, surface="test artifact") is False
    assert declared.read_bytes() == payload


def test_replay_rejects_parent_symlink_retargeted_immediately_after_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_a = tmp_path / "A"
    directory_b = tmp_path / "B"
    directory_a.mkdir()
    directory_b.mkdir()
    (directory_a / "artifact").write_bytes(b"A")
    (directory_b / "artifact").write_bytes(b"B")
    live = tmp_path / "live"
    live.symlink_to(directory_a, target_is_directory=True)
    declared = live / "artifact"
    assert declared.exists()
    assert declared.read_bytes() == b"A"
    real_resolve = Path.resolve
    retargeted = False

    def resolve_then_retarget(self: Path, strict: bool = False) -> Path:
        nonlocal retargeted
        resolved = real_resolve(self, strict=strict)
        if self == declared and not retargeted:
            retargeted = True
            live.unlink()
            live.symlink_to(directory_b, target_is_directory=True)
        return resolved

    monkeypatch.setattr(Path, "resolve", resolve_then_retarget)

    with pytest.raises(ImmutableArtifactError, match="changed while it was being read"):
        install_bytes_once(b"A", declared, surface="test artifact")
    assert retargeted
    assert declared.read_bytes() == b"B"


def test_replay_accepts_hard_link_metadata_change_after_descriptor_bound_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"payload"
    target = tmp_path / "artifact.json"
    target.write_bytes(payload)
    transient_link = tmp_path / "transient-link.json"
    real_stream_hash = immutable_io.stream_hash_file_descriptor_bound

    def hash_then_change_link_count(
        path: str | Path,
        *,
        on_error: Any = None,
        chunk_size: PositiveIntLike = 8_388_608,
    ) -> StreamedFileDigest:
        digest = real_stream_hash(path, on_error=on_error, chunk_size=chunk_size)
        os.link(target, transient_link)
        transient_link.unlink()
        assert target.read_bytes() == payload
        return digest

    monkeypatch.setattr(
        immutable_io,
        "stream_hash_file_descriptor_bound",
        hash_then_change_link_count,
    )

    assert install_bytes_once(payload, target, surface="test artifact") is False
    assert target.read_bytes() == payload


def test_replay_rejects_a_directory_with_stable_read_error(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.mkdir()

    with pytest.raises(
        ImmutableArtifactError,
        match=f"^Could not stably read test artifact at {re.escape(str(target))}$",
    ):
        install_bytes_once(b"payload", target, surface="test artifact")


def test_empty_payload_exact_replay_and_nonempty_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"

    assert install_bytes_once(b"", target, surface="test artifact") is True
    assert install_bytes_once(b"", target, surface="test artifact") is False
    with pytest.raises(
        ImmutableArtifactError,
        match=(
            f"^test artifact at {re.escape(str(target))} differs from the current result; "
            r"refusing to overwrite it\. Use a fresh run tag\.$"
        ),
    ):
        install_bytes_once(b"z", target, surface="test artifact")


@pytest.mark.parametrize(
    ("reason", "expected_message"),
    [
        ("not_a_regular_file", "Could not stably read test artifact"),
        ("read_failed", "Could not stably read test artifact"),
        ("changed_before_open", "test artifact changed while it was being read"),
        ("changed_during_read", "test artifact changed while it was being read"),
        ("changed_after_read", "test artifact changed while it was being read"),
    ],
)
def test_replay_error_factory_returns_a_non_oserror_immutable_artifact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    expected_message: str,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_bytes(b"payload")

    def fail_with_reason(
        path: str | Path,
        *,
        on_error: Any = None,
        chunk_size: PositiveIntLike = 8_388_608,
    ) -> StreamedFileDigest:
        del path, chunk_size
        assert on_error is not None
        raise on_error(reason)

    monkeypatch.setattr(
        immutable_io,
        "stream_hash_file_descriptor_bound",
        fail_with_reason,
    )

    with pytest.raises(ImmutableArtifactError, match=expected_message) as error:
        immutable_io._replay_matches(target, b"payload", surface="test artifact")
    assert not isinstance(error.value, OSError)


def test_replay_streams_a_real_multi_chunk_payload(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    payload = b"A" * (8 * 1024 * 1024 + 1)

    assert install_bytes_once(payload, target, surface="test artifact") is True
    assert install_bytes_once(payload, target, surface="test artifact") is False
    with pytest.raises(
        ImmutableArtifactError,
        match=(
            f"^test artifact at {re.escape(str(target))} differs from the current result; "
            r"refusing to overwrite it\. Use a fresh run tag\.$"
        ),
    ):
        install_bytes_once(payload[:-1] + b"B", target, surface="test artifact")


def test_replay_rejects_equal_length_different_content(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"AAAA")

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            f"^test artifact at {re.escape(str(target))} differs from the current result; "
            r"refusing to overwrite it\. Use a fresh run tag\.$"
        ),
    ):
        install_bytes_once(b"BBBB", target, surface="test artifact")


def test_replay_wraps_a_file_vanishing_after_the_exists_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.json"
    target.write_bytes(b"payload")
    real_exists = Path.exists
    vanished = False

    def vanishing_exists(self: Path) -> bool:
        nonlocal vanished
        if self == target and not vanished:
            vanished = True
            self.unlink()
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", vanishing_exists)

    with pytest.raises(
        ImmutableArtifactError,
        match=f"^Could not stably read test artifact at {re.escape(str(target))}$",
    ):
        install_bytes_once(b"payload", target, surface="test artifact")
    assert vanished


def test_install_replay_detects_a_real_parent_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing-path replay must bind the digest to the opened file."""
    live = tmp_path / "live"
    staged = tmp_path / "staged"
    parked = tmp_path / "parked"
    live.mkdir()
    staged.mkdir()
    target = live / "artifact.json"
    other = staged / "artifact.json"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096
    target.write_bytes(a_bytes)
    other.write_bytes(b_bytes)

    old_before_stat = target.stat()
    old_before = (
        old_before_stat.st_dev,
        old_before_stat.st_ino,
        old_before_stat.st_size,
        old_before_stat.st_mtime_ns,
        old_before_stat.st_ctime_ns,
    )
    swap_depth = 0
    try:
        os.replace(live, parked)
        swap_depth = 1
        os.replace(staged, live)
        swap_depth = 2
        with target.open("rb") as handle:
            payload = handle.read()
        os.replace(live, staged)
        swap_depth = 1
        os.replace(parked, live)
        swap_depth = 0
    finally:
        if swap_depth == 2:
            os.replace(live, staged)
            os.replace(parked, live)
        elif swap_depth == 1:
            os.replace(parked, live)
    old_after_stat = target.stat()
    old_after = (
        old_after_stat.st_dev,
        old_after_stat.st_ino,
        old_after_stat.st_size,
        old_after_stat.st_mtime_ns,
        old_after_stat.st_ctime_ns,
    )
    assert payload == b_bytes
    assert old_before == old_after
    assert target.read_bytes() == a_bytes
    assert other.read_bytes() == b_bytes

    resolved = target.resolve()
    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            swap_depth = 0
            handle: Any = None
            success = False
            try:
                os.replace(live, parked)
                swap_depth = 1
                os.replace(staged, live)
                swap_depth = 2
                handle = real_open(self, *args, **kwargs)  # binds fd to B
                os.replace(live, staged)
                swap_depth = 1
                os.replace(parked, live)
                swap_depth = 0
                success = True
            finally:
                try:
                    if swap_depth == 2:
                        os.replace(live, staged)
                        os.replace(parked, live)
                    elif swap_depth == 1:
                        os.replace(parked, live)
                finally:
                    if not success and handle is not None:
                        handle.close()
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(ImmutableArtifactError, match="changed while it was being read"):
        install_bytes_once(b_bytes, target, surface="test artifact")
    assert swapped["done"]
    assert target.read_bytes() == a_bytes
    assert other.read_bytes() == b_bytes


def test_install_link_race_replay_detects_a_real_parent_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FileExistsError replay must bind the digest to the opened file."""
    live = tmp_path / "live"
    staged = tmp_path / "staged"
    parked = tmp_path / "parked"
    live.mkdir()
    staged.mkdir()
    target = live / "artifact.json"
    other = staged / "artifact.json"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096
    target.write_bytes(a_bytes)
    other.write_bytes(b_bytes)
    resolved = target.resolve()
    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            swap_depth = 0
            handle: Any = None
            success = False
            try:
                os.replace(live, parked)
                swap_depth = 1
                os.replace(staged, live)
                swap_depth = 2
                handle = real_open(self, *args, **kwargs)  # binds fd to B
                os.replace(live, staged)
                swap_depth = 1
                os.replace(parked, live)
                swap_depth = 0
                success = True
            finally:
                try:
                    if swap_depth == 2:
                        os.replace(live, staged)
                        os.replace(parked, live)
                    elif swap_depth == 1:
                        os.replace(parked, live)
                finally:
                    if not success and handle is not None:
                        handle.close()
            return handle
        return real_open(self, *args, **kwargs)

    real_exists = Path.exists

    def force_link_race(self: Path) -> bool:
        if self == target:
            return False
        return real_exists(self)

    link_attempted = False

    def lose_link_race(src: Any, dst: Any) -> None:
        nonlocal link_attempted
        del src
        assert Path(dst) == target
        link_attempted = True
        raise FileExistsError

    monkeypatch.setattr(Path, "open", swapping_open)
    monkeypatch.setattr(Path, "exists", force_link_race)
    monkeypatch.setattr(os, "link", lose_link_race)

    with pytest.raises(ImmutableArtifactError, match="changed while it was being read"):
        install_bytes_once(b_bytes, target, surface="test artifact")
    assert link_attempted
    assert swapped["done"]
    assert target.read_bytes() == a_bytes
    assert other.read_bytes() == b_bytes


def _fifo_replay_worker(path_str: str, results: multiprocessing.Queue[str]) -> None:
    try:
        install_bytes_once(b"payload", Path(path_str), surface="test artifact")
    except ImmutableArtifactError as exc:
        results.put(f"raised:{exc}")
    except Exception as exc:  # pragma: no cover - diagnostic
        results.put(f"unexpected:{type(exc).__name__}")
    else:  # pragma: no cover - regression diagnostic
        results.put("returned")


def test_install_replay_rejects_a_fifo_without_opening_it(tmp_path: Path) -> None:
    """A regression here must time out and fail, never block pytest forever."""
    target = tmp_path / "artifact.fifo"
    os.mkfifo(target)
    ctx = multiprocessing.get_context("spawn")
    results: multiprocessing.Queue[str] = ctx.Queue()
    process = ctx.Process(target=_fifo_replay_worker, args=(str(target), results))
    process.start()
    process.join(timeout=15)
    terminated = not process.is_alive()
    if not terminated:
        process.terminate()
        process.join(timeout=5)

    assert terminated, "FIFO replay child blocked instead of failing loudly"
    assert process.exitcode == 0
    assert results.get(timeout=5) == f"raised:Could not stably read test artifact at {target}"
    results.close()
    results.join_thread()


def _racing_writer(path_str: str, payload: bytes, results: multiprocessing.Queue[str]) -> None:
    try:
        install_bytes_once(payload, Path(path_str), surface="race artifact")
        results.put("ok")
    except ImmutableArtifactError:
        results.put("rejected")
    except Exception as exc:  # pragma: no cover - diagnostic
        results.put(f"unexpected:{type(exc).__name__}")


@pytest.mark.parametrize("same_payload", [True, False])
def test_concurrent_writers_link_race(tmp_path: Path, same_payload: bool):
    target = tmp_path / "artifact.json"
    ctx = multiprocessing.get_context("spawn")
    results: multiprocessing.Queue = ctx.Queue()
    payload_b = b"payload-1" if same_payload else b"payload-2"
    procs = [
        ctx.Process(target=_racing_writer, args=(str(target), b"payload-1", results)),
        ctx.Process(target=_racing_writer, args=(str(target), payload_b, results)),
    ]
    for p in procs:
        p.start()
    outcomes = [results.get(timeout=30) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    results.close()
    results.join_thread()
    if same_payload:
        assert outcomes == ["ok", "ok"]
        assert target.read_bytes() == b"payload-1"
    else:
        # Exactly one writer can win the create-once hard-link race; the loser
        # observes the winner's bytes and rejects. Two "ok"s with different
        # payloads would mean the immutability guarantee was violated.
        assert outcomes.count("ok") <= 1
        assert not any(o.startswith("unexpected:") for o in outcomes)
        assert target.read_bytes() in (b"payload-1", payload_b)


# ==========================================================================
# stream_hash_file_descriptor_bound — Codex B1 fix batch finding 1.
#
# Mirrors steering.conditional.feature_clusters._file_content_identity's
# fd-binding: lstat -> open -> fstat -> stream-read -> fstat -> lstat, with
# st_mode included in the signature, closing the TOCTOU window the oracle's
# pathname-only before/after stat() pair missed.
# ==========================================================================


def test_stream_hash_file_descriptor_bound_matches_manual_sha256_and_captures_stat(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.pt"
    payload = b"x" * 1000
    target.write_bytes(payload)

    digest = stream_hash_file_descriptor_bound(target)

    assert isinstance(digest, StreamedFileDigest)
    assert digest.sha256 == hashlib.sha256(payload).hexdigest()
    assert digest.byte_length == len(payload)
    assert digest.path == target.resolve()
    assert len(digest.stat_signature) == 6
    assert all(type(field) is int for field in digest.stat_signature)
    assert stat.S_ISREG(digest.stat_signature[0])
    assert digest.stat_signature[3] == len(payload)  # size at index 3 (mode, dev, ino, size, ...)
    assert digest.payload is None


@pytest.mark.parametrize("byte_length", [0, 1, 4, 5, 4097], ids=str)
def test_stream_hash_file_descriptor_bound_can_retain_the_exact_hashed_bytes(
    tmp_path: Path,
    byte_length: int,
) -> None:
    target = tmp_path / "retained.bin"
    payload = (bytes(range(251)) * ((byte_length // 251) + 1))[:byte_length]
    target.write_bytes(payload)

    digest = stream_hash_file_descriptor_bound(
        target,
        chunk_size=PositiveInt.parse(4),
        retain_payload=True,
    )

    assert digest.payload == payload
    assert digest.byte_length == len(payload)
    assert digest.sha256 == hashlib.sha256(digest.payload).hexdigest()


@PROPERTY_SETTINGS
@given(
    payload=st.binary(max_size=16_384),
    chunk_size=st.integers(min_value=1, max_value=2_048),
    retain_payload=st.booleans(),
)
def test_stream_hash_file_descriptor_bound_matches_hashlib_for_valid_refined_inputs(
    payload: bytes,
    chunk_size: int,
    retain_payload: bool,
) -> None:
    """The refined boundary preserves hashing for arbitrary valid values."""
    with TemporaryDirectory() as directory:
        target = Path(directory) / "property-payload.bin"
        target.write_bytes(payload)

        digest = stream_hash_file_descriptor_bound(
            target,
            chunk_size=PositiveInt.parse(chunk_size),
            retain_payload=retain_payload,
        )

        assert digest.sha256 == hashlib.sha256(payload).hexdigest()
        assert digest.byte_length == len(payload)
        assert digest.payload == (payload if retain_payload else None)


@pytest.mark.parametrize("bad", [0, 1, None, "true", [], object()])
def test_stream_hash_file_descriptor_bound_rejects_non_bool_retain_payload(
    tmp_path: Path,
    bad: Any,
) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"must not be read")

    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter retain_payload=.*bool",
    ):
        stream_hash_file_descriptor_bound(target, retain_payload=bad)


def test_stream_hash_file_descriptor_bound_hashes_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.pt"
    target.write_bytes(b"")
    digest = stream_hash_file_descriptor_bound(target)
    assert digest.sha256 == hashlib.sha256(b"").hexdigest()
    assert digest.byte_length == 0
    assert digest.stat_signature[3] == 0


@pytest.mark.parametrize("byte_length", [0, 1, 3, 4, 5, 9, 4096], ids=str)
def test_stream_hash_file_descriptor_bound_matches_manual_sha256_across_sizes(
    tmp_path: Path, byte_length: int
) -> None:
    """Multi-chunk coverage with a small chunk_size: exercises 0 chunks, a
    partial chunk, exactly one full chunk, and several full-plus-partial
    chunks through the same ``chunk_size=4`` reader.
    """
    target = tmp_path / "f.bin"
    payload = bytes(range(256)) * ((byte_length // 256) + 1)
    payload = payload[:byte_length]
    target.write_bytes(payload)
    digest = stream_hash_file_descriptor_bound(
        target,
        chunk_size=PositiveInt.parse(4),
    )
    assert digest.sha256 == hashlib.sha256(payload).hexdigest()
    assert digest.byte_length == byte_length


@pytest.mark.parametrize(
    ("chunk_size", "byte_length"),
    [(8, 7), (8, 8), (8, 9)],
    ids=["chunk_minus_1", "chunk_exact", "chunk_plus_1"],
)
def test_stream_hash_file_descriptor_bound_chunk_size_boundary(
    tmp_path: Path, chunk_size: int, byte_length: int
) -> None:
    """8-MiB±1 boundary, exercised via the exposed ``chunk_size`` parameter
    at a small scale instead of allocating real ~8 MiB fixtures (Codex
    finding 5's explicitly sanctioned approach).
    """
    target = tmp_path / "f.bin"
    payload = bytes(range(byte_length))
    target.write_bytes(payload)
    digest = stream_hash_file_descriptor_bound(
        target,
        chunk_size=PositiveInt.parse(chunk_size),
    )
    assert digest.sha256 == hashlib.sha256(payload).hexdigest()
    assert digest.byte_length == byte_length


@pytest.mark.parametrize(
    "bad_chunk_size",
    [0, -1, -1024, True, False, 4.0, 0.0],
    ids=["zero", "negative", "large_negative", "true", "false", "positive_float", "zero_float"],
)
def test_stream_hash_file_descriptor_bound_rejects_an_invalid_chunk_size(
    tmp_path: Path, bad_chunk_size: Any
) -> None:
    """Codex finding 1 (round 2): ``chunk_size`` must be a strict positive
    ``int`` (``type(x) is int and x > 0``). ``bool`` is an ``int`` subclass
    in Python (``True == 1``) but is rejected anyway -- a caller passing a
    flag by mistake where a chunk size was expected must not silently do
    something.  Rejected eagerly, before any filesystem call: the target
    file is never even opened.
    """
    target = tmp_path / "f.bin"
    target.write_bytes(b"not empty")
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter chunk_size=.*PositiveInt",
    ):
        stream_hash_file_descriptor_bound(target, chunk_size=bad_chunk_size)
    # Rejected before any read occurs: the file on disk is untouched.
    assert target.read_bytes() == b"not empty"


def test_stream_hash_file_descriptor_bound_zero_chunk_size_used_to_silently_hash_empty(
    tmp_path: Path,
) -> None:
    """Codex finding 1 (round 2), headline proof: before this fix,
    ``chunk_size=0`` made ``handle.read(0)`` return ``b""`` immediately --
    a 0-byte read is indistinguishable from real EOF via truthiness alone
    -- so the read loop's body never executed even once. A 9-byte file
    silently hashed as ``sha256(b"")`` with ``byte_length=0``, while every
    fstat/lstat checkpoint reported the (genuinely unchanged) 9-byte file as
    "no mutation detected". Now rejected outright before any filesystem
    call is made at all.
    """
    target = tmp_path / "nonempty.bin"
    target.write_bytes(b"not empty")  # 9 bytes
    bad_chunk_size: Any = 0
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter chunk_size=.*PositiveInt",
    ):
        stream_hash_file_descriptor_bound(target, chunk_size=bad_chunk_size)


def test_stream_hash_file_descriptor_bound_detects_a_premature_eof_via_a_wrapped_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 1 (round 2), defense-in-depth proof: even with a
    perfectly valid ``chunk_size``, a reader that signals EOF (returns
    ``b""``) before the real file's bytes are exhausted must still be
    caught. Neither fstat checkpoint alone can see this -- the file's true
    on-disk size, mtime, and ctime never change, only what the ``.read()``
    call chooses to report -- so only comparing the accumulated
    ``byte_length`` against the post-read fstat size catches it. This applies
    the buffered whole-file read's ``len(payload) != before.st_size``
    completeness invariant at the fd-bound checkpoint: the general form of
    the ``chunk_size=0`` failure (same-looking metadata, silently
    short-circuited read), from a source other than an invalid ``chunk_size``.
    """
    target = tmp_path / "model.pt"
    payload = b"0123456789"  # 10 real bytes, untouched on disk throughout
    target.write_bytes(payload)
    resolved = target.resolve()
    real_open = Path.open

    class _PrematureEOFHandle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self._reads = 0

        def read(self, *args: Any, **kwargs: Any) -> bytes:
            self._reads += 1
            if self._reads == 1:
                return self._handle.read(4)  # a genuine partial chunk: b"0123"
            return b""  # lies about EOF; 6 real bytes remain unread on disk

        def fileno(self) -> int:
            return self._handle.fileno()

        def __enter__(self) -> _PrematureEOFHandle:
            return self

        def __exit__(self, *exc_info: Any) -> None:
            self._handle.__exit__(*exc_info)

    def lying_open(self: Path, *args: Any, **kwargs: Any):
        handle = real_open(self, *args, **kwargs)
        if self == resolved:
            return _PrematureEOFHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", lying_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed while it was being read: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    # The real file on disk was never touched by this fake short read. Read
    # through the captured `real_open` rather than `target.read_bytes()`,
    # since `Path.open` is still monkeypatched to `lying_open` here (which
    # would otherwise also intercept this verification read).
    with real_open(target, "rb") as verify_handle:
        assert verify_handle.read() == payload


def test_stream_hash_file_descriptor_bound_empty_file_passes_the_byte_length_invariant(
    tmp_path: Path,
) -> None:
    """The restored ``byte_length == fstat size`` invariant must not reject
    the legitimate zero-byte case: an empty file's read loop hits EOF on
    the very first ``read()`` call (returns ``b""`` immediately, exactly
    like the ``chunk_size=0`` bug -- but here because there is genuinely
    nothing to read), so ``byte_length`` stays ``0`` and the post-read fstat
    size is also ``0``. ``0 == 0`` must pass.
    """
    target = tmp_path / "empty.pt"
    target.write_bytes(b"")
    digest = stream_hash_file_descriptor_bound(target)
    assert digest.byte_length == 0
    assert digest.stat_signature[3] == 0


def test_stream_hash_file_descriptor_bound_raises_file_not_found_for_a_missing_path(
    tmp_path: Path,
) -> None:
    """A missing path raises the raw, unwrapped ``FileNotFoundError`` from
    ``lstat`` — not routed through ``on_error`` — matching
    ``cluster_contract_identity.stream_hash_file_stable``'s pinned
    behavior for this case.
    """
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        stream_hash_file_descriptor_bound(missing)


def test_stream_hash_file_descriptor_bound_rejects_a_directory_with_default_error(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()
    with pytest.raises(
        ImmutableArtifactError,
        match=f"^descriptor-bound hash target is not a regular file: {re.escape(str(directory))}$",
    ):
        stream_hash_file_descriptor_bound(directory)


def test_stream_hash_file_descriptor_bound_rejects_a_symlink_input_directly(
    tmp_path: Path,
) -> None:
    """A symlink is rejected at the top level (lstat sees the link, not the
    target) rather than silently dereferenced — callers that want
    symlink-transparent hashing must resolve the path themselves first.
    """
    target = tmp_path / "real.pt"
    target.write_bytes(b"payload")
    link = tmp_path / "link.pt"
    link.symlink_to(target)
    with pytest.raises(
        ImmutableArtifactError,
        match=f"^descriptor-bound hash target is not a regular file: {re.escape(str(link))}$",
    ):
        stream_hash_file_descriptor_bound(link)


def test_stream_hash_file_descriptor_bound_uses_the_caller_supplied_exception_factory(
    tmp_path: Path,
) -> None:
    """``on_error`` receives the reason keyword (not a pre-formatted
    message) and controls both the exception type and wording — proves the
    dependency-inversion contract this helper is built around.
    """
    directory = tmp_path / "a_directory"
    directory.mkdir()

    class _CustomError(RuntimeError):
        pass

    seen_reasons: list[str] = []

    def on_error(reason: str) -> Exception:
        seen_reasons.append(reason)
        return _CustomError(f"custom:{reason}")

    with pytest.raises(_CustomError, match="^custom:not_a_regular_file$"):
        stream_hash_file_descriptor_bound(directory, on_error=on_error)
    assert seen_reasons == ["not_a_regular_file"]


def test_stream_hash_file_descriptor_bound_propagates_an_os_error_subclass_from_on_error_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 3 (round 2): ``on_error``'s return type is just
    ``Exception`` -- nothing stops a caller's factory from returning an
    ``OSError`` subclass. Before this fix, raising that exception inside
    the single broad ``except OSError`` region wrapping the whole
    open/fstat/read block caused it to be caught a *second* time and
    re-classified as ``"read_failed"``: the factory was invoked twice
    (``["changed_before_open", "read_failed"]``) and the caller observed
    the wrong reason, with the real one only reachable via ``__cause__``.

    Every ``except OSError`` now wraps exactly one filesystem call, and
    none of them wrap the ``if ...: raise _descriptor_bound_error(...)``
    mutation-detection sites -- so a domain exception the factory
    constructs, even one that happens to subclass ``OSError``, propagates
    completely unchanged: exactly one factory invocation, the correct
    reason, and no fabricated ``__cause__``/``__context__``.
    """
    target = tmp_path / "model.pt"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096  # equal size, different content/inode
    target.write_bytes(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)
    resolved = target.resolve()

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)  # A -> B, immediately before open()
            handle = real_open(self, *args, **kwargs)  # binds fd to B
            os.replace(a_restore_source, resolved)  # "B -> A" restore, immediately after
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    class _FactoryOSError(OSError):
        pass

    reasons: list[str] = []

    def factory(reason: str) -> Exception:
        reasons.append(reason)
        return _FactoryOSError(reason)

    with pytest.raises(_FactoryOSError, match="^changed_before_open$") as excinfo:
        stream_hash_file_descriptor_bound(target, on_error=factory)

    assert reasons == ["changed_before_open"]
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_stream_hash_file_descriptor_bound_wraps_os_error_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    resolved = target.resolve()
    real_open = Path.open

    def faulty_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            raise OSError("simulated read failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", faulty_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=f"^could not read descriptor-bound hash target: {re.escape(str(resolved))}$",
    ) as excinfo:
        stream_hash_file_descriptor_bound(target)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_stream_hash_file_descriptor_bound_detects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    resolved = target.resolve()
    real_signature = immutable_io._fd_bound_signature
    calls = 0

    def flaky_signature(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = real_signature(result)
        if calls == 3:  # the post-read fstat checkpoint
            return (
                signature[0],
                signature[1],
                signature[2],
                signature[3],
                signature[4] + 1,
                signature[5],
            )
        return signature

    monkeypatch.setattr(immutable_io, "_fd_bound_signature", flaky_signature)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed while it was being read: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert calls == 3


def test_stream_hash_file_descriptor_bound_detects_a_real_equal_size_swap_around_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 1 (round 1), headline proof: an equal-size A->B swap
    staged immediately before ``open()`` and restored to *fresh* A-content
    immediately after is detected, because the post-open ``fstat`` is bound
    to the descriptor (B's inode) regardless of what the path is later
    restored to.

    Correctness note (round 2, Codex finding 2): this schedule's "restore"
    step replaces the path with a *different* file (``a_restore_source``,
    a distinct inode that merely has A's bytes), not the original A file
    itself — so it is not, on its own, proof that a pathname-only
    before/after ``stat()`` pair would be fooled: a fresh inode changes
    ``st_ino``, which that check *would* have caught too, incidentally.
    This test therefore only proves the new helper correctly detects an
    A->B->A'-content swap (still a real, valuable regression — see the
    ``read_bytes()`` assertion below); it does not by itself demonstrate
    the pathname-only algorithm's actual blind spot. That structural gap
    is proven directly by the companion test below (masks the pathname
    ``stat()`` observation itself) and, as the required real-schedule
    regression, by
    ``test_stream_hash_file_descriptor_bound_detects_a_real_parent_directory_swap``
    (a genuine parent-directory swap that leaves the target's own inode
    completely untouched, so the old algorithm's pathname signature is
    provably identical before and after — no incidental inode change to
    ride along on).
    """
    target = tmp_path / "model.pt"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096  # equal size, different content/inode
    target.write_bytes(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)
    resolved = target.resolve()

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)  # A -> B, immediately before open()
            handle = real_open(self, *args, **kwargs)  # binds fd to B
            os.replace(a_restore_source, resolved)  # "B -> A" restore, immediately after
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed before it could be opened: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert swapped["done"]
    # Left on disk: the restored "A" content — the substitution was
    # transient and would be invisible to anything checking the path after
    # the fact. Only the fd-bound check catches it in the moment.
    assert target.read_bytes() == a_bytes


def _oracle_pathname_only_hash(path: Path) -> str:
    """Faithfully reimplements the *pre-fix* oracle algorithm: a pathname
    ``stat()`` before open, a plain read loop, and a second pathname
    ``stat()`` after — no fd-binding anywhere, no ``st_mode`` in the
    signature. Test-only; not production code. Used to empirically prove
    the swap race below is invisible to that algorithm.
    """
    before = os.stat(path)
    before_sig = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = os.stat(path)
    after_sig = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_sig != after_sig:
        raise ImmutableArtifactError(f"changed while it was being hashed: {path}")
    return digest.hexdigest()


def test_the_same_swap_is_invisible_to_the_oracles_pathname_only_before_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empirical proof the Codex finding describes a real gap, not a
    theoretical one: the identical A->B swap around ``open()`` that the
    fd-bound helper detects and raises on (test above) is silently *missed*
    by a faithful reimplementation of the oracle's pathname-only before/
    after ``stat()`` pair — it returns success, having hashed B's bytes
    under a check that never compared anything to *what open() actually
    returned*.

    The restore-to-"unchanged" half of the race is modeled with a two-phase
    test double on ``os.stat`` (the pathname-only check's only observation
    point) rather than a second real rename: a real double-rename
    empirically bumps ctime on this filesystem (verified separately), which
    would make even the *oracle's* check happen to catch this specific
    reproduction — masking the actual structural gap the fix closes, which
    is that the oracle has no check bound to what ``open()`` returned at
    all, only to two independent, unbound pathname snapshots. A real-world
    substitution has other ways to reproduce that final "looks unchanged"
    observation (coarser mtime/ctime granularity on other filesystems,
    inode reuse, or a restore mechanism that does not itself touch ctime);
    this test isolates the structural property directly rather than betting
    on one filesystem's timestamp resolution.

    This is the test that fails under oracle semantics (no exception; wrong
    hash silently returned) and passes only because it is written to
    *expect* that failure from the reimplementation, contrasted with the
    fix (previous test), which raises.
    """
    target = tmp_path / "model.pt"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096
    target.write_bytes(a_bytes)

    real_open = Path.open
    real_stat = os.stat
    captured_before: list[os.stat_result] = []
    swap_done = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == target and not swap_done["done"]:
            swap_done["done"] = True
            # Same-inode, in-place content swap immediately before open() —
            # the oracle's "before" pathname stat has already been taken.
            # Uses the captured real_open (not target.write_bytes, which
            # would re-enter this same monkeypatched Path.open and recurse).
            with real_open(target, "wb") as writer:
                writer.write(b_bytes)
        return real_open(self, *args, **kwargs)

    def masking_stat(path: Any, *args: Any, **kwargs: Any):
        result = real_stat(path, *args, **kwargs)
        if Path(path) != target:
            return result
        if not captured_before:
            captured_before.append(result)
            return result
        # The "after" observation: a pathname-only check has no binding to
        # what open() actually returned, so this stands in for a restore
        # that reproduces the original pathname snapshot exactly.
        return captured_before[0]

    monkeypatch.setattr(Path, "open", swapping_open)
    monkeypatch.setattr(os, "stat", masking_stat)

    # No exception: the oracle's pathname-only algorithm is fooled...
    sha256_hex = _oracle_pathname_only_hash(target)
    # ...and it silently hashed B's bytes while believing nothing changed.
    assert sha256_hex == hashlib.sha256(b_bytes).hexdigest()
    assert sha256_hex != hashlib.sha256(a_bytes).hexdigest()


def test_stream_hash_file_descriptor_bound_detects_a_real_parent_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 2 (round 2), the required real-schedule regression: the
    two tests above use a *pathname* swap -- replacing the file at the same
    path twice (A -> B -> a fresh A-content file). That second replacement
    lands on a brand-new inode for the "restored" A, which the OLD
    pathname-only ``(dev, ino, size, mtime_ns, ctime_ns)`` check would
    *also* have caught, since a fresh inode changes ``st_ino``. Round 1's
    regression therefore proved nothing about the old design's actual blind
    spot (confirmed empirically below).

    The real miss is a *parent-directory* swap: rename the live directory
    aside, rename a staged sibling directory into its place for exactly the
    duration of ``open()``, then rename both back. The child file named
    ``model.pt`` under the live directory is never itself touched by any of
    this -- ``live/model.pt``'s own inode, size, and timestamps stay
    completely untouched throughout, on both the A side and the B side.
    Only which directory entry the *path* currently resolves to changes,
    and only for the transient window between the two rename pairs. A
    pathname ``stat()`` taken before the dance and one taken after it has
    been fully reversed therefore observes the *same original A file*
    both times and reports an identical tuple.
    """
    live = tmp_path / "live"
    staged = tmp_path / "staged"
    parked = tmp_path / "parked"
    live.mkdir()
    staged.mkdir()
    target = live / "model.pt"
    other = staged / "model.pt"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096  # equal size, different content/inode
    target.write_bytes(a_bytes)
    other.write_bytes(b_bytes)

    # ------------------------------------------------------------------
    # Part 1: explicit proof the OLD pathname-only algorithm would accept
    # this schedule -- computed directly here (not imported from
    # production code) so the comparison tuple is pinned as documentation
    # of exactly what the pre-fix algorithm looked at.
    #
    # State-aware try/finally: an exception anywhere between the first
    # rename and the last restoring rename (including during the
    # open/read of the swapped-in file) must not leave live/staged/parked
    # swapped on disk. ``swap_depth`` tracks how many forward renames are
    # currently outstanding so the finally clause can undo exactly the
    # right ones, in reverse order, from wherever the failure landed.
    # depth 0: canonical layout (nothing outstanding).
    # depth 1: ``live`` is missing; ``parked`` holds live's original
    #          content and ``staged`` holds its own -- reached either
    #          right after the first forward rename or right after the
    #          first restoring rename.
    # depth 2: fully swapped -- ``live`` holds staged's content, ``parked``
    #          holds live's original content, ``staged`` is missing.
    # ------------------------------------------------------------------
    old_before_stat = target.stat()
    old_before = (
        old_before_stat.st_dev,
        old_before_stat.st_ino,
        old_before_stat.st_size,
        old_before_stat.st_mtime_ns,
        old_before_stat.st_ctime_ns,
    )
    swap_depth = 0
    try:
        os.replace(live, parked)
        swap_depth = 1
        os.replace(staged, live)
        swap_depth = 2
        with target.open("rb") as handle:
            payload = handle.read()
        os.replace(live, staged)
        swap_depth = 1
        os.replace(parked, live)
        swap_depth = 0
    finally:
        if swap_depth == 2:
            os.replace(live, staged)
            os.replace(parked, live)
        elif swap_depth == 1:
            os.replace(parked, live)
    old_after_stat = target.stat()
    old_after = (
        old_after_stat.st_dev,
        old_after_stat.st_ino,
        old_after_stat.st_size,
        old_after_stat.st_mtime_ns,
        old_after_stat.st_ctime_ns,
    )
    # The descriptor opened mid-dance really did read the swapped-in B...
    assert payload == b_bytes
    # ...yet the old pathname-only signature reports no change at all...
    assert old_before == old_after
    # ...because by the time of the second stat(), the dance is fully
    # reversed and A is back on disk, byte-for-byte, same inode.
    assert target.read_bytes() == a_bytes
    assert other.read_bytes() == b_bytes

    # ------------------------------------------------------------------
    # Part 2: the identical dance, hooked around the new helper's own
    # open() call -- proves the fd-bound helper raises where the old
    # algorithm (part 1, above) would have silently accepted.
    #
    # Same state-aware ``swap_depth`` tracking as Part 1 above, plus a
    # descriptor-safety guarantee Part 1 doesn't need: ``real_open``
    # binds a live fd to B *before* the two restoring renames run, and
    # that fd is only handed to the caller (the production code under
    # test, which owns closing it) once this closure returns normally.
    # If either restoring rename raises, the fd was never handed off and
    # would otherwise leak -- ``success`` records whether we reached the
    # point of returning it, and the finally closes it ourselves when we
    # didn't. The nested finally ensures the descriptor close still runs
    # even if the directory-restore step itself raises.
    # ------------------------------------------------------------------
    resolved = target.resolve()
    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            swap_depth = 0
            handle: Any = None
            success = False
            try:
                os.replace(live, parked)
                swap_depth = 1
                os.replace(staged, live)
                swap_depth = 2
                handle = real_open(self, *args, **kwargs)  # binds fd to B
                os.replace(live, staged)
                swap_depth = 1
                os.replace(parked, live)
                swap_depth = 0
                success = True
            finally:
                try:
                    if swap_depth == 2:
                        os.replace(live, staged)
                        os.replace(parked, live)
                    elif swap_depth == 1:
                        os.replace(parked, live)
                finally:
                    if not success and handle is not None:
                        handle.close()
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed before it could be opened: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert swapped["done"]
    # Dance fully reversed on disk regardless of the raise.
    assert target.read_bytes() == a_bytes
    assert other.read_bytes() == b_bytes


def test_stream_hash_file_descriptor_bound_detects_a_symlink_substituted_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symlink-target swap: the path is a regular file when first observed
    (lstat), but gets replaced by a symlink to a different file immediately
    before ``open()`` — detected because the descriptor-bound fstat after
    open reports the symlink target's identity, mismatching the original.
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"original content")
    link_target = tmp_path / "elsewhere.pt"
    link_target.write_bytes(b"different content, different length")
    resolved = target.resolve()

    real_open = Path.open
    swapped = {"done": False}

    def symlink_swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            resolved.unlink()
            resolved.symlink_to(link_target)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", symlink_swapping_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed before it could be opened: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert swapped["done"]


def test_stream_hash_file_descriptor_bound_detects_a_rename_at_the_tail_of_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rename-mid-read: the descriptor stays valid and reads the original
    bytes correctly (POSIX rename semantics don't affect an already-open
    fd's readability or content). What *does* surface the tamper is more
    subtle than "the file disappeared": ``rename()`` marks the renamed
    inode's own ctime for update (verified empirically — not just per a
    POSIX reading), and ``fstat`` on the still-open descriptor reports that
    *live* inode state, not a snapshot frozen at ``open()`` time — so the
    post-read fstat checkpoint (bound to the same descriptor as the
    post-open one) already differs and catches this, one checkpoint earlier
    than "the path is simply gone."
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    renamed = tmp_path / "renamed-away.pt"
    resolved = target.resolve()
    real_open = Path.open

    class _RenamingHandle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self._renamed = False

        def read(self, *args: Any, **kwargs: Any) -> bytes:
            chunk = self._handle.read(*args, **kwargs)
            if not chunk and not self._renamed:
                self._renamed = True
                os.replace(resolved, renamed)
            return chunk

        def fileno(self) -> int:
            return self._handle.fileno()

        def __enter__(self) -> _RenamingHandle:
            return self

        def __exit__(self, *exc_info: Any) -> None:
            self._handle.__exit__(*exc_info)

    def renaming_open(self: Path, *args: Any, **kwargs: Any):
        handle = real_open(self, *args, **kwargs)
        if self == resolved:
            return _RenamingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", renaming_open)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            "^descriptor-bound hash target changed while it was being read: "
            f"{re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert renamed.read_bytes() == b"payload"


def test_stream_hash_file_descriptor_bound_final_lstat_catches_a_rename_the_earlier_fstat_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates the *last* checkpoint specifically: forces the post-read
    fstat comparison to observe a masked/pre-rename value (a two-phase test
    double — standing in for e.g. coarser ctime granularity on a filesystem
    that would not register the previous test's real rename at that
    checkpoint) precisely so the rename must instead be caught by the final,
    pathname-based ``lstat`` — proving that checkpoint pulls its own weight
    rather than being redundant with the post-read fstat.
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    renamed = tmp_path / "renamed-away.pt"
    resolved = target.resolve()
    real_signature = immutable_io._fd_bound_signature
    calls = 0

    def masking_signature(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = real_signature(result)
        if calls == 3:
            # This is the post-read fstat checkpoint. Rename the path away
            # as a side effect *after* capturing the (still pre-rename)
            # signature, so this checkpoint reports "unchanged" and control
            # reaches the final lstat, which will find nothing at `resolved`.
            os.replace(resolved, renamed)
        return signature

    monkeypatch.setattr(immutable_io, "_fd_bound_signature", masking_signature)

    with pytest.raises(
        ImmutableArtifactError,
        match=(
            f"^descriptor-bound hash target changed after it was read: {re.escape(str(resolved))}$"
        ),
    ):
        stream_hash_file_descriptor_bound(target)
    assert calls == 3
    assert renamed.read_bytes() == b"payload"
