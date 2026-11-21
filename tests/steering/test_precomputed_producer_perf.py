"""Tests for the precomputed-producer performance fixes (issue #1507).

Covers the side-stream prefetch producer (CPU passthrough, CUDA bitwise
identity under a loaded default stream) and per-node shared-memory
decode-once (publish/attach round trip, cross-process bitwise identity,
unlink-only eviction with live mappings, stale-segment replacement,
owner storage deadlines, attacher fallback, structural gating and namespace
resolution).
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import torch as t
import zstandard

from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import (
    _SHARED_DECODE_MAGIC,
    _SHARED_DECODE_WORKER_PATH,
    _prefetch_wrapper,
    _shared_decode_load_base_shard,
    _shared_decode_load_owner_shard,
    _shared_decode_local_topology,
    _shared_decode_payload_size,
    _shared_decode_publish,
    _shared_decode_run_namespace,
    _shared_decode_segments,
    _shared_decode_wrap,
)
from steering.overlay_io import save_overlay_shard
from steering.packed_batch_io import (
    PackedRowShard,
    save_packed_row_shard,
)

SHARD_KEYS = [
    "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
    "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
]


def _tiny_shard(seed: int = 0, rows: int = 8) -> PackedRowShard:
    rng = np.random.default_rng(seed)
    T, seg = 32, 16
    segs_per_row = T // seg
    n_seg = rows * segs_per_row
    counts = rng.integers(3, 9, size=n_seg)
    offsets = np.zeros(n_seg + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    return PackedRowShard(
        tokens=t.from_numpy(rng.integers(0, 100, size=(rows, T + 1)).astype(np.uint16)),
        row_seg_offsets=t.from_numpy(
            (np.arange(rows + 1) * segs_per_row).astype(np.int32)
        ),
        seg_token_start=t.from_numpy(
            np.tile(np.arange(segs_per_row) * seg, rows).astype(np.uint16)
        ),
        seg_token_end=t.from_numpy(
            (np.tile(np.arange(segs_per_row) * seg, rows) + seg).astype(np.uint16)
        ),
        seg_doc_id=t.from_numpy(np.zeros(n_seg, dtype=np.uint8)),
        seg_feat_offsets=t.from_numpy(offsets),
        feature_ids=t.from_numpy(rng.integers(0, 200, size=total).astype(np.uint16)),
        feature_pcts=t.from_numpy(rng.integers(0, 256, size=total).astype(np.uint8)),
    )


def _assert_shards_equal(a: PackedRowShard, b: PackedRowShard) -> None:
    for key in SHARD_KEYS:
        assert t.equal(getattr(a, key), getattr(b, key)), key


def _owner_for(shard_path: Path, epoch: int, local_world: int) -> int:
    key = f"{Path(os.path.abspath(os.fspath(shard_path)))}:{epoch}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % local_world


def _shared_name(shard_path: Path, epoch: int) -> str:
    normalized = Path(os.path.abspath(os.fspath(shard_path)))
    digest = hashlib.sha256(f"{normalized}:{epoch}".encode()).hexdigest()
    return f"ngptshard_{_shared_decode_run_namespace()}_{digest[:16]}"


def _clear_registry() -> None:
    for name in list(_shared_decode_segments):
        segment, is_owner = _shared_decode_segments.pop(name)
        if is_owner:
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


@pytest.fixture(autouse=True)
def _isolated_registry():
    dataloader_module._shared_decode_retain_floor = dataloader_module._SHARED_DECODE_RETAIN
    _clear_registry()
    yield
    _clear_registry()
    dataloader_module._shared_decode_retain_floor = dataloader_module._SHARED_DECODE_RETAIN


# ---------------------------------------------------------------------------
# Side-stream prefetch producer
# ---------------------------------------------------------------------------


def test_prefetch_wrapper_cpu_passthrough():
    items = [(t.full((4,), i), {"state": i}) for i in range(6)]
    out = list(_prefetch_wrapper(iter(items), prefetch_batches=3, use_cuda=False))
    assert len(out) == 6
    for i, (tensor, state) in enumerate(out):
        assert t.equal(tensor, t.full((4,), i))
        assert state == {"state": i}


def test_prefetch_wrapper_drains_complete_batches_then_reraises_owner_timeout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.2)
    fifo = tmp_path / "prefetch-blocked.safetensors.zst"
    os.mkfifo(fifo)
    complete = [
        (t.tensor([1, 2]), {"state": 1}),
        (t.tensor([3, 4]), {"state": 2}),
    ]

    def inner():
        yield from complete
        _shared_decode_load_base_shard(fifo, 1, 0, 1)

    wrapped = _prefetch_wrapper(inner(), prefetch_batches=4, use_cuda=False)
    for expected in complete:
        actual = next(wrapped)
        assert t.equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    start = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        next(wrapped)
    assert time.monotonic() - start < 3.0
    assert "shared decode owner timed out" in str(error.value)
    assert os.fspath(fifo) in str(error.value)


@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_prefetch_wrapper_cuda_bitwise_under_load():
    reference = [t.randn(64, 64) for _ in range(8)]

    def gen():
        for i, ref in enumerate(reference):
            yield (ref.to("cuda"), i, {"state": i})

    # Keep the default stream busy so a coupled producer would serialize;
    # correctness (not throughput) is asserted here.
    a = t.randn(2048, 2048, device="cuda")
    for _ in range(30):
        a = a @ a * 1e-3
    out = list(_prefetch_wrapper(gen(), prefetch_batches=4, use_cuda=True))
    t.cuda.synchronize()
    assert len(out) == 8
    for (tensor, idx, state), ref in zip(out, reference, strict=True):
        assert tensor.is_cuda
        assert t.equal(tensor.cpu(), ref)
        assert state == {"state": idx}


# ---------------------------------------------------------------------------
# Shared decode: private zstd worker
# ---------------------------------------------------------------------------


def test_shared_decode_owner_and_consumer_timeout_defaults_are_independent():
    assert dataloader_module._SHARED_DECODE_OWNER_TIMEOUT_S == 900.0
    assert dataloader_module._SHARED_DECODE_TIMEOUT_S == 120.0


@pytest.mark.parametrize("path_kind", ["ordinary", "spaces", "symlink"])
@pytest.mark.parametrize("tee_enabled", [False, True], ids=["no-tee", "tee"])
def test_shared_decode_worker_emits_exact_decoded_bytes(tmp_path, path_kind, tee_enabled):
    shard = _tiny_shard(seed=11)
    directory = tmp_path / ("directory with spaces" if path_kind == "spaces" else "data")
    directory.mkdir()
    source = save_packed_row_shard(shard, directory / "shard_00000")
    worker_input = source
    if path_kind == "symlink":
        worker_input = tmp_path / "linked shard.safetensors.zst"
        worker_input.symlink_to(source)

    tee_path = tmp_path / "tee compressed bytes"
    command = [sys.executable, os.fspath(_SHARED_DECODE_WORKER_PATH), os.fspath(worker_input)]
    if tee_enabled:
        command.append(os.fspath(tee_path))
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    expected = zstandard.ZstdDecompressor().decompress(source.read_bytes())
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == expected
    assert completed.stderr == b""
    if tee_enabled:
        assert tee_path.read_bytes() == source.read_bytes()
    else:
        assert not tee_path.exists()


def test_shared_decode_worker_tee_failure_does_not_stop_decode(tmp_path):
    shard = _tiny_shard(seed=18)
    source = save_packed_row_shard(shard, tmp_path / "shard_00000")
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("file")
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(_SHARED_DECODE_WORKER_PATH),
            os.fspath(source),
            os.fspath(not_a_directory / "tee"),
        ],
        capture_output=True,
        check=False,
    )
    expected = zstandard.ZstdDecompressor().decompress(source.read_bytes())
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == expected


def test_shared_decode_worker_mid_write_enospc_preserves_decode(tmp_path):
    if not Path("/dev/full").exists():
        pytest.skip("/dev/full is required for a real mid-write ENOSPC")
    shard = _tiny_shard(seed=19)
    source = save_packed_row_shard(shard, tmp_path / "shard_00000")
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(_SHARED_DECODE_WORKER_PATH),
            os.fspath(source),
            "/dev/full",
        ],
        capture_output=True,
        check=False,
    )
    expected = zstandard.ZstdDecompressor().decompress(source.read_bytes())
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout == expected


def test_shared_decode_worker_failures_are_nonzero(tmp_path):
    missing = tmp_path / "missing shard.safetensors.zst"
    missing_result = subprocess.run(
        [sys.executable, os.fspath(_SHARED_DECODE_WORKER_PATH), os.fspath(missing)],
        capture_output=True,
        check=False,
    )
    assert missing_result.returncode != 0
    assert os.fspath(missing).encode() in missing_result.stderr

    corrupt = tmp_path / "corrupt.safetensors.zst"
    corrupt.write_bytes(b"not a zstd frame")
    corrupt_result = subprocess.run(
        [sys.executable, os.fspath(_SHARED_DECODE_WORKER_PATH), os.fspath(corrupt)],
        capture_output=True,
        check=False,
    )
    assert corrupt_result.returncode != 0


def test_shared_decode_worker_rejects_corruption_after_partial_output(tmp_path):
    decoded = bytes(range(256)) * 16384
    compressed = bytearray(zstandard.ZstdCompressor(write_checksum=True).compress(decoded))
    compressed[-1] ^= 0xFF
    corrupt = tmp_path / "partial-corrupt.safetensors.zst"
    corrupt.write_bytes(compressed)
    tee = tmp_path / "partial-corrupt.tee"

    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(_SHARED_DECODE_WORKER_PATH),
            os.fspath(corrupt),
            os.fspath(tee),
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout
    assert decoded.startswith(completed.stdout)
    assert tee.read_bytes() == corrupt.read_bytes()


def test_shared_decode_worker_imports_stay_lightweight():
    tree = ast.parse(_SHARED_DECODE_WORKER_PATH.read_text())
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots == {"pathlib", "sys", "typing", "zstandard"}


# ---------------------------------------------------------------------------
# Shared decode: single-process protocol
# ---------------------------------------------------------------------------


def test_publish_wrap_roundtrip_bitwise():
    shard = _tiny_shard()
    segment = shared_memory.SharedMemory(
        create=True, size=_shared_decode_payload_size(shard), track=False
    )
    try:
        _shared_decode_publish(segment, shard)
        assert segment.buf is not None
        assert bytes(segment.buf[0:16]) == _SHARED_DECODE_MAGIC
        wrapped = _shared_decode_wrap(segment)
        _assert_shards_equal(wrapped, shard)
        assert wrapped.backing_segment is segment
    finally:
        segment.unlink()


def test_wrap_survives_unlink():
    shard = _tiny_shard(seed=1)
    segment = shared_memory.SharedMemory(
        create=True, size=_shared_decode_payload_size(shard), track=False
    )
    _shared_decode_publish(segment, shard)
    wrapped = _shared_decode_wrap(segment)
    segment.unlink()  # name gone; mapping must stay valid via the shard ref
    _assert_shards_equal(wrapped, shard)


def test_owner_load_and_eviction_unlinks_old_names(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"evict{os.getpid()}")
    shard = _tiny_shard(seed=2)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    names = []
    for epoch in range(1, 7):  # local_world=1 => always owner
        loaded = _shared_decode_load_base_shard(path, epoch, 0, 1)
        _assert_shards_equal(loaded, shard)
        names.append(next(reversed(_shared_decode_segments)))
    assert len(_shared_decode_segments) == 4
    for stale_name in names[:2]:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=stale_name, track=False)


@pytest.mark.parametrize("retain", [4, 12])
def test_eviction_retain_window_is_parameterised(tmp_path, monkeypatch, retain):
    monkeypatch.setenv("SLURM_JOB_ID", f"retain{retain}-{os.getpid()}")
    shard = _tiny_shard(seed=20 + retain)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    names = []
    for epoch in range(1, retain + 4):
        loaded = _shared_decode_load_base_shard(path, epoch, 0, 1, retain=retain)
        _assert_shards_equal(loaded, shard)
        names.append(next(reversed(_shared_decode_segments)))

    assert len(_shared_decode_segments) == retain
    for stale_name in names[:-retain]:
        with pytest.raises(FileNotFoundError, match=stale_name):
            shared_memory.SharedMemory(name=stale_name, track=False)
    for live_name in names[-retain:]:
        attached = shared_memory.SharedMemory(name=live_name, track=False)
        attached.close()


def test_historical_retain_unlinks_a_depth_four_lagging_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"negative-control-{os.getpid()}")
    shard = _tiny_shard(seed=37)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    first_name = None
    for epoch in range(1, 10):
        _shared_decode_load_base_shard(path, epoch, 0, 1, retain=4)
        if first_name is None:
            first_name = next(reversed(_shared_decode_segments))

    assert first_name is not None
    with pytest.raises(FileNotFoundError, match=first_name):
        shared_memory.SharedMemory(name=first_name, track=False)


def test_concurrent_eviction_never_lowers_the_retain_floor(monkeypatch) -> None:
    """The retain floor is a process-global read-modify-write with two real callers.

    The train loader's lookahead producer (retain 12 at depth 4) and a
    validation loader on the main thread (retain 4) evict concurrently: eval
    runs between training steps while the producer keeps going. Unlocked, the
    update interleaves as train-reads-4 / val-reads-4 / train-writes-12 /
    val-writes-4 — silently LOWERING the floor and unlinking names a lagging
    sibling rank still needs, which is the 120s attach timeout the floor exists
    to prevent.

    Reproducing it needs two things, and it is vacuous without either. The real
    window is a few bytecodes wide, so a plain thread race practically never
    lands in it (measured: 400 barrier-synchronised rounds at a 1us switch
    interval never reproduced it) — hence shadowing ``max`` inside the module
    with one that yields, to hold the read open. And once widened, the loser is
    still whichever thread happens to store last, so a single round is a coin
    flip — hence repeating. Verified non-vacuous: this FAILS without the lock.
    """
    real_max = max

    def yielding_max(*args):
        time.sleep(0.005)  # hold the read open across a scheduler switch
        return real_max(*args)

    monkeypatch.setattr(dataloader_module, "max", yielding_max, raising=False)
    try:
        for round_index in range(20):
            dataloader_module._shared_decode_retain_floor = 4
            barrier = threading.Barrier(2)

            def evict(retain: int, gate: threading.Barrier = barrier) -> None:
                gate.wait()
                dataloader_module._shared_decode_evict_old_segments(retain)

            threads = [
                threading.Thread(target=evict, args=(12,)),
                threading.Thread(target=evict, args=(4,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            assert not any(thread.is_alive() for thread in threads)
            assert dataloader_module._shared_decode_retain_floor == 12, (
                f"round {round_index}: retain floor was lowered by a concurrent "
                "lower-retain caller — the read-modify-write is not atomic"
            )
    finally:
        dataloader_module._shared_decode_retain_floor = dataloader_module._SHARED_DECODE_RETAIN


def test_validation_retain_cannot_shrink_process_global_train_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"retain-floor-{os.getpid()}")
    shard = _tiny_shard(seed=41)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    names = []
    for epoch in range(1, 13):
        _shared_decode_load_base_shard(path, epoch, 0, 1, retain=12)
        names.append(next(reversed(_shared_decode_segments)))

    _shared_decode_load_base_shard(path, 13, 0, 1, retain=4)
    names.append(next(reversed(_shared_decode_segments)))

    assert len(_shared_decode_segments) == 12
    for live_name in names[-12:]:
        attached = shared_memory.SharedMemory(name=live_name, track=False)
        attached.close()


def test_stale_committed_segment_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"stale{os.getpid()}")
    shard = _tiny_shard(seed=3)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    name = _shared_name(path, 1)
    stale = shared_memory.SharedMemory(name=name, create=True, size=4096, track=False)
    assert stale.buf is not None
    stale.buf[0:16] = _SHARED_DECODE_MAGIC  # committed-looking garbage
    loaded = _shared_decode_load_base_shard(path, 1, 0, 1)
    _assert_shards_equal(loaded, shard)


def test_owner_symlink_is_lexical_and_launches_one_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"symlink{os.getpid()}")
    shard = _tiny_shard(seed=12)
    source = save_packed_row_shard(shard, tmp_path / "source" / "shard_00000")
    link_dir = tmp_path / "lexical alias"
    link_dir.mkdir()
    link = link_dir / "linked shard.safetensors.zst"
    link.symlink_to(source)

    owner_calls: list[Path] = []
    popen_calls: list[list[str]] = []
    real_owner_load = dataloader_module._shared_decode_load_owner_shard
    real_popen = dataloader_module.subprocess.Popen

    def owner_load(
        path: Path,
        *,
        tee_path: Path | None = None,
    ) -> PackedRowShard:
        assert tee_path is None
        owner_calls.append(path)
        return real_owner_load(path)

    def popen(args, **kwargs):
        popen_calls.append(args)
        return real_popen(args, **kwargs)

    monkeypatch.setattr(dataloader_module, "_shared_decode_load_owner_shard", owner_load)
    monkeypatch.setattr(dataloader_module.subprocess, "Popen", popen)
    with monkeypatch.context() as no_resolve:
        no_resolve.setattr(
            Path,
            "resolve",
            lambda *_args, **_kwargs: pytest.fail("Path.resolve() touched owner setup"),
        )
        loaded = _shared_decode_load_base_shard(link, 1, 0, 1)

    _assert_shards_equal(loaded, shard)
    assert loaded.backing_segment is not None
    normalized = Path(os.path.abspath(os.fspath(link)))
    assert owner_calls == [normalized]
    assert len(popen_calls) == 1
    assert popen_calls[0] == [
        sys.executable,
        os.fspath(_SHARED_DECODE_WORKER_PATH),
        os.fspath(normalized),
    ]


def test_owner_blocked_open_times_out_before_segment_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"fifo{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.25)
    fifo = tmp_path / "blocked.safetensors.zst"
    os.mkfifo(fifo)
    segment_name = _shared_name(fifo, 1)
    registry_before = dict(_shared_decode_segments)

    start = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_base_shard(fifo, 1, 0, 1)
    elapsed = time.monotonic() - start

    message = str(error.value)
    assert 0.20 < elapsed < 3.0
    assert os.fspath(fifo) in message
    assert "timed out after" in message
    assert "limit 0.250s" in message
    assert "helper_pid=" in message
    assert "helper_reaped=true" in message
    assert _shared_decode_segments == registry_before
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=segment_name, track=False)


def test_owner_partial_fifo_progress_still_times_out_without_publication(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLURM_JOB_ID", f"partial{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.35)
    shard = _tiny_shard(seed=13)
    source = save_packed_row_shard(shard, tmp_path / "source")
    compressed = source.read_bytes()
    fifo = tmp_path / "partial.safetensors.zst"
    os.mkfifo(fifo)
    segment_name = _shared_name(fifo, 1)
    writer_opened = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def write_prefix() -> None:
        try:
            with fifo.open("wb", buffering=0) as stream:
                writer_opened.set()
                stream.write(compressed[: max(1, len(compressed) // 2)])
                release_writer.wait(timeout=5)
        except BaseException as exc:
            writer_errors.append(exc)

    writer = threading.Thread(target=write_prefix)
    writer.start()
    try:
        with pytest.raises(RuntimeError, match="shared decode owner timed out"):
            _shared_decode_load_base_shard(fifo, 1, 0, 1)
        assert writer_opened.is_set()
    finally:
        release_writer.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert not writer_errors, writer_errors
    assert not _shared_decode_segments
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=segment_name, track=False)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_owner_worker_failure_is_loud_without_fallback_or_segment(
    tmp_path,
    monkeypatch,
    failure,
):
    monkeypatch.setenv("SLURM_JOB_ID", f"failure{os.getpid()}{failure}")
    path = tmp_path / f"{failure}.safetensors.zst"
    if failure == "corrupt":
        path.write_bytes(b"not a zstd frame")
    monkeypatch.setattr(
        dataloader_module,
        "load_packed_row_shard",
        lambda _path: pytest.fail("owner failure fell back to canonical loading"),
    )
    segment_name = _shared_name(path, 1)

    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_base_shard(path, 1, 0, 1)

    message = str(error.value)
    assert os.fspath(path) in message
    assert "after " in message
    assert "exit_code=" in message
    assert "stderr_tail=" in message
    assert "timed out" not in message
    assert not _shared_decode_segments
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=segment_name, track=False)


def test_owner_discards_partial_stdout_from_nonzero_worker(tmp_path, monkeypatch):
    shard = _tiny_shard(seed=15)
    source = save_packed_row_shard(shard, tmp_path / "source")
    decoded = zstandard.ZstdDecompressor().decompress(source.read_bytes())

    class FailedProcess:
        returncode = 7
        pid = 1234

        def communicate(self, *, timeout):
            assert timeout > 0
            return decoded, b"synthetic worker failure"

    parse_calls = 0

    def forbidden_parse(_decoded: bytes) -> PackedRowShard:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("nonzero worker stdout was parsed")

    monkeypatch.setattr(
        dataloader_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(
        dataloader_module,
        "load_packed_row_shard_safetensors_bytes",
        forbidden_parse,
    )

    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_owner_shard(source)
    assert "exit_code=7" in str(error.value)
    assert "synthetic worker failure" in str(error.value)
    assert parse_calls == 0


def test_owner_safetensors_failure_preserves_type_and_prepublication(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLURM_JOB_ID", f"safetensors{os.getpid()}")
    shard = _tiny_shard(seed=16)
    incomplete = {
        key: value
        for key, value in shard.__dict__.items()
        if key != "feature_pcts" and isinstance(value, t.Tensor)
    }
    path = save_overlay_shard(incomplete, tmp_path / "incomplete-packed")
    segment_name = _shared_name(path, 1)

    with pytest.raises(KeyError) as error:
        _shared_decode_load_base_shard(path, 1, 0, 1)

    assert error.value.args == ("feature_pcts",)
    assert any(os.fspath(path) in note for note in error.value.__notes__)
    assert not _shared_decode_segments
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=segment_name, track=False)


def test_owner_launch_failure_is_contextual_and_prepublication(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"launch{os.getpid()}")
    relative = Path(os.path.relpath(tmp_path / "launch.safetensors.zst"))
    normalized = Path(os.path.abspath(os.fspath(relative)))

    def fail_launch(*_args, **_kwargs):
        raise OSError("exec unavailable")

    monkeypatch.setattr(dataloader_module.subprocess, "Popen", fail_launch)
    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_base_shard(relative, 1, 0, 1)

    message = str(error.value)
    assert os.fspath(normalized) in message
    assert "failed to launch helper" in message
    assert "after " in message
    assert isinstance(error.value.__cause__, OSError)
    assert not _shared_decode_segments


def test_owner_unreaped_timeout_cleanup_is_bounded_and_not_retried(
    tmp_path,
    monkeypatch,
):
    class FakePipe:
        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    class FakeProcess:
        pid = 424242

        def __init__(self) -> None:
            self.stdout = FakePipe()
            self.stderr = FakePipe()
            self.communicate_calls = 0
            self.kill_calls = 0
            self.wait_calls: list[float] = []

        def communicate(self, *, timeout):
            self.communicate_calls += 1
            raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self) -> None:
            self.kill_calls += 1
            raise PermissionError("synthetic kill failure")

        def wait(self, *, timeout):
            self.wait_calls.append(timeout)
            raise subprocess.TimeoutExpired("worker", timeout)

    fake = FakeProcess()
    popen_calls = 0

    def fake_popen(*_args, **_kwargs):
        nonlocal popen_calls
        popen_calls += 1
        return fake

    monkeypatch.setattr(dataloader_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.1)
    monkeypatch.setattr(
        dataloader_module,
        "load_packed_row_shard",
        lambda _path: pytest.fail("owner timeout retried through canonical loading"),
    )
    path = tmp_path / "blocked.safetensors.zst"

    start = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_owner_shard(path)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert "helper_reaped=false" in str(error.value)
    assert popen_calls == 1
    assert fake.communicate_calls == 1
    assert fake.kill_calls == 1
    assert fake.wait_calls == [1.0]
    assert fake.stdout.close_calls == fake.stderr.close_calls == 1
    assert any("kill cleanup failed" in note for note in error.value.__notes__)


def test_owner_launch_time_is_subtracted_before_communicate(tmp_path, monkeypatch):
    class FakePipe:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        pid = 434343

        def __init__(self) -> None:
            self.stdout = FakePipe()
            self.stderr = FakePipe()
            self.communicate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        def communicate(self, *, timeout):
            self.communicate_calls += 1
            raise AssertionError(f"communicate called with exhausted timeout {timeout}")

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, *, timeout):
            assert timeout == 1.0
            self.wait_calls += 1
            return 0

    fake = FakeProcess()

    def slow_popen(*_args, **_kwargs):
        time.sleep(0.08)
        return fake

    monkeypatch.setattr(dataloader_module.subprocess, "Popen", slow_popen)
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.05)

    with pytest.raises(RuntimeError) as error:
        _shared_decode_load_owner_shard(tmp_path / "slow-launch.safetensors.zst")

    assert "limit 0.050s" in str(error.value)
    assert "helper_reaped=true" in str(error.value)
    assert fake.communicate_calls == 0
    assert fake.kill_calls == 1
    assert fake.wait_calls == 1


def test_attacher_timeout_falls_back_to_direct_decode(tmp_path, monkeypatch):
    """An owner that never publishes must NOT be fatal: unequal rank stripes
    (rows_per_shard % local participation != 0, e.g. world 31) make per-rank
    resume cursors legitimately diverge, so a rank ahead of the cursor never
    visits — and never publishes — a shard a behind-rank still needs. The old
    fatal timeout deadlocked every w31 resume (jobs 21388758/21390294,
    2026-08-27); the consumer now warns and direct-decodes, bitwise-equal.
    Its deadline remains independent of the deliberately shorter owner bound."""
    monkeypatch.setenv("SLURM_JOB_ID", f"tmo{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        "steering.conditional.encoder_decoder.dataloader._SHARED_DECODE_TIMEOUT_S",
        0.3,
    )
    shard = _tiny_shard(seed=4)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    owner = _owner_for(path, 1, 2)
    attacher = (owner + 1) % 2
    canonical_load = dataloader_module.load_packed_row_shard
    direct_calls: list[Path] = []

    def direct_load(candidate: Path) -> PackedRowShard:
        direct_calls.append(candidate)
        return canonical_load(candidate)

    monkeypatch.setattr(dataloader_module, "load_packed_row_shard", direct_load)
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_owner_shard",
        lambda _path: pytest.fail("an attacher invoked the owner helper"),
    )
    start = time.monotonic()
    loaded = _shared_decode_load_base_shard(path, 1, attacher, 2)
    elapsed = time.monotonic() - start
    _assert_shards_equal(loaded, shard)
    assert direct_calls == [path]
    assert 0.25 < elapsed < 3.0


def test_attacher_retries_through_empty_segment_window(tmp_path, monkeypatch):
    """The owner's SharedMemory(create=True) opens the segment file BEFORE
    ftruncating it to size; an attacher landing in that window gets
    ValueError("cannot mmap an empty file") from mmap. Observed killing 3
    consumer ranks at the first w31 shard boundary in production (job
    21333055, 2026-08-26). The attach poll must treat it as not-published-yet,
    exactly like FileNotFoundError."""
    monkeypatch.setenv("SLURM_JOB_ID", f"race{os.getpid()}")
    shard = _tiny_shard(seed=5)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    local_world = 2
    owner = _owner_for(path, 1, local_world)
    attacher = (owner + 1) % local_world
    name = _shared_name(path, 1)

    # Reproduce the window deterministically: the segment name exists with
    # zero bytes (post-open, pre-ftruncate). SharedMemory attach on it raises
    # ValueError, not FileNotFoundError.
    empty = Path("/dev/shm") / name
    empty.touch()
    with pytest.raises(ValueError, match="cannot mmap an empty file"):
        shared_memory.SharedMemory(name=name, track=False)

    publish_error: list[BaseException] = []

    def _publish_later() -> None:
        try:
            time.sleep(0.3)
            empty.unlink()
            segment = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=_shared_decode_payload_size(shard),
                track=False,
            )
            _shared_decode_publish(segment, shard)
        except BaseException as exc:  # surfaced after join
            publish_error.append(exc)

    publisher = threading.Thread(target=_publish_later)
    publisher.start()
    try:
        loaded = _shared_decode_load_base_shard(path, 1, attacher, local_world)
        _assert_shards_equal(loaded, shard)
    finally:
        publisher.join()
        try:
            leftover = shared_memory.SharedMemory(name=name, track=False)
            leftover.close()
            leftover.unlink()
        except (FileNotFoundError, ValueError):
            pass
    assert not publish_error, publish_error


def test_local_topology_helper(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    assert _shared_decode_local_topology() is None
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert _shared_decode_local_topology() is None
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert _shared_decode_local_topology() == (3, 8)


def test_run_namespace_chain(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "sj1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "te1")
    assert _shared_decode_run_namespace() == "sj1"
    monkeypatch.delenv("SLURM_JOB_ID")
    assert _shared_decode_run_namespace() == "te1"
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    assert _shared_decode_run_namespace() == str(os.getppid())


# ---------------------------------------------------------------------------
# Shared decode: cross-process bitwise identity
# ---------------------------------------------------------------------------

# Runs in a fresh interpreter per local rank (subprocess, not
# multiprocessing.spawn: pytest-spawned children re-import the whole test
# session context and take minutes; a direct interpreter imports the loader
# module in seconds). The owner must outlive the attacher's attach — its
# atexit hook unlinks the segment name — so the attacher touches an ack file
# once done and the owner holds until it appears. Prints OK/FAIL on the last
# line.
_TWO_RANK_WORKER_SNIPPET = """
import sys, time
shard_path, local_rank, is_owner, ack = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4]
)
import torch as t
from steering.conditional.encoder_decoder.dataloader import (
    _shared_decode_load_base_shard,
)
from steering.packed_batch_io import load_packed_row_shard
from pathlib import Path
keys = [
    "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
    "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
]
loaded = _shared_decode_load_base_shard(Path(shard_path), 1, local_rank, 2)
reference = load_packed_row_shard(Path(shard_path))
equal = all(t.equal(getattr(loaded, k), getattr(reference, k)) for k in keys)
if is_owner:
    deadline = time.monotonic() + 120
    while not Path(ack).exists() and time.monotonic() < deadline:
        time.sleep(0.1)
else:
    Path(ack).touch()
print("OK" if equal else "FAIL")
"""

_TWO_RANK_TIMEOUT_SNIPPET = """
import sys
from pathlib import Path

import torch as t
from steering.conditional.encoder_decoder import dataloader as loader
from steering.packed_batch_io import load_packed_row_shard

shard_path, local_rank, is_owner, slow_worker = (
    Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1", Path(sys.argv[4])
)
keys = [
    "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
    "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
]
if is_owner:
    loader._SHARED_DECODE_OWNER_TIMEOUT_S = 0.25
    loader._SHARED_DECODE_WORKER_PATH = slow_worker
    try:
        loader._shared_decode_load_base_shard(shard_path, 1, local_rank, 2)
    except RuntimeError as exc:
        message = str(exc)
        ok = (
            "shared decode owner timed out" in message
            and str(shard_path) in message
            and "helper_pid=" in message
        )
        print("OWNER_TIMEOUT" if ok else f"OWNER_BAD:{message}")
    else:
        print("OWNER_DID_NOT_TIMEOUT")
else:
    loader._SHARED_DECODE_TIMEOUT_S = 0.6
    def forbidden_owner_helper(_path):
        raise AssertionError("consumer invoked owner helper")
    loader._shared_decode_load_owner_shard = forbidden_owner_helper
    loaded = loader._shared_decode_load_base_shard(shard_path, 1, local_rank, 2)
    reference = load_packed_row_shard(shard_path)
    equal = all(t.equal(getattr(loaded, key), getattr(reference, key)) for key in keys)
    print("CONSUMER_OK" if equal else "CONSUMER_BAD")
"""

_TWO_RANK_NODE_CACHE_SNIPPET = """
import hashlib, json, os, sys, time
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder import dataloader as loader
from steering.node_shard_cache import NodeLocalShardCache, NodeShardCacheConfig
from steering.refined_types import NonNegativeInt, PositiveInt

shard_path, local_rank, cache_dir, barrier_dir = (
    Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
)
cache = NodeLocalShardCache(NodeShardCacheConfig(
    dir=cache_dir,
    max_bytes=PositiveInt.parse(10_000_000),
    reserve_bytes=NonNegativeInt.parse(0),
))
worker_sources = []
real_popen = loader.subprocess.Popen
def observed_popen(args, **kwargs):
    worker_sources.append(os.path.abspath(args[2]))
    return real_popen(args, **kwargs)
loader.subprocess.Popen = observed_popen

keys = [
    "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
    "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
]
def digest(shard):
    h = hashlib.sha256()
    for key in keys:
        h.update(getattr(shard, key).contiguous().numpy().tobytes())
    return h.hexdigest()
def barrier(label):
    (barrier_dir / f"{label}_{local_rank}").touch()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if all((barrier_dir / f"{label}_{rank}").exists() for rank in (0, 1)):
            return
        time.sleep(0.05)
    raise RuntimeError(f"barrier timed out: {label}")

digests = []
for epoch in (1, 2):
    shard = loader._shared_decode_load_base_shard(
        shard_path, epoch, local_rank, 2, node_cache=cache
    )
    digests.append(digest(shard))
    barrier(f"epoch{epoch}")
barrier("done")
print("RESULT " + json.dumps({
    "digests": digests,
    "counters": cache.counters(),
    "worker_sources": worker_sources,
}))
"""


def test_two_process_bitwise_roundtrip(tmp_path):
    shard = _tiny_shard(seed=5)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    owner = _owner_for(path, 1, 2)
    ack = tmp_path / "attacher.ack"
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = f"xproc{os.getpid()}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    procs = [
        subprocess.Popen(
            [
                sys.executable, "-c", _TWO_RANK_WORKER_SNIPPET,
                str(path), str(rank), "1" if rank == owner else "0", str(ack),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=240)
        outputs.append((proc.returncode, stdout.strip().splitlines()[-1:], stderr[-800:]))
    assert all(rc == 0 for rc, _, _ in outputs), outputs
    assert all(last == ["OK"] for _, last, _ in outputs), outputs


def test_two_process_shared_decode_tees_then_hits_shared_node_cache(tmp_path):
    shard = _tiny_shard(seed=31)
    source_dir = tmp_path / "source"
    for index in range(1000):
        candidate = source_dir / f"shard_{index:05d}.safetensors.zst"
        if _owner_for(candidate, 1, 2) != _owner_for(candidate, 2, 2):
            path = save_packed_row_shard(shard, source_dir / f"shard_{index:05d}")
            break
    else:  # pragma: no cover - probability 2^-1000
        raise AssertionError("could not find a shard path with different epoch owners")
    cache_dir = tmp_path / "node-cache"
    barrier_dir = tmp_path / "barriers"
    barrier_dir.mkdir()
    expected_hasher = hashlib.sha256()
    for key in SHARD_KEYS:
        expected_hasher.update(getattr(shard, key).contiguous().numpy().tobytes())
    expected_digest = expected_hasher.hexdigest()
    owners = {epoch: _owner_for(path, epoch, 2) for epoch in (1, 2)}
    assert owners[1] != owners[2]
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = f"xproccache{os.getpid()}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    procs = {}
    for rank in (0, 1):
        rank_env = dict(env)
        rank_env["LOCAL_RANK"] = str(rank)
        rank_env["LOCAL_WORLD_SIZE"] = "2"
        procs[rank] = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _TWO_RANK_NODE_CACHE_SNIPPET,
                os.fspath(path),
                str(rank),
                os.fspath(cache_dir),
                os.fspath(barrier_dir),
            ],
            env=rank_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    results = {}
    for rank, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"rank {rank} failed: {stderr[-2000:]}"
        assert "falling back to direct decode" not in stderr, stderr[-2000:]
        line = next(row for row in stdout.splitlines() if row.startswith("RESULT "))
        results[rank] = json.loads(line.removeprefix("RESULT "))

    for rank in (0, 1):
        assert results[rank]["digests"] == [expected_digest, expected_digest]
    assert results[owners[1]]["counters"] == {
        "hits": 0,
        "misses": 1,
        "fills": 1,
        "fill_skips": 0,
        "verify_failures": 0,
        "fallbacks": 0,
    }
    assert results[owners[2]]["counters"] == {
        "hits": 1,
        "misses": 0,
        "fills": 0,
        "fill_skips": 0,
        "verify_failures": 0,
        "fallbacks": 0,
    }
    canonical = os.path.abspath(path)
    assert results[owners[1]]["worker_sources"][0] == canonical
    epoch_two_owner_sources = results[owners[2]]["worker_sources"]
    epoch_two_source = epoch_two_owner_sources[-1]
    assert Path(epoch_two_source).parent == cache_dir
    assert epoch_two_source != canonical


def test_two_process_owner_timeout_preserves_consumer_direct_fallback(tmp_path):
    shard = _tiny_shard(seed=14)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    owner = _owner_for(path, 1, 2)
    slow_worker = tmp_path / "slow_worker.py"
    slow_worker.write_text("import time\ntime.sleep(10)\n")
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = f"xproctimeout{os.getpid()}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _TWO_RANK_TIMEOUT_SNIPPET,
                os.fspath(path),
                str(rank),
                "1" if rank == owner else "0",
                os.fspath(slow_worker),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in (0, 1)
    ]
    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        outputs.append((proc.returncode, stdout.strip().splitlines()[-1:], stderr))

    assert all(returncode == 0 for returncode, _, _ in outputs), outputs
    owner_output = outputs[owner]
    consumer_output = outputs[(owner + 1) % 2]
    assert owner_output[1] == ["OWNER_TIMEOUT"], outputs
    assert consumer_output[1] == ["CONSUMER_OK"], outputs
    assert "falling back to direct decode" in consumer_output[2]
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=_shared_name(path, 1), track=False)
