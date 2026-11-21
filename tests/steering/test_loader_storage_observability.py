"""Loader storage-health accounting (2026-08-28 hstor013 incidents).

During the incidents, successful-but-slow shard loads were invisible: the
only signals were consumer fallback WARNs and, past the NCCL timeout,
watchdog kills. These tests pin the new accounting: every base-shard load
path (owner decode / shared attach / direct) records a duration, fallbacks
are counted, slow loads WARN even when they succeed, and a periodic one-line
summary aggregates it all.
"""

import hashlib
import os
import threading
import time
from io import StringIO
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import torch as t
from loguru import logger

import steering.conditional.encoder_decoder.dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import (
    _loader_health_record,
    _loader_health_reset,
    _shared_decode_load_base_shard,
    _shared_decode_payload_size,
    _shared_decode_publish,
    _shared_decode_run_namespace,
    _shared_decode_segments,
    format_loader_health,
    loader_health_delta,
    loader_health_scope,
    loader_health_snapshot,
)
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard

SHARD_KEYS = [
    "tokens",
    "row_seg_offsets",
    "seg_token_start",
    "seg_token_end",
    "seg_doc_id",
    "seg_feat_offsets",
    "feature_ids",
    "feature_pcts",
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
        row_seg_offsets=t.from_numpy((np.arange(rows + 1) * segs_per_row).astype(np.int32)),
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
def _isolated_state():
    _clear_registry()
    _loader_health_reset()
    yield
    _clear_registry()
    _loader_health_reset()


def _capture(level: str = "DEBUG"):
    sink = StringIO()
    handler_id = logger.add(sink, format="{level} | {message}", level=level)
    return sink, handler_id


def test_owner_decode_records_health(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-owner{os.getpid()}")
    shard = _tiny_shard(seed=1)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    loaded = _shared_decode_load_base_shard(path, 1, 0, 1)
    _assert_shards_equal(loaded, shard)
    health = loader_health_snapshot()
    assert health["base_loads"] == 1
    assert health["owner_decodes"] == 1
    assert health["shared_attaches"] == 0
    assert health["fallbacks"] == 0
    assert health["owner_decode_s_max"] > 0.0
    assert health["owner_decode_s_sum"] >= health["owner_decode_s_max"]


def test_slow_owner_decode_warns_even_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-slowown{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_SLOW_WARN_S", 0.0)
    shard = _tiny_shard(seed=2)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    sink, handler_id = _capture()
    try:
        loaded = _shared_decode_load_base_shard(path, 1, 0, 1)
    finally:
        logger.remove(handler_id)
    _assert_shards_equal(loaded, shard)
    captured = sink.getvalue()
    assert "shared decode owner slow" in captured
    assert "storage degradation likely" in captured
    assert loader_health_snapshot()["owner_decodes"] == 1


def test_owner_decode_outlives_consumer_timeout_and_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-own-bound{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_OWNER_TIMEOUT_S", 2.0)
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_SLOW_WARN_S", 0.05)
    shard = _tiny_shard(seed=21)
    source = save_packed_row_shard(shard, tmp_path / "source")
    compressed = source.read_bytes()
    fifo = tmp_path / "delayed.safetensors.zst"
    os.mkfifo(fifo)
    writer_errors: list[BaseException] = []

    def write_after_consumer_timeout() -> None:
        try:
            time.sleep(0.15)
            with fifo.open("wb", buffering=0) as stream:
                stream.write(compressed)
        except BaseException as exc:
            writer_errors.append(exc)

    writer = threading.Thread(target=write_after_consumer_timeout, daemon=True)
    writer.start()
    sink, handler_id = _capture()
    start = time.monotonic()
    try:
        loaded = _shared_decode_load_base_shard(fifo, 1, 0, 1)
    finally:
        logger.remove(handler_id)
        writer.join(timeout=5)
    elapsed = time.monotonic() - start

    assert not writer.is_alive()
    assert not writer_errors, writer_errors
    assert elapsed > 0.1
    _assert_shards_equal(loaded, shard)
    assert "shared decode owner slow" in sink.getvalue()


def test_fallback_counts_and_direct_load_recorded(tmp_path, monkeypatch):
    """An attacher whose owner never publishes must count a fallback AND a
    direct load, and still produce a bitwise-equal shard."""
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-fb{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_TIMEOUT_S", 0.2)
    shard = _tiny_shard(seed=3)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    owner = _owner_for(path, 1, 2)
    attacher = (owner + 1) % 2
    sink, handler_id = _capture()
    try:
        loaded = _shared_decode_load_base_shard(path, 1, attacher, 2)
    finally:
        logger.remove(handler_id)
    _assert_shards_equal(loaded, shard)
    health = loader_health_snapshot()
    assert health["base_loads"] == 1
    assert health["fallbacks"] == 1
    assert health["direct_loads"] == 1
    assert health["owner_decodes"] == 0
    assert health["shared_attaches"] == 0
    assert health["direct_load_s_max"] > 0.0
    assert "falling back to direct decode" in sink.getvalue()


def test_attach_wait_recorded_and_warns_when_slow(tmp_path, monkeypatch):
    """A publish that lands late-but-in-time must record the attach wait and
    WARN past the slow threshold — the silent 60-119s waits of the incident."""
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-wait{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_SLOW_WARN_S", 0.05)
    shard = _tiny_shard(seed=4)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    local_world = 2
    owner = _owner_for(path, 1, local_world)
    attacher = (owner + 1) % local_world
    name = _shared_name(path, 1)
    publish_error: list[BaseException] = []

    def _publish_later() -> None:
        try:
            time.sleep(0.3)
            segment = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=_shared_decode_payload_size(shard),
                track=False,
            )
            _shared_decode_publish(segment, shard)
            segment.close()
        except BaseException as exc:  # surfaced via the assertion below
            publish_error.append(exc)

    publisher = threading.Thread(target=_publish_later, daemon=True)
    publisher.start()
    sink, handler_id = _capture()
    try:
        loaded = _shared_decode_load_base_shard(path, 1, attacher, local_world)
    finally:
        logger.remove(handler_id)
        publisher.join(timeout=5.0)
    assert not publish_error, publish_error
    _assert_shards_equal(loaded, shard)
    health = loader_health_snapshot()
    assert health["shared_attaches"] == 1
    assert health["fallbacks"] == 0
    assert health["attach_wait_s_max"] >= 0.05
    captured = sink.getvalue()
    assert "shared decode attach waited" in captured
    assert "storage degradation likely" in captured
    try:  # the attacher never owns the segment; unlink the publisher's name
        stale = shared_memory.SharedMemory(name=name, track=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass


def test_summary_line_cadence_and_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-sum{os.getpid()}")
    monkeypatch.setattr(dataloader_module, "_LOADER_HEALTH_SUMMARY_EVERY", 2)
    sink, handler_id = _capture(level="INFO")
    try:
        for i in range(4):
            shard = _tiny_shard(seed=10 + i)
            path = save_packed_row_shard(shard, tmp_path / f"shard_{i:05d}")
            _shared_decode_load_base_shard(path, 1, 0, 1)
    finally:
        logger.remove(handler_id)
    summaries = [line for line in sink.getvalue().splitlines() if "loader health:" in line]
    assert len(summaries) == 2
    assert "loads=2 owner=2 attach=0 direct=0 fallbacks=0" in summaries[0]
    assert "loads=4 owner=4 attach=0 direct=0 fallbacks=0" in summaries[1]
    assert "owner_decode=" in summaries[1]
    assert "attach_wait=-" in summaries[1]


def test_snapshot_returns_independent_copy():
    before = loader_health_snapshot()
    before["base_loads"] = 999
    assert loader_health_snapshot()["base_loads"] == 0


def test_scope_and_global_health_share_exact_record_semantics() -> None:
    with loader_health_scope() as scoped:
        _loader_health_record("owner_decode", 1.25)
        _loader_health_record("shared_attach", 2.5)
        _loader_health_record("direct_load", 3.75, fallback=True)

    expected = {
        "base_loads": 3.0,
        "owner_decodes": 1.0,
        "shared_attaches": 1.0,
        "direct_loads": 1.0,
        "fallbacks": 1.0,
        "owner_decode_s_sum": 1.25,
        "owner_decode_s_max": 1.25,
        "attach_wait_s_sum": 2.5,
        "attach_wait_s_max": 2.5,
        "direct_load_s_sum": 3.75,
        "direct_load_s_max": 3.75,
    }
    assert scoped == expected
    assert loader_health_snapshot() == expected


def test_scope_is_independent_from_global_snapshot_and_later_records() -> None:
    with loader_health_scope() as scoped:
        _loader_health_record("direct_load", 4.0)
    scoped["base_loads"] = 999.0
    _loader_health_record("owner_decode", 5.0)

    global_health = loader_health_snapshot()
    assert global_health["base_loads"] == 2.0
    assert global_health["direct_loads"] == 1.0
    assert global_health["owner_decodes"] == 1.0
    assert scoped["base_loads"] == 999.0
    assert scoped["owner_decodes"] == 0.0


def test_scope_excludes_concurrent_background_loader_records() -> None:
    background_started = threading.Event()
    allow_background_record = threading.Event()
    background_done = threading.Event()

    def record_training_prefetch() -> None:
        background_started.set()
        assert allow_background_record.wait(timeout=5.0)
        _loader_health_record("direct_load", 7.0, fallback=True)
        background_done.set()

    thread = threading.Thread(
        target=record_training_prefetch,
        name="simulated-training-prefetch",
        daemon=True,
    )
    thread.start()
    assert background_started.wait(timeout=5.0)

    with loader_health_scope() as validation_health:
        allow_background_record.set()
        assert background_done.wait(timeout=5.0)
        _loader_health_record("shared_attach", 1.5)
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert validation_health["base_loads"] == 1.0
    assert validation_health["shared_attaches"] == 1.0
    assert validation_health["direct_loads"] == 0.0
    assert validation_health["fallbacks"] == 0.0

    global_health = loader_health_snapshot()
    assert global_health["base_loads"] == 2.0
    assert global_health["shared_attaches"] == 1.0
    assert global_health["direct_loads"] == 1.0
    assert global_health["fallbacks"] == 1.0


def test_scope_cleanup_preserves_original_exception_and_allows_reentry() -> None:
    original = RuntimeError("validation failed after a recorded load")
    with pytest.raises(RuntimeError) as raised:
        with loader_health_scope() as failed_health:
            _loader_health_record("direct_load", 8.0)
            raise original

    assert raised.value is original
    assert failed_health["base_loads"] == 1.0
    with loader_health_scope() as next_health:
        assert next_health["base_loads"] == 0.0
        _loader_health_record("owner_decode", 0.5)
    assert next_health["owner_decodes"] == 1.0


def test_nested_scopes_fail_before_replacing_outer_accounting() -> None:
    with loader_health_scope() as outer:
        _loader_health_record("owner_decode", 1.0)
        with pytest.raises(RuntimeError, match="may not be nested"):
            with loader_health_scope():
                pass
        _loader_health_record("owner_decode", 2.0)

    assert outer["base_loads"] == 2.0
    assert outer["owner_decodes"] == 2.0
    assert outer["owner_decode_s_sum"] == 3.0
    assert outer["owner_decode_s_max"] == 2.0


def test_additive_delta_does_not_claim_a_cumulative_maximum() -> None:
    _loader_health_record("direct_load", 10.0)
    previous = loader_health_snapshot()
    _loader_health_record("direct_load", 2.0, fallback=True)
    current = loader_health_snapshot()

    delta = loader_health_delta(previous, current)
    assert delta == {
        "base_loads": 1.0,
        "owner_decodes": 0.0,
        "shared_attaches": 0.0,
        "direct_loads": 1.0,
        "fallbacks": 1.0,
        "owner_decode_s_sum": 0.0,
        "attach_wait_s_sum": 0.0,
        "direct_load_s_sum": 2.0,
    }
    assert current["direct_load_s_max"] == 10.0
    assert all(not key.endswith("_max") for key in delta)
    assert format_loader_health(delta, include_max=False) == (
        "loads=1 owner=0 attach=0 direct=1 fallbacks=1 | "
        "mean s: owner_decode=- attach_wait=- direct=2.0s"
    )


def test_delta_rejects_a_counter_reset_instead_of_logging_negative_health() -> None:
    _loader_health_record("owner_decode", 3.0)
    previous = loader_health_snapshot()
    _loader_health_reset()

    with pytest.raises(ValueError, match="moved backwards"):
        loader_health_delta(previous, loader_health_snapshot())


def test_exact_formatter_reports_means_maxima_and_empty_categories() -> None:
    with loader_health_scope() as scoped:
        _loader_health_record("owner_decode", 1.0)
        _loader_health_record("owner_decode", 3.0)
        _loader_health_record("direct_load", 4.0, fallback=True)

    assert format_loader_health(scoped, include_max=True) == (
        "loads=3 owner=2 attach=0 direct=1 fallbacks=1 | "
        "mean/max s: owner_decode=2.0/3.0s attach_wait=- direct=4.0/4.0s"
    )


def test_real_owner_load_is_bitwise_equal_and_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", f"obs-scoped-owner{os.getpid()}")
    shard = _tiny_shard(seed=99)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")

    with loader_health_scope() as scoped:
        loaded = _shared_decode_load_base_shard(path, 1, 0, 1)

    _assert_shards_equal(loaded, shard)
    assert scoped["base_loads"] == 1.0
    assert scoped["owner_decodes"] == 1.0
    assert scoped["owner_decode_s_sum"] > 0.0
    assert scoped["owner_decode_s_max"] == scoped["owner_decode_s_sum"]
