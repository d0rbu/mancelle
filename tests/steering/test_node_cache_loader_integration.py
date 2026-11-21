"""Precomputed-loader composition tests for the node-local shard cache."""

import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.node_shard_cache import NodeLocalShardCache, NodeShardCacheConfig
from steering.refined_types import NonNegativeInt, PositiveInt
from tests.steering.test_precomputed_producer_perf import _assert_shards_equal
from tests.steering.test_precomputed_shard_lookahead import (
    _assert_batches_equal,
    _build_pipeline,
    _write_shards,
)


def _cache(root: Path, *, max_bytes: int = 10_000_000) -> NodeLocalShardCache:
    return NodeLocalShardCache(
        NodeShardCacheConfig(
            dir=root,
            max_bytes=PositiveInt.parse(max_bytes),
            reserve_bytes=NonNegativeInt.parse(0),
        )
    )


def _offer(cache: NodeLocalShardCache, source: Path) -> None:
    source_stat = os.stat(source)
    cache.offer(source, source.read_bytes(), source_stat)


def _loader(
    packed_dir: Path,
    *,
    node_cache: NodeLocalShardCache | None,
    selection_dir: Path | None = None,
    lookahead: int = 0,
    batch_size: int = 3,
    label_dropout: float = 0.0,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    resume_state_dict: dict | None = None,
) -> Iterator[tuple]:
    return precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=batch_size,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        n_sae_features=512,
        selection_dir=selection_dir,
        label_dropout=label_dropout,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
        train_shard_lookahead=lookahead,
        node_cache=node_cache,
    )


def _drain(loader: Iterator[tuple], count: int) -> list[tuple]:
    with contextlib.closing(loader):
        return [next(loader) for _ in range(count)]


def _assert_streams_equal(actual: list[tuple], expected: list[tuple]) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        _assert_batches_equal(actual_batch, expected_batch)


def _shared_decode_owner(shard_path: Path, logical_epoch: int, local_world: int) -> int:
    normalized = Path(os.path.abspath(os.fspath(shard_path)))
    digest = hashlib.sha256(f"{normalized}:{logical_epoch}".encode()).hexdigest()
    return int(digest, 16) % local_world


def test_cache_off_cold_and_warm_match_across_two_epoch_boundaries_with_selection(
    tmp_path: Path,
) -> None:
    packed_dir, selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=2,
        n_rows=3,
        with_selection=True,
    )
    assert selection_dir is not None
    t.manual_seed(91)
    baseline = _drain(
        _loader(
            packed_dir,
            node_cache=None,
            selection_dir=selection_dir,
            lookahead=2,
            batch_size=4,
            label_dropout=0.35,
        ),
        8,
    )
    assert baseline[-1][9]["epoch"] >= 3

    cold_cache = _cache(tmp_path / "cache")
    t.manual_seed(91)
    cold = _drain(
        _loader(
            packed_dir,
            node_cache=cold_cache,
            selection_dir=selection_dir,
            lookahead=2,
            batch_size=4,
            label_dropout=0.35,
        ),
        8,
    )
    warm_cache = _cache(tmp_path / "cache")
    t.manual_seed(91)
    warm = _drain(
        _loader(
            packed_dir,
            node_cache=warm_cache,
            selection_dir=selection_dir,
            lookahead=2,
            batch_size=4,
            label_dropout=0.35,
        ),
        8,
    )
    _assert_streams_equal(cold, baseline)
    _assert_streams_equal(warm, baseline)
    assert cold_cache.counters()["fills"] >= 4
    assert warm_cache.counters()["hits"] >= 4


def test_cache_preserves_each_ddp_stripe_with_lookahead(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=3,
        n_rows=6,
    )
    cache = _cache(tmp_path / "cache")
    rank_streams: dict[int, list[tuple]] = {}
    for rank in (0, 1):
        t.manual_seed(7)
        baseline = _drain(
            _loader(
                packed_dir,
                node_cache=None,
                lookahead=3,
                ddp_rank=rank,
                ddp_world_size=2,
            ),
            8,
        )
        t.manual_seed(7)
        cached = _drain(
            _loader(
                packed_dir,
                node_cache=cache,
                lookahead=3,
                ddp_rank=rank,
                ddp_world_size=2,
            ),
            8,
        )
        _assert_streams_equal(cached, baseline)
        rank_streams[rank] = cached
    assert rank_streams[0][0][9]["batch_rows"] != rank_streams[1][0][9]["batch_rows"]


def test_exact_resume_is_bitwise_with_warm_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=3,
        n_rows=5,
        with_selection=True,
    )
    assert selection_dir is not None
    monkeypatch.setattr(
        dataloader_module,
        "_validate_generic_selection_dir",
        lambda **_kwargs: 20,
    )
    t.manual_seed(19)
    baseline = _drain(
        _loader(
            packed_dir,
            node_cache=None,
            selection_dir=selection_dir,
            label_dropout=0.2,
        ),
        8,
    )
    cache = _cache(tmp_path / "cache")
    t.manual_seed(19)
    _drain(
        _loader(
            packed_dir,
            node_cache=cache,
            selection_dir=selection_dir,
            label_dropout=0.2,
        ),
        5,
    )
    state = json.loads(json.dumps(baseline[3][9]))
    warm_cache = _cache(tmp_path / "cache")
    real_read_bytes = Path.read_bytes
    source_reads: list[Path] = []
    base_loads: list[NodeLocalShardCache | None] = []
    selection_loads: list[NodeLocalShardCache | None] = []
    real_base_load = dataloader_module._load_packed_row_shard_with_node_cache
    real_selection_load = dataloader_module._load_selection_shard_with_node_cache

    def observed_read(path: Path) -> bytes:
        lexical = Path(os.path.abspath(os.fspath(path)))
        if lexical.parent in (packed_dir, selection_dir):
            source_reads.append(lexical)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed_read)

    def observed_base_load(path: Path, node_cache: NodeLocalShardCache | None):
        base_loads.append(node_cache)
        return real_base_load(path, node_cache)

    def observed_selection_load(path: Path, node_cache: NodeLocalShardCache | None):
        selection_loads.append(node_cache)
        return real_selection_load(path, node_cache)

    monkeypatch.setattr(
        dataloader_module,
        "_load_packed_row_shard_with_node_cache",
        observed_base_load,
    )
    monkeypatch.setattr(
        dataloader_module,
        "_load_selection_shard_with_node_cache",
        observed_selection_load,
    )
    t.manual_seed(19)
    resumed = _drain(
        _loader(
            packed_dir,
            node_cache=warm_cache,
            selection_dir=selection_dir,
            label_dropout=0.2,
            lookahead=2,
            resume_state_dict=state,
        ),
        5,
    )
    _assert_streams_equal(resumed, baseline[3:8])
    assert source_reads == []
    # Exact-resume replay loads both base and selection shards through the
    # cache before continuing the ordinary visit stream.
    assert base_loads == [warm_cache] * 9
    assert selection_loads == [warm_cache] * 9
    assert warm_cache.counters()["hits"] == 18
    assert warm_cache.counters()["misses"] == 0


def test_source_read_detector_falsifies_off_before_warm_and_budget_claims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    n_shards = 4
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=n_shards,
        n_rows=4,
    )
    real_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def observed_read(path: Path) -> bytes:
        lexical = Path(os.path.abspath(os.fspath(path)))
        if lexical.parent == packed_dir:
            reads.append(lexical)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed_read)

    # Negative control first: the detector must see the known uncached N reads.
    _drain(_loader(packed_dir, node_cache=None, batch_size=2), n_shards * 2)
    assert len(reads) == n_shards

    full_cache = _cache(tmp_path / "full-cache")
    reads.clear()
    _drain(_loader(packed_dir, node_cache=full_cache, batch_size=2), n_shards * 2)
    assert len(reads) == n_shards
    reads.clear()
    _drain(_loader(packed_dir, node_cache=full_cache, batch_size=2), n_shards * 2)
    assert reads == []

    source_paths = sorted(packed_dir.glob("*.safetensors.zst"))
    k = 2

    def committed_entry_bytes(path: Path) -> int:
        source_stat = os.stat(path)
        compressed = real_read_bytes(path)
        metadata = full_cache._metadata(
            source_abspath=os.path.abspath(path),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            sha256=hashlib.sha256(compressed).hexdigest(),
            byte_length=len(compressed),
        )
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return len(compressed) + len(metadata_bytes)

    capped_bytes = sum(committed_entry_bytes(path) for path in source_paths[:k])
    capped_cache = _cache(tmp_path / "capped-cache", max_bytes=capped_bytes)
    reads.clear()
    _drain(_loader(packed_dir, node_cache=capped_cache, batch_size=2), n_shards * 2)
    assert len(reads) == n_shards
    reads.clear()
    _drain(_loader(packed_dir, node_cache=capped_cache, batch_size=2), n_shards * 2)
    assert len(reads) == n_shards - k
    assert capped_cache.counters()["fills"] == k


def test_consumer_timeout_fallback_uses_warm_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    cache = _cache(tmp_path / "cache")
    _offer(cache, shard_path)
    epoch = 1
    digest = hashlib.sha256(
        f"{Path(os.path.abspath(os.fspath(shard_path)))}:{epoch}".encode()
    ).hexdigest()
    owner = int(digest, 16) % 2
    attacher = (owner + 1) % 2
    monkeypatch.setattr(dataloader_module, "_SHARED_DECODE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        dataloader_module,
        "load_packed_row_shard",
        lambda _path: pytest.fail("fallback reread the canonical source"),
    )
    loaded = dataloader_module._shared_decode_load_base_shard(
        shard_path,
        epoch,
        attacher,
        2,
        node_cache=cache,
    )
    reference = dataloader_module.load_packed_row_shard_bytes(shard_path.read_bytes())
    _assert_shards_equal(loaded, reference)
    assert cache.counters()["hits"] == 1
    assert cache.counters()["fallbacks"] == 1


def test_precomputed_loader_forwards_same_node_cache_to_shared_decode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    cache = _cache(tmp_path / "cache")
    forwarded: list[NodeLocalShardCache | None] = []

    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_local_topology",
        lambda: (0, 2),
    )

    def recorded_shared_decode(
        shard_path: Path,
        _logical_epoch: int,
        _local_rank: int,
        _local_world: int,
        *,
        retain: int,
        node_cache: NodeLocalShardCache | None = None,
    ):
        del retain
        forwarded.append(node_cache)
        return dataloader_module.load_packed_row_shard(shard_path)

    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_base_shard",
        recorded_shared_decode,
    )
    # The two-process rig covers shared behavior; this seam pins only object
    # forwarding from the loader into the shared-decode call.
    _drain(_loader(packed_dir, node_cache=cache), 1)
    assert forwarded == [cache]


def test_in_process_shared_decode_owner_miss_commits_tee_and_matches_cache_off(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    t.manual_seed(0)
    reference = _drain(_loader(packed_dir, node_cache=None, batch_size=2), 1)[0]
    cache = _cache(tmp_path / "cache")
    owner = _shared_decode_owner(shard_path, logical_epoch=1, local_world=2)
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_local_topology",
        lambda: (owner, 2),
    )

    try:
        t.manual_seed(0)
        actual = _drain(_loader(packed_dir, node_cache=cache, batch_size=2), 1)[0]
    finally:
        dataloader_module._shared_decode_unlink_registered()

    _assert_batches_equal(actual, reference)
    _source, payload, sidecar = cache._entry_paths(shard_path)
    assert payload.is_file()
    assert sidecar.is_file()
    assert cache.counters()["fills"] == 1


def test_in_process_shared_decode_owner_failure_discards_tee_and_propagates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    shard_path.write_bytes(b"not a zstd frame")
    cache = _cache(tmp_path / "cache")
    owner = _shared_decode_owner(shard_path, logical_epoch=1, local_world=2)
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_local_topology",
        lambda: (owner, 2),
    )
    loader = _loader(packed_dir, node_cache=cache, batch_size=2)
    try:
        with pytest.raises(RuntimeError, match="shared decode owner helper failed"):
            next(loader)
    finally:
        loader.close()
        dataloader_module._shared_decode_unlink_registered()

    _source, payload, sidecar = cache._entry_paths(shard_path)
    assert not payload.exists()
    assert not sidecar.exists()
    assert not list(cache.dir.glob("*.tmp.*"))


def test_shared_decode_cache_hit_failure_retries_canonical_source_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    cache = _cache(tmp_path / "cache")
    _offer(cache, shard_path)
    _source, payload, _sidecar = cache._entry_paths(shard_path)
    reference = dataloader_module.load_packed_row_shard(shard_path)
    calls: list[tuple[Path, Path | None]] = []

    def observed_owner_load(path: Path, *, tee_path: Path | None = None):
        calls.append((path, tee_path))
        if path == payload:
            payload.unlink()
            raise RuntimeError("cache payload vanished after verification")
        assert path == Path(os.path.abspath(shard_path))
        assert tee_path is None
        return reference

    class FakeSegment:
        def __init__(self, *, name: str, create: bool = False, **_kwargs):
            if not create:
                raise FileNotFoundError(name)
            self.name = name
            self.shard = None

    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_owner_shard",
        observed_owner_load,
    )
    monkeypatch.setattr(dataloader_module, "_QuietSharedMemory", FakeSegment)
    monkeypatch.setattr(dataloader_module, "_shared_decode_payload_size", lambda _shard: 1)
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_publish",
        lambda segment, shard: setattr(segment, "shard", shard),
    )
    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_wrap",
        lambda segment: segment.shard,
    )
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )
    try:
        loaded = dataloader_module._shared_decode_load_base_shard(
            shard_path,
            1,
            0,
            1,
            node_cache=cache,
        )
    finally:
        logger.remove(sink)
        dataloader_module._shared_decode_segments.clear()

    _assert_shards_equal(loaded, reference)
    assert calls == [(payload, None), (Path(os.path.abspath(shard_path)), None)]
    assert any("retrying once from canonical source" in message for message in messages)
    assert cache.counters()["hits"] == 1
    assert cache.counters()["verify_failures"] == 1
    assert cache.counters()["fallbacks"] == 1


def test_shared_decode_cache_hit_keyboard_interrupt_does_not_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    cache = _cache(tmp_path / "cache")
    _offer(cache, shard_path)
    _source, payload, _sidecar = cache._entry_paths(shard_path)
    calls: list[Path] = []

    def interrupted_owner_load(path: Path, *, tee_path: Path | None = None):
        assert tee_path is None
        calls.append(path)
        raise KeyboardInterrupt("cache path interrupt")

    monkeypatch.setattr(
        dataloader_module,
        "_shared_decode_load_owner_shard",
        interrupted_owner_load,
    )
    with pytest.raises(KeyboardInterrupt, match="cache path interrupt"):
        dataloader_module._shared_decode_load_base_shard(
            shard_path,
            1,
            0,
            1,
            node_cache=cache,
        )
    assert calls == [payload]
    assert cache.counters()["verify_failures"] == 0
    assert cache.counters()["fallbacks"] == 0


def test_shared_decode_real_mid_write_tee_failure_does_not_commit(
    tmp_path: Path,
) -> None:
    if not Path("/dev/full").exists():
        pytest.skip("/dev/full is required for a real mid-write ENOSPC")
    packed_dir, _selection_dir = _write_shards(
        tmp_path / "artifacts",
        n_shards=1,
        n_rows=4,
    )
    shard_path = next(packed_dir.glob("*.safetensors.zst"))
    cache = _cache(tmp_path / "cache")
    fill = cache.reserve_fill_tmp(shard_path)
    assert fill is not None
    fill.tmp_path.symlink_to("/dev/full")

    loaded = dataloader_module._shared_decode_load_owner_shard(
        shard_path,
        tee_path=fill.tmp_path,
    )
    reference = dataloader_module.load_packed_row_shard(shard_path)
    cache.commit_fill_tmp(fill)

    _assert_shards_equal(loaded, reference)
    assert cache.read(shard_path) is None
    assert cache.counters()["fills"] == 0
    assert cache.counters()["verify_failures"] == 1
    assert not fill.tmp_path.exists()


@pytest.mark.parametrize(
    "helper_name",
    ["_load_packed_row_shard_with_node_cache", "_load_selection_shard_with_node_cache"],
)
def test_node_cache_load_helpers_preserve_legacy_suffix_error(
    tmp_path: Path,
    helper_name: str,
) -> None:
    wrong_suffix = tmp_path / "shard.bin"
    wrong_suffix.write_bytes(b"not a shard")
    helper = getattr(dataloader_module, helper_name)
    with pytest.raises(ValueError, match=r"Expected \.safetensors\.zst file, got: shard\.bin"):
        helper(wrong_suffix, _cache(tmp_path / f"cache-{helper_name}"))


@pytest.mark.parametrize("materializer_name", ["packed_materializer", "remote_materializer"])
def test_materializer_with_cache_fails_loudly(
    tmp_path: Path,
    materializer_name: str,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / materializer_name,
        n_shards=1,
        n_rows=2,
    )
    # Deliberately invalid runtime input: exercise the loader's rejection path.
    kwargs: dict[str, Any] = {materializer_name: object()}
    loader = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=1,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        n_sae_features=512,
        node_cache=_cache(tmp_path / f"cache-{materializer_name}"),
        **kwargs,
    )
    with pytest.raises(ValueError, match="node-local shard cache.*local packed-row path"):
        next(loader)
