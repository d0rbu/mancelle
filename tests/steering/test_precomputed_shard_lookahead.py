"""Tests for decoded-shard lookahead in the precomputed packed loader."""

import contextlib
import gc
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch as t

from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import save_packed_row_shard, save_selection_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_identity_selection,
    _make_test_shard,
    build_threshold_pipeline,
)


def _lookahead_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "packed-shard-lookahead"]


def _wait_for_no_lookahead_thread(timeout: float | None = None) -> None:
    if timeout is None:
        timeout = dataloader_module._SHARD_LOOKAHEAD_JOIN_SECONDS + 1.0
    deadline = time.monotonic() + timeout
    while _lookahead_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _lookahead_threads()


@pytest.fixture(autouse=True)
def _no_leaked_lookahead_threads():
    dataloader_module._shared_decode_retain_floor = dataloader_module._SHARED_DECODE_RETAIN
    _wait_for_no_lookahead_thread()
    yield
    _wait_for_no_lookahead_thread()
    dataloader_module._shared_decode_retain_floor = dataloader_module._SHARED_DECODE_RETAIN


def _build_pipeline():
    boundaries = (
        t.linspace(
            1 / NUM_BUCKETS,
            1.0 - 1 / NUM_BUCKETS,
            NUM_BUCKETS - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _write_shards(
    root: Path,
    *,
    n_shards: int,
    n_rows: int,
    with_selection: bool = False,
) -> tuple[Path, Path | None]:
    """Write deterministic packed shards and optional identity selections."""
    packed_dir = root / "packed"
    packed_dir.mkdir(parents=True)
    selection_dir = root / "selection" if with_selection else None
    if selection_dir is not None:
        selection_dir.mkdir(parents=True)

    for shard_idx in range(n_shards):
        shard = _make_test_shard(n_rows=n_rows, seed=17 + 6 * shard_idx)
        name = f"shard_{shard_idx:05d}"
        save_packed_row_shard(shard, packed_dir / name)
        if selection_dir is not None:
            save_selection_shard(_make_identity_selection(shard), selection_dir / name)

    (packed_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))
    if selection_dir is not None:
        write_manifest(
            selection_dir,
            "test-selection",
            OverlayKind.PACKED_ROW,
            metadata={
                "packed_rows": {
                    "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
                },
            },
        )
    return packed_dir, selection_dir


def _drain(loader: Iterator[tuple], n_batches: int) -> list[tuple]:
    with contextlib.closing(loader):
        return [next(loader) for _ in range(n_batches)]


def _make_loader(
    packed_dir: Path,
    *,
    lookahead: int,
    batch_size: int = 3,
    selection_dir: Path | None = None,
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
        n_sae_features=N_SAE_FEATURES,
        selection_dir=selection_dir,
        label_dropout=label_dropout,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
        train_shard_lookahead=lookahead,
    )


def _assert_batches_equal(actual: tuple, expected: tuple) -> None:
    assert len(actual) == len(expected) == 10
    for field_idx, (actual_value, expected_value) in enumerate(
        zip(actual[:9], expected[:9], strict=True)
    ):
        if expected_value is None:
            assert actual_value is None, field_idx
        else:
            assert actual_value is not None, field_idx
            assert t.equal(actual_value, expected_value), field_idx

    actual_state, expected_state = actual[9], expected[9]
    assert isinstance(actual_state, dict)
    assert isinstance(expected_state, dict)
    json.dumps(actual_state, sort_keys=True)
    json.dumps(expected_state, sort_keys=True)
    assert actual_state == expected_state


def _load_fixture_batches(
    root: Path,
    *,
    lookahead: int,
    n_batches: int,
    n_shards: int = 3,
    n_rows: int = 5,
    batch_size: int = 3,
    with_selection: bool = False,
    label_dropout: float = 0.0,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    resume_state_dict: dict | None = None,
) -> list[tuple]:
    packed_dir, selection_dir = _write_shards(
        root,
        n_shards=n_shards,
        n_rows=n_rows,
        with_selection=with_selection,
    )
    t.manual_seed(0)
    loader = _make_loader(
        packed_dir,
        lookahead=lookahead,
        batch_size=batch_size,
        selection_dir=selection_dir,
        label_dropout=label_dropout,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
    )
    return _drain(loader, n_batches)


def _capture_prechange_goldens(root: Path) -> tuple[list[tuple], str]:
    packed_dir, _selection_dir = _write_shards(root, n_shards=3, n_rows=5)
    loader = precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=3,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        n_sae_features=N_SAE_FEATURES,
    )
    batches = _drain(loader, 14)
    decisions = [
        (
            state["batch_idx"],
            state["epoch"],
            state["shard"],
            state["shard_batch_offset"],
            tuple(tuple(row) for row in state["batch_rows"]),
        )
        for *_, state in batches
    ]
    digest = hashlib.sha256()
    for batch in batches[:8]:
        for tensor in batch[:9]:
            assert isinstance(tensor, t.Tensor)
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return decisions, digest.hexdigest()


# Captured from the unmodified loader at commit 3528ba362. Re-baseline with:
# uv run python -c 'import tempfile; from pathlib import Path; from tests.steering.\
# test_precomputed_shard_lookahead import _capture_prechange_goldens as c; \
# print(c(Path(tempfile.mkdtemp())))'
_PRECHANGE_TRAVERSAL_GOLDEN: list[tuple[Any, ...]] = [
    (
        0,
        1,
        "shard_00000.safetensors.zst",
        0,
        (
            (1, "shard_00000.safetensors.zst", 0),
            (1, "shard_00000.safetensors.zst", 1),
            (1, "shard_00000.safetensors.zst", 2),
        ),
    ),
    (
        1,
        1,
        "shard_00001.safetensors.zst",
        0,
        (
            (1, "shard_00000.safetensors.zst", 3),
            (1, "shard_00000.safetensors.zst", 4),
            (1, "shard_00001.safetensors.zst", 0),
        ),
    ),
    (
        2,
        1,
        "shard_00001.safetensors.zst",
        1,
        (
            (1, "shard_00001.safetensors.zst", 1),
            (1, "shard_00001.safetensors.zst", 2),
            (1, "shard_00001.safetensors.zst", 3),
        ),
    ),
    (
        3,
        1,
        "shard_00002.safetensors.zst",
        0,
        (
            (1, "shard_00001.safetensors.zst", 4),
            (1, "shard_00002.safetensors.zst", 0),
            (1, "shard_00002.safetensors.zst", 1),
        ),
    ),
    (
        4,
        1,
        "shard_00002.safetensors.zst",
        1,
        (
            (1, "shard_00002.safetensors.zst", 2),
            (1, "shard_00002.safetensors.zst", 3),
            (1, "shard_00002.safetensors.zst", 4),
        ),
    ),
    (
        5,
        2,
        "shard_00000.safetensors.zst",
        0,
        (
            (2, "shard_00000.safetensors.zst", 0),
            (2, "shard_00000.safetensors.zst", 1),
            (2, "shard_00000.safetensors.zst", 2),
        ),
    ),
    (
        6,
        2,
        "shard_00001.safetensors.zst",
        0,
        (
            (2, "shard_00000.safetensors.zst", 3),
            (2, "shard_00000.safetensors.zst", 4),
            (2, "shard_00001.safetensors.zst", 0),
        ),
    ),
    (
        7,
        2,
        "shard_00001.safetensors.zst",
        1,
        (
            (2, "shard_00001.safetensors.zst", 1),
            (2, "shard_00001.safetensors.zst", 2),
            (2, "shard_00001.safetensors.zst", 3),
        ),
    ),
    (
        8,
        2,
        "shard_00002.safetensors.zst",
        0,
        (
            (2, "shard_00001.safetensors.zst", 4),
            (2, "shard_00002.safetensors.zst", 0),
            (2, "shard_00002.safetensors.zst", 1),
        ),
    ),
    (
        9,
        2,
        "shard_00002.safetensors.zst",
        1,
        (
            (2, "shard_00002.safetensors.zst", 2),
            (2, "shard_00002.safetensors.zst", 3),
            (2, "shard_00002.safetensors.zst", 4),
        ),
    ),
    (
        10,
        3,
        "shard_00000.safetensors.zst",
        0,
        (
            (3, "shard_00000.safetensors.zst", 0),
            (3, "shard_00000.safetensors.zst", 1),
            (3, "shard_00000.safetensors.zst", 2),
        ),
    ),
    (
        11,
        3,
        "shard_00001.safetensors.zst",
        0,
        (
            (3, "shard_00000.safetensors.zst", 3),
            (3, "shard_00000.safetensors.zst", 4),
            (3, "shard_00001.safetensors.zst", 0),
        ),
    ),
    (
        12,
        3,
        "shard_00001.safetensors.zst",
        1,
        (
            (3, "shard_00001.safetensors.zst", 1),
            (3, "shard_00001.safetensors.zst", 2),
            (3, "shard_00001.safetensors.zst", 3),
        ),
    ),
    (
        13,
        3,
        "shard_00002.safetensors.zst",
        0,
        (
            (3, "shard_00001.safetensors.zst", 4),
            (3, "shard_00002.safetensors.zst", 0),
            (3, "shard_00002.safetensors.zst", 1),
        ),
    ),
]
_PRECHANGE_PAYLOAD_SHA256 = "942045d4f33fcb3112b03789a38f879e3e99e8ecb6563a1d51d1fa05de219d1f"


def test_prechange_traversal_decision_golden(tmp_path: Path) -> None:
    decisions, _digest = _capture_prechange_goldens(tmp_path)
    assert decisions == _PRECHANGE_TRAVERSAL_GOLDEN


def test_prechange_payload_digest_golden(tmp_path: Path) -> None:
    _decisions, digest = _capture_prechange_goldens(tmp_path)
    assert digest == _PRECHANGE_PAYLOAD_SHA256


def _plan(
    shard_paths: list[Path],
    **overrides: Any,
) -> Iterator[dataloader_module._ShardVisit]:
    kwargs: dict[str, Any] = {
        "start_epoch": 1,
        "resume_shard": None,
        "resume_shard_batch_offset": 0,
        "exact_continuation_shard": None,
        "exact_continuation_after_row": -1,
        "exact_continuation_batch_offset": 0,
        "log_epoch_starts": False,
    }
    kwargs.update(overrides)
    return dataloader_module._precomputed_shard_visits(shard_paths, **kwargs)


def test_fresh_traversal_cycles_shards_then_epochs() -> None:
    paths = [Path("shard_a"), Path("shard_b")]
    visits = list(itertools.islice(_plan(paths), 5))
    assert [(visit.shard_path, visit.epoch) for visit in visits] == [
        (paths[0], 1),
        (paths[1], 1),
        (paths[0], 2),
        (paths[1], 2),
        (paths[0], 3),
    ]
    assert all(
        (visit.first_row_idx, visit.shard_batch_idx, visit.skip_batches) == (0, 0, 0)
        for visit in visits
    )


def test_legacy_resume_offset_applies_to_the_matched_visit_only() -> None:
    paths = [Path("shard_a"), Path("shard_b"), Path("shard_c")]
    visits = list(
        itertools.islice(
            _plan(
                paths,
                start_epoch=4,
                resume_shard="shard_b",
                resume_shard_batch_offset=7,
            ),
            4,
        )
    )
    assert [(visit.shard_path.name, visit.epoch, visit.skip_batches) for visit in visits] == [
        ("shard_b", 4, 7),
        ("shard_c", 4, 0),
        ("shard_a", 5, 0),
        ("shard_b", 5, 0),
    ]


def test_unknown_legacy_resume_warns_and_starts_fresh_next_epoch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = [Path("shard_a"), Path("shard_b")]
    caplog.set_level(logging.WARNING)
    visit = next(
        _plan(
            paths,
            start_epoch=3,
            resume_shard="missing",
            resume_shard_batch_offset=9,
        )
    )
    assert visit == dataloader_module._ShardVisit(paths[0], 4, 0, 0, 0)
    assert [record.message for record in caplog.records] == [
        "PrecomputedBatchLoader: resume shard 'missing' not found in epoch 3. "
        "Starting fresh from epoch 4."
    ]


def test_exact_continuation_offsets_apply_to_the_matched_visit_only() -> None:
    paths = [Path("shard_a"), Path("shard_b"), Path("shard_c")]
    visits = list(
        itertools.islice(
            _plan(
                paths,
                exact_continuation_shard="shard_b",
                exact_continuation_after_row=4,
                exact_continuation_batch_offset=8,
            ),
            3,
        )
    )
    assert visits == [
        dataloader_module._ShardVisit(paths[1], 1, 5, 8, 0),
        dataloader_module._ShardVisit(paths[2], 1, 0, 0, 0),
        dataloader_module._ShardVisit(paths[0], 2, 0, 0, 0),
    ]


@pytest.mark.parametrize("log_epoch_starts", [False, True])
def test_epoch_start_logging_is_optional(
    caplog: pytest.LogCaptureFixture,
    log_epoch_starts: bool,
) -> None:
    caplog.set_level(logging.INFO)
    visits = _plan(
        [Path("shard_a"), Path("shard_b")],
        log_epoch_starts=log_epoch_starts,
    )
    list(itertools.islice(visits, 3))
    records = [record.message for record in caplog.records if "starting epoch" in record.message]
    assert records == (["PrecomputedBatchLoader: starting epoch 2"] if log_epoch_starts else [])


def test_traversal_honours_start_epoch() -> None:
    visit = next(_plan([Path("shard_a")], start_epoch=11))
    assert visit.epoch == 11


def test_background_lookahead_preserves_order_and_finite_termination() -> None:
    stream = dataloader_module._background_lookahead(iter(range(7)), 3)
    with contextlib.closing(stream):
        assert list(stream) == list(range(7))


def test_background_lookahead_reraises_producer_exception_in_order() -> None:
    error = RuntimeError("producer failed")

    def inner() -> Iterator[int]:
        yield 1
        yield 2
        raise error

    stream = dataloader_module._background_lookahead(inner(), 2)
    with contextlib.closing(stream):
        assert next(stream) == 1
        assert next(stream) == 2
        with pytest.raises(RuntimeError, match="producer failed") as caught:
            next(stream)
    assert caught.value is error


def test_background_lookahead_runs_ahead_but_remains_bounded() -> None:
    depth = 2
    started = [threading.Event() for _ in range(8)]

    def inner() -> Iterator[int]:
        for index, event in enumerate(started):
            event.set()
            yield index

    stream = dataloader_module._background_lookahead(inner(), depth)
    with contextlib.closing(stream):
        assert next(stream) == 0
        # After one item is consumed the producer holds `depth` queued plus one
        # blocked in put(), so exactly indices 0..depth+1 start. Wait for the
        # LAST index that should start before asserting the next one has not:
        # checking the negative straight after started[depth] passes spuriously
        # because the producer simply has not reached it yet, which let a
        # maxsize=depth+1 regression (33% more resident shards) through.
        assert started[depth + 1].wait(timeout=5.0)
        time.sleep(0.1)
        assert not started[depth + 2].is_set(), (
            "producer ran further ahead than the queue bound permits"
        )


def test_background_lookahead_close_stops_the_thread() -> None:
    def inner() -> Iterator[int]:
        index = 0
        while True:
            yield index
            index += 1

    stream = dataloader_module._background_lookahead(inner(), 3)
    assert next(stream) == 0
    assert len(_lookahead_threads()) == 1
    assert _lookahead_threads()[0].daemon is True
    stream.close()
    _wait_for_no_lookahead_thread()


def test_background_lookahead_close_warns_for_a_blocked_inner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def inner() -> Iterator[int]:
        yield 0
        entered.set()
        release.wait()
        yield 1

    caplog.set_level(logging.WARNING)
    stream = dataloader_module._background_lookahead(inner(), 1)
    assert next(stream) == 0
    assert entered.wait(timeout=1.0)
    started = time.monotonic()
    try:
        stream.close()
    finally:
        release.set()
    elapsed = time.monotonic() - started
    assert elapsed <= dataloader_module._SHARD_LOOKAHEAD_JOIN_SECONDS + 1.0
    assert any("did not stop within" in record.message for record in caplog.records)


def test_background_lookahead_is_lazy() -> None:
    stream = dataloader_module._background_lookahead(iter([1]), 1)
    assert not _lookahead_threads()
    assert next(stream) == 1
    stream.close()


@pytest.mark.parametrize("depth", [0, -1])
def test_background_lookahead_rejects_nonpositive_depth(depth: int) -> None:
    stream = dataloader_module._background_lookahead(iter(()), depth)
    with pytest.raises(ValueError, match="must be positive"):
        next(stream)


def _assert_streams_equal(actual: list[tuple], expected: list[tuple]) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        _assert_batches_equal(actual_batch, expected_batch)


def test_fused_path_is_bitwise_equal_at_every_lookahead_depth(tmp_path: Path) -> None:
    baseline = _load_fixture_batches(tmp_path / "fused-0", lookahead=0, n_batches=12)
    for depth in (1, 2, 5):
        actual = _load_fixture_batches(
            tmp_path / f"fused-{depth}",
            lookahead=depth,
            n_batches=12,
        )
        _assert_streams_equal(actual, baseline)


def test_cross_epoch_batches_are_bitwise_equal_at_every_lookahead_depth(
    tmp_path: Path,
) -> None:
    baseline = _load_fixture_batches(
        tmp_path / "cross-epoch-0",
        lookahead=0,
        n_batches=6,
        n_shards=1,
        n_rows=3,
        batch_size=4,
    )
    assert all(len({row[0] for row in batch[9]["batch_rows"]}) > 1 for batch in baseline)
    for depth in (1, 3, 5):
        actual = _load_fixture_batches(
            tmp_path / f"cross-epoch-{depth}",
            lookahead=depth,
            n_batches=6,
            n_shards=1,
            n_rows=3,
            batch_size=4,
        )
        _assert_streams_equal(actual, baseline)


def test_selection_sidecar_path_is_bitwise_equal_at_every_lookahead_depth(
    tmp_path: Path,
) -> None:
    baseline = _load_fixture_batches(
        tmp_path / "selection-0",
        lookahead=0,
        n_batches=12,
        with_selection=True,
    )
    for depth in (1, 2, 5):
        actual = _load_fixture_batches(
            tmp_path / f"selection-{depth}",
            lookahead=depth,
            n_batches=12,
            with_selection=True,
        )
        _assert_streams_equal(actual, baseline)


def test_label_dropout_is_bitwise_equal_at_every_lookahead_depth(tmp_path: Path) -> None:
    baseline = _load_fixture_batches(
        tmp_path / "dropout-0",
        lookahead=0,
        n_batches=12,
        label_dropout=0.3,
    )
    for depth in (1, 2, 5):
        actual = _load_fixture_batches(
            tmp_path / f"dropout-{depth}",
            lookahead=depth,
            n_batches=12,
            label_dropout=0.3,
        )
        _assert_streams_equal(actual, baseline)


def test_ddp_stripes_are_bitwise_equal_at_every_lookahead_depth(tmp_path: Path) -> None:
    for rank in (0, 1):
        baseline = _load_fixture_batches(
            tmp_path / f"ddp-rank-{rank}-0",
            lookahead=0,
            n_batches=12,
            ddp_rank=rank,
            ddp_world_size=2,
        )
        for depth in (1, 2, 5):
            actual = _load_fixture_batches(
                tmp_path / f"ddp-rank-{rank}-{depth}",
                lookahead=depth,
                n_batches=12,
                ddp_rank=rank,
                ddp_world_size=2,
            )
            _assert_streams_equal(actual, baseline)


@pytest.mark.parametrize(
    ("save_index", "save_point"),
    [
        (0, "mid-shard"),
        (4, "shard-boundary"),
        (1, "cross-shard-pending"),
        (3, "second-continuation-visit-batch"),
    ],
    ids=lambda value: str(value),
)
def test_exact_resume_is_bitwise_equal_with_and_without_lookahead(
    tmp_path: Path,
    save_index: int,
    save_point: str,
) -> None:
    del save_point
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=5)
    baseline = _drain(_make_loader(packed_dir, lookahead=0), save_index + 5)
    state = json.loads(json.dumps(baseline[save_index][9]))

    resumed_by_depth: dict[int, list[tuple]] = {}
    for depth in (0, 3):
        t.manual_seed(0)
        resumed = _drain(
            _make_loader(
                packed_dir,
                lookahead=depth,
                resume_state_dict=json.loads(json.dumps(state)),
            ),
            5,
        )
        _assert_streams_equal(resumed, baseline[save_index : save_index + 5])
        resumed_by_depth[depth] = resumed
    _assert_streams_equal(resumed_by_depth[3], resumed_by_depth[0])


def test_exact_resume_replay_runs_inline(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=5)
    baseline = _drain(_make_loader(packed_dir, lookahead=0), 2)
    state = json.loads(json.dumps(baseline[0][9]))
    resumed = _make_loader(packed_dir, lookahead=3, resume_state_dict=state)
    try:
        _assert_batches_equal(next(resumed), baseline[0])
        assert not _lookahead_threads()
        _assert_batches_equal(next(resumed), baseline[1])
        assert _lookahead_threads()
    finally:
        resumed.close()


def test_legacy_resume_stream_is_equal_with_and_without_lookahead(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=5)
    baseline = _drain(_make_loader(packed_dir, lookahead=0), 3)
    state = json.loads(json.dumps(baseline[1][9]))
    state.pop("batch_rows")

    streams = []
    for depth in (0, 3):
        t.manual_seed(0)
        streams.append(
            _drain(
                _make_loader(
                    packed_dir,
                    lookahead=depth,
                    resume_state_dict=json.loads(json.dumps(state)),
                ),
                6,
            )
        )
    _assert_streams_equal(streams[1], streams[0])


def test_legacy_offset_without_shard_starts_every_visit_at_zero(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=8)
    # The old loop dropped the first two batches of every shard forever when
    # ``shard`` was absent. Repository checkpoints cannot produce that state:
    # materialization and migration always write both cursor fields. Keep the
    # refactor's non-dropping behavior pinned for malformed external states.
    resume_state = {
        "batch_idx": 0,
        "epoch": 1,
        "shard_batch_offset": 2,
        "max_seg_size": None,
    }
    expected = [
        (1, "shard_00000.safetensors.zst", 0),
        (1, "shard_00000.safetensors.zst", 1),
        (1, "shard_00000.safetensors.zst", 2),
        (1, "shard_00000.safetensors.zst", 3),
        (1, "shard_00001.safetensors.zst", 0),
        (1, "shard_00001.safetensors.zst", 1),
        (1, "shard_00001.safetensors.zst", 2),
        (1, "shard_00001.safetensors.zst", 3),
    ]
    for depth in (0, 3):
        batches = _drain(
            _make_loader(
                packed_dir,
                lookahead=depth,
                batch_size=2,
                resume_state_dict=json.loads(json.dumps(resume_state)),
            ),
            8,
        )
        assert [
            (state["epoch"], state["shard"], state["shard_batch_offset"]) for *_, state in batches
        ] == expected


_SLOW_SECONDS = 1.0


def test_lookahead_absorbs_a_slow_shard_load_and_depth_zero_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = dataloader_module.load_packed_row_shard

    def measure(depth: int) -> list[float]:
        packed_dir, _selection_dir = _write_shards(
            tmp_path / f"slow-{depth}",
            n_shards=3,
            n_rows=8,
        )

        def slow_second_shard(path: Path):
            if Path(path).name.startswith("shard_00001"):
                time.sleep(_SLOW_SECONDS)
            return real_load(path)

        gaps = []
        with monkeypatch.context() as patch:
            patch.setattr(dataloader_module, "load_packed_row_shard", slow_second_shard)
            loader = _make_loader(packed_dir, lookahead=depth, batch_size=2)
            with contextlib.closing(loader):
                for _ in range(8):
                    started = time.monotonic()
                    next(loader)
                    gaps.append(time.monotonic() - started)
                    time.sleep(0.3)
        return gaps

    lookahead_gaps = measure(2)
    inline_gaps = measure(0)
    assert max(lookahead_gaps[1:]) < _SLOW_SECONDS / 3, lookahead_gaps
    assert max(inline_gaps[1:]) > _SLOW_SECONDS * 0.8, inline_gaps


def test_lookahead_starts_loads_before_the_consumer_needs_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = dataloader_module.load_packed_row_shard

    def run(depth: int) -> list[threading.Event]:
        packed_dir, _selection_dir = _write_shards(
            tmp_path / f"events-{depth}",
            n_shards=4,
            n_rows=8,
        )
        events = [threading.Event() for _ in range(4)]

        def record_load(path: Path):
            index = int(Path(path).name.split("_")[1].split(".")[0])
            events[index].set()
            return real_load(path)

        with monkeypatch.context() as patch:
            patch.setattr(dataloader_module, "load_packed_row_shard", record_load)
            loader = _make_loader(packed_dir, lookahead=depth, batch_size=2)
            with contextlib.closing(loader):
                next(loader)
                if depth > 0:
                    assert events[3].wait(timeout=1.0)
                else:
                    assert not events[1].is_set()
        return events

    run(2)
    run(0)


def test_selection_sidecar_failure_arrives_after_prior_shard_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_dir, selection_dir = _write_shards(
        tmp_path,
        n_shards=3,
        n_rows=8,
        with_selection=True,
    )
    assert selection_dir is not None
    real_load = dataloader_module.load_selection_shard
    second_load_started = threading.Event()
    release_second_load = threading.Event()

    def controlled_load(path: Path):
        if (
            Path(path).name.startswith("shard_00001")
            and threading.current_thread().name == "packed-shard-lookahead"
        ):
            second_load_started.set()
            release_second_load.wait()
        return real_load(path)

    monkeypatch.setattr(dataloader_module, "load_selection_shard", controlled_load)
    loader = _make_loader(
        packed_dir,
        lookahead=2,
        batch_size=2,
        selection_dir=selection_dir,
    )
    try:
        first = next(loader)
        assert first[9]["shard"] == "shard_00000.safetensors.zst"
        assert second_load_started.wait(timeout=1.0)
        second_path = next(selection_dir.glob("shard_00001*"))
        second_path.unlink()
        release_second_load.set()
        prior = [first, next(loader), next(loader), next(loader)]
        assert all(batch[9]["shard"] == "shard_00000.safetensors.zst" for batch in prior)
        with pytest.raises(FileNotFoundError, match="shard_00001"):
            next(loader)
    finally:
        release_second_load.set()
        loader.close()


def test_base_shard_failure_arrives_after_prior_shard_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=8)
    real_load = dataloader_module.load_packed_row_shard
    error = OSError("synthetic shard failure")

    def fail_second(path: Path):
        if Path(path).name.startswith("shard_00001"):
            raise error
        return real_load(path)

    monkeypatch.setattr(dataloader_module, "load_packed_row_shard", fail_second)
    loader = _make_loader(packed_dir, lookahead=2, batch_size=2)
    try:
        prior = [next(loader) for _ in range(4)]
        assert all(batch[9]["shard"] == "shard_00000.safetensors.zst" for batch in prior)
        with pytest.raises(OSError, match="synthetic shard failure") as caught:
            next(loader)
        assert caught.value is error
    finally:
        loader.close()


def test_real_loader_close_joins_the_lookahead_thread(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=8)
    loader = _make_loader(packed_dir, lookahead=3, batch_size=2)
    next(loader)
    next(loader)
    assert _lookahead_threads()
    started = time.monotonic()
    loader.close()
    assert time.monotonic() - started < 1.0
    _wait_for_no_lookahead_thread()


def test_abandoning_real_loader_stops_the_lookahead_thread(tmp_path: Path) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=3, n_rows=8)
    loader = _make_loader(packed_dir, lookahead=3, batch_size=2)
    next(loader)
    assert _lookahead_threads()
    del loader
    gc.collect()
    _wait_for_no_lookahead_thread(timeout=5.0)


@pytest.mark.parametrize(
    ("depth", "packed", "remote", "raises"),
    [
        (1, object(), None, True),
        (1, None, object(), True),
        (0, object(), None, False),
        (0, None, object(), False),
        (1, None, None, False),
    ],
)
def test_shard_lookahead_support_boundary(
    depth: int,
    packed: object | None,
    remote: object | None,
    raises: bool,
) -> None:
    if raises:
        with pytest.raises(ValueError, match="train_shard_lookahead"):
            dataloader_module._assert_shard_lookahead_supported(depth, packed, remote)
    else:
        dataloader_module._assert_shard_lookahead_supported(depth, packed, remote)


def test_loader_rejects_remote_materializer_with_lookahead(tmp_path: Path) -> None:
    loader = precomputed_data_loader(
        packed_row_dir=tmp_path / "not-read",
        B=1,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        remote_materializer=object(),
        train_shard_lookahead=1,
    )
    with pytest.raises(ValueError, match="train_shard_lookahead"):
        next(loader)


def test_loader_rejects_packed_materializer_with_lookahead(tmp_path: Path) -> None:
    loader = precomputed_data_loader(
        packed_row_dir=tmp_path / "not-read",
        B=1,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        packed_materializer=object(),
        train_shard_lookahead=1,
    )
    with pytest.raises(ValueError, match="train_shard_lookahead"):
        next(loader)


@pytest.mark.parametrize("invalid_depth", [-1, 1.5, True])
def test_loader_rejects_invalid_shard_lookahead_depth(
    tmp_path: Path,
    invalid_depth: object,
) -> None:
    loader = precomputed_data_loader(
        packed_row_dir=tmp_path / "not-read",
        B=1,
        device="cpu",
        feature_pipeline=_build_pipeline(),
        train_shard_lookahead=invalid_depth,
    )
    with pytest.raises(TypeError, match="Could not parse.*NonNegativeInt"):
        next(loader)


@pytest.mark.parametrize("depth, expected", [(0, 4), (1, 6), (2, 8), (4, 12), (8, 20)])
def test_shared_decode_retain_scales_with_lookahead(depth: int, expected: int) -> None:
    assert dataloader_module._shared_decode_retain_for_lookahead(depth) == expected


@pytest.mark.parametrize("depth, expected_retain", [(0, 4), (4, 12)])
def test_loader_passes_the_scaled_retain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    expected_retain: int,
) -> None:
    packed_dir, _selection_dir = _write_shards(
        tmp_path / f"retain-{depth}",
        n_shards=3,
        n_rows=8,
    )
    real_load = dataloader_module.load_packed_row_shard
    recorded: list[int] = []

    def spy(
        shard_path: Path,
        logical_epoch: int,
        local_rank: int,
        local_world: int,
        *,
        retain: int = dataloader_module._SHARED_DECODE_RETAIN,
        node_cache=None,
    ):
        del logical_epoch, local_rank, local_world, node_cache
        recorded.append(retain)
        return real_load(shard_path)

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(dataloader_module, "_shared_decode_load_base_shard", spy)
    loader = _make_loader(packed_dir, lookahead=depth, batch_size=2)
    with contextlib.closing(loader):
        next(loader)
    assert recorded
    assert set(recorded) == {expected_retain}


@pytest.mark.parametrize(
    ("depth", "level"),
    [
        (1, logging.INFO),
        (15, logging.INFO),
        (16, logging.WARNING),
        (31, logging.WARNING),
        (32, logging.ERROR),
        (63, logging.ERROR),
        (64, logging.ERROR),
    ],
)
def test_shard_lookahead_budget_log_levels(
    caplog: pytest.LogCaptureFixture,
    depth: int,
    level: int,
) -> None:
    retain = dataloader_module._shared_decode_retain_for_lookahead(depth)
    message = f"depth={depth} retain={retain}"
    caplog.set_level(logging.INFO)
    dataloader_module._log_shard_lookahead_budget(depth, message)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == level
    assert message in caplog.records[0].message
    if depth >= 64:
        assert caplog.records[0].exc_info is not None
        assert "NoneType: None" in caplog.records[0].message
    else:
        assert caplog.records[0].message == message


def test_loader_emits_exact_shard_lookahead_budget(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=2, n_rows=4)
    caplog.set_level(logging.INFO)
    loader = _make_loader(packed_dir, lookahead=4, batch_size=2)
    with contextlib.closing(loader):
        next(loader)

    records = [
        record.message for record in caplog.records if "train_shard_lookahead=4" in record.message
    ]
    assert records == [
        "train_shard_lookahead=4: banked shard pairs=4; live decoded shards=7 "
        "(process-private); shared-decode retain window=12; see docs/pipelines/"
        "encoder-decoder-training.md#shard-level-lookahead-train_shard_lookahead"
    ]


def test_depth_zero_emits_no_budget_record_and_starts_no_thread(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=2, n_rows=4)
    caplog.set_level(logging.INFO)
    loader = _make_loader(packed_dir, lookahead=0, batch_size=2)
    with contextlib.closing(loader):
        next(loader)
        assert not _lookahead_threads()
    assert not any("train_shard_lookahead=" in record.message for record in caplog.records)


def test_nonzero_rank_loader_does_not_log_epoch_starts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=1, n_rows=4)
    caplog.set_level(logging.INFO)
    _drain(
        _make_loader(
            packed_dir,
            lookahead=1,
            batch_size=2,
            ddp_rank=1,
            ddp_world_size=2,
        ),
        2,
    )
    assert not any("starting epoch" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Two-process shared decode composed with lookahead (the production topology)
# ---------------------------------------------------------------------------

# Runs a REAL precomputed_data_loader with a positive lookahead depth in each of
# two local ranks, so one process owns each shard's shared-memory segment and the
# other attaches to it -- while both run their lookahead producers ahead of
# consumption. Nothing else in the suite composes those two mechanisms: the
# shared-decode tests use depth 0 and the lookahead tests monkeypatch the shared
# store out. A direct interpreter (not multiprocessing.spawn) keeps it seconds
# rather than minutes. Prints the per-batch digests on the last lines.
_TWO_RANK_LOOKAHEAD_SNIPPET = """
import hashlib, json, sys
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from tests.encdec_test_helpers import build_threshold_pipeline

import time

packed_dir, rank, depth = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ack_dir = Path(sys.argv[4])
NUM_BUCKETS, N_SAE = 10, 512
boundaries = (
    t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
    .unsqueeze(0).expand(N_SAE, -1).contiguous()
)
loader = precomputed_data_loader(
    packed_row_dir=packed_dir,
    B=2,
    device="cpu",
    feature_pipeline=build_threshold_pipeline(boundaries),
    n_sae_features=N_SAE,
    ddp_rank=rank,
    ddp_world_size=2,
    train_shard_lookahead=depth,
)
digests = []
try:
    for _ in range(10):
        batch = next(loader)
        h = hashlib.sha256()
        for field in batch[:9]:
            h.update(b"<None>" if field is None else field.contiguous().numpy().tobytes())
        digests.append(h.hexdigest()[:16])
finally:
    loader.close()

# Each rank OWNS some shards, and exiting runs the atexit hook that unlinks the
# names it owns. If either rank exits while the other is still attaching, the
# attacher waits out the 120s timeout and silently direct-decodes instead --
# byte-identical, so the test would pass while exercising nothing shared. Hold
# both processes until both are done. No production rank exits mid-run.
(ack_dir / f"done_{rank}").touch()
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    if all((ack_dir / f"done_{other}").exists() for other in (0, 1)):
        break
    time.sleep(0.05)
print("DIGESTS " + json.dumps(digests))
"""


def _run_two_rank_lookahead(packed_dir: Path, depth: int, tmp_path: Path) -> dict[int, list[str]]:
    ack_dir = tmp_path / f"ack_depth{depth}"
    ack_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = f"lookahead2p{os.getpid()}{depth}"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["LOCAL_RANK"] = "0"  # overridden per process below
    env["LOCAL_WORLD_SIZE"] = "2"
    procs = {}
    for rank in (0, 1):
        rank_env = dict(env)
        rank_env["LOCAL_RANK"] = str(rank)
        procs[rank] = subprocess.Popen(
            [sys.executable, "-c", _TWO_RANK_LOOKAHEAD_SNIPPET, os.fspath(packed_dir),
             str(rank), str(depth), os.fspath(ack_dir)],
            env=rank_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    out: dict[int, list[str]] = {}
    for rank, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"rank {rank} failed: {stderr[-2000:]}"
        # Non-vacuity guard: the attacher must genuinely attach. Without this a
        # timed-out attach silently falls back to a byte-identical direct decode
        # and the test would still pass while exercising nothing shared.
        assert "falling back to direct decode" not in stderr, (
            f"rank {rank} fell back instead of attaching: {stderr[-2000:]}"
        )
        line = next(row for row in stdout.splitlines() if row.startswith("DIGESTS "))
        out[rank] = json.loads(line.removeprefix("DIGESTS "))
    return out


def test_two_process_shared_decode_with_lookahead_matches_depth_zero(tmp_path: Path) -> None:
    """Shared decode and lookahead compose: one owner, one attacher, both banking.

    Under torchrun with two local ranks, exactly one process decodes each shard
    and publishes it to /dev/shm while the other attaches zero-copy. This runs
    that for real with depth 3 on both ranks and requires each rank's batch
    stream to equal the same rank's depth-0 stream, so neither the attach path
    nor the scaled retain window perturbs the data.
    """
    packed_dir, _selection_dir = _write_shards(tmp_path, n_shards=4, n_rows=6)
    banked = _run_two_rank_lookahead(packed_dir, 3, tmp_path)
    inline = _run_two_rank_lookahead(packed_dir, 0, tmp_path)
    assert banked[0] == inline[0], "rank 0 diverged between depth 3 and depth 0"
    assert banked[1] == inline[1], "rank 1 diverged between depth 3 and depth 0"
    assert banked[0] != banked[1], "ranks must consume disjoint row stripes"
    assert len(banked[0]) == 10
