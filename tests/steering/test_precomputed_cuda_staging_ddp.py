"""Opt-in multi-rank CUDA regressions for issues #842 and #868.

Run the synthetic two-rank compile gate with:

    RUN_ISSUE842_DDP=1 uv run pytest \
      tests/steering/test_precomputed_cuda_staging_ddp.py -m slow

Run the production-artifact eight-H100 gate under ``torchrun`` with:

    RUN_ISSUE868_H100_DDP=1 \
    ISSUE868_PACKED_ROW_DIR=/path/to/packed/rows \
    ISSUE868_SELECTION_DIR=/path/to/selection \
      uv run torchrun --standalone --nproc_per_node=8 -m pytest -q \
      tests/steering/test_precomputed_cuda_staging_ddp.py \
      -m very_slow -k issue868
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import traceback
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.precomputed_cuda_staging import (
    PrecomputedBatch,
    PrecomputedCudaStagingLoader,
    retained_tensor_bytes,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import save_packed_row_shard, save_selection_shard
from tests.encdec_test_helpers import NUM_BUCKETS
from tests.steering.test_precomputed_dataloader_ddp import (
    _make_pipeline,
    _make_selection_shard_for,
    _make_shard,
)

pytestmark = pytest.mark.slow

_WORLD_SIZE = 2
_BATCH_SIZE = 16
_CONTEXT_LEN = 1_024
_N_SEG = 13
_PAD_F = 96
_FEATURE_VOCAB = 2_074
_TOKEN_VOCAB = 32_768
_MAGNITUDE_VOCAB = NUM_BUCKETS + 1
_ROWS_PER_SHARD = 17
_SHARDS = 3
_BATCHES = 72
_HOST_QUEUE_BATCHES = 64
_DEEP_QUEUE_BATCHES = 32
_ISSUE868_WORLD_SIZE = 8
_ISSUE868_HOST_QUEUE_BYTES = 512 * 1024 * 1024
_ISSUE868_PINNED_SLOTS = 3
_ISSUE868_BOUNDARIES = 3
_ISSUE868_MAX_BATCHES = 520


class _StaticConsumer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 32
        self.token_embed = nn.Embedding(_TOKEN_VOCAB, width)
        self.feature_embed = nn.Embedding(_FEATURE_VOCAB, width)
        self.magnitude_embed = nn.Embedding(_MAGNITUDE_VOCAB, width)
        self.output = nn.Linear(width, 1)

    def forward(
        self,
        inputs: t.Tensor,
        targets: t.Tensor,
        enc_fids: t.Tensor,
        enc_mags: t.Tensor,
        enc_mask: t.Tensor,
        enc_pcts: t.Tensor,
        seg_map: t.Tensor,
        _seg_doc_id: t.Tensor,
        fp_targets: t.Tensor,
    ) -> t.Tensor:
        _ = (
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            enc_pcts,
            seg_map,
            fp_targets,
        )
        token_hidden = self.token_embed(inputs)
        feature_hidden = self.feature_embed(enc_fids)
        feature_hidden = feature_hidden + self.magnitude_embed(enc_mags)
        weights = enc_mask.unsqueeze(-1).to(feature_hidden.dtype)
        weights = weights * (enc_pcts.unsqueeze(-1).to(feature_hidden.dtype) + 1)
        segment_hidden = (feature_hidden * weights).sum(dim=2)
        segment_hidden = segment_hidden / weights.sum(dim=2).clamp_min(1)
        token_conditioning = segment_hidden.gather(
            1,
            seg_map.unsqueeze(-1).expand(-1, -1, segment_hidden.shape[-1]),
        )
        prediction = self.output(token_hidden + token_conditioning).squeeze(-1)
        loss = (prediction.float() - targets.remainder(97).to(t.float32).div(97)).square().mean()
        return loss + fp_targets.mean() * 1e-7


def _write_production_shape_artifacts(root: Path) -> tuple[Path, Path]:
    packed_dir = root / "packed"
    selection_dir = root / "selection"
    packed_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)
    packed_paths: list[Path] = []
    for shard_index in range(_SHARDS):
        base = _make_shard(
            n_rows=_ROWS_PER_SHARD,
            context_len=_CONTEXT_LEN,
            segs_per_row=_N_SEG,
            feats_per_seg=_PAD_F,
            seed=842 + shard_index,
        )
        base.feature_ids = base.feature_ids.to(t.int64).remainder(_FEATURE_VOCAB).to(t.uint16)
        packed_paths.append(
            save_packed_row_shard(
                base,
                packed_dir / f"shard_{shard_index:05d}",
            )
        )
        selection = _make_selection_shard_for(
            base,
            sel_per_seg=64,
            neg_per_seg=32,
            seed=1_842 + shard_index,
        )
        selection.sel_fids = selection.sel_fids.to(t.int64).remainder(_FEATURE_VOCAB).to(t.uint16)
        selection.neg_fids = selection.neg_fids.to(t.int64).remainder(_FEATURE_VOCAB).to(t.uint16)
        selection.sel_buckets = selection.sel_buckets.remainder(2)
        save_selection_shard(
            selection,
            selection_dir / f"shard_{shard_index:05d}",
        )
    max_seg_size = _CONTEXT_LEN - (_N_SEG - 1) * (_CONTEXT_LEN // _N_SEG)
    write_manifest(
        packed_dir,
        "issue-842-packed",
        OverlayKind.PACKED_ROW,
        metadata={
            "max_n_seg": _N_SEG,
            "max_seg_size": max_seg_size,
            "context_len": _CONTEXT_LEN,
        },
        base_shard_paths=packed_paths,
    )
    write_manifest(
        selection_dir,
        "issue-842-selection",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
            },
        },
        base_shard_paths=packed_paths,
    )
    return packed_dir, selection_dir


def _loader(
    packed_dir: str,
    selection_dir: str,
    *,
    rank: int,
    device: str,
    world_size: int = _WORLD_SIZE,
    resume_state_dict: dict[str, Any] | None = None,
) -> Iterator[PrecomputedBatch]:
    feature_pipeline = _make_pipeline(_FEATURE_VOCAB)
    return cast(
        Iterator[PrecomputedBatch],
        precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=_BATCH_SIZE,
            feature_pipeline=feature_pipeline,
            device=device,
            selection_dir=selection_dir,
            effective_buckets=feature_pipeline.effective_buckets,
            n_sae_features=_FEATURE_VOCAB,
            ddp_rank=rank,
            ddp_world_size=world_size,
            resume_state_dict=resume_state_dict,
        ),
    )


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, t.Tensor):
        assert isinstance(actual, t.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert t.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        actual_dict = cast(dict[Any, Any], actual)
        expected_dict = cast(dict[Any, Any], expected)
        assert actual_dict.keys() == expected_dict.keys()
        for key in expected_dict:
            _assert_nested_equal(actual_dict[key], expected_dict[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_batch_equal(actual: PrecomputedBatch, expected: PrecomputedBatch) -> None:
    for actual_field, expected_field in zip(actual, expected, strict=True):
        _assert_nested_equal(actual_field, expected_field)


def _close_loader(loader: object) -> None:
    close = getattr(loader, "close", None)
    assert callable(close)
    close()


def _batch_crosses_packed_shard(state: dict[str, Any]) -> bool:
    rows = state["batch_rows"]
    assert isinstance(rows, list)
    return len({row[1] for row in rows}) > 1


def _worker(
    rank: int,
    port: int,
    packed_dir: str,
    selection_dir: str,
    result_queue: Any,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    loader: PrecomputedCudaStagingLoader | None = None
    legacy: Iterator[PrecomputedBatch] | None = None
    try:
        device = t.device("cuda", rank)
        t.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(minutes=5),
        )
        t.manual_seed(842)
        model = _StaticConsumer().to(device=device, dtype=t.bfloat16)
        model = DistributedDataParallel(model, device_ids=[rank])
        compiled = cast(nn.Module, t.compile(model, dynamic=False))
        optimizer = t.optim.SGD(compiled.parameters(), lr=1e-3)
        probe = _loader(
            packed_dir,
            selection_dir,
            rank=rank,
            device="cpu",
        )
        prototype = next(probe)
        _close_loader(probe)
        batch_bytes = retained_tensor_bytes(prototype)
        legacy = _loader(
            packed_dir,
            selection_dir,
            rank=rank,
            device=str(device),
        )
        loader = PrecomputedCudaStagingLoader(
            _loader(
                packed_dir,
                selection_dir,
                rank=rank,
                device="cpu",
            ),
            device=device,
            host_queue_bytes=batch_bytes * _HOST_QUEUE_BATCHES,
            pinned_slots=3,
            expected_batch_size=_BATCH_SIZE,
            expected_context_len=_CONTEXT_LEN,
            expected_pad_f=_PAD_F,
            expected_n_seg=_N_SEG,
        )
        deep_queue_target = batch_bytes * _DEEP_QUEUE_BATCHES
        deep_queue_deadline = time.monotonic() + 60
        while (
            loader.stats().host_queue_peak_bytes < deep_queue_target
            and time.monotonic() < deep_queue_deadline
        ):
            time.sleep(0.01)
        assert loader.stats().host_queue_peak_bytes >= deep_queue_target

        losses: list[float] = []
        crossed_boundary = False
        for _ in range(_BATCHES):
            batch = next(loader)
            expected = next(legacy)
            _assert_batch_equal(batch, expected)
            assert batch[0].device == device
            assert int(batch[0].max().item()) < _TOKEN_VOCAB
            assert int(batch[2].max().item()) < _FEATURE_VOCAB
            assert int(batch[3].max().item()) < _MAGNITUDE_VOCAB
            assert batch[8] is not None
            assert batch[8].shape == (
                _BATCH_SIZE,
                _N_SEG,
                _FEATURE_VOCAB,
            )
            rows = batch[9]["batch_rows"]
            assert all(row[2] % _WORLD_SIZE == rank for row in rows)
            crossed_boundary |= _batch_crosses_packed_shard(batch[9])
            optimizer.zero_grad(set_to_none=True)
            loss = compiled(*batch[:9])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        assert crossed_boundary
        checksum = t.stack(
            [parameter.detach().float().sum() for parameter in compiled.parameters()]
        ).sum()
        gathered = [t.zeros_like(checksum) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered, checksum)
        assert t.equal(gathered[0], gathered[1])
        stats = loader.stats()
        result_queue.put(
            (
                rank,
                {
                    "status": "ok",
                    "losses": losses,
                    "consumed": stats.consumed_batches,
                    "crossed_boundary": crossed_boundary,
                    "host_peak": stats.host_queue_peak_bytes,
                    "host_cap": batch_bytes * _HOST_QUEUE_BATCHES,
                    "deep_queue_target": deep_queue_target,
                },
            )
        )
    except BaseException:
        result_queue.put(
            (
                rank,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        if loader is not None:
            loader.close()
        if legacy is not None:
            _close_loader(legacy)
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_two_ranks(
    packed_dir: Path,
    selection_dir: Path,
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                port,
                str(packed_dir),
                str(selection_dir),
                result_queue,
            ),
        )
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    results: dict[int, dict[str, Any]] = {}
    try:
        for _ in range(_WORLD_SIZE):
            rank, result = result_queue.get(timeout=600)
            results[rank] = result
    except queue.Empty as error:
        raise AssertionError("issue-#842 two-rank workers timed out") from error
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    return [results[rank] for rank in range(_WORLD_SIZE)]


@pytest.mark.skipif(
    os.environ.get("RUN_ISSUE842_DDP") != "1" or t.cuda.device_count() < 2,
    reason="requires RUN_ISSUE842_DDP=1 and two CUDA devices",
)
def test_two_rank_static_compile_production_shapes_are_stream_safe(tmp_path: Path) -> None:
    packed_dir, selection_dir = _write_production_shape_artifacts(tmp_path)
    results = _spawn_two_ranks(packed_dir, selection_dir)

    for rank, result in enumerate(results):
        assert result["status"] == "ok", f"rank {rank} failed:\n{result.get('traceback', '')}"
        assert result["consumed"] == _BATCHES
        assert len(result["losses"]) == _BATCHES
        assert all(math.isfinite(value) for value in result["losses"])
        assert result["crossed_boundary"] is True
        assert result["host_peak"] >= result["deep_queue_target"]
        assert result["host_peak"] <= result["host_cap"]


def _issue868_artifact_dir(environment_key: str) -> Path:
    raw = os.environ.get(environment_key)
    assert raw, f"{environment_key} must name the immutable production artifact directory"
    path = Path(raw).expanduser().resolve()
    assert path.is_dir(), f"{environment_key} is not a directory: {path}"
    return path


def _issue868_count_boundary_transitions(
    state: dict[str, Any],
    previous: tuple[int, str] | None,
) -> tuple[int, tuple[int, str] | None]:
    rows = state["batch_rows"]
    assert isinstance(rows, list)
    transitions = 0
    for row in rows:
        assert isinstance(row, list)
        identity = (row[0], row[1])
        if previous is not None and identity != previous:
            transitions += 1
        previous = identity
    return transitions, previous


@pytest.mark.very_slow
@pytest.mark.skipif(
    os.environ.get("RUN_ISSUE868_H100_DDP") != "1",
    reason="requires the explicit RUN_ISSUE868_H100_DDP=1 production gate",
)
def test_issue868_eight_h100_production_artifact_staging_gate() -> None:
    """Exercise exact staging/resume/cancellation contracts on eight H100 ranks."""
    assert os.environ.get("WORLD_SIZE") == str(_ISSUE868_WORLD_SIZE), (
        "issue #868 gate must be launched with torchrun --nproc_per_node=8"
    )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert t.cuda.device_count() >= _ISSUE868_WORLD_SIZE
    device = t.device("cuda", local_rank)
    assert "H100" in t.cuda.get_device_name(device), (
        f"issue #868 production gate requires H100 devices, got {t.cuda.get_device_name(device)!r}"
    )
    packed_dir = _issue868_artifact_dir("ISSUE868_PACKED_ROW_DIR")
    selection_dir = _issue868_artifact_dir("ISSUE868_SELECTION_DIR")

    staged: PrecomputedCudaStagingLoader | None = None
    legacy: Iterator[PrecomputedBatch] | None = None
    resumed_staged: PrecomputedCudaStagingLoader | None = None
    resumed_legacy: Iterator[PrecomputedBatch] | None = None
    try:
        t.cuda.set_device(device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        legacy = _loader(
            str(packed_dir),
            str(selection_dir),
            rank=rank,
            device=str(device),
            world_size=_ISSUE868_WORLD_SIZE,
        )
        cancel_event = threading.Event()
        staged = PrecomputedCudaStagingLoader(
            _loader(
                str(packed_dir),
                str(selection_dir),
                rank=rank,
                device="cpu",
                world_size=_ISSUE868_WORLD_SIZE,
            ),
            device=device,
            host_queue_bytes=_ISSUE868_HOST_QUEUE_BYTES,
            pinned_slots=_ISSUE868_PINNED_SLOTS,
            expected_batch_size=_BATCH_SIZE,
            expected_context_len=_CONTEXT_LEN,
            expected_pad_f=_PAD_F,
            expected_n_seg=_N_SEG,
            remote_read_cancel_event=cancel_event,
        )

        boundary_transitions = 0
        previous_identity: tuple[int, str] | None = None
        checkpoint_state: dict[str, Any] | None = None
        for _ in range(_ISSUE868_MAX_BATCHES):
            actual = next(staged)
            expected = next(legacy)
            _assert_batch_equal(actual, expected)
            rows = actual[9]["batch_rows"]
            assert all(row[2] % _ISSUE868_WORLD_SIZE == rank for row in rows)
            added, previous_identity = _issue868_count_boundary_transitions(
                actual[9],
                previous_identity,
            )
            boundary_transitions += added
            checkpoint_state = actual[9]
            if boundary_transitions >= _ISSUE868_BOUNDARIES:
                break
        assert boundary_transitions >= _ISSUE868_BOUNDARIES
        assert checkpoint_state is not None

        # Snapshot the exact next batch before closing the non-exhausted loaders.
        continuation = next(legacy)
        _assert_batch_equal(next(staged), continuation)
        staged_stats = staged.stats()
        assert staged_stats.host_queue_peak_bytes <= _ISSUE868_HOST_QUEUE_BYTES
        staged.close()
        staged.close()
        assert cancel_event.is_set()
        assert all(not thread.is_alive() for thread in staged._runtime.threads)
        _close_loader(legacy)
        legacy = None
        staged = None

        resume_cancel_event = threading.Event()
        resumed_legacy = _loader(
            str(packed_dir),
            str(selection_dir),
            rank=rank,
            device=str(device),
            world_size=_ISSUE868_WORLD_SIZE,
            resume_state_dict=checkpoint_state,
        )
        resumed_staged = PrecomputedCudaStagingLoader(
            _loader(
                str(packed_dir),
                str(selection_dir),
                rank=rank,
                device="cpu",
                world_size=_ISSUE868_WORLD_SIZE,
                resume_state_dict=checkpoint_state,
            ),
            device=device,
            host_queue_bytes=_ISSUE868_HOST_QUEUE_BYTES,
            pinned_slots=_ISSUE868_PINNED_SLOTS,
            expected_batch_size=_BATCH_SIZE,
            expected_context_len=_CONTEXT_LEN,
            expected_pad_f=_PAD_F,
            expected_n_seg=_N_SEG,
            remote_read_cancel_event=resume_cancel_event,
        )
        resumed_actual = next(resumed_staged)
        resumed_expected = next(resumed_legacy)
        _assert_batch_equal(resumed_actual, resumed_expected)
        _assert_batch_equal(resumed_actual, continuation)

        # Close while both infinite loaders still have work. This is the
        # interruption path used by an early training shutdown.
        resumed_staged.close()
        resumed_staged.close()
        assert resume_cancel_event.is_set()
        assert all(not thread.is_alive() for thread in resumed_staged._runtime.threads)
        _close_loader(resumed_legacy)
        resumed_legacy = None
        resumed_staged = None

        completed = t.ones((), dtype=t.int64, device=device)
        dist.all_reduce(completed)
        assert completed.item() == _ISSUE868_WORLD_SIZE
    finally:
        if staged is not None:
            staged.close()
        if legacy is not None:
            _close_loader(legacy)
        if resumed_staged is not None:
            resumed_staged.close()
        if resumed_legacy is not None:
            _close_loader(resumed_legacy)
        if dist.is_initialized():
            dist.destroy_process_group()
