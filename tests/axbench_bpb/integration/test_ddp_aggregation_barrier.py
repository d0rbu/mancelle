"""Verify _run_stage(BPB_SCORE) issues dist.barrier() before returning
when running multi-rank.

A monkeypatched dist module records every barrier() call so the test
can assert ordering without spinning up a real process group.

Why these barriers matter: when stage=all, rank 0 transitions from
BPB_SCORE -> BPB_AGGREGATE without waiting for non-rank-0 workers to
finish flushing their batch_NNNNNN.parquet shards. Rank 0 then reads
a subset of the parquet shards and produces a partial leaderboard.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval import axbench_bpb_eval as eval_module
from scripts.eval.axbench_bpb_eval import (
    _maybe_barrier,
    _propagate_rank0_derive,
    _run_stage,
)
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.types import AxBenchBpbError, Stage, SteeringMode
from tests.statistics_doubles import fixed_statistics_double


@dataclass
class _BarrierRecorder:
    """Records dist.barrier() invocations against the monkeypatched dist module.

    Each call appends a sentinel string so tests can assert ordering relative to
    other side-effects (e.g. "stage handler returned" recorded by a stub
    handler).
    """
    events: list[str]

    def barrier(self) -> None:
        self.events.append("barrier")


class _FakeDist:
    """Stand-in for ``torch.distributed`` whose state is fully test-owned.

    Drop-in for the ``dist`` reference inside ``scripts.eval.axbench_bpb_eval``;
    only the four attributes we touch (``is_available``, ``is_initialized``,
    ``barrier``, plus rank/world helpers if needed) are implemented.
    """
    def __init__(
        self, *, available: bool, initialized: bool, recorder: _BarrierRecorder,
    ) -> None:
        self._available = available
        self._initialized = initialized
        self._recorder = recorder

    def is_available(self) -> bool:
        return self._available

    def is_initialized(self) -> bool:
        return self._initialized

    def barrier(self) -> None:
        self._recorder.barrier()


def _make_cfg(*, stage: str = "all") -> AxBenchBpbConfig:
    return AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        offload_between_stages=False,
        stage=Stage.coerce(stage),
    )


def _install_fake_dist(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool = True,
    initialized: bool = True,
) -> _BarrierRecorder:
    """Replace the module-level ``dist`` reference with a fake; return its recorder."""
    recorder = _BarrierRecorder(events=[])
    fake = _FakeDist(
        available=available, initialized=initialized, recorder=recorder,
    )
    monkeypatch.setattr(eval_module, "dist", fake)
    return recorder


def _complete_samples() -> dict:
    """Return both required classes for the mini source concept in both modes."""

    samples = [
        SimpleNamespace(content_class="concept_rich"),
        SimpleNamespace(content_class="concept_avoiding"),
    ]
    return {
        (0, SteeringMode.POSITIVE): samples,
        (0, SteeringMode.NEGATIVE): list(samples),
    }


def test_maybe_barrier_no_op_for_world_size_1(monkeypatch):
    """Single-rank runs must not call dist.barrier() (and shouldn't even
    consult dist availability)."""
    recorder = _install_fake_dist(monkeypatch)
    _maybe_barrier(world_size=1)
    assert recorder.events == []


def test_maybe_barrier_calls_barrier_when_multi_rank_and_initialized(monkeypatch):
    """world_size>1 + dist available + dist initialized => barrier() fires."""
    recorder = _install_fake_dist(monkeypatch, available=True, initialized=True)
    _maybe_barrier(world_size=2)
    assert recorder.events == ["barrier"]


def test_maybe_barrier_no_op_when_dist_not_available(monkeypatch):
    """Even with world_size>1, no barrier is issued when dist is unavailable."""
    recorder = _install_fake_dist(monkeypatch, available=False, initialized=False)
    _maybe_barrier(world_size=2)
    assert recorder.events == []


def test_maybe_barrier_no_op_when_dist_not_initialized(monkeypatch):
    """Even with world_size>1, no barrier is issued when dist is uninitialized."""
    recorder = _install_fake_dist(monkeypatch, available=True, initialized=False)
    _maybe_barrier(world_size=2)
    assert recorder.events == []


def test_bpb_score_stage_barrier_after_run_handler(monkeypatch):
    """In BPB_SCORE handler under multi-rank, dist.barrier() fires AFTER the
    run_bpb_score_stage call returns and before _run_stage itself returns.

    The intent of the barrier is: "all ranks finished writing their parquet
    shards"; therefore it must come AFTER the stage handler. We assert this
    ordering by tracking ``run_bpb_score_stage`` invocation against the
    barrier event in a single events list.
    """
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=2)

    recorder = _install_fake_dist(monkeypatch)

    # Stub all the lazy resources used inside the BPB_SCORE branch so the
    # test never tries to actually load models. We are only asserting the
    # control-flow ordering.
    monkeypatch.setattr(
        eval_module, "_load_context_segments_by_hash", lambda _rs: {},
    )
    monkeypatch.setattr(eval_module, "_build_token_bytes", lambda _tok: None)
    monkeypatch.setattr(
        eval_module,
        "_build_samples_by_concept_mode",
        lambda *a, **k: _complete_samples(),
    )
    monkeypatch.setattr(eval_module, "_build_coefficient_grids",
                        lambda *a, **k: {})
    monkeypatch.setattr(eval_module, "filter_methods", lambda *a, **k: [])
    monkeypatch.setattr(eval_module, "_build_all_methods", list)
    monkeypatch.setattr(
        eval_module, "detect_target_model_metadata",
        lambda _model, _cfg: type("M", (), {"negative_selection": None})(),
    )

    # Stub the bundle accessors so RuntimeState.decoder_only / encdec are no-ops.
    fake_bundle = type(
        "B", (), {
            "model": object(),
            "tokenizer": object(),
            "max_seq_len": 64,
            "target_name": "decoder_only",
        },
    )()
    monkeypatch.setattr(RuntimeState, "decoder_only", lambda self: fake_bundle)
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: fake_bundle)
    monkeypatch.setattr(RuntimeState, "inactive_sampler", lambda self: None)

    # Stub the source-paths concepts.json reader by patching the source_paths
    # accessor to return a fake whose concepts_json returns a tmp file.
    tmp = Path(tempfile.mkdtemp())
    concepts_file = tmp / "concepts.json"
    concepts_file.write_text(json.dumps([]))
    fake_sp = type(
        "SP", (), {"concepts_json": lambda self: concepts_file}
    )()
    monkeypatch.setattr(RuntimeState, "source_paths", lambda self: fake_sp)

    # Capture ordering: stage handler must fire BEFORE the barrier.
    def _fake_run_bpb_score_stage(**kwargs):
        recorder.events.append("run_bpb_score_stage")

    monkeypatch.setattr(
        eval_module, "run_bpb_score_stage", _fake_run_bpb_score_stage,
    )

    _run_stage(Stage.BPB_SCORE, rs)

    assert recorder.events == ["run_bpb_score_stage", "barrier"]


def test_bpb_score_no_barrier_when_world_size_is_1(monkeypatch):
    """Single-rank runs do not call dist.barrier() at the end of BPB_SCORE."""
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    recorder = _install_fake_dist(monkeypatch)

    monkeypatch.setattr(
        eval_module, "_load_context_segments_by_hash", lambda _rs: {},
    )
    monkeypatch.setattr(eval_module, "_build_token_bytes", lambda _tok: None)
    monkeypatch.setattr(
        eval_module,
        "_build_samples_by_concept_mode",
        lambda *a, **k: _complete_samples(),
    )
    monkeypatch.setattr(eval_module, "_build_coefficient_grids",
                        lambda *a, **k: {})
    monkeypatch.setattr(eval_module, "filter_methods", lambda *a, **k: [])
    monkeypatch.setattr(eval_module, "_build_all_methods", list)
    monkeypatch.setattr(
        eval_module, "detect_target_model_metadata",
        lambda _model, _cfg: type("M", (), {"negative_selection": None})(),
    )

    fake_bundle = type(
        "B", (), {
            "model": object(),
            "tokenizer": object(),
            "max_seq_len": 64,
            "target_name": "decoder_only",
        },
    )()
    monkeypatch.setattr(RuntimeState, "decoder_only", lambda self: fake_bundle)
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: fake_bundle)
    monkeypatch.setattr(RuntimeState, "inactive_sampler", lambda self: None)

    tmp = Path(tempfile.mkdtemp())
    concepts_file = tmp / "concepts.json"
    concepts_file.write_text(json.dumps([]))
    fake_sp = type(
        "SP", (), {"concepts_json": lambda self: concepts_file}
    )()
    monkeypatch.setattr(RuntimeState, "source_paths", lambda self: fake_sp)

    monkeypatch.setattr(
        eval_module, "run_bpb_score_stage",
        lambda **kw: recorder.events.append("run_bpb_score_stage"),
    )

    _run_stage(Stage.BPB_SCORE, rs)

    # Stage handler still ran; barrier did NOT.
    assert recorder.events == ["run_bpb_score_stage"]


def test_stage0_emits_barriers_around_rank0_manifest(monkeypatch, tmp_path):
    """Verify rank 0 emits the documented event sequence: build the
    sharded extraction, barrier, write the manifest, second barrier.

    Mirrors the _FakeDist barrier-emission tests for Stages 1/2/3.
    Uses the recorder (rather than a real gloo group) so the test
    deterministically captures the dispatch order under multi-rank.
    """
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=2)

    recorder = _install_fake_dist(monkeypatch, available=True, initialized=True)

    def _fake_build_context_cache(**kwargs):
        recorder.events.append("build_context_cache")

    def _fake_write_manifest(**kwargs):
        recorder.events.append("write_context_cache_manifest")

    monkeypatch.setattr(
        eval_module, "build_context_cache", _fake_build_context_cache,
    )
    monkeypatch.setattr(
        eval_module, "write_context_cache_manifest", _fake_write_manifest,
    )
    # Stub the heavy upstream helpers that fire on every rank (stats +
    # boundaries) and the rank-0 helpers (tokenizer guards + prefix
    # collection) so they don't hit the fixture file system.
    monkeypatch.setattr(
        eval_module,
        "load_statistics",
        lambda *_a, **_kw: fixed_statistics_double(per_feature_stats=[]),
    )
    monkeypatch.setattr(
        eval_module, "build_boundaries_from_stats",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        eval_module, "collect_unique_prefixes", lambda *_a, **_kw: {},
    )
    # Slice 17 audit (2026-05-05): Stage 0 calls decoder_only()
    # plus the tokenizer-checksum + BOS-length guards. Stub them.
    from unittest.mock import MagicMock as _MagicMock

    from steering.eval.axbench_bpb.runtime_state import RuntimeState
    from steering.eval.axbench_bpb.teacher_force import TargetModelBundle

    _fake_bundle = TargetModelBundle(
        model=_MagicMock(), tokenizer=_MagicMock(),
        max_seq_len=2048, target_name="decoder_only",
    )
    monkeypatch.setattr(
        RuntimeState, "decoder_only", lambda self: _fake_bundle,
    )
    monkeypatch.setattr(
        eval_module, "assert_tokenizer_checksum_matches",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        eval_module, "assert_bos_lengths_match", lambda *_a, **_kw: None,
    )

    eval_module._stage_prepare_context_cache(rs)

    # Parallel Stage 1 sequence: build (all-rank sharded), barrier (all
    # ranks finish extraction), manifest write (rank 0), barrier (all
    # ranks see manifest).
    assert recorder.events == [
        "build_context_cache",
        "barrier",
        "write_context_cache_manifest",
        "barrier",
    ], (
        f"Stage 0 on rank 0 must emit build → barrier → manifest → "
        f"barrier under DDP; got {recorder.events}"
    )


def test_stage0_non_rank_0_emits_build_and_barriers_no_manifest(monkeypatch, tmp_path):
    """Non-rank-0 worker participates in build_context_cache (sharded
    extraction) and reaches both barriers, but does NOT write the
    manifest. The fake recorder shows: build → barrier → barrier.
    """
    from tests.axbench_bpb._mini_runtime import build_mini_runtime_state

    rs = build_mini_runtime_state(run_root=tmp_path, rank=1, world_size=2)

    recorder = _install_fake_dist(monkeypatch, available=True, initialized=True)

    def _fake_build_context_cache(**kwargs):
        recorder.events.append("build_context_cache")

    def _fake_write_manifest(**kwargs):
        recorder.events.append("write_context_cache_manifest")

    monkeypatch.setattr(
        eval_module, "build_context_cache", _fake_build_context_cache,
    )
    monkeypatch.setattr(
        eval_module, "write_context_cache_manifest", _fake_write_manifest,
    )
    monkeypatch.setattr(
        eval_module,
        "load_statistics",
        lambda *_a, **_kw: fixed_statistics_double(per_feature_stats=[]),
    )
    monkeypatch.setattr(
        eval_module, "build_boundaries_from_stats",
        lambda *_a, **_kw: None,
    )

    eval_module._stage_prepare_context_cache(rs)

    assert recorder.events == [
        "build_context_cache",
        "barrier",
        "barrier",
    ], (
        f"Non-rank-0 must run sharded build + both barriers, no manifest; "
        f"got {recorder.events}"
    )


class _FakeDistBroadcast:
    """Fake ``dist`` whose ``broadcast_object_list`` stamps a rank-0 outcome.

    ``rank0_failed`` is what rank 0 would have put on the wire; the fake writes it
    into the caller's payload so a non-rank-0 worker sees rank 0's verdict.
    """

    def __init__(self, *, rank0_failed: bool) -> None:
        self._rank0_failed = rank0_failed
        self.broadcast_called = False

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def broadcast_object_list(self, payload, src: int = 0) -> None:
        self.broadcast_called = True
        payload[0] = self._rank0_failed


def test_propagate_rank0_derive_reraises_captured_error_single_rank():
    """Single-rank: a captured derive error is re-raised verbatim (fail loud)."""
    with pytest.raises(RuntimeError, match="derive boom"):
        _propagate_rank0_derive(RuntimeError("derive boom"), world_size=1)


def test_propagate_rank0_derive_noop_on_success_single_rank():
    """Single-rank: no error, no raise."""
    _propagate_rank0_derive(None, world_size=1)


def test_propagate_rank0_derive_broadcasts_before_reraising_rank0_error(monkeypatch):
    """Rank 0's OWN derive error is broadcast BEFORE it is re-raised.

    This is the whole anti-deadlock contract: rank 0 must enter the broadcast
    collective (releasing ranks 1+ from their matching call) and only then raise.
    A refactor that moved ``raise error`` ahead of ``broadcast_object_list`` would
    leave ``broadcast_called`` False here while ranks 1+ hang forever -- so this
    asserts the broadcast actually fired, not merely that the error propagates.
    """
    fake = _FakeDistBroadcast(rank0_failed=True)
    monkeypatch.setattr(eval_module, "dist", fake)

    with pytest.raises(RuntimeError, match="rank0 derive boom"):
        _propagate_rank0_derive(RuntimeError("rank0 derive boom"), world_size=2)

    assert fake.broadcast_called is True


def test_propagate_rank0_derive_aborts_all_ranks_when_rank0_failed(monkeypatch):
    """A non-rank-0 worker (no local error) still raises when rank 0 signals
    failure over the broadcast -- so it aborts loudly instead of sailing into the
    next collective and hanging."""
    fake = _FakeDistBroadcast(rank0_failed=True)
    monkeypatch.setattr(eval_module, "dist", fake)
    with pytest.raises(AxBenchBpbError, match="rank 0 failed to derive feature-DiffMean"):
        _propagate_rank0_derive(None, world_size=2)
    assert fake.broadcast_called is True


def test_propagate_rank0_derive_all_ranks_pass_when_rank0_succeeded(monkeypatch):
    """Broadcast says rank 0 succeeded => every rank continues without raising."""
    fake = _FakeDistBroadcast(rank0_failed=False)
    monkeypatch.setattr(eval_module, "dist", fake)
    _propagate_rank0_derive(None, world_size=2)
    assert fake.broadcast_called is True


def test_barrier_does_not_fire_for_aggregate_stage(monkeypatch):
    """BPB_AGGREGATE itself does not call dist.barrier(): the prior SCORE-end
    barrier covers cross-rank synchronization. Adding another at the start
    of AGGREGATE would be redundant.

    We verify by entering BPB_AGGREGATE on a non-rank-0 worker (which
    short-circuits at the rank-0 check and returns without doing anything,
    including without invoking barrier).
    """
    cfg = _make_cfg()
    # Non-rank-0 worker
    rs = RuntimeState(cfg=cfg, ddp_rank=1, ddp_world_size=2)

    recorder = _install_fake_dist(monkeypatch)

    _run_stage(Stage.BPB_AGGREGATE, rs)

    assert recorder.events == []
