"""Observability contracts for ``steering.extract_features``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from loguru import logger

import steering.extract_features as ef
from steering.extract_features import (
    _ShardProgress,
    _tqdm_disabled,
    process_shard,
)
from steering.feature_paths import PoolingMode
from steering.refined_types import NonNegativeInt
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)
from tests.steering.test_extract_features_loop import _staging, make_parquet, tiny_params


@contextmanager
def _capture_info() -> Iterator[list[str]]:
    lines: list[str] = []
    handler_id = logger.add(lines.append, format="{message}", level="INFO")
    try:
        yield lines
    finally:
        logger.remove(handler_id)


def test_shard_progress_emits_at_document_milestone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_DOCS", 2)
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_SECONDS", 100.0)
    clock = iter((0.0, 1.0, 2.0))
    monkeypatch.setattr(ef.time, "monotonic", lambda: next(clock))

    with _capture_info() as lines:
        progress = _ShardProgress("shard_00007", total_docs=5)
        assert not progress.emit(docs_done=1, segments_done=3)
        assert progress.emit(docs_done=2, segments_done=6)

    assert len(lines) == 1
    assert "shard=shard_00007" in lines[0]
    assert "docs_done=2" in lines[0]
    assert "docs_total=5" in lines[0]
    assert "progress=0.400" in lines[0]
    assert "segments_done=6" in lines[0]


def test_shard_progress_time_fallback_and_empty_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_DOCS", 10_000)
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_SECONDS", 10.0)
    clock = iter((0.0, 5.0, 11.0, 12.0, 13.0))
    monkeypatch.setattr(ef.time, "monotonic", lambda: next(clock))

    with _capture_info() as lines:
        progress = _ShardProgress("slow", total_docs=100)
        assert not progress.emit(docs_done=1, segments_done=2)
        assert progress.emit(docs_done=2, segments_done=4)
        empty = _ShardProgress("empty", total_docs=0)
        assert empty.emit(docs_done=0, segments_done=0, force=True)

    assert len(lines) == 2
    assert "shard=slow" in lines[0]
    assert "progress=0.020" in lines[0]
    assert "shard=empty" in lines[1]
    assert "progress=1.000" in lines[1]


@pytest.mark.parametrize("isatty,expected_disabled", [(False, True), (True, False)])
def test_tqdm_disable_follows_stderr_tty(
    monkeypatch: pytest.MonkeyPatch,
    isatty: bool,
    expected_disabled: bool,
) -> None:
    class _Stderr:
        def isatty(self) -> bool:
            return isatty

    monkeypatch.setattr(ef.sys, "stderr", _Stderr())
    assert _tqdm_disabled() is expected_disabled


@pytest.mark.parametrize("context_independent_features", [False, True])
@pytest.mark.parametrize("pipeline_post", [False, True])
def test_process_shard_logs_start_progress_and_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context_independent_features: bool,
    pipeline_post: bool,
) -> None:
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_DOCS", 1)
    monkeypatch.setattr(ef, "_EXTRACTION_HEARTBEAT_SECONDS", 10_000.0)
    parquet = make_parquet(tmp_path / "shard_00012.parquet", ["alpha", "beta"])
    model = TinyHookedTransformer()
    sae = TinySAE()

    with _capture_info() as lines:
        process_shard(
            parquet,
            model,  # ty: ignore[invalid-argument-type]
            sae,  # ty: ignore[invalid-argument-type]
            output_dir=str(tmp_path / "out"),
            pipeline_post=pipeline_post,
            pooling=PoolingMode.max_pool,
            gpu_batch_size=1,
            cpu_batch_size=1,
            context_window=64,
            segment_size=16,
            context_independent_features=context_independent_features,
        )

    text = "\n".join(lines)
    assert "extract_features shard start: shard=shard_00012" in text
    assert "extract_features heartbeat: shard=shard_00012" in text
    assert "phase=extract_done" in text
    assert "extract_features shard save start: shard=shard_00012" in text


class _SpyTqdm:
    def __init__(self, iterable, **kwargs):
        self._iterable = iterable
        self.disabled_values.append(kwargs["disable"])

    disabled_values: list[bool] = []

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, **kwargs) -> None:
        del kwargs


def test_all_extraction_bars_receive_tty_disable_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _SpyTqdm.disabled_values = []
    monkeypatch.setattr(ef, "tqdm", _SpyTqdm)
    monkeypatch.setattr(ef, "_tqdm_disabled", lambda: True)
    parquet = make_parquet(tmp_path / "shard_00000.parquet", ["alpha", "beta"])

    ef.run_extraction_loop(
        tiny_params(tmp_path / "out"),
        [parquet],
        rank=0,
        world_size=1,
        local_rank=0,
        shard_start=NonNegativeInt.parse(0),
        shard_end=NonNegativeInt.parse(1),
        staging_cfg=_staging(tmp_path / "scratch", 0, 0),
        profile_enabled=False,
        profile_output=str(tmp_path / "profile"),
        flops_per_token=1.0,
        gpu_peak_flops=1.0,
    )

    assert _SpyTqdm.disabled_values == [True, True, True]
