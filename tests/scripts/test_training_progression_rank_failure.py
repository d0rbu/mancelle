"""Tests for multi-rank failure classification in training_progression.

These tests exercise `collect_rank_results`, the helper that dispatches per-rank
work via a `ThreadPoolExecutor`, classifies rank exceptions, and raises if any
rank fails. We use `ThreadPoolExecutor` for deterministic parallel rank
simulation in CI without requiring CUDA.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from loguru import logger
from omegaconf import OmegaConf

from scripts.train.training_progression import (
    CheckpointSpec,
    collect_rank_results,
    collect_results,
)


@pytest.fixture()
def loguru_capture():
    """Capture loguru log records (loguru bypasses stdlib caplog)."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda msg: messages.append(str(msg)),
        level="DEBUG",
    )
    yield messages
    logger.remove(handler_id)


def _make_rank_fn(failing_ranks: set[int], exc_type=RuntimeError):
    def rank_fn(rank: int) -> dict:
        if rank in failing_ranks:
            raise exc_type(f"boom on rank {rank}")
        return {"rank": rank, "value": rank * 10}
    return rank_fn


def test_all_ranks_success_returns_ordered_aggregate():
    fn = _make_rank_fn(set())
    aggregate = collect_rank_results(fn, world_size=4)

    assert aggregate["partial_ranks_failed"] == []
    assert len(aggregate["rank_results"]) == 4
    succeeded_ranks = {r["rank"] for r in aggregate["rank_results"]}
    assert succeeded_ranks == {0, 1, 2, 3}


def test_single_worker_mode_preserves_order():
    calls: list[int] = []

    def rank_fn(rank: int) -> dict:
        calls.append(rank)
        return {"rank": rank}

    aggregate = collect_rank_results(rank_fn, world_size=3, max_workers=1)

    assert calls == [0, 1, 2]
    assert [r["rank"] for r in aggregate["rank_results"]] == [0, 1, 2]


def test_single_rank_failure_raises_with_traceback(loguru_capture):
    """1 of 4 ranks fails: raise with traceback + rank id."""
    fn = _make_rank_fn({2})
    with pytest.raises(RuntimeError) as exc_info:
        collect_rank_results(fn, world_size=4)

    assert "1 of 4 ranks failed" in str(exc_info.value)
    assert "ranks=[2]" in str(exc_info.value)

    combined = "\n".join(loguru_capture)
    assert "rank 2" in combined
    assert "Traceback" in combined
    assert "boom on rank 2" in combined


def test_multiple_rank_failures_raise():
    fn = _make_rank_fn({1, 3})
    with pytest.raises(RuntimeError) as exc_info:
        collect_rank_results(fn, world_size=4)
    assert "2 of 4 ranks failed" in str(exc_info.value)
    assert "ranks=[1, 3]" in str(exc_info.value)


def test_all_ranks_failure_raises():
    """All ranks fail: raise."""
    fn = _make_rank_fn({0, 1, 2, 3})
    with pytest.raises(RuntimeError):
        collect_rank_results(fn, world_size=4)


def test_invalid_world_size_raises():
    with pytest.raises(ValueError, match="world_size must be positive"):
        collect_rank_results(_make_rank_fn(set()), world_size=0)


def test_collect_rank_results_is_wired_into_torchrun_worker(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "model_tag": "d27",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "results_file": str(tmp_path / "results.json"),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
            "evals": "core",
            "num_seeds": 1,
            "base_seed": 1234,
            "dtype": "float32",
            "num_gpus": 2,
            "device_type": "cuda",
            "cleanup_after_eval": False,
        }
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    specs = [CheckpointSpec(step=500, phase="pretrain", subdir="")]

    with (
        patch("scripts.train.training_progression.build_checkpoint_specs", return_value=specs),
        patch("scripts.train.training_progression._needs_eval", return_value=True),
        patch("scripts.train.training_progression._validate_required_context_cache"),
        patch("scripts.train.training_progression.torch.cuda.set_device"),
        patch("scripts.train.training_progression.save_merged_results") as mock_merge,
        patch(
            "scripts.train.training_progression.collect_rank_results",
            side_effect=RuntimeError("sentinel"),
        ) as mock_helper,
    ):
        with pytest.raises(RuntimeError, match="1 of 1 ranks failed"):
            collect_results(cfg)

    assert mock_helper.called
    mock_merge.assert_not_called()
