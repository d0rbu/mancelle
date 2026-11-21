"""Checkpoint save/load timing lives inside checkpoint_manager.

Motivated by the 2026-08-28 hstor013 incidents: checkpoint I/O rank skew is
what trips the NCCL watchdog, and it was invisible. save_checkpoint and
load_checkpoint now time themselves and either log a local line (default,
collective-free — safe for rank-conditional callers like chat_sft's
rank-0-only interval save) or, with report_rank_skew=True, gather durations
across the process group (all-rank callers only).
"""

from io import StringIO

import torch
from loguru import logger

import steering.profiling.rank_skew as rank_skew_module
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint


def _capture(level: str = "INFO"):
    sink = StringIO()
    handler_id = logger.add(sink, format="{level} | {message}", level=level)
    return sink, handler_id


def _tiny_state():
    return {"w": torch.arange(4, dtype=torch.float32)}


def test_save_logs_local_timing_by_default(tmp_path):
    sink, handler_id = _capture()
    try:
        save_checkpoint(str(tmp_path), 7, _tiny_state(), _tiny_state(), {"step": 7})
    finally:
        logger.remove(handler_id)
    captured = sink.getvalue()
    assert "checkpoint save (step 7):" in captured
    assert "(this rank)" in captured
    assert (tmp_path / "model_000007.pt").exists()
    assert (tmp_path / "optim_000007_rank0.pt").exists()


def test_load_logs_local_timing_by_default(tmp_path):
    save_checkpoint(str(tmp_path), 9, _tiny_state(), _tiny_state(), {"step": 9})
    sink, handler_id = _capture()
    try:
        model_data, optimizer_data, meta_data = load_checkpoint(
            str(tmp_path), 9, "cpu", load_optimizer=True
        )
    finally:
        logger.remove(handler_id)
    captured = sink.getvalue()
    assert "checkpoint load (step 9):" in captured
    assert "(this rank)" in captured
    assert torch.equal(model_data["w"], torch.arange(4, dtype=torch.float32))
    assert optimizer_data is not None
    assert meta_data["step"] == 9


def test_save_report_rank_skew_gathers_across_ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_skew_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(rank_skew_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr("nanochat.checkpoint_manager.torch.distributed.is_available", lambda: True)
    monkeypatch.setattr(
        "nanochat.checkpoint_manager.torch.distributed.is_initialized", lambda: True
    )
    monkeypatch.setattr("nanochat.checkpoint_manager.torch.distributed.get_world_size", lambda: 3)

    def fake_all_gather_object(out, value):
        out[:] = [float(value), 2.0, 650.0]

    monkeypatch.setattr(rank_skew_module.dist, "all_gather_object", fake_all_gather_object)
    sink, handler_id = _capture(level="DEBUG")
    try:
        save_checkpoint(
            str(tmp_path),
            11,
            _tiny_state(),
            _tiny_state(),
            {"step": 11},
            report_rank_skew=True,
        )
    finally:
        logger.remove(handler_id)
    captured = sink.getvalue()
    assert "checkpoint save (step 11): rank duration min/median/max" in captured
    assert "slowest rank 2" in captured
    assert "risks the NCCL watchdog" in captured  # 650s skew > 300s threshold
    assert "(this rank)" not in captured


def test_load_resolves_step_none_before_labeling(tmp_path):
    save_checkpoint(str(tmp_path), 13, _tiny_state(), None, {"step": 13})
    sink, handler_id = _capture()
    try:
        load_checkpoint(str(tmp_path), None, "cpu")
    finally:
        logger.remove(handler_id)
    assert "checkpoint load (step 13):" in sink.getvalue()
