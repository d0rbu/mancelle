"""Unit tests for the empty-parent-dir handling of _recover_model_config
and _recover_parent_user_config (issue #443).

The pre-fix bug: ``find_last_step(parent_dir)`` raises ``FileNotFoundError``
on an empty existing dir, which propagates up and crashes ancestor walks
instead of falling through to the documented "Strategy 2" fallback.

Post-fix: both helpers walk through an empty parent dir cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanochat.checkpoint_manager import (
    _recover_model_config,
    _recover_parent_user_config,
)


def _make_meta(tmp_path: Path, step: int, body: dict) -> Path:
    """Write a meta_{step:06d}.json to tmp_path and return its path."""
    meta_path = tmp_path / f"meta_{step:06d}.json"
    meta_path.write_text(json.dumps(body), encoding="utf-8")
    return meta_path


def _make_model_pt(tmp_path: Path, step: int) -> Path:
    """Touch a placeholder model_{step:06d}.pt at tmp_path."""
    model_path = tmp_path / f"model_{step:06d}.pt"
    model_path.write_bytes(b"")
    return model_path


def test_recover_model_config_falls_through_on_empty_parent(tmp_path):
    """An empty parent dir must not crash; the helper should fall through
    to Strategy 2 (reconstruct from user_config)."""
    # Tree: <tmp_path>/empty_parent/<sft_dir>/
    # The parent has no model_*.pt files, but the sft_dir's meta carries
    # a user_config sufficient to reconstruct.
    empty_parent = tmp_path / "empty_parent"
    empty_parent.mkdir()
    sft_dir = empty_parent / "sft"
    sft_dir.mkdir()

    meta_data = {
        "user_config": {
            "depth": 12,
            "aspect_ratio": 64,
            "head_dim": 128,
            "max_seq_len": 2048,
            "window_pattern": "SSSL",
        }
    }

    result = _recover_model_config(str(sft_dir), meta_data)
    # Reconstructed-from-user-config shape.
    assert result["n_layer"] == 12
    assert result["n_embd"] == 12 * 64
    assert result["sequence_len"] == 2048
    assert result["window_pattern"] == "SSSL"
    assert result["vocab_size"] == 32768


def test_recover_model_config_uses_parent_when_populated(tmp_path):
    """Positive control: when the parent has a checkpoint, Strategy 1
    should fire and return the parent's model_config."""
    parent = tmp_path / "parent"
    parent.mkdir()
    sft_dir = parent / "sft"
    sft_dir.mkdir()
    parent_meta = {
        "model_config": {
            "sequence_len": 1024,
            "vocab_size": 32768,
            "n_layer": 24,
            "n_head": 12,
            "n_kv_head": 12,
            "n_embd": 1536,
            "window_pattern": "L",
        },
        "user_config": {},
    }
    _make_meta(parent, step=100, body=parent_meta)
    _make_model_pt(parent, step=100)

    meta_data = {"user_config": {}}
    result = _recover_model_config(str(sft_dir), meta_data)
    assert result["n_layer"] == 24
    assert result["sequence_len"] == 1024


def test_recover_parent_user_config_raises_clear_error_on_empty_parent(tmp_path):
    """An empty parent dir means the helper has nothing to recover from.
    The documented behaviour is to raise FileNotFoundError with the
    'Cannot recover encoder/cross_attention' message — distinct from
    find_last_step's 'No checkpoints found' message.

    Pre-fix bug: the helper would crash inside find_last_step with the
    wrong error message (and at the wrong call site).
    """
    empty_parent = tmp_path / "empty_parent"
    empty_parent.mkdir()
    sft_dir = empty_parent / "sft"
    sft_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        _recover_parent_user_config(str(sft_dir))
    # Locks the post-fix error path: the documented "no parent checkpoint"
    # message, not find_last_step's "No checkpoints found" message.
    assert "Cannot recover encoder/cross_attention config" in str(excinfo.value)


def test_recover_parent_user_config_returns_user_config_when_populated(tmp_path):
    """Positive control: parent has a checkpoint with user_config → helper returns it."""
    parent = tmp_path / "parent"
    parent.mkdir()
    sft_dir = parent / "sft"
    sft_dir.mkdir()
    parent_meta = {
        "user_config": {
            "encoder": {"n_embd": 128, "depth": 2, "head_dim": 64},
            "cross_attention": {"frequency": 2},
            "num_buckets": 5,
        }
    }
    _make_meta(parent, step=50, body=parent_meta)
    _make_model_pt(parent, step=50)

    result = _recover_parent_user_config(str(sft_dir))
    assert result["encoder"]["depth"] == 2
    assert result["num_buckets"] == 5


def test_recover_model_config_handles_parent_dir_with_meta_but_no_model_pt(tmp_path):
    """Edge case: parent has a meta_*.json but no matching model_*.pt
    (incomplete checkpoint). find_last_step would raise; the soft variant
    should return None, falling through to Strategy 2.
    """
    parent = tmp_path / "parent"
    parent.mkdir()
    sft_dir = parent / "sft"
    sft_dir.mkdir()
    # Meta without model.pt — orphaned, simulating an interrupted save.
    _make_meta(parent, step=200, body={"model_config": {"n_layer": 999}})

    meta_data = {
        "user_config": {
            "depth": 10,
            "aspect_ratio": 64,
            "head_dim": 128,
        }
    }
    result = _recover_model_config(str(sft_dir), meta_data)
    # Strategy 2 reconstructed from user_config — NOT the orphan parent meta.
    assert result["n_layer"] == 10
    assert result["n_layer"] != 999


def test_recover_model_config_falls_through_on_missing_parent(tmp_path):
    """Pre-existing edge case (not the bug we're fixing, but a regression
    target): missing parent dir → Strategy 2 fallback. The pre-fix code
    handled this via the ``os.path.isdir`` guard; we must preserve it."""
    nonexistent_parent = tmp_path / "does_not_exist"
    sft_dir = nonexistent_parent / "sft"
    # Note: sft_dir's parent (nonexistent_parent) doesn't exist.
    # The helper inspects parent_dir = os.path.dirname(sft_dir) which is
    # nonexistent_parent.

    meta_data = {
        "user_config": {
            "depth": 8,
            "aspect_ratio": 64,
            "head_dim": 128,
        }
    }
    result = _recover_model_config(str(sft_dir), meta_data)
    assert result["n_layer"] == 8
