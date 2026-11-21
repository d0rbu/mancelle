"""Regression tests for atomic checkpoint save semantics.

These tests guard against a prior bug where ``save_checkpoint`` wrote
``model_{step:06d}.pt`` and ``meta_{step:06d}.json`` directly to their final
paths. A crash mid-write (OOM, SIGKILL, disk full, network partition) left a
torn file that ``find_last_step`` happily returned as the "latest" step,
causing silent corruption on resume.

The fix writes to ``<final>.tmp``, fsync's the file descriptor, then
``os.replace``s into the final path. ``find_last_step`` only globs for
``model_*.pt`` (not ``model_*.pt.tmp``), so a failed mid-write never produces a
discoverable final file.
"""

from __future__ import annotations

import glob
import json
import os
from unittest.mock import patch

import pytest
import torch

from nanochat.checkpoint_manager import (
    find_last_step,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# 1. No torn final file on mid-write crash
# ---------------------------------------------------------------------------


class TestSaveCheckpointAtomic:
    """``save_checkpoint`` must not leave a torn ``model_*.pt`` on crash."""

    def test_model_save_crash_leaves_no_final_file(self, tmp_path):
        """When torch.save raises during write, no model_*.pt exists in the final path."""
        checkpoint_dir = str(tmp_path / "ckpt")
        state = {"w": torch.zeros(4)}
        meta = {"step": 0, "user_config": {}}

        real_torch_save = torch.save

        def broken_save(obj, path, *args, **kwargs):
            # Simulate a partial write: create the file with a few bytes, then raise.
            # The bug: if we write straight to the final path, this debris stays.
            # The fix: we write to path.tmp; the final path must never exist.
            with open(path, "wb") as f:
                f.write(b"\x00" * 16)
            raise RuntimeError("simulated mid-write crash")

        with patch("nanochat.checkpoint_manager.torch.save", side_effect=broken_save):
            with pytest.raises(RuntimeError, match="simulated mid-write crash"):
                save_checkpoint(checkpoint_dir, 0, state, None, meta, rank=0)

        # CRITICAL ASSERTION: no final model_*.pt may exist, otherwise
        # find_last_step would discover torn bytes and return them.
        final_models = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
        final_models = [p for p in final_models if not p.endswith(".tmp")]
        assert final_models == [], (
            f"save_checkpoint left torn file(s) at final path: {final_models}"
        )

    def test_find_last_step_does_not_return_torn_file(self, tmp_path):
        """After a crashed save, find_last_step must not find the torn file."""
        checkpoint_dir = str(tmp_path / "ckpt")
        # First, save a clean checkpoint at step 0.
        state = {"w": torch.zeros(4)}
        meta = {"step": 0, "user_config": {}}
        save_checkpoint(checkpoint_dir, 0, state, None, meta, rank=0)
        assert find_last_step(checkpoint_dir) == 0

        real_torch_save = torch.save

        def broken_save(obj, path, *args, **kwargs):
            with open(path, "wb") as f:
                f.write(b"\x00" * 16)
            raise RuntimeError("simulated crash on step 100")

        # Now, attempt to save step 100 with a simulated crash.
        with patch("nanochat.checkpoint_manager.torch.save", side_effect=broken_save):
            with pytest.raises(RuntimeError, match="simulated crash on step 100"):
                save_checkpoint(
                    checkpoint_dir, 100, state, None,
                    {"step": 100, "user_config": {}}, rank=0,
                )

        # find_last_step must still return 0 (the last *successful* save), NOT 100.
        last = find_last_step(checkpoint_dir)
        assert last == 0, (
            f"find_last_step returned torn step {last}; expected 0 (last valid)"
        )

    def test_successful_save_still_works(self, tmp_path):
        """After the atomicity fix, a normal save still produces a loadable checkpoint."""
        checkpoint_dir = str(tmp_path / "ckpt")
        state = {"w": torch.arange(8, dtype=torch.float32)}
        meta = {"step": 0, "user_config": {"depth": 4}, "val_bpb": 3.14}

        save_checkpoint(checkpoint_dir, 0, state, None, meta, rank=0)

        # Roundtrip load should succeed bit-exact.
        loaded_sd, _, loaded_meta = load_checkpoint(
            checkpoint_dir, 0, torch.device("cpu")
        )
        assert torch.equal(state["w"], loaded_sd["w"])
        assert loaded_meta["step"] == 0
        assert loaded_meta["val_bpb"] == 3.14

    def test_no_tmp_files_left_on_success(self, tmp_path):
        """A successful save must not leave any ``.tmp`` debris behind."""
        checkpoint_dir = str(tmp_path / "ckpt")
        state = {"w": torch.zeros(4)}
        meta = {"step": 0, "user_config": {}}

        save_checkpoint(checkpoint_dir, 0, state, None, meta, rank=0)

        leftover_tmps = glob.glob(os.path.join(checkpoint_dir, "*.tmp"))
        assert leftover_tmps == [], (
            f"save_checkpoint left .tmp debris: {leftover_tmps}"
        )

    def test_meta_json_crash_leaves_no_final_meta(self, tmp_path):
        """A crash while writing meta_*.json must not leave a torn final meta file."""
        checkpoint_dir = str(tmp_path / "ckpt")
        state = {"w": torch.zeros(4)}
        meta = {"step": 0, "user_config": {}}

        # Let torch.save succeed, but crash while writing JSON.
        real_json_dump = json.dump
        call_count = {"n": 0}

        def broken_json_dump(obj, fp, *args, **kwargs):
            call_count["n"] += 1
            # Write a few partial bytes, then raise.
            fp.write('{"partial')
            raise RuntimeError("simulated meta write crash")

        with patch("nanochat.checkpoint_manager.json.dump", side_effect=broken_json_dump):
            with pytest.raises(RuntimeError, match="simulated meta write crash"):
                save_checkpoint(checkpoint_dir, 0, state, None, meta, rank=0)

        # No final meta_*.json may exist.
        final_metas = glob.glob(os.path.join(checkpoint_dir, "meta_*.json"))
        final_metas = [p for p in final_metas if not p.endswith(".tmp")]
        assert final_metas == [], (
            f"save_checkpoint left torn meta file(s): {final_metas}"
        )

    def test_optimizer_crash_leaves_no_final_optim(self, tmp_path):
        """A crash while writing the optimizer file must not leave a torn optim_*.pt."""
        checkpoint_dir = str(tmp_path / "ckpt")
        state = {"w": torch.zeros(4)}
        opt_state = {"state": {}, "param_groups": []}
        meta = {"step": 0, "user_config": {}}

        # We want the MODEL save to succeed (rank 0 writes model first), then
        # the OPTIMIZER save to fail. Track call count.
        real_torch_save = torch.save
        calls = {"n": 0}

        def sometimes_broken_save(obj, path, *args, **kwargs):
            calls["n"] += 1
            if "optim_" in os.path.basename(path):
                with open(path, "wb") as f:
                    f.write(b"\x00" * 16)
                raise RuntimeError("simulated optim crash")
            return real_torch_save(obj, path, *args, **kwargs)

        with patch("nanochat.checkpoint_manager.torch.save", side_effect=sometimes_broken_save):
            with pytest.raises(RuntimeError, match="simulated optim crash"):
                save_checkpoint(
                    checkpoint_dir, 0, state, opt_state, meta, rank=0
                )

        final_optims = glob.glob(os.path.join(checkpoint_dir, "optim_*.pt"))
        final_optims = [p for p in final_optims if not p.endswith(".tmp")]
        assert final_optims == [], (
            f"save_checkpoint left torn optim file(s): {final_optims}"
        )
