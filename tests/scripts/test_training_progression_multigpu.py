"""Tests for the multi-GPU torchrun code path in training_progression.

Covers:
- Torchrun relaunch command construction and filtering logic
- Env var clearing and restoration inside torchrun workers
- Checkpoint striping across ranks
- Per-checkpoint result writing and rank-0 merging
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from scripts.train.training_progression import (
    CheckpointSpec,
    _checkpoint_result_path,
    _needs_eval,
    _results_dir,
    _results_path,
    build_checkpoint_specs,
    collect_results,
    load_checkpoint_result,
    merge_checkpoint_results,
    save_checkpoint_result,
    save_merged_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_cfg(tmp_path: Path):
    """Minimal OmegaConf config for multi-GPU testing."""
    return OmegaConf.create(
        {
            "model_tag": "d27",
            "model_family": "decoder",
            "checkpoint_root": "base_checkpoints",
            "checkpoint_prefix": "",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "results_file": str(tmp_path / "results.json"),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [
                {"phase": "pretrain", "subdir": "", "steps": [500, 1000, 1500]},
                {"phase": "sft", "subdir": "standard_sft_checkpoints", "steps": [1000]},
            ],
            "evals": "core,chat",
            "feature_configs": None,
            "segment_size": None,
            "num_seeds": 2,
            "base_seed": 1234,
            "max_per_task": 10,
            "chat_tasks": "ARC-Easy,MMLU",
            "chat_batch_size": 4,
            "chat_max_new_tokens": 32,
            "chat_temperature": 0.0,
            "chat_top_k": 50,
            "chat_num_samples": 1,
            "chat_max_problems": 5,
            "harm_temperature": 0.7,
            "harm_top_k": 50,
            "harm_max_new_tokens": 32,
            "harm_num_benign": 10,
            "harm_judge_model": "test",
            "cleanup_after_eval": False,
            "device_type": "cuda",
            "dtype": "float32",
            "num_gpus": 4,
            "remote": {
                "host": "user@host",
                "port": 22,
                "ssh_key": "~/.ssh/id_rsa",
                "base_path": "/remote/base_checkpoints",
                "local_temp_dir": str(tmp_path / "tmp"),
            },
        }
    )


@pytest.fixture()
def empty_results():
    """Empty results structure."""
    return {"model_tag": "d27", "checkpoints": []}


def _conditioned_eval_cfg(base_cfg=None):
    conditioned = {
        "conditioning": {
            "conditioned_configs": [{"name": "static_delta", "features": {0: 1}}],
            "context_cache_dir": "/tmp/context-cache",
        }
    }
    if base_cfg is None:
        return OmegaConf.create(conditioned)
    return OmegaConf.merge(base_cfg, conditioned)


def _conditioned_evals(
    *,
    core_seeds: tuple[int, ...] = (),
    chat_seeds: tuple[int, ...] = (),
    harmfulness_seeds: tuple[int, ...] = (),
    config_name: str = "static_delta",
) -> dict:
    evals = {}
    for eval_name, eval_seeds in {
        "core": core_seeds,
        "chat": chat_seeds,
        "harmfulness": harmfulness_seeds,
    }.items():
        if not eval_seeds:
            continue
        evals[eval_name] = {
            "conditioned": {
                "configs": {
                    config_name: {"seeds": {str(seed): {} for seed in eval_seeds}},
                },
            },
        }
    return evals


# ---------------------------------------------------------------------------
# Unit tests: checkpoint striping
# ---------------------------------------------------------------------------


class TestCheckpointStriping:
    """Test that pending_specs[rank::world_size] distributes correctly."""

    def test_stripe_evenly_divisible(self):
        """8 specs across 4 ranks => 2 each."""
        specs = [CheckpointSpec(step=i * 100, phase="pretrain", subdir="") for i in range(8)]
        world_size = 4
        all_assigned = []
        for rank in range(world_size):
            my_specs = specs[rank::world_size]
            assert len(my_specs) == 2
            all_assigned.extend(my_specs)
        # Every spec assigned exactly once.
        assert sorted(s.step for s in all_assigned) == sorted(s.step for s in specs)

    def test_stripe_uneven(self):
        """7 specs across 4 ranks => ranks 0-2 get 2, rank 3 gets 1."""
        specs = [CheckpointSpec(step=i * 100, phase="pretrain", subdir="") for i in range(7)]
        world_size = 4
        counts = []
        all_assigned = []
        for rank in range(world_size):
            my_specs = specs[rank::world_size]
            counts.append(len(my_specs))
            all_assigned.extend(my_specs)
        assert counts == [2, 2, 2, 1]
        assert sorted(s.step for s in all_assigned) == sorted(s.step for s in specs)

    def test_stripe_fewer_specs_than_ranks(self):
        """2 specs across 4 ranks => ranks 0-1 get 1 each, ranks 2-3 get 0."""
        specs = [CheckpointSpec(step=500, phase="pretrain", subdir=""),
                 CheckpointSpec(step=1000, phase="sft", subdir="sft")]
        world_size = 4
        counts = []
        for rank in range(world_size):
            counts.append(len(specs[rank::world_size]))
        assert counts == [1, 1, 0, 0]

    def test_stripe_single_rank(self):
        """Single rank gets all specs."""
        specs = [CheckpointSpec(step=i * 100, phase="pretrain", subdir="") for i in range(5)]
        assert specs[0::1] == specs

    def test_stripe_empty_specs(self):
        """Empty spec list => every rank gets empty."""
        specs: list[CheckpointSpec] = []
        for rank in range(4):
            assert specs[rank::4] == []

    def test_stripe_ordering_preserved(self):
        """Each rank's specs maintain relative order from the original list."""
        specs = [CheckpointSpec(step=i, phase="pretrain", subdir="") for i in range(12)]
        for rank in range(3):
            my_specs = specs[rank::3]
            steps = [s.step for s in my_specs]
            assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# Unit tests: env var clearing and restoration
# ---------------------------------------------------------------------------


class TestEnvVarClearingAndRestoration:
    """Test that torchrun env vars are properly saved, cleared, and restored."""

    TORCHRUN_VARS = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]

    def test_env_vars_saved_and_cleared(self):
        """Simulate the env var clearing logic: vars are removed from os.environ."""
        original_values = {
            "RANK": "2",
            "LOCAL_RANK": "2",
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        # Set env vars.
        for var, val in original_values.items():
            os.environ[var] = val

        try:
            # Replicate the clearing logic from collect_results.
            saved_env = {}
            for var in self.TORCHRUN_VARS:
                saved_env[var] = os.environ.pop(var, None)

            # Verify all vars are removed.
            for var in self.TORCHRUN_VARS:
                assert var not in os.environ, f"{var} should be removed"

            # Verify saved values are correct.
            for var, val in original_values.items():
                assert saved_env[var] == val
        finally:
            # Clean up: remove any leftover vars.
            for var in self.TORCHRUN_VARS:
                os.environ.pop(var, None)

    def test_env_vars_restored(self):
        """After work is done, env vars are restored from saved_env."""
        saved_env = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }

        # Make sure vars are not set beforehand.
        for var in self.TORCHRUN_VARS:
            os.environ.pop(var, None)

        try:
            # Replicate the restoration logic.
            for var, val in saved_env.items():
                if val is not None:
                    os.environ[var] = val

            for var, val in saved_env.items():
                assert os.environ.get(var) == val
        finally:
            for var in self.TORCHRUN_VARS:
                os.environ.pop(var, None)

    def test_missing_env_vars_not_restored(self):
        """Vars that were None (not originally set) are not restored."""
        saved_env = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": None,  # was not set
            "MASTER_PORT": None,  # was not set
        }

        for var in self.TORCHRUN_VARS:
            os.environ.pop(var, None)

        try:
            for var, val in saved_env.items():
                if val is not None:
                    os.environ[var] = val

            assert os.environ.get("RANK") == "0"
            assert "MASTER_ADDR" not in os.environ
            assert "MASTER_PORT" not in os.environ
        finally:
            for var in self.TORCHRUN_VARS:
                os.environ.pop(var, None)

    def test_clear_restore_roundtrip(self):
        """Full roundtrip: set -> clear -> work -> restore -> verify."""
        originals = {
            "RANK": "3",
            "LOCAL_RANK": "3",
            "WORLD_SIZE": "8",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "12345",
        }
        for var, val in originals.items():
            os.environ[var] = val

        try:
            # Clear phase.
            saved_env = {}
            for var in self.TORCHRUN_VARS:
                saved_env[var] = os.environ.pop(var, None)

            for var in self.TORCHRUN_VARS:
                assert var not in os.environ

            # Restore phase.
            for var, val in saved_env.items():
                if val is not None:
                    os.environ[var] = val

            for var, val in originals.items():
                assert os.environ[var] == val
        finally:
            for var in self.TORCHRUN_VARS:
                os.environ.pop(var, None)


# ---------------------------------------------------------------------------
# Functional tests: torchrun relaunch detection
# ---------------------------------------------------------------------------


class TestTorchrunRelaunchDetection:
    """Test that collect_results detects RANK env var to decide relaunch vs worker."""

    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("subprocess.run")
    def test_relaunch_when_rank_not_set(
        self, mock_subproc, mock_detect, mock_needs_eval, mock_specs, sample_cfg, tmp_path
    ):
        """When RANK is not in env and num_gpus > 1, subprocess.run is called."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]
        mock_subproc.return_value = MagicMock(returncode=0)

        # Ensure RANK is not set.
        env_backup = os.environ.pop("RANK", None)
        try:
            collect_results(sample_cfg)
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

        mock_subproc.assert_called_once()
        cmd = mock_subproc.call_args[0][0]
        assert "torch.distributed.run" in cmd
        assert "--standalone" in cmd
        assert "model_family=decoder" in cmd
        assert "checkpoint_root=base_checkpoints" in cmd
        assert not any(arg.startswith("source=") for arg in cmd)

        snapshot_arg = next(arg for arg in cmd if arg.startswith("config_snapshot="))
        snapshot = OmegaConf.load(snapshot_arg.partition("=")[2])
        assert snapshot.model_family == "decoder"
        assert snapshot.checkpoint_root == "base_checkpoints"
        assert snapshot.checkpoint_prefix == ""
        assert "source" not in snapshot

    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("subprocess.run")
    def test_relaunch_raises_on_nonzero_exit(
        self, mock_subproc, mock_detect, mock_needs_eval, mock_specs, sample_cfg
    ):
        """Non-zero return code from torchrun raises RuntimeError."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]
        mock_subproc.return_value = MagicMock(returncode=1)

        env_backup = os.environ.pop("RANK", None)
        try:
            with pytest.raises(RuntimeError, match="torchrun exited with code 1"):
                collect_results(sample_cfg)
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("subprocess.run")
    def test_no_relaunch_on_single_gpu(
        self, mock_subproc, mock_detect, mock_needs_eval, mock_specs, sample_cfg
    ):
        """With num_gpus=1, no torchrun relaunch happens."""
        sample_cfg.num_gpus = 1
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]

        env_backup = os.environ.pop("RANK", None)
        try:
            # This will fall through to the single-GPU path which tries to
            # evaluate. We patch evaluate_single_checkpoint to avoid real work.
            with patch("scripts.train.training_progression.evaluate_single_checkpoint") as mock_eval, \
                 patch("scripts.train.training_progression.save_results"):
                mock_eval.return_value = {"core": {"seeds": {}}}
                collect_results(sample_cfg)
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

        mock_subproc.assert_not_called()

    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("subprocess.run")
    def test_no_relaunch_on_cpu_device(
        self, mock_subproc, mock_detect, mock_needs_eval, mock_specs, sample_cfg
    ):
        """With device_type=cpu (not cuda), no torchrun relaunch even with num_gpus>1."""
        sample_cfg.device_type = "cpu"

        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]

        env_backup = os.environ.pop("RANK", None)
        try:
            with patch("scripts.train.training_progression.evaluate_single_checkpoint") as mock_eval, \
                 patch("scripts.train.training_progression.save_results"):
                mock_eval.return_value = {"core": {"seeds": {}}}
                collect_results(sample_cfg)
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

        mock_subproc.assert_not_called()

    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("subprocess.run")
    def test_relaunch_command_contains_nproc(
        self, mock_subproc, mock_detect, mock_needs_eval, mock_specs, sample_cfg
    ):
        """The relaunch command has the right nproc_per_node value."""
        sample_cfg.num_gpus = 8
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]
        mock_subproc.return_value = MagicMock(returncode=0)

        env_backup = os.environ.pop("RANK", None)
        try:
            collect_results(sample_cfg)
        finally:
            if env_backup is not None:
                os.environ["RANK"] = env_backup

        cmd = mock_subproc.call_args[0][0]
        assert "--nproc_per_node=8" in cmd


# ---------------------------------------------------------------------------
# Functional tests: torchrun worker path
# ---------------------------------------------------------------------------


class TestTorchrunWorkerPath:
    """Test behavior when RANK env var IS set (inside a torchrun worker)."""

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_writes_rank_file(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg, tmp_path
    ):
        """Inside a torchrun worker, results are written to a rank-specific file."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
            CheckpointSpec(step=1000, phase="pretrain", subdir=""),
        ]
        mock_eval.return_value = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        mock_dist.init_process_group = MagicMock()
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        # Simulate rank 0 of 2.
        env_vars = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        # Save original env values and set torchrun vars.
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            collect_results(sample_cfg)

            # Rank 0 should have written a per-checkpoint result file.
            ckpt_path = _checkpoint_result_path(sample_cfg, 500, "pretrain")
            assert ckpt_path.exists()

            # And merged into results.json.
            results_path = _results_path(sample_cfg)
            assert results_path.exists()
        finally:
            # Restore original env.
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_non_rank0_does_not_merge(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg, tmp_path
    ):
        """Rank != 0 does not call load_or_init_results/save_results for merge."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
            CheckpointSpec(step=1000, phase="pretrain", subdir=""),
        ]
        mock_eval.return_value = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        mock_dist.init_process_group = MagicMock()
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        env_vars = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            with patch("scripts.train.training_progression.load_or_init_results") as mock_load, \
                 patch("scripts.train.training_progression.save_results") as mock_save:
                mock_load.return_value = {"model_tag": "d27", "checkpoints": []}
                collect_results(sample_cfg)

            # load_or_init_results is called once at the top of collect_results
            # (line 496) for filtering, but rank 1 should NOT call it a second
            # time for the merge phase, and should NOT call save_results.
            assert mock_load.call_count == 1, (
                f"Expected 1 call (initial load), got {mock_load.call_count}"
            )
            mock_save.assert_not_called()
        finally:
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_stripes_specs_correctly(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg, tmp_path
    ):
        """Rank 0 of 2 gets specs at indices 0, 2 and rank 1 gets 1, 3."""
        all_specs = [
            CheckpointSpec(step=100, phase="pretrain", subdir=""),
            CheckpointSpec(step=200, phase="pretrain", subdir=""),
            CheckpointSpec(step=300, phase="pretrain", subdir=""),
            CheckpointSpec(step=400, phase="pretrain", subdir=""),
        ]
        mock_specs.return_value = all_specs
        mock_eval.return_value = {"core": {"val": 1}}
        mock_dist.init_process_group = MagicMock()
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        # Test as rank 0 of 2.
        env_vars = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            with patch("scripts.train.training_progression.load_or_init_results") as mock_load, \
                 patch("scripts.train.training_progression.save_results"):
                mock_load.return_value = {"model_tag": "d27", "checkpoints": []}
                collect_results(sample_cfg)

            # Rank 0 should evaluate steps at indices 0, 2 => steps 100, 300.
            eval_calls = mock_eval.call_args_list
            evaluated_steps = [c[0][0].step for c in eval_calls]
            assert evaluated_steps == [100, 300]
        finally:
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_env_vars_cleared_during_eval(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg, tmp_path
    ):
        """During evaluation, RANK/LOCAL_RANK/WORLD_SIZE are not in os.environ."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
            CheckpointSpec(step=1000, phase="pretrain", subdir=""),
        ]
        mock_dist.init_process_group = MagicMock()
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        captured_env = {}

        def capture_env_during_eval(*args, **kwargs):
            """Capture env state when evaluate_single_checkpoint is called."""
            captured_env["RANK"] = os.environ.get("RANK")
            captured_env["LOCAL_RANK"] = os.environ.get("LOCAL_RANK")
            captured_env["WORLD_SIZE"] = os.environ.get("WORLD_SIZE")
            captured_env["MASTER_ADDR"] = os.environ.get("MASTER_ADDR")
            captured_env["MASTER_PORT"] = os.environ.get("MASTER_PORT")
            return {"core": {"val": 1}}

        mock_eval.side_effect = capture_env_during_eval

        env_vars = {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            with patch("scripts.train.training_progression.load_or_init_results") as mock_load, \
                 patch("scripts.train.training_progression.save_results"):
                mock_load.return_value = {"model_tag": "d27", "checkpoints": []}
                collect_results(sample_cfg)

            # All torchrun vars should be None during eval.
            for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
                assert captured_env[var] is None, (
                    f"{var} should be cleared during eval, got {captured_env[var]}"
                )
        finally:
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_env_vars_restored_for_barrier(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg, tmp_path
    ):
        """After eval, env vars are restored before init_process_group/barrier."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]
        mock_eval.return_value = {"core": {"val": 1}}

        captured_env_at_barrier = {}

        def capture_at_init(*args, **kwargs):
            """Capture env state when init_process_group is called."""
            for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
                captured_env_at_barrier[var] = os.environ.get(var)

        mock_dist.init_process_group = MagicMock(side_effect=capture_at_init)
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        env_vars = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            collect_results(sample_cfg)

            # At the time of init_process_group, env vars should be restored.
            assert captured_env_at_barrier["RANK"] == "1"
            assert captured_env_at_barrier["LOCAL_RANK"] == "1"
            assert captured_env_at_barrier["WORLD_SIZE"] == "2"
            assert captured_env_at_barrier["MASTER_ADDR"] == "localhost"
            assert captured_env_at_barrier["MASTER_PORT"] == "29500"
        finally:
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original

    @patch("scripts.train.training_progression.dist")
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.build_checkpoint_specs")
    @patch("scripts.train.training_progression._needs_eval", return_value=True)
    @patch("scripts.train.training_progression.autodetect_device_type", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_worker_calls_destroy_process_group(
        self, mock_set_device, mock_detect, mock_needs_eval, mock_specs,
        mock_eval, mock_dist, sample_cfg
    ):
        """After barrier and merge, destroy_process_group is called."""
        mock_specs.return_value = [
            CheckpointSpec(step=500, phase="pretrain", subdir=""),
        ]
        mock_eval.return_value = {"core": {"val": 1}}
        mock_dist.init_process_group = MagicMock()
        mock_dist.barrier = MagicMock()
        mock_dist.destroy_process_group = MagicMock()

        env_vars = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        orig_env = {}
        for var, val in env_vars.items():
            orig_env[var] = os.environ.get(var)
            os.environ[var] = val

        try:
            collect_results(sample_cfg)
            mock_dist.destroy_process_group.assert_called_once()
        finally:
            for var, original in orig_env.items():
                if original is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = original


# ---------------------------------------------------------------------------
# Per-checkpoint result files
# ---------------------------------------------------------------------------


class TestCheckpointResultPath:
    """Test _checkpoint_result_path naming convention."""

    def test_path_format(self, sample_cfg):
        path = _checkpoint_result_path(sample_cfg, 1000, "pretrain")
        assert path.name == "pretrain_step_001000.json"

    def test_path_different_phases(self, sample_cfg):
        p1 = _checkpoint_result_path(sample_cfg, 1000, "pretrain")
        p2 = _checkpoint_result_path(sample_cfg, 1000, "sft")
        assert p1 != p2
        assert "pretrain" in p1.name
        assert "sft" in p2.name

    def test_path_different_steps(self, sample_cfg):
        p1 = _checkpoint_result_path(sample_cfg, 500, "pretrain")
        p2 = _checkpoint_result_path(sample_cfg, 1000, "pretrain")
        assert p1 != p2

    def test_path_under_results_dir(self, sample_cfg):
        path = _checkpoint_result_path(sample_cfg, 500, "pretrain")
        assert _results_dir(sample_cfg) in path.parents


class TestSaveLoadCheckpointResult:
    """Test per-checkpoint file save/load cycle."""

    def test_save_and_load(self, sample_cfg):
        evals = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        save_checkpoint_result(sample_cfg, 1000, "pretrain", evals)
        loaded = load_checkpoint_result(sample_cfg, 1000, "pretrain")
        assert loaded is not None
        assert loaded["step"] == 1000
        assert loaded["phase"] == "pretrain"
        assert loaded["evals"] == evals

    def test_load_missing_returns_none(self, sample_cfg):
        assert load_checkpoint_result(sample_cfg, 9999, "pretrain") is None

    def test_save_creates_directory(self, sample_cfg):
        evals = {"core": {"seeds": {}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)
        assert _results_dir(sample_cfg).exists()

    def test_save_overwrites_existing(self, sample_cfg):
        evals_v1 = {"core": {"seeds": {"1234": {"core_metric": 0.3}}}}
        evals_v2 = {"core": {"seeds": {"1234": {"core_metric": 0.7}}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals_v1)
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals_v2)
        loaded = load_checkpoint_result(sample_cfg, 500, "pretrain")
        assert loaded is not None
        assert loaded["evals"]["core"]["seeds"]["1234"]["core_metric"] == 0.7

    def test_atomic_write(self, sample_cfg):
        """No .tmp files left after save."""
        save_checkpoint_result(sample_cfg, 500, "pretrain", {"core": {}})
        tmp_files = list(_results_dir(sample_cfg).glob("*.tmp"))
        assert len(tmp_files) == 0


class TestMergeCheckpointResults:
    """Test merging per-checkpoint files into combined results."""

    def test_merge_multiple_checkpoints(self, sample_cfg):
        save_checkpoint_result(sample_cfg, 500, "pretrain", {"core": {"seeds": {"1234": {}}}})
        save_checkpoint_result(sample_cfg, 1000, "pretrain", {"core": {"seeds": {"1234": {}}}})
        save_checkpoint_result(sample_cfg, 1000, "sft", {"chat": {"seeds": {"1234": {}}}})

        merged = merge_checkpoint_results(sample_cfg)
        assert len(merged["checkpoints"]) == 3
        steps = [(c["step"], c["phase"]) for c in merged["checkpoints"]]
        assert (500, "pretrain") in steps
        assert (1000, "pretrain") in steps
        assert (1000, "sft") in steps

    def test_merge_empty_directory(self, sample_cfg):
        merged = merge_checkpoint_results(sample_cfg)
        assert merged["checkpoints"] == []

    def test_merge_sorted_by_filename(self, sample_cfg):
        """Results should be sorted (filename sort = phase+step order)."""
        save_checkpoint_result(sample_cfg, 2000, "pretrain", {})
        save_checkpoint_result(sample_cfg, 500, "pretrain", {})
        save_checkpoint_result(sample_cfg, 1000, "pretrain", {})

        merged = merge_checkpoint_results(sample_cfg)
        steps = [c["step"] for c in merged["checkpoints"]]
        assert steps == [500, 1000, 2000]

    def test_save_merged_writes_results_json(self, sample_cfg):
        save_checkpoint_result(sample_cfg, 500, "pretrain", {"core": {}})
        save_checkpoint_result(sample_cfg, 1000, "pretrain", {"chat": {}})

        result = save_merged_results(sample_cfg)
        assert len(result["checkpoints"]) == 2

        # Also written to disk.
        path = _results_path(sample_cfg)
        assert path.exists()
        with open(path) as f:
            on_disk = json.load(f)
        assert len(on_disk["checkpoints"]) == 2


class TestNeedsEvalWithCheckpointFiles:
    """Test that _needs_eval works correctly with per-checkpoint result data."""

    def test_no_eval_needed_without_configured_conditioning(self):
        assert _needs_eval(None, ["core"], [1234]) is False

    def test_needs_eval_when_configured_conditioned_file_missing(self):
        assert _needs_eval(
            None,
            ["core"],
            [1234],
            _conditioned_eval_cfg(),
            "pretrain",
        ) is True

    def test_needs_eval_when_conditioned_incomplete(self):
        existing = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(core_seeds=(1234,)),
        }
        # Needs seed 2234.
        assert _needs_eval(
            existing,
            ["core"],
            [1234, 2234],
            _conditioned_eval_cfg(),
            "pretrain",
        ) is True

    def test_no_eval_needed_when_conditioned_complete(self):
        existing = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(core_seeds=(1234, 2234), chat_seeds=(1234,)),
        }
        assert _needs_eval(
            existing,
            ["core", "chat"],
            [1234, 2234],
            _conditioned_eval_cfg(),
            "pretrain",
        ) is False

    def test_needs_eval_missing_conditioned_suite(self):
        existing = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(core_seeds=(1234,)),
        }
        with pytest.raises(ValueError, match="no longer accepts hand-supplied"):
            _needs_eval(existing, ["core", "harmfulness"], [1234],
                _conditioned_eval_cfg(), "pretrain")


class TestResumeAfterInterruption:
    """Integration tests: simulate interruption and verify resume skips
    already-completed checkpoints."""

    def test_resume_skips_completed_checkpoints(self, sample_cfg):
        """Checkpoints with result files should not appear in pending_specs."""
        sample_cfg = _conditioned_eval_cfg(sample_cfg)
        specs = build_checkpoint_specs(sample_cfg)
        eval_suites = ["core", "chat"]
        seeds = [1234, 2234]

        # Simulate completing the first checkpoint.
        evals = _conditioned_evals(core_seeds=(1234, 2234), chat_seeds=(1234,))
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)

        # Filter pending.
        pending = [
            spec for spec in specs
            if _needs_eval(
                load_checkpoint_result(sample_cfg, spec.step, spec.phase),
                eval_suites,
                seeds,
                sample_cfg,
                spec.phase,
            )
        ]

        completed_steps = [(s.step, s.phase) for s in specs if s not in pending]
        assert (500, "pretrain") in completed_steps
        assert len(pending) == len(specs) - 1

    def test_resume_reruns_partial_checkpoint(self, sample_cfg):
        """A checkpoint with only some seeds done should still be pending."""
        sample_cfg = _conditioned_eval_cfg(sample_cfg)
        evals = _conditioned_evals(core_seeds=(1234,))
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)

        result = load_checkpoint_result(sample_cfg, 500, "pretrain")
        assert _needs_eval(
            result,
            ["core", "chat"],
            [1234, 2234],
            sample_cfg,
            "pretrain",
        ) is True

    def test_multiple_ranks_write_different_files(self, sample_cfg):
        """Two ranks writing results for different checkpoints don't conflict."""
        save_checkpoint_result(sample_cfg, 500, "pretrain", {"core": {"seeds": {"1234": {}}}})
        save_checkpoint_result(sample_cfg, 1000, "pretrain", {"core": {"seeds": {"1234": {}}}})

        # Both files exist independently.
        assert load_checkpoint_result(sample_cfg, 500, "pretrain") is not None
        assert load_checkpoint_result(sample_cfg, 1000, "pretrain") is not None

        # Merge picks up both.
        merged = merge_checkpoint_results(sample_cfg)
        assert len(merged["checkpoints"]) == 2
