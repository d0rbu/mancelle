"""Regression tests for issue #1269 token-boundary worker thread caps."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch as t
from omegaconf import OmegaConf

from scripts.preprocess import precompute_token_boundaries as token_boundaries
from steering.overlay_io import load_overlay_shard
from tests.scripts._token_boundary_fixtures import make_token_boundary_env


def _overlay_payloads(feature_dir: Path) -> dict[str, dict[str, t.Tensor]]:
    payloads: dict[str, dict[str, t.Tensor]] = {}
    for path in sorted((feature_dir / "tok").rglob("*.safetensors.zst")):
        payloads[path.name] = load_overlay_shard(path)
    return payloads


def test_worker_initializer_caps_torch_threads(tmp_path: Path) -> None:
    script = """
from scripts.preprocess.precompute_token_boundaries import _init_worker_process
import torch
_init_worker_process(object(), 5, 128)
print(torch.get_num_threads(), torch.get_num_interop_threads())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "1 1"


def test_apply_worker_thread_env(monkeypatch) -> None:
    for variable in token_boundaries._WORKER_THREAD_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    saved = token_boundaries._apply_worker_thread_env()
    for variable, value in token_boundaries._WORKER_THREAD_ENV_VARS.items():
        assert os.environ[variable] == value
    assert saved == dict.fromkeys(token_boundaries._WORKER_THREAD_ENV_VARS)


def test_apply_and_restore_round_trips_prior_values(monkeypatch) -> None:
    """Snapshot + restore must return the parent env to its exact prior state.

    Review finding (#1269): the original unconditional os.environ.update leaked
    the '1' caps into the parent forever, silently degrading a later workers=1
    variant in the same process (exactly the sequential baseline the benchmark
    sweep compares against).
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    saved = token_boundaries._apply_worker_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    token_boundaries._restore_worker_thread_env(saved)
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in os.environ
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"


def test_run_wires_pool_initializer(monkeypatch, tmp_path: Path) -> None:
    env_parallel = make_token_boundary_env(tmp_path / "parallel", n_shards=1)
    env_sequential = make_token_boundary_env(tmp_path / "sequential", n_shards=1)
    captured: dict[str, object] = {}
    real_executor = ProcessPoolExecutor

    class RecordingExecutor:
        def __init__(self, *args, **kwargs):
            captured["initializer"] = kwargs["initializer"]
            captured["env"] = {
                variable: os.environ.get(variable)
                for variable in token_boundaries._WORKER_THREAD_ENV_VARS
            }
            self._executor = real_executor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._executor, name)

    monkeypatch.setattr(token_boundaries, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(token_boundaries, "get_tokenizer", lambda: env_parallel["tokenizer"])
    # Pre-set values so the post-run restoration assertion has teeth.
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    cfg_parallel = {
        "feature_dir": env_parallel["feature_dir"],
        "data_dir": env_parallel["data_dir"],
        "segment_size": env_parallel["segment_size"],
        "workers": 2,
        "force": False,
        "resume": False,
        "dry_run": False,
        "profile": True,
        "profile_summary_output": str(tmp_path / "parallel-profile.json"),
    }
    assert token_boundaries.run(OmegaConf.create(cfg_parallel)) == 0
    # The caps must span only the pool's lifetime: children saw "1" (asserted
    # below via captured["env"]), and the parent env is restored on return.
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in os.environ
    assert "OPENBLAS_NUM_THREADS" not in os.environ

    monkeypatch.setattr(token_boundaries, "get_tokenizer", lambda: env_sequential["tokenizer"])
    cfg_sequential = {
        "feature_dir": env_sequential["feature_dir"],
        "data_dir": env_sequential["data_dir"],
        "segment_size": env_sequential["segment_size"],
        "workers": 1,
        "force": False,
        "resume": False,
        "dry_run": False,
    }
    assert token_boundaries.run(OmegaConf.create(cfg_sequential)) == 0

    assert captured["initializer"] is token_boundaries._init_worker_process
    assert captured["env"] == token_boundaries._WORKER_THREAD_ENV_VARS
    parallel_payloads = _overlay_payloads(Path(env_parallel["feature_dir"]))
    sequential_payloads = _overlay_payloads(Path(env_sequential["feature_dir"]))
    assert parallel_payloads.keys() == sequential_payloads.keys()
    for stem in parallel_payloads:
        assert parallel_payloads[stem].keys() == sequential_payloads[stem].keys()
        for key in parallel_payloads[stem]:
            assert t.equal(parallel_payloads[stem][key], sequential_payloads[stem][key])
