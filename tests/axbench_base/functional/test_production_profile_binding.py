"""Functional tests for AxBench-Base production profile binding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

import scripts.eval.axbench_base_eval as base_cli
from steering.eval.axbench_base.profiler import get_current_profiler


class _FakeRuntimeState:
    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _minimal_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "feature_dir": str(
                tmp_path
                / "features"
                / "s128"
                / "l8"
                / "a_res"
                / "p_max_pool"
                / "f16k"
                / "m_gemma-2-2b"
                / "base"
            ),
            "feature_statistics_path": str(tmp_path / "stats.safetensors"),
            "decoder_only": {
                "source": "base",
                "model_tag": "d13",
                "step": 10000,
            },
            "encdec": {
                "source": "encdec",
                "model_tag": "d13",
                "step": 10000,
            },
            "device": "cuda:1",
            "output_root": str(tmp_path / "out"),
            "run_tag": "profile-base-test",
            "stage": "0a",
            "profile": {
                "enabled": True,
                "gpu_utilization": False,
                "synchronize_cuda": False,
            },
        }
    )


def test_production_base_profile_honors_explicit_single_process_cuda_device(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_dispatch(cfg_obj, state):  # noqa: ANN001
        seen["profiler_active"] = get_current_profiler() is not None
        seen["run_tag"] = cfg_obj.run_tag
        seen["state_type"] = type(state).__name__

    monkeypatch.setattr(
        base_cli,
        "compute_init",
        lambda *, device_type: (False, 0, 0, 1, "cpu"),  # noqa: ARG005
    )
    monkeypatch.setattr(base_cli, "compute_cleanup", lambda: None)
    monkeypatch.setattr(base_cli, "validate_env_for_stages", lambda *_a, **_k: None)
    monkeypatch.setattr(base_cli, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(base_cli, "_dispatch_stage", fake_dispatch)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    cfg = _minimal_cfg(tmp_path)
    base_cli._run(cfg)

    assert seen == {
        "profiler_active": True,
        "run_tag": "profile-base-test",
        "state_type": "_FakeRuntimeState",
    }
    profile_dir = Path(cfg.output_root) / cfg.run_tag / "profile"
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    metadata = summary["metadata"]
    assert metadata["pipeline"] == "axbench_base"
    assert metadata["profile_gpu_device_index"] == 1
    assert metadata["configured_output_root"] == str(tmp_path / "out")
    assert metadata["effective_output_root"] == str(tmp_path / "out")
    assert metadata["run_output_root"] == str(tmp_path / "out/profile-base-test")
    assert not (tmp_path / "eval/axbench_base/profile-base-test").exists()
