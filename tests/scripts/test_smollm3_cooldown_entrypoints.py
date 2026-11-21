"""Functional wiring tests for SmolLM3 cooldown entrypoints and launcher."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from scripts.preprocess.prepare_smollm3_cooldown import load_config as load_prepare_config
from scripts.train.smollm3_cooldown import load_config as load_train_config
from steering.smollm3_cooldown import SOURCE_MODEL_ID, SOURCE_REVISION

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_both_entrypoints_compose_the_same_exact_config() -> None:
    overrides = ["data.prepared_dir=/pool/prepared", "output_dir=/pool/output", "canary_steps=2"]
    prepare = cast(
        dict[str, Any], OmegaConf.to_container(load_prepare_config(overrides), resolve=True)
    )
    train = cast(dict[str, Any], OmegaConf.to_container(load_train_config(overrides), resolve=True))
    assert prepare == train
    assert train["source"] == {
        "model_id": SOURCE_MODEL_ID,
        "revision": SOURCE_REVISION,
        "local_files_only": True,
    }
    assert train["data"]["prepared_dir"] == "/pool/prepared"
    assert train["output_dir"] == "/pool/output"
    assert train["canary_steps"] == 2


def test_config_requires_explicit_prepared_and_output_paths() -> None:
    config = load_train_config([])
    with pytest.raises(Exception, match="prepared_dir"):
        str(config.data.prepared_dir)
    with pytest.raises(Exception, match="output_dir"):
        str(config.output_dir)


def test_config_name_switch_is_supported_and_not_forwarded() -> None:
    config = load_train_config(
        [
            "--",
            "config_name=smollm3_cooldown",
            "data.prepared_dir=/a",
            "output_dir=/b",
        ]
    )
    assert config.data.prepared_dir == "/a"
    assert config.output_dir == "/b"
    assert "config_name" not in config


def test_entrypoints_are_import_safe_and_main_is_guarded() -> None:
    import scripts.preprocess.prepare_smollm3_cooldown as prepare_module
    import scripts.train.smollm3_cooldown as train_module

    assert callable(prepare_module.main)
    assert callable(train_module.main)
    assert 'if __name__ == "__main__":' in inspect.getsource(prepare_module)
    assert 'if __name__ == "__main__":' in inspect.getsource(train_module)


def test_distributed_acceptance_topology_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.steering.smollm3_cooldown_ddp_real import _expected_topology

    monkeypatch.delenv("SMOLLM3_COOLDOWN_DDP_WORLD_SIZE", raising=False)
    monkeypatch.delenv("SMOLLM3_COOLDOWN_DDP_LOCAL_WORLD_SIZE", raising=False)
    assert _expected_topology() == (2, 2)
    monkeypatch.setenv("SMOLLM3_COOLDOWN_DDP_WORLD_SIZE", "32")
    monkeypatch.setenv("SMOLLM3_COOLDOWN_DDP_LOCAL_WORLD_SIZE", "8")
    assert _expected_topology() == (32, 8)


def test_distributed_acceptance_rejects_invalid_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.steering.smollm3_cooldown_ddp_real import _expected_topology

    monkeypatch.setenv("SMOLLM3_COOLDOWN_DDP_WORLD_SIZE", "31")
    monkeypatch.setenv("SMOLLM3_COOLDOWN_DDP_LOCAL_WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="positive multiple"):
        _expected_topology()


def test_launcher_has_valid_bash_and_exact_distributed_topology() -> None:
    launcher = REPO_ROOT / "scripts" / "slurm" / "smollm3_cooldown.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "#SBATCH --partition=mit_normal_gpu" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --gpus-per-node=h200:1" in text
    assert "--ntasks-per-node=1" in text
    assert '--nproc_per_node="${GPUS_PER_NODE}"' in text
    assert '--node_rank="${SLURM_PROCID}"' in text
    assert "--max_restarts=0" in text
    assert "scripts/slurm/preflight_nccl_transport.sh" in text
    assert "bash -c" in text
    assert "bash -lc" not in text


def test_launcher_is_offline_has_no_install_and_requires_shared_paths() -> None:
    text = (REPO_ROOT / "scripts" / "slurm" / "smollm3_cooldown.sbatch").read_text()
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert ".venv/bin/torchrun" in text
    assert "uv sync" not in text
    assert "pip install" not in text
    assert "PREPARED_DATA_DIR:-" in text
    assert "OUTPUT_DIR:-" in text
    assert '"data.prepared_dir=${PREPARED_DATA_DIR}"' in text
    assert '"output_dir=${OUTPUT_DIR}"' in text


def test_launcher_does_not_hardcode_account_or_qos_identity() -> None:
    lines = (REPO_ROOT / "scripts" / "slurm" / "smollm3_cooldown.sbatch").read_text().splitlines()
    directives = [line for line in lines if line.startswith("#SBATCH")]
    assert not any("--account" in line for line in directives)
    assert not any("--qos" in line for line in directives)
    assert any('--account="$ACCOUNT" --qos="$QOS"' in line for line in lines)
