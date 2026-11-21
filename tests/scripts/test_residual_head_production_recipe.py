"""Production recipe and destructive-launch guards for issue #1681."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / "scripts/slurm/conditional_encdec_sft_residual_head.sbatch"
HISTORICAL_ROOT = Path(
    "/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/claude-sft3b-steered-20260905"
)


def test_depth_two_recipe_resolves_explicit_artifacts_and_matched_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = (
        "RUN_DIR", "STEERED_JSONL", "TRAIN_FEATURES", "VAL_FEATURES",
        "TRAIN_PARQUET", "VAL_PARQUET", "MASK", "CLUSTERS",
    )
    for key in keys:
        monkeypatch.setenv(f"RESIDUAL_HEAD_{key}", str(tmp_path / key))
    with initialize_config_dir(config_dir=str(REPO / "configs/training"), version_base=None):
        cfg = compose(config_name="conditional_encdec_sft_3b_residual_head")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["feature_prediction_head_layers"] == 2
    assert resolved["model_step"] == 250000
    assert resolved["base_checkpoint_dir"] == str(tmp_path / "RUN_DIR/base")
    assert resolved["dataset"] == "standard_steered"
    assert resolved["num_iterations"] == -1
    assert resolved["sft_loss_mask"] == "completion_only"
    assert resolved["missing_percentile_policy"] == "lowest"
    assert resolved["total_batch_size"] == 8 * 2048 * 4 * 8
    assert (resolved["device_batch_size"], resolved["max_seq_len"]) == (8, 2048)
    assert (resolved["sae"]["model"], resolved["sae"]["layer"]) == ("gemma-3-4b-pt", 22)
    assert resolved["feature_mask"]["path"] == str(tmp_path / "MASK")
    assert resolved["feature_clusters"]["path"] == str(tmp_path / "CLUSTERS")
    assert resolved["resume_from_step"] == -1


@pytest.mark.parametrize("depth", [4, 6, 8, 10])
def test_transformer_depths_use_ordinary_overrides_and_the_matched_recipe(depth):
    with initialize_config_dir(config_dir=str(REPO / "configs/training"), version_base=None):
        original = OmegaConf.to_container(compose(config_name="conditional_encdec_sft_3b_residual_head"))
        candidate = OmegaConf.to_container(compose(
            config_name="conditional_encdec_sft_3b_transformer_head",
            overrides=[f"feature_prediction_head_layers={depth}"],
        ))
    assert isinstance(original, dict) and isinstance(candidate, dict)
    assert candidate["feature_prediction_block"] == "transformer"
    assert candidate["feature_prediction_head_layers"] == depth
    for field in ("feature_prediction_block", "feature_prediction_head_layers", "run"):
        original.pop(field)
        candidate.pop(field)
    assert candidate == original


@pytest.fixture()
def launcher_environment(tmp_path: Path) -> dict[str, str]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "--quiet", "--allow-empty", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    return {
        **os.environ,
        "RESIDUAL_HEAD_REPO": str(checkout),
        "RESIDUAL_HEAD_RUN_DIR": str(tmp_path / "run"),
        "RESIDUAL_HEAD_EXPECTED_COMMIT": commit,
        "RESIDUAL_HEAD_PYTHON": "/must/not/be/executed",
        "RESIDUAL_HEAD_RESUME_STEP": "-1",
        "SLURM_GPUS_ON_NODE": "0",
        "SLURM_NNODES": "0",
    }


@pytest.mark.parametrize("suffix", ["", "/new-output"])
def test_launcher_refuses_output_under_historical_control(
    launcher_environment: dict[str, str], suffix: str
) -> None:
    env = {**launcher_environment, "RESIDUAL_HEAD_RUN_DIR": str(HISTORICAL_ROOT) + suffix}
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "outside the historical run" in result.stderr


def test_launcher_resolves_symlink_before_protecting_historical_control(
    tmp_path: Path, launcher_environment: dict[str, str]
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(HISTORICAL_ROOT, target_is_directory=True)
    env = {**launcher_environment, "RESIDUAL_HEAD_RUN_DIR": str(alias / "new-output")}
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "outside the historical run" in result.stderr


@pytest.mark.parametrize("link_name", ["base", "leaf"])
def test_launcher_rejects_symlinked_checkpoint_directories_before_resume(
    tmp_path: Path, launcher_environment: dict[str, str], link_name: str
) -> None:
    target = tmp_path / "existing-checkpoints"
    target.mkdir()
    sentinel = target / "untouched"
    sentinel.write_text("existing output")
    base = Path(launcher_environment["RESIDUAL_HEAD_RUN_DIR"]) / "base"
    link = base if link_name == "base" else base / "standard_steered_sft_conditioned_checkpoints"
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)
    env = {
        **launcher_environment, "RESIDUAL_HEAD_RESUME_STEP": "500",
        "SLURM_GPUS_ON_NODE": "4", "SLURM_NNODES": "1",
        "NODE_STAGE_ROOT": str(tmp_path / "stage"),
    }
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "base and SFT output leaf must be real isolated directories" in result.stderr
    assert list(target.iterdir()) == [sentinel]
    assert sentinel.read_text() == "existing output"


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
@pytest.mark.parametrize("checkpoint_file", [
    "base/model_250000.pt", "base/meta_250000.json",
    "base/standard_steered_sft_conditioned_checkpoints/model_000500.pt",
    "base/standard_steered_sft_conditioned_checkpoints/meta_000500.json",
    "base/standard_steered_sft_conditioned_checkpoints/optim_000500_rank2.pt",
])
def test_launcher_refuses_checkpoint_files_shared_with_another_run(
    tmp_path, launcher_environment, checkpoint_file, link_kind,
):
    migrated = tmp_path / "migrated"
    migrated.mkdir()
    run = Path(launcher_environment["RESIDUAL_HEAD_RUN_DIR"])
    base = run / "base"
    base.mkdir(parents=True)
    for name in ("model_250000.pt", "meta_250000.json"):
        (migrated / name).write_text(name)
        (base / name).write_text(name)
    resume = "conditioned_checkpoints" in checkpoint_file
    if resume:
        leaf = base / "standard_steered_sft_conditioned_checkpoints"
        leaf.mkdir()
        for name in ["model_000500.pt", "meta_000500.json", *[f"optim_000500_rank{rank}.pt" for rank in range(4)]]:
            (leaf / name).write_text(name)
    destination = run / checkpoint_file
    original = tmp_path / "historical-checkpoint"
    original.write_bytes(destination.read_bytes())
    destination.unlink()
    if link_kind == "symlink":
        destination.symlink_to(original)
    else:
        destination.hardlink_to(original)
    before = original.read_bytes(), original.stat().st_mtime_ns
    env = {
        **launcher_environment, "RESIDUAL_HEAD_RESUME_STEP": "500" if resume else "-1",
        "RESIDUAL_HEAD_MIGRATED_BASE": str(migrated),
        "SLURM_GPUS_ON_NODE": "4", "SLURM_NNODES": "1",
        "NODE_STAGE_ROOT": str(tmp_path / "stage"),
    }
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0 and "isolated" in result.stderr
    assert before == (original.read_bytes(), original.stat().st_mtime_ns)


@pytest.mark.parametrize("resume", ["0", "-2", "1.5", "true", "0500", "500;false"])
def test_launcher_rejects_ambiguous_resume_before_creating_outputs(
    launcher_environment: dict[str, str], resume: str
) -> None:
    env = {**launcher_environment, "RESIDUAL_HEAD_RESUME_STEP": resume}
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "resume step must be" in result.stderr
    assert not Path(env["RESIDUAL_HEAD_RUN_DIR"]).exists()


def test_launcher_rejects_unreviewed_code_before_creating_outputs(
    launcher_environment: dict[str, str]
) -> None:
    env = {**launcher_environment, "RESIDUAL_HEAD_EXPECTED_COMMIT": "0" * 40}
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "commit differs" in result.stderr
    assert not Path(env["RESIDUAL_HEAD_RUN_DIR"]).exists()


def test_launcher_rejects_dirty_code_before_creating_outputs(
    launcher_environment: dict[str, str]
) -> None:
    (Path(launcher_environment["RESIDUAL_HEAD_REPO"]) / "unreviewed.txt").write_text("changed")
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=launcher_environment, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert "checkout must be clean" in result.stderr
    assert not Path(launcher_environment["RESIDUAL_HEAD_RUN_DIR"]).exists()


@pytest.mark.parametrize("argument", ["preflight", "train"])
def test_launcher_rejects_removed_modes_before_creating_outputs(
    launcher_environment: dict[str, str], argument: str
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER), argument], env=launcher_environment,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "no positional arguments" in result.stderr
    assert not Path(launcher_environment["RESIDUAL_HEAD_RUN_DIR"]).exists()


@pytest.mark.parametrize("inherited_dtype", [None, "float16", "float32"])
def test_launcher_exports_bfloat16_and_propagates_pipeline_failure(
    tmp_path: Path,
    launcher_environment: dict[str, str],
    inherited_dtype: str | None,
) -> None:
    # The real launcher's allocation guard exits before touching source inputs.
    # Probe its exported dtype and pipeline behavior in that shell's EXIT trap.
    probe = tmp_path / "shell-probe"
    probe.write_text(
        "trap 'printenv NANOCHAT_DTYPE; false | cat; echo pipeline-failure-ignored' EXIT\n"
    )
    env = {**launcher_environment, "BASH_ENV": str(probe)}
    if inherited_dtype is None:
        env.pop("NANOCHAT_DTYPE", None)
    else:
        env["NANOCHAT_DTYPE"] = inherited_dtype
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "requires one node and four GPUs" in result.stderr
    assert result.stdout == "bfloat16\n"
