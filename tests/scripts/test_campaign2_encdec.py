"""CPU artifact and executable launcher contracts for the chained SFT campaign."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.utils.campaign2 import Stage, inventory, latest_complete, prepare_base

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def submission_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "sbatch.jsonl"
    sbatch = binary / "sbatch"
    sbatch.write_text(f"""#!{sys.executable}
import json, os, sys
path = os.environ["SBATCH_LOG"]
with open(path, "a+", encoding="utf-8") as stream:
    stream.seek(0)
    job_id = 100 + len(stream.readlines())
    stream.write(json.dumps({{"args": sys.argv[1:], "mixture": os.environ.get("MIXTURE"),
                             "task": os.environ.get("CAMPAIGN_CPU_TASK")}}) + "\\n")
print(job_id)
""")
    sbatch.chmod(0o755)
    git = binary / "git"
    git.write_text('#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo pinned; fi\n')
    git.chmod(0o755)
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "SBATCH_LOG": str(log),
        "CAMPAIGN_REPO": str(ROOT),
        "CAMPAIGN_ROOT": str(campaign_root),
        "CAMPAIGN_PYTHON": sys.executable,
        "NANOCHAT_BASE_DIR": str(tmp_path),
        "HF_HOME": str(tmp_path),
        "EXPECTED_GIT_REVISION": "pinned",
    }
    return env, log


def _submission_calls(
    submission_env: tuple[dict[str, str], Path], script: str, pool: str,
) -> list[dict]:
    env, log = submission_env
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/campaign2" / script), pool],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.mark.parametrize("pool", ["priority", "preemptable"])
def test_data_submission_preserves_cpu_resources_and_gpu_lane(
    submission_env: tuple[dict[str, str], Path], pool: str,
) -> None:
    submission_env[0]["MIXTURE"] = "safety"
    convert, extract, stats = _submission_calls(submission_env, "submit_data.sh", pool)
    assert convert["task"] is None and convert["args"][-1].endswith("/cpu.sbatch")
    assert {"--partition=mit_normal", "--account=mit_amf_advanced_cpu", "--qos=mit_amf_advanced_cpu",
            "--job-name=c2-convert", "--cpus-per-task=16", "--mem=128G", "--time=08:00:00",
            "--export=ALL,CAMPAIGN_CPU_TASK=convert"} <= set(convert["args"])
    assert extract["task"] is None and extract["args"][-1].endswith("/extract.sbatch")
    expected_pool = "mit_normal_gpu" if pool == "priority" else "mit_preemptable"
    assert expected_pool in extract["args"] and "--array=0-3%4" in extract["args"]
    assert stats["task"] is None and stats["args"][-1].endswith("/cpu.sbatch")
    assert {"--job-name=c2-stats", "--cpus-per-task=16", "--mem=256G", "--time=12:00:00",
            "--export=ALL,CAMPAIGN_CPU_TASK=stats"} <= set(stats["args"])


@pytest.mark.parametrize("pool", ["priority", "preemptable"])
def test_pythia_submission_preserves_manifest_resources_and_gpu_lane(
    submission_env: tuple[dict[str, str], Path], pool: str,
) -> None:
    calls = _submission_calls(submission_env, "submit_pythia.sh", pool)
    manifests = calls[:2]
    assert [(call["mixture"], call["task"]) for call in manifests] == [
        ("safety", None), ("capabilities", None)]
    for call in manifests:
        assert call["args"][-1].endswith("/cpu.sbatch")
        assert {"--partition=mit_normal", "--account=mit_amf_advanced_cpu", "--qos=mit_amf_advanced_cpu",
                "--job-name=c2-hf-manifest", "--cpus-per-task=16", "--mem=128G",
                "--time=24:00:00", "--export=ALL,CAMPAIGN_CPU_TASK=manifest"} <= set(call["args"])
    expected_pool = "mit_normal_gpu" if pool == "priority" else "mit_preemptable"
    assert all(call["task"] is None and expected_pool in call["args"]
               and call["args"][-1].endswith("/hf_stage.sbatch") for call in calls[2:])


def test_documented_direct_campaign_launchers_execute(tmp_path: Path, submission_env: tuple[dict[str, str], Path]) -> None:
    launch = ROOT / "scripts/slurm/campaign2"
    log = tmp_path / "direct.log"
    runner = tmp_path / "python"
    runner.write_text('#!/bin/bash\nprintf "%s|%s\\n" "${CUDA_VISIBLE_DEVICES-unset}" "$*" >> "$DIRECT_LOG"\n')
    runner.chmod(0o755)
    env = {**submission_env[0], "CAMPAIGN_PYTHON": str(runner), "DIRECT_LOG": str(log)}
    assert "#SBATCH" not in (launch / "cpu.sbatch").read_text()
    cpu_env = {**env, "CAMPAIGN_CPU_TASK": "eval-view", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        ["bash", str(launch / "cpu.sbatch"), "sentinel=1"],
        cwd=ROOT, env=cpu_env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        ["bash", str(launch / "eval.sbatch"), "mode=judge", "variant=pythia_prompt"],
        cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == [
        "|-m scripts.utils.campaign2_eval_view sentinel=1",
        "|-m scripts.eval.campaign2 mode=judge variant=pythia_prompt",
    ]
    docs = (ROOT / "docs/pipelines/campaign2.md").read_text()
    assert "`cpu.sbatch` directly" in docs and "`eval.sbatch` directly" in docs
    assert "`variant=pythia_prompt`" in docs


def checkpoint(path: Path, step: int = 7, **changes: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"model_{step:06d}.pt").write_bytes(b"model")
    meta = {
        "step": step, "dataset": "standard_steered", "sft_loss_mask": "completion_only",
        "sft_segment_mode": "turn_delimited_fixed_tokens", "user_config": {"segment_size": 32},
        **changes,
    }
    (path / f"meta_{step:06d}.json").write_text(json.dumps(meta))


def test_stage_parent_is_pinned_and_leaf_is_isolated(tmp_path: Path) -> None:
    previous, base = tmp_path / "instruct", tmp_path / "safety/base"
    checkpoint(previous)
    leaf = prepare_base(previous, 7, base, Stage.SAFETY)
    assert leaf == base / "safety_sft_conditioned_checkpoints"
    assert (base / "model_000007.pt").is_symlink()
    assert (base / "model_000007.pt").resolve() == previous / "model_000007.pt"
    assert prepare_base(previous, 7, base, Stage.SAFETY) == leaf
    checkpoint(leaf, dataset="safety")
    cap = prepare_base(leaf, 7, tmp_path / "capabilities/base", Stage.CAPABILITIES)
    assert cap.name == "capabilities_sft_conditioned_checkpoints"
    checkpoint(previous, step=8)
    with pytest.raises(ValueError, match="unrelated artifacts"):
        prepare_base(previous, 8, base, Stage.SAFETY)


@pytest.mark.parametrize("changes,message", [
    ({"sft_loss_mask": "full_sequence"}, "completion_only"),
    ({"dataset": "safety"}, "standard_steered"),
    ({"selfcond": {}}, "not RL"),
    ({"grpo": {}}, "not RL"),
    ({"sft_segment_mode": "fixed_tokens"}, "turn-delimited"),
    ({"user_config": {"segment_size": 128}}, "segment_size=32"),
])
def test_wrong_parent_fails_before_symlinking(tmp_path: Path, changes: dict, message: str) -> None:
    checkpoint(tmp_path / "source", **changes)
    with pytest.raises(ValueError, match=message):
        prepare_base(tmp_path / "source", 7, tmp_path / "base", Stage.SAFETY)
    assert not (tmp_path / "base").exists()


def test_resume_rejects_partial_and_counts_exact_ranks(tmp_path: Path) -> None:
    assert latest_complete(tmp_path) == -1
    checkpoint(tmp_path)
    for rank in range(3):
        (tmp_path / f"optim_000007_rank{rank}.pt").write_bytes(b"optim")
    (tmp_path / "optim_000007_rank3.pt.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="no complete"):
        latest_complete(tmp_path)
    (tmp_path / "optim_000007_rank3.pt").write_bytes(b"optim")
    checkpoint(tmp_path, 8)
    assert latest_complete(tmp_path) == 7
    (tmp_path / "optim_000007_rank4.pt").write_bytes(b"wrong topology")
    with pytest.raises(ValueError, match="no complete"):
        latest_complete(tmp_path)


def test_inventory_requires_exact_complete_sets(tmp_path: Path) -> None:
    parquet, features = tmp_path / "parquet", tmp_path / "features"
    parquet.mkdir()
    features.mkdir()
    meta = parquet / "conversion_meta.json"
    meta.write_text('{"total_shards":2,"completed_shards":1}')
    with pytest.raises(ValueError, match="conversion must be complete"):
        inventory(parquet)
    meta.write_text('{"total_shards":2,"completed_shards":2}')
    for index in range(2):
        (parquet / f"shard_{index:05d}.parquet").write_bytes(b"parquet")
        (features / f"shard_{index:05d}_features.safetensors.zst").write_bytes(b"features")
    assert inventory(parquet, features) == 2
    (features / "shard_00002_features.safetensors.zst").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="feature shard set"):
        inventory(parquet, features)


@pytest.mark.parametrize("mixture", ["safety", "capabilities"])
def test_encdec_and_extraction_compositions(mixture: str) -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs/training"), version_base=None):
        cfg = compose(config_name="conditional_encdec_sft", overrides=[
            f"dataset={mixture}", "sft_loss_mask=completion_only", "segment_size=32",
            "base_checkpoint_dir=/stages/base", "model_step=7", "model_tag=campaign2",
        ])
    assert cfg.dataset == mixture and cfg.steered_jsonl is None
    assert cfg.base_checkpoint_dir == "/stages/base" and cfg.save_every == 500
    with initialize_config_dir(config_dir=str(ROOT / "configs/extraction"), version_base=None):
        extraction = compose(config_name="extract_features_gemma3_4bpt_l22_f65k_ci_nobos", overrides=[
            f"dataset_phase={mixture}_sft", "segment_mode=turn_delimited_fixed_tokens", "segment_size=32",
        ])
    assert extraction.model == "gemma-3-4b-pt"
    assert extraction.model_dtype == "bfloat16" and extraction.context_independent_features
    assert extraction.sae.num_features == "65k"


@pytest.mark.parametrize("local", [False, True])
def test_extraction_worker_shards_and_batches(tmp_path: Path, local: bool) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    fake_git = binary / "git"
    fake_git.write_text('#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo test; fi\n')
    fake_git.chmod(0o755)
    fake_python = binary / "python"
    fake_python.write_text(
        '#!/bin/bash\nif [[ $2 == scripts.utils.campaign2 ]]; then echo 6; '
        'else printf "%s\\n" "$*" >> "$CALLS"; fi\n'
    )
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "CALLS": str(calls),
           "CAMPAIGN_REPO": str(ROOT), "CAMPAIGN_ROOT": str(tmp_path / "campaign"),
           "NANOCHAT_BASE_DIR": str(tmp_path), "HF_HOME": str(tmp_path), "EXPECTED_GIT_REVISION": "test",
           "MIXTURE": "safety", "CAMPAIGN_PYTHON": str(fake_python), "SLURM_ARRAY_TASK_ID": "2"}
    script = "extract_local.sh" if local else "extract.sbatch"
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/campaign2" / script)],
                            env=env, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == (12 if local else 2)
    assert all("gpu_batch_size=2 " in line if local else "gpu_batch_size=248 " in line for line in lines)
    if not local:
        assert all("start_shard=2 num_shards=3" in line for line in lines)
    assert any("dataset_phase=safety_sft_val" in line for line in lines)
