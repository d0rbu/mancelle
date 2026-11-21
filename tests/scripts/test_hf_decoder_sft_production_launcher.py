"""Execute both launchers with fake Slurm/torchrun; no GPUs or downloads."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["1x4", "4x8"])
def staged_launcher(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, dict[str, str]]:
    topology = request.param
    repo = tmp_path / "repo"
    scripts = repo / "scripts" / "slurm"
    scripts.mkdir(parents=True)
    for name in (f"hf_decoder_sft_h200_{topology}.sbatch", "hf_decoder_sft_preflight.sh"):
        shutil.copy(REPO_ROOT / "scripts" / "slurm" / name, scripts / name)
    binaries = repo / "bin"
    binaries.mkdir()
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)

    def executable(path: Path, body: str) -> None:
        path.write_text(body)
        path.chmod(0o755)

    executable(
        venv / "python",
        f"""#!{sys.executable}
import json
import os
import subprocess
import sys
from pathlib import Path
if sys.argv[1] == "-m":
    Path(os.environ["VERIFY_ARGS"]).write_text(json.dumps(sys.argv[2:]))
    sys.exit(int(os.environ.get("VERIFY_FAILURE", "0")))
if "manifest must be" in sys.argv[2] or "num_iterations" in sys.argv[2] or "campaign2_parent" in sys.argv[2]:
    sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))
print('{{}}')
""",
    )
    executable(venv / "torchrun", '#!/bin/bash\nprintf "%s\\n" "$@" > "${TRAIN_ARGS}"\n')
    executable(
        binaries / "srun", '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n'
    )
    executable(
        binaries / "scontrol",
        '#!/bin/bash\nif [[ "$2" == hostnames ]]; then\nfor ((i=0; i<SLURM_NNODES; i++)); do echo "node${i}"; done\nelse echo job; fi\n',
    )
    executable(binaries / "getent", '#!/bin/bash\necho "127.0.0.1 STREAM node0"\n')
    executable(
        scripts / "preflight_nccl_transport.sh",
        '#!/bin/bash\nmkdir -p "$PREFLIGHT_RESULT_DIR"\necho "unset NCCL_NET_GDR_LEVEL" > "$PREFLIGHT_RESULT_DIR/transport.env"\n',
    )
    (repo / "uv.lock").write_text("test lock\n")
    (repo / ".gitignore").write_text(".venv/\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "identity_conversations.jsonl").write_text("identity\n")
    steered = tmp_path / "steered.jsonl"
    steered.write_text("steered\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"snapshot":"fixture","epoch":{"num_iterations":1123}}\n')
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "HF_", "PROFILE_", "RESTART_"))
    }
    env.update(
        {
            "PATH": f"{binaries}:{env['PATH']}",
            "SLURM_SUBMIT_DIR": str(repo),
            "SLURM_NNODES": "1" if topology == "1x4" else "4",
            "SLURM_GPUS_ON_NODE": "4" if topology == "1x4" else "8",
            "SLURM_JOB_ID": "1234",
            "SLURM_JOB_NODELIST": "node0",
            "SLURM_PROCID": "0",
            "EXPECTED_GIT_REVISION": revision,
            "EXPECTED_UV_LOCK_SHA256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
            "DATA_MANIFEST": str(manifest),
            "DATA_MANIFEST_SHA256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "NANOCHAT_BASE_DIR": str(cache),
            "HF_HOME": str(cache),
            "STEERED_JSONL": str(steered),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "TRAIN_ARGS": str(tmp_path / "train_args"),
            "VERIFY_ARGS": str(tmp_path / "verify_args"),
            **({"NUM_ITERATIONS": "1123"} if topology == "1x4" else {}),
        }
    )
    return scripts / f"hf_decoder_sft_h200_{topology}.sbatch", env


def run_launcher(launcher: Path, env: dict[str, str]) -> tuple[int, str]:
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=launcher.parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = Path(env["OUTPUT_DIR"]) / "logs" / f"slurm_{env['SLURM_JOB_ID']}.err"
    return result.returncode, result.stderr + (log.read_text() if log.exists() else "")


def test_successful_launch_preserves_defaults_and_evidence(
    staged_launcher: tuple[Path, dict[str, str]],
) -> None:
    launcher, env = staged_launcher
    assert run_launcher(launcher, env) == (0, "")
    args = Path(env["TRAIN_ARGS"]).read_text().splitlines()
    output = Path(env["OUTPUT_DIR"])
    assert (output / "data_manifest.json").read_bytes() == Path(env["DATA_MANIFEST"]).read_bytes()
    assert (output / "invocations/job_1234/run_contract.env").is_file()
    assert "--max_restarts=0" in args
    if "1x4" in launcher.name:
        assert "num_iterations=1123" in args
        assert "--nproc_per_node=4" in args
        assert "max_seq_len=2048" in args
        assert "dry_run=false" in args
        assert "huggingface.restart_save_every=500" in args
        assert "huggingface.validate_export=true" in args
        assert "sft_loss_mask=full_sequence" in args
        assert "eval_every=0" not in args and "save_every=0" not in args
        assert (output / "logs/slurm_1234.out").exists()
        verify = json.loads(Path(env["VERIFY_ARGS"]).read_text())
        assert "mode=verify" in verify
        assert "sft_config_name=chat_sft_pythia_2_8b" in verify
        assert "config_name=chat_sft_pythia_2_8b_standard_steered" in args
    else:
        assert "num_iterations=60" in args
        assert "--nproc_per_node=8" in args
        assert "dry_run=true" in args
        assert "eval_every=0" in args and "save_every=0" in args


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("CONFIG_NAME", "chat_sft_pythia_2_8b", "unsupported CONFIG_NAME=chat_sft_pythia_2_8b"),
        ("CONFIG_NAME", "chat_sft_pythia_1_4b_h200_4x8_benchmark_typo", "unsupported CONFIG_NAME"),
        ("EXPECTED_GIT_REVISION", "wrong", "git revision mismatch"),
        ("EXPECTED_UV_LOCK_SHA256", "wrong", "uv.lock digest mismatch"),
        ("DATA_MANIFEST_SHA256", "wrong", "data-manifest digest mismatch"),
        ("DATA_MANIFEST_SHA256", "", "must be a nonempty"),
        ("SLURM_NNODES", "2", "allocated nodes"),
        ("DEVICE_BATCH_SIZE", "3", "must be divisible"),
        ("DEVICE_BATCH_SIZE", "0", "positive integer"),
        ("RESTART_FROM_STEP", "-2", "must be -1 or non-negative"),
        ("RESTART_FROM_STEP", "500", "existing OUTPUT_DIR"),
    ],
)
def test_preflight_failures_do_not_launch(
    staged_launcher: tuple[Path, dict[str, str]], key: str, value: str, message: str
) -> None:
    launcher, env = staged_launcher
    env[key] = value
    code, stderr = run_launcher(launcher, env)
    assert code != 0
    # Production creates only its logs before restart validation.
    if key == "RESTART_FROM_STEP" and value == "500" and "1x4" in launcher.name:
        message = "restart complete marker"
    assert message in stderr
    assert not Path(env["TRAIN_ARGS"]).exists()


@pytest.mark.parametrize("problem", ["dirty", "empty_manifest", "missing_identity", "used_output"])
def test_bad_staging_is_rejected(
    staged_launcher: tuple[Path, dict[str, str]], problem: str
) -> None:
    launcher, env = staged_launcher
    if problem == "dirty":
        (launcher.parents[2] / "untracked.txt").write_text("dirty")
    elif problem == "empty_manifest":
        Path(env["DATA_MANIFEST"]).write_text("{}")
    elif problem == "missing_identity":
        (Path(env["NANOCHAT_BASE_DIR"]) / "identity_conversations.jsonl").unlink()
    else:
        output = Path(env["OUTPUT_DIR"])
        output.mkdir()
        (output / "step_000500").mkdir()
    code, _stderr = run_launcher(launcher, env)
    assert code != 0
    assert not Path(env["TRAIN_ARGS"]).exists()


def test_restart_requires_retained_manifest_and_new_invocation(
    staged_launcher: tuple[Path, dict[str, str]],
) -> None:
    launcher, env = staged_launcher
    assert run_launcher(launcher, env)[0] == 0
    output = Path(env["OUTPUT_DIR"])
    (output / "restart").mkdir()
    (output / "restart/complete_000500.json").write_text('{"step":500}')
    env.update(RESTART_FROM_STEP="500", PROFILE_TIMING="false", SLURM_JOB_ID="1235")
    assert run_launcher(launcher, env)[0] == 0
    assert "huggingface.restart_from_step=500" in Path(env["TRAIN_ARGS"]).read_text()
    assert run_launcher(launcher, env)[0] != 0  # invocation evidence cannot be overwritten
    env["SLURM_JOB_ID"] = "1236"
    (output / "data_manifest.json").write_text('{"snapshot":"different"}')
    assert "differs from the retained" in run_launcher(launcher, env)[1]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("SLURM_GPUS_ON_NODE", "8", "expected 4 GPUs"),
        ("SLURM_GPUS_ON_NODE", "gpu:h200:4", "positive integer"),
        ("VERIFY_FAILURE", "1", "live SFT data differs"),
        ("NUM_ITERATIONS", "999", "NUM_ITERATIONS differs"),
        ("STEERED_JSONL", "/missing", "STEERED_JSONL"),
        ("SFT_LOSS_MASK", "typo", "unsupported SFT_LOSS_MASK"),
    ],
)
def test_production_specific_guards(
    staged_launcher: tuple[Path, dict[str, str]], key: str, value: str, message: str
) -> None:
    launcher, env = staged_launcher
    if "4x8" in launcher.name:
        pytest.skip("production-only contract")
    env[key] = value
    code, stderr = run_launcher(launcher, env)
    assert code != 0 and message in stderr
    assert not Path(env["TRAIN_ARGS"]).exists()


@pytest.mark.parametrize("size", ["1_4b", "2_8b"])
def test_production_preset_selects_the_manifest_tokenizer_source(
    staged_launcher: tuple[Path, dict[str, str]], size: str
) -> None:
    launcher, env = staged_launcher
    if "4x8" in launcher.name:
        pytest.skip("production-only contract")
    env["CONFIG_NAME"] = f"chat_sft_pythia_{size}_standard_steered"
    assert run_launcher(launcher, env) == (0, "")
    verify = json.loads(Path(env["VERIFY_ARGS"]).read_text())
    assert f"sft_config_name=chat_sft_pythia_{size}" in verify
    args = Path(env["TRAIN_ARGS"]).read_text().splitlines()
    assert f"config_name=chat_sft_pythia_{size}_standard_steered" in args
    contract = (Path(env["OUTPUT_DIR"]) / "invocations/job_1234/run_contract.env").read_text()
    assert f"manifest_sft_config_name=chat_sft_pythia_{size}\n" in contract


def test_single_node_is_nonexclusive_and_uses_runtime_gpu_count() -> None:
    text = (REPO_ROOT / "scripts/slurm/hf_decoder_sft_h200_1x4.sbatch").read_text()
    assert "#SBATCH --gres=gpu:h200:4" in text
    assert "#SBATCH --time=24:00:00" in text
    for option in ("exclusive", "account", "qos", "partition"):
        assert f"#SBATCH --{option}" not in text


@pytest.mark.parametrize("stage", ["safety", "capabilities"])
def test_campaign2_uses_local_parent_and_stage_verifier(
    staged_launcher: tuple[Path, dict[str, str]], stage: str,
) -> None:
    launcher, env = staged_launcher
    if "4x8" in launcher.name:
        pytest.skip("Campaign 2 is one node")
    path = Path(env["DATA_MANIFEST"])
    payload = json.loads(path.read_text())
    payload["campaign2_parent"] = {"source_revision": "a" * 40}
    path.write_text(json.dumps(payload))
    env.update(CONFIG_NAME=f"chat_sft_pythia_2_8b_{stage}", SFT_LOSS_MASK="completion_only",
               HF_PARENT_EXPORT="/parent/step_000007", HF_PARENT_REVISION="a" * 40,
               C2_DATA_MANIFEST="/manifest/data.json",
               DATA_MANIFEST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    env.pop("STEERED_JSONL")
    assert run_launcher(launcher, env) == (0, "")
    args = Path(env["TRAIN_ARGS"]).read_text().splitlines()
    assert f"config_name=chat_sft_pythia_2_8b_{stage}" in args
    assert "steered_jsonl=null" in args and "sft_loss_mask=completion_only" in args
    verify = json.loads(Path(env["VERIFY_ARGS"]).read_text())
    assert "scripts.utils.build_hf_sft_data_manifest" in verify
    assert f"mixture={stage}" in verify
    expected_parent = "standard_steered" if stage == "safety" else "safety"
    assert f"expected_parent_mixture={expected_parent}" in verify
