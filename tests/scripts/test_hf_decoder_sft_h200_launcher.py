"""Static fail-closed contract for the 4x8 H200 decoder-SFT launcher."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "slurm" / "hf_decoder_sft_h200_4x8.sbatch"

PREFLIGHT = LAUNCHER.with_name("hf_decoder_sft_preflight.sh")


def test_launcher_pins_the_exact_production_topology_and_no_restart_tolerance() -> None:
    text = LAUNCHER.read_text() + PREFLIGHT.read_text()
    assert "#SBATCH --nodes=4" in text
    assert "#SBATCH --gpus-per-node=h200:8" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "--nnodes=4" in text
    assert "--nproc_per_node=8" in text
    assert '--node_rank="${SLURM_PROCID}"' in text
    assert "--max_restarts=0" in text
    assert "--standalone" not in text


def test_launcher_requires_immutable_code_lock_and_data_identities() -> None:
    text = LAUNCHER.read_text() + PREFLIGHT.read_text()
    for field in (
        "EXPECTED_GIT_REVISION",
        "EXPECTED_UV_LOCK_SHA256",
        "DATA_MANIFEST",
        "DATA_MANIFEST_SHA256",
    ):
        assert field in text
    assert "git status --porcelain)" in text
    assert "--untracked-files=no" not in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "HF_DATASETS_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "manifest must be a nonempty JSON object or array" in text
    assert "identity_conversations.jsonl" in text
    assert '[[ "${OUTPUT_DIR}" == /* ]]' in text
    assert "a fresh run requires a new OUTPUT_DIR" in text
    assert "a restart requires the existing OUTPUT_DIR" in text
    assert "complete_%06d.json" in text
    assert "restart data manifest differs from the retained run manifest" in text


def test_launcher_whitelists_all_full_model_canaries_and_strategy_axes() -> None:
    text = LAUNCHER.read_text() + PREFLIGHT.read_text()
    assert "chat_sft_pythia_1_4b_h200_4x8_benchmark" in text
    assert "chat_sft_pythia_2_8b_h200_4x8_benchmark" in text
    assert "chat_sft_smollm3_3b_h200_4x8_benchmark" in text
    assert "sdpa|flash_attention_4" in text
    assert "dist_muon_adamw|ddp_fused_adamw" in text
    assert '"huggingface.attention_implementation=${ATTENTION_IMPLEMENTATION}"' in text
    assert '"huggingface.distributed_implementation=${DISTRIBUTED_IMPLEMENTATION}"' in text


def test_launcher_runs_transport_preflight_and_retains_rank_local_paid_step_rows() -> None:
    text = LAUNCHER.read_text() + PREFLIGHT.read_text()
    assert "scripts/slurm/preflight_nccl_transport.sh" in text
    assert 'source "${PREFLIGHT_RESULT_DIR}/transport.env"' in text
    assert '"profile_timing=${PROFILE_TIMING}"' in text
    assert '"profile_output_dir=${PROFILE_OUTPUT_DIR}"' in text
    assert '"huggingface.validate_export=${VALIDATE_EXPORT}"' in text
    assert '"${invocation_dir}/run_contract.env"' in text
    assert '"${invocation_dir}/software_versions.json"' in text
    assert "invocations/job_${SLURM_JOB_ID}" in text
    for package in (
        "torch",
        "transformers",
        "flash_attn_4",
        "kernels",
        "cuda_runtime",
        "cudnn",
        "nccl",
    ):
        assert f'"{package}"' in text
    assert "-m scripts.train.chat_sft --" in text
    for field in (
        "restart_from_step",
        "restart_save_every",
        "profile_timing",
        "profile_trace_steps",
        "profile_trace_ranks",
        "torchinductor_compile_threads",
        "output_dir",
    ):
        assert f"printf '{field}" in text


def test_launcher_rejects_invalid_geometry_and_profiled_restart_overwrites() -> None:
    text = LAUNCHER.read_text() + PREFLIGHT.read_text()
    assert "TOTAL_BATCH_SIZE % world_micro_batch_tokens" in text
    assert "expected_nodes=4" in text
    assert "expected_gpus=8" in text
    assert "expected_nodes * gpus_per_node * DEVICE_BATCH_SIZE * MAX_SEQ_LEN" in text
    assert "restart invocations require PROFILE_TIMING=false and empty trace lists" in text
    assert 'TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"' in text
