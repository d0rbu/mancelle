#!/bin/bash
# Sourced by the HF decoder launchers after their topology and defaults are set.
# Keep benchmark and production code/lock/data/restart checks in one place.

die() {
    echo "$*" >&2
    exit 2
}

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

required_environment=(
    EXPECTED_GIT_REVISION
    EXPECTED_UV_LOCK_SHA256
    DATA_MANIFEST
    DATA_MANIFEST_SHA256
    NANOCHAT_BASE_DIR
    HF_HOME
    OUTPUT_DIR
)
for name in "${required_environment[@]}"; do
    [[ -n "${!name:-}" ]] || die "${name} must be a nonempty environment variable"
done
case "${launch_mode}:${CONFIG_NAME}" in
    benchmark:chat_sft_pythia_1_4b_h200_4x8_benchmark|benchmark:chat_sft_pythia_2_8b_h200_4x8_benchmark|benchmark:chat_sft_smollm3_3b_h200_4x8_benchmark) ;;
    production:chat_sft_pythia_2_8b_standard_steered) MANIFEST_SFT_CONFIG_NAME=chat_sft_pythia_2_8b ;;
    production:chat_sft_pythia_1_4b_standard_steered) MANIFEST_SFT_CONFIG_NAME=chat_sft_pythia_1_4b ;;
    production:chat_sft_pythia_2_8b_safety|production:chat_sft_pythia_2_8b_capabilities) MANIFEST_SFT_CONFIG_NAME=chat_sft_pythia_2_8b ;;
    *) die "unsupported CONFIG_NAME=${CONFIG_NAME} for launch_mode=${launch_mode}" ;;
esac
RESTART_FROM_STEP="${RESTART_FROM_STEP:--1}"
[[ "${RESTART_FROM_STEP}" =~ ^-1$|^0$|^[1-9][0-9]*$ ]] || \
    die "RESTART_FROM_STEP must be -1 or non-negative"

[[ "${SLURM_NNODES:-}" == "${expected_nodes}" ]] || die "expected ${expected_nodes} allocated nodes"
if [[ "${launch_mode}" == "production" ]]; then
    [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_GPUS_ON_NODE must be a positive integer"
    [[ "${SLURM_GPUS_ON_NODE}" == "${expected_gpus}" ]] || die "expected ${expected_gpus} GPUs on this node"
    gpus_per_node="${SLURM_GPUS_ON_NODE}"
else
    gpus_per_node="${expected_gpus}"
fi
[[ -x .venv/bin/torchrun ]] || die "the checked-in environment is missing .venv/bin/torchrun"
[[ -f "${DATA_MANIFEST}" ]] || die "DATA_MANIFEST does not exist: ${DATA_MANIFEST}"
if ! .venv/bin/python -c '
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(payload, (dict, list)) or not payload:
    raise ValueError("manifest must be a nonempty JSON object or array")
' "${DATA_MANIFEST}"; then
    die "DATA_MANIFEST must contain a nonempty JSON object or array"
fi
[[ -d "${NANOCHAT_BASE_DIR}" && -d "${HF_HOME}" ]] || \
    die "NANOCHAT_BASE_DIR and HF_HOME must be pre-staged directories"
[[ -f "${NANOCHAT_BASE_DIR}/identity_conversations.jsonl" ]] || \
    die "NANOCHAT_BASE_DIR is missing identity_conversations.jsonl"
[[ "${OUTPUT_DIR}" == /* ]] || die "OUTPUT_DIR must be an absolute shared-filesystem path"
if [[ "${RESTART_FROM_STEP}" == "-1" ]]; then
    if [[ "${launch_mode}" == "production" && -d "${OUTPUT_DIR}" && ! -L "${OUTPUT_DIR}" ]]; then
        [[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 ! -name logs -print -quit)" ]] || \
            die "a fresh run requires a new OUTPUT_DIR (only logs may exist): ${OUTPUT_DIR}"
    else
        [[ ! -e "${OUTPUT_DIR}" ]] || die "a fresh run requires a new OUTPUT_DIR: ${OUTPUT_DIR}"
    fi
else
    [[ -d "${OUTPUT_DIR}" ]] || die "a restart requires the existing OUTPUT_DIR: ${OUTPUT_DIR}"
    printf -v restart_marker '%s/restart/complete_%06d.json' \
        "${OUTPUT_DIR}" "${RESTART_FROM_STEP}"
    [[ -s "${restart_marker}" ]] || die "restart complete marker is missing or empty: ${restart_marker}"
    [[ -f "${OUTPUT_DIR}/data_manifest.json" ]] || \
        die "restart output is missing its retained data manifest"
fi

actual_git_revision="$(git rev-parse HEAD)"
actual_lock_sha256="$(sha256sum uv.lock | awk '{print $1}')"
actual_data_sha256="$(sha256sum "${DATA_MANIFEST}" | awk '{print $1}')"
[[ "${actual_git_revision}" == "${EXPECTED_GIT_REVISION}" ]] || \
    die "git revision mismatch: ${actual_git_revision}"
[[ "${actual_lock_sha256}" == "${EXPECTED_UV_LOCK_SHA256}" ]] || \
    die "uv.lock digest mismatch: ${actual_lock_sha256}"
[[ "${actual_data_sha256}" == "${DATA_MANIFEST_SHA256}" ]] || \
    die "data-manifest digest mismatch: ${actual_data_sha256}"
if [[ "${RESTART_FROM_STEP}" != "-1" ]]; then
    retained_data_sha256="$(sha256sum "${OUTPUT_DIR}/data_manifest.json" | awk '{print $1}')"
    [[ "${retained_data_sha256}" == "${actual_data_sha256}" ]] || \
        die "restart data manifest differs from the retained run manifest"
fi
[[ -z "$(git status --porcelain)" ]] || die "repository files must be clean for HF SFT"

ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"
case "${ATTENTION_IMPLEMENTATION}" in
    sdpa|flash_attention_4) ;;
    *) die "unsupported ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION}" ;;
esac

DISTRIBUTED_IMPLEMENTATION="${DISTRIBUTED_IMPLEMENTATION:-dist_muon_adamw}"
case "${DISTRIBUTED_IMPLEMENTATION}" in
    dist_muon_adamw|ddp_fused_adamw) ;;
    *) die "unsupported DISTRIBUTED_IMPLEMENTATION=${DISTRIBUTED_IMPLEMENTATION}" ;;
esac

PROFILE_TRACE_STEPS="${PROFILE_TRACE_STEPS:-[]}"
PROFILE_TRACE_RANKS="${PROFILE_TRACE_RANKS:-[]}"

[[ "${NUM_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ITERATIONS must be a positive integer"
for name in DEVICE_BATCH_SIZE MAX_SEQ_LEN TOTAL_BATCH_SIZE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || \
        die "${name} must be a positive integer, got ${!name}"
done
[[ "${RESTART_SAVE_EVERY}" =~ ^[0-9]+$ ]] || die "RESTART_SAVE_EVERY must be non-negative"
world_micro_batch_tokens=$((expected_nodes * gpus_per_node * DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
(( TOTAL_BATCH_SIZE % world_micro_batch_tokens == 0 )) || \
    die "TOTAL_BATCH_SIZE must be divisible by ${expected_nodes}*${gpus_per_node}*DEVICE_BATCH_SIZE*MAX_SEQ_LEN"

TORCH_COMPILE="${TORCH_COMPILE:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
for name in DRY_RUN TORCH_COMPILE GRADIENT_CHECKPOINTING PROFILE_TIMING VALIDATE_EXPORT; do
    [[ "${!name}" == "true" || "${!name}" == "false" ]] || \
        die "${name} must be true or false, got ${!name}"
done
if [[ ( "${PROFILE_TRACE_STEPS}" == "[]" && "${PROFILE_TRACE_RANKS}" != "[]" ) || \
      ( "${PROFILE_TRACE_STEPS}" != "[]" && "${PROFILE_TRACE_RANKS}" == "[]" ) ]]; then
    die "PROFILE_TRACE_STEPS and PROFILE_TRACE_RANKS must both be empty or nonempty"
fi
if [[ "${RESTART_FROM_STEP}" != "-1" ]] && \
    [[ "${PROFILE_TIMING}" == "true" || "${PROFILE_TRACE_STEPS}" != "[]" ]]; then
    die "restart invocations require PROFILE_TIMING=false and empty trace lists"
fi

if [[ "${launch_mode}" == "production" ]]; then
    case "${SFT_LOSS_MASK}" in
        full_sequence|completion_only) ;;
        *) die "unsupported SFT_LOSS_MASK=${SFT_LOSS_MASK}" ;;
    esac
    case "${CONFIG_NAME}" in
        chat_sft_pythia_1_4b_standard_steered|chat_sft_pythia_2_8b_standard_steered)
            [[ -s "${STEERED_JSONL:-}" ]] || \
                die "STEERED_JSONL must name the staged nonempty steered JSONL"
            HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                .venv/bin/python -m scripts.utils.build_hf_sft_data_manifest \
                mode=verify "base_dir=${NANOCHAT_BASE_DIR}" \
                "steered_jsonl=${STEERED_JSONL}" "output_path=${DATA_MANIFEST}" \
                "device_batch_size=${DEVICE_BATCH_SIZE}" \
                "sft_loss_mask=${SFT_LOSS_MASK}" \
                "sft_config_name=${MANIFEST_SFT_CONFIG_NAME}" || \
                die "live SFT data differs from DATA_MANIFEST"
            ;;
        chat_sft_pythia_2_8b_safety|chat_sft_pythia_2_8b_capabilities)
            [[ "${SFT_LOSS_MASK}" == completion_only ]] || \
                die "Campaign 2 requires completion_only"
            [[ -z "${STEERED_JSONL:-}" ]] || die "Campaign 2 does not consume STEERED_JSONL"
            stage="${CONFIG_NAME#chat_sft_pythia_2_8b_}"
            expected_parent_mixture=standard_steered
            [[ "${stage}" == safety ]] || expected_parent_mixture=safety
            HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                .venv/bin/python -m scripts.utils.build_hf_sft_data_manifest mode=verify \
                "mixture=${stage}" "base_dir=${NANOCHAT_BASE_DIR}" \
                "parent_export=${HF_PARENT_EXPORT:?}" \
                "expected_parent_mixture=${expected_parent_mixture}" \
                "data_manifest=${C2_DATA_MANIFEST:?}" "output_path=${DATA_MANIFEST}" \
                "device_batch_size=${DEVICE_BATCH_SIZE}" sft_loss_mask=completion_only || \
                die "Campaign 2 parent/data manifest differs"
            parent_revision="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign2_parent"]["source_revision"])' "${DATA_MANIFEST}")"
            [[ "${HF_PARENT_REVISION:?}" == "${parent_revision}" ]] || \
                die "parent revision mismatch"
            STEERED_JSONL=null
            ;;
        *) die "unsupported production CONFIG_NAME=${CONFIG_NAME}" ;;
    esac
    manifest_iterations="$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["epoch"]["num_iterations"])' "${DATA_MANIFEST}")"
    [[ "${NUM_ITERATIONS}" == "${manifest_iterations}" ]] || die "NUM_ITERATIONS differs from the measured manifest epoch"
fi

invocation_dir="${OUTPUT_DIR}/invocations/job_${SLURM_JOB_ID}"
[[ ! -e "${invocation_dir}" ]] || die "refusing to overwrite invocation evidence: ${invocation_dir}"
mkdir -p "${OUTPUT_DIR}/profile" "${OUTPUT_DIR}/preflight" "${invocation_dir}"
if [[ "${RESTART_FROM_STEP}" == "-1" ]]; then
    cp "${DATA_MANIFEST}" "${OUTPUT_DIR}/data_manifest.json"
fi
cp "${DATA_MANIFEST}" "${invocation_dir}/data_manifest.json"
scontrol show job "${SLURM_JOB_ID}" > "${invocation_dir}/slurm_job.txt"
{
    printf 'git_revision=%s\n' "${actual_git_revision}"
    printf 'uv_lock_sha256=%s\n' "${actual_lock_sha256}"
    printf 'data_manifest_sha256=%s\n' "${actual_data_sha256}"
    if [[ "${launch_mode}" == "production" ]]; then
        printf 'steered_jsonl=%s\n' "${STEERED_JSONL}"
        printf 'hf_parent_export=%s\n' "${HF_PARENT_EXPORT:-}"
        printf 'sft_loss_mask=%s\n' "${SFT_LOSS_MASK}"
        printf 'manifest_sft_config_name=%s\n' "${MANIFEST_SFT_CONFIG_NAME}"
        printf 'eval_every=150\nsave_every=500\n'
    fi
    printf 'config_name=%s\n' "${CONFIG_NAME}"
    printf 'attention_implementation=%s\n' "${ATTENTION_IMPLEMENTATION}"
    printf 'distributed_implementation=%s\n' "${DISTRIBUTED_IMPLEMENTATION}"
    printf 'num_iterations=%s\n' "${NUM_ITERATIONS}"
    printf 'device_batch_size=%s\n' "${DEVICE_BATCH_SIZE}"
    printf 'max_seq_len=%s\n' "${MAX_SEQ_LEN}"
    printf 'total_batch_size=%s\n' "${TOTAL_BATCH_SIZE}"
    printf 'torch_compile=%s\n' "${TORCH_COMPILE}"
    printf 'torchinductor_compile_threads=%s\n' "${TORCHINDUCTOR_COMPILE_THREADS:-8}"
    printf 'gradient_checkpointing=%s\n' "${GRADIENT_CHECKPOINTING}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
    printf 'validate_export=%s\n' "${VALIDATE_EXPORT}"
    printf 'restart_from_step=%s\n' "${RESTART_FROM_STEP}"
    printf 'restart_save_every=%s\n' "${RESTART_SAVE_EVERY}"
    printf 'profile_timing=%s\n' "${PROFILE_TIMING}"
    printf 'profile_trace_steps=%s\n' "${PROFILE_TRACE_STEPS}"
    printf 'profile_trace_ranks=%s\n' "${PROFILE_TRACE_RANKS}"
    printf 'output_dir=%s\n' "${OUTPUT_DIR}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
} > "${invocation_dir}/run_contract.env"
.venv/bin/python -c '
import json
import platform
import sys
from importlib.metadata import version

import torch as t
import transformers

print(json.dumps({
    "cuda_runtime": t.version.cuda,
    "cudnn": t.backends.cudnn.version(),
    "flash_attn_4": version("flash-attn-4"),
    "kernels": version("kernels"),
    "nccl": t.cuda.nccl.version(),
    "platform": platform.platform(),
    "python": sys.version,
    "torch": t.__version__,
    "transformers": transformers.__version__,
}, sort_keys=True))
' > "${invocation_dir}/software_versions.json"

export GPUS_PER_NODE="${gpus_per_node}"
export PREFLIGHT_REPO_ROOT="${repo_root}"
export PREFLIGHT_RESULT_DIR="${OUTPUT_DIR}/preflight/nccl_${SLURM_JOB_ID}"
export TORCHRUN_BIN=.venv/bin/torchrun
scripts/slurm/preflight_nccl_transport.sh
source "${PREFLIGHT_RESULT_DIR}/transport.env"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
(( ${#nodes[@]} == expected_nodes )) || die "expected ${expected_nodes} node names, got ${#nodes[@]}"
MASTER_ADDR="$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')"
[[ -n "${MASTER_ADDR}" ]] || die "could not resolve the master node IPv4 address"
MASTER_PORT="${MASTER_PORT:-$((33500 + SLURM_JOB_ID % 1000))}"

export MASTER_ADDR MASTER_PORT CONFIG_NAME ATTENTION_IMPLEMENTATION
export DISTRIBUTED_IMPLEMENTATION NUM_ITERATIONS DEVICE_BATCH_SIZE MAX_SEQ_LEN
export TOTAL_BATCH_SIZE RESTART_FROM_STEP RESTART_SAVE_EVERY
export PROFILE_TRACE_STEPS PROFILE_TRACE_RANKS
export DRY_RUN TORCH_COMPILE GRADIENT_CHECKPOINTING PROFILE_TIMING VALIDATE_EXPORT
export OUTPUT_DIR NANOCHAT_BASE_DIR HF_HOME
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PROFILE_OUTPUT_DIR="${OUTPUT_DIR}/profile"
export DATA_MANIFEST_SHA256="${actual_data_sha256}"
