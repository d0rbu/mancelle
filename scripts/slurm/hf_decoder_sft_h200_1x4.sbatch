#!/bin/bash
#SBATCH --job-name=pythia-sft-h200-1x4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00

# Pass partition/account/QoS and --output/--error (or SBATCH_OUTPUT/ERROR)
# at submission; see docs/pipelines/hf-decoder-sft.md for OUTPUT_DIR/logs.
set -euo pipefail
[[ "${OUTPUT_DIR:-}" == /* ]] || { echo "OUTPUT_DIR must be an absolute path" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}/logs"
exec >>"${OUTPUT_DIR}/logs/slurm_${SLURM_JOB_ID}.out" 2>>"${OUTPUT_DIR}/logs/slurm_${SLURM_JOB_ID}.err"
launch_mode=production
expected_nodes=1
expected_gpus=4
CONFIG_NAME="${CONFIG_NAME:-chat_sft_pythia_2_8b_standard_steered}"
NUM_ITERATIONS="${NUM_ITERATIONS:?export the positive epoch.num_iterations from DATA_MANIFEST}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
MAX_SEQ_LEN=2048
TOTAL_BATCH_SIZE=524288
RESTART_SAVE_EVERY=500
DRY_RUN=false
PROFILE_TIMING=false
VALIDATE_EXPORT=true
SFT_LOSS_MASK="${SFT_LOSS_MASK:-full_sequence}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/hf_decoder_sft_preflight.sh"
export STEERED_JSONL SFT_LOSS_MASK

srun --kill-on-bad-exit=1 \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    bash -c '
        exec .venv/bin/torchrun \
            --nnodes=1 \
            --nproc_per_node="${GPUS_PER_NODE}" \
            --node_rank="${SLURM_PROCID}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            --max_restarts=0 \
            -m scripts.train.chat_sft -- \
            "config_name=${CONFIG_NAME}" \
            "run=dummy" \
            "num_iterations=${NUM_ITERATIONS}" \
            "device_batch_size=${DEVICE_BATCH_SIZE}" \
            "max_seq_len=${MAX_SEQ_LEN}" \
            "total_batch_size=${TOTAL_BATCH_SIZE}" \
            "steered_jsonl=${STEERED_JSONL}" \
            "sft_loss_mask=${SFT_LOSS_MASK}" \
            "dry_run=${DRY_RUN}" \
            "profile_timing=${PROFILE_TIMING}" \
            "profile_trace_steps=${PROFILE_TRACE_STEPS}" \
            "profile_trace_ranks=${PROFILE_TRACE_RANKS}" \
            "profile_output_dir=${PROFILE_OUTPUT_DIR}" \
            "huggingface.attention_implementation=${ATTENTION_IMPLEMENTATION}" \
            "huggingface.distributed_implementation=${DISTRIBUTED_IMPLEMENTATION}" \
            "huggingface.torch_compile=${TORCH_COMPILE}" \
            "huggingface.gradient_checkpointing=${GRADIENT_CHECKPOINTING}" \
            "huggingface.restart_from_step=${RESTART_FROM_STEP}" \
            "huggingface.restart_save_every=${RESTART_SAVE_EVERY}" \
            "huggingface.data_snapshot_sha256=${DATA_MANIFEST_SHA256}" \
            "huggingface.output_dir=${OUTPUT_DIR}" \
            "huggingface.validate_export=${VALIDATE_EXPORT}"
    '
