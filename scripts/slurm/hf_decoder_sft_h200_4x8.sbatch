#!/bin/bash
#SBATCH --job-name=hf-sft-h200-4x8
#SBATCH --output=experiments/logs/hf_sft_h200_4x8_%j.out
#SBATCH --error=experiments/logs/hf_sft_h200_4x8_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --exclusive

# Submit with the approved advanced account/QoS pair. Required environment:
# EXPECTED_GIT_REVISION, EXPECTED_UV_LOCK_SHA256, DATA_MANIFEST,
# DATA_MANIFEST_SHA256, NANOCHAT_BASE_DIR, HF_HOME, and OUTPUT_DIR.

set -euo pipefail
launch_mode=benchmark
expected_nodes=4
expected_gpus=8
CONFIG_NAME="${CONFIG_NAME:-chat_sft_pythia_2_8b_h200_4x8_benchmark}"
NUM_ITERATIONS="${NUM_ITERATIONS:-60}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
RESTART_SAVE_EVERY="${RESTART_SAVE_EVERY:-0}"
DRY_RUN="${DRY_RUN:-true}"
PROFILE_TIMING="${PROFILE_TIMING:-true}"
VALIDATE_EXPORT="${VALIDATE_EXPORT:-false}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/hf_decoder_sft_preflight.sh"

srun --kill-on-bad-exit=1 \
    --nodes=4 \
    --ntasks=4 \
    --ntasks-per-node=1 \
    bash -c '
        exec .venv/bin/torchrun \
            --nnodes=4 \
            --nproc_per_node=8 \
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
            "eval_every=0" \
            "save_every=0" \
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
