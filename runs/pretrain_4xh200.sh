#!/bin/bash
set -euo pipefail

# Pretrain a GPT-2-scale (d26, ~1.4B params) model on 4xH200 GPUs.
# Trains a custom 32K BPE tokenizer (same as speedrun.sh).
# Based on runs/speedrun.sh but adapted for 4xH200 (fewer GPUs, larger batch per device).

# 1) Example launch (simplest):
# bash runs/pretrain_4xh200.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/pretrain_4xh200.log -S pretrain bash runs/pretrain_4xh200.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=d26 screen -L -Logfile runs/pretrain_4xh200.log -S pretrain bash runs/pretrain_4xh200.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# System dependencies (gcc + Python headers needed for Triton/torch.compile)

if ! command -v gcc &> /dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq gcc python3-dev
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

export PATH="$HOME/.local/bin:$PATH"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer training (custom 32K BPE, same as speedrun.sh)

# Download the first ~2B characters of pretraining dataset (8 shards)
python -m nanochat.dataset -n 8
# Kick off downloading the rest of the shards in the background
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# Train the tokenizer with vocab size 2**15 = 32768 on ~2B characters
python -m scripts.preprocess.tok_train
# Evaluate the tokenizer (report compression ratio etc.)
python -m scripts.preprocess.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Key differences from speedrun.sh (8xH100):
#   - 4 processes instead of 8 (4xH200)
#   - device_batch_size=32 (doubled from 16 to compensate for fewer GPUs)
#   - Total tokens per microbatch: 4 * 32 * 2048 = 262,144 (same as 8 * 16 * 2048)
#   - FP8 training enabled (H200 supports FP8 like H100)

torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- device_batch_size=32 fp8=true run=$WANDB_RUN

# Evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=4 -m scripts.eval.base_eval -- device_batch_size=32

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
