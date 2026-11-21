#!/bin/bash
set -eo pipefail

# Single-GPU (4090, 24GB VRAM) full training pipeline.
# No torchrun (single GPU, no DDP). No FP8 (Ada Lovelace lacks support).
# Uses window_pattern=L because SDPA fallback has terrible perf with sliding window.
#
# d26 (1.68B params) OOMs on 24GB VRAM with torch.compile, so we use d24 (1.38B).
# device_batch_size=1 with gradient accumulation to total_batch_size.
#   grad_accum_steps = total_batch_size / (1 * 2048 * 1) = 524288 / 2048 = 256
#
# Usage:
#   bash runs/run_4090.sh
#   WANDB_RUN=debug4090 bash runs/run_4090.sh

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Python venv setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Reset report
python -m nanochat.report reset

# Download data (d24 needs ~295 shards at 8.25 ratio; 370 is safe/conservative)
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!

# Train tokenizer (skip if already trained)
python -m scripts.preprocess.tok_train
python -m scripts.preprocess.tok_eval

# Wait for data
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Pretrain (single GPU, no torchrun)
# Checkpoints saved every 500 steps by default (save_every in base_train.yaml)
uv run python -m scripts.train.base_train \
    depth=24 \
    device_batch_size=1 \
    window_pattern=L \
    run=$WANDB_RUN

# Evaluate
uv run python -m scripts.eval.base_eval device_batch_size=1

# Download identity conversations for SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# SFT (single GPU)
uv run python -m scripts.train.chat_sft \
    device_batch_size=1 \
    run=$WANDB_RUN

# Evaluate SFT model
uv run python -m scripts.eval.chat_eval source=sft

# Generate report
python -m nanochat.report generate
