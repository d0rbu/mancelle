#!/bin/bash
set -euo pipefail

# Train a LARGER GPT model (depth=27, ~1.98B params) on ALL 370 available data shards.
# depth=27 is chosen as the closest match to 1.92B total params.
# target_param_data_ratio=12.6 reflects the effective ratio (~13.9B tokens / ~1.1B scaling params)
# for correct batch size, learning rate, and weight decay scaling.
#
# Model comparison:
#   speedrun (d26): model_dim=1664, 13 heads, ~918M scaling params, ~1.68B total, ~7.6B tokens
#   this    (d27): model_dim=1792, 14 heads, ~1.1B scaling params, ~1.98B total, ~13.9B tokens
#
# Designed for 8xH100 GPUs (approx. 2x the runtime of speedrun.sh).

# 1) Example launch (simplest):
# bash runs/train_370_l.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/train_370_l.log -S train370l bash runs/train_370_l.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=d27 screen -L -Logfile runs/train_370_l.log -S train370l bash runs/train_370_l.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

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
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + Dataset

# Start downloading all 370 shards in the background (369 train + 1 val, auto-split by dataloader)
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -f "$TOKENIZER_DIR/token_bytes.pt" ] && { [ -f "$TOKENIZER_DIR/tokenizer.json" ] || [ -f "$TOKENIZER_DIR/tokenizer.pkl" ]; }; then
    echo "Pretrained tokenizer found at $TOKENIZER_DIR, skipping tokenizer training"
else
    echo "No pretrained tokenizer found, training from scratch..."
    # Wait for at least 8 shards to be available for tokenizer training
    python -m nanochat.dataset -n 8
    python -m scripts.preprocess.tok_train
    python -m scripts.preprocess.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d27 model (~1.98B total params, closest to 1.92B target)
# ratio=12.6 reflects effective tokens:scaling_params for correct lr/wd scaling
# num_iterations=13232 to consume exactly all 369 train shards (at 1M batch size)
# Checkpoint saved to d27/
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- depth=27 target_param_data_ratio=12.6 num_iterations=13232 device_batch_size=16 fp8=true run=$WANDB_RUN

# Evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- device_batch_size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.train.chat_sft -- device_batch_size=16 run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.eval.chat_eval -- source=sft

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
