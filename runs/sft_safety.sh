#!/bin/bash
# Safety SFT training script
# Usage: ./runs/sft_safety.sh [WANDB_RUN_NAME]
#
# Prerequisites:
# - Base model already trained (pretrain step)
# - Identity conversations downloaded to $NANOCHAT_BASE_DIR/identity_conversations.jsonl
# - HuggingFace access to gated datasets (allenai/wildjailbreak, allenai/wildguardmix)
#   Run: huggingface-cli login

set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${1:-safety_sft}"

echo "Running safety SFT with $NPROC_PER_NODE GPUs..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.train.chat_sft -- \
    mixture=safety \
    run=$WANDB_RUN \
    "${@:2}"
