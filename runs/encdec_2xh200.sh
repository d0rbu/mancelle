#!/bin/bash

# Encoder-decoder conditional pretraining on 2xH200 GPUs (FP8).
# Assumes feature extraction (370 shards) and statistics are already complete.
#
# Model: GPT-2-scale decoder (d26, ~1.4B params) + feature encoder
# SAE: GemmaScope 16k features, layer 8, residual activations
# Extraction: max_pool, fixed 128-token segments, context_window=1024
# Thresholding: feature_percentile_segment_topk at 2048.9 (p90 + top-2048 cap)
# FP8: enabled (H200 supports FP8)
#
# Usage:
#   bash runs/encdec_2xh200.sh
#   screen -L -Logfile runs/encdec_2xh200.log -S encdec bash runs/encdec_2xh200.sh

set -euo pipefail

export OMP_NUM_THREADS=1

FEATURE_DIR="out/data/features"

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

# =============================================================================
# Training (2xH200, FP8)
# =============================================================================

echo ""
echo "================================================================"
echo "Encoder-decoder training (2xH200 FP8)"
echo "  Config: conditional_encdec_2xh200"
echo "  Features: ${FEATURE_DIR}"
echo "  depth=26, threshold=2048.9, fp8=true"
echo "================================================================"
echo ""

torchrun --standalone --nproc_per_node=2 \
    -m scripts.train.conditional_encdec_train \
    -- \
    config_name=conditional_encdec_2xh200 \
    feature_dir=${FEATURE_DIR} \
    run=encdec-d26-full-2xh200
