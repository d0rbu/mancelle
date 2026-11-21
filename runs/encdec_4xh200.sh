#!/bin/bash

# Encoder-decoder conditional pretraining pipeline on 4xH200 GPUs.
# Phase 1: Extract SAE features from all 370 shards (4-way parallel)
# Phase 2: Compute corpus-wide feature statistics
# Phase 3: (manual) Run encoder-decoder training with the extracted features
#
# Model: GPT-2-scale decoder (d26, ~1.4B params) + feature encoder
# SAE: GemmaScope 16k features, layer 8, residual activations
# Extraction: max_pool, fixed 128-token segments, context_window=1024
# Thresholding: feature_percentile_segment_topk at 2048.9 (p90 + top-2048 cap, applied at training time)
#
# Usage:
#   bash runs/encdec_4xh200.sh
#   screen -L -Logfile runs/encdec_4xh200.log -S encdec bash runs/encdec_4xh200.sh

set -euo pipefail

export OMP_NUM_THREADS=1

FEATURE_DIR="out/data/features"
NUM_SHARDS=370

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
# Download all shards (kick off in background, wait before extraction)

echo "Downloading ${NUM_SHARDS} shards..."
python -m nanochat.dataset -n ${NUM_SHARDS}

# =============================================================================
# Phase 1: Feature extraction (4xH200, ~93 shards per GPU)
# =============================================================================

echo ""
echo "================================================================"
echo "Phase 1: Extracting SAE features from ${NUM_SHARDS} shards"
echo "  Config: max_pool, fixed_tokens(128), layer=8, residual, 16k"
echo "  GPUs: 4xH200 (gpu_batch=96, cpu_batch=192)"
echo "================================================================"
echo ""

torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    --config-name extract_features_4xh200 \
    num_shards=${NUM_SHARDS} \
    output_dir=${FEATURE_DIR}

echo ""
echo "Feature extraction complete. Output: ${FEATURE_DIR}"
echo ""

# =============================================================================
# Phase 2: Compute feature statistics
# =============================================================================

echo "================================================================"
echo "Phase 2: Computing corpus-wide feature statistics"
echo "================================================================"
echo ""

python -m steering.compute_statistics \
    feature_dir=${FEATURE_DIR} \
    device=cuda

echo ""
echo "Statistics computation complete."
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "================================================================"
echo "Pipeline complete!"
echo ""
echo "  Features:   ${FEATURE_DIR}/"
echo "  Statistics:  ${FEATURE_DIR}/feature_statistics.safetensors"
echo ""
echo "To start encoder-decoder training:"
echo "  torchrun --standalone --nproc_per_node=4 \\"
echo "    -m scripts.train.conditional_encdec_train \\"
echo "    -- \\"
echo "    config_name=conditional_encdec_4xh200 \\"
echo "    feature_dir=${FEATURE_DIR}"
echo "================================================================"
