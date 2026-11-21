#!/bin/bash
# One-command install validation: extracts SAE features from a single pretraining shard.
# Expected runtime: ~2-5 min on a single H100, ~5-10 min on a 4090.
#
# Usage:
#   bash runs/extract_features_quick.sh
#
# Prerequisites:
#   - uv installed (this script will install it if missing)
#   - HF_TOKEN set (or you've run `huggingface-cli login`) for Gemma + GemmaScope downloads
#   - .env populated (cp .env.example .env)

set -euo pipefail

# Load .env if present (don't fail if missing — HF auth may be cached)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
# shellcheck disable=SC1091
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Step 1: download one pretraining data shard (~100MB)
echo "==> Downloading 1 pretraining shard..."
python -m nanochat.dataset -n 1

# -----------------------------------------------------------------------------
# Step 2: extract features
echo "==> Extracting SAE features..."
python -m steering.extract_features --config-name=extract_features_quick

# -----------------------------------------------------------------------------
# Step 3: verify output exists
EXPECTED_DIR="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
if [ -d "$EXPECTED_DIR" ] && [ -n "$(ls -A "$EXPECTED_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "==> SUCCESS: features extracted to $EXPECTED_DIR"
    echo "    $(find "$EXPECTED_DIR" -name '*.safetensors' | wc -l) shard(s) written"
    echo ""
    echo "Next: try the tiny pretraining smoke test:"
    echo "  torchrun --standalone --nproc_per_node=1 -m scripts.train.base_train -- config_name=base_train_tiny_1gpu"
else
    echo ""
    echo "==> FAILED: expected output at $EXPECTED_DIR"
    echo "    Check the extraction log above for errors."
    exit 1
fi
