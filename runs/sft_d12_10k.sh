#!/bin/bash
# Chained SFT for d12_base_10k: safety -> capabilities
# Run after base pretraining (model_tag=d12_base_10k) is complete.
#
# Prerequisites:
# - Base model already trained (model_tag=d12_base_10k)
# - HuggingFace access to gated datasets (allenai/wildjailbreak, allenai/wildguardmix)
#   Run: huggingface-cli login

set -euo pipefail
cd "$(dirname "$0")/.."

# Default intermediate artifacts directory (matches nanochat.common.get_base_dir())
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"

# Activate the project venv if not already activated
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== Stage 1: Safety SFT ==="
torchrun --standalone --nproc_per_node=4 -m scripts.train.chat_sft -- config_name=sft_d12_10k_safety save_every=500

echo "=== Stage 2: Capabilities SFT ==="
torchrun --standalone --nproc_per_node=4 -m scripts.train.chat_sft -- config_name=sft_d12_10k_capabilities save_every=500

echo "=== SFT chain complete ==="

echo "=== Stage 3: Progression eval ==="
uv run python -c "
import os
from hydra import compose, initialize_config_dir
config_dir = os.path.join(os.getcwd(), 'configs/eval')
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name='training_progression_d12_base_10k')
from scripts.train.training_progression import collect_results, plot_results
collect_results(cfg)
plot_results(cfg)
"

echo "=== Progression eval complete ==="
