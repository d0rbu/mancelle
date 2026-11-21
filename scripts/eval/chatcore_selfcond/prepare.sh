#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"

# Fresh runtime roots: fail if a previous campaign already owns these names.
test ! -e "$FIXED_RUNTIME"
test ! -e "$TURN_RUNTIME"
mkdir -p "$FIXED_RUNTIME" "$TURN_RUNTIME/conditional_encdec_checkpoints/$TAG"
ln -s "$G/full3b-w32fp-prod-20260827/nanochat_cache/tokenizer" "$FIXED_RUNTIME/tokenizer"
ln -s "$S/nanochat_cache/tokenizer" "$TURN_RUNTIME/tokenizer"
ln -s "$S/chatcore/tasks/GSM8K/base/base_data" "$FIXED_RUNTIME/base_data"
ln -s "$S/chatcore/tasks/GSM8K/base/base_data" "$TURN_RUNTIME/base_data"
ln -s "$S/eval_leaf_s32" "$TURN_RUNTIME/conditional_encdec_checkpoints/$TURN_TAG"

# Measure fixed K from the actual base TRAIN rows; no numerical K is guessed.
export NANOCHAT_BASE_DIR=$FIXED_RUNTIME
srun -p mit_normal -A mit_amf_advanced_cpu -q mit_amf_advanced_cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=02:00:00 \
  uv run --no-sync python -m scripts.utils.record_fixed_cadence \
  checkpoint_meta_path="$BASE_SOURCE/meta_250000.json" \
  feature_dir="$BASE_FEATURES" parquet_dir="$BASE_FEATURES/_parquets" \
  tokenizer_dir="$FIXED_RUNTIME/tokenizer" max_rows=4096 \
  output_dir="$FIXED_LEAF"

# CPU planning authenticates schema 3 and emits exact extraction inputs.
prepare_fixed_cache() {
  local preparation_stage="$1"
  srun -p mit_normal -A mit_amf_advanced_cpu -q mit_amf_advanced_cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=04:00:00 \
    uv run --no-sync python -m scripts.eval.prepare_chat_self_conditioned_cache \
    stage="$preparation_stage" checkpoint_meta_path="$FIXED_LEAF/meta_250000.json" \
    legacy_cache_dir="$B/cache/core_chat" \
    legacy_contexts_file="$B/contexts/core_chat_contexts.jsonl" \
    legacy_feature_dir="$B/extracted/core_chat" \
    legacy_parquet_dir="$B/contexts/core_chat_parquet" \
    output_dir="$FIXED_CACHE" chat_max_problems=500 \
    'chat_tasks=[ARC-Easy,ARC-Challenge,MMLU,GSM8K,HumanEval,SpellingBee]'
}
prepare_fixed_cache plan
srun -p mit_normal_gpu -A mit_amf_advanced_gpu -q mit_amf_advanced_gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G \
  --gres=gpu:h200:1 --time=12:00:00 \
  bash "$FIXED_CACHE.prepare/extract.sh"
prepare_fixed_cache publish
