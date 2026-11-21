#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"

: "${TURN_STATS_PATH:?Set the local path to the exact turn training statistics}"
FIXED_STATS_PATH=${FIXED_STATS_PATH:-$BASE_FEATURES/feature_statistics.safetensors}

# Six scalar jobs per checkpoint, shown sequentially for unambiguous completion.
# problem_start/max_problems are exact half-open range requests in the new mode.
run_selfcond_family() {
  local chat_base="$1" chat_tag="$2" chat_step="$3" chat_cache="$4" chat_out="$5" chat_stats="$6"
  local chat_spec chat_task chat_count
  export NANOCHAT_BASE_DIR="$chat_base"
  for chat_spec in ARC-Easy:500 ARC-Challenge:500 MMLU:500 GSM8K:500 HumanEval:164 SpellingBee:256; do
    chat_task=${chat_spec%:*}
    chat_count=${chat_spec#*:}
    srun -p mit_normal_gpu -A mit_amf_advanced_gpu -q mit_amf_advanced_gpu \
      --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G \
      --gres=gpu:h200:1 --time=12:00:00 \
      uv run --no-sync python -m scripts.eval.chat_eval \
      --config-name=chat_eval_self_conditioned \
      evaluation_mode=self_conditioned source=encdec \
      model_tag="$chat_tag" step="$chat_step" task_name="$chat_task" \
      device_type=cuda dtype=bfloat16 model_max_seq_len=2048 \
      problem_start=0 max_problems="$chat_count" \
      batch_size=1 num_samples=1 temperature=0 top_k=50 max_new_tokens=512 \
      stats_path="$chat_stats" \
      next_segment_conditioning_offset=false \
      self_conditioned.cache_dir="$chat_cache" self_conditioned.seed=1234 \
      self_conditioned_output_dir="$chat_out/$chat_task"
  done
  uv run --no-sync python -m scripts.eval.merge_chat_eval_target_real \
    "input_files=[$chat_out/ARC-Easy/self_conditioned/result.yaml,$chat_out/ARC-Challenge/self_conditioned/result.yaml,$chat_out/MMLU/self_conditioned/result.yaml,$chat_out/GSM8K/self_conditioned/result.yaml,$chat_out/HumanEval/self_conditioned/result.yaml,$chat_out/SpellingBee/self_conditioned/result.yaml]" \
    'task_totals={ARC-Easy:500,ARC-Challenge:500,MMLU:500,GSM8K:500,HumanEval:164,SpellingBee:256}' \
    output_file="$chat_out/merged/self_conditioned/result.yaml"
}

run_selfcond_family "$FIXED_RUNTIME" "$TAG" 250000 "$FIXED_CACHE" "$B/selfcond_results" "$FIXED_STATS_PATH"
run_selfcond_family "$TURN_RUNTIME" "$TURN_TAG" 1116 "$TURN_CACHE" "$S/chatcore/s32/selfcond_results" "$TURN_STATS_PATH"
