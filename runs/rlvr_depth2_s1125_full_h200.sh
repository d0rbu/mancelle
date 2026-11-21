#!/usr/bin/env bash
# Run from a dedicated immutable checkout; see the RLVR runbook for sbatch.
set -euo pipefail

: "${RLVR_SOURCE_REVISION:?Set the exact reviewed source commit}"
: "${RLVR_BASE_DIR:?Set the authenticated cadence-bearing step-1125 view}"
: "${RLVR_BUNDLE_DIR:?Set the finalized full four-task bundle}"
: "${RLVR_STATS_PATH:?Set the recorded statistics artifact}"
: "${RLVR_OUTPUT_DIR:?Set a separate smoke or production output directory}"
: "${NANOCHAT_BASE_DIR:?Set the pinned RustBPE tokenizer root}"
rlvr_python=${RLVR_PYTHON:-python3}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(git rev-parse HEAD)" = "$RLVR_SOURCE_REVISION"
test -z "$(git status --porcelain)"

export NANOCHAT_ATTN_IMPL=fa4 NANOCHAT_DTYPE=bfloat16
export NCCL_TIMEOUT_SEC=3600 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$rlvr_python" -c 'import torch as t; assert t.cuda.device_count() == 1 and "H200" in t.cuda.get_device_name(0), "This preset requires exactly one visible H200"'

exec "$rlvr_python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m scripts.train.conditional_encdec_rlvr -- \
  config_name=conditional_encdec_rlvr_depth2_s1125_full_h200 \
  "base_checkpoint_dir=$RLVR_BASE_DIR" "bundle_dir=$RLVR_BUNDLE_DIR" \
  "stats_path=$RLVR_STATS_PATH" "output_dir=$RLVR_OUTPUT_DIR" "$@"
