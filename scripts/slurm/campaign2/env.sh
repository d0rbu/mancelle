#!/bin/bash
# Source after setting MIXTURE. All paths are operator-supplied; no remote defaults.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/guard.sh"
for name in CAMPAIGN_ROOT NANOCHAT_BASE_DIR HF_HOME EXPECTED_GIT_REVISION; do
    [[ -n ${!name:-} ]] || die "export $name"
done
[[ $CAMPAIGN_ROOT == /* && $NANOCHAT_BASE_DIR == /* && $HF_HOME == /* ]] || die 'use absolute paths'
case "${MIXTURE:-}" in safety|capabilities) ;; *) die 'MIXTURE must be safety or capabilities';; esac
TORCHRUN=${CAMPAIGN_TORCHRUN:-$REPO/.venv/bin/torchrun}
[[ -d $NANOCHAT_BASE_DIR && -d $HF_HOME ]] || die 'stage environment and data first'
export NANOCHAT_BASE_DIR HF_HOME
export NCCL_TIMEOUT_SEC=3600
FEATURE_ROOT=$CAMPAIGN_ROOT/features
FAMILY=$FEATURE_ROOT/s32/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt
TRAIN_LEAF=$FAMILY/${MIXTURE}_sft/context_independent/no_bos/turn_delimited
VAL_LEAF=$FAMILY/${MIXTURE}_sft_val/context_independent/no_bos/turn_delimited
TRAIN_PARQUET=$NANOCHAT_BASE_DIR/sft_data/$MIXTURE
VAL_PARQUET=$NANOCHAT_BASE_DIR/sft_data/${MIXTURE}_val
BASE=$CAMPAIGN_ROOT/encdec/$MIXTURE/base
SFT_LEAF=$BASE/${MIXTURE}_sft_conditioned_checkpoints
mkdir -p "$CAMPAIGN_ROOT/logs"
