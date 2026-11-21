#!/bin/bash
# Shared local / Slurm extraction body. One process owns each whole parquet shard.
set -euo pipefail
source "${CAMPAIGN_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/campaign2/env.sh"
WORKER_INDEX=${WORKER_INDEX:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
GPU_BATCH_SEGMENTS=${GPU_BATCH_SEGMENTS:-2}
CPU_BATCH_DOCS=${CPU_BATCH_DOCS:-8}
[[ $WORKER_INDEX =~ ^[0-9]+$ && $WORKER_COUNT =~ ^[1-9][0-9]*$ ]] || die 'invalid worker geometry'
(( WORKER_INDEX < WORKER_COUNT )) || die 'worker index outside range'
for split in train val; do
    if [[ $split == train ]]; then parquet=$TRAIN_PARQUET; phase=${MIXTURE}_sft
    else parquet=$VAL_PARQUET; phase=${MIXTURE}_sft_val; fi
    count=$("$PY" -m scripts.utils.campaign2 action=inventory "parquet=$parquet")
    for ((shard=WORKER_INDEX; shard<count; shard+=WORKER_COUNT)); do
        "$PY" -m steering.extract_features --config-name extract_features_gemma3_4bpt_l22_f65k_ci_nobos \
            "dataset_phase=$phase" segment_mode=turn_delimited_fixed_tokens segment_size=32 \
            context_window=4096 "feature_root=$FEATURE_ROOT" "start_shard=$shard" "num_shards=$((shard+1))" \
            "gpu_batch_size=$GPU_BATCH_SEGMENTS" "cpu_batch_size=$CPU_BATCH_DOCS"
    done
done
