#!/bin/bash

# End-to-end prep of the gemma-3-4b-it / l22 / a_res / p_max_pool / f65k / base
# feature dataset for encoder-decoder training, on a single RTX 4090.
#
# The dataset arrived (rsync) with shards 0-368 only; parquet 0-369 exist.
# This script:
#   0. Extracts the missing feature shard 369 (gemma-scope-2, medium tier)
#      so packed-rows can use max_shards=null with parquet 369 as the val
#      split.  Skipped if shard 369 already exists.
#   0b. Validates the full feature directory (all 370 shards) via
#       steering.validate_features (CSR invariants + cross-shard metadata +
#       source re-tokenization). Fails fast on any violation.
#   2. compute_statistics      (exact CPU count table; bounded CUDA count reduction + moments)
#   3. precompute_percentiles  (device=cuda, resume)
#   4. precompute_token_boundaries (resume)
#   5. precompute_packed_rows  (max_shards=null, c1024, resume)
#
# All steps use the dedicated *_gemma3_l22_f65k configs and are resumable.
# Aborts if free disk on / drops below DISK_FLOOR_GB between phases.
#
# Usage (from repo root), backgrounded for the multi-day run:
#   setsid nohup bash runs/gemma3_l22_f65k_pipeline.sh \
#       > logs/gemma3_l22_f65k_pipeline.log 2>&1 < /dev/null &

set -euo pipefail
export OMP_NUM_THREADS=1

FEATURE_DIR="out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base"
SHARD_369="${FEATURE_DIR}/shard_00369_features.safetensors.zst"
STATS_FILE="${FEATURE_DIR}/feature_statistics.safetensors"
LOG_DIR="logs"
DISK_FLOOR_GB=200

mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

check_disk() {
    local free_gb
    free_gb=$(df -BG --output=avail / | tail -n1 | tr -dc '0-9')
    if [ "${free_gb}" -lt "${DISK_FLOOR_GB}" ]; then
        echo "[$(ts)] ABORT: free disk on / is ${free_gb} GB, below floor ${DISK_FLOOR_GB} GB"
        exit 1
    fi
    echo "[$(ts)] disk OK: ${free_gb} GB free"
}

[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "================================================================"
echo "[$(ts)] gemma-3-4b-it l22 f65k pipeline START"
echo "  feature_dir: ${FEATURE_DIR}"
echo "================================================================"
check_disk

# -----------------------------------------------------------------------------
# Step 0: extract the missing feature shard 369 (val parquet shard).
echo ""
echo "[$(ts)] Step 0: extract feature shard 369"
if [ -f "${SHARD_369}" ]; then
    echo "[$(ts)]   shard 369 already present — skipping extraction."
else
    uv run python -m steering.extract_features \
        --config-name extract_features_gemma3_l22_f65k \
        start_shard=369 num_shards=370
    if [ ! -f "${SHARD_369}" ]; then
        echo "[$(ts)] ABORT: extraction finished but ${SHARD_369} missing"
        exit 1
    fi
fi
echo "[$(ts)] Step 0 complete: ${SHARD_369}"
check_disk

# -----------------------------------------------------------------------------
# Step 0b: validate the full feature directory (all 370 shards) via the
# tested steering.validate_features entrypoint — per-document source
# re-tokenization + char-boundary alignment, per-shard CSR/structural
# invariants, and cross-shard metadata (incl. l0_tier) consistency. This
# subsumes the old "shard 369 vs shard 0" heredoc and is unit-tested
# (unlike embedded bash). validate_features exits non-zero on any violation,
# so `set -euo pipefail` aborts the pipeline on bad data.
echo ""
echo "[$(ts)] Step 0b: validate feature directory (steering.validate_features)"
uv run python -m steering.validate_features \
    --config-name validate_features_gemma3_l22_f65k
echo "[$(ts)] Step 0b complete: feature directory validated"
check_disk

# -----------------------------------------------------------------------------
# Step 2: compute_statistics (Pass 1 CUDA, Pass 2 CPU — OOM-safe for 65k feats).
echo ""
echo "[$(ts)] Step 2: compute_statistics"
if [ -f "${STATS_FILE}" ]; then
    echo "[$(ts)]   ${STATS_FILE} already present — skipping."
else
    uv run python -m steering.compute_statistics \
        --config-name compute_statistics_gemma3_l22_f65k
    [ -f "${STATS_FILE}" ] || { echo "[$(ts)] ABORT: stats file missing after step 2"; exit 1; }
fi
echo "[$(ts)] Step 2 complete."
check_disk

# -----------------------------------------------------------------------------
# Step 3: precompute_percentiles (resumable).
echo ""
echo "[$(ts)] Step 3: precompute_percentiles"
uv run python -m scripts.preprocess.precompute_percentiles \
    --config-name precompute_percentiles_gemma3_l22_f65k
echo "[$(ts)] Step 3 complete."
check_disk

# -----------------------------------------------------------------------------
# Step 4: precompute_token_boundaries (resumable).
echo ""
echo "[$(ts)] Step 4: precompute_token_boundaries"
uv run python -m scripts.preprocess.precompute_token_boundaries \
    --config-name precompute_token_boundaries_gemma3_l22_f65k
echo "[$(ts)] Step 4 complete."
check_disk

# -----------------------------------------------------------------------------
# Step 5: precompute_packed_rows (resumable).
echo ""
echo "[$(ts)] Step 5: precompute_packed_rows"
uv run python -m scripts.preprocess.precompute_packed_rows \
    --config-name precompute_packed_rows_gemma3_l22_f65k
echo "[$(ts)] Step 5 complete."
check_disk

echo ""
echo "================================================================"
echo "[$(ts)] PIPELINE COMPLETE — gemma-3-4b-it l22 f65k ready for training"
echo "  Features:     ${FEATURE_DIR}/  (370 shards: 0-369)"
echo "  Statistics:   ${STATS_FILE}"
echo "  Percentiles:  ${FEATURE_DIR}/pct/<hash>/"
echo "  Token bounds: ${FEATURE_DIR}/tok/<hash>/seg_boundaries/"
echo "  Packed rows:  ${FEATURE_DIR}/pct/<hash>/tok/<hash>/c1024/sall/ (+ val_sall)"
echo "================================================================"
