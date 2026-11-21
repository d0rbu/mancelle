#!/usr/bin/env bash
# Launch multi-GPU sweep with config parallelism.
# Each GPU runs an independent sweep process with striped configs.
#
# Usage:
#   bash scripts/launch_sweep.sh NUM_GPUS [sweep args...]
#
# Examples:
#   bash scripts/launch_sweep.sh 2 --all --output-dir out/sweep_h200
#   bash scripts/launch_sweep.sh 2 --all --resume --output-dir out/sweep_h200
#   bash scripts/launch_sweep.sh 2 --all --max-idx 20 --output-dir out/sweep_h200
set -euo pipefail

NUM_GPUS="${1:?Usage: $0 NUM_GPUS [sweep args...]}"
shift

echo "Launching sweep on $NUM_GPUS GPUs..."
PIDS=()
for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  Starting rank $RANK on GPU $RANK"
    uv run python -m scripts.train.run_encdec_sweep_fast \
        --gpu "$RANK" \
        --sweep-rank "$RANK" \
        --sweep-world-size "$NUM_GPUS" \
        "$@" \
        > >(sed "s/^/[GPU$RANK] /") 2>&1 &
    PIDS+=($!)
done

echo "All $NUM_GPUS processes launched. PIDs: ${PIDS[*]}"
echo "Waiting for all processes to finish..."

# Wait for all, track exit codes
EXIT_CODE=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "GPU $i (PID ${PIDS[$i]}) failed with exit code $?"
        EXIT_CODE=1
    fi
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "All sweep processes completed successfully."
else
    echo "Some sweep processes failed." >&2
fi
exit $EXIT_CODE
