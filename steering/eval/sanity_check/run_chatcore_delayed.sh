#!/bin/bash
# Wait for compute_statistics to finish, then run ChatCORE sanity check.
# Usage: bash steering/eval/sanity_check/run_chatcore_delayed.sh

set -euo pipefail

echo "[$(date)] Waiting at least 1 hour before starting..."
sleep 3600

echo "[$(date)] Checking for compute_statistics processes..."
while pgrep -f compute_statistics > /dev/null 2>&1; do
    echo "[$(date)] compute_statistics still running, waiting 60s..."
    sleep 60
done

echo "[$(date)] No compute_statistics running. Starting ChatCORE sanity check..."
uv run python -m steering.eval.sanity_check.run_all --chatcore-only 2>&1
echo "[$(date)] ChatCORE sanity check complete."
