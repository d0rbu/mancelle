#!/usr/bin/env bash
# Usage:
#   ./scripts/bench_compare.sh baseline   # Save baseline
#   ./scripts/bench_compare.sh compare    # Compare to baseline
#   ./scripts/bench_compare.sh full       # Run all benchmarks (incl. slow CUDA)
set -euo pipefail

BENCH_DIR=".benchmarks"
TESTS="tests/steering/test_performance.py"

case "${1:-help}" in
    baseline)
        echo "Saving benchmark baseline..."
        uv run pytest "$TESTS" \
            --model-scale=tiny \
            --benchmark-autosave \
            --benchmark-json="${BENCH_DIR}/baseline.json" \
            -k "not CUDA and not Granularity" \
            -v
        echo "Baseline saved to ${BENCH_DIR}/baseline.json"
        ;;
    compare)
        if [ ! -f "${BENCH_DIR}/baseline.json" ]; then
            echo "No baseline found. Run: $0 baseline"
            exit 1
        fi
        echo "Running benchmarks and comparing to baseline..."
        uv run pytest "$TESTS" \
            --model-scale=tiny \
            --benchmark-compare="${BENCH_DIR}/baseline.json" \
            -k "not CUDA and not Granularity" \
            -v
        ;;
    full)
        echo "Running full benchmark suite (including CUDA)..."
        uv run pytest "$TESTS" \
            --benchmark-autosave \
            -v
        ;;
    help|*)
        echo "Usage: $0 {baseline|compare|full}"
        echo "  baseline  - Save current performance as baseline"
        echo "  compare   - Run and compare against saved baseline"
        echo "  full      - Run all benchmarks (including slow CUDA)"
        ;;
esac
