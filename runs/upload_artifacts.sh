#!/bin/bash
# Push artifact categories up to the project's HF bucket.
# DESTRUCTIVE on the bucket if you pass `--delete`. Read carefully.
#
# Usage:
#   bash runs/upload_artifacts.sh [FLAGS] [CATEGORY...]
#
# Categories (from configs/artifacts/default.yaml):
#   base_checkpoints, sft_checkpoints, encdec_checkpoints, features
#
# Default selection (when no categories are given): everything EXCEPT
# `features`. Feature shards are tens-to-hundreds of GB and are opt-in;
# pass `features` explicitly or `--all` to include them.
#
# Flags:
#   --all       Push every category (including features)
#   --dry-run   Preview only — don't actually transfer
#   --delete    Also remove bucket files that don't exist locally (destructive)
#   --list      Print the resolved category list and exit (no sync)
#   -h, --help  Show this help
#
# Examples:
#   bash runs/upload_artifacts.sh                              # default set
#   bash runs/upload_artifacts.sh features                     # only features
#   bash runs/upload_artifacts.sh --all --dry-run              # everything, preview
#   bash runs/upload_artifacts.sh base_checkpoints features    # specific subset

set -euo pipefail

ALL_CATEGORIES=(base_checkpoints sft_checkpoints encdec_checkpoints features)
# `features` opt-in: shards are huge.
DEFAULT_CATEGORIES=(base_checkpoints sft_checkpoints encdec_checkpoints)

usage() {
    cat <<'EOF'
Usage:
  bash runs/upload_artifacts.sh [FLAGS] [CATEGORY...]

Push artifact categories up to the project's HF bucket.
DESTRUCTIVE on the bucket if you pass --delete. Read carefully.

Categories (from configs/artifacts/default.yaml):
  base_checkpoints, sft_checkpoints, encdec_checkpoints, features

Default selection (no categories given): everything EXCEPT `features`.
Feature shards are tens-to-hundreds of GB and are opt-in; pass `features`
explicitly or --all to include them.

Flags:
  --all       Push every category (including features)
  --dry-run   Preview only — don't actually transfer
  --delete    Also remove bucket files that don't exist locally (destructive)
  --list      Print the resolved category list and exit (no sync)
  -h, --help  Show this help

Examples:
  bash runs/upload_artifacts.sh                              # default set
  bash runs/upload_artifacts.sh features                     # only features
  bash runs/upload_artifacts.sh --all --dry-run              # everything, preview
  bash runs/upload_artifacts.sh base_checkpoints features    # specific subset
EOF
}

is_known_category() {
    local needle=$1
    for known in "${ALL_CATEGORIES[@]}"; do
        if [ "$needle" = "$known" ]; then return 0; fi
    done
    return 1
}

DRY_RUN_FLAG="dry_run=false"
DELETE_FLAG="delete=false"
LIST_ONLY=false
USE_ALL=false
SELECTED=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN_FLAG="dry_run=true" ;;
        --delete)   DELETE_FLAG="delete=true" ;;
        --all)      USE_ALL=true ;;
        --list)     LIST_ONLY=true ;;
        -h|--help)  usage; exit 0 ;;
        --*)
            echo "Unknown flag: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
        *)
            if ! is_known_category "$arg"; then
                echo "Unknown category: $arg" >&2
                echo "Known categories: ${ALL_CATEGORIES[*]}" >&2
                exit 1
            fi
            SELECTED+=("$arg")
            ;;
    esac
done

if [ "$USE_ALL" = true ]; then
    CATEGORIES=("${ALL_CATEGORIES[@]}")
elif [ ${#SELECTED[@]} -gt 0 ]; then
    CATEGORIES=("${SELECTED[@]}")
else
    CATEGORIES=("${DEFAULT_CATEGORIES[@]}")
fi

if [ "$LIST_ONLY" = true ]; then
    echo "Will push: ${CATEGORIES[*]}"
    exit 0
fi

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Match nanochat.common.get_base_dir(): falls back to <repo_root>/nanochat_cache.
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(git rev-parse --show-toplevel)/nanochat_cache}"

if [ "$DELETE_FLAG" = "delete=true" ]; then
    echo "WARNING: --delete will remove bucket files that don't exist locally."
    echo "         Sleeping 5s — Ctrl-C to abort."
    sleep 5
fi

for cat in "${CATEGORIES[@]}"; do
    echo "==> Pushing $cat"
    uv run python -m scripts.utils.hf_push category="$cat" "$DRY_RUN_FLAG" "$DELETE_FLAG"
done
