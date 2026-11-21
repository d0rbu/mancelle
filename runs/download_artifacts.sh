#!/bin/bash
# Pull artifact categories from the project's HF bucket.
# Non-destructive on the local filesystem.
#
# Usage:
#   bash runs/download_artifacts.sh [FLAGS] [CATEGORY...]
#
# Categories (from configs/artifacts/default.yaml):
#   base_checkpoints, sft_checkpoints, encdec_checkpoints, features
#
# Default selection (when no categories are given): everything EXCEPT
# `features`. Feature shards are tens-to-hundreds of GB and are opt-in;
# pass `features` explicitly or `--all` to include them.
#
# Flags:
#   --all       Pull every category (including features)
#   --dry-run   Preview only — don't actually transfer (DRY_RUN=true env var also works)
#   --list      Print the resolved category list and exit (no sync)
#   -h, --help  Show this help
#
# Examples:
#   bash runs/download_artifacts.sh                              # default set
#   bash runs/download_artifacts.sh features                     # only features
#   bash runs/download_artifacts.sh --all --dry-run              # everything, preview
#   bash runs/download_artifacts.sh base_checkpoints features    # specific subset

set -euo pipefail

ALL_CATEGORIES=(base_checkpoints sft_checkpoints encdec_checkpoints features)
# `features` opt-in: shards are huge.
DEFAULT_CATEGORIES=(base_checkpoints sft_checkpoints encdec_checkpoints)

usage() {
    cat <<'EOF'
Usage:
  bash runs/download_artifacts.sh [FLAGS] [CATEGORY...]

Pull artifact categories from the project's HF bucket.
Non-destructive on the local filesystem.

Categories (from configs/artifacts/default.yaml):
  base_checkpoints, sft_checkpoints, encdec_checkpoints, features

Default selection (no categories given): everything EXCEPT `features`.
Feature shards are tens-to-hundreds of GB and are opt-in; pass `features`
explicitly or --all to include them.

Flags:
  --all       Pull every category (including features)
  --dry-run   Preview only — don't actually transfer (DRY_RUN=true env var also works)
  --list      Print the resolved category list and exit (no sync)
  -h, --help  Show this help

Examples:
  bash runs/download_artifacts.sh                              # default set
  bash runs/download_artifacts.sh features                     # only features
  bash runs/download_artifacts.sh --all --dry-run              # everything, preview
  bash runs/download_artifacts.sh base_checkpoints features    # specific subset
EOF
}

is_known_category() {
    local needle=$1
    for known in "${ALL_CATEGORIES[@]}"; do
        if [ "$needle" = "$known" ]; then return 0; fi
    done
    return 1
}

# Backward compat: DRY_RUN=true env var preserved as the initial default.
DRY_RUN_FLAG="dry_run=${DRY_RUN:-false}"
LIST_ONLY=false
USE_ALL=false
SELECTED=()

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN_FLAG="dry_run=true" ;;
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
    echo "Will pull: ${CATEGORIES[*]}"
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

for cat in "${CATEGORIES[@]}"; do
    echo "==> Pulling $cat"
    uv run python -m scripts.utils.hf_pull category="$cat" "$DRY_RUN_FLAG"
done

echo ""
echo "==> Done. Artifacts now under \$NANOCHAT_BASE_DIR (checkpoints) and out/data/features (features)."
