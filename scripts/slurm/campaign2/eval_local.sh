#!/bin/bash
# Local runner, also used inside the one-GPU Slurm allocation. Explicit cell.
# Example: bash .../eval_local.sh variant=pythia stage=safety checkpoint_dir=... output_dir=... mode=generate
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/guard.sh"
exec "$PY" -m scripts.eval.campaign2 "$@"
