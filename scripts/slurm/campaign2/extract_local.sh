#!/bin/bash
# RTX 4090: 2 isolated <=32-token segments/forward, 8 documents/CPU chunk.
# Expected to fit 24 GiB with the inherited BF16/offload policy; not GPU-qualified here.
set -euo pipefail
export WORKER_INDEX=0 WORKER_COUNT=1 GPU_BATCH_SEGMENTS=2 CPU_BATCH_DOCS=8
exec bash "${CAMPAIGN_REPO:-$(pwd)}/scripts/slurm/campaign2/extract.sh"
