#!/bin/bash
# Pass successful stats job IDs and (optionally) the still-running instruct job.
set -euo pipefail
export MIXTURE=safety
source "${CAMPAIGN_REPO:-$(pwd)}/scripts/slurm/campaign2/env.sh"
pool_args "${1:?pool required}"
deps=${SAFETY_STATS_JOB:?set SAFETY_STATS_JOB}
[[ -z ${INSTRUCT_JOB:-} ]] || deps=$deps:$INSTRUCT_JOB
safety=$(sbatch --parsable "${POOL_ARGS[@]}" --export=ALL --dependency="afterok:$deps" \
    --output="$CAMPAIGN_ROOT/logs/encdec-safety-%j.out" "$REPO/scripts/slurm/campaign2/sft_stage.sbatch")
safety=${safety%%;*}
export MIXTURE=capabilities
sbatch --parsable "${POOL_ARGS[@]}" --export=ALL --dependency="afterok:$safety:${CAPABILITIES_STATS_JOB:?set CAPABILITIES_STATS_JOB}" \
    --output="$CAMPAIGN_ROOT/logs/encdec-capabilities-%j.out" "$REPO/scripts/slurm/campaign2/sft_stage.sbatch"
