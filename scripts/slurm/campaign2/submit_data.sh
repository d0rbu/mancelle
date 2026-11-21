#!/bin/bash
# MIXTURE=safety|capabilities bash .../submit_data.sh priority|preemptable
set -euo pipefail
source "${CAMPAIGN_REPO:-$(pwd)}/scripts/slurm/campaign2/env.sh"
pool_args "${1:?pool required}"
launch=$REPO/scripts/slurm/campaign2
convert=$(sbatch --parsable \
    --partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu \
    --job-name=c2-convert --cpus-per-task=16 --mem=128G --time=08:00:00 \
    --export=ALL,CAMPAIGN_CPU_TASK=convert \
    --output="$CAMPAIGN_ROOT/logs/$MIXTURE-convert-%j.out" "$launch/cpu.sbatch")
convert=${convert%%;*}
extract=$(sbatch --parsable "${POOL_ARGS[@]}" --export=ALL --array=0-3%4 --dependency="afterok:$convert" \
    --output="$CAMPAIGN_ROOT/logs/$MIXTURE-extract-%A_%a.out" "$launch/extract.sbatch")
extract=${extract%%;*}
sbatch --parsable --partition=mit_normal --account=mit_amf_advanced_cpu \
    --qos=mit_amf_advanced_cpu --job-name=c2-stats --cpus-per-task=16 \
    --mem=256G --time=12:00:00 --export=ALL,CAMPAIGN_CPU_TASK=stats \
    --dependency="afterok:$extract" \
    --output="$CAMPAIGN_ROOT/logs/$MIXTURE-stats-%j.out" "$launch/cpu.sbatch"
