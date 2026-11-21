#!/bin/bash
set -euo pipefail
export MIXTURE=safety
source "${CAMPAIGN_REPO:-$(pwd)}/scripts/slurm/campaign2/env.sh"
pool_args "${1:?pool required}"
launch=$REPO/scripts/slurm/campaign2
declare -A manifests
for MIXTURE in safety capabilities; do
    export MIXTURE
    job=$(sbatch --parsable --partition=mit_normal --account=mit_amf_advanced_cpu \
        --qos=mit_amf_advanced_cpu --job-name=c2-hf-manifest --cpus-per-task=16 \
        --mem=128G --time=24:00:00 --export=ALL,CAMPAIGN_CPU_TASK=manifest \
        --output="$CAMPAIGN_ROOT/logs/$MIXTURE-manifest-%j.out" "$launch/cpu.sbatch")
    manifests[$MIXTURE]=${job%%;*}
done
export MIXTURE=safety
deps=${manifests[safety]}
[[ -z ${PYTHIA_INSTRUCT_JOB:-} ]] || deps=$deps:$PYTHIA_INSTRUCT_JOB
job=$(sbatch --parsable "${POOL_ARGS[@]}" --export=ALL --dependency="afterok:$deps" \
    --output="$CAMPAIGN_ROOT/logs/pythia-safety-%j.out" "$launch/hf_stage.sbatch")
export MIXTURE=capabilities
sbatch --parsable "${POOL_ARGS[@]}" --export=ALL --dependency="afterok:${job%%;*}:${manifests[capabilities]}" \
    --output="$CAMPAIGN_ROOT/logs/pythia-capabilities-%j.out" "$launch/hf_stage.sbatch"
