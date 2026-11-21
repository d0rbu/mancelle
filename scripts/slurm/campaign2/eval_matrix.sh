#!/bin/bash
# EVAL_POOL=local|priority|preemptable, EVAL_VARIANT=<config variant>.
# Supply three checkpoint dirs, plus encoder steps/statistics and safety targets/caches.
# Safety runs default to all three strengths. No implicit judging.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/guard.sh"
case ${EVAL_POOL:?set EVAL_POOL} in
    local) pool=();;
    priority|preemptable) pool_args "$EVAL_POOL"; pool=("${POOL_ARGS[@]}");;
    *) echo 'invalid EVAL_POOL' >&2; exit 2;;
esac
case ${EVAL_VARIANT:?set EVAL_VARIANT} in encdec|encdec_safety|pythia|pythia_diffmean|pythia_prompt) ;;
    *) echo 'invalid EVAL_VARIANT' >&2; exit 2;; esac
strengths=(unsteered)
if [[ $EVAL_VARIANT == encdec_safety ]]; then
    read -r -a strengths <<< "${EVAL_SAFETY_STRENGTHS-0 3 6}"
    [[ ${#strengths[@]} -gt 0 ]] || { echo 'EVAL_SAFETY_STRENGTHS must be nonempty' >&2; exit 2; }
    strength_result=$(mktemp "${TMPDIR:-/tmp}/campaign2-strengths.XXXXXX")
    trap 'rm -f -- "$strength_result"' EXIT
    CUDA_VISIBLE_DEVICES='' "$PY" -c '
import sys
from pathlib import Path
from steering.eval.campaign2 import strength_leaf
output = Path(sys.argv[1])
values = [float(value) for value in sys.argv[2:]]
leaves = [strength_leaf(value) for value in values]
if len(leaves) != len(set(leaves)):
    raise ValueError("EVAL_SAFETY_STRENGTHS must contain distinct strengths")
lines = ["campaign2_strengths_v1", str(len(values))]
lines.extend(format(value, ".17g") for value in values)
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
' "$strength_result" "${strengths[@]}"
    mapfile -t strength_payload < "$strength_result"
    if [[ ${#strength_payload[@]} -lt 2 || ${strength_payload[0]} != campaign2_strengths_v1 ||
          ! ${strength_payload[1]} =~ ^[0-9]+$ ||
          ${#strength_payload[@]} -ne $((strength_payload[1] + 2)) ]]; then
        echo 'malformed Campaign 2 strength result' >&2
        exit 2
    fi
    strengths=("${strength_payload[@]:2}")
    for strength in "${strengths[@]}"; do
        [[ $strength =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
            echo 'malformed Campaign 2 strength value' >&2
            exit 2
        }
    done
    rm -f -- "$strength_result"
    trap - EXIT
fi
# Validate the complete matrix before submitting its first job.
for stage in INSTRUCT SAFETY CAPABILITIES; do
    key=EVAL_${stage}_CHECKPOINT
    [[ -d ${!key:?set all three EVAL_*_CHECKPOINT paths} ]] || exit 2
    if [[ $EVAL_VARIANT == encdec* ]]; then
        key=EVAL_${stage}_STEP; [[ ${!key:?set all EVAL_*_STEP values} =~ ^[0-9]+$ ]] || exit 2
        key=EVAL_${stage}_STATISTICS; [[ -s ${!key:?set all EVAL_*_STATISTICS paths} ]] || exit 2
    fi
    if [[ $EVAL_VARIANT == encdec_safety ]]; then
        key=EVAL_${stage}_TARGETS; [[ -s ${!key:?set all EVAL_*_TARGETS paths} ]] || exit 2
        key=EVAL_${stage}_PROMPT_CACHE; [[ -d ${!key:?set all EVAL_*_PROMPT_CACHE paths} ]] || exit 2
    fi
done
mkdir -p "${EVAL_OUTPUT_ROOT:?set EVAL_OUTPUT_ROOT}/logs"
# Build identical arguments for preflight and launch. Matrix-owned fields come
# after extra Hydra overrides for safety, so those overrides cannot collapse its cells.
cell_args() {
    local stage=$1 strength=$2 key
    shift 2
    key=EVAL_${stage}_CHECKPOINT
    args=("stage=${stage,,}" "variant=$EVAL_VARIANT" "checkpoint_dir=${!key}"
          "output_dir=$EVAL_OUTPUT_ROOT/${stage,,}/$EVAL_VARIANT" mode=generate)
    if [[ $EVAL_VARIANT == encdec* ]]; then
        key=EVAL_${stage}_STEP; args+=("step=${!key}")
        key=EVAL_${stage}_STATISTICS; args+=("statistics=${!key}")
    fi
    if [[ $EVAL_VARIANT == encdec_safety ]]; then
        args=("$@" "${args[@]}")
        key=EVAL_${stage}_TARGETS; args+=("safety.targets=${!key}")
        key=EVAL_${stage}_PROMPT_CACHE; args+=("conditioning.context_cache_dir=${!key}")
        args+=("safety.strength=$strength")
    else
        args+=("$@")
    fi
}
for stage in INSTRUCT SAFETY CAPABILITIES; do
    for strength in "${strengths[@]}"; do
        cell_args "$stage" "$strength" "$@"
        CUDA_VISIBLE_DEVICES='' "$PY" -m scripts.eval.campaign2 "${args[@]}" mode=preflight
    done
done
for stage in INSTRUCT SAFETY CAPABILITIES; do
    for strength in "${strengths[@]}"; do
        cell_args "$stage" "$strength" "$@"
        if [[ $EVAL_POOL == local ]]; then
            bash "$REPO/scripts/slurm/campaign2/eval_local.sh" "${args[@]}"
        else
            log_name=${stage,,}-$EVAL_VARIANT
            if [[ $EVAL_VARIANT == encdec_safety ]]; then log_name+=-$strength; fi
            sbatch --parsable "${pool[@]}" --export=ALL \
                --output="$EVAL_OUTPUT_ROOT/logs/$log_name-%j.out" \
                "$REPO/scripts/slurm/campaign2/eval.sbatch" "${args[@]}"
        fi
    done
done
