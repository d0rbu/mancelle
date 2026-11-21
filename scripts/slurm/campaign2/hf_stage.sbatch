#!/bin/bash
#SBATCH --job-name=c2-pythia-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/campaign2/env.sh"
mkdir -p "$CAMPAIGN_ROOT/pythia"
exec 9>"$CAMPAIGN_ROOT/pythia/$MIXTURE.lock"
flock -n 9 || die 'another Pythia launcher owns this stage'
export OUTPUT_DIR=$CAMPAIGN_ROOT/pythia/$MIXTURE
[[ ! -f $CAMPAIGN_ROOT/pythia/$MIXTURE-final-export ]] || die 'stage already completed'
if [[ $MIXTURE == safety ]]; then
    export HF_PARENT_EXPORT=${PYTHIA_INSTRUCT_EXPORT:?pin the masked instruct step_XXXXXX export}
else
    export HF_PARENT_EXPORT
    HF_PARENT_EXPORT=$(cat "$CAMPAIGN_ROOT/pythia/safety-final-export")
fi
export C2_DATA_MANIFEST=$CAMPAIGN_ROOT/manifests/$MIXTURE-data.json
export DATA_MANIFEST=$CAMPAIGN_ROOT/manifests/$MIXTURE-stage.json
mode=build
[[ ! -e $DATA_MANIFEST ]] || mode=verify
expected_parent_mixture=standard_steered
[[ $MIXTURE == safety ]] || expected_parent_mixture=safety
"$PY" -m scripts.utils.build_hf_sft_data_manifest "mode=$mode" "mixture=$MIXTURE" \
    sft_loss_mask=completion_only "parent_export=$HF_PARENT_EXPORT" \
    "expected_parent_mixture=$expected_parent_mixture" \
    "data_manifest=$C2_DATA_MANIFEST" "output_path=$DATA_MANIFEST"
export HF_PARENT_REVISION NUM_ITERATIONS DATA_MANIFEST_SHA256
HF_PARENT_REVISION=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign2_parent"]["source_revision"])' "$DATA_MANIFEST")
NUM_ITERATIONS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["epoch"]["num_iterations"])' "$DATA_MANIFEST")
DATA_MANIFEST_SHA256=$(sha256sum "$DATA_MANIFEST" | awk '{print $1}')
export CONFIG_NAME=chat_sft_pythia_2_8b_$MIXTURE SFT_LOSS_MASK=completion_only
unset STEERED_JSONL
# Restart is explicit: supply a complete restart marker's step. A model-only
# export is a parent for the next stage, never a restart for this one.
bash "$REPO/scripts/slurm/hf_decoder_sft_h200_1x4.sbatch"
printf -v final '%s/step_%06d' "$OUTPUT_DIR" "$NUM_ITERATIONS"
[[ -s $final/sft_metadata.json ]] || die 'trainer exited without its final validated HF export'
printf '%s\n' "$final" > "$CAMPAIGN_ROOT/pythia/$MIXTURE-final-export.tmp"
mv "$CAMPAIGN_ROOT/pythia/$MIXTURE-final-export.tmp" "$CAMPAIGN_ROOT/pythia/$MIXTURE-final-export"
