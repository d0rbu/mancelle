#!/bin/bash
#SBATCH --job-name=c2-encdec-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/campaign2/env.sh"
[[ ${SLURM_GPUS_ON_NODE:-} == 4 && ${SLURM_NNODES:-} == 1 ]] || die 'requires one node and four GPUs'
mkdir -p "$CAMPAIGN_ROOT/encdec/$MIXTURE"
exec 9>"$CAMPAIGN_ROOT/encdec/$MIXTURE/launcher.lock"
flock -n 9 || die 'another launcher owns this stage'
if [[ $MIXTURE == safety ]]; then
    PREVIOUS_DIR=${ENCDEC_INSTRUCT_DIR:?set the masked s32 SFT leaf}
    PREVIOUS_STEP=${ENCDEC_INSTRUCT_STEP:?pin the completed instruct step}
else
    PREVIOUS_DIR=$CAMPAIGN_ROOT/encdec/safety/base/safety_sft_conditioned_checkpoints
    PREVIOUS_STEP=$(cat "$CAMPAIGN_ROOT/encdec/safety/final_step")
fi
"$PY" -m scripts.utils.campaign2 action=base "stage=$MIXTURE" "source=$PREVIOUS_DIR" \
    "step=$PREVIOUS_STEP" "base=$BASE"
[[ ! -f $CAMPAIGN_ROOT/encdec/$MIXTURE/final_step ]] || die 'stage already completed'
[[ -s ${FEATURE_MASK:?export FEATURE_MASK} && -s ${FEATURE_CLUSTERS:?export FEATURE_CLUSTERS} ]] || die 'missing mask/clusters'
"$PY" -m scripts.utils.campaign2 action=inventory "parquet=$TRAIN_PARQUET" "features=$TRAIN_LEAF"
"$PY" -m scripts.utils.campaign2 action=inventory "parquet=$VAL_PARQUET" "features=$VAL_LEAF"
[[ -s $TRAIN_LEAF/feature_statistics.safetensors ]] || die 'missing train statistics'
resume=$("$PY" -m scripts.utils.campaign2 action=resume "leaf=$SFT_LEAF")
source "$REPO/scripts/slurm/campaign2/stage_inputs.sh"
"$TORCHRUN" --standalone --nproc_per_node=4 -m scripts.train.conditional_encdec_sft -- \
    model_tag=campaign2 "model_step=$PREVIOUS_STEP" "base_checkpoint_dir=$BASE" run=dummy \
    "dataset=$MIXTURE" "feature_dir=$TRAIN_LEAF" "sft_parquet_dir=$TRAIN_PARQUET" \
    "val_feature_dir=$VAL_LEAF" "val_sft_parquet_dir=$VAL_PARQUET" \
    max_seq_len=2048 device_batch_size=8 total_batch_size=524288 \
    'feature_selection={distribution:geometric,mean:12.0,std:12.0,min_k:1,max_k:49,mode:top_k_percentile,delta_weights:null}' \
    "feature_mask={mode:whitelist,path:'$FEATURE_MASK'}" "feature_clusters={path:'$FEATURE_CLUSTERS'}" \
    num_buckets=2 segment_size=32 missing_percentile_policy=lowest sft_loss_mask=completion_only \
    num_iterations=-1 eval_every=150 save_every=500 sft_prefetch_batches=2 alignment_probe_samples=8 \
    "resume_from_step=$resume"
final=$("$PY" -m scripts.utils.campaign2 action=resume "leaf=$SFT_LEAF")
(( final >= 0 )) || die 'trainer exited without a complete checkpoint'
printf '%s\n' "$final" > "$CAMPAIGN_ROOT/encdec/$MIXTURE/final_step.tmp"
mv "$CAMPAIGN_ROOT/encdec/$MIXTURE/final_step.tmp" "$CAMPAIGN_ROOT/encdec/$MIXTURE/final_step"
