#!/bin/bash
#SBATCH --job-name=sft3b-feature-head
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=06:00:00
# Run the documented alignment audit before submitting this training job.
# Inputs and the qualified environment are operator-selected and remain stable.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
is_isolated_file() { [[ -f $1 && ! -L $1 && $(stat -c %h "$1") == 1 ]]; }
(( $# == 0 )) || die 'no positional arguments; run the alignment audit separately (see runbook)'
: "${RESIDUAL_HEAD_REPO:?set an isolated checkout of the reviewed implementation}"
: "${RESIDUAL_HEAD_RUN_DIR:?set a new shared-storage run directory}"
: "${RESIDUAL_HEAD_EXPECTED_COMMIT:?pin the reviewed implementation commit}"
: "${RESIDUAL_HEAD_PYTHON:?pin the qualified Python executable}"
[[ $RESIDUAL_HEAD_RUN_DIR == /* && $RESIDUAL_HEAD_REPO == /* ]] || die 'run and repo paths must be absolute'
resume=${RESIDUAL_HEAD_RESUME_STEP:--1}
[[ $resume == -1 || $resume =~ ^[1-9][0-9]*$ ]] || die 'resume step must be -1 or a positive integer'

source_root=/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/claude-sft3b-steered-20260905
export RESIDUAL_HEAD_RUN_DIR
RESIDUAL_HEAD_RUN_DIR=$(realpath -m "$RESIDUAL_HEAD_RUN_DIR")
[[ $RESIDUAL_HEAD_RUN_DIR != / && $RESIDUAL_HEAD_RUN_DIR != "$source_root" && $RESIDUAL_HEAD_RUN_DIR != "$source_root/"* ]] || die 'output must be outside the historical run'
cd "$RESIDUAL_HEAD_REPO"
[[ $(git rev-parse HEAD) == "$RESIDUAL_HEAD_EXPECTED_COMMIT" ]] || die 'checkout commit differs from the reviewed commit'
[[ -z $(git status --porcelain) ]] || die 'checkout must be clean'
mkdir -p "$RESIDUAL_HEAD_RUN_DIR"
exec 9>"$RESIDUAL_HEAD_RUN_DIR/launcher.lock"
flock -n 9 || die 'another launcher owns this run'
[[ ! -e $RESIDUAL_HEAD_RUN_DIR/COMPLETED ]] || die 'run already completed'
case ${RESIDUAL_HEAD_PRESET:-mlp} in
    mlp) preset=conditional_encdec_sft_3b_residual_head; default_depth=2 ;;
    transformer) preset=conditional_encdec_sft_3b_transformer_head; default_depth=4 ;;
    *) die 'RESIDUAL_HEAD_PRESET must be mlp or transformer' ;;
esac
depth=${RESIDUAL_HEAD_LAYERS:-$default_depth}
[[ $depth =~ ^(0|[1-9][0-9]*)$ ]] || die 'RESIDUAL_HEAD_LAYERS must be a nonnegative integer'
export PYTHONPATH=$RESIDUAL_HEAD_REPO
export NANOCHAT_BASE_DIR=$source_root/nanochat_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_ATTN_IMPL=fa4
export NCCL_TIMEOUT_SEC=3600
export RESIDUAL_HEAD_MASK=/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/preselection-config/feature_mask_two_stage_v2_covsel100.txt
export RESIDUAL_HEAD_CLUSTERS=/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/preselection-config/feature_clusters_two_stage_v2_covsel100.json
family=$source_root/features/s32/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt
export RESIDUAL_HEAD_TRAIN_FEATURES=$family/standard_steered_sft/context_independent/no_bos/turn_delimited
export RESIDUAL_HEAD_VAL_FEATURES=$family/standard_sft_val/context_independent/no_bos/turn_delimited
export RESIDUAL_HEAD_TRAIN_PARQUET=$NANOCHAT_BASE_DIR/sft_data/standard_steered
export RESIDUAL_HEAD_VAL_PARQUET=$NANOCHAT_BASE_DIR/sft_data/standard_val
export RESIDUAL_HEAD_STEERED_JSONL=$source_root/data/messages_steered.jsonl

[[ ${SLURM_GPUS_ON_NODE:-} == 4 && ${SLURM_NNODES:-} == 1 ]] || die 'requires one node and four GPUs'
: "${NODE_STAGE_ROOT:?set an absolute node-local staging directory}"
[[ $NODE_STAGE_ROOT == /* ]] || die 'NODE_STAGE_ROOT must be absolute'
mkdir -p "$NODE_STAGE_ROOT"
[[ -d $NODE_STAGE_ROOT && -w $NODE_STAGE_ROOT ]] || die 'invalid NODE_STAGE_ROOT'
base=$RESIDUAL_HEAD_RUN_DIR/base
leaf=$base/standard_steered_sft_conditioned_checkpoints
[[ ! -L $base && ! -L $leaf ]] || die 'base and SFT output leaf must be real isolated directories'
: "${RESIDUAL_HEAD_MIGRATED_BASE:?set an isolated canonical step-250000 checkpoint directory}"
[[ $RESIDUAL_HEAD_MIGRATED_BASE == /* ]] || die 'migrated base must be absolute'
source_base=$(realpath -e "$RESIDUAL_HEAD_MIGRATED_BASE")
[[ $source_base != "$source_root" && $source_base != "$source_root/"* ]] || die 'migrated base must be outside the historical run'
mkdir -p "$base"
if [[ $resume == -1 ]]; then
    [[ ! -e $leaf ]] || die 'fresh run refuses an existing SFT output leaf; choose an explicit resume step'
else
    printf -v resume_padded '%06d' "$resume"
    for path in "$leaf/model_$resume_padded.pt" "$leaf/meta_$resume_padded.json" "$leaf"/optim_"$resume_padded"_rank{0,1,2,3}.pt; do
        [[ -s $path ]] && is_isolated_file "$path" || die "incomplete or non-isolated resume checkpoint: $path"
    done
fi
for name in model_250000.pt meta_250000.json; do
    target=$(realpath -e "$source_base/$name")
    if [[ -e $base/$name || -L $base/$name ]]; then
        is_isolated_file "$base/$name" && cmp -s "$target" "$base/$name" || die "base identity changed or file is not isolated: $name"
    else
        cp --reflink=auto "$target" "$base/$name"
    fi
done
runtime=$RESIDUAL_HEAD_RUN_DIR/runtime/${SLURM_JOB_ID:?}
mkdir -p "$runtime"

# Preserve canonical feature paths; lineage reads the resolved directory layout.
# Every allocation gets a fresh stage. Outputs and base copies stay on shared storage.
job_stage=$(mktemp -d "$NODE_STAGE_ROOT/residual-head-${SLURM_JOB_ID}.XXXXXX")
stage_dir=$job_stage/inputs
export TORCHINDUCTOR_CACHE_DIR=$job_stage/torchinductor CUDA_CACHE_PATH=$job_stage/cuda TMPDIR=$job_stage/tmp
mkdir -p "$stage_dir" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
inputs=(RESIDUAL_HEAD_TRAIN_FEATURES RESIDUAL_HEAD_VAL_FEATURES RESIDUAL_HEAD_TRAIN_PARQUET RESIDUAL_HEAD_VAL_PARQUET)
required_kb=$(du -sLk "$RESIDUAL_HEAD_TRAIN_FEATURES" "$RESIDUAL_HEAD_VAL_FEATURES" "$RESIDUAL_HEAD_TRAIN_PARQUET" "$RESIDUAL_HEAD_VAL_PARQUET" "$RESIDUAL_HEAD_STEERED_JSONL" | awk '{n+=$1} END {print n}')
available_kb=$(df -Pk "$stage_dir" | awk 'END {print $4}')
(( available_kb > required_kb + 20*1024*1024 )) || die 'insufficient node-local space plus 20 GiB reserve'
for key in "${inputs[@]}"; do
    mkdir -p "$stage_dir${!key}"
    rsync -aL "${!key}/" "$stage_dir${!key}/"
    printf -v "$key" '%s' "$stage_dir${!key}"
done
mkdir -p "$stage_dir$(dirname "$RESIDUAL_HEAD_STEERED_JSONL")"
rsync -aL "$RESIDUAL_HEAD_STEERED_JSONL" "$stage_dir$RESIDUAL_HEAD_STEERED_JSONL"
export RESIDUAL_HEAD_STEERED_JSONL=$stage_dir$RESIDUAL_HEAD_STEERED_JSONL

"$RESIDUAL_HEAD_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m scripts.train.conditional_encdec_sft -- "config_name=$preset" \
    "feature_prediction_head_layers=$depth" "resume_from_step=$resume" 2>&1 | tee "$runtime/training.log"
final=$("$RESIDUAL_HEAD_PYTHON" -m scripts.utils.campaign2 action=resume "leaf=$leaf")
(( final > 0 )) || die 'trainer exited without a complete checkpoint'
printf '%s\n' "$final" > "$RESIDUAL_HEAD_RUN_DIR/final_step"
printf '%s\n' "$SLURM_JOB_ID" > "$RESIDUAL_HEAD_RUN_DIR/COMPLETED"
