#!/bin/bash
# Source after env.sh. Outputs and checkpoint symlinks remain on shared storage.
# A job-specific directory avoids stale or concurrently modified staged inputs.
[[ -n ${NODE_STAGE_ROOT:-} ]] || die 'export NODE_STAGE_ROOT to writable node-local storage'
[[ $NODE_STAGE_ROOT == /* && -d $NODE_STAGE_ROOT && -w $NODE_STAGE_ROOT ]] || die 'invalid NODE_STAGE_ROOT'
stage_dir=$(mktemp -d "$NODE_STAGE_ROOT/campaign2-${MIXTURE}-${SLURM_JOB_ID:-local}.XXXXXX")
required_kb=$(du -sLk "$TRAIN_LEAF" "$VAL_LEAF" "$TRAIN_PARQUET" "$VAL_PARQUET" | awk '{n+=$1} END {print n}')
available_kb=$(df -Pk "$stage_dir" | awk 'END {print $4}')
(( available_kb > required_kb + 20*1024*1024 )) || die 'insufficient node-local space plus 20 GiB reserve'
# Mirror each input under its own absolute path: the feature-directory parser reads the family layout
# (s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/... leaves) from the path, so a flat copy is rejected
# ("Feature path must have at least 7 components").
for key in TRAIN_LEAF VAL_LEAF TRAIN_PARQUET VAL_PARQUET; do
    [[ ${!key} == /* ]] || die "$key must be an absolute path"
    mkdir -p "$stage_dir${!key}"
    rsync -aL --checksum "${!key}/" "$stage_dir${!key}/"
    printf -v "$key" '%s' "$stage_dir${!key}"
done
echo "Read-only inputs staged at $stage_dir; outputs remain at $SFT_LEAF"
