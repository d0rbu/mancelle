#!/bin/bash
# Resource flags are supplied by sbatch callers; this file only dispatches work.
set -euo pipefail
launch=${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/campaign2

case ${CAMPAIGN_CPU_TASK:?set CAMPAIGN_CPU_TASK} in
    convert)
        source "$launch/env.sh"
        "$PY" -m scripts.preprocess.convert_sft_to_parquet "mixture=$MIXTURE" split=both
        for parquet in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
            "$PY" -m scripts.utils.campaign2 action=inventory "parquet=$parquet"
        done
        ;;
    manifest)
        source "$launch/env.sh"
        mkdir -p "$CAMPAIGN_ROOT/manifests"
        mode=build
        [[ ! -f $CAMPAIGN_ROOT/manifests/$MIXTURE-data.json ]] || mode=verify
        "$PY" -m scripts.utils.build_hf_sft_data_manifest "mode=$mode" "mixture=$MIXTURE" \
            steered_jsonl=null sft_loss_mask=completion_only \
            "output_path=$CAMPAIGN_ROOT/manifests/$MIXTURE-data.json"
        ;;
    stats)
        source "$launch/env.sh"
        "$PY" -m scripts.utils.campaign2 action=inventory "parquet=$TRAIN_PARQUET" "features=$TRAIN_LEAF"
        "$PY" -m scripts.utils.campaign2 action=inventory "parquet=$VAL_PARQUET" "features=$VAL_LEAF"
        "$PY" -m steering.compute_statistics --config-name compute_statistics_gemma3_l22_f65k \
            model=gemma-3-4b-pt segment_size=32 "feature_dir=$TRAIN_LEAF" exclude_last_shard=false device=cpu
        "$PY" -m steering.validate_features --config-name validate_features_gemma3_l22_f65k \
            model=gemma-3-4b-pt "feature_dir=$TRAIN_LEAF" "parquet_dir=$TRAIN_PARQUET" n_spot_check=0
        "$PY" -m steering.validate_features --config-name validate_features_gemma3_l22_f65k \
            model=gemma-3-4b-pt "feature_dir=$VAL_LEAF" "parquet_dir=$VAL_PARQUET" n_spot_check=0
        ;;
    eval-view)
        source "$launch/guard.sh"
        export CUDA_VISIBLE_DEVICES=''
        exec "$PY" -m scripts.utils.campaign2_eval_view "$@"
        ;;
    *)
        echo 'CAMPAIGN_CPU_TASK must be convert, manifest, stats, or eval-view' >&2
        exit 2
        ;;
esac
