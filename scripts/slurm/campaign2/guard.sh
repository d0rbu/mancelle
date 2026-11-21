#!/bin/bash
# Shared repository admission, offline environment and GPU pool selection.
set -euo pipefail

die() { echo "$*" >&2; exit 2; }

REPO=${CAMPAIGN_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "$REPO"
[[ $(git rev-parse HEAD) == "${EXPECTED_GIT_REVISION:?pin the campaign code}" ]] || die 'git revision mismatch'
[[ -z $(git status --porcelain) ]] || die 'repository must be clean'
PY=${CAMPAIGN_PYTHON:-$REPO/.venv/bin/python}
[[ -x $PY ]] || die 'stage the campaign Python environment first'
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

pool_args() {
    case "$1" in
        priority) POOL_ARGS=(-p mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu);;
        preemptable) POOL_ARGS=(-p mit_preemptable --requeue);;
        *) die 'pool must be priority or preemptable';;
    esac
}
