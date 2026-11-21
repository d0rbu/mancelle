#!/bin/bash
#SBATCH --job-name=c2-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
export WORKER_INDEX=${SLURM_ARRAY_TASK_ID:?submit as array 0-3} WORKER_COUNT=4
export GPU_BATCH_SEGMENTS=248 CPU_BATCH_DOCS=128
exec bash "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/campaign2/extract.sh"
