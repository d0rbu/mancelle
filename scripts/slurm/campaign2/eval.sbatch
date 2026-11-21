#!/bin/bash
#SBATCH --job-name=c2-harmfulness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
# This single job accepts mode=generate (default callers) or mode=judge.
# Override Slurm resources at submission for CPU-only judging.
exec bash "${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/campaign2/eval_local.sh" "$@"
