#!/bin/bash
#SBATCH --job-name=base-pretrain
#SBATCH --output=experiments/logs/train_%j.out
#SBATCH --error=experiments/logs/train_%j.err
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00

# Slurm template for a 4-GPU base-model pretraining run.
#
# The modern entry point is ``scripts.train.base_train`` driven by Hydra
# (configs/training/*.yaml). PR #394 reorganized all top-level training
# scripts under ``scripts/train/``.
#
# Override Hydra keys with ``key=value`` after ``--`` (Hydra rejects the
# argparse-style ``--key value`` form). Edit the depth/run/total_batch_size
# overrides below to match your experiment.

source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 \
    -m scripts.train.base_train -- \
    depth=12 \
    run=slurm-${SLURM_JOB_ID:-default}
