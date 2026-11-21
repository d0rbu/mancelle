#!/bin/bash
#SBATCH --job-name=sae-extract
#SBATCH --output=experiments/logs/extract_%j.out
#SBATCH --error=experiments/logs/extract_%j.err
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Slurm template for a single-GPU SAE feature extraction run.
#
# The modern entry point is ``steering.extract_features`` driven by Hydra
# (configs/extraction/*.yaml). The legacy argparse form (--shard-dir / --top-k /
# --device) was removed during the Hydra migration; this template now uses
# python -m + key=value Hydra overrides instead.
#
# Override Hydra keys with ``key=value`` (Hydra rejects argparse-style
# ``--key value``). Edit the layer / num_shards overrides below to match your
# experiment, or pick a hardware variant config via --config-name= (e.g.
# extract_features_4xh200, extract_features_2xb200).

source .venv/bin/activate

torchrun --standalone --nproc_per_node=${SLURM_GPUS_PER_NODE:-1} \
    -m steering.extract_features \
    --config-name=extract_features \
    layer=8 \
    num_shards=1 \
    device=cuda
