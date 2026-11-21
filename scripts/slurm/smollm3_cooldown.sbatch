#!/bin/bash
#SBATCH --job-name=smollm3-cooldown
#SBATCH --output=experiments/logs/smollm3_cooldown_%j.out
#SBATCH --error=experiments/logs/smollm3_cooldown_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

# Submit with the current approved advanced account/QoS pair, for example:
# sbatch --account="$ACCOUNT" --qos="$QOS" scripts/slurm/smollm3_cooldown.sbatch

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

if [[ -z "${PREPARED_DATA_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
    echo "PREPARED_DATA_DIR and OUTPUT_DIR must be explicit shared-storage paths" >&2
    exit 2
fi
if [[ ! -x .venv/bin/torchrun ]]; then
    echo "checked-in environment is missing .venv/bin/torchrun" >&2
    exit 2
fi

gpus_per_node="${GPUS_PER_NODE:-1}"
if ! [[ "${gpus_per_node}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer, got ${gpus_per_node}" >&2
    exit 2
fi

export GPUS_PER_NODE="${gpus_per_node}"
export PREFLIGHT_REPO_ROOT="${repo_root}"
export PREFLIGHT_RESULT_DIR="${OUTPUT_DIR}/nccl_preflight_${SLURM_JOB_ID}"
scripts/slurm/preflight_nccl_transport.sh
source "${PREFLIGHT_RESULT_DIR}/transport.env"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
if (( ${#nodes[@]} != SLURM_NNODES )); then
    echo "expected ${SLURM_NNODES} nodes, got ${#nodes[@]}" >&2
    exit 2
fi
master_addr="$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')"
if [[ -z "${master_addr}" ]]; then
    echo "could not resolve IPv4 address for ${nodes[0]}" >&2
    exit 2
fi

export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${MASTER_PORT:-$((32500 + SLURM_JOB_ID % 1000))}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PREPARED_DATA_DIR
export OUTPUT_DIR
export CANARY_STEPS="${CANARY_STEPS:-null}"

srun --kill-on-bad-exit=1 \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    bash -c '
        exec .venv/bin/torchrun \
            --nnodes="${SLURM_NNODES}" \
            --nproc_per_node="${GPUS_PER_NODE}" \
            --node_rank="${SLURM_PROCID}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            --max_restarts=0 \
            -m scripts.train.smollm3_cooldown -- \
            "data.prepared_dir=${PREPARED_DATA_DIR}" \
            "output_dir=${OUTPUT_DIR}" \
            "canary_steps=${CANARY_STEPS}"
    '
