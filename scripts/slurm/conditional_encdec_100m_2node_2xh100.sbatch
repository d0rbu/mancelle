#!/bin/bash
#SBATCH --job-name=encdec-100m-2node
#SBATCH --output=experiments/logs/encdec_100m_%j.out
#SBATCH --error=experiments/logs/encdec_100m_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

# Two-node, two-H100-per-node launcher for the 100M clustered remote run.
# The cache root (cfg.feature_root) must be on a filesystem shared by both
# nodes; the remote body protocol proves that namespace before spawning its
# owner workers.  Do not add --standalone: rendezvous is cross-node here.

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
if (( ${#nodes[@]} != 2 )); then
    echo "expected exactly two allocated nodes, got ${#nodes[@]}" >&2
    exit 2
fi

master_addr="$(getent ahostsv4 "${nodes[0]}" | awk 'NR == 1 {print $1}')"
if [[ -z "${master_addr}" ]]; then
    echo "could not resolve IPv4 address for master node ${nodes[0]}" >&2
    exit 2
fi
master_port="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"

echo "Launching 4 DDP ranks across ${nodes[*]} (2 H100s/node)"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

srun --ntasks=2 --ntasks-per-node=1 \
    bash -lc '
        exec uv run torchrun \
            --nnodes="${SLURM_NNODES}" \
            --nproc_per_node=2 \
            --node_rank="${SLURM_PROCID}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT}" \
            -m scripts.train.conditional_encdec_train -- \
            config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_100m_2node_2xh100 \
            run="${RUN_NAME:-encdec-100m-2node-${SLURM_JOB_ID}}"
    '
