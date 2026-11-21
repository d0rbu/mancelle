#!/bin/bash
#SBATCH --job-name=encdec-l22-profile
#SBATCH --output=experiments/logs/encdec_l22_profile_%j.out
#SBATCH --error=experiments/logs/encdec_l22_profile_%j.err
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --time-min=00:03:00
#SBATCH --requeue

# Measured counterpart to the analytical topology mode. Submit with explicit
# --nodes/--gpus-per-node and a matching GPUS_PER_NODE export so the same script
# covers the 1x2, 1x4, 2x1, and 2x2 comparison without separate launchers.
# PROFILE_TRIALS repeats the complete fresh-process measurement on one fixed
# allocation; OPTIMIZER_DIAGNOSTIC_STEPS adds separately labeled synchronized
# group attribution after each ordinary throughput trial.
# COMPONENT_DIAGNOSTIC_STEPS adds eager CUDA module-hook attribution plus an
# uninstrumented eager control; it never changes the compiled throughput path.
# PROFILE_*_STEPS and PROFILE_DEVICE_BATCH_SIZE make short scale probes possible
# without another launcher.
# PROFILE_CONFIG_NAME selects a target preset, and colon-separated
# PROFILE_FP8_VARIANTS (for example, false:true) runs a paired alternating A/B.

set -euo pipefail

repo_root="${PROFILE_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${repo_root}"
mkdir -p experiments/logs

gpus_per_node="${GPUS_PER_NODE:?export GPUS_PER_NODE to match --gpus-per-node}"
profile_trials="${PROFILE_TRIALS:-1}"
optimizer_diagnostic_steps="${OPTIMIZER_DIAGNOSTIC_STEPS:-0}"
component_diagnostic_steps="${COMPONENT_DIAGNOSTIC_STEPS:-0}"
component_warmup_steps="${COMPONENT_WARMUP_STEPS:-1}"
profile_static_max_seg_size="${PROFILE_STATIC_MAX_SEG_SIZE:-1024}"
profile_warmup_steps="${PROFILE_WARMUP_STEPS:-2}"
profile_steps="${PROFILE_STEPS:-10}"
profile_validation_warmup_steps="${PROFILE_VALIDATION_WARMUP_STEPS:-1}"
profile_validation_steps="${PROFILE_VALIDATION_STEPS:-0}"
profile_device_batch_size="${PROFILE_DEVICE_BATCH_SIZE:-16}"
profile_total_batch_size="${PROFILE_TOTAL_BATCH_SIZE:-524288}"
profile_config_name="${PROFILE_CONFIG_NAME:-encdec_l22_training}"
profile_fp8_variants="${PROFILE_FP8_VARIANTS:-${PROFILE_FP8:-false}}"
if ! [[ "${profile_trials}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_TRIALS must be a positive integer, got ${profile_trials}" >&2
    exit 2
fi
if ! [[ "${optimizer_diagnostic_steps}" =~ ^[0-9]+$ ]]; then
    echo "OPTIMIZER_DIAGNOSTIC_STEPS must be a non-negative integer, got ${optimizer_diagnostic_steps}" >&2
    exit 2
fi
if ! [[ "${component_diagnostic_steps}" =~ ^[0-9]+$ ]]; then
    echo "COMPONENT_DIAGNOSTIC_STEPS must be a non-negative integer, got ${component_diagnostic_steps}" >&2
    exit 2
fi
if ! [[ "${component_warmup_steps}" =~ ^[0-9]+$ ]]; then
    echo "COMPONENT_WARMUP_STEPS must be a non-negative integer, got ${component_warmup_steps}" >&2
    exit 2
fi
if ! [[ "${profile_static_max_seg_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_STATIC_MAX_SEG_SIZE must be a positive integer, got ${profile_static_max_seg_size}" >&2
    exit 2
fi
if ! [[ "${profile_warmup_steps}" =~ ^[0-9]+$ ]]; then
    echo "PROFILE_WARMUP_STEPS must be a non-negative integer, got ${profile_warmup_steps}" >&2
    exit 2
fi
if ! [[ "${profile_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_STEPS must be a positive integer, got ${profile_steps}" >&2
    exit 2
fi
if ! [[ "${profile_validation_warmup_steps}" =~ ^[0-9]+$ ]]; then
    echo "PROFILE_VALIDATION_WARMUP_STEPS must be a non-negative integer, got ${profile_validation_warmup_steps}" >&2
    exit 2
fi
if ! [[ "${profile_validation_steps}" =~ ^[0-9]+$ ]]; then
    echo "PROFILE_VALIDATION_STEPS must be a non-negative integer, got ${profile_validation_steps}" >&2
    exit 2
fi
if ! [[ "${profile_device_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_DEVICE_BATCH_SIZE must be a positive integer, got ${profile_device_batch_size}" >&2
    exit 2
fi
if ! [[ "${profile_total_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_TOTAL_BATCH_SIZE must be a positive integer, got ${profile_total_batch_size}" >&2
    exit 2
fi
IFS=':' read -r -a fp8_variants <<< "${profile_fp8_variants}"
if (( ${#fp8_variants[@]} == 0 )); then
    echo "PROFILE_FP8_VARIANTS must contain true or false" >&2
    exit 2
fi
for fp8_variant in "${fp8_variants[@]}"; do
    if [[ "${fp8_variant}" != "true" && "${fp8_variant}" != "false" ]]; then
        echo "PROFILE_FP8_VARIANTS entries must be true or false, got ${fp8_variant}" >&2
        exit 2
    fi
done
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
base_master_port="${MASTER_PORT:-$((30500 + SLURM_JOB_ID % 1000))}"

export MASTER_ADDR="${master_addr}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

echo "label=${PROFILE_LABEL:-unset} nodes=${nodes[*]} gpus_per_node=${gpus_per_node}"
echo "MASTER_ADDR=${MASTER_ADDR} BASE_MASTER_PORT=${base_master_port} config_name=${profile_config_name} fp8_variants=${profile_fp8_variants} trials=${profile_trials} optimizer_diagnostic_steps=${optimizer_diagnostic_steps} component_diagnostic_steps=${component_diagnostic_steps} component_warmup_steps=${component_warmup_steps} semantic_max_seg_size=${profile_static_max_seg_size} warmup_steps=${profile_warmup_steps} steps=${profile_steps} validation_warmup_steps=${profile_validation_warmup_steps} validation_steps=${profile_validation_steps} device_batch_size=${profile_device_batch_size} total_batch_size=${profile_total_batch_size}"
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
nvidia-smi topo -m

run_index=0
for ((trial = 1; trial <= profile_trials; trial++)); do
    variant_order=("${fp8_variants[@]}")
    if (( trial % 2 == 0 )); then
        variant_order=()
        for ((i = ${#fp8_variants[@]} - 1; i >= 0; i--)); do
            variant_order+=("${fp8_variants[i]}")
        done
    fi
    for fp8_variant in "${variant_order[@]}"; do
        export PROFILE_FP8_CURRENT="${fp8_variant}"
        export PROFILE_TRIAL_INDEX="${trial}_${fp8_variant}"
        export MASTER_PORT="$((base_master_port + run_index))"
        export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID}_trial${trial}_fp8${fp8_variant}_${SLURM_PROCID:-batch}"
        export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}_trial${trial}_fp8${fp8_variant}_${SLURM_PROCID:-batch}"
        echo "trial=${trial} fp8=${fp8_variant} MASTER_PORT=${MASTER_PORT}"
        srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
        bash -c '
            exec "${TORCHRUN_BIN:-.venv/bin/torchrun}" \
                --nnodes="${SLURM_NNODES}" \
                --nproc_per_node="${GPUS_PER_NODE}" \
                --node_rank="${SLURM_PROCID}" \
                --master_addr="${MASTER_ADDR}" \
                --master_port="${MASTER_PORT}" \
                -m scripts.benchmark.profile_encdec_l22_training \
                --config-name="${PROFILE_CONFIG_NAME:-encdec_l22_training}" \
                mode=distributed \
                distributed.trial_id="${PROFILE_TRIAL_INDEX}" \
                distributed.optimizer_diagnostic_steps="${OPTIMIZER_DIAGNOSTIC_STEPS:-0}" \
                distributed.component_diagnostic_steps="${COMPONENT_DIAGNOSTIC_STEPS:-0}" \
                distributed.component_warmup_steps="${COMPONENT_WARMUP_STEPS:-1}" \
                distributed.static_max_seg_size="${PROFILE_STATIC_MAX_SEG_SIZE:-1024}" \
                distributed.warmup_steps="${PROFILE_WARMUP_STEPS:-2}" \
                distributed.steps="${PROFILE_STEPS:-10}" \
                distributed.validation_warmup_steps="${PROFILE_VALIDATION_WARMUP_STEPS:-1}" \
                distributed.validation_steps="${PROFILE_VALIDATION_STEPS:-0}" \
                distributed.device_batch_size="${PROFILE_DEVICE_BATCH_SIZE:-16}" \
                distributed.total_batch_size="${PROFILE_TOTAL_BATCH_SIZE:-524288}" \
                distributed.fp8="${PROFILE_FP8_CURRENT}"
        '
        run_index=$((run_index + 1))
    done
done
