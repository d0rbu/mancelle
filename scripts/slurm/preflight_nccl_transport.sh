#!/bin/bash
# Run inside an existing Slurm allocation. Probe NCCL defaults first; if that
# fails within the bounded watchdog, retry with host-staged inter-node traffic.

set -uo pipefail

repo_root="${PREFLIGHT_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${repo_root}"

if [[ -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NNODES:-}" ]]; then
    echo "preflight_nccl_transport.sh must run inside a Slurm allocation" >&2
    exit 2
fi
if ! command -v srun >/dev/null 2>&1; then
    echo "srun is unavailable inside the allocation" >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required to bound a stuck launcher" >&2
    exit 2
fi

gpus_per_node="${GPUS_PER_NODE:-8}"
nccl_timeout_sec="${NCCL_TIMEOUT_SEC:-90}"
wall_timeout_sec="${PREFLIGHT_WALL_TIMEOUT_SEC:-150}"
for value_name in gpus_per_node nccl_timeout_sec wall_timeout_sec; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
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

base_master_port="${MASTER_PORT:-$((31500 + SLURM_JOB_ID % 1000))}"
result_dir="${PREFLIGHT_RESULT_DIR:-out/preflight/nccl_${SLURM_JOB_ID}}"
selection_file="${result_dir}/transport.env"
mkdir -p "${result_dir}"
printf '%s\n' '# No transport selected; preflight has not completed.' > "${selection_file}"

export MASTER_ADDR="${master_addr}"
export GPUS_PER_NODE="${gpus_per_node}"
export NCCL_TIMEOUT_SEC="${nccl_timeout_sec}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-60}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-10000}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
export PREFLIGHT_WALL_TIMEOUT_SEC="${wall_timeout_sec}"
export PREFLIGHT_SHARD_MIB="${PREFLIGHT_SHARD_MIB:-[1,43]}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-.venv/bin/torchrun}"

echo "nodes=${nodes[*]} gpus_per_node=${gpus_per_node}"
echo "NCCL_TIMEOUT_SEC=${nccl_timeout_sec} wall_timeout_sec=${wall_timeout_sec}"
if ! srun --kill-on-bad-exit=1 \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    bash -c '
        hostname
        nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv
        nvidia-smi topo -m
    ' 2>&1 | tee "${result_dir}/inventory.log"; then
    echo "failed to inventory every allocated node" >&2
    exit 2
fi

run_probe() {
    local label="$1"
    local master_port="$2"
    local log_file="${result_dir}/${label}.log"
    export MASTER_PORT="${master_port}"
    echo "probe=${label} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    srun --kill-on-bad-exit=1 \
        --nodes="${SLURM_NNODES}" \
        --ntasks="${SLURM_NNODES}" \
        --ntasks-per-node=1 \
        bash -c '
            exec timeout --signal=TERM --kill-after=15s \
                "${PREFLIGHT_WALL_TIMEOUT_SEC}s" \
                "${TORCHRUN_BIN}" \
                --nnodes="${SLURM_NNODES}" \
                --nproc_per_node="${GPUS_PER_NODE}" \
                --node_rank="${SLURM_PROCID}" \
                --master_addr="${MASTER_ADDR}" \
                --master_port="${MASTER_PORT}" \
                -m scripts.benchmark.preflight_nccl_transport \
                "transport.shard_mib=${PREFLIGHT_SHARD_MIB}"
        ' 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
}

unset NCCL_NET_GDR_LEVEL
if run_probe default "${base_master_port}"; then
    printf '%s\n' \
        '# Default NCCL transport passed the bounded preflight.' \
        'unset NCCL_NET_GDR_LEVEL' > "${selection_file}"
    echo "selected_transport=default selection_file=${selection_file}"
    exit 0
fi

echo "default NCCL transport failed; retrying with NCCL_NET_GDR_LEVEL=LOC" >&2
export NCCL_NET_GDR_LEVEL=LOC
if run_probe host_staged "$((base_master_port + 1))"; then
    printf '%s\n' \
        '# Default transport failed; host-staged NCCL passed.' \
        'export NCCL_NET_GDR_LEVEL=LOC' > "${selection_file}"
    echo "selected_transport=host_staged selection_file=${selection_file}"
    exit 0
fi

printf '%s\n' '# Both default and host-staged NCCL preflights failed.' > "${selection_file}"
echo "no viable 64-rank transport; logs=${result_dir}" >&2
exit 1
