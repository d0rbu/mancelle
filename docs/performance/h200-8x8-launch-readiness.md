# 4 x 8 / 8 x 8 H200 pretraining launch readiness

This runbook prepares the planned 3B encoder-decoder pretraining job for a
time-bounded reservation on four or eight nodes with eight H200 GPUs per node.
It does not claim measured 32- or 64-GPU throughput. The goal is to enter the
reservation with an immutable environment, quantify both optimizer contracts,
and select a working NCCL transport in minutes instead of allowing a hung first
collective to consume the slot.

## Hard compatibility boundary

The locked Linux environment is PyTorch 2.13.0+cu130, CUDA 13.0, NCCL 2.29.7,
Triton 3.7.1, and flash-attn-4 4.0.0b26. CUDA 13.x requires an NVIDIA 580-series
or newer driver. Obtain the target driver version before the reservation if
possible. Keep the former PyTorch 2.11/CUDA 12.8 lock or image available as a
fallback rather than attempting a network install during the slot.

- [PyTorch 2.13 release](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [CUDA 13 driver requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

Confirm whether the nodes are HGX H200 with an eight-GPU NVSwitch fabric or an
H200 NVL/PCIe system. The topology and useful GPU/HCA binding are not
interchangeable. NVIDIA's HGX reference uses 900 GB/s GPU-to-GPU NVLink and
NVSwitch connectivity.

- [HGX H100/H200 components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)

## Exact planned model contract

The checked-in profiling preset is
`configs/profiling/encdec_l22_training_3b_h200.yaml`. It represents symmetric
27-layer, width-1,664 encoder and decoder stacks, context 2,048,
cross-attention at every decoder layer, compact feature vocabulary V=2,074,
and the compact feature-prediction head. `topology` mode is analytical;
`distributed` mode builds this exact model for a measured synthetic-row run.

```bash
uv run python -m scripts.benchmark.profile_encdec_l22_training \
  --config-name=encdec_l22_training_3b_h200
```

The exact meta-device inventory is:

| Contract | Value |
|---|---:|
| Parameters | 3,020,935,888 |
| Optimizer logical messages | 42 |
| Optimizer physical collectives | 64 |
| World-8 padded / real payload | 11.625 / 11.254 GiB |
| World-32 padded / real payload | 12.698 / 11.254 GiB |
| World-64 padded / real payload | 13.358 / 11.254 GiB |
| World-32 / world-64 padding amplification | 1.128x / 1.187x |
| World-32 / world-64 idealized padding penalty | 8.83 / 14.28 ms/step |

These times are sensitivity results under the preset's ideal hierarchical-ring
assumptions, not throughput predictions. They exclude overlap and the target
cluster's NCCL algorithms. Muon group coalescing substantially reduced the
former padding problem: at world size 64, Muon now contributes 9.902 GiB of
padded payload versus 7.798 GiB of real tensor bytes.

At context 2,048, per-rank batch eight preserves the 524,288-token global batch
with one microbatch at world size 32. Per-rank batch four preserves it at both
world sizes:

| World size | Per-rank batch | Accumulation steps | Global tokens |
|---:|---:|---:|---:|
| 32 | 8 | 1 | 524,288 |
| 32 | 4 | 2 | 524,288 |
| 32 | 2 | 4 | 524,288 |
| 32 | 1 | 8 | 524,288 |
| 64 | 4 | 1 | 524,288 |
| 64 | 2 | 2 | 524,288 |
| 64 | 1 | 4 | 524,288 |

Start the four-node/world-32 H200 memory canary at B=8. Three fresh-process
four-H200 trials measured 103.13 GiB peak allocation and 107.40 GiB peak
reservation on 140.40 GiB devices; the first eager BF16 validation forward
peaked at 33.80 GiB allocated. B=4 is the first memory fallback and B=2 the
second, both preserving the global batch and learning-rate regime. At world
size 64, B=8 is invalid for this fixed global batch, so start at B=4.

The repeated four-H200 full-model FP8 A/B and bounded 32-GPU runtime projection
are recorded in the
[encoder-decoder component profile](encoder-decoder-training-components.md#full-3b-c2048-h200-fp8-ab).
The projection is planning evidence, not a substitute for the allocation-local
world-size-32 canary below.

## Before the reservation

Prepare these without target compute access:

1. Pin the exact git commit, lockfile, training config, feature vocabulary,
   selection identity, and checkpoint destination.
2. Make both the primary CUDA 13 and fallback CUDA 12.8 environments available
   offline, preferably as immutable images. Do not depend on package indexes
   during the reservation.
3. Verify packed/selection family counts, manifests, and expected compressed
   bytes. Ensure the destination cache and checkpoint paths have sufficient
   quota.
4. Ask the operator for the driver, exact H200 SKU, Slurm launcher contract,
   inter-node fabric, NIC count per node, NCCL network plugin, and recommended
   GPU/HCA binding. These facts are useful even without interactive access.
5. Pre-write the world-32 B=8 production command plus B=4 and B=2 fallbacks.
   Only the batch and accumulation geometry should differ.

## Allocation-local storage recovery gate

Before each torchrun attempt in a retrying production allocation, gate the
attempt on the storage paths that attempt will actually consume:

```bash
.venv/bin/python -m scripts.utils.storage_recovery_gate \
  --checkpoint-file "$CHECKPOINT_LEAF/model_${RESUME_STEP}.pt" \
  --selection-dir "$SELECTION_DIR"
```

Use the exact resume checkpoint file, not a wildcard or a different checkpoint
from the same leaf. The driver must itself run inside the Slurm allocation. It
does no metadata operation on either storage path; each probe is dispatched by
`srun --nodes=1 --ntasks=1 --overlap`, so login-node cache and routing cannot
certify the compute-node path.

One probe reads 64 MiB from a random MiB-aligned checkpoint offset, then times
direct-child discovery/validation plus a 64 KiB read from each of the first 20
sorted `shard_000*.zst` selection files.
Health requires both bulk throughput `>= 100 MB/s` and the complete small-file
loop `< 2s`. The whole allocation-local probe has a 60-second timeout followed
by 15 seconds of kill grace. An unhealthy read or runtime I/O failure waits 300
seconds and retries, for at most 3 probes (2 sleeps). The scheduled recovery
wait is therefore capped at 10 minutes; three probes that each consume the full
timeout and kill grace yield a 13-minute-45-second upper bound before negligible
process-launch overhead. If all 3 are unhealthy, the gate prints
`storage_gate_status=exhausted_proceeding` and returns success so the launcher
attempts once rather than idling forever. Missing/malformed paths, missing
allocation/tooling, malformed worker output, and unexpected worker failures
return 2 immediately; the retry launcher must treat that as a launcher
configuration failure rather than start torchrun.

The full3b/world-32 retry launcher is operational and is not checked into this
repository. Wire this command immediately before every torchrun attempt in
that launcher; invoking it once before the retry loop does not satisfy the
recovery contract.

## Allocation-local transport selection

Run the wrapper from inside the full Slurm allocation:

```bash
GPUS_PER_NODE=8 \
TORCHRUN_BIN=.venv/bin/torchrun \
bash scripts/slurm/preflight_nccl_transport.sh
```

The wrapper:

1. records GPU, driver, and topology information;
2. launches all ranks with a 90-second process-group timeout and a 150-second
   outer launcher timeout;
3. verifies a one-element all-reduce before allocating large buffers;
4. checks all-reduce, reduce-scatter, and all-gather correctness and
   rank-critical timings for 1 MiB and 43 MiB per-rank shards;
5. retries once with `NCCL_NET_GDR_LEVEL=LOC` if the default transport fails;
6. writes the selected setting to
   `out/preflight/nccl_<job-id>/transport.env`.

At world size 64, the 43 MiB shard creates a 2.688 GiB stacked buffer,
bracketing the planned model's largest 2.641 GiB padded Muon groups. Source the
result before the canary and production launch:

```bash
source "out/preflight/nccl_${SLURM_JOB_ID}/transport.env"
```

`NCCL_NET_GDR_LEVEL=LOC` explicitly disables GPUDirect RDMA. It is a working
but potentially slower fallback, not a successful GDR result. The preflight is
a transport selector:

```text
default transport passes -----------------> use default and continue
default fails; host-staged passes --------> use GDR0 and continue
both fail --------------------------------> no viable distributed path
```

NCCL exposes separate PXN controls. Test PXN only after selecting a correct
GDR mode, so one short experiment changes one transport property. For a deeper
bandwidth sweep, run NVIDIA `nccl-tests` in the same allocation; the repository
probe is a bounded health gate rather than a replacement.

- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)

## Topology attribution during the slot

Use allocation subsets before the production canary. Keep software and
transport fixed and retain the same physical node union:

1. one node x eight GPUs establishes the NVSwitch control;
2. eight nodes x one GPU isolates the inter-node route without local NIC
   sharing;
3. four nodes x one GPU, followed by two disjoint two-node x two-GPU runs on
   the same four nodes, revisits the Engaging 4x1/2x2 result;
4. four nodes x eight GPUs verifies the actual world-size-32 optimizer path;
5. eight nodes x eight GPUs verifies the actual world-size-64 optimizer path.

Record `NCCL_DEBUG=INFO` with `INIT,NET,GRAPH` for these short probes, including
the chosen HCA, GDR, and PXN paths. Reduce logging for the long training run.
Do not attribute a topology delta unless the compared samples share node/GPU
identity or use enough rotated placements to expose node effects.

## Rare-stall canary

Run 100-200 synthetic or production-shape steps before committing the remaining
reservation. Retain raw per-step forward, backward, optimizer, and wall samples;
medians alone hide the previously observed two approximately 2.4-second tails.
During this canary, collect:

- GPU clocks, power, temperature, throttling reasons, ECC/Xid events, and
  NVLink/PCIe counters at roughly one-second cadence;
- `TORCH_LOGS=recompiles` to expose unexpected Inductor/Dynamo recompilation;
- PyTorch NCCL collective timing, desynchronization diagnostics, monitoring,
  and the flight-recorder dump configured by the wrapper;
- the Slurm node list, CPU/GPU binding, and any colocated-job evidence.

PyTorch documents the `TORCH_NCCL_ENABLE_TIMING`, monitoring, heartbeat,
desynchronization, and trace-buffer controls used by the wrapper:

- [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)

If stalls recur with normal optimizer/NCCL timing, capture an Nsight Systems
trace in a separate diagnostic repetition. Do not enable a high-overhead trace
for the full pretraining run.

## Remaining unverified gates

- Exact 3B forward/backward/optimizer steps have run on four H200s with random
  weights and synthetic c2048 conditioning rows. No matching c2048 checkpoint
  and production packed rows were available for a real-data validation.
- No 32- or 64-rank NCCL result has been measured.
- The analytical topology model does not include overlap, multi-rail striping,
  NCCL protocol selection, or network contention.
- The transport wrapper cannot repair a cluster where both direct and
  host-staged communication fail. Its purpose is to preserve the reservation
  when the slower fallback remains usable and to produce bounded evidence when
  it does not.
