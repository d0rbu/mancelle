# FlashAttention-4 H100 validation

## Outcome

FlashAttention-4 now works through the same compiled wrapper used by training.
The current compatible Linux stack is PyTorch 2.13.0+cu130, flash-attn-4
4.0.0b26, CuTeDSL 4.6.0.dev0, and torchao 0.18.0+cu130. The project lock
installs this combination, and `bash scripts/setup_fa4.sh` verifies a causal
production forward/backward pass, finite gradients, SDPA parity, and that
runtime probing did not silently disable FA4. The original validation below
used the earlier PyTorch 2.11.0+cu128 / flash-attn-4 4.0.0b22 stack; the 2.13
follow-up re-passed the real-H100 forward/backward and SDPA parity gates before
profiling.

On an H100 at the planned 2,048-token context, a matched attention-only
forward/backward A/B improved throughput by 7.76%. On the exact current 412M
encoder-decoder training shape, a matched compiled 2xH100 A/B improved full-step
throughput by 0.51% and MFU by 0.048 percentage points. The full-step result is
smaller because attention is only one part of the encoder, decoder, optimizer,
and distributed step.

These are measured gains, but they are not a throughput projection for the
proposed 3B model. The context-2048 result isolates its attention geometry; the
full-step result uses the current d12/e12, context-1024 checkpoint shape.

## Root causes

There were three independent problems:

1. The old setup script said FA4 was included in PyTorch and needed no external
   dependency. FlexAttention's `BACKEND=FLASH` path actually imports
   `flash_attn.cute`; the project did not install it.
2. PyTorch 2.10 could import the package but was not API-compatible with
   flash-attn-4 4.0.0b22. The production wrapper's probe failed because FA4
   passed a `seqlen_info` keyword to `score_mod`, then correctly disabled FA4
   and fell back to Triton. PyTorch 2.11 is the first stable release with this
   FlexAttention FA4 backend.
3. The old verifier used sequence length and default block-mask dimensions of
   64. Hopper FA4 requires its block-sparse dimensions to be multiples of 128.
   Training already builds explicit 128-wide masks, so the verifier—not the
   production wrapper—was wrong.

The original PyTorch 2.11 upgrade also required torchao 0.17 or newer. The
current PyTorch 2.13 lock resolves torchao 0.18.0+cu130 from the same CUDA 13.0
PyTorch index; using the matching indexed wheel remains required.

FA3 remains unavailable on Engaging. The kernels-hub extension requires
GLIBC 2.32, while the compute-node image exports an older glibc. The exact
loader failure is now surfaced by optional `scripts/setup_fa4.sh --fa3`; FA4 is
the verified training solution and does not require FA3.

References:

- [PyTorch FA4 FlexAttention announcement](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- [PyTorch 2.11 release](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch 2.13 release](https://github.com/pytorch/pytorch/releases/tag/v2.13.0)
- [flash-attn-4 package](https://pypi.org/project/flash-attn-4/)
- [official FlashAttention installation notes](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
- [torchao releases](https://github.com/pytorch/ao/releases)

## Measured context-2048 attention A/B

Slurm job `20542862` ran both backends sequentially on the same H100 80GB HBM3
with PyTorch 2.11.0+cu128. Shapes were B=4, H=13, S=2,048, D=128, matching a
1,664-wide model's attention geometry. Each row is the median of 30 measured
forward/backward iterations after five warmups; compilation is excluded.

| Backend | Median | p95 | Tokens/s | Peak allocated |
|---|---:|---:|---:|---:|
| Triton FlexAttention | 1.0917 ms | 1.1132 ms | 7,503,700 | 1.908 GiB |
| FA4 | 1.0132 ms | 1.0383 ms | 8,085,657 | 1.908 GiB |

FA4 reduced median time by 7.20% and increased throughput by 7.76%. Output
means were -0.000135740 and -0.000135798, respectively. The separate production
smoke job `20544019` ran the final tracked `scripts/setup_fa4.sh`, compared the
complete output tensor to SDPA, and passed forward/backward with finite
gradients.

## Measured full encoder-decoder A/B

Slurm job `20543624` ran each backend sequentially on the same two H100 80GB
HBM3 GPUs with separate compilation caches. It used the exact 412,256,050-
parameter d12/e12 production shape, B=16 per rank, context 1,024, global batch
524,288 tokens, 16 accumulation steps, and the real distributed optimizer.
Rows are medians over ten post-warmup optimizer steps. Synthetic conditioning
rows remove the input pipeline from this comparison.

| Backend | Forward | Backward | Step | Tokens/s | MFU | Peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| Triton FlexAttention | 899.08 ms | 1,714.86 ms | 2,633.51 ms | 199,083 | 9.466% | 50.158 GiB |
| FA4 | 894.58 ms | 1,706.42 ms | 2,620.23 ms | 200,092 | 9.514% | 50.156 GiB |

FA4 reduced median step time by 0.50%, increased throughput by 0.51%, and
increased MFU by 0.048 percentage points. No FA4 compatibility warning or
runtime fallback occurred in the FA4 phase.

The earlier roughly 4-7% MFU placement pattern should not be interpreted as
inter-node communication making training faster. That comparison mixed H100
subtypes, GPU/NIC affinity, and severe rank-0 optimizer skew. The full A/B here
holds placement and software constant; after the PyTorch/NCCL upgrade, rank
optimizer intervals were 19-26 ms instead of the previous multi-second skew.

## Distributed checks and limitations

- A fresh two-GPU NCCL smoke (`20543543`) completed on one node with
  NCCL 2.28.9, P2P/CUMEM, GDR enabled, and correct all-reduce results. A prior
  allocation on node2642 hung in its first NCCL barrier and timed out; that
  failure did not reproduce on node2803.
- The queued 2-node x 2-GPU job from the original profile (`20541425`) eventually
  ran on node1702 and node2905. NCCL established P2P/CUMEM within each node and
  GDR-enabled InfiniBand across nodes, but the three-minute allocation expired
  before the first timing sample. It provides transport evidence, not a
  four-rank throughput number.
- The A/B excludes packed-row decode, remote I/O, H2D staging, validation, and
  checkpointing. Both phases use the same synthetic input, so those omissions
  do not bias the backend comparison.
- The full-step experiment is the current 412M model at context 1,024. The 3B,
  context-2,048 shape still needs a production-scale memory and throughput
  experiment; no result here is silently scaled to it.

## Reproduction and verification

Sync the locked environment and run the production correctness gate on an
SM90 or SM100 GPU:

```bash
uv sync --locked
bash scripts/setup_fa4.sh
```

SM12 GRPO/RLVR workers use the separate
[native CuTe adapter and qualification](fa4-sm12-rlvr-validation.md).

Use `bash scripts/setup_fa4.sh --fa3` only when validating the optional FA3
KV-cache fallback. It is expected to fail clearly on the current Engaging glibc
rather than mutating the project environment or hiding the loader error.
