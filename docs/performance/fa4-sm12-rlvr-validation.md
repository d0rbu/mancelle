# Native FA4 on SM12 for decoder and feature-head attention

SM12 workers install `steering/conditional/encoder_decoder/sm12_fa4.py` for
causal/windowed decoder and transformer-head full-prefix attention. Native CuTe
avoids the unsupported FlexAttention block-mask route. Shared admission runs at
CUDA model construction and materialization (including ordinary checkpoint
loading) as well as early trainer admission. CPU/meta construction does not
require CUDA dependencies; moving the model to CUDA performs admission.
Installation is idempotent and preserves explicit SDPA reference overrides. SM12 skips FlexAttention block-mask
warming; H200 retains it and uses the FA4/CuTe-backed FlexAttention path.
Explicit FA4 requests require the real dependency/backend, rather than the
implementation label alone. Packed encoder/cross-attention and token-cache
kernels retain their dispatch; cached generation uses FA3 on supported Hopper
environments and SDPA otherwise.

The September 18 receipts below qualify the earlier GRPO/RLVR source snapshots.
Transformer-head SFT qualification is recorded separately in the
[feature-head experiment record](residual-feature-head.md).

## Environment and evidence

September 18, 2026 qualification used RTX PRO 6000 Blackwell **Server Edition**
(SM120), PyTorch 2.13.0+cu130, flash-attn-4 4.0.0b26, CuTeDSL 4.6.0.dev0, and
Triton 3.7.1. No dependency upgrade is required. Evidence paths below are relative
to this directory on Engaging unless explicitly labeled local or pod 12:

```text
/orcd/data/dylanhm/001/nano-gpt-pretrain-steer/codex-issue1687-20260918/
```

Frozen checkouts retain their original source throughout checkpoint execs:

| Checkout | Git reference | `self_conditioned_code_identity()` source SHA-256 | Scope |
|---|---|---|---|
| `repo-v3` | `4fbd639661c05704bf466ed3712a1f895f03bae4` | `8e8d2318c67a7efc67d4bf44bbb9282201420b99610ebf6cc56cc780722bdf9f` | Original tasks/FA4 qualification; memory probe uses a sidecar candidate. |
| `repo-v4`, `repo-v5` | `4d89af9a19fd5fe06766443a8b1995471b17a1e8` | `e08f20f168f7aa15c36c100c6ef8b1c4de21f7816e568e1030388bbe0d3f1a2d` | Production history checkpointing; v5 corrects only the test oracle. |
| `repo-v7` | Archive of `e8c52456d044a924fce0dbae1de2c758b8d75b7f` | `0a5c2c09d6d2699046c987290210e7584eecad189fa872775b4caae4fc3c3f39` | Simplified task/FA4 admission and encoder replay. |
| `repo-v8` | Archive of `38b842604fffb009600767b7ca1d507940a8bcf6` | `cbb9ef3795443e82c51f23574baa49a4f4cb633ed0ea9a70ca8aa7a858e9ead4` | GRPO admission covered by the existing cleanup guard. |
| `repo-v9` | Archive of `a1da0ffc66dd457a435296f6b48941038dcff54d` | `80ca7e170ab4d2a3e36b546dd6ce10ec8f62d7a3d2b73f463522ff0fb1452292` | Final production source, rebased onto main `5b99abc88` with residual feature heads. |

The v7/v8 archive snapshots have local Git IDs `e952c0bf4bbc3ef216e4bf5e2117e360c530501a`
and `2e12557d08a5ec6389c5d798e68c79d14618a7df`, respectively. Source fingerprints
match the reviewed files. Their only production difference is the GRPO cleanup
guard's extent; normal-path statements and their order remain identical.
The v9 snapshot Git ID is `bda4d6de99ff25e97c08510b54c26d6abd145c56`.
GPU receipts belong to these frozen snapshots, not automatically to later code.

## Simplification qualification

Rebased onto main `5b99abc88a7309ed0495284c91cc02b1a602f953`, the implementation
adds 97 net production Python lines, down from 114. Simplification removes six
statements, two `if` branches, and one helper, without removing a regression scenario.
Default task handling reuses settings validation and keeps an immutable selection;
history replay chooses its encoder callable once. Native dependency checks remain
at admission, before loading weights, with the existing cleanup guard extended
to cover admission failures. Compiler-context and weight-lifetime checks remain.

The CPU run passes **656 tests**: 314 RLVR/GRPO contracts plus 342 expanded
residual-head/configuration checks. Its 24 Hydra failures reproduce on untouched
main with identical error output; three CUDA-only cases skip and 49 slow cases
are deselected. All 56 changed executable production lines and their branch arcs
are covered; the adapter covers 22/22 statements and 8/8 branches. All 14 targeted
mutations are detected, and the cleanup regression fails before the guard moves
and passes afterward. Exact pre-simplification resume payload/hash comparisons
pass 258 cases across defaults, every ordered task subset, list/tuple inputs,
and one/two ranks, with code identity held fixed for that semantic comparison.
Ruff passes. Ty 0.0.82 reports 388 diagnostics versus 391 on this main, with no
new normalized diagnostics. Local receipts: `out/issue1687/simplification/`;
the latest CPU results and baseline comparisons are in its `rebased-code/` directory.
The simplified source passes all **46 selected CUDA tests** in job `23057631`.
All six post-rebase CUDA checks pass in `23060217`, including residual-head
eager/compiled forward, backward, and strict checkpoint restore.

## Correctness checks

| Gate | Result and receipt |
|---|---|
| Native SM12 kernels | All 35 committed CUDA cases pass in `23057631` (`repo-v7`). The native attention calculation is unchanged by simplification or the final rebase. |
| Compiled history encoder | Both cases pass in `23057631` and `23060217` (`repo-v7/v9`); original/candidate numerical controls are in `23043005`. |
| Real-tokenizer RLVR CLI and NCCL restarts | Seven tests pass in `23057631`: full/MC-only tasks, one/two GPUs, three steps and two execs, complete per-rank checkpoints, saved task list, unchanged LM head. The two-GPU full-task run also passes in `23060217`. |
| Shared GRPO admission | Both existing CLI tests pass in `23057631` and `23058523` (`repo-v7/v8`), including auxiliary NTP off/on, resume mismatch rejection, and decay. The on/off and resume-rejection case also passes in `23060217` (`repo-v9`). |
| Residual-head rebase | Both upstream packed-model CUDA cases pass in `23060217`: eager/compiled exact upgrade parity, learning, and strict restore. These retain their existing SDPA test configuration. |
| Missing native FA4 | Real NCCL initialization and cleanup pass in `23058523` and `23060217`: admission raises the expected error and destroys the group before loading weights. |
| Existing CUDA policy/compiler contracts | Three tests pass in `22998717` using their FP32/SDPA unit-test configuration. |

The independent attention oracle uses FP64 math attention. Cases cover BF16/FP16,
GQA, 13 heads, lengths 1–2048, irregular tiles, batch one/two, and inclusive left
windows 0/31/128/1024/2048 plus unlimited attention. They check forward output,
Q/K/V gradients, mask boundaries, and the decoder's contiguous normalized Q/K
with strided V and gradients into shared QKV projection storage. Gradient-enabled
and detached no-gradient forward outputs must match exactly.

Kernel error bounds are `atol=rtol=0.02` and relative L2 below 1%; mathematically
zero Q/K derivatives allow `2e-6` absolute FP32 reduction roundoff. RLVR's existing
`2e-5` likelihood replay checks are unchanged. Native SM12 backward rejects hard
determinism in this release, so its recorded setting is disabled, as on SM90.
Bitwise training/resume equality is not guaranteed.

CPU tests retain all 64 nonempty ordered task subsets, invalid-input rejection,
exact legacy sampler equality, unchanged global RNG/rank partitioning, four-task
validation, startup-only rank-zero warnings including excluded tasks, and resume
rejection before state application. Encoder tests compare every output/layout
field and parameter gradient against the original loop for 0/1/4/16 history
segments, preserve no-gradient collection, and measure real MLP weight-cast
lifetimes. The torchrun helper has a real worker-timeout regression; separate
two-worker, interruption, normal-output/exit, and forced-kill probes also pass.
Graceful launcher termination matters because workers occupy separate sessions.

All 14 targeted mutations were detected: ignored/sorted/duplicate tasks, erased
resume task identity, suppressed excluded-task warnings, warnings on every rank,
wrong attention window, detached gradients, forced deterministic SM12 backward,
skipped adapter installation, retained history graphs, lost compiler context,
late-bound loop inputs, and skipped native-dependency admission.

The compiled encoder oracle compares original and candidate **within each grad
mode**, requiring exact outputs/layout and all 12 parameter gradients. In job
`23043005`, two original repeats and the candidate agree at 0/4 history segments.
Inductor inference can feed an unrounded FP32 embedding sum into RMSNorm, while
autograd materializes BF16 intermediates. Thus equality *between* those modes is
not a valid baseline assumption. Diagnostic `emulate_precision_casts=True`
removes that gap but changes baseline computation; production precision settings
remain unchanged. Raw comparisons/tensors and generated kernels are in
`encoder-numerics-23043005/` and `encoder-numerics-compiler-23043005/`.

```bash
uv run pytest tests/steering/test_sm12_fa4.py -m slow -v
uv run pytest tests/steering/selfcond/test_rlvr_encoder_recompute.py -m slow -v
```

`scripts/setup_fa4.sh` tests the generic FlexAttention block-mask route; the first
command above tests the native SM12 adapter. Original kernel logs are
`kernel-tests-22990547.log`, `fa4-shapes-22991253.log`, `long-window-22994168.log`,
and `qkv-layout-22996605.log`; original CLI results are in `cli-tests-22992359.log`.
Current receipts are in `simplified-23057631/`, `admission-23058523/`, and
`rebased-23060217/`, including source fingerprints, `pytest.log`, and admission
failure logs where applicable. Exact job commands are retained in
`qualify-simplified-final.sh`, `qualify-admission.sh`, and `qualify-rebased.sh`
in the evidence root.

### Repository-wide boundary

The broad pytest run remains **incomplete and failing**. With offline HF/W&B,
hidden CUDA, FP32, and two OpenMP/MKL threads, it reached 46,765 passes, 173
failures, 437 skips, 15 errors, seven xfails, and 620 deselections before a TUI
shutdown hang. The same `test_steering_pending_blocks_reentrant_apply_and_escape[asyncio]`
hangs on untouched main. A separate tail run excluding it gave 1,044 passes,
25 failures, and eight errors; these overlapping counts must not be added.

Of the 173 failures, 157 reproduce on untouched main; 15 hit the one-GiB disk
reserve and pass after space is available (affected-file ordering check: 388
passes); one eight-rank Gloo peer-closed error passes in isolation on both
branches, with its broad-run cause unproven. All tail failures/errors reproduce
on main. Of the first 15 errors, 11 lack RustBPE fixtures (17 rendering tests pass
with the existing cache), and four require unavailable offline AnthropicHH data.
Nine baseline failures disappear with BF16 (47 passes, seven skips); the tail's
eight errors use invalid geometric mean/std fixtures. Local failure receipts:
`out/issue1687/evidence/full-suite-failures.json`.

At the previous delivery, Ruff passed all changed Python files and ty 0.0.82
reported 382 diagnostics versus 385 on untouched `b453e530e`, with no new
normalized diagnostics. Those are historical counts, not a clean whole-repo gate.

## Actual SFT checkpoint

Job `22992739` used step-1116 model SHA-256
`a91b0ffb0fefe6d34c2954f72d57bdbe33f73c306177cfc8e23128c7b9cead27`:
27 decoder layers, 13 heads of width 128, context 2048, cadence 33, 844 allowed
features. Its four-task bundle has 32 training/eight validation rows and identity
`11bdc48b92295cf29ab5478581d855c5faf62b86da439b17c8eec8190763f3e6`.

Both one/two-GPU MC-only runs completed three optimizer steps and two execs,
using four global prompts and four trajectories per prompt. Rewards were
0.5/0.125/0.4375, with finite nonzero gradient norms each step. GSM8K had no
training trajectories but remained in validation; startup parse rate 0/2 warned.
All six checkpoints are complete. LM heads are bitwise unchanged, floating
weights finite, encoder/decoder/cross-attention/feature-head families changed,
and nonfloating buffers unchanged. Peak allocation was 62.49 GiB (one GPU) and
44.72 GiB (reporting rank, two GPUs).

One-GPU standalone evaluation exactly matched training metrics and all eight
samples. Two-GPU metrics matched but one GSM8K sample diverged after 765 shared
characters, with zero parse/reward in both. Receipts in `real-3b-22992739/` include
`run-{1,2}/qualification.json`, `evaluation-replay-{1,2}.json`, `train-{1,2}.log`,
and `evaluation-{1,2}.json`. These bounded checks establish neither accuracy nor
full-training throughput gains.

### Fixed-weight evaluation reproducibility

Follow-ups reused two-GPU step-3 model SHA-256
`099f180874f0d2112487276e92a7e65ddd9becfbb4cfa8d2016f35341a849223`, bundle, and
generation settings without optimizer steps or production changes:

| Job | Within-process result | Difference from original evaluation |
|---|---|---|
| `23001910`, node5101 | Six passes identical: fresh, after unchanged-weight backward, and compiler reset; each with cuDNN token-cache SDPA on/off. | Five sample rows; macro accuracy 0.125 versus 0.0. |
| `23004038`, node5106 | Four passes identical: original evaluator, frozen LM head, restored trainability, compiler reset. | Four sample rows; macro accuracy 0.125. Three rows also differ from `23001910`. |
| `23039079`, three cold two-rank runs | Metrics, all eight samples, buffers, and 577 traced events agree; selected ARC-Easy row includes 27 native FA4 and 81 token-cache SDPA calls. | Macro accuracy 0.0; traced row matches original, two other rows differ. CPU-copy hooks synchronize execution. |

The first two jobs check unchanged parameter versions; the second also records
native FA4, BF16, eval mode, no autocast, disabled determinism, and highest FP32
matmul precision. Evidence supports the measured repeatability, but does not
isolate the earlier fresh-process variation or justify a cuDNN workaround.
Receipts: `repeatability-23001910/`, `trainability-23004038/` (matching root logs),
and `cold-trace-23039079/`.

## History encoder memory

Pod 12's original run is `/workspace/claude/rlvr/run_full_1rank_20260918-1332/`.
Its resolved config is in `meta_000060.json`; model, optimizer, and completion
files are `model_000060.pt`, `optim_000060_rank0.pt`, and
`rlvr_complete_000060.json`. Full tracebacks: `/workspace/claude/rlvr/train_full.log`
and `train_full.log.oom_1802`. Original Hydra YAML plus W&B argv reconstruct the
resumed config with only `resume_from_step` changed, 50→60. That GPU is the
Workstation Edition; Engaging probes use the Server Edition.

The original code already checkpointed canonical replay and backpropagated per
prompt. During replay's backward, each history encoder call nevertheless retained
its own autograd graph. The failing compiled encoder has 122 independent BF16
weight casts/transposes totaling 1,794,256,256 bytes (1.671031 GiB) per call,
before activations; its 1664×6656 MLP cast matches the traceback's 22 MiB request.
Inner nonreentrant checkpoints now recompute each history segment under the
captured compiler context, preserving shapes, append order, prompt encoding,
and no-gradient collection. No weight cache, batching policy, or new knob is added.

Job `23040563` compares original, repeated original, and candidate helpers with
the actual 3B model/native FA4. Each synthetic history entry adds 33 ordinary
tokens and one conditioning segment. Real optimizer checkpoint tensors and
preallocated gradients stay resident; each pass runs one full backward with a
nonzero scalar objective, **without an optimizer step or step-67 replay**.

| History segments | Prefix tokens | Repeated original peak (GiB) | Candidate peak (GiB) |
|---:|---:|---:|---:|
| 0 | 52 | 34.3602 | 34.3602 |
| 4 | 184 | 42.2621 | 38.7143 |
| 16 | 580 | 64.1451 | 38.7138 |

All nine passes complete, with exact candidate logits. All gradients match at
four segments; at 0/16, only the magnitude-embedding gradient differs, by at most
`2.91e-11`/`1.17e-10`. The repeated original has the same maximum differences;
this does not prove elementwise equality between those deviations. Gradients
are finite; separate tests establish connectivity without preallocated buffers.
The 16-segment peak falls 25.43 GiB (39.65%), but the original also fits: this
shows reduced allocation, not recovery from the exact reported OOM. Receipt:
`encoder-memory-23040563/memory.json`; the sidecar candidate was independently
checked against the production helper.

Jobs `23042508` (production memory fix) and `23057631` (simplification) each use
real tokenizers and a schema-5 tiny model on two GPUs: three GSM8K optimizer
steps, two execs, four-task validation.
Every rank/step reaches 15 history segments with 135 or 240 inner checkpoints.
All 26 intended gradients are connected and finite but zero because fixture
advantages are zero; separate encoder/3B probes establish nonzero-gradient parity.
All checkpoints are complete, LM heads exact, weights finite. The simplified
run additionally checks LM-head dtype, shape, and raw bytes, and matches each
rank's gradient names against checkpoint metadata. Receipts:
`history-23042508/{result.json,train.log,trace/}` and
`simplified-23057631/{result.json,receipt-audit.json,train.log,trace/}`.

The original matched 256-per-task panel confirms 159/1024 correct at SFT,
137/1024 at step 50, and 115/1024 at step 60. GSM8K parses 9/256→17/256→8/256,
with 1→1→0 correct. A small startup panel's zero parse rate does not prove the
model never emits the marker. Zero-variance groups have zero direct policy
gradients, while optimizer momentum may still update weights. The accuracy
decline's cause and recovery remain unestablished.

## Distributed transport condition

Engaging's default NCCL path hung at the initial barrier on node5202/node5101,
before attention or checkpoint recovery, despite successful NCCL 2.29.7+cuda13.2
communicator/P2P-CUMEM initialization. Job `22992359` reproduces this in plain
PyTorch without project/FA4 imports. Changing only `NCCL_P2P_DISABLE=1` completes
the barrier and exact all-reduce in under 0.6 seconds via SHM. Distributed gates
therefore use this explicit job setting; production does not set it. The driver
or PCIe cause remains unidentified. NVIDIA documents the [override and performance
cost](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-disable).
Paired logs: `transport-22992359/{default.log,without-p2p.log}`.

## Attention timing

Jobs `22990140`/`22994168` compare native FA4 with nanochat SDPA sequentially on
one GPU: BF16, batch one, 13 heads, width 128. Values are medians of five CUDA-event
trials of 30 calls after five warmups; backward includes `out.sum().backward()`
and gradient clearing, excluding compilation. Window 2048 is the model's finite
long-window representation; unlimited attention takes a different CuTe path.

| Tokens | Left window | FA4 forward (ms) | SDPA forward (ms) | FA4 forward/backward (ms) | SDPA forward/backward (ms) |
|---:|---:|---:|---:|---:|---:|
| 128 | Unlimited (-1) | 0.0485 | 0.0154 | 0.9653 | 0.7711 |
| 128 | 1024 | 0.0482 | 0.0159 | 0.9691 | 0.6767 |
| 128 | 2048 | 0.0499 | 0.0161 | 0.9144 | 0.6672 |
| 512 | Unlimited (-1) | 0.0485 | 0.0214 | 1.1153 | 0.8164 |
| 512 | 1024 | 0.0487 | 0.0199 | 0.9595 | 0.6885 |
| 512 | 2048 | 0.0491 | 0.0207 | 0.9726 | 0.6966 |
| 2048 | Unlimited (-1) | 0.0979 | 0.0906 | 0.9832 | 0.6813 |
| 2048 | 1024 | 0.0974 | 0.4159 | 1.0613 | 2.1363 |
| 2048 | 2048 | 0.0979 | 0.0914 | 0.9573 | 0.7166 |

The 2048-token sliding-window case is 4.27× faster forward and 2.01× forward/backward.
SDPA wins the other measured cases. These measurements establish neither overall
RLVR throughput improvement nor accuracy gain. Logs: `fa4-performance-22990140.log`
and `long-window-22994168.log`.
