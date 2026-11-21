# Encoder-decoder training component profile

## Result

The dominant measured training-code bottleneck was padded cross-attention, and
packed FlexAttention removes it. With the real 1024-token semantic segment
bound still enabled, the repeated 1-node x 4-H100 median improved from 1,294.23
to **787.24 ms/step**: **39.17% lower wall time, 64.40% higher throughput and
MFU, and 20.64 GiB less peak allocation**. It also slightly beat the earlier
unsafe-for-production 128-token static-padding ceiling.

The full planned 3,020,935,888-parameter c2048 model has now also run on four
H200s. Across three paired trials, tensorwise FP8 improved throughput from
119,508 to **132,805 tokens/s (+11.13%)**, raised MFU from 25.73% to **28.59%**,
and reduced peak allocation from 82.72 to **70.03 GiB**. These full-size runs
used random weights and synthetic conditioning tensors because no matching
c2048 checkpoint and packed artifact exist yet.

The planned four-node run can also use per-rank batch 8 while preserving the
524,288-token global batch. Three fresh-process B=8 trials reached **143,873
tokens/s and 30.97% MFU**, with 103.13 GiB peak allocation. The first eager
BF16 validation forward peaked at only 33.80 GiB allocated; validation did not
approach an OOM with the optimizer state resident.

Globally packing the encoder and cross-attention K/V moves that same planned
3B/B=8 shape to **201,387 tokens/s and 43.35% MFU** in the paired 4-H200 gate:
**41.65% higher throughput, 29.40% lower step wall time, and 24.92 GiB less
peak allocation**. The two packed fresh-process medians differed by only 0.05%.

A conservative B=16 capacity canary also completed on two H200s. It reached
**110,332 tokens/s and 47.50% MFU**, with a maximum rank peak of **119.93 GiB
allocated / 121.80 GiB reserved** on a 140.40-GiB H200. This is not a
four-GPU throughput estimate: world size two carries more optimizer state per
rank and used a 524,288-token measured batch (eight accumulation microbatches).
It is nevertheless a stronger memory test than the intended four-GPU layout;
the model, backward pass, packed FlexAttention path, DistMuonAdamW, NCCL, and
two eager BF16 validation steps all completed without OOM or rank failure.

An independent exact-checkpoint/real-row A/B confirmed the result. On one H100,
the compiled forward+backward median for 16 real pre-selected rows fell from
159.552 to **95.753 ms**: **66.63% higher throughput**, with peak allocation
falling from 46.00 to **25.36 GiB**. This used the trained 412,256,050-parameter
checkpoint at step 54,600, not random weights.

CUDA now projects exactly one query row per decoder token instead of
`n_seg * static_max_seg_size / T` rows. It flattens the existing encoder K/V
tensor and uses one data-dependent `BlockMask`, shared by all cross-attention
layers. FlexAttention fuses the masked attention and does not materialize a
dense score matrix. The manifest's `max_seg_size` remains a compiled fail-loud
semantic check; it is no longer a CUDA allocation extent.

The model constructs that shared mask before the decoder-layer loop. A CUDA
cross-attention layer now raises if no mask is passed instead of silently
rebuilding it inside the layer.

The follow-up now packs encoder inputs and cross-attention K/V globally as well.
The dataloader contract remains dense `(B,S,F)`, but CUDA gathers only real
features onto one 32-aligned axis, records `(batch, segment)` owners, and adds at
most 31 tail entries for the whole batch. Encoder self-attention is
segment-diagonal; flattened `(B,T)` cross-attention queries route through the
same owner tensors. No nested or jagged tensor enters the model API.

## FA4 and `torch.compile` compatibility

The original exact-query packed cross-attention below used a static FA4
FlexAttention graph. Global encoder packing makes K content-dependent. PyTorch
2.13 FA4 forward accepts that dynamic block mask, but Hopper backward does not:
the CuTe preprocessor fails IR verification. The final global path therefore
uses fused Triton Flex for encoder and cross-attention, while all decoder causal
self-attention remains on FA4. The production feature encoder and
cross-attention are packed and CUDA-only; their padded SDPA oracles live under
`tests/`. Follow-up support and acceptance evidence are tracked in issue #1447.

H100 job 20644151 exercised one packed layer with explicit
`kernel_options={"BACKEND": "FLASH"}`. Its profile contained
`FlashAttentionForwardSm90` and `FlashAttentionBackwardSm90`, rather than a
Triton fallback. Against padded BF16 SDPA, maximum/mean forward absolute error
was 0.0078125/0.000717 and maximum gradient error was 9.54e-7.

Job 20644465 then compiled the complete 412,256,050-parameter model and ran
forward, backward, and `DistMuonAdamW` on two H100s. The 1024-token semantic
bound remained active, repeated compiled loss was stable, and peak allocation
was 5.56 GiB at device batch 2. The repeated throughput jobs below use the same
whole-model compiled path.

### Why packed attention has its own dynamic compile boundary

Replacing the named function with `functools.partial(flex_attention, ...)` was
tested on the real H100 stack, not rejected on style grounds. With
`dynamic=False`, job 20654810 shared Dynamo's underlying FlexAttention code
object with FA4 self-attention, exhausted the eight-graph recompile limit across
Q/K shapes, and fell back to unfused attention. The packed batch-1 median then
regressed from 70.21 to 101.84 ms. Switching the partial to `dynamic=True` was
also not viable: job 20655280 failed because the FA4 CuTe backend cannot inline
the block mask's captured symbolic scalar.

The global implementation no longer retains the trivial wrappers around either
implementation method. The complete packed encoder method and the complete
packed cross-attention projection-plus-attention method are compiled directly
with `dynamic=True`. The K/V projection consumes the content-dependent packed
token extent, although its weight and channel dimensions are static. Q and the
output projection consume fixed `(B,T)` extents and do not inherently need
dynamic shapes; the current baseline keeps them in the same inner graph so each
layer makes one compiled call. A short local split-Q/O experiment was noisy and
bimodal, so it does not establish that either graph partition is faster. That
comparison remains an H100/H200 benchmark gate after Engaging maintenance. One
explicit disabled dispatcher is still required to graph-break the static
caller before entering the dynamic graph.

Packing and BlockMask construction occur outside Dynamo; packed tensor and
sparse-mask extents are marked dynamic before entering the compiled methods.
The BlockMask builder cannot be inlined on PyTorch 2.13: the full-model compile
then fails with `Attempt to trace forbidden callable ... mark_dynamic`. The
outer model remains static when dataloader shapes are static, preserving the
decoder FA4 path.

Rounding each batch to the next multiple of 32 or 64 does not make this region
static: it only reduces the number of possible lengths, so `dynamic=False`
would still compile a graph per occupied bucket. A truly static graph would
need one fixed worst-case K extent (or a managed finite graph cache), restoring
the padding waste this change removes.

This boundary was selected from failed H200 controls, not only local behavior.
Job 20676435 inlined dynamic packed masks into the full model and failed because
FA4 could not capture a symbolic scalar. Moving packed work behind its own
dynamic boundary removed that failure, but jobs 20677054 and 20678117 then
isolated a CuTe backward IR-verification error. Job 20678273 ran encoder and
cross-attention output+gradient differential tests successfully after selecting
Triton for only those dynamic-mask regions.

## Exact checkpoint and real-row validation

Engaging job 20656023 ran padded base `f24857ad` and packed candidate
`6c9cbd33` back-to-back on node1702's H100 with PyTorch 2.13.0+cu130 and FA4.
Both loaded checkpoint
`l22_main_h100_8rank_db16_v1/.../model_054600.pt` and the same 16 rows from
packed shard 0 with selection hash
`e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3`.
The full model used `torch.compile(dynamic=False)`; three warmups preceded seven
synchronized forward+backward trials. The batch was already reconstructed and
resident on device, so these numbers exclude decode, H2D, and optimizer work.

| CUDA cross-attention | Median forward+backward | tokens/s | Peak alloc. |
|---|---:|---:|---:|
| Padded SDPA | 159.552 ms | 102,687 | 46.00 GiB |
| Packed Flex/FA4 | **95.753 ms** | **171,107** | **25.36 GiB** |

Packing improved throughput by 66.63%, reduced forward+backward wall time by
39.99%, and saved 20.64 GiB. All seven padded samples were 159.40-159.86 ms;
all seven packed samples were 95.72-96.03 ms.

Job 20655599 separately compared the complete output and backward pass for one
real row under the same checkpoint and H100 allocation. The packed-versus-
padded combined/language-model/feature-prediction loss differences were
3.03e-5, 3.36e-5, and 3.42e-5. Across the complete 1024x32768 logit tensor,
relative RMSE was 0.927%, cosine similarity was 0.999957, mean KL divergence was
8.60e-4, and token argmax agreement was 98.24%. Ten representative full
gradient tensors spanning the encoder and cross-attention layers had cosine
similarity 0.9971-0.9999 and maximum absolute error at most 3.82e-5.

These are explicit BF16 kernel-change tolerances, not bitwise or tight
elementwise equality: maximum logit error was 0.704, so a blanket
`allclose(atol=0.01)` does not pass after twelve cross-attention injections.
The loss, distribution, and gradient checks do pass the training-relevant
bounds above. FA4 backward on H100 is normally nondeterministic; PyTorch 2.13
only supports deterministic Flex/FLASH block-mask backward on SM100 or newer.

The matching trained checkpoint has `max_seq_len=1024`. No context-2048 trained
checkpoint with the same artifact contract was available, so this validation
does not misrepresent a mismatched c2048 model as real-checkpoint evidence.

### Globally packed encoder real-row follow-up

Job 20678384 repeated the one-row trained-checkpoint probe after globally
packing the encoder and selecting Triton for dynamic-mask attention. The row had
173 real features in a dense `1 x 13 x 96 = 1,248` extent; the CUDA axis was 192
tokens including its alignment tail, a 6.5x reduction in encoder/KV slots. Two
warmups preceded seven synchronized forward+backward trials.

| CUDA encoder/KV layout | Median forward+backward | tokens/s | Peak alloc. |
|---|---:|---:|---:|
| Dense per-segment extent | 74.784 ms | 13,693 | 3.392 GiB |
| Globally packed real features | **72.650 ms** | **14,095** | **2.921 GiB** |

This small batch improved throughput by 2.94% and saved 0.471 GiB. The timing
probe leaves the outer model eager; each implementation still uses its required
compiled attention regions. The full-size compiled result below is the
production-throughput gate.

Against the same dense checkpoint forward, combined/LM/feature-prediction loss
differences were 9.40e-4, 9.51e-4, and 1.09e-4. Across all logits, relative L2
error was 0.987%, cosine similarity was 0.999951, mean KL was 9.86e-4, and
argmax agreement was 97.95%. The sparse feature-embedding gradient had maximum
absolute error 5.36e-7 and cosine 0.9929; nine cross-attention parameter
gradients had cosine 0.9985-0.9998 and maximum absolute error at most 4.58e-5.

## Full 3B c2048 H200 FP8 A/B

The earlier throughput campaigns used the exact 412,256,050-parameter trained
model. Job 20661396 instead measured the complete planned geometry: symmetric
27-layer, width-1,664 encoder and decoder stacks, context 2,048,
3,020,935,888 total parameters, and cross-attention at every decoder layer.
The run used four H200s on node4301, PyTorch 2.13.0+cu130, NCCL 2.29.7, FA4,
per-rank batch 4, and a fixed 524,288-token global batch. World size four
therefore used 16 gradient-accumulation microbatches per optimizer step.

The c2048 production packed artifact and matching checkpoint do not exist yet.
This experiment consequently used random weights and synthetic conditioning
tensors with 26x96 padded slots, 16 real 128-token segments, and 24 selected
features per real segment. It exercises the exact compiled full model and
optimizer allocation, but is throughput evidence rather than a real-data or
training-quality claim.

Each variant ran in three fresh processes, with order reversed on the middle
trial, two complete warmup steps, and five synchronized measured steps. The
table reports the median of the three trial medians on the critical rank.

| Precision | Forward | Backward | Optimizer | Wall | tokens/s | MFU | Peak alloc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1,446.10 ms | 2,860.76 ms | 113.27 ms | 4,387.04 ms | 119,508 | 25.73% | 82.72 GiB |
| Tensorwise FP8 | **1,270.05 ms** | **2,596.98 ms** | **106.65 ms** | **3,947.78 ms** | **132,805** | **28.59%** | **70.03 GiB** |

FP8 converted 297 linear modules and skipped 30 under the trainer's existing
conversion policy. It reduced step wall time by 10.01%, increased throughput by
11.13%, raised measured MFU by 2.86 percentage points, and saved 12.69 GiB of
peak allocated memory. The paired throughput improvements were 11.18%, 11.05%,
and 11.13%; BF16 trial medians spanned 4,385.81-4,387.34 ms and FP8 spanned
3,946.10-3,949.45 ms. The 15 raw critical-rank wall samples spanned
4,357.12-4,399.58 ms for BF16 and 3,923.65-3,958.00 ms for FP8.

Job 20660079 separately supplied the first full-shape compiled FP8 allocation
smoke. Its first step was compile-contaminated and is not a throughput sample;
it is retained only as confirmation that the model, optimizer, backward, and
26x96 conditioning extent fit on all four H200s.

### Per-rank batch 8 and eager validation

The world-size-32 run needs two B=4 microbatches per optimizer step, but one
B=8 microbatch is already the complete 524,288-token global batch. Jobs
20671052, 20671149, and 20671150 measured B=8 in three fresh processes on
node4400. Each completed two warmup steps, five synchronized training samples,
one validation warmup, and three synchronized validation samples. Job
20671052 was preempted only after emitting its complete first-trial result;
the other two jobs completed normally. Job 20671408 supplied a current B=4
control on the same node and code.

The validation probe matches the trainer rather than the compiled throughput
path: `disable_fp8(orig_model)`, eager `orig_model`, `torch.no_grad`, and
`loss_reduction="none"`, with optimizer state still resident.

| Per-rank batch | World-4 accum. | Step wall | tokens/s | MFU | Training peak alloc. / reserved | First validation peak alloc. | Steady validation peak alloc. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 control | 16 | 3,950.98 ms | 132,698 | 28.57% | 70.03 / 73.89 GiB | 30.87 GiB | 27.71 GiB |
| 8, three-trial median | 8 | **3,644.08 ms** | **143,873** | **30.97%** | 103.13 / 107.40 GiB | 33.80 GiB | 27.48 GiB |

B=8 reduced the matched-node step by 7.77% and increased throughput by 8.42%.
Its three step medians spanned 3,637.30-3,644.23 ms, while peak allocation
spanned only 103.1327-103.1341 GiB. The H200s reported 140.40 GiB of physical
memory, leaving 32.95 GiB outside the allocator at the largest observed
reserved peak and 37.27 GiB above the largest live allocation. The first
validation-forward peak spanned 32.78-33.93 GiB. Validation is therefore not
the memory limiter; FP8 forward/backward training is.

This is strong evidence for starting the four-node x eight-H200 canary at B=8.
It remains synthetic-row evidence: a future c2048 packed manifest with a larger
conditioning extent than 26x96 must rerun the canary. B=8 is not valid at world
size 64 with the same global batch because one world microbatch would contain
1,048,576 tokens.

### Global encoder/KV packing at 3B, B=8

Job 20678373 measured the global packing follow-up on node3001 with PyTorch
2.13.0+cu130. It used the same complete 3,020,935,888-parameter model, FP8
policy, c2048 sequence, 26x96 dense conditioning extent, B=8, world size four,
and 524,288-token global batch as the B=8 campaign above. Each row had 16 real
segments with 24 real features, so CUDA reduced 19,968 dense encoder/KV slots
per rank to 3,072 globally packed slots (6.5x) with no alignment tail in this
particular shape.

The order was dense-a, packed-a, packed-b, dense-b. Every variant used a fresh
compiler cache, two complete optimizer-step warmups, and five synchronized
samples. The table averages the two fresh-process medians; peak allocation is
the average of each trial's maximum rank.

| Encoder/KV layout | Forward | Backward | Optimizer | Wall | tokens/s | MFU | Peak alloc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense per-segment extent | 1,192.20 ms | 2,413.73 ms | 192.22 ms | 3,687.65 ms | 142,174 | 30.60% | 102.62 GiB |
| Global real-feature packing | **899.52 ms** | **1,620.30 ms** | **140.98 ms** | **2,603.39 ms** | **201,387** | **43.35%** | **77.70 GiB** |

Global packing reduced forward by 24.55%, backward by 32.87%, wall time by
29.40%, and peak allocation by 24.28% (24.92 GiB). Throughput and measured MFU
rose by 41.65%, or 12.75 MFU percentage points. The two paired throughput gains
were 41.83% and 41.46%; wall reductions were 29.49% and 29.31%. Packed trial
medians were 201,436 and 201,337 tokens/s.

The model parameter set, optimizer messages (42), physical collectives (64),
and 11.501 GiB collective payload were unchanged. The measured optimizer phase
also fell by 26.66%, but that is not an algorithmic optimizer improvement and
should not be extrapolated independently of the end-to-end paired result. No
recompile-limit, unfused-FlexAttention, non-finite-loss, rank failure, or NCCL
failure appeared. Fresh compilation is excluded from all samples.

Job 20663921 repeated the paired campaign on node4900+node5200 with two H200s
per node and the working Engaging host-staged path
(`NCCL_NET_GDR_LEVEL=LOC`, `NCCL_PXN_DISABLE=1`). The same three-fresh-process,
two-warmup, five-sample protocol completed without a failed rank or hang:

| Precision | Forward | Backward | Optimizer | Wall | tokens/s | MFU | Peak alloc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1,450.65 ms | 2,883.70 ms | 376.63 ms | 4,645.54 ms | 112,858 | 24.29% | 77.04 GiB |
| Tensorwise FP8 | **1,267.74 ms** | **2,603.37 ms** | **337.56 ms** | **4,173.57 ms** | **125,620** | **27.04%** | **64.35 GiB** |

FP8 increased 2x2 throughput by 11.31%. Moving from 4x1 to 2x2 increased BF16
wall time by 5.89% and FP8 by 5.72%. Forward changed by +4.56/-2.31 ms and
backward by +22.93/+6.38 ms for BF16/FP8, while optimizer time increased by
263.36/230.91 ms. The placement cost is therefore still transport, but the
exact 3B model did **not** reproduce the earlier 412M model's 29.63x optimizer
blow-up: optimizer time rose 3.32x for BF16 and 3.17x for FP8.

The three 2x2 wall medians were 4,634.01-4,652.38 ms for BF16 and
4,171.54-4,181.75 ms for FP8. Across all 15 raw samples per precision, BF16
spanned 4,593.43-4,658.06 ms and FP8 spanned 4,146.68-4,189.91 ms. No rare
stall appeared in this bounded campaign. The placements used different nodes,
so their delta remains a cross-campaign calibration rather than a matched-node
causal estimate.

NCCL also logged unavailable optional `mlx5` data-direct symbols and localhost
RAS-port conflicts on every fresh group. All rings nevertheless connected as
PXN 0 / GDR 0 and every trial completed. These are cluster-environment
diagnostics rather than benchmark failures, but further limit portability of
the Engaging transport calibration.

### Four-node x eight-H200 projection and token horizon

The profiling preset fixes shape and global batch but deliberately does not set
a training horizon. Applying the existing L22 pretraining policy
`target_param_data_ratio=100` to the trainer's 1,769,543,414 scaling parameters
gives 337,513 optimizer steps and **176,954,015,744 tokens (176.954B)** at the
524,288-token global batch. This is 100 tokens per scaling parameter but 58.58
tokens per total parameter because the trainer excludes the encoder,
cross-attention, and feature-prediction head from its scaling count.

At world size 32, B=4 requires two microbatches per step. Both projections below
divide the measured world-size-four forward/backward phases by eight. They then
anchor optimizer time to the observed 4x1 result rather than incorrectly
treating the analytical communication service as the entire optimizer.

The target-fabric estimate adds the analytical world-size-32 minus world-size-4
service delta to the observed 4x1 optimizer. The Engaging-calibrated estimate
first scales the 25-us service model to reproduce the measured 4x1-to-2x2 GDR0
optimizer delta, then applies that multiplier at world size 32. Both
conservatively hold the remaining optimizer time fixed even though sharded
update work should decrease with world size. The service model assumes
900 GB/s NVSwitch and one 400 Gb/s inter-node rail; neither estimate includes
contention or overlap.

| Inter-node latency | Target-fabric FP8 step / 100x runtime | Engaging-GDR0 FP8 step / 100x runtime |
|---:|---:|---:|
| 5 us | 649.02 ms / 60.85 h | 705.07 ms / 66.10 h |
| 25 us | **654.06 ms / 61.32 h** | **714.90 ms / 67.02 h** |
| 100 us | 672.96 ms / 63.09 h | 751.76 ms / 70.48 h |

At the 25-us center assumption, these correspond to approximately 801,600 and
733,400 tokens/s. The matching BF16 estimates are 67.10 and 74.45 hours, so FP8
retains a material advantage under either transport model.

The B=8 same-node control lowers forward+backward service by 38.99 ms after the
same world-4-to-world-32 division. Its measured optimizer critical path was
19.28 ms higher than the B=4 control, so retaining that full penalty gives a
conservative net 19.71 ms projected step reduction rather than assuming the
optimizer is batch-invariant. Under that sensitivity, the B=8 25-us target and
Engaging-GDR0 optimizer-step runtimes are approximately **59.47 and 65.17
hours**; the 100-us Engaging case is 68.63 hours. These remain projections, not
world-size-32 measurements.

The global-packing gate above subsequently removed another 1,086.11 ms from
world-size-four forward+backward. Applying one eighth of that measured compute
delta to the B=8 projection, while conservatively crediting none of the observed
optimizer reduction, lowers the 25-us target/Engaging optimizer-step estimates
to **46.74/52.44 hours**. The 100-us Engaging case becomes 55.90 hours. This is
still a world-size-four extrapolation; the real world-size-32 canary supersedes
it.

Periodic conditioned-BPB validation is eager BF16, not FP8 or compiled. At
B=8/world-32, `eval_tokens=20,971,520` gives 40 microbatches per validation
stream. The measured three-trial median was 221.44 ms per microbatch, or 8.86
seconds per stream and event. Over 337,513 steps, `eval_every=250` evaluates
1,352 times including step zero and the non-divisible final step: **3.33 hours
per validation stream**, or **6.65 hours** when both original and spliced
validation are active. Conservatively retaining that pre-packing validation
cost, the global-packed B=8 center estimates including dual validation are
therefore **53.4-59.1 hours**; the 100-us Engaging case is 62.6 hours. A single
stream instead gives 50.1-55.8 hours.

Input decode/H2D, checkpoints, CORE/sample/steering evaluation, cluster
contention, and rare stalls remain outside the measurement. A **64-72 hour
reservation** is the updated practical planning envelope for the 100x run, not a
promise of end-to-end wall time. The real allocation's world-size-32 canary
should replace this estimate.

### Real c2048 B16 calibration and long horizon

Engaging job 21208779 ran the exact 2,986,781,878-parameter clustered c2048
model and real packed/selection artifacts on two H200s across two nodes. B16,
FP8, FA4/FlexAttention, a 1,048,576-token global batch, and 16 accumulation
microbatches produced a 9.771-second steady median, 107,312 tokens/s, and
46.36% MFU. The connected NCCL rings reported PXN 0 / GDR 1. B32 exhausted the
H200 before the first training update, confirming B16 as the production point.

Merged-main job 21229452 then ran the full 500,000-step preset with a 32-batch
FIFO on the same two-node, one-H200-per-node geometry. It reached step 168 and
five packed-shard transitions before the intentional timeout. Excluding the
compile step, the ordinary-step median was 9.812 seconds, 106,900 tokens/s,
and 46.16% MFU. Boundary-inclusive throughput was 106,228 tokens/s. Four
transitions added 1.14-1.34 seconds, while the fifth added 5.57 seconds
(15.384 seconds total), so depth 32 reduced but did not eliminate the tail.
The first eight base bodies were all 1,442-1,447 MiB and their selection bodies
12.41-12.47 MiB; body size does not explain the fifth-transition outlier.

The production preset therefore retains 64 complete batches. At world size 32,
each full shard contributes 32 rank-local B16 batches, so this is two full
shard-runs of lookahead rather than one. On the two-H200 calibration geometry it
doubles the decode lead from approximately 20 to 39 seconds. The fixed batch
payload is 1,531,264 bytes, so depth 64 retains about 93.5 MiB per rank, only
46.7 MiB above depth 32. Job 21229452 measured at least 6,338 MiB HBM headroom
and a 42,355,452 KiB per-rank maximum RSS under a 180 GiB per-rank request. This
is an execution-only queue-depth change; it does not alter batches, RNG state,
checkpoint identity, or remote semantic identity. A matched depth-64 boundary
run remains the performance gate; these capacity numbers are not a throughput
result for depth 64.

Perfect 32-GPU scaling of the B16 median is a 28.63-hour compute-only lower
bound for the 176,953,491,456-token 100x run. Retaining the measured world-32
optimizer/transport components above while applying the measured B16 per-GPU
gain gives **36.3-40.9 hours of optimizer updates**. At `eval_every=5000`, the
168,756-step run evaluates both validation streams 35 times (step zero, every
5,000 steps through 165,000, and the final step), adding approximately 0.17
hours under the retained validation calibration. The resulting
model-plus-validation estimate is **36.5-41.1 hours**. Checkpoints, filesystem
variance, and cluster contention justify retaining a **48-60 hour practical
reservation** until a real world-size-32 canary supersedes the projection.

The exact packed inventory contains 1,604 output shards: 1,603 complete
16,384-row shards and a final 15,028-row shard, for 26,278,580 rows or
53,818,531,840 c2048 tokens per epoch. The long preset runs 500,000 complete
1,048,576-token global batches: **524,288,000,000 tokens**, or **9.74177
epochs**. Scaling the calibrated compute cost gives **107.6-121.2 hours** of
optimizer updates. The cadence evaluates both validation streams 101 times
(step zero through step 500,000 inclusive), adding approximately 0.50 hours,
for **108.1-121.7 hours (4.5-5.1 days)** model-plus-validation. A conservative
**5.9-7.4 day reservation** retains the prior operational margin.

For comparison, preserving the old 54,600-step horizon would consume
28,626,124,800 tokens (28.626B) and projects to 9.92-10.84 hours of FP8
optimizer steps plus about 0.39 hours of base validation under the center
assumptions. Inheriting the generic 8.25x base-training default would instead
produce 27,844 steps and 14.598B tokens. The eventual production config must
choose one explicitly.

## Distributed method

The distributed throughput measurements used PyTorch 2.13.0+cu130, NCCL
2.29.7, FA4, H100 80 GB GPUs, device batch 16, sequence length 1024, and global
batch 524,288 tokens. The exact 412M profiling geometry has 12 cross-attention
layers. These distributed rows and weights were synthetic; each row had eight
real 128-token segments in 13 segment slots and 24 real features in 96 feature
slots. The exact-checkpoint real-row result above is a separate single-H100
control and is not folded into the distributed table.

Each placement used three fresh process groups. Every trial compiled from a
fresh cache, warmed two optimizer steps, and measured ten synchronized steps.
Reported values are medians of the three trial medians, with the critical rank
defining each distributed phase. GDR beyond the node and PXN were disabled
(`NCCL_NET_GDR_LEVEL=LOC`, `NCCL_PXN_DISABLE=1`).

The harness excludes packed-body decode, remote I/O, H2D staging, validation,
and checkpointing. It measures model and optimizer work, not the end-to-end
input pipeline.

## Same-node throughput

| CUDA cross-attention | Job | Forward | Backward | Optimizer | Wall | tokens/s | MFU | Peak alloc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Padded SDPA, bound 1024 | 20637800 | 444.00 ms | 838.92 ms | 13.10 ms | 1,294.23 ms | 405,096 | 9.63% | 49.30 GiB |
| Padded SDPA, bound 128 | 20638713 | 290.70 ms | 511.14 ms | 15.16 ms | 812.27 ms | 645,463 | 15.35% | 30.04 GiB |
| Packed Flex/FA4, bound 1024 | 20644766 | 288.44 ms | 489.23 ms | 23.45 ms | **787.24 ms** | **665,983** | **15.83%** | **28.66 GiB** |

The production-bound padded baseline was reproduced separately on node2803 at
1,294.84 ms, within 0.05% of the node2501 row above. The packed run used
node1703, so this is a same-shape cross-campaign comparison rather than a
same-node pair; the two padded replications and the packed fresh-process spread
bound the observed node variance. Packed trial medians were 784.32, 787.24,
and 789.52 ms, and throughput ranged from 664,059 to 668,462 tokens/s.

Relative to padded bound 1024, packing reduced forward by 35.03%, backward by
41.68%, wall by 39.17%, and peak allocation by 41.86%. Relative to the padded
bound-128 counterfactual, it was still 3.08% faster, 3.18% higher-throughput,
and 1.38 GiB smaller while correctly accepting real segments up to 1024 tokens.

## Multi-node throughput

| Placement | NCCL ring | Job | Forward | Backward | Optimizer | Wall | tokens/s | MFU | Peak alloc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node1703, 4 GPUs | PXN 0 / GDR 1 | 20644766 | 288.44 ms | 489.23 ms | 23.45 ms | 787.24 ms | 665,983 | 15.83% | 28.66 GiB |
| node1702+1703, 2 GPUs each | PXN 0 / GDR 0 | 20644768 | 290.27 ms | 488.10 ms | 694.98 ms | 1,463.65 ms | 358,206 | 8.52% | 27.92 GiB |

Moving two ranks off-node added 676.41 ms (85.92%) to the step and reduced
throughput by 46.21%. Forward changed by +1.83 ms and backward by -1.13 ms,
while optimizer time increased by 671.52 ms (29.63x). Packed model compute is
therefore topology-stable; the penalty is in collective transport.

The transport was also highly variable. The three 2x2 optimizer medians were
589.68, 694.98, and 749.12 ms. Across the 30 measured steps, optimizer samples
ranged from 222.25 to 1,578.18 ms and wall time from 993.28 to 2,348.26 ms. All
four ranks saw the same stalls. By contrast, the earlier 2x2 job on
node2900+2901 measured 74.25 ms optimizer time. The current result confirms a
real node-pair/network problem but does not implicate packed attention or
provide a portable 8x8 H200 transport estimate.

## Why this target was selected

Before changing the implementation, three repeated eager diagnostics assigned
48.47-48.48% of forward time to cross-attention. Encoder and decoder blocks
together contributed about 34.8%; the LM head contributed 1.5% and the feature
prediction head 0.32%. Hook overhead against adjacent eager controls stayed
between -0.22% and +0.39%.

The matched node2803 padding counterfactual changed only the validated bound
from 1024 to 128. It reduced wall time by 37.27% and peak allocation by 19.26
GiB without changing optimizer time. Packed FlexAttention recovers and slightly
exceeds that measured ceiling without truncation or a data-dependent compile
bucket. The result therefore confirms the attribution and the proposed cause;
it is not only an operator-profile hypothesis.

## Correctness and remaining gates

Focused tests cover segment isolation, uneven per-item segment counts, the
single global alignment tail, all-empty input, GQA, STE gradients, CPU
forward/backward, packed-versus-padded encoder output and parameter gradients,
an independent cross-attention oracle, compiled raw-bound acceptance, and
fail-loud stale-bound rejection. They also verify that direct CUDA layer calls
with dense K/V or without a shared block mask fail and that production-sized packed-length
changes reuse the warmed compile graphs. Exact-checkpoint loss/logit/gradient
parity covers one real row, but full training-trajectory parity has not been
established.

The next useful measurements are:

1. When matching c2048 data and a trained checkpoint exist, repeat the real-row
   gate at context 2048.
2. Keep optimizer transport separate. Packed attention removes model compute
   waste but cannot repair a cross-node `DistMuonAdamW` transport penalty.
3. Revisit dynamic-mask FA4 only after its Hopper backward accepts these masks;
   carrying a second production backend switch is not justified meanwhile.

The full pytest suite was intentionally not run.
