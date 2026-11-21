# AxBench-Adv GCG progress across the stack

Each column identifies a change. The tables break a complete bounded DiffMean
search into non-overlapping phases on the RTX 4090 workstation. Values are
**seconds**, averaged across the measured runs at that revision; warmups are
excluded. The phase means sum to the total before rounding. The retained
[measurement summary](axbench-adv-gcg-stack-evidence.json) records the raw-source
digests, run counts, call counts, common search configuration, and reviewed PR heads.

Within each cohort, recorded receipts match the full effective configuration,
rendered-layout digest, model/tokenizer identity, training-pair identity, factor
and suffix length within its objective. The common search settings are:
72 prepared pairs, four sampled pairs, physical batch two, one suffix token,
two steps, 16 proposals, top-k 32, temperature 0.1, distance weight 10, and
master seed zero. Timing includes preparation and normal 128-token-budget
sampled generation with a fake judge; model loading and paid judging are excluded.

**These are historical timings, not matched comparisons between adjacent columns.**
Host scheduling, preparation, and allocation varied between experiments. The
first batched-backward whole-search probe also overlapped CPU pytest. Use the
paired comparisons in [the performance report](axbench-adv-gcg.md) to assess
each change causally. A dash means no complete-cell timing was retained for
that PR; values are neither interpolated nor carried forward. Project-only
and compatibility changes therefore leave gaps in the DiffMean table.

## Response margin

| Workflow phase | Before stack | [#1690](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1690) | [#1692](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1692) | [#1693](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1693) | [#1694](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1694) | [#1696](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1696) | [#1697](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1697) | [#1698](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1698) | [#1699](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1699) | [#1700](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1700) | [#1701](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1701) | [#1702](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1702) | [#1703](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1703) | [#1705](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1705) | [#1707](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1707) | [#1708](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1708) | [#1709](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1709) | [#1710](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1710) | [#1712](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1712) | [#1713](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1713) | [#1714](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1714) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Preparation | 3.584 | 3.073 | — | — | — | — | — | 2.610 | 2.094 | 2.082 | — | 2.098 | 2.178 | 2.219 | 2.124 | 1.914 | 1.852 | 2.102 | 1.983 | 2.091 | 2.092 |
| Candidate filtering | 0.715 | 0.710 | — | — | — | — | — | 0.115 | 0.115 | 0.113 | — | 0.113 | 0.116 | 0.128 | 0.122 | 0.116 | 0.114 | 0.109 | 0.173 | 0.110 | 0.110 |
| Gradients | 1.578 | 1.025 | — | — | — | — | — | 0.965 | 0.965 | 0.958 | — | 0.959 | 0.961 | 0.978 | 0.971 | 0.961 | 0.962 | 0.964 | 0.960 | 0.962 | 0.958 |
| Candidate scoring | 12.085 | 11.951 | — | — | — | — | — | 11.152 | 11.151 | 11.134 | — | 11.133 | 11.159 | 11.438 | 11.302 | 11.201 | 11.135 | 11.172 | 11.039 | 10.887 | 10.738 |
| Final generation and fake judge | 2.605 | 2.596 | — | — | — | — | — | 2.576 | 2.576 | 2.578 | — | 2.577 | 2.582 | 2.744 | 2.609 | 2.585 | 2.586 | 2.586 | 2.580 | 2.583 | 2.575 |
| Other driver work | 0.209 | 0.209 | — | — | — | — | — | 0.206 | 0.212 | 0.214 | — | 0.214 | 0.221 | 0.234 | 0.220 | 0.207 | 0.195 | 0.213 | 0.216 | 0.205 | 0.204 |
| Total cell | 20.775 | 19.565 | — | — | — | — | — | 17.623 | 17.112 | 17.080 | — | 17.094 | 17.216 | 17.742 | 17.348 | 16.985 | 16.845 | 17.146 | 16.951 | 16.838 | 16.677 |
| Measured runs | 3 | 3 | — | — | — | — | — | 3 | 3 | 3 | — | 3 | 3 | 3 | 3 | 1 | 1 | 1 | 3 | 3 | 3 |

## Activation L2

| Workflow phase | Before stack | [#1690](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1690) | [#1692](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1692) | [#1693](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1693) | [#1694](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1694) | [#1696](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1696) | [#1697](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1697) | [#1698](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1698) | [#1699](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1699) | [#1700](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1700) | [#1701](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1701) | [#1702](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1702) | [#1703](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1703) | [#1705](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1705) | [#1707](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1707) | [#1708](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1708) | [#1709](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1709) | [#1710](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1710) | [#1712](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1712) | [#1713](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1713) | [#1714](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1714) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Preparation | 6.572 | 6.434 | — | — | — | — | — | 5.813 | 5.224 | 5.350 | — | 4.787 | 4.744 | 4.886 | 4.676 | 4.721 | 4.609 | 4.689 | 4.634 | 4.619 | 4.552 |
| Candidate filtering | 0.589 | 0.545 | — | — | — | — | — | 0.092 | 0.092 | 0.090 | — | 0.091 | 0.091 | 0.100 | 0.091 | 0.096 | 0.090 | 0.088 | 0.089 | 0.089 | 0.089 |
| Gradients | 1.201 | 0.802 | — | — | — | — | — | 0.724 | 0.787 | 0.726 | — | 0.667 | 0.660 | 0.674 | 0.662 | 0.667 | 0.649 | 0.655 | 0.650 | 0.652 | 0.653 |
| Candidate scoring | 10.769 | 10.530 | — | — | — | — | — | 9.607 | 9.539 | 9.549 | — | 7.822 | 7.616 | 7.983 | 7.632 | 7.651 | 7.524 | 7.531 | 7.459 | 7.337 | 7.245 |
| Final generation and fake judge | 2.611 | 2.598 | — | — | — | — | — | 2.577 | 2.578 | 2.579 | — | 2.577 | 2.580 | 2.637 | 2.581 | 2.581 | 2.577 | 2.582 | 2.580 | 2.580 | 2.577 |
| Other driver work | 0.216 | 0.208 | — | — | — | — | — | 0.217 | 0.212 | 0.211 | — | 0.215 | 0.216 | 0.230 | 0.225 | 0.229 | 0.212 | 0.216 | 0.210 | 0.212 | 0.205 |
| Total cell | 21.958 | 21.117 | — | — | — | — | — | 19.030 | 18.431 | 18.505 | — | 16.159 | 15.906 | 16.509 | 15.867 | 15.944 | 15.662 | 15.760 | 15.621 | 15.489 | 15.320 |
| Measured runs | 3 | 3 | — | — | — | — | — | 3 | 3 | 3 | — | 3 | 3 | 3 | 3 | 1 | 1 | 1 | 3 | 3 | 3 |

`Other driver work` is the measured total minus the five timed client calls.
It includes unclassified driver work; it is not an isolated ranking benchmark.

## Project-model component measurements

These are separate SelfConditioning checkpoint measurements, in **milliseconds**:
each cell shows the paired parent median → new median. They use 72 prepared
pairs, four selected pairs, physical batch four, one suffix token, 21 alternating
pairs, and the root Torch 2.13 environment with buffer donation disabled.
They are component calls, not the DiffMean whole-search phases above.

| Component | #1693: reduced L2 logits | #1696: state guard | #1697: response byte weights | #1701: inference-mode correctness |
|---|---:|---:|---:|---:|
| Response-margin gradient | 91.122 → 91.388 | 95.342 → 91.985 | 84.948 → 84.596 | — |
| Response-margin scoring | 97.011 → 96.258 | 98.920 → 95.105 | 85.974 → 84.654 | — |
| Activation-L2 gradient | 357.035 → 355.874 | 346.613 → 345.747 | 335.374 → 334.738 | — |
| Activation-L2 scoring | 119.741 → 114.641 | 125.881 → 120.133 | 108.487 → 107.432 | — |

#1701 fixes prepared weights created under inference mode; it makes no
additional speed claim. #1694 qualifies the locked Blackwell environment and
also makes no standalone speed claim. #1707 additionally reduces the project
token-pool median from 60.328 ms to 42.938 ms (21 exact paired comparisons).
These compatibility and project measurements are detailed in the main report.

## Changes represented by the columns

- [#1690](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1690): perf(axbench-adv): batch baseline GCG backward passes.
- [#1692](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1692): perf(axbench-adv): skip unused activation GCG logits.
- [#1693](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1693): perf(axbench-adv): reduce project activation GCG logits.
- [#1694](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1694): fix(axbench): qualify frozen workers for Blackwell.
- [#1696](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1696): perf(axbench-adv): reduce GCG state restoration overhead.
- [#1697](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1697): perf(axbench-adv): prepare immutable response byte weights.
- [#1698](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1698): perf(axbench-adv): batch GCG candidate prompt tokenization.
- [#1699](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1699): perf(axbench-adv): batch canonical GCG token-pool checks.
- [#1700](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1700): perf(axbench-adv): reuse prepared GCG response tensors.
- [#1701](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1701): fix(axbench-adv): keep prepared GCG weights usable by backward.
- [#1702](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1702): perf(axbench-adv): stop L2 capture after its decoder target.
- [#1703](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1703): perf(axbench-adv): skip unused activation logit projection.
- [#1705](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1705): perf(axbench-adv): normalize singleton token IDs directly.
- [#1707](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1707): perf(axbench-adv): accelerate ASCII checks in GCG token pools.
- [#1708](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1708): Avoid CUDA scalar reads in GCG activation collectors.
- [#1709](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1709): perf(axbench-adv): copy clean references before GCG forwards.
- [#1710](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1710): strict token-ID validation with built-in iterators for plain lists.
- [#1712](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1712): reuse the private MLP gate buffer during BF16 candidate scoring.
- [#1713](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1713): fuse RMSNorm postscaling while preserving native variance and reciprocal square root.
- [#1714](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1714): fuse BF16 conversion and square while preserving native mean and reciprocal square root.

## Direct pre-stack comparison

A fresh paired RTX 4090 run compares the merged 17-PR stack's baseline worker and state
helper with independent complete frozen copies from pre-stack main
`b453e530ea501aeec1de0d41016117633f719b71`. It uses DiffMean, SmolLM3-3B
commit `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`, 72 prepared pairs, four
selected pairs, physical batch four, one suffix token and two scoring candidates.
The Python 3.12.11 / Torch 2.5.1+cu124 environment is held fixed. Each case uses
seven alternating measured pairs after three warmup pairs. All **28 measured
gradient/score comparisons** are exact; prepared payloads and clean states
also match exactly. These are warm worker calls, excluding preparation and
generation. They are not estimates of whole-search or campaign throughput.

| Component | Pre-stack median (ms) | 17-PR stack median (ms) | Median paired speedup |
|---|---:|---:|---:|
| Response margin gradient | 1254.029 | 470.442 | 2.664× |
| Response margin scoring | 424.163 | 373.759 | 1.134× |
| Activation L2 gradient | 998.820 | 344.008 | 2.902× |
| Activation L2 scoring | 417.209 | 283.398 | 1.473× |

The candidate worker SHA-256 is
`d7242ca10f8858a73d1b9e4f2818e12ef63b286168ac7573a3e4835606c4fea1`.
Raw timings and full numerical tensors are retained in
`out/axbench-gcg-performance/loop23/cumulative-main/`.

## Selected-factor receipt cohort

All columns below include the selected-factor L2 policy from #1716 and have
identical recorded receipt contracts. The #1714 and #1717 columns are retained
from the native-pool experiment; the clean-reference column comes from a later
paired experiment against a fresh #1717 reference. Rotary scoring comes from
a third paired experiment against #1717, and native RMS mean from a fourth
against #1720. Adjacent historical columns are therefore not a matched
performance comparison. All four experiments use the
same 72 training pairs, four selected pairs, physical batch two, k=1, two search
steps, and 128-token generation budget. Each objective has one warmup pair and
three measured pairs. Values are mean seconds, including the same trace auditing
in both arms within each experiment.

The [measurement summary](axbench-adv-gcg-stack-evidence.json) retains the
complete paired experiments in `selected_factor_cohort`,
`clean_reference_fusion_cohort`, `rotary_scoring_cohort`,
`native_rms_mean_cohort`, and `input_staging_cohort`. Historical fields and
numbers are unchanged.

The clean-reference change in #1718 is rejected. Its Blackwell checks pass
exact numerical parity but reveal a 15–16% median DiffMean batch-two slowdown
on two nodes. Its column retains the experimental 4090 timings; subsequent work
starts from #1717 without that scope extension.
See the [Blackwell results](axbench-adv-gcg.md#blackwell-parity-passes-one-performance-case-regresses).

### Response margin

| Workflow phase | [#1714](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1714) with selected-factor L2 | [#1717](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1717) | [#1718](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1718): clean-reference fusion (rejected) | [#1720](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1720): rotary scoring | [#1721](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1721): native RMS mean | Input staging (this change) |
|---|---:|---:|---:|---:|---:|---:|
| Preparation | 2.094 | 2.024 | 2.008 | 1.893 | 1.982 | 1.961 |
| Candidate filtering | 0.109 | 0.110 | 0.109 | 0.110 | 0.110 | 0.109 |
| Gradients | 0.959 | 0.959 | 0.961 | 0.960 | 0.960 | 0.959 |
| Candidate scoring | 10.738 | 10.742 | 10.728 | 10.613 | 10.407 | 10.246 |
| Final generation and fake judge | 2.571 | 2.569 | 2.571 | 2.571 | 2.575 | 2.572 |
| Other driver work | 0.208 | 0.207 | 0.208 | 0.191 | 0.198 | 0.192 |
| Total cell | 16.678 | 16.611 | 16.584 | 16.337 | 16.232 | 16.040 |
| Measured runs | 3 | 3 | 3 | 3 | 3 | 3 |

### Activation L2

| Workflow phase | [#1714](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1714) with selected-factor L2 | [#1717](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1717) | [#1718](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1718): clean-reference fusion (rejected) | [#1720](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1720): rotary scoring | [#1721](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1721): native RMS mean | Input staging (this change) |
|---|---:|---:|---:|---:|---:|---:|
| Preparation | 4.615 | 4.546 | 4.390 | 4.504 | 4.425 | 4.374 |
| Candidate filtering | 0.088 | 0.090 | 0.087 | 0.088 | 0.088 | 0.089 |
| Gradients | 0.650 | 0.652 | 0.649 | 0.650 | 0.651 | 0.652 |
| Candidate scoring | 7.238 | 7.234 | 7.233 | 7.096 | 6.983 | 6.983 |
| Final generation and fake judge | 2.572 | 2.572 | 2.572 | 2.576 | 2.574 | 2.574 |
| Other driver work | 0.214 | 0.206 | 0.208 | 0.202 | 0.194 | 0.198 |
| Total cell | 15.377 | 15.299 | 15.140 | 15.116 | 14.915 | 14.869 |
| Measured runs | 3 | 3 | 3 | 3 | 3 | 3 |

Each experiment matched the complete prepared payload, optimizer requests,
filters, gradients, scores, saved search receipts, and generated text exactly
in all eight pairs, including warmups. The clean-fusion experiment also
remeasured its complete #1717 parent; those fresh paired phase means are retained
in `clean_reference_fusion_cohort` in the measurement summary. Phase timings
are descriptive. The [performance report](axbench-adv-gcg.md) records the
controlled pool-construction and clean-reference component results.

The rotary-scoring column is a later paired experiment with the same receipt
contracts. It excludes rejected #1718. Its immediately preceding historical
column is not its reference. `rotary_scoring_cohort` retains the complete
matched #1717 comparison below; all prior JSON fields and table values remain
unchanged.

## Matched rotary-scoring comparison

Both columns use the same 4090, pinned SmolLM3-3B model, 72 prepared pairs,
four selected members, physical batch two, k=1, two steps and 128-token final
generation. Each objective has one warmup and three measured alternating
pairs. Values are mean seconds with identical trace auditing. All eight pairs
match optimizer requests, filters, gradients, all 1,872 scores, complete
prepared payloads, receipt bytes and generated text exactly. These small
whole-search samples are descriptive; the separate seven-pair component
comparisons establish the reported scoring improvements.

### Response margin

| Workflow phase | #1717 remeasured | [#1720](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1720): rotary scoring |
|---|---:|---:|
| Preparation | 1.880 | 1.893 |
| Candidate filtering | 0.111 | 0.110 |
| Gradients | 0.961 | 0.960 |
| Candidate scoring | 10.746 | 10.613 |
| Final generation and fake judge | 2.572 | 2.571 |
| Other driver work | 0.187 | 0.191 |
| Total cell | 16.458 | 16.337 |
| Measured runs | 3 | 3 |

### Activation L2

| Workflow phase | #1717 remeasured | [#1720](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1720): rotary scoring |
|---|---:|---:|
| Preparation | 4.504 | 4.504 |
| Candidate filtering | 0.090 | 0.088 |
| Gradients | 0.651 | 0.650 |
| Candidate scoring | 7.241 | 7.096 |
| Final generation and fake judge | 2.574 | 2.576 |
| Other driver work | 0.200 | 0.202 |
| Total cell | 15.260 | 15.116 |
| Measured runs | 3 | 3 |

## Matched native RMS mean comparison

The native RMS mean column comes from a fourth independent experiment.
`native_rms_mean_cohort` retains its freshly remeasured #1720 parent and both
complete arms. All prior JSON fields and table measurements are preserved.
The same receipt-contract checks cover the model, training pairs, rendered
layout, selected-factor policy and complete effective search configuration.

Values below are mean seconds from three measured alternating pairs per
objective, after one warmup pair. The workload is the same bounded 4090
DiffMean search: 72 prepared pairs, four selected pairs, physical batch two,
k=1, two steps and 128-token generation with a fake judge. All eight pairs
match prepared payloads, optimizer requests, filters, gradients, all 1,872
scores, receipt bytes and generated text exactly. These full-workflow samples
are descriptive; the separate seven-pair component comparisons establish
the reported scoring improvements.

### Response margin

| Workflow phase | #1720 remeasured | [#1721](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1721): native RMS mean |
|---|---:|---:|
| Preparation | 1.974 | 1.982 |
| Candidate filtering | 0.108 | 0.110 |
| Gradients | 0.960 | 0.960 |
| Candidate scoring | 10.548 | 10.407 |
| Final generation and fake judge | 2.575 | 2.575 |
| Other driver work | 0.195 | 0.198 |
| Total cell | 16.360 | 16.232 |
| Measured runs | 3 | 3 |

### Activation L2

| Workflow phase | #1720 remeasured | [#1721](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1721): native RMS mean |
|---|---:|---:|
| Preparation | 4.423 | 4.425 |
| Candidate filtering | 0.088 | 0.088 |
| Gradients | 0.651 | 0.651 |
| Candidate scoring | 7.096 | 6.983 |
| Final generation and fake judge | 2.575 | 2.574 |
| Other driver work | 0.193 | 0.194 |
| Total cell | 15.026 | 14.915 |
| Measured runs | 3 | 3 |

## Matched scoring input comparison

The input-staging column comes from a separate paired experiment.
`input_staging_cohort` retains its freshly remeasured #1721 parent and both
complete arms. All prior JSON fields and table measurements are preserved.
The same receipt-contract checks cover the model, training pairs, rendered
layout, selected-factor policy and complete effective search configuration.

Values below are mean seconds from three measured alternating pairs per
objective, after one warmup pair. The workload is the same bounded 4090
DiffMean search: 72 prepared pairs, four selected pairs, physical batch two,
k=1, two steps and 128-token generation with a fake judge. All eight pairs
match prepared payloads, optimizer requests, filters, gradients, all 1,872
scores, receipt bytes and generated text exactly. These full-workflow samples
are descriptive; the separate seven-pair component comparisons establish
the reported scoring improvements.

### Response margin

| Workflow phase | #1721 remeasured | Input staging (this change) |
|---|---:|---:|
| Preparation | 1.946 | 1.961 |
| Candidate filtering | 0.109 | 0.109 |
| Gradients | 0.959 | 0.959 |
| Candidate scoring | 10.398 | 10.246 |
| Final generation and fake judge | 2.572 | 2.572 |
| Other driver work | 0.192 | 0.192 |
| Total cell | 16.175 | 16.040 |
| Measured runs | 3 | 3 |

### Activation L2

| Workflow phase | #1721 remeasured | Input staging (this change) |
|---|---:|---:|
| Preparation | 4.433 | 4.374 |
| Candidate filtering | 0.087 | 0.089 |
| Gradients | 0.651 | 0.652 |
| Candidate scoring | 6.983 | 6.983 |
| Final generation and fake judge | 2.573 | 2.574 |
| Other driver work | 0.196 | 0.198 |
| Total cell | 14.925 | 14.869 |
| Measured runs | 3 | 3 |
