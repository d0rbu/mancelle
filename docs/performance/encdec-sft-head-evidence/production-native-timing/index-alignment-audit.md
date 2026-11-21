# Production large-trial index and aggregation audit

Scope: offline source and raw-series review of closed job **23377981**, B8/GA8, all four 40-sample trials (160 updates). No input, runtime, timing policy, or instrumentation changes; no GPU, network, scheduler polling, or test rerun.

**Finding: no project-source index or aggregation mismatch was found.** All 640 published phase/wall values exactly equal the same-index rank-0 sample fields. Adjacent-step shifts do not explain the negative host-minus-CUDA remainders. Their cause remains unexplained; this audit neither repairs the old campaign nor qualifies throughput.

## Same-step source path

In the receipt-pinned `scripts/benchmark/profile_encdec_l22_training.py`, relative to the sealed runtime source root:

- Lines 1017–1035 create a fresh event list and eight fresh event triples per call. Each microbatch contributes once to forward and once to backward; the shared boundary event is the forward end/backward start. Lines 1036–1039 record one optimizer interval after the accumulation loop. There is no extra GA multiplier in the phase sums.
- Lines 1020 and 1040–1048 bracket that call with host start, final device synchronization, 17 elapsed-time queries, and host stop. The single returned dictionary contains all four timings. CUDA elapsed times are milliseconds; the host seconds are multiplied by 1,000 once.
- Lines 1391–1405 append each call's complete dictionary directly to `samples`, after an outer synchronization. There is no shift, sorting, queue, or reuse of the preceding call's event list.
- Lines 1563–1570 retain this list in `rank_result`. Lines 1587–1600 aggregate each key at the same `step` index. All four trials contain exactly one rank, so the per-key rank maximum is the identity operation.
- Lines 1727–1748 serialize rank samples in that same order. Raw validation confirms all 160 × 4 values are finite, nonnegative, and exactly equal between `critical_samples_ms` and `rank_summaries[0].samples_ms`.

These are project indexing/aggregation findings, not proof of lower-level event/clock behavior. The existing raw receipts do not contain per-event stream IDs, whole-step GPU envelopes, or independent host-clock anchors.

## Arithmetic using every original sample

Let `G[i] = forward_ms[i] + backward_ms[i] + optimizer_ms[i]` and `R[i] = wall_ms[i] - G[i]`. All negative values below are also below −1 ms.

| Trial | Samples | R < 0 | Minimum R (ms) | Maximum R (ms) | Wall below every same-trial G | Wall below sum of three phase minima |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 40 | 7 | -282.963022 | 153.899466 | 7 | 6 |
| B1 | 40 | 12 | -296.494278 | 372.356181 | 11 | 11 |
| B2 | 40 | 17 | -296.513899 | 396.320183 | 17 | 17 |
| A2 | 40 | 15 | -297.135612 | 372.964241 | 13 | 13 |

There are **51** negative same-index remainders. **48** wall samples are shorter than the minimum complete CUDA phase total in their trial, so no permutation of those intact totals can resolve those 48 deficits. **47** are shorter even than `min(forward) + min(backward) + min(optimizer)` from their own trial; independently reassigning the recorded phase values cannot resolve those 47 deficits either. This is a lower-bound argument, not an attempted correction or a claim about the cause.

| Trial | Minimum wall (ms) | Minimum complete G (ms) | Sum of separate phase minima (ms) |
|---|---:|---:|---:|
| A1 | 4190.148382 | 4463.639206 | 4451.232620 |
| B1 | 4157.683202 | 4453.338760 | 4437.045959 |
| B2 | 4157.124058 | 4448.898361 | 4439.485962 |
| A2 | 4171.064933 | 4461.328018 | 4450.615387 |

## Bounded lag check

For lag `k`, compare `wall[i]` against `G[i+k]`, never across trial boundaries. Lags −3 through +3 were inspected. First use every valid overlapping pair (`40 - abs(k)` per trial); zero lag has both the lowest residual RMSE and highest Pearson correlation in each trial. Then repeat on the same zero-based wall indices 3–36 for every lag (34 pairs per trial) to remove unequal-edge effects. The table gives this common-index RMSE in milliseconds:

| Trial | k = −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 117.440 | 117.310 | 117.569 | 85.247 | 108.952 | 109.213 | 108.450 |
| B1 | 226.996 | 227.647 | 228.511 | 215.084 | 227.890 | 227.838 | 237.687 |
| B2 | 280.914 | 287.159 | 278.035 | 259.230 | 270.849 | 270.066 | 277.176 |
| A2 | 193.140 | 188.061 | 188.149 | 173.084 | 187.797 | 187.157 | 188.315 |

| Trial | Common-index correlation at k = 0 | Range across the six nonzero lags |
|---|---:|---:|
| A1 | 0.598018 | -0.024683 to 0.171154 |
| B1 | 0.219696 | -0.252241 to -0.028043 |
| B2 | 0.442655 | -0.112338 to 0.233615 |
| A2 | 0.237484 | -0.195817 to -0.056746 |

Thus neither a one-step slip nor the other checked short fixed lags improves these two alignment measures. These are descriptive arithmetic checks, not significance tests. The common-index calculation is supplementary: all 160 original observations remain in the primary audit and campaign, with no sample filtering or altered performance estimates.

## Input identity

Raw filenames below are relative to the original `packed-production-triton-v1-23377981/` result bundle; each has an adjacent `<trial>.provenance.json`. Each raw SHA was reverified locally, and each provenance receipt pins the reviewed profiler SHA `5a8cf492a699062e368f1c56623d7d2cbb18e32b6da5a14aa76f65efcc1108cd`. These names and hashes do not depend on where this audit is copied.

- `large-A1.json`: `2c3355dc1be1cae129d68137ccd76785035f58be89d3b09cd2a96e06fb123788`
- `large-B1.json`: `b9001775355bfb82821c8e2bc6e21725462e482befee72fc8d25c7d510cf9b60`
- `large-B2.json`: `9a56be08e0d6523260309421a5f2417e4eacbeeb8f215d2ba7aed4fc0d4cda65`
- `large-A2.json`: `26a19ea4bb84a1eaa6e05babb4d300f2d2b0483f45aa149c99fe1f862bd8e9b2`

The sealed source is `production-packed-v2/src`, manifest `0e89bf87f2fcbea0e77fe8d3eb43d8c9b5071e9cf45666ecf85c272ea7e9c96f`. No source/index explanation was found. Existing timing validity and throughput qualification remain unresolved.
