# ChatCORE compaction tile — 2026-09-21

Changing the existing FP8 cache compactor's byte tile from **1024 to 4096** reduced
warm complete-suite latency from **9.039791 to 8.968406 seconds** in the measured
L40S qualification: **0.78968% / 71.39 ms**. Both candidate processes beat the
faster reference by **0.459–0.593%** using geometric means and **0.450–0.600%**
using medians. Completion bytes, grades, scores and sampled logits were exact.

Reference drift of **+0.532%** materially affects the gain's magnitude: adjacent
AB and BA reductions were **0.459% and 1.119%**. Every conservative process
comparison still favored the candidate. The accepted claim is scoped to these
sources, checkpoint and workload; it is not a universal percentage or confidence
bound. No drift correction, selected-reference estimate or prior-stage gain is
added to the primary result.

## Change and supporting screens

The reference is published commit
`7003bc5247d2a4786556361d8ad1b996716f8d50` ([the existing compactor](chatcore-cache-compaction-20260920.md)).
The only runtime change is `_BLOCK_BYTES = 1024` → `4096` in
`steering/eval/triton_kv_cache_compaction.py`. Admission, fallback, row ordering,
integer byte copies, launch options, storage and view semantics remain unchanged.
There is still one paired K/V launch per layer and the same index allocation/H2D
and scalar position copies. At capacity 859, a row has 1,429,376 bytes; its grid
falls from 1,396 to 349 programs per layer. This changes work per program, not the
number of layer launches. The helper test crosses the 4096 boundary with a
4113-byte row at offset 7, including hidden prefix/tail bytes.

Component job `23354196` compared complete synchronized compaction calls against
the published 1024 helper. Synthetic raw bytes used actual first/late census
geometry and keep mappings, with 26 layers and the full 150-row backing retained
for both cases. The 150→149 case fell from 22.9808 to 21.7139 ms (**5.5127%**);
50→49 fell from 12.7822 to 12.1836 ms (**4.6829%**). All 320 timed calls and 32
excluded warmups were retained, with 352 identical full restorations outside the
timers. All eight comparative quartets per case were positive; orientation,
half and reference-reference checks passed. These component gains motivated
whole-suite measurement and do not predict or add to suite savings.

Seventeen tiny cases comprising 19 steps or error attempts passed independent
integer-byte replay; both full-size fixtures passed GPU backing/position/metadata
comparisons. Full-size tensors were not persisted for CPU replay. Ordering/mask
review covered six saved last-launch compiled bundles, not every specialization;
cubin hashes were bound without a SASS or achieved-occupancy claim.

Resident job `23356531` retained 44 complete suites: four excluded warmups,
32 comparative calls and eight reference-reference calls. Its reduction was
**0.8200% / 74.40 ms**, with 8/8 positive quartets, ABBA/BAAB effects of
0.8456%/0.7944%, and a maximum absolute reference-reference difference of 0.05435%.
Every fixed guard and the qualitative review passed. First excluded warmups
observed 511 decodes, 70 helper admissions and 1,820 layer launches **in each arm**,
with 53 backing descriptors preserved. Witnesses were removed before measured
calls. These are prior resident observations, not fresh qualification route counts.

## Fresh-process qualification

Job `23359170` completed with exit `0:0` in 7:15 on `node3204`: one NVIDIA L40S,
eight CPUs, 64 GiB host memory, PyTorch `2.13.0+cu130`, Triton `3.7.1` and driver
`590.48.01`. The d26 SFT step-999 checkpoint uses 26 layers, width 1664,
13 attention/KV heads and model context 2048. The 4096 value is a byte tile,
not a model context change. The six tasks each use 50 problems, with the existing
512-token generation budget and two-worker HumanEval grading.

After five focused GPU tests, six natural model processes ran the fixed order
`R5, C5, C5, R5, R1-logits, C1-logits`. Both complete source exports used the
unchanged canonical benchmark; there was no resident overlay or helper patch.
Seven phases had 300-second child limits, 14 distinct compiler-cache paths and a
30-minute outer limit. CUDA synchronization brackets each complete `test_chatcore`
call, including live compaction guards/query, allocation/H2D, kernels, position
copies, views and grading. Model loading and artifact serialization are outside.

Exactly repeats 1–4 of the four five-suite processes enter primary timing: eight
warm calls per arm. All four first calls and both instrumented logit audits are
retained and excluded by precommitment. First-call exclusion does not imply cold
hardware, driver or filesystem state.

| Process, execution order | Warm geometric mean | Warm median | Excluded first call |
|---|---:|---:|---:|
| Reference 1 | 9.015832 s | 9.013058 s | 15.444038 s |
| Candidate 1 | 8.974444 s | 8.972488 s | 15.568313 s |
| Candidate 2 | 8.962371 s | 8.958954 s | 15.619965 s |
| Reference 2 | 9.063814 s | 9.066201 s | 15.645544 s |

The primary reduction is `100 × (1 − exp((log(C1/R1) + log(C2/R2))/2))`, using
these warm geometric means. Both candidate processes must beat **both** references
by geometric mean **and** median; both adjacent effects must also be positive.
All conditions passed, with no minimum-percentage threshold.

| Candidate/reference | Geometric-mean reduction | Median reduction |
|---|---:|---:|
| C1/R1 | 0.459061% | 0.450125% |
| C1/R2 | 0.986009% | 1.033658% |
| C2/R1 | 0.592964% | 0.600281% |
| C2/R2 | 1.119203% | 1.182934% |

Reference geometric-mean/median drift was +0.532196%/+0.589627%; candidate drift
was −0.134521%/−0.150835%. This order sensitivity limits precision of the pooled
magnitude, while all faster-reference contrasts preserve the direction. No call
was removed, replaced or repeated. Four timing processes on one additional
allocation are not 16 independent experiments; no confidence interval is claimed.

## Correctness, memory and evidence

All 22 suites matched actual baseline completion/grade bytes and scores: 150 rows,
42,089 tokens, 50 HumanEval verdicts with three passes, one 50-program/two-worker
batch, zero scalar grades and ChatCORE 0.2867. Each separate audit matched eight
samples, 705 IDs and 1,048,576 finite FP32 logits byte-for-byte against the
historical f95 correctness oracle and the other arm. The timed reference remains
7003. Audit times of 15.606170/15.082058 s are excluded; not every trained-checkpoint
cache/logit value was compared.

CUDA allocator peaks matched across arms: 18,571,301,888 bytes allocated in every
call; 26,533,167,104 bytes reserved for first calls/audits and 28,399,632,384 for warm
calls. This is observed parity, not a memory-reduction claim. All 372 ownership
samples had zero observed outsiders; 266 showed owned GPU activity. One-second
sampling cannot exclude subsecond interference. Sampled process-tree RSS can
miss short workers and double-count shared pages; the evidence retains those
observations separately from per-process maxima.

The [evidence JSON](evidence/chatcore-cache-compaction-tile-20260921.json) retains
all 22 timings/classifications, process contrasts, memory, source/input/control
pins, stage audits and the immutable acceptance decision. Its SHA-256 is
`53d33df167cf40ee19d95bd65311bc8bfa491a930dab17d7df19cef7999516a9`.
The final independent audit rehashed 108 outputs, 31 controls and both 3532-file
source exports. Seventy-one focused CPU cases passed; both whole-suite GPU jobs
passed the same five focused cases without skips. Changed-file Ruff/format/ty
and diff checks passed. The full repository test suite is not claimed green.

Qualification changed exactly two paths from 7003: helper
`a6792c14216dc84d504b4d92c6e0bb94ca432b37d7c99fd742bb701e833d3d4b` and helper test
`cf4e1c211604eb1d26d88ae480f3b23bea02e6772d456d00576c17c72e72a8fc`.
The unchanged canonical harness is `8e9dcd2d…`; candidate manifest is
`579cebaa5d064c451fb655203a7aedcbc91e551312e1bceafe91acdb7d8f4589`.
Final packaging adds only the report, evidence and reference links; runtime and
test bytes match the qualified snapshot. The complete later commit was not
GPU-executed as part of qualification. Other devices, FP16 performance and DDP
were not qualified. Earlier optimization percentages are not summed.

For reproduction, use the [canonical benchmark procedure](chatcore-cache-compaction-20260920.md#sources-checks-and-reproduction)
with isolated 7003/4096 source exports and the exact checkpoint/tokenizer hashes
in the evidence. Preserve the six-process order, fresh cache paths, all 22 calls,
fixed exclusions and conservative comparisons. Full artifacts remain under
`/home/henryac/chatcore-perf-20260920/loop25_qual_23359170` on Engaging.
