# ChatCORE FP8 cache compaction — 2026-09-20

The Ada expandable ragged cache now compacts its K/V rows with one paired
Triton launch per layer. Qualified warm complete-suite latency fell from
**9.6541 to 9.0618 seconds**, a **0.5922-second / 6.13% reduction** on one L40S.
Both candidate processes beat the faster reference process by **5.90–6.20%**
using warm geometric means, and **5.90–6.24%** using medians. Completion bytes,
HumanEval verdicts, task scores, and sampled logits matched exactly.

The reference is PR #1731 at
`f95f459d0ab99ead8c63cce03973639931fce709`. Both arms retain its two-worker
HumanEval grader and use the unchanged canonical benchmark. This result is
specific to the workload and allocation below; gains from earlier PRs are not
added to it.

## Behavior and overlap ordering

The helper admits the exact expandable-cache class during inference, with
contiguous E4M3 K/V tensors and contiguous int32 positions on one CUDA device
of compute capability 8.9. Dimensions must be positive integers; the keep list
must be nonempty, strictly increasing, in range, and shorter than the batch.
All complete backing allocations must be disjoint. Unsupported metadata,
layouts, subclasses, dtypes, devices, and partial-growth views retain the
original method body. Non-Linux imports retain the portable path.

Each program owns disjoint byte coordinates across every row. Within those
coordinates it visits destination rows in increasing order. For increasing
keep indices `s`, `s[k] >= k`: an earlier destination cannot destroy a later
source, but a later destination can overlap an earlier source. Preserving that
order within each owner is necessary; independent destination-row programs
would be unsafe. Integer loads/stores copy FP8 bytes without conversion.

Compaction covers the entire allocated logical row capacity, including unused
tails. It retains backing allocations, storage offsets, prefix-view semantics,
and the original scalar position-copy order, with legacy-equivalent bytes.
There is no full-cache temporary or persistent index cache. The outer caller
retains its validation and original full-length early return. Unsupported
direct-method inputs fall through before mutation, preserving the original
error and partial-mutation behavior. Admitted allocation/launch errors
propagate without retrying over potentially mutated buffers.

## Discovery and resident screen

The retained f95 short profile from job `23295288` identified one 150-to-149
compaction at capacity 411: 4,316 K/V copies and 83 position copies. Device
copies occupied 13.8417 ms inside a 47.1289-ms span. These instrumented numbers
identify a target, not achievable suite savings. A separate 511-state canonical
census at capacity 859 reconstructed 70 compactions and 5,122 moved rows; it
does not measure full-suite compaction latency.

The [component screen](evidence/chatcore-cache-compaction-component-20260920.json)
(`23338844`, L40S/node2804) used two synthetic raw-byte fixtures with the exact
first/late census shapes, mappings and positions, retaining the full 150-row
backing in both. Complete synchronized compaction latency fell from 32.1700 to
22.8635 ms (28.93%) for 150-to-149, and from 18.6046 to 12.7553 ms (31.44%) for
50-to-49. Every comparative quartet was positive; maximum absolute
same-implementation control difference was 0.0862%. All 320 timed calls and
32 excluded warmups were retained, with restoration outside each interval.
These are component measurements with synthetic payloads.

Seventeen tiny cases comprising 19 compaction steps/attempts and both full-size
fixtures passed byte/storage/metadata checks. Independent CPU replay covered
persisted tiny storages; the full-size arrays were checked on GPU and were not
persisted for CPU replay. Manual PTX/TTGIR/LLVM review bound six saved last-launch
compiled variants/artifact bundles; cubin hashes were checked without SASS
disassembly. The integrated kernel is byte-identical to the screened kernel;
source normalization accounts for class-dependency and boolean-return plumbing.

The [resident screen](evidence/chatcore-cache-compaction-resident-20260920.json)
(`23341075`, L40S/node3002) ran four excluded warmups, 32 comparative complete
calls in eight randomized ABBA/BAAB quartets, and eight same-implementation
control calls. Geometric means fell from 9.8064 to 9.2183 seconds (**6.00%**).
All eight comparative quartets were positive. Orientation estimates were
5.8050% and 6.1904%; first/second halves were 5.9246% and 6.0712%; leave-one-quartet-out
estimates were 5.8773–6.0818%. Control differences were 0.7652% and 0.1057%.
Every frozen advance condition passed without sample removal or a percentage
threshold. These are observations on one resident allocation, not confidence
intervals or independent process trials.

All 44 resident suites retained baseline artifacts and scores. Each first
excluded warmup recorded 511 decode calls and 70 compactions. The candidate
admitted all 70, launching 1,820 layer kernels without fallback; the reference
used neither helper nor kernel. All 53 backing allocation descriptors stayed
stable at every compaction. Both sampled-logit arrays matched historical f95
bytes. Witnesses were removed before comparative timing. These route counts
are resident evidence; the later qualification adds no runtime witnesses.

## Fresh-process complete-suite qualification

Engaging job `23343419` completed with exit `0:0` in 7:18 on one NVIDIA L40S on
`node3003`, using `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, eight CPUs
and 64 GiB host memory. Software was PyTorch `2.13.0+cu130`, Triton `3.7.1`,
and driver `590.48.01`. The d26 SFT step-999 checkpoint has 26 layers, width
1664, 13 attention/KV heads and context 2048; its SHA-256 is
`d3d320a5b4e3985a6915d70d51f63f4500ace6c12bc7bbd867fec74ab50f7371`.

Every suite includes 50 problems per task across ARC-Easy, ARC-Challenge, MMLU,
GSM8K, HumanEval and SpellingBee, with the production 512-token generation
budget. Five focused GPU tests ran first. Six separate model processes then
ran the fixed sequence `R5, C5, C5, R5, R1-logits, C1-logits`, with distinct
compiler caches for all seven phases. Each child had a 300-second wall limit.
No phase was repeated or added after observing results.

CUDA synchronization brackets each complete `test_chatcore` call. Admission,
capability lookup, index allocation/H2D, kernels, position copies, view updates,
and grading remain inside the timer. Model loading, source/input hashing and
artifact serialization are outside it. The canonical completion/grade capture
is common to both arms. Only repeats 1–4 of the four five-suite processes enter
primary timing; all four first calls and both instrumented audits are retained
and excluded by precommitment.

| Process, execution order | Warm geometric mean | Warm median | Excluded first call |
|---|---:|---:|---:|
| Reference 1 | 9.663067 s | 9.662669 s | 15.887843 s |
| Candidate 1 | 9.047373 s | 9.044999 s | 15.208375 s |
| Candidate 2 | 9.076286 s | 9.076968 s | 15.498626 s |
| Reference 2 | 9.645072 s | 9.646537 s | 15.924754 s |

The primary ABBA estimate is **6.1347% / 0.59225 seconds**. Adjacent AB and BA
reductions are 6.3716% and 5.8972%. Both candidate geometric means and medians
beat both references, passing the prespecified conservative rule. Reference
geometric means drifted by −0.1862%, candidate means by +0.3196%; corresponding
median drift was −0.1669% and +0.3534%. No correction was applied for drift.
All candidate warm samples (9.0340–9.0856 s) were faster than all reference warm
samples (9.6265–9.6647 s). Repetitions within a process are not independent
trials; this is one fresh allocation, not a universal percentage or confidence
bound. First calls describe fresh processes/configured compiler caches, not
globally cold hardware, driver or filesystem state.

All 22 qualification suites retained identical completion/grade JSON bytes and
scores: 150 generated rows, 42,089 tokens, 50 HumanEval verdicts with three passes,
one 50-program/two-worker batch, zero scalar grades, and ChatCORE 0.2867.
Each numerical audit compared 1,048,576 finite FP32 logits and 705 input IDs at
eight decode calls (0, 64, …, 448), byte-for-byte against historical f95 and
each other. This samples actual checkpoint logits; it does not compare every
checkpoint cache/logit value. Their excluded times were 15.8358 and 15.1081 s.

Peak allocated CUDA memory was 18,571,301,888 bytes in every call. Warm reserved
peaks matched at 28,399,632,384 bytes. First-call/audit reserved peaks were higher
for the candidate: 26,533,167,104 versus 24,301,797,376 bytes (about 2.08 GiB more).
This change does not establish memory reduction or general memory parity.
All 383 ownership samples showed no unrelated GPU process; 278 showed owned
GPU activity. Sampling cannot rule out subsecond interference. Process-tree
RSS can miss short workers and double-count shared pages, so it is not an exact
aggregate host-memory peak. Performance on other GPUs, FP16 or DDP was not
qualified by this run.

## Sources, checks and reproduction

The [qualification evidence](evidence/chatcore-cache-compaction-20260920.json)
retains every call, process contrast, source/input pin, memory observation,
independent audit and decision. Its SHA-256 is
`1234feda6232a7e5748d21ec713ddbebed57ca0036cebe1685edd9698453bc2d`.
The frozen independent auditor passed unchanged, rehashing 108 output files,
28 controls, both source manifests and archives. Full artifacts and controllers
remain under `/home/henryac/chatcore-perf-20260920/loop19_qual_23343419` and the
parent campaign directory on Engaging.

Qualification used a stable 3,528-file snapshot: exactly four runtime/test
deltas from the 3,525-file f95 parent, with the unchanged canonical harness.
The source-manifest SHA-256 is
`9a38ba64826071a99ff8db53a11f1c06c89fc82d77ccb8e6eb90ab7d8991f273`;
the source-archive SHA-256 is
`a91cc66d326d61a692d7d3bac47987f51dfcb4ee129deebfe517f41019388c4f`.
Final reporting adds documentation/evidence only. Runtime, helper, tests and
canonical harness must match that qualified snapshot; the entire later commit
was not executed on GPU.

All 71 focused CPU cases passed (27 new and 44 existing). Both resident and
qualification allocations passed the five focused GPU cases without skips:
three raw full-backing byte cases and two tiny GPT generator cases with
nontrivial coefficients, BF16/FP16 and repeated compaction. Test-only terminal
forcing occurs after saving actual model logits. CPU coverage includes malformed
keeps, fallback layouts, aliasing, subclasses, offsets, partial errors, and
non-Linux imports.

Changed-file Ruff/ty and diff checks pass. Paired repository checks used the same
tools/environment: Ruff retained the same two findings; ty retained byte-identical
logs with 446 diagnostics; Vulture passed; the production-caller ratchet retained
the same four symbols (one failed test, four passed). No changed file was
implicated. The full repository suite was not run or claimed green.

For CPU reproduction, use `CUDA_VISIBLE_DEVICES='' NANOCHAT_DTYPE=bfloat16` and
the default non-slow selector on `tests/steering/test_greedy_token_batch.py`,
`tests/steering/test_triton_kv_cache_compaction.py`, and
`tests/steering/test_greedy_cache_compaction_gpu.py`. Run the latter two with
`-m slow` on an Ada CUDA device for the five GPU cases.

For timing, isolate complete parent/candidate source snapshots and use identical
checkpoint/tokenizer assets. Set `NANOCHAT_BASE_DIR` and choose fresh output,
`TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` paths for every process:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --no-sync python -m scripts.benchmark.benchmark_chatcore \
  checkpoint_dir=/path/to/d26/standard_sft_checkpoints step=999 \
  output_dir=out/chatcore/unique-process grading=candidate \
  repeats=5 record_logits=false profile=false
```

Follow reference/candidate/candidate/reference order, retaining every observation
and excluding exactly repeat 0 from each warm process summary. Run two separate
`repeats=1 record_logits=true profile=false` audits. Compare actual completion,
grade and sampled-logit files; score agreement alone is insufficient for this
byte-preserving optimization.
