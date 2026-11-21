# ChatCORE bounded parallel grading — 2026-09-20

Standalone ChatCORE now grades a completed HumanEval chunk with at most two
fresh execution processes. On the qualified L40S run, the two candidate process
medians were **3.13–3.50% faster than the faster reference process**. All tested
completion tokens, program hashes, verdicts, scores, and sampled logits matched.
The reference is PR #1729 at `159f67be5d6b6a937fc30119600087c63071c86b`.

## Scope and execution contract

Parallel admission requires Linux, the existing `fork` context, the main thread
of a non-daemon process, one rank, and no initialized distributed process group.
Actual CPU affinity must permit two workers plus the parent. A declared
`SLURM_CPUS_PER_TASK` caps that budget; a Slurm job without that declaration
retains scalar grading. Unsupported or insufficient CPU/process contexts and
singleton HumanEval chunks also retain scalar grading. DDP remains scalar.

Every program uses a fresh, unchanged `nanochat.execution._unsafe_execute`
worker with its temporary directory, reliability guard, execution alarm, and
default 256 MiB memory limit. Each parent deadline starts after that child's
`start()` returns and allows the existing `timeout + 1` seconds. Queue time
does not consume the allowance. A still-live expired worker returns false even
if it published success before hanging during cleanup. Startup and parent
errors propagate after cleanup; workers are killed, reaped, and closed as needed.

A slow `fork` can delay observation of another child's deadline. Concurrent
scheduling can change outcomes near time limits; finite parity tests do not
establish universal equivalence for deadline-sensitive programs. The supervisor
checks live workers before every launch and never deliberately extends a deadline.

Scalar and batch paths share exact program construction. Submission order and
duplicates are preserved, and all graders finish before the next generation
chunk or collective. Other task rows precede the deferred HumanEval batch within
a mixed chunk, so malformed-input exceptions can change which earlier rows
have run. Successful row mapping and task totals are preserved. Vendored code
is unchanged.

## Evidence and complete-suite comparison

An L40S resident-model probe (`23291953`) attributed 0.449 seconds of a separate
50-program grading profile to joins and 0.267 seconds to process starts. The
subsequent [replay screen](evidence/chatcore-parallel-grading-screen-20260920.json)
(`23292661`, another L40S allocation) ran three serial/parallel/parallel/serial
cycles on the same 50 generated programs. Replay medians fell from 3.1584 to
1.5408 seconds (51.22%). This is a grading-only result, excluding generation;
absolute timings from the two different probe hosts are not a controlled delta.
All 164 canonical solutions and 50 generated programs retained their ordered
verdicts: 164 canonical and three generated programs passed.

Complete-suite job `23295288` completed with exit `0:0` in 8:50 on `node3205`,
using one NVIDIA L40S, `mit_normal_gpu`, and account/QoS `mit_amf_advanced_gpu`.
Software was PyTorch `2.13.0+cu130`, Triton `3.7.1`, and driver `590.48.01`.
The d26 SFT step-999 checkpoint SHA-256 is
`d3d320a5b4e3985a6915d70d51f63f4500ace6c12bc7bbd867fec74ab50f7371`.
Full manifests cover 3,520 reference files and one additional candidate test.
Both arms use the same new harness and `grading=candidate`; runtime differences
are confined to the grader, HumanEval task integration, and standalone runner.

Each complete suite includes 50 problems per task across ARC-Easy,
ARC-Challenge, MMLU, GSM8K, HumanEval, and SpellingBee, with the production
512-token generation budget. Fresh processes run in reference/candidate/
candidate/reference order, with fresh Triton/Inductor cache directories.
Each process runs one first suite and four warm suites. CUDA synchronization
brackets complete calls. Loading the model, hashing inputs, writing artifacts,
profiling, and logit capture are outside the warm timing. Datasets are offline.

| Complete-suite measurement | Reference | Candidate |
|---|---:|---:|
| Warm process medians, in execution order | 10.0286 / 10.5175 s | 9.6774 / 9.7149 s |
| Pooled warm median, eight suites per arm | 10.2680 s | 9.6910 s |
| Warm range, all samples retained | 10.0108–10.5459 s | 9.6702–9.9293 s |
| First suite in each fresh-cache process | 19.4568 / 16.5312 s | 15.8376 / 15.8219 s |
| Peak allocated CUDA memory | 17.30 GiB | 17.30 GiB |

Every candidate warm sample was faster than every reference warm sample. The
reference process medians nevertheless drifted by **4.87%**. The pooled reduction
of 0.5770 seconds (5.62%) is therefore secondary to the conservative block
comparison: both candidate medians are 3.13–3.50% below the faster reference
median. These are observed comparisons, not statistical lower bounds. Repeats
within one process are not independent process-level trials. This qualifies a
positive warm improvement for this workload/allocation, not a stable universal
percentage or a general cold-start or DDP performance claim.

All 22 timing/audit suites produced the same 42,089 completion tokens and bytes,
50 program hashes/verdicts, and six-task scores (ChatCORE 0.2867). The candidate
graded all 50 programs through its two-worker batch path; the reference used
50 scalar calls. Separate audits compared 1,048,576 full-vocabulary logit values
at eight decode calls, with zero maximum absolute difference (`atol=rtol=0`)
and identical input IDs. Their instrumented timings are excluded. All 380 GPU
process samples showed no unrelated work, with matching CUDA/monitor UUIDs.

The separate final short profile recorded one batch of 50 programs, using at
most two concurrent workers and 50 fresh worker starts. It recorded 0.694 seconds
in the batch grader and 0.598 seconds in process starts. This confirms route
selection; it is not a paired full-suite performance measurement.
Host lifetime `ru_maxrss` can include compiler/dataset processes. One-second
process-tree RSS snapshots can miss short workers and double-count shared pages;
neither establishes an exact aggregate host-memory peak.

The [comparison receipt](evidence/chatcore-parallel-grading-20260920.json)
retains every timing, source/input hash, memory observation, score, and route
receipt. Its SHA-256 is
`10dc2bc3bc9ffa8977a91f9b782e2d10a90413a4ca5a1fb753e7840428193a2f`.
Full artifacts and controller/manifests remain in
`/home/henryac/chatcore-perf-20260920/loop4_job_23295288` and its parent directory.

## Tests and startup boundary

All 130 focused local tests passed: 40 worker tests, 54 task/admission/integration
tests, and 36 existing caller tests. The cluster passed 88 preflight tests plus
two supplemental CPU admission tests after all timed/audit processes exited.
The broad preflight selector had also omitted those two mocked admission cases;
the supplement restores coverage without changing any timed code or sample.

Earlier preflight `23293349` stopped before timing with four default-timeout
positive cases failing under `spawn`/`forkserver`, in both scalar and batch
graders. A [paired startup diagnostic](evidence/chatcore-grading-contexts-20260920.json)
on the exact parent and candidate (`23294415`) confirmed the shared boundary:
all 14 default-timeout workers were killed around six seconds before their early
program marker, whereas all 14 workers with a diagnostic 30-second allowance
reached that marker after 10.12–12.61 seconds and returned the expected verdicts.
All tested execution workers were reaped. The marker is not an exact worker-entry
timestamp, and individual import costs were not measured. Production deadlines
and limits were unchanged. Those four remote non-fork positive cases remain
failed/unqualified and are excluded from the fork-only qualification.

Changed-file Ruff/ty, repository Vulture, and diff checks pass. Broad Ruff/ty
logs exactly match the parent findings (two Ruff findings and 442 ty diagnostics),
and the production-caller gate retains its same four baseline findings. The full
repository is not claimed green.

To reproduce the timing, use the same new benchmark harness in isolated parent
and candidate snapshots with `grading=candidate repeats=5`, fresh cache/output
directories, and separate `record_logits=true repeats=1` audits as described in
[profiling](profiling.md). Require all 50 ordered grade records and nonzero
candidate batch receipts. The parent snapshot intentionally lacks the new batch
symbol, which the common harness handles without changing its scalar path.
