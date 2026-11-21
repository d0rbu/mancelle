# Canonical history encoder reuse for RLVR

This optional optimization reuses frozen feature-encoder outputs for repeated
historical actions during canonical behavior collection. It leaves prompt
encoding, token generation and the live feature head unchanged. Set
`hidden_replay_history_encoder=true` alongside a positive
`hidden_replay_max_cache_bytes`; the default remains false.

The measured source is `db7af6728371880fd18b43eb57e7e9b030a161a7`, stacked on
[hidden replay](rlvr-hidden-replay.md). Both benchmark arms retain hidden replay
with a 256 MiB budget and frozen decoder staging with a 2 GiB budget, each with
4 GiB estimated headroom. Only the history flag and output directory differ.

## Complete-update results

The full workload median fell **7.32%**; the short workload median rose
**2.31%**. Both pairs passed exact numerical comparisons and independent
evidence audits. Reuse remains opt-in because its benefit depends on history
repetition. [Machine-readable evidence](evidence/rlvr-history-encoder-4090.json)
contains every sample, configuration, source/runtime identity and comparison.

| Workload | Reference median | History cache median | Median change | Mean change | Faster paired updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 2.440778 s | 2.497250 s | 2.3137% slower | 0.3272% slower | 3 / 5 |
| Full | 27.136654 s | 25.150304 s | 7.3198% lower | 7.5467% lower | 3 / 3 |

The short workload is a regression in this pair. Its ten repeated history
encodes all occur in one measured update; the other four updates offer no
history reuse. Both the median and mean are reported, together with every
paired latency in the evidence. These are one fresh process per arm with fixed
seeds, not randomized repeated trials or a reference/reference noise estimate.
Do not infer a universal speedup or a gain on other hardware.

## Workload and correctness

The RTX 4090 run uses the native roughly 3B encoder-decoder, BF16 deterministic
SDPA, the linear feature head, checkpoint step 1116 and cadence 33. The short
pair uses rows `[0,8,16,24]`, G=2, maximum 64 GSM8K tokens, one warmup update and
five measured updates. The full pair uses rows `[0,10,16,26]`, G=4, maximum 512
GSM8K tokens, one warmup and three measured updates. The row sets differ, so
these are separate workloads. Multiple-choice generation is capped at 16
tokens in both.

Short receipts cover 638 generated tokens and 53 live feature actions across
five measured updates, including three updates with nonzero gradients. All
120 tensor comparisons and 163 floating scalar comparisons are exact at zero
tolerance. The comparator also checks discrete trajectory structure. A separate
exact comparison verifies the new history-disabled reference against the saved
qualified predecessor; no historical timing comparison is used.

Full receipts cover 4,575 generated tokens and 176 live feature actions across
three measured updates, all with nonzero gradients. Its 72 tensors and 291
floating scalars match exactly, as does the separate parent/reference control.
Across both fresh pairs, eight measured updates match in 192 tensors and 454
floating scalars; historical controls are not added to that total. Warmup
non-timing receipts match; full warmup tensor snapshots are not saved.

Focused CPU validation passed 231 unique tests. The final measured source
passed 116 native CUDA tests (42 new history cases and 74 existing replay
cases) and the real RustBPE depth-two CLI checkpoint/re-exec, fork, evaluation
and uninterrupted-training case. The history CUDA suite keeps decoder staging
and hidden replay enabled in
both arms for three generated depth-two updates, changing only history reuse.
It checks exact trajectories, RNG, all six head gradients/updates, optimizer
state and frozen parameters. Additional cases cover packed counts 32/64/96,
complete padded bytes and dynamic annotations, state mutation between behavior
calls, failed canonical calls, partial optional-copy OOM and shared-budget
priority. These are scoped checks, not the complete repository suite or a
natural memory-exhaustion experiment. Two compacted-encoder policy cases
required CUDA and were skipped in the CPU subset; one case was deselected.
Initial Gloo socket failures passed on retry with local socket access.
Production and new-test Ruff/Ty pass; the runtime test file retains the same
18 Ty diagnostics reproduced on the parent source.

## Ownership, limits and measurements

Each group owns exact ordered action-tuple keys and independent copies of all
five packed output tensors. Hits use defensive copies with native dynamic
annotations, while native action tensor construction and append still run.
The original frozen-state guard runs once per nonempty history operation;
new entries publish only after the canonical forward's successful postcheck.
History shares tensor, key and entry limits with hidden replay and yields
capacity to hidden-state admission. Entries are released at collection seal;
all successful groups have zero retained owned tensor objects.

Short worker counters include warmup: 25 history actions, 15 admitted entries,
10 hits, 15 additional guard calls and zero evictions, skips, optional-copy OOM
or retained objects. Peak owned history payload was 751,072 bytes; peak total
cache payload grew from 2,307,170 to 3,058,242 bytes. The observed peak CUDA
allocation delta was zero. These payload figures differ from allocator
reservations and do not establish a memory-recovery guarantee.

Full worker counters include warmup: 1,147 history actions, 971 hits, 176
misses/admissions and 176 additional guard calls, with zero evictions, skips,
optional-copy OOM or retained objects. Peak owned history payload was 7,081,536
bytes; peak total cache payload grew from 51,258,240 to 57,588,704 bytes. These
separate peaks need not occur together. Peak CUDA allocation rose by 7,021,568
bytes. The existing hidden replay counters are unchanged: 192 captured states
and 191 hits, including one state without a live replay.

All admission, guard, equality, clone, live-head and group-cleanup work remains
inside complete-update timing. Persistent decoder setup/cleanup is outside and
reported separately. Worker guard counters include warmup; host accumulator
costs must not be subtracted from GPU-synchronized update timings. The 1 Hz
ownership monitor observed no competing measured compute PID in both
pairs, with measured coverage within 1.5 seconds. It cannot exclude subsecond
processes, graphics load or CPU contention. Frozen-model `.data` writes,
unregistered independent-version aliases and concurrent mutation remain
outside the existing frozen-state contract. Distributed/full-policy runs use
native behavior; no distributed or Hopper acceleration is established.

## Reproduce

Use the [native benchmark and exact comparison](rlvr-head-only-compiler.md#reproduce)
with identical admitted inputs and fresh output directories. Retain
`compiler_reset=null`, `train_feature_head_only=true`,
`frozen_decoder_max_shadow_bytes=2147483648`,
`frozen_decoder_min_free_bytes=4294967296`,
`hidden_replay_max_cache_bytes=268435456`, and
`hidden_replay_min_free_bytes=4294967296` in both arms. Change only
`hidden_replay_history_encoder=false` to `true`. Use the row sets, group sizes
and token budgets above, `measure_phases=false`, and `profile=false`. Save
every measured update with short `'parity_steps=[0,1,2,3,4]'` and full
`'parity_steps=[0,1,2]'`, and compare at zero tolerance. Verify actual history
hits, unchanged hidden replay coverage, and zero retained owned objects after
each successful group.

Complete-update timing includes collection, reward, replay/backward, clipping
and optimization. Loading, checkpoint/restart, validation, telemetry and receipt
serialization are outside this compute-update measure.
