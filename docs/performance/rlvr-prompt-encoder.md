# Canonical prompt encoder reuse for RLVR

Canonical prompt reuse is accepted as an **opt-in optimization**, with
`hidden_replay_prompt_encoder=false` by default. Full-budget update medians fell
0.50% and 2.15% in two paired runs; the short workload median rose 6.25%.
The [audited evidence](evidence/rlvr-prompt-encoder-4090.json) records every
sample, source/runtime identity, comparison and qualification boundary.

The measured source is `fc65cf8fa62bbe53407fcce448c044abab5c03ea`, based on
published [history reuse](rlvr-history-encoder.md) source
`e3d98e9a7ea3e55364f098de1abae3fd3c055d2f`. Both arms retain decoder staging,
hidden replay and history reuse. Only the prompt flag and output directory differ.

| Comparison | Reference median | Prompt reuse median | Median change | Mean change | Faster paired updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short, candidate first | 2.406467 s | 2.556986 s | 6.2548% slower | 1.2292% slower | 2 / 5 |
| Initial full, reference first | 24.882276 s | 24.756945 s | 0.5037% lower | 0.4975% lower | 3 / 3 |
| Repeated full, candidate first | 25.268306 s | 24.724323 s | 2.1528% lower | 1.7348% lower | 3 / 3 |

The full pairs replay the same three seeded updates in fresh processes with
reversed process order. Six of six full-budget comparisons improved; these are
repeated observations of one workload, without additional workload diversity or
statistical-significance evidence. Relative to the initial full run, the repeated
reference median rose 1.5514% and candidate median fell 0.1318%. The improvement's
magnitude varies across runs; these observations do not establish its cause.

The RTX 4090 workload uses the native roughly 3B encoder-decoder, BF16
deterministic SDPA, a **linear feature head**, checkpoint step 1116 and cadence
33. Short uses rows `[0,8,16,24]`, G=2, at most 64 GSM8K tokens, one warmup and
five measured updates. Full uses `[0,10,16,26]`, G=4, at most 512 GSM8K tokens,
one warmup and three measured updates. Multiple-choice generation is capped
at 16 tokens. The different row sets define separate workloads.

All eleven fresh measured comparisons match exactly: **264 tensors and 745
floating scalars**, covering gradients, updated parameters and optimizer state,
plus recursively checked discrete trajectories. Separate prompt-disabled parent
controls and original/repeated same-arm controls also match exactly; they are
excluded from those totals and do not establish historical timing gains.
Warmup non-timing receipts match; full warmup tensor snapshots were not saved.

Reuse retains the first admitted canonical prompt in each group. Admission owns
independent snapshots of the three encoder inputs after `.to(device)` and the
five packed output tensors.
Hits require fresh input metadata and content equality, then return defensive
output copies with native dynamic annotations. Empty-history and token-generation
encodings remain native; native history construction/append and the live head
still run. Changed inputs miss without replacing the first entry.

The existing fresh nonempty-history guard protects reuse. Pending entries publish
only after the complete canonical forward passes its postcheck. Prompt, history
and hidden replay share tensor, entry and key caps; hidden capture has priority.
Memo entries clear at collection seal, and successful group replay releases all
owned payloads and tensor objects.

| Worker scope, including warmup | Prompt calls | Hits | Misses / admissions |
| --- | ---: | ---: | ---: |
| Short candidate | 39 | 15 | 24 / 24 |
| Each full candidate | 192 | 176 | 16 / 16 |

Existing semantic history, hidden replay and guard counters match between arms.
All runs report zero evictions, optional-copy OOMs and retained owned objects.
Worker totals are not measured-only hit counts.

| Workload | Reference peak owned cache bytes | Candidate peak owned cache bytes | Peak owned prompt bytes | Measured peak CUDA allocation increase |
| --- | ---: | ---: | ---: | ---: |
| Short | 3,058,242 | 3,273,888 | 757,872 | 759,808 bytes |
| Each full pair | 57,588,704 | 57,912,904 | 757,872 | 326,656 bytes |

Independent payload peaks need not coincide. Owned payload, CUDA allocation and
allocator reservation measure different things; admission headroom is an
estimate, not an OOM guarantee.

Synchronized complete-update timing includes collection, rewards, replay/backward,
clipping, optimizer work and all per-group guards, admissions, equality checks,
copies and cleanup. Persistent decoder setup/cleanup is outside and separately
reported. Loading, checkpoint/restart, validation, telemetry and parity
serialization are excluded. Host guard accumulators are not additive savings.

Focused CPU checks passed **240 unique tests** during source finalization. Pinned
final-source qualification passed **165 native CUDA cases** and **one real RustBPE
depth-two CLI checkpoint/re-exec test**. The CUDA comparison runs three generated
depth-two updates with prior optimizations enabled in both arms, changing only
prompt reuse; all six head gradients/updates, optimizer, RNG, trajectories and
frozen tensors match. Other cases cover multisegment padding, dynamic annotations,
input mutation, pending publication, native errors, injected optional-copy OOM
and shared-cap priority. CLI qualification covers fork state/RNG, evaluation and
uninterrupted-training equality. Depth-two correctness does not establish
depth-two throughput. An earlier CLI expected-settings dictionary omitted the
new flag; correcting that test expectation required no production change, and
fresh CUDA/CLI qualification passed. Production/new-test Ruff and Ty pass; the
runtime test file retains 18 Ty diagnostics reproduced on the parent.

These checks exclude the full repository suite and natural memory exhaustion.
Frozen-model `.data` writes, unregistered independent-version aliases and
concurrent mutation remain outside the guard contract; prompt contents are
freshly compared even after `.data` or alias writes. Full-policy/distributed
execution falls back to native behavior; Hopper/distributed acceleration is
unqualified. The 1 Hz ownership audit found no competing measured compute PID;
startup coverage gaps lie outside measured intervals. Subsecond processes,
graphics load and CPU contention cannot be excluded.

Reproduce with the [native benchmark and comparator](rlvr-head-only-compiler.md#reproduce),
fresh output directories and the workloads above. Retain `compiler_reset=null`,
`train_feature_head_only=true`, `frozen_decoder_max_shadow_bytes=2147483648`,
`frozen_decoder_min_free_bytes=4294967296`,
`hidden_replay_max_cache_bytes=268435456`,
`hidden_replay_min_free_bytes=4294967296` and
`hidden_replay_history_encoder=true` in both arms. Toggle only
`hidden_replay_prompt_encoder=false` to `true`; use `measure_phases=false`,
`profile=false`, short `'parity_steps=[0,1,2,3,4]'` and full
`'parity_steps=[0,1,2]'`. Compare at zero tolerance, reverse full-pair order in
fresh processes, and verify actual hits, unchanged semantic counters and cleanup.
