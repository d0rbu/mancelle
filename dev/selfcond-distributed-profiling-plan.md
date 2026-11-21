# Self-conditioning distributed qualification and profiling plan

Baseline: `origin/main` at `f5853430ff627e518cab1cceda7bad106e1b32f8`.

This implementation qualifies the existing torchrun/`DistMuonAdamW` path and
adds opt-in profiling. It does not introduce a second distributed-training
system, change training semantics, or modify vendored `nanochat/` code.

## Invariants

1. Every rank enters the same optimizer and validation collectives in the same
   order, including ranks with no predicted slots.
2. Exhausted SFT-validation ranks emit fully masked padding until every rank is
   exhausted; slower-rank tail batches remain in the pooled BPB.
3. Profiling is diagnostic only. It does not enter checkpoint/run identity,
   loader state, resume matching, or model/optimizer state.
4. With profiling disabled, actions, conditioning tensors, losses, gradients,
   metrics, model state, optimizer state, and checkpoint metadata remain exact.
5. Timing and trace modes are separate: synchronized CSV attribution is never
   presented as throughput or as an uncontaminated kernel trace.

## Delivery sequence

1. Restore the skipped two-rank Gloo coverage by installing the existing
   test-owned CPU encoder reference inside each spawned worker.
2. Correct the stale finite-iterator SFT-validation test so its fake loader
   matches production's exhausted-rank padding contract and compare BPB against
   an independent all-real-batch oracle.
3. Add one reusable, opt-in, two-CUDA-device NCCL spawn harness and exercise
   canonical head-only/dense plus full/selected self-conditioning steps.
4. Add four strictly validated profile keys to the self-conditioning config.
5. Reuse `steering.profiling.markers` for top-level synchronized timing and
   nested unsynchronized trace markers; do not pass profiler objects through
   model APIs.
6. Add settings, lifecycle, failure, marker, exact-parity, Hydra, identity,
   Gloo, and opt-in NCCL tests, followed by focused and broad validation.
7. Run real 100M/400M artifact and 32-H200 gates only after local correctness
   is green and the required resources/artifacts are revalidated.

## Planned production files

- `scripts/train/conditional_encdec_selfcond.py`
- `steering/conditional/encoder_decoder/selfcond_training.py`
- `steering/profiling/markers.py` (scoped trace-label listener only; see addendum)
- `configs/training/conditional_encdec_selfcond.yaml`

`nanochat/` remains unchanged.

## Addenda discovered during implementation

- The six skipped distributed training scenarios pass when the existing padded
  CPU reference is installed in spawned workers. The parent pytest autouse
  fixture cannot propagate through the `spawn` start method.
- The skipped SFT-validation test encoded the pre-PR-1472 "first rank wins"
  behavior with a finite generator. Production and PR #1472 intentionally wait
  for every rank and use fully masked padding on early ranks. The corrected
  disposable probe executed four iterations on both ranks and matched an
  independent pooled-nats/bytes oracle exactly.
- `docs/pipelines/selfcond-post-training.md` still describes the obsolete
  first-rank-exhaustion behavior even though the shared evaluator uses
  `ReduceOp.MIN`; documentation correction belongs in the qualification layer.
- Current-main's automatic attention selection chooses FlexAttention in this
  CPU environment, and PyTorch rejects FlexAttention backward on CPU. Profiling
  unit/integration tests therefore use the same explicit SDPA fixture as the
  established self-conditioning training tests; CUDA qualification continues
  to exercise the production GPU attention path.
- A real `main()` Chrome trace proved that PyTorch's
  `profile_all_threads=True` mode leaves the private
  `torch.autograd._profiler_enabled()` listener flag false on the entering
  thread. Operator events were captured but every `selfcond/*` record-function
  label was missing. The shared marker layer now has one explicit scoped
  listener for profiler owners; it emits labels without a timing recorder or
  CUDA synchronization. This is instrumentation-only and is pinned by both a
  focused all-thread test and the exported end-to-end Chrome JSON.
- The previously idle node-2433 device 1 was rechecked before use and now hosts
  a live conditioned-SFT process (8.2 GiB allocated, zero GPU utilization in
  three samples, high CPU use). No co-run is safe for performance evidence.
  Self-conditioning now passes its profile session's live marker switch into
  the existing SFT generator so a separate allocation can distinguish loader
  subphases without adding another data-loader abstraction.
- The first combined focused run found test-only launcher-environment leakage:
  direct `_worker_main` unit calls left `RANK/WORLD_SIZE` in the pytest parent,
  so a later nominally local optimizer selected `DistMuonAdamW` without a
  process group. Spawned production workers cannot mutate the parent. The unit
  module now restores all five launcher coordinates, and the downstream
  optimizer reproducer is rerun in the same pytest process.
- The local RTX 4090 full-checkpoint probes use the exact 100M and 400M SFT
  state dictionaries at sequence length 1024. After the cold first-step cost,
  synchronized timing attributed 4.414/7.848 s (100M head-only/dense) and
  4.249/4.444 s (400M full/selected) to rollout. The two arms use different
  STE forms, so this is not a model-size comparison; it is evidence that each
  arm is rollout-dominated and that cold-start and steady-state measurements
  must remain separate.
- Main advanced during validation with the audited W31-to-W32 topology
  migration. The implementation was rebased before remote GPU qualification;
  the only textual conflict was the CLI command table, which now retains both
  the migration entry and the self-conditioning profiling description.
- Main then advanced with the unrelated Chat TUI 3B inference-memory fix. The
  implementation rebased cleanly a second time; changed-file static checks are
  computed from the single post-rebase commit, not from a moving remote ref.
- The repository-wide default pytest collection is dominated by unrelated
  long-running integration cases. At the operator's direction, validation is
  restricted to affected self-conditioning/profiling/distributed/topology
  subsets plus the opt-in real-checkpoint and NCCL gates; an interrupted broad
  run is not counted as passing evidence.
- An early formatter invocation touched legacy layout outside the semantic
  change. Safe formatting-only reversions were admitted only when the Python
  AST remained exactly equal; behavior-affecting indentation and every mixed
  semantic hunk were left unchanged.
- The first real NCCL allocation failed in test setup before collectives: the
  established deterministic perturbation helper is CPU-only, but the new test
  moved the tiny model to CUDA before calling it. The minimal repair perturbs
  the identically seeded CPU model first and then transfers it to each rank's
  device; production code and helper semantics are unchanged.
- The repaired NCCL run passed head-only/dense. Full/selected reached real
  reduce-scatter/all-gather calls, but rank-specific Muon ownership triggered
  a cold Inductor compile that exceeded the original 300-second worker
  deadline; no rank reported a result before the parent deliberately closed
  the store. The harness deadline is now 900 seconds, and cleanup escalates
  join -> terminate -> kill so an Inductor subprocess cannot outlive pytest.
- The next full/selected qualification did not establish collective safety: one
  worker reported while rank 1 timed out after 600 seconds in optimizer
  all-gather sequence 31. It also exposed a concrete confounder that must be
  removed before interpreting that result: each of two ranks inherited all 16
  allocated CPUs and PyTorch's default `min(32, cpu_count)` created 32 Inductor
  compiler workers per rank, leaving 64 workers contending for 16 CPUs. The
  installed PyTorch source identifies `TORCHINDUCTOR_COMPILE_THREADS` as its
  highest-precedence override. Qualification launchers now set it to eight,
  matching both allocations exactly: two ranks on 16 CPUs and eight ranks per
  node on 64 CPUs. This is launch-only resource control, not a training semantic
  change; the capped rerun remains a required correctness gate.
- The compiler-capped two-H200 rerun (`21354035`) closed that correctness
  gate: both real `DistMuonAdamW` NCCL scenarios passed in 84.71 seconds, and
  the exact 100M/400M checkpoint probes then passed in 83.10 seconds. Live
  process inspection confirmed eight Inductor workers per rank, exactly
  matching the allocation's 16 CPUs. The job later failed when its separate
  trainer smoke arm reached an intentionally isolated tokenizer cache; that
  staging failure does not turn the already-completed pytest gates into a
  whole-job success.
- Successive fail-closed trainer preflights exposed, rather than guessed over,
  the complete real SFT dependency set. The staged cache now contains the
  exact tokenizer (`c3945190...e185`), token-byte table
  (`d9b43832...ffa2`), and 1,000-row identity-conversation asset
  (`85a6a963...878e`). The checkpoint's recorded feature path is no longer
  canonical, so the documented explicit override is used only after verifying
  22/22 feature shards, 22/22 parquet shards, 849,656 aligned documents, no
  broken links, and statistics/conversion hashes
  `dadf6d4f...3b5d`/`fe693dd2...ff0`. This is operational staging, not a
  relaxation of production validation.
- The isolated one-H200 real-main-loop job (`21354430`) completed all four arms
  with exit `0:0`. At sequence length 1024, the settled 100M
  head-only/dense step was 466.824 ms: data 179.498 ms, rollout 215.848 ms
  (195.169 ms model forward), pass-1 forward 24.647 ms, pass-2 LM 15.187 ms,
  backward 13.509 ms, and optimizer 0.256 ms. The settled 400M full/selected
  step was 530.222 ms: data 84.526 ms, rollout 298.272 ms (281.575 ms model
  forward), pass-1 forward 38.521 ms, pass-2 LM 24.915 ms, backward 48.029 ms,
  and optimizer 12.590 ms. Regions are nested and must not be summed. Step 0
  and step 1 were dominated by cold/lazy compilation, so neither is reported
  as steady throughput.
- The bounded real 100M trace contains every intended `selfcond/*` phase and
  the SFT construction markers. Even one batch-size-one, one-step capture with
  shapes, stacks, CUDA, and all-thread collection produced a 2,724,199,573-byte
  JSON file and a 35,065,260-KiB process high-water mark. Trace mode therefore
  remains a forensic single-step arm; any future large-world benchmark should
  use synchronized CSV timing and a separate profiling-disabled throughput run.
- Main advanced again before multi-node submission with two AxBench-only
  changes (#1518 and #1520). The rebase was conflict-free, changed-file Ruff
  and `ty` checks passed, and 55 directly affected profiling/DDP/self-conditioning
  tests passed. The trainer SHA remained byte-identical
  (`eb84d464...c881`), but the complete tracked tree was restaged under a new
  commit-named root before submission so indirect imports also match main.
- The exact four-node/eight-H200-per-node pilot was submitted as `21355216`,
  but it never received an allocation, node assignment, or start time. At the
  operator's direction it was canceled while still pending; Slurm reports
  `CANCELLED`, zero runtime, and no node. No 32-GPU behavior or performance
  claim is made. The completed acceptance evidence is limited to local RTX
  4090 real-checkpoint probes, real two-H200 NCCL tests, and the isolated
  one-H200 main-loop/profile run.
- Before merge, main advanced through #1524, #1510, #1523, #1527, #1522, and
  #1529, including material conditioned-SFT selection, lazy feature-index, and
  loader changes consumed by the self-conditioning path. The branch was
  rebased rather than relying on GitHub's conflict-free merge alone. Changed-file
  Ruff and `ty` passed, and 103 focused self-conditioning/DDP/profiling plus
  current-main SFT identity/vectorization tests passed. This expands the
  integration test scope; it does not change production semantics.

Further addenda must state the evidence that changed the plan and whether the
change affects semantics, test scope, or only implementation mechanics.
