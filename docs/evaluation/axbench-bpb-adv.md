# AxBench-BPB-Adv Evaluation

AxBench-BPB-Adv is an adversarial robustness eval over AxBench-Base Stage 0-3
artifacts. Adversarial prompt search is still BPB-based: Faster-GCG / GEPA build
fixed attacked prompts by degrading the per-prefix BPB class margin. The default
reported steering y-axis is now **roundtrip steering** (`steering_metric:
roundtrip`): after a candidate prompt is fixed, the steered target model
generates a completion and the prompt + completion is probed with the original
SAE model to measure whether the original concept feature reactivates in the
completion.

The legacy BPB class-margin y-axis remains available with
`steering_metric: bpb_class_margin`.

This Faster-GCG campaign remains distinct from sampled-generation
AxBench-Adv's opt-in `gcg` pressure. BPB-Adv searches and reports its BPB or
roundtrip surfaces; AxBench-Adv trains one suffix from prepared concept pairs,
then samples and applies the three-axis LLM judge only to the retained suffixes.
Their artifacts and merge roots are not interchangeable.

For the bounded real-checkpoint clustered acceptance, including the exact
assets, mutation gate, resume contract, and retained evidence, see
[`clustered-gpu-acceptance.md`](clustered-gpu-acceptance.md).

## Default metric: roundtrip steering

Roundtrip scoring uses the shared implementation in
`steering/eval/roundtrip_steering.py`. For each candidate/sample, the scorer:

1. Applies the same steering intervention used by BPB scoring.
2. Generates a new completion from the attacked prompt.
3. Generates a paired control completion for the same adversarial candidate:
   no base steering, plus the same additive counter-direction when the candidate
   is an additive countersteering row.
4. Re-tokenizes each exact generation prompt text plus generated response with
   the probe model tokenizer.
5. Masks response tokens only, preserving BOS behavior exactly as the shared
   roundtrip scorer defines it.
6. Probes `Concept.feature_index`, the original SAE feature for the concept.

Round-trip probing is inference-only: the model activations and downstream SAE
probe math do not record gradients. Production probes are still loaded frozen
for memory efficiency, but hand-built probe models do not need frozen SAE
parameters to use this scorer.

The primary y-axis is a mode-normalized activation percentile:

```text
positive mode:  primary_roundtrip_y = roundtrip_percentile
negative mode:  primary_roundtrip_y = 100 - roundtrip_percentile
```

In both modes a higher value means stronger steering in the requested direction.
`control_primary_roundtrip_y` is computed from the paired control completion at
the **same adversarial strength**. It is never copied from the strength-0 row.
For additive countersteering this means both the primary and control roundtrip
signals are re-evaluated at every counter-direction magnitude. The roundtrip
artifact also carries `loss_concept_rich` and `loss_concept_avoiding`, computed
with the existing BPB scorer, so the roundtrip scatter can use the same
baseline-loss x-axis as the legacy BPB plot.

Roundtrip scoring is more expensive than the legacy BPB metric because the score
stage performs generation plus SAE probing. The implementation generates with
the target model first. For each candidate side (steered and control), all
`samples_per_candidate` completions use one engine construction, one
intervention entry, one shared prompt prefill, and one RNG that samples all
active rows in a single vectorized call per iteration. When rows stop, generation
removes them from the active batch and copies only surviving KV-cache rows into a
fresh uniformly positioned cache; no filler-token decode is performed in the
tail. The wrapped encoder-decoder path expands its stored batch-one encoder state
to the shrinking active batch on every decode call.

PR7 deliberately waived per-row RNG stability. Steered and control cache keys
remain per-sample, including `sample_index`, so deduplication and twin-sharing
semantics are unchanged. Each candidate side instead derives one batch seed from
the same identity payload with `sample_index` removed, records that seed on every
row, and passes it to the single batch RNG. This reseeds sampled completions and
all downstream probes relative to pre-PR7 score tables. Prompt/identity/editable
span columns, teacher-forced losses, row schema, and row order remain exact. Use
`compare_roundtrip_score_tables --expect-reseeding` to report the sanctioned
generation/probe drift while hard-gating those surviving invariants.

Control generation is cached by its resolved semantic determinants rather than
the complete candidate row: target model; exact control prompt tokens;
the resolved context provider's ordered segment content (feature IDs, buckets,
and character bounds), or an explicit decoder-only marker;
`max_new_tokens`, temperature, and top-k; sample index; and the resolved control
`Intervention` kind, sorted feature conditions, direction layer/coefficient,
LoRA path, prompt-prefix tokens, counter layer/coefficient, and counter
provenance. Direction coefficients and active-counter adversarial strengths are
identified by their exact binary64 bits. Active counter provenance consists of
adversarial method, strength, mode, and concept ID; an absent counter uses an
explicit `no_counter` marker. The cache key retains the complete semantic key,
while the batch seed is derived after removing only `sample_index`. Rows sharing
every determinant still share completions per sample index; all rows generated
in one candidate-side batch record the same seed.

An offline audit of the real 16,632-row `m_gcg2` candidate table measured 1,386
unique control batches / 5,544 unique completions, a 12.00x completion reduction:
every audited adversarial strength was 0.0 and resolved to an explicit
additive-counter no-op, so method and concept provenance could not affect the
control. The `axbench_bpb_adv.roundtrip.control_unique_batches` and
`axbench_bpb_adv.roundtrip.control_unique_completions` counters report those two
units separately. Probe scoring separately memoizes exact
`(prompt text, response text, feature IDs)` requests for the duration of one
scoring call. The probe model and percentile table are fixed for that call, so
this removes bit-identical recomputation with zero numerical or semantic drift.
On resume, committed probe rows reconstruct and pre-seed the same memo before
fresh candidates are visited, so restored/fresh twins share work in either
candidate ordering.
Requested probes continue to count before memo lookup; cache-hit and unique-text
counters expose the deduplication.

The stage always unloads target state for the intra-stage
target-to-probe handoff, then loads the probe model for roundtrip scoring to
avoid co-resident target/probe memory.

### Automatic round-trip score resume

Model-backed round-trip scoring checkpoints every candidate automatically.
Checkpointing is intentionally not a configuration surface: it is always on,
adds no Hydra fields, and does not change the run-tag identity. There are two
durable units per `candidate_id`: `generation` stores all
`samples_per_candidate` generation rows before probing, and `probe` stores that
candidate's final score rows after empty-completion filtering and SAE probing.
Each completed unit is installed before work advances to the next candidate.

The immutable store lives at `roundtrip_checkpoint/`. Its canonical
`manifest.json` binds the exact consumed `candidates.parquet` bytes; evaluation
population and source contracts; the cluster feature-contract fingerprint (or
explicit null); resolved target checkpoint source, tag, step, model bytes, and
metadata bytes; the nanochat tokenizer tree; LoRA and direction semantics;
conditional encoder-context and unclustered precomputed-bag trees (clustered
bags are transitively pinned by candidate metadata); both feature-statistics
files; extraction-tokenizer content; the full roundtrip-generation block and
target reference segment sizes; the complete probe config; exact SAE weight
content; the probe base-model snapshot revision and config bytes; probe feature
chunk size; the store algorithm version; and the ordered expected candidate-ID
count and digest. Generation and
probe shards are content-addressed Parquet under
`<family>/groups/<two-hex-bucket>/cand_<candidate-key-sha256>_<payload-sha256>.parquet`.

On restart the store validates the whole manifest and every committed shard in
one pass. Restored generation candidates still rebuild the cheap intervention
and generation inputs, assert their prompt tokens/text/digests against the
shard, and pre-seed both the steered identity cache and semantic control cache;
they perform no model generation. This lets a later fresh semantic twin reuse a
restored completion. Restored probe candidates perform no probe forward. Fresh
candidates write through after each family completes, and final assembly always
follows candidates-table order then `sample_index`, so a resumed final Parquet
table is byte-identical to an uninterrupted run under the same runtime.

PR7 bumps the fingerprinted store algorithm from checkpoint v2 to v3 because
restored-row validation recomputes the new batch seeds. A pre-PR7 checkpoint is
therefore rejected by the normal manifest fence, naming
`store_algorithm_version` and requiring a fresh `run_tag`; stale generation
shards are never mixed with the new reseeding semantics.

Any manifest mismatch, malformed/truncated shard, digest or bucket mismatch,
unknown candidate, missing/duplicate sample, duplicate payload, schema drift,
symlink, or changed re-commit raises and tells the operator to preserve the old
run and choose a fresh `run_tag`. State is never silently overwritten, adopted,
or recomputed under mixed semantics. Runtime numeric fingerprints are recorded
per shard but are diagnostic rather than a semantic fence. Generation shards
use the resolved target device and probe shards use the independently resolved
probe device; mismatches warn separately per family and allow restore, while the
byte-parity guarantee applies only to matching family runtime fingerprints.
Shards remain after success, the final `roundtrip_scores.parquet`
is the sole downstream authority, and exactly one writer per `run_tag` is the
supported contract.

Resume accounting separates durable history from current-process work:
`checkpoint_restored_generation_candidates` and
`checkpoint_restored_probe_candidates` report the restart split, while sample,
probe, unique-work, and cache-hit counters describe requests and reuse in the
current process (including fresh twins that reuse restored cache entries).

## Legacy metric: BPB class margin under one steered intervention

The metric mirrors the AxBench-BPB class margin. A *core prefix* is a
`(concept_id, mode, prefix_text)` triple, identified by
`core_prefix_id = sha256(concept_id|mode|prefix_text)[:16]`. Each core prefix
owns **both** continuations of that prefix:

- `concept_rich`: the concept-favoring continuation.
- `concept_avoiding`: the contrast continuation.

Both completions are teacher-forced under the **same** steered intervention (the
evaluated steering signal plus any adversarial pressure). There is no
neutral/reference side and no second, unsteered pass. The eval records:

```text
loss_concept_rich     = BPB(concept_rich     | attacked prompt, steered intervention)
loss_concept_avoiding = BPB(concept_avoiding | attacked prompt, steered intervention)
```

and reports the signed class margin `primary_margin_y`:

```text
positive mode:  primary_margin_y = BPB(avoid) - BPB(rich)
negative mode:  primary_margin_y = BPB(rich)  - BPB(avoid)
```

In both modes a **higher** `primary_margin_y` means **stronger** steering. The
sign flip makes negative-mode steering (suppressing the concept) read on the
same "higher is better" axis as positive-mode steering.

Both completions of a core prefix are **required**. A prefix that lacks a
`concept_avoiding` (or `concept_rich`) row in the requested mode is *incomplete*
and cannot form a class margin. Incomplete prefixes are always hard source
errors; clean historical sources explicitly before running BPB-Adv. This catches
source runs from the old shared global-negative pool, which could not supply a
per-prefix avoiding continuation.

### Adversarial objective: desired vs contrast

The worst-case adversary **degrades steering**, i.e. it **minimizes**
`primary_margin_y`. The token optimizer (Faster-GCG / GEPA) is written as a
minimization of the unweighted steering-margin search objective `g`:

```text
g      = loss_contrast - loss_desired
grad g = grad_contrast - grad_desired
```

where the **desired** class is the one the steering favors (positive →
`concept_rich`, negative → `concept_avoiding`) and **contrast** is the other.

`g = loss_contrast - loss_desired = primary_margin_y`, so the adversary minimizes
the margin directly. `g` is therefore exactly `primary_margin_y`, the same
quantity that is plotted and aggregated — the y-axis is always
`primary_margin_y`, and `g` never diverges from it.

The `k=0` no-adversary control is **scored once and reused across the attack
rows whose candidates share its scoring inputs**: distinct attack rows whose
candidates feed identical inputs into the dual-completion score (differing only
in fields excluded from the scoring identity — `{candidate_id, attack_row_id}`)
collapse to a single forward-pass evaluation per `(core prefix × steering spec)`
control rather than one per row (issue #511). The attack/candidate/score **rows
are unchanged**; only the redundant forward passes are removed, so all aggregates
and plots are byte-identical.

## Stages

```bash
uv run python -m scripts.eval.axbench_bpb_adv_eval \
  source_run_dir=out/eval/axbench_base/<run_tag> \
  output_root=out/eval/axbench_bpb_adv \
  feature_dir=out/data/features/... \
  feature_statistics_path=out/data/features/.../feature_statistics.safetensors \
  stage=validate_source
```

Valid stages are:

- `validate_source`
- `prepare_context_cache`
- `prepare_attack_rows`
- `build_candidates`
- `score`
- `score_general_val` (optional; gated on `general_val_bpb.enabled`)
- `aggregate`
- `plot`
- `all`

There is no `prepare_neutral_references` stage; the neutral apparatus was removed
when the metric became the per-prefix class margin.

`validate_source` additionally applies the both-classes-per-prefix requirement.
It logs the per-mode complete-prefix counts
(`n_positive_prefixes` / `n_negative_prefixes`) and records `n_dropped=0` in
`source_contract.json` for schema compatibility. Any incomplete prefix raises
before the contract is written. It also loads both AxBench seed files through
the strict schema-v2 loader, requires `seed_pool_manifest.json`, and hashes all
three into unclustered source-contract v6 or clustered v7. Production rejects
v4/v5 and older contracts; use a fresh run tag for new output.
An unversioned/torn AxBench seed pair fails before attack preparation.

`prepare_attack_rows` pairs each prefix's completions into a single core prefix
(`pair_samples_into_prefixes`) before emitting attack rows. A core prefix may now
own **multiple** `concept_rich` and `concept_avoiding` completions; the Cartesian
product of those completions is exploded into one attack row per `(rich_i,
avoid_j)` pair, optionally capped by `pair_subsample` (see
[Cartesian pairing](#cartesian-pairing-pair_subsample)). Every attack row still
carries exactly one rich and one avoid completion's tokens and digests, so each
pair gets a distinct `attack_row_id`. Any incomplete prefix raises before attack
rows are written.

The scored class-margin population is universal across method families.
`prepare_attack_rows` first applies the deterministic per-`(concept, mode)`
`ours_gcg.split_ratio` / `split_seed` optimize-vs-held-out split, then emits rows
only for the held-out prefixes. It also writes `evaluation_population.json`, a
method-independent list of the exact held-out `(core_prefix_id, rich digest,
avoid digest)` pairs. `build_candidates`, concept scoring, roundtrip scoring, and
general-val scoring enforce that artifact for every method, including
decoder-only baselines; stale full-prefix attack/candidate artifacts therefore
fail loudly rather than silently entering plots.

`prepare_context_cache` reuses AxBench-BPB's context-cache extraction layout so
encoder-decoder scoring receives full-context features. Context segments depend
only on the attacked **prompt** (encoder-decoder targets condition at
`<assistant_start>`, never on the completion), so the cache is computed once per
prompt and reused for both completions. Missing context rows are hard errors
during scoring.

`prepare_context_cache` participates on every torchrun rank, matching
AxBench-BPB's sharded context extraction. The remaining stages are rank-0-only
unless a later PR adds score/candidate partitioning.

The clustered cache's in-memory writer boundary is type-owned. The shared
cross-tokenizer alignment primitive accepts only complete lists of strict
non-negative character offsets and a strict non-negative decoded length;
`beartype` checks every list item deterministically before alignment. Raw and
selected cache records use refined domains for character offsets, token counts,
and positive/negative feature counts; normalized raw offsets and record counts
are stored under nominal phantom annotations. Direct metadata writers require
exact prefix-hash and SHA-256 shapes. Relationships that cannot be expressed by
one value's type remain explicit: range ordering, parallel lengths, prefix/hash
correspondence, DDP `rank < world_size`, tensor shape and dtype, and
aggregate-count consistency. File, safetensors, JSON, and mutable runtime-config
ingress still translates malformed external state into contextual
`AxBenchBpbAdvError` messages before it is trusted.

Attack preparation follows the same ownership rule. `BpbAttackSample`,
`BpbAttackPrefix`, and `BpbAttackWorkSpec` store nominal non-negative counts and
non-empty identities, accept only `SteeringMode` members, and validate complete
token-ID, nested completion-token, sample-ID, prefix, and work-spec collections
with deterministic `beartype` predicates. Empty prefix/work-spec inputs now fail
at the typed `prepare_bpb_attack_rows` call boundary. Evaluation-population pair
keys and universal-GCG concept-group keys likewise store nominal count/string
domains, so trigger lookup shares one exact typed seven-field identity instead
of coercing arbitrary values with `int()` or `str()`. Equal rich/avoid ID-to-
completion lengths, exact population group membership, constant editable width,
and canonical key strings remain explicit cross-field checks. Raw synth rows,
Arrow rows, and persisted evaluation-population JSON still translate malformed
external values into contextual `AxBenchBpbAdvError` messages.

When `feature_clusters` resolves, the dispatcher resolves the feature contract
once before stage execution, pins the encoder-decoder checkpoint step to that
contract, validates the clustered source before installing
`feature_contract.json`, then commits `source_contract.json` from that same
preflight validation before dispatching any stage. A later stage revalidation
must match the committed source contract or the run fails before stage output.
The dispatcher threads the feature contract through every stage. Each
independently invocable clustered stage revalidates the current source contract;
attack rows, candidates, universal triggers, and score tables are then checked
or atomically written with clustered identity metadata. Clustered attack-row
construction consumes the exact synth tables retained by that source
validation; it does not reopen the source paths after committing their
contract. Clustered context-cache construction likewise derives its complete
QC-eligible prefix population from those retained tables, broadcasts that
canonical population from rank 0, and never reopens the synth paths after
source-contract validation. The clustered
aggregation stage passes the exact score and general-validation table objects
that passed those identity checks into aggregation, rather than reopening their
paths after validation. Clustered candidate construction receives one paired
stage-source value containing both the exact validation snapshot and its
committed source identity. Its direction-artifact scale preflight and
precomputed-bag prefix setup consume the retained concepts and synth tables;
the unclustered path keeps its legacy path reads. The class-margin, roundtrip,
and general-validation score stages receive the same paired source value,
construct their concept index from its retained payload, and use its single
source identity for candidate validation and output binding. A clustered
candidate read is descriptor-bound and retains the exact Parquet payload, the
table parsed from that payload, and its SHA-256 in one factory-only snapshot.
Candidate construction receives the equivalent snapshot directly from the
immutable writer's exact rendered bytes. Score, roundtrip, general-validation,
universal-trigger, and aggregation identity consumers must use that snapshot;
they cannot substitute an arbitrary digest or reopen `candidates.parquet`
between table validation and downstream stamping. Replacing the candidate path
after snapshot creation therefore changes neither the consumed rows nor their
recorded upstream digest. Candidate-dependent identity operations reject calls
that do not carry this exact snapshot.
Source validation retains every permitted DiffMean/linear-probe baseline leaf
from the same descriptor-bound traversal that produces its directory digest,
for both clustered and unclustered sources. Metadata and tensors are validated
from those retained bytes, so replacing a pathname during validation cannot
mix one version's direction layer with another version's contract hash.
Clustered candidate, class-margin, roundtrip, general-validation, and
sample-generation intervention builders continue to receive that exact
validation owner. Unclustered runtime loaders perform a fresh descriptor-bound
read and repeat the complete identity/layer/tensor validation immediately at
hook construction; unclustered candidate reads also remain path-backed.
The clustered
`prepare_context_cache` path uses the selected-feature cache builder and runs
the tokenizer-checksum and BOS-length guards before cache writes. With no
resolved cluster contract, the dispatcher omits clustered kwargs, markers, and
parquet stamps.

### Rank-zero failure diagnostics

Source and direction preflight runs on rank 0 for clustered and unclustered
launches, before feature-marker creation, model construction, or stage output.
It broadcasts the existing concise failure summary before raising. This
catch-and-broadcast boundary is required so that peer ranks do not wait forever
at a collective when rank 0 fails.

Before broadcasting that summary, rank 0 logs the original exception with its
full traceback. The broadcast payload and the raised `AxBenchBpbAdvError`
remain the same as the existing contract; peers intentionally receive only
that summary. When a guard fails, inspect the rank-zero stderr/log for the
originating file and line rather than expecting the original traceback on
every rank.

### Clustered scoring: runnable, with one recorded coverage limit

The cluster-aware dispatcher, artifact pipeline, and scoring chain are end-to-end
runnable. Two temporary Track-F guards in the shared intervention builder previously
blocked 12 of the 14 clustered methods; both are gone (#1008, PR #1010):

- the eight precomputed fair/roofline Feature-DiffMean and Feature-GCG methods now
  consume their bags, validating each artifact's own
  `feature_contract_fingerprint` on read;
- the four positive/negative decoder-only DiffMean and linear-probe methods consume
  exact source-certified payloads, including coordinate and scoring-layer checks.

Both guards cited prerequisites that had already landed, so they were premise drift
rather than deliberate unsupported-method boundaries. Worth recording because it was
not obvious: `validate_clustered_method_allowlist` certified all 14 methods against
`CLUSTERED_AXBENCH_BPB_ADV_METHODS`, and the shipped `axbench_bpb_adv_clustered.yaml`
listed all 14, so the *config* layer certified a method set the *runtime* refused —
the shipped clustered config died on its first diffmean concept.

#### Bag reuse on resume, and why byte-comparison was dropped

Clustered precomputed bags are immutable within a `run_tag`. A resumed run does **not**
re-derive them and byte-compare: instead it validates the committed bag's *identity*
and reuses it, skipping the derive (#1050).

The reason is that the feature-GCG derive is not bitwise reproducible. Its per-step
gradient is a CUDA backward pass whose atomics do not fix an accumulation order, and
`rank_candidates` amplifies a sub-ULP difference — a reordered top-k pool makes
temperature sampling draw a different candidate set even from an identical RNG stream.
Measured on the real l22 clustered checkpoint with byte-identical inputs, two
consecutive derives of the same bag differed in 2 of 24 feature ids and moved
`optimize_g_steer` by ~20%. Forcing `torch.use_deterministic_algorithms(True)` would
not fix the contract either: resuming on a different GPU, driver, or torch version
would still diverge.

Bag metadata is therefore partitioned three ways:

| category | keys | on resume |
|---|---|---|
| identity | `concept_id`, `mode`, `mode_tag`, `prefix_id` (roofline), `reg_lambda`, the `n_*` input counts, `k_pos`/`k_neg`, buckets, `master_seed`, `split_ratio`, `split_seed`, `max_steps`, `minibatch_size`, `extraction_batch_size`, `feature_contract_fingerprint`, `vocab_fingerprint`, `bag_derive_schema_version` | compared exactly; any difference **raises**, naming every differing key |
| self-consistent | `n_features` | checked against the artifact's own tensors |
| diagnostic | `warm_start_g_steer`, `optimize_g_steer`, `held_out_g_steer`, `in_sample_g_steer`, `warm_penalized_lm_objective`, `optimized_penalized_lm_objective` | ignored |

The partition is fail-closed: a new metadata key counts as identity by default, so a
key that ought to gate reuse cannot be silently ignored by omission. Note that
`split_ratio` and `extraction_threshold` are also stamped as `repr(float(...))` but are
config-derived and stay identity — the discriminator is *measured vs config*, not
"is it a float repr".

Two limits are worth knowing. Identity pins input *counts*, not input *content*; that
is safe only because every clustered stage revalidates the current source contract
against the committed one before a derive is reached. And a bag written before
`bag_derive_schema_version` existed validates as version `1`, so it stays reusable but
a change in derive semantics cannot be detected for it.

Reuse decisions are logged at DEBUG.

### Certified clustered preflight

Clustered contract resolution is producer-certified. Each launch still hashes
the complete compressed selection inventory, but a signature-valid
`.packed-row-fingerprint-cache.json` lets it skip packed-body hashing and
packed/selection tensor decompression. Thus a warm launch still has selection
I/O even when packed-body reads are zero. Missing or structurally valid stale
caches are rebuilt and atomically published after validation; corrupt,
truncated, unreadable, wrong-schema, or internally inconsistent caches fail
before packed-body I/O.

Set `force_cluster_shard_revalidation=true` to run the complete canonical body
validators and packed-body hash from one exact compressed read per packed
shard, then repair the cache after success. The flag is rejected without
`feature_clusters`; its false default is omitted from legacy run-tag/config
serialization and true is recorded. Under DDP, rank 0 is the only cache writer
and the only forced full packed-body reader. Peers independently verify
manifests, selection bytes, signatures, cache, and final fingerprint before any
rank returns. The BPB-Adv sample generator uses this same preflight and accepts
the same flag.

## Configuration Shape

`adversarial_methods` is a list of tagged method configs. A method is enabled by
including its config object; omit it to disable it.

```yaml
adversarial_methods:
  - method: faster_gcg
    optimized_token_counts: [0, 1, 2, 4, 8, 16, 32]  # the universal-trigger lengths k (0 = control)
    max_steps: 256                # per-concept-group optimization budget
    universal_minibatch_size: 16  # rich x avoid pairs dealt per GCG step
```

GEPA and additive direction use the same presence-based convention:

```yaml
adversarial_methods:
  - method: gepa
    metric_call_budgets: [8, 16, 32, 64]
    k_editable_tokens: 16
    reflection_model: ...
  - method: additive_direction
    families: [diffmean, linear_probe, lora, ours]   # which baselines to attack
    counter_direction_family: diffmean               # the specified counter vector
    magnitudes: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
```

`families` selects which baseline families additive attacks (`diffmean`,
`linear_probe`, `lora`, `ours`); `counter_direction_family` (`diffmean` /
`linear_probe`) selects the concept counter-direction pushed on top of each. The
counter-direction is **specified globally** and applied to every attacked baseline
— so with the default `diffmean` counter, a `linear_probe` baseline is now
counter-steered along the concept's **diffmean** direction, not its own probe
direction (a deliberate change from the legacy per-baseline counter). For an
encdec baseline the counter uses that target's own direction
(`baselines/<family>/encdec/...`, the encdec residual basis).

`base_methods: null` means all supported AxBench-Base steering rows for an
unclustered run. A clustered run requires an explicit non-empty subset of the
closed fourteen-method allowlist: the six positive/negative decoder-only
DiffMean, linear-probe, and control methods; positive/negative fair and roofline
Feature-DiffMean; and positive/negative fair and roofline Feature-GCG. Unknown
names, duplicates, and target-model mismatches fail rather than being silently
dropped.
`representative_target_methods` is a separate narrow escape hatch for fair
Feature-DiffMean selector targets: those methods are scored only at each
adversary's explicit no-adversary control row and are not optimized across the
prompt-token attack grid. Use it when a run needs the fair Feature-DiffMean
target for representative-coefficient selection without treating
Feature-DiffMean as a full scored method family. If an adversary grid omits an
explicit no-op control, BPB-Adv raises instead of inventing one.
The research eval default narrows to decoder-only DiffMean/linear-probe/control,
the scored encoder-decoder feature-DiffMean conditioning-bag baselines, and the
fair feature-space GCG baselines (`ours_gcg_encdec` /
`neg_ours_gcg_encdec`). It deliberately omits the replace-all encoder-decoder
controls plus the roofline feature-space GCG and Feature-DiffMean oracles unless
they are explicitly re-added.

Clustering additionally requires top-level `feature_mask` and
`feature_clusters` together, each as a path-backed mapping. Main declares both
keys as `null`, so override the complete mappings; it does not ship the frozen
campaign's `axbench_bpb_adv_clustered` Hydra preset. The default ten-method list
is a legal clustered subset, but it is still subject to the scoring blockers
above.

Feature-DiffMean has the same fair/roofline split as feature-space GCG. The fair
methods (`feature_diffmean_encdec` / `neg_feature_diffmean_encdec`) build one
conditioning bag per concept/mode from the training split and score held-out
prefixes. The roofline methods (`feature_diffmean_roofline_encdec` /
`neg_feature_diffmean_roofline_encdec`) build a separate conditioning bag from
each held-out prefix pair and score that same held-out prefix pair, giving an
in-sample upper-bound estimate for optimal feature conditioning signals while
keeping the scored prefix population identical to the fair methods.

With an active cluster feature contract, both Feature-DiffMean derives accept
that contract and the shared `ClusteredWarmStartCache` built before the target
model becomes the sole large accelerator owner. The cache contains exactly the
same fair `(concept, mode)` and roofline `(concept, mode, core_prefix_id)` bags
that an inline clustered derive would compute; cache/contract fingerprint
mismatches and missing per-prefix roofline keys raise. Clustered bag and
scored-prefix artifacts both carry the feature-contract fingerprint, while the
unclustered call path omits the new kwargs and stamps so its artifact content
remains unchanged.

Pass-through runtime fields mirror AxBench-BPB:

- `steering_metric` (`roundtrip` / `bpb_class_margin`, default `roundtrip`):
  selects the score/aggregate/plot y-axis. Candidate construction and
  Faster-GCG/GEPA objectives remain BPB-based in both modes.
- `roundtrip_generation`: generation settings for roundtrip scoring
  (`samples_per_candidate`, `temperature`, `top_k_sampling`, `master_seed`).
  The token budget is computed before sampling and is not configurable. The
  mode-appropriate continuation (`concept_rich` for positive rows,
  `concept_avoiding` for negative rows) is decoded in the context of the source
  prompt tokens, not in isolation. That contextual reference text is segmented
  in extraction-token space using `context_cache.segment_size`; generation then
  runs for the number of target-model tokens needed to cover the reference text
  through the end of the first segment touched by the continuation. This
  conversion goes through character offsets, so the extraction-token count and
  `max_new_tokens` need not be equal. The score artifact records the final
  target-token budget as
  `reference_segment_token_count`.
- `roundtrip_steering`: roundtrip probe settings specific to AxBench-BPB-Adv.
  defaults `enabled=true`; when `feature_statistics_path` is null, it falls
  back to the top-level `feature_statistics_path`. It always probes each
  candidate's `Concept.feature_index`. If either the steered or
  control completion is empty, the current scorer skips that sample and records
  `axbench_bpb_adv.score.roundtrip_empty_steered`,
  `axbench_bpb_adv.score.roundtrip_empty_control`, and
  `axbench_bpb_adv.score.roundtrip_empty_skipped` counters; it does not
  synthesize a zero-reactivation score. A future contract change to raise on
  empty completions is tracked separately from the roundtrip bottleneck
  investigation. The probe `model_name` must match
  `neuronpedia_model_id`, numeric `probe.l0` is
  rejected because source concept metadata does not encode GemmaScope v1 L0
  variants, and the feature-statistics
  layer/width metadata must match the probe before percentile scoring starts.
- `extraction_max_tokens` / `extraction_threshold`: SAE feature extraction
  knobs used when converting text into sparse SAE feature rows for context
  segments. `extraction_max_tokens` caps the Gemma-side
  extraction context; it is not the target LM scoring length.
  `extraction_threshold` is the SAE activation cutoff applied before sparse
  feature rows are bucketed/stored.
- `decoder_only_target`, `encdec_target`, `lora_adapter`: target bundle and LoRA
  configuration blocks consumed by AxBench-BPB runtime loading and intervention
  construction.
- `filter_unaccepted_samples`: whether synth rows with failed QC are removed
  before attack-row construction.
- `pair_subsample` (int, default `1`, `0` = no cap): per-prefix Cartesian
  rich×avoid pair cap — see [Cartesian pairing](#cartesian-pairing-pair_subsample).
- `scatter_baseline_x` (`baseline_loss` / `min` / `general_val_bpb`, default
  `general_val_bpb`): the x-axis of the steerability-vs-baseline scatter
  families (and the per-coefficient scatters) — see
  [Steerability-vs-baseline scatter](#steerability-vs-baseline-scatter). Needs
  `general_val_bpb.enabled=true` (also the default); switching this to
  `baseline_loss` or `min` does not require disabling `general_val_bpb`, it
  just leaves that scoring pass's output unused for this plot.
- `plot_steering_strength_families` (bool, default `false`): during `plot`,
  additionally writes per-method base-strength families under
  `steering_strength_plots/`; see
  [Steering-strength family plots](#steering-strength-family-plots).

Residual-direction calibration is direction-owned and has no configuration
knob. DiffMean/LinearProbe scoring, roundtrip, general-val, Faster-GCG
objectives, and sample generation read `max_activation` only from the exact
direction artifact they apply. For its unit direction `d`, Stage 3 computed
that value as the maximum signed `h_token · d` over continuation tokens in
both rich and avoiding fit rows at that target model's hook layer, in float32
on CPU without ReLU or absolute value. Missing, invalid, negative,
zero-inconsistent, or wrongly bound metadata raises before expensive work;
there is no feature-statistics or constant fallback. Additive counter-steering
uses the counter artifact's own independently validated value.

## Incomplete-prefix Handling

A core prefix needs **both** class sides (at least one `concept_rich` and one
`concept_avoiding` completion) to form a class margin. A prefix that is missing
an entire side is *incomplete*. Incomplete prefixes are always hard failures in
the BPB-Adv runtime, naming the offending `(concept_id, prefix_text)` groups.
There is no config knob that drops or skips them. Clean older source runs
explicitly before scoring them.

### Cleaning a source for fail-loud validation

If an older source run has incomplete groups, build a filter-derived clean sibling
first:

```bash
uv run python -m scripts.eval.clean_axbench_source \
  --source-run-dir /path/to/axbench_base/source_run
```

The default output is `/path/to/axbench_base/source_run_paired`. The cleaner
rewrites only `synth_training_positive.parquet` and
`synth_training_negative.parquet`; every other source artifact is hardlinked
when possible and copied otherwise. In particular, the schema-v2 train/eval
seed files and `seed_pool_manifest.json` are copied byte-for-byte and then
revalidated; the cleaner never rewrites seed metadata or infers BOS ownership.
Rows are dropped using the same predicates
as validation and attack-row preparation: `qc_accepted=false` rows when
`filter_unaccepted_samples=true`, rows whose `concept_id` is absent from
`concepts.json`, and incomplete `(concept_id, prefix_text)` groups. The cleaner
is the only place incomplete groups are dropped, and it records those drops in
`cleaning_manifest.json`; normal eval validation then records `n_dropped=0`.

Each cleaned source includes `cleaning_manifest.json` with the source/output
paths, filter settings, per-split row/drop counts, sorted dropped groups, and
the self-verification contract hash. The manifest is not part of
`source_contract.json` hashing; the cleaned source has a new contract hash
because the synth parquets and `source_run_dir` changed.

## Cartesian pairing (`pair_subsample`)

A core prefix may own more than one `concept_rich` and/or `concept_avoiding`
completion (the same `(concept_id, mode, prefix_text)` can recur across synth
rows). All rich completions are paired against all avoid completions: the
`N × M` Cartesian product is exploded into one attack row per `(rich_i, avoid_j)`
pair. Each row still owns exactly one rich and one avoid completion, so the
downstream dual-completion scoring and aggregation are unchanged; distinct pairs
differ in their output digests and therefore get distinct `attack_row_id`s (no
collision).

`pair_subsample` bounds the per-prefix explosion:

- `pair_subsample = 1` (default): keep a single pair per prefix.
- `pair_subsample = 0`: keep every pair (no cap).
- `pair_subsample = k > 0`: when `N × M > k`, keep a deterministic, reproducible
  size-`k` subset. The subsample seed is derived **only** from `core_prefix_id`,
  so distinct prefixes draw independent subsets and re-runs are byte-identical.
  When `N × M <= k`, every pair is kept.

**Aggregation identity.** Aggregation averages each side's BPB across a prefix's
pairs and re-derives `primary_margin_y` from those means via the centralized
sign-flip (`_signed_margin`). It never averages per-pair margins directly. This
mean-of-completions-then-margin order is exactly preserved by the Cartesian
explosion because every row carries both sides; capping pairs changes which
completions contribute but not the identity.

## Adversarial Strength

For the default `steering_metric=roundtrip`, the primary plot family is a
connected scatter:

- `x = adversarial_strength`
- `y = primary_roundtrip_y`
- points are connected in increasing `adversarial_strength`
- the no-adversary row's `primary_roundtrip_y` is drawn as a dashed reference
  line in the primary plot

The primary `primary_roundtrip_y_vs_strength.png` line collapses each steering
family to its **representative coefficient** (see
[Representative coefficient](#representative-coefficient) below) so the plot
shows one line per method family instead of one per coefficient, while keeping
that coefficient's full adversarial-strength sweep. The per-metric diagnostic
plots (`mean_roundtrip_percentile`, `mean_roundtrip_post_act`,
`mean_roundtrip_pre_act`, `control_primary_roundtrip_y`) still draw the complete
coefficient grid.

For `steering_metric=bpb_class_margin`, the primary plot family uses the legacy
BPB class-margin y-axis:

- `x = adversarial_strength`
- `y = primary_margin_y`
- points are connected in increasing `adversarial_strength`
- the `k=0` control's `baseline_primary_margin` is drawn as a dashed reference
  line

The primary `primary_margin_vs_strength.png` line collapses each steering family
to its **representative coefficient** (see
[Representative coefficient](#representative-coefficient) below) so the plot shows
one line per method family instead of one per coefficient, while keeping that
coefficient's full adversarial-strength sweep. The per-metric diagnostic plots
(`mean_loss_concept_rich`, `mean_loss_concept_avoiding`, `baseline_primary_margin`)
still draw the complete coefficient grid.

Plots are split into separate files by steering `mode` (`positive` / `negative`)
to keep token-optimizer settings from crowding one image. Within each plot,
panels are grouped by adversarial method and strength
unit, so decoder-only and encoder-decoder rows share a panel when the
adversarial steering type matches (for example, Faster-GCG optimized-token
attacks). Color hue encodes target/steering family; shade encodes adversarial
strength, with the no-adversary control or lowest-strength point drawn as the
lightest shade and higher strengths drawn progressively darker; line style
encodes context mode (`fixed_context` solid; `not_applicable` renders SOLID and
is omitted from the context legend).

The y-axis label states the explicit definition: positive mode plots
`avoid - rich` BPB, negative mode plots `rich - avoid` BPB, and higher means
stronger steering in both modes. There is no `baseline_x_mode`; the x-axis is
always `adversarial_strength`.

### Steering-strength family plots

`plot_steering_strength_families: true` adds a separate plot family whose x-axis
is the **base steering coefficient**, not adversarial pressure. It filters macro
aggregates to no-adversary rows (`adversarial_strength == 0.0`, with
`is_control=true`), groups by `(mode, method_name, target_model, coefficient)`,
and averages duplicate rows from concatenated runs. It writes:

```text
steering_strength_plots/plot_index.csv
steering_strength_plots/plot_series_legend.csv
steering_strength_plots/<mode>/steering_strength_families.png
```

Each PNG is a **single steerability-frontier panel** (PR #627): x = `general_val_bpb`,
y = the active steering metric (`primary_roundtrip_y` for roundtrip runs,
`primary_margin_y` for class-margin runs — labeled "BPB class margin (higher =
stronger)" on the axis). Points within a family are connected in increasing
base-coefficient order to trace a frontier; a family reduced to a single
coefficient (for example a single-config control) renders as one unconnected
marker instead of a line.

The frontier never silently degrades — three situations raise instead:

- **Missing capability x.** Every family drawn on the frontier must have a
  `general_val_bpb` x-coordinate. A family missing one — for example a roofline
  method in a run whose `score_general_val` stage did not score it — raises,
  naming the offenders, rather than being dropped from the capability axis
  (which would let a method vanish from a comparison while others are shown).
  Roofline general-val is implemented (per-document, see
  [General-validation BPB](#general-validation-bpb-general_val_bpb)), so a null
  roofline value means the stage was not run for that method, not that roofline
  is unsupported.
- **Partial-null groups.** If concatenating multiple runs leaves one
  `(mode, method_name, target_model, coefficient)` group with a mix of null and
  non-null `general_val_bpb` rows, plotting raises instead of averaging over
  whichever rows happen to have data — a gv-enabled run cannot be silently
  blended with a gv-less run of the same configuration.
- **Mixed corpus signatures.** Every plotted row (after any `plot_methods`
  whitelist) must carry the same `general_val_corpus` (see
  [General-validation BPB](#general-validation-bpb-general_val_bpb)); mixing
  corpora raises naming the offending corpus/method pairs rather than drawing a
  frontier that compares `general_val_bpb` scored on two different validation
  sets.

For clean production sweeps, use:

```bash
uv run python -m scripts.eval.axbench_bpb_adv_eval \
  --config-name axbench_bpb_adv_strength_sweep \
  source_run_dir=... feature_dir=... feature_statistics_path=...
```

That preset uses `additive_direction` with `magnitudes: [0.0]` and enables
`plot_steering_strength_families`.
Existing finished runs can be replotted without re-scoring:

```bash
uv run python -m scripts.eval.plot_steering_strength_families \
  --output-root out/eval/axbench_bpb_adv \
  --run-tag <run_tag> \
  --metric roundtrip
```

Repeat `--run-tag` to combine several finished runs into one frontier; combining
requires `--plots-dir` (there is no per-run default once more than one run is
involved). Every replotted run — a single run or a combined set — must carry its
validated `source_contract.json` (a missing or malformed one raises, naming the
run): `validate_source` writes it for every real run, so its absence means an
incomplete artifact. When combining, the script additionally requires an equal
`contract_hash` across all runs, raising and naming each run's tag + hash on a
mismatch, so a frontier can never splice methods from divergent source runs.
Combining writes the concatenated `combined_macro.parquet` (only when more than
one `--run-tag` is given) alongside the figures, and always writes a
`provenance.yaml` recording the input runs (macro path, `contract_hash`, row
count), the metric, any `--plot-method` whitelist, the drawn figure's
`general_val_corpus` (`null` if every family lacked one), and the list of files
written.

Method-specific strength meanings:

- Faster-GCG: the length `k` of the shared universal per-concept trigger — the
  number of final non-BOS prompt tokens it overwrites. This is not the optimizer
  step count. One trigger is optimized per concept (see
  [The universal per-concept trigger](#the-universal-per-concept-trigger)) and
  substituted into every prefix's last-k span, so larger `k` is a longer (more
  expressive) adversarial trigger. `k=0` is an exact no-adversary control and
  supplies `baseline_primary_margin`.
- GEPA: metric-call budget. GEPA is omitted from the default method list because
  it needs a reflection model configuration.
- Additive direction: counter-direction magnitude. The additive adversary pushes
  a **specified concept counter-direction** (the concept's `diffmean` or
  `linear_probe` direction, set by `additive.counter_direction_family`, default
  `diffmean`) as a residual hook **on top of the FULL baseline steering**,
  opposing the concept (coefficient `-magnitude` in positive mode, `+magnitude`
  in negative mode). `magnitude = 0` is the pure-baseline control. This composes
  with **any** baseline — `diffmean`, `linear_probe`, `lora`, and the
  encoder-decoder `ours` methods — chosen by `additive.families` (which baseline
  families to attack). The counter magnitude is **scaled by the counter
  direction artifact's own `max_activation`** (#1228), not the base
  artifact's scale. For unit `d`, that field is Stage 3's maximum signed
  continuation-token `h_token · d` over rich and avoiding fit rows at the
  target hook layer (float32 CPU; no ReLU/absolute value). The loader raises on
  absent, invalid, negative, zero-inconsistent, or wrongly bound metadata;
  there is no configuration knob or fallback. `magnitude = 0` returns the
  unchanged pure-baseline object. When a DiffMean baseline is countered along
  the same DiffMean artifact, the hooks superpose to
  `(|base| − magnitude) × max_activation` (up to float non-associativity). The
  counter add-on is an orthogonal field on `Intervention` installed by `apply_intervention` via
  `register_residual_hook`, which works for both decoder-only and encoder-decoder
  decoders. additive is scoring-only (no gradient); faster-GCG remains the
  prompt-token attack.

## Representative coefficient

A steering method is usually swept over a coefficient grid. To keep the primary
line plot readable, each steering family selects a single **representative
coefficient** per
`(mode, method_name, target_model, adversarial_method, context_signal_mode)`.

The selector is not configurable. It always uses the scored fair
encoder-decoder Feature-DiffMean conditioning-bag baseline as the target:

- positive mode target: `feature_diffmean_encdec`
- negative mode target: `neg_feature_diffmean_encdec`
- target model: `encdec`
- same `mode`, `adversarial_method`, and `context_signal_mode`
- no-adversary control rows only (`is_control=true`)
- metric: `primary_roundtrip_y` for roundtrip, `baseline_primary_margin` for
  BPB class-margin

For every other family, the selector computes that family's no-adversary control
score at each coefficient and picks the coefficient whose score is closest to
the feature-DiffMean target. Exact distance ties choose the higher steering
score; remaining ties choose the smaller `abs(coeff)`, then the lexicographically
smaller coefficient string. Missing feature-DiffMean target rows or null target
scores are hard errors, including when the target rows are hidden by
`plot_methods`; the selector reads the full macro table as its target source.
For narrow probes, list the fair Feature-DiffMean target in
`representative_target_methods` instead of `base_methods` so it supplies only
the selector target control row.

The evaluator rejects an impossible fresh `stage=all` configuration before
model initialization and scoring: every selected steering mode must include
its exact fair target method, either in `base_methods` or
`representative_target_methods`, with `encdec` in `target_models`. This is a
configuration preflight only; decoder-only scoring and aggregation remain valid.
`plot_methods` cannot waive the requirement because the full macro table is the
selector's target source. A direct `stage=plot` invocation is artifact-aware
instead: it reads the existing macro first, accepts a replay configuration that
names only decoder-only targets when the artifact contains the fair rows, and
fails before clearing old plots when those rows are absent or null.

The chosen coefficient keeps its **entire** adversarial-strength sweep — only the
coefficient axis is collapsed, never the strength axis. Families whose candidate
control score is entirely `None` still select `None` and are passed through
wholesale rather than silently dropped; this does not relax the feature-DiffMean
target requirement when the new selector is active.

### Plot-readability reduction is NOT a GCG cost lever

`_select_representative_coefficient` and `_filter_rows_to_representative_coefficient`
are a **plot-readability reduction only**: they run *after* scoring and collapse the
plot to one line per family (the representative coefficient), discarding the other
coefficients' rows from the figure. They do **not** reduce the number of
GCG-optimizable attack rows — every coefficient in the grid has already been
generated, optimized, and scored by the time this reduction runs. The plot keeps
one coefficient; the optimizer paid for all of them.

The real GCG cost lever is orthogonal: run a **reduced coefficient grid at
attack-row-generation time**, so fewer coefficients are ever optimized. Distinguish
the two:

- **plot one coefficient** (implemented here): a post-scoring figure reduction;
  zero effect on optimizer cost.
- **optimize one coefficient** (a grid choice, made upstream at attack-row
  generation): the actual `−10×` lever — fewer coefficients enter the GCG loop.

For a Faster-GCG spot-check you want the latter (a single-coefficient grid at
generation time); the representative-coefficient selection then merely makes the
already-reduced result a clean single line.

## Steerability-vs-baseline scatter

Alongside the strength line plots, each `mode` subset emits a
`steerability_vs_baseline.png` scatter that places each steering family on a
steerability-vs-difficulty plane. Each family is reduced to its **representative
coefficient** (the best-at-rest coefficient, see
[Representative coefficient](#representative-coefficient)) and drawn as a
**counter-steering strength sweep** — one point per adversarial strength of that
coefficient, connected light → dark in strength order. Because the swept axis is
the counter-steering strength (not the base coefficient), a single-base-config
method (LoRA, a no-steering control, encoder-decoder `ours`) still gets a full
multi-point curve.

- **y** = `primary_margin_y` (steerability), recomputed from the row's two side
  losses via the same `_signed_margin` sign-flip used everywhere else, so higher
  is better in both modes.
- **x** = the baseline BPB, selected by `scatter_baseline_x`:
  - `general_val_bpb` (default): the held-out generic-validation BPB under this
    row's intervention (a concept-agnostic "how broken is the model" axis).
  - `baseline_loss`: the model's NATURAL (unsteered) completion BPB, which is
    **mode-dependent** — positive mode steers toward the concept so the
    natural completion is concept-avoiding (`mean_loss_concept_avoiding`);
    negative mode steers away so it is concept-rich (`mean_loss_concept_rich`).
  - `min`: `min(mean_loss_concept_rich, mean_loss_concept_avoiding)`.
- **color** encodes the model/steering family; **brightness** encodes the
  counter-steering strength rank (light → dark); the **marker** encodes the mode.
- each family's points are connected with a line in strength order, so the sweep
  reads as a counter-steering trajectory. An unreduced multi-coefficient family
  (no usable control margin) is kept as SEPARATE per-coefficient sweeps rather
  than one line zigzagging across coefficients.

Points whose coefficient is `None`, or whose baseline x is `None` (for `min`,
either side being null makes the minimum undefined), are skipped. If no point
survives, the scatter PNG is not written for that subset. Every emitted scatter
PNG is registered in `plot_index.csv` and the series legend, so the recursive
plot counter includes it alongside the line plots.

For `steering_metric=roundtrip`, the analogous scatter is
`roundtrip_steerability_vs_baseline.png`: x is the BPB baseline selected by
`scatter_baseline_x` (`general_val_bpb` by default; see above), and y is
`primary_roundtrip_y`.

### Per-coefficient steerability scatters

The default scatter collapses each family to its representative coefficient. To
inspect a FIXED magnitude instead, each `mode` subset also
emits one scatter per distinct `coefficient_str` under a `by_coefficient/`
subdirectory:

```text
plots/<mode>/by_coefficient/coeff_<slug>/steerability_vs_baseline.png
```

The coefficient slugs follow `_coefficient_slug`'s digit rules (`None` → `none`,
`0.2` → `0p2`, `6.0` → `6`, `-1.5` → `neg_1p5`) and are listed `none` first, then
ascending magnitude. Each scatter fixes its coefficient across all families
(keeping that coefficient's full counter-steering strength sweep) instead of
choosing the best-at-rest coefficient, so non-representative magnitudes are still
visible. Set `scatter_baseline_x: general_val_bpb` to put the concept-agnostic
validation loss on the x-axis. The per-coefficient scatters are self-contained:
their own `by_coefficient/plot_index.csv` lists every emitted magnitude (so the
top-level `plot_index.csv` stays focused on the default plots). The
per-coefficient PNGs are still counted by the recursive plot counter.

## Per-datapoint sampled completions

A finished run's metric artifacts are aggregate-first: class-margin runs write
`aggregates/`, while default roundtrip runs write `roundtrip_aggregates/`. To
inspect *what the steered model actually writes*, a separate script samples
completions for each selected steering family, at the SAME representative
coefficient the plots select, under the EXACT steered (+ optional additive
counter) intervention the eval scored. Legacy/unclustered runs remain
decoder-only. A clustered run may additionally replay `encdec` families from
the exact scored context cache and persisted fair/roofline conditioning bags.
The generator chooses `aggregates/` or `roundtrip_aggregates/` from
`steering_metric`, not directory-existence order:
class-margin sample YAML emits `primary_margin_y`, and roundtrip sample YAML
emits `primary_roundtrip_y`.

Run it against a finished run (point `source_run_dir` / `output_root` / `run_tag`
at the run the eval produced):

```bash
uv run python -m scripts.eval.axbench_bpb_adv_generate_samples \
  source_run_dir=/path/to/source_run \
  output_root=out/eval/axbench_bpb_adv \
  run_tag=<the eval run_tag> \
  steering_metric=<same as the eval: roundtrip or bpb_class_margin> \
  feature_dir=<same as the eval> \
  feature_statistics_path=<same as the eval>
```

Keep the target blocks, `diffmean_and_probe_layer`, and
`counter_direction_family` identical to the eval so generation reuses the SAME
model + intervention it scored. Keep `steering_metric` identical too, especially
when reusing one run directory across metric switches; a mismatch fails loudly
instead of falling back to stale aggregates. Sample generation does not run the
roundtrip scorer, so do not mirror `roundtrip_steering.enabled` here; legacy
class-margin samples only need `steering_metric=bpb_class_margin`.

For a clustered finished run, also pass the exact path-backed
`feature_mask`/`feature_clusters`, feature/SAE/statistics identity, checkpoint
selection overrides, and requested `target_models`. Startup requires the
existing `feature_contract.json`, source contract, and candidates; it
re-resolves the contract, rejects source/target drift, validates score-table
identity, and rebuilds unstamped aggregate tables from those stamped score
inputs before selecting coefficients. The clustered replay keeps the paired
source validation, exact candidate snapshot, validated score/general-validation
tables, and newly derived aggregate tables in memory. It checks the aggregate
function's returned macro table for exact equality with an independent
in-memory derivation and passes those same aggregate objects to every requested
target; it does not reopen candidate, aggregate, or source-concept paths between
validation and generation. Direction-based interventions likewise consume the
source validation's retained baseline bytes. Clustered YAML is create-once,
carries the feature/source/candidate identity, and is stamped from those
retained owners. `encdec` is accepted only in this validated clustered mode; an
ordinary unclustered `encdec` request remains skipped/rejected.

Use `force_cluster_shard_revalidation=true` only when an explicit full proof or
packed-cache repair is required.

For each selected family at its representative coefficient, the generator
takes the first `n_concepts` concepts (ascending `concept_id`) and their first
`n_prefixes_per_concept` core-prefixes, and at every counter-steering
`strength_index` samples `n_completions` completions from the attacked prompt of
one **representative candidate**. The per-completion seed is `base_seed + i`,
reused ACROSS strengths, so a completion's drift is attributable to the swept axis
(the additive counter intervention for `additive_direction`; the attacked prompt
for `faster_gcg` / `gepa`), not sampling noise. Families with no usable control
margin (no single representative coefficient) are logged and skipped.
For clustered encoder-decoder replay, the generator loads the exact scored
prefix context and validates representative-only target/context bags immediately
before encoder packing. This support still depends on an already-finished
clustered scored run; the current scoring blockers above prevent producing the
advertised full-method run on main.

### Choosing the representative candidate

A datapoint `(concept, prefix, strength)` can carry **more than one** candidate:
one per `(rich_i, avoid_j)` completion pair (see
[Cartesian pairing](#cartesian-pairing-pair_subsample)). The representative is the
candidate with the lowest `objective_value` (`g` is minimized, so lower = more
degraded steering — see
[Adversarial objective](#adversarial-objective-desired-vs-contrast)), breaking
exact ties on `candidate_id`. When no objective is recorded — every additive /
control / GEPA candidate, and the `k=0` faster_gcg control — the selection falls
back to `min(candidate_id)`, which for `additive_direction` reproduces the
pre-feature behaviour exactly (the additive YAML is byte-identical).

Under the universal trigger, a concept's prefixes share **one** optimized trigger,
so a datapoint's prompt-token candidates carry an **identical** attacked prompt /
`replacement_text` and the **same** concept-level shared-trigger `objective_value`
(the group-aggregated mean-over-pairs `g`, not a per-prefix strongest single
attack). The min-`objective_value` selection therefore reduces to the
`candidate_id` tie-break and the shown prompt is well-defined regardless of which
candidate wins. The datapoint's active steering metric (`primary_margin_y` for
class-margin runs, `primary_roundtrip_y` for roundtrip runs) is still the mean
over **all** its candidates.

> GEPA records no `objective_value` today, so a GEPA family's representative is the
> arbitrary-but-deterministic `min(candidate_id)`; surfacing a GEPA search objective
> is a follow-up.

### Attacked-prompt fields (prompt-token attacks only)

For `faster_gcg` / `gepa` families — where the prompt is the adversarial variable —
each datapoint additionally records what the completions were generated from:

- `attacked_prompt_text`: the full GCG/GEPA-attacked prompt (the chosen
  candidate's stored decode). At the `k=0` control this equals the source prefix.
- `replacement_text`: just the optimized editable-span tokens (the adversarial
  edit); `""` at `k=0` (empty span).
- `objective_value`: the representative candidate's search objective `g` (the
  achieved adversarial strength); `null` at the `k=0` control and for GEPA.
- `optimization`: the representative candidate's typed `candidate_metadata_json`
  serialized to a mapping. Faster-GCG emits
  `{candidate: faster_gcg_final, max_steps, accepted_steps, visited_sequences}`;
  these fields are **concept-group-level** under the universal attack (the trigger
  is optimized once per concept and shared across all its prefixes), so every
  candidate in a group reports the same values. The `k=0` control emits
  `{candidate: no_adversary_control}` and GEPA emits `{candidate: gepa_final}`.
  Missing, malformed, or unknown metadata is treated as a corrupt candidate
  artifact and raises before sample YAML is written.

`additive_direction` leaves the prompt unchanged, so it **omits** these four
fields and its datapoints stay byte-identical to the pre-feature output. Each
`mode` keeps its own YAML subdir for prompt-token attacks, so a faster_gcg run
writes one YAML per (mode, method) family.

### `generation:` config block

```yaml
generation:
  n_concepts: 5              # first-N concepts (ascending concept_id)
  n_prefixes_per_concept: 1  # first-N core-prefixes per concept
  n_completions: 5           # completions sampled per datapoint
  temperature: 0.7
  top_k: 50
  max_new_tokens: 128
  base_seed: 0               # seeds base_seed + i, reused across strengths
  methods: null              # optional method_name whitelist; null = all families
```

`counter_direction_family` is a top-level field of the generate-samples config
(default `diffmean`); set it to match the eval's
`adversarial_methods[additive_direction].counter_direction_family`.
`scatter_baseline_x` is echoed verbatim into each YAML so the qualitative sample
records the same baseline-x convention as the plots.

### Per-family YAML schema

One YAML per family is written under
`samples/<mode>/<method>__<adv>.yaml`:

```yaml
family: <mode>|<method_name>|<target_model>|<adversarial_method>|<context_signal_mode>
mode: positive
method_name: diffmean_decoder_only
target_model: decoder_only
adversarial_method: additive_direction
representative_coefficient: "4.0"   # the coefficient the plot selected for this family
scatter_baseline_x: general_val_bpb
strength_unit: counter_direction_magnitude
sampling: {n_completions, temperature, top_k, max_new_tokens, base_seed}
concepts:
  - concept_id: 1
    concept_label: dogs            # from the source run's concepts.json
    prefixes:
      - core_prefix_id: cp_a
        prefix_text: "..."
        datapoints:                # one per distinct strength_index, ascending
          - strength_index: 0
            adversarial_strength: 0.0
            is_control: true
            primary_margin_y: 1.5  # roundtrip runs instead emit primary_roundtrip_y
            general_val_bpb: null  # null unless general_val_bpb was computed
            completions: ["...", "..."]   # n_completions strings
```

A **prompt-token** family (`faster_gcg` / `gepa`) emits three extra fields per
datapoint, right after `adversarial_strength`:

```yaml
        datapoints:
          - strength_index: 2
            adversarial_strength: 2.0
            attacked_prompt_text: "...the GCG-attacked prompt..."  # what was generated from
            replacement_text: "...optimized span..."               # the adversarial edit ("" at k=0)
            objective_value: -0.83   # representative candidate's g (null at k=0 / gepa)
            optimization: {candidate: faster_gcg_final, max_steps: 256, accepted_steps: 12, visited_sequences: 41}
            is_control: false
            primary_margin_y: 0.42   # or primary_roundtrip_y in roundtrip runs
            general_val_bpb: null
            completions: ["...", "..."]
```

Each emitted datapoint MUST have a matching `per_candidate` metrics row (it is a
hard error otherwise) so `is_control` and the active steering metric
(`primary_margin_y` or `primary_roundtrip_y`) are never silently null; only
`general_val_bpb` is legitimately nullable. For a prompt-token family a missing
representative attack is likewise a hard error (a GCG sample must never be shown
without the prompt that produced it).

## The universal per-concept trigger

Faster-GCG optimizes **one shared adversarial trigger per concept**, not a
bespoke attack per prefix. A *concept group* is the set of attack rows sharing
the 7-tuple `(concept_id, mode, method_name, target_model, coefficient_str,
adversarial_method, strength_index)` — i.e. all the rows that
differ only in `core_prefix_id` and the rich×avoid pair. Every row in the group
shares one trigger length `k = editable_token_count = adversarial_strength`.
`strength_index` is only the ordinal in the configured token-count grid (for
example, the default grid is non-linear), and exists to keep artifact keys stable.
The constant-`k` relationship is fail-loud: `editable_token_count` must agree
across every prefix in a group.

For each non-control group (`k > 0`):

1. **Init.** The trigger is initialized to the last-k source tokens of the
   group's lexicographically-smallest `core_prefix_id` prefix (a real,
   in-distribution, fully deterministic start).
2. **Optimize over a minibatch.** Up to `max_steps` (default **256**) steps
   deals `universal_minibatch_size` (default **16**) of the concept's
   `(concept_rich, concept_avoiding)` pairs **without replacement**, so over each
   `ceil(n_pairs / minibatch)`-step epoch every rich and every avoid sample is
   covered at least once. The epoch permutation uses
   `random.Random(sha256(group_key | seed{master_seed} | epoch{epoch}).hexdigest())`.
   Candidate sampling has a separate, step-specific RNG seeded from the first
   16 hex digits of `sha256(group_key | step | master_seed)` interpreted as an
   integer. Per pair the
   trigger is substituted into that pair's last-k span, the objective gradient
   `grad_contrast − grad_desired` is computed (two forwards: rich, avoid) and
   sliced to its `(k, D)` last-k block; the per-pair blocks are **mean**-pooled
   into one shared `(k, D)` trigger gradient. Candidate triggers are ranked in
   trigger-local coordinates (on a BOS-padded view so the shared ranking
   primitive's BOS guard holds; BOS carries no gradient/ranking signal), sampled,
   and the candidate with the best **minibatch-aggregated** objective (mean of
   per-pair `g`) is accepted iff strictly better than the current trigger.
3. **Substitute everywhere.** The final trigger is written into the last-k span
   of **every** prefix-pair in the group (length-preserving
   `replace_editable_tokens`) → one candidate per pair, all carrying the same
   trigger; `candidate_id`/digests stay unique per pair. The group-level
   typed `candidate_metadata_json` payload carries the group's
   `max_steps / accepted_steps / visited_sequences` (so the per-datapoint
   `optimization` YAML field is group-shared).

For a **full-batch group** (`n_pairs <= universal_minibatch_size`), the first
ranked replacement pool for the current trigger is held fixed. Rejected and
empty-sample steps reuse that small proposal mapping, while candidate sampling
still uses the real step index and therefore a fresh deterministic RNG stream.
An accepted candidate immediately invalidates the pool; the next executed step
ranks the first pool for the new trigger. This is intentionally the fixed-pool
algorithm, not a claim of legacy multi-member output identity: the full member
set is reshuffled every step, and float32 mean reduction can change with member
order even when the mathematical set is unchanged.

Sampling remains bounded weighted rejection. An empty return can therefore be
unlucky when a low-weight unvisited sequence remains; emptiness alone never
terminates the search. Only after an empty sample does the optimizer derive every
complete single-token candidate sequence from the cached proposals, digest it in
the same BOS-padded space as `visited`, and stop when all are already visited.
The terminal empty `step_done` event is emitted before `group_done`. Groups with
`n_pairs > universal_minibatch_size` keep the per-step reshuffled-minibatch
gradient and ranking path and never use the exhaustion shortcut, because their
member subset changes across steps. Thus `max_steps` remains the configured
budget and artifact metadata value, while progress events report only executed
steps; `accepted_steps` and `visited_sequences` keep their existing meanings.

The production path batches minibatch gradients and candidate-objective scoring,
while preserving candidate evaluation, strict-improvement acceptance,
full-group retained-best evaluation, and first-on-tie selection. The test suite
pins the single-pair legacy oracle where its fixture does not exercise fixed-pool
ordering drift and separately checks the batched surrogate loss/gradient paths
against single-sample references.

The shared trigger is also surfaced once per group in
`universal_triggers.parquet` (see [Outputs](#outputs)), which both the attacked
`general_val` stage and the samples generator read.

### Automatic Faster-GCG group resume

Production model-backed `build_candidates` runs checkpoint every completed
non-control Faster-GCG concept group automatically. The durable unit is the
existing one-row `UNIVERSAL_TRIGGERS_SCHEMA` trigger record, installed before
the optimizer emits `group_done`; candidate rows are deliberately not stored.
On restart the builder validates and replays each completed trigger through the
same `TokenTriggerEditSpace.finalize` owner, reconstructing candidate IDs,
metadata, substitutions, and row order exactly. A group interrupted before its
trigger shard was installed is recomputed; a group installed immediately before
a crash is restored. Restored groups perform no objective, gradient, ranking, or
sampling calls. If every group is restored, the lazy allowed-embedding pool is
never materialized.

The immutable store lives at
`candidates/universal_gcg_checkpoint/`: canonical `manifest.json` binds the
source contract, attack rows, evaluation population, optional cluster contract,
complete Faster-GCG config and objective settings, resolved target checkpoint
bytes/effective metadata, tokenizer/statistics/context/bag contents, and the
expected group-key set. Group shards are content-addressed Parquet files under
`groups/<two-hex-bucket>/`. Any semantic mismatch or malformed/duplicate shard
raises with the differing field and tells the operator to preserve the old run
and choose a fresh `run_tag`; state is never silently overwritten, adopted,
migrated, or deleted.

The checkpoint identity intentionally has no separate GCG algorithm marker or
Git SHA. A code revision alone therefore does not invalidate same-tag shards.
When crossing the legacy per-step proposal behavior and the fixed-pool behavior
documented above, use a fresh `run_tag` and record the producing Git SHA; reusing
one explicit tag across those revisions is an unguarded provenance risk and can
mix restored legacy groups with newly optimized fixed-pool groups.

The numeric runtime fingerprint is diagnostic rather than semantic. A restart
on another runtime warns and records both hashes but is allowed; byte-identical
final artifacts are guaranteed only for the same runtime fingerprint. Shards
remain after success, one rank-0 writer per `run_tag` is the supported contract,
and the final `candidates.parquet` / `universal_triggers.parquet` remain the sole
downstream authority. GEPA reruns after a restart, while no-op and
additive-direction rows are rebuilt. The model-free `embedding_weight` test path
does not create checkpoint state. A process already running code from before
this feature cannot acquire checkpointing retroactively; let it finish or start
a separately tagged run with current code.

### Universal-GCG cost and the timing-probe config

Faster-GCG is wired into production but is **opt-in**: the practical default
adversary is `additive_direction`. The optimization unit is the concept
**group**, not the attack row, so the cost no longer multiplies by the prefix
count: one trigger is optimized per `(concept × coefficient × k × method)`
group, then cheaply substituted into all that group's prefixes. Each objective
item is evaluated against both the rich and avoiding completion. For `C`
candidates and `M` pair members, one step therefore scores `2 × C × M` model
sequences: `C=64`, `M=7` means **896**, not 448. Pair members cannot be
coalesced because each pair has its own intervention. With candidate device batch
size `B`, candidate evaluation requires `M × ceil(2C / B)` microforwards:
**56** at `B=16` and **28** at `B=32` for the same 64-candidate, 7-member step.
The gradient portion is one grad-enabled forward plus an embedding-only VJP per
completion, with model parameter gradients disabled. Candidate evaluation is a
separate dominant cost: it requires those `M × ceil(2C / B)` microforwards.
The candidate-emission and scoring stages still scale with the full prefix×pair
count, but they do **not** run the optimizer.

The cost levers are multiplicative:

- `optimized_token_counts` `7 → 2–3` (`≈−3×`) — fewer trigger lengths.
- a **reduced coefficient grid at attack-row-generation time** (`10 → 1`,
  `−10×`) — i.e. *optimize* one coefficient, generating fewer groups. This is NOT
  the post-scoring representative-coefficient plot reduction (which optimizes
  the full grid and only collapses the figure); see
  [Representative coefficient](#representative-coefficient).
- `max_steps` (the per-group budget; an eligible group may execute fewer steps
  after exact pool exhaustion) and a single base method / target.

`faster_gcg.candidate_eval_device_batch_size` changes only these candidate-loss
microbatches. `null` inherits the top-level `device_batch_size`; it does not alter
gradient, GEPA, final scoring, or general-validation batch composition.

### Shared-prefix KV candidate evaluation

Shared-prefix KV reuse is the only production candidate-loss implementation for
Faster-GCG and GEPA. The previous batched full-sequence implementation is kept
only as a test oracle. The candidate vector and singleton current-target/
acceptance calls therefore always use the same numerical path, while gradients
remain uncached. Faster-GCG can use its scoped candidate batch override; GEPA
continues to use the top-level device batch size.

For a final length-preserving editable span of width `k`, the cached scorer sets
`S = n_full_prefix - k`, prefills the fixed `[:S]` prefix once, materializes that
cache into independent candidate rows, and teacher-forces only the trigger plus
completion suffix. It refuses non-final/dropped editable spans, `k <= 0`, an
empty reusable prefix, position or rotary overflow, malformed masks, and
non-finite in-mask loss. K/V tensors are call-local: no cache survives in the
objective callback or across a GCG step.

Reuse is exact-grouped, never heuristic. A group must share `S`, exact prefix
token IDs, and, for encoder-decoder targets, the exact prefix `seg_map`, feature
IDs, magnitudes, masks, tensor shapes, generation slot, feature-vocabulary
mapping, and cluster contract. Rich and avoiding completions can require separate
prefills when their lengths cause different left truncation. Encoder routing for
the suffix is sliced from the validated full map rather than rebuilt.

Cached and test-oracle attention may use different kernels and batch shapes, so
objective floats are compared explicitly rather than assumed byte-identical.
Shared-prefix scoring is an intentionally versioned search algorithm, so exact
legacy argmins, accept-margin signs, and trajectories are not acceptance
requirements. Rollout acceptance instead requires finite values, deterministic
shared-prefix reruns, and an explicit counterfactual quality comparison: on each
shared-prefix candidate pool report the legacy-oracle rank and BPB gap of the
chosen candidate; on a common pool also report reciprocal rank and top-k
overlap; and score the final retained trigger and its improvement with both
scorers. The production real-checkpoint drift gate remains at most `0.2` BPB per
sample and no more than 1.25 times the same sample's measured legacy
batch-shape/grad-enabled envelope (with a `1e-4` floor). Each supported
hardware/backend must pass separately. Use the test-only oracle only for this
evidence; it is not a production fallback.

The implementation change does not alter the artifact schema. Always use a fresh
`run_tag`; clustered artifact immutability must reject an attempted overwrite of
a legacy artifact. No artifact migration or deletion is needed.

The token-ranking path has two distinct cache scopes. One **lazy,
search-scoped allowed-embedding cache** belongs to the whole universal
Faster-GCG builder. On the first active ranking step it builds the allowed-id
tensor and gathers the corresponding surrogate embedding rows on the configured
model device; later ranking calls reuse those exact tensors. Separately, each
eligible full-batch group retains only its small Python proposal mapping for one
current trigger. Acceptance drops that proposal mapping, and group completion
drops it unconditionally. Control groups and `max_steps=0` materialize neither
cache. The raw ranking API remains available for edit spaces whose allowed set
changes with the current target.

For `d12_base`, the surrogate concatenates the token embedding and value
embeddings, so its measured shape is `(32,768, 5,376)`, not `(32,768, 768)`.
The 32,734-row float32 allowed pool plus its int64 ids occupies 671.55 MiB and
stays resident until the universal builder returns. PR #1225 moved the ranking
pool and matrix work to the model device, reducing ranking's measured live-sample
share from about 20.3% to about 2.2% in the available post-change profile. The
older CPU-ranking description and roughly 24% expected saving are therefore
obsolete. Fixed-pool reuse primarily removes repeated proposal construction and
terminates exactly exhausted tails; its end-to-end value must be measured at the
complete builder boundary rather than inferred from the pre-#1225 profile.

Issue #1224 evaluated retaining the first ranked pool per current target for
full-batch groups and stopping only after exact derived-sequence exhaustion. The
comparison used baseline `dbbf2101a570817539bdbc391e016e2bec274f64` and
experimental revision `dc9d793f67d7d5d586cdcdce125f0ec2bd7d5321`, before
the group-resume merge above. On a pinned 12-group production manifest (two
eligible groups for each
`k in {1, 2, 4, 8, 16, 32}`), three alternating fresh-process comparisons
reduced the complete builder median from 1,129.04 s to 1,082.49 s: **4.12%**,
with paired improvements from 3.84% to 4.71%. The initial evaluation plan set a
5% retention gate, but the implementation is retained because all three matched
runs improved and the change stays narrowly scoped to eligible full-batch
groups. It reduced executed steps from 3,072 to 870 and ranking calls from 3,072
to 428, while candidate evaluation remained dominant. Its candidate and trigger
tables were byte-identical on this sample; the float-order counterexample above
means that sample result is not a global output-parity claim.

Benchmark one concept group on the real model before committing to a grid — the
per-group cost depends on `max_steps`, `minibatch`, and the model's GCG forward
speed. The committed
`configs/eval/axbench_bpb_adv/axbench_bpb_adv_timing_probe.yaml` is a cheap cost /
progress probe: `faster_gcg` only, `max_steps=2`, `candidate_batch_size=4`,
`top_k=8`, `universal_minibatch_size=16`, `optimized_token_counts=[0, 1, 2]`,
`base_methods=[diffmean_decoder_only]`,
`representative_target_methods=[feature_diffmean_encdec]`, and
`target_models=[decoder_only, encdec]`. It is a **cost / progress probe**, not a
research run — scale the knobs back up for real results.

## Context Mode

Prompt-token attacks use a single context mode, `fixed_context`: keep the
original SAE feature ids/buckets and project the original segment character
spans onto the length-preserving attacked prompt. There is no
`recomputed_context` mode, and context mode is not a swept axis.

Additive direction does not mutate prompt tokens, so its rows record
`context_signal_mode=not_applicable`.

## General-validation BPB (`general_val_bpb`)

The class margin only measures steering on concept-specific completions. It cannot
tell whether a steering signal (or the adversarial pressure used to break it) is
*also* degrading the model's GENERAL language-modelling ability. The optional
`score_general_val` stage answers that: it scores the model's teacher-forced BPB on
a held-out GENERIC validation set (the last FineWeb-Edu shard — the pretraining
distribution) under the **same** steered + adversarial intervention used for the
margin, reusing the identical scoring kernel (`score_triple`). `general_val_bpb` is
therefore an apples-to-apples companion to the active steering metric
(`primary_margin_y` or `primary_roundtrip_y`): same intervention, **under the same
attack**, on different (concept-agnostic) text. Lower is better — less collateral
damage.

**Attack-aware (non-flat across the GCG axis).** The universal per-concept trigger
makes `general_val_bpb` measure general ability *under the attack*. The validation
docs are BOS-prefixed, matching the concept path and the BOS-context the steering
directions were computed in — a change that also shifts the *clean* general_val
baseline, since the model now sees a leading BOS. The BOS is additive rather than
part of the raw text encoding: the code builds
`prefix = [BOS] + real[:prefix_tokens]` and
`output = real[prefix_tokens : prefix_tokens + output_tokens]`, so the scored
continuation window is unchanged. General-val rejects raw validation encodings
that already start with BOS, validates every constructed prefix has exactly one
leading BOS, and rejects trigger substitutions that would create a double-BOS.
For each `(concept, k)` the group's optimized k-token trigger (read from
`universal_triggers.parquet`) is substituted into the **last-k non-BOS tokens** of
every DECODER-ONLY doc — the same length-preserving last-k substitution the
prefixes get. Since the trigger differs per `k`, `general_val_bpb` is **non-flat
across the GCG axis**: `k = 0` is the clean BOS-prefixed baseline, and `k > 0`
reflects the attacked docs. (Encdec general_val is NOT yet attacked — it scores
the clean BOS-prefixed docs; attacked-encdec general_val is a deliberate follow-up.
A missing trigger for an attacked `(concept, k)` is a hard error — fail-loud.)

Configured under `general_val_bpb`:

```yaml
general_val_bpb:
  enabled: true
  n_documents: 64        # held-out docs scored per intervention
  prefix_tokens: 256     # context prefix (must span >= 1 Gemma segment for encdec)
  output_tokens: 128     # scored continuation length
  max_interventions: 0   # cap on distinct interventions scored (0 = no cap)
  seed: 0
```

Enabled by default, since the default `scatter_baseline_x=general_val_bpb`
frontier needs it. Needs a base_data val shard (`python -m nanochat.dataset -n
1`). Set `enabled: false` to fall back to the pre-feature byte-identical
behaviour (the stage no-ops and the `general_val_bpb` aggregate column stays
null) — but then also move `scatter_baseline_x` off `general_val_bpb`, or
`validate_config` raises.

**Document selection (`n_documents`).** `n_documents` means SCORED documents, not
candidates fetched. Validation texts are streamed one at a time (the split is
never materialized in full); the stage keeps the first `n_documents` documents
long enough to supply `prefix_tokens + output_tokens` real tokens (with a
nonzero output byte count) and stops as soon as it has that many. Selection is
therefore deterministic for a fixed corpus and fixed knobs, independent of how
many candidate texts happen to be available — the old contract instead fetched
exactly `n_documents` candidates up front and silently scored however many of
them turned out to be long enough. An empty corpus (no val shard downloaded)
raises an actionable error pointing at `python -m nanochat.dataset -n 1`;
exhausting the stream before finding `n_documents` survivors raises reporting
how many were found of how many were requested (and how many texts were
scanned) — lower `n_documents` / `prefix_tokens` / `output_tokens`, or provide
more/longer documents.

**Cost / dedup.** The metric is computed once per unique intervention SIGNATURE
and broadcast to every matching `(target, method, concept, coefficient,
adversarial, adversarial_strength)` row. For a prompt-token attack the signature now
forks on `faster_gcg:k{k}` (it used to collapse all GCG strengths to one
`base` intervention), because each `k` substitutes a **different** trigger
into the docs — so `general_val_bpb` is genuinely non-flat across the GCG axis.
`k = 0` (no trigger) is the shared clean baseline. The additive counter-direction
(a residual hook, magnitude > 0) additionally moves the steered intervention
itself. So the work is `(#GCG (k>0) forks per concept) + (#steering
settings) + (#additive magnitudes > 0)` distinct interventions, each scored over
`n_documents` docs (the attacked decoder-only forks re-decode and re-score their
substituted docs). `max_interventions` caps it (deterministic seeded subset;
dropped count is logged, never silently truncated).

**Corpus signature (`general_val_corpus`).** Every row the stage writes also
carries a per-target corpus signature, `nd=<n>;nb=<bytes>;h=<32hex>` — scored
document count, total scored-output bytes, and a 32-hex-digit hash over the
ordered scored-doc identities (sample id, prefix/output token ids, byte count)
plus the slicing knobs. It is computed separately for `decoder_only` and
`encdec` from the docs each target actually scored, and describes WHICH
documents were scored — never the intervention (that's still the
`intervention_signature` dedup key from **Cost / dedup** above). Before writing
each row the stage checks the scored footprint against its own target's
signature, and — when both targets are active — raises if `decoder_only` and
`encdec` ended up scoring different corpora.

`general_val_corpus` sits next to `general_val_bpb` wherever the latter appears
— the per-intervention general-val table, and (via the aggregation join)
`per_candidate.parquet` / `macro.parquet` for BOTH the legacy class-margin and
the roundtrip aggregate families — nullable only in lockstep with it (null iff
`general_val_bpb` is null). Concept-averaging at the macro level raises if a
group mixes corpus signatures, and so does every figure that draws
`general_val_bpb`: comparing it across runs or families requires that every
value came from scoring the same held-out documents (issue #635).

Extracting encdec's per-doc Gemma SAE feature segments is fail-loud for the same
reason. A doc that yields no feature segments used to be warned about and
dropped — silently forking the encdec corpus away from the decoder-only one. It
now raises immediately, naming the doc, and suggests raising
`general_val_bpb.prefix_tokens` so every prefix spans at least one Gemma
segment: every scored doc must condition, or the run fails before scoring
begins.

Roofline feature-space methods also get true general-validation capability
values, not plot-time placeholders. For
`feature_diffmean_roofline_encdec`, each validation document's generation-slot
conditioning bag is the document's own real output-window SAE signal, thresholded
and discretized with the same settings as context segments, then capped to the
same positive/negative feature-count budgets used by the fair feature methods
(`ours_gcg.top_k_pos` / `top_k_neg` overrides when set, otherwise the
encoder-decoder checkpoint's `feature_selection` / `negative_selection`
geometric means). For
`ours_gcg_roofline_encdec`, that real output-window bag is the warm start for a
per-document optimization of masked LM loss plus `lambda * |bag|`, where
`lambda` is the method coefficient. The persisted roofline bags are
concept-independent and live under `general_val_bags/`. Bag metadata validates
the validation corpus, GCG knobs, extraction threshold/max length, Neuronpedia
model id/source/layer, feature-statistics path, context segment/bucket settings,
and a digest of the bucket-boundary tensor; stale same-`run_tag` bags written
under a different config are rejected rather than reused. Feature-DiffMean
roofline bags are just the real output-window signal and do not run the
metadata-only LM objective forward; GCG roofline bags still record warm and
optimized objective metadata. Nonzero additive counter-direction interventions
still fork by concept because the residual direction being applied is
concept-specific, but that
direction is loaded once per scored roofline signature and attached to each
per-document bag.

This generic-validation roofline is deliberately different from the concept
class-margin roofline. The concept roofline uses one sparse top-k
Feature-DiffMean/GCG bag per selected `(concept_id, prefix_text)` evaluation
prefix. The generic-validation roofline has no concept prefix, so it uses one
sparse real-signal bag per validation document instead; under the clean/control
setting that value is broadcast across concept rows before macro aggregation.
It is therefore an oracle for conditioning-signal availability, not a guarantee
that the model will assign lower generic BPB than a sparse fair concept bag.

**Schema v2.** `general_val_corpus` is a new column on `GENERAL_VAL_BPB_SCHEMA`,
`PER_CANDIDATE_AGG_SCHEMA`, `MACRO_AGG_SCHEMA`, `ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA`,
and `ROUNDTRIP_MACRO_AGG_SCHEMA` (issue #635), and there is no
legacy-compatibility path: a pre-635 `general_val_bpb.parquet` or
`macro.parquet` (written before the column existed) is rejected with a
schema-mismatch error — at the aggregation join (`stage=aggregate`) or in
`scripts.eval.plot_steering_strength_families`'s macro reader — rather than
being silently read with the column defaulted to null. Re-run
`stage=score_general_val` and then `stage=aggregate` on such a run before
aggregating or (re)plotting it.

**Where it shows up.** Aggregation left-joins the per-intervention table onto the
macro rows and concept-averages it (mirroring `primary_margin_y`). Two views:

- `general_val_bpb_vs_strength.png` — a companion line plot that SHARES the
  primary plot's adversarial-strength x-axis and representative-coefficient
  reduction (one line per family). Now that the trigger is substituted into the
  validation docs too, the GCG line genuinely **moves** along the strength axis.
  Read side-by-side with `primary_margin_vs_strength.png`: as the trigger ramps,
  does steering strength fall *without* general ability collapsing?
- `scatter_baseline_x: general_val_bpb` — selects general-val BPB as the
  steerability-vs-baseline scatter's x-axis (a less noisy, concept-agnostic
  "how broken is the model" axis than the single concept-completion baseline loss).

## Outputs

Run outputs are rooted at:

```text
out/eval/axbench_bpb_adv/<run_tag>/
```

Key artifacts:

- `source_contract.json`
- `evaluation_population.json` (method-independent held-out prefix/pair
  population enforced by candidate and score stages)
- `attack_rows/attack_rows.parquet`
- `candidates/candidates.parquet`
- `candidates/universal_gcg_checkpoint/manifest.json` plus immutable
  `groups/<bucket>/group_<key-sha256>_<payload-sha256>.parquet` shards (automatic
  model-backed Faster-GCG resume; retained after success and never read downstream)
- `universal_triggers.parquet` (one row per concept group; the shared Faster-GCG
  trigger plus optimization counters)
- `general_val_bags/<family>/s<sample_id>/lambda_<slug>.safetensors` (only when
  `general_val_bpb.enabled` scores roofline feature-space methods)
- Default roundtrip metric:
  - `roundtrip_scores.parquet`
  - `roundtrip_aggregates/per_candidate.parquet` (includes `general_val_bpb` when joined)
  - `roundtrip_aggregates/macro.parquet` (includes `general_val_bpb` when joined)
  - `roundtrip_aggregates/aggregation_report.yaml`
  - `roundtrip_plots/plot_series_legend.csv`
  - `roundtrip_plots/plot_index.csv`
  - `roundtrip_plots/<mode>/primary_roundtrip_y_vs_strength.png`
  - `roundtrip_plots/<mode>/roundtrip_percentile_mean_vs_strength.png`
  - `roundtrip_plots/<mode>/roundtrip_post_act_mean_vs_strength.png`
  - `roundtrip_plots/<mode>/roundtrip_pre_act_mean_vs_strength.png`
  - `roundtrip_plots/<mode>/control_primary_roundtrip_y_vs_strength.png`
  - `roundtrip_plots/<mode>/roundtrip_steerability_vs_baseline.png`
- Optional strength-family plots (`plot_steering_strength_families=true`):
  - `steering_strength_plots/plot_series_legend.csv`
  - `steering_strength_plots/plot_index.csv`
  - `steering_strength_plots/<mode>/steering_strength_families.png`
- Legacy `steering_metric=bpb_class_margin`:
  - `scores.parquet`
  - `general_val_bpb.parquet` (only when `general_val_bpb.enabled`)
  - `aggregates/per_candidate.parquet`
  - `aggregates/macro.parquet`
  - `aggregates/aggregation_report.yaml`
  - `plots/plot_series_legend.csv`
  - `plots/plot_index.csv`
  - `plots/<mode>/primary_margin_vs_strength.png`
  - `plots/<mode>/general_val_bpb_vs_strength.png` (only when `general_val_bpb.enabled`)
  - `plots/<mode>/mean_loss_concept_rich_vs_strength.png`
  - `plots/<mode>/mean_loss_concept_avoiding_vs_strength.png`
  - `plots/<mode>/baseline_primary_margin_vs_strength.png`
  - `plots/<mode>/steerability_vs_baseline.png`
  - `plots/<mode>/by_coefficient/coeff_<slug>/steerability_vs_baseline.png`
  - `plots/<mode>/by_coefficient/plot_index.csv`
- `samples/<mode>/<method>__<adv>.yaml` (only from the separate `axbench_bpb_adv_generate_samples` script; see [Per-datapoint sampled completions](#per-datapoint-sampled-completions))

### `source_contract.json` schema

`validate_source`, every standalone stage, and sample generation use one
provenance contract for the upstream Stage 0-3 source run. Unclustered sources
use globally distinct schema **v6**; clustered sources use **v7**. Both require
and hash the AxBench seed-v2 pair and `seed_pool_manifest.json`, validate exact
BOS-free source-slice semantics with the nanochat tokenizer, and bind the
complete baseline tree. Direction semantics and the directory hash are derived
from the same descriptor-bound payload traversal.

All present DiffMean/linear-probe artifacts are validated method-agnostically.
Their path and canonical `concept_id`, `mode`, `target_model`, and `layer`
metadata must agree, and the entire source must have one training layer.
Runtime requirement resolution then proves that every selected scalar baseline
and every nonzero additive counter-direction exists for both modes and all
requested concepts/targets. Directionless control/prompting-only sources are
legal and record `direction_training_layer: null`; selecting any
direction-consuming method against such a source is a hard error. The
configured `diffmean_and_probe_layer` is a strict non-negative integer indexing
the target nanochat/encoder-decoder residual stack, not the Gemma SAE layer,
and must equal the derived source layer.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | `6` for an unclustered source; `7` for a clustered source. |
| `source_eval` | str | Upstream eval that produced the source run (`axbench_base_stage_0_3`). |
| `bpb_only` | bool | Must be `true`. |
| `source_run_dir` | str | Absolute path of the validated source run. |
| `seed_schema_version` | int | Required AxBench seed schema version (`2`). |
| `seed_fingerprint` | str | SHA-256 semantic identity shared by the committed train/eval seed pair. |
| `seed_token_semantics` | str | Exact BOS-free source slices; one pipeline-owned BOS only at model input. |
| `artifact_byte_sizes` | dict[str, int] | Byte size of each required artifact. Files: `stat().st_size`. The `baselines/` directory: recursive sum of every non-directory descendant's byte size. |
| `artifact_entry_counts` | dict[str, int] | Directory artifacts only (today just `baselines/`): recursive count of nested entries (directories + files). |
| `artifact_hashes` | dict[str, str] | SHA-256 of each artifact. Ordinary files use exact bytes. Seed safetensors use canonical metadata plus tensor name/dtype/shape/content, independent of header/data ordering; the seed manifest uses canonical JSON excluding its exact-byte transport checksums. The directory hash folds in each entry's relative path and a dir/file marker, plus each file's byte size and content. |
| `direction_training_layer` | int \| null | The one validated training layer for every present direction, or `null` only when no direction artifact exists. Included in `contract_hash`. |
| `feature_contract_schema_version`, `feature_contract_fingerprint`, `candidate_provenance_sha256`, `stage2_feature_contract_fingerprints`, `baseline_training_corpus_sha256` | strict cluster fields | Required only by schema v7 and forbidden in v6. They bind the clustered feature owner, candidate provenance, both Stage 2 artifacts, and both training corpora. |
| `contract_hash` | str | sha256 over the canonical JSON of all other contract fields (excluding `contract_hash` itself). |
| `n_dropped` | int | Always `0` for successful validation. Retained for schema compatibility; incomplete prefixes raise before a contract is written. |

Field types are exact: booleans cannot stand in for integers, and floats,
numeric strings, negative layers, malformed hashes, or cluster fields in v6
are rejected. Once v6/v7 is installed in a run, revalidation must reproduce the
same canonical payload exactly; source drift requires a new run. Production
evaluators and sample generation reject v4/v5. Run
`uv run python -m scripts.eval.validate_axbench_bpb_adv_output_root` to verify
that the production root contains only current contracts and no migration
control-plane entries. Historical v4/v5 and incomplete smoke outputs from the
issue-1232 closeout are retained in a dated archive outside that root; they are
not rewritten or deleted.

Durable follow-up: after the direction-baseline version is validated, extend the
adversarial family coverage to LoRA, GEPA-as-steering, GCG-as-steering, and
richer SAE-feature additive counter-interventions.

## Profiling

The production eval (`scripts.eval.axbench_bpb_adv_eval`) is instrumented with
opt-in `profile_region(...)` / `record_counter(...)` calls. These are no-ops
unless profiling is enabled, for example with `+profile.enabled=true`, so the
default production path does not write profiling artifacts.

To capture production-faithful wall-clock, CPU RSS, CUDA memory, GPU
utilization, and counter data, run the normal eval CLI with profiling enabled:

```bash
uv run python -m scripts.eval.axbench_bpb_adv_eval \
  source_run_dir=out/eval/axbench_base/<run_tag> \
  output_root=out/eval/axbench_bpb_adv \
  feature_dir=out/data/features/... \
  feature_statistics_path=out/data/features/.../feature_statistics.safetensors \
  stage=all \
  +profile.enabled=true
```

Outputs land alongside the eval artifacts under:

```text
out/eval/axbench_bpb_adv/<run_tag>/profile/
  events.jsonl    # one JSON object per region / counter event (streaming)
  summary.yaml    # aggregate region + counter table (human-readable)
  summary.csv     # flat table for spreadsheets / diff
```

Each of the 8 stages — `validate_source`, `prepare_context_cache`,
`prepare_attack_rows`, `build_candidates`, `score`, `score_general_val`,
`aggregate`, `plot` —
appears as a `region` in the summary with total/min/max/mean wall-clock +
per-region CPU RSS, CUDA memory, and GPU utilization summaries. Per-stage and
nested counters record row counts, cache hits/misses, prompt/output-token
totals, context-cache rebuild/skip counts, candidate construction subregions,
roundtrip generation/probe work, and general-validation signature/doc counts.
The `build_candidates`/`score` token counters are gated on an active profiler
so the default path never pays the per-row scan.

Shared-prefix candidate evaluation adds nested regions for `layout_grouping`,
`encoder_preparation`, `prefill`, `materialized_kv_copy`, and
`suffix_forward_bpb`. Its counters report compatibility groups/prefills, legacy
full versus suffix token positions, copied K/V bytes, maximum source/destination
cache dimensions, the fixed `shared_prefix_kv` algorithm, and effective candidate
device batch size. It does not add per-candidate logs or retain tensors for
profiling.

### Issue #1257 dual-metric benchmark

To benchmark the current bottleneck without changing the candidate population,
run the dedicated harness against an existing finished run:

```bash
  uv run python -m scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip \
  --output-root out/eval/axbench_bpb_adv \
  --run-tag l22_f65k_clustered_std50_m_gcg2 \
  --benchmark-dir out/benchmarks/issue-1257 \
  source_run_dir=... feature_dir=... \
  feature_statistics_path=... decoder_only_target=... encdec_target=...
```

The harness launches the production CLI in a fresh subprocess for
`bpb_class_margin` and `roundtrip`, reuses the same ordered
`candidates/candidates.parquet`, and writes score/profile outputs below the
benchmark directory rather than overwriting the finished run. It records the
candidate file hash, schema hash, ordered-ID hash, wall-clock candidates/sec,
generated-tokens/sec, profiler counters, and a baseline-loss comparison. The
two metric runs are refused if the candidate file changes between them.

Use `--metrics bpb_class_margin` or `--metrics roundtrip` to run only one
scorer; the default `--metrics both` retains the two-run behavior. A
single-metric report explicitly records that the retained-baseline comparison
was skipped, because that comparison is meaningful only when both score tables
were produced.

`--sample-size N` selects a bounded population from the existing candidate
table, stratified by `(method_name, target_model, mode)`. Allocation is
proportional by largest remainder and gives every stratum at least one row;
sizes below the stratum count or above the source row count are rejected rather
than clamped. Within each stratum, candidate IDs establish a stable base order
and a PCG64 stream derived from `--sample-seed` (default 0) selects without
replacement. The selected indices are restored to source-table order, and the
subset preserves the complete Arrow columns and metadata, except that clustered
subsets recompute their row-dependent precomputed-bag-set identity stamp.
Identical source tables, sizes, seeds, and locked NumPy/PyArrow versions
therefore produce identical subset parquet hashes; the report records both
dependency versions in its `sample` block.

Bounded runs never invoke a scorer against the production run directory. Each
metric receives a fresh
`<benchmark-dir>/derived/<metric>/<run_tag>/` tree: immutable inputs are
hardlinked, output artifacts are omitted, and the subset candidate table is
written as a separate inode. The harness fingerprints both that subset and the
production candidate table after the subprocess exits, and fails if either was
mutated. Without `--sample-size`, the scorer continues to consume the original
full candidate table directly.

The generation profile includes engine-construction, encoder/prefill,
decode-step, prompt-token, generated-token, condition-preparation, and
steered/control cache-hit counters. These counters are observational and are
no-ops when profiling is disabled. Before trusting a future optimization
benchmark, run an explicitly slower reference configuration and verify that the
sentinel counters increase; the harness exposes
`validate_regression_sentinel()` for that check.

**Process model:** The production CLI calls `compute_init`; under `torchrun`,
rank 0 alone binds the profiler and nonzero ranks still participate in
distributed stages and barriers without writing profile files. The benchmark
wrapper remains available for lightweight mocked/single-process smoke tests,
but it is not the recommended source of full-run performance numbers.

### Round-trip performance characteristics (issue #1260 campaign)

The issue #1260 campaign measured the production score stage on an RTX 4090
against fixed, method/target/mode-stratified subsets of the real `m_gcg2`
population. PR3 replaced per-sample generation with one shared prefill and
lockstep batched decode per candidate side; PR4 then shared semantically
identical controls and memoized exact probe requests. The measured stack was:

| Population / region | Before PR3 | After PR3 | After PR4 |
|---|---:|---:|---:|
| strat64 score-stage wall, paired median | 431.9 s | 389.1 s | — |
| strat64 generation region | 73.0 s | 19.0 s (3.8x) | — |
| strat256 score-stage wall | 823.5 s | 606.1 s | 581.2 s |
| strat256 generation region | 280.6 s | 75.7 s (3.7x) | — |

PR3 reduced physical engine constructions and prompt prefills by exactly 4x,
matching `samples_per_candidate=4`; requests, seeds, row order, and per-sample
accounting stayed live and unchanged. The PR4 offline audit found 1,386 unique
control batches across 16,632 candidates in `m_gcg2`, a 12.00x reduction. Live
`control_unique_batches`, `control_unique_completions`, requested-probe,
unique-probe, and cache-hit counters make both kinds of sharing observable.

PR7 follows that measured stack by replacing per-row generators and lockstep
filler decode with one vectorized batch RNG plus active-row/KV compaction before
the next decode forward (never after the final budgeted draw). This is an
owner-approved reseeding transition, not an exact sampled-output optimization;
greedy generation is the bitwise forward/KV-copy oracle, and N=1 sampled
generation remains exactly aligned with `generate_stream`.

PR8's sync property is intentionally loop-local: a quiet decode iteration has
one loop-local stop-detection sync at `bool(finished_now.any())`. It is not an
end-to-end per-iteration sync count. The call path still synchronizes in the
nanochat finite-logit `.all()` (`nanochat/sampling.py:22-23`); twice per
successful cached GPT forward through `get_pos()` (`nanochat/gpt.py:545,582`),
whose implementation calls `.item()` (`nanochat/engine.py:152-154`); and in the
SDPA fallback's per-layer B>1 uniform-position `.all()` plus position `.item()`
(`nanochat/flash_attention.py:526-532`). Each active-batch compaction also has
one steering-owned cache-position read
(`steering/eval/inference_engine.py:191`). The SDPA fallback forms transposed
views and casts K/V only when cache and query dtypes differ
(`nanochat/flash_attention.py:539-549`). Removing the nanochat-owned host reads
requires an owner-approved nanochat change and remains outside this work.

`scripts/benchmark/bench_generate_samples.py` measures the production call with
five warmups and synchronized aggregate timing, plus per-iteration median/p95,
end-to-end token throughput, decode-region seconds, and profiler counters. Its
default `--stop-mode none` consumes the full budget. The measurable compaction
workload uses `--stop-mode compaction`, fixed seed 1260, temperature 1.5, and
the deterministic stop set `0, 32, 64, ...` below the tokenizer vocabulary
size; it fails loudly if no finish event or active-batch compaction occurs.
Those constants are calibrated against the pinned `d12_base` step-54600
checkpoint, where two rows finish at different steps (two compaction events)
and the survivors run to the full budget. At the production temperature 0.7
the repetitive benchmark prompt keeps all sampled rows in lockstep, so every
finish is simultaneous and the Case-B compaction path never executes.

Probe lifecycle remained the dominant fixed cost. `probe_load` measured 105.9 s
cold and 75.7–83.2 s warm per score-stage process, with the GPU idle for about
93% of that CPU-bound Hugging Face/TransformerLens deserialization. The 512
probe forwards took about 32.5 s total (63 ms/forward for roughly 450-token
texts). Generation peaked near 17.1 GiB of CUDA memory on the 24 GiB card and
the probe requires about 9 GiB, so loading the probe during the generation tail
is not safe on 24 GiB hardware. The stage's never-co-resident target/probe
handoff remains the contract; overlap is a follow-up only for high-VRAM systems
such as a 98 GiB Blackwell card.

On the 4090, which had neither FA4 nor FA3 available, the no-KV-cache
`flash_attn_func` calls used flex-attention/Triton for teacher-forced baseline
BPB scoring and encoder-decoder encoder forwards. KV-cache decode already used
SDPA throughout. At the identical PR3 revision and pinned strat64 population,
setting `NANOCHAT_ATTN_IMPL=sdpa` reduced wall time from 389.1 s to 342.7 s
(−12%): baseline BPB fell from 32.6 s to under 16 s, probe scoring from 27.6 s to
24.4 s, and generation from 19.0 s to 17.9 s. It also removed the
flex-attention/Dynamo recompilation warnings tracked by issue #1260 items 5–6.
For round-trip evaluation on hardware without FA4/FA3, use
`NANOCHAT_ATTN_IMPL=sdpa`. This is an operator-selected attention backend, not a
Hydra field.

Attention backends are not numerically interchangeable: the SDPA score table
differed from the flex/Triton table, so changing the backend is a measured-drift
change, not a bit-parity-preserving optimization. The campaign differential
method established a seed-null rerun floor, exact-compared the frozen identity
and schema columns, counted token-stream divergences, and measured score deltas.
There were zero first-token divergences. Later token differences were consistent
with TF32/bf16 rounding changing near-tied sampling logits when the shared-prefill
implementation changed GEMM batch shapes. Preserve the backend when exact
reproduction matters; otherwise re-run the differential gate and report drift.

The retained evidence is under `out/benchmarks/issue-1260/`: `before-*` contains
the sequential baseline, `after-p3-*` the shared-prefill/batched-decode and SDPA
A/B runs (including `after-p3-sdpa-strat64` and the independent-seed run), and
`after-p4-*` the control-sharing results. Each benchmark directory retains
`benchmark.json`, the round-trip score table, `profile/events.jsonl`,
`profile/summary.yaml`, and `profile/summary.csv`. See
[`docs/performance/profiling.md`](../performance/profiling.md#axbench-bpb-adv-round-trip-score-stage-benchmark)
for the comparison protocol. FA4/flash-attn-4 installation remains a
hardware-gated Blackwell follow-up.

Issue #657 tracks the full l12 frontier profiling/optimization work. Any cache
reuse or optimization following the profile must keep source contracts, target
identity, feature identity, corpus signatures, and relevant method parameters
in fail-loud cache keys.
