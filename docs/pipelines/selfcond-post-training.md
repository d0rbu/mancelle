# On-Policy Self-Conditioning Post-Training

This stage post-trains an encoder-decoder checkpoint on conditioning selected
by its own feature-prediction (FP) head. The no-grad rollout decides which
features condition later segments; a two-pass, one-hop straight-through
estimator (STE) then trains either the FP head alone or the full model from the
language-model loss.

Use this stage after encoder-decoder pretraining and before
[self-conditioned generation](self-conditioned-generation.md). It is not a
replacement for pretraining. The default data source consumes the same packed
rows and selection artifacts recorded by the base checkpoint; SFT mode instead
consumes the SFT conversations and features recorded by an SFT checkpoint.

## Prerequisites

The base checkpoint must satisfy every row below. Startup fails loudly rather
than silently changing semantics.

| ID | Required checkpoint/data property | Reason |
|---|---|---|
| C1 | A `feature_pred_head` is present and its config is recoverable from metadata. | The head is the rollout policy. |
| C2 | Pretraining mode: the checkpoint has no compact `FeatureVocab`. SFT mode: compact masked/clustered checkpoints ARE supported — the rollout selects among the compact rows (cluster representatives), translates them to original SAE ids through the checkpoint's `FeatureVocab` (the same contract as inference-time compact feedback), and both STE forms operate in compact row space. Canonical mask/cluster metadata must be recoverable and must agree id-for-id with the state dict's vocabulary. | Compact FP scores are indexed by effective rows; SFT mode carries the explicit compact target contract, pretraining mode still does not. |
| C3 | `num_buckets == 2`. | Predicted positives use bucket 0 and negatives use bucket 1. |
| C4 | `negative_selection` is non-null in the checkpoint config. | Bucket 1 must already mean “negative,” not “high positive.” |
| C5 | Pretraining mode: no feature clusters. Both modes: no gradient-attribution ranking; `label_dropout == 0`; `next_segment_conditioning_offset == false`. SFT mode accepts clustered checkpoints (rollout actions are representatives by construction and are asserted representative-only). | Clustered interactions carry an explicit contract only on the SFT path. |
| C6 | In pretraining mode, matching precomputed packed training rows are available and `use_precomputed_batches` is true. In SFT mode, the checkpoint's SFT conversations and extracted features are available. | The pretraining live loader is outside v1; SFT has its own source-aware generator. |
| C7 | The rollout pool mode is `max_pool`, `last_token`, or `full_segment`. | `best_token` needs the target and cannot define an on-policy action. |
| C8 | No splice-counterfactual setup or spliced derived feature directory. | Delta-aware dual validation is outside v1. |

The default encoder-decoder pretraining config produces a feature mask and
disables the FP head, so it does not produce an eligible checkpoint. The
default packed-pretraining data source requires a deliberately trained,
uncompacted FP-head checkpoint. The SFT source may instead consume a compact
masked/clustered SFT checkpoint only when its canonical metadata agrees exactly
with the state-dict vocabulary; neither path widens a compact checkpoint after
the fact.

## Data sources

`selfcond.data_source` is explicit: `pretraining` (the default) preserves the
original packed-row path, while `sft` selects converted SFT conversations and
their extracted features. Setting any SFT-only key while the selector remains
`pretraining` raises instead of being ignored.

SFT mode requires an SFT-shaped base checkpoint leaf passed explicitly through
`base_checkpoint_dir`. Tag discovery stops at the parent pretraining family and
cannot locate SFT leaves. The SFT checkpoint must have current metadata,
including top-level `dataset` and `encoder_config` plus recoverable conditioning
metadata; current writers persist the latter under `pretrain_conditioning`.
Legacy thin SFT checkpoints that omitted those values cannot seed
self-conditioning. Re-run SFT from the eligible pretraining base with current
code to produce a usable leaf.

Prepare the selected dataset (`standard`, `capabilities`, or `safety`) as
converted train conversations, extracted SFT TRAIN features, and train
`feature_statistics.safetensors`. When validation is enabled, also prepare the
matching converted validation conversations and extracted SFT VAL features.
SFT mode never falls back to pretraining feature, packed-row, or validation
artifacts.

The steered mixtures (`standard_steered`, `standard_steered_balanced`) are
**not supported** in SFT mode. `resolve_selfcond_sft_sources` rejects such a
base checkpoint with `self-conditioning SFT mode does not support steered
mixtures (dataset=...)` before any artifact preflight or store load. Their SFT
generator opens a new feature-prediction group at each steered conversation's
assistant turn (see
[encoder-decoder-training.md](encoder-decoder-training.md#steered-sft-dataset));
the rollout would have to reproduce that split slot-for-slot to stay aligned
with the objective the head was trained under, which is out of scope for v1.

The five SFT source overrides resolve as follows:

| Key | `null` resolution | Explicit value |
|---|---|---|
| `selfcond.sft_dataset` | Top-level SFT checkpoint `dataset`. | Must equal the recorded dataset; cross-dataset post-training is unsupported. |
| `selfcond.sft_feature_dir` | Checkpoint `user_config.feature_dir`. | Replaces the train feature directory, subject to the phase/segment guard below. |
| `selfcond.sft_parquet_dir` | Checkpoint `user_config.sft_parquet_dir` when non-null, otherwise `<base_dir>/sft_data/<dataset>`. | Replaces the train parquet directory. |
| `selfcond.val_sft_feature_dir` | When validation is enabled, derive the matching VAL phase from the canonical train feature path. | Replaces the validation feature directory. |
| `selfcond.val_sft_parquet_dir` | When validation is enabled, `<base_dir>/sft_data/<dataset>_val`. | Replaces the validation parquet directory. |

The generator policy also comes from the SFT checkpoint. `sft_loss_mask` is
read from top-level metadata (a missing legacy value warns and resolves to
`full_sequence`). `sft_segment_mode` is read the same way (a missing legacy
value warns and resolves to `fixed_tokens`); `min_loss_tokens`,
`feature_source_bos_char_len`, and
`alignment_probe_samples` come from `user_config`, with legacy defaults 8, 5,
and 8. Feature selection and negative selection use the recovered checkpoint
identity; negative selection remains mandatory. The tokenizer checksum comes
from train statistics when recorded, otherwise it is computed from the loaded
tokenizer.

A turn-delimited SFT base additionally gets a MEASURED generated-assistant
cadence. `measure_turn_self_cond_segment_size_rustbpe`
(`steering/conditional/encoder_decoder/turn_cadence.py`) samples conversations
from the TRAIN store and prepares each exactly as the conditioned-SFT
generator does: it renders the conversation the same way (converter message
slice + `render_conversation_with_mask`), cuts the render and its loss mask to
the packing row with `truncate_conv_and_mask` (`dec_config.sequence_len + 1`,
the generator's `row_capacity`), skips a row with fewer than
`min_loss_tokens` supervised targets under the recorded `sft_loss_mask` exactly
as the generator skips it, applies the generator's OWN coverage check on the
truncated render (`compute_full_coverage_gap` against
`CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`, 20), clips a truncated row's
segments to the retained text the way `_truncate_seg_features` does (an
untruncated row whose features overrun the render by more than the window is
rejected in routing and named), and routes them onto the RustBPE positions with
`build_full_sequence_segment_routing`. It keeps the FULL windows — segments
whose source-token count, the difference of consecutive
`SegmentFeatures.position` values, equals the Gemma `segment_size`, so a
document's trailing segment (and a segment clipped by truncation, which is
the last one retained) is excluded because it has no successor to measure
against, and a window clipped by routing's own alignment because the features
overrun the render within the window is likewise the last measurable one and
excluded — whose every routed INPUT position lies inside a logical assistant
turn (the generator conditions only `len(ids) - 1` positions of a
conversation and leaves its final position unconditioned, so a window that
owns the closing marker is one position narrower than the full routing says)
(`legacy_turn_spans`, not the render loss mask, which marks
`<assistant_start>` and tool output with 0). K is `round(mean(...))` of those
windows' RustBPE widths, taken exactly over a `Fraction` with ties to even.
At most 4096 conversations are measured, chosen at exact even spacing over
the store's sorted document ids, so the pass is bounded and identical on every
rank and resume; the reported `documents` count is the number of sampled
conversations that contributed at least one window. The sample's rounded mean
is not guaranteed to equal the full corpus's rounded mean. A sample with no
full assistant window raises rather than falling back, including when valid
conversations are simply too short.

Because every sampled conversation is head-truncated before the coverage
check, a conversation longer than the extraction window — whose features
legitimately stop short of its FULL render — is measured over the prefix the
generator trains on, exactly as the generator accepts it. A document whose
TRUNCATED render still runs more than 20 characters past its last segment
raises `SelfCondTrainingError` naming that document's store id, the gap, both
artifact directories, and the same re-extraction remedy; the generator would
abort on the same example. Any other per-document routing `ValueError` is
re-raised the same way, with the document id and the underlying message.

There is
no config knob: setting `selfcond.turn_self_cond_segment_size_rustbpe` is an
error. Post-training records K at the checkpoint metadata TOP LEVEL beside
`sft_segment_mode` as deployment identity; inference must read that exact
value, and it is never guessed from either tokenizer. Startup fully parses any
online-SAE lineage before SFT task/data activity, so malformed lineage cannot
survive until the first checkpoint save.

Every self-conditioning checkpoint also re-publishes its SFT base's top-level
identity blocks verbatim — `pretrain_conditioning`, `feature_mask_canonical`,
both `feature_clusters_*` blocks, `feature_vocab`, `feature_prediction_width`,
`missing_percentile_policy`, `dataset` and `sft_loss_mask` — plus its own
top-level `step` taken from `loop_state`. Downstream consumers read those from
the checkpoint they are handed rather than from its base
(`steering/eval/axbench_upstream/run_methods.py` locates
`model_{step:06d}.pt` from `step`; `methods.resolve_raw_feature_delta` and
`resolve_raw_feature_selection` fall back to `pretrain_conditioning`;
`resolve_feature_clusters_from_checkpoint_meta` requires both cluster blocks),
so a checkpoint missing them is unreadable on every AxBench surface.

Before constructing the SFT generators, startup independently derives the
source BOS character width from the first train feature/parquet shard pair and,
when validation is active, the first validation pair. Each result must exactly
match the recovered `feature_source_bos_char_len`. Missing evidence and
disagreement fail the existing rank-synchronized startup stage; the trainer does
not infer an override, rewrite metadata, or continue with a warning.

Train and validation feature-path guards parse the symlink-resolved path before
checking the selected dataset's SFT TRAIN/VAL phase and the checkpoint segment
size and segmentation mode. A canonical-looking symlink into a pretraining or
wrong-SFT-phase tree therefore raises, as does a segment-size or
segmentation-mode mismatch. A non-canonical recovered or
derived path is invalid identity and raises. Explicit non-canonical overrides
remain the only exception for both train and validation feature directories:
each is accepted with a loud warning because phase isolation cannot be
verified. An explicit non-canonical train override additionally requires both
validation directory overrides when validation is enabled. With
`eval_every=0`, those validation overrides are unnecessary.
`train_prefetch_batches` must be `null` in SFT mode because prefetch would move
the conversation cursor ahead of completed optimizer steps.

SFT prediction-target-only segments keep their FP-supervision semantics, but
have no decoder-token span and therefore cannot affect the LM objective through
cross-attention. They are excluded from rollout actions and from the
predicted-slot/count diagnostic denominator; the frozen `continuation_mask`
and FP-loss target predicate are unchanged.

Each SFT evaluation cycle builds two fresh single-pass loaders: one for
`val/bpb` with real conditioning and one for `val/bpb_selfcond` with the final
rollout conditioning. Both clone fresh conditioning bundles from the same
pinned per-eval data seed, and each clone preserves the shared positive/negative
RNG object. The loaders therefore serve byte-identical examples, and repeated
evaluation at unchanged model state is deterministic on the same runtime. This
deliberately differs from the SFT trainer's shared, advancing validation RNG.
Both metrics use the same `accumulate_bpb_nats_bytes` and `reduce_bpb`
accounting contract. Under distributed validation, both evaluators continue
until every rank exhausts its covered slice (`sync_last_step`,
`ReduceOp.MIN`). A rank that finishes early emits fully masked padded batches,
which keep collective call counts aligned while contributing neither nats nor
bytes. This preserves every slower rank's real tail batches without rescoring
documents or changing the two metrics' comparability.

SFT-mode selfcond checkpoints record `selfcond.data_source`, the turn-only
top-level measured `turn_self_cond_segment_size_rustbpe`, an exact
`selfcond_data` block (dataset, four resolved directories, `sft_loss_mask`,
`sft_segment_mode`, `min_loss_tokens`, `feature_source_bos_char_len`, tokenizer
checksum, `percentile_checksum`, and the resolved `feature_selection` and
`negative_selection` dictionaries), and generator-published artifact coverage.
Turn outputs also republish the agreed top-level `sft_segment_mode` and complete
online-SAE lineage so deployment has all three consistent witnesses. Resume
requires exact mode and cadence matches. SFT-mode self-conditioning checkpoints written
before `sft_segment_mode` joined this identity block cannot be resumed and must
start fresh from the SFT base; they remain loadable for inference, where the
absent `selfcond_data.sft_segment_mode` witness resolves as `fixed_tokens` like
every other pre-turn lineage. Resume also requires
`loop_state.sft_cursor_semantics=buffered_v2`; `sft_consumed` and the ordered
per-rank `sft_dataloader_state` snapshot are published only after an optimizer
step commits. Resume restores that cursor plus its in-flight best-fit buffer,
so conversations are neither silently replayed nor skipped. This is the same
fail-closed state contract described under [encoder-decoder SFT
resume](encoder-decoder-training.md#resuming-an-interrupted-sft-run). The
conditioning RNG still restarts at its fixed seed rather than restoring
buffered draws, so resume is not RNG- or gradient-bit-exact.

Compact masked/clustered SFT bases (the l22 production family) are supported:
the mask and cluster consolidation are reconstructed from the checkpoint's
canonical metadata alone (`feature_mask_canonical`,
`feature_clusters_canonical`/`_descriptor`), verified against the state
dict's `FeatureVocab` id-for-id, and drive representative-only conditioning
and negative sampling exactly as conditioned SFT does. A
`feature_delta.mode=splice_counterfactual` pretraining lineage warns and
proceeds (the same hardcoded policy as conditioned SFT), and a base recorded
with `missing_percentile_policy=lowest` continues under `lowest` with the
NON-COMPARABLE pilot warning — the policy is persisted and a cross-policy
resume is rejected.

Unsupported SFT-mode bases include pretraining-shaped checkpoints,
cross-dataset requests, gradient-ranked selection, legacy thin SFT
checkpoints without recoverable negative-selection conditioning metadata, and
non-canonical feature paths recovered from checkpoint metadata. The explicit
non-canonical escape hatch described above is the only exception.

## Startup trace and timing

The self-conditioning entry point emits an always-on INFO trace whose message
starts with `selfcond_startup ` and whose remainder is one canonical JSON
object. This is diagnostic output, not a new config surface or a checkpoint
identity field. The trace is emitted by rank 0 only; records made before
distributed initialization are buffered until the rank is known, and
non-master ranks discard their local records.

All durations are host-wall seconds measured with Python
`time.perf_counter()`. The implementation intentionally does not call
`torch.cuda.synchronize()`, so a phase can include asynchronous CUDA work and
must not be read as synchronized device time. The existing `step ... ms` log
line remains unchanged. The first completed optimizer step is additionally
reported as `first_step_s`; later completed steps contribute to
`steady_state_mean_s` and `steady_state_step_count`.

The JSON record kinds are:

| `kind` | Meaning |
|---|---|
| `phase` | A timed region with `phase`, `status` (`ok` or `error`), `elapsed_s`, `cumulative_s`, and phase-specific metadata. Nested feature-loader phases are allowed; a failed phase is logged before its original exception is re-raised. |
| `event` | An instantaneous boundary or artifact fact with `event`, `cumulative_s`, and structured metadata. |
| `summary` | One closing record with `status`, `scope`, `total_s`, `named_phase_s`, `unattributed_s`, `first_step_s`, `steady_state_mean_s`, and `steady_state_step_count`. |

The successful summary's normal scope is
`main_entry_to_first_completed_step`. On a zero-step run it instead says
`main_entry_to_training_end_without_completed_step`; a failure uses
`main_entry_to_failure`. `named_phase_s` is the union of the phase intervals,
so nested loader phases are counted once. Time in observer-free gaps is
retained as `unattributed_s` rather than being silently assigned to a
neighboring phase.

The SFT path names the expensive boundaries explicitly: checkpoint/model
restore, statistics and compact-identity reconstruction (mask, clusters, and
feature vocabulary), conditioning and percentile reconstruction, spelling
asset preflight, mixture construction, the shared SFT feature-store load,
artifact identity/document/feature-row coverage, content-alignment probe,
initial train-batch materialization, optimizer/loader/W&B setup, and the first
optimizer step. The shared generator reports its lazy train phases as
`sft_feature_store_load`, `sft_artifact_identity`, `sft_document_coverage`,
`sft_feature_row_coverage`, `sft_alignment_probe`, and `sft_initial_batch`.
The validation generator remains lazy and is reported as a skipped
`sft_validation_artifact_index` record; validation artifacts are indexed when
evaluation actually starts. This issue intentionally does not add a separate
eager train/validation percentile-coverage preflight; that broader SFT-trainer
behavior is a follow-up candidate if operators need it after reviewing these
traces.

The standard and capabilities SFT mixtures use `words_alpha.txt`. Startup
resolves the operator cache at `$NANOCHAT_BASE_DIR/words_alpha.txt` when it is
present, otherwise uses the pinned repository copy at
`tasks/assets/words_alpha.txt`; it never performs a live GitHub fetch during
self-conditioning startup. The safety mixture does not contain spelling tasks
and records the word-list phase as skipped. If both local sources are absent,
startup fails with the paths and restoration instructions. The pinned bytes and
upstream commit are recorded in
[`tasks/assets/words_alpha.txt.provenance.md`](../../tasks/assets/words_alpha.txt.provenance.md).

## Algorithm

For each optimizer step, `scripts/train/conditional_encdec_selfcond.py` does:

1. Start from the source batch's real conditioning. Document starts, padding,
   and single-segment documents retain their real feature IDs, buckets, and
   masks byte-for-byte, with only zero padding added if the action width is
   wider.
2. Roll out continuation slots sequentially under `torch.no_grad()`. Slot
   `j` is chosen from FP scores pooled over source slot `j-1`, after all slots
   through `j-1` have already been decided.
3. Draw positive count `k`, then negative count `l`, for every row at each
   live rollout step. Null count configs inherit the checkpoint's recorded
   positive and negative count distributions.
4. Rank pooled float32 FP logits directly, then select the stable top-`k` IDs
   as positives and the disjoint stable bottom-`l` IDs as negatives. This
   avoids false endpoint ties from sigmoid saturation. Positives are bucket 0;
   negatives are bucket 1.
5. Recompute differentiable FP scores in pass 1. Pass 2 computes the existing
   language-model cross-entropy with value-zero STE edges injected into the
   encoder. One backward traverses both graphs.
6. Materialize exact zero FP-head gradients when a rank has no predicted slots,
   then clip, apply the learning-rate schedule, and step the optimizer. This is
   required for `DistMuonAdamW` collectives; ranks must never skip a step
   independently.

SFT mode uses this rollout followed by the same final teacher-forced LM forward
verbatim; only batch construction and source identity differ.

### Dense and selected STEs

`selfcond.ste_form=dense` projects every full-vocabulary score through the
detached feature embedding tables. `selected` gathers only scores for the
chosen feature IDs. They are different estimators, not interchangeable
performance modes.

For a positive column, the score edge has sign `+1`. For a negative column it
has sign `-1`, corresponding to a negative membership weight tied as `1-s`.
Thus an otherwise identical negative slot contributes exactly the negated
score gradient. The embedding tables are detached inside the STE mix, so they
receive only their ordinary lookup gradients.

The discrete rollout and differentiable edge deliberately use different
representations of the same finite logit: rollout actions rank the raw pooled
float32 logit, while the STE retains `s = sigmoid(logit)` as its membership
weight. In exact arithmetic those rankings are monotone-equivalent; ranking
before sigmoid prevents finite logits from collapsing into false `0`/`1` ties
without changing the estimator's values or derivatives.

### What the one-hop estimator captures

The estimator captures the direct pass-2 LM gradients and, for each
continuation slot, the complete chain
`loss → slot j encoder input → score s[j-1] → FP head/upstream model`. In
`full` scope this includes everything upstream of source segment `j-1`,
including decoder self-attention.

It deliberately drops terms where that chain re-enters an earlier slot's STE,
such as
`… → s[j-1] → hidden[j-1] → slot j-1 STE → s[j-2] → …`. It is therefore a
two-pass, one-hop surrogate, not exact full-chain backpropagation through the
entire discrete rollout.

## Running it

This is a compose-API Hydra script. Use `config_name=...`, not
`--config-name`, and retain the `--` separator under `torchrun`.

Single process:

```bash
uv run python -m scripts.train.conditional_encdec_selfcond \
    config_name=conditional_encdec_selfcond \
    model_tag=d12_fp_uncompacted \
    selfcond.train_scope=head_only \
    selfcond.ste_form=dense
```

Multi-process:

```bash
uv run torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_selfcond -- \
    config_name=conditional_encdec_selfcond \
    model_tag=d12_fp_uncompacted \
    selfcond.train_scope=full \
    selfcond.ste_form=selected
```

If the tag/segment/selection locator is inconvenient, point directly at the
pretraining leaf. An explicit base path makes `model_tag` optional:

```bash
uv run python -m scripts.train.conditional_encdec_selfcond \
    base_checkpoint_dir=/abs/path/to/conditional_encdec_checkpoints/d12/s128/dist_geometric/mean_12.0/std_12.0
```

SFT mode must point directly at the SFT leaf:

```bash
uv run python -m scripts.train.conditional_encdec_selfcond \
    config_name=conditional_encdec_selfcond \
    base_checkpoint_dir=/abs/.../safety_sft_conditioned_checkpoints \
    selfcond.data_source=sft \
    selfcond.train_scope=head_only selfcond.ste_form=dense
```

If that SFT leaf records `turn_delimited_fixed_tokens`, the launch takes no
extra flag: startup measures the target-token cadence from the TRAIN artifacts,
logs it with the window count and the mean/min/max it came from, and records it
in the output checkpoint, after which inference reads that value verbatim.
Resuming re-measures and refuses a value different from the resumed
checkpoint's.

The real-artifact smoke gate uses the same command for each dataset's matching
SFT leaf, with one train/eval/save step:

```bash
uv run python -m scripts.train.conditional_encdec_selfcond \
    config_name=conditional_encdec_selfcond \
    base_checkpoint_dir=/abs/.../safety_sft_conditioned_checkpoints \
    selfcond.data_source=sft \
    selfcond.train_scope=head_only selfcond.ste_form=dense \
    num_iterations=1 eval_every=1 save_every=1
```

`model_step=null` means the latest base checkpoint. This is distinct from
`resume_from_step=-1`, which means start a fresh post-training run. Resume a
post-training checkpoint with a positive step:

```bash
uv run python -m scripts.train.conditional_encdec_selfcond \
    base_checkpoint_dir=/abs/path/to/base-leaf \
    resume_from_step=250
```

CORE, sample, and steering evals are not wired in v1. Positive
`core_metric_every`, `sample_every`, or `steering_eval_every` values raise at
startup instead of being ignored.

## Configuration

The shipped config inherits `base_train`, but architecture and data identity
come from checkpoint metadata rather than fresh-model config blocks.

| Key | Default | Meaning |
|---|---|---|
| `model_tag` | required at runtime unless `base_checkpoint_dir` is set | Base encoder-decoder checkpoint tag. Hydra composition yields `null`; the script enforces requiredness. |
| `model_step` | `null` | Base checkpoint step; `null` selects the latest. |
| `base_checkpoint_dir` | `null` | Explicit base checkpoint leaf, bypassing tag/path assembly. |
| `segment_size` | `128` | Locator used before metadata is loaded; must match the checkpoint. |
| `feature_selection` | geometric mean/std 12, min 1, max 48, top-k percentile | Locator used before metadata is loaded. Recorded metadata is authoritative after load. |
| `num_iterations` | `500` | Post-training optimizer steps. |
| `resume_from_step` | `-1` | `-1` starts fresh; a positive integer loads that selfcond step. |
| `lr_multiplier` | `1.0` | Multiplies optimizer learning rates before the flat-then-linear-decay schedule. |
| `grad_clip_max_norm` | `1.0` | Global gradient-norm clip; `null` disables it. |
| `compile` | `false` | Must remain false in v1 because rollout, pass 1, and pass 2 are distinct graphs. |
| `eval_every` | `100` | Real-conditioned and self-conditioned validation cadence. |
| `save_every` | `250` | Checkpoint cadence. |
| `profile_timing` | `false` | Per-rank synchronized region timing; mutually exclusive with tracing. |
| `profile_trace` | `null` | Zero-based absolute first optimizer step for a per-rank Chrome trace. |
| `profile_trace_steps` | `1` | Positive consecutive-step trace length; the full window must execute. |
| `profile_output_dir` | `out/profiles/encdec_selfcond` | Timing/trace output directory, unused and uncreated when diagnostics are off. |

| `selfcond.*` key | Default | Meaning |
|---|---|---|
| `train_scope` | `head_only` | `head_only` freezes everything except `feature_pred_head`; `full` uses the standard full-model optimizer. |
| `ste_form` | `dense` | `dense` full-vocabulary or `selected` gathered-score estimator. |
| `pool_mode` | `null` | Inherit the head's `loss_mode`; explicitly set a supported mode for a `best_token` checkpoint. |
| `positive_counts` | `null` | Inherit `{distribution, mean, std, min_k, max_k}` from checkpoint `feature_selection`. A threshold-trained checkpoint carries no count distribution — resolving raises, and you must set this explicitly (e.g. geometric 12/12, min 1, max 48). No silent fallback. |
| `negative_counts` | `null` | Inherit count fields from checkpoint `negative_selection`. |
| `data_source` | `pretraining` | `pretraining` uses packed rows; `sft` uses SFT conversations and source-aware artifacts. |
| `sft_dataset` | `null` | Recover the SFT checkpoint's dataset; an explicit value must match it. |
| `sft_feature_dir` | `null` | Recover the SFT TRAIN feature directory from checkpoint metadata. |
| `sft_parquet_dir` | `null` | Recover the recorded train parquet override or use `<base_dir>/sft_data/<dataset>`. |
| `val_sft_feature_dir` | `null` | Derive the matching SFT VAL feature directory when validation is enabled. |
| `val_sft_parquet_dir` | `null` | Use `<base_dir>/sft_data/<dataset>_val` when validation is enabled. |

Top-level script knobs beyond the `selfcond` block include
`train_prefetch_batches` (`null` = pretraining parity: prefetch depth equals
the gradient-accumulation step count; an explicit integer ≥ 0 overrides it,
`0` = synchronous loader). SFT mode requires `null`. Validation loaders are
deliberately never prefetch-wrapped — each eval consumes exactly its budget and
drops the loader, so a prefetch thread would leak.

Pretraining data-identity keys (`feature_dir`, selections, seed, segment size,
SAE, bucket count, pooling, dataset phase) default to the checkpoint's
`user_config`. Explicit `+key=value` overrides win and are logged; structural
disagreements in segment size, bucket count, or SAE feature width raise. In SFT
mode, the source keys and scalar recovery rules under
[Data sources](#data-sources) own the conversation/feature directories and
generator settings.

## Profiling and benchmark arms

Use separate runs for attribution, kernel tracing, and throughput. Timing adds
CUDA synchronization around main-thread regions; tracing records shapes and
Python stacks. Either changes the measured step, so neither result is a clean
throughput number and the script rejects enabling both together.

CUDA-synchronized timing CSVs for every rank:

```bash
uv run torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_selfcond -- \
    base_checkpoint_dir=/abs/path/to/base-leaf \
    profile_timing=true profile_trace=null \
    profile_output_dir=out/profiles/selfcond_timing
```

One optimizer-step Chrome trace per rank:

```bash
uv run torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_selfcond -- \
    base_checkpoint_dir=/abs/path/to/base-leaf \
    profile_timing=false profile_trace=10 profile_trace_steps=1 \
    profile_output_dir=out/profiles/selfcond_trace
```

`profile_trace` is the script's zero-based absolute `step`, while ordinary logs
display `step + 1`. A resumed run rejects a trace before its resumed step, and
all runs reject a window whose end exceeds `num_iterations`; this prevents a
silently empty or partial successful capture. A failure during an active trace
attempts a rank-local `_partial.json` export without introducing a new DDP
collective or replacing the training exception.

Timing output is `selfcond_timing_rank<R>.csv`. Trace output is
`selfcond_trace_step<S>_rank<R>.json`; preserving every rank is necessary to
diagnose stragglers and collective imbalance. Static marker names cover batch
loading, every rollout slot/gate/forward/pool/selection/write, both
differentiable passes, backward, head-zero-gradient materialization, clipping,
optimizer work, validation, and checkpoint I/O. Rank, absolute step,
micro-step, and slot are CSV columns rather than interpolated marker names.
Synchronous SFT generator work also reuses its existing `sft/*` regions for
buffer refill, rendering, text tensors, segment construction, feature lookup,
and conditioned-batch assembly; these are activated only while this profiling
session can consume them.

SFT mode materializes its initial batch during startup. Consequently, step
zero's `selfcond/data_loading` row can be near zero for that already-pending
batch; subsequent rows measure live generator work. Pretraining lookahead
likewise attributes only the consumer-side `pop`; background producer work is
not inherited by the main thread's context variables. Compare the per-rank CSV
with the always-on startup trace and a separate diagnostics-disabled throughput
run before attributing a bottleneck. Open Chrome traces in
[Perfetto UI](https://ui.perfetto.dev).

## Evaluation and logging

- In pretraining mode, `val/bpb` uses real validation conditioning through
  `evaluate_encdec_bpb_conditioned`.
- In SFT mode, both validation metrics use fresh, identically seeded
  single-pass conversation streams as described under [Data sources](#data-sources).
- `val/bpb_selfcond` performs a fresh no-grad rollout and measures CE using a
  fixed validation seed. Repeated evaluation at unchanged model state is
  deterministic.
- Training logs `train/loss`, predicted-slot fraction, mean positive/negative
  counts, mean selected positive score, LR multiplier, and step timing.
- Checkpoint reporting tracks both `min_val_bpb` and
  `min_val_bpb_selfcond`.

## Checkpoints: explicit path only

Post-training writes beneath the explicit base checkpoint leaf (a pretraining
leaf for the default source or an SFT leaf for SFT mode):

```text
<base-checkpoint-leaf>/
├── model_005000.pt
├── meta_005000.json
└── selfcond_pooled_fp_logits_v1_<train_scope>_<ste_form>_checkpoints/
    ├── model_000250.pt
    ├── meta_000250.json
    └── optim_000250_rank0.pt
```

Consumers must receive the selfcond leaf as an **explicit directory path**.
Tag-level `_find_checkpoint_leaf` discovery stops at the parent as soon as it
sees `model_*.pt`; the vendored `ModelSource` enum has no
selfcond chain. The script prints the exact output directory at run end.

Metadata keeps SFT-compatible architecture keys (`model_config`,
`encoder_config`, `xattn_config`, `num_buckets`) and rewrites
`feature_prediction_config.loss_mode` to the pool mode actually trained. The
original head mode remains under
`selfcond.original_head_loss_mode`. Resolved counts, scope, STE form, seed,
`action_ranking_policy`, base path, loader cursor, and loop state are also
persisted. SFT mode replaces the packed-loader cursor with the exact
`selfcond_data` and committed SFT cursor identity described above. Model
parameters are saved once and every DDP rank saves its optimizer shard. A
turn-mode leaf additionally carries the RustBPE cadence under `selfcond`, and
resume resolves it from the written metadata before accepting the run.
Profiling settings and diagnostic output paths do not enter this metadata or
the resume-matching contract.

The action-ranking identity is resume-critical. Checkpoints from the historical
sigmoid-ranked rollout do not contain `action_ranking_policy` and cannot be
resumed under `pooled_fp_logits_v1`; start a fresh post-training run instead.

## Cost expectation

At a typical roughly 16-segment row, one optimizer step costs the live rollout
for each continuation step plus two grad-enabled forwards and one backward:
roughly 5–8× a pretraining step. Retaining the pass-1 and pass-2 graphs makes
activation memory roughly 2× pretraining; rollout state is comparatively
small. Reduce `device_batch_size` and use gradient accumulation first when
memory-bound.

Tiny CPU smoke measurement on 2026-07-17 (SDPA, B=1, T=8, two segments,
V=8, dense STE, 20 post-warmup samples): head-only median 6.649 ms (p90
6.692 ms), full-scope median 11.522 ms (p90 11.551 ms). These numbers are a
correctness-smoke reference, not a production performance gate.

## V1 follow-ups

Deferred work includes `num_buckets > 2`, compacted FP heads, exact
full-chain BPTT, compile support, the live loader, splice-counterfactual
training, auxiliary FP-loss mixing, KV-chunked rollout, and generation-side
sampled-count selection matching the post-training action policy.
