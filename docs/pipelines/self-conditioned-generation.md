# Self-Conditioned Generation

The encoder-decoder model has an auxiliary feature-prediction head that learns
to predict the next segment's SAE feature percentiles from the current
segment's hidden state. **Self-conditioned generation** plumbs this head
through to inference: the model drives its own encoder via the head's output,
producing coherent text without further SAE feature extraction after the
prompt's real-feature prefill.

> ### ⚠ Prerequisite: an FP head; compact checkpoints are supported for generation
>
> Self-conditioned **generation** supports uncompacted and compact (mask- or cluster-derived) feature-prediction heads. A compact head returns `V_effective` logits whose row `j` scores original SAE id `feature_vocab.original_ids[j]`. `SelfConditionedFeatureProvider` pairs those scores with `feature_vocab.new_to_old_tensor()` before selection; masking, per-feature thresholds, bucketing, and encoder inputs then remain in original-ID space, and `FeatureEncoder._to_compact` performs the encoder's original-to-compact translation.
>
> No cluster reduction is repeated during generation: the effective vocabulary already contains exactly the surviving representative rows, so an excluded or consolidated-away feature has no head row and cannot be selected. A configured `feature_mask`, including a whitelist, is still reapplied value-wise before selection and may further restrict that effective candidate set. Construction checks the model/config original SAE width, the head/vocabulary width, the persistent vocabulary buffers, and the only two structurally valid active-bucket counts.
>
> **Post-training has a narrower split:** its pretraining-data path still
> requires an uncompacted checkpoint, while its source-aware SFT path supports
> compact mask/cluster vocabularies and performs STE in compact row space. That
> training contract remains separate from consuming compact head logits during
> inference. Higher-level checkpoint loaders must also declare a
> compaction-aware eval stance before building the model. See
> [selfcond-post-training.md](selfcond-post-training.md) and
> [encoder-decoder-training.md § Encoder Feature-Dictionary Compaction](encoder-decoder-training.md#encoder-feature-dictionary-compaction).

## Pipeline at a glance

Both modes prefill the exact cached real-feature prompt, pool only FP rows
routed to the final real source slot, and build S0. Their generated-token
causal contracts then intentionally differ:

```text
fixed_tokens (legacy, unchanged)
  commit <assistant_start> under S0; it is outside the K-row window
  forward rows 1..K under S0
  on row K: FP-only forward under S0 → rewind one KV position
            → build S1 → replay row K under S1
            → returned S1 logits sample token K+1

turn_delimited_fixed_tokens
  discard the marker-only real SAE slot
  forward <assistant_start> once under S0 as assistant-window row 1
  forward every later assistant input exactly once under its outgoing slot
  after row K: retain its outgoing logits, build/publish S1
               → the next forwarded input is row 1 under S1
```

One shared transition owns both branches; the fixed branch's model calls, KV
writes, and boundary replay remain unchanged. Turn mode never replays a
boundary input. The inference engine checks every sampled token before display
or model forwarding: a stop token is neither yielded nor forwarded. An ordinary
token that exactly exhausts `max_new_tokens` is yielded but not forwarded,
because its logits would control no later sample.

Encoder attention is **segment-isolated**: packed owner metadata prevents one
segment from attending to another. Self-conditioning intentionally encodes each
new segment on its own, then appends its real packed values and assigns the next
segment owner without changing any previously encoded value. This is exactly
the established incremental behavior. Joint and standalone encoder calls can
use different FP8 tensorwise scales because their global packed tensors differ,
so they are required to be numerically close, not bitwise identical. Exact
append parity against the removed dense-concat/repack behavior is pinned in
`tests/steering/test_packed_encoder.py`; segment-isolation closeness is pinned in
`tests/steering/test_self_conditioned_provider.py`.

## Choosing `self_cond_segment_size`

This is the cap on nanochat (RustBPE decoder) input rows per predicted segment.
Its authority depends on checkpoint mode:

- General `fixed_tokens` callers retain the historical caller-owned value. The SAE pipeline's
  usual 128-token Gemma source windows motivated a 256-RustBPE heuristic for
  English text, but this is an operational choice rather than checkpoint
  identity. Existing AxBench fixed-mode runs deliberately use the decoder
  sequence length so their 128-token completion never refreshes.
  **Self-conditioned ChatCORE requires measured fixed cadence instead**:
  `record_fixed_cadence` records `fixed_self_cond_segment_size_rustbpe` and
  `fixed_self_cond_cadence_provenance` in an evaluation leaf. The optional CLI
  fixed assertion only checks equality. See the
  [ChatCORE runbooks](evaluation.md#self-conditioned-chatcore).
- `turn_delimited_fixed_tokens` requires the strict positive integer that
  post-training MEASURED from its TRAIN artifacts and recorded at the metadata
  top level as `turn_self_cond_segment_size_rustbpe`. Loaders must use that
  exact value; a CLI/config value cannot replace it (post-training has no such
  knob), and it must never be inferred from the Gemma source `segment_size`.
  Missing, malformed, or conflicting metadata fails before generation.

The Gemma extraction segment size/context window and this RustBPE generated
cadence are deliberately separate token currencies. Independently of either
value, extraction validates the *observed* nanochat/source token ratio of each
non-final segment: `scripts/preprocess/precompute_token_boundaries.py` warns at
4x, errors at 5x, and raises at 6x, with a hard 0.5x lower bound
(`_SEG_LO_RAISE`/`_SEG_HI_RAISE`); the empirical mean lives in the 2x band. Both
bounds are overridable per run via `max_nanochat_tokens_per_segment` /
`min_nanochat_tokens_per_segment` (see
`configs/extraction/precompute_token_boundaries.yaml`).

## Loss-mode contract for the prediction head

The default for `FeaturePredictionConfig.loss_mode` is `MAX_POOL` so
freshly-trained models work out of the box for self-cond eval. The other
modes:

| Mode | Self-cond eval | Notes |
|---|---|---|
| `MAX_POOL` (default) | ✓ | Per-feature max of fp_logits across the segment positions, then BCE/MSE against the next segment's target. Symmetric across the segment, runs identically at training and inference. Verified in `tests/steering/test_max_pool_scrutiny.py` (27 tests covering invariants, gradient flow, edge cases, numerics) plus an end-to-end training-loss-decreases smoke test. |
| `LAST_TOKEN` | ✓ | Reads position −1 of each K-token span. |
| `FULL_SEGMENT` | ✓ | Averages over all K positions. |
| `BEST_TOKEN` | ✗ | Forbidden. Best-token requires the next segment's target features (argmin loss); unavailable at inference. The provider refuses to load such a checkpoint. |

If you have an older checkpoint trained with `BEST_TOKEN`, retrain or
fine-tune with `MAX_POOL`/`LAST_TOKEN`/`FULL_SEGMENT` before running
self-cond evals.

Checkpoints produced by
[on-policy self-conditioning post-training](selfcond-post-training.md) rewrite
`feature_prediction_config.loss_mode` to the pool mode actually trained while
preserving the original mode under
`meta["selfcond"]["original_head_loss_mode"]`. They are the intended targets
for this generation path. Pass the nested selfcond checkpoint leaf by explicit
directory path; tag-level discovery stops at its parent pretraining checkpoint.

## Building a self-conditioned engine

The two generation surfaces in this repo:

### Path A — scalar `InferenceEngine`

`InferenceEngine`'s scalar path drives both the legacy fixed-token transition
and the audited turn-delimited one-shot transition. The current turn-aware
callers are the AxBench-upstream project method and encoder-decoder baselines.
Existing `chat_eval`, harmfulness, and AxBench-Base surfaces
remain fixed-only and reject turn-delimited checkpoint/config metadata before
their generation work; they do not gain turn support merely by using this
engine.

Interactive turn Chat uses the same wrapper/scalar transition in transactional
full-history epochs, described in [serving](serving.md#full-conversation-epochs-turn-delimited-checkpoints).

```python
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.conditional.feature_pipeline import (
    ThresholdSelectionConfig,
    active_bucket_count,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType

active_buckets = active_bucket_count(
    model.encoder.config.num_buckets,
    reserves_negative_bucket=checkpoint_used_negative_sampling,
)
selection = ThresholdSelection(
    threshold_type=ThresholdType.segment_topk,
    threshold=12,
)
self_cond_cfg = SelfCondConfig(
    self_cond_segment_size=256,
    pool_mode=FeaturePredLossMode.MAX_POOL,
    selection=ThresholdSelectionConfig(
        selection=selection,
        precomputed_thresholds=None,
    ),
    selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
    bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
    num_buckets=active_buckets,  # full count, or full-1 when negatives reserve the top row
    n_sae_features=16384,
)
# Alternatively, sampling-mode self-cond accepts a SamplingBundle:
#   selection=SamplingBundle(selection=SamplingSelection(...), percentile_table=..., rng=...)

engine = build_engine_self_conditioned(
    model=model,
    tokenizer=tokenizer,
    device=device,
    self_cond_config=self_cond_cfg,
    context_provider=context_provider,  # real per-segment prompt features
)

# Same generation interface as today
text = engine.generate(prompt, max_new_tokens=256, render_as_chat=True, ...)
```

### Path B — `multiseg_generator` (inline / completion / shuffled steering)

This surface is intentionally `fixed_tokens`-only. Its public self-conditioned
stream — the one entry point both functions below go through — rejects a
turn-delimited `SelfCondConfig` before token RNG, KV-cache, or model work.
Turn-aware generation must use Path A's audited scalar runtime; fixed-only
callers must not silently reinterpret a turn checkpoint with the legacy
boundary-replay geometry.

```python
from steering.eval.multiseg_generator import (
    generate_completion_self_conditioned,
    generate_completion_stream_self_conditioned,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
)

provider = SelfConditionedFeatureProvider(
    model=model, cfg=self_cond_cfg, device=device,
)

tokens = generate_completion_self_conditioned(
    model=model,
    prefix_tokens=prefix,
    provider=provider,
    encoder_inputs=encoder_inputs,  # real per-segment prompt features + routing
    max_new_tokens=256,
    device=device,
    temperature=0.7,
    top_k=50,
    seed=42,
    bos_id=tokenizer.get_bos_token_id(),
)
```

In fixed mode, both paths build their initial encoder state from real
per-segment prompt features, seed one predicted slot from the prompt's tail
logits, and then use the same `SelfConditionedFeatureProvider` and
`SelfCondConfig` contract during decode. The turn branch uses the shared scalar
wrapper (InferenceEngine or turn Chat) with marker-inclusive, no-replay semantics;
the AxBench-upstream project/encoder-decoder callers and turn Chat construct
its exact prompt feature source through one shared producer. Choose a caller
that explicitly advertises the checkpoint geometry; fixed-only callers fail closed.

## Dense target-score interpolation

`SelfCondConfig.score_policy = SelfCondScoreInterpolation(...)` opts into the corrected AxBench path while
`None` preserves the legacy provider and wrapper behavior exactly. The field is
one frozen value object, so the target, weight, and negative policy cannot be
partially configured:

```python
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
)

self_cond_cfg = SelfCondConfig(
    # normal fields omitted
    score_policy=SelfCondScoreInterpolation(
        target_scores=dense_head_row_scores,
        weight=strength / 6,
        negative_selection=checkpoint_negative_selection,
    ),
)
```

The target is snapshotted once as detached contiguous CPU float32. It must be a
materialized rank-1 real tensor, have exactly the FP-head width, contain only
finite values in `[-1, 1]`, and use the head's exact row order.
For a compact checkpoint, target row `j` therefore names original SAE id
`feature_vocab.original_ids[j]`.

AxBench-Upstream can obtain this target from either fitted dense DiffMean or
the optional `NLSteering` retrieval source. Retrieval aggregates unit-normalized
feature-description embeddings into the checkpoint's exact mask/cluster-aware
head rows, computes cosine similarity to the exact concept description, centers
covered rows by PyTorch's lower median, and divides by the largest covered
absolute centered value. Uncovered rows stay exactly zero. Both sources then
enter the same `SelfCondScoreInterpolation` engine contract described below.

This mode requires a MAX_POOL-trained head, MAX_POOL runtime pooling,
`SamplingBundle` with `top_k_percentile`, full-percentile selection, uniform
percentile buckets, one active bucket, and an encoder with exactly two total
buckets. Both positive and negative count distributions need bounded exclusive
`max_k` values whose maximum possible total fits the eligible vocabulary.
Delta-weighted and gradient-ranked live selection remain rejected. It cannot be
combined with `fp_logit_bias`.

For supplied raw rows `L`, let `z = MAX_POOL(L).float()` and
`q = 2 * sigmoid(z) - 1`. The provider computes:

```text
scores = z                                      when weight == 0
scores = target_scores                          when weight == 1
scores = torch.lerp(q, target_scores, weight)   otherwise
```

The exact endpoint branches prevent interpolation rounding. At weight zero only
ordering matters, so ranking `z` avoids false ties when float32 sigmoid would
saturate distinct finite logits. Interior weights retain the historical
float32 `q`/DiffMean interpolation exactly; replacing `q` there with unbounded
logits would change the strength scale. The provider applies any full-width
feature mask to candidate IDs and score rows, draws the positive count before
the negative count, then uses one stable shared top/bottom selector. Highest
rows become active bucket 0; lowest rows among the remainder become inactive
bucket 1. Genuinely equal logits tie by canonical head-row order. Selected IDs
are translated back to original SAE IDs and validated as unique, disjoint,
allowed, in range, and representative-only before encoding.

## Dense target-score sign masking

`SelfCondConfig.score_policy = SelfCondScoreMasking(...)` is the second corrected AxBench policy. It
shares the interpolation mode's entire structural contract (MAX_POOL-trained
head and runtime pooling, `SamplingBundle` with `top_k_percentile`,
full-percentile selection, uniform percentile buckets, one active bucket, an
encoder with exactly two buckets, bounded `max_k` on both count policies, and
a target snapshotted exactly like `SelfCondScoreInterpolation.target_scores`),
and — because both policies share the single union-typed `score_policy` field via the `SelfCondDenseScorePolicy` base — it is structurally exclusive with interpolation and mutually exclusive with
`fp_logit_bias`:

```python
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreMasking,
)

self_cond_cfg = SelfCondConfig(
    # normal fields omitted
    score_policy=SelfCondScoreMasking(
        target_scores=dense_head_row_scores,
        negative_selection=checkpoint_negative_selection,
        positive_pool_limit=PositiveInt.parse(positive_pool_limit),
        negative_pool_limit=PositiveInt.parse(negative_pool_limit),
    ),
)
```

Stable DiffMean rank defines eligibility; pooled model logits perform final
selection. The dense target `d` first defines sign-consistent base pools, after
the feature mask is applied:

```text
positive base = eligible head rows where d > 0
negative base = eligible head rows where d < 0
positive pool = first positive_pool_limit rows ranked by d descending
negative pool = first negative_pool_limit rows ranked by d ascending
```

Both DiffMean sorts are stable, so equal scores resolve by canonical ascending
head-row order. Exact-zero DiffMeans (including `-0.0`) are eligible for
neither side. Pool construction happens once when the provider is built; no
DiffMean sort or mask recomputation enters the token-generation loop.

Counts are drawn through the same `sample_action_counts` call as
interpolation — positive first, negative second from the config's seeded
generator — so the two policies draw identical `(k, l)` for the same seed. A
sampled count larger than its ranked pool is clamped to the pool size with one
`logger.warning`; the empty-segment error still fires if both effective counts
are zero. Selection then runs the stable shared top/bottom primitive once per
pool: highest logit in the positive pool, lowest logit in the negative pool,
ties by head-row order. The feature mask restricts both pools before pool sizes are
measured and before rank truncation. A required empty sign pool, or two empty
pools, is rejected at provider construction rather than producing a provider
that fails on every step. Compact-checkpoint row translation, mask asserts, and
representatives-only cluster checks are identical to the interpolation path.

## Optional feature-prediction logit bias

`SelfCondConfig.fp_logit_bias` is a fixed additive control applied in this
exact order:

```
pool over decoder-token positions
  → detach and cast pooled logits to float32
  → add fp_logit_bias in float32
  → sigmoid
  → selection and bucket assignment
```

The bias is therefore measured in **logit space**, not percentile space. It has
one value per feature-prediction **head row**. Its width is
`n_sae_features` for an uncompacted head and `feature_vocab.size` for a compact
head. On a compact checkpoint, row `j` controls original SAE id
`feature_vocab.original_ids[j]`; passing an original-SAE-width vector fails
loudly instead of broadcasting or silently steering the wrong row.

There are two distinct safe translation recipes:

```python
import torch as t

# Sparse controls authored as original SAE ids: translate ids strictly.
original_ids = t.tensor([17, 4021], dtype=t.int64)
compact_rows = feature_vocab.to_compact(original_ids)  # raises if either has no row
bias = t.zeros(feature_vocab.size, dtype=t.float32)
bias[compact_rows] = t.tensor([2.0, -1.5])

# A dense vector already authored over every original SAE id: select rows.
full_bias = t.zeros(feature_vocab.n_sae_features, dtype=t.float32)
# A gather alone would silently drop excluded nonzero controls. Validate first.
controlled_ids = full_bias.nonzero().flatten()
feature_vocab.to_compact(controlled_ids)  # raises if any controlled id has no row
compact_bias = feature_vocab.select_rows(full_bias)
```

Do not use `select_rows()` without the support validation above: it is a pure
gather and cannot distinguish an intentionally zero excluded row from a
nonzero control it silently dropped. Do not index a bias with
`old_to_new_tensor()` either: an excluded or consolidated-away feature has the
`-1` sentinel there, and `bias[-1]` silently targets the final row. The strict
`to_compact()` validation deliberately raises whenever either kind of control
names a missing row. A configured
`feature_mask` is still applied after scoring, so even an arbitrarily positive
bias cannot resurrect a masked feature.

The beartyped snapshot boundary accepts a phantom-refined domain: a materialized,
strided, real floating 1-D tensor only. It copies and detaches the value into
contiguous float32, then the provider validates the head-row width and takes a
second private copy on its inference device. This
severs aliases, prevents a supplied `Parameter` from rebuilding an autograd
graph, and makes a running provider stable if the caller later mutates either
its source tensor or the config's public tensor. Values must remain finite
after float32 conversion; a finite float64 value that overflows float32 is
rejected. `None` is the default and returns the pooled float32 tensor itself:
no addition, conversion, allocation, version change, or RNG use occurs. A zero
bias has equal numeric outputs but still executes the opt-in addition, so it is
not the literal no-op path.

## Target, selection, and bucket producer contracts

On a compact checkpoint, `fp_logits` has `feature_vocab.size` columns, but
`SelfCondConfig.n_sae_features`, masks, percentile tables, selected ids, and
encoder inputs all remain in the original SAE domain. The provider translates
only the candidate-id vector with `feature_vocab.new_to_old_tensor()`; callers
must not reinterpret compact head row numbers as feature ids.

`SelfCondConfig.num_buckets` is the bucket count for predicted **active**
features. If training reserved the encoder's final bucket for negative/inactive
features, derive the count from the training metadata:

```python
from steering.conditional.feature_pipeline import active_bucket_count

active_buckets = active_bucket_count(
    model.encoder.config.num_buckets,
    reserves_negative_bucket=checkpoint_used_negative_sampling,
)
```

The model object records its embedding-table size but not whether negative
sampling was enabled. The provider can therefore reject counts other than the
full encoder count and `full - 1`, but the caller must choose the correct one
from checkpoint selection metadata. Passing the full count for a checkpoint
that reserved its last row can encode a predicted-active feature as inactive.

Two independent, required fields identify conditioning behavior. Neither has an
inferred default, and neither can be recovered from the head's shape or weights.
Self-conditioned feedback itself has one numeric currency: sigmoid outputs and
all cutpoints are continuous normalized full-distribution ranks in `[0, 1]`.
The adapter does not quantize cutpoints to imitate packed target storage.

`SelfCondConfig.selection_producer` identifies the score used to choose positive
features:

- `FULL_PERCENTILE_RANK` (`"full_percentile_rank_v1"`) consumes the same
  full-distribution rank predicted by the head. Packed pretraining,
  percentile-based sampling, and clustered SFT use this producer.
- `RAW_MAGNITUDE` (`"raw_magnitude_v1"`) identifies raw SAE-magnitude
  selection. Self-conditioning supports only `feature_percentile`: its
  per-feature statistics key has a monotonic percentile-space equivalent.
  `segment_topk`, `segment_percentile`, and
  `feature_percentile_segment_topk` all compare raw magnitudes across features;
  independent per-feature CDF transforms do not preserve that ordering, so
  those combinations fail at `SelfCondConfig` construction.

`SelfCondConfig.bucket_producer` independently identifies discretization:

- `UNIFORM_PERCENTILE` (`"uniform_percentile_v1"`) applies the historical exact
  `floor(score * active_buckets)` rule.
- `RAW_STATS_BOUNDARIES` (`"raw_stats_boundaries_v1"`) maps the persisted raw
  statistics-key grid directly into normalized rank and searches it with
  `right=True`.

The split is necessary for ordinary unclustered percentile sampling: selection
uses `FULL_PERCENTILE_RANK`, but the selected raw magnitudes are bucketed with
`RAW_STATS_BOUNDARIES`.

The distinction is load-bearing. Packed pretraining writes uniform buckets
directly from decoded packed percentiles; it does **not** call
`build_boundaries_from_stats`. Conversely, a raw unshifted three-bucket path
uses statistics keys `p33` and `p66`, and a raw p90-filtered ten-bucket path
uses `p91` through `p99`. Exact raw `p99` maps to rank `99.5`; the other
integer keys keep their rank. Treating every checkpoint as either producer
silently changes valid conditioning rows.

For the raw-stat bucket producer, a score exactly at a translated cutpoint
enters the higher bucket just as a raw activation at the boundary does. Scores
and cutpoints are float32, so sufficiently close ranks can still alias; the
canonical right-inclusive cutoff maps such an alias to the higher bucket.

Checkpoint metadata does not yet stamp both identities for all legacy paths.
Provide evidence from the actual selection and bucket producers; raw strings are
rejected. Do not infer from a path, run name, compactness, or tensor shape.

### Residual quantization skew (cutpoints stay continuous)

Heads whose targets were supervised from packed rows or `pct` overlays learn
uint8-roundtripped values, `round(rank / 100 * 255) / 255`; self-conditioned
cutpoints deliberately remain continuous. For such a pretrained-only head under
a RAW producer, an ideal head disagrees with raw training only within at most
half a uint8 code (≈0.196 percentile points), and it errs on the *exclude*
side. A target sitting exactly on a boundary is the visible case: raw rank 66
encodes to code 168, decodes to `168/255 ≈ 0.65882`, one code below the
continuous `0.66` cutpoint, so it takes the lower bucket.

Quantizing the cutpoints instead would preserve that single boundary value but
would misclassify a band up to ~4× wider on the *include* side (for `p66`,
`[167.5/255, 0.66)` rather than `[0.66, 168.5/255)`), and it would BREAK the
exact replay of packed selection — training compared decoded uint8 scores
against continuous cutoffs — and of SFT-trained heads, whose targets are
continuous. SFT and self-cond post-trained heads are exact under the continuous
contract. See
`tests/steering/test_self_conditioned_compact.py::test_uint8_roundtripped_boundary_target_takes_the_lower_bucket`.

## Representing per-feature threshold availability

Training materializes an original-SAE-width per-feature **magnitude** tensor for
`feature_percentile` and `feature_percentile_segment_topk`: finite rows have
statistics and `+inf` rows do not. Pass that exact tensor to
`ThresholdSelectionConfig.precomputed_thresholds`. The provider preserves its
availability mask, rejects NaN/`-inf`/non-floating/wrong-width inputs, and
replaces each finite raw value with the canonical cutoff for the declared
selection producer. The numeric raw magnitudes are never compared directly
with FP scores.

```python
from steering.conditional.thresholding import build_thresholds_from_stats

training_precomputed_thresholds = build_thresholds_from_stats(
    per_feature_stats,
    model.encoder.config.n_sae_features,
    selection.threshold_type,
    selection.threshold,
)
```

For `segment_topk` and `segment_percentile`, no precomputed value is needed.
Feature-universe restrictions belong in `feature_mask`; whitelist masks are the
replacement for any fixed allowed-ID set.

For `FULL_PERCENTILE_RANK`, the provider compares float32 scores with the parsed
configured percentile directly. This reproduces packed selection exactly: for
example, decoded p95 code `242 / 255` remains below a configured `0.95` cutoff.
For `RAW_MAGNITUDE`, the provider reproduces the artifact-identifying
`int(percentile * 100)` key rule used by `build_thresholds_from_stats` (so
binary-float `.29`, `.57`, and `.58` intentionally select the preceding key),
then divides that rank by 100. Key `p99` uses rank `99.5`. The raw bucket grid
remains shifted from the configured percentile,
not from the truncated key; changing either legacy rule would re-key existing
artifacts.

Do not convert the magnitude tensor with `compute_percentiles` or replace it
with a hand-filled decimal tensor: that loses the missing-statistics exclusions
and cannot identify which producer owned the checkpoint.

### Clustered checkpoints: fold availability over cluster members

For a cluster-derived compact checkpoint, availability may **not** be read off
the representative's own row. Clustered training looks up per-feature
statistics on the ORIGINAL sibling ids and only then max-folds each cluster
into its representative, so a representative whose own threshold row is `+inf`
can still be a legitimately trained, selectable feature — for example a cluster
`{1, 3}` whose representative is `1` but whose statistics live on `3`. Reading
the representative row alone suppresses it silently and forever.

Pass the checkpoint's own consolidation through
`SelfCondConfig.cluster_consolidation`, rebuilt from checkpoint metadata with
`resolve_feature_clusters_from_checkpoint_meta` followed by
`FeatureClusterConsolidation.build` with the checkpoint's fully resolved
`FeatureMask`. The provider then OR-folds availability over each cluster's
mask-surviving members before deriving cutoffs, and validates the consolidation
against the effective vocabulary at construction: it must name the same
original SAE width, the model must be compacted, and every candidate original
id must be its own surviving (not consolidated-away) representative. The field
is consulted only for that fold, so it is a validated no-op for selection types
without per-feature thresholds.

A thresholds tensor that is entirely `+inf` after the fold is rejected at
construction rather than yielding a provider that can never select anything;
that state almost always means the thresholds were built from a statistics
artifact covering none of this model's features (wrong `feature_dir`).

## Prefill: real-feature conditioning

Every prefill starts with the wrapper's full-context conditioning lookup. Real
per-segment SAE features come from the `context_provider`, and prompt tokens are
routed through the same `compute_context_token_segment_assignments` machinery
used by training.

All provider policies use one prefill sequence. The wrapper looks up the full
rendered token tuple, builds conditioning and its authoritative full `seg_map`
once, forwards tokens `[:-1]` under real routing, and selects every FP row
mapped to the final real source slot. That source may be shorter or longer than
generated cadence `K`; prompt pooling follows extracted geometry, not a
decoder-token guess. It then builds S0 and forwards the terminal
`<assistant_start>` under S0, so token 1 already sees S0. No earlier context row
enters the pool.

For fixed mode, the marker remains a boundary outside the first predicted
window and is committed with the historical dummy-target/replay contract. For
turn mode, the supported AxBench surface's transient exact-token context entry
must carry training-identical source-token-aligned ownership and end in an
exact marker-only assistant segment; interior logical-turn starts need not be
window starts, because a Gemma token can straddle an unmarked boundary (tool
output into assistant text) and training keeps the straddled first token with
the preceding window. ChatCORE also admits authenticated schema-5 raw-turn cache rows through
`prepare_turn_context`; source-token straddles and unused source ordinals keep
their original routing. The
marker-only real SAE slot is removed before packing, the marker is left out of
real routing, and that same token is forwarded once as FP row 1 under S0. This
makes the last real user/tool-output chunk predict the first assistant window
exactly as conditioned SFT does. Batch size must be one and the prompt must
contain at least two tokens.

This means:

- **All prompts**: pass a `context_provider` built from real extracted features
  so the prompt receives its training-time per-segment conditioning. Fixed-mode
  consumers use an offline `SafetensorsContextFeatureProvider`; the supported
  turn-mode AxBench path builds a transient `SingleEntryContextProvider` from
  the checkpoint-lineage extraction result. See
  `scripts/eval/build_eval_context_cache.py` for fixed and schema-5 turn
  precompute, and the [ChatCORE runbooks](evaluation.md#self-conditioned-chatcore)
  for checkpoint-derived preparation.

At a fixed-mode decode boundary, the outgoing state's speculative FP row
belongs to the completed old window, while the replayed KV entry belongs to the
incoming state. The fixed branch restores the KV position on speculative,
provider, or commit failure; callers publish the new runtime only after the
commit succeeds. Turn mode performs one outgoing forward, retains those logits,
then publishes the incoming state for the next input. A provider failure
rewinds that one KV write, and wrapper runtime fields are published only from a
complete result.

## Scalar self-conditioned ChatCORE

See the [self-conditioned ChatCORE pipeline](self-conditioned-chatcore.md) for
completion-prefix caches, categorical scoring, calculator generation and result
identity. This path uses the model-owned precision with ambient autocast disabled.

## Limitations

- **Batch size**: B=1 only. The wrapper's existing kv-cache contract enforces
  this, and self-cond inherits.
- **Optimized cross-cell batching**: the boundary-only AxBench kernel remains
  fixed-mode-only. Turn-mode project and encoder-decoder baseline rows dispatch
  through the scalar audited runtime because one completion may cross multiple
  metadata-owned refresh boundaries.
- **Interactive multi-turn Chat**: Chat TUI uses a separate full-history
  extraction/prefill protocol with epoch-scoped rewind. ChatCORE's persistent
  prefix cache and scalar calculator driver do not provide that UI lifecycle.
- **`torch.compile`**: unsupported on the self-cond eval path because the
  encoder grows by one slot per K tokens (dynamic shape). Compile is only
  used during training, so this is not a regression.
- **BPB / loss-only evals** (`base_eval`, `dclm_eval`, `axbench_bpb`): not in
  scope. These score teacher-forced loss; self-conditioning would require
  generation, which is a different code path.
- **Decoder-only models**: cannot self-condition (no encoder).
  `build_engine_self_conditioned` raises if given a `GPT` instance.
- **`BEST_TOKEN` checkpoints**: incompatible. The provider refuses to load
  them — train or fine-tune with `MAX_POOL` / `LAST_TOKEN` / `FULL_SEGMENT`.
- **Compact FP checkpoints**: supported by the inference-time feedback
  provider and target-aware wrapper. Source-aware SFT post-training also
  supports an exact compact mask/cluster vocabulary and performs STE in compact
  row space; the packed-pretraining post-training source remains
  uncompacted-only.
- **Raw-magnitude cross-feature selection**: unsupported. A feature-prediction
  head emits per-feature percentiles, which cannot reproduce cross-feature raw
  ordering. Use full-percentile selection or raw per-feature quantile thresholds.
- **Delta-weighted or gradient-ranked sampling**: rejected. Delta weighting
  needs reference-segment provenance absent from `SelfCondConfig`; gradient
  ranking is offline-only and has no live attribution scores.

## Verification

Run the self-conditioned suite:

```bash
uv run pytest tests/steering/ -k "self_cond or self_conditioned or pool_fp_logits or return_fp_logits or percentile_to_bucket or masking" -v
```

This exercises the provider, wrapper, generation round trip, feature masking,
feature-prediction pooling, packed-state growth, and returned FP-logit contracts
on the test-owned CPU oracle. `tests/steering/test_packed_encoder.py` separately
checks CUDA output/loss/gradient parity against that padded reference.
