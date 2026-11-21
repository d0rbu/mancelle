# Gradient-Attribution Feature Selection (offline-only)

Rank each segment's SAE features by **gradient attribution** instead of
activation percentile: backprop the LM loss of a configured token window to
the SAE's residual hook (layer L), push the gradient through the SAE decoder
(`grad @ W_dec.T`), gate by feature activity, and reduce per feature with
`max over the window's rows of ( gate · ReLU(-g) )` — a per-feature gradient
**sensitivity**: how strongly *increasing* the feature would *decrease* the
loss window (the gate zeroes a feature that did not fire). The offline producer
then multiplies this sensitivity by the feature's **activation percentile** to
form the stored score `percentile(a) · max_r ( gate · ReLU(-g) )`. The
selection precompute keeps the top-k of each segment's **eligible** (score > 0)
features.

The percentile factor is what keeps the ranking topically coherent: a
barely-active feature with a large per-unit gradient (high sensitivity, low
percentile) is demoted below a strongly-firing feature, so the score reflects
both *that the feature was actually present* and *that its presence helped the
future*. Both factors are non-negative, so `score > 0 ⟺ the feature fired AND
its increase lowers the loss`.

This changes only *which* features condition each segment. Conditioning
strength is unchanged: `sel_pcts`/buckets fed to the encoder stay
activation-percentile-based — a controlled experiment on the selection
criterion.

## Sign convention (worked example)

`g = dL/df` is the loss gradient w.r.t. a feature's activation. `g < 0`
means increasing the feature **lowers** the loss window (the feature helps);
`ReLU(-g)` keeps exactly that direction, and the activity gate (`a > 0`) keeps
only features that actually fired. A feature whose decoder row points *against*
the residual-stream loss gradient at some read row scores `|grad|` (the
sensitivity); a feature pointing *along* it, or one that did not fire, scores
exactly 0 and becomes ineligible. The offline producer then weights this
sensitivity by the feature's activation percentile, so a barely-active feature
with a large per-unit gradient no longer outranks a strongly-active one. The
sign tests in `tests/steering/test_gradient_attribution_sign_convention.py`
construct these cases and corroborate with finite differences; the
percentile-weighting (a producer-level multiply) is exercised in the
producer-integration tests.

## Attribution modes (`gradient.mode`, three states)

| Mode | Non-final segments | Doc-final attribution unit |
|---|---|---|
| `future_half_split` | gradient of the post-segment suffix loss (all predictions of tokens after the segment), read at the segment's rows | unit `[u, e)` split at `mid = u + (e-u)//2`: backprop predictions of `[mid, e)`, read rows `[u, mid)` |
| `future_self_full` | identical | backprop the unit's own predictions, read all of its rows |
| `self` | gradient of the segment's OWN predictions, isolated per segment | same (no split) |

**Short-final merge** (`gradient.min_final_segment_tokens`, default 8, **must
be >= 2**): a doc-final segment shorter than the threshold is merged with the
second-to-last segment into ONE attribution unit — both segments' stored
features are scored at the unit's read rows (each conditioning slot still
selects from its own stored support; slots stay separate at training).
Earlier segments' suffixes are unchanged. A single-segment doc has nothing
to merge with and is its own unit; with `doc_len >= 2` it is already
scoreable. The `>= 2` floor guarantees a 1-token final (which is unscoreable
— nothing follows it) always merges; `2` merges only those 1-token finals
(there is no "disable merging" — disabling it would silently zero them).

**Every segment is scored, or the producer fails loud.** There are no
zero-signal placeholders. Structurally unscoreable inputs are made impossible
upstream — the producer rejects any doc with `< 2` tokens, requires
`segment_size >= 2` (a `[0, 1)` first segment has no own predictions in
`self` mode), and requires `min_final_segment_tokens >= 2` — and the schedule
RAISES rather than emit an empty read or a zero row if one slips through.

**The spec quantity**: a segment's attribution gradient is the ordered sum
of per-unit gradients in the schedule order of
`steering.gradient_attribution.build_backward_schedule` (mathematically the
suffix/self loss gradient; numerically pinned so the reference oracle in
`tests/reference_gradient_attribution.py` matches **bit-exactly** on CPU
fp32).

## Pipeline position

```
parquet ─ extraction ─→ base shards ─→ stats ─→ pct/<h>/ ─→ tok/<h>/ ─→ packed rows
              │                                                            │
   (1) GPU, per grad config:                                  packed rows carry their provenance
   {feature_dir}/grad/<checksum16>/shard_*.safetensors.zst    NATIVELY: {packed_row_dir}/provenance/
   fp16 scores 1:1 with base CSR feature_ids                  (base_shard_idx, doc_idx, seg_ordinal)
   (self-contained: offsets + doc_seg_offsets copies)         written by the packer in the same flush
              │                                                            │
              └──────(2) precompute_selected_features joins + ranks ───────┘
                          → sel/<new hash>/ → training (unchanged)
```

Base shards, statistics, percentile overlays, token boundaries, and packed
rows are never recomputed (I3). Segment provenance is recorded by
`precompute_packed_rows` itself (the packing loop is the only code that
knows the mapping) — it is config-independent and serves every gradient
config. A packed dir created before provenance integration heals in place
via `precompute_packed_rows resume=true` (existing packed shards are kept
byte-identical; only the missing provenance is regenerated by the
deterministic replay).

> **Known gap (GA-001; fails loud):** base shards record token counts and
> segment geometry, but no witness for the exact token IDs used during
> extraction. Equal-geometry tokenizer or source drift is therefore
> unverifiable. Any new, missing, or rebuilt gradient shard raises
> `NotImplementedError` after geometry validation and before scoring; adding a
> base-shard identity witness is deferred because it requires a schema change.
> Until then, leave `feature_selection.gradient` unset and use percentile-based
> selection. Existing overlay discovery checks remain unchanged, but cannot
> retroactively prove this identity for an older overlay.

## Quick start

```bash
# 0. The standard pipeline first: extract → compute_statistics →
#    precompute_percentiles → precompute_token_boundaries → precompute_packed_rows
#    (packed rows include segment provenance natively).

# 1. Gradient overlay (GPU; DDP-striped; resumable; expect 1-2 orders of
#    magnitude slower than activation extraction — full forward + per-unit
#    backwards). REQUIRES the percentile overlay from step 0: the stored score
#    is `percentile(a) * sensitivity`, so the producer reads exactly one
#    `pct/<hash>/` overlay and bakes the percentile weight in (fails loud if
#    absent or ambiguous).
torchrun --standalone --nproc_per_node=4 \
    -m scripts.preprocess.precompute_gradient_attribution \
    attribution_mode=future_half_split resume=true

# 2. Selection shards for the TRAIN and VAL packed dirs:
uv run python -m scripts.preprocess.precompute_selected_features \
    packed_row_dir=<train_packed_dir> \
    feature_delta=null feature_selection.delta_weights=null \
    '+feature_selection.gradient.mode=future_half_split'
uv run python -m scripts.preprocess.precompute_selected_features \
    packed_row_dir=<val_packed_dir> \
    feature_delta=null feature_selection.delta_weights=null \
    '+feature_selection.gradient.mode=future_half_split'

# 3. Training (config_name= syntax for torchrun scripts):
torchrun --standalone --nproc_per_node=4 -m scripts.train.conditional_encdec_train -- \
    feature_delta=null feature_selection.delta_weights=null \
    '+feature_selection.gradient.mode=future_half_split'
```

## Configuration

```yaml
feature_selection:
  distribution: geometric
  mean: 5.0
  min_k: 1
  max_k: 25
  mode: top_k_percentile        # REQUIRED for gradient ranking
  delta_weights: null           # REQUIRED: gradient ranking is delta-incompatible
  gradient:                     # presence of this block = gradient ranking
    mode: future_half_split     # future_half_split | future_self_full | self
    model_dtype: bfloat16       # must match the overlay producer run
    min_final_segment_tokens: 8 # short-final merge threshold (must be >= 2)
feature_delta: null             # REQUIRED
```

The overlay producer (`configs/extraction/precompute_gradient_attribution.yaml`)
additionally takes the SAE/model identity (must match the base extraction) and
`batch_size` (documents per padded forward/backward batch). BOS handling is
**inherited** from the base shards' metadata, not configured: when the shards
record `ignore_bos_features=true` (frozen always-on at extraction since #533),
BOS rows are excluded from every read window's max-reduction. A stale
`ignore_bos_features` key passed via config/CLI raises before model
construction or artifact creation; remove the obsolete key.

Before constructing the tokenizer, SAE, or model, the producer also requires
the explicit feature directory to parse as the exact canonical residual path,
validates uniform required identity metadata across every base shard, checks
canonical document blocks and SAE L0/hook identity, and validates percentile
manifest staleness, exact shard coverage, and every percentile payload. These
are rejection-only checks: valid v1/v2 artifacts and healthy-path scoring and
serialization are unchanged.

v1 restrictions (all fail fast): `mode=top_k_percentile` only; no
`delta_weights` / `feature_delta`; no `feature_clustering`; residual SAEs
only (`hook_resid_post`); Gemma-2/Gemma-3 extraction models; sae_lens
`standard`/`jumprelu` architectures with `normalize_activations="none"` and
no finetuning scaling factor (enforced by a build-time decoder-linearity
gate that proves `autograd-through-decode == grad @ W_dec.T` exactly).

> **Gemma-3 VLM model loading (4b/12b/27b).** The gradient attribution
> producer calls `build_gradient_model`, which routes through
> `steering.gemma3_loading.load_gemma3_text_decoder` for gemma-3 4b/12b/27b
> checkpoints — a plain `Gemma3ForCausalLM.from_pretrained` would silently
> random-init the text decoder (issue #549). See
> [`gemma-scope-2.md` § VLM checkpoints](gemma-scope-2.md#vlm-checkpoints-4b12b27b--text-decoder-loading).

## Selection semantics

- **Strict eligibility**: only features with score > 0 are selectable. If a
  segment has fewer eligible features than the drawn k, the selection is
  capped at what's available; zero eligible ⇒ empty selection
  (`sel_counts=0`). Both cases are aggregate-logged per shard. There is no
  percentile fallback and no random fill.
- Exact ties among positive scores keep the existing seeded-RNG tie-break.
- Negative sampling is unchanged: it excludes the segment's full ACTIVE set
  (not just the eligible subset), so an active-but-ineligible feature can
  never be drawn as a negative.
- `sel_pcts` / buckets are recovered from activation percentiles exactly as
  before.

## Hashing, manifests, invariants

- `compute_selection_hash` canonicalizes the gradient block **centrally**:
  absent/null ⇒ the payload is byte-identical to the legacy form (I1 —
  pinned by a frozen golden in `tests/steering/test_selection_hash_gradient.py`);
  present ⇒ `{mode, model_dtype, min_final_segment_tokens, score_reduction,
  split_policy_version}` enters the hash, so `sel/<hash>/` dirs re-address
  across any config or algorithm revision.
- The grad overlay dir name is the SHA-256 of its full canonical descriptor
  (mode, merge threshold, model/SAE identity, dtype, code-versioned
  reduction + split-policy constants); its schema-v2 manifest binds the base
  shard content hash (staleness guard). The selection precompute verifies
  descriptor equality before consuming.
- Provenance is written by the packer in the same flush as each packed shard
  and recorded in the packed manifest (`base_shard_stems` +
  `provenance_shards`), so the two cannot drift; the selection precompute
  requires the manifest record and a 1:1 provenance/packed shard count, and
  a dir packed by pre-integration code fails loud with the heal command.
- The sel/ manifest records a `gradient_attribution` block (I4) which
  training discovery asserts against the configured spec.
- Checkpoints gain a conditional `grad_{mode}_mf{N}[_fp32]` leaf
  (byte-identical legacy paths when off — I2).

## Offline-only (fail-fast surface)

Gradient ranking REQUIRES `use_precomputed_batches=true` with packed rows +
sel shards for the train AND val splits (verified at startup). Every live
selection path raises `GradientRankingUnsupportedError`: the live
dataloader, the fused `reconstruct_and_select` packed path without sel
shards, SFT training (live-only), self-conditioned generation feedback,
eval context caches, the completion TUI,
and `convert_decoder_to_encdec`. The guard is enforced centrally in
`sample_features`/`sample_features_batched` (any caller that merely threads
the parsed config is auto-protected) plus early site-level checks.
Conditioned-BPB eval keeps working — it resolves the same `sel/<hash>/`
shards the training run used.

## Numerical contract

- CPU fp32: the collector's gradient SENSITIVITY matches the reference oracle
  **bit-exactly** (schedule-as-data equality; per-unit gradient fold in spec
  order; the scoring matmul/gate). See `tests/reference_gradient_attribution.py`.
- The GA-011 ragged-batch oracle records raw accumulated gradients, reset/unit
  traces, and final rows for grouped execution versus independent per-document
  VJPs in all three modes. Its CUDA fp32/BF16 envelope lives in the slow-marked
  `tests/steering/test_gradient_attribution_cuda_tiny.py` suite.
- The stored score is `percentile(a) · sensitivity`: the producer reads the
  feature's activation-percentile (uint8 / 255) from the resolved `pct/<hash>/`
  overlay and multiplies in fp32 before the fp16 cast. The grad overlay
  manifest records that `percentile_checksum`, so a resume against a different
  percentile overlay fails loud, and the producer-integration tests pin the
  baked product (`tests/steering/test_gradient_attribution_producer_integration.py`).
- The chunked lm_head loss tail reproduces `model(input_ids).logits` either
  bit-exactly or within a 1-2 ulp GEMM-blocking envelope, platform-probed
  (`test_chunked_tail_vs_full_forward_logits`).
- bf16 production runs carry the usual extraction-grade magnitude noise;
  fp32-vs-bf16 score-set agreement is gated on real models
  (`tests/steering/test_gradient_attribution_real_gate.py`, `very_slow`).
- Scores are stored fp16: non-finite values AND values at/above the
  fp16-safe ceiling (`FP16_SAFE_MAX`, 60000) abort the producer — a finite
  fp32 score >= 65520 would otherwise silently persist as +inf and dominate
  every selection. The zero-score fraction and max are logged per shard.
- CUDA backward uses atomics: overlays are NOT guaranteed bit-reproducible
  across CUDA runs (CPU fp32 is). Resume/skip semantics are unaffected.

## Verification

```bash
# Fast suite (CPU, includes the bit-exact oracle):
uv run pytest tests/reference_gradient_attribution.py tests/steering/ -k gradient -q
# Tiny-model CUDA envelopes (GPU):
uv run pytest tests/steering/test_gradient_attribution_cuda_tiny.py -m slow
# Real gemma-2-2b gates + full real-pipeline integration (GPU, downloads):
uv run pytest tests/steering/test_gradient_attribution_real_gate.py \
    tests/steering/test_gradient_attribution_e2e_real.py -m very_slow
```
