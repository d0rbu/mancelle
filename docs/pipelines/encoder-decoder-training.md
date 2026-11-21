# Encoder-Decoder Conditional Training Pipeline

Train an encoder-decoder model that conditions text generation on SAE features via cross-attention.

## Conceptual Overview

The goal is to make a language model's generation controllable via interpretable SAE features. Instead of modifying the text token sequence, we add a separate encoder that reads SAE feature activations and injects them into the decoder via cross-attention. This keeps the text stream untouched while giving the model a "side channel" of interpretable steering signals.

**Why encoder-decoder?** SAE features are unordered sets of (feature_id, magnitude) pairs per text segment — they're not sequential tokens. A bidirectional encoder processes these bag-of-features representations without imposing artificial ordering, and cross-attention lets each decoder position attend to the features relevant to its segment.

**Why discretize magnitudes?** Raw SAE activation magnitudes vary widely across features and are sensitive to distribution shifts. Discretizing into buckets (via per-feature percentile boundaries) makes the model robust to magnitude scale changes and prevents overfitting to exact activation values.

**Why segment-level?** Features are extracted per segment (e.g., 128 tokens). Each decoder position is mapped to its segment via `seg_map`, so the model can condition on local feature context rather than a single global embedding. This enables fine-grained steering — different parts of a document can be conditioned on different features.

## Pipeline Overview

```
Pre-extracted features + Statistics → Feature Selection (Threshold or Sampling)
    → Discretization → FeatureEncoder → CrossAttention → Decoder → Conditioned LM
```

**Prerequisites:** Feature extraction and statistics computation must be completed first (see `feature-extraction.md`).

**Scripts:** `scripts/train/conditional_encdec_train.py`, `scripts/train/conditional_encdec_sft.py`, `scripts/train/conditional_encdec_selfcond.py`, `scripts/preprocess/convert_decoder_to_encdec.py`
**Configs:** `configs/training/conditional_encdec_train.yaml` and variants; `configs/training/conditional_encdec_selfcond.yaml` for post-training

## Quick Start

```bash
# Train encoder-decoder from scratch (auto-resolves feature dir)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train -- \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base

# Or let feature_dir be auto-constructed from SAE params
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train
```

## Full Pipeline

### Phase 1: Extract Features (see `feature-extraction.md`)

```bash
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    --config-name extract_features_4xh200 \
    num_shards=370 \
    output_dir=out/data/features
```

### Phase 2: Compute Statistics

```bash
python -m steering.compute_statistics \
    feature_dir=out/data/features \
    device=cuda
```

### Phase 3: Train Encoder-Decoder

```bash
# Default config (inherits from base_train.yaml)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train -- \
    feature_dir=out/data/features

# 4xH200 config
torchrun --standalone --nproc_per_node=4 \
    -m scripts.train.conditional_encdec_train -- \
    config_name=conditional_encdec_4xh200 \
    feature_dir=out/data/features

# RTX 4090 (reduced model)
python -m scripts.train.conditional_encdec_train \
    config_name=conditional_encdec_4090 \
    feature_dir=out/data/features

# Custom encoder architecture
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train -- \
    encoder.n_embd=0.75 \
    encoder.depth=0.5 \
    cross_attention.frequency=2

# With label dropout (CFG-style conditioning)
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train -- \
    label_dropout=0.1

# Resume from checkpoint
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_train -- \
    resume_from_step=5000
```

Distributed encoder-decoder resume reads the model archive once on rank 0 and
broadcasts its schema and tensors to the other ranks. Optimizer shards and the
metadata JSON remain per-rank reads. This is an internal all-rank resume-path
optimization, not a configuration mode; single-rank runs retain the plain load.
The checkpoint rank-duration skew line is still emitted. The source model read
does not create a comparable duration skew because receivers block in the
phase-1 broadcast, so their elapsed times include rank 0's read and should be
similar to rank 0's duration.

## Configuration Reference

### Encoder Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder.n_embd` | 0.5 | Float = ratio of decoder dim (rounded to 128), int = absolute |
| `encoder.depth` | 0.5 | Float = ratio of decoder depth, int = absolute |
| `encoder.n_head` | null | Null = enc_dim/head_dim, int = absolute |
| `encoder.n_kv_head` | null | Null = same as n_head (MHA), int = absolute |
| `encoder.head_dim` | 64 | Per-head dimension |
| `initialize_encoder_embeddings_from_sae` | null | `null` = random feature embeddings; `random`/`svd` initialize `encoder.feature_embed.weight` from SAE decoder directions |
| `initialize_encoder_embeddings_cluster_vectors` | mean | With static clusters, `mean` (the default) replaces each representative with the fp32 mean of mask-surviving members at the init mode's fixed stage: `random` after full projection/rescaling and before its final effective-vocab gather; `svd` in oriented `W_dec` before its initial effective-vocab subset/covariance. Without clusters (or with `null` init) it is an inert bitwise no-op. `representative` explicitly selects each representative's own direction instead. |

### Cross-Attention

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cross_attention.frequency` | 4 | Insert every N-th decoder layer |
| `cross_attention.n_head` | null | Null = match decoder n_head, int = absolute |
| `cross_attention.n_kv_head` | null | Null = same as n_head (MHA), int = absolute |

### Feature Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dir` | null | Path to features (null = auto-resolve from SAE params) |
| `packed_row_dir` | null | Producer-only input for `precompute_selected_features`; training rejects a non-null value and auto-discovers packed rows |
| `selection_splice_batch_size` | 64 | Batches ordinary splice rows by default; null/1 explicitly preserves legacy selection identity and bytes (exact enabled size enters the hash/manifest) |
| `feature_root` | out/data/features | Root for auto-constructed paths |
| `pooling` | max_pool | Pooling strategy (must match extraction) |
| `segment_size` | 128 | Segment size (must match extraction) |
| `decoder_segment_size` | null | Training-only compatibility placeholder; non-null values are rejected. Packed-row construction always uses character offsets. |
| `feature_selection.distribution` | geometric | Default sampling arm; `geometric` or `normal` |
| `feature_selection.mean` | 12.0 | Expected positive features per segment |
| `feature_selection.std` | 12.0 | Sampling standard deviation; geometric requires `std == mean` |
| `feature_selection.min_k` / `max_k` | 1 / 49 | Positive feature-count bounds (`max_k` is exclusive) |
| `feature_selection.delta_weights` | percentile_delta | Rank delta-scored segments by positive percentile delta; segments without delta evidence retain activation-percentile ranking |
| `feature_selection.mode` | top_k_percentile | Sampling selection mode |
| `num_buckets` | 2 | Discretization bucket count |
| `label_dropout` | 0.0 | Feature dropout probability (CFG-style) |
| `compile` | true | Enable `torch.compile`; `compile_static=auto` keeps the outer model static when loader shapes are fixed, while packed CUDA work uses separate dynamic graphs |
| `feature_mask_freq_min` | 1e-5 | Auto-blacklist producer, lower bound (was `0.0`; raised in #702 to also kill features too rare to autointerp-score) |
| `feature_mask_freq_max` | 0.5 | Auto-blacklist producer, upper bound (kills features active in >50% of segments) |
| `feature_clusters` | null | Path-only static YAML/JSON cluster definition for selected-feature precomputation and precomputed pretraining. Filtering chooses representatives; the same post-mask consolidation removes away members from the encoder's effective feature vocabulary. |
| `feature_prediction` | **null** | Auxiliary feature-prediction head (`{weight, layer, loss_mode, loss_fn, block, head_layers}`). Off by default after the PR #697 ablations. Full models predict `n_sae_features` columns; masked/clustered models predict `V_effective` columns. Compact FP requires precomputed packed rows with a completed source-integrity certificate. |
| `remote_training` | null | Presence enables the issue-#781 verified remote selection/checkpoint mode for the exact clustered L22/Gemma-3 splice-counterfactual c1024 run. Every non-null block requires issue-#853 strict rank-striped `body_prefetch`; optional `train_packed` enables issue-#806 TRAIN packed-row streaming. It is not a generic remote dataset option. |

### SAE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sae.release` | gemmascope | SAE release family |
| `sae.model` | gemma-2-2b | Model name |
| `sae.activation_type` | residual | Activation site |
| `sae.layer` | 8 | Layer index |
| `sae.width` | 16k | SAE width |
| `sae.l0` | null | v1: null canonical or int average-L0; v2: tier string such as `medium` |

## Feature Selection

Two mutually exclusive modes are available: **sampling** (stochastic, the
root-config default) and **threshold** (deterministic). Hydra selects the root
arm from `configs/training/feature_selection_variant/`; the internal group
keeps shipped YAML presets mutually exclusive. At the CLI, replace the final
mapping explicitly rather than recursively merging keys:

```bash
python -m scripts.train.conditional_encdec_train \
    '~feature_selection' \
    +feature_selection.threshold_type=feature_percentile \
    +feature_selection.threshold=0.9
```

The quoted delete-plus-add sequence works from every shipped pretraining
preset, including presets whose `_self_` block customizes sampling fields.
Pipeline consumers parse both modes through
`steering.conditional.feature_pipeline.build_feature_pipeline()`.

### Threshold Mode

The low-level thresholding helper supports four `threshold_type` values:

| Strategy | Description | `threshold` meaning |
|----------|-------------|---------------------|
| `segment_topk` | Top-k features per segment by magnitude | k value |
| `segment_percentile` | Top fraction of features per segment | Fraction `(0, 1]` |
| `feature_percentile` | Per-feature percentile-based threshold | Fraction `[0, 1]` |
| `feature_percentile_segment_topk` | Percentile filter + segment topk cap | Combined (e.g., 2048.9 = top-2048 + p90) |

The cached-statistics training pipeline supports percentile-based thresholds
(`feature_percentile` and `feature_percentile_segment_topk`). The shipped
`conditional_encdec_2xh200` and `conditional_encdec_2xb200` presets select
`feature_percentile_segment_topk` with `threshold=2048.9`: keep features above
p90 magnitude, then cap at top-2048 per segment.

Feature-universe filtering is handled by `feature_mask`, not by
`threshold_type`. To replace the old top-k-frequency filtering behavior, write
the chosen feature IDs to a mask file and configure `feature_mask` with
`mode: whitelist`. To remove very rare or very common features by activation
frequency, use the `feature_mask_freq_min` / `feature_mask_freq_max` producer
described below; it generates an ordinary blacklist mask from statistics.

### Sampling Mode

Instead of a fixed threshold, sampling mode stochastically selects features each segment using a probability distribution over feature magnitudes. This produces variable feature counts per segment, which can improve generalization.

| Parameter | Description |
|-----------|-------------|
| `distribution` | `geometric` (default) or `normal` |
| `mean` | Expected number of features per segment |
| `std` | Standard deviation (geometric requires `std == mean`) |

```yaml
# Default: geometric sampling with ~12 features per segment
feature_selection:
  distribution: geometric
  mean: 12.0
  std: 12.0
  min_k: 1
  max_k: 49
  delta_weights: percentile_delta
  mode: top_k_percentile
```

Sampling uses a seeded RNG (per DDP rank) for reproducibility. When combined with negative sampling, the positive and negative bundles share the same RNG object.
The positive, negative, threshold, and delta configuration constructors store
phantom-refined scalar attributes and enforce them with `beartype`. Sampling
means must be finite non-boolean numbers, standard deviations finite and
positive, `min_k` strict non-negative integers, and `max_k` null or strict
positive integers. Thresholds are finite and non-negative, while
`feature_delta.neg_delta_ratio` lies in inclusive `[0, 1]`. For these scalar
fields, the remaining imperative checks are relationships: geometric
`mean > 0`, geometric `std == mean`, and `max_k > min_k`.
The same domains are enforced at the low-level scalar and batched sampler,
thresholding, percentile-table, and bucket-boundary call boundaries. Seeded
Torch RNG behavior is unchanged; runtime collection hints use unsubscripted
outer containers so validation does not consume Python's global RNG.
Self-conditioning reuses these selection objects directly and adds only its
local requirement that both count ranges have a finite `max_k`.
`percentile_delta` changes the ranking weights only where the selected
`feature_delta.mode` supplies delta values. The root config continues to use
`prev_segment`; splice-counterfactual runs must separately select
`feature_delta.mode=splice_counterfactual` and satisfy that mode's spliced-data
and provenance requirements.

### Gradient-Attribution Ranking (offline-only)

Adding a `gradient` block to `feature_selection` (with `mode: top_k_percentile`)
switches the per-segment RANKING KEY from activation percentile to precomputed
gradient-attribution scores ("which features were most helpful for the
configured loss window"). Conditioning strength is unchanged — `sel_pcts` and
buckets stay activation-percentile-based.

```yaml
feature_selection:
  distribution: geometric
  mean: 5.0
  mode: top_k_percentile
  delta_weights: null             # gradient ranking is delta-incompatible
  gradient:                       # presence of this block = gradient ranking
    mode: future_half_split       # future_half_split | future_self_full | self
    model_dtype: bfloat16
    min_final_segment_tokens: 8
feature_delta: null
```

Requirements (all fail fast with `GradientRankingUnsupportedError` /
`GradientAttributionError`): `use_precomputed_batches=true`, packed rows +
`sel/<hash>/` shards for BOTH the train and val splits and the gradient
overlay (`scripts.preprocess.precompute_gradient_attribution`); packed rows
carry their segment provenance natively. Every
LIVE selection path (live dataloader, fused packed path, SFT, self-conditioned
generation, eval context caches) refuses gradient-ranked configs. Only
features with score > 0 are eligible: a segment with fewer eligible features
than the drawn k is capped at what's available; zero eligible features yield
an empty selection (aggregate-logged). The gradient block enters
`compute_selection_hash` via a CONDITIONAL canonical key (absent ⇒
byte-identical legacy hash — the same I1 rule as `feature_mask`), and
checkpoints gain a conditional `grad_{mode}_mf{N}[_fp32]` leaf (I2). Full
pipeline, semantics, and numerical contracts:
[gradient-attribution-selection.md](gradient-attribution-selection.md).

## Discretization

Feature magnitudes are discretized into `num_buckets` bins:
1. Compute per-feature percentile boundaries from statistics
2. Map activations to bucket indices via `searchsorted`
3. Encoder receives bucket indices instead of raw magnitudes

## SAE-Based Feature Embedding Initialization

`initialize_encoder_embeddings_from_sae` controls only the initial value of
`encoder.feature_embed.weight` during encoder-decoder pretraining. It is applied
after the model's normal random `init_weights()` pass and before optimizer setup.
Resume runs skip it because checkpoint weights are authoritative.

Modes:

| Mode | Behavior |
|------|----------|
| `null` | No-op; keep the normal random feature embedding initialization. |
| `random` | Load the configured SAE `W_dec`, orient it as `(n_sae_features, sae_dim)`, project **the full table** through a deterministic random matrix to the encoder dimension, rescale, and row-subset to the encoder's vocabulary **last** (if compacted). |
| `svd` | Load the configured SAE `W_dec`, orient it, row-subset to the vocabulary **first** (if compacted), project the surviving decoder directions onto their top right-singular directions, then rescale. Requires `encoder_dim <= min(V, sae_dim)`: the covariance's rank is bounded by **both**, and because the subset happens first, a narrow `feature_mask` can be the binding constraint. A rank-deficient sub-dictionary is **refused** — it would otherwise emit `encoder_dim - rank` identically-zero columns that no later check catches. |

Both non-null modes preserve the average row norm of the random table they
overwrite: the projected SAE directions are scaled so their mean row norm matches
it. Magnitude embeddings, value embeddings, decoder embeddings, cross-attention
weights, and feature-prediction weights are not changed by this option.

`initialize_encoder_embeddings_cluster_vectors` controls the rows retained for
static clusters. Its default is `mean` (absent, `null`, and explicit `mean` are
equivalent). For every source cluster, take its ascending mask-surviving member
IDs `S`, keep `r = min(S)`, and replace only row `r` by the fp32 mean over `S`.
Empty-survivor clusters are skipped, singleton survivors are counted but remain
bitwise no-ops, mask-excluded members never contribute, and every
non-representative row stays verbatim. The matrix and stage are mode-specific:

- **`random`:** let `N` be the normalized fp32 full-dictionary random projection.
  Set `N[r] = N[S].mean(dim=0)` **after** the global rescale and **before** the
  final effective-vocabulary gather. Averaging before rescaling or after gathering
  is a different algorithm.
- **`svd`:** let `D` be the oriented fp32 full-dictionary `W_dec`. Set
  `D[r] = D[S].mean(dim=0)` **before** selecting effective-vocabulary rows. The
  selected averaged dictionary then feeds covariance, `eigh`, projection, and
  rescaling, so the basis itself reflects the averaged dictionary. Averaging after
  the subset or projection is a different algorithm.

`representative` explicitly selects each representative's own direction and adds
no averaging operation to either path. In every inert combination — no clusters,
or `null` init — `mean` has nothing to average and is bitwise the pre-option path,
so the default is safe for every unclustered config. Before either active mean path
touches `W_dec`, the initializer verifies the original SAE width and requires the
live model vocabulary to equal
`effective_feature_vocab(feature_clusters, n_sae_features)`. A cluster with
multiple surviving members necessarily makes that vocabulary compact, so an
uncompacted active-cluster model is rejected rather than given ambiguous rows.

**Compaction (#695).** `n_sae_features` is taken from the encoder *config*, never from the table's row count (which is `V` once compacted). Compact row `j` receives the SAE direction of original id `vocab.original_ids[j]` in **both** modes — but *where the row subset happens* differs between them, and it is load-bearing.

**The rescale target is the same in both modes, and it is never the compact table.** `sae_embedding_init` reads `encoder.feature_embed_full_avg_row_norm`: the average row norm of the **full** `(n_sae_features, d)` random table — the number the *uncompacted* run would have used. On a compacted model that full table exists only as a transient inside `init_weights()`, which captures the norm (after the `COMPUTE_DTYPE` cast) for this purpose. The property **raises** if it was never captured rather than falling back to the compact table's norm, because that fallback is precisely the defect it exists to remove: a `V`-row subset's mean row norm is only a *sample estimate* of the full table's (~2.1e-04 relative at `V=35,275` of 65,536), so rescaling to it would let compaction silently perturb the init. Compaction is purely representational; it must not.

> **Do not "simplify" that property read to `average_row_norm(feature_weight)`.** The compact table is sitting right there and the swap looks harmless. It would silently destroy the bitwise result below.

| | `random` | `svd` |
|---|---|---|
| row subset | **last** — after projection *and* rescale | **first** — before projection |
| matrix projected | the full `(n_sae_features, sae_dim)` | the `(V, sae_dim)` sub-dictionary |
| rescale target | full table's average row norm | full table's average row norm (**identical**) |
| static-cluster vector policy | `representative` or `mean`; `mean` runs after rescale and before the final gather | `representative` or `mean`; `mean` runs on oriented `W_dec` before the subset/covariance |
| vs. uncompacted reference | `representative` is **bitwise identical**; no uncompacted active-cluster `mean` model is valid | differs irreducibly from selected-row covariance; active `mean` additionally changes the dictionary and basis |

> **`svd` is the one place compaction is not an identity.** The covariance `matrixᵀ·matrix` is computed over the **selected rows only**, so the principal directions are those of the *sub-dictionary*: a compacted `svd` init does **not** equal the row-subset of an uncompacted `svd` init. This is a deliberate design decision, not a bug. Under cluster vectors `mean`, the selected rows come from the averaged `W_dec`, so covariance and basis describe the averaged sub-dictionary; `representative` skips that operation bitwise. `random`, by contrast, is **bitwise identical** to the uncompacted reference under `representative` — not merely "the same directions with a different rescale". It runs the *identical* GEMM (the full matrix, through a seeded, `V`-independent projection) and the *identical* rescale (the same target), and only then keeps the vocabulary's rows. Subsetting first would be mathematically equal but would run a different-shaped GEMM, leaving byte-exactness hostage to how BLAS reduces at `M=V` vs `M=n_sae_features` — which is *why* the subset is last. Active `random` + `mean` deliberately replaces retained representative rows after that rescale. `initialize_encoder_embeddings_from_sae` defaults to `null`, so the default path is unaffected either way (and invariant V4 below holds on it).

## Hardware-Specific Configs

### Pretraining variants

| Config | Hardware | Batch | FP8 | Notes |
|--------|----------|-------|-----|-------|
| `conditional_encdec_train.yaml` | Default (8xH100) | 32 | false | Full model |
| `conditional_encdec_4090.yaml` | RTX 4090 | 4 | false | d=12, seq=1024 |
| `conditional_encdec_4xh200.yaml` | 4xH200 | 16 | true | |
| `conditional_encdec_2xh200.yaml` | 2xH200 | 8 | true | |
| `conditional_encdec_2xb200.yaml` | 2xB200 | 32 | false | threshold selection |

### SFT variants

| Config | Hardware | Batch | Seq len | Grad accum | Notes |
|--------|----------|-------|---------|------------|-------|
| `conditional_encdec_sft.yaml` | Default (8xH100) | 32 | 1024 | 2 | Full SFT loop |
| `conditional_encdec_sft_4090.yaml` | RTX 4090 | 4 | 1024 | 128 | Single-GPU SFT; fits 24GB HBM |
| `conditional_encdec_sft_profile_4090.yaml` | RTX 4090 | 4 | 1024 | 8 | Short d8 profiling run with timing CSV and optional trace |
| `conditional_encdec_sft_2xh200.yaml` | 2xH200 | 8 | 1024 | 32 | 2-GPU DDP |
| `conditional_encdec_sft_4xh200.yaml` | 4xH200 | 16 | 1024 | 8 | 4-GPU DDP |
| `conditional_encdec_sft_2xb200.yaml` | 2xB200 | 32 | 1024 | 8 | Larger HBM lets 2 GPUs match 8xH100 device batch |

> **FP8 note**: the pretraining variants set `fp8: true` on H200 and exercise `nanochat.fp8_utils.apply_fp8_training`. The SFT script does not call that helper, so its config inherits `_train_common.yaml` directly and has no `fp8` key. Appending or inheriting `fp8` (or another pretraining-only key) is rejected immediately after Hydra composition and points operators to `conditional_encdec_train.py`. Adding FP8 support to SFT remains a separate workstream.

All hardware SFT variants inherit from `conditional_encdec_sft.yaml` and only override `device_batch_size` (and `max_seq_len` for the 4090). The profiling variant also overrides the model tag, run horizon, and profiling output knobs. Every other SFT-specific knob — `sft_parquet_dir`, `grad_clip_max_norm`, `resume_from_step`, `init_lr_frac`, and `dry_run` — propagates automatically. The effective initial LR is base group LR × `lr_multiplier` × `init_lr_frac`; `dry_run=true` skips checkpoint writes but leaves Wandb unchanged.

## On-Policy Self-Conditioning Post-Training

An eligible checkpoint with an FP head can be post-trained on conditioning
selected by that head. The rollout is on-policy and no-grad; language-model
gradients reach the policy through a two-pass, one-hop STE in either head-only
or full-model scope. The default packed-pretraining source requires an
uncompacted checkpoint and reuses its packed rows and recorded data identity.

With `selfcond.data_source=sft`, the same stage can consume the SFT data family
produced by [Encoder-Decoder SFT](#encoder-decoder-sft): an explicit SFT
checkpoint leaf supplies the recorded dataset and generator policy, while the
matching converted conversations and extracted SFT features supply train and
validation batches.

See [On-Policy Self-Conditioning Post-Training](selfcond-post-training.md) for
the complete prerequisite matrix, algorithm, CLI, configuration, evaluation,
and the explicit-path-only checkpoint consumption rule.

## Evaluation During Training

The training script evaluates:

1. **Conditioned BPB** — Validation bits-per-byte with real encoder features
2. **CORE metric** — Full-context cached conditioning:
   - `absolute_features`: Fixed top-K absolute-feature smoke signal at median magnitude
   - static configs add non-empty target deltas on top of prompt/cache features
3. **Sampling** — Text generation with full-context conditioning and optional target deltas

For normal pretraining runs, conditioned BPB is logged under `val/*` using the
single validation packed-row directory (`val_s*`). For feature-shard splice
runs using the canonical derived layout `<base>/splice/cut_frac*/seed*`, the script
runs two conditioned-BPB validation passes at the same eval cadence. Which
loader flavor backs the two passes depends on `feature_delta.mode`:

- **Packed-row flavor** (`feature_delta=null`, or
  `feature_delta.mode=splice_counterfactual` with complete preselected shards):
  `val/*` reads the original, unspliced base feature directory's
  validation packed rows, matching the baseline validation distribution;
  `val_spliced/*` reads the active derived feature directory's spliced
  validation packed rows. The derived and base packed-row dirs must both
  exist. If the spliced feature dir was written with a custom `output_dir`,
  pass `original_val_feature_dir=<base feature_dir>` so the baseline
  validation rows can be resolved. Packed rows do not support `prev_segment`/
  `cum_max`: combining either with a spliced dual-validation directory raises
  at startup. For `splice_counterfactual`, train and `val_spliced/*` use
  delta-aware `sel/<hash>/` shards over the derived packed rows; `val/*` uses
  the delta-free twin hash over the original/base validation packed rows
  (`feature_delta=null`, `feature_selection.delta_weights` removed).
- **Live flavor** (`feature_delta.mode=splice_counterfactual`, issue #596):
  when `use_precomputed_batches=false`, both passes run through live loaders —
  see the `feature_delta.mode` values below (§ Cross-shard Batch Padding
  Contract).

The canonical seed leaf also accepts
`seed{seed}_opposite_percentile_pool{B}` from activation-based splicing. It
discovers the same original/base validation directory automatically. Pairing
mode changes corpus composition; downstream selection and would-have-followed
provenance retain their existing contracts. See
[opposite-percentile pairing](feature-shard-splice.md#optional-opposite-percentile-pairing).

## Startup timing and effective data path

`scripts.train.conditional_encdec_train` emits an opt-in startup trace for the
standalone entrypoint. The trace is created before Hydra composition and ends
at the first completed optimizer update (or explicitly reports
`no_training_step`/`error`). It does not add CUDA synchronization, DDP barriers,
artifact scans, or hash work. Timings are rank-0-local wall-clock values; they
are not a cross-rank critical-path measurement.

Each completed phase is one INFO line with the stable prefix
`encdec_startup_phase` and `phase`, `status`, `elapsed_s`, and
`cumulative_s` fields. The phase set covers Hydra/config resolution, DDP/device
init, tokenizer and feature/statistics/overlay resolution, boundaries and
identity, model materialization, packed/selection discovery, compile-wrapper
construction, evaluation contexts, W&B setup, checkpoint/resume, the per-config
feature pipeline and optimizer, the final data-path decision, dataloader
construction, first-batch fetch, and pre-first-update control. A phase is
`skipped` when its optional work is disabled and `error` when its work raises;
the original exception is still re-raised.

After the final `_packed_row_inputs_for_pipeline` decision, the trainer emits
one `encdec_startup_data_path` line. `packed_rows=yes` with
`selection=selection_shards` means exact precomputed selection shards;
`selection=fused_base_csr` means packed rows are present but selection is fused
from base CSR. `packed_rows=no` is the assembled/live feature-pipeline route,
with `reason=feature_delta`, `precomputed_disabled`, or
`packed_rows_unavailable` identifying the fallback. `source=remote_certified`
is reserved for the verified remote artifact route; otherwise it is `local`.
The event includes `resolution_s` for shared artifact discovery and
`decision_s` for the per-config route decision.

The first completed update emits `encdec_startup_first_step`. Its
`first_step_s` is exactly the existing post-optimizer-synchronize `dt`, while
`time_to_first_step_s` is measured from trace creation and `update_step` is
invocation-local (so resumed runs do not require step zero). The final
`encdec_startup_summary` reports the invocation-local mean of updates after the
first with numeric `step > 10`, the count used, and a compact JSON map of phase
seconds. These records are observational only and do not enter selection
hashes, checkpoint identity, W&B config identity, RNG state, or fast-sweep JSONL
semantics. For deterministic lifecycle tests, the helper accepts an injected
clock and emitter; normal training uses the repository logger.

## Converting Decoder to Encoder-Decoder

Initialize an encoder-decoder from a pre-trained decoder-only checkpoint:

```bash
# Convert SFT checkpoint
python -m scripts.preprocess.convert_decoder_to_encdec \
    source=sft \
    encoder.n_embd=0.5 \
    encoder.depth=0.5 \
    cross_attention.frequency=4 \
    n_sae_features=16384 \
    num_buckets=10

# Convert base checkpoint
python -m scripts.preprocess.convert_decoder_to_encdec \
    source=base \
    output_tag=my_experiment
```

This:
1. Loads decoder weights from checkpoint
2. Builds EncoderDecoderGPT with fresh encoder + cross-attention
3. Initializes cross-attention, including `c_proj`, with the model's standard uniform init
4. Saves to `conditional_encdec_checkpoints/{output_tag}/`

Cross-attention `c_proj` is intentionally **not** zero-initialized. The older
exact-decoder-output startup promise was removed because all-zero `c_proj`
creates degenerate FP8 activation scaling. Encoder block output projections
remain zero-initialized for residual-friendly encoder startup.

## Encoder-Decoder SFT

Fine-tune an encoder-decoder model on chat tasks using
`scripts/train/conditional_encdec_sft.py`. SFT is always conditioned:
each nanochat-rendered conversation — truncated after its last assistant
turn, exactly as conversion rendered it — is paired with pre-extracted SAE
features; the encoder sees real full-sequence features and the decoder learns
to condition on them. Target supervision is controlled by `sft_loss_mask`;
the default is the historical `full_sequence` objective, with optional
`completion_only` behavior described under [SFT Loss Mask](#sft-loss-mask).
This requires running the same feature extraction pipeline on the SFT dataset
before training.

### Available datasets (`dataset=` CLI arg)

Defined by `tasks.mixtures.SFTMixture` and shared with decoder-only SFT:

| Dataset | Train tasks | Val tasks |
|---------|-------------|-----------|
| `safety` | `SFTMixture.SAFETY` — see [Safety mixture](chat-training.md#safety-mixturesafety) | optional val split from mixture |
| `capabilities` | `SFTMixture.CAPABILITIES` — see [Capabilities mixture](chat-training.md#capabilities-mixturecapabilities) | optional val split from mixture |
| `standard` | `SFTMixture.STANDARD` — see [Standard mixture](chat-training.md#standard-mixturestandard) | optional val split from mixture |
| `standard_steered` | `SFTMixture.STANDARD_STEERED` — standard plus a steered JSONL, evenly spaced — see [Steered mixtures](chat-training.md#steered-mixtures-mixturestandard_steered-mixturestandard_steered_balanced) | shared with `standard` |
| `standard_steered_balanced` | `SFTMixture.STANDARD_STEERED_BALANCED` — the same, with standard down-capped to a 50/50 mix | shared with `standard` |

> Datasets without a val split skip the validation-BPB eval loop entirely (no crash — `has_val` is just False).

> The steered mixtures mint no val phase of their own: `sft_val_family` maps
> both onto `standard`, so validation reads `sft_data/standard_val` and the
> `standard_sft_val` feature phase, and the val fingerprint records
> `mixture=standard`. `dataset=standard_steered*` requires
> `steered_jsonl=/path/to/steered.jsonl`, which must be the SAME file the
> parquet was converted from — the fingerprint gate compares row counts and
> order, not content.

### SFT Feature Extraction

The extraction pipeline is the same as for base pretraining data, but with `dataset_phase` set to the SFT mixture. You must first convert the SFT data to parquet, then extract features from it:

```bash
# 1. Convert train and validation conversations to separate parquet families
python -m scripts.preprocess.convert_sft_to_parquet mixture=safety split=both

# 2. Extract features from SFT data (note: dataset_phase, not base data)
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    dataset_phase=safety_sft \
    segment_mode=turn_delimited_fixed_tokens \
    context_window=2048 \
    num_shards=27

# 3. Extract validation features from the validation parquet family
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    dataset_phase=safety_sft_val \
    segment_mode=turn_delimited_fixed_tokens \
    context_window=2048 \
    num_shards=27

# 4. Compute statistics on TRAIN features only
python -m steering.compute_statistics \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/safety_sft/no_bos/turn_delimited \
    device=cuda
```

`context_window` for SFT phases must be at least **twice the trainer's
`max_seq_len`** (2048 for the default `max_seq_len=1024`, 4096 for a 2048-token
base). The trainer keeps `max_seq_len` *nanochat* tokens of every conversation
and its coverage gate requires the feature shard to reach that prefix; the
feature model's tokenizer (Gemma splits digits, for instance) can need more
tokens than nanochat's for the same text, so a window equal to `max_seq_len`
leaves long rows uncovered and the trainer stops with `feature coverage ends N
chars before the rendered conversation`. Extraction logs a WARNING for SFT
phases below 2048 (#1634).

Omit `segment_mode=turn_delimited_fixed_tokens` to retain fixed-token SFT
extraction. With the turn-delimited setting, auto-constructed feature paths end
in `/no_bos/turn_delimited` (or
`/context_independent/no_bos/turn_delimited` when context-independent features
are enabled).

#### Steered mixtures

A steered mixture converts TRAIN ONLY (`split=val`/`split=both` are rejected)
and borrows standard's validation artifacts, so its val side is converted and
extracted once under `mixture=standard`:

```bash
# 1a. Convert the steered TRAIN mixture (steered_jsonl is required here)
python -m scripts.preprocess.convert_sft_to_parquet \
    mixture=standard_steered split=train \
    steered_jsonl=/path/to/steered.jsonl

# 1b. Convert the shared VAL family once
python -m scripts.preprocess.convert_sft_to_parquet mixture=standard split=val

# 2. Extract the steered TRAIN features
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    dataset_phase=standard_steered_sft \
    segment_mode=turn_delimited_fixed_tokens \
    context_window=2048 \
    num_shards=27

# 3. Extract the SHARED val features (standard's phase, not a steered one)
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    dataset_phase=standard_sft_val \
    segment_mode=turn_delimited_fixed_tokens \
    context_window=2048 \
    num_shards=27
```

Then train with `dataset=standard_steered steered_jsonl=/path/to/steered.jsonl`
(use `standard_steered_balanced_sft` / `dataset=standard_steered_balanced` for
the balanced member). A default `artifact_format=full_nanochat_text`
conversion records `steered_rows` in `conversion_meta.json`; the structured
rollout format does not.

#### Turn-delimited SFT features

Point `feature_dir` at the canonical `…/turn_delimited` feature family to train
with turn-delimited feature segments. When validation is enabled, its feature
directory must use the same segmentation mode as training. The trainer records
the discovered mode as top-level checkpoint metadata `sft_segment_mode` and
requires it to match on resume; legacy checkpoints without the key mean
`fixed_tokens`. Inference resolves the effective mode from every checkpoint
lineage witness and rejects malformed or conflicting modern metadata. The
AxBench-upstream project methods
(`scripts.eval.axbench_upstream.run_methods`) and encoder-decoder baselines
(`scripts.eval.axbench_upstream.run_encdec_baselines`) extract the exact
rendered prompt with turn boundaries. Both surfaces self-condition, so a
turn-delimited conditioned-SFT checkpoint is not runnable on AxBench as it comes
out of this trainer: setup resolution also requires the generated-assistant
cadence `turn_self_cond_segment_size_rustbpe`, which only self-conditioning
post-training measures and records at the metadata top level (see
[`selfcond-post-training.md`](selfcond-post-training.md)). Run that step first.
Target-real ChatCORE now supports turn checkpoints through
`scripts.eval.chat_eval --config-name=chat_eval_turn`, after schema-5 enumeration,
lineage-selected conversion and deep source/routing validation (see the
[turn runbook](evaluation.md#turn-delimited-target-real-context-caches)).
It admits single-process greedy static reference conditioning: pooled current/next,
lockstep current, and single-slot lockstep next. Multi-slot lockstep next remains
rejected. Prompt routing and the marker-inclusive reference schedule use the
unchanged full-sequence SFT routing primitive. Calculator output retains its
actual decoder positions. These static pooled/lockstep paths require no
generated-assistant cadence; a value, if present, must still be valid and does
not affect static outputs. They do not run FP self-conditioning through tool turns.
The AxBench inference surfaces above retain their existing cadence requirement.

Consumers that still depend on static
fixed-window context caches or raw fixed-window extraction fail closed on a
turn-delimited checkpoint or feature artifact; this includes the generic
base/DCLM/steering/progression evaluations, turn `core_target`, AxBench-Base/BPB raw-cache
paths, completion tools, harmfulness evaluation, and the interactive chat TUI.
Persistent eval context-cache schema v4 carries a mandatory fixed-token source
witness. Turn schema v5 is admitted only by the explicit checkpoint-contract
path; changing a cache's mode label or directory cannot grant other consumers
turn support. Distributed turn ChatCORE and encoder-decoder greedy microbatching
also remain rejected. Older caches must be rebuilt because their source
geometry is unauthenticated.
There is no warning-and-continue fallback. Interactive multi-turn support is a
separate follow-up because it must atomically re-extract the whole rendered
conversation and reset generated-assistant self-conditioning at every turn;
see [#1627](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1627).

The `dataset_phase` parameter controls which data split is used. Train phases
are `standard_sft`, `safety_sft`, and `capabilities_sft`; their validation
siblings are `standard_sft_val`, `safety_sft_val`, and
`capabilities_sft_val`. The output is written to a phase-specific subdirectory
in the feature hierarchy. Validation needs no separate statistics run: its
conditioning buckets and percentile tables deliberately reuse the train
statistics that define the trained conditioning scheme.

#### SFT Parquet Format

SFT parquets store `text`, `original_index`, and `turn_char_starts` columns.
`text` is the nanochat-rendered conversation, truncated after its last assistant
turn, with the decoder BOS stripped; `original_index` maps rows back to the
source mixture. `turn_char_starts` contains text-coordinate character offsets
where each logical user, assistant, or tool-output turn begins; element 0 is 0,
and the offsets come from the same render as `text`. Pre-existing conversion
directories lack this column and must be regenerated into a fresh directory.

Conversion and conditioned SFT training share **one** conversation-render
contract, owned by `prepare_messages_for_full_nanochat_render`
(`scripts/preprocess/convert_sft_to_parquet.py`). It truncates every
conversation after its last assistant message and reports a skip reason for
rows with no assistant turn (`no_assistant`) or an empty final assistant
response (`empty_response`). The converter renders that slice into `text`, and
feature extraction stores `char_start`/`char_end` over the same string, so
`conditional_encdec_sft` applies the identical helper before tokenization and
loss-mask construction — via `_prepare_sft_render_messages`, which both the
startup content probe and the data generator call. Rendering the raw row
instead would append a trailing user turn worth at least 22 characters that no
feature segment covers, which the coverage guard rejects (issue #1465).

A converter-skipped row can never reach the generator: the metadata-backed path
walks the parquet's own `original_index` set, and the legacy path dies earlier
on the missing-features guard. Meeting one anyway means the live task drifted
from the conversion behind a composition-only mixture fingerprint, so both
surfaces raise `DataLoadError` rather than silently dropping the row.

The non-conditioned trainer (`scripts/train/chat_sft.py` via
`steering/sft/iteration.py`) deliberately keeps the untrimmed render: it has no
parquet, no feature store and no `original_index` alignment, so nothing
constrains it to this contract.

Conversion resume requires the same requested `max_conversations` (including
`null`), `conversations_per_shard`, effective input-prefix size, mixture, and
split. A mismatch fails before changing shards or metadata; use a fresh output
directory for a different conversion. Existing shards without recorded sizing
identity cannot be upgraded in place, but remain readable by downstream tools.
Empty legacy progress without shards can still start a fresh conversion.
A nonblocking exclusive directory lock covers legacy-converter preflight through
publication; a second legacy writer to the same directory fails before reading
or changing artifacts.

Before reusing any shard, the converter checks its `original_index` sequence
against the live input range and the shared skip predicate, then re-renders
every retained conversation to compare the stored text and logical turn starts.
Resume therefore re-tokenizes existing rows; composition-only mixture identity
cannot prove their content has stayed unchanged. Missing, duplicate, reordered,
out-of-range, or unexpected shards fail before progress publication. Identity
is written before the first shard, and shards and metadata are published
atomically, so a shard left ahead of the progress cursor by an interruption
can be audited and reused.

`total_conversations` retains its input-prefix meaning for downstream coverage
checks. `written_conversations` and `skipped_conversations` record the actual
counts through `completed_shards`, including reused shards. The completion log
reports both counts; resuming cannot bypass the existing >50% skip-rate guard.
The final completed cursor is published only after that guard passes, so a
rejected conversion remains incomplete for downstream loaders. If an older
converter prematurely marked a skip-rejected conversion complete, an audited
resume revokes that marker before raising.

`conversion_meta.json` also content-binds positional indexing with a mixture
fingerprint: mixture name and split, constituent task class names and lengths in
construction order, `len(mixture)`, and a SHA-256 over the deterministically
shuffled `TaskMixture.index_map`. Extraction stamps the canonical fingerprint in
each base shard's metadata. Those uniformly validated shard headers are the
authoritative feature-side carrier for both train and validation families.
`compute_statistics` copies the train fingerprint into
`feature_statistics.safetensors`; when that metadata is present, loading treats
it as corroboration and rejects any disagreement with the shard headers.
Validation continues to require no statistics artifact. Conditioned SFT compares
the parquet and authoritative shard fingerprint, then recomputes the identity
from the live train or validation task before building the first batch. For a
capped conversion (`max_conversations`), `total_conversations` is the attempted
prefix length while the fingerprint still identifies the full source mixture.
For any metadata-backed conversion — capped or not — the reference document set
is the parquet's own `original_index` column, read independently of the feature
shards. SFT consumes exactly those rows, and the feature index must cover them
in *both* directions: a parquet row without features and a feature row without a
parquet row are each rejected. Legitimate converter skips (`no_assistant`,
`empty_response`) therefore remain representable in a full-mixture run instead
of being reported as missing extractions — they are absent from the reference
set, so the generator never renders them. Feature/parquet shard stems must match
in both directions, and feature-shard `n_docs` and parquet row counts must
agree. Legacy artifacts without conversion metadata retain the historical
full-mixture key coverage requirement.

Statistics remain a separate contract: validation reuses train statistics only,
never validation statistics. A small capped train statistics artifact may not
contain every allowed feature activated by the capped validation prefix; the
feature-prediction/percentile path fails loudly in that case. Do not skip those
features or relax the guard; use a sufficiently complete train statistics source
or run the full conversion before relying on validation metrics. In particular,
do not substitute a default percentile for a missing row: the same lookup feeds
the feature-prediction supervision targets, so a substituted value teaches the
model a specific activation level for a feature whose activation is simply
unknown. The percentile lookup queries the RAW pre-selection feature ids, before
feature-mask filtering and before cluster consolidation, so the statistics
artifact must cover every ACTIVE feature, not just the allowed representatives —
rebuilding statistics over the allowed set alone will fail the same guard.

### `missing_percentile_policy`

Coverage is checked ONCE, up front, before the first batch is built, rather than
at the first offending lookup. The preflight walks the covered train documents
(and the validation documents when `eval_every > 0`), unions their raw
pre-selection segment feature ids, and validates each active percentile table
against the set it is actually queried with — these are different sets and are
checked independently:

| table | queried with |
| --- | --- |
| `fp_percentile_table` | the RAW `seg.feature_ids` of every segment, before feature-mask filtering and before cluster consolidation |
| `sampling_bundle.percentile_table` | those ids after feature-mask filtering, still pre-consolidation (the clustered path consolidates *after* the lookup and hands `sample_features` the consolidated percentiles as override weights) |

The two coincide only when the feature-prediction table *is* the sampling
bundle's table. The report always names the COMPLETE distinct uncovered
population per table, not the first few raw hits, because the operator's
decision — rebuild statistics, shrink the mixture, or accept a pilot — needs the
whole picture.

The scan itself is ID-only and rank-once (issue #1495): from each covered
shard it decodes and returns only the document ids, CSR offsets, and feature
ids (no `SegmentFeatures` construction, no magnitude upcast, no
char/token-boundary decoding — the whole compressed shard body is still
decompressed and deserialized, as the storage format requires), unions them
with vectorized tensor ops, runs on rank 0 only, and broadcasts the resulting
bitmap to every rank through the same aligned-failure protocol as the other
startup stages. Any anomaly the full walk would fail on in the fields the
scan reads (a document missing from its shard, a feature id outside the SAE
width, a malformed CSR) re-runs the reference document walk so the raised
error is the same one the pre-#1495 scan produced. Corruption confined to
shard fields the scan never decodes (segment index/position/char-boundary
tensors, and a `feature_mags` length mismatch) is still caught with today's
exact error by the unchanged full loader — on that shard's first access by
the data generator, which for a late shard may be after optimizer steps have
run (the pre-#1495 preflight full-loaded every covered shard up front and
caught it at startup).

`missing_percentile_policy` then decides what happens:

- **`error` (default, production).** Abort at startup with the full count.
  Nothing is substituted anywhere, so every percentile-derived value is
  byte-identical to a run without this preflight.
- **`lowest` (PILOT — explicitly NOT comparable).** Every uncovered feature is
  pinned to the bottom of every statistics-derived quantity, using the values
  the percentile primitive itself emits for a below-p1 magnitude: percentile
  rank `0.5`, normalized rank / feature-prediction target `0.005`, magnitude
  bucket `0` if selected, and an infinite per-feature percentile threshold so
  the unclustered `feature_percentile` selectors agree with the percentile-space
  clustered ones (an uncovered feature's precomputed threshold is otherwise
  `0.0`, which admits every positive magnitude — the opposite of rank `0.5`).
  One contract, every consumer: unclustered threshold selectors, both supported
  clustered percentile selectors, feature-prediction targets, weighted sampling,
  top-k sampling, feature masks, compact FP layouts, and binary bucket spaces.
  Weighted sampling can still draw an uncovered feature — its weight `0.5` is
  positive — which is exactly why the bucket pin exists.

The resolved policy is written to every SFT checkpoint as
`missing_percentile_policy` and a resume under a different policy is rejected,
exactly like `sft_loss_mask` and `feature_mask_canonical`. A `lowest` pilot's
`val/bpb` and `min_val_bpb` must never be silently compared with a production
number, and the startup log line says so.

The preflight does NOT replace `magnitude_to_percentile`'s own raise. That
primitive is untouched and still refuses an all-zero row, so anything that slips
past the bitmap still fails loudly.

Every startup validation above (artifact coverage, statistics coverage, and the
percentile coverage bitmap) is RANK-SYNCHRONIZED. Artifact/store loading and the
scan+coverage check run as two consecutive aligned stages (`percentile coverage
sources`, then `percentile coverage`), and rank 0 computes the ID scan result
before broadcasting it to every rank. On a shared or flaky filesystem exactly
one rank can fail to read a shard while its peers proceed into the next
collective and block until the NCCL watchdog kills the job with an error that
names nothing useful. The preflight all-reduces the failure outcome
(`ReduceOp.MIN` over the failing rank id) and broadcasts the failing rank's
message, so every rank aborts at the same point with a message naming which rank
failed and why. This applies to those startup stages only; it is deliberately
not retrofitted onto arbitrary pre-existing raises.

Fingerprint identity is complemented by an independent content channel. At
generator startup, `alignment_probe_samples` (default `8`) selects a fixed,
endpoint-inclusive spread of indices without RNG, renders each live conversation
through `_prepare_sft_render_messages` +
`render_messages_to_full_nanochat_text` — the same preparation the data
generator now applies to every row it packs — and byte-compares it with the
parquet `text` row for that `original_index`. Only the sampled rows' parquet
shards are read. `alignment_probe_samples=0` explicitly disables the probe with
an INFO while leaving fingerprint checks enabled. A legacy pair with neither
fingerprint emits one WARNING with a fresh-output-directory regeneration command
and proceeds only when the probe is active. Legacy artifacts combined with
`alignment_probe_samples=0` raise because neither verification channel remains;
one-sided or mismatched fingerprints always raise. A conversion with completed
legacy shards cannot be upgraded in place: rerun
`convert_sft_to_parquet` with `output_dir=/path/to/fresh/sft_data`, then extract
into a fresh feature directory.

#### Cross-Tokenizer Segment Boundaries

Feature extraction and SFT training use different tokenizers. Segment boundaries
are stored as character offsets over the rendered source text, truncated after
the conversation's last assistant turn. During SFT training,
`build_full_sequence_segment_routing()` aligns those ranges to nanochat token
offsets and shifts decoder BOS coordinates with
`feature_source_bos_char_len` when the source tokenizer's BOS width differs. The
SFT config defaults this expected value to `5`, matching Gemma's decoded
`<bos>` span.

That value is verified rather than trusted. Before generator construction, the
trainer loads `doc_ids`, `segment_char_ends`, and `doc_token_counts` from the
lexicographically first feature shard, pairs it with the same-stem parquet, and
examines the parquet's first row group. It skips documents whose token count
fills the entire extraction context and computes
`last_segment_char_end - len(parquet_text)` for the first complete document with
a non-negative result. The result must exactly equal the configured value. The
check runs for train and for validation whenever validation is active, so an
explicit validation override cannot silently introduce a different source
coordinate system.

This is a fail-closed startup invariant. Empty/unreadable feature families,
missing paired parquets, a first row group without usable evidence, and value
mismatches all raise `DataLoadError`; startup neither rewrites artifacts nor
silently replaces the configured value. The same check runs in SFT-backed
self-conditioning with the checkpoint-recovered value (legacy default `5`).

#### Head Truncation

Both base and SFT extraction use head truncation: tokens are kept from the start up to the configured training window (`token_ids[:n_kept]`). For SFT, this matches the full-sequence packing used during SFT training.

#### Conditioned-SFT startup trace

The conditioned SFT entrypoint emits an always-on startup trace at `INFO`; no
profiling override is required. It starts at `main()` entry and ends after the
first successful optimizer step, using monotonic wall-clock time. The trace
separates configuration/device/checkpoint/model/compiler setup from the lazy
train and validation feature-store work, artifact identity and document
coverage checks, the source-BOS origin check, the content-alignment probe,
percentile coverage preflight, and first-batch construction. Repeated
train/validation work is tagged with `source`; feature/parquet shard counts and
document counts come from the same loader validation pass rather than a second
scan.

The first batch is lazy: creating `make_sft_data_generator` does not load the
feature store or run the probe. Those phases appear when the first batch is
requested (and validation has its own lazy phases). The first optimizer step is
reported separately as `first_optimizer_step` and `first_step_warmup_s`; it
includes the existing synchronization, first forward/backward/compiler work,
and optimizer update. The closing `time_to_first_step` summary also reports
the measured startup-phase sum and an explicit unattributed residual. A run
that exits before an optimizer update reports `status=not_reached` rather than
inventing a warmup or steady-state value. This telemetry does not alter task
order, RNG state, artifact identity, checkpoint contents, prefetch boundaries,
or existing live-loop timing controls.

### SFT Training

```bash
# Safety SFT with full-sequence feature conditioning
python -m scripts.train.conditional_encdec_sft \
    dataset=safety \
    model_tag=my_model \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/safety_sft

# Distributed
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_sft -- \
    dataset=safety \
    model_tag=my_model \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/safety_sft
```

SFT is a consumer of the base checkpoint's feature dictionary. It has no
`feature_mask_freq_min` / `feature_mask_freq_max` producer: configure the exact
`feature_mask` and optional path-only `feature_clusters` used to train the base.
Recover the mask with
`uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`.
Startup requires the effective mask/cluster vocabulary to equal the state-dict
vocabulary, including the full-width case, and checks cluster metadata
bidirectionally for both the base and resume checkpoints. Cluster siblings are
max-folded in normalized-percentile space before per-conversation live
selection. SFT does not inherit pretraining's `use_precomputed_batches=true`
constraint because it has no packed precomputed-batch path. It does inherit the
sibling threshold-domain constraint: clustered SFT supports sampling and only
`feature_percentile` / `feature_percentile_segment_topk` thresholds, because no
representative raw magnitude exists. Compact feature-prediction targets are
supported through the checkpoint-derived `FeaturePredictionTargetLayout`:
masked sources are dropped, cluster siblings max-reduce into their
representative row, and duplicate original IDs in one conversation segment
fail with example/segment provenance. SFT checkpoints with an FP head persist
`feature_prediction_width`; compact-FP resumes require that width and the
vocabulary/cluster identity as a complete set.

See the **Available datasets** table above for all three options (`safety`,
`capabilities`, `standard`).

For chained experiments, `scripts.train.run_encdec_sft_pipeline` runs each
phase from the previous phase's checkpoint. The default order is
`phase_order=safety,capabilities`; reverse-order runs use
`phase_order=capabilities,safety`. Conditioned chained runs require extracted
features for each phase in the order. All three mixtures (`safety`,
`capabilities`, and `standard`) are accepted; the default chain remains
`safety,capabilities`. Set the pipeline-only
`feature_segment_mode=turn_delimited_fixed_tokens` argument when those feature
directories carry the `/turn_delimited` leaf. The driver currently rejects `feature_clusters`; run clustered
phases individually and pass each prior checkpoint through an explicit
`base_checkpoint_dir` instead (#709 owns the adjacent leaf-composition work).

```bash
# Reverse conditioned chain: capabilities SFT, then safety SFT.
python -m scripts.train.run_encdec_sft_pipeline \
    model_tag=d12_baseline \
    phase_order=capabilities,safety

# Hardware config files use the shared compose-API config_name convention.
python -m scripts.train.run_encdec_sft_pipeline \
    config_name=conditional_encdec_sft_2xh200 \
    model_tag=d12_baseline
```

#### SFT Loss Mask

`sft_loss_mask=full_sequence` is the default and preserves the historical
behavior: every non-padding target token is supervised. Optional
`completion_only` uses nanochat's chat-template mask, supervising assistant
text and tool calls (`<python_start>` through `<python_end>`) while excluding
user turns and injected tool outputs (`<output_start>` through
`<output_end>`). The latter are forced into the inference stream by the tool
engine rather than generated by the model.

The choice changes only decoder cross-entropy targets. Feature segments remain
routed over the complete rendered sequence, and encoder inputs, `seg_map`,
`seg_doc_id`, and feature-prediction targets stay unchanged. Training and
validation share the same resolved mode. Checkpoints record it at top level;
resuming with another mode is rejected, while historical checkpoints without
the key are interpreted as `full_sequence`. Under `completion_only`,
`min_loss_tokens` counts supervised targets after truncation. Both modes retain
the model's mean loss reduction; evaluating a future default flip must revisit
valid-target-count variance at production batch sizes.

Both modes supervise the converted conversation slice, which ends at the last
assistant turn; an unanswered trailing user turn is not rendered at all. Under
`completion_only` this changes nothing, because those tokens were already
unsupervised. Under `full_sequence` such rows lose the trailing user turn's
targets, which is the point: those tokens had no feature coverage.

Note that under `full_sequence` the `min_loss_tokens` filter counts
`len(ids) - 1` rather than supervised targets, so in principle a very short
conversation could newly fall under the threshold and be skipped — which in a
validation pass would move the BPB denominator. Measured over all 268
trimming-affected rows of the published standard mixture, none does.

#### SFT residual feature heads

`feature_prediction_block=mlp feature_prediction_head_layers=0` is the default.
At depth zero both `mlp` and `transformer` preserve the linear module. A positive
depth upgrades a strictly loaded linear source during **fresh** SFT. MLP blocks
are pre-norm `D → 4D → D` ReLU-squared residuals. Transformer blocks prepend
decoder-style causal attention to each MLP residual, using the captured layer's
absolute RoPE positions and window. All added output projections start at zero;
the retained projection, initial predictions, existing-model gradients and
surrounding CPU/CUDA RNG state are preserved under matched arithmetic.

An expanded source must match the requested block kind and total depth. A
headless checkpoint cannot acquire positive depth. Every resume must match both
saved fields, for example `feature_prediction_block=transformer
feature_prediction_head_layers=6 resume_from_step=500`. Checkpoints explicitly
record `block` and `head_layers` in both feature-config mirrors. Generic loading,
pretraining resume, self-conditioning, GRPO and RLVR enforce the same contract.
Old metadata needs [explicit conversion](feature-head-metadata-migration.md);
there are no old CLI/config aliases or implicit legacy loads.

The usual SFT optimizer trains the entire model, including every residual
parameter in its existing feature-head group. This option does not enable
head-only training. Compact head row order, targets, prediction layer,
loss mode/function/weight and LM outputs keep their existing contracts.

The matched-recipe depth-2 instruct run and its historical-control limitations
are recorded in the [experiment runbook](../performance/residual-feature-head.md).

#### SFT Feature-Prediction Segment IDs

When the loaded checkpoint has a feature-prediction head, SFT training builds
dense `fp_targets` for the same next-segment auxiliary loss used during
encoder-decoder pretraining. Uncompacted targets use the original SAE width;
compact targets fold mask-surviving cluster siblings and scatter directly into
the checkpoint's effective-vocabulary layout. Unlike packed-pretraining's
persisted uint8 roundtrip, the SFT producer supervises continuous normalized
`magnitude_to_percentile / 100` values. Self-conditioned inference always treats
head probabilities as continuous normalized ranks; it does not quantize its
selection or bucket cutpoints to replay packed storage. Any supported SFT batch
with real `fp_targets`
also emits `seg_doc_id`, matching the pretraining dataloader contract.

`seg_doc_id` is the **FP prediction-group id**: consecutive segments may predict
each other only inside one group. Each real segment gets a row-local group id
while segment 0 and padding slots use `-1`, and the FP loss only lets segment
`s` predict `s+1` when both ids match and are non-negative. A group boundary
sits at **every packed conversation boundary** and, for a **steered**
conversation, ALSO at its **first assistant-turn segment**. Without steered rows
the group id is exactly the row-local conversation index, so every non-steered
batch is byte-identical to the pre-steering producer. A prediction-target-only
extra segment created by truncation keeps the group of the turn its
`char_start` falls in, letting the last in-range segment of that turn predict it
without exposing that extra segment to decoder cross-attention. Because these
extra target segments have no decoder-token span, the FP loss treats real
no-token targets as length 1 for weighted modes while keeping `seg_doc_id=-1`
padding excluded.

#### Steered SFT dataset

For the steered mixtures (`standard_steered`, `standard_steered_balanced`) the
assistant response was generated under an activation-steering intervention, so
its extracted features are deliberately perturbed. Predicting them from the
(unperturbed) user turn is not a signal the head should learn, so the generator
starts a NEW prediction group at the steered conversation's first assistant-turn
segment. That suppresses exactly one pair — last user segment → first assistant
segment — which leaves both the numerator and the denominator of every loss
mode, exactly like a packed-conversation boundary. User→user and
assistant→assistant pairs stay valid, cross-attention is unaffected, and
self-conditioning's rollout gate follows automatically because
`continuation_mask` is the target-indexed shift of the same gate — though
SFT-mode self-conditioning rejects steered mixtures outright in v1, so that
property is a mechanism guarantee rather than a supported configuration (see
**Self-conditioning** below).

Mechanics:

- **No new tensor and no batch-contract change.** The split rides in
  `seg_doc_id`, keyed off the steered rows' dataset positions
  (`SteeredTaskMixture.steered_positions`), which are the same
  `original_index` values the generator already serves.
- **Exactly two turns.** A steered conversation must render as one user turn
  followed by one assistant turn. Anything else (for example an assistant
  message carrying a python tool part, which renders as a third logical turn)
  raises `DataLoadError` naming the `original_index` and the roles found. Turn
  boundaries are recovered from the FULL render, before truncation, because
  truncation rewrites the token ids but never a segment's `char_start`.
- **Truncation.** When the row capacity cuts inside the assistant turn, the
  prediction-target-only extra segment is still an assistant-turn segment and
  keeps the second group. When it cuts exactly at the end of the user turn, the
  extra segment IS the first assistant-turn segment and opens the second group
  on its own. When it cuts inside the user turn, no segment of turn 1 survives
  and the conversation stays a single group.
- **Turn-delimited features required.** Steered runs need
  `segment_mode=turn_delimited_fixed_tokens` (a `…/turn_delimited` feature
  family), enforced both at trainer startup and in the generator right after
  the feature store loads.
- **Counters.** Per generator instance the first-pass summary reports
  `steered_seen`, the steered rows dropped by the `min_loss_tokens` filter, and
  the steered rows packed with no assistant-turn segment. They never raise.
  The summary emits once after this rank renders its own dataset share,
  including skipped rows, rather than waiting for the global document count.
  Uneven rank shares use their exact stride lengths. Counters restart with a
  new generator, including checkpoint resume. On resume, the window includes
  ordered buffer replay and counts one initial rank-share worth of renders;
  it does not reconstruct a historical epoch or promise the same steered
  composition as the startup coverage line. The packing count can trail the
  render count by up to `buffer_size` conversations still awaiting packing.
  Default console sinks show INFO summaries on rank 0 only; a buffer summary
  with skipped rows is WARNING and remains visible on other ranks. Set
  `NANOCHAT_LOG_LEVEL_NONZERO_RANK=INFO` to see their INFO summaries too.
- **Per-rank coverage.** Steered rows are evenly spaced at the DATASET level, so
  a rank's `dataset_indices[rank::world_size]` stride can still be unevenly
  steered. That is accepted; each rank logs how many steered documents its own
  stride serves, next to its stride length.
- **Self-conditioning.** Steered mixtures are rejected in
  `conditional_encdec_selfcond` SFT mode — see
  [selfcond-post-training.md](selfcond-post-training.md#data-sources).

No new checkpoint key is written for a steered run: `user_config` already
echoes `dataset` and `steered_jsonl` (the two inputs that define the
interleaving) and `sft_artifact_coverage` already pins that interleaving by
mixture fingerprint on resume.

#### Learning-rate warmup

Ordinary conditioned SFT defaults to `lr_warmup_steps=20`. Only a nonnegative
integer is accepted; `lr_warmup_steps=0` restores the historical LR schedule
exactly. Floats such as `20.0`, booleans, strings, null, and containers fail at
startup. The original fractional proposal in issue #1739 was narrowed to
integer updates; even `+lr_warmup_frac=0.1` is rejected.

Let `s` be successfully completed optimizer updates and `W` the warmup length.
Before each update during warmup, every optimizer group uses
`initial_lr × (s + 1) / W`, composed with the existing decay multiplier.
With `W=20`, the first update uses 5%, update 20 reaches 100%, and later
updates retain the original flat-then-decay schedule. For a 100-update run,
`s=80` uses 100%, `s=90` uses 50%, and `s=100` terminates without an update.
The decay axis is never shifted or renormalized. AdamW and Muon groups,
including the decoder, encoder, cross-attention, and all feature-head
parameters, retain their base-rate ratios.

A fixed `num_iterations=N` requires `5 * W < 4 * N` for positive `W`, checked
before device/model initialization. The default rejects `N=25` and accepts
`N=26`. Epoch mode has no known optimizer-step horizon. It checks synchronized
committed data progress at every boundary: progress at or above 80% with
`s <= W` is an error, including at termination and before validation or
checkpoint publication. Reduce `W` or use zero for short runs. There is no
extra counting pass. Accumulation and prefetched but uncommitted batches do
not advance either schedule axis.

Ordinary presets inherit 20; the six-update profiling preset explicitly uses
zero. LR warmup is separate from Muon's momentum ramp (0.85 to 0.95 over 300
updates), gradient clipping, static `lr_multiplier`/`init_lr_frac` scaling,
and compilation timing reported as first-step warmup. RLVR and structured-SFT
GRPO LR warmup remain deferred follow-up work; decoder-only SFT and pretraining
retain their schedules. A finite tiny-model check does not establish that 20
warmup updates resolve transformer-head divergence (#1735).

#### Resuming an interrupted SFT run

``scripts.train.conditional_encdec_sft`` supports ``resume_from_step`` the same way ``conditional_encdec_train.py`` does. Pass a positive integer to pick up an earlier SFT checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_sft -- \
    dataset=safety \
    model_tag=d26 \
    resume_from_step=5000
```

The default is ``resume_from_step=-1`` (fresh start, inherited from `_train_common.yaml`). On resume:

- The base encdec checkpoint (``conditional_encdec_checkpoints/{tag}/s{seg}/{fs}/model_XXXXXX.pt``) is still loaded to recover decoder/encoder/xattn configs.
- The SFT checkpoint under ``{dataset}_sft_conditioned_checkpoints/`` at the specified step provides the fine-tuned model weights, per-rank optimizer shards (DistMuonAdamW uses ZeRO-2 sharding), a top-level ``step``, and a ``loop_state`` sub-dict with ``sft_cursor_semantics: buffered_v2``, the loss/time scalars, mandatory artifact coverage, and ``sft_dataloader_state``. That dataloader-state envelope contains every rank's committed consumed count, next logical striped cursor, and the remaining best-fit buffer positions in their original order. The generator publishes it only after constructing all batch tensors; the training loop snapshots it only after a successful optimizer step and gathers it on every rank before the rank-0 metadata write. It also pins world size, buffer size, per-device batch size, sequence length, gradient-accumulation steps, minimum supervised-token filter, and loss-mask mode; changing any of them fails closed instead of silently changing the resumed conversation stream. ``sft_consumed`` remains as a rank-0 progress/reporting scalar, not the resume authority, and a caller cannot resume from that scalar alone.
- Historical checkpoints without exact ``buffered_v2`` semantics fail closed. In particular, ``committed_v1`` records only a scalar consumed count: best-fit packing may have prefetched later conversations and popped them out of order, and different DDP ranks may have popped different counts, so the missing ordered per-rank state cannot be reconstructed reliably.
- ``sft_artifact_coverage`` records the covered document count, the conversion prefix length, and BOTH the parquet-side and the feature-side mixture fingerprint, so re-pointing ``feature_dir`` at a different extraction of the same mixture (other layer, SAE width or pooling) is rejected.
- The top-level ``sft_loss_mask`` value must match the configured mode. Historical checkpoints without the key are treated as ``full_sequence``; they cannot be resumed into ``completion_only``.
- The top-level ``sft_segment_mode`` value must match the mode discovered from the training features. Historical checkpoints without the key are treated as ``fixed_tokens``; they cannot be resumed with turn-delimited features, and turn-delimited checkpoints cannot resume with fixed-token features.
- The top-level ``missing_percentile_policy`` value must match the configured policy. Historical checkpoints without the key are treated as ``error`` — the only behaviour that existed then. A ``lowest`` pilot and a production run are not comparable, so neither can be resumed as the other.
- ``feature_prediction_block`` and ``feature_prediction_head_layers`` must equal the saved architecture, including depth zero. Resume validates explicit canonical metadata and strictly restores the complete head before optimizer restoration; it never initializes new residual blocks into a learned checkpoint.
- Clustered-threshold runs (``feature_clusters`` + a ``ThresholdSelection``) stamp ``clustered_threshold_selection_semantics`` (currently ``full_distribution_rank_v1``, PR #1321). Resume rejects a checkpoint whose value is missing or does not match, because pre-#1321 checkpoints computed ``selection_percentiles`` by re-ranking within the post-threshold bucket boundaries — a boundary-relative rank that double-applies the threshold — with no version key at all, and resuming that as the current full-distribution rank would silently change which features get selected mid-run. Sampling-based selection and unclustered runs never stamp or check this key.
- ``make_sft_data_generator`` rematerializes the saved ordered buffer first, then continues from that rank's saved next cursor. This preserves the exact conversation order while leaving the fresh-run best-fit algorithm unchanged.
- All ranks participate in ``save_checkpoint`` (previously only rank 0 did), so every Muon shard is persisted — essential for multi-GPU DDP resume.

For deterministic feature selection, a save → load → continue cycle reproduces the next batch tensors and final model state (see ``TestSFTBufferedCursorRecovery`` and ``TestSFTResumeRoundtrip`` in `tests/nanochat/test_encdec_sft_resume.py`). The checkpoint reconstructs buffer entries from their source conversations rather than serializing their tensors. Consequently, this contract guarantees exact conversation order but does not claim bit-exact selected feature tensors for stochastic sampling modes whose RNG state is not checkpointed.

Exact optimizer resume also requires the requested `lr_warmup_steps` and
`num_iterations` to equal the saved `user_config` values. A changed horizon
changes decay and is rejected. Malformed saved values fail without coercion;
a missing legacy warmup field means zero. Resume a legacy checkpoint with its
original horizon and an explicit zero override, for example:

```bash
uv run python -m scripts.train.conditional_encdec_sft \
    model_tag=<base_tag> feature_dir=<features> dataset=safety \
    resume_from_step=10 num_iterations=-1 lr_warmup_steps=0
```

Saved optimizer `initial_lr` values and the saved completed-update counter
remain authoritative. With `W=20`, resuming at `s=10` uses `11/20` on the next
update, exactly as uninterrupted training would. No scheduler object or extra
mutable state is saved. Starting a **fresh** SFT phase from checkpoint model
weights (`base_checkpoint_dir=<checkpoint>`, `resume_from_step=-1`) starts its
own warmup normally. With warmup disabled, identical inputs preserve exact LR,
model tensors, and optimizer state; checkpoint JSON intentionally gains the
configuration field and is not expected to be byte-identical.

#### Conditioned-SFT batch prefetch

`sft_prefetch_batches=0` retains the original synchronous generator path. A
positive depth overlaps complete SFT batch construction with forward/backward
work. It reuses the dataloader's existing, tested
`ClosableBatchPrefetcher` for bounded FIFO, worker-error, and close/join
semantics. The SFT-specific layer has only three ownership concerns:

1. The ordinary `FeatureIndex` remains LRU-1 and schedules its single
   sorted-stem lookahead load after admitting the current feature shard (see
   "Conditioned-SFT shard prefetch and lazy documents" below).
2. One optimizer-step-local producer thread constructs complete SFT batches
   into a bounded FIFO. It owns a private `SFTState`; delivered batches publish
   deep snapshots to the training loop in FIFO order.
3. One private CUDA stream owns producer H2D/device materialization. The main
   thread consumes a batch only after waiting on its producer event and
   registering its CUDA tensors on the consumer stream. The
   checkpoint-authoritative `buffered_v2` snapshot is still updated only after
   the corresponding optimizer step succeeds.

The producer receives exactly the remaining accumulation batches in the
current optimizer step. It is joined before validation, checkpointing, or the
next step. This prevents speculative batches from changing committed cursor or
ordered-buffer state, and prevents the train producer from consuming the
selection/negative RNG stream across a validation boundary. The FIFO holds at
most the configured depth; one consumer batch and one completed producer item
blocked on a full queue can coexist with it. There is no persistent worker
pool, process transport, new cache, or artifact format.

Issue #1475 measured depth 2 on the exact 2xH200, B32, T1024,
accumulation-8 workload: it improved stable throughput by 8.53% on node2433
and by 3.58-4.57% against two synchronous bookends on node3400. Depth 8 was
slightly slower than depth 2. Keep the base default at `0`; use depth `2` as an
explicit, production-shaped optimization only after checking memory and
throughput on the target geometry. Do not interpret this knob as feature-shard
lookahead: the existing feature predictor was already hitting, with measured
transition resolution in tens of microseconds.

#### Conditioned-SFT shard prefetch and lazy documents

Issue #1504 measured `sft/refill_buffer` (zst decompress + per-document graph
build) at ~55% of `sft/data_loading`, itself ~78% of step time. Two changes in
`FeatureIndex` address it, both plain constructor arguments — no Hydra knobs:

- **`lazy_docs=True`** (opted into by `SFTFeatureStore`) replaces the eager
  `_build_docs` pass — which constructs a `SegmentFeatures` object and a tensor
  slice for every segment of the shard, ~1.8M per shard at production geometry —
  with one vectorized `(doc_id, segment_idx)` argsort at load time plus
  per-document materialization on lookup. Values, dtypes, storage independence,
  segment order, iteration order, `len`, and the absent-document `KeyError` all
  match the eager path exactly, and whole-shard consistency (the CSR span and
  the `feature_mags`/`feature_percentiles` lengths that `split_with_sizes`
  implicitly enforced) is checked at load time, not at access time, so a corrupt
  shard still fails where it used to. It supports exactly this configuration and
  rejects `shared_store`/`lazy_shared_docs`/`compact_feature_ids`. Documents are
  deliberately **not** memoized: memoizing rebuilds the whole eager object graph
  on top of the retained contiguous arrays and failed the task's RSS gate.
- **`prefetch=True`** (also opted into by `SFTFeatureStore`) loads the next
  sorted stem on one daemon thread while the current shard is consumed. One
  worker and one published-but-unconsumed shard map exist at a time: scheduling
  and consuming both join the previous worker first, and a miss empties the slot
  *before* the synchronous load, so a reposition can neither stack workers nor
  strand a decompressed shard map. Failures are logged on the worker at failure
  time and re-raised by the consumer's synchronous load.

**Memory model:** at most `max_resident_shards + 1` shards are held — the
resident set plus the single prefetch slot. With `lazy_docs` a resident shard
costs its contiguous arrays rather than the (larger, int64-upcast) per-segment
object graph; the clones handed to consumers are bounded by the conversation
buffer, which at production shard sizes is a negligible fraction of a shard.

One shard of lookahead is what the measurements support, not a placeholder for a
deeper pipeline: with lazy documents the load path is short enough that a single
in-flight shard already hides every steady-state boundary crossing (steady
`sft/refill_buffer` max falls from 268 ms — a blocking whole-shard build — to
7.9 ms, about twice p99, i.e. lookup cost), while depths 2 and 3 bought 1-2%
throughput for +100 MB and +260 MB of peak RSS per rank and were removed again
in the issue's simplification pass. Re-open the question only with fresh
`scripts/benchmark/benchmark_sft_data_pipeline.py` numbers on the target
geometry.

For an attribution run, enable the existing timing controls as needed:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m scripts.train.conditional_encdec_sft -- \
  config_name=conditional_encdec_sft_2xh200 \
  sft_prefetch_batches=0 \
  profile_timing=true \
  profile_trace=null \
  profile_output_dir=out/profiles/sft-control
```

`profile_timing` writes `sft_timing_rank<R>.csv`; its default CUDA
synchronization applies to main-thread regions, deliberately changes overlap,
and must not be used as the throughput arm. Prefetch-worker phase rows remain
unsynchronized CPU wall times. They carry rank/step/microstep tags and travel
with the batch, then merge only when that exact batch is delivered, so
abandoned lookahead cannot pollute the CSV. A Chrome trace is likewise
diagnostic. Compare throughput with separate fresh-process, tracing-disabled
synchronous and prefetched runs at identical model, artifact, batch,
world-size, attention, precision, and cache geometry.

Enable `profile_trace` separately and start with the smallest bounded capture
that answers the question. A trace-only run does not propagate the listener
context used by `timed_region` into the SFT prefetch thread, so its fine-grained
markers are absent. `profile_timing=true` installs the worker-local listener
and supplies those rows (and markers if both diagnostics are enabled), but the
synchronized timing mode changes overlap. The tracer retains shapes and Python
stacks: a four-optimizer-step capture of the issue-#1475 B32/accumulation-8
workload exhausted a 256-GiB cgroup while finalizing and produced no trace.
Synchronized timing is the lower-memory first choice.

#### Conditioned validation

Encoder-decoder SFT validation always runs with real SAE features and logs
`val/bpb`. `min_val_bpb` tracks that conditioned validation BPB.

The validation generator is single-pass: it walks its rank's slice of the
covered documents exactly once and pads instead of wrapping, so no document is
scored twice within one eval pass, and `eval_tokens` only ever bounds the pass
from above. The eval loop stops on a flag all-reduced across ranks
(`sync_last_step`, `ReduceOp.MIN`) rather than on each rank's own cursor:
best-fit packing pops a different number of conversations per rank, so the
loop must continue until every rank has exhausted its slice. A rank that
finishes early emits fully masked padded batches, which keep collective call
counts aligned while contributing neither nats nor bytes to BPB. Stopping when
the first rank finishes would silently omit the remaining ranks' tail batches;
stopping rank-locally would desynchronise the next collective. Every rank must
own at least one covered document; a validation conversion smaller than the
world size is rejected.

```bash
# Run conditioned validation during SFT.
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_sft -- \
    dataset=safety model_tag=d26 \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/safety_sft
```

#### Discrepancy audit helpers

The SFT supervision audit is useful when debugging capability regressions after
conditioned SFT:

```bash
# Quantify how much SFT supervision the conditioned path actually sees.
uv run python -m scripts.analysis.sft_supervision_audit \
    --dataset capabilities --max-examples 50000 --load-features

```

Current encoder-decoder SFT uses conditioned features for both training and
validation, so compare `min_val_bpb` as the conditioned validation BPB.

`sft_supervision_audit.py` writes
`out/analysis/sft_supervision_audit/{dataset}_with_features.md`.
It replays the SFT parquet rows through the nanochat tokenizer and reports how
many full-sequence supervised positions survive optional feature-coverage
filtering, truncation, and `min_loss_tokens`.

#### Overriding the SFT parquet directory

By default, `scripts/train/conditional_encdec_sft.py` reads train SFT parquets
from `{base_dir}/sft_data/{cfg.dataset}` and validation parquets from
`{base_dir}/sft_data/{cfg.dataset}_val` — the two layouts produced by
`scripts/preprocess/convert_sft_to_parquet.py split=both`. The
`sft_parquet_dir` and `val_sft_parquet_dir` config knobs (both default `null`)
let operators override them independently. `val_feature_dir=null` derives a
validation feature path by parsing the train `feature_dir`, replacing its
known train phase with the matching `*_sft_val` phase, and rebuilding through
`steering.feature_paths`:

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_sft -- \
    dataset=safety model_tag=d26 \
    feature_dir=/mnt/features/safety_sft \
    sft_parquet_dir=/mnt/parquets/safety \
    val_feature_dir=/mnt/features/safety_sft_val \
    val_sft_parquet_dir=/mnt/parquets/safety_val
```

When validation is present and `eval_every>0`, startup requires at least one
`shard_*.parquet` in the resolved val parquet directory and at least one
feature shard in the resolved val feature directory. There is no train-artifact
fallback. Set `eval_every=0` to opt out explicitly.

#### Gradient clipping

The SFT training loop uses the same shared clipping helper as decoder SFT. The `grad_clip_max_norm` config knob (default: `null`) controls the max norm:

- `null` (default) disables clipping deliberately. On startup `main()` emits an INFO line recording that clipping is disabled by default.
- A positive float enables clipping; for example, `5.0` loosens the clip for runs with more gradient headroom.
- `1.0` is the historical value chosen in commit [`edc101b`](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/commit/edc101b) as a NaN defense against bad SFT batches containing garbled U+FFFD tokens — `flex_attention`'s Triton fallback has known NaN issues in the cross-attention backward pass. Enable it if a run hits NaN losses.

Enabled clipping sets `error_if_nonfinite=True` and records the returned
pre-clipping norm. In distributed runs each rank clips its accumulated local
gradient before `DistMuonAdamW` performs gradient reduction.

```bash
# Loosen the clip for a run with more headroom.
torchrun --standalone --nproc_per_node=8 \
    -m scripts.train.conditional_encdec_sft -- \
    dataset=safety model_tag=d26 \
    grad_clip_max_norm=5.0
```

The `min_val_bpb` tracker and checkpoint selection logic are unaffected by this knob.

#### Current Limitations & Caveats

These are behaviors of the SFT script as it stands — not bugs per se, but things to be aware of when running it.

**1. FP8 training is not yet supported by the SFT script.** The pretraining script (`conditional_encdec_train.py`) calls `apply_fp8_training(model, recipe=cfg.fp8_recipe)` under `if cfg.fp8:`, but the SFT script has no equivalent code path. Conditioned SFT therefore does not inherit the pretraining-only FP8 surface, and its post-compose guard rejects attempts to append or inherit `fp8`/`fp8_recipe` instead of silently ignoring them. Adding FP8 support to SFT is a separate workstream tracked outside this remediation chain.

> The earlier "no `resume_from_step` for SFT" caveat has been fixed: see the **Resuming an interrupted SFT run** section above. The earlier "silent feature fallback in conditioned mode" caveat has been fixed: missing SFT feature rows and truncated feature coverage now fail before training. The earlier validation-mode caveat has been removed: SFT validation is conditioned-only. The earlier "SFT parquet dir is hard-coded" caveat has been fixed: see the **Overriding the SFT parquet directory** section above. The earlier "Gradient clipping threshold is hard-coded" caveat has been fixed: see the **Gradient clipping** section above. The earlier "No hardware-variant configs for SFT" caveat has been fixed: see the **SFT variants** table in the **Hardware-Specific Configs** section above.

### Harmfulness eval source-string convention

`scripts/eval/harmfulness_eval.py` records `config.source` from a CLI-provided
`source=...` argument verbatim into each YAML output. The operator chooses the
convention. For the `harmfulness_progression` plotting infrastructure to
consume the resulting YAMLs (forward or reverse), the convention used by the
existing `DEFAULT_*_SOURCE_MAP` constants in
`steering/eval/harmfulness_progression.py` must be followed:

| Lineage | Pretrain | First SFT phase | Chained second SFT phase |
|---|---|---|---|
| Forward cond | `base` | `base/safety_cond` | `base/safety_cond/capabilities_cond` |
| Reverse cond | `base` | `base/capabilities_cond` | `base/capabilities_cond/safety_cond` |

Drift (e.g. running a reverse-order eval with a forward-style `source` string)
causes the corresponding records to be silently dropped by the plot's source
map. The drift is caught by `tests/harmfulness_progression/integration/test_plot_reversed_e2e.py::test_e2e_source_drift_skips_record`.

## Checkpoints

### Verified remote c1024 run (issue #781)

`cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote` is the
issue-#781 selection-only preset.
`cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed` extends
it with issue-#806 train packed-row streaming and a separate
collision-resistant run/model tag. Presence of `remote_training` enables the
verified remote selection/checkpoint path and requires
`body_prefetch.lookahead_shards_per_rank`. The optional
`remote_training.train_packed` sub-block additionally enables train packed-row
streaming. `remote_training: null` takes the literal existing all-local loader
and checkpoint branches.

Startup rejects the mode unless all of the following are true:

- `use_precomputed_batches=true`, static clusters are active,
  `feature_delta.mode=splice_counterfactual`, and `max_seq_len=1024`;
- the launch may span nodes, but `LOCAL_WORLD_SIZE` must divide `WORLD_SIZE` and
  every rank must resolve the same cache and `flock` namespace;
- the derived validation and original/base validation packed families are
  complete locally. In selection-only mode the derived train packed family is
  also complete locally. In packed-streaming mode its authenticated manifest,
  signed inventory, and complete `provenance/` sidecars are pre-staged while
  its large direct body files may be absent;
- the exact matching selection `manifest.json` for each family exists in the
  configured HF bucket and has complete cluster/splice, mask, tokenizer,
  percentile, packed-provenance, and selection identities;
- each manifest carries a `remote_loader` certificate binding the selection
  checksum, ordered compressed-body inventory aggregate, packed-row
  fingerprint, and exact global positive-plus-negative CSR width. Every current
  clustered family uses certificate schema v2. Derived clustered-splice
  families also bind the canonical `cluster_delta_semantics`; the delta-free
  base-validation family omits that conditional key. Schema-v1 certificates
  are unsupported and must be reissued by the producer's ordinary validation
  path; this does not recompute selection outputs.

Selection bodies are lazy in both remote presets. In the issue-#781 preset,
packed-row bodies remain local. In the issue-#806 preset, TRAIN packed-row
bodies are lazy too; validation packed rows remain complete and local.

The producer writes `remote_packed_inventory.json` after final compact-FP
certification. Its signed schema binds the exact packed `manifest.json` bytes;
the sorted body names, byte lengths, plain SHA-256 digests, name-salted
fingerprints, and segment counts; the folded packed-row fingerprint; and the
complete local packed-provenance count/fingerprint. Publication also
cross-checks the train selection manifest's `remote_loader` and
`packed_provenance` contracts. Any legitimate manifest restamp invalidates the
inventory and requires republishing it.

At startup the streaming consumer authenticates that inventory, the manifest,
the compact-FP certificate, local `provenance/`, local splice sidecars, and all
three remote selection manifests. It also requires the derived train and
derived-validation manifests to agree on the producer-recorded derived-feature
fingerprint. No missing artifact falls back to a weaker path.

Remote packed mode intentionally does not stage the raw percentile overlay or
the complete derived feature-shard family: training and validation consume the
percentile-derived values already stored in packed rows, while probe percentile
metrics read the statistics table. It does not construct a global raw
`FeatureIndex`.

When `steering_eval_every=0`, no raw derived feature shard is required. When
clustered inline steering is enabled, startup instead fetches the authoritative
`remote_inline_eval_source.json` certificate and requires exactly its one
validation raw shard, token-boundary leaf, and parquet. The producer creates
that certificate only after validating the complete raw family against the
unchanged schema-v2 token manifest. It binds the full derived-feature
fingerprint and global token-overlay base hash plus the exact size/SHA-256 of
the staged triple. The consumer cross-binds those identities to the remote
train/derived-validation selection contracts, rejects extra raw or token-leaf
bodies, and constructs a one-shard lookup index without calling global overlay
discovery. The decoded shard is evicted after the immutable clustered steering
cache is built; cadence-time evaluation does not reread raw data. The small
authenticated catalog is retained so a later config in the in-process fast
sweep can rebuild its own cache from the same certified bytes.

Startup still loads the full percentile and tokenizer checksums from feature
statistics and requires exact equality with the combined checksum and the
separate `metadata.pct_checksum` / `metadata.tok_checksum` fields in each train,
derived-validation, and base-validation packed-row manifest. This full
manifest check is required because the directory layout contains only
16-character checksum prefixes. Local, live, and unclustered modes retain the
existing complete-family raw-overlay base-shard integrity check unchanged. The
sparse certificate does not weaken, rewrite, or reinterpret the schema-v2
overlay hash.

Issue #853 assigns signed inventory index `i` to transport owner
`i % world_size`. Each DDP rank starts exactly one spawned, non-daemon owner
process with its own `HfBucketBackend`. Paired train selection and packed
bodies share the owner. Every rank still consumes the bodies needed by its
unchanged row stripe; training consumers have no backend, download,
publication, adoption, or ownership-takeover path.

Workers combine every rank's exact resume cursor and requester-specific
demands, finalize the nearest configured owner obligations ahead of demand,
and publish selection before packed. A pair-ready record appears only after
both final bodies and publication receipts exist. Validation selection uses
the same workers and requests its startup window explicitly. Retry exhaustion,
publication failure, or worker exit is sticky and aborts the run; no non-owner
fallback exists.

Every call to the parent failure controller still detects an existing sticky
failure immediately. New local worker-pipe errors and unexpected owned-process
exits are sampled immediately on the first call and then no more than once per
monotonic second across the training and remote CPU batch-producer threads.
Complete shared-filesystem status sweeps across all ranks are separately
limited to one successful sweep per ten monotonic seconds, with an immediate
initial sweep. The threads share one nonblocking sweep lock, so neither
duplicates a due scan nor waits behind slow peer-record I/O. A newly published
local failure is therefore visible within one second plus normal scheduling; a
newly published peer error is visible within ten seconds plus the sweep
duration and normal scheduling. Failed local checks and invalid peer control
state remain fail-loud and immediately retryable because they do not advance
their successful-check timestamps. The general 50 ms transport/cancellation
polling cadence is unchanged.

Finalized bodies are opened once with `O_NOFOLLOW|O_NONBLOCK`, classified with
`fstat`, held under the shared body lock, and handed to the existing decoder as
immutable compressed bytes. A fresh publication receipt binds the signed
family/index metadata, owner, generation, and run nonce. The owner trusts a
successful HF/Xet download and performs no post-download content reread before
the one atomic final rename. Strict mode hashes every consumer read. Trusted
mode relies on transport completion and decode; pre-start warm bodies retain
the explicit existing-cache trust boundary.

Before any packed row is consumed, the loader still checks the body's segment
count and paired local provenance tensors. Resume uses the same inventory order
and exact saved row state as the all-local loader. Certified selection width
still supplies `pad_F`, preserving all-local tensor shapes and batch bytes.
Startup/session locking, publication receipts, cancellation, owner-only
eviction, disk admission, and acceptance gates are documented in
[Strict Rank-Striped Remote Body Prefetch](remote-cache-materialization.md).

Every remote checkpoint is a group containing exactly `model_<step>.pt`,
`meta_<step>.json`, and `optim_<step>_rank<R>.pt` for every DDP rank. All ranks
finish their atomic local saves and exchange failures first. With the shipped
`remote_training.async_checkpoint_upload=true` default, rank 0 pins the exact
direct-file identities and enqueues one background publication; every rank may
then continue training. The worker hashes the stable files, writes a local
marker, uploads and verifies every data file, and uploads/verifies
`remote_complete_<step>.json` last. The marker binds step, world size, the
immutable checkpoint lineage, the semantic remote-run fingerprint, the exact
filename set, and per-file
size/SHA-256/Xet identities. It is the sole commit authority. New data objects
live at `_groups/<marker aggregate_sha256>/<logical filename>` beneath the
checkpoint leaf; the canonical marker remains at the leaf root. The marker
schema stays version 1 and local checkpoint filenames and bytes are unchanged.
Before each publication, the checkpoint-leaf walk is O(1) in remote calls with
warm local marker mirrors: marker discovery and per-marker group validation use
one recursive listing, while schema-v2 lineage resolution retains its constant
authenticated-metadata tail.

Only one group may be pending. While it is outstanding, all ranks perform one
nonblocking rank-zero status broadcast at the top of each training iteration.
A completed failure is therefore observed by every rank at the same control
point. Before another local checkpoint save, ranks wait with short repeated
polls rather than holding one collective open across the upload. The final
checkpoint is drained before the run reports success. Exceptional teardown
does not call collectives: rank 0 joins its worker locally, preserving any
already-active training error while recording a simultaneous publication
failure. Set `remote_training.async_checkpoint_upload=false` to use the
unchanged synchronous rank-zero publication branch.

The pinned identity includes device, inode, mode, size, timestamps, and link
count; an input with multiple existing hard links is rejected. Identity is
rechecked around hashing, every data-file upload, and after marker verification.
Replacement, unlink, truncation, hardlinking, or other mutation before worker
completion therefore fails the run. A mutation during marker upload can be
detected only after that marker has linearized remotely; in that case the
remote group is still a valid, marker-bound snapshot and remains resumable, but
the publication failure is not suppressed. There is no checkpoint-retention
path in this trainer; step-specific files remain present. Upload failure leaves
the local checkpoint usable and partial data-file debris uncommitted.
On a plain relaunch, an absent remote marker makes any same-step local marker
non-authoritative: it is atomically reconciled to the newly pinned group, whose
aggregate-derived paths cannot collide with different interrupted bytes.
Exact matching partial generation objects are reused. Legacy flat debris is
never overwritten or deleted, and other unreferenced generations remain
harmless evidence for a separate future garbage collector. Re-publication is
idempotent for the identical complete group; a different committed same-step
marker remains a hard collision detected before local-marker mutation or data
upload. The unchanged single-writer checkpoint-leaf requirement still applies;
simultaneous independent writers are unsupported. The async setting is runtime/W&B
provenance but is stripped from checkpoint `user_config`, so sync and async
modes preserve identical checkpoint bytes and resume identity.

With `resume_latest=true`, rank 0 considers completion markers only, validates
every marker it finds, and restores the highest complete matching group through
temporary files. Marker discovery and per-marker group validation cost one
recursive listing. Marker payloads are re-validated from local
`remote_complete_*.json` mirrors only when their exact size and Xet identity
match that listing, proving byte identity; any mirror miss uses a verified
download. Those authenticated bytes are adopted as a mirror only when the local
path was absent. The walk never replaces an existing stale or malformed local
marker: publish/restore reconciliation still owns that replacement, after every
local data file verifies. Every marked group's completeness and object identity
are checked against the same listing snapshot instead of later per-object stats;
the unchanged single-writer checkpoint-leaf requirement makes that snapshot
authoritative.
Listing records must be unique per path and carry well-formed Xet identities;
conflicting duplicates or malformed identities fail the walk loudly.

Readers probe the complete immutable generation named by the authenticated
marker first and fall back to the complete legacy flat group only when no
generation entry exists. A partial or drifted generation is corruption: it
cannot mix with or fall back to flat objects. Higher unmarked debris is ignored;
a malformed marker or an incomplete marked group is corruption and fails rather
than falling back. A conflicting local data file is not overwritten. After
every local data file verifies, a stale or malformed local marker is atomically
replaced with the authenticated remote marker. Fixed code therefore reads old
flat and new generation checkpoints, while pre-fix binaries cannot read newly
committed generation-layout checkpoints. Do not roll a checkpoint leaf back to
pre-fix code after its first generation commit. The checkpoint metadata's
lineage and semantic identity are checked before model or optimizer mutation.

Fingerprint schema v2 separates semantic identity from that lineage. On a
legacy marker mismatch, rank 0 first parses every marker, verifies world size
and the complete signed remote group, then downloads only the highest marker's
authenticated metadata object. It requires marker/meta lineage agreement,
reconstructs the byte-pinned v1 identity (including the historical omitted
`initialize_encoder_embeddings_cluster_vectors=mean` default), and separately
canonicalizes the saved config under exhaustive v2 against the currently
authenticated selection contracts. Only then does it adopt the old lineage.
A migrated checkpoint with v2 metadata must match its stored v2 record, saved
config, and current identity. Mixed lineages, artifact drift, semantic drift,
malformed metadata, and unknown schemas fail before model assignment or
optimizer loading. Existing markers and production checkpoints are never
rewritten.

Device-batch migration is accepted only when the resolved `total_batch_size`
remains divisible by `device_batch_size * max_seq_len * world_size`; this is
checked before remote restoration. Exact `batch_rows` become an ordered
unconsumed rank-local stream and are regrouped into the new width, continuing
through shard and epoch boundaries. Cursor-only legacy states cannot migrate.
Same-B replay retains its exact path, and every migrated optimizer step consumes
the same ordered rows as the fixed-global-batch reference.

For permitted nonzero `label_dropout`, the saved width becomes
`label_dropout_draw_batch_size`. A dedicated checkpointed generator continues
drawing in that canonical grouping and slices or concatenates masks for the
execution width; its group offset is persisted too. Per-row segment-drop masks
therefore remain exact across repeated B changes. CUDA host staging still
requires `label_dropout=0`.

Cadence-only migration is trajectory-exact, same-B resume remains exact, and B
migration is row/mask exact but floating-point approximate. FP8 mode and recipe
remain semantic and cannot change on resume.

The trainer suppresses the otherwise redundant save at the initial resumed
step;
independent `torch.save` containers are not a byte-stable oracle. Exact bytes
are required only between one locally produced file and its uploaded then
redownloaded copy. Independent local/remote-selection runs compare deserialized
model, optimizer, metadata, RNG, and dataloader state.

Launch the selection-only preset after the three certified c1024 families have
been published and packed rows have been staged locally:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.conditional_encdec_train -- \
  config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote \
  feature_dir=<derived-c1024-feature-dir>
```

Append `remote_training.async_checkpoint_upload=false` to either remote launch
to run the synchronous publication path.

Credentials use standard Hugging Face token discovery; never put a token in
Hydra config or checkpoint metadata. The selection-only path closes #781; the
optional packed-row path implements #806. Production throughput and real-bucket
smokes remain explicit owner-gated acceptance work.

#### Remote TRAIN packed-row publication and launch (issue #806)

Use paths relative to the `features` category root (`out/data/features`) in
the `hf_push`/`hf_pull` include lists below. The producer machine must have the
complete train packed family. Existing remote training also needs a completed
selection family, while the remote pre-selection workflow can create that
family afterward. Certify the compact-FP source first, then build the signed
schema-v2 packed inventory; the inventory itself is selection-independent.
Pass `selection_checksum=<64-hex>` only to cross-check an already completed
selection against the same packed provenance and loader certificate.

```bash
TRAIN_PACKED=<complete-derived-train-packed-dir>

uv run python -m scripts.preprocess.validate_fp_target_source \
  packed_row_dir="$TRAIN_PACKED" n_sae_features=65536
uv run python -m scripts.preprocess.publish_remote_packed_inventory \
  packed_row_dir="$TRAIN_PACKED"

# Required only when steering_eval_every > 0. VAL_STEM must be the one
# derived-validation provenance source consumed by clustered inline steering.
uv run python -m scripts.preprocess.publish_remote_inline_eval_source \
  feature_dir=<complete-derived-feature-dir> \
  data_dir=<derived-parquet-dir> \
  stem=<VAL_STEM> \
  fingerprint_cache_dir="$TRAIN_PACKED"

# Preview, then additively publish the complete authenticated train family.
uv run python -m scripts.utils.hf_push category=features dry_run=true \
  +include='["<derived-rel>/remote_inline_eval_source.json","<train-packed-rel>/manifest.json","<train-packed-rel>/remote_packed_inventory.json","<train-packed-rel>/*.safetensors.zst","<train-packed-rel>/provenance/**","<train-packed-rel>/sel/**"]'
uv run python -m scripts.utils.hf_push category=features \
  +include='["<derived-rel>/remote_inline_eval_source.json","<train-packed-rel>/manifest.json","<train-packed-rel>/remote_packed_inventory.json","<train-packed-rel>/*.safetensors.zst","<train-packed-rel>/provenance/**","<train-packed-rel>/sel/**"]'
```

On the training machine, pre-stage exactly the small train lineage inputs and
the two complete local validation families. Replace every placeholder with a
path relative to `out/data/features`; `<VAL_STEM>` is the exact stem certified
above. The train packed include intentionally omits direct
`*.safetensors.zst` bodies. The four `<VAL_STEM>` source entries are required
only when `steering_eval_every > 0`; omit them when inline steering is disabled.

```bash
uv run python -m scripts.utils.hf_pull category=features \
  +include='["<derived-rel>/feature_statistics.safetensors","<derived-rel>/remote_inline_eval_source.json","<derived-rel>/<VAL_STEM>_features.safetensors.zst","<derived-rel>/tok/<tok-hash16>/seg_boundaries/manifest.json","<derived-rel>/tok/<tok-hash16>/seg_boundaries/<VAL_STEM>.safetensors.zst","<derived-rel>/_parquets/<VAL_STEM>.parquet","<derived-rel>/_splice_provenance/**","<train-packed-rel>/manifest.json","<train-packed-rel>/remote_packed_inventory.json","<train-packed-rel>/provenance/**","<derived-val-packed-rel>/**","<base-rel>/feature_statistics.safetensors","<base-val-packed-rel>/**"]'
```

The resulting local checklist is:

- derived directory: statistics and complete `_splice_provenance/`; with
  inline steering, also the authoritative certificate, its exact validation
  raw shard, its exact token manifest/leaf, and its exact parquet. No unrelated
  raw shard or token leaf may be staged;
- derived train packed directory: `manifest.json`,
  `remote_packed_inventory.json`, and complete `provenance/`, with direct
  packed bodies absent;
- derived `val_sall`: complete packed bodies, manifest, provenance, and
  matching selection family;
- base directory: statistics; base `val_sall`: complete packed bodies,
  manifest, provenance, and matching delta-free selection family.

Launch with the generic preset:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.conditional_encdec_train -- \
  config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed \
  feature_dir=<derived-c1024-feature-dir>
```

For the exact eight-H100 production geometry, use the target preset shown in
the staging section below. It removes the equivalent `feature_dir`,
`device_batch_size=16`, `total_batch_size=524288`, `fp8=false`, and
`steering_eval_every=0` command-line overrides without changing their resolved
values or any packed, selection, mask, cluster, run, model-tag, checkpoint, or
remote-run fingerprint identity.

The shipped cache window is 24 shards and can be overridden with
`remote_training.body_prefetch.cache_shards=<N>`, where `N` is a strict integer
of at least 2. It controls trailing finalized selection and packed retention,
not the `body_prefetch.lookahead_shards_per_rank=2` owner frontier. The preset also
sets `remote_training.train_packed.trust_existing_cache=true`: the owner trusts
successful HF/Xet downloads and atomically publishes them without reopening;
the consumer skips hashing after the publication receipt is present. Stable bodies captured before startup keep the
documented direct-file/single-link/signed-size trust boundary. The observed descriptor
signature is mode, device, inode, size, `mtime_ns`, `ctime_ns`, and link count;
any change in that complete before/after signature restores full content
verification. An unchanged signature cannot prove byte identity, so same-length
bad bytes already named when the descriptor opens or same-length in-place
mutation during the read may be accepted. Only strict mode provides per-read
content authentication; override the option to `false` to hash every cache
read. On the owner's GPU/storage box, the final
operational gate is a real L22 cold-cache smoke, warm-cache throughput and
training-step impact measurement, followed by the mandatory two-rank
real-bucket checkpoint/resume gate below. Those measurements are owner-gated
and are not inferred from the fake-bucket correctness tests.

#### Local packed-row batch lookahead

Ordinary local packed-row training uses the existing complete-batch FIFO. By
default its depth remains `grad_accum_steps`, preserving historical behavior.
An optional strict non-negative `train_prefetch_batches` overrides that depth;
zero leaves the loader synchronous. The setting changes execution overlap only:
it is omitted from checkpoint/resume and remote semantic identity, and exact
batch/state ordering remains the dataloader contract.

The exact `full3b_clustered_c2048` presets use depth 64. Their packed artifacts
contain nominal 16,384-row output shards, while B16, c2048, world size 32, and a
1,048,576-token global batch consume 512 rows per optimizer step. Each shard is
therefore 32 optimizer steps and 32 rank-local batches. A one-step default would
shrink to one batch of decode lead; depth 64 gives each producer up to two
nominal shards of decode lead. This path retains the FIFO's CUDA event handoff
rather than introducing the remote staging stack.

#### Shard-level lookahead (`train_shard_lookahead`)

Complete-batch prefetch cannot absorb the observed packed-shard storage stalls for two structural
reasons. Its GPU-resident FIFO is capped in practice at about 24 batches, or roughly 17 seconds, by
the dense `fp_targets` payload (about 84 MB/batch), and it starts refilling only while that finite
queue drains. `train_shard_lookahead` instead banks decoded base/selection pairs in host or shared
memory. Loading shard `N+k` begins as soon as `N+k-1` finishes decoding, independent of batch
materialization, so the queue carries storage supply across the entire consumption time of the
intervening shards. The shipped `full3b_clustered_c2048*` presets use depth 4. For the FP8
B16/accumulation-1 base that is 128 optimizer steps of supply (32 steps per shard per rank at world
32); its step time is unmeasured, so no wall-clock figure is quoted. For the BF16
B14/accumulation-2 child presets it is about 103 seconds, from the measured world-31 1.41 seconds per
step over ~18.3 steps per shard.

One traversal-plan iterator remains the sole authority for resume filtering, exact continuation,
epoch rollover, and log order. The background producer performs only existing packed/selection I/O;
all reconstruction, selection RNG, label dropout, CUDA work, batch-state construction, and yielding
remain ordered on the consuming thread. Depth zero is the historical inline path. Positive depths
are local-packed-train-only: either packed or selection materialization raises because
`remote_training.body_prefetch` already stages remote bodies. Validation, evaluation, and
self-conditioning loaders deliberately remain at zero because they abandon bare generators after a
bounded evaluation window.

The producer is a daemon thread with a depth-bounded queue. Exceptions occupy their traversal
position in that queue and are re-raised only after the complete preceding batch prefix. Closing the
loader sets a stop event, drains queued shard references, and joins for at most five seconds; a load
still blocked at that deadline warns without masking an original exception. During initial fill an
attacher may transiently outrun its local owner; after the existing 120-second attach timeout it
warns and direct-decodes canonically, then producers reconverge as their queues fill.

| Depth | Banked pairs | Live decoded shards (`K + 3`) | Shared retain floor (`max(4, 2 * (K + 2))`) | Approximate node memory at 4.3 GiB/shard |
|------:|-------------:|------------------------------:|-----------------------------------------------:|------------------------------------------:|
| 4 | 4 | 7 | 12 | 30 GiB live / 52 GiB retained |

The live budget includes `K` queued pairs, one pair held by the producer while it blocks on a full
queue, the visit being consumed, and one predecessor still referenced by cross-shard pending rows.
The retain value is a monotonic process-global floor, so a depth-zero validation loader cannot shrink
the train loader's safety window. At depth 4 the B14 child geometry has a maximum local-rank producer
skew of `K + 3 = 7`; the B16 base FIFO can span three shards and reaches `K + 4 = 8`. The retained 12
names cover both cases, and process residency is bounded by the largest lookahead depth used in that
process.

Enabling or raising the depth takes effect on restart. It changes neither checkpoint bytes nor
resume identity, so an existing checkpoint may resume at any depth without migration.

#### Producer side stream and per-node shared decode (issue #1507)

Two always-on, bitwise-identical producer changes (measured on the world-32
matrix, ~5% step-time each, additive):

- The prefetch producer thread issues its CUDA work (batch `.to(device)`
  copies) on a dedicated side stream. On the default stream every produced
  batch waits behind the entire enqueued training step, capping the producer
  at about one batch per step; the FIFO's event record/wait pair plus a
  consumer-side `record_stream` keep cross-stream ordering and allocator
  safety.
- Under torchrun with more than one local rank, exactly one local rank per
  shard (stable hash of its lexical absolute path and epoch) decodes it and
  publishes the eight `PackedRowShard` tensors into a POSIX shared-memory
  segment; sibling ranks attach zero-copy via `t.frombuffer`, cutting per-node
  NFS reads and zstd decode 8x. Different lexical aliases can select different
  ephemeral names, but sibling ranks receive the same inventory path and the
  segment namespace is launch-local.

  The owner's packed-body open/read and zstd decode run in a disposable tiny
  subprocess with one 900-second storage-stage deadline and no retry. This
  owner deadline is independent of the consumer's 120-second attach wait. The
  deadline starts before process creation and, after `Popen` returns, covers
  worker startup, symlink traversal in `open`, all compressed reads, decode,
  complete decoded-byte transfer, and worker exit. Python cannot preempt the
  synchronous `Popen` constructor/exec handoff; in-parent safetensors parsing,
  `PackedRowShard` construction, shared-memory allocation, and publication are
  also outside this storage deadline and do not read the packed NFS body. A
  timeout or worker/decode failure occurs before new segment creation, so no
  partial magic header is publishable.

  Publication still commits the magic header last. Attachers poll for 120
  seconds, then warn and direct-decode canonically rather than failing; this
  preserves legitimate divergent local resume stripes. Exact-resume pending
  rows and this consumer fallback remain canonical direct loads and are not
  subprocess-bounded. Eviction still unlinks names only — a mapping lives as
  long as a shard references it — with a monotonic process-global retain floor of
  `max(4, 2 * (train_shard_lookahead + 2))` shards (4 until a positive depth raises it).
  Materializer-backed remote paths keep their existing byte-streaming behavior.

#### Loader-health attribution and gate summaries (issue #1569)

Packed validation wraps loader construction and BPB evaluation in same-thread
health accounting, then emits `loader health eval:` with step, split, rank,
outcome, counts, means, and maxima. This excludes concurrent TRAIN prefetch or
lookahead records. Failures log completed loads with `outcome=error` before
propagating; live `FeatureIndex` validation emits no misleading packed-I/O line.

Each completed conditioned-eval and checkpoint-save gate also emits process-local
`loader health gate:` counts and duration means since the previous gate,
including background-thread loads. It omits maxima because cumulative maxima
cannot be subtracted into interval maxima. These loader-health paths add no
collective, config, W&B field, state, RNG draw, checkpoint byte, or identity change.

Remote saves report gathered duration skew only after `ddp_collective_result`
confirms every local save succeeded, so a failed rank cannot strand survivors in
the telemetry collective. Publication follows that report; the checkpoint
manager's existing per-rank timing line remains.

#### Persistent local validation setup (issue #1615)

Local, precomputed, selection-backed conditioned-BPB validation retains one
immutable setup per validation split and process. The first gate still performs
the complete canonical packed/selection validation. It publishes the retained
setup only after a before/after fence proves that every present manifest and
every direct packed, selection, and packed-provenance shard body stayed
unchanged during validation. A failure or interruption before that fence
publishes nothing; later model/traversal failure cannot advance the next
gate's fresh cursor.

Every later gate constructs a new loader generator and therefore a new epoch,
pending-row list, RNG replay state, decoded-shard state, and rank-local cursor.
Only immutable shard discovery, validated static dimensions, and artifact
authority are reused. Before reuse, the loader recomputes the exact request
identity, reads and SHA-256-hashes the required selection manifest and the
packed manifest when present, checks the exact direct packed, selection, and
packed-provenance shard inventories and every body's link/target stat identity,
and checks the large derived-feature and sidecar directories. Unrelated
cache-file publication in the packed directory does not invalidate the retained
setup. Drift raises before traversal and requires a restart so the complete
validator runs again; the cache never silently refreshes itself from partially
changed input.

This intentionally changes the process-lifetime artifact trust boundary. Later
gates do not content-hash packed/selection/provenance bodies again, and they do
not repeat the per-file stat inventory over every already-certified
derived-feature and splice-sidecar body. Thus an in-place derived/sidecar
mutation that leaves its parent directory unchanged, or a direct-body mutation
that preserves every checked link/target stat field, is detected by a fresh
process but can evade a running process. Manifest bytes are not covered by that
relaxation: every manifest present during canonical validation is
content-hashed at every gate, and any later filesystem object at a packed
manifest path that was initially absent is rejected. Treat all validated
bodies as immutable for the life of the trainer.

The reuse is internal and always on for this one validation path; it adds no
Hydra setting, checkpoint field, run identity, RNG draw, or collective. The
trainer never supplies the retained setup to live, fused-selection,
remote/materialized, or training loaders, and the low-level seam rejects
resume, feature-prediction-target, materialized, and lookahead variants. The
node-local compressed-byte cache remains compatible because it affects body
loading after each fresh gate cursor is built.

#### Node-local shard cache

Local packed-row training can cache the exact compressed packed and selection
shard bytes on node-local storage. Set `train_node_cache_dir` together with the
required positive `train_node_cache_max_bytes` and required non-null,
non-negative `train_node_cache_reserve_bytes`; the default reserve of
`3000000000000` refuses any fill that would leave less than 3 TB free. All three
settings are execution-only: W&B retains the resolved operational snapshot,
while checkpoints and resume identity omit them. Validation loaders share the
same cache object as the train loader.

The cache has no eviction. Each admitted source is keyed by its lexical
absolute path and committed as a payload plus a stat-bound SHA-256 sidecar. A
hit rechecks source size and mtime and verifies the local payload hash before
decoding. Budget or reserve pressure therefore yields a stable first-`K`
policy: the first `K` shards by fill order remain cached, and every later shard
continues to stream from its canonical source. `train_node_cache_max_bytes`
accounts for every regular file already in the cache directory: payloads,
sidecars, and live temporaries. Admission pre-counts only the new payload; its
roughly 300-byte sidecar first enters the total at the next admission, which
self-corrects subsequent decisions. Concurrent fills can transiently overshoot
by up to the in-flight set before commits become visible. The exact-resume worst
case is one base shard per local rank, about `n_local_ranks x 1.5 GB`. Cache failures after
startup warn and bypass; failure to create the configured directory at startup
raises because the operator explicitly demanded the cache.

Each shared-decode owner attempt preserves a bounded storage stage. A failed
cache-hit attempt grants the one canonical-source retry a fresh deadline; local
cache admission and the parent's post-worker hash, fsync, and rename run outside
that deadline on local storage. On a miss, the existing lightweight zstd worker
tees the compressed source chunks into a
unique local temporary while continuing to stream decoded bytes to its
training-critical stdout. Tee I/O errors never stop decoding; a stalled,
non-erroring local write can still consume the worker deadline, an accepted
local-ext4 risk. The parent commits the local temporary only after successful
decode and an exact compressed-size check; on a hit, that same worker reads the
verified NVMe payload instead of the network source. Sibling shared-memory
attachment and its existing 120-second fallback protocol are unchanged, and
exact-resume and fallback direct decodes use the same compressed-byte cache.
At cache startup, tee temporaries with dead PIDs are swept immediately. Every
temporary older than 24 hours is also swept, including parseable names whose PID
appears alive, which bounds debris retained by PID reuse; fresher malformed names
are preserved until that age ceiling.

`/scratch` is operationally shared, persistent, and site-managed rather than a
private application cache. Its cleanup behavior is not documented as a stable
contract, so launchers should sweep the configured cache directory between
unrelated runs when space reclamation is required. Do not point the cache at
quota-limited network scratch, and preserve the reserve guard for other users
and jobs. Materializer-backed remote training rejects `train_node_cache_*`
because its own body cache already owns staging and retention.


#### Optional packed-batch CUDA staging (issue #842)

`remote_training.train_packed.cuda_staging` opt-in adds a byte-bounded pageable
CPU lookahead FIFO followed by reusable pinned source slots and a dedicated
rank-local CUDA transfer stream. It is intentionally absent from the generic
remote-packed preset. When absent, remote runs still decode on the explicitly
closable producer thread rather than the training thread, but its
`grad_accum_steps` queue is too shallow to hide the measured H100 decode tail.

The strict block requires `host_queue_bytes` and `pinned_slots`. Startup rejects
local/fused/live/validation use, unindexed CUDA devices, dynamic dimensions,
nonzero label dropout, or dropout RNG metadata in resumed loader state. The
consumer waits on the transfer event and records every CUDA tensor on its
actual stream before the ordinary training loop receives the batch; pinned
sources are not reusable before H2D completion. The operational block remains
excluded from the remote-run semantic fingerprint, so legacy and staged
checkpoints share resume identity when every authenticated artifact and
semantic training setting is unchanged.

The full ownership model, exact batch schema, lifecycle behavior, tuning
instructions, and single-/two-GPU parity gates are documented in
[Precomputed Packed CUDA Staging](precomputed-packed-prefetch.md).

Issue #868 enables that existing mechanism in the target-specific
`conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100` preset. It
pins the existing production feature root, B=16/BF16, the 524,288-token global
batch, and disabled steering eval while retaining the parent `run` and
`model_tag`. Its exact launch is:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m scripts.train.conditional_encdec_train -- \
  config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100
```

The 512 MiB pageable queue admits 211 canonical 2,539,776-byte batches, while a
16,384-row shard contributes 128 rank-local batches. The producer can therefore
enter N+1 with the full 32-step current stripe still consumable instead of the
legacy four-batch/one-step lead. Body workers continue to own only compressed
transport bytes; the loader owns transient decompression/current decoded
shards; the pageable queue, three pinned leases, and ready CUDA batches have
their own explicit bounds. Issue-#826 verification advances with the same
producer and remains mandatory.

Decode and selection validation failures stay fail-loud and retain their
original exception type/traceback, with an added note naming the exact body path
and logical epoch. Issue #867 peer-status polling remains an independent
control-plane cost; this issue-#868 change is stacked on PR #870, which supplies
the polling fix. Both benchmark arms must use that same stacked revision. The
approximately 710 ms H100 step time is still a projection until the
staged/unstaged production benchmark matrix passes.

#### L22 remote-packed production-host runbook

This is the concrete, fresh-machine procedure for the current clustered
Gemma-3-4B-PT L22 splice-counterfactual c1024 training run. It is deliberately
specific to `conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed`; do
not treat it as a generic object-store training recipe.

1. Prepare a persistent local checkout and cache. The host needs a current
   checkout of the intended `main` revision, a completed `uv` environment, a
   writable persistent `$NANOCHAT_BASE_DIR`, and credentials that can read and
   write `d0rb/nano-gpt-pretrain-steer-artifacts`. Keep secrets out of Git and
   Hydra; a mode-600 `.env` is one suitable host-local mechanism.

   ```bash
   cd /workspace/nano-gpt-pretrain-steer
   git fetch origin
   git checkout --detach origin/main
   uv sync

   cat > .env <<'EOF'
   HF_TOKEN=<read-write Hugging Face token>
   WANDB_API_KEY=<optional W&B API key>
   NANOCHAT_BASE_DIR=/workspace/nanochat_cache
   EOF
   chmod 600 .env
   set -a && . ./.env && set +a
   uv run python -c 'import torch; print(torch.cuda.device_count(), torch.__version__)'
   git rev-parse HEAD
   ```

   The token must also be able to fetch the Gemma tokenizer and the
   GemmaScope-2 SAE used by the config. Record the printed commit in the launch
   log. Do not repair a dependency/import failure by editing the vendored
   `nanochat/` tree or by making a one-off site-packages change; fix or pin the
   host environment, then rerun the import check.

2. Pre-stage the small local contract. Large *train* packed bodies and train
   selection bodies remain remote and are fetched on demand, but the following
   local paths are required before launch:

   | Local family | Required contents |
   |---|---|
   | Derived L22 feature root | `feature_statistics.safetensors`, the exact first `*_features.safetensors.zst` named by the train manifest, and all `_splice_provenance/` sidecars. |
   | Derived train `c1024/sall` | `manifest.json`, `remote_packed_inventory.json`, and all `provenance/` files; no direct `shard_*.safetensors.zst` train bodies are required. |
   | Derived validation `c1024/val_sall` | Complete packed bodies, `manifest.json`, `provenance/`, and its matching splice-counterfactual `sel/<hash>/` family. |
   | Base validation `c1024/val_sall` | Complete packed bodies, `manifest.json`, `provenance/`, and its matching delta-free `sel/<hash>/` family. |
   | Run controls | The manual whitelist, schema-v1 cluster artifact, and the pinned eval-context cache named by the selected config. |

   For this L22 run, the derived feature root is
   `out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/seed42`.
   Its train packed leaf is
   `pct/623c372d3f121b18/tok/dc6308e6116cbfe0/c1024/sall`; the derived and
   base validation leaves are the corresponding `c1024/val_sall` paths. Use
   the authenticated train `manifest.json` to obtain the exact first derived
   stem rather than guessing it. The publication command above is the source
   of truth for the bucket contents; use a narrow `hf_pull` include list rather
   than downloading all 584 train bodies.

   If a validation body must be copied from another host rather than pulled,
   copy it to a unique hidden staging name, compare its SHA-256 to the source,
   and only then atomically rename it to `shard_<N>.safetensors.zst`. Never
   stream directly into a final packed-body filename: an interrupted copy is
   correctly rejected later as packed-row fingerprint drift, but wastes a full
   distributed startup.

   The host also needs enough local disk for the two validation families, the
   support inputs above, checkpoints, and the admitted remote cache. Do not
   estimate this as `2 * cache_shards - 1`: a paused rank can delay owner
   eviction while all signed packed bodies become durable. Startup calculates
   the byte peak from the complete signed inventories, lookahead stages,
   active transports, receipts/control data, and its safety reserve before
   spawning workers.

3. Run the metadata/provenance preflight before consuming GPUs. The remote preset
   authenticates the three remote selection manifests, train inventory,
   compact-FP certificate, local train provenance, local splice provenance,
   validation families, selection/cluster/mask identities, and the exact
   percentile/tokenizer checksums. Existing train packed/selection bodies are
   the documented exception: the shipped `trust_existing_cache=true` setting
   accepts an already-present descriptor with an unchanged observed signature
   after cheap direct-file, single-link, signed-size checks. Fresh bodies and
   any change in the observed before/after descriptor signature remain fully
   hashed; an unchanged signature does not authenticate content. A successful
   log contains
   `remote_training startup contract verified` with a run fingerprint plus the
   trusted-cache warning. Never remove an artifact or bypass another failed
   identity check to make startup pass.

4. Pass the two-rank H100 checkpoint/resume gate before an eight-rank launch.
   Use an empty train-body cache and choose a literal gate-specific run/model
   tag that is absent from both local and remote checkpoint namespaces. Record
   that literal and reuse it unchanged for the resume; do not reuse the example
   tag below without first proving it is unused. Use the production microbatch
   of 8 rows per rank, `num_iterations=80`, and `save_every=40`. At two ranks
   the fixed 524,288-token global batch gives 32 accumulation microsteps, so 80
   optimizer steps cross multiple packed and selection body boundaries.

   ```bash
   cd /workspace/nano-gpt-pretrain-steer
   set -a && . ./.env && set +a
   nohup torchrun --standalone --nproc_per_node=2 \
     -m scripts.train.conditional_encdec_train \
     config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed \
     feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/seed42 \
     run=issue831_2xh100_remote_cache_gate \
     model_tag=issue831_2xh100_remote_cache_gate \
     device_batch_size=8 \
     total_batch_size=524288 \
     num_iterations=80 \
     save_every=40 \
     fp8=true \
     steering_eval_every=0 \
     remote_training.body_prefetch.cache_shards=24 \
     remote_training.train_packed.trust_existing_cache=true \
     > /workspace/issue831_2xh100_gate.log 2>&1 &
   echo $! > /workspace/issue831_2xh100_gate.pid
   ```

   Record the PID, log, Git revision, mount/filesystem type, resolved config,
   and remote-run fingerprint. After the marker-last checkpoint for step 40 is
   visible and verified in the bucket, terminate that exact torchrun process,
   verify both workers and GPUs are released, and relaunch the identical
   command. `resume_latest=true` must restore step 40 and continue through step
   80, crossing further shard boundaries. The gate passes only if:

   - every rank makes nonzero optimizer-step progress before and after resume;
   - packed and selection cache misses/hits occur across multiple shard names;
   - `remote_complete_000040.json` is marker-last and the resumed run reports
     the same remote-run fingerprint;
   - no cache `artifact changed`, integrity, Xet, provenance, or clustered
     selection failure appears;
   - no worker hangs during failure propagation, checkpoint coordination, stop,
     or resume.

   Run the focused multiprocess pytest suite with its temporary/cache
   directories on the same production filesystem mount as an additional
   `flock`/rename gate. A synthetic local filesystem result is not a substitute.

5. Only after step 4 passes, launch the eight-rank H100 job. The production
   target uses 16 rows per rank and four gradient-accumulation steps for the
   fixed global batch of 524,288 c1024 tokens. The H100 preset retains the
   existing run/model namespace, disables the remote steering suite, selects
   BF16 (`fp8=false`), and enables the measured 512 MiB/three-slot staging
   geometry.

   ```bash
   cd /workspace/nano-gpt-pretrain-steer
   set -a && . ./.env && set +a
   nohup torchrun --standalone --nproc_per_node=8 \
     -m scripts.train.conditional_encdec_train \
     config_name=cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100 \
     > /workspace/l22_h100_pretrain.log 2>&1 &
   ```

   The first compiled step is much slower than steady state. Confirm the DDP
   world size, the startup-contract fingerprint, the first cached training
   batch, and then nonzero step progress before judging throughput:

   ```bash
   tail -f /workspace/l22_h100_pretrain.log
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
   ```

6. Checkpoint and restart semantics are remote-first but marker-last. A usable
   bucket checkpoint has its complete data group plus
   `remote_complete_<step>.json`; partial uploads are intentionally not
   resumable. Leave `remote_training.resume_latest=true` and
   `resume_from_step=-1` for an interruption restart. Verify that a completion
   marker appears in the bucket after the first scheduled save before relying
   on remote resume.

#### Bare-host rebuild for the L22 remote-packed run

The runbook above assumes a host that already has a checkout, a `uv`
environment, and a populated `$NANOCHAT_BASE_DIR`. A freshly provisioned GPU
pod has **none** of those, and three of the gaps abort startup at different
depths, so a rebuild that only stages feature artifacts still fails. This
section is the complete bare-host procedure. It was derived by rebuilding the
run from nothing on two separate 8xH100 pods.

What a bare host is missing, and how each one fails if skipped:

| Missing | Symptom |
|---|---|
| Repo + `uv` environment | nothing runs |
| `out/data/features/...` staged families | `remote_training` startup contract fails |
| `out/autointerp_filter/...`, `out/artifacts/...`, `out/data/eval_context_features/...` | mask/cluster/eval-cache identity failure at startup |
| `$NANOCHAT_BASE_DIR/tokenizer/` | `FileNotFoundError: .../tokenizer/tokenizer.pkl` in `build_shared_state`, **after** distributed init |
| `$NANOCHAT_BASE_DIR/eval_bundle/` | survives startup, then fails at the first `core_metric_every` step (2000) |

The tokenizer and eval-bundle gaps are the easy ones to miss: they are not
feature artifacts, so a rebuild focused on `hf_pull category=features` sails
past them and dies later.

**1. Get the code onto the host.** The repo is private. If the host has no
GitHub credential, prefer a bundle over installing a PAT, which would persist
in `.git/config`:

```bash
# On a machine that can already reach GitHub:
git fetch origin refs/pull/<N>/head:pr<N>          # or: git fetch origin main
git bundle create pr<N>.bundle pr<N>               # ~19 MB for this repo
scp -P <port> -i <key> pr<N>.bundle root@<host>:/workspace/

# On the target host:
cd /workspace
git clone --branch pr<N> /workspace/pr<N>.bundle nano-gpt-pretrain-steer
cd nano-gpt-pretrain-steer
git remote set-url origin git@github.com:Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer.git
git rev-parse HEAD                                  # record in the launch log
```

**2. Build the environment and prove it imports.** Do not skip the import
check; it is where a bad dependency resolution surfaces cheaply rather than
across every rank:

```bash
export UV_CACHE_DIR=/workspace/.uv-cache
uv sync
.venv/bin/python -c "
from nanochat.checkpoint_manager import load_checkpoint
import torch, transformers; print(torch.cuda.device_count(), transformers.__version__)"
```

`transformers` must resolve to >=5.10.0. Earlier 5.x releases raise
`KeyError: 'flash_attn_interface'` at import on Hopper/Blackwell, because
`nanochat/flash_attention.py` resolves FA3 through the kernels hub and those
versions index `PACKAGE_DISTRIBUTION_MAPPING` unguarded. The floor in
`pyproject.toml` already encodes this; do not lower it.

**3. Stage the feature families.** Paths are relative to `out/data/features`.
This is the same include list as the runbook above, with the `<VAL_STEM>`
entries omitted because production launches with `steering_eval_every=0`:

```bash
DER='s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/seed42'
BASE='s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos'
LEAF='pct/623c372d3f121b18/tok/dc6308e6116cbfe0/c1024'

uv run python -m scripts.utils.hf_pull category=features +include="[
  \"$DER/feature_statistics.safetensors\",
  \"$DER/_splice_provenance/**\",
  \"$BASE/feature_statistics.safetensors\",
  \"$DER/$LEAF/sall/manifest.json\",
  \"$DER/$LEAF/sall/remote_packed_inventory.json\",
  \"$DER/$LEAF/sall/provenance/**\",
  \"$DER/$LEAF/val_sall/**\",
  \"$BASE/$LEAF/val_sall/**\"]"
```

Roughly 19 GB. The train `sall/*.safetensors.zst` bodies are intentionally
excluded — they stream on demand, and staging all 584 would pull 451 GB.

**4. Stage the run controls.** These live under the bucket's `support/`
prefix, which has **no `hf_pull` category**, so they need a direct bucket
download (`HfApi.download_bucket_files`). Their local paths are named
literally by the config and must match exactly:

| Bucket path (under `support/l22_gemma3pt_clustered/`) | Local path |
|---|---|
| `feature_mask/manual_feature_filter_with_descriptions.yaml` | `out/autointerp_filter/gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543/manual_feature_filter_with_descriptions.yaml` |
| `feature_clusters/feature_clusters_two_stage_schema_v1.yaml` | `out/artifacts/feature_clusters_two_stage_schema_v1.yaml` |
| `eval_context_cache/c1024/6be4095ad0116d3e/*` | `out/data/eval_context_features/gemma3_4bpt_l22_clustered/c1024/6be4095ad0116d3e/*` |

The cache leaf is `<context_cache_dir>/c{max_seq_len}/{16-hex}`, where the hex
is the first 16 chars of `conditioning.identity.feature_selection_hash`. The
bucket also holds superseded leaves (`63aeb505...` is the legacy seed=0 cache);
staging the wrong one is an identity failure, not a silent fallback.

**5. Stage `$NANOCHAT_BASE_DIR`.** Mirror the bucket's `tokenizer/` and
`eval_bundle/` prefixes into `$NANOCHAT_BASE_DIR` (4 and 77 files
respectively). `base_data/` is **not** required for this run: it uses
precomputed batches and packed validation families.

**6. Verify before spending GPU time.** Assert counts, not just existence:

- derived `_splice_provenance/*.json` — **370**
- train `sall/provenance/*.zst` — **584**, and **zero** `sall/shard_*.safetensors.zst`
- derived `val_sall` and base `val_sall` — **2** bodies each, plus `manifest.json`, `provenance/`, `sel/`
- base `val_sall/sel/` contains `6be4095ad0116d3e1b7ca93a2fe0a47439eaba770c6d0ef08ab164a0652a19bc`
- `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` and `eval_bundle/core.yaml` exist

When writing such a checker in shell, keep globs **unquoted** in the `ls`
expression — `ls "$dir/*.json"` matches literally and reports a false zero.

**7. Launch.** Two host-specific arguments beyond the runbook command:

- `data_dir=` must point at an existing but empty directory. Left unset, the
  splice-counterfactual path auto-derives `<feature_dir>/_parquets` and raises
  because those parquets are deliberately not staged.
- `run=`/`model_tag=` must be **absent from `checkpoints/conditional_encdec/`
  in the bucket**. The preset ships `resume_latest=true`, so a reused tag
  silently resumes a different run's checkpoints instead of starting fresh.
  List that prefix and pick an unused literal before launching.

Credentials: pass `HF_TOKEN` into the process environment only (for example by
reading it from stdin in the launch wrapper). The token needs **write** role
for checkpoint publication; verify with `whoami-v2` rather than discovering it
at the first save.

Sanity checks in the first minute of a correct fresh start:

- `remote_training startup contract verified: run_fingerprint=<64 hex>`
- `Encoder feature dictionary: 2,074/65,536 rows kept` — the intended clustered
  dictionary; a different number means the mask or cluster artifact is wrong
- `Total batch size 524,288 => gradient accumulation steps: 8` at 8 ranks with
  `device_batch_size=8`
- step 0 loss ~**10.40** (`ln(32768)`), confirming a genuinely fresh init
  rather than an unintended resume

`save_every` is **500**, inherited from `base_train`; it is not set anywhere in
the `cluster_fp` chain.

Encoder-decoder checkpoints nest under `$NANOCHAT_BASE_DIR/conditional_encdec_checkpoints/` with segment size and feature selection subdirectories. The feature selection dirname depends on the mode:

- **Threshold:** `t{threshold}` (single dir, e.g., `t2048.9`)
- **Sampling:** `dist_{distribution}/mean_{mean}/std_{std}` (nested dirs, e.g., `dist_geometric/mean_5.0/std_5.0`)

Downstream SFT checkpoints nest inside the pretraining checkpoint dir. The SFT sub-directory encodes the dataset name **and** whether feature conditioning was enabled:

```
conditional_encdec_checkpoints/
└── d26/
    └── s128/
        ├── t2048.9/                                       # threshold mode
        │   ├── model_005000.pt                            # source="encdec"
        │   ├── meta_005000.json
        │   ├── safety_sft_conditioned_checkpoints/        # conditioned SFT
        │   │   ├── model_000300.pt
        │   │   ├── meta_000300.json
        │   │   └── optim_000300_rank0.pt                  # sharded per DDP rank
        │   └── report/
        └── dist_geometric/                                # sampling mode
            └── mean_5.0/
                └── std_5.0/
                    ├── model_005000.pt
                    └── capabilities_sft_conditioned_checkpoints/
                        └── model_000300.pt
```

The pretraining feature-selection dirname (`t2048.9`, `dist_.../mean_.../std_...`) is computed by `feature_selection_dirname()` in `steering.conditional.feature_pipeline`. The SFT leaf directory name is built directly in `scripts/train/conditional_encdec_sft.py` as `{dataset}_sft_conditioned_checkpoints`.

SFT checkpoint meta (`meta_XXXXXX.json`) records `step`, `dataset`, the entire resolved `user_config` (which includes `feature_dir`, `feature_selection`, `num_buckets`, etc.), and `min_val_bpb`. Optimizer state is saved sharded per DDP rank.

## Feature Mask (Blacklist/Whitelist)

An optional `feature_mask` block restricts which SAE features are eligible to appear as positive conditioning, negative conditioning, FP-prediction targets, or self-conditioned-generation feedback.

The mask operates at **two** layers:

1. **Selection layer** (everything in this section) — which features may be picked. Base extraction shards, statistics, percentile overlays, token boundaries, and packed rows are never recomputed (invariant I3).
2. **Structural layer** (issues #695/#714) — the resolved mask and static-cluster consolidation **size the encoder's feature-id-indexed embedding tables**: one row per effective feature (allowed and not consolidated away), not one per SAE feature. Always derived; no config knob. See [§ Encoder Feature-Dictionary Compaction](#encoder-feature-dictionary-compaction).

### Config schema

```yaml
feature_mask: null          # absent/null ⇒ disabled; no-mask run is bit-identical to legacy (I1/I2/I5)
# feature_mask:
#   mode: blacklist         # blacklist | whitelist
#   ids: [123, 456]         # inline list; XOR with `path`
#   # path: /abs/path/feature_ids.txt   # .txt one int/line ('#' comments + blank lines ok)
#                                        # OR .yaml/.yml flat strict ints or
#                                        # quoted decimal IDs (zero-padded OK)
```

Pretraining passes each configuration’s already resolved mask snapshot into the feature pipeline. The pipeline and negative sampling share its derived mask memo; the next configuration resolves its own mask.

`resolve_feature_mask` (in `steering/conditional/feature_mask.py`) is the single entry point. It validates:
- Exactly one of `ids` (inline) or `path` (file) must be present — both or neither is a `FeatureMaskError`.
- Duplicate IDs → `FeatureMaskError`.
- Any ID outside `[0, n_sae_features)` → `FeatureMaskError`. Here `n_sae_features` is the **ORIGINAL** SAE dictionary width (16,384 / 65,536 / …), *not* the encoder's compacted table height `V` (`model.feature_vocab_size`). Mask ids — like every other feature id crossing an API boundary — are always original SAE ids.
- Empty resolved set → `FeatureMaskError`.

The direct runtime objects express those unary domains in the type system:
`FeatureMask.ids` is a non-empty tuple of strict non-negative integers,
original SAE widths are strict positive integers, frequency bounds reuse the
shared `[0, 1)` / `(0, 1]` phantom intervals, and delta-negative counts and
ratios are non-negative integers / inclusive unit-interval numbers. `beartype`
enforces those call boundaries. Config, checkpoint, and file adapters translate
violations back to their established project error types; upper bounds,
`freq_min < freq_max`, tensor alignment, and cluster-definition relationships
remain explicit checks at the owners that have both sides of each relationship.

### Blacklist vs whitelist semantics

| Mode | Positive selection | Negative universe | FP targets | Self-cond feedback |
|------|-------------------|-------------------|------------|--------------------|
| `blacklist` | Forbidden IDs dropped | Forbidden IDs excluded | Forbidden sources dropped; allowed members map to effective rows | Forbidden IDs filtered |
| `whitelist` | Only whitelisted IDs kept | Only whitelisted IDs eligible | Non-whitelisted sources dropped; allowed members map to effective rows | Non-whitelisted IDs filtered |

In both modes the mask is enforced identically at:
1. **Precompute selection shards** (`scripts/preprocess/precompute_selected_features.py`) — applied when building `sel/<hash>/` shards.
2. **Fused inline sampler** (`reconstruct_and_select` in `dataloader.py`) — applied during packed-row batch assembly.
3. **Live dataloader** (`precomputed_data_loader` / the non-fused path in `dataloader.py`) — applied per-batch.
4. **All three negative-sampling implementations:** `sample_inactive_features`, `sample_inactive_features_batched` (in `steering/conditional/sampling.py`), and `sample_delta_negatives` (in `steering/conditional/delta.py`) — each receives `feature_mask` and restricts the eligible pool via `allowed_universe_mask`.
5. **FP-target building** (`build_fp_targets_from_shard` for pretraining and `make_sft_data_generator` for conditioned SFT) and **FP loss** (`compute_feature_pred_loss`) — full-width targets preserve legacy masked-column zeroing. Compact targets contain only allowed effective rows, so the loss receives no redundant mask.
6. **Self-conditioned generation feedback** (`SelfConditionedFeatureProvider` in `steering/conditional/encoder_decoder/self_conditioned.py`) — full-width feedback filters to the allowed universe. On a compact head, the effective vocabulary already excludes masked and consolidated-away rows; the provider pairs each compact score with its original representative id before the same selection and encoder path. The packed-pretraining post-training source remains uncompacted-only; source-aware SFT supports exact compact mask/cluster identity and compact-row STE.

### Compact feature-prediction targets and source integrity

The internal `FeaturePredictionTargetLayout` is derived from the already-resolved
`FeatureMask`, `FeatureClusterConsolidation`, and model `FeatureVocab`; it is not
another serialized identity. `FeatureVocab is None` means the literal legacy
full-width branch. Otherwise target construction transforms each actual target
segment in this exact order:

```text
raw packed CSR or SFT conversation segment in original SAE IDs
  → validate original-ID bounds and per-segment uniqueness
  → apply the resolved mask
  → map each surviving member to its post-mask representative
  → max-reduce sibling percentiles at that representative
  → map the representative original ID to a compact row
  → scatter directly into [segment, V_effective]
```

For pretraining the source is the base packed shard, never the positive/negative
selection overlay. Splice-counterfactual delta therefore changes conditioning
selection, not the FP target: the head predicts the next actual segment. SFT
uses each conversation segment's full `all_feature_ids` / `all_feature_pcts`
pair before selection. Mask-only compact targets exactly equal the legacy full
masked target sliced at `FeatureVocab.original_ids`; active clusters use one
`amax` scatter. No compact branch allocates an `n_sae_features`-wide dense
intermediate.

Compact-FP pretraining accepts only precomputed packed rows. Before outputs or
the first batch, the loader requires `manifest.json` metadata
`fp_target_source_integrity` semantic version 1. The certificate binds the
original SAE width, ordered shard inventory and exact packed-row fingerprint,
records checked row/segment/feature counts, and certifies CSR structure, bounds,
and unique original IDs per segment. Certify an existing family atomically with:

```bash
uv run python -m scripts.preprocess.validate_fp_target_source \
  packed_row_dir=<packed-row-dir> n_sae_features=<original-sae-width>
```

Future `precompute_packed_rows` runs write the same certificate after production.
A failed scan leaves the manifest unchanged. Conditioned SFT has no packed-row
source to certify, so it performs the equivalent duplicate-original-ID check on
every source segment before folding; legacy full-width FP and headless loaders
do not require the manifest field.

Compact dataloader resume state conditionally records the existing vocabulary
metadata, FP output width, and source-certificate digest. Checkpoint mask,
cluster, and vocabulary metadata remain authoritative. Resume compares all of
them and the strict head tensor shape before `load_state_dict(assign=True)`;
partial metadata, same-width/different-ID vocabularies, changed cluster
membership, changed masks, and changed packed sources fail before a saved row is
consumed. Conditioned-SFT checkpoint metadata similarly records
`feature_prediction_width` whenever the inherited head is active. Old
full-width SFT checkpoints may omit it; a compact-FP resume requires both width
and vocabulary metadata, because pre-#1276 SFT could not create such a
checkpoint.

### No-clobber mechanism

`compute_selection_hash` (in `steering/packed_batch_io.py`) inserts `feature_mask` into the hash payload **only when a mask is configured**. A no-mask run hashes byte-identically to the legacy 6-key payload so every existing `sel/<hash>/` shard stays valid (invariant I1).

The checkpoint path gets an optional `bl_<16hex>` or `wl_<16hex>` leaf appended by the training scripts: `conditional_encdec_train.py`, `conditional_encdec_sft.py`, and `convert_decoder_to_encdec.py` call `append_mask_leaf`; `run_encdec_sft_pipeline.py` calls `feature_mask_leaf_from_cfg`. Clustered pretraining and SFT then append `fc_<64hex>`. The 16-hex mask value is a SHA-256 prefix of the canonical mask JSON (mode + sorted unique IDs). No mask/clusters returns the legacy path byte-identically (invariant I2).

> **The `bl_`/`wl_` leaf is no longer just a path disambiguator.** Since #695 the mask determines the encoder's parameter *shape*, and the vocabulary is a pure function of the mask — so the leaf now identifies **which dictionary a checkpoint's embedding rows are indexed by**. Two sibling leaves under one feature-selection dir hold models whose row *j* denotes a *different feature*. The checkpoint itself is self-describing (it carries `encoder.feature_vocab_ids`), so loaders do not have to trust the path — but a human reading the tree must not treat the leaf as cosmetic.
>
> Caveat: the leaf hashes only `(mode, ids)`, **not** `n_sae_features` (`feature_mask.py`'s `canonical_dict`). The same `{whitelist, ids: [1,2,3]}` on a 16k and a 65k SAE produce the same leaf, the same `original_ids` and the same `V`. That is why `feature_vocab_n_sae` is persisted as its own buffer and compared explicitly on load rather than inferred.

> **nanochat read-only constraint:** `nanochat/checkpoint_manager.py` (`get_checkpoint_dir` / `_find_checkpoint_leaf`) was NOT modified. Read-side checkpoint discovery relies on the existing `_find_checkpoint_leaf` recursion for unpinned callers, which traverses the directory tree and finds the mask leaf automatically. The `harmfulness_eval` pinned-discovery case (explicit `feature_selection` path) does not yet handle the mask leaf — this is a known follow-up (Task 15b).

### sel/ manifest record and assertion

When a `sel/<hash>/` directory is first created, its manifest records `metadata.feature_mask` as the canonical dict (`{mode, ids}`) or `None`. On subsequent discovery, `conditional_encdec_train.py` asserts that the manifest's `feature_mask` matches the currently-configured mask and raises `FeatureMaskError` on mismatch. Legacy manifests without a `feature_mask` key are treated as `None` (backward-compatible with no-mask runs, invariant I5).

### Eval-side enforcement

The same `FeatureMask` primitive is threaded into **three eval paths**. These mirror the training-side sites — a masked model must be evaluated through the same mask-aware pipeline.

1. **Conditioned-BPB eval** (`scripts/eval/base_eval.py` — `_resolve_bpb_selection_dir` at L322, called at L1244): `feature_mask` (passed as `canonical_dict(feature_pipeline.feature_mask)`) is folded into the `compute_selection_hash` call that locates the `sel/<hash>/` dir, so a mask-trained model's BPB eval resolves the **same selection shards** it trained on. After resolving the dir a manifest cross-check asserts that `metadata.feature_mask` in the manifest matches the configured mask — mismatch raises `FeatureMaskError` (same guard as training). Legacy manifests without a `feature_mask` key pass when no mask is configured (I5).

2. **Eval-context-cache identity** (`steering/eval/feature_selection_hash.py` — `compute_canonical_feature_selection_hash` at L35; `steering/eval/cache_identity.py` at L123): when the hash is computed from canonical feature-selection inputs, `feature_mask` is included in the SHA-256 hash that forms the `CacheIdentity.feature_selection_hash`, so a masked run's context cache cannot collide with an unmasked one. An explicitly pinned `identity.feature_selection_hash` is used as-is and therefore must be updated by the operator when the mask changes. When `feature_mask` is `None` the canonical hash is byte-identical to the pre-H7 value — existing unmasked caches remain valid (cache-identity I1).

3. **Completion-steering eval** (`steering/eval/completion_steering_eval.py` — `select_from_tensors` at L1053 and `assert_selections_within_mask` at L1036): `select_from_tensors` restricts the feature-selection universe to mask-allowed IDs via `feature_mask.keep_mask`; `assert_selections_within_mask` is a defense-in-depth check applied every time selections are materialised or loaded, so a stale pre-mask `feature_selections.yaml` is rejected loudly (`FeatureMaskError`) rather than silently steering on a blacklisted feature. The config slot `feature_mask: null` is present (with usage guidance) in both `configs/eval/completion_steering_eval.yaml` and `configs/eval/completion_steering_eval_gemma3.yaml`. No-mask path is bit-identical (I5).

### Eval-correctness contract

> **A model trained with `feature_mask` MUST be evaluated with the same `feature_mask` configured.**

The conditioning pipeline (BPB `sel/<hash>/` discovery, canonically computed
eval-context-cache identity, completion-steering feature selection) is keyed on
the mask. An explicitly pinned context-cache hash remains operator-managed, as
described above. If the mask is **omitted at eval time**:
- The BPB eval resolves a **different** `sel/<hash>/` dir (the unmasked hash), which either does not exist or contains shards the model was never trained on — or, if only an unmasked shard exists from a prior run, conditions on forbidden features.
- A canonically computed eval-context-cache identity resolves a **different**
  hash and misses the masked cache entirely. An explicitly pinned
  `identity.feature_selection_hash` does not change automatically and must be
  updated by the operator.
- The completion-steering eval selects features from the **full unrestricted universe**, including blacklisted / non-whitelisted IDs the model was never trained to respond to.

**Safeguards that exist:**
- The training-time manifest assertion (`sel/<hash>/metadata.feature_mask`) guards against `sel/` drift — if a stale or wrong shard is somehow reused the mismatch is caught with `FeatureMaskError`.
- AxBench-Base Stage 4 folds runtime `conditioning.feature_mask` into a
  context-cache identity only when that identity is computed from canonical
  feature-selection inputs. The shipped AxBench configs instead pin
  `identity.feature_selection_hash`; when using that explicit hash, operators
  must regenerate/update it whenever the mask changes. The runtime mask must
  still match the model's training mask.
- The `feature_selections.yaml` defense-in-depth check in completion-steering eval (`assert_selections_within_mask`) catches a stale file containing masked IDs.
- **On a COMPACTED checkpoint (#695/#714), conditioning on a mask-excluded or consolidated-away feature can no longer be silent.** The feature has no encoder embedding row, so the encoder-boundary remap (`FeatureEncoder._to_compact`) raises via `torch._assert_async` — in eager **and** under `torch.compile`. The failure is loud, but it lands at the **first forward**, not at startup, and the message cannot name the offending ids (naming them would require a host sync on every eager CUDA forward, which was rejected). It names `V`, `n_sae_features`, and the mask/cluster recovery remedy instead.
- **Model shape is recovered from the checkpoint, not from the eval config.** `build_encdec_model` reads `encoder.feature_vocab_ids` / `feature_vocab_n_sae` out of the state dict, so a compacted checkpoint always loads with the right table height and the right *original* `n_sae_features` even if the eval config's mask is wrong or missing. The mask is still needed for the **selection** layer.

**Gap — user responsibility (narrowed, not closed):**
- **SFT now enforces it at startup** on a compacted base checkpoint — the resolved mask's allowed universe must *equal* the checkpoint's vocabulary, and `feature_mask: null` raises. See [§ The equality rule](#the-equality-rule-downstream-conditioning-must-match-the-checkpoint-vocabulary).
- For **standalone BPB eval** (`base_eval.py`), **completion-steering eval**
  (`completion_steering_eval.py`), and **unclustered AxBench-Base Stage 4**
  (`conditioning.feature_mask`), there is still no *startup* enforcement that
  the runtime mask matches training. Clustered AxBench-Base/BPB-Adv are the
  narrow exception: their feature contract compares the path-backed
  mask/clusters with checkpoint and selection provenance before stage work.
  Elsewhere the user must set the mask identically in the eval config. The
  documented contract above is the primary mitigation; on a compacted
  checkpoint the encoder-boundary assert is the backstop (loud, but only once a
  bad id is actually conditioned on). Recover the exact training mask with
  `uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`. See
  [evaluation.md § Feature Mask — Eval-Correctness Contract](../pipelines/evaluation.md#feature-mask--eval-correctness-contract)
  for the quick reference.

### Auto Activation-Frequency Blacklist (producer for `feature_mask`)

`feature_mask_freq_min` / `feature_mask_freq_max` (defaults `1e-5` / `0.5`, siblings of `feature_mask`) make training **produce** the blacklist automatically. Before the first `resolve_feature_mask` call, `maybe_apply_frequency_blacklist` (`steering/conditional/freq_blacklist.py`) reads the precomputed `feature_statistics.safetensors`, blacklists every feature whose per-segment frequency lies *outside* the kept interval **KEEP iff `freq_min < freq <= freq_max`** (LB inclusive — the default `freq_min: 1e-5` kills `count==0` dead features *and* other features too rare to be autointerp-scoreable (`count <= 1e-5 * n_segments`); pass `freq_min: 0.0` explicitly to kill `count==0` dead features only; UB exclusive — so `freq_max: 0.5` kills features active in >50% of segments), using the exact integer `count` + `n_segments` (never the rounded `pct_segments_active`); writes `{feature_dir}/blacklist/freq_min{m}_max{M}.{txt,meta.json}` (DDP rank-0 atomic; (re)written only when the deterministic content changes); and sets `cfg.feature_mask = {mode: blacklist, path: <file>}`. From there it is an ordinary `feature_mask` blacklist — every enforcement/no-clobber/eval rule above applies unchanged.

Source = the statistics artifact, so **the train/val split is whatever `compute_statistics` used** (validation shard excluded by default, `exclude_last_shard: true`); run statistics computation first. Call sites are pretraining-only: `conditional_encdec_train.py` `build_shared_state` (before the discovery selection-hash) **and** the top of `train_one_config` (so the amortized fast sweep's fresh per-config cfg also gets it — idempotent on the same cfg). SFT has no frequency producer and requires the base checkpoint's exact explicit mask. The bounds **never** enter any selection hash — only the resolved id set does — so two bound-pairs yielding the same id set share a `sel/<hash>/` and checkpoint leaf.

No-op (cfg byte-identical ⇒ I1/I2/I5) when: BOTH `feature_mask_freq_min` and `feature_mask_freq_max` are `null` (fully disabled); the computed blacklist is empty; or an explicit `feature_mask` is set (it wins, with a warning). Either bound can independently be `null` to skip that side. **Resume / eval:** keep both bounds + statistics constant; to eval, point `feature_mask.path` at the generated `freq_min{m}_max{M}.txt` (or re-run `scripts/preprocess/precompute_feature_blacklist.py` with the same bounds — output is deterministic). The old single key `feature_mask_freq_threshold` is renamed to the two bounds above (no longer recognized). Full schema: [configuration.md § `feature_mask_freq_min` / `feature_mask_freq_max`](../reference/configuration.md#feature_mask_freq_min--feature_mask_freq_max--auto-activation-frequency-blacklist-producer).

> **MIGRATION — changing a bound re-keys the checkpoint leaf and `sel/<hash>/`, and a pre-#695 run cannot be resumed at all.** The bounds themselves never enter a hash, but the **resolved id set** does, so a band that resolves a different blacklist produces a different `bl_<16hex>` checkpoint leaf *and* a different `sel/<hash>/`. This bit the shipped defaults: `freq_min` moved `0.0 → 1e-5` and `freq_max` moved `0.99 → 0.5` (#702/#695), so **a run started on the old defaults will not resume on the current ones** — it looks under a leaf that holds no checkpoint, and its `sel/` artifacts are orphaned (not deleted) and silently regenerated at full cost.
>
> **Re-pinning the old bounds does not rescue it.** It does reproduce the old id set, hence the old leaf and the old `sel/<hash>/` — the leaf was never the obstacle. The **encoder shape** is: compaction arrived *with* #695, so that checkpoint is physically uncompacted (full-width `encoder.feature_embed`, no `encoder.feature_vocab_ids` buffer), while any mask that resolves now builds a **compacted** encoder. `assert_checkpoint_vocab_compatible` catches the mismatch just before `load_state_dict` and raises `FeatureVocabError` ("resuming would require reshaping the checkpoint"); there is no reshape tool. Do **not** null the band to get around it — that does not fix the run (a no-mask run resolves a different, unmasked `sel/<hash>/` and an un-leafed checkpoint path, so it is a different experiment, not a continuation) and it forfeits compaction as well. **Start the run fresh on the current defaults**; the old artifacts stay on disk, orphaned. Details: [configuration.md § MIGRATION](../reference/configuration.md#migration-the-freq-default-changes-re-key-your-artifacts-and-a-pre-695-run-cannot-be-resumed).

### Autointerp Intruder Whitelist (producer for `feature_mask`)

`scripts/preprocess/precompute_autointerp_filter.py` scores local extracted singleton features with intruder detection and writes `${output_dir}/accepted_feature_ids.txt`. Use it as an ordinary whitelist:

```yaml
feature_mask:
  mode: whitelist
  path: out/autointerp_filter/accepted_feature_ids.txt
```

This step is offline: it never rewrites base shards, statistics, percentile overlays, token boundaries, packed rows, or `sel/<hash>/` directories. Target resolution applies candidate IDs, then an optional input `feature_mask`, then sorted singleton feature targets. Feature shards only identify candidate source documents; scoring re-forwards those documents through the SAE and builds token-level windows. Token-example caching is keyed by source fingerprints, model/SAE/backend/dtype/token-window settings, and target identity. Parsed-trial identity combines target/trial/sample settings with stable-key digests of sampled refs, judge model/CoT, prompt/few-shot names and their truncated content-version fingerprint, and provider context. Those ref digests exclude `token_strings`, and the rendered final prompt is not directly hashed; renderer/token-string changes require an identity-method/version/context bump. The configured prompt-length cap, parse retries, RPM, and concurrency are also deliberately excluded, and the cap is enforced only for cache misses. Prompt construction mirrors EleutherAI Delphi's `IntruderScorer`; intruder rows are target-non-activating token windows with Delphi-style false-positive highlights. Candidate windows are deduped by rendered source-window identity before trial sampling, each sampled trial rejects overlapping same-document windows across all presented options, and only successfully parsed trial results are cached. `trial_audit.jsonl` provides bounded detailed samples with token strings, token activations, highlights, and judge responses; `trial_audit.yaml` keeps the same sanity-check prompts and rendered examples compact for manual review. Prompt/rendering, judge, and enabled Neuronpedia failures abort the run. Full schema and collision boundary: [configuration.md § Autointerp Intruder Filter](../reference/configuration.md#autointerp-intruder-filter--whitelist-producer).

<a id="static-feature-clusters-preselection-only"></a>

## Static Feature Clusters (Preselection + Completion-TUI Consumption)

*Issue #714. Primitive: `steering/conditional/feature_clusters.py`.*

`feature_clusters` restores only the ability to **consume** a static cluster
definition while materializing selected-feature shards. It does not discover
clusters, write a clustering overlay, or restore the removed
`feature_clustering` config/checkpoint/manifest schema. The supported V1
pipeline is deliberately narrow:

```text
original SAE feature rows
  -> fully resolved manual/frequency feature mask
  -> ordinary: representative max(percentile) -> selection
  -> clustered splice-CF: independently compute
       current representative max(percentile)
       reference representative max(percentile)
     -> representative delta = current max - reference max
     -> existing positive/negative selection
  -> representative-only sel/<hash>/ shard in original SAE IDs
  -> effective encoder vocabulary = mask-allowed ids minus consolidated-away ids
  -> FeatureEncoder._to_compact at the model boundary
```

### Definition, filtering order, and representative

Configure a path to a strict schema-v1 YAML or JSON file:

```yaml
feature_clusters:
  path: /path/to/feature_clusters.yaml
```

```yaml
schema_version: 1
clusters:
  - [12, 43, 91]
  - [7, 104]
```

The source mapping has exactly `schema_version` and `clusters`. A cluster has
at least two strict integer original SAE IDs. The source clusters are
non-empty, internally unique, pairwise disjoint, and in range for the original
SAE width. Duplicate mapping keys and unknown fields fail. Member and cluster
ordering are canonicalized, so semantically identical YAML and JSON files have
the same content identity.

The complete explicit or frequency-produced `feature_mask` resolves **first**.
For source cluster `C`, let `S` be its mask-allowed members. If `S` is non-empty,
`min(S)` is the fixed representative for the whole run; every other member of
`S` is consolidated away. The representative is not re-elected from whichever
members happen to occur in one segment. Zero- and one-survivor clusters are
inert. This ordering means changing a whitelist/blacklist can change the
representative while leaving the source cluster file unchanged.

The same frozen consolidation defines the encoder's **effective feature
vocabulary**: every mask-allowed id remains except a member consolidated into a
different representative. Representatives keep physical rows;
`consolidated_away` members do not. `effective_feature_vocab(feature_universe,
n_sae_features)` — where `feature_universe` is a `FeatureClusterConsolidation`,
a `FeatureMask`, or `None` — is the only production constructor for clustered
models. A consolidation carries the mask it was built against, so there is no
separate mask argument to drift out of agreement; a bare mask or `None`
literally delegates to `FeatureVocab.from_mask`. All
cluster definitions, manifests, packed/selection shards, and trainer APIs still
use original SAE IDs. Only `FeatureEncoder._to_compact` translates the final
representative to a compact row, and an away id reaches the ordinary loud
out-of-vocabulary guard.

When SAE-backed initialization and
`initialize_encoder_embeddings_cluster_vectors: mean` (the default) are active,
each retained representative is derived from the fp32 mean of all
mask-surviving cluster members. In `random` mode the mean is applied after the
existing full-dictionary projection/rescale and before the effective-vocab
gather. In `svd` mode it is applied to oriented `W_dec` before the
effective-vocab subset/covariance, so the basis sees the averaged dictionary.
It changes initialization only; selection still emits the same fixed
representative ID and all checkpoint/selection identity remains
cluster-derived as described below.

### Ordinary and splice-counterfactual semantics

For ordinary selection, apply the mask and then max-reduce each representative's
member percentiles before thresholding or sampling. Supported ordinary modes
are `weighted`/`top_k_percentile` sampling and
`feature_percentile`/`feature_percentile_segment_topk` thresholds. Raw-magnitude
thresholds are rejected because a max-reduced percentile is not the raw
magnitude of any one source feature.

For `feature_delta.mode=splice_counterfactual`, resolve the mask on the current
and would-have-followed segments separately. Independently map both surviving
sides through the same fixed `rep_of` tensor and max-reduce each
representative's normalized percentiles. Only then compute
`delta(rep) = current_pct(rep) - reference_pct(rep)`. Missing representatives
use the existing zero/absence treatment. The clustered current percentile
drives serialized percentiles, buckets, and `absolute_percentile` ranking; the
representative delta drives `percentile_delta` ranking. This ordering is
deliberately non-commutative: for one cluster with current `[0.9, 0.1]` and
reference `[0.1, 0.9]`, the correct delta is `0.9 - 0.9 = 0`, not the `+0.8`
produced by max-reducing raw member deltas. The generic cluster primitive does
not import or parse delta configuration; `delta.py` composes its two side
reductions. Only sampling is supported; splice threshold selection is rejected.

Negative selection follows four additional invariants:

- A consolidated-away feature is never emitted as a negative.
- Every allowed sibling contributes only through its side's representative
  maximum; signed-negative evidence is then constructed directly in
  representative space. No raw member-negative pool is remapped after delta.
- A representative in the full consolidated positive-candidate set wins over
  negative evidence even when sampling does not select it as a positive.
- Current/reference presence is mask-filtered and projected to representatives
  before ordinary inactive-universe sampling.

### Artifacts, identity, and resume

Clustering conditionally extends the selection hash with a descriptor containing
the schema version, original SAE width, SHA-256 of canonical cluster content,
and versioned semantics. The source path is provenance only. Clustered
checkpoints append `fc_<full-descriptor-sha256>` after any mask/gradient leaves
and persist both the descriptor and canonical definition; resume requires exact
agreement. Disabled runs add no key or leaf and retain legacy identity exactly.
That `fc_` leaf also pins the model-shaping cluster input. Checkpoints written
before effective-feature compaction used the same leaf but kept a mask-only
vocabulary; current resume recognizes that exact shape and raises a dedicated
pre-effective-compaction diagnostic. Start fresh: no repository tool reshapes
all feature-indexed parameters and optimizer state.

Clustered splice-counterfactual selection has one additional, narrower identity:
`cluster_delta_semantics=mask_then_cluster_each_side_max_percentile_then_delta_v1`.
It is added only when both static clusters and `splice_counterfactual` are
active. Unclustered splice selection retains its exact legacy hash and bytes;
the clustered delta-free base-validation twin retains its exact descriptor-only
hash. The conditional key prevents corrected selected bytes from sharing a
directory with artifacts produced by the former raw-delta-then-max ordering.

The producer writes an `in_progress` selection manifest before shard work and a
`complete` manifest only after validating exact packed-shard stem coverage and
every output tensor. Acceptance requires the exact selection dtypes, valid CSR
offsets and alignment, in-range buckets, per-segment positive/negative
uniqueness and disjointness, and representative-only IDs. A complete clustered
manifest additionally pins the exact sorted shard filenames, compressed byte
lengths, per-shard SHA-256 values, and an aggregate SHA-256. Both the trainer
and `precomputed_data_loader` require and recompute that complete contract,
then scan every clustered selection shard before yielding the first batch.
Before a producer reuses any completed shard, it verifies the same output
fingerprint plus the prior manifest's full selection checksum and packed-row
byte fingerprint. Shards left by an `in_progress` run are deliberately treated
as unverified and recomputed deterministically. A partial, stale, corrupt,
path-copied, or differently clustered selection directory therefore fails
before training consumes it.

Every newly completed clustered family, including the delta-free validation
twin, also carries a schema-v1 `clustered_selection_validation` certificate.
The producer constructs it from the same final stat-fenced evidence used to
validate the compressed selections: exact selection and packed content
digests, per-shard segment counts, maximum positive-plus-negative CSR widths,
cluster/delta/bucket identity, and canonical aggregates. The `complete`
manifest is the atomic commit and is written only after that certificate is
complete; `in_progress` manifests never advertise it.

For clustered production, one strict packed proof is established before shard
processing. Every packed body read used for splice statistics or selection is
bracketed by that proof's exact link/target signature, including spawn-worker
reads, and final certificate construction reuses the same proof rather than
rehashing the packed family. The producer rechecks both exact shard domains and
all signatures immediately before the atomic complete-manifest commit.

The certificate block is derivable evidence and is the only new manifest field
excluded from the selection manifest's semantic hash. Reissuing it changes the
raw manifest audit digest but not the selection checksum, checkpoint feature
contract, or selected shard bytes. A no-op producer rerun with
`selection_resume=true` rehashes and semantically revalidates the compact
selection outputs. An existing schema-v1 proof is rechecked against the current
compressed inventory; a legacy artifact replays semantic validation from the
compact selection tensors. Neither path recomputes unchanged selections. A
completed schema-v3 clustered-splice resume certificate remains an accepted
legacy evaluator proof when its complete statuses, storage
validation, remote-loader-v2 evidence, inventory, packed fingerprint, counts,
width, buckets, and cluster-delta semantics agree exactly.

For evaluation, those producer semantics and the packed-row fingerprint cache
are deliberately separate trust boundaries. The producer certificate says the
certified selection bytes passed the canonical packed/selection validators.
The packed cache says the current packed files still have the certified
content, using exact link and target mode/inode/size/`mtime_ns`/
`ctime_ns` signatures. Evaluation always rehashes the much smaller compressed
selection files. A valid cache allows it to skip packed body hashing and both
packed/selection tensor decompression; it does not claim protection against an
adversary able to forge producer metadata or preserve every filesystem
signature while changing bytes.

#### Clustered-evaluation certificate migration

Completed clustered splice artifacts with the schema-v3 splice proof need no
migration. For an uncertified or delta-free clustered artifact, rerun its exact
producer command with resume enabled; for the documented delta-free shape:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<feature_dir> packed_row_dir=<packed_row_dir> \
    +feature_clusters.path=<clusters.yaml> feature_delta=null \
    feature_selection.delta_weights=null selection_resume=true
```

This revalidates and certifies existing selection bytes. It does not rewrite
them. Missing, stale, malformed, or unreadable packed fingerprint caches are
recomputed from packed bytes; cache writes are best effort. Cache entries are
independent of selection approval and may remain after a later validation
failure. `force_cluster_shard_revalidation=true` still performs the canonical
full body validation, even on a valid warm cache. Packed source changes or
selection mismatches remain errors and never produce an evaluator contract.

The same conditional identity is explicit in recovery metadata. Clustered
splice manifests use schema v2 and carry `cluster_delta_semantics`; their signed
`splice_resume` certificate is schema v3 and binds the same value (schema v2 was
already assigned to post-serialization shard validation). Loader resume state
records it beside the cluster descriptor. Missing or former-semantic values fail
before shard reuse or batch yield. Unclustered splice manifests remain schema v1.
Delta-free clustered selection and splice identity remain on their
pre-correction schemas and carry no conditional semantic key, while their
remote-loader certificate is reissued under the single current schema v2.

### Supported training boundary

V1 requires `use_precomputed_batches=true`, no gradient ranking or
next-segment conditioning, and either `feature_delta=null`
or `splice_counterfactual`. Conditioned validation through the precomputed
training loader remains supported, but each enabled validation split needs its
own complete clustered selection artifacts. Cached CORE and sample generation
are supported when their prompt cache satisfies the cluster-aware contract
below.

In-training clustered steering is supported only for the
`splice_counterfactual` arm with complete derived-validation packed rows,
selection shards, and provenance. The cache is built and cross-rank-digested at
`train_one_config` startup, before a training row is consumed. It reuses the
delta-aware derived selection; it does not use the delta-free CORE/sample cache
and does not re-threshold raw features.

Clustered feature prediction is supported on this precomputed path. It predicts
the next actual segment from the raw base packed CSR, independent of which
positive/negative conditioning rows the selection overlay retained. Compact
targets additionally require the packed manifest's completed
`fp_target_source_integrity` certificate.

Live/fused pretraining selection, serving/chat, and on-policy self-conditioning
post-training reject active `feature_clusters` config or clustered checkpoint
metadata before output. The already-loaded inference provider is the narrower
exception documented below.
Feature-GCG does the same unless its entry point receives a validated cluster
feature contract. The cluster-aware standalone exceptions are AxBench-Base's
source/Stage-4 stage subset and BPB-Adv's dispatcher/sample replay. Both resolve
the contract before stage work; BPB-Adv preflights the clustered source before
marker installation and threads the contract through every stage. Its artifact
and identity layers have landed, and the two stale intervention guards that once
blocked the advertised direction/precomputed-bag scoring set are gone (#1008), so
the clustered score set is end-to-end runnable. A resumed run reuses committed
precomputed bags by identity rather than byte-comparing a nondeterministic derive
(#1050).
Cluster-aware prompt caches support
in-training CORE and sample generation. The completion TUI is the interactive
downstream consumer: it reads the canonical clusters and resolved mask from checkpoint
metadata, requires the matching percentile overlay, and recreates the ordinary
percentile-based selection path for display, resampling, and generation. It has
no `feature_clusters` Hydra key. Clustered checkpoints with
`feature_delta.mode=splice_counterfactual` are accepted only with
`neg_delta_ratio=1.0`, since evaluation has no splice provenance and therefore
reduces exactly to ordinary percentile selection. Other delta modes, partial
ratios, gradient ranking, or raw-magnitude thresholds still fail at TUI launch.
Ordinary/threshold clustered inline steering and other standalone eval paths
remain unsupported unless they consume the validated cluster-aware context
provider. This is an intentional boundary, not an implied compatibility claim.
See
[configuration.md § `feature_clusters`](../reference/configuration.md#feature_clusters--static-preselection-only-clusters)
for the full support matrix.

### Cluster-aware CORE/sample prompt cache

Prompt contexts have no splice current/reference pair. Their identity and
selection are therefore deliberately delta-free even when the training rows use
splice-counterfactual selection:

```text
raw extracted context segment in original SAE IDs
  -> validate IDs, magnitudes, and per-segment uniqueness
  -> resolved training feature mask
  -> each member magnitude -> normalized percentile using its own stats row
  -> representative + sibling amax(percentile)
  -> audited sampling policy with the ordered real-seed RNG stream
  -> percentile_to_bucket (no representative raw-magnitude reinterpretation)
  -> cluster-aware inactive representative universe
  -> cached original representative IDs
  -> FeatureEncoder._to_compact at model entry
```

Only sampling-based clustered caches are supported. The positive cache config is
the audited training positive selection with `delta_weights` RETAINED as an
explicit `null` (not removed -- `compute_selection_hash` hashes raw dict key
presence, so dropping the key entirely produces a different hash than keeping
it null, even though downstream parsing treats both identically); the negative
selection is unchanged and `feature_delta` is `null`. The identity hash is
pinned to `conditioning.seed` -- the checkpoint's real training-preselection
seed, not a hardcoded value -- so it agrees with the packed-row producer's own
delta-free selection identity for the same policy. The canonical cache hash
additionally includes the resolved mask and full cluster descriptor, but
**omits** the statistics-file SHA-256 (unlike the unclustered eval-cache
identity, which does fold it in): statistics identity is independently pinned
by the clustered feature contract and cache manifest instead
(`steering/eval/cache_identity.py`'s clustered branch;
`steering/eval/cluster_feature_contract.py`'s `ClusterFeatureContractRequest`/
`resolve_cluster_feature_contract` add the actual checkpoint-bound
statistics/mask/cluster/seed cross-check). The manifest repeats the
canonical mask/cluster state, effective-vocabulary metadata/fingerprint,
`cluster_selection_semantics=post_mask_representative_max_percentile_v1`, and
explicit `feature_delta=null`. Path/member reordering canonicalizes to the same
identity; changing one cluster membership does not.

At training startup the provider re-derives that hash from the top-level
training selection and authoritative resolved objects. A manually configured
hash must match. It also requires any redundant conditioning SAE width/bucket
count to equal the model values and rejects cached masked or
consolidated-away IDs before constructing entries. CORE and sample generation
then share the ordinary target-aware wrapper; no compact row ID crosses the
cache, log, or reporting boundary.

### Delta-aware clustered inline steering

Prompt caches deliberately remove splice delta. The periodic steering metric
does the opposite: it evaluates the actual training-conditioning distribution,
so it reads the derived/spliced validation `sel/<hash>/` overlay and its packed
provenance. A shared read-only validator binds selection content, packed rows,
provenance, sidecars, derived features, boundaries, mask, clusters, and the
effective vocabulary before the bounded two-pass scan begins.

Pass 1 considers only provenance rows belonging to the derived validation
parquet, rejects duplicate `(doc_idx, segment_ordinal)` keys, and selects unique
positive-bearing target documents plus deterministic different-document donor
segments with the fixed steering RNG. Pass 2 rescans only that validation shard
family and materializes the required target prefixes and donors. IDs/buckets
come from the selection overlay; authoritative character/token boundaries come
from an exact `FeatureIndex` join. Positive and negative conditioning rows stay
original representative IDs through generation.

For each positive representative, the payload also records all mask-surviving
original members that map to it. Raw pre/post diagnostics probe the
representative only. Percentile scoring probes the unique member union once,
uses each member's own statistics row, reduces by member maximum per
representative, and averages representatives so larger clusters receive no
extra weight. Every rank builds the same startup payload and all-gathers its
digest. During a cadence, ranks gather status before scores; an ordinary local
exception therefore becomes the same summarized failure on every rank.

For ordinary pretraining, run the existing packed-row step and then materialize
the exact selection identity used by training:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<feature_dir> packed_row_dir=<packed_row_dir> \
    +feature_clusters.path=<clusters.yaml> feature_delta=null \
    feature_selection.delta_weights=null
```

For packed `splice_counterfactual`, add the same
`+feature_clusters.path=<clusters.yaml>` override to **all three** selection
precomputations in the splice artifact sequence below, including the
delta-free base-validation twin. The resolved mask and cluster file must match
the subsequent training invocation exactly.

### Invariants

| ID | Invariant |
|----|-----------|
| I1 | No mask and no clusters ⇒ byte-identical `sel/<hash>/` path (existing shards stay valid) |
| I2 | No mask and no clusters ⇒ byte-identical checkpoint path |
| I3 | Base shards/stats/pct/tok/packed-rows are NEVER recomputed regardless of mask config |
| I5 | No mask and no clusters ⇒ bit-identical RNG draws and outputs everywhere |

> **Encoder compaction (#695) preserves all four — by construction, not by test.**
>
> - **I1/I2:** the vocabulary is a *pure, deterministic function of `(mask, canonical clusters, n_sae_features)`*. The mask and cluster descriptor already pin those semantics in `compute_selection_hash` and in the `bl_`/`wl_` plus `fc_` checkpoint leaves. The derived vocabulary therefore adds **no** identity key or path component.
> - **I3:** the remap lives at the *encoder boundary*. `sel/` shards and packed rows keep storing raw SAE ids (and keep relying on `uint16`).
> - **I5:** with no clusters, `effective_feature_vocab` literally delegates to `FeatureVocab.from_mask`; with neither mask nor clusters it returns `None`, and `_to_compact` **returns the very same tensor object**. The tables are full-width and `init_weights()` consumes an identical RNG stream.
>
> Compaction adds three of its own:
>
> | ID | Invariant |
> |----|-----------|
> | V1 | `effective_feature_vocab` is a pure function of `(mask, canonical clusters, n_sae_features)` and literally delegates to `FeatureVocab.from_mask` without clusters |
> | V2 | compact row `j` ⇔ original id `vocab.original_ids[j]`; `to_original(to_compact(id)) == id` for every id in the vocab |
> | V3 | a compacted model's logits are **bitwise identical** to a masked-but-uncompacted model's, given the compact tables are the row-subset of the full ones |
> | V4 | thanks to slice-init, a masked **compacted** run is **bitwise identical to a masked uncompacted run** at every step — same RNG stream, same loss trajectory, same surviving weights. **Holds for the default recipe (`fp8: false`, `initialize_encoder_embeddings_from_sae: null`); see [Byte-exactness](#byte-exactness) for the two documented exceptions.** |

For clustered construction the same contract is spelled out as E1–E7: E1 is
literal no-cluster delegation and the effective-universe definition; E2 keeps
the vocabulary derived-only (never a hash/path input); E3 preserves full
no-cluster behavior and RNG identity; E4 requires at least one effective id;
E5 preserves the minimum allowed id, so padding row 0 denotes the same original
id as in the mask-only reference; E6 gives
`V_eff = V_mask - consolidated_away_count`; and E7 keeps every surviving
representative while excluding every away id.

The strict non-empty base-segment invariant (issue #522 — a base extracted segment with zero stored features is corruption; see [feature-extraction.md § Shard Format](feature-extraction.md#shard-format-csr-sparse)) applies **only** to base extracted shards / `FeatureIndex` loads / the base validators. Downstream selection / delta / masking / percentile-filtered stats may legitimately produce empty conditioning segments and are intentionally **EXEMPT** from that check.

## Encoder Feature-Dictionary Compaction

*Issues #695/#714. Primitives: `steering/conditional/feature_vocab.py` and `effective_feature_vocab.py`. Applied in `steering/conditional/encoder_decoder/encoder.py`.*

### What it does

The encoder conceptually starts with one row per SAE feature across **every feature-id-indexed table** — `encoder.feature_embed` **and** each `encoder.value_embeds[i]`. Rows outside the mask universe, and allowed cluster members that preselection always consolidates into a representative, are **exactly-dead weight**: no legal batch can address them. Cross-attention also masks padding positions as keys, so padding gets zero contribution and exactly-zero gradient. Compaction stops allocating those unreachable rows.

> **Zero gradient alone would not be enough — the rows also have to be *frozen*.** AdamW's `weight_decay` is *decoupled*: it is applied to every parameter in the group regardless of gradient, so a non-zero decay would slowly shrink the dead rows and they would no longer be inert. Production is safe because `setup_optimizer` puts `enc_embed_params` in a **`weight_decay=0.0`** AdamW group — and that is now pinned by a test, because the whole "the deleted rows never mattered" argument rests on it.

`FeatureVocab` is the sorted, de-duplicated list of effective original SAE ids. The production constructor is a pure function of the resolved mask, resolved cluster consolidation, and `n_sae_features`: allowed ids minus `consolidated_away`. It sizes those tables to `V = vocab.size`, and ids are remapped **at the encoder boundary** — so **every caller, config, shard, eval and CLI still speaks ORIGINAL SAE feature ids**. Nothing upstream or downstream of the encoder changes.

| | |
|---|---|
| **Trigger** | Whenever the effective universe is smaller than the SAE dictionary: a restrictive blacklist/whitelist, active post-mask cluster consolidation, or both. **No config knob.** The default frequency band therefore makes the default run compacted. |
| **`n_sae_features`** | **Unchanged meaning: the ORIGINAL dictionary width.** Still the number every bounds-check uses (`eval_adapter`, `target_conditioning.validate_feature_dict`, `model_detection`, `harmfulness_eval`, `steering_eval`). Redefining it to `V` would silently corrupt all of them. |
| **`model.feature_vocab_size`** | `V` — the encoder's embedding-table row count. Equals `n_sae_features` when uncompacted. **These are no longer the same number.** |
| **`model.feature_vocab`** | The `FeatureVocab`, or `None` when full-width. Lives on `FeatureEncoder`; the model exposes it as a passthrough property. |
| **Scaling law** | **Unaffected.** `get_scaling_params` sums decoder tensors only and excludes the encoder entirely, so compaction does not move the training-token horizon or the batch-size/LR schedule. |

### Self-describing checkpoints

Two **persistent** buffers are registered on the encoder, and **only when compacted**:

| key | dtype / shape | role |
|---|---|---|
| `encoder.feature_vocab_ids` | `int64[V]` | the durable record — original id of each compact row. **Its ORDER is the row mapping**; a loader that re-sorted it would silently reinterpret every row, so an unsorted buffer is rejected. |
| `encoder.feature_vocab_n_sae` | `int64` scalar | the original width. **Not recoverable** from a compact table's row count, hence persisted. |

(A third, `encoder.feature_vocab_old_to_new` (`int64[n_sae_features]`, `-1` where out-of-vocab), is **non-persistent** — re-derived by `init_vocab_buffers()`, which `init_weights()` and `init_buffers()` both call.)

So a checkpoint reconstructs its own vocabulary **from the state dict alone** — no training config, no sidecar. `feature_vocab_from_state_dict()` is the one source of truth; `build_encdec_model` uses it. `meta_*.json` carries a `feature_vocab: {n_sae_features, vocab_size, sha256}` **fingerprint** mirror (for tooling — not load-bearing, and cross-checked against the buffers on load).

| checkpoint | vocab buffers? | behaviour |
|---|---|---|
| legacy (any, incl. masked) | no | identity vocab; the old row-count inference is still correct; `strict=True` passes unchanged |
| current, no effective compaction | no | **byte-identical to the full-dictionary path** |
| current, mask- and/or cluster-compacted | yes | effective vocab read from the file alone |

### Byte-exactness

Two exactness results hold, and they are what make this a pure refactor rather than a numerical trade:

- **No mask and no clusters ⇒ bit-identical (I5), by construction.** The constructor literally delegates to the old mask-only path, `_to_compact` early-returns *the same tensor object*, and the tables are full-width. Zero added ops.
- **Effective-compacted ⇒ bitwise identical to the corresponding mask-only reference (V3/V4, extended to clustered effective compaction).** `_init_feature_indexed_` uses **slice-init**: it draws the **full** `(n_sae_features, d)` table — consuming the identical RNG stream — and then keeps only the vocab's rows. The pre-PR-1 clustered reference keeps every mask-allowed row; current code removes only consolidated-away rows that representative-only data never addresses. Shared feature rows, every non-feature parameter, forward loss/logits, gradients, AdamW state, and post-step weights are bitwise equal in single-process training. Cost: one transient full-width allocation per table at init, freed immediately; nothing at train time.


#### Where byte-exactness does NOT hold

There are two independent boundaries: FP8 can observe a different padding row,
while SVD deliberately derives a different basis from the selected dictionary.

1. **FP8 (`fp8: true` — `4xh100`, `4xh200`, `2xh200`, sweep).** Not bitwise identical when SAE id 0 is outside the vocabulary. The globally packed CUDA path carries at most 31 tail entries with id `0` (the CPU reference pads each row); `_to_compact` maps those entries to compact row 0 — i.e. to `feature_embed[original_ids[0]]` — whereas the uncompacted reference embeds `feature_embed[0]`. Those are the same vector only when SAE id 0 is in the vocabulary. torchao's **tensorwise** scaling reduces over the whole activation (`tensor_to_amax` is literally `torch.max(torch.abs(x))`), and those tail entries are part of that tensor — so a different padding row can yield a different `amax`, a different quantization scale, and therefore different quantized values for every *real* activation.

   This is **not fixable and not a correctness defect.** A compacted table has no row for SAE id 0 to reproduce; forcing id 0 into the vocabulary would blunt the out-of-vocab guard, and zeroing padded embeddings would break the no-mask path's byte-identity (I5). The *uncompacted* model's padding row is itself an arbitrary never-trained vector, so neither run is "more correct" — the two simply round differently in fp8. Padding still contributes **exactly zero** to the output and **exactly zero** gradient in both.

   The same boundary applies verbatim to **cluster-driven** compaction (which the V1 training gate does not forbid combining with `fp8: true`): the effective table has a different row count than the mask-only reference, so the tensorwise `amax` reduces over a different tensor, and a clustered fp8 run is not bitwise against its mask-only-vocabulary twin either. Everything above about zero output/gradient contribution holds unchanged.

2. **SAE-init `svd` (`initialize_encoder_embeddings_from_sae: svd`).** Not bitwise identical, and irreducibly so for a reason unrelated to padding: the eigenbasis is the covariance of the rows you feed it, and requirement 6 feeds it the **sub-dictionary** deliberately. The principal directions are those of the dictionary you will actually train on, which is the point — but it means a compacted `svd` init cannot equal the row-subset of an uncompacted one. With active cluster vectors `mean`, oriented `W_dec` is averaged before that subset, so covariance and basis are additionally computed over the averaged effective dictionary; `representative` remains bitwise the pre-option SVD path.

   **`random` + cluster vectors `representative` is not a boundary — it is bitwise.** It projects the full `(n_sae, d)` matrix, rescales against the **full** table's average row norm, and subsets **last**, so it executes the identical GEMM and identical rescale as the uncompacted reference. (Earlier it rescaled against the *V-row* table's norm — a sample estimate off by ~2.1e-04 at the real shape — which let compaction silently perturb the init. Compaction is purely representational; it must not.) Cluster vectors `mean` (the default, active only when clusters are configured) deliberately changes only surviving representative rows after that identical rescale; non-cluster rows remain bitwise.

**DDP collective-path scope.** `DistMuonAdamW` chooses all-reduce versus
reduce-scatter/all-gather per parameter from its element count and leading-row
divisibility. `V_mask` and `V_eff` can therefore route the same logical embedding
through different collectives. The claim is single-process bitwise identity always,
and DDP bitwise identity when both shapes choose the same path. At two ranks the
reduction order is identical even when the paths differ (covered by the deferred CPU
Gloo smoke); at three or more ranks, especially under NCCL algorithm selection,
cross-path results are only mathematically equivalent and no bitwise claim is made.

The FP8 and `svd` boundaries are pinned by tests in `tests/steering/test_encdec_compaction_byte_exact_e2e.py`, so they cannot drift silently. The part that *does* hold is proven where production actually runs it: a CUDA test builds both models **bf16-on-CUDA with the production double `init_weights()`** (the second of which slice-inits over already-`COMPUTE_DTYPE`-cast tables, so the RNG draw is bf16, not fp32).

> **Meta-device construction is load-bearing here, not just an optimisation.** `nn.Embedding.reset_parameters()` draws `numel()` randoms *at construction*. On a real device a compact table would therefore draw fewer randoms than a full one and diverge the stream *before* `init_weights()` ever runs. Production is safe two ways: models are built on `meta` (which draws nothing) and the training scripts `manual_seed` **after** construction.

### The out-of-vocabulary guard

Every producer pads `enc_feature_ids` with literal `0` — a real SAE id a whitelist may exclude — and CUDA global packing adds an id-0 alignment tail (the CPU reference retains its row-wise `F.pad`). The remap is therefore **mask-aware**: in-range padding slots map to compact row `0`. `searchsorted` uses clamped indices for a memory-safe vocabulary gather, then checks exact membership against the original ID. Active non-members fail the asynchronous assertion; the returned row is `where(mask & hit, pos, 0)`, never `-1`. Negative or out-of-SAE-range IDs fail even in padding slots. Clamping protects the gather; the membership assertion enforces vocabulary correctness.

The guard is `torch._assert_async`. This is not belt-and-braces:

> **`nn.Embedding(-1)` raises `IndexError` in eager but SILENTLY RETURNS THE LAST ROW under `torch.compile`** (Python negative-index wraparound). A `-1` sentinel alone is not a sufficient guard.

`torch._assert_async` is sync-free (no host sync on the hot path), produces zero graph breaks under `fullgraph=True`, and faults in **both** eager and compiled paths. The trade-off, taken deliberately: it **cannot name the offending ids** — that would need `bool(ok.all())`, a host sync on every eager CUDA forward. The message instead names `V`, `n_sae_features`, and the `dump_feature_mask` remedy. On CUDA under compile it surfaces as a device-side assert and may be truncated by the CUDA runtime. Rich diagnostics live in the once-per-run validators (the dataloader's I4 `assert_no_masked`), which can afford a sync.

### Recovering a checkpoint's mask — `dump_feature_mask`

For SFT and evaluation surfaces that require an explicit downstream mask, use the exact training mask. The default frequency blacklist can otherwise depend on a **generated** `{feature_dir}/blacklist/freq_min*_max*.txt` that a `feature_dir` move or statistics recompute would destroy. The checkpoint-derived `chat_eval_turn` surface reconstructs the canonical mask and clusters from metadata and proves equality with the loaded vocabulary at startup; it does not require re-specifying the mask. Fixed `chat_eval` retains its deferred compaction stance.

```bash
uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir> [--step N] [--format yaml|txt]
```

It recovers the mask from the checkpoint **alone** and prints a paste-ready
`feature_mask:` block to stdout (provenance goes to stderr and `#` comments).
For unclustered checkpoints it retains the original byte-for-byte behavior:
`meta.feature_mask_canonical` round-trips through `FeatureVocab.from_mask` against
the persistent ids. For clustered checkpoints it strictly rebuilds the canonical
definition/descriptor, applies it to the recovered mask, and checks the current
`effective_feature_vocab` first. A mask-only vocabulary is also recognized and
labeled as a checkpoint written by pre-effective-compaction code. If neither
candidate matches, it refuses as corrupt and names both candidate sizes. When
clusters are inert after the mask, both candidates coincide; effective matching
wins and provenance says the configured clusters consolidate nothing.

If unclustered metadata carries no mask but the tables are compacted, it recovers
the mask **including its mode** from the checkpoint path as before. It builds
whitelist and complement-blacklist candidates, then searches the given and resolved
paths and every ancestor for the matching `bl_`/`wl_` leaf.

Cluster evidence changes that rule. An effective vocabulary cannot distinguish an
id excluded by the mask from one consolidated into a representative, so the tool
intercepts cluster metadata or an `fc_<64hex>` path leaf **before** vocab-derived
reconstruction. A missing mask plus cluster metadata and a mask leaf is an
instructive thin-metadata error; cluster metadata with no mask leaf is cluster-only
compaction, so there was no mask to emit; and an `fc_` leaf without canonical cluster
metadata is positive evidence of an ambiguous thinned checkpoint. All three refuse
rather than invent a whitelist.

### Compaction compatibility and guards

Supported paths preserve original-ID semantics; unsupported paths fail loudly.

| Situation | Result |
|---|---|
| **FP head + compaction** | Supported for precomputed pretraining and conditioned SFT. The head and targets use `V_effective`; public IDs remain original representatives. Generic live pretraining compact-FP loading is rejected. |
| **Self-conditioned generation / `return_fp_logits` on a compacted model** | Direct `return_fp_logits`/`fp_logits_only` returns `V_effective` columns. Inference-time self-conditioning translates compact head row `j` to `feature_vocab.original_ids[j]` before selection and hands original ids to the encoder. Head/vocabulary/config widths and persistent vocab buffers are checked before generation. The packed-pretraining post-training source remains uncompacted-only; source-aware SFT supports exact compact mask/cluster identity and compact-row STE. |
| **Universal `feature_gcg` on a compacted model** | Supported for one or many members. Proposals are restricted to the checkpoint vocabulary `V` in **original SAE-ID space**. The edit space translates original IDs to compact `id_weight` rows only while scoring; returned and persisted bags retain original IDs. Static clustered checkpoints require a validated cluster feature contract at the low-level Feature-GCG entry points; without one they remain rejected. AxBench-Base/BPB-Adv can resolve that contract, but BPB-Adv's advertised clustered score chain still hits a stale precomputed-bag intervention guard. |
| **Resuming a legacy MASKED checkpoint into a compacted run** | Raises `FeatureVocabError` with an actionable message *before* `load_state_dict` (`assert_checkpoint_vocab_compatible`) — PyTorch's bare `size mismatch` tells an operator nothing. Also catches a compacted checkpoint loaded with no mask configured, and the case where ids and `V` match but `n_sae_features` does not. |
| **Resuming a pre-effective clustered checkpoint** | After cluster metadata matches, a dedicated guard recognizes the old mask-only vocabulary and raises `FeatureVocabError` naming consolidated-away rows and the start-fresh remedy. It runs before the generic vocab guard; no migration tool exists. |
| **A per-config mask or cluster consolidation that differs from the shared model's** | Raises. The encoder's tables are sized once per model, so the effective vocabulary cannot vary across configs sharing one. |
| **SFT from a checkpoint with the wrong mask/cluster universe** | Raises `FeatureVocabError` **at startup**. A compacted base requires exact effective-vocabulary equality. An uncompacted clustered base always raises; a pre-compaction masked base passes only when present canonical mask metadata exactly matches the configured mask. |
| **SFT/selfcond/eval/GCG from a clustered checkpoint** | SFT is supported when the explicit clusters and mask agree bidirectionally with base/resume metadata and their effective vocabulary equals the state dict; siblings are folded inline before selection and into compact FP targets. Both clustered sampling and threshold selection consolidate full-distribution percentile ranks from the statistics table, then use uniform percentile buckets. An already-loaded compact model can self-condition at inference because excluded siblings have no head row and candidate ids come from its self-describing effective vocabulary; no runtime cluster reduction is inferred. Source-aware SFT post-training supports that same exact cluster/vocabulary contract and compact-row STE, while the packed-pretraining source remains uncompacted-only. Generic eval loaders that still declare a deferred compaction stance can reject the checkpoint before adapter construction. Feature-GCG rejects clusters unless a validated cluster feature contract is supplied to a low-level entry point. AxBench-Base supports a stage/method-limited source and Stage-4 subset; BPB-Adv resolves and threads the contract through every stage and clustered sample replay. |
| **Fast in-process sweep + active cluster vectors `mean`** | Unreachable by design. `run_encdec_sweep_fast.py` rejects `feature_clusters` during shared setup and again for each composed config. The default `mean` policy without clusters remains an inert supported no-op; use the ordinary pretraining entry point for active averaging. |
| **The 315-config sweep (`conditional_encdec_sweep.yaml`)** | Its compact FP model shapes now compose. Any compact-FP cell still requires precomputed rows carrying the new source-integrity certificate. |

### The equality rule: downstream conditioning must match the checkpoint vocabulary

Under compaction the vocabulary is a property of the **checkpoint**, not of the downstream config. SFT *inherits* the dictionary and can neither create nor widen it.

This equality section applies to SFT's complete effective mask/cluster universe.
The narrow clustered AxBench paths use their stronger artifact-level feature
contract. `chat_eval_turn` uses a checkpoint-derived contract: canonical mask and
cluster reconstruction must equal the loaded checkpoint contract and vocabulary.
The explicit-mask advice below applies to SFT and other surfaces that require
operator-supplied masks, not checkpoint-derived turn ChatCORE.

> **The resolved allowed universe must be EQUAL to the checkpoint's vocabulary. Not a subset. Nothing is inferred.**

- **Equality, not subset.** A subset would be *structurally* safe — every id would still have a row — so this is deliberately stricter than the structural minimum. It is the consistent choice: it matches `assert_resume_feature_mask_matches`, which already demands exact equality on resume.
- **`feature_mask: null` against a compacted checkpoint RAISES.** The checkpoint's vocabulary is never silently adopted as the downstream universe.
- **Uncompacted checkpoints are enforced too.** No mask and an allows-all mask remain the byte-identical legacy cell. A pre-compaction masked base may continue only when its present `feature_mask_canonical` metadata exactly matches the configured compacting mask; differing or absent metadata raises with the `dump_feature_mask` recovery route. Configured clusters always raise against an uncompacted base because cluster-trained bases are compacted by construction.

Enforcement, and where it is **not** yet enforced:

| Path | Check |
|---|---|
| **SFT** (`conditional_encdec_sft.py`) | ✅ **At startup.** `_assert_sft_universe_matches_checkpoint_vocab` compares `effective_feature_vocab(consolidation if clustered else mask, n_sae_features)` against compacted state-dict vocabularies and uses canonical mask metadata to verify pre-compaction masked bases. Base and resume cluster metadata are checked bidirectionally; clusters never infer semantics from an uncompacted base. `n_sae_features` and the checkpoint vocab both come from `feature_vocab_from_state_dict`, and the model is built with that checkpoint vocab. |
| **Pretrain resume** (`conditional_encdec_train.py`) | ✅ **Before `load_state_dict`.** `assert_checkpoint_vocab_compatible` + `assert_resume_feature_vocab_matches`. |
| **Turn ChatCORE** (`chat_eval.py`, `chat_eval_turn` surface) | ✅ **At startup.** `assert_conditioned_eval_model` verifies the loaded checkpoint feature contract, exact effective vocabulary and persistent buffers via `model.assert_vocab_consistent()`. `resolve_turn_context_contract` reconstructs canonical mask/cluster metadata as an equality assertion, then adopts the loaded contract. Providers validate all cached IDs; target construction and encoder packing validate original SAE IDs against that vocabulary. Fixed checkpoints still use `EvalConditioning.deferred("chat_eval")`. |
| **Generic/unclustered eval** (`base_eval.py`, completion-steering, unclustered AxBench Stage 4, …) | ⚠️ **No startup equality check yet.** The model still *loads* correctly — `build_encdec_model` recovers the vocab from the state dict — but a wrong/missing eval mask is only caught when a masked-out feature is actually conditioned on, by the encoder-boundary `torch._assert_async` (loud, at the first forward, without an id list). Clustered AxBench is the exception: its stronger feature contract validates mask/cluster/checkpoint/selection agreement at startup. Hardening every other conditioning entry point is a deferred follow-up. |

For surfaces requiring explicit masks, use `dump_feature_mask` (above) rather than depending on the *generated* `{feature_dir}/blacklist/freq_min*_max*.txt` surviving on disk. Checkpoint-derived turn ChatCORE instead requires complete canonical metadata and exact loaded-vocabulary equality.

### Producers of compacted checkpoints

Three scripts build encoder-decoder models and are all vocab-aware:

| Script | Vocab source |
|---|---|
| `conditional_encdec_train.py` | the run's resolved mask plus static-cluster consolidation (`effective_feature_vocab`), both built **before** the model. Logs the realized row/parameter saving and the number consolidated away, then writes the vocab fingerprint and canonical cluster metadata. |
| `conditional_encdec_sft.py` | **inherited from the base checkpoint** (`feature_vocab_from_state_dict`). SFT can neither create, widen, nor narrow the dictionary; its explicit mask/cluster effective universe must *equal* it (see above). |
| `convert_decoder_to_encdec.py` | the config's `feature_mask`, resolved **once** and used for all three of: the encoder's vocab, the `bl_`/`wl_` checkpoint leaf, and `meta.feature_mask_canonical` + `meta.feature_vocab`. Static clusters are rejected immediately after Hydra composition; conversion is not a supported cluster surface. |

### Not touched

`sel/` shards, packed rows and every other producer keep emitting **raw SAE ids** (I3). `magnitude_embed` is bucket-indexed, not feature-indexed, so it is full-size regardless. `encoder_io.build_no_feature_inputs` emits an all-masked `(1,1,1)` input carrying feature id `0`; every slot is padding, so it is exactly-dead and cannot trip the vocab guard even when id `0` is out of vocab.

## Offline topology migration (issue #1514)

`python -m steering.tools.migrate_encdec_train_state` converts one complete,
native encoder-decoder checkpoint leaf between DDP world sizes without changing
`nanochat/`. The currently supported production handoff is the BF16 W31 preset
`full3b_clustered_c2048_long_w31` to
`full3b_clustered_c2048_long_w32_bf16_b14`. The destination preset changes only
`run`, `model_tag`, and `total_batch_size`:

| Geometry | World | B/rank | Context | Accumulation | Tokens/update | Updates |
|---|---:|---:|---:|---:|---:|---:|
| Source | 31 | 14 | 2,048 | 2 | 1,777,664 | 250,000 |
| Destination | 32 | 14 | 2,048 | 2 | 1,835,008 | 250,000 |

The raw checkpoint step and the 250,000-update schedule phase are unchanged.
The destination's canonical optimizer groups are rebuilt from the destination
batch size, so every batch-scaled LR and the initial weight-decay scale increase
by `sqrt(32/31)`; this is intentional. The inherited
`target_param_data_ratio` is unchanged with the rest of the schedule inputs,
while the literal W32 horizon is 458,752,000,000 tokens.

The component contracts are:

- **Model:** the native `model_NNNNNN.pt` file is copied byte-for-byte after its
  complete state-dict key/shape/dtype contract is checked against both configs.
  This reproduces the production CUDA initialization policy explicitly:
  decoder token/value embeddings and encoder feature/magnitude/value embeddings
  are BF16, while all remaining parameters and persistent buffers retain their
  declared dtypes. Validation does not depend on the offline CPU host's
  auto-detected compute dtype.
- **Optimizer:** source parameter groups must exactly equal the groups the
  trainer would have saved at the previous completed update. AdamW state is
  reassembled and re-emitted according to the destination's replicated versus
  leading-dimension-sharded rule; replicated source copies must be bitwise
  equal. Muon rows are reassembled in parameter order, repartitioned with
  `ceil(K/world)`, and destination padding is newly zero-filled. Per-parameter
  Adam steps and every logical state tensor are preserved exactly; only the
  destination's canonical group hyperparameters change.
- **Dataloader:** exact cross-topology continuation is deliberately not
  claimed. Saved `batch_rows` are the **next** batch to train. The migrator
  chooses row zero of the packed shard containing the earliest rank-local next
  row, then emits ordinary destination-striped pending batches from that
  boundary. This can repeat already-consumed rows but cannot skip a row. The
  manifest records the exact boundary, source frontiers, repeats per source
  rank, total repeated rows/tokens, and `skipped_rows: 0`. Use
  `max_replay_rows=N` as an operator-set upper bound.
- **Artifacts/RNG:** the supplied selection directory must be the canonical
  `<packed-row-dir>/sel/<selection-hash>` leaf, its manifest checksum must match
  that leaf, the manifest must bind the exact supplied packed-row bytes, and the
  shard inventories must match. This path is
  restricted to precomputed selection, `label_dropout=0`, and local (not
  `remote_training`) checkpoints. Feature-selection RNG streams are inert on
  this data path and are deterministically re-seeded for destination ranks;
  cross-topology stochastic-stream continuity is out of scope.

Dry-run is the default and performs all reads, content hashes, model/config
construction, group reconstruction, and tensor validation without creating the
destination. Hydra's operational runtime stanza is normalized out through the
same checkpoint-config path the trainer uses, including when the destination
preset is nested inside the tool config. Execute mode requires a nonexistent
destination, writes a private sibling staging directory, copies the model with
SHA-256 verification, writes exactly one optimizer shard per destination rank
plus metadata and an audit manifest, fsyncs them, and atomically renames the
complete directory into place. The source is never modified. See the
[CLI reference](../cli-reference.md) for commands.

The offline suite proves tensor-exact optimizer round trips and all four AdamW
ownership transitions, exercises Muon padding/corruption, property-tests the
no-skip replay rule, feeds synthesized states through the real loader, runs a
native checkpoint prepare/write integration, and compares the next real
`DistMuonAdamW` Gloo update across W2/W3. A real W31 checkpoint dry-run and the
20-step W32 loss/gradient/checkpoint-completeness canary remain mandatory
operator gates before production continuation.

## Optimizer

The encoder-decoder model uses `MuonAdamW` with separate parameter groups for
its model components, allowing independent learning rate tuning:

| Component | Parameters | Optimizer | Default LR |
|-----------|-----------|-----------|------------|
| **Decoder** embeddings (wte, value_embeds) | Token + value embeddings | AdamW | 0.2 * scale |
| **Decoder** unembedding (lm_head) | Output projection | AdamW | 0.004 * scale |
| **Decoder** scalars (resid_lambdas, x0_lambdas) | Per-layer gates | AdamW | 0.5 / 0.005 |
| **Decoder** matrices (attention, MLP) | Block weights | Muon | 0.02 |
| **Feature head** projection + non-matrices | Retained output projection and any residual-block scalars | AdamW | 0.004 * scale |
| **Feature head** matrices (attention, MLP) | 2-D residual-block weights | Muon | 0.02 |
| **Encoder** embeddings (feature_embed, magnitude_embed, **value_embeds**) | Feature + bucket + encoder value embeddings | AdamW | 0.2 * scale |
| **Encoder** matrices (attention, MLP) | Block weights | Muon | 0.02 |
| **Cross-attention** matrices (c_q, c_kv, c_proj) | All cross-attn weights | Muon | 0.02 |

The encoder has its **own** `value_embeds` (`encoder.value_embeds`, ResFormer-style, one table on every other encoder layer via `has_ve`) — distinct from the decoder's token-indexed `value_embeds`. They are **feature-id-indexed**, they live in the encoder-embeddings AdamW group (`model.py::setup_optimizer` → `enc_embed_params`), and they are collectively the **largest** block of feature-id-indexed parameters in the encoder — bigger than `feature_embed` itself. Compaction (#695) shrinks all of them; the group is a plain AdamW group (not shape-grouped like the Muon groups), so no regrouping is needed.

The `scale` factor is `(model_dim / 768)^-0.5` (muP-style LR scaling). Decoder LRs are further adjusted by batch-size-dependent scaling `(batch_size / B_REF)^0.5`.

All per-group LRs — including encoder embeddings, encoder matrices, and cross-attention matrices — are scaled by `batch_lr_scale = (total_batch_size / B_REF)^0.5`. The base LRs for encoder and cross-attention components are configurable via `encoder_embedding_lr`, `encoder_matrix_lr`, and `xattn_matrix_lr` in the training config.

Weight matrices are grouped by shape for efficient batched Muon updates. Shape-identical groups across components are always merged when LR, momentum, Newton-Schulz steps, beta2, weight decay, dtype, and device all match; groups with different update semantics stay separate. The optimizer verifies that every model parameter belongs to exactly one group. This is the only supported layout. Optimizer state written by the older per-component layout cannot be resumed; start a fresh run or load model weights without that optimizer state.

For the planned 3.02B-parameter, depth-27, width-1664 encoder-decoder at logical world size 64, this reduces the Muon layout from 12 groups and 17.824 GiB padded FP32 payload to 6 groups and 9.902 GiB without changing the 7.798 GiB real payload. Engaging jobs measured the exact payload on four H100s with PyTorch 2.13.0, CUDA 13.0, and NCCL 2.29.7. Across three alternating-order trials, the median-of-trial-medians fell from 86.296 ms to 47.844 ms on one node (job 20580339) and from 1,272.388 ms to 706.820 ms across two nodes with GDR and PXN disabled (job 20592478). Both placements show a 44.5% transport-time reduction (1.80x throughput). Physical collectives fell from 24 to 12 and peak allocated memory from 22.285 GiB to 12.379 GiB. The GDR-disabled cross-node route remained about 14.8x slower than same-node NVLink for both layouts.

Full 412M-model A/B trials were neutral, as expected for physical world size 4: coalescing reduced physical collectives from 52 to 40 but padded payload by only 3 KiB. On 4x1 H100 (job 20592479), the median-of-trial-medians was 1,293.145 ms and 405,436 tokens/s before versus 1,293.637 ms and 405,282 tokens/s after. On 2x2 with GDR and PXN disabled (job 20592480), it was 1,367.481 ms and 383,396 tokens/s before versus 1,366.935 ms and 383,549 tokens/s after. Trial-paired effects changed sign, so these are no-regression results, not a measurable four-GPU training speedup. Forward and backward times were placement-invariant, while optimizer time rose from roughly 15-16 ms on 4x1 to 90-91 ms on 2x2, accounting for the approximately 5.4% multi-node throughput penalty. The exact planned-world-64 transport measurements above are the performance evidence for this change; end-to-end world-64 throughput remains unmeasured until the target allocation is available.

## Label Dropout (CFG-Style Conditioning)

When `label_dropout > 0`, entire segments have their encoder mask set to `False` with the given probability during training. This is an encoder-feature robustness knob; current eval/generation still uses full-context conditioning.

Default is 0.0 (no dropout). Set via `label_dropout=0.1` in the training config.

### Conditioning cooldown

Enable `conditioning_cooldown.enabled=true` to linearly reduce the expected
fraction of retained conditioning during encoder-decoder pretraining or a continuation
of a pretraining checkpoint. The default schedule retains 100% of segments at
its start and 50% at its end, over the final 20 billion nominal training tokens.
It is disabled by default, preserving existing constant `label_dropout`
behavior, selection artifacts, and checkpoint identities. Active cooldown
requires `label_dropout=0`; the trainer rejects two simultaneous dropout controls.

```yaml
conditioning_cooldown:
  enabled: false
  start_fraction: 1.0
  end_fraction: 0.5
  start_step: null
  end_step: null
  cooldown_tokens: 20000000000
  granularity: segment  # segment | document
```

The x-axis is the **absolute, zero-based optimizer-update index** already used
by the trainer. With `num_iterations=N`, updates run at `0, ..., N-1`; a saved
checkpoint at step `S` has completed `S` updates and resumes update `S`.
Both endpoints are inclusive: the update at `start_step` uses exactly
`start_fraction`, and the update at `end_step` uses exactly `end_fraction`.
Between them the retained probability is
`start_fraction + (end_fraction - start_fraction) * (step - start_step) / (end_step - start_step)`.
Before and after the interval it holds the corresponding endpoint value.
Fractions are finite numbers in `[0, 1]`; equal fraction endpoints are permitted.
Step endpoints must satisfy `0 <= start_step < end_step < num_iterations`.

A null `end_step` resolves to `num_iterations - 1`. A null `start_step` resolves
backward from the end to include `ceil(cooldown_tokens / total_batch_size)`
updates, so the nominal duration is rounded upward to a whole global batch.
`total_batch_size` counts decoder training tokens across all ranks and all
gradient-accumulation microbatches. It does not count extraction-model tokens,
only selected features, or only retained segments. Supplying `start_step` makes
the step endpoints authoritative instead of `cooldown_tokens`. The inferred
window must contain at least two updates and fit the configured horizon; a
short run fails instead of silently truncating the requested 20B-token window.
Startup reports the resolved interval and nominal token count.

The y-axis is the **expected retained fraction of real units per microbatch**,
not an exact segment, document, or token quota. In `segment` mode the units are
all real segments (`seg_doc_id >= 0`), including real segments with no selected
features; padding is excluded. In `document` mode each row-local packed text
document counts once and all its segments share one decision. Original and
spliced portions of that document remain indivisible. Different units use
independent Bernoulli draws.

Splice **junction segments** are last in line for dropout. The authoritative
`_splice_provenance` sidecars identify these segments; the rest of a spliced
suffix has no special protection. In document mode, a document is protected
when any represented conditioning segment is a junction. Protection follows
the source feature segment, including `s + 1` when
`next_segment_conditioning_offset=true`. A cropped-away junction does not
protect the visible prefix unless its features are actually used by this offset.

For a microbatch with `N` real units, `M` protected units, and requested retained
fraction `r`, the expected retained count is `K = r * N`. When `K >= M`, all
protected units stay and ordinary units use probability `(K - M) / (N - M)`.
When `K < M`, ordinary units all drop and protected units use probability
`K / M`. If only one kind of unit is present, its probability is simply `r`.
Thus ordinary units are exhausted first, while the overall expected fraction
still follows the schedule. These counts are per rank's consumed microbatch,
not global quotas across accumulated batches or DDP ranks.

Both packers consume a document in one row and may discard a cropped tail;
there is no carried remainder needing cross-row dropout state. Packed training
joins the existing `provenance/` overlays to splice sidecars using the manifest's
source-shard order. Known spliced inputs require these artifacts even when
`feature_delta=null`; missing or malformed joins fail before training on that
batch. Live training reads the same sidecars directly by source document.

The trainer applies cooldown to the consumed batch immediately before forward,
after loader prefetch and optional CUDA staging. All microbatches of an optimizer
update use the same scheduled fraction, with independent draws. Only the
encoder mask changes: decoder inputs/targets, segment routing, selected IDs,
magnitudes, and feature-prediction targets remain unchanged. Validation and
generation continue to use their existing full conditioning. The constant
`label_dropout` path retains its historical empty-row fallback; cooldown runs
after that fallback and never reactivates a feature that its own decision
dropped. Retention zero therefore leaves the conditioning mask entirely false.

Cooldown uses a private generator keyed by training seed, absolute update,
rank, and microbatch index. It does not consume the process or feature-selection
RNG. The resolved schedule, seed, global batch, device batch, world size,
context length, training horizon, device type, and fixed `junctions_last` splice
policy are saved in the top-level
`conditioning_cooldown` record. A cooldown checkpoint must resume with matching
recorded settings; disabling the schedule, changing its endpoints, or changing
execution geometry is rejected. Packed replay reproduces the dropout masks
for identical data and geometry. Active loader states additionally bind the
splice-sidecar content; packed states also bind the manifest and provenance
overlays. Changing those inputs on resume or before a lazy read is rejected.
The first identity check reads all provenance-overlay and sidecar bytes;
subsequent checks reuse content fingerprint caches while file identities match.
The loader checks joins against already-loaded packed rows, without an additional
scan of packed tensor bodies. Remote packed loading uses its existing requirement
that the manifest and complete provenance-overlay set be staged locally; cooldown
also requires the matching local splice sidecars. It does not download a missing
provenance family. The live loader's documented approximate
data resume remains approximate. The offline topology-migration tool does not
migrate cooldown checkpoints.

To begin a new cooldown from a legacy local checkpoint, the resumed step must
be no later than the resolved schedule start. Preserve the original checkpoint
and stage its complete model, metadata, and optimizer files under a distinct
continuation `model_tag` using the normal artifact workflow. The initial
resumed checkpoint is not rewritten before an update. Review the source
checkpoint, data identity, optimizer settings, LR schedule, global batch, and
absolute final horizon before launching. For example, after that preparation,
an explicit interval is:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.conditional_encdec_train -- \
    config_name=<matching-source-training-config> \
    model_tag=<prepared-continuation-tag> resume_from_step=100000 \
    num_iterations=110000 label_dropout=0.0 \
    conditioning_cooldown.enabled=true \
    conditioning_cooldown.start_step=100000 conditioning_cooldown.end_step=109999 \
    conditioning_cooldown.start_fraction=1.0 conditioning_cooldown.end_fraction=0.5 \
    conditioning_cooldown.granularity=segment
```

Those example endpoints define 10,000 updates; their token count depends on
the resolved global batch and is not a claim of a 20B-token recipe. To use the
default duration, leave both endpoints null and configure the reviewed absolute
horizon accordingly. Enabling cooldown changes remote semantic identity:
resuming an old remote lineage with it enabled is rejected. A new continuation
requires a separately prepared lineage; this feature does not rewrite remote
markers or provide a remote branching tool.

Conditioned SFT preserves the cooldown provenance in `pretrain_conditioning`
through subsequent SFT checkpoints. It does not apply the pretraining schedule
during SFT. Self-conditioning and structured-SFT GRPO retain their existing
zero-dropout-lineage restrictions: a schedule with either retained endpoint
below one is rejected, even when the scalar `label_dropout` is zero. Relaxing
that post-training contract requires a separate compatibility review.

## Testing the Dataloader

The encoder-decoder dataloader has comprehensive test coverage across several files:

```bash
# Run all dataloader-related tests
uv run pytest tests/ -k "encdec_dataloader or char_level_segments or encdec_dropout_negative or encdec_prefetch_statedict or row_segments or token_segment or char_offset or char_coordinate" -v
```

| Test file | Coverage |
|-----------|----------|
| `test_encdec_dataloader.py` | Shapes, seg_map boundaries, mask consistency, sampling/negative pipeline integration |
| `test_encdec_dataloader_edge_cases.py` | `_pad_segments`, `_build_row_segments_stride`, and `_stack_batch_parts` boundary conditions; contract asserts for cross-shard fp_targets padding |
| `test_precomputed_loader_cross_shard.py` | Cross-shard batch reconstruction shapes; fp_targets shape coherence with `enc_mask`; variable-`n_seg` regression coverage |
| `test_encdec_train_cross_shard_smoke.py` | End-to-end fp-loss training step with cross-shard variable-`n_seg` batch (the production failure scenario) |
| `test_encdec_row_segments.py` | Stride-based segment assignment: single/multi-doc, cropped docs, None features |
| `test_char_level_segments.py` | Character-level segment assignment: hypothesis property tests, multi-byte chars, BOS assignment, cummax monotonicity, empty segments |
| `test_encdec_dropout_negative.py` | Label dropout (0.0/0.5/1.0, segment-level granularity), negative sampling rejection (no overlap, uniqueness, range, bucket index) |
| `test_encdec_prefetch_statedict.py` | `_prefetch_wrapper` FIFO ordering, error propagation, thread termination, state dict keys, prefetch vs non-prefetch equivalence |
| `test_precomputed_cuda_staging.py` | Issue-#842/#868 strict config/eligibility, exact H100 preset/fingerprint parity, deterministic staged-versus-four-batch N+1 decode lead, byte backpressure, static CPU schema, failure/cancellation lifecycle, CUDA source/destination lifetime, and exact legacy output/state parity across same/cross-shard, cross-epoch, rank, FP, and resume cases |
| `test_precomputed_cuda_staging_ddp.py` | Opt-in two-rank NCCL gate using real cross-shard packed rows: exact legacy tensor/state parity on both explicit devices, deep bounded-queue stress, production B/T/n_seg/F/V dimensions, bounds checks, BF16 backward, and static compile. Its separate issue-#868 `torchrun` gate requires eight H100s plus immutable production packed/selection directories and covers exact per-rank parity through three boundaries, exact checkpoint continuation, the 512 MiB/three-slot bound, early cancellation, repeated close, and terminal workers |
| `test_token_segment_assignment.py` | `assign_tokens_to_segments` overlap logic |
| `test_bos_char_offset_consumers.py` | BOS-inclusive coordinate system consistency |
| `test_char_coordinate_invariants.py` | Pretraining vs SFT coordinate spaces |

### Testing the 4090 Config and Memory Gate

`tests/steering/test_encdec_memory.py` has two tiers:

- Fast CPU/meta tests compose `conditional_encdec_4090` and assert the exact
  production architecture resolved by
  `build_encdec_meta_model_from_training_cfg`, including encoder ratios,
  cross-attention layers, `num_buckets`, feature prediction, and static
  feature-width bounds.
- Slow CUDA tests are marked `@pytest.mark.slow`. The live 4090 gate builds the
  same production model and a training-shaped synthetic batch, runs
  forward/backward without ambient autocast, and checks peak CUDA allocation
  delta over the pre-test baseline against the 22 GiB budget. Existing
  process allocations are excluded from the budget; the test still asserts its
  own resident allocations are released after cleanup.

```bash
# Fast config/helper coverage; runs in the default suite.
uv run pytest tests/steering/test_encdec_memory.py -q

# Live hardware gate; run on a clear 24GB 4090-class GPU.
uv run pytest tests/steering/test_encdec_memory.py -m slow -q -s -p no:cacheprovider
```

### Testing the SFT Data Generator

`prepare_sft_example` in `scripts/train/conditional_encdec_sft.py` is shared by
training and turn-cadence measurement. It truncates the render and mask, skips
rows below the supervision minimum before selecting features, checks coverage,
clips segments, and routes only the conditioned input positions (`ids[:-1]`).
The generator buffers this prepared routing through best-fit packing; prediction-
only segments remain available for feature targets and never own decoder tokens.
Cadence measurement supplies empty feature payloads to the same geometry path.


The SFT data generator (`make_sft_data_generator` in `scripts/train/conditional_encdec_sft.py`) is covered by:

| Test file | Coverage |
|-----------|----------|
| `test_encdec_sft.py` | Task loaders (PKU-SafeRLHF, Alpaca, CodeAlpaca, GSM8K), rendering edge cases, conditioned data-generator shape tests, training loop loss decrease |
| `test_encdec_sft_training_loop.py` | `evaluate_encdec_bpb`, complete `sync_last_step` truth table, exhausted-rank padded/RNG invariants, real two-rank Gloo tail aggregation and eval-step bound, compiled-CUDA zero-contribution padding, exact BPB golden, wandb integration, validation-eval trigger logic |
| `test_encdec_sft_mixtures.py` | shared `SFTMixture` selection/building, mixture→parquet conversion with `original_index` tracking, and synthetic train/converter index-alignment parity |
| `test_encdec_sft_step_estimation.py` | `SFTState.consumed` bookkeeping, 1-epoch consumption-based stopping |
| `test_encdec_sft_conditioned_integration.py` | **End-to-end conditioned SFT tier**: synthetic feature injection via `patch_load_sft_features`, shape + multi-turn segment-map invariants, tiny-CPU forward/backward gradient flow through encoder + cross-attention, short-run overfit smoke test, hard-fail-on-missing-SFT-features invariants, and `inputs`/`targets` contiguity invariants. A slow (`@pytest.mark.slow`) GPU sub-suite exercises the real nanochat tokenizer with flash-attn-4. |
| `test_sft_conv_truncation.py` | Full-sequence truncation, segment-feature clipping, shifted-token loss floors, and conditioned generator behavior under truncation |
| `test_sft_conditioning_alignment.py` | Startup fingerprint/content alignment, exact mismatch diagnostics, and order-independent conditioned loader behavior |
| `test_sft_full_sequence_integration.py` | Real feature-store routing through full-sequence batches, missing-row failures, and end-to-end integration invariants |
| `test_sft_patch_isolation.py` | Issue-#1303 patch-lifetime invariant: exact batch/state parity, lazy construction, exception/early-close cleanup, repository AST sweep, and non-vacuous mutation guards |
| `test_encdec_sft_resume.py` | **Resume-from-step invariants**: exact `buffered_v2` per-rank ordered-buffer/cursor recovery, save/load round-trip of both `loop_state` meta and sharded optimizer state, bitwise model-state and per-step-loss equivalence between an interrupted (save → load → continue) run and an uninterrupted baseline, synchronous/prefetched tensor and delivered-state equality, discarded-lookahead resume parity, config default for `resume_from_step`, and source-inspection smoke tests that catch regressions if the resume logic ever gets removed from `main()`. |
| `test_encdec_sft_batch_prefetch.py` | SFT-specific strict depth parsing, private-versus-delivered state visibility and deep-copy isolation (including `buffered_v2` state), optimizer-step production bound, worker timing delivery/tagging/failure isolation, instrumentation-free disabled timing, worker stream-failure propagation, mocked CUDA event plus consumer `record_stream` ordering, and a slow real-CUDA handoff on every visible device without a global synchronize. Shared FIFO/backpressure/closure/error behavior remains covered by `test_remote_prefetch_lifecycle.py`. |
| `test_encdec_sft_val_mode.py` | **Conditioned val invariants**: `_build_sft_val_loader` plumbing and config-surface checks that no validation-mode knob remains. |
| `test_encdec_sft_parquet_dir.py` | **Configurable SFT parquet dir invariants**: `_resolve_sft_parquet_dir` null-returns-default-layout and explicit-override-passthrough cases (including a parametrized check across all three mixtures), Hydra config round-trip, and source inspection asserting that `main()` uses the resolver and no longer inlines the `"sft_data"` string literal. |
| `test_encdec_sft_grad_clip.py` | **Configurable grad clip invariants**: Hydra config round-trip for `grad_clip_max_norm` (default `null` disables, explicit positive override, and explicit `null`), source inspection asserting that `main()` reads `cfg.grad_clip_max_norm`, guards on `is not None`, preserves the clip-between-backward-and-step position, and logs INFO (not WARNING) for the disabled default. The behavioral `clip_grad_norm_` check covers the explicitly-enabled path. Complements `test_gradient_clipping.py` which pins the torch-level behavior and the call-site position. |
| `test_encdec_sft_hardware_configs.py` | **Hardware-variant config invariants**: for each of `conditional_encdec_sft_4090`, `_2xh200`, `_4xh200`, `_2xb200`, plus the 4090 profiling variant — Hydra composition, inheritance from the SFT default, `device_batch_size` override, `total_batch_size` divisibility (so the SFT startup assert doesn't fire), `max_seq_len` per-variant expectations (1024 across the shared default and hardware variants), plus file-level checks (file exists, `defaults` references `conditional_encdec_sft`, usage comment mentions the target hardware). |
| `test_encdec_sft_encoder_stack.py` | CPU-side encoder-input stacking helper: padded shapes, one transfer per tensor, dtype/device preservation, and empty-feature width invariants. |
| `test_record_function_markers.py` | Profiling marker plumbing: SFT generator emits torch profiler events without a recorder, and nested timing rows inherit active training-loop tags when CSV timing is enabled. |

Run the SFT-specific integration suite with:

```bash
uv run pytest tests/steering/test_encdec_sft_conditioned_integration.py -v
# With the slow GPU sub-suite:
uv run pytest tests/steering/test_encdec_sft_conditioned_integration.py -v -m slow
```

All SFT generator wrappers must release `patch_load_sft_features` before any
outer generator yield.  This preserves construction laziness while preventing
order-dependent leaks when a test stops after one batch.  The invariant is
checked statically and at every SFT test boundary.  For the issue-#1303
regression matrix, run the producer and victim files in both directions (and
an interleaved order) with `-p no:randomly` when the optional ordering plugin
is unavailable:

```bash
uv run pytest \
  tests/tasks/test_encdec_sft_step_estimation.py \
  tests/scripts/test_encdec_sft_training_loop.py \
  tests/steering/test_sft_conv_truncation.py \
  tests/steering/test_sft_conditioning_alignment.py \
  tests/steering/test_sft_full_sequence_integration.py \
  -q -p no:randomly
```

## Architecture Diagram

```
Text tokens ──→ Token Embed ──→ [Decoder Block 1-3] ──→ [CrossAttn + Block 4] ──→
                                                               ↑
Features ──→ Feature Embed ──→ [Encoder Block 1-N] ──→ Per-segment outputs

──→ [Decoder Block 5-7] ──→ [CrossAttn + Block 8] ──→ ... ──→ LM Head ──→ Logits
                                    ↑
                          Same encoder outputs (via seg_map)
```

Cross-attention is inserted every `frequency`-th decoder layer (default: every 4th layer).

## Self-Conditioned Generation

The feature-prediction head can drive the encoder at inference time, removing
the dependence on real SAE features for generation. This is exposed via
`EncoderDecoderGPT.forward(..., return_fp_logits=True)` and orchestrated by
the `SelfConditionedFeatureProvider` in
`steering/conditional/encoder_decoder/self_conditioned.py`. See the
[self-conditioned generation pipeline doc](self-conditioned-generation.md) for
the direct `SelfCondConfig` integration pattern, the `loss_mode` contract
(default is `MAX_POOL`; `BEST_TOKEN` is incompatible with self-cond eval), and
the chunked-prefill toggle for base- vs chat-model prompts.

## Cross-shard Batch Padding Contract

The precomputed dataloader (`precomputed_data_loader` in `steering/conditional/encoder_decoder/dataloader.py`) assembles batches of size `B` from packed row shards. When all `B` rows come from the same shard, the loader calls `reconstruct_batch` directly and `_stack_batch_parts` is never reached. When a batch spans a shard boundary, each row is reconstructed individually and `_stack_batch_parts` stitches the parts together. Within `_stack_batch_parts`, a simple `t.cat` fast path applies when `pad_n_seg` (from the manifest) and `pad_F` (from validated selection metadata, an existing validation body scan, or the caller on the fused path) are both set so all per-row parts already share shape; otherwise the slow path re-pads each row to the batch-level maximum.

For local selection families, loader construction reuses the exact global positive-plus-negative CSR width bound by a completed splice or clustered-selection certificate. Legacy or uncertified local manifests compute the same width while performing their required semantic body scan. Remote loaders continue to use their authenticated inventory width.

### Raw `max_seg_size` and packed CUDA cross-attention

The packed-row producer and `backfill_packed_row_shapes` write
`metadata.max_seg_size` as the exact maximum observed
`seg_token_end - seg_token_start`. This is a raw semantic upper bound, not an
FP8-padded shape. Training discovery threads the raw integer through
`SharedState.max_seg_size`; the precomputed loader records and compares that
same raw value in its resume state.

On every static-shape training forward, cross-attention verifies that the
batch's required length (`pos_in_seg.amax() + 1`) does not exceed the raw
manifest bound. The check runs in both eager and compiled execution and occurs
before any allocation policy, so a stale bound fails even when it and the
required length round to the same bucket. CUDA keeps queries packed at `(B,T)`
and uses a shared data-dependent FlexAttention block mask, so the raw global
bound no longer sizes a query/scatter allocation. The test-owned padded SDPA
oracle still rounds its private query extent to
`ceil(raw_max_seg_size / FP8_ALIGNMENT) * FP8_ALIGNMENT`. Existing unaligned
manifests therefore remain valid without regeneration; changing the raw
manifest value still trips the loader's resume-identity guard.

### Globally packed CUDA feature encoder

The dataloader continues to emit dense `(B, n_seg, F)` tensors so artifact and
loader contracts do not change. At the CUDA model boundary, the encoder gathers
only `encoder_mask=True` entries into one globally ordered token axis. It rounds
that single axis up to a multiple of 32 for FP8 linear compatibility, adding at
most 31 tail entries across the whole batch. Owner tensors record each token's
original batch and segment. Encoder self-attention uses a segment-diagonal
FlexAttention mask, and decoder cross-attention uses the same owners to route
each flattened `(B,T)` query. `FeatureEncoder.forward` and precomputed/cached
model calls retain this `PackedEncoderOutput`; no live path restores
`(B, n_seg, F_padded, C)` or accepts a redundant padded mask. The exact padded
FeatureEncoder implementation, dense conversions, and padded cross-attention
are test-owned differential oracles. Production cross-attention has no dense
tensor, encoder-mask, or CPU branch.

This removes both feature-column padding and unused segment-row allocation from
encoder activations and cross-attention K/V. It does not make the dataloader
tensors ragged, so batching and checkpoint formats are unchanged. The packed
`K` is data-dependent, so the encoder and cross-attention execute in small
`dynamic=True` compile regions. The surrounding model retains the normal
`compile_static=auto` decision and stays static when loader shapes are fixed.
Packed tensor and sparse-mask dimensions are marked dynamic before those
regions run, so changing `K` does not specialize the full decoder graph.

Cached evaluation and self-conditioning use only four representation-level
operations: detach values, move values and all owners atomically, append one
non-empty batch-one segment, and materialize a batch-one state for multiple
candidate rows. Append and repeat rebuild batch/segment ownership and retain a
single global 32-aligned tail. Ordinary raw-input pretraining already called
the packed encoder directly and does not pass through these cached-state
operations.

`prev_segment` and `cum_max` are live-dataloader-only. Packed-row training and
`scripts/preprocess/precompute_selected_features.py` fail fast for those modes
because packed rows do not replay the carried document-history state they use.
`splice_counterfactual` can run from packed rows only after the offline
selection producer has materialized complete compatible `sel/<hash>/` shards.
The packed loader rejects missing, partial, manifest-mismatched, or
provenance-mismatched splice selections before yielding a batch; it never runs
fused runtime selection for any `feature_delta` mode.

`feature_delta.mode` values: `prev_segment`, `cum_max` (reference carried
across segments), and `splice_counterfactual` (issue #596). The latter applies
delta conditioning ONLY at post-splice segments of a spliced derived feature
dir — the reference is the would-have-followed segment recorded in the
`_splice_provenance` schema-v2 sidecar — and byte-exact absolute-percentile
conditioning everywhere else. The live loader verifies that every sidecar
splice position resolves to exactly one reference and exactly one delta-scored
segment; in a single-rank full-shard pass, the logged scored count must equal
`params.n_splice_positions`. Empty raw current/reference feature segments raise.
It requires: `feature_dir` = a spliced derived directory and, for the live
loader, `data_dir` = its `_parquets` (auto-derived when unset, D8).
Missing/outdated sidecars, mismatched doc counts, or sidecar↔shard drift all
raise (no silent fallback; the runtime-miss RAISE deliberately deviates from
issue #596 requirement 4's fallback wording).

For packed `splice_counterfactual` training, produce artifacts in this order:

```bash
# 1. Build the spliced derived feature dir + _parquets + _splice_provenance.
uv run python -m scripts.preprocess.splice_feature_shards \
    feature_dir=<base_feature_dir> cut_fraction=<f> seed=<seed>

# 2. For both base and derived dirs: statistics, percentiles, token boundaries.
uv run python -m steering.compute_statistics feature_dir=<dir>
uv run python -m scripts.preprocess.precompute_percentiles feature_dir=<dir>
uv run python -m scripts.preprocess.precompute_token_boundaries feature_dir=<dir>

# 3. Precompute packed rows for derived train/val and original/base val.
uv run python -m scripts.preprocess.precompute_packed_rows feature_dir=<derived_dir>
uv run python -m scripts.preprocess.precompute_packed_rows \
    feature_dir=<base_feature_dir>

# 4. Precompute three selection dirs.
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<derived_dir> packed_row_dir=<derived_train_packed> \
    feature_delta.mode=splice_counterfactual
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<derived_dir> packed_row_dir=<derived_val_packed> \
    feature_delta.mode=splice_counterfactual
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<base_feature_dir> packed_row_dir=<base_val_packed> \
    feature_delta=null feature_selection.delta_weights=null
```

The first two `sel/` dirs use the delta-aware selection hash. The base
`val/*` dir uses the delta-free twin hash, so it intentionally lives under a
different `sel/<hash>/`.

### Multi-segment splice selection

`selection_splice_batch_size` is a semantic producer mode for the two
delta-aware selection commands above. Its default is `64`. The exact enabled
size is part of artifact identity, so selection production and training must
resolve the same value. `null` and `1` explicitly execute the established
one-segment path byte-for-byte, add no hash or manifest field, and keep
existing legacy selection families discoverable. To select a non-default
batched family, pass the same explicit integer to selection production and the
training config, for example:

```bash
uv run python -m scripts.preprocess.precompute_selected_features \
    feature_dir=<derived_dir> packed_row_dir=<derived_train_packed> \
    feature_delta.mode=splice_counterfactual selection_splice_batch_size=64
```

The default of 64 is the measured plateau choice, not an arbitrary maximum.
Across the first three real L22/c1024 packed provenance shards, ordinary runs
had median/p99/p99.9/max lengths 6/35/55/106. A cap of 64 created 45,922
ordinary batches across 375,464 rows, versus 45,908 at an unbounded/128 cap.
In the production-width masked-negative component sweep, 64 reached 1.608x
the one-row throughput while 256 reached 1.661x. These are default-selection
and component measurements, not a claim of end-to-end producer speedup.

The producer retains the established `(source_shard, doc)` execution order.
Within that stream it accumulates ordinary rows until the configured bound, a
splice-delta row, or a precomputed-percentile representation boundary. It then
batch-samples positive cardinalities/rankings and masked inactive negatives.
Every delta row first flushes the ordinary batch and continues through the
existing specialized delta-negative sampler. Thus batching cannot reinterpret
counterfactual candidates, delta weights, reference masks, or the
`neg_delta_ratio` mixture.

The ordinary batched path stays on CPU: source features, percentile overlays,
selection results, and serialized CSR tensors are CPU tensors in this
producer. Removing the old selected-ID `tensor -> Python list -> tensor`
round-trip therefore removes Python boxing/copies, not a GPU host-device
synchronization. Results are packed into local tensor CSR batches and scattered
back to exact packed-segment order; PyTorch lacks CPU indexed assignment for
`uint16`, so IDs are range-checked, scattered as `int64`, then narrowed once in
bulk. Masked negative batching uses sorted active-ID membership and the cached
allowed universe rather than allocating `[batch_size, n_sae_features]`; it is
differentially required to match the retained dense sampler's outputs and final
generator state exactly.

Clustered ordinary output is checked for mask-allowed representative IDs when
each serialized shard is reloaded, before that shard is certified or the
selection manifest can become complete. The ordinary row loop omits duplicate
checks; unclustered mask checks and public negative-sampler guards remain.
Single-segment selection retains its representative and ID-range checks before
conversion to unsigned storage; these also enforce the mask, so it omits the
separate clustered mask checks.

Each enabled shard profile adds a `selection_batching` object with the exact
configured/maximum/mean ordinary batch sizes, ordinary segment and batch
counts, serial delta-row count, algorithm version, and tensor-native CSR flag.
The producer asserts that ordinary plus delta counts cover the packed shard
exactly before it writes output, so these fields are both an operator-visible
utilization diagnostic and a fail-loud accounting gate.

Cross-segment grouping changes which RNG draws belong to which row (even when
the total number of draws is unchanged). The artifact is therefore not claimed
bit-exact to legacy mode. `compute_selection_hash` conditionally adds a
descriptor containing schema version, algorithm version, and exact batch size;
the splice manifest records the same descriptor and resume requires exact
agreement. Different sizes and legacy mode always live in different
`sel/<hash>/` directories. Resource-only worker/cache/shared-store settings
remain outside identity and must produce exact bytes within any one batching
mode. The delta-free original/base validation selection ignores this setting
and retains its existing identity.

Correctness gates cover: exact legacy null/explicit-1 trees; exact one-row
output and generator-state differential parity; exact sparse-vs-dense masked
batch negatives at 65,536-feature width; randomized out-of-order CSR assembly;
fixed-cardinality positive-ranking **and negative-candidate** output/final-RNG
equality to serial selection; masks, clusters, empty rows, storage boundaries,
resume, deterministic rebuild, and real producer routing with multi-splice
documents. A gated real-l22 test runs the
first three production-sized packed shards and checks every serialized row,
mask, representative, manifest, and coverage invariant; this producer is
CPU-only and does not load a GPU model. Performance tests remain separate from
correctness tests: sampler microbenchmarks do not establish an end-to-end
producer speedup.

**`data_dir` auto-derivation (D8):** when `feature_delta.mode=splice_counterfactual`
and `data_dir` is unset, it is auto-derived to `<feature_dir>/_parquets` — the
parquets the splicer wrote alongside the derived shards. An explicit `data_dir`
that does not match that canonical path is accepted but logs a warning (the
per-shard doc-count guard between the sidecar and the parquet shard still
fails fast on a real mismatch).

**Dual validation for `splice_counterfactual` (D12):** with
`use_precomputed_batches=true`, both validation passes use packed rows and
selection shards as described above. With `use_precomputed_batches=false`,
both passes use live loaders: `val_spliced/*` reuses the main counterfactual
pipeline over the active derived `feature_dir`/`data_dir`; `val/*` runs a
delta-free twin pipeline (same selection/negatives/mask, `feature_delta=None`,
`delta_weights` stripped) over the original/base feature directory. That
directory is resolved by `original_val_feature_dir`, auto-detected via the
same canonical `<base>/splice/cut_frac*/seed*` path rules as the packed-row
flavor above; its own text source is `original_val_data_dir` (`null` =
nanochat default corpus). Both flavors reuse the percentile/tokenizer
checksum-equality gate against the original/base directory's
`feature_statistics.safetensors` (ensuring both directories share one
percentile table). The packed flavor additionally requires the derived and
original/base packed-row directories plus the matching `sel/` shards; the live
flavor never discovers or requires packed-row directories.

**Invariant enforced at the stack step:** all batch tensors share the same `(B, n_seg)` leading dimensions:

| Tensor | Shape |
|---|---|
| `enc_fids`, `enc_mags`, `enc_pcts`, `enc_mask` | `(B, n_seg, F)` |
| `seg_doc_id` | `(B, n_seg)` |
| `seg_map` | `(B, T)` |
| `fp_targets` (when feature prediction is enabled) | `(B, n_seg, feature_prediction_width)`; full width is `n_sae_features`, compact width is `V_effective` |

In the slow path, `n_seg` is `max(per-row natural segment counts)` for the batch. Padded positions in `enc_*` are filled with zeros and `enc_mask=False`. Padded positions in `seg_doc_id` are `-1` (filtered out of the feature-prediction loss by the `seg_doc_id[:, :-1] >= 0` guard in `_full_segment_loss`). Padded positions in `fp_targets` are zeros — semantically "no feature present" — which is benign because the corresponding `seg_doc_id == -1` already excludes them from the loss.

**Production failure prevented by this contract:** if `fp_targets` is padded to a different `n_seg` than `enc_*` (e.g., via independent `max()` over a different list of parts), the downstream feature-prediction loss silently mismatches shapes or — in the original failure mode — `torch.cat` raises a `RuntimeError: Sizes of tensors must match except in dimension 0`. The asserts in `_stack_batch_parts` make this fail loudly at the boundary instead of silently propagating.

## Remote/Streaming Clustered Pre-Selection

This opt-in producer is intentionally limited to static clusters plus
precomputed, sampling-based selection. It supports the existing
`feature_delta.mode=splice_counterfactual` transport and ordinary
`feature_delta=null` selection with one worker. `remote_preselection: null`
uses the local producer literally; enabled mode has no local-body or live
fallback.

Both modes require a selection-independent schema-v2
`remote_packed_inventory.json`, built by `publish_remote_packed_inventory`
without a selection checksum. Schema v1 remains valid for existing remote
training, but it lacks the semantic layouts and packed base hash required by
remote pre-selection. Splice-CF trusts the configured completed percentile
family and constructs an ephemeral source contract from remote object
listings plus local layout/statistics/sidecar controls; there is no source
inventory publication or body recomputation step.

Statistics, the packed manifest, and packed provenance remain local controls.
Splice-CF also keeps the percentile manifest and splice sidecars local while it
streams authenticated raw/percentile inputs. Ordinary selection uses no raw
feature bodies, percentile bodies, source inventory, sidecars, or
`FeatureIndex`: the unchanged selector consumes only feature IDs and percentile
bytes already present in each authenticated packed body.

Remote paths are derived from `remote_training.selection_prefix` and the exact
path relative to `feature_root`:

```text
<packed-prefix>/remote_packed_inventory.json
<packed-prefix>/sel/<selection-hash>/<shard>.safetensors.zst
<packed-prefix>/sel/<selection-hash>/_remote_preselection_receipts/<shard>.safetensors.zst.json
<packed-prefix>/sel/<selection-hash>/manifest.json
```

Ordinary mode never accesses or fences the raw/percentile namespace. Its
selection-shard count equals the packed-inventory-v2 count, not the raw
feature-shard count.

Packed streaming holds one active compressed body plus at most one downloaded
lookahead. Every locally serialized selection shard is reopened and validated
by the same clustered CSR validator used locally. Publication is immutable and
ordered: selection body, signed receipt, then the timestamp-free final manifest
after complete coverage and input fences. Conflicts are never overwritten or
deleted automatically.

Ordinary sampling uses one continuous generator across packed shards, shared by
positive and negative sampling. Its schema-v2 receipt signs the exact
post-shard RNG-state digest. A partial restart replays the exact leading receipt
prefix through the unchanged selector and checks body bytes, width, and RNG
state before continuing. Complete receipts can finalize without replay; a
complete manifest performs no packed-body download or upload. Receipt state is
authoritative regardless of `selection_resume`.

The final manifest uses the existing selection schema and consumer
certificates. It contains no transport paths, inventory aggregates, Xet
identities, receipts, RNG state, or staging/cache settings. Therefore
`RemoteSelectionMaterializer` discovers both modes without a producer-specific
consumer branch. Transport settings and receipt-only evidence do not alter the
selection hash or serialized bodies.

## FeatureIndex Fail-Fast Contract

`FeatureIndex.lookup(shard_stem, doc_idx)` raises `KeyError` when the shard is
unknown or the document is absent — it never returns `None` (issue #521).

- **Why:** for encoder-decoder pretraining/precompute, every iterated parquet row
  must have an extracted feature row. A missing one indicates corruption or a
  stale/misaligned shard; continuing with empty conditioning silently changes the
  training distribution.
- **Why it is safe:** extraction rejects zero-token (`extract_features.py`, the
  per-doc tokenize step — the `kept_ids` guard right after
  `_truncate_to_training_window`) and zero-segment (`_precompute_batch_cpu_work`)
  documents, so a successfully written shard has full 1:1 coverage of its parquet
  rows, every doc with >= 1 segment.
- **Enforcing sites:** the live dataloader, `SFTFeatureStore` / conditioned SFT
  (subscript access — never `.get()`, which would swallow the `KeyError`), and
  `precompute_packed_rows`. Eval paths constrain doc indices to present docs
  (`doc_ids_for_shard`) so the raise only fires on genuine corruption; the
  interactive completion TUI pre-checks membership to keep its documented
  `None`-return.
- **Valid zero outcomes** (selecting zero features for a segment) happen *after* a
  successful lookup, never as `lookup(...) is None`.

### Splice-preselection source caches (issue #740)

Cache bounding changes nothing about the fail-fast contract above: `FeatureIndex.lookup`
still raises `KeyError` and `SpliceProvenanceIndex` still raises `SpliceProvenanceError`
for exactly the same conditions at every `max_resident_shards` value — a bigger cache
only changes *when* a shard loads (and whether it must reload), never whether a bad
lookup raises.

- **The problem:** `select_splice_counterfactual_features_for_shard` (the
  `splice_counterfactual` preselection pass in
  `scripts/preprocess/precompute_selected_features.py`) iterates `(source_shard, doc)`
  groups sorted by first packed-segment appearance. A packed shard's groups routinely
  interleave many distinct source stems (p50 6, p99 13 on the real l22/f65k corpus).
  Under the historical single-slot cache, every stem alternation evicted and fully
  reloaded a multi-GB feature shard and re-parsed a multi-MB sidecar.
- **Two producer-only knobs** (`configs/training/conditional_encdec_train.yaml`, read
  only by `precompute_selected_features.py`): `selection_source_cache_shards` (default
  **5**) sizes the `FeatureIndex` resident feature-shard cache;
  `selection_sidecar_cache_shards` (default **16**) sizes the `SpliceProvenanceIndex`
  resident `_splice_provenance` sidecar cache. Both classes still default their own
  `max_resident_shards` to **1** (the legacy single-slot behavior) for every other
  consumer — the live dataloader, packer, evals, TUI, and SFT never pass these keys and
  are unaffected. (SFT does pass the unrelated `prefetch`/`lazy_docs` arguments;
  see "Conditioned-SFT shard prefetch and lazy documents" above. Its resident
  capacity is still 1.)
- **Memory accounting (#758):** the old `cache capacity × per-shard RSS` formula was
  not a hard bound. The observed ≈12.7 GB source figure was a decode transient, not a
  resident stem, and eager tensors can keep the complete decompressed safetensors
  backing allocation alive. A shared-store directory count also omits
  evicted-but-still-mapped tmpfs files. Use
  `scripts/analysis/splice_preselection_locality.py` to characterize load locality,
  but use the producer's admission report—not an RSS sum—to decide whether a parallel
  topology fits.
- **Retain protocol:** before each packed shard's group loop runs, the script drops
  resident entries not referenced by that shard (`drop_except`) and pre-seeds the
  referenced stems' feature shard, then sidecar, per stem, in first-need order, bounded
  by `min(selection_source_cache_shards, selection_sidecar_cache_shards)`. This is what
  turns repeated cross-shard reloads into cache hits — a stem still resident from the
  previous packed shard is never reloaded — while preserving the reference
  implementation's exact [feature shard, then sidecar] per-stem access order, so the
  first artifact to fail on a corrupt input is identical to the reference at any
  capacity. If a packed shard's distinct-stem working set exceeds the **source-shard
  cache** capacity (`selection_source_cache_shards` — the expensive ~GB reloads, as
  opposed to the cheap sidecar cache), the script logs a WARNING naming the shortfall
  and the tail feature shards reload under ordinary LRU eviction — slower, but still
  byte-identical.
- **Capacity 1 is the reference:** at `max_resident_shards=1` on both indices, the
  retain step is skipped entirely and each class reproduces the original single-slot
  load/evict sequence byte-for-byte. Every other capacity produces byte-identical
  `sel/<hash>/` output too — the caches are a resource/performance bound, never a
  correctness switch.
- **Not in the selection hash:** neither key is part of `compute_selection_hash`'s
  payload, so every existing `sel/<hash>/` directory stays valid regardless of cache
  size, and neither key can silently invalidate a run. Both are also popped from the
  resolved config in `_resolved_user_config` (mirroring `packed_row_dir`), so they
  never leak into training checkpoint or W&B metadata.

### Resource-safe parallel splice preselection (issue #758)

`selection_num_workers=1` is still the exact serial selection path. For `>1`,
one parent performs first-source validation and overlay discovery once.
Stat-certified caches return the exact existing
source/packed/provenance/sidecar fingerprints without body reads only when
every input identity matches; cold or stale entries execute the original full
hashing and semantic validation. The parent freezes that result into a versioned
`FeatureIndexCatalog`; spawn workers validate its path/link/target signatures and
hydrate without loading the first source shard or rediscovering overlays. Any drift
between the parent scan and worker use is fatal.

The remaining producer-only controls are:

- `selection_resume=true` is the default. For clustered selection, the parent
  reloads each exact compressed output after its atomic publish, validates all
  representative-only CSR tensors against that shard's signed segment count, and
  hashes the compressed bytes while requiring the file's inode/size/timestamps to
  remain unchanged across load, validation, and hashing. Only then does it atomically
  advance the `in_progress` manifest. The schema-v3 per-shard record binds all three
  splice counters and the successful post-serialization validation to that exact
  content digest; its signed output record also carries the exact running maximum
  positive-plus-negative CSR width. At completion that maximum must agree with the
  separately signed remote-loader certificate before either is published. A whole
  file published in the narrow crash window before the
  manifest checkpoint is overwritten deterministically; a missing, changed, or
  semantically invalid shard fails loudly. Because every shard is certified at its
  checkpoint, completion checks certificate coverage and re-hashes the compressed
  selection files once for current-byte integrity; it does **not** decompress them or
  reopen the large packed-row bodies merely to recover segment counts. An unchanged
  complete rerun still streams compressed selection bytes for exact integrity, but
  does not decompress packed/selection tensors or replay packed provenance and
  sidecar statistics. Pre-#790 schema-v1/v2 resume certificates do not bind the
  corrected cluster-delta ordering and are rejected, even if copied beneath a new
  selection directory. A current-semantic in-progress manifest with no certificate
  replays stats once and validates compact selection shards; a partial schema-v3
  certificate validates only the remaining compact shards. Neither recovery path
  reopens packed-row bodies merely to reconstruct the signed width.
  A malformed or internally disagreeing advertised certificate is fatal. Set
  `selection_resume=false` only to deliberately recompute every in-domain
  shard after artifact-identity checks; this explicit mode also repairs corrupt old
  shard bytes or progress certificates because those bytes will be replaced rather
  than reused.
- The production clustered path requests masked inactive negatives one segment at a
  time. For one-dimensional CPU integer inputs up to 4,096 active IDs, it now uses
  cached read-only allowed IDs plus sorted sparse membership checks instead of allocating a
  `bool[n_sae_features]` scratch tensor for every call. The established dense path is
  retained for every other dtype/device/shape and edge case. Differential tests compare
  both output IDs and final generator state across full-width whitelist/blacklist cases,
  and a three-shard pipeline fixture requires exact compressed-byte equality against
  the dense implementation.
- `selection_packed_spool_dir=null` uses the eager packed reader. An existing
  disk-backed directory selects streaming decompression to one temporary plain
  safetensors file per active worker. The worker maps that file only inside the
  current shard context; root tensors are invalidated, mappings closed, and the file
  unlinked on success or failure. Retaining a tensor view beyond that context is an
  API violation. `tmpfs` and `ramfs` spools are rejected because they merely move the
  same pressure into memory.
- `selection_shared_store=false` keeps private source tensors. With `true`, workers
  map immutable per-stem tensors through `NANOCHAT_SHM_DIR` (default `/dev/shm`).
  A materialized stem also publishes two fixed-size `int64[n_segments]` document
  locator tensors: the exact stable eager row order and its sorted document IDs.
  Each worker holds one mapped stem, binary-searches that locator, and constructs
  only the requested document's zero-copy `SegmentFeatures` views. This preserves
  the original lookup/RNG/error order while making interleaved LRU-1 switches pure
  remaps instead of complete-stem sorts and Python-object rebuilds. Admission costs
  the exact `16 * n_segments` locator payload and a conservative locator-build
  transient.
  `selection_shared_store_capacity=0` resolves to `workers + 4`; an explicit positive
  value caps visible stems. Capacity is not a byte bound: at most `C` visible stems,
  an old resident plus an in-flight next mapping per active worker, and one
  publication temporary may coexist, so admission costs the largest
  `min(n_source_stems, C + 2W + 1)` stem allocations.
  tmpfs is physical memory and can be swapped; increasing its size does not add RAM.
- Spawn workers use compact int64 arrays plus lazy per-document maps for validated
  JSON splice sidecars. The reference parser still performs every schema check in the
  same order before conversion. Values outside int64 retain the legacy Python object
  representation rather than overflow.

Before creating `sel/<hash>/`, an in-progress manifest, or a shared store, parallel
startup streams and validates every packed/provenance/source/percentile safetensors
header. The admission report derives exact declared tensor and serialized sizes and
resolved runtime widths (`uint16` IDs only when `n_sae_features <= 65,536`; otherwise
`int32`). It combines those structural bytes with explicit measurement-derived
private-state and host/OS reserves, current `MemAvailable`, finite cgroup-v2
`memory.max - memory.current`, shared-filesystem free bytes, and disk-spool free bytes.
The result is logged as one `SPLICE_RESOURCE_ADMISSION` JSON object.

A pass means the exact requested topology is within the supported envelope at that
instant; it is not a mathematical promise against later unrelated machine pressure.
A failure is loud and side-effect-free and never silently lowers concurrency.
A complete certified resume is modeled separately: no worker pool is launched
and no packed/provenance/sidecar/selection body is replayed after startup
identity checks. Admission still checks the requested topology and treats the
run as having no selection work.

Shared-store lifetime is identity- and lease-based rather than PID-only:

1. The parent creates an exclusive directory and owner record containing schema,
   UID, boot ID, PID, process start time, canonical root, and creation time.
2. A separate-session janitor monitors the exact parent (using `pidfd` where
   available) and a parent-only control pipe. Each worker registers its exact identity
   atomically and holds a shared store lease before mapping. Termination binds a pidfd,
   revalidates boot ID/PID/start time, and signals through that descriptor; native
   CPython wrappers or registered raw Linux syscalls are accepted, but there is no
   PID-only fallback.
3. Every source materialization path—including a peer-publication/eviction fallback—
   executes under the cross-process materialization lease. Publication and the first
   mapping also occur within that lease, making one decode transient at a time a real
   bound.
4. Normal completion closes the pool, acquires the exclusive lease, removes the exact
   store, verifies removal, and acknowledges cleanup before a complete manifest can be
   written. On parent death, the janitor signals only matching registered worker
   identities, waits for lease release, and then removes the store. A parent cleanup
   request is one-shot and closes both control descriptors even when its deadline
   expires, preventing a second command after an ambiguous result.
5. If the entire coordinator/process group receives `SIGKILL`, no user-space cleanup
   can run. The next invocation sweeps only schema-valid direct children whose exact
   owner is dead and whose exclusive lease is available; symlinks, path escapes,
   ambiguous identities, and live leases are preserved for manual inspection.

Resume and resource modes do not alter selection semantics: fixed-clock differential tests require
byte-for-byte equality of every selection shard and manifest across W=1, private
spawn, shared spawn, disk-spooled spawn, capacities including churn, fresh runs,
partial resumes, masks, inactive/active clusters, zero-splice docs, and multi-splice
docs.
