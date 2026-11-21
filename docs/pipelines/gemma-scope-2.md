# GemmaScope-2 (Gemma 3 SAEs)

`gemma-scope-2` is DeepMind's Gemma 3 follow-up to the original GemmaScope SAE
suite. It coexists with v1 (`gemmascope`) inside the same resolver
(`steering/sae_resolver.py`): the release family is selected by
`SAERelease.gemmascope_2`, and the resolver dispatches to a v2 code path with
its own validation matrix, hook templates, and ID grammar. Everything
downstream of extraction (statistics → percentiles → token boundaries → packed
rows → encoder-decoder training) is identical; only the model family, SAE
identifiers, and L0 representation change.

This page is the canonical reference for the v2 release. For the shared
extraction mechanics (pooling, segmentation, CSR shard format, validation), see
[`feature-extraction.md`](feature-extraction.md).

## Supported Models

Ten Gemma 3 checkpoints are registered in `steering.models.EXTRACTION_MODELS`
and in `GEMMASCOPE_2_MODEL_SIZES` (`steering/sae_resolver_v2_data.py`): five
sizes × {`pt` (pretrained), `it` (instruction-tuned)}.

| Model name | HF id | n_layers | Selected layers |
|------------|-------|----------|-----------------|
| `gemma-3-270m`    | `google/gemma-3-270m`    | 18 | 5, 9, 12, 15 |
| `gemma-3-270m-it` | `google/gemma-3-270m-it` | 18 | 5, 9, 12, 15 |
| `gemma-3-1b-pt`   | `google/gemma-3-1b-pt`   | 26 | 7, 13, 17, 22 |
| `gemma-3-1b-it`   | `google/gemma-3-1b-it`   | 26 | 7, 13, 17, 22 |
| `gemma-3-4b-pt`   | `google/gemma-3-4b-pt`   | 34 | 9, 17, 22, 29 |
| `gemma-3-4b-it`   | `google/gemma-3-4b-it`   | 34 | 9, 17, 22, 29 |
| `gemma-3-12b-pt`  | `google/gemma-3-12b-pt`  | 48 | 12, 24, 31, 41 |
| `gemma-3-12b-it`  | `google/gemma-3-12b-it`  | 48 | 12, 24, 31, 41 |
| `gemma-3-27b-pt`  | `google/gemma-3-27b-pt`  | 62 | 16, 31, 40, 53 |
| `gemma-3-27b-it`  | `google/gemma-3-27b-it`  | 62 | 16, 31, 40, 53 |

> **The 270m PT quirk.** The 270m pretrained model name is `gemma-3-270m` —
> there is **no `-pt` suffix**, because HuggingFace publishes it as
> `google/gemma-3-270m` (the IT variant is `google/gemma-3-270m-it`). The v2
> release token still uses `pt`, so the model→release table reconciles them.
> Passing `model="gemma-3-270m-pt"` raises a `ValueError` that suggests
> `gemma-3-270m`. Every other size keeps the explicit `-pt`/`-it` suffix.

SAEs are only published at the **selected layers** per size — not every
transformer layer. Choosing any other layer raises a `ValueError` listing the
valid layers and the closest one.

The `pt` entries set a `chat_template_source` (the matching `-it` repo) so the
extraction pipeline can apply a chat template to PT models that ship without
one.

## L0 Tiers

v1 indexed SAEs by an **integer average L0**. v2 instead publishes a discrete
set of **tiers**, modelled by `L0Tier` (a `StrEnum` in
`steering/sae_resolver.py`):

| Tier | Availability |
|------|--------------|
| `small`         | all activations / widths |
| `medium`        | all activations / widths |
| `medium_seed_1` | **residual + width 262k only** (an extra-seed variant) |
| `big`           | all activations / widths |

The declaration order (`small → medium → medium_seed_1 → big`) is the canonical
small→big ordering used for closest-match suggestions in error messages.

Set the tier in config via `sae.l0`:

```yaml
sae:
  release: gemmascope_2
  num_features: 16k
  l0: medium          # small | medium | medium_seed_1 | big
```

`gemma-scope-2` has **no canonical (l0=None) variant**. `resolve_sae` raises if
`l0` is omitted, and raises again if an `int` average-L0 is passed — v2 strictly
requires an `L0Tier`. Conversely, passing an `L0Tier` to a v1 `gemmascope`
release also raises.

## Widths

Width availability is per-activation (`GEMMASCOPE_2_VALID_COMBOS`):

| Activation | Widths |
|------------|--------|
| residual   | 16k, 65k, 262k, 1m |
| mlp        | 16k, 65k, 262k (**plus 1m for `gemma-3-1b-{pt,it}`**) |
| attention  | 16k, 65k, 262k |

`medium_seed_1` is the only tier restricted to a single (activation, width)
cell: residual at 262k. All other tiers exist at every published width for
their activation type.

## Hook Names

The resolver returns a TransformerLens-style `hook_name` that nnterp parses by
substring to route to the right accessor (`_GEMMASCOPE_2_ACTIVATION_MAP`):

| Activation | Hook template |
|------------|---------------|
| residual   | `blocks.{L}.hook_resid_post` |
| mlp        | `blocks.{L}.hook_mlp_out` |
| attention  | `blocks.{L}.attn.hook_z` |

> **v2 attention is pre-projection — same as v1.** Both `gemmascope` and
> `gemmascope_2` attention SAEs capture at `blocks.{L}.attn.hook_z`, the
> per-head attention output **before** the `o_proj` linear. `d_in = num_heads
> × head_dim` (e.g. 1024 for the 1b model). This was verified against the
> gemma-scope-2 SAEs' own `cfg.metadata.hook_name` at gemma-3-1b-pt layers 7
> and 13, width 16k. Do **not** confuse this with `hook_attn_out` (the
> post-projection output) — using that hook causes a shape mismatch crash
> (`1152 ≠ 1024`) on `sae.encode`.

## Worked Example

The default Gemma 3 config is `configs/extraction/extract_features_gemma3.yaml`.
It inherits everything from `extract_features.yaml` and overrides only the
deltas: `model: gemma-3-4b-it`, `layer: 22`, and the v2 SAE block
(`release: gemmascope_2`, `num_features: 16k`, `l0: medium`).

```bash
uv run python -m steering.extract_features \
    --config-name extract_features_gemma3 \
    num_shards=10
```

Multi-GPU is identical to v1 — wrap with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    --config-name extract_features_gemma3 \
    num_shards=370
```

For that default combo (`gemma-3-4b-it`, residual, layer 22, width 16k, tier
`medium`), `resolve_sae` returns the following triple — copied verbatim from a
live probe:

| Field | Value |
|-------|-------|
| `release`   | `gemma-scope-2-4b-it-res` |
| `sae_id`    | `layer_22_width_16k_l0_medium` |
| `hook_name` | `blocks.22.hook_resid_post` |

The v2 `sae_id` grammar is `layer_{L}_width_{W}_l0_{tier}` — all
**underscore**-separated (contrast with v1's slash-separated
`layer_{L}/width_{W}/canonical`).

For comparison, the same selection with `activation_type=attention` resolves to
release `gemma-scope-2-4b-it-att`, identical `sae_id`, and hook
`blocks.22.attn.hook_z` (the pre-projection per-head output, same as v1).

Output lands at the standard auto-constructed feature directory
(`steering.feature_paths.build_feature_dir`):

```
out/data/features/s128/l22/a_res/p_max_pool/f16k/m_gemma-3-4b-it/base/
```

i.e. `{root}/s{segment_size}/l{layer}/a_{act}/p_{pooling}/f{width}/m_{model}/{phase}/`.
The L0 tier is **not** part of this path (see next section).

Always-on BOS exclusion applies identically to v2: every v2 run adds a `/no_bos` leaf after `/context_independent` (e.g. `…/base/context_independent/no_bos/`) and records `ignore_bos_features=true` in both shard metadata and the stats file (a plain bool, always written; legacy artifacts load as `false`) — see [`feature-extraction.md` § Ignore BOS Features](feature-extraction.md#ignore-bos-features).

## L0 Tier Lives in Metadata, Not the Path

The feature directory layout has no L0 component, so two extractions that
differ only by tier would otherwise collide. To prevent silently mixing tiers
in one directory, the tier is stored in **shard metadata**
(`MetadataKey.L0_TIER` in `steering/feature_io.py`), and
`assert_l0_tier_compatible(feature_dir, new_l0_tier)` refuses to write into a
directory whose existing shards carry a different tier (raising
`FeatureDirectoryMismatchError` and listing every mismatched shard). v1 shards
have no tier key; `read_l0_tier_from_shard` returns `None` for them, and a
v1↔v2 cross-write is also rejected.

The tier is read by streaming only the few-KB safetensors header
(`read_shard_metadata` → `read_shard_header` in `steering/feature_io.py`),
not by decompressing the multi-GB shard body — so the
`assert_l0_tier_compatible` startup guard does not re-read every existing
shard in full. A corrupt, truncated, or empty shard fails loudly with
`CorruptFeatureShardError` (a `ValueError`) naming the shard, and is never
silently treated as a tier-less v1 shard.

Cross-shard tier consistency is enforced by a single shared rule,
`assert_uniform_l0_tier(shard_paths)` in `steering/feature_io.py`, consumed
by both `verify_feature_dir` and `compute_statistics`. `compute_statistics`
now **stamps the resolved tier into the stats file** (`l0_tier` on
`ExtractionStatistics`, `steering/stats_types.py`) so a v2 stats file
mirrors the v1/v2 split its shards carry (v1 → `None`); a directory with
mixed tiers fails fast with `FeatureDirectoryMismatchError`.

If you want both tiers, point each extraction at a distinct `feature_root` (or
explicit `output_dir`).

## Registry Injection

SAELens 6.42 already registers most v2 releases. A small idempotent shim
(`steering/sae_resolver_v2_registry.py`, wired in via `steering/__init__.py`)
backfills only the releases that exist on HuggingFace but are missing from the
installed SAELens registry. The candidate set (`INJECTED_RELEASES`) is the
27b-pt MLP/attention releases plus the 1b/12b PT res/mlp/att releases (missing
in some environments). Each candidate is checked at import time; already-present
releases are skipped with a debug log, so re-imports never double-register.

## Hardware Note (sm_120)

The nnterp loader for Gemma 3, `_build_nnterp_gemma3` in
`steering/extract_features.py`, pins `attn_implementation="sdpa"` when calling
`load_gemma3_text_decoder(...)`. This avoids routing onto an FA4
attention path that has not been exercised on **sm_120** (the Blackwell
consumer / RTX PRO 6000 generation). The pin is unconditional in the loader, so
no manual flag is needed for the extraction run.

## VLM checkpoints (4b/12b/27b) — text-decoder loading

`gemma-3-4b/12b/27b` ship as `Gemma3ForConditionalGeneration` VLM checkpoints
whose text weights live under `language_model.model.*`. A plain
`Gemma3ForCausalLM.from_pretrained` does **not** remap them and silently
random-initialises the entire text decoder (transformers 5.x; issue #549) —
producing garbage SAE features. All accelerated backends load the text decoder
through `steering.gemma3_loading.load_gemma3_text_decoder`, which remaps
`language_model.model.* → model.*` into a normally-constructed
`Gemma3ForCausalLM` (preserving the non-persistent `embed_scale`/rotary buffers)
and **fails loud** (`ModelWeightLoadError`) if any weight is unmaterialised.
`gemma-3-270m/1b` are native text-only checkpoints and take the direct load
path. A second gate hard-fails `compute_statistics`
(`DegenerateFeatureStatisticsError`) if the aggregated features show the
random-weight signature.

**Memory note (12b/27b):** Normal construction allocates an fp32 model in host
RAM *before* the state dict is loaded, so peak host RAM is roughly
`sizeof(fp32 model) + sizeof(state dict on disk)` — higher than a straight
`from_pretrained` call (which loads checkpoint shards one at a time without
first building a full remapped state dict in host RAM).
`no_init_weights()` suppresses the random-init *compute* overhead but does not
reduce the RAM footprint. A meta-device load path that avoids the double
allocation is a planned follow-up. The **4b** checkpoint is unaffected in
practice — it fits comfortably within typical host RAM budgets.

## v1 vs v2 at a Glance

| Aspect | v1 (`gemmascope`) | v2 (`gemmascope_2`) |
|--------|-------------------|---------------------|
| Models | Gemma 2 (2b/9b/27b) | Gemma 3 (270m/1b/4b/12b/27b × pt/it) |
| `sae_id` grammar | `layer_{L}/width_{W}/canonical` (slashes) | `layer_{L}_width_{W}_l0_{tier}` (underscores) |
| L0 representation | int average L0, or `None` (canonical) | `L0Tier` enum (no canonical) |
| Layer coverage | all/most layers | selected layers only (4 per size) |
| Attention hook | `blocks.{L}.attn.hook_z` (pre-projection) | `blocks.{L}.attn.hook_z` (pre-projection, same as v1) |
| Resolver path | `_resolve_gemmascope` | `_resolve_gemmascope_2` |
| Default config | `extract_features.yaml` | `extract_features_gemma3.yaml` |

## Downstream Pipeline

Every stage after extraction has a `_gemma3` sibling config that inherits its
v1 counterpart and overrides only the Gemma 3 / v2 deltas:

- `configs/extraction/compute_statistics_gemma3.yaml`
- `configs/extraction/precompute_percentiles_gemma3.yaml`
- `configs/extraction/precompute_token_boundaries_gemma3.yaml`
- `configs/extraction/precompute_packed_rows_gemma3.yaml`
- `configs/extraction/validate_token_boundary_overlays_gemma3.yaml`
- `configs/training/conditional_encdec_train_gemma3.yaml`
- `configs/eval/base_eval_gemma3.yaml`,
  `configs/eval/chat_eval_gemma3.yaml`,
  `configs/eval/completion_steering_eval_gemma3.yaml`

A second set of sibling configs targets the **gemma-3-4b-it, layer 22,
width 65k** pipeline specifically:

- `configs/extraction/extract_features_gemma3_l22_f65k.yaml`
- `configs/extraction/compute_statistics_gemma3_l22_f65k.yaml`
- `configs/extraction/precompute_percentiles_gemma3_l22_f65k.yaml`
- `configs/extraction/precompute_token_boundaries_gemma3_l22_f65k.yaml`
- `configs/extraction/precompute_packed_rows_gemma3_l22_f65k.yaml`
- `configs/extraction/validate_features_gemma3_l22_f65k.yaml`

Run each stage exactly as documented in
[`feature-extraction.md`](feature-extraction.md), substituting the matching
`--config-name *_gemma3`. The 5-step precomputation order
(extract → statistics → percentiles → token boundaries → packed rows) and the
CSR shard format are unchanged from v1.

## Gemma-3 l22 f65k End-to-End Pipeline

`runs/gemma3_l22_f65k_pipeline.sh` orchestrates the complete pipeline for
gemma-3-4b-it at layer 22 with 65k-wide SAEs:

```
Step 0  — Extract missing validation shard 369
           (uv run python -m steering.extract_features --config-name extract_features_gemma3_l22_f65k)
Step 0b — Full validation: all 370 shards (CSR invariants + source cross-ref +
           cross-shard metadata including l0_tier)
           (uv run python -m steering.validate_features --config-name validate_features_gemma3_l22_f65k)
Step 2  — Statistics (exact CPU count table; bounded CUDA count reduction + moments)
Step 3  — Percentile overlays
Step 4  — Token boundary overlays
Step 5  — Packed rows
```

> Step 1 — bulk extraction of the already-present shards 0–368 — was completed before this script runs and is therefore not a step here; Step 0 only extracts the single missing val shard 369.

The orchestrator is fail-closed (`set -euo pipefail`): any non-zero exit (e.g.
a validation violation) aborts the pipeline immediately.

**Why does statistics require host-memory headroom?** The exact reducer uses a
dense 65,536 × 31,743 int64 count table (15.4995 GiB) on the host; it never
allocates that table in VRAM. `compute_statistics_gemma3_l22_f65k.yaml` pins
`n_sae_features=65536`, while `device=cuda` accelerates the moment
accumulators and the bounded sparse exact-count reduction. Dense count-table
updates and finalization remain on CPU.

**Prerequisites:**

- `HF_TOKEN` with access to the gated `google/gemma-3` and `google/gemma-scope-2-*`
  repositories.
- ~16 GB host RAM for the offloaded fp32 TransformerLens companion model
  (see [Hardware Note (sm_120)](#hardware-note-sm_120) above and
  [Feature Extraction Pipeline](feature-extraction.md) for the OOM-root-cause
  fix description).

## Feature-filter on a not-yet-ingested v2 SAE

Neuronpedia does not host every gemma-scope-2 SAE. Running the
[feature-filter](feature-filter.md) against a v2 SAE Neuronpedia hasn't
ingested no longer aborts: each feature is recorded with the
`not_published` outcome (HTTP 404), excluded from the error-rate abort, and
the run exits **0** with a warning. Only genuine failures (5xx/auth/judge)
still count. Note this bumped the feature-filter `SCHEMA_VERSION` 2 → 3, so
old in-progress runs cannot be resumed across the change.
