# Architecture

This document describes the model architectures, training systems, and key design decisions in nano-gpt-pretrain-steer.

## Overview

The project implements two complementary model architectures:

1. **Decoder-only GPT** — A modern GPT-2 class causal language model for pretraining
2. **Encoder-Decoder GPT** — Extends the decoder with a bidirectional feature encoder and cross-attention layers for SAE feature conditioning

Both share the same decoder backbone defined in `nanochat/gpt.py`.

## Decoder Architecture (`nanochat/gpt.py`)

### GPTConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_len` | 1024 | Maximum sequence length (project default) |
| `vocab_size` | 32768 | Token vocabulary size |
| `n_layer` | 26 | Number of transformer layers |
| `n_head` | varies | Number of attention heads (derived from `n_embd / head_dim`) |
| `n_kv_head` | varies | KV heads for GQA (defaults to `n_head` = MHA) |
| `n_embd` | varies | Embedding dimension (derived from `depth * aspect_ratio`) |
| `head_dim` | 128 | Per-head dimension |
| `window_pattern` | "SSSL" | Sliding window pattern per layer cycle |

### Key Design Choices

- **No positional embeddings** — Uses Rotary Position Embeddings (RoPE) instead
- **No bias** — All linear layers are bias-free
- **RMSNorm** — Functional (no learnable parameters), used instead of LayerNorm
- **ReLU^2 activation** — MLP uses `relu(x)^2` instead of GELU/SwiGLU
- **Untied embeddings** — Token embedding and lm_head are separate parameters
- **Logit softcap** — Output logits clamped to ±15 via `tanh(x/15)*15`
- **Vocab padding** — Vocabulary padded for DDP and tensor core alignment

### Attention

`CausalSelfAttention` supports:

- **Group-Query Attention (GQA)** — Configurable `n_kv_head` for memory efficiency
- **QK normalization** — L2-normalized queries and keys for training stability
- **Sliding window** — Per-layer "S" (short window) or "L" (full context) pattern
- **Value embeddings** — ResFormer-style alternating layers with learned value bias
- **Flash Attention** — FA4 (Blackwell/Hopper), FA3 (Hopper), Triton FlexAttention,
  then SDPA. FA4 requires the locked `flash-attn-4` CuTeDSL package in addition
  to PyTorch; run `bash scripts/setup_fa4.sh` on the target GPU to exercise the
  production compiled forward and backward path.

### Per-Layer Scalars

Each layer has learnable scalars:
- `resid_lambdas` — Residual stream scaling (initialized to 1.0)
- `x0_lambdas` — Input stream scaling (initialized to 0.1)

These implement a pre-block interpolation with two independent scalars (not a complementary pair):
```
x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # pre-block interpolation
x = x + attn(...)   # standard residual
x = x + mlp(...)    # standard residual
```
`resid_lambdas` scales the running hidden state and `x0_lambdas` scales the original input (`x0`). Both are independent learnable scalars per layer -- they do not sum to 1.

## Encoder-Decoder Architecture (`steering/conditional/encoder_decoder/`)

### FeatureEncoder (`encoder.py`)

A bidirectional transformer that processes SAE features:

- **Input:** Feature IDs + discretized magnitude buckets per segment
- **Embedding:** `feature_embed(id) + magnitude_embed(bucket)` (additive)
- **Value embeddings:** `encoder.value_embeds[i]` — the encoder has its **own** ResFormer-style value embeddings, one table on every other layer (`has_ve`, last layer always included), mixed into `v` through an input-dependent `ve_gate`. **These are feature-ID-indexed too**, and collectively they are the *largest* block of feature-ID-indexed parameters in the encoder — bigger than `feature_embed`. (Distinct from the decoder's token-indexed `value_embeds`.)
- **Self-attention:** No causal mask, no positional embeddings (permutation-invariant)
- **GQA + QK normalization** — Same attention design as decoder
- **Global CUDA packing** — All real features in the batch are compacted onto
  one `K = max(32, ceil(sum(mask) / 32) * 32)` token axis. `(batch, segment)` owner
  tensors and a shared FlexAttention `BlockMask` preserve segment isolation.
  There is at most one 31-token tail for the whole batch instead of padding
  every `(batch, segment)` row to the dataloader's `F` extent.
- **Vectorized sparse-mask construction** — Encoder and cross-attention owners
  are compared as complete boolean routing matrices, reduced exactly into
  partial/full 128×128 tiles, and passed to `BlockMask.from_kv_blocks` with the
  unchanged per-token predicate. This produces the same sparse metadata as
  scalar `create_block_mask` without its nested-vmap CPU overhead or compile
  startup.
- **CUDA-only production path** — `FeatureEncoder` no longer ships a padded CPU
  implementation. `FeatureEncoder.forward` returns `PackedEncoderOutput`, and
  precomputed/cached model calls accept only that packed state; its owner
  metadata replaces the redundant padded mask. Differential and cheap CPU
  semantic tests use a test-owned padded SDPA oracle.

**Table height (issues #695/#714).** The feature-ID-indexed tables — `feature_embed` **and** every `value_embeds[i]` — are sized to `V = model.feature_vocab_size`, the number of **effective features**: mask-allowed original ids minus cluster members consolidated into another representative. With neither a restrictive mask nor active post-mask consolidation, `V == n_sae_features` and the module is byte-identical to the full-dictionary path. Original ids are remapped to compact rows inside `FeatureEncoder.forward` (`_to_compact`), after CUDA packing and before the embedding lookups, so **all callers keep speaking original SAE ids**. `magnitude_embed` is bucket-indexed and is never compacted. See [encoder-decoder-training.md § Encoder Feature-Dictionary Compaction](../pipelines/encoder-decoder-training.md#encoder-feature-dictionary-compaction).

`feature_embed.weight` normally starts from `Normal(0, 1)`. During
encoder-decoder pretraining, `initialize_encoder_embeddings_from_sae` can
replace that table after random initialization with projected SAE decoder
directions (`W_dec`). `random` uses a deterministic random projection from SAE
activation dimension to encoder dimension; `svd` uses the top right-singular
directions. Both modes rescale the projected table so its average row norm
matches that of the freshly randomized **full** `(n_sae_features, d)` table —
never the compact one, even when compacted. When compacted the two modes take the
row subset at *different* points: `svd` **before** projection (its covariance is
therefore the sub-dictionary's), `random` **last**, after projection and rescale,
which is what keeps it bitwise identical to an uncompacted run. `null` is the
default no-op. With static clusters,
`initialize_encoder_embeddings_cluster_vectors: mean` — the DEFAULT policy —
replaces each retained representative with the fp32 mean of its mask-surviving
members at a mode-specific stage. `random` averages after full-table projection
and normalization but before its final effective-vocab gather. `svd` averages
the oriented full `W_dec` first, then selects the effective-vocabulary rows and
computes covariance/eigenvectors over that averaged sub-dictionary. Thus the
model's SVD basis describes the dictionary it actually uses. Non-cluster rows
and singleton survivors remain bitwise identical to the explicit
`representative` policy. Without clusters (or with `null` init) `mean` is an
inert bitwise no-op; a mismatched SAE width or a model built with the wrong
effective vocabulary is refused before `W_dec` is oriented or any weight is
overwritten.

Architecture is configured via ratios of the decoder:
- `n_embd: 0.5` → Encoder dim = 50% of decoder dim, rounded to multiple of 128
- `depth: 0.5` → Encoder depth = 50% of decoder depth

### CrossAttentionLayer (`cross_attention.py`)

Inserted at every `frequency`-th decoder layer:

1. Keep decoder queries packed as `(B, T, C)` and project them exactly once.
2. Consume encoder keys/values on the same global `(1, K, C)` real-feature
   axis. Precomputed encoder state remains packed across model boundaries;
   dense precomputed outputs are not a production compatibility path.
3. Build one data-dependent FlexAttention `BlockMask` from vectorized
   `seg_map`/packed-owner equality, then share it across all cross-attention
   layers.
4. Run block-sparse FlexAttention and project the packed `(B, T, C)` output.

Decoder causal self-attention retains the locked FA4 backend on Hopper and
Blackwell. Both packed data-dependent regions use the fused Triton Flex backend:
PyTorch 2.13's FA4/CuTe backward cannot lower their genuinely dynamic
`BlockMask`s. `FeatureEncoder` and `CrossAttentionLayer` are packed and
CUDA-only; padded SDPA exists only as a test-owned differential oracle.

The packed length is content-dependent even when dataloader `B/S/F` extents are
fixed. Encoder and cross-attention therefore use separate dynamic compile
regions, with their data-dependent tensor and sparse-mask dimensions marked
dynamic. The outer decoder can remain statically compiled, preserving FA4's
static-shape path while crossing packed-length boundaries reuses the inner
graphs. In cross-attention, only the K/V projection's leading token extent is
content-dependent; its weight matrix and input/output channel dimensions remain
static. Q and output projection operate on fixed `(B,T)` extents and do not
mathematically require dynamic shapes. They currently share the same inner
dynamic graph as K/V and attention to keep one compiled call per layer. A short
local split experiment was bimodal and is not evidence that either layout is
faster; a repeated H100/H200 benchmark remains required. The encoder's
projections and MLP all consume the packed K axis, so their leading extent is
dynamic. Rounding K to the next 64 does not permit `dynamic=False`; only a
single fixed maximum extent would do so.

- **seg_map contract** — Non-zero values must be monotonically non-decreasing along the T dimension (real segments are contiguous and ordered). Some callers reserve `seg_map=0` for empty/padding slots, so zero may repeat after real segments. A non-zero value followed by a smaller non-zero value violates this canonical ordered-segment contract. `EncoderDecoderGPT.forward` validates the shared packed routing state once before its decoder-layer loop; direct `CrossAttentionLayer.forward` callers retain the same fail-loud validation. Packed routing itself performs no query scatter.
- **Wrapper context_token_ids contract** — `TargetAwareEncoderDecoderEvalWrapper`
  consumes per-row unpadded prompt token lists via
  `conditioning_context(context_token_ids=...)`. These token IDs are the
  lookup key for the full-context SAE feature cache; missing rows are fatal.
  Enforced by `FeatureIndex.lookup`, which raises `KeyError` naming the shard and
  doc id rather than returning `None` (see "FeatureIndex Fail-Fast Contract" in
  [encoder-decoder-training.md](../pipelines/encoder-decoder-training.md#featureindex-fail-fast-contract)).
- Uniform-initialized `c_proj` — `Uniform(±sqrt(3/(n_head*head_dim)))`. Previously
  zero-initialized for residual-friendly startup, but all-zero weights make FP8
  tensorwise scaling degenerate (`amax=0` → extreme scale factors → inf
  gradients on the first backward). The uniform init keeps magnitude small
  while ensuring `amax > 0`. See `cross_attention.py:init_weights` and
  `tests/nanochat/test_cross_attention.py::TestFP8InitSafety`.
- No separate encoder output projection — `c_kv` projects directly from encoder dim
- Segment-bound validation — packed-row manifests store the **raw** maximum
  observed segment length (`max(seg_token_end - seg_token_start)`).
  Cross-attention still validates
  `pos_in_seg.amax() + 1 <= raw_max_seg_size` in eager and compiled execution,
  but CUDA no longer uses that global bound as an allocation extent. The CPU
  SDPA reference rounds its private extent to `FP8_ALIGNMENT` as before.
- Cross-shard batch padding — when a precomputed-row batch spans multiple
  packed shards, `_stack_batch_parts` pads each row to the batch-level
  `max_n_seg` so all output tensors (`enc_*`, `seg_doc_id`, `seg_map`,
  `fp_targets`) share the same leading `(B, n_seg)` dims. Contract is
  asserted at the stack step; see the "Cross-shard Batch Padding Contract"
  section in `docs/pipelines/encoder-decoder-training.md` for details.

### EncoderDecoderGPT (`model.py`)

Composes all components:

```
Input text → Token embedding → Decoder blocks (with cross-attention) → LM head → Logits
                                     ↑
Feature IDs + buckets → FeatureEncoder → per-segment representations
```

Cross-attention layers are interleaved at regular intervals in the decoder stack.

The model's `setup_optimizer` creates separate parameter groups for decoder, encoder, and cross-attention components, allowing independent learning rates. Encoder and cross-attention matrix weights use the same Muon optimizer as the decoder but can be tuned independently. The `estimate_flops` method accounts for encoder and cross-attention costs, amortizing encoder FLOPs across decoder tokens using the ratio `R = (n_segments * avg_features_per_segment) / sequence_len`.

### Tokenizer Warning

Public HF ChatCORE evaluation is an isolated inference path: Qwen2.5-3B-Instruct
and SmolLM3-3B use their own pinned model/tokenizer/template together through
`steering/eval/hf_chat_eval.py`. Their token IDs never enter nanochat's Engine or
SAE conditioning. See the [HF evaluation contract](../pipelines/evaluation.md#public-instruction-tuned-hf-baselines).

This codebase uses **two tokenizers, never mixed**: Gemma (256k vocab) for `steering/` feature extraction and RustBPE (32,768 vocab) for `nanochat/` training. The bridge between them is a **char-level segment boundary protocol**:

- Extraction stores per-segment character ranges (`segment_char_starts` / `segment_char_ends`) relative to the decoded source text. SFT shards use the rendered conversation text, truncated after its last assistant turn, with the decoder BOS stripped before extraction.
- Training resolves those char ranges against the *training* tokenizer's char offsets via `steering.segmentation.token_to_char_offsets` and `assign_tokens_to_segments`. Full-sequence SFT routing lives in `steering/conditional/encoder_decoder/full_sequence_routing.py`; feature loading lives in `steering/conditional/encoder_decoder/sft_features.py`.
- The canonical char-window `CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS` (defined in `steering/segmentation.py`) absorbs end-of-turn marker residue at segment boundaries. The extraction side (`scripts/preprocess/precompute_token_boundaries.py`) references the constant directly; all consumer sites (full-sequence SFT data prep in `steering/conditional/encoder_decoder/full_sequence_routing.py`, target-real filter in `steering/eval/target_segments.py`, axbench-bpb seg_map in `steering/eval/axbench_bpb/seg_map.py`, target-aware encdec eval in `steering/conditional/encoder_decoder/eval_adapter.py`, completion-steering eval in `steering/eval/multiseg_generator.py:generate_for_document` per-target build) delegate to the centralised helper `align_char_ranges_to_decoded_prefix` (Layer 1, parallel-list shape) or `align_segments_to_decoded_prefix` (Layer 2, `ContextFeatureSegment` tuple shape). Each consumer site emits a site-scoped variant of the canonical `segmentation.char_end_extended` counter (e.g. `segmentation.char_end_extended.full_sft`, `.target_real_filter`, `.completion_eval`, `.encdec_eval_context`, `.bpb_seg_map`) with `direction` + `delta` attributes, and raises `ValueError` for any delta above the window in either direction. See `tests/steering/test_cross_tokenizer_threshold_unified.py` and `tests/steering/test_align_segments_to_decoded_prefix.py` + `test_align_char_ranges_to_decoded_prefix.py` for the Layer 1/2 contracts.
- Tokenizer identity is verified at extraction-time and training-time via
  `compute_tokenizer_checksum` (vocab + merges + BOS id). For conditioned SFT,
  the source tokenizer's decoded BOS width is a separate coordinate-system
  fact: startup derives it from the first train and active-validation
  feature/parquet shard pairs and requires it to equal
  `feature_source_bos_char_len`.
- AxBench Stage 1b/1c exact-boundary helpers obtain the RustBPE nanochat tokenizer through `steering.eval.axbench_base.tokenizer_cache`, which memoizes one instance per process. This is an operational construction cache only: it does not replace the Gemma extraction tokenizer, change the checksum, or enter artifact identity.

Gemma 2 and Gemma 3 ship **different tokenizers**, so v1 (GemmaScope / Gemma-2) and v2 (gemma-scope-2 / Gemma-3) feature shards must never be mixed. They stay isolated because the feature-path convention embeds an `m_{model}` component (distinct model name → distinct directory tree), and the L0Tier overwrite guard (`assert_l0_tier_compatible`) refuses to write into a feature dir whose recorded L0 representation disagrees with the current run — so a v1 (integer-L0) directory and a v2 (tier-L0) directory can never collide on disk.

If you see a "Two tokenizers, never mix" reminder in `AGENTS.md`, this is the section it points to. The full Segment Assignment mechanics are below; the cross-tokenizer behavior for full-sequence SFT lives in `steering/conditional/encoder_decoder/full_sequence_routing.py`.

### Segment Assignment

Decoder tokens are assigned to feature segments by character-offset overlap. The
former fixed-stride assignment path has been removed; character-level assignment
is required because extraction and training can use different tokenizers.

For SFT, segment char boundaries are stored over the converted conversation text
(truncated after its last assistant turn), matching the text column produced by
`scripts/preprocess/convert_sft_to_parquet.py`. During training,
`build_full_sequence_segment_routing()` maps those full-source char segments
onto nanochat token positions and shifts the decoder BOS coordinate via
`feature_source_bos_char_len` when the source tokenizer's BOS has a different
decoded width. Before generator construction, the trainer verifies that value
against an independently derived width from the first train shard and, when
active, the first validation shard; missing evidence or disagreement aborts
startup.

For multi-turn conversations, conditioned SFT always routes features across the whole converted sequence (the conversation up to its last assistant turn). The LM target tensor independently supports `sft_loss_mask=full_sequence` (the default, every non-padding target) or `completion_only` (assistant text and tool calls, excluding user turns and tool outputs). Changing the loss mask never changes segment geometry or feature conditioning.

One-shot turn-aware inference reuses that exact rendering/extraction/alignment
chain. The full prompt ends in a marker-only assistant turn; its extracted real
slot is validated and removed, the final real user/tool-output chunk predicts
S0, and `<assistant_start>` becomes RustBPE generated-window row 1. The source
Gemma segment size comes from the online-SAE lineage, while its context window
comes from the lineage-bound, exact-SHA feature-statistics snapshot. The
generated RustBPE cadence comes only from self-conditioning post-training
metadata.
Interactive multi-turn re-extraction is intentionally separate and tracked in
[#1627](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1627).

### Remote Packed/Selection Body Architecture

The verified remote c1024 mode uses strict rank-striped transport. For signed
inventory index `i`, rank `i % world_size` owns the only downloader process.
That spawned process owns its own Hugging Face/Xet state and publishes the
selection body followed by its paired packed body through hidden,
integrity-verified stages. Every DDP rank still descriptor-pins, decodes, and
consumes every body required by its unchanged row stripe.

Training-side materializers are cache-only: they publish requester-specific
demands and wait for the nominal owner. Per-body locks coordinate owner
publication, readers, and owner-only eviction but never choose an owner.
Receipts bind signed artifact identity to the exact final inode. Owner failure
is terminal, and the remote body protocol permits multiple torchrun nodes when
they expose a demonstrably shared
cache/lock namespace.

This compressed-body transport precedes both complete-batch CPU prefetch and
the optional pageable/pinned/CUDA staging pipeline. Those downstream stages do
not change body ownership, row striping, semantic run identity, or checkpoint
state. See [Strict Rank-Striped Remote Body Prefetch](../pipelines/remote-cache-materialization.md).

## Optimizer (`nanochat/optim.py`)

### MuonAdamW

A mixed optimizer that applies different algorithms per parameter type:

| Parameter Type | Optimizer | Default LR |
|---------------|-----------|------------|
| Embeddings | AdamW | 0.3 |
| Unembedding | AdamW | 0.004 |
| Scalars (gates, biases) | AdamW | 0.5 |
| Weight matrices | Muon | 0.02 |

**Muon** features:
- Nesterov momentum
- Polar Express orthogonalization (5 iterations)
- Variance reduction (NorMuon)
- Cautious weight decay
- Parameters stacked by shape for efficient batched operations

### DistMuonAdamW

Distributed version with ZeRO-2 style sharding:
- Small parameters: all_reduce
- Large parameters: reduce_scatter + all_gather
- 3-phase async communication overlapped with compute

## Inference Engine (`nanochat/engine.py`)

Production inference with:
- **KV cache** — Pre-allocated tensors with full shape
  `[n_layers, batch, time, n_kv_heads, head_dim]`; each layer view passed to
  attention is `(batch, time, n_kv_heads, head_dim)`
- **Prefill → decode** — Initial context processing then autoregressive generation
- **Tool use** — Auto-executes Python expressions inside `<python_start>...<python_end>` blocks
- **Streaming** — Generator yielding `(token_column, token_masks)` tuples
- **Batch generation** — Multiple sequences in parallel
- **Sampling** — `nanochat.sampling.sample_next_token` is the canonical public
  sampler (NaN/Inf guard + greedy/top-k/full-softmax branches) shared by the
  nanochat engine and steering's scalar/shared-RNG paths. The fixed-row,
  independently seeded self-conditioned batch is restricted to fixed-token
  checkpoints whose completions cannot reach a refresh boundary. It checks all
  active rows once with that canonical guard, then mirrors only its finite-input
  row arithmetic in physical order so it can retain one generator per row
  without paying one host synchronization per row. Turn-delimited AxBench rows
  use the scalar runtime because the marker begins a metadata-sized RustBPE
  window and one completion may publish multiple conditioning states.

### Project-owned chat-generation seams

`steering/eval/inference_engine.py` keeps KV-cache K/V tensors on the model
device. On CUDA, its authoritative position counter lives on CPU for the resolved SDPA
cached-attention path, avoiding a CUDA scalar read in every decoder layer.
FA3 keeps device-resident positions. Stacking and compaction use the same
allocation policy; backend selection remains fixed while a cache is live.
See [RLVR position-placement measurements](../performance/rlvr-sdpa-kv-positions.md).

`steering/eval/chat_generation.py` is the shared adapter boundary for the
SFT-style completion prompt. It deliberately does not modify the vendored
Engine. `render_prompt_for_chat_completion()` builds a fresh user/empty-
assistant conversation, preserves the tokenizer's exact
`render_for_completion()` IDs, and validates the `<assistant_end>` stop ID.
Generation consumers preserve the complete Engine token stream; masks remain
available for conditioning and loss-position logic rather than presentation
filtering.

Lockstep target-real routing uses a per-decoder-position assignment vector.
`target_token_assignments[t]` includes mask `0` forced tool-output positions,
not only positions selected for scoring. The vector is forward-filled across
the absolute span between the first and last routed positions, so KV-cache
positions and target slots cannot diverge after a tool result. Pooled targets
and legacy no-gap lockstep outputs are unchanged.

## Flash Attention Backend (`nanochat/flash_attention.py`)

Auto-detects GPU capabilities:

| Backend | GPU | Features |
|---------|-----|----------|
| FA4 | Hopper (sm90) + Blackwell (sm100) | Fastest, sliding window |
| FA3 | Hopper (sm90) | Fast, sliding window, KV cache |
| SDPA | All | Fallback, GQA emulation |

Selection: `attn_impl: auto` tries FA4 → FA3 → SDPA in order.

Faster-GCG and GEPA candidate losses use the KV-cache backend: SDPA on the
current RTX 4090 path and potentially FA3 on Hopper. The test-only full-sequence
oracle can instead use Triton FlexAttention. Identical tokens and K/V state do
not imply identical floating-point results across those kernels, so deployment
is gated per hardware/backend by finite bounded objective drift, deterministic
shared-prefix trajectories, and explicit search-quality comparisons against the
oracle. Exact oracle trajectory equality is not required: shared-prefix scoring
is the versioned production search algorithm, not a numerically transparent
backend substitution. Encoder-decoder cache reuse additionally requires exact
prefix `seg_map` and encoder
feature/vocabulary/cluster state, not merely equal prefix token IDs.

## FP8 Training (`nanochat/fp8_utils.py`)

Optional FP8 quantization via torchao:
- Only applies to Linear layers with dims divisible by 16 and min dim >= 128
- `disable_fp8()` context manager for evaluation (temporarily converts back to BF16)
- Configured via `fp8: true` and `fp8_recipe: tensorwise` in training configs

## Mixed Precision Contract

The codebase uses two dtype-management patterns that must not be confused.

### Canonical paths

1. **Training scripts** wrap the forward pass (and optional loss computation) in a
   `torch.amp.autocast(device_type=..., dtype=torch.bfloat16)` context manager,
   constructed once at startup and aliased as `autocast_ctx`. On non-CUDA devices
   `autocast_ctx` is `contextlib.nullcontext()`. Scripts following this pattern:
   `scripts/train/base_train.py`, `scripts/train/chat_sft.py`, `scripts/train/conditional_encdec_sft.py`,
   `scripts/eval/base_eval.py`, `scripts/chat/chat_cli.py`, `scripts/eval/chat_eval.py`,
   `scripts/chat/chat_web.py`, `scripts/train/chat_rl.py`, `scripts/chat/steer_chat_cli.py`,
   `scripts/train/training_progression.py`, `scripts/benchmark/benchmark_attention.py`, and the
   sanity-check/eval adapter modules under `steering/eval/sanity_check/` and
   `steering/conditional/encoder_decoder/eval_adapter.py`.

   The isolated SmolLM3 cooldown uses the same FP32-parameter/BF16-autocast
   contract through Transformers `Trainer`, with FA4 attention and explicit
   shifted labels. Its three 4,096-token contexts are flattened to one row so
   reset `position_ids` select the FA varlen document path; it supplies no
   attention mask. See
   [`docs/pipelines/smollm3-cooldown.md`](../pipelines/smollm3-cooldown.md).

2. **Inference + KV-cache paths** (`nanochat/engine.py`) do NOT rely on an ambient
   autocast context. Instead the KV cache is pre-allocated with an explicit dtype
   (`torch.bfloat16` on CUDA, `torch.float32` otherwise — see
   `InferenceEngine.generate_batch`). The `Linear` subclass in `nanochat/gpt.py`
   casts its weights to the activation dtype on every forward, so once embeddings
   enter the trunk in `COMPUTE_DTYPE` (via `x = x.to(COMPUTE_DTYPE)` in
   `GPT.forward`), every matmul runs in that dtype without needing autocast.
   AxBench-Upstream's bounded direction-cell batching preserves this rule: it
   stacks independently prefilled batch-one KV/packed-encoder states, then calls
   the encoder-decoder model directly with a uniform-position cache and one
   residual addend per physical row. It is restricted to a decode horizon before
   the first self-conditioning boundary; the general state-growing wrapper
   remains deliberately batch-one.

3. **`conditional_encdec_train.py`** also follows the module-level cast pattern
   and does NOT open an autocast context. This is deliberate: the training loop
   runs entirely off `COMPUTE_DTYPE`-cast embeddings + `Linear` cast-on-use.

### `Linear` cast-on-use

`nanochat/gpt.py::Linear.forward` returns
`F.linear(x, self.weight.to(dtype=x.dtype), ...)`. Master weights stay fp32 for
optimizer precision; matmuls run in `x.dtype`. This is NOT a replacement for
autocast — it is a complement. Autocast additionally handles intermediate ops
(RMSNorm is `F.rms_norm` which is autocast-aware, sigmoid/softmax, etc.) when a
script chooses to open a context.

### `GPT.forward` output dtype contract

The forward method explicitly casts logits back to fp32 at the softcap step
(`logits = logits.float()` before `tanh(logits / softcap) * softcap`). The
returned logits / loss are therefore **always float32**, independent of whether
an autocast context is active. Trunk inputs are cast to `COMPUTE_DTYPE` (bf16 on
modern CUDA, fp32 elsewhere), but CUDA autocast can promote intermediate
RMSNorm outputs to fp32. In particular, decoder Q/K normalization produces
fp32 while V remains bf16 during conditioned SFT validation and training.
FlexAttention casts Q/K/V to the active autocast dtype at its boundary; the
project's native SM12 FA4 adapter applies the same conversion before CuTe,
which requires matching low-precision inputs. Without autocast the adapter
preserves the caller's dtypes.

### When to use which

| Site type | Canonical pattern |
|-----------|-------------------|
| Training script forward/backward | `with autocast_ctx:` |
| Offline eval / CLI forward | `with autocast_ctx:` |
| `InferenceEngine` KV-cache loop | explicit dtype on cache + `Linear` cast-on-use (no autocast needed) |
| `harmfulness_eval.py` native model load | explicit `model.to(dtype=bfloat16)` to align KV cache dtype (documented workaround) |
| `harmfulness_eval.py` repository-HF load | strict local safetensors load directly in BF16 with SDPA; no native `InferenceEngine` or autocast context |

Any off-pattern site must carry an inline `# AUTOCAST_OFFPATTERN: <reason>`
comment explaining why it diverges, and the test at
`tests/test_autocast_canonical.py` must be updated to list the exception.

Repository-HF harmfulness generation is a narrow adapter, not a second serving
stack. It validates one immutable `step_NNNNNN` export, renders the saved chat
template, left-pads one configured outer batch, and invokes one cached
Transformers decode for that batch. A narrow logits processor applies
temperature and top-k to the complete score matrix, then samples every row in
one matrix-wide operation from a batch-level generator. Its RNG trajectory
therefore depends on outer-batch partitioning while a fixed configuration
remains repeatable. SmolLM3 is supported immediately; a
Pythia/GPT-NeoX exporter is compatible when it emits the same shared repository
metadata/tokenizer contract. Arbitrary external or Hub models, HF steering,
and progression evaluation remain outside this boundary.

## Scaling Laws

Base training uses compute-optimal scaling:

- **Batch size:** Auto-computed from model parameters and target token budget
- **Learning rate:** muP extrapolation based on depth
- **Data-to-params ratio:** `target_param_data_ratio` controls training duration (default 8.25, compute-optimal ~10.5)
- **Warmdown schedule:** 50% of training spent in learning rate warmdown

## Evaluation

### CORE Metric (`nanochat/core_eval.py`)

Multiple-choice, schema, and language modeling tasks from the DCLM evaluation suite. Evaluates in-context learning capabilities.

### Bits Per Byte (`nanochat/loss_eval.py`)

Tokenizer-agnostic metric that normalizes loss by UTF-8 byte length. Immune to vocabulary size changes, enabling fair comparison across tokenizers.

### Conditioned BPB (`steering/conditional/encoder_decoder/loss_eval.py`)

For encoder-decoder models, BPB is computed with real encoder feature inputs
from the validation feature rows.

### Feature Prediction Head (`feature_prediction.py`)

An optional auxiliary loss that predicts next-segment SAE features from decoder hidden states. This encourages the decoder to internalize feature information during training.

- **Four loss modes:**
  - `MAX_POOL` (default) — Per-feature max of head logits across the segment positions, then BCE/MSE against the next segment's target. Symmetric across the segment, runs identically at training and inference.
  - `FULL_SEGMENT` — Mean of per-token losses across all source tokens
  - `LAST_TOKEN` — Use only the last token's hidden state per segment
  - `BEST_TOKEN` — Select the token whose hidden state best predicts features (per-element min loss). **Cannot be used at inference** (needs the next segment's target features); not compatible with self-conditioned generation.
- **Two loss functions:** BCE (binary cross-entropy) and MSE (mean squared error)
- **Configurable layer attachment** — The prediction head can tap into any decoder layer's hidden states
- Defined in `steering/conditional/encoder_decoder/feature_prediction.py`: `resolve_layer_index()`, `compute_feature_pred_loss()`, `pool_fp_logits()`

**Off by default.** `feature_prediction: null` in `configs/training/conditional_encdec_train.yaml` reflects the PR #697 ablation result, not a structural restriction. An uncompacted head is `Linear(dec_n_embd, n_sae_features)`; when a mask or static clusters resolve, it is `Linear(dec_n_embd, V_effective)`. Compact targets are scattered directly into the effective width without a full-width dense intermediate: pretraining consumes certified packed CSR rows, while conditioned SFT validates its per-conversation source segments inline.

**Residual head capacity (issues #1681 and #1719).**
`FeaturePredictionConfig.block` selects `mlp` or `transformer`;
`head_layers` is the total nonnegative depth. Defaults are `mlp` and `0`.
Both block choices at depth zero retain the existing linear module. Positive
depth wraps its original `weight`/`bias` parameter objects in
`ResidualFeaturePredictionHead`, preserving device, dtype and training mode.

The model supplies `h = norm(captured_hidden)`. Each MLP block applies
`h = h + c_proj(relu(c_fc(norm(h))) ** 2)` with bias-free `D → 4D → D`
matrices. Each transformer block first applies
`h = h + Attention(norm(h))`, then the same MLP residual. Transformer blocks
reuse the decoder's width, query/KV head counts, RMS normalization, RoPE and
Q/K normalization. They have independent weights, no value-embedding gates,
cross-attention, residual-scaling parameters, or final normalization.
Every head block uses the captured decoder layer's absolute token positions
and causal/sliding window. Decoder self-attention does not reset at document
boundaries; the head preserves that behavior. Existing segment routing and
next-segment target validity still govern the loss.

New input matrices use the decoder's uniform initialization; attention and MLP
output matrices start at zero. Allocation and initialization preserve global
CPU/CUDA RNG state. This retains the original linear predictions and existing
gradients at initialization under matched arithmetic. Optional FP8 conversion
keeps the nonlinear module and its final projection; exact equality to a
separately FP8-converted linear head is not guaranteed.

`EncoderDecoderGPT.apply_feature_prediction_head` owns the position/window
context for training, feature-only/paired output, and head-only RLVR. Hidden-only
output remains normalized captured decoder states, allowing RLVR to freeze the
trunk while differentiating through all head parameters. The complete head uses
the existing AdamW unembedding-rate group, including width scaling and weight
decay. Parameter/FLOP accounting includes head attention.

Transformer KV slots follow decoder slots. `model.num_cache_layers` is decoder
depth plus transformer-head depth (27 + 4 = 31 for the default 3B preset), while
`dec_config.n_layer` still counts only decoder blocks. Allocate the full cache
before self-conditioned prefill. Complete caches execute head blocks for every
forwarded token, including LM-only boundary commits, and skip the feature
projection when unused. Decoder-only caches remain valid for LM-only inference;
they cannot later supply feature predictions. Forward validates cache capacity
and positions before writes, advances the shared position once, and restores
counters after a failure. Existing rewind, copy, stack, reset and compaction
operate on every slot.

Inference and resume require explicit canonical `block` and `head_layers`
metadata and strict state loading. Legacy metadata requires the
[explicit conversion utility](../pipelines/feature-head-metadata-migration.md).
See
[SFT residual feature heads](../pipelines/encoder-decoder-training.md#sft-residual-feature-heads)
for the explicit fresh-SFT upgrade and resume contract.

**Inference-time use:** `EncoderDecoderGPT.forward(idx, targets=None, return_fp_logits=True)` returns `(logits, fp_logits)` with shapes `(B, T, V_tok)` and `(B, T, feature_prediction_width)`. The latter is `n_sae_features` for an uncompacted checkpoint and `V_effective` for a compact checkpoint. `fp_logits_only=True` follows the same rule. In self-conditioned generation, compact head row `j` is paired with `feature_vocab.original_ids[j]` before selection, so every downstream feature id remains in the original SAE domain. An optional fixed `SelfCondConfig.fp_logit_bias` uses that same head-row space and is added to detached float32 pooled logits before sigmoid. The union-typed `score_policy` field's interpolation mode ranks pooled float32 logits directly at weight zero, retains `q = 2*sigmoid(logits)-1` for interpolation toward a strict finite DiffMean vector at interior weights, and performs stable highest-active/lowest-remaining-inactive selection into a two-bucket encoder. Sign masking applies the feature mask first, then uses stable descending-positive/ascending-negative DiffMean rank to cap eligible pools; pooled logits still perform final selection inside those pools. Factor zero removes the rank cap while retaining sign and mask rules. Its target rows retain the same compact vocabulary ordering. See the [self-conditioned generation pipeline](../pipelines/self-conditioned-generation.md#dense-target-score-interpolation) for construction, split-prefill, and selection contracts. On-policy post-training keeps pretraining-source checkpoints uncompacted; its source-aware SFT path supports compact mask/cluster vocabularies and performs STE in compact row space.

`forward(..., logits_to_keep=k)` projects only the final `min(k, T)` token
positions through the LM head when `k > 0`; the default zero retains every
position. This requires `targets=None`. Decoder states, KV-cache positions,
and feature-prediction logits still cover the full input. The feature-only
path already skips the LM head. AxBench-Adv activation-L2 uses `k=1` because
it consumes a captured residual and discards the LM logits; see
[GCG performance](../performance/axbench-adv-gcg.md).

**Prediction, selection, and bucket values:** packed pretraining supervises
percentiles after their persisted uint8 encode/decode roundtrip, while
conditioned SFT supervises continuous `magnitude_to_percentile / 100` values.
Separately, selection consumes either full-distribution ranks or raw magnitudes,
and buckets use either a uniform percentile grid or raw statistics boundaries.
The axes cannot be collapsed: unclustered percentile sampling selects by rank
but buckets the returned raw magnitudes against statistics. None is recoverable
from head shape or weights. Self-conditioned feedback uses one continuous
normalized-rank score space and does not quantize cutpoints to reproduce packed
target storage. `SelfCondConfig` therefore requires versioned
`FeatureSelectionProducer` and `FeatureBucketProducer` values. Raw-magnitude
self-conditioned selection is supported only for per-feature quantile cutoffs;
cross-feature raw ordering fails closed. Clustered SFT thresholding now uses the
same full-distribution percentiles as its FP targets and clustered sampling.
Legacy checkpoints need
evidence from their actual producers because saved metadata does not always
stamp these choices.

Checkpoint segmentation metadata is reconciled rather than chosen by
precedence: top-level `sft_segment_mode`, online-SAE lineage, and SFT
self-conditioning data must all be well-formed and agree. Absence remains the
legacy `fixed_tokens` identity only when no modern SFT witness requires more.
For `turn_delimited_fixed_tokens`, inference additionally requires the positive
top-level `turn_self_cond_segment_size_rustbpe` that post-training measured
from its TRAIN artifacts and recorded beside `sft_segment_mode`. The
turn runtime commits each assistant input exactly once under its outgoing
state, uses the Kth row's outgoing logits for the next sample, and publishes
the incoming slot for the next forwarded input; the historical fixed runtime
retains its two-forward boundary replay unchanged. All chat structural token
IDs are fail-closed stops before forwarding.

Interactive turn Chat reuses that one-shot runtime in a separate epoch for each
submitted user message. Its full logical transcript is re-rendered and real
features are obtained through the same producer as evaluation; previous generated
assistant/tool text is therefore real context in the next epoch. A candidate
owns its wrapper, KV allocation, ledger and RNGs until a successful atomic swap.
Rewinds copy KV into a candidate instead of overwriting committed history, and
never-reused operation revisions reject stale previews/actions. Fixed Chat keeps
its original continuous-stream driver. See [serving](../pipelines/serving.md#full-conversation-epochs-turn-delimited-checkpoints).

AxBench-upstream is the turn-aware evaluation surface: it runs the exact
conditioned-SFT prompt selector and cross-tokenizer converter, revalidates
checkpoint lineage after target load, and uses the scalar turn runtime. Static
fixed-window cache/loss consumers reject turn checkpoints or artifacts before
geometry-sensitive conversion, generation, scoring, or final output rather
than warning and evaluating mismatched geometry. Some entry points first load
checkpoint metadata or a model to obtain the evidence needed by that guard.

**On-policy post-training:** eligible FP-head checkpoints can train the head
policy (or the full model) from rollout-conditioned LM loss with a two-pass,
one-hop STE. The packed-pretraining source requires an uncompacted checkpoint;
the source-aware SFT path supports an exact compact mask/cluster vocabulary and
performs STE in compact row space. Dense and selected-score estimators are both
supported; negative conditioning columns invert the score gradient. See
[`docs/pipelines/selfcond-post-training.md`](../pipelines/selfcond-post-training.md)
for the exact estimator boundary and checkpoint contract.

**Structured-SFT sequential GRPO:** a separate experimental trainer can treat
one authoritative structured SFT conversation as a sequential conditioning
environment. Unlike generation replay, its single segment map is input-aligned:
the action pooled from segment `s` becomes conditioning slot `s+1`, so it first
changes the token prediction after that next segment's first input token. The
completed conditioning history is scored once with detached teacher-forced LM
loss; rewards normalize only across trajectories of the same conversation;
then sparse action traces are replayed differentiably. Weighted ordered
sampling has an exact without-replacement likelihood. Bernoulli-first selection
uses an explicit selected-feature surrogate. The default
[head-only scope](../pipelines/sft-self-conditioning-grpo.md#training-scope)
trains the complete feature-prediction head with all other policy parameters
frozen and NTP disabled. `train_feature_head_only=false` restores full-policy
training except the untied LM head. The optional auxiliary next-token objective
requires that override, also unfreezes the LM head, and accumulates its gradients
with policy replay before the same clip and optimizer step. The runtime
uses DistMuonAdamW's optimizer-owned rank averaging and never wraps the model
in DDP. See
[`docs/pipelines/sft-self-conditioning-grpo.md`](../pipelines/sft-self-conditioning-grpo.md)
for data, causal, resume, initial-conditioning, and qualification contracts.

**Verified-reward RLVR:** the separate turn-delimited RLVR trainer generates
both tokens and feature conditioning from question-only prompts, then scores
the generated answer with the native benchmark verifier. Its canonical
batch-one full-prefix feature policy is shared by collection and gradient
replay; native cached decoding and calculator transitions supply the token
trajectory. Raw prompt encoding and each prior sampled feature segment are
reconstructed in their original order and call shapes. Only
causally live feature decisions enter the rollout mean. Every connected
parameter except the untied LM head trains through feature-action likelihoods;
`train_feature_head_only=true` instead trains the complete feature head while
running the frozen encoder, decoder, and head-input normalization under
`no_grad`, then applying the head with gradients enabled. This skips trunk
backward work and checkpoint recomputation. Token likelihoods
have no gradient objective. This is an intentional
length-normalized partial-gradient surrogate. Prompt-local reward advantages,
local clipping followed by optimizer-owned averaging, immutable prompt-cache
authority, and explicit complete-checkpoint recovery are described in
[`encoder-decoder-rlvr.md`](../pipelines/encoder-decoder-rlvr.md).

### Feature Mask — Selection-Layer + Structural Enforcement (`feature_mask.py`, `feature_vocab.py`)

`steering/conditional/feature_mask.py` is the single source of truth for the optional feature blacklist/whitelist.

**`FeatureMask` dataclass** holds a validated, canonicalized `(mode, ids)` pair (sorted + de-duplicated). Two logically-identical masks from any source are `==` and hash identically.

**Training enforcement sites** — a masked feature can never appear as:
- **Positive conditioning:** `apply_to_positive` filters `feature_ids` (and any co-aligned tensors) to the allowed universe.
- **Negative universe:** all three negative-sampling implementations — `sample_inactive_features`, `sample_inactive_features_batched` (both in `sampling.py`), and `sample_delta_negatives` (in `delta.py`) — call `allowed_universe_mask` to narrow the eligible pool before any random draws. `allowed_universe_mask` returns `~member` for blacklist and `member` for whitelist, gated behind a single `if feature_mask is not None` so no-mask paths are truly bit-identical (I5).
  The batched sampler preserves RNG draw order while deduplicating candidates: repeated candidate IDs keep their first drawn occurrence via a stable per-row sort, which is required for fixed-seed determinism under duplicate candidates.
- **FP-target:** full-width runs retain the legacy masked-column zeroing path. Compact runs apply the mask, map each surviving original member to its post-mask representative, max-reduce sibling percentiles, and scatter directly into effective-vocabulary rows. The compact loss needs no second mask gate because its columns already are exactly the allowed representative universe.
- **Self-conditioned feedback:** `SelfConditionedFeatureProvider._apply_selection` restricts the feature-ID and score tensors to the allowed universe before calling the threshold or sampling selection, so predicted features fed back to the encoder are always within the allowed set.

**Eval enforcement sites** — the mask is also threaded into three eval paths via the same shared primitive:
- **Conditioned-BPB eval** (`scripts/eval/base_eval.py:_resolve_bpb_selection_dir`): `feature_mask` is folded into `compute_selection_hash` so the eval resolves the same masked `sel/<hash>/` dir used at training; a manifest assertion guards against drift.
- **Eval-context-cache identity** (`steering/eval/feature_selection_hash.py:compute_canonical_feature_selection_hash`; `steering/eval/cache_identity.py`): `feature_mask` and the canonical static-cluster descriptor are included conditionally in the cache-identity hash, so different effective prompt-conditioning policies use distinct namespaces; no-mask/no-cluster hashing is byte-identical to the legacy value (cache-identity I1).
- **Completion-steering eval** (`steering/eval/completion_steering_eval.py:select_from_tensors` + `assert_selections_within_mask`): feature selection is restricted to the allowed universe; a defense-in-depth assertion rejects any masked ID (including from a stale `feature_selections.yaml`).
- **Clustered in-training steering** (`steering/eval/clustered_inline_cache.py` + `inline_steering_eval.py`): splice-counterfactual runs consume the already-certified derived validation selection instead of re-thresholding raw features. A bounded two-pass provenance join materializes fixed target prefixes and different-document donors in original representative IDs. Probe raw values retain representative scale; the primary percentile metric converts every surviving member with its own statistics row and takes the maximum within each representative group.

**Structural enforcement (issues #695/#714)** — `steering/conditional/feature_vocab.py` plus `effective_feature_vocab.py`. The resolved mask and static-cluster consolidation size the encoder's parameters. A masked or consolidated-away feature does not merely go unselected: it has no embedding row at all.

- **`FeatureVocab`** (frozen dataclass, `{n_sae_features, original_ids}`) is the encoder's compacted dictionary. `effective_feature_vocab(feature_universe, n_sae)` — `feature_universe` being a `FeatureClusterConsolidation` (which carries its resolved mask), a `FeatureMask`, or `None` — composes the mask-only constructor with the already-resolved consolidation and is a **pure, deterministic function** of `(mask, canonical cluster content, n_sae_features)`. The existing `bl_`/`wl_` and `fc_` leaves already pin those inputs, so the derived vocabulary adds no hash or path identity. With no clusters it literally delegates to `FeatureVocab.from_mask`; identity normalizes to `None`.
- **`FeatureEncoder`** sizes `feature_embed` and every `value_embeds[i]` to `V = vocab.size`, and remaps ids original → compact in `_to_compact` (after the FP8 pad, before the lookups — the one choke point all call sites funnel through, including the five that call `model.encoder(...)` directly). No-vocab ⇒ literal early-return of the same tensor object ⇒ zero added ops (I5). Slice-init (`_init_feature_indexed_`) draws the full-width table so the RNG stream matches an uncompacted run bit for bit.
- **Feature domains:** artifacts and callers use original SAE IDs. Cluster members map to the smallest mask-surviving original ID (the representative). Only model-weight and FP-target boundaries translate representatives to compact rows. The FP head has one output row per effective representative and initializes by drawing the full reference table once, selecting representative rows, and preserving the reference RNG state.
- **Out-of-vocab guard:** `torch._assert_async` — sync-free, zero graph breaks, fires in eager *and* compiled. Necessary because `nn.Embedding(-1)` silently returns the **last row** under `torch.compile` instead of raising.
- **Self-describing checkpoints:** persistent `encoder.feature_vocab_ids` (`int64[V]`) + `encoder.feature_vocab_n_sae` (`int64`) buffers, registered only when compacted. `feature_vocab_from_state_dict()` recovers `(n_sae_features, vocab)` from the state dict alone; `EncoderDecoderGPT.assert_vocab_consistent()` is required after every `load_state_dict(assign=True)` (which replaces the persistent buffer while the derived non-persistent map keeps its old value).
- **Feature-GCG:** the universal optimizer supports uncompacted and compacted
  checkpoints for one or many members. On compacted checkpoints it proposes only
  original IDs in `V`; compact-row translation is internal and returned bags retain
  original IDs. Static clustered checkpoints are accepted only when the caller
  supplies a validated cluster feature contract; without one they remain rejected.
  AxBench-Base and BPB-Adv both have cluster-aware dispatchers that resolve and
  thread that contract. BPB-Adv's shared intervention builder accepts exact
  source-certified DiffMean/linear-probe payloads and identity-bound persisted
  precomputed bags; clustered scoring and sample replay use those validated
  in-memory/source snapshots rather than reopening baseline paths. See
  [AxBench-BPB-Adv](../evaluation/axbench-bpb-adv.md#clustered-scoring-runnable-with-one-recorded-coverage-limit).
- **Compact FP source contract:** compact pretraining targets require precomputed packed rows whose manifest contains a completed `fp_target_source_integrity` certificate. Conditioned SFT instead validates unique original IDs per conversation segment, then applies the same mask/map/max-reduce layout directly to its full per-segment feature lists. Inference-time self-conditioned generation consumes compact head rows by translating its candidate vector back to original SAE ids. The packed-pretraining self-conditioning post-training source remains uncompacted-only; its source-aware SFT path supports exact compact mask/cluster identity and compact-row STE.

See [encoder-decoder-training.md § Eval-side enforcement](../pipelines/encoder-decoder-training.md#eval-side-enforcement) and [§ Eval-correctness contract](../pipelines/encoder-decoder-training.md#eval-correctness-contract) for full details.

**No-clobber via conditional hash key:** `compute_selection_hash` (in `packed_batch_io.py`) adds `feature_mask` to its SHA-256 payload ONLY when the mask is not `None`. A no-mask run's hash is byte-identical to the legacy 6-key payload, so every existing `sel/<hash>/` shard on disk remains valid (invariant I1).

**Checkpoint path:** `feature_mask_dirname` computes `{bl|wl}_{16hex}` from a SHA-256 of the canonical mask JSON. `append_mask_leaf` returns `path` unchanged when the mask is `None`, preserving the legacy checkpoint path byte-identically (invariant I2). Training scripts (`conditional_encdec_train.py`, `conditional_encdec_sft.py`, `convert_decoder_to_encdec.py`, `run_encdec_sft_pipeline.py`) call this helper after resolving the mask.

**Manifest metadata and assertion:** the `sel/<hash>/` manifest records `metadata.feature_mask` as `canonical_dict(mask)` or `None`. On each run, `conditional_encdec_train.py` asserts the manifest value matches the configured mask — a mismatch raises `FeatureMaskError`, catching configuration drift. Legacy manifests without a `feature_mask` key are treated as `None` (backward-compatible with no-mask runs, I5).

**Legacy feature-clustering boundary:** the former `feature_clustering` producer,
overlay, model-aware runtime, and checkpoint/manifest schema remain removed.
Current configs containing that singular key and artifacts carrying its legacy metadata
are rejected before they affect hashes, caches, dataloaders, evals, or reports.

### Static Feature Clusters — Preselection Transform + SFT/TUI Consumers (`feature_clusters.py`)

Issue #714 adds a separate, path-only `feature_clusters` producer contract for
static schema-v1 YAML/JSON definitions. It is used by selected-feature
precomputation and precomputed encoder-decoder pretraining; conditioned SFT
replays the same mask-then-max-percentile fold inline per conversation. SFT
sampling supports `weighted` and `top_k_percentile`; threshold selection supports
only `feature_percentile` and `feature_percentile_segment_topk`, because clustered
selection has percentiles rather than representative raw magnitudes. The
completion TUI also consumes the canonical definition and descriptor persisted
in checkpoint metadata. It neither discovers clusters nor restores the legacy
machinery.

The architectural order is fixed. Ordinary selection resolves the complete
`FeatureMask`, maps allowed members to the minimum allowed ID in each source
cluster, and max-reduces normalized percentiles per representative. Clustered
`splice_counterfactual` selection resolves that same mask on the current and
would-have-followed sides separately, independently max-reduces each side's
normalized percentiles through the same representative map, and only then
subtracts the representative-level values. Positive/negative sampling and
bucket assignment therefore consume representative deltas and clustered
current percentiles. Final rows contain representatives only, still expressed
as original SAE IDs. The generic cluster primitive remains independent of
delta/splice code; `delta.py` owns the narrow two-side orchestration.

The cluster descriptor continues to identify ordinary consolidation. A second
canonical `cluster_delta_semantics` identity is present only when static
clusters and `feature_delta.mode=splice_counterfactual` are both active. This
invalidates every pre-correction clustered-splice selection without changing
unclustered splice hashes or delta-free clustered base-validation hashes.

Clusters participate in model construction only through the derived effective
feature vocabulary. Its ids are the mask-allowed universe with every
`consolidated_away` member removed; representatives remain, so an active cluster
shrinks every feature-indexed embedding table and an away id is out of vocabulary.
With no clusters the constructor literally follows the legacy mask-only path, and
clusters that are inert after masking do not change the shape. The only
original-to-compact translation remains `FeatureEncoder._to_compact`. Cluster
identity also conditions the `sel/<hash>/` artifact namespace, complete manifest,
checkpoint `fc_` leaf, and checkpoint metadata, so the vocabulary is derived from
already-pinned inputs rather than becoming a new identity source. Loaders scan the final clustered CSR artifacts for exact
shape/dtype/alignment, bucket bounds, row uniqueness/disjointness, and
representative-only IDs before yielding data.

The completion TUI resolves the checkpoint mask and clusters before loading
documents, requires the matching base-aligned percentile overlay, and performs
mask-then-max-percentile consolidation in representative space. Composite rows
therefore display and condition on original SAE representative IDs while the
encoder itself uses the cluster-consolidated effective vocabulary derived from
the same consolidation (see the structural-enforcement section above) — the
launch gate rejects a clustered checkpoint carrying a stale mask-only
vocabulary. Generation asserts that every conditioned ID is both mask-allowed
and a representative.

Cluster-aware prompt caches support in-training CORE and sample generation. They
apply the mask, convert each member magnitude with that member's statistics,
max-reduce normalized percentiles at the representative, and retain original
representative IDs for the existing eval adapter and encoder remap. Their
identity is deliberately delta-free, pinned to the checkpoint's real
training-preselection seed, and includes the canonical mask and cluster
descriptor -- but **omits** the statistics SHA-256 from the hash itself
(statistics identity is independently pinned by the clustered feature
contract and cache manifest instead). The manifest separately records the
effective-vocabulary fingerprint.

Unsupported live/fused pretraining selection, serving, on-policy
self-conditioning post-training, and gradient ranking paths reject active
cluster config or clustered artifact metadata at their boundaries. An
already-loaded compact model may use the inference-time feedback provider; that
path consumes the checkpoint vocabulary and does not infer a new cluster
contract. Feature-GCG likewise rejects clusters unless its entry point receives
a validated cluster feature contract. The standalone cluster-aware
orchestrators are narrower than the generic eval stack:

- AxBench-Base accepts one stage at a time for source construction
  (`0a`-`3b`, excluding prompting/LoRA stages) plus exact Stage `3e` context
  enumeration and Stage `4` generation, under stage-specific explicit method
  allowlists.
- BPB-Adv resolves one contract, preflights the clustered source before marker
  installation, and threads the contract through its stage dispatcher. Its
  artifact-production and identity layers are implemented. Source-certified
  DiffMean/linear-probe payloads and persisted precomputed-bag clustered
  encoder-decoder method families are runnable (#1008 removed the stale
  intervention guards; #1050 made resume reuse committed bags by identity).
- Clustered BPB-Adv sample generation can additionally replay encoder-decoder
  families from the validated context cache and persisted bags. One paired
  source snapshot, exact candidate payload/table, and exact in-memory aggregate
  result are shared across target-model generation; candidate, aggregate,
  concept, and direction paths are not reopened after validation. Legacy
  unclustered sample generation remains decoder-only.

Clustered in-training steering is limited to the precomputed
splice-counterfactual validation path described above; ordinary/threshold inline
steering and generic standalone eval surfaces remain unsupported unless they use
a validated cluster-aware contract.
The completion TUI support above is a separate checkpoint-metadata consumer.
See [encoder-decoder-training.md § Static Feature Clusters](../pipelines/encoder-decoder-training.md#static-feature-clusters-preselection-only).

Stage 0c API proposals retain member diversity instead of inventing a blended
cluster label. Each allowed member gets an independently inferred operational
description from its validated raw top-activation evidence; flattened
candidate/seed slots cycle through the sorted member IDs with a versioned global
round-robin assignment. Published Neuronpedia descriptions never enter this
generator, and local cluster-wide `amax` remains authoritative for acceptance.

**nanochat read-only:** `nanochat/checkpoint_manager.py` was not modified. Unpinned callers use the existing `_find_checkpoint_leaf` recursion (which finds the `bl_/wl_` leaf naturally). The `harmfulness_eval` pinned-discovery path is a known follow-up (Task 15b).

### Autointerp Intruder Filter — Whitelist Producer

`scripts/preprocess/precompute_autointerp_filter.py` is an offline producer for `feature_mask.mode=whitelist`. It does not mutate base feature shards, statistics, percentile overlays, token boundaries, packed rows, or selection hashes. It reads local feature shards, scores singleton feature targets with intruder detection, and writes `accepted_feature_ids.txt`.

Feature shards are used to find candidate source documents, then those documents are re-forwarded through the SAE to build token-level windows with token-aligned target activations. Positive rows highlight target-active tokens, and intruder rows are target-non-activating windows with Delphi-style false-positive highlights. The default judge uses the calibrated `highlight_context_token_first_v1` prompt with `sae_realistic_v1` few-shot examples; the Delphi-compatible prompt remains available through explicit judge settings. Candidate windows are deduped by rendered source-window identity before trial sampling, and each sampled trial rejects overlapping same-document windows across all presented options. The token-example cache is keyed by source fingerprints, model/SAE/backend/dtype/token-window settings, and singleton target identity. Parsed-result identity combines the target/trial/sample settings with `token_window_v1` digests of sampled refs, plus judge model/CoT, prompt/few-shot names and their 16-hex content-version fingerprint, and provider render context. Window digests cover token IDs and activations but not `token_strings`, and the rendered final prompt is not directly hashed; renderer/token-string changes therefore require an explicit identity-method/version or context bump. The configured prompt-length cap, parse retries, RPM, and concurrency also do not enter parsed-result identity; the cap is applied only to cache misses, while the cap/retries remain semantic plan fields and RPM/concurrency are operational provenance. Only successfully parsed trial results are cached. `trial_audit.jsonl` records bounded machine-readable samples with token strings, token activations, highlights, and judge responses; `trial_audit.yaml` keeps the same samples concise for human review with the final prompt, rendered examples, prediction, correct index, and response. Prompt/rendering and judge failures abort the run instead of being hidden in output rows; an enabled Neuronpedia fetch failure is non-fatal and instead records a `None` description for that feature. `top_k` counts singleton feature targets.

All stable-key and truncated prompt-version guarantees assume no relevant digest
collision; they are versioned identity contracts, not literal prompt-byte
hashing.

Judge concurrency is a single-process, in-thread-pool layer around that same
scorer -- there is no separate process, plan, or coordinator. `judge.max_concurrent`
(default 8) bounds how many uncached intruder trials `_evaluate_trial_wave`
(`runner.py`) judges at once via a `ThreadPoolExecutor`; `judge.requests_per_minute`
(default 60) rate-limits admission of judge `complete()` calls, and `max_judge_calls`
caps total admitted uncached trials for the whole run despite its name (each
admitted trial can still make `parse_max_retries + 1` `complete()` calls before
transport retries). `max_concurrent=1` takes a serial fallback loop with
identical semantics to the concurrent path.

Determinism does not depend on wall-clock completion order. `_resolve_all_trials`
resolves every (target, trial) globally in one canonical single-threaded
admission pass first: cache hits are filled in place and misses become work
items in strict (target, trial) order, so the exact set of admitted trials is
fixed *before* any concurrency, and a judge-call budget caps the same prefix
regardless of `max_concurrent`. Only judge execution itself runs concurrently.
The calling thread commits each successfully parsed result through
`IntruderTrialCache` as it observes the completion; workers never touch SQLite.
`_evaluate_trial_wave` sorts the resulting success/error outcomes by global
index before returning, so result assembly and the error selected when several
fail remain byte-identical between serial (`max_concurrent=1`) and concurrent
runs. A drained wave caches the same logical successful-result set in either
mode, including successes whose global index follows a failed trial; an external
kill can preserve only commits completed before termination. These contracts are
proven directly by `tests/steering/autointerp_filter/test_concurrency_parity.py`.

A configured `document_blacklist_path` (`document_blacklist.py`) is parsed once
into a canonical set of `(shard_stem, doc_id)` entries and bound to a SHA-256 of
that parsed set, independent of file formatting/comment/whitespace differences.
`TokenExampleSampler.build_trials` filters both positive and non-activating
candidate refs against this set before any quantile grouping or trial sampling,
so blacklisted documents can never be sampled as a positive or intruder example
for any target. The runner manifest binds `document_blacklist_path` and
`document_blacklist_sha256` as identity fields: pre-blacklist manifests missing
both fields, and intermediate manifests with a null path and a missing/null
digest, both normalize to the canonical empty-set digest for compatibility;
any other combination is compared exactly.

One model/SAE/tokenizer loads exactly once per run and is shared across every
target and trial (`scripts/benchmark/measure_autointerp_model_vram.py` measures
its resident VRAM footprint). There is no plan/prepare/launch/combine step, no
shared on-disk source-index or trial-cache coordination across processes, and
no `/proc/self/fd` lock anchoring: the only cross-run protection is a single
`flock`-based lockfile over the whole output directory
(`AutoInterpFilterRunner.run`), guarding against two runs racing into the same
`output_dir`.

Token-example materialization has two in-process execution modes. Batch size
`1` is the unchanged serial reference. The production default `128` groups
singleton targets by feature-ID quotient/remainder into fixed 128-slot cache
blocks, while retaining one independent serial-policy state machine per target.
The model never receives a padded block: each real slot uses its exact serial
`[1, L]` input on one of two persistent worker-local CUDA streams, and dummy
slots run no forward. CPU preparation may overlap through one worker. The main
thread consumes GPU results in canonical slot order and alone owns state
transitions, validation, atomic block cache writes, and logs. This is neither
model replication nor process sharding. The bulk
extension completes all token examples before the existing judge-concurrency
phase begins. Unsupported backend/artifact/runtime combinations fail rather
than adapting or falling back. Normal batch execution also requires a frozen
profile with exact model/SAE/input identity plus runtime and calibration-data
digests; the permissive profile seam exists only inside the calibration
harness. See [Autointerp Intruder Filter](../pipelines/autointerp-filter.md) for exact
commands, file layouts, overwrite policy, and the serial/concurrent parity ladder.

Downstream use is unchanged:

```yaml
feature_mask:
  mode: whitelist
  path: out/autointerp_filter/accepted_feature_ids.txt
```

### Feature Delta Conditioning (`delta.py`)

`steering/conditional/delta.py` provides delta-aware feature selection for tracking how features change across segments.

- **`compute_doc_deltas()`** — Processes a complete document while tracking
  cross-segment percentile history for two of the three `DeltaMode` values:
  - `prev_segment` — Compare against the immediately preceding segment
  - `cum_max` — Compare against the cumulative maximum across all prior segments
- **`compute_doc_counterfactual_deltas()`** — Handles the third `DeltaMode`,
  `splice_counterfactual` (issue #596): no cross-segment state is carried;
  each post-splice segment is compared directly against an externally supplied
  would-have-followed reference read from a spliced derived dir's schema-v2
  `_splice_provenance` sidecar
  (`steering/conditional/splice_provenance.py`). Segments with no reference
  get plain absolute-percentile conditioning instead. Raw delta inputs and
  counterfactual references must be non-empty; empty segments raise. The live
  loader also verifies that the positions/count of delta-scored segments match
  the sidecar splice positions. See
  [encoder-decoder-training.md](../pipelines/encoder-decoder-training.md) and
  [feature-shard-splice.md](../pipelines/feature-shard-splice.md).
- **`sample_delta_negatives()`** — Samples negative features weighted by their delta magnitude
- Supporting types: `DeltaMode`, `DeltaSamplingWeights`, `FeatureDeltaConfig`

This is different from the dataset-level differential selector in
`scripts.preprocess.extract_common_features`. That script compares mean SAE activation
percentiles between two complete datasets, such as `safety_sft` and pretraining
`base`, and writes a static `{feature_id: bucket}` YAML for conditioned evals.
Pass `baseline_dir=.../base` there when you want the eval conditioning signal to
prefer features that activate more on the steering dataset than on a reference
dataset.

**Packed-row limitation:** `prev_segment` and `cum_max` delta modes remain
live-loader-only because packed rows do not encode carried document-history
state. `splice_counterfactual` is the exception: its reference identity is
offline in the spliced feature dir's schema-v2 `_splice_provenance` sidecar, so
`scripts/preprocess/precompute_selected_features.py` can materialize final
`sel/<hash>/` shards from packed-row provenance. At train time
`precomputed_data_loader` accepts `feature_delta.mode=splice_counterfactual`
only with complete compatible selection shards; it never falls back to fused
runtime selection for any delta mode. See
[encoder-decoder-training.md](../pipelines/encoder-decoder-training.md).

Remote ordinary clustered preselection is a packed-only transport wrapper
around that same delta-free selector. Inventory-v2 authenticates compressed
packed bytes and semantic layouts; per-shard schema-v2 receipts authenticate
selection bytes, width, and post-shard RNG state; the final existing consumer
manifest is published last. Raw features, percentile overlays, sidecars, and
`FeatureIndex` are not part of this ordinary execution graph.

Parallel splice preselection is architecturally a producer-only resource
substitution. The parent retains the full first-source validation and freezes
source/overlay discovery into a signed `FeatureIndexCatalog`; workers use
spawn, compact validated sidecars, an optional leased shared source store, and
an optional context-bounded disk mmap for packed rows. Shared-store entries also
publish the eager path's exact stable row order plus a sorted document locator,
so a worker remapping one resident stem constructs only the requested document's
zero-copy views instead of rebuilding every document after each LRU eviction.
Before output state is created, header-derived admission checks the exact
requested topology, including locator payload and construction transients,
against current host/cgroup/filesystem headroom. None of these controls enters
`compute_selection_hash` or the artifact schema, and fixed-clock tests require
byte-identical output against the W=1 reference. See
[resource-safe parallel splice preselection](../pipelines/encoder-decoder-training.md#resource-safe-parallel-splice-preselection-issue-758).

The producer's common masked-negative path also avoids per-segment full-SAE
scratch masks: a narrow CPU single-segment specialization uses cached allowed
IDs and sparse sorted membership, with the dense implementation retained as its
compatibility fallback. Newly serialized clustered splice shards are reloaded,
validated against their signed segment count, and bound to their compressed
digest under an unchanged-file guard. That per-shard certificate replaces the
former completion-time replay of every packed-row body without weakening the
representative-only CSR validation contract. Its signed running maximum CSR
width is required to agree with the remote-loader padding certificate, so the
remote static shape is derived from those same validated selection tensors.

### AxBench-BPB (`steering/eval/axbench_bpb/` + `scripts/eval/axbench_bpb_eval.py`)

Offline, BPB-based variant of AxBench-Base that replaces Stages 4-6
(generation + LLM judge + judge leaderboard) with teacher-forced BPB
scoring on the Stage 1b/1c synth samples. Reuses all of axbench-base's
Stage 0-3 artifacts read-only and writes to a separate output tree. Both synth
corpora must carry the current authenticated-continuation Parquet contract with
the correct positive/negative role; BPB and BPB-Adv fail before scoring on
missing, stale, malformed, or role-swapped continuation provenance. See
[`docs/evaluation/axbench-bpb.md`](../evaluation/axbench-bpb.md) and
[`docs/evaluation/axbench-bpb-advanced.md`](../evaluation/axbench-bpb-advanced.md).
