# Serving & Inference

Interactive chat interfaces for trained models.

## Overview

Five interactive serving modes:

| Script | Interface | GPUs | Features |
|--------|-----------|------|----------|
| `scripts/chat/chat_cli.py` | Terminal CLI | 1 | Basic chat |
| `scripts/chat/chat_web.py` | Web UI (FastAPI) | 1+ | Multi-GPU, streaming |
| `scripts/chat/steer_chat_cli.py` | Terminal CLI | 1 | Fixed-token feature-conditioned generation |
| `scripts/completion_tui/` | Textual TUI | 1 | Fixed-token editable segmented completions, mask-aware conditioning, and clustered-checkpoint inspection |
| `scripts/chat_tui/` | Textual TUI | 1 | Fixed-token self-conditioned multi-turn chat with segment rewind and AxBench steering |

## Chat CLI

Interactive single-GPU chat interface.

```bash
# Interactive chat with SFT model
python -m scripts.chat.chat_cli source=sft

# With RL model
python -m scripts.chat.chat_cli source=rl

# Custom generation parameters
python -m scripts.chat.chat_cli source=sft temperature=0.8 top_k=50

# Single-shot (non-interactive)
python -m scripts.chat.chat_cli prompt="What is 2+2?"

# Specific model checkpoint
python -m scripts.chat.chat_cli source=sft model_tag=d26 step=5000
```

**CLI commands:** `quit`/`exit` to exit, `clear` to reset conversation.

**Config:** `configs/serving/chat_cli.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | sft | Model source (base/sft/rl/safety) |
| `temperature` | 0.6 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `dtype` | bfloat16 | Model dtype |

## Web Chat Server

FastAPI web server with a ChatGPT-style interface.

```bash
# Single GPU
python -m scripts.chat.chat_web

# Multi-GPU (data parallel across workers)
python -m scripts.chat.chat_web num_gpus=4

# Custom settings
python -m scripts.chat.chat_web source=rl temperature=0.8 port=8080 host=0.0.0.0
```

**Config:** `configs/serving/chat_web.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | sft | Model source |
| `num_gpus` | 1 | Number of GPU workers |
| `temperature` | 0.8 | Default sampling temperature |
| `top_k` | 50 | Default top-k |
| `max_tokens` | 1024 | Max tokens per response |
| `port` | 8000 | Server port |
| `host` | 0.0.0.0 | Server host |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat web UI (HTML) |
| `/chat/completions` | POST | Chat API (streaming SSE) |
| `/health` | GET | Health check |
| `/stats` | GET | Worker pool statistics |

### Request Limits

- 500 messages per request
- 8000 characters per message
- 32k total characters
- Temperature: 0-2
- Top-k: 0-200
- Max tokens: 1-4096

## Steered Chat CLI (Encoder-Decoder)

Chat with feature-conditioned generation using encoder-decoder models. The
encoder-decoder path accepts `fixed_tokens` checkpoints only and rejects a
turn-delimited checkpoint immediately after `load_model` returns, before
generation setup. Decoder-only sources are unaffected.

```bash
# Default encoder-decoder model
python -m scripts.chat.steer_chat_cli

# With specific feature conditions
python -m scripts.chat.steer_chat_cli \
    source=encdec \
    feature_conditions='{0: 1, 5: 2}'

# From SFT encoder-decoder
python -m scripts.chat.steer_chat_cli \
    source=sft \
    model_tag=my_model

# Custom generation
python -m scripts.chat.steer_chat_cli \
    temperature=0.6 \
    top_k=50 \
    max_new_tokens=512

# Single-shot
python -m scripts.chat.steer_chat_cli \
    prompt="Tell me about safety" \
    feature_conditions='{42: 3}'
```

**Config:** `configs/serving/steer_chat_cli.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | encdec | Model source (encdec/sft) |
| `feature_conditions` | {} | Dict of {feature_id: bucket_index} |
| `temperature` | 0.6 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `max_new_tokens` | 256 | Max generated tokens |

### Feature Conditions

Feature conditions are a dictionary mapping SAE feature IDs to discretization bucket indices:

```python
# Activate feature 0 at bucket 1, feature 5 at bucket 2
feature_conditions = {0: 1, 5: 2}
```

Bucket indices range from 0 to `num_buckets - 1`. For ordinary positive-only
configs, higher buckets correspond to stronger activation of that feature.
For checkpoints trained with `negative_selection`, the last bucket
`num_buckets - 1` is reserved as the inactive bucket, so stronger positive
activation only ranges over `0 .. num_buckets - 2`.

## Completion TUI

The completion TUI is an interactive editor for segmented encoder-decoder
prompts and SAE feature conditions. Launch it with:

```bash
uv run python -m scripts.completion_tui

# Pin an auto-discovered checkpoint and defer loading its model weights
uv run python -m scripts.completion_tui model_tag=my_model step=5000 lazy=true

# Supply a complete explicit match when discovery cannot find one
uv run python -m scripts.completion_tui \
    model_tag=my_model feature_dir=/path/to/features \
    parquet_dir=/path/to/parquets
```

**Config:** `configs/serving/completion_tui.yaml`. The TUI scans
`$NANOCHAT_BASE_DIR/conditional_encdec_checkpoints/`, recursively finds
checkpoint leaves, matches them to feature directories with the checkpoint's
segment size, and resolves the corresponding parquet corpus. One match is
auto-selected; multiple matches open a terminal chooser. `model_tag`, `step`,
`feature_dir`, and `parquet_dir` select or override a match. `lazy=true` defers
model-weight loading until the first generation request. There are no TUI
`feature_mask` or `feature_clusters` keys: the checkpoint metadata is
authoritative.

The completion TUI is `fixed_tokens`-only. It rejects a turn-delimited
checkpoint before feature-statistics setup; a non-lazy launch may already have
loaded model weights, while lazy launch rejects before that load. It rejects
turn-delimited feature statistics before feature rows or generation are
started. Use the AxBench-upstream one-shot inference path for current turn-aware
generation; interactive turn-aware editing/re-extraction is tracked in
[#1627](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/issues/1627).

### Masked and clustered checkpoints

At launch, the TUI resolves `feature_mask_canonical`, verifies that it exactly
matches the model's compact `FeatureVocab`, and keeps every UI and generation
condition in original SAE-ID space. Masked IDs are absent from document rows,
resampling, and search. A final generation-boundary assertion rejects any
masked ID introduced by corrupted state or a manual edit.

For a clustered checkpoint, the TUI additionally resolves the canonical
schema-v1 clusters, requires the matching base-aligned percentile overlay, and
replays the producer's ordinary live-selection order: apply the mask,
max-reduce member percentiles onto each fixed representative, select positives
with producer-equivalent thresholds or the same batched samplers, sample
negatives with the producer's batched sampler, then derive buckets from the
normalized percentile. Generation accepts representatives only.

A surviving cluster with at least two allowed members appears as one editor
row:

```text
[n composite features] <representative description>
```

The representative's description is the cluster label. Here `n` is the stable
number of mask-allowed members, not the number active in the current segment.
Press Enter on the composite row to open its member submenu. The representative
is marked `*`; each member shows its original SAE ID, cached description, and
segment evidence (`p0.xx` from the stored uint8 percentile divided by 255, or
`—` when absent). Enter opens that member's ordinary detail and per-member live
highlight, Escape returns detail → submenu → editor, and searching for a member
adds or selects its representative instead of creating an invalid sibling row.

Current limitations are fail-loud except where explicitly noted:

- Clustered checkpoints with `feature_delta.mode=splice_counterfactual` are
  accepted only when `neg_delta_ratio=1.0`. Evaluation documents have no splice
  provenance, so this configuration reduces exactly to the ordinary
  percentile-selection path. Other delta modes and partial ratios are rejected
  because the TUI cannot reproduce their live semantics.
- Clustered gradient-ranked selection is rejected, as are clustered
  raw-magnitude thresholds; only ordinary percentile-based clustered selection
  is supported.
- Composite max-across-members live highlighting is not implemented. Member
  detail provides the existing per-member live highlight instead.
- Pre-existing behavior is unchanged for unclustered `feature_delta`
  checkpoints: the TUI currently displays them without delta semantics. This
  is a known limitation, not a clustered-support claim.
- A pre-existing focus-cycling path can leave an open overlay alive across a
  document swap, so the overlay may refer to the previous document. Clustered
  support does not change that behavior; it needs a dedicated follow-up issue
  rather than a compatibility-sensitive fix in this change.
- Unclustered feature search retains its pre-existing asymmetric duplicate
  handling. The clustered path deduplicates in representative space, but the
  legacy path is deliberately unchanged to preserve its behavioral contract.
- Search results can show a cluster member without a composite prefix when its
  representative has no current row. Selecting that result still redirects to
  and adds the representative, so model conditioning remains correct.
- Splitting a segment preserves the left segment's evidence for the original
  pre-split span. This matches the existing feature-copy behavior; evidence is
  not recomputed for either new text span during the split.

## Chat TUI

The chat TUI serves compatible encoder-decoder checkpoints with continuous
fixed-token streams or turn-delimited conversation epochs. `target_meta_path`
is mandatory:

```bash
uv run python -m scripts.chat_tui \
    target_meta_path=/path/to/checkpoint/meta_000749.json
```

Setup reconciles all checkpoint segmentation witnesses. Fixed-token checkpoints
use the continuous stream described below. Turn-delimited checkpoints with a measured cadence
use full-conversation epochs: each new user message re-renders the
complete logical history, obtains its real Gemma/SAE features through the shared
one-shot producer, and prefills a fresh wrapper/KV cache. A raw turn-SFT parent
without measured `turn_self_cond_segment_size_rustbpe` is not a supported target.

An existing conditioned-SFT leaf can acquire that field without weight training:
measure it with `measure_turn_self_cond_segment_size_rustbpe` against the original
TRAIN feature/parquet family and trainer-rendered conversations, preserving its
row capacity and loss-mask settings. Add only the measured top-level integer and
retain the original metadata plus measurement provenance. Do not substitute the
Gemma source window size or a guessed cadence. Such SFT leaves use their complete
online-SAE lineage for extraction identity; stale `user_config.sae` defaults and
an absent pretraining `conditioning` block do not override that identity. Their
canonical feature vocabulary and matching statistics remain required and checked.

The default `feature_statistics_path` is usable only when its SHA-256 matches
the checkpoint binding. Override it when launching a checkpoint bound to a
different statistics artifact.

### Required artifacts

- `target_meta_path` must name either pretraining-shaped encoder-decoder
  metadata with its conditioning block or a conditioned-SFT leaf. For an SFT
  leaf, setup resolves the recoverable conditioning contract from the leaf and
  walks to the nearest pretraining ancestor for inherited conditioning
  identity such as the statistics binding, seed, and SAE.
  A turn post-training leaf instead uses its complete online-SAE lineage and
  the shared one-shot setup loader; ancestor defaults never replace its prompt
  selection, missing-percentile policy, statistics identity, or extraction geometry.
- A raw turn-SFT leaf (no post-training block) may omit `pretrain_meta_path`:
  its online-SAE lineage is then the sole identity witness for the stale default
  `user_config.sae` and an absent `user_config.conditioning`. For turn
  post-training leaves with those gaps, set `pretrain_meta_path` to the exact
  pretraining checkpoint metadata, as with the shared one-shot loader. Whenever
  it is supplied, its `model_tag` and SAE must agree with the target and the
  online-SAE lineage; it is never accepted and ignored. It supplies the missing
  pretraining identity without overriding the turn prompt policy or measured
  cadence. Fixed Chat retains ancestor recovery and rejects a non-null
  `pretrain_meta_path`.
- `feature_statistics_path` must name the statistics file bound by SHA-256 to
  that checkpoint contract. A missing file, a digest mismatch, or incompatible
  statistics metadata is a launch error.
- `prepared_dir` must be a prepared AxBench concept directory. Concept metadata
  is read from `generate/metadata.jsonl`; concept training texts are loaded
  when a steering signal first needs to be fit, and fitted signals are cached
  under `signal_cache_dir`.
- The standard-SFT validation Parquet family, resolved as `standard_sft_val`,
  is required for `/convo`. It is not replaced by `prepared_dir` and has no
  chat-TUI config override.

### Slash commands

**`/convo`.** Lazily opens the standard-SFT validation catalog, lets you choose
a conversation and the user turn at which to start or continue, checks that
the selected prefix leaves generation headroom, and installs the preceding
messages as the conversation prefix. On pick, the TUI re-renders the selected
source row and UTF-8 byte-compares it with the row stored in the
`standard_sft_val` Parquet shards; a missing family, missing row, or mismatch is
an error rather than a best-effort load.

**`/steer`.** Opens the segment and AxBench-signal panes. Rewinding discards
everything after the selected boundary and, when that boundary leaves an
assistant turn open, automatically continues until the turn is closed. In fixed
mode, modifying a later self-conditioned segment rewinds to the preceding boundary,
replays the recorded `self_cond_segment_size` token window, and installs the
edited feature dictionary as an explicit segment at the boundary. Editing the
initial self-conditioned segment or a real-prefix segment instead re-prefills
the original prompt with the explicit override; every modification discards
the old tail and completes an open assistant turn. In turn mode, only predicted
rows of the current assistant response are editable; real history is read-only.
Rewinding a real row starts a fresh full-history extraction epoch at that
exchange's complete user message. The signals pane selects
off, interpolation, or restriction steering, fits enabled AxBench concept
signals on demand, and applies the new policy at the next boundary.

**`/clear`.** Resets the transcript to its conversation-start state: the
prefix chosen by `/convo`, or an empty conversation if no prefix was chosen.
It discards the live wrapper and KV state, restarts the session RNG streams,
and clears the one-step modification retry state; the active steering settings
survive.

**`/seed N`.** Accepts a positive integer and records it as the session seed.
It does not perturb the running conversation's RNG streams; it takes effect at
the next RNG restart, such as `/clear`, loading a new `/convo` prefix, or a
fresh-take rewind. The status bar marks the seed as pending until a reset.

**`/quit`.** Exits the TUI. Modals that mutate the session are disabled while a
generation is in flight; concept fitting remains available. Enter during active
generation requests cooperative cancellation. Fixed mode force-closes the
assistant turn; turn mode discards the provisional response and restores the
last committed conversation, reporting cancellation without consuming a reset.

### Steering blend semantics

For enabled concept signals `d_i`, strengths `f_i`, and the model's predicted
scores `q`, interpolation uses additive pulls. The semantic formula from the
steering contract is:

`scores = (1 − Σ f_i/6)·q + Σ (f_i/6)·d_i`

It requires `Σ f_i ≤ 6 + 1e-9`; the small epsilon only absorbs floating-point
accumulation around the visible strength budget of 6. A zero total is the
explicit no-steer baseline.

Restriction is an AND-of-pools. Its per-concept and pooled formulas are:

`L_i^+ = clamp(round(N × (1 − f_i/6) + 2 × positive_mean × f_i/6), 1, N)`

`L_i^- = clamp(round(N × (1 − f_i/6) + 2 × negative_mean × f_i/6), 1, N)`

`pos_i = top L_i^+ eligible rows ranked stably by d_i descending`

`neg_i = top L_i^- eligible rows ranked stably by d_i ascending`

`pos = ∧ pos_i`, `neg = ∧ neg_i`

`N` is the number of eligible head rows after the effective vocabulary and
feature mask. Each signal is ranked independently before intersection; the TUI
does not average restriction-mode DiffMeans. Python ties-to-even rounding and
the checkpoint's independent positive/negative configured means match the
AxBench policy. Factor zero removes the rank cap but retains each signal's sign
and mask rules. Empty required intersections fail when the policy is activated.

### Continuous-stream conditioning (fixed-token checkpoints)

One conversation is one continuous self-conditioned decode stream. The first
send performs a single prefill of the complete rendered prompt using real
feature segments produced by live Gemma/SAE extraction. Every later forced or
sampled token passes through the self-conditioned decode step, so subsequent
user turns ride the currently predicted segments instead of starting a new
prefill. The checkpoint's `sequence_len` is a hard cap on the whole
conversation; the TUI does not slide or truncate the context. These are the
preserved legacy fixed-window semantics, not a claim of turn-delimited training
parity.

### Full-conversation epochs (turn-delimited checkpoints)

Every submitted user message builds a candidate epoch while the previous
transcript, wrapper, KV cache, ledger, and token/feature RNGs remain committed.
Previous assistant/tool text becomes **real extracted history**, not reused
predicted slots. The exact full RustBPE token tuple keys the context provider. The prepared
context retains its producing segment mode and exact per-token segment ownership
through prefill and same-epoch restarts; the ledger uses that same ownership
instead of reconstructing it from character ranges.
The shared one-shot wrapper validates and removes the terminal marker-only real
slot, seeds S0 from the final real source chunk, and forwards `<assistant_start>`
as row 1. Every Kth assistant input publishes the next slot without replay;
K is the integer recorded by post-training, including the K=1 marker boundary.
Structural chat/tool tokens stop before display as assistant content or model
forwarding. Tool execution is not enabled by this TUI.

Streaming text is a provisional preview. Only a fully generated, validated epoch
is published. Extraction, selection, conversion, model/provider, event, and
cancellation failures discard the candidate. Same-epoch rewind works on a new
KV allocation copied from the committed cache; the original allocation stays
untouched until success. There is no rollback by rewriting old cache contents.
Each attempt has a never-reused revision, and events, ledger actions, and fitting
callbacks are checked against their conversation epoch/revision.

Only the current assistant's predicted rows can be edited. Real history is
read-only. Rewind on a real row chooses its containing user/assistant exchange,
retains that exchange's complete user message, discards its assistant response
and all later messages, and builds a new full-history epoch. It never retains a
partial user message or splices an old predicted slot into new history. Rewind
on S0 restarts the current assistant prefill; later predicted rows restore their
complete runtime/RNG boundary and continue with the shared turn transition.
Edits regenerate the selected predicted slot; replayed recorded inputs are each
forwarded once, without fixed mode's two-forward boundary-token algorithm.
Steering selection and regeneration commit together, including failure rollback.

The complete requested assistant budget plus a logical closing marker must fit
the **RustBPE target** window before prefill. The shared producer separately
checks the full **Gemma extraction** count against the raw-SHA-bound statistics
window before extraction/selection. Neither currency is silently truncated or
slid. Exact prompt entries are cached only within their extraction service;
checkpoint/statistics file replacement invalidates even an exact cache hit.
Cached immutable features may survive a failed attempt, but cannot mutate its
committed runtime or alter a retry's RNG draws. Turn signal-fit keys also include
the lineage's missing-percentile policy; legacy fixed-fit keys are unchanged.

### Targeted turn Chat qualification

Run the Chat and shared turn-runtime subsets, not an unbounded training/eval
suite. Tiny-model CUDA coverage is separate from real artifact acceptance:

```bash
uv run pytest tests/test_chat_tui/ --ignore=tests/test_chat_tui/gpu
uv run pytest tests/test_chat_tui/test_turn_session.py -m slow
CHAT_TURN_META=/path/to/posttraining/meta_000020.json \
CHAT_TURN_STATS=/path/to/matching/feature_statistics.safetensors \
CHAT_TURN_PRETRAIN_META=/path/to/pretraining/meta_054600.json \
uv run pytest tests/test_chat_tui/test_turn_artifacts.py -m slow -v \
    --junitxml=out/chat-turn-acceptance.xml
```

Omit `CHAT_TURN_PRETRAIN_META` only when the target metadata is self-contained.
In an isolated checkout without its own tokenizer cache, also set
`NANOCHAT_BASE_DIR` to the existing cache root containing `tokenizer/tokenizer.pkl`.
The real-artifact test records metadata/weight/statistics SHA-256 values (and
pretraining metadata when supplied) and
compares first-turn tokens, routing, LM/FP bytes, and runtime state with the
scalar one-shot engine. It then checks rewind/edit, full-history second-turn
extraction, failure isolation, and clean-retry equality. Missing environment
variables skip this gate; explicitly supplied invalid paths or incompatible
artifacts fail. The separately named tiny-CUDA harness test is not real Gemma
qualification. A raw SFT parent without measured cadence is not a substitute.
Natural sampling and a separate non-structural long-decode probe run as distinct
cases. The latter masks sampled stop logits identically in Chat and the oracle
to require two real-weight refreshes even when the model's natural response is
short. It is boundary coverage, not evidence of natural response length or
quality; the unmodified LM/FP outputs remain the exact equality oracle.

### Checkpoint and config requirements

The target must be an encoder-decoder model with a live feature-prediction head
using BCE and MAX_POOL and an encoder with exactly two feature buckets.
Decoder-only models, missing heads, incompatible head modes, and other bucket
counts are rejected during setup or model load.

**Config:** `configs/serving/chat_tui.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_meta_path` | mandatory | Pretraining or conditioned-SFT checkpoint metadata path |
| `pretrain_meta_path` | `null` | Turn-only exact pretraining metadata for post-training leaves missing conditioning or carrying stale SAE defaults; a raw turn-SFT leaf may omit it (lineage is the sole witness); validated by the shared loader whenever supplied |
| `feature_statistics_path` | `out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/feature_statistics.safetensors` | SHA-bound feature statistics |
| `prepared_dir` | `out/axbench/prepared` | Prepared AxBench metadata and training texts |
| `signal_cache_dir` | `out/chat_tui/signal_cache` | Content-addressed fitted-signal cache |
| `description_cache_path` | `out/neuronpedia_cache.yaml` | Persistent feature-description cache |
| `neuronpedia_model_id` | `""` | Neuronpedia model ID; descriptions require this and `neuronpedia_sae_source` |
| `neuronpedia_sae_source` | `""` | Neuronpedia SAE source; descriptions require this and `neuronpedia_model_id` |
| `self_cond_segment_size` | `256` | Caller-owned fixed-mode RustBPE cadence. Ignored in turn mode, which uses top-level post-training `turn_self_cond_segment_size_rustbpe`. |
| `max_new_tokens_per_turn` | `256` | Per-assistant-turn sampling limit |
| `temperature` | `0.6` | Sampling temperature |
| `top_k` | `50` | Top-k sampling cutoff; `null` disables it |
| `seed` | `42` | Session token RNG seed; feature selection uses the adjacent stream |
| `deterministic_rewind` | `false` | Restore recorded RNG state on rewind instead of taking a fresh derived stream |
| `device` | `cuda` | Target-model and extraction device |
| `dtype` | `bfloat16` | CUDA target-model inference dtype (`bfloat16` or `float32`); BF16 also bounds the temporary model shell during checkpoint loading to avoid an FP32-sized allocation peak |
| `eager_extraction` | `false` | Load the Gemma extraction stack at startup instead of first use |

The config's `hydra.run.dir`, `hydra.output_subdir`, and `hydra.job.chdir`
settings keep the caller's working directory unchanged.

The dtype setting applies only to the target encoder-decoder model. Gemma/SAE
prompt extraction retains its existing BF16 CUDA path. CPU launches preserve
the checkpoint loader's CPU dtype regardless of this setting. After a CUDA
target is converted, the loader releases inactive construction allocations
before the lazy extraction stack claims the same device.

### Error surfaces

The TUI fails loudly instead of substituting fallback artifacts or semantics:
checkpoint-contract, statistics, prepared-concept, Parquet-verification,
sequence-budget, steering-budget, candidate-pool, and segment-edit errors are
surfaced to the user. In fixed mode, a model-operation failure automatically restores the
last completed self-conditioning boundary, including KV, ledger, override, and
RNG state; if the first prefill fails, that provisional prefill is discarded.
Turn mode instead retains the complete previous committed epoch on failure;
no partially generated assistant or partially advanced RNG/cache is installed.
Inputs and prefixes that exceed the rendering, extraction, or decoder budget
are refused with counts rather than silently truncated.

## Special Tokens

All chat interfaces use these conversation tokens:

| Token | Purpose |
|-------|---------|
| `<bos>` | Beginning of sequence |
| `<user_start>` | Start of user turn |
| `<user_end>` | End of user turn |
| `<assistant_start>` | Start of assistant turn |
| `<assistant_end>` | End of assistant turn |
| `<python_start>` | Start of tool code block |
| `<python_end>` | End of tool code block |
| `<output_start>` | Start of tool output |
| `<output_end>` | End of tool output |

## Tool Use

The inference engine (`nanochat/engine.py`) supports automatic tool use:
- When the model generates `<python_start>...<python_end>`, the expression is evaluated
- Results are injected as `<output_start>result<output_end>` tokens
- Execution is sandboxed with a 3s timeout (default `eval_with_timeout`'s `max_time=3`); there is no memory limit. `use_calculator` only invokes `eval_with_timeout`; the heavier `reliability_guard` sandbox in `nanochat/execution.py` is NOT wired into the engine path.

The Engine yields a mask with each streamed token. Sampled model tokens have
mask `1`; forced tool markers and results have mask `0`, although they still
consume decoder/KV positions. The CLI, web stream, and evaluation adapters
preserve and expose the complete Engine-generated transcript. The mask remains
available to conditioning and loss-position logic; it is not a presentation
filter.
