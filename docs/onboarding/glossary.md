# Glossary

Domain vocabulary used throughout the codebase. Read this if a doc or filename is using a term you don't recognize. Cross-references in **bold**.

## SAE & feature concepts

**SAE** — Sparse Autoencoder. A learned dictionary that decomposes a model's hidden activations into a long, sparse vector of interpretable **features**. We use [GemmaScope](https://huggingface.co/google/gemma-scope) SAEs trained on Gemma-2-2b residual streams. Width options: 16k, 32k, 65k, 131k, 262k, 524k, 1m.

**gemma-scope-2** — DeepMind's Gemma 3 SAE release family; the v2 successor to GemmaScope. Coexists with v1 in the same resolver, selected via `release: gemmascope_2`. See [`docs/pipelines/gemma-scope-2.md`](../pipelines/gemma-scope-2.md).

**gemmascope_2** — The `SAERelease.gemmascope_2` enum value, set in structured SAE config as `release: gemmascope_2` to route extraction to the v2 (Gemma 3) code path. v1 stays `release: gemmascope`.

**L0Tier** — The v2 L0 representation: a discrete tier enum (`small`, `medium`, `medium_seed_1`, `big`) replacing v1's integer average-L0. Set via `sae.l0`. `medium_seed_1` exists only for residual SAEs at width 262k.

**Feature** — A single learned direction in SAE space (one of `width` slots; e.g., feature #4231 might fire on "German articles"). Each feature has an integer **feature ID** and a per-token **magnitude**.

**Feature ID** — Integer index into the SAE's feature dictionary, range `[0, width)` (== `[0, n_sae_features)`). This is the **original** id, and it is the *only* id that ever crosses an API, config, shard, checkpoint-metadata or CLI boundary. Contrast **compact feature id** below.

**Compact feature ID** — The row index into the encoder's feature-ID-indexed tables, range `[0, V)` where `V = model.feature_vocab_size = len(vocab.original_ids)`. The tables have one row per **effective feature**: an original id allowed by `feature_mask` and not consolidated into a static-cluster representative. A compact id equals the rank of that original id within the sorted effective set. Original ↔ compact translation happens **only** inside `FeatureEncoder._to_compact`; nothing outside the encoder should ever hold one. With neither a restrictive mask nor active post-mask consolidation, `V == n_sae_features` and the two id spaces coincide. See **Feature vocabulary**.

**Feature vocabulary (`FeatureVocab`)** — The encoder's compacted feature dictionary: `{n_sae_features, original_ids}`, where `original_ids[j]` is the original SAE id of compact row `j` (sorted, unique). In production it is a pure function of `(feature_mask, feature_clusters, n_sae_features)`: `effective_feature_vocab` keeps exactly the mask-allowed ids that are not consolidated away, and literally delegates to the mask-only `FeatureVocab.from_mask` path when clusters are absent. Existing `bl_`/`wl_<16hex>` and `fc_<64hex>` leaves already pin those inputs, so the vocabulary creates no new hash or path component. Persistent `encoder.feature_vocab_ids` / `encoder.feature_vocab_n_sae` buffers make each checkpoint self-describing. See `steering/conditional/effective_feature_vocab.py` and `feature_vocab.py`. Recover a checkpoint's mask with `uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`; clustered recovery also validates the canonical cluster metadata.

**Magnitude / activation** — How strongly a **feature** fires (a non-negative float). A feature is "active" at a token when its magnitude > 0. Stored as `feature_mags` in shards.

**Percentile** — The rank of a feature's magnitude within the per-feature distribution across the training corpus, normalized to `[0, 1]`. Stored as **uint8 overlays** to make the magnitude → percentile lookup O(1) at training time. Computed by `scripts.preprocess.precompute_percentiles`.

**Pooling** — How per-token magnitudes within a **segment** are aggregated into one number per (segment, feature). Options live in `steering.feature_paths.PoolingMode`: `max_pool` (the default), `mean_pool`, etc.

**Discretization / bucket** — Continuous magnitudes (or percentiles) are bucket-ized into a small number of categorical bins (default `num_buckets: 2`) before being fed to the encoder. Each (feature_id, bucket) pair becomes an embedding in the **feature encoder**.

**Delta mode** — Compares a feature's activation across consecutive **segments**. `prev_segment` compares against the previous segment; `cum_max` compares against the running max across all prior segments; `splice_counterfactual` carries no cross-segment state and instead compares each post-splice segment against its **would-have-followed segment** (see below). Drives document-batched positive-delta filtering and negative-feature sampling.

**Delta-aware negatives** — Inactive features are sampled weighted by their delta — features that *would* have grown but were absent are more informative negatives than random ones.

**Gradient attribution score** — A per-(segment, feature) score `percentile(a) · max over the attribution window's rows of ( gate · ReLU(-dL/df) )`. The collector computes the gated gradient *sensitivity* (`max_r [ (a > 0) · ReLU(-dL/df) ]`, where `dL/df` is the loss gradient at the SAE's residual hook pushed through the decoder `grad @ W_dec^T`) — how strongly increasing the feature would lower the windowed loss; the offline producer then multiplies by the feature's activation **percentile** so a barely-active high-gradient feature is demoted below a strongly-firing one. Both factors are ≥ 0, so `score > 0 ⟺ the feature fired AND its increase helps`. Stored as the `grad/<hash>/` overlay; used as an OFFLINE ranking key by the selection precompute (`feature_selection.gradient`). See [`docs/pipelines/gradient-attribution-selection.md`](../pipelines/gradient-attribution-selection.md).

**future / self attribution mode** — Which loss window a segment's gradient attribution uses: `future_*` modes attribute the post-segment suffix loss (the doc-final unit instead splits in half — `future_half_split` — or self-attributes — `future_self_full`); `self` attributes each segment's own predictions. A doc-final segment shorter than `min_final_segment_tokens` merges with the second-to-last segment into one attribution unit.

**Provenance (packed-segment)** — A config-independent sibling of a packed-row dir (`provenance/`) mapping every packed segment back to `(base_shard_idx, doc_idx, seg_ordinal)`, written by the packer (`precompute_packed_rows`) in the same flush as each packed shard (`resume=true` heals it for dirs packed before the integration). Lets base-aligned overlays (e.g. gradient scores) join packed segments without re-running packing per config.

**Static feature cluster (`feature_clusters`)** — A schema-v1 YAML/JSON grouping
of original SAE feature IDs consumed while precomputing selected-feature
shards and persisted canonically in clustered checkpoint metadata. After the
complete `feature_mask` resolves, the minimum allowed member is the fixed
representative and normalized percentiles are max-reduced for ordinary
selection. In clustered `splice_counterfactual` selection, the
mask and representative max are applied independently to the current and
would-have-followed percentile maps before their representative-level delta is
computed; raw member deltas are never max-reduced. The same resolved
consolidation sizes the encoder's effective feature vocabulary:
representatives keep rows and allowed members
consolidated into them do not. Cluster identity is already pinned by the
selection hash, manifest/checkpoint metadata, and `fc_<64hex>` checkpoint leaf;
the derived vocabulary adds no identity input. The completion TUI is the
supported downstream interactive consumer of that checkpoint metadata; it does
not take a cluster path of its own. See
[`encoder-decoder-training.md`](../pipelines/encoder-decoder-training.md#static-feature-clusters-preselection-only).

**Cluster feature contract** — The immutable, fingerprinted agreement used by
cluster-aware AxBench surfaces. It binds the path-backed mask and canonical
cluster definition to the representative/member maps, checkpoint step and
effective vocabulary, statistics, feature-directory/SAE identity,
selected-feature manifest, and concept-source domain. Artifacts carry the
contract fingerprint so equal widths or equal representative sets cannot hide
different membership/provenance. AxBench-Base supports a stage-limited
source/Stage-4 subset under this contract; BPB-Adv threads it through its
dispatcher and sample replay. BPB-Adv consumes source-certified
DiffMean/linear-probe bytes and identity-bound persisted precomputed bags; the
clustered scoring and sample-replay paths do not reopen those baseline artifacts
after source validation.

**Clustered AxBench method set** — A closed, explicit allowlist rather than
“all methods.” AxBench-Base source stages allow the six positive/negative
decoder-only DiffMean, linear-probe, and control methods; Stages 3e/4 add the
encoder-decoder controls and seven ordinary `ours_*` context-editing methods.
BPB-Adv allows those six decoder-only methods plus positive/negative fair and
roofline Feature-DiffMean and Feature-GCG, fourteen total. Both evaluators
require paired path-backed mask/cluster configs; AxBench-Base additionally
requires one supported stage at a time.

**Legacy feature clustering (`feature_clustering`)** — Removed producer,
overlay, runtime, and artifact schema. The similarly named static
`feature_clusters` consumer does not restore that machinery. Current configs or
artifacts carrying legacy clustering metadata are rejected before use.

## Data & file formats

**Segment** — A contiguous span of tokens; the unit of feature extraction. The default `fixed_tokens` mode splits a document into `ceil(doc_len / segment_size)` chunks (default `segment_size: 128`). SFT extraction can instead use `turn_delimited_fixed_tokens`: turn boundaries are mapped to Gemma token starts, then each turn is chunked independently so a segment never crosses turns.

**Shard** — One `.safetensors` file containing CSR-packed features for a chunk of the corpus. Output of `steering.extract_features`.

**FeatureIndex** — The lazy, bounded-LRU-cached reader of extracted feature **shards** (`steering/conditional/feature_index.py`). `lookup(shard_stem, doc_idx)` is O(1) and fail-fast — `KeyError` for an unknown shard or absent document, never `None` (see [encoder-decoder-training.md § FeatureIndex Fail-Fast Contract](../pipelines/encoder-decoder-training.md#featureindex-fail-fast-contract)). Holds up to `max_resident_shards` shards resident at once (default **1**, reproducing the original single-slot cache byte-for-byte); a shard requested beyond capacity evicts the LRU resident. A shard can also be **pinned** (`pin_shard`) to stay resident permanently, bypassing the LRU entirely — used for the validation shard so periodic eval never evicts the current training shard. `retain(stems)` drops residents not in `stems` and pre-warms the first `max_resident_shards` of them, for callers that know their working set ahead of time. The **splice-preselection retain protocol** (issue #740) is `precompute_selected_features`'s adaptation of that same drop-then-pre-seed pattern, applied to *both* indices at once: while preselecting `feature_delta.mode=splice_counterfactual` shards, it drops non-referenced residents and pre-seeds each packed shard's referenced source stems' feature shard, then sidecar, per stem, in first-need order, bounded by the smaller of the two capacities (not a plain `retain()` call — interleaving both indices this way preserves which artifact fails first under unequal capacities) — sized by the producer-only `selection_source_cache_shards` config key (default 5). Shared-store spawn workers are the exception to the private retain size: they keep one mapped stem, and every store stem includes the stable eager row order plus sorted document IDs. A switch remaps that compact locator and constructs only the requested document, avoiding a complete-stem object rebuild without changing lookup order or values. `SpliceProvenanceIndex` (`steering/conditional/splice_provenance.py`) is the matching bounded-LRU cache for parsed `_splice_provenance` sidecars in that same pass (`selection_sidecar_cache_shards`, default 16): same `max_resident_shards`/`retain`/`ensure_resident`/`drop_except` shape, but no threads, pinning, or pickling support. Both caches are a resource/performance bound only — output is byte-identical at every capacity, and neither config key is part of `compute_selection_hash`.

**FeatureIndex catalog** — An immutable, versioned snapshot of one parent
`FeatureIndex` scan: ordered source paths/stems, segment size, exact overlay
identities and coverage, directory/file link and target signatures, and an
optional validated content fingerprint. Parallel splice-selection workers
validate and hydrate this catalog instead of repeating a full source scan. A
changed, replaced, deleted, retargeted, or permission-changed catalog input is
fatal before worker use.

**Shared stem store** — The optional parallel splice-preselection cache under
`NANOCHAT_SHM_DIR` (normally `/dev/shm`). Workers map one immutable physical
tensor copy per source stem. Its visible `capacity` is not a memory bound:
unlinked mapped files remain allocated, so admission accounts for up to
`capacity + 2 * active_workers + 1` stems (capped by the corpus). The two
per-worker generations cover the old resident mapping plus the next mapping
while it is converted to document views. The store uses
exact process identities and leases for cleanup; all-coordinator `SIGKILL` is
recoverable only by the next run's conservative sweeper. tmpfs consumes real
memory and may be swapped.

**Splice resource admission** — The fail-closed, read-only startup check for
`selection_num_workers > 1`. It derives fixed bytes from validated compressed
safetensors headers/runtime dtypes and combines packed, provenance, source,
shared/private, sidecar, measured worker, host/cgroup, and filesystem terms.
An accepted report is a point-in-time supported envelope, not a guarantee
against later unrelated pressure. Rejection occurs before selection output or
shared-store creation and never silently reduces worker count.

**Packed-row spool** — Optional disk-backed packed loading selected by
`selection_packed_spool_dir`. Each worker streams one compressed packed shard
to a temporary plain safetensors file, maps it only for the current shard
context, then closes and unlinks it on every exit. A tmpfs/ramfs spool is
rejected because it would consume the memory the mode is intended to protect.

**CSR sparse format** — Compressed Sparse Row, the on-disk layout for feature shards. Per shard:
```
offsets:           int64[(n_segments + 1,)]   CSR row pointers
feature_ids:       int32[(n_total_features,)] feature indices, ascending-by-id order per segment
feature_mags:      float16 on disk; float32 after load; magnitudes co-sorted with feature_ids
doc_ids:           int32[(n_segments,)]
segment_indices:   int32[(n_segments,)]
segment_positions: int32[(n_segments,)]
doc_token_counts:  int32[(n_docs,)]
segment_char_starts/ends: int32[(n_segments,)]
```
Ordering: the in-memory output of `extract_sparse_gpu` is **descending-magnitude per segment**; this is the order callers see when they construct a shard before saving. `save_features` re-sorts to **ascending-by-id** (delta-encodable for compression). `load_features` round-trips that on-disk order, so any consumer reading shards from disk receives ascending-by-id arrays.

Invariants: `offsets[0]==0`, `offsets[-1]==len(feature_ids)`, offsets monotonic, post-load `feature_ids` ascending within each segment. For **base** extracted shards there is one more, strict invariant: every segment is non-empty (`offsets[i+1] > offsets[i]` for all `i`) whenever `n_segments > 0` — a base-extracted segment that fired zero SAE features is treated as corruption. `save_features`, `validate_features`, and `FeatureIndex` load all reject an empty base segment (issue #522; a whole-shard `n_segments==0` is exempt). `steering.validate_features` checks all of these. This strict non-empty rule applies **only** to base extracted shards: downstream SELECTED/derived data (selection shards, packed rows, overlays, clustering, delta, masking, percentile-filtered stats) MAY legitimately be empty and is intentionally exempt.

**Overlay** — Auxiliary data computed *on top of* an immutable feature shard, stored alongside it. Examples: `pct/<hash>/` (percentiles), `tok/<hash>/seg_boundaries/` (token boundaries), packed rows under `pct/<hash>/tok/<hash>/c{T}/s{N}/`, `grad/<hash>/` (gradient-attribution scores). The directory hash makes overlays content-addressed — change the input config, get a new overlay dir, never mutate an old one. See [`docs/reference/architecture.md`](../reference/architecture.md) for the full overlay layout.

**Packed row** — A pre-built training row (token IDs + per-segment feature info) ready for the encoder-decoder dataloader. Produced by `scripts.precompute_packed_rows`. Lets training skip per-row Python work.

**Feature-shard splice** — An optional **post-extraction** transform (`scripts.preprocess.splice_feature_shards`) that recombines a base shard's segments into a **derived feature directory**, carrying each segment's stored features verbatim (no model re-run). The algorithm selects a fraction of inter-segment **splice points** as **cuts**, pairs them via a **derangement**, and stitches each cut's prefix to its paired cut's suffix; the result is gated at every stage by round-trip stability. **Root-started** derived docs (opening with a source doc's segment 0) are extraction-equivalent; **new-started** docs (from loops or overflow splits) prepend a synthetic BOS and are not extraction-equivalent. Within-shard and deterministic; the derived dir lives at `<base>/splice/cut_frac{cut_fraction}/seed{seed}/` with `_parquets/` (text) and `_splice_provenance/` (`[base_shard_idx, doc_idx, seg_ordinal]` per segment, plus schema-v2 would-have-followed references — below). See [`docs/pipelines/feature-shard-splice.md`](../pipelines/feature-shard-splice.md).

**Would-have-followed segment** — for a splice junction `…A_k | B_j…`, the base segment `A_{k+1}` that would have continued A's document had the splice not occurred. Recorded per junction in the schema-v2 `_splice_provenance` sidecar.

**Splice-counterfactual conditioning** — `feature_delta.mode=splice_counterfactual`: post-splice segments are conditioned on their percentile delta vs the would-have-followed segment; all other segments use absolute percentiles.

**Splice point** — An inter-segment boundary within a source document that can serve as a cut site. For a document with `n` segments, there are `n-1` splice points (between segments 0-1, 1-2, …, (n-2)-(n-1)). The splicer enumerates all splice points across all source docs in a shard before selection.

**Cut** — A selected splice point. The splicer picks `round(cut_fraction * n_splice_points)` cuts via a single seeded shuffle. A cut at `(doc, boundary)` severs the source-doc adjacency between segment `boundary` and segment `boundary+1`.

**Run** — The maximal contiguous interval of source-doc segments that contains no cuts under the current cut set. Used by the windowed pre-filter to assemble a BOS-led context around each stitch seam.

**Stitch** — The edge created when the derangement pairs cut `i` with cut `σ(i)`: the prefix of source doc `d` through segment boundary `b` is connected to the suffix of source doc `dp` starting at segment `bp+1`. The resulting derived document contains segments from both source docs.

**Derangement** — A permutation `σ` over the `k` selected cuts with no fixed points (`σ(i) != i` for all `i`). Drawn by rejection sampling over uniform permutations. Guarantees that every cut's tail is attached to a DIFFERENT cut's partner, so no stitch is a no-op.

**Opposite-percentile pairing** — Optional `pairing_mode=opposite_percentile` in feature-shard splicing. Starting from the usual derangement, bounded groups of cut slots exchange partners when that increases the two seams' combined L1 percentile distance without creating a fixed point. Percentiles compare the incoming segment against the original continuation it replaces, using the same would-have-followed reference as splice-counterfactual conditioning; inactive features are represented as zero. The search uses base statistics and original SAE IDs; later tokenizer safety repairs take precedence over the initial score improvement. Its output seed leaf adds `_opposite_percentile_pool{B}`. See the [splice algorithm](../pipelines/feature-shard-splice.md#optional-opposite-percentile-pairing).

**Root-started document** — A derived document whose first segment is segment 0 of a source doc. The real BOS token is already present; every interior segment carries exactly `segment_size` tokens. Root-started docs are extraction-equivalent: gradient attribution and source cross-ref validation work on them. All root chains in the successor graph produce root-started docs.

**New-started document** — A derived document formed from a broken cycle (**loop**) or an overflow **split** remainder. A synthetic BOS token is prepended to the concatenated segment tokens; segment 0 carries `segment_size + 1` tokens (BOS + the source segment). New-started docs are NOT extraction-equivalent: their seg-0 size breaks fixed-size re-segmentation, so gradient attribution and `validate_features` source cross-ref raise loudly on derived dirs that contain them.

**Round-trip-stable seam** — A stitch seam whose surrounding token sequence re-tokenizes to itself under the SAE tokenizer: `tokenizer(decode(ids)) == ids`. The feature-shard splice applies a windowed pre-filter on run-local contexts as a fast early gate, then uses the batched full-document oracle as the final authority. Any seam or piece that fails triggers a redraw from the pool, keeping every output document re-tokenizable to exactly its stored segments.

**Phase** — Which dataset / training stage the data is from: `base` (FineWeb-Edu pretraining), `standard_sft` / `safety_sft` / `capabilities_sft` / `standard_steered_sft` / `standard_steered_balanced_sft`. Used in feature directory paths and config overrides.

**Steered conversation** — One single-turn `user`/`assistant` exchange sampled from a steered model and stored as one line of a steered JSONL (`tasks/steered.py`). The `standard_steered` and `standard_steered_balanced` mixtures splice these into the standard mixture at exactly even dataset positions — steered row `k` of `M` lands at `((2k + 1) * L) // (2M)` — so exactly one steered row falls in every half-open interval `[kL/M, (k+1)L/M)` rather than clustering anywhere in the stream. DDP striding by `world_size` can still make a single rank's share uneven; the averaged gradient carries the global fraction.

**Val family** — The mixture whose validation artifacts another mixture validates on (`tasks.mixtures.sft_val_family`). Every mixture is its own family except the steered ones, which both map to `standard`: they mint no `*_sft_val` phase, read `sft_data/standard_val` and `standard_sft_val` features, and record `mixture=standard` in the val fingerprint.

**Feature directory convention** — `{root}/s{seg_size}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}`. Defined in `steering/feature_paths.py`. Example: `out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base`. Optional trailing leaves (ordered): `/context_independent` (opt-in), `/no_bos` (always present on new runs — absent only on legacy pre-freeze artifacts).

**ignore_bos_features / `/no_bos`** — BOS-feature exclusion (every `<bos>`-id token dropped from per-segment pooling), **frozen always-on** in the extraction producer (no longer a config option). New runs add a `/no_bos` directory leaf (after `/context_independent`) and record `ignore_bos_features=true` in shard metadata and the stats file. Legacy shards and stats files without the key load as `false`.

## Model architecture

**Decoder / decoder-only** — The standard causal-LM transformer (the GPT). Defined in `nanochat/gpt.py`.

**Encoder** — The bidirectional transformer that consumes per-segment SAE features. No causal mask, no positional embeddings (permutation-invariant). Defined in `steering/conditional/encoder_decoder/encoder.py`.

**Cross-attention** — Inserted at every `frequency`-th decoder layer. Decoder queries attend to encoder keys/values per **segment**. See `steering/conditional/encoder_decoder/cross_attention.py`.

**Encoder-decoder model** — The composition: token embedding → decoder blocks (with periodic cross-attention) → LM head; encoder runs in parallel on SAE features. See `steering/conditional/encoder_decoder/model.py`.

**seg_map** — A `(B, T)` tensor mapping each decoder token to its segment index. Must be monotonically non-decreasing along T (segments are contiguous). Used by cross-attention to route queries to the right encoder slot.

**Character-level segment assignment** — Decoder tokens are assigned to feature
segments by character-offset overlap. The former fixed-stride path was removed;
character-level assignment is required for cross-tokenizer robustness.

**Conditioned eval** — Encoder-decoder evals require real SAE features for the prompt/context, provided by a context cache keyed by the exact nanochat token IDs.

**Steering** — Choosing which features the encoder sees at inference time to shape generation. Done via `scripts.steer_chat_cli` (CLI) or `scripts.completion_tui` (TUI).

**RoPE / RMSNorm / GQA / SwiGLU / sliding window** — Standard modern transformer components. See [`docs/reference/architecture.md`](../reference/architecture.md) for what nanochat uses.

**Value embeddings** — ResFormer-style alternating layers with a learned value bias, mixed into `v` through an input-dependent gate. There are **two independent sets**: the decoder's, which are **token**-indexed (`nanochat/gpt.py`, and `EncoderDecoderGPT.value_embeds`), and the **encoder's** (`steering/conditional/encoder_decoder/encoder.py`, `FeatureEncoder.value_embeds`), which are **feature-ID**-indexed and therefore compacted by the `feature_mask` (#695). Both use `has_ve(i, n_layer)` — a table on every other layer, last layer always included. The encoder's are collectively the largest block of feature-ID-indexed parameters in the encoder.

**resid_lambdas / x0_lambdas** — Per-layer learnable scalars that interpolate between the residual stream and the original input embedding. Both are independent (not a complementary pair).

**Logit softcap** — Output logits clamped to ±15 via `tanh(x/15)*15`. Stabilizes late training.

## Optimizer & training

**Muon** — A second-order-ish optimizer (Polar Express orthogonalization, NorMuon variance reduction, cautious weight decay) used for matrix weights. Applied via the `MuonAdamW` mixed optimizer in `nanochat/optim.py`.

**MuonAdamW** — The mixed optimizer: AdamW for embeddings/scalars, Muon for weight matrices. Different learning rates per group (`embedding_lr`, `unembedding_lr`, `matrix_lr`, `scalar_lr`).

**DistMuonAdamW** — Distributed (ZeRO-2-style) version with reduce-scatter / all-gather for large parameters.

**FP8** — 8-bit float training via torchao. Enabled with `fp8: true` and `fp8_recipe: tensorwise`. Only applied to Linear layers with dims divisible by 16 and min dim ≥ 128. Auto-disabled for evaluation via `disable_fp8()`.

**BPB** — Bits Per Byte. Tokenizer-agnostic loss metric: NLL normalized by UTF-8 byte length. Lets you compare models with different vocab sizes fairly.

**CORE** — DCLM's standard multiple-choice / schema / language modeling eval suite. Reported by `nanochat.core_eval`.

**Conditioned BPB** — For encoder-decoder models, BPB is reported with real encoder features via `evaluate_encdec_bpb_conditioned()` in `steering/conditional/encoder_decoder/loss_eval.py`.

**Feature prediction (aux loss)** — Optional auxiliary loss that has the decoder predict next-segment SAE features from its own hidden states. Encourages the decoder to internalize feature info during training. See `steering/conditional/encoder_decoder/feature_prediction.py`. Modes: **`MAX_POOL`** (the `FeaturePredictionConfig` default), `FULL_SEGMENT`, `LAST_TOKEN`, `BEST_TOKEN`. Loss fns: BCE, MSE. The head predicts the original SAE width on a full model and the effective representative vocabulary on a compact model. Compact pretraining targets require certified precomputed packed rows; conditioned SFT validates its per-conversation source segments inline. Direct FP-logit return and inference-time self-conditioned generation support both widths: compact head rows are translated to original SAE ids before selection. Pure-model two-sided policies rank pooled float32 logits directly to avoid sigmoid saturation ties; paths that consume numeric percentiles, interpolation values, or buckets retain normalized ranks. Raw cross-feature magnitude ordering is not replayable; clustered SFT thresholding and sampling both use full-distribution ranks. The packed-pretraining self-conditioning post-training source remains uncompacted-only; source-aware SFT supports exact compact mask/cluster vocabularies and compact-row STE.

**target_param_data_ratio** — Tokens per parameter for compute-optimal scaling. Compute-optimal is ~10.5; the speedrun uses 8.25 to slightly undertrain (faster, hits GPT-2 quality).

**warmdown_ratio** — Fraction of training spent in linear LR warmdown to zero. Default `0.5`.

## Distributed & infra

**DDP** — PyTorch Distributed Data Parallel. Launched via `torchrun --standalone --nproc_per_node=N`.

**Compose API vs `@hydra.main`** — Two patterns the codebase uses. `@hydra.main` is Hydra's standard CLI decorator; the compose API is a manual alternative used by torchrun-launched scripts because `@hydra.main` doesn't play well with torchrun. See [`docs/cli-reference.md#hydra-config-selection`](../cli-reference.md#hydra-config-selection).

**FA4 / FA3 / SDPA** — Attention backends, in preference order: Flash-Attention 4 (Hopper+/Blackwell, via `flex_attention`'s FLASH backend), Flash-Attention 3 (Hopper), PyTorch SDPA fallback. Auto-detected by `nanochat/flash_attention.py`.

**`NANOCHAT_BASE_DIR`** — Where downloaded data, tokenizer, checkpoints, and the run report live. Defaults to `~/.cache/nanochat`.

## Evals

**AxBench-Base** — A multi-stage steering benchmark (Stages 0-6) with a final LLM-judge leaderboard. Lives in `steering/eval/axbench_base/`. See [`docs/evaluation/axbench-base.md`](../evaluation/axbench-base.md).

**AxBench-BPB** — Offline, BPB-based variant of AxBench-Base. Replaces Stages 4-6 (generation + judge) with teacher-forced BPB scoring on Stage 1b/1c synth samples. Lives in `steering/eval/axbench_bpb/`. See [`docs/evaluation/axbench-bpb.md`](../evaluation/axbench-bpb.md).

**Steering eval** — Pairwise judge eval that asks whether adding a target feature delta changes the output toward the requested feature under full-context conditioning — `scripts.steering_eval`.

**Completion steering eval** — Ablation/insertion eval measuring causal effect of steering on completion text. `scripts.completion_steering_eval`.

**Shuffled conditioning eval** — Control eval that swaps real conditioning for random feature dicts. `scripts.shuffled_conditioning_eval`.

**Harmfulness eval** — OpenRouter LLM-judge safety eval. Harmful-prompt safety
is the exact-score-1 rate; benign compliance is independently the non-refusal
rate over successfully judged benign pairs. Helpfulness remains a required 1--5
diagnostic and does not affect compliance. The result retains per-benign-pair
reasoning and versioned judge provenance. `scripts.eval.harmfulness_eval`.

**Sanity-check eval** — Minimal/fast variants that exercise the eval pipeline itself (used when modifying judges or inference engines). `steering/eval/sanity_check/`.

**Identity conversations** — A small dataset of synthetic chat data used to give the model a name + personality during early SFT. Downloaded from `karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl`.

## Misc

**`nanochat/`** — Vendored copy of [karpathy/nanochat](https://github.com/karpathy/nanochat). **Read-only.** See [`nanochat/VENDORED.md`](../../nanochat/VENDORED.md).

**`steering/`** — All project-specific code. New code goes here.

**Hydra** — The configuration framework. Uses YAML files in `configs/` with composition via `defaults:`. CLI overrides are `key=value` (or `key.subkey=value`).

**`uv` / `uvx`** — The package manager (replaces pip / venv) and tool runner. `uv sync` installs the project; `uvx ty check` runs the type checker without installing it.

**`ty`** — The type checker (from Astral, the same group as ruff/uv). Run via `uvx ty check`. Configured in `pyproject.toml` under `[tool.ty]`.

**loguru** — Logging library. Use `from loguru import logger` and `logger.info(...)`. Configured DDP-aware in `nanochat/common.py` (rank 0 logs INFO, other ranks log WARN+).

**Hopper / Blackwell** — NVIDIA GPU architectures. Hopper = H100/H200 (sm90). Blackwell = B100/B200 (sm100). FA4 supports both; FA3 supports Hopper only.
