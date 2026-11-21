# Feature-Shard Splice Pipeline

An optional, **CPU-only, weight-free** transform that recombines the
already-computed segments of base SAE feature shards into a **separate derived
feature directory**, carrying each segment's stored features verbatim. It runs
**after** `extract_features` + `compute_statistics`, leaves the base shards
immutable, and produces a derived feature directory that every downstream
producer accepts.

> **See also:** [feature-extraction.md](feature-extraction.md) for the base
> pipeline that produces the inputs.

This replaces the removed pre-extraction splicer: instead of joining raw parquet
text *before* SAE extraction (which re-ran the model), this transform reshuffles
the **post-extraction** feature-shard segments, so no model is run and each
segment keeps its original-context features.

> **Obsolescence notice:** Derived directories produced by the pre-rework
> algorithm (output path `splice/p{rate}/seed{seed}/`) are **obsolete**. Delete
> them and re-run the splice step under the current algorithm to obtain a valid
> derived directory.

## Overview

```
Base feature shard N + parquet N
        │  re-tokenize parquet (CPU) + verify it reproduces the shard's segmentation
        ▼
   per-segment (tokens + stored features)
        │  enumerate splice points; select cut_fraction·N via seeded shuffle;
        │  uniform derangement (no fixed points) over selected cuts;
        │  optional bounded opposite-percentile partner swaps;
        │  stitch prefix-of-cut-i to suffix-of-cut-σ(i);
        │  windowed pre-filter + batched full-document oracle gate
        ▼
   derived doc chains  ──►  derived feature shard N (features verbatim)
                       ──►  derived parquet N (decoded spliced text)
                       ──►  provenance N  [base_shard_idx, doc_idx, seg_ordinal]
```

**Script:** `scripts/preprocess/splice_feature_shards.py`
**Config:** `configs/extraction/splice_feature_shards.yaml`
**Core library:** `steering/feature_shard_splice.py` (+ `steering/feature_shard_pairing.py` and `steering/_roundtrip_seam.py`)

## Quick start

```bash
# Splice all base shards with cut_fraction=0.15 (the default)
uv run -m scripts.preprocess.splice_feature_shards \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    cut_fraction=0.15

# Then run the standard downstream on the derived directory:
DERIVED=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/splice/cut_frac0.15/seed42
uv run -m steering.compute_statistics feature_dir=$DERIVED
uv run -m scripts.preprocess.precompute_percentiles feature_dir=$DERIVED
uv run -m scripts.preprocess.precompute_token_boundaries feature_dir=$DERIVED data_dir=$DERIVED/_parquets
uv run -m scripts.preprocess.precompute_packed_rows feature_dir=$DERIVED data_dir=$DERIVED/_parquets
```

To prefer incoming segments opposite to the original continuations they
replace, use the command and derived directory below instead. The same
downstream commands above then operate on this `DERIVED` value. The base
directory must contain `feature_statistics.safetensors`.

```bash
uv run -m scripts.preprocess.splice_feature_shards \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    pairing_mode=opposite_percentile pairing_candidate_pool_size=64
DERIVED=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/splice/cut_frac0.15/seed42_opposite_percentile_pool64
```

When `scripts.train.conditional_encdec_train` is pointed at the canonical
derived directory (`<base>/splice/cut_frac*/seed*`, including the
`seed42_opposite_percentile_pool64` leaf), it treats the base directory's
last shard as the baseline validation set. Training uses the derived packed
rows; `val/*` reports the original/base validation shard, and `val_spliced/*`
reports the derived validation shard. If the splicer wrote to a custom
`output_dir`, set `original_val_feature_dir=<base feature_dir>` on the training
run so the base validation packed rows can be found. Because packed-row
training does not support the live `feature_delta` path, set
`feature_delta=null` for these splice-training runs — except for
`feature_delta.mode=splice_counterfactual` (issue #596), which runs both
validation passes through live loaders instead of packed rows and requires
`feature_delta` set to that mode rather than `null`; see
[encoder-decoder-training.md](encoder-decoder-training.md).

## Semantics — what changes and what does not

Splicing is a **pure reorganization** of segments:

- **Per-segment features are copied verbatim** (`feature_ids`, `feature_mags`) —
  byte-identical (`t.equal`) to the base source segment.
- The **document grouping** changes: derived documents are formed by stitching
  the prefix of one source document (up to a cut boundary) with the suffix of a
  deranged partner's document (starting from its paired cut boundary).
- Because the features are NOT recomputed, each segment keeps the features it had
  under its **original** causal context. For `context_independent_features=true`
  this is identical to re-extraction; for context-dependent features the segment's
  features describe its original neighbours, not the spliced ones. This is the
  deliberate post-extraction trade-off (no GPU, no re-extraction).

## Algorithm

Each base shard is processed independently (within-shard), seeded deterministically
by `recombination_seed(seed, shard_index, cut_fraction)`.

### Phase A — Re-tokenize + align

The source parquet is re-tokenized (head-truncated exactly as extraction does)
and re-segmented; the result must reproduce the base shard's recorded
`doc_token_counts`, `segment_positions`, and doc-ordered `segment_indices`, or
the shard fails fast (`AlignmentError`). This binds the re-tokenized tokens to
the base shard's features.

### Phase B — Attach features

Each base segment's CSR slice is attached to its re-tokenized token span.

### Phase C — Source-doc repairs

Before recombination, each source document is checked for round-trip stability:
`tokenizer(decode(ids)) == ids`. Two repair paths exist:

- **Gemma-3 U+2581 space-marker normalization** (existing): normalizes literal
  `▁` characters in source text that confuse the Gemma-3 sentencepiece tokenizer.
- **Bounded byte-fallback trim** (new): for documents truncated mid-multibyte-char,
  trailing `<0xNN>` byte-fallback tokens are popped one at a time until the
  document round-trips. If the first non-byte-fallback token would need to be
  removed, `SourceDocNotStableError` is raised. Non-truncated unstable documents
  always raise immediately.

Trimmed-token and trimmed-doc counts are recorded in the shard provenance.

### Phase D — Batch select → derange → stitch

1. **Enumerate splice points.** All `(doc, boundary)` pairs with
   `boundary in 0..n_segs-2` are listed in lexicographic order (these are the
   legal inter-segment cut sites across all source docs).
2. **Select cuts.** `target_k = round(cut_fraction * n_splice_points)`.
   If `target_k < 2`, the shard is returned as identity (a derangement requires
   at least 2 cuts). One seeded shuffle picks the first `target_k` entries as the
   **cut set**; the rest form the **redraw pool** (consumed in order on any redraw).
3. **Draw a derangement.** A uniform derangement `σ` over the `k` cuts is drawn by
   rejection sampling (random permutations filtered until no fixed point). Each
   cut `i` is paired with cut `σ(i)`. This is the final pairing in the default
   `pairing_mode=random`; `opposite_percentile` improves it as described below.
4. **Build the successor map.** Source-doc adjacencies are installed; cut edges
   `(d, b) → (d, b+1)` are replaced by stitch edges `(d, b) → (dp, bp+1)` where
   `cuts[σ(i)] = (dp, bp)`. The successor function is injective: every ref has at
   most one successor and at most one predecessor.
5. **Read chains and loops.** Root walks from `(d, 0)` yield the root chains.
   All unvisited refs form disjoint cycles (loops). Loops arise when a doc
   donates segments to itself via a chain of stitches that wraps around — they
   contain no root `(d, 0)` (proof: stitches only target `ordinal >= 1`, so roots
   are always chain-started).

### Optional opposite-percentile pairing

`pairing_mode=opposite_percentile` favors an incoming segment whose activations
differ from the original continuation it replaces. For a proposed seam
`A_k | B_j`, the would-have-followed segment is `A_{k+1}` and the score is

```text
score(A_k | B_j) = sum over original SAE features f of |pct(A_{k+1}, f) - pct(B_j, f)|
```

An absent feature has percentile zero. Stored activations use the existing
`magnitude_to_percentile` interpolation against the base directory's
`feature_statistics.safetensors`, in percentile-rank units `[0.5, 99.5]`.
Dividing every value by 100 would only rescale the objective. Values remain
unquantized, so there is no uint8-overlay rounding. The reference is the same would-have-followed segment
used by downstream splice-counterfactual conditioning. Feature masks,
cluster representatives, and downstream selection settings do not enter this
corpus transformation. The statistics file must cover each queried active
feature: missing, nonfinite, or invalid input raises rather than receiving a
fabricated percentile. The usual training-only statistics can be used; no
statistics are fitted on validation data by the splicer. A feature seen only
in the validation shard can therefore cause the same missing-statistics error.

The cut selection and initial random derangement remain unchanged. The engine
then processes consecutive groups of at most `pairing_candidate_pool_size`
selected cut slots; their order is already seeded and shuffled. Each group's
candidate suffixes are the suffixes currently assigned to those slots. In one
pass over its slots, the engine accepts the partner swap with the greatest
strictly positive gain in the **combined score of both affected seams**, as
long as neither seam becomes its original uncut adjacency. Zero gains and gains
within the float64 accumulation-error bound keep the current pairing; equal
positive gains within that bound choose the first eligible candidate. Every
suffix is still assigned exactly once. A final group of one slot, or a group
without a legal improving swap, keeps its initial assignments. Two or three
cuts can have no legal two-partner swap, leaving their initial derangement.

Scoring uses sparse overlap: `L1(a,b) = sum(a) + sum(b) - 2*sum(min(a,b))`.
Because every suffix is used once, the two norm terms cancel when comparing
partner swaps; minimizing shared percentile overlap is equivalent to
maximizing L1. The search holds only a bounded group score matrix, sparse
segment features, and feature-width scratch space, alongside the ordinary
percentile table. It never builds a corpus-wide pair matrix or a feature
correlation matrix. For `K` selected cuts, bound `B`, and typical `S` stored
features per segment, scoring is `O(K * B * S)` and group score storage is
`O(B²)`. Increasing `B` trades compute and memory for a wider local search;
64 is the default starting bound, not a global-optimum guarantee.

The positive-sum reduction uses a conservative roundoff bound rather than
rounding the percentiles: an overlap reduction must exceed
`8 * eps64 * (maximum candidate nnz + 1) * (old overlap + new overlap)` before
a swap is eligible. Candidates whose improvements differ by no more than the
sum of their error bounds are treated as tied. This avoids changing a seeded
pairing because differently grouped float64 sums give a mathematically
zero-gain swap a tiny positive value.

Only the initial pairing is optimized. The existing token-stability redraws,
loop breaks, and overflow splits run afterwards and take precedence: they can
reduce or remove the improvement in the final emitted junctions. Redraws keep
their original finite-pool order and termination rules. With fewer than two
selected cuts, the identity result remains unchanged.

The default output leaf is
`<base>/splice/cut_frac{cut_fraction}/seed{seed}_opposite_percentile_pool{B}`.
This separates modes and search bounds while preserving automatic base
validation discovery. The sidecar pins the actual percentile-table checksum
and algorithm version; changing either in an existing directory raises during
resume. Use a fresh `output_dir` to retain multiple statistics versions.
Random mode loads no statistics and preserves the established output path,
RNG behavior, decoded shard contents, and parquet/provenance bytes. Its
search-bound setting is inert. Local feature serialization retains its legacy
behavior; remote publication uses canonical bytes as described below.
Correlation-based pairing is not implemented.

The statistics' available extraction metadata is checked against the first
base shard at setup and against every pending shard before processing it:
segment size, context window, layer, pooling, segmentation mode, L0 tier, and
BOS-feature policy. A source shard's `n_sae_features` header must also match
the statistics when present, including in noncanonical explicit-model paths;
legacy shards may omit that header. Canonical feature paths additionally bind
the SAE width.
Statistics are loaded once per run and reused; these per-shard checks read
only the headers. Completed resumes retain the existing provenance-validation
contract and do not rescan every source body.

### Phase E — Stability gates

All three gates use the same criterion: `tokenizer(decode(ids)) == ids`.

**Windowed pre-filter (run-local context):** For each of the `k` stitch seams,
a run-local BOS-led `(prefix, suffix)` pair is assembled. The prefix runs from
the start of the cut's maximal uncut source interval through the cut boundary;
the suffix runs from the stitch target through the end of its interval. Both are
BOS-led (mid-doc runs get a synthetic BOS). If `WindowedSeamChecker.stitch_is_stable`
fails, the cut at that slot is redrawn from the pool (pool exhaustion →
`PoolExhaustedError`). The pre-filter is a local approximation; the full-doc
oracle below is the final authority.

**Loop resolution (oracle-gated):** Each cycle is broken by rotating it to start
at the lowest-ref stitch candidate that passes `is_roundtrip_stable` on
`(BOS,) + concat_ids`. The broken cycle becomes a fresh **new-started document**
(`new_started=True`, synthetic BOS prepended). If no rotation passes, the
lowest-numbered cut slot in the cycle is redrawn and the outer fixpoint restarts.

**Overflow split (oracle-gated):** Any chain or loop-doc piece exceeding
`limit = min(context_window, sae_max_context)` tokens is split at the maximal
segment boundary `b` such that the kept piece fits within `limit` and both the
kept and `(BOS,) + remainder` pass `is_roundtrip_stable`. If no boundary works,
`UnsplittableChainError` is raised with full diagnostic context (doc id,
`new_started` flag, piece length in segments and tokens, the limit, and the range
of boundaries tried). **An overflowing chain with no stable split boundary is fatal
by design** — recovery-by-redraw was considered and rejected; strictness over yield.
The remainder becomes a new-started document.

**Batched full-document oracle (final authority):** All pieces that are not
source-pure (identical to an unstitched source doc) are checked via a batched
`is_roundtrip_stable` call over their full token ids. If any piece fails, the
earliest stitched junction in that piece is identified and its cut slot is redrawn;
the outer fixpoint restarts. This loop terminates because the pool is finite.

### Phase F — Emit

**Root-started documents** (`new_started=False`): formed from a root chain
`(d, 0) → ...`. The real BOS token is already in segment 0 of the source doc.
Every interior segment carries exactly `segment_size` tokens; the last may be
shorter. Root-started documents are **extraction-equivalent** — their text
round-trips and their segmentation matches what re-extraction would produce.

**New-started documents** (`new_started=True`): formed from a broken loop or an
overflow remainder. A synthetic BOS token is prepended to the concatenated
segment tokens. Segment 0 of a multi-segment new-started doc carries
`segment_size + 1` tokens (synthetic BOS + the source segment's tokens). For a
solo overflow remainder whose only segment is a truncated source last-segment,
seg-0 carries `1 + that segment's length ≤ segment_size + 1`. Features are
copied verbatim; the BOS token is outside any segment window and carries no
features.

**Provenance:** Each derived segment maps back to `[base_shard_idx, doc_idx,
seg_ordinal]` — the same triple packed-row provenance uses — so derived segments
can join base-aligned overlays.

## Root-started vs new-started documents

| Property | Root-started | New-started |
|---|---|---|
| `new_started` flag | `False` | `True` |
| BOS token | real (from source seg 0) | synthetic (prepended) |
| Seg-0 token count | `segment_size` (or less for a 1-seg doc) | `segment_size + 1` (multi-seg); `1 + last-seg-len ≤ segment_size + 1` (solo remainder) |
| Extraction-equivalent | yes | no |
| `precompute_gradient_attribution` | supported | **raises on re-segmentation mismatch** |
| `validate_features` source cross-ref | supported | **raises on re-segmentation mismatch** |

New-started docs arise from loops and overflow splits. At low `cut_fraction`
values they are rare but possible. The `n_loops` and `n_splits` provenance
counters show how many occurred per shard.

## Artifact layout

```
.../m_gemma-2-2b/base/                                  ← immutable base (untouched)
└── splice/cut_frac0.15/seed42/                          ← derived feature_dir (override via output_dir=)
    ├── shard_00000_features.safetensors.zst …          ← derived shards (features verbatim)
    ├── _parquets/shard_00000.parquet …                 ← derived text; pass as data_dir= downstream
    ├── _splice_provenance/shard_00000.json …           ← provenance sidecar
    ├── feature_statistics.safetensors                  ← recomputed downstream
    └── pct/<hash>/ · tok/<hash>/seg_boundaries/ · pct/<hash>/tok/<hash>/cT/sN/ · grad/<hash>/
```

The `s{N}` path component is inherited from the base path (so `verify_segment_size`
passes). Overlays self-isolate via each manifest's `base_shard_content_hash`,
which is pinned to the **derived** shard bytes — so base overlays can never be
silently reused on the derived directory.

## Provenance sidecar schema

Each `_splice_provenance/shard_NNNNN.json` records:

| Field | Description |
|---|---|
| `schema_version` | `2` |
| `shard_name` | e.g., `shard_00000` |
| `shard_index` | integer shard index |
| `shard_seed` | `recombination_seed(seed, shard_index, cut_fraction)` |
| `params.model` | model id the base shard was extracted with |
| `params.segment_size` | tokens per segment (must match the base shard) |
| `params.context_window` | extraction context window (must match the base shard) |
| `params.sae_max_context` | the model's SAE max context (`null` when unbounded) |
| `params.seed` | global RNG seed (folded into `shard_seed`) |
| `params.base_feature_dir` | directory of the base shard this was derived from |
| `params.data_dir` | directory of the source parquet this was derived from |
| `params.cut_fraction` | fraction of splice points selected as cuts |
| `params.pairing_mode` | `opposite_percentile` for activation-based pairing; absent on legacy/default random outputs |
| `params.pairing_candidate_pool_size` | active bounded search group size; absent in random mode |
| `params.pairing_percentile_checksum` | checksum of the actual percentile lookup table used; absent in random mode |
| `params.pairing_algorithm` | `bounded_swap_v1`; absent in random mode |
| `params.n_cuts_final` | number of cuts after all redraws |
| `params.n_redrawn` | total redraw count across all gate phases |
| `params.n_loops` | number of cycles broken into new-started docs |
| `params.n_splits` | number of overflow splits |
| `params.n_trimmed_docs` | source docs with trailing byte-fallback tokens trimmed |
| `params.n_trimmed_tokens` | total tokens trimmed across those docs |
| `params.n_marker_normalized_docs` | docs where gemma-3 U+2581 normalization applied |
| `params.n_splice_positions` | total splice records across all docs (schema v2) |
| `docs[i].provenance` | `[[base_shard_idx, doc_idx, seg_ordinal], …]` per segment |
| `docs[i].segment_token_starts` | content-token offset of each segment's first token within the derived doc; a new-started doc counts a synthetic BOS as token 0 so the first real token's offset is 1 — unlike the shard's `segment_positions` which is BOS-inclusive[^1] |
| `docs[i].n_switches` | number of source-doc switches in derived doc `i` |
| `docs[i].splices` | schema v2: one record per splice position `p` — `{pos, whf_base: [shard, doc, ord], whf_derived: [derived_doc, position]}`; `[]` when the doc has no junctions |

[^1]: Convention divergence from `segment_positions`: the shard's `segment_positions` counts the BOS as an emitted token (position 0), so root-doc `segment_positions[j]` == `segment_token_starts[j]` for all `j`. For a new-started doc the synthetic BOS is NOT a real corpus token — it is not carried in any segment's token tuple — so `segment_token_starts[0] == 1` (first content token) while the BOS-inclusive convention would give 0. Consumed by `steering/conditional/splice_provenance.py`; the loader works in segment positions (derived doc index + ordinal), not token offsets, so this convention never surfaces there.

### Schema v2 upgrade note

Sidecars written before issue #596 (schema_version 1) lack `splices` /
`n_splice_positions`; `feature_delta.mode=splice_counterfactual` refuses them.
Regenerate the derived directory (delete it, or re-run the splicer with
`skip_existing=false`). Re-splicing rewrites the derived `.zst` shards — the
tensors are `t.equal`-identical but the bytes differ (safetensors metadata
order), so consumers that pin `base_shard_content_hash` will treat existing
overlays as stale; recompute overlays after regenerating.

### Counterfactual splice-delta conditioning (issue #596)

Training with `feature_delta: {mode: splice_counterfactual}` conditions the
single segment after each stitch junction on `pct(segment) − pct(would-have-
followed segment)` — the natural continuation the splice displaced — and every
other segment on plain absolute percentiles. Requirements: a schema-v2 derived
dir as `feature_dir`, `data_dir=<derived>/_parquets` (auto-derived when unset),
live dataloader (`use_precomputed_batches: false`). The loader fails if a
splice reference is missing/empty or if the count of delta-scored segments does
not match the sidecar splice positions for the processed docs; a full
single-rank shard pass logs `scored/params.n_splice_positions`. See
[encoder-decoder-training.md](encoder-decoder-training.md).

## Overlay & statistics interaction

| Producer | On the derived dir |
|---|---|
| `compute_statistics` | recompute; within-shard splicing preserves each shard's feature multiset, so per-feature stats equal the base; corpus/segment-length diagnostics legitimately differ for new-started docs |
| `precompute_percentiles` | recompute; each segment's stored percentiles equal its base source segment's |
| `precompute_token_boundaries` | regenerate from derived shards + `_parquets`; char-based, robust to seam token-merging |
| `precompute_packed_rows` | regenerate; provenance points at the derived shards |
| `precompute_gradient_attribution` | **root-started docs only** — raises on any new-started doc (seg-0 size breaks `_verify_doc_alignment`); check `n_loops + n_splits == 0` across all shard provenances before running |
| `validate_features` structural | passes on the derived dir |
| `validate_features` source cross-ref | **root-started docs only** — raises on new-started docs (re-segmentation mismatch); check `n_loops + n_splits == 0` before enabling |

### Validation shard

The last shard by the dataloader's positional convention is the validation shard.
The derived dir's last shard is the spliced val shard automatically. Dual-validation
training wiring (issue #605) shipped in PR #606 (packed-row flavor): a splice run on
the canonical `<base>/splice/cut_frac*/seed*` layout runs two conditioned-BPB
validation passes at each eval cadence — `val/*` over the base (unspliced) validation
rows and `val_spliced/*` over the derived rows. Counterfactual splice-delta
conditioning (issue #596, below) adds a **live** flavor of the same two passes for
`feature_delta.mode=splice_counterfactual`. See the **Evaluation During Training**
section of `encoder-decoder-training.md` (including the `original_val_feature_dir`
override and, for the packed flavor, the `feature_delta=null` requirement).

## Determinism

`(seed, corpus, model, segment_size, cut_fraction)` fully determine random-mode
output. Opposite-percentile mode additionally depends on its search bound,
algorithm, and exact percentile table.
The derived parquets and provenance JSON are byte-identical across reruns, DDP
striping, and `shard_parallelism`. Ordinary local feature-shard `.zst` bytes are
not reproducible (safetensors metadata key order is randomized), so the local
determinism oracle compares **decompressed tensors** (`t.equal`) and metadata.
Remote feature publication opts into canonical safetensors headers before
compression, making its complete bytes reproducible across restarts with the
same inputs and runtime. This changes no decoded tensors or metadata values.

Different `cut_fraction` values produce independent cut selections even with the
same `seed` and `shard_index`, because `recombination_seed(seed, shard_index,
cut_fraction)` hashes all three together.

## Profiling and benchmark harness

Set `profile=true` to write a rank-local JSON sidecar at
`profile_summary_output` (`.rankNNNNN` is inserted before the suffix when
`WORLD_SIZE>1`). The sidecar is written atomically after each completed shard in
serial mode and after each completed future when `shard_parallelism>1`, so a long
run can be inspected without risking torn JSON.

Top-level schema:

| Field | Description |
|---|---|
| `config` | Resolved feature/output dirs, model, segment/context sizes, `cut_fraction`, `seed`, rank/world size, `shard_parallelism`, and `async_feature_save_depth`. |
| `totals` | Wall seconds, ok/failed/skipped counts, docs/switches/segments/features, input/output bytes, CPU seconds, max high-water RSS, summed phase seconds, and summed detailed seconds. |
| `per_shard[]` | One record per processed shard with `shard_stem`, `shard_index`, status, counters, input/output bytes, phase seconds, detailed seconds, final RSS/HWM, CPU seconds, and error type/message on failure. |

Fixed phase keys are:
`load_features`, `read_parquet`, `load_source_segments`, `recombine`,
`build_derived_shard`, `build_provenance`, `write_parquet`,
`write_provenance`, `save_features`, and `total`.

Detailed timing keys live under `detail_seconds` and preserve the top-level
phase keys while splitting expensive phases into stable substeps:

- Source loading: `source.truncate_tokenize`, `source.align_attach`,
  `source.batch_roundtrip`, `source.trim_unstable`.
- Recombination: `recombine.prepare`, `recombine.select_cuts`,
  `recombine.draw_derangement`, `recombine.opposite_pairing` (active mode only),
  `recombine.prefilter_seams`,
  `recombine.read_graph`, `recombine.resolve_loops`,
  `recombine.split_overflow`, `recombine.full_doc_oracle`.
- Derived shard/provenance: `derived.prepare`, `derived.piece_boundaries`,
  `derived.char_text`, `derived.feature_refs`, `derived.tensor_assembly`,
  `provenance.placement`, `provenance.docs`.
- Save path: `save.sort_segments`, `save.delta_encode`,
  `save.assemble_payload`, `save.safetensors_serialize`,
  `save.zstd_compress`, `save.atomic_write`.

`async_feature_save_depth>0` is an opt-in serial-mode pipeline: one background
thread saves the completed feature shard while the main thread starts the next
shard's splice. It is rejected with `shard_parallelism>1` because process-pool
mode already runs whole shards in separate workers. A shard is not counted as
successful, and its profile is not added to the run sidecar, until the save
future completes; any async save exception records a failed shard.

`zstd_threads=0` retains single-threaded compression. A positive integer opts
into that many zstd workers per active shard save. Budget those workers alongside
`shard_parallelism`; decoded feature values are unchanged, but compressed bytes
can differ. Nonzero settings are recorded in provenance, and resume requires the
same setting (an absent legacy key means zero). To change it, use a fresh output
directory or regenerate local outputs with `skip_existing=false`; remote
immutable claims require a fresh output prefix if their bytes would change.

`zstd_level=null` keeps the existing writer default (currently 12) and adds no
provenance key. Integers 1–22 select the writer's existing compression level.
Resume treats an absent/null recorded key as `null`; explicit `12` is distinct.
Use a fresh output directory or remote prefix when changing levels, especially
with immutable upload claims. For example, write a separate derived dataset:

```bash
uv run python -m scripts.preprocess.splice_feature_shards \
  feature_dir=<base-feature-dir> data_dir=<source-parquets> context_window=null \
  output_dir=<new-output-dir> zstd_level=8 zstd_threads=2
```

With `async_feature_save_depth>0`, per-shard `cpu_seconds` measures the
prepare/save threads and excludes native compression-worker CPU. Synchronous
profiles use process CPU time and include those workers; these CPU totals are
not directly comparable. Wall-clock phase timings include compression completion.

Benchmark bounded real inputs without writing into the production feature tree:

```bash
uv run python -m scripts.benchmark.benchmark_splice_feature_shards \
  feature_dir=out/data/features/.../base \
  data_dir=$NANOCHAT_BASE_DIR/base_data \
  max_shards=4 \
  cut_fraction=0.15 \
  report_md=out/bench/splice_feature_shards.md
```

The harness mirrors the first `max_shards` sorted stems into an isolated
`out/bench/splice_feature_shards/run_*` directory using symlinks when possible
(copies otherwise), runs each variant with `skip_existing=false` and
`profile=true`, and reports per-phase median/p90, wall seconds, shards/hour,
output bytes, high-water RSS, CPU seconds, plus a second detail table sorted by
total detailed seconds. A present base statistics file is also mirrored so
variants can set `pairing_mode=opposite_percentile` and a search bound.

### Opposite-percentile CPU measurement

A bounded component benchmark used one real L22 evaluation-feature shard:
24,107 segments, 50,689,690 stored activations, and 65,536 SAE features. At
`cut_fraction=0.15`, the seeded selection chose 2,380 of 15,868 legal cuts.
The table below compares each search bound with the same initial random
derangement, using one CPU thread and the median of three warm repetitions.

| Candidate bound | Pairing time | L1 increase | Shared-overlap reduction |
|---|---:|---:|---:|
| 16 | 2.282 s | 7.21% | 27.92% |
| 64 (default) | 3.313 s | 8.01% | 31.04% |
| 256 | 3.460 s | 8.57% | 33.18% |

These timings include percentile conversion for the selected groups and
partner optimization. Loading the shard took 0.720 s; loading statistics and
building the percentile table took 3.928 s, separately from pairing. The
measurement excludes runner metadata validation, tokenization, safety repairs,
and derived-output serialization/I/O (including canonical remote serialization),
so it establishes the component's cost and initial score improvement on this
shard, not complete-pipeline throughput or final-junction improvements. The
[raw receipt](../performance/evidence/splice-opposite-percentile-l22-cpu.json)
records all repetitions, source/statistics hashes, percentile checksum,
runtime versions, CPU model, and the exact pairing-module SHA-256. Its peak RSS
is the high-water mark of the shared process that ran the frozen and current
implementations sequentially. An explicit comparison produced identical partner
assignments for all three bounds. These timing samples do not establish a
causal speedup from simplification because run order and host variance were
not isolated.

### Repeated singleton decode measurement

A full-splice CPU profile on the same real evaluation shard above found
2,796,774 tokenizer decode calls; character-offset construction accounted for
21.073 s of the profiled run. `token_to_char_offsets` now decodes each distinct
singleton token once per document. This call-local reuse preserves the complete
text comparison, progressive Unicode fallback, and tokenizer changes between
calls; storage is bounded by the document's distinct token IDs.

An unprofiled ABBA comparison against main `a7a66446` measured the following
medians with one CPU thread on a Ryzen 9 9950X:

| Scope | Main | Singleton reuse | Time reduction |
|---|---:|---:|---:|
| Full local splice, including per-run setup and output I/O | 81.057 s | 71.563 s | 11.71% |
| Derived character/text offsets | 12.965 s | 7.894 s | 39.11% |

All feature tensors and scalar metadata, parquet bytes, and provenance JSON
matched exactly; source hashes were unchanged. The
[raw receipt](../performance/evidence/splice-token-offsets-local-cpu.json)
records all four runs, inputs, code hashes, and runtime versions. These are two
samples per arm on one warm-cache shard with other workstation jobs running.
Process imports are excluded; this is local splice evidence, without a training
throughput, compressed-feature-byte equality, or per-arm peak-memory claim.
The existing benchmark harness above can run the same input/configuration on
each revision; keep profiling traces separate from timed comparisons.

### Optional compression workers measurement

The same real evaluation shard (24,107 segments / 50.7 million activations)
spent 10.88 s in compression in a fresh local comparison. Opting into
`zstd_threads=2` reduced this to 6.03 s. An ABBA comparison without cProfile,
using the splice phase timers, gave these medians:

| Scope | Default compression | Two zstd workers | Change |
|---|---:|---:|---:|
| Full local splice, including per-run setup and output I/O | 37.065 s | 34.077 s | 8.06% less time |
| Compression phase | 10.880 s | 6.028 s | 44.59% less time |
| Synchronous shard CPU time | 28.656 s | 29.766 s | 3.87% more CPU |
| Compressed feature bytes | 100,172,343.5 | 100,423,115.5 | 0.25% larger |

The baseline used the exact `feature_io` module from merged main `103f66bf`
inside the candidate runner with the default setting; the candidate enabled
two workers. Every decoded feature tensor and scalar, parquet bytes, and
provenance except the intentional compression parameter matched exactly.
Source hashes were unchanged, and all generated shard outputs were removed.
The default remains zero; choose worker counts alongside shard parallelism.

The [receipt](../performance/evidence/splice-zstd-workers-local-cpu.json)
contains all samples and source/input hashes. These are two samples per arm on
one warm-cache shard on a shared Ryzen 9 9950X workstation. PyTorch/BLAS used
one thread; the candidate added two native compression workers. Imports and
equality/cleanup checks are excluded from timing. There is no per-arm memory,
async/multiprocess throughput, or GPU/training claim. Host conditions differed
from the earlier singleton-decode experiment, so compare within each ABBA run.

### Optional compression level measurement

Eight complete real-shard splices compared `zstd_level=null` (writer default 12)
with opt-in `zstd_level=8`; both used two compression workers. The unchanged
writer processed 24,107 segments / 50,689,690 activations with opposite-percentile
pairing in ABBABAAB order. Four samples per setting gave these medians:

| Scope | Default level 12 | Opt-in level 8 | Change |
|---|---:|---:|---:|
| Full local splice, including per-run setup and output I/O | 30.302 s | 25.734 s | 15.07% less time |
| Compression phase | 5.696 s | 1.489 s | 73.86% less time |
| Full-run process CPU, including compression workers | 34.299 s | 25.892 s | 24.51% less CPU |
| Compressed feature bytes | 100,423,114.5 | 103,890,938.5 | 3.45% larger |

Decoded feature tensors/scalars and parquet bytes matched all seven comparisons;
parsed provenance differed only by the candidate's intentional `zstd_level=8`.
Total feature/parquet/provenance bytes increased 3.19%. Default `null` retains the
unchanged writer and omits the new provenance key. The
[receipt](../performance/evidence/splice-zstd-level-local-cpu.json) preserves all
samples and input/code/driver hashes. These are warm-cache synchronous local CPU
measurements on one shard on a shared Ryzen 9 9950X workstation. Process imports
and validation/cleanup are excluded from timing. There is no GPU/training, network,
async/process throughput, per-arm memory, or other-corpus speedup claim.

### Batched HF singleton decode measurement

The next full-splice profile still recorded 1,289,566 Hugging Face decode calls.
Character-offset construction took 8.552 s in that profiled run. For HF
tokenizers using the standard public `decode` and `batch_decode` methods,
`token_to_char_offsets` now batches the document's unique singleton IDs through
`batch_decode`. An object array avoids Python-list conversion overhead while
preserving integer values, cleanup defaults and special tokens. Overrides,
generic tokenizers, small inputs and the tiktoken byte path retain their
existing behavior; the full-text check and Unicode fallbacks are unchanged.

On the actual derived shard (8,240 documents / 2,495,191 tokens), interleaved
ABBA/BAAB calls within each document measured 13.781 s for the parent function
and 12.222 s for batching: **11.31% less offset time**. These totals include two
complete function calls per arm per document, including array construction.
Every offset matched exactly; 63 of 65 blocks of up to 128 documents were
faster. Inputs were retokenized and verified against stored token counts and
exact text round trips before timing.

Eight complete splice runs also matched every decoded feature tensor/scalar,
parquet bytes and provenance JSON. Their observed medians were 61.625 s versus
50.183 s, but unchanged compression ranged from 5.973 to 14.523 s across runs.
These noisy whole-run timings do **not** establish a causal end-to-end speedup.
The paired calls provide warm-cache offset-workload evidence; they are not
independent full-splice repetitions. Imports, validation, comparison and logging
are outside paired timing; there is no cold-start, GPU/training or per-arm
memory claim. All generated shard outputs were removed after verification.

The [receipt](../performance/evidence/splice-hf-batch-offsets-local-cpu.json)
records the full-run samples, paired totals, input/source hashes and limits.

### HF source-text decode measurement

`to_text` passes a one-dimensional object array to unchanged standard HF decoding.
Integer values, defaults, empty input, single-BOS removal and decoder overrides stay intact.

Across 18,432 live splice helper invocations, paired calls took 3.838 s versus 3.508 s:
**8.59% less helper time**, only **164.81 ms** per one shard's helper traversal.
Alternating ABBA/BAAB used two calls per arm; all strings, decoded features,
parquet bytes and provenance matched. Source-only passes saved 9.59% and 9.57%.

The four-run ABBA candidate median was **35.505 s versus 33.485 s**,
**6.03% slower**. These observations establish no causal end-to-end gain.
The host was shared and warm; checks were outside timed helper calls. No cold-start,
GPU/training throughput or per-arm memory claim. The
[receipt](../performance/evidence/splice-to-text-array-local-cpu.json) records samples, hashes, checks and cleanup.

Of four additional slow seam tests, three passed; an uncached Gemma3-1b noop case failed
before the helper with either implementation. A cached 4b model substitution passed the
same noop assertions with zero helper calls; it does not qualify the original 1b case
or add helper coverage.

### Bounded derived-shard singleton decode cache

`build_derived_shard` shares singleton token decodes only within one build, using
an initially empty dictionary capped at 16,384 entries. Standard HF public
`decode` / `batch_decode` and backend `_decode` methods are required for reuse.
The tokenizer and its default decode settings stay unchanged during that build.
Overflow clears the dictionary and recomputes the document's unique IDs; an
oversized document uses the existing per-document path, leaving the shared cache intact.
The offset helper defaults to no shared cache. Small inputs, custom decoders,
generic tokenizers, tiktoken, full-text checks and Unicode fallbacks retain their
existing paths. No configuration or provenance field changes.

Eight native complete splices used one actual Gemma3-4b shard, CPU execution,
`zstd_level=8`, two zstd workers and ABBA/BAAB order. The measured character/text
phase median was **4.964 s versus 3.740 s**, **24.65% less phase time**;
the two order groups measured 16.76% and 25.61% less. All seven comparisons to
the first baseline matched decoded feature tensors/scalars, parquet bytes and
provenance JSON exactly, without removing keys.

Pooled full-run medians were 34.995 s versus 28.885 s (17.46% lower), but group
changes were only 0.42% and 7.97% lower. The first group's shard wall time was
**1.60% higher**; process CPU fell 2.03% and 7.83% in the two groups. Host load
rose, and unchanged pairing time fell 30.14%. These observations **do not
establish a causal full-pipeline speedup**.
A separate instrumented complete-helper prototype matched all 8,240 offsets and
saved 0.918 s (27.20%); counters inside timing make this feasibility evidence.
The shared host was warm; there is no isolated-host, cold-start, GPU/training,
network throughput or per-arm memory claim. The
[receipt](../performance/evidence/splice-hf-shard-cache-local-cpu.json) records
all eight samples, both order groups, source/input hashes, tests and limits.
All eight generated outputs and the clean baseline fixture were removed after
verification; raw reports, hashes and cleanup receipts were retained.

### Raw-text tokenizer startup

The splice CLI passes `load_chat_template=False` to `resolve_tokenizer` in local
and remote mode. Raw splicing does not format chat messages, so it keeps the base
tokenizer's existing template and skips borrowing a missing one. This intentionally
removes failures caused only by an unavailable/invalid template source; unknown-model
and base-tokenizer failures still propagate. Other callers retain the default `True`.

Eight fresh-process native splices of one cached Gemma3-4b CPU shard, with
`zstd_level=8` and two zstd workers, timed `sfs.run` after imports/config
composition: medians **24.881→22.138 s** (11.02% lower) and process CPU
**25.354→23.134 s** (8.76% lower). The ABBA/BAAB
groups measured 13.73%/11.78% lower wall time. All seven artifact comparisons matched
every decoded feature tensor/scalar, raw metadata key, parquet and provenance exactly.
The per-run outside-shard residual fell 7.401→5.342 s; this includes all setup and
finalization, not only tokenizer resolution. Unchanged shard wall time also fell
4.91%, so these shared-host observations do not isolate a causal 11.02% pipeline gain.

A separate warm-process resolver prototype measured 5.777→2.986 s (48.32% lower),
with fresh tokenizers and exact non-chat state/checksum/encoding/decoding on all
8,239 source documents. Its sole state difference was the borrowed chat template.
The affected gates passed 282 distinct tests, including five cached Gemma integrations.
The [receipt](../performance/evidence/splice-tokenizer-template-optout-local-cpu.json)
records all samples, scopes, source hashes and verified cleanup. No cold-cache/download,
network, GPU/training throughput or per-arm peak-memory claim is made.

### Percentile-table key construction

`build_percentile_table` creates its immutable `p1`…`p99` key tuple once per call,
after the existing table allocation. Per-feature values, tensor construction,
monotonicity checks and assignments keep their existing order and behavior.

In eight fresh-process native splices on one cached Gemma3-4b-pt CPU shard
(`zstd_level=8`, two zstd workers), the complete decorated builder measured
**1.927→1.462 s** (24.11% lower), with 6.14%/35.73% lower time in the two order
groups. Identical harness-only wrappers timed the actual function in both arms;
their clock/list overhead was included in full `sfs.run` wall and process CPU,
timed after imports/config composition. All seven complete artifact comparisons
matched decoded tensors/scalars, all raw metadata, parquet and provenance exactly.

Whole-run timing was inconclusive: pooled wall/CPU fell 6.28%/6.62%, but the first
group was **15.62%/17.03% higher** and the second 17.38%/16.92% lower. Unchanged
shard wall time fell 4.52%; these observations do not establish a net pipeline gain.
A separate warm-process complete-builder prototype measured 1.349→0.844 s
(37.49% lower), with exact float64 tables/checksums on 65,471 actual statistics
rows and unchanged inputs/defaults. Gates passed 223 tests and retained two
pre-existing expected failures separately. The
[receipt](../performance/evidence/splice-percentile-key-hoist-local-cpu.json)
records every sample, source hash, scope and verified cleanup. No cold-cache,
network, GPU/training throughput or per-arm peak-memory claim is made.

### Parity rules

The canonical artifact comparator requires:

- same shard stems, same `_parquets` stems, and same `_splice_provenance` stems;
- feature shards equal after `load_features()` by tensor keys, scalar metadata,
  dtypes, shapes, and `t.equal` values;
- derived parquets byte-equal and parsed row text equal;
- provenance sidecars byte-equal and parsed JSON equal.

Raw local `.safetensors.zst` byte equality is intentionally not an oracle because
safetensors metadata ordering can vary even when loaded tensors are identical.
Remote restart tests additionally require exact feature bytes and unchanged
immutable claims after a full run interruption and recomputation.

### Resume and completion marker

In local mode, the orchestrator writes the derived parquet first, then provenance, and saves
the feature shard last. The feature shard remains the completion marker in both
synchronous and async-save modes: a skip is allowed only when shard, parquet, and
schema-v2 provenance all exist and the sidecar params match the requested run.
Pairing mode, active search bound, algorithm, and percentile-table checksum
participate in that comparison. Legacy sidecars without pairing fields are
accepted only as random-mode outputs.
Local partial shard/parquet pairs, missing sidecars, schema-v1 sidecars, corrupt JSON,
and mismatched params remain loud errors rather than resumable states.

### Bounded HF-backed execution

`remote.enabled=true` keeps the logical `feature_dir` and `output_dir` in the
ordinary `out/data/features/...` namespace while mapping them into one HF bucket.
Source parquets stay local. Each rank seals one source/output listing snapshot,
validates every matching parquet and every completed provenance before publishing
anything, and owns source indices satisfying
`shard_index % WORLD_SIZE == RANK`.

Opposite-percentile mode also requires the local statistics control file at
`<logical feature_dir>/feature_statistics.safetensors`. It is loaded once per
rank; feature bodies still follow the bounded remote pipeline. The output
prefix includes the same pairing suffix as local mode, and remote resumes
verify the same pairing identity fields before publication.

The rank-local staging namespace is
`<staging_dir>/rankNNNNN/{source,output,verify}`. A nonblocking `flock` prevents
two local processes from sharing it. Startup removes only recognized runner
shard/temp residue; an unknown entry blocks the run and is left untouched. The
pipeline uses one download worker and one upload worker: raw shard `N+1` and the
commit of `N-1` may overlap the unchanged synchronous splice/save of `N`.
Backpressure joins the previous commit before submitting the current one, bounding
staging to two raw bodies and two output triplets plus writer scratch.

Each commit uploads the derived parquet, provenance, and feature body in that
order. Before a new canonical path is written, the verified uploader publishes a
small immutable content-addressed claim beneath that path's sibling
`_verified_upload_claims/` namespace. Concurrent identical writers share the
same claim and remain idempotent; differing writers create distinct claims and
fail before either can overwrite the canonical path. Claims persist as the
cross-host collision fence and are ignored by shard discovery. Every canonical
object is size/Xet verified. Only after the feature-last triplet verifies are its
three local outputs unlinked. Raw remote objects are never changed or deleted. A
parquet-only or parquet-plus-provenance publication is resumable by deterministic
recomputation; any state containing a feature body without both prerequisites is
invalid.

Remote feature serialization uses the existing canonical safetensors writer so
an interruption after the feature claim but before its body upload can restart
and reproduce the exact claimed bytes. Ordinary local writers keep their
existing serialization defaults. Completed remote triplets from older writers
still verify and skip without rewriting their bodies or claims.

An orphan claim from a pre-fix run can name noncanonical feature bytes that a
new run cannot reproduce. Such claims are not automatically migrated: recovery
requires the exact originally claimed bytes, or a new output prefix. If those
original bytes are gone, the conflict remains a loud error. The runner never
deletes, ignores, or replaces a conflicting claim.

Use `remote.expected_shard_count=null` while extraction is still append-only.
For the final corpus gate, set the exact count (for example `1400`) and compare
the `source_inventory_digest` recorded by every rank profile. A later all-skip
pass verifies one snapshotted raw body first to resolve model/segment/context
independently, then verifies every completed provenance sidecar against those
source-bound parameters and the requested logical `feature_dir`.

```bash
uv run -m scripts.preprocess.splice_feature_shards \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  data_dir=/path/to/base_data context_window=2048 cut_fraction=0.15 seed=42 \
  remote.enabled=true remote.staging_dir=/scratch/splice-l22-ci \
  remote.expected_shard_count=1400 profile=true
```

## Configuration reference (`configs/extraction/splice_feature_shards.yaml`)

| Key | Default | Description |
|---|---|---|
| `feature_dir` | `null` | Base feature directory (**required**). |
| `data_dir` | `null` | Source parquet dir (`null` = nanochat default base data). |
| `output_dir` | `null` | Random: `<feature_dir>/splice/cut_frac{cut_fraction}/seed{seed}`. Opposite-percentile appends `_opposite_percentile_pool{B}` to the seed leaf. |
| `model` / `segment_size` | `null` | `null` = resolve from base shard metadata + the `m_*` path leaf; an explicit value must match. |
| `context_window` | `1024` | Shared source-document window; must match the base shard metadata exactly. |
| `cut_fraction` | `0.15` | Fraction of splice points selected as cuts, in `[0.0, 1.0]`. NOT a per-boundary probability. |
| `seed` | `42` | Global RNG seed; per-shard seed = `recombination_seed(seed, shard_index, cut_fraction)`. |
| `pairing_mode` | `random` | `random` preserves the existing seeded derangement; `opposite_percentile` improves the incoming segment's L1 percentile distance from the original continuation it replaces, using local base statistics. |
| `pairing_candidate_pool_size` | `64` | Integer ≥ 2 bounding each optimization group in opposite-percentile mode; ignored in random mode. |
| `skip_existing` | `true` | Skip shards whose derived shard + parquet + provenance already exist **and whose sidecar params match this run** (a mismatch raises). Requires `write_provenance: true`. |
| `write_provenance` | `true` | Write the `_splice_provenance/` sidecar. Must stay `true` while `skip_existing` is `true`: `skip_existing: true` + `write_provenance: false` is rejected at config validation (an unverified resume has no sidecar to check recorded params against). Set `skip_existing: false` to recompute without provenance. |
| `shard_parallelism` | `1` | `>1` uses spawn worker processes (one shard each). |
| `zstd_threads` | `0` | Integer ≥ 0: zero preserves single-threaded compression; positive values enable that many workers per active save. Recorded when nonzero; resume requires the same setting. Compressed bytes may change, while decoded features remain exact. |
| `zstd_level` | `null` | `null` keeps the writer default (currently 12) and omits the provenance key. Otherwise a strict integer in `[1, 22]`; recorded values must match exactly on resume, so explicit `12` differs from `null`. Compressed bytes, CPU cost and size can change; use a fresh output directory/prefix when changing levels. |
| `async_feature_save_depth` | `0` | `0` keeps synchronous save behavior. `>0` enables a bounded serial pipeline with one async feature-shard save worker and the requested pending-save depth; rejected with `shard_parallelism>1`. |
| `profile` | `false` | Write the per-shard JSON profile sidecar described above. |
| `profile_summary_output` | `out/traces/splice_feature_shards_profile.json` | Profile JSON path; multi-rank runs write `.rankNNNNN`-suffixed siblings. |
| `remote.*` | disabled | Optional bounded verified HF-backed mode described above; queue depths are fixed at one. |

**All shards are processed** — there is no `start_shard` / `end_shard` range
filter. DDP striping (`RANK`/`WORLD_SIZE`) distributes shards across ranks; each
rank's output is identical to what a single-rank run would produce for those
shards.

**Multi-GPU / torchrun:** honours `RANK`/`WORLD_SIZE` like extraction. Local mode
retains its positional sorted slice. Remote mode uses stable index modulo
ownership. Per-shard output is independent (global-index seeding), so either
striped run is identical to a single-rank run for the same shard.
