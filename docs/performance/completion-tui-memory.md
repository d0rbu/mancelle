# Completion TUI memory investigation

## Observed failure mode

On 2026-08-05, opening the cluster/live-highlight path for the L22 65k
feature set produced approximately 77 GiB of process RSS. `/proc/<pid>/smaps_rollup`
reported approximately 77 GiB of anonymous RSS and 95 GiB of process swap,
while `nvidia-smi` reported only approximately 3.4 GiB of GPU memory. The
machine had 123 GiB of RAM and began thrashing once the process was active.

The 1.5 TiB feature directory is mostly on-disk artifacts; it is not loaded
in full. `DocumentLoader` does eagerly materialize the selected validation
shard through `FeatureIndex.doc_ids_for_shard`. With the current defaults this
keeps feature IDs and magnitudes resident, but that shard is only a secondary
allocation (the compressed shard measured about 1.9 GiB, with about 3.6 GiB
decompressed).

## Likely primary cause

Entering live highlighting lazily loads a probe model. On CUDA,
`load_probe_model` first loads a TransformerLens Gemma model and then calls
`build_nnterp_model`, which constructs a second HuggingFace/nnterp Gemma model.
The returned `ProbeModel` retains both `model` (TransformerLens fallback) and
`nnterp_model` (the normal activation path), after trimming their tails. The
GPU footprint therefore does not reveal the duplicated CPU/offloaded model
weights, which explains the very large anonymous RSS and swap footprint.

The generation model is also lazy in the TUI, so it was not needed to produce
this initial spike. Triggering generation later can add another model.

## Optimization status

The completion TUI now requests `keep_transformerlens_model=False` when it
loads the probe. If nnterp builds successfully, the TransformerLens fallback
is trimmed, released, garbage-collected, and removed from the `ProbeModel`; the
TUI keeps only the nnterp model and SAE. If nnterp is unavailable, the existing
TransformerLens path remains available. A failed nnterp call with the fallback
released now fails explicitly instead of attempting to dereference a missing
model.

Probe activations and SAE encoding still run on the probe model's CUDA device;
only percentile lookup and final highlight assembly are moved to CPU. The
context length is retained as metadata so releasing TransformerLens does not
change context-window clipping.

Clustered completion-TUI launches now also construct `FeatureIndex` with
`skip_feature_mags=True` and `compact_feature_ids=True`. Cluster sampling uses
percentiles and IDs, not raw magnitudes, so this removes the largest unused
resident tensor and keeps IDs at the narrow on-disk dtype. Unclustered launches
retain the previous magnitude-backed path because their runtime samplers need
the raw values.

Document preview and selection loads run in a cancellable background worker.
Each request carries a monotonically increasing sequence number, so a result
that finishes after a newer arrow-key selection is discarded. This keeps shard
decompression and feature sampling off Textual's event loop; the loader also
serializes FeatureIndex lookups because cancellation cannot interrupt a Python
thread already inside a cache operation.

The focused regression suite covers fallback ownership, nnterp failure
behavior, context metadata, and the existing highlight outputs. A real-model
before/after RSS, anonymous-RSS, swap, and GPU-allocation measurement remains
the follow-up acceptance run; `torch.cuda.empty_cache()` alone would not have
addressed the CPU duplication.
