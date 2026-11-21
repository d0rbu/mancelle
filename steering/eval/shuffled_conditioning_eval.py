"""Shuffled conditioning evaluation helpers.

Metric semantics (Wave 3D task 1)
---------------------------------
This evaluator computes ``shuffled_same_seg_delta = real_mean - shuffled_mean``
where:

- ``real`` = correct conditioning (the target document's own features at
  its target segment).
- ``shuffled`` = a DONOR document's features injected at THIS document's
  SAME target segment position. Position is held constant; only the
  feature payload is swapped. This isolates the conditioning PAYLOAD from
  positional effects.

This is **not** the canonical "specificity" metric. The canonical
specificity (``real - random_doc_random_seg``) lives in
``steering/eval/inline_steering_eval.py``. The primary aggregated values
are ``real_mean`` and ``shuffled_mean`` (see
:class:`ShuffledConditioningAggregated`); consumers compute the delta
themselves.
"""

from __future__ import annotations

import enum
import math
import random as _random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import torch as t
import yaml
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanochat.engine import KVCache
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig
from steering.eval._stats_helpers import compute_mean_stderr_or_nan
from steering.eval.completion_steering_eval import (
    _activation_to_percentile,
    _build_percentile_lookup,
)
from steering.eval.feature_probe import build_completion_mask, max_pool_masked
from steering.eval.multiseg_generator import (
    CompletionSample,
    _decode_from_prefill,
    _prefill_condition,
    _segment_bounds_for_assignment,
    build_multiseg_inputs,
    prepare_segments,
)
from steering.eval.probe_activations import get_probe_activations
from steering.sae_resolver import ActivationType
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from steering.stats_types import PerFeatureStat
from steering.tokenizer_protocols import PrefixedTextTokenizer


def assert_no_feature_clustering(meta: dict | None) -> None:
    """Fail fast for every unsupported clustered checkpoint schema."""
    reject_feature_clusters_metadata(meta, "shuffled_conditioning_eval checkpoint")


class FeatureSelectionStrategy(enum.Enum):
    """How to pick which features to probe in the shuffled conditioning eval."""

    top_k = "top_k"
    random_k = "random_k"


def select_top_k_features(
    segment: SegmentFeatures,
    per_feature_stats: dict[str, PerFeatureStat],
    k: int = 16,
) -> list[int]:
    """Select top-k features by percentile with deterministic tie-breaking.

    Converts each feature's raw activation magnitude to a percentile using
    per-feature statistics, then sorts by (percentile descending, feature_id
    ascending) and returns the first ``min(k, n_active)`` feature IDs.

    Features whose stats are missing from *per_feature_stats* are skipped
    with a warning.

    Raises:
        ValueError: If the segment has no active features.
    """
    fids = segment.feature_ids
    mags = segment.feature_mags
    n_active = fids.numel()

    if n_active == 0:
        raise ValueError("no active features in segment")

    # Convert raw magnitudes to percentiles
    pairs: list[tuple[float, int]] = []
    for fid_t, mag_t in zip(fids.tolist(), mags.tolist(), strict=True):
        fid = int(fid_t)
        stat = per_feature_stats.get(str(fid))
        if stat is None:
            logger.warning("No stats for feature {}, skipping in top-k selection", fid)
            continue
        lookup = _build_percentile_lookup(stat.percentiles)
        pct = _activation_to_percentile(float(mag_t), lookup)
        pairs.append((pct, fid))

    n_with_stats = len(pairs)
    if n_with_stats == 0:
        raise ValueError(
            f"Segment has {n_active} active features but none have "
            f"entries in per_feature_stats — cannot rank by percentile"
        )

    if k > n_with_stats:
        logger.warning("Requested k={} but only {} features with stats", k, n_with_stats)

    # Sort by (-percentile, +feature_id) for deterministic ordering
    pairs.sort(key=lambda x: (-x[0], x[1]))
    top = pairs[: min(k, n_with_stats)]
    return [fid for _, fid in top]


def select_random_k_features(
    segment: SegmentFeatures,
    per_feature_stats: dict[str, PerFeatureStat],
    k: int = 16,
    seed: int = 42,
) -> list[int]:
    """Select k random features from a segment's active set.

    Filters to features that have entries in *per_feature_stats*, then
    randomly samples k of them (or all if fewer than k are available).

    Raises:
        ValueError: If the segment has no active features, or none have stats.
    """
    fids = segment.feature_ids
    n_active = fids.numel()

    if n_active == 0:
        raise ValueError("no active features in segment")

    candidates: list[int] = []
    for fid_t in fids.tolist():
        fid = int(fid_t)
        if per_feature_stats.get(str(fid)) is not None:
            candidates.append(fid)

    if not candidates:
        raise ValueError(
            f"Segment has {n_active} active features but none have "
            f"entries in per_feature_stats — cannot select random features"
        )

    if k > len(candidates):
        logger.warning("Requested k={} but only {} features with stats", k, len(candidates))

    rng = _random.Random(seed)
    selected = rng.sample(candidates, min(k, len(candidates)))
    return selected


def replace_target_segment(
    segments: list[SegmentFeatures],
    target_seg_idx: int,
    donor: SegmentFeatures,
) -> list[SegmentFeatures]:
    """Replace a target segment's features with donor features.

    Returns a new list where the segment at *target_seg_idx* has its
    feature_ids and feature_mags replaced by clones of the donor's tensors,
    while keeping the original segment's metadata (segment_idx, position,
    char_start, char_end).  The original list and its segments are not mutated.
    """
    result: list[SegmentFeatures] = []
    for i, seg in enumerate(segments):
        if i == target_seg_idx:
            result.append(
                SegmentFeatures(
                    segment_idx=seg.segment_idx,
                    position=seg.position,
                    feature_ids=donor.feature_ids.clone(),
                    feature_mags=donor.feature_mags.clone(),
                    char_start=seg.char_start,
                    char_end=seg.char_end,
                )
            )
        else:
            result.append(seg)
    return result


def sample_shuffled_segment(
    target_segment_idx: int,
    current_doc_idx: int,
    feature_index: Any,
    val_shard_stem: str,
    all_doc_indices: list[int],
    boundaries: t.Tensor,
    threshold_cfg: ThresholdConfig | None,
    precomputed_thresholds: t.Tensor | None,
    rng: _random.Random,
    max_attempts: int = 100,
) -> tuple[SegmentFeatures, int]:
    """Sample a shuffled segment from a different document.

    Randomly picks documents from *all_doc_indices* (excluding
    *current_doc_idx*), verifies the candidate document has enough
    segments, runs :func:`prepare_segments` on the target segment, and
    checks that at least one feature is active post-threshold.

    Returns:
        A tuple of (prepared segment with int64 bucket magnitudes,
        donor document index).

    Raises:
        ValueError: After *max_attempts* without finding a valid donor.
    """
    candidates = [idx for idx in all_doc_indices if idx != current_doc_idx]
    if not candidates:
        raise ValueError(
            f"Could not find a valid donor document after {max_attempts} "
            f"attempts (no candidates besides current_doc_idx={current_doc_idx})"
        )

    for _ in range(max_attempts):
        donor_doc_idx = rng.choice(candidates)
        # donor_doc_idx is drawn from the shard's own doc keys ⇒ present; lookup
        # raises KeyError only on genuine corruption (issue #521).
        raw_segments = feature_index.lookup(val_shard_stem, donor_doc_idx)
        if len(raw_segments) <= target_segment_idx:
            continue

        # Prepare just the target segment from the donor doc
        target_seg = raw_segments[target_segment_idx]
        prepared = prepare_segments([target_seg], boundaries, threshold_cfg, precomputed_thresholds)
        if not prepared:
            logger.debug(
                f"prepare_segments returned empty for donor doc {donor_doc_idx} "
                f"segment {target_segment_idx} — skipping"
            )
            continue

        seg = prepared[0]
        if seg.feature_ids.numel() == 0:
            logger.debug(
                f"Donor doc {donor_doc_idx} segment {target_segment_idx} "
                f"has no active features after thresholding — skipping"
            )
            continue

        return seg, donor_doc_idx

    raise ValueError(
        f"Could not find a valid donor segment after {max_attempts} attempts "
        f"for target_segment_idx={target_segment_idx}"
    )


def _compute_percentiles_for_features(
    activations: dict[int, float],
    per_feature_stats: dict[str, PerFeatureStat],
) -> dict[int, float]:
    """Convert raw activations to percentiles using per-feature stats.

    For each feature in *activations*, looks up the corresponding
    :class:`PerFeatureStat`, builds a percentile lookup table, and
    converts the raw activation to a percentile in [0, 1].

    Args:
        activations: Mapping of feature_id to raw activation magnitude.
        per_feature_stats: Mapping of stringified feature_id to stats.

    Returns:
        Mapping of feature_id to percentile (0-1 scale).
    """
    result: dict[int, float] = {}
    for fid, act in activations.items():
        stat = per_feature_stats.get(str(fid))
        if stat is None:
            logger.warning("No stats for feature {}, skipping percentile", fid)
            continue
        lookup = _build_percentile_lookup(stat.percentiles)
        result[fid] = _activation_to_percentile(act, lookup)
    return result


def compute_topk_percentile_activation(
    completion_text: str,
    pre_segment_text: str,
    top_k_feature_ids: list[int],
    per_feature_stats: dict[str, PerFeatureStat],
    base_model: Any,
    sae: Any,
    gemma_tokenizer: Any,
    layer: int,
    hook_name: str,
    device: t.device,
    nnterp_model: Any = None,
) -> float:
    """Measure average percentile activation of top-k features in a completion.

    Tokenizes the full text (pre_segment + completion), runs the base model
    and SAE encoder, max-pools each feature's activation over completion
    tokens, converts to percentiles, and returns the mean percentile.

    Returns:
        Mean percentile across top-k features, or NaN if no valid measurement.
    """
    full_text = pre_segment_text + completion_text
    tokens = gemma_tokenizer.encode(full_text)

    if not tokens:
        return float("nan")

    completion_char_start = len(pre_segment_text)
    char_offsets = token_to_char_offsets(gemma_tokenizer, tokens)
    mask = build_completion_mask(char_offsets, completion_char_start)

    if not mask.any():
        return float("nan")

    tokens_tensor = t.tensor([tokens], device=device)
    with t.no_grad():
        act = get_probe_activations(
            base_model,
            tokens_tensor,
            layer,
            hook_name=hook_name,
            activation_type=ActivationType.residual,
            nnterp_model=nnterp_model,
        )
    act = act.squeeze(0).float()

    all_features = sae.encode(act)

    raw_activations: dict[int, float] = {}
    for fid in top_k_feature_ids:
        raw_activations[fid] = max_pool_masked(all_features[:, fid], mask)

    percentiles = _compute_percentiles_for_features(raw_activations, per_feature_stats)

    valid = [v for v in percentiles.values() if not math.isnan(v)]
    if not valid:
        return float("nan")

    return sum(valid) / len(valid)


# ---------------------------------------------------------------------------
# Data structures for shuffled conditioning results
# ---------------------------------------------------------------------------


@dataclass
class ShuffledConditioningResult:
    """Results for a single document in the shuffled conditioning eval."""

    doc_idx: int
    target_segment_idx: int
    n_top_k_features: int
    top_k_feature_ids: list[int]
    ground_truth_text: str
    pre_segment_text: str
    shuffled_doc_idx: int
    top_k_feature_descriptions: list[str] = field(default_factory=list)
    real_samples: list[tuple[str, float]] = field(default_factory=list)
    shuffled_samples: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ShuffledConditioningAggregated:
    """Aggregated statistics across all documents."""

    n_documents: int
    k: int
    real_mean: float
    real_stderr: float
    shuffled_mean: float
    shuffled_stderr: float


def _doc_mean(samples: list[tuple[str, float]]) -> float:
    """Compute mean percentile for a single document's samples, filtering NaN."""
    vals = [pct for _, pct in samples if not math.isnan(pct)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def aggregate_results(
    results: list[ShuffledConditioningResult],
    k: int,
) -> ShuffledConditioningAggregated:
    """Aggregate per-document results into summary statistics.

    Computes per-document means first (since samples within a document are
    correlated), then computes the mean and standard error across documents.
    """
    real_doc_means = [_doc_mean(r.real_samples) for r in results]
    shuffled_doc_means = [_doc_mean(r.shuffled_samples) for r in results]

    real_m, real_se = compute_mean_stderr_or_nan(real_doc_means)
    shuf_m, shuf_se = compute_mean_stderr_or_nan(shuffled_doc_means)

    return ShuffledConditioningAggregated(
        n_documents=len(results),
        k=k,
        real_mean=real_m,
        real_stderr=real_se,
        shuffled_mean=shuf_m,
        shuffled_stderr=shuf_se,
    )


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def serialize_results(results: list[ShuffledConditioningResult], path: Path | str) -> None:
    """Serialize a list of ShuffledConditioningResult to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append(
            {
                "doc_idx": r.doc_idx,
                "target_segment_idx": r.target_segment_idx,
                "n_top_k_features": r.n_top_k_features,
                "top_k_feature_ids": r.top_k_feature_ids,
                "ground_truth_text": r.ground_truth_text,
                "pre_segment_text": r.pre_segment_text,
                "shuffled_doc_idx": r.shuffled_doc_idx,
                "top_k_feature_descriptions": r.top_k_feature_descriptions,
                "real_samples": [[text, pct] for text, pct in r.real_samples],
                "shuffled_samples": [[text, pct] for text, pct in r.shuffled_samples],
            }
        )
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def deserialize_results(path: Path | str) -> list[ShuffledConditioningResult]:
    """Deserialize a list of ShuffledConditioningResult from YAML."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    results: list[ShuffledConditioningResult] = []
    for d in data:
        results.append(
            ShuffledConditioningResult(
                doc_idx=d["doc_idx"],
                target_segment_idx=d["target_segment_idx"],
                n_top_k_features=d["n_top_k_features"],
                top_k_feature_ids=d["top_k_feature_ids"],
                ground_truth_text=d["ground_truth_text"],
                pre_segment_text=d["pre_segment_text"],
                shuffled_doc_idx=d["shuffled_doc_idx"],
                top_k_feature_descriptions=d.get("top_k_feature_descriptions", []),
                real_samples=[(text, float(pct)) for text, pct in d["real_samples"]],
                shuffled_samples=[(text, float(pct)) for text, pct in d["shuffled_samples"]],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Document selection and generation
# ---------------------------------------------------------------------------


def select_eval_documents(
    doc_segments: Mapping[int, list[SegmentFeatures]],
    n_documents: int,
    doc_texts: list[str] | None = None,
    tokenizer: PrefixedTextTokenizer | None = None,
    max_seq_len: int | None = None,
    bos_id: int = 2,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Select documents for evaluation, returning (doc_idx, target_seg_idx) pairs.

    Filters to documents with >= 2 segments, sets target to the last segment
    (list position, not segment_idx attribute), shuffles with seed, and
    returns the first n_documents.

    When *doc_texts*, *tokenizer*, and *max_seq_len* are provided, documents
    whose prefix + target segment tokens exceed the context window are
    excluded before selection.
    """
    ctx_args = (doc_texts, tokenizer, max_seq_len)
    ctx_provided = sum(a is not None for a in ctx_args)
    if ctx_provided not in (0, 3):
        raise ValueError(
            f"Context-window filtering requires all three of doc_texts, "
            f"tokenizer, and max_seq_len (got {ctx_provided}/3)"
        )
    do_ctx_filter = ctx_provided == 3
    # Hoist BOS prefix computation out of the loop (loop-invariant)
    if do_ctx_filter:
        assert doc_texts is not None
        assert tokenizer is not None
        assert max_seq_len is not None
        doc_texts_for_filter = doc_texts
        tokenizer_for_filter = tokenizer
        max_seq_len_for_filter = int(max_seq_len)
        bos_prefix = tokenizer_for_filter.decode([bos_id])
    else:
        doc_texts_for_filter = []
        tokenizer_for_filter = None
        max_seq_len_for_filter = 0
        bos_prefix = ""

    eligible: list[tuple[int, int]] = []
    for doc_idx, segs in doc_segments.items():
        if len(segs) >= 2:
            target = len(segs) - 1  # list position of last segment
            if do_ctx_filter:
                if doc_idx >= len(doc_texts_for_filter):
                    continue
                target_seg = segs[target]
                # char_end is in BOS-prefixed coordinates
                text_through_target = (bos_prefix + doc_texts_for_filter[doc_idx])[
                    len(bos_prefix) : target_seg.char_end
                ]
                assert tokenizer_for_filter is not None
                n_tokens = len(tokenizer_for_filter.encode(text_through_target, prepend=bos_id))
                if n_tokens > max_seq_len_for_filter:
                    continue
            eligible.append((doc_idx, target))

    rng = _random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n_documents]


@t.inference_mode()
def generate_for_shuffled_document(
    model: Any,
    tokenizer: Any,
    doc_text: str,
    segments: list[SegmentFeatures],
    target_segment_idx: int,
    shuffled_segment: SegmentFeatures,
    boundaries: t.Tensor,
    threshold_cfg: ThresholdConfig | None,
    precomputed_thresholds: t.Tensor | None,
    device: t.device,
    samples_per_condition: int = 5,
    temperature: float = 0.7,
    top_k: int | None = 50,
    base_seed: int = 42,
    bos_id: int = 2,
    next_segment_conditioning_offset: bool = False,
) -> tuple[str, str, list[CompletionSample]]:
    """Generate completions under real and shuffled conditions.

    Derives prefix tokens and generation token count from character-level
    segment assignments rather than a fixed ``segment_size`` parameter.

    Returns:
        (pre_segment_text, ground_truth_text, list of CompletionSample)
    """
    target_seg = segments[target_segment_idx]

    # BOS-inclusive coordinate system (same pattern as generate_for_document)
    bos_prefix = tokenizer.decode([bos_id])
    bos_text = bos_prefix + doc_text

    ground_truth = bos_text[target_seg.char_start : target_seg.char_end]
    pre_segment_text = bos_text[len(bos_prefix) : target_seg.char_start]

    # Tokenize all text through end of target segment to derive prefix + gen length
    text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
    all_tokens = tokenizer.encode(text_through_target, prepend=bos_id)

    # Compute character-level segment assignments (same as training dataloader)
    char_offsets = token_to_char_offsets(tokenizer, all_tokens)
    token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
    token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

    seg_char_starts, seg_char_ends = _segment_bounds_for_assignment(
        segments[: target_segment_idx + 1],
        token_starts,
        token_ends,
    )

    assignments = assign_tokens_to_segments(
        token_starts,
        token_ends,
        seg_char_starts,
        seg_char_ends,
    )
    assignments = t.cummax(assignments, dim=0).values

    # Split: prefix = tokens before target segment, gen = tokens in target segment
    target_positions = (assignments == target_segment_idx).nonzero(as_tuple=True)[0]
    if len(target_positions) > 0:
        split_idx = int(target_positions[0].item())
    else:
        split_idx = len(all_tokens)

    prefix_tokens = all_tokens[:split_idx]
    n_gen_tokens = len(all_tokens) - split_idx

    if n_gen_tokens == 0:
        raise ValueError(
            f"Ground truth segment [{target_seg.char_start}:{target_seg.char_end}] "
            f"encodes to 0 tokens"
        )

    n_prefix = len(prefix_tokens)

    # Validate that prefix + generation fits in the context window.
    dec_config = model.dec_config
    max_seq_len = dec_config.sequence_len
    total_len = n_prefix + n_gen_tokens
    if total_len > max_seq_len:
        raise ValueError(
            f"Prefix ({n_prefix} tokens) + gen ({n_gen_tokens}) = "
            f"{total_len} exceeds context window ({max_seq_len}). "
            f"This document/segment should have been filtered out during selection."
        )

    # Choose dtype: bfloat16 on CUDA, float32 otherwise
    dtype = t.bfloat16 if device.type == "cuda" else t.float32

    # Prepare segments up to and including target
    prepared = prepare_segments(
        segments[: target_segment_idx + 1],
        boundaries,
        threshold_cfg,
        precomputed_thresholds,
    )

    conditions: list[tuple[str, list[SegmentFeatures]]] = [
        ("real", prepared),
        (
            "shuffled",
            replace_target_segment(prepared, target_segment_idx, shuffled_segment),
        ),
    ]

    samples: list[CompletionSample] = []
    seed_counter = 0

    for condition_name, cond_segs in conditions:
        encoder_inputs = build_multiseg_inputs(
            segments=cond_segs,
            prefix_tokens=prefix_tokens,
            n_gen_tokens=n_gen_tokens,
            target_segment_idx=target_segment_idx,
            tokenizer=tokenizer,
            next_segment_conditioning_offset=next_segment_conditioning_offset,
        )

        # Encoder forward pass (once per condition)
        enc_fids = encoder_inputs.enc_feature_ids.to(device)
        enc_mags = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        seg_map = encoder_inputs.seg_map.to(device)

        # Allocate base KV cache and prefill once
        base_kv_cache = KVCache(
            batch_size=1,
            num_heads=dec_config.n_kv_head,
            seq_len=total_len,
            head_dim=dec_config.n_embd // dec_config.n_head,
            num_layers=dec_config.n_layer,
            device=device,
            dtype=dtype,
        )

        base_last_logits = _prefill_condition(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_out=encoder_out,
            seg_map=seg_map,
            kv_cache=base_kv_cache,
            device=device,
        )

        for _i in range(samples_per_condition):
            seed = base_seed + seed_counter
            seed_counter += 1

            sample_kv = KVCache(
                batch_size=1,
                num_heads=dec_config.n_kv_head,
                seq_len=total_len,
                head_dim=dec_config.n_embd // dec_config.n_head,
                num_layers=dec_config.n_layer,
                device=device,
                dtype=dtype,
            )
            sample_kv.prefill(base_kv_cache)

            gen_tokens = list(
                _decode_from_prefill(
                    model=model,
                    kv_cache=sample_kv,
                    last_logits=base_last_logits,
                    encoder_out=encoder_out,
                    seg_map=seg_map,
                    n_prefix=n_prefix,
                    max_new_tokens=n_gen_tokens,
                    total_len=total_len,
                    device=device,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                    bos_id=bos_id,
                )
            )
            text = tokenizer.decode(gen_tokens)
            samples.append(
                CompletionSample(
                    condition=condition_name,
                    text=text,
                    seed=seed,
                )
            )

    return pre_segment_text, ground_truth, samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_shuffled_comparison(
    aggregated: ShuffledConditioningAggregated,
    output_path: Path | str,
) -> None:
    """Bar chart comparing real vs shuffled mean percentile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Real", "Shuffled"]
    means = [aggregated.real_mean, aggregated.shuffled_mean]
    stderrs = [aggregated.real_stderr, aggregated.shuffled_stderr]
    colors = ["#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(labels)))
    ax.bar(
        x,
        means,
        yerr=stderrs,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean top-k percentile activation")
    ax.set_title(f"Shuffled Conditioning Eval (n={aggregated.n_documents}, k={aggregated.k})")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved shuffled comparison plot to {output_path}")
