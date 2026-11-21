"""Inline Steering Eval — shuffled completion roundtrip fidelity metric.

Measures whether SAE feature conditioning causally influences the
encoder-decoder model by generating completions under two conditioned variants
(real features and random features from a different document),
then running each completion back through the SAE's base model to measure
how much the *ground-truth* features reactivate.

A positive specificity (real > random) means the model generates text specific
to the *correct* features, not just any conditioning.

Supports DDP: work is partitioned across ranks and results are gathered.
"""

from __future__ import annotations

import math
import random as stdlib_random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from loguru import logger

from nanochat.dataset import list_parquet_files
from steering.conditional.feature_clusters import reject_feature_clusters_metadata
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.percentile_table import magnitude_to_percentile
from steering.eval.clustered_inline_cache import (
    ClusteredInlineEvalPayload,
)
from steering.eval.completion_steering_eval import apply_threshold_to_shard
from steering.eval.feature_probe import (
    apply_sae_activation_batched,
    build_completion_mask,
    compute_raw_probe_batched,
    max_pool_masked_batched,
)
from steering.eval.multiseg_generator import (
    CompletionSample,
    MultisegInputs,
    build_multiseg_inputs,
    generate_samples_for_condition,
    prepare_segments,
)
from steering.eval.probe_activations import get_probe_activations
from steering.eval.probe_model import ProbeModel as _ProbeModel
from steering.feature_io import glob_features, load_features
from steering.segmentation import token_to_char_offsets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


CLUSTERED_PROBE_SEMANTICS = "representative_raw_member_max_percentile_v1"


@dataclass
class SteeringEvalConfig:
    """Configuration for inline steering evaluation during training."""

    n_eval_docs: int = 10
    samples_per_condition: int = 2
    temperature: float = 0.7
    top_k_sampling: int | None = 50
    seed: int = 42


@dataclass(frozen=True)
class InlineEvalCache:
    """Deterministic inputs for :func:`evaluate_steering`, computed once per config.

    Bundles the three expensive-to-compute-but-deterministic values that the
    inline steering eval needs: the randomly-sampled doc/segment pairs, the
    associated scan metrics, and the val parquet's doc_idx -> text map.

    Built once by :func:`prepare_inline_eval_cache` and reused across every
    subsequent ``evaluate_steering`` call within a single training config.
    Frozen so the cached ``selection_metrics`` dict cannot be mutated across
    calls (``evaluate_steering`` always works on a shallow copy).
    """

    selected_docs: list[tuple[int, int]]
    """(doc_idx, target_segment_idx) pairs surviving post-threshold selection."""

    selection_metrics: dict[str, int | float]
    """Metric dict from ``_select_random_docs``: steering/n_docs_available, steering/n_docs_selected."""

    doc_texts: dict[int, str]
    """doc_idx -> raw text from the last validation parquet shard."""

    clustered_payload: ClusteredInlineEvalPayload | None = None
    """Validated offline selection/provenance join for clustered steering."""


# ---------------------------------------------------------------------------
# Document-text loader (extracted for reuse by cache)
# ---------------------------------------------------------------------------


def _load_doc_texts(data_dir: str | None) -> dict[int, str]:
    """Load doc_idx -> text dict from the last validation parquet shard.

    Extracted from the inline block in :func:`evaluate_steering` so that
    :func:`prepare_inline_eval_cache` can call the same code path.
    """
    val_parquet = list_parquet_files(data_dir=data_dir)[-1]
    texts = pq.read_table(val_parquet, columns=["text"])["text"].to_pylist()
    return dict(enumerate(texts))


def prepare_inline_eval_cache(
    feature_dir: Path | str,
    threshold_cfg,
    precomputed_thresholds,
    data_dir: str | None,
    n_eval_docs: int,
    seed: int,
    feature_mask: FeatureMask | None = None,
) -> InlineEvalCache:
    """Build the deterministic inputs for :func:`evaluate_steering` once per config.

    Calls :func:`_select_random_docs` (which loads the validation shard and
    applies thresholding) and :func:`_load_doc_texts` (which reads the val
    parquet). Both are deterministic in the given inputs, so the result can
    be reused across every eval call in a training config.

    When the feature dir has no shards (or no segments survive thresholding),
    returns a cache with empty ``selected_docs`` and ``doc_texts`` — the
    short-circuit at the top of :func:`evaluate_steering` will return the
    scan metrics without touching the parquet.
    """
    feature_dir = Path(feature_dir)
    selected_docs, selection_metrics = _select_random_docs(
        feature_dir=feature_dir,
        threshold_cfg=threshold_cfg,
        precomputed_thresholds=precomputed_thresholds,
        n_eval_docs=n_eval_docs,
        seed=seed,
        feature_mask=feature_mask,
    )
    # Skip parquet read when we have no docs to generate for — this matches
    # the fast-return path inside evaluate_steering and avoids raising on
    # misconfigured data_dir when we'd never use the result anyway.
    doc_texts: dict[int, str] = {}
    if selected_docs:
        doc_texts = _load_doc_texts(data_dir)
    return InlineEvalCache(
        selected_docs=selected_docs,
        selection_metrics=selection_metrics,
        doc_texts=doc_texts,
    )


# ---------------------------------------------------------------------------
# Scan + doc selection
# ---------------------------------------------------------------------------


def _find_val_shard(feature_dir) -> Path | None:
    """Find validation shard (last shard in feature_dir)."""
    shards = glob_features(feature_dir)
    return shards[-1] if shards else None


def _shard_stem(shard_path: Path) -> str:
    """Extract shard stem for FeatureIndex lookup."""
    return shard_path.name.split("_features")[0]


def _select_random_docs(
    feature_dir,
    threshold_cfg,
    precomputed_thresholds,
    n_eval_docs: int,
    seed: int = 42,
    feature_mask: FeatureMask | None = None,
) -> tuple[list[tuple[int, int]], dict[str, int | float]]:
    """Pick random validation docs that have post-threshold features.

    Loads the validation shard, applies thresholding, finds all (doc, segment)
    pairs that survived, then randomly samples ``n_eval_docs`` unique documents.
    For each selected doc, a random segment with features is chosen as the target.

    Returns ``(selected_docs, metrics)`` where ``selected_docs`` is a list of
    ``(doc_idx, target_segment_idx)`` pairs.
    """
    feature_dir = Path(feature_dir)
    shard_path = _find_val_shard(feature_dir)
    if shard_path is None:
        return [], {"steering/n_docs_available": 0, "steering/n_docs_selected": 0}

    shard_data = load_features(shard_path)
    thresholded = apply_threshold_to_shard(shard_data, threshold_cfg, precomputed_thresholds)

    # Find docs that have at least one feature after thresholding.
    # For each such doc, collect the segments that have features.
    doc_segments: dict[int, list[int]] = {}
    offsets = thresholded.offsets
    doc_ids = thresholded.doc_ids
    seg_indices = thresholded.segment_indices

    for i in range(thresholded.n_segments):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        if end > start:  # segment has features after thresholding
            seg_fids = thresholded.feature_ids[start:end]
            if feature_mask is not None:
                seg_fids = feature_mask.apply_to_positive(seg_fids)[0]
            if len(seg_fids) == 0:
                continue
            d = int(doc_ids[i].item())
            s = int(seg_indices[i].item())
            if s >= 1:
                doc_segments.setdefault(d, []).append(s)

    rng = stdlib_random.Random(seed)
    available_docs = list(doc_segments.keys())
    rng.shuffle(available_docs)
    chosen = available_docs[:n_eval_docs]

    # For each chosen doc, pick a random segment with features as target
    selected = [(d, rng.choice(doc_segments[d])) for d in chosen]

    metrics: dict[str, int | float] = {
        "steering/n_docs_available": len(available_docs),
        "steering/n_docs_selected": len(selected),
    }
    return selected, metrics


# ---------------------------------------------------------------------------
# DDP work partitioning
# ---------------------------------------------------------------------------


def _partition_docs(
    selected_docs: list[tuple[int, int]],
    rank: int,
    world_size: int,
) -> list[tuple[int, int]]:
    """Partition (doc_idx, seg_idx) pairs across ranks via round-robin."""
    if world_size <= 1:
        return selected_docs
    return selected_docs[rank::world_size]


def _gather_int(local_val: int, world_size: int, device: torch.device | str = "cpu") -> int:
    """All-reduce (sum) a single int across ranks."""
    if world_size <= 1 or not dist.is_initialized():
        return local_val
    t = torch.tensor([local_val], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


# ---------------------------------------------------------------------------
# Roundtrip data structures
# ---------------------------------------------------------------------------


@dataclass
class RoundtripCompletion:
    """One document's roundtrip eval results: samples under all conditions."""

    doc_idx: int
    target_segment_idx: int
    pre_segment_text: str
    ground_truth_feature_ids: list[int]
    samples: list[CompletionSample] = field(default_factory=list)
    ground_truth_member_ids: list[list[int]] | None = None


# ---------------------------------------------------------------------------
# Roundtrip generation
# ---------------------------------------------------------------------------


def _tokenize_and_split(doc_text, segments, target_seg_idx, tokenizer):
    """Tokenize document through target segment and find prefix/target split.

    Requires the ``tok/<checksum>/seg_boundaries`` overlay — produced by
    ``scripts.preprocess.precompute_token_boundaries`` and loaded into
    ``SegmentFeatures.token_start`` / ``token_end`` by
    ``FeatureIndex._load_shard``. Those indices are the authoritative split
    boundaries in the **training tokenizer's** token space (i.e. indices
    into ``tokenizer.encode(doc_text, prepend=bos_id)``). The precompute
    pipeline already accounts for source-vs-training BOS-length deltas,
    zero-width byte-fallback tokens, and the tiny-trailing-seg repair, so
    consuming its output sidesteps every retokenize-and-realign edge case.

    Coordinate spaces (to prevent future confusion):
      * ``segment.char_start`` / ``char_end`` — **source** (extraction)
        tokenizer's decoded-text char coords (e.g. Gemma), BOS-inclusive.
        Used only to derive the returned ``pre_segment_text`` display
        prefix. Never used for splitting.
      * ``segment.token_start`` / ``token_end`` — **training** tokenizer
        token indices. Used for the authoritative split.

    Fails loud:
      * ``token_start``/``token_end`` is None → the overlay was not loaded.
        Precompute it via ``scripts.preprocess.precompute_token_boundaries`` and
        rerun. The inline eval intentionally has no char-alignment
        fallback: the fallback had a documented degenerate case
        (``n_gen_tokens == 0`` when a token's midpoint aligned with a
        segment boundary under ``searchsorted(..., side="left")``) that
        would silently mask overlay-less configurations.
      * overlay bounds don't fit the tokenized doc → overlay was built on
        a different parquet text than the one passed to eval. Verify the
        feature shard, seg_boundaries overlay, and val parquet all come
        from the same dataset.

    Returns:
        ``(all_tokens, split_idx, pre_segment_text)`` where
        ``all_tokens[:split_idx]`` is the prefix tokens (nanochat space)
        and ``all_tokens[split_idx:]`` is the ground-truth target span.
    """
    bos_id = tokenizer.get_bos_token_id()
    bos_prefix = tokenizer.decode([bos_id])
    target_seg = segments[target_seg_idx]

    if target_seg.token_start is None or target_seg.token_end is None:
        raise ValueError(
            f"SegmentFeatures for target_seg_idx={target_seg_idx} has no "
            f"token_start/token_end, but the inline steering eval requires "
            f"the seg_boundaries overlay to route through the training "
            f"tokenizer's authoritative boundaries. Precompute the overlay "
            f"via `uv run python -m scripts.preprocess.precompute_token_boundaries` "
            f"and construct FeatureIndex with a matching tokenizer_checksum. "
            f"No char-alignment fallback: the fallback had an alignment "
            f"edge case (n_gen_tokens == 0 on searchsorted boundary "
            f"ties) that silently dropped eligible docs."
        )

    full_tokens = tokenizer.encode(doc_text, prepend=bos_id)
    split_idx = int(target_seg.token_start)
    end_idx = int(target_seg.token_end)
    n_full = len(full_tokens)
    # These invariants are enforced by the precompute pipeline
    # (``_extract_ranges_from_monotonic`` raises on empty segs; the
    # trailing-seg repair ensures ``token_end > token_start``). A violation
    # here signals the overlay was computed on a different doc/text than
    # the one we just tokenized — fail loud, with the specific check that
    # failed named so operators can diagnose.
    _overlay_mismatch = (
        "The overlay was likely computed on a different parquet text than "
        "the one passed to evaluate_steering — verify the val parquet and "
        "the feature shard / seg_boundaries overlay all come from the same "
        "dataset."
    )
    if split_idx <= 0:
        raise ValueError(
            f"Overlay token_start must be > 0 (BOS token occupies index 0), "
            f"got token_start={split_idx} for target_seg_idx={target_seg_idx}. "
            + _overlay_mismatch
        )
    if end_idx <= split_idx:
        raise ValueError(
            f"Overlay token_end must be strictly greater than token_start, "
            f"got token_start={split_idx} >= token_end={end_idx} for "
            f"target_seg_idx={target_seg_idx} (empty target segment). "
            + _overlay_mismatch
        )
    if end_idx > n_full:
        raise ValueError(
            f"Overlay token_end={end_idx} exceeds full doc n_tokens={n_full} "
            f"for target_seg_idx={target_seg_idx}. "
            + _overlay_mismatch
        )

    all_tokens = full_tokens[:end_idx]
    # pre_segment_text is the doc text up to the target seg's char_start,
    # used by the caller as a display prefix and by the probe to locate
    # the completion span. Slicing ``(bos_prefix + doc_text)[len(bos_prefix):char_start]``
    # is a valid approximation when source BOS and training BOS have the
    # same decoded length (the production case: Gemma "<bos>" = nanochat
    # "<bos>" = 5 chars). The authoritative split uses token_start, so
    # minor char-arithmetic drift here never propagates to the tokenized
    # prefix / target split.
    pre_segment_text = (bos_prefix + doc_text)[len(bos_prefix) : target_seg.char_start]
    return all_tokens, split_idx, pre_segment_text


def _build_random_inputs(
    real_inputs: MultisegInputs,
    random_prepared_seg: SegmentFeatures,
    target_seg_idx: int,
    *,
    preserve_all_donor_features: bool = False,
) -> MultisegInputs:
    """Build encoder inputs with random features at the conditioning slot.

    Keeps the real document's ``seg_map`` and all non-target encoder features
    intact; only the encoder slot at ``target_seg_idx`` is overwritten with
    the random donor segment's features (zero-padded to ``F_max``).

    ``preserve_all_donor_features`` is enabled only by clustered steering,
    whose donor row is an already-selected, integrity-validated artifact row.
    In that mode, a wider donor expands the final feature dimension instead
    of being silently truncated. The default retains the literal legacy
    unclustered behavior.

    Args:
        real_inputs: Encoder inputs built for the real condition. The seg_map
            is preserved verbatim — the random condition only swaps the
            payload at ``target_seg_idx``.
        random_prepared_seg: Donor segment features (a ``SegmentFeatures``
            from a different document). ``feature_ids`` and ``feature_mags``
            are copied into ``cond_slot``.
        target_seg_idx: Source-coordinate segment index of the target
            segment.

    Returns:
        ``MultisegInputs`` with the random features placed at ``cond_slot``.
    """
    fids = real_inputs.enc_feature_ids.clone()
    mags = real_inputs.enc_magnitudes.clone()
    mask = real_inputs.enc_mask.clone()

    cond_slot = target_seg_idx

    n_encoder_seg = fids.shape[1]
    if cond_slot < 0 or cond_slot >= n_encoder_seg:
        raise ValueError(
            f"_build_random_inputs: target_seg_idx={target_seg_idx} is out of "
            f"range for encoder with {n_encoder_seg} slots."
        )

    F_max = fids.shape[2]
    donor_features = len(random_prepared_seg.feature_ids)
    if preserve_all_donor_features and donor_features > F_max:
        donor_padding = donor_features - F_max
        fids = torch.nn.functional.pad(fids, (0, donor_padding))
        mags = torch.nn.functional.pad(mags, (0, donor_padding))
        mask = torch.nn.functional.pad(mask, (0, donor_padding))
        F_max = donor_features

    fids[0, cond_slot, :] = 0
    mags[0, cond_slot, :] = 0
    mask[0, cond_slot, :] = False

    n_f = min(donor_features, F_max)
    if n_f > 0:
        fids[0, cond_slot, :n_f] = random_prepared_seg.feature_ids[:n_f].to(torch.int64)
        mags[0, cond_slot, :n_f] = random_prepared_seg.feature_mags[:n_f].to(torch.int64)
        mask[0, cond_slot, :n_f] = True

    return MultisegInputs(enc_feature_ids=fids, enc_magnitudes=mags, enc_mask=mask, seg_map=real_inputs.seg_map)


def _pick_random_segment(
    feature_index, val_shard_stem, exclude_doc_idx, boundaries,
    threshold_cfg, precomputed_thresholds, rng,
    feature_mask: FeatureMask | None = None,
) -> SegmentFeatures:
    """Pick a random prepared segment from a different document.

    Raises ValueError if no candidate is available — every selected doc
    must have at least one OTHER doc in the val shard for the random
    condition. This is a hard requirement; production val shards have
    thousands of docs, so a missing candidate is a real configuration bug
    (e.g., a single-doc test fixture that forgot to add a peer).
    """
    try:
        all_docs = feature_index.doc_ids_for_shard(val_shard_stem)
    except (KeyError, AttributeError) as e:
        raise ValueError(
            f"_pick_random_segment: feature_index.doc_ids_for_shard failed for "
            f"shard {val_shard_stem!r}: {e}"
        ) from e
    candidates = [d for d in all_docs if d != exclude_doc_idx]
    if not candidates:
        raise ValueError(
            f"_pick_random_segment: no other docs in shard {val_shard_stem!r} "
            f"besides exclude_doc_idx={exclude_doc_idx}. The val shard must have "
            f"at least 2 docs for the random condition to work."
        )
    random_doc = rng.choice(candidates)
    # random_doc came from doc_ids_for_shard ⇒ present; lookup raises KeyError
    # only on genuine corruption (issue #521).
    segments = feature_index.lookup(val_shard_stem, random_doc)
    random_seg = segments[rng.randrange(len(segments))]
    # prepare_segments always returns a list of length len(input) — single-element
    # input gives single-element output. Defensive check kept as an assertion.
    prepared = prepare_segments(
        [random_seg],
        boundaries,
        threshold_cfg,
        precomputed_thresholds,
        feature_mask=feature_mask,
    )
    assert len(prepared) == 1, "prepare_segments invariant violated"
    return prepared[0]


def _generate_roundtrip_samples(
    model,
    selected_docs: list[tuple[int, int]],
    doc_texts: dict,
    feature_index,
    val_shard_stem: str,
    tokenizer,
    boundaries,
    threshold_cfg,
    precomputed_thresholds,
    device: torch.device,
    cfg: SteeringEvalConfig,
    feature_mask: FeatureMask | None = None,
) -> list[RoundtripCompletion]:
    """Generate completions under real and random conditioned variants.
    """
    bos_id = tokenizer.get_bos_token_id()
    rng = stdlib_random.Random(cfg.seed)
    completions: list[RoundtripCompletion] = []

    for doc_idx, target_seg_idx in selected_docs:
        # doc_idx came from _select_random_docs over present docs; lookup raises
        # KeyError only on genuine corruption (issue #521).
        segments = feature_index.lookup(val_shard_stem, doc_idx)

        doc_text = doc_texts.get(doc_idx)
        if doc_text is None:
            raise ValueError(
                f"doc_texts dict has no entry for selected doc_idx={doc_idx}. "
                f"Available doc_idxs: {sorted(doc_texts.keys())[:20]}{'...' if len(doc_texts) > 20 else ''}. "
                f"The val parquet must contain every doc that _select_random_docs picks."
            )

        if target_seg_idx >= len(segments):
            raise ValueError(
                f"Selected target_seg_idx={target_seg_idx} is out of range for "
                f"doc_idx={doc_idx} which has only {len(segments)} segments. "
                f"_select_random_docs picked this segment; the index must contain it. "
                f"Check for stale caches or shard mismatch between selection and lookup."
            )

        segs = segments[: target_seg_idx + 1]
        prepared = prepare_segments(
            segs,
            boundaries,
            threshold_cfg,
            precomputed_thresholds,
            feature_mask=feature_mask,
        )

        gt_fids = prepared[target_seg_idx].feature_ids.tolist()
        if not gt_fids:
            # _select_random_docs only picks (doc, seg) pairs whose target segment
            # has at least one feature post-thresholding. If we reach this branch,
            # the selection logic disagrees with the probe-time threshold/discretize
            # logic — that's a real bug, not a tolerable edge case. Fail loudly.
            raise ValueError(
                f"Empty ground-truth features for selected (doc={doc_idx}, "
                f"target_seg={target_seg_idx}). _select_random_docs picked this "
                f"pair as having post-threshold features, but prepare_segments "
                f"produced no features at probe time. Check that the threshold_cfg "
                f"and precomputed_thresholds passed to evaluate_steering match the "
                f"ones used by _select_random_docs / prepare_inline_eval_cache."
            )

        try:
            all_tokens, split_idx, pre_text = _tokenize_and_split(
                doc_text, segs, target_seg_idx, tokenizer,
            )
        except Exception:
            logger.opt(exception=True).error(
                f"Tokenization failed for doc {doc_idx}"
            )
            raise

        prefix_tokens = all_tokens[:split_idx]
        n_gen_tokens = len(all_tokens) - split_idx
        # ``_tokenize_and_split`` raises on missing overlay or out-of-range
        # bounds, so both quantities here are guaranteed positive. The
        # assertion is a cheap invariant check to surface any future
        # refactor that weakens the helper's guarantees.
        assert split_idx > 0 and n_gen_tokens > 0, (
            f"_tokenize_and_split invariant violated: split_idx={split_idx}, "
            f"n_gen_tokens={n_gen_tokens} for doc_idx={doc_idx} "
            f"target_seg={target_seg_idx}"
        )

        # Build encoder inputs for two fully conditioned variants. Both are
        # guaranteed non-None: real is built from the target document, and
        # random is built via _pick_random_segment which raises on no-candidate.
        real_inputs = build_multiseg_inputs(
            prepared, prefix_tokens, n_gen_tokens, target_seg_idx, tokenizer,
        )

        random_seg = _pick_random_segment(
            feature_index, val_shard_stem, doc_idx,
            boundaries, threshold_cfg, precomputed_thresholds, rng,
            feature_mask=feature_mask,
        )
        random_inputs = _build_random_inputs(
            real_inputs, random_seg, target_seg_idx,
        )

        samples: list[CompletionSample] = []
        conditions: list[tuple[str, MultisegInputs]] = [
            ("real", real_inputs),
            ("random", random_inputs),
        ]
        seed_counter = 0

        for cond_name, enc_inputs in conditions:
            base_seed = cfg.seed + doc_idx * 1000 + seed_counter
            try:
                gen_tokens_list = generate_samples_for_condition(
                    model=model,
                    prefix_tokens=prefix_tokens,
                    encoder_inputs=enc_inputs,
                    n_samples=cfg.samples_per_condition,
                    device=device,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k_sampling,
                    base_seed=base_seed,
                    bos_id=bos_id,
                )
            except Exception:
                logger.opt(exception=True).error(
                    f"Generation failed: {cond_name} doc {doc_idx}"
                )
                raise
            for i, gen_tokens in enumerate(gen_tokens_list):
                sample_seed = base_seed + i
                text = tokenizer.decode(gen_tokens)
                samples.append(CompletionSample(
                    condition=cond_name, text=text, seed=sample_seed,
                ))
            seed_counter += cfg.samples_per_condition

        completions.append(RoundtripCompletion(
            doc_idx=doc_idx, target_segment_idx=target_seg_idx,
            pre_segment_text=pre_text, ground_truth_feature_ids=gt_fids,
            samples=samples,
        ))

    return completions


def _generate_clustered_roundtrip_samples(
    model,
    selected_docs: list[tuple[int, int]],
    doc_texts: dict[int, str],
    payload: ClusteredInlineEvalPayload,
    tokenizer,
    device: torch.device,
    cfg: SteeringEvalConfig,
) -> list[RoundtripCompletion]:
    """Generate from the validated offline clustered selection payload.

    The payload already contains bucketed original representative IDs for the
    required target-document prefixes and deterministic different-document
    donors.  Re-thresholding raw magnitudes here would discard splice-delta
    selection semantics, so this branch feeds those rows directly to the
    ordinary encoder-input builder.
    """
    bos_id = tokenizer.get_bos_token_id()
    completions: list[RoundtripCompletion] = []

    for doc_idx, target_seg_idx in selected_docs:
        target = payload.target(doc_idx, target_seg_idx)
        overlap = set(target.positive_feature_ids).intersection(
            target.negative_feature_ids
        )
        if overlap:
            raise ValueError(
                "clustered inline steering target marks feature ID "
                f"{min(overlap)} as both positive and negative"
            )
        segments = list(target.prefix_segments)
        if len(segments) != target_seg_idx + 1:
            raise ValueError(
                "clustered inline steering cache prefix length mismatch for "
                f"target {(doc_idx, target_seg_idx)}: got {len(segments)}"
            )
        for ordinal, segment in enumerate(segments):
            if segment.segment_idx != ordinal:
                raise ValueError(
                    "clustered inline steering cache prefix is not aligned to "
                    f"source segment ordinals for target {(doc_idx, target_seg_idx)}: "
                    f"slot {ordinal} contains segment_idx={segment.segment_idx}"
                )

        doc_text = doc_texts.get(doc_idx)
        if doc_text is None:
            raise ValueError(
                "clustered inline steering doc_texts has no entry for selected "
                f"doc_idx={doc_idx}"
            )

        all_tokens, split_idx, pre_text = _tokenize_and_split(
            doc_text,
            segments,
            target_seg_idx,
            tokenizer,
        )
        prefix_tokens = all_tokens[:split_idx]
        n_gen_tokens = len(all_tokens) - split_idx
        assert split_idx > 0 and n_gen_tokens > 0, (
            "_tokenize_and_split invariant violated for clustered inline "
            f"steering target {(doc_idx, target_seg_idx)}: "
            f"split_idx={split_idx}, n_gen_tokens={n_gen_tokens}"
        )

        real_inputs = build_multiseg_inputs(
            segments,
            prefix_tokens,
            n_gen_tokens,
            target_seg_idx,
            tokenizer,
        )
        random_inputs = _build_random_inputs(
            real_inputs,
            target.donor_segment,
            target_seg_idx,
            preserve_all_donor_features=True,
        )

        samples: list[CompletionSample] = []
        conditions: list[tuple[str, MultisegInputs]] = [
            ("real", real_inputs),
            ("random", random_inputs),
        ]
        seed_counter = 0
        for condition, encoder_inputs in conditions:
            base_seed = cfg.seed + doc_idx * 1000 + seed_counter
            try:
                generated = generate_samples_for_condition(
                    model=model,
                    prefix_tokens=prefix_tokens,
                    encoder_inputs=encoder_inputs,
                    n_samples=cfg.samples_per_condition,
                    device=device,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k_sampling,
                    base_seed=base_seed,
                    bos_id=bos_id,
                )
            except Exception:
                logger.opt(exception=True).error(
                    "Clustered inline steering generation failed: "
                    f"{condition} doc {doc_idx}"
                )
                raise
            for sample_idx, generated_tokens in enumerate(generated):
                samples.append(
                    CompletionSample(
                        condition=condition,
                        text=tokenizer.decode(generated_tokens),
                        seed=base_seed + sample_idx,
                    )
                )
            seed_counter += cfg.samples_per_condition

        completions.append(
            RoundtripCompletion(
                doc_idx=doc_idx,
                target_segment_idx=target_seg_idx,
                pre_segment_text=pre_text,
                ground_truth_feature_ids=list(target.positive_feature_ids),
                samples=samples,
                ground_truth_member_ids=[
                    list(group) for group in target.positive_member_groups
                ],
            )
        )

    return completions


# ---------------------------------------------------------------------------
# Roundtrip SAE probe
# ---------------------------------------------------------------------------


def _probe_roundtrip_raw(
    completions: list[RoundtripCompletion],
    probe_model: _ProbeModel,
    percentile_table: torch.Tensor | None = None,
) -> list[tuple[str, float, float, float]]:
    """Run completions through probe model, measuring ground-truth feature activations.

    Returns list of (condition, pre_act_avg, post_act_avg, pct_avg) tuples.
    pct_avg is the mean percentile rank of post-activation values (range [0.5, 99.5]),
    or nan if no percentile_table is provided.
    """
    scores: list[tuple[str, float, float, float]] = []
    device = probe_model.device

    for comp in completions:
        gt_fids = comp.ground_truth_feature_ids
        if not gt_fids:
            # Upstream (_generate_roundtrip_samples) raises ValueError if the
            # selected segment has no post-threshold features, so any completion
            # reaching here SHOULD have a non-empty gt_fids list. An empty list
            # at this point is a caller-side invariant violation.
            raise ValueError(
                f"RoundtripCompletion(doc_idx={comp.doc_idx}, "
                f"target_segment_idx={comp.target_segment_idx}) has empty "
                f"ground_truth_feature_ids. _generate_roundtrip_samples should "
                f"raise on empty GT features upstream — if you constructed this "
                f"completion directly (e.g., in tests), pass at least one feature ID."
            )
        member_groups = comp.ground_truth_member_ids
        if member_groups is not None:
            if len(member_groups) != len(gt_fids):
                raise ValueError(
                    "clustered roundtrip member groups must align one-to-one "
                    f"with representative IDs: {len(member_groups)} != {len(gt_fids)}"
                )
            seen_members: set[int] = set()
            for representative, members in zip(gt_fids, member_groups, strict=True):
                if not members or representative not in members:
                    raise ValueError(
                        "clustered roundtrip member group must be non-empty and "
                        f"contain representative {representative}: {members}"
                    )
                if len(members) != len(set(members)):
                    raise ValueError(
                        "clustered roundtrip member group contains duplicate IDs: "
                        f"representative={representative}, members={members}"
                    )
                overlap = seen_members.intersection(members)
                if overlap:
                    raise ValueError(
                        "clustered roundtrip member groups overlap at feature ID "
                        f"{min(overlap)}"
                    )
                seen_members.update(members)

        completion_char_start = len(comp.pre_segment_text)

        for sample in comp.samples:
            full_text = comp.pre_segment_text + sample.text
            # Empty / whitespace-only text means the sampler produced nothing
            # over a non-empty pre_segment_text, which is impossible on healthy
            # inputs (pre_segment_text is sliced from a real doc). Raise loudly
            # rather than silently dropping the sample.
            if not full_text.strip():
                raise ValueError(
                    f"Empty/whitespace-only full_text for doc={comp.doc_idx} "
                    f"target_seg={comp.target_segment_idx}: "
                    f"pre_segment_text={comp.pre_segment_text!r}, "
                    f"sample.text={sample.text!r}. "
                    f"Either the pre-segment text is degenerate or the sampler "
                    f"produced nothing — both indicate an upstream bug."
                )

            tokens = probe_model.tokenizer.encode(full_text)
            # Empty token list from a non-empty string means the probe model's
            # tokenizer failed (e.g., dropped every char). Raise rather than
            # silently skipping; this is a real configuration bug.
            if not tokens:
                raise ValueError(
                    f"probe_model.tokenizer.encode returned empty tokens for "
                    f"non-empty full_text (len={len(full_text)}) on "
                    f"doc={comp.doc_idx} target_seg={comp.target_segment_idx}. "
                    f"Check the tokenizer configuration."
                )

            char_offsets = token_to_char_offsets(probe_model.tokenizer, tokens)
            mask = build_completion_mask(char_offsets, completion_char_start)

            # An all-False mask means the completion span has zero tokens.
            # Previously this produced NaN-filled probe values that the
            # downstream filter dropped. The user wants loud failures instead:
            # an empty completion span is a real bug (the upstream tokenizer
            # produced a degenerate result, or the char-offset arithmetic is
            # broken) and should never happen on healthy inputs.
            if not bool(mask.any()):
                raise ValueError(
                    f"Completion mask is all-False for doc={comp.doc_idx} "
                    f"target_seg={comp.target_segment_idx} "
                    f"(completion_char_start={completion_char_start}, "
                    f"n_tokens={len(tokens)}). The completion span has zero "
                    f"tokens — check the tokenizer / char-offset alignment."
                )

            tokens_tensor = torch.tensor([tokens], device=device)
            act = get_probe_activations(
                probe_model.model, tokens_tensor,
                layer=probe_model.layer,
                hook_name=probe_model.hook_name,
                activation_type=probe_model.activation_type,
                nnterp_model=probe_model.nnterp_model,
            ).squeeze(0).float()

            if member_groups is not None:
                union_fids = sorted(
                    {feature_id for group in member_groups for feature_id in group}
                )
                union_fids_t = torch.tensor(
                    union_fids,
                    dtype=torch.int64,
                    device=device,
                )
                union_pre_probes = compute_raw_probe_batched(
                    act,
                    probe_model.sae.W_enc,
                    probe_model.sae.b_enc,
                    probe_model.sae.b_dec,
                    union_fids_t,
                    apply_b_dec=probe_model.apply_b_dec,
                )
                union_pre_vals = max_pool_masked_batched(union_pre_probes, mask)
                union_post_probes = apply_sae_activation_batched(
                    union_pre_probes,
                    probe_model.sae,
                    union_fids_t,
                )
                union_post_vals = max_pool_masked_batched(
                    union_post_probes,
                    mask,
                )
                if not bool(torch.isfinite(union_pre_vals).all()):
                    raise ValueError(
                        "Non-finite clustered pre_val in probe for "
                        f"doc={comp.doc_idx} target_seg={comp.target_segment_idx}: "
                        f"member_fids={union_fids}"
                    )
                if not bool(torch.isfinite(union_post_vals).all()):
                    raise ValueError(
                        "Non-finite clustered post_val in probe for "
                        f"doc={comp.doc_idx} target_seg={comp.target_segment_idx}: "
                        f"member_fids={union_fids}"
                    )

                union_position = {
                    feature_id: index for index, feature_id in enumerate(union_fids)
                }
                representative_positions = torch.tensor(
                    [union_position[feature_id] for feature_id in gt_fids],
                    dtype=torch.int64,
                    device=device,
                )
                representative_pre = union_pre_vals[representative_positions]
                representative_post = union_post_vals[representative_positions]
                pre_mean = representative_pre.mean().item()
                post_mean = representative_post.mean().item()

                if percentile_table is None:
                    percentile_mean = float("nan")
                else:
                    member_percentiles = magnitude_to_percentile(
                        torch.tensor(union_fids, dtype=torch.int64),
                        union_post_vals.detach().to(device="cpu", dtype=torch.float32),
                        percentile_table,
                    )
                    representative_percentiles = torch.stack(
                        [
                            member_percentiles[
                                torch.tensor(
                                    [union_position[feature_id] for feature_id in group],
                                    dtype=torch.int64,
                                )
                            ].amax()
                            for group in member_groups
                        ]
                    )
                    percentile_mean = representative_percentiles.mean().item()

                scores.append(
                    (
                        str(sample.condition),
                        pre_mean,
                        post_mean,
                        percentile_mean,
                    )
                )
                continue

            gt_fids_t = torch.tensor(gt_fids, dtype=torch.int64, device=device)

            # One matmul over all GT features (P1.2 vectorization).
            # gt_fids is guaranteed non-empty by the raise at the top of the
            # outer loop, so the empty case is not handled here.
            pre_probes = compute_raw_probe_batched(
                act, probe_model.sae.W_enc, probe_model.sae.b_enc,
                probe_model.sae.b_dec, gt_fids_t,
                apply_b_dec=probe_model.apply_b_dec,
            )  # (seq_len, n_gt)
            pre_vals_t = max_pool_masked_batched(pre_probes, mask)  # (n_gt,)

            post_probes = apply_sae_activation_batched(
                pre_probes, probe_model.sae, gt_fids_t,
            )  # (seq_len, n_gt)
            post_vals_t = max_pool_masked_batched(post_probes, mask)  # (n_gt,)

            # All probe values must be finite. Non-finite (NaN/inf) is a real
            # bug — under realistic SAE semantics with finite inputs and a
            # ReLU/JumpReLU activation, neither pre_val nor post_val can be
            # non-finite. If we see non-finite values here, something upstream
            # produced bad activations (model bug, NaN propagation, etc.) and
            # the loud failure mode is the right default.
            if not bool(torch.isfinite(pre_vals_t).all()):
                raise ValueError(
                    f"Non-finite pre_val in probe for doc={comp.doc_idx} "
                    f"target_seg={comp.target_segment_idx}: pre_vals={pre_vals_t.tolist()}, "
                    f"gt_fids={gt_fids}. Check the SAE encoder weights and the "
                    f"activations from the trimmed base model for NaN/inf propagation."
                )
            if not bool(torch.isfinite(post_vals_t).all()):
                raise ValueError(
                    f"Non-finite post_val in probe for doc={comp.doc_idx} "
                    f"target_seg={comp.target_segment_idx}: post_vals={post_vals_t.tolist()}, "
                    f"gt_fids={gt_fids}. Check the SAE activation function for "
                    f"unexpected NaN/inf production."
                )

            pre_activations: list[float] = pre_vals_t.tolist()
            post_activations: list[float] = post_vals_t.tolist()
            probed_fids: list[int] = list(gt_fids)

            if pre_activations:
                pre_mean = statistics.mean(pre_activations)
                post_mean = statistics.mean(post_activations) if post_activations else 0.0

                if percentile_table is not None and probed_fids:
                    fid_t = torch.tensor(probed_fids, dtype=torch.int64)
                    post_t = torch.tensor(post_activations, dtype=torch.float32)
                    pct_mean = magnitude_to_percentile(
                        fid_t, post_t, percentile_table
                    ).mean().item()
                else:
                    pct_mean = float("nan")

                scores.append((str(sample.condition), pre_mean, post_mean, pct_mean))

    return scores


def _condition_means(
    by_condition: dict[str, list[float]],
) -> tuple[float, float]:
    """Compute per-condition means: (real, random).

    Returns ``0.0`` for any missing or empty condition. The empty case can
    arise after the symmetric-NaN filter in :func:`_aggregate_roundtrip_metrics`
    drops every value for a condition; treating that as a 0.0 mean preserves
    the W&B-friendly behavior (avoid NaN poisoning) while keeping
    ``steering/roundtrip_n_samples`` informative for the consumer.
    """
    def _mean_or_zero(key: str) -> float:
        vals = by_condition.get(key)
        if not vals:
            return 0.0
        return statistics.mean(vals)

    return _mean_or_zero("real"), _mean_or_zero("random")


def _aggregate_roundtrip_metrics(
    raw_scores: list[tuple[str, float, float, float]],
    total_gt_features: int = 0,
    n_docs: int = 0,
) -> dict[str, float]:
    """Aggregate raw roundtrip probe scores into wandb metrics.

    Each score is ``(condition, pre_act_value, post_act_value, pct_value)``.
    Produces pre-activation, post-activation, and percentile metric sets.
    Percentile metrics (``_pct`` suffix) are the primary reported metrics;
    pre- and post-activation variants are also logged for diagnostic use.

    Mixed NaN handling (map B-13 fix): the pre-, post-, and percentile paths
    are filtered SYMMETRICALLY — non-finite values are dropped before the
    per-condition mean is taken. Without this, a single non-finite pre/post
    value would propagate through ``statistics.mean`` and yield NaN for the
    whole condition (silently breaking W&B logging). The percentile path
    already filtered; this commit applies the same filter to pre/post.
    """
    pre_by_cond: dict[str, list[float]] = {}
    post_by_cond: dict[str, list[float]] = {}
    pct_by_cond: dict[str, list[float]] = {}
    for condition, pre_val, post_val, pct_val in raw_scores:
        if math.isfinite(pre_val):
            pre_by_cond.setdefault(condition, []).append(pre_val)
        if math.isfinite(post_val):
            post_by_cond.setdefault(condition, []).append(post_val)
        if math.isfinite(pct_val):
            pct_by_cond.setdefault(condition, []).append(pct_val)

    pre_real, pre_random = _condition_means(pre_by_cond)
    post_real, post_random = _condition_means(post_by_cond)
    pct_real, pct_random = _condition_means(pct_by_cond)

    return {
        # Pre-activation
        "steering/roundtrip_real_mean": pre_real,
        "steering/roundtrip_random_mean": pre_random,
        "steering/roundtrip_specificity": pre_real - pre_random,
        # Post-activation
        "steering/roundtrip_real_mean_post_act": post_real,
        "steering/roundtrip_random_mean_post_act": post_random,
        "steering/roundtrip_specificity_post_act": post_real - post_random,
        # Percentile (primary: post-activation values mapped to [0.5, 99.5])
        "steering/roundtrip_real_mean_pct": pct_real,
        "steering/roundtrip_random_mean_pct": pct_random,
        "steering/roundtrip_specificity_pct": pct_real - pct_random,
        # Shared
        "steering/roundtrip_n_samples": len(raw_scores),
        "steering/roundtrip_n_gt_features_avg": total_gt_features / n_docs if n_docs > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# RNG state preservation
# ---------------------------------------------------------------------------


def save_rng_state(device: torch.device) -> dict:
    """Save CPU and CUDA RNG state for restoration after eval."""
    state: dict = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict, device: torch.device) -> None:
    """Restore CPU and CUDA RNG state after eval."""
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_steering(
    model,
    feature_dir,
    tokenizer,
    device,
    boundaries,
    threshold_cfg,
    precomputed_thresholds,
    cfg: SteeringEvalConfig | None = None,
    data_dir: str | None = None,
    feature_index: FeatureIndex | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    probe_model: _ProbeModel | None = None,
    percentile_table: torch.Tensor | None = None,
    eval_cache: InlineEvalCache | None = None,
    meta: dict | None = None,
    feature_mask: FeatureMask | None = None,
):
    """Run inline roundtrip steering eval. Returns flat dict of metrics.

    Pipeline: select random docs -> generate (real/random)
    -> SAE roundtrip probe -> aggregate metrics.

    When ``eval_cache`` is passed, the deterministic doc selection and val
    parquet read are skipped — their results come from the cache. Pass
    ``None`` (the default) to run the legacy compute-inline path.

    Canonical specificity definition
    --------------------------------
    This evaluator is the canonical emitter of the "specificity" metric
    across the four steering evaluators in the codebase. The contract is
    bit-exactly:

        specificity = real_probe_score − random_doc_random_seg_probe_score

    Both terms are computed on the same probe with the same N samples.
    The ``real`` condition uses features from the held-out document's
    target segment. The ``random`` condition uses features drawn
    independently: a random held-out document (``random_doc``) and, within
    that document, a random segment (``random_seg``) — the two random draws
    are fully independent from the real target. Output keys:
    ``steering/roundtrip_specificity`` (pre-activation),
    ``steering/roundtrip_specificity_post_act``,
    ``steering/roundtrip_specificity_pct``.
    """
    if cfg is None:
        cfg = SteeringEvalConfig()
    assert not model.training, "Model must be in eval mode for steering eval"

    clustered_payload = (
        eval_cache.clustered_payload if eval_cache is not None else None
    )
    if clustered_payload is None:
        reject_feature_clusters_metadata(meta, "inline_steering_eval checkpoint")

    t0 = time.monotonic()
    feature_dir = Path(feature_dir)

    # Select random docs with post-threshold features (cache-aware)
    if eval_cache is None:
        selected_docs, selection_metrics_src = _select_random_docs(
            feature_dir=feature_dir,
            threshold_cfg=threshold_cfg,
            precomputed_thresholds=precomputed_thresholds,
            n_eval_docs=cfg.n_eval_docs,
            seed=cfg.seed,
            feature_mask=feature_mask,
        )
    else:
        selected_docs = eval_cache.selected_docs
        selection_metrics_src = eval_cache.selection_metrics

    # ALWAYS copy before mutating — a cached selection_metrics dict must
    # never be poisoned across calls. This copy is O(2) keys.
    scan_metrics: dict[str, int | float] = dict(selection_metrics_src)
    scan_metrics["steering/eval_cache_hit"] = 1 if eval_cache is not None else 0

    if not selected_docs:
        scan_metrics["steering/eval_time_seconds"] = time.monotonic() - t0
        return scan_metrics

    # Load document texts (cache-aware)
    if eval_cache is None:
        doc_texts = _load_doc_texts(data_dir)
    else:
        doc_texts = eval_cache.doc_texts

    # The legacy branch resolves raw validation features here. The clustered
    # branch was joined and certified at startup and must not re-read or
    # re-threshold that source during a cadence trigger.
    val_shard_stem: str | None = None
    if clustered_payload is None:
        if feature_index is None:
            feature_index = FeatureIndex(str(feature_dir))
        val_shard = _find_val_shard(feature_dir)
        if val_shard is None:
            scan_metrics["steering/eval_time_seconds"] = time.monotonic() - t0
            return scan_metrics
        val_shard_stem = _shard_stem(val_shard)

    # Partition across DDP ranks
    my_docs = _partition_docs(selected_docs, ddp_rank, ddp_world_size)

    if clustered_payload is None:
        # Literal legacy generation/probe behavior: no extra status collective
        # or changed exception ordering on an unclustered run.
        assert feature_index is not None
        assert val_shard_stem is not None
        local_completions = _generate_roundtrip_samples(
            model,
            my_docs,
            doc_texts,
            feature_index,
            val_shard_stem,
            tokenizer,
            boundaries,
            threshold_cfg,
            precomputed_thresholds,
            device,
            cfg,
            feature_mask=feature_mask,
        )
        if probe_model is not None:
            local_scores = _probe_roundtrip_raw(
                local_completions,
                probe_model,
                percentile_table,
            )
        else:
            local_scores = []
            logger.warning("No probe_model — skipping roundtrip probe metrics")
    else:
        # Every initialized rank participates in one status gather before any
        # result gather. Ordinary clustered Python/data/model failures therefore
        # become symmetric instead of leaving peers blocked in a later collective.
        local_completions = []
        local_scores = []
        local_error: Exception | None = None
        try:
            local_completions = _generate_clustered_roundtrip_samples(
                model,
                my_docs,
                doc_texts,
                clustered_payload,
                tokenizer,
                device,
                cfg,
            )
            if probe_model is not None:
                local_scores = _probe_roundtrip_raw(
                    local_completions,
                    probe_model,
                    percentile_table,
                )
            else:
                logger.warning("No probe_model — skipping roundtrip probe metrics")
        except Exception as exc:
            local_error = exc

        local_status: dict[str, Any] = {
            "ok": local_error is None,
            "rank": ddp_rank,
            "error_type": None if local_error is None else type(local_error).__name__,
            "error": None if local_error is None else str(local_error),
        }
        gathered_statuses: list[dict[str, Any] | None]
        if ddp_world_size > 1 and dist.is_initialized():
            gathered_statuses = [None] * ddp_world_size
            dist.all_gather_object(gathered_statuses, local_status)
        else:
            gathered_statuses = [local_status]
        failures = [
            status
            for status in gathered_statuses
            if status is not None and not bool(status.get("ok"))
        ]
        if failures:
            if ddp_world_size <= 1 or not dist.is_initialized():
                assert local_error is not None
                raise local_error
            summary = "; ".join(
                f"rank {status['rank']}: {status['error_type']}: {status['error']}"
                for status in failures
            )
            raise RuntimeError(
                "inline steering evaluation failed on one or more ranks: " + summary
            )

    local_n = len(local_completions)
    local_gt = sum(len(c.ground_truth_feature_ids) for c in local_completions)
    local_members = sum(
        len(group)
        for completion in local_completions
        for group in (completion.ground_truth_member_ids or [])
    )

    # Gather across ranks
    if ddp_world_size > 1 and dist.is_initialized():
        gathered: list[list[tuple[str, float, float, float]] | None] = [None] * ddp_world_size
        dist.all_gather_object(gathered, local_scores)
        all_scores = [s for sub in gathered if sub is not None for s in sub]
    else:
        all_scores = local_scores

    total_n = _gather_int(local_n, ddp_world_size, device)
    total_gt = _gather_int(local_gt, ddp_world_size, device)

    probe_metrics = _aggregate_roundtrip_metrics(all_scores, total_gt, total_n)
    clustered_metrics: dict[str, str | float] = {}
    if clustered_payload is not None:
        total_members = _gather_int(local_members, ddp_world_size, device)
        clustered_metrics = {
            "steering/cluster_probe_semantics": CLUSTERED_PROBE_SEMANTICS,
            "steering/roundtrip_cluster_member_group_size_avg": (
                total_members / total_gt if total_gt > 0 else 0.0
            ),
        }

    elapsed = time.monotonic() - t0
    return {
        **scan_metrics,
        **probe_metrics,
        **clustered_metrics,
        "steering/n_completions_generated": total_n,
        "steering/eval_time_seconds": elapsed,
    }
