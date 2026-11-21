# steering/eval/multiseg_generator.py
"""Multi-segment conditioning builder and generator for completion steering eval.

Builds encoder inputs (feature IDs, magnitudes, mask, seg_map) that match
the training dataloader's character-level segment assignment, and generates
text continuations with KV-cache autoregressive decoding.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype
from phantom.sized import NonEmptyStr

from nanochat.engine import KVCache
from nanochat.sampling import sample_next_token
from steering.conditional.discretization import discretize_activations
from steering.conditional.encoder_decoder.packed_encoder import PackedEncoderOutput
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfConditionedFeatureProvider,
    SelfCondState,
    final_real_prompt_source_mask,
    self_conditioned_decode_step,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.sampling import SamplingBundle, sample_features
from steering.conditional.thresholding import (
    ThresholdConfig,
    apply_threshold,
)
from steering.eval.completion_steering_eval import EvalMode
from steering.eval.segment_mode_guard import require_fixed_segment_mode
from steering.refined_types import (
    NonEmptyNonNegativeIntListLike,
    NonEmptyNonNegativeIntTupleLike,
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    NonNegativeIntListLike,
    NonNegativeIntMappingLike,
    PositiveIntLike,
    StrictIntLike,
)
from steering.segmentation import (
    align_char_ranges_to_decoded_prefix,
    assign_tokens_to_segments,
    strict_segment_bounds_for_assignment,
    token_to_char_offsets,
)

_COMPLETION_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.completion_eval"
)

if TYPE_CHECKING:
    _CompletionSampleList = list["CompletionSample"]
    _GeneratedSamples = list[list[int]]
    _GeneratedTokenList = list[int]
    _KVCacheLike = KVCache
    _SegmentFeatureList = list[SegmentFeatures]
    _SegmentFeatureTuple = tuple[SegmentFeatures, ...]
else:
    # Domain values own their element invariants. Unsubscripted runtime
    # containers avoid beartype's global-RNG-backed element sampling.
    _CompletionSampleList = list
    _GeneratedSamples = list
    _GeneratedTokenList = list
    _KVCacheLike = object
    _SegmentFeatureList = list
    _SegmentFeatureTuple = tuple

# ---------------------------------------------------------------------------
# Data structures for generation outputs
# ---------------------------------------------------------------------------


class CompletionCondition(StrEnum):
    """Fixed completion-generation condition names."""

    REAL = "real"
    ABLATED = "ablated"
    SHUFFLED = "shuffled"
    RANDOM = "random"


class AddedBucketCondition(str):
    """Dynamic completion condition for insertion bucket sweeps."""

    prefix = "added_bucket_"
    bucket_index: int

    @beartype
    def __new__(cls, bucket_index: NonNegativeIntLike):
        obj = str.__new__(cls, f"{cls.prefix}{bucket_index}")
        obj.bucket_index = bucket_index
        return obj

    @classmethod
    def parse(cls, value: str) -> AddedBucketCondition | None:
        if not value.startswith(cls.prefix):
            return None
        suffix = value[len(cls.prefix) :]
        if not suffix.isdigit():
            return None
        return cls(int(suffix))


CompletionConditionLabel = CompletionCondition | AddedBucketCondition


def coerce_completion_condition(
    value: CompletionConditionLabel | str,
) -> CompletionConditionLabel:
    """Normalize a completion condition and reject unknown string labels."""

    if isinstance(value, (CompletionCondition, AddedBucketCondition)):
        return value
    try:
        return CompletionCondition(value)
    except ValueError:
        pass
    parsed = AddedBucketCondition.parse(value)
    if parsed is not None:
        return parsed
    raise ValueError(
        f"Unknown completion condition {value!r}; expected one of "
        f"{[condition.value for condition in CompletionCondition]} or "
        f"{AddedBucketCondition.prefix}<non-negative-int>."
    )


@beartype
def added_bucket_condition(bucket_index: NonNegativeIntLike) -> AddedBucketCondition:
    """Return the dynamic label for an added-feature bucket condition."""

    return AddedBucketCondition(bucket_index)


@dataclass(init=False)
class CompletionSample:
    """A single generated text sample under one condition."""

    condition: CompletionConditionLabel
    text: str
    seed: int
    bucket_index: int | None = None

    @beartype
    def __init__(
        self,
        condition: CompletionConditionLabel | str,
        text: str,
        seed: StrictIntLike,
        bucket_index: NonNegativeIntLike | None = None,
    ) -> None:
        condition = coerce_completion_condition(condition)
        self.condition = condition
        self.text = text
        self.seed = seed
        self.bucket_index = bucket_index
        if isinstance(condition, AddedBucketCondition):
            if self.bucket_index is None:
                self.bucket_index = condition.bucket_index
            elif self.bucket_index != condition.bucket_index:
                raise ValueError(
                    "CompletionSample bucket_index does not match condition: "
                    f"condition={condition!r}, bucket_index={self.bucket_index!r}"
                )
        elif self.bucket_index is not None:
            raise ValueError(
                "CompletionSample.bucket_index is only valid for "
                f"{AddedBucketCondition.prefix}<N> conditions; got {condition!r}"
            )


@beartype
@dataclass(frozen=True)
class GroundTruthFeatureSignals:
    """Post-threshold, discretized feature signals for the ground truth segment."""

    feature_signals: NonNegativeIntMappingLike  # {feature_id: bucket_index}
    n_features_active: NonNegativeIntLike
    avg_magnitude: NonNegativeFiniteNumberLike
    max_magnitude: NonNegativeIntLike
    min_magnitude: NonNegativeIntLike
    feature_of_interest_magnitude: NonNegativeIntLike | None


@beartype
def build_ground_truth_signals(
    prepared_segment: SegmentFeatures,
    feature_id: NonNegativeIntLike,
    mode: EvalMode,
) -> GroundTruthFeatureSignals:
    """Build ground truth signals from a prepared (post-threshold, discretized) segment.

    Args:
        prepared_segment: A SegmentFeatures whose feature_mags are int64 bucket indices
            (output of :func:`prepare_segments`).
        feature_id: The feature being tested in this evaluation.
        mode: ``EvalMode.ABLATION`` — feature must be present in the segment;
              ``EvalMode.INSERTION`` — feature must NOT be present in the segment.

    Returns:
        :class:`GroundTruthFeatureSignals` with the signal dict and summary stats.

    Raises:
        AssertionError: If the feature presence invariant for the given mode is violated.
    """
    fids = prepared_segment.feature_ids.tolist()
    mags = prepared_segment.feature_mags.tolist()

    feature_signals = {int(fid): int(mag) for fid, mag in zip(fids, mags, strict=True)}
    n_active = len(feature_signals)

    # Validate feature-of-interest presence matches mode invariant
    feature_present = feature_id in feature_signals
    if mode is EvalMode.ABLATION:
        assert feature_present, (
            f"Ablation mode: feature {feature_id} must be present in target segment "
            f"(post-threshold feature_ids: {sorted(feature_signals.keys())})"
        )
    else:
        assert not feature_present, (
            f"Insertion mode: feature {feature_id} must NOT be present in target segment "
            f"(post-threshold feature_ids: {sorted(feature_signals.keys())})"
        )

    if n_active == 0:
        return GroundTruthFeatureSignals(
            feature_signals={},
            n_features_active=0,
            avg_magnitude=0.0,
            max_magnitude=0,
            min_magnitude=0,
            feature_of_interest_magnitude=None,
        )

    mag_values = list(feature_signals.values())
    avg_mag = sum(mag_values) / n_active
    max_mag = max(mag_values)
    min_mag = min(mag_values)
    foi_mag = feature_signals.get(feature_id)

    return GroundTruthFeatureSignals(
        feature_signals=feature_signals,
        n_features_active=n_active,
        avg_magnitude=avg_mag,
        max_magnitude=max_mag,
        min_magnitude=min_mag,
        feature_of_interest_magnitude=foi_mag,
    )


@beartype
@dataclass
class DocumentCompletion:
    """All generated samples for a single document + feature combination."""

    doc_idx: NonNegativeIntLike
    feature_id: NonNegativeIntLike
    pre_segment_text: str
    ground_truth_segment: str
    target_segment_idx: NonNegativeIntLike
    samples: _CompletionSampleList = field(default_factory=list)
    ground_truth_signals: GroundTruthFeatureSignals | None = None


@beartype
def ablate_feature(
    segments: _SegmentFeatureList,
    target_seg_idx: NonNegativeIntLike,
    feature_id: NonNegativeIntLike,
) -> _SegmentFeatureList:
    """Return a copy of segments with feature_id removed from the target segment."""
    result = list(segments)
    target = segments[target_seg_idx]

    mask = target.feature_ids != feature_id
    if mask.all():
        raise ValueError(
            f"Feature {feature_id} not present in segment {target_seg_idx} "
            f"(filtered by top-k thresholding)"
        )

    result[target_seg_idx] = SegmentFeatures(
        segment_idx=target.segment_idx,
        position=target.position,
        feature_ids=target.feature_ids[mask].clone(),
        feature_mags=target.feature_mags[mask].clone(),
        char_start=target.char_start,
        char_end=target.char_end,
    )
    return result


@beartype
def add_feature(
    segments: _SegmentFeatureList,
    target_seg_idx: NonNegativeIntLike,
    feature_id: NonNegativeIntLike,
    bucket_index: NonNegativeIntLike,
) -> _SegmentFeatureList:
    """Return a copy of segments with feature_id added to the target segment."""
    result = list(segments)
    target = segments[target_seg_idx]

    if (target.feature_ids == feature_id).any():
        raise ValueError(
            f"Feature {feature_id} already present in segment {target_seg_idx}"
        )

    fids = target.feature_ids
    mags = target.feature_mags

    new_fid = t.tensor([feature_id], dtype=fids.dtype)
    new_mag = t.tensor([bucket_index], dtype=mags.dtype)

    result[target_seg_idx] = SegmentFeatures(
        segment_idx=target.segment_idx,
        position=target.position,
        feature_ids=t.cat([fids, new_fid]),
        feature_mags=t.cat([mags, new_mag]),
        char_start=target.char_start,
        char_end=target.char_end,
    )
    return result


def prepare_segments(
    segments: list[SegmentFeatures],
    boundaries: t.Tensor,  # (n_features, n_boundaries) float64
    feature_cfg: ThresholdConfig | SamplingBundle | None = None,
    precomputed_thresholds: t.Tensor | None = None,
    feature_mask: FeatureMask | None = None,
) -> list[SegmentFeatures]:
    """Apply feature selection and discretization to raw SegmentFeatures.

    Matches the training pipeline: threshold or sample features, then map
    activation magnitudes to discrete bucket indices via per-feature boundaries.

    Args:
        segments: Raw SegmentFeatures with float32 magnitudes.
        boundaries: Per-feature bucket boundaries, shape (n_features, n_boundaries).
        feature_cfg: ThresholdConfig for deterministic thresholding,
            SamplingBundle for stochastic sampling, or None to skip
            feature selection and only discretize.
        precomputed_thresholds: Precomputed per-feature threshold tensor
            (only used with ThresholdConfig).
    Returns:
        New SegmentFeatures list with feature_mags replaced by int64 bucket indices.
    """
    result: list[SegmentFeatures] = []
    for seg in segments:
        fids = seg.feature_ids.clone()
        mags = seg.feature_mags.clone()
        if feature_mask is not None and len(fids) > 0:
            fids, mags = feature_mask.apply_to_positive(fids, mags)

        if feature_cfg is None:
            pass  # skip thresholding/sampling, just discretize
        elif isinstance(feature_cfg, ThresholdConfig):
            fids, mags = apply_threshold(
                feature_cfg, fids, mags, precomputed=precomputed_thresholds
            )
        elif isinstance(feature_cfg, SamplingBundle):
            fids, mags = sample_features(
                fids,
                mags,
                feature_cfg.percentile_table,
                feature_cfg.selection,
                feature_cfg.rng,
            )
        else:
            raise TypeError(
                f"Unsupported feature_cfg type: {type(feature_cfg).__name__}"
            )
        if feature_mask is not None:
            feature_mask.assert_no_masked(fids)

        buckets = discretize_activations(fids, mags, boundaries)

        result.append(
            SegmentFeatures(
                segment_idx=seg.segment_idx,
                position=seg.position,
                feature_ids=fids,
                feature_mags=buckets.to(t.int64),
                char_start=seg.char_start,
                char_end=seg.char_end,
            )
        )
    return result


@dataclass(frozen=True)
class MultisegInputs:
    """Batched encoder inputs and segment map for multi-segment generation."""

    enc_feature_ids: t.Tensor  # (1, n_seg, F_max) int64
    enc_magnitudes: t.Tensor  # (1, n_seg, F_max) int64
    enc_mask: t.Tensor  # (1, n_seg, F_max) bool
    seg_map: t.Tensor  # (1, T) int64


def _segment_bounds_for_assignment(
    segments: _SegmentFeatureList,
    token_starts: t.Tensor,
    token_ends: t.Tensor,
    allow_zero_width_segment_idx: NonNegativeIntLike | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Return strict segment bounds for token-to-segment assignment."""
    seg_char_starts = t.tensor(
        [s.char_start for s in segments],
        dtype=t.int64,
    )
    seg_char_ends = t.tensor(
        [s.char_end for s in segments],
        dtype=t.int64,
    )
    return strict_segment_bounds_for_assignment(
        seg_char_starts=seg_char_starts,
        seg_char_ends=seg_char_ends,
        token_starts=token_starts,
        token_ends=token_ends,
        allow_zero_width_segment_idx=allow_zero_width_segment_idx,
    )


@beartype
def build_multiseg_inputs(
    segments: _SegmentFeatureList,
    prefix_tokens: NonNegativeIntListLike,
    n_gen_tokens: NonNegativeIntLike,
    target_segment_idx: NonNegativeIntLike,
    tokenizer,
    next_segment_conditioning_offset: bool = False,
) -> MultisegInputs:
    """Build encoder inputs and seg_map for multi-segment generation.

    Uses character-level overlap assignment to map nanochat-tokenizer positions
    to SAE-tokenizer segments, matching the training dataloader's
    ``_build_row_segments_char_level``.

    In current-segment mode, generated tokens attend to the same
    source-coordinate slot as ``target_segment_idx``. In next-segment mode,
    encoder slot s contains source segment s+1, so generated target-segment
    tokens attend to slot ``target_segment_idx - 1``.

    Args:
        segments: Prepared SegmentFeatures with char_start/char_end boundaries.
        prefix_tokens: Nanochat token IDs for the prompt (including BOS).
        n_gen_tokens: Number of tokens to generate (mapped to
            *target_segment_idx*).
        target_segment_idx: Segment index for generated tokens.
        tokenizer: Nanochat tokenizer (must support ``decode``).
        next_segment_conditioning_offset: If True, decoder segment s attends
            to feature signals from source segment s + 1.
    """
    if not segments:
        raise ValueError("segments must contain at least one segment")
    if target_segment_idx < 0 or target_segment_idx >= len(segments):
        raise ValueError(
            f"target_segment_idx={target_segment_idx} out of range for "
            f"{len(segments)} segments"
        )
    if next_segment_conditioning_offset and target_segment_idx == 0:
        raise ValueError(
            "next_segment_conditioning_offset=True requires target_segment_idx >= 1"
        )

    n_source_seg = len(segments)
    n_encoder_seg = n_source_seg - 1 if next_segment_conditioning_offset else n_source_seg
    n_prefix = len(prefix_tokens)
    T = n_prefix + n_gen_tokens

    # Build seg_map using character-level overlap assignment
    seg_map = t.zeros(1, T, dtype=t.int64)

    if n_prefix > 0:
        char_offsets = token_to_char_offsets(tokenizer, prefix_tokens)
        token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
        token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

        seg_char_starts, seg_char_ends = _segment_bounds_for_assignment(
            segments,
            token_starts,
            token_ends,
            allow_zero_width_segment_idx=target_segment_idx,
        )

        # Assign each prefix token to the segment with maximum char overlap
        assignments = assign_tokens_to_segments(
            token_starts,
            token_ends,
            seg_char_starts,
            seg_char_ends,
        )
        # Enforce monotonicity (matching training's cummax)
        assignments = t.cummax(assignments, dim=0).values

        if next_segment_conditioning_offset:
            assignments = assignments.clamp(max=n_encoder_seg - 1)

        seg_map[0, :n_prefix] = assignments

    gen_seg = (
        target_segment_idx - 1
        if next_segment_conditioning_offset
        else target_segment_idx
    )
    seg_map[0, n_prefix:] = gen_seg

    # Collect per-segment features
    per_seg_fids: list[t.Tensor] = []
    per_seg_mags: list[t.Tensor] = []
    if next_segment_conditioning_offset:
        source_segments = segments[1:]
    else:
        source_segments = segments
    for seg in source_segments:
        per_seg_fids.append(seg.feature_ids.to(t.int64))
        per_seg_mags.append(seg.feature_mags.to(t.int64))

    # Pad to uniform F_max
    F_max = max((len(f) for f in per_seg_fids), default=1)
    F_max = max(F_max, 1)

    enc_feature_ids = t.zeros(1, n_encoder_seg, F_max, dtype=t.int64)
    enc_magnitudes = t.zeros(1, n_encoder_seg, F_max, dtype=t.int64)
    enc_mask = t.zeros(1, n_encoder_seg, F_max, dtype=t.bool)

    for s_idx in range(n_encoder_seg):
        n_f = len(per_seg_fids[s_idx])
        if n_f > 0:
            enc_feature_ids[0, s_idx, :n_f] = per_seg_fids[s_idx]
            enc_magnitudes[0, s_idx, :n_f] = per_seg_mags[s_idx]
            enc_mask[0, s_idx, :n_f] = True

    return MultisegInputs(
        enc_feature_ids=enc_feature_ids,
        enc_magnitudes=enc_magnitudes,
        enc_mask=enc_mask,
        seg_map=seg_map,
    )


@beartype
def _prefill_condition(
    model,  # EncoderDecoderGPT
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    encoder_out: PackedEncoderOutput,
    seg_map: t.Tensor,
    kv_cache: _KVCacheLike,
    device: t.device,
) -> t.Tensor:
    """Run the prefill forward pass, populating *kv_cache* in-place.

    Args:
        model: EncoderDecoderGPT instance.
        prefix_tokens: List of int token IDs for the prompt.
        encoder_out: Precomputed packed encoder state.
        seg_map: Full seg_map tensor on device, shape (1, T).
        kv_cache: Empty KV cache to fill with prefill state.
        device: Target device.

    Returns:
        last_logits: Logits at the last prefix position, shape (1, vocab_size).
    """
    n_prefix = len(prefix_tokens)
    prefix_ids = t.tensor(
        [prefix_tokens], dtype=t.int64, device=device
    )  # (1, n_prefix)
    seg_map_prefill = seg_map[:, :n_prefix]

    logits = model(
        prefix_ids,
        targets=None,
        seg_map=seg_map_prefill,
        kv_cache=kv_cache,
        encoder_out=encoder_out,
        encoder_strict=True,
    )
    # logits shape: (1, n_prefix, vocab_size); take last position
    return logits[:, -1, :]  # (1, vocab_size)


@beartype
def _decode_from_prefill(
    model,  # EncoderDecoderGPT
    kv_cache: _KVCacheLike,
    last_logits: t.Tensor,
    encoder_out: PackedEncoderOutput,
    seg_map: t.Tensor,
    n_prefix: PositiveIntLike,
    max_new_tokens: NonNegativeIntLike,
    total_len: PositiveIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
    cancel_event: threading.Event | None = None,
) -> Iterator[int]:
    """Yield tokens one at a time from an already-prefilled KV cache.

    Args:
        model: EncoderDecoderGPT instance.
        kv_cache: KV cache already populated by _prefill_condition.
        last_logits: Logits from the last prefill position, shape (1, vocab_size).
        encoder_out: Precomputed packed encoder state.
        seg_map: Full seg_map tensor on device, shape (1, T).
        n_prefix: Number of prefix tokens that were prefilled.
        max_new_tokens: Maximum number of tokens to generate.
        total_len: Total sequence length (prefix + max_new_tokens).
        device: Target device.
        temperature: Sampling temperature; 0.0 for greedy.
        top_k: Top-k sampling cutoff; None for full vocab.
        seed: Random seed for the torch Generator.
        bos_id: Token ID that signals end of generation.
        cancel_event: If set between tokens, stops generation early.

    Yields:
        Token IDs one at a time (not including the stop token).
    """
    # Defensive: ``seg_map`` must cover the entire decode horizon. A caller
    # that plumbs ``max_new_tokens`` larger than what
    # ``build_multiseg_inputs(n_gen_tokens=...)`` was sized for would silently
    # produce a ``(1, 0)`` tensor at ``seg_map[:, cur_pos-1:cur_pos]`` once
    # ``cur_pos > seg_map.shape[1]``, which the decoder would then attempt to
    # forward against — see map SB-13-P0-06. Hard-fail here for clearer
    # attribution.
    if seg_map.shape[1] < total_len:
        raise ValueError(
            f"seg_map covers {seg_map.shape[1]} tokens but decode requires "
            f"total_len={total_len} (n_prefix={n_prefix} + "
            f"max_new_tokens={max_new_tokens}). Rebuild encoder_inputs with "
            "a matching n_gen_tokens budget."
        )
    rng = t.Generator(device=device)
    rng.manual_seed(seed)

    cur_pos = n_prefix

    for _ in range(max_new_tokens):
        if cancel_event is not None and cancel_event.is_set():
            return

        next_token = sample_next_token(last_logits, rng, temperature, top_k)  # (1, 1)
        token_id = int(next_token[0, 0].item())

        if token_id == bos_id:
            return

        yield token_id
        cur_pos += 1

        if cur_pos >= total_len:
            return

        # Decode one token
        token_ids = next_token  # (1, 1)
        seg_map_step = seg_map[:, cur_pos - 1 : cur_pos]

        logits = model(
            token_ids,
            targets=None,
            seg_map=seg_map_step,
            kv_cache=kv_cache,
            encoder_out=encoder_out,
            encoder_strict=True,
        )
        last_logits = logits[:, -1, :]  # (1, vocab_size)


@t.inference_mode()
@beartype
def generate_completion(
    model,  # EncoderDecoderGPT
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    encoder_inputs: MultisegInputs,
    max_new_tokens: NonNegativeIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
    precomputed_encoder: PackedEncoderOutput | None = None,
) -> _GeneratedTokenList:
    """Generate a text continuation conditioned on encoder inputs.

    Precomputes the encoder output once, then prefills the decoder with
    prefix tokens using KV-cache, and autoregressively decodes up to
    max_new_tokens tokens (stopping early on bos_id).

    Args:
        model: EncoderDecoderGPT instance.
        prefix_tokens: List of int token IDs for the prompt.
        encoder_inputs: MultisegInputs with enc_feature_ids, enc_magnitudes, enc_mask, seg_map.
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device.
        temperature: Sampling temperature; 0.0 for greedy.
        top_k: Top-k sampling cutoff; None for full vocab.
        seed: Random seed for the torch Generator.
        bos_id: Token ID that signals end of generation.
        precomputed_encoder: Optional packed encoder state. When provided, the
            encoder forward pass is skipped. Useful when generating multiple
            samples for the same condition.

    Returns:
        List of generated token IDs (not including the stop token).
    """
    dec_config = model.dec_config
    max_seq_len = dec_config.sequence_len

    # Reject prefixes that would overrun the context window. Sister functions
    # ``generate_for_document`` and ``generate_samples_for_condition`` raise
    # explicitly on overflow; previous behaviour here silently left-truncated
    # the prefix (which dropped BOS and broke ``token_to_char_offsets``
    # alignment). See map SB-13-P0-05 / manifest §7. Hard-fail like the
    # sisters so callers see the budget violation at submission rather than
    # debugging garbled completions later.
    if len(prefix_tokens) > max_seq_len - max_new_tokens:
        raise ValueError(
            f"prefix_tokens length ({len(prefix_tokens)}) exceeds "
            f"max_seq_len ({max_seq_len}) - max_new_tokens ({max_new_tokens}); "
            "filter the document/segment during selection rather than "
            "silently truncating (BOS would be dropped, breaking "
            "char-offset alignment)."
        )
    n_prefix = len(prefix_tokens)
    total_len = n_prefix + max_new_tokens

    # seg_map is always needed on device (not part of encoder output)
    seg_map = encoder_inputs.seg_map.to(device)

    # Use precomputed encoder output if provided, otherwise compute it
    if precomputed_encoder is not None:
        encoder_out = precomputed_encoder
    else:
        enc_fids = encoder_inputs.enc_feature_ids.to(device)
        enc_mags = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

    # Choose dtype: bfloat16 on CUDA, float32 otherwise
    dtype = t.bfloat16 if device.type == "cuda" else t.float32

    # Allocate KV cache for full sequence
    kv_cache = KVCache(
        batch_size=1,
        num_heads=dec_config.n_kv_head,
        seq_len=total_len,
        head_dim=dec_config.n_embd // dec_config.n_head,
        num_layers=dec_config.n_layer,
        device=device,
        dtype=dtype,
    )

    # Prefill: pass all prefix tokens at once
    last_logits = _prefill_condition(
        model=model,
        prefix_tokens=prefix_tokens,
        encoder_out=encoder_out,
        seg_map=seg_map,
        kv_cache=kv_cache,
        device=device,
    )

    # Autoregressive decode from the prefilled cache
    return list(
        _decode_from_prefill(
            model=model,
            kv_cache=kv_cache,
            last_logits=last_logits,
            encoder_out=encoder_out,
            seg_map=seg_map,
            n_prefix=n_prefix,
            max_new_tokens=max_new_tokens,
            total_len=total_len,
            device=device,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            bos_id=bos_id,
        )
    )


@t.inference_mode()
@beartype
def generate_completion_stream(
    model,  # EncoderDecoderGPT
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    encoder_inputs: MultisegInputs,
    max_new_tokens: NonNegativeIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
    cancel_event: threading.Event | None = None,
    precomputed_encoder: PackedEncoderOutput | None = None,
) -> Iterator[int]:
    """Yield token IDs one at a time for streaming generation.

    Identical to generate_completion() but yields instead of collecting.
    If cancel_event is set between tokens, stops early and returns.

    Args:
        model: EncoderDecoderGPT instance.
        prefix_tokens: List of int token IDs for the prompt.
        encoder_inputs: MultisegInputs with enc_feature_ids, enc_magnitudes, enc_mask, seg_map.
        max_new_tokens: Maximum number of tokens to generate.
        device: Target device.
        temperature: Sampling temperature; 0.0 for greedy.
        top_k: Top-k sampling cutoff; None for full vocab.
        seed: Random seed for the torch Generator.
        bos_id: Token ID that signals end of generation.
        cancel_event: If set between tokens, stops generation early.
        precomputed_encoder: Optional packed encoder state. When provided, the
            encoder forward pass is skipped and the state is used directly.

    Yields:
        Token IDs one at a time (not including the stop token).
    """
    dec_config = model.dec_config
    max_seq_len = dec_config.sequence_len

    # Reject overrun like ``generate_completion`` (see SB-13-P0-05). Silent
    # left-truncation here would also drop BOS and break char-offset
    # alignment; sister functions hard-fail, so this one does too.
    if len(prefix_tokens) > max_seq_len - max_new_tokens:
        raise ValueError(
            f"prefix_tokens length ({len(prefix_tokens)}) exceeds "
            f"max_seq_len ({max_seq_len}) - max_new_tokens ({max_new_tokens}); "
            "filter the document/segment during selection rather than "
            "silently truncating (BOS would be dropped, breaking "
            "char-offset alignment)."
        )
    n_prefix = len(prefix_tokens)
    total_len = n_prefix + max_new_tokens

    seg_map = encoder_inputs.seg_map.to(device)

    if precomputed_encoder is not None:
        encoder_out = precomputed_encoder
    else:
        enc_fids = encoder_inputs.enc_feature_ids.to(device)
        enc_mags = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

    dtype = t.bfloat16 if device.type == "cuda" else t.float32

    kv_cache = KVCache(
        batch_size=1,
        num_heads=dec_config.n_kv_head,
        seq_len=total_len,
        head_dim=dec_config.n_embd // dec_config.n_head,
        num_layers=dec_config.n_layer,
        device=device,
        dtype=dtype,
    )

    last_logits = _prefill_condition(
        model=model,
        prefix_tokens=prefix_tokens,
        encoder_out=encoder_out,
        seg_map=seg_map,
        kv_cache=kv_cache,
        device=device,
    )

    yield from _decode_from_prefill(
        model=model,
        kv_cache=kv_cache,
        last_logits=last_logits,
        encoder_out=encoder_out,
        seg_map=seg_map,
        n_prefix=n_prefix,
        max_new_tokens=max_new_tokens,
        total_len=total_len,
        device=device,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        bos_id=bos_id,
        cancel_event=cancel_event,
    )


# ---------------------------------------------------------------------------
# Condition-level generation helper (amortizes encoder + prefill)
# ---------------------------------------------------------------------------


@t.inference_mode()
@beartype
def generate_samples_for_condition(
    model,  # EncoderDecoderGPT
    prefix_tokens: NonNegativeIntListLike,
    encoder_inputs: MultisegInputs,
    n_samples: NonNegativeIntLike,
    device: t.device,
    *,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    base_seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
) -> _GeneratedSamples:
    """Generate N samples for one conditioning setup, amortizing encoder + prefill.

    Runs the encoder forward pass and decoder prefill exactly once for the
    given ``encoder_inputs`` + ``prefix_tokens`` combination, then decodes
    ``n_samples`` independent continuations by cloning the prefilled KV cache
    per sample. Each sample uses ``base_seed + i`` as its RNG seed.

    This mirrors the encoder/prefill amortization pattern already used in
    :func:`generate_for_document` but extracted into a reusable callable so
    the inline steering eval (``_generate_roundtrip_samples``) can use the
    same pattern without code duplication.

    Args:
        model: EncoderDecoderGPT instance.
        prefix_tokens: List of int token IDs for the prompt. Must be non-empty
            and strictly shorter than ``encoder_inputs.seg_map.shape[1]``.
        encoder_inputs: MultisegInputs with enc_feature_ids, enc_magnitudes,
            enc_mask, seg_map. The helper derives ``n_gen_tokens`` from
            ``seg_map.shape[1] - len(prefix_tokens)``; callers MUST build
            the seg_map such that this is a positive integer. There is no
            fallback for oversized prefixes — the helper raises ValueError.
        n_samples: Number of samples to decode. Must be >= 0. If 0, returns [].
        device: Target device.
        temperature: Sampling temperature; 0.0 for greedy.
        top_k: Top-k sampling cutoff; None for full vocab.
        base_seed: Seed for sample 0; sample i uses ``base_seed + i``.
        bos_id: Token ID that signals end of generation.

    Returns:
        List of ``n_samples`` token lists. Each list contains tokens up to
        (but excluding) the stop token, matching :func:`generate_completion`.

    Raises:
        ValueError: If ``prefix_tokens`` is empty, if its length is >= the
            seg_map total ``T_seg``, or if ``T_seg`` exceeds the decoder's
            ``max_seq_len``. No silent truncation or fallback — callers must
            build seg_map and prefix_tokens that satisfy these constraints.
    """
    if n_samples == 0:
        return []

    dec_config = model.dec_config
    max_seq_len = dec_config.sequence_len
    T_seg = int(encoder_inputs.seg_map.shape[1])
    n_prefix = len(prefix_tokens)

    # Hard preconditions: no silent truncation, no fallback. Callers
    # (specifically _generate_roundtrip_samples) are responsible for
    # building inputs that satisfy these constraints.
    if n_prefix == 0:
        raise ValueError(
            "generate_samples_for_condition: prefix_tokens is empty. "
            "The caller must pass a non-empty prefix."
        )
    if n_prefix >= T_seg:
        raise ValueError(
            f"generate_samples_for_condition: prefix_tokens length {n_prefix} "
            f"must be strictly less than encoder_inputs.seg_map.shape[1]={T_seg}. "
            f"The caller must build seg_map with room for at least 1 generation "
            f"slot beyond the prefix."
        )
    if T_seg > max_seq_len:
        raise ValueError(
            f"generate_samples_for_condition: encoder_inputs.seg_map.shape[1]={T_seg} "
            f"exceeds the decoder's max_seq_len={max_seq_len}. The caller must "
            f"build seg_map within the decoder's context window."
        )

    n_gen_tokens = T_seg - n_prefix
    total_len = T_seg

    # Move encoder inputs to device
    seg_map = encoder_inputs.seg_map.to(device)
    enc_fids = encoder_inputs.enc_feature_ids.to(device)
    enc_mags = encoder_inputs.enc_magnitudes.to(device)
    enc_mask = encoder_inputs.enc_mask.to(device)

    # 1. Encoder forward ONCE
    encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

    # Choose dtype: bfloat16 on CUDA, float32 otherwise
    dtype = t.bfloat16 if device.type == "cuda" else t.float32

    # 2. Prefill ONCE into a base cache
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

    # 3. Per-sample: clone base cache and decode from it
    results: list[list[int]] = []
    for i in range(n_samples):
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
                seed=base_seed + i,
                bos_id=bos_id,
            )
        )
        results.append(gen_tokens)
    return results


@beartype
@dataclass(frozen=True)
class _TargetAlignedSplit:
    """Result of the M4b alignment + split computation.

    Returned by :func:`_compute_target_aligned_prefix_and_split` and
    consumed by :func:`generate_for_document`. Kept as a frozen dataclass
    so the helper can be unit-tested without standing up a real
    ``EncoderDecoderGPT`` model.
    """

    seg_slice: _SegmentFeatureTuple  # may be shorter than segments[:target+1] under filter
    all_tokens: NonEmptyNonNegativeIntTupleLike  # tokens through target.char_end
    split_idx: NonNegativeIntLike  # index where target tokens start
    ground_truth: str                     # bos_text[target_seg.char_start : target_seg.char_end]
    pre_segment_text: str                 # bos_text[len(bos_prefix) : target_seg.char_start]
    text_through_target: str              # bos_text[len(bos_prefix) : target_seg.char_end]


@beartype
def _compute_target_aligned_prefix_and_split(
    *,
    tokenizer,
    segments: _SegmentFeatureList,
    target_segment_idx: NonNegativeIntLike,
    doc_text: str,
    bos_id: NonNegativeIntLike,
) -> _TargetAlignedSplit:
    """Compute the M4b aligned seg_slice + token-to-segment split.

    Extracted from :func:`generate_for_document` so the alignment + split
    logic is unit-testable in isolation. Production callers receive a
    :class:`_TargetAlignedSplit` containing all derived quantities used
    downstream (encoder pack, generation slot, ground-truth slice).

    Cross-tokenizer mismatch is absorbed via
    :func:`steering.segmentation.align_char_ranges_to_decoded_prefix`
    with the site-scoped counter name
    ``segmentation.char_end_extended.completion_eval``.

    Raises:
        RuntimeError: alignment's reverse filter dropped the target
            segment. Structurally guaranteed not to fire in production
            (segment widths >> 20-char window) but defended for any
            future caller with sub-window segments.
        ValueError: alignment exceeds the canonical window in either
            direction.
    """
    # NOTE: target_seg holds the PRE-alignment char_end. If the M4b
    # alignment helper below forward-extends the target's char_end (rare
    # cross-tokenizer mismatch case), the ground_truth + pre_segment_text
    # computed below will reflect the source-tokenizer's view, not the
    # extended consumer view. Loss computation operates on actual tokens
    # (post-alignment) so this is observability-only - the loss is still
    # correct.
    if not segments:
        raise ValueError(
            "segments must contain at least one segment; production callers "
            "always pass non-empty segments from FeatureIndex"
        )
    if target_segment_idx >= len(segments):
        raise ValueError(
            f"target_segment_idx={target_segment_idx} out of range for "
            f"{len(segments)} segments"
        )
    target_seg = segments[target_segment_idx]

    # Stored char offsets are in BOS-inclusive coordinate system (from extraction).
    # Prepend the decoded BOS to doc_text so slicing uses the same coordinate system.
    bos_prefix = tokenizer.decode([bos_id])
    bos_text = bos_prefix + doc_text

    # Slice in BOS coordinate system - correct word boundaries
    ground_truth = bos_text[target_seg.char_start : target_seg.char_end]
    # Pre-segment text for the DocumentCompletion output
    pre_segment_text = bos_text[len(bos_prefix) : target_seg.char_start]

    # Tokenize all text through end of target segment to derive prefix + gen length
    text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
    all_tokens = tokenizer.encode(text_through_target, prepend=bos_id)

    # Compute character-level segment assignments (same as training dataloader).
    # Defend against cross-tokenizer mismatch via the centralised primitive.
    char_offsets = token_to_char_offsets(tokenizer, all_tokens)
    decoded_len = char_offsets[-1] if char_offsets else 0

    seg_slice = segments[: target_segment_idx + 1]
    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in seg_slice],
        char_ends=[s.char_end for s in seg_slice],
        decoded_len=decoded_len,
        counter_name=_COMPLETION_ALIGNMENT_COUNTER,
    )
    # Skip the dataclasses.replace rebuild when the primitive made no
    # changes (the common no-mismatch hot path). new_starts is filtered
    # by kept_indices; checking those two conditions covers every way
    # the primitive can mutate the seg_slice.
    if kept_indices != list(range(len(seg_slice))) or new_ends != [s.char_end for s in seg_slice]:
        seg_slice = [
            dataclasses.replace(
                seg_slice[orig_idx],
                char_start=new_starts[new_idx],
                char_end=new_ends[new_idx],
            )
            for new_idx, orig_idx in enumerate(kept_indices)
        ]

    # Defensive: alignment must preserve the target index. In production
    # this is structurally guaranteed (segment widths are ~128 tokens, far
    # wider than the 20-char window), but the assertion catches any future
    # caller that passes sub-window segments where reverse-direction
    # filtering could drop the target slot.
    if len(seg_slice) != target_segment_idx + 1:
        raise RuntimeError(
            f"M4b alignment filtered the target segment: "
            f"seg_slice length {len(seg_slice)} != target_segment_idx + 1 "
            f"= {target_segment_idx + 1}. This means decoded_len was so "
            f"short that the alignment helper's reverse-direction filter "
            f"dropped indices including the target. Production code paths "
            f"do not reach this state because segment widths far exceed the "
            f"20-char window."
        )

    token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
    token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

    seg_char_starts, seg_char_ends = _segment_bounds_for_assignment(
        seg_slice,
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

    return _TargetAlignedSplit(
        seg_slice=tuple(seg_slice),
        all_tokens=tuple(all_tokens),
        split_idx=split_idx,
        ground_truth=ground_truth,
        pre_segment_text=pre_segment_text,
        text_through_target=text_through_target,
    )


# ---------------------------------------------------------------------------
# Document-level generation orchestrator
# ---------------------------------------------------------------------------


@t.inference_mode()
@beartype
def generate_for_document(
    model,  # EncoderDecoderGPT
    tokenizer,
    doc_text: str,
    segments: _SegmentFeatureList,
    target_segment_idx: NonNegativeIntLike,
    feature_id: NonNegativeIntLike,
    mode: EvalMode,
    boundaries: t.Tensor,
    feature_cfg: ThresholdConfig | SamplingBundle,
    precomputed_thresholds: t.Tensor | None,
    device: t.device,
    num_buckets: PositiveIntLike,
    doc_idx: NonNegativeIntLike = 0,
    samples_per_condition: PositiveIntLike = 5,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    base_seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
    next_segment_conditioning_offset: bool = False,
    feature_mask: FeatureMask | None = None,
) -> DocumentCompletion:
    """Generate completions for a single document under all conditions.

    For ablation mode: generates under real and ablated conditions.
    For insertion mode: generates under real and per-bucket addition.

    Derives prefix tokens and generation token count from character-level
    segment assignments rather than a fixed ``segment_size`` parameter.

    Args:
        model: EncoderDecoderGPT in eval mode.
        tokenizer: Tokenizer with encode/decode.
        doc_text: Full document text.
        segments: Raw SegmentFeatures from FeatureIndex (up to target segment).
        target_segment_idx: Segment to generate.
        feature_id: Feature being tested.
        mode: EvalMode.ABLATION or EvalMode.INSERTION.
        boundaries: Bucket boundaries for discretization.
        feature_cfg: ThresholdConfig or SamplingBundle for feature selection.
        precomputed_thresholds: From build_thresholds_from_stats (threshold path only).
        device: Model device.
        num_buckets: Number of magnitude buckets.
        doc_idx: Document index for output.
        samples_per_condition: Samples per condition.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        base_seed: Base seed (incremented per sample).
        bos_id: BOS/stop token ID.
        next_segment_conditioning_offset: If True, decoder segment s attends
            to feature signals from source segment s + 1.
    Returns:
        DocumentCompletion with all samples.

    Encoder/prefill reuse scope (no cross-condition amortization):
        Within a single condition, the encoder forward and decoder prefill
        are executed once and the resulting KV state is cloned for each of
        the ``samples_per_condition`` samples. Across conditions there is
        no reuse: encoder and prefill are NOT amortized across the bucket
        loop. In insertion mode with ``num_buckets=N`` the encoder + prefill
        therefore run ``1 + N`` times total (once per condition), even
        though the non-target encoder slots and the prefix prefill are
        invariant across the bucket loop. See
        ``tests/test_multiseg_generator_amortization.py`` for the
        call-count regression lock.

    TODO(wave3d): this does not amortize across conditions. The encoder
        output for non-target segments is invariant across all conditions
        and the prefix prefill is invariant across all conditions (prefix
        tokens never attend to the target segment's encoder slot under the
        ``encoder_strict`` seg_map routing). A real fix would not be
        local: it would require splitting encoder inputs into shared + target-slot edits
        and sharing a base KV cache across all conditions (i.e. true
        cross-condition reuse, not just the current within-condition
        sample reuse). Deferred
        because (a) the encoder is not slot-additive out of the box and
        (b) the sharing path needs a non-trivial refactor of
        ``_prefill_condition`` to accept a pre-filled base cache.
    """
    aligned_split = _compute_target_aligned_prefix_and_split(
        tokenizer=tokenizer,
        segments=segments,
        target_segment_idx=target_segment_idx,
        doc_text=doc_text,
        bos_id=bos_id,
    )
    all_tokens = aligned_split.all_tokens
    split_idx = aligned_split.split_idx
    ground_truth = aligned_split.ground_truth
    pre_segment_text = aligned_split.pre_segment_text

    # Materialise as list because downstream consumers (build_multiseg_inputs,
    # _prefill_condition, generate_samples_for_condition) expect list[int].
    prefix_tokens = list(all_tokens[:split_idx])
    n_gen_tokens = len(all_tokens) - split_idx
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
    if feature_mask is not None:
        feature_mask.assert_no_masked(t.tensor([feature_id], dtype=t.int64))

    # Threshold and discretize all segments up to and including target.
    prepared = prepare_segments(
        segments[: target_segment_idx + 1],
        boundaries,
        feature_cfg,
        precomputed_thresholds,
        feature_mask=feature_mask,
    )

    # Build ground truth signals from the target segment's post-threshold features
    target_prepared = prepared[target_segment_idx]

    ground_truth_signals = build_ground_truth_signals(target_prepared, feature_id, mode)

    # Build condition list
    conditions: list[tuple[CompletionConditionLabel, int | None]] = []
    if mode is EvalMode.ABLATION:
        conditions = [
            (CompletionCondition.REAL, None),
            (CompletionCondition.ABLATED, None),
        ]
    else:  # insertion
        conditions = [(CompletionCondition.REAL, None)]
        conditions.extend((added_bucket_condition(b), b) for b in range(num_buckets))

    samples: list[CompletionSample] = []
    seed_counter = 0

    for condition_name, bucket in conditions:
        if condition_name == CompletionCondition.ABLATED:
            cond_segs = ablate_feature(prepared, target_segment_idx, feature_id)
        elif str(condition_name).startswith("added_bucket_"):
            assert bucket is not None
            cond_segs = add_feature(prepared, target_segment_idx, feature_id, bucket)
        else:
            cond_segs = prepared

        encoder_inputs = build_multiseg_inputs(
            segments=cond_segs,
            prefix_tokens=prefix_tokens,
            n_gen_tokens=n_gen_tokens,
            target_segment_idx=target_segment_idx,
            tokenizer=tokenizer,
            next_segment_conditioning_offset=next_segment_conditioning_offset,
        )

        # Compute encoder output ONCE for this condition (identical across samples)
        enc_fids = encoder_inputs.enc_feature_ids.to(device)
        enc_mags = encoder_inputs.enc_magnitudes.to(device)
        enc_mask = encoder_inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        # seg_map on device
        seg_map = encoder_inputs.seg_map.to(device)

        # Allocate base KV cache and run prefill ONCE for this condition
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

            # Clone the prefilled KV cache for this sample
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
                    bucket_index=bucket,
                )
            )

    return DocumentCompletion(
        doc_idx=doc_idx,
        feature_id=feature_id,
        pre_segment_text=pre_segment_text,
        ground_truth_segment=ground_truth,
        target_segment_idx=target_segment_idx,
        samples=samples,
        ground_truth_signals=ground_truth_signals,
    )


# ---------------------------------------------------------------------------
# Self-conditioned generation
# ---------------------------------------------------------------------------


@beartype
def _self_cond_decode_loop(
    *,
    model,
    provider: SelfConditionedFeatureProvider,
    state,
    kv_cache: _KVCacheLike,
    last_logits: t.Tensor,
    n_prefix: PositiveIntLike,
    max_new_tokens: NonNegativeIntLike,
    total_len: PositiveIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike,
    top_k: PositiveIntLike | None,
    seed: StrictIntLike,
    bos_id: NonNegativeIntLike,
    cancel_event=None,
):
    """Per-step decode loop that drives the SelfCondProvider.

    Non-boundary emitted tokens are measured and committed under the current
    state. A Kth emitted token is measured speculatively under the outgoing
    state, rewound, and committed at the same position under the state built
    from that completed window. The shared helper therefore returns logits for
    the next distribution with no one-token state lag.
    """
    rng = t.Generator(device=device)
    rng.manual_seed(seed)

    fp_buffer: tuple[t.Tensor, ...] = ()
    tokens_in_seg = 0
    cur_pos = n_prefix

    for _ in range(max_new_tokens):
        if cancel_event is not None and cancel_event.is_set():
            return

        next_token = sample_next_token(last_logits, rng, temperature, top_k)
        token_id = int(next_token[0, 0].item())
        if token_id == bos_id:
            return
        yield token_id
        cur_pos += 1
        if cur_pos >= total_len:
            return

        result = self_conditioned_decode_step(
            model=model,
            provider=provider,
            state=state,
            fp_buffer=fp_buffer,
            tokens_in_seg=tokens_in_seg,
            emitted_token=next_token,
            kv_cache=kv_cache,
            device=device,
        )
        last_logits = result.logits[:, -1, :]
        state = result.state
        fp_buffer = result.fp_buffer
        tokens_in_seg = result.tokens_in_seg


@beartype
def _self_cond_prefill(
    *,
    model,
    provider: SelfConditionedFeatureProvider,
    encoder_inputs: MultisegInputs,
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    kv_cache: _KVCacheLike,
    device: t.device,
):
    """Run prefill with REAL per-segment SAE features (the same encoder
    inputs the training dataloader and the eval-cache builder produce),
    capture fp_logits over the final real prompt segment, seed S0, and commit
    the assistant-boundary token under S0.

    The caller supplies ``encoder_inputs: MultisegInputs`` — these are
    the prefix's real ``(enc_feature_ids, enc_magnitudes, enc_mask,
    seg_map)`` extracted from the SAE pipeline (or from a context
    cache). This replaces the previous "chunked prefill in nanochat
    token space" approach and matches the training-time conditioning
    contract exactly: prompt tokens are routed to their real per-segment
    encoder slots via the precomputed seg_map.

    Returns ``(last_logits, state)`` where ``state`` is a
    :class:`SelfCondState` ready for the decode loop. The last slot of
    ``state.encoder_out`` is the seeded prediction; ``target_slot_idx``
    points at it so the FIRST GENERATED token is self-conditioned.
    """
    enc_fids = encoder_inputs.enc_feature_ids.to(device)
    enc_mags = encoder_inputs.enc_magnitudes.to(device)
    enc_mask = encoder_inputs.enc_mask.to(device)
    empty_slots = (~enc_mask.any(dim=-1)).nonzero(as_tuple=False)
    if empty_slots.numel():
        raise EmptySelfConditioningSegmentError(
            "self-conditioned multisegment prefill received empty encoder "
            f"slots {empty_slots[:, 1].tolist()}"
        )
    encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

    n_prefix = len(prefix_tokens)
    seg_map = encoder_inputs.seg_map.to(device)
    if seg_map.shape[1] < n_prefix:
        raise ValueError(
            f"encoder_inputs.seg_map has {seg_map.shape[1]} positions but "
            f"prefix has {n_prefix} tokens; the caller must build seg_map "
            f"covering the full prefix."
        )
    seg_map_prefix = seg_map[:, :n_prefix]
    source_mask = final_real_prompt_source_mask(
        seg_map_prefix,
        encoder_slot_count=encoder_out.layout.n_segments,
    )
    prefix_ids = t.tensor([prefix_tokens], dtype=t.int64, device=device)
    _prefix_logits, prefix_fp_logits = model(
        prefix_ids[:, :-1],
        targets=None,
        seg_map=seg_map_prefix[:, :-1],
        kv_cache=kv_cache,
        encoder_out=encoder_out,
        encoder_strict=True,
        return_fp_logits=True,
    )

    prefill_state = SelfCondState(
        encoder_out=encoder_out.detach(),
        target_slot_idx=encoder_out.layout.n_segments - 1,
    )
    state = provider.seed_from_prompt(
        prefill_state,
        prefix_fp_logits[:, source_mask, :].detach(),
    )
    boundary_map = t.full(
        (1, 1),
        state.target_slot_idx,
        dtype=t.long,
        device=device,
    )
    logits = model(
        prefix_ids[:, -1:],
        targets=None,
        seg_map=boundary_map,
        kv_cache=kv_cache,
        encoder_out=state.encoder_out.to(device),
        encoder_strict=True,
    )
    return logits[:, -1, :], state


@t.inference_mode()
@beartype
def generate_completion_self_conditioned(
    model,
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    provider: SelfConditionedFeatureProvider,
    encoder_inputs: MultisegInputs,
    max_new_tokens: NonNegativeIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
) -> _GeneratedTokenList:
    """Self-conditioned counterpart to :func:`generate_completion`.

    Prefill uses the REAL per-segment SAE features supplied via
    ``encoder_inputs`` (same shape as the training-time encoder input).
    The FP rows routed to the final real prompt segment seed a fresh slot for
    sampled distributions 1..K. If another token is requested, the completed
    K-row window builds the slot that controls distribution K+1 through the
    shared speculative-old/rewind/new-commit boundary helper.
    """
    return list(
        generate_completion_stream_self_conditioned(
            model=model,
            prefix_tokens=prefix_tokens,
            provider=provider,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new_tokens,
            device=device,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            bos_id=bos_id,
        )
    )


@t.inference_mode()
@beartype
def generate_completion_stream_self_conditioned(
    model,
    prefix_tokens: NonEmptyNonNegativeIntListLike,
    provider: SelfConditionedFeatureProvider,
    encoder_inputs: MultisegInputs,
    max_new_tokens: NonNegativeIntLike,
    device: t.device,
    temperature: NonNegativeFiniteNumberLike = 0.7,
    top_k: PositiveIntLike | None = 50,
    seed: StrictIntLike = 42,
    bos_id: NonNegativeIntLike = 2,
    cancel_event=None,
):
    """Streaming counterpart to :func:`generate_completion_self_conditioned`."""
    require_fixed_segment_mode(
        provider.cfg.segment_mode,
        surface="multisegment self-conditioned generation",
        source="SelfCondConfig",
    )
    if max_new_tokens == 0:
        return
    dec_config = model.dec_config
    max_seq_len = dec_config.sequence_len
    prefix_tokens = prefix_tokens[-(max_seq_len - max_new_tokens):]
    n_prefix = len(prefix_tokens)
    total_len = n_prefix + max_new_tokens

    dtype = t.bfloat16 if device.type == "cuda" else t.float32
    kv_cache = KVCache(
        batch_size=1,
        num_heads=dec_config.n_kv_head,
        seq_len=total_len,
        head_dim=dec_config.n_embd // dec_config.n_head,
        num_layers=model.num_cache_layers,
        device=device,
        dtype=dtype,
    )

    last_logits, state = _self_cond_prefill(
        model=model,
        provider=provider,
        encoder_inputs=encoder_inputs,
        prefix_tokens=prefix_tokens,
        kv_cache=kv_cache,
        device=device,
    )

    yield from _self_cond_decode_loop(
        model=model,
        provider=provider,
        state=state,
        kv_cache=kv_cache,
        last_logits=last_logits,
        n_prefix=n_prefix,
        max_new_tokens=max_new_tokens,
        total_len=total_len,
        device=device,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        bos_id=bos_id,
        cancel_event=cancel_event,
    )
