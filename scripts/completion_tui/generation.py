"""Generation orchestration for the completion TUI.

Handles streaming generation, cancellation, and undo/redo of generated text.
Converts TUI state (Segment, FeatureCondition) to encoder inputs (SegmentFeatures,
MultisegInputs) and delegates to the multi-segment generator.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch as t

from scripts.completion_tui.segment_ops import compute_takeover, retokenize_boundaries
from scripts.completion_tui.state import (
    GenerationSnapshot,
    Segment,
    SessionState,
    deep_copy_segments,
    display_to_internal,
)
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask
from steering.eval.multiseg_generator import (
    MultisegInputs,
    build_multiseg_inputs,
    generate_completion_stream,
)
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from steering.tokenizer_protocols import TextTokenizer, require_text_tokenizer

# ---------------------------------------------------------------------------
# GenerationHandle
# ---------------------------------------------------------------------------


@dataclass
class GenerationHandle:
    """Handle returned by GenerationManager.start_generation."""

    cancel_event: threading.Event
    token_iterator: Iterator[int]
    snapshot: GenerationSnapshot
    target_segment_idx: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_target(segments: list[Segment]) -> int:
    """Return the index of the generation-target segment, or raise."""
    for i, seg in enumerate(segments):
        if seg.is_generation_target:
            return i
    raise ValueError("No generation target segment found")


def _segments_to_features(
    segments: list[Segment],
    num_buckets: int,
    has_negatives: bool,
    *,
    feature_mask: FeatureMask | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
) -> list[SegmentFeatures]:
    """Convert TUI Segments to SegmentFeatures for the encoder.

    Only enabled FeatureConditions are included. Display magnitudes are
    converted to internal bucket indices via ``display_to_internal``.
    """
    result: list[SegmentFeatures] = []
    for seg in segments:
        enabled = [f for f in seg.features if f.enabled]

        if enabled:
            fids = t.tensor([f.feature_id for f in enabled], dtype=t.int32)
            mags = t.tensor(
                [display_to_internal(f.magnitude, num_buckets, has_negatives) for f in enabled],
                dtype=t.int64,
            )
        else:
            fids = t.zeros(0, dtype=t.int32)
            mags = t.zeros(0, dtype=t.int64)

        if feature_mask is not None:
            feature_mask.assert_no_masked(fids)
        if consolidation is not None:
            consolidation.assert_representatives_only(
                fids,
                surface=f"completion_tui generation segment {seg.segment_idx}",
            )

        result.append(
            SegmentFeatures(
                segment_idx=seg.segment_idx,
                position=0,
                feature_ids=fids,
                feature_mags=mags,
                char_start=seg.char_start,
                char_end=seg.char_end,
            )
        )
    return result


def _apply_generation(state: SessionState, generated_text: str) -> None:
    """Append generated_text at the target segment's char_end, updating boundaries."""
    target_idx = _find_target(state.segments)
    target = state.segments[target_idx]
    insert_pos = target.char_end

    state.document_text = (
        state.document_text[:insert_pos] + generated_text + state.document_text[insert_pos:]
    )

    gen_len = len(generated_text)
    target.char_end += gen_len
    for seg in state.segments[target_idx + 1 :]:
        seg.char_start += gen_len
        seg.char_end += gen_len


# ---------------------------------------------------------------------------
# build_inputs_from_state
# ---------------------------------------------------------------------------


def build_inputs_from_state(
    state: SessionState,
    tokenizer: TextTokenizer,
    *,
    feature_mask: FeatureMask | None = None,
    consolidation: FeatureClusterConsolidation | None = None,
) -> tuple[list[int], MultisegInputs, int]:
    """Build prefix_tokens and MultisegInputs from SessionState.

    Returns:
        ``(prefix_tokens, inputs, target_segment_idx)`` where
        ``target_segment_idx`` is the index of the generation-target
        segment within the *effective* (post-takeover) segment list.

    Steps:
        1.  ``compute_takeover`` to get effective segments.
        1b. Convert TUI Segments to SegmentFeatures (enabled features only,
            display magnitudes → internal bucket indices).
        2.  Tokenize ``document_text``.
        3.  Assign tokens to segments via character-level overlap.
        4.  Split into prefix (tokens before target) and generation target.
        5.  Call ``build_multiseg_inputs``.
    """
    effective = compute_takeover(state.segments)
    seg_features = _segments_to_features(
        effective,
        state.num_buckets,
        state.has_negatives,
        feature_mask=feature_mask,
        consolidation=consolidation,
    )

    target_idx = _find_target(effective)
    target_seg = effective[target_idx]

    token_ids = tokenizer.encode(state.document_text)

    if token_ids:
        offsets = token_to_char_offsets(tokenizer, token_ids)
        tok_starts = t.tensor(offsets[:-1], dtype=t.int64)
        tok_ends = t.tensor(offsets[1:], dtype=t.int64)

        seg_starts = t.tensor([s.char_start for s in effective], dtype=t.int64)
        seg_ends = t.tensor([s.char_end for s in effective], dtype=t.int64)

        # Segment char boundaries are in source-model-with-BOS coordinates; extend
        # the first segment backward to absorb any pre-segment tokens (source BOS gap).
        if len(seg_starts) > 0 and tok_ends[0] <= seg_starts[0]:
            seg_starts = seg_starts.clone()
            seg_starts[0] = tok_starts[0]

        assignments = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assignments = t.cummax(assignments, dim=0).values

        target_positions = (assignments == target_idx).nonzero(as_tuple=True)[0]
        tokens_in_target = len(target_positions)

        # Include ALL tokens (including those already in the target segment)
        # as prefix context so the model continues from existing text.
        prefix_tokens = list(token_ids)
    else:
        prefix_tokens = []
        tokens_in_target = 0
    if target_seg.target_token_count is not None:
        n_gen_tokens = max(1, target_seg.target_token_count - tokens_in_target)
    else:
        n_gen_tokens = max(tokens_in_target, 1)

    inputs = build_multiseg_inputs(
        segments=seg_features,
        prefix_tokens=prefix_tokens,
        n_gen_tokens=n_gen_tokens,
        target_segment_idx=target_idx,
        tokenizer=tokenizer,
    )

    return prefix_tokens, inputs, target_idx


# ---------------------------------------------------------------------------
# GenerationManager
# ---------------------------------------------------------------------------


class GenerationManager:
    """Orchestrates streaming generation, cancellation, and undo/redo."""

    def __init__(
        self,
        model: Any,
        tokenizer: object,
        device: t.device,
        boundaries: t.Tensor,
        precomputed_thresholds: t.Tensor | None = None,
        temperature: float = 0.7,
        top_k: int | None = 50,
        max_new_tokens: int | None = None,
        *,
        feature_mask: FeatureMask | None = None,
        consolidation: FeatureClusterConsolidation | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = require_text_tokenizer(
            tokenizer,
            surface="completion TUI generation",
        )
        self.device = device
        self.boundaries = boundaries
        self.precomputed_thresholds = precomputed_thresholds
        self.temperature = float(temperature)
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.feature_mask = feature_mask
        self.consolidation = consolidation

    def start_generation(self, state: SessionState) -> GenerationHandle:
        """Prepare inputs, create streaming iterator, return handle.

        1. Synchronous retokenize_boundaries.
        2. Compute takeover and build MultisegInputs.
        3. Create cancel event.
        4. Start generate_completion_stream.
        5. Return GenerationHandle with pre-generation snapshot.
        """
        retokenize_boundaries(state.document_text, state.segments, self.tokenizer)

        prefix_tokens, inputs, target_idx = build_inputs_from_state(
            state,
            self.tokenizer,
            feature_mask=self.feature_mask,
            consolidation=self.consolidation,
        )

        snapshot = GenerationSnapshot(
            document_text=state.document_text,
            segments=state.segments,
            generated_text="",
            generated_token_ids=[],
            seed=state.seed,
        )

        cancel_event = threading.Event()
        max_new_tokens = inputs.seg_map.shape[1] - len(prefix_tokens)
        if self.max_new_tokens is not None:
            max_new_tokens = min(max_new_tokens, int(self.max_new_tokens))
        max_new_tokens = max(1, max_new_tokens)

        token_iter = generate_completion_stream(
            model=self.model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=inputs,
            max_new_tokens=max_new_tokens,
            device=self.device,
            temperature=self.temperature,
            top_k=self.top_k,
            seed=state.seed,
            cancel_event=cancel_event,
        )

        return GenerationHandle(
            cancel_event=cancel_event,
            token_iterator=token_iter,
            snapshot=snapshot,
            target_segment_idx=target_idx,
        )

    def cancel(self, handle: GenerationHandle) -> None:
        """Signal the generator to stop after the current token."""
        handle.cancel_event.set()


# ---------------------------------------------------------------------------
# LazyGenerationManager
# ---------------------------------------------------------------------------


class LazyGenerationManager:
    """Factory-based wrapper that defers model loading to first generation."""

    def __init__(self, factory: Callable[[], GenerationManager]) -> None:
        self._factory = factory
        self._inner: GenerationManager | None = None

    def start_generation(self, state: SessionState) -> GenerationHandle:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner.start_generation(state)

    def cancel(self, handle: GenerationHandle) -> None:
        if self._inner is not None:
            self._inner.cancel(handle)
        else:
            handle.cancel_event.set()


# ---------------------------------------------------------------------------
# undo / redo
# ---------------------------------------------------------------------------


def undo_generation(state: SessionState) -> bool:
    """Pop from history, restore state, push to redo stack. False if empty."""
    if not state.generation_history:
        return False

    snapshot = state.generation_history.pop()
    state.document_text = snapshot.document_text
    state.segments = deep_copy_segments(snapshot.segments)
    state.generation_redo_stack.append(snapshot)
    return True


def redo_generation(state: SessionState) -> bool:
    """Pop from redo stack, re-apply generation, push to history. False if empty."""
    if not state.generation_redo_stack:
        return False

    snapshot = state.generation_redo_stack.pop()
    _apply_generation(state, snapshot.generated_text)
    state.generation_history.append(snapshot)
    return True
