"""Tests for the generation manager module.

Covers: build_inputs_from_state, undo/redo, and GenerationManager with mocks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch as t

from scripts.completion_tui.generation import (
    GenerationManager,
    build_inputs_from_state,
    redo_generation,
    undo_generation,
)
from scripts.completion_tui.state import (
    FeatureCondition,
    Segment,
    SessionState,
)
from tests.test_completion_tui._generation_history import push_generation
from tests.test_completion_tui.conftest import WordMockTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feat(fid: int, mag: int, *, enabled: bool = True) -> FeatureCondition:
    return FeatureCondition(feature_id=fid, magnitude=mag, enabled=enabled)


def _seg(
    idx: int,
    start: int,
    end: int,
    features: list[FeatureCondition] | None = None,
    *,
    enabled: bool = True,
    target: bool = False,
    token_count: int | None = None,
) -> Segment:
    return Segment(
        segment_idx=idx,
        char_start=start,
        char_end=end,
        color_idx=idx,
        features=features or [],
        enabled=enabled,
        is_generation_target=target,
        target_token_count=token_count,
    )


def _state(
    text: str,
    segments: list[Segment],
    *,
    num_buckets: int = 4,
    has_negatives: bool = False,
) -> SessionState:
    return SessionState(
        document_text=text,
        segments=segments,
        num_buckets=num_buckets,
        has_negatives=has_negatives,
    )


# ---------------------------------------------------------------------------
# build_inputs_from_state tests
# ---------------------------------------------------------------------------


class TestBuildInputsFromState:
    """Tests for build_inputs_from_state."""

    def test_segment_count_matches_effective(self, word_tokenizer: WordMockTokenizer) -> None:
        """Effective segments (all enabled) appear in encoder inputs."""
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),
                _seg(1, 6, 11, [_feat(20, 1)], target=True, token_count=5),
            ],
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        assert inputs.enc_feature_ids.shape[1] == 2

    def test_disabled_segment_absorbed(self, word_tokenizer: WordMockTokenizer) -> None:
        """Disabled segment absorbed by neighbor → fewer effective segments."""

        state = _state(
            "hello world",
            [
                _seg(0, 0, 5, [_feat(10, 2)]),
                _seg(1, 5, 6, [_feat(30, 0)], enabled=False),  # disabled " "
                _seg(2, 6, 11, [_feat(20, 1)], target=True, token_count=5),
            ],
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        # 3 segments → 2 effective (seg1 absorbed by seg0)
        assert inputs.enc_feature_ids.shape[1] == 2

    def test_display_magnitudes_converted_no_negatives(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Without negatives, display mag == internal bucket index."""

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),  # display 2 → internal 2
                _seg(1, 6, 11, [_feat(20, 1)], target=True, token_count=5),
            ],
            num_buckets=4,
            has_negatives=False,
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        assert inputs.enc_magnitudes[0, 0, 0].item() == 2

    def test_display_magnitudes_converted_with_negatives(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """With negatives, display -1 → internal = num_buckets - 1."""

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, [_feat(10, -1)]),  # display -1 → internal 3
                _seg(1, 6, 11, [_feat(20, 0)], target=True, token_count=5),
            ],
            num_buckets=4,
            has_negatives=True,
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        assert inputs.enc_magnitudes[0, 0, 0].item() == 3

    def test_disabled_features_excluded(self, word_tokenizer: WordMockTokenizer) -> None:
        """Disabled FeatureConditions are excluded from encoder inputs."""

        state = _state(
            "hello world",
            [
                _seg(
                    0,
                    0,
                    6,
                    [
                        _feat(10, 2, enabled=True),
                        _feat(20, 1, enabled=False),
                    ],
                ),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        # Only feature 10 should be present (masked in)
        assert inputs.enc_mask[0, 0, 0].item() is True
        assert inputs.enc_feature_ids[0, 0, 0].item() == 10
        # No second feature
        if inputs.enc_feature_ids.shape[2] > 1:
            assert inputs.enc_mask[0, 0, 1].item() is False

    def test_seg_map_assigns_tokens_correctly(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """seg_map assigns prefix tokens to correct segments, gen tokens to target."""

        # "hello world" → tokens ["hello", " ", "world"]
        # seg0=[0,6) covers "hello " → tokens "hello" and " "
        # seg1=[6,11) covers "world" → target
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),
                _seg(1, 6, 11, [_feat(20, 1)], target=True, token_count=5),
            ],
        )
        prefix_tokens, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        # All tokens (including target's "world") are now prefix context
        # "hello world" → ["hello", " ", "world"] → 3 tokens
        assert len(prefix_tokens) == 3
        # Prefix tokens map to their respective segments
        assert inputs.seg_map[0, 0].item() == 0  # "hello" → seg0
        assert inputs.seg_map[0, 1].item() == 0  # " " → seg0
        assert inputs.seg_map[0, 2].item() == 1  # "world" → seg1 (target)
        # Remaining generated tokens (5 - 1 = 4) map to target segment
        for i in range(3, 3 + 4):
            assert inputs.seg_map[0, i].item() == 1

    def test_prefix_tokens_are_correct_ids(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Prefix tokens include all text (including target segment's existing text)."""

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )
        prefix_tokens, _inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        # All tokens are prefix context (including "world" in the target)
        expected = word_tokenizer.encode("hello world")  # "hello", " ", "world"
        assert prefix_tokens == expected


class TestDynamicGenTokenCount:
    """n_gen_tokens should subtract tokens already in the target segment."""

    def test_zero_width_target_uses_full_budget(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Zero-width target → n_gen_tokens = target_token_count."""
        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=10),
            ],
        )
        _prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        # prefix = ["hello", " "] → 2 tokens; seg_map total = 2 + 10 = 12
        n_gen = inputs.seg_map.shape[1] - len(_prefix)
        assert n_gen == 10

    def test_partially_filled_target_subtracts_existing(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Partially filled target → fewer generation tokens."""
        # "hello world" → tokens: ["hello", " ", "world"]
        # seg0 = "hello " → ["hello", " "] (2 tokens)
        # seg1 = "world" → ["world"] (1 token already in target)
        # target_token_count = 10, tokens_in_target = 1 → n_gen = 9
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=10),
            ],
        )
        prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        n_gen = inputs.seg_map.shape[1] - len(prefix)
        assert n_gen == 9

    def test_fully_filled_target_clamps_to_one(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Fully filled target → n_gen_tokens clamped to 1."""
        # "hello world" → 3 tokens, target has 1, budget is 1 → 1-1=0 → clamp to 1
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=1),
            ],
        )
        prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        n_gen = inputs.seg_map.shape[1] - len(prefix)
        assert n_gen == 1

    def test_no_token_count_falls_back(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """No target_token_count → fallback to tokens_in_target count."""
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=None),
            ],
        )
        prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        n_gen = inputs.seg_map.shape[1] - len(prefix)
        # tokens_in_target = 1 ("world"), so fallback = max(1, 1) = 1
        assert n_gen == 1


class TestPrefillContextInclusion:
    """Prefix tokens should include existing text in the target segment."""

    def test_partially_filled_target_included_in_prefix(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Tokens already in the target segment are included as prefix context."""
        # "hello world" → ["hello", " ", "world"]
        # "world" is in the target segment and must be in the prefix
        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )
        prefix, _inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        all_tokens = word_tokenizer.encode("hello world")
        assert prefix == all_tokens

    def test_zero_width_target_prefix_is_all_tokens(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Zero-width target: prefix = all tokens (nothing in target)."""
        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        prefix, _inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        expected = word_tokenizer.encode("hello ")
        assert prefix == expected

    def test_multiple_tokens_in_target(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """Multiple tokens in target all appear in prefix."""
        # "hello the quick" → ["hello", " ", "the", " ", "quick"]
        # seg0 = [0,6), seg1 = [6,15) → target has ["the", " ", "quick"] = 3 tokens
        # n_gen_tokens = 10 - 3 = 7
        state = _state(
            "hello the quick",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 15, [], target=True, token_count=10),
            ],
        )
        prefix, inputs, _tidx = build_inputs_from_state(state, word_tokenizer)
        all_tokens = word_tokenizer.encode("hello the quick")
        assert prefix == all_tokens
        n_gen = inputs.seg_map.shape[1] - len(prefix)
        assert n_gen == 7


# ---------------------------------------------------------------------------
# undo / redo tests
# ---------------------------------------------------------------------------


class TestUndoGeneration:
    """Tests for undo_generation."""

    def test_restores_text_and_segments(self) -> None:
        """undo restores document_text and segments exactly."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        original_text = state.document_text
        original_seg1_end = state.segments[1].char_end

        push_generation(state, "world", [100])
        assert state.document_text == "hello world"

        result = undo_generation(state)
        assert result is True
        assert state.document_text == original_text
        assert state.segments[1].char_end == original_seg1_end

    def test_pushes_to_redo_stack(self) -> None:
        """undo pushes snapshot to redo stack."""

        state = _state(
            "hello ",
            [_seg(0, 0, 6, [], target=True, token_count=5)],
        )
        push_generation(state, "world", [100])
        undo_generation(state)

        assert len(state.generation_history) == 0
        assert len(state.generation_redo_stack) == 1

    def test_empty_history_returns_false(self) -> None:
        """undo with empty history returns False."""

        state = _state("hello", [_seg(0, 0, 5, [])])
        assert undo_generation(state) is False


class TestRedoGeneration:
    """Tests for redo_generation."""

    def test_reapplies_text(self) -> None:
        """redo re-applies the generated text."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        push_generation(state, "world", [100])
        post_gen_text = state.document_text

        undo_generation(state)
        assert state.document_text == "hello "

        result = redo_generation(state)
        assert result is True
        assert state.document_text == post_gen_text

    def test_pushes_to_history(self) -> None:
        """redo pushes snapshot back to history."""

        state = _state(
            "hello ",
            [_seg(0, 0, 6, [], target=True, token_count=5)],
        )
        push_generation(state, "world", [100])
        undo_generation(state)
        redo_generation(state)

        assert len(state.generation_history) == 1
        assert len(state.generation_redo_stack) == 0

    def test_extends_target_and_shifts_subsequent_segments(self) -> None:
        state = _state(
            "hello end",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
                _seg(2, 6, 9, []),
            ],
        )
        push_generation(state, "world", [100])
        undo_generation(state)

        assert redo_generation(state) is True
        assert state.segments[1].char_end == 11
        assert state.segments[2].char_start == 11
        assert state.segments[2].char_end == 14

    def test_empty_redo_returns_false(self) -> None:
        """redo with empty redo stack returns False."""

        state = _state("hello", [_seg(0, 0, 5, [])])
        assert redo_generation(state) is False


class TestUndoRedoCycles:
    """Tests for multi-step undo/redo interactions."""

    def test_undo_redo_undo_cycle(self) -> None:
        """undo → redo → undo returns to original state."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        original = state.document_text

        push_generation(state, "world", [100])
        undo_generation(state)
        assert state.document_text == original

        redo_generation(state)
        assert state.document_text == "hello world"

        undo_generation(state)
        assert state.document_text == original

    def test_double_generate_double_undo(self) -> None:
        """generate → generate → undo → undo reverts both."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )

        push_generation(state, "world", [100])
        assert state.document_text == "hello world"

        push_generation(state, "!!", [200])
        assert state.document_text == "hello world!!"

        undo_generation(state)
        assert state.document_text == "hello world"

        undo_generation(state)
        assert state.document_text == "hello "

    def test_generate_undo_new_generate_clears_redo(self) -> None:
        """generate → undo → new generate → redo stack is empty."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        push_generation(state, "world", [100])
        undo_generation(state)

        # New generation should clear redo
        push_generation(state, "earth", [200])
        assert len(state.generation_redo_stack) == 0
        assert redo_generation(state) is False
        assert state.document_text == "hello earth"

    def test_snapshot_segments_not_mutated_by_later_operations(self) -> None:
        """Snapshots deep-copy segments — later mutations don't corrupt history."""

        state = _state(
            "hello ",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),
                _seg(1, 6, 6, [], target=True, token_count=5),
            ],
        )
        push_generation(state, "world", [100])

        # Mutate current state segments
        state.segments[0].features[0].magnitude = 99

        # Undo should restore to pre-mutation snapshot
        undo_generation(state)
        assert state.segments[0].features[0].magnitude == 2


# ---------------------------------------------------------------------------
# GenerationManager tests (mocked model/generator)
# ---------------------------------------------------------------------------


class TestGenerationManager:
    """Tests for GenerationManager.start_generation and cancel."""

    def test_start_generation_yields_tokens(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """start_generation returns handle whose iterator yields tokens."""

        def mock_stream(**kwargs):
            cancel_event = kwargs.get("cancel_event")
            for token_id in [10, 20, 30]:
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield token_id

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )

        with patch(
            "scripts.completion_tui.generation.generate_completion_stream",
            side_effect=mock_stream,
        ):
            mgr = GenerationManager(
                model=MagicMock(),
                tokenizer=word_tokenizer,
                device=t.device("cpu"),
                boundaries=t.zeros(1, 1),
                precomputed_thresholds=None,
            )
            handle = mgr.start_generation(state)
            tokens = list(handle.token_iterator)

        assert tokens == [10, 20, 30]

    def test_cancel_stops_iteration(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """cancel() sets the event, causing the iterator to stop early."""

        def mock_stream(**kwargs):
            cancel_event = kwargs.get("cancel_event")
            for token_id in [10, 20, 30, 40, 50]:
                if cancel_event is not None and cancel_event.is_set():
                    return
                yield token_id

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, []),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )

        with patch(
            "scripts.completion_tui.generation.generate_completion_stream",
            side_effect=mock_stream,
        ):
            mgr = GenerationManager(
                model=MagicMock(),
                tokenizer=word_tokenizer,
                device=t.device("cpu"),
                boundaries=t.zeros(1, 1),
                precomputed_thresholds=None,
            )
            handle = mgr.start_generation(state)

            tokens = []
            for tok in handle.token_iterator:
                tokens.append(tok)
                if len(tokens) == 2:
                    mgr.cancel(handle)

        # Should have at most 3 tokens (2 before cancel + 1 during check)
        assert len(tokens) <= 3
        assert tokens[:2] == [10, 20]

    def test_handle_has_pre_generation_snapshot(
        self, word_tokenizer: WordMockTokenizer
    ) -> None:
        """GenerationHandle snapshot captures pre-generation state."""

        def mock_stream(**kwargs):
            yield from []

        state = _state(
            "hello world",
            [
                _seg(0, 0, 6, [_feat(10, 2)]),
                _seg(1, 6, 11, [], target=True, token_count=5),
            ],
        )

        with patch(
            "scripts.completion_tui.generation.generate_completion_stream",
            side_effect=mock_stream,
        ):
            mgr = GenerationManager(
                model=MagicMock(),
                tokenizer=word_tokenizer,
                device=t.device("cpu"),
                boundaries=t.zeros(1, 1),
                precomputed_thresholds=None,
            )
            handle = mgr.start_generation(state)

        assert handle.snapshot.document_text == "hello world"
        assert handle.target_segment_idx == 1
