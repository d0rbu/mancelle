"""Tests for the single-segment doc skip behavior in _generate_completions.

As of issue #521, ``FeatureIndex.lookup`` raises ``KeyError`` on a missing
doc/shard, so a 0-segment / ``None`` result no longer occurs: a present doc
always has >= 1 segment, and an absent doc raises. The old "0 segments raises
ValueError" path is therefore dead and has been removed.

Verifies that:
  - Documents with only 1 segment are gracefully skipped (not crashed on).
  - A doc absent from the index (lookup raises KeyError) propagates the
    KeyError out of _generate_completions (corruption; issue #521).
  - Documents with >= 2 segments are processed normally.
  - Mixed batches (some single-segment, some multi-segment) produce the
    correct subset of completions.
  - The skip produces a warning log message.
  - Skipped docs don't affect the processing of subsequent docs/features.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch as t

from scripts.eval.completion_steering_eval import _generate_completions
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.completion_steering_eval import EvalMode, FeatureSelection
from steering.eval.multiseg_generator import CompletionSample, DocumentCompletion

# Passthrough config: segment_topk with k large enough to keep all features.
_PASSTHROUGH_CFG = ThresholdConfig(ThresholdType.segment_topk, 99999)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DOC_TEXTS = 100


def _make_fake_segment() -> MagicMock:
    seg = MagicMock()
    seg.segment_idx = 0
    return seg


def _make_doc_completion(
    feature_id: int, doc_idx: int, n_samples: int
) -> DocumentCompletion:
    samples = [
        CompletionSample(
            condition="ablated",
            text=f"text_{feature_id}_{doc_idx}_{i}",
            seed=i,
            bucket_index=None,
        )
        for i in range(n_samples)
    ]
    return DocumentCompletion(
        doc_idx=doc_idx,
        feature_id=feature_id,
        pre_segment_text="context",
        ground_truth_segment="truth",
        target_segment_idx=2,
        samples=samples,
    )


def _make_selections(
    feature_ids: list[int],
    docs_per_feature: list[list[int]],
) -> list[FeatureSelection]:
    selections = []
    for fid, docs in zip(feature_ids, docs_per_feature, strict=True):
        selections.append(
            FeatureSelection(
                feature_id=fid,
                neuronpedia_description=f"Feature {fid}",
                n_docs_activated=len(docs),
                selected_docs=docs,
            )
        )
    return selections


def _common_kwargs(
    selections: list[FeatureSelection],
    feature_index: MagicMock,
) -> dict:
    return {
        "selections": selections,
        "target_segment_maps": None,
        "doc_texts": [f"doc_text_{i}" for i in range(N_DOC_TEXTS)],
        "feature_index": feature_index,
        "val_shard_stem": "shard_0000",
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "mode": EvalMode.ABLATION,
        "boundaries": t.zeros(10),
        "feature_cfg": _PASSTHROUGH_CFG,
        "precomputed_thresholds": None,
        "device": t.device("cpu"),
        "num_buckets": 8,
        "samples_per_condition": 5,
        "temperature": 0.7,
        "top_k_sampling": 50,
        "bos_id": 2,
    }


# ---------------------------------------------------------------------------
# Tests: single-segment skip behavior
# ---------------------------------------------------------------------------


class TestSingleSegmentDocSkip:
    """Verify that single-segment documents are skipped gracefully."""

    def test_single_segment_doc_is_skipped(self) -> None:
        """A document with exactly 1 segment should be skipped, not crash."""
        selections = _make_selections([0], [[0]])
        feature_index = MagicMock()
        # Return a single segment for doc 0
        feature_index.lookup.return_value = [_make_fake_segment()]

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 0, "Single-segment doc should produce no completions"
        assert len(captured_calls) == 0, "generate_for_document should not be called"

    def test_missing_doc_propagates_keyerror(self) -> None:
        """A doc absent from the index is corruption: lookup raises KeyError and
        it propagates out of _generate_completions (issue #521)."""
        selections = _make_selections([0], [[0]])
        feature_index = MagicMock()
        feature_index.lookup.side_effect = KeyError(
            "Document 0 not found in shard 'shard_0000' (shard has 0 docs)"
        )
        with pytest.raises(KeyError, match="not found in shard"):
            _generate_completions(**_common_kwargs(selections, feature_index))

    def test_two_segment_doc_is_processed(self) -> None:
        """A document with exactly 2 segments should be processed normally."""
        selections = _make_selections([0], [[0]])
        feature_index = MagicMock()
        feature_index.lookup.return_value = [_make_fake_segment(), _make_fake_segment()]

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 1
        assert len(captured_calls) == 1
        assert captured_calls[0] == (0, 0)

    def test_three_segment_doc_is_processed(self) -> None:
        """A document with 3 segments should be processed normally."""
        selections = _make_selections([0], [[0]])
        feature_index = MagicMock()
        feature_index.lookup.return_value = [_make_fake_segment() for _ in range(3)]

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 1
        assert len(captured_calls) == 1


class TestMixedSegmentCounts:
    """Verify correct behavior when features have a mix of single- and
    multi-segment documents."""

    def test_mixed_segments_single_feature(self) -> None:
        """Feature with 5 docs where docs 1 and 3 have only 1 segment.

        Expected: 3 completions (docs 0, 2, 4 processed; docs 1, 3 skipped).
        """
        doc_indices = [0, 1, 2, 3, 4]
        single_seg_docs = {1, 3}
        selections = _make_selections([10], [doc_indices])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx in single_seg_docs:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 3
        processed_doc_idxs = [c[1] for c in captured_calls]
        assert processed_doc_idxs == [0, 2, 4]

    def test_mixed_segments_multiple_features(self) -> None:
        """Two features: feature 0 has all multi-seg docs, feature 1 has
        one single-seg doc. Verify each feature is handled independently."""
        # Feature 0: docs [0, 1, 2] all have 3 segments
        # Feature 1: docs [3, 4, 5] where doc 4 has 1 segment
        selections = _make_selections([0, 1], [[0, 1, 2], [3, 4, 5]])
        single_seg_docs = {4}
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx in single_seg_docs:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        # Feature 0: 3 docs processed; Feature 1: 2 docs processed (doc 4 skipped)
        assert len(result) == 5
        feature_0_docs = [c[1] for c in captured_calls if c[0] == 0]
        feature_1_docs = [c[1] for c in captured_calls if c[0] == 1]
        assert feature_0_docs == [0, 1, 2]
        assert feature_1_docs == [3, 5]

    def test_all_docs_single_segment_yields_empty(self) -> None:
        """If every doc for every feature has only 1 segment, result is empty."""
        selections = _make_selections([0, 1], [[0, 1], [2, 3]])
        feature_index = MagicMock()
        feature_index.lookup.return_value = [_make_fake_segment()]

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 0
        assert len(captured_calls) == 0


class TestSingleSegmentSkipDoesNotAffectSubsequent:
    """Verify that skipping a single-segment doc doesn't break subsequent
    processing within the same feature or across features."""

    def test_skip_first_doc_process_rest(self) -> None:
        """First doc is single-segment, remaining 3 should all be processed."""
        doc_indices = [10, 11, 12, 13]
        selections = _make_selections([5], [doc_indices])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx == 10:
                return [_make_fake_segment()]  # single segment
            return [_make_fake_segment() for _ in range(4)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 3
        assert [c[1] for c in captured_calls] == [11, 12, 13]

    def test_skip_last_doc_process_rest(self) -> None:
        """Last doc is single-segment, preceding docs should all be processed."""
        doc_indices = [10, 11, 12, 13]
        selections = _make_selections([5], [doc_indices])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx == 13:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(4)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 3
        assert [c[1] for c in captured_calls] == [10, 11, 12]

    def test_skip_middle_doc_doesnt_break_continuation(self) -> None:
        """Middle doc skipped — verify both before and after are processed."""
        doc_indices = [0, 1, 2]
        selections = _make_selections([0], [doc_indices])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx == 1:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 2
        assert [c[1] for c in captured_calls] == [0, 2]

    def test_skip_in_first_feature_doesnt_affect_second(self) -> None:
        """Skip occurs in feature 0's docs; feature 1's docs must all process."""
        # Feature 0: docs [0, 1] — doc 0 is single-segment
        # Feature 1: docs [2, 3] — both multi-segment
        selections = _make_selections([0, 1], [[0, 1], [2, 3]])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx == 0:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        assert len(result) == 3
        feature_0_docs = [c[1] for c in captured_calls if c[0] == 0]
        feature_1_docs = [c[1] for c in captured_calls if c[0] == 1]
        assert feature_0_docs == [1]
        assert feature_1_docs == [2, 3]


class TestSingleSegmentSkipWithInsertionMode:
    """Verify the skip works the same in INSERTION mode as ABLATION."""

    def test_insertion_mode_skips_single_segment(self) -> None:
        """Single-segment skip should apply regardless of EvalMode."""
        selections = _make_selections([0], [[0, 1, 2]])
        single_seg_docs = {1}
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx in single_seg_docs:
                return [_make_fake_segment()]
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        kwargs = _common_kwargs(selections, feature_index)
        kwargs["mode"] = EvalMode.INSERTION

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**kwargs)

        assert len(result) == 2
        assert [c[1] for c in captured_calls] == [0, 2]


class TestSingleSegmentSkipWarningLog:
    """Verify a warning is logged when a single-segment doc is skipped."""

    def test_warning_logged(self) -> None:
        """The skip should produce a logger.warning with doc index info."""
        selections = _make_selections([0], [[7]])
        feature_index = MagicMock()
        feature_index.lookup.return_value = [_make_fake_segment()]

        with (
            patch(
                "scripts.eval.completion_steering_eval.generate_for_document",
                side_effect=lambda **kw: _make_doc_completion(0, 7, 5),
            ),
            patch("scripts.eval.completion_steering_eval.logger") as mock_logger,
        ):
            _generate_completions(**_common_kwargs(selections, feature_index))

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Doc 7" in warning_msg
        assert "1 segment" in warning_msg


class TestSingleSegmentSkipWithTargetSegmentMaps:
    """Verify the skip interacts correctly with ablation target_segment_maps."""

    def test_skip_doc_with_target_map_entry(self) -> None:
        """Even if a doc has a target_segment_map entry, it should be skipped
        if it has only 1 segment (the map entry becomes irrelevant)."""
        selections = _make_selections([0], [[0, 1]])
        feature_index = MagicMock()

        def lookup_side_effect(shard_stem, doc_idx):
            if doc_idx == 0:
                return [_make_fake_segment()]  # single segment
            return [_make_fake_segment() for _ in range(3)]

        feature_index.lookup.side_effect = lookup_side_effect

        # doc 0 has a target_segment_map entry — should still be skipped
        target_maps = {0: {0: 2, 1: 1}}

        captured_calls: list[tuple[int, int]] = []

        def fake_generate(**kwargs):
            captured_calls.append((kwargs["feature_id"], kwargs["doc_idx"]))
            return _make_doc_completion(kwargs["feature_id"], kwargs["doc_idx"], 5)

        kwargs = _common_kwargs(selections, feature_index)
        kwargs["target_segment_maps"] = target_maps

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate,
        ):
            result = _generate_completions(**kwargs)

        assert len(result) == 1
        assert captured_calls == [(0, 1)]
