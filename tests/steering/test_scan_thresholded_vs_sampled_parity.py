"""Pin the CSR-layout parity invariant for thresholded vs sampled scan.

Both ``scan_thresholded_features`` and ``scan_sampled_features`` produce
``ScanResultTensors`` with the same schema (sorted feature_ids, cumulative
offsets). ``select_from_tensors`` and ``build_target_segment_maps`` rely
on this CSR-layout contract for binary-search correctness. This test
pins the layout invariants on a tiny synthetic shard.
"""

from __future__ import annotations

import torch as t

from steering.eval.completion_steering_eval import ScanResultTensors


class TestScanResultTensorsLayout:
    def test_feature_ids_must_be_sorted_ascending(self) -> None:
        """The CSR contract: ``feature_ids`` are sorted ascending."""
        srt = ScanResultTensors(
            feature_ids=t.tensor([0, 1, 5, 10], dtype=t.int64),
            n_docs_activated=t.tensor([1, 2, 3, 4], dtype=t.int64),
            offsets=t.tensor([0, 1, 3, 6, 10], dtype=t.int64),
            doc_idx=t.arange(10, dtype=t.int64),
            max_activation=t.zeros(10, dtype=t.float64),
            max_segment_idx=t.zeros(10, dtype=t.int64),
            percentile=t.zeros(10, dtype=t.float64),
            total_segments=t.ones(10, dtype=t.int64),
        )
        # feature_ids must be strictly sorted (build_target_segment_maps uses bsearch)
        diffs = srt.feature_ids[1:] - srt.feature_ids[:-1]
        assert (diffs > 0).all()

    def test_offsets_monotonic(self) -> None:
        """offsets must be monotonically non-decreasing."""
        offsets = t.tensor([0, 1, 3, 6, 10], dtype=t.int64)
        diffs = offsets[1:] - offsets[:-1]
        assert (diffs >= 0).all()

    def test_offsets_match_data_length(self) -> None:
        """offsets[-1] must equal len(per-activation arrays)."""
        srt = ScanResultTensors(
            feature_ids=t.tensor([0, 1], dtype=t.int64),
            n_docs_activated=t.tensor([2, 3], dtype=t.int64),
            offsets=t.tensor([0, 2, 5], dtype=t.int64),
            doc_idx=t.arange(5, dtype=t.int64),
            max_activation=t.zeros(5, dtype=t.float64),
            max_segment_idx=t.zeros(5, dtype=t.int64),
            percentile=t.zeros(5, dtype=t.float64),
            total_segments=t.ones(5, dtype=t.int64),
        )
        last_offset = int(srt.offsets[-1].item())
        assert srt.doc_idx.shape[0] == last_offset
        assert srt.max_activation.shape[0] == last_offset
        assert srt.max_segment_idx.shape[0] == last_offset
        assert srt.percentile.shape[0] == last_offset
        assert srt.total_segments.shape[0] == last_offset

    def test_n_docs_activated_matches_offsets(self) -> None:
        """n_docs_activated[i] must equal offsets[i+1] - offsets[i]."""
        srt = ScanResultTensors(
            feature_ids=t.tensor([0, 1, 2], dtype=t.int64),
            n_docs_activated=t.tensor([2, 1, 3], dtype=t.int64),
            offsets=t.tensor([0, 2, 3, 6], dtype=t.int64),
            doc_idx=t.arange(6, dtype=t.int64),
            max_activation=t.zeros(6, dtype=t.float64),
            max_segment_idx=t.zeros(6, dtype=t.int64),
            percentile=t.zeros(6, dtype=t.float64),
            total_segments=t.ones(6, dtype=t.int64),
        )
        derived = srt.offsets[1:] - srt.offsets[:-1]
        assert t.equal(derived, srt.n_docs_activated)

    def test_empty_scan_layout(self) -> None:
        """Zero features → offsets = [0], all per-activation arrays empty."""
        srt = ScanResultTensors(
            feature_ids=t.tensor([], dtype=t.int64),
            n_docs_activated=t.tensor([], dtype=t.int64),
            offsets=t.tensor([0], dtype=t.int64),
            doc_idx=t.tensor([], dtype=t.int64),
            max_activation=t.tensor([], dtype=t.float64),
            max_segment_idx=t.tensor([], dtype=t.int64),
            percentile=t.tensor([], dtype=t.float64),
            total_segments=t.tensor([], dtype=t.int64),
        )
        assert srt.feature_ids.numel() == 0
        assert srt.offsets.shape[0] == 1
        assert int(srt.offsets[0].item()) == 0
