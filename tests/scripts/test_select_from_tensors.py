"""Tests for tensor-native completion-steering feature selection.

Verifies random selection behavior, RawFeatureSelection output, determinism,
edge cases, and the auto-resume logic in the pipeline script.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch as t
import yaml

from steering.eval.completion_steering_eval import (
    EvalMode,
    RawFeatureSelection,
    ScanResultTensors,
    build_target_segment_maps,
    save_scan_results,
    select_from_tensors,
)
from tests.steering._completion_steering_reference import (
    scan_results_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scan_tensors(
    feature_ids: list[int],
    n_docs_per_feature: list[int],
    seed: int = 0,
) -> ScanResultTensors:
    """Build ScanResultTensors with deterministic but realistic data.

    Each feature gets n_docs_per_feature[i] doc activations with predictable
    doc_idx values (unique across features to avoid ambiguity).
    """
    rng = random.Random(seed)
    fids = t.tensor(sorted(feature_ids), dtype=t.int64)  # must be sorted for CSR
    n_docs = t.tensor(
        [n_docs_per_feature[sorted(feature_ids).index(fid)] for fid in fids.tolist()],
        dtype=t.int64,
    )

    total = int(n_docs.sum().item())
    offsets = t.zeros(len(fids) + 1, dtype=t.int64)
    offsets[1:] = n_docs.cumsum(0)

    # Generate unique doc_idx values across all features
    all_doc_idxs: list[int] = []
    doc_counter = 0
    for i in range(len(fids)):
        count = int(n_docs[i].item())
        for _ in range(count):
            all_doc_idxs.append(doc_counter)
            doc_counter += 1

    doc_idx = t.tensor(all_doc_idxs, dtype=t.int64)
    max_activation = t.tensor(
        [rng.uniform(0.1, 10.0) for _ in range(total)], dtype=t.float32
    )
    max_segment_idx = t.tensor([rng.randint(1, 5) for _ in range(total)], dtype=t.int64)
    percentile = t.tensor(
        [rng.uniform(0.01, 0.99) for _ in range(total)], dtype=t.float32
    )
    total_segments = t.tensor([rng.randint(2, 8) for _ in range(total)], dtype=t.int64)

    return ScanResultTensors(
        feature_ids=fids,
        n_docs_activated=n_docs,
        offsets=offsets,
        doc_idx=doc_idx,
        max_activation=max_activation,
        max_segment_idx=max_segment_idx,
        percentile=percentile,
        total_segments=total_segments,
    )


# ---------------------------------------------------------------------------
# Tests: select_from_tensors returns RawFeatureSelection
# ---------------------------------------------------------------------------


class TestSelectFromTensorsReturnType:
    """Verify select_from_tensors returns RawFeatureSelection (no description)."""

    def test_returns_raw_feature_selection(self):
        srt = _build_scan_tensors(
            feature_ids=[10, 20, 30], n_docs_per_feature=[5, 3, 8]
        )
        result = select_from_tensors(srt, 2, 3, EvalMode.ABLATION)
        for r in result:
            assert isinstance(r, RawFeatureSelection)

    def test_raw_selection_has_no_description_field(self):
        srt = _build_scan_tensors(feature_ids=[10], n_docs_per_feature=[5])
        result = select_from_tensors(srt, 1, 3, EvalMode.ABLATION)
        assert not hasattr(result[0], "neuronpedia_description")

    def test_raw_selection_fields(self):
        srt = _build_scan_tensors(feature_ids=[10, 20], n_docs_per_feature=[5, 3])
        result = select_from_tensors(srt, 2, 3, EvalMode.ABLATION)
        for r in result:
            assert isinstance(r.feature_id, int)
            assert isinstance(r.n_docs_activated, int)
            assert isinstance(r.selected_docs, list)


# ---------------------------------------------------------------------------
# Tests: random selection behavior
# ---------------------------------------------------------------------------


class TestRandomSelectionBehavior:
    """Test that selection is truly random (not top-k by n_docs_activated)."""

    def test_not_always_top_n_docs(self):
        """With many seeds, random selection should NOT always pick highest n_docs."""
        # Feature 40 has the most docs; if selection were top-k, it'd always be picked
        srt = _build_scan_tensors(
            feature_ids=[10, 20, 30, 40, 50],
            n_docs_per_feature=[1, 2, 3, 100, 5],
        )
        seeds_without_40 = 0
        for seed in range(50):
            result = select_from_tensors(srt, 1, 1, EvalMode.ABLATION, seed=seed)
            if result[0].feature_id != 40:
                seeds_without_40 += 1
        # With random selection, we should miss feature 40 at least sometimes
        assert seeds_without_40 > 0, "Selection always picks top n_docs — not random"

    def test_deterministic_same_seed(self):
        srt = _build_scan_tensors(
            feature_ids=[1, 2, 3, 4, 5],
            n_docs_per_feature=[20, 15, 10, 25, 8],
        )
        r1 = select_from_tensors(srt, 3, 5, EvalMode.ABLATION, seed=99)
        r2 = select_from_tensors(srt, 3, 5, EvalMode.ABLATION, seed=99)
        assert [s.feature_id for s in r1] == [s.feature_id for s in r2]
        assert [s.selected_docs for s in r1] == [s.selected_docs for s in r2]

    def test_different_seeds_differ(self):
        srt = _build_scan_tensors(
            feature_ids=[1, 2, 3, 4, 5],
            n_docs_per_feature=[50, 50, 50, 50, 50],
        )
        r1 = select_from_tensors(srt, 3, 5, EvalMode.ABLATION, seed=1)
        r2 = select_from_tensors(srt, 3, 5, EvalMode.ABLATION, seed=2)
        fids1 = [s.feature_id for s in r1]
        fids2 = [s.feature_id for s in r2]
        assert fids1 != fids2

    def test_ablation_basic(self):
        srt = _build_scan_tensors(
            feature_ids=[10, 20, 30, 40, 50],
            n_docs_per_feature=[8, 3, 12, 5, 1],
        )
        result = select_from_tensors(srt, 3, 2, EvalMode.ABLATION)
        assert len(result) == 3
        # All selected feature IDs must be valid
        valid_fids = {10, 20, 30, 40, 50}
        for r in result:
            assert r.feature_id in valid_fids

    def test_insertion_basic(self):
        srt = _build_scan_tensors(
            feature_ids=[100, 200, 300],
            n_docs_per_feature=[3, 7, 2],
        )
        all_doc_indices = set(range(30))
        result = select_from_tensors(srt, 2, 4, EvalMode.INSERTION, all_doc_indices)
        assert len(result) == 2

    def test_insertion_requires_all_doc_indices(self):
        srt = _build_scan_tensors(feature_ids=[1], n_docs_per_feature=[3])
        with pytest.raises(ValueError, match="all_doc_indices"):
            select_from_tensors(srt, 1, 2, EvalMode.INSERTION)

    def test_top_k_exceeds_n_features(self):
        srt = _build_scan_tensors(feature_ids=[5, 10], n_docs_per_feature=[3, 7])
        result = select_from_tensors(srt, 100, 5, EvalMode.ABLATION)
        assert len(result) == 2

    def test_empty_scan_tensors(self):
        srt = ScanResultTensors(
            feature_ids=t.empty(0, dtype=t.int64),
            n_docs_activated=t.empty(0, dtype=t.int64),
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        result = select_from_tensors(srt, 10, 5, EvalMode.ABLATION)
        assert len(result) == 0

    def test_top_k_zero(self):
        srt = _build_scan_tensors(feature_ids=[1, 2], n_docs_per_feature=[5, 3])
        result = select_from_tensors(srt, 0, 5, EvalMode.ABLATION)
        assert len(result) == 0

    def test_docs_per_feature_caps_selection(self):
        srt = _build_scan_tensors(feature_ids=[1], n_docs_per_feature=[20])
        result = select_from_tensors(srt, 1, 3, EvalMode.ABLATION)
        assert len(result[0].selected_docs) == 3

    def test_docs_per_feature_exceeds_available(self):
        srt = _build_scan_tensors(feature_ids=[1], n_docs_per_feature=[2])
        result = select_from_tensors(srt, 1, 100, EvalMode.ABLATION)
        assert len(result[0].selected_docs) == 2

    def test_single_feature_single_doc(self):
        srt = _build_scan_tensors(feature_ids=[999], n_docs_per_feature=[1])
        result = select_from_tensors(srt, 1, 1, EvalMode.ABLATION)
        assert len(result) == 1
        assert result[0].feature_id == 999

    def test_feature_with_zero_docs(self):
        srt = ScanResultTensors(
            feature_ids=t.tensor([5, 10], dtype=t.int64),
            n_docs_activated=t.tensor([0, 3], dtype=t.int64),
            offsets=t.tensor([0, 0, 3], dtype=t.int64),
            doc_idx=t.tensor([0, 1, 2], dtype=t.int64),
            max_activation=t.tensor([1.0, 2.0, 3.0], dtype=t.float32),
            max_segment_idx=t.tensor([1, 1, 1], dtype=t.int64),
            percentile=t.tensor([0.5, 0.6, 0.7], dtype=t.float32),
            total_segments=t.tensor([3, 3, 3], dtype=t.int64),
        )
        result = select_from_tensors(srt, 2, 5, EvalMode.ABLATION)
        assert len(result) == 2
        # The feature with 0 docs should have empty selected_docs
        by_fid = {r.feature_id: r for r in result}
        assert by_fid[5].selected_docs == []

    def test_csr_slice_matches_dict_conversion(self):
        """For each feature, the doc_idx extracted from CSR matches scan_results_to_dict."""
        srt = _build_scan_tensors(
            feature_ids=[7, 13, 42],
            n_docs_per_feature=[4, 6, 2],
        )
        d = scan_results_to_dict(srt)

        for i, fid in enumerate(srt.feature_ids.tolist()):
            start = int(srt.offsets[i].item())
            end = int(srt.offsets[i + 1].item())
            csr_docs = srt.doc_idx[start:end].tolist()
            dict_docs = [a.doc_idx for a in d[fid].doc_activations]
            assert csr_docs == dict_docs


class TestRandomSelectionLargeScale:
    """Test with realistic feature counts."""

    def test_1000_features_random_selection(self):
        n_feats = 1000
        rng = random.Random(12345)
        n_docs = [rng.randint(1, 200) for _ in range(n_feats)]
        fids = list(range(n_feats))
        srt = _build_scan_tensors(feature_ids=fids, n_docs_per_feature=n_docs, seed=99)

        result = select_from_tensors(srt, 10, 10, EvalMode.ABLATION, seed=42)
        assert len(result) == 10
        # All selected must be valid
        for r in result:
            assert 0 <= r.feature_id < n_feats

    def test_1000_features_insertion(self):
        n_feats = 1000
        rng = random.Random(54321)
        n_docs = [rng.randint(1, 50) for _ in range(n_feats)]
        fids = list(range(n_feats))
        srt = _build_scan_tensors(feature_ids=fids, n_docs_per_feature=n_docs, seed=77)

        total_doc_count = sum(n_docs) + 500
        all_doc_indices = set(range(total_doc_count))
        result = select_from_tensors(
            srt, 5, 8, EvalMode.INSERTION, all_doc_indices, seed=42
        )
        assert len(result) == 5

    def test_statistical_uniformity(self):
        """Each feature should be selected roughly equally over many seeds."""
        n_feats = 10
        srt = _build_scan_tensors(
            feature_ids=list(range(n_feats)),
            n_docs_per_feature=[5] * n_feats,
        )
        counts: dict[int, int] = dict.fromkeys(range(n_feats), 0)
        n_trials = 500
        for seed in range(n_trials):
            result = select_from_tensors(srt, 1, 1, EvalMode.ABLATION, seed=seed)
            counts[result[0].feature_id] += 1

        # With 500 trials picking 1 of 10, expected ~50 each.
        # Allow wide margin (> 10 each) to avoid flaky tests.
        for fid, count in counts.items():
            assert count > 10, f"Feature {fid} selected only {count}/{n_trials} times"


class TestInsertionEdgeCases:
    """Edge cases specific to insertion mode."""

    def test_insertion_all_docs_activated(self):
        srt = _build_scan_tensors(feature_ids=[1], n_docs_per_feature=[5])
        all_doc_indices = set(range(5))
        result = select_from_tensors(srt, 1, 3, EvalMode.INSERTION, all_doc_indices)
        assert result[0].selected_docs == []

    def test_insertion_sorted_candidate_docs(self):
        srt = _build_scan_tensors(feature_ids=[1], n_docs_per_feature=[3])
        all_doc_indices = set(range(20))
        result = select_from_tensors(srt, 1, 100, EvalMode.INSERTION, all_doc_indices)
        # All non-activated docs should be returned
        activated = set(srt.doc_idx[:3].tolist())
        for doc in result[0].selected_docs:
            assert doc not in activated


# ---------------------------------------------------------------------------
# Auto-resume tests
# ---------------------------------------------------------------------------


class TestAutoResume:
    """Verify that stage=all skips stages whose output files exist."""

    def _make_scan_output(self, output_dir: Path) -> Path:
        srt = _build_scan_tensors(feature_ids=[1, 2, 3], n_docs_per_feature=[5, 3, 8])
        path = output_dir / "validation_scan.safetensors"
        save_scan_results(srt, path)
        return path

    def _make_selections_output(self, output_dir: Path) -> Path:
        selections = [
            {
                "feature_id": 3,
                "neuronpedia_description": "test feature 3",
                "n_docs_activated": 8,
                "selected_docs": [0, 1, 2],
            },
        ]
        path = output_dir / "feature_selections.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(selections, f, default_flow_style=False)
        return path

    def _make_completions_output(self, output_dir: Path) -> Path:
        completions = [
            {
                "doc_idx": 0,
                "feature_id": 3,
                "pre_segment_text": "hello",
                "ground_truth_segment": "world",
                "target_segment_idx": 1,
                "samples": [
                    {
                        "condition": "real",
                        "text": "foo",
                        "seed": 42,
                        "bucket_index": None,
                    }
                ],
                "ground_truth_signals": {
                    "feature_signals": {3: 2},
                    "n_features_active": 1,
                    "avg_magnitude": 2.0,
                    "max_magnitude": 2,
                    "min_magnitude": 2,
                    "feature_of_interest_magnitude": 2,
                },
            }
        ]
        path = output_dir / "completions.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(completions, f, default_flow_style=False)
        return path

    def _make_judge_scores_output(self, output_dir: Path) -> Path:
        scores = {3: {"real": [{"coherence": 4, "feature_alignment": 3}]}}
        path = output_dir / "judge_scores.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(scores, f, default_flow_style=False)
        return path

    def test_scan_output_file_loadable(self, tmp_path):
        from steering.eval.completion_steering_eval import load_scan_results

        self._make_scan_output(tmp_path)
        srt = load_scan_results(tmp_path / "validation_scan.safetensors")
        assert len(srt.feature_ids) == 3

    def test_selections_output_file_loadable(self, tmp_path):
        from scripts.eval.completion_steering_eval import _load_selections_yaml

        self._make_selections_output(tmp_path)
        selections = _load_selections_yaml(tmp_path / "feature_selections.yaml")
        assert len(selections) == 1
        assert selections[0].feature_id == 3

    def test_completions_output_file_loadable(self, tmp_path):
        from scripts.eval.completion_steering_eval import _load_completions_yaml

        self._make_completions_output(tmp_path)
        completions = _load_completions_yaml(tmp_path / "completions.yaml")
        assert len(completions) == 1
        assert completions[0].feature_id == 3
        assert completions[0].ground_truth_signals is not None

    def test_completions_without_ground_truth_signals_raises(self, tmp_path):
        """Old-format YAML without ground_truth_signals should raise KeyError."""
        from scripts.eval.completion_steering_eval import _load_completions_yaml

        old_format = [
            {
                "doc_idx": 0,
                "feature_id": 3,
                "pre_segment_text": "hello",
                "ground_truth_segment": "world",
                "target_segment_idx": 1,
                "samples": [{"condition": "real", "text": "foo", "seed": 42}],
            }
        ]
        path = tmp_path / "old_completions.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(old_format, f)
        with pytest.raises(KeyError):
            _load_completions_yaml(path)

    def test_all_outputs_exist_structure(self, tmp_path):
        self._make_scan_output(tmp_path)
        self._make_selections_output(tmp_path)
        self._make_completions_output(tmp_path)
        self._make_judge_scores_output(tmp_path)

        assert (tmp_path / "validation_scan.safetensors").exists()
        assert (tmp_path / "feature_selections.yaml").exists()
        assert (tmp_path / "completions.yaml").exists()
        assert (tmp_path / "judge_scores.yaml").exists()


# ---------------------------------------------------------------------------
# build_target_segment_maps equivalence tests
# ---------------------------------------------------------------------------


class TestBuildTargetSegmentMaps:
    """Prove build_target_segment_maps matches the old scan_results_to_dict path."""

    def test_matches_dict_conversion(self):
        srt = _build_scan_tensors(
            feature_ids=[10, 20, 30, 40, 50],
            n_docs_per_feature=[8, 3, 12, 5, 1],
        )
        d = scan_results_to_dict(srt)
        selected_fids = [30, 10, 50]

        old_maps: dict[int, dict[int, int]] = {}
        for fid in selected_fids:
            if fid in d:
                old_maps[fid] = {
                    act.doc_idx: act.max_segment_idx for act in d[fid].doc_activations
                }

        new_maps = build_target_segment_maps(srt, selected_fids)
        assert old_maps.keys() == new_maps.keys()
        for fid, old_map in old_maps.items():
            assert old_map == new_maps[fid]

    def test_feature_not_in_scan(self):
        srt = _build_scan_tensors(feature_ids=[10, 20], n_docs_per_feature=[3, 5])
        result = build_target_segment_maps(srt, [10, 999])
        assert 10 in result
        assert 999 not in result

    def test_empty_scan_tensors(self):
        srt = ScanResultTensors(
            feature_ids=t.empty(0, dtype=t.int64),
            n_docs_activated=t.empty(0, dtype=t.int64),
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        assert build_target_segment_maps(srt, [1, 2, 3]) == {}

    def test_empty_feature_list(self):
        srt = _build_scan_tensors(feature_ids=[1, 2], n_docs_per_feature=[5, 3])
        assert build_target_segment_maps(srt, []) == {}

    def test_large_scale_equivalence(self):
        rng = random.Random(42)
        n_feats = 1000
        n_docs = [rng.randint(1, 100) for _ in range(n_feats)]
        fids = list(range(n_feats))
        srt = _build_scan_tensors(feature_ids=fids, n_docs_per_feature=n_docs, seed=42)

        selected = rng.sample(fids, 10)
        d = scan_results_to_dict(srt)

        old_maps: dict[int, dict[int, int]] = {}
        for fid in selected:
            if fid in d:
                old_maps[fid] = {
                    act.doc_idx: act.max_segment_idx for act in d[fid].doc_activations
                }

        new_maps = build_target_segment_maps(srt, selected)
        assert old_maps.keys() == new_maps.keys()
        for fid, old_map in old_maps.items():
            assert old_map == new_maps[fid]
