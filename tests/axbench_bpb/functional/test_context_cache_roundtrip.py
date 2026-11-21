# tests/axbench_bpb/functional/test_context_cache_roundtrip.py
"""Functional tests for context cache extraction + reload."""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_bpb.context_cache import (
    load_context_segments,
    serialize_context_segments,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.target_conditioning import ContextFeatureSegment


def test_serialize_load_roundtrip(tmp_path):
    original = (
        ContextFeatureSegment(
            feature_ids=(10, 11, 12), buckets=(3, 4, 5),
            char_start=0, char_end=20,
        ),
        ContextFeatureSegment(
            feature_ids=(99,), buckets=(7,),
            char_start=20, char_end=40,
        ),
    )
    path = tmp_path / "test.safetensors"
    serialize_context_segments(original, path)
    reloaded = load_context_segments(path)
    assert reloaded == original


def test_serialize_atomic_write(tmp_path):
    """Interrupt mid-write leaves no partial file."""
    original = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=0, char_end=1),
    )
    path = tmp_path / "atomic.safetensors"
    serialize_context_segments(original, path)
    # Final file exists, no .tmp sibling left
    assert path.exists()
    siblings = list(path.parent.glob("*"))
    assert all(not s.name.endswith(".tmp") for s in siblings)


def test_roundtrip_zero_length_segment(tmp_path):
    """Reviewer suggestion from Task 4.2: segment with char_start == char_end."""
    original = (
        ContextFeatureSegment(
            feature_ids=(5,), buckets=(2,),
            char_start=10, char_end=10,  # zero-length span
        ),
    )
    path = tmp_path / "zero_len.safetensors"
    serialize_context_segments(original, path)
    reloaded = load_context_segments(path)
    assert reloaded == original
    assert reloaded[0].char_start == reloaded[0].char_end == 10


def test_roundtrip_empty_segments_tuple(tmp_path):
    """Edge case: empty segment tuple."""
    original: tuple[ContextFeatureSegment, ...] = ()
    path = tmp_path / "empty.safetensors"
    serialize_context_segments(original, path)
    reloaded = load_context_segments(path)
    assert reloaded == original


def test_roundtrip_empty_features_in_segment(tmp_path):
    """Edge case: segment with no features."""
    original = (
        ContextFeatureSegment(
            feature_ids=(), buckets=(),
            char_start=0, char_end=50,
        ),
    )
    path = tmp_path / "empty_feats.safetensors"
    serialize_context_segments(original, path)
    reloaded = load_context_segments(path)
    assert reloaded == original
    assert reloaded[0].feature_ids == ()


def test_extract_prefix_features_delegates_to_existing_helpers(monkeypatch):
    """Verify extract_prefix_features calls the underlying extraction +
    discretization helpers rather than reimplementing char-level segmentation.

    The real ``steering.extract_features.extract_document_features`` returns a
    6-tuple of parallel lists/tensors:

        (all_feat_ids, all_feat_mags, positions, char_starts, char_ends, n_tokens)

    so the mock mirrors that shape (not the plan's aspirational
    ``list[Segment-like-object]`` shape, which predates verification).

    Uses monkeypatch to stub both helpers so we don't need a real SAE or
    Gemma model loaded."""
    from steering.eval.axbench_bpb import context_cache

    counts: dict[str, int] = {"extract": 0, "discretize": 0}
    captured_kwargs: dict[str, dict] = {}

    def _fake_extract(**kwargs):
        counts["extract"] += 1
        captured_kwargs["extract"] = kwargs
        all_feat_ids = [torch.tensor([10, 20], dtype=torch.int64)]
        all_feat_mags = [torch.tensor([0.5, 0.8], dtype=torch.float32)]
        positions = [0]
        char_starts = [0]
        char_ends = [10]
        n_tokens = 16
        return all_feat_ids, all_feat_mags, positions, char_starts, char_ends, n_tokens

    def _fake_discretize(fids, mags, boundaries):
        counts["discretize"] += 1
        # Return bucket tensor parallel to the (thresholded) feature list.
        return torch.tensor([3, 4], dtype=torch.int64)

    # Patch at the import site — context_cache.py binds both names at
    # module-top via ``from ... import ...``, so these local names are
    # what the implementation actually calls.
    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)
    monkeypatch.setattr(context_cache, "discretize_activations", _fake_discretize)

    segs = context_cache.extract_prefix_features(
        prefix_text="x",
        base_model=None,
        base_sae=None,
        boundaries=torch.zeros(64, 9),
        segment_size=128,
        sae_layer=8,
        extraction_max_tokens=2048,
        extraction_threshold=0.0,
    )

    assert counts["extract"] == 1, "extract_document_features should be called once"
    assert counts["discretize"] == 1, (
        "discretize_activations should be called once for the single non-empty segment"
    )
    assert len(segs) == 1
    assert segs[0].feature_ids == (10, 20)
    assert segs[0].buckets == (3, 4)
    assert segs[0].char_start == 0
    assert segs[0].char_end == 10

    # Sanity: the wrapper threads the plan's kwargs through to the real
    # extractor so future wiring in Task 4.5 stays faithful to training
    # pooling/segment-mode.
    ek = captured_kwargs["extract"]
    assert ek["text"] == "x"
    assert ek["layer"] == 8
    assert ek["segment_size"] == 128
    assert ek["context_window"] == 2048


def test_extract_prefix_features_applies_magnitude_threshold(monkeypatch):
    """Features with magnitude <= extraction_threshold are filtered out
    before discretization.

    The plan's ``extraction_threshold`` is a simple scalar magnitude cutoff
    (see AxBenchBpbConfig.extraction_threshold default of 0.0, which keeps
    all non-zero activations). This test asserts that behaviour by passing
    a threshold between the two fake magnitudes and checking only the
    higher-mag feature survives."""
    from steering.eval.axbench_bpb import context_cache

    def _fake_extract(**kwargs):
        return (
            [torch.tensor([10, 20], dtype=torch.int64)],
            [torch.tensor([0.1, 0.9], dtype=torch.float32)],
            [0],
            [0],
            [10],
            16,
        )

    discretize_inputs = {}

    def _fake_discretize(fids, mags, boundaries):
        discretize_inputs["fids"] = fids.clone()
        discretize_inputs["mags"] = mags.clone()
        return torch.tensor([7], dtype=torch.int64)

    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)
    monkeypatch.setattr(context_cache, "discretize_activations", _fake_discretize)

    segs = context_cache.extract_prefix_features(
        prefix_text="x",
        base_model=None,
        base_sae=None,
        boundaries=torch.zeros(64, 9),
        segment_size=128,
        sae_layer=8,
        extraction_max_tokens=2048,
        extraction_threshold=0.5,
    )

    # Only the mag=0.9 feature (id=20) should survive the threshold.
    assert discretize_inputs["fids"].tolist() == [20]
    assert discretize_inputs["mags"].tolist() == pytest.approx([0.9])
    assert segs[0].feature_ids == (20,)
    assert segs[0].buckets == (7,)


def test_threshold_and_discretize_features_raises_on_empty_signal() -> None:
    from steering.eval.axbench_bpb import context_cache

    with pytest.raises(AxBenchBpbError, match="empty sparse feature signal"):
        context_cache.threshold_and_discretize_features(
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.float32),
            boundaries=torch.zeros(4, 1),
            extraction_threshold=0.0,
        )


def test_threshold_and_discretize_features_raises_on_no_survivors() -> None:
    from steering.eval.axbench_bpb import context_cache

    with pytest.raises(AxBenchBpbError, match="zero features above"):
        context_cache.threshold_and_discretize_features(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            boundaries=torch.zeros(4, 1),
            extraction_threshold=0.5,
        )


@pytest.mark.parametrize(
    "feature_ids,feature_mags,match",
    [
        (
            torch.tensor([[1, 2]], dtype=torch.int64),
            torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            "one-dimensional",
        ),
        (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0.1], dtype=torch.float32),
            "mismatched",
        ),
    ],
)
def test_threshold_and_discretize_features_raises_on_malformed_signal(
    feature_ids: torch.Tensor,
    feature_mags: torch.Tensor,
    match: str,
) -> None:
    from steering.eval.axbench_bpb import context_cache

    with pytest.raises(AxBenchBpbError, match=match):
        context_cache.threshold_and_discretize_features(
            feature_ids,
            feature_mags,
            boundaries=torch.zeros(4, 1),
            extraction_threshold=0.0,
        )


@pytest.mark.parametrize(
    "feature_ids,feature_mags,boundaries,match",
    [
        (
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32),
            torch.zeros(4, 1),
            "integer dtype",
        ),
        (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            torch.zeros(4, 1),
            "floating-point dtype",
        ),
        (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([float("inf")], dtype=torch.float32),
            torch.zeros(4, 1),
            "non-finite",
        ),
        (
            torch.tensor([-1], dtype=torch.int64),
            torch.tensor([0.1], dtype=torch.float32),
            torch.zeros(4, 1),
            "non-negative",
        ),
        (
            torch.tensor([4], dtype=torch.int64),
            torch.tensor([0.1], dtype=torch.float32),
            torch.zeros(4, 1),
            "exceed bucket boundary",
        ),
        (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0.1], dtype=torch.float32),
            torch.zeros(4),
            "two-dimensional",
        ),
    ],
)
def test_threshold_and_discretize_features_raises_on_corrupt_signal_state(
    feature_ids: torch.Tensor,
    feature_mags: torch.Tensor,
    boundaries: torch.Tensor,
    match: str,
) -> None:
    from steering.eval.axbench_bpb import context_cache

    with pytest.raises(AxBenchBpbError, match=match):
        context_cache.threshold_and_discretize_features(
            feature_ids,
            feature_mags,
            boundaries=boundaries,
            extraction_threshold=0.0,
        )


def test_extract_prefix_features_raises_on_empty_segment_signal(monkeypatch):
    from steering.eval.axbench_bpb import context_cache

    def _fake_extract(**kwargs):
        return (
            [torch.tensor([], dtype=torch.int64)],
            [torch.tensor([], dtype=torch.float32)],
            [0],
            [100],
            [200],
            128,
        )

    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)

    with pytest.raises(AxBenchBpbError, match="empty sparse feature signal"):
        context_cache.extract_prefix_features(
            prefix_text="x",
            base_model=None,
            base_sae=None,
            boundaries=torch.zeros(64, 9),
            segment_size=128,
            sae_layer=8,
            extraction_max_tokens=2048,
            extraction_threshold=0.5,
        )


def test_extract_prefix_features_raises_on_fully_filtered_segment(monkeypatch):
    from steering.eval.axbench_bpb import context_cache

    def _fake_extract(**kwargs):
        return (
            [torch.tensor([10], dtype=torch.int64)],
            [torch.tensor([0.2], dtype=torch.float32)],
            [0],
            [0],
            [100],
            128,
        )

    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)

    with pytest.raises(AxBenchBpbError, match="zero features above"):
        context_cache.extract_prefix_features(
            prefix_text="x",
            base_model=None,
            base_sae=None,
            boundaries=torch.zeros(64, 9),
            segment_size=128,
            sae_layer=8,
            extraction_max_tokens=2048,
            extraction_threshold=0.5,
        )


def test_extract_prefix_features_matches_legacy_nonempty_threshold_behavior(
    monkeypatch,
) -> None:
    from steering.eval.axbench_bpb import context_cache

    all_feat_ids = [
        torch.tensor([10, 11], dtype=torch.int64),
        torch.tensor([20, 21], dtype=torch.int64),
        torch.tensor([30, 31], dtype=torch.int64),
        torch.tensor([40], dtype=torch.int64),
    ]
    all_feat_mags = [
        torch.tensor([0.1, 0.8], dtype=torch.float32),
        torch.tensor([0.2, 0.9], dtype=torch.float32),
        torch.tensor([0.5, 0.6], dtype=torch.float32),
        torch.tensor([1.2], dtype=torch.float32),
    ]
    char_starts = [0, 10, 20, 30]
    char_ends = [10, 20, 30, 40]
    threshold = 0.5

    def _fake_extract(**kwargs):
        return all_feat_ids, all_feat_mags, [0, 1, 2, 3], char_starts, char_ends, 4

    def _bucket(fids, mags, boundaries):
        return (fids.to(torch.int64) + (mags.to(torch.float32) > 1.0).to(torch.int64)).to(
            torch.int64
        )

    def _legacy_segments() -> tuple[ContextFeatureSegment, ...]:
        segments: list[ContextFeatureSegment] = []
        for seg_fids, seg_mags, char_start, char_end in zip(
            all_feat_ids, all_feat_mags, char_starts, char_ends, strict=True,
        ):
            mask = seg_mags > threshold
            kept_fids = seg_fids[mask]
            kept_mags = seg_mags[mask]
            bucket_tensor = _bucket(
                kept_fids.to(torch.int64),
                kept_mags.to(torch.float32),
                torch.zeros(64, 1),
            )
            feature_ids = tuple(int(x) for x in kept_fids.tolist())
            buckets = tuple(int(x) for x in bucket_tensor.tolist())
            segments.append(
                ContextFeatureSegment(
                    feature_ids=feature_ids,
                    buckets=buckets,
                    char_start=int(char_start),
                    char_end=int(char_end),
                )
            )
        return tuple(segments)

    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)
    monkeypatch.setattr(context_cache, "discretize_activations", _bucket)

    actual = context_cache.extract_prefix_features(
        prefix_text="x",
        base_model=None,
        base_sae=None,
        boundaries=torch.zeros(64, 1),
        segment_size=128,
        sae_layer=8,
        extraction_max_tokens=2048,
        extraction_threshold=threshold,
    )

    assert actual == _legacy_segments()


def test_extract_prefix_features_raises_on_mismatched_feature_magnitude_counts(
    monkeypatch,
) -> None:
    from steering.eval.axbench_bpb import context_cache

    def _fake_extract(**kwargs):
        return (
            [torch.tensor([1, 2], dtype=torch.int64)],
            [torch.tensor([0.9], dtype=torch.float32)],
            [0],
            [0],
            [10],
            1,
        )

    monkeypatch.setattr(context_cache, "extract_document_features", _fake_extract)

    with pytest.raises(AxBenchBpbError, match="mismatched feature id/magnitude"):
        context_cache.extract_prefix_features(
            prefix_text="x",
            base_model=None,
            base_sae=None,
            boundaries=torch.zeros(64, 1),
            segment_size=128,
            sae_layer=8,
            extraction_max_tokens=2048,
            extraction_threshold=0.0,
        )
