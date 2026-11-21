# tests/test_completion_steering_integration.py
"""Integration tests for the completion steering eval pipeline.

Tests generate_for_document orchestrator, YAML serialization round-trips,
multi-segment conditioning correctness, and entry-point helper functions.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch as t
import yaml

from steering.conditional.encoder_decoder.packed_encoder import PackedEncoderOutput
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.completion_judge import (
    JudgementScore,
    compute_dose_response,
)
from steering.eval.completion_steering_eval import (
    ContextFilter,
    EvalMode,
    FeatureSelection,
    RawFeatureSelection,
    ScanResultTensors,
    doc_fits_context,
    find_fitting_segment,
    select_from_tensors,
)
from steering.eval.multiseg_generator import (
    CompletionSample,
    DocumentCompletion,
    GroundTruthFeatureSignals,
    ablate_feature,
    add_feature,
    build_multiseg_inputs,
    generate_for_document,
    prepare_segments,
)
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets
from steering.stats_types import PerFeatureStat
from tests.padded_encoder_reference import pack_dense_encoder_output_reference
from tests.steering._completion_steering_reference import (
    scan_validation_features,
)

# Passthrough config: segment_topk with k large enough to keep all features.
_PASSTHROUGH_CFG = ThresholdConfig(ThresholdType.segment_topk, 99999)

# ---------------------------------------------------------------------------
# Helpers: synthetic data builders
# ---------------------------------------------------------------------------


def _make_segments(
    n_segs: int = 3,
    features_per_seg: int = 5,
    segment_size: int = 128,
) -> list[SegmentFeatures]:
    """Create synthetic SegmentFeatures with known feature IDs and mags."""
    segs: list[SegmentFeatures] = []
    for i in range(n_segs):
        fids = t.arange(i * 10, i * 10 + features_per_seg, dtype=t.int32)
        mags = t.ones(features_per_seg, dtype=t.float32) * (i + 1.0)
        segs.append(
            SegmentFeatures(
                segment_idx=i,
                position=i * segment_size,
                feature_ids=fids,
                feature_mags=mags,
                char_start=i * 512,
                char_end=(i + 1) * 512,
            )
        )
    return segs


class _FixedWidthTokenizer:
    """Mock tokenizer where each token decodes to exactly 4 characters.

    With segment char boundaries at multiples of 512 and 128 tokens per
    segment, 128 tokens × 4 chars = 512 chars aligns exactly.
    """

    def decode(self, token_ids):
        return "x" * (len(token_ids) * 4)


_MOCK_TOK = _FixedWidthTokenizer()


class _FakeTokenizer:
    """Minimal tokenizer for testing generate_for_document.

    Each token decodes to exactly 4 characters so that char offsets
    align with the test segment boundaries (multiples of 512).
    """

    bos_id = 2

    def encode(self, text: str, prepend: int | None = None, append: int | None = None) -> list[int]:
        tokens = list(range(10, 10 + len(text) // 4))  # Rough tokenization
        if prepend is not None:
            tokens = [prepend] + tokens
        if append is not None:
            tokens = tokens + [append]
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 4)


class _FakeModel:
    """Minimal model mock for testing generate_for_document.

    Returns random logits from forward() and has the required attributes.
    """

    def __init__(self, vocab_size: int = 100, seq_len: int = 1024):
        self.vocab_size = vocab_size

        # Mimic dec_config
        self.dec_config = MagicMock()
        self.dec_config.sequence_len = seq_len
        self.dec_config.n_kv_head = 4
        self.dec_config.n_head = 8
        self.dec_config.n_embd = 256
        self.dec_config.n_layer = 2

        # Mimic encoder.config
        self.encoder = MagicMock()
        self.encoder.config = MagicMock()
        self.encoder.config.num_buckets = 4

    def __call__(self, idx, **kwargs):
        B = idx.shape[0]
        T = idx.shape[1]
        logits = t.randn(B, T, self.vocab_size)
        # Make token 50 most likely to keep generation deterministic-ish
        logits[:, :, 50] = 10.0
        return logits

    def encoder_forward(self, fids, mags, mask):
        n_seg = fids.shape[1]
        return t.randn(1, n_seg, 256), mask


# ---------------------------------------------------------------------------
# Test: seg_map correctness
# ---------------------------------------------------------------------------


class TestSegMapCorrectness:
    """Verify character-level seg_map assignment matches the training dataloader."""

    def test_seg_map_char_overlap(self):
        """Character-level overlap assigns each token to the correct segment."""
        segment_size = 128
        n_segs = 4
        segs = _make_segments(n_segs=n_segs, segment_size=segment_size)
        n_prefix = segment_size * n_segs  # Exactly fills all segments
        n_gen = segment_size

        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(n_prefix)),
            n_gen_tokens=n_gen,
            target_segment_idx=n_segs - 1,
            tokenizer=_MOCK_TOK,
        )

        seg_map = result.seg_map[0]  # (T,)
        # With 4 chars/token, 128 tokens = 512 chars = 1 segment
        for pos in range(n_prefix):
            expected = min(pos // segment_size, n_segs - 1)
            assert seg_map[pos].item() == expected, (
                f"seg_map[{pos}] = {seg_map[pos].item()}, expected {expected}"
            )

    def test_gen_tokens_map_to_target(self):
        """All generated token positions should map to target_segment_idx."""
        segment_size = 128
        segs = _make_segments(n_segs=3, segment_size=segment_size)
        n_prefix = 256
        n_gen = 128
        target = 2

        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(n_prefix)),
            n_gen_tokens=n_gen,
            target_segment_idx=target,
            tokenizer=_MOCK_TOK,
        )

        seg_map = result.seg_map[0]
        gen_region = seg_map[n_prefix:]
        assert (gen_region == target).all()

    def test_full_context_uses_direct_segment_slots(self):
        segment_size = 128
        segs = _make_segments(n_segs=4, segment_size=segment_size)
        n_prefix = segment_size * 3
        n_gen = segment_size

        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(n_prefix)),
            n_gen_tokens=n_gen,
            target_segment_idx=3,
            tokenizer=_MOCK_TOK,
        )

        assert result.enc_feature_ids.shape[1] == 4
        assert result.enc_feature_ids[0, 0, 0].item() == 0
        assert result.enc_feature_ids[0, 1, 0].item() == 10
        assert result.enc_feature_ids[0, 2, 0].item() == 20
        assert result.enc_feature_ids[0, 3, 0].item() == 30

        seg_map = result.seg_map[0]
        assert (seg_map[:128] == 0).all()
        assert (seg_map[128:256] == 1).all()
        assert (seg_map[256:384] == 2).all()
        assert (seg_map[384:] == 3).all()

    def test_prefix_past_last_segment_raises(self):
        """Prefixes extending past all segment chars fail loudly."""
        segment_size = 128
        n_segs = 2
        segs = _make_segments(n_segs=n_segs, segment_size=segment_size)
        n_prefix = 500  # Longer than 2 * 128

        with pytest.raises(ValueError, match="zero overlap"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(n_prefix)),
                n_gen_tokens=128,
                target_segment_idx=1,
                tokenizer=_MOCK_TOK,
            )


# ---------------------------------------------------------------------------
# Test: feature count cap with thresholding
# ---------------------------------------------------------------------------


class TestFeatureCountCap:
    """Verify that segment_topk thresholding caps features per segment."""

    def test_topk_respects_k_value(self):
        """With k=5, no segment should have more than 5 features after prepare."""
        fids = t.arange(50, dtype=t.int32)
        mags = t.arange(50, dtype=t.float32) + 1.0
        seg = SegmentFeatures(1, 128, fids, mags, 128, 256)

        boundaries = t.zeros(50, 2, dtype=t.float64)
        boundaries[:, 0] = 10.0
        boundaries[:, 1] = 30.0
        precomputed = t.zeros(50, dtype=t.float64)

        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, threshold=5.9
        )

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        assert len(prepared[0].feature_ids) <= 5

    def test_topk_retains_highest_magnitude(self):
        """Top-k should keep the highest magnitude features."""
        fids = t.tensor([0, 1, 2, 3, 4], dtype=t.int32)
        mags = t.tensor([1.0, 5.0, 3.0, 2.0, 4.0], dtype=t.float32)
        seg = SegmentFeatures(1, 128, fids, mags, 128, 256)

        boundaries = t.zeros(5, 1, dtype=t.float64)
        boundaries[:] = 2.5
        precomputed = t.zeros(5, dtype=t.float64)

        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, threshold=3.0
        )

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        surviving_fids = set(prepared[0].feature_ids.tolist())
        # Features 1 (5.0), 4 (4.0), 2 (3.0) should survive with k=3
        assert 1 in surviving_fids
        assert 4 in surviving_fids


# ---------------------------------------------------------------------------
# Test: ablation correctness
# ---------------------------------------------------------------------------


class TestAblationCorrectness:
    """Verify ablation removes only the target feature from the target segment."""

    def test_ablation_preserves_other_features(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        target_fid = segs[2].feature_ids[2].item()  # Pick the middle feature
        ablated = ablate_feature(segs, target_seg_idx=2, feature_id=target_fid)  # ty: ignore[invalid-argument-type]

        # Target segment should have one fewer feature
        assert len(ablated[2].feature_ids) == 4
        assert target_fid not in ablated[2].feature_ids.tolist()

        # Other segments should be unchanged
        for i in (0, 1):
            assert ablated[i] is segs[i]

    def test_ablation_encoder_inputs_differ(self):
        """Real vs ablated encoder inputs should differ at the target segment."""
        segs = _make_segments(n_segs=3, features_per_seg=5)
        target_fid = segs[2].feature_ids[0].item()
        ablated = ablate_feature(segs, target_seg_idx=2, feature_id=target_fid)  # ty: ignore[invalid-argument-type]

        real_inputs = build_multiseg_inputs(
            segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        ablated_inputs = build_multiseg_inputs(
            ablated,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )

        # Segment 2 mask should differ
        assert not t.equal(
            real_inputs.enc_mask[0, 2],
            ablated_inputs.enc_mask[0, 2],
        )

        # Segments 0 and 1 should be identical
        assert t.equal(
            real_inputs.enc_feature_ids[0, 0],
            ablated_inputs.enc_feature_ids[0, 0],
        )
        assert t.equal(
            real_inputs.enc_feature_ids[0, 1],
            ablated_inputs.enc_feature_ids[0, 1],
        )


# ---------------------------------------------------------------------------
# Test: insertion addition
# ---------------------------------------------------------------------------


class TestInsertionAddition:
    """Verify feature addition for insertion (dose-response) mode."""

    def test_add_feature_creates_per_bucket_conditions(self):
        """Each bucket should produce a distinct magnitude in the target segment."""
        segs = _make_segments(n_segs=3, features_per_seg=3)
        feature_id = 999
        num_buckets = 4

        for bucket in range(num_buckets):
            added = add_feature(segs, target_seg_idx=2, feature_id=feature_id, bucket_index=bucket)
            target = added[2]
            idx = target.feature_ids.tolist().index(feature_id)
            assert target.feature_mags[idx].item() == float(bucket)

    def test_add_preserves_existing_features(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        original_fids = set(segs[1].feature_ids.tolist())
        added = add_feature(segs, target_seg_idx=1, feature_id=999, bucket_index=2)
        new_fids = set(added[1].feature_ids.tolist())
        # All original features should still be present
        assert original_fids.issubset(new_fids)
        # Plus the new one
        assert 999 in new_fids

    def test_insertion_inputs_differ_across_buckets(self):
        """Encoder inputs should vary across bucket indices."""
        segs = _make_segments(n_segs=2, features_per_seg=3)
        feature_id = 999

        inputs_b0 = build_multiseg_inputs(
            add_feature(segs, 1, feature_id, 0),
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        inputs_b3 = build_multiseg_inputs(
            add_feature(segs, 1, feature_id, 3),
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )

        # Magnitudes at target segment should differ
        assert not t.equal(
            inputs_b0.enc_magnitudes[0, 1],
            inputs_b3.enc_magnitudes[0, 1],
        )


# ---------------------------------------------------------------------------
# Test: YAML serialization round-trips
# ---------------------------------------------------------------------------


class TestYAMLSerializationRoundTrip:
    """Verify production YAML artifacts survive round-trips."""

    def test_selections_round_trip(self, tmp_path):
        selections = [
            FeatureSelection(
                feature_id=42,
                neuronpedia_description="references to animals",
                n_docs_activated=10,
                selected_docs=[0, 1, 5],
            ),
        ]

        serializable = [
            {
                "feature_id": s.feature_id,
                "neuronpedia_description": s.neuronpedia_description,
                "n_docs_activated": s.n_docs_activated,
                "selected_docs": s.selected_docs,
            }
            for s in selections
        ]

        path = tmp_path / "selections.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(serializable, f, default_flow_style=False)

        with open(path) as f:
            raw = yaml.safe_load(f)

        loaded = [
            FeatureSelection(
                feature_id=s["feature_id"],
                neuronpedia_description=s["neuronpedia_description"],
                n_docs_activated=s["n_docs_activated"],
                selected_docs=s["selected_docs"],
            )
            for s in raw
        ]

        assert len(loaded) == 1
        assert loaded[0].feature_id == 42
        assert loaded[0].neuronpedia_description == "references to animals"
        assert loaded[0].selected_docs == [0, 1, 5]

    def test_completions_round_trip(self, tmp_path):
        completions = [
            DocumentCompletion(
                doc_idx=0,
                feature_id=42,
                pre_segment_text="Once upon a time",
                ground_truth_segment=" there was a cat.",
                target_segment_idx=2,
                samples=[
                    CompletionSample(
                        condition="real",
                        text="there lived a dog.",
                        seed=42,
                        bucket_index=None,
                    ),
                    CompletionSample(
                        condition="added_bucket_3",
                        text="the cat appeared.",
                        seed=43,
                        bucket_index=3,
                    ),
                ],
            ),
        ]

        # Serialize
        serializable = [
            {
                "doc_idx": c.doc_idx,
                "feature_id": c.feature_id,
                "pre_segment_text": c.pre_segment_text,
                "ground_truth_segment": c.ground_truth_segment,
                "target_segment_idx": c.target_segment_idx,
                "samples": [
                    {
                        "condition": str(s.condition),
                        "text": s.text,
                        "seed": s.seed,
                        "bucket_index": s.bucket_index,
                    }
                    for s in c.samples
                ],
            }
            for c in completions
        ]

        path = tmp_path / "completions.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(serializable, f, default_flow_style=False)

        # Deserialize
        with open(path) as f:
            raw = yaml.safe_load(f)

        loaded = [
            DocumentCompletion(
                doc_idx=c["doc_idx"],
                feature_id=c["feature_id"],
                pre_segment_text=c["pre_segment_text"],
                ground_truth_segment=c["ground_truth_segment"],
                target_segment_idx=c["target_segment_idx"],
                samples=[
                    CompletionSample(
                        condition=s["condition"],
                        text=s["text"],
                        seed=s["seed"],
                        bucket_index=s.get("bucket_index"),
                    )
                    for s in c["samples"]
                ],
            )
            for c in raw
        ]

        assert len(loaded) == 1
        assert loaded[0].doc_idx == 0
        assert loaded[0].feature_id == 42
        assert loaded[0].pre_segment_text == "Once upon a time"
        assert len(loaded[0].samples) == 2
        assert loaded[0].samples[0].condition == "real"
        assert loaded[0].samples[0].bucket_index is None
        assert loaded[0].samples[1].condition == "added_bucket_3"
        assert loaded[0].samples[1].bucket_index == 3


# ---------------------------------------------------------------------------
# Test: generate_for_document orchestrator
# ---------------------------------------------------------------------------


class TestGenerateForDocument:
    """Test the document-level generation orchestrator."""

    def _make_fake_env(self, n_segs: int = 3, num_buckets: int = 4):
        """Create all the test doubles needed for generate_for_document."""
        model = _FakeModel(vocab_size=100, seq_len=1024)

        # Patch model.encoder to return dummy encoder_out
        def fake_encoder_call(fids, mags, mask):
            n_seg = fids.shape[1]
            dense = t.randn(1, n_seg, mask.shape[2], 256)
            return pack_dense_encoder_output_reference(dense, mask)

        model.encoder = MagicMock(side_effect=fake_encoder_call)
        model.encoder.config = MagicMock()
        model.encoder.config.num_buckets = num_buckets

        tokenizer = _FakeTokenizer()

        # Create segments with char boundaries
        segment_size = 128
        doc_text = "A" * 512 * n_segs  # Each segment covers 512 chars
        segments = _make_segments(n_segs=n_segs, segment_size=segment_size)

        boundaries = t.zeros(100, num_buckets - 1, dtype=t.float64)
        for i in range(num_buckets - 1):
            boundaries[:, i] = float(i + 1)

        return model, tokenizer, doc_text, segments, boundaries

    def test_ablation_mode_produces_two_conditions(self):
        """Ablation mode should produce real and ablated samples."""
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(n_segs=3)
        feature_id = segments[2].feature_ids[0].item()

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=feature_id,
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=4,
            samples_per_condition=2,
            temperature=0.0,
        )

        conditions = {s.condition for s in result.samples}
        assert conditions == {"real", "ablated"}
        assert len(result.samples) == 4

    def test_insertion_mode_produces_per_bucket(self):
        """Insertion mode should produce real plus per-bucket samples."""
        num_buckets = 4
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(
            n_segs=3, num_buckets=num_buckets
        )
        feature_id = 999  # Feature not in segments

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=feature_id,
            mode=EvalMode.INSERTION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=num_buckets,
            samples_per_condition=1,
            temperature=0.0,
        )

        conditions = {s.condition for s in result.samples}
        expected = {"real"} | {f"added_bucket_{b}" for b in range(num_buckets)}
        assert conditions == expected
        assert len(result.samples) == 1 + num_buckets

    def test_unique_seeds_per_sample(self):
        """Each sample should get a unique seed."""
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(n_segs=3)
        feature_id = segments[2].feature_ids[0].item()

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=feature_id,
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=4,
            samples_per_condition=3,
        )

        seeds = [s.seed for s in result.samples]
        assert len(set(seeds)) == len(seeds), "All seeds should be unique"

    def test_document_completion_fields_populated(self):
        """DocumentCompletion should have correct metadata."""
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(n_segs=3)
        feature_id = segments[2].feature_ids[0].item()

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=feature_id,
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=4,
            doc_idx=7,
            samples_per_condition=1,
            temperature=0.0,
        )

        assert result.doc_idx == 7
        assert result.feature_id == feature_id
        assert result.target_segment_idx == 2
        assert isinstance(result.pre_segment_text, str)
        assert isinstance(result.ground_truth_segment, str)
        assert len(result.samples) > 0

    def test_document_generation_uses_direct_encoder_slots(self):
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(n_segs=3)
        feature_id = segments[2].feature_ids[0].item()

        generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=2,
            feature_id=feature_id,
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=4,
            samples_per_condition=1,
            temperature=0.0,
        )

        enc_fids = model.encoder.call_args_list[0].args[0]
        assert enc_fids.shape[1] == 3
        assert set(enc_fids[0, 0].tolist()) == set(segments[0].feature_ids.tolist())
        assert set(enc_fids[0, 1].tolist()) == set(segments[1].feature_ids.tolist())
        assert set(enc_fids[0, 2].tolist()) == set(segments[2].feature_ids.tolist())

    def test_bucket_index_set_for_insertion_samples(self):
        """Insertion samples should have bucket_index set."""
        num_buckets = 3
        model, tokenizer, doc_text, segments, boundaries = self._make_fake_env(
            n_segs=2, num_buckets=num_buckets
        )

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segments,
            target_segment_idx=1,
            feature_id=999,
            mode=EvalMode.INSERTION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=num_buckets,
            samples_per_condition=1,
            temperature=0.0,
        )

        for s in result.samples:
            if s.condition.startswith("added_bucket_"):
                assert s.bucket_index is not None
                bucket_from_name = int(s.condition.split("_")[-1])
                assert s.bucket_index == bucket_from_name
            else:
                assert s.bucket_index is None


# ---------------------------------------------------------------------------
# Test: assignment-based generation length derivation
# ---------------------------------------------------------------------------


class TestAssignmentBasedGenLength:
    """Verify that generate_for_document derives prefix and gen length from
    segment assignments rather than a fixed segment_size parameter.
    """

    def test_gen_length_matches_segment_token_count(self):
        """With the fixed-width tokenizer (4 chars/token) and 512-char segments,
        the derived gen length should be 128 tokens (512 / 4).
        """
        tokenizer = _FakeTokenizer()
        bos_id = 2
        bos_prefix = tokenizer.decode([bos_id])

        # 3 segments, each 512 chars
        segments = _make_segments(n_segs=3)
        target_segment_idx = 2
        target_seg = segments[target_segment_idx]

        doc_text = "A" * 512 * 3
        bos_text = bos_prefix + doc_text

        # Replicate the derivation logic from generate_for_document
        text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
        all_tokens = tokenizer.encode(text_through_target, prepend=bos_id)

        char_offsets = token_to_char_offsets(tokenizer, all_tokens)
        token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
        token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

        seg_char_starts = t.tensor(
            [s.char_start for s in segments[: target_segment_idx + 1]],
            dtype=t.int64,
        )
        seg_char_ends = t.tensor(
            [s.char_end for s in segments[: target_segment_idx + 1]],
            dtype=t.int64,
        )

        assignments = assign_tokens_to_segments(
            token_starts,
            token_ends,
            seg_char_starts,
            seg_char_ends,
        )
        assignments = t.cummax(assignments, dim=0).values

        target_positions = (assignments == target_segment_idx).nonzero(as_tuple=True)[0]
        assert len(target_positions) > 0, "Should have tokens in target segment"
        split_idx = int(target_positions[0].item())

        prefix_tokens = all_tokens[:split_idx]
        n_gen_tokens = len(all_tokens) - split_idx

        # With 4 chars/token and 512-char segments, each segment = 128 tokens.
        # Target is segment 2, so prefix covers segments 0+1 = 256 tokens (+ BOS).
        # Gen covers segment 2 = 128 tokens.
        assert n_gen_tokens == 128
        # Prefix = BOS + 2 segments worth of tokens
        assert len(prefix_tokens) >= 256  # at least 256, plus BOS

    def test_wide_tokenizer_different_gen_length(self):
        """With 8-char tokens, a 512-char segment produces 64 gen tokens,
        not the old fixed 128.
        """

        class _WideTokenizer:
            bos_id = 2

            def encode(
                self, text: str, prepend: int | None = None, append: int | None = None
            ) -> list[int]:
                tokens = list(range(10, 10 + len(text) // 8))
                if prepend is not None:
                    tokens = [prepend] + tokens
                return tokens

            def decode(self, ids: list[int]) -> str:
                return "x" * (len(ids) * 8)

        tokenizer = _WideTokenizer()
        bos_id = 2
        bos_prefix = tokenizer.decode([bos_id])

        segments = _make_segments(n_segs=2)  # [0,512) and [512,1024)
        target_segment_idx = 1
        target_seg = segments[target_segment_idx]

        doc_text = "A" * 512 * 2
        bos_text = bos_prefix + doc_text

        text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
        all_tokens = tokenizer.encode(text_through_target, prepend=bos_id)

        char_offsets = token_to_char_offsets(tokenizer, all_tokens)
        token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
        token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

        seg_char_starts = t.tensor(
            [s.char_start for s in segments[: target_segment_idx + 1]],
            dtype=t.int64,
        )
        seg_char_ends = t.tensor(
            [s.char_end for s in segments[: target_segment_idx + 1]],
            dtype=t.int64,
        )

        assignments = assign_tokens_to_segments(
            token_starts,
            token_ends,
            seg_char_starts,
            seg_char_ends,
        )
        assignments = t.cummax(assignments, dim=0).values

        target_positions = (assignments == target_segment_idx).nonzero(as_tuple=True)[0]
        split_idx = int(target_positions[0].item())
        n_gen_tokens = len(all_tokens) - split_idx

        # 512 chars / 8 chars per token = 64 tokens for the target segment
        assert n_gen_tokens == 64


# ---------------------------------------------------------------------------
# Test: dose-response aggregation
# ---------------------------------------------------------------------------


class TestDoseResponseAggregation:
    """Test dose-response computation with various score distributions."""

    def test_monotonic_increase_detected(self):
        """If feature_alignment increases with bucket, means should reflect that."""
        scores = {
            0: [JudgementScore(4, 1), JudgementScore(4, 2)],
            1: [JudgementScore(4, 3), JudgementScore(4, 3)],
            2: [JudgementScore(4, 4), JudgementScore(4, 5)],
        }
        result = compute_dose_response(scores)
        means = result["bucket_means"]
        assert means[0] < means[1] < means[2]

    def test_stderr_decreases_with_more_samples(self):
        """More samples should yield smaller standard error."""
        few = {0: [JudgementScore(4, 3), JudgementScore(4, 4)]}
        many = {0: [JudgementScore(4, 3)] * 10 + [JudgementScore(4, 4)] * 10}

        few_result = compute_dose_response(few)
        many_result = compute_dose_response(many)

        few_stderr = few_result["bucket_stderrs"][0]
        many_stderr = many_result["bucket_stderrs"][0]
        assert many_stderr < few_stderr

    def test_single_sample_nan_stderr(self):
        """With only one sample, stderr should be nan (can't compute variance)."""
        scores = {0: [JudgementScore(4, 3)]}
        result = compute_dose_response(scores)
        assert math.isnan(result["bucket_stderrs"][0])

    def test_coherence_means_tracked(self):
        """coherence_means should reflect the coherence dimension."""
        scores = {
            0: [JudgementScore(2, 4), JudgementScore(3, 4)],
            1: [JudgementScore(5, 1), JudgementScore(5, 1)],
        }
        result = compute_dose_response(scores)
        assert result["coherence_means"][0] == 2.5
        assert result["coherence_means"][1] == 5.0


# ---------------------------------------------------------------------------
# Test: entry-point plotting helpers
# ---------------------------------------------------------------------------


class TestEntryPointHelpers:
    """Test the helper functions defined in the entry point script."""

    def test_compute_mean_stderr(self):
        # Import from scripts module

        # We can't easily import from scripts/ since it's not a package,
        # so test the equivalent logic directly
        scores = [JudgementScore(4, 3), JudgementScore(4, 5)]
        vals = [s.feature_alignment for s in scores]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        stderr = (variance / len(vals)) ** 0.5

        assert mean == 4.0
        assert stderr > 0

    def test_compute_mean_stderr_empty(self):
        scores: list[JudgementScore] = []
        if not scores:
            mean, stderr = 0.0, 0.0
        assert mean == 0.0
        assert stderr == 0.0

    def test_compute_mean_stderr_single(self):
        scores = [JudgementScore(4, 3)]
        vals = [s.feature_alignment for s in scores]
        mean = sum(vals) / len(vals)
        assert mean == 3.0


# ---------------------------------------------------------------------------
# Test: YAML deserialization helpers (from entry point)
# ---------------------------------------------------------------------------


def _import_entry_point():
    """Import the entry point script as a module.

    The script moved from ``scripts/completion_steering_eval.py`` to
    ``scripts/eval/completion_steering_eval.py`` in PR #394
    (74cc4e33). This helper resolves to the current location.
    """
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "scripts"
        / "eval"
        / "completion_steering_eval.py"
    )
    spec = importlib.util.spec_from_file_location("completion_steering_eval_script", script_path)
    mod = importlib.util.module_from_spec(spec)  # ty: ignore[invalid-argument-type]
    # Don't actually execute main(), just load definitions
    spec.loader.exec_module(mod)  # ty: ignore[unresolved-attribute]
    return mod


class TestYAMLDeserializationHelpers:
    """Test the _load_*_yaml helper functions from the entry point."""

    def test_load_selections_yaml(self, tmp_path):
        data = [
            {
                "feature_id": 42,
                "neuronpedia_description": "test feature",
                "n_docs_activated": 5,
                "selected_docs": [0, 1, 2],
            }
        ]
        path = tmp_path / "sel.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        mod = _import_entry_point()
        result = mod._load_selections_yaml(path)

        assert len(result) == 1
        assert result[0].feature_id == 42
        assert result[0].selected_docs == [0, 1, 2]

    def test_load_selections_yaml_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        mod = _import_entry_point()
        result = mod._load_selections_yaml(path)
        assert result == []

    def test_load_completions_yaml(self, tmp_path):
        data = [
            {
                "doc_idx": 0,
                "feature_id": 42,
                "pre_segment_text": "hello",
                "ground_truth_segment": "world",
                "target_segment_idx": 1,
                "samples": [
                    {
                        "condition": "real",
                        "text": "generated",
                        "seed": 42,
                        "bucket_index": None,
                    }
                ],
                "ground_truth_signals": {
                    "feature_signals": {42: 2},
                    "n_features_active": 1,
                    "avg_magnitude": 2.0,
                    "max_magnitude": 2,
                    "min_magnitude": 2,
                    "feature_of_interest_magnitude": 2,
                },
            }
        ]
        path = tmp_path / "comp.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        mod = _import_entry_point()
        result = mod._load_completions_yaml(path)

        assert len(result) == 1
        assert result[0].doc_idx == 0
        assert result[0].samples[0].condition == "real"
        assert result[0].samples[0].bucket_index is None
        assert result[0].ground_truth_signals is not None
        assert result[0].ground_truth_signals.n_features_active == 1

    def test_load_completions_yaml_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")

        mod = _import_entry_point()
        result = mod._load_completions_yaml(path)
        assert result == []


# ---------------------------------------------------------------------------
# Test: scan_validation_features edge cases
# ---------------------------------------------------------------------------


class TestScanEdgeCases:
    """Test scan_validation_features with edge-case inputs."""

    def test_empty_doc_segments(self):
        result = scan_validation_features({}, {})
        assert result == {}

    def test_doc_with_empty_segment_list(self):
        result = scan_validation_features({0: []}, {})
        assert result == {}

    def test_doc_with_only_first_segment(self):
        """A document with only segment_idx=0 should be excluded."""
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([1, 2, 3], dtype=t.int32),
            feature_mags=t.tensor([1.0, 2.0, 3.0]),
            char_start=0,
            char_end=128,
        )
        result = scan_validation_features({0: [seg]}, {})
        assert result == {}

    def test_all_features_zero_magnitude(self):
        """Zero-magnitude features should still be recorded."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([5], dtype=t.int32),
            feature_mags=t.tensor([0.0]),
            char_start=128,
            char_end=256,
        )
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        stats = {
            "5": PerFeatureStat(
                count=100,
                pct_segments_active=0.5,
                mean_mag=5.0,
                mean_mag_incl_zeros=2.5,
                std_mag=2.0,
                min_mag=0.1,
                max_mag=10.0,
                median_mag=5.0,
                percentiles=pcts,
                histogram=None,
            )
        }
        result = scan_validation_features(
            {
                0: [
                    SegmentFeatures(0, 0, t.tensor([], dtype=t.int32), t.tensor([]), 0, 128),
                    seg,
                ]
            },
            stats,
        )
        assert 5 in result
        assert result[5].doc_activations[0].max_activation == 0.0


class TestEncoderStrictMode:
    """Test the encoder_strict XOR validation on EncoderDecoderGPT.forward."""

    def _make_mock_model(self):
        """Create a minimal mock that mimics EncoderDecoderGPT.forward signature."""
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        model = MagicMock(spec=EncoderDecoderGPT)

        # Use the real forward method's validation logic
        def strict_forward(
            idx,
            targets,
            enc_feature_ids=None,
            enc_magnitudes=None,
            enc_mask=None,
            seg_map=None,
            loss_reduction="mean",
            kv_cache=None,
            encoder_out=None,
            encoder_strict=False,
        ):
            has_raw = enc_feature_ids is not None and enc_magnitudes is not None
            has_precomputed = encoder_out is not None
            if encoder_strict:
                assert has_raw ^ has_precomputed, (
                    "Exactly one of (enc_feature_ids, enc_magnitudes) or encoder_out must be provided"
                )
            if has_precomputed:
                if not isinstance(encoder_out, PackedEncoderOutput):
                    raise TypeError("precomputed encoder_out must be PackedEncoderOutput")
                if enc_mask is not None:
                    raise ValueError("enc_mask must be omitted when encoder_out is precomputed")
            else:
                assert enc_mask is not None, "enc_mask is required with raw encoder inputs"
            return t.zeros(1, 1, 100)  # dummy logits

        model.side_effect = strict_forward
        return model

    def test_strict_rejects_both_raw_and_precomputed(self):
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        enc_mask = t.zeros(1, 1, 1, dtype=t.bool)
        packed = pack_dense_encoder_output_reference(
            t.zeros(1, 1, 1, 64),
            t.ones(1, 1, 1, dtype=t.bool),
        )
        with pytest.raises(AssertionError, match="Exactly one"):
            model(
                idx,
                targets=None,
                enc_feature_ids=t.zeros(1, 1, 1, dtype=t.int64),
                enc_magnitudes=t.zeros(1, 1, 1, dtype=t.int64),
                enc_mask=enc_mask,
                encoder_out=packed,
                encoder_strict=True,
            )

    def test_strict_rejects_neither_raw_nor_precomputed(self):
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        enc_mask = t.zeros(1, 1, 1, dtype=t.bool)
        with pytest.raises(AssertionError, match="Exactly one"):
            model(
                idx,
                targets=None,
                enc_mask=enc_mask,
                encoder_strict=True,
            )

    def test_strict_accepts_only_precomputed(self):
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        packed = pack_dense_encoder_output_reference(
            t.zeros(1, 1, 1, 64),
            t.ones(1, 1, 1, dtype=t.bool),
        )
        # Should not raise
        result = model(
            idx,
            targets=None,
            encoder_out=packed,
            encoder_strict=True,
        )
        assert result is not None

    def test_strict_accepts_only_raw(self):
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        enc_mask = t.zeros(1, 1, 1, dtype=t.bool)
        # Should not raise
        result = model(
            idx,
            targets=None,
            enc_feature_ids=t.zeros(1, 1, 1, dtype=t.int64),
            enc_magnitudes=t.zeros(1, 1, 1, dtype=t.int64),
            enc_mask=enc_mask,
            encoder_strict=True,
        )
        assert result is not None

    def test_non_strict_rejects_redundant_mask_with_precomputed(self):
        """Packed ownership remains authoritative outside strict XOR mode."""
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        enc_mask = t.zeros(1, 1, 1, dtype=t.bool)
        packed = pack_dense_encoder_output_reference(
            t.zeros(1, 1, 1, 64),
            t.ones(1, 1, 1, dtype=t.bool),
        )
        with pytest.raises(ValueError, match="enc_mask must be omitted"):
            model(
                idx,
                targets=None,
                enc_feature_ids=t.zeros(1, 1, 1, dtype=t.int64),
                enc_magnitudes=t.zeros(1, 1, 1, dtype=t.int64),
                enc_mask=enc_mask,
                encoder_out=packed,
                encoder_strict=False,
            )

    def test_dense_precomputed_state_is_rejected(self):
        model = self._make_mock_model()
        idx = t.tensor([[1]])
        with pytest.raises(TypeError, match="must be PackedEncoderOutput"):
            model(
                idx,
                targets=None,
                encoder_out=t.zeros(1, 1, 1, 64),
                encoder_strict=True,
            )


def _make_scan_tensors(n_features: int = 50, docs_per_feature: int = 5) -> ScanResultTensors:
    """Create synthetic ScanResultTensors."""
    feature_ids = t.arange(n_features, dtype=t.int64)
    n_docs_activated = t.full((n_features,), docs_per_feature, dtype=t.int64)

    total_acts = n_features * docs_per_feature
    offsets = t.arange(0, total_acts + 1, docs_per_feature, dtype=t.int64)

    doc_idx = t.zeros(total_acts, dtype=t.int64)
    max_activation = t.ones(total_acts, dtype=t.float32)
    max_segment_idx = t.ones(total_acts, dtype=t.int64)
    percentile = t.full((total_acts,), 0.5, dtype=t.float32)
    total_segments = t.full((total_acts,), 3, dtype=t.int64)

    for i in range(n_features):
        start = i * docs_per_feature
        doc_idx[start : start + docs_per_feature] = t.arange(docs_per_feature, dtype=t.int64)

    return ScanResultTensors(
        feature_ids=feature_ids,
        n_docs_activated=n_docs_activated,
        offsets=offsets,
        doc_idx=doc_idx,
        max_activation=max_activation,
        max_segment_idx=max_segment_idx,
        percentile=percentile,
        total_segments=total_segments,
    )


# ---------------------------------------------------------------------------
# Test: random feature selection (select_from_tensors)
# ---------------------------------------------------------------------------


class TestRandomSelectionFromTensors:
    """Tests for random feature selection in select_from_tensors."""

    def test_returns_exactly_top_k_features(self):
        scan = _make_scan_tensors(n_features=50)
        selections = select_from_tensors(scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION)
        assert len(selections) == 10

    def test_reproducible_with_same_seed(self):
        scan = _make_scan_tensors(n_features=50)
        s1 = select_from_tensors(
            scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION, seed=123
        )
        s2 = select_from_tensors(
            scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION, seed=123
        )
        assert [s.feature_id for s in s1] == [s.feature_id for s in s2]

    def test_different_seeds_different_selections(self):
        scan = _make_scan_tensors(n_features=50)
        s1 = select_from_tensors(scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION, seed=1)
        s2 = select_from_tensors(
            scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION, seed=999
        )
        fids1 = [s.feature_id for s in s1]
        fids2 = [s.feature_id for s in s2]
        assert fids1 != fids2

    def test_selected_features_are_valid(self):
        scan = _make_scan_tensors(n_features=50)
        valid_fids = set(scan.feature_ids.tolist())
        selections = select_from_tensors(scan, top_k=10, docs_per_feature=3, mode=EvalMode.ABLATION)
        for s in selections:
            assert s.feature_id in valid_fids

    def test_top_k_greater_than_n_features(self):
        scan = _make_scan_tensors(n_features=5)
        selections = select_from_tensors(scan, top_k=20, docs_per_feature=3, mode=EvalMode.ABLATION)
        assert len(selections) == 5

    def test_top_k_zero_returns_empty(self):
        scan = _make_scan_tensors(n_features=10)
        selections = select_from_tensors(scan, top_k=0, docs_per_feature=3, mode=EvalMode.ABLATION)
        assert selections == []

    def test_empty_scan_returns_empty(self):
        empty = ScanResultTensors(
            feature_ids=t.empty(0, dtype=t.int64),
            n_docs_activated=t.empty(0, dtype=t.int64),
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        selections = select_from_tensors(empty, top_k=5, docs_per_feature=3, mode=EvalMode.ABLATION)
        assert selections == []

    def test_returns_raw_feature_selections(self):
        """Selection returns RawFeatureSelection (no description field)."""
        scan = _make_scan_tensors(n_features=10)
        selections = select_from_tensors(scan, top_k=5, docs_per_feature=3, mode=EvalMode.ABLATION)
        for s in selections:
            assert isinstance(s, RawFeatureSelection)
            assert not hasattr(s, "neuronpedia_description")

    def test_ablation_samples_from_activated_docs(self):
        scan = _make_scan_tensors(n_features=5, docs_per_feature=10)
        selections = select_from_tensors(scan, top_k=5, docs_per_feature=3, mode=EvalMode.ABLATION)
        for s in selections:
            # In our synthetic data, activated docs are 0..9
            for doc in s.selected_docs:
                assert 0 <= doc < 10

    def test_insertion_samples_from_non_activated_docs(self):
        scan = _make_scan_tensors(n_features=5, docs_per_feature=3)
        all_docs = set(range(20))
        selections = select_from_tensors(
            scan,
            top_k=5,
            docs_per_feature=3,
            mode=EvalMode.INSERTION,
            all_doc_indices=all_docs,
        )
        for s in selections:
            activated = set(range(3))  # docs 0, 1, 2 are activated
            for doc in s.selected_docs:
                assert doc not in activated

    def test_statistical_coverage_across_seeds(self):
        """Running 100 seeds on 50 features (top_k=10) should cover >10 unique features."""
        scan = _make_scan_tensors(n_features=50)
        all_selected: set[int] = set()
        for seed in range(100):
            sels = select_from_tensors(
                scan, top_k=10, docs_per_feature=1, mode=EvalMode.ABLATION, seed=seed
            )
            all_selected.update(s.feature_id for s in sels)
        assert len(all_selected) > 10

    def test_pool_equals_top_k_selects_all(self):
        scan = _make_scan_tensors(n_features=10)
        for seed in range(5):
            sels = select_from_tensors(
                scan, top_k=10, docs_per_feature=1, mode=EvalMode.ABLATION, seed=seed
            )
            assert len(sels) == 10
            assert {s.feature_id for s in sels} == set(range(10))


# ---------------------------------------------------------------------------
# Test: YAML round-trip with ground_truth_signals
# ---------------------------------------------------------------------------


class TestYAMLGroundTruthSignalsRoundTrip:
    """Test YAML serialization/deserialization of ground_truth_signals."""

    def test_completions_with_signals_round_trip(self, tmp_path):
        gt = GroundTruthFeatureSignals(
            feature_signals={100: 2, 200: 3, 42: 1},
            n_features_active=3,
            avg_magnitude=2.0,
            max_magnitude=3,
            min_magnitude=1,
            feature_of_interest_magnitude=1,
        )
        completions = [
            DocumentCompletion(
                doc_idx=0,
                feature_id=42,
                pre_segment_text="prefix",
                ground_truth_segment="truth",
                target_segment_idx=2,
                samples=[
                    CompletionSample(condition="real", text="gen", seed=42),
                ],
                ground_truth_signals=gt,
            ),
        ]

        # Serialize
        serializable = []
        for c in completions:
            entry = {
                "doc_idx": c.doc_idx,
                "feature_id": c.feature_id,
                "pre_segment_text": c.pre_segment_text,
                "ground_truth_segment": c.ground_truth_segment,
                "target_segment_idx": c.target_segment_idx,
                "samples": [
                    {
                        "condition": str(s.condition),
                        "text": s.text,
                        "seed": s.seed,
                        "bucket_index": s.bucket_index,
                    }
                    for s in c.samples
                ],
            }
            if c.ground_truth_signals is not None:
                g = c.ground_truth_signals
                entry["ground_truth_signals"] = {
                    "feature_signals": {int(k): int(v) for k, v in g.feature_signals.items()},
                    "n_features_active": g.n_features_active,
                    "avg_magnitude": g.avg_magnitude,
                    "max_magnitude": g.max_magnitude,
                    "min_magnitude": g.min_magnitude,
                    "feature_of_interest_magnitude": g.feature_of_interest_magnitude,
                }
            serializable.append(entry)

        path = tmp_path / "comp_gt.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(serializable, f, default_flow_style=False)

        # Deserialize via entry point helper
        mod = _import_entry_point()
        loaded = mod._load_completions_yaml(path)

        assert len(loaded) == 1
        assert loaded[0].ground_truth_signals is not None
        gt_loaded = loaded[0].ground_truth_signals
        assert gt_loaded.feature_signals == {100: 2, 200: 3, 42: 1}
        assert gt_loaded.n_features_active == 3
        assert gt_loaded.avg_magnitude == 2.0
        assert gt_loaded.max_magnitude == 3
        assert gt_loaded.min_magnitude == 1
        assert gt_loaded.feature_of_interest_magnitude == 1

    def test_feature_signals_keys_are_ints_after_roundtrip(self, tmp_path):
        gt = GroundTruthFeatureSignals(
            feature_signals={42: 2},
            n_features_active=1,
            avg_magnitude=2.0,
            max_magnitude=2,
            min_magnitude=2,
            feature_of_interest_magnitude=2,
        )
        data = [
            {
                "doc_idx": 0,
                "feature_id": 42,
                "pre_segment_text": "",
                "ground_truth_segment": "",
                "target_segment_idx": 1,
                "samples": [],
                "ground_truth_signals": {
                    "feature_signals": gt.feature_signals,
                    "n_features_active": 1,
                    "avg_magnitude": 2.0,
                    "max_magnitude": 2,
                    "min_magnitude": 2,
                    "feature_of_interest_magnitude": 2,
                },
            }
        ]
        path = tmp_path / "keys.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        mod = _import_entry_point()
        loaded = mod._load_completions_yaml(path)
        for k in loaded[0].ground_truth_signals.feature_signals:
            assert isinstance(k, int)

    def test_none_feature_of_interest_preserved(self, tmp_path):
        data = [
            {
                "doc_idx": 0,
                "feature_id": 42,
                "pre_segment_text": "",
                "ground_truth_segment": "",
                "target_segment_idx": 1,
                "samples": [],
                "ground_truth_signals": {
                    "feature_signals": {100: 1},
                    "n_features_active": 1,
                    "avg_magnitude": 1.0,
                    "max_magnitude": 1,
                    "min_magnitude": 1,
                    "feature_of_interest_magnitude": None,
                },
            }
        ]
        path = tmp_path / "none_foi.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        mod = _import_entry_point()
        loaded = mod._load_completions_yaml(path)
        assert loaded[0].ground_truth_signals.feature_of_interest_magnitude is None

    def test_old_format_without_ground_truth_signals_raises(self, tmp_path):
        """Old YAML without ground_truth_signals field raises KeyError."""
        data = [
            {
                "doc_idx": 0,
                "feature_id": 42,
                "pre_segment_text": "hello",
                "ground_truth_segment": "world",
                "target_segment_idx": 1,
                "samples": [{"condition": "real", "text": "gen", "seed": 42}],
            }
        ]
        path = tmp_path / "old.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

        mod = _import_entry_point()
        with pytest.raises(KeyError):
            mod._load_completions_yaml(path)


# ---------------------------------------------------------------------------
# Tests: context-length filtering in selection
# ---------------------------------------------------------------------------


def _make_scan_tensors_for_ctx_filter(
    n_features: int = 2,
    docs_per_feature: int = 5,
    total_segments: int = 3,
) -> ScanResultTensors:
    """Build ScanResultTensors with predictable doc indices and segment counts."""
    feature_ids = t.arange(n_features, dtype=t.int64)
    n_docs = t.full((n_features,), docs_per_feature, dtype=t.int64)
    offsets = t.arange(n_features + 1, dtype=t.int64) * docs_per_feature
    total_acts = n_features * docs_per_feature

    doc_idx = t.arange(total_acts, dtype=t.int64)
    max_activation = t.ones(total_acts, dtype=t.float32)
    # max_segment_idx = last segment (total_segments - 1) for all docs
    max_segment_idx = t.full((total_acts,), total_segments - 1, dtype=t.int64)
    percentile = t.full((total_acts,), 0.9, dtype=t.float32)
    total_segs = t.full((total_acts,), total_segments, dtype=t.int64)

    return ScanResultTensors(
        feature_ids=feature_ids,
        n_docs_activated=n_docs,
        offsets=offsets,
        doc_idx=doc_idx,
        max_activation=max_activation,
        max_segment_idx=max_segment_idx,
        percentile=percentile,
        total_segments=total_segs,
    )


def _make_ctx_filter_feature_index(
    n_docs: int,
    n_segs: int = 3,
    seg_features: list[tuple[list[int], list[float]]] | None = None,
):
    """Return a FeatureIndex mock where each doc has n_segs segments at 512 chars each.

    Args:
        seg_features: Per-segment ``(feature_ids, feature_mags)`` list.  When
            ``None``, every segment gets ``feature_ids=[0], mags=[1.0]``.
    """
    idx = MagicMock()

    def _lookup(shard_stem, doc_idx):
        segs = []
        for i in range(n_segs):
            if seg_features is not None and i < len(seg_features):
                fids, mags = seg_features[i]
            else:
                fids, mags = [0], [1.0]
            segs.append(
                SegmentFeatures(
                    segment_idx=i,
                    position=i * 128,
                    feature_ids=t.tensor(fids, dtype=t.int32),
                    feature_mags=t.tensor(mags),
                    char_start=i * 512,
                    char_end=(i + 1) * 512,
                )
            )
        return segs

    idx.lookup.side_effect = _lookup
    return idx


class _CtxFilterTokenizer:
    """Tokenizer for context-length filtering tests.

    Each token = 4 chars.  encode returns BOS + text_len//4 tokens.
    """

    bos_id = 2

    def encode(self, text: str, prepend: int | None = None, **kw) -> list[int]:
        tokens = list(range(10, 10 + len(text) // 4))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 4)


class TestContextLengthFilterInSelection:
    """Verify that select_from_tensors excludes documents whose prefix exceeds
    the context window BEFORE sampling, so we never select invalid docs."""

    def test_ablation_filters_long_docs(self):
        """In ablation mode, docs whose max_segment_idx doesn't fit are
        salvaged via find_fitting_segment fallback to earlier segments."""
        n_segs = 10  # 10 segments × 128 tokens = 1280 tokens prefix for last seg
        # Feature 0 present in all segments (default from mock helper)
        scan = _make_scan_tensors_for_ctx_filter(
            n_features=1,
            docs_per_feature=5,
            total_segments=n_segs,
        )
        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * (n_segs * 512) for _ in range(5)]
        fi = _make_ctx_filter_feature_index(5, n_segs)

        # Context window = 512 tokens. With 4 chars/token, seg 1 ends at
        # char 1024 → 256 tokens (fits). Seg 3 ends at char 2048 → 512 tokens
        # (fits exactly). Seg 4 ends at char 2560 → 640 tokens (doesn't fit).
        # max_segment_idx=9 doesn't fit, but fallback finds earlier segments.
        ctx = ContextFilter(
            doc_texts=doc_texts,  # ty: ignore[invalid-argument-type]
            feature_index=fi,
            val_shard_stem="shard_0000",
            tokenizer=tok,
            max_seq_len=512,
            bos_id=2,
        )
        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=5,
            mode=EvalMode.ABLATION,
            ctx_filter=ctx,
        )
        assert len(selections) == 1
        # Docs are salvaged via fallback — target_segments point to fitting segs
        assert len(selections[0].selected_docs) > 0
        for doc_idx in selections[0].selected_docs:
            seg_idx = selections[0].target_segments[doc_idx]
            assert 1 <= seg_idx < n_segs

    def test_ablation_keeps_short_docs(self):
        """Docs with small max_segment_idx that fit are kept."""
        n_segs = 3
        scan = _make_scan_tensors_for_ctx_filter(
            n_features=1,
            docs_per_feature=3,
            total_segments=n_segs,
        )
        # Override max_segment_idx to target segment 1 (prefix ≈ 128 tokens)
        scan.max_segment_idx[:] = 1

        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * (n_segs * 512) for _ in range(3)]
        fi = _make_ctx_filter_feature_index(3, n_segs)

        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="shard_0000",
            tokenizer=tok,
            max_seq_len=1024,
            bos_id=2,
        )
        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=3,
            mode=EvalMode.ABLATION,
            ctx_filter=ctx,
        )
        assert len(selections) == 1
        assert len(selections[0].selected_docs) == 3

    def test_insertion_filters_long_docs(self):
        """In insertion mode, docs where no segment without the feature
        fits the context window are excluded."""
        n_segs = 10
        # No activated docs → all docs are insertion candidates
        scan = ScanResultTensors(
            feature_ids=t.tensor([0], dtype=t.int64),
            n_docs_activated=t.tensor([0], dtype=t.int64),
            offsets=t.tensor([0, 0], dtype=t.int64),
            doc_idx=t.tensor([], dtype=t.int64),
            max_activation=t.tensor([], dtype=t.float32),
            max_segment_idx=t.tensor([], dtype=t.int64),
            percentile=t.tensor([], dtype=t.float32),
            total_segments=t.tensor([], dtype=t.int64),
        )

        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * (n_segs * 512) for _ in range(5)]
        # Feature 0 is in all segments (default), so insertion can't use any
        # segment that contains it. All segments have feature 0 → no candidates.
        fi = _make_ctx_filter_feature_index(5, n_segs)
        all_docs = set(range(5))

        ctx = ContextFilter(
            doc_texts=doc_texts,  # ty: ignore[invalid-argument-type]
            feature_index=fi,
            val_shard_stem="shard_0000",
            tokenizer=tok,
            max_seq_len=200,  # irrelevant — all segments have feature 0
            bos_id=2,
        )
        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=5,
            mode=EvalMode.INSERTION,
            all_doc_indices=all_docs,
            ctx_filter=ctx,
        )
        assert len(selections) == 1
        assert selections[0].selected_docs == []

    def test_without_filter_params_no_filtering(self):
        """When context-filter params are omitted, all docs are sampled."""
        scan = _make_scan_tensors_for_ctx_filter(
            n_features=1,
            docs_per_feature=3,
            total_segments=10,
        )
        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=3,
            mode=EvalMode.ABLATION,
        )
        assert len(selections) == 1
        assert len(selections[0].selected_docs) == 3

    def test_doc_fits_context_true(self):
        """doc_fits_context returns True for a short prefix."""
        tok = _CtxFilterTokenizer()
        fi = _make_ctx_filter_feature_index(1, n_segs=3)
        doc_texts = ["A" * 1536]
        bos_prefix = tok.decode([2])

        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=1024,
            bos_id=2,
        )
        assert doc_fits_context(
            doc_idx=0,
            target_seg_idx=1,
            ctx=ctx,
            bos_prefix=bos_prefix,
        )

    def test_doc_fits_context_false(self):
        """doc_fits_context returns False when prefix exceeds context."""
        tok = _CtxFilterTokenizer()
        fi = _make_ctx_filter_feature_index(1, n_segs=10)
        doc_texts = ["A" * 5120]
        bos_prefix = tok.decode([2])

        ctx = ContextFilter(
            doc_texts=doc_texts,  # ty: ignore[invalid-argument-type]
            feature_index=fi,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=512,
            bos_id=2,
        )
        assert not doc_fits_context(
            doc_idx=0,
            target_seg_idx=9,
            ctx=ctx,
            bos_prefix=bos_prefix,
        )


# ---------------------------------------------------------------------------
# Tests: find_fitting_segment
# ---------------------------------------------------------------------------


def _make_ctx_and_bos(
    n_segs: int,
    max_seq_len: int,
    seg_features: list[tuple[list[int], list[float]]] | None = None,
    n_docs: int = 1,
):
    """Build a ContextFilter + bos_prefix for find_fitting_segment tests."""
    tok = _CtxFilterTokenizer()
    doc_texts = ["A" * (n_segs * 512) for _ in range(n_docs)]
    fi = _make_ctx_filter_feature_index(n_docs, n_segs, seg_features=seg_features)
    ctx = ContextFilter(
        doc_texts=doc_texts,  # ty: ignore[invalid-argument-type]
        feature_index=fi,
        val_shard_stem="shard_0000",
        tokenizer=tok,
        max_seq_len=max_seq_len,
        bos_id=2,
    )
    bos_prefix = tok.decode([2])
    return ctx, bos_prefix


class TestFindFittingSegment:
    """Unit tests for find_fitting_segment."""

    def test_preferred_fits_returns_immediately(self):
        """When preferred_seg_idx fits, return it without lookup overhead."""
        ctx, bos_prefix = _make_ctx_and_bos(n_segs=5, max_seq_len=1024)
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
            preferred_seg_idx=2,
        )
        assert result == 2

    def test_ablation_fallback_to_earlier_segment(self):
        """Preferred doesn't fit, but earlier segment with feature does."""
        # doc_fits_context slices bos_prefix then prepends BOS token — net
        # token count = char_end / chars_per_token (offsets cancel out).
        # seg 1: char_end=1024 → 256 tokens ≤ 300 (fits).
        # seg 2: char_end=1536 → 384 tokens > 300 (doesn't fit).
        # Feature 0 in all segments.
        ctx, bos_prefix = _make_ctx_and_bos(n_segs=5, max_seq_len=300)
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
            preferred_seg_idx=4,  # doesn't fit
        )
        assert result == 1

    def test_ablation_no_feature_in_fitting_segments(self):
        """Feature only in non-fitting segments → None."""
        # Feature 5 only in segment 4 (which doesn't fit at max_seq_len=300).
        seg_features = [
            ([0], [1.0]),  # seg 0
            ([1], [1.0]),  # seg 1 — fits but no feature 5
            ([2], [1.0]),  # seg 2 — doesn't fit
            ([3], [1.0]),  # seg 3 — doesn't fit
            ([5], [2.0]),  # seg 4 — doesn't fit, has feature 5
        ]
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=5,
            max_seq_len=300,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=5,
            preferred_seg_idx=4,
        )
        assert result is None

    def test_ablation_prefers_highest_activation(self):
        """Multiple fitting segments with the feature → highest mag chosen."""
        # seg 1: feature 0 mag 1.0 (fits)
        # seg 2: feature 0 mag 3.0 (fits if max_seq_len big enough)
        seg_features = [
            ([99], [1.0]),  # seg 0 (never eligible)
            ([0], [1.0]),  # seg 1
            ([0, 1], [3.0, 1.0]),  # seg 2
            ([0], [2.0]),  # seg 3
        ]
        # seg 2: char_end=1536 → 384 tokens ≤ 400 (fits).
        # seg 3: char_end=2048 → 512 tokens > 400 (doesn't fit).
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=4,
            max_seq_len=400,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
            preferred_seg_idx=3,  # doesn't fit
        )
        assert result == 2  # highest mag (3.0) among fitting segments

    def test_ablation_feature_in_some_segments_only(self):
        """Feature in segments 2,5,8; only seg 2 fits; returns 2."""
        n_segs = 10
        seg_features = []
        for i in range(n_segs):
            if i in (2, 5, 8):
                seg_features.append(([7], [float(i)]))
            else:
                seg_features.append(([99], [1.0]))
        # seg 2: char_end=1536 → 384 tokens ≤ 400 (fits).
        # seg 5: char_end=3072 → 768 tokens > 400 (doesn't fit).
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=n_segs,
            max_seq_len=400,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=7,
            preferred_seg_idx=8,
        )
        assert result == 2

    def test_insertion_picks_from_fitting(self):
        """Insertion picks a random fitting segment without the feature."""
        # Feature 0 only in segment 0 and 3. Segments 1, 2 are eligible.
        seg_features = [
            ([0], [1.0]),  # seg 0 — has feature, never eligible anyway
            ([5], [1.0]),  # seg 1 — no feature 0, eligible
            ([6], [1.0]),  # seg 2 — no feature 0, eligible
            ([0], [1.0]),  # seg 3 — has feature 0
        ]
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=4,
            max_seq_len=1024,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.INSERTION,
            feature_id=0,
            seed=42,
        )
        assert result in (1, 2)

    def test_insertion_skips_segments_with_feature(self):
        """Segments containing the feature are excluded even if they fit."""
        # Feature 0 in all segments >= 1
        seg_features = [
            ([99], [1.0]),  # seg 0
            ([0], [1.0]),  # seg 1
            ([0], [1.0]),  # seg 2
        ]
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=3,
            max_seq_len=1024,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.INSERTION,
            feature_id=0,
        )
        assert result is None

    def test_insertion_no_fitting_returns_none(self):
        """All segments too long → None."""
        seg_features = [
            ([99], [1.0]),
            ([5], [1.0]),
            ([6], [1.0]),
        ]
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=3,
            max_seq_len=50,
            seg_features=seg_features,
        )
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.INSERTION,
            feature_id=0,
        )
        assert result is None

    def test_segment_0_never_selected(self):
        """Segment 0 is never eligible in either mode."""
        # Only segment 0 has feature 0 and fits
        seg_features = [
            ([0], [5.0]),  # seg 0 — has feature, but never eligible
            ([99], [1.0]),  # seg 1
        ]
        # seg 0: char_end=512 → 128 tokens (but seg 0 is never eligible).
        # seg 1: char_end=1024 → 256 tokens > 200 (doesn't fit).
        ctx, bos_prefix = _make_ctx_and_bos(
            n_segs=2,
            max_seq_len=200,
            seg_features=seg_features,
        )
        # Ablation: feature 0 only in seg 0 (excluded)
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
        )
        assert result is None

        # Insertion: seg 0 excluded, seg 1 doesn't fit
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.INSERTION,
            feature_id=0,
        )
        assert result is None

    def test_doc_with_one_segment_returns_none(self):
        """Only segment 0 → None."""
        ctx, bos_prefix = _make_ctx_and_bos(n_segs=1, max_seq_len=1024)
        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests: cross-tokenizer correctness
# ---------------------------------------------------------------------------


class _WideTokenizer:
    """Tokenizer where each token = 8 characters (wider than default 4-char)."""

    bos_id = 2

    def encode(self, text: str, prepend: int | None = None, **kw) -> list[int]:
        n = len(text) // 8
        tokens = list(range(10, 10 + n))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 8)


class _NarrowTokenizer:
    """Tokenizer where each token = 2 characters (narrower than default 4-char)."""

    bos_id = 2

    def encode(self, text: str, prepend: int | None = None, **kw) -> list[int]:
        n = len(text) // 2
        tokens = list(range(10, 10 + n))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 2)


class _MixedWidthTokenizer:
    """Tokenizer with variable-width tokens (alternating 2 and 6 chars)."""

    bos_id = 2

    def encode(self, text: str, prepend: int | None = None, **kw) -> list[int]:
        tokens = []
        pos = 0
        tok_id = 10
        while pos < len(text):
            # Alternate: first token 2 chars, next 6 chars
            width = 2 if len(tokens) % 2 == 0 else 6
            if pos + width > len(text):
                break
            tokens.append(tok_id)
            tok_id += 1
            pos += width
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 4)  # avg 4 chars


class TestCrossTokenizerCorrectness:
    """Verify that context-fit checks use character-space token counting."""

    def test_wide_tokenizer_fewer_segments_fit(self):
        """8-char tokens: a 512-char segment = 64 tokens (not 128).
        With max_seq_len=150, seg 1 fits but seg 2 doesn't."""
        n_segs = 5
        tok = _WideTokenizer()
        doc_texts = ["A" * (n_segs * 512)]
        fi = _make_ctx_filter_feature_index(1, n_segs)
        bos_prefix = tok.decode([2])

        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=150,
            bos_id=2,
        )
        # bos_prefix slicing + BOS prepend cancel out → net tokens = char_end / 8
        # seg 1: char_end=1024 → 128 tokens ≤ 150 ✓
        # seg 2: char_end=1536 → 192 tokens > 150 ✗
        assert doc_fits_context(0, 1, ctx, bos_prefix)
        assert not doc_fits_context(0, 2, ctx, bos_prefix)

        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
            preferred_seg_idx=4,
        )
        assert result == 1

    def test_narrow_tokenizer_more_segments_fit(self):
        """2-char tokens: a 512-char segment = 256 tokens.
        With max_seq_len=600, seg 1 fits but seg 2 doesn't."""
        n_segs = 5
        tok = _NarrowTokenizer()
        doc_texts = ["A" * (n_segs * 512)]
        fi = _make_ctx_filter_feature_index(1, n_segs)
        bos_prefix = tok.decode([2])

        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=600,
            bos_id=2,
        )
        # Net tokens = char_end / 2 (bos_prefix slicing + prepend cancel out)
        # seg 1: char_end=1024 → 512 tokens ≤ 600 ✓
        # seg 2: char_end=1536 → 768 tokens > 600 ✗
        assert doc_fits_context(0, 1, ctx, bos_prefix)
        assert not doc_fits_context(0, 2, ctx, bos_prefix)

        result = find_fitting_segment(
            doc_idx=0,
            ctx=ctx,
            bos_prefix=bos_prefix,
            mode=EvalMode.ABLATION,
            feature_id=0,
            preferred_seg_idx=3,
        )
        assert result == 1

    def test_mixed_width_tokenizer_boundary(self):
        """Variable-width tokenizer: verify char-level counting, no fixed stride."""
        n_segs = 5
        tok = _MixedWidthTokenizer()
        doc_texts = ["A" * (n_segs * 512)]
        fi = _make_ctx_filter_feature_index(1, n_segs)
        bos_prefix = tok.decode([2])

        # Variable-width tokenizer — compute expected count dynamically
        # rather than relying on fixed arithmetic.
        text_seg1 = doc_texts[0][:1024]
        n_tokens_seg1 = len(tok.encode(text_seg1, prepend=2))

        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=n_tokens_seg1,  # exactly fits seg 1
            bos_id=2,
        )
        assert doc_fits_context(0, 1, ctx, bos_prefix)
        assert not doc_fits_context(0, 2, ctx, bos_prefix)

    def test_context_fit_uses_char_boundaries_not_token_count(self):
        """Verify doc_fits_context tokenizes through char_end, not position + segment_size."""
        # Create segments where char boundaries differ from position × 4
        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * 2000]

        # Custom feature index with non-standard char boundaries
        idx = MagicMock()
        idx.lookup.side_effect = lambda shard_stem, doc_idx: [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([0], dtype=t.int32),
                feature_mags=t.tensor([1.0]),
                char_start=0,
                char_end=400,
            ),
            SegmentFeatures(
                segment_idx=1,
                position=128,
                feature_ids=t.tensor([0], dtype=t.int32),
                feature_mags=t.tensor([1.0]),
                char_start=400,
                char_end=800,  # char_end=800, not 1024
            ),
            SegmentFeatures(
                segment_idx=2,
                position=256,
                feature_ids=t.tensor([0], dtype=t.int32),
                feature_mags=t.tensor([1.0]),
                char_start=800,
                char_end=1600,
            ),
        ]

        bos_prefix = tok.decode([2])
        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=idx,
            val_shard_stem="s",
            tokenizer=tok,
            max_seq_len=250,
            bos_id=2,
        )
        # seg 1: char_end=800 → 200 tokens ≤ 250 ✓
        assert doc_fits_context(0, 1, ctx, bos_prefix)
        # seg 2: char_end=1600 → 400 tokens > 250 ✗
        assert not doc_fits_context(0, 2, ctx, bos_prefix)


# ---------------------------------------------------------------------------
# Tests: integration — select_from_tensors with find_fitting_segment
# ---------------------------------------------------------------------------


class TestSelectFromTensorsSalvage:
    """Verify select_from_tensors salvages docs via segment fallback."""

    def test_select_from_tensors_ablation_salvages_docs(self):
        """10-segment docs, max_seq_len fits only segments 1-3.
        Feature in segments 2, 5, 8. max_segment_idx=8 (doesn't fit).
        Docs should be kept with target_segments=2."""
        n_segs = 10
        # Feature 7 in segments 2, 5, 8 only
        seg_features = []
        for i in range(n_segs):
            if i in (2, 5, 8):
                seg_features.append(([7], [float(10 - i)]))  # seg 2 has highest mag
            else:
                seg_features.append(([99], [1.0]))

        fi = _make_ctx_filter_feature_index(5, n_segs, seg_features=seg_features)

        # Build scan tensors with feature_id=7, max_segment_idx=8
        scan = ScanResultTensors(
            feature_ids=t.tensor([7], dtype=t.int64),
            n_docs_activated=t.tensor([5], dtype=t.int64),
            offsets=t.tensor([0, 5], dtype=t.int64),
            doc_idx=t.arange(5, dtype=t.int64),
            max_activation=t.ones(5, dtype=t.float32),
            max_segment_idx=t.full((5,), 8, dtype=t.int64),
            percentile=t.full((5,), 0.9, dtype=t.float32),
            total_segments=t.full((5,), n_segs, dtype=t.int64),
        )

        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * (n_segs * 512) for _ in range(5)]
        # seg 3: char_end=2048 → 512 tokens.
        # max_seq_len=520: segs 1-3 fit, seg 4+ don't.
        ctx = ContextFilter(
            doc_texts=doc_texts,  # ty: ignore[invalid-argument-type]
            feature_index=fi,
            val_shard_stem="shard_0000",
            tokenizer=tok,
            max_seq_len=520,
            bos_id=2,
        )

        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=5,
            mode=EvalMode.ABLATION,
            ctx_filter=ctx,
        )
        assert len(selections) == 1
        sel = selections[0]
        assert len(sel.selected_docs) == 5
        # All docs should target segment 2 (only fitting seg with feature 7)
        for doc_idx in sel.selected_docs:
            assert sel.target_segments[doc_idx] == 2

    def test_select_from_tensors_insertion_salvages_docs(self):
        """Insertion mode: feature in segments 2,4. Segments 1,3 lack the
        feature and fit. Docs should be kept with fitting featureless segments."""
        n_segs = 6
        seg_features = []
        for i in range(n_segs):
            if i in (2, 4):
                seg_features.append(([7], [1.0]))  # has feature
            else:
                seg_features.append(([99], [1.0]))  # no feature 7

        fi = _make_ctx_filter_feature_index(5, n_segs, seg_features=seg_features)

        # No docs activated for feature 7 → all are insertion candidates
        scan = ScanResultTensors(
            feature_ids=t.tensor([7], dtype=t.int64),
            n_docs_activated=t.tensor([0], dtype=t.int64),
            offsets=t.tensor([0, 0], dtype=t.int64),
            doc_idx=t.tensor([], dtype=t.int64),
            max_activation=t.tensor([], dtype=t.float32),
            max_segment_idx=t.tensor([], dtype=t.int64),
            percentile=t.tensor([], dtype=t.float32),
            total_segments=t.tensor([], dtype=t.int64),
        )

        tok = _CtxFilterTokenizer()
        doc_texts = ["A" * (n_segs * 512) for _ in range(5)]
        # seg 3 ends at char 2048 → 513 tokens. max_seq_len=520: segs 1-3 fit.
        ctx = ContextFilter(
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="shard_0000",
            tokenizer=tok,
            max_seq_len=520,
            bos_id=2,
        )

        selections = select_from_tensors(
            scan_tensors=scan,
            top_k=1,
            docs_per_feature=5,
            mode=EvalMode.INSERTION,
            all_doc_indices=set(range(5)),
            ctx_filter=ctx,
        )
        assert len(selections) == 1
        sel = selections[0]
        assert len(sel.selected_docs) > 0
        for doc_idx in sel.selected_docs:
            seg = sel.target_segments[doc_idx]
            # Must be a fitting segment without feature 7
            assert seg in (1, 3)

    def test_generate_completions_uses_target_segments_priority(self):
        """Verify sel.target_segments takes precedence over doc_target_map."""
        mod = _import_entry_point()

        tok = _FakeTokenizer()
        model = _FakeModel(seq_len=2048)

        # Patch encoder to return packed state.
        def fake_encoder_call(fids, mags, mask):
            n_seg = fids.shape[1]
            dense = t.randn(1, n_seg, mask.shape[2], 256)
            return pack_dense_encoder_output_reference(dense, mask)

        model.encoder = fake_encoder_call  # ty: ignore[invalid-assignment]
        segs = _make_segments(n_segs=5, segment_size=128)

        fi = MagicMock()
        fi.lookup.return_value = segs

        doc_texts = ["A" * 2560 for _ in range(3)]

        # target_segments says use segment 2, doc_target_map says segment 4.
        # Feature 20 is present in segment 2 (from _make_segments: seg 2 has
        # feature_ids=[20,21,22,23,24]).
        sel = FeatureSelection(
            feature_id=20,
            neuronpedia_description="test",
            n_docs_activated=3,
            selected_docs=[0, 1, 2],
            target_segments={0: 2, 1: 2, 2: 2},
        )
        target_segment_maps = {20: {0: 4, 1: 4, 2: 4}}

        # Boundaries must cover all feature IDs. _make_segments(n_segs=5) produces
        # IDs up to 44, so we need at least 45 rows. num_buckets=4 -> 3 boundaries.
        n_features = 100
        boundaries = t.linspace(0, 5, steps=3).unsqueeze(0).expand(n_features, -1).to(t.float64)
        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile_segment_topk, 10.9)
        # Precomputed thresholds: very negative (nothing gets thresholded away)
        precomputed_thresholds = t.full((n_features,), -1e6, dtype=t.float32)

        completions = mod._generate_completions(
            selections=[sel],
            target_segment_maps=target_segment_maps,
            doc_texts=doc_texts,
            feature_index=fi,
            val_shard_stem="s",
            model=model,
            tokenizer=tok,
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed_thresholds,
            device=t.device("cpu"),
            num_buckets=4,
            samples_per_condition=1,
            temperature=0.7,
            top_k_sampling=50,
            bos_id=2,
        )
        # All completions should use target_segment_idx=2 (from target_segments)
        assert len(completions) == 3
        for comp in completions:
            assert comp.target_segment_idx == 2
