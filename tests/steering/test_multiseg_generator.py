# tests/test_multiseg_generator.py
"""Tests for multi-segment conditioning builder and generator."""

import pytest
import torch as t

from nanochat.sampling import sample_next_token as _sample_token
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.completion_steering_eval import EvalMode
from steering.eval.multiseg_generator import (
    MultisegInputs,
    _segment_bounds_for_assignment,
    ablate_feature,
    add_feature,
    build_ground_truth_signals,
    build_multiseg_inputs,
    prepare_segments,
)

# Passthrough config: segment_topk with k large enough to keep all features.
_PASSTHROUGH_CFG = ThresholdConfig(ThresholdType.segment_topk, 99999)


class _FixedWidthTokenizer:
    """Mock tokenizer where each token decodes to exactly 4 characters.

    With segment char boundaries at multiples of 512 and 128 tokens per
    segment, 128 tokens × 4 chars = 512 chars aligns exactly.
    """

    def decode(self, token_ids):
        return "x" * (len(token_ids) * 4)


_MOCK_TOK = _FixedWidthTokenizer()


def _make_segments(n_segs: int = 3, features_per_seg: int = 5) -> list[SegmentFeatures]:
    """Create synthetic segments with known features."""
    segs = []
    for i in range(n_segs):
        fids = t.arange(i * 10, i * 10 + features_per_seg, dtype=t.int32)
        mags = t.ones(features_per_seg, dtype=t.float32) * (i + 1.0)
        segs.append(
            SegmentFeatures(
                segment_idx=i,
                position=i * 128,
                feature_ids=fids,
                feature_mags=mags,
                char_start=i * 512,
                char_end=(i + 1) * 512,
            )
        )
    return segs


class TestBuildMultisegInputs:
    def test_seg_map_char_level(self):
        segs = _make_segments(n_segs=3)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        seg_map = result.seg_map
        T = 256 + 128
        assert seg_map.shape == (1, T)
        assert (seg_map[0, :128] == 0).all()
        assert (seg_map[0, 128:256] == 1).all()
        assert (seg_map[0, 256:] == 2).all()

    def test_encoder_input_shapes(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        n_seg = 3
        F_max = 5
        assert result.enc_feature_ids.shape == (1, n_seg, F_max)
        assert result.enc_magnitudes.shape == (1, n_seg, F_max)
        assert result.enc_mask.shape == (1, n_seg, F_max)

    def test_mask_true_for_real_features(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert result.enc_mask.all()

    def test_padding_when_unequal_features(self):
        seg0 = SegmentFeatures(
            0, 0, t.tensor([1, 2, 3], dtype=t.int32), t.tensor([1.0, 2.0, 3.0]), 0, 128
        )
        seg1 = SegmentFeatures(
            1, 128, t.tensor([4], dtype=t.int32), t.tensor([4.0]), 128, 256
        )
        result = build_multiseg_inputs(
            segments=[seg0, seg1],
            prefix_tokens=list(range(64)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert result.enc_feature_ids.shape == (1, 2, 3)
        assert result.enc_mask[0, 1, 0]
        assert not result.enc_mask[0, 1, 1]
        assert not result.enc_mask[0, 1, 2]

class TestCharLevelSegMapEdgeCases:
    """Tests exercising the character-level seg_map logic, including
    cross-tokenizer scenarios that would fail with the old stride-based approach.
    """

    def test_cross_tokenizer_different_width(self):
        """When token width differs from extraction stride, char-level produces
        different (correct) assignments vs the old pos // segment_size formula.

        With 8-char tokens and segments at [0,512) and [512,1024):
        - 64 tokens = 512 chars = 1 segment
        - Old stride-based (pos // 128) would assign token 64 to seg 0
        - Char-level correctly assigns token 64 to seg 1
        """

        class _WideTokenizer:
            """Each token = 8 chars: 64 tokens per segment."""

            def decode(self, token_ids):
                return "x" * (len(token_ids) * 8)

        segs = _make_segments(n_segs=2)  # [0,512) and [512,1024)
        # 128 tokens × 8 chars = 1024 chars, covering both segments
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=64,
            target_segment_idx=1,
            tokenizer=_WideTokenizer(),
        )
        seg_map = result.seg_map[0]
        # First 64 tokens (chars 0–512) → seg 0
        assert (seg_map[:64] == 0).all()
        # Next 64 tokens (chars 512–1024) → seg 1
        assert (seg_map[64:128] == 1).all()
        # Gen tokens → target segment
        assert (seg_map[128:] == 1).all()

    def test_boundary_straddling_token(self):
        """A token straddling a segment boundary is assigned by max overlap."""

        class _SixCharTokenizer:
            """Each token = 6 chars."""

            def decode(self, token_ids):
                return "x" * (len(token_ids) * 6)

        segs = _make_segments(n_segs=2)  # [0,512) and [512,1024)
        # 170 tokens × 6 chars = 1020 chars
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(170)),
            n_gen_tokens=10,
            target_segment_idx=1,
            tokenizer=_SixCharTokenizer(),
        )
        seg_map = result.seg_map[0]
        # Token 85: chars [510, 516) — straddles boundary at 512
        # Overlap with seg 0: 512-510 = 2, seg 1: 516-512 = 4 → seg 1
        assert seg_map[85].item() == 1
        # Token 84: chars [504, 510) — entirely in seg 0
        assert seg_map[84].item() == 0

    def test_long_prefix_covers_all_segments(self):
        """A prefix covering all segment char ranges assigns correctly."""
        segs = _make_segments(n_segs=3)  # [0,512), [512,1024), [1024,1536)
        # 384 tokens × 4 chars = 1536 chars, covering all 3 segments
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(384)),
            n_gen_tokens=64,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        seg_map = result.seg_map[0]
        assert (seg_map[:128] == 0).all()  # chars 0–512
        assert (seg_map[128:256] == 1).all()  # chars 512–1024
        assert (seg_map[256:384] == 2).all()  # chars 1024–1536
        assert (seg_map[384:] == 2).all()  # gen tokens

    def test_empty_prefix(self):
        """Empty prefix — seg_map is all target_segment_idx."""
        segs = _make_segments(n_segs=2)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=[],
            n_gen_tokens=64,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert result.seg_map.shape == (1, 64)
        assert (result.seg_map == 1).all()

    def test_single_segment(self):
        """With one segment, all positions map to index 0."""
        segs = _make_segments(n_segs=1)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=64,
            target_segment_idx=0,
            tokenizer=_MOCK_TOK,
        )
        assert (result.seg_map == 0).all()

    def test_tokens_past_last_segment_raise(self):
        """Prefixes extending beyond available segment chars fail loudly."""
        segs = _make_segments(n_segs=2)  # [0,512) and [512,1024)
        # 400 tokens × 4 = 1600 chars — last 150 tokens are past both segments
        with pytest.raises(ValueError, match="zero overlap"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(400)),
                n_gen_tokens=10,
                target_segment_idx=1,
                tokenizer=_MOCK_TOK,
            )

    def test_exact_leading_bos_span_is_aligned(self):
        """Only the exact BOS span before segment 0 is folded into segment 0."""
        bos_width = 4
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=bos_width,
                char_end=bos_width + 512,
            )
        ]
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(129)),
            n_gen_tokens=1,
            target_segment_idx=0,
            tokenizer=_MOCK_TOK,
        )
        assert (result.seg_map == 0).all()

    def test_multiple_leading_tokens_before_first_segment_raise(self):
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=8,
                char_end=520,
            )
        ]
        with pytest.raises(ValueError, match="token\\(s\\) precede"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(3)),
                n_gen_tokens=1,
                target_segment_idx=0,
                tokenizer=_MOCK_TOK,
            )

    def test_non_contiguous_segment_bounds_raise_before_assignment(self):
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=0,
                char_end=512,
            ),
            SegmentFeatures(
                segment_idx=1,
                position=128,
                feature_ids=t.tensor([2], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=520,
                char_end=1024,
            ),
        ]
        with pytest.raises(ValueError, match="tile contiguously"):
            _segment_bounds_for_assignment(
                segs,
                t.tensor([0, 4], dtype=t.int64),
                t.tensor([4, 8], dtype=t.int64),
            )

    def test_non_bos_leading_gap_raises(self):
        """A leading coordinate gap larger than the first token is an error."""
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=8,
                char_end=520,
            )
        ]
        with pytest.raises(ValueError, match="precede the first segment"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(130)),
                n_gen_tokens=1,
                target_segment_idx=0,
                tokenizer=_MOCK_TOK,
            )

    @pytest.mark.parametrize(
        ("first_end", "second_start"),
        [(512, 520), (512, 500)],
    )
    def test_non_contiguous_segment_ranges_raise(
        self,
        first_end: int,
        second_start: int,
    ):
        """Segment ranges must tile exactly; gaps and overlaps are errors."""
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=0,
                char_end=first_end,
            ),
            SegmentFeatures(
                segment_idx=1,
                position=128,
                feature_ids=t.tensor([2], dtype=t.int32),
                feature_mags=t.tensor([2.0], dtype=t.float32),
                char_start=second_start,
                char_end=1024,
            ),
        ]
        with pytest.raises(ValueError, match="tile contiguously"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(128)),
                n_gen_tokens=1,
                target_segment_idx=1,
                tokenizer=_MOCK_TOK,
            )

    def test_zero_width_generation_target_is_explicitly_allowed(self):
        """A zero-width target slot is valid when only generated tokens use it."""
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=0,
                char_end=512,
            ),
            SegmentFeatures(
                segment_idx=1,
                position=128,
                feature_ids=t.tensor([2], dtype=t.int32),
                feature_mags=t.tensor([2.0], dtype=t.float32),
                char_start=512,
                char_end=512,
            ),
        ]
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=3,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert (result.seg_map[0, :128] == 0).all()
        assert (result.seg_map[0, 128:] == 1).all()

    def test_zero_width_non_target_segment_raises(self):
        """Only the generation target may be zero-width."""
        segs = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([1], dtype=t.int32),
                feature_mags=t.tensor([1.0], dtype=t.float32),
                char_start=0,
                char_end=0,
            ),
            SegmentFeatures(
                segment_idx=1,
                position=128,
                feature_ids=t.tensor([2], dtype=t.int32),
                feature_mags=t.tensor([2.0], dtype=t.float32),
                char_start=0,
                char_end=512,
            ),
        ]
        with pytest.raises(ValueError, match="non-positive char span"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=list(range(128)),
                n_gen_tokens=3,
                target_segment_idx=1,
                tokenizer=_MOCK_TOK,
            )


class TestFullContextSegmentConditioning:
    """Tests for direct full-context routing in build_multiseg_inputs."""

    def test_generated_tokens_attend_to_target_segment_slot(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        seg_map = result.seg_map[0]
        assert (seg_map[:128] == 0).all()
        assert (seg_map[128:] == 2).all()
        assert result.enc_mask.shape[1] == 3
        assert t.equal(
            result.enc_feature_ids[0, 2, :5],
            segs[2].feature_ids.to(t.int64),
        )


class TestNextSegmentConditioningOffset:
    """Tests for next_segment_conditioning_offset in build_multiseg_inputs."""

    def test_offset_seg_map_routes_generated_tokens_to_previous_slot(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
            next_segment_conditioning_offset=True,
        )
        seg_map = result.seg_map[0]
        assert (seg_map[:128] == 0).all()
        assert (seg_map[128:] == 1).all()

    def test_offset_features_shifted_by_one_source_segment(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
            next_segment_conditioning_offset=True,
        )
        assert result.enc_mask.shape[1] == 2
        assert t.equal(
            result.enc_feature_ids[0, 0, :5],
            segs[1].feature_ids.to(t.int64),
        )
        assert t.equal(
            result.enc_feature_ids[0, 1, :5],
            segs[2].feature_ids.to(t.int64),
        )

    def test_offset_requires_nonzero_target_segment(self):
        segs = _make_segments(n_segs=2, features_per_seg=5)
        with pytest.raises(ValueError, match="target_segment_idx >= 1"):
            build_multiseg_inputs(
                segments=segs,
                prefix_tokens=[],
                n_gen_tokens=16,
                target_segment_idx=0,
                tokenizer=_MOCK_TOK,
                next_segment_conditioning_offset=True,
            )

    def test_no_offset_unchanged_when_explicit_false(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        default_result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        explicit_result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
            next_segment_conditioning_offset=False,
        )
        assert t.equal(default_result.seg_map, explicit_result.seg_map)
        assert t.equal(default_result.enc_feature_ids, explicit_result.enc_feature_ids)
        assert t.equal(default_result.enc_mask, explicit_result.enc_mask)


class TestAblateFeature:
    def test_removes_feature_from_target_only(self):
        segs = _make_segments(n_segs=3, features_per_seg=5)
        target_seg_idx = 2
        ablated = ablate_feature(segs, target_seg_idx, feature_id=20)
        target = ablated[target_seg_idx]
        assert 20 not in target.feature_ids.tolist()
        assert len(target.feature_ids) == 4
        assert ablated[0].feature_ids.tolist() == segs[0].feature_ids.tolist()
        assert ablated[1].feature_ids.tolist() == segs[1].feature_ids.tolist()

    def test_ablate_nonexistent_feature_raises(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        with pytest.raises(ValueError, match="not present"):
            ablate_feature(segs, target_seg_idx=1, feature_id=999)


class TestAddFeature:
    def test_adds_feature_to_target(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        added = add_feature(segs, target_seg_idx=1, feature_id=999, bucket_index=5)
        target = added[1]
        assert 999 in target.feature_ids.tolist()
        idx = target.feature_ids.tolist().index(999)
        assert target.feature_mags[idx].item() == 5.0

    def test_add_does_not_modify_other_segments(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        added = add_feature(segs, target_seg_idx=1, feature_id=999, bucket_index=3)
        assert added[0].feature_ids.tolist() == segs[0].feature_ids.tolist()

    def test_add_duplicate_raises(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        existing_fid = segs[1].feature_ids[0].item()
        with pytest.raises(ValueError, match="already present"):
            add_feature(segs, target_seg_idx=1, feature_id=existing_fid, bucket_index=7)  # ty: ignore[invalid-argument-type]


class TestPrepareSegments:
    def test_discretizes_activations(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0, 9.0], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[0] = t.tensor([3.0, 6.0])
        boundaries[1] = t.tensor([3.0, 6.0])
        boundaries[2] = t.tensor([3.0, 6.0])

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        assert len(prepared) == 1
        # segment_topk may reorder features; verify feature→bucket mapping
        fid_to_bucket = dict(
            zip(
                prepared[0].feature_ids.tolist(),
                prepared[0].feature_mags.tolist(), strict=True,
            )
        )
        assert fid_to_bucket == {0: 0, 1: 1, 2: 2}

    def test_thresholding_filters_features(self):
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0, 9.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        precomputed = t.tensor([4.0, 4.0, 4.0], dtype=t.float64)
        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, threshold=0.9)

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        surviving_fids = prepared[0].feature_ids.tolist()
        assert 0 not in surviving_fids
        assert 1 in surviving_fids
        assert 2 in surviving_fids

    def test_feature_mask_filters_before_thresholding(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([10.0, 9.0, 1.0], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])
        threshold_cfg = ThresholdConfig(ThresholdType.segment_topk, threshold=2)
        feature_mask = FeatureMask(FeatureMaskMode.blacklist, (0,))

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            feature_mask=feature_mask,
        )

        assert prepared[0].feature_ids.tolist() == [1, 2]

    def test_segment_topk_caps_features(self):
        fids = t.arange(100, dtype=t.int32)
        mags = t.arange(100, dtype=t.float32) + 1.0
        seg = SegmentFeatures(1, 128, fids, mags, 128, 256)

        boundaries = t.zeros(100, 2, dtype=t.float64)
        boundaries[:, 0] = 30.0
        boundaries[:, 1] = 60.0
        precomputed = t.zeros(100, dtype=t.float64)

        threshold_cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, threshold=10.9
        )

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        assert len(prepared[0].feature_ids) <= 10

    def test_empty_segment_handled(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([], dtype=t.int32),
            feature_mags=t.tensor([], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        boundaries = t.zeros(10, 2, dtype=t.float64)
        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        assert len(prepared[0].feature_ids) == 0
        assert len(prepared[0].feature_mags) == 0

    def test_original_segments_unchanged(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        original_mags = seg.feature_mags.clone()
        boundaries = t.zeros(2, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])
        prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        assert t.equal(seg.feature_mags, original_mags)


class TestSampleToken:
    def test_greedy_returns_argmax(self):
        logits = t.tensor([[0.1, 0.9, 0.3]])
        result = _sample_token(logits, rng=None, temperature=0.0, top_k=None)
        assert result.item() == 1

    def test_topk_restricts_vocabulary(self):
        logits = t.zeros(1, 100)
        logits[0, 50] = 10.0  # dominant token
        rng = t.Generator()
        rng.manual_seed(42)
        result = _sample_token(logits, rng=rng, temperature=1.0, top_k=5)
        assert result.item() == 50


class TestMultisegInputsDataclass:
    """Tests that MultisegInputs is a proper frozen dataclass."""

    def test_returns_multiseg_inputs_type(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert isinstance(result, MultisegInputs)

    def test_frozen_immutable(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        with pytest.raises(AttributeError):
            result.seg_map = t.zeros(1, 1)  # ty: ignore[invalid-assignment]

    def test_all_fields_present(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        result = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        assert result.enc_feature_ids is not None
        assert result.enc_magnitudes is not None
        assert result.enc_mask is not None
        assert result.seg_map is not None


class TestPrepareSegmentsOutputDtype:
    """Verify prepare_segments outputs int64 bucket indices directly."""

    def test_output_dtype_is_int64(self):
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        boundaries = t.zeros(2, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])
        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        assert prepared[0].feature_mags.dtype == t.int64

    def test_output_values_are_bucket_indices(self):
        """Bucket indices should be non-negative integers."""
        seg = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0, 9.0], dtype=t.float32),
            char_start=0,
            char_end=128,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])
        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        buckets = prepared[0].feature_mags
        assert (buckets >= 0).all()
        # segment_topk may reorder features; verify feature→bucket mapping
        fid_to_bucket = dict(
            zip(
                prepared[0].feature_ids.tolist(),
                prepared[0].feature_mags.tolist(), strict=True,
            )
        )
        assert fid_to_bucket == {0: 0, 1: 1, 2: 2}


class TestPrepareAblateIntegration:
    """End-to-end: prepare_segments → ablate_feature must work together."""

    def test_ablate_after_prepare(self):
        """Feature that survives thresholding can be ablated."""
        target_fid = 1
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([5.0, 5.0, 5.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        # Feature 1 should still be present after discretization
        assert target_fid in prepared[0].feature_ids.tolist()

        ablated = ablate_feature(prepared, target_seg_idx=0, feature_id=target_fid)
        assert target_fid not in ablated[0].feature_ids.tolist()
        # Other features should remain
        assert 0 in ablated[0].feature_ids.tolist()
        assert 2 in ablated[0].feature_ids.tolist()

    def test_ablate_after_threshold_removes_feature_raises(self):
        """Feature removed by thresholding cannot be ablated — raises ValueError."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([0.1, 5.0, 9.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        # Threshold that removes feature 0 (activation 0.1 < threshold 4.0)
        precomputed = t.tensor([4.0, 4.0, 4.0], dtype=t.float64)
        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, threshold=0.9)

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        # Feature 0 should have been thresholded out
        assert 0 not in prepared[0].feature_ids.tolist()

        with pytest.raises(ValueError, match="not present"):
            ablate_feature(prepared, target_seg_idx=0, feature_id=0)


class TestPrepareAddIntegration:
    """End-to-end: prepare_segments → add_feature must work together."""

    def test_add_novel_feature_after_prepare(self):
        """Feature not in segment can be added with a bucket index."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1], dtype=t.int32),
            feature_mags=t.tensor([5.0, 5.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(2, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        added = add_feature(prepared, target_seg_idx=0, feature_id=999, bucket_index=2)
        assert 999 in added[0].feature_ids.tolist()
        idx = added[0].feature_ids.tolist().index(999)
        # Bucket index should be stored as int64
        assert added[0].feature_mags[idx].item() == 2
        assert added[0].feature_mags.dtype == t.int64

    def test_add_feature_already_present_after_prepare_raises(self):
        """Feature that survived thresholding cannot be added — raises ValueError."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1], dtype=t.int32),
            feature_mags=t.tensor([5.0, 5.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(2, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        with pytest.raises(ValueError, match="already present"):
            add_feature(prepared, target_seg_idx=0, feature_id=1, bucket_index=5)


class TestBuildMultisegAfterAblation:
    """End-to-end: ablate_feature → build_multiseg_inputs produces correct shapes."""

    def test_ablated_inputs_have_fewer_masked_features(self):
        segs = _make_segments(n_segs=2, features_per_seg=5)
        target_fid = segs[1].feature_ids[2].item()

        real_inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        ablated_segs = ablate_feature(segs, target_seg_idx=1, feature_id=target_fid)  # ty: ignore[invalid-argument-type]
        ablated_inputs = build_multiseg_inputs(
            segments=ablated_segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )

        # Real has 5 features in seg 1, ablated has 4
        real_mask_count = real_inputs.enc_mask[0, 1].sum().item()
        ablated_mask_count = ablated_inputs.enc_mask[0, 1].sum().item()
        assert real_mask_count == 5
        assert ablated_mask_count == 4

        # Seg 0 should be unchanged
        assert t.equal(
            real_inputs.enc_feature_ids[0, 0],
            ablated_inputs.enc_feature_ids[0, 0, : real_inputs.enc_feature_ids.size(2)],
        )


class TestBuildMultisegAfterAddition:
    """End-to-end: add_feature → build_multiseg_inputs produces correct shapes."""

    def test_added_inputs_have_more_masked_features(self):
        segs = _make_segments(n_segs=2, features_per_seg=3)
        added_segs = add_feature(segs, target_seg_idx=1, feature_id=999, bucket_index=5)

        real_inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )
        added_inputs = build_multiseg_inputs(
            segments=added_segs,
            prefix_tokens=list(range(128)),
            n_gen_tokens=128,
            target_segment_idx=1,
            tokenizer=_MOCK_TOK,
        )

        real_mask_count = real_inputs.enc_mask[0, 1].sum().item()
        added_mask_count = added_inputs.enc_mask[0, 1].sum().item()
        assert real_mask_count == 3
        assert added_mask_count == 4


# ---------------------------------------------------------------------------
# Test: build_ground_truth_signals
# ---------------------------------------------------------------------------


class TestBuildGroundTruthSignals:
    """Tests for the build_ground_truth_signals helper."""

    def _make_prepared_segment(
        self, fids: list[int], mags: list[int]
    ) -> SegmentFeatures:
        """Create a prepared segment (int64 mags = bucket indices)."""
        return SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor(fids, dtype=t.int32),
            feature_mags=t.tensor(mags, dtype=t.int64),
            char_start=128,
            char_end=256,
        )

    def test_known_three_features(self):
        seg = self._make_prepared_segment([10, 20, 30], [1, 2, 3])
        result = build_ground_truth_signals(seg, feature_id=20, mode=EvalMode.ABLATION)
        assert result.feature_signals == {10: 1, 20: 2, 30: 3}
        assert result.n_features_active == 3
        assert result.avg_magnitude == 2.0
        assert result.min_magnitude == 1
        assert result.max_magnitude == 3
        assert result.feature_of_interest_magnitude == 2

    def test_feature_of_interest_present(self):
        seg = self._make_prepared_segment([5, 10], [3, 1])
        result = build_ground_truth_signals(seg, feature_id=5, mode=EvalMode.ABLATION)
        assert result.feature_of_interest_magnitude == 3

    def test_feature_of_interest_absent_insertion_mode(self):
        """In insertion mode, feature must NOT be in segment."""
        seg = self._make_prepared_segment([5, 10], [3, 1])
        result = build_ground_truth_signals(
            seg, feature_id=999, mode=EvalMode.INSERTION
        )
        assert result.feature_of_interest_magnitude is None

    def test_empty_segment_insertion_mode(self):
        """Empty segment with insertion mode (feature absent)."""
        seg = self._make_prepared_segment([], [])
        result = build_ground_truth_signals(seg, feature_id=42, mode=EvalMode.INSERTION)
        assert result.feature_signals == {}
        assert result.n_features_active == 0
        assert result.avg_magnitude == 0.0
        assert result.min_magnitude == 0
        assert result.max_magnitude == 0
        assert result.feature_of_interest_magnitude is None

    def test_single_feature(self):
        seg = self._make_prepared_segment([42], [5])
        result = build_ground_truth_signals(seg, feature_id=42, mode=EvalMode.ABLATION)
        assert result.n_features_active == 1
        assert result.avg_magnitude == 5.0
        assert result.min_magnitude == 5
        assert result.max_magnitude == 5
        assert result.feature_of_interest_magnitude == 5

    def test_all_same_bucket(self):
        seg = self._make_prepared_segment([1, 2, 3], [2, 2, 2])
        result = build_ground_truth_signals(seg, feature_id=1, mode=EvalMode.ABLATION)
        assert result.avg_magnitude == 2.0
        assert result.min_magnitude == 2
        assert result.max_magnitude == 2

    def test_n_features_equals_len_signals(self):
        seg = self._make_prepared_segment([10, 20, 30, 40], [0, 1, 2, 3])
        result = build_ground_truth_signals(seg, feature_id=10, mode=EvalMode.ABLATION)
        assert result.n_features_active == len(result.feature_signals)

    def test_feature_of_interest_matches_dict(self):
        seg = self._make_prepared_segment([10, 20, 30], [1, 2, 3])
        result = build_ground_truth_signals(seg, feature_id=20, mode=EvalMode.ABLATION)
        assert result.feature_of_interest_magnitude == result.feature_signals[20]

    def test_ablation_asserts_feature_present(self):
        """Ablation mode raises AssertionError when feature is absent."""
        seg = self._make_prepared_segment([5, 10], [3, 1])
        with pytest.raises(AssertionError, match="must be present"):
            build_ground_truth_signals(seg, feature_id=999, mode=EvalMode.ABLATION)

    def test_insertion_asserts_feature_absent(self):
        """Insertion mode raises AssertionError when feature is present."""
        seg = self._make_prepared_segment([5, 10], [3, 1])
        with pytest.raises(AssertionError, match="must NOT be present"):
            build_ground_truth_signals(seg, feature_id=5, mode=EvalMode.INSERTION)


class TestBuildGroundTruthSignalsWithPrepareSegments:
    """Integration: prepare_segments → build_ground_truth_signals."""

    def test_signals_from_prepared_segment(self):
        """Verify signals use discretized bucket indices from prepare_segments."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([1.0, 5.0, 9.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
        )
        result = build_ground_truth_signals(
            prepared[0], feature_id=1, mode=EvalMode.ABLATION
        )
        # buckets: 1.0 → 0, 5.0 → 1, 9.0 → 2
        assert result.feature_signals == {0: 0, 1: 1, 2: 2}
        assert result.feature_of_interest_magnitude == 1
        assert result.n_features_active == 3

    def test_signals_after_thresholding(self):
        """Features removed by threshold should not appear in signals."""
        seg = SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor([0, 1, 2], dtype=t.int32),
            feature_mags=t.tensor([0.1, 5.0, 9.0], dtype=t.float32),
            char_start=128,
            char_end=256,
        )
        boundaries = t.zeros(3, 2, dtype=t.float64)
        boundaries[:] = t.tensor([3.0, 6.0])
        precomputed = t.tensor([4.0, 4.0, 4.0], dtype=t.float64)
        threshold_cfg = ThresholdConfig(ThresholdType.feature_percentile, threshold=0.9)

        prepared = prepare_segments(
            segments=[seg],
            boundaries=boundaries,
            feature_cfg=threshold_cfg,
            precomputed_thresholds=precomputed,
        )
        # Feature 0 was thresholded out — use insertion mode (feature absent)
        result = build_ground_truth_signals(
            prepared[0], feature_id=0, mode=EvalMode.INSERTION
        )
        assert 0 not in result.feature_signals
        assert result.feature_of_interest_magnitude is None
        assert result.n_features_active == 2


class TestGroundTruthSignalsInvariants:
    """Property-based invariants for GroundTruthFeatureSignals."""

    def _make_prepared_segment(
        self, fids: list[int], mags: list[int]
    ) -> SegmentFeatures:
        return SegmentFeatures(
            segment_idx=1,
            position=128,
            feature_ids=t.tensor(fids, dtype=t.int32),
            feature_mags=t.tensor(mags, dtype=t.int64),
            char_start=128,
            char_end=256,
        )

    def test_n_features_active_non_negative(self):
        # feature present → ABLATION; feature absent → INSERTION
        cases = [
            ([1, 2], [0, 1], 1, EvalMode.ABLATION),
            ([], [], 42, EvalMode.INSERTION),
            ([5], [3], 5, EvalMode.ABLATION),
        ]
        for fids, mags, fid, mode in cases:
            result = build_ground_truth_signals(
                self._make_prepared_segment(fids, mags), fid, mode=mode
            )
            assert result.n_features_active >= 0

    def test_min_le_avg_le_max(self):
        seg = self._make_prepared_segment([1, 2, 3, 4], [0, 1, 2, 3])
        result = build_ground_truth_signals(seg, feature_id=1, mode=EvalMode.ABLATION)
        assert result.min_magnitude <= result.avg_magnitude <= result.max_magnitude

    def test_len_signals_equals_n_active(self):
        seg = self._make_prepared_segment([10, 20, 30], [1, 2, 3])
        result = build_ground_truth_signals(seg, feature_id=10, mode=EvalMode.ABLATION)
        assert len(result.feature_signals) == result.n_features_active

    def test_foi_equals_dict_value(self):
        seg = self._make_prepared_segment([10, 20], [5, 7])
        result = build_ground_truth_signals(seg, feature_id=10, mode=EvalMode.ABLATION)
        if result.feature_of_interest_magnitude is not None:
            assert result.feature_of_interest_magnitude == result.feature_signals[10]
