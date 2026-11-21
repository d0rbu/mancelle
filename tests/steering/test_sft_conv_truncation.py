"""Full-sequence SFT truncation and segment-routing tests."""

from __future__ import annotations

import pytest
import torch as t

from scripts.train.conditional_encdec_sft import (
    SFTState,
    _compute_full_coverage_gap,
    _source_aligned_char_offsets,
    _truncate_seg_features,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder.dataloader import SegFeatures
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft.loss_mask import LossMaskMode, truncate_conv_and_mask
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    _make_boundaries,
    build_synthetic_features,
    make_lazy_patched_sft_generator,
)


def _make_seg(char_start: int, char_end: int, n_features: int = 3) -> SegFeatures:
    return SegFeatures(
        feature_ids=t.arange(n_features, dtype=t.int64),
        buckets=t.ones(n_features, dtype=t.int64),
        percentiles=t.full((n_features,), 0.5, dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


class TinyConversationTask(Task):
    def __init__(self, conversations: list[dict]) -> None:
        super().__init__()
        self._conversations = conversations

    def num_examples(self) -> int:
        return len(self._conversations)

    def get_example(self, index: int) -> dict:
        return self._conversations[index]


def _make_conditioned_generator(
    task: Task,
    tokenizer: SFTMockTokenizer,
    *,
    B: int,
    T: int,
    min_loss_tokens: int,
    loss_mask_mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
):
    features = build_synthetic_features(task, tokenizer)
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=1000,
    )

    return make_lazy_patched_sft_generator(
        features,
        lambda: make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device="cpu",
            sft_state=SFTState(),
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg,
            boundaries=_make_boundaries(),
            min_loss_tokens=min_loss_tokens,
            loss_mask_mode=loss_mask_mode,
            alignment_probe_samples=0,
            expected_split=SFTSplit.TRAIN,
        ),
    )


class TestTruncateConversation:
    def test_short_conv_unchanged(self):
        ids = list(range(500))
        mask = [0] * len(ids)
        t_ids, t_mask, orig_len, was_truncated = truncate_conv_and_mask(ids, mask, 1025)
        assert t_ids == ids
        assert t_mask == mask
        assert orig_len == 500
        assert was_truncated is False

    def test_exact_capacity_unchanged(self):
        ids = list(range(1025))
        mask = [0] * len(ids)
        t_ids, t_mask, orig_len, was_truncated = truncate_conv_and_mask(ids, mask, 1025)
        assert len(t_ids) == 1025
        assert len(t_mask) == 1025
        assert orig_len == 1025
        assert was_truncated is False

    def test_long_conv_truncated_ids(self):
        ids = list(range(2000))
        mask = [i % 2 for i in ids]
        t_ids, t_mask, orig_len, was_truncated = truncate_conv_and_mask(ids, mask, 1025)
        assert t_ids == ids[:1025]
        assert t_mask == mask[:1025]
        assert orig_len == 2000
        assert was_truncated is True

    def test_empty_conv_passthrough(self):
        t_ids, t_mask, orig_len, was_truncated = truncate_conv_and_mask([], [], 1025)
        assert t_ids == []
        assert t_mask == []
        assert orig_len == 0
        assert was_truncated is False


class TestFullSequenceLossFiltering:
    def test_min_loss_tokens_counts_all_shifted_tokens(self):
        tokenizer = SFTMockTokenizer()
        conversation = {
            "messages": [
                {"role": "user", "content": "abcd"},
                {"role": "assistant", "content": "efgh"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        task = TinyConversationTask([conversation])

        gen = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=len(ids) - 1,
            min_loss_tokens=len(ids) - 1,
        )
        _inputs, targets, *_ = next(gen)

        assert (targets != -1).sum().item() == len(ids) - 1
        assert sum(mask[1:]) < len(ids) - 1

    def test_rows_mask_only_padding(self):
        tokenizer = SFTMockTokenizer()
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ]
        task = TinyConversationTask(conversations)

        gen = _make_conditioned_generator(
            task=task,
            tokenizer=tokenizer,
            B=1,
            T=32,
            min_loss_tokens=1,
        )
        inputs, targets, *_ = next(gen)

        supervised = targets[0] != -1
        assert supervised.any()
        first_padding_target = int(supervised.sum().item())
        assert t.equal(
            targets[0, :first_padding_target], inputs[0, 1 : first_padding_target + 1].to(t.int64)
        )
        assert (targets[0, first_padding_target:] == -1).all()


class TestTruncateSegFeatures:
    def test_all_segments_in_range(self):
        segs = [_make_seg(i * 100, (i + 1) * 100) for i in range(5)]
        result = _truncate_seg_features(segs, total_chars=600)
        assert result is not None
        assert len(result) == 5
        assert all(not s.prediction_target_only for s in result)

    def test_truncation_keeps_one_prediction_target_segment(self):
        segs = [_make_seg(i * 100, (i + 1) * 100) for i in range(8)]
        result = _truncate_seg_features(segs, total_chars=350)
        assert result is not None
        assert [(s.char_start, s.char_end) for s in result[:4]] == [
            (0, 100),
            (100, 200),
            (200, 300),
            (300, 350),
        ]
        assert result[4].prediction_target_only
        assert result[4].char_start == 400

    def test_none_and_empty_return_none(self):
        assert _truncate_seg_features(None, total_chars=100) is None
        assert _truncate_seg_features([], total_chars=100) is None


class TestFullConversationFeatureRouting:
    def test_source_aligned_offsets_shift_decoder_bos_to_source_bos(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, 10, 11, 12]

        raw_offsets = _source_aligned_char_offsets(tokenizer, ids, source_bos_char_len=1)
        shifted_offsets = _source_aligned_char_offsets(tokenizer, ids, source_bos_char_len=5)

        assert shifted_offsets[0] == 0
        assert shifted_offsets[1:] == [offset + 4 for offset in raw_offsets[1:]]

    def test_full_conv_segments_route_every_covered_token(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, 10, 11, 12, 13, 14]
        total_chars = _source_aligned_char_offsets(tokenizer, ids)[-1]
        seg_features = (
            [t.tensor([1]), t.tensor([2])],
            [t.tensor([1]), t.tensor([1])],
            [0, total_chars // 2],
            [total_chars // 2, total_chars],
        )

        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=seg_features,
        )
        fids = routing.per_seg_fids
        buckets = routing.per_seg_buckets
        seg_map = routing.seg_map

        assert len(fids) == len(buckets) == 3
        assert set(seg_map.tolist()) == {1, 2}
        assert seg_map[0].item() == 1
        assert seg_map[-1].item() == 2

    def test_full_conv_segments_reject_missing_features(self):
        tokenizer = SFTMockTokenizer()

        with pytest.raises(ValueError, match="seg_features is required"):
            build_full_sequence_segment_routing(
                ids=[0, 10, 11],
                tokenizer=tokenizer,
                seg_features=None,  # ty: ignore[invalid-argument-type]
            )

    def test_full_conv_segments_extend_small_trailing_coverage_gap(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, 10, 11, 12, 13, 14]
        total_chars = _source_aligned_char_offsets(tokenizer, ids)[-1]
        seg_features = (
            [t.tensor([7])],
            [t.tensor([1])],
            [0],
            [total_chars - 1],
        )

        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=seg_features,
        )

        assert len(routing.per_seg_fids) == 2
        assert routing.seg_map.eq(1).all()

    def test_full_conv_segments_reject_large_trailing_coverage_gap(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, *range(10, 45)]
        total_chars = _source_aligned_char_offsets(tokenizer, ids)[-1]
        seg_features = (
            [t.tensor([7])],
            [t.tensor([1])],
            [0],
            [total_chars - 21],
        )

        with pytest.raises(ValueError, match="Forward tokenizer-mismatch"):
            build_full_sequence_segment_routing(
                ids=ids,
                tokenizer=tokenizer,
                seg_features=seg_features,
            )

    def test_full_conv_segments_filter_overhanging_segments_in_lockstep(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, 10, 11, 12]
        seg_features = (
            [t.tensor([11]), t.tensor([22]), t.tensor([33])],
            [t.tensor([1]), t.tensor([2]), t.tensor([3])],
            [0, 2, 4],
            [2, 4, 6],
        )

        routing = build_full_sequence_segment_routing(
            ids=ids,
            tokenizer=tokenizer,
            seg_features=seg_features,
        )

        assert [x.tolist() for x in routing.per_seg_fids] == [[], [11], [22]]
        assert [x.tolist() for x in routing.per_seg_buckets] == [[], [1], [2]]
        assert int(routing.seg_map.max().item()) == 2

    def test_full_conv_segments_support_zero_width_tokens(self):
        class ZeroWidthTokenizer:
            def decode(self, ids):
                return "".join("" if i == 99 else "x" for i in ids)

        routing = build_full_sequence_segment_routing(
            ids=[10, 99, 11],
            tokenizer=ZeroWidthTokenizer(),
            seg_features=([t.tensor([7])], [t.tensor([1])], [0], [2]),
        )

        assert routing.seg_map.tolist() == [1, 1, 1]

    def test_source_aligned_offsets_reject_negative_source_bos_len(self):
        tokenizer = SFTMockTokenizer()
        with pytest.raises(ValueError, match="non-negative"):
            _source_aligned_char_offsets(
                tokenizer,
                [0, 10],
                source_bos_char_len=-1,
            )

    def test_full_conv_segments_reject_gap(self):
        tokenizer = SFTMockTokenizer()
        seg_features = (
            [t.tensor([1]), t.tensor([2])],
            [t.tensor([1]), t.tensor([1])],
            [0, 4],
            [3, 5],
        )
        with pytest.raises(ValueError, match="Gap"):
            build_full_sequence_segment_routing(
                ids=[0, 10, 11, 12, 13],
                tokenizer=tokenizer,
                seg_features=seg_features,
            )

    def test_full_conv_segments_reject_overlap(self):
        tokenizer = SFTMockTokenizer()
        seg_features = (
            [t.tensor([1]), t.tensor([2])],
            [t.tensor([1]), t.tensor([1])],
            [0, 2],
            [3, 5],
        )
        with pytest.raises(ValueError, match="Overlap"):
            build_full_sequence_segment_routing(
                ids=[0, 10, 11, 12, 13],
                tokenizer=tokenizer,
                seg_features=seg_features,
            )

    def test_coverage_gap_uses_full_source_text(self):
        tokenizer = SFTMockTokenizer()
        ids = [0, 10, 11, 12, 13]
        total_chars = _source_aligned_char_offsets(tokenizer, ids)[-1]
        segs = [_make_seg(0, total_chars - 3)]

        assert _compute_full_coverage_gap(ids, segs, tokenizer) == 3
