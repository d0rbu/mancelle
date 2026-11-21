"""Regression tests for ``ignore_bos_features`` in feature extraction."""

from __future__ import annotations

import pytest
import torch as t

from steering.extract_features import extract_document_features
from steering.feature_paths import PoolingMode
from steering.segmentation import SegmentMode
from tests.steering.test_context_independent_extraction import _tiny_model_sae

_TEXT = "Hello World"
_LAYER = 8
_CONTEXT_WINDOW = 64
_SEGMENT_SIZE = 4


class TestFullSequenceIgnoreBosAffectsFirstSegment:
    """Full-sequence extraction starts at token 0, so BOS can affect segment 0."""

    @pytest.mark.parametrize(
        "pooling",
        [PoolingMode.mean_pool, PoolingMode.max_pool],
        ids=["mean_pool", "max_pool"],
    )
    def test_ignore_bos_changes_first_segment(self, pooling: PoolingMode) -> None:
        model, sae = _tiny_model_sae()

        _ids_off, mags_off, *_ = extract_document_features(
            model,
            sae,
            _TEXT,
            layer=_LAYER,
            context_window=_CONTEXT_WINDOW,
            pooling=pooling,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            ignore_bos_features=False,
        )
        _ids_on, mags_on, *_ = extract_document_features(
            model,
            sae,
            _TEXT,
            layer=_LAYER,
            context_window=_CONTEXT_WINDOW,
            pooling=pooling,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            ignore_bos_features=True,
        )

        assert len(mags_off) >= 1 and len(mags_on) >= 1
        assert not t.equal(mags_off[0], mags_on[0]), (
            f"segment 0 {pooling.value} is identical with ignore_bos_features "
            "enabled, but full-sequence extraction should drop BOS from the "
            "first pooled segment"
        )

    def test_last_token_first_segment_is_unchanged_when_bos_is_not_last(self) -> None:
        model, sae = _tiny_model_sae()

        _ids_off, mags_off, *_ = extract_document_features(
            model,
            sae,
            _TEXT,
            layer=_LAYER,
            context_window=_CONTEXT_WINDOW,
            pooling=PoolingMode.last_token,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            ignore_bos_features=False,
        )
        _ids_on, mags_on, *_ = extract_document_features(
            model,
            sae,
            _TEXT,
            layer=_LAYER,
            context_window=_CONTEXT_WINDOW,
            pooling=PoolingMode.last_token,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            ignore_bos_features=True,
        )

        assert len(mags_off) >= 1 and len(mags_on) >= 1
        assert t.equal(mags_off[0], mags_on[0]), (
            "BOS is the first token in segment 0, not the last token; "
            "last_token pooling should therefore be unchanged by dropping it"
        )
