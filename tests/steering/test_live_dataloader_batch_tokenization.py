"""Live dataloader row-group tokenization behavior."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from nanochat.dataset import DatasetSplit
from steering.conditional.encoder_decoder import dataloader as dl
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.refined_types import PositiveInt

_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(128)


class _BatchCountingTokenizer:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.scalar_calls = 0

    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text, prepend=None):
        if isinstance(text, list):
            self.batch_calls += 1
            return [self._encode_one(item, prepend=prepend) for item in text]
        self.scalar_calls += 1
        return self._encode_one(text, prepend=prepend)

    def _encode_one(self, text: str, prepend=None) -> list[int]:
        tokens = [len(part) for part in text.split()]
        if prepend is not None:
            tokens.insert(0, int(prepend))
        return tokens


class _FakeFeatureIndex:
    has_precomputed_percentiles = False
    has_precomputed_token_boundaries = False

    def lookup(self, shard_stem: str, doc_idx: int):
        return [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor([doc_idx % 8, (doc_idx + 1) % 8], dtype=t.int64),
                feature_mags=t.tensor([0.1, 0.2], dtype=t.float32),
                char_start=0,
                char_end=100,
            )
        ]


def _pipeline() -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 1),
            precomputed_thresholds=None,
        ),
        boundaries=t.linspace(0.25, 0.75, 2, dtype=t.float64)
        .unsqueeze(0)
        .expand(8, -1)
        .contiguous(),
        negative_bundle=None,
        effective_buckets=3,
    )


def test_document_batches_tokenizes_row_group_once(monkeypatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "shard_00000.parquet"
    texts = ["alpha beta", "gamma", "delta epsilon zeta", "eta theta"]
    pq.write_table(pa.table({"text": texts}), parquet_path, row_group_size=len(texts))
    monkeypatch.setattr(dl, "list_parquet_files", lambda _data_dir=None: [str(parquet_path)])
    tok = _BatchCountingTokenizer()

    batches = dl._document_batches_with_features(
        split=DatasetSplit.VAL,
        resume_state_dict=None,
        tokenizer_batch_size=16,
        tokenizer=tok,
        feature_index=cast(FeatureIndex, _FakeFeatureIndex()),
        pipeline=_pipeline(),
        segment_size=_SEGMENT_SIZE,
        data_dir=str(tmp_path),
        max_shards=None,
        context_len=16,
    )

    doc_list, state = next(batches)

    assert state["pq_idx"] == 0
    assert state["rg_idx"] == 0
    assert state["epoch"] == 1
    assert state["ddp_world_size"] == 1
    assert state["feature_pipeline_rng_state"] == []
    assert tok.batch_calls == 1
    assert tok.scalar_calls == 0
    assert [tokens for tokens, _features in doc_list] == [
        [0, 5, 4],
        [0, 5],
        [0, 5, 7, 4],
        [0, 3, 5],
    ]


def test_document_batches_match_scalar_tokenizer_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "shard_00000.parquet"
    texts = ["alpha beta", "gamma", "delta epsilon zeta", "eta theta"]
    pq.write_table(pa.table({"text": texts}), parquet_path, row_group_size=len(texts))
    monkeypatch.setattr(dl, "list_parquet_files", lambda _data_dir=None: [str(parquet_path)])

    def _first_batch(tokenizer):
        batches = dl._document_batches_with_features(
            split=DatasetSplit.VAL,
            resume_state_dict=None,
            tokenizer_batch_size=16,
            tokenizer=tokenizer,
            feature_index=cast(FeatureIndex, _FakeFeatureIndex()),
            pipeline=_pipeline(),
            segment_size=_SEGMENT_SIZE,
            data_dir=str(tmp_path),
            max_shards=None,
            context_len=16,
        )
        return next(batches)

    batch_doc_list, batch_state = _first_batch(_BatchCountingTokenizer())
    scalar_tokenizer = _BatchCountingTokenizer()
    scalar_token_lists = [scalar_tokenizer.encode(text, prepend=0) for text in texts]

    assert batch_state["pq_idx"] == 0
    assert batch_state["rg_idx"] == 0
    assert batch_state["epoch"] == 1
    assert batch_state["ddp_world_size"] == 1
    assert batch_state["feature_pipeline_rng_state"] == []
    assert scalar_tokenizer.batch_calls == 0
    assert scalar_tokenizer.scalar_calls == len(texts)
    assert len(batch_doc_list) == len(scalar_token_lists)
    for doc_idx, ((batch_tokens, batch_features), scalar_tokens) in enumerate(
        zip(batch_doc_list, scalar_token_lists, strict=True)
    ):
        assert batch_tokens == scalar_tokens
        assert batch_features is not None
        assert len(batch_features) == 1
        batch_seg = batch_features[0]
        assert t.equal(
            batch_seg.feature_ids,
            t.tensor([(doc_idx + 1) % 8], dtype=t.int64),
        )
        assert batch_seg.char_start == 0
        assert batch_seg.char_end == 100
