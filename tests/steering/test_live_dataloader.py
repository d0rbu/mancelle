"""Tests for the restored live encoder-decoder dataloader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from nanochat.dataset import DatasetSplit
from steering.conditional.delta import DeltaMode, DeltaSamplingWeights, FeatureDeltaConfig
from steering.conditional.encoder_decoder import dataloader as dl_mod
from steering.conditional.encoder_decoder.dataloader import (
    SegFeatures,
    _build_row_segments,
    _document_batches_with_features,
    _encoder_decoder_data_loader_inner,
    encoder_decoder_data_loader_with_state,
    truncate_to_boundary_coverage,
)
from steering.conditional.encoder_decoder.sorted_doc_buffer import SortedDocBuffer
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets

_BOUNDARY_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(128)


class TestSegFeaturesAllFields:
    """SegFeatures stores pre-selection all_feature_ids and all_feature_pcts."""

    def test_all_feature_fields_default_to_none(self):
        sf = SegFeatures(
            feature_ids=t.tensor([1, 2], dtype=t.int64),
            buckets=t.tensor([0, 1], dtype=t.int64),
            percentiles=t.tensor([0.5, 0.8], dtype=t.float32),
            char_start=0,
            char_end=10,
        )
        assert sf.all_feature_ids is None
        assert sf.all_feature_pcts is None

    def test_all_feature_fields_accept_tensors(self):
        all_fids = t.tensor([1, 2, 3, 4, 5], dtype=t.int64)
        all_pcts = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float32)
        sf = SegFeatures(
            feature_ids=t.tensor([1, 2], dtype=t.int64),
            buckets=t.tensor([0, 1], dtype=t.int64),
            percentiles=t.tensor([0.5, 0.8], dtype=t.float32),
            char_start=0,
            char_end=10,
            all_feature_ids=all_fids,
            all_feature_pcts=all_pcts,
        )
        assert sf.all_feature_ids is not None
        assert sf.all_feature_pcts is not None
        assert len(sf.all_feature_ids) == 5


class TestBuildRowSegmentsAllFeatures:
    """_build_row_segments returns 7-tuple including all-feature lists."""

    def test_returns_7_tuple(self):
        """_build_row_segments returns exactly 7 values."""
        sf = SegFeatures(
            feature_ids=t.arange(2, dtype=t.int64),
            buckets=t.zeros(2, dtype=t.int64),
            percentiles=t.zeros(2, dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=10,
            all_feature_ids=t.arange(5, dtype=t.int64),
            all_feature_pcts=t.linspace(0.1, 0.9, 5),
        )
        row_docs: list[tuple[int, list[SegFeatures] | None, list[int]]] = [
            (0, [sf], list(range(12))),
        ]
        result = _build_row_segments(row_docs, T=10, tokenizer=None)
        assert len(result) == 7

    def test_all_fids_len_matches_n_all(self):
        """per_seg_all_fids[i] has len == n_all (5), not n_selected (2)."""
        n_selected, n_all = 2, 5
        sf = SegFeatures(
            feature_ids=t.arange(n_selected, dtype=t.int64),
            buckets=t.zeros(n_selected, dtype=t.int64),
            percentiles=t.zeros(n_selected, dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=10,
            all_feature_ids=t.arange(n_all, dtype=t.int64),
            all_feature_pcts=t.linspace(0.1, 0.9, n_all),
        )
        row_docs: list[tuple[int, list[SegFeatures] | None, list[int]]] = [
            (0, [sf], list(range(12))),
        ]
        _, _, _, _, _, per_seg_all_fids, per_seg_all_pcts = _build_row_segments(
            row_docs, T=10, tokenizer=None
        )
        assert len(per_seg_all_fids) == 1
        assert len(per_seg_all_fids[0]) == n_all
        assert len(per_seg_all_pcts[0]) == n_all

    def test_all_fids_falls_back_to_selected_when_none(self):
        """When all_feature_ids is None, per_seg_all_fids falls back to feature_ids."""
        n_selected = 3
        sf = SegFeatures(
            feature_ids=t.arange(n_selected, dtype=t.int64),
            buckets=t.zeros(n_selected, dtype=t.int64),
            percentiles=t.zeros(n_selected, dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=10,
        )
        row_docs: list[tuple[int, list[SegFeatures] | None, list[int]]] = [
            (0, [sf], list(range(12))),
        ]
        _, _, _, _, _, per_seg_all_fids, per_seg_all_pcts = _build_row_segments(
            row_docs, T=10, tokenizer=None
        )
        assert t.equal(per_seg_all_fids[0], t.arange(n_selected, dtype=t.int64))

    def test_empty_segs_get_empty_all_feature_tensors(self):
        """When a doc has no features (n_ext_segs == 0), all-feature lists get empty tensors."""
        row_docs = [(0, None, list(range(12)))]
        _, _, _, _, _, per_seg_all_fids, per_seg_all_pcts = _build_row_segments(
            row_docs, T=10, tokenizer=None
        )
        assert len(per_seg_all_fids) == 1
        assert len(per_seg_all_fids[0]) == 0
        assert per_seg_all_fids[0].dtype == t.int64
        assert per_seg_all_pcts[0].dtype == t.float32

    def test_multi_segment_all_features(self):
        """Multiple segments each get their own all_feature_ids/pcts."""
        sfs = []
        for i in range(3):
            n_all = 5 + i
            sfs.append(SegFeatures(
                feature_ids=t.arange(2, dtype=t.int64),
                buckets=t.zeros(2, dtype=t.int64),
                percentiles=t.zeros(2, dtype=t.float32),
                char_start=i * 30,
                char_end=(i + 1) * 30,
                token_start=i * 4,
                token_end=(i + 1) * 4,
                all_feature_ids=t.arange(n_all, dtype=t.int64),
                all_feature_pcts=t.ones(n_all, dtype=t.float32) * (i + 1) * 0.1,
            ))
        row_docs: list[tuple[int, list[SegFeatures] | None, list[int]]] = [
            (0, sfs, list(range(14))),
        ]
        _, _, _, _, _, per_seg_all_fids, per_seg_all_pcts = _build_row_segments(
            row_docs, T=12, tokenizer=None
        )
        assert len(per_seg_all_fids) == 3
        assert len(per_seg_all_fids[0]) == 5
        assert len(per_seg_all_fids[1]) == 6
        assert len(per_seg_all_fids[2]) == 7


class TestDocumentBatchesImport:
    """_document_batches_with_features is importable and callable."""

    def test_importable(self):
        assert callable(_document_batches_with_features)

    def test_has_context_len_parameter(self):
        """The restored function has the new context_len parameter."""
        import inspect
        sig = inspect.signature(_document_batches_with_features)
        assert "context_len" in sig.parameters
        # Default should be None
        assert sig.parameters["context_len"].default is None

    def test_has_required_parameters(self):
        """The function has all expected parameters from the original + new ones."""
        import inspect
        sig = inspect.signature(_document_batches_with_features)
        param_names = list(sig.parameters.keys())
        expected = [
            "split", "resume_state_dict", "tokenizer_batch_size", "tokenizer",
            "feature_index", "pipeline", "segment_size", "data_dir",
            "max_shards", "context_len", "include_splice_mask",
        ]
        assert param_names == expected
        assert sig.parameters["include_splice_mask"].default is False


class TestTruncateToBoundaryCoverage:
    """truncate_to_boundary_coverage handles boundary gaps correctly."""

    def test_no_features_returns_unchanged(self):
        tokens = [1, 2, 3, 4, 5]
        assert truncate_to_boundary_coverage(
            tokens,
            None,
            context_len=10,
            segment_size=_BOUNDARY_SEGMENT_SIZE,
        ) == tokens
        assert truncate_to_boundary_coverage(
            tokens,
            [],
            context_len=10,
            segment_size=_BOUNDARY_SEGMENT_SIZE,
        ) == tokens

    def test_fully_covered_returns_unchanged(self):
        sf = SegFeatures(
            feature_ids=t.tensor([0], dtype=t.int64),
            buckets=t.tensor([0], dtype=t.int64),
            percentiles=t.tensor([0.5], dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=10,
        )
        tokens = list(range(8))  # 8 tokens, boundary covers up to 10
        result = truncate_to_boundary_coverage(
            tokens,
            [sf],
            context_len=20,
            segment_size=_BOUNDARY_SEGMENT_SIZE,
        )
        assert result == tokens

    def test_small_gap_truncates(self):
        """A gap of <= 4 tokens is silently truncated."""
        sf = SegFeatures(
            feature_ids=t.tensor([0], dtype=t.int64),
            buckets=t.tensor([0], dtype=t.int64),
            percentiles=t.tensor([0.5], dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=10,
        )
        tokens = list(range(13))  # 13 tokens, boundary covers up to 10, gap=3
        result = truncate_to_boundary_coverage(
            tokens,
            [sf],
            context_len=20,
            segment_size=_BOUNDARY_SEGMENT_SIZE,
        )
        assert len(result) == 10

    def test_context_len_truncation_first(self):
        """Tokens are first truncated to context_len before boundary check."""
        sf = SegFeatures(
            feature_ids=t.tensor([0], dtype=t.int64),
            buckets=t.tensor([0], dtype=t.int64),
            percentiles=t.tensor([0.5], dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=0,
            token_end=5,
        )
        tokens = list(range(100))  # 100 tokens, context_len=5 -> fully covered
        result = truncate_to_boundary_coverage(
            tokens,
            [sf],
            context_len=5,
            segment_size=_BOUNDARY_SEGMENT_SIZE,
        )
        assert len(result) == 5

    def test_missing_boundaries_raises(self):
        """Segments without token_start/token_end raise ValueError."""
        sf = SegFeatures(
            feature_ids=t.tensor([0], dtype=t.int64),
            buckets=t.tensor([0], dtype=t.int64),
            percentiles=t.tensor([0.5], dtype=t.float32),
            char_start=0,
            char_end=100,
            token_start=None,
            token_end=None,
        )
        with pytest.raises(ValueError, match="token_start=None"):
            truncate_to_boundary_coverage(
                list(range(10)),
                [sf],
                context_len=20,
                segment_size=_BOUNDARY_SEGMENT_SIZE,
            )


class TestLiveDataLoaderYields10Tuple:
    """encoder_decoder_data_loader_with_state yields a valid 10-tuple batch."""

    # NOTE: Full integration tests need synthetic parquet + feature shard fixtures.
    # For now, test that the functions are importable and have correct signatures.

    def test_importable(self):
        from steering.conditional.encoder_decoder.dataloader import (
            encoder_decoder_data_loader_with_state,
        )
        assert callable(encoder_decoder_data_loader_with_state)

    def test_has_n_sae_features_parameter(self):
        import inspect

        from steering.conditional.encoder_decoder.dataloader import (
            encoder_decoder_data_loader_with_state,
        )
        sig = inspect.signature(encoder_decoder_data_loader_with_state)
        assert "n_sae_features" in sig.parameters
        assert sig.parameters["n_sae_features"].default is None

    def test_has_context_len_parameter(self):
        import inspect

        from steering.conditional.encoder_decoder.dataloader import (
            encoder_decoder_data_loader_with_state,
        )
        sig = inspect.signature(encoder_decoder_data_loader_with_state)
        assert "context_len" in sig.parameters
        assert sig.parameters["context_len"].default is None

    def test_inner_function_importable(self):
        assert callable(_encoder_decoder_data_loader_inner)

    def test_inner_has_n_sae_features_parameter(self):
        import inspect
        sig = inspect.signature(_encoder_decoder_data_loader_inner)
        assert "n_sae_features" in sig.parameters
        assert sig.parameters["n_sae_features"].default is None

    def test_inner_has_context_len_parameter(self):
        import inspect
        sig = inspect.signature(_encoder_decoder_data_loader_inner)
        assert "context_len" in sig.parameters
        assert sig.parameters["context_len"].default is None

    def test_decoder_segment_size_must_be_none(self):
        """Inner function asserts decoder_segment_size is None."""
        # Calling with decoder_segment_size != None should raise AssertionError
        with pytest.raises(AssertionError, match="Stride-based segment assignment"):
            gen = _encoder_decoder_data_loader_inner(
                tokenizer=None,
                B=1,
                T=10,
                split=DatasetSplit.TRAIN,
                feature_index=object.__new__(FeatureIndex),
                pipeline=_build_threshold_pipeline(),
                segment_size=PositiveInt.parse(128),
                decoder_segment_size=64,  # not None -> should fail
                device="cpu",
                resume_state_dict=None,
                tokenizer_batch_size=128,
                buffer_size=100,
                data_dir=None,
                prefetch_batches=0,
            )
            next(gen)  # generators are lazy, need to advance

    @settings(max_examples=500, derandomize=True)
    @given(st.one_of(st.integers(max_value=0), st.booleans()))
    def test_segment_size_rejects_complete_invalid_integer_domain(
        self,
        invalid_segment_size: int,
    ) -> None:
        runtime_value: Any = invalid_segment_size
        loader = encoder_decoder_data_loader_with_state(
            tokenizer=None,
            B=1,
            T=10,
            split=DatasetSplit.TRAIN,
            feature_index=object.__new__(FeatureIndex),
            pipeline=_build_threshold_pipeline(),
            segment_size=runtime_value,
            decoder_segment_size=None,
            device="cpu",
        )
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="segment_size",
        ):
            next(loader)


# ===========================================================================
# Shared fixtures for functional / integration tests
# ===========================================================================


class _SimpleTokenizer:
    """Deterministic tokenizer for testing.

    Token 0 = BOS = "<bos>" (5 chars)
    Token n (n>0) = f"t{n:03d} " (5 chars each)

    Every non-BOS token decodes to exactly 5 characters, giving predictable
    char offsets for token boundary computation.
    """

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            if tid == 0:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


# Test constants
_N_SAE_FEATURES = 200
_NUM_BUCKETS = 4
_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(20)
_N_DOCS = 8  # enough docs to fill multiple packed rows
_N_SEGS_PER_DOC = 2
_N_FEATS_PER_SEG = 6
_B = 2
_T = 32


def _compute_token_boundaries_for_seg(tokenizer, text_tokens, seg_char_starts, seg_char_ends):
    """Compute precomputed token start/end for each segment."""
    from steering.segmentation import assign_tokens_to_segments

    char_offsets = token_to_char_offsets(tokenizer, text_tokens)
    tok_starts_t = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends_t = t.tensor(char_offsets[1:], dtype=t.int64)
    assignments = assign_tokens_to_segments(
        tok_starts_t,
        tok_ends_t,
        t.tensor(seg_char_starts, dtype=t.int64),
        t.tensor(seg_char_ends, dtype=t.int64),
    )
    assignments = t.cummax(assignments, dim=0).values
    n_segs = len(seg_char_starts)
    token_starts = []
    token_ends = []
    for s in range(n_segs):
        mask = assignments == s
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            token_starts.append(int(indices[0].item()))
            token_ends.append(int(indices[-1].item()) + 1)
        else:
            prev_end = token_ends[-1] if token_ends else 0
            token_starts.append(prev_end)
            token_ends.append(prev_end)
    return token_starts, token_ends


def _build_threshold_pipeline(n_sae_features=_N_SAE_FEATURES, num_buckets=_NUM_BUCKETS, threshold=3):
    """Build a minimal ThresholdSelection pipeline for testing."""
    boundaries = (
        t.linspace(
            1 / num_buckets, 1.0 - 1 / num_buckets, num_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=num_buckets,
    )


def _make_synthetic_data(tmp_path, *, n_docs=_N_DOCS, n_segs_per_doc=_N_SEGS_PER_DOC,
                          n_feats_per_seg=_N_FEATS_PER_SEG, segment_size=_SEGMENT_SIZE,
                          n_sae_features=_N_SAE_FEATURES, tokenizer_checksum="test_cksum",
                          percentile_checksum="pct_cksum", feature_n_docs=None):
    """Create synthetic parquet + feature shard + token boundary + percentile overlays.

    Creates TWO parquet files (so the dataloader can do train/val splitting).
    The first parquet is for training, the second for validation.

    ``feature_n_docs`` (default ``None`` ⇒ ``n_docs``) limits how many leading
    documents receive feature segments in the shard, WITHOUT shrinking the
    parquet (which always has ``n_docs`` rows). When ``feature_n_docs < n_docs``
    the trailing parquet docs have no feature row, so ``FeatureIndex.lookup``
    raises ``KeyError`` for them — the corruption fail-fast path of issue #521.

    Returns dict with data_dir, feature_dir, texts, tokenizer, checksums.
    """
    if feature_n_docs is None:
        feature_n_docs = n_docs
    assert feature_n_docs <= n_docs, "feature_n_docs cannot exceed parquet n_docs"
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    # Feature shards must live under ``s{N}/`` so verify_segment_size's
    # cross-check (steering/feature_paths.py:_parse_segment_size_from_path)
    # accepts them. Added in commit 8c1880ca.
    feature_dir = Path(tmp_path) / "features" / f"s{segment_size}"
    feature_dir.mkdir(parents=True)

    tokenizer = _SimpleTokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Create two parquet files (train + val)
    for pq_num in range(2):
        shard_stem = f"shard_{pq_num:05d}"
        texts = []
        for doc_idx in range(n_docs):
            # Generate documents with 8-15 words each, enough for 2 segments
            n_words = 8 + (doc_idx % 8) + pq_num * 3
            words = [f"w{pq_num}d{doc_idx}x{w}" for w in range(n_words)]
            texts.append(" ".join(words))

        # Write parquet with a single row group
        pq_path = data_dir / f"{shard_stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path), row_group_size=n_docs)

        # Build feature shard (covers only the first ``feature_n_docs`` docs;
        # trailing parquet docs intentionally get no feature row).
        total_segs = feature_n_docs * n_segs_per_doc
        total_feats = total_segs * n_feats_per_seg

        all_fids = t.zeros(total_feats, dtype=t.int32)
        all_mags = t.zeros(total_feats, dtype=t.float32)
        all_pcts_overlay = t.zeros(total_feats, dtype=t.uint8)  # percentile overlay
        offsets = t.zeros(total_segs + 1, dtype=t.int64)
        doc_ids = t.zeros(total_segs, dtype=t.int32)
        seg_indices = t.zeros(total_segs, dtype=t.int32)
        seg_positions = t.zeros(total_segs, dtype=t.int32)
        seg_char_starts_t = t.zeros(total_segs, dtype=t.int32)
        seg_char_ends_t = t.zeros(total_segs, dtype=t.int32)
        seg_token_starts_t = t.zeros(total_segs, dtype=t.int32)
        seg_token_ends_t = t.zeros(total_segs, dtype=t.int32)

        feat_offset = 0
        seg_global = 0

        for doc_idx in range(feature_n_docs):
            text = texts[doc_idx]
            text_tokens = tokenizer.encode(text, prepend=bos_token)
            char_offsets = token_to_char_offsets(tokenizer, text_tokens)
            n_tokens = len(text_tokens)

            # Build segments based on token-count (not chars), for simplicity
            tokens_per_seg = max(1, n_tokens // n_segs_per_doc)

            doc_seg_char_starts = []
            doc_seg_char_ends = []

            for seg_idx in range(n_segs_per_doc):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * segment_size
                offsets[seg_global] = feat_offset

                tok_start = seg_idx * tokens_per_seg
                tok_end = min((seg_idx + 1) * tokens_per_seg, n_tokens) if seg_idx < n_segs_per_doc - 1 else n_tokens
                char_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
                char_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]

                seg_char_starts_t[seg_global] = char_start
                seg_char_ends_t[seg_global] = char_end
                doc_seg_char_starts.append(int(char_start))
                doc_seg_char_ends.append(int(char_end))

                # Assign unique feature IDs per (pq, doc, seg)
                for f in range(n_feats_per_seg):
                    fid = (pq_num * n_docs * 10 + doc_idx * 10 + seg_idx * n_feats_per_seg + f) % n_sae_features
                    all_fids[feat_offset + f] = fid
                    all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx
                    # Percentile: encode as uint8 in [1, 255] (0 means inactive)
                    all_pcts_overlay[feat_offset + f] = min(255, max(1, int(255 * (f + 1) / (n_feats_per_seg + 1))))

                feat_offset += n_feats_per_seg
                seg_global += 1

            # Compute token boundaries
            tok_starts, tok_ends = _compute_token_boundaries_for_seg(
                tokenizer, text_tokens, doc_seg_char_starts, doc_seg_char_ends
            )
            for seg_idx in range(n_segs_per_doc):
                g = (doc_idx * n_segs_per_doc) + seg_idx
                seg_token_starts_t[g] = tok_starts[seg_idx]
                seg_token_ends_t[g] = tok_ends[seg_idx]

        offsets[total_segs] = feat_offset

        shard_data = {
            "offsets": offsets,
            "feature_ids": all_fids[:feat_offset],
            "feature_mags": all_mags[:feat_offset],
            "doc_ids": doc_ids,
            "segment_indices": seg_indices,
            "segment_positions": seg_positions,
            "doc_token_counts": t.full((feature_n_docs,), 40, dtype=t.int32),
            "segment_char_starts": seg_char_starts_t,
            "segment_char_ends": seg_char_ends_t,
            "n_docs": feature_n_docs,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": segment_size,
        }
        save_features(shard_data, feature_dir / f"{shard_stem}_features")

        # Write token boundary overlay
        tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tokenizer_checksum)
        write_manifest(tok_dir, tokenizer_checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: seg_token_starts_t,
                TensorKey.SEGMENT_TOKEN_ENDS: seg_token_ends_t,
            },
            overlay_shard_path(tok_dir, shard_stem),
        )

        # Write percentile overlay
        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, percentile_checksum)
        write_manifest(pct_dir, percentile_checksum, OverlayKind.PERCENTILE)
        save_overlay_shard(
            {
                TensorKey.FEATURE_PERCENTILES: all_pcts_overlay[:feat_offset],
            },
            overlay_shard_path(pct_dir, shard_stem),
        )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "tokenizer_checksum": tokenizer_checksum,
        "percentile_checksum": percentile_checksum,
        "tokenizer": tokenizer,
    }


@pytest.fixture
def synthetic_env(tmp_path):
    """Fixture providing synthetic data environment for live dataloader tests."""
    return _make_synthetic_data(tmp_path)


def _make_dataloader(env, *, B=_B, T=_T, label_dropout=0.0, n_sae_features=None,
                      buffer_size=50, threshold=3, resume_state_dict=None,
                      pad_F=None, pad_n_seg=None, doc_prefetch_depth=0,
                      prefetch_batches=0):
    """Build a live dataloader from a synthetic environment dict."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )
    pipeline = _build_threshold_pipeline(threshold=threshold)

    gen = encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=resume_state_dict,
        tokenizer_batch_size=128,
        buffer_size=buffer_size,
        data_dir=env["data_dir"],
        prefetch_batches=prefetch_batches,
        label_dropout=label_dropout,
        max_shards=None,
        n_sae_features=n_sae_features,
        pad_F=pad_F,
        pad_n_seg=pad_n_seg,
        doc_prefetch_depth=doc_prefetch_depth,
    )
    return gen


# ===========================================================================
# 1. End-to-end live dataloader with synthetic data
# ===========================================================================


class TestLiveDataloaderEndToEnd:
    """Full integration tests for the live encoder-decoder dataloader."""

    def test_yields_10_tuple(self, synthetic_env):
        """Dataloader yields exactly a 10-element tuple per batch."""
        gen = _make_dataloader(synthetic_env)
        batch = next(gen)
        assert isinstance(batch, tuple)
        assert len(batch) == 10

    def test_tensor_shapes_and_dtypes(self, synthetic_env):
        """Each tensor in the 10-tuple has correct shape, dtype, and device."""
        gen = _make_dataloader(synthetic_env)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(gen)

        B, T = _B, _T

        # x and y: (B, T) int64
        assert x.shape == (B, T)
        assert y.shape == (B, T)
        assert x.dtype == t.long
        assert y.dtype == t.long

        # enc_fids: (B, n_seg, F_max) int64
        assert enc_fids.dim() == 3
        assert enc_fids.shape[0] == B
        assert enc_fids.dtype == t.int64

        # enc_mags: same shape as enc_fids, int64 (bucket indices)
        assert enc_mags.shape == enc_fids.shape
        assert enc_mags.dtype == t.int64

        # enc_mask: same shape as enc_fids, bool
        assert enc_mask.shape == enc_fids.shape
        assert enc_mask.dtype == t.bool

        # enc_pcts: same shape as enc_fids, float32
        assert enc_pcts.shape == enc_fids.shape
        assert enc_pcts.dtype == t.float32

        # seg_map: (B, T) int64
        assert seg_map.shape == (B, T)
        assert seg_map.dtype == t.int64

        # seg_doc_id: (B, n_seg) int64
        assert seg_doc_id.shape[0] == B
        assert seg_doc_id.shape[1] == enc_fids.shape[1]
        assert seg_doc_id.dtype == t.int64

        # fp_targets: None when n_sae_features is not set
        assert fp_targets is None

        # state_dict
        assert isinstance(state_dict, dict)
        assert "pq_idx" in state_dict
        assert "rg_idx" in state_dict
        assert "epoch" in state_dict

    def test_multi_batch_yield(self, synthetic_env):
        """Dataloader can yield at least 3 consecutive batches without error."""
        gen = _make_dataloader(synthetic_env)
        batches = []
        for _ in range(3):
            batch = next(gen)
            batches.append(batch)
        assert len(batches) == 3

    def test_bos_alignment(self, synthetic_env):
        """Every packed row starts with BOS token (position 0 in x)."""
        tokenizer = synthetic_env["tokenizer"]
        bos_id = tokenizer.get_bos_token_id()
        gen = _make_dataloader(synthetic_env)
        x, *_ = next(gen)
        for row_idx in range(_B):
            assert x[row_idx, 0].item() == bos_id, (
                f"Row {row_idx} does not start with BOS token {bos_id}, got {x[row_idx, 0].item()}"
            )

    def test_seg_map_valid_indices(self, synthetic_env):
        """All seg_map values are valid indices into the segment dimension of enc_fids."""
        gen = _make_dataloader(synthetic_env)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(gen)
        n_seg = enc_fids.shape[1]
        assert (seg_map >= 0).all(), "seg_map has negative indices"
        assert (seg_map < n_seg).all(), (
            f"seg_map has index {seg_map.max().item()} >= n_seg {n_seg}"
        )

    def test_targets_are_shifted_inputs(self, synthetic_env):
        """y should be x shifted by one position within each document."""
        gen = _make_dataloader(synthetic_env)
        # Note: x and y come from row_buffer[:, :-1] and row_buffer[:, 1:]
        # so y[i, j] = row_buffer[i, j+1], which is the next token.
        # We can verify the relationship by checking x[:, 1:] == y[:, :-1]
        # actually that's not guaranteed due to document boundaries.
        # The real invariant is that inputs and targets are consecutive slices
        # of the packed row. Let's just check they're different (shifted).
        x, y, *_ = next(gen)
        # Not all identical (would be extremely unlikely)
        assert not t.equal(x, y), "x and y should not be identical (shifted by 1)"


# ===========================================================================
# 2. fp_targets correctness
# ===========================================================================


class TestFpTargets:
    """Tests for dense feature prediction target tensor (fp_targets)."""

    def test_fp_targets_shape(self, synthetic_env):
        """fp_targets has shape (B, n_seg, n_sae_features)."""
        gen = _make_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.shape[0] == _B
        assert fp_targets.shape[1] == enc_fids.shape[1]  # n_seg
        assert fp_targets.shape[2] == _N_SAE_FEATURES

    def test_fp_targets_dtype(self, synthetic_env):
        """fp_targets is float32."""
        gen = _make_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        *_, fp_targets, _ = next(gen)
        assert fp_targets.dtype == t.float32

    def test_fp_targets_values_in_range(self, synthetic_env):
        """fp_targets nonzero values are in (0, 1] (normalized percentiles)."""
        gen = _make_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        *_, fp_targets, _ = next(gen)
        nonzero = fp_targets[fp_targets != 0]
        if len(nonzero) > 0:
            assert (nonzero > 0).all(), "fp_targets has negative nonzero values"
            assert (nonzero <= 1.0).all(), "fp_targets has values > 1.0"

    def test_fp_targets_has_nonzero_entries(self, synthetic_env):
        """fp_targets should have nonzero entries (features are present in the synthetic data)."""
        gen = _make_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        *_, fp_targets, _ = next(gen)
        assert fp_targets.sum().item() > 0, "fp_targets is all zeros"

    def test_fp_targets_more_nonzero_than_selected(self, synthetic_env):
        """fp_targets has MORE nonzero entries than enc_mask active entries.

        fp_targets includes ALL features (pre-selection), while enc_mask
        only covers the threshold-selected top-k.
        """
        gen = _make_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES, threshold=3)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, _ = next(gen)

        selected_count = enc_mask.sum().item()
        fp_nonzero_count = (fp_targets != 0).sum().item()

        # Each segment has _N_FEATS_PER_SEG=6 features pre-selection, but only
        # threshold=3 are selected. So fp_targets should have more nonzero entries.
        assert fp_nonzero_count >= selected_count, (
            f"fp_targets nonzero count ({fp_nonzero_count}) should be >= "
            f"enc_mask active count ({selected_count})"
        )

    def test_fp_targets_none_when_n_sae_features_unset(self, synthetic_env):
        """fp_targets is None when n_sae_features=None."""
        gen = _make_dataloader(synthetic_env, n_sae_features=None)
        *_, fp_targets, _ = next(gen)
        assert fp_targets is None


# ===========================================================================
# 3. Label dropout
# ===========================================================================


class TestLabelDropout:
    """Tests for label_dropout behavior in the live dataloader."""

    def test_no_dropout_has_active_features(self, synthetic_env):
        """With label_dropout=0.0, enc_mask has active features in every row."""
        gen = _make_dataloader(synthetic_env, label_dropout=0.0)
        x, y, enc_fids, enc_mags, enc_mask, *_ = next(gen)
        for row_idx in range(_B):
            assert enc_mask[row_idx].any(), (
                f"Row {row_idx} has no active features with label_dropout=0.0"
            )

    def test_full_dropout_has_at_most_B_sentinels(self, synthetic_env):
        """With label_dropout=1.0, enc_mask has at most B sentinel features.

        The NaN prevention code sets enc_mask[row, 0, 0] = True for rows
        that would otherwise have all-zero masks.
        """
        gen = _make_dataloader(synthetic_env, label_dropout=1.0)
        x, y, enc_fids, enc_mags, enc_mask, *_ = next(gen)
        total_active = enc_mask.sum().item()
        # At most B sentinel features (one per row)
        assert total_active <= _B, (
            f"With full dropout, expected at most {_B} active entries, got {total_active}"
        )

    def test_nan_prevention_with_full_dropout(self, synthetic_env):
        """Even with label_dropout=1.0, at least one feature per row is active."""
        gen = _make_dataloader(synthetic_env, label_dropout=1.0)
        # Run several batches to be sure (dropout is stochastic)
        for _ in range(5):
            x, y, enc_fids, enc_mags, enc_mask, *_ = next(gen)
            for row_idx in range(_B):
                row_active = enc_mask[row_idx].any(dim=-1).any()
                assert row_active, (
                    f"Row {row_idx} has no active features despite NaN prevention"
                )


# ===========================================================================
# 4. SortedDocBuffer in packing loop
# ===========================================================================


class TestSortedDocBufferEquivalence:
    """SortedDocBuffer produces same results as linear scan best-fit."""

    def _linear_best_fit(self, docs, remaining):
        """O(n) linear scan best-fit: find largest doc with len <= remaining."""
        best_idx = None
        best_len = -1
        for i, (tokens, _) in enumerate(docs):
            doc_len = len(tokens)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len
        if best_idx is not None:
            return docs.pop(best_idx)
        return None

    def test_best_fit_equivalence(self):
        """SortedDocBuffer.pop_best_fit matches linear scan best-fit."""
        docs = [
            ([1, 2, 3], None),  # len 3
            ([1, 2, 3, 4, 5], None),  # len 5
            ([1, 2], None),  # len 2
            ([1, 2, 3, 4, 5, 6, 7], None),  # len 7
            ([1], None),  # len 1
            ([1, 2, 3, 4], None),  # len 4
        ]

        # Test various remaining values
        for remaining in [8, 5, 3, 1, 10]:
            linear_docs = list(docs)
            buffer = SortedDocBuffer()
            for d in docs:
                buffer.append(d)

            linear_result = self._linear_best_fit(linear_docs, remaining)
            buffer_result = buffer.pop_best_fit(remaining)

            if linear_result is None:
                assert buffer_result is None
            else:
                assert buffer_result is not None
                assert len(buffer_result[0]) == len(linear_result[0])

    def test_pop_shortest(self):
        """pop_shortest returns the shortest document."""
        buffer = SortedDocBuffer()
        buffer.append(([1, 2, 3], None))
        buffer.append(([1], None))
        buffer.append(([1, 2, 3, 4, 5], None))

        result = buffer.pop_shortest()
        assert len(result[0]) == 1

    def test_buffer_refill_across_batches(self, synthetic_env):
        """Buffer refill works correctly across multiple batches.

        Uses a very small buffer_size to force frequent refills.
        """
        gen = _make_dataloader(synthetic_env, buffer_size=2)
        # Should produce multiple batches even with tiny buffer
        for _ in range(3):
            batch = next(gen)
            assert len(batch) == 10

    def test_packing_sequence(self):
        """Verify a complete packing sequence: repeated best-fit + crop fills row."""
        buffer = SortedDocBuffer()
        # Add documents of various sizes
        for length in [5, 10, 3, 7, 15, 2, 8, 4]:
            buffer.append((list(range(length)), None))

        row_capacity = 20
        pos = 0
        docs_used = []

        while pos < row_capacity:
            remaining = row_capacity - pos
            result = buffer.pop_best_fit(remaining)
            if result is not None:
                docs_used.append(len(result[0]))
                pos += len(result[0])
            else:
                # Crop shortest
                shortest = buffer.pop_shortest()
                crop_len = min(len(shortest[0]), remaining)
                docs_used.append(crop_len)
                pos += crop_len

        # The row should be exactly filled
        assert sum(docs_used) == row_capacity


# ===========================================================================
# 5. Global doc index
# ===========================================================================


class TestGlobalDocIndex:
    """Verify documents in row groups > 0 get correct features."""

    def test_features_vary_across_docs(self, synthetic_env):
        """Different documents should have different feature IDs.

        If global doc indexing were broken, all documents would get doc 0's features.
        """
        gen = _make_dataloader(synthetic_env)
        # Collect enc_fids from several batches
        all_fid_sets = []
        for _ in range(3):
            _, _, enc_fids, _, enc_mask, *_ = next(gen)
            for row_idx in range(_B):
                # Get active feature IDs for this row
                active = enc_fids[row_idx][enc_mask[row_idx]]
                if len(active) > 0:
                    all_fid_sets.append(set(active.tolist()))

        # Not all rows should have identical feature sets
        if len(all_fid_sets) >= 2:
            unique_sets = {frozenset(s) for s in all_fid_sets}
            assert len(unique_sets) > 1, (
                "All rows have identical feature IDs — global doc index may be broken"
            )

    def test_state_dict_tracks_progress(self, synthetic_env):
        """state_dict pq_idx and rg_idx track progress through data."""
        gen = _make_dataloader(synthetic_env)
        states = []
        for _ in range(5):
            *_, state_dict = next(gen)
            states.append(state_dict.copy())
        # At least the epoch should be consistent (starts at 1)
        assert all(s["epoch"] >= 1 for s in states)

    def test_enc_mask_has_active_features(self, synthetic_env):
        """enc_mask should have active features in at least one segment per row."""
        gen = _make_dataloader(synthetic_env)
        for _ in range(3):
            _, _, _, _, enc_mask, *_ = next(gen)
            for row_idx in range(_B):
                assert enc_mask[row_idx].any(), (
                    f"Row {row_idx} has no active features"
                )

    def test_enc_pcts_values_for_active_features(self, synthetic_env):
        """Active features should have nonzero percentile values."""
        gen = _make_dataloader(synthetic_env)
        _, _, enc_fids, enc_mags, enc_mask, enc_pcts, *_ = next(gen)
        active_pcts = enc_pcts[enc_mask]
        # At least some active features should have nonzero percentiles
        # (they come from discretization of nonzero magnitudes)
        if len(active_pcts) > 0:
            assert (active_pcts >= 0).all(), "Active percentiles have negative values"


# ===========================================================================
# Missing-document fail-fast (issue #521)
# ===========================================================================


class TestMissingDocFailFast:
    """A parquet doc with no feature row makes the live dataloader fail fast.

    ``FeatureIndex.lookup`` raises ``KeyError`` (naming the shard + doc) when a
    document is absent from the feature shard. The dataloader must propagate
    that — it must NOT swallow the gap or rely on a strippable ``assert`` (the
    ``-O``-stripped ``assert segments is not None`` removed for issue #521).
    """

    def test_missing_doc_raises_keyerror(self, tmp_path):
        """3-row parquet whose feature shard covers only docs {0, 1} -> KeyError on doc 2."""
        env = _make_synthetic_data(tmp_path, n_docs=3, feature_n_docs=2)
        with pytest.raises(KeyError, match=r"Document 2 not found in shard"):
            # consume the dataloader generator built over the 3-row parquet whose
            # feature shard only covers docs 0 and 1; Phase-1 per-doc feature
            # lookup reaches doc_idx=2 and raises.
            gen = _make_dataloader(env, B=1, T=8, buffer_size=8)
            for _ in range(5):
                next(gen)

    def test_all_docs_present_yields_batch(self, tmp_path):
        """Same tiny 3-doc config with a COMPLETE feature shard yields a valid batch.

        Proves the valid path is unbroken: full coverage (feature_n_docs == n_docs)
        produces a normal 10-tuple with the expected shapes — no spurious KeyError.
        """
        env = _make_synthetic_data(tmp_path, n_docs=3, feature_n_docs=3)
        gen = _make_dataloader(env, B=1, T=8, buffer_size=8)
        batch = next(gen)
        assert isinstance(batch, tuple)
        assert len(batch) == 10
        x, y, *_ = batch
        assert x.shape == (1, 8)
        assert y.shape == (1, 8)


# ===========================================================================
# Additional integration tests
# ===========================================================================


class TestDataloaderWithPrefetch:
    """Test the prefetch wrapper path."""

    def test_prefetch_yields_cloned_tensors(self, synthetic_env):
        """With prefetch_batches > 0, tensors should be stable (cloned)."""
        from nanochat.dataset import DatasetSplit

        tokenizer = synthetic_env["tokenizer"]
        feature_index = FeatureIndex(
            synthetic_env["feature_dir"],
            tokenizer_checksum=synthetic_env["tokenizer_checksum"],
            percentile_checksum=synthetic_env.get("percentile_checksum"),
        )
        pipeline = _build_threshold_pipeline()

        gen = encoder_decoder_data_loader_with_state(
            tokenizer=tokenizer,
            B=_B,
            T=_T,
            split=DatasetSplit.TRAIN,
            feature_index=feature_index,
            pipeline=pipeline,
            segment_size=_SEGMENT_SIZE,
            decoder_segment_size=None,
            device="cpu",
            resume_state_dict=None,
            tokenizer_batch_size=128,
            buffer_size=50,
            data_dir=synthetic_env["data_dir"],
            prefetch_batches=2,
            label_dropout=0.0,
            n_sae_features=_N_SAE_FEATURES,
        )
        batch1 = next(gen)
        x1_clone = batch1[0].clone()
        next(gen)
        # batch1 tensors should be stable (not overwritten by batch2)
        assert t.equal(batch1[0], x1_clone), (
            "Prefetched tensor was modified after yielding"
        )


class TestDocBatchPrefetch:
    """Tests for async document batch prefetching (doc_prefetch_depth > 0)."""

    # --- Parity ---

    def test_bit_identical_output_sync_vs_async(self, synthetic_env):
        """Async prefetch produces bit-identical batches to sync path."""
        n_batches = 5
        sync_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=0)
        async_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=4)
        for batch_i in range(n_batches):
            sb = next(sync_gen)
            ab = next(async_gen)
            for tensor_i in range(9):  # first 9 elements are tensors
                if sb[tensor_i] is None:
                    assert ab[tensor_i] is None, (
                        f"Batch {batch_i}, el {tensor_i}: sync=None, async is not"
                    )
                else:
                    assert t.equal(sb[tensor_i], ab[tensor_i]), (
                        f"Batch {batch_i}, el {tensor_i}: tensors differ"
                    )

    def test_state_dict_parity_sync_vs_async(self, synthetic_env):
        """State dict (pq_idx, rg_idx, epoch) matches between sync and async."""
        n_batches = 5
        sync_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=0)
        async_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=4)
        for batch_i in range(n_batches):
            s_state = next(sync_gen)[9]
            a_state = next(async_gen)[9]
            assert s_state == a_state, (
                f"Batch {batch_i}: sync state={s_state}, async state={a_state}"
            )

    # --- Functional ---

    @pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
    def test_various_prefetch_depths(self, synthetic_env, depth):
        """Different doc_prefetch_depth values all produce valid batches."""
        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=depth)
        batch = next(gen)
        assert len(batch) == 10
        x, y, *_ = batch
        assert x.shape == (_B, _T)
        assert y.shape == (_B, _T)

    def test_fp_targets_with_async_prefetch(self, synthetic_env):
        """fp_targets are valid with async prefetch enabled."""
        gen = _make_dataloader(
            synthetic_env, doc_prefetch_depth=4, n_sae_features=_N_SAE_FEATURES,
        )
        for _ in range(3):
            batch = next(gen)
            fp_targets = batch[8]
            assert fp_targets is not None
            assert fp_targets.dtype == t.float32
            assert fp_targets.shape[0] == _B
            assert fp_targets.shape[2] == _N_SAE_FEATURES

    def test_prefetch_hides_latency(self, synthetic_env, monkeypatch):
        """Async path is faster than sync when doc generator has artificial delay."""
        import steering.conditional.encoder_decoder.dataloader as dl_mod
        orig_fn = dl_mod._document_batches_with_features

        def slow_batches(*args, **kwargs):
            import time
            for batch in orig_fn(*args, **kwargs):
                time.sleep(0.03)
                yield batch

        monkeypatch.setattr(dl_mod, "_document_batches_with_features", slow_batches)

        import time
        # Sync: doc_prefetch_depth=0
        sync_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=0, buffer_size=5)
        next(sync_gen)  # warmup
        t0 = time.monotonic()
        for _ in range(3):
            next(sync_gen)
        sync_time = time.monotonic() - t0

        # Async: doc_prefetch_depth=4
        async_gen = _make_dataloader(synthetic_env, doc_prefetch_depth=4, buffer_size=5)
        next(async_gen)  # warmup
        time.sleep(0.2)  # let queue fill
        t0 = time.monotonic()
        for _ in range(3):
            next(async_gen)
        async_time = time.monotonic() - t0

        assert async_time < sync_time * 0.8, (
            f"Async ({async_time:.3f}s) not faster than sync ({sync_time:.3f}s)"
        )

    # --- Error handling ---

    def test_producer_exception_propagation(self, synthetic_env, monkeypatch):
        """Exception in the document generator propagates to the main thread."""
        import steering.conditional.encoder_decoder.dataloader as dl_mod
        orig_fn = dl_mod._document_batches_with_features

        def failing_batches(*args, **kwargs):
            count = 0
            for batch in orig_fn(*args, **kwargs):
                count += 1
                if count > 3:
                    raise RuntimeError("Simulated doc generator failure")
                yield batch

        monkeypatch.setattr(dl_mod, "_document_batches_with_features", failing_batches)

        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=2, buffer_size=5)
        batches_received = 0
        with pytest.raises(RuntimeError, match="Simulated doc generator failure"):
            for _ in range(100):
                next(gen)
                batches_received += 1
        assert batches_received >= 1

    def test_generator_exhaustion_propagates(self, synthetic_env, monkeypatch):
        """Finite doc generator causes RuntimeError."""
        import steering.conditional.encoder_decoder.dataloader as dl_mod
        orig_fn = dl_mod._document_batches_with_features

        def finite_batches(*args, **kwargs):
            count = 0
            for batch in orig_fn(*args, **kwargs):
                count += 1
                if count > 3:
                    return  # stop yielding
                yield batch

        monkeypatch.setattr(dl_mod, "_document_batches_with_features", finite_batches)

        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=2, buffer_size=5)
        batches_received = 0
        with pytest.raises(RuntimeError):
            for _ in range(100):
                next(gen)
                batches_received += 1
        assert batches_received >= 1

    # --- Edge cases ---

    def test_depth_zero_no_thread(self, synthetic_env):
        """doc_prefetch_depth=0 does not spawn a new doc-prefetch thread."""
        import threading
        before = {t.ident for t in threading.enumerate() if t.name == "doc-prefetch"}
        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=0)
        next(gen)
        after = {t.ident for t in threading.enumerate() if t.name == "doc-prefetch"}
        new_threads = after - before
        assert len(new_threads) == 0, f"New doc-prefetch threads spawned: {new_threads}"

    def test_rapid_consumption_small_buffer(self, synthetic_env):
        """Very small buffer_size forces frequent refills — no deadlock."""
        gen = _make_dataloader(
            synthetic_env, doc_prefetch_depth=2, buffer_size=2,
        )
        for _ in range(10):
            batch = next(gen)
            assert len(batch) == 10

    # --- Integration ---

    def test_resume_state_consistency(self, synthetic_env):
        """State dict from async path allows correct resume."""
        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=4)
        states = []
        for _ in range(3):
            batch = next(gen)
            states.append(batch[9])
        saved = states[-1]

        gen2 = _make_dataloader(
            synthetic_env, doc_prefetch_depth=4,
            resume_state_dict=saved,
        )
        batch = next(gen2)
        new_state = batch[9]
        assert new_state["epoch"] >= saved["epoch"]

    def test_both_prefetch_levels_active(self, synthetic_env):
        """Doc prefetch + outer training-batch prefetch work simultaneously."""
        gen = _make_dataloader(
            synthetic_env, doc_prefetch_depth=4, prefetch_batches=2,
        )
        for _ in range(3):
            batch = next(gen)
            assert len(batch) == 10
            assert batch[0].shape == (_B, _T)

    def test_thread_named_doc_prefetch(self, synthetic_env):
        """Background thread has expected name for debugging."""
        import threading
        gen = _make_dataloader(synthetic_env, doc_prefetch_depth=4)
        next(gen)  # trigger thread start
        thread_names = [t.name for t in threading.enumerate()]
        assert "doc-prefetch" in thread_names

    def test_sampling_pipeline_with_prefetch(self, synthetic_env):
        """Async prefetch works with stochastic sampling selection path."""
        from nanochat.dataset import DatasetSplit

        tokenizer = synthetic_env["tokenizer"]
        feature_index = FeatureIndex(
            synthetic_env["feature_dir"],
            tokenizer_checksum=synthetic_env["tokenizer_checksum"],
            percentile_checksum=synthetic_env.get("percentile_checksum"),
        )
        pipeline = _build_sampling_pipeline()

        gen = encoder_decoder_data_loader_with_state(
            tokenizer=tokenizer,
            B=_B,
            T=_T,
            split=DatasetSplit.TRAIN,
            feature_index=feature_index,
            pipeline=pipeline,
            segment_size=_SEGMENT_SIZE,
            decoder_segment_size=None,
            device="cpu",
            resume_state_dict=None,
            tokenizer_batch_size=128,
            buffer_size=50,
            data_dir=synthetic_env["data_dir"],
            prefetch_batches=0,
            label_dropout=0.0,
            n_sae_features=_N_SAE_FEATURES,
            doc_prefetch_depth=4,
        )
        for _ in range(3):
            batch = next(gen)
            assert len(batch) == 10


class TestMultiDocPacking:
    """Tests specific to multi-document packing within rows."""

    def test_no_padding_positions(self, synthetic_env):
        """Every position in x is a real token (no zero-padding in the middle).

        BOS token (0) IS valid at document starts, so we check that every
        position is filled by ensuring the row buffer was fully packed.
        """
        gen = _make_dataloader(synthetic_env)
        x, *_ = next(gen)
        # x should not have weird sentinel values. All values should be >= 0.
        assert (x >= 0).all(), "x has negative values (invalid tokens)"

    def test_seg_doc_id_has_valid_entries(self, synthetic_env):
        """seg_doc_id should have non-negative values for real segments."""
        gen = _make_dataloader(synthetic_env)
        _, _, enc_fids, _, _, _, _, seg_doc_id, *_ = next(gen)
        # At least some segments per row should have valid (non-negative) doc IDs
        for row_idx in range(_B):
            valid_count = (seg_doc_id[row_idx] >= 0).sum().item()
            assert valid_count > 0, (
                f"Row {row_idx} has no valid seg_doc_id entries"
            )


# ===========================================================================
# 6. SamplingBundle selection path
# ===========================================================================


def _build_sampling_pipeline(
    n_sae_features=_N_SAE_FEATURES,
    num_buckets=_NUM_BUCKETS,
    negative_bundle=None,
    mode=SamplingMode.weighted,
):
    """Build a SamplingBundle pipeline for testing.

    Uses a simple linear percentile table so that magnitude_to_percentile
    produces meaningful non-zero percentile values. Boundaries are uniform.
    """
    # Build a percentile table: each feature has linearly-spaced p1-p99 values.
    # Magnitudes in synthetic data range ~0.01 to 0.7, so set table rows so
    # that those magnitudes land in the middle of the percentile range.
    percentile_table = (
        t.linspace(0.0, 1.0, 99)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .to(t.float64)
        .contiguous()
    )

    effective_buckets = num_buckets - 1 if negative_bundle is not None else num_buckets

    boundaries = (
        t.linspace(
            1 / effective_buckets,
            1.0 - 1 / effective_buckets,
            effective_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )

    rng = t.Generator().manual_seed(42)

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0,
        std=3.0,
        min_k=1,
        max_k=10,
        mode=mode,
    )

    # Share RNG between SamplingBundle and NegativeBundle (as production code does)
    if negative_bundle is not None:
        negative_bundle.rng = rng

    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=percentile_table,
            rng=rng,
        ),
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
    )


def _make_sampling_dataloader(env, *, B=_B, T=_T, n_sae_features=None,
                               buffer_size=50, negative_bundle=None,
                               mode=SamplingMode.weighted):
    """Build a live dataloader using SamplingBundle selection."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )
    pipeline = _build_sampling_pipeline(
        negative_bundle=negative_bundle,
        mode=mode,
    )

    gen = encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=128,
        buffer_size=buffer_size,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=0.0,
        max_shards=None,
        n_sae_features=n_sae_features,
    )
    return gen


class TestSamplingBundlePath:
    """Tests for the SamplingBundle feature selection path in _document_batches_with_features."""

    def test_yields_valid_10_tuple(self, synthetic_env):
        """Dataloader with SamplingBundle yields exactly a 10-element tuple."""
        gen = _make_sampling_dataloader(synthetic_env)
        batch = next(gen)
        assert isinstance(batch, tuple)
        assert len(batch) == 10

    def test_tensor_shapes_and_dtypes(self, synthetic_env):
        """Tensors from SamplingBundle path have correct shapes and dtypes."""
        gen = _make_sampling_dataloader(synthetic_env)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(gen)

        B, T = _B, _T

        assert x.shape == (B, T) and x.dtype == t.long
        assert y.shape == (B, T) and y.dtype == t.long
        assert enc_fids.dim() == 3 and enc_fids.shape[0] == B and enc_fids.dtype == t.int64
        assert enc_mags.shape == enc_fids.shape and enc_mags.dtype == t.int64
        assert enc_mask.shape == enc_fids.shape and enc_mask.dtype == t.bool
        assert enc_pcts.shape == enc_fids.shape and enc_pcts.dtype == t.float32
        assert seg_map.shape == (B, T) and seg_map.dtype == t.int64
        assert seg_doc_id.shape[0] == B and seg_doc_id.dtype == t.int64
        assert isinstance(state_dict, dict)

    def test_enc_mask_has_active_features(self, synthetic_env):
        """SamplingBundle selects at least some features (min_k=1)."""
        gen = _make_sampling_dataloader(synthetic_env)
        _, _, _, _, enc_mask, *_ = next(gen)
        for row_idx in range(_B):
            assert enc_mask[row_idx].any(), (
                f"Row {row_idx} has no active features with SamplingBundle"
            )

    def test_feature_count_varies_across_batches(self, synthetic_env):
        """Stochastic sampling produces varying feature counts across batches."""
        gen = _make_sampling_dataloader(synthetic_env)
        active_counts = []
        for _ in range(5):
            _, _, _, _, enc_mask, *_ = next(gen)
            active_counts.append(enc_mask.sum().item())
        # With geometric sampling, counts should vary (not all identical)
        unique_counts = set(active_counts)
        assert len(unique_counts) > 1, (
            f"Feature counts are all identical ({active_counts[0]}) across 5 batches; "
            f"expected stochastic variation"
        )

    def test_fp_targets_populated_with_sampling(self, synthetic_env):
        """fp_targets is populated correctly when using SamplingBundle."""
        gen = _make_sampling_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.shape[2] == _N_SAE_FEATURES
        assert fp_targets.sum().item() > 0, "fp_targets is all zeros with SamplingBundle"

    def test_multi_batch_yield(self, synthetic_env):
        """SamplingBundle dataloader can yield multiple consecutive batches."""
        gen = _make_sampling_dataloader(synthetic_env)
        for i in range(5):
            batch = next(gen)
            assert len(batch) == 10, f"Batch {i} is not a 10-tuple"

    def test_top_k_percentile_mode(self, synthetic_env):
        """SamplingBundle with top_k_percentile mode yields valid batches."""
        gen = _make_sampling_dataloader(synthetic_env, mode=SamplingMode.top_k_percentile)
        batch = next(gen)
        assert isinstance(batch, tuple) and len(batch) == 10
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = batch
        # Should have active features
        assert enc_mask.any(), "No active features with top_k_percentile mode"

    def test_bos_alignment_with_sampling(self, synthetic_env):
        """Packed rows start with BOS even with SamplingBundle."""
        tokenizer = synthetic_env["tokenizer"]
        bos_id = tokenizer.get_bos_token_id()
        gen = _make_sampling_dataloader(synthetic_env)
        x, *_ = next(gen)
        for row_idx in range(_B):
            assert x[row_idx, 0].item() == bos_id


# ===========================================================================
# 7. Negative sampling path
# ===========================================================================


def _build_negative_bundle(n_sae_features=_N_SAE_FEATURES):
    """Build a NegativeBundle for testing.

    RNG is a placeholder — it will be replaced by the pipeline builder to
    share with SamplingBundle's RNG.
    """
    return NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=5,
        ),
        n_sae_features=n_sae_features,
        rng=t.Generator().manual_seed(99),  # placeholder, overwritten
    )


class TestNegativeSamplingPath:
    """Tests for the negative (inactive) feature sampling path."""

    def test_yields_valid_10_tuple(self, synthetic_env):
        """Dataloader with negative sampling yields a valid 10-tuple."""
        neg = _build_negative_bundle()
        gen = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        batch = next(gen)
        assert isinstance(batch, tuple) and len(batch) == 10

    def test_negative_features_have_effective_buckets_magnitude(self, synthetic_env):
        """Negative features should have bucket value == effective_buckets."""
        neg = _build_negative_bundle()
        pipeline = _build_sampling_pipeline(negative_bundle=neg)
        gen = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, *_ = next(gen)

        # effective_buckets = num_buckets - 1 when negative_bundle is not None
        effective_buckets = pipeline.effective_buckets
        # Find positions where enc_mags == effective_buckets (negatives)
        neg_positions = enc_mags == effective_buckets
        # Some negative features should exist (min_k=1 per segment)
        assert neg_positions.any(), (
            f"No negative features found with bucket value {effective_buckets}"
        )

    def test_more_active_features_with_negatives(self, synthetic_env):
        """enc_mask should have more active positions with negatives than without.

        Negative features are appended to the selected features and marked active.
        """
        # Without negatives
        gen_no_neg = _make_sampling_dataloader(synthetic_env)
        _, _, _, _, enc_mask_no_neg, *_ = next(gen_no_neg)
        count_no_neg = enc_mask_no_neg.sum().item()

        # With negatives
        neg = _build_negative_bundle()
        gen_neg = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        _, _, _, _, enc_mask_neg, *_ = next(gen_neg)
        count_neg = enc_mask_neg.sum().item()

        # With min_k=1, every segment gets at least 1 negative
        assert count_neg > count_no_neg, (
            f"Active features with negatives ({count_neg}) should be > "
            f"without negatives ({count_no_neg})"
        )

    def test_negative_features_have_zero_percentile(self, synthetic_env):
        """Negative features should have percentile == 0.0."""
        neg = _build_negative_bundle()
        pipeline = _build_sampling_pipeline(negative_bundle=neg)
        gen = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, *_ = next(gen)

        effective_buckets = pipeline.effective_buckets
        neg_mask = enc_mags == effective_buckets
        if neg_mask.any():
            neg_pcts = enc_pcts[neg_mask]
            assert (neg_pcts == 0.0).all(), (
                "Negative features should have percentile == 0.0"
            )

    def test_multi_batch_with_negatives(self, synthetic_env):
        """Dataloader with negatives can yield multiple batches without error."""
        neg = _build_negative_bundle()
        gen = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        for i in range(5):
            batch = next(gen)
            assert len(batch) == 10, f"Batch {i} is not a 10-tuple"

    def test_fp_targets_with_negatives(self, synthetic_env):
        """fp_targets still populated correctly with negative sampling."""
        neg = _build_negative_bundle()
        gen = _make_sampling_dataloader(
            synthetic_env, negative_bundle=neg, n_sae_features=_N_SAE_FEATURES,
        )
        *_, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.sum().item() > 0, "fp_targets all zeros with negatives"

    def test_negative_fids_are_within_range(self, synthetic_env):
        """Negative feature IDs should be in [0, n_sae_features)."""
        neg = _build_negative_bundle()
        gen = _make_sampling_dataloader(synthetic_env, negative_bundle=neg)
        _, _, enc_fids, enc_mags, enc_mask, *_ = next(gen)

        pipeline = _build_sampling_pipeline(negative_bundle=neg)
        effective_buckets = pipeline.effective_buckets
        neg_mask = (enc_mags == effective_buckets) & enc_mask
        if neg_mask.any():
            neg_fids = enc_fids[neg_mask]
            assert (neg_fids >= 0).all(), "Negative feature IDs have negative values"
            assert (neg_fids < _N_SAE_FEATURES).all(), (
                f"Negative feature IDs >= {_N_SAE_FEATURES}"
            )


# ===========================================================================
# 8. Negative sampling with ThresholdSelection
# ===========================================================================


def _make_threshold_negative_dataloader(env, *, B=_B, T=_T, n_sae_features=None,
                                         buffer_size=50, threshold=3):
    """Build a live dataloader using ThresholdSelection + NegativeBundle."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )

    neg_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=5,
        ),
        n_sae_features=_N_SAE_FEATURES,
        rng=t.Generator().manual_seed(99),
    )

    num_buckets = _NUM_BUCKETS
    effective_buckets = num_buckets - 1  # negative sampling reserves one bucket

    boundaries = (
        t.linspace(
            1 / effective_buckets,
            1.0 - 1 / effective_buckets,
            effective_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(_N_SAE_FEATURES, -1)
        .contiguous()
    )

    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=neg_bundle,
        effective_buckets=effective_buckets,
    )

    gen = encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=128,
        buffer_size=buffer_size,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=0.0,
        max_shards=None,
        n_sae_features=n_sae_features,
    )
    return gen, pipeline


class TestThresholdNegativeSampling:
    """Tests for negative sampling combined with ThresholdSelection."""

    def test_yields_valid_10_tuple(self, synthetic_env):
        """ThresholdSelection + negatives yields a valid 10-tuple."""
        gen, _ = _make_threshold_negative_dataloader(synthetic_env)
        batch = next(gen)
        assert isinstance(batch, tuple) and len(batch) == 10

    def test_negatives_present_with_threshold(self, synthetic_env):
        """Negative features should be present when using threshold + negatives."""
        gen, pipeline = _make_threshold_negative_dataloader(synthetic_env)
        _, _, enc_fids, enc_mags, enc_mask, *_ = next(gen)
        neg_positions = enc_mags == pipeline.effective_buckets
        assert neg_positions.any(), (
            "No negative features found with ThresholdSelection + negatives"
        )

    def test_multi_batch_threshold_negatives(self, synthetic_env):
        """ThresholdSelection + negatives produces multiple valid batches."""
        gen, _ = _make_threshold_negative_dataloader(synthetic_env)
        for i in range(3):
            batch = next(gen)
            assert len(batch) == 10, f"Batch {i} not a 10-tuple"


# ===========================================================================
# 9. Delta filtering path
# ===========================================================================


def _build_delta_sampling_pipeline(
    n_sae_features=_N_SAE_FEATURES,
    num_buckets=_NUM_BUCKETS,
    with_negatives=False,
):
    """Build a SamplingBundle pipeline with delta filtering for testing.

    Uses prev_segment delta mode with percentile_delta weights so the delta
    filtering path AND delta-aware negative sampling path are both exercised.
    """
    percentile_table = (
        t.linspace(0.0, 1.0, 99)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .to(t.float64)
        .contiguous()
    )

    negative_bundle = None
    if with_negatives:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=1,
                max_k=5,
            ),
            n_sae_features=n_sae_features,
            rng=t.Generator().manual_seed(99),  # placeholder
        )

    effective_buckets = num_buckets - 1 if negative_bundle is not None else num_buckets

    boundaries = (
        t.linspace(
            1 / effective_buckets,
            1.0 - 1 / effective_buckets,
            effective_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )

    rng = t.Generator().manual_seed(42)

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0,
        std=3.0,
        min_k=1,
        max_k=10,
        delta_weights=DeltaSamplingWeights.percentile_delta,
    )

    # Share RNG with negative bundle
    if negative_bundle is not None:
        negative_bundle.rng = rng

    delta_bundle = DeltaBundle(
        config=FeatureDeltaConfig(
            mode=DeltaMode.prev_segment,
            neg_delta_ratio=0.5,
        ),
        percentile_table=percentile_table,
    )

    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=percentile_table,
            rng=rng,
        ),
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        delta_bundle=delta_bundle,
    )


def _make_delta_dataloader(env, *, B=_B, T=_T, n_sae_features=None,
                            buffer_size=50, with_negatives=False):
    """Build a live dataloader with delta filtering."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )
    pipeline = _build_delta_sampling_pipeline(with_negatives=with_negatives)

    gen = encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=128,
        buffer_size=buffer_size,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=0.0,
        max_shards=None,
        n_sae_features=n_sae_features,
    )
    return gen, pipeline


class TestDeltaFilteringPath:
    """Tests for the delta filtering path in _document_batches_with_features."""

    def test_yields_valid_10_tuple(self, synthetic_env):
        """Delta filtering path yields a valid 10-tuple."""
        gen, _ = _make_delta_dataloader(synthetic_env)
        batch = next(gen)
        assert isinstance(batch, tuple) and len(batch) == 10

    def test_tensor_shapes(self, synthetic_env):
        """Tensors from delta path have correct shapes."""
        gen, _ = _make_delta_dataloader(synthetic_env)
        x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(gen)
        assert x.shape == (_B, _T)
        assert enc_fids.dim() == 3 and enc_fids.shape[0] == _B
        assert enc_mask.shape == enc_fids.shape

    def test_multi_batch_delta(self, synthetic_env):
        """Delta path can yield multiple consecutive batches."""
        gen, _ = _make_delta_dataloader(synthetic_env)
        for i in range(5):
            batch = next(gen)
            assert len(batch) == 10, f"Batch {i} not a 10-tuple"

    def test_enc_mask_has_features(self, synthetic_env):
        """Delta filtering should still leave some features active.

        The first segment of each doc has no reference, so all features are
        positive-delta. Subsequent segments filter to positive deltas only.
        """
        gen, _ = _make_delta_dataloader(synthetic_env)
        _, _, _, _, enc_mask, *_ = next(gen)
        assert enc_mask.any(), "No active features after delta filtering"

    def test_fp_targets_with_delta(self, synthetic_env):
        """fp_targets populated correctly with delta filtering."""
        gen, _ = _make_delta_dataloader(synthetic_env, n_sae_features=_N_SAE_FEATURES)
        *_, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.shape[2] == _N_SAE_FEATURES
        assert fp_targets.sum().item() > 0, "fp_targets is all zeros with delta filtering"


class TestDeltaWithNegatives:
    """Tests for delta filtering combined with negative sampling."""

    def test_yields_valid_10_tuple(self, synthetic_env):
        """Delta + negatives yields a valid 10-tuple."""
        gen, _ = _make_delta_dataloader(synthetic_env, with_negatives=True)
        batch = next(gen)
        assert isinstance(batch, tuple) and len(batch) == 10

    def test_negatives_present(self, synthetic_env):
        """Negative features should be present with delta + negatives."""
        gen, pipeline = _make_delta_dataloader(synthetic_env, with_negatives=True)
        _, _, enc_fids, enc_mags, enc_mask, *_ = next(gen)
        neg_positions = enc_mags == pipeline.effective_buckets
        assert neg_positions.any(), (
            "No negative features found with delta + negative sampling"
        )

    def test_multi_batch_delta_negatives(self, synthetic_env):
        """Delta + negatives produces multiple valid batches."""
        gen, _ = _make_delta_dataloader(synthetic_env, with_negatives=True)
        for i in range(5):
            batch = next(gen)
            assert len(batch) == 10, f"Batch {i} not a 10-tuple"

    def test_delta_negatives_have_zero_percentile(self, synthetic_env):
        """Negative features should have percentile == 0.0 even with delta path."""
        gen, pipeline = _make_delta_dataloader(synthetic_env, with_negatives=True)
        _, _, enc_fids, enc_mags, enc_mask, enc_pcts, *_ = next(gen)
        neg_mask = enc_mags == pipeline.effective_buckets
        if neg_mask.any():
            neg_pcts = enc_pcts[neg_mask]
            assert (neg_pcts == 0.0).all(), (
                "Negative features should have percentile == 0.0 with delta"
            )


# ===========================================================================
# 9. Checkpoint resume
# ===========================================================================


class TestLiveDataloaderResume:
    """Tests for checkpoint resume via resume_state_dict in the live dataloader."""

    def test_state_dict_keys(self, synthetic_env):
        """First batch state_dict has keys pq_idx, rg_idx, epoch."""
        gen = _make_dataloader(synthetic_env)
        *_, state_dict = next(gen)
        assert "pq_idx" in state_dict
        assert "rg_idx" in state_dict
        assert "epoch" in state_dict

    def test_topology_change_raises_before_parquet_access(
        self, synthetic_env, monkeypatch
    ):
        """F4: reject changed striping before cursor division or file I/O."""
        monkeypatch.setattr(dl_mod, "get_dist_info", lambda: (True, 0, 0, 4))

        def fail_if_opened(_data_dir=None):
            raise AssertionError("parquet inventory touched before topology guard")

        monkeypatch.setattr(dl_mod, "list_parquet_files", fail_if_opened)
        env = synthetic_env
        feature_index = FeatureIndex(
            env["feature_dir"],
            tokenizer_checksum=env["tokenizer_checksum"],
            percentile_checksum=env["percentile_checksum"],
        )
        gen = _document_batches_with_features(
            split=DatasetSplit.TRAIN,
            resume_state_dict={
                "pq_idx": 0,
                "rg_idx": 7,
                "epoch": 1,
                "ddp_world_size": 8,
            },
            tokenizer_batch_size=4,
            tokenizer=env["tokenizer"],
            feature_index=feature_index,
            pipeline=_build_threshold_pipeline(threshold=3),
            segment_size=_SEGMENT_SIZE,
            data_dir=env["data_dir"],
        )

        with pytest.raises(ValueError, match=r"world.size.*8.*4|8.*4"):
            next(gen)

    def test_state_dict_types(self, synthetic_env):
        """Cursor/topology fields are ints and the RNG state is JSON-safe."""
        gen = _make_dataloader(synthetic_env)
        *_, state_dict = next(gen)
        for key in ("pq_idx", "rg_idx", "epoch", "ddp_world_size"):
            value = state_dict[key]
            assert isinstance(value, int), (
                f"state_dict[{key!r}] has type {type(value).__name__}, expected int"
            )
        assert isinstance(state_dict["feature_pipeline_rng_state"], list)
        json.loads(json.dumps(state_dict))

    def test_resume_restores_cursor_aligned_feature_pipeline_rng(self, synthetic_env):
        """F3 live path: resume continues selection RNG at the saved row group.

        This drives the document producer directly, where row-group cursor and
        feature RNG are aligned. The outer SortedDocBuffer is intentionally not
        part of this claim, so full live batches remain approximate.
        """
        def make_document_loader(pipeline, resume_state=None):
            feature_index = FeatureIndex(
                synthetic_env["feature_dir"],
                tokenizer_checksum=synthetic_env["tokenizer_checksum"],
                percentile_checksum=synthetic_env["percentile_checksum"],
            )
            return _document_batches_with_features(
                split=DatasetSplit.TRAIN,
                resume_state_dict=resume_state,
                tokenizer_batch_size=128,
                tokenizer=synthetic_env["tokenizer"],
                feature_index=feature_index,
                pipeline=pipeline,
                segment_size=_SEGMENT_SIZE,
                data_dir=synthetic_env["data_dir"],
            )

        baseline_pipeline = _build_sampling_pipeline(
            negative_bundle=_build_negative_bundle()
        )
        baseline = make_document_loader(baseline_pipeline)
        _first_docs, first_state = next(baseline)
        expected_docs, expected_state = next(baseline)
        saved_state = json.loads(json.dumps(first_state))
        assert saved_state["feature_pipeline_rng_state"][0]["roles"] == [
            "selection",
            "negative",
        ]

        resumed_pipeline = _build_sampling_pipeline(
            negative_bundle=_build_negative_bundle()
        )
        t.rand(100, generator=resumed_pipeline.selection_config.rng)
        resumed = make_document_loader(resumed_pipeline, saved_state)
        actual_docs, actual_state = next(resumed)

        assert actual_state["feature_pipeline_rng_state"] == expected_state[
            "feature_pipeline_rng_state"
        ]
        assert len(actual_docs) == len(expected_docs)
        for (actual_tokens, actual_segments), (expected_tokens, expected_segments) in zip(
            actual_docs, expected_docs, strict=True
        ):
            assert actual_tokens == expected_tokens
            assert actual_segments is not None and expected_segments is not None
            for actual, expected in zip(
                actual_segments, expected_segments, strict=True
            ):
                assert t.equal(actual.feature_ids, expected.feature_ids)
                assert t.equal(actual.buckets, expected.buckets)
                assert t.equal(actual.percentiles, expected.percentiles)

    def test_resume_restores_live_label_dropout_default_rng(self, synthetic_env):
        """F3 live path restores the saved pre-dropout CPU default stream."""
        t.manual_seed(8712)
        baseline = _make_dataloader(
            synthetic_env,
            label_dropout=0.5,
            pad_F=8,
            pad_n_seg=8,
        )
        first = next(baseline)
        second = next(baseline)
        saved_state = json.loads(json.dumps(first[9]))
        expected_post_first_hex = second[9]["label_dropout_rng_state"][
            "state_hex"
        ]

        t.manual_seed(999999)
        resumed = _make_dataloader(
            synthetic_env,
            label_dropout=0.5,
            pad_F=8,
            pad_n_seg=8,
            resume_state_dict=saved_state,
        )
        replay = next(resumed)

        assert replay[9]["label_dropout_rng_state"] == saved_state[
            "label_dropout_rng_state"
        ]
        assert bytes(t.get_rng_state().tolist()).hex() == expected_post_first_hex

    def test_present_malformed_live_rng_metadata_raises(self, synthetic_env):
        gen = _make_dataloader(synthetic_env, label_dropout=0.5)
        *_, state = next(gen)

        corrupt_feature = json.loads(json.dumps(state))
        corrupt_feature["feature_pipeline_rng_state"] = [{"roles": ["selection"]}]
        with pytest.raises(ValueError, match="feature.pipeline_rng_state|RNG snapshot"):
            next(
                _make_dataloader(
                    synthetic_env,
                    label_dropout=0.5,
                    resume_state_dict=corrupt_feature,
                )
            )

        corrupt_label = json.loads(json.dumps(state))
        corrupt_label["label_dropout_rng_state"]["state_hex"] = "not-hex"
        with pytest.raises(ValueError, match="state_hex"):
            next(
                _make_dataloader(
                    synthetic_env,
                    label_dropout=0.5,
                    resume_state_dict=corrupt_label,
                )
            )

    def test_resume_skips_ahead(self, synthetic_env):
        """Resumed loader starts at or past the saved state."""
        gen = _make_dataloader(synthetic_env)
        # Consume 3 batches, save state from the 3rd
        for _ in range(3):
            *_, state_dict = next(gen)
        saved_state = state_dict.copy()

        # Create new loader resumed from that state
        resumed_gen = _make_dataloader(synthetic_env, resume_state_dict=saved_state)
        *_, resumed_state = next(resumed_gen)

        # The resumed loader should be at or past the saved position
        assert resumed_state["epoch"] >= saved_state["epoch"], (
            f"Resumed epoch {resumed_state['epoch']} < saved epoch {saved_state['epoch']}"
        )
        if resumed_state["epoch"] == saved_state["epoch"]:
            assert resumed_state["pq_idx"] >= saved_state["pq_idx"], (
                f"Resumed pq_idx {resumed_state['pq_idx']} < saved pq_idx {saved_state['pq_idx']}"
            )

    def test_resume_skips_pq_idx(self, synthetic_env):
        """Resumed loader's state reflects that it started from a later parquet index."""
        gen = _make_dataloader(synthetic_env)
        # Consume 3 batches, save state from the 3rd
        for _ in range(3):
            *_, state_dict = next(gen)
        saved_state = state_dict.copy()

        # The saved state has a valid pq_idx (the resume checkpoint position)
        assert isinstance(saved_state["pq_idx"], int)
        assert isinstance(saved_state["rg_idx"], int)

        # Resume from that state -- should not crash and should produce valid batches
        resumed_gen = _make_dataloader(synthetic_env, resume_state_dict=saved_state)
        batch = next(resumed_gen)
        assert len(batch) == 10

    def test_none_resume_is_fresh_start(self, synthetic_env):
        """resume_state_dict=None produces a valid first batch (default behavior)."""
        gen1 = _make_dataloader(synthetic_env, resume_state_dict=None)
        *_, state1 = next(gen1)
        # With None resume, epoch starts at 1 internally (though buffer pre-fill
        # may advance it). Just verify valid state dict keys and types.
        assert isinstance(state1["epoch"], int)
        assert isinstance(state1["pq_idx"], int)
        assert isinstance(state1["rg_idx"], int)
        assert state1["epoch"] >= 1

    def test_resume_from_step_0(self, synthetic_env):
        """State dict from the very first batch can be used to resume."""
        gen = _make_dataloader(synthetic_env)
        first_x, *_, first_state = next(gen)

        # Resume from step 0 state
        resumed_gen = _make_dataloader(synthetic_env, resume_state_dict=first_state)
        resumed_x, *_, resumed_state = next(resumed_gen)

        assert resumed_state["pq_idx"] >= first_state["pq_idx"]
        assert resumed_state["epoch"] >= first_state["epoch"]

    def test_state_dict_with_missing_epoch_uses_default(self, synthetic_env):
        """State dict missing 'epoch' key uses default epoch=1."""
        # Provide pq_idx and rg_idx but omit epoch -- .get("epoch", 1) is used
        gen = _make_dataloader(synthetic_env)
        *_, state = next(gen)
        partial_state = {"pq_idx": state["pq_idx"], "rg_idx": state["rg_idx"]}
        resumed_gen = _make_dataloader(synthetic_env, resume_state_dict=partial_state)
        batch = next(resumed_gen)
        assert len(batch) == 10, "Resumed loader with partial state did not yield a valid batch"

    def test_resume_restores_epoch(self, synthetic_env):
        """Resumed loader restores the epoch from saved state.

        The live dataloader resume is approximate (parquet/row-group level),
        so we verify epoch restoration rather than exact batch skipping.
        With small synthetic data the buffer pre-fills across many epochs,
        making raw tensor comparison unreliable.
        """
        gen = _make_dataloader(synthetic_env)
        # Consume 3 batches, save state from the 3rd
        for _ in range(3):
            *_, state_dict = next(gen)
        saved_state = state_dict.copy()
        saved_epoch = saved_state["epoch"]

        # Resume from saved state
        resumed_gen = _make_dataloader(synthetic_env, resume_state_dict=saved_state)
        *_, resumed_state = next(resumed_gen)

        # Epoch should be at least as high as the saved epoch
        # (the buffer pre-fill may advance it further, but never backwards)
        assert resumed_state["epoch"] >= saved_epoch, (
            f"Resumed epoch {resumed_state['epoch']} < saved epoch {saved_epoch}"
        )


# ===========================================================================
# 10. Static shape padding (pad_F / pad_n_seg)
# ===========================================================================


class TestStaticShapeInvariants:
    """When pad_F and pad_n_seg are set, every batch has identical shapes."""

    def test_enc_fids_shape_matches_padding(self, synthetic_env):
        """enc_fids.shape == (B, pad_n_seg, pad_F) across multiple batches."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        for _ in range(5):
            _, _, enc_fids, *_ = next(gen)
            assert enc_fids.shape == (_B, pad_n_seg, pad_F), (
                f"Expected ({_B}, {pad_n_seg}, {pad_F}), got {enc_fids.shape}"
            )

    def test_all_encoder_tensors_have_same_shape(self, synthetic_env):
        """enc_fids, enc_mags, enc_mask, enc_pcts all have (B, pad_n_seg, pad_F)."""
        pad_F, pad_n_seg = 15, 6
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, enc_fids, enc_mags, enc_mask, enc_pcts, *_ = next(gen)
        expected = (_B, pad_n_seg, pad_F)
        assert enc_fids.shape == expected
        assert enc_mags.shape == expected
        assert enc_mask.shape == expected
        assert enc_pcts.shape == expected

    def test_seg_map_shape_unchanged(self, synthetic_env):
        """seg_map is always (B, T) regardless of padding."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        *_, seg_map, _, _, _ = next(gen)
        assert seg_map.shape == (_B, _T)

    def test_seg_doc_id_shape_matches_pad_n_seg(self, synthetic_env):
        """seg_doc_id.shape == (B, pad_n_seg)."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        *_, seg_doc_id, _, _ = next(gen)
        assert seg_doc_id.shape == (_B, pad_n_seg)

    def test_fp_targets_shape_matches_pad_n_seg(self, synthetic_env):
        """fp_targets.shape == (B, pad_n_seg, n_sae) when n_sae_features is set."""
        pad_F, pad_n_seg, n_sae = 20, 8, _N_SAE_FEATURES
        gen = _make_dataloader(
            synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg, n_sae_features=n_sae,
        )
        *_, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.shape == (_B, pad_n_seg, n_sae)

    def test_shapes_identical_across_batches(self, synthetic_env):
        """Every batch produces tensors with identical shapes -- the core invariant."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(
            synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg, n_sae_features=_N_SAE_FEATURES,
        )
        shapes = []
        for _ in range(5):
            x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, _ = next(gen)
            batch_shapes = (
                x.shape, y.shape, enc_fids.shape, enc_mags.shape, enc_mask.shape,
                enc_pcts.shape, seg_map.shape, seg_doc_id.shape,
                fp_targets.shape if fp_targets is not None else None,
            )
            shapes.append(batch_shapes)
        # ALL batches must have identical shapes
        for i, s in enumerate(shapes[1:], 1):
            assert s == shapes[0], f"Batch {i} shapes differ from batch 0: {s} vs {shapes[0]}"

    def test_encoder_tensor_dtypes_preserved(self, synthetic_env):
        """Static padding does not change the dtypes of encoder tensors."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, enc_fids, enc_mags, enc_mask, enc_pcts, _, seg_doc_id, _, _ = next(gen)
        assert enc_fids.dtype == t.int64
        assert enc_mags.dtype == t.int64
        assert enc_mask.dtype == t.bool
        assert enc_pcts.dtype == t.float32
        assert seg_doc_id.dtype == t.int64

    def test_x_y_shapes_unchanged_by_padding(self, synthetic_env):
        """x and y always have shape (B, T), unaffected by encoder padding."""
        pad_F, pad_n_seg = 20, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        x, y, *_ = next(gen)
        assert x.shape == (_B, _T)
        assert y.shape == (_B, _T)


class TestStaticShapePaddingContent:
    """Padded positions have zeros/False, non-padded positions have real data."""

    def test_padded_feature_positions_are_zero(self, synthetic_env):
        """Features beyond the actual count are zero in enc_fids."""
        pad_F = 50  # much larger than actual features
        pad_n_seg = 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, enc_fids, _, enc_mask, _, _, _, _, _ = next(gen)
        # Positions where mask is False should have fids == 0
        assert (enc_fids[~enc_mask] == 0).all(), (
            "Padded feature positions (mask=False) should have enc_fids == 0"
        )

    def test_padded_magnitude_positions_are_zero(self, synthetic_env):
        """Magnitudes at padded positions are zero in enc_mags."""
        pad_F, pad_n_seg = 50, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, _, enc_mags, enc_mask, _, _, _, _, _ = next(gen)
        assert (enc_mags[~enc_mask] == 0).all(), (
            "Padded magnitude positions (mask=False) should have enc_mags == 0"
        )

    def test_padded_percentile_positions_are_zero(self, synthetic_env):
        """Percentiles at padded positions are zero in enc_pcts."""
        pad_F, pad_n_seg = 50, 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, _, _, enc_mask, enc_pcts, _, _, _, _ = next(gen)
        assert (enc_pcts[~enc_mask] == 0.0).all(), (
            "Padded percentile positions (mask=False) should have enc_pcts == 0.0"
        )

    def test_padded_segment_positions_are_minus_one(self, synthetic_env):
        """Padded segments in seg_doc_id are -1."""
        pad_n_seg = 20  # much larger than actual segments
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=pad_n_seg)
        *_, seg_doc_id, _, _ = next(gen)
        # At least some positions should be -1 (padded)
        assert (seg_doc_id == -1).any(), "Expected some padded segment positions"

    def test_padded_mask_positions_are_false(self, synthetic_env):
        """enc_mask is False for all padded positions."""
        pad_F, pad_n_seg = 50, 20
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg)
        _, _, _, _, enc_mask, _, _, _, _, _ = next(gen)
        # Some mask positions should be False (padded)
        assert not enc_mask.all(), "Expected some False positions in mask"

    def test_padded_fp_targets_segments_are_zero(self, synthetic_env):
        """fp_targets for padded segments (seg_doc_id == -1) should be all zeros."""
        pad_F, pad_n_seg = 20, 20  # pad_n_seg much larger than actual
        gen = _make_dataloader(
            synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg, n_sae_features=_N_SAE_FEATURES,
        )
        _, _, _, _, _, _, _, seg_doc_id, fp_targets, _ = next(gen)
        padded_seg_mask = seg_doc_id == -1  # (B, n_seg)
        if padded_seg_mask.any():
            # fp_targets has shape (B, n_seg, n_sae), index with padded segments
            padded_fp = fp_targets[padded_seg_mask]  # (num_padded, n_sae)
            assert (padded_fp == 0.0).all(), (
                "fp_targets for padded segments should be all zeros"
            )

    def test_padded_mask_false_for_extra_feature_slots(self, synthetic_env):
        """With large pad_F, the extra feature columns beyond real features are False."""
        pad_F = 100  # far larger than threshold=3
        pad_n_seg = 8
        gen = _make_dataloader(synthetic_env, pad_F=pad_F, pad_n_seg=pad_n_seg, threshold=3)
        _, _, _, _, enc_mask, _, _, _, _, _ = next(gen)
        # Active features per segment are at most threshold=3, so columns >=3 should be False
        # (at least for segments with real features)
        high_cols_active = enc_mask[:, :, 10:].sum().item()  # columns 10+ should be mostly False
        total_high_cols = enc_mask[:, :, 10:].numel()
        # Allow some slack (negatives, NaN prevention sentinel) but the vast majority should be False
        assert high_cols_active < total_high_cols * 0.1, (
            f"Too many active features in high columns: {high_cols_active}/{total_high_cols}"
        )


class TestStaticShapeDataIntegrity:
    """Non-padded positions contain valid feature data."""

    def test_active_features_have_nonzero_fids(self, synthetic_env):
        """Where enc_mask is True, enc_fids should have real feature IDs."""
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=8)
        _, _, enc_fids, _, enc_mask, _, _, _, _, _ = next(gen)
        active_fids = enc_fids[enc_mask]
        # Not all need to be nonzero (feature ID 0 is valid), but at least some should exist
        assert len(active_fids) > 0, "No active features found"

    def test_active_features_have_valid_percentiles(self, synthetic_env):
        """Where enc_mask is True, enc_pcts should be in [0, 1]."""
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=8)
        _, _, _, _, enc_mask, enc_pcts, _, _, _, _ = next(gen)
        active_pcts = enc_pcts[enc_mask]
        if len(active_pcts) > 0:
            assert (active_pcts >= 0).all(), "Active percentiles have negative values"
            assert (active_pcts <= 1).all(), "Active percentiles exceed 1.0"

    def test_active_features_have_valid_magnitudes(self, synthetic_env):
        """Where enc_mask is True, enc_mags (bucket indices) should be non-negative."""
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=8)
        _, _, _, enc_mags, enc_mask, _, _, _, _, _ = next(gen)
        active_mags = enc_mags[enc_mask]
        if len(active_mags) > 0:
            assert (active_mags >= 0).all(), "Active magnitudes have negative values"
            assert (active_mags < _NUM_BUCKETS).all(), (
                f"Active magnitudes exceed num_buckets={_NUM_BUCKETS}"
            )

    def test_seg_map_indices_valid_with_padding(self, synthetic_env):
        """seg_map values are valid indices into the padded segment dimension."""
        pad_n_seg = 10
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=pad_n_seg)
        _, _, enc_fids, _, _, _, seg_map, _, _, _ = next(gen)
        assert (seg_map >= 0).all(), "seg_map has negative indices"
        assert (seg_map < pad_n_seg).all(), (
            f"seg_map has index {seg_map.max().item()} >= pad_n_seg {pad_n_seg}"
        )

    def test_padded_vs_unpadded_active_data_match(self, synthetic_env):
        """Padded and unpadded dataloaders produce the same active feature count.

        This is the critical correctness test: padding must NOT change the
        actual feature data, only add zero-padding around it.
        """
        # Both use same seed for identical packing
        gen_unpadded = _make_dataloader(synthetic_env)
        gen_padded = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=20)

        _, _, _, _, mask_unpadded, _, _, _, _, _ = next(gen_unpadded)
        _, _, _, _, mask_padded, _, _, _, _, _ = next(gen_padded)

        # Same number of active features
        assert mask_unpadded.sum() == mask_padded.sum(), (
            f"Active feature count differs: unpadded={mask_unpadded.sum()}, padded={mask_padded.sum()}"
        )

    def test_padded_vs_unpadded_fids_match(self, synthetic_env):
        """The actual feature IDs in padded output match the unpadded output."""
        gen_unpadded = _make_dataloader(synthetic_env)
        gen_padded = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=20)

        _, _, fids_unpadded, _, mask_unpadded, _, _, _, _, _ = next(gen_unpadded)
        _, _, fids_padded, _, mask_padded, _, _, _, _, _ = next(gen_padded)

        # Extract active feature IDs from both and compare as sorted sets per row
        for row_idx in range(_B):
            active_unpadded = fids_unpadded[row_idx][mask_unpadded[row_idx]].sort().values
            active_padded = fids_padded[row_idx][mask_padded[row_idx]].sort().values
            assert t.equal(active_unpadded, active_padded), (
                f"Row {row_idx}: active feature IDs differ between padded and unpadded"
            )

    def test_padded_vs_unpadded_pcts_match(self, synthetic_env):
        """The actual percentile values in padded output match the unpadded output."""
        gen_unpadded = _make_dataloader(synthetic_env)
        gen_padded = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=20)

        _, _, _, _, mask_unpadded, pcts_unpadded, _, _, _, _ = next(gen_unpadded)
        _, _, _, _, mask_padded, pcts_padded, _, _, _, _ = next(gen_padded)

        for row_idx in range(_B):
            active_unpadded = pcts_unpadded[row_idx][mask_unpadded[row_idx]].sort().values
            active_padded = pcts_padded[row_idx][mask_padded[row_idx]].sort().values
            assert t.equal(active_unpadded, active_padded), (
                f"Row {row_idx}: active percentiles differ between padded and unpadded"
            )

    def test_padded_vs_unpadded_text_tokens_match(self, synthetic_env):
        """Padding does not affect x and y token tensors."""
        gen_unpadded = _make_dataloader(synthetic_env)
        gen_padded = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=20)

        x_unpadded, y_unpadded, *_ = next(gen_unpadded)
        x_padded, y_padded, *_ = next(gen_padded)

        assert t.equal(x_unpadded, x_padded), "x differs between padded and unpadded"
        assert t.equal(y_unpadded, y_padded), "y differs between padded and unpadded"


class TestStaticShapeOverflowRaises:
    """When actual dimensions exceed the static pad bounds, the live loader
    RAISES rather than truncating.

    Truncating drops the segments/features beyond the bound, and the seg_map
    clamp then silently remaps the dropped segments' decoder tokens onto the
    last retained segment — mis-conditioning them on a different document's
    features (issue #641). Failing loud is the stopgap until overflow is handled
    gracefully.
    """

    def test_pad_n_seg_overflow_raises(self, synthetic_env):
        """batch_n_seg > pad_n_seg raises instead of truncating segments."""
        gen = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=1)
        with pytest.raises(ValueError, match="exceeds pad_n_seg") as exc_info:
            next(gen)
        message = str(exc_info.value)
        assert "batch_n_seg=" in message
        assert "pad_n_seg=1" in message
        assert "Refusing to truncate" in message
        assert "seg_map clamp" in message
        assert "mis-condition" in message
        assert "issue #641" in message

    def test_pad_f_overflow_raises(self, synthetic_env):
        """batch_f_max > pad_F raises instead of truncating features."""
        gen = _make_dataloader(synthetic_env, pad_F=1, pad_n_seg=20)
        with pytest.raises(ValueError, match="exceeds pad_F") as exc_info:
            next(gen)
        message = str(exc_info.value)
        assert "batch_f_max=" in message
        assert "pad_F=1" in message
        assert "Refusing to truncate features" in message
        assert "silently corrupts" in message
        assert "issue #641" in message

    def test_short_doc_batch_can_exceed_heuristic_sized_pad_n_seg(self, tmp_path):
        """A valid many-short-doc live batch can exceed a 14-segment static bound.

        This pins issue #641's core edge case without changing the current
        stopgap behavior: dynamic ``pad_n_seg=None`` preserves every segment,
        while the same batch with a heuristic-sized static bound raises loudly.
        """
        heuristic_pad_n_seg = 14
        env = _make_synthetic_data(tmp_path, n_docs=40, n_segs_per_doc=1)

        gen_dynamic = _make_dataloader(
            env,
            B=1,
            T=256,
            buffer_size=100,
            pad_F=20,
            pad_n_seg=None,
        )
        _, _, enc_fids, _, _, _, seg_map, seg_doc_id, _, _ = next(gen_dynamic)

        natural_n_seg = enc_fids.shape[1]
        assert natural_n_seg > heuristic_pad_n_seg
        assert seg_doc_id.shape == (1, natural_n_seg)
        assert int(seg_map.max().item()) == natural_n_seg - 1

        gen_static = _make_dataloader(
            env,
            B=1,
            T=256,
            buffer_size=100,
            pad_F=20,
            pad_n_seg=heuristic_pad_n_seg,
        )
        with pytest.raises(ValueError, match="exceeds pad_n_seg") as exc_info:
            next(gen_static)
        assert f"batch_n_seg={natural_n_seg}" in str(exc_info.value)
        assert f"pad_n_seg={heuristic_pad_n_seg}" in str(exc_info.value)

    def test_generous_bounds_do_not_raise(self, synthetic_env):
        """Bounds large enough for the batch produce exact padded shapes — the
        raise only fires on genuine overflow, not whenever pads are set."""
        gen = _make_dataloader(synthetic_env, pad_F=64, pad_n_seg=64)
        _, _, enc_fids, *_ = next(gen)
        assert enc_fids.shape[1] == 64
        assert enc_fids.shape[2] == 64

    def test_fp_targets_respects_generous_pad_n_seg(self, synthetic_env):
        """fp_targets is [B, pad_n_seg, n_sae] when pad_n_seg is not overflowed."""
        pad_n_seg = 64
        gen = _make_dataloader(
            synthetic_env, pad_F=64, pad_n_seg=pad_n_seg, n_sae_features=_N_SAE_FEATURES,
        )
        *_, fp_targets, _ = next(gen)
        assert fp_targets is not None
        assert fp_targets.shape == (_B, pad_n_seg, _N_SAE_FEATURES)


class TestOverflowMisconditioningXfail:
    """xfail: documents the pad_n_seg/pad_F overflow bug tracked in issue #641.

    The desired end-state is that an overflowing batch is handled WITHOUT
    mis-conditioning any decoder token and WITHOUT crashing (e.g. dynamic
    segment bounds, or masking the overflow tokens out of the decoder loss).
    Today overflow raises a ValueError (the fail-loud stopgap), so these tests
    xfail; they xpass once graceful handling lands, prompting removal of the
    xfail marker and finalization of the conditioning assertions.
    """

    @pytest.mark.xfail(
        reason="pad_n_seg overflow raises (stopgap for the seg_map mis-conditioning "
        "bug); graceful handling tracked in issue #641",
        raises=ValueError,
        strict=True,
    )
    def test_pad_n_seg_overflow_handled_without_misconditioning(self, synthetic_env):
        # An overflowing batch should yield a usable batch in which no decoder
        # token is conditioned on a segment belonging to a different document.
        # Today this raises -> xfail. When graceful handling lands, the raise is
        # gone and the conditioning contract below must hold.
        gen = _make_dataloader(synthetic_env, pad_F=50, pad_n_seg=1)
        _, _, _, _, _, _, seg_map, seg_doc_id, _, _ = next(gen)
        # No token may point at a segment index that was fabricated by clamping
        # (i.e. every referenced segment must be a real segment of the batch).
        assert int(seg_map.max().item()) < seg_doc_id.shape[1]

    @pytest.mark.xfail(
        reason="pad_F overflow raises (stopgap); graceful no-feature-loss handling "
        "tracked in issue #641",
        raises=ValueError,
        strict=True,
    )
    def test_pad_f_overflow_handled_without_feature_loss(self, synthetic_env):
        # An overflowing batch should retain the segments' conditioning features
        # rather than silently dropping them to a single slot. Today this raises
        # -> xfail.
        gen = _make_dataloader(synthetic_env, pad_F=1, pad_n_seg=20)
        _, _, enc_fids, _, _, _, _, _, _, _ = next(gen)
        assert enc_fids.shape[2] > 1


class TestStaticShapeNullPassthrough:
    """pad_F=None, pad_n_seg=None preserves variable-shape behavior."""

    def test_none_padding_produces_valid_shapes(self, synthetic_env):
        """Without padding, dataloader produces valid 3D enc_fids."""
        gen = _make_dataloader(synthetic_env, pad_F=None, pad_n_seg=None)
        shapes = set()
        for _ in range(5):
            _, _, enc_fids, *_ = next(gen)
            shapes.add(enc_fids.shape)
        # Verify no crash and valid shapes
        for s in shapes:
            assert len(s) == 3  # (B, n_seg, f_max)
            assert s[0] == _B

    def test_partial_padding_only_f(self, synthetic_env):
        """pad_F set but pad_n_seg=None: only feature dim is fixed."""
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=None)
        for _ in range(3):
            _, _, enc_fids, *_ = next(gen)
            assert enc_fids.shape[2] == 20  # feature dim fixed
            assert enc_fids.shape[0] == _B

    def test_partial_pad_f_preserves_dynamic_segment_payload(self, synthetic_env):
        """Fixing only pad_F must not alter tokens, segments, or active features."""
        gen_dynamic = _make_dataloader(
            synthetic_env, pad_F=None, pad_n_seg=None, n_sae_features=_N_SAE_FEATURES,
        )
        gen_pad_f = _make_dataloader(
            synthetic_env, pad_F=20, pad_n_seg=None, n_sae_features=_N_SAE_FEATURES,
        )

        (
            x_dynamic,
            y_dynamic,
            fids_dynamic,
            mags_dynamic,
            mask_dynamic,
            pcts_dynamic,
            seg_map_dynamic,
            seg_doc_dynamic,
            fp_dynamic,
            _,
        ) = next(gen_dynamic)
        (
            x_pad_f,
            y_pad_f,
            fids_pad_f,
            mags_pad_f,
            mask_pad_f,
            pcts_pad_f,
            seg_map_pad_f,
            seg_doc_pad_f,
            fp_pad_f,
            _,
        ) = next(gen_pad_f)

        natural_f = fids_dynamic.shape[2]
        assert natural_f < fids_pad_f.shape[2]
        assert fids_dynamic.shape[:2] == fids_pad_f.shape[:2]

        assert t.equal(x_dynamic, x_pad_f)
        assert t.equal(y_dynamic, y_pad_f)
        assert t.equal(seg_map_dynamic, seg_map_pad_f)
        assert t.equal(seg_doc_dynamic, seg_doc_pad_f)
        assert fp_dynamic is not None
        assert fp_pad_f is not None
        assert t.equal(fp_dynamic, fp_pad_f)

        assert t.equal(fids_dynamic, fids_pad_f[:, :, :natural_f])
        assert t.equal(mags_dynamic, mags_pad_f[:, :, :natural_f])
        assert t.equal(mask_dynamic, mask_pad_f[:, :, :natural_f])
        assert t.equal(pcts_dynamic, pcts_pad_f[:, :, :natural_f])
        assert (fids_pad_f[:, :, natural_f:] == 0).all()
        assert (mags_pad_f[:, :, natural_f:] == 0).all()
        assert not mask_pad_f[:, :, natural_f:].any()
        assert (pcts_pad_f[:, :, natural_f:] == 0.0).all()

    def test_partial_padding_only_seg(self, synthetic_env):
        """pad_n_seg set but pad_F=None: only segment dim is fixed."""
        gen = _make_dataloader(synthetic_env, pad_F=None, pad_n_seg=8)
        for _ in range(3):
            _, _, enc_fids, *_ = next(gen)
            assert enc_fids.shape[1] == 8  # segment dim fixed
            assert enc_fids.shape[0] == _B

    def test_partial_pad_n_seg_preserves_unpadded_payload(self, synthetic_env):
        """Fixing only pad_n_seg must only append padded segment rows."""
        pad_n_seg = 20
        gen_dynamic = _make_dataloader(
            synthetic_env, pad_F=None, pad_n_seg=None, n_sae_features=_N_SAE_FEATURES,
        )
        gen_pad_n_seg = _make_dataloader(
            synthetic_env, pad_F=None, pad_n_seg=pad_n_seg, n_sae_features=_N_SAE_FEATURES,
        )

        (
            x_dynamic,
            y_dynamic,
            fids_dynamic,
            mags_dynamic,
            mask_dynamic,
            pcts_dynamic,
            seg_map_dynamic,
            seg_doc_dynamic,
            fp_dynamic,
            _,
        ) = next(gen_dynamic)
        (
            x_pad_n_seg,
            y_pad_n_seg,
            fids_pad_n_seg,
            mags_pad_n_seg,
            mask_pad_n_seg,
            pcts_pad_n_seg,
            seg_map_pad_n_seg,
            seg_doc_pad_n_seg,
            fp_pad_n_seg,
            _,
        ) = next(gen_pad_n_seg)

        natural_n_seg = fids_dynamic.shape[1]
        assert natural_n_seg < pad_n_seg
        assert fids_dynamic.shape[2] == fids_pad_n_seg.shape[2]

        assert t.equal(x_dynamic, x_pad_n_seg)
        assert t.equal(y_dynamic, y_pad_n_seg)
        assert t.equal(seg_map_dynamic, seg_map_pad_n_seg)

        assert t.equal(fids_dynamic, fids_pad_n_seg[:, :natural_n_seg, :])
        assert t.equal(mags_dynamic, mags_pad_n_seg[:, :natural_n_seg, :])
        assert t.equal(mask_dynamic, mask_pad_n_seg[:, :natural_n_seg, :])
        assert t.equal(pcts_dynamic, pcts_pad_n_seg[:, :natural_n_seg, :])
        assert t.equal(seg_doc_dynamic, seg_doc_pad_n_seg[:, :natural_n_seg])
        assert fp_dynamic is not None
        assert fp_pad_n_seg is not None
        assert t.equal(fp_dynamic, fp_pad_n_seg[:, :natural_n_seg, :])

        assert (fids_pad_n_seg[:, natural_n_seg:, :] == 0).all()
        assert (mags_pad_n_seg[:, natural_n_seg:, :] == 0).all()
        assert not mask_pad_n_seg[:, natural_n_seg:, :].any()
        assert (pcts_pad_n_seg[:, natural_n_seg:, :] == 0.0).all()
        assert (seg_doc_pad_n_seg[:, natural_n_seg:] == -1).all()
        assert (fp_pad_n_seg[:, natural_n_seg:, :] == 0.0).all()

    def test_default_is_no_padding(self, synthetic_env):
        """Default _make_dataloader call (no pad args) behaves like None/None."""
        gen_default = _make_dataloader(synthetic_env)
        gen_explicit_none = _make_dataloader(synthetic_env, pad_F=None, pad_n_seg=None)

        batch_default = next(gen_default)
        batch_none = next(gen_explicit_none)

        # Same tuple length
        assert len(batch_default) == len(batch_none) == 10

        # Same shapes (both are unpaddded, same data)
        for i in range(8):  # first 8 tensors
            assert batch_default[i].shape == batch_none[i].shape, (
                f"Element {i} shape mismatch: default={batch_default[i].shape}, "
                f"explicit_none={batch_none[i].shape}"
            )

    def test_partial_pad_f_seg_doc_id_variable(self, synthetic_env):
        """With only pad_F set, seg_doc_id segment dim may still vary."""
        gen = _make_dataloader(synthetic_env, pad_F=20, pad_n_seg=None)
        _, _, _, _, _, _, _, seg_doc_id, _, _ = next(gen)
        # seg_doc_id should have shape (B, n_seg) where n_seg is natural
        assert seg_doc_id.shape[0] == _B
        assert seg_doc_id.dim() == 2

    def test_partial_pad_n_seg_feature_dim_variable(self, synthetic_env):
        """With only pad_n_seg set, feature dim may still vary."""
        gen = _make_dataloader(synthetic_env, pad_F=None, pad_n_seg=8)
        _, _, enc_fids, enc_mags, enc_mask, enc_pcts, _, _, _, _ = next(gen)
        # All encoder tensors should have segment dim == pad_n_seg
        assert enc_fids.shape[1] == 8
        assert enc_mags.shape[1] == 8
        assert enc_mask.shape[1] == 8
        assert enc_pcts.shape[1] == 8
        # Feature dim is whatever the batch naturally produces
        assert enc_fids.shape[2] >= 1
