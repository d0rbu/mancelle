"""Tests for context-independent feature extraction.

Covers context-independent-only primitives (ContextIndependentSegmentInput, _prepare_context_independent_segment_inputs,
pool_one_segment, _context_independent_forward_and_pool_batch, _run_context_independent_chunk),
the process_shard fork, and process-wide invariants.

Run with:
    uv run pytest tests/test_context_independent_extraction.py -v --model-scale=tiny
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.extract_features as ef_mod
from steering.extract_features import (
    ContextIndependentSegmentInput,
    _context_independent_forward_and_pool_batch,
    _get_sae_activations,
    _prepare_context_independent_segment_inputs,
    _run_context_independent_chunk,
    _truncate_to_training_window,
    _warn_if_path_flag_mismatch,
    extract_document_features,
    find_optimal_batch_sizes,
    pool_one_segment,
    prefetch_shard_data,
    process_shard,
)
from steering.feature_io import glob_features, load_features
from steering.feature_paths import (
    FeatureDirectoryMismatchError,
    PoolingMode,
    resolve_output_dir,
    verify_feature_dir,
)
from steering.refined_types import PositiveInt
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode, segment_tokens
from steering.shard_validation import validate_shard
from steering.stats_io import save_statistics
from steering.validate_features import validate_shard_activations
from tests.reference_extraction_pipeline import ref_extract_document_context_independent
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE
from tests.steering.test_feature_dir_verification import _make_stats


class TestContextIndependentSegmentInput:
    def test_construction_and_fields(self):
        seg_input = ContextIndependentSegmentInput(
            doc_idx=3,
            seg_idx=1,
            model_input=[2, 7, 8, 9],
            bos_injected=True,
            original_length=3,
        )
        assert seg_input.doc_idx == 3
        assert seg_input.seg_idx == 1
        assert seg_input.model_input == [2, 7, 8, 9]
        assert seg_input.bos_injected is True
        assert seg_input.original_length == 3

    def test_frozen(self):
        seg_input = ContextIndependentSegmentInput(
            doc_idx=0, seg_idx=0, model_input=[2, 5],
            bos_injected=True, original_length=1,
        )
        with pytest.raises((AttributeError, TypeError)):
            seg_input.doc_idx = 99  # frozen dataclass rejects assignment  # ty: ignore[invalid-assignment]

    def test_length_invariant_with_bos_injected(self):
        seg_input = ContextIndependentSegmentInput(
            doc_idx=0, seg_idx=0, model_input=[2, 5, 6, 7],
            bos_injected=True, original_length=3,
        )
        assert len(seg_input.model_input) == seg_input.original_length + 1

    def test_length_invariant_without_bos_injected(self):
        seg_input = ContextIndependentSegmentInput(
            doc_idx=0, seg_idx=0, model_input=[2, 5, 6, 7],
            bos_injected=False, original_length=4,
        )
        assert len(seg_input.model_input) == seg_input.original_length


class TestPoolOneSegment:
    def test_max_pool(self):
        acts = t.tensor([[0.0, 1.0, 0.0], [2.0, 0.5, 0.0], [1.0, 0.0, 3.0]])
        out = pool_one_segment(acts, PoolingMode.max_pool)
        assert t.equal(out, t.tensor([2.0, 1.0, 3.0]))

    def test_mean_pool(self):
        acts = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = pool_one_segment(acts, PoolingMode.mean_pool)
        assert t.allclose(out, t.tensor([3.0, 4.0]))

    def test_last_token(self):
        acts = t.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = pool_one_segment(acts, PoolingMode.last_token)
        assert t.equal(out, t.tensor([5.0, 6.0]))

    def test_single_token_segment(self):
        acts = t.tensor([[7.0, 8.0, 9.0]])
        for mode in PoolingMode:
            out = pool_one_segment(acts, mode)
            assert t.equal(out, t.tensor([7.0, 8.0, 9.0])), f"failed for {mode}"

    def test_output_shape(self):
        acts = t.rand(10, 256)
        for mode in PoolingMode:
            out = pool_one_segment(acts, mode)
            assert out.shape == (256,)

    def test_empty_segment_raises(self):
        acts = t.zeros(0, 256)
        with pytest.raises((AssertionError, RuntimeError, IndexError)):
            pool_one_segment(acts, PoolingMode.max_pool)

    def test_2d_required(self):
        acts = t.rand(10)  # 1D
        with pytest.raises(AssertionError, match="2D"):
            pool_one_segment(acts, PoolingMode.max_pool)


class TestPrepareContextIndependentSegmentInputs:
    def test_pretraining_seg0_no_injection(self):
        # kept_ids starts with BOS=0. Segment (0,4) should NOT get injection.
        chunk_token_info = [([0, 5, 6, 7, 8], 5)]
        chunk_precomputed = [([(0, 4)], 5, [0, 0, 0, 0], [1, 2, 3, 4])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert len(result) == 1
        seg_input = result[0]
        assert seg_input.doc_idx == 0
        assert seg_input.seg_idx == 0
        assert seg_input.bos_injected is False
        assert seg_input.model_input == [0, 5, 6, 7]
        assert seg_input.original_length == 4

    def test_pretraining_seg1_injection(self):
        # Two segments: (0,2) and (2,4). First starts with BOS (no inject),
        # second does not (inject).
        chunk_token_info = [([0, 5, 6, 7], 4)]
        chunk_precomputed = [([(0, 2), (2, 4)], 4, [0, 0], [0, 0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert len(result) == 2
        assert result[0].bos_injected is False
        assert result[0].model_input == [0, 5]
        assert result[1].bos_injected is True
        assert result[1].model_input == [0, 6, 7]
        assert result[1].original_length == 2

    def test_sft_all_segments_injected(self):
        # Segment starts at token_offset=2; none of the segments start with BOS.
        chunk_token_info = [([0, 1, 2, 3, 4, 5], 6)]
        chunk_precomputed = [([(2, 4), (4, 6)], 6, [0, 0], [0, 0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert all(ci_input.bos_injected for ci_input in result)
        assert result[0].model_input == [0, 2, 3]
        assert result[1].model_input == [0, 4, 5]

    def test_ordering_preserved_across_docs(self):
        chunk_token_info = [
            ([0, 5, 6], 3),
            ([0, 7, 8, 9], 4),
        ]
        chunk_precomputed = [
            ([(0, 3)], 3, [0], [0]),
            ([(0, 2), (2, 4)], 4, [0, 0], [0, 0]),
        ]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert [(ci_input.doc_idx, ci_input.seg_idx) for ci_input in result] == [
            (0, 0), (1, 0), (1, 1)
        ]

    def test_empty_chunk(self):
        result = _prepare_context_independent_segment_inputs([], [], bos_token_id=0)
        assert result == []

    def test_zero_segment_doc_skipped(self):
        # SFT doc with fully-truncated response produces no segments.
        chunk_token_info = [([0, 1, 2], 3)]
        chunk_precomputed = [([], 3, [], [])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert result == []

    def test_mid_doc_bos_token_skips_injection(self):
        # If a segment happens to start with BOS mid-doc, no injection.
        chunk_token_info = [([0, 5, 0, 6], 4)]
        chunk_precomputed = [([(0, 2), (2, 4)], 4, [0, 0], [0, 0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=0
        )
        assert result[0].bos_injected is False
        assert result[1].bos_injected is False
        assert result[1].model_input == [0, 6]


def _tiny_model_sae(d_model: int = 64, n_features: int = 128, vocab: int = 256):
    model = TinyHookedTransformer(d_model=d_model, vocab_size=vocab, device="cpu")
    sae = TinySAE(d_model=d_model, n_features=n_features, device="cpu")
    return model, sae


class TestContextIndependentForwardAndPoolBatch:
    def test_output_shape(self):
        model, sae = _tiny_model_sae()
        batch = [
            ContextIndependentSegmentInput(0, 0, [0, 1, 2, 3], bos_injected=True, original_length=3),
            ContextIndependentSegmentInput(0, 1, [5, 6, 7], bos_injected=False, original_length=3),
        ]
        out = _context_independent_forward_and_pool_batch(
            model, sae, batch, layer=8,
            pooling=PoolingMode.max_pool,
            hook_name=None, activation_type=None,
            nnterp_model=None,
        )
        assert out.shape == (2, 128)

    def test_bos_drop_vs_no_drop(self):
        """A segment with BOS injected and a segment without should produce
        identical pooled outputs when the underlying real tokens are identical,
        because TinyHookedTransformer returns per-token embeddings with no
        positional encoding or attention — so the activations at the real-token
        positions are unaffected by the prefix."""
        model, sae = _tiny_model_sae()
        batch = [
            ContextIndependentSegmentInput(0, 0, [0, 5, 6], bos_injected=True, original_length=2),
            ContextIndependentSegmentInput(0, 1, [5, 6], bos_injected=False, original_length=2),
        ]
        out = _context_independent_forward_and_pool_batch(
            model, sae, batch, layer=8,
            pooling=PoolingMode.max_pool,
            hook_name=None, activation_type=None,
            nnterp_model=None,
        )
        assert t.allclose(out[0], out[1], atol=1e-5), (
            "BOS drop should produce identical pooled output as no-injection path "
            "for the same real segment tokens, under a positionless tiny model."
        )

    def test_matches_reference_per_segment(self):
        """Compare against one-at-a-time reference forward+pool."""
        model, sae = _tiny_model_sae()
        batch = [
            ContextIndependentSegmentInput(0, 0, [0, 3, 4, 5], bos_injected=True, original_length=3),
            ContextIndependentSegmentInput(0, 1, [0, 7, 8], bos_injected=True, original_length=2),
            ContextIndependentSegmentInput(1, 0, [9, 10, 11], bos_injected=False, original_length=3),
        ]
        batched = _context_independent_forward_and_pool_batch(
            model, sae, batch, layer=8,
            pooling=PoolingMode.mean_pool,
            hook_name=None, activation_type=None,
            nnterp_model=None,
        )
        # Reference: one-at-a-time
        ref_rows = []
        for ci_input in batch:
            tokens = t.tensor(ci_input.model_input, dtype=t.long)
            acts = _get_sae_activations(model, sae, tokens, layer=8)
            real = acts[1:] if ci_input.bos_injected else acts
            assert real.shape[0] == ci_input.original_length
            ref_rows.append(real.mean(dim=0))
        ref = t.stack(ref_rows, dim=0)
        assert t.allclose(batched, ref, atol=1e-5), (
            f"max diff {(batched - ref).abs().max().item()}"
        )

    def test_length_parity_assert_on_wrong_acts(self, monkeypatch):
        """If forward returns wrong-length activations, assert fires."""
        model, sae = _tiny_model_sae()

        def broken_forward(*args, **kwargs):
            # Return activations that are one row too long for each input.
            token_id_lists = args[2]
            return [t.rand(len(ids) + 1, 128) for ids in token_id_lists]

        monkeypatch.setattr(ef_mod, "_get_sae_activations_batched", broken_forward)
        batch = [ContextIndependentSegmentInput(0, 0, [0, 1], bos_injected=True, original_length=1)]
        with pytest.raises(AssertionError, match="Context-independent activation length"):
            _context_independent_forward_and_pool_batch(
                model, sae, batch, layer=8,
                pooling=PoolingMode.max_pool,
                hook_name=None, activation_type=None,
                nnterp_model=None,
            )


class TestRunContextIndependentChunk:
    def test_processes_all_segments(self):
        """Every segment gets processed, results reassembled in (doc, seg) order."""
        model, sae = _tiny_model_sae()
        # Two docs: first has 2 segments, second has 1.
        chunk_token_info = [
            ([0, 1, 2, 3, 4], 5),
            ([0, 5, 6], 3),
        ]
        chunk_precomputed = [
            ([(0, 3), (3, 5)], 5, [0, 10], [10, 20]),
            ([(0, 3)], 3, [0], [30]),
        ]
        all_feature_ids: list[t.Tensor] = []
        all_feature_mags: list[t.Tensor] = []
        doc_segment_info: list = []
        doc_token_counts: list[int] = []

        doc_counter = _run_context_independent_chunk(
            model=model, sae=sae,
            chunk_token_info=chunk_token_info,
            chunk_precomputed=chunk_precomputed,
            layer=8,
            pooling=PoolingMode.max_pool,
            gpu_batch_size=2,
            hook_name=None,
            activation_type=None,
            nnterp_model=None,
            bos_token_id=0,
            device=t.device("cpu"),
            gc_mem_threshold=0.99,
            timer=None,
            all_feature_ids=all_feature_ids,
            all_feature_mags=all_feature_mags,
            doc_segment_info=doc_segment_info,
            doc_token_counts=doc_token_counts,
            doc_counter=0,
        )
        assert doc_counter == 2
        assert len(all_feature_ids) == 3  # 3 total segments
        assert len(all_feature_mags) == 3
        assert len(doc_segment_info) == 2
        # doc 0: 2 segs, doc 1: 1 seg
        assert doc_segment_info[0][1] == 2  # n_segs for doc 0
        assert doc_segment_info[1][1] == 1  # n_segs for doc 1
        # Positions tracked correctly
        assert doc_segment_info[0][2] == [0, 3]
        assert doc_segment_info[1][2] == [0]
        assert doc_token_counts == [5, 3]

    def test_zero_segment_doc_handled(self):
        """SFT-truncated doc with no segments still increments doc_counter."""
        model, sae = _tiny_model_sae()
        chunk_token_info = [
            ([0, 1, 2], 3),
            ([0, 4, 5], 3),
        ]
        chunk_precomputed = [
            ([], 3, [], []),   # no segments
            ([(0, 3)], 3, [0], [30]),
        ]
        all_feature_ids, all_feature_mags = [], []
        doc_segment_info, doc_token_counts = [], []

        doc_counter = _run_context_independent_chunk(
            model=model, sae=sae,
            chunk_token_info=chunk_token_info,
            chunk_precomputed=chunk_precomputed,
            layer=8,
            pooling=PoolingMode.max_pool,
            gpu_batch_size=16,
            hook_name=None, activation_type=None, nnterp_model=None,
            bos_token_id=0, device=t.device("cpu"),
            gc_mem_threshold=0.99, timer=None,
            all_feature_ids=all_feature_ids,
            all_feature_mags=all_feature_mags,
            doc_segment_info=doc_segment_info,
            doc_token_counts=doc_token_counts,
            doc_counter=0,
        )
        assert doc_counter == 2
        assert len(all_feature_ids) == 1  # only 1 segment total
        assert doc_segment_info[0][1] == 0  # zero-seg doc
        assert doc_segment_info[1][1] == 1
        assert doc_token_counts == [3, 3]

    def test_empty_chunk(self):
        model, sae = _tiny_model_sae()
        all_feature_ids, all_feature_mags = [], []
        doc_segment_info, doc_token_counts = [], []
        doc_counter = _run_context_independent_chunk(
            model=model, sae=sae,
            chunk_token_info=[],
            chunk_precomputed=[],
            layer=8,
            pooling=PoolingMode.max_pool,
            gpu_batch_size=16,
            hook_name=None, activation_type=None, nnterp_model=None,
            bos_token_id=0, device=t.device("cpu"),
            gc_mem_threshold=0.99, timer=None,
            all_feature_ids=all_feature_ids,
            all_feature_mags=all_feature_mags,
            doc_segment_info=doc_segment_info,
            doc_token_counts=doc_token_counts,
            doc_counter=0,
        )
        assert doc_counter == 0
        assert all_feature_ids == []
        assert doc_segment_info == []

    def test_batch_size_invariance(self):
        """Running with gpu_batch_size=1 vs 16 produces identical results."""
        model, sae = _tiny_model_sae()
        chunk_token_info = [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], 9),
        ]
        chunk_precomputed = [
            ([(0, 3), (3, 6), (6, 9)], 9, [0, 10, 20], [10, 20, 30]),
        ]

        def run(gpu_bs):
            all_ids, all_mags = [], []
            dsi, dtc = [], []
            _run_context_independent_chunk(
                model=model, sae=sae,
                chunk_token_info=chunk_token_info,
                chunk_precomputed=chunk_precomputed,
                layer=8, pooling=PoolingMode.max_pool,
                gpu_batch_size=gpu_bs,
                hook_name=None, activation_type=None, nnterp_model=None,
                bos_token_id=0, device=t.device("cpu"),
                gc_mem_threshold=0.99, timer=None,
                all_feature_ids=all_ids,
                all_feature_mags=all_mags,
                doc_segment_info=dsi,
                doc_token_counts=dtc,
                doc_counter=0,
            )
            return all_ids, all_mags, dsi

        ids1, mags1, dsi1 = run(1)
        ids2, mags2, dsi2 = run(16)
        assert len(ids1) == len(ids2)
        for a, b in zip(ids1, ids2, strict=True):
            assert t.equal(a, b)
        for a, b in zip(mags1, mags2, strict=True):
            assert t.allclose(a, b, atol=1e-5)
        assert dsi1 == dsi2

    def test_doc_token_counts_aligned_when_mixed_zero_and_nonzero(self):
        """The accumulators stay len==doc_counter aligned across zero/nonzero-seg docs."""
        model, sae = _tiny_model_sae()
        chunk_token_info = [([0, 1, 2], 3), ([0, 4, 5], 3), ([0, 6, 7, 8], 4)]
        chunk_precomputed = [
            ([], 3, [], []),                            # zero-seg doc
            ([(0, 3)], 3, [0], [30]),                   # 1 segment
            ([(0, 2), (2, 4)], 4, [0, 20], [20, 40]),   # 2 segments
        ]
        all_feature_ids, all_feature_mags = [], []
        doc_segment_info, doc_token_counts = [], []

        doc_counter = _run_context_independent_chunk(
            model=model, sae=sae,
            chunk_token_info=chunk_token_info,
            chunk_precomputed=chunk_precomputed,
            layer=8, pooling=PoolingMode.max_pool, gpu_batch_size=16,
            hook_name=None, activation_type=None, nnterp_model=None,
            bos_token_id=0, device=t.device("cpu"),
            gc_mem_threshold=0.99, timer=None,
            all_feature_ids=all_feature_ids,
            all_feature_mags=all_feature_mags,
            doc_segment_info=doc_segment_info,
            doc_token_counts=doc_token_counts,
            doc_counter=0,
        )
        assert doc_counter == 3
        assert len(doc_token_counts) == doc_counter
        assert len(doc_segment_info) == doc_counter


class TestExtractDocumentFeaturesContextIndependent:
    def test_context_independent_returns_same_tuple_shape(self):
        model, sae = _tiny_model_sae()
        ids, mags, positions, cs, ce, n_tokens = extract_document_features(
            model, sae, "abcdefgh",
            layer=8, context_window=128, segment_size=4,
            pooling=PoolingMode.max_pool,
            context_independent_features=True,
        )
        assert isinstance(ids, list)
        assert len(ids) == len(mags) == len(positions) == len(cs) == len(ce)
        assert n_tokens > 0

    def test_context_independent_matches_reference(self):
        """context-independent mode matches the per-segment reference."""
        model, sae = _tiny_model_sae()
        text = "abcdefghij"
        ids, mags, _, _, _, _ = extract_document_features(
            model, sae, text,
            layer=8, context_window=128, segment_size=3,
            pooling=PoolingMode.max_pool,
            context_independent_features=True,
        )
        kept_ids, _ = _truncate_to_training_window(text, model.tokenizer, 128, 2048)
        segments = segment_tokens(
            kept_ids,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(3),
        )
        bos_id = model.tokenizer.bos_token_id
        ref_ids, ref_mags = [], []
        for start, end in segments:
            seg = kept_ids[start:end]
            bos_injected = seg[0] != bos_id
            model_input = ([bos_id] + seg) if bos_injected else seg
            tokens = t.tensor(model_input, dtype=t.long)
            acts = _get_sae_activations(model, sae, tokens, layer=8)
            real = acts[1:] if bos_injected else acts
            pooled = real.max(dim=0).values
            mask = pooled > 0
            fid = mask.nonzero(as_tuple=True)[0].to(t.int32)
            fmag = pooled[mask].to(t.float32)
            order = fmag.argsort(descending=True)
            ref_ids.append(fid[order])
            ref_mags.append(fmag[order])
        assert len(ids) == len(ref_ids)
        for a, b in zip(ids, ref_ids, strict=True):
            assert t.equal(a, b)
        for a, b in zip(mags, ref_mags, strict=True):
            assert t.allclose(a, b, atol=1e-5)

    def test_context_independent_raises_on_missing_bos(self):
        """Tokenizer without bos_token_id → ValueError."""
        model, sae = _tiny_model_sae()
        orig = model.tokenizer.bos_token_id
        model.tokenizer.bos_token_id = None
        try:
            with pytest.raises(ValueError, match="bos_token_id"):
                extract_document_features(
                    model, sae, "abc",
                    layer=8, context_window=128, segment_size=4,
                    context_independent_features=True,
                )
        finally:
            model.tokenizer.bos_token_id = orig

    def test_context_independent_raises_on_segment_too_big_for_ctx(self):
        model, sae = _tiny_model_sae()
        orig = model.cfg.n_ctx
        model.cfg.n_ctx = 4
        try:
            with pytest.raises(ValueError, match="exceeds model context"):
                extract_document_features(
                    model, sae, "abcdefghij",
                    layer=8, context_window=128, segment_size=4,
                    context_independent_features=True,
                )
        finally:
            model.cfg.n_ctx = orig


def _make_parquet(path: str, texts: list[str]):
    pq.write_table(pa.table({"text": texts}), path)


class TestProcessShardContextIndependent:
    def test_context_independent_flag_produces_valid_shard(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, [
            "hello world this is a document",
            "another document for testing",
        ])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4, cpu_batch_size=16,
            context_independent_features=True,
        )
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations
        data = load_features(result.output_path)
        assert data["context_independent_features"] is True

    def test_guard_fires_on_real_desync_context_independent(self, tmp_path, monkeypatch):
        """An induced accumulator desync on the context-independent path is caught
        end-to-end by _validate_doc_accounting (proves the guard is both reached on
        this path AND enforces the invariant, not just the monkeypatched-boom wiring)."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, [
            "hello world this is a document",
            "another document for testing",
        ])
        real_chunk = ef_mod._run_context_independent_chunk

        def desyncing_chunk(*args, **kwargs):
            new_doc_counter = real_chunk(*args, **kwargs)
            # Genuine desync: one extra doc_token_counts row with no doc_counter advance.
            kwargs["doc_token_counts"].append(999)
            return new_doc_counter

        monkeypatch.setattr(ef_mod, "_run_context_independent_chunk", desyncing_chunk)
        with pytest.raises(RuntimeError, match="doc accounting desync"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=4,
                context_independent_features=True,
            )

    def test_context_independent_false_produces_flag_false(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world document"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
        )
        data = load_features(result.output_path)
        assert data["context_independent_features"] is False

    def test_context_independent_segment_metadata_identical_to_cd(self, tmp_path):
        """doc_ids, segment_indices, segment_positions, char boundaries
        must be bit-identical between context-dependent and context-independent on the same input."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, [
            "first document for parity test",
            "second document shorter",
            "third",
        ])
        cd_dir = tmp_path / "cd"
        context_independent_dir = tmp_path / "context_independent"
        cd_dir.mkdir()
        context_independent_dir.mkdir()
        cd = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(cd_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
        )
        context_independent_result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(context_independent_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        cd_data = load_features(cd.output_path)
        context_independent_data = load_features(context_independent_result.output_path)
        for key in [
            "doc_ids", "segment_indices", "segment_positions",
            "segment_char_starts", "segment_char_ends", "doc_token_counts",
        ]:
            assert t.equal(cd_data[key], context_independent_data[key]), (
                f"metadata key {key} differs between context-dependent and context-independent"
            )
        assert cd_data["n_segments"] == context_independent_data["n_segments"]
        assert cd_data["n_docs"] == context_independent_data["n_docs"]

    def test_context_independent_pipeline_post_false_matches_true(self, tmp_path):
        """pipeline_post=False and pipeline_post=True produce identical context-independent shards."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, [
            "some doc a",
            "some doc b",
            "some doc c",
        ])
        out_t = tmp_path / "t"
        out_t.mkdir()
        out_f = tmp_path / "f"
        out_f.mkdir()
        rt = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(out_t),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2,
            context_independent_features=True,
            pipeline_post=True,
        )
        rf = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(out_f),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2,
            context_independent_features=True,
            pipeline_post=False,
        )
        dt = load_features(rt.output_path)
        df = load_features(rf.output_path)
        assert t.equal(dt["feature_ids"], df["feature_ids"])
        assert t.allclose(dt["feature_mags"], df["feature_mags"], atol=1e-5)


class TestFindOptimalBatchSizesContextIndependent:
    def test_context_independent_auto_tuner_runs(self, tmp_path):
        """Auto-tuner runs end-to-end in context-independent mode without crashing."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["sample " * 20] * 8)
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model, sae, layer=8, context_window=128,
            nnterp_model=None,
            max_gpu_batch_size=4,
            max_cpu_batch_size=8,
            min_gpu_batch_size=1,
            min_cpu_batch_size=2,
            n_trials=1, n_docs=4,
            pooling=PoolingMode.max_pool,
            segment_size=8,
            gc_mem_threshold=0.99,
            hook_name=None,
            activation_type=ActivationType.residual,
            parquet_paths=[parquet_path],
            context_independent_features=True,
        )
        assert gpu_bs >= 1 and cpu_bs >= 1


class TestPathFlagMismatchWarning:
    def _capture_warnings(self):
        """Capture loguru warnings into a list; returns (capture, sink_id)."""
        from loguru import logger
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        return messages, sink_id

    def test_context_independent_flag_path_matches_silent(self):
        from loguru import logger
        messages, sink_id = self._capture_warnings()
        try:
            _warn_if_path_flag_mismatch("/root/out/base/context_independent", True)
        finally:
            logger.remove(sink_id)
        assert not any("context_independent" in m for m in messages)

    def test_context_independent_flag_path_missing_warns(self):
        from loguru import logger
        messages, sink_id = self._capture_warnings()
        try:
            _warn_if_path_flag_mismatch("/root/out/base", True)
        finally:
            logger.remove(sink_id)
        assert any("context_independent" in m for m in messages)

    def test_no_flag_path_has_context_independent_warns(self):
        from loguru import logger
        messages, sink_id = self._capture_warnings()
        try:
            _warn_if_path_flag_mismatch("/root/out/base/context_independent", False)
        finally:
            logger.remove(sink_id)
        assert any("context_independent" in m for m in messages)

    def test_no_flag_no_context_independent_silent(self):
        from loguru import logger
        messages, sink_id = self._capture_warnings()
        try:
            _warn_if_path_flag_mismatch("/root/out/base", False)
        finally:
            logger.remove(sink_id)
        assert not any("context_independent" in m for m in messages)


class TestValidateShardActivationsContextIndependent:
    def test_context_independent_reextraction_matches(self, tmp_path):
        """Spot-check re-extraction in context-independent mode matches stored features."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["abcdefgh", "ijklmnop"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2,
            context_independent_features=True,
        )
        shard_data = load_features(result.output_path)
        vr = validate_shard_activations(
            shard_data,
            parquet_path=parquet_path,
            model=model, sae=sae,
            layer=shard_data["layer"],
            context_window=shard_data["context_window"],
            segment_size=shard_data["segment_size"],
            pooling=shard_data["pooling"],
            n_spot_check=2,
        )
        assert vr.valid, vr.violations


class TestContextIndependentVsContextDependentBitIdentity:
    def test_segment0_pretraining_bit_identical(self, tmp_path):
        """Invariant 5: segment 0 of a pretraining doc has bit-identical
        features between context-dependent and context-independent under a positionless tiny model."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world this is a longer document"])
        cd_dir = tmp_path / "cd"
        cd_dir.mkdir()
        context_independent_dir = tmp_path / "context_independent"
        context_independent_dir.mkdir()
        cd = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(cd_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=5, gpu_batch_size=4,
        )
        context_independent_result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(context_independent_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=5, gpu_batch_size=4,
            context_independent_features=True,
        )
        cd_data = load_features(cd.output_path)
        context_independent_data = load_features(context_independent_result.output_path)
        # Segment 0 is the first row in CSR for doc 0.
        cd_s0_ids = cd_data["feature_ids"][
            cd_data["offsets"][0]:cd_data["offsets"][1]
        ]
        cd_s0_mags = cd_data["feature_mags"][
            cd_data["offsets"][0]:cd_data["offsets"][1]
        ]
        context_independent_s0_ids = context_independent_data["feature_ids"][
            context_independent_data["offsets"][0]:context_independent_data["offsets"][1]
        ]
        context_independent_s0_mags = context_independent_data["feature_mags"][
            context_independent_data["offsets"][0]:context_independent_data["offsets"][1]
        ]
        # Both shards store ids ASCENDING within segment and mags in the
        # corresponding order (see feature_io._sort_within_segments).
        # Direct equality works.
        assert t.equal(cd_s0_ids, context_independent_s0_ids), (
            f"Segment 0 IDs differ: cd={cd_s0_ids.tolist()}, "
            f"context_independent={context_independent_s0_ids.tolist()}"
        )
        assert t.allclose(cd_s0_mags, context_independent_s0_mags, atol=1e-5), (
            f"Segment 0 mags differ: max diff "
            f"{(cd_s0_mags - context_independent_s0_mags).abs().max().item()}"
        )

    def test_degenerate_segment_size_eq_context_window(self, tmp_path):
        """When segment_size == context_window, context-independent and context-dependent should be
        bit-identical (single segment covers the whole doc)."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["short"])
        cd_dir = tmp_path / "cd"
        cd_dir.mkdir()
        context_independent_dir = tmp_path / "context_independent"
        context_independent_dir.mkdir()
        cd = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(cd_dir),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=64, gpu_batch_size=4,
        )
        context_independent_result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(context_independent_dir),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=64, gpu_batch_size=4,
            context_independent_features=True,
        )
        cd_data = load_features(cd.output_path)
        context_independent_data = load_features(context_independent_result.output_path)
        assert t.equal(cd_data["feature_ids"], context_independent_data["feature_ids"])
        assert t.allclose(cd_data["feature_mags"], context_independent_data["feature_mags"], atol=1e-5)


class TestContextIndependentMatchesReference:
    def test_context_independent_production_matches_reference(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        texts = ["abcdefghij", "mnopqrstuv", "short"]
        _make_parquet(parquet_path, texts)
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=3, gpu_batch_size=4,
            context_independent_features=True,
        )
        prod = load_features(result.output_path)

        # Build reference shard one doc at a time.
        all_ids_ref = []
        all_mags_ref = []
        for text in texts:
            kept_ids, _ = _truncate_to_training_window(
                text, model.tokenizer, 128, model.cfg.n_ctx,
            )
            segments = segment_tokens(
                kept_ids,
                SegmentMode.fixed_tokens,
                segment_size=PositiveInt.parse(3),
            )
            ref_ids, ref_mags = ref_extract_document_context_independent(
                model, sae, kept_ids, segments,
                pooling=PoolingMode.max_pool,
                bos_token_id=model.tokenizer.bos_token_id,
                layer=8,
            )
            for fid, fmag in zip(ref_ids, ref_mags, strict=True):
                all_ids_ref.append(fid)
                all_mags_ref.append(fmag)

        # Compare CSR content per segment (order should match).
        assert prod["n_segments"] == len(all_ids_ref), (
            f"n_segments mismatch: prod={prod['n_segments']}, "
            f"ref={len(all_ids_ref)}"
        )
        for seg_idx, (fid_ref, fmag_ref) in enumerate(
            zip(all_ids_ref, all_mags_ref, strict=True),
        ):
            s = int(prod["offsets"][seg_idx].item())
            e = int(prod["offsets"][seg_idx + 1].item())
            fid_prod = prod["feature_ids"][s:e]
            fmag_prod = prod["feature_mags"][s:e]
            # Production stores ids ASC within segment; reference sorts by mag DESC.
            # Re-sort both by id for comparison.
            ref_order = fid_ref.argsort()
            prod_order = fid_prod.argsort()
            assert t.equal(fid_ref[ref_order], fid_prod[prod_order]), (
                f"Segment {seg_idx} IDs differ: "
                f"ref={fid_ref[ref_order].tolist()}, "
                f"prod={fid_prod[prod_order].tolist()}"
            )
            assert t.allclose(
                fmag_ref[ref_order], fmag_prod[prod_order], atol=2e-4,
            ), f"Segment {seg_idx} mags differ"

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_context_independent_all_pooling_modes(self, tmp_path, pooling):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["some test text here for pooling"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=pooling,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations


class TestContextIndependentIntegration:
    def test_auto_resolved_path_has_context_independent_leaf(self, tmp_path):
        d = resolve_output_dir(
            output_dir=None,
            feature_root=str(tmp_path),
            segment_size=4,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
            context_independent_features=True,
        )
        assert d.parts[-1] == "context_independent"

    def test_context_independent_and_cd_do_not_collide(self, tmp_path):
        """Running both modes with auto-resolved paths lands in different dirs."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world"])

        cd_dir = resolve_output_dir(
            output_dir=None,
            feature_root=str(tmp_path / "features"),
            segment_size=4, layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        context_independent_dir = resolve_output_dir(
            output_dir=None,
            feature_root=str(tmp_path / "features"),
            segment_size=4, layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
            context_independent_features=True,
        )
        cd_dir.mkdir(parents=True)
        context_independent_dir.mkdir(parents=True)
        process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(cd_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
        )
        process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(context_independent_dir),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        cd_shards = glob_features(cd_dir)
        context_independent_shards = glob_features(context_independent_dir)
        assert len(cd_shards) > 0
        assert len(context_independent_shards) > 0
        # Same filename, but different parent dirs.
        assert {p.name for p in cd_shards} == {p.name for p in context_independent_shards}
        assert cd_shards[0].parent != context_independent_shards[0].parent


class TestContextIndependentPipelining:
    def test_context_independent_with_prefetched_shard(self, tmp_path):
        """process_shard in context-independent mode accepts prefetched data correctly."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["abcdefghij"] * 4)
        prefetched = prefetch_shard_data(parquet_path, cpu_batch_size=2, dataset_phase=None)
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2, cpu_batch_size=2,
            context_independent_features=True,
            prefetched=prefetched,
        )
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations

    def test_context_independent_gpu_batch_size_invariance(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["abcdefghij" * 2, "klmnop"])

        def run(bs):
            d = tmp_path / f"bs{bs}"
            d.mkdir()
            r = process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(d),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=bs,
                context_independent_features=True,
            )
            return load_features(r.output_path)

        d1 = run(1)
        d16 = run(16)
        assert t.equal(d1["feature_ids"], d16["feature_ids"])
        # Allow modest float32 batch-GEMM drift (see Task 25 note).
        assert t.allclose(d1["feature_mags"], d16["feature_mags"], atol=2e-4)


class TestContextIndependentLoudFailures:
    def test_missing_bos_token_raises(self, tmp_path):
        model, sae = _tiny_model_sae()
        model.tokenizer.bos_token_id = None
        # The tokenizer is a MagicMock; null the second BOS contract
        # (``get_bos_token_id``) too so ``resolve_bos_token_id`` sees a
        # genuinely BOS-less tokenizer (a real HF tokenizer has no such
        # method) instead of an auto-vivified phantom Mock attribute.
        model.tokenizer.get_bos_token_id = None
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello"])
        with pytest.raises(ValueError, match="bos_token_id"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=2,
                context_independent_features=True,
            )

    def test_segment_size_plus_one_exceeds_ctx_raises(self, tmp_path):
        model, sae = _tiny_model_sae()
        model.cfg.n_ctx = 4  # tiny, so segment_size+1 > 4 overflows
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello"])
        with pytest.raises(ValueError, match="exceeds model context"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=2,
                context_independent_features=True,
            )

    def test_pretraining_seg0_missing_bos_raises(self, tmp_path):
        """Force tokenizer to produce kept_ids that don't start with BOS
        (simulated broken config) → loud assertion."""
        model, sae = _tiny_model_sae()

        def broken_encode(text: str) -> list[int]:
            return [max(1, ord(c) % 256) for c in text] if text else []

        model.tokenizer.encode = broken_encode

        def broken_call(arg, **kw):
            if isinstance(arg, str):
                return {"input_ids": broken_encode(arg)}
            return {"input_ids": [broken_encode(a) for a in arg]}

        model.tokenizer.side_effect = broken_call

        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello"])
        with pytest.raises(AssertionError, match="kept_ids"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=2,
                context_independent_features=True,
                dataset_phase=None,
            )

    def test_verify_feature_dir_context_independent_metadata_path_mismatch_raises(self, tmp_path):
        """Metadata says context_independent=True but path lacks /context_independent → raise."""
        d = (
            tmp_path
            / "s128" / "l8" / "a_res" / "p_max_pool" / "f16k"
            / "m_gemma-2-2b" / "base"
        )
        d.mkdir(parents=True)
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello"])
        process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(d),
            context_window=1024, pooling=PoolingMode.max_pool,
            segment_size=128, gpu_batch_size=2,
            context_independent_features=True,
        )
        save_statistics(
            _make_stats(segment_size=128, layer=8, pooling_mode="max_pool",
                        n_sae_features=16384),
            d / "feature_statistics",
        )
        with pytest.raises(FeatureDirectoryMismatchError, match="context_independent"):
            verify_feature_dir(d)


class TestCDBackwardCompat:
    def test_cd_extraction_bit_identical_to_golden(self, tmp_path):
        """Extraction with context_independent_features=False on the fixture parquet produces a shard
        whose decoded content matches the committed golden cd shard."""
        fixture_dir = Path("tests/fixtures/cd_golden_shard")
        parquet_path = str(fixture_dir / "source.parquet")
        t.manual_seed(20260422)
        model = TinyHookedTransformer(d_model=64, vocab_size=256, device="cpu")
        sae = TinySAE(d_model=64, n_features=128, device="cpu")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = process_shard(
            parquet_path, model, sae,  # ty: ignore[invalid-argument-type]
            layer=8, output_dir=str(out_dir),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2, cpu_batch_size=4,
            context_independent_features=False,
        )
        regenerated = load_features(result.output_path)
        golden_path = fixture_dir / "source_features.safetensors.zst"
        golden = load_features(str(golden_path))

        # Compare all tensor fields.
        for key in [
            "feature_ids",
            "offsets",
            "doc_ids", "segment_indices", "segment_positions",
            "segment_char_starts", "segment_char_ends", "doc_token_counts",
        ]:
            assert t.equal(regenerated[key], golden[key]), (
                f"cd regression: tensor key {key} differs"
            )
        # Magnitudes: float16 on disk; close-but-not-exact tolerance acceptable.
        assert t.allclose(
            regenerated["feature_mags"], golden["feature_mags"], atol=1e-5,
        ), "cd regression: feature_mags differ"
        # Metadata:
        assert regenerated["context_independent_features"] is False
        assert golden["context_independent_features"] is False
        for meta_key in ["layer", "context_window", "pooling",
                         "segment_mode", "segment_size", "n_docs",
                         "n_segments"]:
            assert regenerated[meta_key] == golden[meta_key], (
                f"cd regression: metadata {meta_key} differs: "
                f"regenerated={regenerated[meta_key]}, golden={golden[meta_key]}"
            )


# ---------------------------------------------------------------------------
# TestCIStrengthening — robustness and invariant verification (Task 40-48)
# ---------------------------------------------------------------------------


def _make_aco_parquet(path: str, texts: list[str], offsets: list[int]):
    """Parquet with text plus a legacy extra assistant_char_offset column."""
    pq.write_table(
        pa.table({"text": texts, "assistant_char_offset": offsets}),
        path,
    )


class TestLegacySFTSchemaRejected:
    """Extraction rejects removed assistant-offset parquet columns in every mode."""

    def test_context_dependent_rejects_legacy_assistant_offset(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "legacy.parquet")
        _make_aco_parquet(parquet_path, ["P" * 10 + "R" * 40], [10])

        with pytest.raises(ValueError, match="removed SFT legacy columns"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=100, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=4,
                dataset_phase=None,
            )

    def test_context_independent_rejects_legacy_assistant_offset(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "legacy.parquet")
        _make_aco_parquet(parquet_path, ["P" * 10 + "R" * 40], [10])

        with pytest.raises(ValueError, match="removed SFT legacy columns"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=100, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=4,
                context_independent_features=True,
                dataset_phase=None,
            )


class TestContextIndependentMultiChunk:
    """Multi-chunk processing (cpu_batch_size < n_docs) (Task 41)."""

    def test_context_independent_multi_chunk_doc_counter_continues(self, tmp_path):
        """doc_counter must increment correctly across chunk boundaries."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        texts = [f"doc number {i}" * 3 for i in range(7)]
        _make_parquet(parquet_path, texts)
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2, cpu_batch_size=2,
            context_independent_features=True,
        )
        data = load_features(result.output_path)
        assert data["n_docs"] == 7
        # doc_ids must span 0..6 contiguously
        unique_docs = set(data["doc_ids"].tolist())
        assert unique_docs == set(range(7))
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations

    def test_context_independent_multi_chunk_equivalent_to_single_chunk(self, tmp_path):
        """Same input, cpu_batch_size=2 vs cpu_batch_size=100 → identical."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        texts = [f"doc {i} content here" for i in range(5)]
        _make_parquet(parquet_path, texts)
        small_dir = tmp_path / "small"
        small_dir.mkdir()
        big_dir = tmp_path / "big"
        big_dir.mkdir()
        rs = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(small_dir),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2, cpu_batch_size=2,
            context_independent_features=True,
        )
        rb = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(big_dir),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2, cpu_batch_size=100,
            context_independent_features=True,
        )
        ds = load_features(rs.output_path)
        db = load_features(rb.output_path)
        assert t.equal(ds["feature_ids"], db["feature_ids"])
        assert t.allclose(ds["feature_mags"], db["feature_mags"], atol=2e-4)
        assert t.equal(ds["doc_ids"], db["doc_ids"])
        assert t.equal(ds["segment_positions"], db["segment_positions"])


class TestContextIndependentEmptyParquet:
    """Empty-parquet edge case in context-independent mode (Task 42)."""

    def test_context_independent_empty_parquet_produces_valid_empty_shard(self, tmp_path):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "empty.parquet")
        _make_parquet(parquet_path, [])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2,
            context_independent_features=True,
        )
        assert result.n_docs == 0
        assert result.n_total_features == 0
        data = load_features(result.output_path)
        assert data["n_docs"] == 0
        assert data["n_segments"] == 0
        assert data["context_independent_features"] is True


class TestContextIndependentBOSMetadataLeakage:
    """Verify injected BOS does NOT leak into shard metadata (Task 44)."""

    def test_context_independent_segment_positions_ignore_injected_bos(self, tmp_path):
        """segment_positions records the ORIGINAL segment start in kept_ids,
        not adjusted for BOS injection."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        # Multi-segment doc so we have both segment 0 (no inject) and
        # segments 1+ (inject).
        _make_parquet(parquet_path, ["this is a longer document text here"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        data = load_features(result.output_path)
        positions = data["segment_positions"].tolist()
        # Positions are indices into kept_ids. First segment starts at 0.
        assert positions[0] == 0
        # Subsequent positions spaced by segment_size (fixed_tokens mode).
        if len(positions) > 1:
            for i in range(1, len(positions)):
                assert positions[i] == positions[i - 1] + 4, (
                    f"seg {i} position {positions[i]} != prev {positions[i-1]} + 4; "
                    f"possibly shifted by injected BOS"
                )

    def test_context_independent_doc_token_counts_ignore_injected_bos(self, tmp_path):
        """doc_token_counts reports the tokenizer's raw token count,
        not inflated by BOS injection across segments."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        text = "ABCDEFGHIJ"  # 10 chars + BOS = 11 tokens
        _make_parquet(parquet_path, [text])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        data = load_features(result.output_path)
        # n_tokens should be the tokenizer's encoded length (11) — not
        # inflated by per-segment BOS injection.
        assert data["doc_token_counts"][0].item() == 11

    def test_context_independent_char_starts_ignore_injected_bos(self, tmp_path):
        """segment_char_starts reports char offsets in the DECODED kept-id
        text, not affected by per-segment BOS injection."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["ABCDEFGHIJKLMN"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=4,
            context_independent_features=True,
        )
        data = load_features(result.output_path)
        # Char offsets should span the doc; first segment's char_start is 0.
        assert data["segment_char_starts"][0].item() == 0
        # char_starts are monotone non-decreasing.
        char_starts = data["segment_char_starts"].tolist()
        for i in range(1, len(char_starts)):
            assert char_starts[i] >= char_starts[i - 1]


class TestContextIndependentVariedBosTokenId:
    """Verify BOS injection rule holds for non-zero bos_token_id (Task 45)."""

    def test_prepare_context_independent_with_bos_id_1(self):
        """Rule: inject iff seg[0] != bos_id. Test with bos_id=1."""
        chunk_token_info = [([1, 5, 6, 7], 4)]
        chunk_precomputed = [([(0, 2), (2, 4)], 4, [0, 0], [0, 0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=1,
        )
        # seg 0: [1, 5] starts with bos_id=1 → no inject
        # seg 1: [6, 7] no bos → inject [1, 6, 7]
        assert result[0].bos_injected is False
        assert result[0].model_input == [1, 5]
        assert result[1].bos_injected is True
        assert result[1].model_input == [1, 6, 7]

    def test_prepare_context_independent_with_bos_id_255(self):
        """Edge: large bos_id at vocab boundary."""
        chunk_token_info = [([255, 5, 6], 3)]
        chunk_precomputed = [([(0, 3)], 3, [0], [0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=255,
        )
        assert result[0].bos_injected is False
        assert result[0].model_input == [255, 5, 6]

    def test_prepare_context_independent_with_mid_doc_match_of_varied_bos(self):
        """Rule only looks at seg[0]; mid-segment bos-matched tokens don't count."""
        chunk_token_info = [([42, 5, 42, 7], 4)]
        chunk_precomputed = [([(0, 2), (2, 4)], 4, [0, 0], [0, 0])]
        result = _prepare_context_independent_segment_inputs(
            chunk_token_info, chunk_precomputed, bos_token_id=42,
        )
        # seg 0: [42, 5] starts with bos_id → no inject
        # seg 1: [42, 7] also starts with bos_id → no inject (not off-by-one)
        assert result[0].bos_injected is False
        assert result[1].bos_injected is False
        assert result[1].model_input == [42, 7]


class TestContextIndependentDownstreamStatistics:
    """Verify compute_statistics runs end-to-end on context-independent shards (Task 46)."""

    def test_compute_statistics_on_context_independent_shards(self, tmp_path):
        from steering.compute_statistics import compute_statistics

        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["doc one", "doc two", "doc three"])
        process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=64, pooling=PoolingMode.max_pool,
            segment_size=4, gpu_batch_size=2,
            context_independent_features=True,
        )
        # Must not raise; must produce valid statistics.
        stats = compute_statistics(str(tmp_path), str(tmp_path / "stats"))
        assert stats.corpus.n_docs == 3
        assert stats.metadata.pooling_mode == "max_pool"
        # Statistics should aggregate context-independent shards without special handling.
        assert stats.features.n_segments > 0


class TestContextIndependentSingleTokenDoc:
    """Doc with exactly 1 SAE token → 1-token segment with BOS injection (Task 47)."""

    @pytest.mark.parametrize("pooling", list(PoolingMode))
    def test_context_independent_single_token_doc_all_pooling_modes(self, tmp_path, pooling):
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        # "a" → encoded as [BOS, 97], truncation keeps all 2 if context_window >= 2.
        # segment_size=8 > len(kept_ids)=2 → single segment (0, 2).
        _make_parquet(parquet_path, ["a"])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=16, pooling=pooling,
            segment_size=8, gpu_batch_size=2,
            context_independent_features=True,
        )
        data = load_features(result.output_path)
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations
        assert data["n_docs"] == 1
        assert data["n_segments"] == 1


class TestContextIndependentDegenerateParams:
    """Loud-failure behavior for malformed/unusual parameter values (Task 43)."""

    def test_context_independent_gpu_batch_size_zero_raises(self, tmp_path):
        """gpu_batch_size=0 must error (infinite-loop safety)."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world"])
        with pytest.raises((ValueError, ZeroDivisionError), match=r"(?i)batch|zero|step"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=64, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=0,
                context_independent_features=True,
            )

    def test_context_independent_gpu_batch_size_negative_raises(self, tmp_path):
        """gpu_batch_size<0 must error."""
        model, sae = _tiny_model_sae()
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world"])
        # range(0, N, -1) is empty → no forward passes → reassembly assertion fires.
        # Any loud failure is acceptable; silent success is not.
        with pytest.raises((AssertionError, ValueError)):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=64, pooling=PoolingMode.max_pool,
                segment_size=4, gpu_batch_size=-1,
                context_independent_features=True,
            )

    def test_context_independent_segment_size_equals_n_ctx_raises(self, tmp_path):
        """segment_size + 1 > n_ctx fires the precondition check exactly at the boundary."""
        model, sae = _tiny_model_sae()
        # Force n_ctx to be segment_size exactly → segment_size+1 overflows.
        model.cfg.n_ctx = 16
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["hello world"])
        with pytest.raises(ValueError, match="exceeds model context"):
            process_shard(
                parquet_path, model, sae,
                layer=8, output_dir=str(tmp_path),
                context_window=128, pooling=PoolingMode.max_pool,
                segment_size=16, gpu_batch_size=4,
                context_independent_features=True,
            )

    def test_context_independent_segment_size_one_less_than_n_ctx_ok(self, tmp_path):
        """segment_size + 1 == n_ctx is the tight-but-legal boundary."""
        model, sae = _tiny_model_sae()
        model.cfg.n_ctx = 16
        parquet_path = str(tmp_path / "shard.parquet")
        _make_parquet(parquet_path, ["x" * 30])
        result = process_shard(
            parquet_path, model, sae,
            layer=8, output_dir=str(tmp_path),
            context_window=128, pooling=PoolingMode.max_pool,
            segment_size=15, gpu_batch_size=4,
            context_independent_features=True,
        )
        vr = validate_shard(result.output_path)
        assert vr.valid, vr.violations

    def test_context_independent_empty_chunk_token_info_handled(self):
        """_run_context_independent_chunk with empty chunk → doc_counter unchanged."""
        model, sae = _tiny_model_sae()
        all_feature_ids, all_feature_mags = [], []
        doc_segment_info, doc_token_counts = [], []
        initial = 42
        result = _run_context_independent_chunk(
            model=model, sae=sae,
            chunk_token_info=[],
            chunk_precomputed=[],
            layer=8, pooling=PoolingMode.max_pool,
            gpu_batch_size=16,
            hook_name=None, activation_type=None, nnterp_model=None,
            bos_token_id=0, device=t.device("cpu"),
            gc_mem_threshold=0.99, timer=None,
            all_feature_ids=all_feature_ids,
            all_feature_mags=all_feature_mags,
            doc_segment_info=doc_segment_info,
            doc_token_counts=doc_token_counts,
            doc_counter=initial,
        )
        assert result == initial  # counter unchanged
        assert all_feature_ids == []
        assert doc_segment_info == []
        assert doc_token_counts == []
