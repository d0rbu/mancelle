"""Edge-case and assertion tests for extract_features.py pure functions.

Tests pool_segments_batched_gpu, extract_sparse_gpu, and extract_document_features
under boundary conditions not covered by the existing test suite.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.extract_features as extract_features_module
from steering.extract_features import (
    PoolingMode,
    extract_sparse_gpu,
    pool_segments_batched_gpu,
)
from steering.segmentation import SegmentMode


class TestPoolSegmentsBatchedGpuEdgeCases:
    """Edge cases for pool_segments_batched_gpu."""

    def test_single_token_segment(self):
        """Segment of exactly one token."""
        sae_acts = t.rand(10, 32)
        segments = [(5, 6)]  # single token
        for mode in PoolingMode:
            result = pool_segments_batched_gpu(sae_acts, segments, mode, SegmentMode.fixed_tokens)
            assert result.shape == (1, 32)
            assert t.allclose(result[0], sae_acts[5], atol=1e-6)

    def test_single_segment_covering_all_tokens(self):
        """One segment covering the entire sequence."""
        sae_acts = t.rand(64, 16)
        segments = [(0, 64)]
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        expected = sae_acts.mean(dim=0)
        assert t.allclose(result[0], expected, atol=1e-6)

    def test_fixed_tokens_short_last_segment_mean_pool(self):
        """Fixed tokens with a short last segment computes correct mean."""
        t.manual_seed(42)
        sae_acts = t.rand(10, 8)
        segments = [(0, 4), (4, 8), (8, 10)]  # last segment has 2 tokens
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.mean_pool, SegmentMode.fixed_tokens
        )
        # The last segment should be the mean of tokens 8 and 9
        expected_last = sae_acts[8:10].mean(dim=0)
        assert t.allclose(result[2], expected_last, atol=1e-5), (
            f"Short last segment mean mismatch: got {result[2][:3]}, expected {expected_last[:3]}"
        )

    def test_fixed_tokens_short_last_segment_max_pool(self):
        """Fixed tokens with a short last segment computes correct max."""
        t.manual_seed(42)
        sae_acts = t.rand(10, 8)
        segments = [(0, 4), (4, 8), (8, 10)]
        result = pool_segments_batched_gpu(
            sae_acts, segments, PoolingMode.max_pool, SegmentMode.fixed_tokens
        )
        expected_last = sae_acts[8:10].max(dim=0).values
        assert t.allclose(result[2], expected_last, atol=1e-5)

    def test_out_of_bounds_segment_asserts(self):
        """Segment extending beyond sae_acts should raise AssertionError."""
        sae_acts = t.rand(10, 8)
        segments = [(0, 5), (5, 15)]  # second segment out of bounds
        with pytest.raises(AssertionError, match="out of bounds"):
            pool_segments_batched_gpu(sae_acts, segments, PoolingMode.mean_pool)

    def test_empty_segment_asserts(self):
        """Segment with start == end should raise AssertionError."""
        sae_acts = t.rand(10, 8)
        segments = [(0, 5), (5, 5)]  # empty segment
        with pytest.raises(AssertionError, match="out of bounds"):
            pool_segments_batched_gpu(sae_acts, segments, PoolingMode.mean_pool)

    def test_1d_input_asserts(self):
        """1D input should raise AssertionError."""
        sae_acts = t.rand(10)
        with pytest.raises(AssertionError, match="must be 2D"):
            pool_segments_batched_gpu(sae_acts, [(0, 5)], PoolingMode.mean_pool)


class TestExtractSparseGpuEdgeCases:
    """Edge cases for extract_sparse_gpu."""

    def test_single_nonzero_element(self):
        """Single segment with exactly one non-zero feature."""
        pooled = t.zeros(1, 100)
        pooled[0, 42] = 3.14
        ids, mags = extract_sparse_gpu(pooled)
        assert len(ids) == 1
        assert ids[0].tolist() == [42]
        assert abs(mags[0][0].item() - 3.14) < 1e-5

    def test_output_count_matches_input_segments(self):
        """Output list lengths always equal input segment count."""
        for n_segs in [1, 5, 100]:
            pooled = t.rand(n_segs, 16)
            pooled[pooled < 0.5] = 0.0
            ids, mags = extract_sparse_gpu(pooled)
            assert len(ids) == n_segs, f"Expected {n_segs} id lists, got {len(ids)}"
            assert len(mags) == n_segs, f"Expected {n_segs} mag lists, got {len(mags)}"

    def test_1d_input_asserts(self):
        """1D input should raise AssertionError."""
        with pytest.raises(AssertionError, match="must be 2D"):
            extract_sparse_gpu(t.rand(10))

    def test_3d_input_asserts(self):
        """3D input should raise AssertionError."""
        with pytest.raises(AssertionError, match="must be 2D"):
            extract_sparse_gpu(t.rand(2, 3, 4))

    def test_magnitudes_sorted_descending(self):
        """Each segment's magnitudes should be sorted descending (CSR invariant)."""
        t.manual_seed(99)
        pooled = t.rand(10, 64)
        pooled[pooled < 0.3] = 0.0
        ids, mags = extract_sparse_gpu(pooled)
        for i in range(10):
            if len(mags[i]) > 1:
                for j in range(len(mags[i]) - 1):
                    assert mags[i][j] >= mags[i][j + 1], (
                        f"Segment {i}: mags not sorted descending at position {j}"
                    )

    def test_all_features_positive(self):
        """All returned magnitudes should be > 0."""
        t.manual_seed(7)
        pooled = t.rand(5, 32)
        pooled[pooled < 0.4] = 0.0
        ids, mags = extract_sparse_gpu(pooled)
        for i in range(5):
            if len(mags[i]) > 0:
                assert (mags[i] > 0).all(), f"Segment {i}: found non-positive magnitude"

    def test_large_sparse_input(self):
        """Large input with high sparsity doesn't crash or produce wrong counts."""
        t.manual_seed(0)
        pooled = t.rand(256, 1024)
        pooled[pooled < 0.95] = 0.0  # ~5% non-zero
        ids, mags = extract_sparse_gpu(pooled)
        assert len(ids) == 256
        total_nnz = sum(len(m) for m in mags)
        expected_nnz = (pooled > 0).sum().item()
        assert total_nnz == expected_nnz


class TestExtractDocumentFeaturesRouting:
    """Routing tests for single-document feature extraction."""

    def _patch_common_extract_document_helpers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        kept_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        kept_ids = kept_ids or [11, 12, 13]
        calls: dict[str, Any] = {
            "batched": [],
            "single": [],
        }

        monkeypatch.setattr(
            extract_features_module,
            "_truncate_to_training_window",
            lambda *args, **kwargs: (kept_ids, len(kept_ids)),
        )

        def fake_precompute(*args, **kwargs):
            del args, kwargs
            return [
                (
                    [(0, 2), (2, len(kept_ids))],
                    len(kept_ids),
                    [0, 2],
                    [2, len(kept_ids)],
                )
            ]

        def fake_pool(sae_acts, segments, pooling, segment_mode, *, keep_mask=None):
            calls["pooled_from"] = sae_acts
            calls["segments"] = segments
            calls["pooling"] = pooling
            calls["segment_mode"] = segment_mode
            calls["keep_mask"] = keep_mask
            return t.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 3.0],
                ]
            )

        def fake_extract_sparse(pooled):
            calls["sparse_from"] = pooled
            return (
                [
                    t.tensor([1], dtype=t.int32),
                    t.tensor([2, 0], dtype=t.int32),
                ],
                [
                    t.tensor([1.0], dtype=t.float32),
                    t.tensor([3.0, 2.0], dtype=t.float32),
                ],
            )

        monkeypatch.setattr(
            extract_features_module,
            "_precompute_batch_cpu_work",
            fake_precompute,
        )
        monkeypatch.setattr(
            extract_features_module,
            "pool_segments_batched_gpu",
            fake_pool,
        )
        monkeypatch.setattr(
            extract_features_module,
            "extract_sparse_gpu",
            fake_extract_sparse,
        )
        return calls

    def test_extract_document_features_uses_nnterp_batched_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._patch_common_extract_document_helpers(monkeypatch)

        def fake_batched(**kwargs):
            calls["batched"].append(kwargs)
            return [t.ones(3, 4)]

        def fake_single(*args, **kwargs):
            calls["single"].append((args, kwargs))
            raise AssertionError("single path should not be used")

        monkeypatch.setattr(
            extract_features_module,
            "_get_sae_activations_batched",
            fake_batched,
        )
        monkeypatch.setattr(
            extract_features_module,
            "_get_sae_activations",
            fake_single,
        )

        sae = t.nn.Linear(4, 4)
        nnterp_model = object()
        model = SimpleNamespace(
            cfg=SimpleNamespace(n_ctx=16),
            tokenizer=object(),
            nnterp_model=nnterp_model,
        )

        ids, mags, positions, char_starts, char_ends, n_tokens = (
            extract_features_module.extract_document_features(
                model,
                sae,
                "abc",
                layer=7,
                context_window=8,
                pooling=PoolingMode.max_pool,
                segment_size=2,
                hook_name="blocks.7.hook_resid_post",
            )
        )

        assert len(calls["batched"]) == 1
        assert calls["single"] == []
        batched_call = calls["batched"][0]
        assert batched_call["model"] is model
        assert batched_call["sae"] is sae
        assert batched_call["token_id_lists"] == [[11, 12, 13]]
        assert batched_call["layer"] == 7
        assert batched_call["nnterp_model"] is nnterp_model
        assert batched_call["hook_name"] == "blocks.7.hook_resid_post"
        assert calls["keep_mask"] is None
        assert [x.tolist() for x in ids] == [[1], [2, 0]]
        assert [x.tolist() for x in mags] == [[1.0], [3.0, 2.0]]
        assert positions == [0, 2]
        assert char_starts == [0, 2]
        assert char_ends == [2, 3]
        assert n_tokens == 3

    def test_context_independent_single_document_forwards_loaded_backends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._patch_common_extract_document_helpers(monkeypatch)
        monkeypatch.setattr(
            extract_features_module,
            "_prepare_context_independent_segment_inputs",
            lambda **_kwargs: [object(), object()],
        )

        def fake_context_independent(*args, **kwargs):
            calls["context_independent_args"] = args
            calls["context_independent_kwargs"] = kwargs
            return t.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])

        monkeypatch.setattr(
            extract_features_module,
            "_context_independent_forward_and_pool_batch",
            fake_context_independent,
        )
        native_model = object()
        nnterp_model = object()
        tokenizer = SimpleNamespace(bos_token_id=1)
        model = SimpleNamespace(
            cfg=SimpleNamespace(n_ctx=16, model_name="gemma-3-test"),
            tokenizer=tokenizer,
            native_model=native_model,
            nnterp_model=nnterp_model,
        )
        sae = t.nn.Linear(4, 4)

        extract_features_module.extract_document_features(
            model,
            sae,
            "abc",
            layer=22,
            context_window=8,
            segment_size=2,
            context_independent_features=True,
        )

        assert calls["context_independent_args"][:2] == (model, sae)
        assert calls["context_independent_kwargs"]["nnterp_model"] is nnterp_model
        assert calls["context_independent_kwargs"]["native_model"] is native_model

    def test_extract_document_features_uses_legacy_single_path_without_nnterp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._patch_common_extract_document_helpers(monkeypatch)

        def fake_batched(**kwargs):
            calls["batched"].append(kwargs)
            raise AssertionError("batched path should not be used")

        def fake_single(model, sae, tokens, layer, hook_name=None):
            calls["single"].append(
                {
                    "model": model,
                    "sae": sae,
                    "tokens": tokens,
                    "layer": layer,
                    "hook_name": hook_name,
                }
            )
            return t.ones(3, 4)

        monkeypatch.setattr(
            extract_features_module,
            "_get_sae_activations_batched",
            fake_batched,
        )
        monkeypatch.setattr(
            extract_features_module,
            "_get_sae_activations",
            fake_single,
        )

        sae = t.nn.Linear(4, 4)
        model = SimpleNamespace(cfg=SimpleNamespace(n_ctx=16), tokenizer=object())

        extract_features_module.extract_document_features(
            model,
            sae,
            "abc",
            layer=5,
            context_window=8,
            segment_size=2,
            hook_name="blocks.5.hook_resid_post",
        )

        assert calls["batched"] == []
        assert len(calls["single"]) == 1
        single_call = calls["single"][0]
        assert single_call["model"] is model
        assert single_call["sae"] is sae
        assert single_call["tokens"].tolist() == [11, 12, 13]
        assert single_call["layer"] == 5
        assert single_call["hook_name"] == "blocks.5.hook_resid_post"
        assert calls["keep_mask"] is None
