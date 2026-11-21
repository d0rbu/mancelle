"""Tests for CPU post-processing pipelining in process_shard.

Tests:
  - _precompute_batch_cpu_work returns correct structure and matches direct calls
  - Pipelined process_shard produces identical output to sequential mode
  - Edge cases: single batch, two batches, oversized batch
  - Error propagation, thread cleanup, timer region coverage

Run with:
    uv run pytest tests/test_extraction_pipelining.py -v
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import (
    PoolingMode,
    _precompute_batch_cpu_work,
    _validate_doc_accounting,
    process_shard,
)
from steering.feature_io import load_features
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, segment_tokens, token_to_char_offsets

# ---------------------------------------------------------------------------
# Tiny model/SAE mocks (same pattern as test_extraction_profiling.py)
# ---------------------------------------------------------------------------


class TinyHookPoint(t.nn.Module):
    def forward(self, x):
        return x


class TinyBlock(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.hook_resid_pre = TinyHookPoint()


class TinySAE:
    """Minimal SAE substitute with random encoder weights."""

    def __init__(self, d_model: int = 64, n_features: int = 256, device: str = "cpu"):
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = t.randn(d_model, n_features, device=device) * 0.1
        self.b_enc = t.randn(n_features, device=device) * 0.01
        self.device = device
        self.cfg = type("obj", (object,), {"d_sae": n_features})()
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x @ self.W_enc + self.b_enc)

    def parameters(self):
        yield self._dummy_param


class TinyHookedTransformer:
    """Minimal HookedTransformer substitute.

    Uses MagicMock for tokenizer, which is not picklable — this causes
    process_shard to automatically fall back to ThreadPoolExecutor instead
    of ProcessPoolExecutor when pipeline_post=True.
    """

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._device = device
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

        self.tokenizer = MagicMock()
        self.tokenizer.encode = self._encode
        self.tokenizer.decode = lambda tids: "".join(
            chr(tok % 128) if tok < 128 else f"<{tok}>" for tok in tids
        )
        # BOS token id exposed for ci extraction. 0 matches
        # conftest.MockTokenizer._bos_id convention.
        self.tokenizer.bos_token_id = 0

        def _tokenizer_call(texts_or_text, **kw):
            if isinstance(texts_or_text, str):
                return {"input_ids": self._encode(texts_or_text)}
            return {"input_ids": [self._encode(text) for text in texts_or_text]}

        self.tokenizer.side_effect = _tokenizer_call

        self.cfg = type("obj", (object,), {"n_ctx": 2048, "model_name": "gpt2"})()

        self._embedding = t.randn(vocab_size, d_model, device=device) * 0.1
        self.blocks = [TinyBlock() for _ in range(12)]

    def _encode(self, text: str) -> list[int]:
        """Byte-pair-ish encoding with BOS prepend.

        Matches real HF tokenizer semantics where ``add_special_tokens=True``
        (the default) causes BOS to be prepended to the output. Maps each
        character to a token id in [1, vocab_size); 0 is reserved for BOS.
        """
        if not text:
            return []
        encoded = [max(1, ord(c) % self.vocab_size) for c in text]
        return [0, *encoded]

    def parameters(self):
        yield self._dummy_param

    def run_with_cache(self, tokens: t.Tensor, names_filter: list[str], **kwargs):
        seq_len = tokens.shape[1]
        token_ids = tokens[0].clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]
        cache = {name: activations.unsqueeze(0) for name in names_filter}
        logits = t.zeros(1, seq_len, self.vocab_size, device=self._device)
        return logits, cache

    def __call__(self, tokens, stop_at_layer=None, **kwargs):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        B, seq_len = tokens.shape
        token_ids = tokens.clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]
        end = stop_at_layer if stop_at_layer is not None else len(self.blocks)
        for block in self.blocks[:end]:
            activations = block.hook_resid_pre(activations)
        if stop_at_layer is not None:
            return activations
        return t.zeros(B, seq_len, self.vocab_size, device=self._device)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_and_sae():
    model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    return model, sae


@pytest.fixture(scope="module")
def sample_parquet(tmp_path_factory):
    """Create a small parquet file with 5 documents."""
    tmp_dir = tmp_path_factory.mktemp("pipelining_test")
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 8,
        "In mathematics, a group is an algebraic structure. " * 6,
        "The theory of relativity was published by Einstein. " * 12,
        "Natural language processing enables computers to understand text. " * 9,
    ]
    path = str(tmp_dir / "test_shard.parquet")
    pq.write_table(pa.table({"text": texts}), path)
    return path


@pytest.fixture(scope="module")
def single_doc_parquet(tmp_path_factory):
    """Parquet with 1 short document."""
    tmp_dir = tmp_path_factory.mktemp("pipelining_single")
    texts = ["Hello world, this is a simple test document. " * 5]
    path = str(tmp_dir / "single.parquet")
    pq.write_table(pa.table({"text": texts}), path)
    return path


@pytest.fixture(scope="module")
def two_doc_parquet(tmp_path_factory):
    """Parquet with 2 documents (minimum for overlap)."""
    tmp_dir = tmp_path_factory.mktemp("pipelining_two")
    texts = [
        "Alpha bravo charlie delta echo foxtrot. " * 8,
        "Golf hotel india juliet kilo lima mike. " * 8,
    ]
    path = str(tmp_dir / "two.parquet")
    pq.write_table(pa.table({"text": texts}), path)
    return path


def _common_kwargs(parquet_path, output_dir, **overrides):
    """Shared process_shard kwargs."""
    base = {
        "parquet_path": parquet_path,
        "layer": 8,
        "output_dir": output_dir,
        "context_window": 256,
        "pooling": PoolingMode.mean_pool,
        "gpu_batch_size": 2,
        "cpu_batch_size": 5,
        "segment_mode": SegmentMode.fixed_tokens,
        "segment_size": 64,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestPrecomputeBatchCpuWork
# ---------------------------------------------------------------------------


class TestPrecomputeBatchCpuWork:
    """Tests for _precompute_batch_cpu_work standalone function."""

    def _make_batch(self, model, texts, context_window=256):
        """Tokenize texts into the batch format expected by _precompute_batch_cpu_work."""
        tokenizer = model.tokenizer
        batch = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            n_tokens = len(token_ids)
            if n_tokens > context_window:
                kept_ids = token_ids[:context_window]
            else:
                kept_ids = token_ids
            batch.append((kept_ids, n_tokens))
        return batch

    def test_returns_correct_structure(self, tiny_model_and_sae):
        """Return type is list of 4-tuples with expected types."""
        model, _ = tiny_model_and_sae
        texts = ["Hello world test document. " * 5]
        batch = self._make_batch(model, texts)

        results = _precompute_batch_cpu_work(
            batch,
            model.tokenizer,
            SegmentMode.fixed_tokens,
            64,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        segments, n_tokens, char_starts, char_ends = results[0]
        assert isinstance(segments, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in segments)
        assert isinstance(n_tokens, int)
        assert isinstance(char_starts, list)
        assert isinstance(char_ends, list)
        assert len(char_starts) == len(segments)
        assert len(char_ends) == len(segments)

    def test_segments_match_direct_call(self, tiny_model_and_sae):
        """Segments from _precompute match calling segment_tokens directly."""
        model, _ = tiny_model_and_sae
        texts = [
            "The quick brown fox. " * 10,
            "Another document with different content. " * 8,
        ]
        batch = self._make_batch(model, texts)

        results = _precompute_batch_cpu_work(
            batch,
            model.tokenizer,
            SegmentMode.fixed_tokens,
            64,
        )

        for (kept_ids, _), (segments, _, _, _) in zip(batch, results, strict=True):
            expected = segment_tokens(
                kept_ids,
                SegmentMode.fixed_tokens,
                PositiveInt.parse(64),
            )
            assert segments == expected

    def test_char_offsets_match_direct_call(self, tiny_model_and_sae):
        """Char starts/ends match computing token_to_char_offsets directly."""
        model, _ = tiny_model_and_sae
        texts = ["Testing character offsets computation. " * 8]
        batch = self._make_batch(model, texts)

        results = _precompute_batch_cpu_work(
            batch,
            model.tokenizer,
            SegmentMode.fixed_tokens,
            64,
        )

        kept_ids, _ = batch[0]
        segments, _, char_starts, char_ends = results[0]
        char_offsets = token_to_char_offsets(model.tokenizer, kept_ids)

        expected_starts = [char_offsets[s] for s, _ in segments]
        expected_ends = [char_offsets[e] for _, e in segments]
        assert char_starts == expected_starts
        assert char_ends == expected_ends

    def test_raises_on_no_segments(self, tiny_model_and_sae):
        """Propagates ValueError when segment_document returns no segments."""
        model, _ = tiny_model_and_sae
        batch = [([1, 2, 3], 3)]

        with patch("steering.extract_features.segment_document", return_value=([], [], [])):
            with pytest.raises(ValueError, match="no segments"):
                _precompute_batch_cpu_work(
                    batch,
                    model.tokenizer,
                    SegmentMode.fixed_tokens,
                    64,
                )


# ---------------------------------------------------------------------------
# TestPipelinedProcessShard
# ---------------------------------------------------------------------------


class TestPipelinedProcessShard:
    """Tests for process_shard with pipeline_post=True vs False."""

    def _run_both_modes(self, model, sae, parquet_path, tmp_path, **extra_kwargs):
        """Run process_shard in both modes and return (sequential, pipelined) results."""
        out_seq = str(tmp_path / "sequential")
        out_pipe = str(tmp_path / "pipelined")
        os.makedirs(out_seq, exist_ok=True)
        os.makedirs(out_pipe, exist_ok=True)

        kwargs_seq = _common_kwargs(parquet_path, out_seq, **extra_kwargs)
        kwargs_pipe = _common_kwargs(parquet_path, out_pipe, **extra_kwargs)

        result_seq = process_shard(
            model=model, sae=sae, pipeline_post=False, **kwargs_seq
        )
        result_pipe = process_shard(
            model=model, sae=sae, pipeline_post=True, **kwargs_pipe
        )

        return result_seq, result_pipe

    def _load_shard_tensors(self, result):
        """Load saved shard data and return key tensors."""
        data = load_features(result.output_path)
        return {
            "offsets": data["offsets"],
            "feature_ids": data["feature_ids"],
            "feature_mags": data["feature_mags"],
            "doc_ids": data["doc_ids"],
            "segment_indices": data["segment_indices"],
            "segment_positions": data["segment_positions"],
            "doc_token_counts": data["doc_token_counts"],
            "segment_char_starts": data["segment_char_starts"],
            "segment_char_ends": data["segment_char_ends"],
        }

    def test_pipelined_output_identical_to_sequential(
        self, tiny_model_and_sae, sample_parquet, tmp_path
    ):
        """Critical: pipelined and sequential modes produce identical output tensors."""
        model, sae = tiny_model_and_sae
        result_seq, result_pipe = self._run_both_modes(
            model,
            sae,
            sample_parquet,
            tmp_path,
        )

        assert result_seq.n_docs == result_pipe.n_docs
        assert result_seq.n_tokens_gpt2 == result_pipe.n_tokens_gpt2
        assert result_seq.n_total_features == result_pipe.n_total_features

        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)

        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), (
                f"Mismatch in '{key}': shapes {tensors_seq[key].shape} vs {tensors_pipe[key].shape}"
            )

    def test_pipelined_single_doc_single_batch(
        self, tiny_model_and_sae, single_doc_parquet, tmp_path
    ):
        """Edge case: single document in a single batch."""
        model, sae = tiny_model_and_sae
        result_seq, result_pipe = self._run_both_modes(
            model,
            sae,
            single_doc_parquet,
            tmp_path,
            gpu_batch_size=8,
        )

        assert result_seq.n_docs == result_pipe.n_docs == 1
        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)
        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), f"Mismatch in '{key}'"

    def test_pipelined_two_batches(self, tiny_model_and_sae, two_doc_parquet, tmp_path):
        """Minimum for pipeline overlap: 2 docs, batch_size=1."""
        model, sae = tiny_model_and_sae
        result_seq, result_pipe = self._run_both_modes(
            model,
            sae,
            two_doc_parquet,
            tmp_path,
            gpu_batch_size=1,
        )

        assert result_seq.n_docs == result_pipe.n_docs == 2
        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)
        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), f"Mismatch in '{key}'"

    def test_pipelined_batch_size_exceeds_docs(
        self, tiny_model_and_sae, single_doc_parquet, tmp_path
    ):
        """Single oversized batch (batch_size > n_docs)."""
        model, sae = tiny_model_and_sae
        result_seq, result_pipe = self._run_both_modes(
            model,
            sae,
            single_doc_parquet,
            tmp_path,
            gpu_batch_size=64,
        )

        assert result_seq.n_docs == result_pipe.n_docs == 1
        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)
        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), f"Mismatch in '{key}'"

    def test_pipelined_error_propagates(
        self, tiny_model_and_sae, sample_parquet, tmp_path
    ):
        """Errors in segment_document propagate out of the pipelined path."""
        model, sae = tiny_model_and_sae
        out = str(tmp_path / "error_test")
        os.makedirs(out, exist_ok=True)

        def _raise(*args, **kwargs):
            raise RuntimeError("injected segmentation error")

        with patch("steering.extract_features.segment_document", side_effect=_raise):
            with pytest.raises(RuntimeError, match="injected segmentation error"):
                process_shard(
                    model=model,
                    sae=sae,
                    pipeline_post=True,
                    **_common_kwargs(sample_parquet, out),
                )

    def test_pipelined_no_thread_leak(
        self, tiny_model_and_sae, sample_parquet, tmp_path
    ):
        """Thread count before == after pipelined run."""
        model, sae = tiny_model_and_sae
        out = str(tmp_path / "thread_leak")
        os.makedirs(out, exist_ok=True)

        threads_before = threading.active_count()

        process_shard(
            model=model,
            sae=sae,
            pipeline_post=True,
            **_common_kwargs(sample_parquet, out),
        )

        threads_after = threading.active_count()
        assert threads_after <= threads_before, (
            f"Thread leak: {threads_before} before, {threads_after} after"
        )

    def test_saved_shard_doc_count_matches_n_docs(
        self, tiny_model_and_sae, sample_parquet, tmp_path
    ):
        """Saved doc_token_counts length == n_docs (the guard's invariant, end-to-end)."""
        model, sae = tiny_model_and_sae
        result_seq, result_pipe = self._run_both_modes(
            model, sae, sample_parquet, tmp_path
        )
        for result in (result_seq, result_pipe):
            tensors = self._load_shard_tensors(result)
            assert tensors["doc_token_counts"].shape[0] == result.n_docs

    @pytest.mark.parametrize("pipeline_post", [False, True])
    def test_doc_accounting_guard_invoked(
        self, tiny_model_and_sae, sample_parquet, tmp_path, monkeypatch, pipeline_post
    ):
        """process_shard's finalize calls _validate_doc_accounting on both
        pipeline_post paths (wiring is live)."""
        model, sae = tiny_model_and_sae

        def boom(*args, **kwargs):
            raise RuntimeError("doc-accounting guard reached")

        monkeypatch.setattr(
            "steering.extract_features._validate_doc_accounting", boom
        )
        out = str(tmp_path / "out")
        os.makedirs(out, exist_ok=True)
        with pytest.raises(RuntimeError, match="doc-accounting guard reached"):
            process_shard(
                model=model,
                sae=sae,
                pipeline_post=pipeline_post,
                **_common_kwargs(sample_parquet, out),
            )


class TestValidateDocAccounting:
    """Unit tests for the doc-accounting invariant guard (issue #523)."""

    def test_aligned_passes(self):
        # doc_counter == len(doc_token_counts) == len(doc_segment_info): no raise.
        _validate_doc_accounting(
            [5, 3],
            [(0, 2, [], [], []), (1, 0, [], [], [])],
            2,
        )

    def test_token_counts_short_raises(self):
        with pytest.raises(RuntimeError, match="doc_token_counts"):
            _validate_doc_accounting(
                [5],
                [(0, 1, [], [], []), (1, 0, [], [], [])],
                2,
            )

    def test_token_counts_long_raises(self):
        with pytest.raises(RuntimeError, match="doc_token_counts"):
            _validate_doc_accounting(
                [5, 3, 1],
                [(0, 1, [], [], []), (1, 0, [], [], [])],
                2,
            )

    def test_segment_info_short_raises(self):
        with pytest.raises(RuntimeError, match="doc_segment_info"):
            _validate_doc_accounting(
                [5, 3],
                [(0, 1, [], [], [])],
                2,
            )

    def test_segment_info_long_raises(self):
        with pytest.raises(RuntimeError, match="doc_segment_info"):
            _validate_doc_accounting(
                [5, 3],
                [(0, 1, [], [], []), (1, 0, [], [], []), (2, 0, [], [], [])],
                2,
            )

    def test_empty_input_passes(self):
        # Degenerate empty-parquet finalize boundary: ([], [], 0) must not raise.
        _validate_doc_accounting([], [], 0)

    def test_both_misaligned_reports_token_count_first(self):
        # When both accumulators are misaligned, the token-count check fires first
        # (the documented ordering the message-matching contract depends on).
        with pytest.raises(RuntimeError, match="doc_token_counts"):
            _validate_doc_accounting([5], [(0, 1, [], [], [])], 2)
