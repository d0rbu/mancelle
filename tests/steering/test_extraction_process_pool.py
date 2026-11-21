"""Tests for ProcessPoolExecutor-based CPU pipelining in process_shard.

Tests:
  - _is_picklable correctly classifies picklable vs non-picklable objects
  - ProcessPoolExecutor used with picklable tokenizer, output matches sequential
  - ThreadPoolExecutor fallback with MagicMock tokenizer
  - Process vs thread output tensor identity
  - Worker initializer sets globals correctly
  - Inline assertions fire on corrupted worker output

Run with:
    uv run pytest tests/test_extraction_process_pool.py -v
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.extract_features import (
    PoolingMode,
    _init_worker,
    _is_picklable,
    _tokenize_and_segment_chunk,
    _tokenize_and_segment_chunk_worker,
    process_shard,
)
from steering.feature_io import load_features
from steering.segmentation import SegmentMode

# ---------------------------------------------------------------------------
# Picklable test doubles
# ---------------------------------------------------------------------------


class PicklableTestTokenizer:
    """Simple picklable tokenizer matching MagicMock interface from TinyHookedTransformer."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tok % 128) if tok < 128 else f"<{tok}>" for tok in token_ids)

    def __call__(self, texts_or_text, **kwargs):
        if isinstance(texts_or_text, str):
            return {"input_ids": self.encode(texts_or_text)}
        return {"input_ids": [self.encode(text) for text in texts_or_text]}


class PicklableBosTokenizer(PicklableTestTokenizer):
    """Picklable character tokenizer with the explicit BOS used by turn mode."""

    bos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *super().encode(text)]

    def decode(self, token_ids: list[int]) -> str:
        prefix = "<bos>" if token_ids and token_ids[0] == self.bos_token_id else ""
        body = token_ids[1:] if prefix else token_ids
        return prefix + super().decode(body)


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


class PicklableTinyHookedTransformer:
    """TinyHookedTransformer variant with a picklable tokenizer (no MagicMock)."""

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._device = device
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

        self.tokenizer = PicklableTestTokenizer(vocab_size=vocab_size)

        self.cfg = type("obj", (object,), {"n_ctx": 2048, "model_name": "gpt2"})()

        self._embedding = t.randn(vocab_size, d_model, device=device) * 0.1
        self.blocks = [TinyBlock() for _ in range(12)]

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


class PicklableTurnHookedTransformer(PicklableTinyHookedTransformer):
    """Picklable tiny model whose tokenizer satisfies turn-mode BOS semantics."""

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        super().__init__(d_model=d_model, vocab_size=vocab_size, device=device)
        self.tokenizer = PicklableBosTokenizer(vocab_size=vocab_size)


class MockMagicTokenizer:
    """Non-picklable tokenizer using MagicMock (same as original TinyHookedTransformer)."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self._mock = MagicMock()
        # MagicMock attributes make this non-picklable

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tok % 128) if tok < 128 else f"<{tok}>" for tok in token_ids)

    def __call__(self, texts_or_text, **kwargs):
        if isinstance(texts_or_text, str):
            return {"input_ids": self.encode(texts_or_text)}
        return {"input_ids": [self.encode(text) for text in texts_or_text]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def picklable_model_and_sae():
    model = PicklableTinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    return model, sae


@pytest.fixture(scope="module")
def picklable_bos_model_and_sae():
    model = PicklableTurnHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    return model, sae


@pytest.fixture(scope="module")
def sample_parquet(tmp_path_factory):
    """Create a small parquet file with 5 documents."""
    tmp_dir = tmp_path_factory.mktemp("process_pool_test")
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
    tmp_dir = tmp_path_factory.mktemp("process_pool_single")
    texts = ["Hello world, this is a simple test document. " * 5]
    path = str(tmp_dir / "single.parquet")
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
# TestIsPicklable
# ---------------------------------------------------------------------------


class TestIsPicklable:
    """Tests for _is_picklable utility function."""

    def test_picklable_tokenizer(self):
        """PicklableTestTokenizer is classified as picklable."""
        tok = PicklableTestTokenizer()
        assert _is_picklable(tok) is True

    def test_magic_mock_not_picklable(self):
        """MagicMock is not picklable."""
        mock = MagicMock()
        assert _is_picklable(mock) is False

    def test_mock_magic_tokenizer_not_picklable(self):
        """Tokenizer containing MagicMock attributes is not picklable."""
        tok = MockMagicTokenizer()
        assert _is_picklable(tok) is False

    def test_builtin_types_picklable(self):
        """Basic Python types are picklable."""
        assert _is_picklable("hello") is True
        assert _is_picklable(42) is True
        assert _is_picklable([1, 2, 3]) is True


# ---------------------------------------------------------------------------
# TestProcessPoolExecutorPath
# ---------------------------------------------------------------------------


class TestProcessPoolExecutorPath:
    """Tests verifying ProcessPoolExecutor is used with picklable tokenizer."""

    def _load_shard_tensors(self, result):
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

    def test_ppe_used_with_picklable_tokenizer(
        self,
        picklable_model_and_sae,
        sample_parquet,
        tmp_path,
    ):
        """ProcessPoolExecutor is chosen when tokenizer is picklable."""
        model, sae = picklable_model_and_sae
        out = str(tmp_path / "ppe_check")
        os.makedirs(out, exist_ok=True)

        with patch(
            "steering.extract_features.ProcessPoolExecutor",
            wraps=ProcessPoolExecutor,
        ) as mock_ppe:
            process_shard(
                model=model,
                sae=sae,
                pipeline_post=True,
                **_common_kwargs(sample_parquet, out),
            )
            assert mock_ppe.called, (
                "ProcessPoolExecutor should be used with picklable tokenizer"
            )

    def test_ppe_output_matches_sequential(
        self,
        picklable_model_and_sae,
        sample_parquet,
        tmp_path,
    ):
        """PPE pipelined output matches sequential (pipeline_post=False)."""
        model, sae = picklable_model_and_sae
        out_seq = str(tmp_path / "ppe_seq")
        out_pipe = str(tmp_path / "ppe_pipe")
        os.makedirs(out_seq, exist_ok=True)
        os.makedirs(out_pipe, exist_ok=True)

        result_seq = process_shard(
            model=model,
            sae=sae,
            pipeline_post=False,
            **_common_kwargs(sample_parquet, out_seq),
        )
        result_pipe = process_shard(
            model=model,
            sae=sae,
            pipeline_post=True,
            **_common_kwargs(sample_parquet, out_pipe),
        )

        assert result_seq.n_docs == result_pipe.n_docs
        assert result_seq.n_tokens_gpt2 == result_pipe.n_tokens_gpt2
        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)
        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), f"Mismatch in '{key}'"

    def test_no_process_leak(
        self,
        picklable_model_and_sae,
        sample_parquet,
        tmp_path,
    ):
        """No child processes remain after pipelined run."""
        model, sae = picklable_model_and_sae
        out = str(tmp_path / "proc_leak")
        os.makedirs(out, exist_ok=True)

        import psutil

        proc = psutil.Process()
        children_before = len(proc.children(recursive=True))

        process_shard(
            model=model,
            sae=sae,
            pipeline_post=True,
            **_common_kwargs(sample_parquet, out),
        )

        children_after = len(proc.children(recursive=True))
        assert children_after <= children_before, (
            f"Process leak: {children_before} before, {children_after} after"
        )

    def test_ppe_turn_mode_matches_sequential(
        self,
        picklable_bos_model_and_sae,
        tmp_path,
    ):
        """Turn boundaries cross the PPE submit/worker seam unchanged."""
        model, sae = picklable_bos_model_and_sae
        parquet = tmp_path / "turn.parquet"
        pq.write_table(
            pa.table(
                {
                    "text": ["abcdefghij", "abcdefghijklmnop", "abcdefghijklm"],
                    "turn_char_starts": [[0, 4, 8], [0, 5, 11], [0, 6]],
                }
            ),
            parquet,
        )
        kwargs = _common_kwargs(
            str(parquet),
            str(tmp_path / "unused"),
            dataset_phase="standard_sft",
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            segment_size=4,
            cpu_batch_size=2,
        )
        with patch(
            "steering.extract_features.ProcessPoolExecutor",
            wraps=ProcessPoolExecutor,
        ) as mock_ppe:
            ppe = process_shard(
                model=model,
                sae=sae,
                pipeline_post=True,
                **(kwargs | {"output_dir": str(tmp_path / "ppe_turn")}),
            )
        assert mock_ppe.called
        sequential = process_shard(
            model=model,
            sae=sae,
            pipeline_post=False,
            **(kwargs | {"output_dir": str(tmp_path / "seq_turn")}),
        )
        ppe_data = load_features(ppe.output_path)
        sequential_data = load_features(sequential.output_path)
        for key in self._load_shard_tensors(ppe):
            assert t.equal(ppe_data[key], sequential_data[key]), f"Mismatch in {key}"
        assert ppe_data["segment_mode"] == sequential_data["segment_mode"] == (
            SegmentMode.turn_delimited_fixed_tokens.value
        )


# ---------------------------------------------------------------------------
# TestThreadPoolFallback
# ---------------------------------------------------------------------------


class TestThreadPoolFallback:
    """Tests verifying ThreadPoolExecutor is used with non-picklable tokenizer."""

    def _load_shard_tensors(self, result):
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

    def test_tpe_used_with_mock_tokenizer(self, sample_parquet, tmp_path):
        """ThreadPoolExecutor fallback when tokenizer is not picklable (MagicMock)."""
        from tests.steering.test_extraction_pipelining import TinyHookedTransformer as MockModel

        model = MockModel(d_model=64, vocab_size=1000, device="cpu")
        sae = TinySAE(d_model=64, n_features=256, device="cpu")
        out = str(tmp_path / "tpe_check")
        os.makedirs(out, exist_ok=True)

        with patch(
            "steering.extract_features.ThreadPoolExecutor",
            wraps=ThreadPoolExecutor,
        ) as mock_tpe:
            process_shard(
                model=model,  # ty: ignore[invalid-argument-type]
                sae=sae,  # ty: ignore[invalid-argument-type]
                pipeline_post=True,
                **_common_kwargs(sample_parquet, out),
            )
            assert mock_tpe.called, (
                "ThreadPoolExecutor should be used with non-picklable tokenizer"
            )

    def test_tpe_output_correct(self, sample_parquet, tmp_path):
        """TPE fallback produces correct output (matches sequential)."""
        from tests.steering.test_extraction_pipelining import TinyHookedTransformer as MockModel

        model = MockModel(d_model=64, vocab_size=1000, device="cpu")
        sae = TinySAE(d_model=64, n_features=256, device="cpu")
        out_seq = str(tmp_path / "tpe_seq")
        out_pipe = str(tmp_path / "tpe_pipe")
        os.makedirs(out_seq, exist_ok=True)
        os.makedirs(out_pipe, exist_ok=True)

        result_seq = process_shard(
            model=model,  # ty: ignore[invalid-argument-type]
            sae=sae,  # ty: ignore[invalid-argument-type]
            pipeline_post=False,
            **_common_kwargs(sample_parquet, out_seq),
        )
        result_pipe = process_shard(
            model=model,  # ty: ignore[invalid-argument-type]
            sae=sae,  # ty: ignore[invalid-argument-type]
            pipeline_post=True,
            **_common_kwargs(sample_parquet, out_pipe),
        )

        assert result_seq.n_docs == result_pipe.n_docs
        tensors_seq = self._load_shard_tensors(result_seq)
        tensors_pipe = self._load_shard_tensors(result_pipe)
        for key in tensors_seq:
            assert t.equal(tensors_seq[key], tensors_pipe[key]), f"Mismatch in '{key}'"


# ---------------------------------------------------------------------------
# TestProcessVsThreadOutputIdentical
# ---------------------------------------------------------------------------


class TestProcessVsThreadOutputIdentical:
    """Verify that process-pool and thread-pool paths produce identical output."""

    def _load_shard_tensors(self, result):
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

    def test_process_vs_thread_identical(self, sample_parquet, tmp_path):
        """Same data through PPE and TPE produces identical 9 output tensors."""
        # PPE path: picklable model
        model_ppe = PicklableTinyHookedTransformer(
            d_model=64, vocab_size=1000, device="cpu"
        )
        sae_ppe = TinySAE(d_model=64, n_features=256, device="cpu")

        # TPE path: force thread pool by patching _is_picklable
        model_tpe = PicklableTinyHookedTransformer(
            d_model=64, vocab_size=1000, device="cpu"
        )
        sae_tpe = TinySAE(d_model=64, n_features=256, device="cpu")

        # Use same random state for SAE weights
        t.manual_seed(42)
        sae_shared = TinySAE(d_model=64, n_features=256, device="cpu")
        sae_ppe.W_enc = sae_shared.W_enc.clone()
        sae_ppe.b_enc = sae_shared.b_enc.clone()
        sae_tpe.W_enc = sae_shared.W_enc.clone()
        sae_tpe.b_enc = sae_shared.b_enc.clone()

        # Use same embedding
        model_tpe._embedding = model_ppe._embedding.clone()

        out_ppe = str(tmp_path / "ppe_path")
        out_tpe = str(tmp_path / "tpe_path")
        os.makedirs(out_ppe, exist_ok=True)
        os.makedirs(out_tpe, exist_ok=True)

        result_ppe = process_shard(
            model=model_ppe,  # ty: ignore[invalid-argument-type]
            sae=sae_ppe,  # ty: ignore[invalid-argument-type]
            pipeline_post=True,
            **_common_kwargs(sample_parquet, out_ppe),
        )

        with patch("steering.extract_features._is_picklable", return_value=False):
            result_tpe = process_shard(
                model=model_tpe,  # ty: ignore[invalid-argument-type]
                sae=sae_tpe,  # ty: ignore[invalid-argument-type]
                pipeline_post=True,
                **_common_kwargs(sample_parquet, out_tpe),
            )

        tensors_ppe = self._load_shard_tensors(result_ppe)
        tensors_tpe = self._load_shard_tensors(result_tpe)

        for key in tensors_ppe:
            assert t.equal(tensors_ppe[key], tensors_tpe[key]), (
                f"PPE vs TPE mismatch in '{key}'"
            )


# ---------------------------------------------------------------------------
# TestWorkerInitializer
# ---------------------------------------------------------------------------


class TestWorkerInitializer:
    """Tests for _init_worker and _tokenize_and_segment_chunk_worker."""

    def test_init_worker_sets_globals(self):
        """_init_worker sets module-level globals correctly."""
        import steering.extract_features as ef

        tok = PicklableTestTokenizer(vocab_size=500)
        _init_worker(tok, 128, "fixed_tokens", 32, 2048)

        assert ef._worker_tokenizer is tok
        assert ef._worker_context_window == 128
        assert ef._worker_segment_mode == "fixed_tokens"
        assert ef._worker_segment_size == 32
        assert ef._worker_sae_max_context == 2048

    def test_worker_function_uses_globals(self):
        """_tokenize_and_segment_chunk_worker reads initialized globals."""

        tok = PicklableTestTokenizer(vocab_size=1000)
        _init_worker(tok, 256, "fixed_tokens", 64, 2048)

        texts = ["Hello world this is a test. " * 10]
        token_info, precomputed = _tokenize_and_segment_chunk_worker(texts)

        # Compare with direct call
        expected_tok, expected_pre = _tokenize_and_segment_chunk(
            tok,
            texts,
            256,
            SegmentMode.fixed_tokens,
            64,
            sae_max_context=2048,
        )

        assert len(token_info) == len(expected_tok)
        assert len(precomputed) == len(expected_pre)
        for (a_ids, a_n), (b_ids, b_n) in zip(token_info, expected_tok, strict=True):
            assert a_ids == b_ids
            assert a_n == b_n

    def test_worker_forwards_turn_char_starts(self):
        """The worker forwards turn boundaries and fails loudly when they are absent."""
        tok = PicklableBosTokenizer(vocab_size=1000)
        texts = ["abcdefghij", "abcdefghijklmnop"]
        starts = [[0, 4, 8], [0, 5, 11]]
        _init_worker(tok, 256, SegmentMode.turn_delimited_fixed_tokens.value, 4, 256)

        actual = _tokenize_and_segment_chunk_worker(texts, starts)
        expected = _tokenize_and_segment_chunk(
            tok,
            texts,
            256,
            SegmentMode.turn_delimited_fixed_tokens,
            4,
            sae_max_context=256,
            turn_char_starts=starts,
        )
        assert actual == expected
        with pytest.raises(ValueError, match="requires texts and turn_char_starts"):
            _tokenize_and_segment_chunk_worker(texts)


# ---------------------------------------------------------------------------
# TestPipelineAssertions
# ---------------------------------------------------------------------------


class TestPipelineAssertions:
    """Tests for inline assertions in the pipelined path."""

    def test_length_mismatch_assertion(
        self, picklable_model_and_sae, sample_parquet, tmp_path
    ):
        """Assertion fires when worker returns mismatched token_info/precomputed lengths."""
        model, sae = picklable_model_and_sae
        out = str(tmp_path / "assert_len")
        os.makedirs(out, exist_ok=True)

        original_fn = _tokenize_and_segment_chunk

        def _corrupted(tokenizer, texts, cw, sm, ss, **kwargs):
            token_info, precomputed = original_fn(
                tokenizer, texts, cw, sm, ss, **kwargs
            )
            # Drop one precomputed entry to create length mismatch
            return token_info, precomputed[:-1] if len(precomputed) > 1 else precomputed

        with patch(
            "steering.extract_features._tokenize_and_segment_chunk",
            side_effect=_corrupted,
        ):
            # Force TPE path so patch works (PPE runs in subprocess)
            with patch("steering.extract_features._is_picklable", return_value=False):
                with pytest.raises(AssertionError):
                    process_shard(
                        model=model,
                        sae=sae,
                        pipeline_post=True,
                        **_common_kwargs(sample_parquet, out),
                    )

    def test_batch_length_assertion(
        self, picklable_model_and_sae, single_doc_parquet, tmp_path
    ):
        """Assertion fires when batch_tok and batch_pre have different lengths."""
        model, sae = picklable_model_and_sae
        out = str(tmp_path / "assert_batch")
        os.makedirs(out, exist_ok=True)

        original_fn = _tokenize_and_segment_chunk

        def _corrupted(tokenizer, texts, cw, sm, ss, **kwargs):
            token_info, precomputed = original_fn(
                tokenizer, texts, cw, sm, ss, **kwargs
            )
            # Add an extra token_info entry
            if token_info:
                token_info = token_info + [token_info[0]]
            return token_info, precomputed

        with patch(
            "steering.extract_features._tokenize_and_segment_chunk",
            side_effect=_corrupted,
        ):
            with patch("steering.extract_features._is_picklable", return_value=False):
                with pytest.raises(AssertionError):
                    process_shard(
                        model=model,
                        sae=sae,
                        pipeline_post=True,
                        **_common_kwargs(single_doc_parquet, out),
                    )
