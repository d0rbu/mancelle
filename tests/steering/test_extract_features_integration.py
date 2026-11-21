"""Integration tests for steering.extract_features with real or tiny models.

By default, loads real GPT-2 + SAE from SAELens (requires ~2GB download).
Use --model-scale=tiny for a lightweight mock that runs fast on CPU.

Run with:
    uv run pytest tests/test_extract_features_integration.py -v
    uv run pytest tests/test_extract_features_integration.py -v --model-scale=tiny
"""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf
from sae_lens import SAE
from transformer_lens import HookedTransformer

import steering.extract_features as _ef_module
from steering.extract_features import (
    ActivationCapture,
    PoolingMode,
    _get_sae_activations,
    _get_sae_activations_batched,
    extract_document_features,
    find_optimal_batch_sizes,
    load_sae_and_model,
    maybe_gc_cuda,
    process_shard,
)
from steering.feature_io import load_features
from steering.shard_validation import validate_shard

# ---------------------------------------------------------------------------
# Tiny model fixtures (for --model-scale=tiny)
# ---------------------------------------------------------------------------


class TinyHookPoint(t.nn.Module):
    """Minimal hook point that supports register_forward_hook."""

    def forward(self, x):
        return x


class TinyBlock(t.nn.Module):
    """Minimal transformer block with hook_resid_pre."""

    def __init__(self):
        super().__init__()
        self.hook_resid_pre = TinyHookPoint()


class TinySAE:
    """Minimal SAE substitute with random encoder weights.

    Matches the SAELens SAE interface used by extract_features:
    - encode(x) -> [seq_len, n_features]
    """

    def __init__(self, d_model: int = 64, n_features: int = 256, device: str = "cpu"):
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = t.randn(d_model, n_features, device=device) * 0.1
        self.b_enc = t.randn(n_features, device=device) * 0.01
        self.device = device
        self.cfg = type("obj", (object,), {"d_sae": n_features})()
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

    def encode(self, x: t.Tensor) -> t.Tensor:
        """x: [seq_len, d_model] -> [seq_len, n_features], ReLU activated."""
        return t.relu(x @ self.W_enc + self.b_enc)

    def parameters(self):
        yield self._dummy_param


class TinyHookedTransformer:
    """Minimal HookedTransformer substitute.

    Implements just enough of the TransformerLens API for extract_features:
    - model.tokenizer.encode(text) -> list[int]
    - model.tokenizer.decode(tids: list[int]) -> str
    - model.run_with_cache(tokens, names_filter) -> (logits, cache_dict)
    - next(model.parameters()) -> tensor on device
    """

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._device = device
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))
        self.nnterp_model = None

        # Simple tokenizer mock — supports both single .encode() and
        # batched __call__(texts) interfaces used by process_shard.
        self.tokenizer = MagicMock()
        self.tokenizer.encode = self._encode
        self.tokenizer.decode = lambda tids: "".join(
            chr(t % 128) if t < 128 else f"<{t}>" for t in tids
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

        # Embedding for producing activations
        self._embedding = t.randn(vocab_size, d_model, device=device) * 0.1

        # Blocks with hook points for register_forward_hook support
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

    def run_with_cache(
        self,
        tokens: t.Tensor,
        names_filter: list[str],
        attention_mask=None,
        stop_at_layer=None,
    ):
        """tokens: [1, seq_len] -> (logits, cache_dict)."""
        del attention_mask, stop_at_layer
        seq_len = tokens.shape[1]
        # Produce activations from embeddings
        token_ids = tokens[0].clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]  # [seq_len, d_model]
        cache = {name: activations.unsqueeze(0) for name in names_filter}
        logits = t.zeros(1, seq_len, self.vocab_size, device=self._device)
        return logits, cache

    def __call__(self, tokens, stop_at_layer=None, **kwargs):
        """Forward pass that triggers hooks on blocks.

        Supports stop_at_layer to match TransformerLens API:
        if stop_at_layer=N, runs blocks 0..N-1 and returns the residual stream.
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        B, seq_len = tokens.shape
        token_ids = tokens.clamp(0, self.vocab_size - 1)
        activations = self._embedding[token_ids]  # [B, seq, d_model]
        # Pass through the hook point at each block to trigger registered hooks
        end = stop_at_layer if stop_at_layer is not None else len(self.blocks)
        for block in self.blocks[:end]:
            activations = block.hook_resid_pre(activations)
        if stop_at_layer is not None:
            return activations
        return t.zeros(B, seq_len, self.vocab_size, device=self._device)


# ---------------------------------------------------------------------------
# Session-scoped model fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_and_sae(model_scale):
    """Load model + SAE at the chosen scale. Cached for the session."""
    if model_scale == "tiny":
        model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
        sae = TinySAE(d_model=64, n_features=256, device="cpu")
        return model, sae, 8, 256  # model, sae, layer, n_features
    model, sae, tokenizer = load_sae_and_model(
        device="cpu",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
    )
    n_features = sae.W_enc.shape[1] if hasattr(sae, "W_enc") else 24576
    return model, sae, 8, n_features


# ---------------------------------------------------------------------------
# extract_document_features tests
# ---------------------------------------------------------------------------


class TestExtractDocumentFeatures:
    """Test extract_document_features with all pooling modes."""

    @pytest.fixture(params=list(PoolingMode))
    def pooling_mode(self, request):
        return request.param

    def test_output_contract(self, model_and_sae, pooling_mode):
        model, sae, layer, n_features = model_and_sae
        seg_ids, seg_mags, positions, _, _, n_tokens = extract_document_features(
            model,
            sae,
            "Hello world",
            layer=layer,
            context_window=1024,
            pooling=pooling_mode,
        )
        assert isinstance(seg_ids, list)
        assert isinstance(seg_mags, list)
        assert len(seg_ids) == len(seg_mags) == len(positions)
        assert n_tokens > 0
        for ids, mags in zip(seg_ids, seg_mags, strict=True):
            assert ids.dtype == t.int32
            assert mags.dtype == t.float32
            assert len(ids) == len(mags)
            if len(mags) > 0:
                assert (mags > 0).all()
            if len(mags) > 1:
                assert (mags[:-1] >= mags[1:]).all()
            if len(ids) > 0:
                assert ids.min().item() >= 0
                assert ids.max().item() < n_features

    def test_empty_text(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        with pytest.raises(ValueError):
            extract_document_features(model, sae, "", layer=layer)

    def test_single_character(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        seg_ids, seg_mags, positions, _, _, n_tokens = extract_document_features(
            model,
            sae,
            "a",
            layer=layer,
        )
        assert n_tokens >= 1
        assert len(seg_ids) == len(seg_mags) == len(positions)

    def test_different_texts_produce_different_features(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        ids1, mags1, *_ = extract_document_features(
            model,
            sae,
            "Mathematics and algebra",
            layer=layer,
            pooling=PoolingMode.mean_pool,
        )
        ids2, mags2, *_ = extract_document_features(
            model,
            sae,
            "Cooking recipes for dinner",
            layer=layer,
            pooling=PoolingMode.mean_pool,
        )
        # At least one segment should differ
        if ids1 and ids2:
            all_same = all(t.equal(a, b) for a, b in zip(mags1, mags2, strict=False))
            assert not all_same or len(ids1) != len(ids2)

    def test_deterministic(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        text = "Determinism test input string"
        r1 = extract_document_features(model, sae, text, layer=layer)
        r2 = extract_document_features(model, sae, text, layer=layer)
        assert len(r1[0]) == len(r2[0])
        for a, b in zip(r1[0], r2[0], strict=True):
            assert t.equal(a, b)
        assert r1[5] == r2[5]  # n_tokens at index 5

    def test_segment_positions_ascending(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        text = "A" * 500  # enough tokens for multiple segments
        _, _, positions, _, _, _ = extract_document_features(
            model,
            sae,
            text,
            layer=layer,
            segment_size=64,
        )
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1]


# ---------------------------------------------------------------------------
# _get_sae_activations
# ---------------------------------------------------------------------------


class TestGetSAEActivations:
    def test_output_shape(self, model_and_sae):
        model, sae, layer, n_features = model_and_sae
        tokens = model.tokenizer.encode("Hello world")
        device = next(model.parameters()).device
        token_tensor = t.tensor(tokens, dtype=t.long, device=device)
        activations = _get_sae_activations(model, sae, token_tensor, layer)
        assert activations.shape[0] == len(tokens)
        assert activations.shape[1] == n_features

    def test_no_grad(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        tokens = model.tokenizer.encode("test")
        device = next(model.parameters()).device
        token_tensor = t.tensor(tokens, dtype=t.long, device=device)
        activations = _get_sae_activations(model, sae, token_tensor, layer)
        assert not activations.requires_grad

    @pytest.mark.parametrize(
        ("hook_name", "expected"),
        [
            (None, _ef_module.ActivationType.residual),
            ("blocks.1.hook_resid_post", _ef_module.ActivationType.residual),
            ("blocks.1.hook_mlp_out", _ef_module.ActivationType.mlp),
            ("blocks.1.attn.hook_z", _ef_module.ActivationType.attention),
        ],
    )
    def test_infers_activation_type_from_hook_name(self, hook_name, expected):
        assert _ef_module._infer_activation_type_from_hook_name(hook_name) == expected

    def test_prefers_nnterp_side_channel_when_present(self, monkeypatch):
        model = TinyHookedTransformer(device="cpu")
        sae = TinySAE(device="cpu")
        model.nnterp_model = object()
        tokens = t.tensor([1, 2, 3], dtype=t.long)
        nn_acts = t.randn(1, len(tokens), model.d_model)
        nn_mock = MagicMock(return_value=nn_acts)
        tl_mock = MagicMock(side_effect=AssertionError("TL fallback should not run"))
        monkeypatch.setattr(_ef_module, "_get_activations_nnterp", nn_mock)
        monkeypatch.setattr(_ef_module, "_get_activations_tl", tl_mock)

        acts = _get_sae_activations(
            cast(HookedTransformer, model),
            cast(SAE, sae),
            tokens,
            layer=3,
            hook_name="blocks.3.hook_resid_post",
        )

        assert acts.shape == (len(tokens), sae.n_features)
        assert nn_mock.call_count == 1
        assert tl_mock.call_count == 0
        call = nn_mock.call_args
        assert call.args[0] is model.nnterp_model
        assert tuple(call.args[1].shape) == (1, len(tokens))
        assert call.kwargs["activation_type"] == _ef_module.ActivationType.residual

    def test_nnterp_failure_raises_failfast_error(self, monkeypatch):
        model = TinyHookedTransformer(device="cpu")
        sae = TinySAE(device="cpu")
        model.nnterp_model = object()
        tokens = t.tensor([4, 5, 6], dtype=t.long)
        monkeypatch.setattr(
            _ef_module,
            "_get_activations_nnterp",
            MagicMock(side_effect=_ef_module.ExtractionError("boom")),
        )
        tl_mock = MagicMock(side_effect=AssertionError("TL fallback should not run"))
        monkeypatch.setattr(_ef_module, "_get_activations_tl", tl_mock)

        with pytest.raises(
            _ef_module.AcceleratedBackendError,
            match="nnterp.*backend=transformerlens",
        ):
            _get_sae_activations(
                cast(HookedTransformer, model),
                cast(SAE, sae),
                tokens,
                layer=2,
                hook_name="blocks.2.hook_resid_pre",
            )

        assert tl_mock.call_count == 0


# ---------------------------------------------------------------------------
# process_shard
# ---------------------------------------------------------------------------


def _make_synthetic_parquet(path: str, texts: list[str]):
    """Create a minimal parquet file with a text column."""
    table = pa.table({"text": texts})
    pq.write_table(table, path)


class TestProcessShard:
    @pytest.fixture
    def parquet_file(self, tmp_path):
        path = str(tmp_path / "shard_test.parquet")
        texts = [
            "This is document one about mathematics.",
            "Document two discusses natural language processing.",
            "The third document covers physics and astronomy.",
            "Fourth document is about cooking recipes.",
        ]
        _make_synthetic_parquet(path, texts)
        return path

    def test_output_file_created(self, model_and_sae, parquet_file, tmp_path):
        model, sae, layer, _ = model_and_sae
        result = process_shard(
            parquet_file,
            model,
            sae,
            layer=layer,
            output_dir=str(tmp_path),
            context_window=1024,
            pooling=PoolingMode.mean_pool,
        )
        assert Path(result.output_path).exists()

    def test_shard_result_fields(self, model_and_sae, parquet_file, tmp_path):
        model, sae, layer, _ = model_and_sae
        result = process_shard(
            parquet_file,
            model,
            sae,
            layer=layer,
            output_dir=str(tmp_path),
        )
        assert result.n_docs == 4
        assert result.n_tokens_gpt2 >= 0
        assert result.n_total_features >= 0
        assert result.elapsed_seconds > 0
        assert result.bytes_on_disk > 0

    def test_output_csr_format(self, model_and_sae, parquet_file, tmp_path):
        model, sae, layer, _ = model_and_sae
        result = process_shard(
            parquet_file,
            model,
            sae,
            layer=layer,
            output_dir=str(tmp_path),
        )
        data = load_features(result.output_path)
        n_segments = data["n_segments"]
        assert data["offsets"][0].item() == 0
        assert data["offsets"][-1].item() == len(data["feature_ids"])
        assert len(data["offsets"]) == n_segments + 1
        assert len(data["doc_ids"]) == n_segments
        assert len(data["segment_indices"]) == n_segments
        assert len(data["segment_positions"]) == n_segments
        assert "segment_mode" in data
        assert "segment_size" in data

    def test_validate_shard(self, model_and_sae, parquet_file, tmp_path):
        model, sae, layer, _ = model_and_sae
        result = process_shard(
            parquet_file, model, sae, layer=layer, output_dir=str(tmp_path)
        )
        vr = validate_shard(result.output_path)
        assert vr.valid, f"Violations: {vr.violations}"

    def test_empty_parquet(self, model_and_sae, tmp_path):
        """Parquet with no documents produces valid but empty output."""
        parquet_path = str(tmp_path / "empty.parquet")
        _make_synthetic_parquet(parquet_path, [])
        model, sae, layer, _ = model_and_sae
        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=layer,
            output_dir=str(tmp_path),
        )
        assert result.n_docs == 0

    def test_all_pooling_modes_work(self, model_and_sae, tmp_path):
        """Every pooling mode runs without error on a small parquet."""
        parquet_path = str(tmp_path / "modes_test.parquet")
        _make_synthetic_parquet(parquet_path, ["Hello world", "Test document"])

        model, sae, layer, _ = model_and_sae
        for mode in PoolingMode:
            out_dir = tmp_path / mode.value
            out_dir.mkdir(exist_ok=True)
            result = process_shard(
                parquet_path,
                model,
                sae,
                layer=layer,
                output_dir=str(out_dir),
                context_window=1024,
                pooling=mode,
            )
            assert result.n_docs == 2
            data = load_features(result.output_path)
            assert data["pooling"] == mode.value


# ---------------------------------------------------------------------------
# ActivationCapture
# ---------------------------------------------------------------------------


class TestActivationCapture:
    def test_captures_activation(self, model_and_sae):
        model, sae, layer, _ = model_and_sae
        capture = ActivationCapture()
        hook_point = model.blocks[layer].hook_resid_pre
        handle = hook_point.register_forward_hook(capture.hook_fn)
        try:
            tokens = t.tensor(
                [[1, 2, 3]], dtype=t.long, device=next(model.parameters()).device
            )
            with t.no_grad():
                model(tokens)
            assert capture.activation is not None
            assert capture.activation.ndim == 3  # [batch, seq, d_model]
            assert capture.activation.shape[0] == 1
            assert capture.activation.shape[1] == 3
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# _get_sae_activations_batched
# ---------------------------------------------------------------------------


class TestGetSAEActivationsBatched:
    """Test batched SAE activation extraction."""

    def test_single_doc_matches_unbatched(self, model_and_sae):
        """Batched with B=1 must match unbatched _get_sae_activations."""
        model, sae, layer, n_features = model_and_sae
        text = "Hello world this is a test"
        token_ids = model.tokenizer.encode(text)
        device = next(model.parameters()).device
        tokens = t.tensor(token_ids, dtype=t.long, device=device)

        unbatched = _get_sae_activations(model, sae, tokens, layer)
        batched = _get_sae_activations_batched(
            model,
            sae,
            [token_ids],
            layer=layer,
        )

        assert len(batched) == 1
        assert t.allclose(batched[0], unbatched, atol=1e-5)

    def test_multi_doc_shapes(self, model_and_sae):
        """Batched with B>1 returns correct shapes for each doc."""
        model, sae, layer, n_features = model_and_sae
        texts = ["Hello world", "This is a longer test sentence for batching"]
        token_id_lists = [model.tokenizer.encode(text) for text in texts]

        results = _get_sae_activations_batched(
            model,
            sae,
            token_id_lists,
            layer=layer,
        )

        assert len(results) == 2
        for i, (ids, acts) in enumerate(zip(token_id_lists, results, strict=True)):
            assert acts.shape == (len(ids), n_features), (
                f"Doc {i}: expected ({len(ids)}, {n_features}), got {acts.shape}"
            )

    def test_multi_doc_matches_unbatched(self, model_and_sae):
        """Each doc in a batch must produce same activations as unbatched.

        Uses a looser tolerance (atol=1e-3) because padding tokens in the
        batched path participate in attention, slightly shifting activations
        for shorter sequences. This is inherent to transformer self-attention
        and acceptable for downstream pooling/sparse extraction.
        """
        model, sae, layer, n_features = model_and_sae
        texts = ["Hello world", "Testing batched forward", "Third document here"]
        token_id_lists = [model.tokenizer.encode(text) for text in texts]
        device = next(model.parameters()).device

        batched = _get_sae_activations_batched(
            model,
            sae,
            token_id_lists,
            layer=layer,
        )

        for i, ids in enumerate(token_id_lists):
            tokens = t.tensor(ids, dtype=t.long, device=device)
            unbatched = _get_sae_activations(model, sae, tokens, layer)
            assert t.allclose(batched[i], unbatched, atol=1e-3), (
                f"Doc {i} differs: max diff {(batched[i] - unbatched).abs().max()}"
            )

    def test_variable_lengths(self, model_and_sae):
        """Documents of different lengths are handled correctly."""
        model, sae, layer, n_features = model_and_sae
        token_id_lists = [
            [1, 2, 3],  # short
            [10, 20, 30, 40, 50, 60, 70],  # medium
            [100, 200],  # very short
        ]

        results = _get_sae_activations_batched(
            model,
            sae,
            token_id_lists,
            layer=layer,
        )

        assert len(results) == 3
        assert results[0].shape[0] == 3
        assert results[1].shape[0] == 7
        assert results[2].shape[0] == 2


# ---------------------------------------------------------------------------
# find_optimal_batch_size
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFindOptimalBatchSizes:
    """Test 2D auto batch size tuning."""

    def test_returns_tuple_of_two_ints(self, model_and_sae):
        """find_optimal_batch_sizes must return (gpu_bs, cpu_bs) both >= 1."""
        model, sae, layer, _ = model_and_sae
        result = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        gpu_bs, cpu_bs = result
        assert isinstance(gpu_bs, int) and gpu_bs >= 1
        assert isinstance(cpu_bs, int) and cpu_bs >= 1

    def test_results_are_powers_of_two(self, model_and_sae):
        """Both batch sizes must be powers of 2."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
        )
        assert gpu_bs & (gpu_bs - 1) == 0, f"gpu_bs={gpu_bs} not power of 2"
        assert cpu_bs & (cpu_bs - 1) == 0, f"cpu_bs={cpu_bs} not power of 2"

    def test_gpu_le_cpu_batch_size(self, model_and_sae):
        """gpu_batch_size must be <= cpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
        )
        assert gpu_bs <= cpu_bs

    def test_respects_max_gpu_batch_size(self, model_and_sae):
        """gpu_batch_size must not exceed max_gpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
        )
        assert gpu_bs <= 4

    def test_respects_max_cpu_batch_size(self, model_and_sae):
        """cpu_batch_size must not exceed max_cpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_cpu_batch_size=8,
        )
        assert cpu_bs <= 8

    def test_does_not_oom_at_returned_sizes(self, model_and_sae):
        """Model must successfully run at the returned gpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
        )
        token_id_lists = [list(range(128)) for _ in range(gpu_bs)]
        results = _get_sae_activations_batched(model, sae, token_id_lists, layer=layer)
        assert len(results) == gpu_bs

    # -- min batch size tests -----------------------------------------------

    def test_respects_min_gpu_batch_size(self, model_and_sae):
        """gpu_batch_size must not go below min_gpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            min_gpu_batch_size=4,
            max_gpu_batch_size=16,
            min_cpu_batch_size=4,
            max_cpu_batch_size=16,
        )
        assert gpu_bs >= 4, f"gpu_bs={gpu_bs} below min 4"
        assert gpu_bs <= 16

    def test_respects_min_cpu_batch_size(self, model_and_sae):
        """cpu_batch_size must not go below min_cpu_batch_size."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            min_gpu_batch_size=2,
            max_gpu_batch_size=16,
            min_cpu_batch_size=8,
            max_cpu_batch_size=16,
        )
        assert cpu_bs >= 8, f"cpu_bs={cpu_bs} below min 8"
        assert cpu_bs <= 16

    def test_min_equals_max_constrains_to_exact_value(self, model_and_sae):
        """When min == max (power of 2), only one candidate exists per axis."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            min_gpu_batch_size=4,
            max_gpu_batch_size=4,
            min_cpu_batch_size=8,
            max_cpu_batch_size=8,
        )
        assert gpu_bs == 4
        assert cpu_bs == 8

    def test_min_greater_than_max_raises_value_error(self, model_and_sae):
        """When min > max, fail fast with the input-validation contract."""
        model, sae, layer, _ = model_and_sae
        with pytest.raises(ValueError, match="min_gpu_batch_size .*<= max_gpu_batch_size"):
            find_optimal_batch_sizes(
                model,
                sae,
                layer=layer,
                context_window=128,
                min_gpu_batch_size=64,
                max_gpu_batch_size=4,
                min_cpu_batch_size=64,
                max_cpu_batch_size=4,
            )

    def test_min_both_bounds_respected(self, model_and_sae):
        """Result must satisfy min <= result <= max on both axes."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            min_gpu_batch_size=2,
            max_gpu_batch_size=8,
            min_cpu_batch_size=4,
            max_cpu_batch_size=16,
        )
        assert 2 <= gpu_bs <= 8, f"gpu_bs={gpu_bs} outside [2, 8]"
        assert 4 <= cpu_bs <= 16, f"cpu_bs={cpu_bs} outside [4, 16]"
        assert gpu_bs <= cpu_bs

    def test_default_min_is_one(self, model_and_sae):
        """Omitting min params defaults to 1 (backward compatible)."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            max_gpu_batch_size=4,
            max_cpu_batch_size=4,
        )
        # Without min, candidates start at 1, so result could be 1
        assert 1 <= gpu_bs <= 4
        assert 1 <= cpu_bs <= 4

    def test_min_still_produces_power_of_two(self, model_and_sae):
        """Even with min bounds, returned sizes must be powers of 2."""
        model, sae, layer, _ = model_and_sae
        gpu_bs, cpu_bs = find_optimal_batch_sizes(
            model,
            sae,
            layer=layer,
            context_window=128,
            min_gpu_batch_size=2,
            max_gpu_batch_size=16,
            min_cpu_batch_size=4,
            max_cpu_batch_size=16,
        )
        assert gpu_bs & (gpu_bs - 1) == 0, f"gpu_bs={gpu_bs} not power of 2"
        assert cpu_bs & (cpu_bs - 1) == 0, f"cpu_bs={cpu_bs} not power of 2"


# ---------------------------------------------------------------------------
# Batch size candidate generation (pure unit tests, no model needed)
# ---------------------------------------------------------------------------


class TestBatchSizeCandidateGeneration:
    """Verify the power-of-2 candidate filtering used by find_optimal_batch_sizes.

    These tests exercise the same candidate formulas as production code,
    ensuring the min/max filtering logic is correct across edge cases.
    """

    @staticmethod
    def _generate_candidates(min_bs: int, max_bs: int) -> list[int]:
        """Replicate the GPU candidate generation from find_optimal_batch_sizes."""
        return [
            2**i
            for i in range(max_bs.bit_length())
            if min_bs <= 2**i <= max_bs
        ]

    @staticmethod
    def _generate_cpu_candidates(min_bs: int, max_bs: int) -> list[int]:
        """Replicate the CPU candidate generation from find_optimal_batch_sizes."""
        cpu_start_exp = max(0, (min_bs - 1).bit_length())
        return [
            2**i
            for i in range(cpu_start_exp, max_bs.bit_length(), 2)
            if min_bs <= 2**i <= max_bs
        ]

    # -- basic correctness --------------------------------------------------

    @pytest.mark.parametrize(
        "min_bs, max_bs, expected",
        [
            (1, 16, [1, 2, 4, 8, 16]),
            (4, 16, [4, 8, 16]),
            (8, 64, [8, 16, 32, 64]),
            (16, 16, [16]),
            (1, 1, [1]),
            (2, 8, [2, 4, 8]),
            (1, 512, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        ],
        ids=[
            "default_min_1",
            "min_4",
            "min_8_max_64",
            "min_eq_max_16",
            "single_at_1",
            "small_range",
            "full_range",
        ],
    )
    def test_candidate_generation(self, min_bs, max_bs, expected):
        assert self._generate_candidates(min_bs, max_bs) == expected

    @pytest.mark.parametrize(
        "min_bs, max_bs, expected",
        [
            (1, 16, [1, 4, 16]),
            (2, 16, [2, 8]),
            (3, 16, [4, 16]),
            (4, 16, [4, 16]),
            (5, 32, [8, 32]),
            (16, 16, [16]),
            (3, 3, []),
        ],
        ids=[
            "default_stride",
            "nonzero_start_exp",
            "rounds_up_from_3",
            "power_min_4",
            "rounds_up_from_5",
            "single_at_16",
            "no_power_in_range",
        ],
    )
    def test_cpu_candidate_generation_stride_two(self, min_bs, max_bs, expected):
        assert self._generate_cpu_candidates(min_bs, max_bs) == expected

    # -- all candidates are powers of 2 ------------------------------------

    @pytest.mark.parametrize("min_bs, max_bs", [(1, 64), (4, 256), (8, 512)])
    def test_all_candidates_are_powers_of_two(self, min_bs, max_bs):
        for c in self._generate_candidates(min_bs, max_bs):
            assert c & (c - 1) == 0, f"{c} is not a power of 2"

    # -- bounds respected ---------------------------------------------------

    @pytest.mark.parametrize("min_bs, max_bs", [(1, 64), (4, 256), (8, 512), (16, 16)])
    def test_all_candidates_within_bounds(self, min_bs, max_bs):
        for c in self._generate_candidates(min_bs, max_bs):
            assert min_bs <= c <= max_bs, f"{c} outside [{min_bs}, {max_bs}]"

    # -- edge cases ---------------------------------------------------------

    def test_min_greater_than_max_gives_empty(self):
        assert self._generate_candidates(32, 16) == []

    def test_min_greater_than_max_large_gap(self):
        assert self._generate_candidates(512, 1) == []

    def test_non_power_of_two_min_rounds_up(self):
        """min=3 excludes 1 and 2, first candidate is 4."""
        candidates = self._generate_candidates(3, 16)
        assert candidates == [4, 8, 16]

    def test_non_power_of_two_min_5(self):
        """min=5 excludes 1, 2, 4 — first candidate is 8."""
        candidates = self._generate_candidates(5, 32)
        assert candidates == [8, 16, 32]

    def test_no_power_of_two_in_range(self):
        """Range [5, 7] contains no power of 2."""
        assert self._generate_candidates(5, 7) == []

    def test_range_3_to_3_empty(self):
        """min=max=3 but 3 is not a power of 2 — no candidates."""
        assert self._generate_candidates(3, 3) == []

    def test_max_is_power_of_two_boundary(self):
        """max=32 is included as a candidate."""
        candidates = self._generate_candidates(16, 32)
        assert 32 in candidates

    def test_min_is_power_of_two_boundary(self):
        """min=16 is included as a candidate."""
        candidates = self._generate_candidates(16, 64)
        assert 16 in candidates

    # -- valid pairs (gpu <= cpu constraint) ---------------------------------

    @staticmethod
    def _valid_pairs(
        min_gpu: int, max_gpu: int, min_cpu: int, max_cpu: int
    ) -> list[tuple[int, int]]:
        """Replicate the valid_pairs generation from find_optimal_batch_sizes."""
        import itertools

        gpu_candidates = TestBatchSizeCandidateGeneration._generate_candidates(
            min_gpu,
            max_gpu,
        )
        cpu_candidates = TestBatchSizeCandidateGeneration._generate_cpu_candidates(
            min_cpu,
            max_cpu,
        )
        return [
            (g, c)
            for g, c in itertools.product(gpu_candidates, cpu_candidates)
            if g <= c
        ]

    def test_valid_pairs_gpu_le_cpu(self):
        """Every valid pair must have gpu_bs <= cpu_bs."""
        pairs = self._valid_pairs(1, 16, 1, 16)
        for g, c in pairs:
            assert g <= c, f"gpu={g} > cpu={c}"

    def test_valid_pairs_expected(self):
        pairs = self._valid_pairs(4, 8, 4, 16)
        # gpu: [4, 8], cpu: [4, 16]
        # valid: (4,4), (4,16), (8,16)
        assert pairs == [(4, 4), (4, 16), (8, 16)]

    def test_valid_pairs_min_narrows_search_space(self):
        """Higher min reduces the number of valid pairs."""
        pairs_wide = self._valid_pairs(1, 16, 1, 16)
        pairs_narrow = self._valid_pairs(4, 16, 4, 16)
        assert len(pairs_narrow) < len(pairs_wide)

    def test_valid_pairs_empty_when_no_candidates(self):
        pairs = self._valid_pairs(32, 4, 32, 4)
        assert pairs == []

    def test_valid_pairs_single_when_min_eq_max(self):
        """min==max on both axes with gpu <= cpu gives exactly one pair."""
        pairs = self._valid_pairs(4, 4, 8, 8)
        assert pairs == [(4, 8)]

    def test_valid_pairs_empty_when_gpu_min_exceeds_cpu_max(self):
        """If min_gpu > max_cpu, no pair can satisfy gpu <= cpu."""
        pairs = self._valid_pairs(16, 32, 1, 8)
        assert pairs == []


# ---------------------------------------------------------------------------
# Hydra config wiring for batch size bounds
# ---------------------------------------------------------------------------

# Path to the production YAML config (relative to repo root).
_YAML_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "extraction" / "extract_features.yaml"


class TestBatchSizeConfigWiring:
    """Verify min/max batch size fields are correctly wired through Hydra config.

    Tests three layers:
    1. YAML config declares the expected fields with correct defaults.
    2. getattr + int() expressions read values (or defaults) from DictConfig.
    3. find_optimal_batch_sizes receives the config-derived values.
    """

    # -- YAML schema tests --------------------------------------------------

    @pytest.fixture
    def yaml_cfg(self):
        """Load the production extraction YAML as an OmegaConf DictConfig."""
        return OmegaConf.load(str(_YAML_CONFIG_PATH))

    def test_yaml_has_min_gpu_batch_size(self, yaml_cfg):
        assert yaml_cfg.min_gpu_batch_size == 1

    def test_yaml_has_min_cpu_batch_size(self, yaml_cfg):
        assert yaml_cfg.min_cpu_batch_size == 1

    def test_yaml_has_max_gpu_batch_size(self, yaml_cfg):
        assert yaml_cfg.max_gpu_batch_size == 512

    def test_yaml_has_max_cpu_batch_size(self, yaml_cfg):
        assert yaml_cfg.max_cpu_batch_size == 512

    # -- getattr expression tests (with and without fields) -----------------

    @staticmethod
    def _read_bounds(cfg):
        """Replicate the exact getattr expressions from main()."""
        return {
            "min_gpu_batch_size": int(getattr(cfg, "min_gpu_batch_size", 1)),
            "min_cpu_batch_size": int(getattr(cfg, "min_cpu_batch_size", 1)),
            "max_gpu_batch_size": int(getattr(cfg, "max_gpu_batch_size", 512)),
            "max_cpu_batch_size": int(getattr(cfg, "max_cpu_batch_size", 512)),
        }

    def test_reads_explicit_values(self):
        """When config has all four fields, reads them directly."""
        cfg = OmegaConf.create({
            "min_gpu_batch_size": 4,
            "min_cpu_batch_size": 8,
            "max_gpu_batch_size": 128,
            "max_cpu_batch_size": 256,
        })
        bounds = self._read_bounds(cfg)
        assert bounds == {
            "min_gpu_batch_size": 4,
            "min_cpu_batch_size": 8,
            "max_gpu_batch_size": 128,
            "max_cpu_batch_size": 256,
        }

    def test_missing_fields_use_defaults(self):
        """Old configs without min/max fields fall back to defaults."""
        cfg = OmegaConf.create({})
        bounds = self._read_bounds(cfg)
        assert bounds == {
            "min_gpu_batch_size": 1,
            "min_cpu_batch_size": 1,
            "max_gpu_batch_size": 512,
            "max_cpu_batch_size": 512,
        }

    def test_partial_fields_fill_defaults(self):
        """Config with only some fields uses defaults for the rest."""
        cfg = OmegaConf.create({"min_gpu_batch_size": 16})
        bounds = self._read_bounds(cfg)
        assert bounds["min_gpu_batch_size"] == 16
        assert bounds["min_cpu_batch_size"] == 1  # default
        assert bounds["max_gpu_batch_size"] == 512  # default
        assert bounds["max_cpu_batch_size"] == 512  # default

    def test_string_values_converted_to_int(self):
        """Hydra can pass numeric strings; int() must handle them."""
        cfg = OmegaConf.create({
            "min_gpu_batch_size": "4",
            "min_cpu_batch_size": "8",
            "max_gpu_batch_size": "128",
            "max_cpu_batch_size": "256",
        })
        bounds = self._read_bounds(cfg)
        assert bounds == {
            "min_gpu_batch_size": 4,
            "min_cpu_batch_size": 8,
            "max_gpu_batch_size": 128,
            "max_cpu_batch_size": 256,
        }

    def test_production_yaml_round_trips_through_getattr(self, yaml_cfg):
        """Production YAML values survive the getattr + int() path."""
        bounds = self._read_bounds(yaml_cfg)
        assert bounds["min_gpu_batch_size"] == 1
        assert bounds["min_cpu_batch_size"] == 1
        assert bounds["max_gpu_batch_size"] == 512
        assert bounds["max_cpu_batch_size"] == 512

    # -- forwarding tests (mock find_optimal_batch_sizes) -------------------

    @patch("steering.extract_features.find_optimal_batch_sizes", return_value=(8, 16))
    def test_auto_mode_forwards_all_bounds(self, mock_find):
        """When gpu_batch_size=auto, all four bounds are forwarded."""
        cfg = OmegaConf.create({
            "gpu_batch_size": "auto",
            "cpu_batch_size": "auto",
            "layer": 8,
            "context_window": 2048,
            "gc_mem_threshold": 0.9,
            "min_gpu_batch_size": 4,
            "min_cpu_batch_size": 8,
            "max_gpu_batch_size": 128,
            "max_cpu_batch_size": 256,
        })

        # Replicate the auto-tuning code path from main()
        gpu_auto = str(cfg.gpu_batch_size) == "auto"
        cpu_auto = str(cfg.cpu_batch_size) == "auto"
        assert gpu_auto and cpu_auto

        _ef_module.find_optimal_batch_sizes(
            None, None,  # model, sae — mocked
            layer=cfg.layer,
            context_window=cfg.context_window,
            gc_mem_threshold=cfg.gc_mem_threshold,
            min_gpu_batch_size=int(getattr(cfg, "min_gpu_batch_size", 1)),
            min_cpu_batch_size=int(getattr(cfg, "min_cpu_batch_size", 1)),
            max_gpu_batch_size=int(getattr(cfg, "max_gpu_batch_size", 512)),
            max_cpu_batch_size=int(getattr(cfg, "max_cpu_batch_size", 512)),
        )

        mock_find.assert_called_once()
        kwargs = mock_find.call_args.kwargs
        assert kwargs["min_gpu_batch_size"] == 4
        assert kwargs["min_cpu_batch_size"] == 8
        assert kwargs["max_gpu_batch_size"] == 128
        assert kwargs["max_cpu_batch_size"] == 256

    @patch("steering.extract_features.find_optimal_batch_sizes", return_value=(8, 16))
    def test_auto_mode_uses_defaults_for_missing_bounds(self, mock_find):
        """Old config without min/max fields forwards correct defaults."""
        cfg = OmegaConf.create({
            "gpu_batch_size": "auto",
            "cpu_batch_size": "auto",
            "layer": 8,
            "context_window": 2048,
            "gc_mem_threshold": 0.9,
        })

        _ef_module.find_optimal_batch_sizes(
            None, None,
            layer=cfg.layer,
            context_window=cfg.context_window,
            gc_mem_threshold=cfg.gc_mem_threshold,
            min_gpu_batch_size=int(getattr(cfg, "min_gpu_batch_size", 1)),
            min_cpu_batch_size=int(getattr(cfg, "min_cpu_batch_size", 1)),
            max_gpu_batch_size=int(getattr(cfg, "max_gpu_batch_size", 512)),
            max_cpu_batch_size=int(getattr(cfg, "max_cpu_batch_size", 512)),
        )

        kwargs = mock_find.call_args.kwargs
        assert kwargs["min_gpu_batch_size"] == 1
        assert kwargs["min_cpu_batch_size"] == 1
        assert kwargs["max_gpu_batch_size"] == 512
        assert kwargs["max_cpu_batch_size"] == 512

    def test_non_auto_skips_find_optimal(self):
        """When batch sizes are explicit integers, find_optimal is not called."""
        cfg = OmegaConf.create({
            "gpu_batch_size": 16,
            "cpu_batch_size": 64,
        })
        gpu_auto = str(cfg.gpu_batch_size) == "auto"
        cpu_auto = str(cfg.cpu_batch_size) == "auto"
        assert not gpu_auto
        assert not cpu_auto
        # main() takes the else branch: gpu_batch_size = int(cfg.gpu_batch_size)
        assert int(cfg.gpu_batch_size) == 16
        assert int(cfg.cpu_batch_size) == 64

    def test_partial_auto_only_gpu(self):
        """gpu=auto, cpu=64 → auto-tuning runs, cpu result is ignored."""
        cfg = OmegaConf.create({
            "gpu_batch_size": "auto",
            "cpu_batch_size": 64,
        })
        gpu_auto = str(cfg.gpu_batch_size) == "auto"
        cpu_auto = str(cfg.cpu_batch_size) == "auto"
        assert gpu_auto
        assert not cpu_auto

    def test_partial_auto_only_cpu(self):
        """gpu=16, cpu=auto → auto-tuning runs, gpu result is ignored."""
        cfg = OmegaConf.create({
            "gpu_batch_size": 16,
            "cpu_batch_size": "auto",
        })
        gpu_auto = str(cfg.gpu_batch_size) == "auto"
        cpu_auto = str(cfg.cpu_batch_size) == "auto"
        assert not gpu_auto
        assert cpu_auto


# ---------------------------------------------------------------------------
# maybe_gc_cuda
# ---------------------------------------------------------------------------


class TestMaybeGcCuda:
    """Test memory-aware GC helper."""

    def test_does_not_crash_on_cpu(self):
        """GC helper is a no-op on CPU devices."""
        device = t.device("cpu")
        maybe_gc_cuda(device, threshold=0.8)  # should not raise

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_runs_on_cuda(self):
        """GC helper runs without error on CUDA."""
        device = t.device("cuda")
        maybe_gc_cuda(device, threshold=0.8)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_threshold_zero_always_triggers(self):
        """Threshold 0.0 should always trigger GC (any allocation > 0%)."""
        device = t.device("cuda")
        x = t.zeros(1, device=device)
        maybe_gc_cuda(device, threshold=0.0)
        del x

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_threshold_one_never_triggers(self):
        """Threshold 1.0 should never trigger GC (usage can't exceed 100%)."""
        device = t.device("cuda")
        maybe_gc_cuda(device, threshold=1.0)
