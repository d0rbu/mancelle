"""Memory usage measurement tests for feature extraction.

Run with:
    uv run pytest tests/test_extraction_memory.py -v --model-scale=tiny -s
"""

import tracemalloc
from unittest.mock import MagicMock

import pytest
import torch as t

from steering.extract_features import (
    PoolingMode,
    extract_document_features,
)

# ---------------------------------------------------------------------------
# Tiny model fixtures (self-contained, no dependency on integration tests)
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
    """Minimal HookedTransformer substitute for memory tests."""

    def __init__(self, d_model: int = 64, vocab_size: int = 1000, device: str = "cpu"):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._device = device
        self._dummy_param = t.nn.Parameter(t.zeros(1, device=device))

        self.tokenizer = MagicMock()
        self.tokenizer.encode = self._encode
        self.tokenizer.decode = lambda tids: "".join(
            chr(tid % 128) if tid < 128 else f"<{tid}>"
            for tid in (tids if isinstance(tids, list) else [tids])
        )
        # BOS token id exposed for ci extraction. 0 matches
        # conftest.MockTokenizer._bos_id convention.
        self.tokenizer.bos_token_id = 0
        self.tokenizer.side_effect = lambda texts, **kw: {
            "input_ids": [self._encode(text) for text in texts]
        }

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

    def run_with_cache(
        self,
        tokens: t.Tensor,
        names_filter: list[str],
        attention_mask: t.Tensor | None = None,
        stop_at_layer: int | None = None,
    ):
        del attention_mask, stop_at_layer
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


@pytest.fixture(scope="session")
def model_and_sae():
    """Tiny model + SAE for memory tests (always uses tiny scale)."""
    model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    return model, sae, 8, 256


# ---------------------------------------------------------------------------
# CPU memory tests
# ---------------------------------------------------------------------------


class TestExtractionMemoryCPU:
    """Measure CPU memory during extraction with tracemalloc."""

    def test_extract_document_memory_bounded(self, model_and_sae):
        """Peak CPU memory for a single doc extraction stays under threshold."""
        model, sae, layer, n_features = model_and_sae
        text = "The quick brown fox. " * 100  # ~500 tokens

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[1]

        extract_document_features(
            model,
            sae,
            text,
            layer=layer,
            context_window=1024,
            pooling=PoolingMode.mean_pool,
        )

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        delta_mb = (peak - baseline) / 1024 / 1024
        print(f"\n  Peak CPU memory delta: {delta_mb:.1f} MB")
        assert delta_mb < 500, f"CPU memory spike too large: {delta_mb:.1f} MB"


# ---------------------------------------------------------------------------
# GPU memory tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
class TestExtractionMemoryGPU:
    """Measure GPU memory during extraction."""

    def test_extract_document_gpu_memory(self, model_and_sae):
        """Peak GPU memory stays bounded during extraction."""
        model, sae, layer, _ = model_and_sae
        text = "The quick brown fox. " * 100

        t.cuda.reset_peak_memory_stats()
        before = t.cuda.max_memory_allocated()

        extract_document_features(
            model,
            sae,
            text,
            layer=layer,
            context_window=1024,
            pooling=PoolingMode.mean_pool,
        )

        peak = t.cuda.max_memory_allocated()
        delta_mb = (peak - before) / 1024 / 1024
        print(f"\n  Peak GPU memory delta: {delta_mb:.1f} MB")
        assert delta_mb < 2000, f"GPU memory spike too large: {delta_mb:.1f} MB"
