"""Unit tests for steering.eval.multiseg_generator.generate_samples_for_condition.

The P1.1 helper that amortizes encoder forward + prefill across samples.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
import steering.eval.multiseg_generator as mg_mod
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.multiseg_generator import (
    MultisegInputs,
    generate_completion,
    generate_samples_for_condition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture
def tiny_model():
    """A tiny EncoderDecoderGPT: 2 layers, dim=64, vocab=256, seq_len=128."""
    dec_config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=50,
        num_buckets=5,
    )
    model.init_weights()
    model.eval()
    return model


def _make_encoder_inputs(n_prefix: int, n_gen: int, n_seg: int = 2, F_max: int = 3):
    """Build a MultisegInputs with deterministic contents."""
    T = n_prefix + n_gen
    seg_map = torch.zeros(1, T, dtype=torch.int64)
    # Spread prefix across segments; put all generated tokens in the last segment
    half = max(1, n_prefix // 2)
    seg_map[0, half:n_prefix] = 1
    seg_map[0, n_prefix:] = n_seg - 1

    enc_feature_ids = torch.zeros(1, n_seg, F_max, dtype=torch.int64)
    enc_magnitudes = torch.zeros(1, n_seg, F_max, dtype=torch.int64)
    enc_mask = torch.zeros(1, n_seg, F_max, dtype=torch.bool)
    # Populate a few features so the encoder output is non-trivial
    enc_feature_ids[0, 0, :] = torch.tensor([1, 2, 3], dtype=torch.int64)
    enc_magnitudes[0, 0, :] = torch.tensor([1, 2, 3], dtype=torch.int64)
    enc_mask[0, 0, :] = True

    return MultisegInputs(
        enc_feature_ids=enc_feature_ids,
        enc_magnitudes=enc_magnitudes,
        enc_mask=enc_mask,
        seg_map=seg_map,
    )


# ---------------------------------------------------------------------------
# Basic return-shape tests
# ---------------------------------------------------------------------------


def test_returns_list_of_token_lists(tiny_model):
    """Returns a list of n_samples token lists."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    result = generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=3,
        device=torch.device("cpu"),
        temperature=0.5,
        top_k=20,
        base_seed=42,
        bos_id=0,
    )
    assert isinstance(result, list)
    assert len(result) == 3
    for gen_tokens in result:
        assert isinstance(gen_tokens, list)
        assert all(isinstance(t, int) for t in gen_tokens)


def test_n_samples_zero_returns_empty_without_touching_model(tiny_model, monkeypatch):
    """n_samples=0 early-exits to [] without running the encoder."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)

    # Monkeypatch encoder forward to raise — if called, the early-exit was skipped
    original_encoder_forward = tiny_model.encoder.forward

    def _boom(*args, **kwargs):
        raise AssertionError("model.encoder.forward should not be called when n_samples=0")

    tiny_model.encoder.forward = _boom  # type: ignore[assignment]
    try:
        result = generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=[1, 2, 3],
            encoder_inputs=enc_inputs,
            n_samples=0,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )
    finally:
        tiny_model.encoder.forward = original_encoder_forward  # type: ignore[assignment]

    assert result == []


def test_n_samples_one_matches_generate_completion_token_for_token(tiny_model):
    """CENTRAL PARITY ANCHOR: n_samples=1 produces identical tokens to generate_completion.

    This is the most important test in the file. If it fails, the helper's
    prefill/decode path has drifted from generate_completion.
    """
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=6)
    prefix_tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    seed = 1234
    n_gen = enc_inputs.seg_map.shape[1] - len(prefix_tokens)

    # Run via the new helper
    torch.manual_seed(0)
    batched_result = generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=1,
        device=torch.device("cpu"),
        temperature=0.7,
        top_k=20,
        base_seed=seed,
        bos_id=0,
    )
    # Run via the reference function
    torch.manual_seed(0)
    reference_tokens = generate_completion(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        max_new_tokens=n_gen,
        device=torch.device("cpu"),
        temperature=0.7,
        top_k=20,
        seed=seed,
        bos_id=0,
    )

    assert len(batched_result) == 1
    assert batched_result[0] == reference_tokens


def test_n_samples_two_produces_different_token_lists(tiny_model):
    """Two samples with temperature=1.0 must diverge — proves KV cache clones are isolated."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=8)
    prefix_tokens = [5, 10, 15, 20, 25, 30, 35, 40]

    result = generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=2,
        device=torch.device("cpu"),
        temperature=1.0,  # high temperature to force divergence
        top_k=50,
        base_seed=42,
        bos_id=0,
    )
    assert len(result) == 2
    assert result[0] != result[1], (
        f"Samples must differ but got identical: {result[0]}"
    )


# ---------------------------------------------------------------------------
# Determinism + seed sequence tests
# ---------------------------------------------------------------------------


def test_determinism_across_two_calls(tiny_model):
    """Two sequential calls with identical arguments produce equal lists."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=5)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    kwargs = {
        "model": tiny_model,
        "prefix_tokens": prefix_tokens,
        "encoder_inputs": enc_inputs,
        "n_samples": 3,
        "device": torch.device("cpu"),
        "temperature": 0.5,
        "top_k": 20,
        "base_seed": 100,
        "bos_id": 0,
    }
    a = generate_samples_for_condition(**kwargs)
    b = generate_samples_for_condition(**kwargs)
    assert a == b


def test_seed_sequence_is_base_plus_i(tiny_model):
    """Sample i uses seed base_seed + i. Verify by comparing against
    independent generate_completion calls with explicit seeds."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=5)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    base_seed = 500
    n_gen = enc_inputs.seg_map.shape[1] - len(prefix_tokens)

    batched = generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=3,
        device=torch.device("cpu"),
        temperature=0.7,
        top_k=20,
        base_seed=base_seed,
        bos_id=0,
    )

    for i in range(3):
        reference = generate_completion(
            model=tiny_model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=enc_inputs,
            max_new_tokens=n_gen,
            device=torch.device("cpu"),
            temperature=0.7,
            top_k=20,
            seed=base_seed + i,
            bos_id=0,
        )
        assert batched[i] == reference, f"Sample {i} differs from seed={base_seed + i}"


# ---------------------------------------------------------------------------
# Amortization tests (encoder + prefill each run once)
# ---------------------------------------------------------------------------


def test_encoder_forward_called_exactly_once_for_n_samples_five(tiny_model):
    """With n_samples=5, model.encoder.forward is invoked exactly 1 time."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)

    original_forward = tiny_model.encoder.forward
    call_count = {"n": 0}

    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    tiny_model.encoder.forward = counting_forward  # type: ignore[assignment]
    try:
        generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
            encoder_inputs=enc_inputs,
            n_samples=5,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )
    finally:
        tiny_model.encoder.forward = original_forward  # type: ignore[assignment]

    assert call_count["n"] == 1, (
        f"Encoder forward called {call_count['n']} times, expected exactly 1 for n_samples=5"
    )


def test_prefill_condition_called_exactly_once_for_n_samples_five(tiny_model, monkeypatch):
    """With n_samples=5, mg_mod._prefill_condition is invoked exactly 1 time."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)

    original = mg_mod._prefill_condition
    call_count = {"n": 0}

    def counting_prefill(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mg_mod, "_prefill_condition", counting_prefill)

    generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        encoder_inputs=enc_inputs,
        n_samples=5,
        device=torch.device("cpu"),
        base_seed=42,
        bos_id=0,
    )
    assert call_count["n"] == 1, (
        f"_prefill_condition called {call_count['n']} times, expected exactly 1 for n_samples=5"
    )


def test_base_kv_cache_not_advanced_by_decode_steps(tiny_model, monkeypatch):
    """Decode writes land on the cloned sample caches, not the base cache.

    Protects against KV cache clone aliasing: if sample_kv.prefill(base_kv_cache)
    shared underlying storage, the decode loop would advance base_kv_cache's
    cache_seqlens beyond n_prefix. We intercept KVCache.__init__ to record the
    first (base) cache instance and verify it stayed at n_prefix after the helper returns.
    """
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=5)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    original_init = KVCache.__init__
    base_cache_holder: dict = {}

    def _recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The first KVCache created inside generate_samples_for_condition is
        # the base cache. Subsequent ones are sample clones.
        if "first" not in base_cache_holder:
            base_cache_holder["first"] = self

    monkeypatch.setattr(KVCache, "__init__", _recording_init)

    generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=3,
        device=torch.device("cpu"),
        base_seed=42,
        bos_id=0,
    )

    base_cache = base_cache_holder["first"]
    # After prefill, cache_seqlens should equal n_prefix. The decode loop
    # runs on cloned caches, so the base should never see n_prefix + k writes.
    assert int(base_cache.cache_seqlens.item()) == len(prefix_tokens), (
        f"Base KV cache was advanced beyond prefill: cache_seqlens="
        f"{int(base_cache.cache_seqlens.item())}, expected {len(prefix_tokens)}"
    )


# ---------------------------------------------------------------------------
# Prefix handling and edge cases
# ---------------------------------------------------------------------------


def test_oversized_prefix_raises(tiny_model):
    """A prefix longer than seg_map total raises ValueError (no silent truncation).

    Earlier this test asserted the helper silently truncated long prefixes.
    The user explicitly removed the fallback / truncation: the helper must
    now raise when the prefix is too long for the seg_map.
    """
    # seg_map total T_seg = 120 + 4 = 124
    enc_inputs = _make_encoder_inputs(n_prefix=120, n_gen=4)
    # Prefix length 150 >= T_seg=124 → must raise
    long_prefix = list(range(1, 151))  # 150 tokens

    with pytest.raises(ValueError, match=r"prefix_tokens length 150 must be strictly less than"):
        generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=long_prefix,
            encoder_inputs=enc_inputs,
            n_samples=1,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )


def test_seg_map_exceeding_max_seq_len_raises(tiny_model):
    """A seg_map longer than the decoder's max_seq_len raises ValueError.

    The decoder context window must contain the full seg_map; otherwise
    the prefill would overflow. The helper raises rather than silently
    truncating.
    """
    # tiny_model.dec_config.sequence_len == 128
    # T_seg = 200 + 4 = 204 > 128
    enc_inputs = _make_encoder_inputs(n_prefix=200, n_gen=4)
    prefix_tokens = list(range(1, 51))  # n_prefix=50, less than T_seg

    with pytest.raises(ValueError, match=r"exceeds the decoder's max_seq_len"):
        generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=enc_inputs,
            n_samples=1,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )


def test_prefix_equal_to_seg_total_raises(tiny_model):
    """prefix_tokens length == T_seg leaves zero generation slots → must raise."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)  # T_seg=12
    prefix_tokens = list(range(1, 13))  # 12 tokens, exactly T_seg

    with pytest.raises(ValueError, match=r"strictly less than"):
        generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=enc_inputs,
            n_samples=1,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )


def test_empty_prefix_raises_clear_error(tiny_model):
    """prefix_tokens=[] is a misuse — helper must raise a clear ValueError."""
    enc_inputs = _make_encoder_inputs(n_prefix=0, n_gen=4)

    with pytest.raises(ValueError, match=r"prefix_tokens is empty"):
        generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=[],
            encoder_inputs=enc_inputs,
            n_samples=1,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )


def test_inference_mode_is_enforced(tiny_model):
    """Helper runs under @torch.inference_mode() — returned tokens computed without grad."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    # Enter grad mode outside — the decorator must still disable grad inside.
    with torch.enable_grad():
        result = generate_samples_for_condition(
            model=tiny_model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=enc_inputs,
            n_samples=1,
            device=torch.device("cpu"),
            base_seed=42,
            bos_id=0,
        )
    assert isinstance(result, list)
    assert len(result) == 1


def test_dtype_fp32_on_cpu(tiny_model, monkeypatch):
    """KV cache dtype is float32 on CPU (bf16 branch only applies on CUDA)."""
    enc_inputs = _make_encoder_inputs(n_prefix=8, n_gen=4)
    prefix_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    original_init = KVCache.__init__
    captured_dtypes: list[torch.dtype] = []

    def _capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured_dtypes.append(self.k_cache.dtype)

    monkeypatch.setattr(KVCache, "__init__", _capturing_init)

    # CPU path — dtype must be float32
    generate_samples_for_condition(
        model=tiny_model,
        prefix_tokens=prefix_tokens,
        encoder_inputs=enc_inputs,
        n_samples=1,
        device=torch.device("cpu"),
        base_seed=42,
        bos_id=0,
    )
    assert len(captured_dtypes) >= 2  # base + at least one sample clone
    assert all(dt == torch.float32 for dt in captured_dtypes), (
        f"Expected all float32 on CPU, got {captured_dtypes}"
    )
