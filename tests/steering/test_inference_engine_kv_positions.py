"""Backend selection and native SDPA parity for host-resident KV positions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch as t

from nanochat import flash_attention
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedDecodeProvider,
    SelfCondState,
    self_conditioned_decode_step,
)
from steering.eval.inference_engine import InferenceEngine, stack_prefilled_kv_caches
from steering.segmentation import SegmentMode
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.slow),
]


def _engine(device: str) -> InferenceEngine:
    return InferenceEngine(
        SimpleNamespace(config=SimpleNamespace(n_kv_head=2, n_embd=16, n_head=2, n_layer=2)),
        SimpleNamespace(get_bos_token_id=lambda: 0),
        t.device(device),
    )


def _reference(cache: KVCache) -> KVCache:
    return KVCache(
        batch_size=cache.batch_size,
        num_heads=cache.n_heads,
        seq_len=cache.max_seq_len,
        head_dim=cache.head_dim,
        num_layers=cache.n_layers,
        device=cache.k_cache.device,
        dtype=cache.k_cache.dtype,
    )


def _assert_equal(actual: KVCache, expected: KVCache) -> None:
    assert type(actual) is KVCache
    assert actual.cache_seqlens.device.type == "cpu"
    assert actual.cache_seqlens.dtype == t.int32
    assert actual.get_pos() == expected.get_pos()
    assert t.equal(actual.cache_seqlens, expected.cache_seqlens.cpu())
    assert t.equal(actual.k_cache, expected.k_cache)
    assert t.equal(actual.v_cache, expected.v_cache)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("has_fa3", "override", "expected_backend"),
    [
        (False, None, "sdpa"),
        (False, FlashAttnImpl.AUTO, "sdpa"),
        (False, FlashAttnImpl.FA3, "sdpa"),
        (False, FlashAttnImpl.FA4, "sdpa"),
        (False, FlashAttnImpl.SDPA, "sdpa"),
        (True, None, "fa3"),
        (True, FlashAttnImpl.AUTO, "fa3"),
        (True, FlashAttnImpl.FA3, "fa3"),
        (True, FlashAttnImpl.FA4, "fa3"),
        (True, FlashAttnImpl.SDPA, "sdpa"),
    ],
)
def test_position_allocation_matches_actual_kv_backend_without_cuda(
    monkeypatch, device, has_fa3, override, expected_backend
):
    """Use meta storage only to inspect CUDA allocation policy without a GPU."""
    monkeypatch.setattr(flash_attention, "HAS_FA3", has_fa3)
    monkeypatch.setattr(flash_attention, "_override_impl", override)
    original_zeros = t.zeros
    allocations = []

    def zeros(*args, **kwargs):
        requested_device = t.device(kwargs.get("device", "cpu"))
        if requested_device.type == "cuda":
            kwargs["device"] = "meta"
        result = original_zeros(*args, **kwargs)
        allocations.append((requested_device, result))
        return result

    monkeypatch.setattr(t, "zeros", zeros)
    cache = _engine(device)._make_kv_cache(batch_size=3, seq_len=9)
    move_position = device == "cuda" and expected_backend == "sdpa"
    assert type(cache) is KVCache
    assert len(allocations) == (4 if move_position else 3)
    assert cache.k_cache is allocations[0][1]
    assert cache.v_cache is allocations[1][1]
    assert cache.cache_seqlens is allocations[-1][1]
    assert allocations[-1][0].type == ("cpu" if move_position else device)
    assert cache.cache_seqlens.shape == (3,)
    assert cache.cache_seqlens.dtype == t.int32

    dispatched = []

    def fa3(q, _k_cache, _v_cache, **kwargs):
        # Native FA3 requires device-resident position counters. Meta stands in
        # for CUDA here, so moving them to CPU would violate this assertion.
        assert kwargs["cache_seqlens"].device == q.device
        dispatched.append("fa3")
        return t.empty_like(q)

    def sdpa(q, _k, _v, _window_size, _enable_gqa):
        # The actual dispatcher has already read the position and checked the
        # batch for uniformity. Meta counters cannot satisfy either operation.
        assert cache.cache_seqlens.device.type == "cpu"
        dispatched.append("sdpa")
        return t.empty_like(q)

    monkeypatch.setattr(flash_attention, "_fa3", SimpleNamespace(flash_attn_with_kvcache=fa3))
    monkeypatch.setattr(flash_attention, "_sdpa_attention", sdpa)
    q = t.empty((3, 1, 2, 8), device=cache.k_cache.device, dtype=cache.k_cache.dtype)
    output = flash_attention.flash_attn_with_kvcache(
        q,
        *cache.get_layer_cache(0),
        k=q,
        v=q,
        cache_seqlens=cache.cache_seqlens,
        causal=True,
    )
    assert output.shape == q.shape
    assert dispatched == [expected_backend]


@pytest.mark.parametrize("device", _DEVICES)
@t.inference_mode()
def test_native_sdpa_positions_match_through_cache_lifecycle(monkeypatch, device):
    if device == "cuda" and not t.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    engine = _engine(device)
    rng = t.Generator().manual_seed(418)

    def forward_pair(actual, expected, length):
        for layer in range(actual.n_layers):
            q, k, v = [
                t.randn(
                    actual.batch_size,
                    length,
                    actual.n_heads,
                    actual.head_dim,
                    generator=rng,
                ).to(device=device, dtype=actual.k_cache.dtype)
                for _ in range(3)
            ]
            outputs = [
                flash_attention.flash_attn_with_kvcache(
                    q,
                    *cache.get_layer_cache(layer),
                    k=k,
                    v=v,
                    cache_seqlens=cache.cache_seqlens,
                    causal=True,
                )
                for cache in (expected, actual)
            ]
            assert t.equal(*outputs)
        actual.advance(length)
        expected.advance(length)
        _assert_equal(actual, expected)

    actual = engine._make_kv_cache(batch_size=2, seq_len=12)
    expected = _reference(actual)
    assert actual.k_cache.device.type == device
    assert actual.v_cache.device.type == device
    assert expected.cache_seqlens.device.type == device
    for length in (3, 1, 2):
        forward_pair(actual, expected, length)
    actual.rewind(2)
    expected.rewind(2)
    forward_pair(actual, expected, 1)
    for cache in (actual, expected):
        cache.reset()
    _assert_equal(actual, expected)
    forward_pair(actual, expected, 2)

    # Both layouts must reject heterogeneous positions before modifying K/V.
    for cache in (actual, expected):
        cache.cache_seqlens[1] += 1
        pinned_k, pinned_v = cache.k_cache.clone(), cache.v_cache.clone()
        q = t.zeros((2, 1, 2, 8), device=device, dtype=cache.k_cache.dtype)
        with pytest.raises(NotImplementedError, match="uniform cache_seqlens"):
            flash_attention.flash_attn_with_kvcache(
                q,
                *cache.get_layer_cache(0),
                k=q,
                v=q,
                cache_seqlens=cache.cache_seqlens,
                causal=True,
            )
        assert t.equal(cache.k_cache, pinned_k)
        assert t.equal(cache.v_cache, pinned_v)

    sources = []
    for _ in range(3):
        source = engine._make_kv_cache(batch_size=1, seq_len=12)
        reference = _reference(source)
        forward_pair(source, reference, 3)
        sources.append(source)
    pinned_sources = [(cache.k_cache.clone(), cache.v_cache.clone()) for cache in sources]

    actual = engine._make_kv_cache(batch_size=3, seq_len=12)
    expected = _reference(actual)
    actual.prefill(sources[0])
    expected.prefill(sources[0])
    _assert_equal(actual, expected)
    forward_pair(actual, expected, 1)
    with pytest.raises(RuntimeError, match="non-empty KV cache"):
        actual.prefill(sources[0])

    actual = stack_prefilled_kv_caches(tuple(sources))
    expected = _reference(actual)
    for row, source in enumerate(sources):
        expected.k_cache[:, row : row + 1, :3] = source.k_cache[:, :, :3]
        expected.v_cache[:, row : row + 1, :3] = source.v_cache[:, :, :3]
    expected.advance(3)
    _assert_equal(actual, expected)
    forward_pair(actual, expected, 1)
    keep = t.tensor([2, 0], device=device)
    actual = engine._compact_kv_cache(actual, keep)
    compact_reference = _reference(actual)
    compact_reference.k_cache[:, :, :4] = expected.k_cache[:, keep, :4]
    compact_reference.v_cache[:, :, :4] = expected.v_cache[:, keep, :4]
    compact_reference.advance(4)
    forward_pair(actual, compact_reference, 2)
    for cache, (pinned_k, pinned_v) in zip(sources, pinned_sources, strict=True):
        assert cache.get_pos() == 3
        assert t.equal(cache.k_cache, pinned_k)
        assert t.equal(cache.v_cache, pinned_v)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "mode", [SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens]
)
@pytest.mark.parametrize("failure", ["model", "provider"])
def test_self_conditioned_failure_rewinds_host_positions_before_retry(
    monkeypatch, device, mode, failure
):
    if device == "cuda" and not t.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    cache = _engine(device)._make_kv_cache(batch_size=1, seq_len=8)
    cache.k_cache[:, :, :3].fill_(2)
    cache.v_cache[:, :, :3].fill_(3)
    cache.advance(3)
    state = SelfCondState(
        encoder_out=pack_dense_encoder_output_reference(
            t.zeros((1, 1, 1, 1)), t.ones((1, 1, 1), dtype=t.bool)
        ),
        target_slot_idx=0,
    )
    fp_buffer = (t.zeros((1, 1, 3), device=device),)
    injected_failure = failure

    def model(_token, _targets, **kwargs):
        position = cache.get_pos()
        cache.k_cache[:, :, position].fill_(5)
        cache.v_cache[:, :, position].fill_(7)
        cache.advance(1)
        if injected_failure == "model":
            raise RuntimeError("injected model failure")
        fp = t.ones((1, 1, 3), device=device)
        logits = t.ones((1, 1, 9), device=device)
        if kwargs.get("fp_logits_only"):
            return fp
        return (logits, fp) if kwargs.get("return_fp_logits") else logits

    def step(old_state, _fp):
        if injected_failure == "provider":
            raise RuntimeError("injected provider failure")
        return old_state

    provider = cast(
        SelfConditionedDecodeProvider,
        SimpleNamespace(
            cfg=SimpleNamespace(self_cond_segment_size=2, segment_mode=mode), step=step
        ),
    )
    kwargs = {
        "model": model,
        "provider": provider,
        "state": state,
        "fp_buffer": fp_buffer,
        "tokens_in_seg": 1,
        "emitted_token": t.tensor([[4]], device=device),
        "kv_cache": cache,
        "device": t.device(device),
    }
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        self_conditioned_decode_step(**kwargs)
    assert cache.cache_seqlens.device.type == "cpu"
    assert cache.cache_seqlens.tolist() == [3]
    assert bool((cache.k_cache[:, :, :3] == 2).all())
    assert bool((cache.v_cache[:, :, :3] == 3).all())
    assert state.target_slot_idx == 0
    assert t.equal(fp_buffer[0], t.zeros_like(fp_buffer[0]))

    injected_failure = None
    result = self_conditioned_decode_step(**kwargs)
    assert cache.get_pos() == 4
    assert cache.cache_seqlens.tolist() == [4]
    assert bool((cache.k_cache[:, :, 3] == 5).all())
    assert bool((cache.v_cache[:, :, 3] == 7).all())
    assert result.tokens_in_seg == 0
    assert result.fp_buffer == ()
