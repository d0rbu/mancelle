from __future__ import annotations

import sys
from typing import Any

import pytest
import torch as t

import nanochat.gpt as gpt_module
import steering.eval.greedy_token_batch as greedy
from nanochat.gpt import GPT, GPTConfig

compaction: Any
if sys.platform == "linux":
    import steering.eval.triton_kv_cache_compaction as compaction
else:
    compaction = None

pytestmark = pytest.mark.skipif(compaction is None, reason="Triton helper requires Linux")


class _Tokenizer:
    def get_bos_token_id(self):
        return 63

    def encode_special(self, name):
        return {"<assistant_end>": 61, "<python_start>": 62}[name]


def _trained_model(device):
    model: Any = (
        GPT(
            GPTConfig(
                sequence_len=128,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_kv_head=1,
                n_embd=256,
                window_pattern="SL",
            ),
            pad_vocab_size_to=64,
        )
        .to(device)
        .eval()
    )
    model.init_weights()
    with t.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.uniform_(-0.4, 0.4)
        model.resid_lambdas.copy_(t.tensor([0.73, 1.27], device=device))
        model.x0_lambdas.copy_(t.tensor([-0.11, 0.19], device=device))
    return model


def _snapshot(cache):
    tensors = [*cache.k_layers, *cache.v_layers, cache.cache_seqlens]
    data = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        data.append(
            t.empty(0, device=tensor.device, dtype=t.uint8)
            .set_(storage, 0, (storage.nbytes(),), (1,))
            .cpu()
            .clone()
        )
    return {
        "data": data,
        "pointers": [tensor.untyped_storage().data_ptr() for tensor in tensors],
        "views": [(tuple(x.shape), x.stride(), x.storage_offset(), x.dtype) for x in tensors],
        "positions": cache.cache_seqlens.cpu().tolist(),
        "metadata": (
            cache.batch_size,
            cache.max_seq_len,
            cache.final_seq_len,
            cache._initial_row_token_budget,
        ),
    }


def _run(monkeypatch, model, *, candidate):
    original_prefill = greedy._prefill_ragged_prompt_batch
    original_decode = greedy._ragged_forward_one
    original_compact = greedy._compact_kv_cache_rows
    original_helper = greedy.try_compact_rows
    original_kernel = compaction._compact_layer_rows_kernel
    assert original_helper is not None
    active = [0, 1, 2, 3]
    frames, events, helper_calls, launches, caches = [], [], [], [], []
    step = 0

    def force_tokens(raw, token_step):
        # Save real model logits first. A deterministic test-only overlay forces
        # stops; random trained-style weights need not naturally emit EOS.
        assert raw.dtype == t.float32 and t.isfinite(raw).all() and t.count_nonzero(raw)
        frames.append((token_step, tuple(active), raw.detach().cpu().clone(), _snapshot(caches[0])))
        forced = t.full_like(raw, -1)
        last = forced.reshape(raw.shape[0], -1, raw.shape[-1])[:, -1]
        for packed_row, original_row in enumerate(active):
            terminal = (original_row == 1 and token_step == 0) or (
                original_row == 0 and token_step == 16
            )
            token = 61 if terminal else 1 + (original_row * 7 + token_step) % 50
            last[packed_row, token] = 1
        return forced

    def prefill(*args, **kwargs):
        cache, raw = original_prefill(*args, **kwargs)
        assert type(cache) is greedy._ExpandableRaggedKVCache
        assert all(x.dtype == t.float8_e4m3fn for x in cache.k_layers + cache.v_layers)
        assert cache.max_seq_len == cache.final_seq_len == 38
        caches.append(cache)
        return cache, force_tokens(raw, 0)

    def decode(*args, **kwargs):
        nonlocal step
        step += 1
        assert id(model) in greedy._ACTIVE_STAGED_LINEAR_MODEL_IDS
        raw = original_decode(*args, **kwargs)
        return force_tokens(raw, step)

    def compact(cache, keep):
        before = _snapshot(cache)
        original_compact(cache, keep)
        after = _snapshot(cache)
        assert before["pointers"] == after["pointers"]
        assert after["positions"] == [before["positions"][i] for i in keep]
        events.append((step, tuple(active), list(keep), before, after))
        active[:] = [active[index] for index in keep]

    def helper(cache, keep, *, cache_type):
        result = original_helper(cache, keep, cache_type=cache_type)
        helper_calls.append(result)
        assert result is True
        return result

    class Witness:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def run(*args, **kwargs):
                result = launch(*args, **kwargs)
                launches.append(grid)
                return result

            return run

    with monkeypatch.context() as patch:
        patch.setattr(greedy, "_prefill_ragged_prompt_batch", prefill)
        patch.setattr(greedy, "_ragged_forward_one", decode)
        patch.setattr(greedy, "_compact_kv_cache_rows", compact)
        patch.setattr(greedy, "try_compact_rows", helper if candidate else None)
        patch.setattr(compaction, "_compact_layer_rows_kernel", Witness())
        completions = greedy.generate_greedy_token_batch(
            model,
            _Tokenizer(),
            [[1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]],
            max_new_tokens=33,
            ragged_cache_initial_len=38,
        )
    assert greedy.try_compact_rows is original_helper
    assert compaction._compact_layer_rows_kernel is original_kernel
    assert id(model) not in greedy._ACTIVE_STAGED_LINEAR_MODEL_IDS
    assert step == 32 and len(frames) == 33
    assert [(event[0], event[2]) for event in events] == [(15, [0, 2, 3]), (31, [1, 2])]
    assert helper_calls == ([True, True] if candidate else [])
    assert len(launches) == (4 if candidate else 0)
    assert completions is not None
    assert [len(row) for row in completions] == [16, 0, 33, 33]
    pointers = frames[0][3]["pointers"]
    assert all(frame[3]["pointers"] == pointers for frame in frames)
    assert [len(frame[1]) for frame in frames] == [4] * 16 + [3] * 16 + [2]
    return completions, frames, events, caches[0]


def _same_state(actual, expected):
    assert all(actual[key] == expected[key] for key in ("views", "positions", "metadata"))
    assert all(t.equal(a, b) for a, b in zip(actual["data"], expected["data"], strict=True))


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16], ids=["bfloat16", "float16"])
def test_trained_generator_preserves_repeated_compaction(monkeypatch, dtype):
    if not t.cuda.is_available() or t.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires Ada CUDA")
    device = t.device("cuda", t.cuda.current_device())
    assert not greedy.HAS_FA3 and greedy._should_use_float8_ragged_cache(device)
    assert greedy._HOST_CHECK_INTERVAL == 16
    monkeypatch.setattr(greedy, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", dtype)
    t.manual_seed(1977)
    model = _trained_model(device)
    with t.inference_mode(), t.autocast("cuda", dtype=dtype):
        expected = _run(monkeypatch, model, candidate=False)
        actual = _run(monkeypatch, model, candidate=True)
    assert actual[0] == expected[0]
    for a, e in zip(actual[1], expected[1], strict=True):
        assert a[:2] == e[:2]
        assert t.equal(a[2].contiguous().view(t.uint8), e[2].contiguous().view(t.uint8))
        _same_state(a[3], e[3])
    for a, e in zip(actual[2], expected[2], strict=True):
        assert a[:3] == e[:3]
        _same_state(a[3], e[3])
        _same_state(a[4], e[4])
    assert set(_snapshot(actual[3])["pointers"]).isdisjoint(_snapshot(expected[3])["pointers"])
    # The real model writes nontrivial cache values before and after packing.
    assert not t.equal(actual[1][0][3]["data"][0], actual[1][-1][3]["data"][0])
