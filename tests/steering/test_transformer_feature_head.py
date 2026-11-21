"""Independent mathematics and learned-head cache/gradient contracts for #1719."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch as t
import torch.nn.functional as F
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torchao.float8.float8_linear import Float8Linear

from nanochat import flash_attention
from nanochat.engine import KVCache
from nanochat.gpt import Block, GPTConfig, Linear, OptimizerKind
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.eval.greedy_token_batch import _compact_kv_cache_rows, _HostPositionKVCache
from steering.eval.inference_engine import stack_prefilled_kv_caches
from tests.steering.test_residual_feature_head_integration import _batch, _inference_batch, _model


@pytest.fixture(autouse=True)
def sdpa(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.SDPA)


def _rotary(length, width, *, start=0, device="cpu", dtype=t.float64):
    frequencies = t.arange(start, start + length, device=device, dtype=t.float64)[:, None] * (
        10000 ** (-t.arange(0, width, 2, device=device, dtype=t.float64) / width)
    )
    return tuple(part[None, :, None, :].to(dtype) for part in (frequencies.cos(), frequencies.sin()))


def _head(depth=2, *, heads=4, kv_heads=2, dim=8, device="cpu", dtype=t.float64):
    config = GPTConfig(n_layer=2, n_head=heads, n_kv_head=kv_heads, n_embd=heads * dim)
    projection = Linear(config.n_embd, 7, device=device, dtype=dtype)
    return ResidualFeaturePredictionHead(projection, depth, FeaturePredictionBlock.TRANSFORMER, decoder_config=config)


def _learned(head):
    with t.no_grad():
        for block in head.residual_blocks:
            block.attn.c_proj.weight.normal_(0, 0.03)
            block.mlp.c_proj.weight.normal_(0, 0.03)
    return head


def _norm(x):
    return x * (x.square().mean(-1, keepdim=True) + t.finfo(x.dtype).eps).rsqrt()


@pytest.mark.parametrize("config", [None, GPTConfig(n_embd=64)])
def test_transformer_head_requires_matching_decoder_geometry(config):
    with pytest.raises(ValueError, match="matching decoder configuration"):
        ResidualFeaturePredictionHead(
            Linear(32, 7), 1, FeaturePredictionBlock.TRANSFORMER, decoder_config=config
        )


@pytest.mark.parametrize("kwargs", [{}, {"cos_sin": _rotary(3, 8)}, {"window_size": (-1, 0)}])
def test_transformer_head_requires_both_rotary_and_window_context(kwargs):
    with pytest.raises(ValueError, match="RoPE and window context"):
        _head()(t.randn(1, 3, 32, dtype=t.float64), **kwargs)


@pytest.mark.parametrize("start", [-1, 1_000_000])
def test_model_head_rejects_positions_outside_its_rotary_capacity(start):
    model = _transformer_model()
    with pytest.raises(ValueError, match="RoPE capacity"):
        model.apply_feature_prediction_head(t.zeros(1, 3, model.dec_config.n_embd), start)


def test_model_head_rejects_headless_application():
    model = _model()
    model.feature_pred_head = None
    with pytest.raises(ValueError, match="no feature-prediction head"):
        model.apply_feature_prediction_head(t.zeros(1, 3, model.dec_config.n_embd))


def test_float8_conversion_preserves_head_and_projection_parameter_ownership():
    head = _head(2, dim=32, dtype=t.float32)
    original = dict(head.named_parameters())
    converted = convert_to_float8_training(head, config=Float8LinearConfig())
    assert converted is head and type(converted) is ResidualFeaturePredictionHead
    assert all(parameter is original[name] for name, parameter in converted.named_parameters())
    for block in converted.residual_blocks:
        assert isinstance(block.attn.c_qkv, Float8Linear)
        assert isinstance(block.attn.c_proj, Float8Linear)
        assert isinstance(block.mlp.c_fc, Float8Linear)
        assert isinstance(block.mlp.c_proj, Float8Linear)


def _reference(x, head, cos_sin, window):
    """Explicit FP64 RMS/RoPE/GQA/softmax/ReLU-square, without decoder helpers."""
    batch, length, width = x.shape
    for block in head.residual_blocks:
        heads, kv_heads, dim = block.attn.n_head, block.attn.n_kv_head, block.attn.head_dim
        qkv = _norm(x) @ block.attn.c_qkv.weight.T
        q, k, v = qkv.split((heads * dim, kv_heads * dim, kv_heads * dim), -1)
        q, k, v = (part.reshape(batch, length, count, dim) for part, count in ((q, heads), (k, kv_heads), (v, kv_heads)))
        cos, sin = cos_sin

        def rotate(part, cos=cos, sin=sin):
            first, second = part.chunk(2, -1)
            return t.cat((first * cos + second * sin, second * cos - first * sin), -1)

        q, k = _norm(rotate(q)), _norm(rotate(k))
        k = k.repeat_interleave(heads // kv_heads, 2)
        v = v.repeat_interleave(heads // kv_heads, 2)
        scores = t.einsum("bqhd,bkhd->bhqk", q, k) / dim**0.5
        delta = t.arange(length, device=x.device)[:, None] - t.arange(length, device=x.device)[None, :]
        allowed = (delta >= 0) & ((delta <= window) if window >= 0 else True)
        probabilities = scores.masked_fill(~allowed, -t.inf).softmax(-1)
        attended = t.einsum("bhqk,bkhd->bqhd", probabilities, v).reshape(batch, length, width)
        x = x + attended @ block.attn.c_proj.weight.T
        intermediate = (_norm(x) @ block.mlp.c_fc.weight.T).clamp_min(0).square()
        x = x + intermediate @ block.mlp.c_proj.weight.T
    return x @ head.weight.T + head.bias


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 6, 8, 10])
def test_identity_retains_projection_rng_gradients_and_only_new_output_matrices_learn_first(depth):
    projection = Linear(32, 7, dtype=t.float64).eval()
    config = GPTConfig(n_layer=27, n_embd=32, n_head=4, n_kv_head=2)
    rng = t.random.get_rng_state().clone()
    head = ResidualFeaturePredictionHead(projection, depth, FeaturePredictionBlock.TRANSFORMER, decoder_config=config)
    assert t.equal(rng, t.random.get_rng_state())
    assert head.weight is projection.weight and head.bias is projection.bias
    assert not head.training
    for index, block in enumerate(head.residual_blocks):
        assert isinstance(block, Block)
        assert block.attn.ve_gate is None and block.attn.layer_idx == 27 + index
    x = t.randn(2, 7, 32, dtype=t.float64, requires_grad=True)
    kwargs = {"cos_sin": _rotary(7, 8), "window_size": (3, 0)}
    expected = projection(x)
    actual = head(x, **kwargs)
    assert t.equal(actual, expected)
    for observed, reference in zip(
        t.autograd.grad(actual.sum(), (x, head.weight, head.bias), retain_graph=True),
        t.autograd.grad(expected.sum(), (x, projection.weight, projection.bias)), strict=True
    ):
        assert t.equal(observed, reference)
    actual.square().mean().backward()
    for block in head.residual_blocks:
        assert isinstance(block, Block)
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None
            assert bool(t.count_nonzero(parameter.grad)) == ("c_proj" in name)
    optimizer = t.optim.SGD(head.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    head(x, **kwargs).square().mean().backward()
    assert all(parameter.grad is not None and t.count_nonzero(parameter.grad) for parameter in head.parameters())


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (13, 13)])
@pytest.mark.parametrize("length,window,start", [(1, 0, 0), (7, 3, 5), (17, -1, 23), (19, 8, 31)])
def test_learned_head_matches_independent_fp64_outputs_and_every_gradient(heads, kv_heads, length, window, start):
    t.manual_seed(1719)
    head = _learned(_head(heads=heads, kv_heads=kv_heads))
    reference = copy.deepcopy(head)
    x = t.randn(2, length, head.in_features, dtype=t.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    cos_sin = _rotary(length, 8, start=start)
    actual = head(x, cos_sin=cos_sin, window_size=(window, 0))
    expected = _reference(other, reference, cos_sin, window)
    t.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    direction = t.randn_like(actual)
    (actual * direction).sum().backward()
    (expected * direction).sum().backward()
    for observed, target in zip((x, *head.parameters()), (other, *reference.parameters()), strict=True):
        t.testing.assert_close(observed.grad, target.grad, rtol=1e-10, atol=1e-10)


def _transformer_model(depth=2, *, compact=False, layer=-1):
    model = _model(compact=compact, layer=layer)
    model.feature_pred_head = _learned(ResidualFeaturePredictionHead(
        model.feature_pred_head, depth, FeaturePredictionBlock.TRANSFORMER, decoder_config=model.dec_config
    ))
    model.feature_pred_config = replace(model.feature_pred_config, block=FeaturePredictionBlock.TRANSFORMER, head_layers=depth)
    return model


def _cache(model, *, batch=2, layers=None, capacity=32, cls=KVCache):
    return cls(
        batch_size=batch, num_heads=model.dec_config.n_kv_head,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        seq_len=capacity, num_layers=model.num_cache_layers if layers is None else layers,
        device=model.get_device(), dtype=model.cos.dtype,
    )


def _slice(batch, start, stop):
    return {**batch, "idx": batch["idx"][:, start:stop], "seg_map": batch["seg_map"][:, start:stop]}


def test_row_compaction_preserves_the_learned_heads_complete_history():
    model = _transformer_model().eval()
    batch = _inference_batch(_batch(model))
    cache = _cache(model, cls=_HostPositionKVCache)
    with t.no_grad():
        expected = model(**batch, fp_logits_only=True)[1:]
        model(**_slice(batch, 0, 6), kv_cache=cache, fp_logits_only=True)
        _compact_kv_cache_rows(cache, [1])
        assert cache.batch_size == 1 and cache.n_layers == model.num_cache_layers
        assert cache.get_pos() == 6
        remaining = {key: value[1:] if isinstance(value, t.Tensor) else value for key, value in batch.items()}
        actual = model(**_slice(remaining, 6, 16), kv_cache=cache, fp_logits_only=True)
    assert cache.get_pos() == 16
    t.testing.assert_close(actual, expected[:, 6:], atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("layer", [0, -1])
def test_learned_head_is_causal_across_document_and_prediction_group_boundaries(compact, layer):
    model = _transformer_model(compact=compact, layer=layer)
    batch = _inference_batch(_batch(model))
    batch["seg_doc_id"][:, 2:] = 1
    before = model(**batch, fp_logits_only=True)
    perturbed = {**batch, "idx": batch["idx"].clone()}
    perturbed["idx"][:, 9:] = (perturbed["idx"][:, 9:] + 17) % 100
    after = model(**perturbed, fp_logits_only=True)
    assert t.equal(before[:, :9], after[:, :9])
    assert not t.equal(before[:, 9:], after[:, 9:])
    embeddings = []

    def capture_embeddings(_module, _inputs, output):
        output.retain_grad()
        embeddings.append(output)

    hook = model._decoder_embedding.register_forward_hook(capture_embeddings)
    try:
        model(**batch, fp_logits_only=True)[:, :9].sum().backward()
    finally:
        hook.remove()
    assert t.count_nonzero(embeddings[0].grad[:, :9])
    assert not t.count_nonzero(embeddings[0].grad[:, 9:])
    hidden = model(**batch, fp_hidden_only=True).detach().requires_grad_()
    model.apply_feature_prediction_head(hidden)[:, :9].sum().backward()
    assert t.count_nonzero(hidden.grad[:, :9])
    assert not t.count_nonzero(hidden.grad[:, 9:])


@pytest.mark.parametrize("chunks", [[16], [3, 1, 5, 7], [1] * 16])
@pytest.mark.parametrize("layer", [0, -1])
@pytest.mark.parametrize("readout", ["paired", "fp_only", "hidden_only"])
def test_learned_cache_matches_full_prefix_absolute_positions_and_advances_once(chunks, layer, readout, monkeypatch):
    model = _transformer_model(layer=layer).eval()
    model.window_sizes = [(4, 0), (7, 0)]
    batch = _inference_batch(_batch(model))
    with t.no_grad():
        expected_lm, expected_fp = model(**batch, return_fp_logits=True)
        expected = model(**batch, fp_hidden_only=True) if readout == "hidden_only" else expected_fp
        reference_cache = _cache(model)
        model(**batch, kv_cache=reference_cache, return_fp_logits=True)
        if readout != "paired":
            def unexpected_lm(*_args, **_kwargs):
                pytest.fail("feature-only readout must skip the LM projection")

            monkeypatch.setattr(model.lm_head, "forward", unexpected_lm)
        if readout == "hidden_only":
            # A projection would now fail, while the learned head blocks must
            # still fill their cache slots and return the pre-head states.
            monkeypatch.setattr(model.feature_pred_head, "weight", None)
        flags = {"paired": "return_fp_logits", "fp_only": "fp_logits_only", "hidden_only": "fp_hidden_only"}
        cache = _cache(model)
        start = 0
        pieces = []
        for count in chunks:
            output = model(**_slice(batch, start, start + count), kv_cache=cache, **{flags[readout]: True})
            if readout == "paired":
                lm, output = output
                t.testing.assert_close(lm, expected_lm[:, start:start+count], atol=3e-6, rtol=3e-6)
            pieces.append(output)
            start += count
            assert cache.get_pos() == start
        t.testing.assert_close(t.cat(pieces, 1), expected, atol=3e-6, rtol=3e-6)
        t.testing.assert_close(cache.k_cache[:, :, :16], reference_cache.k_cache[:, :, :16], atol=3e-6, rtol=3e-6)
        t.testing.assert_close(cache.v_cache[:, :, :16], reference_cache.v_cache[:, :, :16], atol=3e-6, rtol=3e-6)
        for slot in range(model.num_cache_layers):
            assert t.count_nonzero(cache.k_cache[slot, :, :16])


@pytest.mark.parametrize("cache_class", [KVCache, _HostPositionKVCache])
@pytest.mark.parametrize("failure_stage", ["decoder", "head"])
def test_speculate_rewind_lm_only_commit_copy_reset_and_failure_rollback(cache_class, monkeypatch, failure_stage):
    model = _transformer_model().eval()
    batch = _inference_batch(_batch(model))
    cache = _cache(model, cls=cache_class)
    with t.no_grad():
        model(**_slice(batch, 0, 6), kv_cache=cache, return_fp_logits=True)
        model(**_slice(batch, 6, 7), kv_cache=cache, fp_logits_only=True)
        cache.rewind(1)
        # Commit with changed conditioning. The oracle replays this exact history.
        committed = {**batch, "enc_magnitudes": 1 - batch["enc_magnitudes"]}
        model(**_slice(committed, 6, 7), kv_cache=cache)
        copied = _cache(model, cls=cache_class)
        copied.prefill(cache)
        expected_cache = _cache(model)
        model(**_slice(batch, 0, 6), kv_cache=expected_cache, return_fp_logits=True)
        model(**_slice(committed, 6, 7), kv_cache=expected_cache, return_fp_logits=True)
        expected = model(**_slice(committed, 7, 10), kv_cache=expected_cache, fp_logits_only=True)
        actual = model(**_slice(committed, 7, 10), kv_cache=copied, fp_logits_only=True)
        assert t.equal(actual, expected)
        assert t.equal(copied.k_cache[:, :, :10], expected_cache.k_cache[:, :, :10])
        failed_mlp = (
            model._decoder_blocks[0].mlp if failure_stage == "decoder"
            else model.feature_pred_head.residual_blocks[-1].mlp
        )
        original = failed_mlp.forward
        prefix_keys = copied.k_cache[:, :, :10].clone()
        prefix_values = copied.v_cache[:, :, :10].clone()

        def fail(*_args, **_kwargs):
            # The decoder failure precedes the shared position advance; the
            # final head failure follows it. Both leave an uncommitted write.
            assert copied.get_pos() == (10 if failure_stage == "decoder" else 11)
            raise RuntimeError(f"injected failure in {failure_stage}")

        monkeypatch.setattr(failed_mlp, "forward", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            model(**_slice(committed, 10, 11), kv_cache=copied, fp_logits_only=True)
        assert copied.get_pos() == 10 and copied.cache_seqlens.tolist() == [10, 10]
        assert t.equal(copied.k_cache[:, :, :10], prefix_keys)
        assert t.equal(copied.v_cache[:, :, :10], prefix_values)
        monkeypatch.setattr(failed_mlp, "forward", original)
        retry = model(**_slice(committed, 10, 11), kv_cache=copied, fp_logits_only=True)
        reference = model(**_slice(committed, 10, 11), kv_cache=expected_cache, fp_logits_only=True)
        assert t.equal(retry, reference)
        copied.reset()
        assert copied.get_pos() == 0
        assert t.equal(
            model(**_slice(batch, 0, 4), kv_cache=copied, fp_logits_only=True),
            model(**_slice(batch, 0, 4), kv_cache=_cache(model), fp_logits_only=True),
        )


@pytest.mark.parametrize("invalid", ["undersized", "extra_layer", "overflow", "negative", "ragged", "batch"])
def test_bad_cache_fails_before_any_tensor_or_counter_write(invalid):
    model = _transformer_model()
    batch = _inference_batch(_batch(model))
    cache = _cache(model, layers={"undersized": 2, "extra_layer": 5}.get(invalid))
    if invalid == "overflow":
        cache.cache_seqlens.fill_(30)
    elif invalid == "negative":
        cache.cache_seqlens.fill_(-1)
    elif invalid == "ragged":
        cache.cache_seqlens[1] = 1
    elif invalid == "batch":
        cache.batch_size = 3
    before = tuple(x.clone() for x in (cache.k_cache, cache.v_cache, cache.cache_seqlens))
    with pytest.raises(ValueError, match="cache|KV"):
        model(**batch, kv_cache=cache, fp_logits_only=True)
    assert all(t.equal(a, b) for a, b in zip(before, (cache.k_cache, cache.v_cache, cache.cache_seqlens), strict=True))


def test_decoder_only_cache_and_stacked_head_cache_own_correct_slots():
    model = _transformer_model().eval()
    batch = _inference_batch(_batch(model))
    with t.no_grad():
        decoder_only = _cache(model, layers=model.dec_config.n_layer)
        model(**_slice(batch, 0, 5), kv_cache=decoder_only)
        assert decoder_only.get_pos() == 5
        with pytest.raises(ValueError, match="Allocate the complete cache before prefill"):
            model(**_slice(batch, 5, 6), kv_cache=decoder_only, return_fp_logits=True)
        with pytest.raises(RuntimeError, match="KV cache shape mismatch"):
            _cache(model).prefill(decoder_only)
        caches = []
        for row in range(2):
            cache = _cache(model, batch=1)
            row_batch = {key: value[row:row+1] if isinstance(value, t.Tensor) else value for key, value in batch.items()}
            model(**_slice(row_batch, 0, 5), kv_cache=cache, return_fp_logits=True)
            caches.append(cache)
        stacked = stack_prefilled_kv_caches(tuple(caches))
        actual = model(**_slice(batch, 5, 6), kv_cache=stacked, fp_logits_only=True)
        expected = model(**_slice(batch, 0, 6), fp_logits_only=True)[:, -1:]
        t.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        assert stacked.n_layers == model.num_cache_layers and stacked.get_pos() == 6


def test_optimizer_parameter_accounting_and_transformer_attention_flops():
    model = _transformer_model(depth=4)
    optimizer = model.setup_optimizer()
    head = model.feature_pred_head
    ids = {id(parameter) for parameter in head.parameters()}
    block_matrix_ids = {
        id(parameter) for parameter in head.residual_blocks.parameters() if parameter.ndim == 2
    }
    grouped_head = [
        (parameter, group)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in ids
    ]
    assert len(grouped_head) == len(ids)
    assert {id(parameter) for parameter, _group in grouped_head} == ids
    for parameter, group in grouped_head:
        if id(parameter) in block_matrix_ids:
            assert group["kind"] == OptimizerKind.MUON
            assert group["lr"] == pytest.approx(0.02)
        else:
            assert group["kind"] == OptimizerKind.ADAMW
    assert len([parameter for group in optimizer.param_groups for parameter in group["params"]]) == len(list(model.parameters()))
    assert model.num_scaling_params()["feature_pred_head"] == sum(parameter.numel() for parameter in head.parameters())
    baseline = _model()
    delta_weights = sum(parameter.numel() for parameter in head.parameters() if parameter.ndim == 2) - baseline.feature_pred_head.weight.numel()
    expected_delta = 6 * delta_weights + 12 * model.dec_config.n_embd * model.dec_config.sequence_len * 4
    assert model.estimate_flops(3, 4) - baseline.estimate_flops(3, 4) == expected_delta


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_compiled_head_preserves_linear_identity_and_learned_gradients(depth):
    head = _head(depth, dtype=t.float32)
    x = t.randn(2, 7, head.in_features, requires_grad=True)
    kwargs = {"cos_sin": _rotary(7, 8, dtype=t.float32), "window_size": (4, 0)}
    compiled = t.compile(head, backend="aot_eager", fullgraph=True)
    assert t.equal(compiled(x, **kwargs), F.linear(x, head.weight, head.bias))
    _learned(head)
    eager = head(x, **kwargs)
    actual = compiled(x, **kwargs)
    assert t.equal(actual, eager)
    for left, right in zip(t.autograd.grad(eager.sum(), (x, *head.parameters())), t.autograd.grad(actual.sum(), (x, *head.parameters())), strict=True):
        assert t.equal(left, right)
