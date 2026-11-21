"""Residual-head contracts at model, optimizer, generation, and DDP boundaries."""

from __future__ import annotations

import copy
import io
from dataclasses import replace

import pytest
import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import nanochat.flash_attention as flash_attention
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import MLP
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.encoder_decoder.selfcond_training import (
    build_head_only_optimizer,
    freeze_for_head_only,
)
from steering.conditional.feature_vocab import FeatureVocab
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.test_self_cond_generation_roundtrip import (
    _build_engine,
    _build_fp_encdec,
    _runtime_state,
)


@pytest.fixture(autouse=True)
def _sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(t._dynamo.config, "cache_size_limit", 32)


def _model(
    *,
    depth=0,
    block="mlp",
    compact=False,
    mode=FeaturePredLossMode.MAX_POOL,
    loss_fn=FeaturePredLossFn.BCE,
    layer=-1,
):
    vocab = FeatureVocab(32, (1, 4, 7, 12, 21, 30)) if compact else None
    return build_tiny_model(
        n_sae_features=32,
        num_buckets=2,
        n_layer=2,
        seq_len=16,
        feature_pred_config=FeaturePredictionConfig(
            block=FeaturePredictionBlock(block),
            head_layers=depth,
            loss_mode=mode,
            loss_fn=loss_fn,
            layer=layer,
        ),
        feature_vocab=vocab,
    )


def _batch(model):
    rng = t.Generator().manual_seed(123)
    allowed = t.tensor(model.feature_vocab.original_ids if model.feature_vocab else range(32))
    width = model.feature_prediction_width
    return {
        "idx": t.randint(0, 100, (2, 16), generator=rng),
        "targets": t.randint(0, 100, (2, 16), generator=rng),
        "enc_feature_ids": allowed[t.randint(0, len(allowed), (2, 4, 3), generator=rng)],
        "enc_magnitudes": t.randint(0, 2, (2, 4, 3), generator=rng),
        "enc_mask": t.ones(2, 4, 3, dtype=t.bool),
        "seg_map": t.arange(4).repeat_interleave(4).expand(2, -1),
        "seg_doc_id": t.zeros(2, 4, dtype=t.long),
        "fp_targets": t.rand(2, 4, width, generator=rng),
    }


def _inference_batch(batch):
    return {
        **{key: value for key, value in batch.items() if key not in {"targets", "fp_targets"}},
        "targets": None,
    }


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("mode", list(FeaturePredLossMode))
@pytest.mark.parametrize("loss_fn", list(FeaturePredLossFn))
@pytest.mark.parametrize("layer", [0, -1])
@pytest.mark.parametrize("block", ["mlp", "transformer"])
def test_upgrade_preserves_all_forward_seams_and_gradients(compact, mode, loss_fn, layer, block):
    t.manual_seed(174)
    original = _model(compact=compact, mode=mode, loss_fn=loss_fn, layer=layer)
    upgraded = copy.deepcopy(original)
    upgraded.feature_pred_head = ResidualFeaturePredictionHead(
        upgraded.feature_pred_head, 2, block, decoder_config=upgraded.dec_config
    )
    upgraded.feature_pred_config = replace(upgraded.feature_pred_config, block=block, head_layers=2)
    batch = _batch(original)
    before = original(**batch)
    after = upgraded(**batch)
    assert isinstance(before, ForwardResult) and isinstance(after, ForwardResult)
    for name in ("loss", "lm_loss", "feature_pred_loss"):
        assert t.equal(getattr(before, name), getattr(after, name)), name
    before.loss.backward()
    after.loss.backward()
    candidate_params = dict(upgraded.named_parameters())
    for name, parameter in original.named_parameters():
        candidate = candidate_params[name]
        assert t.equal(parameter, candidate), name
        if parameter.grad is None:
            assert candidate.grad is None, name
        else:
            assert t.equal(parameter.grad, candidate.grad), name
    inputs = _inference_batch(batch)
    with t.no_grad():
        old_lm, old_fp = original(**inputs, return_fp_logits=True)
        new_lm, new_fp = upgraded(**inputs, return_fp_logits=True)
        assert t.equal(old_lm, new_lm)
        assert t.equal(old_fp, new_fp)
        assert t.equal(new_fp, upgraded(**inputs, fp_logits_only=True))


def test_existing_head_only_optimizer_covers_complete_module_and_freezes_trunk():
    model = _model(depth=2)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    freeze_for_head_only(model)
    optimizer = build_head_only_optimizer(model, unembedding_lr=0.004, adam_betas=(0.8, 0.95))
    expected = {id(parameter) for parameter in model.feature_pred_head.parameters()}
    assert {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    } == expected
    model(**_batch(model)).loss.backward()
    with t.compiler.set_stance("force_eager"):
        optimizer.step()
    for name, parameter in model.named_parameters():
        if not name.startswith("feature_pred_head."):
            assert not parameter.requires_grad
            assert parameter.grad is None
            assert t.equal(before[name], parameter), name
    for block in model.feature_pred_head.residual_blocks:
        assert t.count_nonzero(block.c_proj.weight) > 0


def test_learned_head_survives_self_conditioned_prefill_decode_and_restore():
    t.manual_seed(0)
    model = _build_fp_encdec()
    model.feature_pred_head = ResidualFeaturePredictionHead(model.feature_pred_head, 2)
    model.feature_pred_config = replace(model.feature_pred_config, head_layers=2)
    assert isinstance(model.feature_pred_head, ResidualFeaturePredictionHead)
    assert model.feature_pred_config is not None
    with t.no_grad():
        for block in model.feature_pred_head.residual_blocks:
            assert isinstance(block, MLP)
            block.c_proj.weight.normal_(std=0.03)
    engine = _build_engine(K=2, model=model)
    tokens = engine._tokenizer.encode("hello", prepend=0)
    before = list(engine.generate_stream(tokens, max_new_tokens=12, temperature=0.0))
    state = _runtime_state(engine)
    assert len(before) > 2
    assert state.encoder_out.layout.n_segments == 2 + (len(before) - 1) // 2
    restored = EncoderDecoderGPT(
        model.dec_config,
        model.enc_config,
        model.xattn_config,
        n_sae_features=model.encoder.config.n_sae_features,
        num_buckets=model.encoder.config.num_buckets,
        pad_vocab_size_to=model.dec_config.vocab_size,
        feature_pred_config=replace(model.feature_pred_config),
    )
    restored.init_buffers()
    restored.load_state_dict(model.state_dict(), strict=True, assign=True)
    restored.eval()
    after = list(
        _build_engine(K=2, model=restored).generate_stream(
            tokens, max_new_tokens=12, temperature=0.0
        )
    )
    assert before == after


def _distributed_head_resume(rank, world_size, block):
    t.set_num_threads(1)
    flash_attention._override_impl = FlashAttnImpl.SDPA
    t.manual_seed(425)
    model = _model(depth=2, block=block)
    head = model.feature_pred_head
    reference = copy.deepcopy(head)
    candidate = DistributedDataParallel(head)
    optimizer = t.optim.AdamW(candidate.parameters(), lr=0.003, weight_decay=0)
    reference_optimizer = t.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0)
    inputs = t.randn(4, 3, 64, generator=t.Generator().manual_seed(877))
    targets = t.randn(4, 3, 32, generator=t.Generator().manual_seed(973))
    local = slice(rank * 2, (rank + 1) * 2)
    context = {"cos_sin": (model.cos[:, :3], model.sin[:, :3]), "window_size": (-1, 0)}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        (candidate(inputs[local], **context) - targets[local]).square().mean().backward()
        (reference(inputs, **context) - targets).square().mean().backward()
        optimizer.step()
        reference_optimizer.step()
        if step == 0:
            buffer = io.BytesIO()
            t.save({"model": head.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = t.load(buffer, weights_only=True)
            restored_head = _model(depth=2, block=block).feature_pred_head
            restored_head.load_state_dict(checkpoint["model"], strict=True, assign=True)
            restored = DistributedDataParallel(restored_head)
            restored_optimizer = t.optim.AdamW(restored.parameters(), lr=0.003, weight_decay=0)
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            restored_optimizer.zero_grad(set_to_none=True)
            (restored(inputs[local], **context) - targets[local]).square().mean().backward()
            restored_optimizer.step()
    for name, parameter in head.named_parameters():
        assert t.equal(parameter, dict(restored_head.named_parameters())[name]), name
        t.testing.assert_close(
            parameter, dict(reference.named_parameters())[name], atol=1e-6, rtol=2e-5
        )
        gathered = [t.empty_like(parameter) for _ in range(world_size)]
        dist.all_gather(gathered, parameter)
        assert all(t.equal(parameter, other) for other in gathered)
    return {
        "rank": rank,
        "depth": len(head.residual_blocks),
        "parameters": len(list(head.parameters())),
    }


@pytest.mark.parametrize("block,parameters", [("mlp", 6), ("transformer", 10)])
def test_two_rank_gradient_aggregation_and_checkpoint_resume(gloo_multiprocess, block, parameters):
    result = gloo_multiprocess(2, _distributed_head_resume, block)
    assert result == [
        {"rank": 0, "depth": 2, "parameters": parameters},
        {"rank": 1, "depth": 2, "parameters": parameters},
    ]


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="real CUDA packed attention required")
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_packed_forward_upgrade_learning_and_restore(compiled):
    t.manual_seed(881)
    original = _model(compact=True).to("cuda")
    original.init_weights()
    upgraded = copy.deepcopy(original)
    upgraded.feature_pred_head = ResidualFeaturePredictionHead(upgraded.feature_pred_head, 2)
    upgraded.feature_pred_config = replace(upgraded.feature_pred_config, head_layers=2)
    batch = {key: value.to("cuda") for key, value in _batch(original).items()}
    batch["static_max_seg_size"] = 4
    original.prewarm_decoder_block_masks(2, 16)
    upgraded.prewarm_decoder_block_masks(2, 16)
    reference = t.compile(original, dynamic=False) if compiled else original
    candidate = t.compile(upgraded, dynamic=False) if compiled else upgraded
    inputs = _inference_batch(batch)
    with t.no_grad():
        expected_lm, expected_fp = reference(**inputs, return_fp_logits=True)
        actual_lm, actual_fp = candidate(**inputs, return_fp_logits=True)
        assert t.equal(expected_lm, actual_lm)
        assert t.equal(expected_fp, actual_fp)
        assert t.equal(actual_fp, candidate(**inputs, fp_logits_only=True))
    optimizer = upgraded.setup_optimizer()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = candidate(**batch)
        assert isinstance(result, ForwardResult)
        assert t.isfinite(result.loss)
        result.loss.backward()
        for block in upgraded.feature_pred_head.residual_blocks:
            assert isinstance(block, MLP)
            for parameter in block.parameters():
                assert parameter.grad is not None
                assert t.isfinite(parameter.grad).all()
            if step:
                assert block.c_fc.weight.grad is not None
                assert t.count_nonzero(block.c_fc.weight.grad) > 0
        optimizer.step()
    restored = _model(depth=2, compact=True).to("cuda")
    restored.load_state_dict(upgraded.state_dict(), strict=True, assign=True)
    restored.assert_vocab_consistent()
    restored.prewarm_decoder_block_masks(2, 16)
    restored_callable = t.compile(restored, dynamic=False) if compiled else restored
    with t.no_grad():
        assert t.equal(
            candidate(**inputs, fp_logits_only=True),
            restored_callable(**inputs, fp_logits_only=True),
        )
