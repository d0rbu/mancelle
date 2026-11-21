"""Independent multi-slot soft-prefill checks with a learned transformer FP head."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.config import FeaturePredictionBlock, FeaturePredLossMode
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
    seed_self_conditioning_from_prompt,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import self_conditioned_decode_step
from steering.eval.axbench_adv.project_soft_prompt import ProjectSoftPromptWrapper
from steering.eval.axbench_adv.self_conditioning import _value_embedding_modules
from steering.eval.chat_generation import empty_assistant_conversation
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SelfConditionedTarget,
    SingleEntryContextProvider,
)
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, token_to_char_offsets
from tests.axbench_adv.unit.test_soft_project_generation import _assert_runtime_equal
from tests.conftest import MockTokenizer
from tests.steering.test_self_conditioned_wrapper import (
    _build_fp_encdec,
    _make_interpolation_config,
)


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cadence", [1, 3])
def test_multislot_prompt_and_learned_head_keep_exact_attacked_cache(
    turn: bool, cadence: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)
    with t.random.fork_rng(), t.no_grad():
        t.manual_seed(9784)
        base = _build_fp_encdec(num_buckets=2, loss_mode=FeaturePredLossMode.MAX_POOL)
        assert base.feature_pred_config is not None
        model = EncoderDecoderGPT(
            replace(base.dec_config, n_layer=12), base.enc_config, base.xattn_config,
            n_sae_features=64, num_buckets=2,
            feature_pred_config=replace(
                base.feature_pred_config,
                block=FeaturePredictionBlock.TRANSFORMER, head_layers=2,
            ),
        ).eval()
        model.init_weights()
        for parameter in model.parameters():
            if parameter.ndim >= 2 and not bool(parameter.ne(0).any()):
                parameter.normal_(0.0, 0.03)
    tokenizer = MockTokenizer.with_collision_check()
    prompt, seed_text = "abcdefgh", "! !"
    clean = tokenizer.render_for_completion(empty_assistant_conversation(prompt))
    attacked = tokenizer.render_for_completion(
        empty_assistant_conversation(prompt + "\n\n" + seed_text)
    )
    offsets = token_to_char_offsets(tokenizer, clean)
    cut = 6
    mode = SegmentMode.turn_delimited_fixed_tokens if turn else SegmentMode.fixed_tokens
    segments = (
        ContextFeatureSegment((1, 2), (1, 1), 0, offsets[cut]),
        ContextFeatureSegment((3, 4), (1, 1), offsets[cut], offsets[-2 if turn else -1]),
    )
    if turn:
        segments += (ContextFeatureSegment((5,), (1,), offsets[-2], offsets[-1]),)
    provider = SingleEntryContextProvider(
        clean, segments, segment_mode=mode,
        token_assignments=(
            (*([0] * cut), *([1] * (len(clean) - cut - 1)), 2) if turn else None
        ),
    )
    config = replace(
        _make_interpolation_config(), segment_mode=mode,
        self_cond_segment_size=PositiveInt.parse(cadence),
    )
    value_modules = _value_embedding_modules(model)
    assert [layer for layer, _ in value_modules] == [1, 3, 5, 7, 9, 11]
    modules: tuple[tuple[str, t.nn.Embedding], ...] = (
        ("wte", cast(t.nn.Embedding, model.transformer["wte"])),
        *((f"value:{layer}", cast(t.nn.Embedding, module)) for layer, module in value_modules),
    )
    seed_ids = cast(list[int], tokenizer.encode(seed_text))
    embeddings = t.cat([module.weight.detach()[seed_ids] for _, module in modules], dim=1)
    embeddings += t.linspace(-0.7, 0.7, embeddings.numel()).reshape_as(embeddings)
    common: dict[str, Any] = {"model": model, "tokenizer": tokenizer, "context_provider": provider}
    actual = ProjectSoftPromptWrapper(
        **common, self_cond_config=copy.deepcopy(config), clean_token_ids=clean,
        attacked_token_ids=attacked, raw_prompt=prompt, seed_text=seed_text,
        embeddings=embeddings, embedding_modules=modules,
    )
    reference = TargetAwareEncoderDecoderEvalWrapper(
        **common, target=SelfConditionedTarget(), self_cond_config=copy.deepcopy(config),
    )

    def cache() -> KVCache:
        return KVCache(
            batch_size=1, num_heads=model.dec_config.n_kv_head, seq_len=64,
            head_dim=model.dec_config.n_embd // model.dec_config.n_head,
            num_layers=model.num_cache_layers, device=t.device("cpu"), dtype=t.float32,
        )

    clean_cache, expected_cache, actual_cache = cache(), cache(), cache()
    with t.inference_mode():
        clean_ids = t.tensor([clean])
        runtime = reference._self_cond
        assert runtime is not None
        with reference.conditioning_context(context_token_ids=[clean]):
            encoder, clean_map = reference._build_conditioning(clean_ids, kv_cache=clean_cache)
            seed = seed_self_conditioning_from_prompt(
                model=model, runtime=runtime, input_ids=clean_ids, encoder_out=encoder,
                seg_map=clean_map, kv_cache=clean_cache,
            )
        assert sorted(set(clean_map[0, :-1].tolist())) == [0, 1]
        suffix_start = 2 + len(prompt) + 2
        # This character-token fixture has no seam merge. Preserve the entire
        # clean content routing and assign only newline/suffix/user_end rows to
        # its final real source slot, independently of project_suffix_prefix.
        prefix_map = t.cat((clean_map[:, :-2], t.ones(1, 2 + len(seed_ids) + 1, dtype=t.long)), 1)
        handles = []
        offset = 0
        try:
            for _, module in modules:
                width = module.weight.shape[1]
                values = embeddings[:, offset:offset + width]
                offset += width

                def inject(
                    _module: Any, _args: Any, output: t.Tensor,
                    *, values: t.Tensor = values,
                ) -> t.Tensor:
                    result = output.clone()
                    result[:, suffix_start:suffix_start + len(seed_ids)] = values
                    return result

                handles.append(module.register_forward_hook(inject))
            model(
                t.tensor([attacked[:-1]]), None, seg_map=prefix_map, kv_cache=expected_cache,
                encoder_out=seed.state.encoder_out, encoder_strict=True, logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()
        marker = t.tensor([[attacked[-1]]])
        if turn:
            result = self_conditioned_decode_step(
                model=model, provider=runtime.provider, state=seed.state, fp_buffer=(),
                tokens_in_seg=0, emitted_token=marker, kv_cache=expected_cache,
                device=t.device("cpu"),
            )
            expected = result.logits
            runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = (
                result.state, result.fp_buffer, result.tokens_in_seg,
            )
        else:
            expected = model(
                marker, None, seg_map=t.full((1, 1), seed.state.target_slot_idx),
                kv_cache=expected_cache, encoder_out=seed.state.encoder_out, encoder_strict=True,
            )
            runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = seed.state, (), 0
        with actual.conditioning_context(context_token_ids=[attacked]):
            observed = actual(t.tensor([attacked]), kv_cache=actual_cache)
        assert t.equal(observed, expected)
        _assert_runtime_equal(actual, reference)
        for token in (44, 45, 46, 47, 48, 49):
            ids = t.tensor([[token]])
            assert t.equal(actual(ids, kv_cache=actual_cache), reference(ids, kv_cache=expected_cache))
            _assert_runtime_equal(actual, reference)
        assert actual_cache.n_layers == expected_cache.n_layers == 14
        assert actual_cache.get_pos() == expected_cache.get_pos() == len(attacked) + 6
        assert t.equal(actual_cache.k_cache, expected_cache.k_cache)
        assert t.equal(actual_cache.v_cache, expected_cache.v_cache)
        for head_layer in (12, 13):
            assert t.count_nonzero(actual_cache.k_cache[head_layer, :, :len(attacked)]) > 0
        assert actual.self_cond_runtime_snapshot().state.target_slot_idx > seed.state.target_slot_idx
