"""Independent clean-seed / attacked-prefill oracle with the real project decoder."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import encoder as encoder_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
    seed_self_conditioning_from_prompt,
)
from steering.conditional.encoder_decoder.self_conditioned import self_conditioned_decode_step
from steering.eval.axbench_adv import project_soft_prompt
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.project_soft_prompt import ProjectSoftPromptWrapper
from steering.eval.axbench_adv.self_conditioning import _value_embedding_modules
from steering.eval.chat_generation import empty_assistant_conversation
from steering.eval.inference_engine import InferenceEngine
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SelfConditionedTarget,
    SingleEntryContextProvider,
)
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, token_to_char_offsets
from tests.conftest import MockTokenizer
from tests.steering.test_self_conditioned_wrapper import (
    _build_fp_encdec,
    _make_interpolation_config,
    _new_kv,
)


@pytest.fixture(autouse=True)
def _fp32_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float32)
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float32)


def _fixture(
    turn: bool, cadence: int, *, perturb: bool = True, suffix_token_count: int = 1
) -> dict[str, Any]:
    with t.random.fork_rng(), t.no_grad():
        t.manual_seed(450)
        model = _build_fp_encdec(num_buckets=2, loss_mode=FeaturePredLossMode.MAX_POOL)
        rng = t.Generator().manual_seed(449)
        for parameter in model.parameters():
            if parameter.ndim >= 2 and not bool(parameter.ne(0).any()):
                parameter.copy_(t.randn(parameter.shape, generator=rng) * 0.02)
    tokenizer = MockTokenizer()
    clean = tokenizer.render_for_completion(empty_assistant_conversation("q"))
    offsets = token_to_char_offsets(tokenizer, clean)
    segment_mode = SegmentMode.turn_delimited_fixed_tokens if turn else SegmentMode.fixed_tokens
    segments = (
        ContextFeatureSegment(
            feature_ids=(0,),
            buckets=(1,),
            char_start=0,
            char_end=int(offsets[-2] if turn else offsets[-1]),
        ),
    )
    if turn:
        segments += (
            ContextFeatureSegment(
                feature_ids=(1,),
                buckets=(1,),
                char_start=int(offsets[-2]),
                char_end=int(offsets[-1]),
            ),
        )
    provider = SingleEntryContextProvider(
        clean,
        segments,
        segment_mode=segment_mode,
        token_assignments=(*([0] * (len(clean) - 1)), 1) if turn else None,
    )
    config = replace(
        _make_interpolation_config(),
        self_cond_segment_size=PositiveInt.parse(cadence),
        segment_mode=segment_mode,
    )
    seed_text = "!" * suffix_token_count
    attacked = tokenizer.render_for_completion(empty_assistant_conversation("q\n\n" + seed_text))
    seed_ids = tokenizer.encode(seed_text)
    modules: tuple[tuple[str, t.nn.Embedding], ...] = (
        ("wte", cast(t.nn.Embedding, model.transformer["wte"])),
        *(
            (f"value:{layer}", cast(t.nn.Embedding, module))
            for layer, module in _value_embedding_modules(model)
        ),
    )
    embeddings = t.cat([module.weight.detach()[seed_ids] for _name, module in modules], dim=1)
    if perturb:
        rng = t.Generator().manual_seed(741)
        embeddings = embeddings + t.randn(embeddings.shape, generator=rng) * 0.7
    common: dict[str, Any] = {"model": model, "tokenizer": tokenizer, "context_provider": provider}
    baseline = TargetAwareEncoderDecoderEvalWrapper(
        **common,
        target=SelfConditionedTarget(),
        self_cond_config=copy.deepcopy(config),
    )
    actual = ProjectSoftPromptWrapper(
        **common,
        self_cond_config=copy.deepcopy(config),
        clean_token_ids=clean,
        attacked_token_ids=attacked,
        raw_prompt="q",
        seed_text=seed_text,
        embeddings=embeddings,
        embedding_modules=modules,
    )
    return {
        "actual": actual,
        "baseline": baseline,
        "model": model,
        "tokenizer": tokenizer,
        "clean": clean,
        "attacked": attacked,
        "embeddings": embeddings,
        "modules": modules,
        "config": config,
        "common": common,
    }


def _reference_prefill(
    fixture: dict[str, Any], cache: Any, *, attack: bool
) -> tuple[t.Tensor, Any]:
    """Spell out the pre-existing clean seed, manual tuple injection, and marker transition."""

    wrapper = fixture["baseline"]
    model = fixture["model"]
    runtime = wrapper._self_cond
    assert runtime is not None
    clean = fixture["clean"]
    clean_ids = t.tensor([clean], dtype=t.long)
    clean_cache = _new_kv(model) if attack else cache
    with wrapper.conditioning_context(context_token_ids=[clean]):
        encoder_out, clean_seg_map = wrapper._build_conditioning(clean_ids, kv_cache=clean_cache)
        seed = seed_self_conditioning_from_prompt(
            model=model,
            runtime=runtime,
            input_ids=clean_ids,
            encoder_out=encoder_out,
            seg_map=clean_seg_map,
            kv_cache=clean_cache,
        )
    if attack:
        attacked = fixture["attacked"]
        # The fixture has one character of user content and no tokenizer seam
        # merge. Construct routing independently of project_suffix_prefix.
        source_slot = int(clean_seg_map[0, -2])
        suffix_count = fixture["embeddings"].shape[0]
        prefix_map = t.cat(
            (clean_seg_map[:, :-2], t.full((1, 3 + suffix_count), source_slot)), dim=1
        )
        prefix_ids = t.tensor([attacked[:-1]], dtype=t.long)
        handles = []
        offset = 0
        try:
            for _name, module in fixture["modules"]:
                width = module.weight.shape[1]
                values = fixture["embeddings"][:, offset : offset + width]
                offset += width

                def inject(
                    _module: Any, _args: Any, output: t.Tensor, *, values: t.Tensor = values
                ) -> t.Tensor:
                    result = output.clone()
                    # Two structural prefix tokens, q, and two newline tokens.
                    result[:, 5 : 5 + values.shape[0]] = values.to(result)
                    return result

                handles.append(module.register_forward_hook(inject))
            model(
                prefix_ids,
                None,
                seg_map=prefix_map,
                kv_cache=cache,
                encoder_out=seed.state.encoder_out,
                encoder_strict=True,
                logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()
        marker = t.tensor([[attacked[-1]]], dtype=t.long)
    else:
        marker = clean_ids[:, -1:]
    if wrapper.is_turn_self_conditioned:
        result = self_conditioned_decode_step(
            model=model,
            provider=runtime.provider,
            state=seed.state,
            fp_buffer=(),
            tokens_in_seg=0,
            emitted_token=marker,
            kv_cache=cache,
            device=t.device("cpu"),
        )
        runtime.state = result.state
        runtime.fp_buffer = result.fp_buffer
        runtime.tokens_in_seg = result.tokens_in_seg
        logits = result.logits
    else:
        logits = model(
            marker,
            None,
            seg_map=t.full((1, 1), seed.state.target_slot_idx),
            kv_cache=cache,
            encoder_out=seed.state.encoder_out,
            encoder_strict=True,
        )
        runtime.state = seed.state
        runtime.fp_buffer = ()
        runtime.tokens_in_seg = 0
    return logits, seed


def _assert_runtime_equal(first: Any, second: Any) -> None:
    left, right = first.self_cond_runtime_snapshot(), second.self_cond_runtime_snapshot()
    assert left.state.target_slot_idx == right.state.target_slot_idx
    assert left.tokens_in_seg == right.tokens_in_seg
    assert t.equal(left.state.encoder_out.values, right.state.encoder_out.values)
    for name in ("flat_indices", "valid_mask", "batch_indices", "segment_indices"):
        assert t.equal(
            getattr(left.state.encoder_out.layout, name),
            getattr(right.state.encoder_out.layout, name),
        )
    assert left.state.encoder_out.layout.n_segments == right.state.encoder_out.layout.n_segments
    assert len(left.fp_buffer) == len(right.fp_buffer)
    for a, b in zip(left.fp_buffer, right.fp_buffer, strict=True):
        assert t.equal(a, b)
    assert t.equal(
        first._self_cond.config.selection.rng.get_state(),
        second._self_cond.config.selection.rng.get_state(),
    )


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cadence", [1, 3])
def test_extracted_original_prefill_preserves_exact_legacy_transition(
    turn: bool, cadence: int
) -> None:
    fixture = _fixture(turn, cadence)
    model = fixture["model"]
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        **fixture["common"],
        target=SelfConditionedTarget(),
        self_cond_config=copy.deepcopy(fixture["config"]),
    )
    expected_cache, actual_cache = _new_kv(model), _new_kv(model)
    with t.inference_mode():
        expected, _seed = _reference_prefill(fixture, expected_cache, attack=False)
        with wrapper.conditioning_context(context_token_ids=[fixture["clean"]]):
            actual = wrapper(t.tensor([fixture["clean"]]), kv_cache=actual_cache)
    assert t.equal(actual, expected)
    assert t.equal(actual_cache.k_cache, expected_cache.k_cache)
    assert t.equal(actual_cache.v_cache, expected_cache.v_cache)
    _assert_runtime_equal(wrapper, fixture["baseline"])


@pytest.mark.parametrize("turn", [False, True])
@pytest.mark.parametrize("cadence", [1, 3])
@pytest.mark.parametrize("suffix_token_count", [1, 3])
def test_soft_prefill_and_dynamic_decode_match_independent_oracle(
    turn: bool,
    cadence: int,
    suffix_token_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(turn, cadence, suffix_token_count=suffix_token_count)
    actual = fixture["actual"]
    expected_cache, actual_cache = _new_kv(fixture["model"]), _new_kv(fixture["model"])
    observed_clean: list[tuple[int, ...]] = []
    seeds: list[Any] = []
    real_seed = project_soft_prompt.seed_self_conditioning_from_prompt
    real_get = actual.context_provider.get

    def get(tokens: Any) -> Any:
        observed_clean.append(tuple(tokens))
        assert tuple(tokens) == tuple(fixture["clean"])
        return real_get(tokens)

    def seed(**kwargs: Any) -> Any:
        assert kwargs["kv_cache"] is not actual_cache
        assert t.equal(kwargs["input_ids"], t.tensor([fixture["clean"]]))
        result = real_seed(**kwargs)
        seeds.append(result)
        return result

    with t.inference_mode():
        expected_logits, expected_seed = _reference_prefill(fixture, expected_cache, attack=True)
        monkeypatch.setattr(actual.context_provider, "get", get)
        monkeypatch.setattr(project_soft_prompt, "seed_self_conditioning_from_prompt", seed)
        with actual.conditioning_context(context_token_ids=[fixture["attacked"]]):
            actual_logits = actual(t.tensor([fixture["attacked"]]), kv_cache=actual_cache)
            assert actual._active_context_token_ids == [fixture["attacked"]]
        assert observed_clean == [tuple(fixture["clean"])]
        assert len(seeds) == 1
        assert t.equal(seeds[0].source_logits, expected_seed.source_logits)
        assert t.equal(seeds[0].state.encoder_out.values, expected_seed.state.encoder_out.values)
        assert t.equal(actual_logits, expected_logits)
        assert actual_cache.get_pos() == expected_cache.get_pos() == len(fixture["attacked"])
        _assert_runtime_equal(actual, fixture["baseline"])
        initial_slot = actual.self_cond_runtime_snapshot().state.target_slot_idx
        # Multiple state refreshes exercise the unmodified inherited decoder.
        for token in (40, 41, 42, 43, 44, 45):
            actual_logits = actual(t.tensor([[token]]), kv_cache=actual_cache)
            expected_logits = fixture["baseline"](t.tensor([[token]]), kv_cache=expected_cache)
            assert t.equal(actual_logits, expected_logits)
            _assert_runtime_equal(actual, fixture["baseline"])
        assert actual.self_cond_runtime_snapshot().state.target_slot_idx > initial_slot
        assert t.equal(actual_cache.k_cache, expected_cache.k_cache)
        assert t.equal(actual_cache.v_cache, expected_cache.v_cache)
    assert actual._active_context_token_ids is None
    assert all(not module._forward_hooks for _name, module in fixture["modules"])


def test_nonseed_embedding_tuple_changes_actual_prefill_logits() -> None:
    changed = _fixture(True, 3, perturb=True)
    seeded = _fixture(True, 3, perturb=False)
    with t.inference_mode():

        def prefill(fixture: dict[str, Any]) -> t.Tensor:
            engine = InferenceEngine(fixture["actual"], fixture["tokenizer"], t.device("cpu"))
            return engine.prepare_self_conditioned_prefill(
                fixture["attacked"], max_new_tokens=4
            ).logits

        assert not t.equal(prefill(changed), prefill(seeded))


@pytest.mark.parametrize("failure_site", ["forward", "embedding"])
def test_failed_attacked_prefill_restores_hooks_and_active_context(
    failure_site: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(True, 1)
    wrapper, model = fixture["actual"], fixture["model"]
    registries = [
        (module._forward_hooks, module._forward_hooks.copy()) for module in model.modules()
    ]
    if failure_site == "forward":
        original = model.forward

        def failing_forward(ids: t.Tensor, *args: Any, **kwargs: Any) -> Any:
            if ids.shape[1] == len(fixture["attacked"]) - 1:
                raise RuntimeError("attacked forward failed")
            return original(ids, *args, **kwargs)

        monkeypatch.setattr(model, "forward", failing_forward)
    else:
        module = fixture["modules"][-1][1]
        original = module.forward

        def failing_embedding(ids: t.Tensor) -> Any:
            if ids.shape[1] == len(fixture["attacked"]) - 1:
                raise RuntimeError("attacked embedding failed")
            return original(ids)

        monkeypatch.setattr(module, "forward", failing_embedding)
    with t.inference_mode(), pytest.raises(RuntimeError, match="attacked .* failed"):
        engine = InferenceEngine(wrapper, fixture["tokenizer"], t.device("cpu"))
        engine.prepare_self_conditioned_prefill(fixture["attacked"], max_new_tokens=4)
    assert wrapper._active_context_token_ids is None
    for (registry, contents), module in zip(registries, model.modules(), strict=True):
        assert module._forward_hooks is registry and registry == contents


def test_soft_generation_checks_exact_prompt_and_context_budget() -> None:
    fixture = _fixture(True, 3)
    wrapper = fixture["actual"]
    with t.inference_mode(), pytest.raises(AxBenchAdvError, match="exact seeded prompt"):
        wrapper(t.tensor([fixture["clean"]]), kv_cache=_new_kv(fixture["model"]))
    engine = InferenceEngine(wrapper, fixture["tokenizer"], t.device("cpu"))
    with pytest.raises(ValueError, match="exceed|length"):
        engine.prepare_self_conditioned_prefill(
            fixture["attacked"],
            max_new_tokens=fixture["model"].dec_config.sequence_len,
        )
