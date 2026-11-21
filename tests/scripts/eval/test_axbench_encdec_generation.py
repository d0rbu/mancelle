"""Pure self-conditioned generation for AxBench encoder-decoder baselines."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
import torch.nn as nn

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.axbench_upstream.encdec_baselines import generation as generation_module
from steering.eval.axbench_upstream.encdec_baselines.generation import (
    DirectionGenerationRequest,
    generate_encdec_baseline,
    generate_encdec_direction_batch,
    steering_delta,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import LORA_RANK, RsLoRAAdapter
from steering.eval.axbench_upstream.methods import MethodSetup, PreparedPromptContext
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import build_tiny_encdec
from tests.encdec_test_helpers import TURN_FIELD_DEFAULTS

_TINY_INTERVENTION_LAYER = 1
_TINY_LORA_LAYERS = (0, 1)


@pytest.fixture(autouse=True)
def _tiny_layer_contracts():
    """Use production mechanisms at layer indices present in the shared tiny model."""

    old_attention = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old_attention


def _block(model: nn.Module, layer: int) -> nn.Module:
    transformer = cast(nn.ModuleDict, model.transformer)
    blocks = cast(nn.ModuleList, transformer["h"])
    return blocks[layer]


def _clone_state_dict(model: nn.Module) -> dict[str, t.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_dict_equal(
    actual: dict[str, t.Tensor],
    expected: dict[str, t.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert t.equal(value, expected[name]), name


def _lora_state(model: nn.Module) -> dict[str, t.Tensor]:
    state: dict[str, t.Tensor] = {}
    for layer in _TINY_LORA_LAYERS:
        projection = cast(Any, _block(model, layer).attn.c_proj)
        prefix = f"layer_{layer}_attn_c_proj_lora_"
        state[f"{prefix}A"] = t.full((LORA_RANK, projection.in_features), 0.25)
        state[f"{prefix}B"] = t.full((projection.out_features, LORA_RANK), -0.5)
    return state


def _setup() -> Any:
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        delta_weights=DeltaSamplingWeights.percentile_delta,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    negative_selection = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=0,
        max_k=4,
    )
    return SimpleNamespace(
        n_sae_features=64,
        percentile_table=t.zeros((64, 99), dtype=t.float64),
        consolidation=None,
        output_selection=selection,
        output_negative_selection=negative_selection,
        feature_mask=None,
        **TURN_FIELD_DEFAULTS,
    )


class _TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(t.zeros(()))
        self.feature_pred_config = SimpleNamespace(loss_mode=FeaturePredLossMode.MAX_POOL)
        self.dec_config = SimpleNamespace(sequence_len=128)
        self.encoder = SimpleNamespace(config=SimpleNamespace(num_buckets=4))


class _HookableTargetModel(_TargetModel):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict({"h": nn.ModuleList([nn.Identity(), nn.Identity()])})


def _prepared(*, token_ids: tuple[int, ...] = (1, 2, 3)) -> PreparedPromptContext:
    return PreparedPromptContext(
        token_ids=token_ids,
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1,), (0,), 0, len(token_ids)),),
    )


@pytest.mark.parametrize("raises_inside", [False, True])
def test_steering_delta_hook_lifecycle_is_exception_safe(raises_inside: bool) -> None:
    model = build_tiny_encdec()
    block = _block(model, _TINY_INTERVENTION_LAYER)
    direction = t.ones(model.dec_config.n_embd)

    assert not block._forward_hooks
    if raises_inside:
        with pytest.raises(RuntimeError, match="intentional hook-context failure"):
            with steering_delta(
                model,
                direction=direction,
                coefficient=1.0,
                layer=_TINY_INTERVENTION_LAYER,
            ):
                assert len(block._forward_hooks) == 1
                raise RuntimeError("intentional hook-context failure")
    else:
        with steering_delta(
            model,
            direction=direction,
            coefficient=1.0,
            layer=_TINY_INTERVENTION_LAYER,
        ):
            assert len(block._forward_hooks) == 1
    assert not block._forward_hooks


@pytest.mark.parametrize("raises_inside", [False, True])
def test_steering_delta_lora_lifecycle_is_byte_exact_and_exception_safe(
    raises_inside: bool,
) -> None:
    model = build_tiny_encdec()
    before = _clone_state_dict(model)
    state = _lora_state(model)

    def assert_attached() -> None:
        for layer in _TINY_LORA_LAYERS:
            adapter = _block(model, layer).attn.c_proj
            assert isinstance(adapter, RsLoRAAdapter)
            prefix = f"layer_{layer}_attn_c_proj_lora_"
            assert t.equal(adapter.lora_A, state[f"{prefix}A"])
            assert t.equal(adapter.lora_B, state[f"{prefix}B"])

    if raises_inside:
        with pytest.raises(RuntimeError, match="intentional lora-context failure"):
            with steering_delta(model, lora_state=state, lora_layers=_TINY_LORA_LAYERS):
                assert_attached()
                raise RuntimeError("intentional lora-context failure")
    else:
        with steering_delta(model, lora_state=state, lora_layers=_TINY_LORA_LAYERS):
            assert_attached()
    _assert_state_dict_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    ("direction", "coefficient"),
    [(t.ones(64), None), (None, 1.0)],
)
def test_steering_delta_rejects_half_set_direction_pair(
    direction: t.Tensor | None,
    coefficient: float | None,
) -> None:
    with pytest.raises(ValueError, match="direction and coefficient must be provided together"):
        with steering_delta(
            build_tiny_encdec(),
            direction=direction,
            coefficient=coefficient,
        ):
            pytest.fail("invalid steering context yielded")


def test_steering_delta_rejects_both_modes() -> None:
    with pytest.raises(ValueError, match="direction and lora_state are mutually exclusive"):
        with steering_delta(
            build_tiny_encdec(),
            direction=t.ones(64),
            coefficient=1.0,
            lora_state={},
        ):
            pytest.fail("invalid steering context yielded")


def test_steering_delta_rejects_no_arguments() -> None:
    with pytest.raises(
        ValueError,
        match="steering_delta requires a direction/coefficient pair or lora_state",
    ):
        with steering_delta(build_tiny_encdec()):
            pytest.fail("invalid steering context yielded")


def test_steering_delta_requires_layer_for_direction() -> None:
    with pytest.raises(ValueError, match="layer is required"):
        with steering_delta(
            build_tiny_encdec(),
            direction=t.ones(64),
            coefficient=1.0,
        ):
            pytest.fail("invalid steering context yielded")


def test_steering_delta_requires_layers_for_lora() -> None:
    with pytest.raises(ValueError, match="lora_layers is required"):
        with steering_delta(build_tiny_encdec(), lora_state={}):
            pytest.fail("invalid steering context yielded")


def _forward_logits(model: nn.Module) -> t.Tensor:
    enc_ids, enc_magnitudes, enc_mask = build_segmented_feature_encoder_inputs(
        [{2: 1, 5: 2}],
        "cpu",
    )
    token_ids = t.tensor([[10, 11, 12, 13]], dtype=t.int64)
    seg_map = t.zeros_like(token_ids)
    with t.no_grad():
        return model(
            token_ids,
            None,
            enc_feature_ids=enc_ids,
            enc_magnitudes=enc_magnitudes,
            enc_mask=enc_mask,
            seg_map=seg_map,
            encoder_strict=True,
        )


def test_steering_delta_zero_is_exact_noop_and_nonzero_changes_logits() -> None:
    t.manual_seed(0)
    model = build_tiny_encdec()
    direction = t.linspace(-1.0, 1.0, model.dec_config.n_embd)

    baseline = _forward_logits(model)
    with steering_delta(model, direction=direction, coefficient=0.0, layer=_TINY_INTERVENTION_LAYER):
        zero = _forward_logits(model)
    with steering_delta(model, direction=direction, coefficient=1.0, layer=_TINY_INTERVENTION_LAYER):
        nonzero = _forward_logits(model)

    assert t.equal(zero, baseline)
    assert not t.equal(nonzero, baseline)


def test_generate_encdec_baseline_mirrors_self_conditioned_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def generate(
        setup: MethodSetup,
        model: Any,
        tokenizer: Any,
        prepared: PreparedPromptContext,
        policy: Any,
        *,
        selection_seed: int,
        seed: int,
        max_new_tokens: int,
        profile_driver: str,
    ) -> str:
        recorded.update(locals())
        return "generated"

    monkeypatch.setattr(generation_module, "_generate_self_conditioning", generate)
    model = _TargetModel()
    tokenizer = object()
    prepared = _prepared()
    setup = _setup()

    result = generate_encdec_baseline(
        setup,
        model,
        tokenizer,
        prepared,
        selection_seed=11,
        seed=12,
        max_new_tokens=16,
    )

    assert result == "generated"
    assert recorded["setup"] is setup
    assert recorded["model"] is model
    assert recorded["tokenizer"] is tokenizer
    assert recorded["prepared"] is prepared
    assert recorded["policy"] is None
    assert recorded["selection_seed"] == 11
    assert recorded["seed"] == 12
    assert recorded["max_new_tokens"] == 16
    assert recorded["profile_driver"] == "encdec"


def test_generate_encdec_baseline_rejects_prompt_plus_budget_over_sequence_length() -> None:
    prepared = _prepared(token_ids=tuple(range(120)))
    with pytest.raises(
        ValueError,
        match="prepared prompt plus generation budget exceeds decoder sequence length",
    ):
        generate_encdec_baseline(
            _setup(),
            _TargetModel(),
            object(),
            prepared,
            selection_seed=1,
            seed=2,
            max_new_tokens=9,
        )


@pytest.mark.parametrize("raises_during_batch", [False, True])
def test_generate_encdec_direction_batch_preserves_row_contract_and_hook_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    raises_during_batch: bool,
) -> None:
    model = _HookableTargetModel()
    block = _block(model, 1)
    prepared = _prepared()
    requests = (
        DirectionGenerationRequest(t.tensor([1.0, 2.0]), 0.5, 1, 11, 101),
        DirectionGenerationRequest(t.tensor([-3.0, 4.0]), -1.5, 1, 12, 102),
    )
    recorded: dict[str, Any] = {"configs": [], "prefills": []}

    class Engine:
        def __init__(self, row: int) -> None:
            self.row = row

        def prepare_self_conditioned_prefill(
            self,
            tokens: list[int],
            *,
            max_new_tokens: int,
        ) -> str:
            assert list(block._forward_hooks) == [next(iter(block._forward_hooks))]
            assert tokens == list(prepared.token_ids)
            assert max_new_tokens == 4
            marker = f"prefill-{self.row}"
            recorded["prefills"].append(marker)
            return marker

        def generate_self_conditioned_batch(self, prefills: list[str], **kwargs: Any):
            assert self.row == 1
            assert len(block._forward_hooks) == 1
            recorded["batch_prefills"] = list(prefills)
            recorded["batch_kwargs"] = kwargs
            if raises_during_batch:
                raise RuntimeError("intentional batched decode failure")
            return [[7], [8, 9]]

    def build_engine(
        builder_model: nn.Module,
        tokenizer: Any,
        device: t.device,
        config: Any,
        *,
        context_provider: Any,
    ) -> Engine:
        del tokenizer
        assert builder_model is model
        assert device == model.anchor.device
        assert not block._forward_hooks
        recorded["configs"].append(config)
        assert context_provider.get(prepared.token_ids) == prepared.segments
        return Engine(len(recorded["configs"]) - 1)

    monkeypatch.setattr(generation_module, "build_engine_self_conditioned", build_engine)
    tokenizer = SimpleNamespace(
        get_bos_token_id=lambda: 1,
        decode=lambda values: f"decoded:{values}",
    )

    if raises_during_batch:
        with pytest.raises(RuntimeError, match="intentional batched decode failure"):
            generate_encdec_direction_batch(
                _setup(), model, tokenizer, prepared, requests, max_new_tokens=4
            )
        assert not block._forward_hooks
        return

    actual = generate_encdec_direction_batch(
        _setup(), model, tokenizer, prepared, requests, max_new_tokens=4
    )

    assert actual == ("decoded:[7]", "decoded:[8, 9]")
    assert [config.selection.rng.initial_seed() for config in recorded["configs"]] == [11, 12]
    assert recorded["prefills"] == ["prefill-0", "prefill-1"]
    assert recorded["batch_prefills"] == recorded["prefills"]
    assert recorded["batch_kwargs"] == {
        "seeds": [101, 102],
        "max_new_tokens": 4,
        "temperature": 1.0,
        "top_k": 50,
        "stop_token_ids": {1, 9},
    }
    assert not block._forward_hooks


def test_generate_encdec_direction_batch_rejects_turn_before_engine_or_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    setup.segment_mode = generation_module.SegmentMode.turn_delimited_fixed_tokens
    requests = (
        DirectionGenerationRequest(t.tensor([1.0, 2.0]), 0.5, 1, 11, 101),
        DirectionGenerationRequest(t.tensor([-3.0, 4.0]), -1.5, 1, 12, 102),
    )
    monkeypatch.setattr(
        generation_module,
        "build_engine_self_conditioned",
        lambda *_args, **_kwargs: pytest.fail("turn mode constructed a batch engine"),
    )
    monkeypatch.setattr(
        generation_module,
        "register_batched_residual_hook",
        lambda *_args, **_kwargs: pytest.fail("turn mode installed a batched hook"),
    )

    with pytest.raises(
        ValueError,
        match="supports only fixed_tokens.*requires the scalar cell dispatcher",
    ):
        generate_encdec_direction_batch(
            setup,
            _HookableTargetModel(),
            object(),
            _prepared(),
            requests,
            max_new_tokens=4,
        )


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        (
            (DirectionGenerationRequest(t.ones(2), 1.0, 1, 11, 101),),
            "requires at least two rows",
        ),
        (
            (
                DirectionGenerationRequest(t.ones(2), 1.0, 0, 11, 101),
                DirectionGenerationRequest(t.ones(2), 1.0, 1, 12, 102),
            ),
            "share one intervention layer",
        ),
        (
            (
                DirectionGenerationRequest(t.ones(1, 2), 1.0, 1, 11, 101),
                DirectionGenerationRequest(t.ones(2), 1.0, 1, 12, 102),
            ),
            "all have shape",
        ),
        (
            (
                DirectionGenerationRequest(t.ones(2), 1.0, 1, 11, 101),
                DirectionGenerationRequest(t.ones(3), 1.0, 1, 12, 102),
            ),
            "share one width",
        ),
    ],
)
def test_generate_encdec_direction_batch_rejects_incompatible_rows(
    requests: tuple[DirectionGenerationRequest, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_encdec_direction_batch(
            _setup(),
            _HookableTargetModel(),
            object(),
            _prepared(),
            requests,
            max_new_tokens=4,
        )
