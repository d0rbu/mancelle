"""Focused contracts for the standalone harmfulness adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.delta import DeltaMode
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.eval.chat_benchmark_targets import BenchmarkTargets
from steering.eval.harmfulness_safety_targets import SafetySteeringSettings, SafetyTargets
from steering.eval.harmfulness_self_conditioning import (
    HarmfulnessSafetySteering,
    generate_harmfulness_response_self_conditioned,
    harmfulness_prompt_seeds,
    parse_harmfulness_safety_settings,
    resolve_harmfulness_self_cond_setup,
)
from steering.eval.target_conditioning import token_context_key
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.test_self_conditioned_interpolation import _run_and_capture


class _ContextProvider:
    def get(self, token_ids):
        del token_ids
        return None


def _meta() -> dict[str, Any]:
    return {
        "user_config": {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 1,
                "max_k": 5,
                "mode": "top_k_percentile",
                "delta_weights": None,
            },
        },
        "pretrain_conditioning": {
            "negative_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "min_k": 0,
                "max_k": 5,
            },
            "feature_delta": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
        },
    }


def _setup(*, targets=None, strength=0.0, loss_fn=FeaturePredLossFn.BCE, meta=None):
    model = build_tiny_model(
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=loss_fn,
        ),
    )
    return resolve_harmfulness_self_cond_setup(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda ids: ids[0]),
        device=t.device("cpu"),
        context_provider=_ContextProvider(),
        meta=_meta() if meta is None else meta,
        feature_contract=CheckpointFeatureContract(None, None),
        segment_size=256,
        safety_steering=None if targets is None else HarmfulnessSafetySteering(
            SafetySteeringSettings(Path("unused-unit-policy-target"), strength), targets),
    )


def _targets(width: int, tokens=(1, 2)) -> SafetyTargets:
    return SafetyTargets(BenchmarkTargets("a" * 64, {"safety": t.linspace(-0.9, 0.9, width)},
        {"safety": (token_context_key(tokens),)}), {})


def test_setup_reuses_dense_interpolation_without_delta() -> None:
    base = _setup()
    setup = _setup(targets=_targets(int(base.policy.n_sae_features)), strength=2.5)
    cfg = setup.make_config(selection_seed=123)
    assert cfg.selection.selection.delta_weights is None
    assert float(cfg.score_policy.weight) == 2.5 / 6
    assert t.equal(cfg.score_policy.target_scores, t.linspace(-0.9, 0.9, int(base.policy.n_sae_features)))


@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_strength_exact_scores_and_selected_buckets(monkeypatch, sign) -> None:
    logits = t.tensor([[[-21., -20., 20., 21., 0., 1.]]])
    target = _targets(6)
    target.bundle.targets["safety"].mul_(sign)
    base = _setup()
    steered = target.policy(base.policy, 0)
    assert isinstance(steered.score_policy, SelfCondScoreInterpolation)
    zero_segments, zero_scores = _run_and_capture(monkeypatch,
        diffmean=t.zeros(6), weight=0, logits=logits)
    segments, scores = _run_and_capture(monkeypatch,
        diffmean=steered.score_policy.target_scores, weight=float(steered.score_policy.weight), logits=logits)
    assert segments == zero_segments
    assert t.equal(scores, zero_scores)
    assert t.equal(scores, logits[:, 0].float())


def test_setup_accepts_serialized_feature_prediction_enums() -> None:
    model = build_tiny_model(
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    assert model.feature_pred_config is not None
    model.feature_pred_config.loss_mode = cast(Any, "max_pool")
    model.feature_pred_config.loss_fn = cast(Any, "bce")

    setup = resolve_harmfulness_self_cond_setup(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda ids: ids[0]),
        device=t.device("cpu"),
        context_provider=_ContextProvider(),
        meta=_meta(),
        feature_contract=CheckpointFeatureContract(None, None),
        segment_size=256,
    )

    assert int(setup.segment_size) == 256


def test_prompt_seed_domains_indices_and_global_rng_are_independent() -> None:
    cpu_before = t.random.get_rng_state().clone()
    token_seed, selection_seed = harmfulness_prompt_seeds(
        base_seed=42,
        stable_pool_index=4,
        prompt="same text",
        pool="benign",
        category="benign",
    )

    assert token_seed != selection_seed
    assert harmfulness_prompt_seeds(
        base_seed=42,
        stable_pool_index=4,
        prompt="same text",
        pool="benign",
        category="benign",
    ) == (token_seed, selection_seed)
    assert harmfulness_prompt_seeds(
        base_seed=42,
        stable_pool_index=5,
        prompt="same text",
        pool="benign",
        category="benign",
    ) != (token_seed, selection_seed)
    assert t.equal(t.random.get_rng_state(), cpu_before)


def test_prompt_order_failure_and_subset_do_not_change_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    monkeypatch.setattr("steering.eval.harmfulness_self_conditioning.render_prompt_for_chat_completion",
        lambda _tokenizer, prompt: SimpleNamespace(token_ids=[1, 2, prompt]))
    builds: list[int] = []

    class _Engine:
        def __init__(self, selected_draw: int) -> None:
            self.selected_draw = selected_draw

        def generate(self, prompt: str, *, seed: int, **_kwargs) -> str:
            if prompt == "fail":
                raise RuntimeError("injected prompt failure")
            return f"{prompt}:{seed}:{self.selected_draw}"

    def build_engine(_model, _tokenizer, _device, config, **_kwargs):
        draw = int(t.randint(0, 2**31, (1,), generator=config.selection.rng).item())
        builds.append(draw)
        return _Engine(draw)

    monkeypatch.setattr(
        "steering.eval.harmfulness_self_conditioning.build_engine_self_conditioned",
        build_engine,
    )
    monkeypatch.setattr("steering.eval.harmfulness_self_conditioning.generate_self_conditioned_chat",
        lambda engine, ids, **kwargs: SimpleNamespace(token_ids=[*ids, engine.generate(ids[-1], **kwargs)]))
    rows = [
        (0, "A", "harmful", "Violence"),
        (1, "B", "benign", "benign"),
    ]

    def run(row: tuple[int, str, str, str]) -> str:
        index, prompt, pool, category = row
        return generate_harmfulness_response_self_conditioned(
            setup=setup,
            stable_pool_index=index,
            prompt=prompt,
            pool=pool,
            category=category,
            base_seed=42,
            max_new_tokens=8,
            temperature=0.7,
            top_k=5,
        )

    forward = {row[1]: run(row) for row in rows}
    reversed_order = {row[1]: run(row) for row in reversed(rows)}
    assert forward == reversed_order
    assert run(rows[1]) == forward["B"]
    with pytest.raises(RuntimeError, match="injected prompt failure"):
        run((0, "fail", "harmful", "Violence"))
    assert run(rows[1]) == forward["B"]
    assert len(builds) == 7


def test_setup_rejects_gradient_ranking_before_generation() -> None:
    meta = _meta()
    meta["user_config"]["feature_selection"]["gradient"] = {
        "schema_version": 1,
        "overlay_hash": "0" * 64,
    }
    model = build_tiny_model(
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )

    with pytest.raises(ValueError, match="gradient"):
        resolve_harmfulness_self_cond_setup(
            model=model,
            tokenizer=SimpleNamespace(decode=lambda ids: ids[0]),
            device=t.device("cpu"),
            context_provider=_ContextProvider(),
            meta=meta,
            feature_contract=CheckpointFeatureContract(None, None),
            segment_size=256,
        )


def test_non_bce_head_rejected_before_sigmoid_interpolation():
    with pytest.raises(ValueError, match="loss_fn=bce"):
        _setup(loss_fn=FeaturePredLossFn.MSE)


@pytest.mark.parametrize("delta", [
    {"mode": DeltaMode.prev_segment.value}, {"mode": DeltaMode.cum_max.value},
    {"mode": "unknown-delta"}, {}, "invalid-delta",
])
def test_incompatible_checkpoint_delta_fails_before_provider(monkeypatch, delta):
    meta = _meta()
    meta["pretrain_conditioning"]["feature_delta"] = delta
    monkeypatch.setattr("steering.eval.harmfulness_self_conditioning.SelfConditionedFeatureProvider",
        lambda *args, **kwargs: pytest.fail("provider before checkpoint delta validation"))
    with pytest.raises(ValueError, match="feature_delta"):
        _setup(meta=meta)


@pytest.mark.parametrize("delta", [None, {"mode": DeltaMode.splice_counterfactual.value}])
def test_admissible_checkpoint_delta_preserves_dense_policy(delta):
    meta = _meta()
    meta["pretrain_conditioning"]["feature_delta"] = delta
    setup = _setup(meta=meta)
    assert isinstance(setup.policy.score_policy, SelfCondScoreInterpolation)
    assert float(setup.policy.score_policy.weight) == 0


@pytest.mark.parametrize("block", [None, {}, {"safety_steering": None}])
def test_absent_safety_settings_are_the_same_unsteered_policy(block):
    assert parse_harmfulness_safety_settings({"self_conditioned": block}) is None


@pytest.mark.parametrize("block", [[], "safety", {"features": {}}, {"safety_steering": None, "typo": True}])
def test_safety_settings_reject_unknown_or_nonschema_options(block):
    with pytest.raises(ValueError, match="accepts only safety_steering"):
        parse_harmfulness_safety_settings({"self_conditioned": block})


@pytest.mark.parametrize("field,value,exception", [
    ("base_seed", True, TypeError),
    ("base_seed", 1.5, TypeError),
    ("stable_pool_index", True, ValueError),
    ("stable_pool_index", -1, ValueError),
    ("stable_pool_index", 1.5, ValueError),
])
def test_prompt_rng_identity_rejects_ambiguous_integer_inputs(field, value, exception):
    arguments = {"base_seed": 42, "stable_pool_index": 0,
        "prompt": "prompt", "pool": "harmful", "category": "test"}
    arguments[field] = value
    with pytest.raises(exception, match=field):
        harmfulness_prompt_seeds(**arguments)


@pytest.mark.parametrize("field,value", [
    ("base_seed", 43), ("stable_pool_index", 1),
    ("prompt", "different prompt"), ("pool", "benign"), ("category", "different category"),
])
def test_every_prompt_identity_field_separates_both_rng_streams(field, value):
    arguments = {"base_seed": 42, "stable_pool_index": 0,
        "prompt": "prompt", "pool": "harmful", "category": "test"}
    baseline = harmfulness_prompt_seeds(**arguments)
    arguments[field] = value
    changed = harmfulness_prompt_seeds(**arguments)
    assert all(first != second for first, second in zip(baseline, changed, strict=True))


def test_prompt_selection_generators_do_not_share_state_with_policy_or_other_prompts():
    setup = _setup()
    policy_state = setup.policy.selection.rng.get_state().clone()
    first = setup.make_config(selection_seed=123)
    second = setup.make_config(selection_seed=123)
    expected = t.randint(0, 2**31, (10,), generator=first.selection.rng)
    # A failed/interrupted prompt may consume arbitrarily many draws.
    t.randint(0, 2**31, (100,), generator=first.selection.rng)
    assert t.equal(t.randint(0, 2**31, (10,), generator=second.selection.rng), expected)
    assert t.equal(setup.policy.selection.rng.get_state(), policy_state)
