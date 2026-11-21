from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.eval.roundtrip_steering as roundtrip_steering
from steering.eval.probe_model import ProbeModel
from steering.eval.roundtrip_steering import (
    RoundtripProbeConfig,
    load_probe_model_for_roundtrip,
    parse_roundtrip_eval_config,
    probe_text_against_features,
)
from steering.sae_resolver import ActivationType, L0Tier

_EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return [ord(ch) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(int(token)) for token in tokens)


class _FakeModel:
    def __init__(self) -> None:
        self._param = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        return iter([self._param])


class _FakeSAECfg:
    activation_fn = "relu"

    def __init__(self, apply_b_dec_to_input: bool = True) -> None:
        self.apply_b_dec_to_input = apply_b_dec_to_input


class _FakeSAE:
    def __init__(self) -> None:
        self.W_enc = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.b_enc = t.zeros(2)
        self.b_dec = t.zeros(2)
        self.cfg = _FakeSAECfg()


class _TrainableFakeSAE(t.nn.Module):
    def __init__(self, *, apply_b_dec_to_input: bool) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(t.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]))
        self.b_enc = t.nn.Parameter(t.tensor([0.5, -0.25, 1.0]))
        self.b_dec = t.nn.Parameter(t.tensor([0.25, -0.5]))
        self.cfg = _FakeSAECfg(apply_b_dec_to_input)


def _fake_probe() -> ProbeModel:
    probe = ProbeModel.__new__(ProbeModel)
    probe.model = _FakeModel()
    probe.sae = _FakeSAE()
    probe.tokenizer = _FakeTokenizer()
    probe.hook_name = "blocks.8.hook_resid_post"
    probe.layer = 8
    probe.nnterp_model = None
    probe.activation_type = ActivationType.residual
    probe.apply_b_dec = True
    return probe


def _trainable_fake_probe(*, apply_b_dec_to_input: bool) -> ProbeModel:
    probe = _fake_probe()
    probe.sae = _TrainableFakeSAE(
        apply_b_dec_to_input=apply_b_dec_to_input,
    )
    probe.apply_b_dec = apply_b_dec_to_input
    return probe


def _fake_activations(
    _model,
    tokens,
    layer,
    hook_name=None,
    activation_type=None,
    nnterp_model=None,
):
    del layer, hook_name, activation_type, nnterp_model
    positions = t.arange(tokens.shape[1], dtype=t.float32)
    return t.stack([positions, positions + 10.0], dim=-1).unsqueeze(0)


def test_probe_text_uses_exact_prompt_plus_response_without_bos_stripping():
    probe = _fake_probe()

    scores = probe_text_against_features(
        prompt_text="<bos>ab",
        response_text="cd",
        feature_ids=[0],
        probe_model=probe,
        percentile_table=None,
        feature_chunk_size=1,
        get_activations_fn=_fake_activations,
    )

    assert probe.tokenizer.encoded_texts == ["<bos>abcd"]
    assert scores.pre_act_mean == pytest.approx(8.0)


@pytest.mark.parametrize("apply_b_dec_to_input", [False, True])
def test_probe_text_disables_grad_for_unfrozen_sae(
    apply_b_dec_to_input: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    probe = _trainable_fake_probe(
        apply_b_dec_to_input=apply_b_dec_to_input,
    )
    frozen_probe = _trainable_fake_probe(
        apply_b_dec_to_input=apply_b_dec_to_input,
    )
    frozen_probe.sae.load_state_dict(probe.sae.state_dict())
    for parameter in frozen_probe.sae.parameters():
        parameter.requires_grad_(False)

    grad_states: list[bool] = []
    activation_is_inference: list[bool] = []
    real_compute = roundtrip_steering.compute_raw_probe_batched

    def _spy_compute(*args, **kwargs):
        grad_states.append(t.is_grad_enabled())
        result = real_compute(*args, **kwargs)
        assert not result.requires_grad
        return result

    def _activations(*args, **kwargs):
        result = _fake_activations(*args, **kwargs)
        activation_is_inference.append(t.is_inference(result))
        return result

    monkeypatch.setattr(
        roundtrip_steering,
        "compute_raw_probe_batched",
        _spy_compute,
    )
    with t.enable_grad():
        scores = probe_text_against_features(
            prompt_text="ab",
            response_text="cd",
            feature_ids=[2, 0, 1],
            probe_model=probe,
            percentile_table=None,
            feature_chunk_size=1,
            get_activations_fn=_activations,
        )
        assert t.is_grad_enabled()

    frozen_scores = probe_text_against_features(
        prompt_text="ab",
        response_text="cd",
        feature_ids=[2, 0, 1],
        probe_model=frozen_probe,
        percentile_table=None,
        feature_chunk_size=1,
        get_activations_fn=_fake_activations,
    )

    assert scores.n_features == frozen_scores.n_features
    assert scores.pre_act_mean == frozen_scores.pre_act_mean
    assert scores.post_act_mean == frozen_scores.post_act_mean
    assert t.isnan(t.tensor(scores.percentile_mean))
    assert t.isnan(t.tensor(frozen_scores.percentile_mean))
    assert activation_is_inference == [True]
    assert grad_states == [False] * 6
    assert all(parameter.grad is None for parameter in probe.sae.parameters())


def test_probe_text_disables_grad_for_percentile_scoring():
    probe = _trainable_fake_probe(apply_b_dec_to_input=False)
    percentile_table = t.stack([t.arange(1, 100, dtype=t.float64) for _ in range(3)])

    with t.enable_grad():
        scores = probe_text_against_features(
            prompt_text="ab",
            response_text="cd",
            feature_ids=[0, 1, 2],
            probe_model=probe,
            percentile_table=percentile_table,
            feature_chunk_size=2,
            get_activations_fn=_fake_activations,
        )
        assert t.is_grad_enabled()

    assert scores.n_features == 3
    assert scores.percentile_mean == pytest.approx((3.5 + 12.75 + 0.5) / 3)
    assert all(parameter.grad is None for parameter in probe.sae.parameters())


@pytest.mark.parametrize(
    "config_name",
    ["harmfulness_eval", "chat_eval", "dump_chat_responses", "training_progression"],
)
def test_generic_eval_configs_do_not_expose_roundtrip_path(config_name: str):
    cfg = OmegaConf.load(_EVAL_CONFIG_DIR / f"{config_name}.yaml")
    assert "roundtrip_steering" not in cfg


def test_probe_text_rejects_empty_completion_mask():
    with pytest.raises(ValueError, match="completion mask is all-False"):
        probe_text_against_features(
            prompt_text="abc",
            response_text="",
            feature_ids=[0],
            probe_model=_fake_probe(),
            percentile_table=None,
            get_activations_fn=_fake_activations,
        )


@pytest.mark.parametrize(
    ("raw_l0", "expected_l0"),
    [
        (None, None),
        (42, 42),
        ("42", 42),
        ("small", L0Tier.small),
        ("medium", L0Tier.medium),
        ("big", L0Tier.big),
    ],
)
def test_roundtrip_probe_config_parses_l0(raw_l0, expected_l0):
    probe = RoundtripProbeConfig(l0=cast(Any, raw_l0))
    assert probe.l0 == expected_l0


@pytest.mark.parametrize("raw_l0", [True, False, "invalid", 1.5])
def test_roundtrip_probe_config_rejects_invalid_l0(raw_l0):
    with pytest.raises(ValueError):
        RoundtripProbeConfig(l0=cast(Any, raw_l0))


def test_parse_roundtrip_config_round_trips_probe_identity():
    cfg = OmegaConf.create(
        {"roundtrip_steering": {"enabled": True, "probe": {"l0": "medium"}}}
    )
    parsed = parse_roundtrip_eval_config(cfg)
    reparsed = parse_roundtrip_eval_config(
        {"roundtrip_steering": {"enabled": parsed.enabled, "probe": asdict(parsed.probe)}}
    )
    assert reparsed == parsed


def test_parse_roundtrip_config_wraps_invalid_chunk_size():
    with pytest.raises(ValueError, match="feature_chunk_size"):
        parse_roundtrip_eval_config(
            {"roundtrip_steering": {"feature_chunk_size": 0}}
        )


def test_numeric_l0_reaches_probe_loader_unchanged(monkeypatch):
    parsed = parse_roundtrip_eval_config(
        {"roundtrip_steering": {"probe": {"l0": "42", "device": "cpu"}}}
    )
    captured: dict[str, object] = {}
    expected_probe = cast(ProbeModel, object())

    def _fake_load_probe_model(**kwargs: object) -> ProbeModel:
        captured.update(kwargs)
        return expected_probe

    monkeypatch.setattr(
        "steering.eval.roundtrip_steering.load_probe_model",
        _fake_load_probe_model,
    )
    assert load_probe_model_for_roundtrip(parsed.probe) is expected_probe
    assert captured["l0"] == 42
