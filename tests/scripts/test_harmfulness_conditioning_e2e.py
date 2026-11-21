"""Focused orchestration tests for standalone harmfulness generation."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval import harmfulness_eval
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.harmfulness_artifact import (
    build_harmfulness_responses_artifact,
    write_harmfulness_responses_artifact,
)
from tests.scripts.test_harmfulness_eval_compliance import CONDITIONED_PROVENANCE
from tests.steering.test_harmfulness_generation_provenance import (
    _generation,
    _self_conditioned_generation,
)


class _EncDecStub(EncoderDecoderGPT):
    def __init__(self) -> None:
        t.nn.Module.__init__(self)


class _DecoderStub(t.nn.Module):
    pass


def _cfg(*, encdec: bool):
    return OmegaConf.create(
        {
            "source": "encdec" if encdec else "base",
            "model_tag": "d-test",
            "step": 1,
            "judge_only": False,
            "responses_file": None,
            "output_file": None,
            "feature_configs": None,
            "self_conditioned": {"safety_steering": None},
            "self_cond_segment_size": 256 if encdec else None,
            "segment_size": None,
            "feature_selection": None,
            "feature_mask": None,
            "num_harmful": 1,
            "num_benign": 1,
            "seed": 17,
            "batch_size": 1 if encdec else 8,
            "max_new_tokens": 4,
            "temperature": 0.25,
            "top_k": 17,
            "judge_model": "test",
            "cache_path": "",
            "requests_per_minute": 1,
        }
    )


def _checkpoint_identity() -> dict:
    return {
        "format": "nanochat_checkpoint",
        "backend": "nanochat",
        "architecture": "GPT",
        "model_type": "decoder_only",
        "step": 1,
        "checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
    }


def _pairs() -> list[dict[str, str]]:
    return [
        {"prompt": "harmful", "pool": "harmful", "category": "Violence"},
        {"prompt": "benign", "pool": "benign", "category": "benign"},
    ]


def _patch_main_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg,
    model: t.nn.Module,
) -> tuple[MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.encode_special.return_value = 3
    tokenizer.render_for_completion.return_value = [1, 2]
    contract = object() if isinstance(model, EncoderDecoderGPT) else None
    identity = _checkpoint_identity()
    if isinstance(model, EncoderDecoderGPT):
        identity.update(architecture="EncoderDecoderGPT", model_type="encoder_decoder")
    monkeypatch.setattr(harmfulness_eval, "initialize_config_dir", lambda **_kw: nullcontext())
    monkeypatch.setattr(harmfulness_eval, "compose", lambda **_kw: cfg)
    monkeypatch.setattr(harmfulness_eval.OmegaConf, "to_yaml", lambda _cfg: "fixture")
    monkeypatch.setattr(harmfulness_eval, "autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr(
        harmfulness_eval,
        "_load_model_for_harmfulness",
        lambda _cfg, _device: (
            model,
            tokenizer,
            {"checkpoint": "metadata"},
            contract,
            identity,
        ),
    )
    monkeypatch.setattr(harmfulness_eval, "_load_expected_prompt_pairs", lambda _cfg: _pairs())
    monkeypatch.setattr(
        harmfulness_eval,
        "_expected_harmfulness_context_records",
        lambda _cfg, _tokenizer: [object(), object()],
    )
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **_kw: MagicMock())
    monkeypatch.setattr(harmfulness_eval.sys, "argv", ["harmfulness_eval.py"])

    current = MagicMock(return_value=({"current": True}, "current artifact"))
    monkeypatch.setattr(harmfulness_eval, "current_harmfulness_payload", current)
    get_responses = MagicMock()
    return current, get_responses


@pytest.mark.parametrize("source", ["encdec", "capabilities_sft_conditioned_checkpoints"])
@pytest.mark.parametrize("safety_strength", [None, 0.0, 2.5])
def test_encoder_decoder_uses_self_conditioning_and_stable_prompt_indices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    source: str,
    safety_strength: float | None,
) -> None:
    cfg = _cfg(encdec=True)
    cfg.source = source
    model = _EncDecStub()
    current, _unused = _patch_main_common(monkeypatch, cfg=cfg, model=model)
    provider = MagicMock(spec=SafetensorsContextFeatureProvider)
    provider.manifest = {"schema_version": 1, "rows": 2}
    coverage = SimpleNamespace(expected_prompts=2)
    setup = MagicMock(context_provider=provider, segment_size=256, safety_strength=0.0, safety_steering=None, assert_target_unchanged=MagicMock())
    digest = hashlib.sha256(b"authenticated target fixture").hexdigest()
    if safety_strength is not None:
        targets = tmp_path / "targets.safetensors"
        targets.write_bytes(b"authenticated target fixture")
        cfg.self_conditioned.safety_steering = {"targets": str(targets), "strength": safety_strength}
        setup.safety_strength = safety_strength
        setup.safety_steering = SimpleNamespace(targets=SimpleNamespace(sha256=digest))
    resolve_setup = MagicMock(return_value=setup)
    generate_one = MagicMock(side_effect=["safe", "helpful"])
    coverage_check = MagicMock(return_value=coverage)
    monkeypatch.setattr(harmfulness_eval, "assert_exact_prompt_coverage", coverage_check)
    monkeypatch.setattr(
        harmfulness_eval,
        "conditioning_provenance",
        lambda *_args, **_kwargs: {
            "conditioning_mode": "conditions_within_vocab",
            "eval_name": "harmfulness_eval",
        },
    )
    monkeypatch.setattr(harmfulness_eval, "prepare_harmfulness_self_conditioning", resolve_setup)
    monkeypatch.setattr(
        harmfulness_eval,
        "generate_harmfulness_response_self_conditioned",
        generate_one,
    )

    def get_responses(
        _cfg,
        condition_name,
        response_generator,
        *,
        generation_conditioning,
        generation_provenance,
    ):
        assert condition_name == ("self_conditioned" if safety_strength is None else "safety_steered")
        assert generation_conditioning["conditioning_mode"] == "conditions_within_vocab"
        expected_completion = generation_provenance["completion"]
        assert len(generation_provenance["context_cache_manifest_sha256"]) == 64
        assert expected_completion["strength"] == (0.0 if safety_strength is None else safety_strength)
        assert expected_completion["safety_steering"] == (None if safety_strength is None else {
            "targets_sha256": digest, "strength": safety_strength})
        assert expected_completion["action_ranking_policy"] == "pooled_fp_logits_v1"
        indexed = list(enumerate(_pairs()))
        responses = [response_generator([item])[0] for item in indexed]
        assert responses == ["safe", "helpful"]
        pairs = [
            {**pair, "response": response}
            for pair, response in zip(_pairs(), responses, strict=True)
        ]
        return (
            condition_name,
            "/tmp/result.yaml",
            pairs,
            generation_conditioning,
            generation_provenance,
        )

    monkeypatch.setattr(harmfulness_eval, "_get_responses", get_responses)
    harmfulness_eval.main()

    coverage_check.assert_called_once()
    resolve_setup.assert_called_once()
    assert [call.kwargs["stable_pool_index"] for call in generate_one.call_args_list] == [0, 1]
    assert all(call.kwargs["setup"] is setup for call in generate_one.call_args_list)
    assert current.call_count == 1


def test_encoder_decoder_missing_context_provider_fails_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(encdec=True)
    model = _EncDecStub()
    _patch_main_common(monkeypatch, cfg=cfg, model=model)
    resolve_setup = MagicMock(return_value=SimpleNamespace(context_provider=None))
    monkeypatch.setattr(harmfulness_eval, "prepare_harmfulness_self_conditioning", resolve_setup)

    with pytest.raises(ValueError, match="resolved no context-feature provider"):
        harmfulness_eval.main()
    resolve_setup.assert_called_once()


def test_decoder_only_keeps_existing_batch_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(encdec=False)
    model = _DecoderStub()
    current, _unused = _patch_main_common(monkeypatch, cfg=cfg, model=model)
    engine = MagicMock()
    engine.generate_batch.return_value = ["safe", "helpful"]
    monkeypatch.setattr(harmfulness_eval, "InferenceEngine", lambda *_args: engine)
    monkeypatch.setattr(
        harmfulness_eval,
        "conditioning_provenance",
        lambda *_args, **_kwargs: {
            "conditioning_mode": "unconditioned",
            "eval_name": "harmfulness_eval",
        },
    )

    def get_responses(
        _cfg,
        condition_name,
        response_generator,
        *,
        generation_conditioning,
        generation_provenance,
    ):
        assert condition_name == "unconditional"
        assert generation_conditioning["conditioning_mode"] == "unconditioned"
        assert generation_provenance["completion"] is None
        responses = response_generator(list(enumerate(_pairs())))
        pairs = [
            {**pair, "response": response}
            for pair, response in zip(_pairs(), responses, strict=True)
        ]
        return (
            condition_name,
            "/tmp/result.yaml",
            pairs,
            generation_conditioning,
            generation_provenance,
        )

    monkeypatch.setattr(harmfulness_eval, "_get_responses", get_responses)
    harmfulness_eval.main()

    engine.generate_batch.assert_called_once_with(
        ["harmful", "benign"],
        max_new_tokens=4,
        temperature=0.25,
        top_k=17,
        seed=17,
        render_as_chat=True,
    )
    assert current.call_count == 1


def test_generate_only_publishes_responses_without_constructing_judge(monkeypatch, tmp_path):
    cfg = _cfg(encdec=False)
    cfg.generate_only = True
    model = _DecoderStub()
    _patch_main_common(monkeypatch, cfg=cfg, model=model)
    engine = MagicMock()
    engine.generate_batch.return_value = ["safe", "helpful"]
    monkeypatch.setattr(harmfulness_eval, "InferenceEngine", lambda *_args: engine)
    monkeypatch.setattr(harmfulness_eval, "conditioning_provenance", lambda *args, **kwargs: {
        "conditioning_mode": "unconditioned", "eval_name": "harmfulness_eval"})
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **kwargs: pytest.fail("paid judge constructed"))
    monkeypatch.setattr(harmfulness_eval, "current_harmfulness_payload", lambda *args, **kwargs: pytest.fail("judge path entered"))
    published = []

    def get_responses(_cfg, name, generator, *, generation_conditioning, generation_provenance):
        responses = generator(list(enumerate(_pairs())))
        published.extend(responses)
        return name, str(tmp_path / "results.yaml"), _pairs(), generation_conditioning, generation_provenance

    monkeypatch.setattr(harmfulness_eval, "_get_responses", get_responses)
    harmfulness_eval.main()
    assert published == ["safe", "helpful"]


@pytest.mark.parametrize("key", ["judge_only", "responses_file"])
def test_generate_only_rejects_incompatible_modes(monkeypatch, key):
    cfg = _cfg(encdec=False)
    cfg.generate_only = True
    cfg[key] = True if key == "judge_only" else "responses.yaml"
    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda _argv: cfg)
    with pytest.raises(ValueError, match="generate_only cannot"):
        harmfulness_eval.main()


@pytest.mark.parametrize("feature_configs", [
    [], {}, False, 0, "",
    *[[{"name": "manual", "features": features, "strength": 0}] for features in ({7: 1}, {})],
])
def test_cli_rejects_mixed_sparse_and_safety_steering_before_model_load(monkeypatch, feature_configs):
    cfg = _cfg(encdec=True)
    cfg.feature_configs = feature_configs
    cfg.self_conditioned.safety_steering = {"targets": "/missing-target", "strength": 2.5}
    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda argv: cfg)
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness",
        lambda *args: pytest.fail("model loaded before mixed steering guard"))
    with pytest.raises(ValueError, match="hand-supplied feature_configs"):
        harmfulness_eval.main()


@pytest.mark.parametrize("value", [1, "true", None])
def test_generate_only_requires_an_explicit_boolean_before_model_load(monkeypatch, value):
    cfg = _cfg(encdec=False)
    cfg.generate_only = value
    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda argv: cfg)
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness",
        lambda *args: pytest.fail("model loaded before mode validation"))
    with pytest.raises(ValueError, match="generate_only must be a boolean"):
        harmfulness_eval.main()


def test_decoder_only_still_rejects_cadence_after_architecture_resolution(monkeypatch):
    cfg = _cfg(encdec=False)
    cfg.self_cond_segment_size = 256
    _patch_main_common(monkeypatch, cfg=cfg, model=_DecoderStub())
    monkeypatch.setattr(harmfulness_eval, "InferenceEngine",
        lambda *args: pytest.fail("engine built before decoder-only cadence rejection"))
    with pytest.raises(ValueError, match="decoder-only harmfulness requires self_cond_segment_size=null"):
        harmfulness_eval.main()


@pytest.mark.parametrize("fault", ["different_digest", "different_strength", "missing_completion"])
def test_judge_only_rejects_responses_for_another_safety_intervention(monkeypatch, tmp_path, fault):
    cfg = _cfg(encdec=True)
    cfg.judge_only = True
    path = tmp_path / "targets.safetensors"
    path.write_bytes(b"requested targets")
    cfg.self_conditioned.safety_steering = {"targets": str(path), "strength": 2.5}
    current, _ = _patch_main_common(monkeypatch, cfg=cfg, model=_EncDecStub())
    generation = _self_conditioned_generation()
    completion = generation["completion"]
    completion.update(strength=2.5, safety_steering={
        "targets_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "strength": 2.5})
    if fault == "different_digest":
        completion["safety_steering"]["targets_sha256"] = "f" * 64
    elif fault == "different_strength":
        completion["strength"] = completion["safety_steering"]["strength"] = 0
    else:
        generation["completion"] = None
    monkeypatch.setattr(harmfulness_eval, "_get_responses", lambda *args, **kwargs: (
        "safety_steered", str(tmp_path / "result.yaml"), _pairs(), {}, generation))
    with pytest.raises(ValueError, match="receipt differs from the requested bundle or strength"):
        harmfulness_eval.main()
    current.assert_not_called()


@pytest.mark.parametrize("cadence", [None, 256])
@pytest.mark.parametrize("strength", [None, 0.0, 2.5])
def test_source_chain_generation_then_judge_only_reuses_same_artifact(monkeypatch, tmp_path, cadence, strength):
    cfg = _cfg(encdec=True)
    cfg.source = "capabilities_sft_conditioned_checkpoints"
    cfg.self_cond_segment_size = cadence
    cfg.generate_only = True
    model = _EncDecStub()
    current, _ = _patch_main_common(monkeypatch, cfg=cfg, model=model)
    metadata = {"encoder_config": {"enabled": True}, "xattn_config": {"enabled": True}}
    (tmp_path / "meta_000001.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(harmfulness_eval, "_resolve_checkpoint_dir_for_harmfulness",
        lambda **kwargs: str(tmp_path))
    provider = MagicMock(spec=SafetensorsContextFeatureProvider)
    provider.manifest = {"schema_version": 1, "rows": 2}
    setup = MagicMock(context_provider=provider, segment_size=256,
        safety_strength=0.0, safety_steering=None, assert_target_unchanged=MagicMock())
    condition_name = "self_conditioned" if strength is None else "safety_steered"
    if strength is not None:
        target_path = tmp_path / "targets.safetensors"
        target_path.write_bytes(b"frozen target fixture")
        cfg.self_conditioned.safety_steering = {"targets": str(target_path), "strength": strength}
        setup.safety_strength = strength
        setup.safety_steering = SimpleNamespace(targets=SimpleNamespace(
            sha256=hashlib.sha256(target_path.read_bytes()).hexdigest()))
    monkeypatch.setattr(harmfulness_eval, "prepare_harmfulness_self_conditioning", lambda *args, **kwargs: setup)
    monkeypatch.setattr(harmfulness_eval, "assert_exact_prompt_coverage", lambda *args, **kwargs: object())
    monkeypatch.setattr(harmfulness_eval, "conditioning_provenance", lambda *args, **kwargs: CONDITIONED_PROVENANCE)
    generator = MagicMock(side_effect=["safe", "helpful"])
    monkeypatch.setattr(harmfulness_eval, "generate_harmfulness_response_self_conditioned", generator)
    harmfulness_eval.main()
    responses = tmp_path / "report/harmfulness_results_responses" / f"{condition_name}.yaml"
    assert responses.exists()
    before = responses.read_bytes()
    cfg.generate_only = False
    cfg.judge_only = True
    cfg.batch_size = 8  # Generation-only defaults must not constrain model-free judging.
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness",
        lambda *args: pytest.fail("judge-only must not load weights"))
    monkeypatch.setattr(harmfulness_eval, "JudgeClient",
        lambda **kwargs: pytest.fail("already judged responses need no judge client"))
    harmfulness_eval.main()
    assert responses.read_bytes() == before
    assert generator.call_count == 2

    current.assert_called_once()
    assert current.call_args.kwargs["expected_generation"]["condition_name"] == condition_name

    # An explicit artifact owns its condition identity, independently of any
    # checkpoint directory or optional request to reassert the target bundle.
    cfg.source = None
    cfg.responses_file = str(responses)
    cfg.self_conditioned.safety_steering = None
    monkeypatch.setattr(harmfulness_eval, "_resolve_native_checkpoint_for_harmfulness",
        lambda *args: pytest.fail("explicit responses must not inspect a checkpoint"))
    read_responses = MagicMock(wraps=harmfulness_eval._load_responses)
    monkeypatch.setattr(harmfulness_eval, "_load_responses", read_responses)
    harmfulness_eval.main()
    read_responses.assert_called_once_with(str(responses), cfg)
    assert current.call_args.kwargs["expected_generation"]["condition_name"] == condition_name
    assert generator.call_count == 2


@pytest.mark.parametrize("fault", ["cadence", "family", "decoder_cadence"])
def test_explicit_response_assertions_fail_before_model_or_judge(monkeypatch, tmp_path, fault):
    cfg = _cfg(encdec=True)
    cfg.source = None
    cfg.responses_file = str(tmp_path / "responses.yaml")
    generation = _generation() if fault == "decoder_cadence" else _self_conditioned_generation()
    if fault == "cadence":
        cfg.self_cond_segment_size = 128
        match = "response receipt cadence"
    elif fault == "family":
        generation["model"]["model_type"] = "decoder_only"
        match = "requires an encoder-decoder model"
    else:
        match = "decoder-only harmfulness requires self_cond_segment_size=null"
    pairs: list[dict[str, object]] = [{**row, "response": "safe"} for row in _pairs()]
    artifact = build_harmfulness_responses_artifact(responses=pairs,
        conditioning=CONDITIONED_PROVENANCE, generation=generation)
    write_harmfulness_responses_artifact(artifact, cfg.responses_file)
    _patch_main_common(monkeypatch, cfg=cfg, model=_EncDecStub())
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **kwargs: pytest.fail("judge before response admission"))
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness", lambda *args: pytest.fail("explicit artifact loaded weights"))
    with pytest.raises(ValueError, match=match):
        harmfulness_eval.main()


@pytest.mark.parametrize("metadata", [None, b"not json", b"[]", b"\xff"])
def test_judge_only_rejects_missing_or_corrupt_metadata_before_judge(monkeypatch, tmp_path, metadata):
    cfg = _cfg(encdec=True)
    cfg.source = "capabilities_sft_conditioned_checkpoints"
    cfg.judge_only = True
    if metadata is not None:
        (tmp_path / "meta_000001.json").write_bytes(metadata)
    _patch_main_common(monkeypatch, cfg=cfg, model=_EncDecStub())
    monkeypatch.setattr(harmfulness_eval, "_resolve_native_checkpoint_for_harmfulness",
        lambda _cfg: (str(tmp_path), 1))
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **kwargs: pytest.fail("judge before metadata admission"))
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness", lambda *args: pytest.fail("judge-only loaded weights"))
    with pytest.raises(ValueError, match="judge-only checkpoint metadata"):
        harmfulness_eval.main()


@pytest.mark.parametrize("fault", ["cadence", "checkpoint_family", "decoder_cadence"])
def test_implicit_judge_only_validates_receipt_against_checkpoint_and_assertions(monkeypatch, tmp_path, fault):
    cfg = _cfg(encdec=True)
    cfg.source = "capabilities_sft_conditioned_checkpoints"
    cfg.judge_only = True
    cfg.batch_size = 8
    is_encdec = fault == "cadence"
    metadata = {"encoder_config": {"enabled": True}, "xattn_config": {"enabled": True}} if is_encdec else {}
    (tmp_path / "meta_000001.json").write_text(json.dumps(metadata))
    generation = _generation() if fault == "decoder_cadence" else _self_conditioned_generation()
    if fault == "cadence":
        cfg.self_cond_segment_size = 128
        match = "response receipt cadence"
    elif fault == "checkpoint_family":
        cfg.self_cond_segment_size = None
        match = "model family differs from the selected checkpoint"
    else:
        match = "decoder-only harmfulness requires self_cond_segment_size=null"
    filename = "self_conditioned.yaml" if is_encdec else "unconditional.yaml"
    response_path = tmp_path / "report/harmfulness_results_responses" / filename
    artifact = build_harmfulness_responses_artifact(
        responses=[{**row, "response": "safe"} for row in _pairs()],
        conditioning=CONDITIONED_PROVENANCE, generation=generation)
    write_harmfulness_responses_artifact(artifact, response_path)
    _patch_main_common(monkeypatch, cfg=cfg, model=_EncDecStub())
    monkeypatch.setattr(harmfulness_eval, "_resolve_checkpoint_dir_for_harmfulness", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", lambda **kwargs: pytest.fail("judge before response assertions"))
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness", lambda *args: pytest.fail("judge-only loaded weights"))
    with pytest.raises(ValueError, match=match):
        harmfulness_eval.main()
