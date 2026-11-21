from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.eval import harmfulness_eval
from steering.eval.harmfulness_artifact import parse_harmfulness_responses_artifact
from steering.eval.harmfulness_generation import (
    HF_BATCHED_SEED_SCHEDULE,
    MAX_TORCH_SEED,
    SELF_CONDITIONED_SEED_SCHEDULE,
    absolute_prompt_batch_seed,
    build_generation_provenance,
    capture_native_artifact_identity,
    native_model_identity,
    validate_absolute_prompt_seed,
    validate_generation_provenance,
)
from steering.immutable_io import ImmutableArtifactError


def _model_identity() -> dict[str, object]:
    return {
        "format": "nanochat_checkpoint",
        "backend": "nanochat",
        "architecture": "GPT",
        "model_type": "decoder_only",
        "step": 7,
        "checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
    }


def _generation(*, prompt_count: int = 17, batch_size: int = 8) -> dict[str, Any]:
    return build_generation_provenance(
        model=_model_identity(),
        condition_name="unconditional",
        device_type="cpu",
        inference_dtype="float32",
        feature_conditions=None,
        target_conditioning_mode="none",
        max_new_tokens=12,
        temperature=0.7,
        top_k=50,
        batch_size=batch_size,
        seed=42,
        prompt_count=prompt_count,
    )


def _self_conditioned_generation() -> dict[str, Any]:
    model = _model_identity()
    model["model_type"] = "encoder_decoder"
    return build_generation_provenance(
        model=model,
        condition_name="self_conditioned",
        device_type="cuda",
        inference_dtype="bfloat16",
        feature_conditions=None,
        target_conditioning_mode="next",
        max_new_tokens=12,
        temperature=0.7,
        top_k=50,
        batch_size=1,
        seed=42,
        prompt_count=17,
        seed_schedule=SELF_CONDITIONED_SEED_SCHEDULE,
        completion={
            "self_cond_segment_size": 256,
            "strength": 0.0,
            "action_ranking_policy": "pooled_fp_logits_v1",
        },
    )


def test_repository_hf_requires_batched_algorithm_identity() -> None:
    model = _model_identity()
    model.update(
        format="repository_hf_sft",
        backend="pythia",
        architecture="GPTNeoXForCausalLM",
        model_type="gpt_neox",
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "condition_name": "unconditional",
        "device_type": "cuda",
        "inference_dtype": "bfloat16",
        "feature_conditions": None,
        "target_conditioning_mode": "none",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_k": 50,
        "batch_size": 8,
        "seed": 42,
        "prompt_count": 17,
    }
    with pytest.raises(ValueError, match="batched outer-batch seed schedule"):
        build_generation_provenance(**kwargs)

    current = build_generation_provenance(
        **kwargs,
        seed_schedule=HF_BATCHED_SEED_SCHEDULE,
    )
    assert current["seed_schedule"] == HF_BATCHED_SEED_SCHEDULE

    stale_native = copy.deepcopy(_generation())
    stale_native["seed_schedule"] = HF_BATCHED_SEED_SCHEDULE
    with pytest.raises(ValueError, match="native harmfulness generation"):
        validate_generation_provenance(stale_native)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("contract_version",), 2),
        (("contract_version",), True),
        (("model", "format"), "external_hf"),
        (("model", "backend"), "other"),
        (("model", "architecture"), ""),
        (("model", "model_type"), ""),
        (("model", "step"), True),
        (("model", "checkpoint_sha256"), "x" * 64),
        (("model", "tokenizer_sha256"), "A" * 64),
        (("condition_name",), ""),
        (("device_type",), "tpu"),
        (("inference_dtype",), "float16"),
        (("feature_conditions",), {2: 1, 1: 1}),
        (("target_conditioning_mode",), "auto"),
        (("max_new_tokens",), -1),
        (("temperature",), 1),
        (("temperature",), float("nan")),
        (("top_k",), 0),
        (("batch_size",), 0),
        (("seed",), -1),
        (("seed_schedule",), "per_batch"),
        (("render_as_chat",), 1),
    ],
)
def test_every_generation_field_is_strict(path: tuple[str, ...], value: object) -> None:
    mutated = copy.deepcopy(_generation())
    target: Any = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_generation_provenance(mutated, prompt_count=17)


def test_generation_schema_rejects_missing_and_extra_fields() -> None:
    missing = copy.deepcopy(_generation())
    del missing["top_k"]
    extra = copy.deepcopy(_generation())
    extra["top_p"] = 1.0
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_generation_provenance(missing)
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_generation_provenance(extra)


def test_self_conditioned_generation_rejects_stale_or_wrong_ranking_policy() -> None:
    current = _self_conditioned_generation()
    completion: Any = current["completion"]
    assert isinstance(completion, dict)
    assert completion["action_ranking_policy"] == "pooled_fp_logits_v1"

    missing = copy.deepcopy(current)
    missing_completion: Any = missing["completion"]
    assert isinstance(missing_completion, dict)
    del missing_completion["action_ranking_policy"]
    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_generation_provenance(missing)

    wrong = copy.deepcopy(current)
    wrong_completion: Any = wrong["completion"]
    assert isinstance(wrong_completion, dict)
    wrong_completion["action_ranking_policy"] = "sigmoid_rank_v0"
    with pytest.raises(ValueError, match="action_ranking_policy"):
        validate_generation_provenance(wrong)


def test_schema_v1_responses_require_regeneration() -> None:
    with pytest.raises(ValueError, match="schema-v1.*regenerate responses"):
        parse_harmfulness_responses_artifact(
            {
                "harmfulness_responses_schema_version": 1,
                "conditioning": {
                    "conditioning_mode": "unconditioned",
                    "eval_name": "harmfulness_eval",
                },
                "responses": [],
            }
        )


def test_seed_range_and_absolute_batch_arithmetic() -> None:
    assert [absolute_prompt_batch_seed(42, index) for index in (0, 8, 16)] == [42, 50, 58]
    assert validate_absolute_prompt_seed(MAX_TORCH_SEED, 0) == MAX_TORCH_SEED
    assert validate_absolute_prompt_seed(MAX_TORCH_SEED, 1) == MAX_TORCH_SEED
    with pytest.raises(ValueError, match="overflows"):
        validate_absolute_prompt_seed(MAX_TORCH_SEED, 2)
    for invalid in (True, -1, 1.0, "1"):
        with pytest.raises(ValueError):
            validate_absolute_prompt_seed(
                invalid,  # ty: ignore[invalid-argument-type]
                1,
            )


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_outer_batches_seed_each_batch_from_its_absolute_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    prompts = [
        {"prompt": f"prompt-{index}", "pool": "harmful", "category": "Violence"}
        for index in range(17)
    ]
    monkeypatch.setattr(
        harmfulness_eval,
        "_load_expected_prompt_pairs",
        lambda _cfg: copy.deepcopy(prompts),
    )
    outer_seeds: list[int] = []

    def generate(batch: list[str], *, seed: int, **_kwargs: object) -> list[str]:
        outer_seeds.append(seed)
        return [f"response-{prompt}" for prompt in batch]

    engine = SimpleNamespace(generate_batch=generate)

    def response_generator(batch: list[tuple[int, dict[str, Any]]]) -> list[str]:
        return engine.generate_batch(
            [str(pair["prompt"]) for _, pair in batch],
            seed=absolute_prompt_batch_seed(42, batch[0][0]),
        )

    cfg = SimpleNamespace(
        # Execution must use the provenance batch size, not a shared config
        # field that can differ for repository-HF generation.
        batch_size=999,
        max_new_tokens=12,
        temperature=0.7,
        top_k=50,
        seed=42,
    )
    harmfulness_eval._generate_responses(
        cfg,
        str(tmp_path / "responses.yaml"),
        condition_name="unconditional",
        response_generator=response_generator,
        conditioning={
            "conditioning_mode": "unconditioned",
            "eval_name": "harmfulness_eval",
        },
        generation=_generation(prompt_count=17, batch_size=batch_size),
    )
    assert outer_seeds == [42 + index for index in range(0, 17, batch_size)]


def test_zero_prompts_does_not_call_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harmfulness_eval, "_load_expected_prompt_pairs", lambda _cfg: [])
    engine = MagicMock()
    cfg = SimpleNamespace(
        batch_size=8,
        max_new_tokens=0,
        temperature=0.0,
        top_k=None,
        seed=0,
    )
    assert (
        harmfulness_eval._generate_responses(
            cfg,
            str(tmp_path / "responses.yaml"),
            condition_name="unconditional",
            response_generator=engine,
            conditioning={
                "conditioning_mode": "unconditioned",
                "eval_name": "harmfulness_eval",
            },
            generation=build_generation_provenance(
                model=_model_identity(),
                condition_name="unconditional",
                device_type="cpu",
                inference_dtype="float32",
                feature_conditions=None,
                target_conditioning_mode="none",
                max_new_tokens=0,
                temperature=0.0,
                top_k=None,
                batch_size=8,
                seed=0,
                prompt_count=0,
            ),
        )
        == []
    )
    engine.assert_not_called()


def _native_tree(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_000007.pt").write_bytes(b"weights")
    (checkpoint / "meta_000007.json").write_text('{"step": 7}\n', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")
    return checkpoint, tokenizer


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("checkpoint/model_000007.pt", b"different weights"),
        ("checkpoint/meta_000007.json", b'{"step": 8}\n'),
        ("tokenizer/tokenizer.pkl", b"different tokenizer"),
    ],
)
def test_native_identity_changes_for_each_load_bearing_input(
    tmp_path: Path,
    relative: str,
    replacement: bytes,
) -> None:
    checkpoint, tokenizer = _native_tree(tmp_path)
    before = capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)
    (tmp_path / relative).write_bytes(replacement)
    after = capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)
    assert (
        before.checkpoint_inventory.sha256,
        before.tokenizer_inventory.sha256,
    ) != (
        after.checkpoint_inventory.sha256,
        after.tokenizer_inventory.sha256,
    )


def test_native_identity_includes_effective_loaded_metadata(tmp_path: Path) -> None:
    checkpoint, tokenizer = _native_tree(tmp_path)
    artifacts = capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)

    first = native_model_identity(
        artifacts,
        architecture="EncoderDecoderGPT",
        model_type="encoder_decoder",
        effective_metadata={"checkpoint_dir": str(checkpoint), "model_config": {"n_layer": 2}},
    )
    second = native_model_identity(
        artifacts,
        architecture="EncoderDecoderGPT",
        model_type="encoder_decoder",
        effective_metadata={"checkpoint_dir": str(checkpoint), "model_config": {"n_layer": 3}},
    )

    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]


def test_native_identity_accepts_symlinked_artifact_roots(tmp_path: Path) -> None:
    checkpoint, tokenizer = _native_tree(tmp_path)
    checkpoint_link = tmp_path / "checkpoint-link"
    tokenizer_link = tmp_path / "tokenizer-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
    tokenizer_link.symlink_to(tokenizer, target_is_directory=True)

    artifacts = capture_native_artifact_identity(
        checkpoint_link,
        step=7,
        tokenizer_dir=tokenizer_link,
    )

    assert artifacts.checkpoint_dir == checkpoint.resolve()
    assert artifacts.tokenizer_inventory.root == tokenizer.resolve()


def test_native_identity_rejects_symlink_and_nonregular_file(tmp_path: Path) -> None:
    checkpoint, tokenizer = _native_tree(tmp_path)
    weights = checkpoint / "model_000007.pt"
    target = checkpoint / "real.pt"
    weights.rename(target)
    weights.symlink_to(target.name)
    with pytest.raises(ValueError, match="regular non-symlink"):
        capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)

    weights.unlink()
    weights.mkdir()
    with pytest.raises(ValueError, match="regular non-symlink"):
        capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)


def test_native_identity_propagates_mid_hash_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, tokenizer = _native_tree(tmp_path)
    monkeypatch.setattr(
        "steering.eval.harmfulness_generation.stream_hash_file_descriptor_bound",
        MagicMock(side_effect=ImmutableArtifactError("changed during read")),
    )
    with pytest.raises(ImmutableArtifactError, match="changed during read"):
        capture_native_artifact_identity(checkpoint, step=7, tokenizer_dir=tokenizer)
