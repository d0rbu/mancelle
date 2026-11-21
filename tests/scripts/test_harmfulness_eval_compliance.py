"""Standalone harmfulness/compliance orchestration tests for issue #1338."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from omegaconf import OmegaConf

import steering.eval.harmfulness_judge as harmfulness_judge
from scripts.eval import harmfulness_eval
from scripts.eval.harmfulness_eval import _judge_and_write
from steering.eval.harmfulness_artifact import (
    HARMFULNESS_RESPONSES_SCHEMA_VERSION,
    build_harmfulness_responses_artifact,
    validate_harmfulness_payload,
)
from steering.eval.harmfulness_generation import (
    SEED_SCHEDULE,
    SELF_CONDITIONED_SEED_SCHEDULE,
    build_generation_provenance,
)
from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult

MODEL = "openai/gpt-5.6-luna"
UNCONDITIONED_PROVENANCE: dict[str, Any] = {
    "conditioning_mode": "unconditioned",
    "eval_name": "harmfulness_eval",
}
CONDITIONED_PROVENANCE: dict[str, Any] = {
    "conditioning_mode": "conditions_within_vocab",
    "eval_name": "harmfulness_eval",
    "feature_vocab": {
        "n_sae_features": 8,
        "vocab_size": 3,
        "sha256": "0" * 64,
    },
    "context_cache": {
        "manifest_path": "/cache/manifest.json",
        "identity": {
            "model_id": "model",
            "sae_id": "sae",
            "layer": 1,
            "pooling": "max",
            "feature_selection_hash": "1" * 64,
        },
        "cached_rows": 4,
    },
    "coverage": {
        "expected_prompts": 4,
        "expected_unique_keys": 4,
        "extra_cached_rows": 0,
    },
}


def _generation(
    condition_name: str = "unconditional",
    *,
    completion: dict[str, Any] | None = None,
    feature_conditions: dict[int, int] | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    return build_generation_provenance(
        model={
            "format": "nanochat_checkpoint",
            "backend": "nanochat",
            "architecture": "GPT",
            "model_type": "decoder_only" if completion is None else "encoder_decoder",
            "step": 123,
            "checkpoint_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
        },
        condition_name=condition_name,
        device_type="cpu",
        inference_dtype="float32",
        feature_conditions=feature_conditions,
        target_conditioning_mode="none" if completion is None else "next",
        max_new_tokens=128,
        temperature=0.25,
        top_k=17,
        batch_size=batch_size,
        seed=42,
        prompt_count=4,
        seed_schedule=(SEED_SCHEDULE if completion is None else SELF_CONDITIONED_SEED_SCHEDULE),
        completion=completion,
    )


GENERATION_PROVENANCE = _generation()


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        source="safety",
        model_tag="d26",
        step=123,
        judge_model=MODEL,
    )


def _pairs() -> list[dict[str, object]]:
    return [
        {
            "prompt": "harmful first",
            "response": "I cannot help.",
            "pool": "harmful",
            "category": "Violence",
            "prompt_text": "rendered harmful first",
            "prompt_token_ids": [1, 2],
        },
        {
            "prompt": "benign first",
            "response": "A complete answer.",
            "pool": "benign",
            "category": "benign",
            "prompt_text": "rendered benign first",
            "prompt_token_ids": [3, 4],
        },
        {
            "prompt": "harmful second",
            "response": "Harmful details.",
            "pool": "harmful",
            "category": "Cybercrime",
            "prompt_text": "rendered harmful second",
            "prompt_token_ids": [5, 6],
        },
        {
            "prompt": "benign second",
            "response": "I refuse.",
            "pool": "benign",
            "category": "benign",
            "prompt_text": "rendered benign second",
            "prompt_token_ids": [7, 8],
        },
    ]


def _client() -> MagicMock:
    client = MagicMock()
    client.score_all.return_value = [
        JudgeResult([], 1, "Safe refusal."),
        JudgeResult([], 1, "Safe benign answer."),
        JudgeResult(["Cybercrime"], 5, "Harmful answer."),
        JudgeResult([], 1, "Safe benign refusal."),
    ]
    client.score_compliance_all.return_value = [
        ComplianceJudgeResult(1, False, "Low helpfulness without refusal."),
        ComplianceJudgeResult(5, True, "Helpful partial refusal."),
    ]
    return client


def _response_cfg(**overrides: object):
    values: dict[str, object] = {
        "source": "safety",
        "model_tag": "d26",
        "output_file": None,
        "responses_file": None,
        "segment_size": None,
        "feature_selection": None,
        "feature_mask": None,
        "judge_only": False,
        "num_harmful": None,
        "num_benign": 2,
        "seed": 42,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _source_prompts() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return (
        [
            {"prompt": "harmful one", "category": "Violence"},
            {"prompt": "harmful two", "category": "Cybercrime"},
        ],
        [
            {"prompt": "benign one", "category": "benign"},
            {"prompt": "benign two", "category": "benign"},
        ],
    )


def _current_cached_pairs() -> list[dict[str, object]]:
    harmful, benign = _source_prompts()
    pairs: list[dict[str, object]] = []
    for pool, prompts in (("harmful", harmful), ("benign", benign)):
        for prompt in prompts:
            pairs.append(
                {
                    **prompt,
                    "pool": pool,
                    "response": f"response to {prompt['prompt']}",
                }
            )
    return pairs


def _responses_artifact(
    pairs: list[dict[str, object]],
    *,
    conditioning: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, object]:
    return build_harmfulness_responses_artifact(
        responses=pairs,
        conditioning=(UNCONDITIONED_PROVENANCE if conditioning is None else conditioning),
        generation=(GENERATION_PROVENANCE if generation is None else generation),
    )


def _stub_prompt_sources(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    harmful, benign = _source_prompts()
    load_harmful = MagicMock(return_value=harmful)
    monkeypatch.setattr(harmfulness_judge, "load_harmful_prompts", load_harmful)

    def load_benign(*, n: int, seed: int) -> list[dict[str, str]]:
        assert n == 2
        assert seed == 42
        return benign

    monkeypatch.setattr(harmfulness_judge, "load_benign_prompts", load_benign)
    return load_harmful


def test_response_cache_accepts_only_exact_current_ordered_prompt_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_harmful = _stub_prompt_sources(monkeypatch)
    pairs = _current_cached_pairs()
    pairs[2]["prompt_text"] = "rendered benign prompt"
    pairs[2]["prompt_token_ids"] = [1, 2, 3]
    path = tmp_path / "responses.yaml"
    path.write_text(yaml.safe_dump(_responses_artifact(pairs)), encoding="utf-8")

    loaded, conditioning, generation = harmfulness_eval._load_responses(
        str(path),
        _response_cfg(),
    )

    assert loaded == pairs
    assert conditioning == UNCONDITIONED_PROVENANCE
    assert generation == GENERATION_PROVENANCE
    assert loaded[2]["prompt_text"] == "rendered benign prompt"
    load_harmful.assert_called_once_with()


def test_response_cache_rejects_different_generation_conditioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_harmful = _stub_prompt_sources(monkeypatch)
    path = tmp_path / "responses.yaml"
    path.write_text(
        yaml.safe_dump(_responses_artifact(_current_cached_pairs())),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the current generation run"):
        harmfulness_eval._load_responses(
            str(path),
            _response_cfg(),
            expected_conditioning=CONDITIONED_PROVENANCE,
        )

    load_harmful.assert_not_called()


def test_explicit_response_file_returns_persisted_conditioned_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_prompt_sources(monkeypatch)
    pairs = _current_cached_pairs()
    response_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "result.yaml"
    response_path.write_text(
        yaml.safe_dump(
            _responses_artifact(
                pairs,
                conditioning=CONDITIONED_PROVENANCE,
                generation=_generation("target"),
            )
        ),
        encoding="utf-8",
    )
    cfg = _response_cfg(
        source=None,
        responses_file=str(response_path),
        output_file=str(output_path),
        judge_only=True,
    )

    name, resolved_output, loaded, conditioning, generation = harmfulness_eval._get_responses(
        cfg,
        "unconditional",
        None,
    )

    assert (name, resolved_output, loaded, conditioning, generation) == (
        "target",
        str(output_path),
        pairs,
        CONDITIONED_PROVENANCE,
        _generation("target"),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda pairs: pairs.pop(), "prompt count"),
        (
            lambda pairs: pairs.__setitem__(2, "not a mapping"),
            "pair 2 must be a mapping",
        ),
        (
            lambda pairs: pairs[2].__setitem__("response", None),
            "pair 2 response must be a string",
        ),
        (
            lambda pairs: pairs[2].__setitem__("prompt", "make a bomb"),
            "pair 2 prompt identity does not match",
        ),
        (
            lambda pairs: pairs[2].__setitem__("pool", "harmful"),
            "pair 2 prompt identity does not match",
        ),
        (
            lambda pairs: pairs[2].__setitem__("category", "Violence"),
            "pair 2 prompt identity does not match",
        ),
        (
            lambda pairs: pairs.__setitem__(slice(None), list(reversed(pairs))),
            "pair 0 prompt identity does not match",
        ),
    ],
)
def test_response_cache_rejects_every_prompt_source_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[list[dict[str, object]]], object],
    message: str,
) -> None:
    _stub_prompt_sources(monkeypatch)
    pairs = _current_cached_pairs()
    mutation(pairs)
    path = tmp_path / "responses.yaml"
    path.write_text(yaml.safe_dump(_responses_artifact(pairs)), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        harmfulness_eval._load_responses(str(path), _response_cfg())


@pytest.mark.parametrize("invalid_document", [None, {}, [], "pairs", 3, True])
def test_response_cache_rejects_invalid_artifact_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_document: object,
) -> None:
    load_harmful = _stub_prompt_sources(monkeypatch)
    path = tmp_path / "responses.yaml"
    path.write_text(yaml.safe_dump(invalid_document), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid generation provenance"):
        harmfulness_eval._load_responses(str(path), _response_cfg())

    load_harmful.assert_not_called()


def test_response_cache_wraps_malformed_yaml_as_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_harmful = _stub_prompt_sources(monkeypatch)
    path = tmp_path / "responses.yaml"
    path.write_text("responses: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid generation provenance"):
        harmfulness_eval._load_responses(str(path), _response_cfg())

    load_harmful.assert_not_called()


def test_generation_and_cache_validation_share_the_same_prompt_pool_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harmful, benign = _source_prompts()
    load_harmful = MagicMock(return_value=harmful)
    load_benign = MagicMock(return_value=benign)
    monkeypatch.setattr(harmfulness_judge, "load_harmful_prompts", load_harmful)
    monkeypatch.setattr(harmfulness_judge, "load_benign_prompts", load_benign)

    selected = harmfulness_eval._load_expected_prompt_pairs(_response_cfg())

    assert selected == [
        {**harmful[0], "pool": "harmful"},
        {**harmful[1], "pool": "harmful"},
        {**benign[0], "pool": "benign"},
        {**benign[1], "pool": "benign"},
    ]
    load_harmful.assert_called_once_with()
    load_benign.assert_called_once_with(n=2, seed=42)


def test_prompt_pool_applies_seeded_harmful_subsample_before_benign_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harmful = [{"prompt": f"harmful {index}", "category": "Violence"} for index in range(5)]
    benign = [{"prompt": "benign", "category": "benign"}]
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        MagicMock(return_value=harmful.copy()),
    )
    load_benign = MagicMock(return_value=benign)
    monkeypatch.setattr(harmfulness_judge, "load_benign_prompts", load_benign)
    cfg = _response_cfg(num_harmful=2, num_benign=1, seed=17)
    expected_harmful = harmful.copy()
    random.Random(17).shuffle(expected_harmful)

    selected = harmfulness_eval._load_expected_prompt_pairs(cfg)

    assert selected == [
        {**expected_harmful[0], "pool": "harmful"},
        {**expected_harmful[1], "pool": "harmful"},
        {**benign[0], "pool": "benign"},
    ]
    load_benign.assert_called_once_with(n=1, seed=17)


def test_generation_persists_the_exact_shared_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {"prompt": "harmful", "category": "Violence", "pool": "harmful"},
        {"prompt": "benign", "category": "benign", "pool": "benign"},
    ]
    load_pool = MagicMock(return_value=selected)
    monkeypatch.setattr(harmfulness_eval, "_load_expected_prompt_pairs", load_pool)
    engine = MagicMock()
    engine.generate_batch.return_value = ["safe response", "helpful response"]
    path = tmp_path / "responses.yaml"
    cfg = OmegaConf.create(
        {
            "batch_size": 8,
            "max_new_tokens": 128,
            "temperature": 0.25,
            "top_k": 17,
            "seed": 91,
        }
    )

    def generate(batch: list[tuple[int, dict[str, Any]]]) -> list[str]:
        return engine.generate_batch(
            [str(pair["prompt"]) for _, pair in batch],
            max_new_tokens=128,
            temperature=0.25,
            top_k=17,
            seed=91,
            render_as_chat=True,
        )

    generated = harmfulness_eval._generate_responses(
        cfg,
        str(path),
        condition_name="unconditional",
        response_generator=generate,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )

    load_pool.assert_called_once_with(cfg)
    engine.generate_batch.assert_called_once_with(
        ["harmful", "benign"],
        max_new_tokens=128,
        temperature=0.25,
        top_k=17,
        seed=91,
        render_as_chat=True,
    )
    assert generated == [
        {
            "prompt": "harmful",
            "category": "Violence",
            "pool": "harmful",
            "response": "safe response",
        },
        {
            "prompt": "benign",
            "category": "benign",
            "pool": "benign",
            "response": "helpful response",
        },
    ]
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "harmfulness_responses_schema_version": (HARMFULNESS_RESPONSES_SCHEMA_VERSION),
        "conditioning": UNCONDITIONED_PROVENANCE,
        "generation": GENERATION_PROVENANCE,
        "responses": generated,
    }


def test_generate_mode_regenerates_a_stale_response_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "result.yaml"
    response_path.write_text("legacy: unsafe\n", encoding="utf-8")
    stale = ValueError("legacy benign pool")
    load = MagicMock(side_effect=stale)
    regenerated = _current_cached_pairs()
    generate = MagicMock(return_value=regenerated)
    warning = MagicMock()
    monkeypatch.setattr(
        harmfulness_eval,
        "resolve_harmfulness_paths",
        lambda **_kwargs: (str(response_path), str(output_path)),
    )
    monkeypatch.setattr(harmfulness_eval, "_load_responses", load)
    monkeypatch.setattr(harmfulness_eval, "_generate_responses", generate)
    monkeypatch.setattr(harmfulness_eval.logger, "warning", warning)
    cfg = _response_cfg()
    response_generator = MagicMock()
    name, returned_output, pairs, conditioning, generation = harmfulness_eval._get_responses(
        cfg,
        "unconditional",
        response_generator,
        generation_conditioning=UNCONDITIONED_PROVENANCE,
        generation_provenance=GENERATION_PROVENANCE,
    )

    assert (name, returned_output, pairs, conditioning, generation) == (
        "unconditional",
        str(output_path),
        regenerated,
        UNCONDITIONED_PROVENANCE,
        GENERATION_PROVENANCE,
    )
    load.assert_called_once_with(
        str(response_path),
        cfg,
        expected_conditioning=UNCONDITIONED_PROVENANCE,
        expected_generation=GENERATION_PROVENANCE,
    )
    generate.assert_called_once_with(
        cfg,
        str(response_path),
        condition_name="unconditional",
        response_generator=response_generator,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )
    assert "regenerating" in warning.call_args.args[0]
    assert "legacy benign pool" in warning.call_args.args[0]


def test_generate_mode_reuses_only_a_validated_current_response_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "result.yaml"
    response_path.write_text("[]\n", encoding="utf-8")
    cached = _current_cached_pairs()
    load = MagicMock(return_value=(cached, UNCONDITIONED_PROVENANCE, GENERATION_PROVENANCE))
    generate = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(
        harmfulness_eval,
        "resolve_harmfulness_paths",
        lambda **_kwargs: (str(response_path), str(output_path)),
    )
    monkeypatch.setattr(harmfulness_eval, "_load_responses", load)
    monkeypatch.setattr(harmfulness_eval, "_generate_responses", generate)
    monkeypatch.setattr(harmfulness_eval.logger, "info", info)
    cfg = _response_cfg()

    name, returned_output, pairs, conditioning, generation = harmfulness_eval._get_responses(
        cfg,
        "unconditional",
        MagicMock(),
        generation_conditioning=UNCONDITIONED_PROVENANCE,
        generation_provenance=GENERATION_PROVENANCE,
    )

    assert (name, returned_output, pairs, conditioning, generation) == (
        "unconditional",
        str(output_path),
        cached,
        UNCONDITIONED_PROVENANCE,
        GENERATION_PROVENANCE,
    )
    load.assert_called_once_with(
        str(response_path),
        cfg,
        expected_conditioning=UNCONDITIONED_PROVENANCE,
        expected_generation=GENERATION_PROVENANCE,
    )
    generate.assert_not_called()
    assert any("Found current cached responses" in logged.args[0] for logged in info.call_args_list)


def test_judge_only_missing_auto_derived_cache_exits_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_path = tmp_path / "missing-responses.yaml"
    output_path = tmp_path / "result.yaml"
    load = MagicMock()
    error = MagicMock()
    monkeypatch.setattr(
        harmfulness_eval,
        "resolve_harmfulness_paths",
        lambda **_kwargs: (str(response_path), str(output_path)),
    )
    monkeypatch.setattr(harmfulness_eval, "_load_responses", load)
    monkeypatch.setattr(harmfulness_eval.logger, "error", error)

    with pytest.raises(SystemExit) as exc_info:
        harmfulness_eval._get_responses(
            _response_cfg(judge_only=True),
            "unconditional",
            None,
        )

    assert exc_info.value.code == 1
    load.assert_not_called()
    assert str(response_path) in error.call_args.args[0]


@pytest.mark.parametrize(
    "cfg",
    [
        _response_cfg(judge_only=True),
        _response_cfg(
            source=None,
            responses_file="explicit-responses.yaml",
            output_file="result.yaml",
        ),
    ],
)
def test_no_generation_mode_never_bypasses_stale_prompt_pool_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg,
) -> None:
    response_path = tmp_path / "responses.yaml"
    output_path = tmp_path / "result.yaml"
    response_path.write_text("legacy: unsafe\n", encoding="utf-8")
    cfg.responses_file = str(response_path) if cfg.responses_file else None
    load = MagicMock(side_effect=ValueError("legacy benign pool"))
    generate = MagicMock()
    monkeypatch.setattr(
        harmfulness_eval,
        "resolve_harmfulness_paths",
        lambda **_kwargs: (str(response_path), str(output_path)),
    )
    monkeypatch.setattr(harmfulness_eval, "_load_responses", load)
    monkeypatch.setattr(harmfulness_eval, "_generate_responses", generate)

    with pytest.raises(ValueError, match="legacy benign pool"):
        harmfulness_eval._get_responses(
            cfg,
            "unconditional",
            None,
        )

    load.assert_called_once_with(str(response_path), cfg)
    generate.assert_not_called()


def test_standalone_judges_compliance_only_for_ordered_benign_subset(
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    client = _client()
    path = tmp_path / "results.yaml"

    returned = _judge_and_write(
        _cfg(),
        "current-cli-label",
        str(path),
        pairs,
        client,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )

    client.score_all.assert_called_once_with(pairs)
    client.score_compliance_all.assert_called_once_with([pairs[1], pairs[3]])
    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert returned == persisted
    validate_harmfulness_payload(
        persisted,
        model=MODEL,
        pairs=pairs,
    )
    assert persisted["harmful_prompts"]["safe_response_rate"] == 0.5
    assert persisted["overall"]["safe_response_rate"] == 0.75
    assert persisted["benign_prompts"]["compliance_rate"] == 0.5
    assert persisted["benign_prompts"]["compliance_num_scored"] == 2
    assert persisted["benign_prompts"]["compliance_num_failed"] == 0
    assert [
        record["is_compliant"] for record in persisted["benign_prompts"]["compliance_judgments"]
    ] == [True, False]
    assert [
        record["instruction"] for record in persisted["benign_prompts"]["compliance_judgments"]
    ] == ["benign first", "benign second"]
    assert persisted["config"] == {
        "source": "safety",
        "model_tag": "d26",
        "step": 123,
        "judge_model": MODEL,
        "feature_configs": "unconditional",
        "feature_conditions": None,
        "num_harmful": 2,
        "num_benign": 2,
        "conditioning": UNCONDITIONED_PROVENANCE,
        "generation": GENERATION_PROVENANCE,
    }


def test_standalone_excludes_partial_compliance_failure_and_persists_it(
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    client = _client()
    client.score_compliance_all.return_value = [
        ComplianceJudgeResult(4, False, "Substantially helpful."),
        None,
    ]

    payload = _judge_and_write(
        _cfg(),
        "unconditional",
        str(tmp_path / "result.yaml"),
        pairs,
        client,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )

    benign = payload["benign_prompts"]
    assert benign["compliance_rate"] == 1.0
    assert benign["compliance_num_scored"] == 1
    assert benign["compliance_num_failed"] == 1
    failed = benign["compliance_judgments"][1]
    assert failed["status"] == "failed"
    assert failed["error"] == "judge_retries_exhausted"
    assert failed["helpfulness_score"] is None


def test_standalone_nonempty_all_failed_compliance_never_writes_result(
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    client = _client()
    client.score_compliance_all.return_value = [None, None]
    path = tmp_path / "must-not-exist.yaml"

    with pytest.raises(ValueError, match="no successful judge results"):
        _judge_and_write(
            _cfg(),
            "unconditional",
            str(path),
            pairs,
            client,
            conditioning=UNCONDITIONED_PROVENANCE,
            generation=GENERATION_PROVENANCE,
        )
    assert not path.exists()


def test_standalone_empty_benign_pool_has_no_compliance_api_calls(
    tmp_path: Path,
) -> None:
    pairs = [_pairs()[0], _pairs()[2]]
    client = MagicMock()
    client.score_all.return_value = [
        JudgeResult([], 1, "Safe."),
        JudgeResult(["Cybercrime"], 5, "Harmful."),
    ]
    client.score_compliance_all.return_value = []

    payload = _judge_and_write(
        _cfg(),
        "unconditional",
        str(tmp_path / "result.yaml"),
        pairs,
        client,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )

    client.score_compliance_all.assert_called_once_with([])
    assert payload["benign_prompts"]["compliance_rate"] == 0.0
    assert payload["benign_prompts"]["compliance_judgments"] == []


def test_standalone_harmfulness_failures_remain_separate_from_compliance(
    tmp_path: Path,
) -> None:
    pairs = _pairs()
    client = _client()
    client.score_all.return_value[1] = None

    payload = _judge_and_write(
        _cfg(),
        "unconditional",
        str(tmp_path / "result.yaml"),
        pairs,
        client,
        conditioning=UNCONDITIONED_PROVENANCE,
        generation=GENERATION_PROVENANCE,
    )

    assert payload["benign_prompts"]["num_scored"] == 1
    assert payload["benign_prompts"]["num_failed"] == 1
    assert payload["benign_prompts"]["compliance_num_scored"] == 2
    assert payload["benign_prompts"]["compliance_num_failed"] == 0


@pytest.mark.parametrize("judge_kind", ["harmfulness", "compliance"])
def test_standalone_judge_cardinality_mismatch_fails_loudly(
    tmp_path: Path,
    judge_kind: str,
) -> None:
    pairs = _pairs()
    client = _client()
    if judge_kind == "harmfulness":
        client.score_all.return_value.pop()
        expected = "Harmfulness judge returned 3 results for 4 prompts"
    else:
        client.score_compliance_all.return_value.pop()
        expected = "Benign compliance judge returned 1 results for 2 prompts"

    with pytest.raises(ValueError, match=expected):
        _judge_and_write(
            _cfg(),
            "unconditional",
            str(tmp_path / "result.yaml"),
            pairs,
            client,
            conditioning=UNCONDITIONED_PROVENANCE,
            generation=GENERATION_PROVENANCE,
        )
    assert not (tmp_path / "result.yaml").exists()


@pytest.mark.parametrize("result_state", ["current", "stale", "fresh"])
@pytest.mark.parametrize("judge_fails", [False, True])
def test_main_resumes_only_current_artifacts_and_rejudges_stale_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_state: str,
    judge_fails: bool,
) -> None:
    (tmp_path / "meta_000001.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(harmfulness_eval, "_resolve_native_checkpoint_for_harmfulness",
        lambda _cfg: (str(tmp_path), 1))
    path = tmp_path / "result.yaml"
    if result_state != "fresh":
        path.write_text(f"{result_state}: true\n", encoding="utf-8")
    pairs = _pairs()
    cfg = OmegaConf.create({
        "source": "base", "hf_checkpoint_dir": None, "judge_only": True,
        "responses_file": None, "feature_configs": None, "self_cond_segment_size": None,
        "batch_size": 1, "max_new_tokens": 128, "temperature": 0.25, "top_k": 17,
        "judge_model": MODEL, "cache_path": "", "requests_per_minute": 60, "seed": 42,
    })
    current = MagicMock(return_value=(
        ({"current": True}, "current artifact") if result_state == "current"
        else (None, "legacy schema" if result_state == "stale" else "result file does not exist")
    ))
    generation = _generation("unconditional", completion=None, feature_conditions=None, batch_size=1)
    judged_payload = {
        "overall": {"mean_score": 1.0, "harmfulness_rate": 0.0, "safe_response_rate": 1.0},
        "harmful_prompts": {"asr": 0.0},
    }
    judge = MagicMock(return_value=judged_payload,
        side_effect=RuntimeError("judge failed") if judge_fails else None)
    client = MagicMock()
    warning = MagicMock()
    get_responses = MagicMock(return_value=(
        "unconditional", str(path), pairs, CONDITIONED_PROVENANCE, generation,
    ))
    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda _argv: cfg)
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", client)
    monkeypatch.setattr(harmfulness_eval, "_get_responses", get_responses)
    monkeypatch.setattr(harmfulness_eval, "current_harmfulness_payload", current)
    monkeypatch.setattr(harmfulness_eval, "_judge_and_write", judge)
    monkeypatch.setattr(harmfulness_eval.logger, "warning", warning)

    if judge_fails and result_state != "current":
        with pytest.raises(RuntimeError, match="judge failed"):
            harmfulness_eval.main()
    else:
        harmfulness_eval.main()

    get_responses.assert_called_once_with(cfg, "unconditional", None,
        generation_conditioning=None, generation_provenance=None)
    current.assert_called_once_with(str(path), model=MODEL, pairs=pairs,
        expected_conditioning=CONDITIONED_PROVENANCE, expected_generation=generation)
    if result_state == "current":
        client.assert_not_called()
        judge.assert_not_called()
    else:
        client.assert_called_once_with(model=MODEL, cache_path=None, requests_per_minute=60)
        judge.assert_called_once_with(cfg, "unconditional", str(path), pairs, client.return_value,
            conditioning=CONDITIONED_PROVENANCE, generation=generation)
    assert warning.call_count == int(result_state == "stale")
    if result_state == "stale":
        assert "legacy schema" in warning.call_args.args[0]
