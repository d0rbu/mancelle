"""Validation tests for harmfulness report conditioning provenance."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.harmfulness_artifact import (
    build_harmfulness_payload,
    validate_harmfulness_payload,
)
from steering.eval.harmfulness_generation import build_generation_provenance
from steering.eval.harmfulness_judge import (
    ComplianceJudgeResult,
    JudgeResult,
    compute_compliance_metrics,
    compute_metrics,
)
from tests.conftest import TEST_CACHE_IDENTITY

MODEL = "test-judge"
VOCAB = FeatureVocab(8, (1, 3, 5))


def _pairs() -> list[dict[str, str]]:
    return [
        {
            "prompt": "harmful",
            "response": "safe",
            "pool": "harmful",
            "category": "Violence",
        },
        {
            "prompt": "benign",
            "response": "helpful",
            "pool": "benign",
            "category": "benign",
        },
    ]


def _conditioned_block() -> dict[str, object]:
    return {
        "conditioning_mode": "conditions_within_vocab",
        "eval_name": "harmfulness_eval",
        "feature_vocab": VOCAB.metadata_dict(),
        "context_cache": {
            "manifest_path": "/cache/c32/sel_0000000000000000/manifest.json",
            "identity": TEST_CACHE_IDENTITY.as_dict(),
            "cached_rows": 2,
        },
        "coverage": {
            "expected_prompts": 2,
            "expected_unique_keys": 2,
            "extra_cached_rows": 0,
        },
    }


def _payload(conditioning: dict[str, object] | None) -> dict:
    pairs = _pairs()
    config = {"judge_model": MODEL}
    if conditioning is not None:
        config["conditioning"] = conditioning
        config["generation"] = build_generation_provenance(
            model={
                "format": "nanochat_checkpoint",
                "backend": "nanochat",
                "architecture": "EncoderDecoderGPT",
                "model_type": "encoder_decoder",
                "step": 1,
                "checkpoint_sha256": "1" * 64,
                "tokenizer_sha256": "2" * 64,
            },
            condition_name="target",
            device_type="cpu",
            inference_dtype="float32",
            feature_conditions={1: 1},
            target_conditioning_mode="current",
            max_new_tokens=8,
            temperature=0.7,
            top_k=50,
            batch_size=8,
            seed=42,
            prompt_count=len(pairs),
        )
    return build_harmfulness_payload(
        metrics=compute_metrics(
            pairs,
            [JudgeResult([], 1, "Safe."), JudgeResult([], 1, "Safe.")],
        ),
        compliance_metrics=compute_compliance_metrics(
            [pairs[1]],
            [ComplianceJudgeResult(5, False, "Helpful.")],
        ),
        model=MODEL,
        pairs=pairs,
        config=config,
    )


def test_complete_conditions_within_vocab_block_is_valid() -> None:
    payload = _payload(_conditioned_block())

    validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())
    assert payload["config"]["conditioning"]["feature_vocab"] == (VOCAB.metadata_dict())


def test_full_width_conditioned_block_records_null_feature_vocab() -> None:
    block = _conditioned_block()
    block["feature_vocab"] = None

    validate_harmfulness_payload(
        _payload(block),
        model=MODEL,
        pairs=_pairs(),
    )


def test_config_requires_conditioning_block() -> None:
    with pytest.raises(ValueError, match="config.conditioning must be a mapping"):
        _payload(None)


@pytest.mark.parametrize("missing", ["feature_vocab", "context_cache", "coverage"])
def test_conditioned_block_rejects_every_missing_required_field(
    missing: str,
) -> None:
    block = _conditioned_block()
    del block[missing]

    with pytest.raises(ValueError, match="config.conditioning keys must be exactly"):
        _payload(block)


def test_unconditioned_block_rejects_conditioned_fields() -> None:
    block = _conditioned_block()
    block["conditioning_mode"] = "unconditioned"

    with pytest.raises(ValueError, match="config.conditioning keys must be exactly"):
        _payload(block)


def test_feature_vocab_fingerprint_must_use_metadata_schema() -> None:
    block = copy.deepcopy(_conditioned_block())
    feature_vocab = cast(dict[str, object], block["feature_vocab"])
    assert isinstance(feature_vocab, dict)
    feature_vocab["sha256"] = "not-a-feature-vocab-fingerprint"

    with pytest.raises(ValueError, match="feature_vocab.sha256"):
        _payload(block)
