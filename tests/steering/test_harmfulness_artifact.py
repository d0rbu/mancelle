"""Artifact, provenance, validation, and resume tests for issue #1338."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import yaml

from steering.eval import harmfulness_artifact
from steering.eval.harmfulness_artifact import (
    HARMFULNESS_EVAL_SCHEMA_VERSION,
    build_harmfulness_payload,
    build_harmfulness_responses_artifact,
    build_judge_provenance,
    compliance_prompt_sha256,
    context_cache_manifest_sha256,
    current_harmfulness_payload,
    harmfulness_prompt_sha256,
    ordered_pairs_sha256,
    validate_harmfulness_contract,
    validate_harmfulness_payload,
    write_harmfulness_payload,
    write_harmfulness_responses_artifact,
)
from steering.eval.harmfulness_generation import (
    build_generation_provenance,
    validate_generation_provenance,
)
from steering.eval.harmfulness_judge import (
    ComplianceJudgeResult,
    JudgeResult,
    compute_compliance_metrics,
    compute_metrics,
)

MODEL = "openai/gpt-5.6-luna"


def _generation() -> dict[str, Any]:
    return build_generation_provenance(
        model={
            "format": "nanochat_checkpoint",
            "backend": "nanochat",
            "architecture": "GPT",
            "model_type": "decoder_only",
            "step": 12,
            "checkpoint_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
        },
        condition_name="unconditional",
        device_type="cpu",
        inference_dtype="float32",
        feature_conditions=None,
        target_conditioning_mode="none",
        max_new_tokens=128,
        temperature=0.0,
        top_k=50,
        batch_size=8,
        seed=42,
        prompt_count=3,
    )


def _pairs() -> list[dict[str, str]]:
    return [
        {
            "prompt": "harmful instruction",
            "response": "I cannot help with that.",
            "pool": "harmful",
            "category": "Violence",
        },
        {
            "prompt": "benign one",
            "response": "Complete answer one.",
            "pool": "benign",
            "category": "benign",
        },
        {
            "prompt": "benign two",
            "response": "I cannot help with that.",
            "pool": "benign",
            "category": "benign",
        },
    ]


def _payload() -> dict:
    pairs = _pairs()
    harm_results = [
        JudgeResult(categories=[], score=1, reasoning="Safe refusal."),
        JudgeResult(categories=[], score=1, reasoning="Benign answer."),
        JudgeResult(categories=[], score=1, reasoning="Benign refusal."),
    ]
    benign_pairs = [pair for pair in pairs if pair["pool"] == "benign"]
    compliance_results = [
        ComplianceJudgeResult(5, False, "Fully helpful."),
        ComplianceJudgeResult(1, True, "Unnecessary refusal."),
    ]
    return build_harmfulness_payload(
        metrics=compute_metrics(pairs, harm_results),
        compliance_metrics=compute_compliance_metrics(
            benign_pairs,
            compliance_results,
        ),
        model=MODEL,
        pairs=pairs,
        config={
            "judge_model": MODEL,
            "seed": 42,
            "conditioning": {
                "conditioning_mode": "unconditioned",
                "eval_name": "harmfulness_eval",
            },
            "generation": _generation(),
        },
    )


def _payload_with_compliance_failure() -> dict:
    pairs = _pairs()
    harm_results: list[JudgeResult | None] = [
        JudgeResult(categories=[], score=1, reasoning="Safe.") for _ in pairs
    ]
    benign_pairs = [pair for pair in pairs if pair["pool"] == "benign"]
    compliance_results = [ComplianceJudgeResult(5, False, "Fully helpful."), None]
    return build_harmfulness_payload(
        metrics=compute_metrics(pairs, harm_results),
        compliance_metrics=compute_compliance_metrics(benign_pairs, compliance_results),
        model=MODEL,
        pairs=pairs,
    )


class TestPairIdentity:
    def test_digest_is_stable_for_same_semantic_ordered_pairs(self) -> None:
        pairs = _pairs()
        reordered_keys = [
            {
                "category": pair["category"],
                "pool": pair["pool"],
                "response": pair["response"],
                "prompt": pair["prompt"],
            }
            for pair in pairs
        ]
        assert ordered_pairs_sha256(pairs) == ordered_pairs_sha256(reordered_keys)

    @pytest.mark.parametrize("field", ["prompt", "response", "pool", "category"])
    def test_digest_changes_for_every_judgment_relevant_field(
        self,
        field: str,
    ) -> None:
        pairs = _pairs()
        changed = copy.deepcopy(pairs)
        changed[0][field] = "benign" if field == "pool" else changed[0][field] + " changed"
        assert ordered_pairs_sha256(pairs) != ordered_pairs_sha256(changed)

    def test_digest_changes_when_pair_order_changes(self) -> None:
        pairs = _pairs()
        assert ordered_pairs_sha256(pairs) != ordered_pairs_sha256([pairs[1], pairs[0], pairs[2]])

    @pytest.mark.parametrize("field", ["prompt", "response", "pool", "category"])
    def test_digest_rejects_missing_or_non_string_fields(self, field: str) -> None:
        missing = _pairs()
        del missing[0][field]
        with pytest.raises(ValueError, match=field):
            ordered_pairs_sha256(missing)

        invalid = _pairs()
        cast(dict[str, Any], invalid[0])[field] = 7
        with pytest.raises(ValueError, match=field):
            ordered_pairs_sha256(invalid)

    def test_digest_rejects_unknown_pool(self) -> None:
        pairs = _pairs()
        pairs[0]["pool"] = "unknown"
        with pytest.raises(ValueError, match="invalid pool"):
            ordered_pairs_sha256(pairs)

    def test_digest_rejects_non_mapping_pair(self) -> None:
        with pytest.raises(ValueError, match="pair 0 must be a mapping"):
            ordered_pairs_sha256(cast(list[dict[str, Any]], ["not-a-mapping"]))


class TestProvenance:
    def test_generation_identity_covers_seed_batch_checkpoint_and_context(self) -> None:
        generation = _generation()
        assert generation["seed"] == 42
        assert generation["batch_size"] == 8

        changed = copy.deepcopy(generation)
        changed["seed"] = 43
        assert changed != generation
        changed = copy.deepcopy(generation)
        changed["model"]["checkpoint_sha256"] = "3" * 64
        assert changed != generation

        first = {"shards": [{"file": "a", "sha256": "1" * 64}], "rows": 1200}
        reordered = {"rows": 1200, "shards": [{"sha256": "1" * 64, "file": "a"}]}
        assert context_cache_manifest_sha256(first) == context_cache_manifest_sha256(reordered)
        reordered["rows"] = 1199
        assert context_cache_manifest_sha256(first) != context_cache_manifest_sha256(reordered)

    def test_generation_schema_rejects_unknown_keys(self) -> None:
        generation = _generation()
        generation["obsolete_mode"] = "legacy"
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_generation_provenance(generation)

    def test_provenance_is_complete_and_exact(self) -> None:
        provenance = build_judge_provenance(model=MODEL, pairs=_pairs())
        assert provenance == {
            "model": MODEL,
            "harmfulness_prompt_sha256": harmfulness_prompt_sha256(),
            "harmfulness_result_schema_version": 1,
            "compliance_prompt_sha256": compliance_prompt_sha256(),
            "compliance_result_schema_version": 2,
            "compliance_cache_namespace": "benign-compliance-v2",
            "ordered_pairs_sha256": ordered_pairs_sha256(_pairs()),
        }
        assert len(provenance["harmfulness_prompt_sha256"]) == 64
        assert len(provenance["compliance_prompt_sha256"]) == 64
        assert provenance["harmfulness_prompt_sha256"] != provenance["compliance_prompt_sha256"]

    @pytest.mark.parametrize("model", ["", " openai/model", "openai/model ", None, 7])
    def test_provenance_rejects_invalid_model(self, model: object) -> None:
        with pytest.raises(ValueError, match="model"):
            build_judge_provenance(model=cast(str, model), pairs=_pairs())

    def test_prompt_hashes_cover_both_system_and_user_templates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harm_before = harmfulness_prompt_sha256()
        compliance_before = compliance_prompt_sha256()
        monkeypatch.setattr(
            harmfulness_artifact,
            "JUDGE_SYSTEM_PROMPT",
            harmfulness_artifact.JUDGE_SYSTEM_PROMPT + " changed",
        )
        assert harmfulness_prompt_sha256() != harm_before
        monkeypatch.setattr(
            harmfulness_artifact,
            "COMPLIANCE_USER_TEMPLATE",
            harmfulness_artifact.COMPLIANCE_USER_TEMPLATE + " changed",
        )
        assert compliance_prompt_sha256() != compliance_before


class TestPayloadBuildAndValidation:
    def test_builds_shared_nested_schema_with_explicit_conditioning(self) -> None:
        payload = _payload()
        assert payload["harmfulness_eval_schema_version"] == 4
        assert payload["config"] == {
            "judge_model": MODEL,
            "seed": 42,
            "conditioning": {
                "conditioning_mode": "unconditioned",
                "eval_name": "harmfulness_eval",
            },
            "generation": _generation(),
        }
        assert payload["harmful_prompts"] == {
            "mean_score": 1.0,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 1.0,
            "asr": 0.0,
            "num_scored": 1,
            "num_failed": 0,
        }
        assert payload["overall"] == {
            "mean_score": 1.0,
            "harmfulness_rate": 0.0,
            "safe_response_rate": 1.0,
            "num_scored": 3,
            "num_failed": 0,
        }
        assert payload["benign_prompts"]["compliance_rate"] == 0.5
        assert payload["benign_prompts"]["compliance_num_scored"] == 2
        assert set(payload["benign_prompts"]) == {
            "mean_score",
            "harmfulness_rate",
            "safe_response_rate",
            "num_scored",
            "num_failed",
            "compliance_rate",
            "compliance_num_scored",
            "compliance_num_failed",
            "compliance_judgments",
        }
        assert len(payload["benign_prompts"]["compliance_judgments"]) == 2

    def test_progression_payload_can_omit_standalone_config(self) -> None:
        payload = _payload()
        payload.pop("config")
        validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_extra_section_does_not_invalidate_judgments(self) -> None:
        payload = _payload()
        payload["extra_metrics"] = {"schema_version": 1, "records": []}
        validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        "field",
        [
            "model",
            "harmfulness_prompt_sha256",
            "harmfulness_result_schema_version",
            "compliance_prompt_sha256",
            "compliance_result_schema_version",
            "compliance_cache_namespace",
            "ordered_pairs_sha256",
        ],
    )
    def test_every_provenance_field_is_resume_critical(self, field: str) -> None:
        payload = _payload()
        payload["judge_provenance"][field] = "stale"
        with pytest.raises(ValueError, match="judge_provenance mismatch"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_missing_and_extra_provenance_fields_are_rejected(self) -> None:
        missing = _payload()
        del missing["judge_provenance"]["model"]
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_harmfulness_payload(missing, model=MODEL, pairs=_pairs())

        extra = _payload()
        extra["judge_provenance"]["unversioned"] = True
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_harmfulness_payload(extra, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize("version", [None, 1, 2, 3, "4", True])
    def test_wrong_artifact_schema_version_is_rejected(self, version: object) -> None:
        payload = _payload()
        payload["harmfulness_eval_schema_version"] = version
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("section", "scored", "failed"),
        [
            ("overall", 2, 0),
            ("harmful_prompts", 0, 0),
            ("benign_prompts", 1, 0),
        ],
    )
    def test_harmfulness_cardinality_mismatch_is_rejected(
        self,
        section: str,
        scored: int,
        failed: int,
    ) -> None:
        payload = _payload()
        payload[section]["num_scored"] = scored
        payload[section]["num_failed"] = failed
        with pytest.raises(ValueError, match="count mismatch"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("section", "field", "value"),
        [
            ("overall", "safe_response_rate", -0.1),
            ("overall", "harmfulness_rate", float("nan")),
            ("harmful_prompts", "asr", 1.1),
            ("benign_prompts", "compliance_rate", float("inf")),
            ("benign_prompts", "compliance_rate", True),
        ],
    )
    def test_invalid_rates_are_rejected(
        self,
        section: str,
        field: str,
        value: object,
    ) -> None:
        payload = _payload()
        payload[section][field] = value
        with pytest.raises(ValueError, match=field):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("section", "value"),
        [
            ("overall", float("nan")),
            ("overall", 0.0),
            ("overall", 5.1),
            ("harmful_prompts", True),
        ],
    )
    def test_invalid_mean_scores_are_rejected(self, section: str, value: object) -> None:
        payload = _payload()
        payload[section]["mean_score"] = value
        with pytest.raises(ValueError, match="mean_score"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_asr_must_equal_harmful_pool_harmfulness_rate(self) -> None:
        payload = _payload()
        payload["harmful_prompts"]["asr"] = 0.5
        with pytest.raises(ValueError, match="must equal"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_per_category_taxonomy_and_rates_are_strict(self) -> None:
        missing = _payload()
        missing["per_category"].pop(next(iter(missing["per_category"])))
        with pytest.raises(ValueError, match="19-category"):
            validate_harmfulness_payload(missing, model=MODEL, pairs=_pairs())

        invalid = _payload()
        category = next(iter(invalid["per_category"]))
        invalid["per_category"][category] = float("inf")
        with pytest.raises(ValueError, match="per_category"):
            validate_harmfulness_payload(invalid, model=MODEL, pairs=_pairs())

    def test_compliance_record_order_and_cardinality_are_rejected(self) -> None:
        payload = _payload()
        records = payload["benign_prompts"]["compliance_judgments"]
        records[0]["pair_index"] = 1
        with pytest.raises(ValueError, match="pair_index"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"].pop()
        with pytest.raises(ValueError, match="cardinality mismatch"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0]["pair_index"] = False
        with pytest.raises(ValueError, match="pair_index"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("is_refusal", True),
            ("instruction", "different input"),
            ("response", "different output"),
        ],
    )
    def test_record_or_derived_value_drift_is_rejected(
        self,
        field: str,
        value: object,
    ) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0][field] = value
        with pytest.raises(ValueError, match="expected False|aggregate/record mismatch"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_helpfulness_change_alone_preserves_valid_payload(self, score: int) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0]["helpfulness_score"] = score
        validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_low_helpfulness_non_refusal_and_high_helpfulness_refusal_validate(self) -> None:
        payload = _payload()
        records = payload["benign_prompts"]["compliance_judgments"]
        records[0]["helpfulness_score"] = 1
        records[1]["helpfulness_score"] = 5
        validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())
        assert records[0]["is_compliant"] is True
        assert records[1]["is_compliant"] is False

    def test_derived_compliance_value_must_be_exact_boolean(self) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0]["is_compliant"] = False
        with pytest.raises(ValueError, match="expected True"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0]["is_compliant"] = 1
        with pytest.raises(ValueError, match="invalid is_compliant"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("compliance_rate", 0.0),
            ("compliance_num_scored", 1),
            ("compliance_num_failed", 1),
        ],
    )
    def test_aggregate_drift_is_rejected(self, field: str, value: object) -> None:
        payload = _payload()
        payload["benign_prompts"][field] = value
        with pytest.raises(ValueError):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_record_schema_is_exact(self) -> None:
        payload = _payload()
        del payload["benign_prompts"]["compliance_judgments"][0]["reasoning"]
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0][7] = "invalid key"
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize("field", ["instruction", "response"])
    def test_record_inputs_must_be_strings(self, field: str) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0][field] = 7
        with pytest.raises(ValueError, match=f"non-string {field}"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0]["confidence"] = 1.0
        with pytest.raises(ValueError, match="keys must be exactly"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_build_rejects_compliance_overwrite(self) -> None:
        pairs = _pairs()
        metrics = compute_metrics(
            pairs,
            [JudgeResult([], 1, "Safe.") for _ in pairs],
        )
        with pytest.raises(ValueError, match="overwrite"):
            build_harmfulness_payload(
                metrics=metrics,
                compliance_metrics={"safe_response_rate": 0.0},
                model=MODEL,
                pairs=pairs,
            )

    def test_build_rejects_non_mapping_benign_metrics(self) -> None:
        pairs = _pairs()
        metrics = compute_metrics(pairs, [JudgeResult([], 1, "Safe.") for _ in pairs])
        metrics["benign_prompts"] = []
        with pytest.raises(ValueError, match="benign_prompts mapping"):
            build_harmfulness_payload(
                metrics=metrics,
                compliance_metrics={},
                model=MODEL,
                pairs=pairs,
            )

    @pytest.mark.parametrize("missing", ["overall", "harmful_prompts", "per_category"])
    def test_build_rejects_every_missing_harmfulness_section(self, missing: str) -> None:
        pairs = _pairs()
        metrics = compute_metrics(pairs, [JudgeResult([], 1, "Safe.") for _ in pairs])
        del metrics[missing]
        with pytest.raises(ValueError, match=missing):
            build_harmfulness_payload(
                metrics=metrics,
                compliance_metrics=compute_compliance_metrics(
                    [pair for pair in pairs if pair["pool"] == "benign"],
                    [ComplianceJudgeResult(5, False, "Helpful."), None],
                ),
                model=MODEL,
                pairs=pairs,
            )

    @pytest.mark.parametrize("payload", [None, [], "mapping"])
    def test_payload_root_must_be_a_mapping(self, payload: object) -> None:
        with pytest.raises(ValueError, match="harmfulness result must be a mapping"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
    def test_counts_must_be_nonnegative_integers(self, value: object) -> None:
        payload = _payload()
        payload["overall"]["num_scored"] = value
        with pytest.raises(ValueError, match="non-negative integer"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_compliance_records_must_be_a_list(self) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"] = {}
        with pytest.raises(ValueError, match="must be a list"):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_every_compliance_field_is_required(self) -> None:
        for field in (
            "compliance_rate",
            "compliance_num_scored",
            "compliance_num_failed",
            "compliance_judgments",
        ):
            payload = _payload()
            del payload["benign_prompts"][field]
            with pytest.raises(ValueError, match="missing compliance fields"):
                validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())

    def test_failed_compliance_record_is_valid_and_strict(self) -> None:
        payload = _payload_with_compliance_failure()
        validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())
        failed = payload["benign_prompts"]["compliance_judgments"][1]
        assert failed == {
            "pair_index": 1,
            "instruction": "benign two",
            "response": "I cannot help with that.",
            "status": "failed",
            "helpfulness_score": None,
            "is_refusal": None,
            "reasoning": None,
            "is_compliant": None,
            "error": "judge_retries_exhausted",
        }

        for field in ("helpfulness_score", "is_refusal", "reasoning", "is_compliant"):
            invalid = _payload_with_compliance_failure()
            invalid["benign_prompts"]["compliance_judgments"][1][field] = 1
            with pytest.raises(ValueError, match="non-null result fields"):
                validate_harmfulness_payload(invalid, model=MODEL, pairs=_pairs())

        invalid = _payload_with_compliance_failure()
        invalid["benign_prompts"]["compliance_judgments"][1]["error"] = "timeout"
        with pytest.raises(ValueError, match="invalid error"):
            validate_harmfulness_payload(invalid, model=MODEL, pairs=_pairs())

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("status", "unknown", "invalid status"),
            ("error", "unexpected", "error=null"),
            ("helpfulness_score", True, "invalid helpfulness_score"),
            ("helpfulness_score", 0, "invalid helpfulness_score"),
            ("helpfulness_score", 6, "invalid helpfulness_score"),
            ("helpfulness_score", 4.0, "invalid helpfulness_score"),
            ("is_refusal", 0, "invalid is_refusal"),
            ("reasoning", "", "invalid reasoning"),
            ("reasoning", "   ", "invalid reasoning"),
            ("reasoning", None, "invalid reasoning"),
        ],
    )
    def test_scored_compliance_record_types_are_strict(
        self,
        field: str,
        value: object,
        match: str,
    ) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"][0][field] = value
        with pytest.raises(ValueError, match=match):
            validate_harmfulness_payload(payload, model=MODEL, pairs=_pairs())


class TestProgressionContractValidation:
    def test_current_contract_is_accepted_without_regenerating_pairs(self) -> None:
        validate_harmfulness_contract(_payload(), model=MODEL)
        validate_harmfulness_contract(_payload_with_compliance_failure(), model=MODEL)

    @pytest.mark.parametrize("payload", [None, [], "mapping"])
    def test_contract_root_must_be_a_mapping(self, payload: object) -> None:
        with pytest.raises(ValueError, match="harmfulness result must be a mapping"):
            validate_harmfulness_contract(payload, model=MODEL)

    def test_contract_schema_and_provenance_shape_are_exact(self) -> None:
        payload = _payload()
        payload["harmfulness_eval_schema_version"] = 1
        with pytest.raises(ValueError, match="schema_version mismatch"):
            validate_harmfulness_contract(payload, model=MODEL)

        payload = _payload()
        payload["judge_provenance"]["extra"] = True
        with pytest.raises(ValueError, match="exact schema"):
            validate_harmfulness_contract(payload, model=MODEL)

    @pytest.mark.parametrize(
        "field",
        [
            "model",
            "harmfulness_prompt_sha256",
            "harmfulness_result_schema_version",
            "compliance_prompt_sha256",
            "compliance_result_schema_version",
            "compliance_cache_namespace",
        ],
    )
    def test_every_static_contract_field_is_checked(self, field: str) -> None:
        payload = _payload()
        payload["judge_provenance"][field] = "stale"
        with pytest.raises(ValueError, match=field):
            validate_harmfulness_contract(payload, model=MODEL)

    @pytest.mark.parametrize("digest", [None, "0" * 63, "0" * 65, 7])
    def test_contract_pair_digest_requires_sha256_shape(self, digest: object) -> None:
        payload = _payload()
        payload["judge_provenance"]["ordered_pairs_sha256"] = digest
        with pytest.raises(ValueError, match="not SHA-256"):
            validate_harmfulness_contract(payload, model=MODEL)

    def test_contract_pair_digest_requires_hexadecimal(self) -> None:
        payload = _payload()
        payload["judge_provenance"]["ordered_pairs_sha256"] = "z" * 64
        with pytest.raises(ValueError, match="not hexadecimal"):
            validate_harmfulness_contract(payload, model=MODEL)

    def test_contract_checks_pool_totals_and_asr_identity(self) -> None:
        payload = _payload()
        payload["harmful_prompts"]["asr"] = 0.5
        with pytest.raises(ValueError, match="must equal"):
            validate_harmfulness_contract(payload, model=MODEL)

        payload = _payload()
        payload["overall"]["num_scored"] = 2
        payload["overall"]["num_failed"] = 0
        with pytest.raises(ValueError, match="overall count"):
            validate_harmfulness_contract(payload, model=MODEL)

    def test_contract_checks_compliance_field_and_record_cardinality(self) -> None:
        payload = _payload()
        del payload["benign_prompts"]["compliance_rate"]
        with pytest.raises(ValueError, match="missing compliance fields"):
            validate_harmfulness_contract(payload, model=MODEL)

        payload = _payload()
        payload["benign_prompts"]["compliance_judgments"].pop()
        payload["benign_prompts"]["compliance_num_scored"] = 1
        payload["benign_prompts"]["compliance_rate"] = 1.0
        with pytest.raises(ValueError, match="cardinality"):
            validate_harmfulness_contract(payload, model=MODEL)

    def test_contract_recomputes_compliance_aggregate(self) -> None:
        payload = _payload()
        payload["benign_prompts"]["compliance_rate"] = 0.0
        with pytest.raises(ValueError, match="aggregate/record mismatch"):
            validate_harmfulness_contract(payload, model=MODEL)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("compliance_rate", True, "must be numeric"),
            ("compliance_num_scored", True, "non-negative integer"),
            ("compliance_num_failed", -1, "non-negative integer"),
        ],
    )
    def test_contract_rejects_type_coercible_compliance_summary_values(
        self,
        field: str,
        value: object,
        match: str,
    ) -> None:
        payload = _payload()
        payload["benign_prompts"][field] = value
        with pytest.raises(ValueError, match=match):
            validate_harmfulness_contract(payload, model=MODEL)


class TestArtifactIOAndResume:
    def test_yaml_round_trip_is_exact_and_unicode_safe(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["config"]["note"] = "日本語"
        path = tmp_path / "nested" / "result.yaml"
        write_harmfulness_payload(payload, path)
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == payload

    def test_writer_delegates_to_canonical_atomic_primitive(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _payload()
        path = tmp_path / "result.yaml"
        with patch("steering.eval.harmfulness_artifact.atomic_write_text") as atomic:
            write_harmfulness_payload(payload, path)
        assert atomic.call_count == 1
        rendered, target = atomic.call_args.args
        assert yaml.safe_load(rendered) == payload
        assert target == path

    def test_response_artifact_round_trip_preserves_conditioning(
        self,
        tmp_path: Path,
    ) -> None:
        responses = cast(list[dict[str, object]], _pairs())
        conditioning = _payload()["config"]["conditioning"]
        artifact = build_harmfulness_responses_artifact(
            responses=responses,
            conditioning=conditioning,
            generation=_generation(),
        )
        path = tmp_path / "nested" / "responses.yaml"

        write_harmfulness_responses_artifact(artifact, path)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == artifact

    def test_current_artifact_is_returned_without_loss(self, tmp_path: Path) -> None:
        payload = _payload()
        path = tmp_path / "result.yaml"
        write_harmfulness_payload(payload, path)
        current, reason = current_harmfulness_payload(
            path,
            model=MODEL,
            pairs=_pairs(),
        )
        assert current == payload
        assert reason == "current artifact"

    def test_response_conditioning_drift_requires_rejudge(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _payload()
        path = tmp_path / "result.yaml"
        write_harmfulness_payload(payload, path)

        current, reason = current_harmfulness_payload(
            path,
            model=MODEL,
            pairs=_pairs(),
            expected_conditioning={
                "conditioning_mode": "unconditioned",
                "eval_name": "training_progression",
            },
        )

        assert current is None
        assert "generation provenance" in reason

    def test_missing_legacy_corrupt_model_and_pair_drift_require_rejudge(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "result.yaml"
        assert (
            current_harmfulness_payload(
                path,
                model=MODEL,
                pairs=_pairs(),
            )[0]
            is None
        )

        path.write_text("overall: {}\n", encoding="utf-8")
        current, reason = current_harmfulness_payload(
            path,
            model=MODEL,
            pairs=_pairs(),
        )
        assert current is None
        assert "schema_version" in reason

        path.write_text("[unterminated", encoding="utf-8")
        assert (
            current_harmfulness_payload(
                path,
                model=MODEL,
                pairs=_pairs(),
            )[0]
            is None
        )

        malformed = _payload()
        malformed["benign_prompts"]["compliance_judgments"][0][7] = "invalid key"
        path.write_text(yaml.dump(malformed), encoding="utf-8")
        current, reason = current_harmfulness_payload(
            path,
            model=MODEL,
            pairs=_pairs(),
        )
        assert current is None
        assert "keys must be exactly" in reason

        write_harmfulness_payload(_payload(), path)
        assert (
            current_harmfulness_payload(
                path,
                model="different-model",
                pairs=_pairs(),
            )[0]
            is None
        )

        changed_pairs = _pairs()
        changed_pairs[0]["response"] += " changed"
        assert (
            current_harmfulness_payload(
                path,
                model=MODEL,
                pairs=changed_pairs,
            )[0]
            is None
        )

    def test_pair_digest_has_expected_sha256_shape(self) -> None:
        digest = ordered_pairs_sha256(_pairs())
        assert len(digest) == 64
        assert int(digest, 16) >= 0


def test_schema_version_is_deliberately_nonlegacy() -> None:
    assert HARMFULNESS_EVAL_SCHEMA_VERSION == 4
