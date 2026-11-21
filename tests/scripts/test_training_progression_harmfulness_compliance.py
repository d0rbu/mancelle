"""Progression contract and resume tests for issue #1338."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import scripts.train.training_progression as progression
import steering.eval.harmfulness_judge as harmfulness_judge
from scripts.eval.harmfulness_eval import _judge_and_write
from steering.eval.harmfulness_artifact import (
    validate_harmfulness_payload,
)
from steering.eval.harmfulness_generation import build_generation_provenance
from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult

MODEL = "openai/gpt-5.6-luna"


def _cfg(**overrides: object) -> DictConfig:
    values: dict[str, object] = {
        "harm_temperature": 0.0,
        "harm_top_k": 0,
        "harm_max_new_tokens": 16,
        "harm_num_benign": 2,
        "harm_num_harmful": 2,
        "harm_judge_model": MODEL,
        "device_type": "cuda",
        "dtype": "bfloat16",
        "model_family": "decoder",
        "source": "base",
        "conditioning": {
            "conditioned_configs": None,
            "target_real_configs": None,
        },
    }
    values.update(overrides)
    cfg = OmegaConf.create(values)
    assert isinstance(cfg, DictConfig)
    return cfg


def _prompt_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    harmful = [
        {"prompt": "harmful one", "category": "Violence"},
        {"prompt": "harmful two", "category": "Cybercrime"},
    ]
    benign = [
        {"prompt": "benign one", "category": "benign"},
        {"prompt": "benign two", "category": "benign"},
    ]
    return harmful, benign


def _judge() -> MagicMock:
    judge = MagicMock()
    judge.score_all.return_value = [
        JudgeResult([], 1, "Safe refusal."),
        JudgeResult(["Cybercrime"], 5, "Harmful compliance."),
        JudgeResult([], 1, "Safe benign answer."),
        JudgeResult([], 2, "Mostly safe benign refusal."),
    ]
    judge.score_compliance_all.return_value = [
        ComplianceJudgeResult(1, False, "Low helpfulness without refusal."),
        ComplianceJudgeResult(5, True, "Helpful partial refusal."),
    ]
    return judge


def _run_progression(
    *,
    cfg: DictConfig | None = None,
    judge: MagicMock | None = None,
    responses: list[str] | None = None,
) -> tuple[dict, MagicMock, MagicMock]:
    resolved_cfg = cfg or _cfg()
    resolved_judge = judge or _judge()
    engine = MagicMock()
    engine.generate_batch.return_value = responses or ["hr1", "hr2", "br1", "br2"]
    harmful, benign = _prompt_rows()
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        patch.object(
            harmfulness_judge,
            "load_harmful_prompts",
            return_value=harmful,
        ),
        patch.object(
            harmfulness_judge,
            "load_benign_prompts",
            return_value=benign,
        ),
        patch.object(progression, "InferenceEngine", return_value=engine),
        patch.object(
            progression.harmfulness_judge,
            "JudgeClient",
            return_value=resolved_judge,
        ),
    ):
        payload = progression.run_harmfulness_eval(
            MagicMock(is_target_aware_encdec=False),
            MagicMock(),
            t.device("cpu"),
            1234,
            resolved_cfg,
        )
    return payload, resolved_judge, engine


def _pairs_from_run() -> list[dict[str, str]]:
    harmful, benign = _prompt_rows()
    rows = [{**row, "pool": "harmful"} for row in harmful] + [
        {**row, "pool": "benign"} for row in benign
    ]
    for row, response in zip(rows, ["hr1", "hr2", "br1", "br2"], strict=True):
        row["response"] = response
    return rows


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"harm_num_harmful": 1.5}, "max_harmful must be an integer or None"),
        ({"harm_num_benign": False}, "num_benign must be an integer"),
    ],
)
def test_progression_rejects_raw_invalid_prompt_pool_controls(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        _run_progression(cfg=_cfg(**override))


class TestProgressionMetrics:
    def test_uses_harmful_pool_safety_and_benign_compliance_explicitly(self) -> None:
        payload, judge, engine = _run_progression()
        pairs = _pairs_from_run()

        validate_harmfulness_payload(payload, model=MODEL, pairs=pairs)
        assert payload["harmful_prompts"]["safe_response_rate"] == 0.5
        assert payload["overall"]["safe_response_rate"] == 0.5
        assert payload["benign_prompts"]["safe_response_rate"] == 0.5
        assert payload["benign_prompts"]["compliance_rate"] == 0.5
        assert payload["benign_prompts"]["compliance_num_scored"] == 2
        assert payload["benign_prompts"]["compliance_num_failed"] == 0
        assert [
            record["is_compliant"] for record in payload["benign_prompts"]["compliance_judgments"]
        ] == [True, False]
        assert "safe_response_rate" not in {
            key for key in payload if key not in {"harmful_prompts", "benign_prompts", "overall"}
        }
        judge.score_compliance_all.assert_called_once()
        compliance_pairs = judge.score_compliance_all.call_args.args[0]
        assert [pair["prompt"] for pair in compliance_pairs] == [
            "benign one",
            "benign two",
        ]
        assert all(pair["pool"] == "benign" for pair in compliance_pairs)
        _, generation_kwargs = engine.generate_batch.call_args
        assert generation_kwargs["render_as_chat"] is True

    def test_deliberately_divergent_pool_rates_cannot_be_mixed(self) -> None:
        judge = _judge()
        judge.score_all.return_value = [
            JudgeResult([], 1, "Safe."),
            JudgeResult([], 1, "Safe."),
            JudgeResult([], 2, "Not score one."),
            JudgeResult([], 2, "Not score one."),
        ]
        payload, _, _ = _run_progression(judge=judge)
        assert payload["harmful_prompts"]["safe_response_rate"] == 1.0
        assert payload["benign_prompts"]["safe_response_rate"] == 0.0
        assert payload["overall"]["safe_response_rate"] == 0.5

    def test_progression_and_standalone_share_exact_metric_sections(
        self,
        tmp_path: Path,
    ) -> None:
        progression_payload, _, _ = _run_progression()
        pairs = _pairs_from_run()
        standalone_judge = _judge()
        standalone_cfg = MagicMock(
            source="base",
            model_tag="d-test",
            step=1,
            judge_model=MODEL,
        )
        standalone_payload = _judge_and_write(
            standalone_cfg,
            "unconditional",
            str(tmp_path / "standalone.yaml"),
            cast(list[dict[str, object]], pairs),
            standalone_judge,
            conditioning={
                "conditioning_mode": "unconditioned",
                "eval_name": "harmfulness_eval",
            },
            generation=build_generation_provenance(
                model={
                    "format": "nanochat_checkpoint",
                    "backend": "nanochat",
                    "architecture": "GPT",
                    "model_type": "decoder_only",
                    "step": 1,
                    "checkpoint_sha256": "1" * 64,
                    "tokenizer_sha256": "2" * 64,
                },
                condition_name="unconditional",
                device_type="cpu",
                inference_dtype="float32",
                feature_conditions=None,
                target_conditioning_mode="none",
                max_new_tokens=16,
                temperature=0.0,
                top_k=None,
                batch_size=4,
                seed=42,
                prompt_count=4,
            ),
        )
        for key in ("overall", "harmful_prompts", "benign_prompts", "per_category"):
            assert progression_payload[key] == standalone_payload[key]

    def test_progression_fails_on_any_compliance_failure(self) -> None:
        judge = _judge()
        judge.score_compliance_all.return_value = [
            ComplianceJudgeResult(5, False, "Complete."),
            None,
        ]
        with pytest.raises(
            ValueError,
            match=r"failed 1 of 2 prompts; failed benign indices=\[1\]",
        ):
            _run_progression(judge=judge)

    def test_progression_fails_on_compliance_cardinality_mismatch(self) -> None:
        judge = _judge()
        judge.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "Complete.")]
        with pytest.raises(ValueError, match="returned 1 results for 2 prompts"):
            _run_progression(judge=judge)

    def test_progression_never_calls_compliance_for_empty_benign_pool(self) -> None:
        cfg = _cfg(harm_num_benign=0)
        judge = MagicMock()
        judge.score_all.return_value = [
            JudgeResult([], 1, "Safe."),
            JudgeResult([], 1, "Safe."),
        ]
        judge.score_compliance_all.return_value = []
        engine = MagicMock()
        engine.generate_batch.return_value = ["r1", "r2"]
        harmful, _ = _prompt_rows()
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch.object(
                harmfulness_judge,
                "load_harmful_prompts",
                return_value=harmful,
            ),
            patch.object(
                harmfulness_judge,
                "load_benign_prompts",
                return_value=[],
            ),
            patch.object(progression, "InferenceEngine", return_value=engine),
            patch.object(
                progression.harmfulness_judge,
                "JudgeClient",
                return_value=judge,
            ),
        ):
            payload = progression.run_harmfulness_eval(
                MagicMock(is_target_aware_encdec=False),
                MagicMock(),
                "cpu",
                1234,
                cfg,
            )
        judge.score_compliance_all.assert_called_once_with([])
        assert payload["benign_prompts"]["compliance_rate"] == 0.0
        assert payload["benign_prompts"]["compliance_judgments"] == []


class TestProgressionSpecIdentity:
    def test_spec_is_stable_for_identical_inputs(self) -> None:
        assert progression._harmfulness_progression_spec_sha256(
            _cfg(), 1234
        ) == progression._harmfulness_progression_spec_sha256(_cfg(), 1234)

    @pytest.mark.parametrize(
        ("field", "changed"),
        [
            ("harm_temperature", 0.7),
            ("harm_top_k", 50),
            ("harm_max_new_tokens", 17),
            ("harm_num_benign", 3),
            ("harm_num_harmful", 1),
            ("device_type", "cpu"),
            ("dtype", "float32"),
        ],
    )
    def test_every_generation_knob_invalidates_spec(
        self,
        field: str,
        changed: object,
    ) -> None:
        before = progression._harmfulness_progression_spec_sha256(_cfg(), 1234)
        after = progression._harmfulness_progression_spec_sha256(
            _cfg(**{field: changed}),
            1234,
        )
        assert before != after

    @pytest.mark.parametrize(
        ("constant", "changed"),
        [
            ("BEAVERTAILS_EVALUATION_REVISION", "different-harmful-revision"),
            ("WILDGUARD_MIX_DATASET_ID", "different-benign-dataset"),
            ("WILDGUARD_MIX_CONFIG", "different-benign-config"),
            ("WILDGUARD_MIX_SPLIT", "different-benign-split"),
            ("WILDGUARD_MIX_REVISION", "different-benign-revision"),
            ("BENIGN_PROMPT_HARM_LABEL", "different-benign-label"),
            ("BENIGN_PROMPT_HARM_AGREEMENT", 2.0),
            ("BENIGN_PROMPT_SUBCATEGORY", "different-benign-subcategory"),
        ],
    )
    def test_seed_and_every_dataset_contract_field_invalidate_spec(
        self,
        monkeypatch: pytest.MonkeyPatch,
        constant: str,
        changed: object,
    ) -> None:
        before = progression._harmfulness_progression_spec_sha256(_cfg(), 1234)
        assert before != progression._harmfulness_progression_spec_sha256(_cfg(), 1235)

        monkeypatch.setattr(
            progression.harmfulness_judge,
            constant,
            changed,
        )
        assert before != progression._harmfulness_progression_spec_sha256(_cfg(), 1234)


class TestProgressionResume:
    def _entry(self, payload: Any, *, conditioned: bool = False) -> dict:
        if conditioned:
            return {
                "evals": {
                    "harmfulness": {
                        "conditioned": {"configs": {"target": {"seeds": {"1234": payload}}}}
                    }
                }
            }
        return {"evals": {"harmfulness": {"seeds": {"1234": payload}}}}

    def test_current_decoder_and_conditioned_seed_are_complete(self) -> None:
        payload, _, _ = _run_progression()
        cfg = _cfg()
        assert progression._decoder_seed_done(self._entry(payload), "harmfulness", 1234, cfg=cfg)
        assert progression._conditioned_seed_done(
            self._entry(payload, conditioned=True),
            "harmfulness",
            "target",
            1234,
            cfg=cfg,
        )

    def test_legacy_flat_seed_is_never_complete(self) -> None:
        legacy = {
            "asr": 0.2,
            "mean_score": 1.5,
            "safe_response_rate": 0.8,
            "num_scored": 100,
            "num_failed": 0,
        }
        cfg = _cfg()
        assert not progression._decoder_seed_done(self._entry(legacy), "harmfulness", 1234, cfg=cfg)
        assert not progression._conditioned_seed_done(
            self._entry(legacy, conditioned=True),
            "harmfulness",
            "target",
            1234,
            cfg=cfg,
        )

    def test_harmfulness_requires_both_current_config_and_mapping_payload(self) -> None:
        payload, _, _ = _run_progression()
        assert not progression._decoder_seed_done(
            self._entry(payload),
            "harmfulness",
            1234,
            cfg=None,
        )
        assert not progression._decoder_seed_done(
            self._entry(["not", "a", "mapping"]),
            "harmfulness",
            1234,
            cfg=_cfg(),
        )

    @pytest.mark.parametrize(
        "mutation",
        [
            "artifact_schema",
            "judge_model",
            "prompt_digest",
            "compliance_result_schema",
            "compliance_cache_namespace",
            "aggregate",
            "record",
            "progression_schema",
            "progression_spec",
        ],
    )
    def test_every_contract_drift_reopens_seed(self, mutation: str) -> None:
        payload, _, _ = _run_progression()
        stale = copy.deepcopy(payload)
        if mutation == "artifact_schema":
            stale["harmfulness_eval_schema_version"] = 2
        elif mutation == "judge_model":
            stale["judge_provenance"]["model"] = "old-model"
        elif mutation == "prompt_digest":
            stale["judge_provenance"]["compliance_prompt_sha256"] = "0" * 64
        elif mutation == "compliance_result_schema":
            stale["judge_provenance"]["compliance_result_schema_version"] = 1
        elif mutation == "compliance_cache_namespace":
            stale["judge_provenance"]["compliance_cache_namespace"] = "benign-compliance-v1"
        elif mutation == "aggregate":
            stale["benign_prompts"]["compliance_rate"] = 0.0
        elif mutation == "record":
            stale["benign_prompts"]["compliance_judgments"][0]["is_compliant"] = False
        elif mutation == "progression_schema":
            stale["progression_provenance"]["schema_version"] = 1
        elif mutation == "progression_spec":
            stale["progression_provenance"]["evaluation_spec_sha256"] = "0" * 64
        assert not progression._decoder_seed_done(
            self._entry(stale),
            "harmfulness",
            1234,
            cfg=_cfg(),
        )

    def test_generation_config_or_seed_change_reopens_seed(self) -> None:
        payload, _, _ = _run_progression()
        entry = self._entry(payload)
        assert not progression._decoder_seed_done(
            entry,
            "harmfulness",
            1234,
            cfg=_cfg(harm_temperature=0.7),
        )
        assert not progression._decoder_seed_done(
            entry,
            "harmfulness",
            1235,
            cfg=_cfg(),
        )

    def test_non_harmfulness_seed_semantics_remain_legacy_compatible(self) -> None:
        entry = {"evals": {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}}
        assert progression._decoder_seed_done(entry, "core", 1234)
        assert not progression._decoder_seed_done(entry, "core", 9999)
