"""Fail-loud regressions for training-progression audit findings."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import scripts.train.training_progression as tp
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.eval.harmfulness_artifact import build_harmfulness_payload
from steering.eval.harmfulness_judge import (
    HARM_CATEGORIES,
    ComplianceJudgeResult,
    JudgeResult,
    compute_compliance_metrics,
    compute_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPO_ROOT / "configs" / "eval"
DECODER_PROGRESSION_CONFIGS = (
    "training_progression_d12_base",
    "training_progression_d12_base_10k",
    "training_progression_d12_base_smoke",
    "training_progression_d13_base",
)


def _base_cfg(
    tmp_path: Path,
    *,
    model_family: str = "decoder",
) -> DictConfig:
    checkpoint_root = (
        "conditional_encdec_checkpoints"
        if model_family == "encoder_decoder"
        else "base_checkpoints"
    )
    cfg = OmegaConf.create(
        {
            "source": "encdec" if model_family == "encoder_decoder" else "base",
            "model_tag": "d12",
            "model_family": model_family,
            "checkpoint_root": checkpoint_root,
            "checkpoint_prefix": "",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "cleanup_after_eval": False,
            "results_file": str(tmp_path / "results.json"),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [
                {"phase": "pretrain", "subdir": "", "steps": [500]},
            ],
            "conditioning": {
                "conditioned_configs": None,
                "context_cache_dir": None,
                "target_real_configs": None,
            },
            "evals": "core,chat,harmfulness",
            "num_seeds": 3,
            "base_seed": 1234,
            "max_per_task": 10,
            "chat_tasks": "ARC-Easy",
            "chat_batch_size": 4,
            "chat_max_new_tokens": 32,
            "chat_temperature": 0.0,
            "chat_top_k": 50,
            "chat_num_samples": 1,
            "chat_max_problems": 5,
            "harm_temperature": 0.7,
            "harm_top_k": 50,
            "harm_max_new_tokens": 32,
            "harm_num_benign": 1,
            "harm_num_harmful": 1,
            "harm_judge_model": "test",
            "device_type": "cpu",
            "dtype": "float32",
            "num_gpus": 1,
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _write_checkpoint(
    cfg,
    *,
    step: int = 500,
    subdir: str = "",
    model_bytes: bytes = b"healthy-model",
) -> Path:
    root = Path(str(cfg.checkpoints_dir)) / str(cfg.checkpoint_root) / str(cfg.model_tag)
    checkpoint_dir = root / subdir if subdir else root
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"model_{step:06d}.pt").write_bytes(model_bytes)
    (checkpoint_dir / f"meta_{step:06d}.json").write_text(
        json.dumps(
            {
                "step": step,
                "model_config": {
                    "vocab_size": 128,
                    "n_layer": 2,
                    "n_head": 2,
                    "n_kv_head": 2,
                    "n_embd": 64,
                    "sequence_len": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_dir


def _decoder_complete_entry(
    *,
    seeds: tuple[int, ...] = (1234, 2234, 3234),
) -> dict:
    return {
        "step": 500,
        "phase": "pretrain",
        "evals": {
            "core": {"seeds": {str(seed): {"core_metric": 0.5} for seed in seeds}},
            "chat": {"seeds": {"1234": {"ARC-Easy": 0.5}}},
            "harmfulness": {
                "seeds": {
                    str(seed): {
                        "harmfulness_eval_schema_version": 4,
                        "harmful_prompts": {
                            "asr": 0.5,
                            "safe_response_rate": 0.25,
                        },
                        "benign_prompts": {"compliance_rate": 0.75},
                        "overall": {"mean_score": 3.0},
                    }
                    for seed in seeds
                }
            },
        },
    }


class TestTP002DecoderVariant:
    @pytest.mark.parametrize("config_name", DECODER_PROGRESSION_CONFIGS)
    def test_shipped_decoder_configs_schedule_every_checkpoint(
        self,
        config_name: str,
    ) -> None:
        with initialize_config_dir(
            config_dir=str(EVAL_CONFIG_DIR),
            version_base=None,
        ):
            cfg = compose(config_name=config_name)

        eval_suites = [name.strip() for name in str(cfg.evals).split(",")]
        seeds = [int(cfg.base_seed) + index * 1000 for index in range(int(cfg.num_seeds))]

        specs = tp.build_checkpoint_specs(cfg)
        assert specs
        for spec in specs:
            assert tp._needs_eval(
                None,
                eval_suites,
                seeds,
                cfg,
                spec.phase,
            )

    def test_pending_decoder_work_uses_suite_seed_policy(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        work = tp._pending_eval_work_keys(
            None,
            ["core", "chat", "harmfulness"],
            [1234, 2234, 3234],
            cfg,
            "pretrain",
        )

        assert all(item[1] == "decoder" for item in work)
        assert [item[0] for item in work].count("core") == 3
        assert [item[0] for item in work].count("chat") == 1
        assert [item[0] for item in work].count("harmfulness") == 3

    def test_decoder_evaluation_runs_and_resume_skips(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        checkpoint_dir = _write_checkpoint(cfg)
        spec = tp.CheckpointSpec(step=500, phase="pretrain", subdir="")
        model = MagicMock()
        tokenizer = MagicMock()

        def suite_result(eval_name, _model, _tokenizer, _device, seed, _cfg, **_kwargs):
            if eval_name == "harmfulness":
                pairs = [
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
                payload = build_harmfulness_payload(
                    metrics=compute_metrics(
                        pairs,
                        [
                            JudgeResult([], 1, "Safe."),
                            JudgeResult([], 1, "Safe."),
                        ],
                    ),
                    compliance_metrics=compute_compliance_metrics(
                        [pairs[1]],
                        [ComplianceJudgeResult(5, False, "Helpful.")],
                    ),
                    model=str(_cfg.harm_judge_model),
                    pairs=pairs,
                )
                payload["progression_provenance"] = tp._harmfulness_progression_provenance(
                    _cfg, seed
                )
                return payload
            return {"metric": f"{eval_name}-{seed}"}

        with (
            patch.object(
                tp,
                "load_progression_model",
                return_value=(
                    model,
                    tokenizer,
                    {"checkpoint_dir": str(checkpoint_dir)},
                    None,
                ),
            ),
            patch.object(tp, "_run_eval_suite", side_effect=suite_result) as run_suite,
        ):
            evals = tp.evaluate_single_checkpoint(
                spec,
                t.device("cpu"),
                t.float32,
                ["core", "chat", "harmfulness"],
                [1234, 2234, 3234],
                cfg,
            )
            resumed = tp.evaluate_single_checkpoint(
                spec,
                t.device("cpu"),
                t.float32,
                ["core", "chat", "harmfulness"],
                [1234, 2234, 3234],
                cfg,
                existing_entry={"evals": evals},
            )

        assert set(evals["core"]["seeds"]) == {"1234", "2234", "3234"}
        assert set(evals["chat"]["seeds"]) == {"1234"}
        assert set(evals["harmfulness"]["seeds"]) == {"1234", "2234", "3234"}
        assert resumed == evals
        assert run_suite.call_count == 7

    def test_decoder_results_plot_as_unconditional_baseline(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        Path(str(cfg.results_file)).write_text(
            json.dumps(
                {
                    "model_tag": "d12",
                    "checkpoints": [_decoder_complete_entry(seeds=(1234,))],
                }
            ),
            encoding="utf-8",
        )

        tp.plot_results(cfg)

        assert (Path(str(cfg.fig_dir)) / "core_metric.png").is_file()
        assert (Path(str(cfg.fig_dir)) / "harmfulness_asr.png").is_file()

    def test_current_conditioned_seed_skips_execution_after_wrapper_build(
        self,
        tmp_path: Path,
    ) -> None:
        """The conditioned execution path honors the seed payload validator."""
        cfg = _base_cfg(tmp_path, model_family="encoder_decoder")
        cfg.conditioning.conditioned_configs = [
            {"name": "static_delta", "features": {0: 1}},
        ]
        cfg.conditioning.context_cache_dir = str(tmp_path / "context-cache")
        checkpoint_dir = _write_checkpoint(cfg)
        spec = tp.CheckpointSpec(step=500, phase="pretrain", subdir="")

        model = MagicMock()
        model.encoder.config = SimpleNamespace(n_sae_features=16, num_buckets=2)
        model.dec_config = SimpleNamespace(sequence_len=128)
        tokenizer = MagicMock()
        wrapper = MagicMock()
        context_provider = SimpleNamespace(cache_dir=tmp_path / "context-cache" / "c128")
        existing_evals = {
            "core": {
                "conditioned": {
                    "configs": {
                        "static_delta": {
                            "kind": "static_yaml",
                            "target_conditioning_mode": "current",
                            "seeds": {"1234": {"core_metric": 0.5}},
                        }
                    }
                }
            }
        }

        with (
            patch.object(tp, "_is_encoder_decoder_model", return_value=True),
            patch.object(
                tp,
                "load_progression_model",
                return_value=(
                    model,
                    tokenizer,
                    {"checkpoint_dir": str(checkpoint_dir)},
                    CheckpointFeatureContract(None, None),
                ),
            ),
            patch.object(tp, "_validate_conditioned_configs"),
            patch.object(tp, "_context_provider_for_cfg", return_value=context_provider),
            # This scheduling test uses a mock tokenizer/provider; cache coverage
            # is exercised by the dedicated real-cache tests, not live CORE data.
            patch.object(tp, "_assert_static_conditioning_context_coverage"),
            patch.object(tp, "_wrap_conditioned_model", return_value=wrapper) as wrap,
            patch.object(tp, "_run_eval_suite") as run_suite,
        ):
            resumed = tp.evaluate_single_checkpoint(
                spec,
                t.device("cpu"),
                t.float32,
                ["core"],
                [1234],
                cfg,
                existing_entry={"evals": existing_evals},
            )

        resumed_config = resumed["core"]["conditioned"]["configs"]["static_delta"]
        assert resumed_config["seeds"] == {"1234": {"core_metric": 0.5}}
        assert resumed_config["display_name"] == "static_delta"
        assert resumed_config["target_features_file"] is None
        assert resumed_config["context_cache_dir"] == str(context_provider.cache_dir)
        assert resumed_config["target_conditioning_mode"] == "current"
        wrap.assert_called_once()
        run_suite.assert_not_called()

    def test_encoder_decoder_without_conditioning_stays_prohibited(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path, model_family="encoder_decoder")

        assert not tp._needs_eval(
            None,
            ["core"],
            [1234],
            cfg,
            "pretrain",
        )


class TestTP008NestedCheckpointAmbiguity:
    def test_ambiguous_nested_candidates_raise_and_list_all(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path, model_family="encoder_decoder")
        root = Path(str(cfg.checkpoints_dir)) / str(cfg.checkpoint_root) / str(cfg.model_tag)
        leaf_a = _write_checkpoint(cfg, subdir="mask_a")
        leaf_b = _write_checkpoint(cfg, subdir="mask_b")

        with pytest.raises(ValueError) as exc_info:
            tp.resolve_local_checkpoint_dir(
                tp.CheckpointSpec(500, "pretrain", ""),
                cfg,
            )

        message = str(exc_info.value)
        assert str(leaf_a) in message
        assert str(leaf_b) in message
        assert "checkpoint_prefix" in message
        assert str(root) in message

    def test_unique_nested_candidate_is_healthy(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path, model_family="encoder_decoder")
        leaf = _write_checkpoint(cfg, subdir="mask_a")

        resolved = tp.resolve_local_checkpoint_dir(
            tp.CheckpointSpec(500, "pretrain", ""),
            cfg,
        )

        assert resolved == str(leaf)

    def test_exact_prefix_selects_intended_healthy_leaf(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path, model_family="encoder_decoder")
        intended = _write_checkpoint(cfg, subdir="mask_z")
        _write_checkpoint(cfg, subdir="mask_a", model_bytes=b"other")
        cfg.checkpoint_prefix = "mask_z"

        resolved = tp.resolve_local_checkpoint_dir(
            tp.CheckpointSpec(500, "pretrain", ""),
            cfg,
        )

        assert resolved == str(intended)


class TestTP012HarmfulnessJudgeFailures:
    @pytest.mark.parametrize("failed_indices", ([1], [0, 1]))
    def test_any_failed_judgment_raises_before_metrics(
        self,
        tmp_path: Path,
        failed_indices: list[int],
    ) -> None:
        cfg = _base_cfg(tmp_path)
        engine = MagicMock()
        engine.generate_batch.return_value = ["response-a", "response-b"]
        judge_results: list[SimpleNamespace | None] = [SimpleNamespace(score=3, categories=[])] * 2
        for index in failed_indices:
            judge_results[index] = None
        judge = MagicMock()
        judge.score_all.return_value = judge_results

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}),
            patch.object(
                tp.harmfulness_judge,
                "load_harmful_prompts",
                return_value=[{"prompt": "bad", "category": "violence"}],
            ),
            patch.object(
                tp.harmfulness_judge,
                "load_benign_prompts",
                return_value=[{"prompt": "good", "category": "none"}],
            ),
            patch.object(tp, "InferenceEngine", return_value=engine),
            patch.object(tp.harmfulness_judge, "JudgeClient", return_value=judge),
            patch.object(tp.harmfulness_judge, "compute_metrics") as compute,
        ):
            with pytest.raises(ValueError) as exc_info:
                tp.run_harmfulness_eval(
                    MagicMock(),
                    MagicMock(),
                    "cpu",
                    42,
                    cfg,
                )

        message = str(exc_info.value)
        assert f"{len(failed_indices)} of 2" in message
        assert str(failed_indices) in message
        compute.assert_not_called()

    def test_judge_result_cardinality_must_match_prompts(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        engine = MagicMock()
        engine.generate_batch.return_value = ["response-a", "response-b"]
        judge = MagicMock()
        judge.score_all.return_value = [
            SimpleNamespace(score=3, categories=[]),
        ]

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}),
            patch.object(
                tp.harmfulness_judge,
                "load_harmful_prompts",
                return_value=[{"prompt": "bad", "category": "violence"}],
            ),
            patch.object(
                tp.harmfulness_judge,
                "load_benign_prompts",
                return_value=[{"prompt": "good", "category": "none"}],
            ),
            patch.object(tp, "InferenceEngine", return_value=engine),
            patch.object(tp.harmfulness_judge, "JudgeClient", return_value=judge),
            patch.object(tp.harmfulness_judge, "compute_metrics") as compute,
        ):
            with pytest.raises(
                ValueError,
                match="returned 1 results for 2 prompts",
            ):
                tp.run_harmfulness_eval(
                    MagicMock(),
                    MagicMock(),
                    "cpu",
                    42,
                    cfg,
                )

        compute.assert_not_called()

    def test_partial_judge_failure_is_not_saved_by_collect(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        cfg.evals = "harmfulness"
        cfg.num_seeds = 1
        checkpoint_dir = _write_checkpoint(cfg)

        with (
            patch.object(
                tp,
                "load_progression_model",
                return_value=(
                    MagicMock(),
                    MagicMock(),
                    {"checkpoint_dir": str(checkpoint_dir)},
                    None,
                ),
            ),
            patch.object(
                tp,
                "run_harmfulness_eval",
                side_effect=ValueError(
                    "Harmfulness judge failed 1 of 2 prompts; failed indices=[1]"
                ),
            ),
            patch.object(tp, "save_checkpoint_result") as save_checkpoint,
        ):
            with pytest.raises(ValueError, match="failed 1 of 2"):
                tp.collect_results(cfg)

        save_checkpoint.assert_not_called()

    def test_fully_scored_run_persists_explicit_counts(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        engine = MagicMock()
        engine.generate_batch.return_value = ["response-a", "response-b"]
        judge = MagicMock()
        judge.score_all.return_value = [
            SimpleNamespace(score=4, categories=["violence"]),
            SimpleNamespace(score=1, categories=[]),
        ]
        judge.score_compliance_all.return_value = [
            ComplianceJudgeResult(5, False, "Complete benign answer.")
        ]
        metrics = {
            "harmful_prompts": {
                "asr": 1.0,
                "mean_score": 4.0,
                "harmfulness_rate": 1.0,
                "safe_response_rate": 0.0,
                "num_scored": 1,
                "num_failed": 0,
            },
            "benign_prompts": {
                "mean_score": 1.0,
                "harmfulness_rate": 0.0,
                "safe_response_rate": 1.0,
                "num_scored": 1,
                "num_failed": 0,
            },
            "overall": {
                "mean_score": 2.5,
                "harmfulness_rate": 0.5,
                "safe_response_rate": 0.5,
                "num_scored": 2,
                "num_failed": 0,
            },
            "per_category": dict.fromkeys(HARM_CATEGORIES, 0.0),
        }

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}),
            patch.object(
                tp.harmfulness_judge,
                "load_harmful_prompts",
                return_value=[{"prompt": "bad", "category": "violence"}],
            ),
            patch.object(
                tp.harmfulness_judge,
                "load_benign_prompts",
                return_value=[{"prompt": "good", "category": "none"}],
            ),
            patch.object(tp, "InferenceEngine", return_value=engine),
            patch.object(tp.harmfulness_judge, "JudgeClient", return_value=judge),
            patch.object(
                tp.harmfulness_judge,
                "compute_metrics",
                return_value=metrics,
            ),
        ):
            result = tp.run_harmfulness_eval(
                MagicMock(),
                MagicMock(),
                "cpu",
                42,
                cfg,
            )

        assert result["harmful_prompts"]["asr"] == 1.0
        assert result["harmful_prompts"]["safe_response_rate"] == 0.0
        assert result["benign_prompts"]["compliance_rate"] == 1.0
        assert result["overall"]["num_scored"] == 2
        assert result["overall"]["num_failed"] == 0


class TestTP013HarmfulnessGenerationIntegrity:
    @pytest.mark.parametrize("rendered_length", (3, 4, 5))
    def test_legacy_target_aware_path_refused_at_every_prompt_length(self, tmp_path, rendered_length):
        model = MagicMock(is_target_aware_encdec=True)
        tokenizer = MagicMock()
        tokenizer.render_for_completion.return_value = list(range(rendered_length))
        with patch.object(tp.harmfulness_judge, "JudgeClient") as judge:
            with pytest.raises(ValueError, match="legacy static wrappers are unsupported"):
                tp.run_harmfulness_eval(model, tokenizer, "cpu", 42, _base_cfg(tmp_path))
        judge.assert_not_called()
        tokenizer.render_for_completion.assert_not_called()

    def test_short_generation_batch_raises_instead_of_partial_judging(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _base_cfg(tmp_path)
        engine = MagicMock()
        engine.generate_batch.return_value = []

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}),
            patch.object(
                tp.harmfulness_judge,
                "load_harmful_prompts",
                return_value=[{"prompt": "bad", "category": "violence"}],
            ),
            patch.object(
                tp.harmfulness_judge,
                "load_benign_prompts",
                return_value=[],
            ),
            patch.object(tp, "InferenceEngine", return_value=engine),
            patch.object(tp.harmfulness_judge, "JudgeClient") as judge,
        ):
            with pytest.raises(
                ValueError,
                match="returned 0 responses for 1 prompts",
            ):
                tp.run_harmfulness_eval(
                    MagicMock(),
                    MagicMock(),
                    "cpu",
                    42,
                    cfg,
                )

        judge.assert_not_called()
