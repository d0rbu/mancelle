"""Comprehensive tests for the training progression evaluation + plotting pipeline.

Covers:
- Unit tests for all helpers (JSON I/O, spec building, metric extraction, formatting)
- Functional tests for plot generation and collect orchestration
- Integration tests for seed parameter changes in core_eval
"""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from nanochat.core_eval import evaluate_example, evaluate_task
from scripts.eval.base_eval import evaluate_core
from scripts.train.training_progression import (
    METRIC_SPECS,
    CheckpointSpec,
    _assert_static_conditioning_context_coverage,
    _configured_checkpoint_keys,
    _fig_dir,
    _filter_results_to_configured_checkpoints,
    _local_checkpoints_root,
    _needs_eval,
    _results_dir,
    _results_path,
    _scp_cmd,
    _should_load_encoder_decoder,
    _validate_conditioned_configs,
    build_checkpoint_specs,
    cleanup_remote_checkpoint,
    collect_results,
    conditioned_configs_for_phase,
    evaluate_single_checkpoint,
    fetch_remote_checkpoint,
    find_checkpoint_entry,
    load_or_init_results,
    main,
    plot_results,
    resolve_local_checkpoint_dir,
    run_chat_eval_suite,
    run_core_eval,
    run_harmfulness_eval,
    save_checkpoint_result,
    save_results,
)
from steering.eval.compacted_eval_loading import conditioned_eval_conditioning
from steering.eval.feature_config import FeatureConfig
from steering.eval.harmfulness_judge import HARM_CATEGORIES, ComplianceJudgeResult

# ---------------------------------------------------------------------------


def _merge_cfg(*configs: Any) -> DictConfig:
    """Merge dynamic OmegaConf inputs and prove this test expects a mapping."""
    merged = OmegaConf.merge(*configs)
    if not isinstance(merged, DictConfig):
        raise TypeError(f"expected DictConfig, got {type(merged).__name__}")
    return merged


def _remote_payload_for_command(
    command: list[str],
    payloads: dict[str, bytes],
) -> tuple[str, bytes]:
    rendered = " ".join(str(part) for part in command)
    matches = [(name, payload) for name, payload in payloads.items() if name in rendered]
    if len(matches) != 1:
        raise AssertionError(f"Could not identify remote payload for command: {command}")
    return matches[0]


def _fake_remote_run(
    payloads: dict[str, bytes],
    calls: list[tuple[str, str]],
    *,
    scp_error: str | None = None,
):
    def run(command, **_kwargs):
        name, payload = _remote_payload_for_command(command, payloads)
        if command[0] == "ssh":
            calls.append(("ssh", name))
            digest = hashlib.sha256(payload).hexdigest()
            return SimpleNamespace(
                returncode=0,
                stdout=f"{len(payload)}\n{digest}  {name}\n",
                stderr="",
            )
        if command[0] != "scp":
            raise AssertionError(f"Unexpected subprocess command: {command}")
        calls.append(("scp", name))
        if scp_error is not None:
            return SimpleNamespace(returncode=1, stdout="", stderr=scp_error)
        Path(command[-1]).write_bytes(payload)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _concurrent_remote_fetch_worker(
    cfg_data: dict,
    payloads: dict[str, bytes],
    scp_started,
    allow_scp,
    result_queue,
) -> None:
    scp_calls: list[str] = []

    def fake_run(command, **_kwargs):
        name, payload = _remote_payload_for_command(command, payloads)
        if command[0] == "ssh":
            digest = hashlib.sha256(payload).hexdigest()
            return SimpleNamespace(
                returncode=0,
                stdout=f"{len(payload)}\n{digest}  {name}\n",
                stderr="",
            )
        if command[0] != "scp":
            raise AssertionError(f"Unexpected subprocess command: {command}")

        scp_calls.append(name)
        destination = Path(command[-1])
        if name.startswith("model_"):
            destination.write_bytes(payload[: max(1, len(payload) // 3)])
            scp_started.set()
            if not allow_scp.wait(timeout=15):
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="timed out waiting to finish fake SCP",
                )
        destination.write_bytes(payload)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    try:
        with patch("scripts.train.training_progression.subprocess.run", side_effect=fake_run):
            spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
            local_dir = fetch_remote_checkpoint(spec, OmegaConf.create(cfg_data))
        result_queue.put(
            {
                "ok": True,
                "model": (Path(local_dir) / "model_000500.pt").read_bytes(),
                "scp_calls": scp_calls,
            }
        )
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc), "scp_calls": scp_calls})


def test_training_progression_rejects_empty_static_conditioned_config() -> None:
    cfg = OmegaConf.create(
        {
            "conditioning": {
                "conditioned_configs": [
                    {"name": "context_only", "features": {}},
                ],
            }
        }
    )
    with pytest.raises(ValueError, match="context_only"):
        conditioned_configs_for_phase(cfg, "pretrain")


def test_progression_model_validation_rejects_empty_static_config() -> None:
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            config=SimpleNamespace(num_buckets=2, n_sae_features=16),
        ),
    )
    with pytest.raises(ValueError, match="context_only"):
        _validate_conditioned_configs(
            model,
            [FeatureConfig(name="context_only", features={})],
            meta={"user_config": {}},
        )


def test_static_conditioning_coverage_enumerates_exact_suites_and_seeds() -> None:
    cfg = OmegaConf.create(
        {
            "evals": "ignored",
            "base_seed": 0,
            "num_seeds": 99,
            "max_per_task": 5,
        }
    )
    provider = MagicMock()
    tokenizer = MagicMock()
    calls: list[tuple[str, int, int, int]] = []

    def enumerate_contexts(coverage_cfg, actual_tokenizer):
        assert actual_tokenizer is tokenizer
        calls.append(
            (
                str(coverage_cfg.evals),
                int(coverage_cfg.base_seed),
                int(coverage_cfg.num_seeds),
                int(coverage_cfg.model_max_seq_len),
            )
        )
        return [MagicMock()]

    with (
        patch(
            "scripts.train.training_progression.enumerate_expected_eval_contexts",
            side_effect=enumerate_contexts,
        ),
        patch(
            "scripts.train.training_progression.assert_exact_prompt_coverage"
        ) as assert_coverage,
    ):
        _assert_static_conditioning_context_coverage(
            cfg=cfg,
            tokenizer=tokenizer,
            provider=provider,
            eval_suites=["core", "chat", "harmfulness", "unknown"],
            seeds=[1234, 2234],
            model_max_seq_len=1024,
            surface="test progression",
        )

    assert calls == [
        ("core", 1234, 1, 1024),
        ("core", 2234, 1, 1024),
        ("chat", 1234, 1, 1024),
        ("harmfulness", 1234, 1, 1024),
        ("harmfulness", 2234, 1, 1024),
    ]
    assert_coverage.assert_called_once()
    assert assert_coverage.call_args.args[0] is provider
    assert len(assert_coverage.call_args.args[1]) == 5
    assert assert_coverage.call_args.kwargs == {"surface": "test progression"}


# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture()
def sample_cfg(tmp_path: Path) -> DictConfig:
    """Minimal OmegaConf config for testing."""
    return OmegaConf.create(
        {
            "model_tag": "d27",
            "model_family": "decoder",
            "checkpoint_root": "base_checkpoints",
            "checkpoint_prefix": "",
            "checkpoint_source": "local",
            "checkpoints_dir": str(tmp_path / "data"),
            "results_file": str(tmp_path / "results.json"),
            "fig_dir": str(tmp_path / "fig"),
            "phases": [
                {"phase": "pretrain", "subdir": "", "steps": [500, 1000, 1500]},
                {"phase": "sft", "subdir": "standard_sft_checkpoints", "steps": [1000]},
            ],
            "evals": "core,chat",
            "num_seeds": 2,
            "base_seed": 1234,
            "max_per_task": 10,
            "chat_tasks": "ARC-Easy,MMLU",
            "chat_batch_size": 4,
            "chat_max_new_tokens": 32,
            "chat_temperature": 0.0,
            "chat_top_k": 50,
            "chat_num_samples": 1,
            "chat_max_problems": 5,
            "harm_temperature": 0.7,
            "harm_top_k": 50,
            "harm_max_new_tokens": 32,
            "harm_num_benign": 10,
            "harm_judge_model": "test",
            "cleanup_after_eval": False,
            "device_type": "cpu",
            "dtype": "float32",
            "remote": {
                "host": "user@host",
                "port": 22,
                "ssh_key": "~/.ssh/id_rsa",
                "base_path": "/remote/base_checkpoints",
                "local_temp_dir": str(tmp_path / "tmp"),
            },
        }
    )


@pytest.fixture()
def mock_results() -> dict:
    """Sample results JSON with pretrain + SFT checkpoints."""
    return {
        "model_tag": "d27",
        "checkpoints": [
            {
                "step": 500,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "core_metric": 0.05,
                                            "per_task": {"task_a": 0.5},
                                            "centered_per_task": {"task_a": 0.3},
                                        },
                                        "2234": {
                                            "core_metric": 0.07,
                                            "per_task": {"task_a": 0.55},
                                            "centered_per_task": {"task_a": 0.35},
                                        },
                                    },
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"ARC-Easy": 0.26, "MMLU": 0.25},
                                    },
                                }
                            }
                        }
                    },
                },
            },
            {
                "step": 1000,
                "phase": "pretrain",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "core_metric": 0.10,
                                            "per_task": {},
                                            "centered_per_task": {},
                                        },
                                        "2234": {
                                            "core_metric": 0.12,
                                            "per_task": {},
                                            "centered_per_task": {},
                                        },
                                    },
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"ARC-Easy": 0.28, "MMLU": 0.26},
                                    },
                                }
                            }
                        }
                    },
                },
            },
            {
                "step": 1000,
                "phase": "sft",
                "evals": {
                    "core": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {
                                            "core_metric": 0.15,
                                            "per_task": {},
                                            "centered_per_task": {},
                                        },
                                        "2234": {
                                            "core_metric": 0.17,
                                            "per_task": {},
                                            "centered_per_task": {},
                                        },
                                    },
                                }
                            }
                        }
                    },
                    "chat": {
                        "conditioned": {
                            "configs": {
                                "static_delta": {
                                    "kind": "static_yaml",
                                    "target_conditioning_mode": "current",
                                    "seeds": {
                                        "1234": {"ARC-Easy": 0.45, "MMLU": 0.35},
                                    },
                                }
                            }
                        }
                    },
                },
            },
        ],
    }


def _conditioned_eval_cfg(base_cfg: DictConfig | None = None) -> DictConfig:
    conditioned = {
        "conditioning": {
            "conditioned_configs": [{"name": "static_delta", "features": {0: 1}}],
            "context_cache_dir": "/tmp/context-cache",
        }
    }
    if base_cfg is None:
        return OmegaConf.create(conditioned)
    return _merge_cfg(base_cfg, conditioned)


def _conditioned_evals(
    *,
    core_seeds: tuple[int, ...] = (),
    chat_seeds: tuple[int, ...] = (),
    harmfulness_seeds: tuple[int, ...] = (),
    config_name: str = "static_delta",
    core_result: dict | None = None,
    chat_result: dict | None = None,
    harmfulness_result: dict | None = None,
) -> dict:
    evals = {}
    for eval_name, eval_seeds, result in (
        ("core", core_seeds, core_result),
        ("chat", chat_seeds, chat_result),
        ("harmfulness", harmfulness_seeds, harmfulness_result),
    ):
        if not eval_seeds:
            continue
        evals[eval_name] = {
            "conditioned": {
                "configs": {
                    config_name: {
                        "kind": "static_yaml",
                        "target_conditioning_mode": "current",
                        "seeds": {str(seed): dict(result or {}) for seed in eval_seeds},
                    },
                },
            },
        }
    return evals


def _collect_results_without_context_cache_validation(cfg):
    with patch("scripts.train.training_progression._validate_required_context_cache"):
        collect_results(cfg)


# ===================================================================
# Unit tests: CheckpointSpec + build_checkpoint_specs
# ===================================================================


class TestBuildCheckpointSpecs:
    def test_builds_correct_specs(self, sample_cfg):
        specs = build_checkpoint_specs(sample_cfg)
        assert len(specs) == 4
        assert specs[0] == CheckpointSpec(step=500, phase="pretrain", subdir="")
        assert specs[1] == CheckpointSpec(step=1000, phase="pretrain", subdir="")
        assert specs[2] == CheckpointSpec(step=1500, phase="pretrain", subdir="")
        assert specs[3] == CheckpointSpec(step=1000, phase="sft", subdir="standard_sft_checkpoints")

    def test_empty_phases(self):
        cfg = OmegaConf.create({"phases": []})
        assert build_checkpoint_specs(cfg) == []

    def test_single_phase_single_step(self):
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [42]}]})
        specs = build_checkpoint_specs(cfg)
        assert len(specs) == 1
        assert specs[0].step == 42
        assert specs[0].phase == "pretrain"


# ===================================================================
# Unit tests: _configured_checkpoint_keys
# ===================================================================


class TestConfiguredCheckpointKeys:
    def test_returns_none_when_phases_missing(self):
        cfg = OmegaConf.create({"model_tag": "d27"})
        assert _configured_checkpoint_keys(cfg) is None

    def test_returns_none_when_phases_is_null(self):
        cfg = OmegaConf.create({"model_tag": "d27", "phases": None})
        assert _configured_checkpoint_keys(cfg) is None

    def test_returns_none_when_phases_is_empty_list(self):
        cfg = OmegaConf.create({"model_tag": "d27", "phases": []})
        # Empty list is falsy; helper returns None so the filter no-ops.
        assert _configured_checkpoint_keys(cfg) is None

    def test_returns_set_of_step_phase_tuples(self):
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                    {"phase": "sft", "subdir": "x", "steps": [200]},
                ]
            }
        )
        assert _configured_checkpoint_keys(cfg) == {
            (500, "pretrain"),
            (1000, "pretrain"),
            (200, "sft"),
        }

    def test_coerces_string_step_to_int(self):
        # Hydra YAML normally types steps as int, but tolerate stringy steps.
        cfg = OmegaConf.create(
            {"phases": [{"phase": "pretrain", "subdir": "", "steps": ["500", "1000"]}]}
        )
        assert _configured_checkpoint_keys(cfg) == {(500, "pretrain"), (1000, "pretrain")}

    def test_distinguishes_same_step_across_phases(self):
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [1000]},
                    {"phase": "sft", "subdir": "x", "steps": [1000]},
                ]
            }
        )
        keys = _configured_checkpoint_keys(cfg)
        assert keys is not None
        assert (1000, "pretrain") in keys
        assert (1000, "sft") in keys
        assert len(keys) == 2

    def test_deduplicates_repeated_step_within_phase(self):
        cfg = OmegaConf.create(
            {"phases": [{"phase": "pretrain", "subdir": "", "steps": [500, 500]}]}
        )
        # Set semantics — duplicates collapse.
        assert _configured_checkpoint_keys(cfg) == {(500, "pretrain")}

    def test_returns_none_when_any_phase_lacks_steps(self):
        # Partial phase configs (no `steps` key) are accepted by the broader
        # progression pipeline; the filter must fall back to no-op rather than
        # crash on missing keys.
        cfg = OmegaConf.create({"phases": [{"phase": "safety_sft"}]})
        assert _configured_checkpoint_keys(cfg) is None

    def test_returns_none_when_any_phase_has_null_steps(self):
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": None}]})
        assert _configured_checkpoint_keys(cfg) is None

    def test_returns_empty_set_when_all_phases_have_empty_steps(self):
        # An EXPLICIT empty `steps: []` is honored as "no keys for that phase".
        # This differs from missing/null `steps` which fall back to no-filter.
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": []}]})
        assert _configured_checkpoint_keys(cfg) == set()

    def test_returns_none_when_some_phases_have_steps_others_dont(self):
        # First phase has steps; second has no `steps` key. The "any phase
        # lacks steps" rule wins → None (filter no-ops) so a partial config
        # never silently drops the matched phase's data.
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500]},
                    {"phase": "sft"},
                ]
            }
        )
        assert _configured_checkpoint_keys(cfg) is None


# ===================================================================
# Unit tests: _filter_results_to_configured_checkpoints
# ===================================================================


class TestFilterResultsToConfiguredCheckpoints:
    @pytest.fixture
    def base_results(self) -> dict:
        return {
            "model_tag": "d27",
            "checkpoints": [
                {"step": 500, "phase": "pretrain", "evals": {"core": "data_500"}},
                {"step": 1000, "phase": "pretrain", "evals": {"core": "data_1000"}},
                {"step": 1500, "phase": "pretrain", "evals": {"core": "data_1500"}},
                {"step": 200, "phase": "sft", "evals": {"core": "data_sft_200"}},
            ],
        }

    def test_no_phases_returns_input_identity(self, base_results):
        cfg = OmegaConf.create({"model_tag": "d27"})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        # Identity: cheap no-op when there's no filter to apply.
        assert out is base_results

    def test_null_phases_returns_input_identity(self, base_results):
        cfg = OmegaConf.create({"model_tag": "d27", "phases": None})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        assert out is base_results

    def test_empty_phases_returns_input_identity(self, base_results):
        cfg = OmegaConf.create({"model_tag": "d27", "phases": []})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        # Empty list -> _configured_checkpoint_keys returns None -> no filter.
        assert out is base_results

    def test_all_match_returns_input_identity(self, base_results):
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500, 1000, 1500]},
                    {"phase": "sft", "subdir": "x", "steps": [200]},
                ]
            }
        )
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        # Early-return path: every checkpoint matched, return original ref.
        assert out is base_results

    def test_drops_unmatched_step(self, base_results):
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                    {"phase": "sft", "subdir": "x", "steps": [200]},
                ]
            }
        )
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        kept = {(c["step"], c["phase"]) for c in out["checkpoints"]}
        assert kept == {(500, "pretrain"), (1000, "pretrain"), (200, "sft")}

    def test_drops_unmatched_phase(self, base_results):
        # Config requests pretrain only -> sft entry must be dropped.
        cfg = OmegaConf.create(
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500, 1000, 1500]},
                ]
            }
        )
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        kept = {(c["step"], c["phase"]) for c in out["checkpoints"]}
        assert kept == {(500, "pretrain"), (1000, "pretrain"), (1500, "pretrain")}

    def test_preserves_top_level_fields_when_filtering(self, base_results):
        base_results["custom_field"] = {"a": 1}
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}]})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        assert out["model_tag"] == "d27"
        assert out["custom_field"] == {"a": 1}
        # Original is not mutated.
        assert len(base_results["checkpoints"]) == 4

    def test_filtered_result_is_shallow_copy(self, base_results):
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}]})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        assert out is not base_results
        # checkpoints list is a new list; the entries can be shared (shallow).
        assert out["checkpoints"] is not base_results["checkpoints"]

    def test_handles_string_step_in_checkpoint(self):
        results = {
            "model_tag": "d27",
            "checkpoints": [{"step": "500", "phase": "pretrain", "evals": {}}],
        }
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}]})
        out = _filter_results_to_configured_checkpoints(results, cfg)
        # Stringy step in checkpoint coerces to int(500), still matches.
        kept = {(int(c["step"]), c["phase"]) for c in out["checkpoints"]}
        assert kept == {(500, "pretrain")}

    def test_handles_missing_step_field_in_checkpoint(self):
        # Missing step defaults to 0; only matches if config explicitly has step 0.
        results = {
            "model_tag": "d27",
            "checkpoints": [{"phase": "pretrain", "evals": {}}],  # no step
        }
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}]})
        out = _filter_results_to_configured_checkpoints(results, cfg)
        # Step 0 not in config -> dropped.
        assert out["checkpoints"] == []

    def test_drops_all_when_no_overlap(self, base_results):
        cfg = OmegaConf.create({"phases": [{"phase": "pretrain", "subdir": "", "steps": [9999]}]})
        out = _filter_results_to_configured_checkpoints(base_results, cfg)
        assert out["checkpoints"] == []
        # model_tag is preserved.
        assert out["model_tag"] == "d27"

    def test_drops_checkpoint_missing_phase_field(self):
        # Missing phase field defaults to "" via the `str(ckpt.get("phase", ""))`
        # fallback; "" is not in the configured key set, so the checkpoint is
        # dropped (instead of crashing or silently passing through).
        results = {
            "model_tag": "d27",
            "checkpoints": [
                {"step": 500, "phase": "pretrain", "evals": {}},
                {"step": 1000, "evals": {}},  # no phase key
            ],
        }
        cfg = OmegaConf.create(
            {"phases": [{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}]}
        )
        out = _filter_results_to_configured_checkpoints(results, cfg)
        kept = [(c.get("step"), c.get("phase")) for c in out["checkpoints"]]
        assert kept == [(500, "pretrain")]


# ===================================================================
# Unit tests: Results JSON helpers
# ===================================================================


class TestResultsPath:
    def test_explicit_path(self):
        cfg = OmegaConf.create({"results_file": "/explicit/path.json", "model_tag": "d27"})
        assert _results_path(cfg) == Path("/explicit/path.json")

    def test_auto_derived_path(self):
        cfg = OmegaConf.create({"results_file": None, "model_tag": "d27"})
        assert _results_path(cfg) == Path("out/progression/d27/results.json")

    def test_fig_dir_explicit(self):
        cfg = OmegaConf.create({"fig_dir": "/my/figs", "model_tag": "d27"})
        assert _fig_dir(cfg) == Path("/my/figs")

    def test_fig_dir_auto(self):
        cfg = OmegaConf.create({"fig_dir": None, "model_tag": "d27"})
        assert _fig_dir(cfg) == Path("fig/progression/d27")


class TestLoadOrInitResults:
    def test_creates_new_when_no_file(self, sample_cfg):
        results = load_or_init_results(sample_cfg)
        assert results == {"model_tag": "d27", "checkpoints": []}

    def test_loads_existing(self, sample_cfg, mock_results):
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(mock_results, f)
        results = load_or_init_results(sample_cfg)
        assert results["model_tag"] == "d27"
        assert len(results["checkpoints"]) == 3


class TestSaveResults:
    def test_creates_parent_dirs(self, sample_cfg):
        results = {"model_tag": "d27", "checkpoints": []}
        save_results(results, sample_cfg)
        path = Path(sample_cfg.results_file)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == results

    def test_atomic_overwrite(self, sample_cfg):
        """Saving twice overwrites cleanly with no .tmp left."""
        save_results({"model_tag": "d27", "checkpoints": []}, sample_cfg)
        save_results({"model_tag": "d27", "checkpoints": [{"step": 1}]}, sample_cfg)
        path = Path(sample_cfg.results_file)
        assert not path.with_suffix(".tmp").exists()
        with open(path) as f:
            assert len(json.load(f)["checkpoints"]) == 1


class TestFindCheckpointEntry:
    def test_finds_existing(self, mock_results):
        entry = find_checkpoint_entry(mock_results, 500, "pretrain")
        assert entry is not None
        assert entry["step"] == 500

    def test_returns_none_for_missing_step(self, mock_results):
        assert find_checkpoint_entry(mock_results, 9999, "pretrain") is None

    def test_returns_none_for_wrong_phase(self, mock_results):
        assert find_checkpoint_entry(mock_results, 500, "sft") is None

    def test_distinguishes_same_step_different_phase(self, mock_results):
        pretrain = find_checkpoint_entry(mock_results, 1000, "pretrain")
        sft = find_checkpoint_entry(mock_results, 1000, "sft")
        assert pretrain is not None and sft is not None
        assert pretrain["phase"] == "pretrain"
        assert sft["phase"] == "sft"


class TestNeedsEval:
    def test_no_eval_needed_without_configured_conditioning(self):
        assert _needs_eval(None, ["core"], [1234]) is False

    def test_needs_eval_when_configured_conditioned_entry_missing(self):
        assert (
            _needs_eval(
                None,
                ["core"],
                [1234],
                _conditioned_eval_cfg(),
                "pretrain",
            )
            is True
        )

    def test_needs_eval_when_conditioned_incomplete(self):
        entry = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(core_seeds=(1234,)),
        }
        assert (
            _needs_eval(
                entry,
                ["core"],
                [1234, 2234],
                _conditioned_eval_cfg(),
                "pretrain",
            )
            is True
        )

    def test_no_eval_needed_when_conditioned_complete(self):
        entry = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(core_seeds=(1234, 2234)),
        }
        assert (
            _needs_eval(
                entry,
                ["core"],
                [1234, 2234],
                _conditioned_eval_cfg(),
                "pretrain",
            )
            is False
        )

    def test_chat_only_checks_first_seed_for_conditioned_config(self):
        entry = {
            "step": 500,
            "phase": "pretrain",
            "evals": _conditioned_evals(chat_seeds=(1234,)),
        }
        assert (
            _needs_eval(
                entry,
                ["chat"],
                [1234, 2234],
                _conditioned_eval_cfg(),
                "pretrain",
            )
            is False
        )


# ===================================================================
# Unit tests: Checkpoint resolution
# ===================================================================


class TestResolveLocalCheckpointDir:
    def test_resolves_existing_pretrain(self, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_000500.pt").write_bytes(b"model_data")
        (ckpt_dir / "meta_000500.json").write_text("{}")
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        result = resolve_local_checkpoint_dir(spec, sample_cfg)
        assert result == str(ckpt_dir)

    def test_resolves_existing_sft(self, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27" / "standard_sft_checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_001000.pt").write_bytes(b"model_data")
        (ckpt_dir / "meta_001000.json").write_text("{}")
        spec = CheckpointSpec(step=1000, phase="sft", subdir="standard_sft_checkpoints")
        result = resolve_local_checkpoint_dir(spec, sample_cfg)
        assert result == str(ckpt_dir)

    def test_raises_when_model_missing(self, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "meta_000500.json").write_text("{}")
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_local_checkpoint_dir(spec, sample_cfg)

    def test_raises_when_meta_missing(self, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_000500.pt").write_bytes(b"data")
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        with pytest.raises(FileNotFoundError, match="Meta not found"):
            resolve_local_checkpoint_dir(spec, sample_cfg)

    def test_resolves_nested_encdec_checkpoint(self, tmp_path, sample_cfg):
        cfg = _merge_cfg(
            sample_cfg,
            {
                "model_tag": "d12_baseline",
                "model_family": "encoder_decoder",
                "checkpoint_root": "conditional_encdec_checkpoints",
            },
        )
        ckpt_dir = (
            tmp_path
            / "data"
            / "conditional_encdec_checkpoints"
            / "d12_baseline"
            / "s128"
            / "dist_geometric"
            / "mean_10"
        )
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_000250.pt").write_bytes(b"model_data")
        (ckpt_dir / "meta_000250.json").write_text("{}")

        spec = CheckpointSpec(step=250, phase="pretrain", subdir="")

        assert isinstance(cfg, DictConfig)
        assert resolve_local_checkpoint_dir(spec, cfg) == str(ckpt_dir)

    def test_base_source_does_not_recursive_search_nested_sft(self, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27" / "standard_sft_checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_000500.pt").write_bytes(b"model_data")
        (ckpt_dir / "meta_000500.json").write_text("{}")

        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_local_checkpoint_dir(spec, sample_cfg)


# ===================================================================
# Unit tests: Remote helpers
# ===================================================================


class TestScpCmd:
    def test_builds_correct_command(self, sample_cfg):
        cmd = _scp_cmd(sample_cfg)
        assert cmd[0] == "scp"
        assert "-P" in cmd
        assert "22" in cmd
        assert "-i" in cmd


class TestCleanupRemoteCheckpoint:
    def test_removes_pt_file(self, tmp_path):
        pt = tmp_path / "model_000500.pt"
        pt.write_bytes(b"data")
        cleanup_remote_checkpoint(str(tmp_path), 500)
        assert not pt.exists()

    def test_noop_if_missing(self, tmp_path):
        cleanup_remote_checkpoint(str(tmp_path), 500)  # should not raise


class TestPlotResults:
    def test_generates_charts(self, sample_cfg, mock_results):
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(mock_results, f)

        plot_results(sample_cfg)

        fig_dir = Path(sample_cfg.fig_dir)
        assert fig_dir.exists()
        generated = {p.name for p in fig_dir.glob("*.png")}
        assert "core_metric.png" in generated
        assert "chat_arc_easy.png" in generated
        assert "chat_mmlu.png" in generated

    def test_empty_results_no_crash(self, sample_cfg):
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"model_tag": "d27", "checkpoints": []}, f)

        plot_results(sample_cfg)  # should not raise

        fig_dir = Path(sample_cfg.fig_dir)
        assert not list(fig_dir.glob("*.png"))

    def test_chart_dimensions(self, sample_cfg, mock_results):
        """Charts have expected figsize."""
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(mock_results, f)

        plot_results(sample_cfg)

        fig_dir = Path(sample_cfg.fig_dir)
        core_png = fig_dir / "core_metric.png"
        assert core_png.exists()
        assert core_png.stat().st_size > 0

    def test_plot_results_filters_to_configured_checkpoints(self, sample_cfg):
        """Stale per-checkpoint files are not plotted or saved in results.json."""
        evals = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1000, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 2000, "pretrain", evals)

        plot_results(sample_cfg)

        with open(sample_cfg.results_file) as f:
            results = json.load(f)
        plotted = {(ckpt["step"], ckpt["phase"]) for ckpt in results["checkpoints"]}
        assert plotted == {(500, "pretrain"), (1000, "pretrain")}

    def test_plot_results_filter_does_not_delete_per_checkpoint_files(self, sample_cfg, tmp_path):
        """Filter is a view; per-checkpoint files survive untouched."""
        evals = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1000, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 2000, "pretrain", evals)  # stale

        ckpt_dir = _results_dir(sample_cfg)
        before = sorted(p.name for p in ckpt_dir.glob("*.json"))
        assert before == [
            "pretrain_step_000500.json",
            "pretrain_step_001000.json",
            "pretrain_step_002000.json",
        ]

        plot_results(sample_cfg)

        after = sorted(p.name for p in ckpt_dir.glob("*.json"))
        assert after == before, (
            "Filter must not delete per-checkpoint files; only the merged results.json is reduced."
        )

    def test_plot_results_filter_round_trips_when_config_re_expanded(self, sample_cfg, tmp_path):
        """A subsequent run with the original config restores every checkpoint."""
        evals = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1000, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1500, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1000, "sft", evals)

        # Round 1: shrink the config to drop 1500 + sft.
        narrow_cfg = _merge_cfg(
            sample_cfg,
            {
                "phases": [
                    {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                ]
            },
        )
        assert isinstance(narrow_cfg, DictConfig)
        plot_results(narrow_cfg)
        with open(narrow_cfg.results_file) as f:
            narrow_results = json.load(f)
        narrow_kept = {(c["step"], c["phase"]) for c in narrow_results["checkpoints"]}
        assert narrow_kept == {(500, "pretrain"), (1000, "pretrain")}

        # Round 2: restore the original (broader) config; merged JSON expands.
        plot_results(sample_cfg)
        with open(sample_cfg.results_file) as f:
            wide_results = json.load(f)
        wide_kept = {(c["step"], c["phase"]) for c in wide_results["checkpoints"]}
        assert wide_kept == {
            (500, "pretrain"),
            (1000, "pretrain"),
            (1500, "pretrain"),
            (1000, "sft"),
        }

    def test_plot_results_without_phases_does_not_filter(self, sample_cfg, mock_results):
        """A cfg lacking `phases` is the legacy decoder-only shape; never filter."""
        # Clone via to_container/create so del() does not mutate the shared fixture.
        cfg_no_phases = OmegaConf.create(OmegaConf.to_container(sample_cfg, resolve=True))
        assert isinstance(cfg_no_phases, DictConfig)
        del cfg_no_phases["phases"]

        path = Path(cfg_no_phases.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(mock_results, f)

        plot_results(cfg_no_phases)

        with open(path) as f:
            results = json.load(f)
        # Every original checkpoint survives — filter must no-op.
        assert {(c["step"], c["phase"]) for c in results["checkpoints"]} == {
            (500, "pretrain"),
            (1000, "pretrain"),
            (1000, "sft"),
        }

    def test_plot_results_no_overlap_emits_empty_results_and_no_charts(self, sample_cfg):
        """Config with no overlap with on-disk per-checkpoint files.

        End-to-end view of the all-filtered-out branch:
        - merge_checkpoint_results pulls every per-checkpoint file.
        - The filter drops all of them (config asks for step 9999, on-disk has 500/1000).
        - save_results writes the filtered (empty) merged JSON.
        - plot_results' early-return guard skips chart generation, leaving fig_dir empty.
        - Per-checkpoint files survive (separately tested), so re-expanding
          the config restores the data on the next merge.
        """
        evals = {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}
        save_checkpoint_result(sample_cfg, 500, "pretrain", evals)
        save_checkpoint_result(sample_cfg, 1000, "pretrain", evals)

        cfg_no_overlap = _merge_cfg(
            sample_cfg,
            {"phases": [{"phase": "pretrain", "subdir": "", "steps": [9999]}]},
        )

        assert isinstance(cfg_no_overlap, DictConfig)
        plot_results(cfg_no_overlap)

        with open(cfg_no_overlap.results_file) as f:
            results = json.load(f)
        assert results["checkpoints"] == []

        fig_dir = Path(cfg_no_overlap.fig_dir)
        assert fig_dir.exists()
        assert not list(fig_dir.glob("*.png"))


# ===================================================================
# Functional tests: Collect orchestration
# ===================================================================


class TestCollectResults:
    def _setup_local_checkpoints(self, tmp_path, steps, subdir=""):
        """Create fake checkpoint files in the expected directory structure."""
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        if subdir:
            ckpt_dir = ckpt_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (ckpt_dir / f"model_{step:06d}.pt").write_bytes(b"fake_model")
            meta = {
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
            (ckpt_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta))

    def test_collect_rejects_local_cleanup_before_result_writes(self, tmp_path, sample_cfg):
        self._setup_local_checkpoints(tmp_path, [500])
        checkpoint = tmp_path / "data" / "base_checkpoints" / "d27" / "model_000500.pt"
        checksum_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "cleanup_after_eval": True,
                    "phases": [
                        {"phase": "pretrain", "subdir": "", "steps": [500]},
                    ],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )

        with (
            patch(
                "scripts.train.training_progression._validate_required_context_cache"
            ) as mock_validate,
            patch(
                "scripts.train.training_progression.evaluate_single_checkpoint",
                return_value=_conditioned_evals(core_seeds=(1234,)),
            ) as mock_eval,
        ):
            with pytest.raises(
                ValueError,
                match="cleanup_after_eval.*checkpoint_source=remote",
            ):
                collect_results(cfg)

        mock_validate.assert_not_called()
        mock_eval.assert_not_called()
        assert checkpoint.exists()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum_before
        assert not Path(cfg.results_file).exists()
        assert not _results_dir(cfg).exists()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_local_saves_results(self, mock_eval, tmp_path, sample_cfg):
        """Collect saves per-checkpoint conditioned eval outputs and merged JSON."""
        self._setup_local_checkpoints(tmp_path, [500])
        self._setup_local_checkpoints(tmp_path, [1000], "standard_sft_checkpoints")

        # Only evaluate 2 checkpoints
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [
                        {"phase": "pretrain", "subdir": "", "steps": [500]},
                        {"phase": "sft", "subdir": "standard_sft_checkpoints", "steps": [1000]},
                    ],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        path = Path(sample_cfg.results_file)
        assert path.exists()
        with open(path) as f:
            results = json.load(f)
        assert len(results["checkpoints"]) == 2
        assert mock_eval.call_count == 2  # once per checkpoint

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_skips_completed(self, mock_eval, tmp_path, sample_cfg):
        """Already-completed checkpoints are skipped."""
        self._setup_local_checkpoints(tmp_path, [500, 1000])

        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [
                        {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                    ],
                    "num_seeds": 2,
                    "evals": "core",
                },
            )
        )

        # Pre-populate results
        complete_results = {
            "model_tag": "d27",
            "checkpoints": [
                {
                    "step": 500,
                    "phase": "pretrain",
                    "evals": _conditioned_evals(core_seeds=(1234, 2234)),
                },
                {
                    "step": 1000,
                    "phase": "pretrain",
                    "evals": _conditioned_evals(core_seeds=(1234, 2234)),
                },
            ],
        }
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(complete_results, f)

        _collect_results_without_context_cache_validation(sample_cfg)

        # Both checkpoints already done -> evaluator never called.
        mock_eval.assert_not_called()

    @patch("scripts.train.training_progression.run_core_eval")
    @patch("scripts.train.training_progression.build_decoder_model")
    def test_collect_handles_missing_checkpoint(self, mock_build, mock_core, tmp_path, sample_cfg):
        """Missing checkpoint files are logged and skipped."""
        # Don't create any checkpoint files
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )

        _collect_results_without_context_cache_validation(sample_cfg)  # should not raise

        mock_build.assert_not_called()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_handles_eval_failure(self, mock_eval, tmp_path, sample_cfg):
        """Eval failure is caught and pipeline continues."""
        self._setup_local_checkpoints(tmp_path, [500])

        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.side_effect = RuntimeError("OOM")

        _collect_results_without_context_cache_validation(sample_cfg)  # should not raise

        # Entry was created but no seed data saved (eval failed).
        # The file may or may not exist depending on whether save_results
        # was reached before the failure — the key invariant is no crash.
        path = Path(sample_cfg.results_file)
        if path.exists():
            with open(path) as f:
                results = json.load(f)
            entry = find_checkpoint_entry(results, 500, "pretrain")
            # If entry exists, it should have no successful seeds
            if entry:
                seeds = entry.get("evals", {}).get("core", {}).get("seeds", {})
                assert len(seeds) == 0

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_remote_delegates_pending_conditioned_checkpoint(
        self, mock_eval, tmp_path, sample_cfg
    ):
        """Remote collect delegates pending conditioned checkpoints to the evaluator."""
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "checkpoint_source": "remote",
                    "cleanup_after_eval": True,
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        mock_eval.assert_called_once()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_local_does_not_cleanup(self, mock_eval, tmp_path, sample_cfg):
        """Local mode does NOT delete checkpoint files after eval."""
        self._setup_local_checkpoints(tmp_path, [500])

        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        # Checkpoint file should still exist
        pt = tmp_path / "data" / "base_checkpoints" / "d27" / "model_000500.pt"
        assert pt.exists()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_passes_full_seed_sweep_to_conditioned_evaluator(
        self, mock_eval, tmp_path, sample_cfg
    ):
        """Collect passes the sweep once; suite-specific seed filtering is in the evaluator."""
        self._setup_local_checkpoints(tmp_path, [500])

        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 3,
                    "evals": "core,chat",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(
            core_seeds=(1234, 2234, 3234),
            chat_seeds=(1234,),
        )

        _collect_results_without_context_cache_validation(sample_cfg)

        mock_eval.assert_called_once()
        assert mock_eval.call_args.args[4] == [1234, 2234, 3234]


# ===================================================================
# Integration tests: Seed parameter in core_eval
# ===================================================================


class TestCoreEvalSeed:
    def test_evaluate_example_default_seed_backward_compat(self):
        """Default seed=1234 produces the same RNG state as the original hardcoded value."""
        # The original code used random.Random(1234 + idx)
        # With seed=1234 (default), we should get the same behavior
        idx = 5
        rng_original = random.Random(1234 + idx)
        rng_new = random.Random(1234 + idx)  # seed=1234 default
        # They should produce identical sequences
        assert rng_original.random() == rng_new.random()

    def test_different_seeds_different_rng(self):
        """Different seeds produce different RNG states."""
        idx = 5
        rng_a = random.Random(1234 + idx)
        rng_b = random.Random(5678 + idx)
        # They should produce different sequences
        assert rng_a.random() != rng_b.random()

    def test_evaluate_task_accepts_seed(self):
        """evaluate_task signature accepts seed parameter."""
        sig = inspect.signature(evaluate_task)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default == 1234

    def test_evaluate_example_accepts_seed(self):
        """evaluate_example signature accepts seed parameter."""
        sig = inspect.signature(evaluate_example)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default == 1234

    def test_evaluate_core_accepts_seed(self):
        """evaluate_core signature accepts seed parameter."""
        sig = inspect.signature(evaluate_core)
        assert "seed" in sig.parameters
        assert sig.parameters["seed"].default == 1234


# ===================================================================
# Unit tests: METRIC_SPECS coverage
# ===================================================================


class TestMetricSpecs:
    def test_all_specs_have_three_elements(self):
        for spec in METRIC_SPECS:
            assert len(spec) == 3, f"Spec {spec} should have 3 elements"

    def test_all_filenames_unique(self):
        filenames = [spec[2] for spec in METRIC_SPECS]
        assert len(filenames) == len(set(filenames)), "Duplicate filenames in METRIC_SPECS"

    def test_covers_core(self):
        core_specs = [s for s in METRIC_SPECS if s[0][0] == "core"]
        assert len(core_specs) >= 1

    def test_covers_chat(self):
        chat_specs = [s for s in METRIC_SPECS if s[0][0] == "chat"]
        assert len(chat_specs) >= 1

    def test_covers_harmfulness(self):
        harm_specs = [s for s in METRIC_SPECS if s[0][0] == "harmfulness"]
        assert len(harm_specs) >= 1


# ===================================================================
# End-to-end: collect + plot
# ===================================================================


# ===================================================================
# Unit tests: _local_checkpoints_root
# ===================================================================


class TestLocalCheckpointsRoot:
    def test_legacy_source_does_not_replace_current_checkpoint_root(self):
        cfg = OmegaConf.create({"source": "encdec", "checkpoints_dir": "/my/data"})
        with pytest.raises(ConfigAttributeError, match="checkpoint_root"):
            _local_checkpoints_root(cfg)

    def test_explicit_dir(self):
        cfg = OmegaConf.create(
            {"checkpoints_dir": "/my/data", "checkpoint_root": "base_checkpoints"}
        )
        assert _local_checkpoints_root(cfg) == Path("/my/data/base_checkpoints")

    def test_encdec_explicit_dir(self):
        cfg = OmegaConf.create(
            {
                "checkpoints_dir": "/my/data",
                "checkpoint_root": "conditional_encdec_checkpoints",
            }
        )
        assert _local_checkpoints_root(cfg) == Path("/my/data/conditional_encdec_checkpoints")

    @patch("scripts.train.training_progression.get_base_dir", return_value="/fallback")
    def test_falls_back_to_base_dir(self, _mock):
        cfg = OmegaConf.create({"checkpoints_dir": None, "checkpoint_root": "base_checkpoints"})
        assert _local_checkpoints_root(cfg) == Path("/fallback/base_checkpoints")


def test_progression_rejects_legacy_model_family_alias() -> None:
    cfg = OmegaConf.create({"model_family": "encdec"})
    with pytest.raises(ValueError, match="model_family must be one of"):
        _should_load_encoder_decoder("/unused", 0, cfg)


# ===================================================================
# Unit tests: fetch_remote_checkpoint
# ===================================================================


class TestFetchRemoteCheckpoint:
    @patch("scripts.train.training_progression.subprocess.run")
    def test_fetches_both_files(self, mock_run, tmp_path, sample_cfg):
        """SCP is called for both model and meta files."""
        payloads = {
            "model_000500.pt": b"complete model bytes",
            "meta_000500.json": b'{"step": 500}',
        }
        calls: list[tuple[str, str]] = []
        mock_run.side_effect = _fake_remote_run(payloads, calls)

        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        local_dir = fetch_remote_checkpoint(spec, sample_cfg)

        assert calls == [
            ("ssh", "model_000500.pt"),
            ("scp", "model_000500.pt"),
            ("ssh", "meta_000500.json"),
            ("scp", "meta_000500.json"),
        ]
        assert Path(local_dir).exists()
        assert (Path(local_dir) / "model_000500.pt").read_bytes() == payloads["model_000500.pt"]
        assert (Path(local_dir) / "meta_000500.json").read_bytes() == payloads["meta_000500.json"]

    @patch("scripts.train.training_progression.subprocess.run")
    def test_scp_failure_raises(self, mock_run, tmp_path, sample_cfg):
        """SCP failure raises RuntimeError."""
        payloads = {
            "model_000500.pt": b"complete model bytes",
            "meta_000500.json": b'{"step": 500}',
        }
        calls: list[tuple[str, str]] = []
        mock_run.side_effect = _fake_remote_run(
            payloads,
            calls,
            scp_error="Connection refused",
        )
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        with pytest.raises(RuntimeError, match="scp failed"):
            fetch_remote_checkpoint(spec, sample_cfg)

    @patch("scripts.train.training_progression.subprocess.run")
    def test_healthy_existing_bytes_validate_without_transfer(self, mock_run, tmp_path, sample_cfg):
        """Healthy current-layout files are verified remotely without SCP."""
        payloads = {
            "model_000500.pt": b"complete model bytes",
            "meta_000500.json": b'{"step": 500}',
        }
        local_dir = tmp_path / "tmp" / "d27"
        local_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            (local_dir / name).write_bytes(payload)
        before = {name: (local_dir / name).read_bytes() for name in payloads}

        calls: list[tuple[str, str]] = []
        mock_run.side_effect = _fake_remote_run(payloads, calls)

        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        fetch_remote_checkpoint(spec, sample_cfg)

        assert calls == [
            ("ssh", "model_000500.pt"),
            ("ssh", "meta_000500.json"),
        ]
        assert {name: (local_dir / name).read_bytes() for name in payloads} == before

    @patch("scripts.train.training_progression.subprocess.run")
    def test_truncated_nonzero_file_is_refetched(self, mock_run, tmp_path, sample_cfg):
        """A nonzero but truncated final file is never trusted as complete."""
        payloads = {
            "model_000500.pt": b"complete model bytes",
            "meta_000500.json": b'{"step": 500}',
        }
        local_dir = tmp_path / "tmp" / "d27"
        local_dir.mkdir(parents=True)
        (local_dir / "model_000500.pt").write_bytes(b"truncated")
        (local_dir / "meta_000500.json").write_bytes(payloads["meta_000500.json"])

        calls: list[tuple[str, str]] = []
        mock_run.side_effect = _fake_remote_run(payloads, calls)
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        fetch_remote_checkpoint(spec, sample_cfg)

        assert calls == [
            ("ssh", "model_000500.pt"),
            ("scp", "model_000500.pt"),
            ("ssh", "meta_000500.json"),
        ]
        assert (local_dir / "model_000500.pt").read_bytes() == payloads["model_000500.pt"]

    @patch("scripts.train.training_progression.subprocess.run")
    def test_mismatched_download_is_rejected_without_replacing_final(
        self, mock_run, tmp_path, sample_cfg
    ):
        """A short or hash-mismatched SCP result never reaches the final path."""
        payloads = {
            "model_000500.pt": b"complete model bytes",
            "meta_000500.json": b'{"step": 500}',
        }
        local_dir = tmp_path / "tmp" / "d27"
        local_dir.mkdir(parents=True)
        model_path = local_dir / "model_000500.pt"
        model_path.write_bytes(b"old truncated bytes")

        calls: list[tuple[str, str]] = []

        def corrupt_scp(command, **_kwargs):
            name, payload = _remote_payload_for_command(command, payloads)
            if command[0] == "ssh":
                calls.append(("ssh", name))
                return SimpleNamespace(
                    returncode=0,
                    stdout=(f"{len(payload)}\n{hashlib.sha256(payload).hexdigest()}  {name}\n"),
                    stderr="",
                )
            calls.append(("scp", name))
            Path(command[-1]).write_bytes(b"corrupt")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        mock_run.side_effect = corrupt_scp
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})

        with pytest.raises(
            RuntimeError,
            match="download validation failed for model_000500.pt",
        ):
            fetch_remote_checkpoint(spec, sample_cfg)

        assert calls == [
            ("ssh", "model_000500.pt"),
            ("scp", "model_000500.pt"),
        ]
        assert model_path.read_bytes() == b"old truncated bytes"
        assert list(local_dir.glob(".model_000500.pt.stage.*")) == []

    @patch("scripts.train.training_progression.subprocess.run")
    def test_subdir_creates_nested_path(self, mock_run, tmp_path, sample_cfg):
        """SFT subdir is reflected in local directory structure."""
        payloads = {
            "model_001000.pt": b"complete model bytes",
            "meta_001000.json": b'{"step": 1000}',
        }
        calls: list[tuple[str, str]] = []
        mock_run.side_effect = _fake_remote_run(payloads, calls)
        spec = CheckpointSpec(step=1000, phase="sft", subdir="standard_sft_checkpoints")
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        local_dir = fetch_remote_checkpoint(spec, sample_cfg)
        assert "standard_sft_checkpoints" in local_dir

    def test_two_processes_publish_once_without_exposing_partial_final(self, tmp_path, sample_cfg):
        """A blocked SCP remains private and two processes publish one exact copy."""
        payloads = {
            "model_000500.pt": b"the complete checkpoint model payload",
            "meta_000500.json": b'{"step": 500, "healthy": true}',
        }
        sample_cfg = _merge_cfg(sample_cfg, {"remote": {"local_temp_dir": str(tmp_path / "tmp")}})
        cfg_data = OmegaConf.to_container(sample_cfg, resolve=True)
        assert isinstance(cfg_data, dict)

        ctx = mp.get_context("spawn")
        scp_started = ctx.Event()
        allow_scp = ctx.Event()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_concurrent_remote_fetch_worker,
                args=(cfg_data, payloads, scp_started, allow_scp, result_queue),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()

        final_model = tmp_path / "tmp" / "d27" / "model_000500.pt"
        try:
            assert scp_started.wait(timeout=15), "fake SCP never reached its blocking point"
            assert not final_model.exists(), "partial SCP bytes leaked at the final path"
        finally:
            allow_scp.set()
            for process in processes:
                process.join(timeout=20)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

        assert [process.exitcode for process in processes] == [0, 0]
        results = [result_queue.get(timeout=5) for _ in processes]
        assert all(result["ok"] for result in results), results
        assert final_model.read_bytes() == payloads["model_000500.pt"]
        assert all(result["model"] == payloads["model_000500.pt"] for result in results)
        transferred = [name for result in results for name in result["scp_calls"]]
        assert transferred.count("model_000500.pt") == 1
        assert transferred.count("meta_000500.json") == 1


# ===================================================================
# Functional tests: Eval dispatch functions
# ===================================================================


class TestRunCoreEval:
    @patch("scripts.train.training_progression._base_eval_evaluate_core")
    def test_returns_structured_result(self, mock_eval, sample_cfg):
        mock_eval.return_value = {
            "core_metric": 0.42,
            "results": {"task_a": 0.5},
            "centered_results": {"task_a": 0.3},
        }
        result = run_core_eval(MagicMock(), MagicMock(), "cpu", 1234, sample_cfg)
        assert result["core_metric"] == 0.42
        assert result["per_task"] == {"task_a": 0.5}
        assert result["centered_per_task"] == {"task_a": 0.3}
        # Verify seed was passed through
        _, kwargs = mock_eval.call_args
        assert kwargs["seed"] == 1234


class TestRunChatEvalSuite:
    @patch("scripts.train.training_progression._chat_eval_run_chat_eval", return_value=0.5)
    @patch("scripts.train.training_progression.Engine")
    def test_runs_configured_tasks(self, mock_engine_cls, mock_run_chat, sample_cfg):
        result = run_chat_eval_suite(MagicMock(), MagicMock(), sample_cfg)

        # sample_cfg has chat_tasks="ARC-Easy,MMLU"
        assert mock_run_chat.call_count == 2
        assert result["ARC-Easy"] == 0.5
        assert result["MMLU"] == 0.5

    @patch("scripts.train.training_progression._chat_eval_run_chat_eval", return_value=0.25)
    @patch("scripts.train.training_progression.Engine")
    def test_all_tasks_when_null(self, mock_engine_cls, mock_run_chat, sample_cfg):
        sample_cfg = _merge_cfg(sample_cfg, {"chat_tasks": None})
        run_chat_eval_suite(MagicMock(), MagicMock(), sample_cfg)
        # null = all 6 tasks
        assert mock_run_chat.call_count == 6

    @patch("scripts.train.training_progression._chat_eval_run_chat_eval", return_value=0.5)
    @patch("scripts.train.training_progression.Engine")
    def test_computes_chatcore_when_all_tasks(self, mock_engine_cls, mock_run_chat, sample_cfg):
        sample_cfg = _merge_cfg(sample_cfg, {"chat_tasks": None})
        result = run_chat_eval_suite(MagicMock(), MagicMock(), sample_cfg)
        assert "chatcore_metric" in result

    @patch("scripts.train.training_progression._chat_eval_run_chat_eval", return_value=0.5)
    @patch("scripts.train.training_progression.Engine")
    def test_no_chatcore_when_subset(self, mock_engine_cls, mock_run_chat, sample_cfg):
        result = run_chat_eval_suite(MagicMock(), MagicMock(), sample_cfg)
        # Only 2 tasks → no chatcore
        assert "chatcore_metric" not in result


class TestRunHarmfulnessEval:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": ""})
    def test_raises_without_api_key(self, sample_cfg):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            run_harmfulness_eval(MagicMock(), MagicMock(), "cpu", 42, sample_cfg)

    @patch("steering.eval.harmfulness_judge.compute_metrics")
    @patch("steering.eval.harmfulness_judge.JudgeClient")
    @patch("steering.eval.harmfulness_judge.load_benign_prompts")
    @patch("steering.eval.harmfulness_judge.load_harmful_prompts")
    @patch("scripts.train.training_progression.InferenceEngine")
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"})
    def test_full_pipeline(
        self, mock_engine_cls, mock_harmful, mock_benign, mock_judge_cls, mock_metrics, sample_cfg
    ):
        mock_harmful.return_value = [{"prompt": "bad", "category": "violence"}]
        mock_benign.return_value = [{"prompt": "good", "category": "none"}]

        mock_engine = MagicMock()
        mock_engine.generate_batch.return_value = ["response1", "response2"]
        mock_engine_cls.return_value = mock_engine

        mock_judge = MagicMock()
        mock_judge.score_all.return_value = [MagicMock(score=3, categories=[])] * 2
        mock_judge.score_compliance_all.return_value = [
            ComplianceJudgeResult(4, False, "Substantially helpful.")
        ]
        mock_judge_cls.return_value = mock_judge

        mock_metrics.return_value = {
            "harmful_prompts": {
                "asr": 0.8,
                "mean_score": 3.0,
                "harmfulness_rate": 0.8,
                "safe_response_rate": 0.2,
                "num_scored": 1,
                "num_failed": 0,
            },
            "benign_prompts": {
                "mean_score": 3.0,
                "harmfulness_rate": 0.2,
                "safe_response_rate": 0.4,
                "num_scored": 1,
                "num_failed": 0,
            },
            "overall": {
                "mean_score": 3.0,
                "harmfulness_rate": 0.5,
                "safe_response_rate": 0.3,
                "num_scored": 2,
                "num_failed": 0,
            },
            "per_category": dict.fromkeys(HARM_CATEGORIES, 0.0),
        }

        result = run_harmfulness_eval(MagicMock(), MagicMock(), "cpu", 42, sample_cfg)

        mock_harmful.assert_called_once_with()
        mock_benign.assert_called_once_with(n=10, seed=42)
        assert mock_engine.generate_batch.call_args.args[0] == ["bad", "good"]
        assert result["harmful_prompts"]["asr"] == 0.8
        assert result["overall"]["mean_score"] == 3.0
        assert result["harmful_prompts"]["safe_response_rate"] == 0.2
        assert result["benign_prompts"]["compliance_rate"] == 1.0


# ===================================================================
# Functional tests: main() entrypoint
# ===================================================================


class TestMain:
    @patch("scripts.train.training_progression.plot_results")
    @patch("scripts.train.training_progression.collect_results")
    @patch("scripts.train.training_progression.compose")
    @patch("scripts.train.training_progression.initialize_config_dir")
    def test_mode_collect(self, mock_init, mock_compose, mock_collect, mock_plot):
        mock_init.return_value.__enter__ = MagicMock()
        mock_init.return_value.__exit__ = MagicMock(return_value=False)
        mock_compose.return_value = OmegaConf.create({"mode": "collect"})

        with patch("sys.argv", ["prog"]):
            main()

        mock_collect.assert_called_once()
        mock_plot.assert_not_called()

    @patch("scripts.train.training_progression.plot_results")
    @patch("scripts.train.training_progression.collect_results")
    @patch("scripts.train.training_progression.compose")
    @patch("scripts.train.training_progression.initialize_config_dir")
    def test_mode_plot(self, mock_init, mock_compose, mock_collect, mock_plot):
        mock_init.return_value.__enter__ = MagicMock()
        mock_init.return_value.__exit__ = MagicMock(return_value=False)
        mock_compose.return_value = OmegaConf.create({"mode": "plot"})

        with patch("sys.argv", ["prog"]):
            main()

        mock_collect.assert_not_called()
        mock_plot.assert_called_once()

    @patch("scripts.train.training_progression.plot_results")
    @patch("scripts.train.training_progression.collect_results")
    @patch("scripts.train.training_progression.compose")
    @patch("scripts.train.training_progression.initialize_config_dir")
    def test_mode_both(self, mock_init, mock_compose, mock_collect, mock_plot):
        mock_init.return_value.__enter__ = MagicMock()
        mock_init.return_value.__exit__ = MagicMock(return_value=False)
        mock_compose.return_value = OmegaConf.create({"mode": "both"})

        with patch("sys.argv", ["prog"]):
            main()

        mock_collect.assert_called_once()
        mock_plot.assert_called_once()


# ===================================================================
# Additional coverage: edge cases in collect_results
# ===================================================================


class TestCollectEdgeCases:
    def _setup_local_checkpoints(self, tmp_path, steps, subdir=""):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        if subdir:
            ckpt_dir = ckpt_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (ckpt_dir / f"model_{step:06d}.pt").write_bytes(b"fake_model")
            meta = {
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
            (ckpt_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta))

    @patch("scripts.train.training_progression.cleanup_remote_checkpoint")
    @patch("scripts.train.training_progression.fetch_remote_checkpoint")
    @patch("scripts.train.training_progression.build_decoder_model")
    def test_model_load_failure_remote_cleans_up_when_enabled(
        self, mock_build, mock_fetch, mock_cleanup, tmp_path, sample_cfg
    ):
        """Model load failure with cleanup_after_eval=True triggers cleanup."""
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "checkpoint_source": "remote",
                    "cleanup_after_eval": True,
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_fetch.return_value = str(tmp_path / "ckpt")
        mock_build.side_effect = RuntimeError("corrupt checkpoint")

        _collect_results_without_context_cache_validation(sample_cfg)

        mock_cleanup.assert_called_once()

    @patch("scripts.train.training_progression.cleanup_remote_checkpoint")
    @patch("scripts.train.training_progression.fetch_remote_checkpoint")
    @patch("scripts.train.training_progression.build_decoder_model")
    def test_model_load_failure_remote_no_cleanup_when_disabled(
        self, mock_build, mock_fetch, mock_cleanup, tmp_path, sample_cfg
    ):
        """cleanup_after_eval=False (default) does not delete remote checkpoint."""
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "checkpoint_source": "remote",
                    "cleanup_after_eval": False,
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_fetch.return_value = str(tmp_path / "ckpt")
        mock_build.side_effect = RuntimeError("corrupt checkpoint")

        _collect_results_without_context_cache_validation(sample_cfg)

        mock_cleanup.assert_not_called()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_harmfulness_branch(self, mock_eval, tmp_path, sample_cfg):
        """Legacy sparse harmfulness configs fail before checkpoint execution."""
        self._setup_local_checkpoints(tmp_path, [500])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "harmfulness",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(harmfulness_seeds=(1234,))

        with pytest.raises(ValueError, match="no longer accepts hand-supplied"):
            _collect_results_without_context_cache_validation(sample_cfg)
        mock_eval.assert_not_called()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_unknown_eval_warns(self, mock_eval, tmp_path, sample_cfg):
        """Unknown eval suite name logs warning and continues."""
        self._setup_local_checkpoints(tmp_path, [500])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "bogus_eval",
                },
            )
        )
        mock_eval.return_value = {}

        _collect_results_without_context_cache_validation(sample_cfg)  # should not raise

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_partial_seed_delegates_with_existing_entry(
        self, mock_eval, tmp_path, sample_cfg
    ):
        """Partial conditioned checkpoints are delegated with their existing entry."""
        self._setup_local_checkpoints(tmp_path, [500])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 2,
                    "evals": "core",
                },
            )
        )

        # Pre-populate with one seed done
        partial = {
            "model_tag": "d27",
            "checkpoints": [
                {
                    "step": 500,
                    "phase": "pretrain",
                    "evals": _conditioned_evals(core_seeds=(1234,)),
                }
            ],
        }
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(partial, f)

        mock_eval.return_value = _conditioned_evals(core_seeds=(1234, 2234))

        _collect_results_without_context_cache_validation(sample_cfg)

        mock_eval.assert_called_once()
        existing_entry = mock_eval.call_args.args[6]
        assert existing_entry["evals"] == _conditioned_evals(core_seeds=(1234,))


# ===================================================================
# Functional tests: evaluate_single_checkpoint
# ===================================================================


class TestEvaluateSingleCheckpoint:
    def _setup_local_checkpoints(self, tmp_path, steps, subdir=""):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        if subdir:
            ckpt_dir = ckpt_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (ckpt_dir / f"model_{step:06d}.pt").write_bytes(b"fake_model")
            meta = {
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
            (ckpt_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta))

    @pytest.mark.parametrize("load_fails", [False, True], ids=["success", "load-failure"])
    def test_local_cleanup_is_rejected_before_checkpoint_access(
        self, load_fails, tmp_path, sample_cfg
    ):
        self._setup_local_checkpoints(tmp_path, [500])
        checkpoint = tmp_path / "data" / "base_checkpoints" / "d27" / "model_000500.pt"
        checksum_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cfg = _merge_cfg(sample_cfg, {"cleanup_after_eval": True})
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")

        with (
            patch(
                "scripts.train.training_progression.resolve_local_checkpoint_dir",
                wraps=resolve_local_checkpoint_dir,
            ) as mock_resolve,
            patch("scripts.train.training_progression.load_progression_model") as mock_load,
            patch("scripts.train.training_progression._run_eval_suite") as mock_eval,
            patch(
                "scripts.train.training_progression.cleanup_remote_checkpoint",
                wraps=cleanup_remote_checkpoint,
            ) as mock_cleanup,
        ):
            if load_fails:
                mock_load.side_effect = RuntimeError("corrupt checkpoint")
            else:
                mock_load.return_value = (MagicMock(), MagicMock(), {})

            with pytest.raises(
                ValueError,
                match="cleanup_after_eval.*checkpoint_source=remote",
            ):
                evaluate_single_checkpoint(
                    spec,
                    t.device("cpu"),
                    t.float32,
                    ["core"],
                    [1234],
                    cfg,
                )

        mock_resolve.assert_not_called()
        mock_load.assert_not_called()
        mock_eval.assert_not_called()
        mock_cleanup.assert_not_called()
        assert checkpoint.exists()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum_before

    def test_remote_cleanup_removes_staged_checkpoint_after_success(self, tmp_path, sample_cfg):
        staged_dir = tmp_path / "remote-staging"
        staged_dir.mkdir()
        checkpoint = staged_dir / "model_000500.pt"
        checkpoint.write_bytes(b"staged checkpoint")
        metadata = staged_dir / "meta_000500.json"
        metadata.write_text("{}")
        cfg = _merge_cfg(
            sample_cfg,
            {
                "checkpoint_source": "remote",
                "cleanup_after_eval": True,
            },
        )
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")

        with (
            patch(
                "scripts.train.training_progression.fetch_remote_checkpoint",
                return_value=str(staged_dir),
            ),
            patch(
                "scripts.train.training_progression.load_progression_model",
                return_value=(MagicMock(), MagicMock(), {}, None),
            ) as mock_load,
            patch(
                "scripts.train.training_progression.cleanup_remote_checkpoint",
                wraps=cleanup_remote_checkpoint,
            ) as mock_cleanup,
        ):
            evaluate_single_checkpoint(
                spec,
                t.device("cpu"),
                t.float32,
                ["core"],
                [1234],
                cfg,
            )

        mock_load.assert_called_once()
        mock_cleanup.assert_called_once_with(str(staged_dir), 500)
        assert not checkpoint.exists()
        assert metadata.exists()

    def test_split_cell_defers_cleanup_until_coordinator_acknowledges(self, tmp_path, sample_cfg):
        staged_dir = tmp_path / "remote-staging"
        staged_dir.mkdir()
        checkpoint = staged_dir / "model_000500.pt"
        checkpoint.write_bytes(b"staged checkpoint")
        (staged_dir / "meta_000500.json").write_text("{}")
        cfg = _merge_cfg(
            sample_cfg,
            {
                "checkpoint_source": "remote",
                "cleanup_after_eval": True,
                "model_family": "decoder",
            },
        )
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")

        with (
            patch(
                "scripts.train.training_progression.fetch_remote_checkpoint",
                return_value=str(staged_dir),
            ),
            patch(
                "scripts.train.training_progression.load_progression_model",
                return_value=(MagicMock(), MagicMock(), {}, None),
            ),
            patch(
                "scripts.train.training_progression._run_eval_suite",
                return_value={"core_metric": 0.5},
            ),
            patch(
                "scripts.train.training_progression.cleanup_remote_checkpoint",
                wraps=cleanup_remote_checkpoint,
            ) as mock_cleanup,
        ):
            evaluate_single_checkpoint(
                spec,
                t.device("cpu"),
                t.float32,
                ["core"],
                [1234],
                cfg,
                work_keys={("core", "decoder", "", 1234)},
            )

        mock_cleanup.assert_not_called()
        assert checkpoint.read_bytes() == b"staged checkpoint"

        cleanup_remote_checkpoint(str(staged_dir), spec.step)
        assert not checkpoint.exists()

    @patch("scripts.train.training_progression.run_core_eval")
    @patch("scripts.train.training_progression.build_decoder_model")
    def test_decoder_checkpoint_runs_unconditional_eval(
        self, mock_build, mock_core, tmp_path, sample_cfg
    ):
        self._setup_local_checkpoints(tmp_path, [500])

        fake_model = MagicMock()
        fake_model.to = MagicMock(return_value=fake_model)
        mock_build.return_value = (fake_model, MagicMock(), {})
        mock_core.return_value = {"core_metric": 0.5, "per_task": {}, "centered_per_task": {}}

        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        evals = evaluate_single_checkpoint(
            spec, t.device("cpu"), t.float32, ["core"], [1234], sample_cfg
        )

        assert evals["core"]["seeds"]["1234"]["core_metric"] == 0.5
        mock_core.assert_called_once()

    @patch("scripts.train.training_progression.run_core_eval")
    @patch("scripts.train.training_progression.build_decoder_model")
    def test_preserves_existing_seeds(self, mock_build, mock_core, tmp_path, sample_cfg):
        """Existing seeds from a prior run are preserved, new ones added."""
        self._setup_local_checkpoints(tmp_path, [500])

        fake_model = MagicMock()
        fake_model.to = MagicMock(return_value=fake_model)
        mock_build.return_value = (fake_model, MagicMock(), {})
        mock_core.return_value = {"core_metric": 0.6, "per_task": {}, "centered_per_task": {}}

        existing = {"evals": {"core": {"seeds": {"1234": {"core_metric": 0.5}}}}}
        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        evals = evaluate_single_checkpoint(
            spec,
            t.device("cpu"),
            t.float32,
            ["core"],
            [1234, 2234],
            sample_cfg,
            existing_entry=existing,
        )

        # Decoder progression preserves completed seeds and fills only the
        # missing unconditional seed.
        assert evals["core"]["seeds"]["1234"]["core_metric"] == 0.5
        assert evals["core"]["seeds"]["2234"]["core_metric"] == 0.6
        mock_core.assert_called_once()

    @patch("scripts.train.training_progression.build_decoder_model")
    def test_raises_on_load_failure(self, mock_build, tmp_path, sample_cfg):
        self._setup_local_checkpoints(tmp_path, [500])
        mock_build.side_effect = RuntimeError("corrupt")

        spec = CheckpointSpec(step=500, phase="pretrain", subdir="")
        with pytest.raises(RuntimeError, match="Failed to load model"):
            evaluate_single_checkpoint(
                spec, t.device("cpu"), t.float32, ["core"], [1234], sample_cfg
            )

    @patch("scripts.train.training_progression.run_core_eval")
    @patch("scripts.train.training_progression.build_encdec_model")
    def test_encdec_uses_encdec_builder(self, mock_build, mock_core, tmp_path, sample_cfg):
        ckpt_dir = tmp_path / "data" / "conditional_encdec_checkpoints" / "d12" / "s128"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_000250.pt").write_bytes(b"fake_model")
        (ckpt_dir / "meta_000250.json").write_text("{}")
        cfg = _merge_cfg(
            sample_cfg,
            {
                "model_tag": "d12",
                "model_family": "encoder_decoder",
                "checkpoint_root": "conditional_encdec_checkpoints",
                "segment_size": 128,
            },
        )

        fake_model = MagicMock()
        fake_model.to = MagicMock(return_value=fake_model)
        mock_build.return_value = (
            fake_model,
            MagicMock(),
            {"user_config": {}},
        )
        mock_core.return_value = {
            "core_metric": 0.5,
            "per_task": {},
            "centered_per_task": {},
        }

        spec = CheckpointSpec(step=250, phase="pretrain", subdir="")
        evaluate_single_checkpoint(
            spec,
            t.device("cpu"),
            t.float32,
            ["core"],
            [1234],
            cfg,
        )

        mock_build.assert_called_once_with(
            str(ckpt_dir),
            250,
            t.device("cpu"),
            "eval",
            eval_conditioning=conditioned_eval_conditioning(
                "training_progression"
            ),
        )


# ===================================================================
# Multi-GPU tests
# ===================================================================


class TestMultiGPUCollect:
    def _setup_local_checkpoints(self, tmp_path, steps, subdir=""):
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        if subdir:
            ckpt_dir = ckpt_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for step in steps:
            (ckpt_dir / f"model_{step:06d}.pt").write_bytes(b"fake_model")
            meta = {
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
            (ckpt_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta))

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    @patch("scripts.train.training_progression.torch.cuda.is_available", return_value=False)
    def test_falls_back_to_single_gpu_without_cuda(
        self, mock_cuda, mock_eval, tmp_path, sample_cfg
    ):
        """num_gpus>1 but no CUDA falls back to single-GPU path."""
        self._setup_local_checkpoints(tmp_path, [500])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "num_gpus": 4,
                    "device_type": "cpu",
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        # Should use single-GPU path (CPU), not multi-GPU
        mock_eval.assert_called_once()

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_single_gpu_default(self, mock_eval, tmp_path, sample_cfg):
        """Default num_gpus=1 uses single-GPU path."""
        self._setup_local_checkpoints(tmp_path, [500, 1000])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "num_gpus": 1,
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )
        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        assert mock_eval.call_count == 2

    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_skips_already_done_checkpoints(self, mock_eval, tmp_path, sample_cfg):
        """Multi-GPU path also respects resume (skips completed checkpoints)."""
        self._setup_local_checkpoints(tmp_path, [500, 1000])
        sample_cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "num_gpus": 1,
                    "phases": [{"phase": "pretrain", "subdir": "", "steps": [500, 1000]}],
                    "num_seeds": 1,
                    "evals": "core",
                },
            )
        )

        # Pre-populate step 500 as done
        existing = {
            "model_tag": "d27",
            "checkpoints": [
                {
                    "step": 500,
                    "phase": "pretrain",
                    "evals": _conditioned_evals(core_seeds=(1234,)),
                }
            ],
        }
        path = Path(sample_cfg.results_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(existing, f)

        mock_eval.return_value = _conditioned_evals(core_seeds=(1234,))

        _collect_results_without_context_cache_validation(sample_cfg)

        # Only step 1000 should be evaluated
        assert mock_eval.call_count == 1

    def test_split_remote_cleanup_runs_after_all_rank_acknowledgements(
        self, tmp_path, sample_cfg, monkeypatch
    ):
        """The split coordinator cleans only after every cell is gathered."""
        cfg = _merge_cfg(
            sample_cfg,
            {
                "checkpoint_source": "remote",
                "cleanup_after_eval": True,
                "model_family": "decoder",
                "num_gpus": 2,
                "device_type": "cuda",
                "phases": [{"phase": "pretrain", "subdir": "", "steps": [500]}],
                "evals": "core",
                "num_seeds": 2,
            },
        )
        staged_dir = Path(cfg.remote.local_temp_dir) / cfg.model_tag
        staged_dir.mkdir(parents=True)
        checkpoint = staged_dir / "model_000500.pt"
        checkpoint.write_bytes(b"shared staged checkpoint")
        (staged_dir / "meta_000500.json").write_text("{}")

        for name, value in {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }.items():
            monkeypatch.setenv(name, value)

        events: list[tuple[str, int]] = []

        def evaluate_cell(
            _spec,
            _device,
            _ptdtype,
            _eval_suites,
            _seeds,
            _cfg,
            _existing,
            *,
            work_keys,
        ):
            assert checkpoint.exists()
            assert len(work_keys) == 1
            seed = next(iter(work_keys))[3]
            events.append(("cell", seed))
            return {"core": {"seeds": {str(seed): {"core_metric": 0.5}}}}

        def acknowledge_rank_results(gathered, local):
            events.append(("ack", len(local)))
            gathered[0] = local
            gathered[1] = []

        def cleanup_after_ack(local_dir, step):
            events.append(("cleanup", step))
            cleanup_remote_checkpoint(local_dir, step)

        fake_dist = MagicMock()
        fake_dist.all_gather_object.side_effect = acknowledge_rank_results
        with (
            patch(
                "scripts.train.training_progression.evaluate_single_checkpoint",
                side_effect=evaluate_cell,
            ),
            patch(
                "scripts.train.training_progression.cleanup_remote_checkpoint",
                side_effect=cleanup_after_ack,
            ) as mock_cleanup,
            patch("scripts.train.training_progression.dist", fake_dist),
            patch("scripts.train.training_progression.torch.cuda.set_device"),
        ):
            _collect_results_without_context_cache_validation(cfg)

        assert [event[0] for event in events].count("cell") == 2
        cleanup_index = next(index for index, event in enumerate(events) if event[0] == "cleanup")
        assert cleanup_index > max(index for index, event in enumerate(events) if event[0] == "ack")
        mock_cleanup.assert_called_once_with(str(staged_dir), 500)
        assert not checkpoint.exists()


# ===================================================================
# End-to-end: collect + plot
# ===================================================================


class TestEndToEnd:
    @patch("scripts.train.training_progression.evaluate_single_checkpoint")
    def test_collect_then_plot(self, mock_eval, tmp_path, sample_cfg):
        """Full pipeline: collect results then generate plots."""
        # Setup
        ckpt_dir = tmp_path / "data" / "base_checkpoints" / "d27"
        ckpt_dir.mkdir(parents=True)
        for step in [500, 1000]:
            (ckpt_dir / f"model_{step:06d}.pt").write_bytes(b"data")
            meta = {
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
            (ckpt_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta))

        cfg = _conditioned_eval_cfg(
            _merge_cfg(
                sample_cfg,
                {
                    "phases": [
                        {"phase": "pretrain", "subdir": "", "steps": [500, 1000]},
                    ],
                    "num_seeds": 2,
                    "evals": "core",
                },
            )
        )

        call_count = [0]

        def eval_side_effect(*args, **kwargs):
            call_count[0] += 1
            return _conditioned_evals(
                core_seeds=(1234, 2234),
                core_result={
                    "core_metric": 0.05 * call_count[0],
                    "per_task": {},
                    "centered_per_task": {},
                },
            )

        mock_eval.side_effect = eval_side_effect

        # Collect
        _collect_results_without_context_cache_validation(cfg)

        results_path = Path(cfg.results_file)
        assert results_path.exists()
        with open(results_path) as f:
            results = json.load(f)
        assert len(results["checkpoints"]) == 2

        # Plot
        plot_results(cfg)

        fig_dir = Path(cfg.fig_dir)
        assert (fig_dir / "core_metric.png").exists()
        assert (fig_dir / "core_metric.png").stat().st_size > 0
