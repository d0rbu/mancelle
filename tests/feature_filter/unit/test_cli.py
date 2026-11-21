"""CLI smoke: Hydra composes the config and run() executes."""
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir

from steering.eval.feature_filter import cli as cli_mod
from steering.eval.feature_filter.cli import _build_cfg, main
from steering.eval.feature_filter.config import (
    FeatureFilterConfig,
    JudgeConfig,
    NeuronpediaInlineConfig,
)
from steering.eval.feature_filter.types import (
    ConcurrentRunnerError,
    ConfigValidationError,
    CostCapExceeded,
)
from steering.eval.neuronpedia_client import NeuronpediaAuthError


def test_cli_help_runs() -> None:
    # The --help flag exits 0 on Hydra apps
    result = subprocess.run(
        [sys.executable, "-m", "steering.eval.feature_filter.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "model_id" in result.stdout or "model_id" in result.stderr


def test_build_cfg_converts_nested_configs(tmp_path) -> None:
    cfg_dir = str(
        Path(__file__).resolve().parents[3] / "configs" / "eval" / "feature_filter"
    )
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg_dict = compose(
            config_name="feature_filter",
            overrides=[
                "model_id=gemma-2-2b",
                "sae_source=8-gemmascope-res-16k",
                "tokenizer_id=google/gemma-2-2b",
                f"output_dir={tmp_path}",
            ],
        )
    cfg = _build_cfg(cfg_dict)
    assert isinstance(cfg, FeatureFilterConfig)
    assert isinstance(cfg.neuronpedia, NeuronpediaInlineConfig)
    assert isinstance(cfg.judge, JudgeConfig)
    assert cfg.judge.model == "openai/gpt-5-nano"
    assert cfg.samples_per_bin == 3
    assert not hasattr(cfg, "percentiles")
    assert cfg.neuronpedia.timeout_seconds == 30.0


def _compose_cfg(tmp_path):
    cfg_dir = str(
        Path(__file__).resolve().parents[3] / "configs" / "eval" / "feature_filter"
    )
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg_dict = compose(
            config_name="feature_filter",
            overrides=[
                "model_id=gemma-2-2b",
                "sae_source=8-gemmascope-res-16k",
                "tokenizer_id=google/gemma-2-2b",
                f"output_dir={tmp_path}",
            ],
        )
    return cfg_dict


@dataclass
class _FakeReport:
    n_total: int = 10
    n_accepted: int = 6
    n_rejected: int = 3
    n_fetch_errors: int = 0
    n_judge_errors: int = 0
    n_prompt_errors: int = 0
    n_not_published: int = 0


def _patch_cli_deps():
    """Patch heavy external deps used by cli.main — tokenizer, cache, LLM client."""
    return [
        patch.object(cli_mod, "AutoTokenizer", MagicMock(from_pretrained=MagicMock(return_value=MagicMock()))),
        patch.object(cli_mod, "NeuronpediaActivationCache", MagicMock()),
        patch.object(cli_mod, "OpenRouterLLMClient", MagicMock()),
        patch.object(cli_mod, "SynthResponseCache", MagicMock()),
        patch.object(cli_mod, "make_interp_judge", MagicMock()),
    ]


def test_main_success_exit_zero(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    runner = MagicMock()
    runner.run.return_value = _FakeReport()
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 0


def test_main_config_validation_error_exit_two(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    with patch.object(cli_mod, "validate_config", side_effect=ConfigValidationError("bad")):
        rc = main.__wrapped__(cfg_dict)
    assert rc == 2


def test_main_neuronpedia_auth_error_exit_three(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    bad_tok = MagicMock()
    bad_tok.from_pretrained.side_effect = NeuronpediaAuthError("unauth")
    with patch.object(cli_mod, "AutoTokenizer", bad_tok):
        rc = main.__wrapped__(cfg_dict)
    assert rc == 3


def test_main_concurrent_runner_error_exit_one(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    runner = MagicMock()
    runner.run.side_effect = ConcurrentRunnerError("lock held")
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 1


def test_main_cost_cap_exit_four(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    runner = MagicMock()
    runner.run.side_effect = CostCapExceeded("over budget")
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 4


def test_main_error_abort_fraction_exit_one(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    # With defaults (error_abort_fraction=0.5), report with 100% errors → exit 1
    report = _FakeReport(
        n_total=4, n_accepted=0, n_rejected=0,
        n_fetch_errors=3, n_judge_errors=1, n_prompt_errors=0,
    )
    runner = MagicMock()
    runner.run.return_value = report
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 1


def test_main_all_not_published_exit_zero_with_warning(tmp_path, caplog) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    report = _FakeReport(
        n_total=10, n_accepted=0, n_rejected=0,
        n_fetch_errors=0, n_judge_errors=0, n_prompt_errors=0,
        n_not_published=10,
    )
    runner = MagicMock()
    runner.run.return_value = report
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 0
    # Strengthened: the operator-facing warning is the ONLY signal a whole
    # SAE was skipped — pin its content (count n/total + sae_source), not
    # just a substring, so a refactor that drops them fails this test.
    np_warnings = [
        r.message for r in caplog.records
        if "not published" in r.message.lower()
    ]
    assert np_warnings, caplog.text
    msg = np_warnings[0]
    assert "10/10" in msg, msg
    assert "8-gemmascope-res-16k" in msg, msg
    assert "excluded from error rate" in msg, msg


def test_main_no_not_published_warning_when_zero(tmp_path, caplog) -> None:
    """Negative: a clean run (n_not_published == 0) must NOT emit the
    not-published warning (guards against `if >= 0`-style guard breakage)."""
    cfg_dict = _compose_cfg(tmp_path)
    report = _FakeReport(
        n_total=10, n_accepted=8, n_rejected=2,
        n_fetch_errors=0, n_judge_errors=0, n_prompt_errors=0,
        n_not_published=0,
    )
    runner = MagicMock()
    runner.run.return_value = report
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 0
    assert not any(
        "not published" in r.message.lower() for r in caplog.records
    ), caplog.text


def test_main_not_published_excluded_from_error_rate(tmp_path) -> None:
    cfg_dict = _compose_cfg(tmp_path)
    # 5 not_published + 1 real fetch_error out of 10 → errored/total =
    # 1/10 = 0.1 <= 0.5 default → exit 0 (not_published must NOT count).
    report = _FakeReport(
        n_total=10, n_accepted=4, n_rejected=0,
        n_fetch_errors=1, n_judge_errors=0, n_prompt_errors=0,
        n_not_published=5,
    )
    runner = MagicMock()
    runner.run.return_value = report
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 0


def test_main_defensive_not_published_catch_exit_zero(tmp_path, caplog) -> None:
    """If a NeuronpediaNotPublishedError somehow escapes runner.run(),
    cli.main treats it as a known limitation (exit 0 + warning), not a crash."""
    from steering.eval.neuronpedia_client import NeuronpediaNotPublishedError

    cfg_dict = _compose_cfg(tmp_path)
    runner = MagicMock()
    runner.run.side_effect = NeuronpediaNotPublishedError("escaped to cli")
    patches = _patch_cli_deps() + [
        patch.object(cli_mod, "FeatureFilterRunner", MagicMock(return_value=runner)),
    ]
    for p in patches:
        p.start()
    try:
        rc = main.__wrapped__(cfg_dict)
    finally:
        for p in patches:
            p.stop()
    assert rc == 0


def test_build_cfg_rejects_non_mapping_top_level() -> None:
    from omegaconf import OmegaConf
    with pytest.raises(ConfigValidationError, match="mapping at top level"):
        _build_cfg(OmegaConf.create([1, 2, 3]))


def test_build_cfg_rejects_non_mapping_nested() -> None:
    from omegaconf import OmegaConf
    bad = OmegaConf.create({
        "model_id": "m", "sae_source": "s", "tokenizer_id": "t", "output_dir": "/tmp",
        "neuronpedia": "not-a-dict", "judge": {},
    })
    with pytest.raises(ConfigValidationError, match="neuronpedia.*must be a mapping"):
        _build_cfg(bad)
