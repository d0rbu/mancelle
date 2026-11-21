"""FeatureFilterConfig validation."""
from dataclasses import replace
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.feature_filter.config import (
    FeatureFilterConfig,
    JudgeConfig,
    validate_config,
)
from steering.eval.feature_filter.types import ConfigValidationError


def _base_cfg(**overrides: Any) -> FeatureFilterConfig:
    output_dir = str(overrides.pop("output_dir", "/tmp/ff"))
    base = FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=output_dir,
        max_concurrent_fetches=4,
        max_concurrent_judges=2,
        shutdown_grace_seconds=10.0,
        neuronpedia_cache_dir=f"{output_dir}/np_cache",
        judge_cache_path=f"{output_dir}/judge_cache.sqlite",
        features_output_path=f"{output_dir}/feature_scores.jsonl",
        accepted_features_path=f"{output_dir}/accepted_feature_ids.txt",
        report_output_path=f"{output_dir}/filter_report.json",
        runner_manifest_path=f"{output_dir}/runner_manifest.json",
        lockfile_path=f"{output_dir}/.lockfile",
        judge=JudgeConfig(),
    )
    return replace(base, **overrides)


def test_valid_config_passes(tmp_path) -> None:
    cfg = _base_cfg(output_dir=str(tmp_path))
    validate_config(cfg)


def test_empty_feature_ids_list_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        _base_cfg(feature_ids=[])


def test_duplicate_feature_ids_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        _base_cfg(feature_ids=[1, 2, 2])


def test_out_of_range_feature_id_rejected(tmp_path) -> None:
    with pytest.raises(ConfigValidationError, match="out of range"):
        validate_config(_base_cfg(output_dir=str(tmp_path), feature_ids=[20000], n_sae_features=16384))


def test_interp_min_out_of_range_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="interp_min_score"):
        _base_cfg(interp_min_score=7)


def test_consistency_min_out_of_range_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="consistency_min_score"):
        _base_cfg(consistency_min_score=-1)


def test_max_judge_calls_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_judge_calls"):
        _base_cfg(max_judge_calls=0)


def test_output_dir_is_file_rejected(tmp_path) -> None:
    p = tmp_path / "outfile"
    p.write_text("x")
    with pytest.raises(ConfigValidationError, match="output_dir"):
        validate_config(_base_cfg(output_dir=str(p)))


def test_model_id_with_path_separator_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="model_id"):
        _base_cfg(model_id="../evil")


def test_empty_model_id_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="model_id"):
        _base_cfg(model_id="")


def test_empty_sae_source_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="sae_source"):
        _base_cfg(sae_source="")


def test_sae_source_with_path_separator_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="sae_source"):
        _base_cfg(sae_source="a/b")


def test_n_sae_features_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        _base_cfg(n_sae_features=0)


def test_max_concurrent_fetches_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_concurrent_fetches"):
        _base_cfg(max_concurrent_fetches=0)


def test_max_concurrent_judges_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_concurrent_judges"):
        _base_cfg(max_concurrent_judges=0)


def test_error_abort_fraction_above_one_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="error_abort_fraction"):
        _base_cfg(error_abort_fraction=1.5)


def test_shutdown_grace_negative_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="shutdown_grace_seconds"):
        _base_cfg(shutdown_grace_seconds=-1.0)


def test_hist_sum_tolerance_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="hist_sum_tolerance"):
        _base_cfg(hist_sum_tolerance=0.0)


def test_judge_parse_max_retries_negative_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="parse_max_retries"):
        JudgeConfig(parse_max_retries=-1)


def test_output_dir_unwritable_rejected(tmp_path, monkeypatch) -> None:
    def _boom(self, *a, **kw):
        raise OSError("read-only filesystem")
    import pathlib
    monkeypatch.setattr(pathlib.Path, "mkdir", _boom)
    with pytest.raises(ConfigValidationError, match="unwritable"):
        validate_config(_base_cfg(output_dir=str(tmp_path / "new")))


def test_unresolved_interpolation_rejected(tmp_path) -> None:
    # Default lockfile_path is "${output_dir}/.lockfile" — if caller overrides
    # output_dir (resolving it) but leaves lockfile_path as the default, we must
    # reject the literal "${output_dir}" token.
    with pytest.raises(ConfigValidationError, match="unresolved interpolation"):
        validate_config(_base_cfg(
            output_dir=str(tmp_path),
            lockfile_path="${output_dir}/.lockfile",
        ))


def test_samples_per_bin_zero_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="samples_per_bin"):
        _base_cfg(samples_per_bin=0)


def test_samples_per_bin_six_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="samples_per_bin"):
        _base_cfg(samples_per_bin=6)


def test_samples_per_bin_five_accepted(tmp_path) -> None:
    validate_config(_base_cfg(output_dir=str(tmp_path), samples_per_bin=5))


def test_samples_per_bin_one_accepted(tmp_path) -> None:
    validate_config(_base_cfg(output_dir=str(tmp_path), samples_per_bin=1))
