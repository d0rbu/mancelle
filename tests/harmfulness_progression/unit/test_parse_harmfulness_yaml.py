"""parse_harmfulness_yaml: strict schema enforcement, no silent defaults."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from steering.eval.harmfulness_progression import (
    HarmfulnessMetrics,
    HarmfulnessRecord,
    _parse_harmfulness_record,
)


def parse_harmfulness_yaml(path: Path) -> HarmfulnessRecord | None:
    return _parse_harmfulness_record(yaml.safe_load(path.read_text()), path)


def _well_formed(
    *,
    model_tag: str = "d12_baseline",
    num_harmful: int = 700,
    source: str = "base",
    feature_configs: str = "safety_sft/f0.9/m0.5",
    asr: float = 0.3,
    mean_score: float = 2.5,
    safe_response_rate: float = 0.6,
    compliance_rate: float = 0.7,
) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "config": {
            "model_tag": model_tag,
            "num_harmful": num_harmful,
            "source": source,
            "feature_configs": feature_configs,
        },
        "harmful_prompts": {
            "asr": asr,
            "safe_response_rate": safe_response_rate,
        },
        "benign_prompts": {"compliance_rate": compliance_rate},
        "overall": {
            "mean_score": mean_score,
            "safe_response_rate": safe_response_rate,
        },
    }


def _write(tmp_path: Path, name: str, data) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data) if data is not None else "")
    return p


def test_parses_well_formed_yaml_to_record(tmp_path: Path) -> None:
    p = _write(tmp_path, "ok.yaml", _well_formed())
    rec = parse_harmfulness_yaml(p)
    assert isinstance(rec, HarmfulnessRecord)
    assert rec.model_tag == "d12_baseline"
    assert rec.source == "base"
    assert rec.signal == "safety_sft/f0.9/m0.5"
    assert rec.full_conditioning is False
    assert rec.metrics == HarmfulnessMetrics(
        asr=0.3,
        mean_score=2.5,
        safe_rate=0.6,
        compliance_rate=0.7,
    )


def test_returns_none_when_config_key_missing(tmp_path: Path) -> None:
    """File outside the harmfulness schema (no `config` key) is silently
    skipped — caller's rglob may have matched an unrelated YAML."""
    p = _write(tmp_path, "unrelated.yaml", {"hello": "world"})
    assert parse_harmfulness_yaml(p) is None


def test_returns_none_when_yaml_is_empty(tmp_path: Path) -> None:
    p = _write(tmp_path, "empty.yaml", None)
    assert parse_harmfulness_yaml(p) is None


@pytest.mark.parametrize("schema_version", [None, 2])
def test_rejects_missing_or_stale_schema(
    tmp_path: Path,
    schema_version: int | None,
) -> None:
    data = _well_formed()
    if schema_version is None:
        del data["harmfulness_eval_schema_version"]
    else:
        data["harmfulness_eval_schema_version"] = schema_version
    p = _write(tmp_path, "stale.yaml", data)

    with pytest.raises(
        ValueError,
        match=r"harmfulness_eval_schema_version must equal 4",
    ):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_model_tag_missing_in_config(tmp_path: Path) -> None:
    data = _well_formed()
    del data["config"]["model_tag"]
    p = _write(tmp_path, "no_tag.yaml", data)
    with pytest.raises(KeyError, match=r"config\.model_tag"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_source_missing_in_config(tmp_path: Path) -> None:
    data = _well_formed()
    del data["config"]["source"]
    p = _write(tmp_path, "no_source.yaml", data)
    with pytest.raises(KeyError, match=r"config\.source"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_harmful_prompts_asr_missing(tmp_path: Path) -> None:
    data = _well_formed()
    del data["harmful_prompts"]["asr"]
    p = _write(tmp_path, "no_asr.yaml", data)
    with pytest.raises(KeyError, match=r"harmful_prompts\.asr"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_overall_mean_score_missing(tmp_path: Path) -> None:
    data = _well_formed()
    del data["overall"]["mean_score"]
    p = _write(tmp_path, "no_mean.yaml", data)
    with pytest.raises(KeyError, match=r"overall\.mean_score"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_harmful_safe_response_rate_missing(
    tmp_path: Path,
) -> None:
    data = _well_formed()
    del data["harmful_prompts"]["safe_response_rate"]
    p = _write(tmp_path, "no_safe.yaml", data)
    with pytest.raises(KeyError, match=r"harmful_prompts\.safe_response_rate"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_benign_compliance_rate_missing(
    tmp_path: Path,
) -> None:
    data = _well_formed()
    del data["benign_prompts"]["compliance_rate"]
    p = _write(tmp_path, "no_compliance.yaml", data)
    with pytest.raises(KeyError, match=r"benign_prompts\.compliance_rate"):
        parse_harmfulness_yaml(p)


def test_raises_when_persisted_judge_run_has_no_successes(
    tmp_path: Path,
) -> None:
    data = _well_formed(
        asr=0.0,
        mean_score=0.0,
        safe_response_rate=0.0,
    )
    data["overall"]["num_scored"] = 0
    data["overall"]["num_failed"] = 3
    p = _write(tmp_path, "all_failed.yaml", data)

    with pytest.raises(ValueError, match="no successful judge results"):
        parse_harmfulness_yaml(p)


@pytest.mark.parametrize("mean_score", [0.0, 5.01, float("nan"), float("inf"), True])
def test_rejects_invalid_successful_mean_score(tmp_path: Path, mean_score: object) -> None:
    data = _well_formed()
    data["overall"]["mean_score"] = mean_score
    p = _write(tmp_path, "invalid_mean.yaml", data)
    with pytest.raises(ValueError, match=r"overall\.mean_score.*(must be numeric|\[1\.0, 5\.0\])"):
        parse_harmfulness_yaml(p)


def test_accepts_partial_judge_run_with_zero_harmfulness(
    tmp_path: Path,
) -> None:
    data = _well_formed(
        asr=0.0,
        mean_score=1.0,
        safe_response_rate=1.0,
    )
    data["overall"]["num_scored"] = 2
    data["overall"]["num_failed"] = 1
    p = _write(tmp_path, "partial_zero_harm.yaml", data)

    rec = parse_harmfulness_yaml(p)

    assert rec is not None
    assert rec.metrics == HarmfulnessMetrics(
        asr=0.0,
        mean_score=1.0,
        safe_rate=1.0,
        compliance_rate=0.7,
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("harmful_prompts", "asr", True),
        ("harmful_prompts", "safe_response_rate", "0.5"),
        ("benign_prompts", "compliance_rate", None),
        ("overall", "mean_score", float("nan")),
        ("overall", "mean_score", 5.1),
        ("overall", "mean_score", True),
    ],
)
def test_rejects_non_numeric_non_finite_and_out_of_range_metrics(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    data = _well_formed()
    data[section][field] = value
    path = _write(tmp_path, "invalid_metric.yaml", data)
    with pytest.raises(ValueError, match=field):
        parse_harmfulness_yaml(path)


def test_raises_yaml_error_on_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("this is :: not :: valid :: yaml\n  bad: indent")
    with pytest.raises(yaml.YAMLError):
        parse_harmfulness_yaml(p)


def test_records_full_conditioning_when_path_contains_full_conditioning(
    tmp_path: Path,
) -> None:
    sub = tmp_path / "harmfulness_results" / "full_conditioning" / "x"
    sub.mkdir(parents=True)
    p = _write(sub, "ok.yaml", _well_formed())
    rec = parse_harmfulness_yaml(p)
    assert rec is not None and rec.full_conditioning is True


def test_raises_keyerror_when_harmful_prompts_is_int_not_dict(
    tmp_path: Path,
) -> None:
    """Cover harmfulness_progression.py:293 — non-dict at harmful_prompts."""
    data = _well_formed()
    data["harmful_prompts"] = 42
    p = _write(tmp_path, "hp_int.yaml", data)
    with pytest.raises(KeyError, match="expected dict at harmful_prompts"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_harmful_prompts_is_list_not_dict(
    tmp_path: Path,
) -> None:
    data = _well_formed()
    data["harmful_prompts"] = [{"asr": 0.1}]
    p = _write(tmp_path, "hp_list.yaml", data)
    with pytest.raises(KeyError, match="expected dict at harmful_prompts"):
        parse_harmfulness_yaml(p)


def test_raises_keyerror_when_overall_is_string_not_dict(
    tmp_path: Path,
) -> None:
    data = _well_formed()
    data["overall"] = "not a dict"
    p = _write(tmp_path, "overall_str.yaml", data)
    with pytest.raises(KeyError, match="expected dict at harmful_prompts"):
        parse_harmfulness_yaml(p)


def test_records_full_conditioning_false_otherwise() -> None:
    # Use tempfile.mkdtemp with a controlled prefix that does NOT contain
    # "full_conditioning", since pytest names tmp_path after the test function
    # name, which may itself contain that substring.
    tmpdir = tempfile.mkdtemp(prefix="hptest_regular_")
    try:
        sub = Path(tmpdir) / "harmfulness_results" / "regular" / "x"
        sub.mkdir(parents=True)
        p = _write(sub, "ok.yaml", _well_formed())
        assert "full_conditioning" not in str(p), (
            f"controlled tmpdir unexpectedly contains 'full_conditioning': {p}"
        )
        rec = parse_harmfulness_yaml(p)
        assert rec is not None and rec.full_conditioning is False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
