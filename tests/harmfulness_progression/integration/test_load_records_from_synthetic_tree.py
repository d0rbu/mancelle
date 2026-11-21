"""End-to-end test of load_harmfulness_records against a synthetic tmp_path
file tree, exercising every distinction between match / filter-skip /
schema-skip / raise."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from steering.eval.harmfulness_progression import (
    DEFAULT_CONFIG_LABEL_MAP,
    HarmfulnessRecord,
    load_harmfulness_records,
)

# HarmfulnessRecord imported for the typed return-list annotation below;
# documents the contract that load_harmfulness_records returns
# list[HarmfulnessRecord] without forcing the test body to repeat the type.

SAFETY_DELTA_FC = (
    "safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/"
    "neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/"
    "r_mean_pct_incl_zeros/m0.5"
)
CAPABILITY_DELTA_FC = (
    "capabilities_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/"
    "neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/"
    "r_mean_pct_incl_zeros/m0.5"
)


def _yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _ok_data(
    *,
    model_tag: str,
    source: str,
    feature_configs: str,
    num_harmful: int = 700,
    asr: float = 0.3,
    mean_score: float = 2.0,
    safe_rate: float = 0.7,
) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "config": {
            "model_tag": model_tag,
            "num_harmful": num_harmful,
            "source": source,
            "feature_configs": feature_configs,
        },
        "harmful_prompts": {"asr": asr, "safe_response_rate": safe_rate},
        "benign_prompts": {"compliance_rate": 0.8},
        "overall": {
            "mean_score": mean_score,
            "safe_response_rate": safe_rate,
        },
    }


def test_load_records_from_synthetic_tree(tmp_path: Path) -> None:
    base = tmp_path

    # (a) matching files for d12_baseline / num_harmful=700 / known signal
    _yaml(
        base / "harmfulness_results" / "a" / "x.yaml",
        _ok_data(
            model_tag="d12_baseline",
            source="base",
            feature_configs=CAPABILITY_DELTA_FC,
        ),
    )
    _yaml(
        base / "harmfulness_results" / "b" / "y.yaml",
        _ok_data(
            model_tag="d12_baseline",
            source="base/safety_cond",
            feature_configs=SAFETY_DELTA_FC,
        ),
    )

    # (b) wrong model_tag — filtered out
    _yaml(
        base / "harmfulness_results" / "c" / "wrong_tag.yaml",
        _ok_data(
            model_tag="d13_base",
            source="base",
            feature_configs=SAFETY_DELTA_FC,
        ),
    )

    # (c) wrong num_harmful — filtered out
    _yaml(
        base / "harmfulness_results" / "d" / "wrong_n.yaml",
        _ok_data(
            model_tag="d12_baseline",
            source="base",
            feature_configs=SAFETY_DELTA_FC,
            num_harmful=100,
        ),
    )

    # (d) _responses noise file — silently skipped by discover()
    _yaml(
        base / "harmfulness_results" / "e" / "x_responses.yaml",
        _ok_data(
            model_tag="d12_baseline",
            source="base",
            feature_configs=SAFETY_DELTA_FC,
        ),
    )

    # (e) feature_configs not in label_map — filtered out
    _yaml(
        base / "harmfulness_results" / "f" / "unknown_signal.yaml",
        _ok_data(model_tag="d12_baseline", source="base", feature_configs="some/unknown/signal"),
    )

    # (f) non-harmfulness YAML *outside* the rglob — never touched
    _yaml(base / "other_dir" / "noise.yaml", {"unrelated": "data"})

    # (g) malformed YAML *inside* the glob with `config` but missing field — raises
    bad_path = base / "harmfulness_results" / "g" / "bad.yaml"
    _yaml(
        bad_path,
        {
            "harmfulness_eval_schema_version": 4,
            "config": {
                "model_tag": "d12_baseline",
                "num_harmful": 700,
                "source": "base",
                "feature_configs": SAFETY_DELTA_FC,
            },
            "harmful_prompts": {},  # missing asr
            "benign_prompts": {"compliance_rate": 0.5},
            "overall": {"mean_score": 1.0, "safe_response_rate": 0.5},
        },
    )

    # The bad file is in the rglob and looks like a harmfulness result, so
    # _parse_harmfulness_record raises. Therefore load_harmfulness_records
    # propagates the exception (no silent skip).
    with pytest.raises(KeyError, match=r"harmful_prompts\.asr"):
        load_harmfulness_records(
            base,
            model_tag="d12_baseline",
            num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP,
        )

    # Now remove the bad file and re-run; the surviving 2 ok files plus the
    # filtered set should yield exactly the 2 matching records.
    bad_path.unlink()

    # (h) YAML *inside* the rglob with NO config key — covers
    # harmfulness_progression.py:347 (_parse_harmfulness_record returning None).
    # _parse_harmfulness_record short-circuits, load_harmfulness_records skips
    # silently and the surviving records remain unaffected.
    _yaml(
        base / "harmfulness_results" / "h" / "no_config.yaml",
        {"hello": "world"},
    )

    records: list[HarmfulnessRecord] = load_harmfulness_records(
        base,
        model_tag="d12_baseline",
        num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP,
    )
    sources = sorted(r.source for r in records)
    signals = sorted(r.signal for r in records)
    assert len(records) == 2
    assert sources == ["base", "base/safety_cond"]
    # Both signals should be the human labels from the default map.
    assert "Capability delta-base" in signals
    assert "Safety delta-base" in signals
