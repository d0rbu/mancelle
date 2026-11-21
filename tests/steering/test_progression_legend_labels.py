"""Legend-label contract tests for progression plots.

The chart legend should read naturally:

- Each conditioned line is labeled **"Conditioned: <Name>"** (capitalized
  prefix, preserved case in the config-supplied name).
- Users can supply a descriptive `name:` per conditioned config in the
  progression YAML by using the inline-dict form with a `features_file:`
  pointing at a `{feature_id: bucket}` YAML. That name flows all the way
  through to the legend.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from steering.eval.feature_config import parse_strict_feature_configs
from steering.eval.progression_plotting import (
    build_multi_series_spec,
)


def _flat_yaml(tmp_path: Path, name: str, mapping: dict[int, int]) -> Path:
    """Write a flat `{feature_id: bucket}` YAML and return its path."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(mapping))
    return path


def test_top_level_seeds_do_not_create_a_series():
    """Top-level seed results are not a conditioning variant."""
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {"seeds": {"1234": {"core_metric": 0.05}}}
            },
        }
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is None


# ---------------------------------------------------------------------------
# Conditioned label prefix
# ---------------------------------------------------------------------------


def test_conditioned_label_has_capitalized_prefix():
    """Each conditioned series label starts with 'Conditioned: ' (capital C)."""
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "seeds": {"1234": {"core_metric": 0.05}},
                    "conditioned": {
                        "configs": {
                            "Safety_p90": {
                                "seeds": {"1234": {"core_metric": 0.06}}
                            },
                            "Capability_p90": {
                                "seeds": {"1234": {"core_metric": 0.07}}
                            },
                        }
                    },
                }
            },
        }
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    conditioned_labels = sorted(s.label for s in spec.series if not s.is_baseline)
    assert conditioned_labels == [
        "Conditioned: Capability_p90",
        "Conditioned: Safety_p90",
    ]


def test_conditioned_label_preserves_supplied_case():
    """Config-supplied name keeps its case (spaces stay collapsed to _ via
    sanitize, but Upper/Lowercase is preserved)."""
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "conditioned": {
                        "configs": {
                            "MyCustom_Mix": {
                                "seeds": {"1234": {"core_metric": 0.1}}
                            }
                        }
                    }
                }
            },
        }
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="m",
        model_tag="t",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    labels = [s.label for s in spec.series if not s.is_baseline]
    assert labels == ["Conditioned: MyCustom_Mix"]


# ---------------------------------------------------------------------------
# features_file: inline-dict support
# ---------------------------------------------------------------------------


def test_inline_dict_accepts_features_file(tmp_path: Path):
    """An inline-dict entry with `features_file:` loads features from YAML
    AND accepts a user-supplied `name:` for the legend."""
    yaml_path = _flat_yaml(tmp_path, "safety.yaml", {0: 0, 1: 1, 2: 1})
    parsed = parse_strict_feature_configs(
        [
            {"name": "Safety p90", "features_file": str(yaml_path)},
        ]
    )
    assert len(parsed) == 1
    cfg = parsed[0]
    # Name is sanitized (spaces become underscores) but case preserved.
    assert cfg.name == "Safety_p90"
    assert cfg.features == {0: 0, 1: 1, 2: 1}
    # `source` records the file path so it can round-trip.
    assert cfg.source == str(yaml_path)


def test_inline_dict_features_file_default_name_from_stem(tmp_path: Path):
    """When no `name:` is supplied, the stem of features_file wins."""
    yaml_path = _flat_yaml(tmp_path, "my_descriptive_name.yaml", {0: 0})
    parsed = parse_strict_feature_configs(
        [
            {"features_file": str(yaml_path)},
        ]
    )
    assert parsed[0].name == "my_descriptive_name"
    assert parsed[0].source == str(yaml_path)


def test_inline_dict_mixed_with_string_paths(tmp_path: Path):
    """Mixing string paths and inline-dict entries both resolve correctly."""
    yaml_a = _flat_yaml(tmp_path, "safety.yaml", {0: 1})
    yaml_b = _flat_yaml(tmp_path, "capability.yaml", {1: 0})
    parsed = parse_strict_feature_configs(
        [
            str(yaml_a),  # string form
            {"name": "Capability p90", "features_file": str(yaml_b)},  # inline
        ]
    )
    names = [cfg.name for cfg in parsed]
    assert names == ["safety", "Capability_p90"]


def test_inline_dict_features_file_missing_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        parse_strict_feature_configs(
            [{"features_file": str(missing)}]
        )


def test_inline_dict_features_file_and_inline_features_is_rejected(tmp_path: Path):
    """Supplying both `features:` (inline dict) and `features_file:` is a
    config error — there's no sensible way to merge a path with a dict."""
    yaml_a = _flat_yaml(tmp_path, "x.yaml", {0: 0})
    with pytest.raises(ValueError, match="features_file.*features"):
        parse_strict_feature_configs(
            [
                {
                    "features_file": str(yaml_a),
                    "features": {1: 1},
                }
            ]
        )
