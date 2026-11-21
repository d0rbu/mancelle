"""Legend labels should preserve spaces and case from user-supplied names.

The `name:` on a conditioned config serves two jobs:
- a stable identifier for JSON keys / filenames (sanitized — spaces become
  underscores, filesystem-hostile chars get collapsed),
- a human-readable label in the chart legend (raw — preserved exactly as
  typed, spaces and all).

The `FeatureConfig` carries both so callers don't have to
guess. Results JSON persists `display_name` alongside the sanitized key
so the plotter can recover it at render time.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from steering.eval.feature_config import (
    FeatureConfig,
    parse_strict_feature_configs,
    uniquify_config_names,
)
from steering.eval.progression_plotting import build_multi_series_spec

# ---------------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------------


def test_dataclass_has_display_name_field():
    cfg = FeatureConfig(
        name="Safety_p90",
        display_name="Safety p90",
        features={},
        source=None,
    )
    assert cfg.name == "Safety_p90"
    assert cfg.display_name == "Safety p90"


def test_dataclass_display_name_defaults_to_name():
    """When constructed without display_name, it mirrors name."""
    cfg = FeatureConfig(name="foo", features={}, source=None)
    assert cfg.display_name == "foo"


# ---------------------------------------------------------------------------
# Parsing preserves spaces in display_name
# ---------------------------------------------------------------------------


def _flat_yaml(tmp_path: Path, name: str, mapping: dict[int, int]) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(mapping))
    return p


def test_inline_dict_name_with_space_preserves_display_name(tmp_path: Path):
    yaml_path = _flat_yaml(tmp_path, "safety.yaml", {0: 0, 1: 1})
    parsed = parse_strict_feature_configs(
        [{"name": "Safety p90", "features_file": str(yaml_path)}]
    )
    assert len(parsed) == 1
    cfg = parsed[0]
    assert cfg.name == "Safety_p90"        # storage-safe
    assert cfg.display_name == "Safety p90"  # original


def test_inline_dict_without_name_defaults_display_to_stem(tmp_path: Path):
    yaml_path = _flat_yaml(tmp_path, "my_features.yaml", {0: 0})
    parsed = parse_strict_feature_configs([{"features_file": str(yaml_path)}])
    cfg = parsed[0]
    assert cfg.name == "my_features"
    assert cfg.display_name == "my_features"


def test_string_path_display_name_is_stem(tmp_path: Path):
    """String entries (YAML paths) still get a display_name — the file stem."""
    yaml_path = _flat_yaml(tmp_path, "safety.yaml", {0: 0})
    parsed = parse_strict_feature_configs([str(yaml_path)])
    assert parsed[0].display_name == "safety"


def test_yaml_structured_config_with_name_key_preserves_space(tmp_path: Path):
    """A structured YAML with `name: "Foo Bar"` keeps the space in display_name."""
    yaml_path = tmp_path / "structured.yaml"
    yaml_path.write_text(
        yaml.safe_dump({"name": "Safety p90", "features": {0: 0, 1: 1}})
    )
    parsed = parse_strict_feature_configs([str(yaml_path)])
    cfg = parsed[0]
    assert cfg.name == "Safety_p90"
    assert cfg.display_name == "Safety p90"


def test_uniquify_carries_display_name_through(tmp_path: Path):
    """Duplicates get `_2` on the sanitized name; display_name is unchanged."""
    configs = [
        FeatureConfig(
            name="Foo", display_name="Foo display", features={0: 0}, source=None
        ),
        FeatureConfig(
            name="Foo", display_name="Foo display", features={1: 0}, source=None
        ),
    ]
    out = uniquify_config_names(configs)
    assert [c.name for c in out] == ["Foo", "Foo_2"]
    assert all(c.display_name == "Foo display" for c in out)


# ---------------------------------------------------------------------------
# Plot uses display_name in legend
# ---------------------------------------------------------------------------


def test_plot_uses_display_name_when_present():
    """build_multi_series_spec reads `display_name` from the results entry."""
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
                                "display_name": "Safety p90",
                                "seeds": {"1234": {"core_metric": 0.06}},
                            }
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
    labels = [s.label for s in spec.series if not s.is_baseline]
    assert labels == ["Conditioned: Safety p90"]


def test_plot_falls_back_to_sanitized_name_when_display_missing():
    """Old results.json files without `display_name` still render readably."""
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "conditioned": {
                        "configs": {
                            "Safety_p90": {
                                "seeds": {"1234": {"core_metric": 0.06}}
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
    assert labels == ["Conditioned: Safety_p90"]


def test_plot_multiple_configs_each_use_own_display_name():
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "conditioned": {
                        "configs": {
                            "Safety_p90": {
                                "display_name": "Safety p90",
                                "seeds": {"1234": {"core_metric": 0.06}},
                            },
                            "Capability_p90": {
                                "display_name": "Capability p90",
                                "seeds": {"1234": {"core_metric": 0.07}},
                            },
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
    labels = sorted(s.label for s in spec.series if not s.is_baseline)  # ty: ignore[unresolved-attribute]
    assert labels == ["Conditioned: Capability p90", "Conditioned: Safety p90"]
