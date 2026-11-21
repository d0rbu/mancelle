"""Tests for scripts/plot_evals.py eval visualization pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scripts.eval.plot_evals import (
    ModelNode,
    ParsedCondition,
    _extract_harmfulness_value,
    _find_encdec_leaf,
    _optimal_value,
    _probe_eval_results,
    collect_lineages,
    discover_model_tree,
    generate_all_plots,
    parse_base_eval_csv,
    parse_condition_label,
    parse_harmfulness_results,
    parse_steering_report,
    plot_cross_family_bars,
    plot_cross_family_harmfulness,
    plot_dose_response_overlay,
    plot_encdec_conditioning_sweep,
    plot_lineage_bars,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


def _axis(figure: Figure | None) -> Axes:
    assert figure is not None
    assert figure.axes
    return figure.axes[0]


def _bar_heights(axis: Axes) -> list[float]:
    heights: list[float] = []
    for patch in axis.patches:
        assert isinstance(patch, Rectangle)
        heights.append(float(patch.get_height()))
    return heights


def _legend_texts(axis: Axes) -> list[str]:
    legend = axis.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def _line_label(line: Line2D) -> str:
    label = line.get_label()
    assert isinstance(label, str)
    return label


def _is_labeled_line(line: Line2D) -> bool:
    label = _line_label(line)
    return bool(label) and not label.startswith("_")


def _line_xdata(line: Line2D) -> list[float]:
    values = np.asarray(line.get_xdata()).reshape(-1).tolist()
    assert isinstance(values, list)
    return [float(value) for value in values]


# ---------------------------------------------------------------------------
# Fixtures: realistic eval output files
# ---------------------------------------------------------------------------

BASE_EVAL_CSV = """\
Task                               , Accuracy  , Centered
hellaswag                          , 0.280000  , 0.040000
arc_easy                           , 0.350000  , 0.100000
CORE                               ,           , 0.070000
"""


@pytest.fixture()
def base_eval_csv(tmp_path: Path) -> Path:
    """Write a realistic base eval CSV and return its path."""
    p = tmp_path / "base_eval" / "base_000500.csv"
    p.parent.mkdir(parents=True)
    p.write_text(BASE_EVAL_CSV)
    return p


# ---------------------------------------------------------------------------
# Tests: parse_base_eval_csv
# ---------------------------------------------------------------------------


class TestParseBaseEvalCsv:
    def test_parses_task_accuracies(self, base_eval_csv: Path) -> None:
        result = parse_base_eval_csv(base_eval_csv)
        assert result["results"]["hellaswag"] == pytest.approx(0.28)
        assert result["results"]["arc_easy"] == pytest.approx(0.35)

    def test_extracts_core_metric(self, base_eval_csv: Path) -> None:
        result = parse_base_eval_csv(base_eval_csv)
        assert result["core_metric"] == pytest.approx(0.07)

    def test_core_not_in_results(self, base_eval_csv: Path) -> None:
        result = parse_base_eval_csv(base_eval_csv)
        assert "CORE" not in result["results"]

    def test_accuracy_in_valid_range(self, base_eval_csv: Path) -> None:
        result = parse_base_eval_csv(base_eval_csv)
        for task, acc in result["results"].items():
            assert 0.0 <= acc <= 1.0, f"{task} accuracy {acc} out of range"

    def test_missing_core_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "no_core.csv"
        p.write_text(
            "Task                               , Accuracy  , Centered\nfoo                                , 0.5       , 0.1\n"
        )
        with pytest.raises(ValueError, match="CORE"):
            parse_base_eval_csv(p)

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_base_eval_csv(Path("/nonexistent/file.csv"))

    def test_accuracy_out_of_range_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_acc.csv"
        p.write_text(
            "Task                               , Accuracy  , Centered\n"
            "foo                                , 1.5       , 0.1\n"
            "CORE                               ,           , 0.3\n"
        )
        with pytest.raises(ValueError, match="out of.*0.*1"):
            parse_base_eval_csv(p)


# ---------------------------------------------------------------------------
# Fixtures: harmfulness eval output
# ---------------------------------------------------------------------------

HARMFULNESS_YAML: dict[str, Any] = {
    "harmfulness_eval_schema_version": 4,
    "config": {
        "source": "safety",
        "model_tag": "d26",
        "step": 998,
        "judge_model": "gpt-5-nano",
        "feature_conditions": None,
        "num_harmful": 700,
        "num_benign": 500,
    },
    "overall": {
        "mean_score": 2.132,
        "harmfulness_rate": 0.268,
        "safe_response_rate": 0.615,
        "num_scored": 1200,
        "num_failed": 0,
    },
    "harmful_prompts": {
        "mean_score": 2.5,
        "harmfulness_rate": 0.324,
        "safe_response_rate": 0.5,
        "asr": 0.324,
        "num_scored": 700,
        "num_failed": 0,
    },
    "benign_prompts": {
        "mean_score": 1.6,
        "harmfulness_rate": 0.19,
        "safe_response_rate": 0.78,
        "num_scored": 500,
        "num_failed": 0,
        "compliance_rate": 0.72,
        "compliance_num_scored": 500,
        "compliance_num_failed": 0,
    },
    "per_category": {
        "Endangering National Security": 0.02,
        "Insulting Behavior": 0.05,
        "Discriminatory Behavior": 0.03,
        "Endangering Public Health": 0.01,
        "Copyright Issues": 0.01,
        "Violence": 0.08,
        "Drugs": 0.04,
        "Privacy Violation": 0.02,
        "Economic Crime": 0.01,
        "Mental Manipulation": 0.03,
        "Human Trafficking": 0.01,
        "Physical Harm": 0.06,
        "Sexual Content": 0.04,
        "Cybercrime": 0.03,
        "Disrupting Public Order": 0.02,
        "Environmental Damage": 0.01,
        "Psychological Harm": 0.03,
        "White-Collar Crime": 0.01,
        "Animal Abuse": 0.01,
    },
}


@pytest.fixture()
def unsupported_harmfulness_flat_file(tmp_path: Path) -> Path:
    """Write the retired flat harmfulness-results layout."""
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    p = report_dir / "harmfulness_results.yaml"
    with open(p, "w") as f:
        yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    return tmp_path  # return checkpoint_dir


@pytest.fixture()
def harmfulness_condition_dir(tmp_path: Path) -> Path:
    """Write per-condition harmfulness results (new format)."""
    results_dir = tmp_path / "report" / "harmfulness_results"
    results_dir.mkdir(parents=True)
    for label in ["default", "common"]:
        p = results_dir / f"{label}.yaml"
        with open(p, "w") as f:
            yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    return tmp_path  # return checkpoint_dir


@pytest.fixture()
def harmfulness_nested_dir(tmp_path: Path) -> Path:
    """Write nested harmfulness results (e.g. f0.5/m0.5.yaml)."""
    results_dir = tmp_path / "report" / "harmfulness_results"
    results_dir.mkdir(parents=True)
    # Top-level default row
    p = results_dir / "default.yaml"
    with open(p, "w") as f:
        yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    # Nested config
    nested = results_dir / "f0.5"
    nested.mkdir()
    p = nested / "m0.5.yaml"
    with open(p, "w") as f:
        yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    return tmp_path


@pytest.fixture()
def harmfulness_phase_prefixed_dir(tmp_path: Path) -> Path:
    """Write phase-prefixed nested harmfulness results (safety_sft/ and capabilities_sft/)."""
    results_dir = tmp_path / "report" / "harmfulness_results"
    results_dir.mkdir(parents=True)
    # safety_sft configs
    for label in ["f0.1/m0.1", "f0.5/m0.5"]:
        p = results_dir / "safety_sft" / (label + ".yaml")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    # capabilities_sft config
    p = results_dir / "capabilities_sft" / "f0.1" / "m0.1.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: parse_harmfulness_results
# ---------------------------------------------------------------------------


class TestParseHarmfulnessResults:
    @staticmethod
    def _write_payload(tmp_path: Path, payload: object) -> None:
        result_dir = tmp_path / "report" / "harmfulness_results"
        result_dir.mkdir(parents=True)
        with open(result_dir / "default.yaml", "w") as handle:
            yaml.dump(payload, handle)

    def test_flat_file_fails_loudly(self, unsupported_harmfulness_flat_file: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported flat harmfulness result"):
            parse_harmfulness_results(unsupported_harmfulness_flat_file)

    def test_condition_dir_returns_per_label(self, harmfulness_condition_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        assert "default" in result
        assert "common" in result
        assert result["default"]["harmful_prompts"]["asr"] == pytest.approx(0.324)

    @pytest.mark.parametrize("schema_version", [None, 2])
    def test_rejects_missing_or_stale_schema(
        self,
        tmp_path: Path,
        schema_version: int | None,
    ) -> None:
        payload = copy.deepcopy(HARMFULNESS_YAML)
        if schema_version is None:
            del payload["harmfulness_eval_schema_version"]
        else:
            payload["harmfulness_eval_schema_version"] = schema_version
        self._write_payload(tmp_path, payload)

        with pytest.raises(
            ValueError,
            match=r"harmfulness_eval_schema_version must equal 4",
        ):
            parse_harmfulness_results(tmp_path)

    def test_nested_dir_returns_relative_labels(self, harmfulness_nested_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_nested_dir)
        assert "default" in result
        assert "f0.5/m0.5" in result
        assert result["f0.5/m0.5"]["overall"]["mean_score"] == pytest.approx(2.132)

    def test_extracts_asr(self, harmfulness_condition_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        assert result["default"]["harmful_prompts"]["asr"] == pytest.approx(0.324)

    def test_extracts_safe_response_rate(self, harmfulness_condition_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        assert result["default"]["overall"]["safe_response_rate"] == pytest.approx(0.615)

    def test_extracts_benign_compliance_rate(
        self,
        harmfulness_condition_dir: Path,
    ) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        assert result["default"]["benign_prompts"]["compliance_rate"] == pytest.approx(0.72)

    def test_per_category_has_19_entries(self, harmfulness_condition_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        assert len(result["default"]["per_category"]) == 19

    def test_mean_score_in_range(self, harmfulness_condition_dir: Path) -> None:
        result = parse_harmfulness_results(harmfulness_condition_dir)
        score = result["default"]["overall"]["mean_score"]
        assert 1.0 <= score <= 5.0

    def test_no_results_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "report").mkdir()
        result = parse_harmfulness_results(tmp_path)
        assert result == {}

    def test_mean_score_out_of_range_raises(self, tmp_path: Path) -> None:
        bad = dict(HARMFULNESS_YAML)
        bad = {
            **HARMFULNESS_YAML,
            "overall": {**HARMFULNESS_YAML["overall"], "mean_score": 6.0},
        }
        hr_dir = tmp_path / "report" / "harmfulness_results"
        hr_dir.mkdir(parents=True)
        with open(hr_dir / "default.yaml", "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="mean_score.*out of"):
            parse_harmfulness_results(tmp_path)

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        bad = {
            "harmfulness_eval_schema_version": 4,
            "overall": HARMFULNESS_YAML["overall"],
        }  # missing other keys
        hr_dir = tmp_path / "report" / "harmfulness_results"
        hr_dir.mkdir(parents=True)
        with open(hr_dir / "default.yaml", "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="Missing required key"):
            parse_harmfulness_results(tmp_path)

    @pytest.mark.parametrize("section", ["overall", "harmful_prompts", "benign_prompts"])
    def test_metric_sections_must_be_mappings(self, tmp_path: Path, section: str) -> None:
        payload = copy.deepcopy(HARMFULNESS_YAML)
        payload[section] = []
        self._write_payload(tmp_path, payload)
        with pytest.raises(ValueError, match="sections must be mappings"):
            parse_harmfulness_results(tmp_path)

    @pytest.mark.parametrize(
        ("section", "field", "value", "match"),
        [
            ("harmful_prompts", "safe_response_rate", None, "Missing or non-numeric"),
            ("harmful_prompts", "safe_response_rate", True, "Missing or non-numeric"),
            ("benign_prompts", "compliance_rate", "0.5", "Missing or non-numeric"),
            ("harmful_prompts", "safe_response_rate", -0.01, "out of"),
            ("benign_prompts", "compliance_rate", 1.01, "out of"),
        ],
    )
    def test_pool_specific_rates_are_numeric_probabilities(
        self,
        tmp_path: Path,
        section: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        payload = copy.deepcopy(HARMFULNESS_YAML)
        payload[section][field] = value
        self._write_payload(tmp_path, payload)
        with pytest.raises(ValueError, match=match):
            parse_harmfulness_results(tmp_path)

    def test_category_cardinality_must_remain_exact(self, tmp_path: Path) -> None:
        payload = copy.deepcopy(HARMFULNESS_YAML)
        payload["per_category"].pop(next(iter(payload["per_category"])))
        self._write_payload(tmp_path, payload)
        with pytest.raises(ValueError, match="Expected 19 harm categories"):
            parse_harmfulness_results(tmp_path)

    def test_phase_prefixed_returns_exact_labels(
        self,
        harmfulness_phase_prefixed_dir: Path,
    ) -> None:
        """Phase-prefixed nested results produce exact label strings."""
        result = parse_harmfulness_results(harmfulness_phase_prefixed_dir)
        assert len(result) == 3
        expected_labels = {
            "safety_sft/f0.1/m0.1",
            "safety_sft/f0.5/m0.5",
            "capabilities_sft/f0.1/m0.1",
        }
        assert set(result.keys()) == expected_labels
        for label in expected_labels:
            assert result[label]["overall"]["mean_score"] == pytest.approx(2.132)

    def test_phase_prefixed_labels_are_unique(
        self,
        harmfulness_phase_prefixed_dir: Path,
    ) -> None:
        """All labels from phase-prefixed results are unique (no collisions)."""
        result = parse_harmfulness_results(harmfulness_phase_prefixed_dir)
        labels = list(result.keys())
        assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# Fixtures: steering eval output
# ---------------------------------------------------------------------------

STEERING_REPORT_YAML = {
    "metadata": {
        "timestamp": "2026-03-07T12:00:00Z",
        "config_file": "config.yaml",
        "num_features": 2,
    },
    "features": [
        {
            "id": 42,
            "confidence": 4,
            "standard_win_rate": 0.72,
            "dose_response": {
                "magnitude_levels": [0, 1, 2, 3, 4],
                "win_rates": [0.5, 0.55, 0.6, 0.68, 0.72],
                "spearman_rho": 0.95,
                "p_val": 0.01,
                "isotonic_r2": 0.91,
            },
        },
        {
            "id": 99,
            "confidence": 2,
            "standard_win_rate": 0.48,
            "dose_response": {
                "magnitude_levels": [0, 1, 2, 3, 4],
                "win_rates": [0.5, 0.49, 0.51, 0.47, 0.48],
                "spearman_rho": -0.1,
                "p_val": 0.8,
                "isotonic_r2": 0.02,
            },
        },
    ],
    "aggregate": {
        "confidence_weighted_win_rate": 0.65,
        "confidence_weighted_monotonicity": 0.8,
        "fraction_monotonic": 0.5,
        "total_features": 2,
        "high_confidence_features": 1,
    },
}


@pytest.fixture()
def steering_report(tmp_path: Path) -> Path:
    """Write a steering eval report.yaml."""
    p = tmp_path / "report.yaml"
    with open(p, "w") as f:
        yaml.dump(STEERING_REPORT_YAML, f, default_flow_style=False)
    return p


# ---------------------------------------------------------------------------
# Tests: parse_steering_report
# ---------------------------------------------------------------------------


class TestParseSteeringReport:
    def test_extracts_aggregate(self, steering_report: Path) -> None:
        result = parse_steering_report(steering_report)
        assert result["aggregate"]["confidence_weighted_win_rate"] == pytest.approx(0.65)

    def test_extracts_features(self, steering_report: Path) -> None:
        result = parse_steering_report(steering_report)
        assert len(result["features"]) == 2
        assert result["features"][0]["id"] == 42

    def test_dose_response_levels_match_win_rates(self, steering_report: Path) -> None:
        result = parse_steering_report(steering_report)
        for feat in result["features"]:
            dr = feat["dose_response"]
            assert len(dr["magnitude_levels"]) == len(dr["win_rates"])

    def test_win_rates_in_valid_range(self, steering_report: Path) -> None:
        result = parse_steering_report(steering_report)
        for feat in result["features"]:
            for wr in feat["dose_response"]["win_rates"]:
                assert 0.0 <= wr <= 1.0

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_steering_report(Path("/nonexistent/report.yaml"))

    def test_win_rate_out_of_range_raises(self, tmp_path: Path) -> None:
        bad = {
            "aggregate": STEERING_REPORT_YAML["aggregate"],
            "features": [
                {
                    "id": 1,
                    "confidence": 3,
                    "standard_win_rate": 0.6,
                    "dose_response": {
                        "magnitude_levels": [0, 1],
                        "win_rates": [0.5, 1.5],  # out of range
                    },
                }
            ],
        }
        p = tmp_path / "report.yaml"
        with open(p, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="win_rate.*out of"):
            parse_steering_report(p)

    def test_mismatched_levels_and_win_rates_raises(self, tmp_path: Path) -> None:
        bad = {
            "aggregate": STEERING_REPORT_YAML["aggregate"],
            "features": [
                {
                    "id": 1,
                    "confidence": 3,
                    "standard_win_rate": 0.6,
                    "dose_response": {
                        "magnitude_levels": [0, 1, 2],
                        "win_rates": [0.5, 0.6],  # length mismatch
                    },
                }
            ],
        }
        p = tmp_path / "report.yaml"
        with open(p, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="magnitude_levels.*win_rates"):
            parse_steering_report(p)


# ---------------------------------------------------------------------------
# Helpers: mock checkpoint tree
# ---------------------------------------------------------------------------


def _write_base_eval(checkpoint_dir: Path) -> None:
    """Write a minimal base eval CSV at the expected location."""
    eval_dir = checkpoint_dir / "base_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv = eval_dir / "base_000500.csv"
    csv.write_text(BASE_EVAL_CSV)


def _write_harmfulness(checkpoint_dir: Path) -> None:
    """Write a minimal harmfulness result at the expected location."""
    results_dir = checkpoint_dir / "report" / "harmfulness_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / "default.yaml"
    with open(p, "w") as f:
        yaml.dump(HARMFULNESS_YAML, f, default_flow_style=False)


def _build_mock_tree(base_dir: Path) -> None:
    """Build a mock checkpoint tree:
    base_checkpoints/
      d26/
        base_eval/base_000500.csv
        report/harmfulness_results/default.yaml
        safety_sft_checkpoints/
          report/harmfulness_results/default.yaml
          capabilities_sft_checkpoints/
            report/harmfulness_results/default.yaml
      d27/
        base_eval/base_000500.csv
    """
    # d26 base
    d26 = base_dir / "base_checkpoints" / "d26"
    d26.mkdir(parents=True)
    _write_base_eval(d26)
    _write_harmfulness(d26)

    # d26 safety SFT
    d26_safety = d26 / "safety_sft_checkpoints"
    d26_safety.mkdir()
    _write_harmfulness(d26_safety)

    # d26 safety -> capabilities SFT
    d26_safety_cap = d26_safety / "capabilities_sft_checkpoints"
    d26_safety_cap.mkdir()
    _write_harmfulness(d26_safety_cap)

    # d27 base
    d27 = base_dir / "base_checkpoints" / "d27"
    d27.mkdir(parents=True)
    _write_base_eval(d27)


# ---------------------------------------------------------------------------
# Tests: discover_model_tree
# ---------------------------------------------------------------------------


class TestDiscoverModelTree:
    def test_discovers_model_tags(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        tags = {n.model_tag for n in nodes}
        assert tags == {"d26", "d27"}

    def test_d26_has_base_eval(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d26 = next(n for n in nodes if n.model_tag == "d26")
        assert "base_eval" in d26.eval_results

    def test_d26_has_harmfulness(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d26 = next(n for n in nodes if n.model_tag == "d26")
        assert "harmfulness" in d26.eval_results

    def test_d26_has_children(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d26 = next(n for n in nodes if n.model_tag == "d26")
        child_chains = {c.source_chain for c in d26.children}
        assert "base/safety" in child_chains

    def test_d26_safety_has_capabilities_child(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d26 = next(n for n in nodes if n.model_tag == "d26")
        safety = next(c for c in d26.children if c.source_chain == "base/safety")
        cap_chains = {c.source_chain for c in safety.children}
        assert "base/safety/capabilities" in cap_chains

    def test_d27_has_no_children(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d27 = next(n for n in nodes if n.model_tag == "d27")
        assert d27.children == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "base_checkpoints").mkdir()
        nodes = discover_model_tree(tmp_path)
        assert nodes == []

    def test_discovers_encdec_checkpoints(self, tmp_path: Path) -> None:
        # conditional_encdec_checkpoints/enc1/s64/t10/ with base eval
        leaf = tmp_path / "conditional_encdec_checkpoints" / "enc1" / "s64" / "t10"
        eval_dir = leaf / "base_eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "results.csv").write_text(BASE_EVAL_CSV)
        nodes = discover_model_tree(tmp_path)
        assert len(nodes) == 1
        assert nodes[0].model_tag == "enc1"
        assert nodes[0].source_chain == "encdec"
        assert "base_eval" in nodes[0].eval_results


# ---------------------------------------------------------------------------
# Tests: collect_lineages
# ---------------------------------------------------------------------------


class TestCollectLineages:
    def test_d26_lineages(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d26 = next(n for n in nodes if n.model_tag == "d26")
        lineages = collect_lineages(d26)
        # Should have two lineages: base->safety and base->safety->capabilities
        assert len(lineages) >= 2
        # Longest lineage should be 3 nodes
        longest = max(lineages, key=len)
        assert len(longest) == 3

    def test_leaf_node_gives_single_lineage(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        nodes = discover_model_tree(tmp_path)
        d27 = next(n for n in nodes if n.model_tag == "d27")
        lineages = collect_lineages(d27)
        assert len(lineages) == 1
        assert len(lineages[0]) == 1


# ---------------------------------------------------------------------------
# Tests: plot_lineage_bars
# ---------------------------------------------------------------------------


class TestPlotLineageBars:
    def _make_lineage(self, tmp_path: Path) -> list[ModelNode]:
        """Build a 3-node lineage with harmfulness results."""
        nodes = []
        for i, label in enumerate(["base", "base/safety", "base/safety/capabilities"]):
            d = tmp_path / f"node_{i}"
            d.mkdir()
            nodes.append(
                ModelNode(
                    source_chain=label,
                    model_tag="d26",
                    checkpoint_dir=d,
                    eval_results={
                        "harmfulness": {
                            "default": {
                                "overall": {
                                    "mean_score": 2.0 + i * 0.3,
                                    "harmfulness_rate": 0.3 - i * 0.05,
                                    "safe_response_rate": 0.5 + i * 0.1,
                                    "num_scored": 1000,
                                    "num_failed": 0,
                                },
                                "harmful_prompts": {
                                    "asr": 0.3 - i * 0.05,
                                    "mean_score": 2.5,
                                    "harmfulness_rate": 0.3,
                                    "safe_response_rate": 0.4,
                                    "num_scored": 700,
                                    "num_failed": 0,
                                },
                                "benign_prompts": {
                                    "mean_score": 1.5,
                                    "harmfulness_rate": 0.1,
                                    "safe_response_rate": 0.8,
                                    "num_scored": 300,
                                    "num_failed": 0,
                                },
                                "per_category": {},
                            },
                        },
                    },
                )
            )
        return nodes

    def test_returns_figure(self, tmp_path: Path) -> None:
        lineage = self._make_lineage(tmp_path)
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        assert isinstance(fig, Figure)

    def test_has_correct_number_of_bars(self, tmp_path: Path) -> None:
        lineage = self._make_lineage(tmp_path)
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        assert len(ax.patches) == 3  # one bar per node

    def test_bar_heights_match_data(self, tmp_path: Path) -> None:
        lineage = self._make_lineage(tmp_path)
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        heights = _bar_heights(ax)
        assert heights[0] == pytest.approx(0.3)
        assert heights[1] == pytest.approx(0.25)
        assert heights[2] == pytest.approx(0.2)

    def test_x_labels_are_source_chains(self, tmp_path: Path) -> None:
        lineage = self._make_lineage(tmp_path)
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["base", "base/safety", "base/safety/capabilities"]

    def test_has_title(self, tmp_path: Path) -> None:
        lineage = self._make_lineage(tmp_path)
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        assert ax.get_title() != ""

    def test_skips_nodes_without_eval(self, tmp_path: Path) -> None:
        """Nodes missing the requested eval type should be skipped."""
        lineage = self._make_lineage(tmp_path)
        lineage[1].eval_results = {}  # remove evals from middle node
        fig = plot_lineage_bars(
            lineage,
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        assert len(ax.patches) == 2  # only 2 bars

    def test_base_eval_core_metric(self, tmp_path: Path) -> None:
        """Test with base_eval type and core_metric."""
        nodes = []
        for i, label in enumerate(["base", "base/safety"]):
            d = tmp_path / f"core_node_{i}"
            d.mkdir()
            nodes.append(
                ModelNode(
                    source_chain=label,
                    model_tag="d26",
                    checkpoint_dir=d,
                    eval_results={"base_eval": {"results": {}, "core_metric": 0.05 + i * 0.02}},
                )
            )
        fig = plot_lineage_bars(nodes, eval_type="base_eval", metric_key="core_metric")
        ax = _axis(fig)
        heights = _bar_heights(ax)
        assert heights[0] == pytest.approx(0.05)
        assert heights[1] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Tests: plot_cross_family_bars
# ---------------------------------------------------------------------------


class TestPlotCrossFamilyBars:
    def _make_roots(self, tmp_path: Path) -> list[ModelNode]:
        """Build two root nodes (d26 and d27), each with 2 variants."""
        roots = []
        for tag in ("d26", "d27"):
            base_dir = tmp_path / tag / "base"
            base_dir.mkdir(parents=True)
            safety_dir = tmp_path / tag / "safety"
            safety_dir.mkdir(parents=True)

            safety_node = ModelNode(
                source_chain="base/safety",
                model_tag=tag,
                checkpoint_dir=safety_dir,
                eval_results={
                    "harmfulness": {
                        "default": {
                            "overall": {
                                "mean_score": 2.0,
                                "harmfulness_rate": 0.2,
                                "safe_response_rate": 0.7,
                                "num_scored": 1000,
                                "num_failed": 0,
                            },
                            "harmful_prompts": {
                                "asr": 0.25 if tag == "d26" else 0.22,
                                "mean_score": 2.5,
                                "harmfulness_rate": 0.3,
                                "safe_response_rate": 0.4,
                                "num_scored": 700,
                                "num_failed": 0,
                            },
                            "benign_prompts": {
                                "mean_score": 1.5,
                                "harmfulness_rate": 0.1,
                                "safe_response_rate": 0.8,
                                "num_scored": 300,
                                "num_failed": 0,
                            },
                            "per_category": {},
                        }
                    }
                },
            )
            base_node = ModelNode(
                source_chain="base",
                model_tag=tag,
                checkpoint_dir=base_dir,
                eval_results={
                    "harmfulness": {
                        "default": {
                            "overall": {
                                "mean_score": 3.0,
                                "harmfulness_rate": 0.5,
                                "safe_response_rate": 0.3,
                                "num_scored": 1000,
                                "num_failed": 0,
                            },
                            "harmful_prompts": {
                                "asr": 0.55 if tag == "d26" else 0.50,
                                "mean_score": 3.5,
                                "harmfulness_rate": 0.5,
                                "safe_response_rate": 0.2,
                                "num_scored": 700,
                                "num_failed": 0,
                            },
                            "benign_prompts": {
                                "mean_score": 2.0,
                                "harmfulness_rate": 0.3,
                                "safe_response_rate": 0.5,
                                "num_scored": 300,
                                "num_failed": 0,
                            },
                            "per_category": {},
                        }
                    }
                },
                children=[safety_node],
            )
            roots.append(base_node)
        return roots

    def test_returns_figure(self, tmp_path: Path) -> None:
        roots = self._make_roots(tmp_path)
        fig = plot_cross_family_bars(
            roots, eval_type="harmfulness", metric_key="asr", condition_label="default"
        )
        assert isinstance(fig, Figure)

    def test_has_groups_for_each_family(self, tmp_path: Path) -> None:
        roots = self._make_roots(tmp_path)
        fig = plot_cross_family_bars(
            roots, eval_type="harmfulness", metric_key="asr", condition_label="default"
        )
        ax = _axis(fig)
        # 2 families x 2 variants = 4 bars
        assert len(ax.patches) == 4

    def test_has_legend(self, tmp_path: Path) -> None:
        roots = self._make_roots(tmp_path)
        fig = plot_cross_family_bars(
            roots, eval_type="harmfulness", metric_key="asr", condition_label="default"
        )
        ax = _axis(fig)
        legend = ax.get_legend()
        assert legend is not None

    def test_single_family_works(self, tmp_path: Path) -> None:
        roots = self._make_roots(tmp_path)[:1]
        fig = plot_cross_family_bars(
            roots, eval_type="harmfulness", metric_key="asr", condition_label="default"
        )
        assert isinstance(fig, Figure)

    def test_separates_dec_and_encdec_groups(self, tmp_path: Path) -> None:
        """Base and encdec roots with same model_tag should appear as
        separate x-axis groups (dec/d26 vs encdec/d26), not merged."""
        d = tmp_path / "sep"
        d.mkdir()
        root_base = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d,
            eval_results={
                "harmfulness": {
                    "default": {
                        "overall": {
                            "mean_score": 3.0,
                            "harmfulness_rate": 0.5,
                            "safe_response_rate": 0.3,
                            "num_scored": 1000,
                            "num_failed": 0,
                        },
                        "harmful_prompts": {
                            "asr": 0.55,
                            "mean_score": 3.5,
                            "harmfulness_rate": 0.5,
                            "safe_response_rate": 0.2,
                            "num_scored": 700,
                            "num_failed": 0,
                        },
                        "benign_prompts": {
                            "mean_score": 2.0,
                            "harmfulness_rate": 0.3,
                            "safe_response_rate": 0.5,
                            "num_scored": 300,
                            "num_failed": 0,
                        },
                        "per_category": {},
                    }
                }
            },
        )
        root_encdec = ModelNode(
            source_chain="encdec",
            model_tag="d26",
            checkpoint_dir=d,
            eval_results={
                "harmfulness": {
                    "default": {
                        "overall": {
                            "mean_score": 2.0,
                            "harmfulness_rate": 0.2,
                            "safe_response_rate": 0.7,
                            "num_scored": 1000,
                            "num_failed": 0,
                        },
                        "harmful_prompts": {
                            "asr": 0.20,
                            "mean_score": 2.5,
                            "harmfulness_rate": 0.2,
                            "safe_response_rate": 0.6,
                            "num_scored": 700,
                            "num_failed": 0,
                        },
                        "benign_prompts": {
                            "mean_score": 1.5,
                            "harmfulness_rate": 0.1,
                            "safe_response_rate": 0.8,
                            "num_scored": 300,
                            "num_failed": 0,
                        },
                        "per_category": {},
                    }
                }
            },
        )
        fig = plot_cross_family_bars(
            [root_base, root_encdec],
            eval_type="harmfulness",
            metric_key="asr",
            condition_label="default",
        )
        ax = _axis(fig)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        # Should have 2 separate groups, not 1 merged group
        assert len(labels) == 2
        assert "dec/d26" in labels
        assert "encdec/d26" in labels
        # "encdec" variant is normalised to "base", so there's only 1 variant
        # with 2 groups = 2 patches total — both blue "base" bars.
        heights = _bar_heights(ax)
        assert len(heights) == 2
        assert heights[0] == pytest.approx(0.55)  # base in dec/d26
        assert heights[1] == pytest.approx(0.20)  # base in encdec/d26


# ---------------------------------------------------------------------------
# Tests: plot_cross_family_harmfulness
# ---------------------------------------------------------------------------


def _make_harmfulness_section(asr: float = 0.40) -> dict[str, Any]:
    """Build a minimal harmfulness section dict with the given ASR."""
    return {
        "overall": {
            "mean_score": 3.0,
            "harmfulness_rate": 0.5,
            "safe_response_rate": 0.3,
            "num_scored": 1000,
            "num_failed": 0,
        },
        "harmful_prompts": {
            "asr": asr,
            "mean_score": 3.5,
            "harmfulness_rate": 0.5,
            "safe_response_rate": 0.2,
            "num_scored": 700,
            "num_failed": 0,
        },
        "benign_prompts": {
            "mean_score": 2.0,
            "harmfulness_rate": 0.3,
            "safe_response_rate": 0.5,
            "num_scored": 300,
            "num_failed": 0,
            "compliance_rate": 0.75,
        },
        "per_category": {},
    }


class TestPlotCrossFamilyHarmfulness:
    def _make_roots_with_variants(self, tmp_path: Path) -> list[ModelNode]:
        """Build a realistic tree: dec/d26 with base→safety→capabilities
        (all default rows) and encdec/d26 with explicit conditioning configs."""
        d = tmp_path / "cf_harm"
        d.mkdir()

        # dec/d26: base → safety → capabilities (all default rows)
        dec_cap = ModelNode(
            source_chain="base/safety/capabilities",
            model_tag="d26",
            checkpoint_dir=d / "dec_cap",
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.20)}},
        )
        dec_safety = ModelNode(
            source_chain="base/safety",
            model_tag="d26",
            checkpoint_dir=d / "dec_safety",
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.35)}},
            children=[dec_cap],
        )
        dec_d26 = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d / "dec_d26",
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.55)}},
            children=[dec_safety],
        )

        # encdec/d26: conditioning configs (common excluded from bars)
        encdec_d26 = ModelNode(
            source_chain="encdec",
            model_tag="d26",
            checkpoint_dir=d / "encdec_d26",
            eval_results={
                "harmfulness": {
                    "common": _make_harmfulness_section(asr=0.25),
                    "safety_sft/f0.1/m0.5": _make_harmfulness_section(asr=0.18),
                    "safety_sft/f0.1/m0.9": _make_harmfulness_section(asr=0.22),
                }
            },
        )
        return [dec_d26, encdec_d26]

    def _make_roots_simple(self, tmp_path: Path) -> list[ModelNode]:
        """Build a simple tree with decoder defaults and encdec conditioning."""
        d = tmp_path / "cf_simple"
        d.mkdir(exist_ok=True)
        dec_d26 = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d / "a",
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.55)}},
        )
        dec_d27 = ModelNode(
            source_chain="base",
            model_tag="d27",
            checkpoint_dir=d / "b",
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.50)}},
        )
        encdec_d26 = ModelNode(
            source_chain="encdec",
            model_tag="d26",
            checkpoint_dir=d / "c",
            eval_results={
                "harmfulness": {
                    "safety_sft/f0.1/m0.5": _make_harmfulness_section(asr=0.20),
                    "safety_sft/f0.1/m0.9": _make_harmfulness_section(asr=0.25),
                }
            },
        )
        return [dec_d26, dec_d27, encdec_d26]

    def test_returns_figure(self, tmp_path: Path) -> None:
        fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), "asr")
        assert isinstance(fig, Figure)

    def test_x_groups_are_families(self, tmp_path: Path) -> None:
        fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), "asr")
        ax = _axis(fig)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["dec/d26", "dec/d27", "encdec/d26"]

    def test_legend_has_variant_and_conditioning_labels(self, tmp_path: Path) -> None:
        """Decoder defaults use variant names; encdec uses dataset_conditioning keys."""
        fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), "asr")
        ax = _axis(fig)
        legend_texts = _legend_texts(ax)
        assert "base" in legend_texts
        assert "base/safety_sft_conditioning" in legend_texts
        # "common" should NOT appear as a bar
        assert "common" not in legend_texts

    def test_encdec_has_optimal_conditioning_bar(self, tmp_path: Path) -> None:
        """encdec/d26 gets only a best-case conditioning bar."""
        fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), "asr")
        ax = _axis(fig)
        # dec/d26: 1 bar, dec/d27: 1 bar, encdec/d26: 1 conditioning bar.
        heights = _bar_heights(ax)
        assert len(heights) == 3
        assert heights[0] == pytest.approx(0.55)  # dec/d26 base
        assert heights[1] == pytest.approx(0.50)  # dec/d27 base
        # Best-case ASR (min) = 0.20 from safety_sft/f0.1/m0.5
        assert heights[2] == pytest.approx(0.20)

    def test_decoder_variants_shown_as_separate_bars(self, tmp_path: Path) -> None:
        """When dec/d26 has base, safety, and capabilities variants (all
        default rows), each should appear as its own bar."""
        fig = plot_cross_family_harmfulness(self._make_roots_with_variants(tmp_path), "asr")
        ax = _axis(fig)
        legend_texts = _legend_texts(ax)
        assert "base" in legend_texts
        assert "base/safety" in legend_texts
        assert "base/safety/capabilities" in legend_texts
        # conditioning bar should appear
        assert "base/safety_sft_conditioning" in legend_texts
        # "common" excluded
        assert "common" not in legend_texts

    def test_decoder_variant_values_are_correct(self, tmp_path: Path) -> None:
        """Verify the actual bar heights for dec variant progression."""
        roots = self._make_roots_with_variants(tmp_path)
        fig = plot_cross_family_harmfulness(roots, "asr")
        ax = _axis(fig)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["dec/d26", "encdec/d26"]

        heights = _bar_heights(ax)
        # dec/d26: 3 bars (base, base/safety, base/safety/capabilities)
        # encdec/d26: 1 bar (safety_sft_conditioning best-case)
        assert len(heights) == 4
        assert heights[0] == pytest.approx(0.55)
        assert heights[1] == pytest.approx(0.35)
        assert heights[2] == pytest.approx(0.20)
        # Best ASR (min) = min(0.18, 0.22) = 0.18
        assert heights[3] == pytest.approx(0.18)

    def test_has_legend(self, tmp_path: Path) -> None:
        fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), "asr")
        ax = _axis(fig)
        assert ax.get_legend() is not None

    def test_single_family_returns_none(self, tmp_path: Path) -> None:
        """A single decoder root with one bar should return None (< 2 data points)."""
        d = tmp_path / "single"
        d.mkdir()
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d,
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.55)}},
        )
        fig = plot_cross_family_harmfulness([root], "asr")
        assert fig is None

    def test_other_metrics_work(self, tmp_path: Path) -> None:
        """mean_score and safe_response_rate should also be extractable."""
        for metric in ("mean_score", "safe_response_rate"):
            fig = plot_cross_family_harmfulness(self._make_roots_simple(tmp_path), metric)
            assert isinstance(fig, Figure)
            ax = _axis(fig)
            heights = _bar_heights(ax)
            assert any(h > 0 for h in heights)

    def test_compliance_label_describes_non_refusal(self, tmp_path: Path) -> None:
        fig = plot_cross_family_harmfulness(
            self._make_roots_simple(tmp_path),
            "compliance_rate",
        )

        ax = _axis(fig)
        assert ax.get_ylabel() == "Benign Compliance Rate (non-refusal)"
        assert "non-refusal" in ax.get_title()

    def test_default_condition_uses_variant_key(self, tmp_path: Path) -> None:
        """Condition labelled 'default' should use the variant name."""
        d = tmp_path / "default_cond"
        d.mkdir()
        roots = [
            ModelNode(
                source_chain="base",
                model_tag="d26",
                checkpoint_dir=d / "a",
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.40)}},
            ),
            ModelNode(
                source_chain="base",
                model_tag="d27",
                checkpoint_dir=d / "b",
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.45)}},
            ),
        ]
        fig = plot_cross_family_harmfulness(roots, "asr")
        assert fig is not None
        ax = _axis(fig)
        legend_texts = _legend_texts(ax)
        assert "base" in legend_texts
        assert "default" not in legend_texts

    def test_no_data_returns_none(self, tmp_path: Path) -> None:
        """Roots with no harmfulness data should return None."""
        d = tmp_path / "empty"
        d.mkdir()
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d,
            eval_results={},
        )
        fig = plot_cross_family_harmfulness([root], "asr")
        assert fig is None

    def test_common_excluded_from_bars(self, tmp_path: Path) -> None:
        """The 'common' condition should not appear as a bar in the chart."""
        d = tmp_path / "common_excl"
        d.mkdir()
        roots = [
            ModelNode(
                source_chain="base",
                model_tag="d26",
                checkpoint_dir=d / "a",
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.55)}},
            ),
            ModelNode(
                source_chain="encdec",
                model_tag="d26",
                checkpoint_dir=d / "b",
                eval_results={
                    "harmfulness": {
                        "common": _make_harmfulness_section(asr=0.25),
                        "safety_sft/f0.1/m0.5": _make_harmfulness_section(asr=0.20),
                    }
                },
            ),
        ]
        fig = plot_cross_family_harmfulness(roots, "asr")
        assert fig is not None
        ax = _axis(fig)
        legend_texts = _legend_texts(ax)
        assert "common" not in legend_texts

    def test_safe_response_rate_uses_max_optimal(self, tmp_path: Path) -> None:
        """For safe_response_rate, optimal = max (most safe responses)."""
        d = tmp_path / "srr_optimal"
        d.mkdir()
        roots = [
            ModelNode(
                source_chain="base",
                model_tag="d26",
                checkpoint_dir=d / "a",
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.55)}},
            ),
            ModelNode(
                source_chain="encdec",
                model_tag="d26",
                checkpoint_dir=d / "b",
                eval_results={
                    "harmfulness": {
                        "safety_sft/f0.1/m0.5": {
                            "overall": {
                                "mean_score": 2.0,
                                "harmfulness_rate": 0.2,
                                "safe_response_rate": 0.7,
                                "num_scored": 1000,
                                "num_failed": 0,
                            },
                            "harmful_prompts": {
                                "asr": 0.2,
                                "mean_score": 2.5,
                                "harmfulness_rate": 0.2,
                                "safe_response_rate": 0.6,
                                "num_scored": 700,
                                "num_failed": 0,
                            },
                            "benign_prompts": {
                                "mean_score": 1.5,
                                "harmfulness_rate": 0.1,
                                "safe_response_rate": 0.8,
                                "num_scored": 300,
                                "num_failed": 0,
                            },
                            "per_category": {},
                        },
                        "safety_sft/f0.1/m0.9": {
                            "overall": {
                                "mean_score": 2.5,
                                "harmfulness_rate": 0.3,
                                "safe_response_rate": 0.5,
                                "num_scored": 1000,
                                "num_failed": 0,
                            },
                            "harmful_prompts": {
                                "asr": 0.3,
                                "mean_score": 3.0,
                                "harmfulness_rate": 0.3,
                                "safe_response_rate": 0.4,
                                "num_scored": 700,
                                "num_failed": 0,
                            },
                            "benign_prompts": {
                                "mean_score": 2.0,
                                "harmfulness_rate": 0.2,
                                "safe_response_rate": 0.6,
                                "num_scored": 300,
                                "num_failed": 0,
                            },
                            "per_category": {},
                        },
                    }
                },
            ),
        ]
        fig = plot_cross_family_harmfulness(roots, "safe_response_rate")
        assert fig is not None
        ax = _axis(fig)
        heights = _bar_heights(ax)
        # _extract_harmfulness_value checks harmful_prompts first
        # dec: safe_response_rate from harmful_prompts = 0.2
        # encdec best conditioning: max(0.6, 0.4) = 0.6
        assert heights[0] == pytest.approx(0.2)  # dec
        assert heights[1] == pytest.approx(0.6)  # best conditioning


# ---------------------------------------------------------------------------
# Tests: plot_dose_response_overlay
# ---------------------------------------------------------------------------


class TestPlotDoseResponseOverlay:
    def _make_steering_nodes(self, tmp_path: Path) -> list[ModelNode]:
        """Build nodes with steering eval results for overlay comparison."""
        nodes = []
        for i, (tag, chain) in enumerate(
            [("d26", "base"), ("d26", "base/safety"), ("d27", "base")]
        ):
            d = tmp_path / f"steer_{i}"
            d.mkdir()
            nodes.append(
                ModelNode(
                    source_chain=chain,
                    model_tag=tag,
                    checkpoint_dir=d,
                    eval_results={
                        "steering": {
                            "aggregate": {
                                "confidence_weighted_win_rate": 0.6 + i * 0.05,
                                "confidence_weighted_monotonicity": 0.7,
                                "fraction_monotonic": 0.5,
                            },
                            "features": [
                                {
                                    "id": 42,
                                    "confidence": 4,
                                    "standard_win_rate": 0.6 + i * 0.05,
                                    "dose_response": {
                                        "magnitude_levels": [0, 1, 2, 3, 4],
                                        "win_rates": [
                                            0.5,
                                            0.52 + i * 0.02,
                                            0.55 + i * 0.03,
                                            0.6 + i * 0.04,
                                            0.65 + i * 0.05,
                                        ],
                                        "spearman_rho": 0.9,
                                        "p_val": 0.01,
                                        "isotonic_r2": 0.85,
                                    },
                                },
                            ],
                        }
                    },
                )
            )
        return nodes

    def test_returns_figure(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        fig = plot_dose_response_overlay(nodes, feature_id=42)
        assert isinstance(fig, Figure)

    def test_has_one_line_per_node(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        fig = plot_dose_response_overlay(nodes, feature_id=42)
        ax = _axis(fig)
        labeled_lines = [line for line in ax.get_lines() if _is_labeled_line(line)]
        assert len(labeled_lines) == 3

    def test_has_legend(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        fig = plot_dose_response_overlay(nodes, feature_id=42)
        ax = _axis(fig)
        legend = ax.get_legend()
        assert legend is not None

    def test_x_axis_is_magnitude_levels(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        fig = plot_dose_response_overlay(nodes, feature_id=42)
        ax = _axis(fig)
        # Get a labeled (data) line, not the axhline
        data_lines = [line for line in ax.get_lines() if _is_labeled_line(line)]
        line = data_lines[0]
        assert _line_xdata(line) == [0, 1, 2, 3, 4]

    def test_skips_nodes_without_feature(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        # Remove the feature from the last node
        nodes[2].eval_results["steering"]["features"] = []
        fig = plot_dose_response_overlay(nodes, feature_id=42)
        ax = _axis(fig)
        labeled_lines = [line for line in ax.get_lines() if _is_labeled_line(line)]
        assert len(labeled_lines) == 2  # only 2 nodes have feature 42

    def test_nonexistent_feature_gives_empty_plot(self, tmp_path: Path) -> None:
        nodes = self._make_steering_nodes(tmp_path)
        fig = plot_dose_response_overlay(nodes, feature_id=9999)
        ax = _axis(fig)
        labeled_lines = [line for line in ax.get_lines() if _is_labeled_line(line)]
        assert len(labeled_lines) == 0


# ---------------------------------------------------------------------------
# Tests: generate_all_plots
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:
    def test_creates_lineage_plots(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        # d26 has harmfulness at base->safety->capabilities lineage
        d26_dir = output_dir / "d26" / "harmfulness"
        assert d26_dir.is_dir()
        pngs = list(d26_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_creates_comparison_plots(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        comp_dir = output_dir / "comparison"
        assert comp_dir.is_dir()
        pngs = list(comp_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_output_files_are_valid_pngs(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        for png in output_dir.rglob("*.png"):
            assert png.stat().st_size > 0
            with open(png, "rb") as f:
                assert f.read(4) == b"\x89PNG"

    def test_empty_tree_no_crash(self, tmp_path: Path) -> None:
        (tmp_path / "base_checkpoints").mkdir()
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        # Should succeed without crashing, may or may not create output_dir

    def test_base_eval_comparison_created(self, tmp_path: Path) -> None:
        _build_mock_tree(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        # d26 and d27 both have base_eval
        comp_dir = output_dir / "comparison"
        base_eval_pngs = list(comp_dir.glob("base_eval_*.png"))
        assert len(base_eval_pngs) >= 1


# ---------------------------------------------------------------------------
# Helpers: steering eval in report/steering_eval/ subdirectory
# ---------------------------------------------------------------------------


def _write_steering_report(checkpoint_dir: Path, *, subdir: bool = False) -> None:
    """Write a steering eval report.yaml.

    If subdir=True, writes to report/steering_eval/report.yaml (new layout).
    If subdir=False, writes to report.yaml (legacy layout).
    """
    if subdir:
        report_dir = checkpoint_dir / "report" / "steering_eval"
    else:
        report_dir = checkpoint_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    p = report_dir / "report.yaml"
    with open(p, "w") as f:
        yaml.dump(STEERING_REPORT_YAML, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Tests: _probe_eval_results steering path discovery
# ---------------------------------------------------------------------------


class TestProbeEvalResultsSteeringPaths:
    """Verify _probe_eval_results finds steering reports in both locations."""

    def test_finds_steering_at_root_report_yaml(self, tmp_path: Path) -> None:
        """Legacy layout: report.yaml at checkpoint root."""
        _write_steering_report(tmp_path, subdir=False)
        results = _probe_eval_results(tmp_path)
        assert "steering" in results
        assert results["steering"]["aggregate"]["confidence_weighted_win_rate"] == pytest.approx(
            0.65
        )

    def test_finds_steering_in_report_steering_eval_subdir(self, tmp_path: Path) -> None:
        """New layout: report/steering_eval/report.yaml."""
        _write_steering_report(tmp_path, subdir=True)
        results = _probe_eval_results(tmp_path)
        assert "steering" in results
        assert results["steering"]["aggregate"]["confidence_weighted_win_rate"] == pytest.approx(
            0.65
        )

    def test_root_report_yaml_takes_precedence(self, tmp_path: Path) -> None:
        """If both exist, root report.yaml should be used (legacy compat)."""
        _write_steering_report(tmp_path, subdir=False)
        _write_steering_report(tmp_path, subdir=True)
        results = _probe_eval_results(tmp_path)
        assert "steering" in results

    def test_no_steering_report_returns_no_steering_key(self, tmp_path: Path) -> None:
        """No steering report at either location."""
        tmp_path.mkdir(exist_ok=True)
        results = _probe_eval_results(tmp_path)
        assert "steering" not in results

    def test_steering_subdir_extracts_features(self, tmp_path: Path) -> None:
        """Verify full parse from subdir path, not just detection."""
        _write_steering_report(tmp_path, subdir=True)
        results = _probe_eval_results(tmp_path)
        features = results["steering"]["features"]
        assert len(features) == 2
        assert features[0]["id"] == 42
        assert features[0]["dose_response"]["win_rates"] == [0.5, 0.55, 0.6, 0.68, 0.72]

    def test_steering_subdir_coexists_with_harmfulness(self, tmp_path: Path) -> None:
        """Steering in subdir + harmfulness results should both be found."""
        _write_steering_report(tmp_path, subdir=True)
        _write_harmfulness(tmp_path)
        results = _probe_eval_results(tmp_path)
        assert "steering" in results
        assert "harmfulness" in results


# ---------------------------------------------------------------------------
# Tests: _find_encdec_leaf depth preference
# ---------------------------------------------------------------------------


class TestFindEncdecLeafDepth:
    """Verify _find_encdec_leaf finds the deepest directory with content."""

    def test_finds_deep_leaf_over_shallow_report_dir(self, tmp_path: Path) -> None:
        """If tag_dir/report/ exists (raw responses) but the real eval
        content is deeper at tag_dir/s128/t2048.9/report/steering_eval/,
        the leaf should be the deeper directory."""
        tag_dir = tmp_path / "enc1"
        # Shallow: tag_dir/report/ with raw response files (no scored results)
        raw_report = tag_dir / "report"
        raw_report.mkdir(parents=True)
        (raw_report / "harmfulness_results_responses_safety_delta.yaml").write_text("- raw data")
        # Deep: tag_dir/s128/t2048.9/ with actual eval content
        deep = tag_dir / "s128" / "t2048.9"
        _write_steering_report(deep, subdir=True)

        leaf = _find_encdec_leaf(tag_dir)
        assert leaf == deep

    def test_finds_deep_leaf_with_base_eval(self, tmp_path: Path) -> None:
        """Deep directory with base_eval/ should be preferred over
        shallow directory with only a report/ dir."""
        tag_dir = tmp_path / "enc1"
        # Shallow report dir
        (tag_dir / "report").mkdir(parents=True)
        (tag_dir / "report" / "raw.yaml").write_text("raw")
        # Deep base_eval
        deep = tag_dir / "s64" / "t100"
        eval_dir = deep / "base_eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "results.csv").write_text(BASE_EVAL_CSV)

        leaf = _find_encdec_leaf(tag_dir)
        assert leaf == deep

    def test_returns_tag_dir_when_it_has_actual_eval_and_no_deeper(self, tmp_path: Path) -> None:
        """If tag_dir itself has base_eval and nothing deeper, return it."""
        tag_dir = tmp_path / "enc1"
        eval_dir = tag_dir / "base_eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "results.csv").write_text(BASE_EVAL_CSV)

        leaf = _find_encdec_leaf(tag_dir)
        assert leaf == tag_dir

    def test_returns_none_for_empty_tree(self, tmp_path: Path) -> None:
        tag_dir = tmp_path / "enc1"
        tag_dir.mkdir()
        leaf = _find_encdec_leaf(tag_dir)
        assert leaf is None

    def test_finds_deepest_with_sft_children(self, tmp_path: Path) -> None:
        """Deep directory with SFT child dirs should be found."""
        tag_dir = tmp_path / "enc1"
        # Shallow report dir (decoy)
        (tag_dir / "report").mkdir(parents=True)
        # Deep directory with SFT children
        deep = tag_dir / "s128" / "t500"
        (deep / "safety_sft_checkpoints").mkdir(parents=True)

        leaf = _find_encdec_leaf(tag_dir)
        assert leaf == deep


# ---------------------------------------------------------------------------
# Tests: end-to-end encdec steering discovery and plotting
# ---------------------------------------------------------------------------


class TestEncdecSteeringEndToEnd:
    """Integration tests ensuring encdec steering evals are discovered and plotted."""

    def _build_encdec_tree_with_steering(self, base_dir: Path) -> None:
        """Build a realistic encdec tree matching the actual on-disk layout:
        conditional_encdec_checkpoints/d26/
          report/                    <-- raw responses (decoy)
            harmfulness_results_responses_safety_delta.yaml
          s128/t2048.9/
            report/steering_eval/    <-- actual scored steering eval
              report.yaml
        """
        tag_dir = base_dir / "conditional_encdec_checkpoints" / "d26"
        # Shallow decoy
        raw_report = tag_dir / "report"
        raw_report.mkdir(parents=True)
        (raw_report / "harmfulness_results_responses_safety_delta.yaml").write_text(
            "- category: test\n  pool: harmful\n  prompt: test\n  response: test\n"
        )
        # Deep steering eval
        deep = tag_dir / "s128" / "t2048.9"
        _write_steering_report(deep, subdir=True)

    def test_discover_finds_encdec_steering(self, tmp_path: Path) -> None:
        self._build_encdec_tree_with_steering(tmp_path)
        roots = discover_model_tree(tmp_path)
        assert len(roots) == 1
        node = roots[0]
        assert node.model_tag == "d26"
        assert node.source_chain == "encdec"
        assert "steering" in node.eval_results

    def test_discover_encdec_steering_has_correct_features(self, tmp_path: Path) -> None:
        self._build_encdec_tree_with_steering(tmp_path)
        roots = discover_model_tree(tmp_path)
        steering = roots[0].eval_results["steering"]
        assert len(steering["features"]) == 2
        assert steering["features"][0]["id"] == 42
        assert steering["aggregate"]["confidence_weighted_win_rate"] == pytest.approx(0.65)

    def test_generate_plots_creates_steering_dose_response(self, tmp_path: Path) -> None:
        """Full pipeline: encdec with steering eval should produce dose-response PNGs."""
        self._build_encdec_tree_with_steering(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        comp_dir = output_dir / "comparison"
        assert comp_dir.is_dir()
        dose_response_pngs = list(comp_dir.glob("steering_dose_response_*.png"))
        assert len(dose_response_pngs) >= 1
        # Verify they're valid PNGs
        for png in dose_response_pngs:
            with open(png, "rb") as f:
                assert f.read(4) == b"\x89PNG"

    def test_generate_plots_skips_single_point_steering_comparison(self, tmp_path: Path) -> None:
        """A single encdec root with steering has < 2 data points, so no comparison plot."""
        self._build_encdec_tree_with_steering(tmp_path)
        output_dir = tmp_path / "fig"
        generate_all_plots(base_dir=tmp_path, output_dir=output_dir)
        comp_dir = output_dir / "comparison"
        steering_pngs = list(comp_dir.glob("steering_confidence_weighted_win_rate*.png"))
        assert len(steering_pngs) == 0

    def test_mixed_tree_base_and_encdec_steering(self, tmp_path: Path) -> None:
        """Both base checkpoints and encdec with steering should coexist."""
        _build_mock_tree(tmp_path)
        self._build_encdec_tree_with_steering(tmp_path)
        roots = discover_model_tree(tmp_path)
        tags = {r.model_tag for r in roots}
        # d26 appears in both base and encdec
        assert "d26" in tags
        assert "d27" in tags
        # At least one root should have steering
        steering_roots = [r for r in roots if "steering" in r.eval_results]
        assert len(steering_roots) >= 1


# ---------------------------------------------------------------------------
# Tests: parse_condition_label
# ---------------------------------------------------------------------------


class TestParseConditionLabel:
    def test_parses_safety_sft_label(self) -> None:
        result = parse_condition_label("safety_sft/f0.1/m0.5")
        assert result == ParsedCondition(
            dataset="safety_sft", feature_ratio="f0.1", magnitude="m0.5"
        )

    def test_parses_capabilities_sft_label(self) -> None:
        result = parse_condition_label("capabilities_sft/f0.5/m0.9")
        assert result is not None
        assert result.dataset == "capabilities_sft"
        assert result.feature_ratio == "f0.5"
        assert result.magnitude == "m0.9"

    def test_returns_none_for_unknown_label(self) -> None:
        assert parse_condition_label("not_a_condition") is None

    def test_returns_none_for_common(self) -> None:
        assert parse_condition_label("common") is None

    def test_returns_none_for_default(self) -> None:
        assert parse_condition_label("default") is None

    def test_returns_none_for_partial_label(self) -> None:
        assert parse_condition_label("safety_sft/f0.1") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_condition_label("") is None


# ---------------------------------------------------------------------------
# Tests: _extract_harmfulness_value
# ---------------------------------------------------------------------------


class TestExtractHarmfulnessValue:
    def test_extracts_from_harmful_prompts(self) -> None:
        data = _make_harmfulness_section(asr=0.42)
        assert _extract_harmfulness_value(data, "asr") == pytest.approx(0.42)

    def test_falls_through_to_overall(self) -> None:
        """Keys present in harmful_prompts are found there first;
        keys only in overall are found via fallback."""
        data = _make_harmfulness_section()
        # harmfulness_rate exists in harmful_prompts (0.5) and overall (0.5)
        # but num_scored only exists in each section with different values
        # Just verify the search order: harmful_prompts → overall → benign_prompts
        assert _extract_harmfulness_value(data, "harmfulness_rate") == pytest.approx(0.5)

    def test_returns_none_for_missing_key(self) -> None:
        data = _make_harmfulness_section()
        assert _extract_harmfulness_value(data, "nonexistent") is None

    def test_explicit_pool_paths_ignore_divergent_duplicate_names(self) -> None:
        data = _make_harmfulness_section()
        data["harmful_prompts"]["safe_response_rate"] = 0.11
        data["overall"]["safe_response_rate"] = 0.22
        data["benign_prompts"]["safe_response_rate"] = 0.33
        data["benign_prompts"]["compliance_rate"] = 0.77
        data["harmful_prompts"]["compliance_rate"] = 0.01
        assert _extract_harmfulness_value(data, "safe_response_rate") == pytest.approx(0.11)
        assert _extract_harmfulness_value(data, "compliance_rate") == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Tests: _optimal_value
# ---------------------------------------------------------------------------


class TestOptimalValue:
    def test_min_for_asr(self) -> None:
        assert _optimal_value([0.3, 0.1, 0.5], "asr") == pytest.approx(0.1)

    def test_min_for_mean_score(self) -> None:
        assert _optimal_value([3.0, 1.5, 2.0], "mean_score") == pytest.approx(1.5)

    def test_max_for_safe_response_rate(self) -> None:
        assert _optimal_value([0.3, 0.7, 0.5], "safe_response_rate") == pytest.approx(0.7)

    def test_max_for_compliance_rate(self) -> None:
        assert _optimal_value([0.3, 0.7, 0.5], "compliance_rate") == pytest.approx(0.7)

    def test_unknown_metric_defaults_to_min(self) -> None:
        assert _optimal_value([0.3, 0.1, 0.5], "unknown_metric") == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: skip single-bar plots
# ---------------------------------------------------------------------------


class TestSkipSingleBarPlots:
    def test_lineage_bars_returns_none_for_single_node(self, tmp_path: Path) -> None:
        """A single-node lineage should return None."""
        node = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=tmp_path,
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.5)}},
        )
        fig = plot_lineage_bars([node], "harmfulness", "asr", condition_label="default")
        assert fig is None

    def test_lineage_bars_returns_figure_for_two_nodes(self, tmp_path: Path) -> None:
        """A two-node lineage with data should return Figure."""
        nodes = [
            ModelNode(
                source_chain="base",
                model_tag="d26",
                checkpoint_dir=tmp_path,
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.5)}},
            ),
            ModelNode(
                source_chain="base/safety",
                model_tag="d26",
                checkpoint_dir=tmp_path,
                eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.3)}},
            ),
        ]
        fig = plot_lineage_bars(nodes, "harmfulness", "asr", condition_label="default")
        assert isinstance(fig, Figure)

    def test_cross_family_bars_returns_none_for_single_point(self, tmp_path: Path) -> None:
        """A single root with one data point should return None."""
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=tmp_path,
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.5)}},
        )
        fig = plot_cross_family_bars([root], "harmfulness", "asr", condition_label="default")
        assert fig is None


# ---------------------------------------------------------------------------
# Tests: plot_cross_family_bars filtering
# ---------------------------------------------------------------------------


class TestCrossFamilyBarsFiltering:
    def test_filters_empty_groups(self, tmp_path: Path) -> None:
        """Groups with no data for the metric should be filtered out."""
        roots = [
            ModelNode(
                source_chain="base",
                model_tag="d26",
                checkpoint_dir=tmp_path,
                eval_results={"base_eval": {"results": {}, "core_metric": 0.07}},
            ),
            ModelNode(
                source_chain="base",
                model_tag="d27",
                checkpoint_dir=tmp_path,
                eval_results={"base_eval": {"results": {}, "core_metric": 0.09}},
            ),
            ModelNode(
                source_chain="base",
                model_tag="d28",
                checkpoint_dir=tmp_path,
                eval_results={},
            ),  # no data
        ]
        fig = plot_cross_family_bars(roots, "base_eval", "core_metric")
        assert fig is not None
        ax = _axis(fig)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "dec/d28" not in labels
        assert "dec/d26" in labels
        assert "dec/d27" in labels

    def test_returns_none_when_all_empty(self, tmp_path: Path) -> None:
        """When no groups have data, return None."""
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=tmp_path,
            eval_results={},
        )
        fig = plot_cross_family_bars([root], "base_eval", "core_metric")
        assert fig is None

    def test_returns_none_for_single_data_point(self, tmp_path: Path) -> None:
        """One total data point is not enough for a comparison chart."""
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=tmp_path,
            eval_results={"base_eval": {"results": {}, "core_metric": 0.07}},
        )
        fig = plot_cross_family_bars([root], "base_eval", "core_metric")
        assert fig is None


# ---------------------------------------------------------------------------
# Tests: plot_encdec_conditioning_sweep
# ---------------------------------------------------------------------------


class TestPlotEncdecConditioningSweep:
    def _make_encdec_roots(self, tmp_path: Path) -> list[ModelNode]:
        d = tmp_path / "sweep"
        d.mkdir(exist_ok=True)
        return [
            ModelNode(
                source_chain="encdec",
                model_tag="d26",
                checkpoint_dir=d,
                eval_results={
                    "harmfulness": {
                        "common": _make_harmfulness_section(asr=0.40),
                        "safety_sft/f0.1/m0.1": _make_harmfulness_section(asr=0.35),
                        "safety_sft/f0.1/m0.5": _make_harmfulness_section(asr=0.25),
                        "safety_sft/f0.1/m0.9": _make_harmfulness_section(asr=0.20),
                        "safety_sft/f0.5/m0.1": _make_harmfulness_section(asr=0.30),
                        "safety_sft/f0.5/m0.5": _make_harmfulness_section(asr=0.22),
                        "safety_sft/f0.5/m0.9": _make_harmfulness_section(asr=0.18),
                    }
                },
            ),
        ]

    def test_returns_figure_with_sweep_data(self, tmp_path: Path) -> None:
        roots = self._make_encdec_roots(tmp_path)
        fig = plot_encdec_conditioning_sweep(roots, "asr")
        assert isinstance(fig, Figure)

    def test_returns_none_for_decoder_only(self, tmp_path: Path) -> None:
        d = tmp_path / "dec_only"
        d.mkdir()
        root = ModelNode(
            source_chain="base",
            model_tag="d26",
            checkpoint_dir=d,
            eval_results={"harmfulness": {"default": _make_harmfulness_section(asr=0.5)}},
        )
        fig = plot_encdec_conditioning_sweep([root], "asr")
        assert fig is None

    def test_has_correct_number_of_lines(self, tmp_path: Path) -> None:
        roots = self._make_encdec_roots(tmp_path)
        fig = plot_encdec_conditioning_sweep(roots, "asr")
        ax = _axis(fig)
        data_lines = [line for line in ax.get_lines() if _is_labeled_line(line)]
        # 2 sweep lines: safety_sft/f0.1, safety_sft/f0.5
        # 1 baseline: common
        assert len(data_lines) == 3  # 2 lines + 1 axhline with label

    def test_x_axis_is_magnitude(self, tmp_path: Path) -> None:
        roots = self._make_encdec_roots(tmp_path)
        fig = plot_encdec_conditioning_sweep(roots, "asr")
        ax = _axis(fig)
        # Find the first data line (not axhline)
        data_lines = [
            line for line in ax.get_lines() if _is_labeled_line(line) and len(_line_xdata(line)) > 1
        ]
        assert len(data_lines) >= 1
        x_vals = sorted(_line_xdata(data_lines[0]))
        assert x_vals == pytest.approx([0.1, 0.5, 0.9])

    def test_returns_none_with_no_encdec_roots(self, tmp_path: Path) -> None:
        fig = plot_encdec_conditioning_sweep([], "asr")
        assert fig is None

    def test_baselines_shown_as_horizontal_lines(self, tmp_path: Path) -> None:
        roots = self._make_encdec_roots(tmp_path)
        fig = plot_encdec_conditioning_sweep(roots, "asr")
        ax = _axis(fig)
        labels = [_line_label(line) for line in ax.get_lines() if _line_label(line)]
        baseline_labels = [label for label in labels if "common" in label]
        assert len(baseline_labels) == 1
