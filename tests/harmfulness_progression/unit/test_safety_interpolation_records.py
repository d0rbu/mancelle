"""Dense receipt discovery and explicit conditioned progression selection."""

from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from scripts.eval import plot_harmfulness_progression as plot_cli
from scripts.eval import plot_harmfulness_standalone_progression as standalone_cli
from steering.conditional.sampling import SELFCOND_ACTION_RANKING_POLICY
from steering.eval.harmfulness_generation import STRENGTH_SWEEP_SPEC_SHA256_KEY
from steering.eval.harmfulness_progression import (
    DEFAULT_CONFIG_LABEL_MAP,
    METRIC_SPECS,
    harmfulness_plot_signals,
    load_harmfulness_records,
    load_progression_data,
    load_safety_progression_runs,
    render_progression_chart,
)
from tests.harmfulness_progression.unit.test_load_progression_data import _seed
from tests.harmfulness_progression.unit.test_parse_harmfulness_yaml import _well_formed


def _completion(strength, digest="a" * 64):
    return {"self_cond_segment_size": 32, "strength": strength,
            "action_ranking_policy": SELFCOND_ACTION_RANKING_POLICY,
            "safety_steering": {"targets_sha256": digest, "strength": strength}}


def _write(tmp_path, strength, digest="a" * 64, *, source="base"):
    path = tmp_path / "harmfulness_results" / f"{strength}-{digest[:8]}.yaml"
    path.parent.mkdir(exist_ok=True)
    data = _well_formed(feature_configs="safety_steered", source=source)
    data["config"]["generation"] = {"completion": _completion(strength, digest)}
    path.write_text(yaml.safe_dump(data))
    return path


def test_dense_records_are_discovered_and_each_strength_gets_a_plot_signal(tmp_path):
    _write(tmp_path, 0)
    _write(tmp_path, 2.5)
    records = load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert {record.signal for record in records} == {
        "Safety interpolation (strength=0)", "Safety interpolation (strength=2.5)"}
    order, styles = harmfulness_plot_signals(records)
    assert all(record.signal in order and record.signal in styles for record in records)


def test_filter_matches_full_bundle_digest_and_strength(tmp_path):
    _write(tmp_path, 0)
    _write(tmp_path, 2.5)
    _write(tmp_path, 2.5, "b" * 64)
    options = {"model_tag": "d12_baseline", "num_harmful": 700, "config_label_map": DEFAULT_CONFIG_LABEL_MAP}
    assert len(load_harmfulness_records(tmp_path, **options,
        safety_steering={"targets_sha256": "a" * 64, "strength": 2.5})) == 1
    assert load_harmfulness_records(tmp_path, **options,
        safety_steering={"targets_sha256": "c" * 64, "strength": 2.5}) == []


def test_different_targets_in_one_plot_cell_fail_with_both_artifact_paths(tmp_path):
    first = _write(tmp_path, 2.5)
    second = _write(tmp_path, 2.5, "b" * 64)
    with pytest.raises(ValueError, match="Conflicting safety targets") as exc:
        load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    message = str(exc.value)
    assert str(first) in message and str(second) in message
    assert "a" * 64 in message and "b" * 64 in message
    assert "safety_steering receipt filter" in message


def test_different_stages_can_use_different_targets_on_one_strength_series(tmp_path):
    sources = {"base/safety_cond", "base/safety_cond/capabilities_cond"}
    _write(tmp_path, 2.5, source="base/safety_cond")
    _write(tmp_path, 2.5, "b" * 64, source="base/safety_cond/capabilities_cond")
    records = load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert len(records) == 2 and {record.source for record in records} == sources
    assert {record.signal for record in records} == {"Safety interpolation (strength=2.5)"}


@pytest.mark.parametrize("strength", [True, -1, 6.5, "2.5", float("nan")])
def test_invalid_dense_receipt_refused(tmp_path, strength):
    _write(tmp_path, strength)
    with pytest.raises(ValueError, match="strength"):
        load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP)


def test_nested_self_conditioned_seeds_remain_readable_with_completion_receipts(tmp_path):
    seed = _seed(asr=0.2, safe_rate=0.7, mean_score=2.0)
    seed["completion"] = _completion(2.5)
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"checkpoints": [{"phase": "safety_sft", "step": 1116,
        "evals": {"harmfulness": {"conditioned": {"configs": {"self_conditioned": {
            "kind": "self_conditioned", "cache_dir": "/cache", "seeds": {"42": seed}}}}}}}]}))
    points = load_progression_data(path, conditioned_config="self_conditioned")
    assert len(points) == 1 and points[0].asr == 0.2
    with pytest.raises(ValueError, match="missing conditioned config"):
        load_progression_data(path, conditioned_config="wrong")


SWEEP_STRENGTHS = (0, 3, 6)


def _progression_run(tmp_path, strength):
    checkpoints = []
    for index, (phase, step) in enumerate((("s32", 1116), ("s32mask", 1125))):
        seeds = {}
        for seed_id, offset in (("42", 0), ("43", 0.1)):
            seed = _seed(asr=0.2 + strength / 20 + offset + index / 10,
                         safe_rate=0.8 - strength / 20 - offset,
                         mean_score=2 + strength / 10 + offset,
                         compliance_rate=0.5 + strength / 20 + offset)
            seed["completion"] = _completion(strength, ("a" if index == 0 else "b") * 64)
            seed["progression_provenance"] = {
                STRENGTH_SWEEP_SPEC_SHA256_KEY: hashlib.sha256(f"{index}/{seed_id}".encode()).hexdigest(),
            }
            seeds[seed_id] = seed
        checkpoints.append({"phase": phase, "step": step, "checkpoint_identity": {"sha256": str(index) * 64},
            "evals": {"harmfulness": {"conditioned": {"configs": {"self_conditioned": {"seeds": seeds}}}}}})
    path = tmp_path / f"strength-{strength}.json"
    path.write_text(json.dumps({"model_tag": "fixture", "checkpoints": checkpoints}))
    return path


@pytest.mark.parametrize("spec", METRIC_SPECS, ids=lambda spec: spec.suffix)
def test_three_strength_chart_from_progression_receipts(tmp_path, spec):
    paths = [_progression_run(tmp_path, strength) for strength in SWEEP_STRENGTHS]
    records, stages = load_safety_progression_runs(paths, model_tag="fixture")
    order, styles = harmfulness_plot_signals(records)
    fig = render_progression_chart(records, source_to_stage=stages, stage_labels=tuple(stages),
        signal_order=order, signal_styles=styles, metric=spec.metric, ylabel=spec.ylabel, title="Fixture sweep")
    try:
        lines = [line for line in fig.axes[0].lines if line.get_label() in {record.signal for record in records}]
        assert len(lines) == 3
        assert len({line.get_color() for line in lines}) == 3
        for line, strength in zip(lines, SWEEP_STRENGTHS, strict=True):
            assert line.get_label() == f"Safety interpolation (strength={strength})"
            assert np.asarray(line.get_xdata()).tolist() == [0, 1]
            expected = {"asr": [0.25 + strength / 20, 0.35 + strength / 20],
                        "safe_rate": [0.75 - strength / 20] * 2,
                        "mean_score": [2.05 + strength / 10] * 2,
                        "compliance_rate": [0.55 + strength / 20] * 2}[spec.metric]
            assert np.asarray(line.get_ydata()).tolist() == pytest.approx(expected)
        assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == list(stages)
        fig.savefig(tmp_path / f"fixture-{spec.suffix}.png")
    finally:
        plt.close(fig)


def test_plot_cli_renders_all_metrics_from_three_progression_files(tmp_path):
    paths = [_progression_run(tmp_path, strength) for strength in SWEEP_STRENGTHS]
    cfg = OmegaConf.create({"base_dir": str(tmp_path), "output_dir": str(tmp_path / "figures"),
                           "model_tag": "fixture", "progression_paths": [str(path) for path in paths]})
    plot_cli._main(cfg)
    assert {p.name for p in (tmp_path / "figures").glob("*.png")} == {
        f"conditioned_{spec.suffix}.png" for spec in METRIC_SPECS}
    cfg.safety_steering = {"targets_sha256": "a" * 64, "strength": 3}
    with pytest.raises(ValueError, match="filter must be null"):
        plot_cli._main(cfg)


@pytest.mark.parametrize("fault", ["model", "checkpoint", "missing_identity", "cells", "missing_receipt", "mixed_seeds",
                                 "duplicate_cell", "duplicate_checkpoint", "unjudged", "malformed_metrics"])
def test_sweep_refuses_ambiguous_or_incomplete_receipts(tmp_path, fault):
    paths = [_progression_run(tmp_path, strength) for strength in (0, 3)]
    data = json.loads(paths[1].read_text())
    checkpoint = data["checkpoints"][0]
    seed = checkpoint["evals"]["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]["42"]
    if fault == "model":
        data["model_tag"] = "other"
    elif fault == "checkpoint":
        checkpoint["checkpoint_identity"]["sha256"] = "c" * 64
    elif fault == "missing_identity":
        del checkpoint["checkpoint_identity"]
    elif fault == "cells":
        data["checkpoints"].pop()
    elif fault == "missing_receipt":
        del seed["completion"]
    elif fault == "mixed_seeds":
        seed["completion"]["safety_steering"]["strength"] = 6
        seed["completion"]["strength"] = 6
    elif fault == "duplicate_cell":
        paths[1] = paths[0]
        data = json.loads(paths[0].read_text())
    elif fault == "duplicate_checkpoint":
        data["checkpoints"].append(checkpoint)
    elif fault == "unjudged":
        data["checkpoints"] = []
    else:
        seed["harmful_prompts"]["asr"] = 5
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_safety_progression_runs(paths, model_tag="fixture")


@pytest.mark.parametrize("strengths", [(2.5000001, 2.5000002), (0.0000010000001, 0.0000010000002)])
def test_nearby_strengths_remain_distinct_plot_cells(tmp_path, strengths):
    for strength in strengths:
        _write(tmp_path, strength)
    records = load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert len({record.signal for record in records}) == 2
    paths = [_progression_run(tmp_path, strength) for strength in strengths]
    sweep, _ = load_safety_progression_runs(paths, model_tag="fixture")
    assert len(sweep) == 4
    assert len({record.signal for record in sweep}) == 2


def test_sweep_refuses_changed_target_between_strengths_for_same_checkpoint(tmp_path):
    paths = [_progression_run(tmp_path, strength) for strength in (0, 3)]
    data = json.loads(paths[1].read_text())
    seeds = data["checkpoints"][0]["evals"]["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]
    for seed in seeds.values():
        seed["completion"]["safety_steering"]["targets_sha256"] = "e" * 64
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="safety targets disagree"):
        load_safety_progression_runs(paths, model_tag="fixture")


@pytest.mark.parametrize("fault", ["missing_spec", "different_spec", "seed_set", "runtime", "strength"])
def test_sweep_refuses_uncontrolled_comparisons(tmp_path, fault):
    paths = [_progression_run(tmp_path, strength) for strength in (0, 3)]
    data = json.loads(paths[1].read_text())
    seeds = data["checkpoints"][0]["evals"]["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]
    seed = seeds["42"]
    if fault == "missing_spec":
        del seed["progression_provenance"]
    elif fault == "different_spec":
        seed["progression_provenance"][STRENGTH_SWEEP_SPEC_SHA256_KEY] = "e" * 64
    elif fault == "seed_set":
        seeds.pop("43")
    elif fault == "runtime":
        seed["completion"]["self_cond_segment_size"] *= 2
    else:
        for seed in seeds.values():
            seed["completion"]["safety_steering"]["strength"] = 6
            seed["completion"]["strength"] = 6
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="strength sweep"):
        load_safety_progression_runs(paths, model_tag="fixture")


@pytest.mark.parametrize("completion", [{}, [], {"safety_steering": None},
    {**_completion(3), "strength": 2}, {**_completion(3), "action_ranking_policy": "wrong"}])
def test_plot_discovery_checks_entire_completion_contract(tmp_path, completion):
    path = _write(tmp_path, 3)
    data = yaml.safe_load(path.read_text())
    data["config"]["generation"]["completion"] = completion
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="completion|safety_steering"):
        load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP)


def test_unsteered_completion_gets_self_conditioned_signal(tmp_path):
    path = _write(tmp_path, 0)
    data = yaml.safe_load(path.read_text())
    data["config"]["generation"]["completion"]["safety_steering"] = None
    path.write_text(yaml.safe_dump(data))
    records = load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert [record.signal for record in records] == ["Self-conditioned"]


@pytest.mark.parametrize("receipt", [{}, {"targets_sha256": "a" * 64},
    {"targets_sha256": "a" * 64, "strength": 7}])
def test_invalid_plot_filter_fails_even_without_input_files(tmp_path, receipt):
    with pytest.raises(ValueError, match="safety_steering"):
        load_harmfulness_records(tmp_path, model_tag="fixture", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP, safety_steering=receipt)


def test_empty_strength_sweep_is_not_a_valid_blank_plot():
    with pytest.raises(ValueError, match="at least one progression file"):
        load_safety_progression_runs([], model_tag="fixture")


@pytest.mark.parametrize("conditioned", [None, [], {}, {"configs": []},
    {"configs": {"self_conditioned": None}}, {"configs": {"self_conditioned": {}}},
    {"configs": {"self_conditioned": {"seeds": []}}}])
def test_malformed_conditioned_progression_has_actionable_error(tmp_path, conditioned):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps({"checkpoints": [{"phase": "safety_sft", "step": 1,
        "evals": {"harmfulness": {"conditioned": conditioned}}}]}))
    with pytest.raises(ValueError, match="conditioned.*(configs|seeds)"):
        load_progression_data(path, conditioned_config="self_conditioned")


def test_dense_duplicate_runs_cannot_silently_replace_metrics(tmp_path):
    first = _write(tmp_path, 3)
    second = first.with_name("repeated-run.yaml")
    data = yaml.safe_load(first.read_text())
    data["harmful_prompts"]["asr"] = 0.9
    second.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Duplicate safety results") as error:
        load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert str(first) in str(error.value) and str(second) in str(error.value)


@pytest.mark.parametrize("generation", [[], "invalid"])
def test_malformed_generation_block_fails_with_artifact_path(tmp_path, generation):
    path = _write(tmp_path, 3)
    data = yaml.safe_load(path.read_text())
    data["config"]["generation"] = generation
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="config.generation") as error:
        load_harmfulness_records(tmp_path, model_tag="d12_baseline", num_harmful=700,
            config_label_map=DEFAULT_CONFIG_LABEL_MAP)
    assert str(path) in str(error.value)


def test_standalone_cli_combines_conditioned_progression_and_dense_signals(tmp_path):
    _write(tmp_path, 2.5, source="base/safety_cond")
    decoder = tmp_path / "decoder.json"
    seed = _seed(asr=0.3, safe_rate=0.7, mean_score=2)
    decoder.write_text(json.dumps({"checkpoints": [{"phase": "pretrain", "step": 53000,
        "evals": {"harmfulness": {"seeds": {"42": seed}}}}]}))
    conditioned = tmp_path / "conditioned.json"
    conditioned.write_text(json.dumps({"checkpoints": [{"phase": "pretrain", "step": 53000,
        "evals": {"harmfulness": {"conditioned": {"configs": {
            "self_conditioned": {"seeds": {"42": seed}}}}}}}]}))
    cfg = OmegaConf.create({"base_dir": str(tmp_path), "output_dir": str(tmp_path / "standalone"),
        "model_tag": "d12_baseline", "num_harmful": 700, "d13_progression_path": str(decoder),
        "d12_progression_path": str(conditioned), "min_step": 0,
        "d12_conditioned_config": "self_conditioned", "last_pretrain_step": 53000})
    standalone_cli.main.__wrapped__(cfg)
    assert {path.name for path in (tmp_path / "standalone").glob("*.png")} == {
        f"progression_{spec.suffix}.png" for spec in METRIC_SPECS}
    for path in (tmp_path / "standalone").glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_yaml_metrics_and_receipt_share_one_snapshot(tmp_path, monkeypatch):
    path = _write(tmp_path, 3)
    load = yaml.safe_load

    def read_then_replace(stream):
        data = load(stream)
        replacement = load(path.read_text())
        replacement["harmful_prompts"]["asr"] = 0.9
        replacement["config"]["generation"]["completion"] = _completion(6)
        path.write_text(yaml.safe_dump(replacement))
        return data

    monkeypatch.setattr(yaml, "safe_load", read_then_replace)
    records = load_harmfulness_records(
        tmp_path, model_tag="d12_baseline", num_harmful=700,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP,
    )
    assert [(record.signal, record.metrics.asr) for record in records] == [
        ("Safety interpolation (strength=3)", 0.3)
    ]


def test_progression_metrics_and_receipts_share_one_snapshot(tmp_path, monkeypatch):
    path = _progression_run(tmp_path, 3)
    load = json.loads

    def read_then_replace(text, **kwargs):
        data = load(text, **kwargs)
        replacement = load(path.read_text())
        for checkpoint in replacement["checkpoints"]:
            seeds = checkpoint["evals"]["harmfulness"]["conditioned"]["configs"]["self_conditioned"]["seeds"]
            for seed in seeds.values():
                seed["harmful_prompts"]["asr"] = 0.9
                seed["completion"] = _completion(6)
        path.write_text(json.dumps(replacement))
        return data

    monkeypatch.setattr(json, "loads", read_then_replace)
    records, _ = load_safety_progression_runs([path], model_tag="fixture")
    assert {record.signal for record in records} == {"Safety interpolation (strength=3)"}
    assert [record.metrics.asr for record in records] == pytest.approx([0.4, 0.5])
