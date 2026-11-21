"""Exercise complete matrix admission and judged three-strength progression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from omegaconf import OmegaConf

from scripts.eval import campaign2_progression as progression
from scripts.utils.campaign2 import Stage, install_json_once
from steering.eval.campaign2 import Variant, safety_record_receipt, strength_leaf
from steering.eval.chat_eval_result_identity import semantic_digest
from steering.eval.harmfulness_artifact import build_harmfulness_payload
from steering.eval.harmfulness_judge import (
    ComplianceJudgeResult,
    JudgeResult,
    compute_compliance_metrics,
    compute_metrics,
)
from steering.eval.harmfulness_progression import (
    METRIC_SPECS,
    HarmfulnessMetric,
    harmfulness_plot_signals,
    render_progression_chart,
)

ROOT = Path(__file__).resolve().parents[2]
STRENGTHS = (0, 3, 6)


@pytest.fixture
def matrix(tmp_path: Path) -> tuple[dict[str, str], Path]:
    repo = tmp_path / "matrix repo"
    binary = repo / "bin"
    binary.mkdir(parents=True)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    logger = f"""#!{sys.executable}
import json
import os
import sys
args = sys.argv[1:]
if args and args[0] == '-c':
    if os.environ.get('STRENGTH_STDOUT_NOISE'):
        print('cache warning: falling back to a writable cache', flush=True)
    if os.environ.get('MALFORM_STRENGTH_RESULT'):
        with open(args[2], 'w') as stream:
            stream.write('not-the-strength-protocol\\n')
        sys.exit(0)
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ['MATRIX_LOG'], 'a') as stream:
    stream.write(json.dumps({{'args': args, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}}) + '\\n')
if 'mode=preflight' in args and os.environ.get('FAIL_CELL'):
    stage, strength = os.environ['FAIL_CELL'].split(':')
    if f'stage={{stage}}' in args and f'safety.strength={{strength}}' in args:
        sys.exit(2)
"""
    python.write_text(logger)
    python.chmod(0o755)
    (binary / "sbatch").write_text(logger)
    (binary / "sbatch").chmod(0o755)
    git = binary / "git"
    git.write_text('#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo pinned; fi\n')
    git.chmod(0o755)
    local = repo / "scripts/slurm/campaign2/eval_local.sh"
    local.parent.mkdir(parents=True)
    local.write_text('#!/bin/bash\nexec "$CAMPAIGN_REPO/.venv/bin/python" -m scripts.eval.campaign2 "$@"\n')
    log = tmp_path / "matrix.jsonl"
    env = {**os.environ, "CAMPAIGN_REPO": str(repo), "EXPECTED_GIT_REVISION": "pinned",
           "PATH": str(binary) + os.pathsep + os.environ["PATH"], "MATRIX_LOG": str(log),
           "EVAL_POOL": "priority", "EVAL_VARIANT": "encdec_safety",
           "EVAL_OUTPUT_ROOT": str(tmp_path / "outputs"), "CUDA_VISIBLE_DEVICES": "7", "PYTHONPATH": str(ROOT)}
    env.pop("EVAL_SAFETY_STRENGTHS", None)
    for index, stage in enumerate(Stage):
        prefix = f"EVAL_{stage.value.upper()}_"
        checkpoint = tmp_path / stage.value / "checkpoint"
        checkpoint.mkdir(parents=True)
        statistics, target, cache = checkpoint.parent / "statistics", checkpoint.parent / "targets", checkpoint.parent / "cache"
        statistics.write_bytes(b"statistics")
        target.write_bytes(b"target")
        cache.mkdir()
        env.update({prefix + "CHECKPOINT": str(checkpoint), prefix + "STEP": str(index + 1),
                    prefix + "STATISTICS": str(statistics), prefix + "TARGETS": str(target),
                    prefix + "PROMPT_CACHE": str(cache)})
    return env, log


def run_matrix(matrix: tuple[dict[str, str], Path], *args: str) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    env, path = matrix
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/campaign2/eval_matrix.sh"), *args],
                            env=env, text=True, capture_output=True, check=False)
    calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return result, calls


@pytest.mark.parametrize("pool", ["local", "priority", "preemptable"])
def test_matrix_preflights_all_nine_before_launch_with_exact_stage_artifacts(matrix, pool: str) -> None:
    env, _ = matrix
    env["EVAL_POOL"] = pool
    result, calls = run_matrix(matrix, "max_new_tokens=17")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 18
    assert all(call["args"][-1] == "mode=preflight" and call["cuda"] == "" for call in calls[:9])
    assert all("mode=preflight" not in call["args"] and call["cuda"] == "7" for call in calls[9:])
    for index, (stage, strength) in enumerate((stage, strength) for stage in Stage for strength in STRENGTHS):
        prefix = f"EVAL_{stage.value.upper()}_"
        for call in (calls[index], calls[index + 9]):
            args = call["args"]
            assert f"stage={stage.value}" in args and f"safety.strength={strength}" in args
            assert f"checkpoint_dir={env[prefix + 'CHECKPOINT']}" in args
            assert f"step={env[prefix + 'STEP']}" in args
            assert f"statistics={env[prefix + 'STATISTICS']}" in args
            assert f"safety.targets={env[prefix + 'TARGETS']}" in args
            assert f"conditioning.context_cache_dir={env[prefix + 'PROMPT_CACHE']}" in args
            assert f"output_dir={env['EVAL_OUTPUT_ROOT']}/{stage.value}/encdec_safety" in args
            assert "max_new_tokens=17" in args
        if pool == "priority":
            assert "mit_normal_gpu" in calls[index + 9]["args"]
            assert "--account=mit_amf_advanced_gpu" in calls[index + 9]["args"]


@pytest.mark.parametrize("failure", [f"{stage.value}:{strength}" for stage in Stage for strength in STRENGTHS])
def test_any_failed_preflight_prevents_every_submission(matrix, failure: str) -> None:
    matrix[0]["FAIL_CELL"] = failure
    result, calls = run_matrix(matrix)
    assert result.returncode != 0 and calls
    assert all(call["args"][-1] == "mode=preflight" for call in calls)


@pytest.mark.parametrize("field", ["CHECKPOINT", "STEP", "STATISTICS", "TARGETS", "PROMPT_CACHE"])
def test_missing_last_stage_input_prevents_preflight_or_submission(matrix, field: str) -> None:
    matrix[0].pop(f"EVAL_CAPABILITIES_{field}")
    result, calls = run_matrix(matrix)
    assert result.returncode != 0 and calls == []


def test_matrix_strength_override_controls_cells(matrix) -> None:
    matrix[0]["EVAL_SAFETY_STRENGTHS"] = "0 2.5"
    result, calls = run_matrix(matrix)
    assert result.returncode == 0 and len(calls) == 12
    assert {arg for call in calls for arg in call["args"] if arg.startswith("safety.strength=")} == {
        "safety.strength=0", "safety.strength=2.5"}


def test_matrix_ignores_stray_python_stdout_when_reading_strengths(matrix) -> None:
    matrix[0]["STRENGTH_STDOUT_NOISE"] = "1"
    result, calls = run_matrix(matrix)
    assert result.returncode == 0 and len(calls) == 18
    assert "cache warning" in result.stdout


def test_matrix_rejects_malformed_strength_result_file(matrix) -> None:
    matrix[0]["MALFORM_STRENGTH_RESULT"] = "1"
    result, calls = run_matrix(matrix)
    assert result.returncode == 2 and calls == []
    assert "malformed Campaign 2 strength result" in result.stderr


@pytest.mark.parametrize("values", ["", "0 0.0", "nan", "inf", "-1", "7", "invalid"])
def test_matrix_invalid_strength_list_fails_without_submissions(matrix, values: str) -> None:
    matrix[0]["EVAL_SAFETY_STRENGTHS"] = values
    result, calls = run_matrix(matrix)
    assert result.returncode != 0 and calls == []


@pytest.mark.parametrize("variant", [
    Variant.ENCDEC, Variant.PYTHIA, Variant.PYTHIA_DIFFMEAN, Variant.PYTHIA_PROMPT,
])
def test_unsteered_matrix_remains_three_cells_without_target_or_cache_inputs(
    matrix, variant: Variant,
) -> None:
    env, _ = matrix
    env["EVAL_VARIANT"] = variant.value
    for stage in Stage:
        for field in ("TARGETS", "PROMPT_CACHE"):
            env.pop(f"EVAL_{stage.value.upper()}_{field}")
    result, calls = run_matrix(matrix)
    assert result.returncode == 0 and len(calls) == 6
    assert all(f"variant={variant.value}" in call["args"] for call in calls)
    assert all(not any(arg.startswith(("safety.strength=", "safety.targets=", "conditioning.context_cache_dir="))
                       for arg in call["args"]) for call in calls)


def write_progression(tmp_path: Path, *, variants: tuple[Variant, ...] = (Variant.ENCDEC_SAFETY,)) -> Any:
    rows = [{"prompt": "harmful question", "pool": "harmful", "category": "test"},
            {"prompt": "benign question", "pool": "benign", "category": "test"}]
    for stage_index, stage in enumerate(Stage):
        for variant in variants:
            for strength in STRENGTHS if variant is Variant.ENCDEC_SAFETY else [None]:
                root = tmp_path / stage.value / variant.value
                config = {"stage": stage.value, "variant": variant.value, "judge_model": "judge"}
                if strength is not None:
                    root /= strength_leaf(strength)
                    config["safety"] = {"strength": strength, "targets": f"/{stage.value}/targets"}
                identity = {"config": config, "checkpoint_files": {"model": str(stage_index)}}
                if strength is not None:
                    identity["safety_binding"] = {"target": {"sha256": str(stage_index) * 64}, "prompt_seeds": [[1, 2], [3, 4]],
                        "completion": {"strength": strength,
                            "safety_steering": {"strength": strength, "targets_sha256": str(stage_index) * 64}}}
                pairs = [{**row, "response": f"answer {strength}"} for row in rows]
                scores = [JudgeResult([], 1 + (strength or 0) * 2 // 3, "Fixture"), JudgeResult([], 1, "Fixture")]
                payload = build_harmfulness_payload(
                    metrics=compute_metrics(pairs, scores),
                    compliance_metrics=compute_compliance_metrics(pairs[1:], [ComplianceJudgeResult(5, False, "Fixture")]),
                    model="judge", pairs=pairs,
                )
                install_json_once(root / "identity.json", identity)
                install_json_once(root / "prompts.json", rows)
                for index, pair in enumerate(pairs):
                    record = pair if strength is None else {**pair, "generation": safety_record_receipt(identity, index, pair)}
                    install_json_once(root / "records" / f"{index:06d}.json", record)
                if strength is not None:
                    install_json_once(root / "responses.json", {"cell_sha256": semantic_digest(identity),
                                                        "records_sha256": semantic_digest(pairs)})
                install_json_once(root / "results.json", payload)
    return OmegaConf.create({"input_root": str(tmp_path), "variants": [variant.value for variant in variants],
                            "output_file": str(tmp_path / "progression.json"), "safety_strengths": list(STRENGTHS)})


def test_progression_collects_all_nine_strength_cells_with_separate_unsteered_line(tmp_path: Path) -> None:
    cfg = write_progression(tmp_path, variants=(Variant.ENCDEC, Variant.ENCDEC_SAFETY))
    progression.main.__wrapped__(cfg)
    cells = json.loads(Path(cfg.output_file).read_text())["cells"]
    assert len(cells) == 12
    assert [(cell["stage"], cell["strength"]) for cell in cells if "strength" in cell] == [
        (stage.value, strength) for stage in Stage for strength in STRENGTHS]
    assert len({record.signal for record in progression.plot_records(cells)}) == 4


@pytest.mark.parametrize("spec", METRIC_SPECS, ids=lambda spec: spec.metric)
def test_progression_plot_has_one_three_stage_line_per_strength(tmp_path: Path, spec) -> None:
    cfg = write_progression(tmp_path)
    records = progression.plot_records(progression.collect_cells(cfg))
    order, styles = harmfulness_plot_signals(records)
    figure = render_progression_chart(
        records, source_to_stage={stage.value: index for index, stage in enumerate(Stage)},
        stage_labels=[stage.value for stage in Stage], signal_order=order, signal_styles=styles,
        metric=spec.metric, ylabel=spec.ylabel, title="test", fmt=spec.fmt,
    )
    try:
        lines = [line for line in figure.axes[0].lines if line.get_label() in {record.signal for record in records}]
        assert len(lines) == 3
        expected = {HarmfulnessMetric.ASR: [0, 0, 1], HarmfulnessMetric.SAFE_RATE: [1, 0, 0],
                    HarmfulnessMetric.MEAN_SCORE: [1, 2, 3], HarmfulnessMetric.COMPLIANCE_RATE: [1, 1, 1]}
        for index, (line, strength) in enumerate(zip(lines, STRENGTHS, strict=True)):
            assert line.get_label() == f"Encoder-decoder safety (strength={strength})"
            assert list(line.get_xdata()) == [0, 1, 2]
            assert list(line.get_ydata()) == [expected[spec.metric][index]] * 3
    finally:
        plt.close(figure)


def test_progression_cli_writes_all_four_plots(tmp_path: Path) -> None:
    cfg = write_progression(tmp_path)
    cfg.plots_dir = str(tmp_path / "plots")
    progression.main.__wrapped__(cfg)
    assert {path.name for path in Path(cfg.plots_dir).glob("*.png")} == {
        f"campaign2_{spec.suffix}.png" for spec in METRIC_SPECS}
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in Path(cfg.plots_dir).glob("*.png"))


@pytest.mark.parametrize("strengths", [[], [0, 0.0], [True], [float("nan")], [float("inf")], [-1], [7]])
def test_invalid_progression_strengths_fail_before_io(tmp_path: Path, strengths: list) -> None:
    cfg = OmegaConf.create({"input_root": str(tmp_path), "variants": ["encdec_safety"], "safety_strengths": strengths})
    with pytest.raises(ValueError, match="strength"):
        progression.collect_cells(cfg)


def test_progression_refuses_missing_strength_cell(tmp_path: Path) -> None:
    cfg = write_progression(tmp_path)
    (tmp_path / "capabilities/encdec_safety/strength-6/identity.json").unlink()
    with pytest.raises(FileNotFoundError):
        progression.collect_cells(cfg)


@pytest.mark.parametrize("fault", ["strength", "variant", "stage", "prompt", "response", "judge"])
def test_progression_refuses_mismatched_cell_identity_or_judgment(tmp_path: Path, fault: str) -> None:
    cfg = write_progression(tmp_path)
    root = tmp_path / "safety/encdec_safety/strength-3"
    path = root / "identity.json"
    identity = json.loads(path.read_text())
    if fault in {"strength", "variant", "stage", "judge"}:
        if fault == "strength":
            identity["config"]["safety"]["strength"] = 6
        else:
            identity["config"]["judge_model" if fault == "judge" else fault] = "wrong"
        path.write_text(json.dumps(identity))
    else:
        path = root / ("prompts.json" if fault == "prompt" else "records/000000.json")
        data = json.loads(path.read_text())
        if fault == "prompt":
            data[0]["prompt"] = "changed"
        else:
            data["response"] = "changed"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        progression.collect_cells(cfg)


@pytest.mark.parametrize("fault", ["target", "checkpoint", "seed", "completion_strength", "receipt_strength"])
def test_progression_strength_sweep_refuses_changed_controls(tmp_path: Path, fault: str) -> None:
    cfg = write_progression(tmp_path)
    path = tmp_path / "safety/encdec_safety/strength-3/identity.json"
    identity = json.loads(path.read_text())
    if fault == "target":
        identity["safety_binding"]["target"]["sha256"] = "f" * 64
    elif fault == "checkpoint":
        identity["checkpoint_files"]["model"] = "changed"
    elif fault == "seed":
        identity["config"]["seed"] = 43
    elif fault == "completion_strength":
        identity["safety_binding"]["completion"]["strength"] = 6
    else:
        identity["safety_binding"]["completion"]["safety_steering"]["strength"] = 6
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="strength|identities"):
        progression.collect_cells(cfg)


@pytest.mark.parametrize("fault", ["missing", "cell_sha256", "records_sha256"])
def test_progression_refuses_missing_or_changed_response_artifact(tmp_path: Path, fault: str) -> None:
    cfg = write_progression(tmp_path)
    path = tmp_path / "capabilities/encdec_safety/strength-6/responses.json"
    if fault == "missing":
        path.unlink()
        with pytest.raises(FileNotFoundError):
            progression.collect_cells(cfg)
    else:
        receipt = json.loads(path.read_text())
        receipt[fault] = "f" * 64
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="response artifact"):
            progression.collect_cells(cfg)
