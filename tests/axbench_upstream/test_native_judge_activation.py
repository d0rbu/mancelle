"""Native judging activation never expands scope or recharges completed results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_upstream import native_judge as judge
from steering.eval.axbench_upstream.native_contract import FORMAT, POLICY, native_input_frame


def write_judged(frame: pd.DataFrame, output: Path) -> None:
    result = {"factor": [1.0]}
    evaluated = frame.copy()
    for suffix, field in judge.FIELDS.items():
        values = ["Rating: [[1]]"] * 10 if field.endswith("completions") else [1.0] * 10
        result[field] = values
        evaluated[f"{judge.METHOD}_{suffix}"] = values
    for summary in judge.SUMMARY:
        result[summary] = [1.0]
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    (output / "steering.jsonl").write_text(
        json.dumps({"concept_id": 0, "results": {"LMJudgeEvaluator": {judge.METHOD: result}}}) + "\n"
    )
    (output / "judge_call_accounting.json").write_text('{"total_calls": 30, "total_cache_hits": 0}\n')


@pytest.fixture
def activation(tmp_path, monkeypatch):
    inputs = tmp_path / "relocated"
    inputs.mkdir()
    input_path = inputs / "steering_data.parquet"
    manifest_path = inputs / "run_manifest.json"
    frame = native_input_frame(pd.DataFrame([{
        **dict.fromkeys(INPUT_COLUMNS, ""),
        "dataset_name": "AlpacaEval", "concept_id": 0, "input_id": index,
        "input_concept": "rental equipment", "original_prompt": f"Question {index}?", "factor": 1.0,
        "steered_input": f"<|im_start|>user\nMention rental equipment.\n\nQuestion: Question {index}?<|im_end|>\n<|im_start|>assistant\n",
    } for index in range(10)]), max_concepts=1)
    frame[f"{judge.METHOD}_steered_generation"] = "Rent a tool."
    frame.to_parquet(input_path, index=False)
    manifest = {"schema_version": 1, "format": FORMAT, "method": judge.METHOD,
                "factors": [1.0], "complete": True, "policy": {"name": POLICY},
                "source": {"max_concepts": 1}, "output": {"path": "/original/host/steering_data.parquet",
                                                         "sha256": judge.sha256(input_path), "rows": 10}}
    manifest_path.write_text(json.dumps(manifest))
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    for relative in ("uv.lock", "evaluate.py", "axbench/evaluators/lm_judge.py",
                     "axbench/evaluators/prompt_templates.py", "axbench/models/language_models.py"):
        path = vendor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned source")
    config_path = vendor / "evaluate.yaml"
    config_path.write_text(yaml.safe_dump({"evaluate": {
        "models": ["DiffMean", "LoRA", "SelfConditioning"], "num_of_workers": 8,
        "steering_evaluators": ["LMJudgeEvaluator"], "lm_model": "openai/gpt-5.6-luna",
    }}))
    monkeypatch.setattr(judge, "VENDOR_ROOT", vendor)
    monkeypatch.setattr(judge, "EVALUATE_SCRIPT", vendor / "evaluate.py")
    monkeypatch.setattr(judge, "EVALUATE_CONFIG", config_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    cfg = OmegaConf.create({"input_path": str(input_path), "manifest_path": str(manifest_path),
                           "output_dir": str(tmp_path / "judged"), "judge_cache": str(tmp_path / "cache.sqlite"),
                           "max_concepts": 1, "num_workers": 8, "cache_only": False,
                           "api_base_url": "https://openrouter.ai/api/v1"})
    calls = []

    def run(command, *, cwd, check, stdout, stderr):
        calls.append((command, cwd, check))
        write_judged(frame, Path(command[command.index("--overwrite_evaluate_dump_dir") + 1]))

    monkeypatch.setattr(judge.subprocess, "run", run)
    return cfg, frame, calls


def test_target_only_frozen_vendor_execution_and_relocated_manifest(activation):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    command, cwd, check = calls[0]
    assert command[:5] == ["uv", "run", "--project", str(cwd), "--frozen"]
    assert check is True
    assert command[command.index("--overwrite_inference_dump_dir") + 1] == str(Path(cfg.input_path).parent)
    generated_config = yaml.safe_load((Path(cfg.output_dir) / "attempts/attempt_00001/native_evaluate.yaml").read_text())
    assert generated_config["evaluate"]["models"] == [judge.METHOD]
    receipt = json.loads((Path(cfg.output_dir) / "native_judge_receipt.json").read_text())
    assert receipt["baseline_requests"] == 0
    assert receipt["rows"] == 10
    assert receipt["call_accounting"]["total_calls"] == 30
    assert receipt["call_accounting"]["reported_noncache_calls"] == 30
    assert receipt["call_accounting"]["accounting_complete"] is True


def test_completed_receipt_resumes_without_key_or_vendor_calls(activation, monkeypatch):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    judge.run_native_evaluation(cfg)
    assert len(calls) == 1


@pytest.mark.parametrize("column,dtype", [("factor", "Float64"), ("dataset_name", "string")])
def test_nullable_source_fails_before_judging(activation, column, dtype):
    cfg, frame, calls = activation
    frame = frame.astype({column: dtype})
    frame.loc[0, column] = pd.NA
    frame.to_parquet(cfg.input_path, index=False)
    manifest_path = Path(cfg.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["output"]["sha256"] = judge.sha256(Path(cfg.input_path))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        judge.run_native_evaluation(cfg)
    assert not calls
    assert not Path(cfg.output_dir).exists()


def test_native_judge_refuses_other_generations(activation):
    cfg, frame, calls = activation
    frame["Other_steered_generation"] = "unrelated generation"
    frame.to_parquet(cfg.input_path, index=False)
    path = Path(cfg.manifest_path)
    manifest = json.loads(path.read_text())
    manifest["output"]["sha256"] = judge.sha256(Path(cfg.input_path))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="other model generations"):
        judge.run_native_evaluation(cfg)
    assert not calls


def test_corrupt_receipt_fails_before_paid_execution(activation):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    (Path(cfg.output_dir) / "judge_call_accounting.json").write_text("{}")
    with pytest.raises(ValueError, match="file hashes changed"):
        judge.run_native_evaluation(cfg)
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("input_sha256", "wrong"), ("scores", {})])
def test_completed_receipt_metadata_is_rechecked(activation, field, value):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    path = Path(cfg.output_dir) / "native_judge_receipt.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        judge.run_native_evaluation(cfg)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["filename", "overlap", "concepts", "unbound_output", "missing_key", "evaluator"])
def test_invalid_judge_execution_never_calls_vendor(activation, monkeypatch, mutation):
    cfg, _, calls = activation
    if mutation == "filename":
        cfg.input_path = str(Path(cfg.input_path).with_name("other.parquet"))
    elif mutation == "overlap":
        cfg.output_dir = str(Path(cfg.input_path).parent)
    elif mutation == "concepts":
        cfg.max_concepts = 2
    elif mutation == "unbound_output":
        output = Path(cfg.output_dir)
        output.mkdir()
        (output / "unrelated.txt").touch()
    elif mutation == "missing_key":
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        judge.EVALUATE_CONFIG.write_text('evaluate: {lm_model: wrong, steering_evaluators: []}')
    with pytest.raises(ValueError):
        judge.run_native_evaluation(cfg)
    assert not calls
    assert not (Path(cfg.output_dir) / "native_judge_receipt.json").exists()


@pytest.mark.parametrize("failure", ["missing_output", "invalid_accounting"])
def test_successful_vendor_exit_is_not_a_completion_receipt(activation, monkeypatch, failure):
    cfg, frame, calls = activation

    def run(command, **kwargs):
        calls.append(command)
        if failure == "invalid_accounting":
            output = Path(command[command.index("--overwrite_evaluate_dump_dir") + 1])
            write_judged(frame, output)
            (output / "judge_call_accounting.json").write_text('{"total_calls": 1, "total_cache_hits": 2}')

    monkeypatch.setattr(judge.subprocess, "run", run)
    with pytest.raises((OSError, ValueError)):
        judge.run_native_evaluation(cfg)
    assert len(calls) == 1
    assert not (Path(cfg.output_dir) / "native_judge_receipt.json").exists()


def test_changed_resume_judge_route_fails_before_paid_execution(activation):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    cfg.api_base_url = "https://different.invalid/api/v1"
    with pytest.raises(ValueError, match="resume identity changed"):
        judge.run_native_evaluation(cfg)
    assert len(calls) == 1


def test_complete_artifacts_finalize_without_receipt_or_live_key(activation, monkeypatch):
    cfg, _, calls = activation
    judge.run_native_evaluation(cfg)
    (Path(cfg.output_dir) / "native_judge_receipt.json").unlink()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    judge.run_native_evaluation(cfg)
    assert len(calls) == 1
    assert (Path(cfg.output_dir) / "native_judge_receipt.json").exists()


def test_partial_vendor_state_restarts_clean_with_cache_and_unknown_accounting(activation, monkeypatch):
    cfg, frame, calls = activation

    def fail(command, **kwargs):
        calls.append((command, kwargs["cwd"], kwargs["check"]))
        attempt = Path(command[command.index("--overwrite_evaluate_dump_dir") + 1])
        (attempt / "steering_evaluate_state.pkl").write_bytes(b"state already advanced")
        (attempt / "steering.jsonl").write_text("partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(judge.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        judge.run_native_evaluation(cfg)

    def recover(command, **kwargs):
        calls.append((command, kwargs["cwd"], kwargs["check"]))
        attempt = Path(command[command.index("--overwrite_evaluate_dump_dir") + 1])
        assert not (attempt / "steering_evaluate_state.pkl").exists()
        write_judged(frame, attempt)
        (attempt / "judge_call_accounting.json").write_text('{"total_calls":30,"total_cache_hits":30}')

    monkeypatch.setattr(judge.subprocess, "run", recover)
    judge.run_native_evaluation(cfg)
    output = Path(cfg.output_dir)
    assert (output / "attempts/attempt_00001/steering.jsonl").read_text() == "partial"
    assert (output / "attempts/attempt_00002/steering_data.parquet").exists()
    report = json.loads((output / "judge_call_accounting.json").read_text())
    assert report["total_calls"] is None
    assert report["accounting_complete"] is False
    assert report["unreported_attempts"] == ["attempts/attempt_00001"]
    assert report["reported_total_calls"] == 30
    assert report["reported_noncache_calls"] == 0
    assert len(calls) == 2


def test_complete_judging_survives_vendor_plot_failure(activation, monkeypatch):
    cfg, frame, calls = activation

    def failed_plot(command, **kwargs):
        calls.append(command)
        attempt = Path(command[command.index("--overwrite_evaluate_dump_dir") + 1])
        write_judged(frame, attempt)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(judge.subprocess, "run", failed_plot)
    judge.run_native_evaluation(cfg)
    report = json.loads((Path(cfg.output_dir) / "judge_call_accounting.json").read_text())
    assert report["total_calls"] == 30
    judge.run_native_evaluation(cfg)
    assert len(calls) == 1
