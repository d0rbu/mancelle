"""Native-only GEPA contracts, durable provider handoff, and original-prompt metrics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_adv import judge_native_points
from steering.eval.axbench_adv import final_judge, native_decoder
from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import build_point_frame
from steering.eval.axbench_adv.gepa_optimization import (
    OptimizationExample,
    RawEvaluation,
    optimize_suffix,
)
from steering.eval.axbench_adv.native_mailbox import (
    ArtifactMailbox,
    digest_json,
    read_json,
    write_json,
)
from steering.eval.axbench_adv.types import METHODS, NATIVE_METHOD, validate_method_subset
from steering.eval.axbench_upstream.native_contract import sha256
from tests.axbench_adv.unit.test_final_judge import _outputs


@pytest.fixture
def offline_final_judge(monkeypatch):
    calls = []

    def run(command, **kwargs):
        inference = Path(command[command.index("--overwrite_inference_dump_dir") + 1])
        output = Path(command[command.index("--overwrite_evaluate_dump_dir") + 1])
        calls.append(inference.parent.name)
        emissions = {(e["method"], e["concept_id"], e["input_id"]): RawEvaluation(**e["raw"])
                     for e in read_json(inference.parent / "emissions.json")["emissions"]}
        _outputs(output, pq.read_table(inference / "steering_data.parquet"), emissions,
                 max_concepts=1, factors={NATIVE_METHOD: 1.0})
        write_json(output / "judge_call_accounting.json", {"total_calls": 30, "total_cache_hits": 20})

    monkeypatch.setattr(final_judge.subprocess, "run", run)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-placeholder")
    return calls


def test_native_only_does_not_change_legacy_methods():
    assert NATIVE_METHOD not in METHODS
    assert validate_method_subset([NATIVE_METHOD]) == (NATIVE_METHOD,)
    with pytest.raises(AxBenchAdvError):
        validate_method_subset([NATIVE_METHOD, "SelfConditioning"])


def test_native_point_preserves_original_judge_instruction():
    frame = pd.DataFrame([{"dataset_name": "AlpacaEval", "concept_id": 0, "input_id": i,
                              "input_concept": "cats", "factor": 1.0, "original_prompt": f"Question {i}"} for i in range(10)])
    emissions = {(NATIVE_METHOD, 0, i): RawEvaluation("long response", 1.0, 2.0, "rationale", 0) for i in range(10)}
    table = build_point_frame(source_frame=frame, best_factors={NATIVE_METHOD: 1.0},
                              emissions=emissions, max_concepts=1)
    assert table.column("original_prompt").to_pylist() == list(frame.original_prompt)
    assert f"{NATIVE_METHOD}_steered_generation" in table.column_names
    with pytest.raises(AxBenchAdvError, match="declared grid"):
        build_point_frame(source_frame=frame, best_factors={NATIVE_METHOD: 2.0}, emissions=emissions, max_concepts=1)


def test_mailbox_response_is_bound_to_request_and_content(tmp_path):
    payload = {"model": "approved", "prompt": "test"}
    request_id = digest_json({"schema_version": 1, "operation": "reflection", "payload": payload})
    result = {"response": "answer", "paid_cache_misses": 1}
    response_path = tmp_path / "responses" / f"{request_id}.json"
    write_json(response_path, {"request_id": request_id, "result": result, "result_sha256": digest_json(result)})
    mailbox = ArtifactMailbox(tmp_path)
    assert mailbox.request("reflection", payload) == {**result, "paid_cache_misses": 0}
    assert read_json(tmp_path / "requests" / f"{request_id}.json")["payload"] == payload
    write_json(response_path, {"request_id": "wrong", "result": result, "result_sha256": digest_json(result)})
    with pytest.raises(AxBenchAdvError, match="identity"):
        mailbox.request("reflection", payload)


def test_native_suffix_preserves_steering_and_rejects_role_injection():
    row = SimpleNamespace(native_user_prompt="Frozen boosted instruction\n\nQuestion: original")
    assert native_decoder.attacked_user_text(row, "  avoid cats  ") == row.native_user_prompt + "\n\navoid cats"
    assert native_decoder.attacked_user_text(row, "") == row.native_user_prompt
    for token in ("<assistant_start>", "<|assistant_start|>", "<|im_start|>"):
        with pytest.raises(AxBenchAdvError):
            native_decoder.attacked_user_text(row, token)


class Reflection:
    paid_cache_misses = 0

    def __call__(self, prompt):
        return "```avoid all references to the concept```"


def test_actual_gepa_zero_generation_rejection_accounting(tmp_path):
    result = optimize_suffix(
        examples=[OptimizationExample(i, f"original {i}") for i in range(3)],
        runtime=lambda suffix, i: RawEvaluation("", 2.0, 0.0, "over context", 0, 0),
        reflection_lm=Reflection(), objective="Avoid cats, stay responsive", run_dir=tmp_path / "gepa",
        proposal_count=1, gepa_seed=42,
    )
    assert result.best_fitness == 0.0
    assert result.suffix == "Please avoid the target concept while remaining helpful and responsive."
    assert result.accounting.actual_evaluator_calls > 0
    assert result.accounting.actual_generation_calls == 0
    assert result.accounting.paid_judge_cache_misses == 0


@pytest.fixture
def native_campaign(tmp_path, monkeypatch):
    frame = pd.DataFrame([dict(dataset_name="AlpacaEval", concept_id=0, input_id=i, input_concept="cats",
                              factor=1.0, original_prompt=f"original {i}",
                              native_user_prompt=f"frozen steering\n\nQuestion: original {i}",
                              **{f"{NATIVE_METHOD}_steered_generation": "Control generation includes cats."}) for i in range(10)])
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.json"
    manifest_path.write_text("{}")
    parquet = source / "source.parquet"
    frame.to_parquet(parquet)
    manifest = {"source": {"max_concepts": 1, "input_path": str(parquet)},
                "checkpoint": {"checkpoint_dir": str(source / "checkpoint"), "step": 1003, "tokenizer_path": str(source / "tokenizer")},
                "generation": {"seed": 42, "max_new_tokens": 128, "temperature": 1.0, "top_k": 50}}
    calls, judged = [], []

    class Adapter:
        def __init__(self, **kwargs):
            self.tokenizer = None
            self.model = SimpleNamespace(config=SimpleNamespace(sequence_len=1024))

        def generate_user_text(self, user_text, **kwargs):
            calls.append((user_text, kwargs))
            return "A helpful fluent response without the target concept."

        def prompt_tokens(self, text):
            return [1, 2]

    def request(self, operation, payload):
        if operation == "reflection":
            return {"response": "```avoid all references to cats```", "paid_cache_misses": 0}
        judged.append(payload)
        return asdict(RawEvaluation(payload["generation"], 0.0 if "without" in payload["generation"] else 2.0,
                                    2.0, '{"concept":"Rating: 0","fluency":"Rating: 2"}', 2))

    monkeypatch.setattr(native_decoder, "validate_native_source", lambda *args: (manifest, frame))
    monkeypatch.setattr(native_decoder, "NativeDecoderAdapter", Adapter)
    monkeypatch.setattr(ArtifactMailbox, "request", request)
    cfg = OmegaConf.create({"source_manifest": str(manifest_path), "source_parquet": str(parquet), "output_dir": str(tmp_path / "adv"),
                               "mailbox_dir": str(tmp_path / "mailbox"), "concept_ids": None, "proposal_counts": [1], "device": "cpu",
                               "gepa_seed": 42, "seed_suffix": "Please avoid the target concept.", "reflection_model": "openai/gpt-5.6-luna",
                               "reflection_minibatch_size": 3, "provider_timeout_seconds": 1.0})
    return SimpleNamespace(cfg=cfg, frame=frame, calls=calls, judged=judged, adapter=Adapter)


@pytest.mark.parametrize("interruption", [None, "before_cell", "before_judge"])
def test_native_gepa_end_to_end_and_resume(tmp_path, monkeypatch, offline_final_judge, native_campaign, interruption):
    cfg, frame, calls, Adapter = native_campaign.cfg, native_campaign.frame, native_campaign.calls, native_campaign.adapter
    if interruption == "before_judge":
        request = ArtifactMailbox.request
        fail_judge = True

        def interrupt_judge(self, operation, payload):
            nonlocal fail_judge
            if fail_judge and operation == "judge" and "without" in payload["generation"]:
                fail_judge = False
                raise OSError("interrupted before judge response")
            return request(self, operation, payload)

        monkeypatch.setattr(ArtifactMailbox, "request", interrupt_judge)
    if interruption == "before_cell":
        def interrupt(path, value):
            if path.name == "c0-p1.json":
                raise OSError("interrupted before GEPA cell publication")
            write_json(path, value)

        monkeypatch.setattr(native_decoder, "write_json", interrupt)
        with pytest.raises(OSError, match="interrupted before GEPA cell publication"):
            native_decoder.run_native_adv(cfg)
        monkeypatch.setattr(native_decoder, "write_json", write_json)
    elif interruption == "before_judge":
        with pytest.raises(OSError, match="interrupted before judge response"):
            native_decoder.run_native_adv(cfg)
        pending = [path for path in (tmp_path / "adv/generations").glob("*.json") if "raw" not in read_json(path)]
        assert len(pending) == len(calls) == 1
    native_decoder.run_native_adv(cfg)
    if interruption == "before_cell":
        accounting = read_json(tmp_path / "adv/cells/c0-p1.json")["accounting"]
        assert accounting["actual_generation_calls"] == 0
        assert accounting["paid_judge_cache_misses"] == 0
    elif interruption == "before_judge":
        assert len(calls) == len({text for text, _ in calls})
        assert read_json(pending[0])["raw"]["generation_calls"] == 0
        assert all("raw" in read_json(path) for path in (tmp_path / "adv/generations").glob("*.json"))
    assert calls
    assert all(call[0].startswith("frozen steering\n\nQuestion: original ") for call in calls)
    point = pd.read_parquet(tmp_path / "adv/points/p1/inference/steering_data.parquet")
    assert point.original_prompt.tolist() == frame.original_prompt.tolist()
    before = len(calls)
    point_bytes = {path: path.read_bytes() for path in (tmp_path / "adv/points").rglob("*") if path.is_file()}
    cache = tmp_path / "cache.sqlite"
    cache.touch()
    judge_cfg = OmegaConf.create({"points_dir": str(tmp_path / "adv/points"), "output_dir": str(tmp_path / "judged"),
                                 "judge_cache": str(cache), "max_concepts": 1, "concept_ids": None, "proposal_counts": [1]})
    judge_native_points.main.__wrapped__(judge_cfg)
    native_decoder.run_native_adv(cfg)
    assert len(calls) == before
    assert all(path.read_bytes() == content for path, content in point_bytes.items())
    judge_native_points.main.__wrapped__(judge_cfg)
    assert offline_final_judge == ["p0", "p1"]
    result_path = tmp_path / "adv/cells/c0-p1.json"
    tampered = read_json(result_path)
    original = dict(tampered)
    tampered["suffix"] = "overflow"
    write_json(result_path, tampered)
    monkeypatch.setattr(Adapter, "prompt_tokens",
                        lambda self, text: list(range(1000)) if text.endswith("overflow") else [1, 2])
    with pytest.raises(AxBenchAdvError, match="selected an over-context suffix"):
        native_decoder.run_native_adv(cfg)
    tampered = original
    tampered["concept_id"] = 1
    write_json(result_path, tampered)
    with pytest.raises(AxBenchAdvError, match="cell identity"):
        native_decoder.run_native_adv(cfg)


@pytest.mark.parametrize("where", ["seed", "proposal"])
def test_native_context_overflow_never_generates_or_judges_rejection(native_campaign, monkeypatch, where):
    cfg = native_campaign.cfg
    monkeypatch.setattr(native_campaign.adapter, "prompt_tokens",
                        lambda self, text: list(range(1000)) if text.endswith("overflow") else [1, 2])
    if where == "seed":
        cfg.seed_suffix = "overflow"
        with pytest.raises(AxBenchAdvError, match="initial GEPA seed"):
            native_decoder.run_native_adv(cfg)
        assert not native_campaign.calls and not native_campaign.judged
        return
    request = ArtifactMailbox.request

    def overflow_proposal(self, operation, payload):
        return ({"response": "```overflow```", "paid_cache_misses": 0}
                if operation == "reflection" else request(self, operation, payload))

    monkeypatch.setattr(ArtifactMailbox, "request", overflow_proposal)
    native_decoder.run_native_adv(cfg)
    rejected = [read_json(path) for path in (Path(cfg.output_dir) / "generations").glob("*.json")
                if read_json(path).get("rejection") == "context_overflow"]
    assert rejected
    assert all(value["raw"]["generation_calls"] == value["raw"]["paid_judge_cache_misses"] == 0 for value in rejected)
    assert all(not text.endswith("overflow") for text, _ in native_campaign.calls)
    assert all(payload["generation"] for payload in native_campaign.judged)


@pytest.mark.parametrize("mutation", ["generation_key", "cell_inputs", "cell_fitness", "identity"])
def test_native_resume_rejects_corrupt_cached_state(native_campaign, mutation):
    cfg = native_campaign.cfg
    native_decoder.run_native_adv(cfg)
    if mutation == "identity":
        cfg.gepa_seed += 1
        before = (len(native_campaign.calls), len(native_campaign.judged))
        with pytest.raises(AxBenchAdvError, match="resume identity"):
            native_decoder.run_native_adv(cfg)
        assert (len(native_campaign.calls), len(native_campaign.judged)) == before
        return
    output = Path(cfg.output_dir)
    path = (next(path for path in (output / "generations").glob("*.json")
                 if read_json(path)["generation"].startswith("Control"))
            if mutation == "generation_key" else output / "cells/c0-p1.json")
    saved = read_json(path)
    if mutation == "generation_key":
        saved["key"] = "wrong"
    elif mutation == "cell_inputs":
        saved["evaluations"].pop()
    else:
        saved["best_fitness"] += 0.1
    write_json(path, saved)
    before = (len(native_campaign.calls), len(native_campaign.judged))
    with pytest.raises(AxBenchAdvError):
        native_decoder.run_native_adv(cfg)
    assert (len(native_campaign.calls), len(native_campaign.judged)) == before


@pytest.mark.parametrize("field,value", [
    ("proposal_counts", [0]), ("concept_ids", [1]), ("reflection_model", "wrong"),
    ("output_dir", "source"),
])
def test_native_rejects_invalid_campaign_before_calls(native_campaign, field, value):
    cfg = native_campaign.cfg
    cfg[field] = str(Path(cfg.source_manifest).parent) if field == "output_dir" else value
    with pytest.raises(AxBenchAdvError):
        native_decoder.run_native_adv(cfg)
    assert not native_campaign.calls and not native_campaign.judged


@pytest.fixture
def staged_points(tmp_path):
    points = tmp_path / "points"
    output = tmp_path / "judged"
    cache = tmp_path / "cache.sqlite"
    cache.touch()
    frame = pd.DataFrame([{"dataset_name": "AlpacaEval", "concept_id": 0, "input_id": i,
                           "input_concept": "cats", "factor": 1.0, "original_prompt": f"original {i}",
                           f"{NATIVE_METHOD}_steered_generation": "generated answer"} for i in range(10)])
    raw = RawEvaluation("generated answer", 1.0, 2.0, '{"concept":"r","fluency":"r"}', 0)
    for level in (0, 1):
        inference = points / f"p{level}" / "inference"
        inference.mkdir(parents=True)
        table = build_point_frame(source_frame=frame, best_factors={NATIVE_METHOD: 1.0},
                                  emissions={(NATIVE_METHOD, 0, i): raw for i in range(10)}, max_concepts=1)
        pq.write_table(table, inference / "steering_data.parquet")
        write_json(inference.parent / "emissions.json", {"emissions": [
            {"method": NATIVE_METHOD, "concept_id": 0, "input_id": i, "raw": asdict(raw)} for i in range(10)]})
    return OmegaConf.create({"points_dir": str(points), "output_dir": str(output), "judge_cache": str(cache),
                            "max_concepts": 1, "concept_ids": None, "proposal_counts": [1]})


def test_final_judge_requires_all_points_and_resumes_completed(staged_points, offline_final_judge):
    cfg = staged_points
    cfg.proposal_counts = [1, 2]
    with pytest.raises(ValueError, match="required native point"):
        judge_native_points.main.__wrapped__(cfg)
    assert not offline_final_judge
    cfg.proposal_counts = [1]
    judge_native_points.main.__wrapped__(cfg)
    # A crash between publishing the complete judge output and its outer receipt is recoverable.
    (Path(cfg.output_dir) / "p0-receipt.json").unlink()
    judge_native_points.main.__wrapped__(cfg)
    assert offline_final_judge == ["p0", "p1"]
    assert len(read_json(Path(cfg.output_dir) / "receipt.json")["points"]) == 2
    emissions_path = Path(cfg.points_dir) / "p0" / "emissions.json"
    changed = read_json(emissions_path)
    changed["emissions"][0]["raw"]["judge_rationale"] = '{"concept":"changed","fluency":"r"}'
    write_json(emissions_path, changed)
    with pytest.raises(ValueError, match="source differs"):
        judge_native_points.main.__wrapped__(cfg)


@pytest.mark.parametrize("levels", [[0], [True], [3]])
def test_final_judge_rejects_invalid_levels_before_calls(staged_points, offline_final_judge, levels):
    staged_points.proposal_counts = levels
    with pytest.raises(ValueError, match="levels"):
        judge_native_points.main.__wrapped__(staged_points)
    assert not offline_final_judge


@pytest.mark.parametrize("mutation,match", [
    ("missing", "coverage"), ("extra", "coverage"), ("duplicate", "duplicate"),
    ("method", "other methods"), ("generation", "generation"), ("frame", "concept/input product"),
    ("rationale", "rationale is not JSON"),
])
def test_all_point_emissions_are_checked_before_any_judging(staged_points, offline_final_judge, mutation, match):
    point = Path(staged_points.points_dir) / "p1"
    path = point / "emissions.json"
    content = read_json(path)
    rows = content["emissions"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({**rows[0], "input_id": 10})
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "method":
        rows[0]["method"] = "DiffMean"
    elif mutation == "generation":
        rows[0]["raw"]["generation"] = "Different fluent answer."
    elif mutation == "rationale":
        rows[0]["raw"]["judge_rationale"] = "malformed"
    else:
        frame_path = point / "inference/steering_data.parquet"
        table = pq.read_table(frame_path)
        pq.write_table(table.slice(0, 9), frame_path)
    write_json(path, content)
    with pytest.raises((ValueError, AxBenchAdvError), match=match):
        judge_native_points.main.__wrapped__(staged_points)
    assert not offline_final_judge


@pytest.mark.parametrize("mutation", ["input", "jsonl", "parquet", "accounting"])
def test_completed_point_manifest_is_verified(staged_points, offline_final_judge, mutation):
    cfg = staged_points
    judge_native_points.main.__wrapped__(cfg)
    output = Path(cfg.output_dir) / "p0"
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path)
    receipt = Path(cfg.output_dir) / "p0-receipt.json"
    receipt.unlink()
    if mutation == "input":
        manifest["input_logical_sha256"] = "0" * 64
    elif mutation == "accounting":
        write_json(output / "judge_call_accounting.json", {"total_calls": 30, "total_cache_hits": 0})
    else:
        entry = read_json(output / "steering.jsonl")
        block = entry["results"]["LMJudgeEvaluator"][NATIVE_METHOD]
        aggregate = 3 / (1 / 1.0 + 1 / 0.5 + 1 / 2.0)
        block["raw_relevance_instruction_ratings"] = [0.5] * 10
        block["relevance_instruction_ratings"] = [0.5]
        block["raw_aggregated_ratings"] = [aggregate] * 10
        block["lm_judge_rating"] = [aggregate]
        write_json(output / "steering.jsonl", entry)
        frame = pd.read_parquet(output / "steering_data.parquet")
        frame[f"{NATIVE_METHOD}_LMJudgeEvaluator_relevance_instruction_ratings"] = 0.5
        frame[f"{NATIVE_METHOD}_LMJudgeEvaluator"] = aggregate
        frame.to_parquet(output / "steering_data.parquet", index=False)
        # Keep the other output's saved hash current, to exercise each hash independently.
        if mutation == "jsonl":
            manifest["files"]["steering_data.parquet"]["logical_sha256"] = logical_table_digest(pq.read_table(output / "steering_data.parquet"))
        else:
            manifest["files"]["steering.jsonl"]["sha256"] = sha256(output / "steering.jsonl")
        point = Path(cfg.points_dir) / "p0"
        emissions = {(e["method"], e["concept_id"], e["input_id"]): RawEvaluation(**e["raw"])
                     for e in read_json(point / "emissions.json")["emissions"]}
        final_judge.validate_final_judge_outputs(input_table=pq.read_table(point / "inference/steering_data.parquet"),
            output_dir=output, best_factors={NATIVE_METHOD: 1.0}, emissions=emissions, max_concepts=1)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest"):
        judge_native_points.main.__wrapped__(cfg)
    assert not receipt.exists()
    assert offline_final_judge == ["p0", "p1"]
