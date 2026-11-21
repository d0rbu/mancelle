"""Counter continuation proves fail-closed admission and exact paid-emission reuse."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from scripts.eval.axbench_adv.merge_slices import _load_run_slice
from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv import counter_resume, pipeline
from steering.eval.axbench_adv.counter_resume import (
    QUALIFIED_PARENT,
    SCIENCE_FIELDS,
    seal_snapshot,
    sha256,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from tests.axbench_adv.functional import test_counter_pipeline as fixture
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import _validated

PRESSURES = (Pressure.LINEAR_PROBE_COUNTER, Pressure.DIFFMEAN_COUNTER)
METHODS = ("SelfConditioning", "SelfConditioningMasked")


class ConsistentCell(fixture._CounterCell):
    def evaluate_counter(self, counter, magnitude, input_ids):
        return tuple(
            replace(
                raw,
                judge_rationale=json.dumps(
                    {
                        "concept": f"concept Rating: [[{raw.concept_rating}]]",
                        "fluency": f"fluency Rating: [[{raw.fluency_rating}]]",
                    }
                ),
            )
            for raw in super().evaluate_counter(counter, magnitude, input_ids)
        )


class ConsistentRuntime(fixture._CounterRuntime):
    def bind_cell(self, *, method, concept_id, **_kwargs):
        cell = ConsistentCell(method, concept_id)
        self.cells.append(cell)
        return cell


def _cache(path, points, concepts):
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE judge_responses (request_sha256 BLOB PRIMARY KEY, request_json TEXT NOT NULL, response TEXT NOT NULL) WITHOUT ROWID"
        )
        templates = counter_resume._templates()
        for point in points:
            doc = json.loads(point.read_text())
            concept = concepts[doc["direction"]["concept_id"]]
            for raw in doc["evaluations"]:
                for kind, template in templates.items():
                    request = {
                        "version": 1,
                        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                        "model": "openai/gpt-5.6-luna",
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "user",
                                "content": template.format(
                                    concept=concept, sentence=raw["generation"]
                                ),
                            }
                        ],
                    }
                    encoded = json.dumps(
                        request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO judge_responses VALUES (?, ?, ?)",
                        (
                            hashlib.sha256(encoded.encode()).digest(),
                            encoded,
                            json.loads(raw["judge_rationale"])[kind],
                        ),
                    )
        # A paid instruction/unfinished-point response must survive too.
        encoded = '{"unfinished":"paid"}'
        db.execute(
            "INSERT INTO judge_responses VALUES (?, ?, ?)",
            (hashlib.sha256(encoded.encode()).digest(), encoded, "preserve me"),
        )


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(METHODS))
    nanochat = tmp_path / "runtime"
    (nanochat / "tokenizer").mkdir(parents=True)
    (nanochat / "tokenizer/tokenizer.pkl").write_bytes(b"immutable tokenizer")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(nanochat))
    configs = []
    original_run = pipeline.run_pipeline
    fit = tmp_path / "fit_manifest.json"
    fit.write_text('{"fit":"verified by fixture"}')

    def capture(config, **kwargs):
        configs.append(config)
        monkeypatch.setattr(
            pipeline,
            "preflight_project_direction_fit",
            lambda *_args: SimpleNamespace(manifest_sha256=sha256(fit)),
        )
        return original_run(config, **kwargs)

    monkeypatch.setattr(fixture, "run_pipeline", capture)
    parent_manifest = fixture._run(
        tmp_path, monkeypatch, "parent", PRESSURES, self_conditioning_factory=ConsistentRuntime
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shutil.copytree(parent_manifest.parent, snapshot / "output")
    (snapshot / "output/manifest.json").unlink()
    shutil.copytree(nanochat / "tokenizer", snapshot / "tokenizer")
    shutil.copyfile(fit, snapshot / "fit_manifest.json")
    config = configs[-1]
    science = {
        key: list(getattr(config, key))
        if key in {"pressures", "counter_magnitudes"}
        else getattr(config, key)
        for key in SCIENCE_FIELDS
    }
    science["pressures"] = list(map(str, science["pressures"]))
    (snapshot / "config.yaml").write_text(yaml.safe_dump(science))
    (snapshot / "launch.json").write_text(
        json.dumps({"source": QUALIFIED_PARENT, "concept_ids": [0, 1], "methods": METHODS})
    )
    concepts = {
        cid: item.concept for cid, item in _validated(tmp_path / "unused").training_texts.items()
    }
    _cache(snapshot / "judge-cache.sqlite", (snapshot / "output").rglob("point.json"), concepts)
    child_cache = tmp_path / "child-judge.sqlite"
    shutil.copyfile(snapshot / "judge-cache.sqlite", child_cache)

    def seal():
        path = snapshot / counter_resume.SNAPSHOT_NAME
        path.unlink(missing_ok=True)
        return seal_snapshot(
            snapshot,
            parent_identity="fixture natural timeout",
            methods=METHODS,
            concept_ids=[0, 1],
            parent_tokenizer_path=str(nanochat / "tokenizer"),
        )

    def run(
        *, name="child", runtime=ConsistentRuntime, config_change=None, judge=None, preflight=False
    ):
        index = snapshot / counter_resume.SNAPSHOT_NAME

        def replay_config(config, **kwargs):
            config = replace(
                config,
                counter_resume_snapshot=index,
                counter_resume_sha256=sha256(index),
                judge_cache=child_cache,
            )
            if config_change:
                config = config_change(config)
            if judge:
                kwargs["final_judge"] = judge
            kwargs["counter_resume_preflight_only"] = preflight
            return capture(config, **kwargs)

        monkeypatch.setattr(fixture, "run_pipeline", replay_config)
        return fixture._run(
            tmp_path, monkeypatch, name, PRESSURES, self_conditioning_factory=runtime
        )

    return SimpleNamespace(
        root=tmp_path,
        snapshot=snapshot,
        seal=seal,
        run=run,
        cache=child_cache,
        parent=parent_manifest,
        tokenizer=nanochat / "tokenizer/tokenizer.pkl",
    )


def forbidden(**kwargs):
    raise AssertionError("model/provider boundary reached")


def test_seal_requires_parent_launch_evidence(campaign):
    (campaign.snapshot / "launch.json").unlink()
    with pytest.raises(AxBenchAdvError, match="launch.json"):
        campaign.seal()


def test_seal_rejects_different_parent_launch_commit(campaign):
    path = campaign.snapshot / "launch.json"
    launch = json.loads(path.read_text())
    launch["source"] = "0" * 40
    path.write_text(json.dumps(launch))
    with pytest.raises(AxBenchAdvError, match="launch source/scope"):
        campaign.seal()


def test_seal_rejects_different_parent_tokenizer(campaign):
    (campaign.snapshot / "tokenizer/tokenizer.pkl").write_bytes(b"child tokenizer")
    with pytest.raises(AxBenchAdvError, match="tokenizer differs"):
        campaign.seal()


def test_seal_rejects_manifest_contradicting_launch(campaign):
    (campaign.snapshot / "output/manifest.json").write_text(
        json.dumps({"implementation": {"repository_git_commit": "0" * 40}})
    )
    with pytest.raises(AxBenchAdvError, match="manifest producer commit"):
        campaign.seal()


def test_full_replay_exact_emissions_no_runtime_and_native_merge(campaign):
    campaign.seal()
    manifest = campaign.run(runtime=forbidden)
    doc = json.loads(manifest.read_text())
    assert doc["call_accounting"]["actual_generation_calls"] == 0
    assert doc["call_accounting"]["control_generation_calls"] == 0
    assert doc["call_accounting"]["paid_judge_cache_misses"] == 0
    assert doc["counter_resume"]["reused_unique_evaluations"] == 200
    for parent in (campaign.snapshot / "output/counter").rglob("point.json"):
        child = manifest.parent / parent.relative_to(campaign.snapshot / "output")
        assert json.loads(child.read_text()) == json.loads(parent.read_text())
    for pressure in PRESSURES:
        _load_run_slice(manifest.parent, pressure)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("reuse_mask", [False, True])
def test_partial_replay_computes_only_missing_point_and_reuses_zero(campaign, reuse_mask, workers):
    point = (
        campaign.snapshot
        / "output/counter/linear_probe_counter/SelfConditioning/c0/m0.25/point.json"
    )
    point.unlink()
    # A temporary interrupted write is retained as evidence, never a completed point.
    point.with_name("point.json.tmp").write_text('{"interrupted":')
    campaign.seal()
    manifest = campaign.run(
        config_change=lambda config: replace(
            config,
            project_judge_inflight=3 if workers == 2 else 2,
            project_judge_workers=workers,
            project_reuse_single_token_mask=reuse_mask,
            project_reuse_diffmean_texts=reuse_mask,
        )
    )
    doc = json.loads(manifest.read_text())
    assert doc["project_judge_inflight"] == (3 if workers == 2 else 2)
    assert doc.get("project_judge_workers", 1) == workers
    assert doc.get("project_reuse_single_token_mask", False) is reuse_mask
    assert doc.get("project_reuse_diffmean_texts", False) is reuse_mask
    assert doc["call_accounting"]["actual_generation_calls"] == 10
    assert doc["call_accounting"]["control_generation_calls"] == 0
    assert doc["counter_resume"]["reused_unique_evaluations"] == 190
    assert json.loads(
        (manifest.parent / point.relative_to(campaign.snapshot / "output")).read_text()
    ) == json.loads(
        (campaign.parent.parent / point.relative_to(campaign.snapshot / "output")).read_text()
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("source", "source contract"),
        ("factor", "selected factors"),
        ("direction", "direction"),
        ("rows", "input rows"),
        ("corrupt", "JSON"),
        ("fitness", "fitness"),
        ("zero", "shared zero"),
        ("tokenizer", "tokenizer"),
        ("cache", "parent paid judgment"),
        ("rationale", "rationale/rating"),
        ("fit", "fit manifest"),
        ("parent", "source/schema"),
    ],
)
def test_mismatch_refused_before_model_or_provider(campaign, mutation, match):
    base = campaign.snapshot / "output"
    path = base / "counter/linear_probe_counter/SelfConditioning/c0/m0.25/point.json"
    if mutation == "source":
        path = base / "source_contract.json"
        doc = json.loads(path.read_text())
        doc["hashes"]["main_parquet"] = "0" * 64
    elif mutation == "factor":
        path = base / "selection/best_points.json"
        doc = json.loads(path.read_text())
        doc["SelfConditioning"]["factor"] = 0.5
    elif mutation == "tokenizer":
        doc = None
    elif mutation == "cache":
        with sqlite3.connect(campaign.cache) as db:
            db.execute("DELETE FROM judge_responses WHERE response = 'preserve me'")
        doc = None
    elif mutation == "fit":
        (campaign.snapshot / "fit_manifest.json").write_text('{"fit":"changed"}')
        doc = None
    elif mutation == "parent":
        doc = None
    else:
        if mutation == "zero":
            path = base / "counter/diffmean_counter/SelfConditioning/c0/m0.0/point.json"
        doc = json.loads(path.read_text())
        if mutation == "direction":
            doc["direction"]["sha256"] = "0" * 64
        if mutation == "rows":
            doc["evaluations"][0]["input_id"] = 1
        if mutation == "fitness":
            doc["mean_fitness"] = -1
        if mutation == "rationale":
            doc["evaluations"][0]["judge_rationale"] = (
                '{"concept":"Rating: [[1]]", "fluency":"Rating: [[1.75]]"}'
            )
        if mutation == "zero":
            doc["evaluations"][0]["generation"] += " different zero"
    if doc is not None:
        path.write_text(json.dumps(doc))
    if mutation == "corrupt":
        path.write_text('{"truncated":')
    index = campaign.seal()
    if mutation == "tokenizer":
        campaign.tokenizer.write_bytes(b"changed tokenizer")
    if mutation == "parent":
        receipt = json.loads(index.read_text())
        receipt["parent_source"] = "0" * 40
        index.write_text(json.dumps(receipt))
    with pytest.raises(AxBenchAdvError, match=match):
        campaign.run(runtime=forbidden, judge=forbidden)
    assert not (campaign.root / "child/manifest.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("master_seed", 43), ("worker_max_prompt_tokens", 512), ("counter_magnitudes", [0.25, 1.0])],
)
def test_config_mismatch_before_factories(campaign, field, value):
    campaign.seal()
    with pytest.raises(AxBenchAdvError, match="config differs"):
        campaign.run(
            runtime=forbidden,
            judge=forbidden,
            config_change=lambda config: replace(config, **{field: value}),
        )


def test_missing_final_judgment_is_not_published(campaign):
    campaign.seal()

    def interrupted_final(**kwargs):
        raise AxBenchAdvError("missing instruction cache entry")

    with pytest.raises(AxBenchAdvError, match="missing instruction"):
        campaign.run(runtime=forbidden, judge=interrupted_final)
    assert not (campaign.root / "child/manifest.json").exists()


def test_parent_mutation_after_preflight_prevents_publication(campaign):
    campaign.seal()
    judge = fixture._CounterJudge()

    def changed_parent(**kwargs):
        result = judge(**kwargs)
        (campaign.snapshot / "config.yaml").write_text("changed")
        return result

    with pytest.raises(AxBenchAdvError, match="snapshot changed"):
        campaign.run(runtime=forbidden, judge=changed_parent)
    assert not (campaign.root / "child/manifest.json").exists()


def test_final_missing_parquet_refuses_publication(campaign):
    campaign.seal()
    judge = fixture._CounterJudge()

    def incomplete(**kwargs):
        result = judge(**kwargs)
        (kwargs["output_dir"] / "steering_data.parquet").unlink()
        return result

    with pytest.raises(AxBenchAdvError, match="both required outputs"):
        campaign.run(runtime=forbidden, judge=incomplete)
    assert not (campaign.root / "child/manifest.json").exists()


def test_missing_retained_judgment_in_both_caches_fails_before_factories(campaign):
    for cache in (campaign.cache, campaign.snapshot / "judge-cache.sqlite"):
        with sqlite3.connect(cache) as db:
            db.execute("DELETE FROM judge_responses WHERE response LIKE 'concept Rating:%'")
    campaign.seal()
    with pytest.raises(AxBenchAdvError, match="retained concept judgment missing"):
        campaign.run(runtime=forbidden, judge=forbidden)


def test_snapshot_hash_tampering_refused(campaign):
    campaign.seal()
    point = next((campaign.snapshot / "output/counter").rglob("point.json"))
    point.write_text(point.read_text() + "\n")
    with pytest.raises(AxBenchAdvError, match="snapshot changed"):
        campaign.run(runtime=forbidden, judge=forbidden)


def test_one_saved_zero_suffices_without_regeneration(campaign):
    point = (
        campaign.snapshot / "output/counter/diffmean_counter/SelfConditioning/c0/m0.0/point.json"
    )
    point.unlink()
    campaign.seal()
    manifest = campaign.run(runtime=forbidden)
    assert json.loads(manifest.read_text())["call_accounting"]["control_generation_calls"] == 0
    assert (manifest.parent / point.relative_to(campaign.snapshot / "output")).is_file()


def test_cpu_preflight_reports_missing_grid_without_factories(campaign):
    point = (
        campaign.snapshot
        / "output/counter/linear_probe_counter/SelfConditioning/c0/m0.25/point.json"
    )
    point.unlink()
    campaign.seal()
    receipt = campaign.run(runtime=forbidden, judge=forbidden, preflight=True)
    doc = json.loads(receipt.read_text())
    assert doc["missing_generation_points"] == [
        ["linear_probe_counter", "SelfConditioning", 0, 0.25]
    ]
    assert doc["remaining_generation_batches"] == 1
    assert doc["remaining_generation_rows"] == 10
    assert not (receipt.parent / "manifest.json").exists()
    assert not (receipt.parent / "counter").exists()
