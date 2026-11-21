"""Opt-in full-model, real-source, real-cache GEPA zero-control replay.

AXBENCH_ADV_MASKED_ZERO_CONFIG points to an operator-pinned JSON containing
reference_root, reference_manifest_sha256, judge_cache, and concept_id. The
reference is a completed masked-step-1125 GEPA slice, not a synthetic fixture.
Every scientific input is authenticated by its original source contract. Both
judging paths are cache-only: a missing recorded response fails, never calls a
provider. This is separate from nonzero direction fitting/qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch as t

from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv.counter import CounterDirection, load_counter_direction
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.pipeline import AxBenchAdvRunConfig, _cell_rows, _worker_payload
from steering.eval.axbench_adv.self_conditioning import SelfConditioningMethodRuntime
from steering.eval.axbench_adv.source_contract import (
    SourceExpectations,
    SourcePaths,
    validate_source_contract,
)
from steering.eval.axbench_adv.types import METHODS, QWEN_METHODS
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient, AxBenchAdvWorkerClient
from steering.eval.axbench_base.config import LLMClientConfig

pytestmark = pytest.mark.slow
_MODEL_SHA256 = "795a5b32687b0c6badb9b849fcadb9acf3d66eca54265ef24bc68ba3b2493da0"
_TARGET_TAG = "full3b_clustered_c2048_long_w32_bf16_b14_fp"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def masked_zero_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    config_path = os.environ.get("AXBENCH_ADV_MASKED_ZERO_CONFIG")
    if config_path is None:
        pytest.skip("set AXBENCH_ADV_MASKED_ZERO_CONFIG for full masked-3B replay")
    assert t.cuda.is_available(), "explicit GPU qualification requires CUDA"
    config = json.loads(Path(config_path).read_text())
    root = Path(config["reference_root"])
    assert _sha256(root / "manifest.json") == config["reference_manifest_sha256"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["max_concepts"] == 50
    assert manifest["runtime_source_hashes"]["target_model"] == _MODEL_SHA256
    selected = (
        "source_contract.json",
        "points/p0/input/steering_data.parquet",
        "points/p0/judged/steering.jsonl",
    )
    for relative in selected:
        assert _sha256(root / relative) == manifest["files"][relative], relative
    contract = json.loads((root / "source_contract.json").read_text())
    paths = SourcePaths(**{
        field.name: Path(contract["paths"][field.name]) for field in fields(SourcePaths)
    })
    expectation_values: dict[str, Any] = {
        **contract["expectations"],
        "project_target_step": 1125,
        "project_target_model_tag": _TARGET_TAG,
        "model_revision": "it-soup-APO",
    }
    expectations = SourceExpectations(**expectation_values)
    temporary = tmp_path_factory.mktemp("masked-zero")
    validated = validate_source_contract(
        paths=paths,
        expectations=expectations,
        expected_hashes=contract["hashes"],
        output_dir=temporary / "qualification",
    )
    assert dict(validated.runtime_hashes) == manifest["runtime_source_hashes"]
    assert len(validated.training_texts) == 50
    assert validated.method_setup.segment_mode.value == "turn_delimited_fixed_tokens"
    assert validated.method_setup.turn_self_cond_segment_size_rustbpe == 33
    assert validated.method_setup.feature_vocab is not None
    assert validated.method_setup.feature_vocab.size == 844
    frame = pd.read_parquet(root / selected[1])
    entries = [json.loads(line) for line in (root / selected[2]).read_text().splitlines()]
    concept_id = config["concept_id"]
    assert type(concept_id) is int and 0 <= concept_id < 50
    assert concept_id in frame.concept_id.unique()
    entries_by_id = {entry["concept_id"]: entry for entry in entries}
    assert len(entries_by_id) == len(entries)
    run_config = AxBenchAdvRunConfig(
        source_paths=paths,
        source_expectations=expectations,
        source_hashes=contract["hashes"],
        output_dir=temporary / "qualification",
        judge_cache=Path(config["judge_cache"]),
        reflection_cache=temporary / "unused-reflection.sqlite3",
        reflection=LLMClientConfig(model="openai/gpt-5.6-luna"),
        master_seed=manifest["master_seed"],
        gepa_seed=manifest["gepa_seed"],
        worker_max_prompt_tokens=2048,
        worker_timeout_seconds=300,
        cache_only=True,
    )
    assert run_config.judge_cache.is_file()
    return {
        "manifest": manifest,
        "validated": validated,
        "config": run_config,
        "frame": frame,
        "entry": entries_by_id[concept_id],
        "concept_id": concept_id,
        "temporary": temporary,
    }


@pytest.fixture(scope="module")
def project_zero_runtime(masked_zero_case: dict[str, Any]):
    case = masked_zero_case
    validated = case["validated"]
    config = case["config"]
    with SelfConditioningMethodRuntime(
        setup=validated.method_setup,
        training=validated.training_texts,
        main_frame=validated.main_frame,
        max_concepts=50,
        max_new_tokens=128,
        master_seed=config.master_seed,
        device="cuda:0",
        extraction_batch_size=validated.extraction_batch_size,
        judge_factory=AxBenchAdvJudgeClient,
        judge_cache=config.judge_cache,
        judge_model="openai/gpt-5.6-luna",
        api_base_url=config.api_base_url,
        cache_only=True,
        timeout_seconds=300,
        stderr_dir=case["temporary"] / "project-logs",
    ) as runtime:
        model = runtime._target_model
        assert sum(parameter.numel() for parameter in model.parameters()) > 2_500_000_000
        assert model.dec_config.n_layer == 27 and model.dec_config.n_embd == 1664
        assert next(model.parameters()).device.type == "cuda"
        assert len(runtime._originals) == 500
        assert set(runtime._dense_diffmeans) == set(range(50))
        yield runtime


def _assert_reference(case: dict[str, Any], method: str, actual: tuple[RawEvaluation, ...]) -> None:
    frame = case["frame"]
    concept_frame = frame.loc[frame.concept_id.eq(case["concept_id"])]
    factor = case["manifest"]["best_factors"][method]
    selected = concept_frame.factor.eq(factor).to_numpy()
    reference = concept_frame.loc[selected]
    assert reference.input_id.tolist() == list(range(10))
    assert len(actual) == 10
    block = case["entry"]["results"]["LMJudgeEvaluator"][method]
    concept_ratings = [value for keep, value in zip(selected, block["raw_relevance_concept_ratings"], strict=True) if keep]
    fluency_ratings = [value for keep, value in zip(selected, block["raw_fluency_ratings"], strict=True) if keep]
    assert [row.generation for row in actual] == reference[f"{method}_steered_generation"].tolist()
    assert [row.concept_rating for row in actual] == concept_ratings
    assert [row.fluency_rating for row in actual] == fluency_ratings
    assert sum(row.paid_judge_cache_misses for row in actual) == 0


@pytest.mark.parametrize("method", METHODS)
def test_full_masked_campaign_exact_gepa_zero_and_counter_zero(
    method: str, masked_zero_case: dict[str, Any], request: pytest.FixtureRequest,
) -> None:
    case = masked_zero_case
    config = case["config"]
    cid = case["concept_id"]
    factor = case["manifest"]["best_factors"][method]
    concept, worker_examples, optimization_examples = _cell_rows(
        case["validated"], method=method, concept_id=cid, factor=factor,
    )
    input_ids = list(range(10))
    if method in QWEN_METHODS:
        with AxBenchAdvWorkerClient(
            init_payload=_worker_payload(config, method=method, concept_id=cid, concept=concept, factor=factor),
            examples=worker_examples,
            master_seed=config.master_seed,
            timeout_seconds=300,
            stderr_path=case["temporary"] / f"{method}.log",
        ) as worker:
            control = worker.evaluate("", input_ids)
            _assert_reference(case, method, control)
            for pressure in (Pressure.LINEAR_PROBE_COUNTER, Pressure.DIFFMEAN_COUNTER):
                counter = load_counter_direction(
                    pressure=pressure, method=method, concept_id=cid,
                    baseline_train=config.source_paths.main_parquet.parent.parent / "train",
                    baseline_layer=28, project_fit=None,
                )
                assert worker.evaluate_counter(counter, 0.0, input_ids) == control
    else:
        runtime = request.getfixturevalue("project_zero_runtime")
        with runtime.bind_cell(
            method=method, concept_id=cid, concept=concept, factor=factor,
            examples=optimization_examples,
        ) as cell:
            control = cell.evaluate("", input_ids)
            _assert_reference(case, method, control)
            # A nonzero sentinel proves magnitude zero bypasses installation;
            # real fitted artifact identity/nonzero effects have separate tests.
            sentinel = CounterDirection(t.ones(1664), {"layer": 21})
            assert cell.evaluate_counter(sentinel, 0.0, input_ids) == control
            assert runtime._target_model._post_cross_attention_eval_callback is None
