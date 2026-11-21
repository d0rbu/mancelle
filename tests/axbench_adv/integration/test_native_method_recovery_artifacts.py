"""Opt-in, read-only audit of authentic failed native campaign artifacts.

Set NATIVE_ADV_RECOVERY_ARTIFACT_ROOT to method-completion-400m. No provider
client is constructed; all SQLite access uses the vendor's cache-only mode.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import socket
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import _EVALUATION_FIELDS, validate_final_judge_outputs
from steering.eval.axbench_adv.native_control_judge import replay_native_control
from steering.eval.axbench_adv.native_mailbox import read_json
from steering.eval.axbench_adv.native_method_judge import _point_specs, _validate_point
from steering.eval.axbench_upstream.native_contract import sha256


@pytest.fixture
def vendor(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "vendor/axbench/axbench"
    for suffix in ("", ".evaluators", ".models", ".utils"):
        name = "recovery_test_axbench" + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*suffix.lstrip(".").split(".")))]
        monkeypatch.setitem(sys.modules, name, module)
    # The root environment need not install the provider SDK. Cache-only paths
    # use none of it; an empty module makes accidental provider access fail.
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    evaluator = importlib.import_module("recovery_test_axbench.evaluators.lm_judge")
    client = importlib.import_module("recovery_test_axbench.models.language_models")
    templates = importlib.import_module("recovery_test_axbench.evaluators.prompt_templates")
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("network forbidden"))
    return evaluator, client, templates


@pytest.fixture
def artifact_root():
    value = os.environ.get("NATIVE_ADV_RECOVERY_ARTIFACT_ROOT")
    if not value:
        pytest.skip("requires authentic local campaign artifacts")
    return Path(value).resolve()


@pytest.mark.parametrize("job", ["decoder-c000-001", "decoder-c001-002"])
def test_actual_cache_mismatch_and_control_recovery(artifact_root, vendor, tmp_path, job):
    evaluator, client, templates = vendor
    source = artifact_root / "decoder-adv-jobs" / job
    run = source / "output"
    identity = read_json(run / "identity.json")
    cache = source / "judge-cache.sqlite"
    before_cache = sha256(cache)
    # The repair deliberately leaves every generation-fingerprinted source alone.
    checkout = Path(__file__).resolve().parents[3]
    assert all(
        sha256(checkout / name) == digest
        for name, digest in identity["implementation_sha256"].items()
    )
    lm = client.LanguageModel(
        "openai/gpt-5.6-luna",
        None,
        use_cache=False,
        judge_cache=cache,
        cache_only=True,
        temperature=0.0,
    )
    counts = Counter()
    missing_instruction_requests = set()
    by_point = {}
    try:
        for relative, coordinate in _point_specs(identity):
            table, emissions = _validate_point(
                run_dir=run, relative=relative, coordinate=coordinate, identity=identity
            )
            point_counts = Counter()
            for (method, concept, input_id), raw in emissions.items():
                frame = table.to_pandas()
                row = frame.loc[frame.concept_id.eq(concept) & frame.input_id.eq(input_id)].iloc[0]
                rationale = json.loads(raw.judge_rationale)
                for axis, template_name in [
                    ("concept", "CONCEPT_RELEVANCE"),
                    ("fluency", "FLUENCY"),
                    ("instruction", "INSTRUCTION_RELEVANCE"),
                ]:
                    template = getattr(
                        templates, f"UNIDIRECTIONAL_PAIRWISE_EVALUATION_{template_name}_TEMPLATE"
                    )
                    prompt = template.format(
                        concept=row.input_concept,
                        sentence=raw.generation,
                        instruction=row.original_prompt,
                    )
                    request = lm._judge_request(prompt)
                    response = lm.judge_cache.get(request)
                    if axis == "instruction":
                        point_counts[
                            "instruction_hit" if response is not None else "instruction_missing"
                        ] += 1
                        if response is None:
                            missing_instruction_requests.add(json.dumps(request, sort_keys=True))
                    else:
                        assert response is not None
                        same = response == rationale[axis]
                        point_counts[axis + ("_exact" if same else "_different")] += 1
                        if relative != Path("points/p0"):
                            assert same, (str(relative), method, concept, input_id, axis)
                            assert evaluator.parse_judge_rating(response) == getattr(
                                raw, axis + "_rating"
                            )
            by_point[str(relative)] = dict(point_counts)
            counts.update(point_counts)
            if relative == Path("points/p0"):
                old = tmp_path / "failed-replay"
                old.mkdir()
                frame = table.to_pandas()
                evaluated = frame.copy()
                blocks = {}
                for method in identity["methods"]:
                    metrics = evaluator.LMJudgeEvaluator(
                        method, lm_model=lm, concept_id=identity["concept_ids"][0]
                    ).compute_metrics(frame)
                    blocks[method] = metrics
                    for suffix, field in _EVALUATION_FIELDS.items():
                        evaluated[f"{method}_{suffix}"] = metrics[field]
                evaluated.to_parquet(old / "steering_data.parquet", index=False)
                (old / "steering.jsonl").write_text(
                    json.dumps(
                        {
                            "concept_id": identity["concept_ids"][0],
                            "results": {"LMJudgeEvaluator": blocks},
                        }
                    )
                    + "\n"
                )
                with pytest.raises(AxBenchAdvError, match="completion differs from worker"):
                    validate_final_judge_outputs(
                        input_table=table,
                        output_dir=old,
                        best_factors=identity["best_factors"],
                        emissions=emissions,
                        max_concepts=100,
                        concept_ids=identity["concept_ids"],
                    )
                assert lm.stats.total_call == lm.stats.total_cache_hit == 90
                copied = tmp_path / "output"
                point = copied / relative
                shutil.copytree(run / relative, point)
                shutil.copy2(run / "identity.json", copied / "identity.json")
                result = replay_native_control(
                    final_root=artifact_root / "decoder100-final",
                    identity=identity,
                    input_table=table,
                    emissions=emissions,
                    output_dir=point / "judged",
                )
                assert result.actual_evaluator_calls == result.paid_judge_cache_misses == 0
        assert counts["concept_different"] + counts["fluency_different"] > 0
        assert len(by_point) == 13
        print(
            json.dumps(
                {
                    "job": job,
                    "counts": dict(counts),
                    "unique_missing_instruction_requests": len(missing_instruction_requests),
                    "by_point": by_point,
                },
                sort_keys=True,
            )
        )
    finally:
        asyncio.run(lm.close())
    assert sha256(cache) == before_cache
