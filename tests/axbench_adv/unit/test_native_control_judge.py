"""Control replay preserves paid results and fails closed on broken lineage."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from steering.eval.axbench_adv import native_control_judge as replay
from steering.eval.axbench_adv import native_method_judge as runner
from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.final_judge import (
    FinalJudgeResult,
    build_point_frame,
    validate_final_judge_outputs,
)
from steering.eval.axbench_adv.native_mailbox import read_json, write_json
from steering.eval.axbench_adv.native_methods import (
    _METHOD_SLUG,
    FORMAT,
    METHOD_FACTORS,
    METHODS,
    UNION_FORMAT,
    _point_emissions_document,
)
from steering.eval.axbench_upstream.native_contract import sha256
from tests.axbench_adv.unit.test_final_judge import _outputs, _raw, _source


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-control-union")
    source = _source(100)
    scores = []
    for method in METHODS:
        destination = root / _METHOD_SLUG[method]
        frames = []
        judged_frames = []
        for factor in METHOD_FACTORS[method]:
            emissions = {
                (method, c, i): _raw(method, c % 2, i) for c in range(100) for i in range(10)
            }
            table = build_point_frame(
                source_frame=source,
                best_factors={method: factor},
                emissions=emissions,
                max_concepts=100,
            )
            part = root.parent / f"{method}-{factor}"
            _outputs(part, table, emissions, max_concepts=100, factors={method: factor})
            frames.append(table.to_pandas())
            judged_frames.append(pd.read_parquet(part / "steering_data.parquet"))
        generation = (
            pd.concat(frames)
            .sort_values(["concept_id", "input_id", "factor"])
            .reset_index(drop=True)
        )
        judged = (
            pd.concat(judged_frames)
            .sort_values(["concept_id", "input_id", "factor"])
            .reset_index(drop=True)
        )
        destination.mkdir()
        generation.to_parquet(destination / "generation.parquet", index=False)
        judged.to_parquet(destination / "steering_data.parquet", index=False)
        entries: list[dict[str, Any]] = []
        method_scores = []
        for concept, group in judged.groupby("concept_id"):
            block = {"factor": list(METHOD_FACTORS[method])}
            block.update(
                {
                    field: group[f"{method}_{suffix}"].tolist()
                    for suffix, field in replay._EVALUATION_FIELDS.items()
                }
            )
            for name, field in replay.SUMMARY.items():
                suffix = next(k for k, v in replay._EVALUATION_FIELDS.items() if v == field)
                block[name] = [
                    group.loc[group.factor.eq(f), f"{method}_{suffix}"].mean()
                    for f in METHOD_FACTORS[method]
                ]
            entries.append(
                {"concept_id": int(str(concept)), "results": {"LMJudgeEvaluator": {method: block}}}
            )
        for index, factor in enumerate(METHOD_FACTORS[method]):
            method_scores.append(
                {
                    "method": method,
                    "factor": factor,
                    "n_concepts": 100,
                    **{
                        name: sum(
                            e["results"]["LMJudgeEvaluator"][method][name][index] for e in entries
                        )
                        / 100
                        for name in replay.SUMMARY
                    },
                }
            )
        (destination / "steering.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
        pd.DataFrame(method_scores).to_csv(destination / "aggregate.csv", index=False)
        scores.extend(method_scores)
    files = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
    write_json(
        root / "union-provenance.json",
        {
            "format": UNION_FORMAT,
            "native_run_manifest": False,
            "concept_ids": list(range(100)),
            "methods": list(METHODS),
            "provider_calls": 0,
            "parents": [{"remote": "fixture/archive", "sha256": "a" * 64}],
            "files": files,
            "scores": scores,
        },
    )
    return root


@pytest.fixture
def control(tmp_path, baseline):
    factors = {"DiffMean": 0.4, "LinearProbe": 1.5, "LoRA": 1.0}
    identity = {
        "format": FORMAT,
        "methods": list(METHODS),
        "concept_ids": [1],
        "best_factors": factors,
        "proposal_counts": [1],
        "counter_magnitudes": [0.25],
        "source_identity": {
            "union_files": read_json(baseline / "union-provenance.json")["files"],
            "union_provenance_sha256": sha256(baseline / "union-provenance.json"),
        },
    }
    emissions = {(m, 1, i): _raw(m, 1, i) for m in METHODS for i in range(10)}
    table = build_point_frame(
        source_frame=_source(2),
        best_factors=factors,
        emissions=emissions,
        max_concepts=100,
        concept_ids=[1],
    )
    run = tmp_path / "output"
    point = run / "points/p0"
    (point / "input").mkdir(parents=True)
    pq.write_table(table, point / "input/steering_data.parquet")
    write_json(point / "emissions.json", _point_emissions_document(emissions))
    write_json(run / "identity.json", identity)
    return {
        "final_root": baseline,
        "identity": identity,
        "input_table": table,
        "emissions": emissions,
        "output_dir": point / "judged",
    }


def test_replay_original_scores_all_methods_and_resume_without_source(control):
    result = replay.replay_native_control(**control)
    assert result.actual_evaluator_calls == result.paid_judge_cache_misses == 0
    validate_final_judge_outputs(
        input_table=control["input_table"],
        output_dir=control["output_dir"],
        best_factors=control["identity"]["best_factors"],
        emissions=control["emissions"],
        max_concepts=100,
        concept_ids=[1],
    )
    output = control["output_dir"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    judge = Mock(side_effect=AssertionError("no provider"))
    for _ in range(2):
        runner._judge_point(
            run_dir=output.parents[2],
            relative=Path("points/p0"),
            table=control["input_table"],
            emissions=control["emissions"],
            identity=control["identity"],
            cache=Path("unused"),
            cache_only=True,
            api_base_url="unused",
            judge=judge,
        )
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert read_json(output.parent / "judge-receipt.json")[
        "control_replay"
    ] == replay.control_lineage(control["identity"])
    judge.assert_not_called()


def test_runtime_control_uses_ordinary_final_judge_and_cache_only_resume(
    control, monkeypatch
):
    identity = {**control["identity"], "zero_control_source": "runtime"}
    write_json(control["output_dir"].parents[2] / "identity.json", identity)
    replay_call = Mock(side_effect=AssertionError("runtime control must not replay archive"))
    monkeypatch.setattr(runner, "replay_native_control", replay_call)
    calls = []

    def ordinary_judge(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_dir"]
        _outputs(
            output,
            kwargs["input_table"],
            kwargs["emissions"],
            max_concepts=100,
            factors=kwargs["best_factors"],
            concept_ids=tuple(kwargs["concept_ids"]),
        )
        write_json(output / "judge_call_accounting.json", {"total_calls": 90, "total_cache_hits": 90})
        write_json(
            output / "manifest.json",
            {
                "schema_version": 1,
                "best_factors": kwargs["best_factors"],
                "call_accounting": {
                    "actual_evaluator_calls": 90,
                    "paid_judge_cache_misses": 0,
                },
                "files": {
                    "judge_call_accounting.json": {
                        "sha256": sha256(output / "judge_call_accounting.json")
                    },
                    "steering.jsonl": {"sha256": sha256(output / "steering.jsonl")},
                    "steering_data.parquet": {
                        "logical_sha256": logical_table_digest(
                            pq.read_table(output / "steering_data.parquet")
                        )
                    },
                },
                "input_logical_sha256": logical_table_digest(kwargs["input_table"]),
            },
        )
        return FinalJudgeResult(output / "steering.jsonl", 90, 0)

    arguments = {
        "run_dir": control["output_dir"].parents[2],
        "relative": Path("points/p0"),
        "table": control["input_table"],
        "emissions": control["emissions"],
        "identity": identity,
        "cache": Path("unused"),
        "cache_only": True,
        "api_base_url": "unused",
        "judge": ordinary_judge,
        "final_root": control["final_root"],
    }
    result = runner._judge_point(**arguments)
    assert result == control["output_dir"] / "steering.jsonl"
    receipt = read_json(control["output_dir"].parent / "judge-receipt.json")
    assert receipt["zero_control_source"] == "runtime"
    assert receipt["run_identity_sha256"] == replay.digest_json(identity)
    assert "control_replay" not in receipt
    entries = [json.loads(line) for line in result.read_text().splitlines()]
    block = entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
    assert set(block["raw_relevance_instruction_ratings"]) == {None, 1.25}
    assert len(calls) == 1
    replay_call.assert_not_called()

    arguments["judge"] = Mock(side_effect=AssertionError("completed cache must resume"))
    assert runner._judge_point(**arguments) == result
    arguments["judge"].assert_not_called()

    manifest = read_json(control["output_dir"] / "manifest.json")
    write_json(
        control["output_dir"] / "judge_call_accounting.json",
        {"total_calls": 0, "total_cache_hits": 0},
    )
    zero_cost = {
        **manifest,
        "call_accounting": {"actual_evaluator_calls": 0, "paid_judge_cache_misses": 0},
    }
    cast(dict[str, Any], zero_cost["files"])["judge_call_accounting.json"] = {
        "sha256": sha256(control["output_dir"] / "judge_call_accounting.json")
    }
    write_json(control["output_dir"] / "manifest.json", zero_cost)
    with pytest.raises(runner.AxBenchAdvError, match="runtime control final-judge accounting"):
        runner._judge_point(**arguments)

    write_json(
        control["output_dir"] / "judge_call_accounting.json",
        {"total_calls": 90, "total_cache_hits": 90},
    )
    cast(dict[str, Any], manifest["files"])["judge_call_accounting.json"] = {
        "sha256": sha256(control["output_dir"] / "judge_call_accounting.json")
    }
    manifest["control_replay"] = {"format": "native-control-judge-replay-v1"}
    write_json(control["output_dir"] / "manifest.json", manifest)
    with pytest.raises(runner.AxBenchAdvError, match="replay lineage"):
        runner._judge_point(**arguments)


def test_runtime_identity_cannot_call_archived_control_replay(control):
    identity = {**control["identity"], "zero_control_source": "runtime"}
    with pytest.raises(replay.AxBenchAdvError, match="cannot claim archived replay"):
        replay.replay_native_control(
            final_root=control["final_root"],
            identity=identity,
            input_table=control["input_table"],
            emissions=control["emissions"],
            output_dir=control["output_dir"],
        )
    assert not control["output_dir"].exists()


@pytest.mark.parametrize(
    "mutation",
    ["provenance", "files", "source_bytes", "prompt", "generation", "rating", "completion"],
)
def test_replay_rejects_corruption_without_publishing(control, tmp_path, mutation):
    if mutation == "provenance":
        control["identity"]["source_identity"]["union_provenance_sha256"] = "0" * 64
    elif mutation == "files":
        control["identity"]["source_identity"]["union_files"] = {}
    elif mutation == "source_bytes":
        copied = tmp_path / "union"
        shutil.copytree(control["final_root"], copied)
        with (copied / "lora/steering.jsonl").open("a") as stream:
            stream.write(" ")
        control["final_root"] = copied
    elif mutation in ("prompt", "generation"):
        frame = control["input_table"].to_pandas()
        frame.loc[
            0, "original_prompt" if mutation == "prompt" else "DiffMean_steered_generation"
        ] = "changed"
        control["input_table"] = type(control["input_table"]).from_pandas(
            frame, schema=control["input_table"].schema, preserve_index=False
        )
    else:
        key = ("DiffMean", 1, 0)
        raw = control["emissions"][key]
        changes = (
            {"concept_rating": 0.0}
            if mutation == "rating"
            else {
                "judge_rationale": json.dumps(
                    {"concept": "changed", "fluency": json.loads(raw.judge_rationale)["fluency"]}
                )
            }
        )
        control["emissions"][key] = replace(raw, **changes)
    with pytest.raises((ValueError, runner.AxBenchAdvError)):
        replay.replay_native_control(**control)
    assert not control["output_dir"].exists()


@pytest.mark.parametrize("mutation", ["lineage", "coordinate", "accounting"])
def test_archive_receipt_rejects_false_replay_lineage(control, mutation):
    replay.replay_native_control(**control)
    output = control["output_dir"]
    manifest = read_json(output / "manifest.json")
    relative = Path("points/p0")
    if mutation == "lineage":
        manifest["control_replay"]["run_identity_sha256"] = "0" * 64
    elif mutation == "coordinate":
        relative = Path("gepa/points/p1")
    else:
        manifest["call_accounting"]["actual_evaluator_calls"] = 90
    write_json(output / "manifest.json", manifest)
    with pytest.raises(runner.AxBenchAdvError, match="lineage/accounting"):
        runner._validated_judge_receipt(
            output=output,
            source={},
            best_factors=control["identity"]["best_factors"],
            relative=relative,
            identity=control["identity"],
        )


def test_all_points_preflight_before_control_or_provider(control, monkeypatch):
    output = control["output_dir"]
    judge = Mock(side_effect=AssertionError("no provider"))
    control_replay = Mock(side_effect=AssertionError("no replay"))
    monkeypatch.setattr(runner, "replay_native_control", control_replay)
    monkeypatch.setattr(
        runner,
        "_point_specs",
        lambda _identity: [
            (Path("points/p0"), {"pressure": "shared_zero"}),
            (Path("gepa/points/p1"), {"pressure": "gepa"}),
        ],
    )
    monkeypatch.setattr(
        runner,
        "_validate_point",
        Mock(
            side_effect=[
                (control["input_table"], control["emissions"]),
                runner.AxBenchAdvError("missing GEPA point"),
            ]
        ),
    )
    cfg = OmegaConf.create(
        {
            "run_dir": str(output.parents[2]),
            "judge_cache": str(output.parents[3] / "cache.sqlite"),
            "api_base_url": "unused",
            "cache_only": False,
        }
    )
    with pytest.raises(runner.AxBenchAdvError, match="missing GEPA"):
        runner.run_native_method_final_judge(cfg, judge=judge)
    control_replay.assert_not_called()
    judge.assert_not_called()


def test_runner_replays_control_and_accepts_relocated_receipt(control, tmp_path):
    output = control["output_dir"]
    judge = Mock(side_effect=AssertionError("no provider"))
    arguments = {
        "relative": Path("points/p0"),
        "table": control["input_table"],
        "emissions": control["emissions"],
        "identity": control["identity"],
        "cache": Path("unused"),
        "cache_only": False,
        "api_base_url": "unused",
        "judge": judge,
        "final_root": control["final_root"],
    }
    runner._judge_point(run_dir=output.parents[2], **arguments)
    receipt_bytes = (output.parent / "judge-receipt.json").read_bytes()
    relocated = tmp_path / "relocated"
    shutil.copytree(output.parents[2], relocated)
    arguments["final_root"] = None
    assert (
        runner._judge_point(run_dir=relocated, **arguments)
        == relocated / "points/p0/judged/steering.jsonl"
    )
    assert (relocated / "points/p0/judge-receipt.json").read_bytes() == receipt_bytes
    receipt = read_json(relocated / "points/p0/judge-receipt.json")
    receipt["steering_jsonl"] = "/wrong/gepa/points/p1/judged/steering.jsonl"
    write_json(relocated / "points/p0/judge-receipt.json", receipt)
    with pytest.raises(runner.AxBenchAdvError, match="receipt differs"):
        runner._judge_point(run_dir=relocated, **arguments)
    judge.assert_not_called()


def test_missing_control_source_never_falls_back_to_paid_judge(control):
    judge = Mock(side_effect=AssertionError("no provider"))
    with pytest.raises(runner.AxBenchAdvError, match="requires final_root"):
        runner._judge_point(
            run_dir=control["output_dir"].parents[2],
            relative=Path("points/p0"),
            table=control["input_table"],
            emissions=control["emissions"],
            identity=control["identity"],
            cache=Path("unused"),
            cache_only=False,
            api_base_url="unused",
            judge=judge,
        )
    judge.assert_not_called()
    assert not control["output_dir"].exists()
