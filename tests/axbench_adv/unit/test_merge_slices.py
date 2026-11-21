"""Completed AxBench-Adv slice merger tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.eval.axbench_adv.merge_slices import main as merge_main
from scripts.eval.axbench_adv.merge_slices import merge_slices
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import build_point_frame
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import ALL_METHODS, METHODS, PLOT_EFFORT_LEVELS

EVALUATION_SUFFIXES = (
    "LMJudgeEvaluator",
    "LMJudgeEvaluator_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_ratings",
    "LMJudgeEvaluator_relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_ratings",
    "LMJudgeEvaluator_fluency_completions",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compatibility() -> dict[str, object]:
    return {
        "schema_version": 2,
        "proposal_counts": [1, 2, 4, 8],
        "max_concepts": 2,
        "master_seed": 42,
        "gepa_seed": 42,
        "attack_config": {
            "reflection": {"provider": "openrouter", "model": "reflection-model"},
            "seed_suffix": "seed suffix",
            "reflection_minibatch_size": 3,
        },
        "implementation": {
            "repository_git_commit": "d" * 40,
            "package_lock_sha256": "e" * 64,
        },
        "worker_max_prompt_tokens": 1024,
        "source_hashes": {"source": "a" * 64},
        "runtime_source_hashes": {"runtime": "b" * 64},
        "runtime_source_byte_lengths": {"runtime": 1},
        "source_project": {
            "repository_git_commit": "c" * 40,
            "package_lock_sha256": "e" * 64,
            "self_conditioning_masked_policy": "fixture-policy",
        },
    }


def _write_run(
    root: Path,
    *,
    methods: tuple[str, ...],
    concept_ids: tuple[int, ...] = (0, 1),
    declare_concept_ids: bool = True,
) -> None:
    factors = dict.fromkeys(methods, 1.0)
    files: dict[str, str] = {}
    source = pd.DataFrame([
        {
            "dataset_name": "AlpacaEval",
            "concept_id": concept_id,
            "input_concept": f"concept-{concept_id}",
            "input_id": input_id,
            "factor": 1.0,
            "original_prompt": f"prompt {concept_id}/{input_id}",
        }
        for concept_id in concept_ids
        for input_id in range(10)
    ])
    for level in PLOT_EFFORT_LEVELS:
        point_root = root / "points" / f"p{level}"
        input_path = point_root / "input" / "steering_data.parquet"
        judged_dir = point_root / "judged"
        input_path.parent.mkdir(parents=True)
        judged_dir.mkdir(parents=True)
        emissions = {
            (method, concept_id, input_id): RawEvaluation(
                generation=f"{method} generated {concept_id}/{input_id}/p{level}",
                concept_rating=1.0 - 0.01 * level,
                fluency_rating=1.5,
                judge_rationale=json.dumps({
                    "concept": f"concept Rating: [[{1.0 - 0.01 * level}]]",
                    "fluency": "fluency Rating: [[1.5]]",
                }),
                paid_judge_cache_misses=0,
            )
            for method in methods
            for concept_id in concept_ids
            for input_id in range(10)
        }
        table = build_point_frame(
            source_frame=source,
            best_factors=factors,
            emissions=emissions,
            max_concepts=2,
            concept_ids=concept_ids,
        )
        pq.write_table(table, input_path)
        evaluated = table.to_pandas()
        entries: list[dict[str, Any]] = []
        for concept_id in concept_ids:
            concept = 1.0 - 0.01 * level
            instruction = 1.25
            fluency = 1.5
            aggregate = 3.0 / (1.0 / concept + 1.0 / instruction + 1.0 / fluency)
            blocks = {
                method: {
                    "factor": [1.0],
                    "lm_judge_rating": [aggregate],
                    "relevance_concept_ratings": [concept],
                    "relevance_instruction_ratings": [instruction],
                    "fluency_ratings": [fluency],
                    "raw_relevance_concept_ratings": [concept] * 10,
                    "raw_relevance_instruction_ratings": [instruction] * 10,
                    "raw_fluency_ratings": [fluency] * 10,
                    "raw_aggregated_ratings": [aggregate] * 10,
                    "relevance_concept_completions": [
                        f"concept Rating: [[{concept}]]"
                    ] * 10,
                    "relevance_instruction_completions": [
                        "instruction Rating: [[1.25]]"
                    ] * 10,
                    "fluency_completions": ["fluency Rating: [[1.5]]"] * 10,
                }
                for method in methods
            }
            entries.append(
                {"concept_id": concept_id, "results": {"LMJudgeEvaluator": blocks}}
            )
        jsonl_path = judged_dir / "steering.jsonl"
        jsonl_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        for method in methods:
            block_rows = [
                entry["results"]["LMJudgeEvaluator"][method]
                for entry in entries
            ]
            for suffix, field in zip(
                EVALUATION_SUFFIXES,
                (
                    "raw_aggregated_ratings",
                    "raw_relevance_concept_ratings",
                    "relevance_concept_completions",
                    "raw_relevance_instruction_ratings",
                    "relevance_instruction_completions",
                    "raw_fluency_ratings",
                    "fluency_completions",
                ),
                strict=True,
            ):
                evaluated[f"{method}_{suffix}"] = [
                    value
                    for block in block_rows
                    for value in block[field]
                ]
        judged_path = judged_dir / "steering_data.parquet"
        evaluated.to_parquet(judged_path, index=False)
        for path in (input_path, jsonl_path, judged_path):
            files[str(path.relative_to(root))] = _sha256(path)
    source_contract = {
        "schema_version": 2,
        "hashes": {"source": "a" * 64},
        "runtime_hashes": {"runtime": "b" * 64},
        "runtime_byte_lengths": {"runtime": 1},
        "project_identity": {
            "repository_git_commit": "c" * 40,
            "package_lock_sha256": "e" * 64,
            "self_conditioning_masked_policy": "fixture-policy",
        },
    }
    attachment = {"schema_version": 1, "identity": "fixture"}
    if "NLSteering" in methods:
        source_contract["nl_source_attachment"] = {
            **attachment,
            "paths": {"fixture": "/source/path"},
            "runtime_byte_lengths": {"fixture": 1},
        }
    source_contract_path = root / "source_contract.json"
    source_contract_path.write_text(json.dumps(source_contract, sort_keys=True))
    files["source_contract.json"] = _sha256(source_contract_path)
    manifest = {
        **_compatibility(),
        "methods": list(methods),
        **({"concept_ids": list(concept_ids)} if declare_concept_ids else {}),
        "best_factors": factors,
        "files": files,
        **({"nl_source_attachment": attachment} if "NLSteering" in methods else {}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))


@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_merge_disjoint_baseline_and_nl_methods_for_same_concepts(
    tmp_path: Path, reverse_inputs: bool,
) -> None:
    baseline = tmp_path / "baseline"
    nl = tmp_path / "nl"
    _write_run(baseline, methods=METHODS)
    _write_run(nl, methods=("NLSteering",))

    roots = (nl, baseline) if reverse_inputs else (baseline, nl)
    manifest_path = merge_slices(roots, tmp_path / "merged")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["methods"] == list(ALL_METHODS)
    assert manifest["concept_ids"] == [0, 1]
    aggregate = pq.read_table(manifest_path.parent / "aggregate.parquet")
    assert aggregate.num_rows == len(ALL_METHODS) * len(PLOT_EFFORT_LEVELS)
    assert set(aggregate.column("method").to_pylist()) == set(ALL_METHODS)
    assert len(list((manifest_path.parent / "plots").glob("*.png"))) == 1 + len(
        ALL_METHODS
    )


def test_merge_refuses_overlapping_method_concept_ownership(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, methods=("DiffMean",))
    _write_run(second, methods=("DiffMean",))

    with pytest.raises(AxBenchAdvError, match=r"overlap the same \(method, concept\)"):
        merge_slices((first, second), tmp_path / "merged")


def test_merge_combines_disjoint_concept_slices_for_same_method(tmp_path: Path) -> None:
    low = tmp_path / "low"
    high = tmp_path / "high"
    _write_run(low, methods=("DiffMean",), concept_ids=(0,))
    _write_run(high, methods=("DiffMean",), concept_ids=(1,))

    manifest_path = merge_slices((low, high), tmp_path / "merged")

    aggregate = pq.read_table(manifest_path.parent / "aggregate.parquet")
    assert aggregate.num_rows == len(PLOT_EFFORT_LEVELS)
    assert set(aggregate.column("n_concepts").to_pylist()) == {2}


def test_merge_requires_matching_final_concept_sets_across_methods(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    nl = tmp_path / "nl"
    _write_run(baseline, methods=METHODS, concept_ids=(0,))
    _write_run(nl, methods=("NLSteering",), concept_ids=(0, 1))

    with pytest.raises(AxBenchAdvError, match="concept sets do not match"):
        merge_slices((baseline, nl), tmp_path / "merged")


@pytest.mark.parametrize(
    "keys,value",
    [
        (("master_seed",), 43),
        (("gepa_seed",), 43),
        (("max_concepts",), 3),
        (("source_hashes", "source"), "f" * 64),
        (("runtime_source_hashes", "runtime"), "f" * 64),
        (("runtime_source_byte_lengths", "runtime"), 2),
        (("source_project", "self_conditioning_masked_policy"), "different-policy"),
    ],
)
def test_merge_rejects_different_scientific_inputs(
    tmp_path: Path, keys: tuple[str, ...], value: object,
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    path = roots[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    target = manifest
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(manifest))

    expected = (
        "masked policies"
        if keys[:2] == ("source_project", "self_conditioning_masked_policy")
        else f"incompatible .*fields.*{keys[0]}"
    )
    with pytest.raises(AxBenchAdvError, match=expected):
        merge_slices(roots, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_records_but_does_not_gate_on_execution_metadata(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    path = roots[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["attack_config"] = {"historical": "unknown"}
    manifest["implementation"] = {"repository_git_commit": "different"}
    manifest["worker_max_prompt_tokens"] = None
    manifest["source_project"]["repository_git_commit"] = "different"
    manifest["source_project"]["package_lock_sha256"] = "different"
    path.write_text(json.dumps(manifest))

    merged = json.loads(merge_slices(roots, tmp_path / "merged").read_text())

    assert [item["manifest_sha256"] for item in merged["input_roots"]] == [
        _sha256(root / "manifest.json") for root in roots
    ]


def test_merge_rejects_different_single_source_nl_identity(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
        identity = {"identity": f"retrieval-{concept_id}"}
        contract_path = root / "source_contract.json"
        contract = json.loads(contract_path.read_text())
        contract["natural_language"] = identity
        contract_path.write_text(json.dumps(contract, sort_keys=True))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["natural_language"] = identity
        manifest["files"]["source_contract.json"] = _sha256(contract_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(AxBenchAdvError, match="incompatible fields.*natural_language"):
        merge_slices(roots, tmp_path / "merged")


@pytest.mark.parametrize(
    "keys",
    [
        ("master_seed",),
        ("gepa_seed",),
        ("source_hashes",),
        ("runtime_source_hashes",),
        ("runtime_source_byte_lengths",),
        ("source_project",),
        ("source_project", "self_conditioning_masked_policy"),
    ],
)
def test_merge_rejects_missing_identity_even_when_both_slices_match(
    tmp_path: Path, keys: tuple[str, ...],
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        target = manifest
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]
        path.write_text(json.dumps(manifest))

    with pytest.raises(AxBenchAdvError, match=keys[0]):
        merge_slices(roots, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_explicit_null_concept_scope(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["concept_ids"] = None
        path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError, match="concept_ids"):
        merge_slices(roots, tmp_path / "merged")


@pytest.mark.parametrize("field,value", [
    ("master_seed", True), ("master_seed", -1), ("gepa_seed", 1 << 63),
    ("source_hashes", None), ("source_hashes", {}), ("runtime_source_hashes", []),
    ("runtime_source_byte_lengths", {}), ("source_project", "missing"),
])
def test_merge_rejects_matching_but_malformed_provenance(
    tmp_path: Path, field: str, value: object,
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest[field] = value
        path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError, match=field):
        merge_slices(roots, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("mutation", ["hash", "level_ids", "methods", "factor", "overlap", "nonfresh"])
def test_merge_rejects_corrupt_cells_or_output_before_publication(
    tmp_path: Path, mutation: str,
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative = "points/p8/judged/steering.jsonl"
    path = roots[1] / relative
    entry = json.loads(path.read_text())
    output = tmp_path / "merged"
    if mutation == "hash":
        path.write_text(path.read_text() + "\n")
    elif mutation == "level_ids":
        entry["concept_id"] = 0
    elif mutation == "methods":
        entry["results"]["LMJudgeEvaluator"] = {}
    elif mutation == "factor":
        manifest["best_factors"]["DiffMean"] = 2.0
    elif mutation == "overlap":
        output = roots[0] / "merged"
    else:
        output.mkdir()
        (output / "keep.txt").write_text("user data")
    if mutation in {"level_ids", "methods"}:
        path.write_text(json.dumps(entry) + "\n")
        manifest["files"][relative] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError):
        merge_slices(roots, output)
    assert not (output / "manifest.json").exists()
    if mutation == "nonfresh":
        assert (output / "keep.txt").read_text() == "user data"


@pytest.mark.parametrize("field,value", [
    ("schema_version", 1), ("proposal_counts", [0, 1, 2]), ("max_concepts", 0),
    ("methods", "DiffMean"), ("methods", [False]),
    ("methods", ["NLSteering", "DiffMean"]),
    ("best_factors", {}), ("best_factors", {"DiffMean": True}),
    ("best_factors", {"DiffMean": 99.0}), ("best_factors", {"DiffMean": float("nan")}),
    ("files", []),
])
def test_merge_rejects_malformed_manifest_fields(tmp_path: Path, field: str, value: object) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    path = roots[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError):
        merge_slices(roots, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("payload", [None, "{", "[]"])
def test_merge_rejects_unreadable_manifest(tmp_path: Path, payload: str | None) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    roots[0].mkdir()
    if payload is not None:
        (roots[0] / "manifest.json").write_text(payload)
    with pytest.raises(AxBenchAdvError):
        merge_slices(roots, tmp_path / "merged")


@pytest.mark.parametrize("entry", [{"concept_id": True}, [], {"concept_id": 0, "results": {}}])
def test_undeclared_slice_rejects_malformed_judged_entries(tmp_path: Path, entry: object) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    manifest_path = roots[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["concept_ids"]
    relative = "points/p0/judged/steering.jsonl"
    path = roots[0] / relative
    path.write_text("\n" + json.dumps(entry) + "\n")
    manifest["files"][relative] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AxBenchAdvError):
        merge_slices(roots, tmp_path / "merged")


@pytest.mark.parametrize("duplicate", [False, True])
def test_merge_requires_multiple_distinct_roots(tmp_path: Path, duplicate: bool) -> None:
    root = tmp_path / "run"
    with pytest.raises(AxBenchAdvError, match="two distinct input roots"):
        merge_slices((root, root) if duplicate else (root,), tmp_path / "merged")


def test_merge_cli_accepts_existing_empty_output_directory(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    output = tmp_path / "merged"
    output.mkdir()
    merge_main(["--input-root", str(roots[0]), "--input-root", str(roots[1]), "--output-root", str(output)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["concept_ids"] == [0, 1]
    assert all(_sha256(output / name) == digest for name, digest in manifest["files"].items())


def test_merge_accepts_undeclared_nonzero_concept_slices(tmp_path: Path) -> None:
    # Slices produced before the manifest declared ``concept_ids`` cover arbitrary concept ranges;
    # their judged files must be validated against the identities they carry, not ``range(n)``.
    low = tmp_path / "low"
    high = tmp_path / "high"
    _write_run(low, methods=("DiffMean",), concept_ids=(1,), declare_concept_ids=False)
    _write_run(high, methods=("DiffMean",), concept_ids=(0,), declare_concept_ids=False)

    manifest_path = merge_slices((low, high), tmp_path / "merged")

    merged = json.loads(manifest_path.read_text())
    assert merged["concept_ids"] == [0, 1]


def test_merge_does_not_require_duplicate_source_contract_values(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    contract_path = roots[1] / "source_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["hashes"]["source"] = "f" * 64
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["source_contract.json"] = _sha256(contract_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    assert merge_slices(roots, tmp_path / "merged").is_file()


def test_merge_ignores_source_contract_runtime_length_copy(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    contract_path = roots[1] / "source_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["runtime_byte_lengths"]["runtime"] = True
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["source_contract.json"] = _sha256(contract_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    assert merge_slices(roots, tmp_path / "merged").is_file()


def test_merge_rejects_bool_alias_in_proposal_counts(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["proposal_counts"][0] = True
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(AxBenchAdvError, match="proposal_counts"):
        merge_slices(roots, tmp_path / "merged")


def test_merge_uses_outer_nl_attachment_identity_without_contract_copy(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    nl = tmp_path / "nl"
    _write_run(baseline, methods=METHODS, concept_ids=(0,))
    _write_run(nl, methods=("NLSteering",), concept_ids=(0,))
    contract_path = nl / "source_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["nl_source_attachment"]["identity"] = "different"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    manifest_path = nl / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["source_contract.json"] = _sha256(contract_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    assert merge_slices((baseline, nl), tmp_path / "merged").is_file()


def test_merge_does_not_require_source_contract_file(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].pop("source_contract.json")
        (root / "source_contract.json").unlink()
        manifest_path.write_text(json.dumps(manifest))

    assert merge_slices(roots, tmp_path / "merged").is_file()


def test_merge_validates_every_gepa_point_frame_not_only_control(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    input_path = roots[1] / "points/p1/input/steering_data.parquet"
    frame = pq.read_table(input_path).to_pandas().iloc[:-1]
    frame.to_parquet(input_path, index=False)
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][str(input_path.relative_to(roots[1]))] = _sha256(input_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(AxBenchAdvError, match="GEPA raw judge fields|canonical union-factor"):
        merge_slices(roots, tmp_path / "merged")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "not valid JSON"),
        ("nonfinite", "not valid JSON"),
        ("alias", "file path is unsafe"),
    ],
)
def test_merge_rejects_ambiguous_json_and_noncanonical_file_aliases(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    manifest_path = roots[1] / "manifest.json"
    if mutation == "duplicate":
        manifest_path.write_text(
            '{"schema_version":2,' + manifest_path.read_text().lstrip("{")
        )
    elif mutation == "nonfinite":
        manifest = json.loads(manifest_path.read_text())
        manifest["master_seed"] = float("nan")
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest = json.loads(manifest_path.read_text())
        canonical = "points/p0/input/steering_data.parquet"
        manifest["files"]["points/./p0/input/steering_data.parquet"] = manifest["files"][
            canonical
        ]
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(AxBenchAdvError, match=match):
        merge_slices(roots, tmp_path / "merged")


def test_merge_rejects_duplicate_keys_in_judged_jsonl(tmp_path: Path) -> None:
    roots = (tmp_path / "low", tmp_path / "high")
    for concept_id, root in enumerate(roots):
        _write_run(root, methods=("DiffMean",), concept_ids=(concept_id,))
    judged_path = roots[1] / "points/p1/judged/steering.jsonl"
    judged_path.write_text(
        judged_path.read_text().replace(
            '"concept_id": 1', '"concept_id": 1, "concept_id": 1', 1
        )
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][str(judged_path.relative_to(roots[1]))] = _sha256(judged_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(AxBenchAdvError, match="invalid JSON.*duplicate JSON object key"):
        merge_slices(roots, tmp_path / "merged")
