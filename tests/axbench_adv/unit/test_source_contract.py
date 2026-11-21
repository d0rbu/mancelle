"""Representative source-fence tests against exact synthetic c2 contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from scripts.eval.axbench_upstream.run_baselines import REQUIRED_PREPARED_PATHS
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.eval import turn_context_contract
from steering.eval.axbench_adv import source_contract as source_contract_module
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ADV_METHOD_FACTORS
from steering.eval.axbench_adv.source_contract import (
    SourceExpectations,
    SourcePaths,
    revalidate_source_hashes,
    validate_source_contract,
    write_source_contract,
)
from steering.eval.axbench_upstream import methods
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from steering.remote_training import build_checkpoint_marker
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from steering.stats_io import content_sha256
from tests.scripts.eval.test_axbench_project_methods import (
    _install_turn_setup_fixture,
    _real_lineage_block,
)

COMMIT = "a" * 40
PROJECT_COMMIT = "9" * 40
REVISION = "b" * 40
MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINGERPRINT = "e" * 64
MODEL_TAG = "l22_main_h100_8rank_db16_v1"
ACCEPTED_SELF_CONDITIONING_POLICY = "prompt_last_segment_max_pool_logit_rank_diffmean_lerp_v2"
ACCEPTED_SELF_CONDITIONING_MASKED_POLICY = (
    "prompt_last_segment_max_pool_logit_rank_diffmean_sign_rank_v3"
)
SELF_CONDITIONING_FACTORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
ELIGIBLE_HEAD_ROWS = 8
POSITIVE_MEAN = 2.0
NEGATIVE_MEAN = 3.0
PRODUCER_METHODS = (
    "DiffMean",
    "LinearProbe",
    "PromptSteering",
    "LoRA",
    "SelfConditioning",
    "SelfConditioningMasked",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_limit(factor: float, mean: float) -> int:
    tau = factor / max(SELF_CONDITIONING_FACTORS)
    real_limit = ELIGIBLE_HEAD_ROWS * (1 - tau) + 2 * mean * tau
    return min(ELIGIBLE_HEAD_ROWS, max(1, round(real_limit)))


def _pool_limits() -> list[dict[str, float | int]]:
    return [
        {
            "factor": factor,
            "positive": _pool_limit(factor, POSITIVE_MEAN),
            "negative": _pool_limit(factor, NEGATIVE_MEAN),
        }
        for factor in SELF_CONDITIONING_FACTORS
    ]


def _input_values(concept_id: int, input_id: int, factor: float) -> dict[str, object]:
    values: dict[str, object] = {
        column: f"{column}-{concept_id}-{input_id}" for column in INPUT_COLUMNS
    }
    values.update(
        {
            "dataset_name": "AlpacaEval",
            "concept_id": concept_id,
            "input_concept": f"concept-{concept_id}",
            "input_id": input_id,
            "factor": float(factor),
            "sae_link": f"repo/{concept_id}",
            "sae_id": concept_id,
        }
    )
    return values


def _metric(method: str) -> str:
    return (
        "strength"
        if method
        in {"DiffMean", "LinearProbe", "SelfConditioning", "SelfConditioningMasked"}
        else "perplexity"
    )


def _frame(source: str) -> pd.DataFrame:
    factors = ADV_FULL_FACTORS if source in {"main", "project"} else (1.0,)
    methods = {
        "main": ("DiffMean", "LinearProbe"),
        "aux": ("PromptSteering", "LoRA"),
        "project": ("SelfConditioning", "SelfConditioningMasked"),
    }[source]
    rows: list[dict[str, object]] = []
    for concept_id in range(2):
        for input_id in range(10):
            for factor in factors:
                row = _input_values(concept_id, input_id, factor)
                if source == "aux":
                    row["steered_input"] = f"aux-steered-{concept_id}-{input_id}"
                for method in methods:
                    row[f"{method}_steered_generation"] = "healthy generated response"
                    row[f"{method}_{_metric(method)}"] = (
                        float(factor) if _metric(method) == "strength" else 1.0
                    )
                rows.append(row)
    columns = list(INPUT_COLUMNS)
    for method in methods:
        columns.extend((f"{method}_steered_generation", f"{method}_{_metric(method)}"))
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    for column in ["factor", *(f"{method}_{_metric(method)}" for method in methods)]:
        frame[column] = frame[column].astype("float64")
    return frame


def _result_block(method: str) -> dict[str, object]:
    factors = ADV_METHOD_FACTORS[method]
    declared = [factor in factors for _input_id in range(10) for factor in ADV_FULL_FACTORS]
    block: dict[str, object] = {
        "factor": list(factors),
        "lm_judge_rating": [1.0] * len(factors),
        "relevance_concept_ratings": [1.0] * len(factors),
        "relevance_instruction_ratings": [1.0] * len(factors),
        "fluency_ratings": [1.0] * len(factors),
    }
    for field in (
        "raw_relevance_concept_ratings",
        "raw_relevance_instruction_ratings",
        "raw_fluency_ratings",
        "raw_aggregated_ratings",
    ):
        block[field] = [1.0 if present else None for present in declared]
    for field in (
        "relevance_concept_completions",
        "relevance_instruction_completions",
        "fluency_completions",
    ):
        block[field] = ["Rating: [[1]]" if present else None for present in declared]
    return block


def _entries() -> list[dict[str, object]]:
    methods = {method: _result_block(method) for method in PRODUCER_METHODS}
    return [
        {"concept_id": concept_id, "results": {"LMJudgeEvaluator": methods}}
        for concept_id in range(2)
    ]


def _write_entries(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text("\n".join(map(json.dumps, entries)) + "\n", encoding="utf-8")


def _prepared_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "axbench_upstream_commit": "d" * 40,
        "benchmark": {},
        "alpaca_eval": {},
        "seed_revisions": {},
        "seed_rows": {},
        "stanza": {},
        "artifacts": {},
    }


def _manifest(source: str, prepared: Path) -> dict[str, object]:
    models = ["DiffMean", "LinearProbe"] if source == "main" else ["PromptSteering", "LoRA"]
    stages = ("train", "latent", "steering") if source == "main" else ("train", "steering")
    return {
        "schema_version": 1,
        "config_name": f"qwen2_5_3b_{source}.yaml",
        "dataset_revision": REVISION,
        "max_concepts": 2,
        "seed": 42,
        "target_model": MODEL,
        "repository_git_commit": COMMIT,
        "repository_git_worktree": {"dirty": False},
        "stage_provenance": {
            stage: {
                "repository_git_commit": COMMIT,
                "repository_git_worktree": {"dirty": False},
            }
            for stage in stages
        },
        "prepared_dir": str(prepared.resolve()),
        "prepared_manifest": _prepared_manifest(),
        "package_lock_sha256": "c" * 64,
        "vendored_axbench_upstream_commit": "d" * 40,
        "config": {
            "train": {"model_name": MODEL, "layer": 28, "use_bf16": True},
            "inference": {
                "models": models,
                "seed": 42,
                "temperature": 1.0,
                "steering_num_of_examples": 10,
                "steering_output_length": 128,
                "model_name": MODEL,
                "steering_model_name": MODEL,
                "steering_datasets": ["AlpacaEval"],
                "steering_factors": list(ADV_FULL_FACTORS if source == "main" else (1.0,)),
                "use_bf16": True,
                "steering_intervention_type": "addition",
                "steering_layer": 28,
                "steering_layers": [28],
                "intervene_on_prompt": True,
            },
        },
    }


def _project_manifest(*, main: Path, meta: Path, statistics: Path) -> dict[str, object]:
    return {
        "schema_version": 3,
        "repository_git_commit": PROJECT_COMMIT,
        "repository_git_worktree": {"dirty": False},
        "vendored_axbench_upstream_commit": "d" * 40,
        "dataset_revision": REVISION,
        "prepared_manifest": _prepared_manifest(),
        "package_lock_sha256": "f" * 64,
        "inputs": {
            "main_steering_parquet_sha256": _sha(main),
            "feature_statistics_sha256": content_sha256(statistics),
        },
        "method_order": [
            "DiffMean",
            "LinearProbe",
            "PromptSteering",
            "LoRA",
            "SelfConditioning",
            "SelfConditioningMasked",
        ],
        "methods": {
            "SelfConditioning": {
                "factors": list(ADV_FULL_FACTORS),
                "policy": ACCEPTED_SELF_CONDITIONING_POLICY,
            },
            "SelfConditioningMasked": {
                "factors": list(ADV_FULL_FACTORS),
                "policy": ACCEPTED_SELF_CONDITIONING_MASKED_POLICY,
                "strength_semantics": "linear_rank_pool_limit_from_N_to_2x_side_mean",
                "rounding": "nearest_ties_to_even",
                "clamp": {"minimum": 1, "maximum": "eligible_head_rows"},
                "eligible_head_rows": ELIGIBLE_HEAD_ROWS,
                "positive_mean": POSITIVE_MEAN,
                "negative_mean": NEGATIVE_MEAN,
                "pool_limits": _pool_limits(),
                "candidate_eligibility": (
                    "post_feature_mask_sign_consistent_stable_diffmean_rank"
                ),
                "final_ranking": "last_prompt_segment_max_pooled_float32_model_logits",
                "exact_zero_diffmean_eligible": False,
            },
        },
        "generation": {"temperature": 1.0, "top_k": 50},
        "target": {"step": 54_600, "model_tag": MODEL_TAG, "checkpoint_meta": str(meta)},
        "configuration": {
            "max_concepts": 2,
            "seed": 42,
            "max_new_tokens": 128,
            "self_conditioning_policy": ACCEPTED_SELF_CONDITIONING_POLICY,
            "self_conditioning_masked_policy": ACCEPTED_SELF_CONDITIONING_MASKED_POLICY,
            "force": False,
            "main_path": str(main),
            "target_meta_path": str(meta),
            "feature_statistics_path": str(statistics),
            "extraction_batch_size": 8,
        },
    }


def _fixture(tmp_path: Path) -> tuple[SourcePaths, SourceExpectations, dict[str, str]]:
    prepared = tmp_path / "prepared"
    for relative in REQUIRED_PREPARED_PATHS:
        path = prepared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix:
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir(exist_ok=True)
    (prepared / "prepare_manifest.json").write_text(
        json.dumps(_prepared_manifest()), encoding="utf-8"
    )

    main_path = tmp_path / "main" / "inference" / "steering_data.parquet"
    aux_path = tmp_path / "aux" / "inference" / "steering_data.parquet"
    project_path = tmp_path / "project" / "inference" / "steering_data.parquet"
    for path, source in ((main_path, "main"), (aux_path, "aux"), (project_path, "project")):
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(_frame(source), preserve_index=False), path)
    for source, path in (("main", main_path), ("aux", aux_path)):
        (path.parent.parent / "run_manifest.json").write_text(
            json.dumps(_manifest(source, prepared)), encoding="utf-8"
        )

    statistics = tmp_path / "feature_statistics.safetensors"
    # A real (if minimal) safetensors file: content_sha256 parses it via
    # safetensors.torch.load_file, which rejects arbitrary bytes.
    save_file({"synthetic": t.zeros(1)}, statistics)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    meta = checkpoint / "meta_054600.json"
    model = checkpoint / "model_054600.pt"
    optimizer = checkpoint / "optim_054600_rank0.pt"
    meta.write_text(
        json.dumps(
            {
                "step": 54_600,
                "remote_run_fingerprint": FINGERPRINT,
                "model_config": {"sequence_len": 1024},
                "user_config": {
                    "model_tag": MODEL_TAG,
                    "conditioning": {"stats_source_sha256": content_sha256(statistics)},
                },
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"synthetic model")
    optimizer.write_bytes(b"synthetic optimizer")
    marker = build_checkpoint_marker(
        checkpoint, step=54_600, world_size=1, remote_run_fingerprint=FINGERPRINT
    )
    (checkpoint / "remote_complete_054600.json").write_text(json.dumps(marker), encoding="utf-8")
    (project_path.parent.parent / "run_manifest.json").write_text(
        json.dumps(_project_manifest(main=main_path, meta=meta, statistics=statistics)),
        encoding="utf-8",
    )

    evaluated = tmp_path / "evaluated" / "steering.jsonl"
    evaluated.parent.mkdir()
    _write_entries(evaluated, _entries())
    paths = SourcePaths(prepared, main_path, aux_path, evaluated, project_path, meta, statistics)
    expectations = SourceExpectations(
        2,
        COMMIT,
        REVISION,
        MODEL,
        42,
        1.0,
        128,
        "c" * 64,
        "d" * 40,
        PROJECT_COMMIT,
        "f" * 64,
        "qwen2_5_3b",
        28,
        True,
    )
    hashes = {key: _sha(path) for key, path in paths.hashed_files().items()}
    return paths, expectations, hashes


@pytest.fixture(autouse=True)
def _fake_self_conditioning_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    setup = SimpleNamespace(
        marker="validated setup",
        n_sae_features=ELIGIBLE_HEAD_ROWS,
        allowed_ids=None,
        feature_mask=None,
        output_selection=SamplingSelection(mean=POSITIVE_MEAN, std=POSITIVE_MEAN),
        output_negative_selection=NegativeSamplingSelection(
            mean=NEGATIVE_MEAN,
            std=NEGATIVE_MEAN,
        ),
    )
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_args, **_kwargs: setup,
    )
    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_args, **_kwargs: {
            concept_id: ConceptTrainingTexts(
                concept_id,
                f"concept-{concept_id}",
                "genre",
                (f"positive-{concept_id}",),
                (f"negative-{concept_id}",),
            )
            for concept_id in range(2)
        },
    )


def _validate(
    paths: SourcePaths,
    expectations: SourceExpectations,
    hashes: dict[str, str],
    output_dir: Path,
):
    return validate_source_contract(
        paths=paths,
        expectations=expectations,
        expected_hashes=hashes,
        output_dir=output_dir,
    )


def test_full_source_contract_accepts_explicitly_allowed_relocated_prepared_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    manifests = [
        path.parent.parent / "run_manifest.json"
        for path in (paths.main_parquet, paths.aux_parquet)
    ]
    original_bytes = [path.read_bytes() for path in manifests]
    relocated = tmp_path / "another-checkout" / "prepared"
    relocated.parent.mkdir()
    paths.prepared_dir.rename(relocated)
    paths = replace(paths, prepared_dir=relocated)
    output = tmp_path / "adv-output"
    monkeypatch.setenv("AXBENCH_ADV_ALLOW_RELOCATED_PREPARED", "1")

    validated = _validate(paths, expectations, hashes, output)
    contract = write_source_contract(validated, output / "source_contract.json")

    assert [path.read_bytes() for path in manifests] == original_bytes
    assert dict(validated.hashes) == hashes
    assert json.loads(contract.read_text())["paths"]["prepared_dir"] == str(relocated.resolve())
    revalidate_source_hashes(validated)


def test_relocated_prepared_data_is_rejected_without_opt_in(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    relocated = tmp_path / "another-checkout" / "prepared"
    relocated.parent.mkdir()
    paths.prepared_dir.rename(relocated)

    with pytest.raises(AxBenchAdvError, match="prepared_dir differs"):
        _validate(
            replace(paths, prepared_dir=relocated),
            expectations,
            hashes,
            tmp_path / "adv-output",
        )


@pytest.mark.parametrize("mutation", ["main", "aux", "missing_prepared_file"])
def test_relocated_prepared_data_still_requires_matching_complete_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    relocated = tmp_path / "restored-prepared"
    paths.prepared_dir.rename(relocated)
    paths = replace(paths, prepared_dir=relocated)
    monkeypatch.setenv("AXBENCH_ADV_ALLOW_RELOCATED_PREPARED", "1")
    if mutation == "missing_prepared_file":
        (relocated / "data/alpaca_eval.json").unlink()
        match = "prepared AxBench directory failed validation"
    else:
        manifest_path = tmp_path / mutation / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["prepared_manifest"]["seed_rows"] = {"different": 1}
        manifest_path.write_text(json.dumps(manifest))
        match = f"{mutation} manifest does not embed the current prepared manifest"
    with pytest.raises(AxBenchAdvError, match=match):
        _validate(paths, expectations, hashes, tmp_path / "adv-output")


def test_full_source_contract_real_shapes_and_publication(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    output_dir = tmp_path / "adv-output"
    validated = _validate(paths, expectations, hashes, output_dir)
    assert validated.main_frame.shape == (240, 18)
    assert validated.aux_frame.shape == (20, 18)
    assert validated.project_frame.shape == (240, 18)
    assert len(validated.steering_entries) == 2
    assert validated.extraction_batch_size == 8
    assert set(validated.runtime_hashes) == {
        "target_meta",
        "target_model",
        "feature_statistics",
    }
    contract_path = write_source_contract(validated, output_dir / "contract.json")
    document = json.loads(contract_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["hashes"] == hashes
    assert document["manifest_identity"]["main"]["repository_git_commit"] == COMMIT
    assert (
        document["project_identity"]["self_conditioning_masked_policy"]
        == ACCEPTED_SELF_CONDITIONING_MASKED_POLICY
    )
    revalidate_source_hashes(validated)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "project_repository_git_commit",
            "not-a-commit",
            "project_repository_git_commit",
        ),
    ],
)
def test_project_source_expectation_identities_are_validated(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    with pytest.raises(AxBenchAdvError, match=match):
        _validate(
            paths,
            replace(expectations, **{field: value}),
            hashes,
            tmp_path / "adv-output",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "steering frame failed"),
        ("hash", "hash mismatch"),
        ("project_hash", "hash mismatch"),
        ("project_manifest_hash", "hash mismatch"),
        ("missing_cell", "steering frame failed"),
        ("project_missing_cell", "project frame failed"),
        ("project_duplicate_cell", "project frame failed"),
        ("factor_coverage", "factor grid differs"),
        ("method_coverage", "method coverage differs"),
        ("manifest_identity", "target_model"),
        ("runtime_config", "train 'layer'"),
        ("prepared_manifest", "missing required keys"),
        ("project_inputs", "do not exactly equal"),
        ("project_strength", "strength values differ"),
        ("project_method", "selected project methods"),
        ("project_factor_grid", "factor grid differs"),
        ("project_policy", "policy differs"),
        ("target_step", "target step or model tag differs"),
        ("target_model_tag", "target step or model tag differs"),
        ("receipt", "receipt failed validation"),
        ("receipt_step", "step differs from metadata filename"),
        ("metadata_digest", "identity differs from receipt"),
        ("model_digest", "identity differs from receipt"),
        ("statistics_manifest", "feature statistics differ from the project manifest"),
        ("path_overlap", "disjoint"),
    ],
)
def test_representative_source_contract_failures(tmp_path: Path, mutation: str, match: str) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    output = tmp_path / "adv-output"
    if mutation in {"schema", "missing_cell"}:
        frame = pd.read_parquet(paths.main_parquet)
        frame = frame.drop(columns="DiffMean_strength") if mutation == "schema" else frame.iloc[:-1]
        frame.to_parquet(paths.main_parquet, index=False)
        hashes["main_parquet"] = _sha(paths.main_parquet)
    elif mutation == "hash":
        hashes["main_parquet"] = "0" * 64
    elif mutation == "project_hash":
        hashes["project_parquet"] = "0" * 64
    elif mutation == "project_manifest_hash":
        hashes["project_manifest"] = "0" * 64
    elif mutation in {"project_missing_cell", "project_duplicate_cell"}:
        frame = pd.read_parquet(paths.project_parquet)
        frame = (
            frame.iloc[:-1]
            if mutation == "project_missing_cell"
            else pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
        )
        frame.to_parquet(paths.project_parquet, index=False)
        hashes["project_parquet"] = _sha(paths.project_parquet)
    elif mutation in {"factor_coverage", "method_coverage"}:
        entries = _entries()
        methods = entries[0]["results"]["LMJudgeEvaluator"]  # ty: ignore[not-subscriptable]
        if mutation == "factor_coverage":
            methods["DiffMean"]["factor"][0] = 99.0  # type: ignore[index]
        else:
            methods.pop("LoRA")  # type: ignore[union-attr]
        _write_entries(paths.evaluated_jsonl, entries)
        hashes["evaluated_jsonl"] = _sha(paths.evaluated_jsonl)
    elif mutation in {"manifest_identity", "runtime_config"}:
        path = paths.main_parquet.parent.parent / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "manifest_identity":
            manifest["target_model"] = "wrong"
        else:
            manifest["config"]["train"]["layer"] = 7
        path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "prepared_manifest":
        manifest_path = paths.prepared_dir / "prepare_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("artifacts")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "project_inputs":
        frame = pd.read_parquet(paths.project_parquet)
        frame.loc[frame["concept_id"].eq(0) & frame["input_id"].eq(0), "original_prompt"] = (
            "different"
        )
        frame.to_parquet(paths.project_parquet, index=False)
        hashes["project_parquet"] = _sha(paths.project_parquet)
    elif mutation == "project_strength":
        frame = pd.read_parquet(paths.project_parquet)
        frame.loc[0, "SelfConditioningMasked_strength"] = 99.0
        frame.to_parquet(paths.project_parquet, index=False)
        hashes["project_parquet"] = _sha(paths.project_parquet)
    elif mutation in {
        "project_method",
        "project_factor_grid",
        "project_policy",
        "target_step",
        "target_model_tag",
    }:
        manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
        if mutation == "project_method":
            manifest["methods"] = {"Wrong": {}}
        elif mutation == "project_factor_grid":
            manifest["methods"]["SelfConditioning"]["factors"] = [1.0]
        elif mutation == "project_policy":
            manifest["methods"]["SelfConditioning"]["policy"] = "wrong"
        elif mutation == "target_step":
            manifest["target"]["step"] = 1
        else:
            manifest["target"]["model_tag"] = "wrong"
        paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        hashes["project_manifest"] = _sha(paths.project_manifest)
    elif mutation == "receipt":
        paths.target_receipt.write_text("{}", encoding="utf-8")
        hashes["target_receipt"] = _sha(paths.target_receipt)
    elif mutation == "receipt_step":
        wrong = tmp_path / "wrong-step"
        wrong.mkdir()
        for name in ("meta_000001.json", "model_000001.pt", "optim_000001_rank0.pt"):
            (wrong / name).write_bytes(name.encode("ascii"))
        marker = build_checkpoint_marker(
            wrong,
            step=1,
            world_size=1,
            remote_run_fingerprint=FINGERPRINT,
        )
        paths.target_receipt.write_text(json.dumps(marker), encoding="utf-8")
        hashes["target_receipt"] = _sha(paths.target_receipt)
    elif mutation == "metadata_digest":
        paths.target_meta.write_text(
            paths.target_meta.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif mutation == "model_digest":
        paths.target_model.write_bytes(b"changed model")
    elif mutation == "statistics_manifest":
        manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
        manifest["inputs"]["feature_statistics_sha256"] = "0" * 64
        paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        hashes["project_manifest"] = _sha(paths.project_manifest)
    else:
        output = paths.prepared_dir / "output"
    with pytest.raises(AxBenchAdvError, match=match):
        _validate(paths, expectations, hashes, output)


def test_metadata_statistics_mismatch_warns_but_does_not_fail_when_well_formed(
    tmp_path: Path,
) -> None:
    """A well-formed but non-matching conditioning.stats_source_sha256 warns,
    not raises -- this checkpoint predates content_sha256, so its recorded
    hash can never match the robust content hash of an unchanged file. The
    project manifest's own recorded hash (checked separately, and matching
    here) is what actually fences this identity.
    """
    paths, expectations, hashes = _fixture(tmp_path)
    meta = json.loads(paths.target_meta.read_text(encoding="utf-8"))
    meta["user_config"]["conditioning"]["stats_source_sha256"] = "0" * 64
    paths.target_meta.write_text(json.dumps(meta), encoding="utf-8")
    marker = build_checkpoint_marker(
        paths.target_meta.parent,
        step=54_600,
        world_size=1,
        remote_run_fingerprint=FINGERPRINT,
    )
    paths.target_receipt.write_text(json.dumps(marker), encoding="utf-8")
    hashes["target_receipt"] = _sha(paths.target_receipt)
    output = tmp_path / "adv-output"
    _validate(paths, expectations, hashes, output)


def test_restored_project_and_checkpoint_paths_are_portable(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    restored = tmp_path / "restored" / "project" / "inference" / paths.project_parquet.name
    restored.parent.mkdir(parents=True)
    restored.write_bytes(paths.project_parquet.read_bytes())
    (restored.parent.parent / "run_manifest.json").write_bytes(paths.project_manifest.read_bytes())
    checkpoint = tmp_path / "restored" / "checkpoint"
    checkpoint.mkdir()
    for path in (paths.target_meta, paths.target_model, paths.target_receipt):
        (checkpoint / path.name).write_bytes(path.read_bytes())
    moved = SourcePaths(
        paths.prepared_dir,
        paths.main_parquet,
        paths.aux_parquet,
        paths.evaluated_jsonl,
        restored,
        checkpoint / paths.target_meta.name,
        paths.feature_statistics,
    )
    _validate(moved, expectations, hashes, tmp_path / "portable-output")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (None, None, None),
        ("aggregate_sha256", "0" * 64, "aggregate mismatch"),
        ("schema_version", True, "schema or kind"),
        ("schema_version", 1.0, "schema or kind"),
        ("step", 54_600.0, "step changed"),
    ],
)
def test_explicit_evaluation_target_receipt_without_training_files(
    tmp_path: Path, field: str | None, value: Any, reason: str | None
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    receipt = json.loads(paths.target_receipt.read_text(encoding="utf-8"))
    del receipt["ddp_world_size"], receipt["aggregate_sha256"]
    receipt["kind"] = "axbench_adv_evaluation_target"
    entries = {entry["name"]: entry for entry in receipt["files"]}
    receipt["files"] = [entries[path.name] for path in (paths.target_meta, paths.target_model)]
    receipt["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    paths.target_receipt.unlink()
    paths.target_meta.with_name("optim_054600_rank0.pt").unlink()
    paths = replace(paths, target_receipt_path=tmp_path / "evaluation_target.json")
    if field is not None:
        receipt[field] = value
    paths.target_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    hashes["target_receipt"] = _sha(paths.target_receipt)
    output_dir = tmp_path / "evaluation-target-output"

    if reason is not None:
        with pytest.raises(
            AxBenchAdvError, match="target checkpoint receipt failed validation"
        ) as exc_info:
            _validate(paths, expectations, hashes, output_dir)
        assert reason in str(exc_info.value.__cause__)
    else:
        validated = _validate(paths, expectations, hashes, output_dir)
        contract_path = write_source_contract(validated, output_dir / "contract.json")
        document = json.loads(contract_path.read_text(encoding="utf-8"))
        assert document["paths"]["target_receipt"] == str(tmp_path / "evaluation_target.json")
        assert validated.runtime_hashes["target_model"] == _sha(paths.target_model)


def test_prepared_symlink_target_is_part_of_disjointness_fence(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    prepared_file = paths.prepared_dir / "data" / "alpaca_eval.json"
    prepared_file.unlink()
    external = tmp_path / "external-source"
    external.mkdir()
    target = external / "alpaca_eval.json"
    target.write_text("{}", encoding="utf-8")
    prepared_file.symlink_to(target)
    with pytest.raises(AxBenchAdvError, match="disjoint"):
        _validate(paths, expectations, hashes, external)


def test_final_hash_fence_detects_pinned_and_transitive_mutations(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    output_dir = tmp_path / "adv-output"
    validated = _validate(paths, expectations, hashes, output_dir)
    original = paths.evaluated_jsonl.read_bytes()
    paths.evaluated_jsonl.write_bytes(original + b"\n")
    with pytest.raises(AxBenchAdvError, match="hash mismatch"):
        revalidate_source_hashes(validated)
    paths.evaluated_jsonl.write_bytes(original)
    paths.target_model.write_bytes(b"changed")
    with pytest.raises(AxBenchAdvError, match="identity differs from receipt"):
        write_source_contract(validated, output_dir / "must-not-publish.json")


def test_publication_path_and_hash_mapping_cannot_be_redirected(tmp_path: Path) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    output_dir = tmp_path / "adv-output"
    validated = _validate(paths, expectations, hashes, output_dir)
    with pytest.raises(TypeError, match="does not support item assignment"):
        validated.hashes["evaluated_jsonl"] = "0" * 64  # type: ignore[index]
    with pytest.raises(AxBenchAdvError, match="inside validated output"):
        write_source_contract(validated, tmp_path / "outside.json")

    output_dir.mkdir()
    redirected = output_dir / "redirected.json"
    redirected.symlink_to(paths.evaluated_jsonl)
    original = paths.evaluated_jsonl.read_bytes()
    with pytest.raises(AxBenchAdvError, match="inside validated output"):
        write_source_contract(validated, redirected)
    assert paths.evaluated_jsonl.read_bytes() == original


def test_validation_does_not_consume_retained_method_setup_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    rng = t.Generator(device="cpu")
    rng.manual_seed(42)
    initial_state = rng.get_state().clone()
    setup = SimpleNamespace(
        prompt_pipeline=SimpleNamespace(selection_config=SimpleNamespace(rng=rng)),
        n_sae_features=ELIGIBLE_HEAD_ROWS,
        allowed_ids=None,
        feature_mask=None,
        output_selection=SamplingSelection(mean=POSITIVE_MEAN, std=POSITIVE_MEAN),
        output_negative_selection=NegativeSamplingSelection(
            mean=NEGATIVE_MEAN,
            std=NEGATIVE_MEAN,
        ),
    )
    monkeypatch.setattr(
        source_contract_module, "load_method_setup", lambda *_args, **_kwargs: setup
    )

    validated = _validate(paths, expectations, hashes, tmp_path / "adv-output")

    assert validated.method_setup is setup
    assert t.equal(rng.get_state(), initial_state)


@pytest.mark.parametrize(
    ("segment_mode", "require_exact"),
    [
        (SegmentMode.fixed_tokens, False),
        (SegmentMode.turn_delimited_fixed_tokens, True),
    ],
)
def test_statistics_identity_mode_follows_checkpoint_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segment_mode: SegmentMode,
    require_exact: bool,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        meta = json.loads(paths.target_meta.read_text(encoding="utf-8"))
        meta["user_config"]["conditioning"]["stats_source_sha256"] = _sha(
            paths.feature_statistics
        )
        paths.target_meta.write_text(json.dumps(meta), encoding="utf-8")
        _refresh_checkpoint_receipt(paths)
        hashes["target_receipt"] = _sha(paths.target_receipt)
    observed: dict[str, object] = {}
    setup = SimpleNamespace(
        marker="validated setup",
        n_sae_features=ELIGIBLE_HEAD_ROWS,
        allowed_ids=None,
        feature_mask=None,
        output_selection=SamplingSelection(mean=POSITIVE_MEAN, std=POSITIVE_MEAN),
        output_negative_selection=NegativeSamplingSelection(
            mean=NEGATIVE_MEAN,
            std=NEGATIVE_MEAN,
        ),
    )

    def load_setup(*args: object, **kwargs: object) -> object:
        observed.update(args=args, kwargs=kwargs)
        return setup

    monkeypatch.setattr(
        source_contract_module,
        "resolve_checkpoint_sft_segment_mode",
        lambda *_args, **_kwargs: segment_mode,
    )
    monkeypatch.setattr(source_contract_module, "load_method_setup", load_setup)
    validated = _validate(paths, expectations, hashes, tmp_path / "adv-output")

    assert validated.method_setup is setup
    assert observed == {
        "args": (paths.target_meta, paths.feature_statistics),
        "kwargs": {"require_exact_statistics_identity": require_exact},
    }


@pytest.mark.parametrize("mismatched_statistics", [False, True])
def test_turn_statistics_identity_through_real_method_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatched_statistics: bool,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    original_payload = paths.feature_statistics.read_bytes()
    block = _real_lineage_block(tmp_path)
    block["statistics_sha256"] = _sha(paths.feature_statistics)
    snapshot_loader = methods._load_turn_statistics_snapshot
    fixture = _install_turn_setup_fixture(
        monkeypatch,
        tmp_path,
        real_lineage_block=block,
    )
    meta = json.loads(fixture.meta_path.read_text(encoding="utf-8"))
    # Exercise detection from lineage alone, without a top-level mode witness.
    del meta["sft_segment_mode"]
    del meta["selfcond_data"]
    meta.update(step=54_600, remote_run_fingerprint=FINGERPRINT)
    meta["user_config"]["model_tag"] = MODEL_TAG
    meta["user_config"]["conditioning"]["stats_source_sha256"] = block[
        "statistics_sha256"
    ]
    assert meta[ONLINE_SAE_LINEAGE_META_KEY] == block
    paths.target_meta.write_text(json.dumps(meta), encoding="utf-8")
    _refresh_checkpoint_receipt(paths)
    hashes["target_receipt"] = _sha(paths.target_receipt)

    if mismatched_statistics:
        save_file({"synthetic": t.ones(1)}, paths.feature_statistics)
        # The replacement matches project provenance, but not training lineage.
        manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
        manifest["inputs"]["feature_statistics_sha256"] = content_sha256(
            paths.feature_statistics
        )
        paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        hashes["project_manifest"] = _sha(paths.project_manifest)

    parsed_payloads: list[bytes] = []
    selected_modes: list[bool] = []

    def parse_statistics(payload: bytes) -> tuple[Any, None, None]:
        parsed_payloads.append(payload)
        return fixture.statistics, None, None

    def load_setup(*args: Any, **kwargs: Any) -> methods.MethodSetup:
        selected_modes.append(kwargs["require_exact_statistics_identity"])
        return methods.load_method_setup(*args, **kwargs)

    # Keep real mode/lineage parsing, setup validation, and raw-byte verification.
    # Reuse the methods suite's CPU collaborators and synthetic decoded stats.
    monkeypatch.setattr(methods, "_load_turn_statistics_snapshot", snapshot_loader)
    monkeypatch.setattr(
        turn_context_contract,
        "load_statistics_and_checksum_from_bytes",
        parse_statistics,
    )
    monkeypatch.setattr(source_contract_module, "load_method_setup", load_setup)
    monkeypatch.setattr(
        source_contract_module,
        "_validate_rank_contract_against_setup",
        lambda *_args: None,
    )

    if mismatched_statistics:
        with pytest.raises(AxBenchAdvError, match="method setup failed validation") as raised:
            _validate(paths, expectations, hashes, tmp_path / "adv-output")
        assert isinstance(raised.value.__cause__, ValueError)
        assert "raw SHA-256 does not match" in str(raised.value.__cause__)
        assert parsed_payloads == []
    else:
        validated = _validate(paths, expectations, hashes, tmp_path / "adv-output")
        assert validated.method_setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens
        assert validated.method_setup.statistics is fixture.statistics
        assert parsed_payloads == [original_payload]
    assert selected_modes == [True]


@pytest.mark.parametrize("metadata_update", [
    {"sft_segment_mode": "unknown-mode"},
    {ONLINE_SAE_LINEAGE_META_KEY: None},
    {ONLINE_SAE_LINEAGE_META_KEY: {"schema_version": 99}},
    {"sft_segment_mode": "fixed_tokens", ONLINE_SAE_LINEAGE_META_KEY: {
        "schema_version": 1, "source_segment_mode": "turn_delimited_fixed_tokens",
    }},
])
def test_source_contract_rejects_corrupt_or_conflicting_segment_witnesses(
    tmp_path: Path, metadata_update: dict[str, Any],
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    meta = json.loads(paths.target_meta.read_text())
    meta.update(metadata_update)
    paths.target_meta.write_text(json.dumps(meta))
    _refresh_checkpoint_receipt(paths)
    hashes["target_receipt"] = _sha(paths.target_receipt)
    with pytest.raises(AxBenchAdvError, match="target checkpoint segmentation failed validation"):
        _validate(paths, expectations, hashes, tmp_path / "adv-output")


def test_source_identity_and_json_helpers_fail_closed(tmp_path: Path) -> None:
    paths, _expectations, hashes = _fixture(tmp_path)
    assert not source_contract_module._is_sha256("g" * 64)
    assert not source_contract_module._is_sha256("A" * 64)
    assert not source_contract_module._is_git_commit("g" * 40)
    assert not source_contract_module._is_git_commit("A" * 40)

    bad_paths = replace(paths, target_meta=paths.target_meta.with_name("meta_latest.json"))
    with pytest.raises(AxBenchAdvError, match="meta_<six digits>"):
        _ = bad_paths.target_step
    with pytest.raises(AxBenchAdvError, match="source file not found"):
        source_contract_module._sha256_file_stable(tmp_path / "missing.bin")
    with pytest.raises(AxBenchAdvError, match="contain exactly"):
        source_contract_module._validate_expected_hashes({})
    bad_hashes = dict(hashes)
    bad_hashes["main_parquet"] = "not-a-digest"
    with pytest.raises(AxBenchAdvError, match="not lowercase SHA-256"):
        source_contract_module._validate_expected_hashes(bad_hashes)

    missing = tmp_path / "missing.json"
    with pytest.raises(AxBenchAdvError, match="not found"):
        source_contract_module._load_json_mapping(missing, label="fixture")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(AxBenchAdvError, match="not valid JSON"):
        source_contract_module._load_json_mapping(malformed, label="fixture")
    nonmapping = tmp_path / "list.json"
    nonmapping.write_text("[]", encoding="utf-8")
    with pytest.raises(AxBenchAdvError, match="JSON object"):
        source_contract_module._load_json_mapping(nonmapping, label="fixture")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("commit", "repository commit"),
        ("worktree", "clean repository"),
        ("worktree_type", "attest a repository worktree state"),
        ("stages", "stage provenance"),
        ("stage_mapping", "must be a mapping"),
        ("stage_commit", "commit does not match"),
        ("stage_worktree", "clean worktree"),
        ("stage_worktree_type", "attest a repository worktree state"),
    ],
)
def test_clean_commit_contract_rejects_each_identity_class(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(
        (paths.main_parquet.parent.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    if mutation == "commit":
        manifest["repository_git_commit"] = "0" * 40
    elif mutation == "worktree":
        manifest["repository_git_worktree"]["dirty"] = True
    elif mutation == "worktree_type":
        manifest["repository_git_worktree"] = []
    elif mutation == "stage_worktree_type":
        manifest["stage_provenance"]["train"]["repository_git_worktree"] = None
    elif mutation == "stages":
        manifest["stage_provenance"].pop("latent")
    elif mutation == "stage_mapping":
        manifest["stage_provenance"]["train"] = None
    elif mutation == "stage_commit":
        manifest["stage_provenance"]["train"]["repository_git_commit"] = "0" * 40
    else:
        manifest["stage_provenance"]["train"]["repository_git_worktree"]["dirty"] = True
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._require_clean_commit(
            manifest,
            COMMIT,
            {"train", "latent", "steering"},
        )


@pytest.mark.parametrize("dirty_at", ["top", "stage"])
def test_clean_commit_contract_accepts_dirty_worktree_when_explicitly_relaxed(
    tmp_path: Path,
    dirty_at: str,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(
        (paths.main_parquet.parent.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    if dirty_at == "top":
        manifest["repository_git_worktree"]["dirty"] = True
    else:
        manifest["stage_provenance"]["train"]["repository_git_worktree"]["dirty"] = True
    # The commit pin and stage provenance are still enforced unchanged --
    # only the "no uncommitted changes were present" attestation is skipped.
    source_contract_module._require_clean_commit(
        manifest,
        COMMIT,
        {"train", "latent", "steering"},
        require_clean=False,
    )
    manifest["repository_git_commit"] = "0" * 40
    with pytest.raises(AxBenchAdvError, match="repository commit"):
        source_contract_module._require_clean_commit(
            manifest,
            COMMIT,
            {"train", "latent", "steering"},
            require_clean=False,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("scalar", "schema_version"),
        ("lock_invalid", "package lock is not"),
        ("lock_mismatch", "package lock does not match"),
        ("vendor_invalid", "vendored AxBench commit is invalid"),
        ("vendor_mismatch", "vendored AxBench commit does not match"),
        ("prepared_dir", "prepared_dir must be a nonblank"),
        ("prepared_manifest", "does not embed"),
        ("config", "config.train"),
        ("inference", "config.inference"),
        ("inference_field", "inference 'models'"),
        ("factors_type", "steering_factors must be a list"),
        ("factors_grid", "steering_factors differ"),
    ],
)
def test_qwen_manifest_semantic_identity_failures(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(
        (paths.main_parquet.parent.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    if mutation == "scalar":
        manifest["schema_version"] = 2
    elif mutation == "lock_invalid":
        manifest["package_lock_sha256"] = "x" * 64
    elif mutation == "lock_mismatch":
        manifest["package_lock_sha256"] = "0" * 64
    elif mutation == "vendor_invalid":
        manifest["vendored_axbench_upstream_commit"] = "x" * 40
    elif mutation == "vendor_mismatch":
        manifest["vendored_axbench_upstream_commit"] = "0" * 40
    elif mutation == "prepared_dir":
        manifest["prepared_dir"] = ""
    elif mutation == "prepared_manifest":
        manifest["prepared_manifest"] = {}
    elif mutation == "config":
        manifest["config"] = None
    elif mutation == "inference":
        manifest["config"]["inference"] = None
    elif mutation == "inference_field":
        manifest["config"]["inference"]["models"] = []
    elif mutation == "factors_type":
        manifest["config"]["inference"]["steering_factors"] = None
    else:
        manifest["config"]["inference"]["steering_factors"] = [1.0]
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_run_manifest(
            manifest,
            source="main",
            prepared_dir=paths.prepared_dir,
            prepared_manifest=_prepared_manifest(),
            expectations=expectations,
        )


@pytest.mark.parametrize("source", ["main", "aux"])
@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("train", "model_revision"),
        ("inference", "model_revision"),
        ("inference", "steering_model_revision"),
    ],
)
def test_run_manifest_requires_matching_baseline_revisions(
    tmp_path: Path, source: str, section: str, field: str
) -> None:
    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = _manifest(source, paths.prepared_dir)
    config = cast(dict[str, Any], manifest["config"])
    config["train"]["model_revision"] = "it-soup-APO"
    config["inference"]["model_revision"] = "it-soup-APO"
    config["inference"]["steering_model_revision"] = "it-soup-APO"
    expectations = replace(expectations, model_revision="it-soup-APO")
    source_contract_module._validate_run_manifest(
        manifest,
        source=source,
        prepared_dir=paths.prepared_dir,
        prepared_manifest=_prepared_manifest(),
        expectations=expectations,
    )
    for wrong_revision in (None, "main", "", 123):
        config[section][field] = wrong_revision
        with pytest.raises(AxBenchAdvError, match=field):
            source_contract_module._validate_run_manifest(
                manifest,
                source=source,
                prepared_dir=paths.prepared_dir,
                prepared_manifest=_prepared_manifest(),
                expectations=expectations,
            )


def test_run_manifest_allows_relocated_prepared_data_with_identical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = _manifest("main", Path("/original/producer/prepared"))
    monkeypatch.setenv("AXBENCH_ADV_ALLOW_RELOCATED_PREPARED", "1")
    source_contract_module._validate_run_manifest(
        manifest,
        source="main",
        prepared_dir=paths.prepared_dir,
        prepared_manifest=_prepared_manifest(),
        expectations=expectations,
    )
    changed = {**_prepared_manifest(), "seed_rows": {"different": 1}}
    with pytest.raises(AxBenchAdvError, match="does not embed"):
        source_contract_module._validate_run_manifest(
            manifest,
            source="main",
            prepared_dir=paths.prepared_dir,
            prepared_manifest=changed,
            expectations=expectations,
        )


def test_run_manifest_config_name_and_layer_are_read_from_expectations_not_hardcoded(
    tmp_path: Path,
) -> None:
    """A non-Qwen target's own config_name/layer must validate cleanly, and
    the historical Qwen values must NOT still be accepted for it -- proving
    these two fields are genuinely sourced from ``expectations`` rather than
    hardcoded to the historical Qwen run."""

    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(
        (paths.main_parquet.parent.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest["config_name"] = "pythia_2_8b_main.yaml"
    manifest["config"]["train"]["layer"] = 24
    manifest["config"]["inference"]["steering_layer"] = 24
    manifest["config"]["inference"]["steering_layers"] = [24]

    pythia_expectations = replace(expectations, config_name_prefix="pythia_2_8b", layer=24)

    # The still-Qwen-shaped expectations must reject this non-Qwen manifest.
    with pytest.raises(AxBenchAdvError, match="config_name"):
        source_contract_module._validate_run_manifest(
            manifest,
            source="main",
            prepared_dir=paths.prepared_dir,
            prepared_manifest=_prepared_manifest(),
            expectations=expectations,
        )

    # The matching non-Qwen expectations must accept it.
    source_contract_module._validate_run_manifest(
        manifest,
        source="main",
        prepared_dir=paths.prepared_dir,
        prepared_manifest=_prepared_manifest(),
        expectations=pythia_expectations,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("concept_count", "must contain 2 concepts"),
        ("results", "only LMJudgeEvaluator"),
        ("block", "result fields differ"),
        ("factor_type", "nonnumeric factor"),
        ("factor_nonfinite", "nonfinite factor"),
        ("summary_type", "must be a list"),
        ("summary_length", "wrong length"),
        ("rating_type", "nonnumeric rating"),
        ("rating_range", r"outside \[0, 2\]"),
        ("raw_length", "wrong length"),
        ("raw_off_grid", "non-null off-grid"),
        ("completion_length", "wrong length"),
        ("completion_declared", "invalid completion"),
        ("completion_off_grid", "non-null off-grid"),
    ],
)
def test_evaluated_jsonl_shape_and_value_failures(
    mutation: str,
    match: str,
) -> None:
    entries: Any = copy.deepcopy(_entries())
    methods = entries[0]["results"]["LMJudgeEvaluator"]
    diffmean = methods["DiffMean"]
    lora = methods["LoRA"]
    if mutation == "concept_count":
        entries.pop()
    elif mutation == "results":
        entries[0]["results"]["extra"] = {}
    elif mutation == "block":
        diffmean.pop("lm_judge_rating")
    elif mutation == "factor_type":
        diffmean["factor"][0] = True
    elif mutation == "factor_nonfinite":
        diffmean["factor"][0] = math.nan
    elif mutation == "summary_type":
        diffmean["lm_judge_rating"] = "wrong"
    elif mutation == "summary_length":
        diffmean["lm_judge_rating"].pop()
    elif mutation == "rating_type":
        diffmean["lm_judge_rating"][0] = True
    elif mutation == "rating_range":
        diffmean["lm_judge_rating"][0] = 3.0
    elif mutation == "raw_length":
        diffmean["raw_aggregated_ratings"].pop()
    elif mutation == "raw_off_grid":
        lora["raw_aggregated_ratings"][0] = 1.0
    elif mutation == "completion_length":
        diffmean["relevance_concept_completions"].pop()
    elif mutation == "completion_declared":
        diffmean["relevance_concept_completions"][0] = None
    else:
        lora["relevance_concept_completions"][0] = "unexpected"
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_evaluated_jsonl(entries, max_concepts=2)


def test_aux_input_projection_mismatch_is_rejected(tmp_path: Path) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    main = pd.read_parquet(paths.main_parquet)
    aux = pd.read_parquet(paths.aux_parquet)
    aux.loc[0, "original_prompt"] = "different"
    with pytest.raises(AxBenchAdvError, match="aux source inputs"):
        source_contract_module._validate_cross_frame_inputs(main, aux)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("scalar", "rank-era schema-3 project artifact"),
        ("project_commit", "project manifest 'repository_git_commit' differs"),
        ("worktree", "clean source worktree"),
        ("worktree_type", "attest a source worktree state"),
        ("prepared", "prepared identity"),
        ("inputs", "no inputs mapping"),
        ("main_hash", "main Parquet hash"),
        ("statistics", "digest is invalid"),
        ("method_block", "method block is invalid"),
        ("ordinary_policy", "SelfConditioning policy differs"),
        ("masked_method_block", "SelfConditioningMasked method block is invalid"),
        ("masked_factor_grid", "SelfConditioningMasked factor grid differs"),
        ("masked_policy", "rank-era schema-3 project artifact"),
        ("masked_block_scale", "rank-era schema-3 project artifact"),
        ("rank_field", "rank contract field 'rounding' differs"),
        ("clamp_type", "rank contract field 'clamp' differs"),
        ("exact_zero_type", "rank contract field 'exact_zero_diffmean_eligible' differs"),
        ("method_order", "method order differs"),
        ("generation", "generation policy differs"),
        ("target", "no target mapping"),
        ("target_filename", "target metadata filename differs"),
        ("configuration", "no configuration mapping"),
        ("config_scalar", "configuration 'seed' differs"),
        ("main_filename", "main Parquet filename differs"),
        ("config_target_filename", "target metadata filename differs"),
        ("statistics_filename", "feature-statistics filename differs"),
        ("config_scale", "rank-era schema-3 project artifact"),
        ("batch_size", "extraction_batch_size must be positive"),
    ],
)
def test_project_manifest_semantic_identity_failures(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    if mutation == "scalar":
        manifest["schema_version"] = 1
    elif mutation == "project_commit":
        manifest["repository_git_commit"] = "0" * 40
    elif mutation == "worktree":
        manifest["repository_git_worktree"]["dirty"] = True
    elif mutation == "worktree_type":
        manifest["repository_git_worktree"] = []
    elif mutation == "prepared":
        manifest["prepared_manifest"] = {}
    elif mutation == "inputs":
        manifest["inputs"] = None
    elif mutation == "main_hash":
        manifest["inputs"]["main_steering_parquet_sha256"] = "0" * 64
    elif mutation == "statistics":
        manifest["inputs"]["feature_statistics_sha256"] = "not-a-digest"
    elif mutation == "method_block":
        manifest["methods"]["SelfConditioning"] = None
    elif mutation == "ordinary_policy":
        manifest["methods"]["SelfConditioning"]["policy"] = (
            "prompt_last_segment_max_pool_diffmean_lerp_v1"
        )
    elif mutation == "masked_method_block":
        manifest["methods"]["SelfConditioningMasked"] = None
    elif mutation == "masked_factor_grid":
        manifest["methods"]["SelfConditioningMasked"]["factors"] = [1.0]
    elif mutation == "masked_policy":
        manifest["methods"]["SelfConditioningMasked"]["policy"] = (
            "prompt_last_segment_max_pool_diffmean_sign_mask_v1"
        )
    elif mutation == "masked_block_scale":
        manifest["methods"]["SelfConditioningMasked"]["masked_threshold_scale"] = True
    elif mutation == "rank_field":
        manifest["methods"]["SelfConditioningMasked"]["rounding"] = "floor"
    elif mutation == "clamp_type":
        manifest["methods"]["SelfConditioningMasked"]["clamp"]["minimum"] = True
    elif mutation == "exact_zero_type":
        manifest["methods"]["SelfConditioningMasked"][
            "exact_zero_diffmean_eligible"
        ] = 0
    elif mutation == "method_order":
        manifest["method_order"] = []
    elif mutation == "generation":
        manifest["generation"] = None
    elif mutation == "target":
        manifest["target"] = None
    elif mutation == "target_filename":
        manifest["target"]["checkpoint_meta"] = "meta_000001.json"
    elif mutation == "configuration":
        manifest["configuration"] = None
    elif mutation == "config_scalar":
        manifest["configuration"]["seed"] = 43
    elif mutation == "main_filename":
        manifest["configuration"]["main_path"] = "different.parquet"
    elif mutation == "config_target_filename":
        manifest["configuration"]["target_meta_path"] = "meta_000001.json"
    elif mutation == "statistics_filename":
        manifest["configuration"]["feature_statistics_path"] = "different.safetensors"
    elif mutation == "config_scale":
        manifest["configuration"]["masked_threshold_scale"] = True
    else:
        manifest["configuration"]["extraction_batch_size"] = 0
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_project_manifest(
            manifest,
            paths=paths,
            prepared_manifest=_prepared_manifest(),
            expectations=expectations,
            main_sha256=_sha(paths.main_parquet),
        )


def test_project_manifest_accepts_dirty_worktree_when_explicitly_relaxed(tmp_path: Path) -> None:
    paths, expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    manifest["repository_git_worktree"]["dirty"] = True
    relaxed = replace(expectations, require_clean_worktree=False)
    # Should not raise: every other field in the fixture manifest is still
    # exactly correct, so only the dirty-worktree attestation was skipped.
    source_contract_module._validate_project_manifest(
        manifest,
        paths=paths,
        prepared_manifest=_prepared_manifest(),
        expectations=relaxed,
        main_sha256=_sha(paths.main_parquet),
    )
    with pytest.raises(AxBenchAdvError, match="clean source worktree"):
        source_contract_module._validate_project_manifest(
            manifest,
            paths=paths,
            prepared_manifest=_prepared_manifest(),
            expectations=expectations,
            main_sha256=_sha(paths.main_parquet),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("eligible_head_rows", "eligible_head_rows differs"),
        ("positive_mean", "positive_mean differs"),
        ("negative_mean", "negative_mean differs"),
        ("pool_limits", "pool-limit table differs"),
        ("pool_limit_type", "pool-limit table differs"),
        ("pool_limit_missing_key", "pool-limit table differs"),
        ("pool_limits_truncated", "pool-limit table differs"),
        ("pool_factor", "pool-limit table differs"),
        ("negative_pool_limit", "pool-limit table differs"),
        ("checkpoint_mean", "positive_mean differs"),
    ],
)
def test_rank_contract_is_recomputed_from_checkpoint_setup(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    masked = manifest["methods"]["SelfConditioningMasked"]
    setup = SimpleNamespace(
        n_sae_features=ELIGIBLE_HEAD_ROWS,
        allowed_ids=None,
        feature_mask=None,
        output_selection=SamplingSelection(mean=POSITIVE_MEAN, std=POSITIVE_MEAN),
        output_negative_selection=NegativeSamplingSelection(
            mean=NEGATIVE_MEAN,
            std=NEGATIVE_MEAN,
        ),
    )
    if mutation == "checkpoint_mean":
        setup.output_selection = SamplingSelection(mean=1.0, std=1.0)
    elif mutation == "pool_limits":
        masked["pool_limits"][1]["positive"] += 1
    elif mutation == "pool_limit_type":
        masked["pool_limits"][0]["positive"] = True
    elif mutation == "pool_limit_missing_key":
        masked["pool_limits"][0].pop("negative")
    elif mutation == "pool_limits_truncated":
        masked["pool_limits"].pop()
    elif mutation == "pool_factor":
        masked["pool_limits"][0]["factor"] = 0.1
    elif mutation == "negative_pool_limit":
        masked["pool_limits"][1]["negative"] += 1
    else:
        masked[mutation] += 1

    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_rank_contract_against_setup(manifest, cast(Any, setup))


def test_rank_contract_uses_compact_post_mask_n_and_independent_side_means(
    tmp_path: Path,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    masked = manifest["methods"]["SelfConditioningMasked"]
    positive_mean = 0.5
    negative_mean = 1.0
    eligible_head_rows = 3
    masked["eligible_head_rows"] = eligible_head_rows
    masked["positive_mean"] = positive_mean
    masked["negative_mean"] = negative_mean
    masked["pool_limits"] = [
        {
            "factor": factor,
            "positive": min(
                eligible_head_rows,
                max(
                    1,
                    round(
                        eligible_head_rows * (1 - factor / 6)
                        + 2 * positive_mean * factor / 6
                    ),
                ),
            ),
            "negative": min(
                eligible_head_rows,
                max(
                    1,
                    round(
                        eligible_head_rows * (1 - factor / 6)
                        + 2 * negative_mean * factor / 6
                    ),
                ),
            ),
        }
        for factor in SELF_CONDITIONING_FACTORS
    ]
    setup = SimpleNamespace(
        n_sae_features=10,
        allowed_ids=t.tensor([1, 3, 5, 7]),
        feature_mask=FeatureMask("blacklist", (1,)),
        output_selection=SamplingSelection(mean=positive_mean, std=positive_mean),
        output_negative_selection=NegativeSamplingSelection(
            mean=negative_mean,
            std=negative_mean,
        ),
    )

    source_contract_module._validate_rank_contract_against_setup(manifest, cast(Any, setup))

    masked["eligible_head_rows"] = 4
    with pytest.raises(AxBenchAdvError, match="eligible_head_rows differs"):
        source_contract_module._validate_rank_contract_against_setup(manifest, cast(Any, setup))


def test_rank_contract_translates_invalid_checkpoint_count_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    setup = SimpleNamespace(
        n_sae_features=ELIGIBLE_HEAD_ROWS,
        allowed_ids=None,
    )
    monkeypatch.setattr(
        source_contract_module,
        "build_self_conditioning_masked_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid mean")),
    )

    with pytest.raises(
        AxBenchAdvError,
        match="checkpoint count policies cannot construct",
    ):
        source_contract_module._validate_rank_contract_against_setup(manifest, cast(Any, setup))


def test_threshold_era_project_is_rejected_before_method_setup_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest["methods"]["SelfConditioningMasked"] = {
        "factors": list(ADV_FULL_FACTORS),
        "policy": "prompt_last_segment_max_pool_diffmean_sign_mask_v1",
        "masked_threshold_scale": 0.1,
    }
    paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    hashes["project_manifest"] = _sha(paths.project_manifest)
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_args: (_ for _ in ()).throw(AssertionError("method setup loaded")),
    )

    with pytest.raises(AxBenchAdvError, match="rank-era schema-3 project artifact"):
        _validate(paths, expectations, hashes, tmp_path / "adv-output")


def _refresh_checkpoint_receipt(paths: SourcePaths, *, fingerprint: str = FINGERPRINT) -> None:
    marker = build_checkpoint_marker(
        paths.target_meta.parent,
        step=paths.target_step,
        world_size=1,
        remote_run_fingerprint=fingerprint,
    )
    paths.target_receipt.write_text(json.dumps(marker), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("target_step", "target metadata step must be exactly"),
        ("receipt_entry", "receipt does not contain"),
        ("metadata_step", "metadata JSON step differs"),
        ("fingerprint", "remote fingerprint differs"),
        ("model_tag", "model tag differs"),
        ("conditioning", "statistics differ from target metadata"),
        ("manifest_inputs", "statistics differ from the project manifest"),
    ],
)
def test_checkpoint_transitive_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    if mutation == "target_step":
        paths = replace(paths, target_meta=paths.target_meta.with_name("meta_000001.json"))
    elif mutation == "receipt_entry":
        monkeypatch.setattr(
            source_contract_module,
            "parse_checkpoint_marker",
            lambda *_args, **_kwargs: {
                "step": 54_600,
                "remote_run_fingerprint": FINGERPRINT,
                "files": [],
            },
        )
    elif mutation == "manifest_inputs":
        manifest["inputs"] = None
    else:
        meta = json.loads(paths.target_meta.read_text(encoding="utf-8"))
        if mutation == "metadata_step":
            meta["step"] = 1
        elif mutation == "fingerprint":
            meta["remote_run_fingerprint"] = "0" * 64
        elif mutation == "model_tag":
            meta["user_config"]["model_tag"] = "wrong"
        else:
            meta["user_config"]["conditioning"] = None
        paths.target_meta.write_text(json.dumps(meta), encoding="utf-8")
        _refresh_checkpoint_receipt(paths)
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_checkpoint_sources(
            paths=paths,
            project_manifest=manifest,
            expectations=_expectations,
        )


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda path: path.unlink(), "feature statistics file failed to load"),
        (lambda path: path.write_bytes(b"not a real safetensors file"), "failed to load"),
    ],
)
def test_missing_or_corrupt_feature_statistics_raises_axbench_adv_error(
    tmp_path: Path,
    corrupt,
    match: str,
) -> None:
    """content_sha256 can raise OSError/SafetensorError; both must surface as
    AxBenchAdvError like every other source-identity failure in this module,
    not as a raw, differently-typed exception a caller wouldn't catch."""
    paths, _expectations, _hashes = _fixture(tmp_path)
    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    corrupt(paths.feature_statistics)
    with pytest.raises(AxBenchAdvError, match=match):
        source_contract_module._validate_checkpoint_sources(
            paths=paths,
            project_manifest=manifest,
            expectations=_expectations,
        )


def test_training_selection_and_concept_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _expectations, _hashes = _fixture(tmp_path)
    frame = pd.read_parquet(paths.main_parquet)
    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("broken")),
    )
    with pytest.raises(AxBenchAdvError, match="training text selection failed"):
        source_contract_module._validate_training_texts(
            paths=paths,
            main_frame=frame,
            max_concepts=2,
        )

    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_args, **_kwargs: {
            0: ConceptTrainingTexts(0, "wrong", "genre", ("positive",), ("negative",))
        },
    )
    with pytest.raises(AxBenchAdvError, match="concept text differs"):
        source_contract_module._validate_training_texts(
            paths=paths,
            main_frame=frame,
            max_concepts=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repository_git_commit", "bad", "repository_git_commit"),
        ("dataset_revision", "bad", "dataset_revision"),
        ("vendored_axbench_upstream_commit", "bad", "vendored_axbench"),
    ],
)
def test_source_expectation_scalar_failures(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    with pytest.raises(AxBenchAdvError, match=match):
        _validate(
            paths,
            replace(expectations, **{field: value}),
            hashes,
            tmp_path / "adv-output",
        )


def test_validation_wraps_prepared_and_method_setup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    monkeypatch.setattr(
        source_contract_module,
        "validate_prepared_dir",
        lambda *_args: (_ for _ in ()).throw(ValueError("broken")),
    )
    with pytest.raises(AxBenchAdvError, match="prepared AxBench directory"):
        _validate(paths, expectations, hashes, tmp_path / "prepared-failure")

    monkeypatch.undo()
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    with pytest.raises(AxBenchAdvError, match="method setup failed"):
        _validate(paths, expectations, hashes, tmp_path / "setup-failure")


def test_validation_detects_in_run_source_and_runtime_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            source_contract_module,
            "assert_source_hashes",
            lambda *_args, **_kwargs: {**hashes, "main_parquet": "0" * 64},
        )
        with pytest.raises(AxBenchAdvError, match="source identities changed"):
            _validate(paths, expectations, hashes, tmp_path / "source-race")

    original = source_contract_module._validate_checkpoint_sources
    calls = 0

    def changing_runtime(**kwargs):
        nonlocal calls
        calls += 1
        runtime_hashes, runtime_lengths, meta = original(**kwargs)
        if calls == 2:
            runtime_hashes = {**runtime_hashes, "target_model": "0" * 64}
        return runtime_hashes, runtime_lengths, meta

    monkeypatch.setattr(source_contract_module, "_validate_checkpoint_sources", changing_runtime)
    with pytest.raises(AxBenchAdvError, match="runtime source identities changed"):
        _validate(paths, expectations, hashes, tmp_path / "runtime-race")


@pytest.mark.parametrize("mutation", ["hash", "length"])
def test_revalidation_compares_runtime_hashes_and_lengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    validated = _validate(paths, expectations, hashes, tmp_path / "adv-output")
    runtime_hashes = dict(validated.runtime_hashes)
    runtime_lengths = dict(validated.runtime_byte_lengths)
    if mutation == "hash":
        runtime_hashes["target_model"] = "0" * 64
    else:
        runtime_lengths["target_model"] += 1
    monkeypatch.setattr(
        source_contract_module,
        "_validate_checkpoint_sources",
        lambda **_kwargs: (runtime_hashes, runtime_lengths, {}),
    )
    match = "runtime source hash" if mutation == "hash" else "runtime source byte length"
    with pytest.raises(AxBenchAdvError, match=match):
        revalidate_source_hashes(validated)


def test_configured_project_checkpoint_pins_cover_validation_and_revalidation(
    tmp_path: Path,
) -> None:
    paths, expectations, hashes = _fixture(tmp_path)
    new_paths = replace(paths, target_meta=paths.target_meta.with_name("meta_000123.json"))
    paths.target_model.rename(new_paths.target_model)
    paths.target_meta.with_name("optim_054600_rank0.pt").rename(
        new_paths.target_meta.with_name("optim_000123_rank0.pt")
    )
    meta = json.loads(paths.target_meta.read_text())
    meta["step"] = 123
    meta["user_config"]["model_tag"] = "alternate-project"
    new_paths.target_meta.write_text(json.dumps(meta))
    manifest = json.loads(paths.project_manifest.read_text())
    manifest["target"].update(
        step=123, model_tag="alternate-project", checkpoint_meta=str(new_paths.target_meta)
    )
    manifest["configuration"]["target_meta_path"] = str(new_paths.target_meta)
    paths.project_manifest.write_text(json.dumps(manifest))
    _refresh_checkpoint_receipt(new_paths)
    hashes["project_manifest"] = _sha(new_paths.project_manifest)
    hashes["target_receipt"] = _sha(new_paths.target_receipt)
    configured = replace(
        expectations, project_target_step=123, project_target_model_tag="alternate-project"
    )
    validated = _validate(new_paths, configured, hashes, tmp_path / "alternate-output")
    revalidate_source_hashes(validated)
    with pytest.raises(AxBenchAdvError, match="target step or model tag differs"):
        _validate(new_paths, expectations, hashes, tmp_path / "wrong-pins-output")
    with pytest.raises(AxBenchAdvError, match="step must be exactly"):
        revalidate_source_hashes(replace(validated, expectations=expectations))
    with pytest.raises(AxBenchAdvError, match="model tag differs"):
        revalidate_source_hashes(
            replace(validated, expectations=replace(configured, project_target_model_tag="wrong"))
        )
