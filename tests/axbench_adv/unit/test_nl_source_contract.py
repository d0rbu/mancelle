"""Optional natural-language AxBench-Adv source-contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch as t

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.eval.axbench_adv import source_contract as source_contract_module
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS
from steering.eval.axbench_adv.source_contract import (
    SourceExpectations,
    SourcePaths,
    revalidate_source_hashes,
    validate_source_contract,
    write_source_contract,
)
from steering.eval.axbench_adv.types import NL_METHOD
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_FACTORS,
    MethodSetup,
    self_conditioning_masked_pool_limit,
)
from steering.eval.axbench_upstream.nl_target_scores import (
    NL_RETRIEVAL_ALGORITHM,
    NL_STEERING_POLICY,
    build_retrieval_target_scores,
    retrieval_feature_identity,
)
from tests.axbench_adv.unit import test_source_contract as base

_DESCRIPTION_TEXTS_SHA256 = "1" * 64
_CONCEPT_TEXTS_SHA256 = "2" * 64
_EMBEDDING_SPACE = {
    "model": "sentence-transformers/all-mpnet-base-v2",
    "revision": "3" * 40,
    "encoding": "mean_pool_normalized_float32",
    "library_versions": {
        "numpy": "2.3.3",
        "sentence-transformers": "5.1.0",
        "torch": "2.8.0",
    },
}
_MASK_IDS = (0, 1, 2, 3, 4, 5, 6)
_CLUSTERS = ((0, 1), (2, 3))


@dataclass(frozen=True)
class _NlFixture:
    paths: SourcePaths
    expectations: SourceExpectations
    hashes: dict[str, str]
    setup: MethodSetup
    description_embeddings: Path
    concept_embeddings: Path
    output_dir: Path


def _setup(
    tmp_path: Path,
    *,
    clusters: tuple[tuple[int, ...], ...] = _CLUSTERS,
) -> MethodSetup:
    mask = FeatureMask(FeatureMaskMode.whitelist, _MASK_IDS)
    definition = FeatureClusterDefinition(clusters=clusters)
    resolved = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=8,
        ),
    )
    contract = CheckpointFeatureContract(
        feature_mask=mask,
        resolved_clusters=resolved,
    )
    consolidation = contract.consolidation
    vocab = effective_feature_vocab(contract.universe, 8)
    assert consolidation is not None
    assert vocab is not None
    return cast(
        MethodSetup,
        SimpleNamespace(
            marker="validated NL setup",
            n_sae_features=8,
            allowed_ids=t.tensor(vocab.original_ids, dtype=t.int64),
            feature_mask=mask,
            feature_contract=contract,
            consolidation=consolidation,
            feature_vocab=vocab,
            output_selection=SamplingSelection(
                mean=base.POSITIVE_MEAN,
                std=base.POSITIVE_MEAN,
            ),
            output_negative_selection=NegativeSamplingSelection(
                mean=base.NEGATIVE_MEAN,
                std=base.NEGATIVE_MEAN,
            ),
        ),
    )


def _training_texts() -> dict[int, ConceptTrainingTexts]:
    return {
        concept_id: ConceptTrainingTexts(
            concept_id,
            f"concept-{concept_id}",
            "genre",
            (f"positive-{concept_id}",),
            (f"negative-{concept_id}",),
        )
        for concept_id in range(2)
    }


def _legacy_setup() -> MethodSetup:
    return cast(
        MethodSetup,
        SimpleNamespace(
            marker="validated legacy setup",
            n_sae_features=base.ELIGIBLE_HEAD_ROWS,
            allowed_ids=None,
            feature_mask=None,
            output_selection=SamplingSelection(
                mean=base.POSITIVE_MEAN,
                std=base.POSITIVE_MEAN,
            ),
            output_negative_selection=NegativeSamplingSelection(
                mean=base.NEGATIVE_MEAN,
                std=base.NEGATIVE_MEAN,
            ),
        ),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_contract(head_width: int) -> dict[str, object]:
    return {
        "eligible_head_rows": head_width,
        "pool_limits": [
            {
                "factor": factor,
                "positive": self_conditioning_masked_pool_limit(
                    factor,
                    n_candidates=head_width,
                    mean=base.POSITIVE_MEAN,
                ),
                "negative": self_conditioning_masked_pool_limit(
                    factor,
                    n_candidates=head_width,
                    mean=base.NEGATIVE_MEAN,
                ),
            }
            for factor in SELF_CONDITIONING_FACTORS
        ],
    }


def _nl_result_block() -> dict[str, object]:
    # NLSteering uses the full interpolation grid and the same judge payload
    # shape as SelfConditioning.
    return copy.deepcopy(base._result_block("SelfConditioning"))


def _embedding_record(path: Path, *, project_root: Path, source_sha256: str) -> dict[str, object]:
    return {
        "path": str(path.relative_to(project_root)),
        "sha256": _sha(path),
        "source_texts_sha256": source_sha256,
        "embedding_space": copy.deepcopy(_EMBEDDING_SPACE),
    }


def _nl_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _NlFixture:
    paths, expectations, hashes = base._fixture(tmp_path)
    setup = _setup(tmp_path)
    monkeypatch.setattr(source_contract_module, "load_method_setup", lambda *_a, **_k: setup)
    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_a, **_k: _training_texts(),
    )

    project_root = paths.project_manifest.parent
    embedding_dir = project_root / "nl_inputs"
    embedding_dir.mkdir()
    description_embeddings = embedding_dir / "descriptions.npz"
    concept_embeddings = embedding_dir / "concepts.npz"
    np.savez(
        description_embeddings,
        fids=np.array([7, 3, 0, 5, 2, 6, 1, 4], dtype=np.int64),
        emb=np.array(
            [
                [0.5, 0.5, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.savez(
        concept_embeddings,
        texts=np.array(["unused-extra", "concept-1", "concept-0"]),
        emb=np.array(
            [[0.1, 0.2, 0.3], [-0.25, 0.5, 1.0], [1.0, -0.5, 0.25]],
            dtype=np.float32,
        ),
    )

    project_frame = pd.read_parquet(paths.project_parquet)
    project_frame[f"{NL_METHOD}_steered_generation"] = "healthy NL generated response"
    project_frame[f"{NL_METHOD}_strength"] = project_frame["factor"].astype("float64")
    project_frame.to_parquet(paths.project_parquet, index=False)

    entries = base._entries()
    for entry in entries:
        methods = cast(dict[str, object], entry["results"])["LMJudgeEvaluator"]
        cast(dict[str, object], methods)[NL_METHOD] = _nl_result_block()
    base._write_entries(paths.evaluated_jsonl, entries)

    manifest = json.loads(paths.project_manifest.read_text(encoding="utf-8"))
    assert setup.allowed_ids is not None
    head_width = int(setup.allowed_ids.numel())
    manifest["methods"]["SelfConditioningMasked"].update(_rank_contract(head_width))
    manifest["methods"][NL_METHOD] = {
        "factors": list(ADV_FULL_FACTORS),
        "policy": NL_STEERING_POLICY,
        "target_source": {
            "algorithm": NL_RETRIEVAL_ALGORITHM,
            "feature_contract": retrieval_feature_identity(setup),
            "concepts": [
                {"concept_id": concept_id, "text": text.concept}
                for concept_id, text in _training_texts().items()
            ],
            "description_embeddings": _embedding_record(
                description_embeddings,
                project_root=project_root,
                source_sha256=_DESCRIPTION_TEXTS_SHA256,
            ),
            "concept_embeddings": _embedding_record(
                concept_embeddings,
                project_root=project_root,
                source_sha256=_CONCEPT_TEXTS_SHA256,
            ),
        },
    }
    manifest["method_order"].append(NL_METHOD)
    paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    hashes = {key: base._sha(path) for key, path in paths.hashed_files().items()}
    return _NlFixture(
        paths=paths,
        expectations=replace(expectations, include_nl_steering=True),
        hashes=hashes,
        setup=setup,
        description_embeddings=description_embeddings,
        concept_embeddings=concept_embeddings,
        output_dir=tmp_path / "adv-output",
    )


def _manifest(case: _NlFixture) -> dict[str, Any]:
    return json.loads(case.paths.project_manifest.read_text(encoding="utf-8"))


def _write_manifest(case: _NlFixture, manifest: dict[str, Any]) -> None:
    case.paths.project_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    case.hashes["project_manifest"] = base._sha(case.paths.project_manifest)


def _rehash(case: _NlFixture, key: str) -> None:
    case.hashes[key] = base._sha(case.paths.hashed_files()[key])


def _validate(
    case: _NlFixture,
    *,
    expectations: SourceExpectations | None = None,
    output_dir: Path | None = None,
    reserved_paths: tuple[Path, ...] = (),
):
    return validate_source_contract(
        paths=case.paths,
        expectations=case.expectations if expectations is None else expectations,
        expected_hashes=case.hashes,
        output_dir=case.output_dir if output_dir is None else output_dir,
        reserved_paths=reserved_paths,
    )


def test_complete_nl_contract_computes_exact_targets_and_publishes_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    expected = build_retrieval_target_scores(
        setup=case.setup,
        description_embeddings_path=case.description_embeddings,
        concept_embeddings_path=case.concept_embeddings,
        concepts={concept_id: text.concept for concept_id, text in _training_texts().items()},
        expected_hashes={
            "description_embeddings": _sha(case.description_embeddings),
            "concept_embeddings": _sha(case.concept_embeddings),
        },
    )

    validated = _validate(case)

    assert validated.project_frame.shape == (240, 20)
    assert validated.nl_target_scores is not None
    assert tuple(validated.nl_target_scores) == (0, 1)
    for concept_id, target in validated.nl_target_scores.items():
        assert t.equal(target, expected[concept_id])
        assert target.shape == (5,)
        assert target.dtype is t.float32
        assert target.device.type == "cpu"
        assert target.is_contiguous()
        assert bool(t.isfinite(target).all())
        assert bool((target.abs() <= 1.0).all())
    assert tuple(validated.steering_entries[0]["results"]["LMJudgeEvaluator"]) == (
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "SelfConditioning",
        "SelfConditioningMasked",
        NL_METHOD,
    )
    assert validated.runtime_hashes["nl_description_embeddings"] == _sha(
        case.description_embeddings
    )
    assert validated.runtime_hashes["nl_concept_embeddings"] == _sha(
        case.concept_embeddings
    )
    assert validated.runtime_byte_lengths["nl_description_embeddings"] == (
        case.description_embeddings.stat().st_size
    )

    contract_path = write_source_contract(validated, case.output_dir / "source_contract.json")
    document = json.loads(contract_path.read_text(encoding="utf-8"))
    nl_block = _manifest(case)["methods"][NL_METHOD]
    assert document["natural_language"] == nl_block
    assert document["natural_language"]["target_source"]["description_embeddings"][
        "embedding_space"
    ]["library_versions"] == _EMBEDDING_SPACE["library_versions"]
    assert document["runtime_hashes"]["nl_description_embeddings"] == _sha(
        case.description_embeddings
    )
    assert document["runtime_hashes"]["nl_concept_embeddings"] == _sha(
        case.concept_embeddings
    )


def test_valid_nl_manifest_survives_producer_sort_keys_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    manifest = _manifest(case)
    case.paths.project_manifest.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    _rehash(case, "project_manifest")

    validated = _validate(case)
    assert validated.nl_target_scores is not None
    assert list(validated.project_manifest["method_order"])[-1] == NL_METHOD
    assert tuple(validated.project_manifest["methods"])[0] == NL_METHOD


def test_disabled_contract_preserves_the_exact_legacy_source_document_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, expectations, hashes = base._fixture(tmp_path)
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_a, **_k: _legacy_setup(),
    )
    monkeypatch.setattr(
        source_contract_module,
        "select_training_texts",
        lambda *_a, **_k: _training_texts(),
    )
    output_dir = tmp_path / "legacy-output"
    validated = validate_source_contract(
        paths=paths,
        expectations=expectations,
        expected_hashes=hashes,
        output_dir=output_dir,
    )
    output_path = write_source_contract(validated, output_dir / "source_contract.json")
    expected = {
        "schema_version": 2,
        "paths": {
            **{key: str(path.resolve()) for key, path in paths.hashed_files().items()},
            "prepared_dir": str(paths.prepared_dir.resolve()),
            **{key: str(path.resolve()) for key, path in paths.runtime_files().items()},
        },
        "expectations": {
            "max_concepts": 2,
            "repository_git_commit": base.COMMIT,
            "dataset_revision": base.REVISION,
            "target_model": base.MODEL,
            "seed": 42,
            "temperature": 1.0,
            "output_length": 128,
            "package_lock_sha256": "c" * 64,
            "vendored_axbench_upstream_commit": "d" * 40,
            "project_repository_git_commit": base.PROJECT_COMMIT,
            "project_package_lock_sha256": "f" * 64,
            "config_name_prefix": "qwen2_5_3b",
            "layer": 28,
            "require_clean_worktree": True,
            "project_target_step": 54_600,
            "project_target_model_tag": base.MODEL_TAG,
            "model_revision": None,
        },
        "hashes": hashes,
        "runtime_hashes": dict(validated.runtime_hashes),
        "runtime_byte_lengths": dict(validated.runtime_byte_lengths),
        "project_identity": {
            "package_lock_sha256": "f" * 64,
            "repository_git_commit": base.PROJECT_COMMIT,
            "target_model_tag": base.MODEL_TAG,
            "target_step": 54_600,
            "self_conditioning_masked_policy": base.ACCEPTED_SELF_CONDITIONING_MASKED_POLICY,
        },
        "manifest_identity": {
            source: {
                "config_name": f"qwen2_5_3b_{source}.yaml",
                "dataset_revision": base.REVISION,
                "package_lock_sha256": "c" * 64,
                "repository_git_commit": base.COMMIT,
                "target_model": base.MODEL,
            }
            for source in ("main", "aux")
        },
    }
    expected_bytes = json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert output_path.read_text(encoding="utf-8") == expected_bytes
    assert "include_nl_steering" not in expected["expectations"]
    assert "natural_language" not in expected


@pytest.mark.parametrize("surface", ["frame", "jsonl", "manifest"])
def test_disabled_contract_rejects_each_optional_nl_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    disabled = replace(case.expectations, include_nl_steering=False)
    expected_error = {
        "frame": "project frame failed validation",
        "jsonl": "method coverage differs",
        "manifest": "method blocks differ",
    }[surface]
    if surface != "frame":
        frame = pd.read_parquet(case.paths.project_parquet).drop(
            columns=[f"{NL_METHOD}_steered_generation", f"{NL_METHOD}_strength"]
        )
        frame.to_parquet(case.paths.project_parquet, index=False)
        _rehash(case, "project_parquet")
    if surface != "jsonl":
        entries = base._entries()
        base._write_entries(case.paths.evaluated_jsonl, entries)
        _rehash(case, "evaluated_jsonl")
    if surface != "manifest":
        manifest = _manifest(case)
        manifest["methods"].pop(NL_METHOD)
        manifest["method_order"].remove(NL_METHOD)
        _write_manifest(case, manifest)

    with pytest.raises(AxBenchAdvError, match=expected_error):
        _validate(case, expectations=disabled)


@pytest.mark.parametrize("surface", ["frame", "jsonl", "manifest"])
def test_enabled_contract_rejects_a_missing_nl_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    if surface == "frame":
        frame = pd.read_parquet(case.paths.project_parquet).drop(
            columns=[f"{NL_METHOD}_steered_generation", f"{NL_METHOD}_strength"]
        )
        frame.to_parquet(case.paths.project_parquet, index=False)
        _rehash(case, "project_parquet")
        match = "missing NLSteering columns"
    elif surface == "jsonl":
        base._write_entries(case.paths.evaluated_jsonl, base._entries())
        _rehash(case, "evaluated_jsonl")
        match = "method coverage differs"
    else:
        manifest = _manifest(case)
        manifest["methods"].pop(NL_METHOD)
        manifest["method_order"].remove(NL_METHOD)
        _write_manifest(case, manifest)
        match = "method blocks differ"

    with pytest.raises(AxBenchAdvError, match=match):
        _validate(case)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("factor_grid", "NLSteering factor grid differs"),
        ("raw_ratings_length", "NLSteering.raw_aggregated_ratings has wrong length"),
    ],
)
def test_enabled_contract_rejects_malformed_nl_evaluated_jsonl_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    entries = [
        json.loads(line)
        for line in case.paths.evaluated_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    evaluator = cast(dict[str, Any], entries[0]["results"])["LMJudgeEvaluator"]
    block = cast(dict[str, list[Any]], cast(dict[str, Any], evaluator)[NL_METHOD])
    if mutation == "factor_grid":
        block["factor"] = block["factor"][:-1]
    else:
        block["raw_aggregated_ratings"] = block["raw_aggregated_ratings"][:-1]
    base._write_entries(case.paths.evaluated_jsonl, entries)
    _rehash(case, "evaluated_jsonl")

    with pytest.raises(AxBenchAdvError, match=match):
        _validate(case)


@pytest.mark.parametrize("schema", [2, 4])
def test_nl_contract_refuses_non_schema_three_project_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: int,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    manifest = _manifest(case)
    manifest["schema_version"] = schema
    _write_manifest(case, manifest)
    with pytest.raises(AxBenchAdvError, match="rank-era schema-3"):
        _validate(case)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("policy", "policy or factor grid differs"),
        ("factors", "policy or factor grid differs"),
        ("algorithm", "retrieval algorithm differs"),
        ("block_extra", "method block fields differ"),
        ("block_missing", "method block fields differ"),
        ("source_extra", "target_source fields differ"),
        ("source_missing", "target_source fields differ"),
        ("concept_reorder", "concept identities differ"),
        ("concept_text", "concept identities differ"),
        ("concept_id_bool", "concept identities differ"),
        ("original_ids", "feature contract differs"),
        ("feature_mask", "feature contract differs"),
        ("feature_clusters", "feature contract differs"),
    ],
)
def test_nl_method_and_semantic_identity_mutations_fail_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    manifest = _manifest(case)
    block = manifest["methods"][NL_METHOD]
    source = block["target_source"]
    if mutation == "policy":
        block["policy"] = "wrong"
    elif mutation == "factors":
        block["factors"] = [1.0]
    elif mutation == "algorithm":
        source["algorithm"] = "wrong"
    elif mutation == "block_extra":
        block["extra"] = None
    elif mutation == "block_missing":
        block.pop("policy")
    elif mutation == "source_extra":
        source["extra"] = None
    elif mutation == "source_missing":
        source.pop("algorithm")
    elif mutation == "concept_reorder":
        source["concepts"].reverse()
    elif mutation == "concept_text":
        source["concepts"][0]["text"] = "different"
    elif mutation == "concept_id_bool":
        source["concepts"][0]["concept_id"] = False
    elif mutation == "original_ids":
        source["feature_contract"]["original_ids"] = [2, 0, 4, 5, 6]
    elif mutation == "feature_mask":
        source["feature_contract"]["feature_mask"]["ids"].remove(6)
    else:
        source["feature_contract"]["feature_clusters"]["clusters"] = [[0, 1, 3]]
    _write_manifest(case, manifest)

    with pytest.raises(AxBenchAdvError, match=match):
        _validate(case)


def test_equal_vocab_but_different_cluster_membership_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    different_members = _setup(tmp_path, clusters=((0, 1, 3),))
    assert different_members.feature_vocab == case.setup.feature_vocab
    assert retrieval_feature_identity(different_members)["feature_clusters"] != (
        retrieval_feature_identity(case.setup)["feature_clusters"]
    )
    monkeypatch.setattr(
        source_contract_module,
        "load_method_setup",
        lambda *_a, **_k: different_members,
    )

    with pytest.raises(AxBenchAdvError, match="feature contract differs"):
        _validate(case)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("space_mismatch", "embedding spaces differ"),
        ("model_blank", "space provenance is invalid"),
        ("encoding_bool", "space provenance is invalid"),
        ("revision", "space provenance is invalid"),
        ("space_missing", "must declare model, revision, encoding"),
        ("versions_empty", "library versions are invalid"),
        ("versions_bool", "library versions are invalid"),
        ("versions_blank_key", "library versions are invalid"),
        ("source_texts", "source-text identity is invalid"),
        ("path_blank", "path or SHA-256 is invalid"),
        ("sha_bool", "path or SHA-256 is invalid"),
        ("record_extra", "description_embeddings record fields differ"),
        ("record_type", "description_embeddings record fields differ"),
        ("same_path", "embedding files must be distinct"),
    ],
)
def test_embedding_provenance_mutations_fail_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    manifest = _manifest(case)
    source = manifest["methods"][NL_METHOD]["target_source"]
    description = source["description_embeddings"]
    concept = source["concept_embeddings"]
    if mutation == "space_mismatch":
        concept["embedding_space"]["model"] = "different/model"
    elif mutation == "model_blank":
        description["embedding_space"]["model"] = " "
    elif mutation == "encoding_bool":
        description["embedding_space"]["encoding"] = True
    elif mutation == "revision":
        description["embedding_space"]["revision"] = "not-a-commit"
    elif mutation == "space_missing":
        description["embedding_space"].pop("library_versions")
    elif mutation == "versions_empty":
        description["embedding_space"]["library_versions"] = {}
    elif mutation == "versions_bool":
        description["embedding_space"]["library_versions"]["torch"] = True
    elif mutation == "versions_blank_key":
        description["embedding_space"]["library_versions"][" "] = "1.0"
    elif mutation == "source_texts":
        description["source_texts_sha256"] = True
    elif mutation == "path_blank":
        description["path"] = " "
    elif mutation == "sha_bool":
        description["sha256"] = False
    elif mutation == "record_extra":
        description["extra"] = None
    elif mutation == "record_type":
        source["description_embeddings"] = []
    else:
        concept["path"] = description["path"]
    _write_manifest(case, manifest)

    with pytest.raises(AxBenchAdvError, match=match):
        _validate(case)


@pytest.mark.parametrize("field", ["revision", "sha256", "source_texts_sha256"])
@pytest.mark.parametrize("nonhex", ["+", "-", " ", "\t", "\n", "１", "0_"])
def test_embedding_identity_requires_ascii_lowercase_hex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    nonhex: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    manifest = _manifest(case)
    source = manifest["methods"][NL_METHOD]["target_source"]
    width = 40 if field == "revision" else 64
    malformed = nonhex + "0" * (width - len(nonhex))
    # Each value parses as an integer but is not a hexadecimal artifact identity.
    assert int(malformed, 16) >= 0
    for name in ("description_embeddings", "concept_embeddings"):
        record = source[name]["embedding_space"] if field == "revision" else source[name]
        record[field] = malformed
    with pytest.raises(AxBenchAdvError, match="invalid"):
        source_contract_module._nl_source_records(manifest, case.paths.project_manifest)


def test_source_hash_bit_drift_and_missing_file_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    case.description_embeddings.write_bytes(case.description_embeddings.read_bytes() + b"x")
    with pytest.raises(AxBenchAdvError, match="retrieval inputs failed validation"):
        _validate(case)

    case = _nl_fixture(tmp_path / "missing", monkeypatch)
    case.concept_embeddings.unlink()
    with pytest.raises(AxBenchAdvError, match="retrieval inputs failed validation"):
        _validate(case)


def test_final_revalidation_detects_changed_nl_embedding_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    validated = _validate(case)
    case.concept_embeddings.write_bytes(case.concept_embeddings.read_bytes() + b"x")
    with pytest.raises(AxBenchAdvError, match="source hash mismatch"):
        revalidate_source_hashes(validated)
    with pytest.raises(AxBenchAdvError, match="source hash mismatch"):
        write_source_contract(validated, case.output_dir / "must-not-publish.json")


@pytest.mark.parametrize("overlap", ["output", "provider_cache"])
def test_nl_inputs_must_be_disjoint_from_output_and_provider_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap: str,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    output = case.output_dir
    reserved: tuple[Path, ...] = ()
    if overlap == "output":
        output = case.description_embeddings.parent
    else:
        reserved = (case.concept_embeddings.parent,)
    with pytest.raises(AxBenchAdvError, match="provider caches must be path-disjoint"):
        _validate(case, output_dir=output, reserved_paths=reserved)


def test_hardlinked_provider_cache_alias_to_nl_input_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _nl_fixture(tmp_path, monkeypatch)
    cache_dir = tmp_path / "provider-cache"
    cache_dir.mkdir()
    cache_alias = cache_dir / "embedding-cache.npz"
    cache_alias.hardlink_to(case.description_embeddings)
    assert cache_alias.samefile(case.description_embeddings)

    with pytest.raises(AxBenchAdvError, match="provider caches must be path-disjoint"):
        _validate(case, reserved_paths=(cache_alias,))


def test_enabled_source_validation_preserves_legacy_global_rng_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _nl_fixture(tmp_path / "enabled", monkeypatch)
    paths, expectations, hashes = base._fixture(tmp_path / "disabled")
    legacy_setup = _legacy_setup()
    before = random.getstate(), cast(tuple[Any, ...], np.random.get_state()), t.random.get_rng_state()
    # Existing checkpoint-marker beartype validation samples Python random.
    # Compare its legacy transition exactly; NL must introduce no extra draws.
    try:
        monkeypatch.setattr(source_contract_module, "load_method_setup", lambda *_a, **_k: legacy_setup)
        validate_source_contract(
            paths=paths, expectations=expectations, expected_hashes=hashes,
            output_dir=tmp_path / "disabled-output",
        )
        legacy = random.getstate(), cast(tuple[Any, ...], np.random.get_state()), t.random.get_rng_state()
        random.setstate(before[0])
        np.random.set_state(before[1])
        t.random.set_rng_state(before[2])
        monkeypatch.setattr(source_contract_module, "load_method_setup", lambda *_a, **_k: case.setup)
        _validate(case)
        assert random.getstate() == legacy[0]
        observed = cast(tuple[Any, ...], np.random.get_state())
        assert observed[0] == legacy[1][0]
        np.testing.assert_array_equal(observed[1], legacy[1][1])
        assert observed[2:] == legacy[1][2:]
        assert t.equal(t.random.get_rng_state(), legacy[2])
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        t.random.set_rng_state(before[2])
