"""Stage 3e exact encoder-decoder context contract tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet, atomic_write_text
from steering.eval.axbench_base import paths
from steering.eval.axbench_base import stage4_contexts as contexts_mod
from steering.eval.axbench_base.cluster_artifacts import (
    ClusterCandidateIdentity,
    install_clustered_bytes_once,
    write_clustered_parquet_once,
)
from steering.eval.axbench_base.stage4_contexts import (
    CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION,
    STAGE4_CONTEXTS_SCHEMA_VERSION,
    enumerate_stage4_contexts,
    load_stage4_contexts_manifest,
    validate_stage4_contexts_for_generation,
)
from steering.eval.axbench_base.stage4_prompts import (
    build_stage4_prompt_tokens,
    prompting_prefix_text,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    SteeringMode,
    TargetModelMetadata,
)
from steering.eval.cluster_feature_contract import stream_hash_file_stable
from steering.immutable_io import ImmutableArtifactError
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config

_CLUSTER_MANIFEST_FIELDS = (
    "feature_contract_fingerprint",
    "candidate_concepts_sha256",
    "candidate_cluster_provenance_sha256",
    "cluster_concept_projection_version",
    "concepts_sha256",
)


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 7

    def encode(self, text: str) -> list[int]:
        return [100 + ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        pieces: list[str] = []
        for token in tokens:
            if token == 1_000:
                pieces.append("A")
            elif token == 1_001:
                pieces.append("B")
            else:
                pieces.append(chr(token - 100))
        return "".join(pieces)


def _concept(concept_id: int) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=100 + concept_id,
        layer=8,
        sae_source="stub",
        description=f"concept {concept_id}",
        corpus_count=10,
        description_source="stub",
    )


def _seed(doc_idx: int = 0) -> SeedRow:
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=3,
        prefix_source_token_start=0,
        prefix_source_token_end=2,
        prefix_text="AB",
        prefix_content_nanochat_tokens=[1_000, 1_001],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="A",
        continuation_content_nanochat_tokens=[1_000],
    )


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=4,
        n_sae_features=16_384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _write_descriptions(run_tag: str, descriptions: dict[int, str]) -> Path:
    path = test_paths.default_paths(run_tag).prompting_descriptions(SteeringMode.POSITIVE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({str(key): value for key, value in descriptions.items()}),
        encoding="utf-8",
    )
    return path


def _cluster_contract(
    fingerprint: str,
    *,
    representative_ids: tuple[int, ...] = (100,),
) -> Any:
    class _Consolidation:
        @staticmethod
        def assert_representatives_only(feature_ids, *, surface: str) -> None:
            assert surface == "AxBench Stage 3e concept list"
            rejected = sorted(set(feature_ids.tolist()) - set(representative_ids))
            if rejected:
                raise AxBenchBaseError(
                    f"{surface} contains non-representative feature IDs: {rejected}"
                )

    return cast(
        Any,
        SimpleNamespace(
            feature_contract_fingerprint=fingerprint,
            consolidation=_Consolidation(),
        ),
    )


def _context_kwargs() -> dict[str, Any]:
    return {
        "concepts": [_concept(0)],
        "eval_seeds_positive": [_seed()],
        "eval_seeds_negative": [],
        "tokenizer": _Tokenizer(),
        "encdec_meta": _meta(),
        "encdec_sequence_len": 512,
    }


def _write_concepts(run_tag: str, payload: str | None = None) -> Path:
    path = test_paths.default_paths(run_tag).concepts_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        payload = json.dumps([asdict(_concept(0))], separators=(",", ":")) + "\n"
    path.write_text(
        payload,
        encoding="utf-8",
    )
    return path


def _install_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fingerprint: str = "a" * 64,
) -> None:
    def candidate_identity(_cfg, *, expected_fingerprint=None, **_kwargs):
        assert expected_fingerprint == fingerprint
        return ClusterCandidateIdentity(
            feature_contract_fingerprint=fingerprint,
            candidate_concepts_sha256="b" * 64,
            candidate_provenance_sha256="c" * 64,
            concept_projection_version="cluster_concept_projection_v1",
        )

    monkeypatch.setattr(
        contexts_mod,
        "load_cluster_candidate_identity",
        candidate_identity,
    )


def _rewrite_manifest_semantics(
    manifest_path: Path,
    mutate,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    semantic_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"contexts_fingerprint", "files"}
    }
    manifest["contexts_fingerprint"] = contexts_mod.sha256_bytes(
        contexts_mod.canonical_json(semantic_payload).encode("utf-8")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_pre_e1_artifacts(
    cfg,
    *,
    concepts: list[Concept],
    eval_seeds_positive: list[SeedRow],
    eval_seeds_negative: list[SeedRow],
    tokenizer: Any,
    encdec_meta: TargetModelMetadata,
    encdec_sequence_len: int,
) -> Path:
    """Execute main's pre-E1 Stage 3e writer calls as a byte oracle."""

    records, expected_semantics = contexts_mod._build_expected_contract(
        cfg,
        concepts=concepts,
        eval_seeds_positive=eval_seeds_positive,
        eval_seeds_negative=eval_seeds_negative,
        tokenizer=tokenizer,
        encdec_meta=encdec_meta,
        encdec_sequence_len=encdec_sequence_len,
    )
    assert expected_semantics["schema_version"] == STAGE4_CONTEXTS_SCHEMA_VERSION
    assert all(field not in expected_semantics for field in _CLUSTER_MANIFEST_FIELDS)
    assert cfg.run_tag is not None
    context_keys = [record["context_key"] for record in records]
    jsonl_path = test_paths.default_paths(cfg.run_tag).stage4_contexts_jsonl()
    parquet_path = test_paths.default_paths(cfg.run_tag).stage4_contexts_parquet()
    manifest_path = test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest()
    atomic_write_text(contexts_mod._jsonl_payload(records), jsonl_path)
    table = pa.table(
        {
            "candidate_index": pa.array(range(len(records)), type=pa.int64()),
            "text": pa.array(
                [record["context_text"] for record in records], type=pa.string()
            ),
            "context_key": pa.array(context_keys, type=pa.string()),
            "token_count": pa.array(
                [len(record["token_ids"]) for record in records], type=pa.int32()
            ),
        }
    )
    atomic_write_parquet(table, parquet_path)
    run_root = paths.AxBenchBasePaths.from_config(cfg).run_output_root()
    manifest = {
        **expected_semantics,
        "files": {
            "jsonl": {
                "relative_path": str(jsonl_path.relative_to(run_root)),
                "sha256": contexts_mod.sha256_file(jsonl_path),
            },
            "parquet": {
                "relative_path": str(parquet_path.relative_to(run_root)),
                "sha256": contexts_mod.sha256_file(parquet_path),
            },
        },
    }
    atomic_json_dump(manifest, manifest_path)
    return manifest_path


def test_stage3e_matches_live_prompt_builder_and_deduplicates_seed_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag="contexts",
        run_negative_mode=False,
        enabled_methods=["diffmean_encdec", "prompting_encdec"],
    )
    _write_descriptions("contexts", {0: "dragons", 1: "knights"})
    tokenizer = _Tokenizer()
    seed = _seed()
    manifest_path = enumerate_stage4_contexts(
        cfg,
        concepts=[_concept(0), _concept(1)],
        eval_seeds_positive=[seed],
        eval_seeds_negative=[],
        tokenizer=tokenizer,
        encdec_meta=_meta(),
        encdec_sequence_len=512,
    )
    manifest = load_stage4_contexts_manifest(manifest_path)
    assert manifest["context_count"] == 3
    rows = [
        json.loads(line)
        for line in test_paths.default_paths("contexts")
        .stage4_contexts_jsonl()
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = [
        build_stage4_prompt_tokens(
            tokenizer=tokenizer,
            seed_content_tokens=seed.prefix_content_nanochat_tokens,
        ),
        build_stage4_prompt_tokens(
            tokenizer=tokenizer,
            prompting_prefix=prompting_prefix_text(
                SteeringMode.POSITIVE, "dragons"
            ),
            seed_content_tokens=seed.prefix_content_nanochat_tokens,
        ),
        build_stage4_prompt_tokens(
            tokenizer=tokenizer,
            prompting_prefix=prompting_prefix_text(
                SteeringMode.POSITIVE, "knights"
            ),
            seed_content_tokens=seed.prefix_content_nanochat_tokens,
        ),
    ]
    assert [row["token_ids"] for row in rows] == expected


def test_stage3e_allows_no_enabled_encoder_decoder_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag="decoder-only",
        run_negative_mode=False,
        enabled_methods=["diffmean_decoder_only"],
    )
    manifest_path = enumerate_stage4_contexts(
        cfg,
        concepts=[_concept(0)],
        eval_seeds_positive=[_seed()],
        eval_seeds_negative=[],
        tokenizer=_Tokenizer(),
        encdec_meta=_meta(),
        encdec_sequence_len=512,
    )
    manifest = load_stage4_contexts_manifest(manifest_path)
    assert manifest["context_count"] == 0
    assert test_paths.default_paths("decoder-only").stage4_contexts_jsonl().read_text() == ""


@pytest.mark.parametrize("mutation", ["missing", "malformed"])
@pytest.mark.parametrize("field", _CLUSTER_MANIFEST_FIELDS)
def test_schema2_requires_every_well_typed_cluster_manifest_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutation: str,
) -> None:
    """Schema 2 owns exactly five mandatory, independently validated fields."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag=f"schema2-{field}-{mutation}",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    manifest_path = enumerate_stage4_contexts(cfg, **_context_kwargs())

    def install_valid_stamp(manifest: dict[str, Any]) -> None:
        manifest.update(
            {
                "schema_version": CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION,
                "feature_contract_fingerprint": "a" * 64,
                "candidate_concepts_sha256": "b" * 64,
                "candidate_cluster_provenance_sha256": "c" * 64,
                "cluster_concept_projection_version": (
                    "cluster_concept_projection_v1"
                ),
                "concepts_sha256": "d" * 64,
            }
        )

    _rewrite_manifest_semantics(manifest_path, install_valid_stamp)
    load_stage4_contexts_manifest(manifest_path)

    def break_one_field(manifest: dict[str, Any]) -> None:
        if mutation == "missing":
            manifest.pop(field)
        elif field == "cluster_concept_projection_version":
            manifest[field] = ""
        else:
            manifest[field] = "not-a-sha256"

    _rewrite_manifest_semantics(manifest_path, break_one_field)
    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        load_stage4_contexts_manifest(manifest_path)


@pytest.mark.parametrize("field", _CLUSTER_MANIFEST_FIELDS)
def test_schema1_rejects_every_cluster_owned_manifest_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Honest schema-1 producers never emit cluster fields; reject forgeries."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag=f"schema1-{field}",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    manifest_path = enumerate_stage4_contexts(cfg, **_context_kwargs())

    def add_cluster_field(manifest: dict[str, Any]) -> None:
        manifest[field] = (
            "cluster_concept_projection_v1"
            if field == "cluster_concept_projection_version"
            else "a" * 64
        )

    _rewrite_manifest_semantics(manifest_path, add_cluster_field)
    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        load_stage4_contexts_manifest(manifest_path)


@pytest.mark.parametrize(
    "schema_version",
    [[], {}, 2.0, True],
    ids=["list", "mapping", "float", "bool"],
)
def test_manifest_rejects_non_strict_int_schema_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: object,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag=f"invalid-schema-{type(schema_version).__name__}",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    manifest_path = enumerate_stage4_contexts(cfg, **_context_kwargs())

    def install_invalid_schema2_stamp(manifest: dict[str, Any]) -> None:
        if schema_version is True:
            manifest["schema_version"] = schema_version
            return
        manifest.update(
            {
                "schema_version": schema_version,
                "feature_contract_fingerprint": "a" * 64,
                "candidate_concepts_sha256": "b" * 64,
                "candidate_cluster_provenance_sha256": "c" * 64,
                "cluster_concept_projection_version": (
                    "cluster_concept_projection_v1"
                ),
                "concepts_sha256": "d" * 64,
            }
        )

    _rewrite_manifest_semantics(manifest_path, install_invalid_schema2_stamp)
    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        load_stage4_contexts_manifest(manifest_path)


def test_stage3e_rejects_tampering_and_stale_prompt_descriptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag="stale",
        run_negative_mode=False,
        enabled_methods=["prompting_encdec"],
    )
    description_path = _write_descriptions("stale", {0: "dragons"})

    def _enumerate_current_contexts() -> Path:
        return enumerate_stage4_contexts(
            cfg,
            concepts=[_concept(0)],
            eval_seeds_positive=[_seed()],
            eval_seeds_negative=[],
            tokenizer=_Tokenizer(),
            encdec_meta=_meta(),
            encdec_sequence_len=512,
        )

    manifest_path = _enumerate_current_contexts()
    jsonl_path = test_paths.default_paths("stale").stage4_contexts_jsonl()
    jsonl_path.write_text(jsonl_path.read_text() + "corrupt\n", encoding="utf-8")
    with pytest.raises(AxBenchBaseError):
        load_stage4_contexts_manifest(manifest_path)

    # Recreate a valid committed set, then change a semantic input.
    manifest_path.unlink()
    jsonl_path.unlink()
    test_paths.default_paths("stale").stage4_contexts_parquet().unlink()
    _enumerate_current_contexts()
    description_path.write_text(json.dumps({"0": "knights"}), encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        _enumerate_current_contexts()


def test_stage3e_fails_prompt_length_preflight_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    cfg = make_valid_config(
        run_tag="oversized",
        run_negative_mode=False,
        enabled_methods=["diffmean_encdec"],
    )
    with pytest.raises(AxBenchBaseError, match="exceeds"):
        enumerate_stage4_contexts(
            cfg,
            concepts=[_concept(0)],
            eval_seeds_positive=[_seed()],
            eval_seeds_negative=[],
            tokenizer=_Tokenizer(),
            encdec_meta=_meta(),
            encdec_sequence_len=2,
        )
    assert not test_paths.default_paths("oversized").stage4_contexts_manifest().exists()


def test_clustered_stage3e_requires_committed_candidate_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    monkeypatch.setattr(
        contexts_mod,
        "load_cluster_candidate_identity",
        lambda _cfg, **_kwargs: None,
    )
    cfg = make_valid_config(
        run_tag="stage3e-missing-candidate-provenance",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts(cfg.run_tag)

    with pytest.raises(
        AxBenchBaseError,
        match="requires the committed Stage 0a candidate provenance",
    ):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=_cluster_contract("a" * 64),
            **_context_kwargs(),
        )


def test_clustered_stage3e_requires_committed_final_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag="stage3e-missing-final-concepts",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    assert not concepts_path.exists()

    with pytest.raises(
        AxBenchBaseError,
        match="requires committed final concepts",
    ):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=_cluster_contract("a" * 64),
            **_context_kwargs(),
        )


def test_clustered_stage3e_rejects_caller_concepts_that_differ_from_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag="stage3e-concept-binding",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts(cfg.run_tag)
    contract = _cluster_contract(
        "a" * 64,
        representative_ids=(100, 101),
    )
    enumerate_stage4_contexts(
        cfg,
        cluster_contract=contract,
        **_context_kwargs(),
    )
    with pytest.raises(AxBenchBaseError, match="committed final concepts"):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=contract,
            concepts=[_concept(1)],
            eval_seeds_positive=[_seed()],
            eval_seeds_negative=[],
            tokenizer=_Tokenizer(),
            encdec_meta=_meta(),
            encdec_sequence_len=512,
        )


def test_clustered_stage3e_rejects_nonrepresentative_concept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag="stage3e-nonrepresentative",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts(cfg.run_tag)

    with pytest.raises(AxBenchBaseError, match="non-representative"):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=_cluster_contract(
                "a" * 64,
                representative_ids=(101,),
            ),
            **_context_kwargs(),
        )
    assert not test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest().exists()


def test_clustered_stage3e_preserves_prompt_bytes_and_binds_full_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustering changes ownership metadata, never model-ready prompt bytes."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    fingerprint = "a" * 64
    _install_candidate_identity(monkeypatch, fingerprint=fingerprint)
    legacy_cfg = make_valid_config(
        run_tag="stage3e-legacy",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    clustered_cfg = legacy_cfg.replace(run_tag="stage3e-clustered")
    concepts_path = _write_concepts("stage3e-clustered")
    kwargs = _context_kwargs()

    legacy_manifest_path = enumerate_stage4_contexts(legacy_cfg, **kwargs)
    clustered_manifest_path = enumerate_stage4_contexts(
        clustered_cfg,
        cluster_contract=_cluster_contract(fingerprint),
        **kwargs,
    )
    legacy_manifest = load_stage4_contexts_manifest(legacy_manifest_path)
    clustered_manifest = load_stage4_contexts_manifest(clustered_manifest_path)

    assert legacy_manifest["schema_version"] == STAGE4_CONTEXTS_SCHEMA_VERSION
    assert clustered_manifest["schema_version"] == CLUSTERED_STAGE4_CONTEXTS_SCHEMA_VERSION
    assert clustered_manifest["feature_contract_fingerprint"] == fingerprint
    assert clustered_manifest["candidate_concepts_sha256"] == "b" * 64
    assert clustered_manifest["candidate_cluster_provenance_sha256"] == "c" * 64
    assert clustered_manifest["concepts_sha256"] == stream_hash_file_stable(concepts_path).sha256
    assert (
        clustered_manifest["cluster_concept_projection_version"] == "cluster_concept_projection_v1"
    )
    assert test_paths.default_paths("stage3e-clustered").stage4_contexts_jsonl().read_bytes() == (
        test_paths.default_paths("stage3e-legacy").stage4_contexts_jsonl().read_bytes()
    )
    assert test_paths.default_paths("stage3e-clustered").stage4_contexts_parquet().read_bytes() == (
        test_paths.default_paths("stage3e-legacy").stage4_contexts_parquet().read_bytes()
    )

    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        validate_stage4_contexts_for_generation(clustered_cfg, **kwargs)


def test_clustered_stage3e_cluster_content_mutation_invalidates_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )

    def candidate_identity(_cfg, *, expected_fingerprint=None, **_kwargs):
        return ClusterCandidateIdentity(
            feature_contract_fingerprint=str(expected_fingerprint),
            candidate_concepts_sha256="b" * 64,
            candidate_provenance_sha256="c" * 64,
            concept_projection_version="cluster_concept_projection_v1",
        )

    monkeypatch.setattr(
        contexts_mod,
        "load_cluster_candidate_identity",
        candidate_identity,
    )
    cfg = make_valid_config(
        run_tag="stage3e-mutation",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts("stage3e-mutation")
    kwargs = _context_kwargs()
    enumerate_stage4_contexts(
        cfg,
        cluster_contract=_cluster_contract("a" * 64),
        **kwargs,
    )

    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=_cluster_contract("d" * 64),
            **kwargs,
        )


def test_clustered_stage3e_rejects_orphan_artifacts_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    run_tag = "stage3e-orphan"
    cfg = make_valid_config(
        run_tag=run_tag,
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts(run_tag)
    orphan = test_paths.default_paths(run_tag).stage4_contexts_jsonl()
    orphan.write_bytes(b"orphan-must-not-be-overwritten\n")

    with pytest.raises(AxBenchBaseError, match="without.*commit manifest"):
        enumerate_stage4_contexts(
            cfg,
            cluster_contract=_cluster_contract("a" * 64),
            **_context_kwargs(),
        )

    assert orphan.read_bytes() == b"orphan-must-not-be-overwritten\n"
    assert not test_paths.default_paths(run_tag).stage4_contexts_manifest().exists()
    assert not test_paths.default_paths(run_tag).stage4_contexts_parquet().exists()


def test_clustered_stage3e_final_concept_bytes_invalidate_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag="stage3e-concept-mutation",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    concept_path = _write_concepts("stage3e-concept-mutation")
    kwargs = {
        **_context_kwargs(),
        "cluster_contract": _cluster_contract("a" * 64),
    }
    enumerate_stage4_contexts(cfg, **kwargs)
    concept_path.write_text(
        json.dumps([{**asdict(_concept(0)), "description": "changed"}]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AxBenchBaseError, match="fresh run tag"):
        enumerate_stage4_contexts(cfg, **kwargs)


def test_unclustered_stage3e_artifact_bytes_match_pre_e1_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled path retains main's schema-1 calls and exact bytes."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    actual_cfg = make_valid_config(
        run_tag="stage3e-unclustered-actual",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    golden_cfg = actual_cfg.replace(run_tag="stage3e-unclustered-pre-e1")
    kwargs = _context_kwargs()

    enumerate_stage4_contexts(actual_cfg, **kwargs)
    _write_pre_e1_artifacts(golden_cfg, **kwargs)

    actual_paths = test_paths.default_paths(actual_cfg.run_tag)
    golden_paths = test_paths.default_paths(golden_cfg.run_tag)
    for artifact_name in (
        "stage4_contexts_manifest",
        "stage4_contexts_jsonl",
        "stage4_contexts_parquet",
    ):
        assert getattr(actual_paths, artifact_name)().read_bytes() == getattr(
            golden_paths,
            artifact_name,
        )().read_bytes()


def test_clustered_and_unclustered_enumeration_and_validation_records_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write-time and read-time paths emit identical prompt tokens and bytes."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    unclustered_cfg = make_valid_config(
        run_tag="stage3e-enumerate-unclustered",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    clustered_cfg = unclustered_cfg.replace(
        run_tag="stage3e-enumerate-clustered"
    )
    _write_concepts(clustered_cfg.run_tag)
    kwargs = _context_kwargs()
    contract = _cluster_contract("a" * 64)

    enumerate_stage4_contexts(unclustered_cfg, **kwargs)
    enumerate_stage4_contexts(
        clustered_cfg,
        cluster_contract=contract,
        **kwargs,
    )
    _, unclustered_records = validate_stage4_contexts_for_generation(
        unclustered_cfg,
        **kwargs,
    )
    _, clustered_records = validate_stage4_contexts_for_generation(
        clustered_cfg,
        cluster_contract=contract,
        **kwargs,
    )

    assert clustered_records == unclustered_records
    assert [record["token_ids"] for record in clustered_records] == [
        record["token_ids"] for record in unclustered_records
    ]
    assert contexts_mod._jsonl_payload(clustered_records).encode(
        "utf-8"
    ) == contexts_mod._jsonl_payload(unclustered_records).encode("utf-8")
    assert test_paths.default_paths(clustered_cfg.run_tag).stage4_contexts_jsonl().read_bytes() == (
        test_paths.default_paths(unclustered_cfg.run_tag).stage4_contexts_jsonl().read_bytes()
    )


def test_clustered_stage3e_routes_writes_through_immutable_installers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact replay is accepted; different bytes retain the immutable cause."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    installed_surfaces: list[str] = []
    parquet_surfaces: list[str] = []

    def install_spy(payload: bytes, path: Path, *, surface: str) -> bool:
        installed_surfaces.append(surface)
        return install_clustered_bytes_once(payload, path, surface=surface)

    def parquet_spy(table: pa.Table, path: Path, *, surface: str) -> bool:
        parquet_surfaces.append(surface)
        return write_clustered_parquet_once(table, path, surface=surface)

    monkeypatch.setattr(contexts_mod, "install_clustered_bytes_once", install_spy)
    monkeypatch.setattr(contexts_mod, "write_clustered_parquet_once", parquet_spy)
    cfg = make_valid_config(
        run_tag="stage3e-write-once",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    _write_concepts(cfg.run_tag)
    enumerate_stage4_contexts(
        cfg,
        cluster_contract=_cluster_contract("a" * 64),
        **_context_kwargs(),
    )

    assert installed_surfaces == [
        "Stage 3e context JSONL",
        "Stage 3e context commit manifest",
    ]
    assert parquet_surfaces == ["Stage 3e extraction contexts"]
    jsonl_path = test_paths.default_paths(cfg.run_tag).stage4_contexts_jsonl()
    jsonl_bytes = jsonl_path.read_bytes()
    assert (
        install_clustered_bytes_once(
            jsonl_bytes,
            jsonl_path,
            surface="Stage 3e context JSONL",
        )
        is False
    )
    with pytest.raises(AxBenchBaseError) as exc_info:
        install_clustered_bytes_once(
            jsonl_bytes + b"different\n",
            jsonl_path,
            surface="Stage 3e context JSONL",
        )
    assert isinstance(exc_info.value.__cause__, ImmutableArtifactError)


def test_clustered_stage3e_uses_stable_hash_for_final_concept_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered concepts identity uses the descriptor-bound stable hash."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        contexts_mod, "load_seed_pool_fingerprint", lambda _run_tag: "f" * 64
    )
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag="stage3e-stable-concepts",
        run_negative_mode=False,
        enabled_methods=["control_encdec"],
    )
    concepts_path = _write_concepts(cfg.run_tag)
    observed_paths: list[Path] = []

    def stable_load(path: Path, *, surface: str) -> Any:
        assert surface == "final concepts"
        observed_paths.append(path)
        return json.loads(path.read_bytes()), "d" * 64

    monkeypatch.setattr(contexts_mod, "_load_stable_json", stable_load)
    manifest_path = enumerate_stage4_contexts(
        cfg,
        cluster_contract=_cluster_contract("a" * 64),
        **_context_kwargs(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert observed_paths == [concepts_path]
    assert manifest["concepts_sha256"] == "d" * 64
