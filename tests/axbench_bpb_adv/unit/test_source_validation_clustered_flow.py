from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

import steering.eval.axbench_bpb_adv.source_validation as source_validation
from steering.eval.axbench_base.direction_scale import encode_max_activation
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
)
from steering.eval.axbench_bpb.sample_loader import collect_unique_prefixes
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    require_clustered_stage_source,
    resolve_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.source_cleaning import clean_axbench_source_run
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
    BpbAdvSourceValidation,
    _load_stage_0_3_source_for_validation,
    _validate_loaded_stage_0_3_source,
    validate_stage_0_3_source_run,
    write_source_contract,
)
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureContractError,
)
from tests.axbench_bpb_adv.source_run_builders import (
    both_class_rows,
    one_sided_rows,
    synth_row,
    write_stage_0_3_source_run,
)

_BASELINE_TRAINING_CORPUS_METADATA_KEY = b"cluster_baseline_training_corpus_sha256"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _write_parquet_with_metadata(path: Path, metadata: dict[str, str]) -> None:
    table = pq.read_table(path)
    encoded = dict(table.schema.metadata or {})
    encoded.update({key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()})
    pq.write_table(table.replace_schema_metadata(encoded), path)


def _cluster_source(
    root: Path,
    *,
    positive_rows: list[dict[str, Any]] | None = None,
    negative_rows: list[dict[str, Any]] | None = None,
) -> str:
    concepts = [
        {
            "concept_id": 1,
            "feature_index": 1,
            "layer": 12,
            "sae_source": "test",
            "description": "concept",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        }
    ]
    write_stage_0_3_source_run(
        root,
        positive_rows=(both_class_rows("positive") if positive_rows is None else positive_rows),
        negative_rows=(both_class_rows("negative") if negative_rows is None else negative_rows),
        concepts=concepts,
    )

    semantic = {
        "schema_version": 1,
        "n_sae_features": 8,
        "algorithms": {
            "concept_projection": "mask_first_min_allowed_representative_v1",
            "qc_target_reduction": "member_token_amax_own_percentile_v1",
        },
        "feature_clusters": {
            "representative_universe": [1, 3],
            "representative_to_allowed_members": [[1, [1, 2]], [3, [3]]],
        },
        "concept_source": {
            "mode": "statistics_domain",
            "domain": [2],
            "source_to_representative": [[2, 1]],
        },
    }
    fingerprint = _canonical_sha(semantic)
    (root / "feature_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_contract_fingerprint": fingerprint,
                "semantic": semantic,
                "provenance": {"feature_clusters_path": "/original/clusters.yaml"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    candidate = [
        {
            "feature_index": 1,
            "layer": 12,
            "sae_source": "test",
            "description": "concept",
            "corpus_count": 100,
            "description_source": "neuronpedia",
        }
    ]
    candidate_path = root / "candidate_concepts.json"
    candidate_path.write_text(
        json.dumps(candidate, sort_keys=True),
        encoding="utf-8",
    )
    candidate_provenance = {
        "schema_version": 1,
        "feature_contract_fingerprint": fingerprint,
        "candidate_concepts_sha256": _sha256(candidate_path),
        "concept_projection_version": ("mask_first_min_allowed_representative_v1"),
        "counts": {
            "raw_source_ids": 1,
            "mask_allowed_source_ids": 1,
            "unique_representatives": 1,
            "representatives_with_statistics": 1,
            "count_survivors": 1,
            "description_fetch_survivors": 1,
            "described_survivors": 1,
        },
        "representatives": [
            {
                "representative_id": 1,
                "allowed_member_ids": [1, 2],
                "projecting_source_ids": [2],
                "statistics_source_id": 1,
                "outcome": "candidate_emitted",
            }
        ],
    }
    provenance_path = root / "candidate_concepts.cluster_provenance.json"
    provenance_path.write_text(
        json.dumps(candidate_provenance, sort_keys=True),
        encoding="utf-8",
    )

    for name in (
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
    ):
        # Model a legitimate Stage 1c drop: Stage 1b's immutable positive
        # corpus metadata can retain a parent-population target that is absent
        # from the final list, while Stage 1c's negative metadata is exact.
        target_members = (
            {"1": [1, 2], "3": [3]} if name == "synth_training_positive.parquet" else {"1": [1, 2]}
        )
        synth_metadata = {
            "feature_contract_fingerprint": fingerprint,
            "cluster_qc_target_algorithm": ("member_token_amax_own_percentile_v1"),
            "cluster_target_members_json": json.dumps(
                target_members,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        _write_parquet_with_metadata(root / name, synth_metadata)

    stage0c_schema = pa.schema([("concept_id", pa.int32())]).with_metadata(
        {
            b"feature_contract_fingerprint": fingerprint.encode("utf-8"),
            b"candidate_cluster_provenance_sha256": _sha256(provenance_path).encode("utf-8"),
        }
    )
    pq.write_table(
        pa.Table.from_pylist([{"concept_id": 1}], schema=stage0c_schema),
        root / "negative_mode_prefixes.parquet",
    )

    stage2 = {
        "feature_ids": t.tensor([[-1, -1], [1, -1]], dtype=t.int32),
        "canonical_feature_id": t.tensor([-1, 1], dtype=t.int32),
    }
    for name in (
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
    ):
        save_file(
            stage2,
            str(root / name),
            metadata={"feature_contract_fingerprint": fingerprint},
        )

    shutil.rmtree(root / "baselines")
    corpus_hashes = {
        "positive": _sha256(root / "synth_training_positive.parquet"),
        "negative": _sha256(root / "synth_training_negative.parquet"),
    }
    for family in ("diffmean", "linear_probe"):
        for mode in ("positive", "negative"):
            path = root / "baselines" / family / "decoder_only" / mode / "concept_0001.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"direction": t.ones(4)},
                str(path),
                metadata={
                    "artifact_kind": f"{family}_direction_v1",
                    "concept_id": "1",
                    "feature_contract_fingerprint": fingerprint,
                    "layer": "6",
                    "source_corpus_sha256": corpus_hashes[mode],
                    "target_model": "decoder_only",
                    "mode": mode,
                    "max_activation": encode_max_activation(2.5),
                },
            )
    return fingerprint


def _legacy_unclustered_contract(
    loaded: source_validation._LoadedStage03Source,
) -> BpbAdvSourceContract:
    """Inline the exact pre-G2c unclustered contract-commit loop."""

    artifact_byte_sizes: dict[str, int] = {}
    artifact_entry_counts: dict[str, int] = {}
    artifact_hashes: dict[str, str] = {}
    for name, path in source_validation._required_artifacts(loaded.paths):
        if path.is_dir():
            byte_size, entry_count = source_validation._directory_byte_size_and_entry_count(path)
            artifact_byte_sizes[name] = byte_size
            artifact_entry_counts[name] = entry_count
            artifact_hashes[name] = source_validation._hash_directory(path)
        else:
            artifact_byte_sizes[name] = path.stat().st_size
            artifact_hashes[name] = source_validation._hash_required_file(
                name,
                path,
            )
    return BpbAdvSourceContract(
        source_run_dir=str(loaded.contract_source_run_dir),
        seed_schema_version=SEED_SCHEMA_VERSION,
        seed_fingerprint=loaded.seed_fingerprint,
        seed_token_semantics=SEED_TOKEN_SEMANTICS,
        artifact_byte_sizes=artifact_byte_sizes,
        artifact_entry_counts=artifact_entry_counts,
        artifact_hashes=artifact_hashes,
        n_dropped=0,
    )


def _resolve_stage_source_fixture(
    *,
    tmp_path: Path,
    source_name: str = "source",
    output_name: str = "out",
) -> tuple[
    Path,
    BpbAdvSourceValidation,
    AxBenchBpbAdvPaths,
    ClusterFeatureContract,
    ClusteredStageSource,
]:
    root = tmp_path / source_name
    _cluster_source(root)
    validation = validate_stage_0_3_source_run(root)
    paths = AxBenchBpbAdvPaths(
        output_root=tmp_path / output_name,
        run_tag="run",
    )
    paths.root.mkdir(parents=True)
    write_source_contract(paths.source_contract(), validation.contract)
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(
            feature_contract_fingerprint=(validation.contract.feature_contract_fingerprint)
        ),
    )
    stage_source = resolve_clustered_stage_source(
        validation=validation,
        paths=paths,
        contract=contract,
        expected_source_run_dir=root,
        expected_filter_unaccepted_samples=True,
    )
    return root, validation, paths, contract, stage_source


def _render_contract(contract: BpbAdvSourceContract) -> bytes:
    return (
        json.dumps(
            contract.to_json_dict(),
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def test_clustered_source_end_to_end_accepts_and_binds_all_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)

    validation = validate_stage_0_3_source_run(root)
    contract = validation.contract

    assert contract.schema_version == CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION
    assert contract.feature_contract_fingerprint == fingerprint
    assert contract.candidate_provenance_sha256 == _sha256(
        root / "candidate_concepts.cluster_provenance.json"
    )
    assert contract.stage2_feature_contract_fingerprints == {
        "concept_features.safetensors": fingerprint,
        "concept_features_neg.safetensors": fingerprint,
    }
    assert contract.baseline_training_corpus_sha256 == {
        "positive": _sha256(root / "synth_training_positive.parquet"),
        "negative": _sha256(root / "synth_training_negative.parquet"),
    }
    assert set(contract.artifact_hashes) == {
        name
        for name, _path in source_validation._required_artifacts(
            SourcePaths(root),
            clustered=True,
        )
    }
    seed_path = root / "seeds_train.safetensors"
    assert contract.artifact_hashes["seeds_train.safetensors"] == (
        source_validation._hash_required_file(
            "seeds_train.safetensors",
            seed_path,
        )
    )
    assert contract.artifact_hashes["seeds_train.safetensors"] != (
        source_validation._hash_file(seed_path)
    )
    assert contract.artifact_hashes["baselines/"] == (
        source_validation._hash_directory(root / "baselines")
    )
    assert validation.n_concepts == 1
    assert validation.n_positive_prefixes == 1
    assert validation.n_negative_prefixes == 1


def test_clustered_validation_rows_survive_post_validation_path_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)

    validation = validate_stage_0_3_source_run(root)
    positive_path = root / "synth_training_positive.parquet"
    negative_path = root / "synth_training_negative.parquet"
    for path in (positive_path, negative_path):
        original = pq.read_table(path)
        replacement_rows = [
            {**row, "prefix_text": f"replacement-{path.stem}"} for row in original.to_pylist()
        ]
        pq.write_table(
            pa.Table.from_pylist(replacement_rows, schema=original.schema),
            path,
        )

    positive_rows = validation.positive_rows()
    negative_rows = validation.negative_rows()

    assert {str(row["prefix_text"]) for row in positive_rows} == {"positive"}
    assert {str(row["prefix_text"]) for row in negative_rows} == {"negative"}
    assert all(
        not str(row["prefix_text"]).startswith("replacement-")
        for row in [*positive_rows, *negative_rows]
    )


def test_clustered_direction_payload_survives_post_validation_path_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    path = (
        root / "baselines" / "diffmean" / "decoder_only" / "positive" / "concept_0001.safetensors"
    )
    original = path.read_bytes()

    validation = validate_stage_0_3_source_run(root)
    save_file(
        {"direction": t.full((4,), 999.0)},
        str(path),
        metadata={
            "concept_id": "1",
            "feature_contract_fingerprint": cast(
                str,
                validation.contract.feature_contract_fingerprint,
            ),
            "layer": "6",
            "mode": "positive",
            "source_corpus_sha256": (
                validation.contract.baseline_training_corpus_sha256["positive"]
            ),
            "target_model": "decoder_only",
        },
    )
    assert path.read_bytes() != original

    retained = validation.direction_artifact_bytes(
        family="diffmean",
        target="decoder_only",
        mode="positive",
        concept_id=1,
    )

    assert retained == original
    assert retained != path.read_bytes()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "family": "lora",
                "target": "decoder_only",
                "mode": "positive",
                "concept_id": 1,
            },
            "family must be",
        ),
        (
            {
                "family": "diffmean",
                "target": "encdec",
                "mode": "positive",
                "concept_id": 1,
            },
            "missing the certified direction payload",
        ),
        (
            {
                "family": "diffmean",
                "target": "decoder_only",
                "mode": "sideways",
                "concept_id": 1,
            },
            "mode must be",
        ),
        (
            {
                "family": "diffmean",
                "target": "decoder_only",
                "mode": "positive",
                "concept_id": True,
            },
            "validated source concept",
        ),
        (
            {
                "family": "diffmean",
                "target": "decoder_only",
                "mode": "positive",
                "concept_id": 999,
            },
            "validated source concept",
        ),
    ],
)
def test_clustered_direction_payload_rejects_invalid_coordinates(
    tmp_path: Path,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    validation = validate_stage_0_3_source_run(root)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validation.direction_artifact_bytes(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["payload_bytes", "drop_payload", "child_kind", "drop_child"],
)
def test_clustered_validation_rejects_replaced_baseline_payload_constituents(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    validation = validate_stage_0_3_source_run(root)
    payloads = list(validation._baseline_artifact_payloads)
    kinds = list(validation._baseline_child_kinds)

    if mutation == "payload_bytes":
        relative, payload = payloads[0]
        payloads[0] = (relative, payload + b"corruption")
    elif mutation == "drop_payload":
        payloads.pop()
    elif mutation == "child_kind":
        relative, _kind = kinds[0]
        kinds[0] = (relative, "file")
    elif mutation == "drop_child":
        kinds.pop()
    else:  # pragma: no cover - parametrization exhaustiveness sentinel
        raise AssertionError(mutation)

    with pytest.raises(
        AxBenchBpbAdvError,
        match=(
            "baseline manifest and retained payloads disagree"
            "|retained baseline payloads do not match"
        ),
    ):
        dataclasses.replace(
            validation,
            _baseline_artifact_payloads=tuple(payloads),
            _baseline_child_kinds=tuple(kinds),
        )


@pytest.mark.parametrize(
    ("child_kinds", "child_payloads", "match"),
    [
        (
            (("z", "file"), ("a", "file")),
            (("z", b"z"), ("a", b"a")),
            "malformed baseline child manifest",
        ),
        (
            (("a", "file"),),
            (("a", b"a"), ("a", b"duplicate")),
            "malformed baseline retained payloads",
        ),
    ],
)
def test_baseline_retained_manifest_rejects_malformed_structure(
    child_kinds: tuple[tuple[str, str], ...],
    child_payloads: tuple[tuple[str, bytes], ...],
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match):
        source_validation._baseline_payload_contract_metrics(
            child_kinds=child_kinds,
            child_payloads=child_payloads,
        )


def test_unclustered_validation_retains_directionless_snapshot_safely(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    validation = validate_stage_0_3_source_run(root)

    assert validation.contract.direction_training_layer is None
    with pytest.raises(AxBenchBpbAdvError, match="retained baseline payloads"):
        dataclasses.replace(
            validation,
            _baseline_artifact_payloads=(("direction", b"payload"),),
            _baseline_child_kinds=(("direction", "file"),),
        )
    with pytest.raises(AxBenchBpbAdvError, match="missing the certified"):
        validation.direction_artifact_bytes(
            family="diffmean",
            target="decoder_only",
            mode="positive",
            concept_id=1,
        )


def test_unclustered_final_snapshot_rejects_direction_cluster_stamp_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    path = (
        root / "baselines" / "diffmean" / "decoder_only" / "positive" / "concept_0001.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    base_metadata = {
        "artifact_kind": "diffmean_direction_v1",
        "concept_id": "1",
        "layer": "6",
        "mode": "positive",
        "target_model": "decoder_only",
    }
    save_file({"direction": t.ones(4)}, str(path), metadata=base_metadata)

    real_snapshot = source_validation._snapshot_artifact
    replaced = False

    def cluster_stamp_before_final_snapshot(name: str, artifact_path: Path) -> Any:
        nonlocal replaced
        if name == "baselines/" and not replaced:
            replaced = True
            save_file(
                {"direction": t.ones(4)},
                str(path),
                metadata={**base_metadata, "source_corpus_sha256": "b" * 64},
            )
        return real_snapshot(name, artifact_path)

    monkeypatch.setattr(
        source_validation,
        "_snapshot_artifact",
        cluster_stamp_before_final_snapshot,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="committed baseline snapshot",
    ):
        validate_stage_0_3_source_run(root)
    assert replaced


def test_unclustered_final_snapshot_rejects_nondirection_cluster_stamp_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    path = root / "baselines" / "lora" / "decoder_only.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"adapter": t.ones(4)}, str(path), metadata={"format": "test"})

    real_snapshot = source_validation._snapshot_artifact
    replaced = False

    def cluster_stamp_before_final_snapshot(name: str, artifact_path: Path) -> Any:
        nonlocal replaced
        if name == "baselines/" and not replaced:
            replaced = True
            save_file(
                {"adapter": t.ones(4)},
                str(path),
                metadata={
                    "format": "test",
                    "feature_contract_fingerprint": "a" * 64,
                },
            )
        return real_snapshot(name, artifact_path)

    monkeypatch.setattr(
        source_validation,
        "_snapshot_artifact",
        cluster_stamp_before_final_snapshot,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="committed baseline snapshot",
    ):
        validate_stage_0_3_source_run(root)
    assert replaced


def test_direction_accessor_fails_if_certified_payload_is_lost(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    validation = validate_stage_0_3_source_run(root)
    relative = "diffmean/decoder_only/positive/concept_0001.safetensors"
    retained = tuple(
        entry for entry in validation._baseline_artifact_payloads if entry[0] != relative
    )
    object.__setattr__(validation, "_baseline_artifact_payloads", retained)

    with pytest.raises(AxBenchBpbAdvError, match="missing the certified"):
        validation.direction_artifact_bytes(
            family="diffmean",
            target="decoder_only",
            mode="positive",
            concept_id=1,
        )


def test_clustered_baseline_path_fallback_matches_retained_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)
    paths = SourcePaths(root)
    concepts = json.loads(paths.concepts_json().read_text(encoding="utf-8"))

    corpus_hashes, direction_validation = source_validation._validate_clustered_baselines(
        paths=paths,
        concepts=concepts,
        expected_fingerprint=fingerprint,
        positive_table=pq.read_table(paths.synth_training_positive()),
        negative_table=pq.read_table(paths.synth_training_negative()),
    )

    assert corpus_hashes == {
        "positive": _sha256(paths.synth_training_positive()),
        "negative": _sha256(paths.synth_training_negative()),
    }
    assert direction_validation.direction_training_layer == 6
    assert len(direction_validation.artifacts) == 4


def test_e10_clustered_baselines_aggregate_missing_max_activation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)
    paths = SourcePaths(root)
    corpus_hashes = {
        "positive": _sha256(paths.synth_training_positive()),
        "negative": _sha256(paths.synth_training_negative()),
    }
    relative_paths: list[str] = []
    for family in ("diffmean", "linear_probe"):
        for mode in ("positive", "negative"):
            relative = (
                f"{family}/decoder_only/{mode}/concept_0001.safetensors"
            )
            relative_paths.append(relative)
            save_file(
                {"direction": t.ones(4)},
                str(paths.baselines_root() / relative),
                metadata={
                    "artifact_kind": f"{family}_direction_v1",
                    "concept_id": "1",
                    "feature_contract_fingerprint": fingerprint,
                    "layer": "6",
                    "source_corpus_sha256": corpus_hashes[mode],
                    "target_model": "decoder_only",
                    "mode": mode,
                },
            )

    concepts = json.loads(paths.concepts_json().read_text(encoding="utf-8"))
    with pytest.raises(
        AxBenchBpbAdvError,
        match="missing 'max_activation'",
    ) as exc_info:
        source_validation._validate_clustered_baselines(
            paths=paths,
            concepts=concepts,
            expected_fingerprint=fingerprint,
            positive_table=pq.read_table(paths.synth_training_positive()),
            negative_table=pq.read_table(paths.synth_training_negative()),
        )

    message = str(exc_info.value)
    assert message.count("missing 'max_activation'") == 4
    assert all(relative in message for relative in relative_paths)


def test_e10_clustered_payload_rejects_zero_value_on_nonzero_direction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)
    paths = SourcePaths(root)
    path = paths.diffmean_direction("decoder_only", "positive", 1)
    save_file(
        {"direction": t.ones(4)},
        str(path),
        metadata={
            "concept_id": "1",
            "artifact_kind": "diffmean_direction_v1",
            "feature_contract_fingerprint": fingerprint,
            "layer": "6",
            "source_corpus_sha256": _sha256(
                paths.synth_training_positive()
            ),
            "target_model": "decoder_only",
            "mode": "positive",
            "max_activation": "0.0",
        },
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="zero-direction max_activation mismatch",
    ):
        validate_stage_0_3_source_run(root)


@pytest.mark.parametrize(
    ("metadata_mutation", "match"),
    [
        ("absent", "missing metadata 'concept_id'"),
        ("noncanonical_layer", "noncanonical layer"),
    ],
)
def test_clustered_baseline_path_fallback_rejects_impossible_metadata_result(
    tmp_path: Path,
    metadata_mutation: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)
    paths = SourcePaths(root)
    concepts = json.loads(paths.concepts_json().read_text(encoding="utf-8"))
    direction_path = paths.diffmean_direction("decoder_only", "positive", 1)
    metadata = None
    if metadata_mutation == "noncanonical_layer":
        metadata = {
            "artifact_kind": "diffmean_direction_v1",
            "concept_id": "1",
            "feature_contract_fingerprint": fingerprint,
            "layer": "01",
            "source_corpus_sha256": _sha256(paths.synth_training_positive()),
            "target_model": "decoder_only",
            "mode": "positive",
        }
    save_file({"direction": t.ones(4)}, str(direction_path), metadata=metadata)
    with pytest.raises(AxBenchBpbAdvError, match=match):
        source_validation._validate_clustered_baselines(
            paths=paths,
            concepts=concepts,
            expected_fingerprint=fingerprint,
            positive_table=pq.read_table(paths.synth_training_positive()),
            negative_table=pq.read_table(paths.synth_training_negative()),
        )


def test_clustered_stage_source_pairs_snapshot_with_committed_identity(
    tmp_path: Path,
) -> None:
    root, validation, _paths, contract, stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )
    SourcePaths(root).concepts_json().write_text(
        '[{"concept_id": 999, "description": "replacement"}]',
        encoding="utf-8",
    )

    assert stage_source.validation is validation
    assert stage_source.source_identity.source_contract_sha256 == validation.contract_hash
    assert validation.concept_ids == frozenset({1})
    assert [entry["concept_id"] for entry in validation.concept_entries()] == [1]
    assert (
        require_clustered_stage_source(
            stage_source=stage_source,
            contract=contract,
            surface="test",
        )
        is stage_source
    )


def test_validated_concept_entries_are_independent_fresh_copies(
    tmp_path: Path,
) -> None:
    _root, validation, _paths, _contract, _stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )

    first = validation.concept_entries()
    first[0]["concept_id"] = 999
    first.append({"concept_id": 1000})

    second = validation.concept_entries()
    assert [entry["concept_id"] for entry in second] == [1]
    assert second is not first
    assert second[0] is not first[0]


def test_clustered_stage_source_rejects_unpaired_or_wrong_coordinates(
    tmp_path: Path,
) -> None:
    root, validation, paths, contract, stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )

    with pytest.raises(TypeError, match="must be built by"):
        ClusteredStageSource(
            validation=validation,
            source_identity=stage_source.source_identity,
        )
    with pytest.raises(AxBenchBpbAdvError, match="different source run"):
        resolve_clustered_stage_source(
            validation=validation,
            paths=paths,
            contract=contract,
            expected_source_run_dir=tmp_path / "other",
            expected_filter_unaccepted_samples=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="different.*policy"):
        resolve_clustered_stage_source(
            validation=validation,
            paths=paths,
            contract=contract,
            expected_source_run_dir=root,
            expected_filter_unaccepted_samples=False,
        )
    wrong_contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(feature_contract_fingerprint="f" * 64),
    )
    with pytest.raises(AxBenchBpbAdvError, match="target feature contract"):
        resolve_clustered_stage_source(
            validation=validation,
            paths=paths,
            contract=wrong_contract,
            expected_source_run_dir=root,
            expected_filter_unaccepted_samples=True,
        )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered"):
        require_clustered_stage_source(
            stage_source=stage_source,
            contract=None,
            surface="test",
        )
    with pytest.raises(AxBenchBpbAdvError, match="requires one"):
        require_clustered_stage_source(
            stage_source=None,
            contract=contract,
            surface="test",
        )
    with pytest.raises(AxBenchBpbAdvError, match="target feature contract"):
        require_clustered_stage_source(
            stage_source=stage_source,
            contract=wrong_contract,
            surface="test",
        )


def test_clustered_stage_source_rejects_replaced_constituents(
    tmp_path: Path,
) -> None:
    _root, validation, _paths, _contract, stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )

    with pytest.raises(TypeError, match="exact BpbAdvSourceValidation"):
        dataclasses.replace(
            stage_source,
            validation=SimpleNamespace(),
        )
    with pytest.raises(TypeError, match="exact ClusteredSourceIdentity"):
        dataclasses.replace(
            stage_source,
            source_identity=SimpleNamespace(),
        )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="contract hash is internally inconsistent",
    ):
        dataclasses.replace(
            validation,
            contract_hash="f" * 64,
        )

    wrong_hash_identity = dataclasses.replace(
        stage_source.source_identity,
        source_contract_sha256="f" * 64,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="snapshot and committed source identity disagree",
    ):
        dataclasses.replace(
            stage_source,
            source_identity=wrong_hash_identity,
        )

    wrong_feature_identity = dataclasses.replace(
        stage_source.source_identity,
        feature_contract_fingerprint="f" * 64,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="snapshot and committed feature identity disagree",
    ):
        dataclasses.replace(
            stage_source,
            source_identity=wrong_feature_identity,
        )


def test_clustered_stage_source_revalidates_on_every_consumption(
    tmp_path: Path,
) -> None:
    _root, validation, _paths, contract, stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )
    object.__setattr__(validation, "contract_hash", "f" * 64)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="contract hash is internally inconsistent",
    ):
        require_clustered_stage_source(
            stage_source=stage_source,
            contract=contract,
            surface="test",
        )


def test_stage_source_resolution_rejects_different_committed_snapshot(
    tmp_path: Path,
) -> None:
    root_a, validation_a, _paths_a, contract_a, _stage_source_a = _resolve_stage_source_fixture(
        tmp_path=tmp_path,
        source_name="source-a",
        output_name="out-a",
    )
    _root_b, _validation_b, paths_b, _contract_b, _stage_source_b = _resolve_stage_source_fixture(
        tmp_path=tmp_path,
        source_name="source-b",
        output_name="out-b",
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="snapshot and committed source identity disagree",
    ):
        resolve_clustered_stage_source(
            validation=validation_a,
            paths=paths_b,
            contract=contract_a,
            expected_source_run_dir=root_a,
            expected_filter_unaccepted_samples=True,
        )


def test_stage_source_resolution_rejects_wrong_validation_type_or_schema(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(
        output_root=tmp_path / "out",
        run_tag="run",
    )
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(feature_contract_fingerprint="f" * 64),
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="exact BpbAdvSourceValidation",
    ):
        resolve_clustered_stage_source(
            validation=cast(BpbAdvSourceValidation, SimpleNamespace()),
            paths=paths,
            contract=contract,
            expected_source_run_dir=tmp_path / "source",
            expected_filter_unaccepted_samples=True,
        )

    legacy_root = tmp_path / "legacy-source"
    write_stage_0_3_source_run(
        legacy_root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    legacy_validation = validate_stage_0_3_source_run(legacy_root)
    assert legacy_validation.contract.schema_version == (SOURCE_CONTRACT_SCHEMA_VERSION)
    with pytest.raises(AxBenchBpbAdvError, match="schema-v7"):
        resolve_clustered_stage_source(
            validation=legacy_validation,
            paths=paths,
            contract=contract,
            expected_source_run_dir=legacy_root,
            expected_filter_unaccepted_samples=True,
        )


def test_unclustered_stage_source_boundary_accepts_only_none() -> None:
    assert (
        require_clustered_stage_source(
            stage_source=None,
            contract=None,
            surface="test",
        )
        is None
    )


def test_validated_concept_entries_reject_internal_shape_corruption(
    tmp_path: Path,
) -> None:
    _root, validation, _paths, _contract, _stage_source = _resolve_stage_source_fixture(
        tmp_path=tmp_path
    )
    object.__setattr__(validation, "_concept_entries_json", "{}")

    with pytest.raises(AxBenchBpbAdvError, match="internally malformed"):
        validation.concept_entries()


@pytest.mark.parametrize("filter_unaccepted", [True, False])
def test_context_prefixes_match_path_reference_and_survive_replacement(
    tmp_path: Path,
    filter_unaccepted: bool,
) -> None:
    root = tmp_path / "source"
    positive_rows = [
        *both_class_rows("positive"),
        *both_class_rows("shared"),
        *both_class_rows("outside-positive", concept_id=999),
        synth_row(
            "rejected-positive",
            content_class="concept_rich",
            qc_accepted=False,
        ),
        synth_row(
            "rejected-positive",
            content_class="concept_avoiding",
            qc_accepted=False,
        ),
    ]
    negative_rows = [
        *both_class_rows("negative"),
        *both_class_rows("shared"),
        *both_class_rows("outside-negative", concept_id=999),
        synth_row(
            "rejected-negative",
            content_class="concept_rich",
            qc_accepted=False,
        ),
        synth_row(
            "rejected-negative",
            content_class="concept_avoiding",
            qc_accepted=False,
        ),
    ]
    _cluster_source(
        root,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
    )
    positive_path = root / "synth_training_positive.parquet"
    negative_path = root / "synth_training_negative.parquet"
    all_prefixes = collect_unique_prefixes(
        positive_path,
        negative_path,
        filter_unaccepted=filter_unaccepted,
    )
    expected = {
        prefix: prefix_hash
        for prefix, prefix_hash in all_prefixes.items()
        if not prefix.startswith("outside-")
    }

    validation = validate_stage_0_3_source_run(
        root,
        filter_unaccepted_samples=filter_unaccepted,
    )
    for path in (positive_path, negative_path):
        original = pq.read_table(path)
        replacement = pa.Table.from_pylist(
            [
                {**row, "prefix_text": f"replacement-{index}"}
                for index, row in enumerate(original.to_pylist())
            ],
            schema=original.schema,
        )
        pq.write_table(replacement, path)

    assert validation.context_prefixes() == expected
    assert "outside-positive" not in expected
    assert "outside-negative" not in expected
    assert ("rejected-positive" in expected) is (not filter_unaccepted)
    assert ("rejected-negative" in expected) is (not filter_unaccepted)
    assert all(not prefix.startswith("replacement-") for prefix in expected)


def test_context_prefixes_reject_non_string_values() -> None:
    positive_table = pa.Table.from_pylist([{"prefix_text": 17, "qc_accepted": True}])
    negative_table = pa.Table.from_pylist(
        [],
        schema=positive_table.schema,
    )
    with pytest.raises(AxBenchBpbAdvError, match="must be strings"):
        source_validation._context_prefixes_from_tables(
            positive_table=positive_table,
            negative_table=negative_table,
            filter_unaccepted=True,
        )


def test_source_validation_cannot_be_constructed_outside_factory() -> None:
    with pytest.raises(TypeError, match="must be built by"):
        source_validation.BpbAdvSourceValidation(
            source_run_dir=Path("/unused"),
            contract=cast(Any, None),
            contract_hash="0" * 64,
            filter_unaccepted_samples=True,
            n_concepts=0,
            n_positive_synth_rows=0,
            n_negative_synth_rows=0,
            n_positive_prefixes=0,
            n_negative_prefixes=0,
            n_dropped=0,
            _positive_table=pa.table({}),
            _negative_table=pa.table({}),
            _concept_ids=frozenset(),
            _concept_entries_json="[]",
        )


def test_clustered_loader_parses_synth_tables_from_retained_snapshot_buffers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    real_read_table = source_validation.pq.read_table
    table_sources: list[object] = []

    def _read_table(source: object, *args: Any, **kwargs: Any) -> pa.Table:
        table_sources.append(source)
        return real_read_table(source, *args, **kwargs)

    monkeypatch.setattr(source_validation.pq, "read_table", _read_table)

    loaded = _load_stage_0_3_source_for_validation(root)

    assert loaded.positive.table.num_rows == 2
    assert loaded.negative.table.num_rows == 2
    assert len(table_sources) == 2
    assert all(isinstance(source, pa.BufferReader) for source in table_sources)


def test_unclustered_loader_preserves_path_backed_synth_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    real_read_table = source_validation.pq.read_table
    table_sources: list[object] = []

    def _read_table(source: object, *args: Any, **kwargs: Any) -> pa.Table:
        table_sources.append(source)
        return real_read_table(source, *args, **kwargs)

    monkeypatch.setattr(source_validation.pq, "read_table", _read_table)

    loaded = _load_stage_0_3_source_for_validation(root)

    paths = SourcePaths(root)
    assert loaded.positive.table.num_rows == 2
    assert loaded.negative.table.num_rows == 2
    assert table_sources == [
        paths.synth_training_positive(),
        paths.synth_training_negative(),
    ]


def test_clustered_baseline_corpus_hash_uses_snapshotted_synth_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    path = root / "synth_training_positive.parquet"
    table = pq.read_table(path)
    snapshotted_sha256 = _sha256(path)

    monkeypatch.setattr(
        source_validation,
        "stream_hash_file_stable",
        lambda _path: pytest.fail("snapshotted corpus must not reopen its path"),
    )

    assert (
        source_validation._baseline_training_corpus_sha256(
            table=table,
            path=path,
            source_file_sha256=snapshotted_sha256,
        )
        == snapshotted_sha256
    )


@pytest.mark.parametrize("invalid_digest", ["", "a" * 63, "A" * 64, "not-sha256"])
def test_clustered_baseline_corpus_hash_rejects_malformed_snapshot_digest(
    tmp_path: Path,
    invalid_digest: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    path = root / "synth_training_positive.parquet"

    with pytest.raises(AxBenchBpbAdvError, match="malformed snapshotted source digest"):
        source_validation._baseline_training_corpus_sha256(
            table=pq.read_table(path),
            path=path,
            source_file_sha256=invalid_digest,
        )


def test_unclustered_source_contract_is_byte_identical_to_pre_g2c_flow(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    loaded = _load_stage_0_3_source_for_validation(root)
    expected = _legacy_unclustered_contract(loaded)

    actual = _validate_loaded_stage_0_3_source(loaded).contract

    assert actual.schema_version == SOURCE_CONTRACT_SCHEMA_VERSION
    assert _render_contract(actual) == _render_contract(expected)
    seed_path = root / "seeds_train.safetensors"
    assert actual.artifact_hashes["seeds_train.safetensors"] == (
        source_validation._hash_required_file(
            "seeds_train.safetensors",
            seed_path,
        )
    )
    assert actual.artifact_hashes["seeds_train.safetensors"] != (
        source_validation._hash_file(seed_path)
    )


def _stamp_unclustered_artifact(root: Path, surface: str) -> None:
    fingerprint = "a" * 64
    if surface == "candidate_provenance":
        (root / "candidate_concepts.cluster_provenance.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return
    if surface in {"positive_parquet", "negative_parquet"}:
        name = (
            "synth_training_positive.parquet"
            if surface == "positive_parquet"
            else "synth_training_negative.parquet"
        )
        _write_parquet_with_metadata(
            root / name,
            {"feature_contract_fingerprint": fingerprint},
        )
        return
    if surface == "negative_prefixes":
        schema = pa.schema([("concept_id", pa.int32())]).with_metadata(
            {b"feature_contract_fingerprint": fingerprint.encode("ascii")}
        )
        pq.write_table(
            pa.Table.from_pylist([{"concept_id": 1}], schema=schema),
            root / "negative_mode_prefixes.parquet",
        )
        return
    if surface in {"stage2_positive", "stage2_negative"}:
        name = (
            "concept_features.safetensors"
            if surface == "stage2_positive"
            else "concept_features_neg.safetensors"
        )
        save_file(
            {"feature_ids": t.tensor([1], dtype=t.int64)},
            str(root / name),
            metadata={"feature_contract_fingerprint": fingerprint},
        )
        return
    if surface in {"baseline", "baseline_source_corpus"}:
        path = root / "baselines" / "stamped.safetensors"
        save_file(
            {"direction": t.ones(1)},
            str(path),
            metadata={
                (
                    "source_corpus_sha256"
                    if surface == "baseline_source_corpus"
                    else "cluster_baseline_training_corpus_sha256"
                ): "b" * 64
            },
        )
        return
    raise AssertionError(surface)


@pytest.mark.parametrize(
    ("surface", "match"),
    [
        ("candidate_provenance", "rejects clustered candidate provenance"),
        ("positive_parquet", "cluster-stamped artifacts"),
        ("negative_parquet", "cluster-stamped artifacts"),
        ("negative_prefixes", "cluster-stamped artifacts"),
        ("stage2_positive", "cluster-stamped artifacts"),
        ("stage2_negative", "cluster-stamped artifacts"),
        ("baseline", "cluster-stamped artifacts"),
        ("baseline_source_corpus", "cluster-stamped artifacts"),
    ],
)
def test_unclustered_source_rejects_each_cluster_stamp_artifact_kind(
    tmp_path: Path,
    surface: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("positive"),
        negative_rows=both_class_rows("negative"),
    )
    _stamp_unclustered_artifact(root, surface)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_stage_0_3_source_run(root)


def test_clustered_source_rejects_mutation_during_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    real_validate_seeds = source_validation._validate_seed_v2_contract

    def validate_then_mutate(paths: SourcePaths) -> str:
        fingerprint = real_validate_seeds(paths)
        manifest_path = paths.seed_pool_manifest()
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        return fingerprint

    monkeypatch.setattr(
        source_validation,
        "_validate_seed_v2_contract",
        validate_then_mutate,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="changed while semantic validation",
    ):
        validate_stage_0_3_source_run(root)


def test_clustered_source_rejects_mutation_before_contract_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    loaded = _load_stage_0_3_source_for_validation(root)
    candidate_path = root / "candidate_concepts.json"
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")

    with pytest.raises(AxBenchBpbAdvError, match="source-contract commit"):
        _validate_loaded_stage_0_3_source(loaded)


@pytest.mark.parametrize("clustered_at_start", [False, True])
def test_clustered_at_start_must_match_resolved_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clustered_at_start: bool,
) -> None:
    root = tmp_path / "source"
    if clustered_at_start:
        _cluster_source(root)
        resolved: object | None = None
    else:
        write_stage_0_3_source_run(
            root,
            positive_rows=both_class_rows("positive"),
            negative_rows=both_class_rows("negative"),
        )
        resolved = object()
    monkeypatch.setattr(
        source_validation,
        "_resolve_clustered_source_identity",
        lambda **_kwargs: resolved,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="cluster ownership changed",
    ):
        _load_stage_0_3_source_for_validation(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("drop_snapshot", "lost its initial artifact snapshot"),
        (
            "drop_identity",
            "unclustered source validation carried a clustered artifact snapshot",
        ),
    ],
)
def test_loaded_cluster_state_must_remain_internally_consistent(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    loaded = _load_stage_0_3_source_for_validation(root)
    if mutation == "drop_snapshot":
        loaded = dataclasses.replace(
            loaded,
            clustered_artifact_snapshot=None,
        )
    elif mutation == "drop_identity":
        loaded = dataclasses.replace(
            loaded,
            cluster_identity=None,
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _validate_loaded_stage_0_3_source(loaded)


@pytest.mark.parametrize(
    ("artifact_name", "match"),
    [
        (
            "candidate_concepts.cluster_provenance.json",
            "missing candidate provenance commit marker",
        ),
        (
            "negative_mode_prefixes.parquet",
            "missing required artifact negative_mode_prefixes.parquet",
        ),
    ],
)
def test_clustered_snapshot_requires_cluster_only_artifacts(
    tmp_path: Path,
    artifact_name: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    (root / artifact_name).unlink()

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_stage_0_3_source_run(root)


def _old_separate_read_candidate_pattern(
    *,
    paths: SourcePaths,
    feature_document: Mapping[str, Any],
    representative_universe: frozenset[int],
) -> str:
    """Inline the oracle's pre-G2-D4 parse-then-re-read hashing pattern."""

    candidate_path = paths.candidate_concepts()
    provenance_path = paths.candidate_cluster_provenance()
    candidate_rows = json.loads(candidate_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(candidate_rows, list):
        raise AxBenchBpbAdvError("clustered candidate_concepts.json must be a list")
    source_validation._require_representative_concepts(
        rows=candidate_rows,
        representative_universe=representative_universe,
        surface="candidate_concepts.json",
    )
    if not isinstance(provenance, Mapping):
        raise AxBenchBpbAdvError("clustered candidate provenance is malformed")
    if provenance["candidate_concepts_sha256"] != (source_validation._hash_file(candidate_path)):
        raise AxBenchBpbAdvError(
            "clustered candidate provenance does not match candidate JSON bytes"
        )
    source_validation.validate_cluster_candidate_provenance_semantics(
        feature_contract=feature_document,
        candidate_rows=candidate_rows,
        provenance=provenance,
    )
    return source_validation._hash_file(provenance_path)


def _scheduled_hash_swap(
    *,
    target: Path,
    swapped_bytes: bytes,
    real_hash: Any,
    calls: list[Path],
) -> Any:
    def hash_with_a_b_a_schedule(path: Path) -> str:
        if path != target:
            return real_hash(path)
        calls.append(path)
        original = target.read_bytes()
        target.write_bytes(swapped_bytes)
        try:
            return real_hash(target)
        finally:
            target.write_bytes(original)

    return hash_with_a_b_a_schedule


def test_d4_candidate_fused_read_rejects_a_b_a_old_pattern_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    paths = SourcePaths(root)
    document = source_validation.load_feature_contract_sidecar(paths.feature_contract())
    universe = source_validation._representative_universe_from_document(document)
    candidate_path = paths.candidate_concepts()
    candidate_a = candidate_path.read_bytes()
    candidate_b_rows = json.loads(candidate_a)
    candidate_b_rows[0]["description"] = "mutated-but-semantically-valid"
    candidate_b = json.dumps(
        candidate_b_rows,
        sort_keys=True,
    ).encode("utf-8")
    provenance_path = paths.candidate_cluster_provenance()
    provenance = json.loads(provenance_path.read_bytes())
    provenance["candidate_concepts_sha256"] = _sha256_bytes(candidate_b)
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True),
        encoding="utf-8",
    )
    real_hash = source_validation._hash_file
    calls: list[Path] = []
    monkeypatch.setattr(
        source_validation,
        "_hash_file",
        _scheduled_hash_swap(
            target=candidate_path,
            swapped_bytes=candidate_b,
            real_hash=real_hash,
            calls=calls,
        ),
    )

    _old_separate_read_candidate_pattern(
        paths=paths,
        feature_document=document,
        representative_universe=universe,
    )
    assert calls == [candidate_path]
    assert candidate_path.read_bytes() == candidate_a

    with pytest.raises(
        AxBenchBpbAdvError,
        match="does not match candidate JSON bytes",
    ):
        source_validation._load_and_validate_candidate_provenance(
            paths=paths,
            feature_document=document,
            representative_universe=universe,
        )


def test_d4_provenance_fused_read_rejects_a_b_a_old_pattern_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    paths = SourcePaths(root)
    document = source_validation.load_feature_contract_sidecar(paths.feature_contract())
    universe = source_validation._representative_universe_from_document(document)
    provenance_path = paths.candidate_cluster_provenance()
    provenance_a = provenance_path.read_bytes()
    provenance_b = provenance_a + b" "
    expected_b_sha = _sha256_bytes(provenance_b)
    _write_parquet_with_metadata(
        paths.negative_mode_prefixes(),
        {"candidate_cluster_provenance_sha256": expected_b_sha},
    )
    real_hash = source_validation._hash_file
    calls: list[Path] = []
    monkeypatch.setattr(
        source_validation,
        "_hash_file",
        _scheduled_hash_swap(
            target=provenance_path,
            swapped_bytes=provenance_b,
            real_hash=real_hash,
            calls=calls,
        ),
    )

    old_digest = _old_separate_read_candidate_pattern(
        paths=paths,
        feature_document=document,
        representative_universe=universe,
    )
    assert old_digest == expected_b_sha
    assert calls == [provenance_path]
    assert provenance_path.read_bytes() == provenance_a

    concepts = json.loads(paths.concepts_json().read_bytes())
    with pytest.raises(
        AxBenchBpbAdvError,
        match="does not match candidate provenance bytes",
    ):
        source_validation._resolve_clustered_source_identity(
            paths=paths,
            concepts=concepts,
            positive_table=pq.read_table(paths.synth_training_positive()),
            negative_table=pq.read_table(paths.synth_training_negative()),
        )


def test_d4_candidate_and_provenance_honest_path_accepts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    paths = SourcePaths(root)
    document = source_validation.load_feature_contract_sidecar(paths.feature_contract())
    universe = source_validation._representative_universe_from_document(document)

    old_digest = _old_separate_read_candidate_pattern(
        paths=paths,
        feature_document=document,
        representative_universe=universe,
    )
    fused_digest = source_validation._load_and_validate_candidate_provenance(
        paths=paths,
        feature_document=document,
        representative_universe=universe,
    )

    assert old_digest == fused_digest == _sha256(paths.candidate_cluster_provenance())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("candidate_unreadable", "candidate artifacts are unreadable"),
        ("provenance_unreadable", "candidate artifacts are unreadable"),
        ("candidate_not_list", "candidate_concepts.json must be a list"),
        ("candidate_nonrepresentative", "non-representative feature IDs"),
        ("provenance_partial", "malformed or partial fields"),
        ("provenance_root_mismatch", "disagrees with the root feature contract"),
        ("provenance_stale_counts", "counts disagree with contract semantics"),
    ],
)
def test_candidate_provenance_flow_rejects_each_contract_boundary(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    candidate_path = root / "candidate_concepts.json"
    provenance_path = root / "candidate_concepts.cluster_provenance.json"
    candidate: object = json.loads(candidate_path.read_bytes())
    provenance = json.loads(provenance_path.read_bytes())

    if mutation == "candidate_unreadable":
        candidate_path.write_bytes(b"{")
    elif mutation == "provenance_unreadable":
        provenance_path.write_bytes(b"{")
    elif mutation == "candidate_not_list":
        candidate = {}
        candidate_path.write_text(
            json.dumps(candidate, sort_keys=True),
            encoding="utf-8",
        )
        provenance["candidate_concepts_sha256"] = _sha256(candidate_path)
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "candidate_nonrepresentative":
        assert isinstance(candidate, list)
        assert isinstance(candidate[0], dict)
        candidate_rows = cast(list[dict[str, Any]], candidate)
        candidate_rows[0]["feature_index"] = 2
        candidate_path.write_text(
            json.dumps(candidate, sort_keys=True),
            encoding="utf-8",
        )
        provenance["candidate_concepts_sha256"] = _sha256(candidate_path)
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "provenance_partial":
        provenance.pop("counts")
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "provenance_root_mismatch":
        provenance["schema_version"] = 2
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "provenance_stale_counts":
        provenance["counts"]["raw_source_ids"] += 1
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True),
            encoding="utf-8",
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_stage_0_3_source_run(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("feature_contract_invalid", "invalid clustered source feature contract"),
        ("concept_nonrepresentative", "concepts.json contains non-representative"),
        ("stage0c_invalid", "could not read negative_mode_prefixes.parquet metadata"),
        ("stage0c_wrong_fingerprint", "fingerprint mismatch"),
        (
            "stage0c_wrong_provenance",
            "does not match candidate provenance bytes",
        ),
        ("synth_wrong_fingerprint", "fingerprint mismatch"),
        ("stage2_invalid", "Stage 2 artifact is invalid"),
        ("stage2_wrong_fingerprint", "fingerprint mismatch"),
        ("stage2_nonrepresentative", "non-representative IDs"),
        ("stage2_nonintegral", "must be integral"),
        ("stage2_missing_tensor", "lacks 'canonical_feature_id'"),
        ("synth_stale_qc", "stale clustered QC target semantics"),
        ("unsupported_baseline", "unknown top-level entry"),
        ("baseline_invalid", "not valid safetensors"),
        ("baseline_wrong_fingerprint", "fingerprint mismatch"),
        ("baseline_wrong_corpus", "exact synth corpus"),
        ("baseline_wrong_concept", "identity mismatch"),
        ("baseline_invalid_layer", "noncanonical layer"),
        ("baseline_nonfinite", "NaN or Inf"),
        ("baseline_nonfloating", "floating point"),
        ("baseline_empty", "non-empty 1D"),
        (
            "baseline_malformed_corpus",
            "malformed clustered baseline-training corpus provenance",
        ),
    ],
)
def test_clustered_semantic_artifact_failures_raise_at_their_owner(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(root)
    stage2_path = root / "concept_features.safetensors"
    baseline_path = (
        root / "baselines" / "diffmean" / "decoder_only" / "positive" / "concept_0001.safetensors"
    )
    if mutation == "feature_contract_invalid":
        (root / "feature_contract.json").write_bytes(b"{")
    elif mutation == "concept_nonrepresentative":
        concepts_path = SourcePaths(root).concepts_json()
        concepts = json.loads(concepts_path.read_bytes())
        concepts[0]["feature_index"] = 2
        concepts_path.write_text(
            json.dumps(concepts, sort_keys=True),
            encoding="utf-8",
        )
    elif mutation == "stage0c_invalid":
        (root / "negative_mode_prefixes.parquet").write_bytes(b"invalid")
    elif mutation == "stage0c_wrong_fingerprint":
        _write_parquet_with_metadata(
            root / "negative_mode_prefixes.parquet",
            {"feature_contract_fingerprint": "f" * 64},
        )
    elif mutation == "stage0c_wrong_provenance":
        _write_parquet_with_metadata(
            root / "negative_mode_prefixes.parquet",
            {"candidate_cluster_provenance_sha256": "f" * 64},
        )
    elif mutation == "synth_wrong_fingerprint":
        _write_parquet_with_metadata(
            root / "synth_training_positive.parquet",
            {"feature_contract_fingerprint": "f" * 64},
        )
    elif mutation == "stage2_invalid":
        stage2_path.write_bytes(b"invalid")
    elif mutation == "stage2_wrong_fingerprint":
        save_file(
            {
                "feature_ids": t.tensor([[1]], dtype=t.int32),
                "canonical_feature_id": t.tensor([1], dtype=t.int32),
            },
            str(stage2_path),
            metadata={"feature_contract_fingerprint": "f" * 64},
        )
    elif mutation == "stage2_nonrepresentative":
        save_file(
            {
                "feature_ids": t.tensor([[2]], dtype=t.int32),
                "canonical_feature_id": t.tensor([1], dtype=t.int32),
            },
            str(stage2_path),
            metadata={"feature_contract_fingerprint": fingerprint},
        )
    elif mutation == "stage2_nonintegral":
        save_file(
            {
                "feature_ids": t.tensor([[1.0]], dtype=t.float32),
                "canonical_feature_id": t.tensor([1], dtype=t.int32),
            },
            str(stage2_path),
            metadata={"feature_contract_fingerprint": fingerprint},
        )
    elif mutation == "stage2_missing_tensor":
        save_file(
            {"feature_ids": t.tensor([[1]], dtype=t.int32)},
            str(stage2_path),
            metadata={"feature_contract_fingerprint": fingerprint},
        )
    elif mutation == "synth_stale_qc":
        _write_parquet_with_metadata(
            root / "synth_training_positive.parquet",
            {"cluster_qc_target_algorithm": "stale"},
        )
    elif mutation == "unsupported_baseline":
        (root / "baselines" / "unknown").mkdir()
    elif mutation == "baseline_invalid":
        baseline_path.write_bytes(b"invalid")
    elif mutation == "baseline_wrong_fingerprint":
        save_file(
            {"direction": t.ones(4)},
            str(baseline_path),
            metadata={
                "artifact_kind": "diffmean_direction_v1",
                "concept_id": "1",
                "feature_contract_fingerprint": "f" * 64,
                "layer": "6",
                "source_corpus_sha256": _sha256(root / "synth_training_positive.parquet"),
                "target_model": "decoder_only",
                "mode": "positive",
            },
        )
    elif mutation == "baseline_wrong_corpus":
        save_file(
            {"direction": t.ones(4)},
            str(baseline_path),
            metadata={
                "artifact_kind": "diffmean_direction_v1",
                "concept_id": "1",
                "feature_contract_fingerprint": fingerprint,
                "layer": "6",
                "source_corpus_sha256": "f" * 64,
                "target_model": "decoder_only",
                "mode": "positive",
            },
        )
    elif mutation in {
        "baseline_wrong_concept",
        "baseline_invalid_layer",
        "baseline_nonfinite",
        "baseline_nonfloating",
        "baseline_empty",
    }:
        direction = {
            "baseline_nonfinite": t.tensor([1.0, float("nan")]),
            "baseline_nonfloating": t.ones(4, dtype=t.int64),
            "baseline_empty": t.empty(0),
        }.get(mutation, t.ones(4))
        save_file(
            {"direction": direction},
            str(baseline_path),
            metadata={
                "artifact_kind": "diffmean_direction_v1",
                "concept_id": "999" if mutation == "baseline_wrong_concept" else "1",
                "feature_contract_fingerprint": fingerprint,
                "layer": "not-an-int" if mutation == "baseline_invalid_layer" else "6",
                "source_corpus_sha256": _sha256(root / "synth_training_positive.parquet"),
                "target_model": "decoder_only",
                "mode": "positive",
            },
        )
    elif mutation == "baseline_malformed_corpus":
        _write_parquet_with_metadata(
            root / "synth_training_positive.parquet",
            {"cluster_baseline_training_corpus_sha256": "not-a-sha256"},
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        validate_stage_0_3_source_run(root)


def test_baseline_training_corpus_stable_hash_failure_is_branded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    _cluster_source(root)
    path = root / "synth_training_positive.parquet"
    table = pq.read_table(path)

    def fail_stable_hash(_path: Path) -> None:
        raise ClusterFeatureContractError("changed")

    monkeypatch.setattr(
        source_validation,
        "stream_hash_file_stable",
        fail_stable_hash,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="could not stably hash clustered baseline-training corpus",
    ):
        source_validation._baseline_training_corpus_sha256(
            table=table,
            path=path,
        )


def test_clustered_cleaning_preserves_baseline_training_corpus_and_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    fingerprint = _cluster_source(
        root,
        positive_rows=(
            both_class_rows("kept")
            + one_sided_rows(
                "dropped",
                content_class="concept_rich",
            )
        ),
    )
    original_hashes = {
        "positive": _sha256(root / "synth_training_positive.parquet"),
        "negative": _sha256(root / "synth_training_negative.parquet"),
    }

    result = clean_axbench_source_run(
        root,
        output_dir=tmp_path / "cleaned",
    )

    assert result.positive.rows_in - result.positive.rows_out == 1
    assert result.validation.contract.schema_version == (CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION)
    assert result.validation.contract.feature_contract_fingerprint == fingerprint
    assert result.validation.contract.baseline_training_corpus_sha256 == original_hashes
    positive_metadata = dict(
        pq.read_schema(result.output_dir / "synth_training_positive.parquet").metadata or {}
    )
    assert positive_metadata[_BASELINE_TRAINING_CORPUS_METADATA_KEY] == (
        original_hashes["positive"].encode("ascii")
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["self_verification"]["feature_contract_fingerprint"] == (fingerprint)
    assert manifest["self_verification"]["baseline_training_corpus_sha256"] == original_hashes


def test_unclustered_cleaning_manifest_retains_legacy_self_verification_shape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=(
            both_class_rows("kept")
            + one_sided_rows(
                "dropped",
                content_class="concept_rich",
            )
        ),
        negative_rows=both_class_rows("negative"),
    )

    result = clean_axbench_source_run(
        root,
        output_dir=tmp_path / "cleaned",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert "feature_contract_fingerprint" not in manifest["self_verification"]
    assert "baseline_training_corpus_sha256" not in manifest["self_verification"]


def test_d10_semantic_provenance_read_is_bound_to_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G2-D10: the two provenance digests must come from the SAME bytes.

    G2-D4 fused the read inside the leaf, but at FLOW level two digests still
    came from independent reads -- ``artifact_hashes`` from the snapshot and
    ``candidate_provenance_sha256`` from the semantic load -- with nothing
    requiring agreement. A coherent transient B present only during the
    semantic reads, with A restored before both snapshots, therefore committed
    a MIXED contract. The snapshot equality checks cannot catch it: they prove
    only that the filesystem is A again, not that the semantic read saw A.

    Modelled deterministically by making the semantic load report a digest for
    bytes that are not the ones on disk -- exactly the state the transient-B
    schedule produces.
    """

    root = tmp_path / "src"
    _cluster_source(root)

    transient_b = "b" * 64
    real_leaf = source_validation._load_and_validate_candidate_provenance
    real_meta = source_validation._require_cluster_parquet_metadata

    def semantic_read_saw_transient_b(**kwargs: Any) -> str:
        real_leaf(**kwargs)
        return transient_b

    def stage0c_agrees_with_transient_b(path: Path, **kwargs: Any) -> Any:
        metadata = dict(real_meta(path, **kwargs))
        # A COHERENT transient B: Stage 0c carries B's provenance digest too,
        # so every internal consistency check inside the semantic load still
        # passes. Only the cross-read binding to the snapshot can catch it.
        metadata[b"candidate_cluster_provenance_sha256"] = transient_b.encode("ascii")
        return metadata

    monkeypatch.setattr(
        source_validation,
        "_load_and_validate_candidate_provenance",
        semantic_read_saw_transient_b,
    )
    monkeypatch.setattr(
        source_validation,
        "_require_cluster_parquet_metadata",
        stage0c_agrees_with_transient_b,
    )

    with pytest.raises(AxBenchBpbAdvError, match="does not match the snapshotted artifact"):
        validate_stage_0_3_source_run(root)


def test_d10_honest_path_still_validates(tmp_path: Path) -> None:
    """Guard against over-correction: the binding must not reject honest input."""

    root = tmp_path / "src"
    _cluster_source(root)

    validation = validate_stage_0_3_source_run(root)

    assert validation.contract.candidate_provenance_sha256 is not None
    assert (
        validation.contract.artifact_hashes["candidate_concepts.cluster_provenance.json"]
        == validation.contract.candidate_provenance_sha256
    )


@pytest.mark.parametrize(
    ("tensor_name", "bad_value", "rejected"),
    [
        ("feature_ids", -1, True),
        ("feature_ids", -2, True),
        ("feature_ids", -999, True),
        ("canonical_feature_id", -1, True),
        ("canonical_feature_id", -2, True),
    ],
)
def test_d9_stage2_negative_ids(
    tmp_path: Path, tensor_name: str, bad_value: int, rejected: bool
) -> None:
    """G2-D9: negatives were filtered OUT of the offender set, not rejected.

    ``feature_ids=[-2]`` and ``canonical_feature_id=[-999]`` validated cleanly
    while the downstream consumer rejects them
    (axbench_base/intervention_dispatch.py:192) -- source validation must not
    certify an artifact its own consumer refuses. ``-1`` is the producer's only
    legal negative. It is valid in trailing ``feature_ids`` unused slots and in
    ``canonical_feature_id`` only for filtered-out concept-ID gap rows; this
    fixture mutates the active row's required slot 0, so ``-1`` remains invalid
    here for either tensor.
    """

    root = tmp_path / "src"
    fingerprint = _cluster_source(root)
    path = SourcePaths(root).concept_features()
    tensors = {
        "feature_ids": t.tensor([[8]], dtype=t.int64),
        "canonical_feature_id": t.tensor([8], dtype=t.int64),
    }
    tensors[tensor_name] = (
        t.tensor([[bad_value]], dtype=t.int64)
        if tensor_name == "feature_ids"
        else t.tensor([bad_value], dtype=t.int64)
    )
    save_file(tensors, str(path), metadata={"feature_contract_fingerprint": fingerprint})

    if rejected:
        expected_error = (
            "canonical_feature_id=-1"
            if tensor_name == "canonical_feature_id" and bad_value == -1
            else "exact canonical feature row"
            if tensor_name == "feature_ids" and bad_value == -1
            else "non-representative IDs"
        )
        with pytest.raises(AxBenchBpbAdvError, match=expected_error):
            source_validation._validate_stage2_artifact(
                path,
                expected_fingerprint=fingerprint,
                representative_universe=frozenset({8}),
                expected_concept_features={0: 8},
            )
    else:
        source_validation._validate_stage2_artifact(
            path,
            expected_fingerprint=fingerprint,
            representative_universe=frozenset({8}),
            expected_concept_features={0: 8},
        )


def test_stage2_sparse_filtered_population_requires_exact_gap_sentinels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "src"
    fingerprint = _cluster_source(root)
    path = SourcePaths(root).concept_features()
    valid = {
        "feature_ids": t.tensor([[8, -1], [-1, -1], [8, -1]], dtype=t.int32),
        "canonical_feature_id": t.tensor([8, -1, 8], dtype=t.int32),
    }
    save_file(
        valid,
        str(path),
        metadata={"feature_contract_fingerprint": fingerprint},
    )

    source_validation._validate_stage2_artifact(
        path,
        expected_fingerprint=fingerprint,
        representative_universe=frozenset({8}),
        expected_concept_features={0: 8, 2: 8},
    )

    invalid = {
        **valid,
        "feature_ids": valid["feature_ids"].clone(),
    }
    invalid["feature_ids"][1, 0] = 8
    save_file(
        invalid,
        str(path),
        metadata={"feature_contract_fingerprint": fingerprint},
    )
    with pytest.raises(AxBenchBpbAdvError, match="not all-sentinel"):
        source_validation._validate_stage2_artifact(
            path,
            expected_fingerprint=fingerprint,
            representative_universe=frozenset({8}),
            expected_concept_features={0: 8, 2: 8},
        )


@pytest.mark.parametrize(
    "bad_active_row",
    [
        [-1, -1],
        [8, 8],
    ],
)
def test_stage2_active_population_requires_exact_canonical_feature_row(
    tmp_path: Path,
    bad_active_row: list[int],
) -> None:
    root = tmp_path / "src"
    fingerprint = _cluster_source(root)
    path = SourcePaths(root).concept_features()
    save_file(
        {
            "feature_ids": t.tensor([bad_active_row], dtype=t.int32),
            "canonical_feature_id": t.tensor([8], dtype=t.int32),
        },
        str(path),
        metadata={"feature_contract_fingerprint": fingerprint},
    )

    with pytest.raises(AxBenchBpbAdvError, match="exact canonical feature row"):
        source_validation._validate_stage2_artifact(
            path,
            expected_fingerprint=fingerprint,
            representative_universe=frozenset({8}),
            expected_concept_features={0: 8},
        )


def test_stage2_rejects_zero_width_feature_rows(tmp_path: Path) -> None:
    root = tmp_path / "src"
    fingerprint = _cluster_source(root)
    path = SourcePaths(root).concept_features()
    save_file(
        {
            "feature_ids": t.empty((1, 0), dtype=t.int32),
            "canonical_feature_id": t.tensor([8], dtype=t.int32),
        },
        str(path),
        metadata={"feature_contract_fingerprint": fingerprint},
    )

    with pytest.raises(AxBenchBpbAdvError, match="n_rank_store>=1"):
        source_validation._validate_stage2_artifact(
            path,
            expected_fingerprint=fingerprint,
            representative_universe=frozenset({8}),
            expected_concept_features={0: 8},
        )
