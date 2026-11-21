from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import steering.eval.axbench_bpb_adv.artifact_identity as artifact_identity
from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredCandidateSnapshot,
    assert_table_cluster_identity,
    bind_clustered_table,
    clustered_artifact_identity,
    load_clustered_candidate_snapshot,
    require_clustered_candidate_snapshot,
    write_clustered_table_atomic,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    ROOFLINE_PRECOMPUTED_BAG_FAMILIES,
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
    contract_hash,
)
from steering.eval.cluster_feature_contract import FEATURE_CONTRACT_SCHEMA_VERSION
from tests.axbench_bpb_adv.artifact_helpers import artifact_identity_metadata

_FP = "a" * 64


def _source_contract_payload(**updates: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "seed_fingerprint": "b" * 64,
        "schema_version": CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
        "source_run_dir": "/test/source",
        "direction_training_layer": 6,
        "feature_contract_schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "feature_contract_fingerprint": _FP,
        "candidate_provenance_sha256": "c" * 64,
        "stage2_feature_contract_fingerprints": {
            "concept_features.safetensors": _FP,
            "concept_features_neg.safetensors": _FP,
        },
        "baseline_training_corpus_sha256": {
            "positive": "d" * 64,
            "negative": "e" * 64,
        },
    }
    values.update(updates)
    return BpbAdvSourceContract(**values).to_json_dict()


def _paths(
    tmp_path: Path, *, payload_extra: dict[str, Any] | None = None
) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=str(tmp_path), run_tag="run")
    paths.root.mkdir(parents=True)
    payload = _source_contract_payload(**(payload_extra or {}))
    paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    return paths


def _contract(fingerprint: str = _FP) -> Any:
    return SimpleNamespace(feature_contract_fingerprint=fingerprint)


def _candidate_row(**updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in CANDIDATES_SCHEMA:
        if pa.types.is_string(field.type):
            row[field.name] = ""
        elif pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        elif pa.types.is_list(field.type):
            row[field.name] = []
        else:  # pragma: no cover - schema growth must make this helper explicit
            raise AssertionError(f"unsupported candidate test field {field}")
    row.update(
        {
            "candidate_id": "candidate-1",
            "attack_row_id": "attack-1",
            "core_prefix_id": "prefix-1",
            "concept_id": 7,
            "mode": "positive",
            "method_name": "ours_gcg_roofline_encdec",
            "coefficient_str": "0.25",
            "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
        }
    )
    row.update(updates)
    return row


def _write_candidate_bag_dependencies(
    paths: AxBenchBpbAdvPaths,
) -> tuple[Path, Path]:
    bag = paths.ours_gcg_bag(
        7,
        "positive",
        0.25,
        "roofline",
        prefix_id="prefix-1",
    )
    scored = paths.ours_gcg_scored_prefixes(7, "positive", "roofline")
    bag.parent.mkdir(parents=True, exist_ok=True)
    scored.parent.mkdir(parents=True, exist_ok=True)
    bag.write_bytes(b"bag-v1")
    scored.write_bytes(b"scored-prefixes-v1")
    return bag, scored


def _candidates_table(
    paths: AxBenchBpbAdvPaths,
    **row_updates: Any,
) -> pa.Table:
    """A real CANDIDATES_SCHEMA table with its bag dependencies on disk.

    Candidates-family binding hashes the bag artifacts a candidate references,
    and the schema check that used to fail open now raises. Fixtures below
    previously passed a throwaway ``pa.table({"x": [1]})`` while testing
    unrelated identity properties; production always passes a real
    CANDIDATES_SCHEMA table, so this makes them faithful rather than merely
    green.
    """
    _write_candidate_bag_dependencies(paths)
    return pa.Table.from_pylist(
        [_candidate_row(**row_updates)], schema=CANDIDATES_SCHEMA
    )


def _write_dependencies_for_row(
    paths: AxBenchBpbAdvPaths,
    row: dict[str, Any],
    *,
    bag_bytes: bytes = b"bag",
    scored_bytes: bytes = b"scored-prefixes",
) -> tuple[Path, Path]:
    family = precomputed_bag_method_family(str(row["method_name"]))
    if family is None:
        raise AssertionError("test row must name a precomputed-bag family")
    if family in {
        PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
        PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
    }:
        reg_lambda = 0.0
    else:
        reg_lambda = float(row["coefficient_str"])
    prefix_id = (
        str(row["core_prefix_id"])
        if family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES
        else None
    )
    bag = paths.ours_gcg_bag(
        int(row["concept_id"]),
        str(row["mode"]),
        reg_lambda,
        family,
        prefix_id=prefix_id,
    )
    scored = paths.ours_gcg_scored_prefixes(
        int(row["concept_id"]),
        str(row["mode"]),
        family,
    )
    bag.parent.mkdir(parents=True, exist_ok=True)
    scored.parent.mkdir(parents=True, exist_ok=True)
    bag.write_bytes(bag_bytes)
    scored.write_bytes(scored_bytes)
    return bag, scored


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _write_candidate_snapshot(
    paths: AxBenchBpbAdvPaths,
    **row_updates: Any,
) -> ClusteredCandidateSnapshot:
    table = bind_clustered_table(
        _candidates_table(paths, **row_updates),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    snapshot = write_clustered_table_atomic(
        table,
        paths.candidates_path(),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    assert type(snapshot) is ClusteredCandidateSnapshot
    return snapshot


def test_unclustered_bind_is_exact_noop_and_reader_rejects_stamp(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    table = _candidates_table(paths)
    assert (
        bind_clustered_table(
            table,
            paths=paths,
            contract=None,
            artifact_family="candidates",
        )
        is table
    )
    assert_table_cluster_identity(
        table,
        paths=paths,
        contract=None,
        artifact_family="candidates",
    )
    stamped = bind_clustered_table(
        table,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered reader rejects"):
        assert_table_cluster_identity(
            stamped,
            paths=paths,
            contract=None,
            artifact_family="candidates",
        )


def test_clustered_table_binds_source_and_feature_contract(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    table = bind_clustered_table(
        pa.table({"x": [1]}),
        paths=paths,
        contract=_contract(),
        artifact_family="scores",
    )
    metadata = artifact_identity_metadata(table)
    assert metadata == {
        "clustered_artifact_family": "scores",
        "feature_contract_fingerprint": _FP,
        "source_contract_hash": json.loads(
            paths.source_contract().read_text(encoding="utf-8")
        )["contract_hash"],
    }
    assert_table_cluster_identity(
        table,
        paths=paths,
        contract=_contract(),
        artifact_family="scores",
    )
    assert (
        clustered_artifact_identity(
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
        )["clustered_artifact_family"]
        == "scores"
    )


def test_clustered_candidates_bind_exact_bag_and_scored_prefix_bytes(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    bag_path, scored_path = _write_candidate_bag_dependencies(paths)
    table = bind_clustered_table(
        pa.Table.from_pylist([_candidate_row()], schema=CANDIDATES_SCHEMA),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    metadata = artifact_identity_metadata(table)
    assert len(metadata["precomputed_bag_set_sha256"]) == 64
    assert_table_cluster_identity(
        table,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )

    bag_path.write_bytes(b"bag-v2")
    with pytest.raises(AxBenchBpbAdvError, match="absent or stale"):
        assert_table_cluster_identity(
            table,
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )

    # Restoring the bag but changing the scored-prefix commit marker is an
    # independent stale-derive mutation and must be caught too.
    bag_path.write_bytes(b"bag-v1")
    scored_path.write_bytes(b"scored-prefixes-v2")
    with pytest.raises(AxBenchBpbAdvError, match="absent or stale"):
        assert_table_cluster_identity(
            table,
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )


@pytest.mark.parametrize(
    "artifact_family",
    [
        "universal_triggers",
        "class_margin_scores",
        "roundtrip_scores",
        "general_val_bpb",
        "samples",
    ],
)
def test_clustered_downstream_binds_exact_validated_candidate_artifact(
    tmp_path: Path,
    artifact_family: str,
) -> None:
    paths = _paths(tmp_path)
    snapshot = _write_candidate_snapshot(paths)
    downstream = bind_clustered_table(
        pa.table({"score": [1.0]}),
        paths=paths,
        contract=_contract(),
        artifact_family=artifact_family,
        candidate_snapshot=snapshot,
    )
    metadata = artifact_identity_metadata(downstream)
    assert metadata["upstream_candidates_sha256"] == snapshot.sha256
    assert_table_cluster_identity(
        downstream,
        paths=paths,
        contract=_contract(),
        artifact_family=artifact_family,
        candidate_snapshot=snapshot,
    )

    replacement = bind_clustered_table(
        _candidates_table(paths, candidate_id="candidate-2"),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    pq.write_table(replacement, paths.candidates_path())
    assert_table_cluster_identity(
        downstream,
        paths=paths,
        contract=_contract(),
        artifact_family=artifact_family,
        candidate_snapshot=snapshot,
    )


def test_candidate_writer_returns_owner_of_exact_installed_parquet(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    snapshot = _write_candidate_snapshot(paths)
    installed = paths.candidates_path().read_bytes()

    assert snapshot.payload == installed
    assert snapshot.sha256 == hashlib.sha256(installed).hexdigest()
    assert snapshot.table.schema.equals(CANDIDATES_SCHEMA)
    assert pq.read_table(pa.BufferReader(installed)).equals(
        snapshot.table,
        check_metadata=True,
    )
    assert (
        require_clustered_candidate_snapshot(
            snapshot,
            paths=paths,
            contract=_contract(),
            source_identity=None,
            surface="writer-return test",
            table=snapshot.table,
        )
        is snapshot
    )


def test_candidate_reader_owns_one_payload_across_path_replacement(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_candidate_snapshot(paths)
    snapshot = load_clustered_candidate_snapshot(
        paths.candidates_path(),
        paths=paths,
        contract=_contract(),
    )
    first_sha256 = snapshot.sha256
    first_rows = snapshot.table.to_pylist()

    replacement_table = bind_clustered_table(
        _candidates_table(paths, candidate_id="candidate-2"),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    pq.write_table(replacement_table, paths.candidates_path())
    replacement_bytes = paths.candidates_path().read_bytes()
    assert hashlib.sha256(replacement_bytes).hexdigest() != first_sha256

    downstream = bind_clustered_table(
        pa.table({"score": [1.0]}),
        paths=paths,
        contract=_contract(),
        artifact_family="class_margin_scores",
        candidate_snapshot=snapshot,
    )

    assert snapshot.table.to_pylist() == first_rows
    assert hashlib.sha256(snapshot.payload).hexdigest() == first_sha256
    assert artifact_identity_metadata(downstream)["upstream_candidates_sha256"] == (
        first_sha256
    )
    assert_table_cluster_identity(
        downstream,
        paths=paths,
        contract=_contract(),
        artifact_family="class_margin_scores",
        candidate_snapshot=snapshot,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement_value"),
    [
        ("sha256", "b" * 64),
        ("feature_contract_fingerprint", "b" * 64),
        ("source_contract_sha256", "b" * 64),
        ("payload", b"different-payload"),
    ],
)
def test_candidate_snapshot_rejects_dataclass_field_replacement(
    tmp_path: Path,
    field_name: str,
    replacement_value: object,
) -> None:
    snapshot = _write_candidate_snapshot(_paths(tmp_path))

    with pytest.raises(
        AxBenchBpbAdvError,
        match="fields changed after validation",
    ):
        replace(snapshot, **{field_name: replacement_value})


def test_candidate_snapshot_rejects_equal_but_different_table_object(
    tmp_path: Path,
) -> None:
    snapshot = _write_candidate_snapshot(_paths(tmp_path))
    equal_table = pa.concat_tables([snapshot.table])
    assert equal_table.equals(snapshot.table, check_metadata=True)
    assert equal_table is not snapshot.table

    with pytest.raises(
        AxBenchBpbAdvError,
        match="fields changed after validation",
    ):
        replace(snapshot, table=equal_table)


def _construct_candidate_snapshot(
    *,
    table: pa.Table,
    payload: Any,
    sha256: str,
    feature_fingerprint: str = _FP,
    source_sha256: str,
) -> ClusteredCandidateSnapshot:
    return ClusteredCandidateSnapshot(
        table=table,
        payload=payload,
        sha256=sha256,
        feature_contract_fingerprint=feature_fingerprint,
        source_contract_sha256=source_sha256,
        _token=artifact_identity._CANDIDATE_SNAPSHOT_TOKEN,
        _seal=(
            id(table),
            id(payload),
            sha256,
            feature_fingerprint,
            source_sha256,
        ),
    )


def test_candidate_snapshot_rejects_direct_construction(tmp_path: Path) -> None:
    snapshot = _write_candidate_snapshot(_paths(tmp_path))
    with pytest.raises(TypeError, match="must be built"):
        ClusteredCandidateSnapshot(
            table=snapshot.table,
            payload=snapshot.payload,
            sha256=snapshot.sha256,
            feature_contract_fingerprint=snapshot.feature_contract_fingerprint,
            source_contract_sha256=snapshot.source_contract_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("payload_type", "malformed exact-byte identity"),
        ("payload_digest", "does not match its digest"),
        ("coordinates", "malformed contract coordinates"),
        ("unreadable", "not readable Parquet"),
        ("table", "table does not match"),
    ],
)
def test_candidate_snapshot_revalidates_every_sealed_constituent(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    snapshot = _write_candidate_snapshot(_paths(tmp_path))
    table = snapshot.table
    payload: Any = snapshot.payload
    sha256 = snapshot.sha256
    feature_fingerprint = snapshot.feature_contract_fingerprint
    source_sha256 = snapshot.source_contract_sha256
    if mutation == "payload_type":
        payload = bytearray(payload)
    elif mutation == "payload_digest":
        sha256 = "b" * 64
    elif mutation == "coordinates":
        feature_fingerprint = "not-a-digest"
    elif mutation == "unreadable":
        payload = b"not-parquet"
        sha256 = hashlib.sha256(payload).hexdigest()
    elif mutation == "table":
        table = table.set_column(
            table.schema.get_field_index("candidate_id"),
            "candidate_id",
            pa.array(["different-candidate"], type=pa.string()),
        )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _construct_candidate_snapshot(
            table=table,
            payload=payload,
            sha256=sha256,
            feature_fingerprint=feature_fingerprint,
            source_sha256=source_sha256,
        )


def test_candidate_snapshot_reader_rejects_noncanonical_and_malformed_inputs(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    snapshot = _write_candidate_snapshot(paths)
    alias = paths.candidates_path().with_name("alias.parquet")
    alias.write_bytes(snapshot.payload)

    with pytest.raises(AxBenchBpbAdvError, match="canonical candidates path"):
        load_clustered_candidate_snapshot(
            alias,
            paths=paths,
            contract=_contract(),
        )

    paths.candidates_path().write_bytes(b"not parquet")
    with pytest.raises(AxBenchBpbAdvError, match="cannot parse candidates"):
        load_clustered_candidate_snapshot(
            paths.candidates_path(),
            paths=paths,
            contract=_contract(),
        )


def test_candidate_snapshot_reader_brands_stable_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        artifact_identity,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
    )

    with pytest.raises(AxBenchBpbAdvError, match="could not stably read"):
        load_clustered_candidate_snapshot(
            paths.candidates_path(),
            paths=paths,
            contract=_contract(),
        )


@pytest.mark.parametrize("missing_payload", [True, False])
def test_candidate_snapshot_reader_rejects_invalid_retained_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_payload: bool,
) -> None:
    paths = _paths(tmp_path)
    payload = None if missing_payload else _parquet_bytes(pa.table({"wrong": [1]}))
    monkeypatch.setattr(
        artifact_identity,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: SimpleNamespace(
            payload=payload,
            sha256="a" * 64,
        ),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=(
            "lost its retained payload"
            if missing_payload
            else "requires CANDIDATES_SCHEMA"
        ),
    ):
        load_clustered_candidate_snapshot(
            paths.candidates_path(),
            paths=paths,
            contract=_contract(),
        )


def test_candidate_snapshot_builder_rejects_table_payload_disagreement(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    snapshot = _write_candidate_snapshot(paths)
    changed_table = snapshot.table.set_column(
        snapshot.table.schema.get_field_index("candidate_id"),
        "candidate_id",
        pa.array(["changed"], type=pa.string()),
    )

    with pytest.raises(AxBenchBpbAdvError, match="changed its validated table"):
        artifact_identity._build_clustered_candidate_snapshot(
            table=changed_table,
            payload=snapshot.payload,
            paths=paths,
            contract=_contract(),
            source_identity=artifact_identity.resolve_clustered_source_identity(
                paths=paths,
                contract=_contract(),
            ),
        )


def test_candidate_snapshot_is_required_only_at_clustered_consumer_boundary(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    snapshot = _write_candidate_snapshot(paths)
    table = pa.table({"score": [1.0]})

    with pytest.raises(AxBenchBpbAdvError, match="requires one exact"):
        require_clustered_candidate_snapshot(
            None,
            paths=paths,
            contract=_contract(),
            source_identity=None,
            surface="boundary test",
        )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered"):
        require_clustered_candidate_snapshot(
            snapshot,
            paths=paths,
            contract=None,
            source_identity=None,
            surface="boundary test",
        )
    with pytest.raises(AxBenchBpbAdvError, match="exact candidate table"):
        require_clustered_candidate_snapshot(
            snapshot,
            paths=paths,
            contract=_contract(),
            source_identity=None,
            surface="boundary test",
            table=pa.concat_tables([snapshot.table]),
        )
    with pytest.raises(AxBenchBpbAdvError, match="different feature contract"):
        require_clustered_candidate_snapshot(
            snapshot,
            paths=paths,
            contract=_contract("b" * 64),
            source_identity=None,
            surface="boundary test",
        )
    resolved_source = artifact_identity.resolve_clustered_source_identity(
        paths=paths,
        contract=_contract(),
    )
    with pytest.raises(AxBenchBpbAdvError, match="different source contract"):
        require_clustered_candidate_snapshot(
            snapshot,
            paths=paths,
            contract=_contract(),
            source_identity=replace(
                resolved_source,
                source_contract_sha256="b" * 64,
            ),
            surface="boundary test",
        )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered"):
        bind_clustered_table(
            table,
            paths=paths,
            contract=None,
            artifact_family="scores",
            candidate_snapshot=snapshot,
        )
    with pytest.raises(AxBenchBpbAdvError, match="valid only"):
        bind_clustered_table(
            table,
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
            candidate_snapshot=snapshot,
        )
    with pytest.raises(AxBenchBpbAdvError, match="unclustered artifact reader"):
        assert_table_cluster_identity(
            table,
            paths=paths,
            contract=None,
            artifact_family="scores",
            candidate_snapshot=snapshot,
        )


def test_candidate_dependent_identity_resolves_source_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _write_candidate_snapshot(paths)
    snapshot = load_clustered_candidate_snapshot(
        paths.candidates_path(),
        paths=paths,
        contract=_contract(),
    )
    real_resolve = artifact_identity.resolve_clustered_source_identity
    calls = 0

    def _resolve_once(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail(  # pragma: no cover - failure-only sentinel
                "one identity operation reopened source_contract.json"
            )
        return real_resolve(**kwargs)

    monkeypatch.setattr(
        artifact_identity,
        "resolve_clustered_source_identity",
        _resolve_once,
    )

    bound = bind_clustered_table(
        pa.table({"score": [1.0]}),
        paths=paths,
        contract=_contract(),
        artifact_family="samples",
        candidate_snapshot=snapshot,
    )

    assert calls == 1
    assert artifact_identity_metadata(bound)["source_contract_hash"] == (
        snapshot.source_contract_sha256
    )


def test_clustered_parquet_resume_is_exact_or_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_candidate_bag_dependencies(paths)
    table = bind_clustered_table(
        pa.Table.from_pylist([_candidate_row()], schema=CANDIDATES_SCHEMA),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    path = paths.candidates_path()
    write_clustered_table_atomic(
        table,
        path,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    original = path.read_bytes()
    write_clustered_table_atomic(
        table,
        path,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    assert path.read_bytes() == original

    changed = table.set_column(
        table.schema.get_field_index("candidate_id"),
        "candidate_id",
        pa.array(["candidate-2"], type=pa.string()),
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="content changed on resume; use a fresh run_tag",
    ):
        write_clustered_table_atomic(
            changed,
            path,
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["source", "feature", "family"])
def test_clustered_reader_rejects_each_identity_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _paths(tmp_path)
    table = bind_clustered_table(
        _candidates_table(paths),
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    contract = _contract()
    family = "candidates"
    if mutation == "source":
        payload = json.loads(paths.source_contract().read_text(encoding="utf-8"))
        payload["n_dropped"] = 1
        payload["contract_hash"] = contract_hash(payload)
        paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "feature":
        contract = _contract("b" * 64)
    else:
        family = "scores"
    with pytest.raises(AxBenchBpbAdvError, match="absent or stale|does not match"):
        assert_table_cluster_identity(
            table,
            paths=paths,
            contract=contract,
            artifact_family=family,
        )


def test_source_contract_self_hash_mutation_fails_before_binding(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    payload = json.loads(paths.source_contract().read_text(encoding="utf-8"))
    payload["n_dropped"] = 1
    paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AxBenchBpbAdvError, match="contract_hash mismatch"):
        bind_clustered_table(
            pa.table({"x": [1]}),
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
        )


def test_rebinding_reserved_cluster_identity_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    table = bind_clustered_table(
        pa.table({"x": [1]}),
        paths=paths,
        contract=_contract(),
        artifact_family="scores",
    )

    with pytest.raises(AxBenchBpbAdvError, match="refusing to overwrite reserved"):
        bind_clustered_table(
            table,
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
        )


def test_bind_without_contract_returns_same_object_without_path_reads(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="missing-run")
    table = pa.table({"x": [1]})

    assert (
        bind_clustered_table(
            table,
            paths=paths,
            contract=None,
            artifact_family="scores",
        )
        is table
    )
    assert not paths.root.exists()


def test_candidate_family_non_candidate_schema_raises(
    tmp_path: Path,
) -> None:
    """The former fail-open gap now raises. This is the inverted anchor test.

    A candidates-family table that is not exactly ``CANDIDATES_SCHEMA`` used to
    return None, so the entire precomputed-bag dependency vanished from the
    identity stamp with no diagnostic -- and because the omission recurred
    identically on write AND read, it could never later surface as a staleness
    error. Four sibling sites already treated the identical condition as fatal
    (scoring.py:107, roundtrip_scoring.py:130, general_val.py:1331,
    scripts/eval/axbench_bpb_adv_eval.py:489); this one now matches them.

    Still not reachable in production -- all five candidate-table constructors
    pass ``schema=CANDIDATES_SCHEMA`` explicitly -- so this closes a fail-open
    asymmetry rather than a live bug. It becomes a live bug the moment a
    constructor is added without the explicit schema, which is exactly what
    this guard now catches.
    """

    paths = _paths(tmp_path)
    _write_candidate_bag_dependencies(paths)

    # (a) the bare lax table -- previously a silent None.
    with pytest.raises(AxBenchBpbAdvError, match="CANDIDATES_SCHEMA"):
        artifact_identity._candidate_bag_set_sha256(
            pa.table({"x": [1]}), paths=paths
        )

    # (b) the realistic shape the Codex cross-review constructed: every real
    # candidate column, one extra schema-drift column, a genuine bag-backed
    # row. Binding previously accepted it with the bag-set key simply absent.
    drifted = pa.Table.from_pylist(
        [_candidate_row()], schema=CANDIDATES_SCHEMA
    ).append_column("unexpected_new_column", pa.array([0]))
    with pytest.raises(AxBenchBpbAdvError, match="CANDIDATES_SCHEMA"):
        bind_clustered_table(
            drifted,
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )


def test_candidate_family_exact_schema_still_stamps_the_bag_set(
    tmp_path: Path,
) -> None:
    """Over-correction guard: the legitimate shape must still be accepted.

    The fix above tightens a schema check that sits on the ONLY path producing
    the bag-set key, so a too-strict version would silently stop stamping every
    real candidates artifact. Asserting the key is PRESENT -- not merely that
    binding did not raise -- is what distinguishes the two.
    """

    paths = _paths(tmp_path)
    _write_candidate_bag_dependencies(paths)

    exact = pa.Table.from_pylist([_candidate_row()], schema=CANDIDATES_SCHEMA)
    stamped = bind_clustered_table(
        exact,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    assert "precomputed_bag_set_sha256" in artifact_identity_metadata(stamped)


def test_non_finite_bag_coefficient_raises(tmp_path: Path) -> None:
    """G1-D4: float() accepts nan/inf, so the numeric guard needed finiteness.

    Untested by the original suite despite 100% statement/branch coverage --
    line coverage is not input-domain coverage.
    """

    paths = _paths(tmp_path)
    _write_candidate_bag_dependencies(paths)

    for spelling in ("nan", "NaN", "inf", "-inf", "Infinity"):
        table = pa.Table.from_pylist(
            [_candidate_row(method_name="ours_gcg_encdec", coefficient_str=spelling)],
            schema=CANDIDATES_SCHEMA,
        )
        with pytest.raises(AxBenchBpbAdvError, match="is not finite"):
            artifact_identity._candidate_bag_set_sha256(table, paths=paths)


def test_writer_rejects_unstamped_and_stale_tables(tmp_path: Path) -> None:
    """Close the surviving mutation the Codex cross-review found.

    Deleting ``assert_table_cluster_identity`` from
    ``write_clustered_table_atomic`` previously survived the whole suite: the
    sole writer test only exercised an already-stamped table, so the mutant
    happily committed an UNSTAMPED table and nothing failed. These two negative
    cases kill that mutation directly.
    """

    paths = _paths(tmp_path)
    _write_candidate_bag_dependencies(paths)
    unstamped = pa.Table.from_pylist([_candidate_row()], schema=CANDIDATES_SCHEMA)

    # 1. Never stamped at all.
    with pytest.raises(AxBenchBpbAdvError, match="identity is absent or stale"):
        write_clustered_table_atomic(
            unstamped,
            paths.candidates_path(),
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )
    assert not paths.candidates_path().exists()

    # 2. Correctly stamped, then the stamp is tampered to a stale value. Note
    #    a mismatched contract cannot be used to build this case -- binding
    #    validates against the source contract and would raise first -- so the
    #    stale state has to be forged directly on the schema metadata.
    stamped = bind_clustered_table(
        unstamped,
        paths=paths,
        contract=_contract(),
        artifact_family="candidates",
    )
    metadata = dict(stamped.schema.metadata or {})
    metadata[b"feature_contract_fingerprint"] = (b"c" * 64)
    stale = stamped.replace_schema_metadata(metadata)
    with pytest.raises(AxBenchBpbAdvError, match="identity is absent or stale"):
        write_clustered_table_atomic(
            stale,
            paths.candidates_path(),
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )
    assert not paths.candidates_path().exists()


def test_candidate_identity_without_table_raises(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="candidate identity requires the candidate table",
    ):
        clustered_artifact_identity(
            paths=paths,
            contract=_contract(),
            artifact_family="candidates",
        )


def test_candidate_without_precomputed_bag_family_is_skipped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    table = pa.Table.from_pylist(
        [_candidate_row(method_name="ours_extract_and_add")],
        schema=CANDIDATES_SCHEMA,
    )

    actual = artifact_identity._candidate_bag_set_sha256(table, paths=paths)

    assert actual == artifact_identity._canonical_sha256(
        {
            "schema_version": 1,
            "bags": [],
            "scored_prefixes": [],
        }
    )


def test_gcg_candidate_non_numeric_coefficient_names_method(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    method_name = "ours_gcg_encdec"
    table = pa.Table.from_pylist(
        [
            _candidate_row(
                method_name=method_name,
                coefficient_str="not-a-number",
            )
        ],
        schema=CANDIDATES_SCHEMA,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=rf"coefficient_str is not numeric: method='{method_name}'",
    ):
        artifact_identity._candidate_bag_set_sha256(table, paths=paths)


def test_fair_and_roofline_prefix_coordinates_produce_different_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    fair_row = _candidate_row(method_name="ours_gcg_encdec")
    roofline_row = _candidate_row(method_name="ours_gcg_roofline_encdec")
    _write_dependencies_for_row(paths, fair_row)
    _write_dependencies_for_row(paths, roofline_row)

    prefix_calls: list[str | None] = []
    real_ours_gcg_bag = paths.ours_gcg_bag

    def _record_ours_gcg_bag(
        concept_id: int,
        mode: Any,
        reg_lambda: float,
        family: Any,
        prefix_id: str | None = None,
        objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
    ) -> Path:
        prefix_calls.append(prefix_id)
        return real_ours_gcg_bag(
            concept_id,
            mode,
            reg_lambda,
            family,
            prefix_id=prefix_id,
            objective_alpha=objective_alpha,
        )

    monkeypatch.setattr(paths, "ours_gcg_bag", _record_ours_gcg_bag)
    fair_manifest = artifact_identity._candidate_bag_set_sha256(
        pa.Table.from_pylist([fair_row], schema=CANDIDATES_SCHEMA),
        paths=paths,
    )
    roofline_manifest = artifact_identity._candidate_bag_set_sha256(
        pa.Table.from_pylist([roofline_row], schema=CANDIDATES_SCHEMA),
        paths=paths,
    )

    assert prefix_calls == [None, "prefix-1"]
    assert fair_manifest != roofline_manifest


def test_candidate_manifest_collapses_shared_bag_but_not_distinct_lambda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    manifests: list[dict[str, Any]] = []
    real_canonical_sha256 = artifact_identity._canonical_sha256

    def _capture_manifest(value: Any) -> str:
        assert isinstance(value, dict)
        manifests.append(value)
        return real_canonical_sha256(value)

    def _path_digest(path: Path, *, label: str) -> str:
        del label
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()

    monkeypatch.setattr(artifact_identity, "_canonical_sha256", _capture_manifest)
    monkeypatch.setattr(artifact_identity, "_stable_file_sha256", _path_digest)

    shared_rows = [_candidate_row(), _candidate_row()]
    artifact_identity._candidate_bag_set_sha256(
        pa.Table.from_pylist(shared_rows, schema=CANDIDATES_SCHEMA),
        paths=paths,
    )
    lambda_rows = [
        _candidate_row(),
        _candidate_row(coefficient_str="0.5"),
    ]
    artifact_identity._candidate_bag_set_sha256(
        pa.Table.from_pylist(lambda_rows, schema=CANDIDATES_SCHEMA),
        paths=paths,
    )

    assert len(manifests[0]["bags"]) == 1
    assert len(manifests[0]["scored_prefixes"]) == 1
    assert len(manifests[1]["bags"]) == 2
    assert len(manifests[1]["scored_prefixes"]) == 1


def test_both_feature_diffmean_families_force_zero_lambda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    manifests: list[dict[str, Any]] = []
    real_canonical_sha256 = artifact_identity._canonical_sha256

    def _capture_manifest(value: Any) -> str:
        assert isinstance(value, dict)
        manifests.append(value)
        return real_canonical_sha256(value)

    monkeypatch.setattr(artifact_identity, "_canonical_sha256", _capture_manifest)
    monkeypatch.setattr(
        artifact_identity,
        "_stable_file_sha256",
        lambda path, *, label: hashlib.sha256(
            f"{label}:{path}".encode()
        ).hexdigest(),
    )
    rows = [
        _candidate_row(
            method_name="feature_diffmean_encdec",
            coefficient_str="not-numeric-and-ignored",
        ),
        _candidate_row(
            method_name="feature_diffmean_roofline_encdec",
            coefficient_str="also-ignored",
        ),
    ]

    artifact_identity._candidate_bag_set_sha256(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
        paths=paths,
    )

    assert {
        (entry["family"], entry["reg_lambda"]) for entry in manifests[0]["bags"]
    } == {
        ("feature_diffmean", "0.0"),
        ("feature_diffmean_roofline", "0.0"),
    }


@pytest.mark.parametrize(
    ("missing", "expected_label"),
    [
        ("bag", "precomputed conditioning bag"),
        ("scored", "precomputed-bag scored-prefix marker"),
    ],
)
def test_missing_candidate_dependency_names_label(
    tmp_path: Path,
    missing: str,
    expected_label: str,
) -> None:
    paths = _paths(tmp_path)
    row = _candidate_row()
    bag, scored = _write_dependencies_for_row(paths, row)
    {"bag": bag, "scored": scored}[missing].unlink()
    table = pa.Table.from_pylist([row], schema=CANDIDATES_SCHEMA)

    with pytest.raises(AxBenchBpbAdvError, match=expected_label):
        artifact_identity._candidate_bag_set_sha256(table, paths=paths)


def test_stable_dependency_hash_failure_names_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unstable.bin"
    path.write_bytes(b"content")

    def _fail_stable_hash(_path: Path) -> Any:
        raise RuntimeError("sentinel mutation")

    monkeypatch.setattr(
        artifact_identity,
        "stream_hash_file_stable",
        _fail_stable_hash,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="could not stably hash sentinel dependency",
    ):
        artifact_identity._stable_file_sha256(
            path,
            label="sentinel dependency",
        )


def test_identity_helpers_match_persisted_parquet_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    table = bind_clustered_table(
        pa.table({"score": [1.0]}),
        paths=paths,
        contract=_contract(),
        artifact_family="scores",
    )
    parquet_path = tmp_path / "scores.parquet"
    pq.write_table(table, parquet_path)
    persisted = pq.read_table(parquet_path)

    # This exact agreement is intentionally for table-independent families.
    # Candidate identity is table-dependent; the table-less diagnostic refuses
    # that call rather than silently omitting its precomputed-bag-set key.
    expected = clustered_artifact_identity(
        paths=paths,
        contract=_contract(),
        artifact_family="scores",
    )
    assert artifact_identity_metadata(table) == expected
    assert artifact_identity_metadata(persisted) == expected


def test_source_contract_must_be_json_object(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.source_contract().write_text("[]", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="must be a JSON object"):
        bind_clustered_table(
            pa.table({"x": [1]}),
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
        )


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_source_contract_must_be_readable(tmp_path: Path, failure: str) -> None:
    paths = _paths(tmp_path)
    if failure == "missing":
        paths.source_contract().unlink()
    else:
        paths.source_contract().write_text("{", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="requires a readable source contract"):
        bind_clustered_table(
            pa.table({"x": [1]}),
            paths=paths,
            contract=_contract(),
            artifact_family="scores",
        )


def test_source_contract_feature_fingerprint_must_match(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="feature fingerprint does not match",
    ):
        bind_clustered_table(
            pa.table({"x": [1]}),
            paths=paths,
            contract=_contract("b" * 64),
            artifact_family="scores",
        )


def test_downstream_identity_requires_candidate_snapshot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires one exact candidate snapshot",
    ):
        bind_clustered_table(
            pa.table({"score": [1.0]}),
            paths=paths,
            contract=_contract(),
            artifact_family="samples",
        )


def test_candidate_dependent_family_membership_is_frozen() -> None:
    # Mutation trap: any family addition/removal changes upstream-candidate
    # binding semantics and must force an explicit test update.
    assert artifact_identity._CANDIDATE_DEPENDENT_FAMILIES == frozenset(
        {
            "universal_triggers",
            "class_margin_scores",
            "roundtrip_scores",
            "general_val_bpb",
            "samples",
        }
    )


def test_candidate_bag_manifest_tracks_observed_file_contents(
    tmp_path: Path,
) -> None:
    """Pin G1-D2's accepted sequential multi-file observation boundary."""

    paths = _paths(tmp_path)
    row = _candidate_row()
    bag, scored = _write_dependencies_for_row(
        paths,
        row,
        bag_bytes=b"bag-observed-v1",
        scored_bytes=b"scored-observed-v1",
    )
    table = pa.Table.from_pylist([row], schema=CANDIDATES_SCHEMA)

    expected_v1 = artifact_identity._canonical_sha256(
        {
            "schema_version": 1,
            "bags": [
                {
                    "concept_id": 7,
                    "mode": "positive",
                    "family": "roofline",
                    "reg_lambda": "0.25",
                    "prefix_id": "prefix-1",
                    "sha256": hashlib.sha256(bag.read_bytes()).hexdigest(),
                }
            ],
            "scored_prefixes": [
                {
                    "concept_id": 7,
                    "mode": "positive",
                    "family": "roofline",
                    "sha256": hashlib.sha256(scored.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    actual_v1 = artifact_identity._candidate_bag_set_sha256(table, paths=paths)
    assert actual_v1 == expected_v1

    # This deliberately proves content sensitivity per observed path read, not
    # an unattainable atomic snapshot across the multi-file family. Production
    # load-time readers remain the authoritative consumption backstop.
    bag.write_bytes(b"bag-observed-v2")
    expected_v2 = artifact_identity._canonical_sha256(
        {
            "schema_version": 1,
            "bags": [
                {
                    "concept_id": 7,
                    "mode": "positive",
                    "family": "roofline",
                    "reg_lambda": "0.25",
                    "prefix_id": "prefix-1",
                    "sha256": hashlib.sha256(bag.read_bytes()).hexdigest(),
                }
            ],
            "scored_prefixes": [
                {
                    "concept_id": 7,
                    "mode": "positive",
                    "family": "roofline",
                    "sha256": hashlib.sha256(scored.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    actual_v2 = artifact_identity._candidate_bag_set_sha256(table, paths=paths)
    assert actual_v2 == expected_v2
    assert actual_v2 != actual_v1


# ─────────────────────────────────────────────────────────────────────────────
# #936 — a stage's stamps must be bound to ONE read of source_contract.json
# ─────────────────────────────────────────────────────────────────────────────


def _swap_source_contract(paths: AxBenchBpbAdvPaths) -> str:
    """Replace the source contract with a DIFFERENT but self-consistent one.

    Same ``feature_contract_fingerprint``, different content, and its own valid
    ``contract_hash`` — so every per-call validation still passes on the new file.
    This is exactly the replacement that made the two reads disagree silently.
    """
    payload = _source_contract_payload(n_dropped=1)
    paths.source_contract().write_text(json.dumps(payload), encoding="utf-8")
    return payload["contract_hash"]


def test_resolved_identity_pins_the_hash_across_a_mid_stage_swap(
    tmp_path: Path,
) -> None:
    """A resolved identity keeps every later stamp on the first read's hash (#936).

    Before this, each entry point re-read ``source_contract.json``, so an artifact
    validated at the start of a stage and one stamped at the end could record
    different source hashes — and nothing downstream compared them.
    """
    paths = _paths(tmp_path)
    contract = _contract()
    original_hash = json.loads(paths.source_contract().read_text(encoding="utf-8"))[
        "contract_hash"
    ]

    identity = artifact_identity.resolve_clustered_source_identity(
        paths=paths, contract=contract
    )
    swapped_hash = _swap_source_contract(paths)
    assert swapped_hash != original_hash

    table = pa.table({"value": pa.array([1], type=pa.int64())})
    bound = bind_clustered_table(
        table,
        paths=paths,
        contract=contract,
        artifact_family="attack_rows",
        source_identity=identity,
    )

    stamped = artifact_identity_metadata(bound)["source_contract_hash"]
    assert stamped == original_hash, (
        "a stamp made with a resolved identity must use the single read it was "
        "resolved from, not a fresh read of a swapped file"
    )

    # Without the resolved identity the entry point re-reads and picks up the swap —
    # this is the pre-#936 behaviour, kept for single-artifact callers, and is what
    # makes the assertion above meaningful rather than vacuous.
    rebound = bind_clustered_table(
        table,
        paths=paths,
        contract=contract,
        artifact_family="attack_rows",
    )
    assert artifact_identity_metadata(rebound)["source_contract_hash"] == swapped_hash


def test_validate_then_stamp_agree_under_a_mid_stage_swap(tmp_path: Path) -> None:
    """The #936 scenario end to end: validate one artifact, swap, stamp another.

    Reproduces the confirmed silent failure — attack rows validated against read A,
    candidates stamped from read B, with no downstream comparison of the two — and
    asserts the two artifacts now carry the SAME source hash.
    """
    paths = _paths(tmp_path)
    contract = _contract()
    identity = artifact_identity.resolve_clustered_source_identity(
        paths=paths, contract=contract
    )

    first = bind_clustered_table(
        pa.table({"value": pa.array([1], type=pa.int64())}),
        paths=paths,
        contract=contract,
        artifact_family="attack_rows",
        source_identity=identity,
    )
    assert_table_cluster_identity(
        first,
        paths=paths,
        contract=contract,
        artifact_family="attack_rows",
        source_identity=identity,
    )

    _swap_source_contract(paths)

    # A second, independently stamped artifact later in the same stage. Deliberately
    # a family with no candidate dependency, so this test isolates the source-read
    # binding rather than dragging in the candidates-on-disk requirement.
    second = bind_clustered_table(
        pa.table({"value": pa.array([2], type=pa.int64())}),
        paths=paths,
        contract=contract,
        artifact_family="attack_rows",
        source_identity=identity,
    )

    assert (
        artifact_identity_metadata(first)["source_contract_hash"]
        == artifact_identity_metadata(second)["source_contract_hash"]
    ), "two artifacts stamped in one stage must share one source-contract read"


def test_resolved_identity_cannot_be_paired_with_another_contract(
    tmp_path: Path,
) -> None:
    """An identity carries the fingerprint it was resolved for and refuses others.

    Resolving once creates a new way to be wrong: reusing a stale identity under a
    different contract. The value object rejects that pairing rather than stamping a
    source hash belonging to some other contract.
    """
    paths = _paths(tmp_path)
    identity = artifact_identity.resolve_clustered_source_identity(
        paths=paths, contract=_contract()
    )

    with pytest.raises(AxBenchBpbAdvError, match="was resolved for feature contract"):
        bind_clustered_table(
            pa.table({"value": pa.array([1], type=pa.int64())}),
            paths=paths,
            contract=_contract("b" * 64),
            artifact_family="attack_rows",
            source_identity=identity,
        )


def test_source_identity_cannot_be_hand_constructed(tmp_path: Path) -> None:
    """Only the resolver may build an identity, so none can skip validation."""
    with pytest.raises(TypeError, match="resolve_clustered_source_identity"):
        artifact_identity.ClusteredSourceIdentity("c" * 64, _FP)
