"""Unit tests for ``steering.eval.axbench_base.cluster_artifacts`` (PR #739
Track C1 port plus the final E3 forward-boundary guard wiring).

Ports the subset of the oracle's ``tests/axbench_base/unit/test_clustered_
stage4_artifacts.py`` tests whose subject-under-test is either a
``cluster_artifacts.py`` function or the final Stage-4 checkpoint guard. The
three ``test_clustered_stage4_checkpoint_guard_*`` tests are included here now
that E3 owns that wiring.

Two of the 14 ported tests are ADAPTED, not byte-identical ports: Finding 1
(see ``cluster_artifacts.py``'s own module docstring) replaces the two-call
``stream_hash_file_stable`` hash-before/read/hash-after bracket with a
single fused hash-then-read primitive, so the oracle's "mutate on the
SECOND hash call" injection mechanism no longer has a second call to
trigger on. Both adapted tests instead mutate the target file as a side
effect of the (now single) hash call, immediately before the subsequent
read -- proven, in comments at each site, to still trigger the exact same
observable failure.

New tests beyond the 14 ported ones cover: the two Finding-1 deviation
sites' own equivalence/bug-fix proofs, Finding 2's error-type-consistency
resolution, the ``_strict_sorted_unique_ids`` asymmetry adjudication, every
remaining raise site in ``install_or_validate_cluster_root_marker[_ddp]``
(untouched by the oracle's own 17 tests), the three ``load_cluster_
candidate_identity`` branches, the ``clustered_stage0c_fingerprint_fields``
no-op, and import-path preservation (constraint 4).
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
import torch as t
import torch.distributed as dist

import scripts.eval.axbench_base_eval as cli
import steering.eval.cluster_checkpoint_pin as cluster_checkpoint_pin_module
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.eval.axbench_base import cluster_artifacts
from steering.eval.axbench_base.cluster_artifacts import (
    ClusterCandidateIdentity,
    assert_live_statistics_identity,
    assert_live_statistics_identity_and_build_percentile_table,
    assert_live_statistics_match_committed_marker,
    build_verified_percentile_table_with_collective_agreement,
    clustered_source_enabled,
    clustered_stage0c_fingerprint_fields,
    commit_cluster_generation_marker,
    install_or_validate_cluster_root_marker,
    install_or_validate_cluster_root_marker_ddp,
    load_cluster_candidate_identity,
    validate_cluster_candidate_identity_with_collective_agreement,
    validate_cluster_generation_resume,
    validate_resume_statistics_identity_with_collective_agreement,
)
from steering.eval.axbench_base.extraction import build_percentile_table_from_stats
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_contract_identity import _sha256_json
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.axbench_base.unit.test_config import make_valid_config
from tests.runtime_attribute_helpers import set_runtime_attribute

# ============================================================================
# Shared fixtures/helpers (ported from the oracle's test_clustered_stage4_
# artifacts.py, minus the `cli`-only ones).
# ============================================================================


def _contract(fingerprint: str) -> Any:
    return cast(Any, SimpleNamespace(feature_contract_fingerprint=fingerprint))


def _cfg(run_tag: str):
    return make_valid_config(
        run_tag=run_tag,
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def _install_candidate_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(_cfg, *, expected_fingerprint=None, **_kwargs):
        return ClusterCandidateIdentity(
            feature_contract_fingerprint=str(expected_fingerprint),
            candidate_concepts_sha256="b" * 64,
            candidate_provenance_sha256="c" * 64,
            concept_projection_version="cluster_concept_projection_v1",
        )

    monkeypatch.setattr(cluster_artifacts, "load_cluster_candidate_identity", load)


def _write_concepts(run_tag: str) -> str:
    path = test_paths.default_paths(run_tag).concepts_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"concept_id":0,"feature_index":1}]\n', encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_inputs(
    *, feature_contract_fingerprint: str, concepts_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature_contract_fingerprint": feature_contract_fingerprint,
        "artifact_sha256": {"concepts.json": concepts_sha256},
    }


def _candidate_contract_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature_contract_fingerprint": "a" * 64,
        "semantic": {
            "algorithms": {"concept_projection": "mask_first_min_allowed_representative_v1"},
            "feature_clusters": {
                "representative_universe": [1, 5],
                "representative_to_allowed_members": [
                    [1, [1, 2]],
                    [5, [5]],
                ],
            },
            "concept_source": {
                "mode": "explicit_whitelist",
                "domain": [2, 5],
                "source_to_representative": [[2, 1], [5, 5]],
            },
        },
        "provenance": {},
    }


def _write_candidate_artifacts(
    run_tag: str,
    *,
    candidate_rows: list[dict[str, Any]] | None = None,
    provenance_mutation: str | None = None,
) -> None:
    rows = candidate_rows or [
        {"feature_index": 1},
        {"feature_index": 5},
    ]
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json()
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(rows), encoding="utf-8")
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "feature_contract_fingerprint": "a" * 64,
        "candidate_concepts_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "concept_projection_version": "mask_first_min_allowed_representative_v1",
        "counts": {
            "raw_source_ids": 2,
            "mask_allowed_source_ids": 2,
            "unique_representatives": 2,
            "representatives_with_statistics": 2,
            "count_survivors": 2,
            "description_fetch_survivors": 2,
            "described_survivors": 2,
        },
        "representatives": [
            {
                "representative_id": 1,
                "allowed_member_ids": [1, 2],
                "projecting_source_ids": [2],
                "statistics_source_id": 1,
                "outcome": "candidate_emitted",
            },
            {
                "representative_id": 5,
                "allowed_member_ids": [5],
                "projecting_source_ids": [5],
                "statistics_source_id": 5,
                "outcome": "candidate_emitted",
            },
        ],
    }
    if provenance_mutation == "member_set":
        provenance["representatives"][0]["allowed_member_ids"] = [1]
    elif provenance_mutation == "projecting_sources":
        provenance["representatives"][0]["projecting_source_ids"] = [5]
    elif provenance_mutation == "outcome":
        provenance["representatives"][0]["outcome"] = "count_survivor"
    elif provenance_mutation == "counts_extra":
        provenance["counts"]["stage0c_survivors"] = 2
    elif provenance_mutation == "missing_representative":
        provenance["representatives"].pop()
    elif provenance_mutation == "count_value":
        provenance["counts"]["described_survivors"] = 1
    elif provenance_mutation == "bool_id":
        provenance["representatives"][0]["representative_id"] = True
    elif provenance_mutation is not None:
        raise AssertionError(f"unknown provenance mutation {provenance_mutation!r}")
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(provenance), encoding="utf-8"
    )


# ============================================================================
# Installer byte-stability/immutability/race tests (direct port).
# ============================================================================


def test_clustered_bytes_are_create_once_and_exact_replay_only(tmp_path: Path) -> None:
    artifact = tmp_path / "immutable.bin"
    assert cluster_artifacts.install_clustered_bytes_once(
        b"first", artifact, surface="test artifact"
    )
    original = artifact.read_bytes()
    assert not cluster_artifacts.install_clustered_bytes_once(
        b"first", artifact, surface="test artifact"
    )

    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        cluster_artifacts.install_clustered_bytes_once(b"second", artifact, surface="test artifact")

    assert artifact.read_bytes() == original
    assert not list(tmp_path.glob(".immutable.bin.tmp.*"))


def test_clustered_bytes_concurrent_writers_cannot_replace_winner(tmp_path: Path) -> None:
    artifact = tmp_path / "raced.bin"
    start = threading.Barrier(2)

    def write(payload: bytes) -> tuple[str, bytes]:
        start.wait()
        try:
            cluster_artifacts.install_clustered_bytes_once(
                payload, artifact, surface="raced artifact"
            )
        except AxBenchBaseError:
            return "rejected", payload
        return "installed", payload

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, (b"left", b"right")))

    assert sorted(status for status, _payload in results) == ["installed", "rejected"]
    winner = next(payload for status, payload in results if status == "installed")
    assert artifact.read_bytes() == winner
    assert not list(tmp_path.glob(".raced.bin.tmp.*"))


def test_clustered_structured_writers_are_byte_stable_and_immutable(tmp_path: Path) -> None:
    parquet_path = tmp_path / "rows.parquet"
    table = pa.table({"feature_id": pa.array([3, 7], type=pa.int64())})
    assert cluster_artifacts.write_clustered_parquet_once(
        table, parquet_path, surface="test parquet"
    )
    parquet_bytes = parquet_path.read_bytes()
    assert not cluster_artifacts.write_clustered_parquet_once(
        table, parquet_path, surface="test parquet"
    )
    assert parquet_path.read_bytes() == parquet_bytes
    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        cluster_artifacts.write_clustered_parquet_once(
            pa.table({"feature_id": pa.array([3, 8], type=pa.int64())}),
            parquet_path,
            surface="test parquet",
        )

    tensor_path = tmp_path / "values.safetensors"
    tensors = {"ids": t.tensor([3, 7], dtype=t.int64)}
    metadata = {"feature_contract_fingerprint": "a" * 64}
    assert cluster_artifacts.write_clustered_safetensors_once(
        tensors, tensor_path, metadata=metadata, surface="test tensors"
    )
    tensor_bytes = tensor_path.read_bytes()
    assert not cluster_artifacts.write_clustered_safetensors_once(
        tensors, tensor_path, metadata=metadata, surface="test tensors"
    )
    assert tensor_path.read_bytes() == tensor_bytes
    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        cluster_artifacts.write_clustered_safetensors_once(
            {"ids": t.tensor([3, 8], dtype=t.int64)},
            tensor_path,
            metadata=metadata,
            surface="test tensors",
        )


# ============================================================================
# Candidate-provenance semantics tests (direct port, except the mutation
# test below which is adapted for Finding 1).
# ============================================================================


def test_clustered_candidate_identity_validates_full_projection_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-semantics"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    identity = load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)

    assert identity == ClusterCandidateIdentity(
        feature_contract_fingerprint=(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
        candidate_concepts_sha256=(
            "204d81a04f8584bf08671e6c5b2db647bd9b5e8f6ceffab6f681aba95ded78df"
        ),
        candidate_provenance_sha256=(
            "7b2bf7b5685044ba060f45c8e6ef5f41bf819aff5e315e5a276930889f29cdd3"
        ),
        concept_projection_version="mask_first_min_allowed_representative_v1",
    )


def test_clustered_candidate_semantics_accepts_masked_statistics_domain() -> None:
    document = _candidate_contract_document()
    document["semantic"]["concept_source"] = {
        "mode": "statistics_domain",
        "domain": [0, 2, 5],
        "source_to_representative": [[2, 1], [5, 5]],
    }
    provenance = {
        "counts": {
            "raw_source_ids": 3,
            "mask_allowed_source_ids": 2,
            "unique_representatives": 2,
            "representatives_with_statistics": 2,
            "count_survivors": 2,
            "description_fetch_survivors": 2,
            "described_survivors": 2,
        },
        "representatives": [
            {
                "representative_id": 1,
                "allowed_member_ids": [1, 2],
                "projecting_source_ids": [2],
                "statistics_source_id": 1,
                "outcome": "candidate_emitted",
            },
            {
                "representative_id": 5,
                "allowed_member_ids": [5],
                "projecting_source_ids": [5],
                "statistics_source_id": 5,
                "outcome": "candidate_emitted",
            },
        ],
    }

    cluster_artifacts.validate_cluster_candidate_provenance_semantics(
        feature_contract=document,
        candidate_rows=[{"feature_index": 5}, {"feature_index": 1}],
        provenance=provenance,
    )


@pytest.mark.parametrize(
    ("outcome", "representatives_with_statistics", "count_survivors", "fetch_survivors"),
    [
        ("missing_representative_statistics", 1, 1, 1),
        ("below_min_corpus_count", 2, 1, 1),
        ("excluded_by_description_fetch_limit", 2, 2, 1),
        ("description_unavailable", 2, 2, 2),
        ("description_below_quality_threshold", 2, 2, 2),
    ],
)
def test_clustered_candidate_semantics_accepts_every_terminal_exclusion(
    outcome: str,
    representatives_with_statistics: int,
    count_survivors: int,
    fetch_survivors: int,
) -> None:
    provenance = {
        "counts": {
            "raw_source_ids": 2,
            "mask_allowed_source_ids": 2,
            "unique_representatives": 2,
            "representatives_with_statistics": representatives_with_statistics,
            "count_survivors": count_survivors,
            "description_fetch_survivors": fetch_survivors,
            "described_survivors": 1,
        },
        "representatives": [
            {
                "representative_id": 1,
                "allowed_member_ids": [1, 2],
                "projecting_source_ids": [2],
                "statistics_source_id": 1,
                "outcome": outcome,
            },
            {
                "representative_id": 5,
                "allowed_member_ids": [5],
                "projecting_source_ids": [5],
                "statistics_source_id": 5,
                "outcome": "candidate_emitted",
            },
        ],
    }

    cluster_artifacts.validate_cluster_candidate_provenance_semantics(
        feature_contract=_candidate_contract_document(),
        candidate_rows=[{"feature_index": 5}],
        provenance=provenance,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "member_set",
        "projecting_sources",
        "outcome",
        "counts_extra",
        "missing_representative",
        "count_value",
        "bool_id",
    ],
)
def test_clustered_candidate_identity_rejects_stale_projection_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = f"candidate-{mutation}"
    _write_candidate_artifacts(run_tag, provenance_mutation=mutation)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    with pytest.raises(AxBenchBaseError, match="candidate provenance"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_nonrepresentative_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-nonrepresentative"
    _write_candidate_artifacts(run_tag, candidate_rows=[{"feature_index": 2}, {"feature_index": 5}])
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    with pytest.raises(AxBenchBaseError, match="non-representative"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_candidate_mutation_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ported from the oracle unchanged in spirit: the file mutates as a
    side effect of the ``before`` hash call -- the BEFORE-to-read window.
    ``_load_stable_json`` now restores the oracle's own two-hash bracket
    byte-for-byte (see module docstring's "Named deviation 1b" -- an
    earlier revision of this port incorrectly simplified it to a single
    hash, which a Codex cross-review caught as a real regression), so this
    scenario is over-determined here: the mutated content is what gets
    read (the payload disagrees with ``before.sha256``), AND the
    subsequent ``after`` hash also disagrees with ``before``. See
    ``test_clustered_candidate_identity_rejects_candidate_swap_after_read``
    immediately below for the complementary POST-READ swap window (Codex
    cross-review Finding 1, HIGH), which ONLY the restored ``after``
    re-hash catches -- this test alone would not have caught that one.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-mutates"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    real_hash = cluster_artifacts.stream_hash_file_stable
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json().resolve()

    def _mutate_after_candidate_hash(path: str | Path):
        resolved = Path(path).resolve()
        result = real_hash(path)
        if resolved == candidate_path:
            candidate_path.write_text(
                candidate_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(cluster_artifacts, "stream_hash_file_stable", _mutate_after_candidate_hash)

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_candidate_swap_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex cross-review Finding 1 (HIGH): the POST-READ swap window.

    The file is A both when hashed AND when read (so the hash-to-read
    window the test above covers sees no mismatch at all) -- but a racing
    writer swaps it to B immediately AFTER the read completes, before the
    function returns. ``_load_stable_json`` must still catch this: it
    restores the oracle's own post-read re-hash (``after = stream_hash_
    file_stable(path)``, cross-checked via ``before != after``) rather
    than accepting an A-backed identity while the file is already B on
    disk. This is the exact race an earlier revision of this file's
    single-hash-then-read shape could not catch (verified: replaying this
    identical scenario against that shape does NOT raise -- see the
    module docstring and task-c1 fix-batch report for the full trace).
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-swapped-after-read"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    real_hash = cluster_artifacts.stream_hash_file_stable
    real_read_bytes = Path.read_bytes
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json().resolve()

    def _passthrough_hash(path: str | Path):
        return real_hash(path)

    def _swap_after_read(self: Path, *args: Any, **kwargs: Any) -> bytes:
        payload = real_read_bytes(self, *args, **kwargs)
        if self.resolve() == candidate_path:
            # Simulate a racing writer that swaps the file to B immediately
            # after THIS read completes -- before any post-read re-hash.
            candidate_path.write_text(
                candidate_path.read_text(encoding="utf-8") + " swapped-after-read",
                encoding="utf-8",
            )
        return payload

    monkeypatch.setattr(cluster_artifacts, "stream_hash_file_stable", _passthrough_hash)
    monkeypatch.setattr(Path, "read_bytes", _swap_after_read)

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_provenance_sidecar_mutation_while_loading_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW (Finding 1 simplification equivalence proof, dict-shaped site):
    none of the oracle's 17 tests exercised a mutation race on
    ``candidate_concepts.cluster_provenance.json`` itself (only on
    ``candidate_concepts.json``) -- this closes that gap for the site that
    now reuses ``cluster_checkpoint_pin._load_json_mapping`` directly.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-mutates"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = (
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().resolve()
    )
    real_hash = cluster_checkpoint_pin_module.stream_hash_file_stable

    def _mutate_after_provenance_hash(path: str | Path):
        resolved = Path(path).resolve()
        result = real_hash(path)
        if resolved == provenance_path:
            provenance_path.write_text(
                provenance_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(
        cluster_checkpoint_pin_module, "stream_hash_file_stable", _mutate_after_provenance_hash
    )

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_provenance_swap_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: reject a persistent provenance swap staged after its read."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-swapped-after-read"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = (
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().resolve()
    )
    real_read_bytes = Path.read_bytes

    def _swap_after_read(self: Path, *args: Any, **kwargs: Any) -> bytes:
        payload = real_read_bytes(self, *args, **kwargs)
        if self.resolve() == provenance_path:
            provenance_path.write_bytes(payload + b" ")
        return payload

    monkeypatch.setattr(Path, "read_bytes", _swap_after_read)

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_provenance_swap_during_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3: the final bind catches a provenance swap after the snapshot."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-swapped-during-semantics"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = (
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().resolve()
    )
    real_validate = cluster_artifacts.validate_cluster_candidate_provenance_semantics

    def _validate_then_swap(*args: Any, **kwargs: Any) -> None:
        real_validate(*args, **kwargs)
        provenance_path.write_bytes(provenance_path.read_bytes() + b" ")

    monkeypatch.setattr(
        cluster_artifacts,
        "validate_cluster_candidate_provenance_semantics",
        _validate_then_swap,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="clustered candidate provenance changed while it was being loaded",
    ):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_candidate_swap_during_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: the final bind catches a candidate swap after its own bracket."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-swapped-during-semantics"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json().resolve()
    real_validate = cluster_artifacts.validate_cluster_candidate_provenance_semantics

    def _validate_then_swap(*args: Any, **kwargs: Any) -> None:
        real_validate(*args, **kwargs)
        candidate_path.write_bytes(candidate_path.read_bytes() + b" ")

    monkeypatch.setattr(
        cluster_artifacts,
        "validate_cluster_candidate_provenance_semantics",
        _validate_then_swap,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="clustered candidate concepts changed while it was being loaded",
    ):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_clustered_candidate_identity_rejects_root_marker_swap_during_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a persistent valid root-marker swap using the real loader.

    Contract A is loaded and supplies the identity fingerprint. Semantic
    validation then installs self-consistent contract B with changed contract
    provenance but the same verified semantic fingerprint. No loader stub is
    installed: both reads parse and validate the real on-disk marker, and the
    final re-load accepts B's fingerprint before the whole-document comparison
    rejects the swap. Returning one stub document twice would make this test
    fail, so it cannot pass vacuously.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "root-marker-swapped-during-semantics"
    _write_candidate_artifacts(run_tag)
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)

    document_a = _candidate_contract_document()
    fingerprint_a = _sha256_json(document_a["semantic"])
    document_a["feature_contract_fingerprint"] = fingerprint_a
    marker.write_text(json.dumps(document_a), encoding="utf-8")

    provenance_path = test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["feature_contract_fingerprint"] = fingerprint_a
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    document_b = copy.deepcopy(document_a)
    document_b["provenance"]["race_variant"] = "contract-b"
    fingerprint_b = _sha256_json(document_b["semantic"])
    assert fingerprint_b == fingerprint_a
    assert document_b != document_a

    real_validate = cluster_artifacts.validate_cluster_candidate_provenance_semantics
    swap_performed = False

    def _validate_then_swap_root_marker(*args: Any, **kwargs: Any) -> None:
        nonlocal swap_performed
        real_validate(*args, **kwargs)
        marker.write_text(json.dumps(document_b), encoding="utf-8")
        swap_performed = True

    monkeypatch.setattr(
        cluster_artifacts,
        "validate_cluster_candidate_provenance_semantics",
        _validate_then_swap_root_marker,
    )

    with pytest.raises(AxBenchBaseError, match="feature-contract marker changed"):
        load_cluster_candidate_identity(
            _cfg(run_tag),
            expected_fingerprint=fingerprint_a,
        )

    assert swap_performed
    assert cluster_artifacts.load_feature_contract_sidecar(
        marker,
        expected_fingerprint=fingerprint_b,
    ) == document_b


def test_clustered_candidate_identity_rejects_provenance_round_trip_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5: the full-digest snapshot catches A-to-B-to-A stat drift."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-round-trip-before-snapshot"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = (
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().resolve()
    )
    original_stat = provenance_path.stat()
    real_read_bytes = Path.read_bytes
    round_trip_performed = False

    def _round_trip_after_read(self: Path, *args: Any, **kwargs: Any) -> bytes:
        nonlocal round_trip_performed
        payload = real_read_bytes(self, *args, **kwargs)
        if self.resolve() == provenance_path:
            provenance_path.write_bytes(payload + b" ")
            provenance_path.write_bytes(payload)
            round_trip_performed = True
            restored_stat = provenance_path.stat()
            assert (
                restored_stat.st_mtime_ns,
                restored_stat.st_ctime_ns,
            ) != (
                original_stat.st_mtime_ns,
                original_stat.st_ctime_ns,
            )
        return payload

    monkeypatch.setattr(Path, "read_bytes", _round_trip_after_read)

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)
    assert round_trip_performed


def test_clustered_candidate_identity_hash_call_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U3: pin the two local binds without conflating imported hash names."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-hash-accounting"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = (
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().resolve()
    )
    candidate_path = test_paths.default_paths(run_tag).candidate_concepts_json().resolve()
    real_hash = cluster_artifacts.stream_hash_file_stable
    hashed_paths: list[Path] = []

    def _spy_hash(path: str | Path):
        digest = real_hash(path)
        hashed_paths.append(digest.path)
        return digest

    monkeypatch.setattr(cluster_artifacts, "stream_hash_file_stable", _spy_hash)

    identity = load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)

    assert identity is not None
    assert hashed_paths.count(provenance_path) == 2
    assert hashed_paths.count(candidate_path) == 3
    assert len(hashed_paths) == 5


def test_clustered_candidate_identity_does_not_post_hash_after_mapping_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5: a mapping failure keeps its wrapper and skips the local hash."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-mapping-failure"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    provenance_path = test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json()
    post_hash_calls: list[Path] = []

    def _mapping_failure(*_args: Any, **_kwargs: Any):
        raise RuntimeError("injected mapping failure")

    def _spy_hash(path: str | Path):
        post_hash_calls.append(Path(path))
        raise AssertionError("post-read hash must not run after mapping failure")

    monkeypatch.setattr(cluster_artifacts, "_load_json_mapping", _mapping_failure)
    monkeypatch.setattr(cluster_artifacts, "stream_hash_file_stable", _spy_hash)

    with pytest.raises(AxBenchBaseError) as exc_info:
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)

    assert str(exc_info.value) == (
        "Clustered candidate JSON is incomplete without valid commit marker "
        f"{provenance_path}: injected mapping failure"
    )
    assert post_hash_calls == []


def test_clustered_candidate_provenance_sidecar_non_object_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW (Finding 1 simplification consequence, dead-code removal): the
    oracle's own downstream ``isinstance(raw, Mapping)`` check
    (independent of ``_load_stable_json``) is now unreachable and was
    removed -- ``_load_json_mapping`` itself guarantees a dict, raising
    before ever returning otherwise. This proves the "provenance must be a
    JSON object" invariant is still enforced (just from inside the reused
    primitive instead of a separate downstream check), with the raise
    still surfacing as ``AxBenchBaseError``.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-not-an-object"
    _write_candidate_artifacts(run_tag)
    # Overwrite the just-written provenance sidecar with a JSON ARRAY
    # (list-shaped, not object-shaped).
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    with pytest.raises(AxBenchBaseError, match="incomplete without valid commit marker"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


# ============================================================================
# _strict_sorted_unique_ids asymmetry adjudication (NEW; owner-lens finding,
# named bug-fix deviation -- see module docstring).
# ============================================================================


def test_clustered_candidate_semantics_rejects_empty_concept_source_domain() -> None:
    """The oracle accepted a well-formed-but-EMPTY ``concept_source.domain``
    (with every other field consistently empty to match) via an ``is None``
    (not falsy) check. This validator now rejects it, symmetric with its
    own ``representative_universe`` check and with the E4 non-emptiness
    invariant main's contract builder enforces upstream (see module
    docstring, Named deviation 3) -- a genuinely-built contract can never
    produce this sidecar shape in the first place.
    """
    document = _candidate_contract_document()
    document["semantic"]["feature_clusters"]["representative_universe"] = []
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = []
    document["semantic"]["concept_source"] = {
        "mode": "statistics_domain",
        "domain": [],
        "source_to_representative": [],
    }
    provenance = {
        "counts": {
            "raw_source_ids": 0,
            "mask_allowed_source_ids": 0,
            "unique_representatives": 0,
            "representatives_with_statistics": 0,
            "count_survivors": 0,
            "description_fetch_survivors": 0,
            "described_survivors": 0,
        },
        "representatives": [],
    }

    # The (also-empty) representative_universe check fires first regardless
    # (both are now falsy-rejected) -- this cell proves the OVERALL
    # all-empty sidecar is rejected; the dedicated concept_source-only cell
    # below isolates the domain check specifically.
    with pytest.raises(AxBenchBaseError, match="representative universe"):
        cluster_artifacts.validate_cluster_candidate_provenance_semantics(
            feature_contract=document,
            candidate_rows=[],
            provenance=provenance,
        )


def test_clustered_candidate_semantics_rejects_empty_domain_with_nonempty_universe() -> None:
    """Isolates the concept_source.domain non-emptiness check itself: a
    NON-empty representative_universe (so that check passes) paired with an
    empty concept_source.domain must still be rejected -- this is the exact
    asymmetry the oracle had (representative_universe: falsy-rejected;
    domain: only None-rejected) and this deviation closes.
    """
    document = _candidate_contract_document()
    document["semantic"]["concept_source"] = {
        "mode": "statistics_domain",
        "domain": [],
        "source_to_representative": [],
    }
    provenance = {
        "counts": {
            "raw_source_ids": 0,
            "mask_allowed_source_ids": 0,
            "unique_representatives": 0,
            "representatives_with_statistics": 0,
            "count_survivors": 0,
            "description_fetch_survivors": 0,
            "described_survivors": 0,
        },
        "representatives": [],
    }

    with pytest.raises(AxBenchBaseError, match="concept domain must be sorted unique IDs"):
        cluster_artifacts.validate_cluster_candidate_provenance_semantics(
            feature_contract=document,
            candidate_rows=[],
            provenance=provenance,
        )


# ============================================================================
# install_or_validate_cluster_root_marker[_ddp] (NEW -- none of the oracle's
# 17 tests exercise this function at all; raise-site + Finding 2 coverage).
# ============================================================================


def test_root_marker_requires_run_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="requires cfg.run_tag"):
        install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))


def test_root_marker_rejects_non_directory_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-not-a-dir")
    root = test_paths.default_paths("root-not-a-dir").run_output_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_bytes(b"not a directory")

    with pytest.raises(AxBenchBaseError, match="is not a directory"):
        install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))


def test_root_marker_unclustered_is_noop_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-unclustered-noop")
    install_or_validate_cluster_root_marker(cfg, None)
    assert not test_paths.default_paths("root-unclustered-noop").feature_contract_json().exists()


def test_root_marker_unclustered_rejects_existing_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-unclustered-rejects")
    marker = test_paths.default_paths("root-unclustered-rejects").feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="Unclustered run rejects cluster-stamped output"):
        install_or_validate_cluster_root_marker(cfg, None)


def test_root_marker_rejects_nonempty_root_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-nonempty-no-marker")
    root = test_paths.default_paths("root-nonempty-no-marker").run_output_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "some_other_artifact.json").write_text("{}", encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="nonempty but has no exact feature-contract"):
        install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))


def test_root_marker_load_wraps_underlying_error_as_axbenchbaseerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 2 (resolved as 'wrap uniformly'): a non-AxBenchBaseError
    raised by ``load_feature_contract_sidecar`` (e.g. the real
    ``ClusterFeatureContractError``) is converted to ``AxBenchBaseError``
    rather than propagating unwrapped.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-load-wraps")
    marker = test_paths.default_paths("root-load-wraps").feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    class _OtherHierarchyError(RuntimeError):
        pass

    def _raise_other(*_args, **_kwargs):
        raise _OtherHierarchyError("simulated Track B validation failure")

    monkeypatch.setattr(cluster_artifacts, "load_feature_contract_sidecar", _raise_other)

    with pytest.raises(AxBenchBaseError, match="Invalid clustered feature-contract marker"):
        install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))


def test_root_marker_load_success_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-load-success")
    marker = test_paths.default_paths("root-load-success").feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    calls: list[Any] = []

    def _fake_load(path, *, expected_fingerprint=None):
        calls.append((path, expected_fingerprint))
        return {"schema_version": 1}

    monkeypatch.setattr(cluster_artifacts, "load_feature_contract_sidecar", _fake_load)

    install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))
    assert len(calls) == 1


def test_root_marker_write_wraps_underlying_error_as_axbenchbaseerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 2, write-side symmetric case: a non-AxBenchBaseError raised
    by ``write_feature_contract_sidecar`` (e.g. a genuine concurrent-writer
    conflict, which that function raises as ``ClusterFeatureContractError``)
    is also converted to ``AxBenchBaseError``.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-write-wraps")

    class _OtherHierarchyError(RuntimeError):
        pass

    def _raise_other(*_args, **_kwargs):
        raise _OtherHierarchyError("simulated concurrent-writer conflict")

    monkeypatch.setattr(cluster_artifacts, "write_feature_contract_sidecar", _raise_other)

    with pytest.raises(
        AxBenchBaseError, match="Could not commit clustered feature-contract marker"
    ):
        install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))


def test_root_marker_write_success_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("root-write-success")
    calls: list[Any] = []

    def _fake_write(path, contract):
        calls.append((path, contract))
        return Path(path)

    monkeypatch.setattr(cluster_artifacts, "write_feature_contract_sidecar", _fake_write)

    install_or_validate_cluster_root_marker(cfg, _contract("a" * 64))
    assert len(calls) == 1


def test_ddp_root_marker_rank_zero_wraps_failure_with_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag=None,
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
    )

    with pytest.raises(AxBenchBaseError, match="cluster run-marker validation failed"):
        install_or_validate_cluster_root_marker_ddp(cfg, _contract("a" * 64), rank=0, world_size=1)


def test_ddp_root_marker_rank_zero_success_single_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("ddp-rank0-ok")
    # world_size=1 skips the broadcast entirely; contract=None is a
    # legitimate unclustered no-op, exercised through the real underlying
    # function with no mocking required.
    install_or_validate_cluster_root_marker_ddp(cfg, None, rank=0, world_size=1)


def test_ddp_root_marker_non_zero_rank_never_touches_disk_and_propagates_broadcast_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    def _fail_if_called(*_args, **_kwargs):
        pytest.fail("install_or_validate_cluster_root_marker must not run on rank != 0")

    monkeypatch.setattr(
        cluster_artifacts, "install_or_validate_cluster_root_marker", _fail_if_called
    )

    def _broadcast_simulated_error(payload_list, src=0):
        assert src == 0
        assert payload_list == [None]
        payload_list[0] = "AxBenchBaseError: simulated rank-0 failure"

    monkeypatch.setattr(cluster_artifacts.dist, "broadcast_object_list", _broadcast_simulated_error)

    with pytest.raises(AxBenchBaseError, match="cluster run-marker validation failed"):
        install_or_validate_cluster_root_marker_ddp(
            _cfg("ddp-rank1-fail"), _contract("a" * 64), rank=1, world_size=2
        )


def test_ddp_root_marker_non_zero_rank_succeeds_on_clean_broadcast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    def _fail_if_called(*_args, **_kwargs):
        pytest.fail("install_or_validate_cluster_root_marker must not run on rank != 0")

    monkeypatch.setattr(
        cluster_artifacts, "install_or_validate_cluster_root_marker", _fail_if_called
    )

    def _broadcast_simulated_success(payload_list, src=0):
        assert payload_list == [None]

    monkeypatch.setattr(
        cluster_artifacts.dist, "broadcast_object_list", _broadcast_simulated_success
    )

    install_or_validate_cluster_root_marker_ddp(
        _cfg("ddp-rank1-ok"), _contract("a" * 64), rank=1, world_size=2
    )


# ============================================================================
# load_cluster_candidate_identity's three legitimate branches (NEW, owner-
# lens empty-return-or-raise table).
# ============================================================================


def test_load_candidate_identity_unclustered_no_run_tag_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag=None)
    assert load_cluster_candidate_identity(cfg) is None


def test_load_candidate_identity_unclustered_with_run_tag_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="unclustered-with-tag")
    assert load_cluster_candidate_identity(cfg) is None


def test_load_candidate_identity_clustered_without_run_tag_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag=None,
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
    )
    with pytest.raises(AxBenchBaseError, match="requires cfg.run_tag"):
        load_cluster_candidate_identity(cfg)


def test_load_candidate_identity_unclustered_rejects_stamped_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "unclustered-stamped"
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    cfg = make_valid_config(run_tag=run_tag)

    with pytest.raises(AxBenchBaseError, match="rejects cluster-stamped candidate artifacts"):
        load_cluster_candidate_identity(cfg)


# ============================================================================
# clustered_stage0c_fingerprint_fields empty-dict no-op (NEW, pinned).
# ============================================================================


def test_stage0c_fingerprint_fields_empty_dict_when_unclustered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="unclustered-fingerprint-fields")
    assert clustered_source_enabled(cfg) is False
    assert clustered_stage0c_fingerprint_fields(cfg) == {}


def test_clustered_source_enabled_true_false() -> None:
    clustered = make_valid_config(conditioning={"feature_clusters": {"path": "/c.yaml"}})
    unclustered_none = make_valid_config(conditioning=None)
    unclustered_missing = make_valid_config(conditioning={"feature_mask": {"mode": "whitelist"}})
    assert clustered_source_enabled(clustered) is True
    assert clustered_source_enabled(unclustered_none) is False
    assert clustered_source_enabled(unclustered_missing) is False


# ============================================================================
# Generation-marker tests (direct port, except the mutation test which is
# adapted, plus one NEW bug-fix regression test).
# ============================================================================


def test_clustered_stage4_checkpoint_guard_requires_initial_live_metadata() -> None:
    with pytest.raises(AxBenchBaseError, match="missing loaded checkpoint metadata"):
        cli._build_cluster_pre_forward_guard(
            _contract("a" * 64),
            SimpleNamespace(),
        )


def test_clustered_stage4_checkpoint_guard_is_one_shot_at_forward_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(_loaded_meta_data={"checkpoint": "metadata"})
    contract = _contract("a" * 64)
    calls: list[tuple[Any, Any, Any, str]] = []
    monkeypatch.setattr(
        cli,
        "assert_loaded_cluster_target",
        lambda resolved, loaded, metadata, *, surface: calls.append(
            (resolved, loaded, metadata, surface)
        ),
    )

    guard = cli._build_cluster_pre_forward_guard(contract, model)
    guard()
    guard()

    assert calls == [
        (
            contract,
            model,
            model._loaded_meta_data,
            "AxBench-Base Stage 4 encoder-decoder checkpoint immediately before first forward",
        )
    ]


def test_clustered_stage4_checkpoint_guard_reads_live_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_metadata = {"checkpoint": "original"}
    replacement_metadata = {"checkpoint": "replacement"}
    model = SimpleNamespace(_loaded_meta_data=original_metadata)
    contract = _contract("a" * 64)
    seen: list[Any] = []
    monkeypatch.setattr(
        cli,
        "assert_loaded_cluster_target",
        lambda _contract, _model, metadata, *, surface: seen.append(metadata),
    )

    guard = cli._build_cluster_pre_forward_guard(contract, model)
    model._loaded_meta_data = replacement_metadata
    guard()

    assert seen == [replacement_metadata]


def test_clustered_stage4_checkpoint_guard_rejects_lost_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(_loaded_meta_data={"checkpoint": "metadata"})
    guard = cli._build_cluster_pre_forward_guard(_contract("a" * 64), model)
    model._loaded_meta_data = None
    monkeypatch.setattr(
        cli,
        "assert_loaded_cluster_target",
        lambda *_args, **_kwargs: pytest.fail(
            "validation must not receive missing metadata"
        ),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="lost.*metadata.*before first forward",
    ):
        guard()


def test_clustered_generation_marker_commits_only_exact_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("clustered-generation")
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts("clustered-generation")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64,
        concepts_sha256=concepts_sha256,
    )
    checkpoint_dir = tmp_path / "same-logical-decoder-ref"
    checkpoint_dir.mkdir()
    model_path = checkpoint_dir / "model_000009.pt"
    (checkpoint_dir / "meta_000009.json").write_text("{}", encoding="utf-8")
    t.save({"weight": t.tensor([1.0])}, model_path)
    pin_a, _metadata, _state = (
        cluster_checkpoint_pin_module.resolve_checkpoint_files_pin(
            checkpoint_dir,
            step=None,
            resolved_tag="d-same",
        )
    )
    generation_inputs["checkpoint_refs"] = {
        "decoder_only": {
            "source": "base",
            "model_tag": "d-same",
            "step": pin_a.step,
            "loaded_checkpoint": pin_a.semantic_dict(),
        }
    }

    validate_cluster_generation_resume(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    commit_cluster_generation_marker(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    marker = test_paths.default_paths("clustered-generation").generations_feature_contract_json()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 2,
        "feature_contract_fingerprint": "a" * 64,
        "contexts_fingerprint": "d" * 64,
        "candidate_concepts_sha256": "b" * 64,
        "candidate_cluster_provenance_sha256": "c" * 64,
        "cluster_concept_projection_version": "cluster_concept_projection_v1",
        "concepts_sha256": concepts_sha256,
        "generation_inputs_fingerprint": hashlib.sha256(
            json.dumps(
                generation_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest(),
        "generation_inputs": generation_inputs,
    }

    validate_cluster_generation_resume(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    t.save({"weight": t.tensor([2.0])}, model_path)
    pin_b, _metadata, _state = (
        cluster_checkpoint_pin_module.resolve_checkpoint_files_pin(
            checkpoint_dir,
            step=None,
            resolved_tag="d-same",
        )
    )
    changed_checkpoint_inputs = copy.deepcopy(generation_inputs)
    changed_checkpoint_inputs["checkpoint_refs"]["decoder_only"][
        "loaded_checkpoint"
    ] = pin_b.semantic_dict()
    assert pin_a.step == pin_b.step == 9
    assert pin_a.model_file.sha256 != pin_b.model_file.sha256
    with pytest.raises(AxBenchBaseError, match="resume identity differs"):
        validate_cluster_generation_resume(
            cfg,
            contract,
            contexts_fingerprint="d" * 64,
            generation_inputs=changed_checkpoint_inputs,
        )
    with pytest.raises(AxBenchBaseError, match="resume identity differs"):
        validate_cluster_generation_resume(
            cfg,
            _contract("e" * 64),
            contexts_fingerprint="d" * 64,
            generation_inputs={**generation_inputs, "feature_contract_fingerprint": "e" * 64},
        )
    with pytest.raises(AxBenchBaseError, match="resume identity differs"):
        validate_cluster_generation_resume(
            cfg, contract, contexts_fingerprint="f" * 64, generation_inputs=generation_inputs
        )
    with pytest.raises(AxBenchBaseError, match="resume identity differs"):
        validate_cluster_generation_resume(
            cfg,
            contract,
            contexts_fingerprint="d" * 64,
            generation_inputs={**generation_inputs, "changed": True},
        )


def test_clustered_generation_rejects_marker_mutation_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADAPTED from the oracle (Finding 1 bug fix -- see module docstring):
    the oracle faked TWO successive ``stream_hash_file_stable`` return
    values directly (its inline block called it twice, before/after). This
    file's ``validate_cluster_generation_resume`` now calls ``_load_json_
    mapping`` (one hash call, resolved via ``cluster_checkpoint_pin``'s own
    namespace, not this module's) -- the equivalent injection wraps that
    single call to mutate the REAL file right after it returns the digest
    of the original content, before the subsequent read.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("mutating-generation-marker")
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts("mutating-generation-marker")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256=concepts_sha256
    )
    commit_cluster_generation_marker(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    marker = (
        test_paths.default_paths("mutating-generation-marker")
        .generations_feature_contract_json()
        .resolve()
    )
    real_hash = cluster_checkpoint_pin_module.stream_hash_file_stable

    def _mutate_after_marker_hash(path: str | Path):
        resolved = Path(path).resolve()
        result = real_hash(path)
        if resolved == marker:
            marker.write_bytes(marker.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        cluster_checkpoint_pin_module, "stream_hash_file_stable", _mutate_after_marker_hash
    )

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        validate_cluster_generation_resume(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )


def test_clustered_generation_rejects_marker_swap_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: reject a persistent Stage-4 marker swap staged after its read."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    run_tag = "generation-marker-swapped-after-read"
    cfg = _cfg(run_tag)
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts(run_tag)
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64,
        concepts_sha256=concepts_sha256,
    )
    commit_cluster_generation_marker(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    marker = test_paths.default_paths(run_tag).generations_feature_contract_json().resolve()
    real_read_bytes = Path.read_bytes

    def _swap_after_read(self: Path, *args: Any, **kwargs: Any) -> bytes:
        payload = real_read_bytes(self, *args, **kwargs)
        if self.resolve() == marker:
            marker.write_bytes(payload + b" ")
        return payload

    monkeypatch.setattr(Path, "read_bytes", _swap_after_read)

    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        validate_cluster_generation_resume(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )


def test_clustered_generation_rejects_marker_swap_during_expected_payload_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject persistent valid marker B installed after the post-read snapshot.

    The expected-payload builder returns A so the in-memory equality check
    succeeds, but installs a marker that is valid for contexts fingerprint B.
    The old function had no binding check after this window and returned while
    that unvalidated B remained live.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    run_tag = "generation-marker-swapped-during-payload-rebuild"
    cfg = _cfg(run_tag)
    contract = _contract("a" * 64)
    contexts_a = "d" * 64
    contexts_b = "e" * 64
    concepts_sha256 = _write_concepts(run_tag)
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64,
        concepts_sha256=concepts_sha256,
    )
    commit_cluster_generation_marker(
        cfg,
        contract,
        contexts_fingerprint=contexts_a,
        generation_inputs=generation_inputs,
    )
    marker = test_paths.default_paths(run_tag).generations_feature_contract_json()
    real_payload = cluster_artifacts._cluster_generation_marker_payload
    swap_performed = False

    def _build_a_then_install_b(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal swap_performed
        expected_a = real_payload(*args, **kwargs)
        marker_b = {**expected_a, "contexts_fingerprint": contexts_b}
        marker.write_text(json.dumps(marker_b), encoding="utf-8")
        swap_performed = True
        return expected_a

    monkeypatch.setattr(
        cluster_artifacts,
        "_cluster_generation_marker_payload",
        _build_a_then_install_b,
    )

    with pytest.raises(AxBenchBaseError, match="changed while it was being validated"):
        validate_cluster_generation_resume(
            cfg,
            contract,
            contexts_fingerprint=contexts_a,
            generation_inputs=generation_inputs,
        )

    assert swap_performed
    assert json.loads(marker.read_text(encoding="utf-8"))["contexts_fingerprint"] == contexts_b

    # B is not merely parseable: with the injection removed, it is a valid
    # marker for its own contexts fingerprint and exact generation inputs.
    monkeypatch.setattr(
        cluster_artifacts,
        "_cluster_generation_marker_payload",
        real_payload,
    )
    validate_cluster_generation_resume(
        cfg,
        contract,
        contexts_fingerprint=contexts_b,
        generation_inputs=generation_inputs,
    )


def test_clustered_generation_does_not_post_hash_after_mapping_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5: marker mapping failure keeps its wrapper and skips local hashing."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    run_tag = "generation-marker-mapping-failure"
    cfg = _cfg(run_tag)
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts(run_tag)
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64,
        concepts_sha256=concepts_sha256,
    )
    commit_cluster_generation_marker(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    marker = test_paths.default_paths(run_tag).generations_feature_contract_json()
    post_hash_calls: list[Path] = []

    def _mapping_failure(*_args: Any, **_kwargs: Any):
        raise RuntimeError("injected mapping failure")

    def _spy_hash(path: str | Path):
        post_hash_calls.append(Path(path))
        raise AssertionError("post-read hash must not run after mapping failure")

    monkeypatch.setattr(cluster_artifacts, "_load_json_mapping", _mapping_failure)
    monkeypatch.setattr(cluster_artifacts, "stream_hash_file_stable", _spy_hash)

    with pytest.raises(AxBenchBaseError) as exc_info:
        validate_cluster_generation_resume(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )

    assert str(exc_info.value) == (
        f"Unreadable clustered generation marker {marker}: injected mapping failure"
    )
    assert post_hash_calls == []


def test_clustered_generation_resume_catches_stale_digest_disagreeing_with_real_bytes_finding1_bugfix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The filesystem-independent "closes a real gap" proof for Finding 1's
    bug-fix deviation.

    A real-filesystem replay of an A->B->A content round-trip does NOT
    isolate the fix: two real ``stream_hash_file_stable`` calls bracketing
    even a single intervening write pick up real mtime_ns/ctime_ns
    advancement from that write, so ``marker_before != marker_after``
    fires via incidental timestamp drift regardless of which pattern is
    used -- the oracle's old before/after-only check would ALSO raise on
    such a replay, just for the wrong reason, and the test would not
    actually distinguish old from new.

    The real, deterministic distinguishing property is this: the oracle's
    inline TOCTOU bracket in ``validate_cluster_generation_resume`` ONLY
    ever compared two ``stream_hash_file_stable`` SNAPSHOTS for equality
    (``marker_before != marker_after``) -- it never cross-checked the
    bytes it went on to parse against either snapshot's own recorded
    hash. Simulate a ``stream_hash_file_stable`` that returns the exact
    same (stale) digest object on every call -- i.e. a coarse-timestamp
    filesystem where two snapshots bracketing an intervening write would
    still compare equal -- whose ``.sha256`` simply disagrees with the
    real bytes on disk. Under that single, fully-controlled input:

    1. The real code (via the fused ``_load_json_mapping`` reuse) still
       raises: it cross-checks the bytes it actually reads against the
       digest's ``.sha256`` directly, independent of any second snapshot.
    2. An inline reference implementation of the oracle's OWN before/
       after-only check -- reconstructed verbatim from oracle:660-667,
       not merely reasoned about -- is replayed against the identical
       monkeypatched calls and does NOT raise (no ``pytest.raises``
       wrapper: if it raised, this test would itself fail at that line).
       That is the genuine old-vs-new discriminator: this exact input is
       silently missed by the old pattern, which never looks at the
       actual read bytes at all.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("stale-digest-generation-marker")
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts("stale-digest-generation-marker")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256=concepts_sha256
    )
    commit_cluster_generation_marker(
        cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    marker = (
        test_paths.default_paths("stale-digest-generation-marker")
        .generations_feature_contract_json()
        .resolve()
    )
    real_bytes = marker.read_bytes()
    real_sha256 = hashlib.sha256(real_bytes).hexdigest()

    # One fixed "stale" digest, returned IDENTICALLY on every call: its
    # .sha256 deliberately disagrees with the real on-disk bytes, but a
    # before/after SNAPSHOT comparison (the oracle's own check) sees no
    # difference at all -- by construction, not by relying on real
    # filesystem timestamp granularity. Duck-typed (SimpleNamespace, not a
    # real StableFileDigest): _load_json_mapping only ever touches
    # .path/.byte_length/.sha256, matching this file's existing precedent
    # for duck-typed stand-ins (e.g. _contract()).
    stale_sha256 = "0" * 64
    assert stale_sha256 != real_sha256
    stale_digest = SimpleNamespace(path=marker, sha256=stale_sha256, byte_length=len(real_bytes))

    def _always_return_stale_digest(_path: str | Path):
        return stale_digest

    monkeypatch.setattr(
        cluster_checkpoint_pin_module, "stream_hash_file_stable", _always_return_stale_digest
    )

    # 1. The real code raises: _load_json_mapping cross-checks the bytes it
    # actually reads (real_bytes) against stale_digest.sha256, which
    # deliberately disagrees.
    with pytest.raises(AxBenchBaseError, match="changed while it was being loaded"):
        validate_cluster_generation_resume(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )

    # 2. Inline reference implementation of the ORACLE'S OWN before/after-
    # only check (validate_cluster_generation_resume's inline block,
    # oracle:660-667 -- not _load_stable_json's separately-named, STRONGER
    # pattern, which already cross-checks payload_sha256 against
    # before.sha256 and is not what Finding 1's bug-fix targets), replayed
    # against the identical monkeypatched stream_hash_file_stable calls.
    # No pytest.raises wrapper: if this raised, the test would itself fail
    # right here. It must NOT raise -- proving this exact stale-digest-vs-
    # real-bytes disagreement is silently missed by the old pattern, which
    # never looks at the actual read bytes at all.
    def _oracle_before_after_only_check(path: Path) -> None:
        marker_before = cluster_checkpoint_pin_module.stream_hash_file_stable(path)
        _marker_bytes = path.read_bytes()  # the old pattern's own unprotected middle read
        marker_after = cluster_checkpoint_pin_module.stream_hash_file_stable(path)
        if marker_before != marker_after:
            raise AxBenchBaseError(
                "Clustered generation resume marker changed while it was being loaded"
            )

    _oracle_before_after_only_check(marker)


def test_clustered_generation_commit_rejects_conflicting_race_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("raced-generation-marker")
    contract = _contract("a" * 64)
    concepts_sha256 = _write_concepts("raced-generation-marker")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256=concepts_sha256
    )
    marker = test_paths.default_paths("raced-generation-marker").generations_feature_contract_json()
    real_validate = cluster_artifacts.validate_cluster_generation_resume

    def validate_then_install_conflict(*args: Any, **kwargs: Any) -> None:
        real_validate(*args, **kwargs)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"{}")

    monkeypatch.setattr(
        cluster_artifacts, "validate_cluster_generation_resume", validate_then_install_conflict
    )

    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        commit_cluster_generation_marker(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )
    assert marker.read_bytes() == b"{}"


def test_clustered_generation_rejects_orphan_rows_and_unclustered_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("orphan-generation")
    concepts_sha256 = _write_concepts("orphan-generation")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256=concepts_sha256
    )
    generations = test_paths.default_paths("orphan-generation").generations_dir()
    generations.mkdir(parents=True)
    (generations / "batch_00000.parquet").write_bytes(b"orphan")

    with pytest.raises(AxBenchBaseError, match="lack.*resume marker"):
        validate_cluster_generation_resume(
            cfg,
            _contract("a" * 64),
            contexts_fingerprint="d" * 64,
            generation_inputs=generation_inputs,
        )

    (generations / "batch_00000.parquet").unlink()
    commit_cluster_generation_marker(
        cfg, _contract("a" * 64), contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
    )
    unclustered = make_valid_config(run_tag="orphan-generation")
    with pytest.raises(AxBenchBaseError, match="rejects cluster-stamped"):
        validate_cluster_generation_resume(unclustered, None, contexts_fingerprint=None)


@pytest.mark.parametrize("fingerprint", [None, "", "D" * 64, "d" * 63, "z" * 64])
def test_clustered_generation_requires_canonical_context_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fingerprint: str | None,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="lowercase SHA-256"):
        validate_cluster_generation_resume(
            _cfg("invalid-context-fingerprint"),
            _contract("a" * 64),
            contexts_fingerprint=fingerprint,
            generation_inputs={},
        )


def test_commit_cluster_generation_marker_is_noop_when_unclustered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="unclustered-commit")
    commit_cluster_generation_marker(cfg, None, contexts_fingerprint=None)
    marker = test_paths.default_paths("unclustered-commit").generations_feature_contract_json()
    assert not marker.exists()


# ============================================================================
# Import-path preservation (constraint 4).
# ============================================================================

# Union of the oracle's own __all__ (12 names) and the 14-consumer symbol
# table from the plan's constraint 4 (adds CLUSTER_CANDIDATE_PROVENANCE_
# SCHEMA_VERSION and validate_cluster_candidate_provenance_semantics, both
# omitted from the oracle's own __all__ despite being imported directly by
# real consumers). None of the 14 documented consumer files import from
# cluster_artifacts on current main today -- this freezes the oracle-
# branch/future-track surface the plan's design spec draws, not a current-
# main call-site fact.
_EXPECTED_PUBLIC_SYMBOLS = frozenset(
    {
        "ClusterCandidateIdentity",
        "CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION",
        "CLUSTER_GENERATION_RESUME_SCHEMA_VERSION",
        "clustered_source_enabled",
        "clustered_stage0c_fingerprint_fields",
        "commit_cluster_generation_marker",
        "install_clustered_bytes_once",
        "install_or_validate_cluster_root_marker",
        "install_or_validate_cluster_root_marker_ddp",
        "load_cluster_candidate_identity",
        "validate_cluster_candidate_provenance_semantics",
        "validate_cluster_generation_resume",
        "write_clustered_parquet_once",
        "write_clustered_safetensors_once",
    }
)


def test_all_documented_consumer_symbols_resolve_by_direct_import() -> None:
    module = importlib.import_module("steering.eval.axbench_base.cluster_artifacts")
    missing = [name for name in sorted(_EXPECTED_PUBLIC_SYMBOLS) if not hasattr(module, name)]
    assert not missing, f"symbols do not resolve from cluster_artifacts: {missing}"
    for name in _EXPECTED_PUBLIC_SYMBOLS:
        # Mirrors `from steering.eval.axbench_base.cluster_artifacts import name`
        # -- __all__ is not trusted alone (constraint 4): it was itself
        # incomplete on the oracle.
        getattr(module, name)
    missing_from_all = _EXPECTED_PUBLIC_SYMBOLS - set(module.__all__)
    assert not missing_from_all, f"missing from __all__: {missing_from_all}"


# ============================================================================
# Raise-site completeness (NEW): every remaining AxBenchBaseError raise site
# not already exercised above, found by an exhaustive manual trace against
# the full test suite (recorded in task-c1-report.md's raise-site map).
# ============================================================================


def _valid_baseline_triple() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """A fully self-consistent ``(document, provenance, candidate_rows)``
    triple that ``validate_cluster_candidate_provenance_semantics`` accepts
    completely unmodified -- the shared baseline the structural-corruption
    matrix below mutates exactly one thing away from, so each case is known
    to trip its OWN specific check rather than an earlier, unrelated one.
    """
    document = _candidate_contract_document()
    provenance = {
        "counts": {
            "raw_source_ids": 2,
            "mask_allowed_source_ids": 2,
            "unique_representatives": 2,
            "representatives_with_statistics": 2,
            "count_survivors": 2,
            "description_fetch_survivors": 2,
            "described_survivors": 2,
        },
        "representatives": [
            {
                "representative_id": 1,
                "allowed_member_ids": [1, 2],
                "projecting_source_ids": [2],
                "statistics_source_id": 1,
                "outcome": "candidate_emitted",
            },
            {
                "representative_id": 5,
                "allowed_member_ids": [5],
                "projecting_source_ids": [5],
                "statistics_source_id": 5,
                "outcome": "candidate_emitted",
            },
        ],
    }
    candidate_rows: list[dict[str, Any]] = [{"feature_index": 1}, {"feature_index": 5}]
    return document, provenance, candidate_rows


def test_baseline_triple_is_actually_valid() -> None:
    """Sanity anchor for the corruption matrix below: the shared baseline
    itself must be accepted before any test can trust that a corrupted
    variant's raise is attributable to the ONE thing it changed.
    """
    document, provenance, candidate_rows = _valid_baseline_triple()
    cluster_artifacts.validate_cluster_candidate_provenance_semantics(
        feature_contract=document, candidate_rows=candidate_rows, provenance=provenance
    )


def test_validate_cluster_candidate_provenance_semantics_rejects_empty_candidate_set_when_all_excluded() -> (
    None
):
    """Codex cross-review Finding 2 (HIGH): when every representative is
    terminally excluded, ``candidate_ids == emitted_ids == set()`` and the
    resulting zero-survivor counts are self-consistent -- without this
    check, the validator accepts an EMPTY committed candidate set. That is
    impossible from the honest producer: ``max_concepts >= 1`` is enforced
    at config validation time (``config.py``, "max_concepts must be >= 1"),
    and the oracle's own Stage 0a producer refuses to commit an
    underfilled candidate population before ever writing these artifacts
    (``concept_list.py``'s ``_raise_clustered_underfill``: "Clustered
    Stage 0a cannot supply the requested concept population"). By the
    point this check runs, ``representative_universe`` is already
    confirmed non-empty (checked earlier in this same function) -- an
    empty candidate set here is anomalous, not a legitimate degenerate
    case, and the fail-loud mandate requires rejecting it rather than
    silently accepting a self-consistent-but-impossible sidecar.

    Exact Codex-verified scenario: a non-empty two-representative contract
    (the shared baseline) with BOTH representatives terminally excluded
    via ``missing_representative_statistics``, an empty committed
    candidate list, and counts recomputed to the genuinely
    self-consistent all-zero-survivors shape.
    """
    document, provenance, _candidate_rows = _valid_baseline_triple()
    for row in provenance["representatives"]:
        row["outcome"] = "missing_representative_statistics"
    provenance["counts"] = {
        "raw_source_ids": 2,
        "mask_allowed_source_ids": 2,
        "unique_representatives": 2,
        "representatives_with_statistics": 0,
        "count_survivors": 0,
        "description_fetch_survivors": 0,
        "described_survivors": 0,
    }
    empty_candidate_rows: list[dict[str, Any]] = []

    # Sanity: this scenario is genuinely self-consistent up to the new
    # check -- candidate_ids and emitted_ids both compute to set(), so the
    # pre-existing "does not exactly match candidate_emitted" check does
    # NOT fire on its own; only the dedicated empty-candidate-set guard
    # catches this.
    with pytest.raises(AxBenchBaseError, match="committed an empty candidate set"):
        cluster_artifacts.validate_cluster_candidate_provenance_semantics(
            feature_contract=document, candidate_rows=empty_candidate_rows, provenance=provenance
        )


def _corrupt_semantic_not_a_mapping(document, _provenance, _candidate_rows):
    document["semantic"] = "not-a-mapping"


def _corrupt_clusters_or_concept_source_not_a_mapping(document, _provenance, _candidate_rows):
    document["semantic"]["feature_clusters"] = None


def _corrupt_members_field_not_a_list(document, _provenance, _candidate_rows):
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = "nope"


def _corrupt_member_row_shape(document, _provenance, _candidate_rows):
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = [
        [1, [1, 2]],
        [5],
    ]


def _corrupt_member_row_internal_consistency(document, _provenance, _candidate_rows):
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = [
        [1, [2, 3]],
        [5, [5]],
    ]


def _corrupt_member_coverage_of_universe(document, _provenance, _candidate_rows):
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = [[1, [1, 2]]]


def _corrupt_concept_source_mode(document, _provenance, _candidate_rows):
    document["semantic"]["concept_source"]["mode"] = "bogus_mode"


def _corrupt_source_projection_not_a_list(document, _provenance, _candidate_rows):
    document["semantic"]["concept_source"]["source_to_representative"] = "nope"


def _corrupt_source_projection_row(document, _provenance, _candidate_rows):
    document["semantic"]["concept_source"]["source_to_representative"] = [[2, 1], [99, 5]]


def _corrupt_source_projection_not_sorted(document, _provenance, _candidate_rows):
    document["semantic"]["concept_source"]["source_to_representative"] = [[5, 5], [2, 1]]


def _corrupt_whitelist_masked_projection(document, _provenance, _candidate_rows):
    document["semantic"]["concept_source"]["source_to_representative"] = [[2, 1]]


def _corrupt_provenance_row_keyset(_document, provenance, _candidate_rows):
    del provenance["representatives"][0]["statistics_source_id"]


def _corrupt_candidate_rows_duplicate_ids(_document, _provenance, candidate_rows):
    candidate_rows[1] = {"feature_index": 1}


def _corrupt_candidate_rows_mismatch_emitted(_document, _provenance, candidate_rows):
    candidate_rows.pop()


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (_corrupt_semantic_not_a_mapping, "lacks semantic content"),
        (_corrupt_clusters_or_concept_source_not_a_mapping, "lacks cluster or concept-source"),
        (_corrupt_members_field_not_a_list, "lacks representative member semantics"),
        (_corrupt_member_row_shape, "malformed representative members"),
        (_corrupt_member_row_internal_consistency, "inconsistent representative members"),
        (_corrupt_member_coverage_of_universe, "does not cover its representative universe"),
        (_corrupt_concept_source_mode, "malformed concept-source semantics"),
        (_corrupt_source_projection_not_a_list, "source projection must be a list"),
        (_corrupt_source_projection_row, "malformed source projection"),
        (_corrupt_source_projection_not_sorted, "sorted unique IDs"),
        (_corrupt_whitelist_masked_projection, "masked or missing projection"),
        (_corrupt_provenance_row_keyset, "representative row is malformed"),
        (_corrupt_candidate_rows_duplicate_ids, "feature IDs must be unique"),
        (_corrupt_candidate_rows_mismatch_emitted, "does not exactly match candidate_emitted"),
    ],
)
def test_validate_cluster_candidate_provenance_semantics_raise_site_matrix(
    corrupt, match: str
) -> None:
    """Raise-site completeness: each case corrupts exactly ONE thing away
    from the shared valid baseline and confirms the validator rejects it
    with the specific message for THAT check (not some other, unrelated
    check firing first) -- closes 14 raise sites the mutation/acceptance
    tests above do not otherwise exercise (a 15th, candidate_rows itself
    not being a list, is its own standalone test right below: it needs a
    non-list REPLACEMENT value, not an in-place list mutation).
    """
    document, provenance, candidate_rows = _valid_baseline_triple()
    corrupt(document, provenance, candidate_rows)

    with pytest.raises(AxBenchBaseError, match=match):
        cluster_artifacts.validate_cluster_candidate_provenance_semantics(
            feature_contract=document, candidate_rows=candidate_rows, provenance=provenance
        )


def test_validate_cluster_candidate_provenance_semantics_rejects_non_list_candidate_rows() -> None:
    document, provenance, _candidate_rows = _valid_baseline_triple()

    with pytest.raises(AxBenchBaseError, match="clustered candidate_concepts.json must be a list"):
        cluster_artifacts.validate_cluster_candidate_provenance_semantics(
            feature_contract=document,
            candidate_rows={"not": "a list"},
            provenance=provenance,
        )


def test_sha256_file_wraps_underlying_hash_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise-site completeness: ``_sha256_file`` (used for ``concepts.json``
    inside ``_cluster_generation_marker_payload``) wraps any failure as
    ``AxBenchBaseError`` -- triggered here by never writing ``concepts.json``
    at all before a marker commit needs to hash it.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("missing-concepts-json")
    contract = _contract("a" * 64)
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256="0" * 64
    )

    with pytest.raises(AxBenchBaseError, match="Could not stably hash clustered AxBench artifact"):
        commit_cluster_generation_marker(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )


def test_load_stable_json_wraps_missing_candidate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise-site completeness: ``_load_stable_json``'s generic read-failure
    wrap (candidate_concepts.json missing entirely, not merely mutated
    mid-load). Starts from a fully valid provenance+candidate pair (so
    every check upstream of the candidate-concepts read passes) and then
    deletes ONLY the candidate file.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-json-missing"
    _write_candidate_artifacts(run_tag)
    test_paths.default_paths(run_tag).candidate_concepts_json().unlink()
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    with pytest.raises(
        AxBenchBaseError, match="Could not stably read clustered candidate concepts"
    ):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_stable_json_wraps_invalid_json_candidate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise-site completeness: ``_load_stable_json``'s invalid-JSON branch
    for ``candidate_concepts.json``. Starts from a fully valid provenance+
    candidate pair and then overwrites ONLY the candidate file's bytes with
    invalid JSON (the provenance's now-stale recorded sha256 is never
    reached: ``_load_stable_json`` raises on the JSON parse first).
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-json-invalid"
    _write_candidate_artifacts(run_tag)
    test_paths.default_paths(run_tag).candidate_concepts_json().write_text(
        "{not valid json", encoding="utf-8"
    )
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )

    with pytest.raises(AxBenchBaseError, match="is not valid JSON"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_wraps_feature_contract_sidecar_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise-site completeness: ``load_cluster_candidate_identity``'s own
    ``load_feature_contract_sidecar`` wrap (distinct from ``install_or_
    validate_cluster_root_marker``'s own, separate wrap of the same
    underlying function) -- triggered by a marker that is present but not
    valid JSON.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-marker-invalid"
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("not json at all", encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="Invalid clustered feature-contract marker"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_rejects_provenance_wrong_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-wrong-fields"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    raw = json.loads(
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().read_text()
    )
    raw["extra_unexpected_field"] = True
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(raw)
    )

    with pytest.raises(AxBenchBaseError, match="wrong fields"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_rejects_unsupported_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-bad-schema"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    raw = json.loads(
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().read_text()
    )
    raw["schema_version"] = 999
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(raw)
    )

    with pytest.raises(AxBenchBaseError, match="unsupported clustered candidate provenance schema"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_rejects_fingerprint_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-fingerprint-mismatch"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    raw = json.loads(
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().read_text()
    )
    raw["feature_contract_fingerprint"] = "f" * 64
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(raw)
    )

    with pytest.raises(AxBenchBaseError, match="root feature contract disagree"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_rejects_wrong_concept_projection_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "provenance-bad-projection-version"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    raw = json.loads(
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().read_text()
    )
    raw["concept_projection_version"] = "some_other_version_v2"
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(raw)
    )

    with pytest.raises(AxBenchBaseError, match="wrong concept-projection version"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_load_candidate_identity_rejects_candidate_bytes_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise-site completeness: the recorded ``candidate_concepts_sha256``
    disagrees with the actual (unmutated, statically stale) file bytes --
    distinct from the mutation-while-loading race tests above, which mutate
    the file DURING the read window rather than starting from an already-
    stale recorded hash.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "candidate-bytes-stale"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts,
        "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    raw = json.loads(
        test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().read_text()
    )
    raw["candidate_concepts_sha256"] = "0" * 64
    test_paths.default_paths(run_tag).candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(raw)
    )

    with pytest.raises(AxBenchBaseError, match="does not match candidate_concepts.json bytes"):
        load_cluster_candidate_identity(_cfg(run_tag), expected_fingerprint="a" * 64)


def test_generation_marker_payload_requires_committed_candidate_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("generation-needs-candidate")
    contract = _contract("a" * 64)
    _write_concepts("generation-needs-candidate")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256="0" * 64
    )
    # No _install_candidate_identity monkeypatch and no real candidate
    # artifacts on disk -- load_cluster_candidate_identity's own real
    # unclustered-branch logic returns None (cfg is clustered via _cfg, so
    # it instead hits its own "requires cfg.run_tag" style checks)... use
    # an explicit monkeypatch returning None to isolate exactly this check.
    monkeypatch.setattr(
        cluster_artifacts, "load_cluster_candidate_identity", lambda *_args, **_kwargs: None
    )

    with pytest.raises(AxBenchBaseError, match="requires committed candidate provenance"):
        commit_cluster_generation_marker(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )


def test_generation_marker_payload_requires_run_tag_direct_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise-site completeness: ``_cluster_generation_marker_payload``'s own
    ``cfg.run_tag is None`` check is unreachable through any real call path
    with an unmocked ``load_cluster_candidate_identity`` -- not only do its
    two real callers (``validate_cluster_generation_resume``/``commit_
    cluster_generation_marker``) already raise their own, earlier "requires
    cfg.run_tag" before ever calling it, but ``load_cluster_candidate_
    identity`` itself (called as this function's very first statement) also
    raises its OWN "requires cfg.run_tag" first whenever cfg.run_tag is
    None and the config is clustered. Ported byte-identical from the oracle
    regardless (not a Finding 1/2/3 site) -- this test mocks ``load_
    cluster_candidate_identity`` to artificially return a valid identity
    despite the missing run_tag, isolating this one specific defensive
    check to confirm it still fires correctly if it were ever reached.
    """
    _install_candidate_identity(monkeypatch)
    cfg = make_valid_config(
        run_tag=None, conditioning={"feature_clusters": {"path": "/clusters.yaml"}}
    )
    contract = _contract("a" * 64)

    with pytest.raises(AxBenchBaseError, match="cluster generation identity requires cfg.run_tag"):
        cluster_artifacts._cluster_generation_marker_payload(
            cfg,
            contract,
            contexts_fingerprint="d" * 64,
            generation_inputs={"feature_contract_fingerprint": "a" * 64},
        )


def test_generation_marker_payload_requires_finite_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("generation-nan-inputs")
    contract = _contract("a" * 64)
    _write_concepts("generation-nan-inputs")

    with pytest.raises(AxBenchBaseError, match="must be finite canonical JSON"):
        cluster_artifacts._cluster_generation_marker_payload(
            cfg,
            contract,
            contexts_fingerprint="d" * 64,
            generation_inputs={"feature_contract_fingerprint": "a" * 64, "bad": float("nan")},
        )


def test_generation_marker_payload_requires_nonempty_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("generation-empty-inputs-commit")
    contract = _contract("a" * 64)

    with pytest.raises(AxBenchBaseError, match="must be a nonempty mapping"):
        commit_cluster_generation_marker(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs={}
        )


def test_generation_marker_payload_requires_matching_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("generation-mismatched-fingerprint")
    contract = _contract("a" * 64)

    with pytest.raises(AxBenchBaseError, match="do not match the feature contract"):
        commit_cluster_generation_marker(
            cfg,
            contract,
            contexts_fingerprint="d" * 64,
            generation_inputs={"feature_contract_fingerprint": "b" * 64},
        )


def test_generation_marker_payload_requires_matching_concepts_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    cfg = _cfg("generation-mismatched-concepts-bytes")
    contract = _contract("a" * 64)
    _write_concepts("generation-mismatched-concepts-bytes")
    generation_inputs = _generation_inputs(
        feature_contract_fingerprint="a" * 64, concepts_sha256="0" * 64
    )

    with pytest.raises(AxBenchBaseError, match="do not match final concepts bytes"):
        commit_cluster_generation_marker(
            cfg, contract, contexts_fingerprint="d" * 64, generation_inputs=generation_inputs
        )


def test_generation_resume_requires_run_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_valid_config(
        run_tag=None, conditioning={"feature_clusters": {"path": "/clusters.yaml"}}
    )
    with pytest.raises(AxBenchBaseError, match="Stage 4 generation marker requires cfg.run_tag"):
        validate_cluster_generation_resume(
            cfg, _contract("a" * 64), contexts_fingerprint="d" * 64, generation_inputs={}
        )


def test_generation_resume_requires_generation_inputs_not_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _cfg("generation-inputs-none")

    with pytest.raises(AxBenchBaseError, match="requires exact generation-input provenance"):
        validate_cluster_generation_resume(
            cfg, _contract("a" * 64), contexts_fingerprint="d" * 64, generation_inputs=None
        )


# ============================================================================
# assert_live_statistics_identity / assert_live_statistics_match_committed_
# marker / validate_cluster_candidate_identity_with_collective_agreement
# (PR #739 FU-D-stage0c fix batch -- Codex cross-review of C4: Bug 1 HIGH,
# resume/merge never re-validated live statistics against the contract's
# pinned identity; Bug 2 MEDIUM, the per-rank cluster-sidecar check ran
# before the DDP collective and could deadlock on rank skew).
# ============================================================================


def _stats_feature_dir(tmp_path: Path, *, name: str = "features") -> Path:
    """A real, canonically-named feature directory (no statistics file yet)."""
    feature_dir = build_feature_dir(
        tmp_path / name,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=False,
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir


def _write_versioned_stats(
    feature_dir: Path,
    *,
    n_sae_features: int = 16_384,
    feature_ids: tuple[int, ...] = (0, 1, 2),
    max_mags: dict[int, float],
) -> Path:
    """Write a real, self-consistent ``feature_statistics.safetensors``
    (valid stored percentile checksum) under ``feature_dir``. Calling this
    twice writes/overwrites the SAME final path -- ``atomic_safetensors_
    save`` is an unconditional tmp+fsync+``os.replace``, not an install-
    once primitive -- so this is how the tests below genuinely swap the
    on-disk bytes at a fixed path, mirroring the plan's exact attack
    scenario (same width/domain, different content).
    """
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags=dict(max_mags),
        absent=[i for i in range(n_sae_features) if i not in feature_ids],
    )
    stats = load_statistics(statistics_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, checksum)
    return statistics_path


def _stats_cfg(run_tag: str, *, feature_dir: Path, statistics_path: Path) -> Any:
    return make_valid_config(
        run_tag=run_tag,
        feature_dir=str(feature_dir),
        feature_statistics_path=str(statistics_path),
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def _write_marker_with_statistics(run_tag: str, *, pinned_statistics: dict[str, Any]) -> None:
    """Hand-write a self-consistent ``feature_contract.json`` root marker
    whose ``semantic.statistics`` block is ``pinned_statistics`` -- enough
    to satisfy ``load_feature_contract_sidecar``'s own schema/fingerprint/
    E4-non-emptiness checks without resolving a full checkpoint/SAE-backed
    ``ClusterFeatureContract``.
    """
    semantic = {
        "statistics": pinned_statistics,
        "concept_source": {
            "mode": "statistics_domain",
            "domain": [0],
            "source_to_representative": [[0, 0]],
        },
    }
    document = {
        "schema_version": 1,
        "feature_contract_fingerprint": _sha256_json(semantic),
        "semantic": semantic,
        "provenance": {},
    }
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(document), encoding="utf-8")


def _fake_contract_with_statistics(fingerprint: str, *, statistics_dict: dict[str, Any]) -> Any:
    """Minimal duck-typed ``ClusterFeatureContract`` stand-in exposing only
    the two attributes ``validate_cluster_candidate_identity_with_
    collective_agreement`` reads -- mirrors this file's own ``_contract``
    helper, extended with a real ``.statistics.canonical_dict()``.
    """
    return cast(
        Any,
        SimpleNamespace(
            feature_contract_fingerprint=fingerprint,
            statistics=SimpleNamespace(canonical_dict=lambda: dict(statistics_dict)),
        ),
    )


# ---- assert_live_statistics_identity ----


def test_assert_live_statistics_identity_noop_when_unclustered(tmp_path: Path) -> None:
    """U-invariant: unconditional no-op when clustering is disabled, even
    with a nonsense pinned identity and a statistics path that does not
    exist -- proves this never touches the filesystem on the unclustered
    path."""
    cfg = make_valid_config(
        run_tag="unclustered-stats",
        feature_dir=str(tmp_path / "does-not-exist"),
        feature_statistics_path=str(tmp_path / "does-not-exist" / "stats.safetensors"),
    )
    assert cfg.conditioning is None
    assert_live_statistics_identity(
        cfg,
        n_sae_features=16_384,
        pinned_statistics={
            "file_sha256": "0" * 64,
            "percentile_table_checksum": "0" * 64,
            "n_sae_features": 16_384,
            "feature_id_domain": [],
        },
        surface="test",
    )


def test_assert_live_statistics_identity_noop_when_statistics_unchanged(
    tmp_path: Path,
) -> None:
    """Normal-path proof: recomputing the identity from the exact same,
    untouched bytes reproduces the pinned values exactly -- no false raise
    on a genuine resume where nothing changed."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg("stats-unchanged", feature_dir=feature_dir, statistics_path=statistics_path)

    assert_live_statistics_identity(
        cfg, n_sae_features=16_384, pinned_statistics=pinned.canonical_dict(), surface="test",
    )


def test_assert_live_statistics_identity_raises_on_swapped_statistics_same_domain(
    tmp_path: Path,
) -> None:
    """RED/GREEN core proof for Bug 1 (HIGH): stats A pinned, the file is
    then swapped IN PLACE (same path, same width/domain -- feature_ids
    unchanged) to stats B with different byte content. Recomputing the
    live identity must disagree with the pinned snapshot and raise loud,
    not silently accept the new bytes."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    # Same path, same n_sae_features/feature_ids domain, different max_mag
    # values -> different bytes, different SHA-256, different rebuilt
    # percentile-table checksum.
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})
    cfg = _stats_cfg("stats-swapped", feature_dir=feature_dir, statistics_path=statistics_path)

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        assert_live_statistics_identity(
            cfg, n_sae_features=16_384, pinned_statistics=pinned_a.canonical_dict(), surface="test",
        )


def test_assert_live_statistics_identity_wraps_recompute_failure(tmp_path: Path) -> None:
    """A statistics file that vanished after the contract pinned it (or any
    other recompute failure) must raise AxBenchBaseError, not leak the
    underlying exception type."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = feature_dir / "feature_statistics.safetensors"
    cfg = _stats_cfg("stats-missing", feature_dir=feature_dir, statistics_path=statistics_path)

    with pytest.raises(AxBenchBaseError, match="could not recompute the live"):
        assert_live_statistics_identity(
            cfg,
            n_sae_features=16_384,
            pinned_statistics={
                "file_sha256": "0" * 64,
                "percentile_table_checksum": "0" * 64,
                "n_sae_features": 16_384,
                "feature_id_domain": [0],
            },
            surface="test",
        )


# ---- assert_live_statistics_identity_and_build_percentile_table ----


def test_assert_live_statistics_identity_and_build_percentile_table_noop_when_unclustered(
    tmp_path: Path,
) -> None:
    """U-invariant: returns ``None`` (so the caller falls back to its
    standard builder) when clustering is disabled -- even with a nonsense
    pinned identity and a statistics path that does not exist. Proves the
    fused helper never touches the filesystem on the unclustered path."""
    cfg = make_valid_config(
        run_tag="fused-unclustered",
        feature_dir=str(tmp_path / "does-not-exist"),
        feature_statistics_path=str(tmp_path / "does-not-exist" / "stats.safetensors"),
    )
    assert cfg.conditioning is None
    result = assert_live_statistics_identity_and_build_percentile_table(
        cfg,
        n_sae_features=16_384,
        pinned_statistics={
            "file_sha256": "0" * 64,
            "percentile_table_checksum": "0" * 64,
            "n_sae_features": 16_384,
            "feature_id_domain": [],
        },
        surface="test",
    )
    assert result is None


def test_assert_live_statistics_identity_and_build_percentile_table_returns_bound_table(
    tmp_path: Path,
) -> None:
    """Normal-path proof: with the statistics genuinely unchanged, the
    fused helper does NOT raise and returns a percentile table that is
    byte-identical to the production ``build_percentile_table_from_stats``
    builder -- i.e. binding the table to the verified identity does not
    change the clustered table's content, only proves its provenance."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg("fused-unchanged", feature_dir=feature_dir, statistics_path=statistics_path)

    result = assert_live_statistics_identity_and_build_percentile_table(
        cfg, n_sae_features=16_384, pinned_statistics=pinned.canonical_dict(), surface="test",
    )

    assert result is not None
    expected = build_percentile_table_from_stats(str(statistics_path), 16_384)
    assert t.equal(result, expected)


def test_assert_live_statistics_identity_and_build_percentile_table_raises_on_swap(
    tmp_path: Path,
) -> None:
    """RED/GREEN core proof for the FRESH-run table-build hole (Codex
    cross-review follow-up, Bug 1 HIGH): stats A pinned, the file is then
    swapped IN PLACE to stats B (same width/domain, different bytes). The
    fused helper must raise loud BEFORE returning any table built from B --
    it must never hand back a table whose provenance disagrees with the
    pinned identity."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    # Same path, same n_sae_features/feature_ids domain, different max_mag
    # values -> different bytes, different SHA-256, different rebuilt
    # percentile-table checksum.
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})
    cfg = _stats_cfg("fused-swapped", feature_dir=feature_dir, statistics_path=statistics_path)

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        assert_live_statistics_identity_and_build_percentile_table(
            cfg,
            n_sae_features=16_384,
            pinned_statistics=pinned_a.canonical_dict(),
            surface="test",
        )


def test_assert_live_statistics_identity_and_build_percentile_table_wraps_recompute_failure(
    tmp_path: Path,
) -> None:
    """A statistics file that vanished after the contract pinned it (or any
    other recompute failure) must raise AxBenchBaseError, not leak the
    underlying exception type or silently return ``None``/an empty table."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = feature_dir / "feature_statistics.safetensors"
    cfg = _stats_cfg("fused-missing", feature_dir=feature_dir, statistics_path=statistics_path)

    with pytest.raises(AxBenchBaseError, match="could not recompute the live"):
        assert_live_statistics_identity_and_build_percentile_table(
            cfg,
            n_sae_features=16_384,
            pinned_statistics={
                "file_sha256": "0" * 64,
                "percentile_table_checksum": "0" * 64,
                "n_sae_features": 16_384,
                "feature_id_domain": [0],
            },
            surface="test",
        )


# ---- assert_live_statistics_match_committed_marker ----


def test_assert_live_statistics_match_committed_marker_noop_when_unclustered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="marker-unclustered")
    assert cfg.conditioning is None
    # No feature_contract.json exists anywhere -- if this were not a no-op
    # it would raise trying to load a nonexistent marker.
    assert_live_statistics_match_committed_marker(cfg, n_sae_features=16_384, surface="test")


def test_assert_live_statistics_match_committed_marker_noop_when_statistics_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal-path proof for the completed-resume entrypoint (Codex-named
    location: negative_prefixes.py resume_negative_prefixes_if_complete /
    _resume_if_artifacts_complete), which never resolves a full
    ClusterFeatureContract and must read the pinned identity from the
    already-committed marker instead."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    run_tag = "marker-unchanged"
    _write_marker_with_statistics(run_tag, pinned_statistics=pinned.canonical_dict())
    cfg = _stats_cfg(run_tag, feature_dir=feature_dir, statistics_path=statistics_path)

    assert_live_statistics_match_committed_marker(cfg, n_sae_features=16_384, surface="test")


def test_assert_live_statistics_match_committed_marker_raises_on_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED/GREEN proof for Bug 1's third named path (completed resume):
    the committed marker pins stats A; the file is swapped in place to
    stats B (same width/domain); the cheap marker-based re-check must
    catch it without ever resolving a full contract."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    run_tag = "marker-swapped"
    _write_marker_with_statistics(run_tag, pinned_statistics=pinned.canonical_dict())
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})
    cfg = _stats_cfg(run_tag, feature_dir=feature_dir, statistics_path=statistics_path)

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        assert_live_statistics_match_committed_marker(cfg, n_sae_features=16_384, surface="test")


def test_assert_live_statistics_match_committed_marker_requires_run_tag() -> None:
    cfg = make_valid_config(
        run_tag=None,
        conditioning={"feature_clusters": {"path": "/clusters.yaml"}},
    )
    with pytest.raises(AxBenchBaseError, match="requires cfg.run_tag"):
        assert_live_statistics_match_committed_marker(cfg, n_sae_features=16_384, surface="test")


def test_assert_live_statistics_match_committed_marker_requires_statistics_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "marker-no-statistics-block"
    marker = test_paths.default_paths(run_tag).feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    semantic = {
        "concept_source": {
            "mode": "statistics_domain",
            "domain": [0],
            "source_to_representative": [[0, 0]],
        },
    }
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_contract_fingerprint": _sha256_json(semantic),
                "semantic": semantic,
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(run_tag)

    with pytest.raises(AxBenchBaseError, match="missing its pinned statistics identity"):
        assert_live_statistics_match_committed_marker(cfg, n_sae_features=16_384, surface="test")


# ---- validate_cluster_candidate_identity_with_collective_agreement ----


def test_validate_cluster_candidate_identity_collective_world_size_1_no_dist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At world_size<=1 this is exactly the two wrapped local calls -- no
    ``torch.distributed`` usage at all, mirroring ``_load_candidate_pool_
    with_collective_identity_agreement``'s own single-rank short-circuit."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    stats_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cluster_artifacts,
        "assert_live_statistics_identity",
        lambda *_a, **kwargs: stats_calls.append(kwargs),
    )

    def _explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("dist.all_gather_object should not be called at ws=1")

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _explode)

    validate_cluster_candidate_identity_with_collective_agreement(
        _cfg("collective-ws1"),
        _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
        candidate_path=tmp_path / "candidate_concepts.json",
        n_sae_features=16384,
        world_size=1,
    )

    assert len(stats_calls) == 1


def test_validate_cluster_candidate_identity_collective_all_ranks_agree_no_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op-on-agreement proof: when every rank's local validation agrees
    (the normal, consistent-view multi-rank case), the collective must not
    fire -- behavior is identical to the pre-fix bare local calls."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    monkeypatch.setattr(
        cluster_artifacts, "assert_live_statistics_identity", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _agreeing_gather(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _agreeing_gather)

    validate_cluster_candidate_identity_with_collective_agreement(
        _cfg("collective-agree"),
        _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
        candidate_path=tmp_path / "candidate_concepts.json",
        n_sae_features=16384,
        world_size=2,
    )


def test_validate_cluster_candidate_identity_collective_raises_together_when_a_peer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug 2 core proof (MEDIUM): THIS rank's own local validation
    succeeds, but the collective shows a peer rank failed (e.g. the
    stale/missing NFS view of the Stage 0a sidecar Codex named, or a
    genuine live-statistics mismatch). This rank must raise TOO, instead
    of returning normally and proceeding into the next collective while
    the failed peer never arrives -- exactly the deadlock the oracle-
    inherited bug allowed.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    monkeypatch.setattr(
        cluster_artifacts, "assert_live_statistics_identity", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _one_rank_failed(out: list[Any], local_status: dict[str, Any]) -> None:
        out[0] = local_status  # this rank's own check succeeded
        out[1] = {
            "ok": False,
            "error_type": "AxBenchBaseError",
            "error": "rank 1 saw a stale NFS view of the candidate sidecar",
        }

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _one_rank_failed)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        validate_cluster_candidate_identity_with_collective_agreement(
            _cfg("collective-peer-fails"),
            _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
            candidate_path=tmp_path / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=2,
        )


def test_validate_cluster_candidate_identity_collective_raises_on_this_ranks_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: THIS rank's own live-statistics check fails (a real
    swapped-statistics detection); the collective failure summary must
    still surface it (not swallow it) even though other ranks agreed."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)

    def _raising_stats_check(*_a: Any, **_k: Any) -> None:
        raise AxBenchBaseError("live feature statistics no longer match")

    monkeypatch.setattr(cluster_artifacts, "assert_live_statistics_identity", _raising_stats_check)
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _reflect_local_status(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _reflect_local_status)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        validate_cluster_candidate_identity_with_collective_agreement(
            _cfg("collective-self-fails"),
            _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
            candidate_path=tmp_path / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=2,
        )


def _collective_identity_gloo_target(
    rank: int, world_size: int, tmp_path_str: str,
) -> str:
    """Worker for the real-gloo no-hang proof. Rank 1's LOCAL validation
    genuinely fails (raises); every rank must still return (raise) from
    the collective -- a hang would time out the whole test instead of
    returning a result list.
    """
    from steering.eval.axbench_base import cluster_artifacts as ca_mod
    from steering.eval.axbench_base.types import AxBenchBaseError as _AxErr

    def _load(_cfg: Any, *, candidate_path: Any = None, expected_fingerprint: Any = None) -> None:
        del candidate_path, expected_fingerprint
        if rank == 1:
            raise _AxErr("rank 1 stale NFS view of the candidate sidecar")
        return None

    set_runtime_attribute(ca_mod, "load_cluster_candidate_identity", _load)
    set_runtime_attribute(
        ca_mod,
        "assert_live_statistics_identity",
        lambda *_a, **_k: None,
    )

    cfg = make_valid_config(
        run_tag="gloo-collective-identity",
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )
    contract = SimpleNamespace(
        feature_contract_fingerprint="a" * 64,
        statistics=SimpleNamespace(canonical_dict=lambda: {"file_sha256": "a" * 64}),
    )
    try:
        ca_mod.validate_cluster_candidate_identity_with_collective_agreement(
            cfg,
            cast(Any, contract),
            candidate_path=Path(tmp_path_str) / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=world_size,
        )
    except _AxErr as exc:
        return str(exc)
    return "unexpected success"


def test_collective_candidate_identity_one_rank_local_failure_all_ranks_raise_under_real_gloo(
    tmp_path: Path, gloo_multiprocess: Any,
) -> None:
    """Empirical no-hang proof: a real 2-process gloo group where rank 1's
    local check genuinely raises. Before the fix this left rank 0 blocked
    on the next collective; the test itself completing (rather than
    hanging until the process-group timeout) is the proof. Both ranks must
    report the SAME collective failure summary.
    """
    results = gloo_multiprocess(2, _collective_identity_gloo_target, str(tmp_path))

    assert all("failed collectively before rank work" in result for result in results)
    assert results[0] == results[1]


# ============================================================================
# PR #739 FU-D-2: validate_cluster_candidate_identity_with_collective_
# agreement now RETURNS the validated identity (resolve-once-at-startup
# design) + the fields-equality proof that the pure derivation helper never
# drifts from a fresh read.
# ============================================================================


def test_validate_cluster_candidate_identity_collective_returns_validated_identity_world_size_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #739 FU-D-2: the vote now RETURNS the validated identity (not
    just validates as a side effect) so a caller can resolve it ONCE at
    startup and thread its fields to hot/per-rank consumers instead of
    re-reading on every later call."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    monkeypatch.setattr(
        cluster_artifacts, "assert_live_statistics_identity", lambda *_a, **_k: None,
    )

    identity = validate_cluster_candidate_identity_with_collective_agreement(
        _cfg("collective-returns-ws1"),
        _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
        candidate_path=tmp_path / "candidate_concepts.json",
        n_sae_features=16384,
        world_size=1,
    )

    assert isinstance(identity, ClusterCandidateIdentity)
    assert identity.feature_contract_fingerprint == "a" * 64


def test_validate_cluster_candidate_identity_collective_all_ranks_agree_returns_own_local_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On agreement, the caller gets back a REAL, correct identity (not
    just a no-raise) -- mirrors ``build_verified_percentile_table_with_
    collective_agreement``'s own "returns a real correct value" proof
    (each agreeing rank returns its OWN locally-validated identity rather
    than broadcasting rank 0's -- see the function's own docstring for
    why this is safe)."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _install_candidate_identity(monkeypatch)
    monkeypatch.setattr(
        cluster_artifacts, "assert_live_statistics_identity", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _agreeing_gather(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _agreeing_gather)

    identity = validate_cluster_candidate_identity_with_collective_agreement(
        _cfg("collective-agree-returns"),
        _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
        candidate_path=tmp_path / "candidate_concepts.json",
        n_sae_features=16384,
        world_size=2,
    )

    assert isinstance(identity, ClusterCandidateIdentity)
    assert identity.feature_contract_fingerprint == "a" * 64


def test_validate_cluster_candidate_identity_collective_raises_when_local_load_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive fail-loud guard: a caller that invokes this vote when
    ``cfg`` is not actually clustered-enabled (an invariant violation --
    both real call sites gate on ``cluster_contract is not None`` first,
    which they also require to imply clustering is enabled) must raise
    loud instead of silently threading empty fingerprint fields
    downstream."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        cluster_artifacts, "load_cluster_candidate_identity", lambda *_a, **_k: None,
    )

    with pytest.raises(AxBenchBaseError, match="requires a resolved cluster contract"):
        validate_cluster_candidate_identity_with_collective_agreement(
            _cfg("collective-none-identity"),
            _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
            candidate_path=tmp_path / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=1,
        )


def test_clustered_fingerprint_fields_from_identity_matches_fresh_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #739 FU-D-2 fields-equality proof: the dict derived ONCE from an
    already-validated ``ClusterCandidateIdentity`` (via
    ``_clustered_fingerprint_fields_from_identity``, the same helper the
    startup collective vote uses to build ``precomputed_clustered_
    fields``) is byte-identical to a completely independent fresh
    ``clustered_stage0c_fingerprint_fields(cfg)`` read of the SAME on-disk
    sidecar state. Proves the cache carries identical content, not just a
    plausible-looking substitute.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "fields-equality"
    _write_candidate_artifacts(run_tag)
    document = _candidate_contract_document()
    monkeypatch.setattr(
        cluster_artifacts, "load_feature_contract_sidecar",
        lambda _path, *, expected_fingerprint=None: copy.deepcopy(document),
    )
    cfg = _cfg(run_tag)

    identity = load_cluster_candidate_identity(cfg, expected_fingerprint="a" * 64)
    assert identity is not None

    once_derived = cluster_artifacts._clustered_fingerprint_fields_from_identity(identity)
    fresh_read = clustered_stage0c_fingerprint_fields(cfg)

    assert once_derived == fresh_read
    assert once_derived == {
        "feature_contract_fingerprint": "a" * 64,
        "candidate_cluster_provenance_sha256": identity.candidate_provenance_sha256,
        "cluster_concept_projection_version": identity.concept_projection_version,
    }


def test_clustered_stage0c_fingerprint_fields_maps_none_load_to_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U-invariant companion, retargeted for the owner's required-
    parameter cleanup (review of PR #884): the ``None`` (no clustered
    sidecar) -> ``{}`` mapping now lives on the wrapper
    (``clustered_stage0c_fingerprint_fields``), not on
    ``_clustered_fingerprint_fields_from_identity`` -- that helper's
    ``identity`` parameter is deliberately non-Optional now (see its own
    docstring: "the CALLER that can actually observe [a None load]
    decides", not a nullable pass-through here).

    Isolates the wrapper's OWN branch from ``load_cluster_candidate_
    identity``'s real unclustered-cfg behavior (already covered by
    ``test_stage0c_fingerprint_fields_empty_dict_when_unclustered`` above)
    by forcing a ``None`` load result directly, regardless of cfg.
    """
    monkeypatch.setattr(
        cluster_artifacts, "load_cluster_candidate_identity", lambda cfg: None,
    )
    cfg = make_valid_config(run_tag="wrapper-none-load-mapping")
    assert clustered_stage0c_fingerprint_fields(cfg) == {}


# ============================================================================
# validate_resume_statistics_identity_with_collective_agreement (PR #739
# FU-D-stage0c fix batch, follow-up closing the LAST known deadlock analog:
# ``stage0c_runner._try_resume_per_rank``'s own bare per-rank copy of the
# Bug 1 check, reached only on the RESUME path -- i.e. only when a rank's
# own partial checkpoint already exists). Real-gloo, end-to-end wiring
# proof lives in ``test_stage0c_runner.py`` (the ``_try_resume_per_rank``
# call site); the tests below cover the helper's own branches directly,
# mirroring ``validate_cluster_candidate_identity_with_collective_
# agreement``'s own mock-based suite above.
# ============================================================================


def test_validate_resume_statistics_identity_collective_world_size_1_no_dist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At world_size<=1 this is exactly the wrapped local call -- no
    ``torch.distributed`` usage at all, mirroring the sibling collective-
    agreement helpers' own single-rank short-circuit."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg("resume-collective-ws1", feature_dir=feature_dir, statistics_path=statistics_path)
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())

    def _explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("dist.all_gather_object should not be called at ws=1")

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _explode)

    validate_resume_statistics_identity_with_collective_agreement(
        cfg, contract, has_local_checkpoint=True, n_sae_features=16384,
        world_size=1, surface="test",
    )


def test_validate_resume_statistics_identity_collective_no_local_checkpoint_is_noop_at_world_size_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``has_local_checkpoint=False`` must skip the live-statistics read
    entirely -- there is no partial resumed state to protect, so this rank
    has nothing to validate. Proven with a deliberately WRONG pinned dict
    that would raise if the check ran; it must not."""
    stats_calls: list[Any] = []
    monkeypatch.setattr(
        cluster_artifacts,
        "assert_live_statistics_identity",
        lambda *a, **k: stats_calls.append((a, k)),
    )
    cfg = _cfg("resume-collective-no-checkpoint-ws1")
    contract = _fake_contract_with_statistics(
        "a" * 64, statistics_dict={"this": "would-mismatch-if-checked"},
    )

    validate_resume_statistics_identity_with_collective_agreement(
        cfg, contract, has_local_checkpoint=False, n_sae_features=16384,
        world_size=1, surface="test",
    )

    assert stats_calls == []


def test_validate_resume_statistics_identity_collective_all_ranks_agree_no_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op-on-agreement proof: when every rank's local validation agrees
    (the normal, consistent-view multi-rank case -- statistics genuinely
    unchanged), the collective must not fire -- behavior is identical to
    the pre-fix bare local call."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg("resume-collective-agree", feature_dir=feature_dir, statistics_path=statistics_path)
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _agreeing_gather(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _agreeing_gather)

    validate_resume_statistics_identity_with_collective_agreement(
        cfg, contract, has_local_checkpoint=True, n_sae_features=16384,
        world_size=2, surface="test",
    )


def test_validate_resume_statistics_identity_collective_no_local_checkpoint_all_ranks_agree_no_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common real-world entry point: a brand-new clustered multi-rank
    run where NEITHER rank has a partial checkpoint yet. Both ranks' local
    steps are trivial passes; the collective must still rendezvous cleanly
    and let every rank proceed -- this is the shape every fresh clustered
    Stage 0c run takes on its very first block."""
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _agreeing_gather(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _agreeing_gather)
    cfg = _cfg("resume-collective-no-checkpoint-agree")
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64})

    validate_resume_statistics_identity_with_collective_agreement(
        cfg, contract, has_local_checkpoint=False, n_sae_features=16384,
        world_size=2, surface="test",
    )


def test_validate_resume_statistics_identity_collective_raises_together_when_a_peer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core proof: THIS rank's own local validation succeeds, but the
    collective shows a peer rank failed (a genuine live-statistics
    mismatch observed only on that peer). This rank must raise TOO,
    instead of returning normally and proceeding into ``run_single_rank``'s
    next real collective while the failed peer never arrives."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg(
        "resume-collective-peer-fails", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _one_rank_failed(out: list[Any], local_status: dict[str, Any]) -> None:
        out[0] = local_status  # this rank's own check succeeded
        out[1] = {
            "ok": False,
            "error_type": "AxBenchBaseError",
            "error": "rank 1 saw a swapped clustered statistics file on resume",
        }

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _one_rank_failed)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        validate_resume_statistics_identity_with_collective_agreement(
            cfg, contract, has_local_checkpoint=True, n_sae_features=16384,
            world_size=2, surface="test",
        )


def test_validate_resume_statistics_identity_collective_no_local_checkpoint_still_raises_when_peer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critical edge case that makes gating the collective CALL itself
    on ``has_local_checkpoint`` unsafe: THIS rank has no local checkpoint
    (nothing to validate, a trivial local pass) but the gathered peer
    status shows a real failure. This rank must raise too -- if a rank
    with nothing to check could return early without posting to (or
    waiting on) the collective, it would sail into ``run_single_rank``'s
    later collective while its resuming peer is dead, reproducing the
    exact deadlock this function exists to prevent."""
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _peer_failed(out: list[Any], local_status: dict[str, Any]) -> None:
        assert local_status == {"ok": True, "error_type": None, "error": None}
        out[0] = local_status  # this rank had nothing to check -- trivial ok
        out[1] = {
            "ok": False,
            "error_type": "AxBenchBaseError",
            "error": "rank 1 saw a swapped clustered statistics file on resume",
        }

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _peer_failed)
    cfg = _cfg("resume-collective-no-checkpoint-peer-fails")
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64})

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        validate_resume_statistics_identity_with_collective_agreement(
            cfg, contract, has_local_checkpoint=False, n_sae_features=16384,
            world_size=2, surface="test",
        )


def test_validate_resume_statistics_identity_collective_raises_on_this_ranks_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: THIS rank's own live-statistics check fails (a real
    swapped-statistics detection on resume); the collective failure summary
    must still surface it (not swallow it) even though other ranks agreed."""

    def _raising_stats_check(*_a: Any, **_k: Any) -> None:
        raise AxBenchBaseError("live feature statistics no longer match")

    monkeypatch.setattr(cluster_artifacts, "assert_live_statistics_identity", _raising_stats_check)
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _reflect_local_status(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _reflect_local_status)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        validate_resume_statistics_identity_with_collective_agreement(
            _cfg("resume-collective-self-fails"),
            _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
            has_local_checkpoint=True,
            n_sae_features=16384,
            world_size=2,
            surface="test",
        )


# ============================================================================
# build_verified_percentile_table_with_collective_agreement (PR #739
# FU-D-stage0c fix batch, Codex RE-CHECK of the Bug 1 HIGH fix -- MEDIUM:
# the fresh-run fused percentile-table build reintroduced a Bug-2-class
# per-rank-raise deadlock). See cluster_artifacts.py's own module-level
# comment block immediately above the function for the full mechanism.
# ============================================================================


def test_build_verified_percentile_table_collective_world_size_1_no_dist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At world_size<=1 this is exactly the wrapped local call -- no
    ``torch.distributed`` usage at all, mirroring ``validate_cluster_
    candidate_identity_with_collective_agreement``'s own single-rank
    short-circuit. Also proves the returned table is real and correct
    (byte-identical to the production builder), not just that no
    exception was raised."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg(
        "fused-collective-ws1", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())

    def _explode(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("dist.all_gather_object should not be called at ws=1")

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _explode)

    result = build_verified_percentile_table_with_collective_agreement(
        cfg, contract, n_sae_features=16_384, surface="test", world_size=1,
    )

    expected = build_percentile_table_from_stats(str(statistics_path), 16_384)
    assert t.equal(result, expected)


def test_build_verified_percentile_table_collective_all_ranks_agree_no_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-op-on-agreement proof: when every rank's local fused build agrees
    (the normal, consistent-view multi-rank case), the collective must not
    fire -- behavior is identical to the pre-fix bare local call, plus a
    real, correctly-built table."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg(
        "fused-collective-agree", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _agreeing_gather(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _agreeing_gather)

    result = build_verified_percentile_table_with_collective_agreement(
        cfg, contract, n_sae_features=16_384, surface="test", world_size=2,
    )

    expected = build_percentile_table_from_stats(str(statistics_path), 16_384)
    assert t.equal(result, expected)


def test_build_verified_percentile_table_collective_raises_together_when_a_peer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core proof: THIS rank's own local fused build succeeds, but the
    collective shows a peer rank failed (e.g. a genuine live-statistics
    mismatch observed only on that peer). This rank must raise TOO,
    instead of returning a table and proceeding into the next collective
    while the failed peer never arrives -- exactly the deadlock the fused
    build's bare per-rank-local shape allowed."""
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _stats_cfg(
        "fused-collective-peer-fails", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned.canonical_dict())
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _one_rank_failed(out: list[Any], local_status: dict[str, Any]) -> None:
        out[0] = local_status  # this rank's own check succeeded
        out[1] = {
            "ok": False,
            "error_type": "AxBenchBaseError",
            "error": "rank 1 saw a swapped clustered statistics file",
        }

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _one_rank_failed)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        build_verified_percentile_table_with_collective_agreement(
            cfg, contract, n_sae_features=16_384, surface="test", world_size=2,
        )


def test_build_verified_percentile_table_collective_raises_on_this_ranks_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror case: THIS rank's own local fused build fails (a real
    swapped-statistics detection); the collective failure summary must
    still surface it (not swallow it) even though other ranks agreed."""

    def _raising_build(*_a: Any, **_k: Any) -> None:
        raise AxBenchBaseError("live feature statistics no longer match")

    monkeypatch.setattr(
        cluster_artifacts,
        "assert_live_statistics_identity_and_build_percentile_table",
        _raising_build,
    )
    monkeypatch.setattr(cluster_artifacts.dist, "is_available", lambda: True)
    monkeypatch.setattr(cluster_artifacts.dist, "is_initialized", lambda: True)

    def _reflect_local_status(out: list[Any], local_status: dict[str, Any]) -> None:
        for i in range(len(out)):
            out[i] = dict(local_status)

    monkeypatch.setattr(cluster_artifacts.dist, "all_gather_object", _reflect_local_status)

    with pytest.raises(AxBenchBaseError, match="failed collectively before rank work"):
        build_verified_percentile_table_with_collective_agreement(
            _cfg("fused-collective-self-fails"),
            _fake_contract_with_statistics("a" * 64, statistics_dict={"file_sha256": "a" * 64}),
            n_sae_features=16384,
            surface="test",
            world_size=2,
        )


def _fused_table_agree_gloo_target(
    rank: int, world_size: int, feature_dir_str: str, statistics_path_str: str,
    pinned_dict: dict[str, Any],
) -> list[list[float]]:
    """Worker for the real-gloo all-agree proof: every rank's local fused
    read+verify+build genuinely agrees (real, unmodified, shared on-disk
    statistics -- no monkeypatch, no swap). Proves the collective wrapper
    returns a real, correctly-built table on the ordinary multi-rank path,
    not just that it correctly detects disagreement.

    No local imports: this function is looked up by qualified name in a
    freshly-spawned child process (``mp.get_context("spawn")``), which
    re-executes this whole module's top-level imports before locating the
    function -- ``cluster_artifacts``/``Path``/``Any``/``cast`` are already
    present in this module's globals by the time this body runs, exactly
    like the pre-existing ``_collective_identity_gloo_target`` relies on
    ``make_valid_config``/``cast``/``Path``/``SimpleNamespace`` being
    already-imported module globals without re-importing them itself.
    """
    feature_dir = Path(feature_dir_str)
    statistics_path = Path(statistics_path_str)
    cfg = _stats_cfg(
        "gloo-fused-table-agree", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned_dict)

    table = cluster_artifacts.build_verified_percentile_table_with_collective_agreement(
        cfg, contract, n_sae_features=16384, surface="test", world_size=world_size,
    )
    return cast(list[list[float]], table.tolist())


def test_build_verified_percentile_table_collective_all_ranks_agree_returns_table_under_real_gloo(
    tmp_path: Path, gloo_multiprocess: Any,
) -> None:
    """Plus an all-ranks-agree proof under a real 2-process gloo group: no
    false failure, and both ranks return the SAME real, correctly-built
    table -- the collective wrapper's normal-path no-op does not corrupt
    or diverge the table itself.
    """
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )

    results = gloo_multiprocess(
        2, _fused_table_agree_gloo_target,
        str(feature_dir), str(statistics_path), pinned.canonical_dict(),
    )

    expected = build_percentile_table_from_stats(str(statistics_path), 16_384).tolist()
    assert results[0] == expected
    assert results[1] == expected


def _fused_table_gloo_target(
    rank: int, world_size: int, feature_dir_str: str, statistics_path_str: str,
    pinned_dict: dict[str, Any],
) -> str:
    """Worker for the fused-percentile-table collective no-hang proof (the
    MEDIUM Codex found on re-check of the Bug 1 HIGH fix: the fresh-run
    fused build closed the TOCTOU hole but ran as a bare per-rank local
    call, reintroducing a Bug-2-class deadlock). Rank 1's local fused
    build genuinely fails: its statistics READ is monkeypatched (not the
    shared on-disk file, so there is no timing race with rank 0's own
    independent read of the SAME real bytes) to return an identity that
    disagrees with the pinned one. Before the fix this left rank 0 free
    to proceed into the dispatcher's next real collective while rank 1
    died raising locally; the fix must make BOTH ranks raise together,
    and NEITHER may ever reach a later collective.

    No local imports (see ``_fused_table_agree_gloo_target``'s docstring
    for why they are unnecessary here): ``Path``/``SimpleNamespace``/``dist``/
    ``cluster_artifacts``/``AxBenchBaseError`` are all already-imported
    module globals by the time a freshly-spawned child process locates
    this function by qualified name.
    """
    feature_dir = Path(feature_dir_str)
    statistics_path = Path(statistics_path_str)

    if rank == 1:
        def _mismatched_stats_identity(
            *, feature_dir: Path, statistics_path: Path, n_sae_features: int,
        ) -> tuple[Any, None, None]:
            del feature_dir, statistics_path, n_sae_features
            return (
                SimpleNamespace(canonical_dict=lambda: {"rank1_saw_a_mismatch": True}),
                None,
                None,
            )

        set_runtime_attribute(
            cluster_artifacts,
            "_statistics_identity",
            _mismatched_stats_identity,
        )

    cfg = _stats_cfg(
        "gloo-fused-table-fail", feature_dir=feature_dir, statistics_path=statistics_path,
    )
    contract = _fake_contract_with_statistics("a" * 64, statistics_dict=pinned_dict)

    try:
        cluster_artifacts.build_verified_percentile_table_with_collective_agreement(
            cfg, contract, n_sae_features=16384, surface="test", world_size=world_size,
        )
    except AxBenchBaseError as exc:
        return f"raised:{exc}"

    # Should be unreachable under the fix: both ranks must raise together
    # before either could get here. If a future regression drops the
    # collective agreement, enter a REAL subsequent collective -- standing
    # in for the dispatcher's own next rendezvous (the candidate-pool
    # gather, or run_single_rank's block-barrier all_gather) -- so the
    # regression manifests as an observable hang/timeout via
    # gloo_multiprocess's own _RESULT_TIMEOUT_S bound, instead of a silent
    # pass.
    out: list[Any] = [None] * world_size
    dist.all_gather_object(out, "reached-next-collective")
    return "reached-next-collective"


def test_build_verified_percentile_table_collective_one_rank_local_failure_all_ranks_raise_under_real_gloo(
    tmp_path: Path, gloo_multiprocess: Any,
) -> None:
    """Empirical no-hang proof for the MEDIUM Codex found on re-check of
    the Bug 1 HIGH fix (the fresh-run fused percentile-table build
    reintroduced a Bug-2-class per-rank-raise deadlock): a real 2-process
    gloo group where rank 1's local fused build genuinely disagrees with
    the pinned identity. Before this fix, rank 1 raised locally while
    rank 0 sailed through to the dispatcher's next real collective and
    blocked forever waiting for a peer that already died -- the test
    itself completing (rather than hanging/timing out) is the proof. Both
    ranks must report the SAME collective failure summary.
    """
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = cluster_checkpoint_pin_module._statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )

    results = gloo_multiprocess(
        2, _fused_table_gloo_target,
        str(feature_dir), str(statistics_path), pinned.canonical_dict(),
    )

    assert all(result.startswith("raised:") for result in results), results
    assert all(
        "no longer match the identity pinned" in result for result in results
    ), results
    assert results[0] == results[1]
