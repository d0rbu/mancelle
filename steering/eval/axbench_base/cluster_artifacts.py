"""Conditional clustered-source artifact identity for AxBench-Base.

Port of the oracle's ``steering/eval/axbench_base/cluster_artifacts.py``
(PR #739 Track C1) onto current ``main``. This module is oracle-only and 0%
redundant on current main -- a full port, unchanged scope, sitting directly
on top of the now-merged Track B contract (``ClusterFeatureContract`` /
``cluster_feature_contract.py``). It consumes only two things from that
layer: the opaque ``contract.feature_contract_fingerprint: str``, and the
fully-resolved ``semantic``/``provenance`` JSON dicts ``load_feature_
contract_sidecar`` returns. It never imports or calls ``effective_feature_
vocab``, ``FeatureClusterConsolidation``, or anything selection/packing-
hash-shaped -- confirmed by direct read (zero occurrences in the 732-line
oracle file) and by Track B's own module docstring, which states the
vocab/splice/packing drift is fully absorbed before the fingerprint ever
reaches this layer.

MAIN-DRIFT CHECK (full table in task-c1-report.md): #759 effective-feature-
vocab, ``splice_batch_size``, and fab4c6bc packing-identity are all real,
present, actively-used on current main, but NOT APPLICABLE to this file --
it never reads any field they touch, only the already-resolved fingerprint
string and sidecar JSON dicts Track B computed and validated. The one real
main-drift surface is ``stream_hash_file_stable``'s internal semantics: it
is now a single fd-bound call (``immutable_io.stream_hash_file_descriptor_
bound``) rather than a pathname-``stat()``-only version, which is what
motivates the two named deviations below.

Named deviation 1 (PR #739 Track C1, Finding 1, SIMPLIFICATION):
``_load_stable_json``'s two dict-shaped call sites (the Stage 0a candidate-
provenance sidecar inside ``load_cluster_candidate_identity``, and the
Stage 4 generation-resume marker inside ``validate_cluster_generation_
resume``) now reuse ``cluster_checkpoint_pin._load_json_mapping`` directly
-- the canonical fused hash-once/read-once/verify-once primitive Track B
already established campaign-wide. Each call site immediately follows that
load with a second ``stream_hash_file_stable`` snapshot and compares the full
``StableFileDigest`` outside the shared exception wrapper. This restores the
oracle's post-read filesystem bracket while retaining the fused primitive's
direct consumed-bytes cross-check. The candidate-provenance flow additionally
re-hashes both the provenance and candidate files immediately before
constructing ``ClusterCandidateIdentity`` and compares the content SHA-256
values with the digests the identity will carry. Any exception from either
the mapping load or its immediate snapshot is caught and re-raised as
``AxBenchBaseError`` at each call site, preserving this module's own
domain-error contract (see Finding 2 below). The downstream
``isinstance(raw, Mapping)`` check the oracle ran on the candidate-provenance
sidecar after loading it is dead code -- ``_load_json_mapping``'s own return
type (``tuple[dict[str, Any], StableFileDigest]``) already guarantees a dict
-- and has been removed rather than kept as unreachable defensive code.

The candidate-identity flow re-loads the root feature-contract marker through
``load_feature_contract_sidecar`` in the final flow-level re-verification.
Thus all three consumed artifacts -- root marker, provenance sidecar, and
candidate concepts -- are re-verified sequentially before constructing the
identity. Each artifact is verified only at the moments it is observed: the
three final checks are not an atomic snapshot, an artifact may change after
its own final check, and this function cannot guarantee the files' state at
any later instant or after it returns. Subject to the parsed-marker caveat
below, every schedule in which a checkpoint observes an artifact difference
is rejected, which is strictly more than the pre-FU-B code detected.

The root marker's final comparison is between parsed documents, not raw bytes.
Python JSON equality can therefore treat byte-distinct values as equal, such
as ``1`` versus ``true`` or ``1`` versus ``1.0``, in ``schema_version`` or
``provenance``. This does not expose the identity-bearing ``semantic`` block:
``validate_feature_contract_document`` recomputes its fingerprint as
``_sha256_json(raw["semantic"])`` and checks the expected fingerprint on every
load. The aliasing residual therefore cannot make the returned
``ClusterCandidateIdentity.feature_contract_fingerprint`` incorrect.

This simplification applies ONLY to these two dict-shaped sites.
``_load_stable_json`` itself (the file's one remaining, list-shaped call
site) is NOT simplified to ``_load_json_mapping``'s shape: an earlier
revision of this port incorrectly did so and a Codex cross-review (HIGH)
caught the resulting regression -- see "Named deviation 1b" immediately
below.

Named deviation 1b (PR #739 Track C1, Codex cross-review fix, HIGH):
``_load_stable_json`` survives as this file's one remaining call site --
the list-shaped ``candidate_concepts.json`` read (oracle:319-320:
"clustered candidate_concepts.json must be a list") -- ``_load_json_
mapping`` cannot be reused there because it hard-requires a JSON object and
would incorrectly reject a valid list-shaped file. An earlier revision of
this port reimplemented it to mirror ``_load_json_mapping``'s single-hash-
then-verify shape (one ``stream_hash_file_stable`` call, one read, one
direct sha256+length cross-check), claiming "identical detection power" to
the oracle's own two-hash bracket. That claim was FALSE: dropping the
oracle's post-read re-hash (``after = stream_hash_file_stable(path)``,
cross-checked via ``before != after``) leaves a real, distinct race
window uncaught -- a racing writer that swaps the file to B strictly
AFTER the read completes (with the read itself still observing the
original A, so the read-vs-pre-read-digest cross-check alone sees no
disagreement) silently returns an A-backed identity while the file is
already B on disk. Independently verified by replaying this exact
scenario: the single-hash shape does not raise; the oracle's own shape
does. ``_load_stable_json`` is now reimplemented to match the oracle's
two-hash bracket byte-for-byte (hash before, read via the *resolved*
``before.path`` -- not the possibly-unresolved/symlinked original
argument, a still-valid closing of the same class of gap for the read
step itself -- hash after, reject on ``before != after`` OR the payload's
own hash disagreeing with ``before``), plus one added strengthening: an
explicit ``byte_length`` cross-check, which can only reject MORE cases,
never fewer. This site therefore carries NO Finding-1 detection-power
deviation at all versus the oracle -- the only remaining Finding 1
SIMPLIFICATION is the two dict-shaped sites' swap to ``_load_json_mapping``
(Named deviation 1, above).

Named deviation 2 (PR #739 Track C1, Finding 1, BUG FIX):
``validate_cluster_generation_resume``'s own inline TOCTOU bracket (the
oracle's separate, *third*, and *weaker* reimplementation of the same
pattern in this same file) only ever compared ``stream_hash_file_stable``
before/after snapshots (``StableFileDigest`` equality); it never separately
cross-checked the actual read bytes against the "before" digest the way
``_load_stable_json`` itself did. That is a real, if narrow, verification
gap: a content round-trip (A -> B -> A) entirely inside the unprotected
middle read leaves before == after (both observe A) while the bytes
actually parsed were the transient B, with no raise at all. Reusing
``_load_json_mapping`` here closes that gap: it directly cross-checks the
read payload's own content hash against the pre-read digest, which catches
this exact race regardless of what the file looks like afterward. See
``test_cluster_artifacts.py``'s dedicated regression test, which fails
under the oracle's original inline shape and passes under this one.

Named deviation 3 (PR #739 Track C1, owner-lens ``_strict_sorted_unique_ids``
asymmetry, BUG FIX): the oracle's ``validate_cluster_candidate_provenance_
semantics`` rejected an empty/malformed ``representative_universe``
(falsy check) but only rejected a *malformed* ``concept_source.domain``,
accepting an empty-but-well-formed one (``is None`` check, not falsy) as
long as every other field was consistently empty to match. This validator
is deliberately independent defense-in-depth over an on-disk JSON sidecar
(see its own docstring below) and must not assume its input was honestly
produced by ``build_cluster_feature_contract`` -- but main's own contract
builder now enforces the analogous non-emptiness invariant on
``source_to_representative`` at construction time (E4 non-emptiness,
``cluster_feature_contract.py``'s "empty_concept_projection_after_masking"
handling), which makes an all-consistently-empty sidecar something no
honestly-built upstream contract could ever produce in the first place.
Bringing this validator's ``concept_source.domain`` check into symmetry
with its own ``representative_universe`` check (falsy, not ``is None``)
closes the same E4 gap defense-in-depth is supposed to guard, for a sidecar
that claims (via fingerprint match) to derive from an honestly-built
contract. Named, not silent; see the dedicated regression test.

Mechanical completeness fix (not a fidelity deviation): the oracle's own
``__all__`` (oracle:719-732) omitted ``CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_
VERSION`` and ``validate_cluster_candidate_provenance_semantics``, both of
which real future consumers import directly (``concept_list.py``,
``axbench_bpb_adv/source_validation.py``). Both are now included.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import torch.distributed as dist

from steering.conditional.percentile_table import build_percentile_table
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.cluster_checkpoint_pin import _load_json_mapping, _statistics_identity
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract as _ClusterFeatureContract,
)
from steering.eval.cluster_feature_contract import (
    load_feature_contract_sidecar,
    stream_hash_file_stable,
    write_feature_contract_sidecar,
)
from steering.immutable_io import ImmutableArtifactError, install_bytes_once, is_sha256_hex
from steering.safetensors_io import deterministic_safetensors_bytes

CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION = 1
CLUSTER_GENERATION_RESUME_SCHEMA_VERSION = 2


def install_clustered_bytes_once(
    payload: bytes,
    path: Path,
    *,
    surface: str,
) -> bool:
    """Create an immutable clustered artifact or validate an exact replay.

    The final path is installed with a same-filesystem hard link from a fully
    flushed temporary file.  Unlike ``os.replace``, the link cannot overwrite
    an artifact created by an earlier run or a concurrent writer.  A racing
    writer is accepted only when it committed byte-for-byte identical content.

    Returns ``True`` when this call installed the file and ``False`` for an
    exact existing replay.
    """

    try:
        return install_bytes_once(
            payload,
            path,
            surface=f"Clustered {surface}",
        )
    except ImmutableArtifactError as exc:
        raise AxBenchBaseError(str(exc)) from exc


def write_clustered_parquet_once(
    table: pa.Table,
    path: Path,
    *,
    surface: str,
) -> bool:
    """Serialize and immutably install a clustered parquet artifact."""

    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return install_clustered_bytes_once(
        sink.getvalue().to_pybytes(),
        path,
        surface=surface,
    )


def write_clustered_safetensors_once(
    tensors: Mapping[str, t.Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str],
    surface: str,
) -> bool:
    """Canonically serialize and immutably install clustered tensors."""

    return install_clustered_bytes_once(
        deterministic_safetensors_bytes(tensors, metadata=metadata),
        path,
        surface=surface,
    )




def clustered_source_enabled(cfg: AxBenchBaseConfig) -> bool:
    conditioning = cfg.conditioning
    return bool(
        isinstance(conditioning, Mapping) and conditioning.get("feature_clusters") is not None
    )


def _sha256_file(path: Path) -> str:
    try:
        return stream_hash_file_stable(path).sha256
    except Exception as exc:
        raise AxBenchBaseError(f"Could not stably hash clustered AxBench artifact {path}") from exc


def _load_stable_json(path: Path, *, surface: str) -> tuple[Any, str]:
    """Read JSON while pinning both the observed file and the consumed bytes.

    Byte-identical in detection power to the oracle's own pattern (hash
    before, read, hash after; reject on EITHER ``before != after`` OR the
    read payload's own hash disagreeing with ``before``) -- see the module
    docstring's "Named deviation 1" for why this file's earlier revision
    incorrectly dropped the post-read re-hash (Codex cross-review Finding
    1, HIGH: a racing writer that swaps the file to B strictly AFTER the
    read completes, with the read itself still observing A, was silently
    accepted -- this function's earlier single-hash-then-read shape had no
    further filesystem check past that read to catch it). Restoring the
    ``after`` snapshot closes that window. Adds one strengthening beyond
    the oracle: an explicit ``byte_length`` cross-check against ``before``,
    which can only reject MORE cases, never fewer.

    This remains the ONE call site ``_load_json_mapping`` itself cannot
    serve directly: ``candidate_concepts.json`` is list-shaped, and
    ``_load_json_mapping`` hard-requires a JSON object.
    """

    try:
        before = stream_hash_file_stable(path)
        payload = before.path.read_bytes()
        after = stream_hash_file_stable(path)
    except Exception as exc:
        raise AxBenchBaseError(f"Could not stably read clustered {surface} {path}") from exc
    if (
        before != after
        or len(payload) != before.byte_length
        or hashlib.sha256(payload).hexdigest() != before.sha256
    ):
        raise AxBenchBaseError(f"clustered {surface} changed while it was being loaded")
    try:
        return json.loads(payload), before.sha256
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(f"clustered {surface} is not valid JSON: {path}") from exc


def _strict_sorted_unique_ids(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(type(item) is not int or item < 0 for item in value):
        return None
    ids = cast(list[int], value)
    if ids != sorted(set(ids)):
        return None
    return ids


def validate_cluster_candidate_provenance_semantics(
    *,
    feature_contract: Mapping[str, Any],
    candidate_rows: object,
    provenance: Mapping[str, Any],
) -> None:
    """Require the Stage 0a audit sidecar to restate the exact contract.

    A matching fingerprint and candidate hash are not enough: stale member or
    source-projection rows could otherwise be carried through every downstream
    AxBench stage while falsely documenting a different target cluster.
    """

    semantic = feature_contract.get("semantic")
    if not isinstance(semantic, Mapping):
        raise AxBenchBaseError("clustered feature contract lacks semantic content")
    clusters = semantic.get("feature_clusters")
    concept_source = semantic.get("concept_source")
    if not isinstance(clusters, Mapping) or not isinstance(concept_source, Mapping):
        raise AxBenchBaseError(
            "clustered feature contract lacks cluster or concept-source semantics"
        )

    representative_universe_list = _strict_sorted_unique_ids(
        clusters.get("representative_universe")
    )
    if not representative_universe_list:
        raise AxBenchBaseError("clustered feature contract has a malformed representative universe")
    representative_universe = set(representative_universe_list)

    raw_members = clusters.get("representative_to_allowed_members")
    if not isinstance(raw_members, list):
        raise AxBenchBaseError("clustered feature contract lacks representative member semantics")
    members_by_representative: dict[int, list[int]] = {}
    for row in raw_members:
        if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int:
            raise AxBenchBaseError(
                "clustered feature contract has malformed representative members"
            )
        representative = row[0]
        members = _strict_sorted_unique_ids(row[1])
        if (
            members is None
            or not members
            or members[0] != representative
            or representative in members_by_representative
        ):
            raise AxBenchBaseError(
                "clustered feature contract has inconsistent representative members"
            )
        members_by_representative[representative] = members
    if set(members_by_representative) != representative_universe:
        raise AxBenchBaseError(
            "clustered feature contract member mapping does not cover its representative universe"
        )

    if set(concept_source) != {"mode", "domain", "source_to_representative"} or (
        concept_source.get("mode") not in {"explicit_whitelist", "statistics_domain"}
    ):
        raise AxBenchBaseError("clustered feature contract has malformed concept-source semantics")
    source_domain = _strict_sorted_unique_ids(concept_source.get("domain"))
    # Named deviation (PR #739 Track C1, owner-lens _strict_sorted_unique_ids
    # asymmetry, BUG FIX -- see module docstring): the oracle only rejected a
    # MALFORMED domain (`is None`), accepting a well-formed EMPTY one. This
    # falsy check brings it into symmetry with representative_universe's own
    # check above and closes the same E4 non-emptiness gap main's contract
    # builder already enforces upstream, for a sidecar that claims (via
    # fingerprint match) to derive from an honestly-built contract.
    if not source_domain:
        raise AxBenchBaseError(
            "clustered feature contract concept domain must be sorted unique IDs"
        )
    source_domain_set = set(source_domain)
    raw_projection = concept_source.get("source_to_representative")
    if not isinstance(raw_projection, list):
        raise AxBenchBaseError("clustered feature contract source projection must be a list")
    projection: list[tuple[int, int]] = []
    for pair in raw_projection:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
            or pair[0] not in source_domain_set
            or pair[1] not in representative_universe
        ):
            raise AxBenchBaseError("clustered feature contract has malformed source projection")
        projection.append((pair[0], pair[1]))
    projected_sources = [source for source, _representative in projection]
    if projected_sources != sorted(set(projected_sources)):
        raise AxBenchBaseError(
            "clustered feature contract source projection must use sorted unique IDs"
        )
    if concept_source["mode"] == "explicit_whitelist" and projected_sources != source_domain:
        raise AxBenchBaseError(
            "clustered explicit concept whitelist has masked or missing projection"
        )

    expected_representatives = list(
        dict.fromkeys(representative for _source, representative in projection)
    )
    projecting_sources: dict[int, list[int]] = {
        representative: [] for representative in expected_representatives
    }
    for source, representative in projection:
        projecting_sources[representative].append(source)

    provenance_rows = provenance.get("representatives")
    if not isinstance(provenance_rows, list) or len(provenance_rows) != len(
        expected_representatives
    ):
        raise AxBenchBaseError(
            "clustered candidate provenance does not cover every projected "
            "representative exactly once"
        )
    expected_fields = {
        "representative_id",
        "allowed_member_ids",
        "projecting_source_ids",
        "statistics_source_id",
        "outcome",
    }
    final_outcomes = {
        "missing_representative_statistics",
        "below_min_corpus_count",
        "excluded_by_description_fetch_limit",
        "description_unavailable",
        "description_below_quality_threshold",
        "candidate_emitted",
    }
    outcomes: dict[int, str] = {}
    for representative, row in zip(expected_representatives, provenance_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise AxBenchBaseError("clustered candidate provenance representative row is malformed")
        outcome = row.get("outcome")
        recorded_representative = row.get("representative_id")
        recorded_statistics_source = row.get("statistics_source_id")
        recorded_members = _strict_sorted_unique_ids(row.get("allowed_member_ids"))
        recorded_projecting_sources = _strict_sorted_unique_ids(row.get("projecting_source_ids"))
        if (
            type(recorded_representative) is not int
            or recorded_representative != representative
            or recorded_members != members_by_representative[representative]
            or recorded_projecting_sources != projecting_sources[representative]
            or type(recorded_statistics_source) is not int
            or recorded_statistics_source != representative
            or outcome not in final_outcomes
        ):
            raise AxBenchBaseError(
                "clustered candidate provenance has stale representative semantics"
            )
        outcomes[representative] = cast(str, outcome)

    if not isinstance(candidate_rows, list):
        raise AxBenchBaseError("clustered candidate_concepts.json must be a list")
    candidate_ids: list[int] = []
    for row in candidate_rows:
        feature_id = row.get("feature_index") if isinstance(row, Mapping) else None
        if type(feature_id) is not int or feature_id not in representative_universe:
            raise AxBenchBaseError(
                "clustered candidate_concepts.json contains a non-representative ID"
            )
        candidate_ids.append(feature_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AxBenchBaseError("clustered candidate_concepts.json feature IDs must be unique")
    emitted_ids = {
        representative
        for representative, outcome in outcomes.items()
        if outcome == "candidate_emitted"
    }
    if set(candidate_ids) != emitted_ids:
        raise AxBenchBaseError(
            "clustered candidate JSON does not exactly match candidate_emitted provenance outcomes"
        )
    # Named deviation (PR #739 Track C1, Codex cross-review Finding 2, BUG
    # FIX, HIGH -- see module docstring): a committed candidate set that is
    # EMPTY (every representative terminally excluded) is self-consistent
    # with candidate_ids == emitted_ids == set() and all-zero-survivor
    # counts, so the checks above never fire for it -- but it is
    # impossible from the honest producer (max_concepts >= 1 is enforced
    # at config validation time; the oracle's own Stage 0a producer
    # refuses to commit an underfilled population before ever writing
    # these artifacts). representative_universe is already confirmed
    # non-empty above, so an empty candidate set here is anomalous, not a
    # legitimate degenerate case -- fail loud rather than silently accept
    # a self-consistent-but-impossible sidecar.
    if not candidate_ids:
        raise AxBenchBaseError(
            "clustered candidate provenance committed an empty candidate set: no "
            "representative was emitted, which the honest producer never commits"
        )

    counts = provenance.get("counts")
    expected_count_fields = {
        "raw_source_ids",
        "mask_allowed_source_ids",
        "unique_representatives",
        "representatives_with_statistics",
        "count_survivors",
        "description_fetch_survivors",
        "described_survivors",
    }
    if (
        not isinstance(counts, Mapping)
        or set(counts) != expected_count_fields
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise AxBenchBaseError("clustered candidate provenance counts are malformed or partial")
    missing_stats = {"missing_representative_statistics"}
    before_count = missing_stats | {"below_min_corpus_count"}
    before_fetch = before_count | {"excluded_by_description_fetch_limit"}
    expected_counts = {
        "raw_source_ids": len(source_domain),
        "mask_allowed_source_ids": len(projection),
        "unique_representatives": len(expected_representatives),
        "representatives_with_statistics": sum(
            outcome not in missing_stats for outcome in outcomes.values()
        ),
        "count_survivors": sum(outcome not in before_count for outcome in outcomes.values()),
        "description_fetch_survivors": sum(
            outcome not in before_fetch for outcome in outcomes.values()
        ),
        "described_survivors": len(emitted_ids),
    }
    if dict(counts) != expected_counts:
        raise AxBenchBaseError(
            "clustered candidate provenance counts disagree with contract "
            f"semantics: recorded={dict(counts)!r}, expected={expected_counts!r}"
        )


@dataclass(frozen=True)
class ClusterCandidateIdentity:
    feature_contract_fingerprint: str
    candidate_concepts_sha256: str
    candidate_provenance_sha256: str
    concept_projection_version: str


def install_or_validate_cluster_root_marker(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract | None,
) -> None:
    """Install a cluster marker only after preflight, rejecting mixed runs."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cluster root marker requires cfg.run_tag")
    root = paths.AxBenchBasePaths.from_config(cfg).run_output_root()
    marker = paths.AxBenchBasePaths.from_config(cfg).feature_contract_json()
    if root.exists() and not root.is_dir():
        raise AxBenchBaseError(f"clustered run root is not a directory: {root}")
    entries = list(root.iterdir()) if root.exists() else []
    if contract is None:
        if marker.exists():
            raise AxBenchBaseError(
                f"Unclustered run rejects cluster-stamped output at {marker}; use a fresh run_tag"
            )
        return
    if entries and not marker.exists():
        raise AxBenchBaseError(
            f"Clustered output root {root} is nonempty but has no exact feature-contract "
            "marker. Use a fresh run_tag; heuristic repair is forbidden."
        )
    if marker.exists():
        # Named deviation (PR #739 Track C1, Finding 2, resolved as "wrap
        # uniformly" -- see module docstring / task-c1-report.md): the
        # oracle left this call unwrapped, so it could raise either
        # AxBenchBaseError or a propagated ClusterFeatureContractError/
        # FeatureClustersError depending on failure mode -- two unrelated
        # exception hierarchies. This file's dominant internal precedent
        # (load_cluster_candidate_identity's own load_feature_contract_
        # sidecar call, below) already wraps uniformly; its DDP wrapper
        # (install_or_validate_cluster_root_marker_ddp, the entrypoint the
        # 14 documented future consumers actually import) already
        # normalizes both types into AxBenchBaseError too. This brings the
        # bare, also-public function into line with both.
        try:
            load_feature_contract_sidecar(
                marker,
                expected_fingerprint=contract.feature_contract_fingerprint,
            )
        except Exception as exc:
            raise AxBenchBaseError(
                f"Invalid clustered feature-contract marker {marker}: {exc}"
            ) from exc
        return
    try:
        write_feature_contract_sidecar(marker, contract)
    except Exception as exc:
        raise AxBenchBaseError(
            f"Could not commit clustered feature-contract marker {marker}: {exc}"
        ) from exc


def install_or_validate_cluster_root_marker_ddp(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Let rank 0 own the marker and broadcast its canonical outcome."""

    error: str | None = None
    if rank == 0:
        try:
            install_or_validate_cluster_root_marker(cfg, contract)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    payload: list[str | None] = [error]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise AxBenchBaseError(f"cluster run-marker validation failed: {payload[0]}")


def load_cluster_candidate_identity(
    cfg: AxBenchBaseConfig,
    *,
    candidate_path: Path | None = None,
    expected_fingerprint: str | None = None,
) -> ClusterCandidateIdentity | None:
    """Validate and bind the Stage 0a clustered candidate identity.

    The provenance mapping gets an immediate full-digest post-read snapshot,
    and after candidate loading and semantic validation the root marker,
    provenance sidecar, and candidate concepts are re-verified sequentially.
    Each artifact is verified only at the moments observed; the three final
    checks are not an atomic snapshot, an artifact may change after its own
    final check, and no file-state guarantee extends past this function's
    return. Subject to the parsed-marker caveat below, every schedule in which
    a checkpoint observes an artifact difference is rejected, strictly more
    than the pre-FU-B code detected.

    The root marker's final equality check compares parsed documents, so JSON
    aliases such as ``1`` versus ``true`` or ``1`` versus ``1.0`` in
    ``schema_version`` or ``provenance`` may compare equal despite different
    bytes. The identity-bearing ``semantic`` block is not exposed: its
    fingerprint is recomputed as ``_sha256_json(raw["semantic"])`` by
    ``validate_feature_contract_document`` and checked against the expected
    fingerprint on every load, so this aliasing cannot make the returned
    ``feature_contract_fingerprint`` incorrect.
    """

    if cfg.run_tag is None:
        if not clustered_source_enabled(cfg):
            return None
        raise AxBenchBaseError("cluster candidate identity requires cfg.run_tag")
    marker = paths.AxBenchBasePaths.from_config(cfg).feature_contract_json()
    provenance_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).candidate_concepts_cluster_provenance_json()
    if not clustered_source_enabled(cfg):
        if marker.exists() or provenance_path.exists():
            raise AxBenchBaseError(
                "Unclustered AxBench reader rejects cluster-stamped candidate artifacts"
            )
        return None
    try:
        feature_contract = load_feature_contract_sidecar(
            marker,
            expected_fingerprint=expected_fingerprint,
        )
    except Exception as exc:
        raise AxBenchBaseError(
            f"Invalid clustered feature-contract marker {marker}: {exc}"
        ) from exc
    # Named deviation (PR #739 Track C1, Finding 1 simplification -- see
    # module docstring): reuses cluster_checkpoint_pin._load_json_mapping
    # (dict-shaped, fused hash-once/read-once/verify-once) in place of the
    # oracle's _load_stable_json for this sidecar. Any exception is caught
    # and re-raised as AxBenchBaseError below, exactly as the oracle's own
    # broad `except Exception` already did for _load_stable_json.
    try:
        raw, provenance_digest = _load_json_mapping(
            provenance_path,
            surface="candidate provenance",
        )
        provenance_after = stream_hash_file_stable(provenance_path)
    except Exception as exc:
        raise AxBenchBaseError(
            f"Clustered candidate JSON is incomplete without valid commit marker "
            f"{provenance_path}: {exc}"
        ) from exc
    if provenance_digest != provenance_after:
        raise AxBenchBaseError(
            "clustered candidate provenance changed while it was being loaded"
        )
    # No isinstance(raw, Mapping) check here: _load_json_mapping's own
    # return type (tuple[dict[str, Any], StableFileDigest]) already
    # guarantees `raw` is a dict, raising internally otherwise (caught and
    # re-raised above) -- the oracle's downstream check here is now
    # unreachable and has been removed rather than kept as dead code.
    provenance_sha256 = provenance_digest.sha256
    required = {
        "schema_version",
        "feature_contract_fingerprint",
        "candidate_concepts_sha256",
        "concept_projection_version",
        "counts",
        "representatives",
    }
    if set(raw) != required:
        raise AxBenchBaseError(
            "clustered candidate provenance has wrong fields: "
            f"expected={sorted(required)}, actual={sorted(raw, key=str)}"
        )
    if raw["schema_version"] != CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION:
        raise AxBenchBaseError(
            f"unsupported clustered candidate provenance schema {raw['schema_version']!r}"
        )
    fingerprint = feature_contract["feature_contract_fingerprint"]
    if raw["feature_contract_fingerprint"] != fingerprint:
        raise AxBenchBaseError("clustered candidate provenance and root feature contract disagree")
    semantic = cast(dict[str, Any], feature_contract["semantic"])
    algorithms = semantic.get("algorithms")
    if not isinstance(algorithms, Mapping) or (
        raw["concept_projection_version"] != algorithms.get("concept_projection")
    ):
        raise AxBenchBaseError(
            "clustered candidate provenance has the wrong concept-projection version"
        )
    candidate = candidate_path or paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    candidate_rows, actual_candidate_sha = _load_stable_json(
        candidate,
        surface="candidate concepts",
    )
    if raw["candidate_concepts_sha256"] != actual_candidate_sha:
        raise AxBenchBaseError(
            "clustered candidate provenance does not match candidate_concepts.json bytes"
        )
    validate_cluster_candidate_provenance_semantics(
        feature_contract=cast(Mapping[str, Any], feature_contract),
        candidate_rows=candidate_rows,
        provenance=cast(Mapping[str, Any], raw),
    )
    try:
        final_contract = load_feature_contract_sidecar(
            marker, expected_fingerprint=str(fingerprint)
        )
    except Exception as exc:
        raise AxBenchBaseError(
            f"Invalid clustered feature-contract marker {marker}: {exc}"
        ) from exc
    if final_contract != feature_contract:
        raise AxBenchBaseError(
            "clustered feature-contract marker changed while it was being loaded"
        )
    try:
        provenance_final = stream_hash_file_stable(provenance_path)
        candidate_final = stream_hash_file_stable(candidate)
    except Exception as exc:
        raise AxBenchBaseError(
            f"Could not re-verify clustered candidate artifacts before "
            f"committing their identity: {exc}"
        ) from exc
    if provenance_final.sha256 != provenance_sha256:
        raise AxBenchBaseError(
            "clustered candidate provenance changed while it was being loaded"
        )
    if candidate_final.sha256 != actual_candidate_sha:
        raise AxBenchBaseError(
            "clustered candidate concepts changed while it was being loaded"
        )
    return ClusterCandidateIdentity(
        feature_contract_fingerprint=str(fingerprint),
        candidate_concepts_sha256=actual_candidate_sha,
        candidate_provenance_sha256=provenance_sha256,
        concept_projection_version=str(raw["concept_projection_version"]),
    )


# ---------------------------------------------------------------------------
# Live feature-statistics identity re-validation (PR #739 FU-D-stage0c fix
# batch, Bug 1 -- HIGH, Codex cross-review of C4).
#
# The oracle-inherited gap: ``ClusterFeatureContract.statistics`` already
# pins the feature-statistics file's content identity (SHA-256 + rebuilt
# percentile-table checksum -- ``cluster_contract_identity.py``'s
# ``FeatureStatisticsIdentity``, folded into the contract fingerprint by
# ``cluster_feature_contract.py``'s ``build_cluster_feature_contract``), but
# every Stage 0c resume/merge/candidate-pool check reconstructs its
# "expected" fingerprint from ``_compute_checkpoint_fingerprint``
# (``_stage0c_internals.py``), which records only ``cfg.feature_
# statistics_path`` -- the path STRING -- never the file's actual bytes.
# Replacing the statistics file at the same path (same width/domain,
# different content) after a clustered contract was resolved therefore went
# completely undetected: a resume would silently accept partial rows
# verified under the ORIGINAL statistics while building its percentile
# table from the REPLACED bytes.
#
# The two functions below close that gap by recomputing the LIVE on-disk
# statistics identity (reusing ``cluster_checkpoint_pin._statistics_
# identity`` -- the exact primitive the contract itself used to pin
# ``FeatureStatisticsIdentity`` in the first place, not a reimplementation)
# and requiring it to exactly equal what was pinned. Both are unconditional
# no-ops when clustering is disabled (``clustered_source_enabled(cfg)`` is
# False): this is a clustered-contract concern only, and the unclustered
# path must remain byte-identical (U-invariant).
# ---------------------------------------------------------------------------


def _assert_live_statistics_identity_snapshot(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
    pinned_statistics: Mapping[str, Any],
    surface: str,
) -> Any:
    """Shared single-read core of the live-statistics identity guard.

    Reads the on-disk statistics file EXACTLY ONCE (via
    ``cluster_checkpoint_pin._statistics_identity``, which fuses the
    hash+read+parse -- see ``_load_statistics_snapshot``), recomputes its
    identity, requires it to equal ``pinned_statistics``, and returns the
    loaded statistics snapshot so a caller can build a percentile table from
    the SAME bytes the identity was just verified from (no separate reopen,
    hence no hash-then-reopen TOCTOU between the check and the build).

    Callers must have already confirmed clustering is enabled; this helper
    performs no ``clustered_source_enabled`` gate of its own.
    """

    try:
        feature_dir = Path(cfg.feature_dir).expanduser().resolve(strict=True)
        statistics_path = Path(cfg.feature_statistics_path).expanduser().resolve(
            strict=True
        )
        live_statistics, _directory_identity, stats = _statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=n_sae_features,
        )
    except Exception as exc:
        raise AxBenchBaseError(
            f"{surface}: could not recompute the live clustered feature-"
            f"statistics identity at {cfg.feature_statistics_path!r}: {exc}"
        ) from exc
    live_dict = live_statistics.canonical_dict()
    pinned_dict = dict(pinned_statistics)
    if live_dict != pinned_dict:
        raise AxBenchBaseError(
            f"{surface}: live feature statistics at {statistics_path} no "
            "longer match the identity pinned by the clustered feature "
            "contract (file SHA-256 and/or rebuilt percentile-table "
            "checksum changed). Rows verified/committed under the original "
            "statistics cannot be safely reused or merged against a "
            "percentile table rebuilt from the new bytes. Use a fresh "
            "run_tag, or restore the original statistics file. "
            f"live={live_dict!r}, pinned={pinned_dict!r}"
        )
    return stats


def assert_live_statistics_identity(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
    pinned_statistics: Mapping[str, Any],
    surface: str,
) -> None:
    """Recompute the live feature-statistics identity and require it to
    exactly equal ``pinned_statistics`` -- typically a clustered feature
    contract's own ``FeatureStatisticsIdentity.canonical_dict()``.

    No-op unless clustering is enabled for ``cfg``. Raises
    :class:`AxBenchBaseError` (fail-loud, not a silent mix) the moment the
    on-disk statistics file's content (SHA-256 and/or its rebuilt
    percentile-table checksum) disagrees with what was pinned -- BEFORE any
    partial rows are accepted or a percentile table built from the live
    file is trusted. On the normal path (statistics genuinely unchanged),
    recomputing the identity from the same bytes reproduces the pinned
    values exactly, so this is a strict no-op.
    """

    if not clustered_source_enabled(cfg):
        return
    _assert_live_statistics_identity_snapshot(
        cfg,
        n_sae_features=n_sae_features,
        pinned_statistics=pinned_statistics,
        surface=surface,
    )


def assert_live_statistics_identity_and_build_percentile_table(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
    pinned_statistics: Mapping[str, Any],
    surface: str,
) -> t.Tensor | None:
    """Fused single-read variant of :func:`assert_live_statistics_identity`
    that ALSO returns the percentile table built from the SAME verified
    bytes.

    This exists to close the FRESH-run hole in the FU-D-stage0c fix batch
    (Codex cross-review, Bug 1 HIGH). ``assert_live_statistics_identity``
    guards the partial-resume and candidate-pool paths, but a fresh
    clustered Stage 0c run (no partial checkpoint) reaches the percentile-
    table build (``dispatch_prepare_negative_prefixes``) having only passed
    the collective candidate-pool check -- an A->B statistics swap in the
    window before the table build would then build the consumed table from B
    with no identity re-check. Binding the table to the validated snapshot
    HERE (one read: identity verified from the same ``stats`` the table is
    built from) removes that hash-then-reopen TOCTOU entirely.

    Returns ``None`` when clustering is disabled for ``cfg`` so callers fall
    back to their standard percentile-table builder and the unclustered path
    stays byte-identical (U-invariant). When clustering is enabled, raises
    :class:`AxBenchBaseError` on any mismatch with ``pinned_statistics``
    (identical message/semantics to :func:`assert_live_statistics_identity`),
    and otherwise returns ``build_percentile_table(stats.per_feature_stats,
    n_sae_features)`` -- the same deterministic builder
    ``extraction.build_percentile_table_from_stats`` wraps, so the table is
    byte-identical to the pre-fix clustered path, only now provably built
    from the validated bytes.
    """

    if not clustered_source_enabled(cfg):
        return None
    stats = _assert_live_statistics_identity_snapshot(
        cfg,
        n_sae_features=n_sae_features,
        pinned_statistics=pinned_statistics,
        surface=surface,
    )
    try:
        return build_percentile_table(stats.per_feature_stats, n_sae_features)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise AxBenchBaseError(
            f"{surface}: could not build the percentile table from the "
            "verified live clustered feature statistics at "
            f"{cfg.feature_statistics_path!r}: {exc}"
        ) from exc


def assert_live_statistics_match_committed_marker(
    cfg: AxBenchBaseConfig,
    *,
    n_sae_features: int,
    surface: str,
) -> None:
    """Like :func:`assert_live_statistics_identity`, for callers that
    intentionally avoid resolving a full :class:`ClusterFeatureContract`.

    The Stage 0c completed-resume CLI check
    (``negative_prefixes.resume_negative_prefixes_if_complete``) is
    deliberately cheap: it must decide whether to reuse completed artifacts
    without loading a checkpoint, SAE, or decoder-only target. Rather than
    require an in-memory contract, this reads the already-committed
    ``feature_contract.json`` root marker directly (a small JSON read +
    self-consistency check -- no heavier than what ``load_cluster_
    candidate_identity`` already does for the same marker) and takes its
    pinned ``statistics`` block as ground truth.

    No-op unless clustering is enabled for ``cfg``.
    """

    if not clustered_source_enabled(cfg):
        return
    if cfg.run_tag is None:
        raise AxBenchBaseError(f"{surface} requires cfg.run_tag")
    marker = paths.AxBenchBasePaths.from_config(cfg).feature_contract_json()
    try:
        feature_contract = load_feature_contract_sidecar(marker)
    except Exception as exc:
        raise AxBenchBaseError(
            f"{surface}: could not load the committed clustered feature-"
            f"contract marker {marker}: {exc}"
        ) from exc
    semantic = feature_contract.get("semantic")
    pinned_statistics = (
        semantic.get("statistics") if isinstance(semantic, Mapping) else None
    )
    if not isinstance(pinned_statistics, Mapping):
        raise AxBenchBaseError(
            f"{surface}: committed feature-contract marker {marker} is "
            "missing its pinned statistics identity"
        )
    assert_live_statistics_identity(
        cfg,
        n_sae_features=n_sae_features,
        pinned_statistics=pinned_statistics,
        surface=surface,
    )


# ---------------------------------------------------------------------------
# Collective candidate-identity agreement (PR #739 FU-D-stage0c fix batch,
# Bug 2 -- MEDIUM, Codex cross-review of C4).
#
# ``load_cluster_candidate_identity`` (and, as of the Bug 1 fix above,
# ``assert_live_statistics_identity``) used to run as a bare per-rank local
# call at both ``negative_prefixes.prepare_negative_prefixes`` and
# ``stage0c_runner.dispatch_prepare_negative_prefixes``, immediately BEFORE
# a DDP collective (the resume broadcast / the block-barrier ``all_gather_
# object`` rendezvous). A single rank hitting a stale/missing NFS view of
# the Stage 0a sidecar (``cluster_artifacts.py``'s own file-identity checks
# above) -- or, now, a genuine live-statistics mismatch -- would raise
# locally while its peers proceeded into that collective and blocked until
# the process-group timeout. ``stage0c_runner._load_candidate_pool_with_
# collective_identity_agreement`` already solved the exact same shape of
# problem for the candidate-pool load; the function below mirrors it
# exactly (do not invent a new protocol): every rank attempts its own local
# validation, then all ranks post a (ok, error) status via ``all_gather_
# object`` and raise TOGETHER if any rank failed.
# ---------------------------------------------------------------------------


def validate_cluster_candidate_identity_with_collective_agreement(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract,
    *,
    candidate_path: Path,
    n_sae_features: int,
    world_size: int,
) -> ClusterCandidateIdentity:
    """Validate the clustered candidate/provenance sidecar AND the live
    feature-statistics identity, then make every rank agree before any
    rank proceeds into a DDP collective. Returns the validated identity.

    At ``world_size <= 1`` (or an uninitialized process group) this is
    exactly the two local calls it wraps -- no ``torch.distributed`` usage
    at all, mirroring ``_load_candidate_pool_with_collective_identity_
    agreement``'s own single-rank short-circuit. Under multi-rank, when
    every rank's local validation agrees (the normal, consistent-view
    case), every rank returns normally with no behavior change versus the
    pre-fix bare local calls -- the fix only changes what happens on
    disagreement: instead of one rank raising alone, every rank raises the
    same collective failure summary.

    PR #739 FU-D-2: this is the ONE place the candidate identity is
    resolved for the whole rank's run. The caller (``stage0c_runner.
    dispatch_prepare_negative_prefixes``) derives ``precomputed_clustered_
    fields`` ONCE from the returned identity (via
    ``_clustered_fingerprint_fields_from_identity``) and threads that dict
    to every hot/per-rank consumer downstream instead of letting them each
    independently re-read the candidate/provenance sidecar -- see the
    "Collective candidate-identity agreement" comment block above this
    function for the full class this closes.

    On agreement, each rank returns its OWN locally-validated identity
    rather than broadcasting rank 0's -- mirroring ``build_verified_
    percentile_table_with_collective_agreement``'s own reasoning (below):
    every rank that reached "ok" validated its live sidecar bytes against
    the exact same ``contract.feature_contract_fingerprint`` pinned value,
    and the identity fields actually consumed downstream (see
    ``_clustered_fingerprint_fields_from_identity``) are read from that
    same contract-scoped marker/provenance sidecar, not from
    ``candidate_path`` -- so an "ok" local read is provably identical
    across every agreeing rank under the same honest-single-writer
    assumption the sibling collectives already rely on.
    """

    def _validate_locally() -> ClusterCandidateIdentity:
        identity = load_cluster_candidate_identity(
            cfg,
            candidate_path=candidate_path,
            expected_fingerprint=contract.feature_contract_fingerprint,
        )
        if identity is None:
            # Invariant violation, not a normal unclustered no-op: both
            # real call sites only invoke this function when
            # ``cluster_contract is not None`` (which they also require to
            # imply ``clustered_source_enabled(cfg)``). A caller that
            # violates this gets a loud raise here rather than silently
            # threading empty fingerprint fields downstream.
            raise AxBenchBaseError(
                "Clustered Stage 0c candidate-identity preflight requires "
                "a resolved cluster contract, but cfg is not clustered-"
                "enabled; this collective wrapper must only be called "
                "when cluster_contract is not None"
            )
        assert_live_statistics_identity(
            cfg,
            n_sae_features=n_sae_features,
            pinned_statistics=contract.statistics.canonical_dict(),
            surface="Clustered Stage 0c candidate-pool preflight",
        )
        return identity

    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return _validate_locally()

    local_identity: ClusterCandidateIdentity | None = None
    try:
        local_identity = _validate_locally()
        local_status: dict[str, Any] = {
            "ok": True,
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        local_status = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_status)
    malformed_ranks = [
        gathered_rank
        for gathered_rank, status in enumerate(gathered)
        if not isinstance(status, dict)
    ]
    if malformed_ranks:
        raise AxBenchBaseError(
            "Clustered Stage 0c candidate-identity collective returned "
            f"malformed state for ranks {malformed_ranks!r}"
        )
    statuses = cast(list[dict[str, Any]], gathered)
    failed = [
        (gathered_rank, status.get("error_type"), status.get("error"))
        for gathered_rank, status in enumerate(statuses)
        if status.get("ok") is not True
    ]
    if failed:
        raise AxBenchBaseError(
            "Clustered Stage 0c candidate/statistics identity check failed "
            f"collectively before rank work: {failed!r}"
        )
    if local_identity is None:
        # Structurally unreachable: this rank posted ok=True above (else the
        # collective raise fired), and ok=True is only set after
        # local_identity was assigned. An explicit raise (not `assert`, which
        # `python -O` strips) so a future protocol bug fails loud here rather
        # than returning None into a non-Optional signature.
        raise AxBenchBaseError(
            "Clustered Stage 0c candidate-identity vote reported all ranks "
            "ok but this rank holds no locally-validated identity -- vote "
            "protocol bug"
        )
    return local_identity


# ---------------------------------------------------------------------------
# Collective agreement for the FUSED percentile-table build (PR #739
# FU-D-stage0c fix batch, Codex RE-CHECK of the Bug 1 HIGH fix -- MEDIUM:
# the fresh-run fused table build reintroduced a Bug-2-class deadlock).
#
# ``assert_live_statistics_identity_and_build_percentile_table`` (the Bug 1
# HIGH follow-up above) closed the fresh-run TOCTOU hole by re-verifying the
# live statistics identity immediately before building the percentile table
# CONSUMED downstream -- but it does so as a bare PER-RANK LOCAL call at its
# one call site (``stage0c_runner.dispatch_prepare_negative_prefixes``),
# immediately BEFORE the dispatcher's later collectives (the candidate-pool
# rendezvous, then ``run_single_rank``'s own block-barrier ``all_gather_
# object``). Both ranks pass the EARLIER collective preflight (``validate_
# cluster_candidate_identity_with_collective_agreement``, above) on
# statistics A; then, in the independent window before each rank's own
# fused read, one rank can still observe a swapped/stale view B while its
# peer observes A. The B-observing rank raises locally and dies; the
# A-observing rank sails on into the next real collective and blocks
# forever waiting for a peer that will never arrive -- the EXACT one-rank-
# raises/peer-enters-collective deadlock the Bug 2 wrapper above exists to
# prevent, just reintroduced at a new call site the Bug 1 follow-up added
# after Bug 2 was already fixed.
#
# The function below closes that gap the same way: every rank attempts its
# own local fused read+verify+build, then ALL ranks post a (ok, error)
# status via an UNCONDITIONAL ``all_gather_object`` -- called regardless of
# local outcome, so a raising rank still rendezvous with its peers instead
# of dying silently -- and every rank raises TOGETHER if any rank failed.
# On all-agree, each rank returns its OWN locally-built table rather than
# broadcasting one: every rank that reached "ok" verified its live bytes
# against the exact same ``contract.statistics.canonical_dict()`` pinned
# value, so an "ok" local read's SHA-256 (and hence its deterministically
# rebuilt percentile table) is provably identical across every agreeing
# rank -- there is nothing left to disagree on past that point.
# ---------------------------------------------------------------------------


def build_verified_percentile_table_with_collective_agreement(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract,
    *,
    n_sae_features: int,
    surface: str,
    world_size: int,
) -> t.Tensor:
    """Fused, identity-verified percentile-table build, made collectively
    failure-agreed so a single rank's local read+verify+build can never
    raise alone while its peers proceed into a later DDP collective.

    At ``world_size <= 1`` (or an uninitialized process group) this is
    exactly the wrapped local call -- no ``torch.distributed`` usage at
    all, mirroring ``validate_cluster_candidate_identity_with_collective_
    agreement``'s own single-rank short-circuit. Under multi-rank, when
    every rank's local build agrees (the normal, consistent-view case,
    including the ordinary unchanged-statistics path), every rank returns
    its own locally-built table with no behavior change versus the pre-fix
    bare local call -- the fix only changes what happens on disagreement:
    instead of one rank raising alone, every rank raises the same
    collective failure summary, and NO rank can reach a later collective
    while a peer is dead.

    Delegates the actual read+verify+build to :func:`assert_live_
    statistics_identity_and_build_percentile_table` (the Bug 1 HIGH fix) --
    this wrapper only adds collective failure-agreement around it, it does
    not change how the table is verified or built. Callers must only invoke
    this when clustering is enabled for ``cfg`` (mirroring ``validate_
    cluster_candidate_identity_with_collective_agreement``'s own contract);
    a caller that violates this gets a loud :class:`AxBenchBaseError`
    rather than a silently swallowed ``None``.
    """

    def _build_locally() -> t.Tensor:
        table = assert_live_statistics_identity_and_build_percentile_table(
            cfg,
            n_sae_features=n_sae_features,
            pinned_statistics=contract.statistics.canonical_dict(),
            surface=surface,
        )
        if table is None:
            raise AxBenchBaseError(
                f"{surface}: expected a verified clustered percentile table "
                "but clustering is not enabled for cfg; this collective "
                "wrapper must only be called when cluster_contract is set"
            )
        return table

    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return _build_locally()

    local_table: t.Tensor | None = None
    try:
        local_table = _build_locally()
        local_status: dict[str, Any] = {
            "ok": True,
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        local_status = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    # Unconditional: every rank posts its status here regardless of local
    # outcome, so a rank whose local build raised still rendezvous with its
    # peers here instead of dying without ever reaching a collective.
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_status)
    malformed_ranks = [
        gathered_rank
        for gathered_rank, status in enumerate(gathered)
        if not isinstance(status, dict)
    ]
    if malformed_ranks:
        raise AxBenchBaseError(
            "Clustered Stage 0c percentile-table collective returned "
            f"malformed state for ranks {malformed_ranks!r}"
        )
    statuses = cast(list[dict[str, Any]], gathered)
    failed = [
        (gathered_rank, status.get("error_type"), status.get("error"))
        for gathered_rank, status in enumerate(statuses)
        if status.get("ok") is not True
    ]
    if failed:
        raise AxBenchBaseError(
            "Clustered Stage 0c percentile-table identity check failed "
            f"collectively before rank work: {failed!r}"
        )
    assert local_table is not None
    return local_table


# ---------------------------------------------------------------------------
# Collective agreement for the RESUME-path per-rank live-statistics check
# (PR #739 FU-D-stage0c fix batch, follow-up closing the LAST known deadlock
# analog: ``stage0c_runner._try_resume_per_rank``'s own copy of the Bug 1
# check).
#
# ``_try_resume_per_rank`` re-validates the live statistics identity against
# the pinned contract immediately before trusting a rank's own partial
# checkpoint (accepted/rejected/verified rows from a PRIOR run) -- but only
# for a rank that actually has a partial checkpoint to resume (``json_path.
# exists()``), and, until this fix, only as a bare PER-RANK LOCAL call.
# ``run_single_rank`` unconditionally proceeds into the block-barrier loop's
# ``_all_gather_state`` collective at least once after this check runs --
# even a rank whose stripe has nothing left to do still reaches it -- so a
# rank that raised here alone would strand every peer that did not observe
# the same swap in the exact same window, INCLUDING a peer with no local
# checkpoint at all (whose copy of this check was, pre-fix, never even
# invoked). This is the same one-rank-raises/peer-enters-collective
# deadlock shape as ``validate_cluster_candidate_identity_with_collective_
# agreement`` and ``build_verified_percentile_table_with_collective_
# agreement`` above close for their own call sites.
#
# This call site survived both of those fixes because it was reasoned to
# always run strictly after an already-collectively-agreed earlier check
# (the dispatcher's own candidate-identity preflight, and now also the
# fused percentile-table build) in the same process invocation -- see this
# fix batch's own report, "Anything else out of scope" section, which
# named this exact residual and flagged it honestly rather than asserting
# it was already covered. That reasoning does not hold: both of those
# earlier collectives complete BEFORE ``run_single_rank`` is even called,
# and non-trivial, per-rank-timing-dependent LOCAL work happens in
# between -- at minimum building the candidate prefix index from the live
# parquet/feature-shard files, optionally also constructing a QC client
# and/or API generator. An A->B statistics swap landing in that window can
# still be observed by only one rank's resume check, exactly the kind of
# TOCTOU window the fused-table-build fix closed for the fresh-run path.
#
# The function below closes it the same way as its two siblings: every
# rank attempts its own local validation (a no-op "trivially ok" pass for a
# rank with no local checkpoint to validate), then ALL ranks post a (ok,
# error) status via an UNCONDITIONAL ``all_gather_object`` -- called by
# every rank regardless of whether it had anything to locally validate, so
# a rank with no checkpoint still rendezvous with a resuming peer instead
# of racing ahead of it -- and every rank raises TOGETHER if any rank
# failed.
# ---------------------------------------------------------------------------


def validate_resume_statistics_identity_with_collective_agreement(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract,
    *,
    has_local_checkpoint: bool,
    n_sae_features: int,
    world_size: int,
    surface: str,
) -> None:
    """Re-validate a resuming rank's live-statistics identity, made
    collectively failure-agreed so a single rank's local per-rank
    checkpoint-resume check can never raise alone while a peer proceeds
    into ``run_single_rank``'s later block-barrier collective.

    ``has_local_checkpoint`` is the caller's own ``json_path.exists()``
    result for THIS rank. A rank with no local checkpoint has nothing to
    validate -- there is no partial resumed state to protect -- so its
    local step below is a trivial pass. It must still CALL this function
    unconditionally whenever ``cluster_contract is not None`` regardless of
    its own ``has_local_checkpoint`` value: a rank that skipped the call
    entirely because it had nothing to check would strand a resuming peer
    that does call it, waiting forever on ``all_gather_object`` for a
    party that never arrives -- indistinguishable from the exact deadlock
    this function exists to prevent.

    At ``world_size <= 1`` (or an uninitialized process group) this is
    exactly the wrapped local call -- no ``torch.distributed`` usage at
    all, mirroring the sibling collective-agreement helpers' own
    single-rank short-circuit. Under multi-rank, when every rank agrees
    (the normal case -- including every rank with no local checkpoint, and
    a genuine resume whose statistics are unchanged), every rank returns
    normally with no behavior change versus the pre-fix bare local call --
    the fix only changes what happens on disagreement: instead of one rank
    raising alone, every rank raises the same collective failure summary,
    and no rank can reach a later collective while a peer is dead.
    """

    def _validate_locally() -> None:
        if not has_local_checkpoint:
            return
        assert_live_statistics_identity(
            cfg,
            n_sae_features=n_sae_features,
            pinned_statistics=contract.statistics.canonical_dict(),
            surface=surface,
        )

    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        _validate_locally()
        return

    try:
        _validate_locally()
        local_status: dict[str, Any] = {
            "ok": True,
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        local_status = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    # Unconditional: every rank posts its status here regardless of local
    # outcome (including a rank with no local checkpoint, whose local step
    # above is a no-op "ok"), so no rank can skip this rendezvous.
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_status)
    malformed_ranks = [
        gathered_rank
        for gathered_rank, status in enumerate(gathered)
        if not isinstance(status, dict)
    ]
    if malformed_ranks:
        raise AxBenchBaseError(
            "Clustered Stage 0c resume-checkpoint statistics collective "
            f"returned malformed state for ranks {malformed_ranks!r}"
        )
    statuses = cast(list[dict[str, Any]], gathered)
    failed = [
        (gathered_rank, status.get("error_type"), status.get("error"))
        for gathered_rank, status in enumerate(statuses)
        if status.get("ok") is not True
    ]
    if failed:
        raise AxBenchBaseError(
            "Clustered Stage 0c resume-checkpoint statistics identity "
            f"check failed collectively before rank work: {failed!r}"
        )


def _clustered_fingerprint_fields_from_identity(
    identity: ClusterCandidateIdentity,
) -> dict[str, str]:
    """Pure derivation of the clustered Stage 0c fingerprint-fields dict
    from an already-validated :class:`ClusterCandidateIdentity` -- no I/O.

    PR #739 FU-D-2: this is the ONE place the field subset/naming is
    decided. Both :func:`clustered_stage0c_fingerprint_fields` (fresh-read
    path, below) and ``stage0c_runner.dispatch_prepare_negative_prefixes``'s
    startup collective vote (cached path -- see ``validate_cluster_
    candidate_identity_with_collective_agreement``'s own docstring) call
    this so the two can never independently drift; see
    ``test_clustered_fingerprint_fields_from_identity_matches_fresh_read``
    for the equality proof.

    ``identity.candidate_concepts_sha256`` is deliberately excluded from
    the returned dict -- the candidate pool's own content identity is
    already tracked separately, and more precisely (it hashes the
    in-memory ordered pool rather than ``candidate_concepts.json`` bytes),
    via ``_stage0c_internals.candidate_pool_identity``.

    Takes a real identity only (parameter is deliberately non-Optional):
    the "no clustered sidecar" case is decided by the CALLER that can
    actually observe it (``clustered_stage0c_fingerprint_fields`` maps a
    ``None`` load result to ``{}``), not by a nullable pass-through here.
    """
    return {
        "feature_contract_fingerprint": identity.feature_contract_fingerprint,
        "candidate_cluster_provenance_sha256": identity.candidate_provenance_sha256,
        "cluster_concept_projection_version": identity.concept_projection_version,
    }


def clustered_stage0c_fingerprint_fields(
    cfg: AxBenchBaseConfig,
) -> dict[str, str]:
    """Fresh-read variant: loads the candidate identity from disk on every
    call.

    PR #739 FU-D-2: used only by callers with no already-validated
    identity in hand -- ``stage0c_runner.run_merge``'s and
    ``negative_prefixes._resume_if_artifacts_complete``'s own live
    re-reads, which are intentionally NOT cached (they are the staleness-
    detection story: identity is pinned collectively at startup; a mid-run
    sidecar mutation is still caught at merge/completed-resume time by
    these live re-reads). Hot/per-rank consumers on the ``dispatch_
    prepare_negative_prefixes`` -> ``run_single_rank`` path instead receive
    an already-derived dict via ``precomputed_clustered_fields`` threading
    and never call this function at all.
    """
    identity = load_cluster_candidate_identity(cfg)
    if identity is None:
        # Not a clustered-source cfg: no sidecar exists and no clustered
        # fields belong in the fingerprint (U-invariant: unclustered
        # fingerprints stay byte-identical to legacy).
        return {}
    return _clustered_fingerprint_fields_from_identity(identity)


def _cluster_generation_marker_payload(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract,
    *,
    contexts_fingerprint: str,
    generation_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = load_cluster_candidate_identity(
        cfg,
        expected_fingerprint=contract.feature_contract_fingerprint,
    )
    if candidate is None:
        raise AxBenchBaseError("Clustered Stage 4 requires committed candidate provenance")
    if cfg.run_tag is None:
        raise AxBenchBaseError("cluster generation identity requires cfg.run_tag")
    try:
        canonical_inputs = json.dumps(
            generation_inputs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        normalized_inputs = json.loads(canonical_inputs)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            "Clustered Stage 4 generation inputs must be finite canonical JSON"
        ) from exc
    if not isinstance(normalized_inputs, dict) or not normalized_inputs:
        raise AxBenchBaseError("Clustered Stage 4 generation inputs must be a nonempty mapping")
    if normalized_inputs.get("feature_contract_fingerprint") != (
        contract.feature_contract_fingerprint
    ):
        raise AxBenchBaseError(
            "Clustered Stage 4 generation inputs do not match the feature contract"
        )
    concepts_sha256 = _sha256_file(paths.AxBenchBasePaths.from_config(cfg).concepts_json())
    artifact_hashes = normalized_inputs.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or (
        artifact_hashes.get("concepts.json") != concepts_sha256
    ):
        raise AxBenchBaseError(
            "Clustered Stage 4 generation inputs do not match final concepts bytes"
        )
    generation_inputs_fingerprint = hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest()
    return {
        "schema_version": CLUSTER_GENERATION_RESUME_SCHEMA_VERSION,
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "contexts_fingerprint": contexts_fingerprint,
        "candidate_concepts_sha256": candidate.candidate_concepts_sha256,
        "candidate_cluster_provenance_sha256": (candidate.candidate_provenance_sha256),
        "cluster_concept_projection_version": (candidate.concept_projection_version),
        "concepts_sha256": concepts_sha256,
        "generation_inputs_fingerprint": generation_inputs_fingerprint,
        "generation_inputs": normalized_inputs,
    }


def validate_cluster_generation_resume(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract | None,
    *,
    contexts_fingerprint: str | None,
    generation_inputs: Mapping[str, Any] | None = None,
) -> None:
    """Reject legacy/mixed generation rows before Stage 4 resume scanning.

    A full-digest snapshot immediately after the marker mapping read brackets
    the consumed document, and a final content re-hash binds it across expected
    payload reconstruction. Success means the marker matched at each moment it
    was observed, not that it still matches at any later instant or after this
    function returns. Every schedule in which a checkpoint observes a marker
    difference is rejected, which is strictly more than the pre-FU-B code
    detected.
    """

    if cfg.run_tag is None:
        raise AxBenchBaseError("Stage 4 generation marker requires cfg.run_tag")
    root = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    marker = paths.AxBenchBasePaths.from_config(cfg).generations_feature_contract_json()
    if contract is None:
        if marker.exists():
            raise AxBenchBaseError("Unclustered Stage 4 rejects cluster-stamped generations")
        return
    if not is_sha256_hex(contexts_fingerprint):
        raise AxBenchBaseError(
            "Clustered Stage 4 requires a lowercase SHA-256 contexts fingerprint"
        )
    assert contexts_fingerprint is not None
    if generation_inputs is None:
        raise AxBenchBaseError("Clustered Stage 4 requires exact generation-input provenance")
    existing = list(root.iterdir()) if root.exists() else []
    if existing and not marker.is_file():
        raise AxBenchBaseError(
            f"Clustered generations at {root} lack their exact feature-contract "
            "resume marker; use a fresh run_tag"
        )
    if not marker.exists():
        return
    # Named deviation (PR #739 Track C1, Finding 1 BUG FIX -- see module
    # docstring): replaces the oracle's own inline hash-before/read/hash-
    # after bracket -- a strictly weaker, third reimplementation of the
    # same TOCTOU pattern that never cross-checked the actual read bytes
    # against the pre-read digest -- with the same fused _load_json_mapping
    # primitive used above. Closes a real content round-trip (A -> B -> A)
    # gap the oracle's before/after-only comparison could miss.
    try:
        actual, marker_digest = _load_json_mapping(
            marker,
            surface="Stage 4 generation resume marker",
        )
        marker_after = stream_hash_file_stable(marker)
    except Exception as exc:
        raise AxBenchBaseError(f"Unreadable clustered generation marker {marker}: {exc}") from exc
    if marker_digest != marker_after:
        raise AxBenchBaseError(
            "clustered Stage 4 generation resume marker changed while it was being loaded"
        )
    expected = _cluster_generation_marker_payload(
        cfg,
        contract,
        contexts_fingerprint=contexts_fingerprint,
        generation_inputs=generation_inputs,
    )
    if actual != expected:
        raise AxBenchBaseError("Clustered generation resume identity differs; use a fresh run_tag")
    try:
        marker_final = stream_hash_file_stable(marker)
    except Exception as exc:
        raise AxBenchBaseError(
            f"Could not re-verify clustered generation marker before resume: {exc}"
        ) from exc
    if marker_final.sha256 != marker_digest.sha256:
        raise AxBenchBaseError(
            "clustered Stage 4 generation resume marker changed while it was being validated"
        )


def commit_cluster_generation_marker(
    cfg: AxBenchBaseConfig,
    contract: _ClusterFeatureContract | None,
    *,
    contexts_fingerprint: str | None,
    generation_inputs: Mapping[str, Any] | None = None,
) -> None:
    """Write the clustered Stage 4 marker only after full generation preflight."""

    validate_cluster_generation_resume(
        cfg,
        contract,
        contexts_fingerprint=contexts_fingerprint,
        generation_inputs=generation_inputs,
    )
    if contract is None:
        return
    assert cfg.run_tag is not None
    assert contexts_fingerprint is not None
    marker = paths.AxBenchBasePaths.from_config(cfg).generations_feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = _cluster_generation_marker_payload(
        cfg,
        contract,
        contexts_fingerprint=contexts_fingerprint,
        generation_inputs=cast(Mapping[str, Any], generation_inputs),
    )
    install_clustered_bytes_once(
        json.dumps(payload, indent=2).encode("utf-8"),
        marker,
        surface="Stage 4 generation resume marker",
    )


# Named deviation (PR #739 Track C1, constraint 4, mechanical completeness
# fix -- plan-authorized, not a fidelity deviation): "CLUSTER_CANDIDATE_
# PROVENANCE_SCHEMA_VERSION" and "validate_cluster_candidate_provenance_
# semantics" below are the two names the oracle's own __all__ omitted
# despite real future consumers importing them directly (concept_list.py,
# axbench_bpb_adv/source_validation.py). Runtime-inert beyond `from module
# import *` and static tooling -- every documented consumer uses a direct
# `from ... import name` either way (see the import-path preservation
# test, which asserts direct-import resolution, not just __all__
# membership).
__all__ = [
    "CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION",
    "CLUSTER_GENERATION_RESUME_SCHEMA_VERSION",
    "ClusterCandidateIdentity",
    "assert_live_statistics_identity",
    "assert_live_statistics_identity_and_build_percentile_table",
    "assert_live_statistics_match_committed_marker",
    "build_verified_percentile_table_with_collective_agreement",
    "clustered_source_enabled",
    "clustered_stage0c_fingerprint_fields",
    "commit_cluster_generation_marker",
    "install_clustered_bytes_once",
    "install_or_validate_cluster_root_marker",
    "install_or_validate_cluster_root_marker_ddp",
    "load_cluster_candidate_identity",
    "validate_cluster_candidate_identity_with_collective_agreement",
    "validate_cluster_candidate_provenance_semantics",
    "validate_cluster_generation_resume",
    "validate_resume_statistics_identity_with_collective_agreement",
    "write_clustered_parquet_once",
    "write_clustered_safetensors_once",
]
