"""Conditional clustered identity for BPB-Adv Parquet artifacts.

Schema metadata is deliberately absent for legacy/unclustered artifacts.  A
clustered writer binds every resumable table to both the exact source contract
and the resolved feature contract; readers validate the binding before using
any rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_family import ALPHA_AXIS_GCG_METHODS
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
from steering.eval.cluster_feature_contract import stream_hash_file_stable
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    is_sha256_hex,
    stream_hash_file_descriptor_bound,
)

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


FEATURE_CONTRACT_METADATA_KEY = b"feature_contract_fingerprint"
SOURCE_CONTRACT_METADATA_KEY = b"source_contract_hash"
ARTIFACT_FAMILY_METADATA_KEY = b"clustered_artifact_family"
PRECOMPUTED_BAG_SET_METADATA_KEY = b"precomputed_bag_set_sha256"
UPSTREAM_CANDIDATES_METADATA_KEY = b"upstream_candidates_sha256"
_RESERVED_KEYS = frozenset(
    {
        FEATURE_CONTRACT_METADATA_KEY,
        SOURCE_CONTRACT_METADATA_KEY,
        ARTIFACT_FAMILY_METADATA_KEY,
        PRECOMPUTED_BAG_SET_METADATA_KEY,
        UPSTREAM_CANDIDATES_METADATA_KEY,
    }
)

_CANDIDATE_DEPENDENT_FAMILIES = frozenset(
    {
        "universal_triggers",
        "class_margin_scores",
        "roundtrip_scores",
        "general_val_bpb",
        "samples",
    }
)
_CANDIDATE_SNAPSHOT_TOKEN = object()


@dataclass(frozen=True)
class ClusteredCandidateSnapshot:
    """One exact, validated read of clustered candidate Parquet bytes."""

    table: pa.Table = field(repr=False, compare=False)
    payload: bytes = field(repr=False, compare=False)
    sha256: str
    feature_contract_fingerprint: str
    source_contract_sha256: str
    _token: object = field(repr=False, compare=False, default=None)
    _seal: tuple[int, int, str, str, str] | None = field(
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._token is not _CANDIDATE_SNAPSHOT_TOKEN:
            raise TypeError(
                "ClusteredCandidateSnapshot must be built by "
                "a clustered candidate reader or writer"
            )
        expected_seal = (
            id(self.table),
            id(self.payload),
            self.sha256,
            self.feature_contract_fingerprint,
            self.source_contract_sha256,
        )
        if self._seal != expected_seal:
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot fields changed after validation"
            )
        if type(self.payload) is not bytes or not is_sha256_hex(self.sha256):
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot has malformed exact-byte identity"
            )
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot payload does not match its digest"
            )
        if (
            not is_sha256_hex(self.feature_contract_fingerprint)
            or not is_sha256_hex(self.source_contract_sha256)
        ):
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot has malformed contract coordinates"
            )
        try:
            parsed = pq.read_table(pa.BufferReader(self.payload))
        except Exception as exc:
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot payload is not readable Parquet"
            ) from exc
        if (
            not parsed.schema.equals(CANDIDATES_SCHEMA)
            or not parsed.equals(self.table, check_metadata=True)
        ):
            raise AxBenchBpbAdvError(
                "clustered candidate snapshot table does not match its exact bytes"
            )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_file_sha256(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise AxBenchBpbAdvError(
            f"clustered artifact identity requires {label} at {path}"
        )
    try:
        return stream_hash_file_stable(path).sha256
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"clustered artifact identity could not stably hash {label} at {path}"
        ) from exc


def _candidate_bag_set_sha256(
    table: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
) -> str:
    """Hash every bag artifact that can affect a clustered candidate forward.

    Only a real ``CANDIDATES_SCHEMA`` table receives this dependency.  The
    manifest is built from logical coordinates plus exact stable file hashes;
    absolute paths are deliberately excluded.  Repeated candidate rows that
    share one fair bag (or one roofline bag) collapse to one manifest entry.
    The per-family scored-prefix commit marker is included as well, so a torn or
    independently edited derive cannot be legitimized by an unchanged root
    feature-contract marker.

    A table that is not exactly ``CANDIDATES_SCHEMA`` RAISES. It previously
    returned ``None``, which made the whole bag-set dependency vanish from the
    identity stamp with no diagnostic -- and because the omission recurred
    identically on write AND read, it could never later surface as a staleness
    error. The sole caller is already inside the ``artifact_family ==
    "candidates"`` branch, so a foreign schema there is an anomaly, not an
    expected-domain case, and four sibling sites already treat the identical
    condition as fatal (``scoring.py:107``, ``roundtrip_scoring.py:130``,
    ``general_val.py:1331``, ``scripts/eval/axbench_bpb_adv_eval.py:489``).

    The return type is therefore ``str``, not ``str | None``: the schema
    mismatch was the ONLY ``None`` path, and an empty bag set still hashes to a
    real digest. Nothing is left for an ``Optional`` to mean.

    Accepted residual (documented, deliberately not deviated from the oracle):
    ``SteeringMode(str(row["mode"]))`` below raises a bare ``ValueError`` rather
    than an ``AxBenchBpbAdvError`` for a malformed mode. It still fails LOUD --
    only the exception family is inconsistent with this module's convention --
    and ``CANDIDATES_SCHEMA`` constrains ``mode`` to a string while every real
    writer emits ``SteeringMode.value``. This matches the owner's standing
    adjudication of the same exception-family wart on D-contract: pre-existing,
    not worsened, no action.

    G1-D2 accepted residual: these path-bound reads form a sequential
    multi-file fence, not an atomic snapshot across every bag and marker.
    Production consumes the same paths through ``read_ours_gcg_bag`` and
    ``read_ours_gcg_scored_prefixes``; those load-time readers are the
    authoritative backstop. Snapshot/temp-copy machinery would exceed
    production parity without closing the irreducible inter-file swap window.
    """

    if not table.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError(
            "clustered candidate identity requires a table with CANDIDATES_SCHEMA"
        )

    # Key: (concept_id, mode, family, reg_lambda, objective_alpha, prefix_id).
    bags: dict[tuple[int, str, str, str, str, str | None], dict[str, Any]] = {}
    scored_prefixes: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in table.to_pylist():
        method_name = str(row["method_name"])
        family = precomputed_bag_method_family(method_name)
        if family is None:
            continue
        concept_id = int(row["concept_id"])
        mode = SteeringMode(str(row["mode"]))
        # The alpha methods share the fair GCG family but their coefficient is
        # the class-margin alpha, so it must be routed to the alpha axis here
        # too: read as a reg_lambda it would resolve to a DIFFERENT bag file and
        # hash that one into the identity stamp.
        is_alpha_axis = method_name in ALPHA_AXIS_GCG_METHODS
        objective_alpha = DEFAULT_OBJECTIVE_ALPHA
        if family in {
            PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        }:
            reg_lambda = 0.0
        else:
            raw_coefficient = row["coefficient_str"]
            try:
                reg_lambda = float(raw_coefficient)
            except (TypeError, ValueError) as exc:
                raise AxBenchBpbAdvError(
                    "clustered candidate has a precomputed GCG bag but its "
                    f"coefficient_str is not numeric: method={method_name!r}, "
                    f"value={raw_coefficient!r}"
                ) from exc
            # G1-D4 (named bug fix). float() happily accepts "nan", "inf" and
            # "Infinity", so the numeric guard above did not enforce
            # finiteness. A non-finite coefficient would be normalized into a
            # path slug and hashed into the identity stamp; every NaN spelling
            # also collapses to one repr, so distinct coefficients could share
            # one manifest entry. In practice the bag-file existence check
            # below catches it, but that is a backstop, not a guarantee.
            if not math.isfinite(reg_lambda):
                raise AxBenchBpbAdvError(
                    "clustered candidate has a precomputed GCG bag but its "
                    f"coefficient_str is not finite: method={method_name!r}, "
                    f"value={raw_coefficient!r}"
                )
            if is_alpha_axis:
                # Same scalar, other axis: alpha methods pin reg_lambda to the
                # single value their derive swept (see the OursGcgConfig guard).
                reg_lambda, objective_alpha = 0.0, reg_lambda
        prefix_id = (
            str(row["core_prefix_id"])
            if family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES
            else None
        )
        # repr(float(...)) is a canonical, round-trippable logical coordinate;
        # AxBenchBpbAdvPaths applies its own stable path formatting separately.
        lambda_identity = repr(float(reg_lambda))
        # alpha is part of the KEY, not just the payload: two alpha coefficients
        # share concept/mode/family/reg_lambda, so without it they would collapse
        # to one manifest entry and only the first bag's bytes would be hashed.
        alpha_identity = repr(float(objective_alpha))
        bag_key = (
            concept_id,
            mode.value,
            family.value,
            lambda_identity,
            alpha_identity,
            prefix_id,
        )
        if bag_key not in bags:
            bag_path = paths.ours_gcg_bag(
                concept_id,
                mode,
                reg_lambda,
                family,
                prefix_id=prefix_id,
                objective_alpha=objective_alpha,
            )
            bags[bag_key] = {
                "concept_id": concept_id,
                "mode": mode.value,
                "family": family.value,
                "reg_lambda": lambda_identity,
                # STRIP-DEFAULT, same rule as the bag's own identity payload and
                # for a harder reason: this dict is hashed WHOLESALE into the
                # candidates identity stamp, so emitting the key unconditionally
                # would change the stamp of every already-scored run and fail
                # them at the next stage boundary. Only a real alpha sweep adds
                # it.
                **(
                    {}
                    if float(objective_alpha) == DEFAULT_OBJECTIVE_ALPHA
                    else {"objective_alpha": alpha_identity}
                ),
                "prefix_id": prefix_id,
                "sha256": _stable_file_sha256(
                    bag_path,
                    label="precomputed conditioning bag",
                ),
            }

        scored_key = (concept_id, mode.value, family.value)
        if scored_key not in scored_prefixes:
            scored_path = paths.ours_gcg_scored_prefixes(
                concept_id,
                mode,
                family,
            )
            scored_prefixes[scored_key] = {
                "concept_id": concept_id,
                "mode": mode.value,
                "family": family.value,
                "sha256": _stable_file_sha256(
                    scored_path,
                    label="precomputed-bag scored-prefix marker",
                ),
            }

    manifest = {
        "schema_version": 1,
        "bags": [bags[key] for key in sorted(bags)],
        "scored_prefixes": [
            scored_prefixes[key] for key in sorted(scored_prefixes)
        ],
    }
    return _canonical_sha256(manifest)


def _read_source_contract_identity(
    paths: AxBenchBpbAdvPaths,
    *,
    feature_contract_fingerprint: str,
) -> str:
    path = paths.source_contract()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            f"clustered artifact identity requires a readable source contract at {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBpbAdvError(
            f"clustered source contract must be a JSON object: {path}"
        )
    payload = cast(dict[str, Any], payload)
    parsed = BpbAdvSourceContract.from_json_dict(payload, path=path)
    if parsed.schema_version != CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION:
        raise AxBenchBpbAdvError(
            "clustered artifact identity requires a schema-v7 source contract"
        )
    if parsed.feature_contract_fingerprint != feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            "clustered source contract feature fingerprint does not match the "
            "validated target contract"
        )
    return contract_hash(payload)


_SOURCE_IDENTITY_TOKEN = object()


@dataclass(frozen=True)
class ClusteredSourceIdentity:
    """One validated read of ``source_contract.json``, bound to its contract.

    ``_read_source_contract_identity`` re-reads the file on every call, so a stage
    that stamps several artifacts assembles their lineage from several independent
    reads. Each read validates itself, but nothing checks that two reads saw the
    *same* contract — an out-of-band replacement between them yields artifacts whose
    recorded source hashes disagree, and that disagreement is not detected downstream
    (issue #936).

    Resolving once and threading this object binds every stamp in a stage to a single
    read. It carries the fingerprint it was resolved for so it cannot be paired with a
    different contract, and only :func:`resolve_clustered_source_identity` can build it
    — a caller cannot hand-assemble one that skips validation.
    """

    source_contract_sha256: str
    feature_contract_fingerprint: str
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _SOURCE_IDENTITY_TOKEN:
            raise TypeError(
                "ClusteredSourceIdentity must be built by "
                "resolve_clustered_source_identity()"
            )


def resolve_clustered_source_identity(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> ClusteredSourceIdentity:
    """Read and validate ``source_contract.json`` exactly once for ``contract``."""

    return ClusteredSourceIdentity(
        _read_source_contract_identity(
            paths,
            feature_contract_fingerprint=contract.feature_contract_fingerprint,
        ),
        contract.feature_contract_fingerprint,
        _SOURCE_IDENTITY_TOKEN,
    )


def _resolved_source_hash(
    *,
    paths: AxBenchBpbAdvPaths,
    fingerprint: str,
    source_identity: ClusteredSourceIdentity | None,
) -> str:
    """Use a pre-resolved identity when given, else read once for this call.

    ``None`` is a real alternate behaviour, not a degraded one: a single-artifact
    validation (for example the scoring stage's lone candidates check) has no second
    read to agree with, so resolving internally is exactly right there. It is only a
    *stage* that stamps several artifacts which must resolve once and thread.
    """

    if source_identity is None:
        return _read_source_contract_identity(
            paths, feature_contract_fingerprint=fingerprint
        )
    if source_identity.feature_contract_fingerprint != fingerprint:
        raise AxBenchBpbAdvError(
            "clustered source identity was resolved for feature contract "
            f"{source_identity.feature_contract_fingerprint}, but is being used with "
            f"{fingerprint}"
        )
    return source_identity.source_contract_sha256


def _expected_metadata(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    artifact_family: str,
    table: pa.Table | None = None,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> dict[bytes, bytes]:
    fingerprint = contract.feature_contract_fingerprint
    resolved_source_identity = (
        resolve_clustered_source_identity(paths=paths, contract=contract)
        if source_identity is None
        else source_identity
    )
    source_hash = _resolved_source_hash(
        paths=paths,
        fingerprint=fingerprint,
        source_identity=resolved_source_identity,
    )
    expected = {
        FEATURE_CONTRACT_METADATA_KEY: fingerprint.encode("ascii"),
        SOURCE_CONTRACT_METADATA_KEY: source_hash.encode("ascii"),
        ARTIFACT_FAMILY_METADATA_KEY: artifact_family.encode("utf-8"),
    }
    if artifact_family == "candidates":
        if table is None:
            raise AxBenchBpbAdvError(
                "clustered candidate identity requires the candidate table"
            )
        # Unconditional: _candidate_bag_set_sha256 now raises rather than
        # returning None, so the bag-set key can no longer silently go missing
        # from a candidates-family stamp.
        expected[PRECOMPUTED_BAG_SET_METADATA_KEY] = _candidate_bag_set_sha256(
            table,
            paths=paths,
        ).encode("ascii")
    if candidate_snapshot is not None and artifact_family not in (
        _CANDIDATE_DEPENDENT_FAMILIES
    ):
        raise AxBenchBpbAdvError(
            "a candidate snapshot is valid only for "
            f"candidate-dependent artifacts, got {artifact_family!r}"
        )
    if artifact_family in _CANDIDATE_DEPENDENT_FAMILIES:
        resolved_snapshot = require_clustered_candidate_snapshot(
            candidate_snapshot,
            paths=paths,
            contract=contract,
            source_identity=resolved_source_identity,
            surface=f"{artifact_family} identity",
        )
        assert resolved_snapshot is not None
        expected[UPSTREAM_CANDIDATES_METADATA_KEY] = resolved_snapshot.sha256.encode("ascii")
    return expected


def _assert_expected_metadata(
    table: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    artifact_family: str,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> None:
    metadata = dict(table.schema.metadata or {})
    expected = _expected_metadata(
        paths=paths,
        contract=contract,
        artifact_family=artifact_family,
        table=table,
        source_identity=source_identity,
        candidate_snapshot=candidate_snapshot,
    )
    actual = {
        key: metadata[key]
        for key in _RESERVED_KEYS
        if key in metadata
    }
    if actual != expected:
        raise AxBenchBpbAdvError(
            f"clustered {artifact_family} identity is absent or stale; use a fresh run_tag"
        )


def load_clustered_candidate_snapshot(
    path: Path,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    source_identity: ClusteredSourceIdentity | None = None,
) -> ClusteredCandidateSnapshot:
    """Read, parse, and identity-check candidate bytes exactly once."""

    resolved_source_identity = (
        resolve_clustered_source_identity(paths=paths, contract=contract)
        if source_identity is None
        else source_identity
    )
    if path != paths.candidates_path():
        raise AxBenchBpbAdvError(
            "clustered candidate snapshot path does not match the run's "
            f"canonical candidates path: {path} != {paths.candidates_path()}"
        )
    try:
        digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
    except (ImmutableArtifactError, OSError) as exc:
        raise AxBenchBpbAdvError(
            f"clustered candidate snapshot could not stably read {path}"
        ) from exc
    if digest.payload is None:
        raise AxBenchBpbAdvError(
            "clustered candidate snapshot lost its retained payload"
        )
    try:
        table = pq.read_table(pa.BufferReader(digest.payload))
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"clustered candidate snapshot cannot parse candidates at {path}"
        ) from exc
    if not table.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError(
            "clustered candidate snapshot requires CANDIDATES_SCHEMA"
        )
    _assert_expected_metadata(
        table,
        paths=paths,
        contract=contract,
        artifact_family="candidates",
        source_identity=resolved_source_identity,
    )
    return _build_clustered_candidate_snapshot(
        table=table,
        payload=digest.payload,
        contract=contract,
        paths=paths,
        source_identity=resolved_source_identity,
        sha256=digest.sha256,
    )


def _build_clustered_candidate_snapshot(
    *,
    table: pa.Table,
    payload: bytes,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    source_identity: ClusteredSourceIdentity,
    sha256: str | None = None,
) -> ClusteredCandidateSnapshot:
    """Seal an already validated candidate table to its exact Parquet bytes."""

    try:
        parsed_table = pq.read_table(pa.BufferReader(payload))
    except Exception as exc:  # pragma: no cover - callers already parsed/rendered
        raise AxBenchBpbAdvError(
            "clustered candidate snapshot payload is not readable Parquet"
        ) from exc
    if (
        not parsed_table.schema.equals(CANDIDATES_SCHEMA)
        or not parsed_table.equals(table)
        or parsed_table.schema.metadata != table.schema.metadata
    ):
        raise AxBenchBpbAdvError(
            "clustered candidate snapshot payload changed its validated table"
        )
    resolved_sha256 = (
        hashlib.sha256(payload).hexdigest() if sha256 is None else sha256
    )
    source_hash = _resolved_source_hash(
        paths=paths,
        fingerprint=contract.feature_contract_fingerprint,
        source_identity=source_identity,
    )
    seal = (
        id(parsed_table),
        id(payload),
        resolved_sha256,
        contract.feature_contract_fingerprint,
        source_hash,
    )
    return ClusteredCandidateSnapshot(
        table=parsed_table,
        payload=payload,
        sha256=resolved_sha256,
        feature_contract_fingerprint=contract.feature_contract_fingerprint,
        source_contract_sha256=source_hash,
        _token=_CANDIDATE_SNAPSHOT_TOKEN,
        _seal=seal,
    )


def require_clustered_candidate_snapshot(
    snapshot: ClusteredCandidateSnapshot | None,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract | None,
    source_identity: ClusteredSourceIdentity | None,
    surface: str,
    table: pa.Table | None = None,
) -> ClusteredCandidateSnapshot | None:
    """Require the exact candidate owner at every clustered consumer boundary."""

    if contract is None:
        if snapshot is not None:
            raise AxBenchBpbAdvError(
                f"unclustered {surface} rejects a clustered candidate snapshot"
            )
        return None
    if type(snapshot) is not ClusteredCandidateSnapshot:
        raise AxBenchBpbAdvError(
            f"clustered {surface} requires one exact candidate snapshot"
        )
    snapshot.__post_init__()
    if snapshot.feature_contract_fingerprint != contract.feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            f"clustered {surface} candidate snapshot belongs to a different "
            "feature contract"
        )
    expected_source_hash = _resolved_source_hash(
        paths=paths,
        fingerprint=contract.feature_contract_fingerprint,
        source_identity=source_identity,
    )
    if snapshot.source_contract_sha256 != expected_source_hash:
        raise AxBenchBpbAdvError(
            f"clustered {surface} candidate snapshot belongs to a different "
            "source contract"
        )
    if table is not None and table is not snapshot.table:
        raise AxBenchBpbAdvError(
            f"clustered {surface} must consume the exact candidate table owned "
            "by its snapshot"
        )
    return snapshot


def bind_clustered_table(
    table: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract | None,
    artifact_family: str,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> pa.Table:
    """Return ``table`` unchanged when unclustered, otherwise stamp identity."""

    if contract is None:
        if candidate_snapshot is not None:
            raise AxBenchBpbAdvError(
                "unclustered artifact binding rejects a clustered candidate snapshot"
            )
        return table
    metadata = dict(table.schema.metadata or {})
    if _RESERVED_KEYS.intersection(metadata):
        raise AxBenchBpbAdvError(
            f"refusing to overwrite reserved clustered identity on {artifact_family}"
        )
    metadata.update(
        _expected_metadata(
            paths=paths,
            contract=contract,
            artifact_family=artifact_family,
            table=table,
            source_identity=source_identity,
            candidate_snapshot=candidate_snapshot,
        )
    )
    return table.replace_schema_metadata(metadata)


def restamp_candidates_bag_set(
    table: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
) -> pa.Table:
    """Replace the row-dependent bag-set stamp for a derived candidate subset.

    The three content-independent reserved stamps (feature contract, source
    contract, artifact family) are preserved byte-for-byte and validated to be
    present; the bag-set stamp is recomputed from THIS table's rows against
    ``paths`` with the same primitive the snapshot loader uses.
    """

    metadata = dict(table.schema.metadata or {})
    required = frozenset(
        {
            FEATURE_CONTRACT_METADATA_KEY,
            SOURCE_CONTRACT_METADATA_KEY,
            ARTIFACT_FAMILY_METADATA_KEY,
            PRECOMPUTED_BAG_SET_METADATA_KEY,
        }
    )
    missing = sorted(key.decode("ascii") for key in required - metadata.keys())
    if missing:
        raise AxBenchBpbAdvError(
            "clustered candidates bag-set restamp requires reserved metadata keys: "
            f"missing={missing}"
        )
    metadata[PRECOMPUTED_BAG_SET_METADATA_KEY] = _candidate_bag_set_sha256(
        table,
        paths=paths,
    ).encode("ascii")
    return table.replace_schema_metadata(metadata)


def assert_table_cluster_identity(
    table: pa.Table,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract | None,
    artifact_family: str,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> None:
    """Validate clustered ownership bidirectionally before consuming a table."""

    metadata = dict(table.schema.metadata or {})
    present = _RESERVED_KEYS.intersection(metadata)
    if contract is None:
        if candidate_snapshot is not None:
            raise AxBenchBpbAdvError(
                "unclustered artifact reader rejects a clustered candidate snapshot"
            )
        if present:
            raise AxBenchBpbAdvError(
                f"unclustered reader rejects cluster-stamped {artifact_family}"
            )
        return
    _assert_expected_metadata(
        table,
        paths=paths,
        contract=contract,
        artifact_family=artifact_family,
        source_identity=source_identity,
        candidate_snapshot=candidate_snapshot,
    )


def write_clustered_table_atomic(
    table: pa.Table,
    path: Path,
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    artifact_family: str,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> ClusteredCandidateSnapshot | None:
    """Commit a clustered parquet once; exact replay is a no-op.

    Clustered artifacts are immutable evidence under one ``run_tag``.  Render
    with the same ``pyarrow.parquet.write_table`` defaults as the legacy atomic
    file writer, compare a pre-existing file byte-for-byte, and refuse any
    mutation before replacement.  Unclustered call sites deliberately retain
    their historical writers and never enter this helper.

    FU-A closed the replay residual by descriptor-binding the comparison.
    Its local-unreachability result is superseded: parent swaps touch no target inode/timestamps.
    """

    resolved_source_identity = source_identity
    if artifact_family == "candidates" and resolved_source_identity is None:
        resolved_source_identity = resolve_clustered_source_identity(
            paths=paths,
            contract=contract,
        )
    assert_table_cluster_identity(
        table,
        paths=paths,
        contract=contract,
        artifact_family=artifact_family,
        source_identity=resolved_source_identity,
        candidate_snapshot=candidate_snapshot,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    rendered = sink.getvalue().to_pybytes()
    try:
        install_bytes_once(
            rendered,
            path,
            surface=f"clustered {artifact_family}",
        )
    except ImmutableArtifactError as exc:
        raise AxBenchBpbAdvError(
            f"clustered {artifact_family} content changed on resume; use a "
            f"fresh run_tag instead of overwriting {path}: {exc}"
        ) from exc
    if artifact_family != "candidates":
        return None
    assert resolved_source_identity is not None
    return _build_clustered_candidate_snapshot(
        table=table,
        payload=rendered,
        paths=paths,
        contract=contract,
        source_identity=resolved_source_identity,
    )


def clustered_artifact_identity(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    artifact_family: str,
    source_identity: ClusteredSourceIdentity | None = None,
    candidate_snapshot: ClusteredCandidateSnapshot | None = None,
) -> dict[str, str]:
    """Return the JSON/YAML form of the same identity used for Parquet."""

    return {
        key.decode("ascii"): value.decode("utf-8")
        for key, value in _expected_metadata(
            paths=paths,
            contract=contract,
            artifact_family=artifact_family,
            table=None,
            source_identity=source_identity,
            candidate_snapshot=candidate_snapshot,
        ).items()
    }


__all__ = [
    "ARTIFACT_FAMILY_METADATA_KEY",
    "ClusteredCandidateSnapshot",
    "FEATURE_CONTRACT_METADATA_KEY",
    "PRECOMPUTED_BAG_SET_METADATA_KEY",
    "SOURCE_CONTRACT_METADATA_KEY",
    "UPSTREAM_CANDIDATES_METADATA_KEY",
    "assert_table_cluster_identity",
    "bind_clustered_table",
    "clustered_artifact_identity",
    "load_clustered_candidate_snapshot",
    "restamp_candidates_bag_set",
    "require_clustered_candidate_snapshot",
    "write_clustered_table_atomic",
]
