"""Stage 0-3 source-run validation for AxBench-BPB-Adv."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from safetensors import SafetensorError
from safetensors.torch import safe_open

from nanochat.tokenizer import get_tokenizer
from steering._common.strict_json import canonical_json_str_fallback
from steering.eval.axbench_base.cluster_artifacts import (
    validate_cluster_candidate_provenance_semantics,
)
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactIdentity,
    DirectionArtifactRequest,
    DirectionTreeValidation,
    require_direction_artifacts,
    validate_direction_tree,
)
from steering.eval.axbench_base.direction_scale import parse_max_activation
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    load_seed_pool,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb.sample_loader import (
    make_prefix_hash,
    synth_rows_from_table,
)
from steering.eval.axbench_bpb.source_paths import (
    SourcePaths,
    validate_stage_0_3_artifacts,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError, SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.cluster_feature_contract import (
    FEATURE_CONTRACT_SCHEMA_VERSION,
    ClusterFeatureContractError,
    load_feature_contract_sidecar,
    stat_signature_now,
    stream_hash_file_stable,
)
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    is_sha256_hex,
    stream_hash_file_descriptor_bound,
)
from steering.safetensors_io import read_safetensors_metadata_from_bytes

# v6/v7 bind every source direction to one canonical training layer. Production
# evaluators accept only these globally distinct current schemas.
SOURCE_CONTRACT_SCHEMA_VERSION = 6
CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION = 7

_SHA256_HEX_LENGTH = 64
_FEATURE_CONTRACT_METADATA_KEY = "feature_contract_fingerprint"
_SOURCE_CORPUS_METADATA_KEY = "source_corpus_sha256"
_BASELINE_TRAINING_CORPUS_METADATA_KEY = "cluster_baseline_training_corpus_sha256"
_RETAINED_SOURCE_INPUTS = frozenset(
    {
        "concepts.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
    }
)
_SOURCE_VALIDATION_TOKEN = object()

type ArtifactMetadata = Mapping[str, str] | Mapping[bytes, bytes]


@dataclass(frozen=True)
class BpbAdvSourceContract:
    seed_fingerprint: str
    schema_version: int = SOURCE_CONTRACT_SCHEMA_VERSION
    source_eval: str = "axbench_base_stage_0_3"
    bpb_only: bool = True
    source_run_dir: str | None = None
    seed_schema_version: int = SEED_SCHEMA_VERSION
    seed_token_semantics: str = SEED_TOKEN_SEMANTICS
    # Bytes for EVERY artifact. Files: stat().st_size. Directories ("baselines/"):
    # recursive sum of every non-directory child's st_size.
    artifact_byte_sizes: dict[str, int] = field(default_factory=dict)
    # Recursive entry count (nested dirs + files), DIRECTORY artifacts only. A plain
    # file has no meaningful "entry count", so files are intentionally absent here.
    artifact_entry_counts: dict[str, int] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    n_dropped: int = 0
    # ``None`` is reserved for a source whose baselines tree contains no
    # DiffMean/linear-probe directions.
    direction_training_layer: int | None = None
    # Cluster-only schema-v7 fields. ``to_json_dict`` removes them entirely for
    # schema v6.
    feature_contract_schema_version: int | None = None
    feature_contract_fingerprint: str | None = None
    candidate_provenance_sha256: str | None = None
    stage2_feature_contract_fingerprints: dict[str, str] = field(default_factory=dict)
    baseline_training_corpus_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (
            SOURCE_CONTRACT_SCHEMA_VERSION,
            CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
        ):
            raise AxBenchBpbAdvError(
                "production BPB-Adv source contracts require schema v6 "
                "(unclustered) or v7 (clustered); use a fresh run tag"
            )
        if type(self.source_eval) is not str or self.source_eval != "axbench_base_stage_0_3":
            raise AxBenchBpbAdvError(
                "BPB-Adv source contract source_eval must be 'axbench_base_stage_0_3'"
            )
        if type(self.bpb_only) is not bool or not self.bpb_only:
            raise AxBenchBpbAdvError("BPB-Adv source contract bpb_only must be true")
        if self.source_run_dir is not None and (
            type(self.source_run_dir) is not str or not self.source_run_dir
        ):
            raise AxBenchBpbAdvError(
                "BPB-Adv source contract source_run_dir must be a non-empty string or null"
            )
        if type(self.seed_schema_version) is not int or (
            self.seed_schema_version != SEED_SCHEMA_VERSION
        ):
            raise AxBenchBpbAdvError("BPB-Adv source contracts require AxBench seed schema v2")
        if type(self.seed_token_semantics) is not str or (
            self.seed_token_semantics != SEED_TOKEN_SEMANTICS
        ):
            raise AxBenchBpbAdvError("BPB-Adv source contract seed token semantics mismatch")
        if (
            type(self.seed_fingerprint) is not str
            or (len(self.seed_fingerprint) != _SHA256_HEX_LENGTH)
            or any(char not in "0123456789abcdef" for char in self.seed_fingerprint)
        ):
            raise AxBenchBpbAdvError("BPB-Adv source contract seed fingerprint must be SHA-256")
        if type(self.n_dropped) is not int or self.n_dropped < 0:
            raise AxBenchBpbAdvError("BPB-Adv source contract n_dropped must be a non-negative int")
        if self.direction_training_layer is not None and (
            type(self.direction_training_layer) is not int or self.direction_training_layer < 0
        ):
            raise AxBenchBpbAdvError(
                "BPB-Adv source contract direction_training_layer must be a "
                "non-negative int or null"
            )
        for name, mapping in (
            ("artifact_byte_sizes", self.artifact_byte_sizes),
            ("artifact_entry_counts", self.artifact_entry_counts),
        ):
            if not isinstance(mapping, dict) or any(
                type(key) is not str or type(value) is not int or value < 0
                for key, value in mapping.items()
            ):
                raise AxBenchBpbAdvError(
                    f"BPB-Adv source contract {name} has malformed strict field types"
                )
        if not isinstance(self.artifact_hashes, dict) or any(
            type(key) is not str or type(value) is not str or not is_sha256_hex(value)
            for key, value in self.artifact_hashes.items()
        ):
            raise AxBenchBpbAdvError(
                "BPB-Adv source contract artifact_hashes has malformed strict field types"
            )
        if self.schema_version == SOURCE_CONTRACT_SCHEMA_VERSION:
            if (
                self.feature_contract_schema_version is not None
                or self.feature_contract_fingerprint is not None
                or self.candidate_provenance_sha256 is not None
                or not isinstance(self.stage2_feature_contract_fingerprints, dict)
                or self.stage2_feature_contract_fingerprints
                or not isinstance(self.baseline_training_corpus_sha256, dict)
                or self.baseline_training_corpus_sha256
            ):
                raise AxBenchBpbAdvError(
                    "unclustered BPB-Adv source contract schema v6 cannot carry "
                    "cluster ownership fields"
                )
            return
        if (
            type(self.feature_contract_schema_version) is not int
            or self.feature_contract_schema_version != FEATURE_CONTRACT_SCHEMA_VERSION
        ):
            raise AxBenchBpbAdvError(
                "clustered BPB-Adv source contract has the wrong feature-contract schema version"
            )
        for name, value in (
            ("feature_contract_fingerprint", self.feature_contract_fingerprint),
            ("candidate_provenance_sha256", self.candidate_provenance_sha256),
        ):
            if not is_sha256_hex(value):
                raise AxBenchBpbAdvError(
                    f"clustered BPB-Adv source contract {name} must be SHA-256"
                )
        expected_stage2 = {
            "concept_features.safetensors",
            "concept_features_neg.safetensors",
        }
        if not isinstance(self.stage2_feature_contract_fingerprints, dict) or (
            set(self.stage2_feature_contract_fingerprints) != expected_stage2
        ):
            raise AxBenchBpbAdvError(
                "clustered BPB-Adv source contract must bind both Stage 2 "
                "feature-contract fingerprints"
            )
        if any(
            type(value) is not str or not is_sha256_hex(value)
            for value in self.stage2_feature_contract_fingerprints.values()
        ) or any(
            value != self.feature_contract_fingerprint
            for value in self.stage2_feature_contract_fingerprints.values()
        ):
            raise AxBenchBpbAdvError(
                "clustered BPB-Adv source Stage 2 fingerprints disagree with "
                "the root feature contract"
            )
        if not isinstance(self.baseline_training_corpus_sha256, dict) or (
            set(self.baseline_training_corpus_sha256) != {"positive", "negative"}
        ):
            raise AxBenchBpbAdvError(
                "clustered BPB-Adv source contract must bind the positive and "
                "negative baseline-training corpora"
            )
        for mode, digest in self.baseline_training_corpus_sha256.items():
            if not is_sha256_hex(digest):
                raise AxBenchBpbAdvError(
                    "clustered BPB-Adv source contract baseline-training corpus "
                    f"digest for {mode!r} must be SHA-256"
                )

    def to_json_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        if self.schema_version == SOURCE_CONTRACT_SCHEMA_VERSION:
            for key in (
                "feature_contract_schema_version",
                "feature_contract_fingerprint",
                "candidate_provenance_sha256",
                "stage2_feature_contract_fingerprints",
                "baseline_training_corpus_sha256",
            ):
                payload.pop(key)
        payload["contract_hash"] = contract_hash(payload)
        return payload

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        path: Path | None = None,
    ) -> BpbAdvSourceContract:
        """Strictly parse a production v6/v7 source-contract document."""

        location = "" if path is None else f" at {path}"
        if not isinstance(payload, dict):
            raise AxBenchBpbAdvError(f"BPB-Adv source contract{location} must be a JSON object")
        schema_version = payload.get("schema_version")
        if type(schema_version) is int and schema_version in {4, 5}:
            raise AxBenchBpbAdvError(
                f"legacy BPB-Adv source contract schema v{schema_version}{location} "
                "is not accepted by production evaluators; use a fresh run tag"
            )
        cluster_fields = {
            "feature_contract_schema_version",
            "feature_contract_fingerprint",
            "candidate_provenance_sha256",
            "stage2_feature_contract_fingerprints",
            "baseline_training_corpus_sha256",
        }
        common_fields = {
            "seed_fingerprint",
            "schema_version",
            "source_eval",
            "bpb_only",
            "source_run_dir",
            "seed_schema_version",
            "seed_token_semantics",
            "artifact_byte_sizes",
            "artifact_entry_counts",
            "artifact_hashes",
            "n_dropped",
            "direction_training_layer",
            "contract_hash",
        }
        expected_fields = (
            common_fields
            if schema_version == SOURCE_CONTRACT_SCHEMA_VERSION
            else common_fields | cluster_fields
            if schema_version == CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION
            else common_fields
        )
        actual_fields: set[str] = set(cast(Mapping[str, Any], payload).keys())
        if actual_fields != expected_fields:
            raise AxBenchBpbAdvError(
                f"BPB-Adv source contract{location} field set mismatch: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        recorded_hash = payload.get("contract_hash")
        computed_hash = contract_hash(dict(payload))
        if type(recorded_hash) is not str or recorded_hash != computed_hash:
            raise AxBenchBpbAdvError(
                f"BPB-Adv source contract{location} contract_hash mismatch: "
                f"recorded={recorded_hash!r}, computed={computed_hash}"
            )
        kwargs = dict(payload)
        kwargs.pop("contract_hash")
        try:
            parsed = cls(**kwargs)
        except TypeError as exc:
            raise AxBenchBpbAdvError(
                f"BPB-Adv source contract{location} has invalid field types: {exc}"
            ) from exc
        if parsed.to_json_dict() != dict(payload):
            raise AxBenchBpbAdvError(
                f"BPB-Adv source contract{location} is not canonical v6/v7 JSON"
            )
        return parsed


def _baseline_payload_contract_metrics(
    *,
    child_kinds: tuple[tuple[str, str], ...],
    child_payloads: tuple[tuple[str, bytes], ...],
) -> tuple[int, int, str]:
    """Recompute the persisted ``baselines/`` contract from retained bytes."""

    if (
        tuple(sorted(child_kinds, key=lambda item: Path(item[0]))) != child_kinds
        or len({relative for relative, _kind in child_kinds}) != len(child_kinds)
        or any(kind not in {"dir", "file"} for _relative, kind in child_kinds)
    ):
        raise AxBenchBpbAdvError(
            "clustered source validation has a malformed baseline child manifest"
        )
    if len({relative for relative, _payload in child_payloads}) != len(child_payloads) or any(
        type(payload) is not bytes for _relative, payload in child_payloads
    ):
        raise AxBenchBpbAdvError(
            "clustered source validation has malformed baseline retained payloads"
        )
    payload_by_relative = dict(child_payloads)
    expected_files = {relative for relative, kind in child_kinds if kind == "file"}
    if set(payload_by_relative) != expected_files:
        raise AxBenchBpbAdvError(
            "clustered source validation baseline manifest and retained payloads disagree"
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, kind in child_kinds:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0" + kind.encode("ascii"))
        if kind == "file":
            payload = payload_by_relative[relative]
            total_bytes += len(payload)
            digest.update(str(len(payload)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
    return total_bytes, len(child_kinds), digest.hexdigest()


@dataclass(frozen=True)
class BpbAdvSourceValidation:
    source_run_dir: Path
    contract: BpbAdvSourceContract
    contract_hash: str
    filter_unaccepted_samples: bool
    n_concepts: int
    n_positive_synth_rows: int
    n_negative_synth_rows: int
    n_positive_prefixes: int
    n_negative_prefixes: int
    n_dropped: int
    _positive_table: pa.Table = field(repr=False, compare=False)
    _negative_table: pa.Table = field(repr=False, compare=False)
    _concept_ids: frozenset[int] = field(repr=False, compare=False)
    _concept_entries_json: str = field(repr=False, compare=False)
    _baseline_artifact_payloads: tuple[tuple[str, bytes], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _baseline_child_kinds: tuple[tuple[str, str], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _direction_validation: DirectionTreeValidation = field(
        default_factory=lambda: DirectionTreeValidation((), None),
        repr=False,
        compare=False,
    )
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _SOURCE_VALIDATION_TOKEN:
            raise TypeError(
                "BpbAdvSourceValidation must be built by validate_stage_0_3_source_run()"
            )
        if self.contract_hash != contract_hash(self.contract.to_json_dict()):
            raise AxBenchBpbAdvError("source validation contract hash is internally inconsistent")
        byte_size, entry_count, digest = _baseline_payload_contract_metrics(
            child_kinds=self._baseline_child_kinds,
            child_payloads=self._baseline_artifact_payloads,
        )
        if (
            self.contract.artifact_byte_sizes.get("baselines/") != byte_size
            or self.contract.artifact_entry_counts.get("baselines/") != entry_count
            or self.contract.artifact_hashes.get("baselines/") != digest
        ):
            raise AxBenchBpbAdvError(
                "clustered source validation retained baseline payloads do not "
                "match the committed source contract"
            )
        if (
            self.contract.direction_training_layer
            != self._direction_validation.direction_training_layer
        ):
            raise AxBenchBpbAdvError(
                "source validation direction layer does not match the committed contract"
            )

    @property
    def concept_ids(self) -> frozenset[int]:
        return self._concept_ids

    def concept_entries(self) -> list[Any]:
        """Return a fresh copy of the concepts parsed during validation."""

        entries = json.loads(self._concept_entries_json)
        if not isinstance(entries, list):
            raise AxBenchBpbAdvError("validated concepts payload is internally malformed")
        return entries

    def direction_identities(self) -> tuple[DirectionArtifactIdentity, ...]:
        """Canonical direction identities derived from the retained tree."""

        return tuple(artifact.identity for artifact in self._direction_validation.artifacts)

    def positive_rows(self) -> list[dict[str, Any]]:
        """Rows derived from the positive table loaded during validation."""

        return _filter_to_source_concepts(
            synth_rows_from_table(
                self._positive_table,
                filter_unaccepted=self.filter_unaccepted_samples,
            ),
            concept_ids=set(self._concept_ids),
        )

    def negative_rows(self) -> list[dict[str, Any]]:
        """Rows derived from the negative table loaded during validation."""

        return _filter_to_source_concepts(
            synth_rows_from_table(
                self._negative_table,
                filter_unaccepted=self.filter_unaccepted_samples,
            ),
            concept_ids=set(self._concept_ids),
        )

    def context_prefixes(self) -> dict[str, str]:
        """Prefix population from the exact synth tables loaded for validation.

        The final filtered concept artifact is authoritative here too: rows
        belonging to concepts excluded by Stage 1c cannot enter the context
        cache merely because the positive parquet was committed earlier.
        """

        return _context_prefixes_from_tables(
            positive_table=self._positive_table,
            negative_table=self._negative_table,
            filter_unaccepted=self.filter_unaccepted_samples,
            concept_ids=set(self._concept_ids),
        )

    def direction_artifact_bytes(
        self,
        *,
        family: str,
        target: str,
        mode: str,
        concept_id: int,
    ) -> bytes:
        """Return one exact direction payload certified by source validation."""

        if family not in {"diffmean", "linear_probe"}:
            raise AxBenchBpbAdvError(
                f"direction family must be diffmean or linear_probe, got {family!r}"
            )
        if target not in {"decoder_only", "encdec"}:
            raise AxBenchBpbAdvError(
                f"direction target must be decoder_only or encdec, got {target!r}"
            )
        if mode not in {"positive", "negative"}:
            raise AxBenchBpbAdvError(
                f"clustered direction mode must be positive or negative, got {mode!r}"
            )
        if type(concept_id) is not int or concept_id not in self._concept_ids:
            raise AxBenchBpbAdvError(
                f"direction concept_id must be a validated source concept, got {concept_id!r}"
            )
        relative = f"{family}/{target}/{mode}/concept_{concept_id:04d}.safetensors"
        try:
            return dict(self._baseline_artifact_payloads)[relative]
        except KeyError as exc:
            raise AxBenchBpbAdvError(
                f"source validation is missing the certified direction payload baselines/{relative}"
            ) from exc

    def require_directions(
        self,
        requirements: list[DirectionArtifactRequest],
        *,
        surface: str,
    ) -> None:
        """Prove a runtime's exact selected coordinates exist at its layer."""

        require_direction_artifacts(
            self._direction_validation,
            requirements,
            surface=surface,
            error_type=AxBenchBpbAdvError,
        )


def _context_prefixes_from_tables(
    *,
    positive_table: pa.Table,
    negative_table: pa.Table,
    filter_unaccepted: bool,
    concept_ids: set[int] | None = None,
) -> dict[str, str]:
    rows = synth_rows_from_table(
        positive_table,
        filter_unaccepted=filter_unaccepted,
    )
    rows.extend(
        synth_rows_from_table(
            negative_table,
            filter_unaccepted=filter_unaccepted,
        )
    )
    if concept_ids is not None:
        rows = _filter_to_source_concepts(rows, concept_ids=concept_ids)
    result: dict[str, str] = {}
    for row in rows:
        prefix_text = row["prefix_text"]
        if not isinstance(prefix_text, str):
            raise AxBenchBpbAdvError("source synth prefix_text values must be strings")
        result.setdefault(prefix_text, make_prefix_hash(prefix_text))
    return result


@dataclass(frozen=True)
class _SourceValidationSplit:
    table_name: str
    mode: SteeringMode
    table: Any
    filtered_rows: list[dict[str, Any]]
    all_prefixes: frozenset[tuple[int, str]]
    incomplete_prefixes: frozenset[tuple[int, str]]

    @property
    def n_complete_prefixes(self) -> int:
        return len(self.all_prefixes - self.incomplete_prefixes)


@dataclass(frozen=True)
class _LoadedStage03Source:
    root: Path
    contract_source_run_dir: Path
    paths: SourcePaths
    concepts: list[Any]
    concept_ids: set[int]
    positive: _SourceValidationSplit
    negative: _SourceValidationSplit
    seed_fingerprint: str
    cluster_identity: _ClusteredSourceIdentity | None
    clustered_artifact_snapshot: dict[str, _ArtifactSnapshot] | None
    filter_unaccepted_samples: bool


@dataclass(frozen=True)
class _ClusteredSourceIdentity:
    schema_version: int
    feature_contract_fingerprint: str
    candidate_provenance_sha256: str
    stage2_feature_contract_fingerprints: dict[str, str]
    baseline_training_corpus_sha256: dict[str, str]
    representative_universe: frozenset[int]
    direction_validation: DirectionTreeValidation = field(repr=False, compare=False)


@dataclass(frozen=True)
class _ArtifactSnapshot:
    """Stable content/stat identity for one required clustered artifact."""

    byte_size: int
    entry_count: int | None
    sha256: str
    contract_sha256: str
    stat_signatures: tuple[tuple[str, tuple[int, int, int, int, int, int]], ...]
    payload: bytes | None = field(default=None, repr=False, compare=False)
    child_payloads: tuple[tuple[str, bytes], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    child_kinds: tuple[tuple[str, str], ...] = field(default=())


def contract_hash(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("contract_hash", None)
    return hashlib.sha256(canonical_json_str_fallback(normalized).encode("utf-8")).hexdigest()


def write_source_contract(path: Path, contract: BpbAdvSourceContract) -> Path:
    rendered = (
        json.dumps(
            contract.to_json_dict(),
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    )
    if path.exists():
        try:
            existing_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing_payload = None
        if isinstance(existing_payload, dict) and existing_payload.get("schema_version") in {4, 5}:
            raise AxBenchBpbAdvError(
                f"legacy BPB-Adv source contract at {path} cannot be overwritten; use a fresh run tag"
            )
    try:
        install_bytes_once(
            rendered.encode("utf-8"),
            path,
            surface="BPB-Adv source contract",
        )
    except ImmutableArtifactError as exc:
        raise AxBenchBpbAdvError(
            "BPB-Adv source contract changed on resume; source drift requires "
            f"migration or a fresh run_tag instead of overwriting {path}: {exc}"
        ) from exc
    return path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_value(
    metadata: ArtifactMetadata | None,
    key: str,
) -> str | None:
    if metadata is None:
        return None
    untyped = cast(Mapping[Any, Any], metadata)
    raw = untyped.get(key)
    if raw is None:
        raw = untyped.get(key.encode("utf-8"))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AxBenchBpbAdvError(f"cluster artifact metadata {key!r} is not UTF-8") from exc
    return str(raw)


def _safetensors_metadata(path: Path, *, required: bool) -> dict[str, str] | None:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except (OSError, SafetensorError) as exc:
        if not required:
            return None
        raise AxBenchBpbAdvError(
            f"clustered source artifact is not a valid safetensors file: {path}: {exc}"
        ) from exc


def _require_artifact_fingerprint(
    *,
    surface: str,
    metadata: ArtifactMetadata | None,
    expected: str,
) -> str:
    actual = _metadata_value(metadata, _FEATURE_CONTRACT_METADATA_KEY)
    if actual != expected:
        raise AxBenchBpbAdvError(
            f"{surface} feature-contract fingerprint mismatch: "
            f"expected={expected}, actual={actual!r}; use a fresh source run"
        )
    # G2-D7: an explicit raise, not a bare `assert`. Under `python -O`
    # assertions are stripped, so the oracle's `assert actual is not None` let a
    # function typed `-> str` and named "require" silently return None whenever a
    # caller passed `expected=None` -- exactly the case the equality check above
    # cannot distinguish. Current production callers supply a string, but this
    # fail-loud guard must not depend on how Python was invoked.
    if actual is None:
        raise AxBenchBpbAdvError(
            f"{surface} feature-contract fingerprint is absent; a clustered "
            "artifact must carry it and the expected value must be a SHA-256"
        )
    return actual


def _representative_universe_from_document(
    document: Mapping[str, Any],
) -> frozenset[int]:
    semantic = document.get("semantic")
    clusters = semantic.get("feature_clusters") if isinstance(semantic, Mapping) else None
    raw = clusters.get("representative_universe") if isinstance(clusters, Mapping) else None
    if (
        not isinstance(raw, list)
        or any(type(value) is not int or value < 0 for value in raw)
        or raw != sorted(set(cast(list[int], raw)))
        or not raw
    ):
        raise AxBenchBpbAdvError(
            "clustered source feature contract has a malformed representative universe"
        )
    return frozenset(int(value) for value in raw)


def _representative_members_from_document(
    document: Mapping[str, Any],
) -> dict[int, tuple[int, ...]]:
    # G2-D8: defensive traversal, matching the sibling
    # _representative_universe_from_document. The oracle used bare subscripts
    # here, so a document missing "semantic"/"feature_clusters" -- or carrying a
    # non-mapping at either level -- leaked a raw KeyError/TypeError instead of
    # this module's AxBenchBpbAdvError. The sibling already guarded the
    # identical nested path; the two now agree.
    semantic = document.get("semantic")
    clusters = semantic.get("feature_clusters") if isinstance(semantic, Mapping) else None
    raw = (
        clusters.get("representative_to_allowed_members") if isinstance(clusters, Mapping) else None
    )
    if not isinstance(raw, list):
        raise AxBenchBpbAdvError(
            "clustered source feature contract is missing representative members"
        )
    # G2-D6: the oracle validated member rows only for strict-int shape,
    # ordering, uniqueness and representative-first. It never bounded members by
    # the SAE width, never required the mapping's keys to BE the representative
    # universe, and accepted an empty mapping outright -- so a self-consistently
    # stamped document like [[1, [1, 999]], [5, [5]]] with universe {1, 3} and
    # n_sae_features=8 was admissible: 999 is outside the SAE domain, 5 is not a
    # representative, and 3 is missing entirely. Downstream candidate-provenance
    # catches missing/extra representatives but its own ID helper checks only
    # non-negativity, not the width bound, so the out-of-domain member survives.
    # Guarded like the traversal above: unreachable with a non-Mapping semantic
    # (raw would be None and we would already have raised), but ty cannot narrow
    # through two ternaries, and an unguarded .get() here would be the only
    # un-narrowed access left in the function.
    n_sae_features = semantic.get("n_sae_features") if isinstance(semantic, Mapping) else None
    if type(n_sae_features) is not int or n_sae_features <= 0:
        raise AxBenchBpbAdvError("clustered source feature contract has a malformed n_sae_features")
    parsed: dict[int, tuple[int, ...]] = {}
    for row in raw:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not int
            or not isinstance(row[1], list)
            or any(type(value) is not int for value in row[1])
        ):
            raise AxBenchBpbAdvError(
                "clustered source feature contract has malformed representative members"
            )
        representative = row[0]
        members = tuple(row[1])
        if (
            representative in parsed
            or not members
            or members != tuple(sorted(set(members)))
            or members[0] != representative
            or any(value < 0 or value >= n_sae_features for value in members)
        ):
            raise AxBenchBpbAdvError(
                "clustered source feature contract has inconsistent representative members"
            )
        parsed[representative] = members
    if not parsed:
        raise AxBenchBpbAdvError(
            "clustered source feature contract has an empty representative-member mapping"
        )
    universe = _representative_universe_from_document(document)
    if set(parsed) != universe:
        raise AxBenchBpbAdvError(
            "clustered source feature contract representative-member keys do not "
            f"match the representative universe: members={sorted(parsed)}, "
            f"universe={sorted(universe)}"
        )
    return parsed


def _require_representative_concepts(
    *,
    rows: list[Any],
    representative_universe: frozenset[int],
    surface: str,
) -> None:
    offenders: list[object] = []
    for row in rows:
        feature_id = row.get("feature_index") if isinstance(row, Mapping) else None
        if type(feature_id) is not int or feature_id not in representative_universe:
            offenders.append(feature_id)
    if offenders:
        raise AxBenchBpbAdvError(
            f"{surface} contains non-representative feature IDs: {offenders[:20]}"
        )


def _load_and_validate_candidate_provenance(
    *,
    paths: SourcePaths,
    feature_document: Mapping[str, Any],
    representative_universe: frozenset[int],
) -> str:
    """Load, hash, and validate the two candidate artifacts from exact bytes."""

    candidate_path = paths.candidate_concepts()
    provenance_path = paths.candidate_cluster_provenance()
    try:
        # G2-D4: each mutable path is read exactly once. The bytes parsed below
        # are therefore the same bytes bound by the candidate/provenance hashes.
        candidate_bytes = candidate_path.read_bytes()
        provenance_bytes = provenance_path.read_bytes()
        candidate_rows = json.loads(candidate_bytes)
        provenance = json.loads(provenance_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            f"clustered source candidate artifacts are unreadable: {exc}"
        ) from exc
    if not isinstance(candidate_rows, list):
        raise AxBenchBpbAdvError("clustered candidate_concepts.json must be a list")
    _require_representative_concepts(
        rows=candidate_rows,
        representative_universe=representative_universe,
        surface="candidate_concepts.json",
    )
    expected_fields = {
        "schema_version",
        "feature_contract_fingerprint",
        "candidate_concepts_sha256",
        "concept_projection_version",
        "counts",
        "representatives",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_fields:
        raise AxBenchBpbAdvError("clustered candidate provenance has malformed or partial fields")
    fingerprint = feature_document["feature_contract_fingerprint"]
    algorithms = feature_document["semantic"].get("algorithms")
    if (
        (type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1)
        or provenance["feature_contract_fingerprint"] != fingerprint
        or not isinstance(algorithms, Mapping)
        or provenance["concept_projection_version"] != algorithms.get("concept_projection")
    ):
        raise AxBenchBpbAdvError(
            "clustered candidate provenance disagrees with the root feature contract"
        )
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    if provenance["candidate_concepts_sha256"] != candidate_sha256:
        raise AxBenchBpbAdvError(
            "clustered candidate provenance does not match candidate JSON bytes"
        )
    try:
        validate_cluster_candidate_provenance_semantics(
            feature_contract=feature_document,
            candidate_rows=candidate_rows,
            provenance=provenance,
        )
    except AxBenchBaseError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc
    return hashlib.sha256(provenance_bytes).hexdigest()


def _require_cluster_parquet_metadata(
    path: Path,
    *,
    expected_fingerprint: str,
    surface: str,
) -> dict[bytes, bytes]:
    try:
        metadata = dict(pq.read_schema(path).metadata or {})
    except (OSError, ValueError) as exc:
        raise AxBenchBpbAdvError(f"could not read {surface} metadata: {exc}") from exc
    _require_artifact_fingerprint(
        surface=surface,
        metadata=metadata,
        expected=expected_fingerprint,
    )
    return metadata


def _validate_stage2_artifact(
    path: Path,
    *,
    expected_fingerprint: str,
    representative_universe: frozenset[int],
    expected_concept_features: Mapping[int, int],
) -> str:
    if not expected_concept_features:
        raise AxBenchBpbAdvError(
            "clustered Stage 2 validation requires a non-empty final concept population"
        )
    expected_rows = max(expected_concept_features) + 1
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            _require_artifact_fingerprint(
                surface=path.name,
                metadata=metadata,
                expected=expected_fingerprint,
            )
            tensors: dict[str, t.Tensor] = {}
            for tensor_name in ("feature_ids", "canonical_feature_id"):
                if tensor_name not in handle.keys():
                    raise AxBenchBpbAdvError(
                        f"clustered Stage 2 artifact {path} lacks {tensor_name!r}"
                    )
                values = handle.get_tensor(tensor_name)
                tensors[tensor_name] = values
                if values.dtype not in {
                    t.int8,
                    t.int16,
                    t.int32,
                    t.int64,
                    t.uint8,
                }:
                    raise AxBenchBpbAdvError(
                        f"clustered Stage 2 tensor {tensor_name!r} must be integral"
                    )
                # G2-D9: the oracle filtered `int(value) >= 0`, which dropped
                # every negative id OUT of the offender set rather than
                # rejecting it -- so `feature_ids=[-2]` or
                # `canonical_feature_id=[-999]` validated cleanly here while
                # the downstream consumer rejects them. Source validation must
                # not certify an artifact its own consumer refuses. Issue
                # #1057 adds one legal canonical negative: `-1` on a tensor row
                # whose original concept ID was removed from the final Stage 1
                # population. The row-structural checks below prove the
                # sentinel occurs only on such gaps.
                allowed_sentinels = {-1}
                offenders = sorted(
                    {
                        int(value)
                        for value in values.reshape(-1).tolist()
                        if int(value) not in allowed_sentinels
                        and int(value) not in representative_universe
                    }
                )
                if offenders:
                    raise AxBenchBpbAdvError(
                        f"clustered Stage 2 artifact {path} contains "
                        f"non-representative IDs: {offenders[:20]}"
                    )
            feature_ids = tensors["feature_ids"]
            canonical_feature_ids = tensors["canonical_feature_id"]
            if (
                feature_ids.ndim != 2
                or tuple(feature_ids.shape[:1]) != (expected_rows,)
                or feature_ids.shape[1] < 1
            ):
                raise AxBenchBpbAdvError(
                    f"clustered Stage 2 artifact {path} has feature_ids shape "
                    f"{tuple(feature_ids.shape)}, expected "
                    f"({expected_rows}, n_rank_store>=1)"
                )
            if canonical_feature_ids.ndim != 1 or tuple(canonical_feature_ids.shape) != (
                expected_rows,
            ):
                raise AxBenchBpbAdvError(
                    f"clustered Stage 2 artifact {path} has "
                    f"canonical_feature_id shape "
                    f"{tuple(canonical_feature_ids.shape)}, expected "
                    f"({expected_rows},)"
                )
            for concept_id in range(expected_rows):
                expected_feature = expected_concept_features.get(concept_id)
                actual_canonical = int(canonical_feature_ids[concept_id].item())
                if expected_feature is None:
                    if actual_canonical != -1 or bool((feature_ids[concept_id] != -1).any()):
                        raise AxBenchBpbAdvError(
                            f"clustered Stage 2 artifact {path} row "
                            f"concept_id={concept_id} is absent from the final "
                            "concept population but is not all-sentinel"
                        )
                elif actual_canonical != expected_feature:
                    raise AxBenchBpbAdvError(
                        f"clustered Stage 2 artifact {path} row "
                        f"concept_id={concept_id} has canonical_feature_id="
                        f"{actual_canonical}, expected {expected_feature}"
                    )
                elif int(feature_ids[concept_id, 0].item()) != expected_feature or bool(
                    (feature_ids[concept_id, 1:] != -1).any()
                ):
                    raise AxBenchBpbAdvError(
                        f"clustered Stage 2 artifact {path} row "
                        f"concept_id={concept_id} is not the exact canonical "
                        f"feature row [{expected_feature}, -1, ...]"
                    )
    except AxBenchBpbAdvError:
        raise
    except (OSError, SafetensorError) as exc:
        raise AxBenchBpbAdvError(f"clustered Stage 2 artifact is invalid: {path}: {exc}") from exc
    return expected_fingerprint


def _validate_clustered_baselines(
    *,
    paths: SourcePaths,
    concepts: list[Any],
    expected_fingerprint: str,
    positive_table: Any,
    negative_table: Any,
    source_file_sha256: Mapping[str, str] | None = None,
    clustered_snapshot: Mapping[str, _ArtifactSnapshot] | None = None,
) -> tuple[dict[str, str], DirectionTreeValidation]:
    root = paths.baselines_root()
    baseline_snapshot = None if clustered_snapshot is None else clustered_snapshot.get("baselines/")
    payload_by_relative = (
        {} if baseline_snapshot is None else dict(baseline_snapshot.child_payloads)
    )
    corpus_hashes = {
        "positive": _baseline_training_corpus_sha256(
            table=positive_table,
            path=paths.synth_training_positive(),
            source_file_sha256=(
                None if source_file_sha256 is None else source_file_sha256["positive"]
            ),
        ),
        "negative": _baseline_training_corpus_sha256(
            table=negative_table,
            path=paths.synth_training_negative(),
            source_file_sha256=(
                None if source_file_sha256 is None else source_file_sha256["negative"]
            ),
        ),
    }
    direction_validation = validate_direction_tree(
        root,
        retained_payloads=(None if baseline_snapshot is None else payload_by_relative),
        retained_child_kinds=(None if baseline_snapshot is None else baseline_snapshot.child_kinds),
        error_type=AxBenchBpbAdvError,
    )
    source_concept_ids = {int(concept["concept_id"]) for concept in concepts}
    max_activation_offenders: list[str] = []
    for loaded_artifact in direction_validation.artifacts:
        path = loaded_artifact.path
        relative = path.relative_to(root).as_posix()
        metadata = loaded_artifact.metadata_dict()
        _require_artifact_fingerprint(
            surface=relative,
            metadata=metadata,
            expected=expected_fingerprint,
        )
        identity = loaded_artifact.identity
        if (
            identity.concept_id not in source_concept_ids
            or metadata.get(_SOURCE_CORPUS_METADATA_KEY) != corpus_hashes[identity.mode]
        ):
            raise AxBenchBpbAdvError(
                "clustered direction/probe metadata does not match its final "
                f"source concept and exact synth corpus for mode "
                f"{identity.mode!r}: {path}"
            )
        try:
            max_activation = parse_max_activation(
                metadata,
                surface=str(path),
                error_cls=AxBenchBpbAdvError,
            )
            if max_activation < 0.0:
                raise AxBenchBpbAdvError(
                    f"{path}: negative max_activation ({max_activation}) is invalid"
                )
            direction_is_zero = bool((loaded_artifact.direction == 0.0).all().item())
            if (max_activation == 0.0) != direction_is_zero:
                raise AxBenchBpbAdvError(
                    f"{path}: zero-direction max_activation mismatch: "
                    f"direction_zero={direction_is_zero}, "
                    f"max_activation={max_activation}"
                )
        except AxBenchBpbAdvError as exc:
            max_activation_offenders.append(str(exc))
    if max_activation_offenders:
        details = "\n".join(
            f"  - {offender}" for offender in max_activation_offenders
        )
        raise AxBenchBpbAdvError(
            "clustered decoder direction/probe max_activation validation failed "
            f"for {len(max_activation_offenders)} artifact(s):\n{details}"
        )
    return corpus_hashes, direction_validation


def _baseline_training_corpus_sha256(
    *,
    table: Any,
    path: Path,
    source_file_sha256: str | None = None,
) -> str:
    """Resolve the exact corpus bytes used to train clustered scalar baselines.

    A source cleaner may remove rows needed only for paired BPB scoring after
    Stage 3 has already trained the scalar baselines. In that case the cleaned
    parquet carries the immutable hash of its ancestor training corpus. The
    copied baseline remains bound to those real training bytes; it is never
    relabelled as if it had been retrained on the filtered table.
    """

    recorded = _metadata_value(
        table.schema.metadata,
        _BASELINE_TRAINING_CORPUS_METADATA_KEY,
    )
    if recorded is None:
        if source_file_sha256 is not None:
            if not is_sha256_hex(source_file_sha256):
                raise AxBenchBpbAdvError(f"{path.name} has a malformed snapshotted source digest")
            return source_file_sha256
        try:
            return stream_hash_file_stable(path).sha256
        except (ClusterFeatureContractError, OSError) as exc:
            raise AxBenchBpbAdvError(
                f"could not stably hash clustered baseline-training corpus {path}"
            ) from exc
    if not is_sha256_hex(recorded):
        raise AxBenchBpbAdvError(
            f"{path.name} has malformed clustered baseline-training corpus provenance"
        )
    return recorded


def _reject_unclustered_cluster_stamps(
    *,
    paths: SourcePaths,
    positive_table: Any,
    negative_table: Any,
) -> None:
    if paths.candidate_cluster_provenance().exists():
        raise AxBenchBpbAdvError(
            "unclustered BPB-Adv source rejects clustered candidate provenance "
            "without an exact feature-contract root marker"
        )
    metadata_surfaces: list[tuple[str, ArtifactMetadata | None]] = [
        ("synth_training_positive.parquet", positive_table.schema.metadata),
        ("synth_training_negative.parquet", negative_table.schema.metadata),
    ]
    if paths.negative_mode_prefixes().exists():
        metadata_surfaces.append(
            (
                "negative_mode_prefixes.parquet",
                pq.read_schema(paths.negative_mode_prefixes()).metadata,
            )
        )
    for name, path in (
        ("concept_features.safetensors", paths.concept_features()),
        ("concept_features_neg.safetensors", paths.concept_features_neg()),
    ):
        metadata_surfaces.append((name, _safetensors_metadata(path, required=False)))
    for path in paths.baselines_root().rglob("*.safetensors"):
        metadata_surfaces.append(
            (
                str(path.relative_to(paths.source_run_dir)),
                _safetensors_metadata(path, required=False),
            )
        )
    stamped = [
        name
        for name, metadata in metadata_surfaces
        if any(
            _metadata_value(metadata, key) is not None
            for key in (
                _FEATURE_CONTRACT_METADATA_KEY,
                _SOURCE_CORPUS_METADATA_KEY,
                _BASELINE_TRAINING_CORPUS_METADATA_KEY,
            )
        )
    ]
    if stamped:
        raise AxBenchBpbAdvError(
            f"unclustered BPB-Adv source rejects cluster-stamped artifacts: {stamped[:20]}"
        )


def _resolve_clustered_source_identity(
    *,
    paths: SourcePaths,
    concepts: list[Any],
    positive_table: Any,
    negative_table: Any,
    clustered_snapshot: Mapping[str, _ArtifactSnapshot] | None = None,
) -> _ClusteredSourceIdentity | None:
    marker = paths.feature_contract()
    if not marker.exists():
        _reject_unclustered_cluster_stamps(
            paths=paths,
            positive_table=positive_table,
            negative_table=negative_table,
        )
        return None
    try:
        document = load_feature_contract_sidecar(marker)
    except (ClusterFeatureContractError, OSError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid clustered source feature contract {marker}: {exc}"
        ) from exc
    fingerprint = str(document["feature_contract_fingerprint"])
    representative_universe = _representative_universe_from_document(document)
    _require_representative_concepts(
        rows=concepts,
        representative_universe=representative_universe,
        surface="concepts.json",
    )
    # G2-D11: hoisted ABOVE _load_and_validate_candidate_provenance. Sited
    # after it, this check was DEAD -- that call already rejects a non-Mapping
    # `algorithms` as one OR-branch of a combined condition, so this specific
    # diagnostic could never fire. Hoisting keeps the guard and restores its
    # precise message instead of the vaguer "provenance disagrees" one.
    algorithms = document["semantic"].get("algorithms")
    if not isinstance(algorithms, Mapping):
        raise AxBenchBpbAdvError("clustered feature contract lacks algorithm versions")
    candidate_provenance_sha256 = _load_and_validate_candidate_provenance(
        paths=paths,
        feature_document=document,
        representative_universe=representative_universe,
    )
    stage0c_metadata = _require_cluster_parquet_metadata(
        paths.negative_mode_prefixes(),
        expected_fingerprint=fingerprint,
        surface="negative_mode_prefixes.parquet",
    )
    if (
        _metadata_value(
            stage0c_metadata,
            "candidate_cluster_provenance_sha256",
        )
        != candidate_provenance_sha256
    ):
        raise AxBenchBpbAdvError(
            "clustered Stage 0c artifact does not match candidate provenance bytes"
        )
    members = _representative_members_from_document(document)
    expected_target_members = {
        str(int(concept["feature_index"])): list(members[int(concept["feature_index"])])
        for concept in concepts
    }
    expected_concept_features = {
        int(concept["concept_id"]): int(concept["feature_index"]) for concept in concepts
    }
    for table, surface, allow_parent_population_entries in (
        (
            positive_table,
            "synth_training_positive.parquet",
            True,
        ),
        (
            negative_table,
            "synth_training_negative.parquet",
            False,
        ),
    ):
        metadata = dict(table.schema.metadata or {})
        _require_artifact_fingerprint(
            surface=surface,
            metadata=metadata,
            expected=fingerprint,
        )
        raw_target_members = _metadata_value(
            metadata,
            "cluster_target_members_json",
        )
        try:
            actual_target_members = json.loads(raw_target_members or "")
        except json.JSONDecodeError:
            actual_target_members = None
        target_members_valid = isinstance(actual_target_members, dict)
        if target_members_valid:
            actual_mapping = cast(dict[Any, Any], actual_target_members)
            for raw_feature_id, raw_members in actual_mapping.items():
                try:
                    feature_id = int(raw_feature_id)
                except (TypeError, ValueError):
                    target_members_valid = False
                    break
                if (
                    not isinstance(raw_feature_id, str)
                    or raw_feature_id != str(feature_id)
                    or feature_id not in members
                    or raw_members != list(members[feature_id])
                ):
                    target_members_valid = False
                    break
            if target_members_valid and any(
                actual_mapping.get(feature_id) != expected_members
                for feature_id, expected_members in expected_target_members.items()
            ):
                target_members_valid = False
            if (
                target_members_valid
                and not allow_parent_population_entries
                and set(actual_mapping) != set(expected_target_members)
            ):
                target_members_valid = False
        if (
            _metadata_value(metadata, "cluster_qc_target_algorithm")
            != algorithms.get("qc_target_reduction")
            or not target_members_valid
        ):
            raise AxBenchBpbAdvError(f"{surface} has stale clustered QC target semantics")
    stage2_fingerprints = {
        paths.concept_features().name: _validate_stage2_artifact(
            paths.concept_features(),
            expected_fingerprint=fingerprint,
            representative_universe=representative_universe,
            expected_concept_features=expected_concept_features,
        ),
        paths.concept_features_neg().name: _validate_stage2_artifact(
            paths.concept_features_neg(),
            expected_fingerprint=fingerprint,
            representative_universe=representative_universe,
            expected_concept_features=expected_concept_features,
        ),
    }
    (
        baseline_training_corpus_sha256,
        direction_validation,
    ) = _validate_clustered_baselines(
        paths=paths,
        concepts=concepts,
        expected_fingerprint=fingerprint,
        positive_table=positive_table,
        negative_table=negative_table,
        source_file_sha256=(
            None
            if clustered_snapshot is None
            else {
                "positive": clustered_snapshot["synth_training_positive.parquet"].sha256,
                "negative": clustered_snapshot["synth_training_negative.parquet"].sha256,
            }
        ),
        clustered_snapshot=clustered_snapshot,
    )
    return _ClusteredSourceIdentity(
        schema_version=int(document["schema_version"]),
        feature_contract_fingerprint=fingerprint,
        candidate_provenance_sha256=candidate_provenance_sha256,
        stage2_feature_contract_fingerprints=stage2_fingerprints,
        baseline_training_corpus_sha256=baseline_training_corpus_sha256,
        representative_universe=representative_universe,
        direction_validation=direction_validation,
    )


def _hash_length_prefixed(h: Any, value: bytes) -> None:
    h.update(len(value).to_bytes(8, byteorder="big", signed=False))
    h.update(value)


def _hash_safetensors_semantic(path: Path) -> str:
    """Hash safetensors metadata and tensors, independent of header/data order."""

    try:
        with path.open("rb") as handle:
            header_length_bytes = handle.read(8)
            if len(header_length_bytes) != 8:
                raise ValueError("missing 8-byte header length")
            header_length = int.from_bytes(header_length_bytes, byteorder="little")
            header = json.loads(handle.read(header_length))
            data_start = 8 + header_length
            file_size = path.stat().st_size
            if not isinstance(header, dict):
                raise ValueError("header must be a mapping")

            metadata = header.get("__metadata__", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("__metadata__ must be a mapping")

            digest = hashlib.sha256()
            digest.update(b"axbench-safetensors-semantic-v1\0")
            _hash_length_prefixed(
                digest,
                canonical_json_str_fallback(metadata).encode("utf-8"),
            )
            tensor_names = sorted(name for name in header if name != "__metadata__")
            for name in tensor_names:
                descriptor = header[name]
                if not isinstance(descriptor, dict):
                    raise ValueError(f"tensor {name!r} descriptor must be a mapping")
                dtype = descriptor.get("dtype")
                shape = descriptor.get("shape")
                offsets = descriptor.get("data_offsets")
                if (
                    not isinstance(dtype, str)
                    or not isinstance(shape, list)
                    or any(
                        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                        for dim in shape
                    )
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(
                        not isinstance(offset, int) or isinstance(offset, bool)
                        for offset in offsets
                    )
                ):
                    raise ValueError(f"tensor {name!r} descriptor is malformed")
                start, end = offsets
                if start < 0 or end < start or data_start + end > file_size:
                    raise ValueError(f"tensor {name!r} data offsets are out of range")
                _hash_length_prefixed(
                    digest,
                    canonical_json_str_fallback({"name": name, "dtype": dtype, "shape": shape}).encode(
                        "utf-8"
                    ),
                )
                digest.update((end - start).to_bytes(8, byteorder="big", signed=False))
                handle.seek(data_start + start)
                remaining = end - start
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"tensor {name!r} data is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            f"cannot compute semantic safetensors hash for {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _hash_seed_manifest_semantic(path: Path) -> str:
    """Hash seed-manifest semantics while excluding exact transport checksums."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            f"cannot compute semantic seed-manifest hash for {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AxBenchBpbAdvError(
            f"cannot compute semantic seed-manifest hash for {path}: payload must be a mapping"
        )
    files = payload.get("files")
    if not isinstance(files, dict):
        raise AxBenchBpbAdvError(
            f"cannot compute semantic seed-manifest hash for {path}: files must be a mapping"
        )
    for filename, entry in files.items():
        if not isinstance(entry, dict):
            raise AxBenchBpbAdvError(
                f"cannot compute semantic seed-manifest hash for {path}: "
                f"entry {filename!r} must be a mapping"
            )
        entry.pop("sha256", None)
    return hashlib.sha256(
        b"axbench-seed-manifest-semantic-v1\0" + canonical_json_str_fallback(payload).encode("utf-8")
    ).hexdigest()


def _hash_required_file(name: str, path: Path) -> str:
    if name in {"seeds_train.safetensors", "seeds_eval.safetensors"}:
        return _hash_safetensors_semantic(path)
    if name == "seed_pool_manifest.json":
        return _hash_seed_manifest_semantic(path)
    return _hash_file(path)


def _hash_directory(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        h.update(relative.encode("utf-8"))
        if child.is_dir():
            h.update(b"\0dir")
            continue
        h.update(b"\0file")
        h.update(str(child.stat().st_size).encode("utf-8"))
        h.update(b"\0")
        h.update(_hash_file(child).encode("ascii"))
    return h.hexdigest()


def _snapshot_artifact(name: str, path: Path) -> _ArtifactSnapshot:
    """Hash one file/directory and prove its entries stayed fixed throughout."""

    if path.is_symlink():
        raise AxBenchBpbAdvError(f"clustered source artifacts must not be symbolic links: {path}")
    if path.is_file():
        try:
            digest = stream_hash_file_stable(path)
            payload = None
            if name in _RETAINED_SOURCE_INPUTS:
                payload = path.read_bytes()
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                if len(payload) != digest.byte_length or payload_sha256 != digest.sha256:
                    raise AxBenchBpbAdvError(
                        "clustered source artifact changed while retaining its "
                        f"semantic payload after the stable snapshot: {path}"
                    )
                # These three artifacts use their exact byte hash as the
                # persisted source-contract hash. Derive it from the retained
                # payload so hashing and later semantic parsing share bytes.
                contract_sha256 = payload_sha256
            else:
                contract_sha256 = _hash_required_file(name, path)
            # G2-D5: recheck the ORIGINAL pathname, not digest.path.
            # stream_hash_file_stable canonicalizes via resolve(strict=True), so
            # digest.path is the RESOLVED target -- rechecking that would follow
            # a symlink substituted at the declared path after the guard above
            # and never look at the declared path again. stat_signature_now uses
            # lstat, so checking the original surfaces a substituted symlink via
            # its mode field. For an honest regular file both describe the same
            # inode, so this is behaviourally identical on the happy path.
            current_signature = stat_signature_now(path)
        except (ClusterFeatureContractError, OSError) as exc:
            raise AxBenchBpbAdvError(
                f"could not stably snapshot clustered source artifact {path}: {exc}"
            ) from exc
        if current_signature != digest.stat_signature:
            raise AxBenchBpbAdvError(
                f"clustered source artifact changed while it was being hashed: {path}"
            )
        return _ArtifactSnapshot(
            byte_size=digest.byte_length,
            entry_count=None,
            sha256=digest.sha256,
            contract_sha256=contract_sha256,
            stat_signatures=(("", digest.stat_signature),),
            payload=payload,
        )
    if not path.is_dir():
        raise AxBenchBpbAdvError(
            f"clustered source artifact is not a regular file or directory: {path}"
        )

    try:
        root_signature = stat_signature_now(path)
    except OSError as exc:
        raise AxBenchBpbAdvError(
            f"could not stably snapshot clustered source directory {path}: {exc}"
        ) from exc
    # Preserve `_hash_directory`'s historical Path ordering exactly. Lexically
    # sorting the POSIX strings is observably different for a directory `a/`
    # beside entries such as `a-c` (`Path("a/b") < Path("a-c")`, while the
    # strings sort in the opposite order), and would silently change existing
    # source-contract hashes.
    children_before = sorted(path.rglob("*"))
    signatures: list[tuple[str, tuple[int, int, int, int, int, int]]] = [("", root_signature)]
    child_kinds: list[tuple[str, str]] = []
    child_payloads: list[tuple[str, bytes]] = []
    for child in children_before:
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise AxBenchBpbAdvError(
                f"clustered source directory artifacts must not contain symlinks: {child}"
            )
        try:
            initial_signature = stat_signature_now(child)
        except OSError as exc:
            raise AxBenchBpbAdvError(
                f"could not stably snapshot clustered source artifact {child}: {exc}"
            ) from exc
        if child.is_dir():
            child_kinds.append((relative, "dir"))
            signatures.append((relative, initial_signature))
            continue
        if not child.is_file():
            raise AxBenchBpbAdvError(
                "clustered source directory artifacts may contain only regular "
                f"files and directories: {child}"
            )
        child_kinds.append((relative, "file"))
        try:
            digest = stream_hash_file_descriptor_bound(
                child.resolve(strict=True),
                retain_payload=True,
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise AxBenchBpbAdvError(
                f"could not stably snapshot clustered source artifact {child}: {exc}"
            ) from exc
        if digest.stat_signature != initial_signature or digest.payload is None:
            raise AxBenchBpbAdvError(
                f"clustered source artifact changed before it was retained: {child}"
            )
        signatures.append((relative, digest.stat_signature))
        child_payloads.append((relative, digest.payload))

    (
        total_bytes,
        entry_count,
        contract_sha256,
    ) = _baseline_payload_contract_metrics(
        child_kinds=tuple(child_kinds),
        child_payloads=tuple(child_payloads),
    )
    children_after = sorted(path.rglob("*"))
    kinds_after = [
        (
            child.relative_to(path).as_posix(),
            "dir" if child.is_dir() else "file" if child.is_file() else "other",
        )
        for child in children_after
    ]
    if child_kinds != kinds_after:
        raise AxBenchBpbAdvError(
            f"clustered source directory changed while it was being hashed: {path}"
        )
    try:
        changed = [
            relative
            for relative, signature in signatures
            if stat_signature_now(path / relative) != signature
        ]
    except OSError as exc:
        raise AxBenchBpbAdvError(
            f"clustered source directory changed while it was being hashed: {path}"
        ) from exc
    if changed:
        raise AxBenchBpbAdvError(
            "clustered source directory file changed while it was being hashed: "
            f"{path / changed[0]}"
        )
    return _ArtifactSnapshot(
        byte_size=total_bytes,
        entry_count=entry_count,
        sha256=contract_sha256,
        contract_sha256=contract_sha256,
        stat_signatures=tuple(signatures),
        child_payloads=tuple(child_payloads),
        child_kinds=tuple(child_kinds),
    )


def _directory_byte_size_and_entry_count(path: Path) -> tuple[int, int]:
    """Recursive ``(byte_size, entry_count)`` for a directory artifact.

    ``byte_size`` sums ``st_size`` over every non-directory descendant — the SAME
    set of children whose ``st_size`` ``_hash_directory`` folds into the directory
    hash, so the byte total and the hash always agree on what counts as a file.
    ``entry_count`` is the recursive count of all descendants (nested directories
    AND files), identical to the pre-#512 ``sum(1 for _ in path.rglob("*"))``.
    """
    total_bytes = 0
    n_entries = 0
    for child in path.rglob("*"):
        n_entries += 1
        if not child.is_dir():
            total_bytes += child.stat().st_size
    return total_bytes, n_entries


def _required_artifacts(
    paths: SourcePaths,
    *,
    clustered: bool = False,
) -> list[tuple[str, Path]]:
    # ``concepts.json`` is the stable logical source-contract key. Its path is
    # intentionally SourcePaths.concepts_json(), which now resolves to the
    # final ``concepts_stage1c.json`` population. Keeping the logical key avoids
    # an unrelated source-contract schema/hash migration while changing the
    # bytes and semantics consumed by BPB-Adv.
    artifacts = [
        ("concepts.json", paths.concepts_json()),
        ("synth_training_positive.parquet", paths.synth_training_positive()),
        ("synth_training_negative.parquet", paths.synth_training_negative()),
        ("seeds_train.safetensors", paths.seeds_train()),
        ("seeds_eval.safetensors", paths.seeds_eval()),
        ("seed_pool_manifest.json", paths.seed_pool_manifest()),
        ("concept_features.safetensors", paths.concept_features()),
        ("concept_features_neg.safetensors", paths.concept_features_neg()),
        ("baselines/", paths.baselines_root()),
    ]
    if clustered:
        artifacts.extend(
            [
                ("feature_contract.json", paths.feature_contract()),
                ("candidate_concepts.json", paths.candidate_concepts()),
                (
                    "candidate_concepts.cluster_provenance.json",
                    paths.candidate_cluster_provenance(),
                ),
                (
                    "negative_mode_prefixes.parquet",
                    paths.negative_mode_prefixes(),
                ),
            ]
        )
    return artifacts


def _snapshot_clustered_source_artifacts(
    paths: SourcePaths,
) -> dict[str, _ArtifactSnapshot]:
    """Freeze every clustered schema-v7 input as one comparable snapshot."""

    snapshots: dict[str, _ArtifactSnapshot] = {}
    for name, path in _required_artifacts(paths, clustered=True):
        if not path.exists():
            if name == "candidate_concepts.cluster_provenance.json":
                raise AxBenchBpbAdvError(
                    "clustered source candidate artifacts are unreadable: "
                    f"missing candidate provenance commit marker {path}"
                )
            raise AxBenchBpbAdvError(
                f"clustered source is missing required artifact {name}: {path}"
            )
        snapshots[name] = _snapshot_artifact(name, path)
    return snapshots


def _required_snapshot_payload(
    snapshots: Mapping[str, _ArtifactSnapshot],
    name: str,
) -> bytes:
    snapshot = snapshots.get(name)
    if snapshot is None or snapshot.payload is None:
        raise AxBenchBpbAdvError(f"clustered source snapshot is missing retained payload {name}")
    return snapshot.payload


def _validate_seed_v2_contract(paths: SourcePaths) -> str:
    """Require the exact committed seed-v2 pair and return its fingerprint."""

    try:
        tokenizer = get_tokenizer()
        load_seed_pool(paths.seeds_train(), expected_split="train", tokenizer=tokenizer)
        load_seed_pool(paths.seeds_eval(), expected_split="eval", tokenizer=tokenizer)
    except (AxBenchBaseError, OSError, RuntimeError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            "BPB-Adv source requires committed AxBench seed schema v2; "
            "legacy or torn seeds cannot establish BOS ownership. Regenerate "
            f"Stages 0b-3e under a fresh run tag: {exc}"
        ) from exc
    try:
        manifest = json.loads(paths.seed_pool_manifest().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError("AxBench seed_pool_manifest.json is unreadable") from exc
    if manifest.get("schema_version") != SEED_SCHEMA_VERSION:
        raise AxBenchBpbAdvError(
            "BPB-Adv source seed schema is not v2; use a fresh AxBench run tag"
        )
    if manifest.get("token_semantics") != SEED_TOKEN_SEMANTICS:
        raise AxBenchBpbAdvError("BPB-Adv source seed token semantics do not match schema v2")
    fingerprint = manifest.get("seed_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise AxBenchBpbAdvError("BPB-Adv source seed fingerprint is not a SHA-256 digest")
    return fingerprint


_BASE_SYNTH_COLUMNS = frozenset(
    {
        "concept_id",
        "content_class",
        "prefix_text",
        "output_text",
        "qc_accepted",
    }
)
_NEGATIVE_SYNTH_COLUMNS = _BASE_SYNTH_COLUMNS | {
    "prefix_source_shard",
    "prefix_source_doc",
}


def _require_columns(table_name: str, table: Any, required: frozenset[str]) -> None:
    missing = sorted(required - set(table.column_names))
    if missing:
        raise AxBenchBpbAdvError(f"{table_name} is missing required columns: {', '.join(missing)}")


_REQUIRED_CLASSES = ("concept_rich", "concept_avoiding")


def _class_sides_by_prefix(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], set[str]]:
    by_prefix: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        key = (int(row["concept_id"]), str(row["prefix_text"]))
        by_prefix.setdefault(key, set()).add(str(row["content_class"]))
    return by_prefix


def _find_incomplete_class_prefixes(
    rows: list[dict[str, Any]],
) -> set[tuple[int, str]]:
    """Return incomplete ``(concept_id, prefix_text)`` groups.

    This helper exists for the explicit source-cleaning CLI. Runtime validation
    and attack-row preparation call the fail-loud requirement instead.
    """

    by_prefix = _class_sides_by_prefix(rows)
    return {
        (concept_id, prefix_text)
        for concept_id, prefix_text in by_prefix
        if any(cls not in by_prefix[(concept_id, prefix_text)] for cls in _REQUIRED_CLASSES)
    }


def _incomplete_prefix_offenders(
    by_prefix: dict[tuple[int, str], set[str]],
) -> list[str]:
    offenders: list[str] = []
    for (concept_id, prefix_text), classes in sorted(by_prefix.items()):
        missing = [cls for cls in _REQUIRED_CLASSES if cls not in classes]
        if missing:
            offenders.append(
                f"(concept_id={concept_id}, prefix_text={prefix_text!r}) "
                f"missing {', '.join(missing)}"
            )
    return offenders


def _source_concept_ids(concepts: list[Any]) -> set[int]:
    """Concept-id set derived from concepts.json.

    Callers that have a ``concepts.json`` path should use
    :func:`_read_source_concept_ids` so the JSON parsing/type errors are
    reported consistently across validation, preparation, optimization, and the
    cleaner.
    """

    return {int(concept["concept_id"]) for concept in concepts}


def _read_source_concepts(path: Path) -> list[Any]:
    try:
        concepts = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AxBenchBpbAdvError(f"failed to read concepts.json: {exc}") from exc
    if not isinstance(concepts, list):
        raise AxBenchBpbAdvError("concepts.json must contain a list")
    return concepts


def _read_source_concepts_bytes(payload: bytes) -> list[Any]:
    """Parse retained concepts bytes with the path reader's UTF-8 semantics."""

    try:
        concepts = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise AxBenchBpbAdvError(f"failed to read concepts.json: {exc}") from exc
    if not isinstance(concepts, list):
        raise AxBenchBpbAdvError("concepts.json must contain a list")
    return concepts


def _read_source_concept_ids(path: Path) -> set[int]:
    """Read ``concepts.json`` and return its concept-id set."""

    return _source_concept_ids(_read_source_concepts(path))


def _filter_to_source_concepts(
    rows: list[dict[str, Any]],
    *,
    concept_ids: set[int],
) -> list[dict[str, Any]]:
    """Drop rows whose ``concept_id`` is not in ``concept_ids``.

    Same predicate as
    ``scripts.eval.axbench_bpb_adv_eval._filter_synth_rows_to_source_concepts``
    so this validator and the prepare stage count over the identical population.
    """

    return [row for row in rows if int(row["concept_id"]) in concept_ids]


def _source_validation_split(
    *,
    table_name: str,
    path: Path,
    table: Any,
    required_columns: frozenset[str],
    concept_ids: set[int],
    mode: SteeringMode,
    filter_unaccepted_samples: bool,
) -> _SourceValidationSplit:
    if table.num_rows == 0:
        raise AxBenchBpbAdvError(f"{table_name} is empty")
    _require_columns(table_name, table, required_columns)
    filtered_rows = _filter_to_source_concepts(
        synth_rows_from_table(
            table,
            filter_unaccepted=filter_unaccepted_samples,
        ),
        concept_ids=concept_ids,
    )
    all_prefixes = frozenset(
        (int(row["concept_id"]), str(row["prefix_text"])) for row in filtered_rows
    )
    incomplete_prefixes = frozenset(_find_incomplete_class_prefixes(filtered_rows))
    return _SourceValidationSplit(
        table_name=path.name,
        mode=mode,
        table=table,
        filtered_rows=filtered_rows,
        all_prefixes=all_prefixes,
        incomplete_prefixes=incomplete_prefixes,
    )


def _load_stage_0_3_source_for_validation(
    source_run_dir: Path | str,
    *,
    filter_unaccepted_samples: bool = True,
    contract_source_run_dir: Path | str | None = None,
) -> _LoadedStage03Source:
    root = Path(source_run_dir)
    if not root.exists():
        raise AxBenchBpbAdvError(f"source_run_dir does not exist: {root}")
    paths = SourcePaths(root)
    try:
        validate_stage_0_3_artifacts(
            paths,
            filter_unaccepted_samples=filter_unaccepted_samples,
            validate_population=False,
        )
    except AxBenchBpbError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc
    clustered_at_start = paths.feature_contract().is_file()
    clustered_snapshot = _snapshot_clustered_source_artifacts(paths) if clustered_at_start else None
    seed_fingerprint = _validate_seed_v2_contract(paths)
    if clustered_snapshot is None:
        concepts = _read_source_concepts(paths.concepts_json())
        positive_table = pq.read_table(paths.synth_training_positive())
        negative_table = pq.read_table(paths.synth_training_negative())
    else:
        concepts = _read_source_concepts_bytes(
            _required_snapshot_payload(clustered_snapshot, "concepts.json")
        )
        positive_table = pq.read_table(
            pa.BufferReader(
                _required_snapshot_payload(
                    clustered_snapshot,
                    "synth_training_positive.parquet",
                )
            )
        )
        negative_table = pq.read_table(
            pa.BufferReader(
                _required_snapshot_payload(
                    clustered_snapshot,
                    "synth_training_negative.parquet",
                )
            )
        )
    try:
        validate_positive_qc_contract(
            paths.synth_training_positive(),
            consumer="AxBench-BPB-Adv source validation",
            schema=positive_table.schema,
        )
        validate_positive_selection_contract(
            paths.synth_training_positive(),
            consumer="AxBench-BPB-Adv source validation",
            schema=positive_table.schema,
        )
        validate_stage1_continuation_contract(
            paths.synth_training_positive(),
            consumer="AxBench-BPB-Adv source validation",
            expected_role="stage1b_positive",
            schema=positive_table.schema,
        )
        validate_stage1_continuation_contract(
            paths.synth_training_negative(),
            consumer="AxBench-BPB-Adv source validation",
            expected_role="stage1c_negative",
            schema=negative_table.schema,
        )
    except AxBenchBaseError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc
    concept_ids = _source_concept_ids(concepts)
    positive = _source_validation_split(
        table_name="synth_training_positive.parquet",
        path=paths.synth_training_positive(),
        table=positive_table,
        required_columns=_BASE_SYNTH_COLUMNS,
        concept_ids=concept_ids,
        mode=SteeringMode.POSITIVE,
        filter_unaccepted_samples=filter_unaccepted_samples,
    )
    negative = _source_validation_split(
        table_name="synth_training_negative.parquet",
        path=paths.synth_training_negative(),
        table=negative_table,
        required_columns=_NEGATIVE_SYNTH_COLUMNS,
        concept_ids=concept_ids,
        mode=SteeringMode.NEGATIVE,
        filter_unaccepted_samples=filter_unaccepted_samples,
    )
    cluster_identity = _resolve_clustered_source_identity(
        paths=paths,
        concepts=concepts,
        positive_table=positive.table,
        negative_table=negative.table,
        clustered_snapshot=clustered_snapshot,
    )
    if clustered_at_start != (cluster_identity is not None):
        raise AxBenchBpbAdvError("source cluster ownership changed while it was being validated")
    if clustered_snapshot is not None and cluster_identity is not None:
        # G2-D10: bind the SEMANTICALLY-loaded provenance digest to the
        # SNAPSHOT digest for the same artifact.
        #
        # G2-D4 fused the read inside _load_and_validate_candidate_provenance,
        # but that only guarantees the leaf hashed the bytes it parsed. At the
        # FLOW level two digests still came from two independent reads --
        # artifact_hashes from the snapshot, candidate_provenance_sha256 from
        # the semantic load -- and nothing required them to agree. A coherent
        # transient B present only during the semantic reads, with A restored
        # before both snapshots, therefore committed a MIXED contract: B's
        # candidate_provenance_sha256 alongside A's artifact_hashes. The
        # snapshot equality checks cannot see this, because they only prove the
        # filesystem is A again, not that the semantic read also saw A.
        snapshot_provenance = clustered_snapshot.get("candidate_concepts.cluster_provenance.json")
        if snapshot_provenance is None:
            raise AxBenchBpbAdvError(
                "clustered source snapshot is missing the candidate provenance "
                "artifact required to bind the semantic read"
            )
        if snapshot_provenance.contract_sha256 != cluster_identity.candidate_provenance_sha256:
            raise AxBenchBpbAdvError(
                "clustered candidate provenance read during semantic validation "
                "does not match the snapshotted artifact; the source changed "
                "mid-validation. Use a fresh source run: "
                f"snapshot={snapshot_provenance.contract_sha256}, "
                f"semantic={cluster_identity.candidate_provenance_sha256}"
            )
    if clustered_snapshot is not None:
        snapshot_after_semantic_load = _snapshot_clustered_source_artifacts(paths)
        if snapshot_after_semantic_load != clustered_snapshot:
            raise AxBenchBpbAdvError(
                "clustered source artifacts changed while semantic validation was running"
            )
    contract_root = Path(contract_source_run_dir) if contract_source_run_dir is not None else root
    return _LoadedStage03Source(
        root=root,
        contract_source_run_dir=contract_root,
        paths=paths,
        concepts=concepts,
        concept_ids=concept_ids,
        positive=positive,
        negative=negative,
        seed_fingerprint=seed_fingerprint,
        cluster_identity=cluster_identity,
        clustered_artifact_snapshot=clustered_snapshot,
        filter_unaccepted_samples=filter_unaccepted_samples,
    )


def _require_split_has_complete_prefixes(
    split: _SourceValidationSplit,
    *,
    expected_concept_ids: set[int],
) -> int:
    present_concept_ids = {concept_id for concept_id, _prefix_text in split.all_prefixes}
    absent = sorted(expected_concept_ids - present_concept_ids)
    if absent:
        raise AxBenchBpbAdvError(
            f"{split.table_name} ({split.mode.value} mode) has no accepted "
            f"source rows for authoritative final concept IDs {absent}; "
            "concepts_stage1c.json is the population authority and corpus "
            "absence cannot silently filter it"
        )
    if split.incomplete_prefixes:
        by_prefix = _class_sides_by_prefix(split.filtered_rows)
        offenders = _incomplete_prefix_offenders(by_prefix)
        shown = "; ".join(offenders[:10])
        raise AxBenchBpbAdvError(
            f"{split.table_name} ({split.mode.value} mode) has "
            f"{len(split.incomplete_prefixes)} prefix(es) missing a class side; "
            "the class margin requires both concept_rich and concept_avoiding "
            "per (concept_id, prefix_text) (this catches an old/global-pool "
            f"source run): {shown}"
        )
    return len(split.all_prefixes)


def _validate_loaded_stage_0_3_source(
    loaded: _LoadedStage03Source,
) -> BpbAdvSourceValidation:
    n_positive_prefixes = _require_split_has_complete_prefixes(
        loaded.positive,
        expected_concept_ids=loaded.concept_ids,
    )
    n_negative_prefixes = _require_split_has_complete_prefixes(
        loaded.negative,
        expected_concept_ids=loaded.concept_ids,
    )
    n_dropped = 0

    clustered_snapshot: dict[str, _ArtifactSnapshot] | None = None
    if loaded.cluster_identity is not None:
        if loaded.clustered_artifact_snapshot is None:
            raise AxBenchBpbAdvError(
                "clustered source validation lost its initial artifact snapshot"
            )
        clustered_snapshot = _snapshot_clustered_source_artifacts(loaded.paths)
        if clustered_snapshot != loaded.clustered_artifact_snapshot:
            raise AxBenchBpbAdvError(
                "clustered source artifacts changed between semantic validation "
                "and source-contract commit"
            )
    elif loaded.clustered_artifact_snapshot is not None:
        raise AxBenchBpbAdvError(
            "unclustered source validation carried a clustered artifact snapshot"
        )

    # The baselines directory is descriptor-snapshotted in BOTH schemas.  Its
    # contract hash and direction semantics are derived from this one retained
    # traversal, so a path replacement cannot mix one version's layer with
    # another version's hash.
    baseline_snapshot = (
        clustered_snapshot["baselines/"]
        if clustered_snapshot is not None
        else _snapshot_artifact("baselines/", loaded.paths.baselines_root())
    )
    direction_validation = validate_direction_tree(
        loaded.paths.baselines_root(),
        retained_payloads=dict(baseline_snapshot.child_payloads),
        retained_child_kinds=baseline_snapshot.child_kinds,
        error_type=AxBenchBpbAdvError,
    )
    if loaded.cluster_identity is None:
        # Reassert unclustered ownership from the SAME retained payloads that
        # supply both the committed baselines hash and direction semantics.
        # The earlier whole-source check gives operators a useful diagnostic
        # for every artifact kind, but it must not be the only guard: any
        # safetensors baseline can be replaced between that path read and this
        # descriptor-bound snapshot. In particular, source_corpus_sha256 is
        # independently sufficient to identify a clustered artifact.
        stamped_baselines: list[str] = []
        for relative, payload in baseline_snapshot.child_payloads:
            if Path(relative).suffix.lower() != ".safetensors":
                continue
            try:
                metadata = dict(read_safetensors_metadata_from_bytes(payload) or {})
            except ValueError:
                # Direction artifacts are parsed strictly above. Other
                # baseline families retain their existing loader-owned format
                # validation; an unreadable header cannot establish a cluster
                # stamp here.
                continue
            if any(
                key in metadata
                for key in (
                    _FEATURE_CONTRACT_METADATA_KEY,
                    _SOURCE_CORPUS_METADATA_KEY,
                    _BASELINE_TRAINING_CORPUS_METADATA_KEY,
                )
            ):
                stamped_baselines.append(f"baselines/{relative}")
        if stamped_baselines:
            raise AxBenchBpbAdvError(
                "unclustered BPB-Adv source rejects cluster-stamped artifacts "
                f"from its committed baseline snapshot: {stamped_baselines[:20]}"
            )
    if loaded.cluster_identity is not None:
        initial_identities = tuple(
            artifact.identity for artifact in loaded.cluster_identity.direction_validation.artifacts
        )
        committed_identities = tuple(
            artifact.identity for artifact in direction_validation.artifacts
        )
        if (
            initial_identities != committed_identities
            or loaded.cluster_identity.direction_validation.direction_training_layer
            != direction_validation.direction_training_layer
        ):
            raise AxBenchBpbAdvError(
                "clustered source direction identities changed before source-contract commit"
            )

    artifact_byte_sizes: dict[str, int] = {}
    artifact_entry_counts: dict[str, int] = {}
    artifact_hashes: dict[str, str] = {}
    for name, path in _required_artifacts(
        loaded.paths,
        clustered=loaded.cluster_identity is not None,
    ):
        snapshot = (
            baseline_snapshot
            if name == "baselines/"
            else None
            if clustered_snapshot is None
            else clustered_snapshot[name]
        )
        if snapshot is not None:
            artifact_byte_sizes[name] = snapshot.byte_size
            if snapshot.entry_count is not None:
                artifact_entry_counts[name] = snapshot.entry_count
            # G2-D3: this is the semantic/path-order contract digest shared with
            # the unclustered loop, not the raw stable snapshot digest.
            artifact_hashes[name] = snapshot.contract_sha256
            continue
        if not path.exists():
            raise AxBenchBpbAdvError(f"source is missing required artifact {name}: {path}")
        if path.is_dir():
            byte_size, entry_count = _directory_byte_size_and_entry_count(path)
            artifact_byte_sizes[name] = byte_size
            artifact_entry_counts[name] = entry_count
            artifact_hashes[name] = _hash_directory(path)
        else:
            artifact_byte_sizes[name] = path.stat().st_size
            artifact_hashes[name] = _hash_required_file(name, path)

    cluster_kwargs: dict[str, Any] = {}
    schema_version = SOURCE_CONTRACT_SCHEMA_VERSION
    if loaded.cluster_identity is not None:
        schema_version = CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION
        cluster_kwargs = {
            "feature_contract_schema_version": loaded.cluster_identity.schema_version,
            "feature_contract_fingerprint": (loaded.cluster_identity.feature_contract_fingerprint),
            "candidate_provenance_sha256": (loaded.cluster_identity.candidate_provenance_sha256),
            "stage2_feature_contract_fingerprints": dict(
                loaded.cluster_identity.stage2_feature_contract_fingerprints
            ),
            "baseline_training_corpus_sha256": dict(
                loaded.cluster_identity.baseline_training_corpus_sha256
            ),
        }
    contract = BpbAdvSourceContract(
        schema_version=schema_version,
        source_run_dir=str(loaded.contract_source_run_dir),
        seed_schema_version=SEED_SCHEMA_VERSION,
        seed_fingerprint=loaded.seed_fingerprint,
        seed_token_semantics=SEED_TOKEN_SEMANTICS,
        artifact_byte_sizes=artifact_byte_sizes,
        artifact_entry_counts=artifact_entry_counts,
        artifact_hashes=artifact_hashes,
        n_dropped=n_dropped,
        direction_training_layer=direction_validation.direction_training_layer,
        **cluster_kwargs,
    )
    return BpbAdvSourceValidation(
        source_run_dir=loaded.contract_source_run_dir,
        contract=contract,
        contract_hash=contract_hash(contract.to_json_dict()),
        filter_unaccepted_samples=loaded.filter_unaccepted_samples,
        n_concepts=len(loaded.concepts),
        n_positive_synth_rows=loaded.positive.table.num_rows,
        n_negative_synth_rows=loaded.negative.table.num_rows,
        n_positive_prefixes=n_positive_prefixes,
        n_negative_prefixes=n_negative_prefixes,
        n_dropped=n_dropped,
        _positive_table=loaded.positive.table,
        _negative_table=loaded.negative.table,
        _concept_ids=frozenset(loaded.concept_ids),
        _concept_entries_json=json.dumps(
            loaded.concepts,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        _baseline_artifact_payloads=baseline_snapshot.child_payloads,
        _baseline_child_kinds=baseline_snapshot.child_kinds,
        _direction_validation=direction_validation,
        _token=_SOURCE_VALIDATION_TOKEN,
    )


def validate_stage_0_3_source_run(
    source_run_dir: Path | str,
    *,
    filter_unaccepted_samples: bool = True,
) -> BpbAdvSourceValidation:
    """Validate the BPB source contract without Stage 4/5 artifacts.

    Incomplete prefixes are always a hard error: the class-margin metric cannot
    score a prefix unless both ``concept_rich`` and ``concept_avoiding`` sides are
    present. Dirty historical sources must be cleaned explicitly before running
    BPB-Adv.

    ``filter_unaccepted_samples`` MUST match ``cfg.filter_unaccepted_samples``:
    completeness is checked over the SAME FILTERED population the prepare stage
    pairs (drop ``qc_accepted=False`` rows when set, then drop rows whose
    ``concept_id`` is not in ``concepts.json``).
    """

    loaded = _load_stage_0_3_source_for_validation(
        source_run_dir,
        filter_unaccepted_samples=filter_unaccepted_samples,
    )
    return _validate_loaded_stage_0_3_source(loaded)
