"""Crash-safe per-candidate checkpoints for BPB-Adv round-trip scoring.

Resume counters deliberately split durable and current-process work. The
``checkpoint_restored_*`` counters report candidates recovered from shards;
generation/probe request, unique-work, and cache-hit counters describe only the
current process. Restored generation and probe rows seed their semantic caches,
so a fresh twin can count as a current-process cache hit without a model call.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import try_to_load_from_cache
from loguru import logger
from pyarrow import ArrowInvalid, ArrowTypeError
from sae_lens.loading.pretrained_saes_directory import get_repo_id_and_folder_name

from nanochat.common import get_base_dir
from steering.eval.axbench_base.config import bare_model_id
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    get_additive_direction_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.universal_gcg_checkpoint import (
    ResolvedTargetCheckpoint,
    canonical_json_bytes,
    canonical_sha256,
    current_runtime_fingerprint,
    tree_sha256,
)
from steering.eval.roundtrip_steering import RoundtripProbeConfig
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    stream_hash_file_descriptor_bound,
)
from steering.sae_resolver import ActivationType, SAERelease, SAEWidth, resolve_sae
from steering.sae_resolver_v2_data import GEMMASCOPE_2_HF_MODEL_IDS

CHECKPOINT_SCHEMA_VERSION = 1
STORE_ALGORITHM_VERSION = "axbench_bpb_adv_roundtrip_candidate_checkpoint_v3"
STORE_KIND = "axbench_bpb_adv_roundtrip_candidate_checkpoint_v1"
MANIFEST_FILENAME = "manifest.json"
GROUPS_DIRNAME = "groups"


class CheckpointFamily(StrEnum):
    """Artifact families stored by :class:`RoundtripCheckpointStore`."""

    GENERATION = "generation"
    PROBE = "probe"

_PROBE_COLUMNS = frozenset(
    {
        "n_roundtrip_features",
        "roundtrip_pre_act",
        "roundtrip_post_act",
        "roundtrip_percentile",
        "control_n_roundtrip_features",
        "control_roundtrip_pre_act",
        "control_roundtrip_post_act",
        "control_roundtrip_percentile",
    }
)
GENERATION_ROW_SCHEMA: pa.Schema = pa.schema(
    [
        ROUNDTRIP_SCORE_SCHEMA.field(name)
        for name in ROUNDTRIP_SCORE_SCHEMA.names
        if name not in _PROBE_COLUMNS
    ]
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_RE = re.compile(r"^[0-9a-f]{2}$")
_SHARD_RE = re.compile(r"^cand_(?P<key>[0-9a-f]{64})_(?P<payload>[0-9a-f]{64})\.parquet$")
_INSTALLER_TEMP_RE = re.compile(
    r"^\.(?:manifest\.json|cand_[0-9a-f]{64}_[0-9a-f]{64}\.parquet)"
    r"\.tmp\.\d+\.[0-9a-f]{8}$"
)

_META_SCHEMA_VERSION = b"checkpoint_schema_version"
_META_ROOT_FINGERPRINT = b"manifest_fingerprint_sha256"
_META_FAMILY = b"roundtrip_checkpoint_family"
_META_CANDIDATE_ID = b"candidate_id"
_META_RUNTIME_FINGERPRINT = b"runtime_numeric_fingerprint_sha256"
_METADATA_KEYS = {
    _META_SCHEMA_VERSION,
    _META_ROOT_FINGERPRINT,
    _META_FAMILY,
    _META_CANDIDATE_ID,
    _META_RUNTIME_FINGERPRINT,
}

_STATIC_CANDIDATE_COLUMNS = tuple(
    name for name in GENERATION_ROW_SCHEMA.names if name in CANDIDATES_SCHEMA.names
)
_PROBE_METRIC_COLUMNS = tuple(
    name
    for name in ROUNDTRIP_SCORE_SCHEMA.names
    if name in _PROBE_COLUMNS
    and name not in {"n_roundtrip_features", "control_n_roundtrip_features"}
)


def _checkpoint_error(message: str, path: Path) -> AxBenchBpbAdvError:
    return AxBenchBpbAdvError(
        f"{message}; checkpoint={path}. Preserve the old run and use a fresh run_tag."
    )


def _hash_file(path: Path, *, label: str) -> str:
    try:
        return stream_hash_file_descriptor_bound(
            path,
            on_error=lambda reason: _checkpoint_error(
                f"could not stably hash {label} ({reason}) at {path}", path
            ),
        ).sha256
    except (ImmutableArtifactError, OSError) as exc:
        raise _checkpoint_error(f"could not stably hash {label} at {path}: {exc}", path) from exc


def _cached_hf_file(repo_id: str, filename: str, *, label: str) -> Path:
    """Resolve one immutable HF-cache file without permitting a network fetch."""

    try:
        cached = try_to_load_from_cache(repo_id, filename)
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"could not resolve local Hugging Face cache content for {label} "
            f"({repo_id}/{filename}): {exc}"
        ) from exc
    if not isinstance(cached, str):
        raise AxBenchBpbAdvError(
            f"{label} ({repo_id}/{filename}) is not present in the local Hugging Face "
            "cache; checkpoint identity requires the artifact content locally and will "
            "not fetch it from the network"
        )
    path = Path(cached)
    if not path.exists() or not path.is_file():
        raise AxBenchBpbAdvError(
            f"resolved local Hugging Face cache file for {label} does not exist: {path}"
        )
    return path


def _snapshot_identity_from_file(path: Path, *, label: str) -> tuple[Path, str]:
    """Return the containing ``snapshots/<revision>`` directory and revision."""

    for parent in (path.parent, *path.parents):
        if parent.parent.name == "snapshots":
            if not parent.exists() or not parent.is_dir():
                raise AxBenchBpbAdvError(
                    f"resolved Hugging Face snapshot directory for {label} does not exist: {parent}"
                )
            if not parent.name:
                break
            return parent, parent.name
    raise AxBenchBpbAdvError(
        f"resolved Hugging Face cache file for {label} is not under snapshots/<revision>: {path}"
    )


def resolve_extraction_tokenizer_content_identity(tokenizer_id: str) -> dict[str, Any]:
    """Pin the exact local/Hub tokenizer content loaded by the production loader."""

    configured = Path(tokenizer_id)
    if configured.exists():
        if configured.is_symlink() or not configured.is_dir():
            raise AxBenchBpbAdvError(
                f"extraction tokenizer local path must be a real directory: {configured}"
            )
        # Local tokenizer directories are already content-only trees.
        get_extraction_tokenizer(tokenizer_id)
        return {
            "tokenizer_id": str(tokenizer_id),
            "source": "local_tree",
            "content_tree_sha256": tree_sha256(configured),
        }

    tokenizer = get_extraction_tokenizer(tokenizer_id)
    commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    cached = _cached_hf_file(
        tokenizer_id,
        "tokenizer_config.json",
        label="extraction tokenizer",
    )
    snapshot_dir, snapshot_revision = _snapshot_identity_from_file(
        cached, label="extraction tokenizer"
    )
    if commit is not None and str(commit) != snapshot_revision:
        raise AxBenchBpbAdvError(
            "extraction tokenizer resolved commit differs from its local snapshot: "
            f"loader={commit!r}, snapshot={snapshot_revision!r}"
        )
    tokenizer_filenames = {
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        *(
            str(filename)
            for filename in getattr(tokenizer, "vocab_files_names", {}).values()
            if filename is not None
        ),
    }
    descriptors: list[dict[str, str]] = []
    for filename in sorted(tokenizer_filenames):
        candidate = snapshot_dir / str(filename)
        if not candidate.exists() or not candidate.is_file():
            continue
        descriptors.append(
            {
                "path": str(filename),
                "sha256": _hash_file(candidate.resolve(), label=f"extraction tokenizer {filename}"),
            }
        )
    if not descriptors:
        raise AxBenchBpbAdvError(
            f"extraction tokenizer {tokenizer_id!r} resolved snapshot "
            f"{snapshot_revision!r} but no tokenizer files were resolvable"
        )
    return {
        "tokenizer_id": str(tokenizer_id),
        "source": "hf_snapshot",
        "snapshot_revision": snapshot_revision,
        "resolved_tokenizer_files_sha256": canonical_sha256(descriptors),
    }


def resolve_probe_artifact_identity(probe_config: RoundtripProbeConfig) -> dict[str, Any]:
    """Resolve and content-pin the exact base-model snapshot and SAE weight file."""

    resolved_sae = resolve_sae(
        model=bare_model_id(probe_config.model_name),
        release=SAERelease(probe_config.sae_release),
        activation_type=ActivationType(probe_config.activation_type),
        layer=int(probe_config.layer),
        width=SAEWidth(probe_config.sae_width),
        l0=probe_config.l0,
    )
    repo_id, folder_name = get_repo_id_and_folder_name(resolved_sae.release, resolved_sae.sae_id)
    weight_name = (
        "params.safetensors" if probe_config.sae_release == "gemmascope_2" else "params.npz"
    )
    weight_filename = f"{folder_name.rstrip('/')}/{weight_name}"
    sae_weight_path = _cached_hf_file(repo_id, weight_filename, label="roundtrip probe SAE weights")
    sae_snapshot_dir, sae_revision = _snapshot_identity_from_file(
        sae_weight_path, label="roundtrip probe SAE weights"
    )

    # The HF cache is keyed on the ORG-PREFIXED repo id while probe configs may
    # carry the registry's bare id (the same bare-vs-prefixed class as the
    # resolve_sae sites); resolve through the canonical mapping before the
    # cache lookup so a bare id pins the same snapshot the loader consumes.
    base_model_repo_id = GEMMASCOPE_2_HF_MODEL_IDS.get(
        bare_model_id(probe_config.model_name),
        str(probe_config.model_name),
    )
    model_config_path = _cached_hf_file(
        base_model_repo_id,
        "config.json",
        label="roundtrip probe base-model config",
    )
    model_snapshot_dir, model_revision = _snapshot_identity_from_file(
        model_config_path, label="roundtrip probe base model"
    )
    # The immutable snapshot revision pins the base-model weights and tokenizer;
    # only the small config file is re-hashed on every run.
    return {
        "sae": {
            "release": resolved_sae.release,
            "sae_id": resolved_sae.sae_id,
            "repo_id": repo_id,
            "snapshot_revision": sae_revision,
            "snapshot_subdir": str(Path(folder_name)),
            "weight_file": weight_name,
            "weight_sha256": _hash_file(
                sae_weight_path.resolve(), label="roundtrip probe SAE weights"
            ),
        },
        "base_model": {
            "configured_model_name": str(probe_config.model_name),
            "repo_id": base_model_repo_id,
            "snapshot_revision": model_revision,
            "config_sha256": _hash_file(
                model_config_path.resolve(), label="roundtrip probe base-model config"
            ),
        },
        "probe_tokenizer": {
            "covered_by_base_model_snapshot_revision": model_revision,
        },
    }


def candidates_payload_sha256(
    *,
    candidates: pa.Table,
    paths: AxBenchBpbAdvPaths,
    snapshot_payload: bytes | None = None,
    snapshot_sha256: str | None = None,
) -> str:
    """Bind the exact candidates Parquet payload represented by ``candidates``."""

    if (snapshot_payload is None) != (snapshot_sha256 is None):
        raise AxBenchBpbAdvError(
            "roundtrip candidate snapshot payload and sha256 must be supplied together"
        )
    if snapshot_payload is None:
        path = paths.candidates_path()
        try:
            digest = stream_hash_file_descriptor_bound(
                path,
                retain_payload=True,
                on_error=lambda reason: _checkpoint_error(
                    f"candidates.parquet changed while reading ({reason})", path
                ),
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(
                f"could not stably read candidates.parquet: {exc}", path
            ) from exc
        payload = digest.payload
        observed_sha256 = digest.sha256
    else:
        payload = snapshot_payload
        observed_sha256 = cast(str, snapshot_sha256)
        computed = hashlib.sha256(payload).hexdigest()
        if computed != observed_sha256:
            raise _checkpoint_error(
                "clustered candidate snapshot payload digest differs: "
                f"expected {observed_sha256!r}, observed {computed!r}",
                paths.candidates_path(),
            )
    if payload is None:
        raise _checkpoint_error(
            "candidates.parquet read retained no payload", paths.candidates_path()
        )
    try:
        parsed = pq.read_table(pa.BufferReader(payload))
    except (pa.ArrowException, OSError) as exc:
        raise _checkpoint_error(
            f"candidates.parquet payload is not valid Parquet: {exc}",
            paths.candidates_path(),
        ) from exc
    if not parsed.schema.equals(CANDIDATES_SCHEMA) or not parsed.equals(candidates):
        raise _checkpoint_error(
            "candidates.parquet payload does not reconstruct the exact consumed candidate table",
            paths.candidates_path(),
        )
    return observed_sha256


def build_roundtrip_checkpoint_fingerprint(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    candidates_sha256: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    cluster_feature_contract_fingerprint: str | None,
    resolved_targets: Mapping[str, ResolvedTargetCheckpoint],
    reference_segment_size_by_target: Mapping[str, int],
    probe_statistics_path: str | Path,
) -> dict[str, Any]:
    """Build the complete semantic fence for both round-trip shard families."""

    expected_ids = [str(row["candidate_id"]) for row in candidate_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise AxBenchBpbAdvError("expected roundtrip candidate_id values must be unique")
    if _SHA256_RE.fullmatch(candidates_sha256) is None:
        raise AxBenchBpbAdvError(
            f"candidates_sha256 must be a lowercase SHA-256 digest, got {candidates_sha256!r}"
        )
    targets = [
        {
            "target_name": target_name,
            "source": target.source,
            "model_tag": target.model_tag,
            "step": target.step,
            "model_sha256": target.model_sha256,
            "metadata_sha256": target.metadata_sha256,
        }
        for target_name, target in sorted(resolved_targets.items())
    ]
    uses_encdec = any(str(row["target_model"]) == "encdec" for row in candidate_rows)
    uses_bags = any(
        precomputed_bag_method_family(str(row["method_name"])) is not None for row in candidate_rows
    )
    additive_cfg = get_additive_direction_config(cfg)
    counter_direction_family = (
        additive_cfg.counter_direction_family if additive_cfg is not None else "diffmean"
    )
    # Clustered bag bytes are already transitively pinned by the exact candidates
    # payload's candidate_metadata_json. Unclustered bag-backed methods consume the
    # live tree and therefore require its own conditional content fence.
    if not uses_bags:
        bags_tree_sha256 = "unused"
    elif cluster_feature_contract_fingerprint is not None:
        bags_tree_sha256 = "covered_by_clustered_candidate_metadata"
    else:
        bags_tree_sha256 = tree_sha256(paths.ours_gcg_bags_dir())
    # max_new_tokens is derived per candidate from the reference segment sizes and
    # the candidates payload's reference-token digests; both are fenced below.
    generation_semantics = {
        "lora_adapter": dict(cfg.lora_adapter),
        "diffmean_and_probe_layer": int(cfg.diffmean_and_probe_layer),
        "counter_direction_family": str(counter_direction_family),
        "context_cache_tree_sha256": (
            tree_sha256(paths.context_cache_dir()) if uses_encdec else "unused"
        ),
        "unclustered_precomputed_bags_tree_sha256": bags_tree_sha256,
        "feature_statistics_sha256": _hash_file(
            Path(cfg.feature_statistics_path), label="feature_statistics.safetensors"
        ),
        "extraction_tokenizer": resolve_extraction_tokenizer_content_identity(
            cfg.extraction_tokenizer_id
        ),
        "roundtrip_generation": asdict(cfg.roundtrip_generation),
        "reference_segment_size_by_target": {
            str(name): int(size) for name, size in sorted(reference_segment_size_by_target.items())
        },
    }
    fingerprint: dict[str, Any] = {
        "store_algorithm_version": STORE_ALGORITHM_VERSION,
        "candidates_sha256": candidates_sha256,
        "evaluation_population_sha256": _hash_file(
            paths.evaluation_population(), label="evaluation_population.json"
        ),
        "source_contract_sha256": _hash_file(paths.source_contract(), label="source_contract.json"),
        "cluster_feature_contract_fingerprint": cluster_feature_contract_fingerprint,
        "targets": targets,
        "tokenizer_tree_sha256": tree_sha256(Path(get_base_dir()) / "tokenizer"),
        "generation_semantics": generation_semantics,
        "probe": {
            "config": asdict(cfg.roundtrip_steering.probe),
            "artifacts": resolve_probe_artifact_identity(cfg.roundtrip_steering.probe),
            "feature_statistics_sha256": _hash_file(
                Path(probe_statistics_path), label="feature_statistics.safetensors"
            ),
            "feature_chunk_size": int(cfg.roundtrip_steering.feature_chunk_size),
        },
        "expected_candidate_ids": {
            "count": len(expected_ids),
            "sha256": canonical_sha256(expected_ids),
        },
    }
    canonical_json_bytes(fingerprint)
    return fingerprint


def _first_difference(
    expected: Any,
    observed: Any,
    path: str = "fingerprint",
) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(observed):
        return path, expected, observed
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            return (
                f"{path}.keys",
                sorted(str(key) for key in expected),
                sorted(str(key) for key in observed),
            )
        for key in sorted(expected):
            difference = _first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}.length", len(expected), len(observed)
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return path, expected, observed
    return None


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def _values_bits_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return _float_bits_equal(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_bits_equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_bits_equal(left[key], right[key]) for key in left
        )
    return type(left) is type(right) and left == right


def rows_bits_equal(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    """Compare row payloads with IEEE-754 bit identity for every float."""

    return len(left) == len(right) and all(
        _values_bits_equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
    )


def _is_known_installer_temp(path: Path) -> bool:
    if _INSTALLER_TEMP_RE.fullmatch(path.name) is None:
        return False
    if path.is_symlink() or not path.is_file():
        raise _checkpoint_error(
            f"installer temporary entry is not a real regular file: {path.name!r}", path
        )
    return True


class RoundtripCheckpointStore:
    """Validated immutable indexes for generation and probe candidate shards."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        fingerprint: Mapping[str, Any],
        expected_candidate_rows: Sequence[Mapping[str, Any]],
        samples_per_candidate: int,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.manifest_path = checkpoint_dir / MANIFEST_FILENAME
        self.fingerprint = cast(
            dict[str, Any],
            json.loads(canonical_json_bytes(dict(fingerprint)).decode("utf-8")),
        )
        self.fingerprint_sha256 = canonical_sha256(self.fingerprint)
        if isinstance(samples_per_candidate, bool) or samples_per_candidate < 1:
            raise _checkpoint_error(
                f"samples_per_candidate must be a positive integer, got {samples_per_candidate!r}",
                checkpoint_dir,
            )
        self.samples_per_candidate = int(samples_per_candidate)
        self._candidate_rows = {
            str(row["candidate_id"]): dict(row) for row in expected_candidate_rows
        }
        self._candidate_ids = [str(row["candidate_id"]) for row in expected_candidate_rows]
        if len(self._candidate_ids) != len(set(self._candidate_ids)):
            raise _checkpoint_error("expected candidate_id values are not unique", checkpoint_dir)
        self._key_hash_to_id: dict[str, str] = {}
        for candidate_id in self._candidate_ids:
            key_hash = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
            previous = self._key_hash_to_id.setdefault(key_hash, candidate_id)
            if previous != candidate_id:
                raise _checkpoint_error(
                    f"SHA-256 collision between candidate ids {previous!r} and {candidate_id!r}",
                    checkpoint_dir,
                )
        self._generation_rows: dict[str, list[dict[str, Any]]] = {}
        self._probe_rows: dict[str, list[dict[str, Any]]] = {}
        self._runtime_hashes: dict[CheckpointFamily, dict[str, str]] = {
            CheckpointFamily.GENERATION: {},
            CheckpointFamily.PROBE: {},
        }
        self._payloads: dict[CheckpointFamily, dict[str, bytes]] = {
            CheckpointFamily.GENERATION: {},
            CheckpointFamily.PROBE: {},
        }

    @classmethod
    def prepare(
        cls,
        *,
        checkpoint_dir: Path,
        fingerprint: Mapping[str, Any],
        expected_candidate_rows: Sequence[Mapping[str, Any]],
        samples_per_candidate: int,
    ) -> RoundtripCheckpointStore:
        store = cls(
            checkpoint_dir=checkpoint_dir,
            fingerprint=fingerprint,
            expected_candidate_rows=expected_candidate_rows,
            samples_per_candidate=samples_per_candidate,
        )
        store._prepare()
        return store

    @property
    def expected_count(self) -> int:
        return len(self._candidate_ids)

    @property
    def restored_generation_count(self) -> int:
        return len(self._generation_rows)

    @property
    def restored_probe_count(self) -> int:
        return len(self._probe_rows)

    @property
    def pending_count(self) -> int:
        return self.expected_count - self.restored_probe_count

    @property
    def observed_runtime_hashes(self) -> frozenset[str]:
        return frozenset(
            digest for hashes in self._runtime_hashes.values() for digest in hashes.values()
        )

    def observed_runtime_hashes_for_family(self, family: CheckpointFamily) -> frozenset[str]:
        return frozenset(self._runtime_hashes[family].values())

    def load_generation_rows(self) -> dict[str, list[dict[str, Any]]]:
        return deepcopy(self._generation_rows)

    def load_probe_rows(self) -> dict[str, list[dict[str, Any]]]:
        return deepcopy(self._probe_rows)

    def _prepare(self) -> None:
        if self.checkpoint_dir.exists() and (
            self.checkpoint_dir.is_symlink() or not self.checkpoint_dir.is_dir()
        ):
            raise _checkpoint_error("checkpoint root is not a real directory", self.checkpoint_dir)
        if self.checkpoint_dir.exists() and not self.manifest_path.exists():
            try:
                entries = list(self.checkpoint_dir.iterdir())
            except OSError as exc:
                raise _checkpoint_error(
                    f"cannot enumerate checkpoint root: {exc}", self.checkpoint_dir
                ) from exc
            if any(not _is_known_installer_temp(entry) for entry in entries):
                raise _checkpoint_error(
                    "checkpoint contains committed state but no manifest", self.checkpoint_dir
                )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "store_kind": STORE_KIND,
            "fingerprint": self.fingerprint,
            "fingerprint_sha256": self.fingerprint_sha256,
        }
        payload = canonical_json_bytes(manifest)
        if not self.manifest_path.exists():
            try:
                install_bytes_once(payload, self.manifest_path, surface="roundtrip manifest")
            except (ImmutableArtifactError, OSError) as exc:
                raise _checkpoint_error(str(exc), self.manifest_path) from exc
        self._validate_manifest(payload)
        self._scan_all_shards()

    def _validate_manifest(self, expected_payload: bytes) -> None:
        try:
            digest = stream_hash_file_descriptor_bound(
                self.manifest_path,
                retain_payload=True,
                on_error=lambda reason: _checkpoint_error(
                    f"manifest changed while reading ({reason})", self.manifest_path
                ),
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(str(exc), self.manifest_path) from exc
        raw = digest.payload
        if raw is None:
            raise _checkpoint_error("manifest read retained no payload", self.manifest_path)
        try:
            observed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _checkpoint_error(
                f"manifest is malformed JSON: {exc}", self.manifest_path
            ) from exc
        if not isinstance(observed, dict):
            raise _checkpoint_error("manifest root must be an object", self.manifest_path)
        expected_keys = {
            "schema_version",
            "store_kind",
            "fingerprint",
            "fingerprint_sha256",
        }
        if set(observed) != expected_keys:
            raise _checkpoint_error(
                f"manifest keys differ: expected {sorted(expected_keys)}, observed {sorted(observed)}",
                self.manifest_path,
            )
        if isinstance(observed["schema_version"], bool) or not isinstance(
            observed["schema_version"], int
        ):
            raise _checkpoint_error(
                f"manifest.schema_version has invalid type/value {observed['schema_version']!r}",
                self.manifest_path,
            )
        if observed["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise _checkpoint_error(
                "manifest.schema_version differs: "
                f"expected {CHECKPOINT_SCHEMA_VERSION!r}, observed {observed['schema_version']!r}",
                self.manifest_path,
            )
        if observed["store_kind"] != STORE_KIND:
            raise _checkpoint_error(
                f"manifest.store_kind differs: expected {STORE_KIND!r}, "
                f"observed {observed['store_kind']!r}",
                self.manifest_path,
            )
        observed_sha = observed["fingerprint_sha256"]
        if not isinstance(observed_sha, str) or _SHA256_RE.fullmatch(observed_sha) is None:
            raise _checkpoint_error(
                f"manifest.fingerprint_sha256 is invalid: {observed_sha!r}", self.manifest_path
            )
        if not isinstance(observed["fingerprint"], dict):
            raise _checkpoint_error("manifest.fingerprint must be an object", self.manifest_path)
        recomputed = canonical_sha256(observed["fingerprint"])
        if recomputed != observed_sha:
            raise _checkpoint_error(
                "manifest.fingerprint_sha256 differs from canonical fingerprint: "
                f"expected {recomputed!r}, observed {observed_sha!r}",
                self.manifest_path,
            )
        difference = _first_difference(self.fingerprint, observed["fingerprint"])
        if difference is not None:
            field, expected, actual = difference
            raise _checkpoint_error(
                f"manifest {field} differs: expected {expected!r}, observed {actual!r}",
                self.manifest_path,
            )
        if raw != canonical_json_bytes(observed):
            raise _checkpoint_error("manifest payload is not canonical JSON", self.manifest_path)
        if raw != expected_payload:
            raise _checkpoint_error(
                "manifest canonical payload differs from the current contract", self.manifest_path
            )

    def _family_dir(self, family: CheckpointFamily) -> Path:
        return self.checkpoint_dir / family

    def _scan_all_shards(self) -> None:
        self._generation_rows.clear()
        self._probe_rows.clear()
        for hashes in self._runtime_hashes.values():
            hashes.clear()
        for payloads in self._payloads.values():
            payloads.clear()
        try:
            entries = list(self.checkpoint_dir.iterdir())
        except OSError as exc:
            raise _checkpoint_error(
                f"cannot enumerate checkpoint root: {exc}", self.checkpoint_dir
            ) from exc
        allowed = {
            MANIFEST_FILENAME,
            CheckpointFamily.GENERATION.value,
            CheckpointFamily.PROBE.value,
        }
        for entry in entries:
            if entry.name == MANIFEST_FILENAME:
                if entry.is_symlink() or not entry.is_file():
                    raise _checkpoint_error("manifest is not a real regular file", entry)
            elif entry.name in (
                CheckpointFamily.GENERATION.value,
                CheckpointFamily.PROBE.value,
            ):
                if entry.is_symlink() or not entry.is_dir():
                    raise _checkpoint_error(f"{entry.name} entry is not a real directory", entry)
            elif _is_known_installer_temp(entry):
                continue
            elif entry.name not in allowed:
                raise _checkpoint_error(f"unknown checkpoint entry {entry.name!r}", entry)
        self._scan_family(CheckpointFamily.GENERATION)
        self._scan_family(CheckpointFamily.PROBE)

    def _scan_family(self, family: CheckpointFamily) -> None:
        groups_dir = self._family_dir(family) / GROUPS_DIRNAME
        family_dir = self._family_dir(family)
        if not family_dir.exists():
            return
        try:
            family_entries = list(family_dir.iterdir())
        except OSError as exc:
            raise _checkpoint_error(f"cannot enumerate {family} family: {exc}", family_dir) from exc
        for entry in family_entries:
            if entry.name != GROUPS_DIRNAME or entry.is_symlink() or not entry.is_dir():
                raise _checkpoint_error(f"unexpected {family} family entry {entry.name!r}", entry)
        try:
            buckets = sorted(groups_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise _checkpoint_error(f"cannot enumerate {family} groups: {exc}", groups_dir) from exc
        for bucket in buckets:
            if (
                bucket.is_symlink()
                or not bucket.is_dir()
                or _BUCKET_RE.fullmatch(bucket.name) is None
            ):
                raise _checkpoint_error(
                    f"unexpected checkpoint bucket entry {bucket.name!r}", bucket
                )
            try:
                shards = sorted(bucket.iterdir(), key=lambda path: path.name)
            except OSError as exc:
                raise _checkpoint_error(
                    f"cannot enumerate checkpoint bucket: {exc}", bucket
                ) from exc
            for shard in shards:
                if _is_known_installer_temp(shard):
                    continue
                self._load_shard(family, shard, expected_bucket=bucket.name)

    def _load_shard(
        self,
        family: CheckpointFamily,
        path: Path,
        *,
        expected_bucket: str,
    ) -> None:
        if path.is_symlink() or not path.is_file():
            raise _checkpoint_error(
                f"checkpoint shard is not a real regular file: {path.name}", path
            )
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise _checkpoint_error(f"unknown committed checkpoint filename {path.name!r}", path)
        key_hash = match.group("key")
        payload_hash = match.group("payload")
        if key_hash[:2] != expected_bucket:
            raise _checkpoint_error(
                f"checkpoint shard bucket differs: expected {key_hash[:2]!r}, "
                f"observed {expected_bucket!r}",
                path,
            )
        candidate_id = self._key_hash_to_id.get(key_hash)
        if candidate_id is None:
            raise _checkpoint_error(
                f"checkpoint shard has unknown candidate-key hash {key_hash}", path
            )
        destination = (
            self._generation_rows
            if family is CheckpointFamily.GENERATION
            else self._probe_rows
        )
        if candidate_id in destination:
            raise _checkpoint_error(
                f"multiple committed payloads exist for candidate {candidate_id!r} in {family}",
                path,
            )
        try:
            digest = stream_hash_file_descriptor_bound(
                path,
                retain_payload=True,
                on_error=lambda reason: _checkpoint_error(
                    f"checkpoint shard changed while reading ({reason})", path
                ),
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(str(exc), path) from exc
        if digest.sha256 != payload_hash:
            raise _checkpoint_error(
                f"checkpoint shard payload digest differs: expected {payload_hash!r}, "
                f"observed {digest.sha256!r}",
                path,
            )
        if digest.payload is None:
            raise _checkpoint_error("checkpoint shard read retained no payload", path)
        try:
            table = pq.read_table(pa.BufferReader(digest.payload))
        except (pa.ArrowException, OSError) as exc:
            raise _checkpoint_error(f"checkpoint shard is not valid Parquet: {exc}", path) from exc
        rows, runtime_hash = self._validate_table(
            family, table, candidate_id=candidate_id, path=path
        )
        destination[candidate_id] = rows
        self._runtime_hashes[family][candidate_id] = runtime_hash
        self._payloads[family][candidate_id] = digest.payload

    def _validate_table(
        self,
        family: CheckpointFamily,
        table: pa.Table,
        *,
        candidate_id: str,
        path: Path,
    ) -> tuple[list[dict[str, Any]], str]:
        schema = (
            GENERATION_ROW_SCHEMA
            if family is CheckpointFamily.GENERATION
            else ROUNDTRIP_SCORE_SCHEMA
        )
        if not table.schema.remove_metadata().equals(schema):
            raise _checkpoint_error(
                f"checkpoint shard logical schema differs for {family}: expected {schema}, "
                f"observed {table.schema.remove_metadata()}",
                path,
            )
        metadata = table.schema.metadata
        if metadata is None or set(metadata) != _METADATA_KEYS:
            raise _checkpoint_error(
                f"checkpoint shard metadata keys differ: expected {sorted(_METADATA_KEYS)!r}, "
                f"observed {sorted(metadata or {})!r}",
                path,
            )
        expected_metadata = {
            _META_SCHEMA_VERSION: str(CHECKPOINT_SCHEMA_VERSION).encode("ascii"),
            _META_ROOT_FINGERPRINT: self.fingerprint_sha256.encode("ascii"),
            _META_FAMILY: family.encode("ascii"),
            _META_CANDIDATE_ID: candidate_id.encode("utf-8"),
        }
        for key, expected in expected_metadata.items():
            if metadata[key] != expected:
                raise _checkpoint_error(
                    f"checkpoint shard metadata {key.decode()} differs: expected {expected!r}, "
                    f"observed {metadata[key]!r}",
                    path,
                )
        try:
            runtime_hash = metadata[_META_RUNTIME_FINGERPRINT].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _checkpoint_error("checkpoint shard runtime hash is not ASCII", path) from exc
        if _SHA256_RE.fullmatch(runtime_hash) is None:
            raise _checkpoint_error(
                f"checkpoint shard runtime hash is invalid: {runtime_hash!r}", path
            )
        if any(column.null_count for column in table.columns):
            raise _checkpoint_error("checkpoint shard contains null values", path)
        for field in schema:
            if pa.types.is_list(field.type):
                column = table.column(field.name).combine_chunks()
                if column.values.null_count:
                    raise _checkpoint_error(
                        f"checkpoint shard {field.name} contains null values", path
                    )
        rows = table.to_pylist()
        observed_ids = {str(row["candidate_id"]) for row in rows}
        if observed_ids - {candidate_id}:
            raise _checkpoint_error(
                f"checkpoint shard candidate_id differs: expected {candidate_id!r}, "
                f"observed {sorted(observed_ids)!r}",
                path,
            )
        sample_indices = [int(row["sample_index"]) for row in rows]
        if len(sample_indices) != len(set(sample_indices)):
            raise _checkpoint_error(
                f"checkpoint shard has duplicate sample_index values {sample_indices!r}", path
            )
        if family is CheckpointFamily.GENERATION:
            expected_indices = list(range(self.samples_per_candidate))
        else:
            generation = self._generation_rows.get(candidate_id)
            if generation is None:
                raise _checkpoint_error(
                    f"probe shard for candidate {candidate_id!r} has no generation shard", path
                )
            expected_indices = [
                int(row["sample_index"])
                for row in generation
                if str(row["generated_text"]) != "" and str(row["control_generated_text"]) != ""
            ]
            generation_by_index = {int(row["sample_index"]): row for row in generation}
            for row in rows:
                generation_row = {name: row[name] for name in GENERATION_ROW_SCHEMA.names}
                generation_source = generation_by_index.get(int(row["sample_index"]))
                if generation_source is None or not rows_bits_equal(
                    [generation_row], [generation_source]
                ):
                    raise _checkpoint_error(
                        f"probe shard generation payload differs for candidate {candidate_id!r} "
                        f"sample_index={row['sample_index']}",
                        path,
                    )
        if sample_indices != expected_indices:
            raise _checkpoint_error(
                f"checkpoint shard sample_index coverage differs for {candidate_id!r}: "
                f"expected {expected_indices!r}, observed {sample_indices!r}",
                path,
            )
        for row in rows:
            self._validate_row(family, row, candidate_id=candidate_id, path=path)
        return rows, runtime_hash

    def _validate_row(
        self,
        family: CheckpointFamily,
        row: Mapping[str, Any],
        *,
        candidate_id: str,
        path: Path,
    ) -> None:
        candidate = self._candidate_rows[candidate_id]
        mismatches = [
            name
            for name in _STATIC_CANDIDATE_COLUMNS
            if not _values_bits_equal(row[name], candidate[name])
        ]
        if mismatches:
            raise _checkpoint_error(
                f"checkpoint shard static candidate projection differs for "
                f"{candidate_id!r}: {mismatches}",
                path,
            )
        max_tokens = int(row["reference_segment_token_count"])
        if max_tokens < 1:
            raise _checkpoint_error(
                "checkpoint shard reference_segment_token_count must be >= 1, "
                f"observed {max_tokens}",
                path,
            )
        for side_prefix in ("", "control_"):
            count = int(row[f"{side_prefix}n_generated_tokens"])
            text = str(row[f"{side_prefix}generated_text"])
            stop_reason = str(row[f"{side_prefix}stop_reason"])
            if count < 0 or count > max_tokens:
                raise _checkpoint_error(
                    f"checkpoint shard {side_prefix}n_generated_tokens must be in "
                    f"[0, {max_tokens}], observed {count}",
                    path,
                )
            if stop_reason not in {"bos_token", "max_tokens"}:
                raise _checkpoint_error(
                    f"checkpoint shard {side_prefix}stop_reason is invalid: {stop_reason!r}",
                    path,
                )
            if (count == 0) != (text == ""):
                raise _checkpoint_error(
                    f"checkpoint shard {side_prefix}generated_text/token-count "
                    f"consistency differs: text_empty={text == ''}, count={count}",
                    path,
                )
        if family is CheckpointFamily.PROBE:
            for name in _PROBE_METRIC_COLUMNS:
                if not math.isfinite(float(row[name])):
                    raise _checkpoint_error(
                        f"checkpoint shard probe metric {name} must be finite, "
                        f"observed {row[name]!r}",
                        path,
                    )

    def _commit(
        self,
        family: CheckpointFamily,
        candidate_id: str,
        rows: Sequence[Mapping[str, Any]],
        runtime_fingerprint: str,
    ) -> bool:
        if candidate_id not in self._candidate_ids:
            raise _checkpoint_error(
                f"cannot commit unknown candidate_id={candidate_id!r}", self.checkpoint_dir
            )
        if (
            not isinstance(runtime_fingerprint, str)
            or _SHA256_RE.fullmatch(runtime_fingerprint) is None
        ):
            raise _checkpoint_error(
                f"runtime fingerprint is invalid: {runtime_fingerprint!r}", self.checkpoint_dir
            )
        schema = (
            GENERATION_ROW_SCHEMA
            if family is CheckpointFamily.GENERATION
            else ROUNDTRIP_SCORE_SCHEMA
        )
        for row in rows:
            if set(row) != set(schema.names):
                raise _checkpoint_error(
                    f"{family} row columns differ: expected {schema.names}, observed {sorted(row)}",
                    self.checkpoint_dir,
                )
        try:
            logical = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
        except (ArrowInvalid, ArrowTypeError, OverflowError, TypeError, ValueError) as exc:
            raise _checkpoint_error(
                f"{family} rows do not conform to their schema: {exc}", self.checkpoint_dir
            ) from exc
        metadata = {
            _META_SCHEMA_VERSION: str(CHECKPOINT_SCHEMA_VERSION).encode("ascii"),
            _META_ROOT_FINGERPRINT: self.fingerprint_sha256.encode("ascii"),
            _META_FAMILY: family.encode("ascii"),
            _META_CANDIDATE_ID: candidate_id.encode("utf-8"),
            _META_RUNTIME_FINGERPRINT: runtime_fingerprint.encode("ascii"),
        }
        table = logical.replace_schema_metadata(metadata)
        normalized, _runtime = self._validate_table(
            family, table, candidate_id=candidate_id, path=self.checkpoint_dir
        )
        try:
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            payload = sink.getvalue().to_pybytes()
        except (pa.ArrowException, OSError) as exc:
            raise _checkpoint_error(
                f"could not serialize {family} rows as Parquet: {exc}", self.checkpoint_dir
            ) from exc
        destination = (
            self._generation_rows
            if family is CheckpointFamily.GENERATION
            else self._probe_rows
        )
        existing = destination.get(candidate_id)
        if existing is not None:
            if self._payloads[family][candidate_id] == payload:
                return False
            if not rows_bits_equal(existing, normalized):
                raise _checkpoint_error(
                    f"{family} rows differ from committed candidate {candidate_id!r}",
                    self.checkpoint_dir,
                )
            return False
        key_hash = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        payload_hash = hashlib.sha256(payload).hexdigest()
        bucket = self._family_dir(family) / GROUPS_DIRNAME / key_hash[:2]
        path = bucket / f"cand_{key_hash}_{payload_hash}.parquet"
        try:
            installed = install_bytes_once(
                payload, path, surface=f"roundtrip {family} candidate {candidate_id}"
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(str(exc), path) from exc
        try:
            matches = [
                candidate
                for candidate in bucket.iterdir()
                if not _is_known_installer_temp(candidate)
                and candidate.name.startswith(f"cand_{key_hash}_")
            ]
        except OSError as exc:
            raise _checkpoint_error(f"cannot enumerate checkpoint bucket: {exc}", bucket) from exc
        if len(matches) != 1:
            raise _checkpoint_error(
                f"expected exactly one committed {family} payload for candidate "
                f"{candidate_id!r}, observed {[item.name for item in sorted(matches)]}",
                bucket,
            )
        self._load_shard(family, matches[0], expected_bucket=key_hash[:2])
        return installed

    def commit_generation_rows(
        self,
        candidate_id: str,
        rows: Sequence[Mapping[str, Any]],
        runtime_fingerprint: str,
    ) -> bool:
        return self._commit(
            CheckpointFamily.GENERATION,
            candidate_id,
            rows,
            runtime_fingerprint,
        )

    def commit_probe_rows(
        self,
        candidate_id: str,
        rows: Sequence[Mapping[str, Any]],
        runtime_fingerprint: str,
    ) -> bool:
        return self._commit(
            CheckpointFamily.PROBE,
            candidate_id,
            rows,
            runtime_fingerprint,
        )


def warn_on_roundtrip_runtime_mismatch(
    store: RoundtripCheckpointStore,
    family: CheckpointFamily,
    current_runtime_sha256: str,
) -> None:
    differing = sorted(store.observed_runtime_hashes_for_family(family) - {current_runtime_sha256})
    if differing:
        logger.warning(
            "ROUNDTRIP {} RESUME IS MIXING NUMERIC RUNTIMES: current={} restored={}. "
            "Reuse is allowed and auditable, but byte-identical parity is guaranteed "
            "only when runtime fingerprints match.",
            family.upper(),
            current_runtime_sha256,
            differing,
        )


__all__ = [
    "CheckpointFamily",
    "GENERATION_ROW_SCHEMA",
    "RoundtripCheckpointStore",
    "build_roundtrip_checkpoint_fingerprint",
    "candidates_payload_sha256",
    "current_runtime_fingerprint",
    "resolve_extraction_tokenizer_content_identity",
    "resolve_probe_artifact_identity",
    "rows_bits_equal",
    "warn_on_roundtrip_runtime_mismatch",
]
