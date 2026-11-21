"""Crash-safe, immutable per-group checkpoints for universal Faster-GCG.

The checkpoint record is deliberately only the already-public universal-trigger
row. Candidate rows remain owned by ``TokenTriggerEditSpace.finalize`` and the
canonical stage artifacts remain the only downstream authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger
from pyarrow import ArrowInvalid, ArrowTypeError

from nanochat.checkpoint_manager import (
    find_largest_model,
    find_last_step,
    get_checkpoint_dir,
)
from nanochat.common import get_base_dir
from steering.eval.autointerp_filter.batch_validation import runtime_numeric_fingerprint
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.schemas import UNIVERSAL_TRIGGERS_SCHEMA
from steering.eval.axbench_bpb_adv.universal_gcg import ConceptGroup, group_key
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    stream_hash_file_descriptor_bound,
)

CHECKPOINT_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "axbench_bpb_adv_universal_gcg_group_checkpoint_v1"
MANIFEST_FILENAME = "manifest.json"
GROUPS_DIRNAME = "groups"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_RE = re.compile(r"^[0-9a-f]{2}$")
_SHARD_RE = re.compile(
    r"^group_(?P<key>[0-9a-f]{64})_(?P<payload>[0-9a-f]{64})\.parquet$"
)
_INSTALLER_TEMP_RE = re.compile(
    r"^\.(?:manifest\.json|group_[0-9a-f]{64}_[0-9a-f]{64}\.parquet)"
    r"\.tmp\.\d+\.[0-9a-f]{8}$"
)

_META_SCHEMA_VERSION = b"checkpoint_schema_version"
_META_ROOT_FINGERPRINT = b"manifest_fingerprint_sha256"
_META_GROUP_KEY = b"group_key_str"
_META_RUNTIME_FINGERPRINT = b"runtime_numeric_fingerprint_sha256"
_METADATA_KEYS = {
    _META_SCHEMA_VERSION,
    _META_ROOT_FINGERPRINT,
    _META_GROUP_KEY,
    _META_RUNTIME_FINGERPRINT,
}


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON used by every persisted/digested checkpoint payload."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(f"checkpoint value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def _is_known_installer_temp(path: Path) -> bool:
    """Return true only for a real regular file with the installer's temp name."""

    if _INSTALLER_TEMP_RE.fullmatch(path.name) is None:
        return False
    if path.is_symlink() or not path.is_file():
        raise _checkpoint_error(
            f"installer temporary entry is not a real regular file: {path.name!r}",
            path,
        )
    return True


def _enumerate_regular_tree(root: Path) -> list[Path]:
    """Return sorted relative regular-file paths, rejecting every other leaf."""

    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise _checkpoint_error(f"cannot inspect required tree {root}: {exc}", root) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _checkpoint_error(f"required tree root is not a real directory: {root}", root)

    relative: list[Path] = []

    def _walk(directory: Path, prefix: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise _checkpoint_error(f"cannot enumerate tree directory {directory}: {exc}", root) from exc
        for entry in entries:
            rel = prefix / entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise _checkpoint_error(f"cannot inspect tree entry {rel}: {exc}", root) from exc
            if stat.S_ISDIR(entry_stat.st_mode):
                _walk(Path(entry.path), rel)
            elif stat.S_ISREG(entry_stat.st_mode):
                relative.append(rel)
            else:
                raise _checkpoint_error(
                    f"tree entry {rel} is a symlink or non-regular file", root
                )

    _walk(root, Path())
    return sorted(relative, key=lambda path: path.as_posix())


def tree_sha256(root: Path) -> str:
    """Digest a stable, sorted directory tree using descriptor-bound file hashes."""

    before = _enumerate_regular_tree(root)
    descriptors: list[dict[str, Any]] = []
    for relative in before:
        path = root / relative
        try:
            digest = stream_hash_file_descriptor_bound(
                path,
                on_error=lambda reason, rel=relative: _checkpoint_error(
                    f"tree file {rel.as_posix()} changed while hashing ({reason})", root
                ),
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(f"could not hash tree file {relative}: {exc}", root) from exc
        descriptors.append(
            {
                "path": relative.as_posix(),
                "byte_length": digest.byte_length,
                "sha256": digest.sha256,
            }
        )
    after = _enumerate_regular_tree(root)
    if before != after:
        raise _checkpoint_error(
            f"tree file set changed while hashing: expected {[p.as_posix() for p in before]}, "
            f"observed {[p.as_posix() for p in after]}",
            root,
        )
    return canonical_sha256(descriptors)


@dataclass(frozen=True)
class ResolvedTargetCheckpoint:
    target_name: str
    source: str
    model_tag: str
    step: int
    checkpoint_dir: Path
    model_path: Path
    metadata_path: Path
    model_sha256: str
    metadata_sha256: str


def resolve_and_pin_target_checkpoints(
    cfg: AxBenchBpbConfig,
    target_names: set[str],
) -> dict[str, ResolvedTargetCheckpoint]:
    """Resolve nullable tag/step once, pin them into ``cfg``, and hash leaves."""

    resolved: dict[str, ResolvedTargetCheckpoint] = {}
    for target_name in sorted(target_names):
        if target_name == "decoder_only":
            target_cfg = dict(cfg.decoder_only)
        elif target_name == "encdec":
            target_cfg = dict(cfg.encdec)
        else:
            raise AxBenchBpbAdvError(f"unsupported Faster-GCG target {target_name!r}")
        source = target_cfg.get("source")
        if not isinstance(source, str) or not source:
            raise AxBenchBpbAdvError(
                f"Faster-GCG target {target_name!r} requires a non-empty source"
            )
        configured_tag = target_cfg.get("model_tag")
        if configured_tag is not None and (
            not isinstance(configured_tag, str) or not configured_tag
        ):
            raise AxBenchBpbAdvError(
                f"Faster-GCG target {target_name!r} model_tag must be null or a non-empty string"
            )
        if configured_tag is None:
            root_name = (
                "conditional_encdec_checkpoints"
                if source.split("/", 1)[0] == "encdec"
                else "base_checkpoints"
            )
            configured_tag = str(find_largest_model(str(Path(get_base_dir()) / root_name)))
        checkpoint_dir = Path(get_checkpoint_dir(source, model_tag=configured_tag))
        configured_step = target_cfg.get("step")
        if configured_step is None:
            configured_step = int(find_last_step(str(checkpoint_dir)))
        if (
            isinstance(configured_step, bool)
            or not isinstance(configured_step, int)
            or configured_step < 0
        ):
            raise AxBenchBpbAdvError(
                f"Faster-GCG target {target_name!r} step must be null or a non-negative integer"
            )
        target_cfg["model_tag"] = configured_tag
        target_cfg["step"] = configured_step
        if target_name == "decoder_only":
            cfg.decoder_only = target_cfg
        else:
            cfg.encdec = target_cfg
        model_path = checkpoint_dir / f"model_{configured_step:06d}.pt"
        metadata_path = checkpoint_dir / f"meta_{configured_step:06d}.json"
        resolved[target_name] = ResolvedTargetCheckpoint(
            target_name=target_name,
            source=source,
            model_tag=configured_tag,
            step=configured_step,
            checkpoint_dir=checkpoint_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            model_sha256=_hash_file(model_path, label=f"{target_name} model checkpoint"),
            metadata_sha256=_hash_file(
                metadata_path, label=f"{target_name} metadata checkpoint"
            ),
        )
    return resolved


def assert_target_checkpoints_unchanged(
    resolved: Mapping[str, ResolvedTargetCheckpoint],
) -> None:
    for target_name, target in sorted(resolved.items()):
        for field, path, expected in (
            ("model_sha256", target.model_path, target.model_sha256),
            ("metadata_sha256", target.metadata_path, target.metadata_sha256),
        ):
            observed = _hash_file(path, label=f"{target_name} {field}")
            if observed != expected:
                raise _checkpoint_error(
                    f"target identity changed after runtime setup at targets.{target_name}.{field}: "
                    f"expected {expected!r}, observed {observed!r}",
                    path,
                )


def _effective_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    # ``load_model`` appends a machine-local path after resolving/recovering the
    # semantic metadata. Paths are not identity; the recovered content is.
    semantic = {key: value for key, value in metadata.items() if key != "checkpoint_dir"}
    return canonical_sha256(semantic)


def build_checkpoint_fingerprint(
    *,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    faster_cfg: FasterGcgConfig,
    groups: Sequence[ConceptGroup],
    resolved_targets: Mapping[str, ResolvedTargetCheckpoint],
    loaded_metadata: Mapping[str, Mapping[str, Any]],
    cluster_feature_contract_fingerprint: str | None,
) -> dict[str, Any]:
    """Build the complete semantic identity for Faster-GCG group restoration."""

    group_keys = sorted(group.key_str for group in groups if not group.is_control)
    if len(group_keys) != len(set(group_keys)):
        raise AxBenchBpbAdvError("expected Faster-GCG group keys must be unique")
    targets: list[dict[str, Any]] = []
    for target_name, target in sorted(resolved_targets.items()):
        metadata = loaded_metadata.get(target_name)
        if metadata is None:
            raise AxBenchBpbAdvError(
                f"loaded metadata is missing for Faster-GCG target {target_name!r}"
            )
        targets.append(
            {
                "target_name": target_name,
                "source": target.source,
                "model_tag": target.model_tag,
                "step": target.step,
                "model_sha256": target.model_sha256,
                "metadata_sha256": target.metadata_sha256,
                "effective_loaded_metadata_sha256": _effective_metadata_sha256(metadata),
            }
        )
    uses_encdec = any(target.target_name == "encdec" for target in resolved_targets.values())
    uses_bags = any(
        precomputed_bag_method_family(str(group.rows[0]["method_name"])) is not None
        for group in groups
        if not group.is_control
    )
    tokenizer_dir = Path(get_base_dir()) / "tokenizer"
    fingerprint: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "source_contract_sha256": _hash_file(
            paths.source_contract(), label="source_contract.json"
        ),
        "attack_rows_sha256": _hash_file(
            paths.attack_rows_path(), label="attack_rows.parquet"
        ),
        "evaluation_population_sha256": _hash_file(
            paths.evaluation_population(), label="evaluation_population.json"
        ),
        "cluster_feature_contract_fingerprint": cluster_feature_contract_fingerprint,
        "faster_gcg_config": asdict(faster_cfg),
        "objective": {
            "device_batch_size": int(cfg.device_batch_size),
            "diffmean_and_probe_layer": int(cfg.diffmean_and_probe_layer),
            "next_segment_conditioning_offset": bool(
                cfg.next_segment_conditioning_offset
            ),
            "lora_adapter": dict(cfg.lora_adapter),
        },
        "targets": targets,
        "tokenizer_tree_sha256": tree_sha256(tokenizer_dir),
        "feature_statistics_sha256": _hash_file(
            Path(cfg.feature_statistics_path), label="feature_statistics.safetensors"
        ),
        "context_cache_tree_sha256": (
            tree_sha256(paths.context_cache_dir()) if uses_encdec else "unused"
        ),
        "ours_gcg_bags_tree_sha256": (
            tree_sha256(paths.ours_gcg_bags_dir()) if uses_bags else "unused"
        ),
        "expected_group_keys": {
            "count": len(group_keys),
            "sha256": canonical_sha256(group_keys),
        },
    }
    # Make non-JSON values fail at fingerprint construction, not at installation.
    canonical_json_bytes(fingerprint)
    return fingerprint


def current_runtime_fingerprint(device: str | t.device) -> tuple[dict[str, Any], str]:
    resolved = t.device(device)
    if resolved.type == "cuda":
        payload = runtime_numeric_fingerprint(resolved)
    else:
        payload = {
            "device_type": resolved.type,
            "torch": t.__version__,
            "float32_matmul_precision": t.get_float32_matmul_precision(),
            "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        }
    return payload, canonical_sha256(payload)


def _first_difference(expected: Any, observed: Any, path: str = "fingerprint") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(observed):
        return (path, expected, observed)
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return (
                f"{path}.keys",
                sorted(str(key) for key in expected_keys),
                sorted(str(key) for key in observed_keys),
            )
        for key in sorted(expected):
            difference = _first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return (f"{path}.length", len(expected), len(observed))
        for index, (lhs, rhs) in enumerate(zip(expected, observed, strict=True)):
            difference = _first_difference(lhs, rhs, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return (path, expected, observed)
    return None


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


class UniversalGcgCheckpointStore:
    """One-pass validated immutable index of completed Faster-GCG groups."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        fingerprint: Mapping[str, Any],
        expected_groups: Sequence[ConceptGroup],
        tokenizer: Any,
        max_steps: int,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.manifest_path = checkpoint_dir / MANIFEST_FILENAME
        self.groups_dir = checkpoint_dir / GROUPS_DIRNAME
        self.fingerprint = cast(
            dict[str, Any],
            json.loads(canonical_json_bytes(dict(fingerprint)).decode("utf-8")),
        )
        self.fingerprint_sha256 = canonical_sha256(self.fingerprint)
        self.tokenizer = tokenizer
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise _checkpoint_error(
                f"max_steps must be a non-negative integer, observed {max_steps!r}",
                checkpoint_dir,
            )
        self.max_steps = max_steps
        self._groups: dict[str, ConceptGroup] = {
            group.key_str: group for group in expected_groups if not group.is_control
        }
        if len(self._groups) != len([g for g in expected_groups if not g.is_control]):
            raise _checkpoint_error("expected group keys are not unique", checkpoint_dir)
        self._key_hash_to_key: dict[str, str] = {}
        for key in self._groups:
            key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
            previous = self._key_hash_to_key.setdefault(key_hash, key)
            if previous != key:
                raise _checkpoint_error(
                    f"SHA-256 collision between expected group keys {previous!r} and {key!r}",
                    checkpoint_dir,
                )
        self._rows: dict[str, dict[str, Any]] = {}
        self._runtime_hashes: dict[str, str] = {}

    @classmethod
    def prepare(
        cls,
        *,
        checkpoint_dir: Path,
        fingerprint: Mapping[str, Any],
        expected_groups: Sequence[ConceptGroup],
        tokenizer: Any,
        max_steps: int,
    ) -> UniversalGcgCheckpointStore:
        store = cls(
            checkpoint_dir=checkpoint_dir,
            fingerprint=fingerprint,
            expected_groups=expected_groups,
            tokenizer=tokenizer,
            max_steps=max_steps,
        )
        store._prepare()
        return store

    @property
    def observed_runtime_hashes(self) -> frozenset[str]:
        return frozenset(self._runtime_hashes.values())

    def load_triggers(
        self,
        expected_groups: Sequence[ConceptGroup] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if expected_groups is None:
            return deepcopy(self._rows)
        keys = {group.key_str for group in expected_groups if not group.is_control}
        unknown = keys - set(self._groups)
        if unknown:
            raise _checkpoint_error(
                f"load_triggers received unknown expected groups {sorted(unknown)}",
                self.checkpoint_dir,
            )
        return {key: deepcopy(self._rows[key]) for key in keys if key in self._rows}

    def _prepare(self) -> None:
        if self.checkpoint_dir.exists():
            if self.checkpoint_dir.is_symlink() or not self.checkpoint_dir.is_dir():
                raise _checkpoint_error(
                    "checkpoint root is not a real directory",
                    self.checkpoint_dir,
                )
        if self.checkpoint_dir.exists() and not self.manifest_path.exists():
            try:
                entries = list(self.checkpoint_dir.iterdir())
            except OSError as exc:
                raise _checkpoint_error(
                    f"cannot enumerate checkpoint root: {exc}", self.checkpoint_dir
                ) from exc
            committed = [entry for entry in entries if not _is_known_installer_temp(entry)]
            if committed:
                raise _checkpoint_error(
                    "checkpoint contains committed state but no manifest",
                    self.checkpoint_dir,
                )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "fingerprint": self.fingerprint,
            "fingerprint_sha256": self.fingerprint_sha256,
        }
        payload = canonical_json_bytes(manifest)
        if not self.manifest_path.exists():
            try:
                install_bytes_once(payload, self.manifest_path, surface="universal GCG manifest")
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
            raise _checkpoint_error(f"manifest is malformed JSON: {exc}", self.manifest_path) from exc
        if not isinstance(observed, dict):
            raise _checkpoint_error("manifest root must be an object", self.manifest_path)
        expected_keys = {
            "schema_version",
            "algorithm_version",
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
                f"manifest.schema_version differs: expected {CHECKPOINT_SCHEMA_VERSION!r}, "
                f"observed {observed['schema_version']!r}",
                self.manifest_path,
            )
        if observed["algorithm_version"] != ALGORITHM_VERSION:
            raise _checkpoint_error(
                f"manifest.algorithm_version differs: expected {ALGORITHM_VERSION!r}, "
                f"observed {observed['algorithm_version']!r}",
                self.manifest_path,
            )
        observed_sha = observed["fingerprint_sha256"]
        if not isinstance(observed_sha, str) or _SHA256_RE.fullmatch(observed_sha) is None:
            raise _checkpoint_error(
                f"manifest.fingerprint_sha256 is invalid: {observed_sha!r}",
                self.manifest_path,
            )
        if not isinstance(observed["fingerprint"], dict):
            raise _checkpoint_error("manifest.fingerprint must be an object", self.manifest_path)
        try:
            recomputed = canonical_sha256(observed["fingerprint"])
        except AxBenchBpbAdvError as exc:
            raise _checkpoint_error(
                f"manifest.fingerprint is not canonical JSON: {exc}",
                self.manifest_path,
            ) from exc
        if recomputed != observed_sha:
            raise _checkpoint_error(
                f"manifest.fingerprint_sha256 differs from canonical fingerprint: "
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
        try:
            canonical_observed = canonical_json_bytes(observed)
        except AxBenchBpbAdvError as exc:
            raise _checkpoint_error(str(exc), self.manifest_path) from exc
        if raw != canonical_observed:
            raise _checkpoint_error("manifest payload is not canonical JSON", self.manifest_path)
        if raw != expected_payload:
            raise _checkpoint_error(
                "manifest canonical payload differs from the current contract",
                self.manifest_path,
            )

    def _scan_all_shards(self) -> None:
        self._rows.clear()
        self._runtime_hashes.clear()
        try:
            root_entries = list(self.checkpoint_dir.iterdir())
        except OSError as exc:
            raise _checkpoint_error(
                f"cannot enumerate checkpoint root: {exc}", self.checkpoint_dir
            ) from exc
        for entry in root_entries:
            if entry.name == MANIFEST_FILENAME:
                if not entry.is_file() or entry.is_symlink():
                    raise _checkpoint_error("manifest is not a real regular file", entry)
                continue
            if entry.name == GROUPS_DIRNAME:
                if not entry.is_dir() or entry.is_symlink():
                    raise _checkpoint_error("groups entry is not a real directory", entry)
                continue
            if _is_known_installer_temp(entry):
                continue
            raise _checkpoint_error(f"unknown checkpoint entry {entry.name!r}", entry)
        if not self.groups_dir.exists():
            return
        try:
            buckets = sorted(self.groups_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise _checkpoint_error(
                f"cannot enumerate checkpoint groups: {exc}", self.groups_dir
            ) from exc
        for bucket in buckets:
            if not bucket.is_dir() or bucket.is_symlink() or _BUCKET_RE.fullmatch(bucket.name) is None:
                raise _checkpoint_error(f"unexpected checkpoint bucket entry {bucket.name!r}", bucket)
            try:
                shards = sorted(bucket.iterdir(), key=lambda path: path.name)
            except OSError as exc:
                raise _checkpoint_error(
                    f"cannot enumerate checkpoint bucket: {exc}", bucket
                ) from exc
            for shard in shards:
                if _is_known_installer_temp(shard):
                    continue
                self._load_shard(shard, expected_bucket=bucket.name)

    def _load_shard(self, path: Path, *, expected_bucket: str) -> None:
        if not path.is_file() or path.is_symlink():
            raise _checkpoint_error(f"checkpoint shard is not a real regular file: {path.name}", path)
        match = _SHARD_RE.fullmatch(path.name)
        if match is None:
            raise _checkpoint_error(f"unknown committed checkpoint filename {path.name!r}", path)
        key_hash = match.group("key")
        payload_hash = match.group("payload")
        if key_hash[:2] != expected_bucket:
            raise _checkpoint_error(
                f"checkpoint shard bucket differs: expected {key_hash[:2]!r}, observed {expected_bucket!r}",
                path,
            )
        key = self._key_hash_to_key.get(key_hash)
        if key is None:
            raise _checkpoint_error(f"checkpoint shard has unknown group-key hash {key_hash}", path)
        if key in self._rows:
            raise _checkpoint_error(f"multiple committed payloads exist for group {key!r}", path)
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
        payload = digest.payload
        if payload is None:
            raise _checkpoint_error("checkpoint shard read retained no payload", path)
        try:
            table = pq.read_table(pa.BufferReader(payload))
        except (pa.ArrowException, OSError) as exc:
            raise _checkpoint_error(f"checkpoint shard is not valid Parquet: {exc}", path) from exc
        row, runtime_hash = self._validate_table(table, key=key, path=path)
        self._rows[key] = row
        self._runtime_hashes[key] = runtime_hash

    def _validate_table(
        self,
        table: pa.Table,
        *,
        key: str,
        path: Path,
    ) -> tuple[dict[str, Any], str]:
        if table.num_rows != 1:
            raise _checkpoint_error(
                f"checkpoint shard must contain exactly one row, observed {table.num_rows}", path
            )
        if not table.schema.remove_metadata().equals(UNIVERSAL_TRIGGERS_SCHEMA):
            raise _checkpoint_error(
                f"checkpoint shard logical schema differs: expected {UNIVERSAL_TRIGGERS_SCHEMA}, "
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
            _META_GROUP_KEY: key.encode("utf-8"),
        }
        for meta_key, expected in expected_metadata.items():
            if metadata[meta_key] != expected:
                raise _checkpoint_error(
                    f"checkpoint shard metadata {meta_key.decode()} differs: "
                    f"expected {expected!r}, observed {metadata[meta_key]!r}",
                    path,
                )
        runtime_bytes = metadata[_META_RUNTIME_FINGERPRINT]
        try:
            runtime_hash = runtime_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise _checkpoint_error("checkpoint shard runtime hash is not ASCII", path) from exc
        if _SHA256_RE.fullmatch(runtime_hash) is None:
            raise _checkpoint_error(
                f"checkpoint shard runtime hash is invalid: {runtime_hash!r}", path
            )
        if any(column.null_count for column in table.columns):
            raise _checkpoint_error("checkpoint shard contains null values", path)
        trigger_tokens = table.column("trigger_tokens").combine_chunks()
        if trigger_tokens.values.null_count:
            raise _checkpoint_error("checkpoint shard trigger_tokens contains null values", path)
        row = table.to_pylist()[0]
        self._validate_row(row, key=key, path=path)
        return row, runtime_hash

    def _validate_row(self, row: Mapping[str, Any], *, key: str, path: Path) -> None:
        group = self._groups[key]
        try:
            observed_key = group_key(row)
        except (AxBenchBpbAdvError, KeyError, TypeError, ValueError) as exc:
            raise _checkpoint_error(
                f"checkpoint shard has invalid group identity for {key!r}: {exc}", path
            ) from exc
        if observed_key != group.key:
            raise _checkpoint_error(
                f"checkpoint shard group identity differs for {key!r}: "
                f"expected {group.key!r}, observed {observed_key!r}",
                path,
            )
        expected_strength = float(group.rows[0]["adversarial_strength"])
        if not _float_bits_equal(float(row["adversarial_strength"]), expected_strength):
            raise _checkpoint_error(
                f"checkpoint shard adversarial_strength differs for {key!r}: "
                f"expected {expected_strength!r}, observed {row['adversarial_strength']!r}",
                path,
            )
        if int(row["max_steps"]) != self.max_steps:
            raise _checkpoint_error(
                f"checkpoint shard max_steps differs for {key!r}: expected {self.max_steps!r}, "
                f"observed {row['max_steps']!r}",
                path,
            )
        if int(row["n_pairs"]) != len(group.rows):
            raise _checkpoint_error(
                f"checkpoint shard n_pairs differs for {key!r}: expected {len(group.rows)!r}, "
                f"observed {row['n_pairs']!r}",
                path,
            )
        accepted_steps = int(row["accepted_steps"])
        if accepted_steps < 0 or accepted_steps > self.max_steps:
            raise _checkpoint_error(
                f"checkpoint shard accepted_steps must be in [0, {self.max_steps}], "
                f"observed {accepted_steps}",
                path,
            )
        visited = int(row["visited_sequences"])
        if visited < 1:
            raise _checkpoint_error(
                f"checkpoint shard visited_sequences must be >= 1, observed {visited}", path
            )
        if accepted_steps >= visited:
            raise _checkpoint_error(
                "checkpoint shard accepted_steps must be smaller than "
                f"visited_sequences, observed accepted_steps={accepted_steps}, "
                f"visited_sequences={visited}",
                path,
            )
        tokens = [int(token_id) for token_id in row["trigger_tokens"]]
        if len(tokens) != int(group.editable_token_count):
            raise _checkpoint_error(
                f"checkpoint shard trigger length differs for {key!r}: expected "
                f"{int(group.editable_token_count)}, observed {len(tokens)}",
                path,
            )
        try:
            decoded = str(self.tokenizer.decode(tokens))
        except Exception as exc:
            raise _checkpoint_error(
                f"checkpoint shard trigger tokens cannot be decoded for {key!r}: {exc}", path
            ) from exc
        if row["trigger_text"] != decoded:
            raise _checkpoint_error(
                f"checkpoint shard trigger_text differs for {key!r}: expected {decoded!r}, "
                f"observed {row['trigger_text']!r}",
                path,
            )
        objective = float(row["objective_value"])
        if math.isnan(objective) or objective == -math.inf:
            raise _checkpoint_error(
                f"checkpoint shard objective_value is impossible: {objective!r}", path
            )

    def commit_trigger(
        self,
        group_key_str: str,
        trigger_row: Mapping[str, Any],
        runtime_fingerprint: str,
    ) -> bool:
        """Durably install one completed group before its ``group_done`` event."""

        if not isinstance(group_key_str, str):
            raise _checkpoint_error(
                f"group key must be a string, observed {group_key_str!r}",
                self.checkpoint_dir,
            )
        group = self._groups.get(group_key_str)
        if group is None:
            raise _checkpoint_error(
                f"cannot commit unknown or control group {group_key_str!r}", self.checkpoint_dir
            )
        if (
            not isinstance(runtime_fingerprint, str)
            or _SHA256_RE.fullmatch(runtime_fingerprint) is None
        ):
            raise _checkpoint_error(
                f"runtime fingerprint is invalid: {runtime_fingerprint!r}", self.checkpoint_dir
            )
        if set(trigger_row) != set(UNIVERSAL_TRIGGERS_SCHEMA.names):
            raise _checkpoint_error(
                "completed trigger columns differ from UNIVERSAL_TRIGGERS_SCHEMA: "
                f"expected {UNIVERSAL_TRIGGERS_SCHEMA.names}, observed {sorted(trigger_row)}",
                self.checkpoint_dir,
            )
        try:
            logical = pa.Table.from_pylist([dict(trigger_row)], schema=UNIVERSAL_TRIGGERS_SCHEMA)
        except (ArrowInvalid, ArrowTypeError, OverflowError, TypeError, ValueError) as exc:
            raise _checkpoint_error(
                f"completed trigger does not conform to UNIVERSAL_TRIGGERS_SCHEMA: {exc}",
                self.checkpoint_dir,
            ) from exc
        metadata = {
            _META_SCHEMA_VERSION: str(CHECKPOINT_SCHEMA_VERSION).encode("ascii"),
            _META_ROOT_FINGERPRINT: self.fingerprint_sha256.encode("ascii"),
            _META_GROUP_KEY: group_key_str.encode("utf-8"),
            _META_RUNTIME_FINGERPRINT: runtime_fingerprint.encode("ascii"),
        }
        table = logical.replace_schema_metadata(metadata)
        normalized, _runtime = self._validate_table(
            table,
            key=group_key_str,
            path=self.checkpoint_dir,
        )
        existing = self._rows.get(group_key_str)
        if existing is not None:
            if any(
                (
                    not _float_bits_equal(existing[name], normalized[name])
                    if isinstance(existing[name], float)
                    and isinstance(normalized[name], float)
                    else existing[name] != normalized[name]
                )
                for name in UNIVERSAL_TRIGGERS_SCHEMA.names
            ):
                raise _checkpoint_error(
                    f"completed trigger differs from committed group {group_key_str!r}",
                    self.checkpoint_dir,
                )
            return False
        try:
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            payload = sink.getvalue().to_pybytes()
        except (pa.ArrowException, OSError) as exc:
            raise _checkpoint_error(
                f"could not serialize completed trigger as Parquet: {exc}",
                self.checkpoint_dir,
            ) from exc
        key_hash = hashlib.sha256(group_key_str.encode("utf-8")).hexdigest()
        payload_hash = hashlib.sha256(payload).hexdigest()
        bucket = self.groups_dir / key_hash[:2]
        path = bucket / f"group_{key_hash}_{payload_hash}.parquet"
        try:
            installed = install_bytes_once(
                payload,
                path,
                surface=f"universal GCG group {group_key_str}",
            )
        except (ImmutableArtifactError, OSError) as exc:
            raise _checkpoint_error(str(exc), path) from exc
        matches: list[Path] = []
        key_prefix = f"group_{key_hash}_"
        try:
            candidates = list(bucket.iterdir())
        except OSError as exc:
            raise _checkpoint_error(
                f"cannot enumerate checkpoint bucket: {exc}", bucket
            ) from exc
        for candidate in candidates:
            if _is_known_installer_temp(candidate):
                continue
            if not candidate.name.startswith(key_prefix):
                continue
            match = _SHARD_RE.fullmatch(candidate.name)
            if match is None:
                raise _checkpoint_error(
                    f"unknown committed checkpoint filename {candidate.name!r}",
                    candidate,
                )
            matches.append(candidate)
        if len(matches) != 1:
            raise _checkpoint_error(
                f"expected exactly one committed payload for group {group_key_str!r}, "
                f"observed {[candidate.name for candidate in sorted(matches)]}",
                bucket,
            )
        self._load_shard(matches[0], expected_bucket=key_hash[:2])
        return installed


def warn_on_runtime_mismatch(
    store: UniversalGcgCheckpointStore,
    current_runtime_sha256: str,
) -> None:
    differing = sorted(store.observed_runtime_hashes - {current_runtime_sha256})
    if differing:
        logger.warning(
            "FASTER-GCG RESUME IS MIXING NUMERIC RUNTIMES: current={} restored={}. "
            "Reuse is allowed and auditable, but byte-identical parity is guaranteed "
            "only when runtime fingerprints match.",
            current_runtime_sha256,
            differing,
        )
