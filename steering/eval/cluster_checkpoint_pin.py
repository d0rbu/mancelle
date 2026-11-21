"""Checkpoint and artifact identity checks for clustered evaluation.

``cluster_feature_contract`` orchestrates these helpers to pin a numeric
checkpoint step and assemble checkpoint, statistics, SAE, and selection
identities. This module does not run the model or SAE, or establish training
provenance beyond the recorded configuration and artifact identities checked.

Checkpoint and vocabulary agreement
-----------------------------------
The model file is hashed and loaded through the same open file object on
CPU. Metadata payload bytes are checked against their stable file digest;
subsequent file-signature checks bound the load window. Persistent state-dict
vocabulary buffers are authoritative, including compiled-key normalization.
The configured mask and static clusters must reproduce that compact
vocabulary; a present metadata vocabulary mirror must also agree.

Statistics must belong to the configured feature directory, match its
extraction settings and SAE width, and reproduce the stored percentile-table
checksum. Requested and loaded SAE widths, hook, model, and activation
metadata are checked against the checkpoint and directory. These are
identity and compatibility checks, not numerical SAE validation.

Selection identity and validation
---------------------------------
Selection settings are reconstructed from checkpoint metadata. The manifest
must agree with that identity, mask, cluster semantics, percentile provenance,
and exact packed/selection shard domains. Compressed selection bodies are
rehashed against their declared inventory. A complete producer validation
certificate supplies the prior canonical tensor proof, segment counts, and
maximum selection width for those artifact identities. This checks recorded
settings, not which selection bytes were actually consumed during training.

The normal path can reuse packed content fingerprints only when cache schema,
digests, and exact link/target file signatures validate. Missing, stale, or
malformed caches are recomputed from source bytes. Forced shard revalidation
bypasses the cache and decodes packed and selection bodies to repeat canonical
representative, segment-count, and width checks. Optional cache writes are
best effort and independent of final selection approval;
``allow_packed_cache_write=False`` prevents writes, not source validation.

Accepted threat-model boundary
------------------------------
(a) Final checks recheck manifest signatures, shard namespaces, per-file
signatures, and the packed fingerprint resolution.

(b) Signature-preserving mutations, including forged or insufficiently
precise filesystem timestamps, can escape stat-based checks and cache
invalidation.

(c) Checks are sequential by path, not an atomic cross-file snapshot: an
earlier shard or manifest can change while a later item is checked, with no
second check of the earlier item. Existing tests call these schedules the
"Codex pass-5 finding"; reordering checks does not close this boundary.

OWNER SCOPE DECISION
--------------------
Inputs are expected to remain immutable during evaluation. The evaluation
dataloader independently enforces its own load-time artifact and canonical
validation checks; this preflight does not replace that backstop or make
concurrent filesystem mutation safe. A successful contract is an identity
and compatibility result within these boundaries, not a guarantee of all
data contents or downstream model behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from sae_lens import SAE

from errors import FeatureClustersError, FeatureVocabError
from nanochat.checkpoint_manager import (
    _extract_encdec_fields,
    find_last_step,
    get_checkpoint_dir,
)
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    assert_validation_packed_inventory,
    resolve_clustered_selection_validation,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig, parse_feature_delta
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.sae_embedding_init import coerce_sae_l0
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    assert_clustered_selection_shard_integrity,
    assert_feature_clusters_selection_metadata,
    assert_resume_feature_clusters_match,
    canonical_feature_clusters_descriptor,
)
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_mask import canonical_dict as canonical_feature_mask_dict
from steering.conditional.feature_vocab import META_KEY as FEATURE_VOCAB_META_KEY
from steering.conditional.feature_vocab import (
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
    feature_vocab_from_state_dict,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.splice_counterfactual import (
    PackedFingerprintResolution,
    assert_packed_row_fingerprint_resolution_current,
    resolve_packed_row_fingerprint_strict,
)
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.conditional.thresholding import ThresholdType

# Reuse the identity layer's private hashing and signature helpers.
# `_stat_signature` supplies the same lstat tuple for manifests and shards;
# `SaeIdentity` is the public return type of `_resolve_and_validate_sae`.
from steering.eval.cluster_config import (
    CLUSTERED_ENCDEC_TARGET_METHODS,
    CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS,
)
from steering.eval.cluster_contract_identity import (
    MAX_SAFE_SELECTED_FEATURE_WIDTH,
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    _assert_digest_stat_unchanged,
    _canonical_json,
    _sha256_json,
    _stat_signature,
    stream_hash_file_stable,
)
from steering.feature_paths import parse_feature_dir
from steering.immutable_io import is_sha256_hex
from steering.overlay_io import COMPRESSED_EXT, MANIFEST_FILENAME, glob_overlay_shards
from steering.packed_batch_io import (
    SelectionShard,
    compute_selection_hash,
    load_packed_row_shard_bytes,
    load_selection_shard,
)
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    resolve_neuronpedia_ids,
    resolve_sae,
)
from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics_and_checksum_from_bytes

# Local to this module (not imported from B1's cluster_contract_identity,
# whose own equivalent is a private implementation detail of a primitive —
# stream_hash_file_descriptor_bound — this module cannot reuse for the
# model file; see _hash_and_load_checkpoint_model).
_MODEL_HASH_CHUNK_SIZE = 8 * 1024 * 1024

# ``num_buckets`` is deliberately NOT in this set (Codex pass-2 residual
# finding 1, MEDIUM): it already has its own dedicated dual-layout
# normalization through ``_extract_encdec_fields`` below, which must be
# the single place that decides its value. Also
# listing it here let the RAW, un-normalized ``user_config["num_buckets"]``
# leak back into the load-bearing fingerprint a second time through the
# ``user_config`` sub-dict -- present for pretrain-layout metadata (nested
# under ``user_config``), absent for SFT-layout metadata (top-level only,
# so never copied into ``selected_user_config``) -- even when both
# checkpoints carry the exact same logical bucket count. That made
# ``_checkpoint_load_bearing_metadata``'s two layouts disagree on
# ``user_config`` alone, defeating the finding-4 normalization one field
# over. See ``test_checkpoint_load_bearing_metadata_parity_between_
# pretrain_and_sft_layouts``, which now asserts the two layouts' ENTIRE
# normalized dicts are equal (not a hand-picked subset of fields) so this
# class of gap cannot recur silently.
_CHECKPOINT_USER_CONFIG_KEYS = frozenset(
    {
        "decoder_segment_size",
        "feature_delta",
        "feature_prediction",
        "feature_selection",
        "negative_selection",
        "next_segment_conditioning_offset",
        "sae",
        "seed",
        "segment_size",
    }
)


def _mapping_get(mapping: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


@dataclass(frozen=True)
class CheckpointFilesPin:
    """Exact on-disk identity shared by decoder-only and encdec checkpoints."""

    checkpoint_dir: Path
    resolved_tag: str
    step: int
    model_file: StableFileDigest
    metadata_file: StableFileDigest

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "resolved_tag": self.resolved_tag,
            "step": self.step,
            "model_sha256": self.model_file.sha256,
            "metadata_sha256": self.metadata_file.sha256,
        }

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_path": str(self.model_file.path),
            "metadata_path": str(self.metadata_file.path),
            "resolved_step": self.step,
            "model_byte_length": self.model_file.byte_length,
            "metadata_byte_length": self.metadata_file.byte_length,
        }


@dataclass(frozen=True)
class CheckpointPin(CheckpointFilesPin):
    n_sae_features: int
    feature_vocab: FeatureVocab | None
    load_bearing_metadata_json: str

    def semantic_dict(self) -> dict[str, Any]:
        if self.feature_vocab is None:
            feature_vocab: dict[str, Any] = {
                "persistent": False,
                "n_sae_features": self.n_sae_features,
            }
        else:
            feature_vocab = {
                "persistent": True,
                "canonical": self.feature_vocab.canonical_dict(),
                "sha256": self.feature_vocab.fingerprint(),
            }
        return {
            **super().semantic_dict(),
            "n_sae_features": self.n_sae_features,
            "feature_vocab": feature_vocab,
            "load_bearing_metadata": json.loads(self.load_bearing_metadata_json),
        }


@dataclass(frozen=True)
class CheckpointSelectionContract:
    """The checkpoint's frozen selection and delta identity.

    Ported byte-identical from the oracle (no ``__post_init__`` there either
    — malformed-field validation happens earlier, while resolving each field
    from checkpoint metadata in ``_resolve_selection_identity``/
    ``_selection_contract``, not here).
    """

    positive_selection_json: str
    negative_selection_json: str
    num_buckets: int
    effective_positive_buckets: int
    preselection_seed: int
    selection_hash: str
    # SHA-256 of the semantic manifest after removing producer timestamp and
    # cluster source-path provenance.  This is safe to fingerprint across hosts.
    selection_manifest_sha256: str
    selection_shards_sha256: str
    packed_row_fingerprint: str
    feature_delta_json: str | None = field(default=None, compare=False)
    # Exact source-file bytes remain audit provenance, never semantic identity.
    selection_manifest_file_sha256: str | None = field(default=None, compare=False)

    def parsed_feature_delta(self) -> FeatureDeltaConfig | None:
        """The delta config as a validated object rather than a JSON string.

        The field itself stays canonical JSON, like its ``*_selection_json``
        siblings: this structure's fingerprint is a SHA-256 over canonical JSON,
        so byte-stable storage is what keeps identity stable. That is a storage
        concern, though, not a call-site concern -- consumers want the real
        type, and open-coding ``json.loads`` plus ``parse_feature_delta`` at
        each one both duplicates the validation and loses it from the signature.
        """
        if self.feature_delta_json is None:
            return None
        return parse_feature_delta(json.loads(self.feature_delta_json))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "positive_selection": json.loads(self.positive_selection_json),
            "negative_selection": json.loads(self.negative_selection_json),
            "num_buckets": self.num_buckets,
            "effective_positive_buckets": self.effective_positive_buckets,
            "preselection_seed": self.preselection_seed,
            "selection_hash": self.selection_hash,
            "selection_manifest_sha256": self.selection_manifest_sha256,
            "selection_shards_sha256": self.selection_shards_sha256,
            "packed_row_fingerprint": self.packed_row_fingerprint,
            # Deliberately always null: the delta CONFIG is replay detail, not
            # identity. Whether a selection is delta-weighted is already carried
            # in positive_selection_json's ``delta_weights``, so emitting the
            # config here would add no distinguishing power while migrating every
            # existing clustered artifact's fingerprint (measured: the real l22
            # contract moved 979a0874... -> 419ea3a8...). Retained on the
            # dataclass for reconstruction, mirroring
            # selection_manifest_file_sha256's provenance-not-identity role.
            "feature_delta": None,
        }

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "selection_manifest_file_sha256": self.selection_manifest_file_sha256,
        }


def _load_json_mapping(path: Path, *, surface: str) -> tuple[dict[str, Any], StableFileDigest]:
    """Load one JSON object with fused hash+read mutation detection.

    Hash ``path`` through ``stream_hash_file_stable``, then verify the
    subsequent payload read's SHA-256 and byte length against that digest.
    """
    digest = stream_hash_file_stable(path)
    try:
        payload = digest.path.read_bytes()
    except OSError as exc:
        raise ClusterFeatureContractError(f"could not read {surface} {path}: {exc}") from exc
    if len(payload) != digest.byte_length or hashlib.sha256(payload).hexdigest() != digest.sha256:
        raise ClusterFeatureContractError(f"{surface} changed while it was being loaded: {path}")
    try:
        raw = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ClusterFeatureContractError(f"could not read {surface} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ClusterFeatureContractError(f"{surface} must contain a JSON object")
    return cast(dict[str, Any], raw), digest


def _checkpoint_load_bearing_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    user_config = metadata.get("user_config")
    if not isinstance(user_config, Mapping):
        raise ClusterFeatureContractError(
            "clustered checkpoint metadata requires a user_config mapping"
        )
    selected_user_config = {
        key: user_config[key] for key in sorted(_CHECKPOINT_USER_CONFIG_KEYS) if key in user_config
    }
    # Codex fix batch finding 4: _extract_encdec_fields normalizes BOTH the
    # pretrain layout (user_config.encoder/user_config.cross_attention) and
    # the SFT layout (top-level encoder_config/xattn_config/num_buckets).
    # metadata may be any Mapping (e.g. the
    # checkpoint's parsed JSON); _extract_encdec_fields expects a dict.
    encdec_fields = _extract_encdec_fields(
        metadata if isinstance(metadata, dict) else dict(metadata)
    )
    return {
        "model_config": metadata.get("model_config"),
        "encoder_config": encdec_fields.get("encoder"),
        "xattn_config": encdec_fields.get("cross_attention"),
        "num_buckets": encdec_fields.get("num_buckets"),
        "feature_mask_canonical": metadata.get("feature_mask_canonical"),
        "feature_vocab": metadata.get(FEATURE_VOCAB_META_KEY),
        "feature_clusters_descriptor": metadata.get("feature_clusters_descriptor"),
        "feature_clusters_canonical": metadata.get("feature_clusters_canonical"),
        "user_config": selected_user_config,
    }


def resolve_eval_checkpoint_directory(
    *,
    checkpoint_dir: str | Path | None,
    checkpoint_source: str | None,
    checkpoint_model_tag: str | None,
) -> tuple[Path, str]:
    """Resolve an eval checkpoint directory through the public path utility.

    Callers may supply a direct directory for tests/offline artifacts, or the
    same ``source``/``model_tag`` pair accepted by runtime loading. Mixing the
    two forms is rejected so preflight cannot inspect one checkpoint while the
    runtime later loads another.
    """

    has_direct = checkpoint_dir is not None
    has_runtime_ref = checkpoint_source is not None or checkpoint_model_tag is not None
    if has_direct == has_runtime_ref:
        raise ClusterFeatureContractError(
            "cluster checkpoint preflight requires exactly one of checkpoint_dir "
            "or checkpoint_source/checkpoint_model_tag"
        )
    if has_direct:
        assert checkpoint_dir is not None
        directory = Path(checkpoint_dir).expanduser().resolve(strict=True)
        return directory, directory.name
    if not isinstance(checkpoint_source, str) or not checkpoint_source:
        raise ClusterFeatureContractError(
            "checkpoint_source must be a non-empty string when checkpoint_dir is null"
        )
    if not isinstance(checkpoint_model_tag, str) or not checkpoint_model_tag:
        raise ClusterFeatureContractError(
            "checkpoint_model_tag must be explicit for clustered evaluation"
        )
    try:
        resolved = get_checkpoint_dir(
            checkpoint_source,
            model_tag=checkpoint_model_tag,
        )
        directory = Path(resolved).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ClusterFeatureContractError(
            "could not resolve clustered checkpoint from "
            f"source={checkpoint_source!r}, model_tag={checkpoint_model_tag!r}: {exc}"
        ) from exc
    return directory, checkpoint_model_tag


def _fd_bound_stat_signature(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """``(mode, dev, ino, size, mtime_ns, ctime_ns)`` from an already-obtained
    ``lstat``/``fstat`` result. Field order mirrors ``StableFileDigest.
    stat_signature`` and ``steering.immutable_io._fd_bound_signature``.
    """
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _hash_and_load_checkpoint_model(path: Path) -> tuple[StableFileDigest, Any]:
    """Hash and load the checkpoint model file from exactly ONE file object.

    Keep the descriptor open between hashing and CPU loading so the load
    cannot reopen a substituted pathname. The shared stable-hash helper
    closes its descriptor before returning; ``mmap=True`` requires a path
    rather than this open file object.
    """
    resolved = path.expanduser().resolve(strict=True)
    before_result = resolved.lstat()
    if not stat.S_ISREG(before_result.st_mode):
        raise ClusterFeatureContractError(
            f"checkpoint model file is not a regular file: {resolved}"
        )
    before = _fd_bound_stat_signature(before_result)

    digest = hashlib.sha256()
    byte_length = 0
    try:
        handle = resolved.open("rb")
    except OSError as exc:
        raise ClusterFeatureContractError(
            f"could not open checkpoint model file {resolved}: {exc}"
        ) from exc
    try:
        try:
            opened = _fd_bound_stat_signature(os.fstat(handle.fileno()))
        except OSError as exc:
            raise ClusterFeatureContractError(
                f"could not hash checkpoint model file {resolved}: {exc}"
            ) from exc
        if opened != before:
            # Same wording as the during-read/after-read checks below: a
            # caller does not need to know WHICH of the three checkpoints
            # tripped, only that the file changed somewhere in the hash
            # window — mirrors B1's own _STREAM_HASH_ERROR_MESSAGES, which
            # unifies "changed_before_open" and "changed_during_read" onto
            # the identical text for the same reason.
            raise ClusterFeatureContractError(
                f"checkpoint model file changed while it was being hashed: {resolved}"
            )

        while True:
            try:
                chunk = handle.read(_MODEL_HASH_CHUNK_SIZE)
            except OSError as exc:
                raise ClusterFeatureContractError(
                    f"could not hash checkpoint model file {resolved}: {exc}"
                ) from exc
            if not chunk:
                break
            byte_length += len(chunk)
            digest.update(chunk)

        try:
            after_read = _fd_bound_stat_signature(os.fstat(handle.fileno()))
        except OSError as exc:
            raise ClusterFeatureContractError(
                f"could not hash checkpoint model file {resolved}: {exc}"
            ) from exc
        if after_read != opened or byte_length != after_read[3]:
            raise ClusterFeatureContractError(
                f"checkpoint model file changed while it was being hashed: {resolved}"
            )

        handle.seek(0)
        try:
            model_data = t.load(handle, map_location="cpu", weights_only=True, mmap=False)
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
            # pickle.UnpicklingError: eager (mmap=False) loading reaches
            # torch's weights_only unpickler for malformed input that
            # mmap=True's zip-format precheck used to reject earlier as a
            # plain RuntimeError -- same "not a valid checkpoint" failure
            # class, different exception type for the same garbage input.
            raise ClusterFeatureContractError(
                f"could not safely load checkpoint state dict {resolved}: {exc}"
            ) from exc
    finally:
        handle.close()

    model_digest = StableFileDigest(
        path=resolved,
        sha256=digest.hexdigest(),
        byte_length=byte_length,
        stat_signature=before,
    )
    return model_digest, model_data


def _assert_vocab_buffers_not_bool_dtype(model_data: Mapping[str, Any]) -> None:
    """Reject boolean vocabulary buffers before integer coercion can hide them."""
    for key in (VOCAB_IDS_KEY, VOCAB_N_SAE_KEY):
        buf = model_data.get(key)
        if isinstance(buf, t.Tensor) and buf.dtype is t.bool:
            raise ClusterFeatureContractError(
                f"corrupt checkpoint: {key!r} must be an integer tensor, got torch.bool. "
                "A boolean buffer's dtype is neither floating-point nor complex, so it "
                "silently passes the dependency's own dtype gate and int(...) coerces "
                "True/False to 1/0 downstream — producing a different, plausible-looking "
                "vocabulary instead of raising."
            )


def resolve_checkpoint_files_pin(
    checkpoint_dir: str | Path,
    *,
    step: int | None,
    resolved_tag: str | None = None,
) -> tuple[CheckpointFilesPin, dict[str, Any], Mapping[str, Any]]:
    """Freeze a numeric checkpoint step and bind its exact model/meta bytes."""

    directory = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ClusterFeatureContractError(f"checkpoint_dir is not a directory: {directory}")
    resolved_step = find_last_step(str(directory)) if step is None else step
    if type(resolved_step) is not int or resolved_step < 0:
        raise ClusterFeatureContractError(
            f"checkpoint step must be a non-negative strict int, got {resolved_step!r}"
        )
    model_path = directory / f"model_{resolved_step:06d}.pt"
    metadata_path = directory / f"meta_{resolved_step:06d}.json"
    metadata, metadata_digest = _load_json_mapping(metadata_path, surface="checkpoint metadata")
    model_digest, model_data = _hash_and_load_checkpoint_model(model_path)
    # Cold rechecks, run *after* the model hash-and-load. The model's own
    # recheck bounds the window from when _hash_and_load_checkpoint_model
    # closed its descriptor to now (a later on-disk mutation than what it
    # already verified internally). Rechecking metadata after the model
    # load also covers that load window.
    _assert_digest_stat_unchanged(model_digest)
    _assert_digest_stat_unchanged(metadata_digest)
    if not isinstance(model_data, Mapping):
        raise ClusterFeatureContractError(
            f"checkpoint {model_path} did not contain a state-dict mapping"
        )
    pin = CheckpointFilesPin(
        checkpoint_dir=directory,
        resolved_tag=resolved_tag or directory.name,
        step=resolved_step,
        model_file=model_digest,
        metadata_file=metadata_digest,
    )
    return pin, metadata, model_data


def assert_checkpoint_files_pin_unchanged(pin: CheckpointFilesPin) -> None:
    """Reject checkpoint file-identity or content drift across a load window.

    Matching the pathname, digest, byte length, and six-field ``lstat``
    signature lets a caller infer that a pathname-based loader observed the
    pinned files throughout its bracket. This is an inference from stable file
    identity, not descriptor-bound byte injection into that loader.

    ``st_ctime_ns`` is the load-bearing field of that signature; the rest are
    cheap defence in depth and must not be mistaken for the guarantee.
    ``st_ino`` does not discriminate on its own -- filesystems reuse inode
    numbers, and an A->B->A rename round trip was measured reusing the original
    number on the first attempt. ``st_mtime_ns`` is forgeable: ``shutil.copy2``
    carries it across a restoring rename, and ``os.utime`` restores it after an
    in-place rewrite. ``st_mode`` catches only symlink substitution. ``ctime``
    survives both evasions because the kernel bumps it on every content write
    and every rename into the path, and POSIX exposes no setter for it
    (``utime``/``utimensat`` move atime and mtime, and bump ctime doing so).
    ``tests/steering/test_cluster_checkpoint_pin.py`` pins this directly.

    Two residuals, neither closed here: an adversary writing below the VFS
    (raw device access, a crafted FUSE or network mount) can forge ``ctime``;
    and a filesystem whose ``ctime`` granularity cannot separate the two writes
    degrades the check. On ext4 the measured separation was ~511us.
    """

    for surface, expected in (
        ("model", pin.model_file),
        ("metadata", pin.metadata_file),
    ):
        actual = stream_hash_file_stable(expected.path)
        if (
            actual.path != expected.path
            or actual.sha256 != expected.sha256
            or actual.byte_length != expected.byte_length
            or actual.stat_signature != expected.stat_signature
        ):
            raise ClusterFeatureContractError(
                f"checkpoint {surface} bytes changed after they were pinned: {expected.path}"
            )


def resolve_checkpoint_pin(
    checkpoint_dir: str | Path,
    *,
    step: int | None,
    resolved_tag: str | None = None,
) -> tuple[CheckpointPin, dict[str, Any], Mapping[str, Any]]:
    """Freeze an encdec checkpoint and recover its feature vocabulary."""

    files_pin, metadata, model_data = resolve_checkpoint_files_pin(
        checkpoint_dir,
        step=step,
        resolved_tag=resolved_tag,
    )
    # Normalize torch.compile's state-dict prefix before recovering vocab.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    _assert_vocab_buffers_not_bool_dtype(model_data)
    try:
        n_sae_features, vocab = feature_vocab_from_state_dict(model_data)
    except FeatureVocabError as exc:
        raise ClusterFeatureContractError(f"invalid checkpoint feature vocabulary: {exc}") from exc
    if n_sae_features > MAX_SAFE_SELECTED_FEATURE_WIDTH:
        raise ClusterFeatureContractError(
            "clustered evaluation currently stores selected feature IDs in uint16; "
            f"original SAE width {n_sae_features} exceeds {MAX_SAFE_SELECTED_FEATURE_WIDTH}"
        )
    pin = CheckpointPin(
        checkpoint_dir=files_pin.checkpoint_dir,
        resolved_tag=files_pin.resolved_tag,
        step=files_pin.step,
        model_file=files_pin.model_file,
        metadata_file=files_pin.metadata_file,
        n_sae_features=n_sae_features,
        feature_vocab=vocab,
        load_bearing_metadata_json=_canonical_json(_checkpoint_load_bearing_metadata(metadata)),
    )
    return pin, metadata, model_data


def _assert_checkpoint_agreement(
    *,
    checkpoint: CheckpointPin,
    metadata: Mapping[str, Any],
    feature_mask: FeatureMask | None,
    feature_clusters: ResolvedFeatureClusters | None,
) -> None:
    if (
        feature_clusters is not None
        and checkpoint.n_sae_features != feature_clusters.n_sae_features
    ):
        raise ClusterFeatureContractError(
            "checkpoint original SAE width does not match the configured "
            "feature-cluster descriptor: "
            f"checkpoint={checkpoint.n_sae_features}, "
            f"clusters={feature_clusters.n_sae_features}"
        )
    configured_mask = canonical_feature_mask_dict(feature_mask)
    if metadata.get("feature_mask_canonical") != configured_mask:
        raise ClusterFeatureContractError(
            "configured feature mask does not exactly match checkpoint "
            f"feature_mask_canonical: configured={configured_mask!r}, "
            f"checkpoint={metadata.get('feature_mask_canonical')!r}"
        )

    # Named deviation from the oracle (Codex fix batch finding 1, CRITICAL —
    # "main-drift adaptation"; see task-b2a-report.md's "## Codex fix batch"
    # for the #759 evidence this is grounded in). The oracle predates PR
    # #759/#760's effective-feature-vocabulary training and compares the
    # checkpoint's persistent vocabulary against ``FeatureVocab.from_mask``
    # alone — mask-aware but cluster-BLIND. On current main, a clustered
    # model's encoder is sized by ``effective_feature_vocab`` (mask AND
    # static-cluster-consolidation aware; delegates to ``FeatureVocab.
    # from_mask`` only when clusters are absent — see
    # ``steering/conditional/effective_feature_vocab.py``), so the oracle's
    # comparison REJECTS every valid current clustered checkpoint (whose
    # persistent vocab is narrower than mask-only, since consolidated-away
    # rows carry no embedding) and ACCEPTS an obsolete pre-#759 checkpoint
    # whose vocab happens to be the wider mask-only shape. Rewired to derive
    # ``expected_vocab`` through the SAME ``effective_feature_vocab``
    # production path every other cluster-aware consumer already uses
    # (``context_cache_io.clustered_context_cache_metadata``,
    # ``FeaturePredictionTargetLayout.build``, ``sae_embedding_init``).
    feature_universe: FeatureClusterConsolidation | FeatureMask | None
    if feature_clusters is None:
        feature_universe = feature_mask
    else:
        feature_universe = FeatureClusterConsolidation.build(feature_clusters, feature_mask)
    expected_vocab = effective_feature_vocab(feature_universe, checkpoint.n_sae_features)
    if checkpoint.feature_vocab != expected_vocab:
        if feature_clusters is not None and checkpoint.feature_vocab == FeatureVocab.from_mask(
            feature_mask, checkpoint.n_sae_features
        ):
            raise ClusterFeatureContractError(
                "checkpoint carries a mask-only feature vocabulary that ignores the "
                "configured static feature-cluster consolidation: this checkpoint "
                "predates effective-vocabulary training (#759); retrain or use a "
                "post-#759 checkpoint"
            )
        raise ClusterFeatureContractError(
            "configured feature mask and authoritative state-dict FeatureVocab do not agree"
        )
    # Codex fix batch finding 5 (MEDIUM): key-PRESENCE semantics, not just
    # value truthiness. ``metadata.get(FEATURE_VOCAB_META_KEY)`` returns
    # ``None`` for BOTH "the key is absent" (a legitimately optional mirror)
    # AND "the key is present with an explicit JSON ``null``" — the latter
    # is a WRITER ASSERTING "no vocab", which directly contradicts a
    # compact (non-``None``) persistent vocabulary. The oracle's ``is not
    # None`` guard could not tell these apart, so an explicit-null mirror
    # silently passed agreement no matter what the persistent vocab was.
    # ``key in metadata`` distinguishes them: absence still skips the check
    # (unchanged); a PRESENT value — including ``None`` — is now compared.
    if FEATURE_VOCAB_META_KEY in metadata:
        metadata_vocab = metadata[FEATURE_VOCAB_META_KEY]
        expected_metadata_vocab = (
            None if checkpoint.feature_vocab is None else checkpoint.feature_vocab.metadata_dict()
        )
        if metadata_vocab != expected_metadata_vocab:
            raise ClusterFeatureContractError(
                "checkpoint feature-vocabulary metadata mirror does not match its "
                "authoritative state-dict buffers"
            )
    try:
        assert_resume_feature_clusters_match(
            feature_clusters,
            metadata,
            checkpoint_dir=str(checkpoint.checkpoint_dir),
        )
    except FeatureClustersError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc


def _load_statistics_snapshot(
    path: Path, *, surface: str = "feature statistics"
) -> tuple[Any, str | None, t.Tensor | None, StableFileDigest]:
    """Fused hash+read+parse for one statistics file.

    Codex fix batch finding 3 (HIGH): applies the exact same "FUSE
    check+read" shape as ``_load_json_mapping`` (hash ``path`` exactly once,
    verify the one subsequent payload read's content against that digest)
    to the statistics file, then parses tensors, metadata, the stored
    percentile checksum, AND the raw (pre-collapse) ``feature_ids`` tensor
    ALL from that single verified ``payload`` via ``stats_io.
    load_statistics_and_checksum_from_bytes``. This closes the oracle's
    two-more-reopens gap: ``load_statistics(path)`` independently opened
    the safetensors body + header, and a SEPARATE ``load_percentile_
    checksum(path)`` reopened the header again *after* the only stability
    check (``_assert_digest_stat_unchanged``) — a window in which an
    A-digest/B-content or A-content/B-checksum swap at either former reopen
    point would have gone undetected.
    """
    digest = stream_hash_file_stable(path)
    try:
        payload = digest.path.read_bytes()
    except OSError as exc:
        raise ClusterFeatureContractError(f"could not read {surface} {path}: {exc}") from exc
    if len(payload) != digest.byte_length or hashlib.sha256(payload).hexdigest() != digest.sha256:
        raise ClusterFeatureContractError(f"{surface} changed while it was being loaded: {path}")
    stats, checksum, raw_feature_ids = load_statistics_and_checksum_from_bytes(payload)
    return stats, checksum, raw_feature_ids, digest


def _assert_raw_feature_ids_valid(raw_feature_ids: t.Tensor | None, *, n_sae_features: int) -> None:
    """Validate dtype/range/uniqueness on the RAW (pre-collapse) safetensors
    ``feature_ids`` tensor.

    Codex fix batch finding 6 (MEDIUM): ``_reconstruct_statistics`` collapses
    ``feature_ids`` into a ``dict[str, PerFeatureStat]`` keyed by
    ``str(feature_id)``, where a LATER, larger row silently wins any
    tensor-level duplicate — so a genuine duplicate raw ID is invisible by
    the time ``stats.per_feature_stats`` exists, and invisible to the
    percentile-table checksum too (computed from that same already-collapsed
    dict). This runs on the tensor ``_load_statistics_snapshot`` exposes
    BEFORE that collapse, from the same digest-verified snapshot.
    """
    if raw_feature_ids is None:
        return
    if raw_feature_ids.ndim != 1:
        raise ClusterFeatureContractError(
            "statistics raw feature_ids tensor must be one-dimensional, got "
            f"shape={tuple(raw_feature_ids.shape)}"
        )
    if (
        raw_feature_ids.dtype is t.bool
        or raw_feature_ids.is_floating_point()
        or raw_feature_ids.is_complex()
    ):
        raise ClusterFeatureContractError(
            "statistics raw feature_ids tensor must use an integer dtype, got "
            f"{raw_feature_ids.dtype}"
        )
    ids = raw_feature_ids.tolist()
    for feature_id in ids:
        if not (0 <= feature_id < n_sae_features):
            raise ClusterFeatureContractError(
                f"statistics raw feature_ids tensor contains ID {feature_id} out of "
                f"range [0, {n_sae_features})"
            )
    if len(set(ids)) != len(ids):
        raise ClusterFeatureContractError(
            "statistics raw feature_ids tensor contains a duplicate feature ID "
            "(pre-collapse; the safetensors row order lets a later, larger row "
            "silently win in the collapsed per_feature_stats dict, concealing it "
            "from every checksum-based check downstream)"
        )


def _statistics_identity(
    *,
    feature_dir: Path,
    statistics_path: Path,
    n_sae_features: int,
) -> tuple[FeatureStatisticsIdentity, FeatureDirectoryIdentity, Any]:
    if statistics_path.parent.resolve() != feature_dir.resolve():
        raise ClusterFeatureContractError(
            "feature_statistics_path must be the statistics artifact inside the "
            f"configured feature_dir; got {statistics_path} vs {feature_dir}"
        )
    try:
        stats, stored_checksum, raw_feature_ids, stats_digest = _load_statistics_snapshot(
            statistics_path
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ClusterFeatureContractError(
            f"could not load feature statistics {statistics_path}: {exc}"
        ) from exc
    _assert_digest_stat_unchanged(stats_digest)
    if stats.metadata.n_sae_features != n_sae_features:
        raise ClusterFeatureContractError(
            "statistics original SAE width does not match checkpoint: "
            f"{stats.metadata.n_sae_features} != {n_sae_features}"
        )
    _assert_raw_feature_ids_valid(raw_feature_ids, n_sae_features=n_sae_features)
    components = parse_feature_dir(feature_dir)
    if components.segment_mode is not SegmentMode.fixed_tokens:
        raise ClusterFeatureContractError(
            f"clustered evaluation supports fixed_tokens feature directories only (segment_mode={components.segment_mode.value!r})"
        )
    directory_identity = FeatureDirectoryIdentity.from_components(components)
    expected_width = SAE_WIDTH_TO_NUM_FEATURES[components.width]
    mismatches: list[str] = []
    if expected_width != n_sae_features:
        mismatches.append(f"path width={expected_width}, checkpoint width={n_sae_features}")
    if stats.metadata.segment_size != components.segment_size:
        mismatches.append(
            f"stats segment_size={stats.metadata.segment_size}, path={components.segment_size}"
        )
    if stats.metadata.layer != components.layer:
        mismatches.append(f"stats layer={stats.metadata.layer}, path={components.layer}")
    if stats.metadata.pooling_mode != components.pooling.value:
        mismatches.append(
            f"stats pooling={stats.metadata.pooling_mode!r}, path={components.pooling.value!r}"
        )
    if stats.metadata.segment_mode is not components.segment_mode:
        mismatches.append(
            "stats segment_mode="
            f"{stats.metadata.segment_mode.value!r}, "
            f"path={components.segment_mode.value!r}"
        )
    if bool(stats.ignore_bos_features) != components.ignore_bos_features:
        mismatches.append("stats ignore_bos_features does not match the feature-directory leaf")
    if mismatches:
        raise ClusterFeatureContractError(
            "feature-directory/statistics identity mismatch: " + "; ".join(mismatches)
        )
    try:
        percentile_table = build_percentile_table(
            stats.per_feature_stats,
            n_sae_features,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ClusterFeatureContractError(
            f"could not rebuild percentile table from statistics: {exc}"
        ) from exc
    computed_checksum = compute_percentile_checksum(percentile_table)
    if stored_checksum is None:
        raise ClusterFeatureContractError(
            "clustered evaluation requires feature statistics with a stored "
            "percentile_checksum from the selection producer"
        )
    if stored_checksum != computed_checksum:
        raise ClusterFeatureContractError(
            "stored feature-statistics percentile_checksum does not match the recomputed table"
        )
    domain: list[int] = []
    for raw_feature_id in stats.per_feature_stats:
        try:
            feature_id = int(raw_feature_id)
        except (TypeError, ValueError) as exc:
            raise ClusterFeatureContractError(
                f"statistics feature ID {raw_feature_id!r} is not an integer"
            ) from exc
        if str(feature_id) != str(raw_feature_id):
            raise ClusterFeatureContractError(
                f"statistics feature ID {raw_feature_id!r} is not canonical decimal form"
            )
        domain.append(feature_id)
    feature_id_domain = tuple(sorted(domain))
    if len(feature_id_domain) != len(set(feature_id_domain)):
        raise ClusterFeatureContractError("statistics feature-ID domain contains duplicates")
    identity = FeatureStatisticsIdentity(
        file_sha256=stats_digest.sha256,
        percentile_table_checksum=computed_checksum,
        n_sae_features=n_sae_features,
        feature_id_domain=feature_id_domain,
    )
    return identity, directory_identity, stats


def _default_sae_loader(resolved: ResolvedSAE) -> Any:
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cpu",
    )
    return sae[0] if isinstance(sae, tuple) else sae


def _model_names_agree(loaded: str, resolved: str) -> bool:
    """Compare an SAE-reported model name against the checkpoint's model token.

    The two sides use different conventions for the same model: SAELens
    gemma-scope-2 configs report a fully-qualified HuggingFace id
    (``google/gemma-3-4b-pt``), while the checkpoint records the bare
    feature-path token (``gemma-3-4b-pt``) that ``sanitize_model_name``
    round-trips through the ``m_*`` directory component. Agreement therefore
    ignores an optional ``org/`` prefix on either side.

    This deliberately cannot distinguish ``google/gemma-3-4b-pt`` from a
    hypothetical ``other-org/gemma-3-4b-pt``: the checkpoint simply does not
    record an org, so no stricter comparison is available from the data. Every
    other SAE coordinate (release, sae_id, hook_name, layer, width, loaded
    width, and the Neuronpedia id when published) is still compared exactly.
    """

    return loaded.rpartition("/")[2] == resolved.rpartition("/")[2]


def _resolve_and_validate_sae(
    *,
    metadata: Mapping[str, Any],
    feature_directory: FeatureDirectoryIdentity,
    n_sae_features: int,
    sae_loader: Callable[[ResolvedSAE], Any],
) -> SaeIdentity:
    """Resolve the checkpoint's SAE config and validate the loaded SAE against it.

    Two width checks, never against each other directly but transitively
    equal whenever both pass: the checkpoint-*requested* width
    (``SAE_WIDTH_TO_NUM_FEATURES[width]``, from ``user_config.sae.width``,
    checked before ``sae_loader`` is even called) and the actually *loaded*
    SAELens decoder's row count (``matrix.shape[0]``, known only after
    ``sae_loader`` returns) are each pinned against the checkpoint's already-
    resolved ``n_sae_features``.
    """
    user_config = metadata.get("user_config")
    if not isinstance(user_config, Mapping):
        raise ClusterFeatureContractError("checkpoint user_config is missing")
    sae_cfg = user_config.get("sae")
    if not isinstance(sae_cfg, Mapping):
        raise ClusterFeatureContractError("checkpoint user_config.sae must be a mapping")
    try:
        release = SAERelease(str(sae_cfg["release"]))
        model = str(sae_cfg["model"])
        activation_type = ActivationType(str(sae_cfg["activation_type"]))
        # Codex combined-review finding 5 (MEDIUM): plain `int(...)` accepts
        # booleans (`isinstance(True, int)`), numeric strings ("8" -> 8), and
        # silently TRUNCATES floats (8.9 -> 8) -- a checkpoint layer of 8.9
        # became 8 and passed both the feature-directory and hook agreement
        # checks below. `type(...) is int` is the strict-int idiom already
        # used everywhere else in this module (resolved_step, seed,
        # num_buckets); it excludes bool (`type(True) is bool`, not `int`)
        # and any non-strict-int input by construction.
        raw_layer = sae_cfg["layer"]
        if type(raw_layer) is not int or raw_layer < 0:
            raise ValueError(f"sae.layer must be a non-negative strict int, got {raw_layer!r}")
        layer = raw_layer
        width = SAEWidth(str(sae_cfg["width"]))
        l0 = coerce_sae_l0(release, sae_cfg.get("l0"))
        resolved = resolve_sae(
            model=model,
            release=release,
            activation_type=activation_type,
            layer=layer,
            width=width,
            l0=l0,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ClusterFeatureContractError(f"invalid checkpoint SAE config: {exc}") from exc
    if (
        # The feature-directory ``m_*`` token is always bare, while the checkpoint
        # SAE block may record a fully-qualified id; compare prefix-insensitively
        # for the same reason as every other model-id comparison in this module.
        not _model_names_agree(model, feature_directory.model)
        or layer != feature_directory.layer
        or activation_type.value != feature_directory.activation_type
        or width.value != feature_directory.width
    ):
        raise ClusterFeatureContractError(
            "checkpoint SAE config does not match the parsed feature-directory identity"
        )
    if SAE_WIDTH_TO_NUM_FEATURES[width] != n_sae_features:
        raise ClusterFeatureContractError(
            "checkpoint SAE configured width does not match original checkpoint width"
        )
    try:
        sae = sae_loader(resolved)
    except Exception as exc:
        raise ClusterFeatureContractError(
            f"could not load SAE {resolved.release}/{resolved.sae_id} on CPU: {exc}"
        ) from exc
    matrix = getattr(sae, "W_dec", None)
    if not isinstance(matrix, t.Tensor) or matrix.ndim != 2:
        raise ClusterFeatureContractError("loaded SAE does not expose a rank-2 W_dec tensor")
    # Codex combined-review finding 2 (HIGH): assert CPU placement of the
    # loaded weights directly rather than trusting sae_loader's own device
    # kwarg. _default_sae_loader always requests device="cpu", but this
    # function accepts an arbitrary caller-supplied sae_loader, and CPU
    # loading is contractual for this preflight check, which does not run
    # the transformer forward pass.
    if matrix.device.type != "cpu":
        raise ClusterFeatureContractError(
            "loaded SAE W_dec must be a CPU tensor for this preflight check, "
            f"got device={matrix.device}"
        )
    dimensions = tuple(int(value) for value in matrix.shape)
    loaded_width = dimensions[0]
    if loaded_width != n_sae_features:
        raise ClusterFeatureContractError(
            "loaded SAELens W_dec must store one decoder row per original SAE "
            f"feature; expected shape[0]={n_sae_features}, got {dimensions}"
        )
    loaded_cfg = getattr(sae, "cfg", None)
    loaded_metadata = _mapping_get(loaded_cfg, "metadata", None)
    loaded_d_in = _mapping_get(loaded_cfg, "d_in", None)
    if loaded_d_in is not None and (
        type(loaded_d_in) is not int or loaded_d_in <= 0 or dimensions[1] != loaded_d_in
    ):
        raise ClusterFeatureContractError(
            "loaded SAE W_dec input dimension does not match sae.cfg.d_in: "
            f"shape={dimensions}, d_in={loaded_d_in!r}"
        )
    # Codex combined-review finding 2 (HIGH): d_sae/n_features cross-check,
    # when the loaded cfg exposes either name (SAELens has used both across
    # versions — see axbench_base/extraction.py's _infer_n_sae_features).
    # Independent of the W_dec-shape check above: a cfg that LIES about its
    # own tensor shape is a genuine "loaded identity disagrees" case a shape
    # check alone cannot see (cfg is metadata, not derived from the tensor).
    loaded_d_sae = _mapping_get(loaded_cfg, "d_sae", None)
    if loaded_d_sae is None:
        loaded_d_sae = _mapping_get(loaded_cfg, "n_features", None)
    if loaded_d_sae is not None and (
        type(loaded_d_sae) is not int or loaded_d_sae != n_sae_features
    ):
        raise ClusterFeatureContractError(
            "loaded SAE cfg d_sae/n_features does not match original checkpoint "
            f"width: cfg={loaded_d_sae!r}, expected={n_sae_features}"
        )

    # Codex combined-review finding 2 (HIGH) — hook_name: the original port's
    # lookup used cfg.metadata as the fallback DEFAULT of one combined read
    # (``_mapping_get(_mapping_get(loaded_cfg, "metadata", loaded_cfg),
    # "hook_name", _mapping_get(loaded_cfg, "hook_name", None))``), which
    # silently gave cfg.metadata.hook_name PRECEDENCE over the direct
    # cfg.hook_name whenever metadata was present — backwards from the
    # actual runtime accessor (axbench_base/extraction.py's _sae_hook_name;
    # read, not imported — that accessor requires a dict-like cfg.metadata
    # via ``.get``/``[]`` and cannot read a plain attribute-holding object,
    # which real SAELens SAEMetadata supports but this module's own
    # ``_mapping_get`` handles more broadly; its own docstring states "The
    # direct attribute takes precedence when both are present", the
    # precedence this now matches). A loaded SAE with a WRONG direct hook
    # and a correct metadata hook passed this preflight while runtime
    # extraction, which reads the direct attribute first, would have used
    # the wrong hook. Read both values separately first so a genuine
    # disagreement between them is itself rejected (not just silently
    # resolved by precedence), THEN apply the runtime's own direct-wins
    # precedence.
    direct_hook = _mapping_get(loaded_cfg, "hook_name", None)
    metadata_hook = _mapping_get(loaded_metadata, "hook_name", None)
    if direct_hook and metadata_hook and direct_hook != metadata_hook:
        raise ClusterFeatureContractError(
            f"loaded SAE cfg.hook_name {direct_hook!r} disagrees with "
            f"cfg.metadata.hook_name {metadata_hook!r}"
        )
    loaded_hook = direct_hook or metadata_hook
    if not isinstance(loaded_hook, str) or not loaded_hook:
        raise ClusterFeatureContractError(
            "loaded SAE config does not expose the resolved hook_name"
        )
    if loaded_hook != resolved.hook_name:
        raise ClusterFeatureContractError(
            f"loaded SAE hook {loaded_hook!r} does not match resolved hook {resolved.hook_name!r}"
        )

    # Codex combined-review finding 2 (HIGH) — release/Neuronpedia identity,
    # validated when available (SAELens 6.37+ drops cfg.release in favor of
    # cfg.metadata["neuronpedia_id"]). The two representations are different namespaces/
    # formats (a SAELens release string vs. "{model}/{source}"), so each is
    # validated against the resolver independently rather than against each
    # other.
    direct_release = _mapping_get(loaded_cfg, "release", None)
    if direct_release and direct_release != resolved.release:
        raise ClusterFeatureContractError(
            f"loaded SAE cfg.release {direct_release!r} does not match "
            f"resolved release {resolved.release!r}"
        )
    metadata_neuronpedia_id = _mapping_get(loaded_metadata, "neuronpedia_id", None)
    if metadata_neuronpedia_id:
        try:
            expected_model_id, expected_source = resolve_neuronpedia_ids(
                model=model,
                release=release,
                activation_type=activation_type,
                layer=layer,
                width=width,
                l0=l0,
            )
        except ValueError:
            pass  # not published for this combo; cannot validate this way
        else:
            expected_neuronpedia_id = f"{expected_model_id}/{expected_source}"
            if metadata_neuronpedia_id != expected_neuronpedia_id:
                raise ClusterFeatureContractError(
                    "loaded SAE cfg.metadata.neuronpedia_id "
                    f"{metadata_neuronpedia_id!r} does not match resolved "
                    f"Neuronpedia id {expected_neuronpedia_id!r}"
                )

    # Codex combined-review finding 2 (HIGH) — model identity, when
    # available, same direct-wins/agreement treatment as hook_name.
    #
    # Compared through `_model_names_agree`: gemma-scope-2 SAEs report the
    # fully-qualified HF id ("google/gemma-3-4b-pt") while the checkpoint's
    # own `sae_cfg["model"]` carries the bare feature-path token
    # ("gemma-3-4b-pt"), the form `sanitize_model_name` round-trips. GemmaScope
    # v1 reports the bare name on both sides, which is why an exact comparison
    # held until gemma-scope-2.
    direct_model_name = _mapping_get(loaded_cfg, "model_name", None)
    metadata_model_name = _mapping_get(loaded_metadata, "model_name", None)
    if direct_model_name and metadata_model_name and direct_model_name != metadata_model_name:
        raise ClusterFeatureContractError(
            f"loaded SAE cfg.model_name {direct_model_name!r} disagrees with "
            f"cfg.metadata.model_name {metadata_model_name!r}"
        )
    loaded_model_name = direct_model_name or metadata_model_name
    if loaded_model_name is not None and not _model_names_agree(loaded_model_name, model):
        raise ClusterFeatureContractError(
            f"loaded SAE model_name {loaded_model_name!r} does not match resolved model {model!r}"
        )
    return SaeIdentity(
        release=resolved.release,
        sae_id=resolved.sae_id,
        hook_name=resolved.hook_name,
        model=model,
        layer=layer,
        activation_type=activation_type.value,
        width=width.value,
        actual_loaded_width=loaded_width,
    )


@dataclass(frozen=True)
class _ResolvedSelectionIdentity:
    positive: dict[str, Any]
    negative: dict[str, Any]
    feature_delta_json: str | None
    num_buckets: int
    seed: int
    selection_hash: str
    feature_delta: dict[str, Any] | None


#: The single feature-delta configuration clustered evaluation supports.
#:
#: Splice-counterfactual training with a full negative-delta ratio produces the
#: same conditioning semantics as an undelta'd run for the feature-space
#: methods clustered evaluation drives (encoder-decoder DiffMean and GCG), so
#: its checkpoints are evaluable. Any other mode, or a partial ratio, changes
#: those semantics and stays rejected.
_CLUSTERED_SUPPORTED_DELTA_MODE = DeltaMode.splice_counterfactual
_CLUSTERED_SUPPORTED_NEG_DELTA_RATIO = 1.0


def _supported_clustered_feature_delta(
    user_config: Mapping[str, Any],
) -> tuple[FeatureDeltaConfig | None, dict[str, Any] | None]:
    """Validate the checkpoint's ``feature_delta`` for clustered evaluation.

    Returns ``(parsed, raw_dict)`` for a supported delta and ``(None, None)``
    when the checkpoint carries none. The raw dict is returned alongside the
    parsed config because ``compute_selection_hash`` hashes the raw block —
    re-serializing the parsed dataclass could diverge from the exact bytes the
    producer hashed.

    Raises:
        ClusterFeatureContractError: for a malformed block, or for any
            mode/ratio combination outside the supported one.
    """

    raw = user_config.get("feature_delta")
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        raise ClusterFeatureContractError(
            f"checkpoint feature_delta must be a mapping or null, got {type(raw).__name__}"
        )
    delta_dict = {str(key): value for key, value in raw.items()}
    try:
        parsed = parse_feature_delta(delta_dict)
    except (BeartypeCallHintParamViolation, TypeError, ValueError) as exc:
        raise ClusterFeatureContractError(f"invalid checkpoint feature_delta: {exc}") from exc
    if parsed is None:
        return None, None
    if (
        parsed.mode is not _CLUSTERED_SUPPORTED_DELTA_MODE
        or parsed.neg_delta_ratio != _CLUSTERED_SUPPORTED_NEG_DELTA_RATIO
    ):
        raise ClusterFeatureContractError(
            "clustered evaluation supports only feature_delta.mode="
            f"{_CLUSTERED_SUPPORTED_DELTA_MODE.value} with neg_delta_ratio="
            f"{_CLUSTERED_SUPPORTED_NEG_DELTA_RATIO}; got mode={parsed.mode.value!r}, "
            f"neg_delta_ratio={parsed.neg_delta_ratio!r}"
        )
    return parsed, delta_dict


def _resolve_selection_identity(
    *,
    metadata: Mapping[str, Any],
    feature_mask: FeatureMask,
    feature_clusters: ResolvedFeatureClusters,
    gcg_top_k_pos: int | None,
    gcg_top_k_neg: int | None,
    selection_splice_batch_size: int | None = None,
    enabled_methods: Sequence[str] | None = None,
) -> _ResolvedSelectionIdentity:
    user_config = metadata.get("user_config")
    if not isinstance(user_config, Mapping):
        raise ClusterFeatureContractError("checkpoint user_config is missing")
    positive_raw = user_config.get("feature_selection")
    negative_raw = user_config.get("negative_selection")
    if not isinstance(positive_raw, Mapping):
        raise ClusterFeatureContractError(
            "clustered checkpoint requires feature_selection metadata"
        )
    if not isinstance(negative_raw, Mapping):
        raise ClusterFeatureContractError(
            "clustered checkpoint requires negative_selection metadata"
        )
    positive_dict = {str(key): value for key, value in positive_raw.items()}
    negative_dict = {str(key): value for key, value in negative_raw.items()}
    try:
        positive = parse_feature_selection(positive_dict)
        negative = parse_negative_selection(negative_dict)
    except ValueError as exc:
        raise ClusterFeatureContractError(f"invalid checkpoint selection config: {exc}") from exc
    if negative is None:
        raise ClusterFeatureContractError("clustered checkpoint negative_selection is disabled")
    # Raises for any feature_delta outside the supported splice-counterfactual
    # case; returns (None, None) when the checkpoint carries no delta at all.
    supported_delta, delta_dict = _supported_clustered_feature_delta(user_config)
    if isinstance(positive, SamplingSelection):
        if positive.gradient is not None:
            raise ClusterFeatureContractError(
                "clustered evaluation does not support gradient-attribution selection"
            )
        if positive.delta_weights is not None and supported_delta is None:
            raise ClusterFeatureContractError(
                "clustered evaluation requires delta_weights disabled unless the "
                "checkpoint was also trained with "
                f"feature_delta.mode={DeltaMode.splice_counterfactual.value} and "
                f"neg_delta_ratio={_CLUSTERED_SUPPORTED_NEG_DELTA_RATIO}; got "
                f"delta_weights={positive.delta_weights.value!r} with "
                "feature_delta=null"
            )
    elif isinstance(positive, ThresholdSelection):
        if positive.threshold_type not in {
            ThresholdType.feature_percentile,
            ThresholdType.feature_percentile_segment_topk,
        }:
            raise ClusterFeatureContractError(
                "clustered evaluation supports only percentile-space threshold selection"
            )
        if gcg_top_k_pos is None or gcg_top_k_neg is None:
            raise ClusterFeatureContractError(
                "threshold-selected clustered checkpoints require explicit "
                "gcg_top_k_pos and gcg_top_k_neg overrides"
            )
    # feature_delta is validated by _supported_clustered_feature_delta above,
    # which raises for every mode/ratio outside the supported case. There is no
    # blanket feature_delta=null requirement any more.
    if user_config.get("feature_prediction") is not None:
        # The auxiliary feature-prediction head (`feature_pred_head.*`) lives on
        # the ENCODER-DECODER checkpoint this contract pins. Only methods whose
        # target is that checkpoint can reach it; the decoder-only methods run
        # against a separate model that has no such head, so they are out of
        # scope here by construction rather than by allowlist.
        #
        # The check is therefore keyed on the target, not on a flat name list:
        # a newly-added encdec method is caught automatically instead of
        # silently slipping through a hand-maintained constant. Method names
        # are already validated upstream by
        # `validate_clustered_method_allowlist`, so an unrecognized name cannot
        # reach here and be quietly treated as decoder-only.
        if enabled_methods is None:
            # Undeclared means "every method", which includes encdec ones.
            raise ClusterFeatureContractError(
                "checkpoint carries feature_prediction; clustered evaluation "
                "requires the caller to declare enabled_methods so any "
                "encoder-decoder-targeting method can be checked against "
                f"{list(CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS)}"
            )
        encdec_methods = set(enabled_methods) & set(CLUSTERED_ENCDEC_TARGET_METHODS)
        offenders = sorted(encdec_methods - set(CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS))
        if offenders:
            raise ClusterFeatureContractError(
                "checkpoint carries feature_prediction, which clustered "
                "evaluation supports only for the head-agnostic encoder-decoder "
                f"methods {list(CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS)}; "
                f"enabled_methods additionally requests {offenders}"
            )
    # Codex combined-review finding 5 (MEDIUM): `value not in (None, False)`
    # is a value-EQUALITY test, and Python's `0 == False` / `0.0 == False`
    # are both True -- a checkpoint carrying `self_conditioning: 0` (or
    # `0.0`) therefore silently passed as "disabled" without ever actually
    # being the null/False sentinel this check means to require. Identity
    # checks (`is None`, `is False`) accept only the two literal disabled
    # spellings and treat every other falsy-but-distinct value (0, 0.0, "",
    # []) as still enabled, which is what fails loudly here.
    for key in ("self_conditioning", "selfcond", "self_conditioned"):
        value = user_config.get(key)
        if value is not None and value is not False:
            raise ClusterFeatureContractError(
                f"clustered evaluation requires checkpoint {key}=null/false"
            )
    num_buckets = user_config.get("num_buckets", metadata.get("num_buckets"))
    seed = user_config.get("seed")
    if type(num_buckets) is not int or num_buckets != 2:
        raise ClusterFeatureContractError(
            f"clustered evaluation requires num_buckets == 2, got {num_buckets!r}"
        )
    if type(seed) is not int:
        raise ClusterFeatureContractError(
            f"checkpoint preselection seed must be a strict int, got {seed!r}"
        )
    # ANTI-DESYNC (Codex combined-review finding 6, MEDIUM — this is the
    # fifth site, added by this PR after the original four; see
    # tests/steering/test_selection_hash_cross_site_consistency.py, which
    # now also enumerates and behaviorally-parity-checks this site): this
    # selection-hash is the checkpoint-preflight-side
    # ``compute_selection_hash`` call at:
    #   1. scripts/preprocess/precompute_selected_features.py
    #   2. scripts/train/conditional_encdec_train.py:build_shared_state
    #   3. scripts/eval/base_eval.py:_resolve_bpb_selection_dir
    #   4. steering/eval/feature_selection_hash.py:compute_canonical_feature_selection_hash
    #   5. steering/eval/cluster_checkpoint_pin.py:_resolve_selection_identity (THIS site)
    # All five must agree on feature_mask=canonical_dict(...) semantics.
    # Change ALL FIVE or none. (Sites 1-4's own comments still literally say
    # "ALL FOUR"/"ALL 4"; they predate this site and are not edited here —
    # out of this fix batch's scope. tests/steering/test_selection_hash_
    # cross_site_consistency.py is the actual enforcement mechanism, not
    # this prose.)
    selection_hash = compute_selection_hash(
        positive_selection=positive_dict,
        negative_selection=negative_dict,
        # `delta_config` is an unconditional payload key, so a splice-trained
        # checkpoint MUST hash its real block here or this site recomputes a
        # hash the producer never wrote and manifest discovery finds nothing.
        # `_supported_clustered_feature_delta` has already rejected every
        # unsupported mode/ratio, so this is either None or the supported
        # splice-counterfactual block.
        delta_config=delta_dict,
        num_buckets=num_buckets,
        seed=seed,
        ddp_rank=0,
        feature_mask=canonical_feature_mask_dict(feature_mask),
        feature_clusters=canonical_feature_clusters_descriptor(feature_clusters),
        # Producer-side parallelism knob, NOT recorded in checkpoint metadata,
        # so it must be supplied by the caller when the producer batched. None
        # resolves to the legacy batch size, which `splice_selection_batching`
        # omits from identity, reproducing every unbatched producer exactly.
        # A wrong value surfaces loudly as "resolved 0 manifests" at discovery
        # rather than as a silently wrong selection.
        splice_batch_size=selection_splice_batch_size,
    )
    return _ResolvedSelectionIdentity(
        positive=positive_dict,
        negative=negative_dict,
        feature_delta_json=(
            None if delta_dict is None else _canonical_json(delta_dict)
        ),
        num_buckets=num_buckets,
        seed=seed,
        selection_hash=selection_hash,
        feature_delta=delta_dict,
    )


def resolve_selection_manifest_path(
    *,
    feature_dir: str | Path,
    metadata: Mapping[str, Any],
    feature_mask: FeatureMask,
    feature_clusters: ResolvedFeatureClusters,
    gcg_top_k_pos: int | None = None,
    gcg_top_k_neg: int | None = None,
    selection_splice_batch_size: int | None = None,
    enabled_methods: Sequence[str] | None = None,
) -> Path:
    """Derive the exact clustered ``sel/<hash>/manifest.json`` artifact.

    Packed-row layout coordinates are historical producer provenance, not
    public evaluation config. The recomputed selection hash is therefore used
    to discover the single matching manifest below ``feature_dir``. Zero or
    multiple matches fail rather than picking a heuristic candidate.

    Codex combined-review finding 4 (MEDIUM): layout names used to be
    checked BEFORE ``path.resolve()``, but the resolved candidate was never
    checked for containment inside ``feature_dir`` afterward — an in-root
    ``sel/<hash>/manifest.json`` symlink to an external, unrelated matching
    artifact tree resolved to that outside target and was accepted as a
    genuine candidate. Every discovered ``manifest.json`` is now rejected
    outright if it is itself a symlink, and every resolved candidate must be
    contained within the resolved feature root before it is even counted.
    """

    identity = _resolve_selection_identity(
        metadata=metadata,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        gcg_top_k_pos=gcg_top_k_pos,
        gcg_top_k_neg=gcg_top_k_neg,
        selection_splice_batch_size=selection_splice_batch_size,
        enabled_methods=enabled_methods,
    )
    root = Path(feature_dir).expanduser().resolve(strict=True)
    candidates: list[Path] = []
    for path in root.rglob("manifest.json"):
        if path.parent.name != identity.selection_hash or path.parent.parent.name != "sel":
            continue
        if path.is_symlink():
            raise ClusterFeatureContractError(
                f"clustered checkpoint selection manifest must not be a symlink: {path}"
            )
        resolved_candidate = path.resolve(strict=True)
        if not resolved_candidate.is_relative_to(root):
            raise ClusterFeatureContractError(
                "clustered checkpoint selection manifest resolved outside the "
                f"feature root: {path} -> {resolved_candidate} (root={root})"
            )
        candidates.append(resolved_candidate)
    candidates = sorted(candidates)
    if len(candidates) != 1:
        raise ClusterFeatureContractError(
            "clustered checkpoint selection identity resolved "
            f"{len(candidates)} manifests for sel/{identity.selection_hash} under "
            f"{root}; expected exactly one. Re-run precompute_selected_features "
            "or remove duplicate/stale packed-row trees."
        )
    return candidates[0]


def _selection_manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a selection manifest without host/path/timestamp provenance."""

    semantic = json.loads(_canonical_json(manifest))
    if not isinstance(semantic, dict):
        raise ClusterFeatureContractError("selection manifest must be a mapping")
    semantic.pop("created_at", None)
    metadata = semantic.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop(CLUSTERED_SELECTION_VALIDATION_KEY, None)
        cluster_metadata = metadata.get("feature_clusters")
        if isinstance(cluster_metadata, dict):
            cluster_metadata.pop("source_path", None)
    return _sha256_json(semantic)


def _unique_shard_stem_map(paths: Sequence[Path], *, surface: str) -> dict[str, Path]:
    """Map each shard path to its exact, full-suffix-stripped stem.

    Codex convergence-pass residual 2 (MEDIUM): the previous extraction,
    ``path.stem.split(".")[0]``, collapsed any filename whose logical stem
    itself contains a dot (e.g. ``x.aaa.safetensors.zst`` and
    ``x.safetensors.zst`` both normalize to ``x``) into one dict key —
    coalescing silently, since a plain dict comprehension just keeps
    whichever path glob order produced last for that key, and the coverage
    check downstream never notices a shard went missing from the map.
    Stripping the exact, fully-known ``COMPRESSED_EXT`` suffix is already
    injective for every path here (all came from ``glob_overlay_shards``,
    which only ever matches ``*{COMPRESSED_EXT}``); the explicit raise below
    is defense in depth against any other future collision source, rather
    than a silent coalesce.
    """
    stems: dict[str, Path] = {}
    for path in paths:
        stem = path.name.removesuffix(COMPRESSED_EXT)
        if stem in stems:
            raise ClusterFeatureContractError(
                f"{surface} has two shards that normalize to the same stem "
                f"{stem!r}: {stems[stem].name} and {path.name}"
            )
        stems[stem] = path
    return stems


def _shard_stat_signatures(
    paths: Sequence[Path],
) -> dict[Path, tuple[int, int, int, int, int, int]]:
    """Capture every shard's ``lstat``-based stat signature via B1's
    ``_stat_signature`` — the exact 6-tuple ``_assert_digest_stat_unchanged``
    already uses for the two manifest digests — so a later recheck can
    detect a torn write, truncation, or different-inode/mtime swap without
    reading any shard's bytes a second time. O(shards) ``lstat`` calls, not a
    body read. The module's "Accepted threat-model boundary" describes
    the limits of these signature checks.
    """
    try:
        return {path: _stat_signature(path) for path in paths}
    except OSError as exc:
        raise ClusterFeatureContractError(
            f"could not stat a selection/packed shard before validating it: {exc}"
        ) from exc


def _assert_shard_stat_signatures_unchanged(
    before: Mapping[Path, tuple[int, int, int, int, int, int]],
) -> None:
    """Re-``lstat`` every shard captured by ``_shard_stat_signatures`` and
    reject any whose signature changed.

    Mirrors ``_assert_digest_stat_unchanged``'s cold recheck of the two
    manifest digests, extended per-shard: there is no ``StableFileDigest``
    for a shard body here (content identity comes from the compressed
    selection inventory, producer semantic certificate, and strict packed
    cache in ``_selection_contract``) — this closes only the window around
    those proofs, the same class of gap the manifest cold-checks close for
    the two JSON files, at O(shards) stat calls rather than a second byte
    read.

    This loop is itself sequential, not atomic: an earlier path's recheck
    below is not repeated once a later path's recheck has started, so a swap
    of an already-cleared earlier path timed to land during a later path's
    check is not caught by this function — see the module docstring's
    "Codex pass-5 finding" / "Accepted threat-model boundary" sections
    (point (c)) and ``test_selection_contract_accepts_a_sequential_shard_
    fence_swap_of_an_already_rechecked_shard`` for the accepted, tested,
    dataloader-parity-backstopped boundary this leaves.
    """
    for path, signature in before.items():
        try:
            current = _stat_signature(path)
        except OSError as exc:
            raise ClusterFeatureContractError(
                f"a selection/packed shard disappeared while this contract was "
                f"validating it: {path}"
            ) from exc
        if current != signature:
            raise ClusterFeatureContractError(
                f"a selection/packed shard changed while this contract was validating it: {path}"
            )


def _selection_shard_max_features_per_segment(selection: SelectionShard) -> int:
    """Return the exact positive-plus-negative CSR width for forced proof."""

    selected = selection.sel_offsets[1:].to(t.int64) - selection.sel_offsets[
        :-1
    ].to(t.int64)
    negative = selection.neg_offsets[1:].to(t.int64) - selection.neg_offsets[
        :-1
    ].to(t.int64)
    if selected.shape != negative.shape:
        raise ClusterFeatureContractError(
            "selection positive/negative CSR offsets describe different "
            f"segment counts: {selected.numel()} != {negative.numel()}"
        )
    widths = selected + negative
    return int(widths.max().item()) if widths.numel() else 0


def _assert_selection_contract_files_current(
    *,
    manifest_digest: StableFileDigest,
    packed_manifest_digest: StableFileDigest,
    packed_dir: Path,
    selection_dir: Path,
    expected_packed_names: tuple[str, ...],
    expected_selection_names: tuple[str, ...],
    shard_signatures: Mapping[Path, tuple[int, int, int, int, int, int]],
    packed_resolution: PackedFingerprintResolution,
) -> None:
    """Recheck manifests, exact shard sets, signatures, and packed proof."""

    _assert_digest_stat_unchanged(manifest_digest)
    _assert_digest_stat_unchanged(packed_manifest_digest)
    actual_packed_names = tuple(
        sorted(path.name for path in glob_overlay_shards(packed_dir))
    )
    actual_selection_names = tuple(
        sorted(path.name for path in glob_overlay_shards(selection_dir))
    )
    if actual_packed_names != expected_packed_names:
        raise ClusterFeatureContractError(
            "packed-row shard inventory changed while this contract was "
            f"validating it: expected={expected_packed_names[:5]}, "
            f"actual={actual_packed_names[:5]}"
        )
    if actual_selection_names != expected_selection_names:
        raise ClusterFeatureContractError(
            "selection shard inventory changed while this contract was "
            f"validating it: expected={expected_selection_names[:5]}, "
            f"actual={actual_selection_names[:5]}"
        )
    _assert_shard_stat_signatures_unchanged(shard_signatures)
    try:
        assert_packed_row_fingerprint_resolution_current(packed_resolution)
    except SpliceProvenanceError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc


def _selection_contract(
    *,
    metadata: Mapping[str, Any],
    manifest_path: Path,
    feature_mask: FeatureMask,
    feature_clusters: ResolvedFeatureClusters,
    consolidation: FeatureClusterConsolidation,
    percentile_checksum: str,
    gcg_top_k_pos: int | None,
    gcg_top_k_neg: int | None,
    selection_splice_batch_size: int | None = None,
    enabled_methods: Sequence[str] | None = None,
    force_cluster_shard_revalidation: bool = False,
    allow_packed_cache_write: bool = True,
) -> CheckpointSelectionContract:
    """Assemble and validate the checkpoint's selection identity against its
    on-disk manifest/shards. This checks recorded settings and canonical
    validity, not which artifact bytes training actually consumed.

    ``_load_json_mapping`` binds the manifest payload to its file digest;
    no separate preliminary hash is needed.

    The certified path rehashes compressed selection bytes and validates the
    strict packed cache without decompression. Forced validation additionally
    executes the canonical packed/selection loaders and representative checks.
    Both paths retain the same final manifest and shard-signature fences and
    return an identical semantic contract.
    """
    if not isinstance(force_cluster_shard_revalidation, bool):
        raise ClusterFeatureContractError(
            "force_cluster_shard_revalidation must be a bool"
        )
    if not isinstance(allow_packed_cache_write, bool):
        raise ClusterFeatureContractError("allow_packed_cache_write must be a bool")
    identity = _resolve_selection_identity(
        metadata=metadata,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        gcg_top_k_pos=gcg_top_k_pos,
        gcg_top_k_neg=gcg_top_k_neg,
        selection_splice_batch_size=selection_splice_batch_size,
        enabled_methods=enabled_methods,
    )
    manifest, manifest_digest = _load_json_mapping(manifest_path, surface="selection manifest")
    # Codex combined-review finding 3 (HIGH): the cold recheck of this digest
    # used to run immediately here, BEFORE every subsequent shard/packed-row
    # read this function performs -- an earlier-hashed file could change
    # after being checked but before the transaction finishes, and nothing
    # would notice. Moved down to join the packed-manifest digest's own cold
    # recheck (see the digest cold-check batch below, immediately ahead of
    # the shard stat-fence's own genuinely-final recheck -- Codex pass-4),
    # after the last read that depends on the manifest still being the
    # content this function already parsed -- the exact same "cold-check
    # widens to cover everything the transaction still has left to do"
    # pattern resolve_checkpoint_pin already applies to the checkpoint
    # metadata digest after the model load.
    if manifest.get("checksum") != identity.selection_hash:
        raise ClusterFeatureContractError(
            "selection manifest checksum does not match the checkpoint selection identity"
        )
    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, Mapping):
        raise ClusterFeatureContractError("selection manifest metadata is missing")
    if manifest_metadata.get("feature_mask") != canonical_feature_mask_dict(feature_mask):
        raise ClusterFeatureContractError(
            "selection manifest feature mask does not match the checkpoint/config"
        )
    selection_dir = manifest_path.parent
    # Moved earlier than the oracle/original port (was derived only after the
    # packed_rows_meta check below): Codex combined-review finding 1 (HIGH)
    # needs packed_dir to cross-check the packed-row shard domain before this
    # function trusts anything the manifest itself claims about it.
    packed_dir = selection_dir.parent.parent
    try:
        assert_feature_clusters_selection_metadata(
            consolidation,
            manifest_metadata,
            surface=f"selection manifest {manifest_path}",
        )
    except FeatureClustersError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc
    shard_integrity = manifest_metadata.get("selection_shards")
    if not isinstance(shard_integrity, Mapping) or not is_sha256_hex(
        shard_integrity.get("aggregate_sha256")
    ):
        raise ClusterFeatureContractError(
            "selection manifest is missing its selected-shard aggregate identity"
        )
    packed_rows_meta = manifest_metadata.get("packed_rows")
    if not isinstance(packed_rows_meta, Mapping) or not is_sha256_hex(
        packed_rows_meta.get("packed_row_fingerprint")
    ):
        raise ClusterFeatureContractError("selection manifest is missing packed-row provenance")

    # Codex combined-review finding 1 (HIGH): the byte-integrity check below
    # only proves the manifest's OWN declared shard inventory is
    # byte-self-consistent -- a coordinated rewrite that replaces a shard AND
    # re-signs the manifest's inventory/aggregate to match passes it
    # trivially, and it has no knowledge of packed_dir at all, so it cannot
    # detect an incomplete selection/packed coverage mismatch either. Two
    # independent, manifest-content-blind checks close both gaps:
    #   (a) exact packed/selection shard STEM equality against what is
    #       actually on disk in both directories -- mirrors the producer's
    #       own invariant (precompute_selected_features.py's
    #       `final_integrity_stems != base_stem_set` check at manifest-write
    #       time) and the dataloader's own coverage check
    #       (steering/conditional/encoder_decoder/dataloader.py's
    #       missing/extra stem check before it ever yields a batch).
    #   (b) producer authority that binds canonical semantic validation to
    #       the exact compressed inventory and current cluster/mask
    #       descriptor. The certified path rehashes that inventory; the
    #       forced path additionally repeats
    #       FeatureClusterConsolidation.assert_selection_shard against every
    #       decoded tensor, mirroring the dataloader's per-shard call.
    packed_shard_paths = glob_overlay_shards(packed_dir)
    if not packed_shard_paths:
        raise ClusterFeatureContractError(f"packed-row directory has no shards: {packed_dir}")
    selection_shard_paths = glob_overlay_shards(selection_dir)
    if not selection_shard_paths:
        raise ClusterFeatureContractError(f"selection directory has no shards: {selection_dir}")
    expected_packed_names = tuple(
        sorted(path.name for path in packed_shard_paths)
    )
    expected_selection_names = tuple(
        sorted(path.name for path in selection_shard_paths)
    )
    # Codex convergence-pass residual 2 (MEDIUM): see `_unique_shard_stem_map`
    # above -- `path.stem.split(".")[0]` used to collapse distinct filenames
    # whose stem itself contains a dot into the same key, silently coalescing
    # in the dict comprehension instead of raising.
    packed_by_stem = _unique_shard_stem_map(
        packed_shard_paths, surface=f"packed directory {packed_dir}"
    )
    selection_by_stem = _unique_shard_stem_map(
        selection_shard_paths, surface=f"selection directory {selection_dir}"
    )
    missing = sorted(set(packed_by_stem) - set(selection_by_stem))
    extra = sorted(set(selection_by_stem) - set(packed_by_stem))
    if missing or extra:
        raise ClusterFeatureContractError(
            f"selection shards at {selection_dir} must match packed shard stems "
            f"exactly: missing={missing[:5]}, extra={extra[:5]}"
        )
    # `_resolve_selection_identity` above unconditionally requires
    # negative_selection to be enabled (its own "negative_selection is
    # disabled" raise) and num_buckets == 2, so this contract's domain always
    # has effective_buckets = num_buckets - 1 = 1 -- the exact closed form of
    # `feature_pipeline.build_feature_pipeline`'s own
    # `num_buckets - 1 if negative_selection is not None else num_buckets`,
    # and the same value `CheckpointSelectionContract.canonical_dict()`
    # already hardcodes as `effective_positive_buckets=1` below.
    effective_buckets = identity.num_buckets - 1

    try:
        producer_authority = resolve_clustered_selection_validation(
            manifest_metadata,
            selection_checksum=identity.selection_hash,
            selection_integrity=shard_integrity,
            packed_row_fingerprint=str(
                packed_rows_meta["packed_row_fingerprint"]
            ),
            feature_clusters_descriptor=canonical_feature_clusters_descriptor(
                feature_clusters
            )
            or {},
            feature_delta=identity.feature_delta,
            effective_buckets=effective_buckets,
            surface=f"selection manifest {manifest_path}",
        )
    except FeatureClustersError as exc:
        if "clustered validation packed_row_fingerprint" in str(exc):
            raise ClusterFeatureContractError(
                "producer validation packed-row fingerprint does not match "
                "current bytes declared by the selection manifest"
            ) from exc
        raise ClusterFeatureContractError(str(exc)) from exc

    # Capture the exact on-disk shard set before any content proof. The final
    # fence below retains the documented sequential by-path boundary; it is
    # intentionally not claimed to be an atomic multi-file snapshot.
    all_shard_paths = [*selection_shard_paths, *packed_shard_paths]
    shard_signatures_before = _shard_stat_signatures(all_shard_paths)

    try:
        assert_clustered_selection_shard_integrity(
            selection_dir,
            manifest_metadata,
            surface=f"selection manifest {manifest_path}",
        )
    except FeatureClustersError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc

    # Codex combined-review finding 3 (HIGH): route the packed manifest read
    # through the same digest-bound path the selection manifest already
    # uses, rather than the unbound (pathname-check-then-separate-open)
    # ``overlay_io.read_manifest``. ``overlay_io`` itself stays untouched
    # (main-owned, shared by many other callers, out of this surgical
    # module's scope, same "wrap don't edit" precedent already applied to
    # ``read_manifest``'s own sibling gaps elsewhere) -- wrapped here, at the
    # call site, by reusing this module's own ``_load_json_mapping`` (B1's
    # ``stream_hash_file_stable``, hash-once/read-once/content-verified) and
    # preserving ``read_manifest``'s exact "missing or malformed -> the same
    # one error" observable behavior via the surrounding try/except and the
    # explicit ``"checksum" in packed_manifest`` shape gate below (mirroring
    # ``read_manifest``'s own ``"checksum" not in data -> None`` contract).
    # This manifest is a distinct file from every shard body above, outside
    # the shard stat-fence -- it keeps its own independent fused hash+read
    # and its own cold recheck below.
    try:
        packed_manifest, packed_manifest_digest = _load_json_mapping(
            packed_dir / MANIFEST_FILENAME, surface="packed-row manifest"
        )
    except OSError as exc:
        raise ClusterFeatureContractError(
            f"selection parent is missing a valid packed-row manifest: {packed_dir}"
        ) from exc
    if "checksum" not in packed_manifest:
        raise ClusterFeatureContractError(
            f"selection parent is missing a valid packed-row manifest: {packed_dir}"
        )
    packed_metadata = packed_manifest.get("metadata")
    if not isinstance(packed_metadata, Mapping) or (
        packed_metadata.get("pct_checksum") != percentile_checksum
    ):
        raise ClusterFeatureContractError(
            "packed-row percentile provenance does not match feature statistics"
        )
    # Packed content caching is independent of selection approval. Forced
    # validation still completes before any optional cache write.
    certified_segments = dict(producer_authority.segment_counts_by_name)
    certified_widths = (
        None
        if producer_authority.max_features_by_name is None
        else dict(producer_authority.max_features_by_name)
    )
    forced_validated_names: set[str] = set()
    forced_max_features_per_segment = 0

    def _validate_forced_packed_body(
        packed_path: Path,
        compressed: bytes,
    ) -> None:
        nonlocal forced_max_features_per_segment

        stem = packed_path.name.removesuffix(COMPRESSED_EXT)
        selection_path = selection_by_stem.get(stem)
        if selection_path is None:
            raise ClusterFeatureContractError(
                f"forced packed validation found no selection shard for {packed_path.name}"
            )
        packed_shard = load_packed_row_shard_bytes(compressed)
        selection_shard = load_selection_shard(selection_path)
        certified_segments_for_shard = certified_segments.get(
            selection_path.name
        )
        if packed_shard.total_segments != certified_segments_for_shard:
            raise ClusterFeatureContractError(
                "forced validation found a producer-certificate segment-count "
                f"mismatch for {selection_path.name}: "
                f"certificate={certified_segments_for_shard!r}, "
                f"actual={packed_shard.total_segments}"
            )
        try:
            consolidation.assert_selection_shard(
                sel_offsets=selection_shard.sel_offsets,
                sel_fids=selection_shard.sel_fids,
                sel_buckets=selection_shard.sel_buckets,
                sel_pcts=selection_shard.sel_pcts,
                neg_offsets=selection_shard.neg_offsets,
                neg_fids=selection_shard.neg_fids,
                expected_segments=packed_shard.total_segments,
                effective_buckets=effective_buckets,
                surface=f"selection shard {selection_path.name}",
            )
        except FeatureClustersError as exc:
            raise ClusterFeatureContractError(str(exc)) from exc
        actual_width = _selection_shard_max_features_per_segment(
            selection_shard
        )
        if (
            certified_widths is not None
            and certified_widths.get(selection_path.name) != actual_width
        ):
            raise ClusterFeatureContractError(
                "forced validation found a producer-certificate width "
                f"mismatch for {selection_path.name}: "
                f"certificate={certified_widths.get(selection_path.name)!r}, "
                f"actual={actual_width}"
            )
        forced_max_features_per_segment = max(
            forced_max_features_per_segment,
            actual_width,
        )
        forced_validated_names.add(packed_path.name)

    try:
        packed_resolution = resolve_packed_row_fingerprint_strict(
            packed_dir,
            force=force_cluster_shard_revalidation,
            allow_cache_write=allow_packed_cache_write,
            body_validator=(
                _validate_forced_packed_body
                if force_cluster_shard_revalidation
                else None
            ),
        )
    except SpliceProvenanceError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc
    actual_packed_fingerprint = packed_resolution.aggregate_sha256
    expected_packed_fingerprint = str(packed_rows_meta["packed_row_fingerprint"])
    if actual_packed_fingerprint != expected_packed_fingerprint:
        raise ClusterFeatureContractError(
            "selection manifest packed-row fingerprint does not match current bytes"
        )
    if producer_authority.packed_row_fingerprint != actual_packed_fingerprint:
        raise ClusterFeatureContractError(
            "producer validation packed-row fingerprint does not match current bytes"
        )
    try:
        assert_validation_packed_inventory(
            producer_authority,
            packed_resolution.content_inventory(),
            surface=f"selection manifest {manifest_path}",
        )
    except FeatureClustersError as exc:
        raise ClusterFeatureContractError(str(exc)) from exc

    if force_cluster_shard_revalidation and (
        forced_validated_names != set(expected_packed_names)
        or forced_max_features_per_segment
        != producer_authority.max_features_per_segment
    ):
        raise ClusterFeatureContractError(
            "forced validation did not reproduce the producer certificate's "
            "exact packed-shard domain and global maximum selection width"
        )
    # ==================================================================
    # RETURN FENCES (Codex combined-review finding 3 -> pass-4 -> pass-5;
    # see the module docstring's "Accepted threat-model boundary" section
    # for the full history and the honest account this comment summarizes).
    # Recheck source, manifest, and selection identity before returning.
    # The earlier best-effort cache write is independent of this approval.
    #
    # NOT atomic. Each check here is its own sequential userspace
    # lstat/read, matching (not exceeding) the sequential by-path validation
    # `dataloader._validate_generic_selection_dir` already performs against
    # these identical files. An item already rechecked earlier in this batch
    # -- a manifest digest, or an earlier shard in the stat-fence's own loop
    # below -- can still be swapped while a LATER item in this SAME batch is
    # being rechecked; nothing loops back to re-verify it. Codex pass-5
    # proved this deterministically, twice: a manifest swap during the shard
    # fence's loop, and a shard swap during that same loop's later
    # iterations. See
    # `test_selection_contract_accepts_a_selection_manifest_swap_staged_during_the_shard_fence_loop`
    # and
    # `test_selection_contract_accepts_a_sequential_shard_fence_swap_of_an_already_rechecked_shard`.
    #
    # No ordering of these checks closes this window -- it is a structural
    # property of validating N>1 files sequentially by path, not a bug in
    # this batch's order (reordering only relocates which item is exposed).
    # The eval-time dataloader's own load-time integrity check is the
    # AUTHORITATIVE backstop for this residual class; this batch is early
    # warning and strictly additive defense-in-depth beyond dataloader
    # parity, never a substitute for it -- see "OWNER SCOPE DECISION" above.
    # ==================================================================
    _assert_selection_contract_files_current(
        manifest_digest=manifest_digest,
        packed_manifest_digest=packed_manifest_digest,
        packed_dir=packed_dir,
        selection_dir=selection_dir,
        expected_packed_names=expected_packed_names,
        expected_selection_names=expected_selection_names,
        shard_signatures=shard_signatures_before,
        packed_resolution=packed_resolution,
    )
    return CheckpointSelectionContract(
        positive_selection_json=_canonical_json(identity.positive),
        negative_selection_json=_canonical_json(identity.negative),
        feature_delta_json=identity.feature_delta_json,
        num_buckets=identity.num_buckets,
        effective_positive_buckets=1,
        preselection_seed=identity.seed,
        selection_hash=identity.selection_hash,
        selection_manifest_sha256=_selection_manifest_semantic_sha256(manifest),
        selection_shards_sha256=str(shard_integrity["aggregate_sha256"]),
        packed_row_fingerprint=actual_packed_fingerprint,
        selection_manifest_file_sha256=manifest_digest.sha256,
    )


__all__ = [
    "CheckpointFilesPin",
    "CheckpointPin",
    "CheckpointSelectionContract",
    "ClusterFeatureContractError",
    "assert_checkpoint_files_pin_unchanged",
    "resolve_checkpoint_files_pin",
    "resolve_checkpoint_pin",
    "resolve_eval_checkpoint_directory",
    "resolve_selection_manifest_path",
]
