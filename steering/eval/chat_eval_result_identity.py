"""Checkpoint and execution receipt for mergeable turn ChatCORE results."""

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import torch as t
import zstandard

from nanochat.common import COMPUTE_DTYPE
from steering._common.file_inventory import file_signature
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY,
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
)
from steering.conditional.feature_mask import canonical_dict
from steering.conditional.feature_vocab import META_KEY as FEATURE_VOCAB_META_KEY
from steering.eval.context_cache_io import (
    SCHEMA_VERSION,
    TURN_SCHEMA_VERSION,
    SafetensorsContextFeatureProvider,
)
from steering.eval.self_conditioning_policy import SELF_CONDITIONING_SEED_VERSION
from steering.eval.turn_checkpoint_provenance import _model_digest, turn_checkpoint_provenance
from steering.eval.turn_context_contract import (
    TurnContextContract,
    canonical_turn_payload,
    turn_context_contract_hash,
)
from steering.safetensors_io import read_safetensors_metadata_from_bytes
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from steering.sft.segment_mode_identity import SEGMENTATION_IDENTITY_META_KEYS

GENERATION_SETTING_KEYS = frozenset({
    "dtype", "compute_dtype", "device_type", "num_samples", "max_new_tokens", "temperature",
    "top_k", "max_problems", "batch_size", "greedy_microbatch_size", "greedy_python_tools",
})


class ChatEvalIdentityVersion(IntEnum):
    LEGACY = 1
    PROBLEM_RANGES = 2
    SELF_CONDITIONED = 3


class SelfConditionedPrecision(StrEnum):
    MODEL_OWNED = "model_owned_no_autocast_v1"


RANGED_GENERATION_SETTING_KEYS = GENERATION_SETTING_KEYS - {"max_problems"}

# These values are checkpoint authority, even when absent or null. Loader
# defaults and ancestor recovery must never add to or fill this contract.
_CONTRACT_PATHS = tuple((key,) for key in (
    *SEGMENTATION_IDENTITY_META_KEYS,
    FEATURE_VOCAB_META_KEY,
    "feature_mask_canonical",
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
    CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY,
    "num_buckets",
    "n_sae_features",
)) + (("model_config", "sequence_len"),)
_MISSING = object()


def _metadata_at_path(metadata: dict, path: tuple[str, ...]) -> Any:
    value = metadata
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _assert_metadata_extension(
    expected: Any, loaded: Any, path: str = "", *, allow_extensions: bool = True,
) -> None:
    """Preserve existing values and keys; lists retain exact contents and order."""
    if allow_extensions and expected is None and loaded is not _MISSING:
        return
    if isinstance(expected, dict) and isinstance(loaded, dict):
        keys = expected.keys() if allow_extensions else expected.keys() | loaded.keys()
        for key in sorted(keys):
            _assert_metadata_extension(
                expected.get(key, _MISSING), loaded.get(key, _MISSING),
                f"{path}.{key}" if path else key, allow_extensions=allow_extensions,
            )
        return
    if isinstance(expected, list) and isinstance(loaded, list) and len(expected) == len(loaded):
        for index, (before, after) in enumerate(zip(expected, loaded, strict=True)):
            _assert_metadata_extension(before, after, f"{path}[{index}]", allow_extensions=False)
        return
    if type(expected) is not type(loaded) or expected != loaded:
        raise ValueError(f"chat_eval loaded checkpoint metadata differs from pre-load metadata at {path}")


def _assert_loaded_metadata_contract(expected: dict, loaded: dict) -> None:
    for path in _CONTRACT_PATHS:
        _assert_metadata_extension(
            _metadata_at_path(expected, path), _metadata_at_path(loaded, path),
            ".".join(path), allow_extensions=False,
        )
    # SFT user_config.num_buckets=null means inherit the authoritative top-level
    # value. Recovery may populate redundant copies only with that same value.
    buckets = expected.get("num_buckets", _metadata_at_path(expected, ("user_config", "num_buckets")))
    for block in ("user_config", "pretrain_conditioning"):
        recovered = _metadata_at_path(loaded, (block, "num_buckets"))
        if recovered is not _MISSING and recovered is not None:
            _assert_metadata_extension(buckets, recovered, f"{block}.num_buckets", allow_extensions=False)


@dataclass(frozen=True)
class TurnChatEvalCheckpointSnapshot:
    """Fence the weights from before loading until each receipt is constructed."""

    path: Path
    payload: bytes
    model_path: Path
    model_signature: tuple[int, ...]

    @classmethod
    def capture(cls, path: Path, payload: bytes, step: int) -> "TurnChatEvalCheckpointSnapshot":
        model_path = path.with_name(f"model_{step:06d}.pt")
        return cls(path, payload, model_path, file_signature(model_path))

    def assert_stable(self) -> None:
        if file_signature(self.model_path) != self.model_signature:
            raise ValueError("chat_eval checkpoint model changed between load and receipt")
        if self.path.read_bytes() != self.payload:
            raise ValueError("chat_eval checkpoint metadata changed between load and receipt")

    def _assert_loaded(self, metadata: dict) -> None:
        """Bind original bytes while allowing only non-contract loader extensions.

        build_encdec_model patches missing model-config defaults and recovers
        null SFT user-config fields from ancestors in place. Existing non-null
        values may not change or disappear; contract fields stay exactly equal.
        """
        # checkpoint_dir is runtime loader provenance, not a persisted model setting.
        expected = json.loads(self.payload)
        loaded = {key: value for key, value in metadata.items() if key != "checkpoint_dir"}
        expected.pop("checkpoint_dir", None)
        json.dumps(loaded, allow_nan=False)  # Reject non-JSON and non-finite loader additions too.
        _assert_loaded_metadata_contract(expected, loaded)
        _assert_metadata_extension(expected, loaded)
        if Path(metadata["checkpoint_dir"]).resolve() != self.path.parent:
            raise ValueError("chat_eval loaded checkpoint directory differs from pinned directory")
        self.assert_stable()

    def bind_loaded(self, metadata: dict) -> dict[str, Any]:
        self._assert_loaded(metadata)
        provenance = turn_checkpoint_provenance(self.path, self.payload, metadata)
        self.assert_stable()
        if provenance["checkpoint_model_sha256"] is None:
            raise ValueError("mergeable ChatCORE results require checkpoint model bytes")
        return {
            "metadata_sha256": provenance["checkpoint_metadata_sha256"],
            "model_sha256": provenance["checkpoint_model_sha256"],
            "resolved_tag": provenance["checkpoint_resolved_tag"],
            "step": provenance["checkpoint_step"],
        }


def effective_architecture_sha256(model) -> str:
    """Digest the configuration specification the model was built from.

    The leaf metadata bytes are hashed separately; ancestor recovery can fill
    shape-compatible settings (e.g. cross-attention heads) that change the
    computation without touching the leaf files, so they are bound here. The
    specification is digested as written (ratio vs absolute encoder sizes are
    distinct), so equivalent encodings deliberately never merge.
    """
    architecture = {
        "decoder": asdict(model.dec_config),
        "encoder": asdict(model.enc_config),
        "cross_attention": asdict(model.xattn_config),
    }
    payload = json.dumps(architecture, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SelfConditionedChatEvalCheckpointSnapshot(TurnChatEvalCheckpointSnapshot):
    """Same byte/loader fences, with no fabricated online-SAE lineage for fixed models."""

    def bind_loaded(self, metadata: dict) -> dict[str, Any]:
        self._assert_loaded(metadata)
        step = int(self.path.stem.removeprefix("meta_"))
        if "step" in metadata and (type(metadata["step"]) is not int or metadata["step"] != step):
            raise ValueError("checkpoint metadata step disagrees with filename")
        digest, _ = _model_digest(self.model_path)
        if digest is None:
            raise ValueError("self-conditioned receipt requires checkpoint model bytes")
        self.assert_stable()
        tag = self.path.parent.name
        for parent in self.path.parents:
            if parent.name == "conditional_encdec_checkpoints":
                tag = self.path.parent.relative_to(parent).as_posix()
                break
        return {"metadata_sha256": hashlib.sha256(self.payload).hexdigest(),
                "model_sha256": digest, "step": step, "resolved_tag": tag}


def build_turn_chat_eval_run_identity(
    cfg, *, checkpoint_identity: dict, model,
    turn_contract: TurnContextContract, device_type: str, provider: SafetensorsContextFeatureProvider,
) -> dict[str, Any]:
    """Record effective settings; task ranges and output paths are not identity.

    Checkpoint hashes use the selected metadata bytes and stable on-disk weights,
    plus the effective architecture the loader built. ``compute_dtype`` is the
    process-wide precision the encoder-decoder actually computes in, which the
    requested ``dtype`` does not determine on the target-real path.
    The conditioning digest covers the full authenticated turn contract, including
    tokenizers, statistics, vocabulary, selection policy and selection seed.
    Cache contents come from the provider's loaded, checksum-verified manifest,
    including ordered source population and input/output shard digests.
    Version 2 records problem ranges and counts separately in ``task_details``.
    """
    if provider.turn_contract != turn_contract:
        raise ValueError("ChatCORE receipt provider disagrees with the turn contract")
    metadata = provider.manifest["metadata"]
    inputs = metadata["input_provenance"]
    return {
        "version": ChatEvalIdentityVersion.PROBLEM_RANGES.value,
        "checkpoint": {**checkpoint_identity, "architecture_sha256": effective_architecture_sha256(model)},
        "conditioning": {"turn_context_contract_sha256": turn_context_contract_hash(turn_contract)},
        "cache": {
            "resolved_path": str(provider.cache_dir.expanduser().resolve(strict=True)),
            "contexts_fingerprint": metadata["contexts_fingerprint"],
            "input_provenance": deepcopy({key: inputs[key] for key in (
                "jsonl_sha256", "parquet_files", "feature_files",
            )}),
            "shards": [{"file": shard["file"], "sha256": shard["sha256"]}
                       for shard in provider.manifest["shards"]],
        },
        "generation": {
            "dtype": cfg.dtype, "compute_dtype": str(COMPUTE_DTYPE), "device_type": str(device_type),
            "num_samples": cfg.num_samples, "max_new_tokens": cfg.max_new_tokens,
            "temperature": float(cfg.temperature), "top_k": cfg.top_k,
            "batch_size": cfg.batch_size,
            "greedy_microbatch_size": cfg.get("greedy_microbatch_size", 1),
            "greedy_python_tools": cfg.get("greedy_python_tools", False),
        },
    }


SELF_CONDITIONED_RUNTIME_VERSION = "chatcore_scalar_self_conditioned_v1"


def semantic_digest(value) -> str:
    return hashlib.sha256(canonical_turn_payload(value)).hexdigest()


@dataclass(frozen=True)
class ChatCacheSnapshot:
    """Bind the manifest actually loaded and all compressed shard bytes, including headers."""

    root: Path
    manifest_bytes: bytes
    signatures: tuple[tuple[Path, tuple[int, ...]], ...]

    @classmethod
    def capture(cls, root: Path) -> "ChatCacheSnapshot":
        root = root.resolve(strict=True)
        manifest_path = root / "manifest.json"
        signature = file_signature(manifest_path)
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        signatures = [(manifest_path, signature)]
        for index, shard in enumerate(manifest["shards"]):
            name = shard["file"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("cache shard name must be a basename")
            path = root / name
            state = file_signature(path)
            compressed = path.read_bytes()
            if hashlib.sha256(compressed).hexdigest() != shard["sha256"]:
                raise ValueError("cache shard hash mismatch")
            header = read_safetensors_metadata_from_bytes(zstandard.ZstdDecompressor().decompress(compressed))
            if header != {"schema_version": str(manifest["schema_version"]), "format": manifest["format"], "shard_index": str(index)}:
                raise ValueError("cache shard header/schema mismatch")
            signatures.append((path, state))
        snapshot = cls(root, payload, tuple(signatures))
        snapshot.assert_stable()
        return snapshot

    def assert_stable(self) -> None:
        if any(file_signature(path) != signature for path, signature in self.signatures):
            raise ValueError("cache changed during load/evaluation/receipt")
        if (self.root / "manifest.json").read_bytes() != self.manifest_bytes:
            raise ValueError("cache manifest changed during load/evaluation/receipt")

    def bind_loaded(self, provider: SafetensorsContextFeatureProvider) -> None:
        self.assert_stable()
        if provider.manifest != json.loads(self.manifest_bytes):
            raise ValueError("loaded cache manifest differs from authenticated snapshot")


def self_conditioned_code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ("steering", "nanochat", "tasks", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return {"runtime_version": SELF_CONDITIONED_RUNTIME_VERSION,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": digest.hexdigest()}


def build_self_conditioned_run_identity(
    cfg, *, settings, model, meta, mode, cadence, tokenizer, policy, feature_contract,
    checkpoint_identity, provider, cache_snapshot, turn_contract, fixed_contract, device_type,
    benchmark_targets_sha256=None,
) -> dict:
    cache_snapshot.bind_loaded(provider)
    # Turn eval copies may use a producer-specific measurement key. Bind every
    # cadence/measurement witness, including the recorded scalar itself.
    provenance = {key: value for key, value in meta.items() if "cadence" in key or "self_cond_segment_size_rustbpe" in key}
    mask = canonical_dict(feature_contract.feature_mask)
    clusters = None if feature_contract.resolved_clusters is None else feature_contract.resolved_clusters.descriptor_dict()
    resolved_vocab = effective_feature_vocab(feature_contract.universe, policy.n_sae_features)
    vocab = None if resolved_vocab is None else resolved_vocab.metadata_dict()
    if fixed_contract is not None:
        source = fixed_contract.source
        preselection = fixed_contract.metadata["prompt_preselection"]
    else:
        source = turn_contract.semantic
        preselection = {"policy": source["selection_semantics"], "seed": turn_contract.lineage.selection_seed}
    interpolation = policy.score_policy
    if interpolation is None:
        raise ValueError("self-conditioned receipt requires two-sided score policy")
    identity = {
        "version": ChatEvalIdentityVersion.SELF_CONDITIONED.value,
        "checkpoint": {**checkpoint_identity, "architecture_sha256": effective_architecture_sha256(model)},
        "conditioning": {
            "evaluation_mode": "self_conditioned", "segment_mode": mode.value,
            "recorded_k": cadence, "measurement_provenance_sha256": semantic_digest(provenance),
            "cadence_fidelity": "measured_rustbpe_approximation",
            "selection_policy": "axbench_pure_model",
            "positive_selection": asdict(policy.selection.selection),
            "negative_selection": asdict(interpolation.negative_selection),
            "pool_mode": policy.pool_mode.value,
            "selection_producer": policy.selection_producer.value, "bucket_producer": policy.bucket_producer.value,
            "score_policy": {"kind": "two_sided_interpolation", "weight": float(interpolation.weight),
                "target_scores": "zeros", "ranking": "raw_float32_logits", "positive_bucket": 0, "negative_bucket": 1},
            "feature_mask_sha256": semantic_digest(mask), "feature_clusters_sha256": semantic_digest(clusters),
            "feature_vocab_sha256": semantic_digest(vocab),
            "source_geometry": source, "decoder_tokenizer_checksum": compute_tokenizer_checksum(tokenizer),
            "base_seed": settings.seed, "seed_derivation_version": SELF_CONDITIONING_SEED_VERSION,
            "cache_preselection": preselection,
        },
        "cache": {"resolved_path": str(cache_snapshot.root), "identity": provider.identity.as_dict(),
            "schema_version": provider.manifest["schema_version"],
            "manifest_sha256": hashlib.sha256(cache_snapshot.manifest_bytes).hexdigest(),
            "shards": deepcopy(provider.manifest["shards"]),
            "population_and_preparation": deepcopy(provider.manifest["metadata"]),
            "n_contexts": len(provider)},
        "generation": {"precision_contract": SelfConditionedPrecision.MODEL_OWNED.value,
            "autocast_enabled": t.is_autocast_enabled(device_type), "dtype": cfg.dtype, "compute_dtype": str(COMPUTE_DTYPE), "device_type": device_type,
            "num_samples": cfg.num_samples, "max_new_tokens": cfg.max_new_tokens,
            "temperature": float(cfg.temperature), "top_k": cfg.top_k,
            "batch_size": cfg.batch_size, "greedy_microbatch_size": 1, "greedy_python_tools": False},
        "code": self_conditioned_code_identity(),
    }
    if settings.benchmark_steering is not None:
        strength = settings.benchmark_steering.strength
        conditioning = identity["conditioning"]
        conditioning["selection_policy"] = "benchmark_interpolation"
        conditioning["benchmark_steering"] = {"targets_sha256": benchmark_targets_sha256, "strength": strength}
        score_policy = conditioning["score_policy"]
        assert isinstance(score_policy, dict)
        score_policy.update(target_scores="per_task_bundle", ranking=(
            "raw_float32_logits" if strength == 0 else "target_scores" if strength == 6
            else "lerp_signed_sigmoid_target"))
    # Convert enum/refined numeric values once, into canonical JSON primitives.
    identity = json.loads(canonical_turn_payload(identity))
    validate_self_conditioned_run_identity(identity)
    return identity


def validate_self_conditioned_run_identity(identity) -> None:
    """Require a complete merge identity; runtime admission owns model validation.

    The merger compares the complete canonical payload between shards. It need
    not re-parse checkpoint selection, source geometry or cache tensor counts.
    """
    if not isinstance(identity, dict) or set(identity) != {
        "version", "checkpoint", "conditioning", "cache", "generation", "code",
    } or type(identity["version"]) is not int or identity["version"] != ChatEvalIdentityVersion.SELF_CONDITIONED:
        raise ValueError("self_conditioned requires receipt version 3 and complete identity")
    required = {
        "checkpoint": "metadata_sha256 model_sha256 architecture_sha256 resolved_tag step",
        "conditioning": "evaluation_mode segment_mode recorded_k measurement_provenance_sha256 cadence_fidelity "
            "selection_policy positive_selection negative_selection pool_mode selection_producer bucket_producer score_policy "
            "feature_mask_sha256 feature_clusters_sha256 feature_vocab_sha256 source_geometry decoder_tokenizer_checksum "
            "base_seed seed_derivation_version cache_preselection",
        "cache": "resolved_path identity schema_version manifest_sha256 shards population_and_preparation n_contexts",
        "generation": " ".join(RANGED_GENERATION_SETTING_KEYS | {"precision_contract", "autocast_enabled"}),
        "code": "runtime_version git_commit source_sha256",
    }
    for name, fields in required.items():
        if not isinstance(identity[name], dict) or not set(fields.split()) <= set(identity[name]):
            raise ValueError(f"self_conditioned requires complete {name} identity")
    cond, cache = identity["conditioning"], identity["cache"]
    if type(cond.get("recorded_k")) is not int or cond["recorded_k"] <= 0:
        raise ValueError("self_conditioned recorded_k must be a positive integer")
    steering = cond.get("benchmark_steering")
    if cond["selection_policy"] == "benchmark_interpolation":
        if not isinstance(steering, dict) or set(steering) != {"targets_sha256", "strength"}:
            raise ValueError("benchmark_interpolation requires complete steering identity")
        strength = steering["strength"]
        if (type(strength) not in (int, float) or not 0 <= strength <= 6
                or not isinstance(steering["targets_sha256"], str)
                or re.fullmatch("[0-9a-f]{64}", steering["targets_sha256"]) is None):
            raise ValueError("benchmark_interpolation requires a valid strength and target digest")
        expected_score = {"kind": "two_sided_interpolation", "weight": strength / 6.0,
            "target_scores": "per_task_bundle", "ranking": "raw_float32_logits" if strength == 0 else
            "target_scores" if strength == 6 else "lerp_signed_sigmoid_target", "positive_bucket": 0, "negative_bucket": 1}
        if canonical_turn_payload(cond["score_policy"]) != canonical_turn_payload(expected_score):
            raise ValueError("benchmark_interpolation score policy disagrees with strength")
    elif steering is not None:
        raise ValueError("benchmark steering identity requires benchmark_interpolation")
    mode = SegmentMode(cond["segment_mode"])
    if cache["schema_version"] != (SCHEMA_VERSION if mode is SegmentMode.fixed_tokens else TURN_SCHEMA_VERSION):
        raise ValueError("self_conditioned cache schema disagrees with checkpoint mode")
    generation = identity["generation"]
    if (generation["precision_contract"] != SelfConditionedPrecision.MODEL_OWNED
            or generation["autocast_enabled"] is not False):
        raise ValueError("self_conditioned inference requires model-owned precision with autocast disabled")
    for block, keys in (
        (identity["checkpoint"], ("metadata_sha256", "model_sha256", "architecture_sha256")),
        (cache, ("manifest_sha256",)), (identity["code"], ("source_sha256",)),
    ):
        if any(not isinstance(block.get(key), str) or re.fullmatch("[0-9a-f]{64}", block[key]) is None for key in keys):
            raise ValueError("self_conditioned receipt requires SHA-256 digests")
    json.dumps(identity, allow_nan=False)
