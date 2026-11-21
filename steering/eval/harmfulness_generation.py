"""Generation provenance and immutable artifact identity for harmfulness eval."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from beartype import BeartypeConf, beartype

from steering._common.strict_json import canonical_json
from steering.conditional.sampling import SELFCOND_ACTION_RANKING_POLICY
from steering.immutable_io import StreamedFileDigest, stream_hash_file_descriptor_bound
from steering.refined_types import (
    LowercaseSha256,
    LowercaseSha256Like,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeFiniteNumber,
    NonNegativeFiniteNumberLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
    UnitInterval,
)

GENERATION_CONTRACT_VERSION = 1
ENCODER_DECODER_MODEL_TYPE = "encoder_decoder"
DECODER_ONLY_MODEL_TYPE = "decoder_only"
SEED_SCHEDULE = "base_plus_absolute_prompt_index"
HF_BATCHED_SEED_SCHEDULE = "base_plus_batch_start_hf_batch_v2"
SELF_CONDITIONED_SEED_SCHEDULE = "harmfulness_prompt_rng_v1"
STRENGTH_SWEEP_SPEC_SHA256_KEY = "strength_sweep_spec_sha256"
MAX_TORCH_SEED = 2**64 - 1
_VALUE_ERROR_CONF = BeartypeConf(violation_param_type=ValueError)

_GENERATION_KEYS = {
    "contract_version",
    "condition_name",
    "model",
    "device_type",
    "inference_dtype",
    "feature_conditions",
    "target_conditioning_mode",
    "max_new_tokens",
    "temperature",
    "top_k",
    "batch_size",
    "seed",
    "seed_schedule",
    "render_as_chat",
    "context_cache_manifest_sha256",
    "completion",
}
_MODEL_KEYS = {
    "format",
    "backend",
    "architecture",
    "model_type",
    "step",
    "checkpoint_sha256",
    "tokenizer_sha256",
}
_SELF_CONDITIONED_KEYS = {
    "self_cond_segment_size",
    "strength",
    "action_ranking_policy",
}

class HarmfulnessResponseIdentityError(ValueError):
    """A cached response belongs to a different prompt identity."""


def validate_harmfulness_response_record(
    record: Any,
    expected: Mapping[str, Any],
    *,
    index: int,
    source: str,
    require_exact_identity: bool,
) -> dict[str, Any]:
    """Validate one cached response without deciding whether its cache is admissible."""

    if not isinstance(record, dict):
        raise ValueError(
            f"{source} pair {index} must be a mapping, got {type(record).__name__}"
        )
    response = record.get("response")
    if not isinstance(response, str):
        raise ValueError(
            f"{source} pair {index} response must be a string, "
            f"got {type(response).__name__}"
        )
    actual_identity = {key: value for key, value in record.items() if key != "response"}
    expected_identity = dict(expected)
    if not require_exact_identity:
        actual_identity = {key: record.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise HarmfulnessResponseIdentityError(
            f"{source} pair {index} prompt identity does not match the current "
            f"pinned prompt pool; expected={expected_identity!r}, got={actual_identity!r}"
        )
    return record


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} keys must be exactly {sorted(expected)}, got {sorted(value)}")


@beartype(conf=_VALUE_ERROR_CONF)
def validate_absolute_prompt_seed(
    seed: NonNegativeIntLike,
    prompt_count: NonNegativeIntLike,
) -> int:
    """Validate the base seed and its largest per-prompt derived seed."""

    largest = seed if prompt_count == 0 else seed + prompt_count - 1
    if largest > MAX_TORCH_SEED:
        raise ValueError(
            "generation.seed overflows PyTorch's valid seed range under the "
            f"absolute prompt schedule: base={seed}, prompt_count={prompt_count}, "
            f"largest={largest}, maximum={MAX_TORCH_SEED}"
        )
    return seed


@beartype(conf=_VALUE_ERROR_CONF)
def absolute_prompt_batch_seed(
    seed: NonNegativeIntLike,
    batch_start_index: NonNegativeIntLike,
) -> int:
    """Return the engine seed for one outer batch of the absolute schedule."""

    result = seed + batch_start_index
    if result > MAX_TORCH_SEED:
        raise ValueError(
            "absolute prompt seed exceeds PyTorch's valid range: "
            f"base={seed}, batch_start_index={batch_start_index}, maximum={MAX_TORCH_SEED}"
        )
    return result


def validate_harmfulness_completion_provenance(
    completion: Any, *, feature_conditions: Mapping | None = None,
) -> dict[str, Any]:
    """Validate legacy and dense safety completion receipts for every consumer."""
    if not isinstance(completion, dict):
        raise ValueError("generation.completion must be a mapping or null")
    # Historical receipts remain readable; new receipts always bind the
    # optional dense safety bundle, even at the zero-strength endpoint.
    expected_keys = _SELF_CONDITIONED_KEYS | ({"safety_steering"} if "safety_steering" in completion else set())
    _require_exact_keys(completion, expected_keys, "generation.completion")
    try:
        PositiveInt.parse(completion["self_cond_segment_size"])
        if "safety_steering" not in completion:
            UnitInterval.parse(completion["strength"])
        else:
            strength = completion["strength"]
            if type(strength) not in (float, int) or not 0 <= strength <= 6:
                raise ValueError("generation.completion.strength must lie in [0, 6]")
            safety = completion["safety_steering"]
            if safety is None:
                if strength != 0:
                    raise ValueError("unsteered self-conditioning requires strength=0")
            else:
                if not isinstance(safety, dict):
                    raise ValueError("generation.completion.safety_steering must be a mapping or null")
                _require_exact_keys(safety, {"targets_sha256", "strength"}, "generation.completion.safety_steering")
                LowercaseSha256.parse(safety["targets_sha256"])
                if type(safety["strength"]) not in (float, int) or safety["strength"] != strength:
                    raise ValueError("safety_steering strength must equal completion strength")
            if feature_conditions is not None:
                raise ValueError("dense safety steering cannot mix with feature_conditions")
    except TypeError as exc:
        raise ValueError(f"invalid generation.completion scalar: {exc}") from exc
    if completion["action_ranking_policy"] != SELFCOND_ACTION_RANKING_POLICY:
        raise ValueError(
            "generation.completion action_ranking_policy must equal "
            f"{SELFCOND_ACTION_RANKING_POLICY!r}"
        )
    return completion


def validate_generation_provenance(
    generation: Any,
    *,
    prompt_count: int | None = None,
) -> dict[str, Any]:
    """Validate the exact schema-v2 generation block and return it unchanged."""

    if not isinstance(generation, dict):
        raise ValueError(f"generation must be a mapping, got {type(generation).__name__}")
    _require_exact_keys(generation, _GENERATION_KEYS, "generation")
    try:
        PositiveInt.parse(generation["contract_version"])
        NonEmptyStr.parse(generation["condition_name"])
    except TypeError as exc:
        raise ValueError(f"invalid generation scalar: {exc}") from exc
    if generation["contract_version"] != GENERATION_CONTRACT_VERSION:
        raise ValueError(
            "generation.contract_version must equal "
            f"{GENERATION_CONTRACT_VERSION}, got {generation['contract_version']!r}"
        )

    model = generation["model"]
    if not isinstance(model, dict):
        raise ValueError(f"generation.model must be a mapping, got {type(model).__name__}")
    _require_exact_keys(model, _MODEL_KEYS, "generation.model")
    model_format = model["format"]
    try:
        for field in ("format", "backend", "architecture", "model_type"):
            NonEmptyStr.parse(model[field])
        NonNegativeInt.parse(model["step"])
        LowercaseSha256.parse(model["checkpoint_sha256"])
        LowercaseSha256.parse(model["tokenizer_sha256"])
        NonNegativeInt.parse(generation["max_new_tokens"])
        PositiveInt.parse(generation["batch_size"])
        if generation["top_k"] is not None:
            PositiveInt.parse(generation["top_k"])
        if generation["context_cache_manifest_sha256"] is not None:
            LowercaseSha256.parse(generation["context_cache_manifest_sha256"])
    except TypeError as exc:
        raise ValueError(f"invalid generation scalar: {exc}") from exc
    temperature = generation["temperature"]
    if type(temperature) is not float or not isinstance(temperature, NonNegativeFiniteNumber):
        raise ValueError("generation.temperature must be a finite non-negative float")
    if model_format not in {"nanochat_checkpoint", "repository_hf_sft"}:
        raise ValueError(f"unsupported generation model format {model_format!r}")
    if model_format == "nanochat_checkpoint" and model["backend"] != "nanochat":
        raise ValueError("generation.model.backend must be 'nanochat' for native checkpoints")
    if generation["device_type"] not in {"cpu", "cuda", "mps"}:
        raise ValueError("unsupported generation device_type")
    if generation["inference_dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("unsupported generation inference_dtype")
    if generation["target_conditioning_mode"] not in {"none", "current", "next"}:
        raise ValueError("unsupported generation target_conditioning_mode")
    if generation["seed_schedule"] not in {
        SEED_SCHEDULE,
        HF_BATCHED_SEED_SCHEDULE,
        SELF_CONDITIONED_SEED_SCHEDULE,
    }:
        raise ValueError("unsupported generation seed_schedule")
    if generation["render_as_chat"] is not True:
        raise ValueError("generation.render_as_chat must be exactly true")

    feature_conditions = generation["feature_conditions"]
    if feature_conditions is not None:
        if not isinstance(feature_conditions, dict):
            raise ValueError("generation.feature_conditions must be a mapping or null")
        for feature_id, score in feature_conditions.items():
            if (
                not isinstance(feature_id, NonNegativeInt)
                or type(score) is not int
                or abs(score) != 1
            ):
                raise ValueError(
                    "generation.feature_conditions must map non-negative integer "
                    "ids to signed scores in {-1, +1}"
                )
        if list(feature_conditions) != sorted(feature_conditions):
            raise ValueError("generation.feature_conditions keys must be integers in sorted order")

    target_mode = generation["target_conditioning_mode"]
    if model_format == "repository_hf_sft" and (
        feature_conditions is not None or target_mode != "none"
    ):
        raise ValueError("repository HF harmfulness generation does not support conditioning")

    seed_schedule = generation["seed_schedule"]
    if prompt_count is not None and seed_schedule in {SEED_SCHEDULE, HF_BATCHED_SEED_SCHEDULE}:
        validate_absolute_prompt_seed(generation["seed"], prompt_count)
    else:
        validate_absolute_prompt_seed(generation["seed"], 1)

    completion = generation["completion"]
    if completion is not None:
        validate_harmfulness_completion_provenance(completion, feature_conditions=feature_conditions)
        if target_mode != "next":
            raise ValueError(
                "self-conditioned harmfulness generation requires target_conditioning_mode='next'"
            )
        if seed_schedule != SELF_CONDITIONED_SEED_SCHEDULE:
            raise ValueError(
                "self-conditioned harmfulness generation requires its prompt-identity seed schedule"
            )
    elif model_format == "repository_hf_sft":
        if seed_schedule != HF_BATCHED_SEED_SCHEDULE:
            raise ValueError(
                "repository HF harmfulness generation requires the batched outer-batch "
                "seed schedule"
            )
    elif seed_schedule != SEED_SCHEDULE:
        raise ValueError("native harmfulness generation requires the absolute prompt seed schedule")
    return generation


@beartype(conf=_VALUE_ERROR_CONF)
def build_generation_provenance(
    *,
    condition_name: NonEmptyStrLike,
    model: Mapping,
    device_type: Literal["cpu", "cuda", "mps"],
    inference_dtype: Literal["float32", "bfloat16"],
    feature_conditions: Mapping | None,
    target_conditioning_mode: Literal["none", "current", "next"],
    max_new_tokens: NonNegativeIntLike,
    temperature: NonNegativeFiniteNumberLike,
    top_k: PositiveIntLike | None,
    batch_size: PositiveIntLike,
    seed: NonNegativeIntLike,
    prompt_count: NonNegativeIntLike,
    seed_schedule: Literal[
        "base_plus_absolute_prompt_index",
        "base_plus_batch_start_hf_batch_v2",
        "harmfulness_prompt_rng_v1",
    ] = SEED_SCHEDULE,
    context_cache_manifest_sha256: LowercaseSha256Like | None = None,
    completion: Mapping | None = None,
) -> dict[str, Any]:
    """Normalize and validate the persisted harmfulness generation contract."""

    generation = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "condition_name": condition_name,
        "model": dict(model),
        "device_type": device_type,
        "inference_dtype": inference_dtype,
        "feature_conditions": (
            None if not feature_conditions else dict(sorted(feature_conditions.items()))
        ),
        "target_conditioning_mode": target_conditioning_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": float(temperature),
        "top_k": top_k,
        "batch_size": batch_size,
        "seed": seed,
        "seed_schedule": seed_schedule,
        "render_as_chat": True,
        "context_cache_manifest_sha256": context_cache_manifest_sha256,
        "completion": None if completion is None else dict(completion),
    }
    return validate_generation_provenance(generation, prompt_count=prompt_count)


@dataclass(frozen=True)
class ArtifactInventory:
    root: Path
    domain: str
    entries: tuple[StreamedFileDigest, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "domain": self.domain,
            "files": [
                {
                    "name": entry.path.relative_to(self.root).as_posix(),
                    "byte_length": entry.byte_length,
                    "sha256": entry.sha256,
                }
                for entry in self.entries
            ],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def assert_unchanged(self, observed_paths: Sequence[Path] | None = None) -> None:
        if observed_paths is not None:
            observed = tuple(path.relative_to(self.root).as_posix() for path in observed_paths)
            expected = tuple(entry.path.relative_to(self.root).as_posix() for entry in self.entries)
            if observed != expected:
                raise ValueError(f"{self.domain} inventory membership changed while loading")
        for entry in self.entries:
            if _regular_file_signature(entry.path, domain=self.domain) != entry.stat_signature:
                raise ValueError(f"{self.domain} inventory changed while loading: {entry.path}")


def _require_real_directory(path: Path, *, surface: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{surface} directory is missing or unreadable: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{surface} directory must not be a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{surface} path must be a directory: {path}")
    return path.resolve(strict=True)


def _regular_file_signature(path: Path, *, domain: str) -> tuple[int, int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{domain} inventory file is missing or unreadable: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{domain} inventory requires regular non-symlink files: {path}")
    return (
        int(info.st_mode),
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def capture_file_inventory(
    root: Path,
    paths: Sequence[Path],
    *,
    domain: str,
) -> ArtifactInventory:
    """Hash an explicit sorted set of direct regular files below ``root``."""

    root = _require_real_directory(root, surface=domain)
    normalized = tuple(sorted((Path(path) for path in paths), key=lambda item: str(item)))
    if not normalized:
        raise ValueError(f"{domain} inventory must contain at least one file")
    entries: list[StreamedFileDigest] = []
    seen: set[str] = set()
    for path in normalized:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{domain} inventory file escapes its root: {path}") from exc
        if relative in seen:
            raise ValueError(f"{domain} inventory contains duplicate path {relative!r}")
        seen.add(relative)
        _regular_file_signature(path, domain=domain)
        entries.append(stream_hash_file_descriptor_bound(path))
    return ArtifactInventory(root=root, domain=domain, entries=tuple(entries))


def recursive_file_paths(root: Path, *, exclude_top_level: set[str] | None = None) -> list[Path]:
    """List a symlink-free recursive file tree without following links."""

    root = _require_real_directory(root, surface="artifact")
    excluded = set(exclude_top_level or ())
    paths: list[Path] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            dirnames[:] = [name for name in dirnames if name not in excluded]
        for dirname in list(dirnames):
            child = current_path / dirname
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"artifact inventory contains symlink directory: {child}")
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"artifact inventory contains non-directory entry: {child}")
        for filename in filenames:
            path = current_path / filename
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"artifact inventory requires regular non-symlink files: {path}")
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


@dataclass(frozen=True)
class NativeArtifactIdentity:
    checkpoint_dir: Path
    step: int
    checkpoint_inventory: ArtifactInventory
    tokenizer_inventory: ArtifactInventory

    def assert_unchanged(self) -> None:
        self.checkpoint_inventory.assert_unchanged()
        self.tokenizer_inventory.assert_unchanged(
            recursive_file_paths(self.tokenizer_inventory.root)
        )


@beartype(conf=_VALUE_ERROR_CONF)
def native_checkpoint_metadata_path(checkpoint_dir: str | Path, step: NonNegativeIntLike) -> Path:
    """The native metadata filename used by both loading and model-free routing."""
    return Path(checkpoint_dir) / f"meta_{step:06d}.json"


@beartype(conf=_VALUE_ERROR_CONF)
def capture_native_artifact_identity(
    checkpoint_dir: str | Path,
    *,
    step: NonNegativeIntLike,
    tokenizer_dir: str | Path,
) -> NativeArtifactIdentity:
    """Capture the exact native weight/meta pair and external tokenizer tree."""

    # Native operators commonly expose immutable scratch artifacts through a
    # directory symlink. Resolve that declared root once; inventory entries
    # below the concrete root must still be real files/directories.
    root = _require_real_directory(
        Path(checkpoint_dir).resolve(),
        surface="native checkpoint",
    )
    checkpoint_paths = [
        root / f"model_{step:06d}.pt",
        native_checkpoint_metadata_path(root, step),
    ]
    checkpoint_inventory = capture_file_inventory(
        root,
        checkpoint_paths,
        domain="harmfulness_native_checkpoint_v1",
    )
    tokenizer_root = _require_real_directory(
        Path(tokenizer_dir).resolve(),
        surface="native tokenizer",
    )
    tokenizer_inventory = capture_file_inventory(
        tokenizer_root,
        recursive_file_paths(tokenizer_root),
        domain="harmfulness_native_tokenizer_v1",
    )
    return NativeArtifactIdentity(
        checkpoint_dir=root,
        step=step,
        checkpoint_inventory=checkpoint_inventory,
        tokenizer_inventory=tokenizer_inventory,
    )


@beartype(conf=_VALUE_ERROR_CONF)
def native_model_identity(
    artifacts: NativeArtifactIdentity,
    *,
    architecture: NonEmptyStrLike,
    model_type: NonEmptyStrLike,
    effective_metadata: Mapping,
) -> dict[str, Any]:
    """Build the persisted content identity after the native model is loaded."""

    # The loader may inherit configuration from ancestor metadata. Hash the
    # resulting metadata it actually used instead of reimplementing that walk.
    loaded_metadata = {
        key: value for key, value in effective_metadata.items() if key != "checkpoint_dir"
    }
    checkpoint_sha256 = hashlib.sha256(
        canonical_json(
            {
                "domain": "harmfulness_native_loaded_model_v1",
                "checkpoint_inventory_sha256": artifacts.checkpoint_inventory.sha256,
                "effective_metadata": loaded_metadata,
            }
        ).encode("utf-8")
    ).hexdigest()
    model = {
        "format": "nanochat_checkpoint",
        "backend": "nanochat",
        "architecture": architecture,
        "model_type": model_type,
        "step": artifacts.step,
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": artifacts.tokenizer_inventory.sha256,
    }
    return model
