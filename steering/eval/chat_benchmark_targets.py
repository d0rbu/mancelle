"""Frozen benchmark targets and their checkpoint/population admission contract.

This module consumes artifacts only. Dataset preparation and SAE extraction
belong to the offline builder, never to an evaluation process.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Self

import torch as t
from safetensors.torch import load

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfCondScoreInterpolation,
)
from steering.conditional.feature_mask import canonical_dict
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound
from steering.refined_types import UnitInterval
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    read_safetensors_metadata_from_bytes,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum

TARGET_FORMAT_VERSION = 1
TARGET_METADATA_KEY = "chatcore_benchmark_targets"


class BackgroundOverlapPolicy(StrEnum):
    EMBEDDED = "embedded"
    EXACT_MESSAGE = "exact_message"


def normalize_input(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class BenchmarkSteeringSettings:
    block_name: ClassVar[str] = "benchmark_steering"
    targets: Path
    strength: float

    @classmethod
    def parse(cls, value: Any) -> Self:
        if not isinstance(value, Mapping) or set(value) != {"targets", "strength"}:
            raise ValueError(f"{cls.block_name} requires exactly targets and strength")
        path, strength = value["targets"], value["strength"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{cls.block_name}.targets must be a nonblank path")
        if type(strength) not in (int, float) or not 0 <= strength <= 6:
            raise ValueError(f"{cls.block_name}.strength must be a finite number in [0, 6]")
        return cls(Path(path), 0.0 if strength == 0 else float(strength))


def benchmark_fit_contract(
    *, feature_contract: CheckpointFeatureContract, n_sae_features: int, tokenizer: Any,
    turn_contract: Any = None, fixed_contract: Any = None,
) -> tuple[dict[str, Any], t.Tensor]:
    """Use the same resolved feature/source authority for fitting and evaluation."""
    if (turn_contract is None) == (fixed_contract is None):
        raise ValueError("benchmark targets require exactly one resolved source contract")
    vocab = effective_feature_vocab(feature_contract.universe, n_sae_features)
    original_ids = (t.arange(n_sae_features, dtype=t.int64) if vocab is None
                    else t.tensor(vocab.original_ids, dtype=t.int64))
    mode = SegmentMode.fixed_tokens if fixed_contract is not None else SegmentMode.turn_delimited_fixed_tokens
    contract = {
        "segment_mode": mode.value,
        "source": fixed_contract.source if fixed_contract is not None else turn_contract.semantic,
        "statistics_sha256": (fixed_contract.statistics_sha256 if fixed_contract is not None
                              else turn_contract.lineage.statistics_sha256),
        "decoder_tokenizer_checksum": compute_tokenizer_checksum(tokenizer),
        "n_sae_features": n_sae_features,
        "feature_mask": canonical_dict(feature_contract.feature_mask),
        "feature_clusters": (None if feature_contract.resolved_clusters is None
                             else feature_contract.resolved_clusters.descriptor_dict()),
        "feature_vocab": None if vocab is None else vocab.metadata_dict(),
        "missing_percentile_policy": ("error" if fixed_contract is not None
                                      else turn_contract.lineage.missing_percentile_policy),
    }
    return json.loads(canonical_turn_payload(contract)), original_ids


@dataclass(frozen=True)
class BenchmarkTargets:
    sha256: str
    targets: dict[str, t.Tensor]
    evaluation_prompt_keys: dict[str, tuple[str, ...]]

    def validate_example(self, name: str, task_size: int, index: int, token_ids: Sequence[int]) -> None:
        keys = self.evaluation_prompt_keys[name]
        if task_size != len(keys) or not 0 <= index < len(keys) or token_context_key(token_ids) != keys[index]:
            raise ValueError(f"{name} problem {index}: benchmark exclusion population mismatch")

    def policy(self, base: SelfCondConfig, name: str, strength: float) -> SelfCondConfig:
        if not isinstance(base.score_policy, SelfCondScoreInterpolation):
            raise ValueError("benchmark steering requires the existing interpolation policy")
        return replace(base, score_policy=replace(base.score_policy,
            target_scores=self.targets[name], weight=UnitInterval.parse(strength / 6.0)))


def decode_benchmark_targets(
    payload: bytes, *, expected_contract: dict, expected_feature_ids: t.Tensor,
    task_names: Sequence[str],
) -> BenchmarkTargets:
    """Hash, parse and validate the same captured bytes, before any forward."""
    metadata = read_safetensors_metadata_from_bytes(payload)
    if metadata is None or set(metadata) != {TARGET_METADATA_KEY}:
        raise ValueError("benchmark target metadata is missing or has unknown keys")
    manifest = json.loads(metadata[TARGET_METADATA_KEY])
    fields = {"format_version", "fit_contract", "preparation", "evaluation_prompt_keys", "fit_provenance"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("benchmark target manifest has missing or unknown sections")
    if canonical_turn_payload(manifest).decode() != metadata[TARGET_METADATA_KEY]:
        raise ValueError("benchmark target manifest must be canonical JSON")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != TARGET_FORMAT_VERSION:
        raise ValueError("unsupported benchmark target format version")
    if canonical_turn_payload(manifest["fit_contract"]) != canonical_turn_payload(expected_contract):
        raise ValueError("benchmark target fit contract disagrees with checkpoint")
    for name in ("preparation", "fit_provenance"):
        if not isinstance(manifest[name], dict) or not manifest[name]:
            raise ValueError(f"benchmark targets require {name}")
    population = manifest["evaluation_prompt_keys"]
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("benchmark targets require unique task names")
    if not isinstance(population, dict) or set(population) != set(task_names):
        raise ValueError("benchmark target evaluation population tasks disagree")
    for name, keys in population.items():
        if (not isinstance(keys, list) or not keys
                or any(not isinstance(key, str) or re.fullmatch(r"tokens_sha256:[0-9a-f]{64}", key) is None for key in keys)):
            raise ValueError(f"benchmark targets require valid ordered prompt keys for {name}")
    tensors = load(payload)
    if set(tensors) != {"feature_ids", *(f"target.{name}" for name in task_names)}:
        raise ValueError("benchmark target tensor names disagree")
    ids = tensors["feature_ids"]
    if ids.dtype != t.int64 or ids.ndim != 1 or ids.numel() == 0 or not t.equal(ids, expected_feature_ids):
        raise ValueError("benchmark target original feature IDs disagree with FP-head row order")
    targets = {}
    for name in task_names:
        target = tensors[f"target.{name}"]
        if (target.dtype != t.float32 or target.shape != ids.shape or not bool(t.isfinite(target).all())
                or not bool(((target > -1) & (target < 1)).all())):
            raise ValueError(f"benchmark target {name} must be finite float32[V] strictly inside (-1, 1)")
        targets[name] = target.detach().clone().contiguous()
    return BenchmarkTargets(hashlib.sha256(payload).hexdigest(), targets,
                            {name: tuple(keys) for name, keys in population.items()})


def load_benchmark_targets(path: Path, **expected: Any) -> BenchmarkTargets:
    snapshot = stream_hash_file_descriptor_bound(path.resolve(strict=True), retain_payload=True)
    if snapshot.payload is None:
        raise RuntimeError("benchmark target snapshot did not retain its bytes")
    return decode_benchmark_targets(snapshot.payload, **expected)


def write_benchmark_targets(
    path: Path, *, targets: dict[str, t.Tensor], feature_ids: t.Tensor, manifest: dict,
    task_names: Sequence[str],
) -> BenchmarkTargets:
    payload = deterministic_safetensors_bytes(
        {"feature_ids": feature_ids, **{f"target.{name}": value for name, value in targets.items()}},
        metadata={TARGET_METADATA_KEY: canonical_turn_payload(manifest).decode()},
    )
    bundle = decode_benchmark_targets(payload, expected_contract=manifest["fit_contract"],
                                     expected_feature_ids=feature_ids, task_names=task_names)
    install_bytes_once(payload, path, surface="ChatCORE benchmark targets")
    return bundle
