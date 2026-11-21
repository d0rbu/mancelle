"""Held-out safe-response targets using the shared dense bundle and score policy."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import torch as t
from datasets import load_dataset

from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.eval.chat_benchmark_targets import (
    TARGET_METADATA_KEY,
    BackgroundOverlapPolicy,
    BenchmarkSteeringSettings,
    BenchmarkTargets,
    decode_benchmark_targets,
    normalize_input,
    write_benchmark_targets,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.dataset_revision import validate_dataset_revision
from steering.eval.harmfulness_judge import (
    BEAVERTAILS_EVALUATION_REVISION,
    WILDGUARD_MIX_CONFIG,
    WILDGUARD_MIX_DATASET_ID,
    WILDGUARD_MIX_REVISION,
    WILDGUARD_MIX_SPLIT,
)
from steering.eval.target_conditioning import token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.feature_io import ExtractionBackend
from steering.immutable_io import stream_hash_file_descriptor_bound
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.safetensors_io import read_safetensors_metadata_from_bytes
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from tasks.smoltalk import SmolTalk

BACKGROUND_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
TRAIN_CONFIG = "wildguardtrain"
TRAIN_SPLIT = "train"
FIT_SHUFFLE_SEED = 42
UNRECORDED_SOURCE_WEIGHT_REVISION = "unrecorded_by_existing_loader"
SAFETY_RENDER_REDUCTION = "native_complete_chat_bos_excluded_document_percentile_diffmean_v1"
SAFETY_FIT_ORDER = "numpy_default_rng_permutation_original_rows_v1"
SAFETY_FIT_SOURCE = "wildguard_train_safe_responses_v2"
SAFETY_TARGET_NAME = "safety"


class SafetySubpopulation(StrEnum):
    SAFE_HARMFUL = "harmful_prompt_safe_response"
    BENIGN_NON_REFUSAL = "benign_prompt_safe_non_refusal"


class PromptHarmLabel(StrEnum):
    HARMFUL = "harmful"
    BENIGN = "unharmful"


class ResponseRefusalLabel(StrEnum):
    REFUSAL = "refusal"
    NON_REFUSAL = "compliance"


class SafeResponseLabel(StrEnum):
    SAFE = "unharmful"
    UNSAFE = "harmful"


class ExclusionReason(StrEnum):
    EMPTY_PROMPT = "empty_prompt"
    EVALUATION = "evaluation_overlap"
    UNSAFE = "unsafe_response"
    UNLABELLED = "unlabelled_response"
    UNLABELLED_PROMPT = "unlabelled_prompt"
    UNLABELLED_REFUSAL = "unlabelled_benign_refusal"
    BENIGN_REFUSAL = "benign_refusal"
    EMPTY = "empty_response"
    DUPLICATE = "duplicate_input"
    OVERLENGTH = "source_overlength"
    CAP = "positive_cap"
    SOURCE_OVERLAP = "source_overlap"


def safety_fit_selection(row: Mapping[str, Any]) -> SafetySubpopulation | ExclusionReason:
    """The sole v2 label rule, shared by preparation and bundle admission.

    Null is an explicitly unlabelled value, not an unknown annotation. Validate
    every field even when another label already excludes this row.
    """
    for field, enum in (("prompt_harm_label", PromptHarmLabel),
                        ("response_harm_label", SafeResponseLabel),
                        ("response_refusal_label", ResponseRefusalLabel)):
        if field not in row:
            raise ValueError(f"safety fit row missing {field}")
        value = row[field]
        if value is not None and (not isinstance(value, str) or value not in tuple(enum)):
            raise ValueError(f"safety fit row unknown {field} {value!r}")
    if row["response_harm_label"] is None:
        return ExclusionReason.UNLABELLED
    if row["response_harm_label"] == SafeResponseLabel.UNSAFE:
        return ExclusionReason.UNSAFE
    if row["prompt_harm_label"] is None:
        return ExclusionReason.UNLABELLED_PROMPT
    if row["prompt_harm_label"] == PromptHarmLabel.HARMFUL:
        return SafetySubpopulation.SAFE_HARMFUL
    if row["response_refusal_label"] is None:
        return ExclusionReason.UNLABELLED_REFUSAL
    if row["response_refusal_label"] == ResponseRefusalLabel.REFUSAL:
        return ExclusionReason.BENIGN_REFUSAL
    return SafetySubpopulation.BENIGN_NON_REFUSAL


class SafetySourceName(StrEnum):
    TRAINING = "training"
    EVALUATION_BEAVERTAILS = "evaluation_beavertails"
    EVALUATION_WILDGUARD = "evaluation_wildguard"
    BACKGROUND = "background"


@dataclass(frozen=True)
class SafetySourceSpec:
    dataset: str
    config: str
    split: str
    revision: str

    def metadata(self, rows: int) -> dict[str, Any]:
        return {"dataset": self.dataset, "config": self.config, "split": self.split,
                "revision": self.revision, "rows": rows}


SAFETY_SOURCE_SPECS: Mapping[SafetySourceName, SafetySourceSpec] = MappingProxyType({
    SafetySourceName.TRAINING: SafetySourceSpec(WILDGUARD_MIX_DATASET_ID, TRAIN_CONFIG, TRAIN_SPLIT, WILDGUARD_MIX_REVISION),
    SafetySourceName.EVALUATION_BEAVERTAILS: SafetySourceSpec("PKU-Alignment/BeaverTails-Evaluation", "default", "test", BEAVERTAILS_EVALUATION_REVISION),
    SafetySourceName.EVALUATION_WILDGUARD: SafetySourceSpec(WILDGUARD_MIX_DATASET_ID, WILDGUARD_MIX_CONFIG, WILDGUARD_MIX_SPLIT, WILDGUARD_MIX_REVISION),
    SafetySourceName.BACKGROUND: SafetySourceSpec("HuggingFaceTB/smol-smoltalk", "default", TRAIN_SPLIT, BACKGROUND_REVISION),
})
SAFETY_EVALUATION_SOURCE_NAMES = (
    SafetySourceName.EVALUATION_BEAVERTAILS,
    SafetySourceName.EVALUATION_WILDGUARD,
)


def load_populations(
    names: Sequence[SafetySourceName] = tuple(SafetySourceName),
) -> tuple[dict[SafetySourceName, Any], dict[str, dict[str, Any]]]:
    """Load pinned safety populations with cache-revision attestation."""
    selected = tuple(names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("safety population selection must be nonempty and unique")
    loaded: dict[SafetySourceName, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for name in selected:
        spec = SAFETY_SOURCE_SPECS[name]
        if name is SafetySourceName.BACKGROUND:
            loaded[name] = SmolTalk(spec.split, revision=spec.revision)
            rows = loaded[name].ds
        else:
            args = () if name is SafetySourceName.EVALUATION_BEAVERTAILS else (spec.config,)
            rows = load_dataset(
                spec.dataset,
                *args,
                split=spec.split,
                revision=spec.revision,
            )
            loaded[name] = rows
        validate_dataset_revision(rows, spec.revision)
        provenance[name.value] = spec.metadata(len(rows))
    return loaded, provenance


def safety_evaluation_input_keys(evaluation: Sequence[Mapping]) -> set[str]:
    """Normalize the full exclusion population shared by safety target producers."""
    forbidden = set()
    for row in evaluation:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("evaluation prompt must be nonblank text")
        forbidden.add(normalize_input(prompt))
    if not forbidden:
        raise ValueError("evaluation exclusion population must be nonempty")
    return forbidden


def _positive_integer(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"safety target {field} must be a positive integer")
    return value


def _digest(value: Any, field: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"safety target {field} must be a {length}-character hexadecimal digest")
    return value


def _source_row_index(row: Mapping[str, Any], *, background: bool) -> int:
    source_id = row.get("source_id")
    prefix = f"{SafetySourceName.BACKGROUND if background else TRAIN_CONFIG}/{TRAIN_SPLIT}/"
    if not isinstance(source_id, str) or not source_id.startswith(prefix):
        raise ValueError("safety target fit audit source_id disagrees with its training source")
    suffix = source_id.removeprefix(prefix)
    if not suffix.isdecimal() or str(int(suffix)) != suffix:
        raise ValueError("safety target source_id requires a canonical nonnegative row index")
    return int(suffix)


def validate_safety_sources(sources: Any) -> None:
    if not isinstance(sources, dict) or set(sources) != set(SAFETY_SOURCE_SPECS):
        raise ValueError("safety target requires pinned training, evaluation and background sources")
    for name, spec in SAFETY_SOURCE_SPECS.items():
        source = sources[name]
        if not isinstance(source, dict) or set(source) != set(spec.metadata(1)):
            raise ValueError(f"safety target invalid source provenance: {name}")
        rows = _positive_integer(source["rows"], f"source {name} rows")
        if source != spec.metadata(rows):
            raise ValueError(f"safety target source {name} disagrees with pinned dataset/config/split/revision")


@dataclass(frozen=True)
class SafetySteeringSettings(BenchmarkSteeringSettings):
    block_name: ClassVar[str] = "safety_steering"


def validate_safety_preparation(preparation: dict) -> None:
    """Require positive safe references and an explicit, disjoint evaluation audit."""
    if isinstance(preparation, dict) and preparation.get("source_kind") != SAFETY_FIT_SOURCE:
        raise ValueError(f"stale or unsupported safety fit rule: refit with {SAFETY_FIT_SOURCE} "
                         "(safe harmful-prompt responses plus safe benign non-refusals)")
    fields = {"source_kind", "training_rows", "positive_cap", "background_count", "background_overlap_policy",
              "evaluation_rows", "evaluation_input_keys", "evaluation_prompts", "accepted", "excluded",
              "exclusion_counts", "background_excluded", "background_exclusion_counts",
              "fit_order", "shuffle_seed", "subpopulation_counts"}
    if not isinstance(preparation, dict) or set(preparation) not in (fields, fields | {"sources"}):
        raise ValueError("safety target preparation has missing or unknown audit fields")
    if (preparation["fit_order"] != SAFETY_FIT_ORDER
            or type(preparation["shuffle_seed"]) is not int or preparation["shuffle_seed"] != FIT_SHUFFLE_SEED):
        raise ValueError("safety target requires natural order after the seed-42 shuffle")
    for field in ("training_rows", "positive_cap", "background_count", "evaluation_rows"):
        _positive_integer(preparation[field], field)
    if preparation["background_overlap_policy"] != BackgroundOverlapPolicy.EXACT_MESSAGE:
        raise ValueError("safety target requires exact-message background exclusion")
    raw_prompts = preparation["evaluation_prompts"]
    if (not isinstance(raw_prompts, list) or not raw_prompts
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in raw_prompts)
            or len(raw_prompts) != preparation["evaluation_rows"]):
        raise ValueError("safety target requires the full raw evaluation prompt population and matching count")
    evaluation = preparation["evaluation_input_keys"]
    accepted = preparation["accepted"]
    if evaluation != sorted({normalize_input(prompt) for prompt in raw_prompts}):
        raise ValueError("safety target evaluation exclusion keys disagree with raw evaluation prompts")
    if (not isinstance(accepted, dict) or set(accepted) != {SAFETY_TARGET_NAME, SafetySourceName.BACKGROUND}
            or any(not isinstance(rows, list) or not rows for rows in accepted.values())):
        raise ValueError("safety target requires nonempty fit and background populations")
    forbidden = set(evaluation)
    # Validate fit first so background exclusion is independent of JSON key order.
    for name in (SAFETY_TARGET_NAME, SafetySourceName.BACKGROUND):
        rows = accepted[name]
        row_fields = {"source_id", "input_key", "text_sha256"}
        if name == SAFETY_TARGET_NAME:
            row_fields |= {"prompt_harm_label", "response_harm_label", "response_refusal_label",
                           "subpopulation", "response_sha256"}
        keys = set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != row_fields
                    or not isinstance(row["input_key"], str) or not row["input_key"]):
                raise ValueError("safety target fit audit has missing provenance")
            _digest(row["text_sha256"], "text_sha256")
            key = row["input_key"]
            if normalize_input(key) != key:
                raise ValueError("safety target input keys must be normalized")
            if key in forbidden:
                raise ValueError("safety target fit/background populations have overlap")
            if key in keys:
                raise ValueError("safety target fit set has duplicate input")
            keys.add(key)
            if name == SAFETY_TARGET_NAME:
                selection = safety_fit_selection(row)
                if not isinstance(selection, SafetySubpopulation) or row["subpopulation"] != selection:
                    raise ValueError("safety target positive violates the v2 fit selection/subpopulation rule")
                _digest(row["response_sha256"], "response_sha256")
        forbidden.update(keys)
    if len(accepted[SAFETY_TARGET_NAME]) > preparation["positive_cap"]:
        raise ValueError("safety target accepted fit population exceeds positive cap")
    counts = preparation["subpopulation_counts"]
    expected = Counter(row["subpopulation"] for row in accepted[SAFETY_TARGET_NAME])
    if (not isinstance(counts, dict) or set(counts) != set(SafetySubpopulation)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts != {name.value: expected[name] for name in SafetySubpopulation}):
        raise ValueError("safety target subpopulation_counts disagree with accepted rows")
    if len(accepted[SafetySourceName.BACKGROUND]) != preparation["background_count"]:
        raise ValueError("safety target accepted background count disagrees with audit")
    for background in (False, True):
        rejected_field = "background_excluded" if background else "excluded"
        count_field = "background_exclusion_counts" if background else "exclusion_counts"
        excluded = preparation[rejected_field]
        allowed_reasons = ({ExclusionReason.SOURCE_OVERLAP, ExclusionReason.DUPLICATE, ExclusionReason.OVERLENGTH}
                           if background else set(ExclusionReason) - {ExclusionReason.SOURCE_OVERLAP})
        if (not isinstance(excluded, list) or any(not isinstance(row, dict) or set(row) != {"source_id", "reason"}
                or not isinstance(row["reason"], str) or row["reason"] not in allowed_reasons for row in excluded)):
            raise ValueError(f"safety target {rejected_field} contains malformed exclusions")
        counts = preparation[count_field]
        if (not isinstance(counts, dict) or any(type(value) is not int or value <= 0 for value in counts.values())
                or counts != dict(Counter(row["reason"] for row in excluded))):
            raise ValueError(f"safety target {count_field} disagrees with excluded rows")
        rows = accepted[SafetySourceName.BACKGROUND if background else SAFETY_TARGET_NAME]
        indices = [_source_row_index(row, background=background) for row in [*rows, *excluded]]
        expected_count = len(indices) if background else preparation["training_rows"]
        if sorted(indices) != list(range(expected_count)):
            raise ValueError("safety target source rows must account for each scanned row exactly once")
        scan_order = (range(expected_count) if background else
                      np.random.default_rng(FIT_SHUFFLE_SEED).permutation(expected_count))
        positions = {int(index): position for position, index in enumerate(scan_order)}
        for population in (indices[:len(rows)], indices[len(rows):]):
            order = [positions[index] for index in population]
            if order != sorted(order):
                raise ValueError("safety target source order disagrees with the recorded scan policy")


def safety_evaluation_prompt_keys(prompts: Sequence[str], tokenizer: Any) -> list[str]:
    # Runtime benign loading strips text; BeaverTails preserves source text.
    return sorted({token_context_key(render_prompt_for_chat_completion(tokenizer, text).token_ids)
                   for prompt in prompts for text in (prompt, prompt.strip())})


def _validate_population(manifest: dict, tokenizer: Any) -> None:
    preparation = manifest["preparation"]
    validate_safety_preparation(preparation)
    sources = preparation.get("sources")
    validate_safety_sources(sources)
    if (sources[SafetySourceName.TRAINING]["rows"] != preparation["training_rows"]
            or sum(sources[name]["rows"] for name in (SafetySourceName.EVALUATION_BEAVERTAILS,
                                                      SafetySourceName.EVALUATION_WILDGUARD)) != preparation["evaluation_rows"]
            or sources[SafetySourceName.BACKGROUND]["rows"] < preparation["background_count"] + len(preparation["background_excluded"])):
        raise ValueError("safety target source counts disagree with preparation audit")
    expected_keys = safety_evaluation_prompt_keys(preparation["evaluation_prompts"], tokenizer)
    if manifest["evaluation_prompt_keys"] != {SAFETY_TARGET_NAME: expected_keys}:
        raise ValueError("safety target evaluation token membership disagrees with raw evaluation prompts")
    if manifest["fit_contract"].get("decoder_tokenizer_checksum") != compute_tokenizer_checksum(tokenizer):
        raise ValueError("safety target decoder tokenizer disagrees with fit contract")


def _validate_fit_provenance(manifest: dict) -> None:
    provenance = manifest["fit_provenance"]
    fields = {"checkpoint_metadata_sha256", "extraction_tokenizer_checksum", "sae", "source_weight_revision",
              "extraction_batch_size", "source_window", "render_reduction", "corpus_sha256", "code",
              "backend", "device", "resolved_sae"}
    if not isinstance(provenance, dict) or set(provenance) != fields:
        raise ValueError("safety target fit provenance has missing or unknown fields")
    for field in ("checkpoint_metadata_sha256", "extraction_tokenizer_checksum", "corpus_sha256"):
        _digest(provenance[field], field)
    for field in ("extraction_batch_size", "source_window"):
        _positive_integer(provenance[field], field)
    if provenance["render_reduction"] != SAFETY_RENDER_REDUCTION:
        raise ValueError("safety target fit provenance uses an unsupported render/reduction")
    if (provenance["source_weight_revision"] != UNRECORDED_SOURCE_WEIGHT_REVISION
            and (not isinstance(provenance["source_weight_revision"], str)
                 or re.fullmatch(r"[0-9a-f]{40}", provenance["source_weight_revision"]) is None)):
        raise ValueError("safety target source weight revision must be pinned or explicitly unrecorded")
    if not isinstance(provenance["device"], str) or not provenance["device"].strip():
        raise ValueError("safety target fit device must be nonblank")
    if provenance["backend"] not in (ExtractionBackend.nnterp, ExtractionBackend.transformerlens):
        raise ValueError("safety target extraction backend must be explicit")
    code = provenance["code"]
    if (not isinstance(code, dict) or set(code) != {"runtime_version", "git_commit", "source_sha256"}
            or not isinstance(code["runtime_version"], str) or not code["runtime_version"].strip()):
        raise ValueError("safety target fit provenance requires source code identity")
    _digest(code["git_commit"], "source code git_commit", length=40)
    _digest(code["source_sha256"], "source code sha256")
    contract = manifest["fit_contract"]
    source = contract.get("source")
    if not isinstance(source, dict):
        raise ValueError("safety target fit contract requires source geometry")
    fields = ("model", "release", "activation_type", "layer", "width", "l0")
    mode = contract["segment_mode"]
    if mode == SegmentMode.turn_delimited_fixed_tokens:
        lineage = source.get("source_lineage")
        if not isinstance(lineage, dict) or any(field not in lineage for field in fields):
            raise ValueError("safety target turn fit provenance requires source lineage")
        expected_sae = {field: lineage[field] for field in fields}
        if (provenance["source_window"] != source.get("source_context_window")
                or provenance["extraction_tokenizer_checksum"] != source.get("extraction_tokenizer_checksum")):
            raise ValueError("safety target source window/tokenizer disagrees with turn geometry")
    elif mode == SegmentMode.fixed_tokens:
        expected_sae = source.get("sae")
    else:
        raise ValueError("safety target source segment mode is unsupported")
    config = provenance["sae"]
    if (not isinstance(config, dict) or any(field not in config for field in fields)
            or canonical_turn_payload(config) != canonical_turn_payload(expected_sae)):
        raise ValueError("safety target SAE extraction provenance disagrees with checkpoint source")
    release = SAERelease(config["release"])
    resolved = resolve_sae(model=config["model"], release=release, activation_type=ActivationType(config["activation_type"]),
        layer=config["layer"], width=SAEWidth(config["width"]), l0=coerce_config_l0(release, config["l0"]))
    if provenance["resolved_sae"] != {"release": resolved.release, "sae_id": resolved.sae_id, "hook_name": resolved.hook_name}:
        raise ValueError("safety target resolved SAE provenance disagrees with configured source")


@dataclass(frozen=True)
class SafetyTargets:
    bundle: BenchmarkTargets
    manifest: dict[str, Any]

    @property
    def sha256(self) -> str:
        return self.bundle.sha256

    def assert_matches_file(self, path: Path) -> None:
        if stream_hash_file_descriptor_bound(path.resolve(strict=True)).sha256 != self.sha256:
            raise ValueError("safety target bytes changed during evaluation")

    def policy(self, base: SelfCondConfig, strength: float) -> SelfCondConfig:
        return self.bundle.policy(base, SAFETY_TARGET_NAME, strength)

    def validate_prompts(self, prompts: Sequence[Sequence[int]]) -> None:
        allowed = set(self.bundle.evaluation_prompt_keys[SAFETY_TARGET_NAME])
        if not prompts or any(token_context_key(ids) not in allowed for ids in prompts):
            raise ValueError("safety steering evaluation prompt was not excluded from fitting")


def decode_safety_targets(payload: bytes, *, expected_contract: dict, expected_feature_ids: t.Tensor, tokenizer: Any) -> SafetyTargets:
    bundle = decode_benchmark_targets(payload, expected_contract=expected_contract,
        expected_feature_ids=expected_feature_ids, task_names=(SAFETY_TARGET_NAME,))
    metadata = read_safetensors_metadata_from_bytes(payload)
    assert metadata is not None  # Shared decoder authenticated metadata on these exact bytes.
    manifest = json.loads(metadata[TARGET_METADATA_KEY])
    _validate_population(manifest, tokenizer)
    _validate_fit_provenance(manifest)
    return SafetyTargets(bundle, manifest)


def load_safety_targets(path: Path, *, expected_contract: dict, expected_feature_ids: t.Tensor, tokenizer: Any) -> SafetyTargets:
    snapshot = stream_hash_file_descriptor_bound(path.resolve(strict=True), retain_payload=True)
    assert snapshot.payload is not None  # retain_payload=True is the immutable I/O contract.
    return decode_safety_targets(snapshot.payload, expected_contract=expected_contract, expected_feature_ids=expected_feature_ids, tokenizer=tokenizer)


def write_safety_targets(path: Path, *, target: t.Tensor, feature_ids: t.Tensor, manifest: dict[str, Any], tokenizer: Any) -> SafetyTargets:
    _validate_population(manifest, tokenizer)
    _validate_fit_provenance(manifest)
    bundle = write_benchmark_targets(
        path,
        targets={SAFETY_TARGET_NAME: target},
        feature_ids=feature_ids,
        manifest=manifest,
        task_names=(SAFETY_TARGET_NAME,),
    )
    return SafetyTargets(bundle, json.loads(canonical_turn_payload(manifest)))
