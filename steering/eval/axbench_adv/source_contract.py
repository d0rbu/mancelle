"""Disk-hash-fenced source validation for upstream AxBench c2 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
import torch as t
from beartype import beartype
from loguru import logger
from safetensors import SafetensorError

from nanochat.atomic_io import atomic_write_text
from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    validate_steering_frame,
)
from scripts.eval.axbench_upstream.run_baselines import (
    REQUIRED_PREPARED_PATHS,
    validate_prepared_dir,
)
from steering._common.strict_json import (
    canonical_json,
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.eval.axbench_adv.best_point import load_steering_entries
from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.eval.axbench_adv.method_grid import (
    ADV_FULL_FACTORS,
    ADV_INPUTS_PER_CONCEPT,
    ADV_METHOD_FACTORS,
    ALL_METHOD_FACTORS,
)
from steering.eval.axbench_adv.types import (
    ALL_METHODS,
    METHODS,
    NL_METHOD,
    ROOT_METHODS,
    SELF_CONDITIONING_METHODS,
)
from steering.eval.axbench_upstream.data import ConceptTrainingTexts, select_training_texts
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_FACTORS,
    SELF_CONDITIONING_MASKED_POLICY,
    SELF_CONDITIONING_POLICY,
    TEMPERATURE,
    TOP_K,
    MethodSetup,
    build_self_conditioning_masked_policy,
    load_method_setup,
    self_conditioning_masked_pool_limit,
)
from steering.eval.axbench_upstream.nl_target_scores import (
    NL_RETRIEVAL_ALGORITHM,
    NL_STEERING_POLICY,
    build_retrieval_target_scores,
    retrieval_feature_identity,
)
from steering.eval.axbench_upstream.run_methods import _tokenizer_artifact_identity
from steering.refined_types import (
    LowercaseSha256Like,
    NonBlankStrLike,
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    PositiveIntLike,
)
from steering.remote_training import RemoteTrainingError, parse_checkpoint_marker
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import resolve_checkpoint_sft_segment_mode
from steering.stats_io import content_sha256

SOURCE_HASH_KEYS: tuple[str, ...] = (
    "main_parquet",
    "aux_parquet",
    "evaluated_jsonl",
    "project_parquet",
    "project_manifest",
    "target_receipt",
)
_BASE_SOURCE_HASH_KEYS = SOURCE_HASH_KEYS[:3]
_PROJECT_SOURCE_HASH_KEYS = SOURCE_HASH_KEYS[3:]
_SHA256_HEX_LENGTH = 64
_RANK_ERA_REQUIRED = (
    "AxBench-Adv SelfConditioningMasked requires a rank-era schema-3 project "
    "artifact; threshold-era artifacts are unsupported"
)
_TARGET_META_RE = re.compile(r"^meta_(?P<step>[0-9]{6})\.json$")
_EVALUATION_TARGET_KIND = "axbench_adv_evaluation_target"
_PRODUCER_METHODS = (
    "DiffMean",
    "LinearProbe",
    "PromptSteering",
    "LoRA",
    *SELF_CONDITIONING_METHODS,
)
_SUMMARY_FIELDS = (
    "lm_judge_rating",
    "relevance_concept_ratings",
    "relevance_instruction_ratings",
    "fluency_ratings",
)
_RAW_RATING_FIELDS = (
    "raw_relevance_concept_ratings",
    "raw_relevance_instruction_ratings",
    "raw_fluency_ratings",
    "raw_aggregated_ratings",
)
_COMPLETION_FIELDS = (
    "relevance_concept_completions",
    "relevance_instruction_completions",
    "fluency_completions",
)
_RESULT_FIELDS = {
    "factor",
    *_SUMMARY_FIELDS,
    *_RAW_RATING_FIELDS,
    *_COMPLETION_FIELDS,
}
_PREPARED_MANIFEST_REQUIRED_KEYS = {
    "schema_version",
    "axbench_upstream_commit",
    "benchmark",
    "alpaca_eval",
    "seed_revisions",
    "seed_rows",
    "stanza",
    "artifacts",
}


@dataclass(frozen=True)
class SourcePaths:
    """The six hashed sources plus explicit runtime setup inputs."""

    prepared_dir: Path
    main_parquet: Path
    aux_parquet: Path
    evaluated_jsonl: Path
    project_parquet: Path
    target_meta: Path
    feature_statistics: Path
    target_receipt_path: Path | None = None

    @property
    def project_manifest(self) -> Path:
        return Path(self.project_parquet).parent.parent / "run_manifest.json"

    @property
    def target_step(self) -> int:
        match = _TARGET_META_RE.fullmatch(Path(self.target_meta).name)
        if match is None:
            raise AxBenchAdvError("target metadata filename must be exactly meta_<six digits>.json")
        return int(match.group("step"))

    @property
    def target_model(self) -> Path:
        return Path(self.target_meta).with_name(f"model_{self.target_step:06d}.pt")

    @property
    def target_receipt(self) -> Path:
        if self.target_receipt_path is not None:
            return Path(self.target_receipt_path)
        return Path(self.target_meta).with_name(f"remote_complete_{self.target_step:06d}.json")

    def hashed_files(self) -> dict[str, Path]:
        """Return the canonical source-hash key to path mapping."""

        return {
            "main_parquet": Path(self.main_parquet),
            "aux_parquet": Path(self.aux_parquet),
            "evaluated_jsonl": Path(self.evaluated_jsonl),
            "project_parquet": Path(self.project_parquet),
            "project_manifest": self.project_manifest,
            "target_receipt": self.target_receipt,
        }

    def runtime_files(self) -> dict[str, Path]:
        return {
            "target_meta": Path(self.target_meta),
            "target_model": self.target_model,
            "feature_statistics": Path(self.feature_statistics),
        }


@dataclass(frozen=True)
class NLAttachmentPaths:
    """Completed project generations, retrieval inputs and NL judgments."""

    project_parquet: Path
    project_manifest: Path
    evaluated_jsonl: Path

    def hashed_files(self) -> dict[str, Path]:
        return {
            "project_parquet": Path(self.project_parquet),
            "project_manifest": Path(self.project_manifest),
            "evaluated_jsonl": Path(self.evaluated_jsonl),
        }


@beartype(conf=AXBENCH_ADV_BEARTYPE)
@dataclass(frozen=True)
class NLAttachmentExpectations:
    """Size of the completed source population before campaign selection."""

    max_concepts: PositiveIntLike = 100


@dataclass(frozen=True)
class NLSourceAttachment:
    """Validated NL-only campaign view over a completed source population."""

    paths: NLAttachmentPaths
    expectations: NLAttachmentExpectations
    hashes: Mapping[str, str]
    runtime_hashes: Mapping[str, str]
    runtime_byte_lengths: Mapping[str, int]
    selected_frame: pd.DataFrame
    steering_entries: list[dict[str, Any]]
    target_scores: Mapping[int, t.Tensor]
    identity: Mapping[str, Any]


@beartype(conf=AXBENCH_ADV_BEARTYPE)
@dataclass(frozen=True)
class SourceExpectations:
    """Exact identities for a baseline target's main/aux sources plus the
    shared regenerated project source.

    ``config_name_prefix`` and ``layer`` are baseline-target-specific (e.g.
    ``qwen2_5_3b``/28 for Qwen2.5-3B-Instruct, a different prefix/layer for
    Pythia or SmolLM3) -- every other field was already a per-target
    expectation supplied via config; these two used to be hardcoded to the
    historical Qwen run instead of coming from here.

    ``require_clean_worktree`` defaults to ``false``: source manifests are
    NOT required to attest ``dirty: false`` unless a config opts into the
    stricter, audited-provenance behavior by setting it ``true``. The exact
    recorded commit and every content hash are always enforced unchanged
    regardless of this flag; it governs only the "no uncommitted local
    changes were present at generation time" attestation. Set ``true`` when
    a source's exact code provenance (not just its file identity) needs to
    be independently auditable.
    """

    max_concepts: PositiveIntLike
    repository_git_commit: str
    dataset_revision: str
    target_model: NonBlankStrLike
    seed: NonNegativeIntLike
    temperature: NonNegativeFiniteNumberLike
    output_length: PositiveIntLike
    package_lock_sha256: LowercaseSha256Like
    vendored_axbench_upstream_commit: str
    project_repository_git_commit: str
    project_package_lock_sha256: LowercaseSha256Like
    config_name_prefix: NonBlankStrLike
    layer: PositiveIntLike
    require_clean_worktree: bool = False
    project_target_step: NonNegativeIntLike = 54_600
    project_target_model_tag: NonBlankStrLike = "l22_main_h100_8rank_db16_v1"
    model_revision: NonBlankStrLike | None = None
    include_nl_steering: bool = False

    @property
    def methods(self) -> tuple:
        # Avoid beartype sampling this fixed tuple and consuming global RNG.
        return ALL_METHODS if self.include_nl_steering else METHODS

    @property
    def project_methods(self) -> tuple:
        return ROOT_METHODS if self.include_nl_steering else SELF_CONDITIONING_METHODS


@dataclass(frozen=True)
class ValidatedSources:
    """Disk-hash-fenced sources; callers must treat retained frames as read-only."""

    paths: SourcePaths
    expectations: SourceExpectations
    output_dir: Path
    hashes: Mapping[str, str]
    main_frame: pd.DataFrame
    aux_frame: pd.DataFrame
    project_frame: pd.DataFrame
    steering_entries: list[dict[str, Any]]
    project_manifest: Mapping[str, Any]
    method_setup: MethodSetup
    training_texts: Mapping[int, ConceptTrainingTexts]
    runtime_hashes: Mapping[str, str]
    runtime_byte_lengths: Mapping[str, int]
    extraction_batch_size: int
    nl_target_scores: Mapping[int, t.Tensor] | None = None
    nl_attachment: NLSourceAttachment | None = None
    target_meta: Mapping[str, Any] | None = None

    @property
    def methods(self) -> tuple[str, ...]:
        if self.expectations.include_nl_steering or self.nl_attachment is not None:
            return ALL_METHODS
        return METHODS

    @property
    def project_methods(self) -> tuple[str, ...]:
        if self.expectations.include_nl_steering or self.nl_attachment is not None:
            return ROOT_METHODS
        return SELF_CONDITIONING_METHODS


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _sha256_file_stable(path: Path) -> str:
    """Hash one source file in bounded-memory chunks."""

    path = Path(path)
    if not path.is_file():
        raise AxBenchAdvError(f"source file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_hashes(expected_hashes: Mapping[str, str]) -> None:
    if set(expected_hashes) != set(SOURCE_HASH_KEYS):
        raise AxBenchAdvError(f"source hashes must contain exactly {SOURCE_HASH_KEYS}")
    for key, value in expected_hashes.items():
        if not _is_sha256(value):
            raise AxBenchAdvError(f"expected source hash {key!r} is not lowercase SHA-256")


def _assert_hash_keys(
    paths: SourcePaths,
    expected_hashes: Mapping[str, str],
    keys: Sequence[str],
) -> dict[str, str]:
    files = paths.hashed_files()
    observed = {key: _sha256_file_stable(files[key]) for key in keys}
    mismatches = {
        key: {"expected": expected_hashes[key], "observed": value}
        for key, value in observed.items()
        if value != expected_hashes[key]
    }
    if mismatches:
        raise AxBenchAdvError(f"source file hash mismatch: {mismatches}")
    return observed


def _require_paths_disjoint(
    sources: Sequence[Path],
    reserved: Sequence[Path],
    *,
    error: str,
) -> None:
    resolved_sources = tuple(Path(path).resolve() for path in sources)
    for target in (Path(path).resolve() for path in reserved):
        for source in resolved_sources:
            if (
                source == target
                or source.is_relative_to(target)
                or target.is_relative_to(source)
                or (source.exists() and target.exists() and source.samefile(target))
            ):
                raise AxBenchAdvError(f"{error}: {target} vs {source}")


def assert_source_hashes(paths: SourcePaths, expected_hashes: Mapping[str, str]) -> dict[str, str]:
    """Require exactly the six ratified source hashes."""

    _validate_expected_hashes(expected_hashes)
    return _assert_hash_keys(paths, expected_hashes, SOURCE_HASH_KEYS)


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AxBenchAdvError(f"{label} not found: {path}")
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_json_constant,
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AxBenchAdvError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise AxBenchAdvError(f"{label} must be a JSON object: {path}")
    return dict(loaded)


def _require_clean_commit(
    manifest: Mapping[str, Any],
    expected_commit: str,
    expected_stages: set[str],
    *,
    # Intentionally the OPPOSITE of SourceExpectations.require_clean_worktree's
    # default (False): this internal helper's only current caller always
    # passes the value explicitly, so this default exists purely as a
    # fail-safe for any future caller that forgets to -- strict, not
    # permissive, by default at this layer.
    require_clean: bool = True,
) -> None:
    if manifest.get("repository_git_commit") != expected_commit:
        raise AxBenchAdvError("source manifest repository commit does not match the pin")
    worktree = manifest.get("repository_git_worktree")
    if not isinstance(worktree, Mapping):
        raise AxBenchAdvError("source manifest must attest a repository worktree state")
    if require_clean and worktree.get("dirty") is not False:
        raise AxBenchAdvError("source manifest must attest a clean repository worktree")
    provenance = manifest.get("stage_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != expected_stages:
        raise AxBenchAdvError(
            f"source manifest stage provenance must be exactly {sorted(expected_stages)}"
        )
    for stage, value in provenance.items():
        if not isinstance(value, Mapping):
            raise AxBenchAdvError(f"source stage provenance {stage!r} must be a mapping")
        if value.get("repository_git_commit") != expected_commit:
            raise AxBenchAdvError(f"source stage {stage!r} commit does not match the pin")
        stage_worktree = value.get("repository_git_worktree")
        if not isinstance(stage_worktree, Mapping):
            raise AxBenchAdvError(f"source stage {stage!r} must attest a repository worktree state")
        if require_clean and stage_worktree.get("dirty") is not False:
            raise AxBenchAdvError(f"source stage {stage!r} must attest a clean worktree")


def _matches_exact_json_scalar(observed: Any, expected: object) -> bool:
    """Compare JSON scalars without allowing ``bool`` to impersonate a number."""

    if type(expected) is int:
        return type(observed) is int and observed == expected
    if type(expected) is float:
        return (
            not isinstance(observed, bool)
            and isinstance(observed, (int, float))
            and math.isfinite(float(observed))
            and float(observed) == expected
        )
    return type(observed) is type(expected) and observed == expected


def _matches_exact_json_value(observed: Any, expected: object) -> bool:
    """Recursively compare a small JSON contract with strict scalar types."""

    if isinstance(expected, dict):
        return (
            type(observed) is dict
            and set(observed) == set(expected)
            and all(
                _matches_exact_json_value(observed[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            type(observed) is list
            and len(observed) == len(expected)
            and all(
                _matches_exact_json_value(observed_value, expected_value)
                for observed_value, expected_value in zip(observed, expected, strict=True)
            )
        )
    return _matches_exact_json_scalar(observed, expected)


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    source: str,
    prepared_dir: Path,
    prepared_manifest: Mapping[str, Any],
    expectations: SourceExpectations,
) -> None:
    expected_models = {
        "main": ["DiffMean", "LinearProbe"],
        "aux": ["PromptSteering", "LoRA"],
    }[source]
    expected_factors = ADV_FULL_FACTORS if source == "main" else (1.0,)
    expected_stages = {"train", "latent", "steering"} if source == "main" else {"train", "steering"}
    expected_config_name = f"{expectations.config_name_prefix}_{source}.yaml"
    scalar_expectations = {
        "schema_version": 1,
        "config_name": expected_config_name,
        "dataset_revision": expectations.dataset_revision,
        "max_concepts": expectations.max_concepts,
        "seed": expectations.seed,
        "target_model": expectations.target_model,
    }
    for field, expected in scalar_expectations.items():
        if not _matches_exact_json_scalar(manifest.get(field), expected):
            raise AxBenchAdvError(
                f"{source} manifest {field!r} must be {expected!r}, got {manifest.get(field)!r}"
            )
    _require_clean_commit(
        manifest,
        expectations.repository_git_commit,
        expected_stages,
        require_clean=expectations.require_clean_worktree,
    )
    if not _is_sha256(manifest.get("package_lock_sha256")):
        raise AxBenchAdvError(f"{source} manifest package lock is not SHA-256")
    if manifest.get("package_lock_sha256") != expectations.package_lock_sha256:
        raise AxBenchAdvError(f"{source} manifest package lock does not match the pin")
    if not _is_git_commit(manifest.get("vendored_axbench_upstream_commit")):
        raise AxBenchAdvError(f"{source} manifest vendored AxBench commit is invalid")
    if (
        manifest.get("vendored_axbench_upstream_commit")
        != expectations.vendored_axbench_upstream_commit
    ):
        raise AxBenchAdvError(f"{source} manifest vendored AxBench commit does not match the pin")
    recorded_prepared_dir = manifest.get("prepared_dir")
    if not isinstance(recorded_prepared_dir, str) or not recorded_prepared_dir.strip():
        raise AxBenchAdvError(f"{source} manifest prepared_dir must be a nonblank source path")
    if (
        Path(recorded_prepared_dir).resolve() != prepared_dir.resolve()
        and os.environ.get("AXBENCH_ADV_ALLOW_RELOCATED_PREPARED") != "1"
    ):
        raise AxBenchAdvError(f"{source} manifest prepared_dir differs from the source path")
    if manifest.get("prepared_manifest") != prepared_manifest:
        raise AxBenchAdvError(
            f"{source} manifest does not embed the current prepared manifest at {prepared_dir}"
        )

    config = manifest.get("config")
    train = config.get("train") if isinstance(config, Mapping) else None
    inference = config.get("inference") if isinstance(config, Mapping) else None
    if not isinstance(train, Mapping):
        raise AxBenchAdvError(f"{source} manifest has no config.train mapping")
    if not isinstance(inference, Mapping):
        raise AxBenchAdvError(f"{source} manifest has no config.inference mapping")
    exact_train = {
        "model_name": expectations.target_model,
        "model_revision": expectations.model_revision,
        "layer": expectations.layer,
        "use_bf16": True,
    }
    for field, expected in exact_train.items():
        observed = train.get(field)
        if not _matches_exact_json_scalar(observed, expected):
            raise AxBenchAdvError(
                f"{source} train {field!r} must be {expected!r}, got {observed!r}"
            )
    exact_inference = {
        "models": expected_models,
        "seed": expectations.seed,
        "temperature": expectations.temperature,
        "steering_num_of_examples": ADV_INPUTS_PER_CONCEPT,
        "steering_output_length": expectations.output_length,
        "model_name": expectations.target_model,
        "steering_model_name": expectations.target_model,
        "model_revision": expectations.model_revision,
        "steering_model_revision": expectations.model_revision,
        "steering_datasets": ["AlpacaEval"],
        "use_bf16": True,
        "steering_intervention_type": "addition",
        "steering_layer": expectations.layer,
        "steering_layers": [expectations.layer],
        "intervene_on_prompt": True,
    }
    for field, expected in exact_inference.items():
        observed = inference.get(field)
        matches = (
            _matches_exact_json_scalar(observed, expected)
            if isinstance(expected, (str, int, float))
            else type(observed) is type(expected) and observed == expected
        )
        if not matches:
            raise AxBenchAdvError(
                f"{source} inference {field!r} must be {expected!r}, got {observed!r}"
            )
    raw_factors = inference.get("steering_factors")
    if not isinstance(raw_factors, list):
        raise AxBenchAdvError(f"{source} inference steering_factors must be a list")
    observed_factors = tuple(
        _numeric_factor(value, field=f"{source}.steering_factors") for value in raw_factors
    )
    if observed_factors != tuple(float(v) for v in expected_factors):
        raise AxBenchAdvError(f"{source} inference steering_factors differ from the grid")


def _numeric_rating(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AxBenchAdvError(f"evaluation JSONL field {field} contains a nonnumeric rating")
    rating = float(value)
    if not math.isfinite(rating) or not 0.0 <= rating <= 2.0:
        raise AxBenchAdvError(f"evaluation JSONL field {field} contains a rating outside [0, 2]")
    return rating


def _numeric_factor(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AxBenchAdvError(f"{field} contains a nonnumeric factor")
    factor = float(value)
    if not math.isfinite(factor):
        raise AxBenchAdvError(f"{field} contains a nonfinite factor")
    return factor


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AxBenchAdvError(f"evaluation JSONL field {field} must be a list")
    return value


def _validate_evaluated_jsonl(
    entries: Sequence[Mapping[str, Any]],
    *,
    max_concepts: int,
    include_nl_steering: bool = False,
    expected_methods: Sequence[str] | None = None,
) -> None:
    if len(entries) != max_concepts:
        raise AxBenchAdvError(
            f"evaluation JSONL must contain {max_concepts} concepts and represent "
            f"each concept exactly once in order; got {len(entries)}"
        )
    expected_methods = tuple(expected_methods) if expected_methods is not None else (
        (*_PRODUCER_METHODS, NL_METHOD) if include_nl_steering else _PRODUCER_METHODS
    )
    for entry in entries:
        results = entry.get("results")
        if not isinstance(results, Mapping) or set(results) != {"LMJudgeEvaluator"}:
            raise AxBenchAdvError("evaluation JSONL must contain only LMJudgeEvaluator results")
        methods = results["LMJudgeEvaluator"]
        if not isinstance(methods, Mapping) or tuple(methods) != expected_methods:
            raise AxBenchAdvError("evaluation JSONL method coverage differs from upstream")
        for method in expected_methods:
            expected_factors = (
                ADV_FULL_FACTORS
                if method == "SelfConditioningPrompt"
                else ALL_METHOD_FACTORS[method]
            )
            block = methods[method]
            if not isinstance(block, Mapping) or set(block) != _RESULT_FIELDS:
                raise AxBenchAdvError(f"evaluation JSONL {method} result fields differ")
            factors = _list(block["factor"], field=f"{method}.factor")
            if tuple(
                _numeric_factor(value, field=f"{method}.factor") for value in factors
            ) != tuple(expected_factors):
                raise AxBenchAdvError(f"evaluation JSONL {method} factor grid differs")
            summaries: dict[str, list[float]] = {}
            for field in _SUMMARY_FIELDS:
                values = _list(block[field], field=f"{method}.{field}")
                if len(values) != len(expected_factors):
                    raise AxBenchAdvError(f"evaluation JSONL {method}.{field} has wrong length")
                summaries[field] = [
                    _numeric_rating(value, field=f"{method}.{field}") for value in values
                ]

            raw_length = ADV_INPUTS_PER_CONCEPT * len(ADV_FULL_FACTORS)
            declared = [
                factor in expected_factors
                for _input_id in range(ADV_INPUTS_PER_CONCEPT)
                for factor in ADV_FULL_FACTORS
            ]
            raw_ratings: dict[str, list[float | None]] = {}
            for field in _RAW_RATING_FIELDS:
                values = _list(block[field], field=f"{method}.{field}")
                if len(values) != raw_length:
                    raise AxBenchAdvError(f"evaluation JSONL {method}.{field} has wrong length")
                numeric_values: list[float | None] = []
                for value, is_declared in zip(values, declared, strict=True):
                    if is_declared:
                        numeric_values.append(
                            _numeric_rating(value, field=f"{method}.{field}")
                        )
                    elif value is not None:
                        raise AxBenchAdvError(
                            f"evaluation JSONL {method}.{field} is non-null off-grid"
                        )
                    else:
                        numeric_values.append(None)
                raw_ratings[field] = numeric_values
            summary_to_raw = {
                "lm_judge_rating": "raw_aggregated_ratings",
                "relevance_concept_ratings": "raw_relevance_concept_ratings",
                "relevance_instruction_ratings": "raw_relevance_instruction_ratings",
                "fluency_ratings": "raw_fluency_ratings",
            }
            for summary_field, raw_field in summary_to_raw.items():
                for factor_index, factor in enumerate(expected_factors):
                    present = [
                        value
                        for position, value in enumerate(raw_ratings[raw_field])
                        if ADV_FULL_FACTORS[position % len(ADV_FULL_FACTORS)] == factor
                    ]
                    if len(present) != ADV_INPUTS_PER_CONCEPT or any(
                        value is None for value in present
                    ):
                        raise AxBenchAdvError(
                            f"evaluation JSONL {method}.{raw_field} is incomplete"
                        )
                    present_values = [value for value in present if value is not None]
                    expected = sum(present_values) / len(present_values)
                    if not math.isclose(
                        summaries[summary_field][factor_index],
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise AxBenchAdvError(
                            f"evaluation JSONL {method}.{summary_field} summary is inconsistent"
                        )
            for field in _COMPLETION_FIELDS:
                values = _list(block[field], field=f"{method}.{field}")
                if len(values) != raw_length:
                    raise AxBenchAdvError(f"evaluation JSONL {method}.{field} has wrong length")
                for value, is_declared in zip(values, declared, strict=True):
                    if is_declared and not (isinstance(value, str) and bool(value.strip())):
                        raise AxBenchAdvError(
                            f"evaluation JSONL {method}.{field} has an invalid completion"
                        )
                    if not is_declared and value is not None:
                        raise AxBenchAdvError(
                            f"evaluation JSONL {method}.{field} is non-null off-grid"
                        )


def _validate_cross_frame_inputs(main: pd.DataFrame, aux: pd.DataFrame) -> None:
    main_factor_one = main.loc[main["factor"].eq(1.0)].sort_values(
        ["concept_id", "input_id"], kind="stable"
    )
    aux_sorted = aux.sort_values(["concept_id", "input_id"], kind="stable")
    stable_columns = [column for column in INPUT_COLUMNS if column != "steered_input"]
    if (
        not main_factor_one.loc[:, stable_columns]
        .reset_index(drop=True)
        .equals(aux_sorted.loc[:, stable_columns].reset_index(drop=True))
    ):
        raise AxBenchAdvError("aux source inputs do not match main factor-1 inputs")


def _validate_project_manifest(
    manifest: Mapping[str, Any],
    *,
    paths: SourcePaths,
    prepared_manifest: Mapping[str, Any],
    expectations: SourceExpectations,
    main_sha256: str,
) -> int:
    if not _matches_exact_json_scalar(manifest.get("schema_version"), 3):
        raise AxBenchAdvError(_RANK_ERA_REQUIRED)
    exact_scalars = {
        "repository_git_commit": expectations.project_repository_git_commit,
        "vendored_axbench_upstream_commit": expectations.vendored_axbench_upstream_commit,
        "dataset_revision": expectations.dataset_revision,
        "package_lock_sha256": expectations.project_package_lock_sha256,
    }
    for field, expected in exact_scalars.items():
        if not _matches_exact_json_scalar(manifest.get(field), expected):
            raise AxBenchAdvError(f"project manifest {field!r} differs from the pinned contract")
    worktree = manifest.get("repository_git_worktree")
    if not isinstance(worktree, Mapping):
        raise AxBenchAdvError("project manifest must attest a source worktree state")
    if expectations.require_clean_worktree and worktree.get("dirty") is not False:
        raise AxBenchAdvError("project manifest must attest a clean source worktree")
    if manifest.get("prepared_manifest") != prepared_manifest:
        raise AxBenchAdvError(
            "project manifest prepared identity differs from current prepared data"
        )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise AxBenchAdvError("project manifest has no inputs mapping")
    if inputs.get("main_steering_parquet_sha256") != main_sha256:
        raise AxBenchAdvError("project manifest main Parquet hash differs from the pinned source")
    statistics_sha256 = inputs.get("feature_statistics_sha256")
    if not _is_sha256(statistics_sha256):
        raise AxBenchAdvError("project manifest feature-statistics digest is invalid")

    methods = manifest.get("methods")
    expected_project_methods = expectations.project_methods
    if not isinstance(methods, Mapping) or set(methods) != set(expected_project_methods):
        raise AxBenchAdvError("project manifest method blocks differ from the selected project methods")
    ordinary = methods["SelfConditioning"]
    masked = methods["SelfConditioningMasked"]
    if not isinstance(ordinary, Mapping):
        raise AxBenchAdvError("project manifest SelfConditioning method block is invalid")
    if not _matches_exact_json_value(
        ordinary.get("factors"), list(ADV_METHOD_FACTORS["SelfConditioning"])
    ):
        raise AxBenchAdvError("project manifest SelfConditioning factor grid differs")
    if ordinary.get("policy") != SELF_CONDITIONING_POLICY:
        raise AxBenchAdvError("project manifest SelfConditioning policy differs")
    if not isinstance(masked, Mapping):
        raise AxBenchAdvError("project manifest SelfConditioningMasked method block is invalid")
    if not _matches_exact_json_value(
        masked.get("factors"), list(ADV_METHOD_FACTORS["SelfConditioningMasked"])
    ):
        raise AxBenchAdvError("project manifest SelfConditioningMasked factor grid differs")
    if masked.get("policy") != SELF_CONDITIONING_MASKED_POLICY:
        raise AxBenchAdvError(_RANK_ERA_REQUIRED)
    if "masked_threshold_scale" in masked:
        raise AxBenchAdvError(_RANK_ERA_REQUIRED)
    expected_rank_contract = {
        "strength_semantics": "linear_rank_pool_limit_from_N_to_2x_side_mean",
        "rounding": "nearest_ties_to_even",
        "clamp": {"minimum": 1, "maximum": "eligible_head_rows"},
        "candidate_eligibility": "post_feature_mask_sign_consistent_stable_diffmean_rank",
        "final_ranking": "last_prompt_segment_max_pooled_float32_model_logits",
        "exact_zero_diffmean_eligible": False,
    }
    for field, expected in expected_rank_contract.items():
        if not _matches_exact_json_value(masked.get(field), expected):
            raise AxBenchAdvError(
                f"project manifest SelfConditioningMasked rank contract field {field!r} differs"
            )
    producer_methods = (*_PRODUCER_METHODS, NL_METHOD) if expectations.include_nl_steering else _PRODUCER_METHODS
    if manifest.get("method_order") != list(producer_methods):
        raise AxBenchAdvError("project manifest method order differs")

    generation = manifest.get("generation")
    if not isinstance(generation, Mapping) or generation != {
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
    }:
        raise AxBenchAdvError("project manifest generation policy differs")
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise AxBenchAdvError("project manifest has no target mapping")
    if (
        target.get("step") != expectations.project_target_step
        or target.get("model_tag") != expectations.project_target_model_tag
    ):
        raise AxBenchAdvError("project manifest target step or model tag differs")
    if Path(str(target.get("checkpoint_meta", ""))).name != Path(paths.target_meta).name:
        raise AxBenchAdvError("project manifest target metadata filename differs")

    config = manifest.get("configuration")
    if not isinstance(config, Mapping):
        raise AxBenchAdvError("project manifest has no configuration mapping")
    exact_config = {
        "max_concepts": expectations.max_concepts,
        "seed": expectations.seed,
        "max_new_tokens": expectations.output_length,
        "self_conditioning_policy": SELF_CONDITIONING_POLICY,
        "self_conditioning_masked_policy": SELF_CONDITIONING_MASKED_POLICY,
        "force": False,
    }
    for field, expected in exact_config.items():
        if not _matches_exact_json_scalar(config.get(field), expected):
            raise AxBenchAdvError(f"project manifest configuration {field!r} differs")
    if Path(str(config.get("main_path", ""))).name != Path(paths.main_parquet).name:
        raise AxBenchAdvError("project manifest main Parquet filename differs")
    if Path(str(config.get("target_meta_path", ""))).name != Path(paths.target_meta).name:
        raise AxBenchAdvError("project manifest target metadata filename differs")
    if (
        Path(str(config.get("feature_statistics_path", ""))).name
        != Path(paths.feature_statistics).name
    ):
        raise AxBenchAdvError("project manifest feature-statistics filename differs")
    if "masked_threshold_scale" in config:
        raise AxBenchAdvError(_RANK_ERA_REQUIRED)
    extraction_batch_size = config.get("extraction_batch_size")
    if type(extraction_batch_size) is not int or extraction_batch_size <= 0:
        raise AxBenchAdvError("project manifest extraction_batch_size must be positive")
    return extraction_batch_size


def _nl_source_records(
    manifest: Mapping[str, Any],
    project_manifest_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Path], dict[str, str]]:
    """Read the optional producer-attested retrieval inputs, without inference."""
    methods = manifest["methods"]
    block = methods[NL_METHOD]
    if not isinstance(block, Mapping) or set(block) != {"factors", "policy", "target_source"}:
        raise AxBenchAdvError("NLSteering method block fields differ")
    if block["policy"] != NL_STEERING_POLICY or not _matches_exact_json_value(
        block["factors"], list(ADV_FULL_FACTORS),
    ):
        raise AxBenchAdvError("NLSteering policy or factor grid differs")
    source = block["target_source"]
    if not isinstance(source, Mapping) or set(source) != {
        "algorithm", "feature_contract", "concepts", "description_embeddings", "concept_embeddings",
    }:
        raise AxBenchAdvError("NLSteering target_source fields differ")
    if source["algorithm"] != NL_RETRIEVAL_ALGORITHM:
        raise AxBenchAdvError("NLSteering retrieval algorithm differs")
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    spaces = []
    for name in ("description_embeddings", "concept_embeddings"):
        record = source[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "embedding_space", "source_texts_sha256"}:
            raise AxBenchAdvError(f"NLSteering {name} record fields differ")
        if not _is_sha256(record["source_texts_sha256"]):
            raise AxBenchAdvError(f"NLSteering {name} source-text identity is invalid")
        path = record["path"]
        if not isinstance(path, str) or not path.strip() or not _is_sha256(record["sha256"]):
            raise AxBenchAdvError(f"NLSteering {name} path or SHA-256 is invalid")
        files[name] = (project_manifest_path.parent / path).resolve()
        hashes[name] = record["sha256"]
        space = record["embedding_space"]
        if not isinstance(space, Mapping) or set(space) != {"model", "revision", "encoding", "library_versions"}:
            raise AxBenchAdvError("NLSteering embedding space must declare model, revision, encoding")
        if not _is_git_commit(space["revision"]) or any(
            not isinstance(space[key], str) or not space[key].strip() for key in ("model", "encoding")
        ):
            raise AxBenchAdvError("NLSteering embedding space provenance is invalid")
        versions = space["library_versions"]
        if not isinstance(versions, Mapping) or not versions or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in versions.items()
        ):
            raise AxBenchAdvError("NLSteering embedding library versions are invalid")
        spaces.append(space)
    if spaces[0] != spaces[1]:
        raise AxBenchAdvError("NLSteering description and concept embedding spaces differ")
    if len(set(files.values())) != 2:
        raise AxBenchAdvError("NLSteering embedding files must be distinct")
    return source, files, hashes


def _nl_runtime_hashes(
    manifest: Mapping[str, Any],
    project_manifest_path: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    _source, files, expected = _nl_source_records(manifest, project_manifest_path)
    hashes: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for name, path in files.items():
        digest = _sha256_file_stable(path)
        if digest != expected[name]:
            raise AxBenchAdvError(f"NLSteering {name} source hash mismatch")
        hashes[f"nl_{name}"] = digest
        lengths[f"nl_{name}"] = path.stat().st_size
    return hashes, lengths


def _validate_nl_targets(
    *, manifest: Mapping[str, Any], project_manifest_path: Path, setup: MethodSetup,
    training: Mapping[int, ConceptTrainingTexts], output_dir: Path,
    reserved_paths: Sequence[Path],
    surface: str = "NLSteering",
) -> dict[int, t.Tensor]:
    source, files, hashes = _nl_source_records(manifest, project_manifest_path)
    _require_paths_disjoint(
        tuple(files.values()),
        (output_dir, *reserved_paths),
        error=f"{surface} inputs, output, and provider caches must be path-disjoint",
    )
    concepts = {concept_id: text.concept for concept_id, text in training.items()}
    if not _matches_exact_json_value(source["concepts"], [
        {"concept_id": concept_id, "text": concepts[concept_id]} for concept_id in sorted(concepts)
    ]):
        raise AxBenchAdvError(f"{surface} concept identities differ from the source")
    if not _matches_exact_json_value(source["feature_contract"], retrieval_feature_identity(setup)):
        raise AxBenchAdvError(f"{surface} feature contract differs from the checkpoint")
    retrieval_surface = surface if surface.endswith("retrieval") else f"{surface} retrieval"
    try:
        return build_retrieval_target_scores(
            setup=setup,
            description_embeddings_path=files["description_embeddings"],
            concept_embeddings_path=files["concept_embeddings"],
            concepts=concepts,
            expected_hashes=hashes,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise AxBenchAdvError(f"{retrieval_surface} inputs failed validation") from exc


def _validate_nl_attachment(
    *,
    paths: NLAttachmentPaths,
    expectations: NLAttachmentExpectations,
    base_paths: SourcePaths,
    base_expectations: SourceExpectations,
    base_main_frame: pd.DataFrame,
    setup: MethodSetup,
    output_dir: Path,
    reserved_paths: Sequence[Path],
) -> NLSourceAttachment:
    """Load an NL-only view from trusted, completed local research outputs.

    Validate the data consumed by the experiment, not the producer's code,
    resume store, unused methods or historical metadata-repair procedure.
    Input files must remain unchanged for the duration of the run.
    """
    if base_expectations.include_nl_steering:
        raise AxBenchAdvError("NL attachment cannot be combined with single-source NL")
    if expectations.max_concepts < base_expectations.max_concepts:
        raise AxBenchAdvError("NL attachment has fewer concepts than the campaign")
    files = paths.hashed_files()
    _require_paths_disjoint(
        tuple(files.values()),
        (output_dir, *reserved_paths),
        error="NL attachment sources, output, and caches must be path-disjoint",
    )
    columns = [*INPUT_COLUMNS, f"{NL_METHOD}_steered_generation", f"{NL_METHOD}_strength"]
    try:
        full_frame = pd.read_parquet(paths.project_parquet).loc[:, columns]
        validate_steering_frame(
            full_frame,
            source="project",
            max_concepts=expectations.max_concepts,
            methods=(NL_METHOD,),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AxBenchAdvError("NL attachment project frame failed validation") from exc
    strength = full_frame[f"{NL_METHOD}_strength"]
    if not strength.astype(float).equals(full_frame["factor"].astype(float)):
        raise AxBenchAdvError("NL attachment strength values differ from factors")
    selected = full_frame.loc[
        full_frame["concept_id"].lt(base_expectations.max_concepts)
    ].reset_index(drop=True)
    if not selected.loc[:, list(INPUT_COLUMNS)].equals(
        base_main_frame.loc[:, list(INPUT_COLUMNS)].reset_index(drop=True)
    ):
        raise AxBenchAdvError("NL attachment selected inputs differ from the main source")

    manifest = _load_json_mapping(paths.project_manifest, label="NL attachment project manifest")
    if not isinstance(manifest.get("methods"), Mapping) or NL_METHOD not in manifest["methods"]:
        raise AxBenchAdvError("NL attachment manifest is missing NLSteering retrieval inputs")
    entries = load_steering_entries(
        paths.evaluated_jsonl,
        expected_concept_ids=range(expectations.max_concepts),
    )
    try:
        nl_entries = [
            {
                "concept_id": entry["concept_id"],
                "results": {"LMJudgeEvaluator": {
                    NL_METHOD: entry["results"]["LMJudgeEvaluator"][NL_METHOD],
                }},
            }
            for entry in entries
        ]
    except (KeyError, TypeError) as exc:
        raise AxBenchAdvError("NL attachment judged results are missing NLSteering") from exc
    _validate_evaluated_jsonl(
        nl_entries,
        max_concepts=expectations.max_concepts,
        expected_methods=(NL_METHOD,),
    )
    selected_entries = nl_entries[:int(base_expectations.max_concepts)]
    training = _validate_training_texts(
        paths=base_paths,
        main_frame=full_frame,
        max_concepts=expectations.max_concepts,
    )
    full_targets = _validate_nl_targets(
        manifest=manifest,
        project_manifest_path=paths.project_manifest,
        setup=setup,
        training=training,
        output_dir=output_dir,
        reserved_paths=reserved_paths,
        surface="NL attachment retrieval",
    )
    target_scores = {
        concept_id: full_targets[concept_id]
        for concept_id in range(int(base_expectations.max_concepts))
    }
    # Record actual inputs once for reproducibility, without demanding another
    # receipt or checking an unused checkpoint-local tokenizer copy.
    hashes = {name: _sha256_file_stable(path) for name, path in files.items()}
    runtime_hashes, runtime_lengths = _nl_runtime_hashes(manifest, paths.project_manifest)
    identity = {
        "schema_version": 1,
        "method": NL_METHOD,
        "hashes": hashes,
        "selection": {"concept_ids": list(range(int(base_expectations.max_concepts)))},
        "runtime_hashes": runtime_hashes,
        "tokenizer": _tokenizer_artifact_identity(),
    }
    return NLSourceAttachment(
        paths=paths,
        expectations=expectations,
        hashes=MappingProxyType(hashes),
        runtime_hashes=MappingProxyType(runtime_hashes),
        runtime_byte_lengths=MappingProxyType(runtime_lengths),
        selected_frame=selected,
        steering_entries=selected_entries,
        target_scores=MappingProxyType(target_scores),
        identity=MappingProxyType(identity),
    )


def _validate_project_frame(frame: pd.DataFrame, *, expectations: SourceExpectations) -> None:
    """Apply the existing full-grid interpolation-column contract to the NL arm."""
    nl_columns = [f"{NL_METHOD}_steered_generation", f"{NL_METHOD}_strength"]
    if expectations.include_nl_steering and not set(nl_columns).issubset(frame.columns):
        raise AxBenchAdvError("project frame is missing NLSteering columns")
    base = frame.drop(columns=nl_columns) if expectations.include_nl_steering else frame
    validate_steering_frame(
        base, source="project", max_concepts=expectations.max_concepts,
        methods=SELF_CONDITIONING_METHODS,
    )
    if expectations.include_nl_steering:
        # NL has exactly the same column and generation-health contract as ordinary
        # interpolation. Validate a view; the source frame is never rewritten.
        nl = frame.loc[:, [*INPUT_COLUMNS, *nl_columns]].rename(columns={
            column: column.replace(NL_METHOD, "SelfConditioning", 1) for column in nl_columns
        })
        validate_steering_frame(
            nl, source="project", max_concepts=expectations.max_concepts,
            methods=("SelfConditioning",),
        )


def _validate_rank_contract_against_setup(
    manifest: Mapping[str, Any],
    setup: MethodSetup,
) -> None:
    """Recompute the schema-3 rank table from loaded checkpoint policies."""

    # _validate_project_manifest has already established this exact structure.
    methods = cast(Mapping[str, Any], manifest["methods"])
    masked = cast(Mapping[str, Any], methods["SelfConditioningMasked"])
    head_width = (
        int(setup.n_sae_features)
        if setup.allowed_ids is None
        else int(setup.allowed_ids.numel())
    )
    dummy_target = t.zeros(head_width, dtype=t.float32)
    try:
        factor_zero_policy = build_self_conditioning_masked_policy(
            setup,
            dummy_target,
            0.0,
        )
    except (TypeError, ValueError) as exc:
        raise AxBenchAdvError(
            "checkpoint count policies cannot construct the rank-era SelfConditioningMasked contract"
        ) from exc
    eligible_head_rows = int(factor_zero_policy.positive_pool_limit)
    expected_limits = [
        {
            "factor": factor,
            "positive": self_conditioning_masked_pool_limit(
                factor,
                n_candidates=eligible_head_rows,
                mean=float(setup.output_selection.mean),
            ),
            "negative": self_conditioning_masked_pool_limit(
                factor,
                n_candidates=eligible_head_rows,
                mean=float(setup.output_negative_selection.mean),
            ),
        }
        for factor in SELF_CONDITIONING_FACTORS
    ]
    expected_scalars = {
        "eligible_head_rows": eligible_head_rows,
        "positive_mean": float(setup.output_selection.mean),
        "negative_mean": float(setup.output_negative_selection.mean),
    }
    for field, expected in expected_scalars.items():
        if not _matches_exact_json_scalar(masked.get(field), expected):
            raise AxBenchAdvError(
                f"project manifest SelfConditioningMasked {field} differs from checkpoint setup"
            )
    if not _matches_exact_json_value(masked.get("pool_limits"), expected_limits):
        raise AxBenchAdvError(
            "project manifest SelfConditioningMasked pool-limit table differs from checkpoint setup"
        )


def _validate_evaluation_target_receipt(
    raw: Mapping[str, Any], *, paths: SourcePaths
) -> None:
    if set(raw) != {
        "schema_version", "kind", "step", "remote_run_fingerprint", "files", "aggregate_sha256"
    }:
        raise ValueError("evaluation-target receipt keys changed")
    if (
        not _matches_exact_json_scalar(raw.get("schema_version"), 1)
        or raw.get("kind") != _EVALUATION_TARGET_KIND
    ):
        raise ValueError("evaluation-target receipt schema or kind changed")
    if not _matches_exact_json_scalar(raw.get("step"), paths.target_step):
        raise ValueError("evaluation-target receipt step changed")
    if not _is_sha256(raw["remote_run_fingerprint"]):
        raise ValueError("evaluation-target receipt fingerprint is invalid")
    files = raw.get("files")
    expected_paths = (Path(paths.target_meta), paths.target_model)
    if not isinstance(files, list) or len(files) != len(expected_paths):
        raise ValueError("evaluation-target receipt must contain exactly metadata and model")
    for index, (entry, expected_path) in enumerate(zip(files, expected_paths, strict=True)):
        if not isinstance(entry, Mapping) or set(entry) != {
            "name", "byte_length", "content_sha256", "xet_hash"
        }:
            raise ValueError(f"evaluation-target receipt file entry {index} has invalid keys")
        if entry.get("name") != expected_path.name:
            raise ValueError(f"evaluation-target receipt file entry {index} has the wrong name")
        byte_length = entry.get("byte_length")
        if type(byte_length) is not int or byte_length < 0:
            raise ValueError("evaluation-target receipt byte length is invalid")
        if not all(_is_sha256(entry.get(key)) for key in ("content_sha256", "xet_hash")):
            raise ValueError("evaluation-target receipt file digest is invalid")
    signed = {key: value for key, value in raw.items() if key != "aggregate_sha256"}
    aggregate = hashlib.sha256(canonical_json(signed).encode("utf-8")).hexdigest()
    if raw.get("aggregate_sha256") != aggregate:
        raise ValueError("evaluation-target receipt aggregate mismatch")


def _validate_checkpoint_sources(
    *,
    paths: SourcePaths,
    project_manifest: Mapping[str, Any],
    expectations: SourceExpectations,
) -> tuple[dict[str, str], dict[str, int], dict[str, Any]]:
    if paths.target_step != expectations.project_target_step:
        raise AxBenchAdvError(
            f"target metadata step must be exactly {expectations.project_target_step}"
        )
    receipt = _load_json_mapping(paths.target_receipt, label="target checkpoint receipt")
    try:
        if receipt.get("kind") == _EVALUATION_TARGET_KIND:
            _validate_evaluation_target_receipt(receipt, paths=paths)
        else:
            receipt = parse_checkpoint_marker(receipt, surface="AxBench-Adv target")
    except (RemoteTrainingError, TypeError, ValueError) as exc:
        raise AxBenchAdvError("target checkpoint receipt failed validation") from exc
    if receipt["step"] != paths.target_step:
        raise AxBenchAdvError("target checkpoint receipt step differs from metadata filename")
    receipt_entries = {entry["name"]: entry for entry in receipt["files"]}
    required = {
        paths.target_meta.name: paths.target_meta,
        paths.target_model.name: paths.target_model,
    }
    runtime_hashes: dict[str, str] = {}
    runtime_lengths: dict[str, int] = {}
    for key, path in paths.runtime_files().items():
        # feature_statistics is a safetensors file: atomic_safetensors_save
        # serializes its metadata HashMap in a randomized key order on every
        # save, so a plain file-byte hash is not reproducible across
        # independent saves of identical decoded content -- hash the
        # decoded content for the project manifest and runtime identity.
        # Turn checkpoint lineage separately pins the raw file bytes.
        if key == "feature_statistics":
            try:
                runtime_hashes[key] = content_sha256(path)
            except (OSError, SafetensorError) as exc:
                raise AxBenchAdvError(
                    f"feature statistics file failed to load: {path}: {exc}"
                ) from exc
        else:
            runtime_hashes[key] = _sha256_file_stable(path)
        runtime_lengths[key] = path.stat().st_size
    for name, path in required.items():
        entry = receipt_entries.get(name)
        if not isinstance(entry, Mapping):
            raise AxBenchAdvError(f"target receipt does not contain {name}")
        runtime_key = "target_meta" if path == paths.target_meta else "target_model"
        if (
            entry.get("byte_length") != runtime_lengths[runtime_key]
            or entry.get("content_sha256") != runtime_hashes[runtime_key]
        ):
            raise AxBenchAdvError(f"local target file identity differs from receipt: {name}")

    meta = _load_json_mapping(paths.target_meta, label="target metadata")
    if meta.get("step") != paths.target_step:
        raise AxBenchAdvError("target metadata JSON step differs from its filename")
    if meta.get("remote_run_fingerprint") != receipt["remote_run_fingerprint"]:
        raise AxBenchAdvError("target metadata remote fingerprint differs from receipt")
    user_config = meta.get("user_config")
    if (
        not isinstance(user_config, Mapping)
        or user_config.get("model_tag") != expectations.project_target_model_tag
    ):
        raise AxBenchAdvError("target metadata model tag differs from the pinned contract")
    conditioning = user_config.get("conditioning")
    metadata_statistics = (
        conditioning.get("stats_source_sha256") if isinstance(conditioning, Mapping) else None
    )
    manifest_inputs = project_manifest.get("inputs")
    manifest_statistics = (
        manifest_inputs.get("feature_statistics_sha256")
        if isinstance(manifest_inputs, Mapping)
        else None
    )
    if runtime_hashes["feature_statistics"] != manifest_statistics:
        raise AxBenchAdvError("feature statistics differ from the project manifest")
    try:
        segment_mode = resolve_checkpoint_sft_segment_mode(
            meta, checkpoint_dir=str(paths.target_meta.parent)
        )
    except ValueError as exc:
        raise AxBenchAdvError("target checkpoint segmentation failed validation") from exc
    metadata_actual = (
        _sha256_file_stable(paths.feature_statistics)
        if segment_mode is SegmentMode.turn_delimited_fixed_tokens
        else runtime_hashes["feature_statistics"]
    )
    if metadata_actual != metadata_statistics:
        if not _is_sha256(metadata_statistics):
            raise AxBenchAdvError("feature statistics differ from target metadata")
        # Preserve the companion metadata warning for legacy hashes. The
        # shared loader additionally enforces the authoritative raw lineage
        # digest for turn checkpoints, regardless of this companion field.
        logger.warning(
            "feature statistics hash differs from the target checkpoint's recorded "
            f"conditioning.stats_source_sha256 (expected={metadata_statistics}, actual="
            f"{metadata_actual}) -- project manifest content hash matches; "
            "checkpoint lineage identity, when present, is enforced by method setup"
        )
    return runtime_hashes, runtime_lengths, meta


def _validate_training_texts(
    *,
    paths: SourcePaths,
    main_frame: pd.DataFrame,
    max_concepts: int,
) -> dict[int, ConceptTrainingTexts]:
    try:
        training = select_training_texts(paths.prepared_dir, max_concepts=max_concepts)
    except (OSError, TypeError, ValueError) as exc:
        raise AxBenchAdvError("SelfConditioning training text selection failed") from exc
    source_concepts = (
        main_frame.loc[main_frame["factor"].eq(1.0)]
        .groupby("concept_id", sort=False)["input_concept"]
        .first()
        .to_dict()
    )
    for concept_id, examples in training.items():
        if source_concepts.get(concept_id) != examples.concept:
            raise AxBenchAdvError("prepared concept text differs from validated main source")
    return training


def _require_output_disjoint(paths: SourcePaths, output_dir: Path) -> None:
    sources = [
        Path(paths.prepared_dir),
        *(
            Path(paths.prepared_dir) / relative_path
            for relative_path in REQUIRED_PREPARED_PATHS
        ),
        *paths.hashed_files().values(),
        *paths.runtime_files().values(),
    ]
    _require_paths_disjoint(
        sources,
        (output_dir,),
        error="output path must be disjoint from every source path",
    )


def validate_source_contract(
    *,
    paths: SourcePaths,
    expectations: SourceExpectations,
    expected_hashes: Mapping[str, str],
    output_dir: Path,
    reserved_paths: Sequence[Path] = (),
    nl_attachment_paths: NLAttachmentPaths | None = None,
    nl_attachment_expectations: NLAttachmentExpectations | None = None,
) -> ValidatedSources:
    """Validate prepared data, manifests, frames, JSONL, hashes, and disjointness."""

    if not _is_git_commit(expectations.repository_git_commit):
        raise AxBenchAdvError("repository_git_commit must be a lowercase 40-character hex commit")
    if not _is_git_commit(expectations.dataset_revision):
        raise AxBenchAdvError("dataset_revision must be a lowercase 40-character hex commit")
    if not _is_git_commit(expectations.vendored_axbench_upstream_commit):
        raise AxBenchAdvError(
            "vendored_axbench_upstream_commit must be a lowercase 40-character hex commit"
        )
    if not _is_git_commit(expectations.project_repository_git_commit):
        raise AxBenchAdvError(
            "project_repository_git_commit must be a lowercase 40-character hex commit"
        )
    _require_output_disjoint(paths, output_dir)
    try:
        prepared_manifest = validate_prepared_dir(Path(paths.prepared_dir))
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise AxBenchAdvError("prepared AxBench directory failed validation") from exc
    missing_prepared_keys = _PREPARED_MANIFEST_REQUIRED_KEYS - set(prepared_manifest)
    if missing_prepared_keys:
        raise AxBenchAdvError(
            f"prepared AxBench manifest is missing required keys: {sorted(missing_prepared_keys)}"
        )
    _validate_expected_hashes(expected_hashes)
    initial_hashes = _assert_hash_keys(paths, expected_hashes, _BASE_SOURCE_HASH_KEYS)
    try:
        main_frame = pd.read_parquet(paths.main_parquet)
        aux_frame = pd.read_parquet(paths.aux_parquet)
        validate_steering_frame(main_frame, source="main", max_concepts=expectations.max_concepts)
        validate_steering_frame(aux_frame, source="aux", max_concepts=expectations.max_concepts)
    except (OSError, ValueError, TypeError) as exc:
        raise AxBenchAdvError("source steering frame failed validation") from exc
    _validate_cross_frame_inputs(main_frame, aux_frame)

    main_manifest = _load_json_mapping(
        Path(paths.main_parquet).parent.parent / "run_manifest.json",
        label="main run manifest",
    )
    aux_manifest = _load_json_mapping(
        Path(paths.aux_parquet).parent.parent / "run_manifest.json",
        label="aux run manifest",
    )
    _validate_run_manifest(
        main_manifest,
        source="main",
        prepared_dir=Path(paths.prepared_dir),
        prepared_manifest=prepared_manifest,
        expectations=expectations,
    )
    _validate_run_manifest(
        aux_manifest,
        source="aux",
        prepared_dir=Path(paths.prepared_dir),
        prepared_manifest=prepared_manifest,
        expectations=expectations,
    )
    steering_entries = load_steering_entries(Path(paths.evaluated_jsonl))
    _validate_evaluated_jsonl(
        steering_entries, max_concepts=expectations.max_concepts,
        include_nl_steering=expectations.include_nl_steering,
    )

    initial_hashes.update(_assert_hash_keys(paths, expected_hashes, _PROJECT_SOURCE_HASH_KEYS))
    try:
        project_frame = pd.read_parquet(paths.project_parquet)
        _validate_project_frame(project_frame, expectations=expectations)
    except (OSError, ValueError, TypeError) as exc:
        raise AxBenchAdvError("project frame failed validation") from exc
    for method in expectations.project_methods:
        strength = project_frame[f"{method}_strength"]
        if not strength.astype(float).equals(project_frame["factor"].astype(float)):
            raise AxBenchAdvError(f"project {method} strength values differ from factors")
    if not project_frame.loc[:, list(INPUT_COLUMNS)].equals(
        main_frame.loc[:, list(INPUT_COLUMNS)]
    ):
        raise AxBenchAdvError("project SelfConditioning inputs do not exactly equal the main frame")
    project_manifest = _load_json_mapping(paths.project_manifest, label="project run manifest")
    extraction_batch_size = _validate_project_manifest(
        project_manifest,
        paths=paths,
        prepared_manifest=prepared_manifest,
        expectations=expectations,
        main_sha256=initial_hashes["main_parquet"],
    )
    runtime_hashes, runtime_lengths, target_meta = _validate_checkpoint_sources(
        paths=paths,
        project_manifest=project_manifest,
        expectations=expectations,
    )
    try:
        # Turn lineage pins raw statistics bytes. Legacy pretrain checkpoints
        # retain the content-hash override fenced by the project manifest.
        segment_mode = resolve_checkpoint_sft_segment_mode(
            target_meta, checkpoint_dir=str(paths.target_meta.parent)
        )
        method_setup = load_method_setup(
            paths.target_meta,
            paths.feature_statistics,
            require_exact_statistics_identity=(
                segment_mode is SegmentMode.turn_delimited_fixed_tokens
            ),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise AxBenchAdvError("SelfConditioning method setup failed validation") from exc
    _validate_rank_contract_against_setup(project_manifest, method_setup)
    training_texts = _validate_training_texts(
        paths=paths,
        main_frame=main_frame,
        max_concepts=expectations.max_concepts,
    )
    nl_target_scores = None
    if expectations.include_nl_steering:
        nl_target_scores = _validate_nl_targets(
            manifest=project_manifest,
            project_manifest_path=paths.project_manifest,
            setup=method_setup,
            training=training_texts, output_dir=output_dir, reserved_paths=reserved_paths,
        )
        nl_hashes, nl_lengths = _nl_runtime_hashes(project_manifest, paths.project_manifest)
        runtime_hashes.update(nl_hashes)
        runtime_lengths.update(nl_lengths)
    final_hashes = assert_source_hashes(paths, expected_hashes)
    if final_hashes != initial_hashes:
        raise AxBenchAdvError("source identities changed during validation")
    final_runtime_hashes, final_runtime_lengths, _target_meta = _validate_checkpoint_sources(
        paths=paths,
        project_manifest=project_manifest,
        expectations=expectations,
    )
    if expectations.include_nl_steering:
        nl_hashes, nl_lengths = _nl_runtime_hashes(project_manifest, paths.project_manifest)
        final_runtime_hashes.update(nl_hashes)
        final_runtime_lengths.update(nl_lengths)
    if final_runtime_hashes != runtime_hashes or final_runtime_lengths != runtime_lengths:
        raise AxBenchAdvError("runtime source identities changed during validation")
    if (nl_attachment_paths is None) != (nl_attachment_expectations is None):
        raise AxBenchAdvError("NL attachment paths and expectations are both required")
    attachment = None
    if (
        nl_attachment_paths is not None
        and nl_attachment_expectations is not None
    ):
        attachment = _validate_nl_attachment(
            paths=nl_attachment_paths,
            expectations=nl_attachment_expectations,
            base_paths=paths,
            base_expectations=expectations,
            base_main_frame=main_frame,
            setup=method_setup,
            output_dir=output_dir,
            reserved_paths=reserved_paths,
        )
        nl_target_scores = attachment.target_scores
    return ValidatedSources(
        paths=paths,
        expectations=expectations,
        output_dir=Path(output_dir).resolve(),
        hashes=MappingProxyType(final_hashes),
        main_frame=main_frame,
        aux_frame=aux_frame,
        project_frame=project_frame,
        steering_entries=steering_entries,
        project_manifest=MappingProxyType(project_manifest),
        method_setup=method_setup,
        training_texts=MappingProxyType(training_texts),
        runtime_hashes=MappingProxyType(runtime_hashes),
        runtime_byte_lengths=MappingProxyType(runtime_lengths),
        extraction_batch_size=extraction_batch_size,
        nl_target_scores=None if nl_target_scores is None else MappingProxyType(nl_target_scores),
        nl_attachment=attachment,
        target_meta=MappingProxyType(target_meta),
    )


def revalidate_source_hashes(validated: ValidatedSources) -> None:
    """Final publication fence for pinned and transitively pinned runtime bytes."""

    assert_source_hashes(validated.paths, validated.hashes)
    runtime_hashes, runtime_lengths, _target_meta = _validate_checkpoint_sources(
        paths=validated.paths,
        project_manifest=validated.project_manifest,
        expectations=validated.expectations,
    )
    if validated.expectations.include_nl_steering:
        nl_hashes, nl_lengths = _nl_runtime_hashes(
            validated.project_manifest, validated.paths.project_manifest,
        )
        runtime_hashes.update(nl_hashes)
        runtime_lengths.update(nl_lengths)
    if runtime_hashes != dict(validated.runtime_hashes):
        raise AxBenchAdvError("runtime source hash changed after validation")
    if runtime_lengths != dict(validated.runtime_byte_lengths):
        raise AxBenchAdvError("runtime source byte length changed after validation")


def write_source_contract(validated: ValidatedSources, output_path: Path) -> Path:
    """Persist the resolved, serializable source identity evidence."""

    output_path = Path(output_path).resolve()
    if output_path == validated.output_dir or not output_path.is_relative_to(validated.output_dir):
        raise AxBenchAdvError(
            f"publication path must be inside validated output directory: {output_path}"
        )
    _require_output_disjoint(validated.paths, output_path)
    revalidate_source_hashes(validated)
    expectations = asdict(validated.expectations)
    if not validated.expectations.include_nl_steering:
        expectations.pop("include_nl_steering")
    document = {
        "schema_version": 2,
        "paths": {key: str(path.resolve()) for key, path in validated.paths.hashed_files().items()}
        | {
            "prepared_dir": str(Path(validated.paths.prepared_dir).resolve()),
            **{key: str(path.resolve()) for key, path in validated.paths.runtime_files().items()},
        },
        "expectations": expectations,
        "hashes": dict(validated.hashes),
        "runtime_hashes": dict(validated.runtime_hashes),
        "runtime_byte_lengths": dict(validated.runtime_byte_lengths),
        "project_identity": {
            "package_lock_sha256": validated.project_manifest["package_lock_sha256"],
            "repository_git_commit": validated.project_manifest["repository_git_commit"],
            "target_model_tag": validated.project_manifest["target"]["model_tag"],
            "target_step": validated.project_manifest["target"]["step"],
            "self_conditioning_masked_policy": validated.project_manifest["methods"]
            ["SelfConditioningMasked"]["policy"],
        },
        "manifest_identity": {
            source: {
                "config_name": f"{validated.expectations.config_name_prefix}_{source}.yaml",
                "dataset_revision": validated.expectations.dataset_revision,
                "package_lock_sha256": validated.expectations.package_lock_sha256,
                "repository_git_commit": validated.expectations.repository_git_commit,
                "target_model": validated.expectations.target_model,
            }
            for source in ("main", "aux")
        },
    }
    if validated.expectations.include_nl_steering:
        document["natural_language"] = validated.project_manifest["methods"][NL_METHOD]
    if validated.nl_attachment is not None:
        document["nl_source_attachment"] = {
            **dict(validated.nl_attachment.identity),
            "paths": {
                key: str(path.resolve())
                for key, path in validated.nl_attachment.paths.hashed_files().items()
            },
            "runtime_byte_lengths": dict(validated.nl_attachment.runtime_byte_lengths),
        }
    atomic_write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        output_path,
    )
    return output_path


__all__ = [
    "NLAttachmentExpectations",
    "NLAttachmentPaths",
    "NLSourceAttachment",
    "SOURCE_HASH_KEYS",
    "SourceExpectations",
    "SourcePaths",
    "ValidatedSources",
    "assert_source_hashes",
    "revalidate_source_hashes",
    "validate_source_contract",
    "write_source_contract",
]
