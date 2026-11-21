"""Build filter-derived clean AxBench-BPB source siblings for BPB-Adv.

Production BPB-Adv runs fail loudly on dirty Stage 0-3 sources with one-sided
``(concept_id, prefix_text)`` groups. This module creates a clean sibling source
by applying the exact same filters the validator and attack-row loader already
use, then self-validates the result with the fail-loud source validator.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
from loguru import logger

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from steering.eval.axbench_base.direction_artifacts import validate_direction_tree
from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_validation import (
    _BASELINE_TRAINING_CORPUS_METADATA_KEY,
    BpbAdvSourceValidation,
    _load_stage_0_3_source_for_validation,
    _LoadedStage03Source,
    _SourceValidationSplit,
    _validate_loaded_stage_0_3_source,
)

CLEANING_MANIFEST_SCHEMA_VERSION = 1
CLEANING_MANIFEST_NAME = "cleaning_manifest.json"
_SYNTH_FILENAMES = frozenset(
    {
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
    }
)


@dataclass(frozen=True)
class SourceCleaningSplitResult:
    """Cleaning summary for one synth-training parquet."""

    table_name: str
    mode: str
    rows_in: int
    rows_out: int
    qc_rejected_rows: int
    out_of_source_concept_rows: int
    incomplete_group_rows: int
    dropped_incomplete_groups: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceCleaningSourceInspection:
    """Pre-clean source inspection used to build the explicit drop plan."""

    n_concepts: int
    n_positive_synth_rows: int
    n_negative_synth_rows: int
    n_positive_complete_prefixes: int
    n_negative_complete_prefixes: int
    n_dropped_incomplete_groups: int


@dataclass(frozen=True)
class SourceCleaningResult:
    """Result returned by ``clean_axbench_source_run``."""

    source_run_dir: Path
    output_dir: Path
    manifest_path: Path
    filter_unaccepted_samples: bool
    positive: SourceCleaningSplitResult
    negative: SourceCleaningSplitResult
    source_inspection: SourceCleaningSourceInspection
    validation: BpbAdvSourceValidation

    @property
    def n_dropped_incomplete_groups(self) -> int:
        return (
            len(self.positive.dropped_incomplete_groups)
            + len(self.negative.dropped_incomplete_groups)
        )

    def to_manifest(self) -> dict[str, Any]:
        self_verification: dict[str, Any] = {
            "n_dropped": self.validation.n_dropped,
            "contract_hash": self.validation.contract_hash,
            "n_concepts": self.validation.n_concepts,
            "n_positive_synth_rows": self.validation.n_positive_synth_rows,
            "n_negative_synth_rows": self.validation.n_negative_synth_rows,
            "n_positive_prefixes": self.validation.n_positive_prefixes,
            "n_negative_prefixes": self.validation.n_negative_prefixes,
        }
        fingerprint = self.validation.contract.feature_contract_fingerprint
        if fingerprint is not None:
            self_verification["feature_contract_fingerprint"] = fingerprint
            self_verification["baseline_training_corpus_sha256"] = dict(
                self.validation.contract.baseline_training_corpus_sha256
            )
        return {
            "schema_version": CLEANING_MANIFEST_SCHEMA_VERSION,
            "source_run_dir": str(self.source_run_dir),
            "output_dir": str(self.output_dir),
            "filter_unaccepted_samples": self.filter_unaccepted_samples,
            "splits": {
                "positive": dataclasses.asdict(self.positive),
                "negative": dataclasses.asdict(self.negative),
            },
            "source_inspection": dataclasses.asdict(self.source_inspection),
            "self_verification": self_verification,
        }


def default_clean_output_dir(source_run_dir: Path | str) -> Path:
    """Return the default ``<source>_paired`` sibling path."""

    source = Path(source_run_dir).resolve()
    if not source.name:
        raise AxBenchBpbAdvError(
            "cannot derive default output_dir for a filesystem root; pass --output-dir"
        )
    return source.with_name(f"{source.name}_paired")


def clean_axbench_source_run(
    source_run_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    filter_unaccepted_samples: bool = True,
) -> SourceCleaningResult:
    """Create a cleaned AxBench-BPB source sibling and self-verify it.

    The cleaned source preserves the source contract artifacts, but rewrites the
    two synth-training parquets after explicitly dropping rows that cannot form
    class-margin prefix pairs.
    """

    source_root = Path(source_run_dir)
    if output_dir is None:
        out_root = default_clean_output_dir(source_root)
    else:
        out_root = Path(output_dir)
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    _validate_output_path(source_root=source_root, out_root=out_root)

    logger.info("cleaning AxBench-BPB source {} -> {}", source_root, out_root)
    loaded_source = _load_stage_0_3_source_for_validation(
        source_root,
        filter_unaccepted_samples=filter_unaccepted_samples,
    )
    _require_cleanable_source(loaded_source)
    source_inspection = _inspection_from_loaded_source(loaded_source)
    source_directions = validate_direction_tree(
        loaded_source.paths.baselines_root(),
        error_type=AxBenchBpbAdvError,
    )

    temp_root = _make_temp_output_dir(out_root)
    installed = False
    try:
        _copy_non_synth_artifacts(source_root=source_root, out_root=temp_root)

        positive = _clean_synth_parquet(
            split=loaded_source.positive,
            output_path=temp_root / "synth_training_positive.parquet",
            concept_ids=loaded_source.concept_ids,
            filter_unaccepted_samples=filter_unaccepted_samples,
            baseline_training_corpus_sha256=(
                loaded_source.cluster_identity.baseline_training_corpus_sha256[
                    "positive"
                ]
                if loaded_source.cluster_identity is not None
                else None
            ),
        )
        negative = _clean_synth_parquet(
            split=loaded_source.negative,
            output_path=temp_root / "synth_training_negative.parquet",
            concept_ids=loaded_source.concept_ids,
            filter_unaccepted_samples=filter_unaccepted_samples,
            baseline_training_corpus_sha256=(
                loaded_source.cluster_identity.baseline_training_corpus_sha256[
                    "negative"
                ]
                if loaded_source.cluster_identity is not None
                else None
            ),
        )

        cleaned_source = _load_stage_0_3_source_for_validation(
            temp_root,
            filter_unaccepted_samples=filter_unaccepted_samples,
            contract_source_run_dir=out_root,
        )
        _assert_cleaned_output_has_no_incomplete_groups(cleaned_source)
        validation = _validate_loaded_stage_0_3_source(cleaned_source)
        source_identities = tuple(
            artifact.identity for artifact in source_directions.artifacts
        )
        if (
            validation.contract.direction_training_layer
            != source_directions.direction_training_layer
            or validation.direction_identities() != source_identities
        ):
            raise AxBenchBpbAdvError(
                "cleaned source did not preserve the original direction layer "
                "and artifact identities exactly"
            )

        result = SourceCleaningResult(
            source_run_dir=source_root,
            output_dir=out_root,
            manifest_path=out_root / CLEANING_MANIFEST_NAME,
            filter_unaccepted_samples=filter_unaccepted_samples,
            positive=positive,
            negative=negative,
            source_inspection=source_inspection,
            validation=validation,
        )
        atomic_json_dump(result.to_manifest(), temp_root / CLEANING_MANIFEST_NAME)
        _install_temp_output_dir(temp_root=temp_root, out_root=out_root)
        installed = True
    finally:
        if not installed and temp_root.exists():
            shutil.rmtree(temp_root)

    logger.info(
        "cleaned source {}: dropped {} incomplete group(s); contract_hash={}",
        out_root,
        result.n_dropped_incomplete_groups,
        validation.contract_hash,
    )
    return result


def _inspection_from_loaded_source(
    loaded: _LoadedStage03Source,
) -> SourceCleaningSourceInspection:
    return SourceCleaningSourceInspection(
        n_concepts=len(loaded.concepts),
        n_positive_synth_rows=loaded.positive.table.num_rows,
        n_negative_synth_rows=loaded.negative.table.num_rows,
        n_positive_complete_prefixes=loaded.positive.n_complete_prefixes,
        n_negative_complete_prefixes=loaded.negative.n_complete_prefixes,
        n_dropped_incomplete_groups=(
            len(loaded.positive.incomplete_prefixes)
            + len(loaded.negative.incomplete_prefixes)
        ),
    )


def _format_incomplete_groups(groups: frozenset[tuple[int, str]]) -> str:
    shown = [
        f"(concept_id={concept_id}, prefix_text={prefix_text!r})"
        for concept_id, prefix_text in sorted(groups)[:10]
    ]
    return "; ".join(shown)


def _require_cleanable_split(split: _SourceValidationSplit) -> None:
    if split.n_complete_prefixes > 0:
        return
    if split.incomplete_prefixes:
        raise AxBenchBpbAdvError(
            f"{split.table_name} ({split.mode.value} mode) has no complete "
            "class-margin pairs to clean; every in-source prefix group is "
            "missing a class side, so cleaning would produce an empty synth "
            f"split. Offending group(s): {_format_incomplete_groups(split.incomplete_prefixes)}"
        )
    raise AxBenchBpbAdvError(
        f"{split.table_name} ({split.mode.value} mode) has no source rows after "
        "QC/source-concept filtering; cleaning would produce an empty synth split"
    )


def _require_cleanable_source(loaded: _LoadedStage03Source) -> None:
    _require_cleanable_split(loaded.positive)
    _require_cleanable_split(loaded.negative)


def _assert_cleaned_output_has_no_incomplete_groups(
    loaded: _LoadedStage03Source,
) -> None:
    incomplete = {
        "positive": loaded.positive.incomplete_prefixes,
        "negative": loaded.negative.incomplete_prefixes,
    }
    n_incomplete = sum(len(groups) for groups in incomplete.values())
    if n_incomplete == 0:
        return
    details = [
        f"{name}: {_format_incomplete_groups(groups)}"
        for name, groups in incomplete.items()
        if groups
    ]
    raise AxBenchBpbAdvError(
        "internal source-cleaning error: cleaned output still has "
        f"{n_incomplete} incomplete group(s): {'; '.join(details)}"
    )


def _validate_output_path(*, source_root: Path, out_root: Path) -> None:
    if not source_root.exists():
        raise AxBenchBpbAdvError(f"source_run_dir does not exist: {source_root}")
    if not source_root.is_dir():
        raise AxBenchBpbAdvError(f"source_run_dir is not a directory: {source_root}")
    if out_root == source_root:
        raise AxBenchBpbAdvError("output_dir must differ from source_run_dir")
    try:
        out_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise AxBenchBpbAdvError(
            "output_dir must not be inside source_run_dir; refusing recursive copy"
        )
    if out_root.exists() and not out_root.is_dir():
        raise AxBenchBpbAdvError(f"output_dir exists and is not a directory: {out_root}")
    if out_root.exists() and any(out_root.iterdir()):
        raise AxBenchBpbAdvError(f"output_dir already exists and is not empty: {out_root}")


def _make_temp_output_dir(out_root: Path) -> Path:
    out_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{out_root.name}.tmp-",
            dir=out_root.parent,
        )
    )


def _install_temp_output_dir(*, temp_root: Path, out_root: Path) -> None:
    if out_root.exists():
        if any(out_root.iterdir()):
            raise AxBenchBpbAdvError(f"output_dir already exists and is not empty: {out_root}")
        out_root.rmdir()
    temp_root.rename(out_root)


def _copy_non_synth_artifacts(*, source_root: Path, out_root: Path) -> None:
    for child in sorted(source_root.iterdir(), key=lambda path: path.name):
        if child.name in _SYNTH_FILENAMES or child.name == CLEANING_MANIFEST_NAME:
            continue
        destination = out_root / child.name
        if child.is_dir():
            shutil.copytree(
                child,
                destination,
                copy_function=_copy_file,
            )
        else:
            _copy_file(child, destination)


def _copy_file(source: Path | str, destination: Path | str) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _clean_synth_parquet(
    *,
    split: _SourceValidationSplit,
    output_path: Path,
    concept_ids: set[int],
    filter_unaccepted_samples: bool,
    baseline_training_corpus_sha256: str | None = None,
) -> SourceCleaningSplitResult:
    raw_table = split.table
    concept_values = raw_table.column("concept_id").to_pylist()
    prefix_values = raw_table.column("prefix_text").to_pylist()
    qc_values = raw_table.column("qc_accepted").to_pylist()
    keep_values: list[bool] = []
    qc_rejected = 0
    out_of_source_concept = 0
    incomplete_group = 0
    for concept_id_raw, prefix_text_raw, qc_accepted_raw in zip(
        concept_values,
        prefix_values,
        qc_values,
        strict=True,
    ):
        if filter_unaccepted_samples and not bool(qc_accepted_raw):
            qc_rejected += 1
            keep_values.append(False)
            continue
        concept_id = int(concept_id_raw)
        if concept_id not in concept_ids:
            out_of_source_concept += 1
            keep_values.append(False)
            continue
        key = (concept_id, str(prefix_text_raw))
        if key in split.incomplete_prefixes:
            incomplete_group += 1
            keep_values.append(False)
            continue
        keep_values.append(True)

    output_table = raw_table.filter(pa.array(keep_values, type=pa.bool_()))
    if baseline_training_corpus_sha256 is not None:
        metadata = dict(output_table.schema.metadata or {})
        metadata[_BASELINE_TRAINING_CORPUS_METADATA_KEY.encode("utf-8")] = (
            baseline_training_corpus_sha256.encode("ascii")
        )
        output_table = output_table.replace_schema_metadata(metadata)
    if split.mode == SteeringMode.POSITIVE:
        validate_positive_qc_contract(
            output_path,
            consumer="AxBench-BPB-Adv cleaned-source publication",
            schema=output_table.schema,
        )
        validate_positive_selection_contract(
            output_path,
            consumer="AxBench-BPB-Adv cleaned-source publication",
            schema=output_table.schema,
        )
    validate_stage1_continuation_contract(
        output_path,
        consumer="AxBench-BPB-Adv cleaned-source publication",
        expected_role=(
            "stage1b_positive"
            if split.mode == SteeringMode.POSITIVE
            else "stage1c_negative"
        ),
        schema=output_table.schema,
    )
    atomic_write_parquet(output_table, output_path)
    return SourceCleaningSplitResult(
        table_name=split.table_name,
        mode=split.mode.value,
        rows_in=raw_table.num_rows,
        rows_out=output_table.num_rows,
        qc_rejected_rows=qc_rejected,
        out_of_source_concept_rows=out_of_source_concept,
        incomplete_group_rows=incomplete_group,
        dropped_incomplete_groups=tuple(
            {"concept_id": concept_id, "prefix_text": prefix_text}
            for concept_id, prefix_text in sorted(split.incomplete_prefixes)
        ),
    )
