"""Validate and merge the ten AxBench steering-method result columns."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, cast

import pandas as pd
import pyarrow as pa
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet

FULL_FACTORS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
INPUTS_PER_CONCEPT = 10
KEY_COLUMNS = ("concept_id", "input_id", "factor")

#: The filename the merged ten-method frame must carry. It is part of the
#: vendored evaluator's contract -- ``evaluate.py`` is pointed at the merged
#: frame's DIRECTORY (``--overwrite_inference_dump_dir``) and finds the frame
#: by this name inside it -- so every consumer of a merged frame refuses any
#: other name rather than scoring an unrelated Parquet.
MERGED_INPUT_FILENAME = "steering_data.parquet"
INPUT_COLUMNS = (
    "dataset_name",
    "concept_id",
    "input_concept",
    "input_id",
    "factor",
    "original_prompt",
    "steered_input",
    "input",
    "suppress_original",
    "suppress_rewrite",
    "steered_prompt",
    "defense",
    "sae_link",
    "sae_id",
)
#: Per-method ceiling on the fraction of a method's DECLARED cells that may
#: carry a whitespace-stripped-empty generation before
#: :func:`validate_steering_frame` refuses the frame.
#:
#: This is a wiring-flood bound, NOT a quality gate -- collapse detection is
#: :data:`MIN_MEDIAN_GENERATION_CHARS` (below), which reads the whole length
#: distribution instead of only its zero bin.
#:
#: Calibrated against three real populations, all measured on the l22 target:
#:
#:   * c10 run 2, post-fix .......  3 blank / 1200 declared cells  (0.25%)
#:   * projected at concept500 ...  ~150 / 60,000                  (0.25%)
#:   * c10 run 2, PRE-fix ........ 33 blank / 1200                 (2.75%)
#:
#: The observed honest sampling rate (0.25%) sits at exactly a QUARTER of the
#: bound, and the rate the real collapse produced (2.75%) is nearly 3x ABOVE
#: it. So the bound tolerates a stochastic tail with room to spare without
#: tolerating the defect this mechanism exists to catch.
#:
#: Value set by the owner, 2026-08-09: "1% or smaller". This is a pure
#: fraction bound with NO absolute-count floor -- deliberately, and applied
#: identically to all declared methods, including those with the SMALLEST
#: declared populations: the aux-like methods declare only ``factor == 1.0``
#: (see :data:`METHOD_FACTORS`), so
#: their declared-cell count is just ``INPUTS_PER_CONCEPT * max_concepts`` --
#: 20 cells at ``max_concepts=2``, the size the test suite runs at. Below 100
#: declared cells, a SINGLE blank generation already exceeds 1% (1/100 =
#: 1.0%, not > 1.0%; 1/99 and smaller populations exceed it), so a lone
#: sampler-honest blank raises there. That is the intended, owner-ruled
#: strictness, not an oversight: an earlier revision of this bound carried an
#: absolute-count floor (``max(1, floor(0.01 * n))``) that tolerated up to
#: 10% blanks at ``n=10`` -- ten times looser than the ratified bound -- and
#: was removed rather than kept as a small-n exception. Operationally this
#: means aux-method runs (and any other small-population declaration) need a
#: correspondingly cleaner sampler than a large sweep can shrug off; that
#: trade was made explicitly, not discovered later.
MAX_BLANK_GENERATION_FRACTION = 0.01

#: A declared method whose MEDIAN stripped generation length falls below this
#: many characters is treated as BROKEN, not as low-quality.
#:
#: This is a defect tripwire, deliberately NOT a quality score: it exists to
#: catch a conditioning/decode regression that empties the generations, which
#: every other guard in this module structurally cannot see (the schema
#: contract only rejects cells that are NULL, and a healthy-looking frame can
#: still be one where every generation collapsed to a token or two).
#:
#: 8 sits in the wide empty gap between the two regimes measured on the l22
#: checkpoint:
#:
#:   BROKEN
#:     c10 pre-fix, 1200 rows      median 2, p75 11, 879/1200 (73.2%) <= 5 chars
#:     dev pre-fix, 240 rows       median 2, min 0, 179/240 <= 5 chars, 9 blank
#:   HEALTHY
#:     dev regeneration, 120 SelfConditioning rows median 580.5 (min 144)
#:     bisect sweep, 25 SelfConditioning cells      median 536 (min 42)
#:
#: The widest broken observation is p75 = 11 (c10) and the smallest healthy
#: PER-CELL SelfConditioning length is 42 in these populations; nothing
#: lands between. The threshold is therefore an order of magnitude below every
#: healthy MEDIAN and still above every broken one -- it fires on collapse and
#: cannot fire on a merely terse method.
MIN_MEDIAN_GENERATION_CHARS = 8


class FrameSource(StrEnum):
    """Validated source identities used while assembling the merged frame."""

    MAIN = "main"
    AUX = "aux"
    ENCDEC_MAIN = "encdec_main"
    ENCDEC_AUX = "encdec_aux"
    PROJECT_INPUT = "project_input"
    PROJECT = "project"
    MERGED = "merged"

    @classmethod
    def coerce(cls, value: FrameSource | str) -> FrameSource:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown steering-frame source: {value!r}") from exc


class MetricKind(StrEnum):
    """Metric-column kinds declared by AxBench steering methods."""

    STRENGTH = "strength"
    PERPLEXITY = "perplexity"


class MethodSpec(NamedTuple):
    """One steering method's source, factor grid, and metric contract."""

    name: str
    source: FrameSource
    factors: tuple[float, ...]
    metric: MetricKind


_STRENGTH = MetricKind.STRENGTH
_PERPLEXITY = MetricKind.PERPLEXITY
_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec("DiffMean", FrameSource.MAIN, FULL_FACTORS, _STRENGTH),
    MethodSpec("LinearProbe", FrameSource.MAIN, FULL_FACTORS, _STRENGTH),
    MethodSpec("PromptSteering", FrameSource.AUX, (1.0,), _PERPLEXITY),
    MethodSpec("LoRA", FrameSource.AUX, (1.0,), _PERPLEXITY),
    MethodSpec("EncDecDiffMean", FrameSource.ENCDEC_MAIN, FULL_FACTORS, _STRENGTH),
    MethodSpec("EncDecLinearProbe", FrameSource.ENCDEC_MAIN, FULL_FACTORS, _STRENGTH),
    MethodSpec("EncDecPromptSteering", FrameSource.ENCDEC_AUX, (1.0,), _STRENGTH),
    MethodSpec("EncDecLoRA", FrameSource.ENCDEC_AUX, (1.0,), _STRENGTH),
    MethodSpec("SelfConditioning", FrameSource.PROJECT, FULL_FACTORS, _STRENGTH),
    MethodSpec("SelfConditioningMasked", FrameSource.PROJECT, FULL_FACTORS, _STRENGTH),
)
_NL_METHOD_SPEC = MethodSpec("NLSteering", FrameSource.PROJECT, FULL_FACTORS, _STRENGTH)
_METHOD_SPEC_BY_NAME = {spec.name: spec for spec in (*_METHOD_SPECS, _NL_METHOD_SPEC)}
METHOD_FACTORS = {spec.name: spec.factors for spec in _METHOD_SPECS}
SUPPORTED_METHOD_FACTORS = {**METHOD_FACTORS, _NL_METHOD_SPEC.name: _NL_METHOD_SPEC.factors}
_SOURCE_METHODS: Mapping[FrameSource, tuple[str, ...]] = {
    **{
        source: tuple(spec.name for spec in _METHOD_SPECS if spec.source is source)
        for source in FrameSource
        if source is not FrameSource.MERGED
    },
    FrameSource.MERGED: tuple(spec.name for spec in _METHOD_SPECS),
}
NL_PROJECT_METHODS = (*_SOURCE_METHODS[FrameSource.PROJECT], "NLSteering")
_STRENGTH_METHODS = frozenset(
    spec.name for spec in _METHOD_SPECS if spec.metric is MetricKind.STRENGTH
)
_PERPLEXITY_METHODS = frozenset(
    spec.name for spec in _METHOD_SPECS if spec.metric is MetricKind.PERPLEXITY
)
_CONCEPT_METADATA_COLUMNS = ("input_concept", "sae_link", "sae_id")
_METHOD_SUFFIXES = ("_steered_generation", "_strength", "_perplexity")


#: Per-surface wording for :func:`read_canonical_merged_frame`'s filename
#: check. The two consumers word this ONE check differently -- the vendored
#: evaluator names the requirement's origin, the roundtrip stage names its own
#: flag and the offending path -- and both wordings are pinned by their suites,
#: so the surface selects a template rather than the shared reader inventing a
#: third phrasing. Every other check the reader applies is worded identically
#: for both: there is nothing surface-specific about "this frame is empty" or
#: "these rows are not in canonical order".
class MergedFrameSurface(StrEnum):
    """Consumers of the canonical merged frame that need distinct CLI wording."""

    EVALUATION = "evaluation"
    ROUNDTRIP = "roundtrip"

    @classmethod
    def coerce(cls, value: MergedFrameSurface | str) -> MergedFrameSurface:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"unknown merged-frame reader surface: {value!r}; expected one of "
                f"{[surface.value for surface in cls]}"
            ) from exc


_MERGED_INPUT_FILENAME_ERRORS: Mapping[MergedFrameSurface, str] = MappingProxyType(
    {
        MergedFrameSurface.EVALUATION: (
            "--input must be named {filename} for the vendored evaluator"
        ),
        MergedFrameSurface.ROUNDTRIP: ("roundtrip --input must be named {filename}: {path}"),
    }
)


def _metric_column(method: str) -> str:
    try:
        metric = _METHOD_SPEC_BY_NAME[method].metric
    except KeyError:
        raise ValueError(f"unknown steering method: {method!r}") from None
    return f"{method}_{metric.value}"


def method_columns(methods: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        column
        for method in methods
        for column in (f"{method}_steered_generation", _metric_column(method))
    )


def methods_for_source(source: FrameSource | str) -> tuple[str, ...]:
    return _SOURCE_METHODS[FrameSource.coerce(source)]


def _factors(source: FrameSource) -> tuple[float, ...]:
    return (1.0,) if source in {FrameSource.AUX, FrameSource.ENCDEC_AUX} else FULL_FACTORS


def _require_exact_columns(
    frame: pd.DataFrame,
    source: FrameSource,
    methods: tuple[str, ...],
) -> None:
    expected = (*INPUT_COLUMNS, *method_columns(methods))
    if not frame.columns.is_unique:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{source} frame has duplicate columns: {duplicates}")

    actual = tuple(str(column) for column in frame.columns)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    unknown_methods = [column for column in extra if column.endswith(_METHOD_SUFFIXES)]
    if unknown_methods:
        raise ValueError(f"{source} frame has unknown method result columns: {unknown_methods}")
    if missing or extra:
        raise ValueError(
            f"{source} frame columns do not match the contract; missing={missing}, extra={extra}"
        )


def _require_dtypes(
    frame: pd.DataFrame,
    source: FrameSource,
    methods: tuple[str, ...],
) -> None:
    integer_columns = ("concept_id", "input_id", "sae_id")
    float_columns = (
        "factor",
        *(_metric_column(method) for method in methods),
    )
    text_columns = tuple(
        column
        for column in (*INPUT_COLUMNS, *method_columns(methods))
        if column not in integer_columns and column not in float_columns
    )

    for column in integer_columns:
        if str(frame[column].dtype) != "int64":
            raise ValueError(
                f"{source} column {column!r} must have dtype int64, got {frame[column].dtype}"
            )
    for column in float_columns:
        if str(frame[column].dtype) != "float64":
            raise ValueError(
                f"{source} column {column!r} must have dtype float64, got {frame[column].dtype}"
            )
    for column in text_columns:
        series = frame[column]
        if not pd.api.types.is_string_dtype(series.dtype):
            raise ValueError(
                f"{source} column {column!r} must have a string dtype, got {series.dtype}"
            )
        present = series[series.notna()]
        if not present.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{source} column {column!r} contains a non-string value")


def _require_key_product(
    frame: pd.DataFrame, source: FrameSource, concept_ids: tuple[int, ...],
) -> None:
    if frame.duplicated(list(KEY_COLUMNS)).any():
        duplicate = frame.loc[frame.duplicated(list(KEY_COLUMNS), keep=False), list(KEY_COLUMNS)]
        raise ValueError(
            f"{source} frame has duplicate (concept_id, input_id, factor) keys: "
            f"{duplicate.head().to_dict(orient='records')}"
        )

    factors = _factors(source)
    expected = pd.MultiIndex.from_product(
        (concept_ids, range(INPUTS_PER_CONCEPT), factors),
        names=KEY_COLUMNS,
    )
    actual = pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"{source} frame does not have the exact key product; "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}"
        )


def _require_input_invariants(frame: pd.DataFrame, source: FrameSource) -> None:
    for column in INPUT_COLUMNS:
        null_count = int(frame[column].isna().sum())
        if null_count:
            raise ValueError(f"{source} required input column {column!r} has {null_count} nulls")

    if not frame["dataset_name"].eq("AlpacaEval").all():
        raise ValueError(f"{source} dataset_name must be exactly 'AlpacaEval'")

    invariant_columns = tuple(column for column in INPUT_COLUMNS if column not in KEY_COLUMNS)
    grouped_inputs = frame.groupby(["concept_id", "input_id"], sort=False, dropna=False)
    for column in invariant_columns:
        if (grouped_inputs[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"{source} input column {column!r} varies across factors")

    grouped_concepts = frame.groupby("concept_id", sort=False, dropna=False)
    for column in _CONCEPT_METADATA_COLUMNS:
        if (grouped_concepts[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"{source} concept metadata column {column!r} varies across inputs")

    if (
        source in {FrameSource.AUX, FrameSource.ENCDEC_AUX}
        and frame["steered_input"].str.strip().eq("").any()
    ):
        raise ValueError(f"{source} steered_input values must be nonblank")


def _declared_mask(frame: pd.DataFrame, source: FrameSource, method: str) -> pd.Series:
    if source is FrameSource.MERGED:
        return frame["factor"].isin(SUPPORTED_METHOD_FACTORS[method])
    return pd.Series(True, index=frame.index, dtype=bool)


def _require_method_values(
    frame: pd.DataFrame,
    source: FrameSource,
    methods: tuple[str, ...],
) -> None:
    for method in methods:
        generation_column = f"{method}_steered_generation"
        metric_column = _metric_column(method)
        declared = _declared_mask(frame, source, method)

        for column in (generation_column, metric_column):
            if frame.loc[declared, column].isna().any():
                raise ValueError(f"{source} column {column!r} is null at a declared factor")
            if source is FrameSource.MERGED and frame.loc[~declared, column].notna().any():
                raise ValueError(f"merged column {column!r} must be null outside factor 1.0")

        metric = frame.loc[declared, metric_column]
        if not metric.map(math.isfinite).all():
            raise ValueError(f"{source} column {metric_column!r} must be finite where declared")
        if method in _PERPLEXITY_METHODS and not metric.gt(0.0).all():
            raise ValueError(f"{source} column {metric_column!r} must be positive where declared")


def _require_generation_health(
    frame: pd.DataFrame,
    source: FrameSource,
    methods: tuple[str, ...],
    *,
    max_blank_fraction: float = MAX_BLANK_GENERATION_FRACTION,
) -> None:
    """Collapse tripwires over every method's DECLARED generation cells.

    Two independent checks, run per method (over that method's own
    ``_declared_mask`` -- pooling methods would let one method's blanks or
    collapse mask or manufacture another's):

    1. blank flood: the fraction of whitespace-stripped-empty generations
       must stay at or below ``max_blank_fraction`` (default
       :data:`MAX_BLANK_GENERATION_FRACTION`, the owner-ratified bound
       calibrated on chat-tuned-model runs). A blank generation is honest
       sampler output (the model really did stop immediately); this bound
       catches a wiring flood, not any single blank. A caller MAY pass a
       different, separately-documented bound for a population this
       calibration was never derived from -- e.g. a base (non-chat) model,
       which emits an immediate stop token on out-of-distribution
       instruction-formatted prompts at a measurably higher honest rate even
       with zero steering applied (see
       ``merge_baseline_methods.BASELINE_MAX_BLANK_GENERATION_FRACTION``) --
       but the module-level constant itself, and every existing caller that
       omits this argument, is unaffected.
    2. length collapse: the MEDIAN stripped generation length must stay at
       or above :data:`MIN_MEDIAN_GENERATION_CHARS`. This is the check that
       sees a run where every cell is populated and every factor declared,
       but the generations are one or two characters long because the
       conditioning signal drove the decoder straight to its stop token --
       a shape :func:`_require_method_values`'s null check structurally
       cannot see.

    Null cells at declared factors already raise in
    :func:`_require_method_values`, which runs first; this function reads
    only the string values it leaves behind.
    """
    for method in methods:
        generation_column = f"{method}_steered_generation"
        declared = _declared_mask(frame, source, method)

        generations = frame.loc[declared, generation_column]
        lengths = generations.str.strip().str.len()
        cell_count = int(declared.sum())
        blank_count = int(lengths.eq(0).sum())
        blank_fraction = (blank_count / cell_count) if cell_count else 0.0
        if blank_fraction > max_blank_fraction:
            raise ValueError(
                f"{source} column {generation_column!r} for method {method!r} has "
                f"{blank_count} of {cell_count} blank generations after stripping "
                f"(fraction {blank_fraction:.4f}) exceeding the "
                f"{max_blank_fraction} bound"
            )

        median_length = float(lengths.median()) if cell_count else 0.0
        if median_length < MIN_MEDIAN_GENERATION_CHARS:
            raise ValueError(
                f"{source} column {generation_column!r} for method {method!r} has "
                f"collapsed output; median stripped length is {median_length:g}, "
                f"minimum is {MIN_MEDIAN_GENERATION_CHARS}"
            )


def resolve_concept_ids(max_concepts: int, concept_ids: object = None) -> tuple[int, ...]:
    """Resolve an explicit global subset without renumbering or accepting duplicates."""
    if type(max_concepts) is not int or max_concepts <= 0:
        raise ValueError("max_concepts must be a positive integer")
    if concept_ids is None:
        return tuple(range(max_concepts))
    if not isinstance(concept_ids, (list, tuple)) or not concept_ids:
        raise ValueError("concept_ids must be a nonempty list or tuple")
    if any(type(value) is not int or not 0 <= value < max_concepts for value in concept_ids):
        raise ValueError("concept_ids must contain global integers in [0, max_concepts)")
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("concept_ids must not contain duplicates")
    return tuple(sorted(cast(Sequence[int], concept_ids)))


def validate_steering_frame(
    frame: pd.DataFrame,
    *,
    source: FrameSource | str,
    max_concepts: int,
    check_method_values: bool = True,
    methods: tuple[str, ...] | None = None,
    max_blank_fraction: float = MAX_BLANK_GENERATION_FRACTION,
    concept_ids: object = None,
) -> None:
    """Validate one unmerged source frame or the final merged frame.

    ``methods`` overrides the source's declared method set. It exists for
    consumers pinned to a narrower frame shape than the source's full
    declared set: the axbench_adv source contract validates its ratified
    single-method project Parquet with ``methods=("SelfConditioning",)``, and
    a baseline-only (no project/encoder-decoder methods) merged frame -- e.g.
    Pythia -- validates with ``methods=("DiffMean", "LinearProbe",
    "PromptSteering", "LoRA")``; ``None`` uses the current table.

    ``max_blank_fraction`` overrides the blank-generation tripwire's bound
    (see :func:`_require_generation_health`); it exists for the same reason
    as ``methods`` -- a population this owner-ratified default was never
    calibrated against. Every existing caller keeps the default.
    """

    source = FrameSource.coerce(source)
    selected_ids = resolve_concept_ids(max_concepts, concept_ids)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if type(check_method_values) is not bool:
        raise TypeError("check_method_values must be a boolean")
    if methods is not None:
        if source not in {FrameSource.PROJECT, FrameSource.MERGED}:
            raise ValueError(
                "methods override exists only for consumers pinned to a "
                "narrower frame shape than the source's full declared set; "
                f"it is not supported for source={source.value!r}"
            )
        if (
            isinstance(methods, str)
            or not isinstance(methods, (tuple, list))
            or not methods
            or not all(isinstance(method, str) and method for method in methods)
        ):
            raise ValueError(
                "methods override must be a nonempty tuple of nonempty method-name "
                f"strings, got {methods!r}"
            )
        if len(set(methods)) != len(methods):
            raise ValueError(f"methods override must not contain duplicates, got {methods!r}")
        for method in methods:
            spec = _METHOD_SPEC_BY_NAME.get(method)
            if spec is None:
                raise ValueError(f"methods override contains unknown steering method: {method!r}")
            if source is FrameSource.PROJECT and spec.source is not FrameSource.PROJECT:
                raise ValueError(
                    f"methods override contains method {method!r} from source "
                    f"{spec.source.value!r}, not 'project'"
                )
    resolved_methods = _SOURCE_METHODS[source] if methods is None else tuple(methods)

    _require_exact_columns(frame, source, resolved_methods)
    _require_dtypes(frame, source, resolved_methods)
    _require_key_product(frame, source, selected_ids)
    _require_input_invariants(frame, source)
    if check_method_values:
        _require_method_values(frame, source, resolved_methods)
        _require_generation_health(
            frame, source, resolved_methods, max_blank_fraction=max_blank_fraction
        )


def validate_project_input_frame(frame: pd.DataFrame, *, max_concepts: int) -> None:
    """Validate the model-free input carrier used by project-only methods."""

    validate_steering_frame(
        frame,
        source=FrameSource.PROJECT_INPUT,
        max_concepts=max_concepts,
    )


def read_canonical_merged_frame(
    input_path: Path,
    *,
    surface: MergedFrameSurface | str,
    methods: tuple[str, ...] | None = None,
    max_blank_fraction: float = MAX_BLANK_GENERATION_FRACTION,
) -> pd.DataFrame:
    """Read and fully preflight a merged steering Parquet.

    THE single reader every consumer of a merged frame goes through, so the
    vendored judge and the roundtrip metric provably score the same rows, in
    the same order, under the same contract. In order: the required filename
    (:data:`MERGED_INPUT_FILENAME`), the file's existence, non-emptiness, the
    presence of ``concept_id``, the full merged-frame contract
    (:func:`validate_steering_frame`), and canonical
    ``(concept_id, input_id, factor)`` row order. An already-evaluated frame is
    rejected by the column contract.

    ``methods`` overrides the full ten-method declared set, for a merged
    frame pinned to a narrower shape (e.g. the baseline-only Pythia
    comparison); ``None`` validates the full ten-method contract.

    ``max_blank_fraction`` overrides the blank-generation tripwire's bound
    (see :func:`validate_steering_frame`); the default is the owner-ratified
    chat-model bound.

    ``concept_id`` is checked EXPLICITLY, before ``validate_steering_frame``,
    because it is the column ``max_concepts`` is inferred from: without the
    check a frame missing it inferred ``max_concepts = 0`` and surfaced as
    "max_concepts must be a positive integer", which names an argument the
    caller never passed instead of the column the frame lacks.

    ``surface`` selects the filename check's wording (see
    :data:`_MERGED_INPUT_FILENAME_ERRORS`); every other message is shared.

    TRUST MODEL. ``max_concepts`` is inferred FROM THE FRAME ITSELF rather than
    supplied, so this reader cannot detect that a validated artifact was
    truncated to a 0-based prefix of its concepts: a 500-concept frame cut down
    to concepts 0..9 is a complete, self-consistent 10-concept grid and
    self-validates here. That is accepted deliberately -- the merged Parquet is
    covered by the repo's immutable-artifact contract (it is written once,
    atomically, and never edited in place), and the roundtrip manifest pins the
    input's sha256, so a truncated input is identifiable after the fact from the
    manifest even though it is not refused here.

    Raises:
        ValueError: unknown ``surface``, wrong filename, empty frame, missing
            ``concept_id``, any :func:`validate_steering_frame` violation, or
            non-canonical row order.
        FileNotFoundError: no file at ``input_path``.
    """
    resolved_surface = MergedFrameSurface.coerce(surface)
    template = _MERGED_INPUT_FILENAME_ERRORS[resolved_surface]
    if input_path.name != MERGED_INPUT_FILENAME:
        raise ValueError(template.format(filename=MERGED_INPUT_FILENAME, path=input_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"merged steering Parquet not found: {input_path}")

    frame = pd.read_parquet(input_path)
    if frame.empty:
        raise ValueError("merged steering Parquet must not be empty")
    if "concept_id" not in frame.columns:
        raise ValueError(
            "merged steering Parquet is missing the required 'concept_id' column, so the "
            f"run's concept count cannot be inferred from it: {input_path}"
        )
    validate_steering_frame(
        frame,
        source="merged",
        max_concepts=int(frame["concept_id"].nunique()),
        methods=methods,
        max_blank_fraction=max_blank_fraction,
    )
    keys = frame.loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
    canonical_keys = keys.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if not keys.equals(canonical_keys):
        raise ValueError(
            "merged steering Parquet rows must already be in canonical "
            "(concept_id, input_id, factor) order"
        )
    return frame


def require_fresh_output_dir(path: Path, *, what: str) -> None:
    """Refuse anything but an absent or empty output directory.

    Every stage downstream of the merge writes a FIXED set of filenames, so
    writing into a populated directory would leave one run's artifacts next to
    another's with nothing to tell them apart. ``what`` names the stage in both
    messages (e.g. ``"evaluation output directory must be empty: ..."``).
    """
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"{what} output path is not a directory: {path}")
    if next(path.iterdir(), None) is not None:
        raise FileExistsError(f"{what} output directory must be empty: {path}")


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _require_matching_inputs(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_name: str,
    columns: Sequence[str],
) -> None:
    reference_view = _sorted(reference).loc[:, list(columns)]
    candidate_view = _sorted(candidate).loc[:, list(columns)]
    if not reference_view.equals(candidate_view):
        raise ValueError(f"{candidate_name} inputs do not match the main frame")


def merge_steering_frames(
    main: pd.DataFrame,
    aux: pd.DataFrame,
    encdec_main: pd.DataFrame,
    encdec_aux: pd.DataFrame,
    project: pd.DataFrame,
    *,
    max_concepts: int,
) -> pd.DataFrame:
    """Validate and merge baseline and project method frames."""

    validate_steering_frame(main, source=FrameSource.MAIN, max_concepts=max_concepts)
    validate_steering_frame(aux, source=FrameSource.AUX, max_concepts=max_concepts)
    validate_steering_frame(
        encdec_main,
        source=FrameSource.ENCDEC_MAIN,
        max_concepts=max_concepts,
    )
    validate_steering_frame(
        encdec_aux,
        source=FrameSource.ENCDEC_AUX,
        max_concepts=max_concepts,
    )
    validate_steering_frame(
        project,
        source=FrameSource.PROJECT,
        max_concepts=max_concepts,
    )

    _require_matching_inputs(
        main,
        project,
        candidate_name="project",
        columns=INPUT_COLUMNS,
    )
    _require_matching_inputs(
        main,
        encdec_main,
        candidate_name="encdec_main",
        columns=INPUT_COLUMNS,
    )
    main_factor_one = main.loc[main["factor"].eq(1.0)]
    stable_aux_columns = tuple(column for column in INPUT_COLUMNS if column != "steered_input")
    _require_matching_inputs(
        main_factor_one,
        aux,
        candidate_name="aux",
        columns=stable_aux_columns,
    )
    _require_matching_inputs(
        main_factor_one,
        encdec_aux,
        candidate_name="encdec_aux",
        columns=stable_aux_columns,
    )

    merged = _sorted(main)
    for source_frame, methods in (
        (aux, _SOURCE_METHODS[FrameSource.AUX]),
        (encdec_main, _SOURCE_METHODS[FrameSource.ENCDEC_MAIN]),
        (encdec_aux, _SOURCE_METHODS[FrameSource.ENCDEC_AUX]),
        (project, _SOURCE_METHODS[FrameSource.PROJECT]),
    ):
        additions = source_frame.loc[:, [*KEY_COLUMNS, *method_columns(methods)]]
        merged = merged.merge(
            additions,
            on=list(KEY_COLUMNS),
            how="left",
            sort=False,
            validate="one_to_one",
        )

    merged = merged.loc[
        :,
        [*INPUT_COLUMNS, *method_columns(_SOURCE_METHODS[FrameSource.MERGED])],
    ]
    validate_steering_frame(
        merged,
        source=FrameSource.MERGED,
        max_concepts=max_concepts,
    )
    return merged


def merge_steering_parquets(
    main_path: Path,
    aux_path: Path,
    encdec_main_path: Path,
    encdec_aux_path: Path,
    project_path: Path,
    output_path: Path,
    *,
    max_concepts: int,
    force: bool = False,
) -> pd.DataFrame:
    """Read, validate, merge, then write five steering Parquets."""

    if type(force) is not bool:
        raise TypeError("force must be a boolean")
    resolved_inputs = {
        Path(main_path).resolve(),
        Path(aux_path).resolve(),
        Path(encdec_main_path).resolve(),
        Path(encdec_aux_path).resolve(),
        Path(project_path).resolve(),
    }
    resolved_output = Path(output_path).resolve()
    if resolved_output in resolved_inputs:
        raise ValueError("merged output path must differ from every input path")
    if resolved_output.exists() and not force:
        raise FileExistsError(
            f"merged output already exists: {resolved_output}; pass force=True to replace it"
        )

    merged = merge_steering_frames(
        pd.read_parquet(main_path),
        pd.read_parquet(aux_path),
        pd.read_parquet(encdec_main_path),
        pd.read_parquet(encdec_aux_path),
        pd.read_parquet(project_path),
        max_concepts=max_concepts,
    )
    atomic_write_parquet(pa.Table.from_pandas(merged, preserve_index=False), resolved_output)
    return merged


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True, help="main baseline steering Parquet")
    parser.add_argument("--aux", type=Path, required=True, help="aux baseline steering Parquet")
    parser.add_argument(
        "--encdec-main",
        type=Path,
        required=True,
        help="encoder-decoder main baseline steering Parquet",
    )
    parser.add_argument(
        "--encdec-aux",
        type=Path,
        required=True,
        help="encoder-decoder aux baseline steering Parquet",
    )
    parser.add_argument("--project", type=Path, required=True, help="project-method Parquet")
    parser.add_argument("--output", type=Path, required=True, help="merged output Parquet")
    parser.add_argument("--max-concepts", type=_positive_int, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing merged output")
    args = parser.parse_args(argv)

    merged = merge_steering_parquets(
        args.main,
        args.aux,
        args.encdec_main,
        args.encdec_aux,
        args.project,
        args.output,
        max_concepts=args.max_concepts,
        force=args.force,
    )
    logger.info("Wrote {} validated rows to {}", len(merged), args.output)


if __name__ == "__main__":
    main()
