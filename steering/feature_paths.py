"""Hierarchical feature directory construction and parsing.

Builds and parses structured paths of the form:
    {root}/s{segment_size}/l{layer}/a_{act}/p_{pooling}/f{width}/m_{model}/{phase}

Used across extraction, statistics, and steering pipelines to ensure
consistent directory layouts for SAE feature data.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path

from beartype import beartype

from nanochat.common import get_base_dir
from steering.feature_io import (
    COMPRESSED_EXT,
    CorruptFeatureShardError,
    FeatureDirectoryMismatchError,
    assert_uniform_l0_tier,
    glob_features,
    read_shard_metadata,
)
from steering.refined_types import (
    NonBlankStrLike,
    NonNegativeIntLike,
    PositiveIntLike,
)
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, ActivationType, SAEWidth
from steering.segmentation import SegmentMode, segment_mode_from_metadata
from steering.sft_schema import SFTSplit, sft_split_dir
from steering.stats_io import (
    _read_metadata as _read_stats_metadata,
)
from steering.stats_io import (
    find_statistics_path,
)
from tasks.mixture_names import STEERED_MIXTURES


class PoolingMode(Enum):
    """How to aggregate SAE activations across token positions."""

    last_token = "last_token"  # activations of the final token only
    max_pool = "max_pool"  # max-pool over last context_window tokens
    mean_pool = "mean_pool"  # mean-pool over last context_window tokens


class DatasetPhase(StrEnum):
    """Training phase that produced the data being featurized."""

    BASE = "base"
    STANDARD_SFT = "standard_sft"
    STANDARD_SFT_VAL = "standard_sft_val"
    SAFETY_SFT = "safety_sft"
    SAFETY_SFT_VAL = "safety_sft_val"
    CAPABILITIES_SFT = "capabilities_sft"
    CAPABILITIES_SFT_VAL = "capabilities_sft_val"
    CHATCORE_SFT = "chatcore_sft"
    CHATCORE_SFT_VAL = "chatcore_sft_val"
    # The steered mixtures validate on STANDARD_SFT_VAL, so they mint no val
    # phase of their own (``tasks.mixtures.sft_val_family``).
    STANDARD_STEERED_SFT = "standard_steered_sft"
    STANDARD_STEERED_BALANCED_SFT = "standard_steered_balanced_sft"


@dataclass(frozen=True)
class SFTPhaseInfo:
    """SFT mixture-directory and split identity for one dataset phase."""

    mixture_dir: str
    split: SFTSplit


_SFT_PHASE_INFO: dict[DatasetPhase, SFTPhaseInfo] = {
    DatasetPhase.STANDARD_SFT: SFTPhaseInfo("standard", SFTSplit.TRAIN),
    DatasetPhase.STANDARD_SFT_VAL: SFTPhaseInfo("standard", SFTSplit.VAL),
    DatasetPhase.SAFETY_SFT: SFTPhaseInfo("safety", SFTSplit.TRAIN),
    DatasetPhase.SAFETY_SFT_VAL: SFTPhaseInfo("safety", SFTSplit.VAL),
    DatasetPhase.CAPABILITIES_SFT: SFTPhaseInfo("capabilities", SFTSplit.TRAIN),
    DatasetPhase.CAPABILITIES_SFT_VAL: SFTPhaseInfo("capabilities", SFTSplit.VAL),
    DatasetPhase.CHATCORE_SFT: SFTPhaseInfo("chatcore", SFTSplit.TRAIN),
    DatasetPhase.CHATCORE_SFT_VAL: SFTPhaseInfo("chatcore", SFTSplit.VAL),
    DatasetPhase.STANDARD_STEERED_SFT: SFTPhaseInfo("standard_steered", SFTSplit.TRAIN),
    DatasetPhase.STANDARD_STEERED_BALANCED_SFT: SFTPhaseInfo(
        "standard_steered_balanced", SFTSplit.TRAIN
    ),
}

# What a recovery command shows when the emitter cannot know the operator's
# actual steered JSONL: the required override, never a silently missing one.
STEERED_JSONL_PLACEHOLDER = "<path/to/steered.jsonl>"

# DatasetPhase -> byte-compatible SFT data subdirectory name, derived from the
# typed phase table and the sole split-directory convention.
_PHASE_TO_SFT_DIR: dict[DatasetPhase, str] = {
    phase: str(sft_split_dir(info.mixture_dir, info.split))
    for phase, info in _SFT_PHASE_INFO.items()
}


@beartype
def sft_phase_split(phase: DatasetPhase) -> SFTSplit | None:
    """Return the concrete split for an SFT phase, or None for non-SFT phases."""

    info = _SFT_PHASE_INFO.get(phase)
    return None if info is None else info.split


@beartype
def sft_phase_for(mixture_dir: str, split: SFTSplit) -> DatasetPhase:
    """Return the dataset phase identified by an SFT mixture directory and split."""

    for phase, info in _SFT_PHASE_INFO.items():
        if info.mixture_dir == mixture_dir and info.split is split:
            return phase
    valid = ", ".join(
        f"({info.mixture_dir!r}, {info.split.value!r})" for info in _SFT_PHASE_INFO.values()
    )
    raise ValueError(
        f"Unknown SFT phase identity ({mixture_dir!r}, {split.value!r}); expected one of {valid}."
    )


# Maps ActivationType -> short directory label, matching
# _GEMMASCOPE_ACTIVATION_MAP labels in steering/sae_resolver.py.
ACTIVATION_LABELS: dict[ActivationType, str] = {
    ActivationType.residual: "res",
    ActivationType.mlp: "mlp",
    ActivationType.attention: "att",
}

# Reverse mapping for parse_feature_dir.
_LABEL_TO_ACTIVATION: dict[str, ActivationType] = {v: k for k, v in ACTIVATION_LABELS.items()}

# Frozen producer behavior for current feature extraction artifacts. Callers
# that auto-construct paths to freshly extracted features should pass this;
# explicit feature_dir overrides remain the escape hatch for legacy artifacts.
PRODUCER_IGNORE_BOS_FEATURES = True

# Canonical trailing leaf that marks a turn-delimited feature directory. Owned
# here because :func:`build_feature_dir` writes it and :func:`parse_feature_dir`
# reads it; fail-closed consumers import this rather than re-spelling it.
TURN_DELIMITED_LEAF = "turn_delimited"


@dataclass(frozen=True)
class FeatureDirectoryComponents:
    """Structural components parsed from a hierarchical feature directory path.

    Returned by :func:`parse_feature_dir` to replace error-prone magic-key
    dict access with typed attribute access.

    Invariants:
        - ``segment_size > 0``
        - ``layer >= 0``
        - ``model`` is non-empty
    """

    segment_size: int
    layer: int
    activation_type: ActivationType
    pooling: PoolingMode
    width: SAEWidth
    model: str
    dataset_phase: DatasetPhase
    context_independent_features: bool
    ignore_bos_features: bool = False
    segment_mode: SegmentMode = SegmentMode.fixed_tokens


class SegmentSizeMismatch(Exception):
    """Raised when ``segment_size`` disagrees across config, path, and metadata."""


# Escape sentinels for model-name sanitization (see `sanitize_model_name`).
#
# Rationale: HuggingFace-style model IDs contain `/` (e.g. "google/gemma-2-2b"),
# which is a path separator and would create unwanted directory nesting that
# breaks `parse(build(x)) == x` round-trip. We encode each `/` on write and
# decode it on read. To keep the encoding bijective even when the input
# itself contains the escape sentinel, we first escape the literal `_`
# character (with `__`), then encode `/` as `_S`. Decoding is a single
# left-to-right scanner that consumes `_` + next char as an escape pair.
#
# Legacy model names that don't contain `/` or `_` (e.g. "gemma-2-2b",
# "gemma-2-9b") encode to themselves — so existing on-disk directories
# keep working without migration.
_MODEL_ESCAPE = "_"  # escape-leader character
_MODEL_ESC_UNDERSCORE = "__"  # encodes a literal `_`
_MODEL_ESC_SLASH = "_S"  # encodes a literal `/`


def sanitize_model_name(name: str) -> str:
    """Encode a model name so it fits in a single filesystem component.

    `/` -> `_S` and literal `_` -> `__`. Underscore escaping runs first so
    that input strings which happen to contain `_S` (looking like an encoded
    slash) do not collide with genuine slash encodings on decode.

    Invariant: `unsanitize_model_name(sanitize_model_name(x)) == x` for
    every non-empty `str` `x`.
    """
    return name.replace("_", _MODEL_ESC_UNDERSCORE).replace("/", _MODEL_ESC_SLASH)


def unsanitize_model_name(encoded: str) -> str:
    """Inverse of :func:`sanitize_model_name`.

    Left-to-right single-pass scanner: at each `_` we look at the next
    character to determine the escape meaning. `__` -> `_` and `_S` -> `/`.
    Any unrecognised escape raises `ValueError` to catch silent corruption
    rather than guessing.
    """
    out: list[str] = []
    i = 0
    n = len(encoded)
    while i < n:
        c = encoded[i]
        if c != _MODEL_ESCAPE:
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            raise ValueError(
                f"Malformed encoded model name {encoded!r}: "
                f"trailing escape character at position {i}"
            )
        nxt = encoded[i + 1]
        if nxt == "_":
            out.append("_")
        elif nxt == "S":
            out.append("/")
        else:
            raise ValueError(
                f"Malformed encoded model name {encoded!r}: unknown escape '_{nxt}' at position {i}"
            )
        i += 2
    return "".join(out)


def _parse_strict_bool_metadata(raw: str, *, field: str) -> bool:
    """Parse a bool from a metadata string, rejecting unknown variants.

    Production shards store `"true"` / `"false"` (lowercase) per
    :func:`steering.feature_io.save_features`. Anything else is a corruption
    or schema drift and must surface, not silently coerce.

    Args:
        raw: Raw metadata string.
        field: Name of the field, used in the error message for traceability.

    Returns:
        ``True`` if ``raw == "true"``, ``False`` if ``raw == "false"``.

    Raises:
        ValueError: For any other input (including ``"True"``, ``"YES"``,
            ``"tru"``, the empty string).
    """
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError(f"Invalid bool metadata for {field!r}: {raw!r}; expected 'true' or 'false'")


@beartype
def build_feature_dir(
    root: str | Path,
    segment_size: PositiveIntLike,
    layer: NonNegativeIntLike,
    activation_type: ActivationType,
    pooling: PoolingMode,
    width: SAEWidth,
    model: NonBlankStrLike,
    dataset_phase: DatasetPhase,
    context_independent_features: bool = False,
    ignore_bos_features: bool = False,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> Path:
    """Build a hierarchical feature directory path.

    Args:
        root: Base directory for all feature data.
        segment_size: Number of tokens per segment (must be > 0).
        layer: Transformer layer index (must be >= 0).
        activation_type: Which activation site (residual/mlp/attention).
        pooling: Pooling strategy over token positions.
        width: SAE dictionary width.
        model: Model name (e.g. "gemma-2-2b" or "google/gemma-2-2b").
            Must be non-empty. HuggingFace-style ``org/model`` IDs are
            encoded via :func:`sanitize_model_name` so the `m_*` component
            stays a single filesystem segment (and `parse_feature_dir` inverts).
        dataset_phase: Training phase of the source data.
        context_independent_features: When true, appends a trailing
            ``context_independent`` leaf: the resulting path has 8 components
            instead of 7. When false (default), the path is bit-identical to
            pre-context-independent-feature output.
        ignore_bos_features: When true, appends a trailing ``no_bos`` leaf
            AFTER the ``context_independent`` leaf (if present). The canonical
            order begins ``…/{phase}[/context_independent][/no_bos]``.
        segment_mode: Appends ``turn_delimited`` after all other leaves when
            set to ``turn_delimited_fixed_tokens``.

    Returns:
        Canonical path of the form
        ``{root}/s{segment_size}/l{layer}/a_{label}/p_{pooling}/f{width}/m_{model}/``
        ``{phase}[/context_independent][/no_bos][/turn_delimited]``.

    Raises:
        BeartypeCallHintParamViolation: If ``segment_size``, ``layer``, or
            ``model`` violates its annotated domain.
    """
    act_label = ACTIVATION_LABELS[activation_type]
    encoded_model = sanitize_model_name(model)

    base = (
        Path(root)
        / f"s{segment_size}"
        / f"l{layer}"
        / f"a_{act_label}"
        / f"p_{pooling.value}"
        / f"f{width.value}"
        / f"m_{encoded_model}"
        / dataset_phase.value
    )
    if context_independent_features:
        base = base / "context_independent"
    if ignore_bos_features:
        base = base / "no_bos"
    if segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        base = base / TURN_DELIMITED_LEAF
    return base


# Regex patterns for each path component.
_RE_SEGMENT = re.compile(r"^s(\d+)$")
_RE_LAYER = re.compile(r"^l(\d+)$")
_RE_ACTIVATION = re.compile(r"^a_(.+)$")
_RE_POOLING = re.compile(r"^p_(.+)$")
_RE_WIDTH = re.compile(r"^f(.+)$")
_RE_MODEL = re.compile(r"^m_(.+)$")


def parse_feature_dir(path: str | Path) -> FeatureDirectoryComponents:
    """Parse a hierarchical feature directory path into its components.

    Accepts the canonical generalized layout
    ``.../s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/``
    ``{phase}[/context_independent][/no_bos][/turn_delimited]``.

    Args:
        path: A feature directory path (str or Path). Only trailing components
            are used; any leading prefix is ignored.

    Returns:
        A :class:`FeatureDirectoryComponents` dataclass with typed fields.

    Raises:
        ValueError: If the path components do not match the expected format.
    """
    parts = Path(path).parts
    stripped = 0
    segment_mode = SegmentMode.fixed_tokens
    ignore_bos_flag = False
    context_independent_flag = False
    if parts and parts[-1] == TURN_DELIMITED_LEAF:
        segment_mode = SegmentMode.turn_delimited_fixed_tokens
        parts = parts[:-1]
        stripped += 1
    if parts and parts[-1] == "no_bos":
        ignore_bos_flag = True
        parts = parts[:-1]
        stripped += 1
    if parts and parts[-1] == "context_independent":
        context_independent_flag = True
        parts = parts[:-1]
        stripped += 1

    # Detect leaves that were not stripped because they appeared in the wrong
    # order (e.g. …/no_bos/context_independent instead of the canonical
    # …/context_independent/no_bos).  If either leaf name is still present in
    # the trailing position after stripping, it will land in the phase slot of
    # the 7-tuple and produce a confusing parse error.  Raise an explicit
    # "dataset phase" error so the failure is recognisably actionable.
    _KNOWN_LEAVES = frozenset({"context_independent", "no_bos", TURN_DELIMITED_LEAF})
    if parts and parts[-1] in _KNOWN_LEAVES:
        raise ValueError(
            f"Unknown dataset phase '{parts[-1]}'; "
            f"valid phases: {[p.value for p in DatasetPhase]} "
            f"(leaf '{parts[-1]}' is in an invalid position — canonical order is "
            f"…/{{phase}}[/context_independent][/no_bos][/turn_delimited])"
        )

    if len(parts) < 7:
        raise ValueError(
            f"Feature path must have at least 7 components (up to 10 with "
            f"context_independent, no_bos, and/or turn_delimited leaves), got "
            f"{len(parts) + stripped}: {path}"
        )

    seg_s, layer_s, act_s, pool_s, width_s, model_s, phase_s = parts[-7:]

    m = _RE_SEGMENT.match(seg_s)
    if not m:
        raise ValueError(f"Cannot parse segment size from '{seg_s}'; expected 's{{int}}'")
    segment_size = int(m.group(1))

    m = _RE_LAYER.match(layer_s)
    if not m:
        raise ValueError(f"Cannot parse layer from '{layer_s}'; expected 'l{{int}}'")
    layer = int(m.group(1))

    m = _RE_ACTIVATION.match(act_s)
    if not m:
        raise ValueError(f"Cannot parse activation type from '{act_s}'; expected 'a_{{label}}'")
    act_label = m.group(1)
    if act_label not in _LABEL_TO_ACTIVATION:
        raise ValueError(
            f"Unknown activation label '{act_label}'; "
            f"valid labels: {sorted(_LABEL_TO_ACTIVATION.keys())}"
        )
    activation_type = _LABEL_TO_ACTIVATION[act_label]

    m = _RE_POOLING.match(pool_s)
    if not m:
        raise ValueError(f"Cannot parse pooling mode from '{pool_s}'; expected 'p_{{label}}'")
    pool_label = m.group(1)
    try:
        pooling = PoolingMode(pool_label)
    except ValueError as exc:
        raise ValueError(
            f"Unknown pooling mode '{pool_label}'; valid modes: {[p.value for p in PoolingMode]}"
        ) from exc

    m = _RE_WIDTH.match(width_s)
    if not m:
        raise ValueError(f"Cannot parse width from '{width_s}'; expected 'f{{label}}'")
    width_label = m.group(1)
    try:
        width = SAEWidth(width_label)
    except ValueError as exc:
        raise ValueError(
            f"Unknown width '{width_label}'; valid widths: {[w.value for w in SAEWidth]}"
        ) from exc

    m = _RE_MODEL.match(model_s)
    if not m:
        raise ValueError(f"Cannot parse model from '{model_s}'; expected 'm_{{name}}'")
    encoded_model = m.group(1)
    try:
        model = unsanitize_model_name(encoded_model)
    except ValueError as exc:
        raise ValueError(f"Cannot parse model from '{model_s}': {exc}") from exc

    try:
        dataset_phase = DatasetPhase(phase_s)
    except ValueError as exc:
        raise ValueError(
            f"Unknown dataset phase '{phase_s}'; valid phases: {[p.value for p in DatasetPhase]}"
        ) from exc

    return FeatureDirectoryComponents(
        segment_size=segment_size,
        layer=layer,
        activation_type=activation_type,
        pooling=pooling,
        width=width,
        model=model,
        dataset_phase=dataset_phase,
        context_independent_features=context_independent_flag,
        ignore_bos_features=ignore_bos_flag,
        segment_mode=segment_mode,
    )


def resolve_output_dir(
    output_dir: str | None,
    feature_root: str,
    segment_size: int,
    layer: int,
    activation_type: ActivationType,
    pooling: PoolingMode,
    width: SAEWidth,
    model: str,
    dataset_phase: str | None,
    context_independent_features: bool = False,
    ignore_bos_features: bool = False,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
) -> Path:
    """Resolve the feature output directory.

    If ``output_dir`` is explicitly set, use it as-is (the ``context_independent_features``
    and ``ignore_bos_features`` flags are NOT appended — users who set an explicit path
    are presumed to have encoded their intent there). Otherwise auto-construct via
    ``build_feature_dir``, appending the ``context_independent`` and/or ``no_bos``
    leaves when the respective flags are true.
    """
    if output_dir is not None:
        return Path(output_dir)

    if dataset_phase:
        try:
            phase = DatasetPhase(dataset_phase)
        except ValueError as exc:
            raise ValueError(
                f"Unknown dataset_phase {dataset_phase!r}; "
                f"valid phases: {[p.value for p in DatasetPhase]}"
            ) from exc
    else:
        phase = DatasetPhase.BASE
    return build_feature_dir(
        root=feature_root,
        segment_size=segment_size,
        layer=layer,
        activation_type=activation_type,
        pooling=pooling,
        width=width,
        model=model,
        dataset_phase=phase,
        context_independent_features=context_independent_features,
        ignore_bos_features=ignore_bos_features,
        segment_mode=segment_mode,
    )


def resolve_data_dir(
    data_dir: str | None,
    dataset_phase: str | None,
) -> Path:
    """Resolve the input data directory based on dataset phase.

    - explicit data_dir: use it for any phase.
    - phase=None or "base": use default base data location.
    - phase=SFT variant: use {base_dir}/sft_data/{mixture}/ by default.
    """
    if data_dir is not None:
        return Path(data_dir)

    if dataset_phase is None or dataset_phase == DatasetPhase.BASE.value:
        # Reconstruct default base data path without importing DATA_DIR
        # (matches nanochat/dataset.py: DATA_DIR = os.path.join(base_dir, "base_data"))
        return Path(get_base_dir()) / "base_data"

    phase = DatasetPhase(dataset_phase)
    sft_subdir = _PHASE_TO_SFT_DIR[phase]
    return Path(get_base_dir()) / "sft_data" / sft_subdir


def check_sft_data_ready(data_dir: str | Path, dataset_phase: str) -> tuple[bool, str]:
    """Check if SFT parquet data exists and is complete.

    Returns (ok, message). ok=True if data is ready, False with helpful msg.
    """
    data_dir = Path(data_dir)
    phase = DatasetPhase(dataset_phase)
    info = _SFT_PHASE_INFO[phase]
    sft_subdir = _PHASE_TO_SFT_DIR[phase]
    convert_cmd = (
        "uv run python -m scripts.preprocess.convert_sft_to_parquet "
        f"mixture={info.mixture_dir} split={info.split.value}"
        # A steered conversion cannot run without its JSONL, so the command
        # names the required override instead of failing on paste.
        + (
            f" steered_jsonl={STEERED_JSONL_PLACEHOLDER}"
            if info.mixture_dir in STEERED_MIXTURES
            else ""
        )
    )

    if not data_dir.exists():
        return False, (
            f"SFT parquet data for '{sft_subdir}' not found at {data_dir}\n  Run: {convert_cmd}"
        )

    meta_path = data_dir / "conversion_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        completed = meta.get("completed_shards", 0)
        total = meta.get("total_shards", 0)
        if completed < total:
            return False, (
                f"SFT parquet conversion for '{sft_subdir}' is incomplete "
                f"({completed}/{total} shards)\n"
                f"  Run: {convert_cmd}  (will resume)"
            )

    parquets = list(data_dir.glob("*.parquet"))
    if not parquets:
        return False, (
            f"SFT parquet data for '{sft_subdir}' not found at {data_dir}\n  Run: {convert_cmd}"
        )

    return True, "OK"


def verify_feature_dir(feature_dir: str | Path) -> None:
    """Verify that feature files match the parameters encoded in their directory path.

    Checks:
        1. Directory exists and contains at least one feature shard.
        2. A statistics file (feature_statistics.safetensors) exists.
        3. Each .safetensors.zst shard's metadata (layer, segment_size, pooling)
           matches the values parsed from the directory path.
        4. The statistics file's metadata (layer, segment_size, pooling_mode,
           n_sae_features) matches the directory path.

    Args:
        feature_dir: Path to a hierarchical feature directory.

    Raises:
        FeatureDirectoryMismatchError: If any check fails. The message lists
            all detected mismatches, not just the first.
    """
    feature_dir = Path(feature_dir)
    errors: list[str] = []

    if not feature_dir.is_dir():
        raise FeatureDirectoryMismatchError(f"Feature directory does not exist: {feature_dir}")

    # Parse expected values from path
    expected = parse_feature_dir(feature_dir)

    # Discover shards
    shards = glob_features(feature_dir)
    if not shards:
        raise FeatureDirectoryMismatchError(f"No feature shard files found in {feature_dir}")

    # Check statistics file. Accumulate (rather than early-raise) so that
    # other invariants — notably cross-shard l0_tier consistency below —
    # are still surfaced together in a single error report.
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        errors.append(f"Statistics file not found in {feature_dir}")

    # Verify shard metadata (compressed safetensors only)
    for shard_path in shards:
        if not shard_path.name.endswith(COMPRESSED_EXT):
            continue
        shard_name = shard_path.name
        try:
            meta = read_shard_metadata(shard_path)
        except CorruptFeatureShardError as exc:
            errors.append(f"Shard {shard_name}: {exc}")
            continue

        if "layer" in meta:
            actual_layer = int(meta["layer"])
            if actual_layer != expected.layer:
                errors.append(
                    f"Shard {shard_name}: layer={actual_layer}, expected {expected.layer} from path"
                )

        if "segment_size" in meta:
            actual_seg = int(meta["segment_size"])
            if actual_seg != expected.segment_size:
                errors.append(
                    f"Shard {shard_name}: segment_size={actual_seg}, "
                    f"expected {expected.segment_size} from path"
                )

        if "pooling" in meta:
            actual_pool = meta["pooling"]
            expected_pool = expected.pooling.value
            if actual_pool != expected_pool:
                errors.append(
                    f"Shard {shard_name}: pooling={actual_pool!r}, "
                    f"expected {expected_pool!r} from path"
                )

        if "segment_mode" not in meta:
            if expected.segment_mode is SegmentMode.turn_delimited_fixed_tokens:
                errors.append(
                    f"Shard {shard_name}: shard predates segment_mode metadata but "
                    "the directory carries the turn_delimited leaf"
                )
        else:
            try:
                shard_mode = segment_mode_from_metadata(meta, source=f"Shard {shard_name}")
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if shard_mode is not expected.segment_mode:
                    errors.append(
                        f"Shard {shard_name}: segment_mode={shard_mode.value!r}, "
                        f"expected {expected.segment_mode.value!r} from path"
                    )

        # NEW: context_independent_features consistency.
        # We require strict 'true'/'false' (lowercase) per the production
        # encoder in feature_io._optimize_dtype/save_features. Anything else
        # is corruption or a schema drift and surfaces as a hard error rather
        # than silently coercing to False.
        meta_context_independent_raw = meta.get("context_independent_features", "false")
        try:
            if isinstance(meta_context_independent_raw, str):
                meta_context_independent = _parse_strict_bool_metadata(
                    meta_context_independent_raw,
                    field="context_independent_features",
                )
            else:
                meta_context_independent = bool(meta_context_independent_raw)
        except ValueError as exc:
            errors.append(
                f"Shard {shard_name}: context_independent_features metadata is malformed: {exc}"
            )
            continue
        if meta_context_independent != expected.context_independent_features:
            errors.append(
                f"Shard {shard_name}: context_independent_features={meta_context_independent}, "
                f"expected {expected.context_independent_features} from path "
                f"(path leaf '/context_independent' present: "
                f"{expected.context_independent_features})"
            )

        # ignore_bos_features consistency (same strict-bool contract).
        meta_ignore_bos_raw = meta.get("ignore_bos_features", "false")
        try:
            if isinstance(meta_ignore_bos_raw, str):
                meta_ignore_bos = _parse_strict_bool_metadata(
                    meta_ignore_bos_raw, field="ignore_bos_features"
                )
            else:
                meta_ignore_bos = bool(meta_ignore_bos_raw)
        except ValueError as exc:
            errors.append(f"Shard {shard_name}: ignore_bos_features metadata is malformed: {exc}")
            continue
        if meta_ignore_bos != expected.ignore_bos_features:
            errors.append(
                f"Shard {shard_name}: ignore_bos_features={meta_ignore_bos}, "
                f"expected {expected.ignore_bos_features} from path "
                f"(path leaf '/no_bos' present: "
                f"{expected.ignore_bos_features})"
            )

    # Verify statistics metadata (only when a stats file was found)
    stats_meta = _read_stats_metadata(stats_path) if stats_path is not None else {}

    if "layer" in stats_meta:
        actual_layer = int(stats_meta["layer"])
        if actual_layer != expected.layer:
            errors.append(f"Statistics: layer={actual_layer}, expected {expected.layer} from path")

    if "segment_size" in stats_meta:
        actual_seg = int(stats_meta["segment_size"])
        if actual_seg != expected.segment_size:
            errors.append(
                f"Statistics: segment_size={actual_seg}, expected {expected.segment_size} from path"
            )

    if "pooling_mode" in stats_meta:
        actual_pool = stats_meta["pooling_mode"]
        expected_pool = expected.pooling.value
        if actual_pool != expected_pool:
            errors.append(
                f"Statistics: pooling_mode={actual_pool!r}, expected {expected_pool!r} from path"
            )

    if "n_sae_features" in stats_meta:
        actual_n = int(stats_meta["n_sae_features"])
        expected_n = SAE_WIDTH_TO_NUM_FEATURES[expected.width]
        if actual_n != expected_n:
            errors.append(
                f"Statistics: n_sae_features={actual_n}, "
                f"expected {expected_n} from path (width={expected.width.value})"
            )

    if stats_path is not None:
        if "segment_mode" not in stats_meta:
            if expected.segment_mode is SegmentMode.turn_delimited_fixed_tokens:
                errors.append(
                    "Statistics: statistics predates segment_mode metadata but the "
                    "directory carries the turn_delimited leaf"
                )
        else:
            try:
                stats_mode = segment_mode_from_metadata(stats_meta, source="Statistics")
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if stats_mode is not expected.segment_mode:
                    errors.append(
                        f"Statistics: segment_mode={stats_mode.value!r}, "
                        f"expected {expected.segment_mode.value!r} from path"
                    )

    # Guarded stats-level ignore_bos_features check. Pre-existing stats files
    # lack the key → not flagged (back-compat).
    if "ignore_bos_features" in stats_meta:
        try:
            stats_ignore_bos = _parse_strict_bool_metadata(
                stats_meta["ignore_bos_features"], field="ignore_bos_features"
            )
        except ValueError as exc:
            errors.append(f"Statistics: ignore_bos_features malformed: {exc}")
        else:
            if stats_ignore_bos != expected.ignore_bos_features:
                errors.append(
                    f"Statistics: ignore_bos_features={stats_ignore_bos}, "
                    f"expected {expected.ignore_bos_features} from path"
                )

    # Single cross-shard l0_tier consistency rule (shared with
    # compute_statistics via steering.feature_io.assert_uniform_l0_tier).
    # Accumulate rather than early-raise to preserve the
    # all-problems-in-one-report contract. Granularity note: the shared
    # helper raises on the FIRST shard with an unrecognized tier value
    # (rather than the pre-refactor per-shard enumeration); the surviving
    # contract is "a mixed/invalid-tier dir is surfaced as one accumulated
    # error alongside any other errors", which the regression tests pin.
    # A corrupt shard intentionally surfaces TWICE: once from the per-shard
    # loop above (``Shard <name>: ...``) and once here, because
    # assert_uniform_l0_tier -> read_l0_tier_from_shard re-raises
    # CorruptFeatureShardError (a ValueError) which this except catches.
    # Both lines name the shard path; "loud and doubled" beats silent.
    # Do NOT de-dupe: it would re-couple this shared cross-shard l0 rule.
    try:
        assert_uniform_l0_tier(shards)
    except (FeatureDirectoryMismatchError, ValueError) as exc:
        errors.append(str(exc))

    if errors:
        raise FeatureDirectoryMismatchError(
            f"Feature directory verification failed for {feature_dir}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _parse_segment_size_from_path(path: Path) -> int | None:
    """Walk up the path looking for an ``s{int}`` component."""
    for part in path.parts:
        m = _RE_SEGMENT.match(part)
        if m:
            return int(m.group(1))
    return None


def verify_segment_size(
    shard_path_or_dir: str | Path,
    expected_segment_size: int,
) -> None:
    """Cross-check ``segment_size`` across config, directory, and shard metadata."""
    path = Path(shard_path_or_dir)

    if path.is_dir():
        shards = glob_features(path)
        if not shards:
            raise FileNotFoundError(f"verify_segment_size: no feature shards found under {path}")
        shard_path = shards[0]
    else:
        shard_path = path

    dir_segment_size = _parse_segment_size_from_path(shard_path.parent)
    if dir_segment_size is None:
        raise ValueError(
            f"verify_segment_size: path {shard_path} has no 's{{N}}' "
            "component; cannot cross-check against directory convention"
        )

    meta = read_shard_metadata(shard_path)
    meta_raw = meta.get("segment_size")
    if meta_raw is None:
        raise SegmentSizeMismatch(
            f"verify_segment_size: shard {shard_path} is missing 'segment_size' "
            f"in safetensors metadata (expected cfg={expected_segment_size}, "
            f"dir={dir_segment_size})"
        )
    try:
        meta_segment_size = int(meta_raw)
    except (TypeError, ValueError) as exc:
        raise SegmentSizeMismatch(
            f"verify_segment_size: shard {shard_path} has non-integer "
            f"'segment_size' metadata {meta_raw!r}: {exc}"
        ) from exc

    if (
        expected_segment_size != dir_segment_size
        or expected_segment_size != meta_segment_size
        or dir_segment_size != meta_segment_size
    ):
        raise SegmentSizeMismatch(
            f"segment_size mismatch for {shard_path}: "
            f"cfg={expected_segment_size}, "
            f"dir={dir_segment_size} (from 's{{N}}' path component), "
            f"metadata={meta_segment_size} (from shard safetensors header). "
            "All three must agree."
        )
