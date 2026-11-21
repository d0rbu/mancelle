"""Per-direction max-activation steering scale (issue #1228).

Owns the statistic that replaces the SAE-feature ``max_mag`` scale (#588):
each DiffMean/LinearProbe direction's OWN maximal activation at its hook
site, bound into the direction artifact at fit time.

Definition (owner decision 2026-08-02; AxBench-faithful — see the spec's §2/§4
deviation table): for unit direction d at ``layer_hooks.diffmean_and_probe_layer``
of the target model,

    max_activation = max over the concept's concept_rich AND concept_avoiding
                     fit rows of max over continuation-token positions of
                     (h_token . d)

Signed (no ReLU, no abs), float32 on CPU, no probe intercept, no rounding.
A ZERO direction (the documented no-op case) stores 0.0. A NONZERO direction
whose max is <= 0 or non-finite RAISES at fit time — writing it would make
factor grids meaningless and silently flip negative-mode steering downstream.
The separate signed-statistic helper also serves raw additive counters, which
do not use calibration as a scale. There is no fallback of any kind (AxBench's own ``else 50`` /
``.get(id, 1.0)`` fallbacks are deliberately NOT replicated).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch as t
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load as load_safetensors_bytes

from nanochat.atomic_io import atomic_write_text
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.safetensors_io import read_safetensors_metadata_from_bytes

__all__ = (
    "MAX_ACTIVATION_METADATA_KEY",
    "STEERING_SCALE_MARKER_NAME",
    "DirectionScalePreflight",
    "DirectionScaleRequest",
    "compute_direction_max_activation",
    "compute_signed_direction_max_activation",
    "direction_scale_identity",
    "encode_max_activation",
    "enforce_steering_scale_identity",
    "normalize_direction",
    "parse_max_activation",
    "preflight_direction_scales",
    "require_direction_max_activation",
)

#: Safetensors text-metadata key on stage-3 direction artifacts. Presence of
#: this key IS the schema epoch — artifacts written before #1228 lack it and
#: every scaled consumer refuses them (re-run stage 3a/3b).
MAX_ACTIVATION_METADATA_KEY = "max_activation"

#: Kept byte-for-byte so existing run directories are checked at the same path.
#: The CONTENT epoch changes from ``maxact:`` to ``dirmaxact:`` below.
STEERING_SCALE_MARKER_NAME = ".steering_scale_identity"

_RERUN_REMEDY = (
    "artifacts predating issue #1228 do not carry max_activation; re-run stage "
    "3a/3b under a fresh run tag"
)


@dataclass(frozen=True)
class DirectionScaleRequest:
    """One direction artifact a stage will consume during this invocation."""

    family: str
    target_model: str
    mode: str
    concept_id: int
    path: Path | None = None
    payload: bytes | None = None


@dataclass(frozen=True)
class DirectionScalePreflight:
    """Validated direction scales and their resume-guard identity."""

    values: dict[tuple[str, str, str, int], float]
    identity: str


def compute_signed_direction_max_activation(
    row_token_acts: Iterable[t.Tensor],
    direction: t.Tensor,
    *,
    surface: str,
) -> float:
    """Return the signed statistic without interpreting it as a steering scale.

    ``row_token_acts``: one ``(T_i, C)`` float tensor per fit row (positive and
    negative rows of the concept), continuation tokens only. ``direction`` is
    the final saved direction (unit norm, or exactly zero for the documented
    zero-direction case). ``surface`` names the concept/mode for errors.
    """
    if direction.ndim != 1 or direction.numel() == 0:
        raise AxBenchBaseError(
            f"{surface}: direction must be a non-empty 1-D tensor, got shape "
            f"{tuple(direction.shape)}"
        )
    direction_f32 = direction.detach().to(dtype=t.float32, device="cpu")
    if not bool(t.isfinite(direction_f32).all().item()):
        raise AxBenchBaseError(f"{surface}: direction contains non-finite values")
    is_zero_direction = bool((direction_f32 == 0.0).all().item())

    overall: float | None = None
    n_rows = 0
    for row_idx, acts in enumerate(row_token_acts):
        n_rows += 1
        if acts.ndim != 2 or acts.shape[1] != direction_f32.shape[0]:
            raise AxBenchBaseError(
                f"{surface}: row activation shape {tuple(acts.shape)} does not "
                f"match direction width {direction_f32.shape[0]}"
            )
        if acts.shape[0] == 0:
            raise AxBenchBaseError(
                f"{surface}: a fit row has no continuation tokens; the stage-3 "
                "capture contract guarantees at least one"
            )
        proj = acts.detach().to(dtype=t.float32, device="cpu") @ direction_f32
        # Guard EVERY row before folding into the running max: Python's
        # max(finite, nan) returns the finite value, so a NaN/-inf projection
        # in any row after the first would otherwise be silently swallowed
        # (order-dependent E2 violation; found in PR-1a adversarial review).
        # Finite activations can also overflow to +/-inf in the f32 dot
        # product, so upstream activation checks cannot replace this.
        if not bool(t.isfinite(proj).all().item()):
            raise AxBenchBaseError(
                f"{surface}: non-finite projection in fit row {row_idx}; the "
                "target model produced non-finite activations on this row, or "
                "the projection overflowed float32"
            )
        row_max = float(proj.max().item())
        overall = row_max if overall is None else max(overall, row_max)
    if n_rows == 0:
        raise AxBenchBaseError(f"{surface}: no fit rows provided")
    if is_zero_direction:
        return 0.0
    # Every per-row projection was finite, so `overall` is finite here by
    # construction. Scale consumers impose their sign rule separately.
    assert overall is not None
    return overall


def compute_direction_max_activation(
    row_token_acts: Iterable[t.Tensor],
    direction: t.Tensor,
    *,
    surface: str,
) -> float:
    """Require a positive steering scale for every nonzero direction."""
    overall = compute_signed_direction_max_activation(row_token_acts, direction, surface=surface)
    if overall <= 0.0 and bool(
        (direction.detach().to(dtype=t.float32, device="cpu") != 0.0).any().item()
    ):
        raise AxBenchBaseError(
            f"{surface}: non-positive max_activation ({overall}) for a nonzero "
            "direction; a non-positive scale would zero or flip the applied "
            "steering. Inspect the fit corpus/model."
        )
    return overall


def encode_max_activation(value: float) -> str:
    """Exact-round-trip encoding for the artifact metadata (repr of float)."""
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise AxBenchBaseError(f"refusing to encode non-finite max_activation ({v})")
    return repr(v)


def normalize_direction(
    direction: t.Tensor,
    *,
    error_cls: type[Exception] = AxBenchBaseError,
) -> t.Tensor:
    """Return a unit-norm direction, preserving the zero-vector object."""
    if not bool(t.isfinite(direction).all().item()):
        raise error_cls("direction tensor contains non-finite values")
    norm = direction.float().norm()
    if norm.item() == 0.0:
        return direction
    return direction / norm.to(device=direction.device, dtype=direction.dtype)


def parse_max_activation(
    metadata: Mapping[str, str] | None,
    *,
    surface: str,
    error_cls: type[Exception] = AxBenchBaseError,
) -> float:
    """Parse the artifact field, distinguishing absent and invalid values."""
    if metadata is None or MAX_ACTIVATION_METADATA_KEY not in metadata:
        raise error_cls(
            f"{surface}: missing {MAX_ACTIVATION_METADATA_KEY!r} metadata; {_RERUN_REMEDY}"
        )
    raw_value = metadata[MAX_ACTIVATION_METADATA_KEY]
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise error_cls(
            f"{surface}: invalid max_activation {raw_value!r}; expected a finite "
            f"float. {_RERUN_REMEDY}"
        ) from exc
    if not math.isfinite(value):
        raise error_cls(
            f"{surface}: invalid max_activation {raw_value!r}; value must be finite. "
            f"{_RERUN_REMEDY}"
        )
    return value


def require_direction_max_activation(
    metadata: Mapping[str, str] | None,
    *,
    direction: t.Tensor,
    expected_concept_id: int,
    expected_mode: str,
    expected_target_model: str,
    expected_layer: int,
    surface: str,
    error_cls: type[Exception] = AxBenchBaseError,
) -> float:
    """Validate artifact binding and return its direction-owned scale."""
    actual = {} if metadata is None else metadata
    expected = {
        "concept_id": str(expected_concept_id),
        "mode": expected_mode,
        "target_model": expected_target_model,
        "layer": str(expected_layer),
    }
    mismatches = [
        f"{field}: expected {expected_value!r}, got {actual.get(field)!r}"
        for field, expected_value in expected.items()
        if actual.get(field) != expected_value
    ]
    if mismatches:
        raise error_cls(
            f"{surface}: direction metadata mismatch ({'; '.join(mismatches)}). {_RERUN_REMEDY}"
        )

    value = parse_max_activation(actual, surface=surface, error_cls=error_cls)
    if value < 0.0:
        raise error_cls(
            f"{surface}: negative max_activation ({value}) is invalid; a negative "
            f"scale would flip steering sign. {_RERUN_REMEDY}"
        )
    if not bool(t.isfinite(direction).all().item()):
        raise error_cls(f"{surface}: direction tensor contains non-finite values; {_RERUN_REMEDY}")
    is_zero_direction = bool((direction == 0.0).all().item())
    if (value == 0.0) != is_zero_direction:
        raise error_cls(
            f"{surface}: zero-direction max_activation mismatch: direction_zero="
            f"{is_zero_direction}, max_activation={value}; zero is valid exactly "
            f"when the direction tensor is all-zero. {_RERUN_REMEDY}"
        )
    return value


def direction_scale_identity(
    values: Mapping[tuple[str, str, str, int], float],
) -> str:
    """Fingerprint the exact consumed direction/value set for resume guards."""
    items: list[tuple[list[str | int], str]] = []
    for key, raw_value in sorted(values.items()):
        value = float(raw_value)
        if not math.isfinite(value):
            raise AxBenchBaseError(
                f"direction-scale identity cannot encode non-finite value {value} for {key!r}"
            )
        family, target_model, mode, concept_id = key
        items.append(([family, target_model, mode, int(concept_id)], repr(value)))
    payload = json.dumps(items, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"dirmaxact:{digest}"


def preflight_direction_scales(
    requests: Iterable[DirectionScaleRequest],
    *,
    expected_layer: int,
    stage_label: str,
    error_cls: type[Exception] = AxBenchBaseError,
) -> DirectionScalePreflight:
    """Validate every unique direction artifact a stage will consume.

    Durable-resume BPB and Stage-4 generation callers also compare the value
    reopened by each scalar consumer with this preflight snapshot. That closes
    the unclustered artifact-regeneration window for the stamped rows. BPB-Adv
    whole-stage runs have no resume marker and retain a bounded preflight-to-
    consumer window equal to their pre-existing unguarded direction-vector
    replacement window.
    """
    values: dict[tuple[str, str, str, int], float] = {}
    offenders: list[str] = []
    seen: set[tuple[str, str, str, int]] = set()
    for request in requests:
        key = (
            request.family,
            request.target_model,
            request.mode,
            int(request.concept_id),
        )
        if key in seen:
            continue
        seen.add(key)
        source = str(request.path) if request.path is not None else "certified payload"
        surface = (
            f"{stage_label} {request.family} direction artifact {source} "
            f"for concept={request.concept_id} mode={request.mode!r} "
            f"target_model={request.target_model!r}"
        )
        try:
            if (request.path is None) == (request.payload is None):
                raise error_cls(f"{surface}: exactly one of path or payload must be supplied")
            if request.path is not None:
                with safe_open(str(request.path), framework="pt", device="cpu") as handle:
                    metadata = handle.metadata()
                    direction = handle.get_tensor("direction")
            else:
                assert request.payload is not None
                tensors = load_safetensors_bytes(request.payload)
                metadata = read_safetensors_metadata_from_bytes(request.payload)
                direction = tensors["direction"]
            values[key] = require_direction_max_activation(
                metadata,
                direction=direction,
                expected_concept_id=request.concept_id,
                expected_mode=request.mode,
                expected_target_model=request.target_model,
                expected_layer=expected_layer,
                surface=surface,
                error_cls=error_cls,
            )
        except Exception as exc:
            offenders.append(f"{source}: {exc}")

    if offenders:
        details = "\n".join(f"  - {offender}" for offender in offenders)
        raise error_cls(
            f"{stage_label} direction-scale preflight found invalid artifact(s):\n{details}"
        )

    for (family, target_model, mode, concept_id), value in sorted(values.items()):
        logger.info(
            "{} direction max_activation: family={} target_model={} mode={} concept_id={} value={}",
            stage_label,
            family,
            target_model,
            mode,
            concept_id,
            value,
        )
    return DirectionScalePreflight(
        values=values,
        identity=direction_scale_identity(values),
    )


def enforce_steering_scale_identity(
    marker_path: Path, *, identity: str, resume_active: bool
) -> None:
    """Stamp or validate the scoring determinant for safe stage resume."""
    if marker_path.exists():
        stored = marker_path.read_text(encoding="utf-8").strip()
        if stored != identity:
            if stored.startswith("maxact:"):
                regime = (
                    "the pre-#1228 SAE max_act regime, while this run resolves "
                    "the per-direction max_activation regime"
                )
            else:
                regime = "a DIFFERENT steering-scale regime"
            raise AxBenchBaseError(
                f"steering_scale resume guard: {marker_path} records identity "
                f"{stored!r} from {regime}; this run resolves {identity!r}. "
                "Resuming would blend incomparable steering magnitudes under "
                "identical coefficient labels. Use a fresh run_tag or clear the "
                "run directory."
            )
        return
    if resume_active:
        raise AxBenchBaseError(
            f"steering_scale resume guard: {marker_path.parent} holds existing "
            "scaled rows but no steering-scale identity marker "
            f"({marker_path.name}); they predate the issue #1228 per-direction "
            "max_activation guard or the marker was removed. Resuming would blend "
            "incomparable steering magnitudes. Use a fresh run_tag or clear the "
            "run directory."
        )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(identity, marker_path)
