"""Target-segment conditioning helpers for encoder-decoder evals.

Progression evals need two different conditioning sources:

* real extracted features for already-observed prompt/context tokens;
* an explicit feature dictionary for the final generated/scored segment.

This module owns the small file formats used by that path.  It intentionally
does not run Gemma/GemmaScope extraction; it only consumes deterministic cache
artifacts created ahead of evaluation.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import torch as t
from beartype import beartype

from errors import FeatureVocabError
from steering.conditional.checkpoint_feature_contract import (
    assert_feature_vocab_width,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.pooling_op import PoolingOp
from steering.refined_types import (
    NonEmptyNonNegativeIntMappingListLike,
    NonNegativeIntLike,
    NonNegativeIntMappingLike,
    NonNegativeIntTupleLike,
    PositiveIntLike,
)
from steering.segmentation_primitives import SegmentMode

if TYPE_CHECKING:
    from steering.eval.turn_context_contract import TurnContextContract


class ContextRoutingMode(StrEnum):
    """Coordinate and boundary ownership used by the static target wrapper."""

    DECODER_ALIGNED = "decoder_aligned"
    TURN_RAW_SOURCE = "turn_raw_source"


class TargetConditioningMode(StrEnum):
    """How target/generated tokens select encoder conditioning slots."""

    CURRENT = "current"
    NEXT = "next"

    @classmethod
    def parse(cls, value) -> TargetConditioningMode:
        if isinstance(value, cls):
            return value
        if value is None:
            raise ValueError(
                f"target_conditioning_mode must be one of current or next; got {value!r}"
            )
        normalized = str(value).lower()
        if normalized in {"current", "current_segment"}:
            return cls.CURRENT
        if normalized in {"next", "next_segment"}:
            return cls.NEXT
        raise ValueError(
            f"target_conditioning_mode must be one of current or next; got {value!r}"
        )

    @classmethod
    def from_next_segment_offset(cls, enabled: bool) -> TargetConditioningMode:
        return cls.NEXT if bool(enabled) else cls.CURRENT


def context_cache_generation_instructions() -> str:
    """Return operator-facing instructions for shared cached-context evals."""

    return (
        "Generate this cache before running conditioned progression eval:\n"
        "  1. Render the exact eval contexts that will be passed to the model "
        "(same templates, seeds, BOS policy, and CORE oversized-example "
        "filtering). Use evals=core,chat,harmfulness as needed.\n"
        "  2. Record the exact nanochat token_ids for each context; these token "
        "IDs are the cache key.\n"
        "  3. Run the Gemma/GemmaScope SAE feature extraction pipeline on the "
        "matching context text using the same extraction settings as the "
        "encoder-decoder training features.\n"
        "  4. Filter/discretize the extracted features into encoder bucket IDs "
        "and write a safetensors context cache rooted at "
        "conditioning.context_cache_dir/c{model_max_seq_len}/{feature_selection_hash[:16]} "
        "with manifest.json plus contexts_*.safetensors.zst shards.\n"
        "  5. Use uv run python -m scripts.eval.build_eval_context_cache "
        "stage=enumerate contexts_file=... context_parquet_dir=... "
        "model_max_seq_len=..., then run steering.extract_features on that "
        "parquet, then run stage=convert contexts_file=... feature_dir=... "
        "stats_path=... output_dir=... n_sae_features=... num_buckets=..., "
        "then run the same script with stage=validate. For CORE, set "
        "model_max_seq_len to the checkpoint "
        "metadata max_seq_len/model.max_seq_len; stage=convert writes "
        "output_dir/c{model_max_seq_len}/{feature_selection_hash[:16]}/manifest.json.\n"
        "  6. Run the cache builder's validation/parity tests before launching "
        "progression eval; missing or stale rows are fatal."
    )


def target_real_cache_generation_instructions() -> str:
    """Return operator-facing instructions for target_real eval caches."""

    return (
        "Generate this target_real cache before running target_real eval:\n"
        "  1. Enumerate target_real rows with evals=core_target, "
        "evals=chat_target, or evals=core_target,chat_target.\n"
        "  2. Render exact full eval token sequences and their exact "
        "prompt/context prefixes using the same templates, seeds, BOS policy, "
        "and CORE oversized-example filtering used at eval time; token IDs "
        "are the cache key.\n"
        "  3. Run uv run python -m scripts.eval.build_eval_context_cache "
        "stage=enumerate contexts_file=... context_parquet_dir=... "
        "model_max_seq_len=..., then run steering.extract_features on that "
        "parquet, then run stage=convert contexts_file=... feature_dir=... "
        "stats_path=... output_dir=... n_sae_features=... num_buckets=..., "
        "then run stage=validate.\n"
        "  4. Point target_real_configs[*].cache_dir at the target cache root, "
        "or omit cache_dir and let the eval auto-resolve "
        "eval_caches/{core_target|chat_target}/.../c{model_max_seq_len}/"
        "{feature_selection_hash[:16]}. stage=convert writes manifest.json "
        "plus contexts_*.safetensors.zst shards under "
        "output_dir/c{model_max_seq_len}/{feature_selection_hash[:16]}."
    )


@beartype
@dataclass(frozen=True)
class ContextFeatureSegment:
    """Selected, bucketed SAE features for one observed context segment.

    Unary element and character-offset domains are enforced by the annotated
    phantom types. Cache-specific relationships such as equal feature/bucket
    lengths, unique feature IDs, ordered bounds, and manifest upper bounds stay
    with the cache validators that have the context needed to report them.
    """

    feature_ids: NonNegativeIntTupleLike
    buckets: NonNegativeIntTupleLike
    char_start: NonNegativeIntLike
    char_end: NonNegativeIntLike


def validate_context_token_assignments(
    token_ids: Sequence[int],
    segments: Sequence[ContextFeatureSegment],
    assignments: tuple[int, ...] | None,
) -> None:
    """Check persisted ownership against the exact token and segment counts."""
    if assignments is None:
        return
    if len(assignments) != len(token_ids):
        raise ValueError("context token ownership length differs from token count")
    if any(type(index) is not int or index < 0 or index >= len(segments) for index in assignments):
        raise ValueError("context token ownership contains an invalid segment index")
    if any(left > right for left, right in pairwise(assignments)):
        raise ValueError("context token ownership must be monotone")


@dataclass(frozen=True)
class PreparedTurnContext:
    """Exact lookup key plus prompt-only raw slots and SFT input assignments."""

    token_ids: tuple[int, ...]
    segments: tuple[ContextFeatureSegment, ...]
    token_assignments: t.Tensor
    target_start_pos: int


class SingleEntryContextProvider:
    """Context provider mapping one exact token sequence to feature segments."""

    def __init__(
        self,
        token_ids: Sequence[int],
        segments: tuple[ContextFeatureSegment, ...],
        *,
        segment_mode: SegmentMode = SegmentMode.fixed_tokens,
        token_assignments: tuple[int, ...] | None = None,
        turn_contract: TurnContextContract | None = None,
    ) -> None:
        validate_context_token_assignments(token_ids, segments, token_assignments)
        self._token_ids = tuple(int(token) for token in token_ids)
        self._segments = segments
        self.segment_mode = segment_mode
        self.token_assignments = token_assignments
        self.turn_contract = turn_contract

    def get(
        self,
        token_ids: Sequence[int],
    ) -> tuple[ContextFeatureSegment, ...] | None:
        if tuple(int(token) for token in token_ids) != self._token_ids:
            return None
        return self._segments


def token_context_key(token_ids: Sequence[int]) -> str:
    """Return the stable manifest key for a nanochat token sequence."""

    h = hashlib.sha256()
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return f"tokens_sha256:{h.hexdigest()}"


@beartype
def validate_feature_dict(
    features: NonNegativeIntMappingLike | None,
    n_sae_features: PositiveIntLike,
    num_buckets: PositiveIntLike,
    *,
    name: str = "feature config",
    feature_vocab: FeatureVocab | None = None,
) -> None:
    """Validate feature ids and pre-discretized bucket indices.

    A non-None ``feature_vocab`` additionally requires every requested ORIGINAL
    SAE id to be present in the compacted vocabulary. A compacted encoder has no
    embedding row for an excluded id, and its ``-1`` remap sentinel is unsafe
    under ``torch.compile`` because ``nn.Embedding(-1)`` silently returns the
    last row. Excluded ids therefore raise rather than degrade.
    """

    assert_feature_vocab_width(feature_vocab, n_sae_features, name=name)
    if not features:
        return
    for fid, bucket in features.items():
        if fid >= n_sae_features:
            raise ValueError(f"{name}: feature id {fid} is out of range [0, {n_sae_features})")
        if bucket >= num_buckets:
            raise ValueError(f"{name}: bucket {bucket} is out of range [0, {num_buckets})")
    if feature_vocab is None:
        return

    original_ids = feature_vocab.original_ids
    outside_ids = []
    for feature_id in features:
        index = bisect_left(original_ids, feature_id)
        if index == len(original_ids) or original_ids[index] != feature_id:
            outside_ids.append(feature_id)
    if not outside_ids:
        return

    outside_ids.sort()
    reported = outside_ids[:8]
    omitted = len(outside_ids) - len(reported)
    suffix = "" if omitted == 0 else f" … (+{omitted} more)"
    raise FeatureVocabError(
        f"{name}: {len(outside_ids)} requested feature ids are outside the "
        f"loaded feature vocabulary: {reported}{suffix}"
    )


@beartype
@dataclass(frozen=True)
class SingleTargetSegment:
    """One target slot. Features broadcast to every assistant position.

    Used by static-YAML conditioning, by Phase 2 'pooled' mode, and by
    empty target-delta conditioning (features={}). Features may be empty
    only when the wrapper still receives real prompt/context features from
    a context provider.

    Note: ``frozen=True`` prevents field rebinding only; the ``features``
    dict's contents remain mutable post-construction.
    """

    features: NonNegativeIntMappingLike


@beartype
@dataclass(frozen=True)
class MultiTargetSegments:
    """N target slots; per-position routing across the assistant span.

    Used by Phase 2 'lockstep' mode. ``token_assignments[t]`` is the segment
    index (0-indexed into ``segments``) for the t-th decoder position in the
    complete target span. The vector includes mask=0 forced/tool-output
    positions between routed mask=1 positions because those positions still
    consume decoder/KV positions. The wrapper clamps ``t`` to
    ``len(token_assignments) - 1`` for decode steps past the ground-truth
    length.

    Turn raw-source routing starts this vector at the terminal prompt's
    ``<assistant_start>`` input. Its marker consumes assignment 0; the first
    generated input consumes assignment 1. Fixed routing retains its historical
    prefill-relative decode origin.

    Note: ``frozen=True`` prevents field rebinding only; ``segments`` list
    contents and ``token_assignments`` tensor elements remain mutable
    post-construction.
    """

    segments: NonEmptyNonNegativeIntMappingListLike
    token_assignments: t.Tensor

    def __post_init__(self) -> None:
        if self.token_assignments.ndim != 1 or self.token_assignments.dtype != t.int64:
            raise ValueError(
                "token_assignments must be 1-D int64; got "
                f"ndim={self.token_assignments.ndim}, "
                f"dtype={self.token_assignments.dtype}"
            )
        if self.token_assignments.numel() == 0:
            raise ValueError("token_assignments must be non-empty")
        max_idx = int(self.token_assignments.max().item())
        min_idx = int(self.token_assignments.min().item())
        if min_idx < 0 or max_idx >= len(self.segments):
            raise ValueError(
                f"token_assignments range [{min_idx}, {max_idx}] is outside "
                f"[0, {len(self.segments)})"
            )
        # Monotonic non-decreasing: a non-monotonic assignment vector cannot
        # describe a contiguous target-segment
        # tiling (would imply a token rejoining an earlier segment) and
        # would silently produce wrong cross-attention routing if it
        # reached the wrapper unchecked.
        if self.token_assignments.numel() > 1:
            diffs = self.token_assignments[1:] - self.token_assignments[:-1]
            if int(diffs.min().item()) < 0:
                bad_idx = int((diffs < 0).nonzero(as_tuple=True)[0][0].item()) + 1
                raise ValueError(
                    f"token_assignments must be monotonic non-decreasing; "
                    f"position {bad_idx} ({int(self.token_assignments[bad_idx].item())}) "
                    f"is below the previous position "
                    f"({int(self.token_assignments[bad_idx - 1].item())})"
                )
        # Defensive clone: isolate from post-construction in-place mutation by
        # the caller. ``frozen=True`` prevents field rebinding but not tensor
        # element mutation. Cloning here ensures the wrapper's view of
        # token_assignments is stable regardless of what the caller does with
        # the original tensor after construction.
        object.__setattr__(self, "token_assignments", self.token_assignments.detach().clone())


@beartype
@dataclass(frozen=True)
class SelfConditionedTarget:
    """Self-conditioned generation: the encoder is driven by the model's
    own feature-prediction head, not by operator-supplied features.

    The wrapper detects this type via ``isinstance(target,
    SelfConditionedTarget)`` and routes to the self-conditioning code path.
    It carries no static target slots: prefill encodes only real context
    features, then the wrapper appends the first non-empty predicted slot to
    the dynamically growing encoder state owned by
    :class:`steering.conditional.encoder_decoder.self_conditioned.SelfCondRuntime`.

    **Prefill contract.** The prompt itself is conditioned with real
    per-segment features from ``context_provider``. Self-conditioning
    starts only at the FIRST GENERATED token, where the wrapper seeds
    a fresh slot from the prompt's tail fp_logits and routes generated
    tokens onto it.

    **Decode contract.** Every ``self_cond_segment_size`` (nanochat)
    tokens, pool the buffered fp_logits and append a fresh predicted
    slot. seg_map for each decoded token points at the latest slot.
    """


type TargetConditioning = SingleTargetSegment | MultiTargetSegments | SelfConditionedTarget


# ---- PR-C: target_real_configs (one-off eval consumer types) ----


@dataclass(frozen=True)
class LockstepTargetRealConfig:
    name: str
    cache_dir: Path


@dataclass(frozen=True)
class PooledTargetRealConfig:
    name: str
    cache_dir: Path
    pooling_op: PoolingOp = PoolingOp.BUCKET_MAX


type TargetRealConfig = LockstepTargetRealConfig | PooledTargetRealConfig


def parse_target_real_configs(
    raw_list,
    *,
    validate_cache_dir: bool = False,
    auto_resolve_cache_dir: Callable[[dict], str | Path] | None = None,
) -> list[TargetRealConfig]:
    """Parse a list of raw dicts (from Hydra YAML) into TargetRealConfig variants.

    Each entry must have ``name`` and ``cache_dir``. The ``mode`` field selects
    the variant: ``"lockstep"`` or ``"pooled"``. Pooled variants may carry
    ``pooling_op`` (default: ``bucket_max``); lockstep variants must not. Both
    Context features are always loaded from the independently extracted exact
    prompt-prefix row; there is no context mode switch.

    ``cache_dir`` is NOT validated for existence here by default; the cache may
    be on a remote filesystem that is unavailable at parse time. The consumer
    (``SafetensorsContextFeatureProvider`` construction) raises a clear
    ``FileNotFoundError`` if the cache is missing at eval time.

    Args:
        raw_list: List of raw config dicts (or None/empty for no configs).
        validate_cache_dir: If True, check that each ``cache_dir`` path exists
            at parse time and raise ``FileNotFoundError`` if it is missing.
            Useful for local-only evals where early validation is desirable;
            leave False (default) for remote/cluster evals.
        auto_resolve_cache_dir: Optional callable invoked when an entry has
            ``cache_dir`` set to ``None`` (or missing entirely). The callable
            receives the entry dict and must return a ``str`` or ``Path``.
            When ``None`` (default), missing/null ``cache_dir`` raises the
            existing ValueError. Callers that want to derive the path
            from cfg.conditioning + suite name should pass a closure here
            (see :func:`steering.eval.target_real_runner.parse_target_real_configs_from_cfg`).
    """
    if not raw_list:
        return []
    out: list[TargetRealConfig] = []
    for idx, raw in enumerate(raw_list):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"target_real_configs[{idx}]: each entry must be a YAML mapping with "
                f"'name'/'mode'/'cache_dir' keys; got {type(raw).__name__} {raw!r}. "
                f"Example: '- {{name: pooled, mode: pooled}}' (mode='pooled' or 'lockstep')."
            )
        if "name" not in raw:
            raise ValueError(f"target_real_configs[{idx}]: missing 'name' field")
        # Determine cache_dir, optionally auto-resolving when missing/null.
        raw_cache_dir = raw.get("cache_dir")
        if raw_cache_dir is None:
            if auto_resolve_cache_dir is None:
                raise ValueError(
                    f"target_real_configs[{idx}] ({raw.get('name')}): missing 'cache_dir' field"
                )
            resolved = auto_resolve_cache_dir(raw)
            cache_dir = Path(str(resolved))
        elif str(raw_cache_dir) == "???":
            raise ValueError(
                f"target_real_configs[{idx}] ({raw.get('name')}): cache_dir is mandatory; "
                "replace ??? with the target cache directory, set cache_dir=null for "
                "identity-based auto-resolution, or set target_real_configs=null to "
                "disable target_real eval"
            )
        else:
            cache_dir = Path(str(raw_cache_dir))
        if "mode" not in raw:
            name_hint = raw.get("name", f"<index {idx}>")
            raise ValueError(
                f"target_real_configs[{idx}] ({name_hint}): missing 'mode' field "
                f"(must be 'lockstep' or 'pooled')"
            )
        mode = raw.get("mode")
        name = str(raw["name"])
        if "context_mode" in raw:
            raise ValueError(
                f"target_real_configs[{idx}] ({name}): context_mode is no longer "
                "supported. Target-real eval always conditions on cached full-text "
                "prompt/context features."
            )
        if validate_cache_dir and not cache_dir.exists():
            raise FileNotFoundError(
                f"target_real_configs[{idx}] ({name}): cache_dir does not exist: {cache_dir}"
            )
        if mode == "lockstep":
            if "pooling_op" in raw:
                raise ValueError(
                    f"target_real_configs[{idx}] ({name}): mode='lockstep' must not "
                    f"set pooling_op (lockstep dispatch routes per-token, no pooling)"
                )
            out.append(
                LockstepTargetRealConfig(
                    name=name,
                    cache_dir=cache_dir,
                )
            )
        elif mode == "pooled":
            pooling_op_str = raw.get("pooling_op", "bucket_max")
            try:
                pooling_op = PoolingOp(pooling_op_str)
            except ValueError as e:
                raise ValueError(
                    f"target_real_configs[{idx}] ({name}): invalid pooling_op {pooling_op_str!r}"
                ) from e
            out.append(
                PooledTargetRealConfig(
                    name=name,
                    cache_dir=cache_dir,
                    pooling_op=pooling_op,
                )
            )
        else:
            raise ValueError(
                f"target_real_configs[{idx}] ({name}): unknown mode {mode!r} "
                f"(must be 'lockstep' or 'pooled')"
            )
    # Validate name uniqueness: the name is the discriminator for output yaml
    # paths ({output_dir}/target_real/{name}.yaml). A duplicate would silently
    # overwrite an earlier result.
    seen: dict[str, int] = {}
    for entry_idx, trc in enumerate(out):
        if trc.name in seen:
            raise ValueError(
                f"target_real_configs: duplicate name {trc.name!r} (entries "
                f"{seen[trc.name]} and {entry_idx}). Each TargetRealConfig must "
                f"have a unique name; the name is the discriminator for output "
                f"yaml paths ({{output_dir}}/target_real/{{name}}.yaml). If you "
                f"want both pooled and lockstep evals on the same target cache, "
                f"give them distinct names (e.g. 'safety_pooled', 'safety_lockstep')."
            )
        seen[trc.name] = entry_idx
    return out
