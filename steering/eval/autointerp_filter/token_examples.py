"""Token-level example construction and sampling for intruder scoring."""

from __future__ import annotations

import bisect
import hashlib
import json
import random
import re
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import pyarrow.parquet as pq
import torch as t
from beartype import beartype
from beartype.vale import Is
from loguru import logger
from sae_lens import SAE
from transformer_lens import HookedTransformer

from nanochat.dataset import list_parquet_files
from steering.eval.autointerp_filter.batch_validation import (
    PRODUCTION_BATCH_128_PROFILE,
    BatchCalibrationContract,
    BatchSurfaceKind,
    BatchToleranceProfile,
    assert_batch_difference_within_profile,
    assert_candidate_repeat_exact,
    assert_runtime_matches_profile,
    runtime_numeric_fingerprint,
)
from steering.eval.autointerp_filter.batching import (
    MaterializationPhase,
    PreparedBatch,
    PreparedDoc,
    TargetBlock,
    build_target_blocks,
    target_plan_hash,
)
from steering.eval.autointerp_filter.cache import TokenExampleCache
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig
from steering.eval.autointerp_filter.contracts import intruder_trial_key
from steering.eval.autointerp_filter.document_blacklist import (
    ResolvedDocumentBlacklist,
    document_key,
    resolve_document_blacklist,
)
from steering.eval.autointerp_filter.source_index import (
    SourceIndexData,
    SourceIndexRequest,
    _validated_source_doc_ref,
    load_or_build_source_index,
    validate_source_index_cache,
)
from steering.eval.autointerp_filter.targets import target_payload_hash
from steering.eval.autointerp_filter.types import (
    AutoInterpConfigError,
    AutoInterpDataError,
    IntruderTrial,
    ScoringTarget,
    SourceDocRef,
    TokenExampleProvider,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.extract_features import (
    _get_activations_native,
    _truncate_to_training_window,
    build_native_model,
)
from steering.feature_io import (
    MetadataKey,
    TensorKey,
    glob_features,
    load_features,
    read_shard_metadata,
)
from steering.feature_paths import parse_feature_dir, resolve_data_dir
from steering.models import resolve_tokenizer
from steering.refined_types import (
    NonEmptyNonNegativeIntTuple,
    NonEmptyNonNegativeIntTupleLike,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntMappingLike,
    NonNegativeIntTuple,
    NonNegativeIntTupleLike,
    PositiveInt,
    PositiveIntLike,
    StrTuple,
    StrTupleLike,
    UnitIntervalLike,
)
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    resolve_sae,
)
from steering.segmentation import (
    SegmentMode,
    resolve_bos_token_id,
    segment_mode_from_metadata,
)
from steering.stats_io import find_statistics_path, load_statistics

_SHARD_INDEX_RE = re.compile(r"shard_(\d+)")
_TOKEN_CACHE_METHOD = "token_examples_v6"
_BATCH_TOKEN_CACHE_METHOD = "token_examples_v7_ci_block_two_stream_singleton_v1"
_BATCH_GPU_SINGLETON_CONCURRENCY = 2
_TRIAL_METHOD = "delphi_intruder_token_v3"
# Character budget for the per-row-group source-text cache. Deliberately close
# to the peak of the previous whole-shard LRU-2 (two ~240 MiB columns ≈ 480 MiB
# resident) so this change does NOT regress token-example peak RAM: the win
# comes from reading one ~1024-row group instead of a whole 53k-row column
# (~1/52 the I/O), not from a bigger cache. Measured in text characters, not
# resident bytes (CPython str overhead makes RSS a small multiple). Pure
# performance knob: eviction only changes how often a row group is re-read,
# never which text a doc resolves to.
_ROW_GROUP_CACHE_CHAR_BUDGET = 512 * 1024**2


if TYPE_CHECKING:
    _NonEmptyStrFrozenSetLike = frozenset[str]
    _BatchPairTimingSinkLike = list[dict[str, object]]
else:
    _NonEmptyStrFrozenSetLike = Annotated[
        frozenset,
        Is[
            lambda value: (
                isinstance(value, frozenset)
                and bool(value)
                and all(isinstance(item, NonEmptyStr) for item in value)
            )
        ],
    ]
    _BatchPairTimingSinkLike = list


def _exception_chain_contains_cuda_oom(exc: BaseException) -> bool:
    """Recognize CUDA OOM after backend adapters wrap the original error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, t.cuda.OutOfMemoryError):
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


@dataclass(frozen=True, init=False)
class _DocActivations:
    source_text_hash: NonEmptyStr
    token_offset: NonNegativeInt
    token_ids: NonNegativeIntTuple
    token_strings: StrTuple
    sae_feature_ids: NonEmptyNonNegativeIntTuple
    sae_acts: t.Tensor

    @beartype
    def __init__(
        self,
        source_text_hash: NonEmptyStrLike,
        token_offset: NonNegativeIntLike,
        token_ids: NonNegativeIntTupleLike,
        token_strings: StrTupleLike,
        sae_feature_ids: NonEmptyNonNegativeIntTupleLike,
        sae_acts: t.Tensor,
    ) -> None:
        object.__setattr__(
            self,
            "source_text_hash",
            NonEmptyStr.parse(source_text_hash),
        )
        object.__setattr__(self, "token_offset", NonNegativeInt.parse(token_offset))
        object.__setattr__(self, "token_ids", NonNegativeIntTuple.parse(token_ids))
        object.__setattr__(self, "token_strings", StrTuple.parse(token_strings))
        object.__setattr__(
            self,
            "sae_feature_ids",
            NonEmptyNonNegativeIntTuple.parse(sae_feature_ids),
        )
        object.__setattr__(self, "sae_acts", sae_acts)


def infer_n_sae_features(feature_dir: Path, configured: int | None) -> int:
    """Resolve n_sae_features from config, stats, or feature-dir width."""
    if configured is not None:
        return int(configured)
    stats_path = find_statistics_path(feature_dir)
    if stats_path is not None:
        return int(load_statistics(stats_path).metadata.n_sae_features)
    return SAE_WIDTH_TO_NUM_FEATURES[parse_feature_dir(feature_dir).width]


@beartype
def positive_highlight_indices(
    ref: TokenWindowRef,
    *,
    activation_threshold_ratio: UnitIntervalLike,
) -> tuple[int, ...]:
    """Highlight active tokens above the per-example activation threshold."""
    max_activation = max(ref.activations, default=0.0)
    if max_activation <= 0.0:
        return ()
    threshold = activation_threshold_ratio * max_activation
    return tuple(idx for idx, value in enumerate(ref.activations) if value >= threshold)


@beartype
def deterministic_intruder_highlight_indices(
    ref: TokenWindowRef,
    *,
    n_highlights: NonNegativeIntLike,
) -> tuple[int, ...]:
    """Return the pinned false-positive highlight policy for one intruder."""
    if n_highlights == 0:
        return ()
    candidates = [idx for idx, activation in enumerate(ref.activations) if activation <= 0.0]
    if not candidates:
        raise AutoInterpDataError(
            f"cannot sample intruder highlights for {ref.stable_key()}: "
            "no non-activating tokens are available"
        )
    n = min(n_highlights, len(candidates))
    token_pos = len(ref.token_strings) - len(ref.token_strings) // 4
    if token_pos in candidates:
        selected = [token_pos]
        remaining = [idx for idx in candidates if idx != token_pos]
        if n > 1:
            selected.extend(random.Random(22).sample(remaining, min(n - 1, len(remaining))))
    else:
        selected = random.Random(22).sample(candidates, n)
    return tuple(sorted(selected))


def _shard_index_from_stem(stem: str) -> int:
    match = _SHARD_INDEX_RE.search(stem)
    if match is None:
        raise AutoInterpDataError(f"feature shard stem {stem!r} does not contain a shard index")
    return int(match.group(1))


def _source_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate_to_focused_window(
    text: str,
    sae_tokenizer,
    context_window: int,
    sae_max_context: int,
    focus_token_end: int,
) -> tuple[list[int], int, int]:
    """Tokenize text and keep a context window that includes the focus token."""
    if focus_token_end is None:
        raise AutoInterpDataError("token-level autointerp requires a focused segment token end")
    if not text:
        raise AutoInterpDataError("cannot build focused token window from empty source text")
    sae_encoding = sae_tokenizer(text, padding=False)
    sae_ids = sae_encoding["input_ids"]
    if sae_ids and isinstance(sae_ids[0], list):
        sae_ids = sae_ids[0]
    n_sae_tokens = len(sae_ids)
    limit = min(context_window, sae_max_context)
    if n_sae_tokens <= limit:
        return (sae_ids, n_sae_tokens, 0)
    end = max(1, min(int(focus_token_end), n_sae_tokens))
    start = max(0, end - limit)
    return (sae_ids[start:end], n_sae_tokens, start)


def _row_aligned_i64_tensor(
    data: dict,
    key: TensorKey,
    *,
    shard_path: Path,
    n_rows: int,
) -> t.Tensor | None:
    value = data.get(key)
    if value is None:
        return None
    tensor = value.to(t.int64)
    if int(tensor.numel()) != n_rows:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has {key} length {int(tensor.numel())}, expected {n_rows}"
        )
    return tensor


def _segment_token_bounds(
    data: dict,
    *,
    shard_path: Path,
    n_rows: int,
) -> tuple[t.Tensor, t.Tensor]:
    starts = _row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_TOKEN_STARTS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    ends = _row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_TOKEN_ENDS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    if (starts is None) != (ends is None):
        raise AutoInterpDataError(
            f"feature shard {shard_path} must contain both segment_token_starts "
            "and segment_token_ends when either is present"
        )
    if starts is not None and ends is not None:
        if bool((starts < 0).any()) or bool((ends <= starts).any()):
            raise AutoInterpDataError(
                f"feature shard {shard_path} has invalid segment token bounds"
            )
        return starts, ends

    segment_size = int(data.get("segment_size", 0))
    if segment_size <= 0:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has invalid segment_size={segment_size}; "
            "token-level autointerp requires positive segment size"
        )

    segment_mode = segment_mode_from_metadata(data, source=shard_path)
    if segment_mode is not SegmentMode.fixed_tokens:
        raise AutoInterpDataError(
            f"feature shard {shard_path} has no explicit segment token provenance (segment_mode={segment_mode.value!r})"
        )

    positions = _row_aligned_i64_tensor(
        data,
        TensorKey.SEGMENT_POSITIONS,
        shard_path=shard_path,
        n_rows=n_rows,
    )
    if positions is not None:
        starts = positions
    else:
        segment_indices = _row_aligned_i64_tensor(
            data,
            TensorKey.SEGMENT_INDICES,
            shard_path=shard_path,
            n_rows=n_rows,
        )
        if segment_indices is None:
            raise AutoInterpDataError(
                f"feature shard {shard_path} is missing segment token provenance; "
                "expected explicit token bounds, segment_positions, or fixed-token "
                "segment_indices"
            )
        starts = segment_indices * segment_size

    if bool((starts < 0).any()):
        raise AutoInterpDataError(f"feature shard {shard_path} has negative segment token starts")
    return starts, starts + segment_size


def _selected_sae_activations(
    *,
    sae,
    acts: t.Tensor,
    feature_ids: tuple[int, ...],
) -> t.Tensor:
    """Encode only requested SAE feature columns from model activations."""
    sae_device = next(sae.parameters()).device
    acts = acts.to(sae_device)
    _validate_selected_sae_inputs(
        sae=sae,
        acts=acts,
        feature_ids=feature_ids,
    )
    target_ids = t.tensor(feature_ids, dtype=t.long, device=sae_device)
    return _project_selected_sae_activations(
        sae=sae,
        acts=acts,
        target_ids=target_ids,
    )


@beartype
def _validate_selected_sae_inputs(
    *,
    sae,
    acts: t.Tensor,
    feature_ids: NonEmptyNonNegativeIntTupleLike,
) -> None:
    """Validate compact SAE projection inputs without a device synchronization."""
    if acts.ndim != 2:
        raise AutoInterpDataError(
            f"token-level autointerp expected 2D SAE input activations, got {acts.shape}"
        )
    sae_width = int(sae.W_enc.shape[1])
    invalid = [feature_id for feature_id in feature_ids if feature_id >= sae_width]
    if invalid:
        raise AutoInterpDataError(
            f"target feature id out of range for SAE width {sae_width}: {feature_ids}"
        )


def _project_selected_sae_activations(
    *,
    sae,
    acts: t.Tensor,
    target_ids: t.Tensor,
) -> t.Tensor:
    """Common compact SAE projection kernel after all structural validation."""
    sae_in = sae.process_sae_in(acts).to(t.float32)
    selected_w_enc = sae.W_enc.index_select(1, target_ids).to(t.float32)
    selected_b_enc = sae.b_enc.index_select(0, target_ids).to(t.float32)
    hidden_pre = sae_in @ selected_w_enc
    hidden_pre = hidden_pre + selected_b_enc
    feature_acts = sae.activation_fn(hidden_pre)
    threshold = getattr(sae, "threshold", None)
    if threshold is not None:
        selected_threshold = threshold.index_select(0, target_ids).to(hidden_pre.dtype)
        feature_acts = feature_acts * (hidden_pre > selected_threshold).to(feature_acts.dtype)
    if feature_acts.ndim != 2:
        raise AutoInterpDataError(f"SAE target activations ndim {feature_acts.ndim} != 2")
    if feature_acts.shape[0] != acts.shape[0]:
        raise AutoInterpDataError(
            f"SAE target activation seq_len {feature_acts.shape[0]} "
            f"!= model activations {acts.shape[0]}"
        )
    if feature_acts.shape[1] != int(target_ids.numel()):
        raise AutoInterpDataError(
            "SAE target activation feature count mismatch: "
            f"{feature_acts.shape[1]} != {int(target_ids.numel())}"
        )
    return feature_acts.detach()


def _target_sae_activations(
    *,
    model,
    sae,
    tokens: t.Tensor,
    layer: int,
    hook_name: str | None,
    feature_ids: tuple[int, ...],
) -> t.Tensor:
    """Compute token activations only for requested SAE feature IDs."""
    if hook_name is None:
        hook_name = f"blocks.{layer}.hook_resid_pre"
    try:
        tl_device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        tl_device = tokens.device
    token_batch = tokens.unsqueeze(0).to(tl_device)
    with t.no_grad():
        _, cache = model.run_with_cache(
            token_batch,
            names_filter=[hook_name],
            stop_at_layer=layer + 1,
            return_type=None,
        )
        acts = cache[hook_name].squeeze(0)
        return _selected_sae_activations(sae=sae, acts=acts, feature_ids=feature_ids)


def _load_transformerlens_token_example_model(
    *,
    model_name: str,
    resolved: ResolvedSAE,
    device: str,
    target_layer: int,
) -> tuple[HookedTransformer, SAE, object]:
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        first_n_layers=int(target_layer) + 1,
    )
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device=device,
    )
    return model, sae, model.tokenizer


def _fingerprint_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        payload = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _shard_row_group_offsets(parquet_path: Path) -> list[int]:
    """Cumulative first-doc offsets per row group.

    ``offsets[i]`` is the doc_id of the first row of row group ``i``;
    ``offsets[-1]`` is the shard's total row count. Read from the parquet footer
    only (no column data), so it is cheap and exact for non-uniform final groups.
    """
    metadata = pq.ParquetFile(parquet_path).metadata
    offsets = [0]
    for row_group_index in range(metadata.num_row_groups):
        offsets.append(offsets[-1] + metadata.row_group(row_group_index).num_rows)
    return offsets


def _load_shard_row_group(parquet_path: Path, row_group_index: int) -> list[str]:
    """Load one row group's text column.

    Byte-identical to the matching slice of the whole-shard reference loader in
    ``tests/steering/autointerp_filter/_token_example_reference.py`` (same
    parquet rows, same ``str(...)`` coercion) at ~1/n_row_groups the I/O.
    """
    table = pq.ParquetFile(parquet_path).read_row_group(row_group_index)
    return [str(x) for x in table.column("text").to_pylist()]


def _row_group_text_chars(texts: list[str]) -> int:
    return sum(len(text) for text in texts)


def _coerce_l0(release: SAERelease, raw: int | str | None) -> int | L0Tier | None:
    if raw is None:
        return None
    if release is SAERelease.gemmascope_2:
        return L0Tier(str(raw))
    return int(raw)


def _window_bounds(token_pos: int, n_tokens: int, ctx_len: int) -> tuple[int, int]:
    if n_tokens <= ctx_len:
        return 0, n_tokens
    preferred_left = token_pos - (ctx_len - ctx_len // 4)
    start = max(0, min(preferred_left, n_tokens - ctx_len))
    return start, start + ctx_len


def _window_identity(ref: TokenWindowRef) -> tuple:
    """Source-window identity, excluding activation values."""
    return (
        ref.shard_stem,
        ref.shard_index,
        ref.doc_id,
        ref.source_text_hash,
        ref.token_start,
        ref.token_end,
        ref.token_ids,
    )


def _target_activation_column_from_compact(
    sae_acts: t.Tensor,
    sae_feature_ids: tuple[int, ...],
    target: ScoringTarget,
) -> t.Tensor:
    feature_to_column = {fid: idx for idx, fid in enumerate(sae_feature_ids)}
    try:
        target_column = feature_to_column[target.feature_ids[0]]
    except KeyError as exc:
        raise AutoInterpDataError(
            f"target feature {exc.args[0]} missing from compact SAE activation "
            f"columns {sae_feature_ids}"
        ) from exc
    return sae_acts[:, target_column].detach().cpu()


def _target_activation_column(
    doc: _DocActivations,
    target: ScoringTarget,
) -> t.Tensor:
    return _target_activation_column_from_compact(
        doc.sae_acts,
        doc.sae_feature_ids,
        target,
    )


def _make_token_window_ref(
    doc_ref: SourceDocRef,
    doc: _DocActivations,
    target: ScoringTarget,
    *,
    start: int,
    end: int,
    activations: t.Tensor,
    active: bool,
) -> TokenWindowRef:
    token_ids = doc.token_ids[start:end]
    token_strings = doc.token_strings[start:end]
    activation_tuple = tuple(float(x) for x in activations.tolist())
    return TokenWindowRef(
        shard_stem=doc_ref.shard_stem,
        shard_index=doc_ref.shard_index,
        doc_id=doc_ref.doc_id,
        source_text_hash=doc.source_text_hash,
        token_start=doc.token_offset + start,
        token_end=doc.token_offset + end,
        token_ids=token_ids,
        token_strings=token_strings,
        target_feature_ids=target.feature_ids,
        activations=activation_tuple,
        max_activation=max(activation_tuple, default=0.0),
        quantile=None,
        active=active,
    )


def _positive_windows_from_doc(
    doc_ref: SourceDocRef,
    doc: _DocActivations,
    target: ScoringTarget,
    *,
    example_ctx_len: int,
    max_windows_per_source_doc: int,
) -> list[TokenWindowRef]:
    """Pure window selection shared by serial and fixed-block executors."""
    target_acts = _target_activation_column(doc, target)
    active_positions = (target_acts > 0).nonzero(as_tuple=True)[0].tolist()
    active_positions.sort(key=lambda idx: (-float(target_acts[idx].item()), idx))
    refs: list[TokenWindowRef] = []
    seen_windows: set[tuple[int, int]] = set()
    for pos in active_positions:
        start, end = _window_bounds(int(pos), len(doc.token_ids), example_ctx_len)
        if (start, end) in seen_windows:
            continue
        seen_windows.add((start, end))
        window_acts = target_acts[start:end].to(t.float32)
        refs.append(
            _make_token_window_ref(
                doc_ref,
                doc,
                target,
                start=start,
                end=end,
                activations=window_acts,
                active=True,
            )
        )
        if len(refs) >= max_windows_per_source_doc:
            break
    return refs


def _non_activating_windows_from_doc(
    doc_ref: SourceDocRef,
    doc: _DocActivations,
    target: ScoringTarget,
    *,
    example_ctx_len: int,
    max_windows_per_source_doc: int,
) -> list[TokenWindowRef]:
    """Pure zero-window selection shared by serial and batch executors."""
    target_acts = _target_activation_column(doc, target)
    refs: list[TokenWindowRef] = []
    for start in range(0, len(doc.token_ids), example_ctx_len):
        end = min(start + example_ctx_len, len(doc.token_ids))
        window_acts = target_acts[start:end].to(t.float32)
        if float(window_acts.max().item()) != 0.0:
            continue
        refs.append(
            _make_token_window_ref(
                doc_ref,
                doc,
                target,
                start=start,
                end=end,
                activations=window_acts,
                active=False,
            )
        )
        if len(refs) >= max_windows_per_source_doc:
            break
    return refs


@dataclass
class _TargetBuildState:
    """One exact serial builder state advanced one source document at a time."""

    target: ScoringTarget
    positive_docs: tuple[SourceDocRef, ...]
    intruder_docs: tuple[SourceDocRef, ...]
    example_ctx_len: int
    max_windows_per_source_doc: int
    max_positive_windows: int
    max_non_activating_windows: int
    positive_cursor: int = 0
    intruder_cursor: int = 0
    phase: MaterializationPhase = MaterializationPhase.positive
    positives: list[TokenWindowRef] = field(default_factory=list)
    non_activating: list[TokenWindowRef] = field(default_factory=list)
    positive_keys: set[str] = field(default_factory=set)
    positive_windows: set[tuple] = field(default_factory=set)
    expected_doc: tuple[MaterializationPhase, SourceDocRef] | None = None
    done: bool = False

    def next_doc(self) -> tuple[MaterializationPhase, SourceDocRef] | None:
        if self.expected_doc is not None:
            raise AutoInterpDataError(
                f"target state {self.target.target_key} requested new work before "
                "accepting its previous document"
            )
        while not self.done:
            if self.phase is MaterializationPhase.positive:
                if (
                    self.positive_cursor >= len(self.positive_docs)
                    or len(self.positives) >= self.max_positive_windows
                ):
                    self._finalize_positives()
                    self.phase = MaterializationPhase.intruder
                    continue
                doc_ref = self.positive_docs[self.positive_cursor]
                self.positive_cursor += 1
                self.expected_doc = (self.phase, doc_ref)
                return self.expected_doc

            if (
                self.intruder_cursor >= len(self.intruder_docs)
                or len(self.non_activating) >= self.max_non_activating_windows
            ):
                self.non_activating = _dedupe_refs(self.non_activating)
                self.done = True
                return None
            doc_ref = self.intruder_docs[self.intruder_cursor]
            self.intruder_cursor += 1
            self.expected_doc = (self.phase, doc_ref)
            return self.expected_doc
        return None

    def accept_doc(
        self,
        *,
        phase: MaterializationPhase,
        doc_ref: SourceDocRef,
        doc: _DocActivations,
    ) -> None:
        if self.expected_doc != (phase, doc_ref):
            raise AutoInterpDataError(
                f"target state {self.target.target_key} received an unexpected "
                f"document: expected={self.expected_doc!r} actual={(phase, doc_ref)!r}"
            )
        if doc.sae_feature_ids != self.target.feature_ids:
            raise AutoInterpDataError(
                f"target state {self.target.target_key} received compact SAE columns "
                f"{doc.sae_feature_ids}, expected {self.target.feature_ids}"
            )
        if phase is MaterializationPhase.positive:
            self.positives.extend(
                _positive_windows_from_doc(
                    doc_ref,
                    doc,
                    self.target,
                    example_ctx_len=self.example_ctx_len,
                    max_windows_per_source_doc=self.max_windows_per_source_doc,
                )
            )
        elif phase is MaterializationPhase.intruder:
            for ref in _non_activating_windows_from_doc(
                doc_ref,
                doc,
                self.target,
                example_ctx_len=self.example_ctx_len,
                max_windows_per_source_doc=self.max_windows_per_source_doc,
            ):
                if (
                    ref.stable_key() in self.positive_keys
                    or _window_identity(ref) in self.positive_windows
                ):
                    continue
                self.non_activating.append(ref)
                if len(self.non_activating) >= self.max_non_activating_windows:
                    break
        else:  # pragma: no cover - StrEnum exhaustiveness guard
            raise AutoInterpDataError(f"unknown materialization phase {phase!r}")
        self.expected_doc = None

    def result(self) -> TokenExampleSet:
        if not self.done or self.expected_doc is not None:
            raise AutoInterpDataError(
                f"target state {self.target.target_key} was finalized before completion"
            )
        return TokenExampleSet(
            positives=tuple(self.positives),
            non_activating=tuple(self.non_activating),
        )

    def _finalize_positives(self) -> None:
        self.positives = sorted(
            _dedupe_refs(self.positives),
            key=lambda ref: (-ref.max_activation, ref.stable_key()),
        )[: self.max_positive_windows]
        self.positive_keys = {ref.stable_key() for ref in self.positives}
        self.positive_windows = {_window_identity(ref) for ref in self.positives}


@dataclass
class _BlockBuild:
    """Mutable main-thread state for one at-most-two-lane block build."""

    block: TargetBlock
    states_by_slot: tuple[_TargetBuildState | None, ...]
    future: Future[PreparedBatch] | None = None
    completed: dict[str, TokenExampleSet] | None = None
    last_surfaces_by_slot: dict[int, tuple[PreparedDoc, _DocActivations]] = field(
        default_factory=dict
    )


class CachedTokenExampleProvider(TokenExampleProvider):
    """Build token examples by re-forwarding source documents, with cache reuse."""

    @beartype
    def __init__(
        self,
        cfg: AutoInterpFilterConfig,
        *,
        feature_dir: Path,
        n_sae_features: PositiveIntLike,
        materialization_prefetch: bool = True,
        batch_tolerance_profile: BatchToleranceProfile | None = None,
        batch_validation_anchor_target_keys: _NonEmptyStrFrozenSetLike | None = None,
        batch_pair_timing_sink: _BatchPairTimingSinkLike | None = None,
        allow_unvalidated_batch_profile: bool = False,
    ) -> None:
        self._cfg = cfg
        self._feature_dir = feature_dir
        self._n_sae_features = PositiveInt.parse(n_sae_features)
        self._targets: list[ScoringTarget] = []
        self._targets_by_feature: dict[int, list[ScoringTarget]] = defaultdict(list)
        self._positive_docs: dict[str, list[SourceDocRef]] = defaultdict(list)
        self._positive_doc_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self._intruder_docs: list[SourceDocRef] = []
        self._intruder_doc_keys: set[str] = set()
        self._intruder_doc_seen = 0
        self._prepared = False
        self._feature_fingerprint = ""
        self._data_fingerprint = ""
        self._parquets: list[Path] = []
        # Source text is cached at ROW-GROUP granularity (keyed by
        # ``(shard_index, row_group_index)``): materialization touches a few docs
        # per shard, so a full-shard cache re-read whole ~240 MiB columns. The
        # cache is bounded by ``_ROW_GROUP_CACHE_CHAR_BUDGET``; per-shard row-group
        # offsets are tiny and kept unbounded. These structures are mutated
        # without a lock: serial materialization reaches ``_full_text`` only on
        # the main thread; fixed-block preparation uses exactly one lane (the main
        # thread with prefetch off, or one ``max_workers=1`` worker with prefetch
        # on), and its GPU/state lane never calls ``_full_text`` concurrently.
        # Both finish before the judge ThreadPoolExecutor, which never calls the
        # provider. Each cache entry is ``(texts, char_count)`` so the running
        # total cannot drift.
        self._row_group_offsets: dict[int, list[int]] = {}
        self._row_group_cache: OrderedDict[tuple[int, int], tuple[list[str], int]] = OrderedDict()
        self._row_group_cache_total_chars = 0
        self._doc_cache: OrderedDict[str, _DocActivations] = OrderedDict()
        self._model = None
        self._native_model = None
        self._sae = None
        self._tokenizer = None
        self._layer = 0
        self._activation_type = None
        self._hook_name: str | None = None
        self._context_window = 0
        self._sae_max_context = 0
        self._activation_backend = "unloaded"
        self._activation_feature_ids: tuple[int, ...] = ()
        self._context_independent_features = False
        self._ignore_bos_features = False
        self._ignore_bos_token_id: int | None = None
        self._source_index_request: SourceIndexRequest | None = None
        self._source_index_loaded = False
        self._target_plan_hash = ""
        self._batch_segment_size = 0
        self._batch_segment_mode = ""
        self._first_shard_path: Path | None = None
        self._materialization_prefetch = materialization_prefetch
        self._batch_tolerance_profile = (
            PRODUCTION_BATCH_128_PROFILE
            if batch_tolerance_profile is None
            else batch_tolerance_profile
        )
        self._batch_verified = False
        self._batch_shadow_counter = 0
        self._batch_validation_observations: list[dict[str, object]] = []
        self._batch_validation_anchor_target_keys = batch_validation_anchor_target_keys
        self._batch_pair_timing_sink = batch_pair_timing_sink
        self._allow_unvalidated_batch_profile = allow_unvalidated_batch_profile
        self._batch_runtime_fingerprint_cache: dict[str, object] | None = None
        self._batch_companion_token_strings: dict[int, str] = {}
        self._batch_gpu_thread_local = threading.local()
        self._batch_gpu_streams_ready = False

    def prepare(
        self,
        targets: list[ScoringTarget],
        *,
        scoring_targets: list[ScoringTarget] | None = None,
    ) -> None:
        self._prepared = False
        full_targets = list(targets)
        selected_targets = full_targets if scoring_targets is None else list(scoring_targets)
        full_targets_by_key = {target.target_key: target for target in full_targets}
        if len(full_targets_by_key) != len(full_targets):
            raise AutoInterpDataError("autointerp full target universe contains duplicates")
        selected_keys: set[str] = set()
        for target in selected_targets:
            full_target = full_targets_by_key.get(target.target_key)
            if full_target is None or full_target != target:
                raise AutoInterpDataError(
                    f"scoring target {target.target_key!r} is not an exact member "
                    "of the full target universe"
                )
            if target.target_key in selected_keys:
                raise AutoInterpDataError(
                    f"scoring target subset contains duplicate {target.target_key!r}"
                )
            selected_keys.add(target.target_key)

        self._targets = selected_targets
        self._target_plan_hash = target_plan_hash(selected_targets)
        self._batch_verified = False
        self._batch_shadow_counter = 0
        self._batch_validation_observations.clear()
        self._batch_runtime_fingerprint_cache = None
        self._batch_companion_token_strings.clear()
        self._batch_gpu_thread_local = threading.local()
        self._batch_gpu_streams_ready = False
        if self._batch_pair_timing_sink is not None:
            self._batch_pair_timing_sink.clear()
        if self._batch_validation_anchor_target_keys is not None:
            missing_validation_anchors = sorted(
                self._batch_validation_anchor_target_keys
                - {target.target_key for target in selected_targets}
            )
            if missing_validation_anchors:
                raise AutoInterpDataError(
                    "batch validation anchors are absent from the prepared target plan: "
                    f"{missing_validation_anchors}"
                )
        self._targets_by_feature.clear()
        self._positive_docs.clear()
        self._positive_doc_scores.clear()
        self._intruder_docs.clear()
        self._intruder_doc_keys.clear()
        self._intruder_doc_seen = 0
        self._source_index_request = None
        self._source_index_loaded = False
        self._activation_feature_ids = tuple(
            sorted({fid for target in selected_targets for fid in target.feature_ids})
        )
        full_feature_ids = tuple(
            sorted({fid for target in full_targets for fid in target.feature_ids})
        )
        for target in selected_targets:
            for fid in target.feature_ids:
                self._targets_by_feature[fid].append(target)

        shard_paths = glob_features(self._feature_dir)
        if not shard_paths:
            raise AutoInterpDataError(f"No feature shards found in {self._feature_dir}")
        (
            self._context_independent_features,
            self._ignore_bos_features,
        ) = self._feature_flags_from_path_or_shard(shard_paths[0])
        self._first_shard_path = Path(shard_paths[0])
        try:
            parsed_feature_dir = parse_feature_dir(self._feature_dir)
        except ValueError:
            self._batch_segment_size = 0
            self._batch_segment_mode = ""
        else:
            self._batch_segment_size = int(parsed_feature_dir.segment_size)
            self._batch_segment_mode = "fixed_tokens"
        if (
            self._cfg.token_examples.materialization_batch_size > 1
            and self._cfg.token_examples.activation_backend == "native"
            and self._cfg.token_examples.device.startswith("cuda")
        ):
            self._load_batch_segment_metadata_if_needed(force=True)
        if not self._ignore_bos_features:
            raise AutoInterpDataError(
                "autointerp token examples require feature shards extracted with "
                "ignore_bos_features=true"
            )
        data_dir = (
            Path(self._cfg.data_dir)
            if self._cfg.data_dir is not None
            else resolve_data_dir(
                None,
                parse_feature_dir(self._feature_dir).dataset_phase.value,
            )
        )
        self._parquets = [Path(path) for path in list_parquet_files(str(data_dir))]
        if not self._parquets:
            raise AutoInterpDataError(f"No parquet shards found in {data_dir}")
        self._feature_fingerprint = _fingerprint_paths([Path(p) for p in shard_paths])
        self._data_fingerprint = _fingerprint_paths(self._parquets)

        if self._cfg.source_index.enabled:
            self._source_index_request = SourceIndexRequest(
                feature_dir=self._feature_dir,
                data_dir=self._cfg.data_dir,
                index_path=Path(self._cfg.source_index.path),
                manifest_path=Path(self._cfg.source_index.manifest_path),
                shared_cache_dir=(
                    Path(self._cfg.source_index.shared_cache_dir)
                    if self._cfg.source_index.shared_cache_dir is not None
                    else None
                ),
                shard_paths=tuple(Path(path) for path in shard_paths),
                parquet_paths=tuple(self._parquets),
                target_feature_ids=full_feature_ids,
                n_sae_features=self._n_sae_features,
                max_positive_refs_per_feature=(
                    self._cfg.token_examples.max_positive_source_docs_per_target
                ),
                max_intruder_refs=self._cfg.token_examples.max_intruder_source_docs,
                scan_workers=self._cfg.source_index.scan_workers,
                seed=self._cfg.seed,
                feature_fingerprint=self._feature_fingerprint,
                data_fingerprint=self._data_fingerprint,
                context_independent_features=self._context_independent_features,
                ignore_bos_features=self._ignore_bos_features,
                chunk_size=self._cfg.source_index.chunk_size,
                force_rebuild=self._cfg.source_index.force_rebuild,
                load_target_feature_ids=(
                    None if scoring_targets is None else self._activation_feature_ids
                ),
            )
            self._source_index_request = validate_source_index_cache(self._source_index_request)
            if self._cfg.source_index.force_rebuild:
                self._ensure_source_index_loaded()
            self._prepared = True
            return

        for shard_path in shard_paths:
            data = load_features(shard_path)
            self._scan_shard(Path(shard_path), data)
        if not self._intruder_docs:
            raise AutoInterpDataError(f"No source documents were discovered in {self._feature_dir}")
        self._prepared = True

    def cache_context(self) -> dict:
        if not self._prepared:
            raise AutoInterpDataError("token example provider was not prepared")
        context = {
            "method": _TOKEN_CACHE_METHOD,
            "feature_dir": str(self._feature_dir),
            "data_dir": self._cfg.data_dir,
            "feature_fingerprint": self._feature_fingerprint,
            "data_fingerprint": self._data_fingerprint,
            "n_sae_features": self._n_sae_features,
            "token_examples": {
                "device": self._cfg.token_examples.device,
                "model": self._cfg.token_examples.model,
                "sae_release": self._cfg.token_examples.sae_release,
                "sae_l0": self._cfg.token_examples.sae_l0,
                "activation_backend": self._cfg.token_examples.activation_backend,
                "model_dtype": self._cfg.token_examples.model_dtype,
                "context_window": self._cfg.token_examples.context_window,
                "example_ctx_len": self._cfg.token_examples.example_ctx_len,
                "activation_threshold_ratio": (self._cfg.token_examples.activation_threshold_ratio),
            },
            "source_index": {
                "enabled": self._cfg.source_index.enabled,
                "path": self._cfg.source_index.path if self._cfg.source_index.enabled else None,
                "manifest_path": (
                    self._cfg.source_index.manifest_path if self._cfg.source_index.enabled else None
                ),
            },
            "context_independent_features": self._context_independent_features,
            "ignore_bos_features": self._ignore_bos_features,
        }
        batch_size = self._cfg.token_examples.materialization_batch_size
        if batch_size == 1:
            return context
        context["method"] = _BATCH_TOKEN_CACHE_METHOD
        context["materialization"] = {
            "batch_size": batch_size,
            "block_mapping": "feature_id_divmod_batch_size_v1",
            "partial_block_policy": "rebuild_entire_block_v1",
            "gpu_singleton_concurrency": _BATCH_GPU_SINGLETON_CONCURRENCY,
            "model_input_policy": "true_length_singleton_v1",
            "companion_leakage_probe": "hold_probe_toggle_other_singleton_v1",
            "target_plan_sha256": self._target_plan_hash,
            "validation_profile": self._batch_tolerance_profile.identity(),
            "runtime_numeric_fingerprint": self._runtime_fingerprint_for_cache(),
            "max_positive_source_docs_per_target": (
                self._cfg.token_examples.max_positive_source_docs_per_target
            ),
            "max_intruder_source_docs": self._cfg.token_examples.max_intruder_source_docs,
            "max_windows_per_source_doc": (self._cfg.token_examples.max_windows_per_source_doc),
            "max_positive_windows_per_target": (
                self._cfg.token_examples.max_positive_windows_per_target
            ),
            "max_non_activating_windows_per_target": (
                self._cfg.token_examples.max_non_activating_windows_per_target
            ),
        }
        return context

    def _runtime_fingerprint_for_cache(self) -> dict[str, object]:
        cached = self._batch_runtime_fingerprint_cache
        if cached is None:
            cached = runtime_numeric_fingerprint(
                self._cfg.token_examples.device
                if self._cfg.token_examples.device.startswith("cuda")
                else None
            )
            self._batch_runtime_fingerprint_cache = cached
        return dict(cached)

    def _apply_source_index(self, indexed: SourceIndexData) -> None:
        for target in self._targets:
            feature_id = target.feature_ids[0]
            for doc_ref, magnitude in indexed.positives_by_feature.get(feature_id, []):
                self._add_positive_doc(target, doc_ref, magnitude)
        self._intruder_docs = list(indexed.intruder_refs)
        self._intruder_doc_keys = {ref.stable_key() for ref in self._intruder_docs}
        self._intruder_doc_seen = len(self._intruder_doc_keys)

    def _feature_flags_from_path_or_shard(self, first_shard_path: Path) -> tuple[bool, bool]:
        try:
            parsed = parse_feature_dir(self._feature_dir)
            return parsed.context_independent_features, parsed.ignore_bos_features
        except ValueError:
            metadata = read_shard_metadata(first_shard_path)
            return (
                metadata.get(MetadataKey.CONTEXT_INDEPENDENT_FEATURES, "false") == "true",
                metadata.get(MetadataKey.IGNORE_BOS_FEATURES, "false") == "true",
            )

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        if not self._prepared:
            raise AutoInterpDataError("token example provider was not prepared")
        namespace = self._cache_namespace(target, serial_reference=True)
        cache_path = Path(self._cfg.token_example_cache_path)
        with TokenExampleCache(cache_path) as cache:
            cached = cache.get(namespace=namespace, target_key=target.target_key)
            if cached is not None:
                self._validate_cached_examples(target, cached)
                return cached
            self._ensure_source_index_loaded()
            examples = self._build_examples_for_target(target)
            cache.set(
                namespace=namespace,
                target_key=target.target_key,
                examples=examples,
            )
            return examples

    def examples_for_targets(
        self,
        targets: list[ScoringTarget],
    ) -> dict[str, TokenExampleSet]:
        """Materialize the exact prepared target plan through fixed ID blocks.

        Batch size one deliberately delegates to the unchanged public serial
        method. For a real batch, complete blocks are lazy cache hits; a block
        with even one missing row is rebuilt in full and committed atomically.
        """
        bulk_started_ns = (
            time.perf_counter_ns() if self._batch_pair_timing_sink is not None else None
        )
        timing_start_index = (
            len(self._batch_pair_timing_sink) if self._batch_pair_timing_sink is not None else 0
        )
        if not self._prepared:
            raise AutoInterpDataError("token example provider was not prepared")
        if targets != self._targets:
            raise AutoInterpDataError(
                "bulk token-example request must exactly match the ordered target "
                "plan passed to prepare()"
            )
        batch_size = self._cfg.token_examples.materialization_batch_size
        if batch_size == 1:
            return {target.target_key: self.examples_for_target(target) for target in targets}
        if not targets:
            return {}
        self._load_batch_segment_metadata_if_needed()
        self._validate_batch_support(batch_size=batch_size)
        blocks = build_target_blocks(targets, batch_size=batch_size)
        namespaces: dict[str, str] = {}
        lookup_elapsed_ns_by_block: dict[int, int] = {}
        results: dict[str, TokenExampleSet] = {}
        pending_blocks: list[TargetBlock] = []
        cache_path = Path(self._cfg.token_example_cache_path)
        with TokenExampleCache(cache_path) as cache:
            for block in blocks:
                lookup_started_ns = (
                    time.perf_counter_ns() if self._batch_pair_timing_sink is not None else None
                )
                for target in block.targets:
                    namespaces[target.target_key] = self._cache_namespace(
                        target,
                        serial_reference=False,
                    )
                keys = [
                    (namespaces[target.target_key], target.target_key) for target in block.targets
                ]
                cached = cache.get_many(keys)
                if lookup_started_ns is not None:
                    lookup_elapsed_ns_by_block[block.block_id] = (
                        time.perf_counter_ns() - lookup_started_ns
                    )
                if len(cached) == len(keys):
                    for target in block.targets:
                        examples = cached[(namespaces[target.target_key], target.target_key)]
                        self._validate_cached_examples(target, examples)
                        results[target.target_key] = examples
                    continue
                pending_blocks.append(block)

            if pending_blocks:
                self._ensure_source_index_loaded()
                self._ensure_model_loaded()
                self._validate_loaded_batch_model(batch_size=batch_size)
                self._prepare_batch_companion_token_strings()
                built = self._build_and_commit_blocks(
                    pending_blocks,
                    cache=cache,
                    namespaces=namespaces,
                )
                overlap = set(results).intersection(built)
                if overlap:
                    raise AutoInterpDataError(
                        f"batched token-example results duplicated cache-hit targets: {overlap}"
                    )
                results.update(built)

        expected = {target.target_key for target in targets}
        if set(results) != expected:
            raise AutoInterpDataError(
                "batched token-example materialization returned the wrong target set: "
                f"missing={sorted(expected - set(results))} "
                f"extra={sorted(set(results) - expected)}"
            )
        if self._batch_pair_timing_sink is not None:
            assert bulk_started_ns is not None
            self._reconcile_batch_pair_timings(
                timing_start_index=timing_start_index,
                bulk_elapsed_ns=time.perf_counter_ns() - bulk_started_ns,
                lookup_elapsed_ns_by_block=lookup_elapsed_ns_by_block,
            )
        return results

    @beartype
    def _reconcile_batch_pair_timings(
        self,
        *,
        timing_start_index: NonNegativeIntLike,
        bulk_elapsed_ns: PositiveIntLike,
        lookup_elapsed_ns_by_block: NonNegativeIntMappingLike,
    ) -> None:
        """Make pair timings partition the complete bulk call exactly.

        Raw pair timers own build/validation/commit/log work. Per-block
        namespace/cache lookup is added to its pair; all one-time setup and
        teardown is assigned to the first pair, which the steady-state policy
        excludes symmetrically with serial's first two feature-ID blocks.
        """
        assert self._batch_pair_timing_sink is not None
        if timing_start_index > len(self._batch_pair_timing_sink):
            raise AutoInterpDataError("batch timing reconciliation start index is invalid")
        pair_timings = self._batch_pair_timing_sink[timing_start_index:]
        if not pair_timings:
            return
        raw_elapsed_ns = 0
        distributed_lookup_ns = 0
        for index, timing in enumerate(pair_timings):
            block_ids = timing.get("block_ids")
            elapsed_ns = timing.get("elapsed_ns")
            if (
                not isinstance(block_ids, tuple)
                or not block_ids
                or any(
                    isinstance(block_id, bool) or not isinstance(block_id, int)
                    for block_id in block_ids
                )
                or any(block_id not in lookup_elapsed_ns_by_block for block_id in block_ids)
                or isinstance(elapsed_ns, bool)
                or not isinstance(elapsed_ns, int)
                or elapsed_ns <= 0
            ):
                raise AutoInterpDataError(
                    f"internal batch pair timing {index} is malformed before reconciliation"
                )
            normalized_block_ids = cast(tuple[int, ...], block_ids)
            pair_lookup_ns = sum(
                lookup_elapsed_ns_by_block[block_id] for block_id in normalized_block_ids
            )
            timing["elapsed_ns"] = elapsed_ns + pair_lookup_ns
            timing["scope"] = "complete_bulk_partition_v1"
            raw_elapsed_ns += elapsed_ns
            distributed_lookup_ns += pair_lookup_ns
        one_time_ns = bulk_elapsed_ns - raw_elapsed_ns - distributed_lookup_ns
        if one_time_ns < 0:
            raise AutoInterpDataError(
                "internal batch pair timings exceed their enclosing bulk elapsed time"
            )
        first_elapsed_ns = pair_timings[0]["elapsed_ns"]
        assert isinstance(first_elapsed_ns, int)
        pair_timings[0]["elapsed_ns"] = first_elapsed_ns + one_time_ns
        reconciled_elapsed_ns = 0
        for timing in pair_timings:
            elapsed_ns = timing.get("elapsed_ns")
            assert isinstance(elapsed_ns, int)
            reconciled_elapsed_ns += elapsed_ns
        if reconciled_elapsed_ns != bulk_elapsed_ns:
            raise AutoInterpDataError(
                "reconciled batch pair timings do not partition the bulk elapsed time"
            )

    def _validate_batch_support(self, *, batch_size: int) -> None:
        """Reject every configuration not covered by the fixed-block contract."""
        cfg = self._cfg.token_examples
        reasons: list[str] = []
        if cfg.activation_backend != "native":
            reasons.append(f"activation_backend={cfg.activation_backend!r} (requires 'native')")
        if not cfg.device.startswith("cuda"):
            reasons.append(f"device={cfg.device!r} (requires CUDA)")
        if not self._context_independent_features:
            reasons.append("feature shards are not context-independent")
        if not self._ignore_bos_features:
            reasons.append("feature shards do not ignore BOS features")
        if self._batch_segment_mode != "fixed_tokens":
            reasons.append(f"segment_mode={self._batch_segment_mode!r} (requires 'fixed_tokens')")
        if self._batch_segment_size < 1:
            reasons.append(f"segment_size={self._batch_segment_size} (requires positive)")
        if self._batch_tolerance_profile.batch_size != batch_size:
            reasons.append(
                "validation profile batch_size="
                f"{self._batch_tolerance_profile.batch_size} does not match {batch_size}"
            )
        if not self._allow_unvalidated_batch_profile:
            if self._batch_tolerance_profile.runtime_fingerprint_sha256 is None:
                reasons.append("validation profile has no calibrated runtime fingerprint")
            if self._batch_tolerance_profile.calibration_observations_sha256 is None:
                reasons.append("validation profile has no calibration-observation digest")
        if reasons:
            raise AutoInterpDataError(
                "token_examples.materialization_batch_size>1 is unsupported for this "
                f"configuration ({'; '.join(reasons)}). Set "
                "token_examples.materialization_batch_size=1 for the serial reference path."
            )

        parsed = parse_feature_dir(self._feature_dir)
        model_name = cfg.model or parsed.model
        release = SAERelease(cfg.sae_release)
        resolved = resolve_sae(
            model=model_name,
            release=release,
            activation_type=parsed.activation_type,
            layer=parsed.layer,
            width=parsed.width,
            l0=_coerce_l0(release, cfg.sae_l0),
        )
        self._should_use_native_backend(
            backend=cfg.activation_backend,
            model_name=model_name,
            hook_name=resolved.hook_name,
            device=cfg.device,
        )
        actual_contract = BatchCalibrationContract(
            model_name=model_name,
            model_dtype=cfg.model_dtype,
            activation_type=parsed.activation_type.value,
            layer=parsed.layer,
            segment_size=self._batch_segment_size,
            sae_release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        )
        expected_contract = self._batch_tolerance_profile.calibration_contract
        if actual_contract != expected_contract:
            raise AutoInterpDataError(
                "batched token-example model/SAE/input surface is not covered by "
                "the frozen tolerance profile: "
                f"profile={self._batch_tolerance_profile.name} "
                f"expected={expected_contract.identity()} "
                f"actual={actual_contract.identity()}. Set "
                "token_examples.materialization_batch_size=1 for the serial "
                "reference path."
            )
        assert_runtime_matches_profile(
            self._batch_tolerance_profile,
            device=cfg.device,
        )

    def _load_batch_segment_metadata_if_needed(self, *, force: bool = False) -> None:
        if not force and self._batch_segment_size > 0 and self._batch_segment_mode:
            return
        if self._first_shard_path is None:
            raise AutoInterpDataError("batch support validation has no feature shard")
        metadata = read_shard_metadata(self._first_shard_path)
        raw_size = metadata.get(MetadataKey.SEGMENT_SIZE)
        raw_mode = metadata.get(MetadataKey.SEGMENT_MODE, "fixed_tokens")
        path_segment_size = self._batch_segment_size
        try:
            shard_segment_size = 0 if raw_size is None else int(raw_size)
        except ValueError as exc:
            raise AutoInterpDataError(
                f"feature shard {self._first_shard_path} has invalid segment_size={raw_size!r}"
            ) from exc
        if path_segment_size > 0 and shard_segment_size != path_segment_size:
            raise AutoInterpDataError(
                "feature shard segment_size disagrees with feature directory: "
                f"path={path_segment_size} shard={shard_segment_size} "
                f"({self._first_shard_path})"
            )
        self._batch_segment_size = shard_segment_size
        self._batch_segment_mode = str(raw_mode)

    def _prepare_batch_companion_token_strings(self) -> None:
        """Freeze synthetic probe decoding before the prefetch worker starts.

        The leakage probe can run on the main thread while the one CPU lane is
        tokenizing the next block. Decoding its two possible replacement token
        IDs here keeps *all* tokenizer access on one thread at a time instead
        of assuming a particular tokenizer implementation is thread-safe.
        """
        if self._tokenizer is None or self._ignore_bos_token_id is None:
            raise AutoInterpDataError(
                "batch companion leakage probe requires a loaded tokenizer and BOS id"
            )
        bos_token_id = int(self._ignore_bos_token_id)
        if bos_token_id == 0:
            raise AutoInterpDataError(
                "batch companion leakage probe BOS id collides with pad/dummy token id 0"
            )
        token_ids = (0, bos_token_id)
        self._batch_companion_token_strings = {
            token_id: str(self._tokenizer.decode([token_id])) for token_id in token_ids
        }

    def _validate_loaded_batch_model(self, *, batch_size: int) -> None:
        if self._native_model is None or self._sae is None or self._tokenizer is None:
            raise AutoInterpDataError(
                "fixed-block token examples require a loaded native model, SAE, and tokenizer"
            )
        if self._activation_backend != "native":
            raise AutoInterpDataError(
                f"fixed-block token examples loaded backend={self._activation_backend!r}"
            )
        if batch_size != self._batch_tolerance_profile.batch_size:
            raise AutoInterpDataError(
                "loaded batch size does not match the numeric validation profile"
            )
        expected_dtype = {
            "float32": t.float32,
            "bfloat16": t.bfloat16,
        }[self._cfg.token_examples.model_dtype]
        model_parameters = tuple(self._native_model.parameters())
        loaded_reasons: list[str] = []
        configured_device = t.device(self._cfg.token_examples.device)

        def wrong_device(value: t.Tensor) -> bool:
            return value.device.type != configured_device.type or (
                configured_device.index is not None
                and value.device.index != configured_device.index
            )

        if not model_parameters:
            loaded_reasons.append("native model has no parameters")
        else:
            wrong_model_dtypes = [
                index
                for index, parameter in enumerate(model_parameters)
                if parameter.is_floating_point() and parameter.dtype != expected_dtype
            ]
            wrong_model_devices = [
                index for index, parameter in enumerate(model_parameters) if wrong_device(parameter)
            ]
            if wrong_model_dtypes:
                loaded_reasons.append(
                    "model dtype mismatch at parameter indices="
                    f"{wrong_model_dtypes} expected={expected_dtype}"
                )
            if wrong_model_devices:
                loaded_reasons.append(
                    "model device mismatch at parameter indices="
                    f"{wrong_model_devices} expected={configured_device}"
                )
        sae_parameters = tuple(self._sae.parameters())
        wrong_sae_dtypes = [
            index
            for index, parameter in enumerate(sae_parameters)
            if parameter.is_floating_point() and parameter.dtype != expected_dtype
        ]
        wrong_sae_devices = [
            index for index, parameter in enumerate(sae_parameters) if wrong_device(parameter)
        ]
        if wrong_sae_dtypes:
            loaded_reasons.append(
                f"SAE dtype mismatch at parameter indices={wrong_sae_dtypes} "
                f"expected={expected_dtype}"
            )
        if wrong_sae_devices:
            loaded_reasons.append(
                f"SAE device mismatch at parameter indices={wrong_sae_devices} "
                f"expected={configured_device}"
            )
        w_enc = getattr(self._sae, "W_enc", None)
        b_enc = getattr(self._sae, "b_enc", None)
        b_dec = getattr(self._sae, "b_dec", None)
        threshold = getattr(self._sae, "threshold", None)
        if not isinstance(w_enc, t.Tensor) or w_enc.ndim != 2:
            loaded_reasons.append("SAE W_enc is not a rank-2 tensor")
            sae_input_width = sae_width = -1
        else:
            sae_input_width, sae_width = (int(w_enc.shape[0]), int(w_enc.shape[1]))
        if not isinstance(b_enc, t.Tensor) or tuple(b_enc.shape) != (sae_width,):
            loaded_reasons.append(
                f"SAE b_enc shape={getattr(b_enc, 'shape', None)} expected=({sae_width},)"
            )
        if not isinstance(b_dec, t.Tensor) or tuple(b_dec.shape) != (sae_input_width,):
            loaded_reasons.append(
                f"SAE b_dec shape={getattr(b_dec, 'shape', None)} expected=({sae_input_width},)"
            )
        if threshold is not None and (
            not isinstance(threshold, t.Tensor) or tuple(threshold.shape) != (sae_width,)
        ):
            loaded_reasons.append(
                f"SAE threshold shape={getattr(threshold, 'shape', None)} expected=({sae_width},)"
            )
        if getattr(self._native_model, "model_name", None) != (
            self._batch_tolerance_profile.calibration_contract.model_name
        ):
            loaded_reasons.append(
                f"native model_name={getattr(self._native_model, 'model_name', None)!r}"
            )
        if getattr(self._native_model, "truncated_layer", None) != self._layer:
            loaded_reasons.append(
                "native truncated layer="
                f"{getattr(self._native_model, 'truncated_layer', None)!r} "
                f"expected={self._layer}"
            )
        if getattr(self._native_model, "residual_only", None) is not True:
            loaded_reasons.append("native collector is not residual-only")
        if getattr(self._native_model, "_use_compiled", None) is not False:
            loaded_reasons.append("native compiled forward must remain inactive")
        base = getattr(self._native_model, "_base", None)
        if getattr(base, "training", None) is not False:
            loaded_reasons.append("native model is not in eval mode")
        layers = getattr(base, "layers", None)
        expected_layers = self._layer + 1
        if not isinstance(layers, t.nn.ModuleList) or len(layers) != expected_layers:
            loaded_reasons.append(
                "native truncated layers="
                f"{len(layers) if isinstance(layers, t.nn.ModuleList) else None} "
                f"expected={expected_layers}"
            )
        if not isinstance(getattr(base, "norm", None), t.nn.Identity):
            loaded_reasons.append("native truncated model norm is not Identity")
        base_config = getattr(base, "config", None)
        if getattr(base_config, "num_hidden_layers", None) != expected_layers:
            loaded_reasons.append(
                "native config num_hidden_layers="
                f"{getattr(base_config, 'num_hidden_layers', None)!r} "
                f"expected={expected_layers}"
            )
        layer_types = getattr(base_config, "layer_types", None)
        if layer_types is not None:
            try:
                layer_types_length = len(layer_types)
            except TypeError:
                layer_types_length = None
            if layer_types_length != expected_layers:
                loaded_reasons.append(
                    f"native config layer_types length={layer_types_length} "
                    f"expected={expected_layers}"
                )
        hidden_size = getattr(base_config, "hidden_size", None)
        if hidden_size is not None and hidden_size != sae_input_width:
            loaded_reasons.append(
                f"model hidden_size={hidden_size} does not match SAE input width={sae_input_width}"
            )
        attention_implementation = getattr(
            base_config,
            "_attn_implementation",
            None,
        )
        if attention_implementation != "sdpa":
            loaded_reasons.append(
                f"native attention implementation={attention_implementation!r} expected='sdpa'"
            )
        resolved_bos_id = resolve_bos_token_id(self._tokenizer)
        if resolved_bos_id is None or resolved_bos_id != self._ignore_bos_token_id:
            loaded_reasons.append(
                f"tokenizer BOS id={resolved_bos_id!r} expected={self._ignore_bos_token_id!r}"
            )
        if resolved_bos_id == 0:
            loaded_reasons.append("tokenizer BOS id collides with pinned pad/dummy token id 0")
        if loaded_reasons:
            raise AutoInterpDataError(
                "loaded fixed-block model violates its calibrated execution "
                f"contract ({'; '.join(loaded_reasons)})"
            )
        invalid = [
            target.feature_ids[0] for target in self._targets if target.feature_ids[0] >= sae_width
        ]
        if invalid:
            raise AutoInterpDataError(
                f"batched target feature IDs are out of range for SAE width {sae_width}: {invalid}"
            )

    def _new_target_build_state(self, target: ScoringTarget) -> _TargetBuildState:
        positive_docs = tuple(
            sorted(
                self._positive_docs.get(target.target_key, []),
                key=lambda ref: (
                    -self._positive_doc_scores[target.target_key].get(
                        ref.stable_key(),
                        0.0,
                    ),
                    ref.stable_key(),
                ),
            )
        )
        intruder_docs = list(self._intruder_docs)
        rng = random.Random(
            self._cfg.seed + int(hashlib.sha256(target.target_key.encode()).hexdigest()[:8], 16)
        )
        rng.shuffle(intruder_docs)
        token_cfg = self._cfg.token_examples
        return _TargetBuildState(
            target=target,
            positive_docs=positive_docs,
            intruder_docs=tuple(intruder_docs),
            example_ctx_len=token_cfg.example_ctx_len,
            max_windows_per_source_doc=token_cfg.max_windows_per_source_doc,
            max_positive_windows=token_cfg.max_positive_windows_per_target,
            max_non_activating_windows=(token_cfg.max_non_activating_windows_per_target),
        )

    def _new_block_build(self, block: TargetBlock) -> _BlockBuild:
        states = tuple(
            None if target is None else self._new_target_build_state(target)
            for target in block.targets_by_slot
        )
        return _BlockBuild(block=block, states_by_slot=states)

    def _build_and_commit_blocks(
        self,
        blocks: list[TargetBlock],
        *,
        cache: TokenExampleCache,
        namespaces: dict[str, str],
    ) -> dict[str, TokenExampleSet]:
        """Run deterministic two-lane pairs and persist in canonical order."""
        results: dict[str, TokenExampleSet] = {}
        prep_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="autointerp-token-prep")
            if self._materialization_prefetch
            else None
        )
        gpu_executor: ThreadPoolExecutor | None = None
        try:
            gpu_executor = ThreadPoolExecutor(
                max_workers=_BATCH_GPU_SINGLETON_CONCURRENCY,
                thread_name_prefix="autointerp-token-gpu",
            )
            for pair_start in range(0, len(blocks), 2):
                pair_started_ns = (
                    time.perf_counter_ns() if self._batch_pair_timing_sink is not None else None
                )
                builds = [
                    self._new_block_build(block) for block in blocks[pair_start : pair_start + 2]
                ]
                for build in builds:
                    self._schedule_next_prepared_batch(build, executor=prep_executor)

                turn = 0
                while any(build.completed is None for build in builds):
                    build = builds[turn % len(builds)]
                    turn += 1
                    if build.completed is not None:
                        continue
                    prepared = self._take_prepared_batch(build)
                    if not prepared.has_work:
                        self._validate_final_block_shadow(build)
                        build.completed = {
                            state.target.target_key: state.result()
                            for state in build.states_by_slot
                            if state is not None
                        }
                        continue
                    docs_by_slot = self._execute_prepared_batch(
                        prepared,
                        gpu_executor=gpu_executor,
                    )
                    for slot, row in enumerate(prepared.rows):
                        if row is None:
                            if docs_by_slot[slot] is not None:
                                raise AutoInterpDataError(
                                    f"batch executor produced a document for dummy slot {slot}"
                                )
                            continue
                        state = build.states_by_slot[slot]
                        doc = docs_by_slot[slot]
                        if state is None or doc is None:
                            raise AutoInterpDataError(
                                f"batch executor lost real slot {slot} in block {build.block.block_id}"
                            )
                        state.accept_doc(
                            phase=row.phase,
                            doc_ref=row.doc_ref,
                            doc=doc,
                        )
                        build.last_surfaces_by_slot[slot] = (row, doc)
                    self._schedule_next_prepared_batch(build, executor=prep_executor)

                for build in builds:
                    assert build.completed is not None
                    for target in build.block.targets:
                        self._validate_cached_examples(
                            target,
                            build.completed[target.target_key],
                        )
                    cache.set_many(
                        (
                            namespaces[target.target_key],
                            target.target_key,
                            build.completed[target.target_key],
                        )
                        for target in build.block.targets
                    )
                    for target in build.block.targets:
                        examples = build.completed[target.target_key]
                        results[target.target_key] = examples
                        logger.info(
                            f"Built token examples for {target.display_name}: "
                            f"positives={len(examples.positives)} "
                            f"non_activating={len(examples.non_activating)}"
                        )
                if self._batch_pair_timing_sink is not None:
                    assert pair_started_ns is not None
                    self._batch_pair_timing_sink.append(
                        {
                            "block_ids": tuple(build.block.block_id for build in builds),
                            "target_keys": tuple(
                                target.target_key
                                for build in builds
                                for target in build.block.targets
                            ),
                            "elapsed_ns": time.perf_counter_ns() - pair_started_ns,
                        }
                    )
        finally:
            try:
                if gpu_executor is not None:
                    gpu_executor.shutdown(wait=True, cancel_futures=True)
            finally:
                if prep_executor is not None:
                    prep_executor.shutdown(wait=True, cancel_futures=True)
        return results

    def _validate_final_block_shadow(self, build: _BlockBuild) -> None:
        if not self._batch_verified or not build.last_surfaces_by_slot:
            return
        slot = min(build.last_surfaces_by_slot)
        row, doc = build.last_surfaces_by_slot[slot]
        reference, _residual = self._serial_surfaces_for_prepared_doc(
            row,
            capture_residual=False,
        )
        candidate = doc.sae_acts[:, 0].detach().to("cpu", dtype=t.float32)
        self._validate_batch_surface(
            reference,
            candidate,
            target_key=row.target_key,
            context=(f"final block={build.block.block_id} slot={slot} target={row.target_key}"),
            surface="selected_sae_final_shadow",
            surface_kind=BatchSurfaceKind.SELECTED_SAE,
        )

    def _schedule_next_prepared_batch(
        self,
        build: _BlockBuild,
        *,
        executor: ThreadPoolExecutor | None,
    ) -> None:
        if build.future is not None:
            raise AutoInterpDataError(
                f"block {build.block.block_id} already has an outstanding prep future"
            )
        work_by_slot = tuple(
            None if state is None else state.next_doc() for state in build.states_by_slot
        )
        if executor is None:
            future: Future[PreparedBatch] = Future()
            try:
                future.set_result(self._prepare_batch(build.block, work_by_slot))
            except BaseException as exc:
                future.set_exception(exc)
            build.future = future
            return
        build.future = executor.submit(self._prepare_batch, build.block, work_by_slot)

    @staticmethod
    def _take_prepared_batch(build: _BlockBuild) -> PreparedBatch:
        future = build.future
        if future is None:
            raise AutoInterpDataError(
                f"block {build.block.block_id} has no prepared batch to consume"
            )
        build.future = None
        return future.result()

    def _prepare_batch(
        self,
        block: TargetBlock,
        work_by_slot: tuple[tuple[MaterializationPhase, SourceDocRef] | None, ...],
    ) -> PreparedBatch:
        if len(work_by_slot) != block.batch_size:
            raise AutoInterpDataError(
                f"CPU prep work row count {len(work_by_slot)} != {block.batch_size}"
            )
        rows: list[PreparedDoc | None] = []
        for slot, work in enumerate(work_by_slot):
            target = block.targets_by_slot[slot]
            if work is None:
                rows.append(None)
                continue
            if target is None:
                raise AutoInterpDataError(f"CPU prep received work for empty slot {slot}")
            phase, doc_ref = work
            rows.append(
                self._prepare_doc(
                    block=block,
                    slot=slot,
                    target=target,
                    phase=phase,
                    doc_ref=doc_ref,
                )
            )
        return PreparedBatch(
            block=block,
            rows=tuple(rows),
            model_width=self._batch_segment_size + 1,
        )

    def _prepare_doc(
        self,
        *,
        block: TargetBlock,
        slot: int,
        target: ScoringTarget,
        phase: MaterializationPhase,
        doc_ref: SourceDocRef,
    ) -> PreparedDoc:
        if self._tokenizer is None:
            raise AutoInterpDataError("CPU batch prep requires a loaded tokenizer")
        text = self._full_text(doc_ref)
        model_input_ids, real_token_ids, token_offset = self._reforward_surface(text, doc_ref)
        if not model_input_ids or not real_token_ids:
            raise AutoInterpDataError(f"Document {doc_ref.stable_key()} produced no SAE tokens")
        activation_slice_start = len(model_input_ids) - len(real_token_ids)
        token_strings = tuple(
            str(self._tokenizer.decode([int(token_id)])) for token_id in real_token_ids
        )
        return PreparedDoc(
            block_ordinal=block.ordinal,
            slot=slot,
            target_key=target.target_key,
            feature_id=target.feature_ids[0],
            phase=phase,
            doc_ref=doc_ref,
            source_text_hash=_source_text_hash(text),
            model_token_ids=tuple(int(token_id) for token_id in model_input_ids),
            real_token_ids=tuple(int(token_id) for token_id in real_token_ids),
            token_strings=token_strings,
            token_offset=token_offset,
            activation_slice_start=activation_slice_start,
        )

    def _execute_prepared_batch(
        self,
        prepared: PreparedBatch,
        *,
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[_DocActivations | None, ...]:
        real_slots = [slot for slot, row in enumerate(prepared.rows) if row is not None]
        if not real_slots:
            raise AutoInterpDataError("batch executor was called with an empty prepared batch")
        startup_probe_slots = (
            self._startup_probe_slots(prepared) if not self._batch_verified else ()
        )
        calibration_residual_slots_list: list[int] = []
        if self._batch_validation_anchor_target_keys is not None:
            for slot in real_slots:
                row = prepared.rows[slot]
                assert row is not None
                if row.target_key in self._batch_validation_anchor_target_keys:
                    calibration_residual_slots_list.append(slot)
        calibration_residual_slots = tuple(calibration_residual_slots_list)
        residual_capture_slots = tuple(
            sorted(set(startup_probe_slots) | set(calibration_residual_slots))
        )
        candidate, candidate_residuals = self._run_native_projection_batch(
            prepared,
            capture_residual_slots=residual_capture_slots,
            gpu_executor=gpu_executor,
        )

        if not self._batch_verified:
            repeated, repeated_residuals = self._run_native_projection_batch(
                prepared,
                capture_residual_slots=startup_probe_slots,
                gpu_executor=gpu_executor,
            )
            assert_candidate_repeat_exact(
                candidate,
                repeated,
                context=f"startup block={prepared.block.block_id}",
            )
            if set(candidate_residuals) != set(residual_capture_slots) or set(
                repeated_residuals
            ) != set(startup_probe_slots):
                raise AutoInterpDataError("startup residual capture returned the wrong probe slots")
            for slot in startup_probe_slots:
                row = prepared.rows[slot]
                assert row is not None
                reference, reference_residual = self._serial_surfaces_for_prepared_doc(
                    row,
                    capture_residual=True,
                )
                assert reference_residual is not None
                batched = self._real_activation_slice(candidate[slot], row)
                self._validate_batch_surface(
                    reference,
                    batched,
                    target_key=row.target_key,
                    context=(
                        f"startup block={prepared.block.block_id} slot={slot} "
                        f"target={row.target_key} phase={row.phase.value}"
                    ),
                    surface="selected_sae_startup",
                    surface_kind=BatchSurfaceKind.SELECTED_SAE,
                )
                candidate_residual = self._real_residual_slice(
                    candidate_residuals[slot],
                    row,
                )
                repeated_residual = self._real_residual_slice(
                    repeated_residuals[slot],
                    row,
                )
                assert_candidate_repeat_exact(
                    candidate_residual,
                    repeated_residual,
                    context=(
                        f"startup residual repeat block={prepared.block.block_id} slot={slot}"
                    ),
                )
                self._validate_batch_surface(
                    reference_residual,
                    candidate_residual,
                    target_key=row.target_key,
                    context=(
                        f"startup residual block={prepared.block.block_id} slot={slot} "
                        f"target={row.target_key} phase={row.phase.value}"
                    ),
                    surface="raw_residual_startup",
                    surface_kind=BatchSurfaceKind.RAW_RESIDUAL,
                    require_support_bound=False,
                )
            companion_probe = self._same_probe_companion_batch(
                prepared,
                probe_slot=real_slots[0],
            )
            original_tokens = prepared.token_tensor()
            companion_tokens = companion_probe.token_tensor()
            if not t.equal(original_tokens[real_slots[0]], companion_tokens[real_slots[0]]):
                raise AutoInterpDataError("companion leakage probe changed the probe row")
            if not any(
                not t.equal(original_tokens[slot], companion_tokens[slot])
                for slot in range(prepared.block.batch_size)
                if slot != real_slots[0]
            ):
                raise AutoInterpDataError(
                    "companion leakage probe did not change any companion row"
                )
            companion_candidate, companion_residuals = self._run_native_projection_batch(
                companion_probe,
                capture_residual_slots=(real_slots[0],),
                gpu_executor=gpu_executor,
            )
            probe_row = prepared.rows[real_slots[0]]
            assert probe_row is not None
            original_probe = self._real_activation_slice(candidate[real_slots[0]], probe_row)
            changed_companions_probe = self._real_activation_slice(
                companion_candidate[real_slots[0]],
                probe_row,
            )
            assert_candidate_repeat_exact(
                original_probe,
                changed_companions_probe,
                context=(f"companion-leakage block={prepared.block.block_id} slot={real_slots[0]}"),
            )
            assert_candidate_repeat_exact(
                self._real_residual_slice(
                    candidate_residuals[real_slots[0]],
                    probe_row,
                ),
                self._real_residual_slice(
                    companion_residuals[real_slots[0]],
                    probe_row,
                ),
                context=(
                    f"residual companion-leakage block={prepared.block.block_id} "
                    f"slot={real_slots[0]}"
                ),
            )
            self._batch_verified = True

        self._batch_shadow_counter += 1
        if self._batch_validation_anchor_target_keys is not None:
            shadow_slots = []
            for slot in real_slots:
                row = prepared.rows[slot]
                assert row is not None
                if row.target_key in self._batch_validation_anchor_target_keys:
                    shadow_slots.append(slot)
        elif self._batch_shadow_counter <= 4 or self._batch_shadow_counter % 128 == 0:
            shadow_slots = [real_slots[(self._batch_shadow_counter - 1) % len(real_slots)]]
        else:
            shadow_slots = []
        for slot in shadow_slots:
            row = prepared.rows[slot]
            assert row is not None
            capture_calibration_residual = (
                self._batch_validation_anchor_target_keys is not None
                and row.target_key in self._batch_validation_anchor_target_keys
            )
            reference, reference_residual = self._serial_surfaces_for_prepared_doc(
                row,
                capture_residual=capture_calibration_residual,
            )
            batched = self._real_activation_slice(candidate[slot], row)
            self._validate_batch_surface(
                reference,
                batched,
                target_key=row.target_key,
                context=(
                    f"shadow batch={self._batch_shadow_counter} "
                    f"block={prepared.block.block_id} slot={slot} target={row.target_key}"
                ),
                surface="selected_sae_periodic_shadow",
                surface_kind=BatchSurfaceKind.SELECTED_SAE,
            )
            if capture_calibration_residual:
                if reference_residual is None or slot not in candidate_residuals:
                    raise AutoInterpDataError(
                        "calibration residual shadow is missing a reference or "
                        f"candidate surface for slot {slot}"
                    )
                self._validate_batch_surface(
                    reference_residual,
                    self._real_residual_slice(candidate_residuals[slot], row),
                    target_key=row.target_key,
                    context=(
                        f"calibration residual batch={self._batch_shadow_counter} "
                        f"block={prepared.block.block_id} slot={slot} target={row.target_key}"
                    ),
                    surface="raw_residual_calibration_shadow",
                    surface_kind=BatchSurfaceKind.RAW_RESIDUAL,
                    require_support_bound=False,
                )

        docs: list[_DocActivations | None] = []
        for slot, row in enumerate(prepared.rows):
            if row is None:
                docs.append(None)
                continue
            real_acts = self._real_activation_slice(candidate[slot], row)
            if not bool(t.isfinite(real_acts).all()):
                raise AutoInterpDataError(
                    f"batched SAE activations contain NaN/Inf for {row.target_key}"
                )
            docs.append(
                _DocActivations(
                    source_text_hash=row.source_text_hash,
                    token_offset=row.token_offset,
                    token_ids=row.real_token_ids,
                    token_strings=row.token_strings,
                    sae_feature_ids=(row.feature_id,),
                    sae_acts=real_acts.unsqueeze(1),
                )
            )
        return tuple(docs)

    @beartype
    def _validate_batch_surface(
        self,
        reference: t.Tensor,
        candidate: t.Tensor,
        *,
        target_key: NonEmptyStrLike,
        context: str,
        surface: str,
        surface_kind: BatchSurfaceKind,
        require_support_bound: bool = True,
    ) -> None:
        metrics = assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=self._batch_tolerance_profile,
            context=context,
            surface_kind=surface_kind,
            require_support_bound=require_support_bound,
        )
        self._batch_validation_observations.append(
            {
                "target_key": target_key,
                "surface": surface,
                "surface_kind": surface_kind,
                "context": context,
                "require_support_bound": require_support_bound,
                **asdict(metrics),
            }
        )

    def _run_native_projection_batch(
        self,
        prepared: PreparedBatch,
        *,
        capture_residual_slots: tuple[int, ...] = (),
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        if self._native_model is None or self._sae is None or self._activation_type is None:
            raise AutoInterpDataError("native batch projection dependencies are not loaded")
        if len(capture_residual_slots) != len(set(capture_residual_slots)):
            raise AutoInterpDataError("residual capture slots contain duplicates")
        bad_capture_slots = [
            slot
            for slot in capture_residual_slots
            if slot < 0 or slot >= prepared.block.batch_size or prepared.rows[slot] is None
        ]
        if bad_capture_slots:
            raise AutoInterpDataError(
                f"residual capture requested invalid/dummy slots {bad_capture_slots}"
            )
        self._synchronize_before_gpu_singleton_streams()
        owned_executor = gpu_executor is None
        executor = (
            ThreadPoolExecutor(
                max_workers=_BATCH_GPU_SINGLETON_CONCURRENCY,
                thread_name_prefix="autointerp-token-gpu",
            )
            if gpu_executor is None
            else gpu_executor
        )
        try:
            futures: list[
                tuple[
                    int,
                    PreparedDoc,
                    Future[tuple[t.Tensor, t.Tensor | None]],
                ]
            ] = []

            def cancel_and_drain(*, skip: Future | None = None) -> None:
                for _slot, _row, pending in futures:
                    if pending is not skip:
                        pending.cancel()
                for _slot, _row, pending in futures:
                    if pending is skip:
                        continue
                    try:
                        pending.result()
                    except Exception:
                        pass

            capture_set = set(capture_residual_slots)
            selected = t.zeros(
                (prepared.block.batch_size, prepared.model_width),
                dtype=t.float32,
                device="cpu",
            )
            try:
                for slot, row in enumerate(prepared.rows):
                    if row is None:
                        continue
                    futures.append(
                        (
                            slot,
                            row,
                            executor.submit(
                                self._run_native_projection_singleton,
                                row,
                                capture_residual=slot in capture_set,
                            ),
                        )
                    )
            except Exception:
                cancel_and_drain()
                raise
            residuals: dict[int, t.Tensor] = {}
            for slot, row, future in futures:
                try:
                    projected, residual = future.result()
                except Exception as exc:
                    cancel_and_drain(skip=future)
                    if _exception_chain_contains_cuda_oom(exc):
                        raise AutoInterpDataError(
                            "two-stream singleton token-example materialization ran out of "
                            "CUDA memory; no smaller concurrency or serial fallback was "
                            "attempted: "
                            f"block={prepared.block.block_id} slot={slot} "
                            f"model_length={len(row.model_token_ids)}"
                        ) from exc
                    raise
                model_length = len(row.model_token_ids)
                if (
                    projected.device.type != "cpu"
                    or projected.dtype != t.float32
                    or projected.ndim != 1
                    or projected.shape[0] != model_length
                ):
                    cancel_and_drain()
                    raise AutoInterpDataError(
                        "native singleton projection contract mismatch: "
                        f"block={prepared.block.block_id} slot={slot} "
                        f"device={projected.device} dtype={projected.dtype} "
                        f"shape={tuple(projected.shape)} expected=cpu/float32/({model_length},)"
                    )
                selected[slot, :model_length].copy_(projected)
                if slot in capture_set:
                    if residual is None:
                        cancel_and_drain()
                        raise AutoInterpDataError(
                            f"native singleton omitted requested residual for slot {slot}"
                        )
                    if (
                        residual.device.type != "cpu"
                        or residual.dtype != t.float32
                        or residual.ndim != 2
                        or residual.shape[0] != model_length
                    ):
                        cancel_and_drain()
                        raise AutoInterpDataError(
                            "native singleton residual contract mismatch: "
                            f"block={prepared.block.block_id} slot={slot} "
                            f"device={residual.device} dtype={residual.dtype} "
                            f"shape={tuple(residual.shape)} "
                            f"expected_prefix=cpu/float32/({model_length},)"
                        )
                    residuals[slot] = residual
                elif residual is not None:
                    cancel_and_drain()
                    raise AutoInterpDataError(
                        f"native singleton returned an unrequested residual for slot {slot}"
                    )
            return selected, residuals
        finally:
            if owned_executor:
                executor.shutdown(wait=True, cancel_futures=True)

    def _synchronize_before_gpu_singleton_streams(self) -> None:
        """Publish loaded weights before independent worker streams first read them."""
        if self._batch_gpu_streams_ready:
            return
        if self._native_model is None:
            raise AutoInterpDataError("GPU singleton stream setup requires a loaded native model")
        device = t.device(self._native_model.device)
        if device.type == "cuda":
            t.cuda.synchronize(device)
        self._batch_gpu_streams_ready = True

    def _run_native_projection_singleton(
        self,
        row: PreparedDoc,
        *,
        capture_residual: bool,
    ) -> tuple[t.Tensor, t.Tensor | None]:
        """Forward one true-length serial surface on this worker's persistent stream."""
        if self._native_model is None or self._sae is None or self._activation_type is None:
            raise AutoInterpDataError("native singleton projection dependencies are not loaded")
        device = t.device(self._native_model.device)
        device_context = t.cuda.device(device) if device.type == "cuda" else nullcontext()
        with device_context:
            if device.type == "cuda":
                stream = getattr(self._batch_gpu_thread_local, "stream", None)
                if stream is None:
                    stream = t.cuda.Stream(device=device)
                    self._batch_gpu_thread_local.stream = stream
                stream_context = t.cuda.stream(stream)
            else:
                stream_context = nullcontext()
            with stream_context, t.no_grad():
                tokens = t.tensor(row.model_token_ids, dtype=t.long, device=device)
                acts = _get_activations_native(
                    self._native_model,
                    tokens.unsqueeze(0),
                    self._layer,
                    activation_type=self._activation_type,
                    hook_name=self._hook_name,
                )
                expected_prefix = (1, len(row.model_token_ids))
                if acts.ndim != 3 or tuple(acts.shape[:2]) != expected_prefix:
                    raise AutoInterpDataError(
                        "native singleton activation shape mismatch: "
                        f"slot={row.slot} got={tuple(acts.shape)} "
                        f"expected_prefix={expected_prefix}"
                    )
                row_acts = acts[0]
                _validate_selected_sae_inputs(
                    sae=self._sae,
                    acts=row_acts,
                    feature_ids=(row.feature_id,),
                )
                target_ids = t.tensor(
                    (row.feature_id,),
                    dtype=t.long,
                    device=device,
                )
                projected = _project_selected_sae_activations(
                    sae=self._sae,
                    acts=row_acts,
                    target_ids=target_ids,
                )[:, 0]
                projected_cpu = projected.detach().to("cpu", dtype=t.float32)
                residual_cpu = (
                    row_acts.detach().to("cpu", dtype=t.float32) if capture_residual else None
                )
        return projected_cpu, residual_cpu

    def _serial_surfaces_for_prepared_doc(
        self,
        row: PreparedDoc,
        *,
        capture_residual: bool,
    ) -> tuple[t.Tensor, t.Tensor | None]:
        if self._native_model is None or self._sae is None or self._activation_type is None:
            raise AutoInterpDataError("serial batch-validation dependencies are not loaded")
        tokens = t.tensor(
            row.model_token_ids,
            dtype=t.long,
            device=self._native_model.device,
        )
        with t.no_grad():
            acts = _get_activations_native(
                self._native_model,
                tokens.unsqueeze(0),
                self._layer,
                activation_type=self._activation_type,
                hook_name=self._hook_name,
            ).squeeze(0)
            projected = (
                _selected_sae_activations(
                    sae=self._sae,
                    acts=acts,
                    feature_ids=(row.feature_id,),
                )[:, 0]
                .detach()
                .to("cpu", dtype=t.float32)
            )
            residual = acts.detach().to("cpu", dtype=t.float32) if capture_residual else None
        return (
            self._real_activation_slice(projected, row),
            None if residual is None else self._real_residual_slice(residual, row),
        )

    def _real_activation_slice(self, projected: t.Tensor, row: PreparedDoc) -> t.Tensor:
        start = row.activation_slice_start
        end = start + len(row.real_token_ids)
        if projected.ndim != 1 or end > int(projected.shape[0]):
            raise AutoInterpDataError(
                f"projected activation surface cannot satisfy prepared slice for "
                f"{row.target_key}: shape={tuple(projected.shape)} slice={start}:{end}"
            )
        real = projected[start:end].detach().to("cpu", dtype=t.float32)
        if self._ignore_bos_features:
            if self._ignore_bos_token_id is None:
                raise AutoInterpDataError("ignore_bos_features requires a resolved BOS id")
            bos_mask = t.tensor(
                [token_id == self._ignore_bos_token_id for token_id in row.real_token_ids],
                dtype=t.bool,
            )
            if bool(bos_mask.any()):
                real = real.clone()
                real[bos_mask] = 0.0
        return real

    @staticmethod
    def _real_residual_slice(residual: t.Tensor, row: PreparedDoc) -> t.Tensor:
        start = row.activation_slice_start
        end = start + len(row.real_token_ids)
        if residual.ndim != 2 or end > int(residual.shape[0]):
            raise AutoInterpDataError(
                f"residual activation surface cannot satisfy prepared slice for "
                f"{row.target_key}: shape={tuple(residual.shape)} slice={start}:{end}"
            )
        real = residual[start:end].detach().to("cpu", dtype=t.float32)
        if not bool(t.isfinite(real).all()):
            raise AutoInterpDataError(f"residual activations contain NaN/Inf for {row.target_key}")
        return real

    @staticmethod
    def _startup_probe_slots(prepared: PreparedBatch) -> tuple[int, ...]:
        real = [(slot, row) for slot, row in enumerate(prepared.rows) if row is not None]
        if not real:
            return ()
        selected = {
            real[0][0],
            real[len(real) // 2][0],
            real[-1][0],
            min(real, key=lambda item: len(item[1].real_token_ids))[0],
            max(real, key=lambda item: len(item[1].real_token_ids))[0],
        }
        for slice_start in (0, 1):
            match = next(
                (slot for slot, row in real if row.activation_slice_start == slice_start),
                None,
            )
            if match is not None:
                selected.add(match)
        return tuple(sorted(selected))

    def _same_probe_companion_batch(
        self,
        prepared: PreparedBatch,
        *,
        probe_slot: int,
    ) -> PreparedBatch:
        probe = prepared.rows[probe_slot]
        if probe is None:
            raise AutoInterpDataError(f"companion probe slot {probe_slot} is empty")
        if self._sae is None:
            raise AutoInterpDataError("companion leakage probe requires a loaded SAE")
        targets_by_slot = list(prepared.block.targets_by_slot)
        real_slots = [slot for slot, target in enumerate(targets_by_slot) if target is not None]
        if len(real_slots) == 1:
            sae_width = int(self._sae.W_enc.shape[1])
            companion_slot = next(
                (
                    slot
                    for slot, target in enumerate(targets_by_slot)
                    if target is None
                    and prepared.block.block_id * prepared.block.batch_size + slot < sae_width
                ),
                None,
            )
            if companion_slot is None:
                raise AutoInterpDataError(
                    "companion leakage probe cannot populate a second valid feature slot"
                )
            probe_target = targets_by_slot[probe_slot]
            assert probe_target is not None
            companion_feature_id = (
                prepared.block.block_id * prepared.block.batch_size + companion_slot
            )
            targets_by_slot[companion_slot] = replace(
                probe_target,
                feature_ids=(companion_feature_id,),
                target_key=f"__batch_companion__:{companion_feature_id}",
                display_name=f"batch_companion_{companion_feature_id}",
            )
        companion_block = replace(
            prepared.block,
            targets_by_slot=tuple(targets_by_slot),
        )
        rows: list[PreparedDoc | None] = []
        for slot, target in enumerate(companion_block.targets_by_slot):
            if target is None:
                rows.append(None)
                continue
            rows.append(
                replace(
                    probe,
                    slot=slot,
                    target_key=target.target_key,
                    feature_id=target.feature_ids[0],
                )
            )
        companion = PreparedBatch(
            block=companion_block,
            rows=tuple(rows),
            model_width=prepared.model_width,
            pad_token_id=prepared.pad_token_id,
        )
        original_tokens = prepared.token_tensor()
        companion_tokens = companion.token_tensor()
        if any(
            not t.equal(original_tokens[slot], companion_tokens[slot])
            for slot in range(prepared.block.batch_size)
            if slot != probe_slot
        ):
            return companion

        alter_slot = next(
            slot
            for slot, row in enumerate(companion.rows)
            if slot != probe_slot and row is not None
        )
        alter_row = companion.rows[alter_slot]
        assert alter_row is not None
        model_token_ids = list(alter_row.model_token_ids)
        replacement_token_id = 0
        if model_token_ids[0] == replacement_token_id:
            if self._ignore_bos_token_id is None or self._ignore_bos_token_id == 0:
                raise AutoInterpDataError(
                    "companion leakage probe cannot choose a distinct valid token ID"
                )
            replacement_token_id = self._ignore_bos_token_id
        model_token_ids[0] = replacement_token_id
        real_token_ids = list(alter_row.real_token_ids)
        token_strings = list(alter_row.token_strings)
        if alter_row.activation_slice_start == 0:
            replacement_token_string = self._batch_companion_token_strings.get(replacement_token_id)
            if replacement_token_string is None:
                raise AutoInterpDataError(
                    "companion leakage probe has no predecoded string for replacement "
                    f"token ID {replacement_token_id}"
                )
            real_token_ids[0] = replacement_token_id
            token_strings[0] = replacement_token_string
        rows[alter_slot] = replace(
            alter_row,
            model_token_ids=tuple(model_token_ids),
            real_token_ids=tuple(real_token_ids),
            token_strings=tuple(token_strings),
        )
        return PreparedBatch(
            block=companion_block,
            rows=tuple(rows),
            model_width=prepared.model_width,
            pad_token_id=prepared.pad_token_id,
        )

    def _ensure_source_index_loaded(self) -> None:
        if self._source_index_loaded:
            return
        if self._source_index_request is None:
            return
        indexed = load_or_build_source_index(self._source_index_request)
        self._apply_source_index(indexed)
        if not self._intruder_docs:
            raise AutoInterpDataError(f"No source documents were discovered in {self._feature_dir}")
        self._source_index_loaded = True

    def _scan_shard(self, shard_path: Path, data: dict) -> None:
        offsets = data[TensorKey.OFFSETS]
        fids_all = data[TensorKey.FEATURE_IDS].to(t.int64)
        doc_ids = data[TensorKey.DOC_IDS]
        stem = shard_path.stem.removesuffix(".safetensors")
        shard_index = _shard_index_from_stem(stem)
        n_rows = int(offsets.numel()) - 1
        segment_token_starts, segment_token_ends = _segment_token_bounds(
            data,
            shard_path=shard_path,
            n_rows=n_rows,
        )

        def doc_ref_for_row(row_index: int) -> SourceDocRef:
            return _validated_source_doc_ref(
                shard_stem=stem,
                shard_index=shard_index,
                doc_id=int(doc_ids[row_index].item()),
                segment_token_start=int(segment_token_starts[row_index].item()),
                segment_token_end=int(segment_token_ends[row_index].item()),
                context=f"feature shard {shard_path} row {row_index}",
            )

        for row_index in range(n_rows):
            self._add_intruder_doc(doc_ref_for_row(row_index))

        target_feature_ids = tuple(sorted(self._targets_by_feature))
        if not target_feature_ids:
            return
        match_mask = t.zeros_like(fids_all, dtype=t.bool)
        for feature_id in target_feature_ids:
            match_mask |= fids_all == int(feature_id)
        matching_entry_indices = match_mask.nonzero(as_tuple=False).flatten().to(t.int64)
        if matching_entry_indices.numel() == 0:
            return
        offsets_i64 = offsets.to(t.int64)
        row_indices = t.searchsorted(offsets_i64[1:], matching_entry_indices, right=True)
        for entry_index, row_index in zip(
            matching_entry_indices.tolist(),
            row_indices.tolist(),
            strict=True,
        ):
            fid = int(fids_all[int(entry_index)].item())
            magnitude = float(data[TensorKey.FEATURE_MAGS][int(entry_index)].item())
            doc_ref = doc_ref_for_row(int(row_index))
            for target in self._targets_by_feature[fid]:
                self._add_positive_doc(target, doc_ref, magnitude)

    def _add_positive_doc(
        self,
        target: ScoringTarget,
        doc_ref: SourceDocRef,
        magnitude: float,
    ) -> None:
        key = doc_ref.stable_key()
        scores = self._positive_doc_scores[target.target_key]
        old_score = scores.get(key)
        if old_score is not None:
            if magnitude > old_score:
                scores[key] = magnitude
            return
        bucket = self._positive_docs[target.target_key]
        max_items = self._cfg.token_examples.max_positive_source_docs_per_target
        scores[key] = magnitude
        if len(bucket) < max_items:
            bucket.append(doc_ref)
            return
        worst_index, worst_ref = min(
            enumerate(bucket),
            key=lambda item: (
                scores[item[1].stable_key()],
                item[1].stable_key(),
            ),
        )
        worst_key = worst_ref.stable_key()
        worst_score = scores[worst_key]
        if magnitude > worst_score or (magnitude == worst_score and key < worst_key):
            bucket[worst_index] = doc_ref
            del scores[worst_key]
        else:
            del scores[key]

    def _add_intruder_doc(self, doc_ref: SourceDocRef) -> None:
        key = doc_ref.stable_key()
        if key in self._intruder_doc_keys:
            return
        self._intruder_doc_keys.add(key)
        self._intruder_doc_seen += 1
        max_items = self._cfg.token_examples.max_intruder_source_docs
        if len(self._intruder_docs) < max_items:
            self._intruder_docs.append(doc_ref)
            return
        replacement = random.Random(
            f"{self._cfg.seed}:intruder:{self._intruder_doc_seen}"
        ).randrange(self._intruder_doc_seen)
        if replacement < max_items:
            self._intruder_docs[replacement] = doc_ref

    def _cache_namespace(
        self,
        target: ScoringTarget,
        *,
        serial_reference: bool = True,
    ) -> str:
        context = self.cache_context()
        if serial_reference and self._cfg.token_examples.materialization_batch_size > 1:
            context = dict(context)
            context["method"] = _TOKEN_CACHE_METHOD
            context.pop("materialization", None)
        payload = {
            **context,
            "target_hash": target_payload_hash(target),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validate_cached_examples(
        self,
        target: ScoringTarget,
        examples: TokenExampleSet,
    ) -> None:
        for ref in (*examples.positives, *examples.non_activating):
            if ref.target_feature_ids != target.feature_ids:
                raise AutoInterpDataError(
                    f"cached token ref {ref.stable_key()} target ids "
                    f"{ref.target_feature_ids} != {target.feature_ids}"
                )

    def _build_examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        positive_docs = sorted(
            self._positive_docs.get(target.target_key, []),
            key=lambda ref: (
                -self._positive_doc_scores[target.target_key].get(ref.stable_key(), 0.0),
                ref.stable_key(),
            ),
        )
        positives: list[TokenWindowRef] = []
        for doc_ref in positive_docs:
            positives.extend(self._positive_windows_for_doc(doc_ref, target))
            if len(positives) >= self._cfg.token_examples.max_positive_windows_per_target:
                break
        positives = sorted(
            _dedupe_refs(positives),
            key=lambda ref: (-ref.max_activation, ref.stable_key()),
        )
        positives = positives[: self._cfg.token_examples.max_positive_windows_per_target]

        intruder_docs = list(self._intruder_docs)
        rng = random.Random(
            self._cfg.seed + int(hashlib.sha256(target.target_key.encode()).hexdigest()[:8], 16)
        )
        rng.shuffle(intruder_docs)
        non_activating: list[TokenWindowRef] = []
        positive_keys = {ref.stable_key() for ref in positives}
        positive_windows = {_window_identity(ref) for ref in positives}
        for doc_ref in intruder_docs:
            for ref in self._non_activating_windows_for_doc(doc_ref, target):
                if ref.stable_key() in positive_keys or _window_identity(ref) in positive_windows:
                    continue
                non_activating.append(ref)
                if (
                    len(non_activating)
                    >= self._cfg.token_examples.max_non_activating_windows_per_target
                ):
                    break
            if (
                len(non_activating)
                >= self._cfg.token_examples.max_non_activating_windows_per_target
            ):
                break
        non_activating = _dedupe_refs(non_activating)

        logger.info(
            f"Built token examples for {target.display_name}: "
            f"positives={len(positives)} non_activating={len(non_activating)}"
        )
        return TokenExampleSet(
            positives=tuple(positives),
            non_activating=tuple(non_activating),
        )

    def _positive_windows_for_doc(
        self,
        doc_ref: SourceDocRef,
        target: ScoringTarget,
    ) -> list[TokenWindowRef]:
        doc = self._doc_activations(doc_ref, target.feature_ids)
        return _positive_windows_from_doc(
            doc_ref,
            doc,
            target,
            example_ctx_len=self._cfg.token_examples.example_ctx_len,
            max_windows_per_source_doc=(self._cfg.token_examples.max_windows_per_source_doc),
        )

    def _non_activating_windows_for_doc(
        self,
        doc_ref: SourceDocRef,
        target: ScoringTarget,
    ) -> list[TokenWindowRef]:
        doc = self._doc_activations(doc_ref, target.feature_ids)
        return _non_activating_windows_from_doc(
            doc_ref,
            doc,
            target,
            example_ctx_len=self._cfg.token_examples.example_ctx_len,
            max_windows_per_source_doc=(self._cfg.token_examples.max_windows_per_source_doc),
        )

    @beartype
    def _doc_activations(
        self,
        doc_ref: SourceDocRef,
        feature_ids: NonEmptyNonNegativeIntTupleLike | None = None,
    ) -> _DocActivations:
        if feature_ids is None:
            feature_ids = self._activation_feature_ids
        if not feature_ids:
            raise AutoInterpDataError("cannot re-forward token examples with no feature ids")
        key = f"{doc_ref.stable_key()}:features={','.join(str(fid) for fid in feature_ids)}"
        if key in self._doc_cache:
            self._doc_cache.move_to_end(key)
            return self._doc_cache[key]
        self._ensure_model_loaded()
        assert self._sae is not None
        assert self._tokenizer is not None
        text = self._full_text(doc_ref)
        source_hash = _source_text_hash(text)
        model_input_ids, real_token_ids, token_offset = self._reforward_surface(text, doc_ref)
        if not model_input_ids or not real_token_ids:
            raise AutoInterpDataError(f"Document {doc_ref.stable_key()} produced no SAE tokens")
        if self._native_model is not None:
            if self._activation_type is None:
                raise AutoInterpDataError("native token example backend has no activation type")
            tokens = t.tensor(model_input_ids, dtype=t.long, device=self._native_model.device)
            with t.no_grad():
                acts = _get_activations_native(
                    self._native_model,
                    tokens.unsqueeze(0),
                    self._layer,
                    activation_type=self._activation_type,
                    hook_name=self._hook_name,
                ).squeeze(0)
                sae_acts = _selected_sae_activations(
                    sae=self._sae,
                    acts=acts,
                    feature_ids=feature_ids,
                )
        else:
            if self._model is None:
                raise AutoInterpDataError("token example activation backend was not loaded")
            device = next(self._sae.parameters()).device
            sae_acts = _target_sae_activations(
                model=self._model,
                sae=self._sae,
                tokens=t.tensor(model_input_ids, dtype=t.long, device=device),
                layer=self._layer,
                hook_name=self._hook_name,
                feature_ids=feature_ids,
            )
        if self._context_independent_features and len(sae_acts) != len(real_token_ids):
            if len(sae_acts) == len(real_token_ids) + 1:
                sae_acts = sae_acts[1:]
            else:
                raise AutoInterpDataError(
                    f"CI token example activation length {len(sae_acts)} does not match "
                    f"real segment length {len(real_token_ids)} for {doc_ref.stable_key()}"
                )
        if len(sae_acts) != len(real_token_ids):
            raise AutoInterpDataError(
                f"token example activation length {len(sae_acts)} does not match "
                f"token length {len(real_token_ids)} for {doc_ref.stable_key()}"
            )
        sae_acts = sae_acts.detach().to("cpu", dtype=t.float32)
        if self._ignore_bos_features:
            if self._ignore_bos_token_id is None:
                raise AutoInterpDataError("ignore_bos_features requires a resolved BOS id")
            bos_mask = t.tensor(
                [token_id == self._ignore_bos_token_id for token_id in real_token_ids],
                dtype=t.bool,
            )
            if bool(bos_mask.any()):
                sae_acts = sae_acts.clone()
                sae_acts[bos_mask] = 0.0
        token_strings = tuple(
            str(self._tokenizer.decode([int(token_id)])) for token_id in real_token_ids
        )
        doc = _DocActivations(
            source_text_hash=source_hash,
            token_offset=token_offset,
            token_ids=tuple(int(token_id) for token_id in real_token_ids),
            token_strings=token_strings,
            sae_feature_ids=feature_ids,
            sae_acts=sae_acts,
        )
        self._doc_cache[key] = doc
        self._doc_cache.move_to_end(key)
        while len(self._doc_cache) > 2:
            self._doc_cache.popitem(last=False)
        return doc

    def _reforward_surface(
        self,
        text: str,
        doc_ref: SourceDocRef,
    ) -> tuple[list[int], list[int], int]:
        if not self._context_independent_features:
            kept_ids, _n_tokens, token_offset = _truncate_to_focused_window(
                text,
                self._tokenizer,
                self._context_window,
                self._sae_max_context,
                doc_ref.segment_token_end,
            )
            return list(kept_ids), list(kept_ids), token_offset

        kept_ids, n_tokens = _truncate_to_training_window(
            text,
            self._tokenizer,
            self._context_window,
            self._sae_max_context,
        )
        start = int(doc_ref.segment_token_start)
        raw_end = int(doc_ref.segment_token_end)
        if start < 0 or raw_end <= start or start >= len(kept_ids):
            raise AutoInterpDataError(
                f"CI token example ref {doc_ref.stable_key()} has invalid token "
                f"bounds for kept length {len(kept_ids)} (full tokens={n_tokens})"
            )
        end = min(raw_end, len(kept_ids))
        segment_ids = list(kept_ids[start:end])
        if not segment_ids:
            raise AutoInterpDataError(
                f"CI token example ref {doc_ref.stable_key()} produced empty segment"
            )
        if self._ignore_bos_token_id is None:
            raise AutoInterpDataError("context_independent token examples require a BOS id")
        bos_id = int(self._ignore_bos_token_id)
        if segment_ids[0] == bos_id:
            return segment_ids, segment_ids, start
        return [bos_id, *segment_ids], segment_ids, start

    def release_model(self) -> None:
        """Drop loaded execution state and free model/SAE CUDA memory.

        Safe once token-example building is finished: nothing in judge-phase
        scoring calls back into the provider. `_ensure_model_loaded` reloads
        transparently if this provider is ever asked to build examples again;
        fixed-block startup proofs are invalidated so that reload is rechecked.
        """
        self._model = None
        self._native_model = None
        self._sae = None
        self._tokenizer = None
        self._ignore_bos_token_id = None
        self._activation_backend = "unloaded"
        self._batch_verified = False
        self._batch_shadow_counter = 0
        self._batch_companion_token_strings.clear()
        self._batch_gpu_thread_local = threading.local()
        self._batch_gpu_streams_ready = False
        if t.cuda.is_available():
            t.cuda.empty_cache()

    def _ensure_model_loaded(self) -> None:
        if self._sae is not None:
            return
        parsed = parse_feature_dir(self._feature_dir)
        stats_path = find_statistics_path(self._feature_dir)
        stats = load_statistics(stats_path) if stats_path is not None else None
        self._context_window = (
            int(self._cfg.token_examples.context_window)
            if self._cfg.token_examples.context_window is not None
            else int(stats.metadata.context_window)
            if stats is not None
            else 2048
        )
        model_name = self._cfg.token_examples.model or parsed.model
        release = SAERelease(self._cfg.token_examples.sae_release)
        l0 = _coerce_l0(release, self._cfg.token_examples.sae_l0)
        resolved = resolve_sae(
            model=model_name,
            release=release,
            activation_type=parsed.activation_type,
            layer=parsed.layer,
            width=parsed.width,
            l0=l0,
        )
        backend = self._cfg.token_examples.activation_backend
        use_native = self._should_use_native_backend(
            backend=backend,
            model_name=model_name,
            hook_name=resolved.hook_name,
            device=self._cfg.token_examples.device,
        )
        if use_native:
            model_dtype = {
                "float32": t.float32,
                "bfloat16": t.bfloat16,
            }[self._cfg.token_examples.model_dtype]
            native_model = build_native_model(
                model_name,
                self._cfg.token_examples.device,
                model_dtype,
                parsed.layer,
            )
            if native_model is None:
                raise AutoInterpDataError(
                    f"native token example backend failed to load {model_name}"
                )
            sae = SAE.from_pretrained(
                release=resolved.release,
                sae_id=resolved.sae_id,
                device=self._cfg.token_examples.device,
            )
            if model_dtype != t.float32:
                sae.to(model_dtype)
            self._native_model = native_model
            self._model = None
            self._sae = sae
            self._tokenizer = resolve_tokenizer(model_name)
            self._sae_max_context = self._context_window
            self._activation_backend = "native"
            logger.info(
                "token examples using native activation backend "
                f"for {model_name} layer {parsed.layer}"
            )
        else:
            model, sae, _tokenizer = _load_transformerlens_token_example_model(
                model_name=model_name,
                resolved=resolved,
                device=self._cfg.token_examples.device,
                target_layer=parsed.layer,
            )
            self._model = model
            self._native_model = None
            self._sae = sae
            self._tokenizer = model.tokenizer
            if self._tokenizer is None:
                raise AutoInterpDataError("loaded model has no tokenizer")
            self._sae_max_context = int(model.cfg.n_ctx)
            self._activation_backend = "transformerlens"
            logger.info(
                "token examples using TransformerLens activation backend "
                f"for {model_name} layer {parsed.layer} "
                f"(first_n_layers={parsed.layer + 1})"
            )
        if self._context_independent_features or self._ignore_bos_features:
            self._ignore_bos_token_id = resolve_bos_token_id(self._tokenizer)
            if self._ignore_bos_token_id is None:
                raise AutoInterpDataError(
                    "context-independent/no-BOS token examples require tokenizer "
                    f"to define a BOS token id for model={model_name!r}"
                )
        self._layer = parsed.layer
        self._activation_type = parsed.activation_type
        self._hook_name = resolved.hook_name

    def _should_use_native_backend(
        self,
        *,
        backend: str,
        model_name: str,
        hook_name: str | None,
        device: str,
    ) -> bool:
        if backend == "transformerlens":
            return False
        if backend == "auto":
            return False
        compatible = (
            device.startswith("cuda")
            and model_name.startswith(("gemma-2", "gemma-3"))
            and hook_name is not None
            and "hook_resid_post" in hook_name
        )
        if backend == "native" and not compatible:
            raise AutoInterpDataError(
                "token_examples.activation_backend=native requires a CUDA Gemma "
                f"residual-post SAE; got model={model_name!r}, device={device!r}, "
                f"hook_name={hook_name!r}"
            )
        return compatible

    def _full_text(self, doc_ref: SourceDocRef) -> str:
        shard_index = doc_ref.shard_index
        if shard_index < 0 or shard_index >= len(self._parquets):
            raise AutoInterpDataError(
                f"Cannot resolve shard_index={shard_index}; "
                f"{len(self._parquets)} parquet shards are available"
            )
        offsets = self._row_group_offsets.get(shard_index)
        if offsets is None:
            offsets = _shard_row_group_offsets(self._parquets[shard_index])
            self._row_group_offsets[shard_index] = offsets
        total_docs = offsets[-1]
        if doc_ref.doc_id < 0 or doc_ref.doc_id >= total_docs:
            raise AutoInterpDataError(
                f"doc_id={doc_ref.doc_id} out of range for shard_index="
                f"{shard_index} ({total_docs} docs)"
            )
        # Row group holding this doc: the last offset <= doc_id.
        row_group_index = bisect.bisect_right(offsets, doc_ref.doc_id) - 1
        key = (shard_index, row_group_index)
        entry = self._row_group_cache.get(key)
        if entry is None:
            texts = _load_shard_row_group(self._parquets[shard_index], row_group_index)
            entry = (texts, _row_group_text_chars(texts))
            self._row_group_cache[key] = entry
            self._row_group_cache_total_chars += entry[1]
            self._evict_row_groups()
        self._row_group_cache.move_to_end(key)
        texts = entry[0]
        return texts[doc_ref.doc_id - offsets[row_group_index]]

    def _evict_row_groups(self) -> None:
        """Bound the row-group text cache to ``_ROW_GROUP_CACHE_CHAR_BUDGET``.

        Never evicts the most-recently-inserted entry (``len > 1`` guard), so the
        row group a caller just loaded is always still resident on return. Each
        entry carries its own char count, so the running total can never drift
        from the cache contents.
        """
        while (
            self._row_group_cache_total_chars > _ROW_GROUP_CACHE_CHAR_BUDGET
            and len(self._row_group_cache) > 1
        ):
            _evicted_key, (_evicted_texts, evicted_chars) = self._row_group_cache.popitem(
                last=False
            )
            self._row_group_cache_total_chars -= evicted_chars


class TokenExampleSampler:
    """Construct deterministic token-level intruder trials."""

    def __init__(
        self,
        cfg: AutoInterpFilterConfig,
        *,
        provider: TokenExampleProvider,
        document_blacklist: ResolvedDocumentBlacklist | None = None,
    ) -> None:
        self._cfg = cfg
        self._provider = provider
        resolved_blacklist = (
            resolve_document_blacklist(cfg.document_blacklist_path)
            if document_blacklist is None
            else document_blacklist
        )
        if resolved_blacklist.configured_path != cfg.document_blacklist_path:
            raise AutoInterpConfigError(
                "resolved document blacklist path does not match AutoInterpFilterConfig: "
                f"resolved={resolved_blacklist.configured_path!r}, "
                f"configured={cfg.document_blacklist_path!r}"
            )
        self._document_blacklist = resolved_blacklist.entries

    def build_trials(self, target: ScoringTarget) -> list[IntruderTrial]:
        examples = self._provider.examples_for_target(target)
        return self._build_trials_from_examples(target, examples)

    def build_trials_many(
        self,
        targets: list[ScoringTarget],
    ) -> list[list[IntruderTrial]]:
        """Build trials after optional bulk token-example materialization.

        The duck-typed extension keeps existing/fake ``TokenExampleProvider``
        implementations source-compatible. The single-target public method is
        intentionally left as the serial oracle.
        """
        expected_keys = [target.target_key for target in targets]
        if len(expected_keys) != len(set(expected_keys)):
            raise AutoInterpDataError("bulk token-example request contains duplicate targets")
        bulk_method = getattr(self._provider, "examples_for_targets", None)
        if not callable(bulk_method):
            return [self.build_trials(target) for target in targets]
        examples_by_target = bulk_method(targets)
        actual_keys = set(examples_by_target)
        if actual_keys != set(expected_keys):
            missing = sorted(set(expected_keys) - actual_keys)
            extra = sorted(actual_keys - set(expected_keys))
            raise AutoInterpDataError(
                "bulk token-example provider returned the wrong target set: "
                f"missing={missing} extra={extra}"
            )
        return [
            self._build_trials_from_examples(target, examples_by_target[target.target_key])
            for target in targets
        ]

    def _build_trials_from_examples(
        self,
        target: ScoringTarget,
        examples: TokenExampleSet,
    ) -> list[IntruderTrial]:
        if self._document_blacklist:
            examples = replace(
                examples,
                positives=tuple(
                    ref
                    for ref in examples.positives
                    if document_key(shard_stem=ref.shard_stem, doc_id=ref.doc_id)
                    not in self._document_blacklist
                ),
                non_activating=tuple(
                    ref
                    for ref in examples.non_activating
                    if document_key(shard_stem=ref.shard_stem, doc_id=ref.doc_id)
                    not in self._document_blacklist
                ),
            )
        quantiles = self._quantile_groups(examples.positives)
        usable_quantiles = [
            (quantile, refs)
            for quantile, refs in quantiles
            if len(refs) >= self._cfg.n_positive_examples
        ]
        if not usable_quantiles or not examples.non_activating:
            return []

        rng = random.Random(
            self._cfg.seed + int(hashlib.sha256(target.target_key.encode()).hexdigest()[:8], 16)
        )
        trials: list[IntruderTrial] = []
        for trial_index in range(self._cfg.n_trials_per_target):
            quantile, bucket = usable_quantiles[trial_index % len(usable_quantiles)]
            positive_refs = self._sample_positive_refs(bucket, rng)
            if positive_refs is None:
                continue
            intruder = self._sample_intruder(examples.non_activating, positive_refs, rng)
            if intruder is None:
                continue
            active_token_counts = [
                sum(1 for activation in ref.activations if activation > 0.0)
                for ref in positive_refs
            ]
            avg_highlights = sum(active_token_counts) // len(active_token_counts)
            intruder_highlights = self._sample_intruder_highlights(
                intruder,
                n_highlights=avg_highlights,
            )
            intruder_index = rng.randrange(self._cfg.n_positive_examples + 1)
            trial_key = self._trial_key(
                target,
                trial_index=trial_index,
                positive_refs=positive_refs,
                intruder_ref=intruder,
                intruder_index=intruder_index,
                quantile=quantile,
                intruder_highlight_indices=intruder_highlights,
            )
            trials.append(
                IntruderTrial(
                    target=target,
                    trial_index=trial_index,
                    positive_refs=positive_refs,
                    intruder_ref=intruder,
                    intruder_index=intruder_index,
                    quantile=quantile,
                    intruder_highlight_indices=intruder_highlights,
                    activation_threshold_ratio=(
                        self._cfg.token_examples.activation_threshold_ratio
                    ),
                    trial_key=trial_key,
                )
            )
        return trials

    def _quantile_groups(
        self,
        positives: tuple[TokenWindowRef, ...],
    ) -> list[tuple[int, list[TokenWindowRef]]]:
        positives = tuple(_dedupe_refs(list(positives)))
        sorted_refs = sorted(
            positives,
            key=lambda ref: (-ref.max_activation, ref.stable_key()),
        )
        n = len(sorted_refs)
        if n == 0:
            return []
        groups: list[tuple[int, list[TokenWindowRef]]] = []
        for quantile in range(self._cfg.n_quantiles):
            start = (quantile * n) // self._cfg.n_quantiles
            end = ((quantile + 1) * n) // self._cfg.n_quantiles
            if end <= start:
                continue
            group = [replace(ref, quantile=quantile) for ref in sorted_refs[start:end]]
            groups.append((quantile, group))
        return groups

    def _sample_positive_refs(
        self,
        bucket: list[TokenWindowRef],
        rng: random.Random,
    ) -> tuple[TokenWindowRef, ...] | None:
        for _attempt in range(16):
            candidates = list(bucket)
            rng.shuffle(candidates)
            selected: list[TokenWindowRef] = []
            for ref in candidates:
                if any(_source_windows_overlap(ref, old) for old in selected):
                    continue
                selected.append(ref)
                if len(selected) == self._cfg.n_positive_examples:
                    return tuple(selected)
        return None

    def _sample_intruder(
        self,
        non_activating: tuple[TokenWindowRef, ...],
        positive_refs: tuple[TokenWindowRef, ...],
        rng: random.Random,
    ) -> TokenWindowRef | None:
        positive_keys = {ref.stable_key() for ref in positive_refs}
        positive_windows = {_window_identity(ref) for ref in positive_refs}
        candidates = list(non_activating)
        rng.shuffle(candidates)
        for ref in candidates:
            if ref.stable_key() in positive_keys or _window_identity(ref) in positive_windows:
                continue
            if any(_source_windows_overlap(ref, positive_ref) for positive_ref in positive_refs):
                continue
            if ref.max_activation != 0.0:
                continue
            return ref
        return None

    def _sample_intruder_highlights(
        self,
        ref: TokenWindowRef,
        *,
        n_highlights: int,
    ) -> tuple[int, ...]:
        return deterministic_intruder_highlight_indices(
            ref,
            n_highlights=n_highlights,
        )

    def _trial_key(
        self,
        target: ScoringTarget,
        *,
        trial_index: int,
        positive_refs: tuple[TokenWindowRef, ...],
        intruder_ref: TokenWindowRef,
        intruder_index: int,
        quantile: int | None,
        intruder_highlight_indices: tuple[int, ...],
    ) -> str:
        return intruder_trial_key(
            target_hash=target_payload_hash(target),
            trial_index=trial_index,
            positive_ref_keys=[ref.stable_key() for ref in positive_refs],
            intruder_ref_key=intruder_ref.stable_key(),
            intruder_index=intruder_index,
            quantile=quantile,
            intruder_highlight_indices=list(intruder_highlight_indices),
            n_positive_examples=self._cfg.n_positive_examples,
            n_quantiles=self._cfg.n_quantiles,
            example_ctx_len=self._cfg.token_examples.example_ctx_len,
            activation_threshold_ratio=(self._cfg.token_examples.activation_threshold_ratio),
        )


def _dedupe_refs(refs: list[TokenWindowRef]) -> list[TokenWindowRef]:
    best: dict[tuple, TokenWindowRef] = {}
    for ref in refs:
        key = _window_identity(ref)
        old = best.get(key)
        if old is None or (ref.max_activation, ref.stable_key()) > (
            old.max_activation,
            old.stable_key(),
        ):
            best[key] = ref
    return list(best.values())


def _source_windows_overlap(left: TokenWindowRef, right: TokenWindowRef) -> bool:
    if (
        left.shard_stem != right.shard_stem
        or left.shard_index != right.shard_index
        or left.doc_id != right.doc_id
        or left.source_text_hash != right.source_text_hash
    ):
        return False
    start = max(left.token_start, right.token_start)
    end = min(left.token_end, right.token_end)
    if end <= start:
        return False
    left_start = start - left.token_start
    left_end = end - left.token_start
    right_start = start - right.token_start
    right_end = end - right.token_start
    left_tokens = left.token_ids[left_start:left_end]
    right_tokens = right.token_ids[right_start:right_end]
    if left_tokens != right_tokens:
        raise AutoInterpDataError(
            "overlapping token windows from the same source disagree on token IDs: "
            f"{left.stable_key()} vs {right.stable_key()}"
        )
    return True
