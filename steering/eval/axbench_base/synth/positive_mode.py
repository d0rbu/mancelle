"""Stage 1b deterministic complete-pair target/refill synthesis.

Each concept owns a one-pass candidate source: shuffle original seed indices,
apply the bounded suffix view, admit headroom-safe stored-clean candidates in
shuffle order, then admissible dirty candidates by stable ascending
percentile. The initial complete-pair target is preflighted for every concept
before provider access. Later candidates are drawn lazily and never rescored or
reused.

One request chain owns a seed from rich attempt 1 through natural avoiding QC
and, when needed, LLM-avoiding QC. Rich and LLM-avoiding sides independently
allow ``positive_max_attempts_per_seed`` total semantic attempts. Protocol,
special-token, and QC rejection consume an attempt; exhausted seeds are
discarded atomically and replaced. Natural avoiding QC consumes no API attempt.
Exhausted HTTP transport retries abort the stage immediately without charging
the semantic attempt or admitting a replacement.

``n_positives_per_concept`` is the accepted complete-pair target. The fallback
policy is consulted only when pool exhaustion or the zero-hit guard makes that
target unreachable; ``warn`` and ``best`` retain complete pairs only. Final
rows are sorted by canonical concept and candidate order, with each rich row
immediately followed by its avoiding row, independent of worker completion
order. Stage 0c and Stage 1c retain the separate ``max_retries`` contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import signal
import struct
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from steering.eval.axbench_base import paths
from steering.eval.axbench_base._stage0c_internals import concepts_identity
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.cluster_artifacts import (
    clustered_source_enabled,
    load_cluster_candidate_identity,
    write_clustered_parquet_once,
)
from steering.eval.axbench_base.config import (
    MAX_RAW_PERCENTILE,
    AxBenchBaseConfig,
    derive_positive_qc_thresholds,
    effective_extraction_model_id,
)
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.extraction import (
    _infer_n_sae_features,
    build_percentile_table_from_stats,
    feature_statistics_file_sha256,
)
from steering.eval.axbench_base.percentile_units import (
    percentile_delta_meets_or_exceeds,
)
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
    validate_positive_qc_rows,
)
from steering.eval.axbench_base.positive_selection_contract import (
    POSITIVE_CHECKPOINT_SELECTION_VERSION_V1,
    POSITIVE_INITIAL_COMPAT_TARGET,
    POSITIVE_SELECTION_CONTRACT_VERSION_V3,
    positive_artifact_selection_contract_payload,
    positive_selection_contract_payload,
    schema_with_positive_selection_contract,
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.prefix_text import (
    validate_prefix_char_bounds,
    window_prefix_text,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth import positive_checkpoint
from steering.eval.axbench_base.synth.batching import (
    SynthCompleter,
    SynthRequest,
    SynthRequestWithPayload,
    SynthResult,
    complete_synth_request_chains,
)
from steering.eval.axbench_base.synth.concept_yield import (
    ConceptYieldRecord,
    filter_positive_concept_rows,
    prepare_concept_yield_report,
    write_concept_yield_report,
    write_filtered_concepts,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    Stage1ContinuationRequest,
    build_stage1_continuation_request,
    parse_stage1_continuation_or_none,
)
from steering.eval.axbench_base.synth.llm_response_cache import (
    SynthResponseCache,
    make_cache_key,
)
from steering.eval.axbench_base.synth.openrouter_client import OpenRouterSynthClient
from steering.eval.axbench_base.synth.prefix_percentile_store import (
    PrefixPercentileStore,
)
from steering.eval.axbench_base.synth.qc import (
    AvoidContinuationOutcome,
    ContinuationQcInput,
    PrefixPercentileCache,
    classify_avoid_continuation,
    classify_avoid_continuations,
    fc_percentile_on_prefix,
    output_fc_percentile_for_continuation,
    output_fc_percentiles_for_continuations,
)
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
    stage1_continuation_contract_payload,
)
from steering.eval.axbench_base.templates import load_template
from steering.eval.axbench_base.tokenizer_cache import (
    get_nanochat_tokenizer as get_tokenizer,
)
from steering.eval.axbench_base.tokenizer_utils import tokenize_labeled_pair
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    DOC_CONTINUATION_GEN_SOURCE_PREFIX,
    MIN_RAW_PERCENTILE,
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from steering.immutable_io import is_sha256_hex
from steering.segmentation import compute_tokenizer_checksum

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import (
        ClusterFeatureContract,
        ClusterFeatureTarget,
    )


def _cluster_target_kwargs(
    cluster_target: ClusterFeatureTarget | None,
) -> dict[str, Any]:
    return {} if cluster_target is None else {"cluster_target": cluster_target}


def _cluster_cache_suffix(cluster_target: ClusterFeatureTarget | None) -> str:
    if cluster_target is None:
        return ""
    return f"_fc{cluster_target.feature_contract_fingerprint}"


def _stage1b_continuation_request(
    template: str,
    *,
    concept: Concept,
    prefix_text: str,
    seed_idx: int,
    attempt: int,
    content_class: str,
    cluster_target: ClusterFeatureTarget | None,
) -> Stage1ContinuationRequest:
    """Build one deterministic rich/avoid request for a Stage 1b attempt."""
    class_kind = "RICH" if content_class == CONCEPT_RICH_LABEL else "AVOID"
    cluster_fingerprint = (
        None
        if cluster_target is None
        else cluster_target.feature_contract_fingerprint
    )
    return build_stage1_continuation_request(
        template,
        label=f"Stage 1b {content_class}",
        kind=f"STAGE1B_{class_kind}",
        identity={
            "stage": "1b",
            "content_class": content_class,
            "concept_id": concept.concept_id,
            "feature_id": concept.feature_index,
            "seed_idx": seed_idx,
            "attempt": attempt,
            "prefix_sha256": hashlib.sha256(prefix_text.encode("utf-8")).hexdigest(),
            "feature_contract_fingerprint": cluster_fingerprint,
        },
        prefix_text=prefix_text,
        description=concept.description,
    )


_CONCEPT_SEED_OFFSET = 0xE70001  # distinct from global-negatives offset
# Distinct from _CONCEPT_SEED_OFFSET (rich) and negative_mode's
# _NEG_SYNTH_SEED_OFFSET (0xE90001); salts the paired concept_avoiding rows.
#: Minimum wall-clock gap between Stage 1b in-flight progress logs. Chosen so
#: a long stage stays observable without the heartbeat itself becoming noise
#: on a fast one; it gates only logging, never scheduling.
_PROGRESS_HEARTBEAT_SECONDS = 60.0
# A hard-coded durability cadence is part of the Stage 1b checkpoint
# contract.  Terminal callbacks run on the caller thread after deterministic
# replacement admission, so a checkpoint here never races mutable tracker
# state and every newly returned provider request is already durable.
_STAGE1B_CHECKPOINT_TERMINAL_INTERVAL = 64

_POS_AVOID_SEED_OFFSET = 0xEA0001


_EffectivePrefixes = tuple[str | None, ...]


@contextmanager
def _defer_sigint_for_checkpoint_state() -> Iterator[None]:
    """Defer Ctrl-C across one short multi-field scheduler-state commit.

    Provider calls, parsing, tokenization, and SAE QC remain interruptible. On
    POSIX, a SIGINT arriving during a tracker/counter commit is delivered as
    soon as the previous signal mask is restored, after the checkpoint state
    is coherent. Platforms without ``pthread_sigmask`` retain normal behavior.
    """

    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous_mask = pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _hash_frame(digest: Any, label: str, payload: bytes) -> None:
    """Hash one named byte string with unambiguous length boundaries."""

    label_bytes = label.encode("utf-8")
    digest.update(struct.pack(">Q", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _hash_int_field(digest: Any, label: str, value: int) -> None:
    _hash_frame(digest, label, b"int:" + str(value).encode("ascii"))


def _hash_text_field(digest: Any, label: str, value: str) -> None:
    _hash_frame(digest, label, b"str:" + value.encode("utf-8"))


def _hash_int_list_field(
    digest: Any,
    label: str,
    values: list[int],
) -> None:
    _hash_int_field(digest, f"{label}.length", len(values))
    for index, value in enumerate(values):
        _hash_int_field(digest, f"{label}[{index}]", value)


def _hash_float_list_field(
    digest: Any,
    label: str,
    values: list[float],
) -> None:
    _hash_int_field(digest, f"{label}.length", len(values))
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise AxBenchBaseError(
                f"Stage 1b seed identity found non-numeric {label}[{index}]="
                f"{value!r}"
            )
        if isinstance(value, int):
            payload = b"int:" + str(value).encode("ascii")
        else:
            payload = b"float64:" + struct.pack(">d", value)
        _hash_frame(digest, f"{label}[{index}]", payload)


def _seed_rows_identity(seeds_train: list[SeedRow]) -> str:
    """Migration-only v1 identity for legacy Stage 1b checkpoints."""

    digest = hashlib.sha256()
    _hash_text_field(digest, "identity_version", "stage1b_seed_rows_v1")
    _hash_int_field(digest, "row_count", len(seeds_train))
    for row_index, seed in enumerate(seeds_train):
        prefix = f"rows[{row_index}]"
        _hash_int_field(digest, f"{prefix}.shard_idx", seed.shard_idx)
        _hash_int_field(digest, f"{prefix}.doc_idx", seed.doc_idx)
        _hash_int_field(
            digest,
            f"{prefix}.source_nanochat_token_count",
            seed.source_nanochat_token_count,
        )
        _hash_int_field(
            digest,
            f"{prefix}.prefix_source_token_start",
            seed.prefix_source_token_start,
        )
        _hash_int_field(
            digest,
            f"{prefix}.prefix_source_token_end",
            seed.prefix_source_token_end,
        )
        _hash_text_field(digest, f"{prefix}.prefix_text", seed.prefix_text)
        _hash_int_list_field(
            digest,
            f"{prefix}.prefix_content_nanochat_tokens",
            seed.prefix_content_nanochat_tokens,
        )
        _hash_int_list_field(
            digest,
            f"{prefix}.sae_feature_ids",
            seed.sae_feature_ids,
        )
        _hash_float_list_field(
            digest,
            f"{prefix}.sae_feature_mags",
            seed.sae_feature_mags,
        )
        _hash_text_field(
            digest,
            f"{prefix}.continuation_text",
            seed.continuation_text,
        )
        _hash_int_list_field(
            digest,
            f"{prefix}.continuation_content_nanochat_tokens",
            seed.continuation_content_nanochat_tokens,
        )
    return digest.hexdigest()


def _stage1b_checkpoint_fingerprint(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    seed_rows_sha256: str,
    target_model_meta: TargetModelMetadata,
    *,
    cluster_contract: ClusterFeatureContract | None,
) -> dict[str, Any]:
    """Bind a partial Stage 1b transaction to every semantic input."""

    rich_template = load_template("synth_continue_with_concept.txt")
    avoiding_template = load_template("synth_continue_avoiding_concept.txt")
    cluster_fingerprint = (
        None
        if cluster_contract is None
        else cluster_contract.feature_contract_fingerprint
    )
    return {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
        ),
        "positive_selection_contract": (
            positive_selection_contract_payload(cfg)
        ),
        "ordered_concepts_sha256": concepts_identity(concepts),
        "seed_rows_sha256": seed_rows_sha256,
        "feature_statistics_sha256": feature_statistics_file_sha256(
            cfg.feature_statistics_path
        ),
        "extraction_model_id": effective_extraction_model_id(cfg),
        "sae_identity": {
            "neuronpedia_model_id": cfg.neuronpedia_model_id,
            "neuronpedia_source": cfg.neuronpedia_source,
            "sae_layer": target_model_meta.sae_layer,
        },
        "extraction_tokenizer_id": cfg.extraction_tokenizer_id,
        "nanochat_tokenizer_checksum": compute_tokenizer_checksum(
            get_tokenizer()
        ),
        "master_seed": cfg.generation.master_seed,
        "output_length": cfg.output_length,
        "synth_provider": cfg.synth.provider,
        "synth_model": cfg.synth.model,
        "rich_prompt_template_sha256": hashlib.sha256(
            rich_template.encode("utf-8")
        ).hexdigest(),
        "avoiding_prompt_template_sha256": hashlib.sha256(
            avoiding_template.encode("utf-8")
        ).hexdigest(),
        "continuation_contract": stage1_continuation_contract_payload(
            "stage1b_positive"
        ),
        "cluster_feature_contract_fingerprint": cluster_fingerprint,
        "extraction_threshold": cfg.extraction_threshold,
        # These gates run after the provider pool drains, but they still
        # decide whether the checkpointed logical history may be published.
        # Resuming under different values could turn the same shortfall into
        # success, failure, or a different surviving concept population.
        "qc_fallback_on_exhaust": cfg.qc_fallback_on_exhaust,
        "max_concept_drop_fraction": cfg.max_concept_drop_fraction,
        "max_dropped_concept_fraction": (
            cfg.max_dropped_concept_fraction
        ),
    }


def _stage1b_checkpoint_fingerprint_v1(
    cfg: AxBenchBaseConfig,
    checkpoint_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Derive the exact Phase 1 target-72 identity without rehashing inputs."""

    fingerprint = dict(checkpoint_fingerprint)
    fingerprint["checkpoint_algorithm_version"] = (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V1
    )
    target_72_cfg = cfg.replace(
        n_positives_per_concept=POSITIVE_INITIAL_COMPAT_TARGET
    )
    legacy_selection = positive_artifact_selection_contract_payload(target_72_cfg)
    legacy_selection.pop("post_provider_qc_batch_size")
    legacy_selection.pop("post_provider_qc_batching")
    legacy_selection["contract_version"] = (
        POSITIVE_SELECTION_CONTRACT_VERSION_V3
    )
    fingerprint["positive_selection_contract"] = legacy_selection
    return fingerprint


def _stage1b_checkpoint_fingerprint_v2(
    checkpoint_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Derive the exact pre-post-provider-batching schema-v2 identity."""

    fingerprint = dict(checkpoint_fingerprint)
    fingerprint["checkpoint_algorithm_version"] = (
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V2
    )
    selection = dict(fingerprint["positive_selection_contract"])
    selection.pop("post_provider_qc_batch_size")
    selection.pop("post_provider_qc_batching")
    selection["contract_version"] = (
        POSITIVE_CHECKPOINT_SELECTION_VERSION_V1
    )
    fingerprint["positive_selection_contract"] = selection
    return fingerprint


def _legacy_stage1b_fingerprint_factory(
    cfg: AxBenchBaseConfig,
    seeds_train: list[SeedRow],
    checkpoint_fingerprint: dict[str, Any],
) -> Callable[[str], dict[str, Any]]:
    """Build legacy fingerprints lazily, with one exact v1 row scan."""

    legacy_fingerprint_v3: dict[str, Any] | None = None

    def factory(checkpoint_algorithm_version: str) -> dict[str, Any]:
        nonlocal legacy_fingerprint_v3
        if legacy_fingerprint_v3 is None:
            legacy_fingerprint_v3 = dict(checkpoint_fingerprint)
            legacy_fingerprint_v3["checkpoint_algorithm_version"] = (
                positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
            )
            legacy_fingerprint_v3["seed_rows_sha256"] = _seed_rows_identity(
                seeds_train
            )
        if checkpoint_algorithm_version == (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
        ):
            return dict(legacy_fingerprint_v3)
        if checkpoint_algorithm_version == (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V2
        ):
            return _stage1b_checkpoint_fingerprint_v2(legacy_fingerprint_v3)
        if checkpoint_algorithm_version == (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V1
        ):
            return _stage1b_checkpoint_fingerprint_v1(
                cfg,
                legacy_fingerprint_v3,
            )
        raise AxBenchBaseError(
            "Unsupported Stage 1b legacy checkpoint algorithm "
            f"{checkpoint_algorithm_version!r}"
        )

    return factory


def _resolve_stage1b_prefixes(
    cfg: AxBenchBaseConfig,
    seeds_train: list[SeedRow],
) -> _EffectivePrefixes:
    """Resolve bounded Stage 1b prefix views without mutating Stage 0b seeds."""
    validate_prefix_char_bounds(
        cfg.min_prefix_chars,
        cfg.max_prefix_chars,
    )
    resolved: list[str | None] = []
    n_truncated = 0
    n_below_min = 0
    for seed_idx, seed in enumerate(seeds_train):
        prefix_text = window_prefix_text(
            seed.prefix_text,
            len(seed.prefix_text),
            cfg.max_prefix_chars,
        )
        if len(seed.prefix_text) > cfg.max_prefix_chars:
            n_truncated += 1
        if len(prefix_text) < cfg.min_prefix_chars:
            n_below_min += 1
            resolved.append(None)
            continue
        if not (cfg.min_prefix_chars <= len(prefix_text) <= cfg.max_prefix_chars):
            raise AxBenchBaseError(
                "Stage 1b resolved an out-of-bounds positive prefix: "
                f"seed_idx={seed_idx}, length={len(prefix_text)}, expected "
                f"[{cfg.min_prefix_chars}, {cfg.max_prefix_chars}]"
            )
        resolved.append(prefix_text)

    n_eligible = len(resolved) - n_below_min
    record_counter("stage1b.seed_prefixes_eligible", n_eligible)
    record_counter("stage1b.seed_prefixes_truncated", n_truncated)
    record_counter("stage1b.seed_prefixes_below_min", n_below_min)
    if n_eligible == 0:
        raise AxBenchBaseError(
            "Stage 1b has no eligible positive seed prefixes after applying "
            f"decoded-character bounds [{cfg.min_prefix_chars}, "
            f"{cfg.max_prefix_chars}] to {len(seeds_train)} Stage 0b seeds"
        )
    logger.info(
        "Stage 1b positive seed prefix bounds: eligible={}, truncated={}, "
        "below_min={}, bounds=[{}, {}]",
        n_eligible,
        n_truncated,
        n_below_min,
        cfg.min_prefix_chars,
        cfg.max_prefix_chars,
    )
    return tuple(resolved)


def _exact_output_start(prefix_text: str, output_text: str) -> int:
    """Return the owned-BOS-inclusive output boundary stored in synth rows."""

    _tokens, output_start = tokenize_labeled_pair(get_tokenizer(), prefix_text, output_text)
    return output_start


def _doc_continuation_gen_source(seed: SeedRow) -> str:
    """Return the gen_source provenance string for a document-continuation row.

    The natural-tier avoiding row's output text originated from the corpus
    document ``(shard_idx, doc_idx)``, not from any model. We record that
    provenance instead of a model tag so downstream aggregation can
    distinguish these free, deterministic rows from model-generated rows.

    Ported from the (now-removed) ``global_negatives`` module.
    """
    return (
        f"{DOC_CONTINUATION_GEN_SOURCE_PREFIX}"
        f"{seed.shard_idx}/{seed.doc_idx}"
    )


@dataclass(frozen=True)
class _SelectedPositiveSeed:
    """One concept-assigned seed with its exact canonical prefix baseline."""

    seed_idx: int
    seed: SeedRow
    prefix_text: str
    seed_percentile: float
    stored_clean: bool
    # Zero-based position in the concept's canonical admissible candidate
    # stream. It is deliberately not serialized into the unchanged row schema.
    candidate_rank: int = -1


@dataclass
class _PositivePending:
    order: int
    selected: _SelectedPositiveSeed
    accepted_row: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.accepted_row is not None

    @property
    def seed_idx(self) -> int:
        return self.selected.seed_idx


@dataclass
class _PositiveWorkItem:
    concept: Concept
    pending: _PositivePending
    cluster_target: ClusterFeatureTarget | None = None
    rich_row: dict[str, Any] | None = None
    avoiding_row: dict[str, Any] | None = None
    avoid_pending: _PositiveAvoidPending | None = None
    phase: str = "rich"
    terminal: bool = False
    rich_semantic_attempts: int = 0
    avoid_semantic_attempts: int = 0
    acceptance_sequence: int | None = None

    @property
    def order(self) -> int:
        return self.pending.order


@dataclass
class _PositiveAvoidPending:
    work: _PositiveWorkItem
    accepted_row: dict[str, Any] | None = None

    @property
    def order(self) -> int:
        return self.work.order

    @property
    def done(self) -> bool:
        return self.accepted_row is not None


@dataclass
class _PreparedPositiveBatchResult:
    """Parsed/scored Stage 1b result awaiting ordered state application."""

    position: int
    request: SynthRequestWithPayload
    result: SynthResult
    work: _PositiveWorkItem
    phase: str
    attempt: int
    continuation: str | None = None
    output_percentile: float | None = None
    natural_outcome: AvoidContinuationOutcome | None = None
    avoid_outcome: AvoidContinuationOutcome | None = None


@dataclass(frozen=True)
class _PositiveDiscardRecord:
    """Terminal seed state needed to rebuild admission and counters."""

    selected: _SelectedPositiveSeed
    side: str
    rich_semantic_attempts: int
    avoid_semantic_attempts: int


def _positive_rich_chain_request(
    cfg: AxBenchBaseConfig,
    template: str,
    work: _PositiveWorkItem,
    *,
    attempt: int,
) -> SynthRequest:
    pending = work.pending
    continuation_request = _stage1b_continuation_request(
        template,
        concept=work.concept,
        prefix_text=pending.selected.prefix_text,
        seed_idx=pending.seed_idx,
        attempt=attempt,
        content_class=CONCEPT_RICH_LABEL,
        cluster_target=work.cluster_target,
    )
    return SynthRequestWithPayload(
        index=work.order,
        prompt=continuation_request.prompt,
        cache_key=make_cache_key(
            model=cfg.synth.model,
            prompt=continuation_request.prompt,
            tag=(
                f"pos_c{work.concept.concept_id}"
                f"_s{pending.seed_idx}_a{attempt}"
                f"{_cluster_cache_suffix(work.cluster_target)}"
            ),
        ),
        attrs={
            "concept_id": work.concept.concept_id,
            "attempt": attempt,
            "seed_idx": pending.seed_idx,
        },
        payload=continuation_request,
    )


# Parquet schema for the positive-mode training corpus. Each accepted
# ``concept_rich`` row is followed by a paired ``concept_avoiding`` row of
# the SAME prefix. ``qc_zero_activation_verified`` mirrors negative mode's
# strict-zero QC column: rich rows are ``False`` (they pass a delta QC, not
# a strict-zero QC); avoiding rows are ``True`` (the concept feature was
# verified NOT to fire on the continuation).
TRAINING_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("category", pa.string()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("output_concept", pa.string()),
        ("n_prefix_nanochat_tokens", pa.int32()),
        ("gen_source", pa.string()),
        ("gen_seed", pa.int64()),
        ("qc_feature_id", pa.int32()),
        ("qc_seed_percentile", pa.float32()),
        ("qc_output_percentile", pa.float32()),
        ("qc_delta", pa.float32()),
        ("qc_attempts", pa.int32()),
        ("qc_accepted", pa.bool_()),
        ("qc_zero_activation_verified", pa.bool_()),
    ]
)


def _build_synth_client(
    cfg: AxBenchBaseConfig,
    response_cache: Any = None,
) -> OpenRouterSynthClient:
    """Thin factory so tests can monkey-patch without constructing a real client.

    ``response_cache`` is forwarded to the underlying
    :class:`OpenRouterSynthClient`. Tests that don't care about the cache
    may call this with the default ``response_cache=None``.
    """
    return OpenRouterSynthClient(
        cfg.synth,
        max_retries=cfg.synth.transport_max_retries,
        response_cache=response_cache,
    )


def _seed_has_feature(seed: SeedRow, feature_id: int) -> bool:
    """Return whether ``feature_id`` is present in a SeedRow's sorted features."""
    ids = seed.sae_feature_ids
    if not ids:
        return False
    if len(ids) < 64:
        return feature_id in ids
    pos = bisect_left(ids, feature_id)
    return pos < len(ids) and ids[pos] == feature_id


def _validate_seed_feature_ids_sorted(seeds_train: list[SeedRow]) -> None:
    """Fail fast if Stage 1b receives unsorted feature ids.

    The fast membership path in ``_seed_has_feature`` relies on sorted
    ``SeedRow.sae_feature_ids`` for large rows. Validate the invariant once
    per stage rather than paying a per-row fallback cost during selection.
    """
    for seed_idx, seed in enumerate(seeds_train):
        ids = seed.sae_feature_ids
        if any(ids[i] > ids[i + 1] for i in range(len(ids) - 1)):
            raise AxBenchBaseError(
                "Stage 1b requires sorted SeedRow.sae_feature_ids; "
                f"seed_idx={seed_idx} doc_idx={seed.doc_idx} shard_idx={seed.shard_idx}"
            )


def _validate_raw_positive_percentile(
    percentile: object,
    *,
    scorer: str,
    concept: Concept,
    seed_idx: int,
) -> float:
    if isinstance(percentile, bool) or not isinstance(percentile, Real):
        raise AxBenchBaseError(
            f"Stage 1b {scorer} returned an invalid raw percentile: "
            f"concept_id={concept.concept_id}, feature_id={concept.feature_index}, "
            f"seed_idx={seed_idx}, percentile={percentile!r}, expected 0.0 or "
            f"[{MIN_RAW_PERCENTILE}, {MAX_RAW_PERCENTILE}]"
        )
    value = float(percentile)
    if not math.isfinite(value) or not (
        value == 0.0
        or MIN_RAW_PERCENTILE <= value <= MAX_RAW_PERCENTILE
    ):
        raise AxBenchBaseError(
            f"Stage 1b {scorer} returned an invalid raw percentile: "
            f"concept_id={concept.concept_id}, feature_id={concept.feature_index}, "
            f"seed_idx={seed_idx}, percentile={value!r}, expected 0.0 or "
            f"[{MIN_RAW_PERCENTILE}, {MAX_RAW_PERCENTILE}]"
        )
    return value


def _score_positive_seed_candidates(
    candidates: list[tuple[int, SeedRow, str, bool]],
    concept: Concept,
    cfg: AxBenchBaseConfig,
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    *,
    sae_layer: int,
    prefix_cache: Any | None,
    cluster_target: ClusterFeatureTarget | None,
) -> list[_SelectedPositiveSeed]:
    """Score candidate prefixes in order and bind each score to its exact text."""

    if not candidates:
        return []
    if prefix_cache is not None:
        percentiles = prefix_cache.get_many(
            [prefix_text for _idx, _seed, prefix_text, _clean in candidates],
            concept.feature_index,
            **_cluster_target_kwargs(cluster_target),
        )
    else:
        percentiles = [
            fc_percentile_on_prefix(
                prefix_text,
                concept.feature_index,
                base_model,
                base_sae,
                percentile_table,
                sae_layer=sae_layer,
                tokenizer_id=cfg.extraction_tokenizer_id,
                **_cluster_target_kwargs(cluster_target),
            )
            for _idx, _seed, prefix_text, _clean in candidates
        ]
    if len(percentiles) != len(candidates):
        raise AxBenchBaseError(
            "Stage 1b prefix scorer returned "
            f"{len(percentiles)} scores for {len(candidates)} candidates"
        )
    return [
        _SelectedPositiveSeed(
            seed_idx=seed_idx,
            seed=seed,
            prefix_text=prefix_text,
            seed_percentile=_validate_raw_positive_percentile(
                percentile,
                scorer="prefix scorer",
                concept=concept,
                seed_idx=seed_idx,
            ),
            stored_clean=stored_clean,
        )
        for (seed_idx, seed, prefix_text, stored_clean), percentile in zip(
            candidates,
            percentiles,
            strict=True,
        )
    ]


class _PositiveCandidateSource:
    """Stateful, one-pass Stage 1b candidate stream for one concept.

    Every target first fills toward the historical target-72 boundary using
    the exact v3 scorer calls. Targets above 72 then continue in full clean
    scorer-batch chunks. Only physical clean-pool exhaustion may shorten a
    post-compatibility chunk; dirty candidates are then scored once and stably
    ordered by percentile (shuffled order breaks ties). Surplus admissible
    candidates remain buffered for deterministic replacement or target growth.
    """

    def __init__(
        self,
        concept: Concept,
        seeds_train: list[SeedRow],
        cfg: AxBenchBaseConfig,
        base_model: Any,
        base_sae: Any,
        percentile_table: Any,
        *,
        sae_layer: int,
        prefix_cache: Any | None,
        cluster_target: ClusterFeatureTarget | None,
        effective_prefixes: _EffectivePrefixes,
    ) -> None:
        if len(effective_prefixes) != len(seeds_train):
            raise AxBenchBaseError(
                "Stage 1b resolved prefix table length does not match "
                f"seeds_train: {len(effective_prefixes)} != {len(seeds_train)}"
            )
        self.concept = concept
        self.cfg = cfg
        self._base_model = base_model
        self._base_sae = base_sae
        self._percentile_table = percentile_table
        self._sae_layer = sae_layer
        self._prefix_cache = prefix_cache
        self._cluster_target = cluster_target
        self._thresholds = derive_positive_qc_thresholds(
            cfg.accept_threshold,
            cfg.positive_prefix_headroom_ceiling,
        )
        self._cache_batch_size = max(
            1,
            int(8 if prefix_cache is None else prefix_cache.batch_size),
        )

        rng = random.Random(
            cfg.generation.master_seed
            + _CONCEPT_SEED_OFFSET
            + concept.concept_id
        )
        indices = list(range(len(seeds_train)))
        rng.shuffle(indices)
        clean: list[tuple[int, SeedRow, str, bool]] = []
        dirty: list[tuple[int, SeedRow, str, bool]] = []
        too_short = 0
        member_ids = (
            (concept.feature_index,)
            if cluster_target is None
            else cluster_target.allowed_member_ids
        )
        with profile_region(
            "stage1b.classify_seed_cleanliness",
            concept_id=concept.concept_id,
            rows=len(indices),
        ):
            for idx in indices:
                prefix_text = effective_prefixes[idx]
                if prefix_text is None:
                    too_short += 1
                    continue
                seed = seeds_train[idx]
                if any(
                    _seed_has_feature(seed, member_id)
                    for member_id in member_ids
                ):
                    dirty.append((idx, seed, prefix_text, False))
                else:
                    clean.append((idx, seed, prefix_text, True))

        self.pool_size = len(seeds_train)
        self.clean_count = len(clean)
        self.dirty_count = len(dirty)
        self.too_short = too_short
        self.clean_scored = 0
        self.dirty_scored = 0
        self.headroom_rejected = 0
        self._clean = clean
        self._dirty = dirty
        self._eligible_by_seed_idx = {
            seed_idx: (seed, prefix_text, stored_clean)
            for seed_idx, seed, prefix_text, stored_clean in (*clean, *dirty)
        }
        self._clean_cursor = 0
        self._dirty_scored_once = False
        self._buffer: deque[_SelectedPositiveSeed] = deque()
        self._next_candidate_rank = 0

        record_counter(
            "stage1b.clean_seed_candidates",
            self.clean_count,
            concept_id=concept.concept_id,
        )
        record_counter(
            "stage1b.dirty_seed_candidates",
            self.dirty_count,
            concept_id=concept.concept_id,
        )
        record_counter(
            "stage1b.short_seed_candidates",
            self.too_short,
            concept_id=concept.concept_id,
        )
        if not clean and not dirty:
            raise AxBenchBaseError(
                f"Concept {concept.concept_id}: Stage 1b has no eligible "
                "positive seed prefixes"
            )

    @property
    def scored_count(self) -> int:
        return self.clean_scored + self.dirty_scored

    @property
    def admissible_count(self) -> int:
        return self._next_candidate_rank

    @property
    def exhausted(self) -> bool:
        """Return whether every physical candidate has been scored and drawn."""
        return (
            self.physically_exhausted and not self._buffer
        )

    @property
    def physically_exhausted(self) -> bool:
        """Return whether every clean and dirty physical row was scored."""

        return (
            self._clean_cursor == len(self._clean)
            and self._dirty_scored_once
        )

    def _append_admissible(
        self,
        candidates: list[_SelectedPositiveSeed],
    ) -> None:
        for candidate in candidates:
            if (
                candidate.seed_percentile
                > self._thresholds.maximum_seed_percentile
            ):
                self.headroom_rejected += 1
                continue
            self._buffer.append(
                _SelectedPositiveSeed(
                    seed_idx=candidate.seed_idx,
                    seed=candidate.seed,
                    prefix_text=candidate.prefix_text,
                    seed_percentile=candidate.seed_percentile,
                    stored_clean=candidate.stored_clean,
                    candidate_rank=self._next_candidate_rank,
                )
            )
            self._next_candidate_rank += 1

    def _score_clean(self, width: int) -> None:
        batch = self._clean[
            self._clean_cursor : self._clean_cursor + width
        ]
        scored = _score_positive_seed_candidates(
            batch,
            self.concept,
            self.cfg,
            self._base_model,
            self._base_sae,
            self._percentile_table,
            sae_layer=self._sae_layer,
            prefix_cache=self._prefix_cache,
            cluster_target=self._cluster_target,
        )
        # Advance the durable cursor only after the entire scorer call
        # succeeds.  An exception must leave a snapshot-able source whose next
        # resume retries this exact batch rather than recording it as scored.
        with _defer_sigint_for_checkpoint_state():
            self._clean_cursor += len(batch)
            self.clean_scored += len(scored)
            self._append_admissible(scored)

    def _score_dirty_once(self) -> None:
        if self._dirty_scored_once:
            return
        with profile_region(
            "stage1b.score_dirty_prefixes",
            concept_id=self.concept.concept_id,
            rows=len(self._dirty),
            cache=self._prefix_cache is not None,
        ):
            scored = _score_positive_seed_candidates(
                self._dirty,
                self.concept,
                self.cfg,
                self._base_model,
                self._base_sae,
                self._percentile_table,
                sae_layer=self._sae_layer,
                prefix_cache=self._prefix_cache,
                cluster_target=self._cluster_target,
            )
        # Python's sort is stable, so shuffled order is the exact tie-break.
        scored.sort(key=lambda candidate: candidate.seed_percentile)
        # As with clean scoring, publish the durable completion flag only once
        # the whole deterministic scoring operation has succeeded.
        with _defer_sigint_for_checkpoint_state():
            self._dirty_scored_once = True
            self.dirty_scored = len(scored)
            self._append_admissible(scored)

    def _fill_buffer(
        self,
        count: int,
        *,
        preserve_initial_batching: bool,
    ) -> None:
        with profile_region(
            "stage1b.score_clean_prefixes",
            concept_id=self.concept.concept_id,
            rows=max(0, len(self._clean) - self._clean_cursor),
            cache=self._prefix_cache is not None,
        ):
            while (
                len(self._buffer) < count
                and self._clean_cursor < len(self._clean)
            ):
                remaining = count - len(self._buffer)
                width = (
                    min(self._cache_batch_size, remaining)
                    if preserve_initial_batching
                    else min(
                        self._cache_batch_size,
                        len(self._clean) - self._clean_cursor,
                    )
                )
                self._score_clean(width)
        if self._clean_cursor == len(self._clean) and (
            len(self._buffer) < count or not preserve_initial_batching
        ):
            self._score_dirty_once()

    def preflight_initial_target(self, target: int) -> None:
        """Build the target-independent prefix and prove ``target``."""

        # This call is deliberately identical to the Phase 1 target-72 path,
        # including rejection-driven short scorer calls and the dirty fallback.
        self._fill_buffer(
            POSITIVE_INITIAL_COMPAT_TARGET,
            preserve_initial_batching=True,
        )
        if (
            target > POSITIVE_INITIAL_COMPAT_TARGET
            and len(self._buffer) >= POSITIVE_INITIAL_COMPAT_TARGET
        ):
            # Growth uses full scorer-batch chunks and therefore may preserve
            # a surplus beyond the requested target in the source buffer.
            self._fill_buffer(target, preserve_initial_batching=False)
        self._record_diagnostics()
        self._require_target_capacity(
            target=target,
            already_owned=0,
            required_buffer=target,
        )

    def preflight_restored_target(
        self,
        *,
        target: int,
        already_owned: int,
    ) -> None:
        """Prove a restored target before growth becomes checkpoint authority."""

        if type(already_owned) is not int or not 0 <= already_owned <= target:
            raise AxBenchBaseError(
                "Stage 1b restored candidate preflight has invalid owned "
                f"count={already_owned!r} for target={target!r}"
            )
        required_buffer = target - already_owned
        self._fill_buffer(
            required_buffer,
            preserve_initial_batching=False,
        )
        self._record_diagnostics()
        self._require_target_capacity(
            target=target,
            already_owned=already_owned,
            required_buffer=required_buffer,
        )

    def _require_target_capacity(
        self,
        *,
        target: int,
        already_owned: int,
        required_buffer: int,
    ) -> None:
        """Fail before provider access when physical candidates cannot suffice."""

        if len(self._buffer) < required_buffer:
            self.record_selection_counters()
            cluster_fingerprint = (
                None
                if self._cluster_target is None
                else self._cluster_target.feature_contract_fingerprint
            )
            raise AxBenchBaseError(
                "Stage 1b positive seed selection found too few admissible "
                "prefixes: "
                f"concept_id={self.concept.concept_id}, "
                f"feature_id={self.concept.feature_index}, required={target}, "
                f"selected={already_owned + len(self._buffer)}, "
                f"pool={self.pool_size}, "
                f"clean={self.clean_count}, dirty={self.dirty_count}, "
                f"too_short={self.too_short}, "
                f"clean_scored={self.clean_scored}, "
                f"dirty_scored={self.dirty_scored}, "
                f"headroom_rejected={self.headroom_rejected}, "
                f"accept_threshold={self._thresholds.accept_fraction}, "
                f"required_delta={self._thresholds.required_delta}, "
                "positive_prefix_headroom_ceiling="
                f"{self._thresholds.headroom_ceiling_fraction}, "
                f"completion_ceiling={self._thresholds.completion_ceiling}, "
                "maximum_seed_percentile="
                f"{self._thresholds.maximum_seed_percentile}, "
                f"cluster_fingerprint={cluster_fingerprint!r}"
            )

    def prepare_next(self) -> _SelectedPositiveSeed | None:
        """Score as needed, then expose the next candidate without consuming it."""

        if not self._buffer:
            self._fill_buffer(1, preserve_initial_batching=False)
        if not self._buffer:
            return None
        return self._buffer[0]

    def commit_prepared(self, selected: _SelectedPositiveSeed) -> None:
        """Consume a candidate previously returned by :meth:`prepare_next`."""

        if not self._buffer or self._buffer[0] is not selected:
            raise AxBenchBaseError(
                "Stage 1b candidate source cannot commit a stale prepared "
                f"candidate: concept_id={self.concept.concept_id}, "
                f"seed_idx={selected.seed_idx}, "
                f"candidate_rank={selected.candidate_rank}"
            )
        self._buffer.popleft()

    def selected_from_checkpoint(
        self,
        descriptor: dict[str, Any],
        *,
        context: str,
    ) -> _SelectedPositiveSeed:
        """Rebind a compact descriptor to the current immutable SeedRow."""

        seed_idx = int(descriptor["seed_idx"])
        partition = self._eligible_by_seed_idx.get(seed_idx)
        if partition is None:
            raise AxBenchBaseError(
                f"{context}: seed_idx={seed_idx} is not an eligible current "
                f"seed for concept_id={self.concept.concept_id}"
            )
        seed, current_prefix, stored_clean = partition
        if descriptor["prefix_text"] != current_prefix:
            raise AxBenchBaseError(
                f"{context}: persisted prefix differs from the current bounded "
                f"prefix for concept_id={self.concept.concept_id}, "
                f"seed_idx={seed_idx}"
            )
        if descriptor["stored_clean"] is not stored_clean:
            raise AxBenchBaseError(
                f"{context}: persisted clean/dirty partition differs for "
                f"concept_id={self.concept.concept_id}, seed_idx={seed_idx}"
            )
        seed_percentile = _validate_raw_positive_percentile(
            descriptor["seed_percentile"],
            scorer="checkpoint",
            concept=self.concept,
            seed_idx=seed_idx,
        )
        return _SelectedPositiveSeed(
            seed_idx=seed_idx,
            seed=seed,
            prefix_text=current_prefix,
            seed_percentile=seed_percentile,
            stored_clean=stored_clean,
            candidate_rank=int(descriptor["candidate_rank"]),
        )

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        context: str,
    ) -> None:
        """Restore counters and buffer without calling the prefix scorer."""

        clean_cursor = int(state["clean_cursor"])
        if not 0 <= clean_cursor <= len(self._clean):
            raise AxBenchBaseError(
                f"{context}: clean_cursor={clean_cursor} is outside "
                f"[0, {len(self._clean)}]"
            )
        dirty_scored_once = bool(state["dirty_scored_once"])
        clean_scored = int(state["clean_scored"])
        dirty_scored = int(state["dirty_scored"])
        if clean_scored != clean_cursor:
            raise AxBenchBaseError(
                f"{context}: clean_scored={clean_scored} does not equal "
                f"clean_cursor={clean_cursor}"
            )
        if dirty_scored_once:
            if dirty_scored != len(self._dirty):
                raise AxBenchBaseError(
                    f"{context}: dirty scoring is complete but dirty_scored="
                    f"{dirty_scored}, expected {len(self._dirty)}"
                )
        elif dirty_scored != 0:
            raise AxBenchBaseError(
                f"{context}: dirty_scored={dirty_scored} before dirty scoring"
            )
        self._clean_cursor = clean_cursor
        self._dirty_scored_once = dirty_scored_once
        self.clean_scored = clean_scored
        self.dirty_scored = dirty_scored
        self.headroom_rejected = int(state["headroom_rejected"])
        self._next_candidate_rank = int(state["next_candidate_rank"])
        persisted_exhausted = bool(state["exhausted"])
        self._buffer = deque(
            self.selected_from_checkpoint(
                descriptor,
                context=f"{context}.buffered_candidates[{index}]",
            )
            for index, descriptor in enumerate(state["buffered_candidates"])
        )
        if persisted_exhausted != self.exhausted:
            raise AxBenchBaseError(
                f"{context}: persisted exhausted={persisted_exhausted} does "
                f"not match the reconstructed source state {self.exhausted}"
            )

    def _record_diagnostics(self) -> None:
        logger.info(
            "Concept {} positive candidate source: pool={}, clean={}, dirty={}, "
            "too_short={}, scored={}, headroom_rejected={}, admissible={}, "
            "initial_target={}, required_delta={:.6g}, "
            "completion_ceiling={:.6g}, maximum_seed_percentile={:.6g}",
            self.concept.concept_id,
            self.pool_size,
            self.clean_count,
            self.dirty_count,
            self.too_short,
            self.scored_count,
            self.headroom_rejected,
            self.admissible_count,
            self.cfg.n_positives_per_concept,
            self._thresholds.required_delta,
            self._thresholds.completion_ceiling,
            self._thresholds.maximum_seed_percentile,
        )

    def record_selection_counters(self) -> None:
        """Emit final legacy selection counters exactly once per run path."""
        record_counter(
            "stage1b.headroom_rejected_seeds",
            self.headroom_rejected,
            concept_id=self.concept.concept_id,
        )
        record_counter(
            "stage1b.prefix_candidates_scored",
            self.scored_count,
            concept_id=self.concept.concept_id,
        )
        if self.dirty_scored == 0:
            record_counter(
                "stage1b.dirty_seed_scoring_skipped",
                self.dirty_count,
                concept_id=self.concept.concept_id,
            )


@dataclass
class _PositiveTargetTracker:
    """Invariant-checking complete-pair target state for one concept."""

    concept: Concept
    source: _PositiveCandidateSource
    target: int
    zero_hit_bail_after: int | None
    cluster_target: ClusterFeatureTarget | None
    started_at: float
    seeds_drawn: int = 0
    discarded_rich: int = 0
    discarded_avoid: int = 0
    semantic_rich_attempts: int = 0
    semantic_avoid_attempts: int = 0
    natural_avoid_acceptances: int = 0
    termination_reason: str | None = None
    accepted_items: list[_PositiveWorkItem] = field(default_factory=list)
    discard_records: list[_PositiveDiscardRecord] = field(default_factory=list)
    outstanding: dict[int, _PositiveWorkItem] = field(default_factory=dict)
    admitted_seed_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._assert_invariants(allow_refill_deficit=True)

    @property
    def accepted(self) -> int:
        return len(self.accepted_items)

    @property
    def deficit(self) -> int:
        return self.target - self.accepted - len(self.outstanding)

    def _guard_blocks_admission(self) -> bool:
        # This is deliberately a zero-hit guard, not a lifetime spend cap.
        # Once one atomic pair proves the concept is viable, the finite,
        # one-pass candidate source remains the deterministic upper bound.
        return (
            self.accepted == 0
            and self.zero_hit_bail_after is not None
            and self.seeds_drawn >= self.zero_hit_bail_after
        )

    def _assert_invariants(
        self,
        *,
        allow_refill_deficit: bool = False,
    ) -> None:
        if self.target < 1:
            raise AxBenchBaseError(
                "Stage 1b target tracker requires target >= 1"
            )
        if self.accepted + len(self.outstanding) > self.target:
            raise AxBenchBaseError(
                "Stage 1b target tracker exceeded its complete-pair target: "
                f"concept_id={self.concept.concept_id}, "
                f"accepted={self.accepted}, "
                f"outstanding={len(self.outstanding)}, target={self.target}"
            )
        outstanding_ids = set(self.outstanding)
        if not outstanding_ids <= self.admitted_seed_indices:
            raise AxBenchBaseError(
                "Stage 1b target tracker has an outstanding unadmitted seed"
            )
        accepted_ids = {
            item.pending.seed_idx for item in self.accepted_items
        }
        if len(accepted_ids) != len(self.accepted_items):
            raise AxBenchBaseError(
                "Stage 1b target tracker contains duplicate accepted seeds"
            )
        if not accepted_ids <= self.admitted_seed_indices:
            raise AxBenchBaseError(
                "Stage 1b target tracker has an accepted unadmitted seed"
            )
        if accepted_ids & outstanding_ids:
            raise AxBenchBaseError(
                "Stage 1b target tracker has a seed both accepted and "
                "outstanding"
            )
        discarded_ids = {
            record.selected.seed_idx for record in self.discard_records
        }
        if len(discarded_ids) != len(self.discard_records):
            raise AxBenchBaseError(
                "Stage 1b target tracker contains duplicate discarded seeds"
            )
        if not discarded_ids <= self.admitted_seed_indices:
            raise AxBenchBaseError(
                "Stage 1b target tracker has a discarded unadmitted seed"
            )
        if (
            discarded_ids & accepted_ids
            or discarded_ids & outstanding_ids
        ):
            raise AxBenchBaseError(
                "Stage 1b target tracker has overlapping accepted, discarded, "
                "or outstanding seeds"
            )
        owned_ids = accepted_ids | discarded_ids | outstanding_ids
        if owned_ids != self.admitted_seed_indices:
            raise AxBenchBaseError(
                "Stage 1b target tracker admitted-seed ownership is "
                f"inconsistent: stored={sorted(self.admitted_seed_indices)}, "
                f"derived={sorted(owned_ids)}"
            )
        if self.seeds_drawn != len(self.admitted_seed_indices):
            raise AxBenchBaseError(
                "Stage 1b target tracker seed-draw count is inconsistent: "
                f"seeds_drawn={self.seeds_drawn}, "
                f"unique={len(self.admitted_seed_indices)}"
            )
        if (
            self.deficit > 0
            and not allow_refill_deficit
            and not self._guard_blocks_admission()
            and self.termination_reason not in {"pool_exhausted", "zero_hit_guard"}
        ):
            raise AxBenchBaseError(
                "Stage 1b target tracker has an unexplained target deficit: "
                f"concept_id={self.concept.concept_id}, deficit={self.deficit}"
            )

    def admit_next(
        self,
        *,
        order: int,
        on_committed: Callable[[_PositiveWorkItem, int], None] | None = None,
    ) -> _PositiveWorkItem | None:
        if self.deficit <= 0:
            self._assert_invariants()
            return None
        if self._guard_blocks_admission():
            if not self.outstanding:
                self.termination_reason = "zero_hit_guard"
            self._assert_invariants()
            return None
        selected = self.source.prepare_next()
        if selected is None:
            self.termination_reason = "pool_exhausted"
            self._assert_invariants()
            return None
        if selected.seed_idx in self.admitted_seed_indices:
            raise AxBenchBaseError(
                "Stage 1b candidate source attempted duplicate seed admission: "
                f"concept_id={self.concept.concept_id}, "
                f"seed_idx={selected.seed_idx}"
            )
        with _defer_sigint_for_checkpoint_state():
            self.source.commit_prepared(selected)
            work = _PositiveWorkItem(
                concept=self.concept,
                pending=_PositivePending(order=order, selected=selected),
                cluster_target=self.cluster_target,
            )
            self.admitted_seed_indices.add(selected.seed_idx)
            self.outstanding[selected.seed_idx] = work
            self.seeds_drawn += 1
            if on_committed is not None:
                # Scheduler-local chain ownership is part of the same durable
                # admission commit.  Publishing it after SIGINT was unblocked
                # left a tiny interval where the tracker owned ``order`` but
                # the emergency snapshot still saw ``next_chain_index ==
                # order`` and therefore could not serialize the outstanding
                # descriptor.
                on_committed(work, order + 1)
        self._assert_invariants(allow_refill_deficit=True)
        return work

    def complete_success(
        self,
        work: _PositiveWorkItem,
        *,
        natural_avoid: bool,
        acceptance_sequence: int,
    ) -> None:
        if work.rich_row is None or work.avoiding_row is None:
            raise AxBenchBaseError(
                "Stage 1b target tracker cannot accept an incomplete pair"
            )
        if (
            isinstance(acceptance_sequence, bool)
            or not isinstance(acceptance_sequence, int)
            or acceptance_sequence < 0
        ):
            raise AxBenchBaseError(
                "Stage 1b acceptance sequence must be a non-negative int"
            )
        if work.acceptance_sequence is not None:
            raise AxBenchBaseError(
                "Stage 1b target tracker received a duplicate or stale "
                "completion and cannot resequence its accepted pair"
            )
        self._finish_outstanding(work)
        work.acceptance_sequence = acceptance_sequence
        self.accepted_items.append(work)
        if natural_avoid:
            self.natural_avoid_acceptances += 1
        # The first success permanently lifts the zero-hit guard.
        if self.termination_reason == "zero_hit_guard":
            self.termination_reason = None
        self._assert_invariants(allow_refill_deficit=True)

    def complete_failure(
        self,
        work: _PositiveWorkItem,
        *,
        side: str,
    ) -> None:
        if side not in {"rich", "avoid"}:
            raise AxBenchBaseError(
                f"Stage 1b target tracker received unknown failure side {side!r}"
            )
        self._finish_outstanding(work)
        self.discard_records.append(
            _PositiveDiscardRecord(
                selected=work.pending.selected,
                side=side,
                rich_semantic_attempts=work.rich_semantic_attempts,
                avoid_semantic_attempts=work.avoid_semantic_attempts,
            )
        )
        if side == "rich":
            self.discarded_rich += 1
        else:
            self.discarded_avoid += 1
        if self._guard_blocks_admission() and not self.outstanding:
            self.termination_reason = "zero_hit_guard"
        self._assert_invariants(allow_refill_deficit=True)

    def _finish_outstanding(self, work: _PositiveWorkItem) -> None:
        seed_idx = work.pending.seed_idx
        if work.terminal or self.outstanding.get(seed_idx) is not work:
            raise AxBenchBaseError(
                "Stage 1b target tracker received a duplicate or stale "
                f"completion: concept_id={self.concept.concept_id}, "
                f"seed_idx={seed_idx}"
            )
        del self.outstanding[seed_idx]
        work.terminal = True


def _admit_positive_work_items(
    tracker: _PositiveTargetTracker,
    *,
    start_order: int,
    on_admitted: Callable[[_PositiveWorkItem, int], None] | None = None,
) -> tuple[list[_PositiveWorkItem], int]:
    """Fill one target deficit through the production admission path."""
    works: list[_PositiveWorkItem] = []
    next_order = start_order
    while tracker.deficit > 0:
        work = tracker.admit_next(
            order=next_order,
            on_committed=on_admitted,
        )
        if work is None:
            break
        works.append(work)
        next_order += 1
    tracker._assert_invariants()
    return works, next_order


def _prepare_positive_target_tracker(
    cfg: AxBenchBaseConfig,
    concept: Concept,
    seeds_train: list[SeedRow],
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    *,
    sae_layer: int,
    prefix_cache: Any | None,
    cluster_target: ClusterFeatureTarget | None,
    effective_prefixes: _EffectivePrefixes,
) -> _PositiveTargetTracker:
    """Build and statically preflight one production target tracker."""
    source = _PositiveCandidateSource(
        concept,
        seeds_train,
        cfg,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        prefix_cache=prefix_cache,
        cluster_target=cluster_target,
        effective_prefixes=effective_prefixes,
    )
    source.preflight_initial_target(cfg.n_positives_per_concept)
    return _PositiveTargetTracker(
        concept=concept,
        source=source,
        target=cfg.n_positives_per_concept,
        zero_hit_bail_after=cfg.positive_zero_hit_bail_after,
        cluster_target=cluster_target,
        started_at=time.perf_counter(),
    )


def _synth_max_concurrent(cfg: AxBenchBaseConfig) -> int:
    return max(1, int(cfg.synth.max_concurrent or 1))


def _positive_rich_threshold_passes(
    cfg: AxBenchBaseConfig,
    selected: _SelectedPositiveSeed,
    output_percentile: float,
) -> bool:
    """Return the one canonical Stage 1b rich delta decision."""
    thresholds = derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    )
    return percentile_delta_meets_or_exceeds(
        output_percentile - selected.seed_percentile,
        thresholds.required_delta,
    )


def _parse_positive_rich_continuation(
    result: SynthResult,
    item: _PositivePending,
    request: Stage1ContinuationRequest,
    cfg: AxBenchBaseConfig,
    concept: Concept,
    tokenizer: Any,
    *,
    attempt: int,
) -> str | None:
    """Parse one authenticated rich response without mutating tracker state."""
    selected = item.selected
    if result.error is not None:
        logger.warning(
            f"Concept {concept.concept_id} seed {item.seed_idx} attempt "
            f"{attempt + 1}: synth LLM call failed: {result.error}"
        )
        return None
    assert result.response_text is not None
    with profile_region(
        "stage1b.parse_synth_response",
        concept_id=concept.concept_id,
        attempt=attempt,
    ):
        continuation = parse_stage1_continuation_or_none(
            prefix_text=selected.prefix_text,
            response=result.response_text,
            request=request,
            output_length=cfg.output_length,
            tokenizer=tokenizer,
            subject="Stage 1b concept_rich continuation",
            log_label=(f"Concept {concept.concept_id} seed {item.seed_idx} attempt {attempt + 1}"),
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": concept.concept_id,
                "attempt": attempt,
                "content_class": CONCEPT_RICH_LABEL,
            },
        )
    return continuation


def _apply_positive_rich_result(
    continuation: str,
    output_percentile: object,
    item: _PositivePending,
    cfg: AxBenchBaseConfig,
    concept: Concept,
    gen_source: str,
    *,
    attempt: int,
) -> None:
    """Apply a parsed, pre-scored rich result using the scalar row contract."""
    if item.done:
        return
    selected = item.selected
    out_pct = _validate_raw_positive_percentile(
        output_percentile,
        scorer="output scorer",
        concept=concept,
        seed_idx=selected.seed_idx,
    )
    seed_pct = selected.seed_percentile
    delta = out_pct - seed_pct
    row: dict[str, Any] = {
        "concept_id": concept.concept_id,
        "category": "positive",
        "content_class": CONCEPT_RICH_LABEL,
        "prefix_text": selected.prefix_text,
        "output_text": continuation,
        "output_concept": concept.description,
        "n_prefix_nanochat_tokens": _exact_output_start(selected.prefix_text, continuation),
        "gen_source": gen_source,
        # Compatibility provenance retained byte-for-byte from Stage 1b v2.
        # It is deterministic, but is not a globally unique row identifier;
        # consumers must use the row's explicit concept/content fields.
        "gen_seed": (
            cfg.generation.master_seed
            + _CONCEPT_SEED_OFFSET
            + concept.concept_id * 1000
            + item.seed_idx
        ),
        "qc_feature_id": concept.feature_index,
        "qc_seed_percentile": float(seed_pct),
        "qc_output_percentile": float(out_pct),
        "qc_delta": float(delta),
        "qc_attempts": attempt + 1,
        "qc_accepted": True,
        "qc_zero_activation_verified": False,
    }
    if _positive_rich_threshold_passes(cfg, selected, out_pct):
        record_counter(
            "stage1b.qc_accepts",
            1,
            concept_id=concept.concept_id,
            attempt=attempt,
        )
        record_counter(
            "stage1b.rich_qc_attempts_to_accept",
            attempt + 1,
            concept_id=concept.concept_id,
            seed_idx=item.seed_idx,
            outcome="accepted",
        )
        item.accepted_row = row
        return
    record_counter(
        "stage1b.qc_rejects",
        1,
        concept_id=concept.concept_id,
        attempt=attempt,
    )


def _avoiding_gen_seed(cfg: AxBenchBaseConfig, concept_id: int, seed_idx: int) -> int:
    """Deterministic provenance seed for a paired concept_avoiding row.

    Uses ``_POS_AVOID_SEED_OFFSET`` (distinct from the rich-row offset) so
    rich/avoid rows of the same seed get different ``gen_seed`` values, and
    re-runs are byte-identical. This compatibility field is not a globally
    unique row identifier.
    """
    return cfg.generation.master_seed + _POS_AVOID_SEED_OFFSET + concept_id * 1000 + seed_idx


def _make_positive_avoiding_row(
    cfg: AxBenchBaseConfig,
    concept: Concept,
    selected: _SelectedPositiveSeed,
    output_text: str,
    *,
    attempts: int,
    gen_source: str,
) -> dict[str, Any]:
    """Build a paired ``concept_avoiding`` row dict (positive-mode schema)."""
    seed_idx = selected.seed_idx
    return {
        "concept_id": concept.concept_id,
        "category": "negative",
        "content_class": CONCEPT_AVOIDING_LABEL,
        "prefix_text": selected.prefix_text,
        "output_text": output_text,
        "output_concept": concept.description,
        "n_prefix_nanochat_tokens": _exact_output_start(selected.prefix_text, output_text),
        "gen_source": gen_source,
        "gen_seed": _avoiding_gen_seed(cfg, concept.concept_id, seed_idx),
        "qc_feature_id": concept.feature_index,
        "qc_seed_percentile": float("nan"),
        "qc_output_percentile": float("nan"),
        "qc_delta": float("nan"),
        "qc_attempts": attempts,
        "qc_accepted": True,
        "qc_zero_activation_verified": True,
    }


def _apply_positive_natural_avoid_outcome(
    cfg: AxBenchBaseConfig,
    concept: Concept,
    selected: _SelectedPositiveSeed,
    outcome: AvoidContinuationOutcome,
) -> dict[str, Any] | None:
    """Apply a precomputed natural strict-zero outcome and build its row."""
    seed = selected.seed
    seed_idx = selected.seed_idx
    natural = seed.continuation_text
    if outcome is AvoidContinuationOutcome.INACTIVE:
        record_counter(
            "stage1b.avoid_natural_accepts",
            1,
            concept_id=concept.concept_id,
        )
        record_counter(
            "stage1b.avoid_qc_attempts_to_accept",
            0,
            concept_id=concept.concept_id,
            seed_idx=seed_idx,
            outcome="accepted_natural",
        )
        return _make_positive_avoiding_row(
            cfg,
            concept,
            selected,
            natural,
            attempts=0,
            gen_source=_doc_continuation_gen_source(seed),
        )
    if outcome is AvoidContinuationOutcome.ACTIVE:
        record_counter(
            "stage1b.avoid_natural_rejects",
            1,
            concept_id=concept.concept_id,
        )
    return None


def _positive_avoid_chain_request(
    cfg: AxBenchBaseConfig,
    avoiding_template: str,
    item: _PositiveAvoidPending,
    *,
    attempt: int,
) -> SynthRequest:
    work = item.work
    concept = work.concept
    selected = work.pending.selected
    continuation_request = _stage1b_continuation_request(
        avoiding_template,
        concept=concept,
        prefix_text=selected.prefix_text,
        seed_idx=selected.seed_idx,
        attempt=attempt,
        content_class=CONCEPT_AVOIDING_LABEL,
        cluster_target=work.cluster_target,
    )
    return SynthRequestWithPayload(
        index=item.order,
        prompt=continuation_request.prompt,
        cache_key=make_cache_key(
            model=cfg.synth.model,
            prompt=continuation_request.prompt,
            tag=(
                f"pos_avoid_c{concept.concept_id}"
                f"_s{selected.seed_idx}_a{attempt}"
                f"{_cluster_cache_suffix(work.cluster_target)}"
            ),
        ),
        attrs={
            "concept_id": concept.concept_id,
            "attempt": attempt,
            "seed_idx": selected.seed_idx,
            "content_class": CONCEPT_AVOIDING_LABEL,
        },
        payload=continuation_request,
    )


def _parse_positive_avoid_continuation(
    result: SynthResult,
    item: _PositiveAvoidPending,
    continuation_request: Stage1ContinuationRequest,
    cfg: AxBenchBaseConfig,
    tokenizer: Any,
    *,
    attempt: int,
) -> str | None:
    """Parse one authenticated LLM-avoid response without state mutation."""
    work = item.work
    concept = work.concept
    selected = work.pending.selected
    if result.error is not None:
        logger.warning(
            f"Concept {concept.concept_id} seed {selected.seed_idx} "
            f"avoid-synth attempt {attempt + 1}: synth LLM call failed: "
            f"{result.error}"
        )
        return None
    assert result.response_text is not None
    with profile_region(
        "stage1b.avoid_parse_synth_response",
        concept_id=concept.concept_id,
        attempt=attempt,
    ):
        continuation = parse_stage1_continuation_or_none(
            prefix_text=selected.prefix_text,
            response=result.response_text,
            request=continuation_request,
            output_length=cfg.output_length,
            tokenizer=tokenizer,
            subject="Stage 1b concept_avoiding continuation",
            log_label=(
                f"Concept {concept.concept_id} seed {selected.seed_idx} "
                f"avoid-synth attempt {attempt + 1}"
            ),
            protocol_counter="stage1b.continuation_protocol_failures",
            special_token_counter="stage1b.special_token_refusals",
            counter_attrs={
                "concept_id": concept.concept_id,
                "attempt": attempt,
                "content_class": CONCEPT_AVOIDING_LABEL,
            },
        )
    return continuation


def _apply_positive_avoid_outcome(
    item: _PositiveAvoidPending,
    continuation: str,
    outcome: AvoidContinuationOutcome,
    cfg: AxBenchBaseConfig,
    *,
    attempt: int,
) -> None:
    """Apply one parsed, precomputed LLM-avoid strict-zero outcome."""
    if item.done:
        return
    work = item.work
    concept = work.concept
    selected = work.pending.selected
    if outcome is AvoidContinuationOutcome.NO_TOKENS:
        record_counter(
            "stage1b.avoid_llm_empty_rejects",
            1,
            concept_id=concept.concept_id,
            attempt=attempt,
        )
        return
    if outcome is AvoidContinuationOutcome.INACTIVE:
        record_counter(
            "stage1b.avoid_llm_accepts",
            1,
            concept_id=concept.concept_id,
            attempt=attempt,
        )
        record_counter(
            "stage1b.avoid_qc_attempts_to_accept",
            attempt + 1,
            concept_id=concept.concept_id,
            seed_idx=selected.seed_idx,
            outcome="accepted_llm",
        )
        item.accepted_row = _make_positive_avoiding_row(
            cfg,
            concept,
            selected,
            continuation,
            attempts=attempt + 1,
            gen_source=f"openrouter:{cfg.synth.model}",
        )
        return
    if outcome is AvoidContinuationOutcome.ACTIVE:
        record_counter(
            "stage1b.avoid_llm_rejects",
            1,
            concept_id=concept.concept_id,
            attempt=attempt,
        )


def _prepare_positive_target_trackers(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    seeds_train: list[SeedRow],
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    target_model_meta: TargetModelMetadata,
    *,
    prefix_cache: Any | None,
    effective_prefixes: _EffectivePrefixes,
    cluster_contract: ClusterFeatureContract | None,
) -> list[_PositiveTargetTracker]:
    """Build and statically preflight every concept before provider access."""

    trackers: list[_PositiveTargetTracker] = []
    for concept in concepts:
        cluster_target = (
            None
            if cluster_contract is None
            else cluster_contract.target_for_representative(
                concept.feature_index
            )
        )
        logger.info(
            "Preparing Stage 1b target-refill source for concept {} "
            "(feature {}, description={!r})",
            concept.concept_id,
            concept.feature_index,
            concept.description,
        )
        try:
            trackers.append(
                _prepare_positive_target_tracker(
                    cfg,
                    concept,
                    seeds_train,
                    base_model,
                    base_sae,
                    percentile_table,
                    sae_layer=target_model_meta.sae_layer,
                    prefix_cache=prefix_cache,
                    cluster_target=cluster_target,
                    effective_prefixes=effective_prefixes,
                )
            )
        except Exception:
            # A later concept can fail static preflight before the scheduler
            # exists. Preserve final selection diagnostics for every earlier
            # concept that was already prepared.
            for prepared_tracker in trackers:
                prepared_tracker.source.record_selection_counters()
            raise
        finally:
            release_cuda_iteration_memory()
    return trackers


def _selected_checkpoint_descriptor(
    selected: _SelectedPositiveSeed,
) -> dict[str, Any]:
    return {
        "seed_idx": selected.seed_idx,
        "candidate_rank": selected.candidate_rank,
        "prefix_text": selected.prefix_text,
        "seed_percentile": float(selected.seed_percentile),
        "stored_clean": selected.stored_clean,
    }


def _accepted_work_uses_natural_avoid(work: _PositiveWorkItem) -> bool:
    if work.avoiding_row is None:
        raise AxBenchBaseError(
            "Stage 1b cannot classify an accepted item without its avoiding row"
        )
    return str(work.avoiding_row.get("gen_source", "")).startswith(
        DOC_CONTINUATION_GEN_SOURCE_PREFIX
    )


def _validate_checkpoint_ordinary_rows(
    cfg: AxBenchBaseConfig,
    rows: list[dict[str, Any]],
) -> None:
    """Run the expensive ordinary-row checks once per immutable pair."""

    _validate_emitted_positive_rows(cfg, rows)
    validate_positive_qc_rows(
        rows,
        accept_threshold_points=derive_positive_qc_thresholds(
            cfg.accept_threshold,
            cfg.positive_prefix_headroom_ceiling,
        ).required_delta,
    )


def _checkpoint_context(
    tracker: _PositiveTargetTracker,
    selected: _SelectedPositiveSeed | None = None,
) -> str:
    base = (
        f"concept_id={tracker.concept.concept_id}, "
        f"feature_id={tracker.concept.feature_index}"
    )
    if selected is None:
        return base
    return (
        f"{base}, seed_idx={selected.seed_idx}, "
        f"candidate_rank={selected.candidate_rank}"
    )


def _stage1b_checkpoint_snapshot(
    cfg: AxBenchBaseConfig,
    trackers: list[_PositiveTargetTracker],
    *,
    fingerprint: dict[str, Any],
    highest_requested_target: int,
    next_chain_index: int,
    next_acceptance_sequence: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate live scheduler state and build its canonical transaction."""

    if (
        type(highest_requested_target) is not int
        or highest_requested_target < 1
    ):
        raise AxBenchBaseError(
            "Stage 1b checkpoint has invalid highest_requested_target="
            f"{highest_requested_target!r}"
        )

    if (
        isinstance(next_chain_index, bool)
        or not isinstance(next_chain_index, int)
        or next_chain_index < 0
    ):
        raise AxBenchBaseError(
            f"Stage 1b checkpoint has invalid next_chain_index={next_chain_index!r}"
        )
    if (
        isinstance(next_acceptance_sequence, bool)
        or not isinstance(next_acceptance_sequence, int)
        or next_acceptance_sequence < 0
    ):
        raise AxBenchBaseError(
            "Stage 1b checkpoint has invalid next_acceptance_sequence="
            f"{next_acceptance_sequence!r}"
        )
    concept_ids = [tracker.concept.concept_id for tracker in trackers]
    if len(concept_ids) != len(set(concept_ids)):
        raise AxBenchBaseError(
            "Stage 1b checkpoint tracker order contains duplicate concept IDs: "
            f"{concept_ids!r}"
        )

    concept_states: list[dict[str, Any]] = []
    accepted_by_sequence: dict[int, tuple[_PositiveTargetTracker, _PositiveWorkItem]] = {}
    for tracker in trackers:
        tracker._assert_invariants(allow_refill_deficit=True)
        context = _checkpoint_context(tracker)
        source = tracker.source
        if source.clean_scored != source._clean_cursor:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint source mismatch ({context}): "
                f"clean_scored={source.clean_scored}, "
                f"clean_cursor={source._clean_cursor}"
            )
        if source._dirty_scored_once:
            if source.dirty_scored != len(source._dirty):
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint source mismatch ({context}): dirty "
                    f"scoring completed with {source.dirty_scored}/"
                    f"{len(source._dirty)} rows"
                )
        elif source.dirty_scored != 0:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint source mismatch ({context}): "
                f"dirty_scored={source.dirty_scored} before dirty scoring"
            )
        if source.headroom_rejected != (
            source.scored_count - source.admissible_count
        ):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint source mismatch ({context}): "
                f"headroom_rejected={source.headroom_rejected}, scored="
                f"{source.scored_count}, next_candidate_rank="
                f"{source.admissible_count}"
            )

        buffered = list(source._buffer)
        accepted = sorted(
            tracker.accepted_items,
            key=lambda work: (
                -1
                if work.acceptance_sequence is None
                else work.acceptance_sequence
            ),
        )
        discarded = sorted(
            tracker.discard_records,
            key=lambda record: record.selected.candidate_rank,
        )
        outstanding = sorted(
            tracker.outstanding.values(), key=lambda work: work.order
        )
        collections = {
            "buffered": [candidate.candidate_rank for candidate in buffered],
            "accepted": [
                work.pending.selected.candidate_rank for work in accepted
            ],
            "discarded": [
                record.selected.candidate_rank for record in discarded
            ],
            "outstanding": [
                work.pending.selected.candidate_rank for work in outstanding
            ],
        }
        all_ranks: list[int] = []
        for name, ranks in collections.items():
            if len(ranks) != len(set(ranks)):
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint contains duplicate {name} candidate "
                    f"ranks ({context}): {ranks!r}"
                )
            all_ranks.extend(ranks)
        if len(all_ranks) != len(set(all_ranks)):
            raise AxBenchBaseError(
                "Stage 1b checkpoint candidate ownership sets overlap "
                f"({context}): {collections!r}"
            )
        expected_ranks = set(range(source.admissible_count))
        actual_ranks = set(all_ranks)
        if actual_ranks != expected_ranks:
            raise AxBenchBaseError(
                "Stage 1b checkpoint candidate ranks do not cover the exact "
                f"admissible prefix ({context}): missing="
                f"{sorted(expected_ranks - actual_ranks)}, unexpected="
                f"{sorted(actual_ranks - expected_ranks)}"
            )

        rich_discards = sum(record.side == "rich" for record in discarded)
        avoid_discards = sum(record.side == "avoid" for record in discarded)
        if (
            rich_discards != tracker.discarded_rich
            or avoid_discards != tracker.discarded_avoid
        ):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint discard counters mismatch ({context})"
            )

        terminal_rich_attempts = sum(
            work.rich_semantic_attempts for work in accepted
        ) + sum(record.rich_semantic_attempts for record in discarded)
        terminal_avoid_attempts = sum(
            work.avoid_semantic_attempts for work in accepted
        ) + sum(record.avoid_semantic_attempts for record in discarded)
        live_rich_attempts = terminal_rich_attempts + sum(
            work.rich_semantic_attempts for work in outstanding
        )
        live_avoid_attempts = terminal_avoid_attempts + sum(
            work.avoid_semantic_attempts for work in outstanding
        )
        if tracker.semantic_rich_attempts != live_rich_attempts or (
            tracker.semantic_avoid_attempts != live_avoid_attempts
        ):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint semantic counters mismatch ({context}): "
                f"tracker=({tracker.semantic_rich_attempts}, "
                f"{tracker.semantic_avoid_attempts}), derived="
                f"({live_rich_attempts}, {live_avoid_attempts})"
            )
        natural_acceptances = sum(
            _accepted_work_uses_natural_avoid(work) for work in accepted
        )
        if natural_acceptances != tracker.natural_avoid_acceptances:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint natural-avoid counter mismatch ({context})"
            )

        for work in accepted:
            selected = work.pending.selected
            item_context = _checkpoint_context(tracker, selected)
            if (
                work.acceptance_sequence is None
                or work.acceptance_sequence in accepted_by_sequence
            ):
                raise AxBenchBaseError(
                    "Stage 1b checkpoint accepted pair has missing/duplicate "
                    f"acceptance sequence ({item_context}): "
                    f"{work.acceptance_sequence!r}"
                )
            if work.rich_row is None or work.avoiding_row is None:
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint cannot persist a half-pair ({item_context})"
                )
            if work.rich_semantic_attempts != work.rich_row.get("qc_attempts"):
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint rich-attempt mismatch ({item_context})"
                )
            if work.avoid_semantic_attempts != work.avoiding_row.get(
                "qc_attempts"
            ):
                raise AxBenchBaseError(
                    f"Stage 1b checkpoint avoid-attempt mismatch ({item_context})"
                )
            accepted_by_sequence[work.acceptance_sequence] = (tracker, work)
        for record in discarded:
            if record.side not in {"rich", "avoid"}:
                raise AxBenchBaseError(
                    "Stage 1b checkpoint has invalid discard side "
                    f"{record.side!r} ({_checkpoint_context(tracker, record.selected)})"
                )
        for work in outstanding:
            if work.terminal:
                raise AxBenchBaseError(
                    "Stage 1b checkpoint marks an outstanding chain terminal "
                    f"({_checkpoint_context(tracker, work.pending.selected)})"
                )
            if work.acceptance_sequence is not None:
                raise AxBenchBaseError(
                    "Stage 1b checkpoint gives an outstanding chain parquet "
                    f"state ({_checkpoint_context(tracker, work.pending.selected)})"
                )
            if next_chain_index <= work.order:
                raise AxBenchBaseError(
                    "Stage 1b checkpoint next_chain_index does not exceed "
                    f"outstanding chain order {work.order} ({context})"
                )

        if tracker.termination_reason == "pool_exhausted" and not source.exhausted:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint declares pool exhaustion prematurely ({context})"
            )
        if tracker.termination_reason == "zero_hit_guard" and not (
            tracker.accepted == 0
            and tracker.zero_hit_bail_after is not None
            and tracker.seeds_drawn >= tracker.zero_hit_bail_after
            and not tracker.outstanding
        ):
            raise AxBenchBaseError(
                f"Stage 1b checkpoint has inconsistent zero-hit termination ({context})"
            )
        if tracker.termination_reason not in {
            None,
            "pool_exhausted",
            "zero_hit_guard",
        }:
            raise AxBenchBaseError(
                f"Stage 1b checkpoint has invalid termination reason "
                f"{tracker.termination_reason!r} ({context})"
            )

        concept_states.append(
            {
                "concept_id": tracker.concept.concept_id,
                "feature_id": tracker.concept.feature_index,
                "source": {
                    "clean_cursor": source._clean_cursor,
                    "dirty_scored_once": source._dirty_scored_once,
                    "clean_scored": source.clean_scored,
                    "dirty_scored": source.dirty_scored,
                    "headroom_rejected": source.headroom_rejected,
                    "next_candidate_rank": source.admissible_count,
                    "exhausted": source.exhausted,
                    "buffered_candidates": [
                        _selected_checkpoint_descriptor(candidate)
                        for candidate in buffered
                    ],
                },
                "tracker": {
                    "seeds_drawn": tracker.seeds_drawn,
                    "discarded_rich": rich_discards,
                    "discarded_avoid": avoid_discards,
                    "semantic_rich_attempts": terminal_rich_attempts,
                    "semantic_avoid_attempts": terminal_avoid_attempts,
                    "natural_avoid_acceptances": natural_acceptances,
                    "termination_reason": tracker.termination_reason,
                    "accepted": [
                        {
                            **_selected_checkpoint_descriptor(
                                work.pending.selected
                            ),
                            "acceptance_sequence": work.acceptance_sequence,
                            "natural_avoid": (
                                _accepted_work_uses_natural_avoid(work)
                            ),
                        }
                        for work in accepted
                    ],
                    "discarded": [
                        {
                            **_selected_checkpoint_descriptor(record.selected),
                            "side": record.side,
                            "rich_semantic_attempts": (
                                record.rich_semantic_attempts
                            ),
                            "avoid_semantic_attempts": (
                                record.avoid_semantic_attempts
                            ),
                        }
                        for record in discarded
                    ],
                    "outstanding": [
                        {
                            **_selected_checkpoint_descriptor(
                                work.pending.selected
                            ),
                            "chain_order": work.order,
                        }
                        for work in outstanding
                    ],
                },
            }
        )

    accepted_pair_count = len(accepted_by_sequence)
    expected_sequences = set(range(accepted_pair_count))
    actual_sequences = set(accepted_by_sequence)
    if actual_sequences != expected_sequences:
        raise AxBenchBaseError(
            "Stage 1b checkpoint global acceptance sequences are not "
            f"contiguous: missing={sorted(expected_sequences - actual_sequences)}, "
            f"unexpected={sorted(actual_sequences - expected_sequences)}"
        )
    if next_acceptance_sequence != accepted_pair_count:
        raise AxBenchBaseError(
            "Stage 1b checkpoint next_acceptance_sequence mismatch: "
            f"{next_acceptance_sequence} != {accepted_pair_count}"
        )

    checkpoint_rows: list[dict[str, Any]] = []
    for sequence in range(accepted_pair_count):
        tracker, work = accepted_by_sequence[sequence]
        selected = work.pending.selected
        rich_row = work.rich_row
        avoiding_row = work.avoiding_row
        if rich_row is None or avoiding_row is None:
            raise AxBenchBaseError(
                "Stage 1b checkpoint accepted pair became incomplete during "
                f"snapshot ({_checkpoint_context(tracker, selected)})"
            )
        for row in (rich_row, avoiding_row):
            ordinary = dict(row)
            checkpoint_rows.append(
                {
                    **ordinary,
                    "checkpoint_seed_idx": selected.seed_idx,
                    "checkpoint_candidate_rank": selected.candidate_rank,
                    "checkpoint_acceptance_sequence": sequence,
                }
            )
        if tracker.concept.concept_id != work.concept.concept_id:
            raise AxBenchBaseError(
                "Stage 1b checkpoint accepted work belongs to the wrong tracker"
            )
    document = {
        "schema_version": positive_checkpoint.CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "highest_requested_target": highest_requested_target,
        "accepted_pair_count": accepted_pair_count,
        "next_chain_index": next_chain_index,
        "next_acceptance_sequence": next_acceptance_sequence,
        "concept_states": concept_states,
    }
    return document, checkpoint_rows


def _write_stage1b_checkpoint(
    cfg: AxBenchBaseConfig,
    trackers: list[_PositiveTargetTracker],
    *,
    fingerprint: dict[str, Any],
    highest_requested_target: int,
    next_chain_index: int,
    next_acceptance_sequence: int,
    json_path: Path,
    parquet_path: Path,
) -> None:
    document, checkpoint_rows = _stage1b_checkpoint_snapshot(
        cfg,
        trackers,
        fingerprint=fingerprint,
        highest_requested_target=highest_requested_target,
        next_chain_index=next_chain_index,
        next_acceptance_sequence=next_acceptance_sequence,
    )
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=checkpoint_rows,
        training_schema=TRAINING_SCHEMA,
    )


def _restore_positive_target_trackers(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    seeds_train: list[SeedRow],
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    target_model_meta: TargetModelMetadata,
    *,
    prefix_cache: Any | None,
    effective_prefixes: _EffectivePrefixes,
    cluster_contract: ClusterFeatureContract | None,
    checkpoint_document: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
) -> list[_PositiveTargetTracker]:
    """Rebind persisted descriptors without performing prefix scoring."""

    states = checkpoint_document["concept_states"]
    if len(states) != len(concepts):
        raise AxBenchBaseError(
            "Stage 1b checkpoint concept population length mismatch: "
            f"stored={len(states)}, current={len(concepts)}"
        )
    ordinary = positive_checkpoint.ordinary_rows(checkpoint_rows)
    _validate_checkpoint_ordinary_rows(cfg, ordinary)
    rows_by_sequence = {
        sequence: (ordinary[2 * sequence], ordinary[2 * sequence + 1])
        for sequence in range(int(checkpoint_document["accepted_pair_count"]))
    }
    trackers: list[_PositiveTargetTracker] = []
    for concept_index, (concept, state) in enumerate(
        zip(concepts, states, strict=True)
    ):
        context = (
            f"Stage 1b checkpoint concept_states[{concept_index}] at "
            f"concept_id={concept.concept_id}, feature_id={concept.feature_index}"
        )
        if state["concept_id"] != concept.concept_id or (
            state["feature_id"] != concept.feature_index
        ):
            raise AxBenchBaseError(
                f"{context} does not match current canonical concept identity: "
                f"stored=({state['concept_id']}, {state['feature_id']})"
            )
        cluster_target = (
            None
            if cluster_contract is None
            else cluster_contract.target_for_representative(
                concept.feature_index
            )
        )
        source = _PositiveCandidateSource(
            concept,
            seeds_train,
            cfg,
            base_model,
            base_sae,
            percentile_table,
            sae_layer=target_model_meta.sae_layer,
            prefix_cache=prefix_cache,
            cluster_target=cluster_target,
            effective_prefixes=effective_prefixes,
        )
        source.restore_checkpoint_state(
            state["source"], context=f"{context}.source"
        )
        tracker_state = state["tracker"]
        tracker = _PositiveTargetTracker(
            concept=concept,
            source=source,
            target=cfg.n_positives_per_concept,
            zero_hit_bail_after=cfg.positive_zero_hit_bail_after,
            cluster_target=cluster_target,
            started_at=time.perf_counter(),
        )
        for descriptor_index, descriptor in enumerate(
            tracker_state["accepted"]
        ):
            selected = source.selected_from_checkpoint(
                descriptor,
                context=f"{context}.tracker.accepted[{descriptor_index}]",
            )
            sequence = int(descriptor["acceptance_sequence"])
            rich_row, avoiding_row = rows_by_sequence[sequence]
            work = _PositiveWorkItem(
                concept=concept,
                pending=_PositivePending(order=-1, selected=selected),
                cluster_target=cluster_target,
                rich_row=dict(rich_row),
                avoiding_row=dict(avoiding_row),
                terminal=True,
                rich_semantic_attempts=int(rich_row["qc_attempts"]),
                avoid_semantic_attempts=int(avoiding_row["qc_attempts"]),
                acceptance_sequence=sequence,
            )
            if (
                _accepted_work_uses_natural_avoid(work)
                is not descriptor["natural_avoid"]
            ):
                raise AxBenchBaseError(
                    f"{context}.tracker.accepted[{descriptor_index}] has a "
                    "natural-avoid mismatch"
                )
            tracker.accepted_items.append(work)
        for descriptor_index, descriptor in enumerate(
            tracker_state["discarded"]
        ):
            selected = source.selected_from_checkpoint(
                descriptor,
                context=f"{context}.tracker.discarded[{descriptor_index}]",
            )
            tracker.discard_records.append(
                _PositiveDiscardRecord(
                    selected=selected,
                    side=str(descriptor["side"]),
                    rich_semantic_attempts=int(
                        descriptor["rich_semantic_attempts"]
                    ),
                    avoid_semantic_attempts=int(
                        descriptor["avoid_semantic_attempts"]
                    ),
                )
            )
        for descriptor_index, descriptor in enumerate(
            tracker_state["outstanding"]
        ):
            selected = source.selected_from_checkpoint(
                descriptor,
                context=f"{context}.tracker.outstanding[{descriptor_index}]",
            )
            work = _PositiveWorkItem(
                concept=concept,
                pending=_PositivePending(
                    order=int(descriptor["chain_order"]),
                    selected=selected,
                ),
                cluster_target=cluster_target,
            )
            tracker.outstanding[selected.seed_idx] = work
        tracker.admitted_seed_indices = {
            work.pending.seed_idx for work in tracker.accepted_items
        } | {
            record.selected.seed_idx for record in tracker.discard_records
        } | set(tracker.outstanding)
        tracker.seeds_drawn = int(tracker_state["seeds_drawn"])
        tracker.discarded_rich = int(tracker_state["discarded_rich"])
        tracker.discarded_avoid = int(tracker_state["discarded_avoid"])
        tracker.semantic_rich_attempts = int(
            tracker_state["semantic_rich_attempts"]
        )
        tracker.semantic_avoid_attempts = int(
            tracker_state["semantic_avoid_attempts"]
        )
        tracker.natural_avoid_acceptances = int(
            tracker_state["natural_avoid_acceptances"]
        )
        tracker.termination_reason = tracker_state["termination_reason"]
        tracker._assert_invariants(allow_refill_deficit=True)
        trackers.append(tracker)

    rebuilt_document, _rebuilt_rows = _stage1b_checkpoint_snapshot(
        cfg,
        trackers,
        fingerprint=checkpoint_document["fingerprint"],
        highest_requested_target=int(
            checkpoint_document["highest_requested_target"]
        ),
        next_chain_index=int(checkpoint_document["next_chain_index"]),
        next_acceptance_sequence=int(
            checkpoint_document["next_acceptance_sequence"]
        ),
    )
    if rebuilt_document != checkpoint_document:
        raise AxBenchBaseError(
            "Stage 1b checkpoint restored state does not round-trip to the "
            "committed JSON document; retain both partial files for diagnosis"
        )
    for tracker in trackers:
        tracker.source.preflight_restored_target(
            target=tracker.target,
            already_owned=tracker.accepted + len(tracker.outstanding),
        )
    return trackers


def _run_positive_target_refill_engine(
    cfg: AxBenchBaseConfig,
    trackers: list[_PositiveTargetTracker],
    base_model: Any,
    base_sae: Any,
    llm_client: SynthCompleter,
    percentile_table: Any,
    target_model_meta: TargetModelMetadata,
    *,
    checkpoint_fingerprint: dict[str, Any] | None = None,
    checkpoint_json_path: Path | None = None,
    checkpoint_parquet_path: Path | None = None,
    highest_requested_target: int | None = None,
    resume_next_chain_index: int = 0,
    resume_next_acceptance_sequence: int = 0,
) -> dict[int, list[dict[str, Any]]]:
    """Resolve rich/natural-avoid/LLM-avoid chains with lazy replacements."""

    if not trackers:
        return {}
    seen_concept_ids: set[int] = set()
    duplicate_concept_ids: set[int] = set()
    for tracker in trackers:
        concept_id = tracker.concept.concept_id
        if concept_id in seen_concept_ids:
            duplicate_concept_ids.add(concept_id)
        seen_concept_ids.add(concept_id)
    if duplicate_concept_ids:
        raise AxBenchBaseError(
            "Stage 1b target-refill scheduler received duplicate concept IDs: "
            f"{sorted(duplicate_concept_ids)}"
        )
    rich_template = load_template("synth_continue_with_concept.txt")
    avoiding_template = load_template(
        "synth_continue_avoiding_concept.txt"
    )
    tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
    sae_layer = target_model_meta.sae_layer
    gen_source = f"openrouter:{cfg.synth.model}"
    work_by_order: dict[int, _PositiveWorkItem] = {}
    terminal_chain_indices: set[int] = set()
    tracker_by_concept = {
        tracker.concept.concept_id: tracker for tracker in trackers
    }
    checkpoint_args = (
        checkpoint_fingerprint,
        checkpoint_json_path,
        checkpoint_parquet_path,
    )
    checkpoint_enabled = all(value is not None for value in checkpoint_args)
    if checkpoint_enabled != any(value is not None for value in checkpoint_args):
        raise AxBenchBaseError(
            "Stage 1b checkpoint integration requires fingerprint, JSON path, "
            "and parquet path together"
        )
    next_chain_index = resume_next_chain_index
    next_acceptance_sequence = resume_next_acceptance_sequence
    checkpoint_highest_requested_target = (
        cfg.n_positives_per_concept
        if highest_requested_target is None
        else highest_requested_target
    )
    max_concurrent = _synth_max_concurrent(cfg)
    released_serial_concepts: set[int] = set()

    def commit_checkpoint() -> None:
        if not checkpoint_enabled:
            return
        if (
            checkpoint_fingerprint is None
            or checkpoint_json_path is None
            or checkpoint_parquet_path is None
        ):
            raise AxBenchBaseError(
                "Stage 1b checkpoint arguments changed after validation"
            )
        _write_stage1b_checkpoint(
            cfg,
            trackers,
            fingerprint=checkpoint_fingerprint,
            highest_requested_target=checkpoint_highest_requested_target,
            next_chain_index=next_chain_index,
            next_acceptance_sequence=next_acceptance_sequence,
            json_path=checkpoint_json_path,
            parquet_path=checkpoint_parquet_path,
        )

    def admit_refills(
        tracker: _PositiveTargetTracker,
    ) -> list[_PositiveWorkItem]:
        nonlocal next_chain_index

        def publish_admission(
            _work: _PositiveWorkItem,
            published_next_order: int,
        ) -> None:
            nonlocal next_chain_index
            # Publish the next order immediately after each successful
            # admission.  If later admission or request construction fails,
            # every outstanding chain still lies below the value captured by
            # the emergency checkpoint.
            next_chain_index = published_next_order

        works, next_chain_index = _admit_positive_work_items(
            tracker,
            start_order=next_chain_index,
            on_admitted=publish_admission,
        )
        for work in works:
            order = work.pending.order
            if order in work_by_order or order in terminal_chain_indices:
                raise AxBenchBaseError(
                    "Stage 1b allocated a duplicate request-chain index "
                    f"{order}"
                )
            work_by_order[order] = work
        return works

    def rich_requests_for(
        works: list[_PositiveWorkItem],
    ) -> list[SynthRequest]:
        return [
            _positive_rich_chain_request(
                cfg,
                rich_template,
                work,
                attempt=0,
            )
            for work in works
        ]

    resumed_works = sorted(
        (
            work
            for tracker in trackers
            for work in tracker.outstanding.values()
        ),
        key=lambda work: work.order,
    )
    for work in resumed_works:
        if work.order in work_by_order:
            raise AxBenchBaseError(
                "Stage 1b checkpoint contains duplicate outstanding chain "
                f"order {work.order}"
            )
        if not 0 <= work.order < next_chain_index:
            raise AxBenchBaseError(
                "Stage 1b checkpoint outstanding chain order is outside the "
                f"committed range: order={work.order}, "
                f"next_chain_index={next_chain_index}"
            )
        work_by_order[work.order] = work
    max_attempts = cfg.positive_max_attempts_per_seed

    def shortfall_error(tracker: _PositiveTargetTracker) -> AxBenchBaseError:
        return AxBenchBaseError(
            "Stage 1b complete-pair target shortfall: "
            f"concept_id={tracker.concept.concept_id}, "
            f"feature_id={tracker.concept.feature_index}, "
            f"accepted={tracker.accepted}, target={tracker.target}, "
            f"reason={tracker.termination_reason}, "
            f"seeds_drawn={tracker.seeds_drawn}, "
            f"rich_discards={tracker.discarded_rich}, "
            f"avoid_discards={tracker.discarded_avoid}"
        )

    def terminal_context(
        work: _PositiveWorkItem,
        *,
        side: str,
        attempt: int,
    ) -> str:
        return (
            f"concept_id={work.concept.concept_id}, "
            f"feature_id={work.concept.feature_index}, "
            f"seed_idx={work.pending.seed_idx}, side={side}, "
            f"semantic_attempt={attempt + 1}"
        )

    def abort_transport(
        work: _PositiveWorkItem,
        error: BaseException,
        *,
        side: str,
        attempt: int,
    ) -> NoReturn:
        raise AxBenchBaseError(
            "Stage 1b provider transport exhausted; aborting without "
            "advancing the semantic attempt or drawing a replacement "
            f"({terminal_context(work, side=side, attempt=attempt)}): "
            f"{error}"
        ) from error

    def discard_exhausted_candidate(
        tracker: _PositiveTargetTracker,
        work: _PositiveWorkItem,
        *,
        side: str,
    ) -> None:
        record_counter(
            f"stage1b.{side}_qc_attempts_on_exhaustion",
            max_attempts,
            concept_id=work.concept.concept_id,
            seed_idx=work.pending.seed_idx,
            outcome="discarded",
        )
        with _defer_sigint_for_checkpoint_state():
            tracker.complete_failure(work, side=side)

    def commit_success(
        tracker: _PositiveTargetTracker,
        work: _PositiveWorkItem,
        *,
        avoiding_row: dict[str, Any],
        natural_avoid: bool,
    ) -> None:
        nonlocal next_acceptance_sequence
        if work.rich_row is None:
            raise AxBenchBaseError(
                "Stage 1b cannot commit success without a rich row"
            )
        _validate_checkpoint_ordinary_rows(
            cfg,
            [dict(work.rich_row), dict(avoiding_row)],
        )
        with _defer_sigint_for_checkpoint_state():
            work.avoiding_row = avoiding_row
            tracker.complete_success(
                work,
                natural_avoid=natural_avoid,
                acceptance_sequence=next_acceptance_sequence,
            )
            next_acceptance_sequence += 1

    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> SynthRequest | None:
        # Keep the scheduler's scalar callback contract as the
        # max_concurrent=1 path, while sharing one preflight/QC/application
        # state machine with the batched callback. A singleton plural scorer
        # bottoms out in the same one-item ragged extractor forward.
        return handle_results([(request, result)], scalar_qc=True)[0]

    def handle_results(
        pairs: list[tuple[SynthRequest, SynthResult]],
        *,
        scalar_qc: bool = False,
    ) -> list[SynthRequest | None]:
        """Prepare batched GPU QC, then apply results in stable chain order."""
        emit_progress_heartbeat()
        prepared: list[_PreparedPositiveBatchResult] = []
        seen_indices: set[int] = set()

        # Preflight the complete callback before parsing, scoring, or changing
        # any tracker-owned state. The scheduler supplies stable-index order,
        # but retain explicit positions so the aligned return contract remains
        # correct even under a direct test double.
        for position, (request, result) in enumerate(pairs):
            if request.index in seen_indices:
                raise AxBenchBaseError(
                    "Stage 1b received duplicate request chains in one batch: "
                    f"{request.index}"
                )
            seen_indices.add(request.index)
            if result.index != request.index:
                raise AxBenchBaseError(
                    "Stage 1b request/result index mismatch: "
                    f"request={request.index}, result={result.index}"
                )
            work = work_by_order.get(request.index)
            if work is None:
                if request.index in terminal_chain_indices:
                    raise AxBenchBaseError(
                        "Stage 1b received a duplicate result for terminal chain "
                        f"{request.index}"
                    )
                raise AxBenchBaseError(
                    "Stage 1b received a result for an unknown request chain "
                    f"{request.index}"
                )
            if work.terminal:
                raise AxBenchBaseError(
                    "Stage 1b received a duplicate result for terminal chain "
                    f"{request.index}"
                )
            if not isinstance(request, SynthRequestWithPayload) or not isinstance(
                request.payload,
                Stage1ContinuationRequest,
            ):
                raise AxBenchBaseError(
                    "Stage 1b request chain lost its continuation protocol payload"
                )
            raw_attempt = request.attrs.get("attempt")
            if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int):
                raise AxBenchBaseError(
                    "Stage 1b request chain has an invalid attempt attribute: "
                    f"{raw_attempt!r}"
                )
            phase = work.phase
            if phase == "rich":
                expected_attempt = work.rich_semantic_attempts
            elif phase == "avoid" and work.avoid_pending is not None:
                expected_attempt = work.avoid_semantic_attempts
            else:
                raise AxBenchBaseError(
                    "Stage 1b request chain entered an invalid phase "
                    f"{phase!r}"
                )
            if raw_attempt != expected_attempt:
                raise AxBenchBaseError(
                    "Stage 1b request-chain attempt is not sequential for its "
                    f"{phase} phase: chain={request.index}, "
                    f"expected={expected_attempt}, got={raw_attempt}"
                )
            prepared.append(
                _PreparedPositiveBatchResult(
                    position=position,
                    request=request,
                    result=result,
                    work=work,
                    phase=phase,
                    attempt=raw_attempt,
                )
            )

        # Stage 1b transport exhaustion is fatal. Scan every prepared result
        # before authenticating or scoring any sibling so the callback is
        # recoverable as one outstanding batch through the existing checkpoint.
        for item in prepared:
            if item.result.error is not None:
                abort_transport(
                    item.work,
                    item.result.error,
                    side=item.phase,
                    attempt=item.attempt,
                )

        rich_qc_inputs: list[ContinuationQcInput] = []
        rich_qc_items: list[_PreparedPositiveBatchResult] = []
        for item in prepared:
            work = item.work
            if item.phase == "rich":
                continuation = _parse_positive_rich_continuation(
                    item.result,
                    work.pending,
                    item.request.payload,
                    cfg,
                    work.concept,
                    tokenizer,
                    attempt=item.attempt,
                )
                item.continuation = continuation
                if continuation is not None:
                    rich_qc_inputs.append(
                        ContinuationQcInput(
                            prefix_text=work.pending.selected.prefix_text,
                            continuation_text=continuation,
                            feature_id=work.concept.feature_index,
                            cluster_target=work.cluster_target,
                        )
                    )
                    rich_qc_items.append(item)
                continue
            assert work.avoid_pending is not None
            item.continuation = _parse_positive_avoid_continuation(
                item.result,
                work.avoid_pending,
                item.request.payload,
                cfg,
                tokenizer,
                attempt=item.attempt,
            )

        if rich_qc_inputs:
            with profile_region(
                "stage1b.compute_qc_percentiles_batch",
                rows=len(rich_qc_inputs),
            ):
                rich_percentiles = (
                    [
                        output_fc_percentile_for_continuation(
                            item.prefix_text,
                            item.continuation_text,
                            item.feature_id,
                            base_model,
                            base_sae,
                            percentile_table,
                            sae_layer=sae_layer,
                            tokenizer_id=cfg.extraction_tokenizer_id,
                            **_cluster_target_kwargs(item.cluster_target),
                        )
                        for item in rich_qc_inputs
                    ]
                    if scalar_qc
                    else output_fc_percentiles_for_continuations(
                        rich_qc_inputs,
                        base_model,
                        base_sae,
                        percentile_table,
                        sae_layer=sae_layer,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                    )
                )
            if len(rich_percentiles) != len(rich_qc_items):
                raise AxBenchBaseError(
                    "Stage 1b rich batch scorer returned "
                    f"{len(rich_percentiles)} scores for "
                    f"{len(rich_qc_items)} continuations"
                )
            for item, percentile in zip(
                rich_qc_items,
                rich_percentiles,
                strict=True,
            ):
                item.output_percentile = _validate_raw_positive_percentile(
                    percentile,
                    scorer="output scorer",
                    concept=item.work.concept,
                    seed_idx=item.work.pending.selected.seed_idx,
                )

        strict_qc_inputs: list[ContinuationQcInput] = []
        strict_qc_tags: list[
            tuple[_PreparedPositiveBatchResult, str]
        ] = []
        for item in prepared:
            work = item.work
            continuation = item.continuation
            if item.phase == "rich":
                if continuation is None or item.output_percentile is None:
                    continue
                if not _positive_rich_threshold_passes(
                    cfg,
                    work.pending.selected,
                    item.output_percentile,
                ):
                    continue
                natural = work.pending.selected.seed.continuation_text
                if not natural.strip():
                    continue
                strict_qc_inputs.append(
                    ContinuationQcInput(
                        prefix_text=work.pending.selected.prefix_text,
                        continuation_text=natural,
                        feature_id=work.concept.feature_index,
                        cluster_target=work.cluster_target,
                    )
                )
                strict_qc_tags.append((item, "natural"))
                continue
            if continuation is None:
                continue
            strict_qc_inputs.append(
                ContinuationQcInput(
                    prefix_text=work.pending.selected.prefix_text,
                    continuation_text=continuation,
                    feature_id=work.concept.feature_index,
                    cluster_target=work.cluster_target,
                )
            )
            strict_qc_tags.append((item, "llm"))

        if strict_qc_inputs:
            strict_outcomes = (
                [
                    classify_avoid_continuation(
                        item.prefix_text,
                        item.continuation_text,
                        item.feature_id,
                        base_model,
                        base_sae,
                        sae_layer=sae_layer,
                        extraction_threshold=cfg.extraction_threshold,
                        tokenizer_id=cfg.extraction_tokenizer_id,
                        **_cluster_target_kwargs(item.cluster_target),
                    )
                    for item in strict_qc_inputs
                ]
                if scalar_qc
                else classify_avoid_continuations(
                    strict_qc_inputs,
                    base_model,
                    base_sae,
                    sae_layer=sae_layer,
                    extraction_threshold=cfg.extraction_threshold,
                    tokenizer_id=cfg.extraction_tokenizer_id,
                    profile_region_name=(
                        "stage1b.avoid_strict_zero_qc_batch"
                    ),
                )
            )
            if len(strict_outcomes) != len(strict_qc_tags):
                raise AxBenchBaseError(
                    "Stage 1b strict-zero batch scorer returned "
                    f"{len(strict_outcomes)} outcomes for "
                    f"{len(strict_qc_tags)} continuations"
                )
            for (item, kind), outcome in zip(
                strict_qc_tags,
                strict_outcomes,
                strict=True,
            ):
                if kind == "natural":
                    item.natural_outcome = outcome
                else:
                    item.avoid_outcome = outcome

        next_by_position: list[SynthRequest | None] = [None] * len(prepared)
        for item in sorted(prepared, key=lambda candidate: candidate.request.index):
            emit_progress_heartbeat()
            work = item.work
            tracker = tracker_by_concept[work.concept.concept_id]
            attempt = item.attempt

            if item.phase == "rich":
                with _defer_sigint_for_checkpoint_state():
                    tracker.semantic_rich_attempts += 1
                    work.rich_semantic_attempts += 1
                record_counter(
                    "stage1b.rich_semantic_attempts",
                    1,
                    concept_id=work.concept.concept_id,
                    seed_idx=work.pending.seed_idx,
                    attempt=attempt,
                )
                if item.continuation is not None:
                    if item.output_percentile is None:
                        raise AxBenchBaseError(
                            "Stage 1b rich batch lost its aligned percentile"
                        )
                    _apply_positive_rich_result(
                        item.continuation,
                        item.output_percentile,
                        work.pending,
                        cfg,
                        work.concept,
                        gen_source,
                        attempt=attempt,
                    )
                if work.pending.accepted_row is None:
                    if attempt + 1 < max_attempts:
                        next_by_position[item.position] = (
                            _positive_rich_chain_request(
                                cfg,
                                rich_template,
                                work,
                                attempt=attempt + 1,
                            )
                        )
                    else:
                        discard_exhausted_candidate(
                            tracker,
                            work,
                            side="rich",
                        )
                    continue

                work.rich_row = work.pending.accepted_row
                natural_row: dict[str, Any] | None = None
                natural = work.pending.selected.seed.continuation_text
                if natural.strip():
                    if item.natural_outcome is None:
                        raise AxBenchBaseError(
                            "Stage 1b rich pass lost its aligned natural-avoid "
                            "strict-zero outcome"
                        )
                    natural_row = _apply_positive_natural_avoid_outcome(
                        cfg,
                        work.concept,
                        work.pending.selected,
                        item.natural_outcome,
                    )
                if natural_row is not None:
                    commit_success(
                        tracker,
                        work,
                        avoiding_row=natural_row,
                        natural_avoid=True,
                    )
                    continue
                work.phase = "avoid"
                work.avoid_pending = _PositiveAvoidPending(work=work)
                next_by_position[item.position] = _positive_avoid_chain_request(
                    cfg,
                    avoiding_template,
                    work.avoid_pending,
                    attempt=0,
                )
                continue

            if work.phase != "avoid" or work.avoid_pending is None:
                raise AxBenchBaseError(
                    "Stage 1b request chain entered an invalid phase "
                    f"{work.phase!r}"
                )
            with _defer_sigint_for_checkpoint_state():
                tracker.semantic_avoid_attempts += 1
                work.avoid_semantic_attempts += 1
            record_counter(
                "stage1b.avoid_semantic_attempts",
                1,
                concept_id=work.concept.concept_id,
                seed_idx=work.pending.seed_idx,
                attempt=attempt,
            )
            if item.continuation is not None:
                if item.avoid_outcome is None:
                    raise AxBenchBaseError(
                        "Stage 1b LLM-avoid batch lost its aligned strict-zero "
                        "outcome"
                    )
                _apply_positive_avoid_outcome(
                    work.avoid_pending,
                    item.continuation,
                    item.avoid_outcome,
                    cfg,
                    attempt=attempt,
                )
            if work.avoid_pending.accepted_row is not None:
                commit_success(
                    tracker,
                    work,
                    avoiding_row=work.avoid_pending.accepted_row,
                    natural_avoid=False,
                )
                continue
            if attempt + 1 < max_attempts:
                next_by_position[item.position] = _positive_avoid_chain_request(
                    cfg,
                    avoiding_template,
                    work.avoid_pending,
                    attempt=attempt + 1,
                )
            else:
                discard_exhausted_candidate(
                    tracker,
                    work,
                    side="avoid",
                )

        return next_by_position

    terminal_chains_since_checkpoint = 0

    def refill_terminal_chain(
        request: SynthRequest,
    ) -> list[SynthRequest]:
        nonlocal terminal_chains_since_checkpoint
        work = work_by_order.pop(request.index, None)
        if work is None:
            state = (
                "duplicate terminal"
                if request.index in terminal_chain_indices
                else "unknown"
            )
            raise AxBenchBaseError(
                "Stage 1b terminal-chain refill was invoked for a "
                f"{state} chain {request.index}"
            )
        if not work.terminal:
            raise AxBenchBaseError(
                "Stage 1b terminal-chain refill was invoked for a live chain "
                f"{request.index}"
            )
        terminal_chain_indices.add(request.index)
        tracker = tracker_by_concept[work.concept.concept_id]
        requests = rich_requests_for(admit_refills(tracker))
        terminal_chains_since_checkpoint += 1
        if (
            terminal_chains_since_checkpoint
            >= _STAGE1B_CHECKPOINT_TERMINAL_INTERVAL
        ):
            commit_checkpoint()
            terminal_chains_since_checkpoint = 0
        if (
            max_concurrent == 1
            and not tracker.outstanding
            and (
                tracker.accepted == tracker.target
                or tracker.termination_reason is not None
            )
            and tracker.concept.concept_id not in released_serial_concepts
        ):
            release_cuda_iteration_memory()
            released_serial_concepts.add(tracker.concept.concept_id)
        if (
            cfg.qc_fallback_on_exhaust == "error"
            and tracker.accepted < tracker.target
            and tracker.termination_reason is not None
        ):
            raise shortfall_error(tracker)
        return requests

    heartbeat_state = {"last": time.monotonic()}

    def emit_progress_heartbeat(*, force: bool = False) -> None:
        """Log aggregate Stage 1b progress while concepts are still in flight.

        ``record_tracker_summary`` reports a concept only once it TERMINATES.
        A run whose concepts are all mid-flight therefore emits nothing for as
        long as the stage takes, which makes a slow stage indistinguishable
        from a stalled one and hides the accept/attempt ratio until the very
        end. This heartbeat closes that window; it reads only counters the
        trackers already maintain and performs no extra work per request.
        """
        now = time.monotonic()
        if not force and now - heartbeat_state["last"] < _PROGRESS_HEARTBEAT_SECONDS:
            return
        heartbeat_state["last"] = now
        done = sum(1 for t in trackers if t.termination_reason is not None)
        accepted = sum(t.accepted for t in trackers)
        target = sum(t.target for t in trackers)
        drawn = sum(t.seeds_drawn for t in trackers)
        discards = sum(t.discarded_rich + t.discarded_avoid for t in trackers)
        attempts = sum(
            t.semantic_rich_attempts + t.semantic_avoid_attempts for t in trackers
        )
        in_flight = sum(len(t.outstanding) for t in trackers)
        # INFO: at one line per _PROGRESS_HEARTBEAT_SECONDS this is the ONLY
        # default-verbosity signal that a long stage is converging rather than
        # stalled. It was briefly DEBUG, which made it invisible at default
        # verbosity and therefore useless; the per-attempt rejection spam that
        # motivated hiding it now lives at DEBUG instead (see
        # continuation_protocol), so this line no longer competes with it.
        logger.info(
            "Stage 1b progress: concepts_terminated={}/{}, "
            "accepted_pairs={}/{}, seeds_drawn={}, discards={}, "
            "semantic_attempts={}, in_flight={}, attempts_per_accept={}",
            done,
            len(trackers),
            accepted,
            target,
            drawn,
            discards,
            attempts,
            in_flight,
            f"{attempts / accepted:.1f}" if accepted else "n/a",
        )
        record_counter("stage1b.progress.accepted_pairs", accepted)
        record_counter("stage1b.progress.concepts_terminated", done)
        record_counter("stage1b.progress.semantic_attempts", attempts)

    def record_tracker_summary(
        tracker: _PositiveTargetTracker,
        *,
        stage_aborted: bool = False,
    ) -> None:
        termination_reason = tracker.termination_reason
        if termination_reason is None and stage_aborted:
            termination_reason = "stage_abort"
        summary = {
            "eligible": (
                tracker.source.clean_count + tracker.source.dirty_count
            ),
            "scored": tracker.source.scored_count,
            "headroom_rejected": tracker.source.headroom_rejected,
            "seeds_drawn": tracker.seeds_drawn,
            "rich_discards": tracker.discarded_rich,
            "avoid_discards": tracker.discarded_avoid,
            "rich_semantic_attempts": tracker.semantic_rich_attempts,
            "avoid_semantic_attempts": tracker.semantic_avoid_attempts,
            "natural_avoid_acceptances": (
                tracker.natural_avoid_acceptances
            ),
            "accepted_complete_pairs": tracker.accepted,
            "shortfall": tracker.target - tracker.accepted,
            "termination_reason": termination_reason,
            "elapsed_seconds": time.perf_counter() - tracker.started_at,
        }
        logger.info(
            "Stage 1b concept target-refill summary: concept_id={} {}",
            tracker.concept.concept_id,
            summary,
        )
        tracker.source.record_selection_counters()
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                record_counter(
                    f"stage1b.summary.{key}",
                    value,
                    concept_id=tracker.concept.concept_id,
                    termination_reason=termination_reason,
                )
        if termination_reason is not None:
            record_counter(
                "stage1b.summary.terminal",
                1,
                concept_id=tracker.concept.concept_id,
                termination_reason=termination_reason,
            )

    pool_completed = False
    try:
        newly_admitted: list[_PositiveWorkItem] = []
        for tracker in trackers:
            newly_admitted.extend(admit_refills(tracker))
        # Fresh admissions and resumed outstanding descriptors become durable
        # before provider requests are even constructed.
        commit_checkpoint()
        initial_requests = rich_requests_for(
            sorted(
                [*resumed_works, *newly_admitted],
                key=lambda work: work.order,
            )
        )
        if not initial_requests and any(
            tracker.accepted < tracker.target
            and tracker.termination_reason is None
            for tracker in trackers
        ):
            raise AxBenchBaseError(
                "Stage 1b target-refill scheduler produced no initial request "
                "chains despite an unterminated target deficit"
            )
        if initial_requests:
            scheduler_kwargs: dict[str, Any]
            if max_concurrent > 1 and cfg.qc_batch_size > 1:
                scheduler_kwargs = {
                    "handle_results": handle_results,
                    "result_batch_size": cfg.qc_batch_size,
                }
            else:
                scheduler_kwargs = {"handle_result": handle_result}
            complete_synth_request_chains(
                llm_client,
                initial_requests,
                terminal_chain_refill=refill_terminal_chain,
                max_concurrent=max_concurrent,
                profile_region_name="stage1b.synth_llm_complete",
                pool_region_name="stage1b.complete_pair_request_chain_pool",
                **scheduler_kwargs,
            )
        # A complete (or tolerated-shortfall) scheduler state is durable before
        # final artifact preparation begins.
        commit_checkpoint()
        pool_completed = True
    except BaseException as stage_error:
        try:
            commit_checkpoint()
        except Exception as checkpoint_error:
            stage_error.add_note(
                "Stage 1b emergency checkpoint failed; the previous committed "
                f"checkpoint remains authoritative: {checkpoint_error!r}"
            )
            logger.opt(exception=checkpoint_error).error(
                "Stage 1b emergency checkpoint failed while preserving the "
                "original stage error"
            )
        for tracker in trackers:
            try:
                record_tracker_summary(tracker, stage_aborted=True)
            except Exception as summary_error:
                stage_error.add_note(
                    "Stage 1b aborted-summary recording also failed for "
                    f"concept_id={tracker.concept.concept_id}: "
                    f"{summary_error!r}"
                )
                logger.opt(exception=summary_error).error(
                    "Stage 1b aborted-summary recording failed for concept {}",
                    tracker.concept.concept_id,
                )
        raise
    finally:
        if max_concurrent != 1 or not pool_completed:
            release_cuda_iteration_memory()

    rows_by_concept: dict[int, list[dict[str, Any]]] = {}
    for tracker in trackers:
        if tracker.outstanding:
            raise AxBenchBaseError(
                "Stage 1b scheduler drained with outstanding chains: "
                f"concept_id={tracker.concept.concept_id}, "
                f"seeds={sorted(tracker.outstanding)}"
            )
        record_tracker_summary(tracker)
        if tracker.accepted < tracker.target:
            if tracker.termination_reason is None:
                raise AxBenchBaseError(
                    "Stage 1b target shortfall has no terminal reason: "
                    f"concept_id={tracker.concept.concept_id}, "
                    f"accepted={tracker.accepted}, target={tracker.target}"
                )
            error = shortfall_error(tracker)
            if cfg.qc_fallback_on_exhaust == "error":
                raise error
            logger.warning(
                "{}; qc_fallback_on_exhaust={!r} retains only complete "
                "accepted pairs",
                error,
                cfg.qc_fallback_on_exhaust,
            )
        accepted_items = sorted(
            tracker.accepted_items,
            key=lambda item: item.pending.selected.candidate_rank,
        )
        rows: list[dict[str, Any]] = []
        for work in accepted_items:
            if work.rich_row is None or work.avoiding_row is None:
                raise AxBenchBaseError(
                    "Stage 1b accepted-item assembly found a half-pair"
                )
            rows.extend((work.rich_row, work.avoiding_row))
        rows_by_concept[tracker.concept.concept_id] = rows
    return rows_by_concept


def _write_training_parquet(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    tokenizer_id: str,
    cfg: AxBenchBaseConfig,
    cluster_contract: ClusterFeatureContract | None = None,
    concepts: list[Concept] | None = None,
) -> None:
    """Serialize rows to parquet using the training-corpus schema."""
    required_delta = derive_positive_qc_thresholds(
        cfg.accept_threshold,
        cfg.positive_prefix_headroom_ceiling,
    ).required_delta
    validate_positive_qc_rows(
        rows,
        accept_threshold_points=required_delta,
    )
    schema = schema_with_extraction_tokenizer_metadata(
        TRAINING_SCHEMA,
        tokenizer_id,
    )
    schema = schema_with_positive_qc_contract(
        schema,
        accept_threshold=cfg.accept_threshold,
    )
    schema = schema_with_positive_selection_contract(schema, cfg)
    schema = schema_with_stage1_continuation_contract(
        schema,
        role="stage1b_positive",
    )
    if cluster_contract is not None:
        if concepts is None:
            raise AxBenchBaseError("clustered synth writer requires its concept population")
        targets = [
            cluster_contract.target_for_representative(concept.feature_index)
            for concept in concepts
        ]
        metadata = dict(schema.metadata or {})
        metadata[b"feature_contract_fingerprint"] = (
            cluster_contract.feature_contract_fingerprint.encode("utf-8")
        )
        metadata[b"cluster_qc_target_algorithm"] = str(
            cluster_contract.semantic_dict()["algorithms"]["qc_target_reduction"]
        ).encode("utf-8")
        metadata[b"cluster_target_members_json"] = json.dumps(
            {str(target.representative_id): list(target.allowed_member_ids) for target in targets},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        schema = schema.with_metadata(metadata)
    table = pa.Table.from_pylist(
        rows,
        schema=schema,
    )
    if cluster_contract is None:
        pq.write_table(table, path)
    else:
        write_clustered_parquet_once(
            table,
            path,
            surface="Stage 1b positive synth corpus",
        )


def _validate_emitted_positive_rows(
    cfg: AxBenchBaseConfig,
    rows: list[dict[str, Any]],
) -> None:
    """Fail before publication if a Stage 1b row violates prefix contracts."""
    for row_idx, row in enumerate(rows):
        prefix_text = row.get("prefix_text")
        if not isinstance(prefix_text, str):
            raise AxBenchBaseError(
                f"Stage 1b row {row_idx} has non-string prefix_text {prefix_text!r}"
            )
        prefix_length = len(prefix_text)
        if not (cfg.min_prefix_chars <= prefix_length <= cfg.max_prefix_chars):
            raise AxBenchBaseError(
                f"Stage 1b row {row_idx} concept_id={row.get('concept_id')!r} "
                f"content_class={row.get('content_class')!r} has prefix length "
                f"{prefix_length}, outside [{cfg.min_prefix_chars}, "
                f"{cfg.max_prefix_chars}]"
            )
        output_text = row.get("output_text")
        if not isinstance(output_text, str) or not output_text.strip():
            raise AxBenchBaseError(f"Stage 1b row {row_idx} has empty or non-string output_text")
        expected_output_start = _exact_output_start(prefix_text, output_text)
        if row.get("n_prefix_nanochat_tokens") != expected_output_start:
            raise AxBenchBaseError(
                f"Stage 1b row {row_idx} has stale nanochat output boundary "
                f"{row.get('n_prefix_nanochat_tokens')!r}; recomputed "
                f"{expected_output_start}"
            )
        if row.get("content_class") != CONCEPT_AVOIDING_LABEL:
            continue
        if row_idx == 0:
            raise AxBenchBaseError(
                "Stage 1b concept_avoiding row appears without a preceding concept_rich row"
            )
        rich_row = rows[row_idx - 1]
        if (
            rich_row.get("content_class") != CONCEPT_RICH_LABEL
            or rich_row.get("concept_id") != row.get("concept_id")
            or rich_row.get("prefix_text") != prefix_text
        ):
            raise AxBenchBaseError(
                f"Stage 1b row {row_idx} concept_avoiding prefix is not paired "
                "with the immediately preceding concept_rich row"
            )


def synthesize_positive_mode(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
    seeds_train: list[SeedRow],
    base_model: Any,
    base_sae: Any,
    target_model_meta: TargetModelMetadata,
    *,
    seed_rows_sha256: str,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Path:
    """Stage 1b entrypoint. Returns the path to the written parquet.

    The output parquet contains per-concept rows ONLY, in concept_id
    order, with each accepted ``concept_rich`` row immediately followed by
    its paired ``concept_avoiding`` row of the same prefix. The legacy
    Stage 1a global-negative prepend is gone: each prefix now owns both
    completions, so positive mode is symmetric with negative mode.

    ``target_model_meta`` carries the SAE layer (via ``sae_layer``)
    used by every QC / percentile helper. The CLI (Task 43) builds
    this once via :func:`detect_target_model_metadata` and threads
    it through every Stage 0c / Stage 1 / Stage 2 entrypoint.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before calling synthesize_positive_mode")
    if not concepts:
        raise AxBenchBaseError("synthesize_positive_mode called with empty concepts list")
    if not seeds_train:
        raise AxBenchBaseError("synthesize_positive_mode called with empty seeds_train list")
    if not is_sha256_hex(seed_rows_sha256):
        raise AxBenchBaseError(
            "synthesize_positive_mode requires seed_rows_sha256 to be a "
            "lowercase 64-character SHA-256 digest"
        )
    if clustered_source_enabled(cfg) != (cluster_contract is not None):
        raise AxBenchBaseError(
            "Clustered Stage 1b requires config and validated feature-contract "
            "state to be present together"
        )
    if cluster_contract is not None:
        load_cluster_candidate_identity(
            cfg,
            expected_fingerprint=cluster_contract.feature_contract_fingerprint,
        )
        for concept in concepts:
            cluster_contract.target_for_representative(concept.feature_index)
    _validate_seed_feature_ids_sorted(seeds_train)
    effective_prefixes = _resolve_stage1b_prefixes(cfg, seeds_train)
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    partial_json_path = output_paths.stage1b_partial_json()
    completed_positive_path = output_paths.synth_training_positive()
    if completed_positive_path.is_file():
        # A committed corpus is evidence, not an implicit growth checkpoint.
        # This remains true in the crash window where its partial JSON marker
        # has not yet been cleaned up: that marker can resume the same target,
        # but it cannot authorize replacing already-committed final evidence.
        completed_schema = pq.read_schema(completed_positive_path)
        completed_contract = validate_positive_selection_contract(
            completed_positive_path,
            consumer="Stage 1b completed-output preflight",
            schema=completed_schema,
        )
        completed_target = int(
            completed_contract["n_positives_per_concept"]
        )
        if completed_target != cfg.n_positives_per_concept:
            raise AxBenchBaseError(
                "Stage 1b completed positive corpus target mismatch at "
                f"{completed_positive_path}: stored target={completed_target}, "
                f"current target={cfg.n_positives_per_concept}. Completed "
                "evidence cannot be extended in place, even when "
                "stage1b_partial.json remains after a final-publication crash "
                "window. Use a fresh explicit run_tag."
            )
        # Preserve the ordinary validator error for corruption or same-target
        # semantic drift instead of misdiagnosing either as target growth.
        validate_positive_selection_contract(
            completed_positive_path,
            consumer="Stage 1b completed-output preflight",
            schema=completed_schema,
            expected_cfg=cfg,
        )

    with profile_region("stage1b.build_percentile_table"):
        percentile_table = build_percentile_table_from_stats(
            cfg.feature_statistics_path,
            n_sae_features=_infer_n_sae_features(base_sae),
        )
    checkpoint_fingerprint = _stage1b_checkpoint_fingerprint(
        cfg,
        concepts,
        seed_rows_sha256,
        target_model_meta,
        cluster_contract=cluster_contract,
    )
    legacy_fingerprint_factory = _legacy_stage1b_fingerprint_factory(
        cfg,
        seeds_train,
        checkpoint_fingerprint,
    )
    loaded_checkpoint = positive_checkpoint.load_checkpoint(
        json_path=partial_json_path,
        parquet_path=output_paths.stage1b_partial_parquet(),
        expected_fingerprint=checkpoint_fingerprint,
        training_schema=TRAINING_SCHEMA,
        current_target=cfg.n_positives_per_concept,
        legacy_fingerprint_factory=legacy_fingerprint_factory,
    )
    with profile_region("stage1b.build_prefix_percentile_cache"):
        # Durable backing for the in-process cache. Prefix scores are a pure
        # function of the cache key, so persisting them makes a relaunch skip
        # the pre-provider scoring pass (~8.5 min of the ~11.7 min startup on a
        # 50-concept run) instead of repeating it. The store lives beside the
        # run directory so every run_tag on the same model/SAE/statistics
        # shares it.
        prefix_store = (
            None
            if not cfg.enable_prefix_percentile_store
            else PrefixPercentileStore(
                Path(paths.AxBenchBasePaths.from_config(cfg).run_output_root())
                / "_prefix_percentile_store.sqlite"
            )
        )
        prefix_cache = PrefixPercentileCache(
            base_model=base_model,
            base_sae=base_sae,
            percentile_table=percentile_table,
            sae_layer=target_model_meta.sae_layer,
            tokenizer_id=cfg.extraction_tokenizer_id,
            percentile_namespace=cfg.feature_statistics_path,
            batch_size=cfg.qc_batch_size,
            store=prefix_store,
        )
    if loaded_checkpoint is None:
        # Candidate selection and exact-target sufficiency are resolved for
        # every concept before the response cache/client can make a provider
        # call. This is the byte-identical legacy fresh-start path.
        trackers = _prepare_positive_target_trackers(
            cfg,
            concepts,
            seeds_train,
            base_model,
            base_sae,
            percentile_table,
            target_model_meta,
            prefix_cache=prefix_cache,
            effective_prefixes=effective_prefixes,
            cluster_contract=cluster_contract,
        )
        resume_next_chain_index = 0
        resume_next_acceptance_sequence = 0
    else:
        checkpoint_document, checkpoint_rows = loaded_checkpoint
        trackers = _restore_positive_target_trackers(
            cfg,
            concepts,
            seeds_train,
            base_model,
            base_sae,
            percentile_table,
            target_model_meta,
            prefix_cache=prefix_cache,
            effective_prefixes=effective_prefixes,
            cluster_contract=cluster_contract,
            checkpoint_document=checkpoint_document,
            checkpoint_rows=checkpoint_rows,
        )
        resume_next_chain_index = int(
            checkpoint_document["next_chain_index"]
        )
        resume_next_acceptance_sequence = int(
            checkpoint_document["next_acceptance_sequence"]
        )
        logger.info(
            "Resuming Stage 1b from {} complete accepted pairs and {} "
            "outstanding chains at {} (stored highest target={}, current "
            "target={})",
            checkpoint_document["accepted_pair_count"],
            sum(len(tracker.outstanding) for tracker in trackers),
            partial_json_path,
            checkpoint_document["highest_requested_target"],
            cfg.n_positives_per_concept,
        )

    all_positive_rows: list[dict[str, Any]] = []
    yield_records: list[ConceptYieldRecord] = []

    def consume_concept_rows(
        concept: Concept,
        rows: list[dict[str, Any]],
    ) -> None:
        _validate_emitted_positive_rows(cfg, rows)
        kept_rows, yield_record = filter_positive_concept_rows(
            concept,
            rows,
            attempted=cfg.n_positives_per_concept,
            max_drop_fraction=cfg.max_concept_drop_fraction,
        )
        yield_records.append(yield_record)
        all_positive_rows.extend(kept_rows)
        logger.info(
            "Stage 1b concept yield: concept_id={} attempted={} "
            "accepted={} dropped={} drop_fraction={:.6f} excluded={} "
            "reasons={}",
            concept.concept_id,
            yield_record.attempted,
            yield_record.accepted,
            yield_record.dropped,
            yield_record.drop_fraction,
            yield_record.excluded,
            yield_record.reason_counts,
        )
        record_counter(
            "stage1b.positive_rows",
            len(kept_rows),
            concept_id=concept.concept_id,
        )
        record_counter(
            "stage1b.logical_items_accepted",
            yield_record.accepted,
            concept_id=concept.concept_id,
        )
        record_counter(
            "stage1b.logical_items_dropped",
            yield_record.dropped,
            concept_id=concept.concept_id,
        )
        if yield_record.excluded:
            record_counter(
                "stage1b.concepts_excluded",
                1,
                concept_id=concept.concept_id,
            )

    # The response cache lives for the duration of the stage; individual
    # LLM calls hit and write to it, so a mid-stage crash (e.g., concept
    # or request-chain failure) leaves every successful response committed
    # immediately and queryable on re-run without re-spending.
    with SynthResponseCache(
        paths.AxBenchBasePaths.from_config(cfg).synth_response_cache()
    ) as cache:
        with profile_region("stage1b.build_synth_client"):
            llm_client = _build_synth_client(cfg, response_cache=cache)
        with profile_region(
            "stage1b.synthesize_complete_pair_targets",
            concepts=len(concepts),
            target_pairs=(
                len(concepts) * cfg.n_positives_per_concept
            ),
            max_concurrent=_synth_max_concurrent(cfg),
        ):
            rows_by_concept = _run_positive_target_refill_engine(
                cfg,
                trackers,
                base_model,
                base_sae,
                llm_client,
                percentile_table,
                target_model_meta,
                checkpoint_fingerprint=checkpoint_fingerprint,
                checkpoint_json_path=partial_json_path,
                checkpoint_parquet_path=(
                    output_paths.stage1b_partial_parquet()
                ),
                highest_requested_target=cfg.n_positives_per_concept,
                resume_next_chain_index=resume_next_chain_index,
                resume_next_acceptance_sequence=(
                    resume_next_acceptance_sequence
                ),
            )
        for concept in concepts:
            consume_concept_rows(
                concept,
                rows_by_concept.get(concept.concept_id, []),
            )

    report_path = output_paths.stage1b_concept_yield_report()
    surviving_concept_ids = prepare_concept_yield_report(
        cfg,
        stage="stage1b",
        records=yield_records,
        path=report_path,
    )
    surviving_concepts = write_filtered_concepts(
        concepts,
        surviving_concept_ids,
        output_paths.stage1b_concepts(),
    )
    _validate_emitted_positive_rows(cfg, all_positive_rows)
    out_path = output_paths.synth_training_positive()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_region(
        "stage1b.write_training_parquet",
        rows=len(all_positive_rows),
    ):
        if cluster_contract is None:
            _write_training_parquet(
                all_positive_rows,
                out_path,
                cfg=cfg,
                tokenizer_id=cfg.extraction_tokenizer_id,
            )
        else:
            _write_training_parquet(
                all_positive_rows,
                out_path,
                cfg=cfg,
                tokenizer_id=cfg.extraction_tokenizer_id,
                cluster_contract=cluster_contract,
                concepts=surviving_concepts,
            )
    write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=yield_records,
        path=report_path,
    )
    partial_parquet_path = output_paths.stage1b_partial_parquet()
    try:
        positive_checkpoint.delete_checkpoint(
            json_path=partial_json_path,
            parquet_path=partial_parquet_path,
        )
    except OSError as exc:
        json_marker_remains = partial_json_path.exists()
        parquet_remains = partial_parquet_path.exists()
        logger.warning(
            "Stage 1b final artifacts committed, but partial-checkpoint "
            "cleanup failed: {}. Current partial state: JSON marker exists={}, "
            "parquet exists={}. A remaining JSON marker is still an "
            "authoritative committed checkpoint; parquet is an ignored orphan "
            "only when JSON is absent.",
            exc,
            json_marker_remains,
            parquet_remains,
        )
    record_counter("stage1b.rows_written", len(all_positive_rows))
    logger.info(
        f"Wrote {len(all_positive_rows)} per-concept training rows "
        "(each accepted concept_rich row paired with a concept_avoiding row) "
        f"to {out_path}"
    )
    return out_path
