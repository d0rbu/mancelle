"""Stage 0c shared internals (PR-4 cycle break).

This module is the canonical home for the private helpers, dataclasses,
constants, and schema definitions that BOTH ``negative_prefixes`` (the
public Stage 0c entrypoint module) AND ``stage0c_runner`` (the per-rank
runner) need to share. Pulling them here breaks the structural import
cycle: ``negative_prefixes`` -> ``stage0c_runner`` -> ``negative_prefixes``
that previously required a deferred bottom-of-file import.

Module-level dependency rule: ``_stage0c_internals`` MUST NOT import
from ``negative_prefixes`` or ``stage0c_runner``. It depends only on:

- standard library
- third-party (``pyarrow``, ``torch``, ``loguru``)
- nanochat sibling helpers (``atomic_io``, ``dataset``)
- non-circular axbench_base modules (``paths``, ``config``,
  ``concept_examples_qc``, ``concept_list``, ``extraction``,
  ``_log_helpers``, ``profiler``, ``synth.*``, ``types``,
  ``validate``, ``batched_generation``)
- ``steering.feature_io`` and ``steering.feature_paths``

The leading underscore on the module name signals that its API surface
is package-private. ``negative_prefixes`` owns the public Stage 0c entrypoint,
while shared implementation and test patch boundaries live here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_write_bytes
from nanochat.dataset import list_parquet_files
from steering.eval.axbench_base import paths
from steering.eval.axbench_base._log_helpers import is_log_rank
from steering.eval.axbench_base.batched_generation import (
    batched_generate_seeds,
    cross_prefix_batched_generate_seeds,
)
from steering.eval.axbench_base.cluster_artifacts import (
    clustered_stage0c_fingerprint_fields,
)
from steering.eval.axbench_base.concept_examples_qc import (
    QCFeatureScore,
    QcOutcome,
    QCReport,
)
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    VerificationPrefillMode,
    effective_extraction_model_id,
)
from steering.eval.axbench_base.percentile_units import (
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.prefix_text import window_prefix_text
from steering.eval.axbench_base.profiler import (
    process_memory_snapshot,
    profile_region,
    record_counter,
)
from steering.eval.axbench_base.stage0c_api_generation import (
    Stage0cApiGenerator,
    api_generation_fingerprint_config,
)
from steering.eval.axbench_base.synth.qc import compute_fc_percentiles
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.types import (
    STAGE0C_API_CHECKPOINT_SCHEMA_VERSION,
    STAGE0C_LOCAL_CHECKPOINT_SCHEMA_VERSION,
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    RejectedConceptReason,
)
from steering.feature_io import load_features
from steering.feature_paths import parse_feature_dir, resolve_data_dir
from steering.parquet_io import read_parquet_texts

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureTarget

_VERIFY_SEED_OFFSET = 0xE80001  # distinct from other pipeline offsets
_VSEED_INCREMENT = 100  # per-verification-seed stride within a concept
_CONCEPT_STRIDE = 10_000  # per-concept stride in the seed space

# Emit a within-concept progress heartbeat every N candidates OR every T
# seconds, whichever comes first. Per-candidate logging would be too
# chatty (hundreds of candidates per concept) but concept-level silence
# for 10-20 minutes makes the pipeline look hung. These thresholds aim
# for ~1 line per 30-90s of wall time on a typical GPU.
_PROGRESS_EVERY_N_CANDIDATES = 10
_PROGRESS_EVERY_SECONDS = 30.0

# Heartbeat thresholds for the per-shard-feature segment-scanning loop in
# _CandidatePrefixIndex._materialize_shard_candidates. Counter is per-index
# (cross-call) so many small features still trigger a heartbeat once
# aggregate scan crosses _HEARTBEAT_SEGMENTS; _HEARTBEAT_SECONDS is the
# wall-time fallback so a long stretch of below-threshold features is not
# silent forever.
_HEARTBEAT_SEGMENTS = 1000
_HEARTBEAT_SECONDS = 30.0


_NEG_PREFIXES_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("prefix_text", pa.string()),
        ("prefix_source_shard", pa.int32()),
        ("prefix_source_doc", pa.int32()),
        ("prefix_activating_seg_idx", pa.int32()),
        ("verification_continuations", pa.list_(pa.string())),
        ("verification_seeds", pa.list_(pa.int64())),
        ("verification_percentiles", pa.list_(pa.float32())),
    ]
)
_NEG_PREFIXES_TOKENIZER_METADATA_KEY = "extraction_tokenizer_id"
_NEG_PREFIXES_EXTRACTOR_METADATA_KEYS = (
    "neuronpedia_model_id",
    "feature_extraction_model_id",
    "neuronpedia_source",
)
_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY = "candidate_pool_identity"
_NEG_PREFIXES_CONCEPTS_METADATA_KEY = "concepts_identity"
_NEG_PREFIXES_REJECTED_CONCEPTS_METADATA_KEY = "rejected_concepts_identity"
_NEG_PREFIXES_QC_REPORT_METADATA_KEY = "concept_examples_qc_report_identity"
_NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY = "stage0c_fingerprint_identity"
_NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS = (
    _NEG_PREFIXES_CONCEPTS_METADATA_KEY,
    _NEG_PREFIXES_REJECTED_CONCEPTS_METADATA_KEY,
    _NEG_PREFIXES_QC_REPORT_METADATA_KEY,
    _NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY,
)


class _CompletedStage0cExtractorIdentityError(AxBenchBaseError):
    """A completed Stage 0c artifact has missing or stale provenance.

    This is distinct from ordinary completed-artifact corruption.  An
    automatic Stage 0c-only rerun is unsafe because ``candidate_concepts.json``
    is produced by Stage 0a under the Neuronpedia identity but does not itself
    persist that full identity.
    """


class _CompletedStage0cCandidatePoolIdentityError(AxBenchBaseError):
    """A completed Stage 0c artifact is not tied to the live candidate pool."""


class _CompletedStage0cArtifactSetIdentityError(AxBenchBaseError):
    """The completed parquet is not tied to its JSON sidecars/config."""


# ---------------------------------------------------------------------------
# Internal dataclasses (replace dict-with-magic-keys pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PrefixCandidate:
    """A candidate prefix pulled from a validation shard segment.

    Replaces the previous ``dict[str, Any]`` with the keys
    ``{doc_idx, seg_idx, shard_idx, prefix_text, prefix_gemma_tokens}``.
    ``source_bridge_text`` is a bounded, exact slice immediately following the
    prefix in the source document.  It is ignored by the legacy local sampler
    and lets the API path prove that a generated continuation crosses the real
    source boundary coherently.
    """

    shard_idx: int
    doc_idx: int
    seg_idx: int
    prefix_text: str
    prefix_gemma_tokens: list[int]
    source_bridge_text: str = ""


def _canonical_cluster_candidate_union(
    candidates: Iterable[_PrefixCandidate],
    *,
    per_shard_limit: int | None,
) -> list[_PrefixCandidate]:
    """Deduplicate a clustered target's member hits, then cap per shard.

    The legacy scan limit is a per-feature, per-shard budget.  A clustered
    representative is one logical target, so applying that budget to every
    member independently would give larger clusters more verification chances.
    Member scans therefore run uncapped and this shared union boundary applies
    the configured limit only after canonical location deduplication.
    """

    if per_shard_limit is not None and per_shard_limit < 0:
        raise AxBenchBaseError(
            "cluster candidate per-shard limit must be None or non-negative"
        )
    by_location: dict[tuple[int, int, int], _PrefixCandidate] = {}
    for candidate in candidates:
        key = (candidate.shard_idx, candidate.doc_idx, candidate.seg_idx)
        by_location.setdefault(key, candidate)

    result: list[_PrefixCandidate] = []
    accepted_by_shard: dict[int, int] = {}
    for key in sorted(by_location):
        shard_idx = key[0]
        accepted = accepted_by_shard.get(shard_idx, 0)
        if per_shard_limit is not None and accepted >= per_shard_limit:
            continue
        result.append(by_location[key])
        accepted_by_shard[shard_idx] = accepted + 1
    return result


class _CandidatePrefixLookup(Protocol):
    """Minimal concept-owned candidate lookup used by ``_verify_concept``."""

    def candidates_for_feature(self, feature_id: int) -> list[_PrefixCandidate]:
        """Return candidates for ``feature_id`` in canonical scan order.

        Implementations MUST return a freshly materialized, concept-owned list
        on every call. The caller may shuffle that list in place and release it
        after the concept without mutating fixed run state or retaining prior
        concepts' prefix payloads.
        """
        ...


@dataclass(frozen=True)
class _LoadedCandidateShard:
    """Fixed decoded validation-shard state reused by candidate lookups."""

    shard_idx: int
    feature_shard_path: Path
    parquet_path: Path
    features: dict[str, Any]
    texts: list[str]


@dataclass(frozen=True)
class _VerifiedNegativePrefix:
    """A candidate prefix that passed the N-seed consistency check.

    Field layout matches the parquet schema column order so
    ``dataclasses.asdict`` yields a dict pyarrow can consume directly.
    """

    concept_id: int
    prefix_text: str
    prefix_source_shard: int
    prefix_source_doc: int
    prefix_activating_seg_idx: int
    verification_continuations: list[str]
    verification_seeds: list[int]
    verification_percentiles: list[float]


@dataclass(frozen=True)
class _RejectedConcept:
    """A Stage 0c candidate concept rejected by LLM QC or verification."""

    candidate_index: int
    feature_index: int
    description: str
    reason: RejectedConceptReason
    n_verified_found: int | None
    llm_judge_score: int | None




# ---------------------------------------------------------------------------
# Stage 0c checkpoint identity helpers
# ---------------------------------------------------------------------------


# The explicit legacy/local mode stays on schema 8 so
# ``verification_api_model=null`` does not invalidate or byte-change existing
# Stage 0c checkpoints/artifacts. API generation is a distinct semantic
# algorithm and uses schema 9.
_STAGE0C_CHECKPOINT_SCHEMA_VERSION = STAGE0C_LOCAL_CHECKPOINT_SCHEMA_VERSION


#: Fingerprint fields that older Stage 0c checkpoints persisted but that the
#: current canonical fingerprint no longer computes. A stored checkpoint may
#: still carry them; their presence is ignored and their VALUE is never
#: compared, because the field is no longer part of Stage 0c identity.
#:
#: This exists so raising `max_concepts` on an existing run resumes instead of
#: restarting. Without it, the strict field-set equality check below would
#: reject every checkpoint written before the field was dropped.
_LEGACY_FINGERPRINT_KEYS = frozenset({"max_concepts"})


def _stage0c_checkpoint_schema_version(cfg: AxBenchBaseConfig) -> int:
    if cfg.verification_api_model is None:
        return _STAGE0C_CHECKPOINT_SCHEMA_VERSION
    return STAGE0C_API_CHECKPOINT_SCHEMA_VERSION


def _stage0c_extractor_identity(
    cfg: AxBenchBaseConfig,
    *,
    precomputed_clustered_fields: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the canonical extractor identity used by Stage 0c.

    ``feature_extraction_model_id`` is stored in its effective form so
    ``None`` and an explicit value equal to ``neuronpedia_model_id`` describe
    the same extractor.  Keeping this normalization in one helper prevents
    partial-checkpoint fingerprints and completed-artifact metadata from
    drifting apart.

    ``precomputed_clustered_fields`` (PR #739 FU-D-2): when supplied
    (typically the dict ``cluster_artifacts._clustered_fingerprint_fields_
    from_identity`` derives ONCE from the dispatcher's startup collective
    vote), used verbatim in place of a fresh
    :func:`cluster_artifacts.clustered_stage0c_fingerprint_fields` read.
    ``None`` (the default) preserves the exact pre-FU-D-2 fresh-read
    behavior -- every caller that has not been updated to thread a cached
    value (including the intentionally-live re-reads ``stage0c_runner.run_merge`` /
    ``negative_prefixes._resume_if_artifacts_complete`` keep -- see their
    own comments for why) is unaffected byte-for-byte.
    """
    fingerprint = {
        "neuronpedia_model_id": cfg.neuronpedia_model_id,
        "feature_extraction_model_id": effective_extraction_model_id(cfg),
        "neuronpedia_source": cfg.neuronpedia_source,
    }
    fingerprint.update(
        clustered_stage0c_fingerprint_fields(cfg)
        if precomputed_clustered_fields is None
        else dict(precomputed_clustered_fields)
    )
    return fingerprint


def _stage0c_concept_examples_qc_config(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any] | None:
    """Return every cfg-derived input to Stage 0c examples QC.

    Disabled QC is normalized to ``None`` so dormant tuning values do not
    invalidate checkpoints. When enabled, retaining the full judge dataclass
    and fetch-worker input avoids silently reusing judgments produced under a
    different provider, model, request policy, prompt example count, threshold,
    survivor cap, or fetch concurrency.
    """
    if not cfg.enable_concept_examples_qc:
        return None
    return {
        "n_activation_examples": cfg.n_activation_examples,
        "activation_example_judge_min_score": (
            cfg.activation_example_judge_min_score
        ),
        "concept_examples_qc_target_surviving": (
            cfg.concept_examples_qc_target_surviving
        ),
        "judge": asdict(cfg.concept_examples_qc_judge),
        "neuronpedia_max_concurrent": cfg.neuronpedia.max_concurrent,
    }


def _compute_checkpoint_fingerprint(
    cfg: AxBenchBaseConfig,
    *,
    precomputed_clustered_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the canonical fingerprint dict for a Stage 0c checkpoint.

    Legacy/local mode deliberately retains the exact schema-8 field set.
    API mode adds one conditional nested identity and uses schema 9. Any
    future semantic identity change must bump the corresponding schema.

    ``precomputed_clustered_fields``: forwarded verbatim to
    :func:`_stage0c_extractor_identity` -- see its own docstring (PR #739
    FU-D-2 threading; ``None`` preserves the pre-FU-D-2 fresh-read path).
    """
    extractor_identity = _stage0c_extractor_identity(
        cfg, precomputed_clustered_fields=precomputed_clustered_fields,
    )
    fingerprint = {
        # This version is part of the canonical fingerprint, not only the
        # partial-checkpoint envelope, so completed parquet metadata also
        # invalidates when Stage 0c's semantic algorithm changes.
        "stage0c_algorithm_version": _stage0c_checkpoint_schema_version(cfg),
        "feature_dir": cfg.feature_dir,
        "feature_statistics_path": cfg.feature_statistics_path,
        "validation_shards": sorted(cfg.validation_shards),
        **extractor_identity,
        "extraction_tokenizer_id": cfg.extraction_tokenizer_id,
        # `max_concepts` is deliberately NOT part of this fingerprint. It is a
        # STOPPING CONDITION ("how many accepted concepts do we want?"), not a
        # semantic property of how any individual candidate is verified. Two
        # runs that differ only in `max_concepts` verify each candidate
        # identically and walk the same ordered candidate pool; the larger run
        # simply walks further. Binding it into identity therefore forced a
        # full re-verification of already-decided candidates whenever the
        # target was raised, discarding valid work for no correctness gain.
        # See `_LEGACY_FINGERPRINT_KEYS` for how pre-existing checkpoints that
        # still carry the field stay valid.
        "n_verified_prefixes_per_concept_target": (
            cfg.n_verified_prefixes_per_concept_target
        ),
        "n_verification_seeds": cfg.n_verification_seeds,
        "min_consistency_passes": cfg.min_consistency_passes,
        "verification_percentile_threshold": (
            cfg.verification_percentile_threshold
        ),
        "verification_length": cfg.verification_length,
        "min_prefix_chars": cfg.min_prefix_chars,
        "max_prefix_chars": cfg.max_prefix_chars,
        "verification_zero_hit_bail_after": (
            cfg.verification_zero_hit_bail_after
        ),
        "verification_candidate_attempt_limit": (
            cfg.verification_candidate_attempt_limit
        ),
        "verification_candidate_scan_limit": (
            cfg.verification_candidate_scan_limit
        ),
        "enable_concept_examples_qc": cfg.enable_concept_examples_qc,
        "concept_examples_qc_config": _stage0c_concept_examples_qc_config(cfg),
        "generation_master_seed": cfg.generation.master_seed,
        "generation_temperature": cfg.generation.temperature,
        "generation_top_k_sampling": cfg.generation.top_k_sampling,
        "qc_batch_size": cfg.qc_batch_size,
        "qc_pipeline": cfg.qc_pipeline,
        "verification_prefill_mode": cfg.verification_prefill_mode,
        "stage0c_multi_gpu_block_size": cfg.stage0c_multi_gpu_block_size,
        "verification_cross_prefix_batch_size": (
            cfg.verification_cross_prefix_batch_size
        ),
    }
    api_config = api_generation_fingerprint_config(cfg)
    if api_config is not None:
        # The local GPU batch size is dormant in API mode. Keep it in the
        # explicit-null schema-8 fingerprint exactly as before, but do not let
        # an irrelevant local-memory knob invalidate API artifacts.
        fingerprint.pop("verification_cross_prefix_batch_size")
        # CONDITIONAL by design: explicit-null local mode retains the exact
        # schema-8 key set and identity.
        fingerprint["verification_api_generation"] = api_config
    return fingerprint


def _semantic_json_identity(payload: object) -> str:
    """SHA-256 of canonical semantic JSON, independent of source formatting."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidate_pool_identity(candidate_concepts: list[CandidateConcept]) -> str:
    """Content hash of the Stage 0c candidate pool in file order.

    Stage 0c indexes ``candidate_concepts`` positionally and its examples QC
    judges each row's description, so both order and per-row content are part
    of the resume contract. Hashing the ordered rows lets a resume detect a
    regenerated pool (changed concept_feature_whitelist_path, min_corpus_count,
    concept_list_seed, or descriptions) and refuse to resume against stale
    accepted concepts.
    """
    return _semantic_json_identity([asdict(c) for c in candidate_concepts])


def concepts_identity(concepts: list[Concept]) -> str:
    """Semantic identity of the ordered, validated final concept list."""
    return _semantic_json_identity([asdict(concept) for concept in concepts])


def rejected_concepts_identity(
    rejected_concepts: list[dict[str, Any]],
) -> str:
    """Semantic identity of ordered, validated Stage 0c rejection records."""
    return _semantic_json_identity(rejected_concepts)


def concept_examples_qc_report_identity(
    report: dict[str, Any] | None,
) -> str:
    """Semantic identity of the QC report, or of the disabled sentinel.

    ``None`` is intentional state: it means examples QC was disabled and no
    report belongs to the completed artifact set. Hashing that sentinel lets
    the canonical parquet bind both enabled and disabled runs explicitly.
    """
    return _semantic_json_identity(report)


def stage0c_fingerprint_identity(fingerprint: dict[str, Any]) -> str:
    """Semantic identity of the full agreed per-rank Stage 0c fingerprint."""
    return _semantic_json_identity(fingerprint)


def _fingerprint_with_pool(
    cfg: AxBenchBaseConfig,
    *,
    candidate_pool_identity: str,
    precomputed_clustered_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Config checkpoint fingerprint plus the candidate-pool identity. Used by
    the live per-rank Stage 0c checkpoint write/validate path so a regenerated
    candidate pool invalidates resume.

    ``precomputed_clustered_fields`` (PR #739 FU-D-2): forwarded verbatim
    to :func:`_compute_checkpoint_fingerprint` -- see its docstring.
    Threaded end-to-end from ``stage0c_runner.dispatch_prepare_negative_
    prefixes``'s startup collective vote, through ``run_single_rank``, to
    both hot/per-rank call sites (``_build_payload``, ``_try_resume_per_
    rank``). ``None`` (the default -- used by ``run_merge`` and
    ``negative_prefixes._resume_if_artifacts_complete``'s own
    intentionally-live re-reads) preserves the exact pre-FU-D-2 fresh-read
    behavior: a fresh read from disk on every call. Any non-``None``
    ``Mapping`` -- including ``{}`` for an unclustered cfg -- is instead
    the startup-resolved cache and is used verbatim with no further I/O
    (owner review of PR #884: this parameter deliberately KEPT its
    Optional default, unlike its now-required callers above, because
    ``None`` here is a real second mode -- live staleness detection --
    not gratuitous nullability).
    """
    return {
        **_compute_checkpoint_fingerprint(
            cfg, precomputed_clustered_fields=precomputed_clustered_fields,
        ),
        "candidate_pool_identity": candidate_pool_identity,
    }



def _base_context_window(base_model: Any) -> int:
    """Return the base SAE LM context window in Gemma tokens."""
    cfg = getattr(base_model, "cfg", None)
    n_ctx = getattr(cfg, "n_ctx", None)
    if n_ctx is None:
        raise AxBenchBaseError(
            "base_model must expose cfg.n_ctx so Stage 0c can pre-filter "
            "candidate prefixes against the Gemma context window"
        )
    return int(n_ctx)



def _tokenizer_stop_token_ids(tokenizer: Any) -> set[int]:
    """Special token IDs that should terminate Gemma verification generation."""
    stop_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            stop_ids.add(value)
    for value in getattr(tokenizer, "all_special_ids", []) or []:
        if isinstance(value, int):
            stop_ids.add(value)
    return stop_ids



def _list_parquets_for_feature_dir(cfg: AxBenchBaseConfig) -> list[Path]:
    """Resolve the parquet data directory from the feature_dir convention.

    Mirrors the helper in ``seed_pools.py`` but is duplicated here so each
    test patches its own module.
    """
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    phase_value = parsed.dataset_phase.value
    data_dir = resolve_data_dir(None, phase_value)
    return [Path(p) for p in list_parquet_files(str(data_dir))]


def _resolve_parquet(parquets: list[Path], shard_idx: int) -> Path:
    if shard_idx < 0 or shard_idx >= len(parquets):
        raise AxBenchBaseError(
            f"validation_shards contains {shard_idx}, but only "
            f"{len(parquets)} parquet shards exist"
        )
    return parquets[shard_idx]


def _resolve_feature_shard(
    feature_shards: list[Path], shard_idx: int, feature_dir: str,
) -> Path:
    target_prefix = f"shard_{shard_idx:05d}_features"
    for p in feature_shards:
        if p.name.startswith(target_prefix):
            return p
    raise AxBenchBaseError(
        f"No feature shard matching parquet index {shard_idx} in {feature_dir}"
    )


# ---------------------------------------------------------------------------
# Candidate scanning
# ---------------------------------------------------------------------------


_SOURCE_BRIDGE_MAX_CHARS = 64
_SOURCE_BRIDGE_SENTENCE_END_RE = re.compile(
    r"[.!?…](?:[\"'”’)\]]*)(?=\s|$)"
)


def _source_bridge_text(full_text: str, seg_end_char: int) -> str:
    """Return a bounded exact source slice that repairs an arbitrary cut.

    Feature segments end at character offsets rather than linguistic
    boundaries, so candidate prefixes routinely stop inside a word or clause.
    API continuations cannot be trusted to guess the missing source text.  The
    bridge therefore comes from the local document itself: prefer the first
    sentence boundary in the next 64 characters, otherwise end at the last
    available whitespace boundary.  No text is synthesized or normalized.
    """
    remaining = full_text[seg_end_char : seg_end_char + _SOURCE_BRIDGE_MAX_CHARS]
    if not remaining:
        return ""

    sentence_end = _SOURCE_BRIDGE_SENTENCE_END_RE.search(remaining)
    if sentence_end is not None:
        end = sentence_end.end()
        while end < len(remaining) and remaining[end].isspace():
            end += 1
        return remaining[:end]

    whitespace_ends = [
        index + 1 for index, character in enumerate(remaining) if character.isspace()
    ]
    if whitespace_ends:
        return remaining[: whitespace_ends[-1]]
    return remaining


def _find_candidate_prefixes(
    cfg: AxBenchBaseConfig,
    shard_idx: int,
    feature_shard_path: Path,
    parquet_path: Path,
    f_c: int,
    base_context_window: int,
) -> list[_PrefixCandidate]:
    """Scan one validation shard for segments where ``f_c`` is active.

    For each matching segment, take the parquet text up to the segment's
    char end as a candidate prefix, clamped to
    ``[cfg.min_prefix_chars, cfg.max_prefix_chars]``. Returns a list of
    :class:`_PrefixCandidate` instances with ``prefix_text``,
    ``doc_idx``, ``seg_idx``, ``shard_idx``, and pre-tokenized
    ``prefix_gemma_tokens``.
    """
    max_prompt_tokens = base_context_window - cfg.verification_length

    with profile_region(
        "stage0c.load_feature_shard_for_candidates",
        shard=shard_idx,
    ):
        feat = load_features(feature_shard_path)
    doc_ids = feat["doc_ids"]
    offsets = feat["offsets"]
    fids_all = feat["feature_ids"]
    seg_ends = feat["segment_char_ends"]
    n_segments = doc_ids.shape[0]

    with profile_region("stage0c.load_shard_text", shard=shard_idx):
        texts = read_parquet_texts(parquet_path)
    with profile_region("stage0c.load_extraction_tokenizer"):
        tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)

    candidates: list[_PrefixCandidate] = []
    n_overflow_dropped = 0
    with profile_region(
        "stage0c.scan_candidate_segments",
        shard=shard_idx,
        feature_id=f_c,
        rows=n_segments,
    ):
        for seg_i in range(n_segments):
            lo = int(offsets[seg_i].item())
            hi = int(offsets[seg_i + 1].item())
            seg_fids = fids_all[lo:hi].tolist()
            if f_c not in seg_fids:
                continue
            doc_idx = int(doc_ids[seg_i].item())
            if doc_idx >= len(texts):
                continue  # defensive: feature shard / parquet row mismatch
            full_text = texts[doc_idx]
            seg_end_char = int(seg_ends[seg_i].item())
            prefix_text_chars = window_prefix_text(
                full_text, seg_end_char, cfg.max_prefix_chars,
            )
            if len(prefix_text_chars) < cfg.min_prefix_chars:
                continue
            prefix_tokens = tokenizer.encode(
                prefix_text_chars, add_special_tokens=True,
            )
            if not prefix_tokens:
                continue
            if len(prefix_tokens) > max_prompt_tokens:
                n_overflow_dropped += 1
                continue
            candidates.append(
                _PrefixCandidate(
                    shard_idx=shard_idx,
                    doc_idx=doc_idx,
                    seg_idx=seg_i,
                    prefix_text=prefix_text_chars,
                    prefix_gemma_tokens=prefix_tokens,
                    source_bridge_text=_source_bridge_text(full_text, seg_end_char),
                )
            )
            if (
                cfg.verification_candidate_scan_limit is not None
                and len(candidates) >= cfg.verification_candidate_scan_limit
            ):
                break
    logger.info(
        f"Candidate scan for shard {shard_idx}, feature {f_c}: "
        f"yielded {len(candidates)} prefixes; dropped "
        f"{n_overflow_dropped} prefixes as overflow "
        f"(> {max_prompt_tokens} tokens)"
    )
    record_counter(
        "stage0c.candidate_prefixes",
        len(candidates),
        shard=shard_idx,
        feature_id=f_c,
    )
    record_counter(
        "stage0c.candidate_prefixes_overflow_dropped",
        n_overflow_dropped,
        shard=shard_idx,
        feature_id=f_c,
    )
    return candidates


class _CandidatePrefixIndex:
    """Lazy per-run candidate index for Stage 0c verification.

    The legacy Stage 0c loop loaded and scanned the same validation shard
    once per concept. This index preserves the legacy candidate order and
    filtering semantics while caching only fixed run state: decoded validation
    shards, their parquet texts, and the extraction tokenizer. Feature-segment
    lookup arrays and fully materialized prefix candidates are concept-owned
    temporaries, so peak candidate memory is bounded by the active concept
    rather than every previously visited concept.

    Construction is intentionally cheap and side-effect free. Shards are
    loaded only when the first post-QC concept reaches verification, so
    all-LLM-QC-rejected runs retain the old "no shard IO" behavior.
    """

    def __init__(
        self,
        cfg: AxBenchBaseConfig,
        parquets: list[Path],
        feature_shards: list[Path],
        *,
        candidate_feature_ids: set[int],
        base_context_window: int,
    ) -> None:
        self._cfg = cfg
        self._parquets = parquets
        self._feature_shards = feature_shards
        self._candidate_feature_ids = {
            int(feature_id) for feature_id in candidate_feature_ids
        }
        self._base_context_window = int(base_context_window)
        self._loaded_shards: dict[int, _LoadedCandidateShard] = {}
        self._tokenizer: Any | None = None
        # Per-index (cross-call) heartbeat state for
        # _materialize_shard_candidates. Tracking on the instance — rather
        # than as a function-local — means many small (<1000 segments)
        # features still produce heartbeats once aggregate scan crosses
        # the segment threshold, and time-based fallback fires for slow
        # scans that never approach the count threshold.
        self._heartbeat_segments_scanned = 0
        self._heartbeat_emitted_count = 0
        self._heartbeat_last_emit_at = time.monotonic()

    def candidates_for_feature(self, feature_id: int) -> list[_PrefixCandidate]:
        """Return candidates for ``feature_id`` in canonical scan order.

        "Canonical order" = ``(validation_shard_index ASC, doc_id ASC,
        seg_idx ASC)`` — deterministic across runs because:

        - Validation shards are traversed in sorted order, matching the
          canonical checkpoint fingerprint regardless of config-list order.
        - :meth:`_materialize_shard_candidates` walks the shard's
          parquet rows in file order, which pyarrow guarantees matches
          write order.

        The returned list and all of its prefix/token payloads are freshly
        materialized for this call. The caller owns that list, may shuffle it
        in place, and should release it at the concept boundary.
        """
        f_c = int(feature_id)
        with profile_region(
            "stage0c.candidate_index_lookup",
            feature_id=f_c,
            validation_shards=len(self._cfg.validation_shards),
        ):
            all_candidates: list[_PrefixCandidate] = []
            for shard_idx in sorted(self._cfg.validation_shards):
                shard = self._load_shard(shard_idx)
                all_candidates.extend(
                    self._materialize_shard_candidates(shard, f_c)
                )

        record_counter(
            "stage0c.candidate_prefix_cache_misses",
            1,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes_by_feature",
            len(all_candidates),
            feature_id=f_c,
        )
        return all_candidates

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            with profile_region("stage0c.load_extraction_tokenizer"):
                self._tokenizer = get_extraction_tokenizer(
                    self._cfg.extraction_tokenizer_id,
                )
            record_counter("stage0c.extraction_tokenizer_loads", 1)
        return self._tokenizer

    def _load_shard(self, shard_idx: int) -> _LoadedCandidateShard:
        cached = self._loaded_shards.get(shard_idx)
        if cached is not None:
            record_counter(
                "stage0c.candidate_index_shard_cache_hits",
                1,
                shard=shard_idx,
            )
            return cached

        parquet_path = _resolve_parquet(self._parquets, shard_idx)
        feature_shard_path = _resolve_feature_shard(
            self._feature_shards, shard_idx, self._cfg.feature_dir,
        )
        logger.info(
            "[stage0c] lazy first load of shard {idx}: "
            "parquet={pq} feature_shard={fs}",
            idx=shard_idx,
            pq=parquet_path.name,
            fs=feature_shard_path.name,
        )
        with profile_region(
            "stage0c.candidate_index_load_shard",
            shard=shard_idx,
            candidate_features=len(self._candidate_feature_ids),
        ):
            with profile_region(
                "stage0c.load_feature_shard_for_candidates",
                shard=shard_idx,
            ):
                features = load_features(feature_shard_path)
            with profile_region("stage0c.load_shard_text", shard=shard_idx):
                texts = read_parquet_texts(parquet_path)
        logger.info(
            "[stage0c] shard {idx} loaded: {n_segments} segment(s), "
            "{n_texts} document(s)",
            idx=shard_idx,
            n_segments=int(features["doc_ids"].shape[0]),
            n_texts=len(texts),
        )

        shard = _LoadedCandidateShard(
            shard_idx=shard_idx,
            feature_shard_path=feature_shard_path,
            parquet_path=parquet_path,
            features=features,
            texts=texts,
        )
        self._loaded_shards[shard_idx] = shard
        record_counter(
            "stage0c.candidate_index_shard_cache_misses",
            1,
            shard=shard_idx,
        )
        return shard

    def _lookup_feature_segments(
        self,
        shard: _LoadedCandidateShard,
        f_c: int,
    ) -> array:
        features = shard.features
        offsets = features["offsets"]
        fids_all = features["feature_ids"]
        doc_ids = features["doc_ids"]
        n_segments = int(doc_ids.shape[0])
        n_feature_entries = int(fids_all.numel())

        if int(offsets.numel()) != n_segments + 1:
            raise AxBenchBaseError(
                "Candidate index requires CSR offsets length "
                f"{n_segments + 1}, got {int(offsets.numel())} for "
                f"{shard.feature_shard_path}"
            )
        if int(offsets[0].item()) != 0:
            raise AxBenchBaseError(
                "Candidate index requires CSR offsets[0] == 0 for "
                f"{shard.feature_shard_path}, got {int(offsets[0].item())}"
            )
        if not bool((offsets[:-1] <= offsets[1:]).all().item()):
            raise AxBenchBaseError(
                "Candidate index requires monotonic CSR offsets for "
                f"{shard.feature_shard_path}"
            )
        if int(offsets[-1].item()) != n_feature_entries:
            raise AxBenchBaseError(
                "Candidate index requires CSR offsets[-1] == len(feature_ids) "
                f"for {shard.feature_shard_path}; got "
                f"{int(offsets[-1].item())} vs {n_feature_entries}"
            )

        with profile_region(
            "stage0c.candidate_index_lookup_segments",
            shard=shard.shard_idx,
            feature_id=f_c,
            rows=n_segments,
            feature_entries=n_feature_entries,
        ):
            matches = t.nonzero(fids_all == int(f_c), as_tuple=False).flatten()
            if matches.numel() == 0:
                return array("I")
            match_segments = t.searchsorted(
                offsets[1:],
                matches.to(offsets.device),
                # A match at offsets[i + 1] belongs to segment i + 1.
                right=True,
            )
            match_segments = t.unique_consecutive(match_segments)
            return array("I", (int(seg_i) for seg_i in match_segments.tolist()))

    def _materialize_shard_candidates(
        self,
        shard: _LoadedCandidateShard,
        f_c: int,
    ) -> list[_PrefixCandidate]:
        cfg = self._cfg
        max_prompt_tokens = self._base_context_window - cfg.verification_length
        tokenizer = self._get_tokenizer()
        doc_ids = shard.features["doc_ids"]
        seg_ends = shard.features["segment_char_ends"]
        # This compact array is local to the active feature lookup. Retaining
        # one such array per feature provided no production reuse because
        # candidate features and schema-v1 cluster memberships are unique.
        segment_indices = self._lookup_feature_segments(shard, f_c)
        record_counter(
            "stage0c.candidate_index_feature_cache_misses",
            1,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_index_feature_matches",
            len(segment_indices),
            shard=shard.shard_idx,
            feature_id=f_c,
        )

        candidates: list[_PrefixCandidate] = []
        n_missing_doc = 0
        n_short_prefix = 0
        n_empty_tokenization = 0
        n_overflow_dropped = 0
        n_considered = 0
        with profile_region(
            "stage0c.candidate_filter_and_tokenize",
            shard=shard.shard_idx,
            feature_id=f_c,
            rows=len(segment_indices),
        ):
            for seg_i in segment_indices:
                n_considered += 1
                self._heartbeat_segments_scanned += 1
                # Heartbeat fires when aggregate (cross-call) segments
                # scanned crosses the next _HEARTBEAT_SEGMENTS multiple,
                # OR when wall-time since the last emit exceeds
                # _HEARTBEAT_SECONDS — whichever comes first. Gated on
                # rank 0 so torchrun workers stay silent without
                # depending on the loguru sink configuration.
                next_count_milestone = (
                    self._heartbeat_emitted_count + 1
                ) * _HEARTBEAT_SEGMENTS
                hit_count = (
                    self._heartbeat_segments_scanned >= next_count_milestone
                )
                now = time.monotonic()
                hit_time = (
                    (now - self._heartbeat_last_emit_at) >= _HEARTBEAT_SECONDS
                )
                if hit_count or hit_time:
                    self._heartbeat_emitted_count += 1
                    self._heartbeat_last_emit_at = now
                    if is_log_rank():
                        logger.info(
                            "[stage0c] materialize heartbeat: "
                            "shard {idx} feature {fid}: "
                            "considered={c} matched={m} dropped={d}",
                            idx=shard.shard_idx,
                            fid=f_c,
                            c=self._heartbeat_segments_scanned,
                            m=len(candidates),
                            d=n_overflow_dropped + n_short_prefix
                              + n_empty_tokenization + n_missing_doc,
                        )
                doc_idx = int(doc_ids[seg_i].item())
                if doc_idx >= len(shard.texts):
                    n_missing_doc += 1
                    continue  # defensive: feature shard / parquet row mismatch
                full_text = shard.texts[doc_idx]
                seg_end_char = int(seg_ends[seg_i].item())
                prefix_text_chars = window_prefix_text(
                    full_text, seg_end_char, cfg.max_prefix_chars,
                )
                if len(prefix_text_chars) < cfg.min_prefix_chars:
                    n_short_prefix += 1
                    continue
                prefix_tokens = tokenizer.encode(
                    prefix_text_chars,
                    add_special_tokens=True,
                )
                if not prefix_tokens:
                    n_empty_tokenization += 1
                    continue
                if len(prefix_tokens) > max_prompt_tokens:
                    n_overflow_dropped += 1
                    continue
                candidates.append(
                    _PrefixCandidate(
                        shard_idx=shard.shard_idx,
                        doc_idx=doc_idx,
                        seg_idx=int(seg_i),
                        prefix_text=prefix_text_chars,
                        prefix_gemma_tokens=prefix_tokens,
                        source_bridge_text=_source_bridge_text(full_text, seg_end_char),
                    )
                )
                if (
                    cfg.verification_candidate_scan_limit is not None
                    and len(candidates) >= cfg.verification_candidate_scan_limit
                ):
                    break

        logger.info(
            f"Candidate scan for shard {shard.shard_idx}, feature {f_c}: "
            f"yielded {len(candidates)} prefixes; dropped "
            f"{n_overflow_dropped} prefixes as overflow "
            f"(> {max_prompt_tokens} tokens)"
        )
        record_counter(
            "stage0c.candidate_segments_considered",
            n_considered,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes",
            len(candidates),
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes_overflow_dropped",
            n_overflow_dropped,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes_missing_doc_dropped",
            n_missing_doc,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes_short_dropped",
            n_short_prefix,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        record_counter(
            "stage0c.candidate_prefixes_empty_tokenization_dropped",
            n_empty_tokenization,
            shard=shard.shard_idx,
            feature_id=f_c,
        )
        return candidates


# ---------------------------------------------------------------------------
# Verification loop
# ---------------------------------------------------------------------------


def _verification_seeds_for_candidate(
    cfg: AxBenchBaseConfig,
    *,
    concept_id: int,
    candidate: _PrefixCandidate,
) -> list[int]:
    return [
        cfg.generation.master_seed
        + _VERIFY_SEED_OFFSET
        + concept_id * _CONCEPT_STRIDE
        + vseed_idx * _VSEED_INCREMENT
        + candidate.doc_idx
        for vseed_idx in range(cfg.n_verification_seeds)
    ]


def _generate_token_lists_for_candidate(
    cfg: AxBenchBaseConfig,
    candidate: _PrefixCandidate,
    base_model: Any,
    *,
    seeds: list[int],
    stop_token_ids: set[int],
) -> list[list[int]]:
    if (
        VerificationPrefillMode(cfg.verification_prefill_mode)
        is VerificationPrefillMode.SHARED
    ):
        return cross_prefix_batched_generate_seeds(
            base_model,
            [candidate.prefix_gemma_tokens],
            seed_rows=[seeds],
            max_new_tokens=cfg.verification_length,
            temperature=cfg.generation.temperature,
            top_k=cfg.generation.top_k_sampling,
            stop_token_ids=stop_token_ids,
            prefill_mode=cfg.verification_prefill_mode,
        )[0]
    return batched_generate_seeds(
        base_model,
        candidate.prefix_gemma_tokens,
        seeds=seeds,
        max_new_tokens=cfg.verification_length,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k_sampling,
        stop_token_ids=stop_token_ids,
    )


def _generate_token_lists_for_candidate_batch(
    cfg: AxBenchBaseConfig,
    candidates: list[_PrefixCandidate],
    base_model: Any,
    *,
    seed_rows: list[list[int]],
    stop_token_ids: set[int],
) -> list[list[list[int]]]:
    return cross_prefix_batched_generate_seeds(
        base_model,
        [candidate.prefix_gemma_tokens for candidate in candidates],
        seed_rows=seed_rows,
        max_new_tokens=cfg.verification_length,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k_sampling,
        stop_token_ids=stop_token_ids,
        prefill_mode=cfg.verification_prefill_mode,
    )


def _verified_prefix_from_token_lists(
    cfg: AxBenchBaseConfig,
    candidate: _PrefixCandidate,
    f_c: int,
    concept_id: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    tokenizer: Any,
    seeds: list[int],
    token_lists: list[list[int]],
    release_memory: bool = True,
    cluster_target: ClusterFeatureTarget | None = None,
) -> _VerifiedNegativePrefix | None:
    raw_threshold = percentile_fraction_to_points(
        cfg.verification_percentile_threshold,
        field_name="verification_percentile_threshold",
    )
    continuations: list[str] = []
    percentile_results: list[float] = []
    seeds_used: list[int] = []
    for s, cont_tokens in zip(seeds, token_lists, strict=True):
        with profile_region(
            "stage0c.verify_candidate_decode",
            concept_id=concept_id,
            token_count=len(cont_tokens),
        ):
            cont_text = tokenizer.decode(cont_tokens)
        continuations.append(cont_text)
        seeds_used.append(s)
        # compute_fc_percentiles returns (seed_pct, out_pct) on a 0-100 scale.
        # We only care about the continuation percentile for verification.
        with profile_region(
            "stage0c.verify_candidate_compute_percentile",
            concept_id=concept_id,
            feature_id=f_c,
        ):
            cluster_kwargs: dict[str, Any] = (
                {} if cluster_target is None else {"cluster_target": cluster_target}
            )
            try:
                _, out_pct = compute_fc_percentiles(
                    candidate.prefix_text, cont_text, f_c,
                    base_model, base_sae, percentile_table,
                    sae_layer=sae_layer,
                    tokenizer_id=cfg.extraction_tokenizer_id,
                    release_memory=release_memory,
                    **cluster_kwargs,
                )
            except TypeError as exc:
                # Some tests monkeypatch compute_fc_percentiles with an exact
                # pre-release_memory signature. Keep those monkeypatch sites
                # stable while letting production Stage 0c defer cleanup to the
                # cross-prefix batch boundary.
                if "release_memory" not in str(exc):
                    raise
                _, out_pct = compute_fc_percentiles(
                    candidate.prefix_text, cont_text, f_c,
                    base_model, base_sae, percentile_table,
                    sae_layer=sae_layer,
                    tokenizer_id=cfg.extraction_tokenizer_id,
                    **cluster_kwargs,
                )
        percentile_results.append(float(out_pct))

    n_passed = sum(p > raw_threshold for p in percentile_results)
    record_counter(
        "stage0c.verification_attempts",
        len(percentile_results),
        concept_id=concept_id,
    )
    if n_passed < cfg.min_consistency_passes:
        return None

    # Keep only the passing continuations / seeds / percentiles.
    kept_continuations: list[str] = []
    kept_seeds: list[int] = []
    kept_percentiles: list[float] = []
    for cont, seed_used, pct in zip(
        continuations, seeds_used, percentile_results, strict=True,
    ):
        if pct > raw_threshold:
            kept_continuations.append(cont)
            kept_seeds.append(seed_used)
            kept_percentiles.append(pct)

    return _VerifiedNegativePrefix(
        concept_id=concept_id,
        prefix_text=candidate.prefix_text,
        prefix_source_shard=candidate.shard_idx,
        prefix_source_doc=candidate.doc_idx,
        prefix_activating_seg_idx=candidate.seg_idx,
        verification_continuations=kept_continuations,
        verification_seeds=kept_seeds,
        verification_percentiles=kept_percentiles,
    )


def _verify_candidate(
    cfg: AxBenchBaseConfig,
    candidate: _PrefixCandidate,
    f_c: int,
    concept_id: int,
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    cluster_target: ClusterFeatureTarget | None = None,
) -> _VerifiedNegativePrefix | None:
    """Run N-seed verification on a candidate prefix.

    Returns ``None`` if the candidate fails the consistency check;
    otherwise returns a :class:`_VerifiedNegativePrefix` with the
    passing continuations and their seeds/percentiles.

    The continuation percentile threshold is
    ``cfg.verification_percentile_threshold * 100`` because
    :func:`compute_fc_percentiles` returns raw percentiles in
    ``[0.5, 99.5]`` while the config stores the threshold in ``[0, 1]``.

    The comparison is **strict** (``p > raw_threshold``). Equality
    with the threshold is treated as a miss. This matters at
    ``threshold = 0.0`` specifically: ``compute_fc_percentiles``
    returns exactly ``0.0`` only when the feature literally never
    fires in the continuation (empty activation or all-zero max),
    which is the "truly inactive" case operators want to exclude.
    An inclusive (``>=``) check would admit these cases and produce
    meaningless "verified" rows for features that never actually
    activated. See PR history for the 175k-candidate pathological
    case that surfaced this.
    """
    with profile_region("stage0c.verify_candidate_tokenizer"):
        tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
        stop_token_ids = _tokenizer_stop_token_ids(tokenizer)

    seeds = _verification_seeds_for_candidate(
        cfg, concept_id=concept_id, candidate=candidate,
    )

    with profile_region(
        "stage0c.verify_candidate_batched_generate",
        concept_id=concept_id,
        n_seeds=len(seeds),
    ):
        token_lists = _generate_token_lists_for_candidate(
            cfg,
            candidate,
            base_model,
            seeds=seeds,
            stop_token_ids=stop_token_ids,
        )

    cluster_kwargs: dict[str, Any] = (
        {} if cluster_target is None else {"cluster_target": cluster_target}
    )
    return _verified_prefix_from_token_lists(
        cfg,
        candidate,
        f_c,
        concept_id,
        base_model,
        base_sae,
        percentile_table,
        sae_layer=sae_layer,
        tokenizer=tokenizer,
        seeds=seeds,
        token_lists=token_lists,
        **cluster_kwargs,
    )


def _generate_candidate_batch_for_verification(
    cfg: AxBenchBaseConfig,
    candidates: list[_PrefixCandidate],
    concept_id: int,
    base_model: Any,
) -> tuple[Any, list[list[int]], list[list[list[int]]]]:
    if not candidates:
        tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
        return tokenizer, [], []
    with profile_region("stage0c.verify_candidate_tokenizer"):
        tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
        stop_token_ids = _tokenizer_stop_token_ids(tokenizer)
    seed_rows = [
        _verification_seeds_for_candidate(
            cfg, concept_id=concept_id, candidate=candidate,
        )
        for candidate in candidates
    ]
    with profile_region(
        "stage0c.verify_candidate_cross_prefix_batched_generate",
        concept_id=concept_id,
        n_prefixes=len(candidates),
        n_seeds=sum(len(seeds) for seeds in seed_rows),
    ):
        token_rows = _generate_token_lists_for_candidate_batch(
            cfg,
            candidates,
            base_model,
            seed_rows=seed_rows,
            stop_token_ids=stop_token_ids,
        )
    return tokenizer, seed_rows, token_rows


def _verify_concept(
    cfg: AxBenchBaseConfig,
    concept: Concept,
    parquets: list[Path],
    feature_shards: list[Path],
    base_model: Any,
    base_sae: Any,
    percentile_table: t.Tensor,
    *,
    sae_layer: int,
    base_context_window: int,
    candidate_index: _CandidatePrefixLookup | None = None,
    api_generator: Stage0cApiGenerator | None = None,
    cluster_target: ClusterFeatureTarget | None = None,
) -> list[_VerifiedNegativePrefix]:
    """Run candidate scan + N-seed verification for one concept.

    Returns the verified prefixes found for this candidate. Stage 0c's
    entrypoint decides whether this is enough to accept the candidate
    into the final ``concepts.json``.
    """
    f_c = concept.feature_index
    scan_started_at = time.monotonic()
    if candidate_index is not None:
        all_candidates = candidate_index.candidates_for_feature(f_c)
    elif cluster_target is None:
        # Preserve the legacy scalar discovery body literally.
        all_candidates = []
        for shard_idx in sorted(cfg.validation_shards):
            parquet_path = _resolve_parquet(parquets, shard_idx)
            feature_shard_path = _resolve_feature_shard(
                feature_shards, shard_idx, cfg.feature_dir,
            )
            shard_candidates = _find_candidate_prefixes(
                cfg,
                shard_idx,
                feature_shard_path,
                parquet_path,
                f_c,
                base_context_window,
            )
            all_candidates.extend(shard_candidates)
    else:
        all_candidates = []
        uncapped_cfg = (
            cfg
            if cfg.verification_candidate_scan_limit is None
            else cfg.replace(verification_candidate_scan_limit=None)
        )
        for shard_idx in sorted(cfg.validation_shards):
            parquet_path = _resolve_parquet(parquets, shard_idx)
            feature_shard_path = _resolve_feature_shard(
                feature_shards, shard_idx, cfg.feature_dir,
            )
            member_candidates: list[_PrefixCandidate] = []
            for member_id in cluster_target.allowed_member_ids:
                member_candidates.extend(
                    _find_candidate_prefixes(
                        uncapped_cfg,
                        shard_idx,
                        feature_shard_path,
                        parquet_path,
                        member_id,
                        base_context_window,
                    )
                )
            all_candidates.extend(
                _canonical_cluster_candidate_union(
                    member_candidates,
                    per_shard_limit=cfg.verification_candidate_scan_limit,
                )
            )
    scan_elapsed = time.monotonic() - scan_started_at
    logger.info(
        f"Concept {concept.concept_id} (feature {f_c}): "
        f"candidate scan yielded {len(all_candidates)} prefixes "
        f"across {len(cfg.validation_shards)} validation shard(s) "
        f"(scan elapsed {scan_elapsed:.1f}s)"
    )
    record_counter(
        "stage0c.concept_candidate_prefixes",
        len(all_candidates),
        concept_id=concept.concept_id,
        feature_id=f_c,
    )
    record_counter(
        "stage0c.concept_candidate_scan_seconds",
        scan_elapsed,
        concept_id=concept.concept_id,
        feature_id=f_c,
    )

    rng = random.Random(
        cfg.generation.master_seed + _VERIFY_SEED_OFFSET + concept.concept_id,
    )
    rng.shuffle(all_candidates)
    candidates_to_verify = all_candidates
    if (
        cfg.verification_candidate_attempt_limit is not None
        and len(candidates_to_verify) > cfg.verification_candidate_attempt_limit
    ):
        candidates_to_verify = candidates_to_verify[
            : cfg.verification_candidate_attempt_limit
        ]
        logger.info(
            f"Concept {concept.concept_id}: applying "
            f"verification_candidate_attempt_limit="
            f"{cfg.verification_candidate_attempt_limit}; "
            f"verifying {len(candidates_to_verify)}/{len(all_candidates)} "
            "candidate prefixes"
        )
    record_counter(
        "stage0c.concept_candidates_to_verify",
        len(candidates_to_verify),
        concept_id=concept.concept_id,
        feature_id=f_c,
    )

    target = cfg.n_verified_prefixes_per_concept_target
    verify_started_at = time.monotonic()
    last_progress_at = verify_started_at
    verified: list[_VerifiedNegativePrefix] = []
    n_tried = 0
    n_passed_observed = 0
    first_hit_attempt: int | None = None
    candidate_pos = 0
    stop_verification = False
    cross_prefix_batch_size = int(
        cfg.verification_cross_prefix_batch_size
    )
    api_prefix_wave_size = int(
        cfg.verification_api_prefix_wave_size
    )

    while (
        candidate_pos < len(candidates_to_verify)
        and len(verified) < target
        and not stop_verification
    ):
        cluster_kwargs: dict[str, Any] = (
            {} if cluster_target is None else {"cluster_target": cluster_target}
        )
        if api_generator is not None:
            # Provider calls are cheap to queue relative to a local Gemma
            # decode. Deliberately allow a speculative wave larger than the
            # remaining accepted-prefix target so low-hit concepts can keep
            # all API workers occupied. Every generated candidate is scored
            # and counted; only the first ``target`` passes in canonical
            # candidate order are retained.
            chunk_limit = max(1, api_prefix_wave_size)
            bail_after = cfg.verification_zero_hit_bail_after
            if bail_after is not None and not verified:
                bail_remaining = bail_after - n_tried
                if bail_remaining <= 0:
                    stop_verification = True
                    break
                chunk_limit = min(chunk_limit, bail_remaining)
            chunk = candidates_to_verify[
                candidate_pos : candidate_pos + chunk_limit
            ]
            seed_rows = [
                _verification_seeds_for_candidate(
                    cfg,
                    concept_id=concept.concept_id,
                    candidate=candidate,
                )
                for candidate in chunk
            ]
            result_rows = []
            with profile_region(
                "stage0c.verify_candidate_api_batch",
                concept_id=concept.concept_id,
                n_prefixes=len(chunk),
                n_requests=sum(len(seeds) for seeds in seed_rows),
            ):
                # LIVE CLUSTER MEMBER-CYCLE WIRING (PR #739 Track C5):
                # ``cluster_target`` supplies the representative and its
                # canonical mask-allowed members. ``proposal_sample_offset``
                # is this wave's global flattened (candidate, verification
                # seed) starting index; multiplying the global candidate
                # boundary by the fixed seed width makes member assignment
                # invariant to how candidates are partitioned into API waves.
                api_cluster_kwargs: dict[str, Any] = {}
                if cluster_target is not None:
                    api_cluster_kwargs = {
                        "cluster_target": cluster_target,
                        "proposal_sample_offset": (
                            candidate_pos * cfg.n_verification_seeds
                        ),
                    }
                token_rows = api_generator.generate_token_rows(
                    feature_id=f_c,
                    candidates=chunk,
                    seed_rows=seed_rows,
                    **api_cluster_kwargs,
                )
                for candidate, seeds, token_lists in zip(
                    chunk,
                    seed_rows,
                    token_rows,
                    strict=True,
                ):
                    result_rows.append(
                        _verified_prefix_from_token_lists(
                            cfg,
                            candidate,
                            f_c,
                            concept.concept_id,
                            base_model,
                            base_sae,
                            percentile_table,
                            sae_layer=sae_layer,
                            tokenizer=api_generator.tokenizer,
                            seeds=seeds,
                            token_lists=token_lists,
                            release_memory=False,
                            **cluster_kwargs,
                        )
                    )
            candidate_pos += len(chunk)
        elif cross_prefix_batch_size <= 1:
            cand = candidates_to_verify[candidate_pos]
            with profile_region(
                "stage0c.verify_candidate",
                concept_id=concept.concept_id,
            ):
                result_rows = [
                    _verify_candidate(
                        cfg, cand, f_c, concept.concept_id,
                        base_model, base_sae, percentile_table,
                        sae_layer=sae_layer,
                        **cluster_kwargs,
                    )
                ]
            candidate_pos += 1
        else:
            chunk_limit = min(cross_prefix_batch_size, target - len(verified))
            bail_after = cfg.verification_zero_hit_bail_after
            if bail_after is not None and not verified:
                bail_remaining = bail_after - n_tried
                if bail_remaining <= 0:
                    stop_verification = True
                    break
                chunk_limit = min(chunk_limit, bail_remaining)
            chunk = candidates_to_verify[
                candidate_pos : candidate_pos + chunk_limit
            ]
            result_rows = []
            with profile_region(
                "stage0c.verify_candidate_cross_prefix_batch",
                concept_id=concept.concept_id,
                n_prefixes=len(chunk),
            ):
                tokenizer, seed_rows, token_rows = (
                    _generate_candidate_batch_for_verification(
                        cfg,
                        chunk,
                        concept.concept_id,
                        base_model,
                    )
                )
                for candidate, seeds, token_lists in zip(
                    chunk, seed_rows, token_rows, strict=True,
                ):
                    result_rows.append(
                        _verified_prefix_from_token_lists(
                            cfg,
                            candidate,
                            f_c,
                            concept.concept_id,
                            base_model,
                            base_sae,
                            percentile_table,
                            sae_layer=sae_layer,
                            tokenizer=tokenizer,
                            seeds=seeds,
                            token_lists=token_lists,
                            release_memory=False,
                            **cluster_kwargs,
                        )
                    )
            candidate_pos += len(chunk)

        for row in result_rows:
            if api_generator is None and len(verified) >= target:
                # Preserve the legacy local path exactly: a local batch may
                # already have computed later rows, but they never enter the
                # attempt counters once the retained target is reached.
                stop_verification = True
                break
            n_tried += 1
            if row is not None:
                n_passed_observed += 1
                if first_hit_attempt is None:
                    first_hit_attempt = n_tried
                if api_generator is not None:
                    if len(verified) < target:
                        verified.append(row)
                else:
                    verified.append(row)

            bail_after = cfg.verification_zero_hit_bail_after
            if (
                bail_after is not None
                and n_passed_observed == 0
                and n_tried >= bail_after
            ):
                record_counter(
                    "stage0c.verification_zero_hit_bailouts",
                    1,
                    concept_id=concept.concept_id,
                    feature_id=f_c,
                )
                logger.info(
                    "[Stage 0c] Concept {concept_id} "
                    "(feature {feature_id}): zero-hit bailout triggered after "
                    "{n_tried} verification attempts with zero verified "
                    "prefixes (threshold={bail_after}). Treating as "
                    "not_enough_verified_prefixes.",
                    concept_id=concept.concept_id,
                    feature_id=f_c,
                    n_tried=n_tried,
                    bail_after=bail_after,
                )
                stop_verification = True
                break

            now = time.monotonic()
            hit_count_boundary = n_tried % _PROGRESS_EVERY_N_CANDIDATES == 0
            hit_time_boundary = (
                now - last_progress_at
            ) >= _PROGRESS_EVERY_SECONDS
            if hit_count_boundary or hit_time_boundary:
                elapsed = now - verify_started_at
                pass_rate = (
                    (
                        n_passed_observed
                        if api_generator is not None
                        else len(verified)
                    )
                    / n_tried
                    if n_tried
                    else 0.0
                )
                rate = (n_tried / elapsed) if elapsed > 0 else 0.0
                # Extrapolate a completion estimate: either hitting the
                # target (at current pass rate) or exhausting the candidate
                # pool, whichever comes first. Handy when pass rate is low
                # and the user is deciding whether to adjust verification
                # thresholds or the candidate pool.
                remaining_by_target = (
                    (target - len(verified)) / pass_rate
                    if pass_rate > 0 else float("inf")
                )
                remaining_by_pool = len(candidates_to_verify) - n_tried
                eta_candidates = min(remaining_by_target, remaining_by_pool)
                eta_seconds = (
                    (eta_candidates / rate) if rate > 0 else float("inf")
                )
                if api_generator is not None:
                    logger.info(
                        "Concept {concept_id}: "
                        "verified {n_verified}/{target} after "
                        "{n_tried}/{n_candidates} candidates "
                        "(observed_passes={n_passed_observed}, "
                        "pass_rate={pass_rate:.1f}%, "
                        "rate={rate:.2f} cand/s, "
                        "elapsed={elapsed:.0f}s, "
                        "eta≈{eta_seconds:.0f}s)",
                        concept_id=concept.concept_id,
                        n_verified=len(verified),
                        target=target,
                        n_passed_observed=n_passed_observed,
                        n_tried=n_tried,
                        n_candidates=len(candidates_to_verify),
                        pass_rate=pass_rate * 100.0,
                        rate=rate,
                        elapsed=elapsed,
                        eta_seconds=eta_seconds,
                    )
                else:
                    logger.info(
                        "Concept {concept_id}: "
                        "verified {n_verified}/{target} after "
                        "{n_tried}/{n_candidates} candidates "
                        "(pass_rate={pass_rate:.1f}%, "
                        "rate={rate:.2f} cand/s, "
                        "elapsed={elapsed:.0f}s, "
                        "eta≈{eta_seconds:.0f}s)",
                        concept_id=concept.concept_id,
                        n_verified=len(verified),
                        target=target,
                        n_tried=n_tried,
                        n_candidates=len(candidates_to_verify),
                        pass_rate=pass_rate * 100.0,
                        rate=rate,
                        elapsed=elapsed,
                        eta_seconds=eta_seconds,
                    )
                last_progress_at = now

        if api_generator is not None and len(verified) >= target:
            stop_verification = True

    verify_elapsed = time.monotonic() - verify_started_at
    final_pass_rate = (
        (
            n_passed_observed
            if api_generator is not None
            else len(verified)
        )
        / n_tried
        if n_tried
        else 0.0
    )
    record_counter(
        "stage0c.concept_verification_attempts",
        n_tried,
        concept_id=concept.concept_id,
        feature_id=f_c,
    )
    record_counter(
        "stage0c.concept_verified_prefixes",
        len(verified),
        concept_id=concept.concept_id,
        feature_id=f_c,
    )
    if api_generator is not None:
        record_counter(
            "stage0c.concept_verification_passes_observed",
            n_passed_observed,
            concept_id=concept.concept_id,
            feature_id=f_c,
        )
    record_counter(
        "stage0c.concept_verification_seconds",
        verify_elapsed,
        concept_id=concept.concept_id,
        feature_id=f_c,
    )
    record_counter(
        "stage0c.concept_verification_pass_rate",
        final_pass_rate,
        concept_id=concept.concept_id,
        feature_id=f_c,
    )
    if first_hit_attempt is not None:
        record_counter(
            "stage0c.concept_first_hit_attempt",
            first_hit_attempt,
            concept_id=concept.concept_id,
            feature_id=f_c,
        )
    if api_generator is not None:
        logger.info(
            f"Concept {concept.concept_id} done: "
            f"verified {len(verified)}/{target} from {n_tried} tried "
            f"(observed_passes={n_passed_observed}, "
            f"pass_rate={final_pass_rate * 100:.1f}%, "
            f"elapsed={verify_elapsed:.0f}s)"
        )
    else:
        logger.info(
            f"Concept {concept.concept_id} done: "
            f"verified {len(verified)}/{target} from {n_tried} tried "
            f"(pass_rate={final_pass_rate * 100:.1f}%, "
            f"elapsed={verify_elapsed:.0f}s)"
        )

    if not verified:
        msg = (
            f"Concept {concept.concept_id} (feature {f_c}, "
            f"description={concept.description!r}) has ZERO verified "
            f"prefixes after {n_tried} verification attempts. "
            "This concept is unusable for negative mode."
        )
        logger.warning(f"{msg} Rejecting candidate and continuing.")
        record_counter(
            "stage0c.concept_rejections.zero_verified",
            1,
            concept_id=concept.concept_id,
            feature_id=f_c,
        )
        return []

    if len(verified) < cfg.n_verified_prefixes_per_concept_target:
        logger.warning(
            f"Concept {concept.concept_id}: only {len(verified)} verified "
            f"prefixes (target {cfg.n_verified_prefixes_per_concept_target})"
        )
        record_counter(
            "stage0c.concept_rejections.not_enough_verified_prefixes",
            1,
            concept_id=concept.concept_id,
            feature_id=f_c,
        )
    return verified


def _concept_from_candidate(
    candidate: CandidateConcept, *, concept_id: int,
) -> Concept:
    """Wrap a pre-QC candidate as a ``Concept`` for verification."""
    return Concept(
        concept_id=concept_id,
        feature_index=candidate.feature_index,
        layer=candidate.layer,
        sae_source=candidate.sae_source,
        description=candidate.description,
        corpus_count=candidate.corpus_count,
        description_source=candidate.description_source,
    )


def _build_qc_report(
    *,
    n_total: int,
    per_feature: list[QCFeatureScore],
    judge_min_score: int,
) -> QCReport:
    n_accepted = 0
    n_rejected = 0
    n_fetch_errors = 0
    n_judge_errors = 0
    n_empty_examples = 0
    score_histogram: dict[int, int] = {}
    for row in per_feature:
        if row.outcome == QcOutcome.ACCEPTED:
            n_accepted += 1
        elif row.outcome == QcOutcome.REJECTED:
            n_rejected += 1
        elif row.outcome == QcOutcome.FETCH_ERROR:
            n_fetch_errors += 1
        elif row.outcome == QcOutcome.JUDGE_ERROR:
            n_judge_errors += 1
        elif row.outcome == QcOutcome.EMPTY_EXAMPLES:
            n_empty_examples += 1
        else:
            raise AxBenchBaseError(
                f"Unexpected concept examples QC outcome: {row.outcome!r}"
            )
        if row.score is not None:
            score_histogram[row.score] = score_histogram.get(row.score, 0) + 1
    return QCReport(
        n_total=n_total,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_fetch_errors=n_fetch_errors,
        n_judge_errors=n_judge_errors,
        n_empty_examples=n_empty_examples,
        score_histogram=score_histogram,
        per_feature=per_feature,
        judge_min_score=judge_min_score,
        n_processed=len(per_feature),
    )


def _write_qc_report(
    output_paths: paths.AxBenchBasePaths,
    report: QCReport,
) -> None:
    report_path = output_paths.concept_examples_qc_report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        json.dumps(
            asdict(report),
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8"),
        report_path,
    )
    logger.info(f"Wrote concept examples QC report to {report_path}")


def _rejected_concept_to_json(row: _RejectedConcept) -> dict[str, Any]:
    """Serialize a rejection row with the public string reason value."""
    data = asdict(row)
    data["reason"] = row.reason.value
    return data


def _rejected_concept_from_json(data: dict[str, Any]) -> _RejectedConcept:
    """Reconstruct a ``_RejectedConcept`` from a JSON-loaded dict.

    Inverse of :func:`_rejected_concept_to_json`. Coerces the ``reason``
    string back into the :class:`RejectedConceptReason` enum.
    """
    return _RejectedConcept(
        candidate_index=int(data["candidate_index"]),
        feature_index=int(data["feature_index"]),
        description=str(data["description"]),
        reason=RejectedConceptReason(data["reason"]),
        n_verified_found=(
            None if data["n_verified_found"] is None
            else int(data["n_verified_found"])
        ),
        llm_judge_score=(
            None if data["llm_judge_score"] is None
            else int(data["llm_judge_score"])
        ),
    )


def _project_completion_eta_seconds(
    *,
    stage_elapsed: float,
    n_visited: int,
    n_accepted: int,
    max_concepts: int,
    n_candidates: int,
) -> float:
    """Estimate wall-time until Stage 0c hits ``max_concepts`` accepts.

    The prior formula ``avg_per_candidate * remaining_pool_size`` projected
    to walking the entire candidate pool — nonsensical when
    ``n_candidates`` is ~16k but we only need 5 accepts. This projection
    instead extrapolates the observed accept rate forward:

        candidates_per_accept = n_visited / n_accepted
        projected_remaining_candidates = accepts_still_needed * candidates_per_accept
        eta = avg_per_candidate * projected_remaining_candidates

    Corner cases:

    - Already hit the target (``accepts_still_needed == 0``) → 0.0.
    - Pool exhausted (``remaining_pool == 0``) → 0.0.
    - Zero accepts so far → no accept-rate evidence; fall back to the
      pool-walk bound so the operator sees a finite upper bound rather
      than ``inf``.
    - Projected remaining candidates exceed the remaining pool → cap at
      pool size. The loop can never process more candidates than exist.

    Returns seconds. Pure function; unit-tested independently.
    """
    if n_visited <= 0:
        return 0.0
    remaining_accepts_needed = max(0, max_concepts - n_accepted)
    remaining_pool = max(0, n_candidates - n_visited)
    if remaining_accepts_needed == 0 or remaining_pool == 0:
        return 0.0
    avg_per_candidate = stage_elapsed / n_visited
    if n_accepted == 0:
        projected_remaining = remaining_pool
    else:
        candidates_per_accept = n_visited / n_accepted
        projected_remaining = min(
            candidates_per_accept * remaining_accepts_needed,
            remaining_pool,
        )
    return avg_per_candidate * projected_remaining


def _log_candidate_completion(
    *,
    stage_started_at: float,
    concept_idx: int,
    n_candidates: int,
    accepted_concepts: list[Concept],
    rejected_concepts: list[_RejectedConcept],
    all_rows: list[_VerifiedNegativePrefix],
    max_concepts: int,
) -> None:
    """Emit the per-candidate Stage 0c completion heartbeat.

    Deduplicated across the two call sites (LLM-reject-before-verify and
    verify-accept-or-reject). The ETA uses
    :func:`_project_completion_eta_seconds` which projects to
    ``max_concepts`` accepts rather than exhausting the full pool.
    """
    stage_elapsed = time.monotonic() - stage_started_at
    n_visited = concept_idx + 1
    avg_per_candidate = stage_elapsed / n_visited
    n_accepted = len(accepted_concepts)
    eta_seconds = _project_completion_eta_seconds(
        stage_elapsed=stage_elapsed,
        n_visited=n_visited,
        n_accepted=n_accepted,
        max_concepts=max_concepts,
        n_candidates=n_candidates,
    )
    memory = process_memory_snapshot()
    memory_fields = ""
    if memory is not None:
        named_values = (
            ("current_rss_kb", memory.rss_kb),
            ("anonymous_rss_kb", memory.rss_anon_kb),
            ("swap_kb", memory.swap_kb),
            ("anonymous_total_kb", memory.anonymous_total_kb),
            ("peak_rss_kb", memory.peak_rss_kb),
        )
        available_values = [
            f"{name}={value}"
            for name, value in named_values
            if value is not None
        ]
        if available_values:
            memory_fields = ", memory after cleanup: " + ", ".join(
                available_values
            )
        for name, value in named_values:
            if value is not None:
                record_counter(
                    f"stage0c.memory.{name}",
                    value,
                    candidate_index=concept_idx,
                    candidates_visited=n_visited,
                )
    logger.info(
        f"[Stage 0c] Completed candidate {n_visited}/{n_candidates}: "
        f"{n_accepted}/{max_concepts} accepted concepts, "
        f"{len(rejected_concepts)} rejected concepts, "
        f"{len(all_rows)} total verified prefixes so far "
        f"(stage elapsed {stage_elapsed:.0f}s, "
        f"avg {avg_per_candidate:.0f}s/candidate, "
        f"eta remaining ≈ {eta_seconds:.0f}s"
        f"{memory_fields})"
    )


# ---------------------------------------------------------------------------
# Schema metadata helpers
# ---------------------------------------------------------------------------


def _schema_with_negative_prefixes_metadata(tokenizer_id: str) -> pa.Schema:
    """Return the Stage 0c parquet schema tagged with tokenizer identity."""
    existing = _NEG_PREFIXES_SCHEMA.metadata or {}
    metadata = dict(existing)
    metadata[_NEG_PREFIXES_TOKENIZER_METADATA_KEY.encode("utf-8")] = str(
        tokenizer_id
    ).encode("utf-8")
    return _NEG_PREFIXES_SCHEMA.with_metadata(metadata)


def _schema_with_completed_negative_prefixes_metadata(
    cfg: AxBenchBaseConfig,
    *,
    candidate_pool_identity: str,
) -> pa.Schema:
    """Return the canonical completed Stage 0c parquet schema.

    Partial parquets are paired with JSON fingerprints, but the final
    ``negative_mode_prefixes.parquet`` outlives those checkpoints.  Persist
    the extractor and candidate-pool identities on the final artifact itself
    so the lightweight completed-resume path can reject stale output before
    model loading.  The pool identity must outlive successful merge cleanup,
    which deletes the per-rank checkpoints that also carry it.

    PR #739 FU-D-2 assessment: this function is reached ONLY from
    ``stage0c_runner.run_merge`` (confirmed by grep -- its one production
    call site), which the dispatcher only ever invokes on rank 0 (or the
    single-rank direct path), with any raise caught and re-raised through
    rank 0's own try/broadcast wrapper in ``dispatch_prepare_negative_
    prefixes`` -- already deadlock-safe by construction, the same category
    as ``run_merge``'s own fingerprint re-read (:1530-ish, ``_fingerprint_
    with_pool`` called with no ``precomputed_clustered_fields``). Both
    ``_stage0c_extractor_identity(cfg)`` and ``clustered_stage0c_
    fingerprint_fields(cfg)`` below are therefore deliberately left as LIVE
    re-reads rather than threaded with the dispatcher's cached fields --
    this is the intentional other half of the staleness-detection story: a
    mid-run sidecar mutation the startup vote could not have observed is
    still caught here, at the exact moment the completed artifact is
    committed.
    """
    _validate_candidate_pool_identity_digest(
        candidate_pool_identity,
        consumer="completed Stage 0c parquet writer",
    )
    schema = _schema_with_negative_prefixes_metadata(
        cfg.extraction_tokenizer_id,
    )
    metadata = dict(schema.metadata or {})
    for key, value in _stage0c_extractor_identity(cfg).items():
        metadata[key.encode("utf-8")] = value.encode("utf-8")
    metadata[_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY.encode("utf-8")] = (
        candidate_pool_identity.encode("utf-8")
    )
    for key, value in clustered_stage0c_fingerprint_fields(cfg).items():
        metadata[key.encode("utf-8")] = value.encode("utf-8")
    return schema.with_metadata(metadata)


def _schema_with_completed_artifact_set_identities(
    schema: pa.Schema,
    *,
    concepts: list[Concept],
    rejected_concepts: list[dict[str, Any]],
    concept_examples_qc_report: dict[str, Any] | None,
    stage0c_fingerprint: dict[str, Any],
) -> pa.Schema:
    """Bind a completed parquet to every JSON sidecar and the full config.

    This deliberately layers over an existing completed schema rather than
    changing :func:`_schema_with_completed_negative_prefixes_metadata` so
    Stage 1-only schema fixtures do not need transaction metadata. The caller
    passes the exact agreed per-rank fingerprint selected by ``run_merge``.
    """
    if not isinstance(stage0c_fingerprint, dict):
        raise _CompletedStage0cArtifactSetIdentityError(
            "completed Stage 0c artifact writer: stage0c_fingerprint must be "
            f"a dict, got {type(stage0c_fingerprint).__name__}"
        )
    try:
        schema_pool_identity = (
            _negative_prefixes_candidate_pool_identity_from_schema(schema)
        )
    except UnicodeDecodeError as exc:
        raise _CompletedStage0cCandidatePoolIdentityError(
            "completed Stage 0c artifact writer: existing schema has non-UTF-8 "
            f"{_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY!r} metadata"
        ) from exc
    if schema_pool_identity is None:
        raise _CompletedStage0cCandidatePoolIdentityError(
            "completed Stage 0c artifact writer: existing schema is missing "
            f"{_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY!r} metadata"
        )
    _validate_candidate_pool_identity_digest(
        schema_pool_identity,
        consumer="completed Stage 0c artifact writer schema",
    )
    fingerprint_pool_identity = _validate_candidate_pool_identity_digest(
        stage0c_fingerprint.get("candidate_pool_identity"),
        consumer="completed Stage 0c artifact writer fingerprint",
    )
    if fingerprint_pool_identity != schema_pool_identity:
        raise _CompletedStage0cArtifactSetIdentityError(
            "completed Stage 0c artifact writer: agreed fingerprint has "
            f"candidate_pool_identity={fingerprint_pool_identity!r}, but the "
            f"existing schema has {schema_pool_identity!r}"
        )

    metadata = dict(schema.metadata or {})
    identities = {
        _NEG_PREFIXES_CONCEPTS_METADATA_KEY: concepts_identity(concepts),
        _NEG_PREFIXES_REJECTED_CONCEPTS_METADATA_KEY: (
            rejected_concepts_identity(rejected_concepts)
        ),
        _NEG_PREFIXES_QC_REPORT_METADATA_KEY: (
            concept_examples_qc_report_identity(concept_examples_qc_report)
        ),
        _NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY: (
            stage0c_fingerprint_identity(stage0c_fingerprint)
        ),
    }
    for key, identity in identities.items():
        _validate_completed_artifact_identity_digest(
            identity,
            metadata_key=key,
            consumer="completed Stage 0c artifact writer",
        )
        metadata[key.encode("utf-8")] = identity.encode("utf-8")
    return schema.with_metadata(metadata)


def _negative_prefixes_tokenizer_metadata_from_schema(
    schema: pa.Schema,
) -> str | None:
    metadata = schema.metadata or {}
    raw = metadata.get(_NEG_PREFIXES_TOKENIZER_METADATA_KEY.encode("utf-8"))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _negative_prefixes_extractor_identity_from_schema(
    schema: pa.Schema,
) -> dict[str, str | None]:
    """Read the completed Stage 0c extractor identity from ``schema``."""
    metadata = schema.metadata or {}
    identity: dict[str, str | None] = {}
    for key in _NEG_PREFIXES_EXTRACTOR_METADATA_KEYS:
        raw = metadata.get(key.encode("utf-8"))
        if raw is None:
            identity[key] = None
        elif isinstance(raw, bytes):
            identity[key] = raw.decode("utf-8")
        else:
            identity[key] = str(raw)
    return identity


def _negative_prefixes_candidate_pool_identity_from_schema(
    schema: pa.Schema,
) -> str | None:
    """Read the completed Stage 0c candidate-pool digest from ``schema``."""
    metadata = schema.metadata or {}
    raw = metadata.get(_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY.encode("utf-8"))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _negative_prefixes_completed_artifact_identities_from_schema(
    schema: pa.Schema,
) -> dict[str, str | None]:
    """Read semantic sidecar/config identities from completed metadata."""
    metadata = schema.metadata or {}
    identities: dict[str, str | None] = {}
    for key in _NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS:
        raw = metadata.get(key.encode("utf-8"))
        if raw is None:
            identities[key] = None
        elif isinstance(raw, bytes):
            identities[key] = raw.decode("utf-8")
        else:
            identities[key] = str(raw)
    return identities


def _validate_candidate_pool_identity_digest(
    value: object, *, consumer: str,
) -> str:
    """Require the lowercase SHA-256 representation emitted by the hasher."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise _CompletedStage0cCandidatePoolIdentityError(
            f"{consumer}: {_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY} must be a "
            f"64-character lowercase SHA-256 hex digest, got {value!r}"
        )
    return value


def _validate_completed_artifact_identity_digest(
    value: object,
    *,
    metadata_key: str,
    consumer: str,
) -> str:
    """Require a lowercase SHA-256 identity for completed transaction state."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise _CompletedStage0cArtifactSetIdentityError(
            f"{consumer}: {metadata_key} must be a 64-character lowercase "
            f"SHA-256 hex digest, got {value!r}"
        )
    return value


def validate_negative_prefixes_tokenizer_metadata(
    path: Path,
    *,
    expected_tokenizer_id: str,
    consumer: str,
    schema: pa.Schema | None = None,
) -> None:
    """Validate that a Stage 0c prefix parquet matches the current tokenizer.

    Older Stage 0c artifacts without tokenizer metadata are intentionally
    stale because candidate tokenization, Gemma generation decoding, and
    verification QC all depend on the extraction tokenizer.
    """
    parquet_schema = schema if schema is not None else pq.read_schema(path)
    try:
        actual = _negative_prefixes_tokenizer_metadata_from_schema(parquet_schema)
    except UnicodeDecodeError as exc:
        raise AxBenchBaseError(
            f"{consumer}: negative_mode_prefixes parquet at {path} has "
            f"non-UTF-8 {_NEG_PREFIXES_TOKENIZER_METADATA_KEY!r} metadata. "
            "Delete or regenerate the file by rerunning Stage 0c "
            "(PREPARE_NEGATIVE_PREFIXES)."
        ) from exc
    if actual is None:
        raise AxBenchBaseError(
            f"{consumer}: negative_mode_prefixes parquet at {path} is missing "
            f"{_NEG_PREFIXES_TOKENIZER_METADATA_KEY!r} metadata. Delete or "
            "regenerate the file by rerunning Stage 0c "
            "(PREPARE_NEGATIVE_PREFIXES)."
        )
    if actual != expected_tokenizer_id:
        raise AxBenchBaseError(
            f"{consumer}: negative_mode_prefixes parquet at {path} was created "
            f"with extraction tokenizer {actual!r}, but the current config uses "
            f"{expected_tokenizer_id!r}. Rerun Stage 0c "
            "(PREPARE_NEGATIVE_PREFIXES) for this tokenizer."
        )


def validate_negative_prefixes_extractor_identity_metadata(
    path: Path,
    *,
    cfg: AxBenchBaseConfig,
    consumer: str,
    schema: pa.Schema | None = None,
) -> None:
    """Validate a completed Stage 0c artifact's extractor identity.

    Artifacts predating this metadata are intentionally stale: there is no
    safe way to infer which model/SAE identity produced their verification
    continuations.  Validation therefore fails loudly instead of silently
    reusing the output or automatically rerunning Stage 0c against a possibly
    stale Stage 0a candidate pool.

    PR #739 FU-D-2 assessment (corrected by review): this function has
    exactly THREE call sites, with two distinct DDP postures.
    (1)+(2) ``negative_prefixes._validate_negative_prefixes_table`` and
    ``negative_prefixes._validate_completed_artifact_identity_and_schema``
    -- the completed-artifact "resume"/merge family, reached rank-0-only
    and wrapped by ``_broadcast_resume_decision`` (or inside ``run_merge``'s
    rank-0-only post-write validation): deadlock-safe by the same
    construction as ``_schema_with_completed_negative_prefixes_metadata``
    above.
    (3) ``synth/negative_mode.py::_load_negative_prefixes`` (Stage 1c) --
    NOT deadlock-safe today: ``scripts/eval/axbench_base_eval.py``'s
    ``_dispatch_stage`` runs SYNTH_NEGATIVE_MODE rank-0-only with a bare
    ``dist.barrier()`` before/after and NO try/except + broadcast wrap, so
    a raise here on rank 0 (e.g. a stale parquet's identity mismatch)
    strands the other ranks in the barrier until the NCCL/torchrun timeout.
    That gap is PRE-EXISTING stage-dispatch architecture (untouched by
    FU-D-2, whose scope is the Stage-0c runner family) and is tracked as a
    named follow-up: wrap rank-0-only stage execution in the same
    try/except + broadcast pattern PREPARE_NEGATIVE_PREFIXES already uses.
    ``_stage0c_extractor_identity(cfg)`` below is deliberately left as a
    live re-read, not threaded with the dispatcher's cached fields --
    consistent with the resume/merge family's staleness-detection role.
    """
    parquet_schema = schema if schema is not None else pq.read_schema(path)
    try:
        actual = _negative_prefixes_extractor_identity_from_schema(parquet_schema)
    except UnicodeDecodeError as exc:
        raise _CompletedStage0cExtractorIdentityError(
            f"{consumer}: negative_mode_prefixes parquet at {path} has "
            "non-UTF-8 extractor identity metadata. Refusing automatic "
            "Stage 0c reuse or rerun because candidate_concepts.json is also "
            "scoped by this identity. Use a new `run_tag`, or delete the "
            "Stage 0 artifacts and rerun Stage 0a then Stage 0c."
        ) from exc
    expected = _stage0c_extractor_identity(cfg)
    for key in _NEG_PREFIXES_EXTRACTOR_METADATA_KEYS:
        actual_value = actual[key]
        if actual_value is None:
            raise _CompletedStage0cExtractorIdentityError(
                f"{consumer}: negative_mode_prefixes parquet at {path} is "
                f"missing {key!r} metadata. Refusing automatic Stage 0c reuse "
                "or rerun because candidate_concepts.json is also scoped by "
                "this identity. Use a new `run_tag`, or delete the Stage 0 "
                "artifacts and rerun Stage 0a then Stage 0c."
            )
        if actual_value != expected[key]:
            raise _CompletedStage0cExtractorIdentityError(
                f"{consumer}: negative_mode_prefixes parquet at {path} was "
                f"created with {key}={actual_value!r}, but the current config "
                f"uses {expected[key]!r}. Refusing automatic Stage 0c reuse "
                "or rerun because candidate_concepts.json is also scoped by "
                "this identity. Use a new `run_tag`, or delete the Stage 0 "
                "artifacts and rerun Stage 0a then Stage 0c."
            )


def validate_negative_prefixes_candidate_pool_identity_metadata(
    path: Path,
    *,
    consumer: str,
    expected_candidate_pool_identity: str | None = None,
    schema: pa.Schema | None = None,
) -> None:
    """Fail closed when completed output is not tied to the live Stage 0a pool.

    Passing no expected digest performs the cheap provenance-presence check
    used before the feature-statistics header and candidate JSON are loaded.
    Passing an expected digest additionally compares the completed artifact to
    the freshly loaded, ordered ``candidate_concepts.json`` pool.
    """
    parquet_schema = schema if schema is not None else pq.read_schema(path)
    try:
        actual = _negative_prefixes_candidate_pool_identity_from_schema(
            parquet_schema
        )
    except UnicodeDecodeError as exc:
        raise _CompletedStage0cCandidatePoolIdentityError(
            f"{consumer}: negative_mode_prefixes parquet at {path} has non-UTF-8 "
            f"{_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY!r} metadata. Refusing "
            "completed Stage 0c reuse. Use a new `run_tag`, or delete the Stage "
            "0 artifacts and rerun Stage 0a then Stage 0c."
        ) from exc
    if actual is None:
        raise _CompletedStage0cCandidatePoolIdentityError(
            f"{consumer}: negative_mode_prefixes parquet at {path} is missing "
            f"{_NEG_PREFIXES_CANDIDATE_POOL_METADATA_KEY!r} metadata. Refusing "
            "completed Stage 0c reuse because the per-rank checkpoints that "
            "carried the candidate-pool identity have already been removed. "
            "Use a new `run_tag`, or delete the Stage 0 artifacts and rerun "
            "Stage 0a then Stage 0c."
        )
    _validate_candidate_pool_identity_digest(
        actual,
        consumer=f"{consumer}: negative_mode_prefixes parquet at {path}",
    )
    if expected_candidate_pool_identity is None:
        return
    _validate_candidate_pool_identity_digest(
        expected_candidate_pool_identity,
        consumer=f"{consumer}: live candidate_concepts.json",
    )
    if actual != expected_candidate_pool_identity:
        raise _CompletedStage0cCandidatePoolIdentityError(
            f"{consumer}: negative_mode_prefixes parquet at {path} was created "
            f"with candidate_pool_identity={actual!r}, but the current ordered "
            "candidate_concepts.json has "
            f"candidate_pool_identity={expected_candidate_pool_identity!r}. "
            "Refusing mixed-pool completed artifacts. Use a new `run_tag`, or "
            "delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c."
        )


def validate_negative_prefixes_completed_artifact_identities(
    path: Path,
    *,
    consumer: str,
    expected_concepts: list[Concept] | None = None,
    expected_rejected_concepts: list[dict[str, Any]] | None = None,
    expected_concept_examples_qc_report: dict[str, Any] | None = None,
    expected_stage0c_fingerprint: dict[str, Any] | None = None,
    schema: pa.Schema | None = None,
) -> None:
    """Fail closed unless completed parquet metadata binds all final state.

    With no expected payloads this performs the cheap presence/format check
    used before any JSON sidecar or feature-statistics load. Supplying expected
    payloads compares semantic hashes, so JSON whitespace and object-key order
    are irrelevant while list order and validated field values remain binding.
    """
    expected_values = (
        expected_concepts,
        expected_rejected_concepts,
        expected_stage0c_fingerprint,
    )
    if any(value is not None for value in expected_values) and not all(
        value is not None for value in expected_values
    ):
        raise ValueError(
            "completed artifact identity validation requires all expected "
            "concepts, rejected concepts, and Stage 0c fingerprint payloads"
        )

    parquet_schema = schema if schema is not None else pq.read_schema(path)
    try:
        actual = _negative_prefixes_completed_artifact_identities_from_schema(
            parquet_schema
        )
    except UnicodeDecodeError as exc:
        raise _CompletedStage0cArtifactSetIdentityError(
            f"{consumer}: negative_mode_prefixes parquet at {path} has non-UTF-8 "
            "completed artifact identity metadata. Refusing completed Stage 0c "
            "reuse. Use a new `run_tag`, or delete the Stage 0 artifacts and "
            "rerun Stage 0a then Stage 0c."
        ) from exc

    for key in _NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS:
        value = actual[key]
        if value is None:
            raise _CompletedStage0cArtifactSetIdentityError(
                f"{consumer}: negative_mode_prefixes parquet at {path} is "
                f"missing {key!r} metadata. Refusing completed Stage 0c reuse "
                "because the canonical parquet is not bound to the full final "
                "artifact set. Use a new `run_tag`, or delete the Stage 0 "
                "artifacts and rerun Stage 0a then Stage 0c."
            )
        _validate_completed_artifact_identity_digest(
            value,
            metadata_key=key,
            consumer=f"{consumer}: negative_mode_prefixes parquet at {path}",
        )

    if expected_concepts is None:
        return
    assert expected_rejected_concepts is not None
    assert expected_stage0c_fingerprint is not None
    expected = {
        _NEG_PREFIXES_CONCEPTS_METADATA_KEY: concepts_identity(expected_concepts),
        _NEG_PREFIXES_REJECTED_CONCEPTS_METADATA_KEY: (
            rejected_concepts_identity(expected_rejected_concepts)
        ),
        _NEG_PREFIXES_QC_REPORT_METADATA_KEY: (
            concept_examples_qc_report_identity(
                expected_concept_examples_qc_report
            )
        ),
        _NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY: (
            stage0c_fingerprint_identity(expected_stage0c_fingerprint)
        ),
    }
    source_names = {
        _NEG_PREFIXES_CONCEPTS_METADATA_KEY: "current semantic concepts.json",
        _NEG_PREFIXES_REJECTED_CONCEPTS_METADATA_KEY: (
            "current semantic rejected_concepts.json"
        ),
        _NEG_PREFIXES_QC_REPORT_METADATA_KEY: (
            "current semantic concept_examples_qc_report.json state"
        ),
        _NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY: (
            "current full Stage 0c fingerprint"
        ),
    }
    # Backward compatibility for artifacts written while `max_concepts` was
    # still part of the fingerprint. Their stored identity hashes a dict that
    # included the field, so a straight comparison against the current
    # (field-free) fingerprint would reject otherwise-valid completed runs.
    #
    # The legacy value is recoverable rather than guessed: a COMPLETED Stage 0c
    # artifact set contains exactly `max_concepts` accepted concepts -- that
    # invariant is asserted by the resume path itself -- so the field's value
    # must have been `len(expected_concepts)`.
    legacy_fingerprint = {
        **expected_stage0c_fingerprint,
        "max_concepts": len(expected_concepts),
    }
    legacy_fingerprint_identity = stage0c_fingerprint_identity(legacy_fingerprint)

    for key in _NEG_PREFIXES_COMPLETED_ARTIFACT_IDENTITY_KEYS:
        if (
            key == _NEG_PREFIXES_STAGE0C_FINGERPRINT_METADATA_KEY
            and actual[key] == legacy_fingerprint_identity
        ):
            # Pre-removal artifact whose identity is otherwise equivalent.
            continue
        if actual[key] != expected[key]:
            raise _CompletedStage0cArtifactSetIdentityError(
                f"{consumer}: negative_mode_prefixes parquet at {path} was "
                f"created with {key}={actual[key]!r}, but the "
                f"{source_names[key]} has {key}={expected[key]!r}. Refusing "
                "mixed completed artifacts. Use a new `run_tag`, or delete the "
                "Stage 0 artifacts and rerun Stage 0a then Stage 0c."
            )
