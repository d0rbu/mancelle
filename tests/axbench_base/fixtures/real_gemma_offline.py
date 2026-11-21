"""Protocol-aware local oracle for the real-Gemma OpenRouter integration.

This module is test-only.  It keeps provider request/response semantics at the
real client boundary while using the already-loaded Gemma + SAE objects for
the same local parsing, percentile, and strict-zero gates used by production.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from steering.eval.axbench_base import _stage0c_internals as stage0c_internals
from steering.eval.axbench_base import paths as axbench_paths
from steering.eval.axbench_base import stage0c_api_generation as stage0c_api
from steering.eval.axbench_base._stats_reader import read_n_sae_features_from_stats
from steering.eval.axbench_base.batched_generation import batched_generate_seeds
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.extraction import build_percentile_table_from_stats
from steering.eval.axbench_base.judge_parser import parse_axbench_rating
from steering.eval.axbench_base.percentile_units import (
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.synth.batching import SynthCompleter
from steering.eval.axbench_base.synth.continuation_guards import (
    ResponseEnvelope,
    prefix_echo_text,
    prefix_suffix_echo_chars,
)
from steering.eval.axbench_base.synth.continuation_protocol import (
    parse_stage1_continuation_response,
)
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.synth.qc import (
    AvoidContinuationOutcome,
    classify_avoid_continuation,
    compute_fc_percentiles,
)
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.types import Concept, TargetModelMetadata
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureTarget,
)
from tests.axbench_base.fixtures.offline_openrouter import (
    OfflineOpenRouterError,
    OfflineOpenRouterTransport,
    OpenRouterRequest,
)
from tests.axbench_base.stage0c_api_helpers import parse_continuation_response

_CONCEPT_TAG_RE = re.compile(r"(?:pos|neg|stage1)[^_]*_c(\d+)")
_SEED_RE = re.compile(r"Variation key:\s*(-?\d+)")
_BEGIN_END_RE = re.compile(
    r"(?P<begin>\[\[AXBENCH_(?P<kind>[A-Z0-9_]+)_BEGIN_"
    r"(?P<nonce>[A-F0-9]+)\]\])"
    r".*?"
    r"(?P<end>\[\[AXBENCH_(?P=kind)_END_(?P=nonce)\]\])",
    re.DOTALL,
)


def _recurring_peak_from_evidence(evidence: list[dict[str, Any]]) -> str | None:
    """Mirror production's exact-or-casefolded recurring-peak contract."""

    if len(evidence) < 2:
        return None
    normalized = [
        " ".join(str(row["peak_token_text"]).split()) for row in evidence
    ]
    if not normalized[0]:
        return None
    if all(value == normalized[0] for value in normalized[1:]):
        return normalized[0]
    casefolded = [value.casefold() for value in normalized]
    if all(value == casefolded[0] for value in casefolded[1:]):
        return casefolded[0]
    return None


def _activation_evidence_prefixes(
    evidence: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_scored_tokens: int,
) -> tuple[str, ...]:
    """Return token-aligned real excerpts whose measured peak is scoreable."""

    if max_scored_tokens < 1:
        raise ValueError("max_scored_tokens must be positive")

    prefixes: list[str] = []
    for row in evidence:
        through_peak = str(row.get("context_before_peak", "")) + str(
            row.get("peak_token_text", "")
        )
        through_peak_ids = list(
            tokenizer.encode(through_peak, add_special_tokens=False)
        )
        if not through_peak_ids:
            continue
        # Keep the peak and as much of its real left context as fits while
        # reserving up to two scored tokens after it. Character slicing can
        # begin in the middle of a subword (for example, ``create`` ->
        # ``eate``) and destroy the very activation this evidence is meant to
        # exercise.
        keep = min(len(through_peak_ids), max(1, max_scored_tokens - 2))
        through_peak_suffix = tokenizer.decode(through_peak_ids[-keep:])
        text = (
            through_peak_suffix
            + str(row.get("context_after_peak", ""))[:192]
        ).strip()
        if text and text not in prefixes:
            prefixes.append(text)
    return tuple(prefixes)


def _without_leading_prefix_echo(source_text: str, generated_text: str) -> str:
    """Normalize a local-model echo before emulating a compliant API response."""

    normalized = generated_text
    while True:
        overlap = prefix_suffix_echo_chars(source_text, normalized)
        if overlap == 0:
            return normalized
        normalized = normalized.lstrip()[overlap:]


def _with_required_peak(
    source_text: str,
    generated_text: str,
    recurring_peak: str | None,
) -> str:
    """Insert a required peak without reconstructing a source-suffix echo."""

    if recurring_peak is None or recurring_peak in generated_text:
        return generated_text
    with_peak = f"{recurring_peak} {generated_text}"
    normalized = _without_leading_prefix_echo(source_text, with_peak)
    if recurring_peak in normalized:
        return normalized
    return f". {recurring_peak} {normalized}"


def _local_generation_budget(
    tokenizer: Any,
    leading_text: str,
    *,
    scored_tokens: int,
) -> int:
    """Generate only the tokens missing from the scored response plus cushion."""

    if scored_tokens < 1:
        raise ValueError("scored_tokens must be positive")
    leading_tokens = len(tokenizer.encode(leading_text, add_special_tokens=False))
    missing = max(0, scored_tokens - leading_tokens)
    return max(2, missing + 2)


@dataclass(frozen=True)
class _StageContext:
    cfg: AxBenchBaseConfig
    base_model: Any
    base_sae: Any
    target_model_meta: TargetModelMetadata
    concepts: dict[int, Concept]
    percentile_table: Any
    cluster_contract: ClusterFeatureContract | None


_feature_context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "real_gemma_offline_feature", default=None
)
_stage0c_member_context: contextvars.ContextVar[int | None] = (
    contextvars.ContextVar("real_gemma_offline_stage0c_member", default=None)
)
_candidate_sources: contextvars.ContextVar[dict[str, tuple[str, str]] | None] = (
    contextvars.ContextVar("real_gemma_offline_candidate_sources", default=None)
)


class RealGemmaOfflineOracle:
    """Deterministic bounded local response oracle for all five request classes."""

    def __init__(self, transport: OfflineOpenRouterTransport) -> None:
        self.transport = transport
        self._context: _StageContext | None = None
        self._cluster_contract: ClusterFeatureContract | None = None
        self._lock = threading.RLock()
        self._definitions: dict[int, tuple[str, str | None]] = {}
        self._activation_evidence: dict[int, tuple[dict[str, Any], ...]] = {}
        self._failures: list[str] = []
        self._max_seeds = 16

    @contextlib.contextmanager
    def stage_context(
        self,
        cfg: AxBenchBaseConfig,
        *,
        base_model: Any,
        base_sae: Any,
        target_model_meta: TargetModelMetadata,
        concepts: list[Concept],
        cluster_contract: ClusterFeatureContract | None = None,
    ) -> Iterator[None]:
        previous = self._context
        if cluster_contract is not None:
            self._cluster_contract = cluster_contract
        self._context = _StageContext(
            cfg=cfg,
            base_model=base_model,
            base_sae=base_sae,
            target_model_meta=target_model_meta,
            concepts={concept.concept_id: concept for concept in concepts},
            percentile_table=build_percentile_table_from_stats(
                cfg.feature_statistics_path,
                n_sae_features=read_n_sae_features_from_stats(Path(cfg.feature_statistics_path)),
            ),
            cluster_contract=cluster_contract,
        )
        try:
            yield
        finally:
            self._context = previous

    @contextlib.contextmanager
    def feature_context(self, feature_id: int) -> Iterator[None]:
        token = _feature_context.set(int(feature_id))
        try:
            yield
        finally:
            _feature_context.reset(token)

    @contextlib.contextmanager
    def stage0c_member_context(self, feature_id: int) -> Iterator[None]:
        """Bind the member selected by production's clustered proposal cycle."""

        token = _stage0c_member_context.set(int(feature_id))
        try:
            yield
        finally:
            _stage0c_member_context.reset(token)

    @contextlib.contextmanager
    def candidate_context(self, candidates: Iterable[Any]) -> Iterator[None]:
        """Bind exact live Stage 0c prefix/bridge pairs to this request context."""

        source_map = {
            str(candidate.prefix_text) + str(candidate.source_bridge_text): (
                str(candidate.prefix_text),
                str(candidate.source_bridge_text),
            )
            for candidate in candidates
        }
        token = _candidate_sources.set(source_map)
        try:
            yield
        finally:
            _candidate_sources.reset(token)

    def response(self, request: OpenRouterRequest) -> str:
        with self._lock:
            request_class = request.request_class
            if request_class is None:
                request_class = self._classify_prompt(request)
            if request_class == "stage0c_definition":
                return self._definition_response(request)
            if request_class == "stage0c_continuation":
                return self._stage0c_continuation_response(request)
            if request_class == "stage1_synth_rich":
                return self._stage1_rich_response(request)
            if request_class == "stage1_synth_avoid":
                return self._stage1_avoid_response(request)
            if request_class == "stage5_judge":
                return self._judge_response(request)
            raise self._fail(request, f"unknown request class {request_class!r}")

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(self._failures)

    @property
    def cluster_contract(self) -> ClusterFeatureContract | None:
        """Return the clustered contract observed on the live stage wrappers."""

        return self._cluster_contract

    def _classify_prompt(self, request: OpenRouterRequest) -> str | None:
        prompt = request.messages[-1]["content"] if request.messages else ""
        if "AXBENCH_DEFINITION_BEGIN_" in prompt:
            return "stage0c_definition"
        if "AXBENCH_CONTINUATION_BEGIN_" in prompt:
            return "stage0c_continuation"
        if "AXBENCH_STAGE1B_RICH_BEGIN_" in prompt:
            return "stage1_synth_rich"
        if (
            "AXBENCH_STAGE1B_AVOID_BEGIN_" in prompt
            or "AXBENCH_STAGE1C_AVOID_BEGIN_" in prompt
        ):
            return "stage1_synth_avoid"
        if "Return the final content in plain text" in prompt:
            return "stage1_synth_rich"
        if "gracefully transitions away" in prompt.lower():
            return "stage1_synth_avoid"
        if "rating" in prompt.lower():
            return "stage5_judge"
        return None

    def _ctx(self, request: OpenRouterRequest) -> _StageContext:
        if self._context is None:
            raise self._fail(request, "no live Gemma/SAE stage context is bound")
        return self._context

    def _feature_id(self, request: OpenRouterRequest) -> int:
        member_id = _stage0c_member_context.get()
        if member_id is not None:
            return member_id
        feature_id = _feature_context.get()
        if feature_id is not None:
            return feature_id
        raise self._fail(
            request,
            "feature id was not propagated to the oracle through the live "
            "Stage 0c call context",
        )

    def _concept_id(self, request: OpenRouterRequest) -> int:
        match = _CONCEPT_TAG_RE.search(request.cache_key or "")
        if match is not None:
            return int(match.group(1))
        prompt = self._user_prompt(request)
        ctx = self._ctx(request)
        matches = [
            concept.concept_id
            for concept in ctx.concepts.values()
            if concept.description and concept.description in prompt
        ]
        if len(matches) == 1:
            return matches[0]
        raise self._fail(
            request,
            "concept id was not uniquely recoverable from request metadata or "
            f"the prompt (matches={matches!r})",
        )

    def _user_prompt(self, request: OpenRouterRequest) -> str:
        if not request.messages:
            raise self._fail(request, "request has no messages")
        return request.messages[-1]["content"]

    def _definition_response(self, request: OpenRouterRequest) -> str:
        ctx = self._ctx(request)
        feature_id = self._feature_id(request)
        prompt = self._user_prompt(request)
        envelope = self._envelope(prompt)
        evidence = self._evidence_rows(prompt, request)
        peak = _recurring_peak_from_evidence(evidence)
        if peak is None:
            definition = (
                "This writing pattern appears as a recurring contextual expression "
                "used naturally in ordinary prose."
            )
        else:
            definition = (
                f"The form {peak!r} recurs in ordinary contextual prose as a "
                "natural lexical or grammatical element."
            )
        parsed = stage0c_api.parse_definition_response(
            envelope.begin + definition + envelope.end,
            envelope=envelope,
        )
        self._definitions[feature_id] = (parsed, peak)
        self._activation_evidence[feature_id] = tuple(dict(row) for row in evidence)
        del ctx
        return envelope.begin + parsed + envelope.end

    def _cluster_target(
        self,
        ctx: _StageContext,
        representative_id: int,
    ) -> ClusterFeatureTarget | None:
        if ctx.cluster_contract is None:
            return None
        return ctx.cluster_contract.target_for_representative(representative_id)

    def _stage0c_continuation_response(self, request: OpenRouterRequest) -> str:
        ctx = self._ctx(request)
        definition_feature_id = self._feature_id(request)
        public_feature_id = _feature_context.get()
        if public_feature_id is None:
            public_feature_id = definition_feature_id
        cluster_target = self._cluster_target(ctx, public_feature_id)
        prompt = self._user_prompt(request)
        definition, recurring_peak = self._definitions.get(
            definition_feature_id,
            ("This writing pattern appears in ordinary contextual prose.", None),
        )
        source = self._source_context(prompt, request)
        prefix, source_bridge = self._source_parts(source, request)
        envelope = self._envelope(prompt)
        seed_match = _SEED_RE.search(prompt)
        requested_seed = int(seed_match.group(1)) if seed_match else 0
        tokenizer = get_extraction_tokenizer(ctx.cfg.extraction_tokenizer_id)
        last_failure = "no candidate seed attempted"
        raw_evidence = self._activation_evidence.get(definition_feature_id)
        if raw_evidence:
            bridge_token_count = len(
                tokenizer.encode(source_bridge, add_special_tokens=False)
            )
            scored_evidence_budget = max(
                1,
                ctx.cfg.verification_length - bridge_token_count,
            )
            generation_prefixes = _activation_evidence_prefixes(
                list(raw_evidence),
                tokenizer=tokenizer,
                max_scored_tokens=scored_evidence_budget,
            )
        else:
            generation_prefixes = (
                "In the following years, the project continued to grow, and",
            )
        for offset in range(self._max_seeds):
            seed = requested_seed + offset * 1_000_003
            try:
                generation_prefix = generation_prefixes[offset % len(generation_prefixes)]
                generated_tail = self._generate_text(
                    ctx,
                    generation_prefix,
                    seed=seed,
                    # The trusted bridge plus real evidence may already fill
                    # most or all of the exact scored window. Generate only
                    # what remains plus a small EOS cushion: a long unscored
                    # PT-model tail can degenerate and would test irrelevant
                    # tail length rather than the production parser/QC contract.
                    max_new_tokens=_local_generation_budget(
                        tokenizer,
                        source_bridge + generation_prefix,
                        scored_tokens=ctx.cfg.verification_length,
                    ),
                )
                generated_tail = _without_leading_prefix_echo(
                    generation_prefix,
                    generated_tail,
                )
                generated = (generation_prefix + " " + generated_tail).strip()
                generated = _without_leading_prefix_echo(
                    prefix + source_bridge,
                    generated,
                )
                if not generated.strip():
                    raise ValueError("local Gemma generation contained only a source echo")
                generated = _with_required_peak(
                    prefix + source_bridge,
                    generated,
                    recurring_peak,
                )
                response = (
                    envelope.begin
                    + "\n"
                    + prefix_echo_text(prefix + source_bridge)
                )
                response += generated.strip() + " " + envelope.end
                token_ids = parse_continuation_response(
                    response,
                    envelope=envelope,
                    prefix_text=prefix,
                    output_length=ctx.cfg.verification_length,
                    tokenizer=tokenizer,
                    concept_definition=definition,
                    recurring_peak_text=recurring_peak,
                    required_source_bridge_text=source_bridge,
                )
                if len(token_ids) != ctx.cfg.verification_length:
                    raise ValueError(
                        "production Stage 0c parser returned an unexpected token "
                        f"count={len(token_ids)} expected={ctx.cfg.verification_length}"
                    )
                continuation = tokenizer.decode(token_ids)
                _seed_pct, output_pct = compute_fc_percentiles(
                    prefix,
                    continuation,
                    public_feature_id,
                    ctx.base_model,
                    ctx.base_sae,
                    ctx.percentile_table,
                    sae_layer=ctx.target_model_meta.sae_layer,
                    tokenizer_id=ctx.cfg.extraction_tokenizer_id,
                    cluster_target=cluster_target,
                )
                verification_threshold_points = percentile_fraction_to_points(
                    ctx.cfg.verification_percentile_threshold,
                    field_name="verification_percentile_threshold",
                )
                if output_pct <= verification_threshold_points:
                    raise ValueError(
                        f"output percentile {output_pct:.4f} did not exceed "
                        f"{verification_threshold_points:.4f}"
                    )
                return response
            except Exception as exc:
                last_failure = f"seed={seed}: {type(exc).__name__}: {exc}"
        raise self._fail(
            request,
            f"Stage 0c oracle exhausted feature={public_feature_id} "
            f"requested_seed={requested_seed}; last_failure={last_failure}",
        )

    def _stage1_rich_response(self, request: OpenRouterRequest) -> str:
        ctx = self._ctx(request)
        concept_id = self._concept_id(request)
        concept = ctx.concepts.get(concept_id)
        if concept is None:
            raise self._fail(request, f"unknown concept_id={concept_id}")
        prefix = self._stage1_prefix(self._user_prompt(request), request)
        tokenizer = get_extraction_tokenizer(ctx.cfg.extraction_tokenizer_id)
        cluster_target = self._cluster_target(ctx, concept.feature_index)
        evidence_prefixes = self._stage1_evidence_prefixes(
            ctx,
            representative_id=concept.feature_index,
            tokenizer=tokenizer,
        )
        last_failure = "no candidate seed attempted"
        for offset, evidence_prefix in enumerate(
            evidence_prefixes[: self._max_seeds]
        ):
            seed = ctx.cfg.generation.master_seed + concept_id * 1009 + offset
            try:
                generated_tail = self._generate_text(
                    ctx,
                    prefix + " " + evidence_prefix,
                    seed=seed,
                    max_new_tokens=2,
                )
                # Lead with a real top-activating member excerpt, then retain a
                # tiny locally generated suffix so the provider boundary still
                # exercises real Gemma decoding.  The production parser and
                # real max-pooled SAE QC remain the only acceptance authority.
                candidate = " " + (evidence_prefix + " " + generated_tail).strip()
                response = self._stage1_response(
                    self._user_prompt(request),
                    prefix,
                    candidate,
                )
                continuation = parse_stage1_continuation_response(
                    response=response,
                    envelope=self._envelope(self._user_prompt(request)),
                    prefix_text=prefix,
                    output_length=ctx.cfg.output_length,
                    tokenizer=tokenizer,
                    subject="real-Gemma Stage 1 rich continuation",
                )
                if not continuation.strip():
                    raise ValueError("rich oracle produced an empty continuation")
                seed_pct, output_pct = compute_fc_percentiles(
                    prefix,
                    continuation,
                    concept.feature_index,
                    ctx.base_model,
                    ctx.base_sae,
                    ctx.percentile_table,
                    sae_layer=ctx.target_model_meta.sae_layer,
                    tokenizer_id=ctx.cfg.extraction_tokenizer_id,
                    cluster_target=cluster_target,
                )
                accept_threshold_points = percentile_fraction_to_points(
                    ctx.cfg.accept_threshold,
                    field_name="accept_threshold",
                )
                if output_pct - seed_pct < accept_threshold_points:
                    raise ValueError(
                        f"delta={output_pct - seed_pct:.4f} below "
                        f"{accept_threshold_points:.4f} percentile points "
                        f"(accept_threshold={ctx.cfg.accept_threshold:.6g})"
                    )
                return response
            except Exception as exc:
                last_failure = f"seed={seed}: {type(exc).__name__}: {exc}"
        raise self._fail(
            request,
            f"Stage 1 rich oracle exhausted concept={concept_id} "
            f"feature={concept.feature_index}; last_failure={last_failure}",
        )

    def _stage1_evidence_prefixes(
        self,
        ctx: _StageContext,
        *,
        representative_id: int,
        tokenizer: Any,
    ) -> tuple[str, ...]:
        """Round-robin real activation excerpts across allowed members."""

        cluster_target = self._cluster_target(ctx, representative_id)
        member_ids = (
            cluster_target.allowed_member_ids
            if cluster_target is not None
            else (representative_id,)
        )
        per_member: list[tuple[str, ...]] = []
        for member_id in member_ids:
            evidence = self._activation_evidence.get(member_id)
            if evidence is None:
                evidence = self._load_stage0c_evidence(
                    ctx,
                    feature_id=member_id,
                    tokenizer=tokenizer,
                )
            prefixes = _activation_evidence_prefixes(
                list(evidence),
                tokenizer=tokenizer,
                max_scored_tokens=ctx.cfg.output_length,
            )
            if prefixes:
                per_member.append(prefixes)
        if not per_member:
            raise ValueError(
                "Stage 1 real-Gemma oracle found no activation evidence for "
                f"representative {representative_id}"
            )
        # Rank-major traversal gives each real member one proposal before a
        # second example from any member, matching the diversity goal of the
        # production member-description cycle.
        return tuple(
            prefixes[rank]
            for rank in range(max(len(prefixes) for prefixes in per_member))
            for prefixes in per_member
            if rank < len(prefixes)
        )

    def _load_stage0c_evidence(
        self,
        ctx: _StageContext,
        *,
        feature_id: int,
        tokenizer: Any,
    ) -> tuple[dict[str, Any], ...]:
        """Load validated real evidence committed by the preceding Stage 0c."""

        if ctx.cfg.run_tag is None:
            raise ValueError("Stage 1 real-Gemma oracle requires cfg.run_tag")
        cache_path = axbench_paths.AxBenchBasePaths.from_config(ctx.cfg).synth_response_cache()
        if not cache_path.is_file():
            raise ValueError(
                "Stage 1 real-Gemma oracle requires the committed Stage 0c "
                f"response cache, missing at {cache_path}"
            )
        with SynthResponseCache(cache_path) as cache:
            generator = stage0c_api.Stage0cApiGenerator(
                ctx.cfg,
                response_cache=cache,
                llm_client=cast(SynthCompleter, object()),
                tokenizer=tokenizer,
            )
            examples = generator._load_or_fetch_examples(feature_id)
        evidence = tuple(
            stage0c_api._evidence_row(example, rank=rank, tokenizer=tokenizer)
            for rank, example in enumerate(examples, start=1)
        )
        if not evidence:
            raise ValueError(
                "Stage 1 real-Gemma oracle loaded empty activation evidence "
                f"for feature {feature_id}"
            )
        self._activation_evidence[feature_id] = evidence
        return evidence

    def _stage1_avoid_response(self, request: OpenRouterRequest) -> str:
        ctx = self._ctx(request)
        concept_id = self._concept_id(request)
        concept = ctx.concepts.get(concept_id)
        if concept is None:
            raise self._fail(request, f"unknown concept_id={concept_id}")
        prefix = self._stage1_prefix(self._user_prompt(request), request)
        tokenizer = get_extraction_tokenizer(ctx.cfg.extraction_tokenizer_id)
        cluster_target = self._cluster_target(ctx, concept.feature_index)
        last_failure = "no candidate seed attempted"
        for offset in range(self._max_seeds):
            seed = ctx.cfg.generation.master_seed + concept_id * 2027 + offset
            try:
                generated = self._generate_text(
                    ctx,
                    prefix,
                    seed=seed,
                    # The oracle is only the proposal source. Keep a real
                    # Gemma decode in this boundary, but do not turn every
                    # strict-zero retry into a 192-token autoregressive run.
                    # The production parser explicitly accepts responses
                    # shorter than ``output_length``; the real SAE forward
                    # below remains the sole acceptance authority.
                    max_new_tokens=2,
                )
                candidate = " " + generated.strip()
                response = self._stage1_response(
                    self._user_prompt(request),
                    prefix,
                    candidate,
                )
                continuation = parse_stage1_continuation_response(
                    response=response,
                    envelope=self._envelope(self._user_prompt(request)),
                    prefix_text=prefix,
                    output_length=ctx.cfg.output_length,
                    tokenizer=tokenizer,
                    subject="real-Gemma Stage 1 avoiding continuation",
                )
                if not continuation.strip():
                    raise ValueError("avoid oracle produced an empty continuation")
                outcome = classify_avoid_continuation(
                    prefix_text=prefix,
                    continuation_text=continuation,
                    f_c=concept.feature_index,
                    base_model=ctx.base_model,
                    base_sae=ctx.base_sae,
                    sae_layer=ctx.target_model_meta.sae_layer,
                    extraction_threshold=ctx.cfg.extraction_threshold,
                    tokenizer_id=ctx.cfg.extraction_tokenizer_id,
                    cluster_target=cluster_target,
                )
                if outcome is not AvoidContinuationOutcome.INACTIVE:
                    raise ValueError(f"strict-zero outcome={outcome}")
                return response
            except Exception as exc:
                last_failure = f"seed={seed}: {type(exc).__name__}: {exc}"
        raise self._fail(
            request,
            f"Stage 1 avoid oracle exhausted concept={concept_id} "
            f"feature={concept.feature_index}; last_failure={last_failure}",
        )

    def _judge_response(self, request: OpenRouterRequest) -> str:
        raw = "Rating: [[1]]"
        if parse_axbench_rating(raw) is None:
            raise self._fail(request, "internal judge response failed production parser")
        return raw

    def _generate_text(
        self,
        ctx: _StageContext,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        tokenizer = get_extraction_tokenizer(ctx.cfg.extraction_tokenizer_id)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = batched_generate_seeds(
            ctx.base_model,
            [int(token_id) for token_id in token_ids],
            seeds=[seed],
            max_new_tokens=max_new_tokens,
            temperature=ctx.cfg.generation.temperature,
            top_k=ctx.cfg.generation.top_k_sampling,
            stop_token_ids=stage0c_internals._tokenizer_stop_token_ids(tokenizer),
        )
        generated_ids = generated_ids[0]
        if not generated_ids:
            raise ValueError("local Gemma generation returned no tokens")
        return tokenizer.decode(generated_ids).strip()

    def _evidence_rows(self, prompt: str, request: OpenRouterRequest) -> list[dict[str, Any]]:
        label = "Untrusted activation evidence (JSON):"
        if label not in prompt:
            raise self._fail(request, "definition prompt has no activation evidence payload")
        payload = prompt.split(label, 1)[1].lstrip()
        try:
            decoded, _end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError as exc:
            raise self._fail(request, f"could not parse definition evidence: {exc}") from exc
        if not isinstance(decoded, list) or not all(isinstance(row, dict) for row in decoded):
            raise self._fail(request, "definition evidence is not a list of objects")
        return decoded

    def _source_context(self, prompt: str, request: OpenRouterRequest) -> str:
        start = "Untrusted quoted source to continue:\n"
        envelope = self._envelope(prompt)
        end = f"\n\nEND SOURCE.\n\nLiteral begin marker: {envelope.begin}"
        start_at = prompt.find(start)
        end_at = prompt.find(end, start_at + len(start))
        if start_at < 0 or end_at < 0:
            raise self._fail(request, "continuation prompt has no bounded source context")
        source = prompt[start_at + len(start) : end_at]
        if not source:
            raise self._fail(request, "continuation source context is empty")
        return source

    def _stage1_prefix(self, prompt: str, request: OpenRouterRequest) -> str:
        return self._source_context(prompt, request)

    def _stage1_response(
        self,
        prompt: str,
        prefix: str,
        continuation: str,
    ) -> str:
        """Frame one proposal exactly as the production Stage 1 protocol."""
        envelope = self._envelope(prompt)
        return (
            envelope.begin
            + "\n"
            + prefix_echo_text(prefix)
            + continuation
            + " "
            + envelope.end
        )

    def _source_parts(self, source: str, request: OpenRouterRequest) -> tuple[str, str]:
        candidates = _candidate_sources.get()
        if candidates is not None:
            pair = candidates.get(source)
            if pair is not None:
                return pair
        raise self._fail(
            request,
            "Stage 0c source context did not match an exact live candidate/bridge pair",
        )

    def _envelope(self, prompt: str) -> ResponseEnvelope:
        match = _BEGIN_END_RE.search(prompt)
        if match is None:
            raise OfflineOpenRouterError(
                "AxBench prompt did not contain a complete nonce envelope"
            )
        return ResponseEnvelope(
            begin=match.group("begin"),
            end=match.group("end"),
        )

    def _fail(self, request: OpenRouterRequest, reason: str) -> OfflineOpenRouterError:
        message = (
            f"{reason}; ordinal={request.ordinal} url={request.url!r} "
            f"model={request.model!r} class={request.request_class!r} "
            f"cache_key_prefix={request.cache_key_prefix!r} "
            f"roles={request.message_roles!r} prompt_hash={request.prompt_hash[:16]}"
        )
        self._failures.append(message)
        return OfflineOpenRouterError(message)


__all__ = ["RealGemmaOfflineOracle"]
