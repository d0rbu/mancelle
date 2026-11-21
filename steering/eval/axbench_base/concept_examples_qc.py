"""Concept examples QC: validate candidate concepts by comparing their
description against their top activating examples from Neuronpedia, via
an LLM judge.

This is a Stage 0c lazy sub-step enabled by default. When
``cfg.enable_concept_examples_qc`` is False, the QC step is skipped
and the pipeline behaves identically to the pre-fix-F version. When
enabled (the default), each visited candidate is validated by:

1. Fetching its top-N max-activating examples from Neuronpedia via
   ``neuronpedia.SAEFeature.get``.
2. Asking an LLM judge whether the human-readable description matches
   those examples (0-2 scale, matching AxBench's judge convention).
3. Dropping concepts whose judge score falls below
   ``cfg.activation_example_judge_min_score``.

Review-fix G additions:
- ``ActivatingExample.text`` is the properly detokenized form obtained
  via ``tokenizer.convert_tokens_to_string``, not ``"".join(tokens)``.
  The latter leaks SentencePiece/BPE whitespace markers (``▁``, ``Ġ``),
  fails to decode byte-fallback tokens (``<0x0A>``) or merge WordPiece
  subword continuations (``##``).
- ``filter`` now returns a ``(survivors, QCReport)`` tuple so callers
  can inspect per-concept outcomes and persist the report to disk.
"""

from __future__ import annotations

import concurrent.futures
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger
from neuronpedia.np_sae_feature import SAEFeature

from steering.eval.axbench_base.judge_parser import parse_axbench_rating
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.llm_response_cache import make_cache_key
from steering.eval.axbench_base.templates import _safe_substitute, load_template
from steering.eval.axbench_base.types import AxBenchBaseError, QcOutcome
from steering.eval.feature_filter.types import ActivatingExample

# Default parse-retry budget for the examples-QC judge.
# 4 total attempts (1 initial + 3 retries) absorbs the ~17% bare-format
# rate observed against openai/gpt-5-nano plus a safety margin. Callers
# can override via ``make_llm_concept_examples_judge(..., parse_max_retries=N)``.
_DEFAULT_PARSE_MAX_RETRIES = 3


@dataclass(frozen=True)
class QCFeatureScore:
    """Per-feature QC outcome.

    Attributes:
        feature_index: Neuronpedia feature id.
        description: human-readable description under evaluation.
        outcome: a :class:`QcOutcome` member (``QcOutcome.ACCEPTED``,
            ``QcOutcome.REJECTED``, ``QcOutcome.FETCH_ERROR``,
            ``QcOutcome.JUDGE_ERROR``, ``QcOutcome.EMPTY_EXAMPLES``).
            Compares equal to the matching bare-string value because
            :class:`QcOutcome` is a :class:`StrEnum`.
        score: judge score (0-2 scale), or ``None`` when no judge was
            invoked (fetch error, empty examples).
        n_examples: number of examples actually fetched (0 on errors).
        error_message: populated for :attr:`QcOutcome.FETCH_ERROR` /
            :attr:`QcOutcome.JUDGE_ERROR`, ``None`` otherwise.
    """

    feature_index: int
    description: str
    outcome: QcOutcome
    score: int | None
    n_examples: int
    error_message: str | None


@dataclass(frozen=True)
class QCReport:
    """Aggregated concept-examples QC statistics.

    Holds both per-feature outcomes and summary counts. Serialized to
    ``AxBenchBasePaths.concept_examples_qc_report()`` as JSON after
    Stage 0c runs, so researchers can inspect how many
    concepts were dropped and why.

    ``n_total`` is the number of candidates passed into
    :meth:`ConceptExamplesQC.filter`; ``n_processed`` is how many were
    actually evaluated before early termination (==
    ``len(per_feature)``). They match when no ``target_surviving`` is
    given — under early termination ``n_processed < n_total``.
    """

    n_total: int
    n_accepted: int
    n_rejected: int
    n_fetch_errors: int
    n_judge_errors: int
    n_empty_examples: int
    score_histogram: dict[int, int]
    per_feature: list[QCFeatureScore]
    judge_min_score: int
    n_processed: int = 0


def _fetch_activating_examples_raw(
    model_id: str,
    sae_source: str,
    feature_id: int,
    n_examples: int,
    tokenizer: Any,
) -> list[ActivatingExample]:
    """Fetch top-N activating examples for a feature from Neuronpedia.

    Uses ``neuronpedia.SAEFeature.get`` which returns a feature object whose
    ``jsonData`` field contains an ``activations`` list. Sorts by
    ``maxValue`` descending and returns the top ``n_examples`` wrapped as
    ``ActivatingExample`` instances.

    Each example's ``text`` field is computed via
    ``tokenizer.convert_tokens_to_string(tokens)``; the
    ``peak_char_offset`` is the char position of the peak token in
    that decoded text. Using ``"".join(tokens)`` here would leak
    SentencePiece/BPE whitespace markers into the judge prompt, so we
    require a real tokenizer API.

    Entries with no ``tokens`` are skipped. A non-list ``activations``
    field raises ``AxBenchBaseError``. The ``maxValueTokenIndex`` is
    clamped to ``[0, len(tokens) - 1]`` to defend against malformed
    Neuronpedia payloads.
    """
    with profile_region(
        "concept_examples_qc.fetch_neuronpedia_activation_examples",
        feature_id=feature_id,
    ):
        feature = SAEFeature.get(model_id, sae_source, str(feature_id))
    raw = feature.jsonData
    data: Any = json.loads(raw) if isinstance(raw, str) else raw
    activations = data.get("activations") or []
    if not isinstance(activations, list):
        raise AxBenchBaseError(
            f"feature {feature_id} activations field is not a list"
        )
    samples: list[ActivatingExample] = []
    for entry in activations:
        # Malformed entries come from EXTERNAL Neuronpedia payloads, so we skip
        # rather than abort the whole run — but log at ERROR so a payload that is
        # unexpectedly malformed is visible instead of silently dropped.
        if not isinstance(entry, dict):
            logger.error(
                "concept examples QC: feature {} has a non-dict activation entry "
                "(malformed Neuronpedia payload); skipping it.",
                feature_id,
            )
            continue
        raw_tokens = entry.get("tokens") or []
        if not raw_tokens:
            logger.error(
                "concept examples QC: feature {} has an activation entry with no "
                "tokens (malformed Neuronpedia payload); skipping it.",
                feature_id,
            )
            continue
        tokens = [str(t) for t in raw_tokens]
        text = tokenizer.convert_tokens_to_string(tokens)
        max_idx = int(entry.get("maxValueTokenIndex", 0))
        # Clamp to [0, len(tokens) - 1] so malformed payloads don't crash us.
        max_idx = max(0, min(max_idx, len(tokens) - 1))
        if max_idx == 0:
            peak_char = 0
        else:
            peak_char = len(tokenizer.convert_tokens_to_string(tokens[:max_idx]))
        samples.append(
            ActivatingExample(
                tokens=tokens,
                text=text,
                max_value=float(entry.get("maxValue", 0.0)),
                max_value_token_index=max_idx,
                peak_char_offset=peak_char,
            )
        )
    samples.sort(key=lambda s: s.max_value, reverse=True)
    return samples[:n_examples]


class ConceptExamplesJudge(Protocol):
    """Callable-style protocol for judging a description against examples.

    Implementations take ``(description, examples)`` and return either an
    integer score (higher = better, 0-2 scale matching AxBench) or
    ``None`` if the judge could not produce a parseable rating.

    ``None`` is NOT the same as a score of 0 -- ``None`` means "could
    not parse", while ``0`` means "judge explicitly said no match".
    Historic conflation of these two silently rejected 16k features in
    a 9.6-hour production run (see commit history for
    ``judge_parser.py``)."""

    def __call__(
        self,
        description: str,
        examples: list[ActivatingExample],
    ) -> int | None: ...


FetchFn = Callable[[str, str, int, int, Any], list[ActivatingExample]]


class ConceptExamplesQC:
    """Orchestrate the fetch + judge flow for a batch of candidate concepts.

    Parameters:
        model_id: Neuronpedia ``model_id`` (e.g. ``"gemma-2-2b"``).
        sae_source: Neuronpedia ``sae_source`` (e.g. ``"8-gemmascope-res-16k"``).
        n_examples: how many top-activating examples to fetch per concept.
        judge: a ``ConceptExamplesJudge`` callable (or compatible).
        tokenizer: HuggingFace tokenizer used to detokenize Neuronpedia
            token lists into readable text. Must support
            ``convert_tokens_to_string(list[str]) -> str``.
        min_score: concepts with judge score ``< min_score`` are dropped.
        fetch_fn: optional override for the Neuronpedia fetch (for testing).
        max_workers: how many Neuronpedia fetches run concurrently inside
            a single batch. Default 10 matches
            :class:`~steering.eval.neuronpedia_config.NeuronpediaConfig`;
            values above ~20 risk account-level 429 throttling even
            with a retry loop. Used by :meth:`filter` to parallelise
            the I/O-bound fetch step.
    """

    def __init__(
        self,
        model_id: str,
        sae_source: str,
        n_examples: int,
        judge: ConceptExamplesJudge,
        tokenizer: Any,
        min_score: int = 1,
        fetch_fn: FetchFn | None = None,
        max_workers: int = 10,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._model_id = model_id
        self._sae_source = sae_source
        self._n_examples = n_examples
        self._judge = judge
        self._tokenizer = tokenizer
        self._min_score = min_score
        self._fetch_fn: FetchFn = fetch_fn or _fetch_activating_examples_raw
        self._max_workers = max_workers

    def filter(
        self,
        candidates: list[tuple[int, str]],
        target_surviving: int | None = None,
    ) -> tuple[list[tuple[int, str]], QCReport]:
        """Run the QC step on a list of ``(feature_id, description)`` pairs.

        Returns a ``(survivors, report)`` tuple, where ``survivors`` is
        the subset of ``candidates`` that passes the judge's ``min_score``
        threshold and ``report`` captures per-concept outcomes for
        inspection. Concepts whose Neuronpedia fetch fails, return zero
        examples, or whose judge raises are classified as
        ``fetch_error``, ``empty_examples``, and ``judge_error``
        respectively.

        Parallelism and early termination
        ---------------------------------
        Candidates are processed in batches of ``max_workers``. Within
        each batch, fetches run concurrently via a
        :class:`~concurrent.futures.ThreadPoolExecutor`; judges run
        serially (the judge is typically rate-limited and parallelising
        it would risk 429s against a single provider account). When
        ``target_surviving`` is set, the batch loop stops as soon as
        the survivor count reaches the target — so a Stage 0a scan
        that only needs ``max_concepts=5`` does not have to evaluate
        all 16k+ candidates. Fetch errors, empty-example rows, and
        judge errors do NOT count toward the target (only rows with
        ``score >= min_score`` do), so the caller always gets a full
        target worth of real survivors when enough candidates exist.

        Args:
            candidates: list of ``(feature_id, description)`` pairs.
            target_surviving: optional early-termination threshold. When
                ``None`` (default), every candidate is evaluated
                (backward-compatible with the pre-early-term contract).
                ``0`` returns empty immediately; negative values raise
                :class:`ValueError`.

        Raises:
            ValueError: if ``target_surviving`` is negative.
        """
        if target_surviving is not None and target_surviving < 0:
            raise ValueError(
                f"target_surviving must be >= 0 or None, "
                f"got {target_surviving}"
            )

        n_total = len(candidates)
        if target_surviving == 0:
            # Trivial fast path: caller asked for zero survivors.
            return [], QCReport(
                n_total=n_total, n_accepted=0, n_rejected=0,
                n_fetch_errors=0, n_judge_errors=0, n_empty_examples=0,
                score_histogram={}, per_feature=[],
                judge_min_score=self._min_score,
                n_processed=0,
            )

        surviving: list[tuple[int, str]] = []
        per_feature: list[QCFeatureScore] = []
        n_fetch_errors = 0
        n_judge_errors = 0
        n_empty_examples = 0
        score_histogram: dict[int, int] = {}

        batch_size = self._max_workers
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            fetch_results = self._fetch_batch(batch)

            for (fid, desc), examples, fetch_exc in fetch_results:
                if fetch_exc is not None:
                    n_fetch_errors += 1
                    per_feature.append(
                        QCFeatureScore(
                            feature_index=fid,
                            description=desc,
                            outcome=QcOutcome.FETCH_ERROR,
                            score=None,
                            n_examples=0,
                            error_message=str(fetch_exc),
                        )
                    )
                    logger.debug(
                        f"Skipping concept feature={fid}: "
                        f"neuronpedia activations fetch failed: {fetch_exc}"
                    )
                    continue

                if not examples:
                    n_empty_examples += 1
                    per_feature.append(
                        QCFeatureScore(
                            feature_index=fid,
                            description=desc,
                            outcome=QcOutcome.EMPTY_EXAMPLES,
                            score=None,
                            n_examples=0,
                            error_message=None,
                        )
                    )
                    logger.debug(
                        f"Skipping concept feature={fid}: no activations returned"
                    )
                    continue

                judge_raised = False
                judge_error: str | None = None
                raw_score: int | None = None
                try:
                    raw_score = self._judge(desc, examples)
                except Exception as exc:
                    n_judge_errors += 1
                    judge_raised = True
                    judge_error = str(exc)
                    logger.warning(
                        f"Judge raised on feature={fid}: {exc}"
                    )

                if not judge_raised and raw_score is None:
                    # Parse failure: the judge returned ``None`` after
                    # exhausting its parse-retry budget. This is NOT a
                    # rubric-zero rating -- treat it as a judge_error so
                    # the QCReport accurately reflects "we couldn't score
                    # this feature" rather than "the judge said 0".
                    n_judge_errors += 1
                    judge_raised = True
                    judge_error = (
                        "judge returned None (could not parse any response "
                        "after retries)"
                    )
                    logger.warning(
                        f"Judge parse failure on feature={fid} after retries; "
                        "marking as judge_error (not score=0)"
                    )

                if judge_raised:
                    per_feature.append(
                        QCFeatureScore(
                            feature_index=fid,
                            description=desc,
                            outcome=QcOutcome.JUDGE_ERROR,
                            score=None,
                            n_examples=len(examples),
                            error_message=judge_error,
                        )
                    )
                    continue

                # At this point raw_score is a valid int in [0, JUDGE_MAX_SCORE].
                assert raw_score is not None
                score = raw_score
                score_histogram[score] = score_histogram.get(score, 0) + 1

                if score >= self._min_score:
                    surviving.append((fid, desc))
                    per_feature.append(
                        QCFeatureScore(
                            feature_index=fid,
                            description=desc,
                            outcome=QcOutcome.ACCEPTED,
                            score=score,
                            n_examples=len(examples),
                            error_message=None,
                        )
                    )
                else:
                    per_feature.append(
                        QCFeatureScore(
                            feature_index=fid,
                            description=desc,
                            outcome=QcOutcome.REJECTED,
                            score=score,
                            n_examples=len(examples),
                            error_message=None,
                        )
                    )
                    logger.debug(
                        f"Concept feature={fid} rejected by QC judge "
                        f"(score={score})"
                    )

            # Early termination: break once we've hit the target. We
            # always finish the current batch (all fetches were already
            # in flight) to keep the accounting simple — this means we
            # may overshoot by up to ``batch_size - 1``, but the caller
            # can trivially truncate.
            if target_surviving is not None and len(surviving) >= target_surviving:
                logger.info(
                    f"Concept examples QC: reached target_surviving="
                    f"{target_surviving} after {len(per_feature)}/"
                    f"{n_total} candidates; stopping early"
                )
                break

        n_accepted = len(surviving)
        n_processed = len(per_feature)
        # ``n_rejected`` counts only actually-rejected (score<min_score)
        # rows — under early termination, unprocessed candidates must
        # NOT be lumped in here.
        n_rejected = (
            n_processed
            - n_accepted
            - n_fetch_errors
            - n_judge_errors
            - n_empty_examples
        )
        report = QCReport(
            n_total=n_total,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_fetch_errors=n_fetch_errors,
            n_judge_errors=n_judge_errors,
            n_empty_examples=n_empty_examples,
            score_histogram=score_histogram,
            per_feature=per_feature,
            judge_min_score=self._min_score,
            n_processed=n_processed,
        )
        logger.info(
            f"Concept examples QC: accepted {n_accepted}/{n_processed} "
            f"(of {n_total} candidates; fetch_errors={n_fetch_errors}, "
            f"empty_examples={n_empty_examples}, judge_errors={n_judge_errors}, "
            f"rejected={n_rejected}, score_histogram={score_histogram})"
        )
        return surviving, report

    def _fetch_batch(
        self, batch: list[tuple[int, str]],
    ) -> list[tuple[tuple[int, str], list[ActivatingExample] | None, Exception | None]]:
        """Fetch examples for every (fid, desc) in ``batch`` concurrently.

        Returns a list of ``((fid, desc), examples, exception)`` triples
        in the same order as ``batch``. On a per-feature fetch failure,
        ``examples`` is ``None`` and ``exception`` carries the error;
        on success, ``exception`` is ``None`` and ``examples`` is the
        fetched list (which may be empty — that's an ``empty_examples``
        signal for the caller, not an error).

        The worker pool is sized to ``min(max_workers, len(batch))`` so
        a tiny trailing batch doesn't spin up pointlessly large pools.
        """
        if not batch:
            return []
        worker_count = min(self._max_workers, len(batch))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
        ) as executor:
            futures = [
                executor.submit(
                    self._fetch_fn,
                    self._model_id,
                    self._sae_source,
                    fid,
                    self._n_examples,
                    self._tokenizer,
                )
                for fid, _desc in batch
            ]
            results: list[
                tuple[tuple[int, str], list[ActivatingExample] | None, Exception | None]
            ] = []
            for pair, fut in zip(batch, futures, strict=True):
                try:
                    with profile_region("concept_examples_qc.neuronpedia_fetch_result"):
                        examples = fut.result()
                    results.append((pair, examples, None))
                except Exception as exc:
                    results.append((pair, None, exc))
            return results


def make_llm_concept_examples_judge(
    llm_client: LLMJudgeClient,
    example_window_chars: int = 240,
    max_examples_in_prompt: int = 5,
    parse_max_retries: int = _DEFAULT_PARSE_MAX_RETRIES,
    *,
    model: str,
) -> ConceptExamplesJudge:
    """Build a ``ConceptExamplesJudge`` from an ``LLMJudgeClient``.

    Loads the ``judge_concept_examples.txt`` template, formats it with the
    description and a joined block of example snippets, calls the LLM,
    and parses the response via
    :func:`~steering.eval.axbench_base.judge_parser.parse_axbench_rating`
    (the shared authoritative parser).

    Parse retry policy
    ------------------

    If the response does not contain a parseable rating, the judge
    retries up to ``parse_max_retries`` additional times. Each retry
    uses a DISTINCT ``cache_key`` (suffixed with ``_a{n}``) so that a
    cached bad first-attempt response doesn't short-circuit every
    retry into re-reading the same unparseable content. The default
    budget (3 retries -> 4 total attempts) absorbs the ~17% bare-
    format rate observed against ``openai/gpt-5-nano`` plus margin.

    After the budget is exhausted, the judge returns ``None`` (NOT
    score 0). Callers must distinguish these cases: ``None`` means
    "parse failed", ``0`` means "judge explicitly said no match".

    If the underlying ``llm_client.complete`` raises
    :class:`AxBenchBaseError` (transport-level failure that the LLM
    client's own retry budget did not overcome), the judge returns
    ``None`` immediately -- retrying here would just hammer a broken
    service that the lower layer already gave up on.

    Parameters
    ----------
    llm_client:
        The underlying LLM client.
    model:
        Provider model identity used to namespace persistent cache keys.
    example_window_chars:
        Characters on either side of each example's peak activation
        token to include in the prompt. 240 gives ~40-50 English words
        on each side, enough to disambiguate most descriptions.
    max_examples_in_prompt:
        Cap on example snippets passed to the LLM.
    parse_max_retries:
        Additional attempts beyond the first when the response does not
        contain a parseable rating. Total attempts = ``parse_max_retries + 1``.
        Must be ``>= 0``; a negative value would cause every concept to
        be silently rejected with zero LLM calls.

    Raises:
        ValueError: if ``parse_max_retries < 0``. Catching this at
            constructor time prevents the silent-100%-reject failure
            mode that ``parse_max_retries=-1`` would otherwise produce.
    """
    if parse_max_retries < 0:
        raise ValueError(
            f"parse_max_retries must be >= 0, got {parse_max_retries}"
        )

    def _judge(
        description: str, examples: list[ActivatingExample],
    ) -> int | None:
        examples_block = "\n---\n".join(
            f"Example {i + 1}:\n{ex.example_text(window_chars=example_window_chars)}"
            for i, ex in enumerate(examples[:max_examples_in_prompt])
        )
        user_prompt = _safe_substitute(
            load_template("judge_concept_examples.txt"),
            DESCRIPTION=description,
            EXAMPLES=examples_block,
        )
        messages = [{"role": "user", "content": user_prompt}]
        base_tag = f"concept_qc_{description[:40]}"
        max_total_attempts = parse_max_retries + 1
        for attempt in range(max_total_attempts):
            # Per-attempt cache key so a cached bad response doesn't
            # short-circuit every retry to the same unparseable content.
            # Attempt 0 uses the canonical (no-suffix) key for
            # backwards-compatibility with existing warm caches.
            tag = base_tag if attempt == 0 else f"{base_tag}_a{attempt}"
            cache_key = make_cache_key(
                model=model,
                prompt=user_prompt,
                tag=tag,
            )
            try:
                with profile_region(
                    "concept_examples_qc.llm_judge",
                    attempt=attempt,
                ):
                    content = llm_client.complete(messages, cache_key=cache_key)
            except AxBenchBaseError as exc:
                # Transport failure: the LLM client's own retry budget
                # is exhausted. Hammering it further here would not
                # help; surface None so the QC layer marks judge_error.
                logger.warning(
                    f"LLM call failed for concept QC: {exc}"
                )
                return None
            score = parse_axbench_rating(content)
            if score is not None:
                record_counter("concept_examples_qc.judge_success", 1)
                return score
            record_counter("concept_examples_qc.judge_parse_failure", 1)
            logger.warning(
                f"Could not parse judge response "
                f"(attempt {attempt + 1}/{max_total_attempts}); "
                f"response: {content[:200]!r}"
            )
        # All parse attempts exhausted.
        return None

    return _judge


__all__ = [
    "ConceptExamplesJudge",
    "ConceptExamplesQC",
    "QCFeatureScore",
    "QCReport",
    "QcOutcome",
    "make_llm_concept_examples_judge",
]
