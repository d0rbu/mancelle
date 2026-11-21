"""Runner for token-level intruder-detection autointerp filtering."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from beartype import beartype
from loguru import logger

from nanochat.atomic_io import atomic_write_text
from steering.eval.autointerp_filter.artifacts import (
    finalize_scores,
    passes_threshold,
    write_final_outputs,
)
from steering.eval.autointerp_filter.cache import IntruderTrialCache
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig
from steering.eval.autointerp_filter.contracts import (
    runner_manifest_source_index,
    runner_manifest_token_examples,
)
from steering.eval.autointerp_filter.document_blacklist import (
    EMPTY_DOCUMENT_BLACKLIST_SHA256,
    resolve_document_blacklist,
)
from steering.eval.autointerp_filter.intruder import (
    IntruderDetectionJudge,
    build_intruder_messages,
    judge_cache_key,
    render_trial_examples,
)
from steering.eval.autointerp_filter.renderers import TokenWindowRenderer
from steering.eval.autointerp_filter.targets import resolve_targets, target_cache_payload
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    TokenExampleSampler,
    infer_n_sae_features,
    positive_highlight_indices,
)
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCostCapExceeded,
    AutoInterpFilterReport,
    ExampleRenderer,
    IntruderTrial,
    IntruderTrialResult,
    RenderedExample,
    ScoringTarget,
    TargetOutcome,
    TargetScore,
    TokenExampleProvider,
)
from steering.eval.neuronpedia_client import NeuronpediaFetchError, fetch_feature_description
from steering.eval.neuronpedia_config import NeuronpediaConfig
from steering.refined_types import PositiveInt
from steering.removed_features import reject_autointerp_clustering_manifest


@runtime_checkable
class _IntruderTrialJudge(Protocol):
    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: ExampleRenderer,
    ) -> IntruderTrialResult: ...


@dataclass(frozen=True)
class _TrialJudgeWorkItem:
    global_index: int
    target_index: int
    trial_index: int
    trial: IntruderTrial


@dataclass(frozen=True)
class _TrialJudgeOutcome:
    global_index: int
    result: IntruderTrialResult | None = None
    error: BaseException | None = None


class AutoInterpFilterRunner:
    """End-to-end token-level autointerp filter runner."""

    def __init__(
        self,
        config: AutoInterpFilterConfig,
        *,
        judge: IntruderDetectionJudge,
        token_example_provider: TokenExampleProvider | None = None,
        renderer: ExampleRenderer | None = None,
    ) -> None:
        self._cfg = config
        self._judge = judge
        self._provider = token_example_provider
        self._renderer = renderer or TokenWindowRenderer()
        self._uncached_judge_calls = 0
        self._document_blacklist = resolve_document_blacklist(config.document_blacklist_path)

    def run(self) -> AutoInterpFilterReport:
        lock_path = Path(self._cfg.lockfile_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            try:
                lock_file.seek(0)
                lock_metadata = lock_file.read().strip() or "uninitialized"
            finally:
                lock_file.close()
            raise RuntimeError(
                f"Another autointerp runner holds {lock_path} (lock metadata={lock_metadata})"
            ) from exc

        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            return self._run_locked()
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def _run_locked(self) -> AutoInterpFilterReport:
        feature_dir = Path(self._cfg.feature_dir)
        n_sae_features = infer_n_sae_features(feature_dir, self._cfg.n_sae_features)
        targets = resolve_targets(
            self._cfg,
            n_sae_features=n_sae_features,
            feature_dir=feature_dir,
        )
        if not targets and self._cfg.fail_on_empty:
            raise RuntimeError("autointerp target resolution produced no targets")

        provider = (
            CachedTokenExampleProvider(
                self._cfg,
                feature_dir=feature_dir,
                n_sae_features=n_sae_features,
            )
            if self._provider is None
            else self._provider
        )
        provider.prepare(targets)
        bulk_materialization = callable(getattr(provider, "examples_for_targets", None))

        self._ensure_manifest(
            n_sae_features=n_sae_features,
            targets=targets,
            provider_context=provider.cache_context(),
            bulk_materialization=bulk_materialization,
            write_if_missing=False,
        )

        scores, audit_rows = self._score_targets(provider, targets)

        accepted = self._accepted_scores(scores)
        accepted_ids = sorted({fid for score in accepted for fid in score.feature_ids})
        if not accepted_ids and self._cfg.fail_on_empty:
            logger.warning("autointerp filter accepted no features")

        self._ensure_manifest(
            n_sae_features=n_sae_features,
            targets=targets,
            provider_context=provider.cache_context(),
            bulk_materialization=bulk_materialization,
            write_if_missing=True,
        )
        report = write_final_outputs(
            self._cfg,
            scores=scores,
            audit_rows=audit_rows,
            n_targets_total=len(targets),
        )
        return report

    def _score_targets(
        self,
        provider: TokenExampleProvider,
        targets: list[ScoringTarget],
    ) -> tuple[list[TargetScore], list[dict]]:
        sampler = TokenExampleSampler(
            self._cfg,
            provider=provider,
            document_blacklist=self._document_blacklist,
        )
        bulk_trial_builder = getattr(sampler, "build_trials_many", None)
        if not targets:
            trials_by_target = []
        elif callable(bulk_trial_builder):
            trials_by_target = bulk_trial_builder(targets)
        else:
            trials_by_target = [sampler.build_trials(target) for target in targets]
        if isinstance(provider, CachedTokenExampleProvider):
            provider.release_model()
            logger.info("Released token-example model/SAE from VRAM before judge phase")
        cache_judge_key = judge_cache_key(
            self._cfg.judge,
            render_context=provider.cache_context(),
        )
        with IntruderTrialCache(Path(self._cfg.judge_cache_path)) as cache:
            resolved = self._resolve_all_trials(
                targets,
                trials_by_target,
                cache,
                cache_judge_key=cache_judge_key,
            )

        scores: list[TargetScore] = []
        audit_rows: list[dict] = []
        for index, (target, trials, trial_results) in enumerate(
            zip(targets, trials_by_target, resolved, strict=True)
        ):
            if index % 50 == 0:
                logger.info(f"Scoring target {index + 1}/{len(targets)}")
            scores.append(
                self._score_target(
                    target,
                    trials,
                    trial_results,
                    audit_rows=audit_rows,
                )
            )
        return scores, audit_rows

    def _resolve_all_trials(
        self,
        targets: list[ScoringTarget],
        trials_by_target: list[list[IntruderTrial]],
        cache: IntruderTrialCache,
        *,
        cache_judge_key: str,
    ) -> list[list[IntruderTrialResult | None]]:
        """Resolve every (target, trial) globally in one deterministic pass.

        Admission is single-threaded and canonical: cache hits are filled in
        place, misses consume the judge budget in (target, trial) order, and each
        admitted miss becomes one work item.  Only execution is concurrent, so the
        set of admitted trials is fixed *before* any concurrency.  Each successfully
        parsed result is committed from the calling thread as its completion is
        observed; outcomes are still consumed in canonical order after the wave, so
        error precedence and final artifacts do not depend on completion order.  A
        budget or cache-read error raised after work was admitted is deferred until
        the wave finishes.  Insufficient-trial targets are skipped entirely (no
        cache lookup, no budget spend), mirroring the legacy early return.
        """
        min_trials = self._min_trials_for_score()
        results: list[list[IntruderTrialResult | None]] = [
            [None] * len(trials) for trials in trials_by_target
        ]
        work: list[_TrialJudgeWorkItem] = []
        delayed_error: BaseException | None = None
        for target_index, (target, trials) in enumerate(
            zip(targets, trials_by_target, strict=True)
        ):
            if len(trials) < min_trials:
                continue
            for trial_index, trial in enumerate(trials):
                try:
                    cached = cache.get(
                        trial_key=trial.trial_key,
                        target_key=target.target_key,
                        judge_key=cache_judge_key,
                    )
                except BaseException as exc:
                    if work:
                        delayed_error = exc
                        break
                    raise
                if cached is not None:
                    results[target_index][trial_index] = cached
                    continue
                try:
                    self._check_judge_budget()
                except BaseException as exc:
                    if work:
                        delayed_error = exc
                        break
                    raise
                work.append(
                    _TrialJudgeWorkItem(
                        global_index=len(work),
                        target_index=target_index,
                        trial_index=trial_index,
                        trial=trial,
                    )
                )
            if delayed_error is not None:
                break

        def persist_success(
            item: _TrialJudgeWorkItem,
            result: IntruderTrialResult,
        ) -> None:
            if result.error_message is not None:
                return
            cache.set(
                target_key=targets[item.target_index].target_key,
                judge_key=cache_judge_key,
                result=result,
            )

        outcomes = _evaluate_trial_wave(
            self._judge,
            renderer=self._renderer,
            work=work,
            max_concurrent=PositiveInt.parse(self._cfg.judge.max_concurrent),
            on_success=persist_success,
        )
        for outcome in outcomes:
            if outcome.error is not None:
                raise outcome.error
            result = outcome.result
            if result is None:
                raise RuntimeError("internal error: judge outcome missing result")
            item = work[outcome.global_index]
            results[item.target_index][item.trial_index] = result

        if delayed_error is not None:
            raise delayed_error

        return results

    def _score_target(
        self,
        target: ScoringTarget,
        trials: list[IntruderTrial],
        trial_results: list[IntruderTrialResult | None],
        *,
        audit_rows: list[dict],
    ) -> TargetScore:
        descriptions = self._descriptions(target)
        min_trials = self._min_trials_for_score()
        if len(trials) < min_trials:
            return TargetScore(
                target_key=target.target_key,
                display_name=target.display_name,
                kind=target.kind,
                feature_ids=target.feature_ids,
                score=None,
                n_correct=0,
                n_trials=0,
                outcome=TargetOutcome.insufficient_examples,
                descriptions=descriptions,
                error_message=self._insufficient_trials_message(len(trials), min_trials),
            )

        results: list[IntruderTrialResult] = []
        for trial, result in zip(trials, trial_results, strict=True):
            if result is None:
                raise RuntimeError("internal error: autointerp trial finished without a result")
            results.append(result)
            if len(audit_rows) < self._cfg.audit_sample_limit:
                audit_rows.append(self._trial_audit_row(target, trial, result))

        n_correct = sum(1 for result in results if result.correct)
        score = n_correct / len(results)
        error_message = None
        if len(results) < self._cfg.n_trials_per_target:
            error_message = (
                f"partial score from {len(results)}/{self._cfg.n_trials_per_target} "
                "requested token trials"
            )
        return TargetScore(
            target_key=target.target_key,
            display_name=target.display_name,
            kind=target.kind,
            feature_ids=target.feature_ids,
            score=score,
            n_correct=n_correct,
            n_trials=len(results),
            outcome=(TargetOutcome.accepted if self._passes(score) else TargetOutcome.rejected),
            descriptions=descriptions,
            error_message=error_message,
        )

    def _min_trials_for_score(self) -> int:
        return min(self._cfg.n_trials_per_target, self._cfg.min_trials_for_score)

    def _insufficient_trials_message(self, n_built: int, min_trials: int) -> str:
        if min_trials == self._cfg.n_trials_per_target:
            requirement = f"{self._cfg.n_trials_per_target} full token trials"
        else:
            requirement = (
                f"at least {min_trials}/{self._cfg.n_trials_per_target} requested token trials"
            )
        return f"needed {requirement}, built {n_built}"

    def _check_judge_budget(self) -> None:
        if (
            self._cfg.max_judge_calls is not None
            and self._uncached_judge_calls >= self._cfg.max_judge_calls
        ):
            raise AutoInterpCostCapExceeded(f"max_judge_calls={self._cfg.max_judge_calls} reached")
        self._uncached_judge_calls += 1

    def _passes(self, score: float) -> bool:
        return passes_threshold(score, self._cfg.score_threshold)

    def _accepted_scores(self, scores: list[TargetScore]) -> list[TargetScore]:
        return list(
            finalize_scores(
                scores,
                score_threshold=self._cfg.score_threshold,
                top_k=self._cfg.top_k,
            ).accepted_scores
        )

    def _descriptions(self, target: ScoringTarget) -> dict[int, str | None]:
        if not self._cfg.neuronpedia.enabled:
            return dict.fromkeys(target.feature_ids)
        if self._cfg.neuronpedia.model_id is None or self._cfg.neuronpedia.sae_source is None:
            raise RuntimeError(
                "neuronpedia.enabled=true requires neuronpedia.model_id and neuronpedia.sae_source"
            )
        np_cfg = NeuronpediaConfig(
            timeout_seconds=self._cfg.neuronpedia.timeout_seconds,
            max_retries=self._cfg.neuronpedia.max_retries,
            backoff_base_seconds=self._cfg.neuronpedia.backoff_base_seconds,
            backoff_max_seconds=self._cfg.neuronpedia.backoff_max_seconds,
        )
        out: dict[int, str | None] = {}
        for fid in target.feature_ids:
            try:
                out[fid] = fetch_feature_description(
                    self._cfg.neuronpedia.model_id,
                    self._cfg.neuronpedia.sae_source,
                    fid,
                    config=np_cfg,
                )
            except NeuronpediaFetchError as exc:
                logger.warning(
                    f"Neuronpedia description unavailable for feature {fid}; "
                    f"recording null description: {exc}"
                )
                out[fid] = None
        return out

    def _trial_audit_row(
        self,
        target: ScoringTarget,
        trial: IntruderTrial,
        result: IntruderTrialResult,
    ) -> dict:
        examples = result.rendered_examples
        if not examples:
            examples = tuple(render_trial_examples(trial, self._renderer))
        final_user_prompt = result.final_user_prompt
        if final_user_prompt is None:
            final_user_prompt = build_intruder_messages(
                list(examples),
                cot=self._cfg.judge.cot,
                prompt_variant=self._cfg.judge.prompt_variant,
                few_shot_set=self._cfg.judge.few_shot_set,
            )[-1]["content"]
        return {
            "target_key": target.target_key,
            "display_name": target.display_name,
            "kind": target.kind.value,
            "feature_ids": list(target.feature_ids),
            "prompt_variant": self._cfg.judge.prompt_variant,
            "few_shot_set": self._cfg.judge.few_shot_set,
            "trial_key": trial.trial_key,
            "trial_index": trial.trial_index,
            "quantile": trial.quantile,
            "intruder_index": trial.intruder_index,
            "prediction": result.prediction,
            "correct_index": result.correct_index,
            "correct": result.correct,
            "cached": result.cached,
            "response": result.response,
            "final_user_prompt_sent_to_judge": final_user_prompt,
            "examples": [
                _example_audit_payload(
                    idx,
                    example,
                    role="intruder" if idx == trial.intruder_index else "positive",
                )
                for idx, example in enumerate(examples)
            ],
        }

    def _ensure_manifest(
        self,
        *,
        n_sae_features: int,
        targets: list[ScoringTarget],
        provider_context: dict,
        bulk_materialization: bool = False,
        write_if_missing: bool,
    ) -> None:
        path = Path(self._cfg.runner_manifest_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_dir": self._cfg.feature_dir,
            "data_dir": self._cfg.data_dir,
            "n_sae_features": n_sae_features,
            "judge": asdict(self._cfg.judge),
            "token_examples": runner_manifest_token_examples(
                self._cfg,
                bulk_materialization=bulk_materialization,
            ),
            "token_example_cache_path": self._cfg.token_example_cache_path,
            "source_index": runner_manifest_source_index(self._cfg),
            "provider_context": provider_context,
            "seed": self._cfg.seed,
            "document_blacklist_path": self._cfg.document_blacklist_path,
            "document_blacklist_sha256": self._document_blacklist.sha256,
            "n_trials_per_target": self._cfg.n_trials_per_target,
            "min_trials_for_score": self._cfg.min_trials_for_score,
            "n_positive_examples": self._cfg.n_positive_examples,
            "n_quantiles": self._cfg.n_quantiles,
            "score_threshold": self._cfg.score_threshold,
            "top_k": self._cfg.top_k,
            "targets": [target_cache_payload(target) for target in targets],
            "created_at": datetime.now(UTC).isoformat(),
        }
        if path.is_file():
            existing = json.loads(path.read_text())
            reject_autointerp_clustering_manifest(
                existing,
                f"autointerp manifest {path}",
            )
            comparable_existing = _manifest_compare_payload(existing)
            comparable_new = _manifest_compare_payload(manifest)
            if comparable_existing != comparable_new:
                raise RuntimeError(
                    f"Existing autointerp manifest at {path} does not match "
                    "the current run configuration"
                )
            return
        if not write_if_missing:
            return
        atomic_write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", path)


_TRIAL_WAVE_PROGRESS_LOG_INTERVAL = 50


def _log_trial_wave_progress(completed: int, total: int) -> None:
    """Emit progress after the runner thread handles a completed outcome."""
    if completed == total or completed % _TRIAL_WAVE_PROGRESS_LOG_INTERVAL == 0:
        logger.info(f"Judged trial {completed}/{total}")


@beartype
def _evaluate_trial_wave(
    judge: _IntruderTrialJudge,
    *,
    renderer: ExampleRenderer,
    work: list[_TrialJudgeWorkItem],
    max_concurrent: PositiveInt,
    on_success: Callable[[_TrialJudgeWorkItem, IntruderTrialResult], None],
) -> list[_TrialJudgeOutcome]:
    """Evaluate fixed work and handle each success on the calling thread.

    Worker threads only run judge evaluation.  ``on_success`` runs synchronously
    on the calling thread as each result is observed, before progress is logged.
    Its failures are attached to the same global work coordinate so the final
    sort preserves canonical error precedence.
    """
    if not work:
        return []
    total = len(work)
    worker_count = min(max(1, max_concurrent), len(work))
    if worker_count == 1:
        outcomes: list[_TrialJudgeOutcome] = []
        for completed, item in enumerate(work, start=1):
            try:
                result = judge.evaluate(item.trial, renderer=renderer)
            except BaseException as exc:
                outcomes.append(
                    _TrialJudgeOutcome(
                        global_index=item.global_index,
                        error=exc,
                    )
                )
            else:
                outcomes.append(
                    _handle_trial_success(
                        item,
                        result,
                        on_success=on_success,
                    )
                )
            _log_trial_wave_progress(completed, total)
        return outcomes

    outcomes = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[IntruderTrialResult], _TrialJudgeWorkItem] = {}
        for item in work:
            context = copy_context()
            future = cast(
                Future[IntruderTrialResult],
                executor.submit(
                    context.run,
                    _evaluate_one_trial,
                    judge,
                    item.trial,
                    renderer,
                ),
            )
            futures[future] = item

        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                outcomes.append(
                    _TrialJudgeOutcome(
                        global_index=item.global_index,
                        error=exc,
                    )
                )
            else:
                outcomes.append(
                    _handle_trial_success(
                        item,
                        result,
                        on_success=on_success,
                    )
                )
            completed += 1
            _log_trial_wave_progress(completed, total)

    outcomes.sort(key=lambda outcome: outcome.global_index)
    return outcomes


def _handle_trial_success(
    item: _TrialJudgeWorkItem,
    result: IntruderTrialResult,
    *,
    on_success: Callable[[_TrialJudgeWorkItem, IntruderTrialResult], None],
) -> _TrialJudgeOutcome:
    """Run the completion side effect and retain its canonical coordinate."""
    try:
        on_success(item, result)
    except BaseException as exc:
        return _TrialJudgeOutcome(global_index=item.global_index, error=exc)
    return _TrialJudgeOutcome(global_index=item.global_index, result=result)


def _evaluate_one_trial(
    judge: _IntruderTrialJudge,
    trial: IntruderTrial,
    renderer: ExampleRenderer,
) -> IntruderTrialResult:
    return judge.evaluate(trial, renderer=renderer)


def _manifest_compare_payload(payload: dict) -> dict:
    comparable = dict(payload)
    comparable.pop("created_at", None)
    has_blacklist_path = "document_blacklist_path" in comparable
    has_blacklist_sha256 = "document_blacklist_sha256" in comparable
    if not has_blacklist_path and not has_blacklist_sha256:
        # Pre-blacklist manifests necessarily represent the disabled filter.
        # Normalize them to the current explicit empty-set identity. Serial
        # reruns retain the original bytes; successful shards rewrite the exact
        # current schema before completion publication.
        comparable["document_blacklist_path"] = None
        comparable["document_blacklist_sha256"] = EMPTY_DOCUMENT_BLACKLIST_SHA256
    elif (
        has_blacklist_path
        and comparable["document_blacklist_path"] is None
        and (not has_blacklist_sha256 or comparable["document_blacklist_sha256"] is None)
    ):
        # Compatibility with the intermediate path-only/null-digest schema is
        # also safe only for the disabled filter. A non-null path without a
        # digest remains unnormalized and therefore incompatible.
        comparable["document_blacklist_sha256"] = EMPTY_DOCUMENT_BLACKLIST_SHA256
    if comparable.get("clustering_enabled") is False:
        comparable.pop("clustering_enabled")
    if comparable.get("schema_version") == 2:
        comparable["schema_version"] = SCHEMA_VERSION
    judge = comparable.get("judge")
    if isinstance(judge, dict):
        normalized_judge = dict(judge)
        normalized_judge.pop("max_concurrent", None)
        comparable["judge"] = normalized_judge
    source_index = comparable.get("source_index")
    if isinstance(source_index, dict):
        # scan_workers is a scheduling knob (like judge.max_concurrent): it
        # cannot change any output byte, so a rerun that only bumps it must not
        # be rejected as a config mismatch.
        normalized_source_index = dict(source_index)
        normalized_source_index.pop("scan_workers", None)
        normalized_source_index.pop("shared_cache_dir", None)
        comparable["source_index"] = normalized_source_index
    return comparable


def _example_audit_payload(
    index: int,
    example: RenderedExample,
    *,
    role: str,
) -> dict:
    ref = example.ref
    positive_indices = positive_highlight_indices(
        ref,
        activation_threshold_ratio=1.0,
    )
    return {
        "index": index,
        "role": role,
        "activating": example.activating,
        "text": example.text,
        "token_strings": list(ref.token_strings),
        "token_ids": list(ref.token_ids),
        "activations": list(ref.activations),
        "highlighted_token_indices": list(example.highlighted_token_indices),
        "max_activation": ref.max_activation,
        "positive_max_token_indices": list(positive_indices),
        "ref": {
            "stable_key": ref.stable_key(),
            "shard_stem": ref.shard_stem,
            "shard_index": ref.shard_index,
            "doc_id": ref.doc_id,
            "source_text_hash": ref.source_text_hash,
            "token_start": ref.token_start,
            "token_end": ref.token_end,
            "target_feature_ids": list(ref.target_feature_ids),
            "quantile": ref.quantile,
            "active": ref.active,
        },
    }
