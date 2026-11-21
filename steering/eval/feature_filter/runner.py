"""Orchestrator for the feature_filter pipeline."""
from __future__ import annotations

import fcntl
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from steering.eval.feature_filter.bin_sampler import sample_by_bins
from steering.eval.feature_filter.config import FeatureFilterConfig
from steering.eval.feature_filter.interp_judge import (
    InterpJudge,
    marked_snippet,
)
from steering.eval.feature_filter.neuronpedia_activations import (
    NeuronpediaActivationCache,
)
from steering.eval.feature_filter.types import (
    SCHEMA_VERSION,
    BinSample,
    BinSampleRecord,
    ConcurrentRunnerError,
    CostCapExceeded,
    FeatureFilterReport,
    FeatureInterpScore,
    NoNonzeroActivationsError,
    OutcomeType,
    PromptTooLongError,
    ResumeMismatchError,
    SchemaVersionError,
)
from steering.eval.neuronpedia_client import NeuronpediaNotPublishedError

__all__ = ["FeatureFilterRunner"]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    tmp.replace(path)


class FeatureFilterRunner:
    def __init__(
        self,
        config: FeatureFilterConfig,
        *,
        np_cache: NeuronpediaActivationCache,
        judge: InterpJudge,
        tokenizer: Any,
    ) -> None:
        self._cfg = config
        self._cache = np_cache
        self._judge = judge
        self._tokenizer = tokenizer
        self._write_lock = threading.Lock()
        self._judge_calls_lock = threading.Lock()
        self._judge_calls: int = 0
        self._already_written: set[int] = set()
        self._shutdown = threading.Event()

    def _feature_list(self) -> list[int]:
        if self._cfg.feature_ids is not None:
            return list(self._cfg.feature_ids)
        return list(range(self._cfg.n_sae_features))

    def _write_manifest(self) -> None:
        path = Path(self._cfg.runner_manifest_path)
        if path.is_file():
            existing = json.loads(path.read_text())
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"runner_manifest.json schema_version={existing.get('schema_version')} "
                    f"!= {SCHEMA_VERSION}"
                )
            if existing.get("model_id") != self._cfg.model_id:
                raise ResumeMismatchError(
                    f"runner_manifest.json model_id={existing.get('model_id')!r} "
                    f"!= {self._cfg.model_id!r}"
                )
            if existing.get("sae_source") != self._cfg.sae_source:
                raise ResumeMismatchError(
                    f"runner_manifest.json sae_source={existing.get('sae_source')!r} "
                    f"!= {self._cfg.sae_source!r}"
                )
            if existing.get("samples_per_bin") != self._cfg.samples_per_bin:
                raise ResumeMismatchError(
                    f"runner_manifest.json samples_per_bin={existing.get('samples_per_bin')!r} "
                    f"!= {self._cfg.samples_per_bin!r}"
                )
            if existing.get("judge_model") != self._cfg.judge.model:
                raise ResumeMismatchError(
                    f"runner_manifest.json judge_model={existing.get('judge_model')!r} "
                    f"!= {self._cfg.judge.model!r}"
                )
            return
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_id": self._cfg.model_id,
            "sae_source": self._cfg.sae_source,
            "samples_per_bin": self._cfg.samples_per_bin,
            "thresholds": {
                "interpretability": self._cfg.interp_min_score,
                "consistency": self._cfg.consistency_min_score,
            },
            "judge_model": self._cfg.judge.model,
            "started_at": datetime.now(UTC).isoformat(),  # allow-nondeterministic: provenance timestamp
        }
        _atomic_write(path, json.dumps(manifest, indent=2))

    def _load_resume_set(self) -> set[int]:
        path = Path(self._cfg.features_output_path)
        if not path.is_file():
            return set()
        seen: set[int] = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt JSONL line skipped in {path}: {line[:80]!r}")
                continue
            if row.get("schema_version") != SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"JSONL schema_version mismatch in {path}"
                )
            seen.add(row["feature_id"])
        return seen

    def _append_row(self, row: FeatureInterpScore) -> None:
        path = Path(self._cfg.features_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            if row.feature_id in self._already_written:
                raise RuntimeError(
                    f"Bug: duplicate write for feature_id={row.feature_id}"
                )
            self._already_written.add(row.feature_id)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "feature_id": row.feature_id,
                    "description": row.description,
                    "outcome": row.outcome,
                    "interpretability": row.interpretability,
                    "consistency": row.consistency,
                    "samples": [asdict(s) for s in row.samples],
                    "error_message": row.error_message,
                    "schema_version": row.schema_version,
                    "model_id": self._cfg.model_id,
                    "sae_source": self._cfg.sae_source,
                }) + "\n")

    def _record_sample(self, s: BinSample, window_chars: int) -> BinSampleRecord:
        if s.is_absent:
            snippet = s.example.example_text(window_chars)
        else:
            snippet = marked_snippet(s.example, window_chars)
        return BinSampleRecord(
            bin_index=s.bin_index,
            band_label=s.band_label,
            is_absent=s.is_absent,
            max_value=s.max_value,
            snippet=snippet,
        )

    def _process_one(self, feature_id: int) -> FeatureInterpScore:
        try:
            payload = self._cache.get(feature_id)
        except NeuronpediaNotPublishedError as exc:
            return FeatureInterpScore(
                feature_id=feature_id, description=None, outcome="not_published",
                interpretability=None, consistency=None, samples=[],
                error_message=str(exc)[:500], schema_version=SCHEMA_VERSION,
            )
        except Exception as exc:
            return FeatureInterpScore(
                feature_id=feature_id, description=None, outcome="fetch_error",
                interpretability=None, consistency=None, samples=[],
                error_message=str(exc)[:500], schema_version=SCHEMA_VERSION,
            )

        try:
            samples = sample_by_bins(
                payload,
                samples_per_bin=self._cfg.samples_per_bin,
                tokenizer=self._tokenizer,
            )
        except NoNonzeroActivationsError as exc:
            return FeatureInterpScore(
                feature_id=feature_id, description=payload.description,
                outcome="no_nonzero_activations",
                interpretability=None, consistency=None, samples=[],
                error_message=str(exc)[:500], schema_version=SCHEMA_VERSION,
            )

        with self._judge_calls_lock:
            if (self._cfg.max_judge_calls is not None
                    and self._judge_calls >= self._cfg.max_judge_calls):
                raise CostCapExceeded(
                    f"max_judge_calls={self._cfg.max_judge_calls} reached before "
                    f"judging feature_id={feature_id}"
                )
            self._judge_calls += 1

        try:
            rating = self._judge(payload.description or "", samples)
        except PromptTooLongError as exc:
            return FeatureInterpScore(
                feature_id=feature_id, description=payload.description,
                outcome="prompt_error",
                interpretability=None, consistency=None, samples=[],
                error_message=str(exc)[:500], schema_version=SCHEMA_VERSION,
            )

        sample_records = [
            self._record_sample(s, self._cfg.judge.example_window_chars)
            for s in samples
        ]
        if rating is None:
            return FeatureInterpScore(
                feature_id=feature_id, description=payload.description,
                outcome="judge_error",
                interpretability=None, consistency=None, samples=sample_records,
                error_message="judge returned None", schema_version=SCHEMA_VERSION,
            )

        if (rating.interpretability >= self._cfg.interp_min_score
                and rating.consistency >= self._cfg.consistency_min_score):
            outcome: OutcomeType = "accepted"
        else:
            outcome = "rejected"
        return FeatureInterpScore(
            feature_id=feature_id, description=payload.description,
            outcome=outcome,
            interpretability=rating.interpretability,
            consistency=rating.consistency,
            samples=sample_records,
            error_message=None,
            schema_version=SCHEMA_VERSION,
        )

    def run(self) -> FeatureFilterReport:
        lock_path = Path(self._cfg.lockfile_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                try:
                    lock_file.seek(0)
                    holder = lock_file.read().strip() or "holder PID unknown"
                except OSError:
                    holder = "holder PID unknown"
                raise ConcurrentRunnerError(
                    f"Another runner already holds {lock_path} (holder PID: {holder})"
                ) from exc
            try:
                lock_file.seek(0)
                lock_file.truncate()
                lock_file.write(str(os.getpid()))
                lock_file.flush()
                self._write_manifest()
                resume = self._load_resume_set()
                todo = [f for f in self._feature_list() if f not in resume]
                self._already_written |= resume

                scored: list[FeatureInterpScore] = []
                with ThreadPoolExecutor(
                    max_workers=self._cfg.max_concurrent_fetches
                ) as fetch_pool:
                    futures = [fetch_pool.submit(self._process_one, fid) for fid in todo]
                    for fut in as_completed(futures):
                        if self._shutdown.is_set():
                            break
                        row = fut.result()
                        self._append_row(row)
                        scored.append(row)
                return self._build_report(scored)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _build_report(self, scored: list[FeatureInterpScore]) -> FeatureFilterReport:
        n_accepted = sum(1 for s in scored if s.outcome == "accepted")
        n_rejected = sum(1 for s in scored if s.outcome == "rejected")
        n_fetch = sum(1 for s in scored if s.outcome == "fetch_error")
        n_judge = sum(1 for s in scored if s.outcome == "judge_error")
        n_noz = sum(1 for s in scored if s.outcome == "no_nonzero_activations")
        n_prompt = sum(1 for s in scored if s.outcome == "prompt_error")
        n_not_pub = sum(1 for s in scored if s.outcome == "not_published")

        interp_hist: dict[int, int] = {}
        cons_hist: dict[int, int] = {}
        for s in scored:
            if s.interpretability is not None:
                interp_hist[s.interpretability] = interp_hist.get(s.interpretability, 0) + 1
            if s.consistency is not None:
                cons_hist[s.consistency] = cons_hist.get(s.consistency, 0) + 1

        accepted_ids = sorted(s.feature_id for s in scored if s.outcome == "accepted")
        _atomic_write(
            Path(self._cfg.accepted_features_path),
            "\n".join(str(i) for i in accepted_ids) + ("\n" if accepted_ids else ""),
        )

        report = FeatureFilterReport(
            model_id=self._cfg.model_id,
            sae_source=self._cfg.sae_source,
            samples_per_bin=self._cfg.samples_per_bin,
            thresholds={
                "interpretability": self._cfg.interp_min_score,
                "consistency": self._cfg.consistency_min_score,
            },
            n_total=len(scored),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_fetch_errors=n_fetch,
            n_judge_errors=n_judge,
            n_no_nonzero=n_noz,
            n_prompt_errors=n_prompt,
            n_not_published=n_not_pub,
            interpretability_histogram=interp_hist,
            consistency_histogram=cons_hist,
            per_feature_path=self._cfg.features_output_path,
            accepted_feature_ids_path=self._cfg.accepted_features_path,
            schema_version=SCHEMA_VERSION,
        )
        _atomic_write(
            Path(self._cfg.report_output_path),
            json.dumps(asdict(report), indent=2),
        )
        return report
