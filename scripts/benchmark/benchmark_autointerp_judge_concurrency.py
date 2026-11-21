"""Local proof harness for autointerp intruder-judge concurrency.

This benchmark uses the real ``AutoInterpFilterRunner`` with synthetic token
examples and a delayed fake judge. It never calls OpenRouter. The harness runs
serial and concurrent scoring, asserts normalized artifact equality, and prints
the observed wall-time speedup plus peak in-flight judge calls.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import yaml

from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    IntruderJudgeConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.intruder import (
    IntruderDetectionJudge,
    render_trial_examples,
)
from steering.eval.autointerp_filter.renderers import TokenWindowRenderer
from steering.eval.autointerp_filter.runner import AutoInterpFilterRunner
from steering.eval.autointerp_filter.types import (
    IntruderTrial,
    IntruderTrialResult,
    ScoringTarget,
    TokenExampleProvider,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import NonNegativeInt, PositiveInt


class _SyntheticProvider:
    def __init__(self, *, n_windows: int) -> None:
        self._n_windows = n_windows
        self._targets: list[ScoringTarget] = []

    def prepare(self, targets: list[ScoringTarget]) -> None:
        self._targets = list(targets)

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        fid = target.feature_ids[0]
        positives = tuple(
            _ref(
                fid=fid,
                prefix=f"p{fid}-{idx}-",
                doc_id=fid * 10_000 + idx,
                active=True,
                activation=10.0 - float(idx % 5),
            )
            for idx in range(self._n_windows)
        )
        intruders = tuple(
            _ref(
                fid=fid,
                prefix=f"n{fid}-{idx}-",
                doc_id=fid * 10_000 + 5000 + idx,
                active=False,
                activation=0.0,
            )
            for idx in range(self._n_windows)
        )
        return TokenExampleSet(positives=positives, non_activating=intruders)

    def cache_context(self) -> dict:
        return {
            "provider": "synthetic_autointerp_concurrency_benchmark_v1",
            "n_windows": self._n_windows,
            "targets": [target.target_key for target in self._targets],
        }


class _DelayedJudge:
    def __init__(self, *, delay_s: float) -> None:
        self.calls = 0
        self.peak_concurrent = 0
        self._delay_s = delay_s
        self._in_flight = 0
        self._lock = threading.Lock()

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: TokenWindowRenderer,
    ) -> IntruderTrialResult:
        with self._lock:
            self.calls += 1
            self._in_flight += 1
            self.peak_concurrent = max(self.peak_concurrent, self._in_flight)
        try:
            time.sleep(self._delay_s)
            examples = render_trial_examples(trial, renderer)
            prediction = trial.intruder_index
            return IntruderTrialResult(
                trial_key=trial.trial_key,
                prediction=prediction,
                correct_index=trial.intruder_index,
                correct=True,
                response=f"[RESPONSE]: {prediction}",
                cached=False,
                rendered_examples=tuple(examples),
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def _ref(
    *,
    fid: int,
    prefix: str,
    doc_id: int,
    active: bool,
    activation: float,
) -> TokenWindowRef:
    activations = (0.0, activation, 0.0, 0.0)
    return TokenWindowRef(
        shard_stem="synthetic",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"hash-{doc_id}",
        token_start=0,
        token_end=len(activations),
        token_ids=tuple(range(len(activations))),
        token_strings=tuple(f"{prefix}{idx}" for idx in range(len(activations))),
        target_feature_ids=(fid,),
        activations=activations,
        max_activation=max(activations),
        quantile=None,
        active=active,
    )


def _cfg(
    *,
    root: Path,
    feature_dir: Path,
    n_targets: int,
    n_trials: int,
    max_concurrent: int,
) -> AutoInterpFilterConfig:
    output_dir = root / f"mc{max_concurrent}"
    return AutoInterpFilterConfig(
        feature_dir=str(feature_dir),
        output_dir=str(output_dir),
        n_sae_features=PositiveInt.parse(n_targets),
        feature_ids=list(range(n_targets)),
        n_trials_per_target=PositiveInt.parse(n_trials),
        n_positive_examples=PositiveInt.parse(3),
        n_quantiles=PositiveInt.parse(4),
        fail_on_empty=False,
        audit_sample_limit=NonNegativeInt.parse(n_targets * n_trials),
        token_example_cache_path=str(output_dir / "token_example_cache.sqlite"),
        judge_cache_path=str(output_dir / "intruder_trial_cache.sqlite"),
        target_scores_path=str(output_dir / "target_scores.jsonl"),
        ranked_yaml_path=str(output_dir / "autointerp_ranked.yaml"),
        accepted_features_path=str(output_dir / "accepted_feature_ids.txt"),
        manual_feature_filter_path=str(output_dir / "manual_feature_filter.yaml"),
        manual_feature_filter_meta_path=str(output_dir / "manual_feature_filter.meta.json"),
        report_output_path=str(output_dir / "report.json"),
        trial_audit_path=str(output_dir / "trial_audit.jsonl"),
        trial_audit_yaml_path=str(output_dir / "trial_audit.yaml"),
        runner_manifest_path=str(output_dir / "runner_manifest.json"),
        lockfile_path=str(output_dir / ".lockfile"),
        judge=IntruderJudgeConfig(
            model="synthetic-delayed-judge",
            requests_per_minute=NonNegativeInt.parse(0),
            max_concurrent=PositiveInt.parse(max_concurrent),
        ),
        token_examples=TokenExamplesConfig(example_ctx_len=PositiveInt.parse(4)),
    )


def _run_once(
    *,
    root: Path,
    feature_dir: Path,
    n_targets: int,
    n_trials: int,
    n_windows: int,
    delay_s: float,
    max_concurrent: int,
) -> tuple[AutoInterpFilterConfig, _DelayedJudge, float]:
    cfg = _cfg(
        root=root,
        feature_dir=feature_dir,
        n_targets=n_targets,
        n_trials=n_trials,
        max_concurrent=max_concurrent,
    )
    provider = _SyntheticProvider(n_windows=n_windows)
    judge = _DelayedJudge(delay_s=delay_s)
    start = time.perf_counter()
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=cast(TokenExampleProvider, provider),
        renderer=TokenWindowRenderer(),
    ).run()
    elapsed = time.perf_counter() - start
    return cfg, judge, elapsed


def _snapshot(cfg: AutoInterpFilterConfig) -> dict[str, object]:
    return {
        "target_scores": [
            json.loads(line)
            for line in Path(cfg.target_scores_path).read_text().splitlines()
        ],
        "ranked_yaml": yaml.safe_load(Path(cfg.ranked_yaml_path).read_text()),
        "accepted_features": Path(cfg.accepted_features_path).read_text(),
        "manual_feature_filter": Path(cfg.manual_feature_filter_path).read_text(),
        "manual_feature_filter_meta": _normalize_output_dir(
            _without_generated_at(
                json.loads(Path(cfg.manual_feature_filter_meta_path).read_text())
            ),
            output_dir=Path(cfg.output_dir),
        ),
        "report": _normalize_output_dir(
            json.loads(Path(cfg.report_output_path).read_text()),
            output_dir=Path(cfg.output_dir),
        ),
        "trial_audit_jsonl": [
            json.loads(line)
            for line in Path(cfg.trial_audit_path).read_text().splitlines()
        ],
        "trial_audit_yaml": yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()),
    }


def _normalize_output_dir(value: object, *, output_dir: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _normalize_output_dir(item, output_dir=output_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_output_dir(item, output_dir=output_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(output_dir), "<OUTPUT_DIR>")
    return value


def _without_generated_at(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("generated_at", None)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-targets", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--delay-s", type=float, default=0.05)
    parser.add_argument("--keep-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.n_targets <= 0 or args.n_trials <= 0:
        raise SystemExit("n-targets and n-trials must be > 0")
    if args.max_concurrent <= 1:
        raise SystemExit("max-concurrent must be > 1 for a concurrency proof")
    n_windows = max(args.n_trials * 3, 16)

    with tempfile.TemporaryDirectory() as tmp:
        root = args.keep_dir or Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        feature_dir = root / "features"
        feature_dir.mkdir(exist_ok=True)

        serial_cfg, serial_judge, serial_elapsed = _run_once(
            root=root,
            feature_dir=feature_dir,
            n_targets=args.n_targets,
            n_trials=args.n_trials,
            n_windows=n_windows,
            delay_s=args.delay_s,
            max_concurrent=1,
        )
        async_cfg, async_judge, async_elapsed = _run_once(
            root=root,
            feature_dir=feature_dir,
            n_targets=args.n_targets,
            n_trials=args.n_trials,
            n_windows=n_windows,
            delay_s=args.delay_s,
            max_concurrent=args.max_concurrent,
        )

        artifacts_equal = _snapshot(serial_cfg) == _snapshot(async_cfg)
        payload = {
            "n_targets": args.n_targets,
            "n_trials": args.n_trials,
            "delay_s": args.delay_s,
            "serial": {
                "elapsed_s": serial_elapsed,
                "calls": serial_judge.calls,
                "peak_concurrent": serial_judge.peak_concurrent,
            },
            "concurrent": {
                "elapsed_s": async_elapsed,
                "calls": async_judge.calls,
                "peak_concurrent": async_judge.peak_concurrent,
                "max_concurrent": args.max_concurrent,
            },
            "speedup": serial_elapsed / async_elapsed if async_elapsed > 0 else None,
            "artifacts_equal": artifacts_equal,
            "output_root": str(root),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if artifacts_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
