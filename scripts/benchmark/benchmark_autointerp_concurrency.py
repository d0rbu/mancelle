"""In-process concurrency benchmark for the autointerp intruder-judge filter.

Single-process proof that raising ``IntruderJudgeConfig.max_concurrent`` speeds up a
rate-limited-judge workload without changing a single output byte. Runs the real
``AutoInterpFilterRunner`` twice over the SAME deterministic synthetic workload --
once serially (``max_concurrent=1``) and once concurrently (``max_concurrent=N``) --
against a fake judge that ``time.sleep(delay_seconds)`` per call (modelling a
rate-limited network judge) and a deterministic in-memory token-example provider. It
never calls OpenRouter.

Replaces the deleted process-sharding benchmark (``benchmark_autointerp_target_sharding.py``)
now that target sharding itself has been removed in favor of in-process judge
concurrency (see
``docs/superpowers/plans/2026-07-10-autointerp-single-process-concurrency.md``).

No multiprocessing. No plan/combine. No sharding.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
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
    ExampleRenderer,
    IntruderTrial,
    IntruderTrialResult,
    ScoringTarget,
    TokenExampleProvider,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import NonNegativeInt, PositiveInt


class _DeterministicProvider:
    """Synthetic, deterministic token-example provider standing in for
    ``CachedTokenExampleProvider``: no shard files are ever read. Each target's
    windows are derived purely from its own feature id, so a given
    ``(n_targets, n_trials)`` call always yields the same trials regardless of how
    many workers score them.
    """

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
        non_activating = tuple(
            _ref(
                fid=fid,
                prefix=f"n{fid}-{idx}-",
                doc_id=fid * 10_000 + 5_000 + idx,
                active=False,
                activation=0.0,
            )
            for idx in range(self._n_windows)
        )
        return TokenExampleSet(positives=positives, non_activating=non_activating)

    def cache_context(self) -> dict:
        return {
            "provider": "benchmark_autointerp_concurrency_v1",
            "n_windows": self._n_windows,
            "targets": [target.target_key for target in self._targets],
        }


class _SleepingJudge:
    """Fake judge modelling a rate-limited network call: sleeps ``delay_seconds``
    per trial, then returns a verdict determined solely by the trial's own
    ``intruder_index`` -- never by wall-clock arrival order -- so a serial
    (``max_concurrent=1``) and a concurrent (``max_concurrent=N``) run over the same
    trials always agree, regardless of completion order.
    """

    def __init__(self, *, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: ExampleRenderer,
    ) -> IntruderTrialResult:
        time.sleep(self._delay_seconds)
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


def _ref(*, fid: int, prefix: str, doc_id: int, active: bool, activation: float) -> TokenWindowRef:
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
            model="benchmark-sleeping-judge",
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
    delay_seconds: float,
    max_concurrent: int,
) -> tuple[AutoInterpFilterConfig, float]:
    cfg = _cfg(
        root=root,
        feature_dir=feature_dir,
        n_targets=n_targets,
        n_trials=n_trials,
        max_concurrent=max_concurrent,
    )
    provider = _DeterministicProvider(n_windows=n_windows)
    judge = _SleepingJudge(delay_seconds=delay_seconds)
    start = time.perf_counter()
    AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=cast(TokenExampleProvider, provider),
        renderer=TokenWindowRenderer(),
    ).run()
    elapsed = time.perf_counter() - start
    return cfg, elapsed


def _snapshot(cfg: AutoInterpFilterConfig) -> dict[str, object]:
    """Canonical-artifact snapshot, normalized the same way the runner-level
    parity tests compare serial vs. concurrent output (see
    ``tests/steering/autointerp_filter/test_concurrency_parity.py``): strip the
    single-run-specific ``generated_at`` timestamp and normalize the (distinct,
    per-run) ``output_dir`` prefix out of embedded paths.
    """
    return {
        "target_scores": [
            json.loads(line) for line in Path(cfg.target_scores_path).read_text().splitlines()
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
            json.loads(line) for line in Path(cfg.trial_audit_path).read_text().splitlines()
        ],
        "trial_audit_yaml": yaml.safe_load(Path(cfg.trial_audit_yaml_path).read_text()),
    }


def _normalize_output_dir(value: object, *, output_dir: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _normalize_output_dir(item, output_dir=output_dir) for key, item in value.items()
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


def _peak_rss_bytes() -> int:
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is reported in kilobytes on Linux but bytes on macOS/BSD.
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def run_benchmark(
    *,
    n_targets: int,
    n_trials: int,
    max_concurrent: int,
    delay_seconds: float,
) -> dict[str, object]:
    """Run the same deterministic autointerp workload serially and concurrently,
    in this one process, and prove the on-disk artifacts are byte-identical.

    Raises ``AssertionError`` if the serial and concurrent canonical artifacts
    ever diverge -- this benchmark IS the byte-identity proof, so a mismatch must
    fail loudly rather than be reported as a data point.
    """
    if n_targets <= 0:
        raise ValueError("n_targets must be > 0")
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if max_concurrent <= 1:
        raise ValueError("max_concurrent must be > 1 for a concurrency proof")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be >= 0")

    n_windows = max(n_trials * 3, 16)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        feature_dir = root / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)

        serial_cfg, serial_seconds = _run_once(
            root=root,
            feature_dir=feature_dir,
            n_targets=n_targets,
            n_trials=n_trials,
            n_windows=n_windows,
            delay_seconds=delay_seconds,
            max_concurrent=1,
        )
        concurrent_cfg, concurrent_seconds = _run_once(
            root=root,
            feature_dir=feature_dir,
            n_targets=n_targets,
            n_trials=n_trials,
            n_windows=n_windows,
            delay_seconds=delay_seconds,
            max_concurrent=max_concurrent,
        )

        if _snapshot(serial_cfg) != _snapshot(concurrent_cfg):
            raise AssertionError(
                "serial (max_concurrent=1) and concurrent "
                f"(max_concurrent={max_concurrent}) runs produced diverging canonical "
                "artifacts -- in-process judge concurrency is not byte-identical to "
                "the serial baseline"
            )

        peak_rss_bytes = _peak_rss_bytes()

    return {
        "serial_seconds": serial_seconds,
        "concurrent_seconds": concurrent_seconds,
        "speedup": serial_seconds / concurrent_seconds if concurrent_seconds > 0 else float("inf"),
        "artifact_parity": True,
        "max_concurrent": max_concurrent,
        "n_targets": n_targets,
        "n_trials": n_trials,
        "peak_rss_bytes": peak_rss_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-targets", type=int, default=12)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    args = parser.parse_args()

    result = run_benchmark(
        n_targets=args.n_targets,
        n_trials=args.n_trials,
        max_concurrent=args.max_concurrent,
        delay_seconds=args.delay_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
