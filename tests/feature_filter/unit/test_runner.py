"""FeatureFilterRunner unit tests."""
import fcntl
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from steering.eval.feature_filter.config import FeatureFilterConfig, JudgeConfig
from steering.eval.feature_filter.interp_judge import InterpJudgeRating
from steering.eval.feature_filter.neuronpedia_activations import (
    FeatureActivationPayload,
)
from steering.eval.feature_filter.runner import FeatureFilterRunner
from steering.eval.feature_filter.types import (
    SCHEMA_VERSION,
    ConcurrentRunnerError,
    CostCapExceeded,
    FeatureInterpScore,
    PromptTooLongError,
    ResumeMismatchError,
    SchemaVersionError,
)
from steering.eval.neuronpedia_client import (
    NeuronpediaFetchError,
    NeuronpediaNotPublishedError,
)
from tests.feature_filter.fixtures.payload_builder import build_entry, build_payload


class _FakeTokenizer:
    @staticmethod
    def convert_tokens_to_string(toks):
        return "".join(toks)


def _fake_cache(payloads: dict[int, FeatureActivationPayload]) -> MagicMock:
    cache = MagicMock()
    cache.get.side_effect = lambda fid: payloads[fid]
    cache.get_cached.side_effect = lambda fid: payloads.get(fid)
    return cache


def _fake_judge(ratings: dict[int, InterpJudgeRating | None]):
    call_log: list[int] = []

    def _j(description, samples):
        fid = sorted(k for k in ratings if k not in call_log)[0]
        call_log.append(fid)
        return ratings[fid]

    return _j


def _make_cfg(tmp_path: Path, **overrides: Any) -> FeatureFilterConfig:
    d = str(tmp_path)
    base = FeatureFilterConfig(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
        tokenizer_id="google/gemma-2-2b",
        output_dir=d,
        feature_ids=[42],
        max_concurrent_fetches=1,
        max_concurrent_judges=1,
        neuronpedia_cache_dir=f"{d}/np_cache",
        features_output_path=f"{d}/feature_scores.jsonl",
        accepted_features_path=f"{d}/accepted_feature_ids.txt",
        report_output_path=f"{d}/filter_report.json",
        runner_manifest_path=f"{d}/runner_manifest.json",
        lockfile_path=f"{d}/.lockfile",
        judge_cache_path=f"{d}/judge_cache.sqlite",
        judge=JudgeConfig(),
    )
    return replace(base, **overrides)


def test_runner_accepted_outcome(tmp_path) -> None:
    payload = build_payload(feature_id=42, description="German articles")
    cache = _fake_cache({42: payload})
    judge = _fake_judge({42: InterpJudgeRating(3, 4)})
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    report = runner.run()
    assert report.n_accepted == 1
    line = Path(cfg.features_output_path).read_text().strip()
    row = json.loads(line)
    assert row["feature_id"] == 42
    assert row["outcome"] == "accepted"
    assert row["interpretability"] == 3
    assert row["consistency"] == 4
    assert row["schema_version"] == SCHEMA_VERSION


def test_runner_rejected_outcome(tmp_path) -> None:
    payload = build_payload(feature_id=42)
    cache = _fake_cache({42: payload})
    judge = _fake_judge({42: InterpJudgeRating(0, 0)})
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    r = runner.run()
    assert r.n_rejected == 1 and r.n_accepted == 0


def test_runner_fetch_error_outcome(tmp_path) -> None:
    cache = MagicMock()
    cache.get.side_effect = RuntimeError("network")
    cache.get_cached.return_value = None
    judge = _fake_judge({})
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    r = runner.run()
    assert r.n_fetch_errors == 1


def test_runner_no_nonzero_outcome(tmp_path) -> None:
    # Payload where all entries are sentinel (bin_min == -1) → no non-sentinel bins
    payload = FeatureActivationPayload(
        feature_id=42,
        description="desc",
        entries=[build_entry(max_value=0.0, bin_min=-1.0, bin_max=1.0, tokens=["x"])],
        freq_hist_bar_heights=[1.0],
        freq_hist_bar_values=[0.0],
        frac_nonzero=0.0,
        max_act_approx=0.0,
    )
    cache = _fake_cache({42: payload})
    judge = _fake_judge({})
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    r = runner.run()
    assert r.n_no_nonzero == 1


def test_runner_judge_error_outcome(tmp_path) -> None:
    payload = build_payload(feature_id=42)
    cache = _fake_cache({42: payload})
    judge = _fake_judge({42: None})  # parse failure
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    r = runner.run()
    assert r.n_judge_errors == 1


def test_runner_prompt_error_outcome(tmp_path) -> None:
    payload = build_payload(feature_id=42)
    cache = _fake_cache({42: payload})

    def _raise(description, samples):
        raise PromptTooLongError("too long")

    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_raise, tokenizer=_FakeTokenizer())
    r = runner.run()
    assert r.n_prompt_errors == 1


def test_runner_duplicate_submission_fails_loud(tmp_path) -> None:
    payload = build_payload(feature_id=42)
    cache = _fake_cache({42: payload})
    cfg = _make_cfg(tmp_path, feature_ids=[42])
    judge = _fake_judge({42: InterpJudgeRating(3, 3)})
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=judge, tokenizer=_FakeTokenizer())
    # pre-populate the set to simulate a dedup-bypass
    runner._already_written.add(42)
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.run()


def test_runner_resume_skips_recorded(tmp_path) -> None:
    # Pre-populate JSONL with feature 42 as accepted
    outfile = tmp_path / "feature_scores.jsonl"
    outfile.write_text(json.dumps({
        "feature_id": 42, "description": "x", "outcome": "accepted",
        "interpretability": 3, "consistency": 4, "samples": [],
        "error_message": None, "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b", "sae_source": "8-gemmascope-res-16k",
    }) + "\n")
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 3,
        "thresholds": {"interpretability": 1, "consistency": 3},
        "judge_model": "openai/gpt-5-nano",
        "started_at": "2026-04-20T00:00:00Z",
    }))
    cache = _fake_cache({})
    # If we call judge, test fails — we should skip 42 entirely.
    called = []

    def _judge(desc, samples):
        called.append(desc)
        return InterpJudgeRating(3, 3)

    cfg = _make_cfg(tmp_path, feature_ids=[42])
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_FakeTokenizer())
    report = runner.run()
    assert called == []
    # resumed feature isn't counted again
    assert report.n_total == 0


def test_runner_manifest_mismatch_refuses(tmp_path) -> None:
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "different-model",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 3,
        "thresholds": {"interpretability": 1, "consistency": 3},
        "judge_model": "openai/gpt-5-nano",
        "started_at": "2026-04-20T00:00:00Z",
    }))
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer())
    with pytest.raises(ResumeMismatchError):
        runner.run()


def test_runner_manifest_samples_per_bin_mismatch_refuses(tmp_path) -> None:
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 99,
        "judge_model": "openai/gpt-5-nano",
    }))
    cfg = _make_cfg(tmp_path, samples_per_bin=3)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ResumeMismatchError, match="samples_per_bin"):
        runner.run()


def test_runner_manifest_judge_model_mismatch_refuses(tmp_path) -> None:
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 3,
        "judge_model": "some/other-model",
    }))
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ResumeMismatchError, match="judge_model"):
        runner.run()


def test_runner_cost_cap_halts_before_next_call(tmp_path) -> None:
    payloads = {i: build_payload(feature_id=i) for i in range(5)}
    cache = _fake_cache(payloads)
    judge_calls: list[int] = []

    def _judge(desc, samples):
        judge_calls.append(1)
        return InterpJudgeRating(3, 3)

    cfg = _make_cfg(tmp_path, feature_ids=list(range(5)), max_judge_calls=2, max_concurrent_fetches=1)
    runner = FeatureFilterRunner(cfg, np_cache=cache, judge=_judge, tokenizer=_FakeTokenizer())
    with pytest.raises(CostCapExceeded):
        runner.run()
    # Exactly 2 judge calls happened before the cap halted the run
    assert len(judge_calls) == 2


def test_runner_schema_version_mismatch_refuses(tmp_path) -> None:
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": 99,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
    }))
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer())
    with pytest.raises(SchemaVersionError):
        runner.run()


def test_runner_lockfile_concurrent_refused(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    # Simulate another runner holding the lockfile
    with open(cfg.lockfile_path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pytest.skip("flock unavailable")
        lock.write(str(os.getpid()))
        lock.flush()
        runner = FeatureFilterRunner(
            cfg,
            np_cache=_fake_cache({}),
            judge=_fake_judge({}),
            tokenizer=_FakeTokenizer(),
        )
        with pytest.raises(ConcurrentRunnerError, match=str(os.getpid())):
            runner.run()


def test_runner_feature_list_defaults_to_range(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, feature_ids=None, n_sae_features=5)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    assert runner._feature_list() == [0, 1, 2, 3, 4]


def test_runner_sae_source_mismatch_refuses(tmp_path) -> None:
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "different-source",
    }))
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ResumeMismatchError, match="sae_source"):
        runner.run()


def test_runner_resume_skips_blank_and_corrupt_lines(tmp_path) -> None:
    # JSONL with blank line + corrupt line + valid line
    outfile = tmp_path / "feature_scores.jsonl"
    outfile.write_text(
        "\n"
        "{this is not json\n"
        + json.dumps({
            "feature_id": 42, "outcome": "accepted", "schema_version": SCHEMA_VERSION,
        }) + "\n"
    )
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 3,
        "judge_model": "openai/gpt-5-nano",
    }))
    cfg = _make_cfg(tmp_path, feature_ids=[42])
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    report = runner.run()
    assert report.n_total == 0  # 42 was resumed


def test_runner_resume_schema_version_mismatch_in_jsonl(tmp_path) -> None:
    outfile = tmp_path / "feature_scores.jsonl"
    outfile.write_text(json.dumps({
        "feature_id": 42, "outcome": "accepted", "schema_version": 999,
    }) + "\n")
    (tmp_path / "runner_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
        "samples_per_bin": 3,
        "judge_model": "openai/gpt-5-nano",
    }))
    cfg = _make_cfg(tmp_path, feature_ids=[42])
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(SchemaVersionError):
        runner.run()


def test_runner_lockfile_read_oserror_fallback(tmp_path, monkeypatch) -> None:
    # Exercise the OSError branch when reading the existing holder PID.
    cfg = _make_cfg(tmp_path)
    with open(cfg.lockfile_path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pytest.skip("flock unavailable")

        original_open = open  # capture builtin

        real_seek = io.IOBase.seek

        def bad_seek(self, *a, **kw):
            # Only fail for our target lock file object, fallback to real for others
            if getattr(self, "_force_seek_fail", False):
                raise OSError("seek failed")
            return real_seek(self, *a, **kw)

        # Patch only the runner's file by wrapping open to mark it.
        def patched_open(path, *a, **kw):
            f = original_open(path, *a, **kw)
            if str(path) == cfg.lockfile_path:
                f._force_seek_fail = True
            return f

        monkeypatch.setattr(
            "steering.eval.feature_filter.runner.open",
            patched_open,
            raising=False,
        )
        monkeypatch.setattr(io.IOBase, "seek", bad_seek)
        runner = FeatureFilterRunner(
            cfg,
            np_cache=_fake_cache({}),
            judge=_fake_judge({}),
            tokenizer=_FakeTokenizer(),
        )
        with pytest.raises(ConcurrentRunnerError, match="holder PID unknown"):
            runner.run()


def test_runner_report_written_atomically(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer(),
    )
    runner.run()
    # No .tmp files left over
    leftover = list(tmp_path.glob("**/*.tmp"))
    assert leftover == []
    assert Path(cfg.report_output_path).is_file()


def test_runner_not_published_outcome(tmp_path) -> None:
    cache = MagicMock()
    cache.get.side_effect = NeuronpediaNotPublishedError("np does not host feature 42")
    cache.get_cached.return_value = None
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=cache, judge=_fake_judge({}), tokenizer=_FakeTokenizer()
    )
    r = runner.run()
    assert r.n_not_published == 1
    assert r.n_fetch_errors == 0
    row = json.loads(Path(cfg.features_output_path).read_text().strip())
    assert row["outcome"] == "not_published"
    assert row["schema_version"] == SCHEMA_VERSION
    assert "42" in (row["error_message"] or "")


def test_runner_generic_fetch_error_still_fetch_error(tmp_path) -> None:
    cache = MagicMock()
    cache.get.side_effect = NeuronpediaFetchError("transient 503")
    cache.get_cached.return_value = None
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=cache, judge=_fake_judge({}), tokenizer=_FakeTokenizer()
    )
    r = runner.run()
    assert r.n_fetch_errors == 1
    assert r.n_not_published == 0


def test_build_report_counts_n_not_published(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    runner = FeatureFilterRunner(
        cfg, np_cache=_fake_cache({}), judge=_fake_judge({}), tokenizer=_FakeTokenizer()
    )
    scored = [
        FeatureInterpScore(feature_id=1, description=None, outcome="not_published",
                           interpretability=None, consistency=None, samples=[],
                           error_message="x", schema_version=SCHEMA_VERSION),
        FeatureInterpScore(feature_id=2, description=None, outcome="not_published",
                           interpretability=None, consistency=None, samples=[],
                           error_message="y", schema_version=SCHEMA_VERSION),
        FeatureInterpScore(feature_id=3, description=None, outcome="fetch_error",
                           interpretability=None, consistency=None, samples=[],
                           error_message="z", schema_version=SCHEMA_VERSION),
    ]
    report = runner._build_report(scored)
    assert report.n_not_published == 2
    assert report.n_fetch_errors == 1


def test_runner_not_published_persisted_and_skipped_on_resume(tmp_path) -> None:
    """A not_published row must persist to JSONL and be SKIPPED on resume
    (no re-fetch). Locks the safety that compensates for 404 not being
    negatively cached: an un-ingested SAE is not re-hit on every resume."""
    cfg = _make_cfg(tmp_path, feature_ids=[42])

    # Pass 1: feature 42 → not_published; runner persists the row + manifest.
    cache1 = MagicMock()
    cache1.get.side_effect = NeuronpediaNotPublishedError("np 404 for 42")
    cache1.get_cached.return_value = None
    r1 = FeatureFilterRunner(
        cfg, np_cache=cache1, judge=_fake_judge({}), tokenizer=_FakeTokenizer()
    ).run()
    assert r1.n_not_published == 1
    row = json.loads(Path(cfg.features_output_path).read_text().strip())
    assert row["outcome"] == "not_published"

    # Pass 2: same output_dir/cfg. The cache MUST NOT be hit for 42.
    cache2 = MagicMock()
    cache2.get_cached.return_value = None
    cache2.get.side_effect = AssertionError(
        "resume must skip the persisted not_published feature — no re-fetch"
    )
    cfg2 = _make_cfg(tmp_path, feature_ids=[42])
    FeatureFilterRunner(
        cfg2, np_cache=cache2, judge=_fake_judge({}), tokenizer=_FakeTokenizer()
    ).run()
    cache2.get.assert_not_called()
    # The persisted row is unchanged (still exactly one not_published row).
    lines = Path(cfg.features_output_path).read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["outcome"] == "not_published"
