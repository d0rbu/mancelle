"""The score stage must report progress (follow-up to #1245's derive logging).

`build_bpb_adv_score_table` is the score stage's entire cost -- 1.9 h for 160k
attack rows on a real clustered sweep -- and it logged NOTHING between its setup
lines and the finished table. On that run the last log line landed at 17:46:48
and the next at 19:33:36: 107 minutes in which a hang and normal progress were
indistinguishable, and the only ETA available was extrapolated from PREVIOUS
runs' logs.

The memo matters as much as the count. The loop runs `score_pair` once per
scoring identity and reuses it for identical rows (#511, k=0 control twins), so
memo hits skip the forward passes entirely. Counting them as work would
understate the per-row cost and project a finish that arrives late.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

import steering.eval.axbench_bpb_adv.progress as progress_mod
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    lines: list[str] = []

    class _Logger:
        @staticmethod
        def info(message: str, *args: object, **kwargs: object) -> None:
            del args, kwargs
            lines.append(str(message))

        @staticmethod
        def debug(*args: object, **kwargs: object) -> None:
            del args, kwargs

    monkeypatch.setattr(progress_mod, "logger", _Logger)
    return lines


def _score_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("bpb-adv score")]


def test_score_stage_emits_progress_with_a_total(captured, monkeypatch) -> None:
    """Without a denominator a progress line cannot be turned into an ETA."""
    calls: list[int] = []

    def _fake_builder(*, candidates: pa.Table, log_progress_every: int, **_: Any):
        reporter = progress_mod.StageProgress(
            "bpb-adv score", candidates.num_rows, "row", log_every=log_progress_every
        )
        for index in range(candidates.num_rows):
            reporter.record(reused=index % 2 == 1)
            calls.append(index)
        progress_mod.logger.info(reporter.summary())

    table = pa.table({"candidate_id": [str(i) for i in range(10)]})
    _fake_builder(candidates=table, log_progress_every=5)

    lines = _score_lines(captured)
    assert lines, "the score stage must emit progress lines"
    assert "5/10 rows" in lines[0]
    assert "10/10 rows" in lines[1]
    assert "eta=" in lines[0], "a progress line without an ETA is not actionable"


def test_memo_hits_are_reported_as_reuse_not_work(captured) -> None:
    """Memo hits skip the forward passes; they must not inflate the rate."""
    reporter = progress_mod.StageProgress("bpb-adv score", 4, "row", log_every=1)
    reporter.record(reused=True)
    reporter.record(reused=True)
    reporter.record(reused=False)
    lines = _score_lines(captured)
    assert "reused=2" in lines[-1]
    summary = reporter.summary()
    assert "1 derived" in summary and "2 reused" in summary


def test_builder_accepts_and_honours_log_progress_every() -> None:
    """The knob must reach the reporter, not sit unread in the signature.

    `log_progress_every` was already threaded from config into this stage and had
    no emitter on this path -- the gap this change closes -- so a signature that
    accepts it while ignoring it would look identical from the outside.
    """
    import inspect

    sig = inspect.signature(build_bpb_adv_score_table)
    assert "log_progress_every" in sig.parameters
    src = inspect.getsource(build_bpb_adv_score_table)
    assert "StageProgress(" in src, "the builder must construct a reporter"
    assert "log_every=log_progress_every" in src, (
        "the parameter must be passed to the reporter; accepting it and "
        "dropping it silently reverts to the default cadence"
    )
    assert "progress.record(reused=True)" in src and "progress.record(reused=False)" in src, (
        "both the memo-hit and the real-work branches must be recorded, or the "
        "reuse split is wrong"
    )
