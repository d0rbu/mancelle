"""Long-running BPB-Adv stages must report progress (#1245 follow-up).

These derives run for HOURS -- a real clustered run measured ~7 min per fair
(concept, mode) group over 100 groups, with ~693 roofline bags behind them --
and they logged ONE line at the start and nothing again until they finished.
A slow run and a hung run were indistinguishable from the log, and there was no
way to project a finish time; the only usable signal was watching bag files
appear on disk from outside the process.

The subtle part is reuse. A resumed run skips already-committed bags in ~0 time.
Folding those into the rate makes the first moments look arbitrarily fast and
projects an ETA that never arrives -- exactly when the ETA matters most, since
resume is the reason a multi-day run is restartable at all.
"""

from __future__ import annotations

import re

import pytest

from steering.eval.axbench_bpb_adv.progress import StageProgress, format_duration


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collect the reporter's emitted lines."""
    lines: list[str] = []
    import steering.eval.axbench_bpb_adv.progress as progress_mod

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


def _eta_seconds(line: str) -> float | None:
    match = re.search(r"eta=([0-9.]+)([smh])", line)
    if match is None:
        return None
    value = float(match.group(1))
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]


def test_every_unit_emits_a_line_with_count_and_total(captured) -> None:
    progress = StageProgress("fair feature-GCG derive", 4, "bag")
    for _ in range(4):
        progress.record(reused=False)
    assert len(captured) == 4, "one line per completed unit"
    assert "1/4 bags" in captured[0]
    assert "4/4 bags" in captured[-1]
    assert "fair feature-GCG derive progress" in captured[0]


def test_percentage_tracks_completion(captured) -> None:
    progress = StageProgress("d", 4, "bag")
    for _ in range(4):
        progress.record(reused=False)
    assert "(25%)" in captured[0]
    assert "(100%)" in captured[-1]


class _FakeClock:
    """Deterministic clock: each read advances by a fixed step."""

    def __init__(self, step: float) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


def test_eta_is_exactly_remaining_times_the_measured_rate(captured) -> None:
    """The projection is arithmetic, not vibes -- so pin the arithmetic.

    With a deterministic clock the expected ETA is computable in closed form,
    which catches an off-by-one in `remaining` (projecting the unit just
    finished, or omitting the last one) that a "goes down over time" assertion
    would sail straight past.
    """
    # Construction reads the clock once (t=0); each record() reads it again.
    clock = _FakeClock(step=60.0)
    progress = StageProgress("d", 10, "bag", clock=clock)

    progress.record(reused=False)  # elapsed=60s, worked=1 -> 60s/bag, 9 remain
    assert _eta_seconds(captured[-1]) == pytest.approx(9 * 60.0)

    progress.record(reused=False)  # elapsed=120s, worked=2 -> 60s/bag, 8 remain
    assert _eta_seconds(captured[-1]) == pytest.approx(8 * 60.0)


def test_eta_shrinks_as_work_completes(captured) -> None:
    """A projection that does not fall as work lands is not a projection."""
    progress = StageProgress("d", 10, "bag", clock=_FakeClock(step=30.0))
    for _ in range(4):
        progress.record(reused=False)
    etas = [_eta_seconds(line) for line in captured]
    numeric_etas = [eta for eta in etas if eta is not None]
    assert len(numeric_etas) == len(etas), "every worked unit must project an ETA"
    assert numeric_etas[-1] < numeric_etas[0], f"ETA did not fall across units: {etas}"


def test_reused_units_do_not_inflate_the_rate(captured) -> None:
    """The resume case, and the reason `reused` is a separate counter.

    Skipped bags cost ~0. If they counted as work, the measured rate would
    approach zero and the ETA would collapse to nothing while ~all the real work
    was still ahead -- worse than no ETA, because it looks authoritative.
    """
    clock = _FakeClock(step=60.0)
    progress = StageProgress("d", 100, "bag", clock=clock)
    for _ in range(50):
        progress.record(reused=True)
    assert _eta_seconds(captured[-1]) is None, (
        "with no real work done yet there is no rate to project from; the line "
        "must omit the ETA rather than invent one from reuse hits"
    )
    assert "reused=50" in captured[-1]
    assert "50/100 bags" in captured[-1]

    progress.record(reused=False)
    eta = _eta_seconds(captured[-1])
    assert eta is not None, "the first worked unit establishes the rate"
    # 51 clock reads have elapsed (1 at construction + 51 records) = 3060s, all
    # of it charged to the ONE worked unit, with 49 units remaining. The point
    # is that the rate is per-WORKED-unit: had the 50 reuse hits counted as
    # work, per-unit would be ~60s and the ETA ~49 min instead of ~41 h.
    # rel tolerance because the assertion reads the FORMATTED line, which
    # rounds to 0.1h (41.65h -> "41.6h"); that is display granularity, not
    # arithmetic error. The distinction being pinned is 41 HOURS vs 49 minutes.
    assert eta == pytest.approx(49 * 3060.0, rel=0.01)


def test_reuse_is_reported_but_kept_out_of_the_rate(captured) -> None:
    """Half reused, half worked: the count includes both, the rate only work."""
    progress = StageProgress("d", 10, "bag")
    for _ in range(5):
        progress.record(reused=True)
    for _ in range(5):
        progress.record(reused=False)
    assert "10/10 bags" in captured[-1]
    assert "reused=5" in captured[-1]
    summary = progress.summary()
    assert "5 derived" in summary and "5 reused" in summary


def test_summary_reports_totals_and_the_split(captured) -> None:
    progress = StageProgress("roofline feature-GCG derive", 3, "bag")
    progress.record(reused=True)
    progress.record(reused=False)
    summary = progress.summary()
    assert summary.startswith("roofline feature-GCG derive complete: 2 bags in ")
    assert "(1 derived, 1 reused)" in summary


def test_zero_total_does_not_divide_by_zero(captured) -> None:
    """An empty derive must log, not raise: a crash here would mask the real
    reason the derive had nothing to do."""
    progress = StageProgress("d", 0, "bag")
    progress.record(reused=False)
    assert captured, "a unit was recorded, so a line must be emitted"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0s"),
        (45.0, "45s"),
        (89.9, "90s"),
        (90.0, "1.5m"),
        (600.0, "10.0m"),
        (5399.0, "90.0m"),
        (5400.0, "1.5h"),
        (86400.0, "24.0h"),
    ],
)
def test_duration_formatting_switches_units_at_the_documented_bounds(
    seconds: float, expected: str
) -> None:
    assert format_duration(seconds) == expected


def test_negative_duration_is_clamped() -> None:
    """A monotonic clock should never go backwards, but an ETA of '-3s' would be
    a reporting bug shown to a user watching a multi-day run."""
    assert format_duration(-5.0) == "0s"


# --- log_every: throttle OUTPUT, never the accounting ----------------------
#
# The score stage records 160k rows. Logging each one is unusable, but a
# throttle that also skipped the COUNTING would silently corrupt the rate and
# the reuse split -- a much worse failure than verbose logs, because the numbers
# would still look plausible.


def test_log_every_throttles_output(captured) -> None:
    progress = StageProgress("s", 100, "row", log_every=10)
    for _ in range(30):
        progress.record(reused=False)
    assert len(captured) == 3, "expected one line per 10 units"
    assert "10/100 rows" in captured[0]
    assert "30/100 rows" in captured[-1]


class _WallClock:
    """Advances when the test says time passed -- NOT when it is read.

    `_FakeClock` advances per read, which is fine for tests that read it a fixed
    number of times. It is wrong here: throttling legitimately reads the clock
    less often, so a read-driven clock would report less elapsed time for the
    throttled run and manufacture a difference that `time.monotonic()` would
    never produce.
    """

    def __init__(self, step: float) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        return self._now

    def tick(self) -> None:
        self._now += self._step


def test_log_every_still_counts_every_unit(captured) -> None:
    """The regression the throttle could introduce.

    Rate and ETA must be identical whether or not output was throttled: they are
    computed from counters, and the counters must advance on every record().
    """
    c1 = _WallClock(step=60.0)
    fast = StageProgress("s", 100, "row", log_every=10, clock=c1)
    for _ in range(10):
        c1.tick()
        fast.record(reused=False)
    throttled_eta = _eta_seconds(captured[-1])

    captured.clear()
    c2 = _WallClock(step=60.0)
    every = StageProgress("s", 100, "row", log_every=1, clock=c2)
    for _ in range(10):
        c2.tick()
        every.record(reused=False)
    unthrottled_eta = _eta_seconds(captured[-1])

    assert throttled_eta == pytest.approx(unthrottled_eta), (
        "throttling output changed the projection; the counters are being "
        "advanced only on emitted lines"
    )


def test_final_unit_always_emits_even_off_cadence(captured) -> None:
    """7 units at log_every=10 would otherwise report nothing at all."""
    progress = StageProgress("s", 7, "row", log_every=10)
    for _ in range(7):
        progress.record(reused=False)
    assert captured, "a completed stage must emit at least its final line"
    assert "7/7 rows" in captured[-1]


def test_throttled_summary_still_reports_full_totals(captured) -> None:
    progress = StageProgress("s", 50, "row", log_every=10)
    for _ in range(25):
        progress.record(reused=True)
    for _ in range(25):
        progress.record(reused=False)
    summary = progress.summary()
    assert "50 rows" in summary
    assert "25 derived" in summary and "25 reused" in summary


def test_log_every_below_one_is_clamped_not_a_zero_division(captured) -> None:
    progress = StageProgress("s", 3, "row", log_every=0)
    progress.record(reused=False)
    assert captured, "log_every=0 must behave as 1, not divide by zero"
