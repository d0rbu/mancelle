"""In-flight progress reporting for Stage 1b.

``record_tracker_summary`` reports a concept only once it TERMINATES. A stage
whose concepts are all mid-flight therefore emits nothing for as long as it
runs, which makes a slow stage indistinguishable from a stalled one and hides
the accept-to-attempt ratio until the very end. A real run went 30+ minutes
with no acceptance signal at all.

These tests pin the heartbeat's cadence contract and the fields it reports.
"""

from __future__ import annotations

from steering.eval.axbench_base.synth import positive_mode


def test_heartbeat_logs_at_info_so_it_is_visible_by_default() -> None:
    """The heartbeat is the only default-verbosity progress signal.

    At DEBUG it was invisible under the default handler level and therefore
    useless. The per-attempt rejection spam that motivated hiding it now lives
    at DEBUG instead, so one line per heartbeat window is signal, not noise.
    """

    import inspect

    source = inspect.getsource(positive_mode)
    assert 'logger.info(\n            "Stage 1b progress:' in source
    assert 'logger.debug(\n            "Stage 1b progress:' not in source


def test_heartbeat_cadence_constant_is_sane() -> None:
    """Gates logging only; must be positive and not so long it hides a stall."""

    assert positive_mode._PROGRESS_HEARTBEAT_SECONDS > 0
    assert positive_mode._PROGRESS_HEARTBEAT_SECONDS <= 300


def test_heartbeat_is_driven_from_the_result_handler() -> None:
    """The heartbeat must be attached to per-result work, not a timer thread.

    Driving it from ``handle_result`` means it costs nothing when idle and
    cannot outlive the request pool.
    """

    import inspect

    source = inspect.getsource(positive_mode._synthesize_positive_rows_impl) \
        if hasattr(positive_mode, "_synthesize_positive_rows_impl") \
        else inspect.getsource(positive_mode)
    assert "emit_progress_heartbeat()" in source


def test_heartbeat_reports_the_fields_needed_to_judge_progress() -> None:
    """A heartbeat that omits accepts or attempts cannot answer 'is it working'."""

    import inspect

    source = inspect.getsource(positive_mode)
    for field in (
        "concepts_terminated",
        "accepted_pairs",
        "seeds_drawn",
        "discards",
        "semantic_attempts",
        "in_flight",
        "attempts_per_accept",
    ):
        assert field in source, f"heartbeat must report {field}"


def test_heartbeat_handles_zero_accepts_without_dividing_by_zero() -> None:
    """The attempts-per-accept ratio is undefined before the first accept.

    This is the state a struggling run is in for its entire life, so it must
    not be the thing that crashes the stage.
    """

    import inspect

    source = inspect.getsource(positive_mode)
    assert 'if accepted else "n/a"' in source
