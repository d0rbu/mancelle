"""Tests for wandb step-axis configuration helper.

Background: base_train.py was logging "step" as a regular metric field rather than
using wandb's native step axis. As a result, runs appeared truncated in the WandB UI
because charts defaulted to wandb's internal `_step` counter (one tick per log call)
instead of the real training step. The fix is to call `define_metric` after init so
that all metrics are plotted against the "step" field.
"""

from nanochat.common import DummyWandb, configure_wandb_step_metric


class RecordingWandb:
    """Minimal stand-in that records define_metric calls."""

    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def define_metric(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_configure_declares_step_as_a_metric():
    run = RecordingWandb()
    configure_wandb_step_metric(run)
    assert any(call[0] == ("step",) and not call[1] for call in run.calls), (
        f"expected define_metric('step') call, got {run.calls}"
    )


def test_configure_sets_step_as_x_axis_for_all_metrics():
    run = RecordingWandb()
    configure_wandb_step_metric(run)
    wildcard_calls = [
        call for call in run.calls if call[0] == ("*",)
    ]
    assert len(wildcard_calls) == 1, (
        f"expected exactly one wildcard define_metric call, got {run.calls}"
    )
    assert wildcard_calls[0][1].get("step_metric") == "step", (
        f"expected step_metric='step', got {wildcard_calls[0][1]}"
    )


def test_configure_declares_step_before_wildcard():
    """Wandb requires the step metric to be declared before referencing it."""
    run = RecordingWandb()
    configure_wandb_step_metric(run)
    step_idx = next(i for i, c in enumerate(run.calls) if c[0] == ("step",))
    wildcard_idx = next(i for i, c in enumerate(run.calls) if c[0] == ("*",))
    assert step_idx < wildcard_idx, (
        f"'step' must be declared before '*' wildcard, got {run.calls}"
    )


def test_configure_is_noop_safe_with_DummyWandb():
    """Non-master ranks and dummy runs use DummyWandb; helper must not raise."""
    configure_wandb_step_metric(DummyWandb())
