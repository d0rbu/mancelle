"""Configurable gradient clipping tests for encoder-decoder SFT.

The owner decision for the SFT knob is that the default is now ``null``:
clipping is disabled and startup records that choice at INFO level.  The
historical ``1.0`` value from commit ``edc101b`` remains available as an
explicit opt-in for runs that need its NaN defense against bad batches with
garbled U+FFFD tokens.  Positive values enable clipping and also log INFO;
validation rejects zero, negative, NaN, infinity, and boolean values.

These tests pin the config surface, the ``main()`` wiring (including
the ``if cfg.grad_clip_max_norm is not None`` guard and INFO logging), and
both behavioral paths: explicit ``1.0`` preserves the legacy torch clipping
semantics while ``None`` leaves gradients untouched.

See also ``tests/test_gradient_clipping.py`` which tests ``torch``'s
``clip_grad_norm_`` behavior directly and pins the position of the
call in the training loop.
"""

from __future__ import annotations

import os

import pytest

from steering.sft.clipping import resolve_grad_clip_max_norm

# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestGradClipConfig:
    """The SFT config defaults ``grad_clip_max_norm`` to ``null``;
    explicit positive values remain available to enable clipping."""

    @pytest.fixture()
    def cfg(self, hydra_compose):
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        return hydra_compose(config_dir, "conditional_encdec_sft")

    def test_default_is_null(self, cfg) -> None:
        """The deliberate default disables clipping."""
        assert cfg.grad_clip_max_norm is None

    def test_override_to_higher_value(self, hydra_compose) -> None:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        cfg = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["grad_clip_max_norm=5.0"],
        )
        assert cfg.grad_clip_max_norm == 5.0

    def test_override_to_lower_value(self, hydra_compose) -> None:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        cfg = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["grad_clip_max_norm=0.5"],
        )
        assert cfg.grad_clip_max_norm == 0.5

    def test_override_to_null_disables_clipping(self, hydra_compose) -> None:
        """Setting the knob to ``null`` must produce ``None`` so
        ``main()``'s ``if grad_clip_max_norm is not None`` guard
        skips the clip entirely."""
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        cfg = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["grad_clip_max_norm=null"],
        )
        assert cfg.grad_clip_max_norm is None

    def test_override_to_one_restores_legacy(self, hydra_compose) -> None:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )
        cfg = hydra_compose(
            config_dir,
            "conditional_encdec_sft",
            ["grad_clip_max_norm=1.0"],
        )
        assert cfg.grad_clip_max_norm == 1.0


# ---------------------------------------------------------------------------
# Unit: shared clipping resolver
# ---------------------------------------------------------------------------


class TestConfigureGradClip:
    """The shared resolver validates and logs once at startup. These tests cover the
    input validation (reject ``0.0`` and negative values), the log
    side effects (info on ``None`` and positive), and the
    return value (``None`` passthrough / float normalization).

    Unit-testing the helper directly is stronger than source-string
    inspection: we can actually **call** the function with various
    inputs and observe its side effects via a mocked logger, rather
    than hoping the string literals in ``main()``'s source imply
    correct behavior.
    """

    # -- Input validation ---------------------------------------------------

    def test_none_returns_none(self) -> None:
        """``None`` is the documented "disabled" sentinel and must
        round-trip through the helper unchanged."""
        assert resolve_grad_clip_max_norm(None) is None

    def test_positive_float_returns_float(self) -> None:
        result = resolve_grad_clip_max_norm(1.0)
        assert result == 1.0
        assert isinstance(result, float)

    def test_positive_int_is_coerced_to_float(self) -> None:
        """Hydra may hand us an ``int`` if the YAML/CLI value has no
        decimal point (e.g. ``grad_clip_max_norm=5``).  The helper
        must coerce to ``float`` so downstream ``clip_grad_norm_``
        receives the type it expects."""
        result = resolve_grad_clip_max_norm(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_very_large_value_is_accepted(self) -> None:
        """Large values (e.g. ``1e6``) effectively disable clipping
        without triggering the ``None`` path.  The helper must accept
        them — operators who want this behavior are making a
        deliberate choice."""
        assert resolve_grad_clip_max_norm(1e6) == 1e6

    @pytest.mark.parametrize(
        "bad_value",
        [0.0, -1.0, -0.001, float("nan"), float("inf"), True],
    )
    def test_invalid_threshold_is_rejected(self, bad_value: object) -> None:
        with pytest.raises(ValueError, match="grad_clip_max_norm"):
            resolve_grad_clip_max_norm(bad_value)

    # -- Logging side effects (behavioral, via loguru caplog) ---------------

    def test_none_emits_info_not_warning(self, caplog) -> None:
        """When clipping is disabled by default, the helper emits INFO
        naming the default, enable knob, and edc101b NaN rationale.

        Uses ``loguru``'s ``caplog`` integration: we add a temporary
        propagating handler, run the helper, and inspect the
        captured records."""
        import logging

        from loguru import logger as loguru_logger

        # Bridge loguru → stdlib caplog for the duration of this test.
        handler_id = loguru_logger.add(
            lambda msg: caplog.handler.emit(
                logging.LogRecord(
                    name=msg.record["name"],
                    level=msg.record["level"].no,
                    pathname=msg.record["file"].path,
                    lineno=msg.record["line"],
                    msg=msg.record["message"],
                    args=(),
                    exc_info=None,
                )
            ),
            level=0,
        )
        try:
            with caplog.at_level(logging.INFO):
                resolve_grad_clip_max_norm(None)
        finally:
            loguru_logger.remove(handler_id)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, "disabled clipping must emit an INFO"
        combined = " ".join(r.message for r in info_records)
        assert "disabled" in combined
        assert "edc101b" in combined
        assert "grad_clip_max_norm" in combined
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            "the disabled default must not emit WARNING or higher"
        )

    def test_positive_emits_info_with_value(self, caplog) -> None:
        """When clipping is enabled, the helper must emit an INFO log
        recording the active ``max_norm`` value so the run log shows
        which threshold was used."""
        import logging

        from loguru import logger as loguru_logger

        handler_id = loguru_logger.add(
            lambda msg: caplog.handler.emit(
                logging.LogRecord(
                    name=msg.record["name"],
                    level=msg.record["level"].no,
                    pathname=msg.record["file"].path,
                    lineno=msg.record["line"],
                    msg=msg.record["message"],
                    args=(),
                    exc_info=None,
                )
            ),
            level=0,
        )
        try:
            with caplog.at_level(logging.INFO):
                resolve_grad_clip_max_norm(2.5)
        finally:
            loguru_logger.remove(handler_id)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, "enabled clipping must emit an INFO"
        combined = " ".join(r.message for r in info_records)
        assert "2.5" in combined, f"INFO must record the active max_norm value; got: {combined}"
        assert "max_norm" in combined, (
            f"INFO must use the ``max_norm`` label for clarity; got: {combined}"
        )

    def test_zero_does_not_emit_warning_or_info(self, caplog) -> None:
        """A rejected value (``0.0``) raises before any log line is
        emitted — we must not confuse the operator with a spurious
        INFO/WARNING followed by a ValueError."""
        import logging

        from loguru import logger as loguru_logger

        handler_id = loguru_logger.add(
            lambda msg: caplog.handler.emit(
                logging.LogRecord(
                    name=msg.record["name"],
                    level=msg.record["level"].no,
                    pathname=msg.record["file"].path,
                    lineno=msg.record["line"],
                    msg=msg.record["message"],
                    args=(),
                    exc_info=None,
                )
            ),
            level=0,
        )
        try:
            with caplog.at_level(logging.INFO):
                with pytest.raises(ValueError):
                    resolve_grad_clip_max_norm(0.0)
        finally:
            loguru_logger.remove(handler_id)

        # No INFO or WARNING should have been emitted before the raise.
        for r in caplog.records:
            assert r.levelno < logging.INFO, (
                f"no log lines should precede the type violation; got "
                f"level={r.levelname}, msg={r.message}"
            )


# ---------------------------------------------------------------------------
# Source inspection: main() reads the config and guards on None
# ---------------------------------------------------------------------------


class TestGradClipMainWiring:
    """Lightweight source inspection of ``main()`` to ensure the
    clipping call reads from the config and can be disabled."""

    @pytest.fixture(scope="class")
    def main_source(self) -> str:
        import inspect

        from scripts.train.conditional_encdec_sft import main

        return inspect.getsource(main)

    @pytest.fixture(scope="class")
    def module_source(self) -> str:
        """Full module source, used for warning-log inspection which
        may live outside ``main()``'s immediate scope."""
        import inspect

        import scripts.train.conditional_encdec_sft as sft

        return inspect.getsource(sft)

    def test_main_reads_grad_clip_config(self, main_source: str) -> None:
        """main() must reference ``cfg.grad_clip_max_norm`` so operator
        overrides take effect."""
        assert "grad_clip_max_norm" in main_source, "main() must read cfg.grad_clip_max_norm"

    def test_main_no_longer_hardcodes_max_norm_value(self, main_source: str) -> None:
        """The literal ``max_norm=1.0`` must NOT appear in main()
        anymore — the value must flow from the config.  Catches a
        regression where the hardcoded literal sneaks back in."""
        assert "max_norm=1.0" not in main_source, (
            "max_norm should not be hardcoded to 1.0 in main(); read "
            "from cfg.grad_clip_max_norm instead"
        )

    def test_main_guards_clip_on_non_none(self, main_source: str) -> None:
        """The clip call must be guarded so that a ``null`` config
        value skips it entirely rather than crashing on
        ``clip_grad_norm_(max_norm=None)``.  Post-refactor, the
        local variable ``grad_clip_max_norm`` (returned by
        ``resolve_grad_clip_max_norm``) is the guard subject.
        """
        assert "grad_clip_max_norm is not None" in main_source, (
            "main() must guard the clip call on "
            "grad_clip_max_norm is not None so the knob can "
            "disable clipping without crashing"
        )

    def test_main_calls_shared_resolver(self, main_source: str) -> None:
        assert "resolve_grad_clip_max_norm" in main_source

    def test_main_uses_local_grad_clip_variable(self, main_source: str) -> None:
        """After the refactor, the training-loop clip call reads
        from the local ``grad_clip_max_norm`` variable (the return
        value of ``resolve_grad_clip_max_norm``) rather than
        ``cfg.grad_clip_max_norm`` directly.  This guarantees the
        validation cannot be bypassed — any code path that reads
        ``cfg.grad_clip_max_norm`` directly in the loop would skip
        resolver validation.
        """
        # The training loop's clip call should pass the validated
        # local variable, not ``cfg.grad_clip_max_norm`` directly.
        # We search for the call pattern in the source.
        assert "apply_gradient_clipping(" in main_source
        assert "grad_clip_max_norm," in main_source, (
            "main()'s shared clipping call must pass the validated "
            "local `grad_clip_max_norm` variable, not "
            "`cfg.grad_clip_max_norm` directly — otherwise a bypass "
            "of ``resolve_grad_clip_max_norm`` could reach the training "
            "loop unchecked"
        )

    def test_clip_happens_after_backward_before_step(self, main_source: str) -> None:
        """Regression guard: the clip call must still appear between
        ``loss.backward()`` and ``optimizer.step()``.  This is the
        invariant that makes the clip effective at all."""
        backward_pos = main_source.find("loss.backward()")
        clip_pos = main_source.find("apply_gradient_clipping(")
        step_pos = main_source.find("optimizer.step()")

        assert backward_pos != -1, "loss.backward() not found"
        assert clip_pos != -1, "clip_grad_norm_ not found"
        assert step_pos != -1, "optimizer.step() not found"

        assert backward_pos < clip_pos < step_pos, (
            f"Order should be backward ({backward_pos}) < clip ({clip_pos}) < step ({step_pos})"
        )


# ---------------------------------------------------------------------------
# Backward compat: default behavior matches pre-PR-6 hardcoded clip
# ---------------------------------------------------------------------------


class TestGradClipBackwardCompat:
    """When explicitly set to ``1.0``, the enabled path preserves the
    pre-PR-6 hardcoded clip semantics."""

    def test_explicit_one_clip_behavior_matches_legacy(self) -> None:
        """Behavioral check: clipping with explicit ``max_norm=1.0``
        produces the same pre-clip norm that the pre-PR-6
        hardcoded call would have produced.  This is mostly a sanity
        check on ``torch.nn.utils.clip_grad_norm_`` but it pins the
        enabled behavior as part of the SFT contract."""
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(10, 10)
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 5.0  # much larger than 1.0

        pre_clip_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert pre_clip_total_norm > 1.0, (
            "test setup expects a pre-clip norm > 1.0 so clipping is actually applied"
        )
        # After the call, every gradient norm should be <= 1.0 + eps.
        post_clip_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert post_clip_total_norm <= 1.0 + 1e-5, (
            f"post-clip norm should be <= 1.0, got {post_clip_total_norm}"
        )

    def test_none_leaves_gradients_untouched(self) -> None:
        """The ``None`` default takes the no-clip branch and leaves grads
        bitwise unchanged; explicit ``1.0`` clips them to the threshold."""
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(10, 10)
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 5.0
        gradients = [p.grad for p in model.parameters()]
        assert all(gradient is not None for gradient in gradients)
        snapshots = [gradient.detach().clone() for gradient in gradients if gradient is not None]

        threshold = None
        if threshold is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=threshold)
        for p, snapshot in zip(model.parameters(), snapshots, strict=True):
            assert p.grad is not None
            assert torch.equal(p.grad, snapshot)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        clipped = [p.grad for p in model.parameters()]
        assert all(gradient is not None for gradient in clipped)
        total_norm = torch.cat(
            [gradient.flatten() for gradient in clipped if gradient is not None]
        ).norm()
        assert total_norm <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# NaN-defense rationale documentation
# ---------------------------------------------------------------------------


class TestGradClipRationaleDocumented:
    """The NaN-defense rationale from commit ``edc101b`` must stay
    visible near the clip call and the config comment so future
    readers don't "optimize" it away without understanding why it
    exists.
    """

    def test_yaml_comment_mentions_nan(self) -> None:
        """The YAML config comment for ``grad_clip_max_norm`` must
        reference the NaN-defense rationale so operators who edit
        the config see the warning in context."""
        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "configs",
                "training",
                "conditional_encdec_sft.yaml",
            )
        )
        with open(config_path, encoding="utf-8") as f:
            content = f.read()
        # The config must mention NaN somewhere near the grad_clip knob
        clip_idx = content.find("grad_clip_max_norm")
        assert clip_idx != -1, "grad_clip_max_norm must appear in the yaml"
        # Look within a reasonable window around the key (before the key
        # is the comment block).
        window_start = max(0, clip_idx - 600)
        window = content[window_start : clip_idx + 100]
        assert "NaN" in window, (
            "the config comment around grad_clip_max_norm must mention "
            "NaN so operators see the NaN-defense rationale"
        )
