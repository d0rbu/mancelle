"""Pin the ``compile_static`` semantics across pretrain and SFT scripts.

Slice-12 audit map P1 #2 finding: ``conditional_encdec_train.py`` and
``conditional_encdec_sft.py`` both expose a ``compile_static`` config knob
but resolve it to ``torch.compile``'s ``dynamic`` flag using DIFFERENT
truth tables.  An operator who copies ``compile_static="auto"`` from a
pretrain config to an SFT config gets *opposite* compile semantics
because SFT is always full-feature conditioned.

Resolution: extract a single helper
``steering._common.compile_helpers.resolve_compile_dynamic`` that both
scripts consume, then pin its truth table here.

Pretrain semantics:
* ``True`` / ``"true"``  -> ``dynamic=False`` (force static)
* ``False`` / ``"false"`` -> ``dynamic=True``  (force dynamic)
* ``"auto"``             -> ``dynamic = not static_shapes_available``

SFT semantics (specialisation):
* ``True``               -> ``dynamic=False`` (force static)
* ``False`` / ``"false"`` -> ``dynamic=True``  (force dynamic)
* ``"auto"``             -> ``dynamic=None``  (auto-dynamic)
  (decoder rows are a fixed shape; the varying packed tensors are
  ``mark_dynamic``'d, so auto-dynamic keeps decoder dims concrete as the
  torch>=2.13 flex FLASH backend requires while packed extents stay
  symbolic from the first trace).

The shared helper supports both modes via a ``mode`` keyword.
"""

from __future__ import annotations

import pytest
import torch as t
from torch._dynamo.utils import counters

# ---------------------------------------------------------------------------
# Pretrain mode
# ---------------------------------------------------------------------------


class TestPretrainMode:
    """``mode='pretrain'`` honours the static_shapes_available branch."""

    def test_true_forces_static(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(True, mode="pretrain", static_shapes_available=False) is False
        assert resolve_compile_dynamic(True, mode="pretrain", static_shapes_available=True) is False
        assert resolve_compile_dynamic("true", mode="pretrain", static_shapes_available=False) is False

    def test_false_forces_dynamic(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(False, mode="pretrain", static_shapes_available=True) is True
        assert resolve_compile_dynamic("false", mode="pretrain", static_shapes_available=True) is True

    def test_auto_with_static_shapes(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=True) is False

    def test_auto_without_static_shapes(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=False) is True


# ---------------------------------------------------------------------------
# SFT mode
# ---------------------------------------------------------------------------


class TestSFTMode:
    """``mode='sft'`` uses auto-dynamic compile for the conditioned auto branch."""

    def test_true_forces_static(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(True, mode="sft") is False

    def test_false_forces_dynamic(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic(False, mode="sft") is True
        assert resolve_compile_dynamic("false", mode="sft") is True

    def test_auto_is_auto_dynamic(self) -> None:
        """``"auto"`` must resolve to ``None`` (torch.compile auto-dynamic).

        ``None`` keeps the fixed-shape decoder dimensions concrete while the
        ``mark_dynamic``'d packed tensors compile symbolically.  ``True`` is a
        regression: it symbolizes the decoder causal mask's captured scalar,
        which the torch>=2.13 flex FLASH (CuteDSL) backend refuses to inline
        ("NYI: score_mod or mask_mod captures a dynamic scalar"), crashing
        FA4 at the first compiled SFT forward on Hopper.
        """
        from steering._common.compile_helpers import resolve_compile_dynamic
        resolved = resolve_compile_dynamic("auto", mode="sft")
        assert resolved is None
        # Guard the None-vs-True distinction explicitly: both are truthy-ish
        # sentinels for "not forced static", but only None is correct.
        assert resolved is not True

    def test_auto_ignores_static_shapes_keyword(self) -> None:
        """The pretrain-only keyword must not change the SFT auto branch."""
        from steering._common.compile_helpers import resolve_compile_dynamic
        assert resolve_compile_dynamic("auto", mode="sft", static_shapes_available=True) is None
        assert resolve_compile_dynamic("auto", mode="sft", static_shapes_available=False) is None


# ---------------------------------------------------------------------------
# Argument validation — bad mode / missing kwargs raise loudly
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_unknown_mode_raises(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        # Use ``"auto"`` because the True/False branches short-circuit before
        # the mode check (those branches are mode-independent).
        with pytest.raises(ValueError, match=r"mode"):
            resolve_compile_dynamic("auto", mode="rl")  # ty: ignore[invalid-argument-type]

    def test_pretrain_mode_requires_static_shapes_available(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        with pytest.raises(TypeError, match=r"static_shapes_available"):
            resolve_compile_dynamic("auto", mode="pretrain")

# ---------------------------------------------------------------------------
# Pretrain truth-table parity — exercise every (compile_static,
# static_shapes_available) combo to pin the auto/never/always semantics.
# ---------------------------------------------------------------------------


class TestPretrainParity:
    """``mode='pretrain'`` covers the full (compile_static, static_shapes_available) matrix."""

    def test_pretrain_truth_table(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        # auto branch: dynamic = not static_shapes_available
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=True) is False
        assert resolve_compile_dynamic("auto", mode="pretrain", static_shapes_available=False) is True
        # force-static and force-dynamic short-circuit regardless of shapes
        assert resolve_compile_dynamic(True, mode="pretrain", static_shapes_available=False) is False
        assert resolve_compile_dynamic(False, mode="pretrain", static_shapes_available=True) is True


# ---------------------------------------------------------------------------
# SFT truth-table parity.
# ---------------------------------------------------------------------------


class TestSFTTruthTable:
    """Pin the conditioned-only SFT compile truth table by exact identity.

    Identity (``is``) rather than truthiness: ``None`` (auto-dynamic) and
    ``False`` (force static) are both falsy, and ``bool(None) == bool(False)``
    would let a force-static regression slide through the auto branch.
    """

    @pytest.mark.parametrize(
        ("compile_static", "expected"),
        [
            (True, False),  # force static
            ("true", False),  # force static
            (False, True),  # force whole-graph dynamic (explicit escape hatch)
            ("false", True),  # force whole-graph dynamic (explicit escape hatch)
            ("auto", None),  # auto-dynamic: concrete decoder, symbolic packed
        ],
    )
    def test_helper_matches_truth_table(self, compile_static, expected) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic
        helper = resolve_compile_dynamic(compile_static, mode="sft")
        assert helper is expected, (
            f"compile_static={compile_static!r}: helper={helper!r} expected={expected!r}"
        )


# ---------------------------------------------------------------------------
# Auto-dynamic compile semantics — functional pin of the torch behaviors the
# SFT auto branch relies on.  CPU-only; no flex_attention involved.
# ---------------------------------------------------------------------------


class TestAutoDynamicCompileSemantics:
    """Pin the two ``dynamic=None`` properties conditioned SFT depends on.

    1. A ``mark_dynamic``'d dimension compiles symbolically from the first
       trace, so varying packed extents reuse one graph (no per-batch
       recompiles — the reason the old code reached for ``dynamic=True``).
    2. Unmarked dimensions stay concrete in that first graph — the property
       the torch>=2.13 flex FLASH (CuteDSL) backend requires of the decoder
       self-attention subgraph ("NYI: score_mod or mask_mod captures a
       dynamic scalar" otherwise).  Concreteness is observed via the
       recompile that a later unmarked-dim change provokes.
    """

    @staticmethod
    def _compiled_sum(dynamic: bool | None):
        def fn(marked: t.Tensor, unmarked: t.Tensor) -> t.Tensor:
            return marked.sum() + unmarked.sum()

        return t.compile(fn, dynamic=dynamic)

    def test_marked_dim_reuses_graph_and_unmarked_dim_stays_concrete(self) -> None:
        from steering._common.compile_helpers import resolve_compile_dynamic

        dynamic = resolve_compile_dynamic("auto", mode="sft")
        t._dynamo.reset()
        counters.clear()
        try:
            fn = self._compiled_sum(dynamic)
            for marked_extent in (8, 12, 16):
                marked = t.randn(marked_extent, 3)
                t._dynamo.mark_dynamic(marked, 0)
                fn(marked, t.randn(2, 3))
            base_graphs = counters["stats"]["unique_graphs"]
            assert base_graphs == 1, (
                f"marked-dynamic extents 8/12/16 must share one graph, got {base_graphs}"
            )
            # Changing the UNMARKED shape must recompile: the first graph
            # specialized it as a concrete int (what FLASH inlines).
            marked = t.randn(8, 3)
            t._dynamo.mark_dynamic(marked, 0)
            fn(marked, t.randn(4, 3))
            assert counters["stats"]["unique_graphs"] == base_graphs + 1
        finally:
            t._dynamo.reset()

    def test_detects_the_old_forced_dynamic_policy(self) -> None:
        """Known-gap check: the assertions above discriminate None from True.

        Under the OLD resolution (``dynamic=True``) every dimension is
        symbolic from the first trace, so an unmarked-dim change does NOT
        recompile.  If this test ever fails, the previous test has lost its
        power to detect a regression back to whole-graph dynamic.
        """
        t._dynamo.reset()
        counters.clear()
        try:
            fn = self._compiled_sum(True)
            fn(t.randn(8, 3), t.randn(2, 3))
            base_graphs = counters["stats"]["unique_graphs"]
            fn(t.randn(12, 3), t.randn(4, 3))
            assert counters["stats"]["unique_graphs"] == base_graphs
        finally:
            t._dynamo.reset()
