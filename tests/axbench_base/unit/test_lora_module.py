"""Unit tests for the hand-rolled LoRAAdapter (Task 23).

Covers ``steering.eval.axbench_base.baselines.lora.module.LoRAAdapter``,
a rank-r low-rank adaptation wrapper around ``nanochat.gpt.Linear``.

Invariants pinned down by these tests:
  * At init, ``lora_B`` is bit-exact zero so ``adapter(x) == base(x)``.
  * ``lora_A`` is Kaiming-uniform-initialised (non-zero, not constant).
  * ``scaling == alpha / rank``.
  * Gradients flow to ``lora_A`` and ``lora_B`` through the adapter
    output; base weights are frozen (``requires_grad=False``).
  * ``forward`` must delegate to ``self.base(x)`` so nanochat.gpt.Linear's
    dtype-cast behaviour is preserved (bf16 input -> bf16 output).
  * Constructor validates ``rank >= 1``, ``alpha > 0``, and
    ``isinstance(base, nn.Linear)``.
  * ``list(adapter.parameters())`` with ``requires_grad=True`` exposes
    only ``lora_A`` and ``lora_B``.
  * All fields (base, lora_A, lora_B) are instance-level, not shared
    across instances.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch as t
import torch.nn as nn
from beartype.roar import BeartypeCallHintParamViolation

from nanochat.gpt import Linear as NanochatLinear
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


IN_FEATURES: int = 16
OUT_FEATURES: int = 8
RANK: int = 4
ALPHA: float = 8.0


def _make_base(
    in_features: int = IN_FEATURES,
    out_features: int = OUT_FEATURES,
    bias: bool = True,
    dtype: t.dtype = t.float32,
) -> NanochatLinear:
    """Build a deterministic NanochatLinear on CPU."""
    t.manual_seed(0xD100)
    linear = NanochatLinear(in_features, out_features, bias=bias, dtype=dtype)
    return linear


def _make_adapter(
    rank: int = RANK,
    alpha: float = ALPHA,
    base: NanochatLinear | None = None,
) -> LoRAAdapter:
    """Build a LoRAAdapter around a fresh NanochatLinear."""
    if base is None:
        base = _make_base()
    return LoRAAdapter(base=base, rank=rank, alpha=alpha)


# ---------------------------------------------------------------------------
# Forward-at-init == base(x)
# ---------------------------------------------------------------------------


class TestForwardAtInit:
    def test_forward_at_init_equals_base(self) -> None:
        """With lora_B zero-initialised, the adapter forward is bit-exact
        identical to the base forward regardless of ``lora_A`` values.
        """
        base = _make_base()
        adapter = _make_adapter(base=base)

        x = t.randn(3, IN_FEATURES)
        with t.no_grad():
            adapter_out = adapter(x)
            base_out = base(x)

        assert t.equal(adapter_out, base_out), (
            "At init (lora_B=0), adapter forward must exactly equal base forward."
        )

    def test_forward_at_init_equals_base_for_batched_3d_input(self) -> None:
        """Time-dimension inputs (B, T, C) also pass through unchanged."""
        base = _make_base()
        adapter = _make_adapter(base=base)

        x = t.randn(2, 5, IN_FEATURES)
        with t.no_grad():
            adapter_out = adapter(x)
            base_out = base(x)

        assert t.equal(adapter_out, base_out)

    def test_forward_at_init_equals_base_no_bias(self) -> None:
        """Sanity-check the no-bias codepath of nanochat.gpt.Linear."""
        base = _make_base(bias=False)
        adapter = _make_adapter(base=base)

        x = t.randn(4, IN_FEATURES)
        with t.no_grad():
            adapter_out = adapter(x)
            base_out = base(x)

        assert t.equal(adapter_out, base_out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_after_backward_gradients_flow_to_lora_params(self) -> None:
        adapter = _make_adapter()
        x = t.randn(2, IN_FEATURES, requires_grad=False)
        out = adapter(x)
        loss = out.sum()
        loss.backward()

        assert adapter.lora_A.grad is not None
        assert adapter.lora_B.grad is not None
        # Kaiming-init lora_A is non-zero, and we multiplied by it in the
        # B-accumulator, so lora_B's grad must be non-zero.
        assert t.any(adapter.lora_B.grad != 0), (
            "lora_B.grad should be non-zero after backward on a nonzero input."
        )

    def test_base_weight_has_no_grad_after_backward(self) -> None:
        adapter = _make_adapter()
        x = t.randn(2, IN_FEATURES)
        out = adapter(x)
        loss = out.sum()
        loss.backward()

        # Base weights are frozen via requires_grad=False. Autograd should
        # therefore never even populate .grad for them (it stays None).
        assert adapter.base.weight.grad is None, (
            "Frozen base.weight must not receive gradient."
        )
        if adapter.base.bias is not None:
            assert adapter.base.bias.grad is None, (
                "Frozen base.bias must not receive gradient."
            )


# ---------------------------------------------------------------------------
# Type / constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_base_is_subclass_of_nn_linear(self) -> None:
        """A plain nn.Linear (not NanochatLinear) must still be accepted
        because NanochatLinear IS an nn.Linear subclass and the adapter
        only needs the nn.Linear contract."""
        plain = nn.Linear(IN_FEATURES, OUT_FEATURES)
        adapter = LoRAAdapter(base=plain, rank=RANK, alpha=ALPHA)
        assert adapter.base is plain

    def test_non_linear_base_raises(self) -> None:
        class NotLinear(nn.Module):
            in_features = IN_FEATURES
            out_features = OUT_FEATURES

            def forward(self, x: t.Tensor) -> t.Tensor:
                return x

        bad = NotLinear()
        with pytest.raises(BeartypeCallHintParamViolation, match="base"):
            LoRAAdapter(base=cast(nn.Linear, bad), rank=RANK, alpha=ALPHA)

    def test_nanochat_linear_base_accepted(self) -> None:
        base = _make_base()
        adapter = LoRAAdapter(base=base, rank=RANK, alpha=ALPHA)
        assert adapter.base is base

    @pytest.mark.parametrize("bad_rank", [0, -1, -100])
    def test_rank_below_one_raises(self, bad_rank: int) -> None:
        base = _make_base()
        with pytest.raises(BeartypeCallHintParamViolation, match="rank"):
            LoRAAdapter(base=base, rank=bad_rank, alpha=ALPHA)

    @pytest.mark.parametrize(
        "bad_alpha",
        [0.0, -1.0, -100.0, float("nan"), float("inf")],
    )
    def test_alpha_non_positive_raises(self, bad_alpha: float) -> None:
        base = _make_base()
        with pytest.raises(BeartypeCallHintParamViolation, match="alpha"):
            LoRAAdapter(base=base, rank=RANK, alpha=bad_alpha)


# ---------------------------------------------------------------------------
# Dtype preservation (bf16)
# ---------------------------------------------------------------------------


class TestDtypePreservation:
    def test_forward_preserves_bf16_input_dtype(self) -> None:
        """The nanochat.gpt.Linear base casts weight to input dtype inside
        its forward. Because the adapter delegates to ``self.base(x)``
        rather than re-implementing F.linear, a bf16 input must stay bf16
        all the way through the LoRA output.
        """
        base = _make_base()
        adapter = _make_adapter(base=base)

        x = t.randn(2, IN_FEATURES, dtype=t.bfloat16)
        with t.no_grad():
            out = adapter(x)

        assert out.dtype == t.bfloat16, (
            f"bf16 input must produce bf16 output; got {out.dtype}."
        )

    def test_forward_preserves_fp32_input_dtype(self) -> None:
        base = _make_base()
        adapter = _make_adapter(base=base)

        x = t.randn(2, IN_FEATURES, dtype=t.float32)
        with t.no_grad():
            out = adapter(x)

        assert out.dtype == t.float32


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScaling:
    def test_scaling_equals_alpha_over_rank(self) -> None:
        base = _make_base()
        adapter = LoRAAdapter(base=base, rank=8, alpha=16.0)
        assert adapter.scaling == pytest.approx(2.0)

    def test_scaling_fractional(self) -> None:
        base = _make_base()
        adapter = LoRAAdapter(base=base, rank=32, alpha=8.0)
        assert adapter.scaling == pytest.approx(0.25)

    def test_scaling_applied_mathematically(self) -> None:
        """Manually set ``lora_B`` to a known value and verify the output
        equals ``base(x) + scaling * (x @ lora_A.T @ lora_B.T)``.
        """
        base = _make_base()
        adapter = _make_adapter(base=base, rank=RANK, alpha=ALPHA)

        # Override lora_B with a known non-zero value.
        t.manual_seed(0xBEEF)
        with t.no_grad():
            adapter.lora_B.copy_(t.randn_like(adapter.lora_B))

        x = t.randn(3, IN_FEATURES)
        with t.no_grad():
            out = adapter(x)
            base_out = base(x)
            # Expected LoRA path: x @ lora_A.T -> (3, r), then @ lora_B.T -> (3, out).
            lora_hidden = x @ adapter.lora_A.t()
            lora_out = lora_hidden @ adapter.lora_B.t()
            expected = base_out + adapter.scaling * lora_out

        assert t.allclose(out, expected, atol=1e-5, rtol=1e-5), (
            "LoRA forward does not match base(x) + scaling * (x @ A.T @ B.T)."
        )


# ---------------------------------------------------------------------------
# Initialisation invariants
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_lora_B_init_zero(self) -> None:
        adapter = _make_adapter()
        assert t.equal(
            adapter.lora_B,
            t.zeros_like(adapter.lora_B),
        ), "lora_B must be bit-exact zero at init."

    def test_lora_A_init_nonzero_non_constant(self) -> None:
        """Kaiming-uniform init must yield a non-zero, non-constant tensor."""
        adapter = _make_adapter()
        assert t.any(adapter.lora_A != 0), (
            "lora_A must be non-zero (Kaiming-uniform)."
        )
        # Non-constant: at least two elements differ.
        first = adapter.lora_A.flatten()[0]
        assert not t.all(adapter.lora_A == first), (
            "lora_A must not be constant."
        )

    def test_lora_A_shape(self) -> None:
        adapter = _make_adapter()
        assert adapter.lora_A.shape == (RANK, IN_FEATURES)

    def test_lora_B_shape(self) -> None:
        adapter = _make_adapter()
        assert adapter.lora_B.shape == (OUT_FEATURES, RANK)


# ---------------------------------------------------------------------------
# Parameter freezing / trainable set
# ---------------------------------------------------------------------------


class TestParameterFreezing:
    def test_base_weights_frozen_after_init(self) -> None:
        adapter = _make_adapter()
        assert adapter.base.weight.requires_grad is False
        if adapter.base.bias is not None:
            assert adapter.base.bias.requires_grad is False

    def test_lora_parameters_are_trainable(self) -> None:
        adapter = _make_adapter()
        assert adapter.lora_A.requires_grad is True
        assert adapter.lora_B.requires_grad is True

    def test_trainable_parameters_are_only_lora_ab(self) -> None:
        """``[p for p in parameters() if p.requires_grad]`` must contain
        exactly lora_A and lora_B — nothing else.
        """
        adapter = _make_adapter()
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) == 2
        trainable_ids = {id(p) for p in trainable}
        assert id(adapter.lora_A) in trainable_ids
        assert id(adapter.lora_B) in trainable_ids


# ---------------------------------------------------------------------------
# Instance-level fields (Item 5: type annotations are NOT shared state)
# ---------------------------------------------------------------------------


class TestInstanceLevelFields:
    """Each LoRAAdapter instance must have its own base, lora_A, lora_B.

    The ``base: nn.Linear`` / ``lora_A: nn.Parameter`` / ``lora_B:
    nn.Parameter`` entries at the top of the ``LoRAAdapter`` class are
    TYPE ANNOTATIONS (PEP 526) — they do NOT create class-level
    attributes with values, so two instances never share storage.
    These tests pin that invariant so nobody "promotes" them to
    class-level defaults later and introduces cross-instance state
    leaks.
    """

    def test_lora_adapter_fields_are_instance_level_not_shared(self) -> None:
        base1 = _make_base()
        # Re-seed before building the second base so its weights differ
        # from the first (otherwise the two NanochatLinears would have
        # the same Kaiming-uniform init and the assert could be a
        # non-event; we want the bases to be clearly distinct objects).
        t.manual_seed(0xD101)
        base2 = NanochatLinear(
            IN_FEATURES, OUT_FEATURES, bias=True, dtype=t.float32,
        )

        instance1 = LoRAAdapter(base=base1, rank=RANK, alpha=ALPHA)
        instance2 = LoRAAdapter(base=base2, rank=RANK, alpha=ALPHA)

        # Different base modules entirely (distinct Python objects).
        assert instance1.base is not instance2.base

        # lora_A and lora_B are separate Parameter objects per instance.
        assert instance1.lora_A is not instance2.lora_A
        assert instance1.lora_B is not instance2.lora_B
        assert id(instance1.lora_A) != id(instance2.lora_A)
        assert id(instance1.lora_B) != id(instance2.lora_B)

        # Mutating one instance's lora_A does NOT affect the other.
        with t.no_grad():
            instance1.lora_A.fill_(42.0)
        # After the fill, instance1.lora_A is all 42s but instance2.lora_A
        # must still be its original (Kaiming-uniform, not 42s) values.
        assert t.all(instance1.lora_A == 42.0)
        assert not t.any(instance2.lora_A == 42.0), (
            "instance2.lora_A must be unaffected by mutation of "
            "instance1.lora_A — fields must be instance-level."
        )

        # Mutating one instance's lora_B does NOT affect the other.
        with t.no_grad():
            instance1.lora_B.fill_(7.0)
        assert t.all(instance1.lora_B == 7.0)
        assert t.all(instance2.lora_B == 0.0), (
            "instance2.lora_B must still be zero-initialised after "
            "mutating instance1.lora_B."
        )
