"""Tests for conditioned-SFT gradient clipping.

Validates that PyTorch clipping and the shared SFT helper work correctly:
- Gradients are clipped to the expected max norm
- Edge cases: zero gradients, single parameters, extreme values
- The training loop clips accumulated raw-model gradients before optimizer step
"""

import inspect
import math
import re

import torch
import torch.nn as nn

from scripts.train import conditional_encdec_sft
from steering.sft import clipping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_with_grads(
    param_shapes: list[tuple[int, ...]], grad_scale: float = 1.0
) -> nn.Module:
    """Create a simple model with manually-set gradients.

    Each parameter gets a gradient filled with grad_scale, so the total
    gradient norm is grad_scale * sqrt(total_numel).
    """
    model = nn.Module()
    for i, shape in enumerate(param_shapes):
        p = nn.Parameter(torch.randn(shape))
        p.grad = torch.full(shape, grad_scale)
        model.register_parameter(f"p{i}", p)
    return model


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of all gradients in the model."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat([g.flatten() for g in grads]).norm().item()


def _require_grad(tensor: torch.Tensor) -> torch.Tensor:
    grad = tensor.grad
    assert grad is not None
    return grad


# ---------------------------------------------------------------------------
# Test: clip_grad_norm_ basic behavior
# ---------------------------------------------------------------------------


class TestClipGradNormBasic:
    """Verify torch.nn.utils.clip_grad_norm_ clips correctly with max_norm=1.0."""

    def test_large_grads_are_clipped_to_max_norm(self):
        """Gradients with norm >> 1.0 should be clipped to exactly 1.0."""
        model = _make_model_with_grads([(100,), (200,)], grad_scale=10.0)
        pre_norm = _compute_grad_norm(model)
        assert pre_norm > 1.0, f"Pre-clip norm should be > 1.0, got {pre_norm}"

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        post_norm = _compute_grad_norm(model)
        assert abs(post_norm - 1.0) < 1e-5, (
            f"Post-clip norm should be 1.0, got {post_norm}"
        )

    def test_returned_norm_is_pre_clip_norm(self):
        """clip_grad_norm_ returns the total norm BEFORE clipping."""
        model = _make_model_with_grads([(100,)], grad_scale=5.0)
        expected_pre_norm = 5.0 * math.sqrt(100)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert abs(total_norm.item() - expected_pre_norm) < 1e-3, (
            f"Returned norm should be pre-clip norm {expected_pre_norm}, got {total_norm.item()}"
        )

    def test_small_grads_are_not_modified(self):
        """Gradients with norm < 1.0 should not be changed."""
        model = _make_model_with_grads([(10,)], grad_scale=0.01)
        pre_grads = {n: _require_grad(p).clone() for n, p in model.named_parameters()}

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for name, param in model.named_parameters():
            assert torch.allclose(_require_grad(param), pre_grads[name]), (
                f"Gradient for {name} should not change when norm < max_norm"
            )

    def test_exactly_at_max_norm_is_unchanged(self):
        """Gradients with norm == max_norm should not be clipped."""
        # Create a single parameter where grad norm is exactly 1.0
        model = nn.Module()
        p = nn.Parameter(torch.randn(1))
        p.grad = torch.tensor([1.0])
        model.register_parameter("p0", p)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert abs(p.grad.item() - 1.0) < 1e-6
        assert abs(total_norm.item() - 1.0) < 1e-6

    def test_gradient_direction_preserved_after_clipping(self):
        """Clipping should only scale magnitude, not change direction."""
        model = nn.Module()
        p = nn.Parameter(torch.randn(4))
        original_grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm = 5.0
        p.grad = original_grad.clone()
        model.register_parameter("p0", p)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Direction should be same: [3/5, 4/5, 0, 0] scaled to norm 1
        expected_dir = original_grad / original_grad.norm()
        actual_dir = p.grad / p.grad.norm()
        assert torch.allclose(actual_dir, expected_dir, atol=1e-6), (
            "Gradient direction should be preserved after clipping"
        )

    def test_multiple_parameters_clipped_jointly(self):
        """All parameters should be scaled by the same factor (joint norm)."""
        model = nn.Module()
        p1 = nn.Parameter(torch.randn(3))
        p1.grad = torch.tensor([3.0, 0.0, 0.0])  # contributes 9 to squared norm
        p2 = nn.Parameter(torch.randn(3))
        p2.grad = torch.tensor([0.0, 4.0, 0.0])  # contributes 16 to squared norm
        model.register_parameter("p1", p1)
        model.register_parameter("p2", p2)
        # Joint norm = sqrt(9 + 16) = 5.0

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert abs(total_norm.item() - 5.0) < 1e-5
        # Both should be scaled by 1/5
        assert abs(p1.grad[0].item() - 3.0 / 5.0) < 1e-6
        assert abs(p2.grad[1].item() - 4.0 / 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Test: logging threshold (total_norm > 5.0)
# ---------------------------------------------------------------------------


class TestGradientClippingLogging:
    """Test the logging threshold: total_norm > 5.0 triggers error log."""

    def test_norm_above_5_triggers_log(self):
        """When pre-clip norm > 5.0, the condition should trigger."""
        model = _make_model_with_grads([(100,)], grad_scale=1.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # norm = 1.0 * sqrt(100) = 10.0, which is > 5.0
        assert total_norm > 5.0, f"Expected norm > 5.0, got {total_norm}"

    def test_norm_below_5_does_not_trigger_log(self):
        """When pre-clip norm <= 5.0, the condition should not trigger."""
        model = _make_model_with_grads([(4,)], grad_scale=1.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # norm = 1.0 * sqrt(4) = 2.0, which is < 5.0
        assert total_norm <= 5.0, f"Expected norm <= 5.0, got {total_norm}"

    def test_norm_exactly_5_does_not_trigger_log(self):
        """When pre-clip norm == 5.0, the condition (> 5.0) should NOT trigger."""
        model = nn.Module()
        p = nn.Parameter(torch.randn(1))
        p.grad = torch.tensor([5.0])
        model.register_parameter("p0", p)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Strict > 5.0, not >= 5.0
        should_log = total_norm > 5.0
        assert not should_log, "Norm exactly 5.0 should NOT trigger the log (strict >)"

    def test_norm_slightly_above_5_triggers_log(self):
        """Norm 5.001 should trigger the log."""
        model = nn.Module()
        p = nn.Parameter(torch.randn(1))
        p.grad = torch.tensor([5.001])
        model.register_parameter("p0", p)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        should_log = total_norm > 5.0
        assert should_log, "Norm 5.001 should trigger the log"


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestGradientClippingEdgeCases:
    """Edge cases for gradient clipping."""

    def test_zero_gradients(self):
        """Zero gradients should remain zero after clipping."""
        model = _make_model_with_grads([(10,), (20,)], grad_scale=0.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert total_norm.item() == 0.0
        for p in model.parameters():
            assert torch.all(_require_grad(p) == 0), "Zero gradients should remain zero"

    def test_single_parameter(self):
        """Clipping should work with a single parameter."""
        model = _make_model_with_grads([(1,)], grad_scale=100.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert abs(total_norm.item() - 100.0) < 1e-3
        post_norm = _compute_grad_norm(model)
        assert abs(post_norm - 1.0) < 1e-5

    def test_many_small_parameters(self):
        """Many small parameters with collectively large norm get clipped."""
        # 1000 params of size 1, each with grad=1.0 → total norm = sqrt(1000) ≈ 31.6
        shapes: list[tuple[int, ...]] = [(1,)] * 1000
        model = _make_model_with_grads(shapes, grad_scale=1.0)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert abs(total_norm.item() - math.sqrt(1000)) < 0.1
        post_norm = _compute_grad_norm(model)
        assert abs(post_norm - 1.0) < 1e-4

    def test_very_large_gradients(self):
        """Extremely large gradients should be clipped without overflow."""
        model = _make_model_with_grads([(10,)], grad_scale=1e6)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        assert math.isfinite(total_norm.item())
        post_norm = _compute_grad_norm(model)
        assert abs(post_norm - 1.0) < 1e-4

    def test_mixed_grad_and_no_grad_params(self):
        """Parameters without gradients should be ignored."""
        model = nn.Module()
        p1 = nn.Parameter(torch.randn(10))
        p1.grad = torch.full((10,), 5.0)  # norm = 5*sqrt(10) ≈ 15.8
        p2 = nn.Parameter(torch.randn(10))
        p2.grad = None  # no gradient
        model.register_parameter("p1", p1)
        model.register_parameter("p2", p2)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        expected = 5.0 * math.sqrt(10)
        assert abs(total_norm.item() - expected) < 0.1
        assert p2.grad is None, "No-grad param should remain None"


# ---------------------------------------------------------------------------
# Test: interaction with optimizer step
# ---------------------------------------------------------------------------


class TestClipGradNormWithOptimizer:
    """Verify clipping integrates correctly with optimizer.step()."""

    def test_clip_then_step_uses_clipped_gradients(self):
        """optimizer.step() should use the clipped gradients, not original."""
        model = nn.Linear(10, 10, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        # Forward/backward to get gradients
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()

        # Record pre-clip gradient norm
        pre_norm = _compute_grad_norm(model)

        if pre_norm > 1.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            post_norm = _compute_grad_norm(model)
            assert abs(post_norm - 1.0) < 1e-5

        # Step should use clipped gradients
        w_before = model.weight.data.clone()
        optimizer.step()
        w_after = model.weight.data.clone()

        # The update should be: w_after = w_before - lr * clipped_grad
        actual_update = w_before - w_after
        update_norm = actual_update.norm().item()

        if pre_norm > 1.0:
            # With lr=1.0 and clipped norm=1.0, update norm should be ~1.0
            assert abs(update_norm - 1.0) < 1e-4, (
                f"Update norm should be ~1.0 (clipped), got {update_norm}"
            )

    def test_clip_before_step_prevents_large_update(self):
        """Without clipping, a large gradient causes a proportionally large update.
        With clipping, the update is bounded."""
        model = nn.Linear(5, 5, bias=False)

        # Create artificially large gradients
        x = torch.randn(2, 5) * 100
        loss = model(x).sum()
        loss.backward()

        pre_norm = _compute_grad_norm(model)
        # Scale up gradients further to ensure they're large
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= 100.0
        pre_norm = _compute_grad_norm(model)
        assert pre_norm > 1.0, f"Gradients should be large, got norm {pre_norm}"

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        post_norm = _compute_grad_norm(model)
        assert abs(post_norm - 1.0) < 1e-5, (
            f"Post-clip norm should be 1.0, got {post_norm}"
        )

    def test_clipping_is_idempotent(self):
        """Clipping twice should give same result as clipping once."""
        model = _make_model_with_grads([(50,), (50,)], grad_scale=10.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grads_after_first = {
            n: _require_grad(p).clone() for n, p in model.named_parameters()
        }

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for name, param in model.named_parameters():
            assert torch.allclose(
                _require_grad(param), grads_after_first[name], atol=1e-6
            ), (
                f"Second clip changed gradient for {name}"
            )


# ---------------------------------------------------------------------------
# Test: gradient accumulation interaction
# ---------------------------------------------------------------------------


class TestClipWithGradAccumulation:
    """Verify clipping works correctly with gradient accumulation
    (loss.backward() called multiple times before clip + step)."""

    def test_accumulated_grads_are_clipped_jointly(self):
        """After multiple backward() calls, clip should act on accumulated norm."""
        model = nn.Linear(10, 1, bias=False)
        torch.optim.SGD(model.parameters(), lr=1.0)
        model.zero_grad()

        # Simulate 4 micro-steps of gradient accumulation
        grad_accum_steps = 4
        for _ in range(grad_accum_steps):
            x = torch.randn(1, 10) * 10
            loss = model(x).sum() / grad_accum_steps
            loss.backward()

        # Now clip the accumulated gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # The accumulated gradient should be clipped
        post_norm = _compute_grad_norm(model)
        if total_norm > 1.0:
            assert abs(post_norm - 1.0) < 1e-5, (
                f"Accumulated gradients should be clipped to 1.0, got {post_norm}"
            )

    def test_clip_after_accumulation_not_per_step(self):
        """Clipping should happen once after all accumulation, not per micro-step.

        This mirrors the SFT training loop: backward() in the micro-step loop,
        then clip_grad_norm_() after the loop, then optimizer.step().
        """
        model = nn.Linear(10, 1, bias=False)

        # Approach 1: clip once after accumulation (correct, matches SFT loop)
        model.zero_grad()
        for _ in range(4):
            x = torch.randn(1, 10)
            loss = model(x).sum() / 4
            loss.backward()
        norm_after_accum = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        grads_correct = _require_grad(model.weight).clone()

        # Approach 2: clip per micro-step (wrong — would under-clip)
        model.zero_grad()
        torch.manual_seed(0)
        for _ in range(4):
            x = torch.randn(1, 10)
            loss = model(x).sum() / 4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # The two approaches should generally give different results
        # (per-step clipping destroys the direction of accumulation)
        # We just verify the correct approach clips to norm=1.0
        post_norm = grads_correct.norm().item()
        if norm_after_accum > 1.0:
            assert abs(post_norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Test: SFT training loop structure
# ---------------------------------------------------------------------------


class TestSFTGradClipPosition:
    """Verify the shared clipping call's training-loop position and inputs."""

    def test_clip_grad_norm_in_source(self):
        assert "clip_grad_norm_" in inspect.getsource(clipping.apply_gradient_clipping)
        assert "apply_gradient_clipping" in inspect.getsource(conditional_encdec_sft.main)

    def test_max_norm_reads_from_config(self):
        """max_norm must be read from ``cfg.grad_clip_max_norm`` rather
        than hardcoded.  The default is ``null`` in
        ``configs/training/conditional_encdec_sft.yaml``; explicit
        ``1.0`` remains the historical opt-in — see
        ``tests/test_encdec_sft_grad_clip.py`` for the config-surface
        and backward-compat tests.

        This test was originally ``test_max_norm_is_1`` which asserted
        the literal ``"max_norm=1.0"`` appeared in the source.  After
        PR 6 made the value configurable, the literal no longer
        appears in ``main()`` — the config reference
        ``cfg.grad_clip_max_norm`` does instead.
        """
        main_source = inspect.getsource(conditional_encdec_sft.main)

        # The literal must no longer be hardcoded
        assert "max_norm=1.0" not in main_source, (
            "max_norm should not be hardcoded to 1.0; read from "
            "cfg.grad_clip_max_norm instead (the default lives in "
            "the YAML config)"
        )
        # And the config reference must be present in its place
        assert "grad_clip_max_norm" in main_source, (
            "main() must reference cfg.grad_clip_max_norm"
        )

    def test_clip_happens_after_backward_before_step(self):
        """The shared clip should appear after backward and before optimizer step."""
        source = inspect.getsource(conditional_encdec_sft.main)

        backward_pos = source.find("loss.backward()")
        clip_pos = source.find("apply_gradient_clipping(")
        step_pos = source.find("optimizer.step()")

        assert backward_pos != -1, "loss.backward() not found"
        assert clip_pos != -1, "apply_gradient_clipping not found"
        assert step_pos != -1, "optimizer.step() not found"

        assert backward_pos < clip_pos < step_pos, (
            f"Order should be backward ({backward_pos}) < clip ({clip_pos}) < step ({step_pos}), "
            "but the positions are wrong"
        )

    def test_clip_uses_orig_model_parameters(self):
        """The shared helper should receive raw, uncompiled model parameters.

        ``model`` may be a compiled wrapper; ``orig_model`` owns the same
        accumulated gradients and is the stable parameter traversal.

        Whitespace-normalised so the source can format the call across
        multiple lines without breaking the regression check.
        """
        source = inspect.getsource(conditional_encdec_sft.main)
        normalised = re.sub(r"\s+", " ", source)

        assert "apply_gradient_clipping( orig_model.parameters()," in normalised, (
            "apply_gradient_clipping should use orig_model.parameters(), "
            "not model.parameters()"
        )

    def test_pre_clip_norm_is_recorded_when_enabled(self):
        source = inspect.getsource(conditional_encdec_sft.main)
        assert 'wandb_payload["train/pre_clip_grad_norm"]' in source


# ---------------------------------------------------------------------------
# Test: flash-attn-4 is now a required dependency
# ---------------------------------------------------------------------------
