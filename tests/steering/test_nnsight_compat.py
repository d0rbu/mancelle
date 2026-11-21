"""Tests for steering.nnsight_compat.unpatch_nnsight.

Verifies that nnsight's global Tensor.backward patch is correctly
detected and removed, and that the original backward function is restored.

Tests are ordered: patched state → unpatch → verify → idempotent check.
"""

import subprocess
import sys

import torch as t


class TestUnpatchNnsight:
    @staticmethod
    def _backward_patch():
        import nnsight

        for patch in nnsight.DEFAULT_PATCHER.patches:
            if getattr(patch, "key", "") == "backward":
                return patch
        raise AssertionError("No backward patch found")

    def test_01_nnsight_patches_backward_on_import(self):
        """Importing nnsight replaces Tensor.backward with a wrapper.

        Runs in a fresh subprocess: nnsight patches torch globally on import,
        so if any earlier test already imported it, the in-process check
        becomes a no-op. A subprocess guarantees a clean baseline.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch as t;"
                    "orig_id = id(t.Tensor.backward);"
                    "import nnsight;"
                    "assert id(t.Tensor.backward) != orig_id, "
                    "'nnsight should have patched Tensor.backward'"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_02_backward_is_known_wrapper_or_original(self):
        """Current Tensor.backward is either nnsight's wrapper or its original."""
        patch = self._backward_patch()
        assert t.Tensor.backward in {patch.replacement, patch.orig}

    def test_03_unpatch_restores_original(self):
        """unpatch_nnsight restores the original Tensor.backward."""
        patch = self._backward_patch()
        t.Tensor.backward = patch.replacement

        from steering.nnsight_compat import unpatch_nnsight
        result = unpatch_nnsight()

        assert result is True
        assert t.Tensor.backward is patch.orig

    def test_04_unpatch_idempotent(self):
        """Second call returns False (already unpatched)."""
        from steering.nnsight_compat import unpatch_nnsight
        result = unpatch_nnsight()
        assert result is False

    def test_05_backward_still_works(self):
        """Backward pass works correctly after unpatching."""
        x = t.tensor([2.0, 3.0], requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert x.grad is not None
        assert t.allclose(x.grad, t.tensor([4.0, 6.0]))

    def test_06_no_nnsight_wrapper(self):
        """After unpatch, backward is the original, not wrap_backward."""
        assert "wrap_backward" not in str(t.Tensor.backward)
