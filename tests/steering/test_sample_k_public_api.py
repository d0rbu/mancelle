"""Slice 07 — pin ``sample_k`` / ``sample_k_batched`` as the public API.

Manifest §7 slice 7: "Promote ``_sample_k`` to public API for slice 6 consumer".

Public names: ``sample_k`` and ``sample_k_batched``.
- The audit (per the user's "no deprecated aliases" directive) renames the
  formerly private ``_sample_k`` / ``_sample_k_batched`` to their public
  forms in-place. There is NO backwards-compatibility alias — every
  in-tree caller has been updated to import the public names.
- ``__all__`` exports the public names plus the existing public surface.

Seed source: deterministic — fixed seeds documented per test.
"""

from __future__ import annotations

import importlib

import pytest
import torch as t

from steering.conditional import sampling as sampling_mod
from steering.conditional.sampling import (
    Distribution,
    sample_k,
    sample_k_batched,
)


class TestSampleKPublic:
    """``sample_k`` is the canonical public name for k-sampling."""

    def test_sample_k_callable(self):
        rng = t.Generator().manual_seed(42)
        k = sample_k(
            Distribution.geometric, mean=5.0, std=5.0, rng=rng,
            min_k=0, max_k=10,
        )
        assert isinstance(k, int)
        assert 0 <= k < 10

    def test_sample_k_geometric_min_k_only(self):
        """Same seed + same params -> deterministic draws."""
        rng_a = t.Generator().manual_seed(0)
        rng_b = t.Generator().manual_seed(0)
        for _ in range(50):
            ka = sample_k(
                Distribution.geometric, mean=4.0, std=4.0, rng=rng_a,
                min_k=1, max_k=8,
            )
            kb = sample_k(
                Distribution.geometric, mean=4.0, std=4.0, rng=rng_b,
                min_k=1, max_k=8,
            )
            assert ka == kb
            assert 1 <= ka < 8


class TestSampleKBatchedPublic:
    """``sample_k_batched`` is the canonical batched public name."""

    def test_sample_k_batched_callable(self):
        rng = t.Generator().manual_seed(123)
        ks = sample_k_batched(
            Distribution.normal, mean=4.0, std=2.0, rng=rng, n=64,
            min_k=2, max_k=10,
        )
        assert ks.shape == (64,)
        assert ks.dtype == t.int64
        assert ((ks >= 2) & (ks < 10)).all()

    def test_sample_k_batched_deterministic(self):
        rng_a = t.Generator().manual_seed(7)
        rng_b = t.Generator().manual_seed(7)
        ka = sample_k_batched(
            Distribution.geometric, mean=3.0, std=3.0, rng=rng_a, n=128,
            min_k=0, max_k=12,
        )
        kb = sample_k_batched(
            Distribution.geometric, mean=3.0, std=3.0, rng=rng_b, n=128,
            min_k=0, max_k=12,
        )
        assert t.equal(ka, kb)


class TestModuleAllExports:
    """``__all__`` lists the public names."""

    def test_sample_k_in_all(self):
        assert "sample_k" in sampling_mod.__all__

    def test_sample_k_batched_in_all(self):
        assert "sample_k_batched" in sampling_mod.__all__


class TestNoUnderscoreAliases:
    """Per the user's "no deprecated aliases" directive (slice-07 audit),
    the underscore-prefixed forms must not exist."""

    def test_no_underscore_sample_k(self):
        assert not hasattr(sampling_mod, "_sample_k"), (
            "Slice-07 audit removed ``_sample_k`` — use ``sample_k``."
        )

    def test_no_underscore_sample_k_batched(self):
        assert not hasattr(sampling_mod, "_sample_k_batched"), (
            "Slice-07 audit removed ``_sample_k_batched`` — use ``sample_k_batched``."
        )

    def test_underscore_imports_fail(self):
        """The underscore-prefixed names are absent from the module.

        Uses ``importlib.import_module`` + ``getattr`` to probe at runtime
        without leaving a static reference that the type checker would flag.
        """
        mod = importlib.import_module("steering.conditional.sampling")
        for name in ("_sample_k", "_sample_k_batched"):
            with pytest.raises(AttributeError):
                getattr(mod, name)
