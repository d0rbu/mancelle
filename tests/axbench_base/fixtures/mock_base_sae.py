"""Deterministic mock base SAE.

Produces per-token activations hashed from ``(batch_index, token_index)``
so repeated calls with the same residual shape yield identical output —
tests can use the mock in place of a real SAE (e.g.
``sae_lens.SAE``) whenever they need per-token sparsity without pulling
in a multi-GB checkpoint.

The mock is deliberately minimal: only the attributes we have seen called
from the axbench_base pipeline are implemented (``encode``, ``cfg``,
``d_sae``). Add more here if a new caller needs them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch as t


@dataclass(frozen=True)
class _SAECfg:
    """Minimal stand-in for :class:`sae_lens.SAEConfig`.

    Only the attributes the axbench_base pipeline actually reads are
    populated. Fields mirror the subset of the real SAEConfig that flows
    through :class:`steering.eval.axbench_base.model_detection.detect_target_model_metadata`.
    """
    d_sae: int
    hook_name: str
    release: str = "mock_release"
    n_ctx: int = 2048


@dataclass
class MockBaseSAE:
    """Deterministic mock SAE over residual-stream activations.

    ``encode`` yields a sparse-ish ``(B, T, F)`` float tensor — each
    ``(batch_idx, token_idx)`` pair deterministically activates 5 randomly
    chosen features with magnitudes drawn from a seeded generator. Every
    other entry is zero.

    Attributes:
        feature_dim: Number of SAE features (``F`` in the output shape).
        seed: Base seed combined with ``(batch_idx, token_idx)`` to drive
            the per-token Generator. Different instances with different
            seeds produce different activation patterns for the same
            input.
    """

    feature_dim: int = 16384
    seed: int = 42

    def encode(self, residual: t.Tensor) -> t.Tensor:
        """Return deterministic SAE activations for ``residual``.

        Args:
            residual: ``(B, T, C)`` tensor. Only the shape is inspected —
                values are intentionally ignored so tests with arbitrary
                inputs still get stable outputs.

        Returns:
            ``(B, T, feature_dim)`` tensor with exactly 5 non-zero entries
            per ``(b, t)`` slice. Magnitudes are in ``[0, 5)``.
        """
        if residual.ndim != 3:
            raise ValueError(
                f"MockBaseSAE.encode expects a (B, T, C) tensor, got "
                f"shape {tuple(residual.shape)}"
            )
        batch_size, seq_len, _ = residual.shape
        out = t.zeros(batch_size, seq_len, self.feature_dim)
        gen = t.Generator()
        for b in range(batch_size):
            for ti in range(seq_len):
                gen.manual_seed(self.seed + b * 1000 + ti)
                active = t.randint(
                    0, self.feature_dim, (5,), generator=gen,
                ).tolist()
                mags = t.rand(5, generator=gen) * 5.0
                for fid, mag in zip(active, mags, strict=True):
                    out[b, ti, fid] = mag
        return out

    def to(self, device: object) -> MockBaseSAE:
        """Device-move stub for ``RuntimeState`` CPU-offload contract.

        ``RuntimeState.base_sae()`` calls ``.to(self._device)`` on enter
        and ``.to("cpu")`` on exit. The mock ignores the device since its
        tensors are always on CPU, but it must implement the method so
        the refcount machinery doesn't raise ``AttributeError``.
        """
        return self

    def parameters(self) -> Iterator[t.Tensor]:
        """Yield a single dummy parameter tensor.

        ``steering.extract_features._get_sae_activations_batched``
        calls ``next(sae.parameters()).device`` to discover the SAE's
        compute device. The real ``sae_lens.SAE`` inherits from
        ``nn.Module`` and has this method natively; the mock must
        provide it too.
        """
        yield t.zeros(1)

    @property
    def cfg(self) -> _SAECfg:
        """sae_lens-compatible config object."""
        return _SAECfg(
            d_sae=self.feature_dim,
            hook_name="blocks.0.hook_resid_post",
        )

    @property
    def d_sae(self) -> int:
        """Alias for ``feature_dim``; matches sae_lens naming."""
        return self.feature_dim
