"""Micro-benchmarks for P1 inline steering eval perf.

Comparing scalar-loop vs. batched implementations for the three probe
helpers. Gated with @pytest.mark.slow so they don't run in the default
CI loop. Opt-in:
    uv run pytest tests/test_p1_benchmarks.py --benchmark-enable -m slow
"""

from __future__ import annotations

import pytest
import torch

from steering.eval.feature_probe import (
    apply_sae_activation,
    apply_sae_activation_batched,
    compute_raw_probe,
    compute_raw_probe_batched,
    max_pool_masked,
    max_pool_masked_batched,
)

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


class _MockSAEReLU:
    """Minimal SAE with ReLU activation (no threshold)."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"activation_fn": "relu"})()


@pytest.fixture
def probe_fixture():
    """Random probe tensors at realistic Gemma-2-2B + 16k SAE scale."""
    torch.manual_seed(7)
    d_model = 2304  # Gemma-2-2B residual width
    n_features = 16384
    seq_len = 200
    return {
        "act": torch.randn(seq_len, d_model),
        "W_enc": torch.randn(d_model, n_features),
        "b_enc": torch.randn(n_features),
        "b_dec": torch.randn(d_model),
        "mask": torch.rand(seq_len) > 0.5,
    }


# ---------------------------------------------------------------------------
# compute_raw_probe: scalar vs batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_compute_raw_probe_scalar(benchmark, probe_fixture, n_gt_features):
    """Scalar compute_raw_probe called in a Python loop over n_gt_features."""
    gt_fids = list(range(n_gt_features))

    def _run():
        results = []
        for fid in gt_fids:
            results.append(
                compute_raw_probe(
                    probe_fixture["act"],
                    probe_fixture["W_enc"],
                    probe_fixture["b_enc"],
                    probe_fixture["b_dec"],
                    fid,
                    apply_b_dec=True,
                )
            )
        return torch.stack(results, dim=1)

    benchmark(_run)


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_compute_raw_probe_batched(benchmark, probe_fixture, n_gt_features):
    """Batched compute_raw_probe_batched over n_gt_features at once."""
    gt_fids_t = torch.tensor(list(range(n_gt_features)), dtype=torch.int64)

    def _run():
        return compute_raw_probe_batched(
            probe_fixture["act"],
            probe_fixture["W_enc"],
            probe_fixture["b_enc"],
            probe_fixture["b_dec"],
            gt_fids_t,
            apply_b_dec=True,
        )

    benchmark(_run)


# ---------------------------------------------------------------------------
# apply_sae_activation: scalar vs batched (ReLU fallback path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_apply_sae_activation_scalar(benchmark, probe_fixture, n_gt_features):
    """Scalar apply_sae_activation called in a Python loop over n_gt_features."""
    sae = _MockSAEReLU()
    gt_fids = list(range(n_gt_features))
    # Precompute pre_probes so we're only measuring the ReLU gate
    pre_probes = compute_raw_probe_batched(
        probe_fixture["act"],
        probe_fixture["W_enc"],
        probe_fixture["b_enc"],
        probe_fixture["b_dec"],
        torch.tensor(gt_fids, dtype=torch.int64),
        apply_b_dec=True,
    )

    def _run():
        return [apply_sae_activation(pre_probes[:, i], sae, gt_fids[i]) for i in range(n_gt_features)]

    benchmark(_run)


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_apply_sae_activation_batched(benchmark, probe_fixture, n_gt_features):
    """Batched apply_sae_activation_batched over n_gt_features at once."""
    sae = _MockSAEReLU()
    gt_fids_t = torch.tensor(list(range(n_gt_features)), dtype=torch.int64)
    pre_probes = compute_raw_probe_batched(
        probe_fixture["act"],
        probe_fixture["W_enc"],
        probe_fixture["b_enc"],
        probe_fixture["b_dec"],
        gt_fids_t,
        apply_b_dec=True,
    )

    def _run():
        return apply_sae_activation_batched(pre_probes, sae, gt_fids_t)

    benchmark(_run)


# ---------------------------------------------------------------------------
# max_pool_masked: scalar vs batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_max_pool_masked_scalar(benchmark, probe_fixture, n_gt_features):
    """Scalar max_pool_masked called in a Python loop over n_gt_features."""
    gt_fids = list(range(n_gt_features))
    probes = torch.stack(
        [
            compute_raw_probe(
                probe_fixture["act"],
                probe_fixture["W_enc"],
                probe_fixture["b_enc"],
                probe_fixture["b_dec"],
                fid,
                apply_b_dec=True,
            )
            for fid in gt_fids
        ],
        dim=1,
    )
    mask = probe_fixture["mask"]

    def _run():
        return [max_pool_masked(probes[:, i], mask) for i in range(n_gt_features)]

    benchmark(_run)


@pytest.mark.parametrize("n_gt_features", [3, 5, 10, 20, 50])
def test_benchmark_max_pool_masked_batched(benchmark, probe_fixture, n_gt_features):
    """Batched max_pool_masked_batched over n_gt_features at once."""
    gt_fids_t = torch.tensor(list(range(n_gt_features)), dtype=torch.int64)
    probes = compute_raw_probe_batched(
        probe_fixture["act"],
        probe_fixture["W_enc"],
        probe_fixture["b_enc"],
        probe_fixture["b_dec"],
        gt_fids_t,
        apply_b_dec=True,
    )
    mask = probe_fixture["mask"]

    def _run():
        return max_pool_masked_batched(probes, mask)

    benchmark(_run)
