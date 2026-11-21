"""Exactness and production-shape tests for sparse masked batch negatives."""

from __future__ import annotations

from statistics import median
from time import perf_counter

import pytest
import torch as t

import steering.conditional.sampling as sampling
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    sample_inactive_features_batched,
    sample_k_batched,
)

N_SAE = 65_536


def _dense_reference(
    monkeypatch: pytest.MonkeyPatch,
    active: list[t.Tensor],
    config: NegativeSamplingSelection,
    rng: t.Generator,
    mask: FeatureMask,
) -> list[t.Tensor]:
    """Force the retained dense implementation without changing its logic."""
    with monkeypatch.context() as context:
        context.setattr(sampling, "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT", -1)
        return sample_inactive_features_batched(
            active,
            N_SAE,
            config,
            rng,
            feature_mask=mask,
        )


def _active_rows(
    *,
    n_rows: int,
    max_active: int,
    seed: int,
) -> list[t.Tensor]:
    rng = t.Generator().manual_seed(seed)
    rows = []
    for row in range(n_rows):
        size = (row * 37 + 11) % (max_active + 1)
        values = t.randint(0, N_SAE, (size,), generator=rng)
        if row % 4 == 0 and values.numel() > 0:
            values = t.cat((values, values[: min(3, values.numel())]))
        rows.append(values)
    return rows


def _configs() -> tuple[NegativeSamplingSelection, ...]:
    return (
        NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=12.0,
            std=12.0,
            min_k=0,
            max_k=49,
        ),
        NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=20.0,
            std=7.0,
            min_k=0,
            max_k=49,
        ),
    )


def _masks() -> tuple[FeatureMask, ...]:
    return (
        FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE, 4))),
        FeatureMask(FeatureMaskMode.blacklist, tuple(range(0, N_SAE, 7))),
    )


@pytest.mark.parametrize("n_rows", [1, 2, 7, 64])
@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("mask", _masks())
def test_sparse_multisegment_is_exact_dense_output_and_rng(
    monkeypatch: pytest.MonkeyPatch,
    n_rows: int,
    config: NegativeSamplingSelection,
    mask: FeatureMask,
) -> None:
    """Differential gate over production width, masks, distributions, and rows."""
    active = _active_rows(n_rows=n_rows, max_active=1024, seed=19_003 + n_rows)
    for seed in range(8):
        sparse_rng = t.Generator().manual_seed(seed)
        dense_rng = t.Generator().manual_seed(seed)
        sparse = sample_inactive_features_batched(
            active,
            N_SAE,
            config,
            sparse_rng,
            feature_mask=mask,
        )
        dense = _dense_reference(
            monkeypatch,
            active,
            config,
            dense_rng,
            mask,
        )
        assert len(sparse) == len(dense) == n_rows
        for row_idx, (actual, expected) in enumerate(zip(sparse, dense, strict=True)):
            assert t.equal(actual, expected), (
                mask.mode,
                config.distribution,
                n_rows,
                row_idx,
                seed,
            )
        assert t.equal(sparse_rng.get_state(), dense_rng.get_state()), (
            mask.mode,
            config.distribution,
            n_rows,
            seed,
        )


def test_negative_rows_are_exact_serial_when_cardinality_rng_is_factored_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate allowed output drift to cardinality-draw regrouping.

    With the k draws replaced by a fixed vector, one matrix-shaped candidate
    draw and consecutive one-row draws consume the same CPU RNG stream. Exact
    row outputs and final state then prove that batching did not alter active
    exclusion, mask eligibility, candidate order, deduplication, or selection.
    """

    def fixed_k(
        distribution,
        mean,
        std,
        rng,
        n,
        min_k=0,
        max_k=None,
    ):
        del distribution, mean, std, rng, min_k, max_k
        return t.full((n,), 4, dtype=t.int64)

    monkeypatch.setattr(sampling, "sample_k_batched", fixed_k)
    active = _active_rows(n_rows=64, max_active=128, seed=7_177)
    mask = FeatureMask(FeatureMaskMode.blacklist, (60_000,))
    config = _configs()[0]
    for seed in range(8):
        serial_rng = t.Generator().manual_seed(seed)
        batched_rng = t.Generator().manual_seed(seed)
        expected = [
            sample_inactive_features_batched(
                [row],
                N_SAE,
                config,
                serial_rng,
                feature_mask=mask,
            )[0]
            for row in active
        ]
        actual = sample_inactive_features_batched(
            active,
            N_SAE,
            config,
            batched_rng,
            feature_mask=mask,
        )
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert t.equal(actual_row, expected_row)
        assert t.equal(batched_rng.get_state(), serial_rng.get_state())


def test_sparse_batch_does_not_allocate_segments_by_sae_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _active_rows(n_rows=128, max_active=512, seed=71)
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE, 5)))
    config = _configs()[0]
    original_zeros = sampling.t.zeros

    def guarded_zeros(*args, **kwargs):
        shape = (
            tuple(args[:2])
            if len(args) >= 2 and all(isinstance(value, int) for value in args[:2])
            else (args[0] if args else kwargs.get("size"))
        )
        if shape == (len(active), N_SAE):
            raise AssertionError("dense [segments, SAE width] allocation reached")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(sampling.t, "zeros", guarded_zeros)
    rows = sample_inactive_features_batched(
        active,
        N_SAE,
        config,
        t.Generator().manual_seed(8),
        feature_mask=mask,
    )
    assert len(rows) == len(active)


def test_outside_sparse_domain_uses_established_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = [t.arange(4097, dtype=t.int64), t.tensor([1, 2, 3])]
    mask = FeatureMask(FeatureMaskMode.blacklist, (60_000,))
    config = _configs()[0]
    original_zeros = sampling.t.zeros
    observed_dense_shape = False

    def observed_zeros(*args, **kwargs):
        nonlocal observed_dense_shape
        shape = (
            tuple(args[:2])
            if len(args) >= 2 and all(isinstance(value, int) for value in args[:2])
            else (args[0] if args else kwargs.get("size"))
        )
        if shape == (len(active), N_SAE):
            observed_dense_shape = True
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(sampling.t, "zeros", observed_zeros)
    rows = sample_inactive_features_batched(
        active,
        N_SAE,
        config,
        t.Generator().manual_seed(9),
        feature_mask=mask,
    )
    assert observed_dense_shape is True
    assert len(rows) == 2


@pytest.mark.parametrize(
    "bad_active",
    [
        t.tensor([True, False]),
        t.tensor([1.0, 2.0]),
        t.tensor([-1, 2]),
        t.tensor([1, N_SAE]),
        t.ones((2, 2), dtype=t.int64),
    ],
)
def test_unsupported_active_shapes_preserve_dense_behavior(
    monkeypatch: pytest.MonkeyPatch,
    bad_active: t.Tensor,
) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (60_000,))
    config = _configs()[0]
    sparse_rng = t.Generator().manual_seed(88)
    dense_rng = t.Generator().manual_seed(88)
    try:
        sparse = sample_inactive_features_batched(
            [bad_active],
            N_SAE,
            config,
            sparse_rng,
            feature_mask=mask,
        )
    except Exception as sparse_error:  # noqa: BLE001 - exact fallback surface
        with pytest.raises(type(sparse_error)) as dense_exc:
            _dense_reference(
                monkeypatch,
                [bad_active],
                config,
                dense_rng,
                mask,
            )
        assert str(dense_exc.value) == str(sparse_error)
    else:
        dense = _dense_reference(
            monkeypatch,
            [bad_active],
            config,
            dense_rng,
            mask,
        )
        assert len(sparse) == len(dense) == 1
        assert t.equal(sparse[0], dense[0])
        assert t.equal(sparse_rng.get_state(), dense_rng.get_state())


def test_pool_size_clamping_matches_exact_available_whitelist() -> None:
    allowed = (2, 4, 6, 8, 10)
    mask = FeatureMask(FeatureMaskMode.whitelist, allowed)
    config = NegativeSamplingSelection(
        distribution=Distribution.normal,
        mean=100.0,
        std=0.01,
        min_k=0,
        max_k=101,
    )
    active = [
        t.tensor([], dtype=t.int64),
        t.tensor([2]),
        t.tensor([2, 4, 6, 8]),
        t.tensor([2, 4, 6, 8, 10]),
    ]
    rows = sample_inactive_features_batched(
        active,
        N_SAE,
        config,
        t.Generator().manual_seed(771),
        feature_mask=mask,
    )
    assert [row.numel() for row in rows] == [5, 4, 1, 0]
    for active_row, sampled in zip(active, rows, strict=True):
        assert set(sampled.tolist()) == set(allowed) - set(active_row.tolist())


def test_sampled_cardinalities_equal_the_first_batched_k_draw() -> None:
    n_rows = 512
    config = _configs()[0]
    mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(0, N_SAE, 101)))
    active = _active_rows(n_rows=n_rows, max_active=64, seed=19)
    sampler_rng = t.Generator().manual_seed(31_337)
    oracle_rng = t.Generator().manual_seed(31_337)
    expected = sample_k_batched(
        config.distribution,
        config.mean,
        config.std,
        oracle_rng,
        n_rows,
        min_k=config.min_k,
        max_k=config.max_k,
    )
    rows = sample_inactive_features_batched(
        active,
        N_SAE,
        config,
        sampler_rng,
        feature_mask=mask,
    )
    actual = t.tensor([row.numel() for row in rows], dtype=t.int64)
    assert t.equal(actual, expected)


def test_every_row_is_unique_allowed_and_inactive_at_production_width() -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE, 3)))
    config = _configs()[1]
    active = _active_rows(n_rows=256, max_active=2048, seed=991)
    rows = sample_inactive_features_batched(
        active,
        N_SAE,
        config,
        t.Generator().manual_seed(992),
        feature_mask=mask,
    )
    allowed = mask.allowed_universe_mask(N_SAE)
    for active_row, sampled in zip(active, rows, strict=True):
        assert sampled.dtype == t.int64
        assert sampled.ndim == 1
        assert sampled.unique().numel() == sampled.numel()
        assert set(sampled.tolist()).isdisjoint(active_row.tolist())
        assert bool(allowed[sampled].all()) if sampled.numel() else True


@pytest.mark.benchmark
@pytest.mark.slow
def test_production_shape_batched_masked_sampling_benchmark() -> None:
    """Non-headline guard: batching must beat 256 one-row dispatches.

    This is excluded from the default suite. It measures only this sampler on
    CPU and cannot establish an end-to-end producer speedup.
    """
    active = _active_rows(n_rows=256, max_active=512, seed=818)
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE, 37)))
    config = _configs()[0]

    def serial() -> None:
        rng = t.Generator().manual_seed(919)
        for row in active:
            sample_inactive_features_batched(
                [row],
                N_SAE,
                config,
                rng,
                feature_mask=mask,
            )

    def batched() -> None:
        sample_inactive_features_batched(
            active,
            N_SAE,
            config,
            t.Generator().manual_seed(919),
            feature_mask=mask,
        )

    serial()
    batched()
    serial_times = []
    batched_times = []
    for _ in range(9):
        start = perf_counter()
        serial()
        serial_times.append(perf_counter() - start)
        start = perf_counter()
        batched()
        batched_times.append(perf_counter() - start)
    assert median(batched_times) < median(serial_times) * 0.9
