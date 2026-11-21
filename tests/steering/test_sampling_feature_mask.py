# tests/steering/test_sampling_feature_mask.py
from __future__ import annotations

import pytest
import torch as t

import steering.conditional.sampling as sampling
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    sample_inactive_features,
    sample_inactive_features_batched,
)

CFG = NegativeSamplingSelection(mean=5.0, std=5.0, min_k=5, max_k=6)


def _dense_masked_batched_reference(
    monkeypatch: pytest.MonkeyPatch,
    active: t.Tensor,
    n_sae: int,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    mask: FeatureMask,
) -> list[t.Tensor]:
    """Execute the retained dense path by disabling only the sparse domain."""
    with monkeypatch.context() as context:
        context.setattr(sampling, "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT", -1)
        return sample_inactive_features_batched(
            [active],
            n_sae,
            config,
            rng,
            feature_mask=mask,
        )


def test_single_blacklist_excluded():
    rng = t.Generator().manual_seed(0)
    m = FeatureMask("blacklist", tuple(range(50)))  # only 50..99 allowed
    out = sample_inactive_features(t.tensor([], dtype=t.int64), 100, CFG, rng, feature_mask=m)
    assert out.numel() > 0 and int(out.min()) >= 50


def test_single_whitelist_universe():
    rng = t.Generator().manual_seed(0)
    m = FeatureMask("whitelist", (10, 20, 30, 40, 50))
    out = sample_inactive_features(t.tensor([20], dtype=t.int64), 100, CFG, rng, feature_mask=m)
    assert set(out.tolist()).issubset({10, 30, 40, 50})  # whitelist minus active(20)


def test_batched_blacklist_excluded():
    rng = t.Generator().manual_seed(0)
    m = FeatureMask("blacklist", (7,))
    res = sample_inactive_features_batched(
        [t.tensor([], dtype=t.int64)], 50, CFG, rng, feature_mask=m
    )
    assert 7 not in res[0].tolist()


def test_single_segment_sparse_masked_path_is_exact_dense_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-width values and final RNG state match the dense oracle.

    This covers both mask modes, both k distributions, duplicate/empty active
    IDs, and active-set sizes spanning the observed splice-selection regime.
    Comparing generator state catches an implementation that happens to return
    the same IDs while consuming a different number of draws.
    """
    n_sae = 65_536
    masks = (
        FeatureMask("whitelist", tuple(range(0, n_sae, 4))),
        FeatureMask("blacklist", tuple(range(0, n_sae, 7))),
    )
    configs = (
        NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=12.0,
            std=12.0,
            min_k=1,
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
    setup_rng = t.Generator().manual_seed(91_003)
    active_cases = [
        t.tensor([], dtype=t.int64),
        t.tensor([0], dtype=t.int64),
        t.tensor([7, 7, 7, 12], dtype=t.int64),
        t.randint(0, n_sae, (64,), generator=setup_rng),
        t.randint(0, n_sae, (256,), generator=setup_rng),
        t.randint(0, n_sae, (1024,), generator=setup_rng),
    ]

    for mask in masks:
        for config in configs:
            for active in active_cases:
                for seed in range(12):
                    sparse_rng = t.Generator().manual_seed(seed)
                    dense_rng = t.Generator().manual_seed(seed)
                    sparse = sample_inactive_features_batched(
                        [active],
                        n_sae,
                        config,
                        sparse_rng,
                        feature_mask=mask,
                    )
                    dense = _dense_masked_batched_reference(
                        monkeypatch,
                        active,
                        n_sae,
                        config,
                        dense_rng,
                        mask,
                    )
                    assert len(sparse) == len(dense) == 1
                    assert t.equal(sparse[0], dense[0]), (
                        mask.mode,
                        config.distribution,
                        active.numel(),
                        seed,
                    )
                    assert t.equal(sparse_rng.get_state(), dense_rng.get_state()), (
                        mask.mode,
                        config.distribution,
                        active.numel(),
                        seed,
                    )


def test_direct_sparse_masked_sampler_is_exact_dense_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_sae = 65_536
    config = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=12.0,
        std=12.0,
        min_k=1,
        max_k=49,
    )
    setup_rng = t.Generator().manual_seed(19_771)
    for mask in (
        FeatureMask("whitelist", tuple(range(0, n_sae, 4))),
        FeatureMask("blacklist", tuple(range(0, n_sae, 7))),
    ):
        for active_size in (0, 1, 64, 256, 1024):
            active = t.randint(0, n_sae, (active_size,), generator=setup_rng)
            for seed in range(12):
                sparse_rng = t.Generator().manual_seed(seed)
                dense_rng = t.Generator().manual_seed(seed)
                sparse = sample_inactive_features(
                    active,
                    n_sae,
                    config,
                    sparse_rng,
                    feature_mask=mask,
                )
                with monkeypatch.context() as context:
                    context.setattr(
                        sampling,
                        "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT",
                        -1,
                    )
                    dense = sample_inactive_features(
                        active,
                        n_sae,
                        config,
                        dense_rng,
                        feature_mask=mask,
                    )
                assert t.equal(sparse, dense)
                assert t.equal(sparse_rng.get_state(), dense_rng.get_state())


@pytest.mark.parametrize(
    "active",
    [
        t.tensor([1.25, 7.75], dtype=t.float32),
        t.tensor([True, False, True], dtype=t.bool),
        t.arange(4097, dtype=t.int64),
        t.tensor([-1, 3, 9], dtype=t.int64),
    ],
)
def test_sparse_domain_edges_fall_back_to_exact_dense_behavior(
    monkeypatch: pytest.MonkeyPatch,
    active: t.Tensor,
) -> None:
    n_sae = 65_536
    mask = FeatureMask("whitelist", tuple(range(0, n_sae, 4)))
    sparse_rng = t.Generator().manual_seed(98)
    dense_rng = t.Generator().manual_seed(98)
    actual = sample_inactive_features_batched(
        [active],
        n_sae,
        CFG,
        sparse_rng,
        feature_mask=mask,
    )
    expected = _dense_masked_batched_reference(
        monkeypatch,
        active,
        n_sae,
        CFG,
        dense_rng,
        mask,
    )
    assert t.equal(actual[0], expected[0])
    assert t.equal(sparse_rng.get_state(), dense_rng.get_state())


def test_positive_oob_sparse_domain_edge_preserves_dense_failure_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_sae = 65_536
    active = t.tensor([n_sae], dtype=t.int64)
    mask = FeatureMask("whitelist", tuple(range(0, n_sae, 4)))
    sparse_rng = t.Generator().manual_seed(38)
    dense_rng = t.Generator().manual_seed(38)
    with pytest.raises(IndexError) as sparse_error:
        sample_inactive_features_batched(
            [active],
            n_sae,
            CFG,
            sparse_rng,
            feature_mask=mask,
        )
    with pytest.raises(IndexError) as dense_error:
        _dense_masked_batched_reference(
            monkeypatch,
            active,
            n_sae,
            CFG,
            dense_rng,
            mask,
        )
    assert str(sparse_error.value) == str(dense_error.value)
    assert t.equal(sparse_rng.get_state(), dense_rng.get_state())


def test_sparse_masked_retry_path_is_exact_dense_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny allowed universe forces the established retry sampler path."""
    n_sae = 65_536
    mask = FeatureMask("whitelist", (101, 203, 307, 409, 503, 601))
    active = t.tensor([101], dtype=t.int64)
    config = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=12.0,
        std=12.0,
        min_k=5,
        max_k=6,
    )
    for seed in range(40):
        sparse_rng = t.Generator().manual_seed(seed)
        dense_rng = t.Generator().manual_seed(seed)
        sparse = sample_inactive_features_batched(
            [active],
            n_sae,
            config,
            sparse_rng,
            feature_mask=mask,
        )
        dense = _dense_masked_batched_reference(
            monkeypatch,
            active,
            n_sae,
            config,
            dense_rng,
            mask,
        )
        assert t.equal(sparse[0], dense[0])
        assert t.equal(sparse_rng.get_state(), dense_rng.get_state())
        assert len(sparse[0]) == 5
        assert set(sparse[0].tolist()) == set(mask.ids) - {101}


def test_sparse_masked_sampling_cannot_mutate_public_mask_results() -> None:
    n_sae = 65_536
    mask = FeatureMask("whitelist", tuple(range(0, n_sae, 4)))
    expected = mask.allowed_universe_mask(n_sae)
    sample_inactive_features_batched(
        [t.arange(0, 512, 8, dtype=t.int64)],
        n_sae,
        CFG,
        t.Generator().manual_seed(71),
        feature_mask=mask,
    )
    returned = mask.allowed_universe_mask(n_sae)
    assert t.equal(returned, expected)
    returned.zero_()
    assert t.equal(mask.allowed_universe_mask(n_sae), expected)


def test_sparse_masked_path_eliminates_per_call_full_width_zero_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_sae = 65_536
    mask = FeatureMask("whitelist", tuple(range(0, n_sae, 4)))
    mask._allowed_feature_ids_cached(n_sae)  # isolate steady-state allocations
    active = t.arange(0, 768, 3, dtype=t.int64)
    zero_shapes: list[tuple[int, ...]] = []
    real_zeros = t.zeros

    def recording_zeros(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            normalized = tuple(shape[0])
        else:
            normalized = tuple(shape)
        zero_shapes.append(normalized)
        return real_zeros(*shape, **kwargs)

    monkeypatch.setattr(sampling.t, "zeros", recording_zeros)
    sample_inactive_features_batched(
        [active],
        n_sae,
        CFG,
        t.Generator().manual_seed(712),
        feature_mask=mask,
    )
    assert (n_sae,) not in zero_shapes
    assert (1, n_sae) not in zero_shapes

    zero_shapes.clear()
    monkeypatch.setattr(sampling, "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT", -1)
    sample_inactive_features_batched(
        [active],
        n_sae,
        CFG,
        t.Generator().manual_seed(712),
        feature_mask=mask,
    )
    assert (n_sae,) in zero_shapes
    assert (1, n_sae) in zero_shapes


def test_no_mask_bit_identical():
    # I5: feature_mask=None must produce the exact legacy sequence.
    r1 = t.Generator().manual_seed(123)
    r2 = t.Generator().manual_seed(123)
    a = sample_inactive_features(t.tensor([1, 2]), 100, CFG, r1)
    b = sample_inactive_features(t.tensor([1, 2]), 100, CFG, r2, feature_mask=None)
    assert t.equal(a, b)


def test_no_mask_rng_state_bit_identical():
    """I5 (strict form): feature_mask=None must leave the torch Generator state
    byte-identical to the legacy (no-feature_mask-kwarg) path.

    This test proves the no-mask path consumes the RNG identically — not just
    that the output sequences are equal, but that the generator state after the
    call is the same.  A codepath that produces equal outputs but consumes a
    different number of random values would desync downstream draws.

    Scope: sample_inactive_features uses only the supplied torch.Generator (rng).
    It does not consume numpy or Python random state.  The generator state check
    is therefore sufficient.

    The test also re-asserts output equality (belt-and-suspenders).
    """
    seed = 456
    active = t.tensor([1, 2, 5], dtype=t.int64)
    n_sae = 100

    # Legacy path (no feature_mask kwarg)
    gen_legacy = t.Generator().manual_seed(seed)
    out_legacy = sample_inactive_features(active, n_sae, CFG, gen_legacy)
    state_legacy = gen_legacy.get_state()  # uint8 tensor

    # No-mask path (explicit feature_mask=None)
    gen_nomask = t.Generator().manual_seed(seed)
    out_nomask = sample_inactive_features(active, n_sae, CFG, gen_nomask, feature_mask=None)
    state_nomask = gen_nomask.get_state()  # uint8 tensor

    # Outputs must be identical
    assert t.equal(out_legacy, out_nomask), (
        f"I5: no-mask output differs from legacy output: "
        f"{out_legacy.tolist()} vs {out_nomask.tolist()}"
    )

    # Generator state must be byte-identical (same RNG draw count)
    assert t.equal(state_legacy, state_nomask), (
        "I5 RNG-state violation: feature_mask=None consumed a different number of "
        "random values than the legacy (no-kwarg) path. "
        "This would desync downstream RNG draws even when outputs happen to match."
    )


def test_batched_no_mask_rng_state_bit_identical():
    """I5 (strict form, batched): sample_inactive_features_batched with feature_mask=None
    must leave the torch Generator state byte-identical to the legacy no-kwarg path.

    Same rationale as test_no_mask_rng_state_bit_identical but for the batched variant.
    """
    seed = 789
    active_list = [t.tensor([0, 3], dtype=t.int64), t.tensor([10], dtype=t.int64)]
    n_sae = 100

    # Legacy path (no feature_mask kwarg)
    gen_legacy = t.Generator().manual_seed(seed)
    out_legacy = sample_inactive_features_batched(active_list, n_sae, CFG, gen_legacy)
    state_legacy = gen_legacy.get_state()

    # No-mask path (explicit feature_mask=None)
    gen_nomask = t.Generator().manual_seed(seed)
    out_nomask = sample_inactive_features_batched(
        active_list, n_sae, CFG, gen_nomask, feature_mask=None
    )
    state_nomask = gen_nomask.get_state()

    # Outputs must be identical per segment
    assert len(out_legacy) == len(out_nomask), "Number of output segments must match"
    for i, (a, b) in enumerate(zip(out_legacy, out_nomask, strict=True)):
        assert t.equal(a, b), (
            f"I5 batched: segment {i} output differs from legacy: "
            f"{a.tolist()} vs {b.tolist()}"
        )

    # Generator state must be byte-identical
    assert t.equal(state_legacy, state_nomask), (
        "I5 batched RNG-state violation: feature_mask=None consumed a different number of "
        "random values than the legacy (no-kwarg) path in sample_inactive_features_batched."
    )


# ---------------------------------------------------------------------------
# Small-pool undersample guard (TestSmallPoolUndersample)
# ---------------------------------------------------------------------------


class TestSmallPoolUndersample:
    """Pin graceful degradation when the mask shrinks the allowed universe below k.

    When the allowed negative-sample pool is SMALLER than the requested k,
    the sampler must:
      (a) not raise / crash;
      (b) return only allowed ids (zero blacklisted / only-whitelisted);
      (c) return ≤ pool size (clamp, not fabricate/pad with masked ids);
      (d) be deterministic for a fixed seed.

    This complements H3's live-delta clamp (test_delta_negatives_mask.py).
    """

    def test_single_blacklist_small_pool_no_crash(self) -> None:
        """Blacklist leaves only 3 ids; k > 3 → sampler clamps, no exception."""
        # n_sae=10, blacklist 0-6 → only {7,8,9} allowed (pool=3)
        n_sae = 10
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(7)))  # 0-6 blacklisted
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        rng = t.Generator().manual_seed(0)
        out = sample_inactive_features(t.tensor([], dtype=t.int64), n_sae, cfg, rng, feature_mask=mask)
        # (a) no exception
        # (b) only allowed ids
        assert set(out.tolist()).issubset({7, 8, 9}), (
            f"Blacklist small-pool: output {out.tolist()} contains non-allowed ids"
        )
        # (c) result length ≤ pool size (3)
        assert out.numel() <= 3, (
            f"Blacklist small-pool: result length {out.numel()} > pool size 3"
        )

    def test_single_whitelist_small_pool_no_crash(self) -> None:
        """Whitelist of 2 ids; k > 2 → sampler clamps, no exception."""
        n_sae = 50
        mask = FeatureMask(FeatureMaskMode.whitelist, (10, 20))  # only 10 and 20 allowed (pool=2)
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        rng = t.Generator().manual_seed(0)
        out = sample_inactive_features(t.tensor([], dtype=t.int64), n_sae, cfg, rng, feature_mask=mask)
        # (a) no exception
        # (b) only allowed ids
        assert set(out.tolist()).issubset({10, 20}), (
            f"Whitelist small-pool: output {out.tolist()} contains non-whitelisted ids"
        )
        # (c) result length ≤ pool size (2)
        assert out.numel() <= 2, (
            f"Whitelist small-pool: result length {out.numel()} > pool size 2"
        )

    def test_single_small_pool_deterministic(self) -> None:
        """Same seed → same result for small-pool blacklist case."""
        n_sae = 10
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(7)))  # pool={7,8,9}
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        rng_a = t.Generator().manual_seed(42)
        rng_b = t.Generator().manual_seed(42)
        out_a = sample_inactive_features(t.tensor([], dtype=t.int64), n_sae, cfg, rng_a, feature_mask=mask)
        out_b = sample_inactive_features(t.tensor([], dtype=t.int64), n_sae, cfg, rng_b, feature_mask=mask)
        assert t.equal(out_a, out_b), (
            f"Small-pool: non-deterministic result: {out_a.tolist()} vs {out_b.tolist()}"
        )

    def test_batched_blacklist_small_pool_no_crash(self) -> None:
        """Batched: blacklist leaves tiny pool; every segment respects the mask + pool clamp."""
        n_sae = 10
        mask = FeatureMask(FeatureMaskMode.blacklist, tuple(range(7)))  # pool={7,8,9} (size 3)
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        rng = t.Generator().manual_seed(0)
        active_list = [t.tensor([], dtype=t.int64), t.tensor([7], dtype=t.int64)]
        results = sample_inactive_features_batched(active_list, n_sae, cfg, rng, feature_mask=mask)
        assert len(results) == 2
        # Segment 0: allowed pool = {7,8,9}, active=[] → pool still 3
        assert set(results[0].tolist()).issubset({7, 8, 9}), (
            f"Batched seg0 blacklist: {results[0].tolist()} contains disallowed id"
        )
        assert results[0].numel() <= 3
        # Segment 1: allowed pool = {7,8,9}, active=[7] → pool = {8,9} (size 2)
        assert set(results[1].tolist()).issubset({8, 9}), (
            f"Batched seg1 blacklist: {results[1].tolist()} contains disallowed/active id"
        )
        assert results[1].numel() <= 2

    def test_batched_whitelist_small_pool_no_crash(self) -> None:
        """Batched: whitelist of 2 ids; every segment respects the whitelist + pool clamp."""
        n_sae = 50
        mask = FeatureMask(FeatureMaskMode.whitelist, (10, 20))  # allowed={10,20}
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        rng = t.Generator().manual_seed(0)
        # Two segments; segment 1 has 10 active (leaving only {20} allowed)
        active_list = [t.tensor([], dtype=t.int64), t.tensor([10], dtype=t.int64)]
        results = sample_inactive_features_batched(active_list, n_sae, cfg, rng, feature_mask=mask)
        assert len(results) == 2
        # Segment 0: pool={10,20}
        assert set(results[0].tolist()).issubset({10, 20}), (
            f"Batched seg0 whitelist: {results[0].tolist()} contains non-whitelisted id"
        )
        assert results[0].numel() <= 2
        # Segment 1: pool={20} (10 is active)
        assert set(results[1].tolist()).issubset({20}), (
            f"Batched seg1 whitelist: {results[1].tolist()} contains non-whitelisted/active id"
        )
        assert results[1].numel() <= 1

    def test_batched_whitelist_partial_underfill_falls_back(self) -> None:
        """If oversampling finds only one valid id from a two-id whitelist, retry."""
        n_sae = 50
        mask = FeatureMask(FeatureMaskMode.whitelist, (10, 20))
        cfg = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=10.0,
            std=1.0,
            min_k=5,
            max_k=20,
        )
        rng = t.Generator().manual_seed(2)

        results = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)],
            n_sae,
            cfg,
            rng,
            feature_mask=mask,
        )

        assert len(results) == 1
        assert set(results[0].tolist()) == {10, 20}

    def test_batched_small_pool_deterministic(self) -> None:
        """Same seed → same batched result for small-pool whitelist case."""
        n_sae = 50
        mask = FeatureMask(FeatureMaskMode.whitelist, (10, 20))
        cfg = NegativeSamplingSelection(mean=10.0, std=10.0, min_k=5, max_k=20)
        active_list = [t.tensor([], dtype=t.int64), t.tensor([10], dtype=t.int64)]
        rng_a = t.Generator().manual_seed(99)
        rng_b = t.Generator().manual_seed(99)
        out_a = sample_inactive_features_batched(active_list, n_sae, cfg, rng_a, feature_mask=mask)
        out_b = sample_inactive_features_batched(active_list, n_sae, cfg, rng_b, feature_mask=mask)
        assert len(out_a) == len(out_b)
        for i, (a, b) in enumerate(zip(out_a, out_b, strict=True)):
            assert t.equal(a, b), (
                f"Batched small-pool: non-deterministic at segment {i}: "
                f"{a.tolist()} vs {b.tolist()}"
            )
        assert t.equal(rng_a.get_state(), rng_b.get_state())
