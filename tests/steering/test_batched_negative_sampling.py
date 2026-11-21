"""Tests for sample_inactive_features_batched correctness."""

from hashlib import sha256

import pytest
import torch as t

import steering.conditional.sampling as sampling
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    sample_inactive_features_batched,
)

N_SAE = 1000


def _tensor_bytes(tensor: t.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _tensor_sha256(tensor: t.Tensor) -> str:
    return sha256(_tensor_bytes(tensor)).hexdigest()


def _origin_main_masked_batched_reference(
    active_ids_list: list[t.Tensor],
    n_sae_features: int,
    config: NegativeSamplingSelection,
    rng: t.Generator,
    feature_mask: FeatureMask,
) -> list[t.Tensor]:
    """Frozen copy of origin/main's masked dense batched result loop."""
    n_seg = len(active_ids_list)
    if n_seg == 0:
        return []

    all_ks = sampling.sample_k_batched(
        config.distribution,
        config.mean,
        config.std,
        rng,
        n_seg,
        min_k=config.min_k,
        max_k=config.max_k,
    )
    masks = t.zeros(n_seg, n_sae_features, dtype=t.bool)
    for row_index, active_ids in enumerate(active_ids_list):
        if active_ids.numel() > 0:
            masks[row_index, active_ids.long()] = True
    allowed = feature_mask.allowed_universe_mask(n_sae_features)
    masks |= (~allowed).unsqueeze(0)
    pool_sizes = (~masks).sum(dim=1)
    all_ks = t.minimum(all_ks, pool_sizes).clamp(min=0)

    max_k = int(all_ks.max().item())
    if max_k == 0:
        return [t.tensor([], dtype=t.int64)] * n_seg

    candidates = t.randint(
        0,
        n_sae_features,
        (n_seg, max_k * 3),
        generator=rng,
    )
    segment_indices = t.arange(n_seg).unsqueeze(1).expand_as(candidates)
    is_collision = masks[segment_indices, candidates]

    sorted_candidates, sort_indices = candidates.sort(dim=1, stable=True)
    duplicate_in_sorted_order = t.zeros_like(candidates, dtype=t.bool)
    duplicate_in_sorted_order[:, 1:] = (
        sorted_candidates[:, 1:] == sorted_candidates[:, :-1]
    )
    is_duplicate = t.zeros_like(duplicate_in_sorted_order)
    is_duplicate.scatter_(1, sort_indices, duplicate_in_sorted_order)
    is_valid = ~is_collision & ~is_duplicate
    valid_rank = is_valid.long().cumsum(dim=1)
    selected = is_valid & (valid_rank <= all_ks.unsqueeze(1))

    results: list[t.Tensor] = []
    for row_index, active_ids in enumerate(active_ids_list):
        k = int(all_ks[row_index].item())
        selected_mask = selected[row_index]
        n_selected = int(selected_mask.sum().item())
        if bool(selected_mask.any()) and (
            n_selected == k or feature_mask is None
        ):
            results.append(candidates[row_index, selected_mask].to(t.int64))
        elif k > 0:
            results.append(
                sampling.sample_inactive_features(
                    active_ids,
                    n_sae_features,
                    config,
                    rng,
                    feature_mask=feature_mask,
                )
            )
        else:
            results.append(t.tensor([], dtype=t.int64))
    return results


def _assert_well_formed_row(
    sampled_ids: t.Tensor,
    active_ids: t.Tensor,
    n_sae_features: int,
    config: NegativeSamplingSelection,
    feature_mask: FeatureMask | None,
) -> None:
    if sampled_ids.numel() == 0:
        return
    assert sampled_ids.unique().numel() == sampled_ids.numel()
    assert bool(((sampled_ids >= 0) & (sampled_ids < n_sae_features)).all())
    if feature_mask is not None:
        assert bool(feature_mask.keep_mask(sampled_ids).all())
        eligible = feature_mask.allowed_universe_mask(n_sae_features)
    else:
        eligible = t.ones(n_sae_features, dtype=t.bool)
    if active_ids.numel() > 0:
        eligible[active_ids.long()] = False
    assert not bool(t.isin(sampled_ids, active_ids).any())
    pool_size = int(eligible.sum().item())
    maximum = (
        pool_size
        if config.max_k is None
        else min(config.max_k, pool_size)
    )
    assert sampled_ids.numel() <= maximum


@pytest.fixture
def neg_config():
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=8.0,
        std=8.0,
        min_k=0,
    )


class TestBatchedNegativeSampling:
    """Tests for sample_inactive_features_batched."""

    def test_first_unique_candidate_mask_preserves_draw_order(self):
        """Duplicate candidate IDs keep their first RNG-drawn column."""
        candidates = t.tensor([
            [5, 7, 5, 8, 7, 9],
            [3, 3, 4, 4, 4, 5],
        ])
        mask = sampling._first_unique_candidate_mask(candidates)
        expected = t.tensor([
            [True, True, False, True, False, True],
            [True, False, True, False, False, True],
        ])
        assert t.equal(mask, expected)

    def test_first_unique_candidate_mask_matches_python_reference(self):
        """Exhaustive small-shape check against a direct first-seen reference."""
        gen = t.Generator().manual_seed(0)
        for n_rows in range(1, 5):
            for n_cols in range(12):
                for n_ids in range(1, 7):
                    candidates = t.randint(0, n_ids, (n_rows, n_cols), generator=gen)
                    expected_rows = []
                    for row in candidates.tolist():
                        seen = set()
                        expected_row = []
                        for value in row:
                            expected_row.append(value not in seen)
                            seen.add(value)
                        expected_rows.append(expected_row)
                    expected = t.tensor(expected_rows, dtype=t.bool)
                    mask = sampling._first_unique_candidate_mask(candidates)
                    assert t.equal(mask, expected), (
                        f"first-unique mismatch for candidates={candidates.tolist()}"
                    )

    def test_duplicate_candidates_selected_in_first_unique_draw_order(
        self, monkeypatch, neg_config
    ):
        """The batched sampler selects the first k unique valid candidates in draw order."""
        forced_candidates = t.tensor([[5, 7, 5, 8, 7, 9, 10, 11, 12, 13, 14, 15]])

        def fake_sample_k_batched(*args, **kwargs):
            return t.tensor([4], dtype=t.int64)

        def fake_randint(low, high, size, *, generator=None):
            assert (low, high, size) == (0, N_SAE, forced_candidates.shape)
            return forced_candidates.clone()

        monkeypatch.setattr(sampling, "sample_k_batched", fake_sample_k_batched)
        monkeypatch.setattr(sampling.t, "randint", fake_randint)

        results = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)],
            N_SAE,
            neg_config,
            t.Generator().manual_seed(0),
        )

        assert len(results) == 1
        assert t.equal(results[0], t.tensor([5, 7, 8, 9]))

    def test_duplicate_candidates_respect_collision_and_mask_order(
        self, monkeypatch, neg_config
    ):
        """First-unique filtering composes with active-id and feature-mask exclusion."""
        forced_candidates = t.tensor([[5, 8, 5, 7, 8, 9, 10, 11, 12]])

        def fake_sample_k_batched(*args, **kwargs):
            return t.tensor([3], dtype=t.int64)

        def fake_randint(low, high, size, *, generator=None):
            assert (low, high, size) == (0, N_SAE, forced_candidates.shape)
            return forced_candidates.clone()

        monkeypatch.setattr(sampling, "sample_k_batched", fake_sample_k_batched)
        monkeypatch.setattr(sampling.t, "randint", fake_randint)

        results = sample_inactive_features_batched(
            [t.tensor([5], dtype=t.int64)],
            N_SAE,
            neg_config,
            t.Generator().manual_seed(0),
            feature_mask=FeatureMask(FeatureMaskMode.blacklist, (7,)),
        )

        assert len(results) == 1
        assert t.equal(results[0], t.tensor([8, 9, 10]))

    def test_exclusion_correctness(self, neg_config):
        """No returned ID should be in the active set for any segment."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [
            t.tensor([0, 1, 2, 10, 50], dtype=t.int64),
            t.tensor([100, 200, 300], dtype=t.int64),
            t.tensor([5], dtype=t.int64),
        ]
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        for i, (active, neg) in enumerate(zip(active_ids_list, results, strict=True)):
            active_set = set(active.tolist())
            neg_set = set(neg.tolist())
            assert neg_set.isdisjoint(active_set), (
                f"Segment {i}: overlap between active {active_set & neg_set}"
            )

    def test_uniqueness(self, neg_config):
        """No duplicate IDs per segment."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [t.randperm(N_SAE)[:50] for _ in range(20)]
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        for i, neg in enumerate(results):
            assert len(set(neg.tolist())) == len(neg), f"Segment {i}: duplicates in {neg.tolist()}"

    def test_id_range(self, neg_config):
        """All returned IDs in [0, n_sae_features)."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [t.tensor([0, 1, 2], dtype=t.int64)] * 10
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        for i, neg in enumerate(results):
            if len(neg) > 0:
                assert neg.min().item() >= 0, f"Segment {i}: negative ID below 0"
                assert neg.max().item() < N_SAE, f"Segment {i}: ID >= {N_SAE}"

    def test_k_distribution(self):
        """Geometric mean=8 produces empirical mean within [5, 12] over many segments."""
        config = NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=8.0, std=8.0, min_k=0,
        )
        rng = t.Generator().manual_seed(42)
        n_seg = 2000
        active_ids_list = [t.tensor([0, 1], dtype=t.int64)] * n_seg
        results = sample_inactive_features_batched(active_ids_list, N_SAE, config, rng)
        ks = [len(r) for r in results]
        empirical_mean = sum(ks) / len(ks)
        assert 5.0 < empirical_mean < 12.0, f"Mean k={empirical_mean} outside [5, 12]"

    def test_pool_size_clamping(self, neg_config):
        """When pool_size < requested k, returns pool_size features."""
        rng = t.Generator().manual_seed(0)
        # Only 3 features total, 2 active → pool_size=1
        active_ids_list = [t.tensor([0, 1], dtype=t.int64)]
        results = sample_inactive_features_batched(active_ids_list, 3, neg_config, rng)
        assert len(results[0]) <= 1

    def test_all_features_active(self, neg_config):
        """pool_size=0 returns empty for all segments."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [t.arange(N_SAE, dtype=t.int64)] * 5
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        for i, neg in enumerate(results):
            assert len(neg) == 0, f"Segment {i}: expected empty, got {len(neg)}"

    def test_empty_active_list(self, neg_config):
        """All features eligible when active set is empty."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [t.tensor([], dtype=t.int64)] * 5
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        for _i, neg in enumerate(results):
            if len(neg) > 0:
                assert neg.min().item() >= 0
                assert neg.max().item() < N_SAE

    def test_empty_input_list(self, neg_config):
        """Empty input list returns empty output list."""
        rng = t.Generator().manual_seed(0)
        results = sample_inactive_features_batched([], N_SAE, neg_config, rng)
        assert results == []

    def test_mixed_empty_and_nonempty_active(self, neg_config):
        """Handles segments with varying active set sizes."""
        rng = t.Generator().manual_seed(0)
        active_ids_list = [
            t.tensor([], dtype=t.int64),
            t.arange(N_SAE, dtype=t.int64),  # all active
            t.tensor([5, 10, 15], dtype=t.int64),
        ]
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        assert len(results) == 3
        # Second segment: all active → empty
        assert len(results[1]) == 0
        # Third segment: negatives disjoint from active
        if len(results[2]) > 0:
            assert set(results[2].tolist()).isdisjoint({5, 10, 15})

    def test_single_segment(self, neg_config):
        """Single segment produces valid results."""
        rng = t.Generator().manual_seed(0)
        active = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        results = sample_inactive_features_batched([active], N_SAE, neg_config, rng)
        assert len(results) == 1
        neg = results[0]
        if len(neg) > 0:
            assert set(neg.tolist()).isdisjoint(set(active.tolist()))
            assert len(set(neg.tolist())) == len(neg)

    def test_large_scale(self, neg_config):
        """1000 segments — all properties hold."""
        rng = t.Generator().manual_seed(42)
        t.manual_seed(99)
        n_seg = 1000
        active_ids_list = [t.randperm(N_SAE)[:t.randint(1, 50, (1,)).item()] for _ in range(n_seg)]
        results = sample_inactive_features_batched(active_ids_list, N_SAE, neg_config, rng)
        assert len(results) == n_seg
        for i, (active, neg) in enumerate(zip(active_ids_list, results, strict=True)):
            if len(neg) > 0:
                active_set = set(active.tolist())
                neg_set = set(neg.tolist())
                assert neg_set.isdisjoint(active_set), f"Segment {i}: overlap"
                assert len(neg_set) == len(neg), f"Segment {i}: duplicates"
                assert neg.min().item() >= 0
                assert neg.max().item() < N_SAE


class TestBatchedNegativeFallbackContract:
    @pytest.mark.parametrize(
        ("stride", "count"),
        [
            pytest.param(2, None, id="whitelist-50-percent"),
            pytest.param(4, None, id="whitelist-25-percent"),
            pytest.param(10, None, id="whitelist-10-percent"),
            pytest.param(None, 1024, id="whitelist-1024-ids"),
        ],
    )
    @pytest.mark.parametrize(
        "force_dense",
        [False, True],
        ids=["sparse", "dense"],
    )
    def test_masked_outputs_and_rng_match_frozen_origin_main_over_20_seeds(
        self,
        monkeypatch,
        stride,
        count,
        force_dense,
    ):
        if force_dense:
            monkeypatch.setattr(
                sampling,
                "_SINGLE_SEGMENT_SPARSE_ACTIVE_LIMIT",
                -1,
            )
        n_sae_features = 65_536
        whitelist = (
            tuple(range(count))
            if count is not None
            else tuple(range(0, n_sae_features, stride))
        )
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=32.0,
            std=8.0,
            min_k=0,
            max_k=49,
        )
        active_ids_list = [
            t.tensor([], dtype=t.int64),
            t.arange(0, 257, 17, dtype=t.int64),
            t.arange(511, 1537, 29, dtype=t.int64),
            t.tensor(
                [0, 0, 1, 1023, 1024, n_sae_features - 1],
                dtype=t.int64,
            ),
        ]

        actual_mask = FeatureMask(FeatureMaskMode.whitelist, whitelist)
        reference_mask = FeatureMask(FeatureMaskMode.whitelist, whitelist)
        for seed in range(20):
            actual_rng = t.Generator().manual_seed(seed)
            reference_rng = t.Generator().manual_seed(seed)
            actual = sample_inactive_features_batched(
                active_ids_list,
                n_sae_features,
                config,
                actual_rng,
                feature_mask=actual_mask,
            )
            expected = _origin_main_masked_batched_reference(
                active_ids_list,
                n_sae_features,
                config,
                reference_rng,
                reference_mask,
            )

            assert len(actual) == len(expected)
            for actual_row, expected_row in zip(actual, expected, strict=True):
                assert actual_row.dtype == expected_row.dtype
                assert actual_row.shape == expected_row.shape
                assert actual_row.device == expected_row.device
                assert _tensor_bytes(actual_row) == _tensor_bytes(expected_row)
            assert t.equal(actual_rng.get_state(), reference_rng.get_state())

    def test_narrow_whitelist_succeeds_and_is_well_formed_over_25_seeds(
        self,
    ):
        n_sae_features = 65_536
        feature_mask = FeatureMask(
            FeatureMaskMode.whitelist,
            tuple(range(1024)),
        )
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=32.0,
            std=8.0,
            min_k=0,
            max_k=49,
        )
        active_ids_list = [
            t.tensor([], dtype=t.int64),
            t.arange(0, 128, dtype=t.int64),
            t.arange(900, 1100, dtype=t.int64),
            t.tensor([0, 0, 512, 2048, 65_535], dtype=t.int64),
        ]

        for seed in range(25):
            rows = sample_inactive_features_batched(
                active_ids_list,
                n_sae_features,
                config,
                t.Generator().manual_seed(seed),
                feature_mask=feature_mask,
            )
            assert len(rows) == len(active_ids_list)
            for row, active_ids in zip(rows, active_ids_list, strict=True):
                _assert_well_formed_row(
                    row,
                    active_ids,
                    n_sae_features,
                    config,
                    feature_mask,
                )

    def test_no_mask_partial_row_is_replaced_by_scalar_fallback(
        self,
        monkeypatch,
    ):
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=0.01,
            min_k=2,
            max_k=3,
        )
        active_ids = t.tensor([0], dtype=t.int64)
        forced_candidates = t.ones((1, 6), dtype=t.int64)
        old_partial_row = forced_candidates[
            0,
            sampling._first_unique_candidate_mask(forced_candidates)[0],
        ]
        assert old_partial_row.numel() == 1

        monkeypatch.setattr(
            sampling,
            "sample_k_batched",
            lambda *args, **kwargs: t.tensor([2], dtype=t.int64),
        )
        real_randint = sampling.t.randint

        def force_only_batched_draw(low, high, size, *, generator=None):
            if size == forced_candidates.shape:
                return forced_candidates.clone()
            return real_randint(low, high, size, generator=generator)

        monkeypatch.setattr(sampling.t, "randint", force_only_batched_draw)
        real_scalar_sampler = sampling.sample_inactive_features
        scalar_calls = 0

        def record_scalar_fallback(*args, **kwargs):
            nonlocal scalar_calls
            scalar_calls += 1
            return real_scalar_sampler(*args, **kwargs)

        monkeypatch.setattr(
            sampling,
            "sample_inactive_features",
            record_scalar_fallback,
        )

        rows = sample_inactive_features_batched(
            [active_ids],
            10,
            config,
            t.Generator().manual_seed(0),
        )

        assert scalar_calls == 1
        assert rows[0].numel() == 2
        _assert_well_formed_row(rows[0], active_ids, 10, config, None)

    @pytest.mark.parametrize(
        ("sampled_ids", "active_ids", "feature_mask", "pool_size", "invariant"),
        [
            pytest.param(
                t.tensor([1, 1], dtype=t.int64),
                t.tensor([], dtype=t.int64),
                None,
                10,
                "uniqueness",
                id="unique",
            ),
            pytest.param(
                t.tensor([-1, 2], dtype=t.int64),
                t.tensor([], dtype=t.int64),
                None,
                10,
                "id range",
                id="range",
            ),
            pytest.param(
                t.tensor([2], dtype=t.int64),
                t.tensor([], dtype=t.int64),
                FeatureMask(FeatureMaskMode.whitelist, (1, 3)),
                2,
                "feature-mask eligibility",
                id="mask-allowed",
            ),
            pytest.param(
                t.tensor([2], dtype=t.int64),
                t.tensor([2], dtype=t.int64),
                None,
                9,
                "active-id disjointness",
                id="inactive",
            ),
            pytest.param(
                t.tensor([1, 2, 3], dtype=t.int64),
                t.tensor([], dtype=t.int64),
                None,
                2,
                "maximum count",
                id="bounded-count",
            ),
        ],
    )
    def test_final_row_invariant_names_row_and_violation(
        self,
        sampled_ids,
        active_ids,
        feature_mask,
        pool_size,
        invariant,
    ):
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=2.0,
            std=1.0,
            min_k=0,
            max_k=4,
        )

        with pytest.raises(
            ValueError,
            match=rf"row 7.*{invariant} invariant",
        ):
            sampling._require_well_formed_batched_negative_row(
                row_index=7,
                sampled_ids=sampled_ids,
                active_ids=active_ids,
                n_sae_features=10,
                config=config,
                feature_mask=feature_mask,
                pool_size=pool_size,
            )

    def test_final_row_invariant_consumes_no_rng(self):
        global_state = t.random.get_rng_state().clone()
        sampling._require_well_formed_batched_negative_row(
            row_index=0,
            sampled_ids=t.tensor([1, 3], dtype=t.int64),
            active_ids=t.tensor([2], dtype=t.int64),
            n_sae_features=10,
            config=NegativeSamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=1.0,
                min_k=0,
                max_k=4,
            ),
            feature_mask=FeatureMask(
                FeatureMaskMode.whitelist,
                (1, 3, 5),
            ),
            pool_size=3,
        )
        assert t.equal(global_state, t.random.get_rng_state())

    def test_fallback_output_is_checked_by_final_row_invariant(
        self,
        monkeypatch,
    ):
        forced_candidates = t.ones((1, 6), dtype=t.int64)
        monkeypatch.setattr(
            sampling,
            "sample_k_batched",
            lambda *args, **kwargs: t.tensor([2], dtype=t.int64),
        )
        monkeypatch.setattr(
            sampling.t,
            "randint",
            lambda *args, **kwargs: forced_candidates.clone(),
        )
        monkeypatch.setattr(
            sampling,
            "sample_inactive_features",
            lambda *args, **kwargs: t.tensor([3, 3], dtype=t.int64),
        )

        with pytest.raises(
            ValueError,
            match=r"row 0.*uniqueness invariant",
        ):
            sample_inactive_features_batched(
                [t.tensor([], dtype=t.int64)],
                10,
                NegativeSamplingSelection(
                    distribution=Distribution.normal,
                    mean=2.0,
                    std=0.01,
                    min_k=2,
                    max_k=3,
                ),
                t.Generator().manual_seed(0),
            )

    def test_fast_output_is_checked_by_final_row_invariant(
        self,
        monkeypatch,
    ):
        forced_candidates = t.ones((1, 6), dtype=t.int64)
        monkeypatch.setattr(
            sampling,
            "sample_k_batched",
            lambda *args, **kwargs: t.tensor([2], dtype=t.int64),
        )
        monkeypatch.setattr(
            sampling.t,
            "randint",
            lambda *args, **kwargs: forced_candidates.clone(),
        )
        monkeypatch.setattr(
            sampling,
            "_first_unique_candidate_mask",
            lambda candidates: t.ones_like(candidates, dtype=t.bool),
        )

        with pytest.raises(
            ValueError,
            match=r"row 0.*uniqueness invariant",
        ):
            sample_inactive_features_batched(
                [t.tensor([], dtype=t.int64)],
                10,
                NegativeSamplingSelection(
                    distribution=Distribution.normal,
                    mean=2.0,
                    std=0.01,
                    min_k=2,
                    max_k=3,
                ),
                t.Generator().manual_seed(0),
            )

    def test_successful_masked_fast_path_matches_frozen_origin_main(self):
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=4.0,
            std=0.01,
            min_k=4,
            max_k=5,
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, (999,))
        active = [
            t.tensor([0, 1], dtype=t.int64),
            t.tensor([2, 3, 4], dtype=t.int64),
            t.tensor([], dtype=t.int64),
        ]
        rng = t.Generator().manual_seed(123)

        actual = sample_inactive_features_batched(
            active,
            1000,
            config,
            rng,
            feature_mask=mask,
        )
        expected = [
            t.tensor([699, 754, 411, 721], dtype=t.int64),
            t.tensor([350, 16, 655, 497], dtype=t.int64),
            t.tensor([23, 179, 554, 516], dtype=t.int64),
        ]

        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert actual_row.dtype == expected_row.dtype
            assert actual_row.shape == expected_row.shape
            assert actual_row.device == expected_row.device
            assert _tensor_bytes(actual_row) == _tensor_bytes(expected_row)
        assert _tensor_sha256(rng.get_state()) == (
            "2a64b2c6f885af3f76d90306781456ff456b9d6fc3f74656d3b857e3f83f3220"
        )

    def test_successful_no_mask_path_matches_frozen_origin_main(self):
        config = NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=3.0,
            std=0.01,
            min_k=3,
            max_k=4,
        )
        active = [
            t.tensor([0, 1], dtype=t.int64),
            t.tensor([2], dtype=t.int64),
            t.tensor([], dtype=t.int64),
        ]
        rng = t.Generator().manual_seed(321)

        actual = sample_inactive_features_batched(
            active,
            1000,
            config,
            rng,
        )
        expected = [
            t.tensor([258, 760, 644], dtype=t.int64),
            t.tensor([203, 138, 628], dtype=t.int64),
            t.tensor([506, 589, 551], dtype=t.int64),
        ]

        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected, strict=True):
            assert actual_row.dtype == expected_row.dtype
            assert actual_row.shape == expected_row.shape
            assert actual_row.device == expected_row.device
            assert _tensor_bytes(actual_row) == _tensor_bytes(expected_row)
        assert _tensor_sha256(rng.get_state()) == (
            "3be8e55207bde6fdae52315750711c27dc8992bd13277b681c744f92edc6c1bb"
        )
