"""Tests for partition.py (Task 5) — partition bounds + L1 count penalty.

Coverage:
- partition_bounds_from_model: num_buckets from model, has_negative from
  negative_selection presence, all min_k/max_k from cfg (not metadata);
  mean_pos/mean_neg no longer exist in PartitionBounds
- is_active: partitions at num_buckets-1; all treated as active when no neg
- count_penalty(bag, reg_lambda): L1 formula = reg_lambda * len(bag),
  reg_lambda=0 → 0.0, monotone in len(bag)
- Error cases: missing _loaded_meta_data, missing num_buckets
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from errors import FeatureVocabError
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.partition import (
    PartitionBounds,
    count_penalty,
    is_active,
    partition_bounds_from_model,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.cluster_feature_contract import ClusterFeatureContractError
from tests.steering.feature_gcg.unit.test_clustered_optimizer_contract import (
    _contract_model_pair,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_model(
    *,
    num_buckets: int = 8,
    feature_selection: dict | None = None,
    negative_selection: dict | None = None,
) -> SimpleNamespace:
    """Lightweight EncoderDecoderGPT stand-in with _loaded_meta_data."""
    encoder_config = SimpleNamespace(num_buckets=num_buckets)
    model = SimpleNamespace(encoder=SimpleNamespace(config=encoder_config))
    user_config: dict = {}
    if feature_selection is not None:
        user_config["feature_selection"] = feature_selection
    if negative_selection is not None:
        user_config["negative_selection"] = negative_selection
    model._loaded_meta_data = {"user_config": user_config}
    return model


_DEFAULT_CFG = FeatureGcgConfig()


# ─────────────────────────────────────────────────────────────────────────────
# partition_bounds_from_model: exact bounds
# ─────────────────────────────────────────────────────────────────────────────


class TestPartitionBoundsFromModel:
    def test_num_buckets_from_model(self):
        model = _make_model(num_buckets=16)
        b = partition_bounds_from_model(model, _DEFAULT_CFG)
        assert b.num_buckets == 16

    def test_has_negative_false_when_no_negative_selection(self):
        model = _make_model()
        b = partition_bounds_from_model(model, _DEFAULT_CFG)
        assert b.has_negative is False

    def test_has_negative_true_when_negative_selection_present(self):
        model = _make_model(negative_selection={"distribution": "geometric", "mean": 3.0})
        b = partition_bounds_from_model(model, _DEFAULT_CFG)
        assert b.has_negative is True

    def test_min_k_pos_from_cfg(self):
        model = _make_model()
        cfg = FeatureGcgConfig(min_k_pos=3, max_k_pos=10)
        b = partition_bounds_from_model(model, cfg)
        assert b.min_k_pos == 3

    def test_max_k_pos_from_cfg(self):
        model = _make_model()
        cfg = FeatureGcgConfig(min_k_pos=0, max_k_pos=20)
        b = partition_bounds_from_model(model, cfg)
        assert b.max_k_pos == 20

    def test_min_k_neg_from_cfg(self):
        model = _make_model()
        cfg = FeatureGcgConfig(min_k_neg=2, max_k_neg=10)
        b = partition_bounds_from_model(model, cfg)
        assert b.min_k_neg == 2

    def test_max_k_neg_from_cfg(self):
        model = _make_model()
        cfg = FeatureGcgConfig(min_k_neg=0, max_k_neg=12)
        b = partition_bounds_from_model(model, cfg)
        assert b.max_k_neg == 12

    def test_bounds_have_no_mean_fields(self):
        model = _make_model()
        b = partition_bounds_from_model(model, _DEFAULT_CFG)
        assert not hasattr(b, "mean_pos")
        assert not hasattr(b, "mean_neg")

    def test_threshold_selection_not_rejected(self):
        """ThresholdSelection is no longer rejected (feature_selection is not parsed)."""
        model = _make_model()
        # This used to raise but must NOT raise now
        b = partition_bounds_from_model(model, _DEFAULT_CFG)
        assert isinstance(b, PartitionBounds)


# ─────────────────────────────────────────────────────────────────────────────
# is_active
# ─────────────────────────────────────────────────────────────────────────────


class TestIsActive:
    def test_buckets_below_last_are_active(self):
        # num_buckets=8 → buckets 0..6 active, bucket 7 inactive
        for bucket in range(7):
            assert is_active(bucket, 8) is True

    def test_last_bucket_is_inactive(self):
        assert is_active(7, 8) is False

    def test_two_bucket_boundary(self):
        # num_buckets=2: bucket 0 active, bucket 1 inactive
        assert is_active(0, 2) is True
        assert is_active(1, 2) is False

    def test_partitions_at_num_buckets_minus_1_for_various_sizes(self):
        """is_active partitions exactly at num_buckets-1."""
        for nb in (2, 4, 8, 16, 64):
            for b in range(nb):
                expected = b < nb - 1
                assert is_active(b, nb) is expected, (
                    f"is_active({b}, {nb}) should be {expected}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# count_penalty
# ─────────────────────────────────────────────────────────────────────────────


class TestCountPenalty:
    def test_reg_lambda_zero_returns_zero(self):
        bag = FeatureConditioning.from_dict({0: 3, 1: 5})
        assert count_penalty(bag, 0.0) == 0.0

    def test_reg_lambda_zero_empty_bag_returns_zero(self):
        bag = FeatureConditioning(entries=())
        assert count_penalty(bag, 0.0) == 0.0

    def test_exact_l1_formula_single_entry(self):
        bag = FeatureConditioning.from_dict({0: 3})
        assert count_penalty(bag, 1.0) == 1.0

    def test_exact_l1_formula_multiple_entries(self):
        bag = FeatureConditioning.from_dict({0: 3, 1: 5, 2: 6})
        assert count_penalty(bag, 2.0) == 6.0  # 2.0 * 3

    def test_exact_l1_formula_five_entries(self):
        bag = FeatureConditioning.from_dict({0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
        assert count_penalty(bag, 3.0) == 15.0  # 3.0 * 5

    def test_monotone_increasing_with_entries(self):
        bag0 = FeatureConditioning(entries=())
        bag1 = FeatureConditioning(entries=((0, 0),))
        bag2 = FeatureConditioning(entries=((0, 0), (1, 0)))
        assert count_penalty(bag0, 1.0) < count_penalty(bag1, 1.0) < count_penalty(bag2, 1.0)

    def test_reg_lambda_scales_linearly(self):
        bag = FeatureConditioning.from_dict({0: 3})
        p1 = count_penalty(bag, 1.0)
        p2 = count_penalty(bag, 2.0)
        assert p2 == 2.0 * p1

    def test_active_and_inactive_entries_both_count(self):
        """All entries count equally regardless of bucket (no partition split)."""
        # bucket 7 would be 'inactive' in old API; in new L1 it just counts as 1
        bag_active = FeatureConditioning(entries=((0, 0), (1, 0)))  # two 'active'
        bag_mixed = FeatureConditioning(entries=((0, 0), (2, 7)))   # one 'active', one 'inactive'
        # Same len → same penalty regardless of which bucket entries are in
        assert count_penalty(bag_active, 1.0) == count_penalty(bag_mixed, 1.0)

    def test_teeth_l1_formula_differs_from_n_minus_1(self):
        """Teeth: count_penalty != reg_lambda * (len(bag) - 1) for non-empty bags.

        If count_penalty were reg_lambda * (len(bag) - 1) instead of reg_lambda * len(bag),
        a 1-entry bag would give 0, not 1.0. This test catches that bug.
        """
        bag = FeatureConditioning(entries=((5, 0),))
        penalty = count_penalty(bag, 1.0)
        assert penalty == 1.0  # must NOT be 0 (which (len-1)*lambda would give)
        assert penalty != 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Error cases
# ─────────────────────────────────────────────────────────────────────────────


class TestErrors:
    def test_missing_meta_data_raises(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=8))
        )
        with pytest.raises(FeatureGcgError, match="_loaded_meta_data"):
            partition_bounds_from_model(model, _DEFAULT_CFG)

    def test_missing_num_buckets_raises(self):
        """Model without encoder.config.num_buckets raises FeatureGcgError."""
        model = SimpleNamespace(
            encoder=SimpleNamespace(config=SimpleNamespace()),  # no num_buckets
            _loaded_meta_data={"user_config": {}},
        )
        with pytest.raises(FeatureGcgError, match="num_buckets"):
            partition_bounds_from_model(model, _DEFAULT_CFG)


class TestClusterContract:
    def test_cluster_contract_rejects_non_mapping_metadata(
        self,
        tmp_path,
    ):
        contract, model, _ = _contract_model_pair(tmp_path)
        model._loaded_meta_data = []

        with pytest.raises(FeatureGcgError) as exc_info:
            partition_bounds_from_model(
                model,
                FeatureGcgConfig(),
                cluster_contract=contract,
            )

        assert (
            str(exc_info.value)
            == "feature_gcg partition checkpoint is missing mapping metadata"
        )

    def test_clustered_bounds_match_unclustered_bounds_exactly(
        self,
        tmp_path,
    ):
        contract, model, _ = _contract_model_pair(tmp_path)
        cfg = FeatureGcgConfig(
            min_k_pos=1,
            max_k_pos=7,
            min_k_neg=2,
            max_k_neg=8,
        )
        clustered = partition_bounds_from_model(
            model,
            cfg,
            cluster_contract=contract,
        )
        model._loaded_meta_data = {
            "user_config": {
                "negative_selection": {
                    "distribution": "geometric",
                    "mean": 1.0,
                }
            }
        }
        unclustered = partition_bounds_from_model(model, cfg)

        assert clustered == unclustered

    @pytest.mark.parametrize(
        ("case", "match"),
        [
            ("width", "original SAE width"),
            ("buckets", "bucket count"),
            ("vocab", "in-memory FeatureVocab"),
            ("missing_consistency", "does not expose assert_vocab_consistent"),
            ("failing_consistency", "inconsistent persistent FeatureVocab buffers"),
        ],
    )
    def test_clustered_loaded_state_mismatch_is_wrapped_with_cause(
        self,
        tmp_path,
        case,
        match,
    ):
        contract, model, _ = _contract_model_pair(tmp_path)
        if case == "width":
            model.encoder.config.n_sae_features += 1
        elif case == "buckets":
            model.encoder.config.num_buckets += 1
        elif case == "vocab":
            model.feature_vocab = FeatureVocab(4, (0, 3))
        elif case == "missing_consistency":
            del model.assert_vocab_consistent
        elif case == "failing_consistency":

            def _fail_consistency():
                raise FeatureVocabError("sentinel buffer mismatch")

            model.assert_vocab_consistent = _fail_consistency
        else:
            raise AssertionError(case)

        with pytest.raises(FeatureGcgError, match=match) as exc_info:
            partition_bounds_from_model(
                model,
                FeatureGcgConfig(),
                cluster_contract=contract,
            )
        assert isinstance(exc_info.value.__cause__, ClusterFeatureContractError)

    @pytest.mark.parametrize(
        ("model", "expected_type", "expected_message"),
        [
            (
                SimpleNamespace(
                    encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=8))
                ),
                FeatureGcgError,
                "Model lacks _loaded_meta_data attribute. Attach meta_data to the "
                "model after calling build_encdec_model(): "
                "`model._loaded_meta_data = meta_data`.",
            ),
            (
                SimpleNamespace(
                    encoder=SimpleNamespace(config=SimpleNamespace()),
                    _loaded_meta_data={"user_config": {}},
                ),
                FeatureGcgError,
                "Cannot read num_buckets from model.encoder.config: "
                "'types.SimpleNamespace' object has no attribute 'num_buckets'. "
                "Is this an encoder-decoder model with a loaded encoder?",
            ),
        ],
    )
    def test_none_contract_keeps_error_type_and_message_exact(
        self,
        model,
        expected_type,
        expected_message,
    ):
        def _call(*, explicit_none):
            kwargs = {"cluster_contract": None} if explicit_none else {}
            with pytest.raises(BaseException) as exc_info:
                partition_bounds_from_model(
                    model,
                    FeatureGcgConfig(),
                    **kwargs,
                )
            return exc_info.value

        omitted = _call(explicit_none=False)
        explicit = _call(explicit_none=True)
        for actual in (omitted, explicit):
            assert type(actual) is expected_type
            assert str(actual) == expected_message

    def test_clustered_validation_reads_real_loaded_state_without_io(
        self,
        tmp_path,
        monkeypatch,
    ):
        contract, model, _ = _contract_model_pair(tmp_path)
        calls = 0

        def _consistent():
            nonlocal calls
            calls += 1

        model.assert_vocab_consistent = _consistent
        monkeypatch.setattr(
            t,
            "load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("partition validation unexpectedly touched disk")
            ),
        )

        partition_bounds_from_model(
            model,
            FeatureGcgConfig(),
            cluster_contract=contract,
        )

        assert calls == 1
