"""Tests for label dropout and NaN prevention in precomputed_data_loader.

Covers:
- dropout=0.0 preserves all features in enc_mask
- dropout=1.0 NaN-prevention guard ensures every row has >= 1 feature
- dropout=0.5 statistical behavior across multiple batches
- NaN prevention guarantees at least one feature per row when dropout=1.0
"""

import json

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
)
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_threshold_pipeline,
    make_precomputed_dataloader,
)
from tests.steering.test_precomputed_loader_fused_path import (
    _make_sampling_pipeline_for_test,
    _write_shard_dir,
)


def _stochastic_pipeline(seed: int):
    t.manual_seed(123456)
    pipeline = _make_sampling_pipeline_for_test(
        mean=3.0,
        min_k=1,
        max_k=5,
        seed=seed,
    )
    assert isinstance(pipeline.selection_config, SamplingBundle)
    pipeline.negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
        ),
        n_sae_features=200,
        rng=pipeline.selection_config.rng,
    )
    return pipeline


def _threshold_pipeline():
    boundaries = (
        t.linspace(
            1 / NUM_BUCKETS,
            1.0 - 1 / NUM_BUCKETS,
            NUM_BUCKETS - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _assert_payloads_equal(actual: tuple, expected: tuple) -> None:
    for payload_idx, (actual_value, expected_value) in enumerate(
        zip(actual[:9], expected[:9], strict=True)
    ):
        if expected_value is None:
            assert actual_value is None
            continue
        assert actual_value is not None
        assert actual_value.dtype == expected_value.dtype, payload_idx
        assert actual_value.shape == expected_value.shape, payload_idx
        assert actual_value.stride() == expected_value.stride(), payload_idx
        assert t.equal(actual_value, expected_value), payload_idx


def _assert_concatenated_payloads_equal(parts: list[tuple], expected: tuple) -> None:
    for payload_idx, expected_value in enumerate(expected[:9]):
        actual_values = [part[payload_idx] for part in parts]
        if expected_value is None:
            assert all(value is None for value in actual_values)
            continue
        assert all(value is not None for value in actual_values)
        assert t.equal(t.cat(actual_values, dim=0), expected_value), payload_idx


class TestLabelDropout:
    """Label dropout and NaN prevention in the precomputed data loader."""

    def test_zero_dropout_preserves_all_features(self, tmp_path):
        """With dropout=0.0, enc_mask should contain real features (not all False)."""
        loader = make_precomputed_dataloader(tmp_path, B=4, n_rows=16, label_dropout=0.0)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(loader)

        # enc_mask is (B, n_seg, F_max) dtype bool
        assert enc_mask.dtype == t.bool
        # With zero dropout, features are fully preserved — at least some should be True
        assert enc_mask.any(), "Expected features in enc_mask with dropout=0.0 but got all False"
        # Every row should have at least one active feature
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), (
            "With dropout=0.0 all rows should have at least one active feature"
        )

    def test_full_dropout_masks_most_features(self, tmp_path):
        """With dropout=1.0, NaN-prevention ensures every row still has >= 1 feature.

        At dropout=1.0 every segment is dropped. The NaN guard then sets
        enc_mask[row, 0, 0] = True for each empty row, so total True count == B.
        """
        B = 4
        loader = make_precomputed_dataloader(tmp_path, B=B, n_rows=16, label_dropout=1.0)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = next(loader)

        assert enc_mask.dtype == t.bool
        # Total True positions should equal exactly B (one sentinel per row)
        total_true = enc_mask.sum().item()
        assert total_true == B, (
            f"Expected exactly {B} True values (one NaN-guard sentinel per row) "
            f"with dropout=1.0, got {total_true}"
        )
        # Every row must have at least one True (NaN prevention)
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), "Some rows have zero features after dropout=1.0 — NaN guard failed"

    def test_partial_dropout_statistical_behavior(self, tmp_path):
        """With dropout=0.5, the observed dropout rate across batches should be 0.2–0.8.

        Collects stats over 10 batches (B=4, n_rows=16). The fraction of segments
        dropped (relative to the no-dropout baseline) should be within this range.
        """
        B = 4
        n_rows = 16
        n_batches = 10

        # Baseline: count total active mask entries at dropout=0
        loader_base = make_precomputed_dataloader(
            tmp_path / "base", B=B, n_rows=n_rows, label_dropout=0.0
        )
        baseline_counts = []
        for _ in range(n_batches):
            _, _, _, _, enc_mask, *_ = next(loader_base)
            baseline_counts.append(enc_mask.sum().item())
        baseline_total = sum(baseline_counts)

        # With dropout=0.5, count active features
        loader_drop = make_precomputed_dataloader(
            tmp_path / "drop", B=B, n_rows=n_rows, label_dropout=0.5
        )
        drop_counts = []
        for _ in range(n_batches):
            _, _, _, _, enc_mask, *_ = next(loader_drop)
            drop_counts.append(enc_mask.sum().item())
        drop_total = sum(drop_counts)

        # How much was kept relative to baseline
        if baseline_total > 0:
            kept_fraction = drop_total / baseline_total
            # Expected: ~50% kept. Allow wide range 0.2–0.8 to avoid flakiness.
            assert 0.2 <= kept_fraction <= 0.8, (
                f"Partial dropout=0.5 kept {kept_fraction:.2%} of features "
                f"(expected 0.20–0.80). drop_total={drop_total}, baseline_total={baseline_total}"
            )

    def test_nan_prevention_ensures_one_feature_per_row(self, tmp_path):
        """With dropout=1.0, every row must have row_has_any=True across 5 batches."""
        B = 4
        n_rows = 16
        loader = make_precomputed_dataloader(tmp_path, B=B, n_rows=n_rows, label_dropout=1.0)

        for batch_idx in range(5):
            _, _, _, _, enc_mask, *_ = next(loader)
            # Check NaN prevention: every row has at least one True feature
            row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
            assert row_has_any.all(), (
                f"Batch {batch_idx}: {(~row_has_any).sum().item()} row(s) have zero features "
                f"after dropout=1.0 — NaN prevention failed"
            )

    def test_resume_restores_label_dropout_and_all_selection_rngs(self, tmp_path):
        """F3: replay and continuation match after both RNG streams are perturbed."""
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=16, feats_per_seg=12, seed=73
        )
        baseline_pipeline = _stochastic_pipeline(seed=991)
        t.manual_seed(8801)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=baseline_pipeline,
            label_dropout=0.5,
            pad_F=8,
        )
        baseline = [next(uninterrupted) for _ in range(3)]
        checkpoint_state = json.loads(json.dumps(baseline[1][9]))
        assert checkpoint_state["feature_pipeline_rng_state"]
        assert checkpoint_state["label_dropout_rng_state"]
        expected_pipeline_post = baseline_pipeline.selection_config.rng.get_state()
        expected_dropout_post = t.get_rng_state()

        resumed_pipeline = _stochastic_pipeline(seed=991)
        t.manual_seed(999999)  # prove the checkpoint, not process state, is authoritative
        resumed = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=resumed_pipeline,
            label_dropout=0.5,
            pad_F=8,
            resume_state_dict=checkpoint_state,
        )
        replay = next(resumed)
        continuation = next(resumed)

        _assert_payloads_equal(replay, baseline[1])
        _assert_payloads_equal(continuation, baseline[2])
        assert t.equal(
            resumed_pipeline.selection_config.rng.get_state(),
            expected_pipeline_post,
        )
        assert t.equal(t.get_rng_state(), expected_dropout_post)

    @pytest.mark.parametrize("new_batch_size", [2, 8])
    def test_device_batch_migration_preserves_rows_and_dropout_masks(
        self,
        tmp_path,
        new_batch_size,
    ):
        t.manual_seed(4401)
        baseline = make_precomputed_dataloader(
            tmp_path / "baseline",
            B=4,
            n_rows=16,
            label_dropout=0.5,
        )
        reference = [next(baseline), next(baseline)]
        state = json.loads(json.dumps(reference[0][9]))
        state["device_batch_size"] = 4

        t.manual_seed(999999)
        make_precomputed_dataloader(
            tmp_path / "resumed",
            B=new_batch_size,
            n_rows=16,
            label_dropout=0.5,
            seed=42,
        )
        resumed = precomputed_data_loader(
            packed_row_dir=tmp_path / "resumed" / "packed",
            B=new_batch_size,
            device="cpu",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )

        if new_batch_size == 2:
            migrated = [next(resumed), next(resumed)]
            _assert_concatenated_payloads_equal(migrated, reference[0])
            second_state = json.loads(json.dumps(migrated[1][9]))
            replay = precomputed_data_loader(
                packed_row_dir=tmp_path / "resumed" / "packed",
                B=2,
                device="cpu",
                feature_pipeline=_threshold_pipeline(),
                label_dropout=0.5,
                n_sae_features=N_SAE_FEATURES,
                resume_state_dict=second_state,
            )
            _assert_payloads_equal(next(replay), migrated[1])
        else:
            migrated = next(resumed)
            expected = tuple(
                (
                    None
                    if reference[0][idx] is None
                    else t.cat([reference[0][idx], reference[1][idx]], dim=0)
                )
                for idx in range(9)
            )
            _assert_payloads_equal(migrated, expected)

    def test_nondivisor_and_second_generation_migration_preserve_dropout_stream(
        self,
        tmp_path,
    ):
        t.manual_seed(4401)
        baseline = make_precomputed_dataloader(
            tmp_path / "baseline",
            B=4,
            n_rows=32,
            label_dropout=0.5,
        )
        reference = [next(baseline) for _ in range(3)]
        state = json.loads(json.dumps(reference[0][9]))
        state["device_batch_size"] = 4

        make_precomputed_dataloader(
            tmp_path / "resumed",
            B=3,
            n_rows=32,
            label_dropout=0.5,
            seed=42,
        )
        first_migration = precomputed_data_loader(
            packed_row_dir=tmp_path / "resumed" / "packed",
            B=3,
            device="cpu",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        b3_batches = [next(first_migration) for _ in range(4)]
        _assert_concatenated_payloads_equal(b3_batches, tuple(
            None
            if reference[0][idx] is None
            else t.cat([batch[idx] for batch in reference], dim=0)
            for idx in range(9)
        ))
        assert [batch[9]["label_dropout_draw_offset"] for batch in b3_batches] == [
            0,
            3,
            2,
            1,
        ]

        second_state = json.loads(json.dumps(b3_batches[2][9]))
        second_state["device_batch_size"] = 3
        second_migration = precomputed_data_loader(
            packed_row_dir=tmp_path / "resumed" / "packed",
            B=5,
            device="cpu",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=second_state,
        )
        b5_batches = [next(second_migration) for _ in range(3)]
        uninterrupted_tail = [
            b3_batches[2],
            b3_batches[3],
            next(first_migration),
            next(first_migration),
            next(first_migration),
        ]
        for payload_idx, actual in enumerate(b5_batches[0][:9]):
            if actual is None:
                assert all(batch[payload_idx] is None for batch in uninterrupted_tail)
                continue
            assert t.equal(
                t.cat([batch[payload_idx] for batch in b5_batches], dim=0),
                t.cat([batch[payload_idx] for batch in uninterrupted_tail], dim=0)[:15],
            ), payload_idx

        replay_state = json.loads(json.dumps(b5_batches[1][9]))
        replay_state["device_batch_size"] = 5
        replay = precomputed_data_loader(
            packed_row_dir=tmp_path / "resumed" / "packed",
            B=5,
            device="cpu",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=replay_state,
        )
        _assert_payloads_equal(next(replay), b5_batches[1])

    @pytest.mark.parametrize(
        ("mutation", "error_type", "error"),
        [
            ("missing_rng", ValueError, "requires checkpointed label_dropout_rng_state"),
            ("draw_width_zero", BeartypeCallHintParamViolation, "draw_batch_size"),
            ("draw_width_bool", BeartypeCallHintParamViolation, "draw_batch_size"),
            ("draw_offset_negative", BeartypeCallHintParamViolation, "draw_offset"),
            ("draw_offset_equal_width", ValueError, "draw_offset must be a strict int"),
        ],
    )
    def test_device_batch_migration_rejects_invalid_canonical_dropout_state(
        self,
        tmp_path,
        mutation,
        error_type,
        error,
    ):
        t.manual_seed(90210)
        baseline = make_precomputed_dataloader(
            tmp_path,
            B=4,
            n_rows=16,
            label_dropout=0.5,
        )
        state = json.loads(json.dumps(next(baseline)[9]))
        state["device_batch_size"] = 4
        if mutation == "missing_rng":
            del state["label_dropout_rng_state"]
        elif mutation == "draw_width_zero":
            state["label_dropout_draw_batch_size"] = 0
        elif mutation == "draw_width_bool":
            state["label_dropout_draw_batch_size"] = True
        elif mutation == "draw_offset_negative":
            state["label_dropout_draw_offset"] = -1
        elif mutation == "draw_offset_equal_width":
            state["label_dropout_draw_offset"] = 4
        else:
            raise AssertionError(mutation)
        resumed = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=2,
            device="cpu",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        with pytest.raises(error_type, match=error):
            next(resumed)

    def test_stochastic_cross_epoch_resume_preserves_slow_path_draw_order(
        self, tmp_path
    ):
        """Exact replay keeps epoch-qualified shard instances distinct."""
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=3, feats_per_seg=12, seed=78
        )
        baseline_pipeline = _stochastic_pipeline(seed=996)
        t.manual_seed(8806)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=baseline_pipeline,
            label_dropout=0.5,
            pad_F=8,
        )
        baseline = [next(uninterrupted) for _ in range(3)]
        state = json.loads(json.dumps(baseline[0][9]))
        assert state["batch_rows"][0][0] == 1
        assert state["batch_rows"][-1][0] == 2
        expected_pipeline_post = baseline_pipeline.selection_config.rng.get_state()
        expected_dropout_post = t.get_rng_state()

        resumed_pipeline = _stochastic_pipeline(seed=996)
        t.manual_seed(999999)
        resumed = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=resumed_pipeline,
            label_dropout=0.5,
            pad_F=8,
            resume_state_dict=state,
        )
        for expected in baseline:
            _assert_payloads_equal(next(resumed), expected)
        assert t.equal(
            resumed_pipeline.selection_config.rng.get_state(),
            expected_pipeline_post,
        )
        assert t.equal(t.get_rng_state(), expected_dropout_post)

    def test_resume_restores_distinct_selection_and_negative_generators(
        self, tmp_path
    ):
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=12, feats_per_seg=12, seed=79
        )

        def distinct_pipeline():
            pipeline = _stochastic_pipeline(seed=997)
            assert pipeline.negative_bundle is not None
            pipeline.negative_bundle.rng = t.Generator().manual_seed(1997)
            return pipeline

        baseline_pipeline = distinct_pipeline()
        uninterrupted = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=baseline_pipeline,
            label_dropout=0.0,
            pad_F=8,
        )
        baseline = [next(uninterrupted) for _ in range(3)]
        state = json.loads(json.dumps(baseline[1][9]))
        assert [entry["roles"] for entry in state["feature_pipeline_rng_state"]] == [
            ["selection"],
            ["negative"],
        ]
        selection_post = baseline_pipeline.selection_config.rng.get_state()
        assert baseline_pipeline.negative_bundle is not None
        negative_post = baseline_pipeline.negative_bundle.rng.get_state()

        resumed_pipeline = distinct_pipeline()
        t.rand(100, generator=resumed_pipeline.selection_config.rng)
        assert resumed_pipeline.negative_bundle is not None
        t.rand(100, generator=resumed_pipeline.negative_bundle.rng)
        resumed = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=resumed_pipeline,
            label_dropout=0.0,
            pad_F=8,
            resume_state_dict=state,
        )
        _assert_payloads_equal(next(resumed), baseline[1])
        _assert_payloads_equal(next(resumed), baseline[2])
        assert t.equal(
            resumed_pipeline.selection_config.rng.get_state(), selection_post
        )
        assert t.equal(resumed_pipeline.negative_bundle.rng.get_state(), negative_post)

    def test_zero_dropout_fresh_path_does_not_touch_default_rng(self, tmp_path):
        """I5: capture metadata is read-only and dropout=0 remains an RNG no-op."""
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=8, feats_per_seg=12, seed=74
        )
        pipeline = _stochastic_pipeline(seed=992)
        loader = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            label_dropout=0.0,
            pad_F=8,
        )
        t.manual_seed(8802)
        before = t.get_rng_state().clone()
        batch = next(loader)
        after = t.get_rng_state()

        assert t.equal(after, before)
        assert "label_dropout_rng_state" not in batch[9]

    def test_positive_dropout_fresh_path_matches_prechange_draw_order(self, tmp_path):
        """I5: state capture adds no draw and preserves the default RNG post-state."""
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=8, feats_per_seg=12, seed=75
        )
        no_dropout_pipeline = _stochastic_pipeline(seed=993)
        dropout_pipeline = _stochastic_pipeline(seed=993)

        base = next(
            precomputed_data_loader(
                packed_row_dir=packed,
                B=4,
                device="cpu",
                feature_pipeline=no_dropout_pipeline,
                label_dropout=0.0,
                pad_F=8,
            )
        )

        t.manual_seed(8803)
        initial_state = t.get_rng_state().clone()
        expected_mask = base[4].clone()
        drop_mask = t.rand(4, expected_mask.shape[1]) < 0.5
        expected_mask[drop_mask] = False
        empty_rows = ~expected_mask.any(dim=-1).any(dim=-1)
        expected_mask[empty_rows, 0, 0] = True
        expected_post_state = t.get_rng_state().clone()

        t.manual_seed(8803)
        actual = next(
            precomputed_data_loader(
                packed_row_dir=packed,
                B=4,
                device="cpu",
                feature_pipeline=dropout_pipeline,
                label_dropout=0.5,
                pad_F=8,
            )
        )
        expected = list(base)
        expected[4] = expected_mask
        _assert_payloads_equal(actual, tuple(expected))
        assert actual[9]["label_dropout_rng_state"]["state_hex"] == bytes(
            initial_state.tolist()
        ).hex()
        assert t.equal(t.get_rng_state(), expected_post_state)
        assert t.equal(
            dropout_pipeline.selection_config.rng.get_state(),
            no_dropout_pipeline.selection_config.rng.get_state(),
        )

    def test_malformed_rng_metadata_raises_and_legacy_missing_fields_loads(
        self, tmp_path
    ):
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=8, feats_per_seg=12, seed=76
        )
        t.manual_seed(8804)
        batch = next(
            precomputed_data_loader(
                packed_row_dir=packed,
                B=4,
                device="cpu",
                feature_pipeline=_stochastic_pipeline(seed=994),
                label_dropout=0.5,
                pad_F=8,
            )
        )
        state = json.loads(json.dumps(batch[9]))

        partial_pipeline = json.loads(json.dumps(state))
        del partial_pipeline["feature_pipeline_rng_state"][0]["state_hex"]
        invalid_hex = json.loads(json.dumps(state))
        invalid_hex["feature_pipeline_rng_state"][0]["state_hex"] = "zz"
        wrong_device = json.loads(json.dumps(state))
        wrong_device["label_dropout_rng_state"]["device_type"] = "cuda"
        wrong_device["label_dropout_rng_state"]["device_index"] = 0

        for malformed in (partial_pipeline, invalid_hex, wrong_device):
            loader = precomputed_data_loader(
                packed_row_dir=packed,
                B=4,
                device="cpu",
                feature_pipeline=_stochastic_pipeline(seed=994),
                label_dropout=0.5,
                pad_F=8,
                resume_state_dict=malformed,
            )
            with pytest.raises(ValueError, match="RNG|state_hex|device"):
                next(loader)

        wrong_length = json.loads(json.dumps(state))
        wrong_length["label_dropout_rng_state"]["state_hex"] = "00"
        with pytest.raises(ValueError, match="RNG state length"):
            next(
                precomputed_data_loader(
                    packed_row_dir=packed,
                    B=4,
                    device="cpu",
                    feature_pipeline=_stochastic_pipeline(seed=994),
                    label_dropout=0.0,
                    pad_F=8,
                    resume_state_dict=wrong_length,
                )
            )

        no_rows_but_rng = json.loads(json.dumps(state))
        del no_rows_but_rng["batch_rows"]
        corrupt_no_rows = json.loads(json.dumps(no_rows_but_rng))
        corrupt_no_rows["feature_pipeline_rng_state"][0]["state_hex"] = "zz"
        with pytest.raises(ValueError, match="state_hex"):
            next(
                precomputed_data_loader(
                    packed_row_dir=packed,
                    B=4,
                    device="cpu",
                    feature_pipeline=_stochastic_pipeline(seed=994),
                    label_dropout=0.5,
                    pad_F=8,
                    resume_state_dict=corrupt_no_rows,
                )
            )

        t.manual_seed(999999)
        partial_new_replay = next(
            precomputed_data_loader(
                packed_row_dir=packed,
                B=4,
                device="cpu",
                feature_pipeline=_stochastic_pipeline(seed=994),
                label_dropout=0.5,
                pad_F=8,
                resume_state_dict=no_rows_but_rng,
            )
        )
        _assert_payloads_equal(partial_new_replay, batch)

        legacy = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "batch_rows",
                "ddp_world_size",
                "feature_pipeline_rng_state",
                "label_dropout_rng_state",
            }
        }
        legacy_loader = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cpu",
            feature_pipeline=_stochastic_pipeline(seed=994),
            label_dropout=0.5,
            pad_F=8,
            resume_state_dict=legacy,
        )
        assert len(next(legacy_loader)) == 10

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
    def test_cuda_resume_uses_and_restores_cuda_default_stream(self, tmp_path):
        packed = _write_shard_dir(
            tmp_path / "data", n_rows=12, feats_per_seg=12, seed=77
        )
        t.cuda.manual_seed_all(8805)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cuda",
            feature_pipeline=_stochastic_pipeline(seed=995),
            label_dropout=0.5,
            pad_F=8,
        )
        expected = [next(uninterrupted) for _ in range(2)]
        state = json.loads(json.dumps(expected[0][9]))
        assert state["label_dropout_rng_state"]["device_type"] == "cuda"
        expected_post_state = t.cuda.get_rng_state().clone()

        t.cuda.manual_seed_all(999999)
        resumed = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cuda",
            feature_pipeline=_stochastic_pipeline(seed=995),
            label_dropout=0.5,
            pad_F=8,
            resume_state_dict=state,
        )
        replay = next(resumed)
        continuation = next(resumed)

        _assert_payloads_equal(replay, expected[0])
        _assert_payloads_equal(continuation, expected[1])
        assert t.equal(t.cuda.get_rng_state(), expected_post_state)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("new_batch_size", [2, 8])
    def test_cuda_device_batch_migration_preserves_dropout_masks(
        self,
        tmp_path,
        new_batch_size,
    ):
        packed = _write_shard_dir(
            tmp_path / f"data-{new_batch_size}",
            n_rows=16,
            feats_per_seg=12,
            seed=177,
        )
        t.cuda.manual_seed_all(18105)
        baseline = precomputed_data_loader(
            packed_row_dir=packed,
            B=4,
            device="cuda",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            pad_F=8,
        )
        expected = [next(baseline) for _ in range(2)]
        state = json.loads(json.dumps(expected[0][9]))
        state["device_batch_size"] = 4

        t.cuda.manual_seed_all(999999)
        resumed = precomputed_data_loader(
            packed_row_dir=packed,
            B=new_batch_size,
            device="cuda",
            feature_pipeline=_threshold_pipeline(),
            label_dropout=0.5,
            n_sae_features=N_SAE_FEATURES,
            pad_F=8,
            resume_state_dict=state,
        )
        if new_batch_size == 2:
            migrated = [next(resumed), next(resumed)]
            _assert_concatenated_payloads_equal(migrated, expected[0])
            replay_state = json.loads(json.dumps(migrated[1][9]))
            replay_state["device_batch_size"] = 2
            replay = precomputed_data_loader(
                packed_row_dir=packed,
                B=2,
                device="cuda",
                feature_pipeline=_threshold_pipeline(),
                label_dropout=0.5,
                n_sae_features=N_SAE_FEATURES,
                pad_F=8,
                resume_state_dict=replay_state,
            )
            _assert_payloads_equal(next(replay), migrated[1])
        else:
            migrated = next(resumed)
            concatenated = tuple(
                None
                if expected[0][idx] is None
                else t.cat([batch[idx] for batch in expected], dim=0)
                for idx in range(9)
            )
            _assert_payloads_equal(migrated, concatenated)
