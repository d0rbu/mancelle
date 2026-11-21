"""Tests for cross-shard batch boundary reconstruction.

Verifies that batches spanning multiple packed-row shards are reconstructed
correctly — shapes, mask validity, and continuation across shard boundaries.
"""

import json
from pathlib import Path

import pytest
import torch as t

from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard, save_selection_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    _make_identity_selection,
    _make_test_shard,
    build_threshold_pipeline,
)


def _assert_payloads_equal(actual: tuple, expected: tuple) -> None:
    """Compare all nine training payloads, including metadata-relevant layout."""
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


def _assert_concatenated_payloads_equal(parts: list[tuple], expected: list[tuple]) -> None:
    for payload_idx in range(9):
        actual_values = [part[payload_idx] for part in parts]
        expected_values = [part[payload_idx] for part in expected]
        if expected_values[0] is None:
            assert all(value is None for value in actual_values)
            assert all(value is None for value in expected_values)
            continue
        assert all(value is not None for value in actual_values)
        assert all(value is not None for value in expected_values)
        assert t.equal(
            t.cat(actual_values, dim=0),
            t.cat(expected_values, dim=0),
        ), payload_idx


def _build_pipeline():
    """Build a standard threshold pipeline with NUM_BUCKETS evenly spaced boundaries."""
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


def _write_two_shards(shard_dir: Path, n_rows: int, seed_a: int = 42, seed_b: int = 99) -> None:
    """Write two packed-row shards to shard_dir and create a manifest.json."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_a = _make_test_shard(n_rows=n_rows, seed=seed_a)
    shard_b = _make_test_shard(n_rows=n_rows, seed=seed_b)
    save_packed_row_shard(shard_a, shard_dir / "shard_00000")
    save_packed_row_shard(shard_b, shard_dir / "shard_00001")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))


def _write_selection_manifest(sel_dir: Path, shard_dir: Path) -> None:
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(shard_dir),
            },
        },
    )


class TestCrossShardBatches:
    """Batches that span multiple shard boundaries reconstruct correctly."""

    def test_cross_shard_batch_shapes(self, tmp_path):
        """2 shards with 3 rows each, B=4: batch should have shape (4, SEQ_LEN)."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=3)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        inputs, targets, *_ = next(loader)

        assert inputs.shape == (4, SEQ_LEN), (
            f"Expected inputs shape (4, {SEQ_LEN}), got {inputs.shape}"
        )
        assert targets.shape == (4, SEQ_LEN), (
            f"Expected targets shape (4, {SEQ_LEN}), got {targets.shape}"
        )

    def test_cross_shard_mask_validity(self, tmp_path):
        """2 shards (2 rows each), B=3: every row has features AND fp_targets
        shape matches enc_mask shape."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=2)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        (
            _inputs, _targets, _enc_fids, _enc_mags, enc_mask, _enc_pcts,
            _seg_map, _seg_doc_id, fp_targets, _state,
        ) = next(loader)

        # enc_mask has shape (B, n_seg, F) — check every row has at least one True
        # (NaN guard guarantees this even when label_dropout=0)
        row_has_feature = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_feature.all(), (
            f"Some rows have no features after cross-shard reconstruction: "
            f"{row_has_feature.tolist()}"
        )

        # fp_targets shape must be coherent with enc_mask
        assert fp_targets is not None
        assert fp_targets.shape[:2] == enc_mask.shape[:2]
        assert fp_targets.shape[2] == N_SAE_FEATURES

    def test_second_batch_continues_correctly(self, tmp_path):
        """2 shards (3 rows each), B=2: verify the second batch is valid and
        advances state correctly. This pins that the loader continues to
        produce well-formed batches past the first one (state tracking,
        epoch/shard advance, no stale state)."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=3, seed_a=7, seed_b=13)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        # First batch (within shard A)
        b1 = next(loader)
        # Second batch — may or may not span shards depending on row layout
        b2 = next(loader)

        inputs1, _t1, _ef1, _em1, em1, _ep1, _sm1, _sd1, fp1, state1 = b1
        inputs2, _t2, _ef2, _em2, em2, _ep2, _sm2, _sd2, fp2, state2 = b2

        # Both batches have full shape
        assert inputs1.shape[0] == 2
        assert inputs2.shape[0] == 2

        # Both have fp_targets coherent with enc_mask
        assert fp1 is not None and fp2 is not None
        assert fp1.shape[:2] == em1.shape[:2]
        assert fp2.shape[:2] == em2.shape[:2]
        assert fp1.shape[2] == N_SAE_FEATURES
        assert fp2.shape[2] == N_SAE_FEATURES

        # State actually advanced
        assert state1["batch_idx"] == 0
        assert state2["batch_idx"] == 1, (
            f"second batch_idx should be 1, got {state2['batch_idx']}"
        )

        # The two batches produced different content (rows differ across batches)
        assert not t.equal(inputs1, inputs2), (
            "consecutive batches produced identical inputs — state not advancing"
        )

    def test_resume_reconstructs_cross_shard_pending_rows_exactly(self, tmp_path):
        """F2: ordered row identities reproduce a boundary-spanning batch."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=13, seed_a=17, seed_b=23)

        uninterrupted = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        baseline = [next(uninterrupted) for _ in range(6)]
        checkpoint = baseline[3]
        assert checkpoint[9]["batch_rows"] == [
            [1, "shard_00000.safetensors.zst", 12],
            [1, "shard_00001.safetensors.zst", 0],
            [1, "shard_00001.safetensors.zst", 1],
            [1, "shard_00001.safetensors.zst", 2],
        ]

        state = json.loads(json.dumps(checkpoint[9]))
        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        replay = [next(resumed) for _ in range(3)]
        for actual, expected in zip(replay, baseline[3:6], strict=True):
            _assert_payloads_equal(actual, expected)

    def test_exact_resume_pending_rows_bypass_shared_owner_helper(
        self,
        tmp_path,
        monkeypatch,
    ):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=13, seed_a=17, seed_b=23)
        baseline_loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        baseline = [next(baseline_loader) for _ in range(4)]
        state = json.loads(json.dumps(baseline[3][9]))
        assert len({row[1] for row in state["batch_rows"]}) == 2

        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        monkeypatch.setattr(
            dataloader_module,
            "_shared_decode_load_owner_shard",
            lambda _path: pytest.fail("exact resume invoked the owner helper"),
        )
        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        _assert_payloads_equal(next(resumed), baseline[3])

    def test_single_process_local_loading_stays_canonical(
        self,
        tmp_path,
        monkeypatch,
    ):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=4)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
        monkeypatch.setattr(
            dataloader_module,
            "_shared_decode_load_owner_shard",
            lambda _path: pytest.fail("single-process loading invoked the owner helper"),
        )
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        assert next(loader)[0].shape == (4, SEQ_LEN)

    @pytest.mark.parametrize("new_batch_size", [2, 8])
    @pytest.mark.parametrize("lookahead", [0, 3])
    def test_device_batch_migration_preserves_cross_shard_row_stream(
        self,
        tmp_path,
        new_batch_size,
        lookahead,
    ):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=13, seed_a=17, seed_b=23)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        baseline = [next(uninterrupted) for _ in range(6)]
        uninterrupted.close()
        state = json.loads(json.dumps(baseline[3][9]))
        state["device_batch_size"] = 4
        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=new_batch_size,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
            train_shard_lookahead=lookahead,
        )

        try:
            if new_batch_size == 2:
                parts = [next(resumed), next(resumed)]
                for payload_idx, expected in enumerate(baseline[3][:9]):
                    if expected is None:
                        assert all(part[payload_idx] is None for part in parts)
                    else:
                        assert t.equal(
                            t.cat([part[payload_idx] for part in parts], dim=0),
                            expected,
                        )
            else:
                migrated = next(resumed)
                for payload_idx, actual in enumerate(migrated[:9]):
                    expected_parts = [baseline[3][payload_idx], baseline[4][payload_idx]]
                    if actual is None:
                        assert expected_parts == [None, None]
                    else:
                        assert t.equal(actual, t.cat(expected_parts, dim=0))
        finally:
            resumed.close()

    @pytest.mark.parametrize(
        ("old_batch_size", "new_batch_size", "old_batches", "new_batches"),
        [
            (6, 4, 2, 3),
            (6, 8, 4, 3),
            (16, 8, 1, 2),
            (16, 32, 2, 1),
        ],
    )
    def test_device_batch_migration_preserves_nondivisor_and_production_row_groups(
        self,
        tmp_path,
        old_batch_size,
        new_batch_size,
        old_batches,
        new_batches,
    ):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=37, seed_a=101, seed_b=103)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=old_batch_size,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        baseline = [next(uninterrupted) for _ in range(old_batches)]
        state = json.loads(json.dumps(baseline[0][9]))
        state["device_batch_size"] = old_batch_size
        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=new_batch_size,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        migrated = [next(resumed) for _ in range(new_batches)]

        _assert_concatenated_payloads_equal(migrated, baseline)
        actual_rows = [row for batch in migrated for row in batch[9]["batch_rows"]]
        expected_rows = [row for batch in baseline for row in batch[9]["batch_rows"]]
        assert actual_rows == expected_rows

    @pytest.mark.parametrize(
        ("mutation", "error"),
        [
            ("missing_topology", "requires checkpointed ddp_world_size"),
            ("missing_saved_batch", "requires checkpointed device_batch_size"),
            ("zero_saved_batch", "positive strict int"),
            ("bool_saved_batch", "positive strict int"),
            ("row_count_mismatch", "length must equal"),
            ("missing_rows", "requires exact batch_rows"),
        ],
    )
    def test_device_batch_migration_rejects_incomplete_header_before_artifact_access(
        self,
        tmp_path,
        mutation,
        error,
    ):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=8)
        baseline = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        state = json.loads(json.dumps(next(baseline)[9]))
        state["device_batch_size"] = 4
        if mutation == "missing_topology":
            del state["ddp_world_size"]
        elif mutation == "missing_saved_batch":
            del state["device_batch_size"]
        elif mutation == "zero_saved_batch":
            state["device_batch_size"] = 0
        elif mutation == "bool_saved_batch":
            state["device_batch_size"] = True
        elif mutation == "row_count_mismatch":
            state["device_batch_size"] = 3
        elif mutation == "missing_rows":
            del state["batch_rows"]
        else:
            raise AssertionError(mutation)
        for path in shard_dir.iterdir():
            path.unlink()
        shard_dir.rmdir()

        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=state,
        )
        with pytest.raises(ValueError, match=error):
            next(resumed)

    def test_resume_reconstructs_cross_epoch_pending_rows_exactly(self, tmp_path):
        """F2: epochs qualify otherwise-identical row references."""
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True)
        shard = _make_test_shard(n_rows=3, seed=31)
        save_packed_row_shard(shard, shard_dir / "shard_00000")
        (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        uninterrupted = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
        )
        baseline = [next(uninterrupted) for _ in range(3)]
        state = baseline[0][9]
        assert state["batch_rows"] == [
            [1, "shard_00000.safetensors.zst", 0],
            [1, "shard_00000.safetensors.zst", 1],
            [1, "shard_00000.safetensors.zst", 2],
            [2, "shard_00000.safetensors.zst", 0],
        ]

        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            resume_state_dict=json.loads(json.dumps(state)),
        )
        for expected in baseline:
            _assert_payloads_equal(next(resumed), expected)

    def test_malformed_exact_batch_rows_raise_before_yield(self, tmp_path):
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=5)
        source = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
        )
        state = next(source)[9]
        assert "batch_rows" in state

        corruptions = []
        out_of_range = json.loads(json.dumps(state))
        out_of_range["batch_rows"][0][2] = 999
        corruptions.append((out_of_range, "row.*bounds|out of range"))
        wrong_rank = json.loads(json.dumps(state))
        wrong_rank["ddp_world_size"] = 2
        wrong_rank["batch_rows"][0][2] = 1
        corruptions.append((wrong_rank, "world.size|strip"))
        reordered = json.loads(json.dumps(state))
        reordered["batch_rows"][0], reordered["batch_rows"][1] = (
            reordered["batch_rows"][1],
            reordered["batch_rows"][0],
        )
        corruptions.append((reordered, "ordered|consecutive"))
        missing = json.loads(json.dumps(state))
        missing["batch_rows"][0][1] = "missing.safetensors.zst"
        corruptions.append((missing, "inventory|missing"))

        for corrupt, message in corruptions:
            loader = precomputed_data_loader(
                packed_row_dir=shard_dir,
                B=4,
                device="cpu",
                feature_pipeline=_build_pipeline(),
                resume_state_dict=corrupt,
            )
            with pytest.raises(ValueError, match=message):
                next(loader)

        rank_zero_source = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            ddp_rank=0,
            ddp_world_size=2,
        )
        wrong_rank_same_topology = json.loads(json.dumps(next(rank_zero_source)[9]))
        wrong_rank_same_topology["batch_rows"][0][2] = 1
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            ddp_rank=0,
            ddp_world_size=2,
            resume_state_dict=wrong_rank_same_topology,
        )
        with pytest.raises(ValueError, match="does not belong to rank"):
            next(loader)


def _write_variable_seg_shards(
    shard_dir: Path,
    segs_a: list[int],
    segs_b: list[int],
    seed_a: int = 42,
    seed_b: int = 99,
) -> None:
    """Write two shards with per-row variable segment counts.

    Produces the heterogeneous n_seg scenario that triggered the
    production crash before the fp_targets padding fix.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_a = _make_test_shard(n_rows=len(segs_a), segs_per_row=segs_a, seed=seed_a)
    shard_b = _make_test_shard(n_rows=len(segs_b), segs_per_row=segs_b, seed=seed_b)
    save_packed_row_shard(shard_a, shard_dir / "shard_00000")
    save_packed_row_shard(shard_b, shard_dir / "shard_00001")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))


class TestCrossShardFpTargets:
    """fp_targets coherence and correctness across shard boundaries."""

    def test_cross_shard_fp_targets_shape_matches_enc_mask(self, tmp_path):
        """fp_targets.shape[:2] == enc_mask.shape[:2] for cross-shard batches."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=2)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        _i, _t, _ef, _em, enc_mask, _ep, _sm, _sd, fp_targets, _state = next(loader)
        assert fp_targets is not None, "n_sae_features set; fp_targets must not be None"
        assert fp_targets.shape[:2] == enc_mask.shape[:2], (
            f"fp_targets.shape[:2]={fp_targets.shape[:2]} != "
            f"enc_mask.shape[:2]={enc_mask.shape[:2]}"
        )
        assert fp_targets.shape[2] == N_SAE_FEATURES

    def test_cross_shard_variable_n_seg_regression(self, tmp_path):
        """REGRESSION: reproduce the production crash.

        Two shards with heterogeneous per-row segment counts. Pre-fix, the
        cross-shard fp_targets stacking would raise:
          RuntimeError: Sizes of tensors must match except in dimension 0.
          Expected size N but got size M ...
        """
        shard_dir = tmp_path / "packed"
        _write_variable_seg_shards(
            shard_dir,
            segs_a=[2, 4],  # shard A: row 0 has 2 segs, row 1 has 4
            segs_b=[3, 5],  # shard B: row 0 has 3 segs, row 1 has 5
        )
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,  # forces cross-shard: 2 rows from A + 1 from B
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        # Must not crash
        _i, _t, _ef, _em, enc_mask, _ep, _sm, _sd, fp_targets, _state = next(loader)

        assert fp_targets is not None
        # Output max_ns is the max over the rows in this batch:
        # rows from A: 2 and 4 segs; row 0 from B: 3 segs → max = 4
        assert fp_targets.shape == (3, enc_mask.shape[1], N_SAE_FEATURES)
        # Coherence with enc_mask
        assert fp_targets.shape[:2] == enc_mask.shape[:2]

    def test_cross_shard_fp_targets_values_preserved(self, tmp_path):
        """fp_targets values from cross-shard parts match per-row reconstruction.

        Builds the same shards twice, runs the loader once with cross-shard
        batches and once with per-row reconstruction (B=1), and compares the
        fp_targets values for the first row. They must be equal for the
        unpadded slots.
        """
        # Two single-row shards so any batch B >= 2 spans both.
        shard_dir = tmp_path / "packed"
        _write_variable_seg_shards(
            shard_dir,
            segs_a=[3],
            segs_b=[3],
            seed_a=11,
            seed_b=22,
        )
        pipeline = _build_pipeline()

        # Cross-shard B=2 batch
        loader_x = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_x, _ = next(loader_x)

        # Per-row B=1 batches (no cross-shard stacking)
        loader_1 = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=1,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_row_a, _ = next(loader_1)
        *_, fp_row_b, _ = next(loader_1)

        # The cross-shard batch's row 0 must equal the standalone row A
        assert fp_x is not None and fp_row_a is not None and fp_row_b is not None
        # Both rows have n_seg=3, so no padding involved on this comparison.
        assert t.equal(fp_x[0, :3], fp_row_a[0, :3])
        assert t.equal(fp_x[1, :3], fp_row_b[0, :3])

    def test_cross_shard_fp_targets_zero_padding_for_short_rows(self, tmp_path):
        """Rows with fewer segments than batch max have zero-padded slots."""
        shard_dir = tmp_path / "packed"
        # Force a batch with very different per-row n_seg
        _write_variable_seg_shards(
            shard_dir,
            segs_a=[2],  # shard A: single row with 2 segs
            segs_b=[5],  # shard B: single row with 5 segs
        )
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,  # 1 row from A + 1 row from B
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        _i, _t, _ef, _em, enc_mask, _ep, _sm, _sd, fp_targets, _state = next(loader)
        assert fp_targets is not None
        # batch_n_seg max is 5 (from shard B's row)
        assert fp_targets.shape[1] == 5
        # Row 0 (from shard A) had only 2 segments; positions [2:] are zero.
        assert (fp_targets[0, 2:] == 0).all(), (
            "padding positions for short row should be zero"
        )

    def test_cross_shard_none_fp_when_n_sae_not_set(self, tmp_path):
        """n_sae_features=None → fp_targets is None even with cross-shard batches."""
        shard_dir = tmp_path / "packed"
        _write_two_shards(shard_dir, n_rows=2)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=None,
        )
        _i, _t, _ef, _em, _enc_mask, _ep, _sm, _sd, fp_targets, _state = next(loader)
        assert fp_targets is None

    def test_cross_shard_seg_doc_id_shape_matches_enc_mask(self, tmp_path):
        """seg_doc_id shape coherence with enc_mask across cross-shard batches.

        Currently equal by construction in _stack_batch_parts (both use the
        slow-path max_ns), but this regression-prevention assertion locks
        in the (B, n_seg) coherence contract specified in the design doc.
        """
        shard_dir = tmp_path / "packed"
        _write_variable_seg_shards(
            shard_dir,
            segs_a=[2, 4],
            segs_b=[3, 5],
        )
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )

        _i, _t, _ef, _em, enc_mask, _ep, _sm, seg_doc_id, _fp, _state = next(loader)
        assert seg_doc_id.shape == enc_mask.shape[:2], (
            f"seg_doc_id.shape={seg_doc_id.shape} != "
            f"enc_mask.shape[:2]={enc_mask.shape[:2]}"
        )


def _make_ground_truth_shard(
    n_rows: int,
    context_len: int,
    feats_per_seg: int,
    feature_ids: list[list[int]],
    feature_pcts_uint8: list[list[int]],
) -> PackedRowShard:
    """Build a PackedRowShard with KNOWN feature_ids and feature_pcts values.

    Each row has exactly `len(feature_ids[r])` segments, and each segment has
    exactly `feats_per_seg` features. Tokens are deterministic but irrelevant.

    feature_ids[r][s*feats_per_seg + k] is the feature id for row r, segment s,
    feature slot k. Same indexing for feature_pcts_uint8.
    """
    T = context_len
    segs_per_row_list = [len(fids) // feats_per_seg for fids in feature_ids]
    total_segs = sum(segs_per_row_list)
    total_feats = total_segs * feats_per_seg

    # Sanity check: each row's feature_ids and feature_pcts are total_segs * feats_per_seg
    for r, fids in enumerate(feature_ids):
        assert len(fids) == segs_per_row_list[r] * feats_per_seg, (
            f"feature_ids[{r}] length mismatch"
        )
        assert len(feature_pcts_uint8[r]) == segs_per_row_list[r] * feats_per_seg, (
            f"feature_pcts_uint8[{r}] length mismatch"
        )

    tokens = t.zeros((n_rows, T + 1), dtype=t.uint16)
    # Use distinct nonzero tokens so they're valid model inputs
    for r in range(n_rows):
        for j in range(T + 1):
            tokens[r, j] = 1 + ((r * (T + 1) + j) % 50)

    # CSR row offsets
    row_seg_offsets = t.zeros(n_rows + 1, dtype=t.int32)
    cum = 0
    for r in range(n_rows):
        cum += segs_per_row_list[r]
        row_seg_offsets[r + 1] = cum

    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id_arr = t.zeros(total_segs, dtype=t.uint8)
    for r in range(n_rows):
        n_seg_r = segs_per_row_list[r]
        seg_size = T // n_seg_r
        start_idx = int(row_seg_offsets[r].item())
        for s in range(n_seg_r):
            seg_token_start[start_idx + s] = s * seg_size
            seg_token_end[start_idx + s] = T if s == n_seg_r - 1 else (s + 1) * seg_size
            seg_doc_id_arr[start_idx + s] = 0

    seg_feat_offsets = t.arange(
        0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32
    )

    # Flatten feature_ids and feature_pcts in row-major order
    flat_fids: list[int] = []
    flat_pcts: list[int] = []
    for r in range(n_rows):
        flat_fids.extend(feature_ids[r])
        flat_pcts.extend(feature_pcts_uint8[r])
    assert len(flat_fids) == total_feats
    assert len(flat_pcts) == total_feats

    feature_ids_t = t.tensor(flat_fids, dtype=t.uint16)
    feature_pcts_t = t.tensor(flat_pcts, dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_arr,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids_t,
        feature_pcts=feature_pcts_t,
    )


class TestCrossShardFpTargetsGroundTruth:
    """Ground-truth fp_targets value tests.

    The cross-path self-consistency tests (B=1 vs B=2) would mask any shared
    bug in build_fp_targets_from_shard. These tests compute expected values
    directly from known feature_ids / feature_pcts and assert exact equality.
    """

    def test_cross_shard_fp_targets_exact_values(self, tmp_path):
        """fp_targets[b, s, f_id] == feature_pcts_uint8[b, s, f_id] / 255.0."""
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True, exist_ok=True)

        feats_per_seg = 2
        # Shard A: 1 row, 2 segments, fids [3, 7] then [11, 13]; pcts [10, 20] then [30, 40]
        shard_a = _make_ground_truth_shard(
            n_rows=1,
            context_len=SEQ_LEN,
            feats_per_seg=feats_per_seg,
            feature_ids=[[3, 7, 11, 13]],
            feature_pcts_uint8=[[10, 20, 30, 40]],
        )
        # Shard B: 1 row, 2 segments, fids [5, 17] then [19, 23]; pcts [50, 60] then [70, 80]
        shard_b = _make_ground_truth_shard(
            n_rows=1,
            context_len=SEQ_LEN,
            feats_per_seg=feats_per_seg,
            feature_ids=[[5, 17, 19, 23]],
            feature_pcts_uint8=[[50, 60, 70, 80]],
        )
        save_packed_row_shard(shard_a, shard_dir / "shard_00000")
        save_packed_row_shard(shard_b, shard_dir / "shard_00001")
        (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        pipeline = _build_pipeline()
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,  # forces cross-shard
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
        )
        *_, fp_targets, _state = next(loader)
        assert fp_targets is not None
        # Row 0 (from shard A): segment 0 features {3: 10/255, 7: 20/255}
        #                       segment 1 features {11: 30/255, 13: 40/255}
        # Row 1 (from shard B): segment 0 features {5: 50/255, 17: 60/255}
        #                       segment 1 features {19: 70/255, 23: 80/255}
        assert abs(fp_targets[0, 0, 3].item() - 10.0 / 255.0) < 1e-6
        assert abs(fp_targets[0, 0, 7].item() - 20.0 / 255.0) < 1e-6
        assert abs(fp_targets[0, 1, 11].item() - 30.0 / 255.0) < 1e-6
        assert abs(fp_targets[0, 1, 13].item() - 40.0 / 255.0) < 1e-6
        assert abs(fp_targets[1, 0, 5].item() - 50.0 / 255.0) < 1e-6
        assert abs(fp_targets[1, 0, 17].item() - 60.0 / 255.0) < 1e-6
        assert abs(fp_targets[1, 1, 19].item() - 70.0 / 255.0) < 1e-6
        assert abs(fp_targets[1, 1, 23].item() - 80.0 / 255.0) < 1e-6
        # All other positions are zero
        # Build a mask of nonzero positions and ensure no surprises
        nonzero_positions = fp_targets.nonzero(as_tuple=False).tolist()
        # We expect exactly 8 nonzero positions, one per known feature
        assert len(nonzero_positions) == 8, (
            f"Expected 8 nonzero positions, got {len(nonzero_positions)}: "
            f"{nonzero_positions}"
        )


def _write_two_shards_with_selection(
    shard_dir: Path,
    n_rows: int,
    seed_a: int = 42,
    seed_b: int = 99,
) -> Path:
    """Write two packed-row shards AND matching selection shards.

    Returns the selection_dir path so the loader can find them.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_a = _make_test_shard(n_rows=n_rows, seed=seed_a)
    shard_b = _make_test_shard(n_rows=n_rows, seed=seed_b)
    save_packed_row_shard(shard_a, shard_dir / "shard_00000")
    save_packed_row_shard(shard_b, shard_dir / "shard_00001")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

    # Build sibling selection-shard dir
    sel_dir = shard_dir.parent / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    sel_a = _make_identity_selection(shard_a)
    sel_b = _make_identity_selection(shard_b)
    save_selection_shard(sel_a, sel_dir / "shard_00000")
    save_selection_shard(sel_b, sel_dir / "shard_00001")
    _write_selection_manifest(sel_dir, shard_dir)
    return sel_dir


class TestCrossShardSelectionPath:
    """Cross-shard batches via the selection-shard branch.

    When a `selection_dir` is provided, the loader takes a different
    branch in cross-shard reconstruction: it calls `reconstruct_batch`
    with the selection shard (not `reconstruct_and_select`). Production
    typically uses this path when selection shards are precomputed. The
    other cross-shard tests in this file only exercise the fused branch.
    """

    def test_cross_shard_selection_path_fp_targets_coherent(self, tmp_path):
        """Cross-shard batch with selection shards: fp_targets shape matches enc_mask."""
        shard_dir = tmp_path / "packed"
        sel_dir = _write_two_shards_with_selection(shard_dir, n_rows=2)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,  # 2 from shard A + 1 from shard B → cross-shard
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
            selection_dir=sel_dir,
        )

        (
            _i, _t, _ef, _em, enc_mask, _ep, _sm, seg_doc_id, fp_targets, _state,
        ) = next(loader)

        assert fp_targets is not None
        assert fp_targets.shape[:2] == enc_mask.shape[:2]
        assert fp_targets.shape[2] == N_SAE_FEATURES
        assert seg_doc_id.shape == enc_mask.shape[:2]

    def test_cross_shard_selection_path_none_fp_when_n_sae_not_set(self, tmp_path):
        """Selection-shard cross-shard batch with n_sae_features=None → fp_targets is None."""
        shard_dir = tmp_path / "packed"
        sel_dir = _write_two_shards_with_selection(shard_dir, n_rows=2)
        pipeline = _build_pipeline()

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=3,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=None,
            selection_dir=sel_dir,
        )

        *_, fp_targets, _state = next(loader)
        assert fp_targets is None

    def test_cross_shard_selection_path_resume_is_exact(self, tmp_path):
        shard_dir = tmp_path / "packed"
        sel_dir = _write_two_shards_with_selection(shard_dir, n_rows=13)
        uninterrupted = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            selection_dir=sel_dir,
        )
        baseline = [next(uninterrupted) for _ in range(6)]
        state = json.loads(json.dumps(baseline[3][9]))

        resumed = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=4,
            device="cpu",
            feature_pipeline=_build_pipeline(),
            n_sae_features=N_SAE_FEATURES,
            selection_dir=sel_dir,
            resume_state_dict=state,
        )
        for expected in baseline[3:6]:
            _assert_payloads_equal(next(resumed), expected)
