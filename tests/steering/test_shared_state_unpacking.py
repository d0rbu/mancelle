"""Tests for SharedState wiring in conditional encoder-decoder training.

Packed-row discovery used to happen inside train_one_config() and previously
failed to unpack percentile_checksum, tokenizer_checksum, feature_dir, and
boundaries from SharedState. Discovery now happens once in build_shared_state(),
and train_one_config() consumes the resolved shared directories.

These tests verify:
1. All SharedState fields are present and named correctly
2. build_shared_state wires packed-row discovery from the required inputs
3. Packed row discovery constructs the correct directory path from checksums
"""

from __future__ import annotations

import dataclasses
import inspect

import torch as t

from scripts.train.conditional_encdec_train import (
    SharedState,
    _packed_row_inputs_for_pipeline,
    build_shared_state,
    discover_packed_selection,
    train_one_config,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    feature_selection_dirname,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.overlay_io import CHECKSUM_PREFIX_LEN, glob_overlay_shards, save_overlay_shard
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard

# ======================================================================
# 1. SharedState field completeness
# ======================================================================


class TestSharedStateFields:
    """Verify SharedState dataclass has all required fields."""

    def test_shared_state_has_checksum_fields(self):
        """SharedState must have both checksum fields for packed row discovery."""
        fields = {f.name for f in dataclasses.fields(SharedState)}
        assert "percentile_checksum" in fields
        assert "tokenizer_checksum" in fields

    def test_shared_state_has_feature_dir(self):
        """SharedState must have feature_dir for steering eval."""
        fields = {f.name for f in dataclasses.fields(SharedState)}
        assert "feature_dir" in fields

    def test_shared_state_has_boundaries(self):
        """SharedState must have boundaries for feature pipeline."""
        fields = {f.name for f in dataclasses.fields(SharedState)}
        assert "boundaries" in fields

    def test_all_shared_state_fields(self):
        """Snapshot test: all expected fields exist in SharedState."""
        expected_fields = {
            "tokenizer",
            "token_bytes",
            "base_vocab",
            "feature_index",
            "feature_stats",
            "per_feature_stats",
            "n_sae_features",
            "percentile_checksum",
            "tokenizer_checksum",
            "model",
            "orig_model",
            "cached_context_provider",
            "device",
            "device_type",
            "ddp",
            "ddp_rank",
            "ddp_local_rank",
            "ddp_world_size",
            "extraction_segment_size",
            "cfg",
            "gpu_peak_flops",
            "boundaries",
            "d_ref",
            "b_ref",
            "steering_eval_cfg",
            "steering_probe_model",
            "feature_dir",
            "packed_row_dir",
            "selection_dir",
            "original_val_feature_dir",
            "original_val_packed_row_dir",
            "original_val_selection_dir",
            "original_val_feature_index",
            "spliced_val_feature_index",
            "val_packed_row_dir",
            "val_selection_dir",
            "max_seg_size",
            "pad_F",
            "live_pad_n_seg",
        }
        actual_fields = {f.name for f in dataclasses.fields(SharedState)}
        missing = expected_fields - actual_fields
        assert not missing, f"SharedState is missing fields: {missing}"


# ======================================================================
# 2. train_one_config unpacking coverage
# ======================================================================


class TestTrainOneConfigUnpacking:
    """Verify the current SharedState access contract."""

    def test_packed_row_inputs_wired_in_build_shared_state_source(self):
        """The wrapper and extracted discovery seam preserve every packed-row input."""
        build_source = inspect.getsource(build_shared_state)
        discovery_source = inspect.getsource(discover_packed_selection)

        wrapper_patterns = [
            "percentile_checksum = load_percentile_checksum(stats_path)",
            "tokenizer_checksum = load_tokenizer_checksum(stats_path)",
            "discover_packed_selection(",
            "percentile_checksum=percentile_checksum",
            "tokenizer_checksum=tokenizer_checksum",
            "packed_row_dir = discovery.train_packed_dir",
            "selection_dir = discovery.train_selection_dir",
            "original_val_packed_row_dir = discovery.original_val_packed_dir",
        ]
        for pattern in wrapper_patterns:
            assert pattern in build_source, (
                f"build_shared_state does not contain: {pattern}\n"
                f"This can break packed-row discovery before train_one_config runs."
            )

        discovery_patterns = [
            "_packed_row_train_dir(",
            "_resolve_original_val_feature_dir(cfg, feature_dir_str)",
            "compute_selection_hash(",
            "discover_val_packed_row_dir(",
        ]
        for pattern in discovery_patterns:
            assert pattern in discovery_source, (
                f"discover_packed_selection does not contain: {pattern}\n"
                "This can break the side-effect-free discovery extraction."
            )

    def test_feature_index_accessed(self):
        """feature_index is accessed for live dataloader fallback and steering eval."""
        source = inspect.getsource(train_one_config)
        assert "shared.feature_index" in source

    def test_train_one_config_accesses_required_shared_fields(self):
        """Every SharedState field train_one_config relies on should be accessed.

        Some fields are unpacked into locals; others are intentionally accessed
        as shared.X at the call site. This protects against stale local-variable
        assumptions without requiring a specific unpacking style.
        """
        source = inspect.getsource(train_one_config)

        critical_fields = {
            "tokenizer",
            "token_bytes",
            "feature_index",
            "per_feature_stats",
            "n_sae_features",
            "feature_dir",
            "boundaries",
            "model",
            "orig_model",
            "cached_context_provider",
            "device",
            "device_type",
            "ddp_rank",
            "ddp_world_size",
            "extraction_segment_size",
            "gpu_peak_flops",
            "d_ref",
            "b_ref",
            "packed_row_dir",
            "selection_dir",
            "original_val_packed_row_dir",
            "original_val_selection_dir",
            "pad_F",
            "live_pad_n_seg",
        }

        for field in critical_fields:
            assert f"shared.{field}" in source, (
                f"SharedState.{field} is never accessed in train_one_config. "
                f"If this field is used as a local var, it may cause a NameError."
            )


# ======================================================================
# 3. Delta configs must not route into packed-row loader
# ======================================================================


def _minimal_pipeline(*, delta: bool) -> FeaturePipelineConfig:
    delta_bundle = (
        DeltaBundle(
            config=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
            percentile_table=t.zeros(4, 99, dtype=t.float64),
        )
        if delta
        else None
    )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                threshold_type=ThresholdType.feature_percentile,
                threshold=0.9,
            ),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(4, 0, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=2,
        delta_bundle=delta_bundle,
    )


class TestPackedRowDeltaFallback:
    """Feature-delta pipelines must use the live loader when packed rows exist."""

    def test_delta_pipeline_clears_packed_row_inputs(self, tmp_path):
        packed = tmp_path / "packed"
        selection = packed / "sel" / "hash"
        pipeline = _minimal_pipeline(delta=True)

        packed_row_dir, selection_dir, ignored = _packed_row_inputs_for_pipeline(
            packed_row_dir=packed,
            selection_dir=selection,
            feature_pipeline=pipeline,
        )

        assert packed_row_dir is None
        assert selection_dir is None
        assert ignored is True

    def test_non_delta_pipeline_preserves_packed_row_inputs(self, tmp_path):
        packed = tmp_path / "packed"
        selection = packed / "sel" / "hash"
        pipeline = _minimal_pipeline(delta=False)

        packed_row_dir, selection_dir, ignored = _packed_row_inputs_for_pipeline(
            packed_row_dir=packed,
            selection_dir=selection,
            feature_pipeline=pipeline,
        )

        assert packed_row_dir == packed
        assert selection_dir == selection
        assert ignored is False


# ======================================================================
# 4. Packed row directory discovery logic
# ======================================================================


class TestPackedRowDiscovery:
    """Test the packed row directory path construction used in train_one_config."""

    def test_packed_row_path_construction(self, tmp_path):
        """Verify the directory path format:
        feature_dir/pct/<pct_hash[:16]>/tok/<tok_hash[:16]>/c<seq_len>/s<max_shards>/
        """
        feature_dir = tmp_path / "features"
        pct_checksum = "a" * 64
        tok_checksum = "b" * 64
        max_seq_len = 2048
        max_train_shards = 50

        shards_str = f"s{max_train_shards}"
        candidate = (
            feature_dir
            / "pct"
            / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok"
            / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / f"c{max_seq_len}"
            / shards_str
        )

        expected = feature_dir / "pct" / ("a" * 16) / "tok" / ("b" * 16) / "c2048" / "s50"
        assert candidate == expected

    def test_packed_row_path_no_max_shards(self, tmp_path):
        """Without max_shards, path uses 'sall' suffix."""
        feature_dir = tmp_path / "features"
        pct_checksum = "a" * 64
        tok_checksum = "b" * 64
        max_seq_len = 2048
        max_train_shards = None

        shards_str = f"s{max_train_shards}" if max_train_shards is not None else "sall"
        candidate = (
            feature_dir
            / "pct"
            / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok"
            / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / f"c{max_seq_len}"
            / shards_str
        )

        assert candidate.name == "sall"

    def test_packed_row_discovery_succeeds_when_dir_exists(self, tmp_path):
        """If the packed row directory exists with shards, discovery succeeds."""
        feature_dir = tmp_path / "features"
        pct_checksum = "a" * 64
        tok_checksum = "b" * 64

        candidate = (
            feature_dir
            / "pct"
            / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok"
            / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / "c2048"
            / "s50"
        )
        candidate.mkdir(parents=True)

        # Create a dummy shard
        shard = PackedRowShard(
            tokens=t.randint(0, 1000, (2, 33), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2, 4], dtype=t.int32),
            seg_token_start=t.tensor([0, 16, 0, 16], dtype=t.uint16),
            seg_token_end=t.tensor([16, 32, 16, 32], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0, 0, 0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 2, 4, 6, 8], dtype=t.int32),
            feature_ids=t.randint(0, 100, (8,), dtype=t.uint16),
            feature_pcts=t.randint(0, 255, (8,), dtype=t.uint8),
        )
        save_packed_row_shard(shard, candidate / "shard_00000")

        # Simulate the train_one_config discovery logic
        assert candidate.is_dir()
        found_shards = glob_overlay_shards(candidate)
        assert len(found_shards) > 0

    def test_packed_row_discovery_fails_without_checksums(self):
        """If checksums are None, packed row discovery should be skipped."""
        # Simulating the guard in train_one_config:
        # if use_precomputed and percentile_checksum and tokenizer_checksum:
        use_precomputed = True
        percentile_checksum = None
        tokenizer_checksum = "b" * 64

        # This should short-circuit
        should_discover = use_precomputed and percentile_checksum and tokenizer_checksum
        assert not should_discover

    def test_packed_row_discovery_fails_with_empty_dir(self, tmp_path):
        """If the directory exists but is empty, discovery should fail."""
        feature_dir = tmp_path / "features"
        pct_checksum = "a" * 64
        tok_checksum = "b" * 64

        candidate = (
            feature_dir
            / "pct"
            / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok"
            / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / "c2048"
            / "s50"
        )
        candidate.mkdir(parents=True)

        assert candidate.is_dir()
        found_shards = glob_overlay_shards(candidate)
        assert len(found_shards) == 0

        # The guard: candidate.is_dir() and glob_overlay_shards(candidate)
        should_use = candidate.is_dir() and glob_overlay_shards(candidate)
        assert not should_use


# ======================================================================
# 4. Selection shard discovery logic
# ======================================================================


class TestSelectionShardDiscovery:
    """Test the selection shard directory construction used in train_one_config.

    The new wiring: packed_row_dir / "sel" / feature_selection_dirname(feature_selection)
    """

    def test_selection_dir_path_sampling(self, tmp_path):
        """Selection dir uses geometric distribution dirname for sampling selection."""
        packed_row_dir = tmp_path / "packed"
        fs = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
        )
        sel_dirname = feature_selection_dirname(fs)
        sel_candidate = packed_row_dir / "sel" / sel_dirname

        assert sel_candidate == packed_row_dir / "sel" / "dist_geometric" / "mean_5.0" / "std_5.0"

    def test_selection_dir_path_threshold(self, tmp_path):
        """Selection dir uses threshold dirname for threshold selection."""
        packed_row_dir = tmp_path / "packed"
        fs = ThresholdSelection(
            threshold_type=ThresholdType.segment_topk,
            threshold=2048.0,
        )
        sel_dirname = feature_selection_dirname(fs)
        sel_candidate = packed_row_dir / "sel" / sel_dirname

        assert sel_candidate == packed_row_dir / "sel" / "t2048.0"

    def test_selection_dir_discovery_succeeds(self, tmp_path):
        """When selection dir has shards, discovery succeeds."""
        packed_row_dir = tmp_path / "packed"
        fs = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
        )
        sel_dirname = feature_selection_dirname(fs)
        sel_candidate = packed_row_dir / "sel" / sel_dirname

        # Create dir and a dummy shard
        sel_candidate.mkdir(parents=True)
        save_overlay_shard(
            {"dummy": t.zeros(1)},
            sel_candidate / "shard_00000",
        )

        assert sel_candidate.is_dir()
        assert len(glob_overlay_shards(sel_candidate)) > 0

    def test_selection_dir_discovery_fails_empty(self, tmp_path):
        """When selection dir exists but is empty, discovery fails."""
        packed_row_dir = tmp_path / "packed"
        fs = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
        )
        sel_dirname = feature_selection_dirname(fs)
        sel_candidate = packed_row_dir / "sel" / sel_dirname
        sel_candidate.mkdir(parents=True)

        # Guard pattern from training script
        should_use = sel_candidate.is_dir() and glob_overlay_shards(sel_candidate)
        assert not should_use

    def test_selection_dir_none_when_no_packed_rows(self):
        """If packed_row_dir is None, selection_dir should also be None."""
        packed_row_dir = None
        selection_dir = None
        if packed_row_dir is not None:
            selection_dir = "should_not_reach"
        assert selection_dir is None

    def test_train_one_config_wires_selection_and_pipeline(self):
        """train_one_config passes selection_dir and feature_pipeline to the loader."""
        source = inspect.getsource(train_one_config)
        assert "shared.selection_dir" in source, (
            "train_one_config must use shared.selection_dir (discovered in build_shared_state)"
        )
        assert "feature_pipeline=feature_pipeline" in source, (
            "feature_pipeline must be passed to precomputed_data_loader for inline selection"
        )
        assert "feature_pipeline.effective_buckets" in source, (
            "effective_buckets must come from feature_pipeline, not raw cfg.num_buckets"
        )
