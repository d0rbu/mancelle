"""Slow integration tests for compute_statistics with real Gemma-2-2b features.

Requires CUDA GPU and parquet data. Skipped by default.
Run with: uv run pytest -m slow tests/test_compute_statistics_slow.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from nanochat.dataset import list_parquet_files
from steering.compute_statistics import ExtractionStatistics, compute_statistics
from steering.extract_features import (
    PoolingMode,
    load_sae_and_model,
    process_shard,
)
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    resolve_sae,
)
from steering.segmentation import SegmentMode

_CUDA_AVAILABLE = t.cuda.is_available()
_HAS_PARQUET_DATA = bool(list_parquet_files(None))


@pytest.mark.slow
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not _HAS_PARQUET_DATA, reason="No parquet data shards found")
class TestComputeStatisticsSlow:
    """Run compute_statistics on real Gemma-2-2b extracted features."""

    @pytest.fixture(scope="class")
    def extracted_features_dir(self, tmp_path_factory):
        """Extract features from one real shard (shared across tests in class)."""
        tmp = tmp_path_factory.mktemp("features")
        parquet_path = str(list_parquet_files(None)[0])

        resolved = resolve_sae(
            model="gemma-2-2b",
            release=SAERelease.gemmascope,
            activation_type=ActivationType.residual,
            layer=12,
            width=SAEWidth.W_16K,
        )

        model, sae, tokenizer = load_sae_and_model(
            "gemma-2-2b",
            layer=12,
            device="cuda",
            resolved=resolved,
        )

        process_shard(
            parquet_path,
            model,
            sae,
            layer=12,
            output_dir=str(tmp),
            context_window=1024,
            pooling=PoolingMode.mean_pool,
            shard_position=0,
            gpu_batch_size=4,
            cpu_batch_size=32,
            gc_mem_threshold=0.9,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            hook_name=resolved.hook_name,
            activation_type=ActivationType.residual,
        )

        return str(tmp)

    def test_no_filtering(self, extracted_features_dir, tmp_path):
        """compute_statistics with no filtering on real features."""
        output = str(tmp_path / "stats.json")
        result = compute_statistics(
            feature_dir=extracted_features_dir,
            output_path=output,
            activation_percentile=0.0,
        )
        assert isinstance(result, ExtractionStatistics)
        assert result.metadata.layer == 12
        assert result.metadata.activation_percentile == 0.0
        assert result.per_feature_thresholds is None
        assert result.corpus.n_docs > 0
        assert result.features.n_segments > 0

        # Verify safetensors is loadable
        from steering.stats_io import load_statistics

        stats = load_statistics(Path(output).with_suffix(".safetensors"))
        assert stats.metadata.layer == 12

    def test_with_percentile_filtering(self, extracted_features_dir, tmp_path):
        """compute_statistics with percentile filtering on real features."""
        output = str(tmp_path / "stats_filtered.json")
        result = compute_statistics(
            feature_dir=extracted_features_dir,
            output_path=output,
            activation_percentile=0.5,
        )
        assert result.metadata.activation_percentile == 0.5
        assert result.per_feature_thresholds is not None
        assert len(result.per_feature_thresholds) > 0

    def test_filtered_and_unfiltered_same_segment_count(
        self, extracted_features_dir, tmp_path
    ):
        """Total segment count should be identical regardless of filtering."""
        out_base = str(tmp_path / "base.json")
        out_filt = str(tmp_path / "filt.json")

        result_base = compute_statistics(
            feature_dir=extracted_features_dir,
            output_path=out_base,
            activation_percentile=0.0,
        )
        result_filt = compute_statistics(
            feature_dir=extracted_features_dir,
            output_path=out_filt,
            activation_percentile=0.5,
        )

        assert result_base.features.n_segments == result_filt.features.n_segments
        assert result_base.corpus.n_docs == result_filt.corpus.n_docs
