"""Regression tests for the high-severity validate_features audit findings."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf

import steering.validate_features as vf
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.feature_io import TensorKey, glob_features, load_features, save_features
from steering.overlay_io import (
    OverlayKind,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.refined_types import PositiveInt
from steering.sae_resolver import L0Tier
from steering.segmentation import (
    SegmentMode,
    segment_document,
    truncate_to_training_window,
)
from steering.stats_io import (
    save_percentile_checksum,
    save_statistics,
)
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)


class _CharTokenizer:
    """One reversible token per Unicode code point."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def __call__(self, text: str, *, padding: bool = False) -> dict[str, list[int]]:
        assert padding is False
        return {"input_ids": self.encode(text)}

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _ByteFallbackTokenizer:
    """One token per UTF-8 byte, including replacement-char partial decodes."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def __call__(self, text: str, *, padding: bool = False) -> dict[str, list[int]]:
        assert padding is False
        return {"input_ids": self.encode(text)}

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


def _source_shard(
    texts: list[str],
    tokenizer,
    *,
    segment_size: int = 4,
    context_window: int = 64,
    sae_max_context: int | None = None,
) -> dict:
    """Build segmentation metadata through the exact production helpers."""
    doc_ids: list[int] = []
    segment_indices: list[int] = []
    segment_positions: list[int] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    doc_token_counts: list[int] = []

    for doc_id, text in enumerate(texts):
        kept_ids, n_full = truncate_to_training_window(
            text,
            tokenizer,
            context_window,
            sae_max_context,
        )
        segments, starts, ends = segment_document(
            kept_ids,
            tokenizer,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(segment_size),
        )
        doc_ids.extend([doc_id] * len(segments))
        segment_indices.extend(range(len(segments)))
        segment_positions.extend(start for start, _ in segments)
        char_starts.extend(starts)
        char_ends.extend(ends)
        doc_token_counts.append(n_full)

    n_segments = len(doc_ids)
    return {
        "offsets": t.arange(n_segments + 1, dtype=t.int64),
        "feature_ids": t.tensor(
            [idx % 4 for idx in range(n_segments)],
            dtype=t.int32,
        ),
        "feature_mags": t.tensor(
            [float(idx + 1) for idx in range(n_segments)],
            dtype=t.float32,
        ),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(segment_indices, dtype=t.int32),
        "segment_positions": t.tensor(segment_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": len(texts),
        "n_segments": n_segments,
        "layer": 8,
        "context_window": context_window,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_independent_features": False,
        "ignore_bos_features": False,
    }


def _write_source_family(
    tmp_path: Path,
    texts: list[str],
    *,
    tokenizer=None,
    segment_size: int = 4,
) -> tuple[Path, Path, object]:
    tokenizer = tokenizer or _CharTokenizer()
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    shard = _source_shard(texts, tokenizer, segment_size=segment_size)
    save_features(shard, feature_dir / "shard_00000")
    pq.write_table(
        pa.table({"text": texts}),
        parquet_dir / "shard_00000.parquet",
    )
    return feature_dir, parquet_dir, tokenizer


def _base_shard(
    *,
    n_docs: int = 1,
    segs_per_doc: int = 2,
    segment_size: int = 4,
    magnitude_offset: float = 0.0,
) -> dict:
    n_segments = n_docs * segs_per_doc
    doc_ids = [
        doc_id
        for doc_id in range(n_docs)
        for _ in range(segs_per_doc)
    ]
    segment_indices = list(range(segs_per_doc)) * n_docs
    segment_positions = [
        seg_idx * segment_size
        for _ in range(n_docs)
        for seg_idx in range(segs_per_doc)
    ]
    char_starts = list(segment_positions)
    char_ends = [position + segment_size for position in segment_positions]
    return {
        "offsets": t.arange(n_segments + 1, dtype=t.int64),
        "feature_ids": t.tensor(
            [idx % 4 for idx in range(n_segments)],
            dtype=t.int32,
        ),
        "feature_mags": t.tensor(
            [magnitude_offset + 10.0 + idx for idx in range(n_segments)],
            dtype=t.float32,
        ),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(segment_indices, dtype=t.int32),
        "segment_positions": t.tensor(segment_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(
            [segs_per_doc * segment_size] * n_docs,
            dtype=t.int32,
        ),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 8,
        "context_window": 64,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_independent_features": False,
        "ignore_bos_features": True,
        "l0_tier": "medium",
    }


def _write_base_shard(
    feature_dir: Path,
    shard_index: int,
    **kwargs,
) -> dict:
    shard = _base_shard(**kwargs)
    save_features(shard, feature_dir / f"shard_{shard_index:05d}")
    return shard


def _per_feature_stats(n_sae_features: int) -> dict[str, PerFeatureStat]:
    percentiles = {f"p{pct}": float(pct) for pct in range(1, 100)}
    return {
        str(feature_id): PerFeatureStat(
            count=1,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=1.0,
            min_mag=1.0,
            max_mag=99.0,
            median_mag=50.0,
            percentiles=dict(percentiles),
        )
        for feature_id in range(n_sae_features)
    }


def _make_statistics(
    domain_shards: list[dict],
    *,
    pooling_mode: str = "max_pool",
    segment_size: int = 4,
    context_window: int = 64,
    layer: int = 8,
    n_sae_features: int = 4,
    l0_tier: L0Tier | None = L0Tier.medium,
    ignore_bos_features: bool = True,
) -> ExtractionStatistics:
    n_docs = sum(int(shard["n_docs"]) for shard in domain_shards)
    n_segments = sum(int(shard["n_segments"]) for shard in domain_shards)
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode=pooling_mode,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=segment_size,
            context_window=context_window,
            layer=layer,
            n_sae_features=n_sae_features,
            generated_at="2026-07-24T00:00:00+00:00",
        ),
        corpus=CorpusStats(
            n_shards_processed=len(domain_shards),
            n_docs=n_docs,
            n_tokens_gpt2=sum(
                int(shard["doc_token_counts"].sum().item())
                for shard in domain_shards
            ),
            mean_tokens_per_doc=8.0,
            median_tokens_per_doc=8,
        ),
        features=SegmentFeatureStats(
            n_segments=n_segments,
            n_features_seen=n_sae_features,
            mean_active_per_segment=1.0,
            median_active_per_segment=1,
            p90_active_per_segment=1,
            max_active_per_segment=1,
            min_active_per_segment=1,
            n_total_sae_features=n_sae_features,
            pct_features_active=1.0,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1,
            mean_bytes_per_doc=1.0,
            mean_bytes_per_token=1.0,
        ),
        per_feature_stats=_per_feature_stats(n_sae_features),
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
        l0_tier=l0_tier,
        ignore_bos_features=ignore_bos_features,
    )


def _write_statistics(
    feature_dir: Path,
    domain_shards: list[dict],
    **kwargs,
) -> tuple[ExtractionStatistics, Path]:
    stats = _make_statistics(domain_shards, **kwargs)
    stats_path = save_statistics(stats, feature_dir / "feature_statistics")
    return stats, stats_path


def _write_percentile_overlay(
    feature_dir: Path,
    stats: ExtractionStatistics,
    stats_path: Path,
) -> tuple[str, Path]:
    table = build_percentile_table(
        stats.per_feature_stats,
        stats.metadata.n_sae_features,
    )
    checksum = compute_percentile_checksum(table)
    base_paths = glob_features(feature_dir)
    output_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    write_manifest(
        output_dir,
        checksum,
        OverlayKind.PERCENTILE,
        metadata={
            "n_sae_features": stats.metadata.n_sae_features,
            "stats_file": stats_path.name,
        },
        base_shard_paths=base_paths,
        cache_dir=feature_dir,
    )
    for base_path in base_paths:
        base = load_features(base_path)
        values = percentile_to_uint8(
            magnitude_to_percentile(
                base["feature_ids"].to(t.int64),
                base["feature_mags"],
                table,
            )
        )
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES.value: values},
            overlay_shard_path(output_dir, vf.shard_stem_from_path(base_path)),
        )
    save_percentile_checksum(stats_path, checksum)
    return checksum, output_dir


def _current_layout_family(
    tmp_path: Path,
    *,
    n_shards: int = 2,
    exclude_final_from_stats: bool = True,
    write_overlay: bool = True,
) -> tuple[
    Path,
    list[dict],
    ExtractionStatistics,
    Path,
    str,
    Path,
]:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    shards = [
        _write_base_shard(feature_dir, shard_index)
        for shard_index in range(n_shards)
    ]
    stats_domain = shards[:-1] if exclude_final_from_stats and n_shards > 1 else shards
    stats, stats_path = _write_statistics(feature_dir, stats_domain)
    table = build_percentile_table(
        stats.per_feature_stats,
        stats.metadata.n_sae_features,
    )
    checksum = compute_percentile_checksum(table)
    output_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    if write_overlay:
        checksum, output_dir = _write_percentile_overlay(
            feature_dir,
            stats,
            stats_path,
        )
    else:
        save_percentile_checksum(stats_path, checksum)
    return feature_dir, shards, stats, stats_path, checksum, output_dir


class TestSV003ExactSourceSegmentation:
    def test_rejects_shifted_internal_boundary_that_still_tiles(
        self,
        tmp_path: Path,
    ) -> None:
        tokenizer = _CharTokenizer()
        text = "abcdefghijkl"
        shard = _source_shard([text], tokenizer, segment_size=4)
        shard["segment_char_ends"][0] += 1
        shard["segment_char_starts"][1] += 1
        structural = vf.validate_shard_segments(shard)
        assert structural.valid, structural.violations
        parquet_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": [text]}), parquet_path)

        result = vf.validate_shard_against_source(
            shard,
            parquet_path,
            tokenizer,
            context_window=64,
            segment_size=4,
        )

        assert not result.valid
        assert any("char_end" in violation for violation in result.violations)

    @pytest.mark.parametrize(
        ("field", "index", "delta", "message"),
        [
            ("segment_indices", 1, 1, "segment_indices"),
            ("segment_positions", 1, 1, "segment_positions"),
            ("segment_char_starts", 1, 1, "segment_char_starts"),
            ("segment_char_ends", 1, 1, "segment_char_ends"),
        ],
    )
    def test_source_validator_compares_each_production_boundary_field(
        self,
        tmp_path: Path,
        field: str,
        index: int,
        delta: int,
        message: str,
    ) -> None:
        tokenizer = _CharTokenizer()
        text = "abcdefghijkl"
        shard = _source_shard([text], tokenizer, segment_size=4)
        shard[field][index] += delta
        parquet_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": [text]}), parquet_path)

        result = vf.validate_shard_against_source(
            shard,
            parquet_path,
            tokenizer,
            context_window=64,
            segment_size=4,
        )

        assert not result.valid
        assert any(message in violation for violation in result.violations)

    @pytest.mark.parametrize(
        ("tokenizer", "text", "segment_size"),
        [
            (_CharTokenizer(), "healthy ASCII segmentation", 5),
            (_ByteFallbackTokenizer(), "ab𐨀cd 🎉", 3),
        ],
    )
    def test_accepts_exact_production_segmentation(
        self,
        tmp_path: Path,
        tokenizer,
        text: str,
        segment_size: int,
    ) -> None:
        shard = _source_shard(
            [text],
            tokenizer,
            segment_size=segment_size,
        )
        parquet_path = tmp_path / "shard_00000.parquet"
        pq.write_table(pa.table({"text": [text]}), parquet_path)

        result = vf.validate_shard_against_source(
            shard,
            parquet_path,
            tokenizer,
            context_window=64,
            segment_size=segment_size,
        )

        assert result.valid, result.violations


class TestSV004StatisticsIdentity:
    @pytest.mark.parametrize(
        ("stats_kwargs", "expected_field"),
        [
            ({"layer": 9}, "layer"),
            ({"pooling_mode": "mean_pool"}, "pooling"),
            ({"segment_size": 8}, "segment_size"),
            ({"context_window": 32}, "context_window"),
            ({"n_sae_features": 5}, "n_sae_features"),
            ({"l0_tier": L0Tier.big}, "l0_tier"),
            ({"ignore_bos_features": False}, "ignore_bos_features"),
        ],
    )
    def test_rejects_independent_statistics_identity_mismatch(
        self,
        tmp_path: Path,
        stats_kwargs: dict,
        expected_field: str,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shard = _write_base_shard(feature_dir, 0)
        _write_statistics(feature_dir, [shard], **stats_kwargs)

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert not result.valid
        assert any(expected_field in violation for violation in result.violations)

    def test_accepts_exact_statistics_identity(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shard = _write_base_shard(feature_dir, 0)
        _write_statistics(feature_dir, [shard])

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations


class TestSV005DocumentCoverage:
    @pytest.mark.parametrize("missing_doc_id", [1, 2])
    def test_rejects_missing_middle_or_final_document(
        self,
        missing_doc_id: int,
    ) -> None:
        shard = _source_shard(
            ["abcdefgh", "ijklmnop", "qrstuvwx"],
            _CharTokenizer(),
            segment_size=4,
        )
        keep = shard["doc_ids"] != missing_doc_id
        for field in (
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "segment_char_starts",
            "segment_char_ends",
            "feature_ids",
            "feature_mags",
        ):
            shard[field] = shard[field][keep]
        shard["n_segments"] = int(keep.sum().item())
        shard["offsets"] = t.arange(shard["n_segments"] + 1, dtype=t.int64)

        result = vf.validate_shard_segments(shard)

        assert not result.valid
        assert any("document IDs" in violation for violation in result.violations)

    def test_rejects_all_zero_segment_documents(self) -> None:
        shard = {
            "offsets": t.tensor([0], dtype=t.int64),
            "feature_ids": t.tensor([], dtype=t.int32),
            "feature_mags": t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor([], dtype=t.int32),
            "segment_indices": t.tensor([], dtype=t.int32),
            "segment_positions": t.tensor([], dtype=t.int32),
            "segment_char_starts": t.tensor([], dtype=t.int32),
            "segment_char_ends": t.tensor([], dtype=t.int32),
            "doc_token_counts": t.tensor([1], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 0,
            "layer": 8,
            "context_window": 64,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 4,
        }

        result = vf.validate_shard_segments(shard)

        assert not result.valid
        assert any("document IDs" in violation for violation in result.violations)

    def test_rejects_null_source_text(self, tmp_path: Path) -> None:
        tokenizer = _CharTokenizer()
        shard = _source_shard(["abcd"], tokenizer, segment_size=4)
        parquet_path = tmp_path / "shard_00000.parquet"
        pq.write_table(
            pa.table({"text": pa.array([None], type=pa.string())}),
            parquet_path,
        )

        result = vf.validate_shard_against_source(
            shard,
            parquet_path,
            tokenizer,
            context_window=64,
            segment_size=4,
        )

        assert not result.valid
        assert any("null text" in violation for violation in result.violations)

    def test_accepts_exact_document_id_range(self) -> None:
        shard = _source_shard(
            ["abcdefgh", "ijklmnop", "qrstuvwx"],
            _CharTokenizer(),
            segment_size=4,
        )

        result = vf.validate_shard_segments(shard)

        assert result.valid, result.violations


def _main_cfg(**overrides):
    values = {
        "feature_dir": "/features",
        "parquet_dir": None,
        "resolve_parquet_default": False,
        "model": None,
        "n_spot_check": 0,
        "sae_release": None,
        "sae_id": None,
        "model_dtype": "float32",
        "feature_count_rtol": 0.01,
        "jaccard_threshold": 0.99,
        "n_sae_features": None,
    }
    values.update(overrides)
    return OmegaConf.create(values)


class TestSV007ParquetRequiresTokenizer:
    def test_api_raises_before_shard_discovery_without_tokenizer(
        self,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(
                vf,
                "glob_features",
                side_effect=AssertionError("shard discovery must not run"),
            ) as glob_mock,
            pytest.raises(ValueError, match="tokenizer.*parquet_dir"),
        ):
            vf.validate_feature_directory(
                tmp_path / "features",
                parquet_dir=tmp_path / "parquets",
            )
        glob_mock.assert_not_called()

    def test_cli_rejects_parquet_without_model_before_validation(
        self,
        tmp_path: Path,
    ) -> None:
        validate_mock = MagicMock()
        with (
            patch.object(vf, "validate_feature_directory", validate_mock),
            patch.object(vf.HydraConfig, "initialized", return_value=False),
            pytest.raises(ValueError, match="model.*parquet_dir"),
        ):
            vf.main.__wrapped__(
                _main_cfg(parquet_dir=str(tmp_path / "parquets"))
            )
        validate_mock.assert_not_called()

    def test_matching_stem_cannot_hide_corrupt_source_without_tokenizer(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir, parquet_dir, _ = _write_source_family(
            tmp_path,
            ["healthy source"],
        )
        pq.write_table(
            pa.table({"text": ["different corrupt source"]}),
            parquet_dir / "shard_00000.parquet",
        )

        with pytest.raises(ValueError, match="tokenizer.*parquet_dir"):
            vf.validate_feature_directory(
                feature_dir,
                parquet_dir=parquet_dir,
            )

    def test_parquet_with_tokenizer_accepts_healthy_source(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir, parquet_dir, tokenizer = _write_source_family(
            tmp_path,
            ["abcdefgh", "ijklmnop"],
        )

        result = vf.validate_feature_directory(
            feature_dir,
            parquet_dir=parquet_dir,
            tokenizer=tokenizer,
        )

        assert result.valid, result.violations


class TestSV008ActivationSpotCheckPreflight:
    @pytest.mark.parametrize(
        ("missing_key", "expected_name"),
        [
            ("parquet_dir", "parquet_dir"),
            ("model", "model"),
            ("sae_release", "sae_release"),
            ("sae_id", "sae_id"),
        ],
    )
    def test_positive_spot_check_rejects_each_missing_prerequisite(
        self,
        tmp_path: Path,
        missing_key: str,
        expected_name: str,
    ) -> None:
        values = {
            "parquet_dir": str(tmp_path / "parquets"),
            "model": "gemma-test",
            "sae_release": "release",
            "sae_id": "sae-id",
            "n_spot_check": 1,
        }
        values[missing_key] = None
        load_mock = MagicMock()
        validate_mock = MagicMock()
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=False),
            patch.object(vf, "load_sae_and_model", load_mock),
            patch.object(vf, "resolve_tokenizer") as tokenizer_mock,
            patch.object(vf, "validate_feature_directory", validate_mock),
            pytest.raises(ValueError, match=expected_name),
        ):
            vf.main.__wrapped__(_main_cfg(**values))
        load_mock.assert_not_called()
        tokenizer_mock.assert_not_called()
        validate_mock.assert_not_called()

    def test_zero_is_the_explicit_accepted_skip(self) -> None:
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=False),
            patch.object(
                vf,
                "validate_feature_directory",
                return_value=vf.ValidationResult(valid=True, violations=[]),
            ) as validate_mock,
            patch.object(vf, "load_sae_and_model") as load_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            vf.main.__wrapped__(_main_cfg(n_spot_check=0))

        assert exc_info.value.code == 0
        validate_mock.assert_called_once()
        load_mock.assert_not_called()

    def test_activation_api_default_is_zero(self) -> None:
        default = inspect.signature(
            vf.validate_shard_activations
        ).parameters["n_spot_check"].default
        assert default == 0

    def test_absent_main_key_defaults_to_zero(self) -> None:
        cfg = _main_cfg()
        del cfg["n_spot_check"]
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=False),
            patch.object(
                vf,
                "validate_feature_directory",
                return_value=vf.ValidationResult(valid=True, violations=[]),
            ) as validate_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            vf.main.__wrapped__(cfg)

        assert exc_info.value.code == 0
        validate_mock.assert_called_once()

    def test_hydra_default_is_zero_but_explicit_override_is_honored(self) -> None:
        cfg = _main_cfg(n_spot_check=5)
        no_overrides = SimpleNamespace(
            overrides=SimpleNamespace(task=[]),
        )
        explicit = SimpleNamespace(
            overrides=SimpleNamespace(task=["n_spot_check=5"]),
        )
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=True),
            patch.object(vf.HydraConfig, "get", return_value=no_overrides),
        ):
            assert vf._resolve_n_spot_check(cfg) == 0
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=True),
            patch.object(vf.HydraConfig, "get", return_value=explicit),
        ):
            assert vf._resolve_n_spot_check(cfg) == 5

    def test_complete_configuration_invokes_activation_validation(
        self,
        tmp_path: Path,
    ) -> None:
        parquet_dir = tmp_path / "parquets"
        parquet_dir.mkdir()
        pq.write_table(
            pa.table({"text": ["abcd"]}),
            parquet_dir / "shard_00000.parquet",
        )
        shard_path = tmp_path / "shard_00000_features.safetensors.zst"
        save_features(
            {"offsets": t.tensor([0, 1]), "feature_ids": t.tensor([1]),
             "feature_mags": t.tensor([1.0])},
            shard_path,
        )
        model = MagicMock(cfg=SimpleNamespace(n_ctx=64))
        sae = MagicMock()
        tokenizer = _CharTokenizer()
        shard_data = {
            "layer": 8,
            "context_window": 64,
            "segment_size": 4,
            "pooling": "max_pool",
        }
        with (
            patch.object(vf.HydraConfig, "initialized", return_value=False),
            patch.object(
                vf,
                "load_sae_and_model",
                return_value=(model, sae, tokenizer),
            ),
            # The backend attach (#1633) builds a real nnterp model; that is
            # covered by test_validate_features_spotcheck_backend.py. Here it
            # is stubbed so the preflight->activation wiring stays the subject.
            patch.object(
                vf, "attach_spot_check_backend", return_value="nnterp"
            ) as backend_mock,
            patch.object(
                vf,
                "validate_feature_directory",
                return_value=vf.ValidationResult(valid=True, violations=[]),
            ),
            patch.object(vf, "glob_features", return_value=[shard_path]),
            patch.object(vf, "load_features", return_value=shard_data),
            patch.object(
                vf,
                "validate_shard_activations",
                return_value=vf.ValidationResult(valid=True, violations=[]),
            ) as activation_mock,
            pytest.raises(SystemExit) as exc_info,
        ):
            vf.main.__wrapped__(
                _main_cfg(
                    feature_dir=str(tmp_path),
                    parquet_dir=str(parquet_dir),
                    model="gemma-test",
                    sae_release="release",
                    sae_id="sae-id",
                    n_spot_check=1,
                    model_dtype="bfloat16",
                )
            )

        assert exc_info.value.code == 0
        activation_mock.assert_called_once()
        backend_mock.assert_called_once()
        assert backend_mock.call_args.args == (model,)
        assert backend_mock.call_args.kwargs == {
            "backend": vf.Backend.auto,
            # main() picks the device the same way the spot-check loads the model.
            "device": "cuda" if t.cuda.is_available() else "cpu",
            # The configured extraction dtype must reach the backend build, or the
            # recompute runs in a different precision than the shard (#1633).
            "model_dtype": t.bfloat16,
        }


class TestSV009CurrentPercentileOverlay:
    def test_accepts_healthy_current_layout_overlay(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir, _, _, _, _, _ = _current_layout_family(tmp_path)

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations

    def test_rejects_missing_overlay(self, tmp_path: Path) -> None:
        feature_dir, _, _, _, checksum, _ = _current_layout_family(
            tmp_path,
            write_overlay=False,
        )

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any(
            "percentile overlay" in violation.lower()
            and checksum[:16] in violation
            for violation in result.violations
        )

    def test_rejects_missing_overlay_shard(self, tmp_path: Path) -> None:
        feature_dir, _, _, _, _, output_dir = _current_layout_family(tmp_path)
        overlay_shard_path(output_dir, "shard_00001").unlink()

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any(
            "missing percentile overlay shard" in violation.lower()
            and "shard_00001" in violation
            for violation in result.violations
        )

    def test_rejects_stale_base_content_hash(self, tmp_path: Path) -> None:
        feature_dir, _, _, _, _, _ = _current_layout_family(tmp_path)
        _write_base_shard(
            feature_dir,
            0,
            magnitude_offset=100.0,
        )

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any(
            "base shard" in violation.lower()
            and "hash" in violation.lower()
            for violation in result.violations
        )

    def test_rejects_wrong_overlay_tensor_length(self, tmp_path: Path) -> None:
        feature_dir, _, _, _, _, output_dir = _current_layout_family(tmp_path)
        payload_path = overlay_shard_path(output_dir, "shard_00000")
        values = load_overlay_shard(payload_path)[
            TensorKey.FEATURE_PERCENTILES.value
        ]
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES.value: values[:-1]},
            payload_path,
        )

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any("numel" in violation for violation in result.violations)

    def test_rejects_wrong_manifest_checksum(self, tmp_path: Path) -> None:
        feature_dir, _, _, _, _, output_dir = _current_layout_family(tmp_path)
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checksum"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest))

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any(
            "percentile overlay" in violation.lower()
            and "checksum" in violation.lower()
            for violation in result.violations
        )

    def test_rejects_single_altered_percentile_value(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir, _, _, _, _, output_dir = _current_layout_family(tmp_path)
        payload_path = overlay_shard_path(output_dir, "shard_00000")
        values = load_overlay_shard(payload_path)[
            TensorKey.FEATURE_PERCENTILES.value
        ].clone()
        values[0] = (int(values[0].item()) + 1) % 256
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES.value: values},
            payload_path,
        )

        result = vf.validate_feature_directory(feature_dir, n_sae_features=4)

        assert not result.valid
        assert any(
            "percentile values don't match recomputation" in violation
            for violation in result.violations
        )


class TestSV010StatisticsShardDomain:
    def test_accepts_default_final_shard_exclusion(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shards = [
            _write_base_shard(feature_dir, shard_index)
            for shard_index in range(3)
        ]
        _write_statistics(feature_dir, shards[:-1])

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations

    def test_accepts_all_shards_statistics(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shards = [
            _write_base_shard(feature_dir, shard_index)
            for shard_index in range(3)
        ]
        _write_statistics(feature_dir, shards)

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations

    def test_one_shard_exemption_uses_the_only_shard(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shard = _write_base_shard(feature_dir, 0)
        _write_statistics(feature_dir, [shard])

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations

    def test_mutating_included_shard_changes_validated_totals(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shards = [
            _write_base_shard(feature_dir, shard_index)
            for shard_index in range(3)
        ]
        _write_statistics(feature_dir, shards[:-1])
        _write_base_shard(
            feature_dir,
            0,
            n_docs=2,
            segs_per_doc=3,
        )

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert not result.valid
        assert any(
            "Statistics n_docs" in violation
            or "Statistics n_segments" in violation
            for violation in result.violations
        )

    def test_mutating_excluded_shard_does_not_change_statistics_totals(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        shards = [
            _write_base_shard(feature_dir, shard_index)
            for shard_index in range(3)
        ]
        _write_statistics(feature_dir, shards[:-1])
        _write_base_shard(
            feature_dir,
            2,
            n_docs=2,
            segs_per_doc=3,
        )

        result = vf.validate_feature_directory(
            feature_dir,
            n_sae_features=4,
        )

        assert result.valid, result.violations
