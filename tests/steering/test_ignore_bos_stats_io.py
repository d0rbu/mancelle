"""ignore_bos_features plain-bool stats-file field.

Contract: always written in fresh stats files; a stats file written before
the field existed simply lacks the key and loads as ``False``; any non
true/false value is loud (``ValueError``). No nullable/"unknown" state.
"""

import json
import struct

import pytest

from steering.segmentation import SegmentMode
from steering.stats_io import (
    _parse_bool_metadata,
    _read_metadata,
    load_statistics,
    save_statistics,
)
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    SegmentFeatureStats,
    StorageStats,
)


def _stats(ignore_bos: bool = False) -> ExtractionStatistics:
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=128,
            context_window=2048,
            layer=8,
            n_sae_features=16,
            generated_at="2026-05-17T00:00:00+00:00",
        ),
        corpus=CorpusStats(1, 1, 1, 1.0, 1),
        features=SegmentFeatureStats(
            n_segments=1, n_features_seen=1, mean_active_per_segment=1.0,
            median_active_per_segment=1, p90_active_per_segment=1,
            max_active_per_segment=1, min_active_per_segment=1,
            n_total_sae_features=16, pct_features_active=1.0,
            count_percentiles={}, magnitude_stats={},
            active_per_segment_histogram={}, active_per_segment_percentiles={},
        ),
        storage=StorageStats(1, 1.0, 1.0),
        per_feature_stats={},
        extrapolation_10b=Extrapolation(1, 1, 1.0, 1.0),
        ignore_bos_features=ignore_bos,
    )


def _rewrite_stats_metadata(path, mutate) -> None:
    """Rewrite a plain-safetensors stats file's __metadata__ via ``mutate``."""
    raw = path.read_bytes()
    header_len = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + header_len])
    mutate(header.setdefault("__metadata__", {}))
    new_hdr = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(new_hdr)) + new_hdr + raw[8 + header_len :])


def test_parse_bool_metadata():
    assert _parse_bool_metadata("true") is True
    assert _parse_bool_metadata("false") is False
    for bad in ("garbage", "True", "1", "", "TRUE", "yes"):
        with pytest.raises(ValueError, match="expected 'true' or 'false'"):
            _parse_bool_metadata(bad)


def test_dataclass_default_is_false():
    # Construct WITHOUT passing ignore_bos_features → plain-bool default.
    s = ExtractionStatistics(
        metadata=_stats().metadata,
        corpus=_stats().corpus,
        features=_stats().features,
        storage=_stats().storage,
        per_feature_stats={},
        extrapolation_10b=_stats().extrapolation_10b,
    )
    assert s.ignore_bos_features is False


@pytest.mark.parametrize("val", [True, False])
def test_roundtrip_concrete(tmp_path, val):
    p = save_statistics(_stats(val), tmp_path / "feature_statistics.safetensors")
    assert load_statistics(p).ignore_bos_features is val
    # Always written (plain bool, like every other scalar metadata).
    assert _read_metadata(p)["ignore_bos_features"] == ("true" if val else "false")


def test_pre_existing_file_without_key_loads_false(tmp_path):
    """A stats file written before this field existed lacks the key and must
    load as False (the pre-feature corpus had ignore_bos off)."""
    p = save_statistics(_stats(True), tmp_path / "feature_statistics.safetensors")
    _rewrite_stats_metadata(p, lambda m: m.pop("ignore_bos_features", None))
    assert "ignore_bos_features" not in _read_metadata(p)
    assert load_statistics(p).ignore_bos_features is False


def test_garbage_value_in_stats_file_raises(tmp_path):
    """A stats file whose ignore_bos_features is neither 'true' nor 'false'
    must raise loudly from load_statistics, never silently coerce."""
    p = save_statistics(_stats(False), tmp_path / "feature_statistics.safetensors")
    _rewrite_stats_metadata(
        p, lambda m: m.__setitem__("ignore_bos_features", "garbage_value")
    )
    with pytest.raises(ValueError, match="expected 'true' or 'false'"):
        load_statistics(p)
