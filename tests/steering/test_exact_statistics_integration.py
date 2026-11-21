"""Integration and production-data gates for exact feature statistics.

The retired sorted-spill implementation lives under ``tests`` as an independent
reference. These tests prove that the production streaming route has no legacy
spill surface and that its complete artifact remains identical to the eager
implementation under frozen volatile metadata.
"""

from __future__ import annotations

import gc
import json
import math
import os
import weakref
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
import torch as t
from safetensors import safe_open
from safetensors.torch import load_file

import steering.compute_statistics as statistics_module
from steering.compute_statistics import (
    PERCENTILES,
    _canonical_feature_width,
    compute_per_feature_percentiles,
    compute_statistics,
    exact_pass1,
    load_all_shards,
)
from steering.conditional.freq_blacklist import compute_activation_frequency_blacklist
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.exact_statistics import (
    DEFAULT_COUNT_CHUNK_SIZE,
    ExactMagnitudeCounts,
    ExactStatisticsError,
    encode_f16_count_keys,
)
from steering.feature_io import glob_features, load_features, save_features
from steering.stats_io import load_statistics
from tests.steering._compute_statistics_reference import (
    merge_per_feature_percentiles,
    optimized_pass1,
)

_CPU_DEVICE = t.device("cpu")


class _FrozenDateTime:
    @classmethod
    def now(cls, tz=None) -> datetime:
        return datetime(2026, 8, 11, 12, 34, 56, 789, tzinfo=tz)


class _WeakrefableShard(dict):
    """Dictionary subclass used to observe exact shard-object lifetimes."""


def _shard(
    feature_ids: list[int],
    feature_mags: list[float],
    offsets: list[int],
    *,
    doc_token_counts: list[int] | None = None,
    segment_size: int = 4,
) -> dict:
    n_segments = len(offsets) - 1
    if doc_token_counts is None:
        doc_token_counts = [n_segments * segment_size]
    return {
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "offsets": t.tensor(offsets, dtype=t.int64),
        "n_segments": n_segments,
        "n_docs": len(doc_token_counts),
        "doc_ids": t.zeros(n_segments, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "segment_indices": t.arange(n_segments, dtype=t.int32),
        "segment_positions": t.arange(n_segments, dtype=t.int32) * segment_size,
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_window": 1024,
        "pooling": "max_pool",
        "layer": 8,
    }


def _write_shards(directory: Path) -> list[Path]:
    shards = [
        _shard(
            [0, 1, 1, 3, 0, 2, 3, 3],
            [1.0, 2.0, 4.0, 8.0, 1.5, 0.5, 16.0, 8.0],
            [0, 4, 8],
            doc_token_counts=[7],
        ),
        _shard(
            [3, 2, 1, 0, 3, 2, 1, 0, 0],
            [32.0, 1.0, 8.0, 3.0, 16.0, 2.0, 2.0, 6.0, 12.0],
            [0, 3, 6, 9],
            doc_token_counts=[11],
        ),
        _shard(
            [2, 2, 3, 1, 0, 3, 1, 0],
            [4.0, 8.0, 64.0, 16.0, 24.0, 128.0, 32.0, 48.0],
            [0, 2, 5, 8],
            doc_token_counts=[10],
        ),
    ]
    paths = []
    for index, shard in enumerate(shards):
        paths.append(save_features(shard, directory / f"shard_{index:05d}_features"))
    return paths


def _expanded(counter: Counter[int | float]) -> list[int | float]:
    values: list[int | float] = []
    for value, multiplicity in counter.items():
        values.extend([value] * multiplicity)
    return values


def _freeze_volatile_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(statistics_module, "datetime", _FrozenDateTime)
    monkeypatch.setattr(statistics_module.time, "monotonic", lambda: 1000.0)


def _assert_safetensors_exact(actual: Path, expected: Path) -> None:
    """Compare the artifact contract despite nondeterministic header key order."""
    with safe_open(actual, framework="pt") as actual_file:
        actual_metadata = actual_file.metadata()
        actual_keys = set(actual_file.keys())
    with safe_open(expected, framework="pt") as expected_file:
        expected_metadata = expected_file.metadata()
        expected_keys = set(expected_file.keys())
    assert actual_metadata == expected_metadata
    assert actual_keys == expected_keys

    actual_tensors = load_file(actual)
    expected_tensors = load_file(expected)
    for key in sorted(expected_keys):
        assert actual_tensors[key].dtype == expected_tensors[key].dtype, key
        assert actual_tensors[key].shape == expected_tensors[key].shape, key
        assert t.equal(actual_tensors[key], expected_tensors[key]), key


_CUDA_MOMENT_TENSOR_KEYS = frozenset(
    {"mean_mag", "mean_mag_incl_zeros", "std_mag"}
)
_CUDA_MOMENT_RTOL = 1e-6
_CUDA_MOMENT_ATOL = 1e-6


def _assert_nested_cuda_moments_close(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        actual_dict = cast(dict[str, object], actual)
        expected_dict = cast(dict[str, object], expected)
        assert set(actual_dict) == set(expected_dict), path
        for key in sorted(expected_dict):
            _assert_nested_cuda_moments_close(
                actual_dict[key],
                expected_dict[key],
                f"{path}.{key}",
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        assert float(actual) == pytest.approx(
            float(expected),
            rel=_CUDA_MOMENT_RTOL,
            abs=_CUDA_MOMENT_ATOL,
        ), path
        return
    assert actual == expected, path


def _assert_safetensors_cuda_close(actual: Path, expected: Path) -> None:
    """Require exact artifacts except for established CUDA moment tolerances."""
    with safe_open(actual, framework="pt") as actual_file:
        actual_metadata = actual_file.metadata()
        actual_keys = set(actual_file.keys())
    with safe_open(expected, framework="pt") as expected_file:
        expected_metadata = expected_file.metadata()
        expected_keys = set(expected_file.keys())

    assert actual_metadata is not None
    assert expected_metadata is not None
    assert set(actual_metadata) == set(expected_metadata)
    assert actual_keys == expected_keys
    assert _CUDA_MOMENT_TENSOR_KEYS <= actual_keys

    for key in sorted(expected_metadata):
        if key == "magnitude_stats":
            _assert_nested_cuda_moments_close(
                json.loads(actual_metadata[key]),
                json.loads(expected_metadata[key]),
                "metadata.magnitude_stats",
            )
        else:
            assert actual_metadata[key] == expected_metadata[key], key

    actual_tensors = load_file(actual)
    expected_tensors = load_file(expected)
    for key in sorted(expected_keys):
        assert actual_tensors[key].dtype == expected_tensors[key].dtype, key
        assert actual_tensors[key].shape == expected_tensors[key].shape, key
        if key in _CUDA_MOMENT_TENSOR_KEYS:
            t.testing.assert_close(
                actual_tensors[key],
                expected_tensors[key],
                rtol=_CUDA_MOMENT_RTOL,
                atol=_CUDA_MOMENT_ATOL,
                msg=lambda message, key=key: f"{key}: {message}",
            )
        else:
            assert t.equal(actual_tensors[key], expected_tensors[key]), key


def _real_feature_width(feature_dir: Path) -> int:
    """Resolve the canonical width for a base or derived real feature directory."""
    for candidate in (feature_dir, *feature_dir.parents):
        width = _canonical_feature_width(candidate)
        if width is not None:
            return width
    pytest.fail(f"real shard directory has no canonical feature ancestor: {feature_dir}")


class TestExactPassOne:
    def test_requires_resolved_torch_device(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="torch.device"):
            exact_pass1(
                str(tmp_path),
                n_sae_features=4,
                device=cast(t.device, "cpu"),
            )

    def test_reads_every_shard_body_exactly_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        calls: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            calls.append(Path(path))
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        result = exact_pass1(str(tmp_path), n_sae_features=4)

        assert calls == paths
        assert result.shard_paths == paths
        assert result.n_shards == len(paths)

    def test_routes_resolved_cpu_to_every_exact_count_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_add = ExactMagnitudeCounts.add
        observed: list[t.device] = []

        def tracked_add(
            reducer: ExactMagnitudeCounts,
            feature_ids: t.Tensor,
            feature_mags: t.Tensor,
            *,
            chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
            reduction_device: t.device = _CPU_DEVICE,
        ) -> None:
            observed.append(reduction_device)
            real_add(
                reducer,
                feature_ids,
                feature_mags,
                chunk_size=chunk_size,
                reduction_device=reduction_device,
            )

        monkeypatch.setattr(ExactMagnitudeCounts, "add", tracked_add)
        exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert observed == [t.device("cpu")] * len(paths)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_routes_resolved_cuda_to_every_exact_count_update(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_add = ExactMagnitudeCounts.add
        observed: list[t.device] = []

        def tracked_add(
            reducer: ExactMagnitudeCounts,
            feature_ids: t.Tensor,
            feature_mags: t.Tensor,
            *,
            chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
            reduction_device: t.device = _CPU_DEVICE,
        ) -> None:
            observed.append(reduction_device)
            real_add(
                reducer,
                feature_ids,
                feature_mags,
                chunk_size=chunk_size,
                reduction_device=reduction_device,
            )

        monkeypatch.setattr(ExactMagnitudeCounts, "add", tracked_add)
        exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cuda"))

        assert observed == [t.device("cuda")] * len(paths)

    def test_releases_complete_shard_before_loading_the_next(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        prior_refs: list[weakref.ReferenceType[object]] = []
        calls: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            gc.collect()
            assert all(ref() is None for ref in prior_refs), (
                "the previous decompressed shard survived until the next load"
            )
            data = _WeakrefableShard(real_load(path))
            calls.append(Path(path))
            prior_refs[:] = [
                weakref.ref(data),
                weakref.ref(data["feature_ids"]),
                weakref.ref(data["feature_mags"]),
            ]
            return data

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        result = exact_pass1(str(tmp_path), n_sae_features=4)

        gc.collect()
        assert calls == paths
        assert all(ref() is None for ref in prior_refs), (
            "the final decompressed shard survived exact_pass1() return"
        )
        assert result.n_shards == len(paths)

    def test_aggregate_moments_equal_old_streaming_reference_exactly(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        old, spills, old_sample, old_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "old-spills",
            n_sae_features=4,
            device="cpu",
        )
        new = exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert t.equal(new.acc.occurrence, old.acc.occurrence)
        assert t.equal(new.acc.mag_sum, old.acc.mag_sum)
        assert t.equal(new.acc.mag_sum_sq, old.acc.mag_sum_sq)
        assert t.equal(new.acc.mag_min, old.acc.mag_min)
        assert t.equal(new.acc.mag_max, old.acc.mag_max)
        assert new.total_docs == old.total_docs
        assert new.total_tokens == old.total_tokens
        assert new.total_storage == old.total_storage
        assert new.n_shards == old.n_shards
        assert new.n_sae_features == old.n_sae_features
        assert t.equal(new.mag_subsample, old_sample)
        assert new.global_mag_min == old_min
        assert sum(path.mags_path.stat().st_size for path in spills if path.mags_path) > 0

    def test_weighted_segment_state_expands_to_old_lists_exactly(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        old, _, _, _ = optimized_pass1(
            str(tmp_path),
            tmp_path / "old-spills",
            n_sae_features=4,
        )
        new = exact_pass1(str(tmp_path), n_sae_features=4)

        assert sorted(_expanded(new.seg_acc.features_per_segment.counts)) == sorted(
            old.seg_acc.features_per_segment
        )
        assert sorted(_expanded(new.seg_acc.tokens_per_doc.counts)) == sorted(
            old.seg_acc.tokens_per_doc
        )
        assert sorted(_expanded(new.seg_acc.segment_lengths.counts)) == sorted(
            old.seg_acc.segment_lengths
        )
        assert sorted(_expanded(new.seg_acc.features_per_token_ratios.counts)) == sorted(
            old.seg_acc.features_per_token_ratios
        )
        assert new.seg_acc.features_per_segment.population_size == len(
            old.seg_acc.features_per_segment
        )
        assert new.seg_acc.tokens_per_doc.population_size == len(old.seg_acc.tokens_per_doc)

    def test_count_table_total_and_every_cell_match_loaded_shards(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        result = exact_pass1(str(tmp_path), n_sae_features=4)
        loaded = load_all_shards(str(tmp_path))

        assert result.magnitude_counts is not None
        assert result.magnitude_counts.total_count == len(loaded.feature_ids)
        assert result.magnitude_counts.total_count == int(result.acc.occurrence.sum().item())

        expected_keys = encode_f16_count_keys(
            loaded.feature_ids,
            loaded.feature_mags,
            n_sae_features=4,
        )
        expected = Counter(expected_keys.tolist())
        actual_flat = result.magnitude_counts.counts.view(-1)
        assert t.count_nonzero(actual_flat).item() == len(expected)
        for key, count in expected.items():
            assert actual_flat[key].item() == count

    def test_finalized_detail_equals_old_merge_reference_without_tolerance(
        self, tmp_path: Path
    ) -> None:
        _write_shards(tmp_path)
        old, spills, old_sample, old_min = optimized_pass1(
            str(tmp_path),
            tmp_path / "old-spills",
            n_sae_features=4,
        )
        reference, reference_edges = merge_per_feature_percentiles(
            spills,
            old.n_sae_features,
            old.acc.occurrence,
            old_sample,
            old_min,
        )
        new = exact_pass1(str(tmp_path), n_sae_features=4)
        assert new.magnitude_counts is not None
        actual, actual_edges = new.magnitude_counts.finalize(
            occurrence=new.acc.occurrence,
            mag_subsample=new.mag_subsample,
            global_mag_min=new.global_mag_min,
            percentiles=PERCENTILES,
        )

        assert actual_edges == reference_edges
        assert set(actual) == set(reference)
        for feature_id, expected in reference.items():
            assert actual[feature_id].percentiles == expected.percentiles
            assert actual[feature_id].median == expected.median
            assert actual[feature_id].histogram == expected.histogram

    def test_subsample_step_and_per_shard_restart_match_legacy_rule(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        monkeypatch.setattr(statistics_module, "P99_SUBSAMPLE_TARGET", 5)
        result = exact_pass1(str(tmp_path), n_sae_features=4)

        first = load_features(paths[0])["feature_mags"]
        step = max(1, len(first) * len(paths) // 5)
        expected = t.cat([load_features(path)["feature_mags"][::step] for path in paths])
        assert step == 4
        assert t.equal(result.mag_subsample, expected)

    def test_exclude_last_shard_excludes_all_of_its_state(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        result = exact_pass1(
            str(tmp_path),
            n_sae_features=4,
            exclude_last_shard=True,
        )
        expected_activations = sum(len(load_features(path)["feature_ids"]) for path in paths[:-1])
        expected_tokens = sum(
            int(load_features(path)["doc_token_counts"].sum().item()) for path in paths[:-1]
        )

        assert result.shard_paths == paths[:-1]
        assert result.n_shards == 2
        assert result.magnitude_counts is not None
        assert result.magnitude_counts.total_count == expected_activations
        assert int(result.acc.occurrence.sum().item()) == expected_activations
        assert result.total_tokens == expected_tokens
        assert result.total_storage == sum(path.stat().st_size for path in paths[:-1])

    def test_skip_detail_never_allocates_count_table(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)

        def forbidden_allocate(*args, **kwargs):
            raise AssertionError("count table must not be allocated")

        monkeypatch.setattr(ExactMagnitudeCounts, "allocate", forbidden_allocate)
        result = exact_pass1(
            str(tmp_path),
            n_sae_features=4,
            skip_per_feature_detail=True,
        )

        assert result.magnitude_counts is None
        assert result.mag_subsample.numel() == 0
        assert int(result.acc.occurrence.sum().item()) == 25

    def test_explicit_memory_admission_precedes_first_body_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        events: list[str] = []

        def rejected_allocate(*args, **kwargs):
            events.append("allocate")
            raise ExactStatisticsError("deliberate admission rejection")

        def forbidden_load(*args, **kwargs):
            events.append("load")
            raise AssertionError("body load must follow admission")

        monkeypatch.setattr(ExactMagnitudeCounts, "allocate", rejected_allocate)
        monkeypatch.setattr(statistics_module, "load_features", forbidden_load)
        with pytest.raises(ExactStatisticsError, match="admission rejection"):
            exact_pass1(str(tmp_path), n_sae_features=65_536)
        assert events == ["allocate"]

    def test_noncanonical_auto_width_discovers_later_higher_feature_id(
        self, tmp_path: Path
    ) -> None:
        save_features(
            _shard([0, 0], [1.0, 2.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        save_features(
            _shard([0, 3], [3.0, 4.0], [0, 2]),
            tmp_path / "shard_00001_features",
        )

        result = exact_pass1(str(tmp_path))
        assert result.n_sae_features == 4
        assert result.acc.occurrence.tolist() == [3, 0, 0, 1]
        assert result.magnitude_counts is not None
        assert result.magnitude_counts.total_count == 4

    def test_explicit_width_handles_later_higher_feature_id(self, tmp_path: Path) -> None:
        save_features(
            _shard([0, 0], [1.0, 2.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        save_features(
            _shard([0, 3], [3.0, 4.0], [0, 2]),
            tmp_path / "shard_00001_features",
        )
        result = exact_pass1(str(tmp_path), n_sae_features=4)
        assert result.n_sae_features == 4
        assert result.acc.occurrence.tolist() == [3, 0, 0, 1]
        assert result.magnitude_counts is not None
        assert result.magnitude_counts.total_count == 4

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (
                "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base",
                65_536,
            ),
            (
                "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/no_bos",
                16_384,
            ),
            ("/tmp/noncanonical", None),
        ],
    )
    def test_canonical_width_resolution_never_reads_or_allocates(
        self,
        path: str,
        expected: int | None,
    ) -> None:
        assert _canonical_feature_width(path) == expected


class TestExactPassContracts:
    def test_noncanonical_compatibility_prepass_reads_each_body_twice(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        calls: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            calls.append(Path(path))
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        result = exact_pass1(str(tmp_path))

        assert result.n_sae_features == 4
        assert calls == paths + paths

    def test_canonical_skip_detail_resolves_width_and_reads_each_body_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canonical = (
            tmp_path / "s4" / "l8" / "a_res" / "p_max_pool" / "f16k" / "m_gemma-2-2b" / "base"
        )
        canonical.mkdir(parents=True)
        paths = _write_shards(canonical)
        real_load = statistics_module.load_features
        calls: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            calls.append(Path(path))
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        result = exact_pass1(str(canonical), skip_per_feature_detail=True)

        assert result.n_sae_features == 16_384
        assert len(result.acc.occurrence) == 16_384
        assert result.magnitude_counts is None
        assert calls == paths

    @pytest.mark.parametrize("width", [0, -1, -65_536])
    def test_skip_detail_still_rejects_invalid_explicit_width(
        self,
        tmp_path: Path,
        width: int,
    ) -> None:
        _write_shards(tmp_path)
        with pytest.raises(ExactStatisticsError, match="must be positive"):
            exact_pass1(
                str(tmp_path),
                n_sae_features=width,
                skip_per_feature_detail=True,
            )

    @pytest.mark.parametrize("width", [True, 4.0, "4"])
    def test_skip_detail_rejects_noninteger_explicit_width(
        self,
        tmp_path: Path,
        width: object,
    ) -> None:
        _write_shards(tmp_path)
        with pytest.raises(TypeError, match="n_sae_features must be an int"):
            exact_pass1(
                str(tmp_path),
                n_sae_features=width,  # ty: ignore[invalid-argument-type]
                skip_per_feature_detail=True,
            )

    def test_explicit_width_rejects_high_id_even_when_detail_is_skipped(
        self, tmp_path: Path
    ) -> None:
        save_features(
            _shard([0, 4], [1.0, 2.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        with pytest.raises(ExactStatisticsError, match=r"max=4, valid=\[0, 4\)"):
            exact_pass1(
                str(tmp_path),
                n_sae_features=4,
                skip_per_feature_detail=True,
            )

    def test_explicit_width_error_does_not_mutate_source_shard(self, tmp_path: Path) -> None:
        path = save_features(
            _shard([0, 4], [1.0, 2.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        before = path.read_bytes()
        with pytest.raises(ExactStatisticsError, match="outside the declared"):
            exact_pass1(str(tmp_path), n_sae_features=4)
        assert path.read_bytes() == before

    def test_single_shard_is_not_excluded_by_exclude_last(self, tmp_path: Path) -> None:
        path = save_features(
            _shard([0, 1], [1.0, 2.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        result = exact_pass1(
            str(tmp_path),
            n_sae_features=2,
            exclude_last_shard=True,
        )
        assert result.shard_paths == [path]
        assert result.n_shards == 1
        assert result.magnitude_counts is not None
        assert result.magnitude_counts.total_count == 2

    def test_metadata_is_copied_exactly_from_first_shard(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        result = exact_pass1(str(tmp_path), n_sae_features=4)
        assert result.metadata.pooling == "max_pool"
        assert result.metadata.segment_mode.value == "fixed_tokens"
        assert result.metadata.segment_size == 4
        assert result.metadata.context_window == 1024
        assert result.metadata.layer == 8

    def test_hand_computed_moment_accumulators_are_exact(self, tmp_path: Path) -> None:
        save_features(
            _shard([0, 0, 1, 1], [1.0, 3.0, 2.0, 4.0], [0, 2, 4]),
            tmp_path / "shard_00000_features",
        )
        result = exact_pass1(str(tmp_path), n_sae_features=3)
        assert result.acc.occurrence.tolist() == [2, 2, 0]
        assert result.acc.mag_sum.tolist() == [4.0, 6.0, 0.0]
        assert result.acc.mag_sum_sq.tolist() == [10.0, 20.0, 0.0]
        assert result.acc.mag_min[:2].tolist() == [1.0, 2.0]
        assert result.acc.mag_max[:2].tolist() == [3.0, 4.0]
        assert result.acc.mag_min[2].item() == math.inf
        assert result.acc.mag_max[2].item() == -math.inf

    def test_global_minimum_uses_all_shards_not_subsample(self, tmp_path: Path) -> None:
        save_features(
            _shard([0, 0], [4.0, 8.0], [0, 2]),
            tmp_path / "shard_00000_features",
        )
        save_features(
            _shard([0, 1], [2**-10, 16.0], [0, 2]),
            tmp_path / "shard_00001_features",
        )
        result = exact_pass1(str(tmp_path), n_sae_features=2)
        assert result.global_mag_min == 2**-10
        assert result.global_mag_min == result.acc.mag_min[result.acc.occurrence > 0].min().item()

    def test_segment_distributions_have_exact_population_and_sums(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        result = exact_pass1(str(tmp_path), n_sae_features=4)
        segments = result.seg_acc.features_per_segment
        tokens = result.seg_acc.tokens_per_doc
        lengths = result.seg_acc.segment_lengths
        ratios = result.seg_acc.features_per_token_ratios
        assert segments.population_size == 8
        assert segments.exact_integer_sum() == 25
        assert tokens.population_size == 3
        assert tokens.exact_integer_sum() == 28
        assert lengths.population_size == 8
        assert lengths.exact_integer_sum() == 28
        assert ratios.population_size == 8
        assert sum(value * count for value, count in ratios.counts.items()) == 91 / 12

    def test_no_shards_is_a_typed_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No feature shard files"):
            exact_pass1(str(tmp_path), n_sae_features=4)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_moment_state_returns_to_cpu_before_result(self, tmp_path: Path) -> None:
        _write_shards(tmp_path)
        result = exact_pass1(
            str(tmp_path),
            n_sae_features=4,
            device=t.device("cuda"),
            gpu_chunk_size=3,
        )
        assert result.acc.occurrence.device.type == "cpu"
        assert result.acc.mag_sum.device.type == "cpu"
        assert result.acc.mag_sum_sq.device.type == "cpu"
        assert result.acc.mag_min.device.type == "cpu"
        assert result.acc.mag_max.device.type == "cpu"
        assert result.magnitude_counts is not None
        assert result.magnitude_counts.counts.device.type == "cpu"


class TestProductionRoutingAndArtifactParity:
    def test_complete_streaming_command_reads_each_body_exactly_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        calls: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            calls.append(Path(path))
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        compute_statistics(
            str(tmp_path),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )

        assert calls == paths

    def test_canonical_width_is_identical_in_eager_and_streaming_modes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canonical = (
            tmp_path / "s4" / "l8" / "a_res" / "p_max_pool" / "f16k" / "m_gemma-2-2b" / "base"
        )
        canonical.mkdir(parents=True)
        _write_shards(canonical)
        _freeze_volatile_metadata(monkeypatch)

        eager = compute_statistics(
            str(canonical),
            str(tmp_path / "canonical-eager"),
            streaming=False,
            skip_per_feature_detail=True,
            allow_degenerate_statistics=True,
        )
        streaming = compute_statistics(
            str(canonical),
            str(tmp_path / "canonical-streaming"),
            streaming=True,
            skip_per_feature_detail=True,
            allow_degenerate_statistics=True,
        )

        assert eager.metadata.n_sae_features == 16_384
        assert streaming.metadata.n_sae_features == 16_384
        assert eager.features.n_total_sae_features == 16_384
        assert streaming.features.n_total_sae_features == 16_384
        assert eager.features.pct_features_active == streaming.features.pct_features_active
        assert eager == streaming
        _assert_safetensors_exact(
            tmp_path / "canonical-streaming.safetensors",
            tmp_path / "canonical-eager.safetensors",
        )

    def test_activation_filtering_preserves_implicit_canonical_width(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canonical = (
            tmp_path / "s4" / "l8" / "a_res" / "p_max_pool" / "f16k" / "m_gemma-2-2b" / "base"
        )
        canonical.mkdir(parents=True)
        _write_shards(canonical)
        _freeze_volatile_metadata(monkeypatch)

        implicit = compute_statistics(
            str(canonical),
            str(tmp_path / "implicit-width"),
            activation_percentile=0.5,
            streaming=False,
            skip_per_feature_detail=True,
            allow_degenerate_statistics=True,
        )
        explicit = compute_statistics(
            str(canonical),
            str(tmp_path / "explicit-width"),
            n_sae_features=16_384,
            activation_percentile=0.5,
            streaming=False,
            skip_per_feature_detail=True,
            allow_degenerate_statistics=True,
        )

        assert implicit.metadata.n_sae_features == 16_384
        assert implicit.features.n_total_sae_features == 16_384
        assert implicit == explicit
        _assert_safetensors_exact(
            tmp_path / "implicit-width.safetensors",
            tmp_path / "explicit-width.safetensors",
        )

    def test_complete_streaming_artifact_is_exactly_equal_to_eager_reference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _freeze_volatile_metadata(monkeypatch)

        eager = compute_statistics(
            str(tmp_path),
            str(tmp_path / "eager"),
            n_sae_features=4,
            streaming=False,
            device="cpu",
        )
        streaming = compute_statistics(
            str(tmp_path),
            str(tmp_path / "streaming"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )

        eager_path = tmp_path / "eager.safetensors"
        streaming_path = tmp_path / "streaming.safetensors"
        assert streaming == eager
        _assert_safetensors_exact(streaming_path, eager_path)
        assert load_statistics(streaming_path) == load_statistics(eager_path)

    def test_downstream_percentile_table_checksum_and_overlay_values_are_exact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _freeze_volatile_metadata(monkeypatch)
        eager = compute_statistics(
            str(tmp_path),
            str(tmp_path / "eager"),
            n_sae_features=4,
            streaming=False,
            device="cpu",
        )
        streaming = compute_statistics(
            str(tmp_path),
            str(tmp_path / "streaming"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )

        eager_table = build_percentile_table(eager.per_feature_stats, 4)
        streaming_table = build_percentile_table(streaming.per_feature_stats, 4)
        assert eager_table.dtype == streaming_table.dtype == t.float64
        assert eager_table.shape == streaming_table.shape == (4, 99)
        assert t.equal(streaming_table, eager_table)
        assert compute_percentile_checksum(streaming_table) == (
            compute_percentile_checksum(eager_table)
        )

        loaded = load_all_shards(str(tmp_path))
        eager_scores = magnitude_to_percentile(
            loaded.feature_ids,
            loaded.feature_mags,
            eager_table,
        )
        streaming_scores = magnitude_to_percentile(
            loaded.feature_ids,
            loaded.feature_mags,
            streaming_table,
        )
        assert t.equal(streaming_scores, eager_scores)
        assert t.equal(
            percentile_to_uint8(streaming_scores),
            percentile_to_uint8(eager_scores),
        )

    def test_downstream_frequency_blacklist_is_exactly_unchanged(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _freeze_volatile_metadata(monkeypatch)
        eager_dir = tmp_path / "eager-output"
        streaming_dir = tmp_path / "streaming-output"
        compute_statistics(
            str(tmp_path),
            str(eager_dir / "feature_statistics"),
            n_sae_features=4,
            streaming=False,
            device="cpu",
        )
        compute_statistics(
            str(tmp_path),
            str(streaming_dir / "feature_statistics"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )

        eager_blacklist = compute_activation_frequency_blacklist(
            eager_dir,
            4,
            freq_min=0.1,
            freq_max=0.8,
        )
        streaming_blacklist = compute_activation_frequency_blacklist(
            streaming_dir,
            4,
            freq_min=0.1,
            freq_max=0.8,
        )
        assert streaming_blacklist == eager_blacklist
        assert streaming_blacklist.ids == (0, 3)

    def test_production_streaming_route_cannot_call_any_spill_primitive(
        self,
        tmp_path: Path,
    ) -> None:
        _write_shards(tmp_path)
        for retired_name in (
            "optimized_pass1",
            "_spill_sorted_mags",
            "merge_per_feature_percentiles",
            "vectorized_pass2_percentiles",
        ):
            assert not hasattr(statistics_module, retired_name)

        result = compute_statistics(
            str(tmp_path),
            str(tmp_path / "stats"),
            n_sae_features=4,
            streaming=True,
        )
        assert result.features.n_features_seen == 4
        assert (tmp_path / "stats.safetensors").is_file()

    def test_skip_detail_production_route_has_no_count_table_or_percentile_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)

        def forbidden_allocate(*args, **kwargs):
            raise AssertionError("count table must not be allocated")

        monkeypatch.setattr(ExactMagnitudeCounts, "allocate", forbidden_allocate)
        result = compute_statistics(
            str(tmp_path),
            str(tmp_path / "stats"),
            n_sae_features=4,
            streaming=True,
            skip_per_feature_detail=True,
        )

        assert result.features.mag_histogram_bin_edges is None
        for feature in result.per_feature_stats.values():
            assert feature.percentiles == {}
            assert feature.median_mag == 0.0
            assert feature.histogram is None

    def test_invalid_exact_magnitude_leaves_no_output_artifact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features

        def corrupt_load(path: str | Path) -> dict:
            data = real_load(path)
            if Path(path) == paths[0]:
                data["feature_mags"] = data["feature_mags"].clone()
                data["feature_mags"][0] = 0.1
            return data

        monkeypatch.setattr(statistics_module, "load_features", corrupt_load)
        output = tmp_path / "stats.safetensors"
        with pytest.raises(ExactStatisticsError, match="non-float16-exact"):
            compute_statistics(
                str(tmp_path),
                str(output),
                n_sae_features=4,
                streaming=True,
            )
        assert not output.exists()

    def test_mid_pass_read_failure_leaves_no_output_or_spill_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        calls = 0

        def fail_second(path: str | Path) -> dict:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("deliberate shard read failure")
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", fail_second)
        output = tmp_path / "stats.safetensors"
        with pytest.raises(OSError, match="deliberate shard read failure"):
            compute_statistics(
                str(tmp_path),
                str(output),
                n_sae_features=4,
                streaming=True,
            )
        assert calls == 2
        assert not output.exists()
        assert list(tmp_path.glob("stats_sorted_*")) == []
        assert all(path.exists() for path in paths)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_full_artifact_cpu_and_cuda_routes_are_exactly_equal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _freeze_volatile_metadata(monkeypatch)

        compute_statistics(
            str(tmp_path),
            str(tmp_path / "cpu"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )
        compute_statistics(
            str(tmp_path),
            str(tmp_path / "cuda"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
        )

        _assert_safetensors_exact(
            tmp_path / "cuda.safetensors",
            tmp_path / "cpu.safetensors",
        )

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_full_cuda_artifact_matches_prechange_cpu_exact_reduction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _freeze_volatile_metadata(monkeypatch)
        real_add = ExactMagnitudeCounts.add

        def force_cpu_exact_add(
            reducer: ExactMagnitudeCounts,
            feature_ids: t.Tensor,
            feature_mags: t.Tensor,
            *,
            chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
            reduction_device: t.device = _CPU_DEVICE,
        ) -> None:
            del reduction_device
            real_add(
                reducer,
                feature_ids,
                feature_mags,
                chunk_size=chunk_size,
                reduction_device=t.device("cpu"),
            )

        with monkeypatch.context() as reference:
            reference.setattr(ExactMagnitudeCounts, "add", force_cpu_exact_add)
            reference.setattr(
                statistics_module,
                "_should_prefetch_exact_loads",
                lambda **_kwargs: False,
            )
            compute_statistics(
                str(tmp_path),
                str(tmp_path / "cpu-exact"),
                n_sae_features=4,
                streaming=True,
                device="cuda",
            )
        compute_statistics(
            str(tmp_path),
            str(tmp_path / "cuda-exact"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
        )

        _assert_safetensors_cuda_close(
            tmp_path / "cuda-exact.safetensors",
            tmp_path / "cpu-exact.safetensors",
        )


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
def test_full_real_artifact_cuda_exact_matches_prechange_cpu_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare complete artifacts on untruncated production feature shards."""
    configured = os.environ.get("FEATURE_STATISTICS_REAL_SHARD_DIR")
    if configured is None:
        pytest.skip("FEATURE_STATISTICS_REAL_SHARD_DIR is not configured")
    feature_dir = Path(configured)
    shard_paths = glob_features(feature_dir)
    if not shard_paths:
        pytest.fail(f"no raw feature shards found in {feature_dir}")
    n_sae_features = _real_feature_width(feature_dir)

    _freeze_volatile_metadata(monkeypatch)
    real_add = ExactMagnitudeCounts.add

    def force_cpu_exact_add(
        reducer: ExactMagnitudeCounts,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        *,
        chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
        reduction_device: t.device = _CPU_DEVICE,
    ) -> None:
        del reduction_device
        real_add(
            reducer,
            feature_ids,
            feature_mags,
            chunk_size=chunk_size,
            reduction_device=t.device("cpu"),
        )

    with monkeypatch.context() as reference:
        reference.setattr(ExactMagnitudeCounts, "add", force_cpu_exact_add)
        reference.setattr(
            statistics_module,
            "_should_prefetch_exact_loads",
            lambda **_kwargs: False,
        )
        reference_stats = compute_statistics(
            str(feature_dir),
            str(tmp_path / "cpu-exact"),
            n_sae_features=n_sae_features,
            streaming=True,
            device="cuda",
            exclude_last_shard=False,
        )
    candidate_stats = compute_statistics(
        str(feature_dir),
        str(tmp_path / "cuda-exact"),
        n_sae_features=n_sae_features,
        streaming=True,
        device="cuda",
        exclude_last_shard=False,
    )

    assert reference_stats.features.n_segments > 100_000
    assert candidate_stats.features.n_segments == reference_stats.features.n_segments
    assert sum(stat.count for stat in reference_stats.per_feature_stats.values()) > 100_000_000
    _assert_safetensors_cuda_close(
        tmp_path / "cuda-exact.safetensors",
        tmp_path / "cpu-exact.safetensors",
    )


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
def test_full_real_read_zstd_pipeline_matches_sequential_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare complete artifacts on multiple untruncated production shards."""
    configured = os.environ.get("FEATURE_STATISTICS_REAL_SHARD_DIR")
    if configured is None:
        pytest.skip("FEATURE_STATISTICS_REAL_SHARD_DIR is not configured")
    feature_dir = Path(configured)
    shard_paths = glob_features(feature_dir)
    if len(shard_paths) < 2:
        pytest.skip("read+zstd pipeline parity requires at least two real shards")
    n_sae_features = _real_feature_width(feature_dir)
    _freeze_volatile_metadata(monkeypatch)

    with monkeypatch.context() as sequential:
        sequential.setattr(
            statistics_module,
            "_should_prefetch_exact_loads",
            lambda **_kwargs: False,
        )
        reference = compute_statistics(
            str(feature_dir),
            str(tmp_path / "sequential"),
            n_sae_features=n_sae_features,
            streaming=True,
            device="cuda",
            exclude_last_shard=False,
        )
    candidate = compute_statistics(
        str(feature_dir),
        str(tmp_path / "pipelined"),
        n_sae_features=n_sae_features,
        streaming=True,
        device="cuda",
        exclude_last_shard=False,
    )

    assert reference.features.n_segments > 100_000
    assert candidate.features.n_segments == reference.features.n_segments
    assert sum(stat.count for stat in reference.per_feature_stats.values()) > 100_000_000
    _assert_safetensors_cuda_close(
        tmp_path / "pipelined.safetensors",
        tmp_path / "sequential.safetensors",
    )


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
def test_full_real_extracted_shard_exact_counts_and_selected_percentiles() -> None:
    """Run the reducer on a complete production shard from a full Gemma model.

    Set ``FEATURE_STATISTICS_REAL_SHARD_DIR`` to a directory containing exactly
    one or more compatible raw base shards. This gate consumes the complete
    shard bodies; it does not truncate activations, shrink the SAE dictionary,
    substitute a toy model, or synthesize magnitudes.
    """
    configured = os.environ.get("FEATURE_STATISTICS_REAL_SHARD_DIR")
    if configured is None:
        pytest.skip("FEATURE_STATISTICS_REAL_SHARD_DIR is not configured")
    feature_dir = Path(configured)
    shard_paths = glob_features(feature_dir)
    if not shard_paths:
        pytest.fail(f"no raw feature shards found in {feature_dir}")

    n_sae_features = _real_feature_width(feature_dir)
    result = exact_pass1(
        str(feature_dir),
        n_sae_features=n_sae_features,
        device=t.device("cuda"),
    )
    assert result.magnitude_counts is not None
    assert result.n_sae_features == n_sae_features
    assert result.magnitude_counts.total_count == int(result.acc.occurrence.sum().item())
    assert result.magnitude_counts.total_count > 100_000_000
    assert result.seg_acc.features_per_segment.population_size > 100_000

    details, edges = result.magnitude_counts.finalize(
        occurrence=result.acc.occurrence,
        mag_subsample=result.mag_subsample,
        global_mag_min=result.global_mag_min,
        percentiles=PERCENTILES,
    )
    assert edges is not None
    assert len(edges) == 51
    assert len(details) == int((result.acc.occurrence > 0).sum().item())

    loaded = load_all_shards(str(feature_dir))
    active_ids = (result.acc.occurrence > 0).nonzero(as_tuple=True)[0]
    selected_positions = sorted({0, len(active_ids) // 2, len(active_ids) - 1})
    for position in selected_positions:
        feature_id = int(active_ids[position].item())
        mask = loaded.feature_ids == feature_id
        reference, reference_edges = compute_per_feature_percentiles(
            loaded.feature_ids[mask],
            loaded.feature_mags[mask],
            percentiles=PERCENTILES,
        )
        assert reference_edges is not None
        expected = reference[feature_id]
        actual = details[feature_id]
        assert actual.percentiles == expected.percentiles
        assert actual.median == expected.median
        assert sum(actual.histogram) == int(mask.sum().item())
